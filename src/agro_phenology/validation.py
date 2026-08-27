"""End-to-end retrospective validation and summary metrics."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .degree_days import add_degree_days
from .open_meteo import WeatherResponse
from .weather import aggregate_hourly_temperature

HourlyLoader = Callable[[float, float, object, object, tuple[str, ...]], WeatherResponse]


def validate_records(
    accepted_observations: pd.DataFrame,
    config: ExperimentConfig,
    hourly_loader: HourlyLoader,
    lookahead_days: int = 45,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Calculate predicted dates for every accepted record and temperature method."""

    records: list[dict[str, Any]] = []
    daily_tables: dict[str, pd.DataFrame] = {}
    variables = ("temperature_2m", *config.optional_hourly_variables)
    for row in accepted_observations.itertuples(index=False):
        start_date = row.accumulation_start_date
        end_date = row.observation_date + timedelta(days=lookahead_days)
        try:
            response = hourly_loader(row.latitude, row.longitude, start_date, end_date, variables)
        except Exception as exc:  # preserve per-record failure instead of losing the batch
            records.append(
                {
                    "observation_id": row.observation_id,
                    "calculation_status": "weather_error",
                    "calculation_error": str(exc),
                    "observed_date": row.observation_date,
                    "accumulation_start_date": start_date,
                    "accumulation_start_rule": row.accumulation_start_rule,
                }
            )
            continue
        timezone_name = str(response.metadata.get("timezone") or "UTC")
        for method in config.daily_temperature_methods:
            daily = aggregate_hourly_temperature(
                response.hourly,
                timezone=timezone_name,
                method=method,
                minimum_valid_hours=config.minimum_valid_hours,
                allow_incomplete_days=config.allow_incomplete_days,
            )
            calculated, predicted = add_degree_days(
                daily,
                base_temperature_c=config.model.base_temperature_c,
                threshold=config.model.degree_day_threshold,
            )
            key = f"{row.observation_id}__{method}"
            daily_tables[key] = calculated
            insufficient_days = int((~daily["accepted"]).sum())
            blocking_incomplete_days = daily.loc[~daily["accepted"], "date"]
            if predicted is not None:
                blocking_incomplete_days = blocking_incomplete_days[
                    blocking_incomplete_days.map(lambda value: value <= predicted)
                ]
            if not config.allow_incomplete_days and not blocking_incomplete_days.empty:
                predicted = None
                status = "insufficient_weather_coverage"
            else:
                status = "threshold_not_reached" if predicted is None else "ok"
            error_days = (predicted - row.observation_date).days if predicted else np.nan
            response_model = str(response.metadata.get("model") or "unknown")
            weather_dataset = "ERA5-Land" if response_model == "era5_land" else response_model
            records.append(
                {
                    "observation_id": row.observation_id,
                    "latitude": row.latitude,
                    "longitude": row.longitude,
                    "region": row.region,
                    "pest_name": row.pest_name,
                    "observed_stage": row.observed_stage,
                    "target_stage_code": config.model.target_stage_code,
                    "observed_date": row.observation_date,
                    "predicted_date": predicted,
                    "error_days": error_days,
                    "absolute_error_days": abs(error_days) if pd.notna(error_days) else np.nan,
                    "base_temperature_c": config.model.base_temperature_c,
                    "degree_day_threshold_c_day": config.model.degree_day_threshold,
                    "daily_temperature_method": method,
                    "accumulation_start_date": start_date,
                    "accumulation_start_rule": row.accumulation_start_rule,
                    "minimum_valid_hours": config.minimum_valid_hours,
                    "allow_incomplete_days": config.allow_incomplete_days,
                    "insufficient_day_count": insufficient_days,
                    "weather_dataset": weather_dataset,
                    "weather_cache_hit": response.cache_hit,
                    "requested_latitude": response.metadata.get("requested_latitude", row.latitude),
                    "requested_longitude": response.metadata.get("requested_longitude", row.longitude),
                    "returned_latitude": response.metadata.get("returned_latitude"),
                    "returned_longitude": response.metadata.get("returned_longitude"),
                    "elevation": response.metadata.get("elevation"),
                    "timezone": timezone_name,
                    "timezone_abbreviation": response.metadata.get("timezone_abbreviation"),
                    "utc_offset_seconds": response.metadata.get("utc_offset_seconds"),
                    "calculation_status": status,
                    "calculation_error": "",
                }
            )
    return pd.DataFrame(records), daily_tables


def calculate_summary_metrics(records: pd.DataFrame) -> pd.DataFrame:
    """Calculate date-error metrics, separately for each temperature method."""

    if records.empty or "calculation_status" not in records or "daily_temperature_method" not in records:
        return pd.DataFrame()
    valid = records[records["calculation_status"] == "ok"].copy()
    rows: list[dict[str, Any]] = []
    for method, group in valid.groupby("daily_temperature_method", dropna=False):
        errors = pd.to_numeric(group["error_days"], errors="coerce").dropna()
        if errors.empty:
            continue
        rows.append(
            {
                "daily_temperature_method": method,
                "valid_observation_count": int(errors.size),
                "mean_absolute_error_days": float(errors.abs().mean()),
                "median_absolute_error_days": float(errors.abs().median()),
                "bias_days": float(errors.mean()),
                "rmse_days": float(np.sqrt(np.mean(np.square(errors)))),
                "within_3_days_fraction": float((errors.abs() <= 3).mean()),
                "within_5_days_fraction": float((errors.abs() <= 5).mean()),
                "within_7_days_fraction": float((errors.abs() <= 7).mean()),
                "interpretation": "предварительная оценка на пилотной выборке",
            }
        )
    return pd.DataFrame(rows)


def metrics_by_region(records: pd.DataFrame, minimum_group_size: int = 3) -> pd.DataFrame:
    """Return region/method metrics only for groups large enough to describe."""

    if records.empty or "region" not in records:
        return pd.DataFrame()
    valid = records[records["calculation_status"] == "ok"].copy()
    parts: list[pd.DataFrame] = []
    for (region, method), group in valid.groupby(["region", "daily_temperature_method"], dropna=False):
        if len(group) < minimum_group_size:
            continue
        metrics = calculate_summary_metrics(group)
        metrics.insert(0, "region", region)
        parts.append(metrics)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def methodology_payload(config: ExperimentConfig, execution_timestamp: str) -> dict[str, Any]:
    """Build the machine-readable methodology record."""

    return {
        "weather_provider": "Open-Meteo Historical Weather API",
        "weather_dataset": "ERA5-Land",
        "API parameters": {
            "hourly": ["temperature_2m", *config.optional_hourly_variables],
            "models": "era5_land",
            "timezone": "auto",
            "cell_selection": "land",
            "temperature_unit": "celsius",
        },
        "base_temperature_c": config.model.base_temperature_c,
        "degree_day_threshold": config.model.degree_day_threshold,
        "degree_day_unit": "°C·day",
        "target_pest": config.model.pest_name,
        "target_stage": config.model.target_stage_name,
        "target_stage_code": config.model.target_stage_code,
        "accumulation_start_rule": config.accumulation_start_rule,
        "daily_temperature_method": list(config.daily_temperature_methods),
        "minimum_hourly_coverage": config.minimum_valid_hours,
        "allow_incomplete_days": config.allow_incomplete_days,
        "calculation_version": config.calculation_version,
        "execution_timestamp": execution_timestamp,
    }


def save_results(
    records: pd.DataFrame,
    metrics: pd.DataFrame,
    excluded: pd.DataFrame,
    methodology: dict[str, Any],
    results_dir: str | Path,
) -> None:
    """Save the required reproducibility artifacts."""

    destination = Path(results_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "plots").mkdir(exist_ok=True)
    records.to_csv(destination / "validation_records.csv", index=False)
    metrics.to_csv(destination / "summary_metrics.csv", index=False)
    excluded.to_csv(destination / "excluded_records.csv", index=False)
    (destination / "methodology.json").write_text(
        json.dumps(methodology, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
