"""Reproducible orchestration for the two explicitly limited 2026 pilot reports."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from typing import Any

import pandas as pd

from .late_blight import (
    HuttonConfig,
    PolyakovConfig,
    aggregate_polyakov_daily,
    classify_hutton_days,
    classify_hutton_periods,
    classify_polyakov_windows,
    extract_hutton_episodes,
    extract_polyakov_episodes,
    merge_polyakov_weather_sources,
)
from .late_blight_validation import (
    summarize_hutton_lookbacks,
    summarize_polyakov_activation_interval,
    summarize_polyakov_observation,
)
from .open_meteo import OpenMeteoClient


WEATHER_SOURCE_PROFILE = (
    "open_meteo_era5_land_temperature_humidity__era5_precipitation_v1"
)


PILOT_FALLBACK_ANALYSIS_POINTS: tuple[dict[str, object], ...] = (
    {
        "observation_id": "LB-PILOT-001",
        "analysis_point_id": "MO-NW-proxy",
        "analysis_latitude": 56.20,
        "analysis_longitude": 36.70,
        "coordinate_rule": "fixed_regional_sensitivity_grid_not_field_coordinate",
    },
    {
        "observation_id": "LB-PILOT-001",
        "analysis_point_id": "MO-NE-proxy",
        "analysis_latitude": 56.20,
        "analysis_longitude": 38.50,
        "coordinate_rule": "fixed_regional_sensitivity_grid_not_field_coordinate",
    },
    {
        "observation_id": "LB-PILOT-001",
        "analysis_point_id": "MO-W-proxy",
        "analysis_latitude": 55.60,
        "analysis_longitude": 36.50,
        "coordinate_rule": "fixed_regional_sensitivity_grid_not_field_coordinate",
    },
    {
        "observation_id": "LB-PILOT-001",
        "analysis_point_id": "MO-E-proxy",
        "analysis_latitude": 55.60,
        "analysis_longitude": 39.20,
        "coordinate_rule": "fixed_regional_sensitivity_grid_not_field_coordinate",
    },
    {
        "observation_id": "LB-PILOT-001",
        "analysis_point_id": "MO-S-proxy",
        "analysis_latitude": 54.90,
        "analysis_longitude": 37.60,
        "coordinate_rule": "fixed_regional_sensitivity_grid_not_field_coordinate",
    },
    {
        "observation_id": "LB-PILOT-002",
        "analysis_point_id": "Sergiev-Posad-center-proxy",
        "analysis_latitude": 56.3153,
        "analysis_longitude": 38.1358,
        "coordinate_rule": "city_center_proxy_not_field_coordinate",
    },
)


def build_pilot_analysis_points(observations: pd.DataFrame) -> pd.DataFrame:
    """Use supplied coordinates when present and fixed proxies only as explicit fallbacks."""

    required = {"observation_id", "latitude", "longitude"}
    missing_columns = required - set(observations.columns)
    if missing_columns:
        raise ValueError(f"Pilot observations are missing columns: {sorted(missing_columns)}")
    fallback = pd.DataFrame(PILOT_FALLBACK_ANALYSIS_POINTS)
    rows: list[dict[str, object]] = []
    for observation in observations.itertuples(index=False):
        has_latitude = pd.notna(observation.latitude)
        has_longitude = pd.notna(observation.longitude)
        if has_latitude != has_longitude:
            raise ValueError(f"Incomplete coordinate pair for {observation.observation_id}")
        if has_latitude:
            location_precision = getattr(
                observation,
                "location_precision",
                "reported_coordinate_not_confirmed_field",
            )
            rows.append(
                {
                    "observation_id": observation.observation_id,
                    "analysis_point_id": (
                        "Bunyatino-village-coordinate"
                        if observation.observation_id == "LB-PILOT-001"
                        else f"{observation.observation_id}-reported-coordinate"
                    ),
                    "analysis_latitude": float(observation.latitude),
                    "analysis_longitude": float(observation.longitude),
                    "coordinate_rule": str(location_precision),
                }
            )
            continue
        observation_fallback = fallback[
            fallback["observation_id"].eq(observation.observation_id)
        ]
        if observation_fallback.empty:
            raise ValueError(f"No coordinates or pilot fallback for {observation.observation_id}")
        rows.extend(observation_fallback.to_dict(orient="records"))
    points = pd.DataFrame(rows)
    return points.merge(observations, on="observation_id", how="left", validate="many_to_one")


def run_late_blight_pilot(
    observations: pd.DataFrame,
    client: OpenMeteoClient,
    analysis_start_date: date | str = "2026-05-01",
    hutton_config: HuttonConfig = HuttonConfig(),
    polyakov_config: PolyakovConfig = PolyakovConfig(),
    activation_sensitivity_dates: tuple[str, ...] = ("2026-06-15", "2026-06-22", "2026-06-30"),
    activation_sensitivity_source: str = "fallback_manual_sensitivity",
) -> dict[str, Any]:
    """Run both indicators while retaining coordinate, phenophase, and weather limitations."""

    points = build_pilot_analysis_points(observations)
    daily_tables: list[pd.DataFrame] = []
    period_tables: list[pd.DataFrame] = []
    event_tables: list[pd.DataFrame] = []
    case_tables: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []
    land_variables = ("temperature_2m", "relative_humidity_2m")

    for point in points.itertuples(index=False):
        land_response = client.fetch_hourly(
            point.analysis_latitude,
            point.analysis_longitude,
            analysis_start_date,
            point.observation_date,
            land_variables,
        )
        precipitation_response = client.fetch_era5_precipitation_hourly(
            point.analysis_latitude,
            point.analysis_longitude,
            analysis_start_date,
            point.observation_date,
        )
        land_timezone = str(land_response.metadata.get("timezone") or "Europe/Moscow")
        precipitation_timezone = str(
            precipitation_response.metadata.get("timezone") or "Europe/Moscow"
        )
        if land_timezone != precipitation_timezone:
            raise ValueError(
                "ERA5-Land and ERA5 responses use different timezones: "
                f"{land_timezone!r} != {precipitation_timezone!r}"
            )
        timezone = land_timezone
        polyakov_hourly = merge_polyakov_weather_sources(
            land_response.hourly,
            precipitation_response.hourly,
        )
        common = {
            "observation_id": point.observation_id,
            "analysis_point_id": point.analysis_point_id,
            "location_text": point.location_text,
            "analysis_latitude": point.analysis_latitude,
            "analysis_longitude": point.analysis_longitude,
            "coordinate_rule": point.coordinate_rule,
            "observation_date": point.observation_date,
            "observation_outcome": point.observation_outcome,
            "weather_source_profile": WEATHER_SOURCE_PROFILE,
        }
        hutton_daily = classify_hutton_days(
            land_response.hourly,
            timezone,
            hutton_config,
            start_date=analysis_start_date,
            end_date=point.observation_date,
        )
        hutton_periods = classify_hutton_periods(hutton_daily, hutton_config)
        hutton_daily = hutton_daily.assign(**common, timezone=timezone)
        hutton_periods = hutton_periods.assign(**common, timezone=timezone)
        polyakov_daily = aggregate_polyakov_daily(
            polyakov_hourly,
            timezone,
            polyakov_config,
            start_date=analysis_start_date,
            end_date=point.observation_date,
        )
        combined_daily = hutton_daily.merge(
            polyakov_daily,
            on="date",
            how="outer",
            suffixes=("_hutton", "_polyakov"),
        ).assign(
            temperature_source_model="era5_land",
            relative_humidity_source_model="era5_land",
            precipitation_source_model="era5",
        )
        daily_tables.append(combined_daily)
        period_tables.append(hutton_periods)

        hutton_summary = summarize_hutton_lookbacks(
            point.observation_id,
            point.observation_date,
            point.observation_outcome,
            hutton_periods,
        ).assign(
            model="hutton_criteria",
            analysis_point_id=point.analysis_point_id,
            location_text=point.location_text,
            coordinate_rule=point.coordinate_rule,
            analysis_latitude=point.analysis_latitude,
            analysis_longitude=point.analysis_longitude,
            result_scope="pilot_association_not_accuracy",
            temperature_source_model="era5_land",
            relative_humidity_source_model="era5_land",
            precipitation_source_model=pd.NA,
            weather_source_profile=WEATHER_SOURCE_PROFILE,
        )
        case_tables.append(hutton_summary)

        hutton_episodes = extract_hutton_episodes(hutton_daily)
        if not hutton_episodes.empty:
            hutton_episodes = hutton_episodes.assign(
                model="hutton_criteria",
                event_status="pass_episode",
                **common,
            )
            event_tables.append(hutton_episodes)

        interval_start_raw = getattr(point, "phenophase_interval_start_date", pd.NA)
        interval_end_raw = getattr(point, "phenophase_interval_end_date", pd.NA)
        has_interval_start = pd.notna(interval_start_raw)
        has_interval_end = pd.notna(interval_end_raw)
        if has_interval_start != has_interval_end:
            raise ValueError(f"Incomplete phenophase interval for {point.observation_id}")
        if has_interval_start:
            interval_start = pd.Timestamp(interval_start_raw).date()
            interval_end = pd.Timestamp(interval_end_raw).date()
            if interval_end < interval_start:
                raise ValueError(f"Invalid phenophase interval for {point.observation_id}")
            activation_dates = tuple(
                value.strftime("%Y-%m-%d")
                for value in pd.date_range(interval_start, interval_end, freq="D")
            )
            activation_source = str(
                getattr(point, "phenophase_interval_basis", "author_confirmed_interval")
            )
            phenophase_status = str(
                getattr(point, "phenophase_status", "AUTHOR_CONFIRMED_REGIONAL_INTERVAL")
            )
            phenophase_name = str(getattr(point, "phenophase_name", "бутонизация"))
            field_specific_date_known = bool(
                getattr(point, "field_specific_activation_date_known", False)
            )
            scenario_scope = "primary_activation_interval_scenario"
        else:
            interval_start = None
            interval_end = None
            activation_dates = activation_sensitivity_dates
            activation_source = activation_sensitivity_source
            phenophase_status = "MISSING_PHENOPHASE"
            phenophase_name = ""
            field_specific_date_known = False
            scenario_scope = "assumption_based_sensitivity_not_validation"
            primary_polyakov = classify_polyakov_windows(polyakov_daily, None, polyakov_config)
            primary_summary = summarize_polyakov_observation(
                point.observation_id,
                point.observation_date,
                point.observation_outcome,
                primary_polyakov,
                "missing_observed_phenophase",
            )
            case_tables.append(
                pd.DataFrame([primary_summary]).assign(
                    model="polyakov_late_blight_10d_v1",
                    analysis_point_id=point.analysis_point_id,
                    location_text=point.location_text,
                    coordinate_rule=point.coordinate_rule,
                    analysis_latitude=point.analysis_latitude,
                    analysis_longitude=point.analysis_longitude,
                    result_scope="primary_not_evaluable",
                    temperature_source_model="era5_land",
                    relative_humidity_source_model="era5_land",
                    precipitation_source_model="era5",
                    weather_source_profile=WEATHER_SOURCE_PROFILE,
                )
            )

        interval_summaries: list[dict[str, object]] = []
        for activation_date in activation_dates:
            classified = classify_polyakov_windows(polyakov_daily, activation_date, polyakov_config)
            summary = summarize_polyakov_observation(
                point.observation_id,
                point.observation_date,
                point.observation_outcome,
                classified,
                f"{activation_source}_{activation_date}",
            )
            summary.update(
                {
                    "activation_date": pd.Timestamp(activation_date).date(),
                    "activation_date_start": interval_start,
                    "activation_date_end": interval_end,
                    "phenophase_name": phenophase_name,
                    "phenophase_status": phenophase_status,
                    "field_specific_activation_date_known": field_specific_date_known,
                    "phenophase_interval_basis": activation_source,
                }
            )
            interval_summaries.append(summary)
            case_tables.append(
                pd.DataFrame([summary]).assign(
                    model="polyakov_late_blight_10d_v1",
                    analysis_point_id=point.analysis_point_id,
                    location_text=point.location_text,
                    coordinate_rule=point.coordinate_rule,
                    analysis_latitude=point.analysis_latitude,
                    analysis_longitude=point.analysis_longitude,
                    result_scope=scenario_scope,
                    temperature_source_model="era5_land",
                    relative_humidity_source_model="era5_land",
                    precipitation_source_model="era5",
                    weather_source_profile=WEATHER_SOURCE_PROFILE,
                )
            )
            episodes = extract_polyakov_episodes(classified, polyakov_config)
            if not episodes.empty:
                episodes = episodes.assign(
                    model="polyakov_late_blight_10d_v1",
                    event_status="critical_weather_episode",
                    activation_rule=f"{activation_source}_{activation_date}",
                    activation_date=pd.Timestamp(activation_date).date(),
                    activation_date_start=interval_start,
                    activation_date_end=interval_end,
                    phenophase_name=phenophase_name,
                    phenophase_status=phenophase_status,
                    field_specific_activation_date_known=field_specific_date_known,
                    phenophase_interval_basis=activation_source,
                    temperature_source_model="era5_land",
                    relative_humidity_source_model="era5_land",
                    precipitation_source_model="era5",
                    **common,
                )
                event_tables.append(episodes)

        if interval_start is not None and interval_end is not None:
            interval_summary = summarize_polyakov_activation_interval(
                point.observation_id,
                point.observation_date,
                point.observation_outcome,
                pd.DataFrame(interval_summaries),
                interval_start,
                interval_end,
                phenophase_status=phenophase_status,
            )
            case_tables.append(
                pd.DataFrame([interval_summary]).assign(
                    model="polyakov_late_blight_10d_v1",
                    analysis_point_id=point.analysis_point_id,
                    location_text=point.location_text,
                    coordinate_rule=point.coordinate_rule,
                    analysis_latitude=point.analysis_latitude,
                    analysis_longitude=point.analysis_longitude,
                    phenophase_name=phenophase_name,
                    phenophase_interval_basis=activation_source,
                    result_scope="primary_activation_interval_summary",
                    temperature_source_model="era5_land",
                    relative_humidity_source_model="era5_land",
                    precipitation_source_model="era5",
                    weather_source_profile=WEATHER_SOURCE_PROFILE,
                )
            )

        land_available = land_response.hourly.copy()
        land_available["time"] = pd.to_datetime(land_available["time"], errors="coerce")
        precipitation_available = precipitation_response.hourly.copy()
        precipitation_available["time"] = pd.to_datetime(
            precipitation_available["time"], errors="coerce"
        )
        valid_temperature = land_available[land_available["temperature_2m"].notna()]
        valid_humidity = land_available[land_available["relative_humidity_2m"].notna()]
        valid_precipitation = precipitation_available[
            precipitation_available["precipitation"].notna()
        ]
        metadata_rows.append(
            {
                **common,
                "requested_start_date": str(analysis_start_date),
                "requested_end_date": str(point.observation_date),
                "era5_land_returned_latitude": land_response.metadata.get("returned_latitude"),
                "era5_land_returned_longitude": land_response.metadata.get("returned_longitude"),
                "era5_land_elevation": land_response.metadata.get("elevation"),
                "era5_returned_latitude": precipitation_response.metadata.get("returned_latitude"),
                "era5_returned_longitude": precipitation_response.metadata.get("returned_longitude"),
                "era5_elevation": precipitation_response.metadata.get("elevation"),
                "timezone": timezone,
                "temperature_valid_hour_count": len(valid_temperature),
                "humidity_valid_hour_count": len(valid_humidity),
                "precipitation_valid_hour_count": len(valid_precipitation),
                "last_temperature_local_time": valid_temperature["time"].max() if not valid_temperature.empty else pd.NaT,
                "last_humidity_local_time": valid_humidity["time"].max() if not valid_humidity.empty else pd.NaT,
                "last_precipitation_local_time": valid_precipitation["time"].max() if not valid_precipitation.empty else pd.NaT,
                "temperature_source_model": land_response.metadata.get("model"),
                "relative_humidity_source_model": land_response.metadata.get("model"),
                "precipitation_source_model": precipitation_response.metadata.get("model"),
                "source_alignment_status": "exact_timestamp_match",
                "source_alignment_hour_count": len(polyakov_hourly),
                "era5_land_cache_hit": land_response.cache_hit,
                "era5_precipitation_cache_hit": precipitation_response.cache_hit,
                "era5_land_retrieved_at": land_response.metadata.get("retrieved_at"),
                "era5_precipitation_retrieved_at": precipitation_response.metadata.get(
                    "retrieved_at"
                ),
                "era5_land_raw_cache_file": land_response.raw_path.name,
                "era5_precipitation_raw_cache_file": precipitation_response.raw_path.name,
            }
        )

    daily_features = pd.concat(daily_tables, ignore_index=True) if daily_tables else pd.DataFrame()
    period_features = pd.concat(period_tables, ignore_index=True) if period_tables else pd.DataFrame()
    model_events = pd.concat(event_tables, ignore_index=True) if event_tables else pd.DataFrame()
    case_results = (
        pd.concat(
            [table.dropna(axis=1, how="all") for table in case_tables],
            ignore_index=True,
            sort=False,
        )
        if case_tables
        else pd.DataFrame()
    )
    return {
        "analysis_points": points,
        "case_results": case_results,
        "daily_features": daily_features,
        "period_features": period_features,
        "model_events": model_events,
        "weather_metadata": pd.DataFrame(metadata_rows),
        "configuration": {
            "hutton": asdict(hutton_config),
            "polyakov": asdict(polyakov_config),
            "analysis_start_date": str(analysis_start_date),
            "fallback_activation_sensitivity_dates": list(activation_sensitivity_dates),
            "fallback_activation_sensitivity_source": activation_sensitivity_source,
            "phenophase_interval_policy": "evaluate_every_date_inclusive_without_voting",
            "weather_sources": {
                "temperature_2m": "era5_land",
                "relative_humidity_2m": "era5_land",
                "precipitation": "era5",
            },
            "weather_source_profile": WEATHER_SOURCE_PROFILE,
        },
    }
