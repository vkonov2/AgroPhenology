"""Shared causal data layer for apple early-warning experiments.

The module deliberately models the first *registered* target event.  Missing
visits and unlisted organisms remain unassessed; they are never converted into
confirmed biological absence.  All weather features end two calendar days
before the issue date, as frozen in each experiment contract.
"""
from __future__ import annotations

from datetime import time
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


CALENDAR_FEATURES = ["doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2"]
COMMON_WINDOWS = (7, 14, 30)
COMMON_VARIABLES = (
    "tmean",
    "tmin",
    "tmax",
    "rhmean",
    "rain",
    "rain_days",
    "active_temperature_sum",
)
COMMON_WEATHER_FEATURES = [
    f"nasa_{name}_{days}d" for days in COMMON_WINDOWS for name in COMMON_VARIABLES
]
CODLING_FEATURES = [
    "codling_degree_days_base10_7d",
    "codling_degree_days_base10_14d",
    "codling_degree_days_base10_30d",
    "codling_degree_days_from_apr1",
    "codling_dd_apr1_margin_126",
    "codling_dd_apr1_margin_230",
    "codling_dd_apr1_pass_126",
    "codling_dd_apr1_pass_230",
    "codling_adult_biofix_known",
    "codling_days_since_adult_biofix",
    "codling_degree_days_since_adult_biofix",
    "codling_dd_biofix_margin_126",
    "codling_dd_biofix_margin_230",
    "codling_dd_biofix_pass_126",
    "codling_dd_biofix_pass_230",
]
SCAB_FEATURES = [
    "scab_warm_wet_proxy_days_3d",
    "scab_warm_wet_proxy_days_7d",
    "scab_warm_wet_proxy_days_14d",
    "scab_warm_humid_days_7d",
    "scab_warm_humid_days_14d",
    "scab_rain_sum_3d",
    "scab_rain_sum_7d",
    "scab_proxy_current_run_days",
    "scab_days_since_proxy",
]


def build_field_seasons(
    visits: pd.DataFrame,
    *,
    snapshot_date: str | pd.Timestamp,
    season_start_month: int = 4,
    season_start_day: int = 1,
    season_end_month: int = 10,
    season_end_day: int = 31,
) -> pd.DataFrame:
    """Create one immutable registry row per field-year."""
    snapshot = pd.Timestamp(snapshot_date).normalize()
    rows: list[dict] = []
    for field_season, group in visits.groupby("field_season", sort=True):
        group = group.sort_values(["observation_date", "visit_id"])
        first = group.iloc[0]
        year = int(first["crop_season"])
        first_visit = pd.Timestamp(group["observation_date"].min()).normalize()
        last_visit = pd.Timestamp(group["observation_date"].max()).normalize()
        nominal_start = pd.Timestamp(year=year, month=season_start_month, day=season_start_day)
        nominal_end = pd.Timestamp(year=year, month=season_end_month, day=season_end_day)
        entry_date = max(first_visit + pd.Timedelta(days=1), nominal_start)
        decision_end = min(nominal_end, snapshot) if year == snapshot.year else nominal_end
        if year > snapshot.year:
            decision_end = snapshot

        positives = group[group["label_status"].eq("positive")]
        event_date = positives["observation_date"].min() if len(positives) else pd.NaT
        event_available = event_date + pd.Timedelta(days=1) if pd.notna(event_date) else pd.NaT
        prior = group[group["observation_date"].lt(event_date)] if pd.notna(event_date) else group.iloc[0:0]
        positive_at_entry = bool(pd.notna(event_date) and pd.Timestamp(event_date) == first_visit)
        warnable = bool(
            pd.notna(event_date)
            and not positive_at_entry
            and entry_date <= pd.Timestamp(event_date) - pd.Timedelta(days=3)
        )

        adult_date = pd.NaT
        if "adult_trap_positive" in group:
            adult = group[group["adult_trap_positive"].fillna(False).astype(bool)]
            if len(adult):
                adult_date = adult["observation_date"].min()
        adult_available = adult_date + pd.Timedelta(days=1) if pd.notna(adult_date) else pd.NaT

        if positive_at_entry:
            category = "positive_known_at_entry"
        elif pd.notna(event_date) and warnable:
            category = "warnable_first_event"
        elif pd.notna(event_date):
            category = "late_entry_first_event"
        elif last_visit >= entry_date + pd.Timedelta(days=10):
            category = "no_event_with_observable_horizon"
        else:
            category = "no_event_short_followup"

        rows.append(
            {
                "field_season": field_season,
                "field_uid": first["final_field_uid"],
                "season": year,
                "crop_code": int(first["crop_code"]),
                "first_visit_date": first_visit,
                "first_visit_available_date": entry_date,
                "last_visit_date": last_visit,
                "decision_end_date": decision_end,
                "first_recorded_event_date": event_date,
                "first_recorded_event_available_date": event_available,
                "first_adult_trap_positive_date": adult_date,
                "first_adult_trap_positive_available_date": adult_available,
                "first_visit_label_status": first["label_status"],
                "positive_at_first_visit": positive_at_entry,
                "warnable_first_event": warnable,
                "all_actionable_leads_available": bool(
                    warnable and entry_date <= pd.Timestamp(event_date) - pd.Timedelta(days=10)
                ),
                "entry_category": category,
                "previous_visit_before_event": bool(len(prior)),
                "previous_visit_date": prior["observation_date"].max() if len(prior) else pd.NaT,
                "previous_visit_gap_days": int(
                    (pd.Timestamp(event_date) - pd.Timestamp(prior["observation_date"].max())).days
                )
                if len(prior)
                else np.nan,
                "previous_explicit_target_absence": bool(
                    prior["label_status"].eq("explicit_target_absent").any()
                ),
                "previous_generic_absence": bool(prior["label_status"].eq("generic_absent").any()),
                "visit_count": int(len(group)),
                "registration_followup_days": int((last_visit - first_visit).days),
                "coordinate_scope": "A_direct"
                if group["final_coordinate_class"].eq("A_direct").all()
                else "includes_B",
                "final_latitude": float(first["final_latitude"]),
                "final_longitude": float(first["final_longitude"]),
                "status_counts": json.dumps(
                    {str(k): int(v) for k, v in group["label_status"].value_counts().items()},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "censoring_basis": "last_registry_visit_for_outcome_mask_only",
            }
        )
    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values(["season", "field_season"]).reset_index(drop=True)
    return result


def _issue_timestamp(day: pd.Timestamp, timezone: str, issue_time: str) -> str:
    clock = time.fromisoformat(issue_time)
    naive = pd.Timestamp.combine(pd.Timestamp(day).date(), clock)
    return naive.tz_localize(ZoneInfo(timezone)).isoformat()


def build_daily_decisions(
    field_seasons: pd.DataFrame,
    *,
    timezone: str,
    issue_time: str,
    minimum_lead: int = 3,
    maximum_lead: int = 10,
    weather_cutoff_days: int = 2,
) -> pd.DataFrame:
    """Build daily issue dates independently of the date of a future visit."""
    rows: list[dict] = []
    for season in field_seasons.itertuples(index=False):
        start = pd.Timestamp(season.first_visit_available_date)
        end = pd.Timestamp(season.decision_end_date)
        if end < start:
            continue
        event_date = (
            pd.Timestamp(season.first_recorded_event_date)
            if pd.notna(season.first_recorded_event_date)
            else pd.NaT
        )
        event_available = (
            pd.Timestamp(season.first_recorded_event_available_date)
            if pd.notna(season.first_recorded_event_available_date)
            else pd.NaT
        )
        for issue_date in pd.date_range(start, end, freq="D"):
            service_active = bool(pd.isna(event_available) or issue_date < event_available)
            target_class = "unknown"
            target_observable = False
            outcome_reason = "insufficient_registry_followup"
            days_to_event = np.nan
            if pd.notna(event_date) and issue_date <= event_date:
                days_to_event = int((event_date - issue_date).days)
                if 0 <= days_to_event <= minimum_lead - 1:
                    target_class = "imminent"
                    target_observable = True
                    outcome_reason = "observed_first_registration"
                elif minimum_lead <= days_to_event <= maximum_lead:
                    target_class = "actionable"
                    target_observable = True
                    outcome_reason = "observed_first_registration"
                elif days_to_event > maximum_lead:
                    target_class = "no_record_in_horizon"
                    target_observable = True
                    outcome_reason = "later_first_registration_proves_registry_horizon"
            elif pd.isna(event_date) and issue_date + pd.Timedelta(days=maximum_lead) <= pd.Timestamp(
                season.last_visit_date
            ):
                target_class = "no_record_in_horizon"
                target_observable = True
                outcome_reason = "followed_registry_through_full_horizon"
            elif pd.notna(event_date) and issue_date > event_date:
                outcome_reason = "after_first_registration"

            rows.append(
                {
                    "field_season": season.field_season,
                    "field_uid": season.field_uid,
                    "season": int(season.season),
                    "issue_date": issue_date,
                    "issued_at": _issue_timestamp(issue_date, timezone, issue_time),
                    "feature_cutoff_nasa_date": issue_date - pd.Timedelta(days=weather_cutoff_days),
                    "label_interval_start": issue_date + pd.Timedelta(days=minimum_lead),
                    "label_interval_end": issue_date + pd.Timedelta(days=maximum_lead),
                    "target_class": target_class,
                    "target_observable": target_observable,
                    "target_observability_reason": outcome_reason,
                    "days_to_first_recorded_event": days_to_event,
                    "service_active": service_active,
                    "evaluation_field_day": service_active,
                    "first_recorded_event_date": event_date,
                    "warnable_first_event": bool(season.warnable_first_event),
                    "positive_at_first_visit": bool(season.positive_at_first_visit),
                    "coordinate_scope": season.coordinate_scope,
                    "previous_visit_gap_days": season.previous_visit_gap_days,
                    "first_adult_trap_positive_date": season.first_adult_trap_positive_date,
                    "first_adult_trap_positive_available_date": season.first_adult_trap_positive_available_date,
                    "final_latitude": season.final_latitude,
                    "final_longitude": season.final_longitude,
                    "outcome_semantics": "first_registered_event_not_biological_onset_or_absence",
                }
            )
    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values(["season", "field_season", "issue_date"]).reset_index(drop=True)
    return result


def _parse_nasa_cell(cell: str) -> tuple[float, float]:
    left, right = str(cell).split("_", maxsplit=1)
    return float(left), float(right)


def derive_nasa_mapping(decisions: pd.DataFrame, nasa_daily: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Map apple coordinates to the nearest already frozen NASA grid cell."""
    cells = sorted(nasa_daily["nasa_cell"].dropna().astype(str).unique())
    cell_values = [(cell, *_parse_nasa_cell(cell)) for cell in cells]
    coordinates = decisions[["final_latitude", "final_longitude"]].drop_duplicates().copy()
    rows: list[dict] = []
    for coordinate in coordinates.itertuples(index=False):
        latitude = float(coordinate.final_latitude)
        longitude = float(coordinate.final_longitude)
        cosine = math.cos(math.radians(latitude))
        nearest = min(
            cell_values,
            key=lambda item: (item[1] - latitude) ** 2 + ((item[2] - longitude) * cosine) ** 2,
        )
        distance = math.sqrt((nearest[1] - latitude) ** 2 + ((nearest[2] - longitude) * cosine) ** 2)
        rows.append(
            {
                "final_latitude": latitude,
                "final_longitude": longitude,
                "nasa_latitude": nearest[1],
                "nasa_longitude": nearest[2],
                "nasa_cell": nearest[0],
                "mapping_method": "nearest_frozen_nasa_cell_equirectangular_v1",
                "angular_distance_degrees": distance,
            }
        )
    mapping = pd.DataFrame(rows).sort_values(["final_latitude", "final_longitude"]).reset_index(drop=True)
    audit = {
        "coordinate_count": int(len(mapping)),
        "available_frozen_cells": int(len(cells)),
        "assigned_cells": int(mapping["nasa_cell"].nunique()),
        "maximum_angular_distance_degrees": float(mapping["angular_distance_degrees"].max()),
        "median_angular_distance_degrees": float(mapping["angular_distance_degrees"].median()),
        "mapping_method": "nearest_frozen_nasa_cell_equirectangular_v1",
        "network_used": False,
    }
    return mapping, audit


def _consecutive_true(values: pd.Series) -> pd.Series:
    result: list[float] = []
    run = 0
    for value in values:
        if pd.isna(value):
            run = 0
            result.append(np.nan)
        elif bool(value):
            run += 1
            result.append(float(run))
        else:
            run = 0
            result.append(0.0)
    return pd.Series(result, index=values.index)


def _days_since_true(values: pd.Series, dates: pd.DatetimeIndex, limit: int = 60) -> pd.Series:
    output: list[float] = []
    last: pd.Timestamp | None = None
    for day, value in zip(dates, values):
        if pd.notna(value) and bool(value):
            last = pd.Timestamp(day)
            output.append(0.0)
        elif last is not None and (pd.Timestamp(day) - last).days <= limit:
            output.append(float((pd.Timestamp(day) - last).days))
        else:
            output.append(np.nan)
    return pd.Series(output, index=values.index)


def build_nasa_features(nasa_daily: pd.DataFrame) -> pd.DataFrame:
    """Build common and target-specific features for every frozen cell/date."""
    frames: list[pd.DataFrame] = []
    for cell, raw_group in nasa_daily.groupby("nasa_cell", sort=False):
        group = raw_group.sort_values("date").set_index("date").asfreq("D")
        features = pd.DataFrame(index=group.index)
        tmean = group["T2M"]
        rain = group["PRECTOTCORR"]
        degree_days = (tmean - 10.0).clip(lower=0).where(tmean.notna())
        active_temperature = tmean.where(tmean.gt(10.0), 0.0).where(tmean.notna())
        for days in COMMON_WINDOWS:
            roller = group.rolling(days, min_periods=days)
            features[f"nasa_tmean_{days}d"] = roller["T2M"].mean()
            features[f"nasa_tmin_{days}d"] = roller["T2M_MIN"].min()
            features[f"nasa_tmax_{days}d"] = roller["T2M_MAX"].max()
            features[f"nasa_rhmean_{days}d"] = roller["RH2M"].mean()
            features[f"nasa_rain_{days}d"] = roller["PRECTOTCORR"].sum()
            features[f"nasa_rain_days_{days}d"] = (
                rain.ge(0.1).where(rain.notna()).rolling(days, min_periods=days).sum()
            )
            features[f"nasa_active_temperature_sum_{days}d"] = active_temperature.rolling(
                days, min_periods=days
            ).sum()
            features[f"codling_degree_days_base10_{days}d"] = degree_days.rolling(
                days, min_periods=days
            ).sum()

        year = pd.Series(group.index.year, index=group.index)
        after_april = pd.Series(
            group.index >= pd.to_datetime(year.astype(str) + "-04-01"), index=group.index
        )
        seasonal_dd = degree_days.where(after_april, 0.0)
        features["codling_degree_days_from_apr1"] = seasonal_dd.groupby(year).cumsum()

        temperature_ok = tmean.between(5.0, 25.0)
        humid = group["RH2M"].ge(80.0)
        wet_proxy = (temperature_ok & humid & rain.ge(0.1)).where(
            tmean.notna() & group["RH2M"].notna() & rain.notna()
        )
        warm_humid = (temperature_ok & humid).where(tmean.notna() & group["RH2M"].notna())
        for days in (3, 7, 14):
            features[f"scab_warm_wet_proxy_days_{days}d"] = wet_proxy.astype(float).rolling(
                days, min_periods=days
            ).sum()
        for days in (7, 14):
            features[f"scab_warm_humid_days_{days}d"] = warm_humid.astype(float).rolling(
                days, min_periods=days
            ).sum()
        features["scab_rain_sum_3d"] = rain.rolling(3, min_periods=3).sum()
        features["scab_rain_sum_7d"] = rain.rolling(7, min_periods=7).sum()
        features["scab_proxy_current_run_days"] = _consecutive_true(wet_proxy)
        features["scab_days_since_proxy"] = _days_since_true(wet_proxy, features.index)
        features["nasa_cell"] = cell
        features["cutoff_date"] = features.index
        frames.append(features.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


def _lookup_feature(
    lookup: pd.Series, cells: pd.Series, dates: pd.Series
) -> np.ndarray:
    keys = pd.MultiIndex.from_arrays([cells.astype(str), pd.to_datetime(dates)])
    return lookup.reindex(keys).to_numpy(dtype=float)


def add_daily_features(
    decisions: pd.DataFrame,
    *,
    nasa_daily_path: str | Path,
    target_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Attach frozen, past-only NASA features and return the derived mapping."""
    frame = decisions.copy()
    day_of_year = frame["issue_date"].dt.dayofyear
    for harmonic in (1, 2):
        frame[f"doy_sin{harmonic}"] = np.sin(2 * np.pi * harmonic * day_of_year / 365.25)
        frame[f"doy_cos{harmonic}"] = np.cos(2 * np.pi * harmonic * day_of_year / 365.25)

    nasa = pd.read_parquet(nasa_daily_path).copy()
    nasa["date"] = pd.to_datetime(nasa["date"]).dt.normalize()
    mapping, mapping_audit = derive_nasa_mapping(frame, nasa)
    frame = frame.merge(
        mapping[["final_latitude", "final_longitude", "nasa_cell"]],
        on=["final_latitude", "final_longitude"],
        how="left",
        validate="many_to_one",
    )
    weather = build_nasa_features(nasa)
    frame = frame.merge(
        weather,
        left_on=["nasa_cell", "feature_cutoff_nasa_date"],
        right_on=["nasa_cell", "cutoff_date"],
        how="left",
        validate="many_to_one",
    ).drop(columns="cutoff_date")
    frame["common_weather_complete"] = frame[COMMON_WEATHER_FEATURES].notna().all(axis=1)

    if target_key == "codling_moth":
        seasonal = frame["codling_degree_days_from_apr1"]
        for threshold in (126.0, 230.0):
            frame[f"codling_dd_apr1_margin_{int(threshold)}"] = seasonal - threshold
            frame[f"codling_dd_apr1_pass_{int(threshold)}"] = seasonal.ge(threshold).astype(float)
        known = frame["first_adult_trap_positive_available_date"].notna() & frame["issue_date"].ge(
            frame["first_adult_trap_positive_available_date"]
        )
        frame["codling_adult_biofix_known"] = known.astype(float)
        frame["codling_days_since_adult_biofix"] = (
            frame["issue_date"] - frame["first_adult_trap_positive_date"]
        ).dt.days.where(known)
        seasonal_lookup = weather.set_index(["nasa_cell", "cutoff_date"])[
            "codling_degree_days_from_apr1"
        ]
        before_biofix = frame["first_adult_trap_positive_date"] - pd.Timedelta(days=1)
        base = _lookup_feature(seasonal_lookup, frame["nasa_cell"], before_biofix)
        biofix_dd = (seasonal.to_numpy(dtype=float) - base)
        biofix_dd = np.maximum(biofix_dd, 0.0)
        frame["codling_degree_days_since_adult_biofix"] = pd.Series(
            biofix_dd, index=frame.index
        ).where(known)
        for threshold in (126.0, 230.0):
            value = frame["codling_degree_days_since_adult_biofix"]
            frame[f"codling_dd_biofix_margin_{int(threshold)}"] = value - threshold
            frame[f"codling_dd_biofix_pass_{int(threshold)}"] = value.ge(threshold).where(
                value.notna()
            ).astype(float)
        target_features = CODLING_FEATURES
    elif target_key == "apple_scab":
        target_features = SCAB_FEATURES
    else:
        raise ValueError(f"Unsupported target_key: {target_key}")

    # Missing biofix is a genuine as-of state, not missing weather.  CatBoost
    # and the imputed linear control can consume it without changing the common
    # comparison population.
    frame["target_feature_weather_complete"] = frame["common_weather_complete"]
    frame["candidate_comparison_complete"] = frame["common_weather_complete"]
    audit = {
        "decision_rows": int(len(frame)),
        "service_field_days": int(frame["service_active"].sum()),
        "observable_target_rows": int(frame["target_observable"].sum()),
        "common_weather_computable_service_days": int(
            (frame["service_active"] & frame["common_weather_complete"]).sum()
        ),
        "common_weather_computable_fraction": float(
            frame.loc[frame["service_active"], "common_weather_complete"].mean()
        )
        if frame["service_active"].any()
        else np.nan,
        "feature_cutoff_days": int(
            (frame["issue_date"] - frame["feature_cutoff_nasa_date"]).dt.days.min()
        ),
        "target_feature_count": len(target_features),
        "mapping": mapping_audit,
        "daily_source_rows": int(len(nasa)),
        "daily_source_cells": int(nasa["nasa_cell"].nunique()),
        "daily_source_date_min": str(nasa["date"].min().date()),
        "daily_source_date_max": str(nasa["date"].max().date()),
    }
    return frame, mapping, audit
