"""Causal data preparation for the potato late-blight early-warning study.

The module models the first *registered* late-blight record.  It deliberately
keeps registry observability separate from biological absence and never uses a
future visit to decide when the simulated service runs.
"""
from __future__ import annotations

from datetime import time
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


POTATO_CODE = 166
LATE_BLIGHT_ID = 640
GEO_CLASSES = {"A_direct", "B_new_subtraction"}
CALENDAR_FEATURES = ["doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2"]
COMMON_VARIABLES = [
    "tmean",
    "tmin",
    "tmax",
    "rhmean",
    "rain",
    "rain_days",
    "active_temperature_sum",
]
COMMON_WINDOWS = (7, 14, 30)
NASA_COMMON_FEATURES = [f"nasa_{name}_{days}d" for days in COMMON_WINDOWS for name in COMMON_VARIABLES]
ERA_COMMON_FEATURES = [f"era_{name}_{days}d" for days in COMMON_WINDOWS for name in COMMON_VARIABLES]
EPISODE_FEATURES = [
    "era_hutton_pass_days_7d",
    "era_hutton_pass_days_14d",
    "era_hutton_pass_days_22d",
    "era_hutton_pairs_7d",
    "era_hutton_pairs_14d",
    "era_hutton_pairs_22d",
    "era_smith_pass_days_7d",
    "era_smith_pass_days_14d",
    "era_smith_pass_days_22d",
    "era_smith_pairs_7d",
    "era_smith_pairs_14d",
    "era_smith_pairs_22d",
    "era_high_humidity_hours_7d",
    "era_high_humidity_hours_14d",
    "era_wet_days_7d",
    "era_wet_days_14d",
    "era_dry_break_days_7d",
    "era_polyakov_t10_c",
    "era_polyakov_rh10_pct",
    "era_polyakov_p10_mm",
    "era_polyakov_temp_lower_margin",
    "era_polyakov_temp_upper_margin",
    "era_polyakov_rh_margin",
    "era_polyakov_rain_margin",
    "era_polyakov_critical_run_days",
    "era_days_since_hutton_pair_21d",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_snapshot_module(path: str | Path, module_name: str) -> ModuleType:
    """Load an audited snapshot source under a unique namespace."""
    source = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load snapshot module: {source}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses and a few other stdlib helpers resolve annotations through
    # sys.modules while a module is executed.  Register the isolated snapshot
    # namespace before exec_module, as the normal import machinery does.
    sys.modules[module_name] = module
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    return module


def prepare_potato_visits(csv_path: str | Path, snapshot_source: str | Path) -> tuple[pd.DataFrame, dict]:
    """Apply the conservative-v2 label semantics to potato records only."""
    snapshot = load_snapshot_module(snapshot_source, "agro_snapshot_vaad_diseases")
    raw = pd.read_csv(csv_path, low_memory=False)
    raw["observation_date"] = pd.to_datetime(raw["observation_date"], errors="coerce")
    raw["growth_stage_code"] = pd.to_numeric(raw["growth_stage_code"], errors="coerce")
    raw["crop_code"] = pd.to_numeric(raw["crop_code"], errors="coerce")

    parsed: list[list[dict] | None] = []
    invalid_json = 0
    for text_value in raw["detected_organisms"]:
        try:
            value = json.loads(text_value)
            if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
                raise ValueError("not an organism list")
        except (TypeError, ValueError, json.JSONDecodeError):
            invalid_json += 1
            value = None
        parsed.append(value)

    generic = raw["explicit_no_harmful_organisms"].map(snapshot.truth) | raw["organisms_raw"].fillna("").str.contains(
        r"(?:kaitīgie\s+organismi|kaitīgo\s+organismu\s+klātbūtne|slimības)\s+(?:nav|netika)\s+konstat",
        case=False,
        regex=True,
    )
    labels = [
        snapshot.label_organism(items, LATE_BLIGHT_ID, bool(no_harmful))
        if items is not None
        else ("invalid_json", np.nan, False)
        for items, no_harmful in zip(parsed, generic)
    ]
    statuses = pd.Series([item[0] for item in labels], index=raw.index)
    prevalence = pd.Series([item[1] for item in labels], index=raw.index)
    listed = pd.Series([item[2] for item in labels], index=raw.index)
    potato = raw[raw["crop_code"].eq(POTATO_CODE) | listed].copy()
    potato["label_status"] = statuses.loc[potato.index]
    potato["prevalence_pct"] = prevalence.loc[potato.index]
    potato["listed_in_source"] = listed.loc[potato.index]
    potato["host_supported"] = potato["crop_code"].eq(POTATO_CODE)
    potato["calendar_year"] = potato["observation_date"].dt.year.astype("Int64")
    potato["crop_season"] = potato["calendar_year"]
    potato["geo_valid"] = (
        potato["final_coordinate_class"].isin(GEO_CLASSES)
        & potato["final_field_uid"].notna()
        & potato["final_latitude"].between(55, 59)
        & potato["final_longitude"].between(20, 29)
    )
    potato["weather_latitude"] = (potato["final_latitude"] * 4).round() / 4
    potato["weather_longitude"] = (potato["final_longitude"] * 4).round() / 4
    potato["weather_cell"] = (
        potato["weather_latitude"].map(lambda value: f"{value:.2f}" if pd.notna(value) else "")
        + "_"
        + potato["weather_longitude"].map(lambda value: f"{value:.2f}" if pd.notna(value) else "")
    )
    potato["field_season"] = (
        potato["final_field_uid"].fillna(potato["observation_id"]).astype(str)
        + "_"
        + potato["crop_code"].astype("Int64").astype(str)
        + "_"
        + potato["crop_season"].astype("Int64").astype(str)
    )
    potato["exclusion_reason"] = ""
    potato.loc[~potato["geo_valid"], "exclusion_reason"] = "no_safe_final_coordinates"
    potato.loc[~potato["host_supported"], "exclusion_reason"] = "unsupported_host"
    potato.loc[potato["observation_date"].isna(), "exclusion_reason"] = "invalid_date"
    storage = potato["growth_stage_code"].eq(99) | potato["crop_stage_raw"].fillna("").str.contains(
        r"noliktav|uzglabāšanas", case=False, regex=True
    )
    potato.loc[storage, "exclusion_reason"] = "storage_or_postharvest"
    potato.loc[~potato["observation_date"].dt.month.between(5, 9), "exclusion_reason"] = "outside_potato_field_season"
    potato.loc[potato["growth_stage_code"].ge(97), "exclusion_reason"] = "dead_or_removed_canopy"
    usable = potato[potato["exclusion_reason"].eq("")].copy()

    visits: list[dict] = []
    keys = ["final_field_uid", "crop_code", "observation_date"]
    for _, group in usable.groupby(keys, sort=False, dropna=False):
        row = group.iloc[0].to_dict()
        group_statuses = set(group["label_status"])
        if "conflict" in group_statuses or (
            "positive" in group_statuses
            and group_statuses.intersection({"explicit_target_absent", "generic_absent"})
        ):
            status = "conflict"
        else:
            status = next(
                (
                    candidate
                    for candidate in ["positive", "explicit_target_absent", "generic_absent", "invalid_json", "unassessed"]
                    if candidate in group_statuses
                ),
                "unassessed",
            )
        stages = sorted(set(group["growth_stage_code"].dropna()))
        row.update(
            {
                "label_status": status,
                "source_row_count": int(len(group)),
                "visit_id": hashlib.sha256(
                    f"{row['final_field_uid']}|{int(row['crop_code'])}|{row['observation_date']}".encode()
                ).hexdigest()[:20],
                "final_coordinate_class": "A_direct"
                if group["final_coordinate_class"].eq("A_direct").any()
                else "B_new_subtraction",
                "growth_stage_code": stages[0] if len(stages) == 1 else np.nan,
                "stage_conflict": len(stages) > 1,
                "observed_bbch51": 51 in stages,
            }
        )
        visits.append(row)
    result = pd.DataFrame(visits).sort_values(["field_season", "observation_date", "visit_id"]).reset_index(drop=True)
    keep = [
        "visit_id",
        "observation_date",
        "field_season",
        "final_field_uid",
        "crop_code",
        "crop_name",
        "growth_stage_code",
        "stage_conflict",
        "observed_bbch51",
        "municipality",
        "parish",
        "final_latitude",
        "final_longitude",
        "final_coordinate_class",
        "final_coordinate_method",
        "weather_latitude",
        "weather_longitude",
        "weather_cell",
        "calendar_year",
        "crop_season",
        "label_status",
        "listed_in_source",
        "prevalence_pct",
        "source_row_count",
    ]
    result = result[keep]
    counts = result["label_status"].value_counts(dropna=False).to_dict()
    audit = {
        "input_path": str(Path(csv_path).resolve()),
        "sha256": sha256_file(csv_path),
        "source_rows": int(len(raw)),
        "source_columns": int(len(raw.columns)),
        "date_min": str(raw["observation_date"].min().date()),
        "date_max": str(raw["observation_date"].max().date()),
        "invalid_organism_json": int(invalid_json),
        "potato_or_late_blight_source_rows": int(len(potato)),
        "usable_source_rows": int(len(usable)),
        "deduplicated_visits": int(len(result)),
        "field_seasons": int(result["field_season"].nunique()),
        "fields": int(result["final_field_uid"].nunique()),
        "label_status_counts": {str(key): int(value) for key, value in counts.items()},
        "exclusion_counts": {
            str(key): int(value)
            for key, value in potato.loc[potato["exclusion_reason"].ne(""), "exclusion_reason"].value_counts().items()
        },
        "historical_reference": {
            "visits": 1706,
            "field_seasons": 488,
            "positive_visits": 788,
            "matches_current": bool(
                len(result) == 1706
                and result["field_season"].nunique() == 488
                and counts.get("positive", 0) == 788
            ),
        },
        "snapshot_adapter": str(Path(snapshot_source).resolve()),
    }
    return result, audit


def build_field_seasons(visits: pd.DataFrame, snapshot_date: str | pd.Timestamp) -> pd.DataFrame:
    """Create one registry row per field-season without redefining first events."""
    snapshot = pd.Timestamp(snapshot_date).normalize()
    rows: list[dict] = []
    for field_season, group in visits.groupby("field_season", sort=True):
        group = group.sort_values(["observation_date", "visit_id"])
        first = group.iloc[0]
        positives = group[group["label_status"].eq("positive")]
        event_date = positives["observation_date"].min() if len(positives) else pd.NaT
        prior = group[group["observation_date"].lt(event_date)] if pd.notna(event_date) else group.iloc[0:0]
        first_visit = pd.Timestamp(group["observation_date"].min()).normalize()
        last_visit = pd.Timestamp(group["observation_date"].max()).normalize()
        entry_date = first_visit + pd.Timedelta(days=1)
        year = int(first["crop_season"])
        nominal_end = pd.Timestamp(year=year, month=9, day=30)
        decision_end = min(nominal_end, snapshot) if year == snapshot.year else nominal_end
        if year > snapshot.year:
            decision_end = snapshot
        first_bbch = group.loc[group["observed_bbch51"], "observation_date"].min()
        event_available = event_date + pd.Timedelta(days=1) if pd.notna(event_date) else pd.NaT
        positive_at_entry = bool(pd.notna(event_date) and event_date == first_visit)
        warnable = bool(pd.notna(event_date) and not positive_at_entry and entry_date <= event_date - pd.Timedelta(days=3))
        all_actionable = bool(warnable and entry_date <= event_date - pd.Timedelta(days=10))
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
                "first_visit_label_status": first["label_status"],
                "positive_at_first_visit": positive_at_entry,
                "warnable_first_event": warnable,
                "all_actionable_leads_available": all_actionable,
                "entry_category": category,
                "previous_visit_before_event": bool(len(prior)),
                "previous_visit_date": prior["observation_date"].max() if len(prior) else pd.NaT,
                "previous_visit_gap_days": int((event_date - prior["observation_date"].max()).days)
                if len(prior)
                else np.nan,
                "previous_explicit_target_absence": bool(prior["label_status"].eq("explicit_target_absent").any()),
                "previous_generic_absence": bool(prior["label_status"].eq("generic_absent").any()),
                "first_observed_bbch51_date": first_bbch,
                "first_observed_bbch51_available_date": first_bbch + pd.Timedelta(days=1)
                if pd.notna(first_bbch)
                else pd.NaT,
                "visit_count": int(len(group)),
                "registration_followup_days": int((last_visit - first_visit).days),
                "coordinate_scope": "A_direct"
                if group["final_coordinate_class"].eq("A_direct").all()
                else "includes_B",
                "final_latitude": float(first["final_latitude"]),
                "final_longitude": float(first["final_longitude"]),
                "weather_cell": first["weather_cell"],
                "status_counts": json.dumps(
                    {str(key): int(value) for key, value in group["label_status"].value_counts().items()},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "censoring_basis": "last_registry_visit_for_outcome_mask_only",
            }
        )
    return pd.DataFrame(rows).sort_values(["season", "field_season"]).reset_index(drop=True)


def _issue_timestamp(day: pd.Timestamp, timezone: str, issue_time: str) -> pd.Timestamp:
    clock = time.fromisoformat(issue_time)
    naive = pd.Timestamp.combine(pd.Timestamp(day).date(), clock)
    return naive.tz_localize(ZoneInfo(timezone))


def build_daily_decisions(
    field_seasons: pd.DataFrame,
    timezone: str = "Europe/Riga",
    issue_time: str = "08:00:00",
    minimum_lead: int = 3,
    maximum_lead: int = 10,
) -> pd.DataFrame:
    """Generate a service calendar independently of future inspection dates."""
    rows: list[dict] = []
    for season in field_seasons.itertuples(index=False):
        start = pd.Timestamp(season.first_visit_available_date)
        end = pd.Timestamp(season.decision_end_date)
        if end < start:
            continue
        for issue_date in pd.date_range(start, end, freq="D"):
            event_date = pd.Timestamp(season.first_recorded_event_date) if pd.notna(season.first_recorded_event_date) else pd.NaT
            event_available = (
                pd.Timestamp(season.first_recorded_event_available_date)
                if pd.notna(season.first_recorded_event_available_date)
                else pd.NaT
            )
            service_active = bool(pd.isna(event_available) or issue_date < event_available)
            target_class = "unknown"
            target_observable = False
            outcome_reason = "insufficient_registry_followup"
            days_to_event = np.nan
            if pd.notna(event_date) and issue_date <= event_date:
                days_to_event = int((event_date - issue_date).days)
                if 0 <= days_to_event <= 2:
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
            elif pd.isna(event_date) and issue_date + pd.Timedelta(days=maximum_lead) <= pd.Timestamp(season.last_visit_date):
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
                    "issued_at": _issue_timestamp(issue_date, timezone, issue_time).isoformat(),
                    "feature_cutoff_nasa_date": issue_date - pd.Timedelta(days=2),
                    "feature_cutoff_era_common_date": issue_date - pd.Timedelta(days=2),
                    # The first-cycle C2-C5 ablation uses one conservative
                    # common cutoff.  ERA5 may be available sooner in a real
                    # service, but no historical publication log proves that.
                    "feature_cutoff_era_episode_date": issue_date - pd.Timedelta(days=2),
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
                    "first_observed_bbch51_date": season.first_observed_bbch51_date,
                    "first_observed_bbch51_available_date": season.first_observed_bbch51_available_date,
                    "final_latitude": season.final_latitude,
                    "final_longitude": season.final_longitude,
                    "weather_cell": season.weather_cell,
                    "outcome_semantics": "registration_proxy_not_biological_absence",
                }
            )
    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values(["season", "field_season", "issue_date"]).reset_index(drop=True)
    return result


def _rolling_common(
    daily: pd.DataFrame,
    cell_column: str,
    prefix: str,
    columns: dict[str, str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for cell, group in daily.groupby(cell_column, sort=False):
        group = group.sort_values("date").set_index("date").asfreq("D")
        features = pd.DataFrame(index=group.index)
        for days in COMMON_WINDOWS:
            roller = group.rolling(days, min_periods=days)
            features[f"{prefix}_tmean_{days}d"] = roller[columns["tmean"]].mean()
            features[f"{prefix}_tmin_{days}d"] = roller[columns["tmin"]].min()
            features[f"{prefix}_tmax_{days}d"] = roller[columns["tmax"]].max()
            features[f"{prefix}_rhmean_{days}d"] = roller[columns["rhmean"]].mean()
            features[f"{prefix}_rain_{days}d"] = roller[columns["rain"]].sum()
            rain = group[columns["rain"]]
            features[f"{prefix}_rain_days_{days}d"] = rain.ge(0.1).where(rain.notna()).rolling(days, min_periods=days).sum()
            tmean = group[columns["tmean"]]
            active = tmean.where(tmean.gt(10), 0).where(tmean.notna())
            features[f"{prefix}_active_temperature_sum_{days}d"] = active.rolling(days, min_periods=days).sum()
        features[cell_column] = cell
        features["cutoff_date"] = features.index
        frames.append(features.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


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


def _days_since_recent_true(values: pd.Series, index: pd.DatetimeIndex, limit: int = 30) -> pd.Series:
    result: list[float] = []
    last: pd.Timestamp | None = None
    for date_value, value in zip(index, values):
        if pd.notna(value) and bool(value):
            last = pd.Timestamp(date_value)
            result.append(0.0)
        elif last is not None and (pd.Timestamp(date_value) - last).days <= limit:
            result.append(float((pd.Timestamp(date_value) - last).days))
        else:
            result.append(np.nan)
    return pd.Series(result, index=values.index)


def _episode_features(era_daily: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for cell, group in era_daily.groupby("weather_cell", sort=False):
        group = group.sort_values("date").set_index("date").asfreq("D")
        accepted = group["accepted"].eq(True)
        pass_day = group["day_status"].eq("pass").where(group["day_status"].notna())
        pair = (pass_day.eq(True) & pass_day.shift(1).eq(True)).where(pass_day.notna() & pass_day.shift(1).notna())
        smith_day = (
            group["minimum_temperature_c"].ge(10) & group["high_humidity_hours"].ge(11)
        ).where(accepted)
        smith_pair = (
            smith_day.eq(True) & smith_day.shift(1).eq(True)
        ).where(smith_day.notna() & smith_day.shift(1).notna())
        wet_day = (
            group["relative_humidity_mean_pct"].ge(75) & group["precipitation_sum_mm"].ge(0.1)
        ).where(accepted)
        dry_break = (
            group["relative_humidity_mean_pct"].lt(75) | group["precipitation_sum_mm"].lt(0.1)
        ).where(accepted)
        features = pd.DataFrame(index=group.index)
        for days in (7, 14, 22):
            features[f"era_hutton_pass_days_{days}d"] = pass_day.astype(float).rolling(days, min_periods=days).sum()
            features[f"era_hutton_pairs_{days}d"] = pair.astype(float).rolling(days, min_periods=days).sum()
            features[f"era_smith_pass_days_{days}d"] = smith_day.astype(float).rolling(days, min_periods=days).sum()
            features[f"era_smith_pairs_{days}d"] = smith_pair.astype(float).rolling(days, min_periods=days).sum()
        for days in (7, 14):
            features[f"era_high_humidity_hours_{days}d"] = group["high_humidity_hours"].rolling(days, min_periods=days).sum()
            features[f"era_wet_days_{days}d"] = wet_day.astype(float).rolling(days, min_periods=days).sum()
        features["era_dry_break_days_7d"] = dry_break.astype(float).rolling(7, min_periods=7).sum()
        complete10 = accepted.rolling(10, min_periods=10).sum().eq(10)
        features["era_polyakov_t10_c"] = group["temperature_mean_c"].rolling(10, min_periods=10).mean().where(complete10)
        features["era_polyakov_rh10_pct"] = group["relative_humidity_mean_pct"].rolling(10, min_periods=10).mean().where(complete10)
        features["era_polyakov_p10_mm"] = group["precipitation_sum_mm"].rolling(10, min_periods=10).sum().where(complete10)
        features["era_polyakov_temp_lower_margin"] = features["era_polyakov_t10_c"] - 13.0
        features["era_polyakov_temp_upper_margin"] = 20.0 - features["era_polyakov_t10_c"]
        features["era_polyakov_rh_margin"] = features["era_polyakov_rh10_pct"] - 75.0
        features["era_polyakov_rain_margin"] = features["era_polyakov_p10_mm"] - 20.0
        critical = (
            features["era_polyakov_t10_c"].between(13, 20)
            & features["era_polyakov_rh10_pct"].ge(75)
            & features["era_polyakov_p10_mm"].ge(20)
        ).where(complete10)
        features["era_polyakov_critical_run_days"] = _consecutive_true(critical)
        since_pair = _days_since_recent_true(pair, features.index, 21)
        pair_history_known = pair.notna().rolling(22, min_periods=22).sum().eq(22)
        features["era_days_since_hutton_pair_21d"] = since_pair.where(since_pair.notna(), 22.0).where(pair_history_known)
        features["era_hutton_pair_now"] = pair.astype("Float64")
        features["era_smith_pair_now"] = smith_pair.astype("Float64")
        features["weather_cell"] = cell
        features["cutoff_date"] = features.index
        frames.append(features.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


def add_daily_features(
    decisions: pd.DataFrame,
    nasa_daily_path: str | Path,
    nasa_mapping_path: str | Path,
    era_daily_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Attach only weather ending before each issue time."""
    frame = decisions.copy()
    day_of_year = frame["issue_date"].dt.dayofyear
    for harmonic in (1, 2):
        frame[f"doy_sin{harmonic}"] = np.sin(2 * np.pi * harmonic * day_of_year / 365.25)
        frame[f"doy_cos{harmonic}"] = np.cos(2 * np.pi * harmonic * day_of_year / 365.25)

    mapping = pd.read_parquet(nasa_mapping_path)
    frame = frame.merge(
        mapping[["final_latitude", "final_longitude", "nasa_cell"]],
        on=["final_latitude", "final_longitude"],
        how="left",
        validate="many_to_one",
    )
    nasa = pd.read_parquet(nasa_daily_path)
    nasa_features = _rolling_common(
        nasa,
        "nasa_cell",
        "nasa",
        {
            "tmean": "T2M",
            "tmin": "T2M_MIN",
            "tmax": "T2M_MAX",
            "rhmean": "RH2M",
            "rain": "PRECTOTCORR",
        },
    )
    frame = frame.merge(
        nasa_features,
        left_on=["nasa_cell", "feature_cutoff_nasa_date"],
        right_on=["nasa_cell", "cutoff_date"],
        how="left",
        validate="many_to_one",
    ).drop(columns="cutoff_date")

    era = pd.read_parquet(era_daily_path)
    era_common = _rolling_common(
        era,
        "weather_cell",
        "era",
        {
            "tmean": "temperature_mean_c",
            "tmin": "minimum_temperature_c",
            "tmax": "maximum_temperature_c",
            "rhmean": "relative_humidity_mean_pct",
            "rain": "precipitation_sum_mm",
        },
    )
    frame = frame.merge(
        era_common,
        left_on=["weather_cell", "feature_cutoff_era_common_date"],
        right_on=["weather_cell", "cutoff_date"],
        how="left",
        validate="many_to_one",
    ).drop(columns="cutoff_date")
    episodes = _episode_features(era)
    frame = frame.merge(
        episodes,
        left_on=["weather_cell", "feature_cutoff_era_episode_date"],
        right_on=["weather_cell", "cutoff_date"],
        how="left",
        validate="many_to_one",
    ).drop(columns="cutoff_date")
    frame["nasa_common_complete"] = frame[NASA_COMMON_FEATURES].notna().all(axis=1)
    frame["era_common_complete"] = frame[ERA_COMMON_FEATURES].notna().all(axis=1)
    frame["common_weather_complete"] = frame["nasa_common_complete"] & frame["era_common_complete"]
    frame["episode_weather_complete"] = frame[EPISODE_FEATURES].notna().all(axis=1)
    frame["candidate_comparison_complete"] = (
        frame["common_weather_complete"] & frame["episode_weather_complete"]
    )
    frame["hutton_score"] = frame["era_hutton_pair_now"].astype(float)
    frame["smith_score"] = frame["era_smith_pair_now"].astype(float)
    audit = {
        "decision_rows": int(len(frame)),
        "service_field_days": int(frame["evaluation_field_day"].sum()),
        "observable_target_rows": int(frame["target_observable"].sum()),
        "common_weather_observable_rows": int((frame["target_observable"] & frame["common_weather_complete"]).sum()),
        "nasa_computable_service_days": int((frame["evaluation_field_day"] & frame["nasa_common_complete"]).sum()),
        "era_common_computable_service_days": int((frame["evaluation_field_day"] & frame["era_common_complete"]).sum()),
        "episode_computable_service_days": int((frame["evaluation_field_day"] & frame["episode_weather_complete"]).sum()),
        "candidate_comparison_service_days": int(
            (frame["evaluation_field_day"] & frame["candidate_comparison_complete"]).sum()
        ),
    }
    return frame, audit


def add_polyakov_baseline(
    decisions: pd.DataFrame,
    field_seasons: pd.DataFrame,
    era_daily_path: str | Path,
    snapshot_late_blight_source: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate the documented phase-gated rule with causal signal availability."""
    late_blight = load_snapshot_module(snapshot_late_blight_source, "agro_snapshot_late_blight")
    era = pd.read_parquet(era_daily_path)
    frame = decisions.copy()
    frame["polyakov_score"] = np.nan
    frame["polyakov_status"] = "not_evaluable_missing_observed_bbch51"
    season_lookup = field_seasons.set_index("field_season")
    evaluable_seasons = 0
    for field_season, index in frame.groupby("field_season").groups.items():
        season = season_lookup.loc[field_season]
        activation = season["first_observed_bbch51_date"]
        activation_available = season["first_observed_bbch51_available_date"]
        if pd.isna(activation):
            continue
        evaluable_seasons += 1
        subset = era[era["weather_cell"].eq(season["weather_cell"])].copy()
        subset = subset[subset["date"].dt.year.eq(int(season["season"]))]
        if subset.empty:
            frame.loc[index, "polyakov_status"] = "not_evaluable_missing_weather"
            continue
        daily = subset[
            ["date", "temperature_mean_c", "relative_humidity_mean_pct", "precipitation_sum_mm", "accepted"]
        ].copy()
        daily["date"] = daily["date"].dt.date
        classified = late_blight.classify_polyakov_windows(daily, pd.Timestamp(activation).date())
        classified["cutoff_date"] = pd.to_datetime(classified["date"])
        classified = classified.set_index("cutoff_date")
        for row_index in index:
            issue = pd.Timestamp(frame.at[row_index, "issue_date"])
            if issue < pd.Timestamp(activation_available):
                frame.at[row_index, "polyakov_status"] = "phenophase_not_yet_available"
                continue
            cutoff = pd.Timestamp(frame.at[row_index, "feature_cutoff_era_episode_date"])
            if cutoff < pd.Timestamp(activation):
                frame.at[row_index, "polyakov_status"] = "weather_cutoff_before_activation"
                continue
            if cutoff not in classified.index:
                frame.at[row_index, "polyakov_status"] = "not_evaluable_missing_weather"
                continue
            state = classified.loc[cutoff]
            if isinstance(state, pd.DataFrame):
                state = state.iloc[-1]
            status = str(state["status"])
            frame.at[row_index, "polyakov_status"] = status
            if status == "INSUFFICIENT_DATA":
                continue
            frame.at[row_index, "polyakov_score"] = float(status in {"OUTBREAK_EXPECTED", "PROLONGED_RISK"})
    audit = {
        "field_seasons_with_observed_bbch51": int(evaluable_seasons),
        "service_days_computable": int((frame["evaluation_field_day"] & frame["polyakov_score"].notna()).sum()),
        "warnable_events_with_any_computable_day": int(
            frame[
                frame["warnable_first_event"]
                & frame["evaluation_field_day"]
                & frame["polyakov_score"].notna()
                & frame["days_to_first_recorded_event"].between(3, 10)
            ]["field_season"].nunique()
        ),
    }
    return frame, audit
