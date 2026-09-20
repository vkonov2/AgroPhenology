"""Retrospective A/B/C early-warning experiment with archived GFS releases.

The experiment reuses the first-cycle event registry and complete daily
decision calendar. It never loads or changes the frozen C6 bundle. Archived
forecast features are joined by local decision date and a public 0.25-degree
GFS cell. Model thresholds are selected on each fold's validation years, then
the unchanged sequential P0 simulator is replayed on the full calendar.

GDEX exposes model initialisation times but no proven historical publication
timestamps. The four-hour main lag and seven-hour sensitivity lag are explicit
assumptions, not observed publication facts.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .early_warning_core import CALENDAR_FEATURES, NASA_COMMON_FEATURES, sha256_file
from .early_warning_cycle2_diagnostics import event_intersections
from .early_warning_cycle2_reporting import (
    leave_one_year_out_cycle2,
    paired_year_bootstrap_cycle2,
)
from .early_warning_cycle3_pipeline import verify_frozen_run
from .early_warning_models import (
    Policy,
    TARGET_TO_INT,
    burden_metrics,
    event_metrics,
    fit_model,
    score_model,
    select_threshold,
    simulate_policy,
)
from .early_warning_reporting import aggregate_pooled_metrics
from .gfs_archive import canonical_gfs_cell_id
from .gfs_feature_builder import (
    AUDITED_DDS_ACCESS_MODE,
    CHECKPOINT_SCHEMA_VERSION,
    FAST_NCSS_ACCESS_MODE,
    PLAN_FILE_NAME,
    PLAN_MANIFEST_FILE_NAME,
    PLAN_SCHEMA_VERSION,
    SOURCE_MANIFEST_FILE_NAME,
    SOURCE_MANIFEST_SCHEMA_VERSION,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V3 = REPO_ROOT / "results/late_blight_early_warning/20260910_first_cycle_v3"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/late_blight_early_warning"
V3_MANIFEST_SHA256 = "c7976317e6d94b8d1ca55b0bd618226c769914280bd7896ca36b5e0ca2cf3c80"

# Frozen contract names. Step-count surrogates are deliberately excluded:
# selected values are six-hour synoptic samples and are not Hutton hours.
GFS_FORECAST_FEATURES = [
    f"fcst_{name}_{band}"
    for band in ("d1_3", "d4_7")
    for name in ("t_mean", "t_min", "t_max", "rh_mean", "rh_max", "precip_sum")
]

GFS_REQUIRED_COLUMNS = [
    "issue_date",
    "issue_time_utc",
    "gfs_cell_id",
    "availability_scenario_hours",
    "selected_init_utc",
    "assumed_available_at_utc",
    "selection_rule_id",
    "source_dataset_id",
    "source_hashes_json",
    "first_lead_h",
    "last_lead_h",
    "native_step_hours",
    "expected_steps_1_3d",
    "observed_steps_1_3d",
    "expected_steps_4_7d",
    "observed_steps_4_7d",
    "complete_1_3d",
    "complete_4_7d",
    "forecast_available",
]

MODEL_A = "A"
MODEL_A_MATCHED_C0 = "A_matched_C0"
MODEL_A_STRONG = "A_strong_calendar_catboost"
MODEL_B = "B"
MODEL_C = "C"
MODEL_CODES = (MODEL_A, MODEL_A_MATCHED_C0, MODEL_A_STRONG, MODEL_B, MODEL_C)
PRIMARY_MODELS = (MODEL_A, MODEL_B, MODEL_C)
PRIMARY_SCOPE = "gfs_common_days"
SERVICE_SCOPE = "service_calendar"

_FORBIDDEN_FEATURE_FRAGMENTS = (
    "target",
    "event",
    "label",
    "visit",
    "outcome",
    "field_uid",
    "field_season",
    "latitude",
    "longitude",
    "prevalence",
)

_EXPECTED_FOLDS = [
    {"id": "test_2020", "train_years": [2015, 2017], "validation_years": [2018, 2019], "test_years": [2020, 2020]},
    {"id": "test_2021", "train_years": [2015, 2018], "validation_years": [2019, 2020], "test_years": [2021, 2021]},
    {"id": "test_2022", "train_years": [2015, 2019], "validation_years": [2020, 2021], "test_years": [2022, 2022]},
    {"id": "test_2023", "train_years": [2015, 2020], "validation_years": [2021, 2022], "test_years": [2023, 2023]},
    {"id": "test_2024", "train_years": [2015, 2021], "validation_years": [2022, 2023], "test_years": [2024, 2024]},
    {"id": "test_2025", "train_years": [2015, 2022], "validation_years": [2023, 2024], "test_years": [2025, 2025]},
]

_EXPECTED_CATBOOST_FIXED = {
    "iterations": 220,
    "depth": 3,
    "learning_rate": 0.05,
    "l2_leaf_reg": 8.0,
    "loss_function": "MultiClass",
    "random_seed": 20260910,
}

_KNOWN_SOURCE_ACCESS_MODES = {AUDITED_DDS_ACCESS_MODE, FAST_NCSS_ACCESS_MODE}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _year_mask(frame: pd.DataFrame, bounds: Sequence[int]) -> pd.Series:
    if len(bounds) != 2:
        raise ValueError("year bounds must contain exactly two values")
    return frame["season"].between(int(bounds[0]), int(bounds[1]))


def _bool(values: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    converted = values.map(
        {
            True: True,
            False: False,
            "True": True,
            "False": False,
            "true": True,
            "false": False,
            1: True,
            0: False,
        }
    )
    if (values.notna() & converted.isna()).any():
        raise ValueError(f"{name} contains a non-boolean value")
    return converted.astype("boolean").fillna(False).astype(bool)


def assert_safe_feature_names(features: Sequence[str]) -> list[str]:
    """Reject evaluator-only, outcome, or field-identifying model inputs."""

    names = [str(value) for value in features]
    if not names or len(names) != len(set(names)):
        raise ValueError("feature list must be non-empty and unique")
    unsafe = [
        name
        for name in names
        if any(fragment in name.lower() for fragment in _FORBIDDEN_FEATURE_FRAGMENTS)
    ]
    if unsafe:
        raise ValueError(f"evaluator-only or identifying model features are forbidden: {unsafe}")
    return names


def contract_feature_columns(contract: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Resolve frozen NASA and GFS feature whitelists without implicit expansion."""

    forecast = assert_safe_feature_names(contract.get("forecast_features", GFS_FORECAST_FEATURES))
    configured_past = contract.get("past_weather", {}).get("features", "NASA_COMMON_FEATURES")
    if configured_past == "NASA_COMMON_FEATURES":
        past = list(NASA_COMMON_FEATURES)
    elif isinstance(configured_past, Sequence) and not isinstance(configured_past, str):
        past = assert_safe_feature_names(configured_past)
    else:
        raise ValueError("past_weather.features must be NASA_COMMON_FEATURES or a column list")
    return assert_safe_feature_names(past), forecast


def availability_scenarios(contract: Mapping[str, Any]) -> tuple[int, tuple[int, ...]]:
    settings = contract["forecast_availability"]
    primary = int(settings["main_assumed_hours_after_initialization"])
    sensitivity = int(settings["delay_sensitivity_total_hours_after_initialization"])
    scenarios = tuple(dict.fromkeys((primary, sensitivity)))
    if any(value < 0 for value in scenarios):
        raise ValueError("forecast availability lags must be non-negative")
    return primary, scenarios


def validate_contract(contract: Mapping[str, Any]) -> None:
    """Fail before scoring if the frozen experiment meaning has drifted."""

    def require_equal(actual: Any, expected: Any, label: str) -> None:
        if actual != expected:
            raise ValueError(f"frozen contract {label} changed: {actual!r} != {expected!r}")

    require_equal(contract.get("contract_version"), "1.0.1", "version")
    if contract.get("experiment_id") != "gfs_archive_forecast_increment":
        raise ValueError("unexpected GFS experiment_id")
    if not bool(contract.get("frozen_before_external_scoring")):
        raise ValueError("contract must be frozen before external scoring")
    require_equal(
        contract.get("research_target"),
        "first_recorded_potato_late_blight_in_field_season",
        "research target",
    )
    require_equal(contract.get("timezone"), "Europe/Riga", "timezone")
    require_equal(contract.get("daily_issue_time"), "08:00:00", "daily issue time")
    if contract.get("timeliness_window_days") != {"minimum": 3, "maximum": 10}:
        raise ValueError("the event-success window must remain 3-10 days")
    source = contract.get("source_dataset", {})
    for key, expected in {
        "id": "d084001",
        "doi": "10.5065/D65D8PWK",
        "gdex_url": "https://gdex.ucar.edu/datasets/d084001/",
        "thredds_catalog": "https://tds.gdex.ucar.edu/thredds/catalog/catalog_d084001.xml",
        "access_method": "unauthenticated_THREDDS_NCSS_regional_subset",
        "archive_start": "2015-01-15",
        "spatial_grid_degrees": 0.25,
        "field_to_public_grid_join": "local_private_only",
    }.items():
        require_equal(source.get(key), expected, f"source_dataset.{key}")

    availability = contract.get("forecast_availability", {})
    for key, expected in {
        "selection_rule_id": "latest_cycle_assumed_published_before_issue_v1",
        "cycle_hours_utc": [0, 6, 12, 18],
        "historical_publication_log_available": False,
        "publication_time_status": "assumption_not_observed_historical_timestamp",
        "main_assumed_hours_after_initialization": 4,
        "delay_sensitivity_total_hours_after_initialization": 7,
        "no_backdating": True,
        "no_reanalysis_fill": True,
    }.items():
        require_equal(availability.get(key), expected, f"forecast_availability.{key}")
    sampling = contract["forecast_sampling"]
    require_equal(
        sampling.get("native_archive_steps_hours_through_168"),
        3,
        "forecast_sampling.native archive step",
    )
    if int(sampling["selected_step_hours"]) != 6:
        raise ValueError("the fixed experiment uses selected six-hour GFS samples")
    observed_bands = [
        (
            str(item["id"]),
            int(item["lead_hours_after_decision_open"]),
            int(item["lead_hours_after_decision_closed"]),
        )
        for item in sampling["bands"]
    ]
    if observed_bands != [("d1_3", 0, 72), ("d4_7", 72, 168)]:
        raise ValueError("forecast bands must remain (0,72] and (72,168] from decision time")
    for key, expected in {
        "precipitation_interval_assignment": "six-hour accumulation is assigned by its interval end valid time",
        "complete_case_rule": "all expected six-hour slots and all required variables in both bands",
        "spatial_join": "nearest 0.25-degree GFS grid cell",
    }.items():
        require_equal(sampling.get(key), expected, f"forecast_sampling.{key}")
    if bool(contract.get("optuna", {}).get("enabled", False)):
        raise ValueError("Optuna is outside this fixed first-pass experiment")
    if list(contract.get("primary_comparison", [])) != ["A", "B", "C"]:
        raise ValueError("primary comparison must remain A/B/C")
    policy = contract["notification_policy"]
    if policy.get("family") != "existing_P0_only":
        raise ValueError("only the existing P0 policy family is allowed")
    if int(policy["active_days_per_message"]) != 7 or int(policy["cooldown_days"]) != 15:
        raise ValueError("P0 active/cooldown settings must remain 7/15 days")
    require_equal(
        policy.get("threshold_selection"),
        "validation_only_fixed_grid_and_score_quantiles",
        "notification threshold selection",
    )
    budget = policy.get("research_budget", {})
    require_equal(
        budget.get("messages_per_30_field_days_max"),
        2.0,
        "message budget",
    )
    require_equal(
        budget.get("active_alarm_fraction_max"),
        0.5,
        "active-alarm budget",
    )
    past = contract.get("past_weather", {})
    for key, expected in {
        "source": "existing_NASA_POWER_daily_features_from_parent_run",
        "features": "NASA_COMMON_FEATURES",
        "cutoff_days_before_issue": 2,
        "same_features_for_B_and_C": True,
    }.items():
        require_equal(past.get(key), expected, f"past_weather.{key}")
    _, forecast = contract_feature_columns(contract)
    if forecast != GFS_FORECAST_FEATURES:
        raise ValueError("forecast feature whitelist must remain the frozen 12 aggregates")
    require_equal(contract.get("rolling_origin_folds"), _EXPECTED_FOLDS, "rolling folds")
    require_equal(contract.get("catboost_fixed"), _EXPECTED_CATBOOST_FIXED, "CatBoost settings")
    require_equal(contract.get("random_seed"), 20260910, "global random seed")
    availability_scenarios(contract)


def _normalise_forecast_schema(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    aliases = {
        "selected_init_time_utc": "selected_init_utc",
        "availability_assumption_id": "selection_rule_id",
        "expected_slots_d1_3": "expected_steps_1_3d",
        "observed_slots_d1_3": "observed_steps_1_3d",
        "complete_d1_3": "complete_1_3d",
        "expected_slots_d4_7": "expected_steps_4_7d",
        "observed_slots_d4_7": "observed_steps_4_7d",
        "complete_d4_7": "complete_4_7d",
        "selected_step_hours": "native_step_hours",
    }
    for source, target in aliases.items():
        if target not in result and source in result:
            result = result.rename(columns={source: target})
    if "source_dataset_id" not in result:
        result["source_dataset_id"] = "d084001"
    if "availability_scenario_hours" not in result and {
        "selected_init_utc",
        "assumed_available_at_utc",
    }.issubset(result.columns):
        initialisation = pd.to_datetime(result["selected_init_utc"], errors="coerce", utc=True)
        available = pd.to_datetime(result["assumed_available_at_utc"], errors="coerce", utc=True)
        result["availability_scenario_hours"] = (
            (available - initialisation).dt.total_seconds() / 3600.0
        )
    if "forecast_available" not in result and {
        "complete_1_3d",
        "complete_4_7d",
    }.issubset(result.columns):
        result["forecast_available"] = (
            _bool(result["complete_1_3d"], name="complete_1_3d")
            & _bool(result["complete_4_7d"], name="complete_4_7d")
        )
    return result


def _latest_assumed_available_init(issue_time: pd.Series, lag_hours: pd.Series) -> pd.Series:
    cutoff = issue_time - pd.to_timedelta(lag_hours, unit="h")
    return cutoff.dt.floor("6h")


def validate_forecast_feature_table(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = GFS_FORECAST_FEATURES,
) -> pd.DataFrame:
    """Validate one immutable aggregate row per issue/cell/total-lag scenario."""

    features = assert_safe_feature_names(feature_columns)
    result = _normalise_forecast_schema(frame)
    missing = sorted(set(GFS_REQUIRED_COLUMNS).union(features).difference(result.columns))
    if missing:
        raise ValueError(f"GFS forecast feature table misses required columns: {missing}")
    forbidden = {
        "field_season",
        "field_uid",
        "final_latitude",
        "final_longitude",
        "target_class",
        "first_recorded_event_date",
    }.intersection(result.columns)
    if forbidden:
        raise ValueError(f"public forecast table contains field/outcome columns: {sorted(forbidden)}")

    result["issue_date"] = pd.to_datetime(result["issue_date"], errors="raise").dt.normalize()
    for column in (
        "issue_time_utc",
        "selected_init_utc",
        "assumed_available_at_utc",
        "publication_time_utc",
    ):
        if column in result:
            result[column] = pd.to_datetime(result[column], errors="coerce", utc=True)
    if result["issue_date"].isna().any() or result["issue_time_utc"].isna().any():
        raise ValueError("issue_date and issue_time_utc must be present")
    result["gfs_cell_id"] = result["gfs_cell_id"].astype("string").str.strip()
    if result["gfs_cell_id"].isna().any() or result["gfs_cell_id"].eq("").any():
        raise ValueError("gfs_cell_id must be present")
    scenario = pd.to_numeric(result["availability_scenario_hours"], errors="raise")
    if scenario.isna().any() or (scenario < 0).any() or (scenario % 1 != 0).any():
        raise ValueError("availability_scenario_hours must be a non-negative integer total lag")
    result["availability_scenario_hours"] = scenario.astype(int)
    for column in ("forecast_available", "complete_1_3d", "complete_4_7d"):
        result[column] = _bool(result[column], name=column)

    numeric = (
        "first_lead_h",
        "last_lead_h",
        "native_step_hours",
        "expected_steps_1_3d",
        "observed_steps_1_3d",
        "expected_steps_4_7d",
        "observed_steps_4_7d",
    )
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if not result["native_step_hours"].eq(6).all():
        raise ValueError("feature table must contain selected six-hour samples")
    for band, fixed_expected in {"1_3d": 12, "4_7d": 16}.items():
        expected = result[f"expected_steps_{band}"]
        observed = result[f"observed_steps_{band}"]
        complete = result[f"complete_{band}"]
        if not expected.eq(fixed_expected).all():
            raise ValueError(f"{band} must declare {fixed_expected} six-hour valid slots")
        if observed.isna().any() or (observed < 0).any() or (observed > expected).any():
            raise ValueError(f"observed_steps_{band} is outside [0, expected]")
        if not complete.eq(observed.eq(expected)).all():
            raise ValueError(f"complete_{band} disagrees with observed valid-slot count")
    if not result["forecast_available"].eq(
        result["complete_1_3d"] & result["complete_4_7d"]
    ).all():
        raise ValueError("forecast_available must equal both band-completeness flags")

    available = result["forecast_available"]
    if result.loc[available, ["selected_init_utc", "assumed_available_at_utc"]].isna().any().any():
        raise ValueError("available forecasts require initialization and assumed availability")
    actual_lag = (
        result.loc[available, "assumed_available_at_utc"]
        - result.loc[available, "selected_init_utc"]
    ).dt.total_seconds() / 3600.0
    if not np.allclose(
        actual_lag.to_numpy(dtype=float),
        result.loc[available, "availability_scenario_hours"].to_numpy(dtype=float),
        atol=1e-9,
        rtol=0,
    ):
        raise ValueError("availability_scenario_hours must be total init-to-publication-assumption lag")
    if (
        result.loc[available, "assumed_available_at_utc"]
        > result.loc[available, "issue_time_utc"]
    ).any():
        raise ValueError("selected forecast was not assumed available at decision time")
    expected_init = _latest_assumed_available_init(
        result.loc[available, "issue_time_utc"],
        result.loc[available, "availability_scenario_hours"],
    )
    if not result.loc[available, "selected_init_utc"].eq(expected_init).all():
        raise ValueError("selected_init_utc is not the latest cycle allowed by the frozen rule")
    if "publication_time_utc" in result and result["publication_time_utc"].notna().any():
        raise ValueError("GDEX d084001 has no proven historical publication timestamp")
    if not result.loc[available, "source_dataset_id"].astype(str).eq("d084001").all():
        raise ValueError("available rows must come from GDEX dataset d084001")
    selection = result.loc[available, "selection_rule_id"].astype(str)
    if selection.str.strip().eq("").any() or not selection.str.contains(
        "latest_cycle_assumed_published_before_issue", regex=False
    ).all():
        raise ValueError("selection_rule_id does not identify the frozen latest-cycle rule")
    hashes = result.loc[available, "source_hashes_json"].astype(str)
    if hashes.str.strip().isin({"", "{}", "[]"}).any():
        raise ValueError("available rows require non-empty immutable source hashes")
    for value in hashes:
        try:
            json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("source_hashes_json is not valid JSON") from error
    if not result.loc[available, "first_lead_h"].gt(0).all():
        raise ValueError("available rows require positive first lead")
    if not result.loc[available, "last_lead_h"].ge(168).all():
        raise ValueError("available rows do not cover decision-relative day 7")

    local_date = (
        result["issue_time_utc"].dt.tz_convert("Europe/Riga").dt.tz_localize(None).dt.normalize()
    )
    if not result["issue_date"].eq(local_date).all():
        raise ValueError("issue_date disagrees with issue_time_utc in Europe/Riga")
    for column in features:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    values = result.loc[available, features].to_numpy(dtype=float)
    if values.size and not np.isfinite(values).all():
        raise ValueError("available forecasts require finite values for every declared feature")
    keys = ["issue_date", "gfs_cell_id", "availability_scenario_hours"]
    if result.duplicated(keys).any():
        raise ValueError("duplicate issue_date/gfs_cell_id/availability_scenario_hours row")
    return result.sort_values(keys, kind="mergesort").reset_index(drop=True)


def load_forecast_feature_table(
    path: str | Path,
    *,
    feature_columns: Sequence[str] = GFS_FORECAST_FEATURES,
) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source)
    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        raise ValueError("forecast feature table must be Parquet or CSV")
    return validate_forecast_feature_table(frame, feature_columns=feature_columns)


def resolve_forecast_input_bundle_paths(
    forecast_features_path: str | Path,
    *,
    source_manifest_path: str | Path | None = None,
    request_plan_path: str | Path | None = None,
    request_plan_manifest_path: str | Path | None = None,
) -> dict[str, Path]:
    """Resolve the four immutable bundle members without searching other runs."""

    forecast = _resolve(forecast_features_path).resolve()

    def explicit_or_sibling(value: str | Path | None, sibling_name: str) -> Path:
        return (
            _resolve(value).resolve()
            if value is not None
            else (forecast.parent / sibling_name).resolve()
        )

    return {
        "forecast_features": forecast,
        "source_manifest": explicit_or_sibling(
            source_manifest_path, SOURCE_MANIFEST_FILE_NAME
        ),
        "request_plan": explicit_or_sibling(request_plan_path, PLAN_FILE_NAME),
        "request_plan_manifest": explicit_or_sibling(
            request_plan_manifest_path, PLAN_MANIFEST_FILE_NAME
        ),
    }


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def validate_forecast_input_bundle(
    forecast: pd.DataFrame,
    *,
    bundle_paths: Mapping[str, Path],
    contract_path: Path,
    parent_decisions_path: Path,
    expected_scenarios: Sequence[int],
    feature_columns: Sequence[str] = GFS_FORECAST_FEATURES,
) -> dict[str, Any]:
    """Prove that every frozen plan checkpoint was assembled before fitting.

    A fetched checkpoint may still document an unavailable archive snapshot;
    that is genuine source computability.  A checkpoint that was never fetched
    is an unfinished extraction and cannot enter a quality run.
    """

    required_bundle_names = {
        "forecast_features",
        "source_manifest",
        "request_plan",
        "request_plan_manifest",
    }
    missing_names = sorted(required_bundle_names.difference(bundle_paths))
    if missing_names:
        raise ValueError(f"forecast input bundle paths are missing: {missing_names}")
    paths = {name: Path(bundle_paths[name]).resolve() for name in required_bundle_names}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing forecast input bundle member {name}: {path}")

    source_manifest = _read_json_object(paths["source_manifest"], "GFS source manifest")
    plan_manifest = _read_json_object(
        paths["request_plan_manifest"], "GFS request-plan manifest"
    )
    plan = pd.read_parquet(paths["request_plan"])
    features = validate_forecast_feature_table(
        forecast, feature_columns=feature_columns
    )

    if source_manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("GFS source manifest schema is not the frozen builder schema")
    if plan_manifest.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("GFS request-plan manifest schema is not the frozen builder schema")
    required_plan_columns = {
        "plan_schema_version",
        "checkpoint_id",
        "issue_date",
        "issue_time_utc",
        "availability_scenario_hours",
        "selection_rule_id",
        "selected_init_utc",
        "assumed_available_at_utc",
        "publication_time_utc",
        "required_gfs_cell_ids_json",
        "required_gfs_cell_count",
        "requested_snapshot_count",
        "expected_steps_1_3d",
        "expected_steps_4_7d",
        "source_dataset_id",
    }
    missing_plan = sorted(required_plan_columns.difference(plan.columns))
    if missing_plan:
        raise ValueError(f"frozen GFS request plan misses columns: {missing_plan}")
    if plan.empty or plan["checkpoint_id"].duplicated().any():
        raise ValueError("frozen GFS request plan is empty or has duplicate checkpoints")
    if not plan["plan_schema_version"].astype(str).eq(PLAN_SCHEMA_VERSION).all():
        raise ValueError("request-plan rows use a non-frozen schema version")
    plan_scenarios = sorted(
        pd.to_numeric(plan["availability_scenario_hours"], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )
    expected_scenario_list = sorted({int(value) for value in expected_scenarios})
    if plan_scenarios != expected_scenario_list:
        raise ValueError(
            f"request plan scenarios differ from contract: {plan_scenarios} != "
            f"{expected_scenario_list}"
        )
    if not plan["source_dataset_id"].astype(str).eq("d084001").all():
        raise ValueError("request plan contains a source other than GDEX d084001")
    if not pd.to_numeric(plan["expected_steps_1_3d"], errors="raise").eq(12).all():
        raise ValueError("request plan does not require all 12 d1-3 samples")
    if not pd.to_numeric(plan["expected_steps_4_7d"], errors="raise").eq(16).all():
        raise ValueError("request plan does not require all 16 d4-7 samples")
    if pd.to_datetime(plan["publication_time_utc"], errors="coerce", utc=True).notna().any():
        raise ValueError("request plan claims an unproven historical publication time")

    feature_sha = sha256_file(paths["forecast_features"])
    plan_sha = sha256_file(paths["request_plan"])
    contract_sha = sha256_file(contract_path)
    parent_sha = sha256_file(parent_decisions_path)
    for actual, expected, label in (
        (source_manifest.get("feature_table_sha256"), feature_sha, "source feature-table hash"),
        (source_manifest.get("request_plan_sha256"), plan_sha, "source request-plan hash"),
        (plan_manifest.get("request_plan_sha256"), plan_sha, "plan-manifest plan hash"),
        (plan_manifest.get("contract_sha256"), contract_sha, "plan-manifest contract hash"),
        (
            plan_manifest.get("parent_daily_decisions_sha256"),
            parent_sha,
            "plan-manifest parent decisions hash",
        ),
    ):
        if actual != expected:
            raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")

    planned_count = int(len(plan))
    assembled_count = int(features["checkpoint_id"].nunique()) if "checkpoint_id" in features else -1
    declared_planned = int(source_manifest.get("planned_checkpoints", -1))
    declared_assembled = int(source_manifest.get("assembled_checkpoints", -1))
    missing_checkpoint_count = int(source_manifest.get("missing_checkpoint_count", -1))
    if not (
        missing_checkpoint_count == 0
        and source_manifest.get("missing_checkpoint_ids") == []
        and planned_count == assembled_count == declared_planned == declared_assembled
    ):
        raise ValueError(
            "GFS extraction is incomplete: quality run requires zero missing checkpoints "
            f"and planned=assembled; planned={planned_count}, assembled={assembled_count}, "
            f"declared={declared_planned}/{declared_assembled}, "
            f"missing={missing_checkpoint_count}"
        )
    if int(plan_manifest.get("checkpoints", -1)) != planned_count:
        raise ValueError("request-plan manifest checkpoint count disagrees with the plan")
    if sorted(plan_manifest.get("scenario_hours", [])) != expected_scenario_list:
        raise ValueError("request-plan manifest scenarios disagree with the contract")
    if int(plan_manifest.get("archive_subset_requests", -1)) != int(
        pd.to_numeric(plan["requested_snapshot_count"], errors="raise").sum()
    ):
        raise ValueError("request-plan manifest archive request count disagrees with the plan")
    for manifest_name, privacy_manifest in (
        ("request-plan", plan_manifest),
        ("source", source_manifest),
    ):
        if privacy_manifest.get("contains_field_ids") is not False:
            raise ValueError(f"{manifest_name} manifest must confirm no field IDs")
        if privacy_manifest.get("contains_outcomes") is not False:
            raise ValueError(f"{manifest_name} manifest must confirm no outcomes")
        if privacy_manifest.get("contains_coarse_grid_coordinates") is not True:
            raise ValueError(
                f"{manifest_name} manifest must label its coarse grid coordinates"
            )
        if privacy_manifest.get("distribution") != "local_private_do_not_publish":
            raise ValueError(f"{manifest_name} manifest must remain local/private")

    compatibility_schema = plan_manifest.get("compatibility_schema_version")
    if compatibility_schema is not None:
        if compatibility_schema != "gfs_request_plan_manifest_compatibility_v1":
            raise ValueError("unknown request-plan compatibility manifest schema")
        legacy_path = Path(str(plan_manifest.get("legacy_manifest_path", "")))
        if not legacy_path.is_absolute():
            legacy_path = paths["request_plan_manifest"].parent / legacy_path
        if not legacy_path.is_file():
            raise FileNotFoundError(
                f"missing frozen legacy request-plan manifest: {legacy_path}"
            )
        legacy_sha = sha256_file(legacy_path)
        if plan_manifest.get("legacy_manifest_sha256") != legacy_sha:
            raise ValueError("request-plan compatibility legacy hash mismatch")
        legacy_manifest = _read_json_object(
            legacy_path, "legacy GFS request-plan manifest"
        )
        for key in (
            "schema_version",
            "request_plan_sha256",
            "contract_sha256",
            "parent_daily_decisions_sha256",
            "checkpoints",
            "archive_subset_requests",
            "scenario_hours",
        ):
            if plan_manifest.get(key) != legacy_manifest.get(key):
                raise ValueError(
                    f"request-plan compatibility manifest changed frozen {key}"
                )

    required_checkpoint_columns = {
        "checkpoint_schema_version",
        "checkpoint_id",
        "checkpoint_complete",
        "source_access_mode",
        "requested_snapshot_count",
        "retrieved_snapshot_count",
        "failed_snapshot_count",
        "failed_requests_json",
    }
    missing_checkpoint_columns = sorted(required_checkpoint_columns.difference(features.columns))
    if missing_checkpoint_columns:
        raise ValueError(
            "assembled GFS features miss checkpoint provenance columns: "
            f"{missing_checkpoint_columns}"
        )
    if features[list(required_checkpoint_columns)].isna().any().any():
        raise ValueError("assembled GFS checkpoint provenance contains null values")
    if not features["checkpoint_schema_version"].astype(str).eq(
        CHECKPOINT_SCHEMA_VERSION
    ).all():
        raise ValueError("assembled GFS checkpoint rows use a non-frozen schema")
    access_modes = set(features["source_access_mode"].astype(str).str.strip())
    if not access_modes or "" in access_modes or not access_modes.issubset(
        _KNOWN_SOURCE_ACCESS_MODES
    ):
        raise ValueError(f"unknown or missing GFS source_access_mode: {sorted(access_modes)}")
    if sorted(source_manifest.get("source_access_modes", [])) != sorted(access_modes):
        raise ValueError("source manifest access modes disagree with assembled rows")

    expected_rows: list[dict[str, Any]] = []
    for row in plan.to_dict(orient="records"):
        try:
            cells = json.loads(str(row["required_gfs_cell_ids_json"]))
        except json.JSONDecodeError as error:
            raise ValueError("request plan contains invalid required-cell JSON") from error
        if not isinstance(cells, list) or len(cells) != int(row["required_gfs_cell_count"]):
            raise ValueError("request-plan required cell count is inconsistent")
        if len(cells) != len(set(cells)) or not cells:
            raise ValueError("request-plan required GFS cells are empty or duplicated")
        for cell in cells:
            expected_rows.append(
                {
                    "checkpoint_id": str(row["checkpoint_id"]),
                    "gfs_cell_id": str(cell),
                    "issue_date_expected": pd.Timestamp(row["issue_date"]).normalize(),
                    "scenario_expected": int(row["availability_scenario_hours"]),
                    "issue_time_expected": pd.Timestamp(row["issue_time_utc"]),
                    "init_expected": pd.Timestamp(row["selected_init_utc"]),
                    "available_expected": pd.Timestamp(row["assumed_available_at_utc"]),
                    "selection_expected": str(row["selection_rule_id"]),
                    "requested_expected": int(row["requested_snapshot_count"]),
                }
            )
    expected = pd.DataFrame(expected_rows)
    actual_keys = features[["checkpoint_id", "gfs_cell_id"]].astype(str)
    expected_keys = expected[["checkpoint_id", "gfs_cell_id"]].astype(str)
    actual_key_set = set(map(tuple, actual_keys.itertuples(index=False, name=None)))
    expected_key_set = set(map(tuple, expected_keys.itertuples(index=False, name=None)))
    if actual_key_set != expected_key_set or len(features) != len(expected):
        missing_keys = sorted(expected_key_set - actual_key_set)[:5]
        extra_keys = sorted(actual_key_set - expected_key_set)[:5]
        raise ValueError(
            "assembled GFS rows do not cover every planned checkpoint/cell exactly once: "
            f"missing={missing_keys}, extra={extra_keys}"
        )
    merged = features.merge(
        expected,
        on=["checkpoint_id", "gfs_cell_id"],
        how="left",
        validate="one_to_one",
    )
    metadata_checks = {
        "issue_date": pd.to_datetime(merged["issue_date"]).dt.normalize().eq(
            merged["issue_date_expected"]
        ),
        "availability_scenario_hours": pd.to_numeric(
            merged["availability_scenario_hours"], errors="raise"
        ).astype(int).eq(merged["scenario_expected"]),
        "issue_time_utc": pd.to_datetime(merged["issue_time_utc"], utc=True).eq(
            pd.to_datetime(merged["issue_time_expected"], utc=True)
        ),
        "selected_init_utc": pd.to_datetime(merged["selected_init_utc"], utc=True).eq(
            pd.to_datetime(merged["init_expected"], utc=True)
        ),
        "assumed_available_at_utc": pd.to_datetime(
            merged["assumed_available_at_utc"], utc=True
        ).eq(pd.to_datetime(merged["available_expected"], utc=True)),
        "selection_rule_id": merged["selection_rule_id"].astype(str).eq(
            merged["selection_expected"]
        ),
        "requested_snapshot_count": pd.to_numeric(
            merged["requested_snapshot_count"], errors="raise"
        ).astype(int).eq(merged["requested_expected"]),
    }
    mismatched_metadata = [name for name, valid in metadata_checks.items() if not valid.all()]
    if mismatched_metadata:
        raise ValueError(
            f"assembled GFS rows disagree with frozen request plan: {mismatched_metadata}"
        )

    requested = pd.to_numeric(features["requested_snapshot_count"], errors="raise").astype(int)
    retrieved = pd.to_numeric(features["retrieved_snapshot_count"], errors="raise").astype(int)
    failed = pd.to_numeric(features["failed_snapshot_count"], errors="raise").astype(int)
    if (retrieved < 0).any() or (failed < 0).any() or not (retrieved + failed).eq(requested).all():
        raise ValueError("GFS checkpoint retrieved/failed counts do not sum to requested")
    parsed_failures: list[list[Any]] = []
    for value in features["failed_requests_json"].astype(str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("failed_requests_json is invalid") from error
        if not isinstance(parsed, list):
            raise ValueError("failed_requests_json must encode a list")
        parsed_failures.append(parsed)
    if any(len(items) != count for items, count in zip(parsed_failures, failed, strict=True)):
        raise ValueError("failed_requests_json count disagrees with failed_snapshot_count")
    available = _bool(features["forecast_available"], name="forecast_available")
    checkpoint_complete = _bool(features["checkpoint_complete"], name="checkpoint_complete")
    if not checkpoint_complete.eq(available & failed.eq(0)).all():
        raise ValueError("checkpoint_complete disagrees with availability and fetch failures")

    unavailable_ids = sorted(
        features.loc[~available, "checkpoint_id"].astype(str).unique().tolist()
    )
    unavailability_link = source_manifest.get("source_unavailability_manifest")
    if unavailable_ids:
        if not isinstance(unavailability_link, dict):
            raise ValueError(
                "unavailable GFS rows require a versioned source-unavailability manifest"
            )
        unavailable_path = Path(str(unavailability_link.get("path", "")))
        if not unavailable_path.is_absolute():
            unavailable_path = paths["source_manifest"].parent / unavailable_path
        if not unavailable_path.is_file():
            raise FileNotFoundError(
                f"missing GFS source-unavailability manifest: {unavailable_path}"
            )
        unavailable_sha = sha256_file(unavailable_path)
        if unavailability_link.get("sha256") != unavailable_sha:
            raise ValueError("GFS source-unavailability manifest hash mismatch")
        unavailable_manifest = _read_json_object(
            unavailable_path, "GFS source-unavailability manifest"
        )
        if (
            unavailable_manifest.get("schema_version")
            != "gfs_source_unavailability_manifest_v1"
            or unavailability_link.get("schema_version")
            != "gfs_source_unavailability_manifest_v1"
        ):
            raise ValueError("unknown GFS source-unavailability manifest schema")
        if unavailable_manifest.get("request_plan_sha256") != plan_sha:
            raise ValueError("GFS source-unavailability plan hash mismatch")
        declared_unavailable_ids = sorted(
            str(value) for value in unavailable_manifest.get("checkpoint_ids", [])
        )
        if declared_unavailable_ids != unavailable_ids:
            raise ValueError(
                "GFS source-unavailability checkpoints disagree with unavailable rows"
            )
        if int(unavailability_link.get("checkpoint_count", -1)) != len(
            unavailable_ids
        ) or int(unavailable_manifest.get("checkpoint_count", -1)) != len(
            unavailable_ids
        ):
            raise ValueError("GFS source-unavailability checkpoint count mismatch")
        frozen_method_flags = {
            "no_data_substitution": True,
            "terminal_scope": "frozen_gdex_thredds_ncss_access_method_only",
            "archive_wide_absence_claimed": False,
            "alternate_transport_used": False,
            "alternate_cycle_used": False,
            "reanalysis_used": False,
            "imputation_used": False,
        }
        if any(
            unavailable_manifest.get(key) != value
            for key, value in frozen_method_flags.items()
        ):
            raise ValueError("GFS source-unavailability substitution policy changed")
        declared_snapshots = unavailable_manifest.get("unavailable_snapshots")
        if not isinstance(declared_snapshots, list):
            raise ValueError("GFS source-unavailability records must be a list")
        expected_failures: list[dict[str, Any]] = []
        for checkpoint_id, group in features.loc[~available].groupby(
            "checkpoint_id", sort=True
        ):
            encoded = group["failed_requests_json"].astype(str).unique().tolist()
            if len(encoded) != 1:
                raise ValueError(
                    "failed_requests_json differs within unavailable checkpoint"
                )
            failures_for_checkpoint = json.loads(encoded[0])
            for failure in failures_for_checkpoint:
                expected_failures.append(
                    {
                        "checkpoint_id": str(checkpoint_id),
                        "lead_hours": int(failure["lead_hours"]),
                        "archive_path": str(failure["archive_path"]),
                        "error_type": str(failure["error_type"]),
                        "error": str(failure["error"]),
                    }
                )
        declared_failures = [
            {
                "checkpoint_id": str(item.get("checkpoint_id")),
                "lead_hours": int(item.get("lead_hours", -1)),
                "archive_path": str(item.get("archive_path")),
                "error_type": str(item.get("error_type")),
                "error": str(item.get("error")),
            }
            for item in declared_snapshots
            if isinstance(item, dict)
        ]
        sort_key = lambda item: (item["checkpoint_id"], item["lead_hours"])
        if sorted(declared_failures, key=sort_key) != sorted(
            expected_failures, key=sort_key
        ):
            raise ValueError(
                "GFS source-unavailability evidence disagrees with checkpoint failures"
            )
        snapshot_count = len(expected_failures)
        if (
            len(declared_failures) != len(declared_snapshots)
            or int(unavailability_link.get("unavailable_snapshot_count", -1))
            != snapshot_count
            or int(unavailable_manifest.get("unavailable_snapshot_count", -1))
            != snapshot_count
        ):
            raise ValueError("GFS source-unavailability snapshot count mismatch")
    elif unavailability_link not in (None, {}):
        raise ValueError(
            "source-unavailability manifest is present but all GFS rows are available"
        )

    declared_rows = int(source_manifest.get("feature_rows", -1))
    declared_complete = int(source_manifest.get("complete_feature_rows", -1))
    if declared_rows != len(features) or declared_complete != int(available.sum()):
        raise ValueError("source manifest feature-row counts disagree with the table")
    expected_fraction = float(available.mean())
    if not np.isclose(
        float(source_manifest.get("complete_feature_fraction", np.nan)),
        expected_fraction,
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("source manifest complete-feature fraction disagrees with the table")
    if source_manifest.get("forecast_features") != list(feature_columns):
        raise ValueError("source manifest forecast feature whitelist changed")
    if source_manifest.get("expected_steps") != {"d1_3": 12, "d4_7": 16}:
        raise ValueError("source manifest expected-step contract changed")
    raw_hashes: set[str] = set()
    for value in features["source_hashes_json"].dropna().astype(str):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("source_hashes_json must encode a list")
        for digest in parsed:
            digest = str(digest)
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("source_hashes_json contains a non-SHA256 value")
            raw_hashes.add(digest)
    if int(source_manifest.get("raw_subset_hash_count", -1)) != len(raw_hashes):
        raise ValueError("source manifest raw-subset hash count disagrees with features")
    if sorted(source_manifest.get("raw_subset_sha256", [])) != sorted(raw_hashes):
        raise ValueError("source manifest raw-subset hashes disagree with features")

    return {
        "status": "complete_and_verified",
        "privacy": "local_private_do_not_publish",
        "contains_coarse_location_footprint": True,
        "planned_checkpoints": planned_count,
        "assembled_checkpoints": assembled_count,
        "planned_feature_rows": int(len(expected)),
        "assembled_feature_rows": int(len(features)),
        "complete_feature_rows": int(available.sum()),
        "unavailable_feature_rows": int((~available).sum()),
        "source_access_modes": sorted(access_modes),
        "hashes": {name: sha256_file(path) for name, path in paths.items()},
        "paths": {name: str(path) for name, path in paths.items()},
    }


def gfs_cell_id_from_weather_cell(values: pd.Series) -> pd.Series:
    """Convert the parent's private rounded cell to the public GFS cell key."""

    parsed = values.astype("string").str.extract(r"^([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)$")
    latitude = pd.to_numeric(parsed[0], errors="coerce")
    longitude = pd.to_numeric(parsed[1], errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="string")
    valid = latitude.notna() & longitude.notna()
    result.loc[valid] = [
        canonical_gfs_cell_id(float(lat), float(lon))
        for lat, lon in zip(latitude.loc[valid], longitude.loc[valid], strict=True)
    ]
    return result


def attach_forecast_features(
    decisions: pd.DataFrame,
    forecast_features: pd.DataFrame,
    *,
    scenario_hours: int,
    feature_columns: Sequence[str] = GFS_FORECAST_FEATURES,
    past_feature_columns: Sequence[str] = NASA_COMMON_FEATURES,
    decision_cell_column: str = "weather_cell",
) -> pd.DataFrame:
    """Join one lag scenario and create the exact A/B/C complete-case mask."""

    features = assert_safe_feature_names(feature_columns)
    past = assert_safe_feature_names(past_feature_columns)
    required = {
        "field_season",
        "season",
        "issue_date",
        "issued_at",
        "service_active",
        "target_observable",
        "target_class",
        "nasa_common_complete",
        decision_cell_column,
        *CALENDAR_FEATURES,
        *past,
    }
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"daily decisions miss required columns: {missing}")
    validated = validate_forecast_feature_table(forecast_features, feature_columns=features)
    selected = validated[
        validated["availability_scenario_hours"].eq(int(scenario_hours))
    ].copy()
    if selected.empty:
        raise ValueError(f"forecast table has no total-lag scenario {scenario_hours}h")

    left = decisions.copy()
    left["issue_date"] = pd.to_datetime(left["issue_date"], errors="raise").dt.normalize()
    if "gfs_cell_id" not in left:
        left["gfs_cell_id"] = gfs_cell_id_from_weather_cell(left[decision_cell_column])
    if left["gfs_cell_id"].isna().any():
        raise ValueError("one or more private decision rows cannot be mapped to a GFS cell")
    left["_decision_row_order"] = np.arange(len(left), dtype=np.int64)
    left["_original_index"] = left.index
    join_columns = [column for column in selected.columns if column != "issue_date"]
    result = left.merge(
        selected[["issue_date", *join_columns]],
        on=["issue_date", "gfs_cell_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_gfs"),
    ).sort_values("_decision_row_order", kind="mergesort")
    result.index = result.pop("_original_index")
    result = result.drop(columns="_decision_row_order")
    result.index.name = decisions.index.name

    joined = result["issue_time_utc"].notna()
    available = _bool(result["forecast_available"], name="forecast_available_after_join")
    finite = result[features].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    result["gfs_feature_row_present"] = joined
    result["gfs_forecast_complete"] = joined & available & finite
    result["gfs_missing_reason"] = np.select(
        [~joined, joined & ~available, joined & available & ~finite],
        ["no_forecast_feature_row", "incomplete_forecast_horizon", "nonfinite_forecast_feature"],
        default="available",
    )
    result["abc_complete"] = (
        result["service_active"].astype(bool)
        & result["nasa_common_complete"].astype(bool)
        & result["gfs_forecast_complete"].astype(bool)
    )
    result["availability_scenario_hours"] = int(scenario_hours)
    timing_rows = result["gfs_forecast_complete"]
    issued_at = pd.to_datetime(result["issued_at"], errors="raise", utc=True)
    if not result.loc[timing_rows, "issue_time_utc"].eq(issued_at.loc[timing_rows]).all():
        raise ValueError("joined GFS issue_time_utc differs from the saved daily decision time")
    return result


def model_training_rows(
    frame: pd.DataFrame,
    years: Sequence[int],
    *,
    common_mask_column: str = "abc_complete",
) -> pd.DataFrame:
    """Exact common observable fit rows shared by A_matched_C0, B, and C."""

    mask = (
        _year_mask(frame, years)
        & frame["target_observable"].astype(bool)
        & frame["service_active"].astype(bool)
        & frame[common_mask_column].astype(bool)
        & frame["target_class"].isin(TARGET_TO_INT)
    )
    return frame.loc[mask].copy()


def full_calendar_training_rows(frame: pd.DataFrame, end_year: int) -> pd.DataFrame:
    """All permitted calendar-only rows from 2010 through fold train end."""

    mask = (
        frame["season"].between(2010, int(end_year))
        & frame["target_observable"].astype(bool)
        & frame["service_active"].astype(bool)
        & frame["target_class"].isin(TARGET_TO_INT)
        & frame[CALENDAR_FEATURES].notna().all(axis=1)
    )
    return frame.loc[mask].copy()


def calendar_window_bounds(
    seasons: pd.DataFrame,
    decisions: pd.DataFrame,
    train_years: Sequence[int],
    *,
    matched_to_common_mask: bool = True,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> tuple[int, int, int]:
    """Estimate empirical bounds from training first events only."""

    registry = seasons[
        _year_mask(seasons, train_years)
        & seasons["warnable_first_event"].astype(bool)
        & seasons["first_recorded_event_date"].notna()
    ].copy()
    if matched_to_common_mask:
        eligible = decisions.loc[
            _year_mask(decisions, train_years) & decisions["abc_complete"].astype(bool),
            "field_season",
        ].unique()
        registry = registry[registry["field_season"].isin(eligible)]
    if registry.empty:
        raise ValueError("calendar window has no eligible training first events")
    event_doy = pd.to_datetime(registry["first_recorded_event_date"]).dt.dayofyear
    lower = int(event_doy.quantile(lower_quantile))
    upper = int(event_doy.quantile(upper_quantile))
    if not 1 <= lower <= upper <= 366:
        raise AssertionError("invalid learned calendar bounds")
    return lower, upper, int(len(registry))


def calendar_window_score(frame: pd.DataFrame, lower_doy: int, upper_doy: int) -> pd.Series:
    day = pd.to_datetime(frame["issue_date"]).dt.dayofyear
    return day.between(int(lower_doy), int(upper_doy)).astype(float)


def _fixed_catboost_params(contract: Mapping[str, Any]) -> dict[str, Any]:
    fixed = contract["catboost_fixed"]
    result = {
        "iterations": int(fixed["iterations"]),
        "depth": int(fixed["depth"]),
        "learning_rate": float(fixed["learning_rate"]),
        "l2_leaf_reg": float(fixed["l2_leaf_reg"]),
    }
    if result["depth"] > 5 or result["iterations"] > 500:
        raise ValueError("fixed first-pass CatBoost must remain shallow and bounded")
    return result


def _actionable_probability(model: Any, values: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(values), dtype=float)
    classes = [int(value) for value in model.classes_]
    actionable = TARGET_TO_INT["actionable"]
    if actionable not in classes:
        return np.zeros(len(values), dtype=float)
    return probabilities[:, classes.index(actionable)]


def _save_model_roundtrip(
    model: Any,
    *,
    kind: str,
    path: Path,
    sample: pd.DataFrame,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "catboost":
        from catboost import CatBoostClassifier

        model.save_model(path)
        loaded = CatBoostClassifier()
        loaded.load_model(path)
    elif kind == "logistic":
        joblib.dump(model, path)
        loaded = joblib.load(path)
    else:  # pragma: no cover
        raise ValueError(kind)
    before = _actionable_probability(model, sample) if len(sample) else np.array([])
    after = _actionable_probability(loaded, sample) if len(sample) else np.array([])
    difference = float(np.max(np.abs(before - after))) if len(sample) else np.nan
    if len(sample) and difference > 1e-12:
        raise AssertionError(f"model reload changed actionable scores by {difference}")
    return {
        "artifact": str(path),
        "artifact_sha256": sha256_file(path),
        "sample_rows": int(len(sample)),
        "max_abs_actionable_score_difference": difference,
        "status": "prediction_roundtrip_checked" if len(sample) else "serialized_only",
    }


def _policy_settings(contract: Mapping[str, Any]) -> dict[str, float | int]:
    policy = contract["notification_policy"]
    budget = policy["research_budget"]
    return {
        "active_days": int(policy["active_days_per_message"]),
        "cooldown_days": int(policy["cooldown_days"]),
        "max_messages": float(budget["messages_per_30_field_days_max"]),
        "max_alarm": float(budget["active_alarm_fraction_max"]),
    }


def _validation_policy_record(
    frame: pd.DataFrame,
    seasons: pd.DataFrame,
    score: pd.Series,
    policy: Policy,
    model_code: str,
    fold_id: str,
    scenario_hours: int,
    *,
    selection_kind: str,
    feasible: bool,
) -> dict[str, Any]:
    mask = frame["abc_complete"].astype(bool)
    states = simulate_policy(frame, score.where(mask), policy, PRIMARY_SCOPE, mask)
    events, _ = event_metrics(
        states, seasons, model_code, fold_id, evaluation_scope=PRIMARY_SCOPE
    )
    burden = burden_metrics(states, model_code, fold_id, evaluation_scope=PRIMARY_SCOPE)
    return {
        "availability_scenario_hours": int(scenario_hours),
        "model_code": model_code,
        "fold_id": fold_id,
        "selection_scope": PRIMARY_SCOPE,
        "selection_kind": selection_kind,
        "threshold": float(policy.threshold),
        "active_days": int(policy.active_days),
        "cooldown_days": int(policy.cooldown_days),
        "validation_policy_feasible": bool(feasible),
        "validation_events": events["events_with_warning_opportunity"],
        "validation_timely_hits": events["timely_hits"],
        "validation_timely_recall": events["timely_recall"],
        "validation_field_days": burden["field_days"],
        "validation_messages": burden["messages"],
        "validation_messages_per_30": burden["messages_per_30_field_days"],
        "validation_alarm_fraction": burden["active_alarm_fraction"],
        "validation_computable_fraction": burden["computable_fraction"],
    }


def select_validation_policy(
    validation: pd.DataFrame,
    validation_seasons: pd.DataFrame,
    score: pd.Series,
    *,
    model_code: str,
    fold_id: str,
    scenario_hours: int,
    contract: Mapping[str, Any],
    fixed_binary_calendar: bool = False,
) -> tuple[Policy, dict[str, Any]]:
    """Select from validation only; empirical A retains its binary threshold."""

    settings = _policy_settings(contract)
    if fixed_binary_calendar:
        policy = Policy(
            0.5,
            int(settings["active_days"]),
            int(settings["cooldown_days"]),
            "gfs_abc_P0_fixed_binary_calendar",
        )
        record = _validation_policy_record(
            validation,
            validation_seasons,
            score,
            policy,
            model_code,
            fold_id,
            scenario_hours,
            selection_kind="fixed_binary_score_with_training_only_calendar_bounds",
            feasible=True,
        )
        record["validation_policy_feasible"] = bool(
            record["validation_messages_per_30"] <= settings["max_messages"]
            and record["validation_alarm_fraction"] <= settings["max_alarm"]
        )
        return policy, record

    mask = validation["abc_complete"].astype(bool)
    policy, details = select_threshold(
        validation,
        score,
        validation_seasons,
        int(settings["active_days"]),
        int(settings["cooldown_days"]),
        float(settings["max_messages"]),
        float(settings["max_alarm"]),
        PRIMARY_SCOPE,
        mask,
    )
    record = _validation_policy_record(
        validation,
        validation_seasons,
        score,
        policy,
        model_code,
        fold_id,
        scenario_hours,
        selection_kind="validation_threshold_grid_under_existing_P0",
        feasible=bool(details["any_feasible"]),
    )
    if not np.isclose(
        record["validation_timely_recall"],
        details["selected"]["timely_recall"],
        equal_nan=True,
    ):
        raise AssertionError("saved validation policy does not reproduce threshold selection")
    return policy, record


def _evaluate_model(
    *,
    frame: pd.DataFrame,
    seasons: pd.DataFrame,
    raw_score: pd.Series,
    policy: Policy,
    model_code: str,
    fold_id: str,
    scenario_hours: int,
    feature_provenance: str,
    evaluation_scope: str,
) -> dict[str, Any]:
    if evaluation_scope == PRIMARY_SCOPE:
        evaluation_mask: pd.Series | None = frame["abc_complete"].astype(bool)
        score = raw_score.where(evaluation_mask)
    elif evaluation_scope == SERVICE_SCOPE:
        evaluation_mask = None
        score = raw_score
    else:
        raise ValueError(f"unsupported evaluation scope {evaluation_scope!r}")
    # Always pass the original daily calendar. Missing days abstain but are not
    # removed, so a 15-day cooldown remains 15 calendar days.
    states = simulate_policy(frame, score, policy, evaluation_scope, evaluation_mask)
    states["model_code"] = model_code
    states["fold_id"] = fold_id
    states["model_version"] = "gfs_abc_fixed_first_pass_v1"
    states["feature_provenance"] = feature_provenance
    states["availability_scenario_hours"] = int(scenario_hours)
    for column in (
        "gfs_cell_id",
        "gfs_feature_row_present",
        "gfs_forecast_complete",
        "gfs_missing_reason",
        "abc_complete",
        "issue_time_utc",
        "assumed_available_at_utc",
        "selected_init_utc",
        "selection_rule_id",
    ):
        if column in frame:
            states[column] = frame[column].reindex(states.index)

    event_rows: list[dict[str, Any]] = []
    event_hits: list[dict[str, Any]] = []
    burden_rows: list[dict[str, Any]] = []
    for slice_name in ("A_plus_B", "direct_A"):
        metric, hits = event_metrics(
            states, seasons, model_code, fold_id, slice_name, evaluation_scope
        )
        metric["availability_scenario_hours"] = int(scenario_hours)
        event_rows.append(metric)
        for hit in hits:
            hit["availability_scenario_hours"] = int(scenario_hours)
        event_hits.extend(hits)
        burden = burden_metrics(states, model_code, fold_id, slice_name, evaluation_scope)
        burden["availability_scenario_hours"] = int(scenario_hours)
        burden_rows.append(burden)
    return {
        "states": states,
        "event_metrics": event_rows,
        "event_hits": event_hits,
        "burden_metrics": burden_rows,
    }


def _fit_fold_models(
    *,
    decisions: pd.DataFrame,
    seasons: pd.DataFrame,
    fold: Mapping[str, Any],
    contract: Mapping[str, Any],
    past_features: Sequence[str],
    forecast_features: Sequence[str],
    scenario_hours: int,
    model_dir: Path,
    fold_index: int,
) -> tuple[
    dict[str, dict[str, pd.Series]],
    dict[str, Policy],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    fold_id = str(fold["id"])
    train = model_training_rows(decisions, fold["train_years"])
    train_end = int(fold["train_years"][1])
    train_strong = full_calendar_training_rows(decisions, train_end)
    validation = decisions[_year_mask(decisions, fold["validation_years"])].copy()
    test = decisions[_year_mask(decisions, fold["test_years"])].copy()
    validation_seasons = seasons[_year_mask(seasons, fold["validation_years"])].copy()
    if train.empty or train_strong.empty or validation.empty or test.empty:
        raise ValueError(f"fold {fold_id} has an empty train, validation, or test block")
    for name, sample in (("common", train), ("strong calendar", train_strong)):
        classes = set(sample["target_class"].map(TARGET_TO_INT).dropna().unique())
        if classes != set(TARGET_TO_INT.values()):
            raise ValueError(f"fold {fold_id} {name} fit rows lack one or more target classes")

    past = assert_safe_feature_names(past_features)
    forecast = assert_safe_feature_names(forecast_features)
    b_features = list(CALENDAR_FEATURES) + past
    c_features = b_features + forecast
    for name, required in {"B": b_features, "C": c_features}.items():
        missing = sorted(set(required).difference(decisions.columns))
        if missing:
            raise ValueError(f"{name} feature columns are missing: {missing}")
    if not train[c_features].notna().all(axis=1).all():
        raise AssertionError("common A/B/C training set contains an incomplete model row")

    catboost_params = _fixed_catboost_params(contract)
    base_seed = int(contract.get("random_seed", 20260910)) + fold_index * 100
    # B and C form a paired feature ablation.  Their CatBoost stochastic state
    # must be identical, otherwise C-B also measures a seed change.
    paired_catboost_seed = int(
        contract.get("catboost_fixed", {}).get(
            "random_seed", contract.get("random_seed", 20260910)
        )
    )
    specs: dict[str, dict[str, Any]] = {
        MODEL_A_MATCHED_C0: {
            "kind": "logistic",
            "features": list(CALENDAR_FEATURES),
            "train": train,
            "params": {"C": 0.1},
            "availability": None,
            "seed": base_seed + 1,
        },
        MODEL_A_STRONG: {
            "kind": "catboost",
            "features": list(CALENDAR_FEATURES),
            "train": train_strong,
            "params": catboost_params,
            "availability": None,
            "seed": paired_catboost_seed,
        },
        MODEL_B: {
            "kind": "catboost",
            "features": b_features,
            "train": train,
            "params": catboost_params,
            "availability": "nasa_common_complete",
            "seed": paired_catboost_seed,
        },
        MODEL_C: {
            "kind": "catboost",
            "features": c_features,
            "train": train,
            "params": catboost_params,
            "availability": "abc_complete",
            "seed": paired_catboost_seed,
        },
    }
    score_sets: dict[str, dict[str, pd.Series]] = {"validation": {}, "test": {}}
    reload_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for model_code, spec in specs.items():
        model = fit_model(
            spec["train"], spec["features"], spec["kind"], spec["params"], spec["seed"]
        )
        for split, frame in (("validation", validation), ("test", test)):
            score_sets[split][model_code] = score_model(
                model, frame, spec["features"], spec["availability"]
            )
        sample = test.loc[score_sets["test"][model_code].notna(), spec["features"]].head(50)
        if sample.empty:
            sample = spec["train"][spec["features"]].head(50)
        suffix = ".cbm" if spec["kind"] == "catboost" else ".joblib"
        saved = _save_model_roundtrip(
            model,
            kind=spec["kind"],
            path=model_dir / f"{fold_id}_{model_code}{suffix}",
            sample=sample,
        )
        reload_rows.append(
            {
                "availability_scenario_hours": int(scenario_hours),
                "fold_id": fold_id,
                "model_code": model_code,
                **saved,
            }
        )
        fit_frame = spec["train"]
        model_rows.append(
            {
                "availability_scenario_hours": int(scenario_hours),
                "fold_id": fold_id,
                "model_code": model_code,
                "model_kind": spec["kind"],
                "features_json": json.dumps(spec["features"], separators=(",", ":")),
                "feature_count": len(spec["features"]),
                "training_year_start": int(fit_frame["season"].min()),
                "training_year_end": int(fit_frame["season"].max()),
                "training_rows": int(len(fit_frame)),
                "training_field_seasons": int(fit_frame["field_season"].nunique()),
                "params_json": json.dumps(spec["params"], sort_keys=True, separators=(",", ":")),
                "seed": int(spec["seed"]),
                "availability_column": spec["availability"] or "calendar_always_available",
                "artifact": saved["artifact"],
                "artifact_sha256": saved["artifact_sha256"],
            }
        )

    lower, upper, matched_events = calendar_window_bounds(
        seasons, decisions, fold["train_years"], matched_to_common_mask=True
    )
    for split, frame in (("validation", validation), ("test", test)):
        score_sets[split][MODEL_A] = calendar_window_score(frame, lower, upper)
    calendar_artifact = model_dir / f"{fold_id}_{MODEL_A}.json"
    calendar_payload = {
        "model_code": MODEL_A,
        "fold_id": fold_id,
        "availability_scenario_hours": int(scenario_hours),
        "lower_day_of_year": lower,
        "upper_day_of_year": upper,
        "quantiles": [0.05, 0.95],
        "training_first_events": matched_events,
        "selection_population": "training_first_events_with_any_abc_complete_service_day",
    }
    _write_json(calendar_artifact, calendar_payload)
    model_rows.append(
        {
            "availability_scenario_hours": int(scenario_hours),
            "fold_id": fold_id,
            "model_code": MODEL_A,
            "model_kind": "empirical_calendar_window",
            "features_json": json.dumps(["issue_date_day_of_year"]),
            "feature_count": 1,
            "training_year_start": int(fold["train_years"][0]),
            "training_year_end": train_end,
            "training_rows": np.nan,
            "training_field_seasons": matched_events,
            "params_json": json.dumps({"lower_doy": lower, "upper_doy": upper}),
            "seed": np.nan,
            "availability_column": "abc_complete_in_primary_scope",
            "artifact": str(calendar_artifact),
            "artifact_sha256": sha256_file(calendar_artifact),
        }
    )

    policies: dict[str, Policy] = {}
    policy_rows: list[dict[str, Any]] = []
    for model_code in MODEL_CODES:
        policy, record = select_validation_policy(
            validation,
            validation_seasons,
            score_sets["validation"][model_code],
            model_code=model_code,
            fold_id=fold_id,
            scenario_hours=scenario_hours,
            contract=contract,
            fixed_binary_calendar=model_code == MODEL_A,
        )
        policies[model_code] = policy
        policy_rows.append(record)
    return score_sets, policies, policy_rows, reload_rows, model_rows


def _add_scenario(frame: pd.DataFrame, scenario_hours: int) -> pd.DataFrame:
    result = frame.copy()
    if "availability_scenario_hours" in result:
        if len(result) and not result["availability_scenario_hours"].eq(int(scenario_hours)).all():
            raise AssertionError("mixed lag scenarios inside one scenario result")
    else:
        result.insert(0, "availability_scenario_hours", int(scenario_hours))
    return result


def run_scenario_experiments(
    decisions: pd.DataFrame,
    seasons: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    past_features: Sequence[str],
    forecast_features: Sequence[str],
    scenario_hours: int,
    model_root: Path,
) -> dict[str, pd.DataFrame]:
    """Fit and evaluate all fixed candidates for one total-lag scenario."""

    output: dict[str, list[Any]] = {
        "predictions": [],
        "alarm_states": [],
        "event_metrics": [],
        "event_hits": [],
        "burden_metrics": [],
        "policy_selection": [],
        "model_reload_verification": [],
        "model_registry": [],
    }
    folds = [
        fold
        for fold in contract["rolling_origin_folds"]
        if 2020 <= int(fold["test_years"][0]) <= int(fold["test_years"][1]) <= 2025
    ]
    if not folds:
        raise ValueError("contract has no 2020-2025 outer folds")
    scenario_model_dir = model_root / f"lag_{int(scenario_hours)}h"
    feature_labels = {
        MODEL_A: "empirical_calendar_window_from_GFS_complete_training_events",
        MODEL_A_MATCHED_C0: "calendar_logit_fit_on_same_ABC_complete_rows",
        MODEL_A_STRONG: "calendar_catboost_fit_on_all_allowed_past_rows_from_2010",
        MODEL_B: "calendar_plus_existing_NASA_past_weather",
        MODEL_C: "calendar_plus_same_NASA_past_plus_archived_GFS_release",
    }

    for fold_index, fold in enumerate(folds):
        fold_id = str(fold["id"])
        test = decisions[_year_mask(decisions, fold["test_years"])].copy()
        test_seasons = seasons[_year_mask(seasons, fold["test_years"])].copy()
        score_sets, policies, policy_rows, reload_rows, model_rows = _fit_fold_models(
            decisions=decisions,
            seasons=seasons,
            fold=fold,
            contract=contract,
            past_features=past_features,
            forecast_features=forecast_features,
            scenario_hours=scenario_hours,
            model_dir=scenario_model_dir,
            fold_index=fold_index,
        )
        output["policy_selection"].extend(policy_rows)
        output["model_reload_verification"].extend(reload_rows)
        output["model_registry"].extend(model_rows)
        for model_code in MODEL_CODES:
            for scope in (PRIMARY_SCOPE, SERVICE_SCOPE):
                evaluated = _evaluate_model(
                    frame=test,
                    seasons=test_seasons,
                    raw_score=score_sets["test"][model_code],
                    policy=policies[model_code],
                    model_code=model_code,
                    fold_id=fold_id,
                    scenario_hours=scenario_hours,
                    feature_provenance=feature_labels[model_code],
                    evaluation_scope=scope,
                )
                states = evaluated["states"]
                output["alarm_states"].append(states)
                prediction_columns = [
                    "field_season",
                    "season",
                    "issue_date",
                    "issued_at",
                    "score",
                    "score_status",
                    "evaluation_scope",
                    "evaluation_scope_day",
                    "model_code",
                    "fold_id",
                    "model_version",
                    "feature_provenance",
                    "availability_scenario_hours",
                    "gfs_cell_id",
                    "gfs_feature_row_present",
                    "gfs_forecast_complete",
                    "gfs_missing_reason",
                    "abc_complete",
                    "issue_time_utc",
                    "selected_init_utc",
                    "assumed_available_at_utc",
                    "selection_rule_id",
                ]
                output["predictions"].append(
                    states[[column for column in prediction_columns if column in states]].copy()
                )
                output["event_metrics"].extend(evaluated["event_metrics"])
                output["event_hits"].extend(evaluated["event_hits"])
                output["burden_metrics"].extend(evaluated["burden_metrics"])

    result: dict[str, pd.DataFrame] = {}
    for name, values in output.items():
        if name in {"predictions", "alarm_states"}:
            result[name] = pd.concat(values, ignore_index=True) if values else pd.DataFrame()
        else:
            result[name] = pd.DataFrame(values)
    pooled = aggregate_pooled_metrics(
        result["event_metrics"],
        result["burden_metrics"],
        event_hits=result["event_hits"],
        alarm_states=result["alarm_states"],
        periods={"2020_2025": (2020, 2025)},
    )
    result.update({name: _add_scenario(frame, scenario_hours) for name, frame in pooled.items()})

    comparisons = (
        ((MODEL_C,), (MODEL_A, MODEL_B)),
        ((MODEL_B,), (MODEL_A,)),
        ((MODEL_C,), (MODEL_A_MATCHED_C0, MODEL_A_STRONG)),
    )
    annual_parts: list[pd.DataFrame] = []
    bootstrap_parts: list[pd.DataFrame] = []
    uncertainty = contract.get("uncertainty", {})
    seed = int(uncertainty.get("paired_year_bootstrap_seed", 20260910))
    repeats = int(uncertainty.get("paired_year_bootstrap_repeats", 2000))
    for candidates, baselines in comparisons:
        annual, bootstrap = paired_year_bootstrap_cycle2(
            result["event_hits"],
            result["alarm_states"],
            candidates=candidates,
            baselines=baselines,
            scopes=(PRIMARY_SCOPE, SERVICE_SCOPE),
            slice_name="A_plus_B",
            years=(2020, 2025),
            seed=seed,
            n_bootstrap=repeats,
        )
        annual_parts.append(annual)
        bootstrap_parts.append(bootstrap)
    result["annual_paired"] = _add_scenario(
        pd.concat(annual_parts, ignore_index=True), scenario_hours
    )
    result["paired_year_bootstrap"] = _add_scenario(
        pd.concat(bootstrap_parts, ignore_index=True), scenario_hours
    )
    result["leave_one_year_out"] = _add_scenario(
        leave_one_year_out_cycle2(result["annual_paired"]), scenario_hours
    )

    intersection_parts: list[pd.DataFrame] = []
    membership_parts: list[pd.DataFrame] = []
    for scope in (PRIMARY_SCOPE, SERVICE_SCOPE):
        summary, membership = event_intersections(
            result["event_hits"],
            pairs=((MODEL_A, MODEL_C), (MODEL_B, MODEL_C), (MODEL_A_STRONG, MODEL_C)),
            scope=scope,
            slice_name="A_plus_B",
            years=(2020, 2025),
            event_key_namespace=f"gfs_abc_{scenario_hours}h",
        )
        intersection_parts.append(summary)
        membership_parts.append(membership)
    result["event_intersections"] = _add_scenario(
        pd.concat(intersection_parts, ignore_index=True), scenario_hours
    )
    result["event_intersection_membership"] = _add_scenario(
        pd.concat(membership_parts, ignore_index=True), scenario_hours
    )
    result["computability"] = result["burden_metrics"].merge(
        result["event_metrics"][[
            "availability_scenario_hours",
            "model_code",
            "fold_id",
            "evaluation_scope",
            "slice",
            "events_with_warning_opportunity",
            "computable_events",
            "computable_event_fraction",
        ]],
        on=[
            "availability_scenario_hours",
            "model_code",
            "fold_id",
            "evaluation_scope",
            "slice",
        ],
        how="left",
        validate="one_to_one",
    )
    return result


def availability_sensitivity_summary(
    decisions: pd.DataFrame,
    forecast_features: pd.DataFrame,
    scenarios: Sequence[int],
    *,
    feature_columns: Sequence[str] = GFS_FORECAST_FEATURES,
    past_feature_columns: Sequence[str] = NASA_COMMON_FEATURES,
) -> pd.DataFrame:
    reported_periods = {
        "training_support_2015_2019": (2015, 2019, "training_support"),
        "2020_2025": (2020, 2025, "external_evaluation_previously_studied"),
    }
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        joined = attach_forecast_features(
            decisions,
            forecast_features,
            scenario_hours=int(scenario),
            feature_columns=feature_columns,
            past_feature_columns=past_feature_columns,
        )
        for season, group in joined.groupby("season", sort=True):
            season_number = int(season)
            period_name = next(
                (
                    name
                    for name, (lower, upper, _) in reported_periods.items()
                    if lower <= season_number <= upper
                ),
                None,
            )
            if period_name is None:
                continue
            service = group[group["service_active"].astype(bool)]
            actionable = service[
                service["warnable_first_event"].astype(bool)
                & service["days_to_first_recorded_event"].between(3, 10)
            ]
            rows.append(
                {
                    "availability_scenario_hours": int(scenario),
                    "aggregation": "year",
                    "period": str(season_number),
                    "population_role": reported_periods[period_name][2],
                    "season": season_number,
                    "service_field_days": int(len(service)),
                    "forecast_complete_field_days": int(service["gfs_forecast_complete"].sum()),
                    "abc_complete_field_days": int(service["abc_complete"].sum()),
                    "forecast_complete_fraction": float(service["gfs_forecast_complete"].mean())
                    if len(service)
                    else np.nan,
                    "abc_complete_fraction": float(service["abc_complete"].mean())
                    if len(service)
                    else np.nan,
                    "warnable_events": int(
                        service.loc[
                            service["warnable_first_event"].astype(bool), "field_season"
                        ].nunique()
                    ),
                    "warnable_events_with_any_abc_actionable_day": int(
                        actionable.loc[actionable["abc_complete"], "field_season"].nunique()
                    ),
                }
            )
    annual = pd.DataFrame(rows)
    pooled_rows: list[dict[str, Any]] = []
    for scenario, scenario_group in annual.groupby(
        "availability_scenario_hours", sort=True
    ):
        for period, (lower, upper, role) in reported_periods.items():
            group = scenario_group[scenario_group["season"].between(lower, upper)]
            if group.empty:
                continue
            days = int(group["service_field_days"].sum())
            forecast_days = int(group["forecast_complete_field_days"].sum())
            abc_days = int(group["abc_complete_field_days"].sum())
            pooled_rows.append(
                {
                    "availability_scenario_hours": int(scenario),
                    "aggregation": "pooled_year_counts",
                    "period": period,
                    "population_role": role,
                    "season": pd.NA,
                    "service_field_days": days,
                    "forecast_complete_field_days": forecast_days,
                    "abc_complete_field_days": abc_days,
                    "forecast_complete_fraction": forecast_days / days if days else np.nan,
                    "abc_complete_fraction": abc_days / days if days else np.nan,
                    "warnable_events": int(group["warnable_events"].sum()),
                    "warnable_events_with_any_abc_actionable_day": int(
                        group["warnable_events_with_any_abc_actionable_day"].sum()
                    ),
                }
            )
    return pd.concat([annual, pd.DataFrame(pooled_rows)], ignore_index=True)


def _source_snapshot(paths: Sequence[Path], destination: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in dict.fromkeys(item.resolve() for item in paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "utf8_content": path.read_text(encoding="utf-8"),
            }
        )
    payload = {"format": "agro_phenology_gfs_abc_source_snapshot_v1", "files": files}
    _write_json(destination, payload)
    return {
        "path": destination.name,
        "sha256": sha256_file(destination),
        "files": [{"path": row["path"], "sha256": row["sha256"]} for row in files],
    }


def _output_hashes(run_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        relative = str(path.relative_to(run_dir))
        if relative == "execution_manifest.json":
            continue
        result[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def verify_run(run_dir: str | Path) -> dict[str, Any]:
    directory = Path(run_dir)
    manifest_path = directory / "execution_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for relative, expected in manifest.get("output_hashes", {}).items():
        path = directory / relative
        if not path.is_file() or sha256_file(path) != expected["sha256"]:
            failures.append(relative)
    return {
        "status": "passed" if manifest.get("status") == "complete" and not failures else "failed",
        "run_id": manifest.get("run_id"),
        "manifest_sha256": sha256_file(manifest_path),
        "outputs_checked": len(manifest.get("output_hashes", {})),
        "failures": failures,
    }


def _write_frames(run_dir: Path, outputs: Mapping[str, pd.DataFrame]) -> None:
    parquet = {"predictions", "alarm_states", "event_hits", "event_intersection_membership"}
    for name, frame in outputs.items():
        if name in parquet:
            frame.to_parquet(run_dir / f"{name}.parquet", index=False)
        else:
            frame.to_csv(run_dir / f"{name}.csv", index=False)


def _environment_versions() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package in (
        "numpy",
        "pandas",
        "pyarrow",
        "catboost",
        "scikit-learn",
        "joblib",
        "pytest",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "Нет вычислимых результатов."
    selected = frame[[column for column in columns if column in frame]].copy()
    headers = [str(column) for column in selected.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in selected.itertuples(index=False, name=None):
        values: list[str] = []
        for value in row:
            rendered = _fmt(value) if isinstance(value, (float, np.floating)) else str(value)
            values.append(rendered.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report_ru(
    path: Path,
    *,
    pooled: pd.DataFrame,
    annual: pd.DataFrame,
    bootstrap: pd.DataFrame,
    availability: pd.DataFrame,
    primary_scenario: int,
    scenarios: Sequence[int],
    contract: Mapping[str, Any],
) -> None:
    primary = pooled[
        pooled["availability_scenario_hours"].eq(primary_scenario)
        & pooled["period"].eq("2020_2025")
        & pooled["evaluation_scope"].eq(PRIMARY_SCOPE)
        & pooled["slice"].eq("A_plus_B")
        & pooled["model_code"].isin(PRIMARY_MODELS)
    ].copy()
    service = pooled[
        pooled["availability_scenario_hours"].eq(primary_scenario)
        & pooled["period"].eq("2020_2025")
        & pooled["evaluation_scope"].eq(SERVICE_SCOPE)
        & pooled["slice"].eq("A_plus_B")
        & pooled["model_code"].isin(PRIMARY_MODELS)
    ].copy()
    all_scenarios = pooled[
        pooled["period"].eq("2020_2025")
        & pooled["evaluation_scope"].eq(PRIMARY_SCOPE)
        & pooled["slice"].eq("A_plus_B")
        & pooled["model_code"].isin(PRIMARY_MODELS)
    ].copy()
    secondary = pooled[
        pooled["availability_scenario_hours"].eq(primary_scenario)
        & pooled["period"].eq("2020_2025")
        & pooled["evaluation_scope"].eq(PRIMARY_SCOPE)
        & pooled["slice"].eq("A_plus_B")
        & pooled["model_code"].isin((MODEL_A_MATCHED_C0, MODEL_A_STRONG))
    ].copy()
    comparisons = bootstrap[
        bootstrap["availability_scenario_hours"].eq(primary_scenario)
        & bootstrap["evaluation_scope"].eq(PRIMARY_SCOPE)
        & bootstrap["candidate"].eq(MODEL_C)
        & bootstrap["baseline"].isin((MODEL_A, MODEL_B))
    ].copy()
    annual_primary = annual[
        annual["availability_scenario_hours"].eq(primary_scenario)
        & annual["evaluation_scope"].eq(PRIMARY_SCOPE)
        & annual["candidate"].eq(MODEL_C)
        & annual["baseline"].isin((MODEL_A, MODEL_B))
    ].copy()
    budget = contract["notification_policy"]["research_budget"]
    messages_limit = float(budget["messages_per_30_field_days_max"])
    alarm_limit = float(budget["active_alarm_fraction_max"])
    annual_primary["candidate_budget_status"] = np.where(
        (annual_primary["candidate_messages_per_30_field_days"] <= messages_limit)
        & (annual_primary["candidate_active_alarm_fraction"] <= alarm_limit),
        "within_both_limits",
        "exceeds_at_least_one_limit",
    )
    main_columns = (
        "model_code",
        "timely_hits",
        "events_with_warning_opportunity",
        "timely_recall",
        "messages",
        "messages_per_30_field_days",
        "active_alarm_days",
        "active_alarm_fraction",
        "computable_fraction",
    )
    comparison_columns = (
        "baseline",
        "candidate_hits",
        "baseline_hits",
        "opportunities",
        "delta_timely_recall",
        "delta_timely_recall_low",
        "delta_timely_recall_high",
        "candidate_messages_per_30_field_days",
        "baseline_messages_per_30_field_days",
        "candidate_active_alarm_fraction",
        "baseline_active_alarm_fraction",
        "interval_status",
    )
    annual_columns = (
        "season",
        "baseline",
        "candidate_hits",
        "baseline_hits",
        "opportunities",
        "delta_timely_recall",
        "candidate_messages",
        "baseline_messages",
        "candidate_messages_per_30_field_days",
        "baseline_messages_per_30_field_days",
        "candidate_alarm_days",
        "baseline_alarm_days",
        "candidate_active_alarm_fraction",
        "baseline_active_alarm_fraction",
        "candidate_budget_status",
    )
    by_model = primary.set_index("model_code") if len(primary) else pd.DataFrame()
    if all(code in by_model.index for code in PRIMARY_MODELS):
        c, a, b = by_model.loc[MODEL_C], by_model.loc[MODEL_A], by_model.loc[MODEL_B]
        delta_a = float(c["timely_recall"] - a["timely_recall"])
        delta_b = float(c["timely_recall"] - b["timely_recall"])
        verdict = (
            "На точечных оценках C дала выигрыш и сверх календаря, и сверх прошлой погоды."
            if delta_a > 0 and delta_b > 0
            else "C не дала одновременного выигрыша сверх календаря и сверх прошлой погоды."
        )
        within_budget = bool(
            c["messages_per_30_field_days"] <= messages_limit
            and c["active_alarm_fraction"] <= alarm_limit
        )
        annual_c = annual_primary[annual_primary["baseline"].eq(MODEL_A)]
        annual_exceeded = annual_c[
            annual_c["candidate_budget_status"].eq("exceeds_at_least_one_limit")
        ]
        exceeded_years = sorted(
            pd.to_numeric(annual_exceeded["season"], errors="raise")
            .astype(int)
            .tolist()
        )
        annual_budget_note = (
            f" В {len(exceeded_years)} из {len(annual_c)} внешних лет C превысила "
            "хотя бы один годовой лимит: "
            + ", ".join(str(year) for year in exceeded_years)
            + "."
            if exceeded_years
            else f" Во всех {len(annual_c)} внешних годах C осталась внутри обоих годовых лимитов."
        )
        interval_rows = comparisons.set_index("baseline") if len(comparisons) else pd.DataFrame()
        uncertainty_note = ""
        if all(name in interval_rows.index for name in (MODEL_A, MODEL_B)):
            includes_zero = any(
                float(interval_rows.loc[name, "delta_timely_recall_low"]) <= 0
                <= float(interval_rows.loc[name, "delta_timely_recall_high"])
                for name in (MODEL_A, MODEL_B)
            )
            uncertainty_note = (
                " Парный интервал хотя бы одного основного сравнения включает ноль."
                if includes_zero
                else " Парные интервалы обоих основных сравнений не включают ноль."
            )
        conclusion = (
            verdict
            + uncertainty_note
            + f" Наблюдаемая разница C против A: {_fmt(delta_a)} "
            f"по своевременному покрытию и {_fmt(c['messages_per_30_field_days'] - a['messages_per_30_field_days'])} "
            "сообщения на 30 поле-дней. "
            f"Разница C против B: {_fmt(delta_b)} и "
            f"{_fmt(c['messages_per_30_field_days'] - b['messages_per_30_field_days'])} соответственно. "
            f"Совокупный бюджет 2020–2025 для C {'соблюдён' if within_budget else 'превышен'}."
            + annual_budget_note
            + " Порог после внешнего результата не менялся."
        )
    else:
        conclusion = "Прямой вывод недоступен: основная A/B/C-таблица неполна."
    availability_pooled = availability[
        availability["aggregation"].eq("pooled_year_counts")
    ]
    lines = [
        "# Архивные прогнозы GFS: законченный A/B/C-эксперимент",
        "",
        "Это ретроспективная оценка на уже изученных внешних годах 2020–2025. Она не является новым независимым тестом и оценивает предупреждение первой зарегистрированной записи, а не биологическое начало заражения.",
        "",
        f"Основной сценарий предполагает доступность GFS через {primary_scenario} ч после инициализации. Проверена чувствительность для total-lag сценариев: {', '.join(map(str, scenarios))} ч. Исторического журнала публикации нет; это явно обозначенное допущение.",
        "",
        "## Основное сравнение на общей complete-case маске",
        "",
        _markdown_table(primary, main_columns),
        "",
        "A — эмпирическое календарное окно, заново оценённое только по training first events с GFS-complete покрытием. B и C используют один и тот же неглубокий CatBoost; C добавляет 12 агрегатов настоящего архивного выпуска GFS к тем же календарным и NASA past-only признакам.",
        "",
        "## Парная неопределённость C против A и B",
        "",
        _markdown_table(comparisons, comparison_columns),
        "",
        "## Результаты C против A и B по внешним годам",
        "",
        _markdown_table(annual_primary, annual_columns),
        "",
        "## Контроли календаря",
        "",
        _markdown_table(secondary, main_columns),
        "",
        "A_matched_C0 — логистический календарный контроль на тех же ABC-complete fit rows. A_strong_calendar_catboost обучен на всех разрешённых календарных строках с 2010 года до конца training-блока. Оба являются вторичными контролями; основной A остаётся эмпирическим календарным окном.",
        "",
        "## Полный сервисный replay",
        "",
        _markdown_table(service, main_columns),
        "",
        "В replay сохранён исходный ежедневный календарь. Дни без прогноза не удалялись и cooldown не сжимался. В service-сценарии A вычислим на календарных днях, B — при наличии NASA, C — только при наличии NASA и GFS.",
        "",
        "## Вычислимость и задержка",
        "",
        _markdown_table(availability_pooled, availability_pooled.columns),
        "",
        "Строка 2020_2025 относится только к уже изученным внешним годам. training_support_2015_2019 показана отдельно и не смешана с внешней оценкой; годы 2010–2014 и 2026 в эту сводку GFS не входят.",
        "",
        "## Чувствительность A/B/C к total lag",
        "",
        _markdown_table(
            all_scenarios,
            ("availability_scenario_hours", *main_columns),
        ),
        "",
        "## Прямой результат",
        "",
        conclusion,
        "",
        "Равный допустимый бюджет не означает равной фактической нагрузки. Годовые превышения бюджета показаны в таблице как результат и не исправлялись перенастройкой. Отсутствующие прогнозы не заполнялись будущим реанализом. Шестичасовые значения не интерпретируются как точные почасовые длительности Хаттона.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    *,
    contract_path: str | Path,
    run_id: str,
    forecast_features_path: str | Path | None = None,
    source_manifest_path: str | Path | None = None,
    request_plan_path: str | Path | None = None,
    request_plan_manifest_path: str | Path | None = None,
    v3_dir: str | Path = DEFAULT_V3,
) -> Path:
    contract_source = _resolve(contract_path).resolve()
    contract = json.loads(contract_source.read_text(encoding="utf-8"))
    validate_contract(contract)
    past_features, forecast_features = contract_feature_columns(contract)
    primary_scenario, scenarios = availability_scenarios(contract)
    output_root = _resolve(contract.get("output_root", DEFAULT_OUTPUT_ROOT))
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"immutable run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    v3 = _resolve(v3_dir).resolve()
    configured_forecast = contract.get("inputs", {}).get("gfs_forecast_features")
    if forecast_features_path is None and configured_forecast is None:
        raise ValueError("--forecast-features is required because the frozen contract has no data path")
    forecast_path = _resolve(forecast_features_path or configured_forecast).resolve()
    bundle_paths = resolve_forecast_input_bundle_paths(
        forecast_path,
        source_manifest_path=source_manifest_path,
        request_plan_path=request_plan_path,
        request_plan_manifest_path=request_plan_manifest_path,
    )
    expected_parent = contract.get("parent_manifest_sha256", V3_MANIFEST_SHA256)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "mode": "retrospective_archived_gfs_A_B_C_fixed_first_pass",
        "started_at_utc": _utc_now(),
        "command": [sys.executable, *sys.argv],
        "environment": _environment_versions(),
        "primary_availability_scenario_hours": primary_scenario,
        "availability_scenarios_hours": list(scenarios),
        "new_optuna_or_ranker": False,
        "notifications_sent": 0,
        "schedule_activated": False,
        "frozen_c6_loaded_or_modified": False,
    }
    _write_json(run_dir / "execution_manifest.json", manifest)
    try:
        parent_before = verify_frozen_run(v3, str(expected_parent))
        _write_json(run_dir / "parent_integrity_before.json", parent_before)
        shutil.copy2(contract_source, run_dir / "evaluation_contract.json")
        protocol = contract_source.with_name("gfs_forecast_protocol.md")
        if protocol.is_file():
            shutil.copy2(protocol, run_dir / "protocol.md")

        source_paths = [
            Path(__file__),
            REPO_ROOT / "download_gfs_archive.py",
            REPO_ROOT / "pyproject.toml",
            REPO_ROOT / "requirements-early-warning.txt",
            REPO_ROOT / "src/agro_phenology/gfs_archive.py",
            REPO_ROOT / "src/agro_phenology/gfs_feature_builder.py",
            REPO_ROOT / "src/agro_phenology/early_warning_core.py",
            REPO_ROOT / "src/agro_phenology/early_warning_models.py",
            REPO_ROOT / "src/agro_phenology/early_warning_cycle2_diagnostics.py",
            REPO_ROOT / "src/agro_phenology/early_warning_cycle2_reporting.py",
            REPO_ROOT / "src/agro_phenology/early_warning_cycle3_pipeline.py",
            REPO_ROOT / "src/agro_phenology/early_warning_reporting.py",
            REPO_ROOT / "tests/test_early_warning_gfs_experiment.py",
            REPO_ROOT / "tests/test_gfs_archive.py",
            REPO_ROOT / "tests/test_gfs_feature_builder.py",
            REPO_ROOT / "tests/test_download_gfs_archive.py",
            REPO_ROOT / "tests/test_early_warning_policy.py",
            contract_source,
        ]
        manifest["source_snapshot"] = _source_snapshot(
            source_paths, run_dir / "source_snapshot.json"
        )
        _write_json(run_dir / "execution_manifest.json", manifest)

        decisions = pd.read_parquet(v3 / "daily_decisions.parquet")
        seasons = pd.read_parquet(v3 / "field_seasons.parquet")
        forecast = load_forecast_feature_table(
            forecast_path, feature_columns=forecast_features
        )
        bundle_audit = validate_forecast_input_bundle(
            forecast,
            bundle_paths=bundle_paths,
            contract_path=contract_source,
            parent_decisions_path=v3 / "daily_decisions.parquet",
            expected_scenarios=scenarios,
            feature_columns=forecast_features,
        )
        _write_json(run_dir / "gfs_input_bundle_audit.json", bundle_audit)
        manifest["gfs_input_bundle"] = {
            "status": bundle_audit["status"],
            "planned_checkpoints": bundle_audit["planned_checkpoints"],
            "assembled_checkpoints": bundle_audit["assembled_checkpoints"],
            "planned_feature_rows": bundle_audit["planned_feature_rows"],
            "assembled_feature_rows": bundle_audit["assembled_feature_rows"],
            "privacy": "local_private_do_not_publish",
            "contains_coarse_location_footprint": True,
        }
        _write_json(run_dir / "execution_manifest.json", manifest)
        missing_scenarios = sorted(
            set(scenarios).difference(forecast["availability_scenario_hours"].unique())
        )
        if missing_scenarios:
            raise ValueError(f"forecast table misses declared total-lag scenarios: {missing_scenarios}")
        availability = availability_sensitivity_summary(
            decisions,
            forecast,
            scenarios,
            feature_columns=forecast_features,
            past_feature_columns=past_features,
        )

        all_outputs: dict[str, list[pd.DataFrame]] = {}
        split_rows: list[dict[str, Any]] = []
        for scenario in scenarios:
            joined = attach_forecast_features(
                decisions,
                forecast,
                scenario_hours=scenario,
                feature_columns=forecast_features,
                past_feature_columns=past_features,
            )
            private_columns = [
                column
                for column in (
                    "field_season",
                    "season",
                    "issue_date",
                    "issued_at",
                    "service_active",
                    "target_class",
                    "target_observable",
                    "days_to_first_recorded_event",
                    "coordinate_scope",
                    "previous_visit_gap_days",
                    "gfs_cell_id",
                    "nasa_common_complete",
                    "gfs_forecast_complete",
                    "gfs_missing_reason",
                    "abc_complete",
                    "selected_init_utc",
                    "assumed_available_at_utc",
                    "selection_rule_id",
                    *CALENDAR_FEATURES,
                    *past_features,
                    *forecast_features,
                )
                if column in joined
            ]
            joined[private_columns].to_parquet(
                run_dir / f"model_matrix_private_lag_{scenario}h.parquet", index=False
            )
            outputs = run_scenario_experiments(
                joined,
                seasons,
                contract,
                past_features=past_features,
                forecast_features=forecast_features,
                scenario_hours=scenario,
                model_root=run_dir / "models",
            )
            for name, frame in outputs.items():
                all_outputs.setdefault(name, []).append(frame)
            for fold in contract["rolling_origin_folds"]:
                for role, key in (
                    ("train", "train_years"),
                    ("validation", "validation_years"),
                    ("test", "test_years"),
                ):
                    selected = joined[_year_mask(joined, fold[key])]
                    fit_rows = model_training_rows(joined, fold[key]) if role == "train" else None
                    split_rows.append(
                        {
                            "availability_scenario_hours": scenario,
                            "fold_id": fold["id"],
                            "role": role,
                            "years": json.dumps(fold[key]),
                            "daily_rows": int(len(selected)),
                            "service_field_days": int(selected["service_active"].sum()),
                            "abc_complete_field_days": int(selected["abc_complete"].sum()),
                            "observable_abc_fit_rows": int(len(fit_rows)) if fit_rows is not None else np.nan,
                            "field_seasons": int(selected["field_season"].nunique()),
                        }
                    )

        combined = {
            name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            for name, frames in all_outputs.items()
        }
        combined["availability_sensitivity"] = availability
        combined["split_manifest"] = pd.DataFrame(split_rows)
        _write_frames(run_dir, combined)
        states = combined["alarm_states"]
        notifications = states[
            states["message_issued"].astype(bool) | states["suppressed_repeat"].astype(bool)
        ].copy()
        notifications.to_parquet(run_dir / "notification_log.parquet", index=False)

        feature_manifest = {
            "past_weather_features": past_features,
            "forecast_features": forecast_features,
            "forecast_sampling": "selected_six_hour_valid_slots",
            "forecast_bands_decision_relative_hours": {
                "d1_3": "(0,72]",
                "d4_7": "(72,168]",
            },
            "humidity_semantics": "synoptic_six_hour_samples_not_exact_hourly_duration",
            "precipitation_semantics": "non_overlapping_six_hour_accumulations_assigned_by_interval_end",
            "past_weather_source": "NASA features saved by 20260910_first_cycle_v3",
            "forecast_source_dataset": "GDEX d084001 NCEP GFS archived releases",
            "common_mask": "service_active AND nasa_common_complete AND gfs_forecast_complete",
            "primary_models_share_exact_mask": True,
            "full_daily_calendar_used_for_replay": True,
            "cooldown_calendar_compressed": False,
            "target_columns_excluded_from_model_features": True,
            "frozen_c6_reused_or_modified": False,
            "contains_coarse_location_footprint": True,
            "distribution": "local_private_do_not_publish",
        }
        _write_json(run_dir / "feature_manifest.json", feature_manifest)
        _write_json(
            run_dir / "input_audit.json",
            {
                "parent_v3": {
                    "path": str(v3),
                    "manifest_sha256": expected_parent,
                    "daily_decisions_sha256": sha256_file(v3 / "daily_decisions.parquet"),
                    "field_seasons_sha256": sha256_file(v3 / "field_seasons.parquet"),
                    "events_sha256": sha256_file(v3 / "events.parquet"),
                },
                "gfs_forecast_features": {
                    "path": str(forecast_path),
                    "sha256": sha256_file(forecast_path),
                    "rows": int(len(forecast)),
                    "issue_date_min": forecast["issue_date"].min().date().isoformat(),
                    "issue_date_max": forecast["issue_date"].max().date().isoformat(),
                    "cells": int(forecast["gfs_cell_id"].nunique()),
                    "scenarios_total_lag_hours": sorted(
                        forecast["availability_scenario_hours"].unique().tolist()
                    ),
                    "historical_retrieval_time_is_publication_evidence": False,
                    "source_manifest_sha256": bundle_audit["hashes"]["source_manifest"],
                    "request_plan_sha256": bundle_audit["hashes"]["request_plan"],
                    "request_plan_manifest_sha256": bundle_audit["hashes"][
                        "request_plan_manifest"
                    ],
                    "planned_checkpoints": bundle_audit["planned_checkpoints"],
                    "assembled_checkpoints": bundle_audit["assembled_checkpoints"],
                    "contains_coarse_location_footprint": True,
                    "distribution": "local_private_do_not_publish",
                },
            },
        )
        _write_report_ru(
            run_dir / "report_ru.md",
            pooled=combined["pooled_summary"],
            annual=combined["annual_paired"],
            bootstrap=combined["paired_year_bootstrap"],
            availability=availability,
            primary_scenario=primary_scenario,
            scenarios=scenarios,
            contract=contract,
        )
        reproduce = (
            "# Воспроизведение GFS A/B/C-эксперимента\n\n"
            "```bash\n"
            f"{sys.executable} -m agro_phenology.early_warning_gfs_experiment run \\\n"
            f"  --contract {contract_source} \\\n"
            f"  --v3-run {v3} \\\n"
            f"  --forecast-features {forecast_path} \\\n"
            f"  --gfs-source-manifest {bundle_paths['source_manifest']} \\\n"
            f"  --gfs-request-plan {bundle_paths['request_plan']} \\\n"
            f"  --gfs-request-plan-manifest {bundle_paths['request_plan_manifest']} \\\n"
            "  --run-id <new_unique_run_id>\n"
            "```\n\n"
            "Каталог запуска должен быть новым. Команда не запускает Optuna, не изменяет frozen C6, не активирует расписание и не отправляет уведомления.\n"
        )
        (run_dir / "REPRODUCE.md").write_text(reproduce, encoding="utf-8")

        tests = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_early_warning_gfs_experiment.py",
                "tests/test_gfs_archive.py",
                "tests/test_gfs_feature_builder.py",
                "tests/test_early_warning_policy.py",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        test_result = {
            "status": "passed" if tests.returncode == 0 else "failed",
            "command": tests.args,
            "returncode": tests.returncode,
            "stdout": tests.stdout,
            "stderr": tests.stderr,
        }
        _write_json(run_dir / "test_results.json", test_result)
        if tests.returncode:
            raise RuntimeError("GFS experiment tests failed")

        parent_after = verify_frozen_run(v3, str(expected_parent))
        _write_json(run_dir / "parent_integrity_after.json", parent_after)
        if parent_before["execution_manifest_sha256"] != parent_after["execution_manifest_sha256"]:
            raise AssertionError("parent manifest changed during GFS experiment")
        manifest.update(
            status="complete",
            completed_at_utc=_utc_now(),
            parent_manifest_sha256=str(expected_parent),
            forecast_features_sha256=sha256_file(forecast_path),
            output_hashes=_output_hashes(run_dir),
            scientific_status="retrospective_already_studied_years_not_independent_test",
        )
        _write_json(run_dir / "execution_manifest.json", manifest)
        return run_dir
    except Exception as error:
        manifest.update(
            status="failed",
            failed_at_utc=_utc_now(),
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
            partial_output_hashes=_output_hashes(run_dir),
        )
        _write_json(run_dir / "execution_manifest.json", manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run fixed archived-GFS A/B/C experiment")
    run.add_argument("--contract", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--forecast-features")
    run.add_argument("--gfs-source-manifest")
    run.add_argument("--gfs-request-plan")
    run.add_argument("--gfs-request-plan-manifest")
    run.add_argument("--v3-run", default=str(DEFAULT_V3))
    verify = subparsers.add_parser("verify-run", help="verify saved output hashes")
    verify.add_argument("--run-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        result = {
            "status": "complete",
            "run_dir": str(
                run_experiment(
                    contract_path=args.contract,
                    run_id=args.run_id,
                    forecast_features_path=args.forecast_features,
                    source_manifest_path=args.gfs_source_manifest,
                    request_plan_path=args.gfs_request_plan,
                    request_plan_manifest_path=args.gfs_request_plan_manifest,
                    v3_dir=args.v3_run,
                )
            ),
        }
    else:
        result = verify_run(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return 0 if result["status"] in {"complete", "passed"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
