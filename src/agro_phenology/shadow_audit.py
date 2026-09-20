"""Operational input audit for the potato late-blight shadow contour.

The audit is intentionally local and read-only.  It separates facts measured
from the frozen workspace from provider documentation and never probes a
location unless an explicit active-field registry is supplied by an operator.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .shadow_sources import (
    FROZEN_ERA5_PROFILE_ID,
    FROZEN_ERA5_VARIABLES,
    MINIMUM_EPISODE_HISTORY_DAYS,
    source_profile_contract,
)


AUDIT_VERSION = "1.0.0"
DECISION_TIMEZONE = "Europe/Riga"
SCHEDULED_LOCAL_TIME = "08:00:00"
OFFICIAL_SOURCE_PAGES = {
    "ecmwf_era5_update_frequency": (
            "https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation"
    ),
    "open_meteo_historical_weather": (
        "https://open-meteo.com/en/docs/historical-weather-api"
    ),
    "open_meteo_single_runs": "https://open-meteo.com/en/docs/single-runs-api",
    "nasa_power_sources": "https://power.larc.nasa.gov/docs/methodology/data/sources/",
    "nasa_power_daily_api": "https://power.larc.nasa.gov/docs/services/api/temporal/daily/",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scheduled_slot(issue_local_date: date) -> tuple[str, str]:
    local = datetime.combine(
        issue_local_date,
        time(8, 0),
        tzinfo=ZoneInfo(DECISION_TIMEZONE),
    )
    return local.isoformat(), local.astimezone(timezone.utc).isoformat()


def _days_between(later: date, earlier: date) -> int:
    return int((later - earlier).days)


def _cache_payload_keys(project_root: Path) -> set[str]:
    """Return SHA-like filenames for local payloads without reading their bytes."""

    keys: set[str] = set()
    cache_root = project_root / "data/cache"
    if not cache_root.is_dir():
        return keys
    for path in cache_root.rglob("*"):
        if not path.is_file() or ".metadata." in path.name:
            continue
        name = path.name
        for ending in (".json.gz", ".json", ".gz"):
            if name.endswith(ending):
                name = name[: -len(ending)]
                break
        if len(name) == 64 and all(character in "0123456789abcdef" for character in name):
            keys.add(name)
    return keys


def _snapshot_record(
    path: Path,
    frame: pd.DataFrame,
    *,
    project_root: Path,
    date_column: str,
) -> dict[str, Any]:
    values = pd.to_datetime(frame[date_column], errors="raise")
    return {
        "relative_path": path.resolve().relative_to(project_root.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "date_min": str(values.min().date()),
        "date_max": str(values.max().date()),
    }


def find_active_field_registry(
    project_root: str | Path,
    explicit_path: str | Path | None = None,
) -> dict[str, Any]:
    """Find only an explicit operational registry, never historical field tables."""

    root = Path(project_root).resolve()
    if explicit_path is not None:
        candidates = [Path(explicit_path).expanduser().resolve()]
        discovery = "operator_supplied_path"
    else:
        candidates = [
            root / "config/shadow_active_fields.json",
            root / "data/private/shadow_active_fields.json",
        ]
        discovery = "allowlisted_operational_paths_only"
    existing = [path for path in candidates if path.is_file()]
    incompatible = [path for path in existing if path.suffix.lower() != ".json"]
    status = (
        "incompatible_schema_format"
        if incompatible
        else "available"
        if len(existing) == 1
        else "ambiguous"
        if len(existing) > 1
        else "missing"
    )
    return {
        "status": status,
        "discovery": discovery,
        "accepted_formats": ["json"],
        "searched_paths": [
            str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            for path in candidates
        ],
        "matched_paths": [
            str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            for path in existing
        ],
        "historical_tables_considered_active": False,
    }


def build_source_compatibility_manifest(
    project_root: str | Path,
    *,
    issue_local_date: date | str,
    audited_at_utc: str | None = None,
    active_field_registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a machine-readable operational audit without live requests."""

    root = Path(project_root).resolve()
    issue = date.fromisoformat(issue_local_date) if isinstance(issue_local_date, str) else issue_local_date
    cutoff_date = issue - timedelta(days=2)
    local_slot, utc_slot = _scheduled_slot(issue)
    audited_at = audited_at_utc or datetime.now(timezone.utc).isoformat()

    frozen_root = root / "docs/extra/vaad_pipeline_repro_20260905/frozen_external"
    era_path = frozen_root / "era5_potato_daily.parquet"
    era_meta_path = frozen_root / "era5_metadata.csv"
    nasa_path = frozen_root / "nasa_daily.parquet"
    required = [era_path, era_meta_path, nasa_path]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required frozen source audit files are missing: {missing}")

    era = pd.read_parquet(era_path)
    era_meta = pd.read_csv(era_meta_path)
    nasa = pd.read_parquet(nasa_path)
    era_snapshot = _snapshot_record(
        era_path, era, project_root=root, date_column="date"
    )
    era_snapshot.update(
        {
            "weather_cells": int(era["weather_cell"].nunique()),
            "accepted_rows": int(era["accepted"].eq(True).sum()),
            "model_metadata_values": sorted(era_meta["model"].astype(str).unique()),
            "downscaling_metadata_values": sorted(
                bool(value) for value in era_meta["downscaling"].unique()
            ),
            "request_metadata_rows": int(len(era_meta)),
            "metadata_sha256": sha256_file(era_meta_path),
            "returned_grid_matches_requested_rows": int(
                (
                    era_meta["requested_latitude"].eq(era_meta["returned_latitude"])
                    & era_meta["requested_longitude"].eq(era_meta["returned_longitude"])
                ).sum()
            ),
            "provider_or_data_version": None,
            "raw_training_responses_present": False,
            "retrieval_timestamps_present": False,
            "latest_valid_date_gap_to_required_days": _days_between(
                cutoff_date, pd.Timestamp(era["date"].max()).date()
            ),
        }
    )
    nasa_snapshot = _snapshot_record(
        nasa_path, nasa, project_root=root, date_column="date"
    )
    nasa_snapshot.update(
        {
            "cells": int(nasa["nasa_cell"].nunique()),
            "model_value_missing_cells": {
                column: int(nasa[column].isna().sum())
                for column in ["T2M", "T2M_MIN", "T2M_MAX", "RH2M", "PRECTOTCORR"]
            },
            "provider_or_data_version": None,
            "raw_training_responses_present": False,
            "retrieval_timestamps_present": False,
            "latest_valid_date_gap_to_required_days": _days_between(
                cutoff_date, pd.Timestamp(nasa["date"].max()).date()
            ),
        }
    )
    cache_keys = _cache_payload_keys(root)
    expected_era_keys = set(era_meta["cache_key"].dropna().astype(str))
    expected_nasa_keys = set(nasa["cache_key"].dropna().astype(str))
    found_era_keys = expected_era_keys & cache_keys
    found_nasa_keys = expected_nasa_keys & cache_keys

    common_episode_requirements = {
        "variables": ["temperature_2m", "relative_humidity_2m", "precipitation"],
        "units": ["degree_C", "percent", "mm"],
        "heights": ["2_m", "2_m", "surface_accumulation"],
        "grid": "ERA5 0.25 degree; nearest cell; elevation=nan; no downscaling",
        "source_time_resolution": "hourly_UTC",
        "aggregation": "Europe/Riga local calendar day, including 23/25-hour days",
        "required_last_local_date": str(cutoff_date),
        "cutoff_rule": "issue_local_date_minus_2_calendar_days",
        "minimum_complete_history_local_days": MINIMUM_EPISODE_HISTORY_DAYS,
    }
    calendar_req = {
        "variables": ["issue_local_date"],
        "aggregation": "Europe/Riga local calendar date and day of year",
        "weather_required": False,
    }

    participants = [
        {
            "model_id": "calendar_window",
            "role": "primary",
            "requirements": calendar_req,
            "operational_source": "system_clock_and_active_field_registry",
            "status": "ready_calendar_input_but_live_registry_missing",
        },
        {
            "model_id": "C0",
            "role": "primary",
            "requirements": calendar_req,
            "operational_source": "system_clock_and_active_field_registry",
            "status": "ready_calendar_input_but_live_registry_missing",
        },
        {
            "model_id": "C1",
            "role": "diagnostic",
            "requirements": calendar_req,
            "operational_source": "system_clock_and_active_field_registry",
            "status": "ready_calendar_input_but_live_registry_missing",
        },
        {
            "model_id": "C4",
            "role": "diagnostic",
            "requirements": common_episode_requirements,
            "training_source": FROZEN_ERA5_PROFILE_ID,
            "operational_source": "not_confirmed_compatible_at_t_minus_2",
            "status": "blocked_exact_era5_t_minus_2_abstain",
        },
        {
            "model_id": "C5",
            "role": "diagnostic",
            "requirements": common_episode_requirements,
            "training_source": FROZEN_ERA5_PROFILE_ID,
            "operational_source": "not_confirmed_compatible_at_t_minus_2",
            "status": "blocked_exact_era5_t_minus_2_abstain",
        },
        {
            "model_id": "C6_weather",
            "role": "primary",
            "requirements": common_episode_requirements,
            "training_source": FROZEN_ERA5_PROFILE_ID,
            "operational_source": "not_confirmed_compatible_at_t_minus_2",
            "status": "weather_blocked_exact_c0_fallback_ready_if_registry_available",
            "weather_correction_active_today": False,
            "effective_origin_without_weather": "C0_fallback",
            "one_policy_history_across_weather_and_fallback": True,
        },
        {
            "model_id": "C6_calibration_control",
            "role": "primary_control",
            "requirements": calendar_req,
            "operational_source": "system_clock_and_active_field_registry",
            "status": "ready_calendar_input_but_live_registry_missing",
            "alpha": 0.0,
            "effective_origin": "C0",
        },
    ]

    references = [
        {
            "reference_id": "Hutton",
            "requirements": {
                **common_episode_requirements,
                "minimum_complete_history_local_days": 2,
                "rule": "two adjacent days with Tmin >= 10 C and RH >= 90% for >= 6 h/day",
            },
            "status": "blocked_exact_era5_t_minus_2",
        },
        {
            "reference_id": "Smith",
            "requirements": {
                **common_episode_requirements,
                "minimum_complete_history_local_days": 2,
                "rule": "two adjacent days with Tmin >= 10 C and RH >= 90% for >= 11 h/day",
            },
            "status": "blocked_exact_era5_t_minus_2",
        },
        {
            "reference_id": "Polyakov_current_formalisation",
            "requirements": {
                **common_episode_requirements,
                "minimum_complete_history_local_days": 10,
                "additional_input": "observed_and_available_BBCH51_with_persistent_state",
            },
            "status": "blocked_weather_and_live_bbch_registry",
        },
        {
            "reference_id": "periodic_30d",
            "requirements": calendar_req,
            "status": "ready_reference_but_not_primary_shadow_participant",
        },
    ]

    active_registry = find_active_field_registry(root, active_field_registry_path)
    if active_registry["status"] == "available":
        for entry in participants:
            if entry["status"] == "ready_calendar_input_but_live_registry_missing":
                entry["status"] = "ready_calendar_input_registry_present_not_yet_live_verified"

    source_rows = [
        {
            "source_id": "ERA5_frozen_episode_weather",
            "required": "forced ERA5 0.25 degree, UTC hourly, nearest, elevation=nan; Europe/Riga local days through t-2",
            "actual": "training-derived daily table through 2026-08-20; no operational response probed",
            "freshness": f"required {cutoff_date}; local frozen maximum 2026-08-20; documented operational lag about 5 days",
            "status": "blocked_exact_era5_t_minus_2",
            "evidence": "local snapshot hash plus official documentation; no live endpoint measurement",
        },
        {
            "source_id": "NASA_POWER_daily_auxiliary",
            "required": "not required by the selected shadow participants; C2 is outside registry",
            "actual": "frozen UTC daily table through 2026-08-31; no live response probed",
            "freshness": f"local frozen maximum is {nasa_snapshot['latest_valid_date_gap_to_required_days']} days before required cutoff",
            "status": "diagnostic_only_requires_source_version_tracking",
            "evidence": "local snapshot; historical end date is not an API latency measurement",
        },
        {
            "source_id": "active_field_registry",
            "required": "current enrolled potato fields with pseudonymous IDs and protected weather-location references",
            "actual": ", ".join(active_registry["matched_paths"]) or "not found",
            "freshness": "not measurable without a current registry",
            "status": active_registry["status"],
            "evidence": active_registry["discovery"],
        },
        {
            "source_id": "current_repository_open_meteo_client",
            "required": FROZEN_ERA5_PROFILE_ID,
            "actual": "ERA5 precipitation-only plus timezone=auto, cell_selection=land and no elevation=nan",
            "freshness": "not probed",
            "status": "incompatible_with_frozen_training_profile",
            "evidence": "source-code audit",
        },
        {
            "source_id": "future_weather_forecast_archive",
            "required": "separate immutable run snapshots with initialization, publication if known, retrieval, first-seen and valid times",
            "actual": "not present in frozen package",
            "freshness": "not measured",
            "status": "archive_not_started_requires_separate_source_bridge_study_for_scoring",
            "evidence": "workspace file audit",
        },
    ]
    fallback_by_model = {
        "C4": "abstain",
        "C5": "abstain",
        "C6_weather": "exact C0 fallback in the same policy history",
    }
    branch_rows = [
        {
            "model_id": entry["model_id"],
            "role": entry["role"],
            "required_inputs": entry["requirements"].get("variables", []),
            "status": entry["status"],
            "fallback": fallback_by_model.get(entry["model_id"], "none required"),
            "reason": (
                "documented ERA5 latency is incompatible with t-2"
                if entry["model_id"] in {"C4", "C5", "C6_weather"}
                else "calendar input is available; actual fields are missing"
            ),
        }
        for entry in participants
    ]

    return {
        "schema_version": AUDIT_VERSION,
        "audited_at_utc": audited_at,
        "issue_local_date": str(issue),
        "scheduled_slot_local": local_slot,
        "scheduled_slot_utc": utc_slot,
        "required_weather_last_local_date": str(cutoff_date),
        "audit_mode": "offline_readiness_no_live_location_request",
        "weather_branch_executable_today": False,
        "prospective_live_run_performed": False,
        "current_api_probe": {
            "performed": False,
            "reason": "no_explicit_current_active_field_registry_or_approved_non_sensitive_location",
            "measured_live_freshness": None,
        },
        "active_field_registry": active_registry,
        "source_profile": source_profile_contract(),
        "documented_availability": {
            "era5_era5t": {
                "lag": "approximately_5_days",
                "fixed_daily_publication_time": False,
                "typical_D_minus_5_availability_utc": "about_12:00",
                "evidence_kind": "official_documentation_not_endpoint_measurement",
                "compatible_with_required_t_minus_2": False,
            },
            "nasa_power": {
            "lag": "generally_about_2_days_for_GEOS_IT_meteorological_tail",
                "fixed_availability_before_05_UTC_proven": False,
                "evidence_kind": "official_documentation_not_endpoint_measurement",
                "can_replace_era5_episode_features": False,
            },
        },
        "local_snapshot_measurements": {
            "era5": era_snapshot,
            "nasa_power": nasa_snapshot,
            "interpretation": (
                "planned historical coverage only; file end dates are not measured provider latency"
            ),
        },
        "training_raw_response_audit": {
            "exact_era5_raw_responses_found": int(len(found_era_keys)),
            "expected_era5_request_keys": int(len(expected_era_keys)),
            "exact_nasa_raw_responses_found": int(len(found_nasa_keys)),
            "expected_nasa_request_keys": int(len(expected_nasa_keys)),
            "derived_tables_available": True,
            "retrieval_or_publication_time_recoverable": False,
        },
        "current_repository_client": {
            "relative_path": "src/agro_phenology/open_meteo.py",
            "sha256": sha256_file(root / "src/agro_phenology/open_meteo.py"),
            "compatible_with_frozen_profile": False,
            "incompatibilities": [
                "era5 profile restricted to precipitation only",
                "timezone=auto instead of UTC",
                "cell_selection=land instead of nearest",
                "elevation=nan absent",
                "temperature relative humidity and precipitation are not forced to one ERA5 product",
            ],
        },
        "participants": participants,
        "sources": source_rows,
        "branches": branch_rows,
        "references": references,
        "forecast_archive": {
            "status": "not_present",
            "expected_files_missing": [
                "frozen_external/archived_weather_forecasts.parquet",
                "frozen_external/weather_publication_log.csv",
            ],
            "automatic_use_in_frozen_past_only_features": False,
            "alternative_products_status": "requires_separate_source_bridge_study",
            "run_initialization_is_provider_publication": False,
        },
        "today_conclusion": {
            "exact_era5_t_minus_2_available_by_documented_latency": False,
            "C4_C5": "abstain",
            "C6_weather": "exact_C0_fallback_only_if_active_field_registry_is_available",
            "calendar_participants": "input_compatible_but_live_execution_blocked_without_active_field_registry",
            "weather_branch_prospectively_tested": False,
        },
        "official_sources": OFFICIAL_SOURCE_PAGES,
    }


__all__ = [
    "AUDIT_VERSION",
    "DECISION_TIMEZONE",
    "OFFICIAL_SOURCE_PAGES",
    "SCHEDULED_LOCAL_TIME",
    "build_source_compatibility_manifest",
    "find_active_field_registry",
    "sha256_file",
]
