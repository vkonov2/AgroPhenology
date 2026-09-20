"""CLI for the first potato late-blight early-warning research cycle."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd

from .early_warning_core import (
    add_daily_features,
    add_polyakov_baseline,
    build_daily_decisions,
    build_field_seasons,
    load_snapshot_module,
    prepare_potato_visits,
    sha256_file,
)
from .early_warning_models import _model_rows, _year_mask, run_experiments, tune_c4


REPO_ROOT = Path(__file__).resolve().parents[2]


def _json_default(value: Any):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_source_snapshot(path: Path) -> dict[str, Any]:
    """Persist the exact uncommitted research code needed to reproduce a run."""
    candidates = [
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "requirements-early-warning.txt",
        REPO_ROOT / "src/agro_phenology/early_warning_core.py",
        REPO_ROOT / "src/agro_phenology/early_warning_models.py",
        REPO_ROOT / "src/agro_phenology/early_warning_pipeline.py",
        REPO_ROOT / "src/agro_phenology/early_warning_reporting.py",
        REPO_ROOT / "tests/test_early_warning_core.py",
        REPO_ROOT / "tests/test_early_warning_policy.py",
        REPO_ROOT / "tests/test_early_warning_reporting.py",
    ]
    missing = [str(item.relative_to(REPO_ROOT)) for item in candidates if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Research source snapshot is incomplete: {missing}")
    files = []
    for source in candidates:
        files.append(
            {
                "relative_path": str(source.relative_to(REPO_ROOT)),
                "sha256": sha256_file(source),
                "utf8_content": source.read_text(encoding="utf-8"),
            }
        )
    payload = {
        "format": "agro_phenology_research_source_snapshot_v1",
        "restore_note": "write each utf8_content to relative_path under a clean checkout of git_revision",
        "git_revision": _git_text("rev-parse", "HEAD"),
        "files": files,
    }
    _write_json(path, payload)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "files": [{"relative_path": item["relative_path"], "sha256": item["sha256"]} for item in files],
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def _git_text(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else f"unavailable: {completed.stderr.strip()}"


def _environment_versions() -> dict:
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "pyarrow",
        "catboost",
        "optuna",
        "joblib",
        "threadpoolctl",
        "pytest",
    ]
    versions: dict[str, str] = {}
    for package in packages:
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


def _manifest_hashes(path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})\s+(.+)", line.strip())
        if match:
            expected[match.group(2)] = match.group(1)
    return expected


def _format_details(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        details: dict[str, Any] = {
            "format": "parquet",
            "rows": parquet.metadata.num_rows,
            "columns": parquet.metadata.num_columns,
            "row_groups": parquet.metadata.num_row_groups,
            "created_by": parquet.metadata.created_by,
            "schema": {field.name: str(field.type) for field in schema},
        }
        if "date" in schema.names:
            dates = pd.read_parquet(path, columns=["date"])["date"]
            details["date_min"] = str(pd.to_datetime(dates).min().date())
            details["date_max"] = str(pd.to_datetime(dates).max().date())
        for cell in ("weather_cell", "nasa_cell"):
            if cell in schema.names:
                values = pd.read_parquet(path, columns=[cell])[cell]
                details[f"unique_{cell}"] = int(values.nunique())
        return details
    if suffix == ".csv":
        frame = pd.read_csv(path, low_memory=False)
        details = {
            "format": "csv",
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "schema": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        }
        for column in ("observation_date", "date"):
            if column in frame:
                dates = pd.to_datetime(frame[column], errors="coerce")
                details["date_min"] = str(dates.min().date()) if dates.notna().any() else None
                details["date_max"] = str(dates.max().date()) if dates.notna().any() else None
                details["invalid_dates"] = int(dates.isna().sum())
                break
        if "observation_id" in frame:
            details["duplicate_observation_ids"] = int(frame["observation_id"].duplicated().sum())
        return details
    if suffix in {".tif", ".tiff"}:
        try:
            from PIL import Image

            with Image.open(path) as image:
                return {"format": "geotiff", "width": image.width, "height": image.height, "mode": image.mode}
        except Exception as error:
            return {"format": "geotiff", "inspection_error": f"{type(error).__name__}: {error}"}
    return {"format": suffix.removeprefix(".") or "unknown"}


def _inventory_record(
    path: Path,
    *,
    role: str,
    provenance: str,
    source_kind: str,
    expected_sha256: str | None = None,
    inspect_format: bool = True,
    required_for_first_cycle: bool = False,
) -> dict:
    present = path.is_file()
    record: dict[str, Any] = {
        "path": str(path),
        "relative_path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else None,
        "role": role,
        "provenance": provenance,
        "source_kind": source_kind,
        "status": "present" if present else "missing",
        "present": present,
        "expected_sha256": expected_sha256,
        "required_for_first_cycle": required_for_first_cycle,
    }
    if present:
        actual = sha256_file(path)
        record.update(
            {
                "bytes": path.stat().st_size,
                "sha256": actual,
                "hash_matches_manifest": actual == expected_sha256 if expected_sha256 else None,
            }
        )
        if inspect_format:
            record["details"] = _format_details(path)
    return record


def build_data_inventory(contract: dict) -> dict:
    inputs = contract["inputs"]
    package = _resolve(Path(inputs["conservative_v2_csv"]).parents[1])
    checksum_path = _resolve(inputs["sha256_manifest"])
    expected = _manifest_hashes(checksum_path)
    records: list[dict] = []
    required_manifest_paths = {
        "data/vaad_observations_geocoded_conservative_v2.csv",
        "frozen_external/era5_potato_daily.parquet",
        "frozen_external/nasa_coordinate_mapping.parquet",
        "frozen_external/nasa_daily.parquet",
    }
    for relative, digest in expected.items():
        path = package / relative
        role = "primary_observations" if relative.startswith("data/") else "frozen_external_input"
        records.append(
            _inventory_record(
                path,
                role=role,
                provenance="Valery frozen reproduction package",
                source_kind="source_input" if role == "primary_observations" else "frozen_derived_external",
                expected_sha256=digest,
                required_for_first_cycle=relative in required_manifest_paths,
            )
        )
    records.append(
        _inventory_record(
            _resolve(inputs["original_vaad_csv"]),
            role="provenance_and_label_sensitivity_only",
            provenance="original local VAAD extraction; never merged with conservative v2",
            source_kind="source_input_distinct_version",
            expected_sha256="34a4c64f047d829374c99e848ac99d1b93dd8a13acade09d909dbf1944d26ecf",
            required_for_first_cycle=False,
        )
    )
    for key, role in (
        ("snapshot_vaad_source", "audited_label_adapter"),
        ("snapshot_late_blight_source", "audited_rule_adapter"),
    ):
        records.append(
            _inventory_record(
                _resolve(inputs[key]),
                role=role,
                provenance="source snapshot shipped in Valery reproduction package",
                source_kind="source_code",
                inspect_format=False,
                required_for_first_cycle=True,
            )
        )

    validation_path = package / "reference_results/artifact_validation.json"
    artifact_expected: dict[str, str] = {}
    if validation_path.is_file():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        artifact_expected = {
            str(name): str(digest)
            for name, digest in validation.get("artifact_sha256", {}).items()
        }
    control_paths = sorted((package / "reference_results").glob("*")) + sorted(
        (package / "reference_docs").glob("*")
    )
    missing_calculation = package / "reference_results/calculation_results.csv"
    if missing_calculation not in control_paths:
        control_paths.append(missing_calculation)
    for path in control_paths:
        if not path.is_file() and path != missing_calculation:
            continue
        role = (
            "historical_control_manifest"
            if path.name == "artifact_validation.json"
            else "missing_historical_row_level_control"
            if path == missing_calculation and not path.is_file()
            else "historical_reference_document"
            if path.parent.name == "reference_docs"
            else "historical_control_artifact"
        )
        records.append(
            _inventory_record(
                path,
                role=role,
                provenance="Valery reproduction package",
                source_kind="control_artifact",
                expected_sha256=artifact_expected.get(path.name),
                inspect_format=path.suffix in {".csv", ".parquet"},
            )
        )

    for path, role, required in (
        (package / "README_RUN.md", "reproduction_instructions", True),
        (checksum_path, "input_hash_manifest", True),
        (package / "setup.ps1", "reproduction_launcher", False),
        (package / "run_pipeline.ps1", "reproduction_launcher", False),
        (package / "verify_reproduction.py", "reproduction_verifier", False),
    ):
        records.append(
            _inventory_record(
                path,
                role=role,
                provenance="Valery reproduction package",
                source_kind="source_code_or_manifest",
                inspect_format=False,
                required_for_first_cycle=required,
            )
        )

    inventoried_paths = {Path(record["path"]) for record in records}
    project_files = sorted(
        path
        for path in (package / "project").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    for path in project_files:
        if path in inventoried_paths:
            continue
        if "tests" in path.parts:
            role = "snapshot_test_source"
        elif "src" in path.parts:
            role = "snapshot_pipeline_source"
        else:
            role = "snapshot_project_config_or_documentation"
        records.append(
            _inventory_record(
                path,
                role=role,
                provenance="source snapshot shipped in Valery reproduction package",
                source_kind="source_code",
                inspect_format=False,
            )
        )
    exact_missing = [
        package / "frozen_external/era5_potato_hourly.parquet",
        package / "frozen_external/archived_weather_forecasts.parquet",
        package / "frozen_external/weather_publication_log.csv",
        package / "reference_results/calculation_results.csv",
        package / "reference_docs/hutton_criteria_pipeline_ru.pdf",
        package / "reference_docs/fitoftoroz_kartofelya_pipeline_polyakov.pdf",
        package / "reference_docs/prognoz-razvitiya-vrediteley-i-bolezney-selskokhozyaystvenny.pdf",
    ]
    failed_required = [
        record["relative_path"] or record["path"]
        for record in records
        if record["required_for_first_cycle"]
        and (not record["present"] or record.get("hash_matches_manifest") is False)
    ]
    artifact_validation_checks = []
    artifact_locations = {
        "calculation_results.csv": package / "reference_results/calculation_results.csv",
        "report_ru.pdf": package / "reference_results/report_ru.pdf",
        "report_ru.html": package / "reference_results/report_ru.html",
        "model_metrics.csv": package / "reference_results/model_metrics.csv",
        "latvia_observation_area_glo90.tif": package / "frozen_external/latvia_observation_area_glo90.tif",
    }
    for name, digest in artifact_expected.items():
        path = artifact_locations.get(name, package / "reference_results" / name)
        present = path.is_file()
        actual = sha256_file(path) if present else None
        artifact_validation_checks.append(
            {
                "artifact": name,
                "present": present,
                "expected_sha256": digest,
                "actual_sha256": actual,
                "matches_saved_validation": bool(present and actual == digest),
            }
        )
    return {
        "created_at_utc": _utc_now(),
        "package_path": str(package),
        "package_git_status": "untracked" if "?? docs/extra/" in _git_text("status", "--short") else "tracked_or_partial",
        "records": records,
        "package_files_inventoried": len(records),
        "saved_artifact_validation_checks": artifact_validation_checks,
        "required_input_failures": failed_required,
        "exact_missing_files_or_capabilities": [str(path) for path in exact_missing if not path.is_file()],
        "interpretation": {
            "frozen_daily_weather": "derived daily retrospective data, not archived forecast issues",
            "historical_reference_results": "saved controls, not a replacement for row-level inputs or a fresh run",
            "original_and_v2": "related versions; never pooled as independent observations",
        },
    }


def _label_evidence(path: Path, snapshot_source: Path) -> tuple[dict, pd.Series]:
    snapshot = load_snapshot_module(snapshot_source, f"agro_label_audit_{path.stem}")
    header = pd.read_csv(path, nrows=0).columns
    coordinate_columns = (
        ["final_field_uid", "final_latitude", "final_longitude", "final_coordinate_class"]
        if "final_field_uid" in header
        else ["field_uid", "latitude", "longitude"]
    )
    columns = [
        "observation_id",
        "observation_date",
        "crop_code",
        "crop_stage_raw",
        "growth_stage_code",
        "detected_organisms",
        "explicit_no_harmful_organisms",
        "organisms_raw",
    ] + coordinate_columns
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    parsed: list[list[dict] | None] = []
    invalid = 0
    for raw_value in frame["detected_organisms"]:
        try:
            value = json.loads(raw_value)
            if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
                raise ValueError("invalid organism list")
        except (TypeError, ValueError, json.JSONDecodeError):
            value = None
            invalid += 1
        parsed.append(value)
    generic = frame["explicit_no_harmful_organisms"].map(snapshot.truth) | frame["organisms_raw"].fillna("").str.contains(
        r"(?:kaitīgie\s+organismi|kaitīgo\s+organismu\s+klātbūtne|slimības)\s+(?:nav|netika)\s+konstat",
        case=False,
        regex=True,
    )
    labels = [
        snapshot.label_organism(items, 640, bool(no_harmful))[0] if items is not None else "invalid_json"
        for items, no_harmful in zip(parsed, generic)
    ]
    statuses = pd.Series(labels, index=frame["observation_id"].astype(str), name="label_status")
    listed = pd.Series(
        [bool(items and any(item.get("organism_id") == 640 for item in items)) for items in parsed],
        index=frame.index,
    )
    potato = pd.to_numeric(frame["crop_code"], errors="coerce").eq(166) | listed
    selected = pd.Series(labels, index=frame.index)[potato]
    candidate = frame.loc[potato].copy()
    candidate["label_status"] = selected
    dates = pd.to_datetime(candidate["observation_date"], errors="coerce")
    stages = pd.to_numeric(candidate["growth_stage_code"], errors="coerce")
    storage = stages.eq(99) | candidate["crop_stage_raw"].fillna("").str.contains(
        r"noliktav|uzglabāšanas", case=False, regex=True
    )
    if "final_field_uid" in candidate:
        geo = (
            candidate["final_coordinate_class"].isin(["A_direct", "B_new_subtraction"])
            & candidate["final_field_uid"].notna()
            & candidate["final_latitude"].between(55, 59)
            & candidate["final_longitude"].between(20, 29)
        )
        geo_basis = "conservative_v2_final_A_or_B"
    else:
        geo = (
            candidate["field_uid"].notna()
            & candidate["latitude"].between(55, 59)
            & candidate["longitude"].between(20, 29)
        )
        geo_basis = "original_existing_or_recovered_coordinates_without_v2_confidence_class"
    eligible = candidate[
        geo & dates.dt.month.between(5, 9) & ~stages.ge(97) & ~storage & dates.notna()
    ].copy()
    eligible["_observation_date"] = dates.loc[eligible.index].dt.normalize()
    eligible["_crop_code"] = pd.to_numeric(eligible["crop_code"], errors="coerce")
    eligible["_field_uid"] = (
        eligible["final_field_uid"] if "final_field_uid" in eligible else eligible["field_uid"]
    )
    deduplicated_statuses: list[str] = []
    for _, group in eligible.groupby(
        ["_field_uid", "_crop_code", "_observation_date"], sort=False, dropna=False
    ):
        group_statuses = set(group["label_status"])
        if "conflict" in group_statuses or (
            "positive" in group_statuses
            and group_statuses.intersection({"explicit_target_absent", "generic_absent"})
        ):
            status = "conflict"
        else:
            status = next(
                (
                    candidate_status
                    for candidate_status in (
                        "positive",
                        "explicit_target_absent",
                        "generic_absent",
                        "invalid_json",
                        "unassessed",
                    )
                    if candidate_status in group_statuses
                ),
                "unassessed",
            )
        deduplicated_statuses.append(status)
    deduplicated_counts = pd.Series(deduplicated_statuses, dtype="object").value_counts()
    return (
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "all_rows": int(len(frame)),
            "potato_or_listed_rows": int(potato.sum()),
            "eligible_source_rows_before_daily_deduplication": int(len(eligible)),
            "eligible_geo_basis": geo_basis,
            "eligible_label_counts": {
                str(key): int(value) for key, value in eligible["label_status"].value_counts().items()
            },
            "eligible_rows_after_daily_deduplication": int(len(deduplicated_statuses)),
            "eligible_label_counts_after_daily_deduplication": {
                str(key): int(value) for key, value in deduplicated_counts.items()
            },
            "invalid_json_all_rows": int(invalid),
            "potato_label_counts": {str(key): int(value) for key, value in selected.value_counts().items()},
        },
        statuses,
    )


def compare_input_versions(contract: dict) -> dict:
    snapshot = _resolve(contract["inputs"]["snapshot_vaad_source"])
    v2_summary, v2 = _label_evidence(_resolve(contract["inputs"]["conservative_v2_csv"]), snapshot)
    original_path = _resolve(contract["inputs"]["original_vaad_csv"])
    if not original_path.is_file():
        return {
            "status": "not_run_missing_original_vaad",
            "original": {"path": str(original_path), "status": "missing"},
            "conservative_v2": v2_summary,
            "target_specific_negative_evidence": {
                "population": "eligible_potato_rows_after_daily_deduplication",
                "original_explicit_target_absent": None,
                "v2_explicit_target_absent": int(
                    v2_summary["eligible_label_counts_after_daily_deduplication"].get(
                        "explicit_target_absent", 0
                    )
                ),
            },
            "usage": "comparison not run; original is optional provenance/sensitivity input",
        }
    original_summary, original = _label_evidence(original_path, snapshot)
    joined = pd.concat([original.rename("original"), v2.rename("v2")], axis=1, join="inner")
    differences = joined["original"].ne(joined["v2"])
    return {
        "status": "complete",
        "original": original_summary,
        "conservative_v2": v2_summary,
        "shared_observation_ids": int(len(joined)),
        "label_status_differences_on_shared_ids": int(differences.sum()),
        "original_only_ids": int(len(original.index.difference(v2.index))),
        "v2_only_ids": int(len(v2.index.difference(original.index))),
        "target_specific_negative_evidence": {
            "population": "eligible_potato_rows_after_daily_deduplication",
            "original_explicit_target_absent": int(
                original_summary["eligible_label_counts_after_daily_deduplication"].get(
                    "explicit_target_absent", 0
                )
            ),
            "v2_explicit_target_absent": int(
                v2_summary["eligible_label_counts_after_daily_deduplication"].get(
                    "explicit_target_absent", 0
                )
            ),
            "all_potato_or_listed_rows_original": int((original == "explicit_target_absent").sum()),
            "all_potato_or_listed_rows_v2": int((v2 == "explicit_target_absent").sum()),
        },
        "usage": "provenance/sensitivity only; versions were not concatenated",
    }


def build_split_manifest(decisions: pd.DataFrame, seasons: pd.DataFrame, contract: dict, smoke: bool) -> dict:
    folds = contract["rolling_origin_folds"][:1] if smoke else contract["rolling_origin_folds"]
    rows: list[dict] = []
    for fold in folds:
        for role, years_key in (
            ("train", "train_years"),
            ("validation", "validation_years"),
            ("test", "test_years"),
        ):
            years = fold[years_key]
            daily = decisions[_year_mask(decisions, years)]
            registry = seasons[_year_mask(seasons, years)]
            model_rows = _model_rows(decisions, years)
            rows.append(
                {
                    "fold_id": fold["id"],
                    "role": role,
                    "years": years,
                    "incomplete": bool(fold.get("incomplete", False) and role == "test"),
                    "decision_rows": int(len(daily)),
                    "service_field_days": int(daily["service_active"].sum()),
                    "paired_candidate_field_days": int(
                        (daily["service_active"] & daily["candidate_comparison_complete"]).sum()
                    ),
                    "observable_model_rows": int(len(model_rows)),
                    "field_seasons": int(len(registry)),
                    "all_first_events": int(registry["first_recorded_event_date"].notna().sum()),
                    "warnable_first_events": int(registry["warnable_first_event"].sum()),
                    "positive_at_first_visit": int(registry["positive_at_first_visit"].sum()),
                    "target_class_counts": {
                        str(key): int(value) for key, value in daily["target_class"].value_counts().items()
                    },
                }
            )
    return {
        "strategy": "expanding_window_with_two_year_validation_and_one_year_test",
        "rows": rows,
        "note": "2023-2025 and partial 2026 were previously studied and are retrospective tests, not untouched holdouts",
    }


def run_invariants(
    visits: pd.DataFrame,
    seasons: pd.DataFrame,
    decisions: pd.DataFrame,
    inventory: dict,
) -> dict:
    checks: list[dict] = []

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks.append({"name": name, "status": "passed" if condition else "failed", "evidence": evidence})

    check("required_inputs_and_hashes", not inventory["required_input_failures"], inventory["required_input_failures"])
    check(
        "historical_visit_landmarks",
        len(visits) == 1706 and visits["field_season"].nunique() == 488 and visits["label_status"].eq("positive").sum() == 788,
        {
            "visits": len(visits),
            "field_seasons": visits["field_season"].nunique(),
            "positive_visits": int(visits["label_status"].eq("positive").sum()),
        },
    )
    check("one_registry_row_per_field_season", seasons["field_season"].is_unique, len(seasons))
    check(
        "single_first_event_per_registry_row",
        seasons.loc[seasons["first_recorded_event_date"].notna(), "field_season"].is_unique,
        int(seasons["first_recorded_event_date"].notna().sum()),
    )
    check(
        "decision_cutoffs_are_strictly_past",
        bool(
            decisions["feature_cutoff_nasa_date"].lt(decisions["issue_date"]).all()
            and decisions["feature_cutoff_era_common_date"].lt(decisions["issue_date"]).all()
            and decisions["feature_cutoff_era_episode_date"].lt(decisions["issue_date"]).all()
        ),
        "all weather cutoffs precede issue_date",
    )
    check(
        "common_two_day_cutoff",
        bool(
            decisions["feature_cutoff_nasa_date"].eq(decisions["issue_date"] - pd.Timedelta(days=2)).all()
            and decisions["feature_cutoff_era_common_date"].eq(decisions["issue_date"] - pd.Timedelta(days=2)).all()
            and decisions["feature_cutoff_era_episode_date"].eq(decisions["issue_date"] - pd.Timedelta(days=2)).all()
        ),
        "NASA, ERA common and episode features end at issue_date-2",
    )
    check(
        "unknown_is_not_model_class",
        "unknown" not in {"no_record_in_horizon", "imminent", "actionable"},
        decisions["target_class"].value_counts().to_dict(),
    )
    available = decisions["first_recorded_event_date"] + pd.Timedelta(days=1)
    should_stop = decisions["first_recorded_event_date"].notna() & decisions["issue_date"].ge(available)
    check(
        "service_stops_after_record_available",
        bool((~decisions.loc[should_stop, "service_active"]).all()),
        int(should_stop.sum()),
    )
    failed = [item["name"] for item in checks if item["status"] != "passed"]
    return {"status": "passed" if not failed else "failed", "checks": checks, "failed_checks": failed}


def _write_experiment_outputs(run_dir: Path, outputs: dict[str, pd.DataFrame]) -> None:
    parquet_names = {"predictions", "alarm_states", "event_hits"}
    for name, frame in outputs.items():
        if name in parquet_names:
            frame.to_parquet(run_dir / f"{name}.parquet", index=False)
        else:
            frame.to_csv(run_dir / f"{name}.csv", index=False)
    states = outputs["alarm_states"]
    notifications = states[states["message_issued"] | states["suppressed_repeat"]].copy()
    notification_columns = [
        "field_season",
        "season",
        "issued_at",
        "issue_date",
        "model_code",
        "model_version",
        "evaluation_scope",
        "score",
        "score_status",
        "policy_version",
        "policy_threshold",
        "message_issued",
        "action_reason",
        "active_from",
        "active_through",
        "forecast_window_start",
        "forecast_window_end",
        "suppressed_repeat",
    ]
    notifications[notification_columns].to_parquet(run_dir / "notification_log.parquet", index=False)


def _output_hashes(run_dir: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "execution_manifest.json":
            continue
        relative = str(path.relative_to(run_dir))
        result[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def _run_tests(smoke: bool) -> dict:
    commands: list[tuple[list[str], Path, dict[str, str]]] = [
        ([
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_early_warning_core.py",
            "tests/test_early_warning_policy.py",
            "tests/test_early_warning_reporting.py",
        ], REPO_ROOT, {})
    ]
    if not smoke:
        commands.extend(
            [
                ([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "not integration", "tests"], REPO_ROOT, {}),
                ([
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "-m",
                    "not integration",
                    "tests",
                ], REPO_ROOT / "docs/extra/vaad_pipeline_repro_20260905/project", {
                    "PYTHONPATH": str(REPO_ROOT / "docs/extra/vaad_pipeline_repro_20260905/project/src")
                }),
            ]
        )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    results: list[dict] = []
    for command, working_directory, additions in commands:
        command_environment = dict(environment)
        command_environment.update(additions)
        completed = subprocess.run(
            command,
            cwd=working_directory,
            text=True,
            capture_output=True,
            check=False,
            env=command_environment,
        )
        results.append(
            {
                "command": command,
                "cwd": str(working_directory),
                "returncode": completed.returncode,
                "status": "passed" if completed.returncode == 0 else "failed",
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )
    return {
        "status": "passed" if all(item["returncode"] == 0 for item in results) else "failed",
        "commands": results,
    }


def _historical_quality_context(inventory: dict) -> dict:
    package = Path(inventory["package_path"])
    validation_path = package / "reference_results/artifact_validation.json"
    selected_path = package / "reference_results/selected_models.csv"
    context: dict[str, Any] = {
        "status": "saved_historical_controls_not_refit_in_the_new_task",
        "comparability": "old seven-day visit-level task includes repeat positives; metrics are not comparable to first-event warning",
    }
    if validation_path.is_file():
        saved = json.loads(validation_path.read_text(encoding="utf-8"))
        potato = next(
            (row for row in saved.get("test_metrics_recomputed_from_export", []) if row.get("organism_id") == 640),
            None,
        )
        context.update(
            {
                "saved_artifact_validation_status": saved.get("status"),
                "saved_artifact_checked_at_utc": saved.get("checked_at_utc"),
                "old_potato_selected_boosting_test": potato,
                "missing_calculation_results_csv": not (package / "reference_results/calculation_results.csv").is_file(),
                "saved_test_suite_claim": saved.get("test_suite"),
            }
        )
    if selected_path.is_file():
        selected = pd.read_csv(selected_path)
        context["saved_selected_models"] = selected.to_dict(orient="records")
    context["old_calendar_test_AP_from_frozen_report"] = 0.6702
    context["old_boosting_test_AP_from_frozen_export"] = 0.6398826019668467
    return context


def _load_reporting_inputs(run_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    artifacts: dict[str, pd.DataFrame] = {}
    for name in (
        "event_metrics",
        "burden_metrics",
        "daily_diagnostics",
        "policy_selection",
        "budget_grid_metrics",
        "optuna_trials",
        "optuna_seed_checks",
    ):
        artifacts[name] = pd.read_csv(run_dir / f"{name}.csv")
    for name in ("event_hits", "alarm_states"):
        artifacts[name] = pd.read_parquet(run_dir / f"{name}.parquet")
    seasons = pd.read_parquet(run_dir / "field_seasons.parquet")
    audits = {
        "data_inventory": json.loads((run_dir / "data_inventory.json").read_text(encoding="utf-8")),
        "data_audit": json.loads((run_dir / "data_audit.json").read_text(encoding="utf-8")),
        "input_version_comparison": json.loads((run_dir / "input_version_comparison.json").read_text(encoding="utf-8")),
        "test_results": json.loads((run_dir / "test_results.json").read_text(encoding="utf-8")),
    }
    return artifacts, seasons, audits


def generate_report_from_saved(
    run_dir: Path, output_dir: Path | None = None
) -> dict[str, pd.DataFrame]:
    """Generate aggregates from a saved run, optionally into a separate directory."""
    from .early_warning_reporting import build_reporting_artifacts, write_report_ru

    destination = run_dir if output_dir is None else output_dir
    destination.mkdir(parents=True, exist_ok=True)
    artifacts, seasons, audits = _load_reporting_inputs(run_dir)
    contract = json.loads((run_dir / "evaluation_contract.json").read_text(encoding="utf-8"))
    reporting = build_reporting_artifacts(
        artifacts,
        seasons,
        seed=int(contract["random_seed"]),
        n_bootstrap=2000,
    )
    for name, frame in reporting.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(destination / f"{name}.csv", index=False)
    write_report_ru(
        destination / "report_ru.md",
        pooled_summary=reporting.get("pooled_summary"),
        paired_comparisons=reporting.get("paired_comparisons"),
        delay_sensitivity=reporting.get("registration_delay_sensitivity"),
        audits=audits,
        contract=contract,
        daily_diagnostics=artifacts["daily_diagnostics"],
        policy_selection=artifacts["policy_selection"],
        budget_grid_pooled=reporting.get("budget_grid_pooled"),
        optuna_trials=artifacts["optuna_trials"],
        optuna_seed_checks=artifacts["optuna_seed_checks"],
    )
    return reporting


def run_cycle(contract_path: Path, run_id: str, smoke: bool) -> Path:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    output_root = _resolve(contract["output_root"])
    run_dir = output_root / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty and will not be overwritten: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "mode": "smoke" if smoke else "full",
        "status": "running",
        "started_at_utc": started,
        "command": [sys.executable, *sys.argv],
        "git": {
            "revision": _git_text("rev-parse", "HEAD"),
            "branch": _git_text("branch", "--show-current"),
            "status_short": _git_text("status", "--short").splitlines(),
        },
        "environment": _environment_versions(),
        "thread_limits": {"catboost_thread_count": 2, "optuna_n_jobs": 1},
        "random_seed": int(contract["random_seed"]),
        "fold_definitions": contract["rolling_origin_folds"][:1] if smoke else contract["rolling_origin_folds"],
    }
    manifest["source_snapshot"] = _write_source_snapshot(run_dir / "source_snapshot.json")
    _write_json(run_dir / "execution_manifest.json", manifest)
    try:
        _write_json(run_dir / "evaluation_contract.json", contract)
        protocol_source = contract_path.with_name("protocol.md")
        if protocol_source.is_file():
            shutil.copy2(protocol_source, run_dir / "protocol.md")
        inventory = build_data_inventory(contract)
        _write_json(run_dir / "data_inventory.json", inventory)
        if inventory["required_input_failures"]:
            raise RuntimeError(f"Required inputs failed inventory: {inventory['required_input_failures']}")
        input_comparison = compare_input_versions(contract)
        _write_json(run_dir / "input_version_comparison.json", input_comparison)

        inputs = contract["inputs"]
        visits, visit_audit = prepare_potato_visits(
            _resolve(inputs["conservative_v2_csv"]), _resolve(inputs["snapshot_vaad_source"])
        )
        seasons = build_field_seasons(visits, contract["global_snapshot_date"])
        decisions = build_daily_decisions(
            seasons,
            timezone=contract["timezone"],
            issue_time=contract["daily_issue_time"],
            minimum_lead=int(contract["timeliness_window_days"]["minimum"]),
            maximum_lead=int(contract["timeliness_window_days"]["maximum"]),
        )
        frozen = _resolve(inputs["frozen_external_dir"])
        decisions, weather_audit = add_daily_features(
            decisions,
            frozen / "nasa_daily.parquet",
            frozen / "nasa_coordinate_mapping.parquet",
            frozen / "era5_potato_daily.parquet",
        )
        decisions, polyakov_audit = add_polyakov_baseline(
            decisions,
            seasons,
            frozen / "era5_potato_daily.parquet",
            _resolve(inputs["snapshot_late_blight_source"]),
        )
        invariants = run_invariants(visits, seasons, decisions, inventory)
        if invariants["status"] != "passed":
            raise AssertionError(f"Pipeline invariants failed: {invariants['failed_checks']}")
        data_audit = {
            "visit_preparation": visit_audit,
            "field_seasons": {
                "rows": int(len(seasons)),
                "entry_categories": {
                    str(key): int(value) for key, value in seasons["entry_category"].value_counts().items()
                },
                "first_events": int(seasons["first_recorded_event_date"].notna().sum()),
                "warnable_first_events": int(seasons["warnable_first_event"].sum()),
                "positive_at_first_visit": int(seasons["positive_at_first_visit"].sum()),
                "no_event_observable_horizon": int(
                    seasons["entry_category"].eq("no_event_with_observable_horizon").sum()
                ),
            },
            "daily_decisions": {
                "rows": int(len(decisions)),
                "service_field_days": int(decisions["service_active"].sum()),
                "target_counts": {
                    str(key): int(value) for key, value in decisions["target_class"].value_counts().items()
                },
                "observable_rows": int(decisions["target_observable"].sum()),
            },
            "weather": weather_audit,
            "polyakov": polyakov_audit,
            "historical_quality_context": _historical_quality_context(inventory),
            "invariants": invariants,
        }
        _write_json(run_dir / "data_audit.json", data_audit)
        visits.to_parquet(run_dir / "potato_visits.parquet", index=False)
        seasons.to_parquet(run_dir / "field_seasons.parquet", index=False)
        seasons.loc[seasons["first_recorded_event_date"].notna()].to_parquet(
            run_dir / "events.parquet", index=False
        )
        decisions.to_parquet(run_dir / "daily_decisions.parquet", index=False)
        split_manifest = build_split_manifest(decisions, seasons, contract, smoke)
        _write_json(run_dir / "split_manifest.json", split_manifest)

        outputs = run_experiments(decisions, seasons, contract, run_dir, smoke=smoke)
        _write_experiment_outputs(run_dir, outputs)
        tests = _run_tests(smoke)
        _write_json(run_dir / "test_results.json", tests)
        if tests["status"] != "passed":
            raise RuntimeError("One or more recorded test commands failed")
        generate_report_from_saved(run_dir)
        manifest.update(
            {
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "input_hashes": {
                    record["relative_path"] or record["path"]: record.get("sha256")
                    for record in inventory["records"]
                    if record.get("present") and record["source_kind"] in {"source_input", "source_input_distinct_version", "frozen_derived_external"}
                },
                "output_hashes": _output_hashes(run_dir),
                "tests_status": tests["status"],
                "scientific_status": "retrospective_registration_proxy_only",
            }
        )
        _write_json(run_dir / "execution_manifest.json", manifest)
        return run_dir
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "partial_output_hashes": _output_hashes(run_dir),
            }
        )
        _write_json(run_dir / "execution_manifest.json", manifest)
        raise


def resume_optuna(source_run_dir: Path, additional_trials_per_fold: int) -> Path:
    """Continue saved studies in a new sibling directory without changing the run."""
    if additional_trials_per_fold < 0:
        raise ValueError("additional_trials_per_fold must be non-negative")
    contract = json.loads((source_run_dir / "evaluation_contract.json").read_text(encoding="utf-8"))
    decisions = pd.read_parquet(source_run_dir / "daily_decisions.parquet")
    seasons = pd.read_parquet(source_run_dir / "field_seasons.parquet")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    extension = source_run_dir.parent / f"{source_run_dir.name}__optuna_resume_{stamp}"
    extension.mkdir(parents=True, exist_ok=False)
    rows: list[pd.DataFrame] = []
    summaries: list[dict] = []
    available_folds = [
        fold
        for fold in contract["rolling_origin_folds"]
        if (source_run_dir / "optuna" / f"{fold['id']}.sqlite3").is_file()
    ]
    if not available_folds:
        raise FileNotFoundError(f"No saved Optuna studies under {source_run_dir / 'optuna'}")
    for fold_index, fold in enumerate(available_folds):
        fold_id = fold["id"]
        source_db = source_run_dir / "optuna" / f"{fold_id}.sqlite3"
        target_db = extension / f"{fold_id}.sqlite3"
        shutil.copy2(source_db, target_db)
        import optuna

        existing = optuna.load_study(study_name=f"C4_{fold_id}", storage=f"sqlite:///{target_db.resolve()}")
        before = len(existing.trials)
        target = before + additional_trials_per_fold
        train = _model_rows(decisions, fold["train_years"])
        validation_service = decisions[_year_mask(decisions, fold["validation_years"])].copy()
        validation_seasons = seasons[_year_mask(seasons, fold["validation_years"])].copy()
        _, study = tune_c4(
            train,
            validation_service,
            validation_seasons,
            contract,
            target_db,
            f"C4_{fold_id}",
            target,
            int(contract["random_seed"]) + 1000 + fold_index,
        )
        frame = pd.DataFrame(
            [
                {
                    "fold_id": fold_id,
                    "number": trial.number,
                    "state": trial.state.name,
                    "value": trial.value,
                    "params": json.dumps(trial.params, sort_keys=True),
                    "user_attrs": json.dumps(trial.user_attrs, sort_keys=True),
                    "datetime_start": trial.datetime_start,
                    "datetime_complete": trial.datetime_complete,
                }
                for trial in study.trials
            ]
        )
        rows.append(frame)
        summaries.append(
            {
                "fold_id": fold_id,
                "trials_before": before,
                "trials_after": len(study.trials),
                "added": len(study.trials) - before,
                "best_value": study.best_value,
                "best_params": study.best_params,
            }
        )
    pd.concat(rows, ignore_index=True).to_csv(extension / "optuna_trials.csv", index=False)
    _write_json(
        extension / "resume_manifest.json",
        {
            "created_at_utc": _utc_now(),
            "source_run_dir": str(source_run_dir),
            "additional_trials_per_fold": additional_trials_per_fold,
            "folds": summaries,
            "status": "study_extended",
            "test_metrics_status": "not_recomputed; create a new full run before comparing test results",
            "output_hashes": _output_hashes(extension),
        },
    )
    return extension


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a new immutable research cycle")
    run.add_argument("--contract", required=True, type=Path)
    run.add_argument("--run-id", required=True)
    run.add_argument("--smoke", action="store_true")
    report = subparsers.add_parser(
        "report", help="build a report derivative from saved artifacts without changing the source run"
    )
    report.add_argument("--run-dir", required=True, type=Path)
    resume = subparsers.add_parser("resume-optuna", help="extend persistent Optuna studies in a new directory")
    resume.add_argument("--run-dir", required=True, type=Path)
    resume.add_argument("--additional-trials-per-fold", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        path = run_cycle(_resolve(args.contract), args.run_id, args.smoke)
        print(json.dumps({"status": "complete", "run_dir": str(path)}, ensure_ascii=False))
        return 0
    if args.command == "report":
        run_dir = _resolve(args.run_dir)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        report_dir = run_dir.parent / f"{run_dir.name}__report_{stamp}"
        report_dir.mkdir(parents=True, exist_ok=False)
        generate_report_from_saved(run_dir, report_dir)
        source_manifest = run_dir / "execution_manifest.json"
        _write_json(
            report_dir / "report_manifest.json",
            {
                "status": "report_derivative_complete",
                "created_at_utc": _utc_now(),
                "source_run_dir": str(run_dir),
                "source_execution_manifest_sha256": sha256_file(source_manifest)
                if source_manifest.is_file()
                else None,
                "output_hashes": _output_hashes(report_dir),
            },
        )
        print(
            json.dumps(
                {
                    "status": "report_derivative_complete",
                    "source_run_dir": str(run_dir),
                    "report_dir": str(report_dir),
                },
                ensure_ascii=False,
            )
        )
        return 0
    extension = resume_optuna(_resolve(args.run_dir), args.additional_trials_per_fold)
    print(json.dumps({"status": "study_extended", "extension_dir": str(extension)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
