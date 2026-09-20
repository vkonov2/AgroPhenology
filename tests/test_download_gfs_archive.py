from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from agro_phenology.early_warning_gfs_experiment import (
    load_forecast_feature_table,
    validate_forecast_input_bundle,
)
from agro_phenology.gfs_feature_builder import build_request_plan, requests_for_plan_row


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "download_gfs_archive.py"
SPEC = importlib.util.spec_from_file_location("download_gfs_archive", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
download = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = download
SPEC.loader.exec_module(download)


def _plan() -> pd.DataFrame:
    contract = json.loads(download.DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    decisions = pd.DataFrame(
        [
            {
                "season": 2024,
                "issue_date": pd.Timestamp("2024-06-01"),
                "issued_at": pd.Timestamp("2024-06-01T05:00:00Z"),
                "service_active": True,
                "weather_cell": "57.00_24.00",
            }
        ]
    )
    return build_request_plan(decisions, contract, start_year=2024, end_year=2024)


def _write_frozen_plan(run_root: Path) -> pd.DataFrame:
    run_root.mkdir(parents=True)
    plan = _plan()
    plan_path = run_root / download.PLAN_FILE_NAME
    plan.to_parquet(plan_path, index=False)
    manifest = {
        "schema_version": download.PLAN_SCHEMA_VERSION,
        "request_plan_sha256": download.sha256_file(plan_path),
        "contract_sha256": download.sha256_file(download.DEFAULT_CONTRACT),
        "parent_daily_decisions_sha256": download.sha256_file(
            download.DEFAULT_PARENT_DECISIONS
        ),
        "checkpoints": 2,
        "archive_subset_requests": 56,
        "scenario_hours": [4, 7],
        # Real v2 was frozen before the explicit privacy keys were split.
        "contains_field_ids_or_outcomes": False,
        "privacy": "local_project_grid_selection_do_not_publish",
    }
    (run_root / download.PLAN_MANIFEST_FILE_NAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return plan


def _complete_checkpoint(row: pd.Series) -> pd.DataFrame:
    source_hashes = json.dumps([f"{index:064x}" for index in range(28)])
    record = {
        "checkpoint_schema_version": download.CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": row["checkpoint_id"],
        "gfs_cell_id": "gfs025_lat+57.00_lon+024.00",
        "issue_date": row["issue_date"],
        "issue_time_utc": row["issue_time_utc"],
        "availability_scenario_hours": row["availability_scenario_hours"],
        "selected_init_utc": row["selected_init_utc"],
        "assumed_available_at_utc": row["assumed_available_at_utc"],
        "publication_time_utc": pd.NaT,
        "selection_rule_id": row["selection_rule_id"],
        "source_dataset_id": "d084001",
        "first_lead_h": 6,
        "last_lead_h": 168,
        "native_step_hours": 6,
        "expected_steps_1_3d": 12,
        "observed_steps_1_3d": 12,
        "expected_steps_4_7d": 16,
        "observed_steps_4_7d": 16,
        "complete_1_3d": True,
        "complete_4_7d": True,
        "forecast_available": True,
        "checkpoint_complete": True,
        "source_access_mode": download.FAST_NCSS_ACCESS_MODE,
        "requested_snapshot_count": 28,
        "retrieved_snapshot_count": 28,
        "failed_snapshot_count": 0,
        "failed_requests_json": "[]",
        "source_hashes_json": source_hashes,
    }
    record.update({column: 1.0 for column in download.FORECAST_FEATURE_COLUMNS})
    return pd.DataFrame([record])


def _incomplete_checkpoint(row: pd.Series, *, error_type: str = "GFSArchiveError") -> pd.DataFrame:
    request = requests_for_plan_row(row)[0]
    if error_type == "GFSArchiveError":
        error = (
            "HTTP 404 for https://tds.gdex.ucar.edu/thredds/ncss/grid/"
            f"{request.archive_path}?var=x: FileNotFound: No such file or directory"
        )
    else:
        error = "temporary timeout"
    failure = {
        "lead_hours": request.lead_hours,
        "archive_path": request.archive_path,
        "error_type": error_type,
        "error": error,
    }
    record = _complete_checkpoint(row).iloc[0].to_dict()
    record.update(
        {
            "forecast_available": False,
            "checkpoint_complete": False,
            "retrieved_snapshot_count": 27,
            "failed_snapshot_count": 1,
            "failed_requests_json": json.dumps([failure], separators=(",", ":")),
            "source_hashes_json": json.dumps(
                [f"{index:064x}" for index in range(27)]
            ),
            "observed_steps_1_3d": 11,
            "complete_1_3d": False,
        }
    )
    for column in download.FORECAST_FEATURE_COLUMNS:
        record[column] = np.nan
    return pd.DataFrame([record])


def test_existing_legacy_plan_is_validated_without_rewriting(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_frozen_plan(run_root)
    manifest_path = run_root / download.PLAN_MANIFEST_FILE_NAME
    legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy_manifest["created_at_utc"] = "2026-09-10T00:00:00Z"
    manifest_path.write_text(
        json.dumps(legacy_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    before = manifest_path.read_bytes()

    plan, manifest = download.ensure_and_validate_plan(run_root)

    assert len(plan) == 2
    assert manifest["privacy"] == "local_project_grid_selection_do_not_publish"
    assert manifest_path.read_bytes() == before

    compatibility_path, compatibility = download.ensure_plan_compatibility_manifest(
        run_root, plan, manifest
    )
    assert manifest_path.read_bytes() == before
    assert compatibility["legacy_manifest_sha256"] == download.sha256_file(
        manifest_path
    )
    assert compatibility["contains_field_ids"] is False
    assert compatibility["contains_outcomes"] is False
    assert compatibility["contains_coarse_grid_coordinates"] is True
    assert compatibility["distribution"] == "local_private_do_not_publish"
    first = compatibility_path.read_bytes()
    download.ensure_plan_compatibility_manifest(run_root, plan, manifest)
    assert compatibility_path.read_bytes() == first


def test_checkpoint_validation_catches_corruption(tmp_path: Path) -> None:
    row = _plan().iloc[0]
    path = tmp_path / "checkpoint.parquet"
    _complete_checkpoint(row).to_parquet(path, index=False)
    assert download.inspect_checkpoint(path, row).state == "complete"

    corrupt = _complete_checkpoint(row)
    corrupt.loc[0, "source_hashes_json"] = "[]"
    corrupt.to_parquet(path, index=False)
    inspection = download.inspect_checkpoint(path, row)
    assert inspection.state == "invalid"
    assert "хеш" in inspection.reason


def test_source_unavailability_rejects_transient_error() -> None:
    row = _plan().iloc[1]
    frame = _incomplete_checkpoint(row, error_type="GFSArchiveError")
    failure = json.loads(frame["failed_requests_json"].iloc[0])[0]
    failure["error"] = "HTTP 503 for frozen NCSS request"
    with pytest.raises(ValueError, match="не является разрешённым"):
        download._terminal_source_evidence(failure)


def test_documented_source_gap_assembles_and_passes_model_bundle_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    plan = _write_frozen_plan(run_root)
    legacy_path = run_root / download.PLAN_MANIFEST_FILE_NAME
    legacy_before = legacy_path.read_bytes()
    legacy_manifest = json.loads(legacy_before)
    compatibility_path, _ = download.ensure_plan_compatibility_manifest(
        run_root, plan, legacy_manifest
    )

    complete_path = download._checkpoint_path(run_root, plan.iloc[0])
    incomplete_path = download._checkpoint_path(run_root, plan.iloc[1])
    complete_path.parent.mkdir(parents=True, exist_ok=True)
    incomplete_path.parent.mkdir(parents=True, exist_ok=True)
    _complete_checkpoint(plan.iloc[0]).to_parquet(complete_path, index=False)
    _incomplete_checkpoint(plan.iloc[1]).to_parquet(incomplete_path, index=False)
    checkpoint_id = str(plan.iloc[1]["checkpoint_id"])
    with (run_root / download.LOG_FILE_NAME).open("w", encoding="utf-8") as handle:
        for second in range(3):
            handle.write(
                f"2026-09-11T00:00:0{second}Z  {checkpoint_id}: checkpoint пока "
                "не полон: получено 27/28, ошибок 1\n"
            )

    original_validate = download.validate_incomplete_checkpoint

    def validate_without_physical_cache(
        frame: pd.DataFrame, row: object, *, run_root: Path | None = None
    ) -> list[dict[str, object]]:
        return original_validate(frame, row, run_root=None)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(
        download, "validate_incomplete_checkpoint", validate_without_physical_cache
    )
    inspections = {
        str(plan.iloc[0]["checkpoint_id"]): download.CheckpointInspection(
            "complete", "проверен"
        ),
        checkpoint_id: download.CheckpointInspection("incomplete", "27/28"),
    }
    source_unavailability = download.ensure_source_unavailability_manifest(
        run_root,
        plan=plan,
        inspections=inspections,
        compatibility_manifest_path=compatibility_path,
    )
    assert source_unavailability is not None
    unavailable_path, unavailable_manifest = source_unavailability
    assert unavailable_manifest["checkpoint_ids"] == [checkpoint_id]
    assert unavailable_manifest["unavailable_snapshot_count"] == 1
    assert unavailable_manifest["archive_wide_absence_claimed"] is False

    download.assemble_feature_table(run_root=run_root)
    download.bind_assembly_manifests(
        run_root,
        compatibility_manifest_path=compatibility_path,
        source_unavailability=source_unavailability,
    )
    assert download.validate_existing_assembly(
        run_root,
        plan,
        compatibility_manifest_path=compatibility_path,
        source_unavailability=source_unavailability,
    )
    features = load_forecast_feature_table(run_root / download.FEATURE_FILE_NAME)
    assert set(
        features.loc[~features["forecast_available"], "checkpoint_id"].astype(str)
    ) == {checkpoint_id}
    audit = validate_forecast_input_bundle(
        features,
        bundle_paths={
            "forecast_features": run_root / download.FEATURE_FILE_NAME,
            "source_manifest": run_root / download.SOURCE_MANIFEST_FILE_NAME,
            "request_plan": run_root / download.PLAN_FILE_NAME,
            "request_plan_manifest": compatibility_path,
        },
        contract_path=download.DEFAULT_CONTRACT,
        parent_decisions_path=download.DEFAULT_PARENT_DECISIONS,
        expected_scenarios=(4, 7),
    )
    assert audit["status"] == "complete_and_verified"
    assert audit["unavailable_feature_rows"] == 1
    assert legacy_path.read_bytes() == legacy_before

    tampered = json.loads(unavailable_path.read_text(encoding="utf-8"))
    tampered["unavailable_snapshots"][0]["archive_path"] = "wrong/path"
    unavailable_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot evidence"):
        download.validate_source_unavailability_manifest(
            unavailable_path,
            run_root=run_root,
            plan=plan,
            inspections=inspections,
            compatibility_manifest_path=compatibility_path,
        )


def test_interrupted_raw_cache_entries_are_preserved_in_quarantine(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    leaf = run_root / "gfs_raw_cache/d084001/2024/cycle/f006/key"
    leaf.mkdir(parents=True)
    (leaf / "subset.nc").write_bytes(b"partial")
    temporary = run_root / "gfs_raw_cache/.candidate.interrupted.nc"
    temporary.write_bytes(b"temporary")
    log = download.RunLog(run_root / download.LOG_FILE_NAME)

    moved = download.recover_interrupted_raw_writes(run_root, log)

    assert moved == 2
    assert not leaf.exists()
    assert not temporary.exists()
    quarantined = list((run_root / download.QUARANTINE_DIR_NAME).rglob("*"))
    assert any(path.name == "subset.nc" for path in quarantined)
    assert any(path.name == ".candidate.interrupted.nc" for path in quarantined)


def test_kernel_lock_rejects_parallel_runner_but_not_stale_file(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "download.lock"
    with download.exclusive_run_lock(lock):
        with pytest.raises(RuntimeError, match="другой загрузчик"):
            with download.exclusive_run_lock(lock):
                pass
    with download.exclusive_run_lock(lock):
        assert "pid=" in lock.read_text(encoding="utf-8")


def test_check_only_uses_no_network_and_reports_resume_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    plan = _write_frozen_plan(run_root)
    first_path = download._checkpoint_path(run_root, plan.iloc[0])
    first_path.parent.mkdir(parents=True)
    _complete_checkpoint(plan.iloc[0]).to_parquet(first_path, index=False)

    def forbidden_fetch(**_kwargs: object) -> None:
        raise AssertionError("network fetch must not be called")

    monkeypatch.setattr(download, "fetch_request_plan", forbidden_fetch)
    monkeypatch.setattr(
        download, "validate_raw_cache_for_checkpoint", lambda *_args: None
    )
    args = argparse.Namespace(
        run_root=run_root,
        workers=8,
        attempts=3,
        retry_backoff_seconds=0.0,
        outage_after=4,
        outage_pause_seconds=0.0,
        minimum_free_gb=0.0,
        check_only=True,
        allow_sleep=True,
    )

    assert download.run(args) == 0
    summary = json.loads(
        (run_root / download.SUMMARY_FILE_NAME).read_text(encoding="utf-8")
    )
    assert summary["state"] == "check_only"
    assert summary["checkpoint_counts"] == {
        "complete": 1,
        "incomplete": 0,
        "invalid": 0,
        "missing": 1,
    }
    assert summary["model_training_started"] is False


def test_full_runner_skips_complete_checkpoint_and_resumes_missing_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    plan = _write_frozen_plan(run_root)
    first_path = download._checkpoint_path(run_root, plan.iloc[0])
    first_path.parent.mkdir(parents=True)
    _complete_checkpoint(plan.iloc[0]).to_parquet(first_path, index=False)
    fetched_scenarios: list[int] = []

    def fake_fetch(**kwargs: object) -> dict[str, int]:
        scenario = int(kwargs["scenario_hours"][0])  # type: ignore[index]
        fetched_scenarios.append(scenario)
        row = plan.loc[plan["availability_scenario_hours"].eq(scenario)].iloc[0]
        path = download._checkpoint_path(run_root, row)
        path.parent.mkdir(parents=True, exist_ok=True)
        _complete_checkpoint(row).to_parquet(path, index=False)
        return {"written": 1}

    monkeypatch.setattr(download, "fetch_request_plan", fake_fetch)
    monkeypatch.setattr(
        download, "validate_existing_assembly", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        download, "validate_raw_cache_for_checkpoint", lambda *_args: None
    )
    args = argparse.Namespace(
        run_root=run_root,
        workers=8,
        attempts=2,
        retry_backoff_seconds=0.0,
        outage_after=4,
        outage_pause_seconds=0.0,
        minimum_free_gb=0.0,
        check_only=False,
        allow_sleep=True,
    )

    assert download.run(args) == 0
    assert fetched_scenarios == [7]
    summary = json.loads(
        (run_root / download.SUMMARY_FILE_NAME).read_text(encoding="utf-8")
    )
    assert summary["state"] == "complete"
    assert summary["checkpoint_counts"]["complete"] == 2
