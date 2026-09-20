from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
import sqlite3
import zipfile

import pytest

from agro_phenology.shadow_readiness_pipeline import (
    export_review_package,
    next_readiness_run_id,
    run_shadow_readiness,
    verify_readiness_run,
)
from agro_phenology.shadow_registry import PARENT_RUNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CREATED_AT = "2026-09-10T19:00:00Z"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_offline_orchestration_preserves_parents_and_validates_real_demo_decisions(
    tmp_path,
) -> None:
    parent_hashes_before = {
        cycle: _sha256(
            PROJECT_ROOT
            / "results/late_blight_early_warning"
            / metadata["run_id"]
            / "execution_manifest.json"
        )
        for cycle, metadata in PARENT_RUNS.items()
    }
    run_id = "20260910_shadow_readiness_v91"
    result = run_shadow_readiness(
        PROJECT_ROOT,
        results_root=tmp_path,
        run_id=run_id,
        issue_local_date="2026-09-10",
        created_at_utc=CREATED_AT,
        test_summary={"status": "passed", "command": "offline-test-fixture"},
    )
    run = Path(result["run_dir"])

    assert result["status"] == "complete"
    assert result["weather_branch_executable_today"] is False
    assert result["prospective_live_run_executed"] is False
    assert result["network_used"] is False
    assert result["notifications_sent"] == 0
    assert result["schedule_activated"] is False
    assert (run / "technical_demo/shadow.sqlite").is_file()
    assert (run / "review_package.zip").is_file()
    assert (run / "contracts/shadow_readiness_contract.json").is_file()
    snapshot = json.loads((run / "source_snapshot.json").read_text())
    snap_paths = {row["relative_path"] for row in snapshot["files"]}
    assert {
        "src/agro_phenology/shadow_readiness_pipeline.py",
        "src/agro_phenology/shadow_reporting.py",
        "src/agro_phenology/shadow_audit.py",
        "docs/research/late_blight_early_warning/shadow_readiness_contract.json",
        "docs/research/late_blight_early_warning/prospective_logging_spec.md",
    }.issubset(snap_paths)

    before = json.loads((run / "parent_integrity_before.json").read_text())
    after = json.loads((run / "parent_integrity_after.json").read_text())
    assert before["status"] == after["status"] == "passed"
    assert after["identical_to_before"] is True
    assert {
        cycle: before["parents"][cycle]["manifest_sha256"] for cycle in PARENT_RUNS
    } == parent_hashes_before
    parent_hashes_after = {
        cycle: _sha256(
            PROJECT_ROOT
            / "results/late_blight_early_warning"
            / metadata["run_id"]
            / "execution_manifest.json"
        )
        for cycle, metadata in PARENT_RUNS.items()
    }
    assert parent_hashes_after == parent_hashes_before

    verification = verify_readiness_run(run)
    assert verification["status"] == "passed"
    assert verification["outputs_checked"] == result["outputs_checked"]

    demo = json.loads((run / "technical_demo_summary.json").read_text())
    assert demo["mode"] == "retrospective_replay"
    assert demo["field_registry_status"] == "synthetic_only"
    assert demo["decision_records"] == demo["expected_decisions"] == 234
    assert demo["scheduled_slots"] == 18
    assert demo["missed_slots"] == 0
    assert demo["late_slots"] == 0
    assert demo["weather_correction_fraction"] == pytest.approx(1 / 9)
    assert demo["effective_c0_fraction"] == pytest.approx(8 / 9)
    assert demo["virtual_messages"] > 0
    assert demo["active_alarm_days"] > 0
    assert demo["abstentions_by_reason"]

    blockers = json.loads((run / "blockers.json").read_text())
    blocker_ids = {item["id"] for item in blockers["blockers"]}
    assert {
        "readiness_lock_after_current_scheduled_slot",
        "missing_actual_active_field_registry",
        "exact_era5_t_minus_2_unavailable",
        "live_source_freshness_not_measured",
        "independent_field_observation_workflow_not_supplied",
    }.issubset(blocker_ids)
    source_manifest = json.loads(
        (run / "source_compatibility_manifest.json").read_text()
    )
    boundary = source_manifest["shadow_start_boundary"]
    assert boundary["current_issue_slot_can_be_prospective"] is False
    assert boundary["backdating_allowed"] is False
    assert boundary["first_honest_scheduled_slot_local"] == (
        "2026-09-11T08:00:00+03:00"
    )
    report = (run / "operational_readiness_ru.md").read_text()
    assert "не может быть превращён в prospective задним числом" in report
    assert "2026-09-11T08:00:00+03:00" in report

    # Validate actual persisted decision payloads, including fallback and alpha=0,
    # against the emitted schema rather than checking the schema text alone.
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((run / "schemas/decision.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)
    with sqlite3.connect(run / "technical_demo/shadow.sqlite") as connection:
        payloads = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT decision_payload_json FROM decisions ORDER BY decision_slot_utc"
            )
        ]
    assert len(payloads) == 234
    assert isinstance(payloads[0]["input_hashes"], dict)
    assert any(
        isinstance(row["transition"]["score_comparison_segment_id"], int)
        for row in payloads
        if row["transition"]["score_comparison_segment_id"] is not None
    )
    assert {row["score_origin"] for row in payloads}.issubset(
        set(schema["properties"]["score_origin"]["enum"])
    )
    errors = [error for payload in payloads for error in validator.iter_errors(payload)]
    assert errors == []

    with zipfile.ZipFile(run / "review_package.zip") as archive:
        assert not any(name.endswith(".sqlite") for name in archive.namelist())
        assert "technical_demo_raw.json" not in archive.namelist()

    exported = tmp_path / "shared-review.zip"
    exported_result = export_review_package(run, exported)
    assert exported_result["status"] == "exported"
    assert exported_result["sha256"] == _sha256(run / "review_package.zip")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        export_review_package(run, exported)
    with pytest.raises(FileExistsError, match="already exists"):
        run_shadow_readiness(
            PROJECT_ROOT,
            results_root=tmp_path,
            run_id=run_id,
            issue_local_date="2026-09-10",
            created_at_utc=CREATED_AT,
        )


def test_next_run_id_skips_existing_versions_and_rejects_parent_output_root(tmp_path) -> None:
    (tmp_path / "20260910_shadow_readiness_v1").mkdir()
    (tmp_path / "20260910_shadow_readiness_v3").mkdir()
    (tmp_path / "other").mkdir()
    assert next_readiness_run_id(tmp_path, date.fromisoformat("2026-09-10")) == (
        "20260910_shadow_readiness_v4"
    )

    parent = (
        PROJECT_ROOT
        / "results/late_blight_early_warning"
        / PARENT_RUNS["cycle3"]["run_id"]
    )
    with pytest.raises(ValueError, match="immutable parent"):
        run_shadow_readiness(
            PROJECT_ROOT,
            results_root=parent,
            run_id="20260910_shadow_readiness_v99",
            issue_local_date="2026-09-10",
            created_at_utc=CREATED_AT,
        )
