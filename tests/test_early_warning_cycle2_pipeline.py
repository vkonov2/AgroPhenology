from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from agro_phenology.early_warning_cycle2_pipeline import (
    DEFAULT_V3,
    _pick_alpha,
    _select_alpha_and_policy,
    _verify_source_snapshot,
    _write_source_snapshot,
    verify_v3,
)
from agro_phenology.early_warning_models import Policy


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cycle2_freezes_cycle1_window_folds_and_budgets() -> None:
    cycle1 = json.loads(
        (DEFAULT_V3 / "evaluation_contract.json").read_text(encoding="utf-8")
    )
    cycle2 = json.loads(
        (
            REPO_ROOT
            / "docs/research/late_blight_early_warning/cycle2_evaluation_contract.json"
        ).read_text(encoding="utf-8")
    )
    assert cycle2["parent_target_and_labels"]["actionable_window_days"] == [
        cycle1["timeliness_window_days"]["minimum"],
        cycle1["timeliness_window_days"]["maximum"],
    ]
    assert cycle2["parent_target_and_labels"]["weather_cutoff_days_before_issue"] == cycle1[
        "first_cycle_feature_cutoff_days_before_issue"
    ]
    limits = cycle2["evaluation"]["selection_constraints"]
    assert limits["messages_per_30_field_days_max"] == cycle1["notification_policy"][
        "research_budget"
    ]["messages_per_30_field_days_max"]
    assert limits["active_alarm_fraction_max"] == cycle1["notification_policy"][
        "research_budget"
    ]["active_alarm_fraction_max"]
    assert cycle2["outer_folds"] == "exactly_cycle1_rolling_origin_folds"


def test_completed_v3_hashes_are_currently_unchanged() -> None:
    contract = json.loads(
        (
            REPO_ROOT
            / "docs/research/late_blight_early_warning/cycle2_evaluation_contract.json"
        ).read_text(encoding="utf-8")
    )
    result = verify_v3(
        DEFAULT_V3,
        contract["frozen_parent_cycle"]["execution_manifest_sha256"],
    )
    assert result["status"] == "passed"
    assert result["outputs_checked"] == 87
    assert result["all_output_hashes_match"]


def test_source_snapshot_is_taken_before_run_and_verified_against_current_files(
    tmp_path: Path,
) -> None:
    snapshot = _write_source_snapshot(tmp_path / "source_snapshot.json")
    result = _verify_source_snapshot(snapshot)
    assert result["status"] == "passed"
    assert result["files_checked"] == len(snapshot["files"])

    corrupted = {**snapshot, "files": [dict(item) for item in snapshot["files"]]}
    corrupted["files"][0]["sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="source changed during execution"):
        _verify_source_snapshot(corrupted)


def test_alpha_selection_respects_feasibility_then_predeclared_tie_break() -> None:
    common = {
        "timely_recall": 0.5,
        "messages_per_30_field_days": 1.5,
        "active_alarm_fraction": 0.3,
        "suppressed_repeats": 3,
        "threshold": 0.2,
    }
    records = [
        {**common, "alpha": 0.5, "feasible": True, "constraint_violation": 0.0},
        {**common, "alpha": 0.1, "feasible": True, "constraint_violation": 0.0},
        {
            **common,
            "alpha": 0.0,
            "feasible": False,
            "constraint_violation": 0.01,
            "timely_recall": 1.0,
        },
    ]
    assert _pick_alpha(records)["alpha"] == 0.1

    infeasible = [
        {**common, "alpha": 0.1, "feasible": False, "constraint_violation": 0.2},
        {**common, "alpha": 0.5, "feasible": False, "constraint_violation": 0.1},
    ]
    assert _pick_alpha(infeasible)["alpha"] == 0.5


def test_inactive_rows_cannot_change_validation_selected_policy() -> None:
    decisions = pd.read_parquet(DEFAULT_V3 / "daily_decisions.parquet")
    seasons = pd.read_parquet(DEFAULT_V3 / "field_seasons.parquet")
    frame = decisions[decisions["season"].eq(2018)].copy()
    validation_seasons = seasons[seasons["season"].eq(2018)].copy()
    score = pd.Series(
        0.001 + 0.8 * pd.Series(range(len(frame)), index=frame.index) / max(1, len(frame) - 1),
        index=frame.index,
    )
    contract = json.loads(
        (
            REPO_ROOT
            / "docs/research/late_blight_early_warning/cycle2_evaluation_contract.json"
        ).read_text(encoding="utf-8")
    )
    base, _ = _select_alpha_and_policy(
        frame,
        validation_seasons,
        {0.0: score},
        Policy(0.2, 7, 15),
        "service_calendar",
        None,
        contract,
    )

    inactive = pd.concat([frame.head(1)] * 20, ignore_index=True)
    inactive.index = range(int(frame.index.max()) + 1, int(frame.index.max()) + 21)
    inactive["field_season"] = [f"inactive-{value}" for value in range(len(inactive))]
    inactive["service_active"] = False
    augmented = pd.concat([frame, inactive], axis=0)
    augmented_score = pd.concat(
        [score, pd.Series([0.999999] * len(inactive), index=inactive.index)]
    )
    after, _ = _select_alpha_and_policy(
        augmented,
        validation_seasons,
        {0.0: augmented_score},
        Policy(0.2, 7, 15),
        "service_calendar",
        None,
        contract,
    )
    assert after["validation_selected"]["alpha"] == base["validation_selected"]["alpha"]
    assert after["validation_selected"]["threshold"] == base["validation_selected"]["threshold"]
