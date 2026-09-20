from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agro_phenology.target_early_warning_core import (
    add_daily_features,
    build_daily_decisions,
    build_field_seasons,
)
from agro_phenology.target_early_warning_models import (
    _policy_grid_row,
    select_policy_fixed_grid,
)


def _visits() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "visit_id": "v1",
                "field_season": "f_41_2025",
                "final_field_uid": "f",
                "crop_code": 41,
                "crop_season": 2025,
                "observation_date": pd.Timestamp("2025-04-10"),
                "label_status": "unassessed",
                "adult_trap_positive": False,
                "final_coordinate_class": "A_direct",
                "final_latitude": 56.1,
                "final_longitude": 24.1,
            },
            {
                "visit_id": "v2",
                "field_season": "f_41_2025",
                "final_field_uid": "f",
                "crop_code": 41,
                "crop_season": 2025,
                "observation_date": pd.Timestamp("2025-05-01"),
                "label_status": "unassessed",
                "adult_trap_positive": True,
                "final_coordinate_class": "A_direct",
                "final_latitude": 56.1,
                "final_longitude": 24.1,
            },
            {
                "visit_id": "v3",
                "field_season": "f_41_2025",
                "final_field_uid": "f",
                "crop_code": 41,
                "crop_season": 2025,
                "observation_date": pd.Timestamp("2025-06-10"),
                "label_status": "positive",
                "adult_trap_positive": False,
                "final_coordinate_class": "A_direct",
                "final_latitude": 56.1,
                "final_longitude": 24.1,
            },
            {
                "visit_id": "v4",
                "field_season": "f_41_2025",
                "final_field_uid": "f",
                "crop_code": 41,
                "crop_season": 2025,
                "observation_date": pd.Timestamp("2025-06-20"),
                "label_status": "positive",
                "adult_trap_positive": False,
                "final_coordinate_class": "A_direct",
                "final_latitude": 56.1,
                "final_longitude": 24.1,
            },
        ]
    )


def test_registry_uses_only_first_event_and_next_day_availability() -> None:
    seasons = build_field_seasons(_visits(), snapshot_date="2026-08-31")
    assert len(seasons) == 1
    row = seasons.iloc[0]
    assert row.first_recorded_event_date == pd.Timestamp("2025-06-10")
    assert row.first_recorded_event_available_date == pd.Timestamp("2025-06-11")
    assert row.first_adult_trap_positive_date == pd.Timestamp("2025-05-01")
    assert row.first_adult_trap_positive_available_date == pd.Timestamp("2025-05-02")
    assert bool(row.warnable_first_event)


def test_decision_calendar_does_not_end_at_next_or_last_visit() -> None:
    visits = _visits().iloc[:2].copy()
    seasons = build_field_seasons(visits, snapshot_date="2026-08-31")
    decisions = build_daily_decisions(
        seasons, timezone="Europe/Riga", issue_time="08:00:00"
    )
    assert decisions.issue_date.min() == pd.Timestamp("2025-04-11")
    assert decisions.issue_date.max() == pd.Timestamp("2025-10-31")
    assert (decisions.issue_date - decisions.feature_cutoff_nasa_date).dt.days.eq(2).all()


def _nasa(path: Path, future_delta: float = 0.0) -> None:
    dates = pd.date_range("2025-03-01", "2025-06-30", freq="D")
    temp = np.full(len(dates), 15.0)
    temp[dates > pd.Timestamp("2025-05-10")] += future_delta
    pd.DataFrame(
        {
            "date": dates,
            "T2M": temp,
            "T2M_MIN": temp - 3,
            "T2M_MAX": temp + 3,
            "RH2M": 85.0,
            "PRECTOTCORR": 1.0,
            "WS2M": 2.0,
            "nasa_cell": "56.000_24.000",
            "cache_key": "test",
        }
    ).to_parquet(path, index=False)


def test_features_use_only_weather_through_issue_minus_two(tmp_path: Path) -> None:
    seasons = build_field_seasons(_visits(), snapshot_date="2026-08-31")
    decisions = build_daily_decisions(
        seasons, timezone="Europe/Riga", issue_time="08:00:00"
    )
    decisions = decisions.loc[decisions.issue_date.le("2025-05-10")].copy()
    first, second = tmp_path / "first.parquet", tmp_path / "second.parquet"
    _nasa(first, 0.0)
    _nasa(second, 100.0)
    left, mapping_left, _ = add_daily_features(
        decisions, nasa_daily_path=first, target_key="codling_moth"
    )
    right, mapping_right, _ = add_daily_features(
        decisions, nasa_daily_path=second, target_key="codling_moth"
    )
    columns = [column for column in left if column.startswith(("nasa_", "codling_", "scab_"))]
    pd.testing.assert_frame_equal(left[columns], right[columns])
    pd.testing.assert_frame_equal(mapping_left, mapping_right)


def test_codling_biofix_is_missing_until_observation_is_available(tmp_path: Path) -> None:
    seasons = build_field_seasons(_visits(), snapshot_date="2026-08-31")
    decisions = build_daily_decisions(
        seasons, timezone="Europe/Riga", issue_time="08:00:00"
    )
    source = tmp_path / "nasa.parquet"
    _nasa(source)
    frame, _, _ = add_daily_features(
        decisions.loc[decisions.issue_date.between("2025-04-25", "2025-05-10")],
        nasa_daily_path=source,
        target_key="codling_moth",
    )
    before = frame.loc[frame.issue_date.lt("2025-05-02")]
    after = frame.loc[frame.issue_date.ge("2025-05-02")]
    assert before.codling_adult_biofix_known.eq(0).all()
    assert before.codling_degree_days_since_adult_biofix.isna().all()
    assert after.codling_adult_biofix_known.eq(1).all()


def test_contracts_keep_window_folds_and_disable_optuna() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "docs/research/codling_moth_early_warning/evaluation_contract.json",
        "docs/research/apple_scab_early_warning/evaluation_contract.json",
    ):
        contract = json.loads((root / relative).read_text(encoding="utf-8"))
        assert contract["timeliness_window_days"] == {"minimum": 3, "maximum": 10}
        assert len(contract["rolling_origin_folds"]) == 7
        assert contract["rolling_origin_folds"][-1]["id"] == "test_2026_partial"
        assert contract["rolling_origin_folds"][-1]["primary_pooled_result"] is False
        assert contract["optuna"]["enabled"] is False
        assert contract["notification_policy"]["threshold_grid"][-1] > 1


def test_fixed_policy_grid_includes_no_alert_control() -> None:
    # An empty score above all ordinary thresholds must remain selectable as
    # the explicit no-alert policy under the load budget.
    frame = pd.DataFrame(
        {
            "field_season": ["f"] * 4,
            "season": [2020] * 4,
            "issue_date": pd.date_range("2020-05-01", periods=4),
            "issued_at": ["2020-05-01T08:00:00+03:00"] * 4,
            "service_active": [True] * 4,
            "evaluation_field_day": [True] * 4,
            "target_class": ["no_record_in_horizon"] * 4,
            "target_observable": [True] * 4,
            "days_to_first_recorded_event": [np.nan] * 4,
            "warnable_first_event": [False] * 4,
            "coordinate_scope": ["A_direct"] * 4,
            "previous_visit_gap_days": [np.nan] * 4,
            "common_weather_complete": [True] * 4,
            "candidate_comparison_complete": [True] * 4,
        }
    )
    seasons = pd.DataFrame(
        {
            "field_season": ["f"],
            "season": [2020],
            "coordinate_scope": ["A_direct"],
            "previous_visit_gap_days": [np.nan],
            "first_recorded_event_date": [pd.NaT],
            "warnable_first_event": [False],
            "positive_at_first_visit": [False],
        }
    )
    contract = {
        "notification_policy": {
            "threshold_grid": [0.5, 1.000001],
            "active_days_per_message": 7,
            "cooldown_days": 15,
            "research_budget": {
                "messages_per_30_field_days_max": 0,
                "active_alarm_fraction_max": 0,
            },
        }
    }
    policy, details = select_policy_fixed_grid(
        frame, seasons, pd.Series(1.0, index=frame.index), contract
    )
    assert policy.threshold > 1
    assert details["any_feasible"]


def test_policy_grid_row_preserves_actual_model_code() -> None:
    row = _policy_grid_row(
        "test_2020",
        "O3",
        {
            "fold_id": "validation",
            "model_code": "candidate",
            "threshold": 0.4,
            "timely_hits": 7,
        },
        0.4,
    )
    assert row["fold_id"] == "test_2020"
    assert row["model_code"] == "O3"
    assert row["evaluation_fold_id"] == "validation"
    assert row["evaluation_model_code"] == "candidate"
    assert row["selected"] is True
