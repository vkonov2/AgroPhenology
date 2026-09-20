"""Checks that reporting pools counts and preserves paired populations."""
from __future__ import annotations

import pandas as pd
import pytest

from agro_phenology.early_warning_reporting import (
    PAIRED_COMPARISONS,
    aggregate_budget_grid_metrics,
    paired_year_bootstrap,
    strong_effect_decision,
    write_report_ru,
)


def _strong_effect_contract():
    return {
        "notification_policy": {
            "research_budget": {
                "messages_per_30_field_days_max": 2.0,
                "active_alarm_fraction_max": 0.5,
            }
        },
        "selection": {
            "strong_effect": {
                "timely_recall_gain_min": 0.15,
                "candidate_must_meet_external_research_budget": True,
                "comparable_burden": {
                    "delta_messages_per_30_field_days_max": 0.0,
                    "delta_active_alarm_fraction_max": 0.0,
                },
                "efficiency_branch": {
                    "relative_message_reduction_min": 0.30,
                    "timely_recall_loss_max": 0.05,
                    "delta_active_alarm_fraction_max": 0.0,
                },
            }
        },
    }


def _comparison(**overrides):
    values = {
        "delta_timely_recall": 0.15,
        "delta_messages_per_30_field_days": 0.0,
        "relative_message_reduction": 0.0,
        "delta_active_alarm_fraction": 0.0,
        "candidate_messages_per_30_field_days": 1.5,
        "candidate_active_alarm_fraction": 0.4,
    }
    values.update(overrides)
    return values


def test_budget_grid_pools_external_counts_before_calculating_rates():
    source = pd.DataFrame(
        [
            {
                "model_code": "C4", "fold_id": "test_2020", "messages_budget_per_30": 2.0,
                "alarm_fraction_budget": 0.5, "validation_feasible": True,
                "test_events": 2, "test_timely_hits": 1, "test_field_days": 10,
                "test_messages": 2, "test_alarm_fraction": 0.3,
            },
            {
                "model_code": "C4", "fold_id": "test_2021", "messages_budget_per_30": 2.0,
                "alarm_fraction_budget": 0.5, "validation_feasible": False,
                "test_events": 8, "test_timely_hits": 2, "test_field_days": 30,
                "test_messages": 2, "test_alarm_fraction": 0.2,
            },
        ]
    )

    row = aggregate_budget_grid_metrics(source).query("period == '2020_2025'").iloc[0]

    assert row.test_events == 10
    assert row.test_timely_hits == 3
    assert row.test_timely_recall == pytest.approx(0.3)
    assert row.test_messages_per_30 == pytest.approx(3.0)
    assert row.test_active_alarm_days == 9
    assert row.test_alarm_fraction == pytest.approx(9 / 40)
    assert row.validation_feasible_folds == 1
    assert row.validation_total_folds == 2


def test_default_paired_comparisons_include_all_model_candidates_against_calendar():
    expected = {
        (model, "calendar_window")
        for model in ("C0", "C1", "C2", "C3", "C4", "C5", "C4_optuna")
    }
    assert expected.issubset(set(PAIRED_COMPARISONS))


def test_calendar_comparison_uses_exact_same_events_and_days():
    event_rows = []
    day_rows = []
    for model, hit, message in (("C0", True, True), ("calendar_window", False, False)):
        event_rows.append({
            "season": 2020, "field_season": "field-2020", "model_code": model,
            "evaluation_scope": "paired_candidate_days", "slice": "A_plus_B",
            "warnable_event": True, "timely_hit": hit,
        })
        for day in pd.date_range("2020-06-01", periods=2):
            day_rows.append({
                "season": 2020, "field_season": "field-2020", "issue_date": day,
                "model_code": model, "evaluation_scope": "paired_candidate_days",
                "evaluation_scope_day": True, "service_active": True,
                "coordinate_scope": "A_direct", "message_issued": message and day.day == 1,
                "alarm_active": message, "score": 0.8,
            })

    result = paired_year_bootstrap(
        pd.DataFrame(event_rows), pd.DataFrame(day_rows), n_bootstrap=10,
        periods={"2020": (2020, 2020)}, comparisons=(("C0", "calendar_window"),),
        slices=("A_plus_B",),
    ).iloc[0]

    assert result.status == "paired"
    assert result.delta_timely_recall == pytest.approx(1.0)
    assert result.delta_messages_per_30_field_days == pytest.approx(15.0)
    assert result.candidate_messages_per_30_field_days == pytest.approx(15.0)
    assert result.baseline_messages_per_30_field_days == pytest.approx(0.0)
    assert result.candidate_active_alarm_fraction == pytest.approx(1.0)
    assert result.baseline_active_alarm_fraction == pytest.approx(0.0)
    assert result.interval_status == "not_estimable_single_year"


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"delta_messages_per_30_field_days": 0.001}, False),
        ({"delta_active_alarm_fraction": 0.001}, False),
        ({"candidate_messages_per_30_field_days": 2.001}, False),
        ({"candidate_active_alarm_fraction": 0.501}, False),
        ({
            "delta_timely_recall": -0.05,
            "delta_messages_per_30_field_days": -0.5,
            "relative_message_reduction": 0.30,
        }, True),
        ({
            "delta_timely_recall": -0.051,
            "delta_messages_per_30_field_days": -0.5,
            "relative_message_reduction": 0.30,
        }, False),
        ({
            "delta_timely_recall": -0.05,
            "delta_messages_per_30_field_days": -0.5,
            "relative_message_reduction": 0.299,
        }, False),
        ({
            "delta_timely_recall": -0.05,
            "delta_messages_per_30_field_days": -0.5,
            "relative_message_reduction": 0.30,
            "delta_active_alarm_fraction": 0.001,
        }, False),
        ({
            "delta_timely_recall": 0.046,
            "delta_messages_per_30_field_days": 0.066,
            "relative_message_reduction": -0.037,
            "delta_active_alarm_fraction": 0.0181,
        }, False),
    ],
)
def test_strong_effect_requires_comparable_burden_and_external_budget(overrides, expected):
    decision = strong_effect_decision(_comparison(**overrides), _strong_effect_contract())
    assert decision["strong_effect"] is expected


@pytest.mark.parametrize(
    ("delta_messages", "expected_phrase"),
    [
        (0.01, "Сильное превосходство над календарным окном в первом цикле не подтверждено"),
        (0.0, "Заранее заданный практический порог эффекта достигнут"),
    ],
)
def test_report_uses_contract_strong_effect_rule(tmp_path, delta_messages, expected_phrase):
    row = {
        "period": "2020_2025", "evaluation_scope": "paired_candidate_days",
        "slice": "A_plus_B", "candidate": "C1", "baseline": "calendar_window",
        "status": "paired", "bootstrap_repetitions": 10, "bootstrap_seed": 1,
        "delta_timely_recall_low": 0.0, "delta_timely_recall_high": 0.2,
        "interval_status": "test",
        **_comparison(delta_messages_per_30_field_days=delta_messages),
    }
    destination = write_report_ru(
        tmp_path / "report.md", pooled_summary=pd.DataFrame(),
        paired_comparisons=pd.DataFrame([row]), delay_sensitivity=pd.DataFrame(),
        audits={}, contract=_strong_effect_contract(),
    )
    text = destination.read_text(encoding="utf-8")
    assert expected_phrase in text


def test_report_surfaces_historical_context_budget_grid_optuna_and_c6(tmp_path):
    pooled = pd.DataFrame([
        {
            "period": "2020_2025", "evaluation_scope": "service_calendar", "slice": slice_name,
            "model_code": "C0", "timely_hits": hits, "events_with_warning_opportunity": 10,
            "timely_recall": hits / 10,
        }
        for slice_name, hits in (("prior_gap_le14", 4), ("prior_gap_le21", 5))
    ])
    paired = pd.DataFrame([{
        "period": "2020_2025", "evaluation_scope": "paired_candidate_days", "slice": "A_plus_B",
        "candidate": "C1", "baseline": "calendar_window", "status": "paired",
        "bootstrap_repetitions": 2000, "bootstrap_seed": 20260910,
        "delta_timely_recall": 0.04, "delta_timely_recall_low": -0.03,
        "delta_timely_recall_high": 0.12, "delta_messages_per_30_field_days": 0.1,
        "relative_message_reduction": -0.05, "delta_active_alarm_fraction": 0.02,
        "interval_status": "few_year_blocks_unadjusted_percentile_interval",
    }])
    grid = pd.DataFrame([{
        "period": "2020_2025", "model_code": "C1", "messages_budget_per_30": 2.0,
        "alarm_fraction_budget": 0.5, "test_timely_hits": 5, "test_events": 10,
        "test_messages_per_30": 1.5, "test_alarm_fraction": 0.3,
        "validation_feasible_folds": 5, "validation_total_folds": 6,
    }])
    trials = pd.DataFrame({"fold_id": ["test_2020", "test_2020"], "state": ["COMPLETE", "COMPLETE"]})
    seeds = pd.DataFrame({
        "fold_id": ["test_2020", "test_2020"], "seed": [1, 2],
        "validation_timely_recall": [0.5, 0.6],
    })

    destination = write_report_ru(
        tmp_path / "report.md", pooled_summary=pooled, paired_comparisons=paired,
        delay_sensitivity=pd.DataFrame(), audits={"data_audit": {
            "old_calendar_test_AP_from_frozen_report": 0.6702,
            "old_boosting_test_AP_from_frozen_export": 0.6398826,
        }}, budget_grid_pooled=grid, optuna_trials=trials, optuna_seed_checks=seeds,
    )
    text = destination.read_text(encoding="utf-8")

    assert "AP=0.6702" in text and "AP=0.6399" in text
    assert "нельзя численно сравнивать" in text
    assert "Сильное превосходство над календарным окном в первом цикле не подтверждено" in text
    assert "Завершено 2 из 2" in text
    assert "Сетка бюджетов уведомлений" in text
    assert "Чувствительность к процессу визитов" in text
    assert "4/10; 40.0%" in text and "5/10; 50.0%" in text
    assert "не исправляет observation bias" in text
    assert "C6: календарный logit" in text
