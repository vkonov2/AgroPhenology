from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agro_phenology.early_warning_cycle2_reporting import (
    BOOTSTRAP_REPETITIONS,
    _markdown_table,
    build_cycle2_reporting_artifacts,
    classify_c6_vs_c0,
    format_paired_bootstrap,
    leave_one_year_out_cycle2,
    paired_year_bootstrap_cycle2,
    write_cycle2_report_ru,
)


C6 = "C6_weather__c0_policy_replay"


def test_markdown_table_escapes_header_pipes_and_formats_integer_floats() -> None:
    rendered = _markdown_table(
        pd.DataFrame({"season": [2020.0], "delta": [0.125]}),
        (("season", "Год"), ("delta", "Максимальное |Δ|")),
    )
    assert rendered[0] == r"| Год | Максимальное \|Δ\| |"
    assert rendered[2] == "| 2020 | 0.125 |"


def _synthetic_artifacts() -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = {
        2020: {C6: True, "C0": False},
        2021: {C6: True, "C0": True},
        2022: {C6: False, "C0": True},
    }
    events = []
    states = []
    for year, by_model in outcomes.items():
        field = f"private-field-{year}"
        event_date = pd.Timestamp(year, 7, 20)
        for model, hit in by_model.items():
            events.append(
                {
                    "field_season": field,
                    "season": year,
                    "model_code": model,
                    "model_family": "C6_weather" if model == C6 else model,
                    "evaluation_scope": "service_calendar",
                    "slice": "A_plus_B",
                    "warnable_event": True,
                    "timely_hit": hit,
                    "computable_in_actionable_window": True,
                }
            )
            # First date changes the actual message; the second is a score
            # change that stays below threshold and does not alter the policy.
            for offset, low_score in ((5, False), (4, True)):
                if low_score:
                    score = 0.30 if model == C6 else 0.20
                    message = False
                    alarm = False
                else:
                    score = 0.80 if hit else 0.20
                    message = hit
                    alarm = hit
                states.append(
                    {
                        "field_season": field,
                        "season": year,
                        "issue_date": event_date - pd.Timedelta(days=offset),
                        "model_code": model,
                        "evaluation_scope": "service_calendar",
                        "evaluation_scope_day": True,
                        "service_active": True,
                        "score": score,
                        "message_issued": message,
                        "alarm_active": alarm,
                        "suppressed_repeat": False,
                        "action_reason": "issued" if message else "below_threshold",
                        "policy_threshold": 0.5,
                        "policy_active_days": 7,
                        "policy_cooldown_days": 15,
                        "days_to_first_recorded_event": offset,
                        "fallback_to_c0": model == C6 and year == 2022,
                    }
                )
    return pd.DataFrame(events), pd.DataFrame(states)


def test_paired_bootstrap_and_leave_one_year_out_are_exact_and_deterministic() -> None:
    events, states = _synthetic_artifacts()
    annual, first = paired_year_bootstrap_cycle2(
        events,
        states,
        candidates=(C6,),
        baselines=("C0",),
        scopes=("service_calendar",),
    )
    _, second = paired_year_bootstrap_cycle2(
        events,
        states,
        candidates=(C6,),
        baselines=("C0",),
        scopes=("service_calendar",),
    )
    pd.testing.assert_frame_equal(first, second)
    assert len(annual) == 3
    row = first.iloc[0]
    assert row["bootstrap_repetitions"] == BOOTSTRAP_REPETITIONS == 2000
    assert row["candidate_hits"] == 2
    assert row["baseline_hits"] == 2
    assert row["opportunities"] == 3
    assert row["delta_timely_recall"] == pytest.approx(0.0)
    assert row["candidate_fallback_fraction"] == pytest.approx(2 / 6)
    assert row["delta_timely_recall_low"] <= 0 <= row["delta_timely_recall_high"]

    loo = leave_one_year_out_cycle2(annual)
    assert set(loo["omitted_year"]) == {2020, 2021, 2022}
    without_2020 = loo[loo["omitted_year"].eq(2020)].iloc[0]
    assert without_2020["candidate_hits"] == 1
    assert without_2020["baseline_hits"] == 2
    assert without_2020["delta_timely_recall"] == pytest.approx(-0.5)

    formatted = format_paired_bootstrap(first)
    assert "95%" in formatted.loc[0, "Δ recall, п.п. [95%]"]
    assert formatted.loc[0, "Сценарий"] == "полный сервис с fallback"


def test_pairing_rejects_non_matching_days_instead_of_intersecting() -> None:
    events, states = _synthetic_artifacts()
    missing = states.drop(
        states[(states["model_code"].eq("C0")) & states["season"].eq(2022)].index[:1]
    )
    with pytest.raises(ValueError, match="Non-matching evaluation-day population"):
        paired_year_bootstrap_cycle2(
            events,
            missing,
            candidates=(C6,),
            baselines=("C0",),
            scopes=("service_calendar",),
            n_bootstrap=10,
        )


def test_gained_lost_and_policy_path_use_pseudonymous_keys() -> None:
    events, states = _synthetic_artifacts()
    result = classify_c6_vs_c0(
        events, states, candidate=C6, scope="service_calendar"
    )
    pooled = result["gained_lost_summary"].query("aggregation == 'pooled'").iloc[0]
    assert pooled["gained_by_c6"] == 1
    assert pooled["lost_by_c6"] == 1
    assert pooled["both_hit"] == 1
    assert pooled["net_gain"] == 0
    assert pooled["score_change_altered_message_days_in_event_windows"] == 2
    assert pooled["score_change_hidden_below_threshold_days_in_event_windows"] == 3
    assert "field_season" not in result["gained_lost_events"]
    assert "field_season" not in result["message_change_days"]
    assert result["gained_lost_events"]["event_key"].str.startswith("evt_").all()
    assert result["message_change_days"]["day_key"].str.startswith("day_").all()


def test_policy_path_requires_real_suppression_and_checks_below_threshold_first() -> None:
    events, states = _synthetic_artifacts()

    # Both policies issue the same message with different scores.  The active
    # alarm on that date is an outcome, not a blocker.
    same_message = states["season"].eq(2021) & states[
        "days_to_first_recorded_event"
    ].eq(5)
    states.loc[same_message & states["model_code"].eq(C6), "score"] = 0.75

    # A previous alarm remains active, but neither score reaches its threshold.
    below = states["season"].eq(2021) & states[
        "days_to_first_recorded_event"
    ].eq(4)
    states.loc[below, "alarm_active"] = True

    # Here the candidate really crosses and is explicitly suppressed.
    suppressed = states["season"].eq(2020) & states[
        "days_to_first_recorded_event"
    ].eq(4)
    candidate_suppressed = suppressed & states["model_code"].eq(C6)
    states.loc[candidate_suppressed, "score"] = 0.8
    states.loc[candidate_suppressed, "suppressed_repeat"] = True
    states.loc[candidate_suppressed, "alarm_active"] = True
    states.loc[candidate_suppressed, "action_reason"] = "suppressed_cooldown"

    days = classify_c6_vs_c0(
        events, states, candidate=C6, scope="service_calendar"
    )["message_change_days"]
    by_day = days.set_index(["season", "issue_date"])["score_policy_effect"]
    assert by_day.loc[(2021, pd.Timestamp("2021-07-15"))] == (
        "score_change_without_message_change_other"
    )
    assert by_day.loc[(2021, pd.Timestamp("2021-07-16"))] == (
        "score_change_hidden_below_threshold"
    )
    assert by_day.loc[(2020, pd.Timestamp("2020-07-16"))] == (
        "score_change_hidden_by_cooldown_or_active_alarm"
    )


def test_build_api_combines_new_and_parent_artifacts() -> None:
    events, states = _synthetic_artifacts()
    built = build_cycle2_reporting_artifacts(
        events[events["model_code"].eq(C6)],
        states[states["model_code"].eq(C6)],
        reference_event_hits=events[events["model_code"].eq("C0")],
        reference_alarm_states=states[states["model_code"].eq("C0")],
        candidates=(C6,),
        baselines=("C0",),
        scopes=("service_calendar",),
        n_bootstrap=25,
    )
    assert set(built) == {
        "paired_annual_metrics",
        "paired_year_bootstrap",
        "leave_one_year_out",
        "gained_lost_events",
        "gained_lost_summary",
        "message_change_days",
        "message_change_summary",
    }
    assert len(built["paired_year_bootstrap"]) == 1


def _report_artifacts() -> dict[str, pd.DataFrame]:
    events, states = _synthetic_artifacts()
    built = build_cycle2_reporting_artifacts(
        events,
        states,
        candidates=(C6,),
        baselines=("C0",),
        scopes=("service_calendar",),
        n_bootstrap=50,
    )
    built["paired_bootstrap"] = pd.DataFrame(
        [
            {
                "period": "2020_2025",
                "slice": "A_plus_B",
                "candidate": f"C6_weather__{mode}",
                "baseline": f"{control}__{mode}",
                "evaluation_scope": scope,
                "candidate_hits": 45 if mode == "c0_policy_replay" else 44,
                "baseline_hits": 43 if control == "C6_calibration_control" else 42,
                "delta_timely_recall": 2 / 87,
                "delta_timely_recall_low": -0.02,
                "delta_timely_recall_high": 0.08,
                "delta_messages_per_30_field_days": 0.01,
                "delta_active_alarm_fraction": 0.002,
            }
            for mode in ("c0_policy_replay", "validation_selected")
            for scope in ("service_calendar", "paired_candidate_days")
            for control in ("C6_calibration_control", "C6_calendar_control")
        ]
    )
    built.update(
        {
            "v3_event_intersections": pd.DataFrame(
                {
                    "aggregation": ["pooled"],
                    "pair_id": ["calendar_window__vs__C4"],
                    "events": [87],
                    "both_hit": [30],
                    "baseline_only_hit": [15],
                    "candidate_only_hit": [15],
                    "neither_hit": [27],
                    "oracle_union_hits": [60],
                }
            ),
            "v3_miss_reason_summary": pd.DataFrame(
                {
                    "aggregation": ["pooled"],
                    "pair_id": ["calendar_window__vs__C4"],
                    "overlap_category": ["neither_hit"],
                    "missed_model": ["C4"],
                    "missed_events": [27],
                    "reason_score_below_threshold": [20],
                    "reason_cooldown_suppression": [2],
                }
            ),
            "v3_yearly_funnel": pd.DataFrame(
                {
                    "aggregation": ["pooled"],
                    "season": [pd.NA],
                    "field_seasons_total": [488],
                    "all_first_events": [332],
                    "events_after_connection": [199],
                    "events_with_actionable_time": [199],
                    "main_registry_events": [87],
                    "events_with_any_common_weather_in_window": [87],
                    "service_field_days": [1000],
                    "common_candidate_days": [800],
                    "calendar_window_timely_hits": [45],
                    "C0_timely_hits": [37],
                    "C1_timely_hits": [49],
                    "C4_timely_hits": [45],
                    "C5_timely_hits": [45],
                }
            ),
            "v3_periodic_30d": pd.DataFrame(
                {
                    "aggregation": ["pooled"],
                    "season": [pd.NA],
                    "evaluation_scope": ["service_calendar"],
                    "timely_hits": [19],
                    "events_with_warning_opportunity": [87],
                    "messages_per_30_field_days": [0.611],
                    "active_alarm_fraction": [0.138],
                    "computable_fraction": [1.0],
                }
            ),
            "v3_periodic_17d_phase_summary": pd.DataFrame(
                {
                    "aggregation": ["year"],
                    "season": [2020],
                    "evaluation_scope": ["service_calendar"],
                    "phases": [17],
                    "timely_hits_min": [20],
                    "timely_hits_median": [40],
                    "timely_hits_max": [50],
                    "messages_per_30_min": [1.6],
                    "messages_per_30_median": [1.7],
                    "messages_per_30_max": [1.8],
                }
            ),
            "v3_polyakov_computability": pd.DataFrame(
                {
                    "aggregation": ["pooled"],
                    "season": [pd.NA],
                    "evaluation_scope": ["service_calendar"],
                    "events_with_warning_opportunity": [87],
                    "timely_hits": [2],
                    "computable_events": [7],
                    "computable_event_fraction": [7 / 87],
                    "conditional_timely_recall": [2 / 7],
                    "computable_fraction": [0.0446],
                    "bbch_causally_available_events": [12],
                    "common_weather_events": [87],
                    "positive_score_events": [2],
                }
            ),
            "v3_polyakov_status_counts": pd.DataFrame(
                {
                    "aggregation": ["pooled", "pooled"],
                    "season": [pd.NA, pd.NA],
                    "polyakov_status": [
                        "not_evaluable_missing_observed_bbch51",
                        "LOW_WEATHER_RISK",
                    ],
                    "field_days": [650, 117],
                    "field_seasons_with_observed_bbch51": [12, 12],
                }
            ),
            "oof_provenance": pd.DataFrame(
                {
                    "outer_fold_id": ["test_2020"],
                    "forecast_year": [2012],
                    "baseline_fit_year_end": [2011],
                    "forecast_rows": [100],
                    "status": ["predicted"],
                }
            ),
            "policy_selection": pd.DataFrame(
                {
                    "fold_id": ["test_2020"],
                    "evaluation_scope": ["service_calendar"],
                    "model_code": ["C6_weather"],
                    "policy_mode": ["c0_policy_replay"],
                    "alpha": [0.0],
                    "threshold": [0.1],
                    "active_days": [7],
                    "cooldown_days": [15],
                }
            ),
            "annual_metrics": pd.DataFrame(
                {
                    "model_code": [C6, C6],
                    "model_family": ["C6_weather", "C6_weather"],
                    "fold_id": ["test_2020", "test_2026_partial"],
                    "policy_mode": ["c0_policy_replay", "c0_policy_replay"],
                    "evaluation_scope": ["service_calendar", "service_calendar"],
                    "alpha": [0.0, 0.1],
                    "timely_hits": [2, 1],
                    "events_with_warning_opportunity": [3, 2],
                    "messages_per_30_field_days": [10.0, 1.2],
                    "active_alarm_fraction": [1 / 3, 0.2],
                    "computable_fraction": [1.0, 1.0],
                    "fallback_day_fraction": [1 / 3, 0.25],
                }
            ),
            "c4_optuna_intersections": pd.DataFrame(
                {
                    "aggregation": ["pooled"],
                    "evaluation_scope": ["service_calendar"],
                    "events": [87],
                    "both_hit": [30],
                    "baseline_only_hit": [13],
                    "candidate_only_hit": [11],
                    "neither_hit": [33],
                    "baseline_hits": [43],
                    "candidate_hits": [41],
                }
            ),
            "c4_optuna_policy_selection": pd.DataFrame(
                {
                    "fold_id": ["test_2020"],
                    "evaluation_scope": ["service_calendar"],
                    "model_code": ["C4_optuna"],
                    "threshold": [0.022],
                    "active_days": [7],
                    "cooldown_days": [15],
                    "validation_policy_feasible": [True],
                    "validation_timely_recall": [2 / 3],
                    "validation_messages_per_30": [1.02],
                    "validation_alarm_fraction": [0.23],
                    "model_params": [
                        '{"depth":4,"iterations":160,"l2_leaf_reg":1.8}'
                    ],
                    "model_seed": [20261910],
                    "fit_population": ["train_only_common_complete_cases"],
                }
            ),
            "c4_optuna_external_metrics": pd.DataFrame(
                {
                    "aggregation": ["pooled"],
                    "model_code": ["C4_optuna"],
                    "evaluation_scope": ["service_calendar"],
                    "timely_hits": [39],
                    "events_with_warning_opportunity": [87],
                    "messages_per_30_field_days": [1.7],
                    "active_alarm_fraction": [0.36],
                }
            ),
            "alpha0_identity_checks": pd.DataFrame(
                {
                    "status": ["passed", "passed"],
                    "threshold_reselected": [False, False],
                    "max_abs_c0_score_difference": [0.0, 0.0],
                }
            ),
            "model_reload_verification": pd.DataFrame(
                {
                    "status": ["passed", "passed"],
                    "max_abs_base_logits_difference": [0.0, 0.0],
                    "max_abs_correction_logits_difference": [0.0, 0.0],
                    "max_abs_probabilities_difference": [0.0, 0.0],
                    "policy_roundtrip_equal": [True, True],
                    "sample_rows": [50, 50],
                    "fallback_rows_checked": [25, 0],
                    "fallback_mask_roundtrip_equal": [True, True],
                }
            ),
            # These rows are intentionally never rendered directly.
            "private_detail": pd.DataFrame(
                {
                    "field_season": ["SECRET-FIELD-UUID"],
                    "final_latitude": [55.123456],
                    "final_longitude": [37.654321],
                }
            ),
        }
    )
    return built


def test_russian_report_answers_four_questions_and_excludes_raw_identifiers(tmp_path: Path) -> None:
    artifacts = _report_artifacts()
    destination = write_cycle2_report_ru(
        tmp_path / "report_ru.md",
        artifacts,
        metadata={
            "run_id": "synthetic_cycle2",
            "question_answers": {
                "different_hits": "Да, разнонаправленные события есть.",
                "c6_addition": "Устойчивой добавки не показано.",
                "weather_specificity": "Погода не отделилась от контролей.",
                "next_experiment": "Проверить наблюдаемость погоды.",
            },
            "next_priority": "Проверить наблюдаемость погоды на дату выпуска.",
        },
    )
    text = destination.read_text(encoding="utf-8")
    assert "## Четыре ответа" in text
    assert "## Один следующий приоритет" in text
    assert "общая маска v3" in text
    assert "полный сервис с fallback" in text
    assert "Поляков" in text
    assert "Optuna" in text
    assert "OOF" in text
    assert "Парные годовые значения 2020–2025" in text
    assert "Неполный 2026 год" in text
    assert "test_2026_partial" in text
    assert "Контрольные проверки alpha=0 и сериализации" in text
    assert "Попадания календаря" in text
    assert "BBCH доступна в окне" in text
    assert "not_evaluable_missing_observed_bbch51" in text
    assert "Порог, нагрузка validation и параметры сохранённых C4" in text
    assert "SECRET-FIELD-UUID" not in text
    assert "55.123456" not in text
    assert "37.654321" not in text


def test_automatic_weather_specificity_answer_uses_control_pairs(tmp_path: Path) -> None:
    destination = write_cycle2_report_ru(
        tmp_path / "automatic.md",
        _report_artifacts(),
        metadata={"next_priority": "Один заранее заданный следующий опыт."},
    )
    text = destination.read_text(encoding="utf-8")
    assert "При replay на общей маске: погода 45, калибровка 43" in text
    assert "календарный CatBoost 42" in text
    assert "нет полной пары контролей" not in text


def test_pipeline_facade_loads_saved_tables(tmp_path: Path) -> None:
    artifacts = _report_artifacts()
    run_dir = tmp_path / "run"
    parent = tmp_path / "v3"
    run_dir.mkdir()
    parent.mkdir()
    for name in (
        "v3_event_intersections",
        "v3_miss_reason_summary",
        "v3_yearly_funnel",
        "v3_periodic_30d",
        "v3_periodic_17d_phase_summary",
        "v3_polyakov_computability",
        "v3_polyakov_status_counts",
        "c4_optuna_intersections",
        "c4_optuna_policy_selection",
        "oof_provenance",
        "policy_selection",
        "annual_metrics",
        "c4_optuna_external_metrics",
        "paired_annual_metrics",
        "paired_year_bootstrap",
        "paired_bootstrap",
        "leave_one_year_out",
        "alpha0_identity_checks",
        "model_reload_verification",
    ):
        artifacts[name].to_csv(run_dir / f"{name}.csv", index=False)
    destination = write_cycle2_report_ru(
        run_dir, v3_dir=parent, contract={"contract_version": "2.0.0"}
    )
    assert destination == run_dir / "report_ru.md"
    assert destination.is_file()
    text = destination.read_text(encoding="utf-8")
    assert "synthetic" not in text
    assert "test_2026_partial" in text
    assert "Парные годовые значения 2020–2025" in text
    assert "нет полной пары контролей" not in text
