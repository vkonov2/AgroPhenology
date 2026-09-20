"""Synthetic checks for second-cycle diagnostics over immutable v3 artefacts."""
from __future__ import annotations

import pandas as pd
import pytest

from agro_phenology.early_warning_cycle2_diagnostics import (
    c4_optuna_diagnostics,
    event_intersections,
    event_policy_details,
    miss_reason_diagnostics,
    periodic_k_all_phases,
    polyakov_computability,
    stable_event_key,
    yearly_funnel,
)
from agro_phenology.early_warning_models import Policy, event_metrics, simulate_policy


def _season_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "field_season": "raw-field-suppressed",
                "season": 2020,
                "first_visit_available_date": pd.Timestamp("2020-06-01"),
                "first_recorded_event_date": pd.Timestamp("2020-06-20"),
                "warnable_first_event": True,
                "positive_at_first_visit": False,
                "coordinate_scope": "A_direct",
                "previous_visit_gap_days": 10,
                "first_observed_bbch51_available_date": pd.Timestamp("2020-06-02"),
            },
            {
                "field_season": "raw-field-low",
                "season": 2020,
                "first_visit_available_date": pd.Timestamp("2020-07-01"),
                "first_recorded_event_date": pd.Timestamp("2020-07-20"),
                "warnable_first_event": True,
                "positive_at_first_visit": False,
                "coordinate_scope": "A_direct",
                "previous_visit_gap_days": 10,
                "first_observed_bbch51_available_date": pd.NaT,
            },
        ]
    )


def _decision_rows(seasons: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in seasons.itertuples(index=False):
        event_date = pd.Timestamp(season.first_recorded_event_date)
        for issue_date in pd.date_range(event_date - pd.Timedelta(days=12), event_date, freq="D"):
            lead = int((event_date - issue_date).days)
            rows.append(
                {
                    "field_season": season.field_season,
                    "season": season.season,
                    "issue_date": issue_date,
                    "issued_at": f"{issue_date.date().isoformat()}T08:00:00+03:00",
                    "service_active": True,
                    "evaluation_field_day": True,
                    "target_class": "actionable" if 3 <= lead <= 10 else "no_record_in_horizon",
                    "target_observable": True,
                    "days_to_first_recorded_event": lead,
                    "warnable_first_event": True,
                    "coordinate_scope": "A_direct",
                    "previous_visit_gap_days": 10,
                    "common_weather_complete": True,
                    "nasa_common_complete": True,
                    "era_common_complete": True,
                    "episode_weather_complete": True,
                    "candidate_comparison_complete": True,
                }
            )
    return pd.DataFrame(rows)


def _diagnostic_saved_frames():
    seasons = _season_rows()
    decisions = _decision_rows(seasons)
    states_by_model = []
    hit_rows = []
    for model in ("A", "B"):
        if model == "A":
            score = decisions["days_to_first_recorded_event"].eq(5).astype(float)
        else:
            score = pd.Series(0.0, index=decisions.index)
            suppressed_field = decisions["field_season"].eq("raw-field-suppressed")
            score.loc[suppressed_field] = 1.0
        states = simulate_policy(
            decisions,
            score,
            Policy(threshold=0.5, active_days=7, cooldown_days=15),
            evaluation_scope="paired_candidate_days",
            evaluation_mask=decisions["candidate_comparison_complete"],
        )
        states["model_code"] = model
        states["fold_id"] = "test_2020"
        states_by_model.append(states)
        _, model_hits = event_metrics(
            states, seasons, model, "test_2020", "A_plus_B", "paired_candidate_days"
        )
        hit_rows.extend(model_hits)
    return seasons, decisions, pd.concat(states_by_model, ignore_index=True), pd.DataFrame(hit_rows)


def test_event_intersections_are_paired_and_privacy_safe():
    seasons, _, _, hits = _diagnostic_saved_frames()

    summary, membership = event_intersections(
        hits,
        pairs=(("A", "B"),),
        years=(2020, 2020),
        event_key_namespace="synthetic",
    )
    pooled = summary.query("aggregation == 'pooled'").iloc[0]

    assert pooled.events == 2
    assert pooled.baseline_only_hit == 2
    assert pooled.both_hit == pooled.candidate_only_hit == pooled.neither_hit == 0
    assert pooled.oracle_union_hits == 2
    assert not {"field_season", "field_uid", "final_latitude", "final_longitude"} & set(membership.columns)
    expected = stable_event_key(seasons.iloc[0].field_season, 2020, "synthetic")
    assert expected in set(membership.event_key)
    assert "raw-field" not in "|".join(membership.event_key)


def test_miss_reasons_separate_low_score_from_cooldown_and_keep_timing_multilabel():
    seasons, _, states, hits = _diagnostic_saved_frames()

    detail, summary = miss_reason_diagnostics(
        hits,
        states,
        seasons,
        models=("A", "B"),
        pairs=(("A", "B"),),
        years=(2020, 2020),
        event_key_namespace="synthetic",
    )
    misses = detail[(detail.model_code == "B") & ~detail.timely_hit].sort_values("primary_reason")

    assert set(misses.primary_reason) == {"cooldown_suppression", "score_below_threshold"}
    suppressed = misses[misses.primary_reason.eq("cooldown_suppression")].iloc[0]
    assert suppressed.reason_only_early_messages
    assert suppressed.reason_active_alarm_from_earlier_message
    assert suppressed.weather_available_dates == suppressed.actionable_dates
    low = misses[misses.primary_reason.eq("score_below_threshold")].iloc[0]
    assert low.reason_no_message_at_any_time
    pooled = summary.query("aggregation == 'pooled' and missed_model == 'B'").iloc[0]
    assert pooled.reason_cooldown_suppression == 1
    assert pooled.reason_score_below_threshold == 1
    assert "field_season" not in detail.columns


def test_event_policy_detail_links_exact_prediction_rows_without_raw_identifier():
    seasons, _, states, hits = _diagnostic_saved_frames()
    notification = states[states.message_issued | states.suppressed_repeat].copy()
    predictions = states[
        [
            "field_season",
            "season",
            "issue_date",
            "issued_at",
            "score",
            "score_status",
            "target_class",
            "target_observable",
            "evaluation_scope",
            "evaluation_scope_day",
            "common_weather_complete",
            "candidate_comparison_complete",
            "model_code",
            "fold_id",
        ]
    ].copy()

    detail = event_policy_details(
        hits,
        states,
        notification,
        predictions,
        seasons,
        models=("A", "B"),
        years=(2020, 2020),
        event_key_namespace="synthetic",
    )

    assert len(detail) == 4
    assert detail.daily_prediction_refs_json.str.contains("predictions.parquet#row=").all()
    assert not {"field_season", "field_uid", "weather_cell"} & set(detail.columns)
    assert detail.event_key.str.startswith("evt_").all()


def test_yearly_funnel_counts_entry_loss_and_common_days():
    seasons, decisions, _, hits = _diagnostic_saved_frames()
    positive_at_entry = seasons.iloc[[0]].copy()
    positive_at_entry["field_season"] = "raw-positive-at-entry"
    positive_at_entry["first_visit_available_date"] = pd.Timestamp("2020-08-21")
    positive_at_entry["first_recorded_event_date"] = pd.Timestamp("2020-08-20")
    positive_at_entry["warnable_first_event"] = False
    seasons = pd.concat([seasons, positive_at_entry], ignore_index=True)

    funnel, losses = yearly_funnel(
        seasons,
        decisions,
        hits,
        hit_models=("A",),
        years=(2020, 2020),
    )
    pooled = funnel.query("aggregation == 'pooled'").iloc[0]

    assert pooled.field_seasons_total == 3
    assert pooled.field_seasons_with_service_days == 2
    assert pooled.all_first_events == 3
    assert pooled.events_after_connection == 2
    assert pooled.events_with_actionable_time == 2
    assert pooled.events_with_all_common_weather_in_window == 2
    entry_loss = losses.query(
        "aggregation == 'pooled' and loss_reason == 'first_event_positive_known_at_entry'"
    ).iloc[0]
    assert entry_loss["count"] == 1


def test_periodic_17d_reports_every_phase_without_selecting_best():
    event_date = pd.Timestamp("2020-06-20")
    seasons = pd.DataFrame(
        [
            {
                "field_season": "periodic-field",
                "season": 2020,
                "first_recorded_event_date": event_date,
                "warnable_first_event": True,
                "positive_at_first_visit": False,
                "coordinate_scope": "A_direct",
                "previous_visit_gap_days": 10,
            }
        ]
    )
    rows = []
    for issue_date in pd.date_range("2020-05-01", periods=34):
        lead = int((event_date - issue_date).days)
        rows.append(
            {
                "field_season": "periodic-field",
                "season": 2020,
                "issue_date": issue_date,
                "issued_at": f"{issue_date.date().isoformat()}T08:00:00+03:00",
                "service_active": True,
                "evaluation_field_day": True,
                "target_class": "actionable" if 3 <= lead <= 10 else "no_record_in_horizon",
                "target_observable": True,
                "days_to_first_recorded_event": lead,
                "warnable_first_event": True,
                "coordinate_scope": "A_direct",
                "previous_visit_gap_days": 10,
                "common_weather_complete": True,
                "candidate_comparison_complete": True,
            }
        )
    decisions = pd.DataFrame(rows)

    phases, summary = periodic_k_all_phases(decisions, seasons, years=(2020, 2020))
    service = phases[
        phases.evaluation_scope.eq("service_calendar") & phases.aggregation.eq("pooled")
    ]

    assert set(service.phase) == set(range(17))
    assert len(service) == 17
    assert service.messages.sum() == 34
    assert service.messages_per_30_field_days.mean() == pytest.approx(30 / 17)
    assert summary.query("aggregation == 'pooled'").phase_selection.eq(
        "none_all_phases_are_diagnostic"
    ).all()
    assert not hasattr(phases, "selected_phase")


def test_polyakov_reports_unconditional_and_conditional_recall():
    hit_rows = []
    state_rows = []
    for scope in ("service_calendar", "paired_candidate_days"):
        for field, computed, hit in (("f1", True, True), ("f2", False, False)):
            hit_rows.append(
                {
                    "field_season": field,
                    "season": 2020,
                    "model_code": "polyakov",
                    "fold_id": "test_2020",
                    "evaluation_scope": scope,
                    "slice": "A_plus_B",
                    "warnable_event": True,
                    "timely_hit": hit,
                    "computable_in_actionable_window": computed,
                }
            )
            state_rows.append(
                {
                    "field_season": field,
                    "season": 2020,
                    "model_code": "polyakov",
                    "evaluation_scope": scope,
                    "evaluation_scope_day": True,
                    "message_issued": hit,
                    "alarm_active": hit,
                    "score": 1.0 if computed else float("nan"),
                    "suppressed_repeat": False,
                }
            )
    daily = pd.DataFrame(
        [
            {
                "field_season": "f1", "season": 2020, "issue_date": pd.Timestamp("2020-06-10"),
                "evaluation_field_day": True, "days_to_first_recorded_event": 5,
                "candidate_comparison_complete": True,
                "polyakov_status": "OUTBREAK_EXPECTED", "polyakov_score": 1.0,
            },
            {
                "field_season": "f2", "season": 2020, "issue_date": pd.Timestamp("2020-06-10"),
                "evaluation_field_day": True, "days_to_first_recorded_event": 5,
                "candidate_comparison_complete": True,
                "polyakov_status": "not_evaluable_missing_observed_bbch51", "polyakov_score": float("nan"),
            },
        ]
    )
    seasons = pd.DataFrame(
        [
            {
                "field_season": "f1", "season": 2020, "warnable_first_event": True,
                "first_observed_bbch51_available_date": pd.Timestamp("2020-06-01"),
            },
            {
                "field_season": "f2", "season": 2020, "warnable_first_event": True,
                "first_observed_bbch51_available_date": pd.NaT,
            },
        ]
    )

    summary, statuses = polyakov_computability(
        pd.DataFrame(hit_rows), pd.DataFrame(state_rows), daily, seasons, years=(2020, 2020)
    )
    pooled = summary.query("aggregation == 'pooled' and evaluation_scope == 'service_calendar'").iloc[0]

    assert pooled.timely_recall == pytest.approx(0.5)
    assert pooled.computable_events == 1
    assert pooled.conditional_timely_recall == pytest.approx(1.0)
    assert pooled.bbch_causally_available_events == 1
    assert pooled.positive_score_events == 1
    assert statuses.query("aggregation == 'pooled'").field_days.sum() == 2


def test_c4_optuna_summary_uses_best_complete_internal_trial_only():
    hit_rows = []
    state_rows = []
    for scope in ("service_calendar", "paired_candidate_days"):
        for model, values in (("C4", (True, False)), ("C4_optuna", (False, True))):
            for field, hit in zip(("f1", "f2"), values):
                hit_rows.append(
                    {
                        "field_season": field,
                        "season": 2020,
                        "model_code": model,
                        "fold_id": "test_2020",
                        "evaluation_scope": scope,
                        "slice": "A_plus_B",
                        "warnable_event": True,
                        "timely_hit": hit,
                    }
                )
                state_rows.append(
                    {
                        "field_season": field,
                        "season": 2020,
                        "model_code": model,
                        "evaluation_scope": scope,
                        "evaluation_scope_day": True,
                        "message_issued": hit,
                        "alarm_active": hit,
                        "score": 0.8,
                        "suppressed_repeat": False,
                    }
                )
    policy = pd.DataFrame(
        [
            {"model_code": model, "fold_id": "test_2020", "evaluation_scope": scope, "threshold": 0.2}
            for model in ("C4", "C4_optuna")
            for scope in ("service_calendar", "paired_candidate_days")
        ]
    )
    trials = pd.DataFrame(
        [
            {"fold_id": "test_2020", "state": "COMPLETE", "value": 0.4, "params": "{\"depth\":2}", "user_attrs": "{}"},
            {"fold_id": "test_2020", "state": "COMPLETE", "value": 0.6, "params": "{\"depth\":3}", "user_attrs": "{}"},
            {"fold_id": "test_2020", "state": "FAIL", "value": 9.0, "params": "{}", "user_attrs": "{}"},
        ]
    )
    seeds = pd.DataFrame(
        [
            {
                "fold_id": "test_2020",
                "validation_timely_recall": value,
                "validation_messages_per_30": 1.0,
                "validation_alarm_fraction": 0.2,
            }
            for value in (0.4, 0.6)
        ]
    )

    result = c4_optuna_diagnostics(
        pd.DataFrame(hit_rows), pd.DataFrame(state_rows), policy, trials, seeds, years=(2020, 2020)
    )
    paired = result["c4_optuna_intersections"].query(
        "aggregation == 'pooled' and evaluation_scope == 'paired_candidate_days'"
    ).iloc[0]

    assert paired.baseline_only_hit == 1
    assert paired.candidate_only_hit == 1
    assert paired.oracle_union_hits == 2
    trial = result["c4_optuna_trial_summary"].iloc[0]
    assert trial.complete_trials == 2
    assert trial.best_internal_value == pytest.approx(0.6)
    assert "\"depth\":3" in trial.best_internal_params
