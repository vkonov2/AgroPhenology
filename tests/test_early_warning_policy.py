"""Notification and event-denominator checks using synthetic service calendars."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agro_phenology.early_warning_models import (
    ALWAYS_ON_ACTIVE_DAYS,
    ALWAYS_ON_COOLDOWN_DAYS,
    Policy,
    _baseline_policy,
    _baseline_scores,
    burden_metrics,
    event_metrics,
    simulate_policy,
)


def _decisions(days=30, field="synthetic-a", event="2024-06-20"):
    dates = pd.date_range("2024-06-01", periods=days)
    leads = (pd.Timestamp(event) - dates).days if event else np.full(days, np.nan)
    return pd.DataFrame(
        {
            "field_season": field,
            "season": 2024,
            "issue_date": dates,
            "issued_at": [day.tz_localize("Europe/Riga").replace(hour=8).isoformat() for day in dates],
            "service_active": True,
            "evaluation_field_day": True,
            "target_class": "unknown",
            "target_observable": False,
            "days_to_first_recorded_event": leads,
            "warnable_first_event": event is not None,
            "coordinate_scope": "A_direct",
            "previous_visit_gap_days": 7.0,
            "common_weather_complete": True,
        }
    )


def _registry(*fields):
    return pd.DataFrame(
        [
            {
                "field_season": field,
                "season": 2024,
                "first_recorded_event_date": pd.Timestamp("2024-06-20"),
                "warnable_first_event": True,
                "positive_at_first_visit": False,
                "coordinate_scope": "A_direct",
                "previous_visit_gap_days": 7.0,
            }
            for field in fields
        ]
    )


def test_constant_score_standard_policy_counts_messages_separately_from_active_days():
    frame = _decisions(event=None)
    states = simulate_policy(frame, pd.Series(1.0, index=frame.index), Policy(0.5, active_days=7, cooldown_days=15))
    assert states.loc[states.message_issued, "issue_date"].tolist() == [pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-16")]
    burden = burden_metrics(states, "constant_score_standard_policy", "synthetic")
    assert burden["field_days"] == 30
    assert burden["messages"] == 2
    assert burden["active_alarm_days"] == 14
    assert burden["messages_per_30_field_days"] == 2.0
    assert burden["active_alarm_fraction"] == pytest.approx(14 / 30)
    assert burden["suppressed_repeats"] == 28


def test_baseline_scores_keep_standard_constant_and_always_on_distinct():
    frame = _decisions(event=None)
    frame["hutton_score"] = 0.0
    frame["smith_score"] = 0.0
    frame["polyakov_score"] = np.nan
    scores = _baseline_scores(frame, _registry("synthetic-a"))
    assert "constant" not in scores
    assert scores["constant_score_standard_policy"].eq(1.0).all()
    assert scores["always_on_alarm"].eq(1.0).all()


@pytest.mark.parametrize("days", [365, 366])
def test_always_on_alarm_uses_one_message_and_covers_full_calendar_year(days):
    frame = _decisions(days=days, event=None)
    policy = _baseline_policy("always_on_alarm", active_days=7, cooldown_days=15)
    states = simulate_policy(frame, pd.Series(1.0, index=frame.index), policy)
    burden = burden_metrics(states, "always_on_alarm", "synthetic")

    assert policy.active_days == ALWAYS_ON_ACTIVE_DAYS == 366
    assert policy.cooldown_days == ALWAYS_ON_COOLDOWN_DAYS == 367
    assert states.message_issued.sum() == 1
    assert states.message_issued.iloc[0]
    assert states.alarm_active.all()
    assert burden["active_alarm_days"] == days
    assert burden["active_alarm_fraction"] == 1.0


def test_always_on_alarm_state_is_independent_per_field_season():
    frame = pd.concat(
        [_decisions(30, "synthetic-a", event=None), _decisions(30, "synthetic-b", event=None)],
        ignore_index=True,
    ).sample(frac=1, random_state=42)
    policy = _baseline_policy("always_on_alarm", active_days=7, cooldown_days=15)
    states = simulate_policy(frame, pd.Series(1.0, index=frame.index), policy)

    assert states.alarm_active.all()
    assert states.groupby("field_season")["message_issued"].sum().eq(1).all()


def test_always_on_alarm_does_not_turn_alarm_days_into_timely_messages():
    frame = _decisions(days=20, event="2024-06-20")
    policy = _baseline_policy("always_on_alarm", active_days=7, cooldown_days=15)
    states = simulate_policy(frame, pd.Series(1.0, index=frame.index), policy)
    metrics, events = event_metrics(
        states,
        _registry("synthetic-a"),
        "always_on_alarm",
        "synthetic",
    )

    assert states.alarm_active.all()
    assert states.message_issued.sum() == 1
    assert metrics["timely_hits"] == 0
    assert events[0]["too_early_messages"] == 1


def test_cooldown_expires_on_calendar_boundary_and_allows_repeat():
    frame = _decisions(days=8)
    states = simulate_policy(frame, pd.Series(0.5, index=frame.index), Policy(0.5, active_days=2, cooldown_days=4))
    assert states.message_issued.tolist() == [True, False, False, False, True, False, False, False]
    assert states.alarm_active.tolist() == [True, True, False, False, True, True, False, False]
    assert states.loc[3, "action_reason"] == "suppressed_cooldown"
    assert states.loc[4, "active_through"] == pd.Timestamp("2024-06-06")


def test_refresh_extends_existing_alarm_when_policy_allows_shorter_cooldown():
    frame = _decisions(days=8)
    states = simulate_policy(frame, pd.Series(1.0, index=frame.index), Policy(0.5, active_days=5, cooldown_days=3))
    assert states.loc[states.message_issued, "issue_date"].dt.day.tolist() == [1, 4, 7]
    assert states.alarm_active.all()
    assert states.loc[3, "active_through"] == pd.Timestamp("2024-06-08")
    assert states.loc[6, "active_through"] == pd.Timestamp("2024-06-11")


@pytest.mark.parametrize("later_score", [0.0, np.nan])
def test_existing_alarm_expires_without_silent_cancellation_or_extension(later_score):
    frame = _decisions(days=5)
    score = pd.Series([1.0, later_score, later_score, later_score, later_score])
    states = simulate_policy(frame, score, Policy(0.5, active_days=3, cooldown_days=15))
    assert states.message_issued.tolist() == [True, False, False, False, False]
    assert states.alarm_active.tolist() == [True, True, True, False, False]
    if pd.isna(later_score):
        assert states.loc[1:, "score_status"].eq("abstained").all()
        assert states.loc[1:, "action_reason"].eq("abstained_missing_input").all()


def test_available_positive_record_stops_messages_and_clears_alarm():
    frame = _decisions(days=5)
    frame.loc[2:, ["service_active", "evaluation_field_day"]] = False
    states = simulate_policy(frame, pd.Series(1.0, index=frame.index), Policy(0.5, active_days=7, cooldown_days=15))
    assert states.alarm_active.tolist() == [True, True, False, False, False]
    assert states.loc[2:, "action_reason"].eq("stopped_after_record_available").all()
    assert not states.loc[2:, "message_issued"].any()
    assert burden_metrics(states, "constant_score_standard_policy", "synthetic")["field_days"] == 2


def test_policy_is_chronological_and_has_independent_state_per_field():
    frame = pd.concat([_decisions(5, "synthetic-a"), _decisions(5, "synthetic-b")], ignore_index=True)
    frame = frame.sample(frac=1, random_state=42)
    states = simulate_policy(frame, pd.Series(1.0, index=frame.index), Policy(0.5, active_days=2, cooldown_days=4))
    for _, group in states.groupby("field_season"):
        assert group.loc[group.message_issued, "issue_date"].sort_values().dt.day.tolist() == [1, 5]
        assert group.alarm_active.sum() == 3


def test_future_scores_and_evaluator_labels_cannot_change_past_policy_decisions():
    frame = _decisions(days=20)
    policy = Policy(0.5, active_days=3, cooldown_days=5)
    original = simulate_policy(frame, pd.Series(0.8, index=frame.index), policy)
    changed = frame.copy()
    changed["target_class"] = "actionable"
    changed["target_observable"] = True
    changed["days_to_first_recorded_event"] = 3
    score = pd.Series(0.8, index=frame.index)
    score.loc[10:] = np.nan
    alternate = simulate_policy(changed, score, policy)
    outputs = ["score", "message_issued", "alarm_active", "active_from", "active_through", "action_reason", "suppressed_repeat"]
    pd.testing.assert_frame_equal(original.loc[:9, outputs], alternate.loc[:9, outputs])


@pytest.mark.parametrize(("lead", "hit"), [(0, False), (2, False), (3, True), (10, True), (11, False)])
def test_timely_event_coverage_uses_message_date_not_active_alarm_day(lead, hit):
    frame = _decisions(days=20)
    score = pd.Series(0.0, index=frame.index)
    score.loc[frame.days_to_first_recorded_event.eq(lead)] = 1.0
    states = simulate_policy(frame, score, Policy(0.5, active_days=7, cooldown_days=30))
    metrics, events = event_metrics(states, _registry("synthetic-a"), "synthetic", "test")
    assert metrics["events_with_warning_opportunity"] == 1
    assert metrics["timely_hits"] == int(hit)
    assert events[0]["timely_hit"] is hit
    if lead == 11:
        assert states.loc[states.days_to_first_recorded_event.between(3, 10), "alarm_active"].any()
        assert events[0]["too_early_messages"] == 1
    if lead < 3:
        assert events[0]["late_messages"] == 1


@pytest.mark.parametrize("remove_unscorable_rows", [False, True])
def test_abstention_cannot_remove_event_from_common_denominator(remove_unscorable_rows):
    frame = pd.concat([_decisions(20, "synthetic-a"), _decisions(20, "synthetic-b")], ignore_index=True)
    score = pd.Series(np.nan, index=frame.index)
    score.loc[frame.field_season.eq("synthetic-a")] = 0.0
    score.loc[frame.field_season.eq("synthetic-a") & frame.days_to_first_recorded_event.eq(5)] = 1.0
    states = simulate_policy(frame, score, Policy(0.5))
    if remove_unscorable_rows:
        states = states.loc[states.score.notna()]
    metrics, events = event_metrics(states, _registry("synthetic-a", "synthetic-b"), "synthetic", "test")
    assert metrics["first_events"] == metrics["events_with_warning_opportunity"] == 2
    assert metrics["timely_hits"] == metrics["computable_events"] == 1
    assert metrics["timely_recall"] == metrics["coverage_all_first_events"] == 0.5
    missing = next(event for event in events if event["field_season"] == "synthetic-b")
    assert not missing["timely_hit"]
    assert not missing["computable_in_actionable_window"]


def test_array_simulation_preserves_index_alignment_calendar_gaps_and_scope_mask():
    """Regression for scalar-to-array optimization, with explicit expected states."""
    frame = _decisions(days=8, event=None)
    frame.index = [f"sample-{number}" for number in range(8)]
    frame["issue_date"] = pd.to_datetime(
        ["2024-06-01", "2024-06-02", "2024-06-04", "2024-06-05",
         "2024-06-08", "2024-06-09", "2024-06-13", "2024-06-14"]
    )
    frame["issued_at"] = [
        day.tz_localize("Europe/Riga").replace(hour=8).isoformat() for day in frame.issue_date
    ]
    frame.loc["sample-1", ["service_active", "evaluation_field_day"]] = False
    frame["episode_weather_complete"] = True
    score = pd.Series({f"sample-{number}": 1.0 for number in range(8) if number != 4})
    score.loc["sample-6"] = 0.1
    score.loc["extra-score-without-a-decision"] = np.inf
    score = score.sample(frac=1, random_state=17)
    frame = frame.sample(frac=1, random_state=31)
    # Evaluation masks do not erase state transitions inside simulate_policy.
    mask = pd.Series({"sample-1": True, "sample-5": True}, dtype="boolean")
    states = simulate_policy(frame, score, Policy(1.0, active_days=3, cooldown_days=5), "subset", mask)
    pd.testing.assert_index_equal(states.index, frame.index)
    ordered = states.sort_values("issue_date")
    assert ordered.message_issued.tolist() == [True, False, False, False, False, True, False, True]
    assert ordered.alarm_active.tolist() == [True, False, False, False, False, True, False, True]
    assert ordered.suppressed_repeat.tolist() == [False, False, True, True, False, False, False, False]
    assert ordered.evaluation_scope_day.tolist() == [False, False, False, False, False, True, False, False]
    assert ordered.action_reason.tolist() == [
        "issued_threshold_crossing_or_refresh", "stopped_after_record_available",
        "suppressed_cooldown", "suppressed_cooldown", "abstained_missing_input",
        "issued_threshold_crossing_or_refresh", "below_threshold", "issued_threshold_crossing_or_refresh",
    ]
    pd.testing.assert_series_equal(
        ordered.active_through,
        pd.Series(pd.to_datetime(["2024-06-03", None, None, None, None, "2024-06-11", None, "2024-06-16"]),
                  index=ordered.index, name="active_through"),
    )
    assert ordered.loc["sample-4", "score_status"] == "abstained"
    assert ordered.episode_weather_complete.all()
