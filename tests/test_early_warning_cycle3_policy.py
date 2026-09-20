"""Cycle-three frozen-score policy checks on synthetic calendars."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agro_phenology.early_warning_cycle3_policy import (
    GrowthPolicy,
    clipped_logit,
    dumps_runtime_states,
    loads_runtime_states,
    serialize_runtime_states,
    simulate_growth_policy,
)
from agro_phenology.early_warning_models import (
    Policy,
    burden_metrics,
    event_metrics,
    simulate_policy,
)


def _decisions(
    days: int = 30,
    *,
    field: str = "synthetic-a",
    start: str = "2024-06-01",
    event: str | None = "2024-06-25",
    timezone: str | None = None,
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=days, tz=timezone)
    event_date = pd.Timestamp(event, tz=timezone) if event else None
    leads = (event_date - dates).days if event_date is not None else np.full(days, np.nan)
    return pd.DataFrame(
        {
            "field_season": field,
            "season": dates.year,
            "issue_date": dates,
            "issued_at": dates + pd.Timedelta(hours=8),
            "service_active": True,
            "evaluation_field_day": True,
            "target_class": "unknown",
            "target_observable": False,
            "days_to_first_recorded_event": leads,
            "warnable_first_event": event is not None,
            "coordinate_scope": "A_direct",
            "previous_visit_gap_days": 7.0,
            "common_weather_complete": True,
            "episode_weather_complete": True,
            "candidate_comparison_complete": True,
        }
    )


def _registry(field: str = "synthetic-a") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "field_season": [field],
            "season": [2024],
            "first_recorded_event_date": [pd.Timestamp("2024-06-25")],
            "warnable_first_event": [True],
            "positive_at_first_visit": [False],
            "coordinate_scope": ["A_direct"],
            "previous_visit_gap_days": [7.0],
        }
    )


def _growth_policy(delta: float = np.log(2.0), **kwargs) -> GrowthPolicy:
    return GrowthPolicy(
        threshold=0.1,
        active_days=7,
        cooldown_days=15,
        minimum_repeat_interval_days=7,
        growth_override_enabled=True,
        growth_logit_delta=float(delta),
        **kwargs,
    )


def test_disabled_growth_exactly_reproduces_v3_policy_states_and_metrics() -> None:
    first = _decisions(22, field="a")
    second = _decisions(22, field="b")
    frame = pd.concat([first, second], ignore_index=True).sample(frac=1, random_state=19)
    score = pd.Series(0.8, index=frame.index)
    score.loc[frame["issue_date"].dt.day.isin([4, 5])] = np.nan
    score.loc[frame["issue_date"].dt.day.isin([9, 10])] = 0.1
    frame.loc[
        frame["field_season"].eq("b") & frame["issue_date"].dt.day.ge(20),
        "service_active",
    ] = False
    mask = frame["issue_date"].dt.day.ne(12)
    old_policy = Policy(0.5, 7, 15, "identity_policy")
    old = simulate_policy(frame, score, old_policy, "paired", mask)
    new, _ = simulate_growth_policy(
        frame,
        score,
        GrowthPolicy(
            threshold=0.5,
            active_days=7,
            cooldown_days=15,
            minimum_repeat_interval_days=7,
            growth_override_enabled=False,
            version="identity_policy",
        ),
        "paired",
        mask,
        score_origin="C0",
        model_id="C0",
        model_version="frozen",
    )
    common = [
        "score",
        "score_status",
        "message_issued",
        "alarm_active",
        "active_from",
        "active_through",
        "action_reason",
        "suppressed_repeat",
        "evaluation_scope",
        "evaluation_scope_day",
        "policy_threshold",
        "policy_active_days",
        "policy_cooldown_days",
        "policy_version",
    ]
    pd.testing.assert_frame_equal(old[common], new[common], check_exact=True)

    registry = pd.concat([_registry("a"), _registry("b")], ignore_index=True)
    old_event, old_hits = event_metrics(old, registry, "model", "fold", evaluation_scope="paired")
    new_event, new_hits = event_metrics(new, registry, "model", "fold", evaluation_scope="paired")
    assert old_event == new_event
    assert old_hits == new_hits
    assert burden_metrics(old, "model", "fold", evaluation_scope="paired") == burden_metrics(
        new, "model", "fold", evaluation_scope="paired"
    )


def test_growth_uses_last_real_message_and_updates_reference_only_on_issue() -> None:
    frame = _decisions(24, event=None)
    score = pd.Series(0.05, index=frame.index)
    score.iloc[0] = 0.2
    score.iloc[7] = 0.3  # Suppressed: odds grew by less than x3.
    score.iloc[8] = 0.5  # x4 odds versus 0.2, but only x2.33 versus 0.3.
    states, _ = simulate_growth_policy(
        frame,
        score,
        _growth_policy(np.log(3.0)),
        score_origin="weather",
        model_id="C6",
        model_version="one",
    )
    assert states.loc[states.message_issued, "issue_date"].dt.day.tolist() == [1, 9]
    assert states.loc[7, "action_reason"] == "suppressed_growth_below_delta"
    assert states.loc[8, "action_reason"] == "issued_growth_override"
    assert states.loc[8, "previous_message_score"] == pytest.approx(0.2)
    assert states.loc[8, "logit_growth_from_previous_message"] == pytest.approx(
        clipped_logit(0.5) - clipped_logit(0.2)
    )
    assert states.loc[8, "cumulative_messages_field_season"] == 2


def test_exact_g_and_c_boundaries_threshold_equality_and_alarm_accounting() -> None:
    frame = _decisions(23, event=None)
    score = pd.Series(0.01, index=frame.index)
    score.iloc[0] = 0.1  # equality to theta is a signal
    score.iloc[6] = 0.9  # too early for growth
    score.iloc[7] = 0.9  # exact g boundary
    score.iloc[22] = 0.9  # exact C boundary after the growth message
    states, _ = simulate_growth_policy(
        frame,
        score,
        _growth_policy(np.log(1.5)),
        score_origin="calendar",
        model_id="C0",
        model_version="one",
    )
    assert states.loc[states.message_issued, "issue_date"].dt.day.tolist() == [1, 8, 23]
    assert states.loc[0, "message_kind"] == "ordinary"
    assert states.loc[6, "action_reason"] == "suppressed_growth_minimum_interval"
    assert states.loc[7, "message_kind"] == "growth_override"
    assert states.loc[22, "message_kind"] == "ordinary"
    assert states.loc[0:13, "alarm_active"].all()
    assert not states.loc[14:21, "alarm_active"].any()
    assert states["message_issued"].sum() == 3
    assert states["alarm_active"].sum() == 15


def test_constant_high_score_has_only_ordinary_cooldown_repeats() -> None:
    frame = _decisions(31, event=None)
    states, _ = simulate_growth_policy(
        frame,
        pd.Series(0.8, index=frame.index),
        _growth_policy(np.log(1.5)),
        score_origin="calendar",
        model_id="C1",
        model_version="one",
    )
    assert states.loc[states.message_issued, "issue_date"].tolist() == [
        pd.Timestamp("2024-06-01"),
        pd.Timestamp("2024-06-16"),
        pd.Timestamp("2024-07-01"),
    ]
    assert not states["growth_override_used"].any()
    assert states.loc[7, "logit_growth_from_previous_message"] == pytest.approx(0.0)


def test_smooth_small_growth_drop_and_return_do_not_trigger_override() -> None:
    frame = _decisions(15, event=None)
    score = pd.Series(0.05, index=frame.index)
    score.iloc[0] = 0.4
    score.iloc[7:11] = [0.45, 0.35, 0.4, 0.45]
    states, _ = simulate_growth_policy(
        frame,
        score,
        _growth_policy(np.log(2.0)),
        score_origin="weather",
        model_id="C6",
        model_version="one",
    )
    assert states["message_issued"].sum() == 1
    assert states.loc[7:10, "action_reason"].eq("suppressed_growth_below_delta").all()


def test_source_transition_and_missing_gap_break_comparability_without_reset() -> None:
    frame = _decisions(16, event=None)
    score = pd.Series(0.05, index=frame.index)
    score.iloc[0] = 0.2
    score.iloc[1] = np.nan
    score.iloc[7] = 0.9
    score.iloc[15] = 0.9
    origin = pd.Series("weather", index=frame.index)
    origin.iloc[2] = "C0_fallback"
    states, _ = simulate_growth_policy(
        frame,
        score,
        _growth_policy(np.log(1.5)),
        score_origin=origin,
        model_id="C6",
        model_version="one",
    )
    assert states.loc[7, "action_reason"] == "growth_reference_not_comparable"
    assert states.loc[7, "previous_message_date"] == pd.Timestamp("2024-06-01")
    assert states.loc[7, "elapsed_calendar_days_since_message"] == 7
    assert states.loc[15, "message_issued"]
    assert states.loc[15, "message_kind"] == "ordinary"
    assert states.loc[15, "elapsed_calendar_days_since_message"] == 15


def test_alpha_zero_origin_normalization_makes_weather_availability_irrelevant() -> None:
    frame = _decisions(20, event=None)
    score = pd.Series(np.linspace(0.15, 0.85, len(frame)), index=frame.index)
    weather_available = pd.Series([number % 3 != 0 for number in range(len(frame))])
    c0, _ = simulate_growth_policy(
        frame,
        score,
        _growth_policy(np.log(1.5)),
        score_origin="C0",
        model_id="C0",
        model_version="fold-model",
    )
    # The caller deliberately normalizes effective alpha=0 origin to C0; the
    # unused weather flag is present only to prove that it cannot affect state.
    alternate = frame.assign(episode_weather_complete=weather_available)
    alpha_zero, _ = simulate_growth_policy(
        alternate,
        score,
        _growth_policy(np.log(1.5)),
        score_origin="C0",
        model_id="C0",
        model_version="fold-model",
    )
    columns = [
        "score",
        "message_issued",
        "message_kind",
        "alarm_active",
        "action_reason",
        "suppressed_repeat",
        "score_comparison_segment_id",
    ]
    pd.testing.assert_frame_equal(c0[columns], alpha_zero[columns], check_exact=True)


def test_every_message_restarts_cooldown_and_no_pair_is_closer_than_g() -> None:
    frame = _decisions(30, event=None)
    score = pd.Series(0.01, index=frame.index)
    score.iloc[[0, 7, 13, 14, 21]] = [0.1, 0.3, 0.9, 0.9, 0.999]
    states, _ = simulate_growth_policy(
        frame,
        score,
        _growth_policy(np.log(1.5)),
        score_origin="same",
        model_id="same",
        model_version="same",
    )
    dates = states.loc[states.message_issued, "issue_date"]
    assert dates.dt.day.tolist() == [1, 8, 15, 22]
    assert dates.diff().dropna().dt.days.ge(7).all()
    assert states.loc[13, "action_reason"] == "suppressed_growth_minimum_interval"


def test_runtime_state_roundtrip_and_resume_equal_continuous_replay() -> None:
    frame = _decisions(25, event=None)
    score = pd.Series(np.linspace(0.1, 0.95, len(frame)), index=frame.index)
    origin = pd.Series("weather", index=frame.index)
    origin.iloc[9:11] = "C0_fallback"
    policy = _growth_policy(np.log(1.5))
    continuous, continuous_state = simulate_growth_policy(
        frame,
        score,
        policy,
        score_origin=origin,
        model_id="C6",
        model_version="one",
    )
    first, checkpoint = simulate_growth_policy(
        frame.iloc[:12],
        score.iloc[:12],
        policy,
        score_origin=origin.iloc[:12],
        model_id="C6",
        model_version="one",
    )
    serialized = dumps_runtime_states(checkpoint)
    json.loads(serialized)
    resumed_state = loads_runtime_states(serialized)
    second, final_state = simulate_growth_policy(
        frame.iloc[12:],
        score.iloc[12:],
        policy,
        score_origin=origin.iloc[12:],
        model_id="C6",
        model_version="one",
        initial_states=resumed_state,
    )
    resumed = pd.concat([first, second]).loc[continuous.index]
    compare = [
        "message_issued",
        "message_kind",
        "alarm_active",
        "active_from",
        "active_through",
        "action_reason",
        "previous_message_date",
        "previous_message_score",
        "score_comparison_segment_id",
        "cumulative_messages_field_season",
        "cumulative_alarm_days_field_season",
    ]
    pd.testing.assert_frame_equal(continuous[compare], resumed[compare], check_exact=True)
    assert serialize_runtime_states(continuous_state) == serialize_runtime_states(final_state)


def test_future_scores_labels_visits_and_availability_cannot_change_prefix() -> None:
    frame = _decisions(20)
    score = pd.Series(np.linspace(0.1, 0.8, len(frame)), index=frame.index)
    policy = _growth_policy(np.log(1.5))
    original, _ = simulate_growth_policy(
        frame,
        score,
        policy,
        score_origin="weather",
        model_id="C6",
        model_version="one",
    )
    changed = frame.copy()
    changed.loc[10:, "target_class"] = "actionable"
    changed.loc[10:, "target_observable"] = True
    changed.loc[10:, "days_to_first_recorded_event"] = 3
    changed.loc[10:, "episode_weather_complete"] = False
    changed_score = score.copy()
    changed_score.loc[10:] = changed_score.loc[10:].iloc[::-1].to_numpy()
    alternate, _ = simulate_growth_policy(
        changed,
        changed_score,
        policy,
        score_origin="weather",
        model_id="C6",
        model_version="one",
    )
    columns = [
        "message_issued",
        "alarm_active",
        "active_from",
        "active_through",
        "action_reason",
        "previous_message_date",
        "previous_message_score",
    ]
    pd.testing.assert_frame_equal(
        original.loc[:9, columns], alternate.loc[:9, columns], check_exact=True
    )


def test_multiple_growth_messages_still_count_one_first_event_hit() -> None:
    frame = _decisions(25, event="2024-06-25")
    score = pd.Series(0.01, index=frame.index)
    score.loc[frame["issue_date"].eq(pd.Timestamp("2024-06-15"))] = 0.2
    score.loc[frame["issue_date"].eq(pd.Timestamp("2024-06-22"))] = 0.8
    states, _ = simulate_growth_policy(
        frame,
        score,
        _growth_policy(np.log(1.5)),
        score_origin="weather",
        model_id="C6",
        model_version="one",
    )
    metrics, hits = event_metrics(states, _registry(), "C6", "fold")
    assert metrics["events_with_warning_opportunity"] == 1
    assert metrics["timely_hits"] == 1
    assert hits[0]["timely_message_count"] == 2


def test_calendar_day_boundaries_use_local_dates_across_dst() -> None:
    frame = _decisions(
        8,
        event=None,
        start="2024-03-29",
        timezone="Europe/Riga",
    )
    score = pd.Series(0.01, index=frame.index)
    score.iloc[0] = 0.2
    score.iloc[7] = 0.8
    states, _ = simulate_growth_policy(
        frame,
        score,
        _growth_policy(np.log(1.5)),
        score_origin="weather",
        model_id="C6",
        model_version="one",
    )
    assert states.loc[7, "elapsed_calendar_days_since_message"] == 7
    assert states.loc[7, "message_kind"] == "growth_override"


def test_field_seasons_are_independent_and_inactive_service_clears_only_alarm() -> None:
    frame = pd.concat(
        [_decisions(18, field="a", event=None), _decisions(18, field="b", event=None)],
        ignore_index=True,
    ).sample(frac=1, random_state=8)
    score = pd.Series(0.8, index=frame.index)
    inactive = frame["field_season"].eq("a") & frame["issue_date"].dt.day.eq(4)
    frame.loc[inactive, "service_active"] = False
    states, _ = simulate_growth_policy(
        frame,
        score,
        _growth_policy(np.log(1.5)),
        score_origin="calendar",
        model_id="C0",
        model_version="one",
    )
    for field, group in states.groupby("field_season"):
        messages = group.loc[group.message_issued, "issue_date"].sort_values()
        assert messages.dt.day.tolist() == [1, 16]
        if field == "a":
            assert not group.loc[
                group["issue_date"].dt.day.eq(4), "alarm_active"
            ].any()


def test_duplicate_field_date_and_invalid_probability_are_rejected() -> None:
    frame = _decisions(2, event=None)
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="at most one decision"):
        simulate_growth_policy(duplicate, pd.Series([0.5, 0.5, 0.5]), GrowthPolicy(0.5))
    with pytest.raises(ValueError, match="finite probabilities"):
        simulate_growth_policy(frame, pd.Series([0.5, np.inf]), GrowthPolicy(0.5))


def test_resume_rejects_overlap_or_backdating() -> None:
    frame = _decisions(5, event=None)
    score = pd.Series(0.8, index=frame.index)
    _, checkpoint = simulate_growth_policy(frame.iloc[:3], score.iloc[:3], GrowthPolicy(0.5))
    with pytest.raises(ValueError, match="resume after"):
        simulate_growth_policy(
            frame.iloc[2:], score.iloc[2:], GrowthPolicy(0.5), initial_states=checkpoint
        )
