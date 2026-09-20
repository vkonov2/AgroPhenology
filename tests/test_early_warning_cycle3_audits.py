from __future__ import annotations

import pandas as pd
import pytest

from agro_phenology.early_warning_cycle3_audits import (
    audit_event_populations,
    audit_suppression_paths,
)


CANDIDATE = "C6_weather__validation_selected"


def _event_hits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = {
        "both": (True, True, True, False),
        "gained": (True, False, True, False),
        "lost": (False, True, True, False),
        "neither": (False, False, True, False),
        "entry-a": (False, False, False, True),
        "entry-b": (False, False, False, True),
    }
    candidate = []
    baseline = []
    registry = []
    for field, (candidate_hit, baseline_hit, warnable, at_entry) in outcomes.items():
        common = {
            "field_season": field,
            "season": 2020,
            "fold_id": "test_2020",
            "evaluation_scope": "service_calendar",
            "slice": "A_plus_B",
            "warnable_event": warnable,
            "positive_at_entry": at_entry,
        }
        candidate.append(
            {**common, "model_code": CANDIDATE, "timely_hit": candidate_hit}
        )
        baseline.append({**common, "model_code": "C0", "timely_hit": baseline_hit})
        registry.append(
            {
                "field_season": field,
                "season": 2020,
                "first_recorded_event_date": pd.Timestamp("2020-07-20"),
                "first_recorded_event_available_date": pd.Timestamp("2020-07-21"),
                "entry_category": (
                    "positive_known_at_entry" if at_entry else "observed_after_entry"
                ),
            }
        )
    return pd.DataFrame(candidate), pd.DataFrame(baseline), pd.DataFrame(registry)


def test_population_audit_separates_entry_events_and_enforces_invariants() -> None:
    candidate, baseline, registry = _event_hits()
    artifacts = audit_event_populations(
        candidate,
        baseline,
        candidate=CANDIDATE,
        event_registry=registry,
    )
    pooled = artifacts["population_audit_summary"].query(
        "aggregation == 'pooled'"
    ).iloc[0]
    assert pooled["all_first_events"] == 6
    assert pooled["eligible_events"] == 4
    assert pooled["positive_at_entry_events"] == 2
    assert pooled["both_hit"] == 1
    assert pooled["gained_by_candidate"] == 1
    assert pooled["lost_by_candidate"] == 1
    assert pooled["neither_hit"] == 1
    assert pooled["candidate_timely_events"] == 2
    assert pooled["baseline_timely_events"] == 2
    assert bool(pooled["all_invariants_pass"])

    eligible = artifacts["eligible_gained_lost_events"]
    diagnostic = artifacts["all_first_event_diagnostics"]
    assert len(eligible) == 4
    assert len(diagnostic) == 6
    assert (diagnostic["eligibility_reason"] == "positive_known_at_entry").sum() == 2
    assert "field_season" not in eligible
    assert "field_season" not in diagnostic
    assert eligible["event_key"].str.startswith("evt_").all()


def test_population_audit_rejects_mismatched_real_event_identity_or_status() -> None:
    candidate, baseline, _ = _event_hits()
    with pytest.raises(ValueError, match="Non-matching first-event population"):
        audit_event_populations(
            candidate,
            baseline.iloc[:-1],
            candidate=CANDIDATE,
        )

    changed = baseline.copy()
    changed.loc[changed["field_season"].eq("both"), "warnable_event"] = False
    with pytest.raises(ValueError, match="disagree on warnable_event"):
        audit_event_populations(candidate, changed, candidate=CANDIDATE)


def _alarm_states() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []

    def add_year(
        year: int,
        field: str,
        candidate_scores: list[float],
        baseline_scores: list[float],
        fallback: list[bool],
        alpha: float,
        messages: list[bool],
        suppressed: list[bool],
    ) -> None:
        event_date = pd.Timestamp(year, 7, 12)
        for offset, (candidate_score, baseline_score) in enumerate(
            zip(candidate_scores, baseline_scores), start=1
        ):
            common = {
                "field_season": field,
                "season": year,
                "fold_id": f"test_{year}",
                "issue_date": pd.Timestamp(year, 7, offset),
                "evaluation_scope": "service_calendar",
                "evaluation_scope_day": True,
                "policy_threshold": 0.5,
                "message_issued": messages[offset - 1],
                "suppressed_repeat": suppressed[offset - 1],
                "model_version": "frozen-v4",
            }
            candidate_rows.append(
                {
                    **common,
                    "model_code": CANDIDATE,
                    "score": candidate_score,
                    "alpha": alpha,
                    "fallback_to_c0": fallback[offset - 1],
                    "days_to_first_recorded_event": (
                        event_date - pd.Timestamp(year, 7, offset)
                    ).days,
                    "warnable_first_event": True,
                }
            )
            baseline_rows.append(
                {**common, "model_code": "C0", "score": baseline_score}
            )

    add_year(
        2020,
        "field-weather-2020",
        candidate_scores=[0.4, 0.7, 0.8, 0.9, 0.4, 0.6, 0.7],
        baseline_scores=[0.4, 0.6, 0.6, 0.6, 0.4, 0.6, 0.7],
        fallback=[False, False, False, False, False, True, True],
        alpha=0.25,
        messages=[False, True, False, False, False, False, False],
        suppressed=[False, False, True, True, False, True, True],
    )
    add_year(
        2021,
        "field-alpha0-2021",
        candidate_scores=[0.4, 0.6, 0.7],
        baseline_scores=[0.4, 0.6, 0.7],
        fallback=[True, False, True],
        alpha=0.0,
        messages=[False, True, False],
        suppressed=[False, False, True],
    )
    return pd.DataFrame(candidate_rows), pd.DataFrame(baseline_rows)


def test_suppression_audit_distinguishes_state_crossing_and_source_boundary() -> None:
    candidate, baseline = _alarm_states()
    artifacts = audit_suppression_paths(
        candidate,
        baseline,
        candidate=CANDIDATE,
        years=(2020, 2021),
    )
    yearly = artifacts["suppression_audit_summary"].query(
        "aggregation == 'year'"
    ).set_index("season")
    weather = yearly.loc[2020]
    assert weather["weather_available_days"] == 5
    assert weather["weather_correction_applied_days"] == 5
    assert weather["score_changed_days"] == 3
    assert weather["changed_score_above_threshold_days"] == 3
    assert weather["changed_score_any_message_days"] == 1
    assert weather["changed_score_suppressed_days"] == 2
    assert weather["changed_score_suppressed_field_seasons"] == 1
    assert weather["changed_score_suppressed_episodes"] == 1
    assert weather["posthoc_timely_suppressed_days"] == 2
    assert weather["posthoc_timely_suppressed_events"] == 1
    assert weather["candidate_upward_crossings_raw"] == 2
    assert weather["candidate_upward_crossings_comparable"] == 1
    assert weather["baseline_upward_crossings_comparable"] == 2
    assert weather["score_source_transition_days"] == 1
    assert weather["raw_crossings_on_source_transition_days"] == 1
    assert weather["candidate_suppressed_episodes"] == 2

    alpha_zero = yearly.loc[2021]
    assert alpha_zero["weather_correction_applied_days"] == 0
    assert alpha_zero["score_changed_days"] == 0
    assert alpha_zero["score_source_transition_days"] == 0
    assert alpha_zero["candidate_upward_crossings_comparable"] == 1
    assert alpha_zero["alpha_zero_score_mismatch_days"] == 0

    days = artifacts["suppression_audit_days"]
    assert "field_season" not in days
    assert days["day_key"].str.startswith("day_").all()
    transition = days[
        days["season"].eq(2020) & days["issue_date"].eq(pd.Timestamp("2020-07-06"))
    ].iloc[0]
    assert bool(transition["candidate_upward_crossing_raw"])
    assert not bool(transition["candidate_upward_crossing_comparable"])

    diagnostic = artifacts["suppression_episodes"].query(
        "episode_kind == 'changed_score_suppressed' and season == 2020"
    )
    assert len(diagnostic) == 1
    assert diagnostic.iloc[0]["days"] == 2


def test_suppression_audit_reports_alpha_zero_identity_failure() -> None:
    candidate, baseline = _alarm_states()
    mask = candidate["season"].eq(2021) & candidate["issue_date"].eq(
        pd.Timestamp("2021-07-03")
    )
    candidate.loc[mask, "score"] = 0.71
    summary = audit_suppression_paths(
        candidate,
        baseline,
        candidate=CANDIDATE,
        years=(2020, 2021),
    )["suppression_audit_summary"]
    row = summary.query("aggregation == 'year' and season == 2021").iloc[0]
    assert row["alpha_zero_score_mismatch_days"] == 1


def test_suppression_audit_rejects_compressed_or_unpaired_calendar() -> None:
    candidate, baseline = _alarm_states()
    baseline = baseline.drop(baseline.index[0])
    with pytest.raises(ValueError, match="Non-matching evaluation-day population"):
        audit_suppression_paths(
            candidate,
            baseline,
            candidate=CANDIDATE,
            years=(2020, 2021),
        )
