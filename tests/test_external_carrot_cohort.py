"""Synthetic observations only; no real locations or field identifiers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agro_phenology.early_warning_models import Policy, simulate_policy
from agro_phenology.external_carrot_cohort import (
    COORDINATE_SCOPE, SOURCE_ID, UNIT, build_decisions, build_registry,
)


def observations(rows, *, site="synthetic_a", latitude="1", longitude="2"):
    return pd.DataFrame([
        dict(source_id=SOURCE_ID, site_id=site, observation_date=pd.Timestamp(day),
             organism="Psila rosae", value=value, unit=UNIT,
             source_latitude=latitude, source_longitude=longitude)
        for day, value in rows
    ])


def test_first_zero_followed_by_positive_is_one_warnable_registration_not_repeated_events():
    frame = observations([("2020-05-18", 0), ("2020-05-25", 0.4), ("2020-06-01", 2)])
    event = build_registry(frame).iloc[0]
    assert event.cohort_included and event.warnable_first_event
    assert event.first_recorded_event_date == pd.Timestamp("2020-05-25")
    assert event.last_complete_zero_date == pd.Timestamp("2020-05-18")
    assert event.last_assessment_date == pd.Timestamp("2020-06-01")
    assert event.previous_visit_gap_days == 7
    assert event.eligibility_category == "eligible"
    assert event.coordinate_scope == COORDINATE_SCOPE


@pytest.mark.parametrize("delay,eligible", [(3, False), (4, True)])
def test_warning_opportunity_is_measured_from_next_day_issue(delay, eligible):
    start = pd.Timestamp("2020-05-18")
    registry = build_registry(observations([(start, 0), (start + pd.Timedelta(days=delay), 1)]))
    assert bool(registry.iloc[0].warnable_first_event) is eligible
    assert registry.iloc[0].cohort_included


def test_late_entry_is_retained_outside_cohort_and_cannot_be_rescued_by_future_zero():
    frame = observations([("2020-06-01", 0), ("2020-06-08", 1)])
    entry = build_registry(frame).iloc[0]
    assert entry.eligibility_category == "outside_spring_entry"
    assert not entry.cohort_included and not entry.warnable_first_event
    assert len(build_decisions(build_registry(frame))) == 0


def test_positive_at_entry_is_left_censored_even_if_a_later_zero_then_positive_exists():
    frame = observations([("2020-05-01", 1), ("2020-05-08", 0), ("2020-05-15", 1)])
    event = build_registry(frame).iloc[0]
    assert event.entry_date == event.first_recorded_event_date == pd.Timestamp("2020-05-01")
    assert event.positive_at_first_visit
    assert event.eligibility_category == "positive_at_entry_left_censored"
    assert not event.cohort_included and not event.warnable_first_event


def test_missing_count_is_unknown_never_zero_or_valid_followup():
    frame = observations([("2020-05-18", 0), ("2020-05-25", None), ("2020-06-01", None)])
    event = build_registry(frame).iloc[0]
    assert event.cohort_included and not event.warnable_first_event
    assert event.eligibility_category == "right_censored"
    assert event.last_assessment_date == event.last_complete_zero_date == pd.Timestamp("2020-05-18")
    assert event.spring_missing_count_rows == event.series_missing_count_rows == 2
    assert event.spring_measured_rows == 1
    decisions = build_decisions(build_registry(frame))
    assert decisions.service_active.all()
    assert not decisions.evaluation_field_day.any()


def test_missing_first_record_cannot_be_replaced_by_later_convenient_zero():
    frame = observations([("2020-05-18", np.nan), ("2020-05-25", 0), ("2020-06-01", 1)])
    event = build_registry(frame).iloc[0]
    assert event.entry_date == pd.Timestamp("2020-05-18")
    assert event.eligibility_category == "missing_at_entry"
    assert not event.cohort_included and not event.warnable_first_event


def test_july_positive_does_not_remove_spring_zero_series_or_become_a_spring_event():
    frame = observations([("2020-05-18", 0), ("2020-06-15", 0), ("2020-07-01", 1)])
    event = build_registry(frame).iloc[0]
    assert event.cohort_included
    assert event.eligibility_category == "right_censored"
    assert pd.isna(event.first_recorded_event_date)
    assert event.last_assessment_date == pd.Timestamp("2020-06-15")
    assert event.last_complete_zero_date == pd.Timestamp("2020-06-15")
    assert event.spring_observation_rows == 2 and event.series_observation_rows == 3


def test_may31_entry_is_included_but_june1_is_outside():
    a = observations([("2020-05-31", 0), ("2020-06-07", 1)])
    b = observations([("2020-06-01", 0), ("2020-06-08", 1)], site="synthetic_b")
    registry = build_registry(pd.concat([a, b], ignore_index=True)).set_index("site_id")
    assert registry.loc["synthetic_a", "cohort_included"]
    assert not registry.loc["synthetic_b", "cohort_included"]


def test_other_organism_is_not_mixed_into_psila_event_or_series():
    fly = observations([("2020-05-18", 0), ("2020-05-25", 1)])
    moth = observations([("2020-05-01", 2)], site="other_insect")
    moth["organism"] = "Agrotis ipsilon"
    registry = build_registry(pd.concat([fly, moth], ignore_index=True))
    assert len(registry) == 1 and registry.iloc[0].site_id == "synthetic_a"


@pytest.mark.parametrize("change,match", [
    (lambda f: f.assign(source_id="another_dataset"), "dataset"),
    (lambda f: f.assign(unit="count_per_day"), "unit"),
    (lambda f: f.assign(value=-1), "nonnegative"),
    (lambda f: f.assign(value=np.inf), "nonnegative"),
    (lambda f: f.assign(value="not_a_number"), "Invalid catch"),
    (lambda f: f.assign(observation_date=pd.NaT), "Missing observation"),
    (lambda f: f.assign(observation_date="invalid_date"), "Invalid observation"),
    (lambda f: f.assign(site_id=None), "site identifier"),
    (lambda f: f.assign(source_latitude=None), "coordinates"),
    (lambda f: f.assign(source_longitude="unknown"), "coordinates"),
    (lambda f: f.assign(source_latitude=91), "coordinates"),
])
def test_invalid_or_ambiguous_input_is_rejected(change, match):
    with pytest.raises(ValueError, match=match):
        build_registry(change(observations([("2020-05-18", 0)])))


def test_duplicate_same_day_even_with_different_times_is_rejected():
    frame = observations([("2020-05-18 00:00", 0), ("2020-05-18 12:00", 1)])
    with pytest.raises(ValueError, match="Duplicate"):
        build_registry(frame)


def test_spatial_groups_are_stable_for_numeric_coordinate_notation_and_row_order():
    a = observations([("2020-05-18", 0), ("2020-05-25", 1)])
    b = observations([("2021-05-18", 0), ("2021-05-25", 1)], site="synthetic_b", latitude="1.000", longitude="2.0")
    c = observations([("2021-05-18", 0), ("2021-05-25", 1)], site="synthetic_c", latitude="3", longitude="4")
    all_rows = pd.concat([a, b, c], ignore_index=True)
    registry = build_registry(all_rows)
    reversed_registry = build_registry(all_rows.iloc[::-1])
    pd.testing.assert_frame_equal(registry, reversed_registry)
    assert registry.field_season.nunique() == 3
    assert registry.spatial_group.nunique() == 2
    assert registry.spatial_group.iloc[0] == registry.spatial_group.iloc[1]
    assert all(key.startswith("spatial_") for key in registry.spatial_group)


def test_same_site_different_years_remain_separate_and_location_cannot_change_within_series():
    a = observations([("2020-05-18", 0), ("2020-05-25", 1)])
    b = observations([("2021-05-18", 0), ("2021-05-25", 1)])
    registry = build_registry(pd.concat([a, b], ignore_index=True))
    assert len(registry) == registry.field_season.nunique() == 2
    assert registry.spatial_group.nunique() == 1
    a.loc[1, "source_latitude"] = "3"
    with pytest.raises(ValueError, match="inconsistent"):
        build_registry(a)


def test_fixed_grid_and_issue_times_do_not_depend_on_future_observation_or_event_dates():
    registry = build_registry(observations([("2020-05-18", 0), ("2020-05-25", 1)]))
    decisions = build_decisions(registry)
    assert decisions.issue_date.min() == pd.Timestamp("2020-05-19")
    assert decisions.issue_date.max() == pd.Timestamp("2020-06-30")
    assert decisions.issue_date.diff().dropna().eq(pd.Timedelta(days=1)).all()
    assert decisions.issued_at.str.endswith("08:00:00+02:00").all()
    changed = registry.copy()
    changed["first_recorded_event_date"] += pd.Timedelta(days=20)
    changed["last_assessment_date"] += pd.Timedelta(days=30)
    changed["previous_visit_gap_days"] = 100
    again = build_decisions(changed)
    static = ["field_season", "site_id", "entry_date", "issue_date", "issued_at"]
    pd.testing.assert_frame_equal(decisions[static], again[static])
    assert decisions.target_class.eq("unknown").all()
    assert not decisions.target_observable.any()


def test_existing_simulator_gets_no_message_after_event_available_and_no_negative_imputation():
    registry = build_registry(observations([("2020-05-18", 0), ("2020-05-25", 1), ("2020-06-01", 2)]))
    decisions = build_decisions(registry)
    assert decisions.loc[decisions.issue_date.eq("2020-05-25"), "service_active"].all()
    assert not decisions.loc[decisions.issue_date.ge("2020-05-26"), "service_active"].any()
    state = simulate_policy(decisions, pd.Series(1., index=decisions.index), Policy(.5, 7, 15),
                            evaluation_mask=decisions.evaluation_field_day)
    assert state.message_issued.sum() == 1
    assert not state.loc[state.issue_date.ge("2020-05-26"), "message_issued"].any()
    assert not state.loc[state.issue_date.ge("2020-05-26"), "alarm_active"].any()
    assert state.evaluation_scope_day.sum() == 7


def test_registry_duplicates_are_rejected_by_decision_builder():
    registry = build_registry(observations([("2020-05-18", 0), ("2020-05-25", 1)]))
    with pytest.raises(ValueError, match="Duplicate registry"):
        build_decisions(pd.concat([registry, registry], ignore_index=True))
