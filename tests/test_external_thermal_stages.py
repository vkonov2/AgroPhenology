from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agro_phenology.external_thermal_stages import (
    MODEL_SPECS, daily_thermal_units, model_registry, predict_stage_dates,
)


def _weather(start="2020-03-01", end="2020-05-01", temperature=16.11111111111111):
    dates = pd.date_range(start, end)
    return pd.DataFrame(dict(date=dates, tmin_c=temperature, tmax_c=temperature,
                             available_at=(dates+pd.Timedelta(days=1)).tz_localize("UTC")))


def _predict(frame=None, *, model="cabbage_maggot_uw", anchor="2020-03-01", issue="2020-03-12T08:00Z", **kwargs):
    spec = MODEL_SPECS[model]
    return predict_stage_dates(
        _weather() if frame is None else frame, model, anchor, issue,
        biofix_kind=kwargs.pop("biofix_kind", spec.biofix_kind),
        biofix_available_at=kwargs.pop("biofix_available_at", "2020-03-02T00:00Z"), **kwargs,
    )


def test_published_milestones_and_distinct_generation_endpoints_are_preserved():
    codling = MODEL_SPECS["codling_ncstate_thresholds"]
    assert codling.lower_c == 10
    assert codling.upper_c == pytest.approx(31.11111111111111)
    assert [m.published_value for m in codling.milestones] == [350, 1050, 1300, 2600]
    assert [m.threshold_c_days for m in codling.milestones] == pytest.approx(np.array([350, 1050, 1300, 2600])*5/9)
    delia = MODEL_SPECS["cabbage_maggot_uw"]
    assert delia.lower_c == pytest.approx(55/9)
    assert delia.upper_c is None
    assert [m.published_value for m in delia.milestones] == [300, 1476, 2652]
    assert all("appearance" in m.stage_id and "peak" not in m.stage_id for m in delia.milestones)
    assert MODEL_SPECS["diamondback_harcourt_cabbage"].milestones[0].threshold_c_days == 283
    assert MODEL_SPECS["diamondback_butts_brussels_sprouts"].milestones[0].threshold_c_days == 293
    assert MODEL_SPECS["diamondback_harcourt_cabbage"].host != MODEL_SPECS["diamondback_butts_brussels_sprouts"].host
    assert all(r["biological_validation_status"].startswith("blocked_") for r in model_registry())


def test_verified_integrator_has_no_invented_physical_upper_cutoff_for_delia():
    assert daily_thermal_units([50], [50], "cabbage_maggot_uw")[0] == pytest.approx(50-55/9)
    assert daily_thermal_units([50], [50], "codling_ncstate_thresholds")[0] == pytest.approx((88-50)*5/9)
    assert np.isnan(daily_thermal_units([np.nan], [20], "cabbage_maggot_uw")[0])


def test_constant_temperature_gives_analytical_crossing_and_window():
    # Ten C DD/day from March1; 166 2/3 requires seventeen days, March17.
    result = _predict()
    first = result.iloc[0]
    assert first.known_cumulative_c_days == pytest.approx(110)
    assert first.forecast_daily_c_days == pytest.approx(10)
    assert first.predicted_date == pd.Timestamp("2020-03-17")
    assert first.score == 1
    assert first.feature_cutoff_date == pd.Timestamp("2020-03-11")


@pytest.mark.parametrize("anchor,kind,available,reason", [
    (None, "observed_ground_thaw", "2020-03-02T00:00Z", "abstained_biofix_missing"),
    ("2020-03-01", None, "2020-03-02T00:00Z", "abstained_biofix_kind_missing_or_incompatible"),
    ("2020-03-01", "calendar_January_1", "2020-03-02T00:00Z", "abstained_biofix_kind_missing_or_incompatible"),
    ("2020-03-01", "observed_ground_thaw", None, "abstained_biofix_availability_unknown"),
    ("2020-03-01", "observed_ground_thaw", "2020-03-13T00:00Z", "abstained_biofix_not_yet_available"),
    ("2020-03-12", "observed_ground_thaw", "2020-03-12T00:00Z", "abstained_biofix_not_yet_available"),
    ("2020-03-01", "observed_ground_thaw", "2020-02-01T00:00Z", "abstained_biofix_published_before_observation_date"),
])
def test_missing_incompatible_or_unavailable_biofix_abstains(anchor, kind, available, reason):
    frame = _predict(anchor=anchor, biofix_kind=kind, biofix_available_at=available)
    assert frame.status.eq(reason).all()
    assert frame.score.isna().all()


def test_adult_trap_biofix_cannot_replace_diamondback_egg_cohort():
    frame = _predict(model="diamondback_harcourt_cabbage", biofix_kind="observed_adult_trap_biofix")
    assert frame.status.eq("abstained_biofix_kind_missing_or_incompatible").all()


def test_biofix_is_included_but_pre_biofix_weather_only_sets_persistence():
    frame = _weather()
    first = _predict(frame, anchor="2020-03-10", biofix_available_at="2020-03-11T00:00Z").iloc[0]
    assert first.known_cumulative_c_days == pytest.approx(20)
    assert first.predicted_date == pd.Timestamp("2020-03-26")


def test_future_and_late_available_temperatures_metadata_and_duplicates_do_not_change_decision():
    base = _weather()
    original = _predict(base)
    future = base.date.ge("2020-03-12")
    changed = base.copy()
    changed["tmin_c"] = changed.tmin_c.astype(object)
    changed.loc[future, "tmin_c"] = "invalid_future_temperature"
    changed.loc[future, "tmax_c"] = -9999
    changed = pd.concat([changed, changed.loc[future]], ignore_index=True)
    pd.testing.assert_frame_equal(original, _predict(changed))
    # It is historical by valid date, but only published after this decision.
    late_revision = pd.DataFrame(dict(date=[pd.Timestamp("2020-03-05")], tmin_c=["bad"], tmax_c=[-10],
                                     available_at=[pd.Timestamp("2020-03-20", tz="UTC")]))
    pd.testing.assert_frame_equal(original, _predict(pd.concat([base, late_revision], ignore_index=True)))


def test_incomplete_current_day_is_excluded_even_if_marked_available():
    base = _weather()
    original = _predict(base)
    current = pd.DataFrame(dict(date=[pd.Timestamp("2020-03-12")], tmin_c=[-100], tmax_c=[-200],
                               available_at=[pd.Timestamp("2020-03-12T07:00Z")]))
    pd.testing.assert_frame_equal(original, _predict(pd.concat([base, current], ignore_index=True)))


def test_availability_lag_forecasts_gap_without_reading_real_recent_weather():
    frame = _weather()
    frame["available_at"] += pd.Timedelta(days=4)
    original = _predict(frame)
    assert original.feature_cutoff_date.eq(pd.Timestamp("2020-03-07")).all()
    changed = frame.copy()
    recent = changed.date.gt("2020-03-07")
    changed.loc[recent, ["tmin_c", "tmax_c"]] = 100
    pd.testing.assert_frame_equal(original, _predict(changed))
    assert original.predicted_date.iloc[0] == pd.Timestamp("2020-03-17")


@pytest.mark.parametrize("kind", ["missing_row", "missing_min", "invalid_range", "duplicate"])
def test_unknown_or_invalid_known_history_is_not_treated_as_zero(kind):
    frame = _weather()
    target = frame.date.eq("2020-03-05")
    if kind == "missing_row":
        frame = frame.loc[~target]
    elif kind == "missing_min":
        frame.loc[target, "tmin_c"] = np.nan
    elif kind == "invalid_range":
        frame.loc[target, "tmin_c"] = 100
    else:
        frame = pd.concat([frame, frame.loc[target]], ignore_index=True)
    result = _predict(frame)
    assert result.status.str.startswith("abstained_").all()
    assert result.score.isna().all()


def test_unknown_publication_time_is_unavailable_not_invented():
    frame = _weather()
    frame["available_at"] = pd.NaT
    result = _predict(frame)
    assert result.status.eq("abstained_no_weather_available").all()


def test_forecast_published_before_valid_day_is_not_observed_past_weather():
    frame = _weather()
    frame["available_at"] -= pd.Timedelta(days=2)
    result = _predict(frame)
    assert result.status.eq("abstained_weather_published_before_day_complete").all()
    assert result.score.isna().all()


def test_only_seven_consecutive_available_days_define_persistence():
    frame = _weather(start="2020-02-01")
    frame = frame.loc[~frame.date.eq("2020-03-08")]
    # Missing rate day is before this recent biofix; cannot backfill by February.
    result = _predict(frame, anchor="2020-03-10", biofix_available_at="2020-03-11T00:00Z")
    assert result.status.eq("abstained_missing_persistence_weather").all()
    assert result.forecast_daily_c_days.isna().all()


def test_zero_rate_abstains_until_threshold_but_known_crossing_returns_past_date():
    frame = _weather(temperature=5)
    result = _predict(frame)
    assert result.status.eq("abstained_zero_persistence_rate").all()
    frame = _weather(temperature=16.11111111111111)
    frame.loc[frame.date.ge("2020-03-20"), ["tmin_c", "tmax_c"]] = 5
    result = _predict(frame, issue="2020-03-30T08:00Z")
    assert result.status.iloc[0] == "computed_past_crossing"
    assert result.predicted_date.iloc[0] == pd.Timestamp("2020-03-17")
    assert result.score.iloc[0] == 0
    assert result.status.iloc[1] == "abstained_zero_persistence_rate"


@pytest.mark.parametrize("issue,score", [("2020-03-06T08:00Z", 0), ("2020-03-07T08:00Z", 1),
                                         ("2020-03-14T08:00Z", 1), ("2020-03-15T08:00Z", 0)])
def test_inclusive_three_to_ten_day_window(issue, score):
    # Supply pre-biofix rate weather so March6 has a complete seven-day rate.
    row = _predict(_weather(start="2020-02-20"), issue=issue).iloc[0]
    assert row.predicted_date == pd.Timestamp("2020-03-17")
    assert row.score == score


def test_model_and_time_schema_fail_closed():
    with pytest.raises(ValueError, match="Unknown thermal model"):
        predict_stage_dates(_weather(), "not_a_model", "2020-03-01", "2020-03-12T08:00Z")
    with pytest.raises(ValueError, match="timezone-aware"):
        _predict(issue="2020-03-12T08:00")
    with pytest.raises(ValueError, match="timezone-aware"):
        _predict(biofix_available_at="2020-03-02")
    with pytest.raises(ValueError, match="Missing daily weather columns"):
        _predict(_weather().drop(columns="tmin_c"))


def test_availability_column_can_be_named_explicitly_without_guessing_semantics():
    frame = _weather().rename(columns={"available_at": "received_at"})
    result = _predict(frame, availability_column="received_at")
    pd.testing.assert_frame_equal(result, _predict())


def test_known_anchor_before_weather_end_is_forecast_from_anchor_not_from_cutoff():
    # Lagged source has no days at/after the observed anchor yet, so DD starts
    # on the anchor rather than filling thermal development before the anchor.
    frame = _weather(start="2020-02-20")
    frame["available_at"] += pd.Timedelta(days=4)
    row = _predict(frame, anchor="2020-03-10", biofix_available_at="2020-03-11T00:00Z").iloc[0]
    assert row.feature_cutoff_date == pd.Timestamp("2020-03-07")
    assert row.known_cumulative_c_days == 0
    assert row.predicted_date == pd.Timestamp("2020-03-26")
