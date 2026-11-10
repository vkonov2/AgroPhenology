from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agro_phenology.external_cpb_model import (
    HATCH_THRESHOLD_C_DAYS,
    LOWER_THRESHOLD_C,
    UPPER_THRESHOLD_C,
    build_cpb_predictions,
)


def _hourly(start="2020-05-01", end="2020-07-15", *, daily_dd=5.0, timezone="Europe/Berlin"):
    lower = pd.Timestamp(start).tz_localize(timezone).tz_convert("UTC")
    upper = pd.Timestamp(end).tz_localize(timezone).tz_convert("UTC")
    return pd.DataFrame({
        "valid_time_utc": pd.date_range(lower, upper, freq="h", inclusive="left"),
        "temperature_2m": LOWER_THRESHOLD_C + daily_dd,
    })


def _local_dates(frame, timezone="Europe/Berlin"):
    return frame.valid_time_utc.dt.tz_convert(timezone).dt.tz_localize(None).dt.normalize()


def test_fixed_hourly_clipped_prediction_and_score():
    result = build_cpb_predictions(_hourly(), ["2020-06-08"], "2020-06-01").iloc[0]
    assert result.issue_date == pd.Timestamp("2020-06-08")
    assert result.issued_at == "2020-06-08T08:00:00+02:00"
    assert result.feature_cutoff_date == pd.Timestamp("2020-06-03")
    assert result.known_cumulative_dd == pytest.approx(15)
    assert result.persistence_daily_dd == pytest.approx(5)
    assert result.predicted_event_date == pd.Timestamp("2020-06-14")
    assert result.score == 1
    assert result.score_status == "computed"


@pytest.mark.parametrize("prediction,score", [
    ("2020-06-22", 0), ("2020-06-23", 1), ("2020-06-30", 1), ("2020-07-01", 0),
])
def test_score_uses_exact_inclusive_three_to_ten_day_window(prediction, score):
    number_of_days = (pd.Timestamp(prediction) - pd.Timestamp("2020-06-01")).days + 1
    frame = _hourly(daily_dd=HATCH_THRESHOLD_C_DAYS / number_of_days)
    result = build_cpb_predictions(frame, ["2020-06-20"], "2020-06-01").iloc[0]
    assert result.predicted_event_date == pd.Timestamp(prediction)
    assert result.score == score


def test_temperature_is_clipped_at_fixed_upper_threshold():
    frame = _hourly(daily_dd=100)
    result = build_cpb_predictions(frame, ["2020-06-03"], "2020-06-01").iloc[0]
    assert result.known_cumulative_dd == 0
    assert result.persistence_daily_dd == pytest.approx(UPPER_THRESHOLD_C - LOWER_THRESHOLD_C)
    assert result.predicted_event_date == pd.Timestamp("2020-06-04")


def test_no_accumulation_before_biofix_when_cutoff_is_earlier():
    result = build_cpb_predictions(_hourly(daily_dd=10), ["2020-06-04"], "2020-06-01").iloc[0]
    assert result.feature_cutoff_date == pd.Timestamp("2020-05-30")
    assert result.known_cumulative_dd == 0
    assert result.predicted_event_date == pd.Timestamp("2020-06-07")
    assert result.score == 1


def test_future_weather_is_never_used_even_for_forecast_gap():
    frame = _hourly()
    result = build_cpb_predictions(frame, ["2020-06-08"], "2020-06-01")
    changed = frame.copy()
    future = _local_dates(changed).gt(pd.Timestamp("2020-06-03"))
    changed.loc[future, "temperature_2m"] = -1000
    changed.loc[future, "provider_latitude"] = 5
    changed.loc[future, "provider_longitude"] = 5
    pd.testing.assert_frame_equal(result, build_cpb_predictions(changed, ["2020-06-08"], "2020-06-01"))
    # Removing or appending unavailable temperatures gives the same decision.
    pd.testing.assert_frame_equal(result, build_cpb_predictions(frame.loc[~future], ["2020-06-08"], "2020-06-01"))


def test_future_temperature_need_not_even_be_numeric():
    frame = _hourly()
    result = build_cpb_predictions(frame, ["2020-06-08"], "2020-06-01")
    changed = frame.astype({"temperature_2m": object})
    changed.loc[_local_dates(changed).gt(pd.Timestamp("2020-06-03")), "temperature_2m"] = "unavailable"
    pd.testing.assert_frame_equal(result, build_cpb_predictions(changed, ["2020-06-08"], "2020-06-01"))


def test_lag_cuts_off_full_local_day_not_utc_day_or_issue_time():
    frame = _hourly(daily_dd=5)
    # June 3 22:00 UTC already belongs to June 4 locally.
    frame.loc[frame.valid_time_utc.eq(pd.Timestamp("2020-06-03T22:00Z")), "temperature_2m"] = np.nan
    result = build_cpb_predictions(frame, ["2020-06-08"], "2020-06-01").iloc[0]
    assert result.known_cumulative_dd == pytest.approx(15)
    assert result.score == 1
    changed_lag = build_cpb_predictions(frame, ["2020-06-08"], "2020-06-01", lag_days=4).iloc[0]
    assert pd.isna(changed_lag.score)


@pytest.mark.parametrize("missing", ["row", "temperature", "infinite_temperature", "whole_day"])
def test_incomplete_cutoff_is_unknown_not_zero_and_no_old_day_replacement(missing):
    frame = _hourly()
    cutoff = _local_dates(frame).eq(pd.Timestamp("2020-06-03"))
    one_hour = cutoff & frame.valid_time_utc.dt.hour.eq(12)
    if missing == "row":
        frame = frame.loc[~one_hour]
    elif missing == "whole_day":
        frame = frame.loc[~cutoff]
    else:
        frame.loc[one_hour, "temperature_2m"] = np.nan if missing == "temperature" else np.inf
    result = build_cpb_predictions(frame, ["2020-06-08"], "2020-06-01").iloc[0]
    assert pd.isna(result.known_cumulative_dd)
    assert pd.isna(result.persistence_daily_dd)
    assert pd.isna(result.predicted_event_date)
    assert pd.isna(result.score)
    assert result.score_status == "abstained_missing_cumulative_weather"


def test_missing_older_cumulative_day_is_not_silently_filled():
    frame = _hourly(daily_dd=1)
    frame = frame.loc[~_local_dates(frame).eq(pd.Timestamp("2020-06-02"))]
    result = build_cpb_predictions(frame, ["2020-06-20"], "2020-06-01").iloc[0]
    assert pd.isna(result.known_cumulative_dd)
    assert result.persistence_daily_dd == pytest.approx(1)
    assert pd.isna(result.score)


def test_missing_persistence_weather_is_not_filled_when_biofix_is_after_cutoff():
    frame = _hourly(start="2020-05-29", daily_dd=10)
    result = build_cpb_predictions(frame, ["2020-06-04"], "2020-06-01").iloc[0]
    assert result.known_cumulative_dd == 0
    assert pd.isna(result.persistence_daily_dd)
    assert pd.isna(result.score)
    assert result.score_status == "abstained_missing_persistence_weather"


def test_biofix_is_available_only_on_next_local_day():
    result = build_cpb_predictions(_hourly(), ["2020-05-31", "2020-06-01", "2020-06-02"], "2020-06-01")
    assert result.score.iloc[:2].isna().all()
    assert result.known_cumulative_dd.iloc[:2].isna().all()
    assert result.persistence_daily_dd.iloc[:2].isna().all()
    assert result.score_status.iloc[:2].eq("abstained_biofix_not_yet_available").all()
    assert pd.notna(result.score.iloc[2])


def test_unknown_or_not_yet_known_biofix_does_not_read_weather():
    missing = build_cpb_predictions(pd.DataFrame(), ["2020-06-01"], None).iloc[0]
    assert pd.isna(missing.score)
    assert missing.score_status == "abstained_biofix_missing"
    unknown = build_cpb_predictions(pd.DataFrame(), ["2020-06-01"], "2020-06-01").iloc[0]
    assert unknown.score_status == "abstained_biofix_not_yet_available"


def test_zero_persistence_rate_abstains_rather_than_predicting_no_event():
    result = build_cpb_predictions(_hourly(daily_dd=0), ["2020-06-08"], "2020-06-01").iloc[0]
    assert result.persistence_daily_dd == 0
    assert result.known_cumulative_dd == 0
    assert pd.isna(result.predicted_event_date) and pd.isna(result.score)
    assert result.score_status == "abstained_zero_persistence_rate"


def test_known_past_threshold_returns_actual_crossing_date_and_zero_score():
    frame = _hourly(daily_dd=10)
    result = build_cpb_predictions(frame, ["2020-06-15"], "2020-06-01").iloc[0]
    assert result.known_cumulative_dd == pytest.approx(100)
    assert result.predicted_event_date == pd.Timestamp("2020-06-07")
    assert result.score == 0


def test_known_past_threshold_needs_no_positive_future_rate():
    frame = _hourly(daily_dd=20)
    frame.loc[_local_dates(frame).ge(pd.Timestamp("2020-06-05")), "temperature_2m"] = LOWER_THRESHOLD_C
    result = build_cpb_predictions(frame, ["2020-06-20"], "2020-06-01").iloc[0]
    assert result.persistence_daily_dd == 0
    assert result.predicted_event_date == pd.Timestamp("2020-06-04")
    assert result.score == 0


@pytest.mark.parametrize("cutoff,expected_hours,offset", [
    ("2020-03-29", 167, "+02:00"), ("2020-10-25", 169, "+01:00"),
])
def test_dst_uses_complete_actual_local_hours_with_fixed_hourly_denominator(cutoff, expected_hours, offset):
    cutoff = pd.Timestamp(cutoff)
    biofix = cutoff - pd.Timedelta(days=6)
    frame = _hourly(str(biofix.date()), str((cutoff + pd.Timedelta(days=1)).date()), daily_dd=1)
    result = build_cpb_predictions(frame, [cutoff + pd.Timedelta(days=5)], biofix).iloc[0]
    assert result.known_cumulative_dd == pytest.approx(expected_hours / 24)
    assert result.persistence_daily_dd == pytest.approx(expected_hours / (24 * 7))
    assert result.issued_at.endswith(offset)
    assert pd.notna(result.score)


def test_aware_dates_are_interpreted_in_local_timezone():
    result = build_cpb_predictions(_hourly(), [pd.Timestamp("2020-06-07T23:30Z")], pd.Timestamp("2020-05-31T23:30Z")).iloc[0]
    assert result.issue_date == pd.Timestamp("2020-06-08")
    assert result.known_cumulative_dd == pytest.approx(15)


@pytest.mark.parametrize("name", ["lag_days", "lookback_days"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_noncausal_or_fractional_configuration_is_rejected(name, value):
    with pytest.raises(ValueError, match="positive integer"):
        build_cpb_predictions(_hourly(), ["2020-06-08"], "2020-06-01", **{name: value})


def test_duplicate_timestamps_and_multiple_grid_cells_are_rejected():
    frame = _hourly()
    duplicate = pd.concat([frame, frame.iloc[[800]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate hourly"):
        build_cpb_predictions(duplicate, ["2020-06-08"], "2020-06-01")
    frame["provider_latitude"], frame["provider_longitude"] = 52, 13
    frame.loc[800, "provider_latitude"] = 53
    with pytest.raises(ValueError, match="multiple provider"):
        build_cpb_predictions(frame, ["2020-06-08"], "2020-06-01")


def test_ambiguous_calendar_and_missing_hourly_columns_are_rejected():
    with pytest.raises(ValueError, match="unique local"):
        build_cpb_predictions(_hourly(), ["2020-06-08", "2020-06-08T10:00"], "2020-06-01")
    with pytest.raises(ValueError, match="missing dates"):
        build_cpb_predictions(_hourly(), [pd.NaT], "2020-06-01")
    with pytest.raises(ValueError, match="Missing hourly columns"):
        build_cpb_predictions(pd.DataFrame(), ["2020-06-08"], "2020-06-01")


def test_empty_calendar_has_stable_output_columns():
    result = build_cpb_predictions(pd.DataFrame(), [], "2020-06-01")
    assert result.empty
    assert result.columns.tolist() == [
        "issue_date", "issued_at", "feature_cutoff_date", "known_cumulative_dd",
        "persistence_daily_dd", "predicted_event_date", "score", "score_status",
    ]
