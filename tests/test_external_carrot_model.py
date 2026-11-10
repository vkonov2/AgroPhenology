from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agro_phenology.external_carrot_model import (
    BASE_TEMPERATURE_C,
    FLIGHT_THRESHOLD_C_DAYS,
    build_carrot_predictions,
)


def _hourly(start="2020-02-01", end="2020-07-01", *, daily_dd=5.0):
    zone = "Europe/Copenhagen"
    times = pd.date_range(pd.Timestamp(start, tz=zone).tz_convert("UTC"),
                          pd.Timestamp(end, tz=zone).tz_convert("UTC"), freq="h", inclusive="left")
    return pd.DataFrame({"valid_time_utc": times, "temperature_2m": BASE_TEMPERATURE_C + daily_dd})


def _local_dates(frame):
    return frame.valid_time_utc.dt.tz_convert("Europe/Copenhagen").dt.tz_localize(None).dt.normalize()


def test_fixed_source_parameters_and_full_local_march_accumulation():
    assert BASE_TEMPERATURE_C == 5.0
    assert FLIGHT_THRESHOLD_C_DAYS == 260.0
    frame = _hourly()
    prediction = build_carrot_predictions(frame, ["2020-03-20"], 2020).iloc[0]
    assert prediction.issued_at == "2020-03-20T08:00:00+01:00"
    assert prediction.feature_cutoff_date == pd.Timestamp("2020-03-15")
    assert prediction.known_cumulative_dd == 75
    assert prediction.persistence_daily_dd == 5
    assert prediction.predicted_event_date == pd.Timestamp("2020-04-21")
    assert prediction.score == 0
    assert prediction.score_status == "computed"


def test_daily_mean_is_taken_before_lower_threshold_not_hourly_clipping():
    frame = _hourly(daily_dd=0)
    # On ordinary days, 12h at 0 C and 12h at 10 C have a daily mean of 5 C.
    # Daily DD is zero; summing hourly positive excess would instead yield 2.5.
    march_days = _local_dates(frame).between("2020-03-01", "2020-03-15")
    hours = frame.valid_time_utc.dt.tz_convert("Europe/Copenhagen").dt.hour
    frame.loc[march_days, "temperature_2m"] = np.where(hours[march_days] < 12, 0., 10.)
    prediction = build_carrot_predictions(frame, ["2020-03-20"], 2020).iloc[0]
    assert prediction.known_cumulative_dd == 0
    assert prediction.persistence_daily_dd == 0
    assert pd.isna(prediction.score)
    assert prediction.score_status == "abstained_zero_persistence_rate"


def test_february_is_not_counted_towards_march_thermal_total():
    frame = _hourly()
    frame.loc[_local_dates(frame).lt(pd.Timestamp("2020-03-01")), "temperature_2m"] = 1000
    prediction = build_carrot_predictions(frame, ["2020-03-20"], 2020).iloc[0]
    assert prediction.known_cumulative_dd == 75
    assert prediction.predicted_event_date == pd.Timestamp("2020-04-21")


def test_no_unpublished_upper_temperature_threshold_is_added():
    prediction = build_carrot_predictions(_hourly(daily_dd=100), ["2020-03-02"], 2020).iloc[0]
    assert prediction.known_cumulative_dd == 0
    assert prediction.persistence_daily_dd == 100
    assert prediction.predicted_event_date == pd.Timestamp("2020-03-03")


@pytest.mark.parametrize("target_date,score", [
    ("2020-04-22", 0), ("2020-04-23", 1), ("2020-04-30", 1), ("2020-05-01", 0),
])
def test_forecast_score_uses_inclusive_three_through_ten_days(target_date, score):
    days = (pd.Timestamp(target_date) - pd.Timestamp("2020-03-01")).days + 1
    frame = _hourly(daily_dd=FLIGHT_THRESHOLD_C_DAYS / days)
    prediction = build_carrot_predictions(frame, ["2020-04-20"], 2020).iloc[0]
    assert prediction.predicted_event_date == pd.Timestamp(target_date)
    assert prediction.score == score


def test_unavailable_recent_gap_and_all_future_temperatures_are_not_predictors():
    frame = _hourly()
    issue = "2020-04-15"
    before = build_carrot_predictions(frame, [issue], 2020)
    after_cutoff = _local_dates(frame).gt(pd.Timestamp("2020-04-10"))
    modified = frame.astype({"temperature_2m": object})
    modified.loc[after_cutoff, "temperature_2m"] = "not available"
    modified.loc[after_cutoff, "provider_latitude"] = 0.125
    modified.loc[after_cutoff, "provider_longitude"] = -20.25
    pd.testing.assert_frame_equal(before, build_carrot_predictions(modified, [issue], 2020))
    pd.testing.assert_frame_equal(before, build_carrot_predictions(frame.loc[~after_cutoff], [issue], 2020))


def test_prefix_and_single_issue_are_identical_despite_later_invalid_metadata():
    frame = _hourly()
    frame["provider_latitude"], frame["provider_longitude"] = 0.125, -20.25
    issues = ["2020-04-15", "2020-04-25"]
    baseline = build_carrot_predictions(frame, issues, 2020)
    modified = frame.copy()
    modified.loc[_local_dates(frame).gt(pd.Timestamp("2020-04-10")), "provider_latitude"] = 0.25
    changed = build_carrot_predictions(modified, issues, 2020)
    pd.testing.assert_frame_equal(baseline.iloc[[0]], changed.iloc[[0]])
    pd.testing.assert_frame_equal(changed.iloc[[0]], build_carrot_predictions(modified, issues[:1], 2020))
    assert changed.score_status.iloc[1] == "abstained_invalid_hourly_input"
    assert pd.isna(changed.score.iloc[1])


def test_later_duplicate_hours_cannot_break_an_earlier_batch_decision():
    frame = _hourly()
    duplicate = frame.loc[_local_dates(frame).eq(pd.Timestamp("2020-04-11"))].iloc[[0]]
    changed = pd.concat([frame, duplicate], ignore_index=True)
    prediction = build_carrot_predictions(changed, ["2020-04-15", "2020-04-25"], 2020)
    pd.testing.assert_frame_equal(prediction.iloc[[0]], build_carrot_predictions(frame, ["2020-04-15"], 2020))
    assert prediction.score_status.iloc[1] == "abstained_invalid_hourly_input"


def test_cutoff_respects_local_midnight_and_declared_lag():
    frame = _hourly()
    # April 10 22:00 UTC is already April 11 in Copenhagen.
    frame.loc[frame.valid_time_utc.eq(pd.Timestamp("2020-04-10T22:00Z")), "temperature_2m"] = np.nan
    unchanged = build_carrot_predictions(frame, ["2020-04-15"], 2020).iloc[0]
    assert unchanged.known_cumulative_dd == 41 * 5
    assert unchanged.score_status == "computed"
    shorter_lag = build_carrot_predictions(frame, ["2020-04-15"], 2020, lag_days=4).iloc[0]
    assert pd.isna(shorter_lag.score)
    assert shorter_lag.score_status == "abstained_missing_cumulative_weather"


@pytest.mark.parametrize("missing", ["row", "temperature", "infinite_temperature", "whole_day"])
def test_missing_history_is_unknown_not_zero_or_an_older_replacement(missing):
    frame = _hourly()
    day = _local_dates(frame).eq(pd.Timestamp("2020-03-02"))
    hour = day & frame.valid_time_utc.dt.hour.eq(12)
    if missing == "row":
        frame = frame.loc[~hour]
    elif missing == "whole_day":
        frame = frame.loc[~day]
    else:
        frame.loc[hour, "temperature_2m"] = np.nan if missing == "temperature" else np.inf
    prediction = build_carrot_predictions(frame, ["2020-03-20"], 2020).iloc[0]
    assert pd.isna(prediction.known_cumulative_dd)
    assert prediction.persistence_daily_dd == 5  # recent seven days are complete
    assert pd.isna(prediction.predicted_event_date)
    assert pd.isna(prediction.score)
    assert prediction.score_status == "abstained_missing_cumulative_weather"


def test_missing_day_in_exact_persistence_window_before_march_abstains():
    frame = _hourly(start="2020-02-25")
    prediction = build_carrot_predictions(frame, ["2020-03-04"], 2020).iloc[0]
    assert prediction.known_cumulative_dd == 0
    assert pd.isna(prediction.persistence_daily_dd)
    assert prediction.score_status == "abstained_missing_persistence_weather"


def test_reached_threshold_returns_past_date_even_if_recent_rate_is_zero():
    frame = _hourly(daily_dd=20)
    frame.loc[_local_dates(frame).ge(pd.Timestamp("2020-03-14")), "temperature_2m"] = 5
    prediction = build_carrot_predictions(frame, ["2020-04-01"], 2020).iloc[0]
    assert prediction.known_cumulative_dd == 260
    assert prediction.persistence_daily_dd == 0
    assert prediction.predicted_event_date == pd.Timestamp("2020-03-13")
    assert prediction.score == 0


def test_zero_recent_rate_below_threshold_abstains_rather_than_predicting_absence():
    prediction = build_carrot_predictions(_hourly(daily_dd=0), ["2020-04-01"], 2020).iloc[0]
    assert prediction.known_cumulative_dd == 0
    assert prediction.persistence_daily_dd == 0
    assert prediction.score_status == "abstained_zero_persistence_rate"
    assert pd.isna(prediction.score)


@pytest.mark.parametrize("start,end,issue,days,offset", [
    ("2020-03-01", "2020-03-30", "2020-04-03", 29, "+02:00"),
    ("2020-03-01", "2020-10-26", "2020-10-30", 239, "+01:00"),
])
def test_dst_mean_is_one_daily_value_despite_twenty_three_or_twenty_five_hours(start, end, issue, days, offset):
    frame = _hourly(start=start, end=end, daily_dd=1)
    prediction = build_carrot_predictions(frame, [issue], 2020).iloc[0]
    assert prediction.known_cumulative_dd == days
    assert prediction.persistence_daily_dd == 1
    assert prediction.issued_at.endswith(offset)
    assert prediction.predicted_event_date == pd.Timestamp("2020-11-15")


def test_one_missing_hour_on_dst_day_is_not_mistaken_for_a_complete_day():
    frame = _hourly(daily_dd=1)
    missing = frame.loc[_local_dates(frame).eq(pd.Timestamp("2020-03-29"))].index[0]
    prediction = build_carrot_predictions(frame.drop(missing), ["2020-04-03"], 2020).iloc[0]
    assert pd.isna(prediction.score)
    assert prediction.score_status == "abstained_missing_cumulative_weather"


def test_timezone_aware_issue_is_normalized_to_local_calendar_date():
    prediction = build_carrot_predictions(_hourly(), [pd.Timestamp("2020-04-14T23:00Z")], 2020).iloc[0]
    assert prediction.issue_date == pd.Timestamp("2020-04-15")
    assert prediction.feature_cutoff_date == pd.Timestamp("2020-04-10")


@pytest.mark.parametrize("name", ["lag_days", "lookback_days"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_noncausal_or_fractional_configuration_is_rejected(name, value):
    with pytest.raises(ValueError, match="positive integer"):
        build_carrot_predictions(_hourly(), ["2020-04-01"], 2020, **{name: value})


def test_ambiguous_issue_season_and_missing_schema_are_rejected():
    with pytest.raises(ValueError, match="unique local"):
        build_carrot_predictions(_hourly(), ["2020-04-01", "2020-04-01T10:00"], 2020)
    with pytest.raises(ValueError, match="missing dates"):
        build_carrot_predictions(_hourly(), [pd.NaT], 2020)
    with pytest.raises(ValueError, match="declared local season"):
        build_carrot_predictions(_hourly(), ["2021-04-01"], 2020)
    with pytest.raises(ValueError, match="positive integer"):
        build_carrot_predictions(_hourly(), ["2020-04-01"], True)
    with pytest.raises(ValueError, match="Missing hourly columns"):
        build_carrot_predictions(pd.DataFrame(), ["2020-04-01"], 2020)


def test_empty_calendar_has_stable_output_without_requiring_weather():
    result = build_carrot_predictions(pd.DataFrame(), [], 2020)
    assert result.empty
    assert result.columns.tolist() == [
        "issue_date", "issued_at", "feature_cutoff_date", "known_cumulative_dd",
        "persistence_daily_dd", "predicted_event_date", "score", "score_status",
    ]


def test_empty_hourly_weather_with_schema_causes_explicit_abstention():
    frame = _hourly().iloc[:0]
    prediction = build_carrot_predictions(frame, ["2020-04-01"], 2020).iloc[0]
    assert prediction.score_status == "abstained_missing_cumulative_weather"
    assert pd.isna(prediction.score)


def test_calendar_before_accumulation_uses_persistence_but_never_accumulates_february():
    frame = _hourly()
    prediction = build_carrot_predictions(frame, ["2020-02-20"], 2020).iloc[0]
    assert prediction.known_cumulative_dd == 0
    assert prediction.persistence_daily_dd == 5
    assert prediction.predicted_event_date == pd.Timestamp("2020-04-21")
