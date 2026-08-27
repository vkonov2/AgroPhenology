import pandas as pd

from agro_phenology.weather import aggregate_hourly_temperature


def test_missing_hours_are_reported_and_rejected() -> None:
    frame = pd.DataFrame(
        {"time": pd.date_range("2023-04-01", periods=19, freq="h"), "temperature_2m": [12.0] * 19}
    )
    daily = aggregate_hourly_temperature(frame, "Europe/Moscow", minimum_valid_hours=20)
    assert daily.loc[0, "valid_hour_count"] == 19
    assert daily.loc[0, "expected_hour_count"] == 24
    assert daily.loc[0, "coverage_fraction"] == 19 / 24
    assert not bool(daily.loc[0, "accepted"])
    assert daily.loc[0, "completeness_warning"] == "insufficient_valid_hours"


def test_incomplete_day_can_be_explicitly_allowed() -> None:
    frame = pd.DataFrame({"time": pd.date_range("2023-04-01", periods=4, freq="h"), "temperature_2m": [8, 10, 12, 14]})
    daily = aggregate_hourly_temperature(
        frame, "Europe/Moscow", method="min_max_mean", minimum_valid_hours=20, allow_incomplete_days=True
    )
    assert bool(daily.loc[0, "accepted"])
    assert daily.loc[0, "temperature_mean_c"] == 11


def test_utc_timestamps_form_local_days_across_dst() -> None:
    timestamps = pd.date_range("2023-03-25T23:00:00Z", "2023-03-27T21:00:00Z", freq="h")
    frame = pd.DataFrame({"time": timestamps, "temperature_2m": [11.0] * len(timestamps)})
    daily = aggregate_hourly_temperature(frame, "Europe/Berlin", minimum_valid_hours=20)
    assert daily["expected_hour_count"].tolist() == [23, 24]
    assert daily["accepted"].all()
