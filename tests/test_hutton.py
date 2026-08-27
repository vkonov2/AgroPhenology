from __future__ import annotations

import numpy as np
import pandas as pd

from agro_phenology.late_blight import (
    HuttonConfig,
    classify_hutton_days,
    classify_hutton_periods,
    extract_hutton_episodes,
)


def hourly_day(day: str, temperature: float = 10.0, humidity: float = 90.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range(day, periods=24, freq="h"),
            "temperature_2m": [temperature] * 24,
            "relative_humidity_2m": [humidity] * 24,
        }
    )


def test_hutton_inclusive_boundaries_and_two_day_period() -> None:
    frame = pd.concat([hourly_day("2026-07-01"), hourly_day("2026-07-02")], ignore_index=True)
    # Exactly six qualifying hours per day; the remaining hours are below RH threshold.
    frame["relative_humidity_2m"] = [90.0] * 6 + [80.0] * 18 + [90.0] * 6 + [80.0] * 18
    daily = classify_hutton_days(frame, "Europe/Moscow")
    periods = classify_hutton_periods(daily)
    assert daily["day_status"].tolist() == ["pass", "pass"]
    assert periods.iloc[0]["period_status"] == "pass"


def test_hutton_temperature_and_humidity_failures() -> None:
    temperature_fail = hourly_day("2026-07-01", temperature=12.0)
    temperature_fail.loc[3, "temperature_2m"] = 9.9
    humidity_fail = hourly_day("2026-07-02", temperature=12.0, humidity=80.0)
    humidity_fail.loc[:4, "relative_humidity_2m"] = 90.0
    daily = classify_hutton_days(pd.concat([temperature_fail, humidity_fail]), "Europe/Moscow")
    assert daily.iloc[0]["temperature_status"] == "fail"
    assert daily.iloc[1]["humidity_status"] == "fail"
    assert daily["day_status"].tolist() == ["fail", "fail"]


def test_hutton_missing_hours_are_indeterminate_when_they_can_change_result() -> None:
    frame = hourly_day("2026-07-01", temperature=12.0, humidity=80.0)
    frame.loc[:4, "relative_humidity_2m"] = 90.0
    frame.loc[5:6, "relative_humidity_2m"] = np.nan
    frame.loc[23, "temperature_2m"] = np.nan
    result = classify_hutton_days(frame, "Europe/Moscow").iloc[0]
    assert result["temperature_status"] == "indeterminate"
    assert result["humidity_status"] == "indeterminate"
    assert result["day_status"] == "indeterminate"


def test_hutton_period_is_fail_dominant_for_logical_and() -> None:
    daily = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-07-01").date(), pd.Timestamp("2026-07-02").date()],
            "day_status": ["fail", "indeterminate"],
        }
    )
    assert classify_hutton_periods(daily).iloc[0]["period_status"] == "fail"


def test_overlapping_hutton_pairs_are_merged_into_one_episode() -> None:
    frame = pd.concat([hourly_day(f"2026-07-0{day}") for day in range(1, 4)], ignore_index=True)
    daily = classify_hutton_days(frame, "Europe/Moscow")
    episodes = extract_hutton_episodes(daily)
    assert len(episodes) == 1
    assert episodes.iloc[0]["qualified_day_count"] == 3
    assert episodes.iloc[0]["hutton_pair_count"] == 2


def test_half_hour_intervals_sum_duration_not_rows() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-07-01", periods=48, freq="30min"),
            "temperature_2m": [12.0] * 48,
            "relative_humidity_2m": [90.0] * 12 + [80.0] * 36,
        }
    )
    result = classify_hutton_days(
        frame, "Europe/Moscow", HuttonConfig(interval_hours=0.5)
    ).iloc[0]
    assert result["high_humidity_hours"] == 6.0
    assert result["day_status"] == "pass"


def test_out_of_range_humidity_is_missing_not_clipped() -> None:
    frame = hourly_day("2026-07-01", temperature=12.0, humidity=120.0)
    result = classify_hutton_days(frame, "Europe/Moscow").iloc[0]
    assert result["humidity_valid_hours"] == 0
    assert result["humidity_status"] == "indeterminate"
