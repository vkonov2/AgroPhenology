from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agro_phenology.late_blight import (
    PolyakovConfig,
    aggregate_polyakov_daily,
    classify_polyakov_windows,
    extract_polyakov_episodes,
    merge_polyakov_weather_sources,
)
from agro_phenology.late_blight_validation import (
    summarize_polyakov_activation_interval,
    summarize_polyakov_observation,
)


def daily_frame(days: int = 17) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [value.date() for value in pd.date_range("2026-06-01", periods=days, freq="D")],
            "temperature_mean_c": [16.0] * days,
            "relative_humidity_mean_pct": [80.0] * days,
            "precipitation_sum_mm": [2.0] * days,
            "accepted": [True] * days,
        }
    )


def test_polyakov_threshold_and_persistence_window() -> None:
    classified = classify_polyakov_windows(daily_frame(), "2026-06-01")
    first_critical = classified[classified["status"] == "CRITICAL_CONDITIONS"].iloc[0]
    expected = classified[classified["status"] == "OUTBREAK_EXPECTED"].iloc[0]
    assert first_critical["date"] == pd.Timestamp("2026-06-10").date()
    assert expected["date"] == pd.Timestamp("2026-06-16").date()
    assert expected["expected_manifestation_start"] == pd.Timestamp("2026-06-16").date()
    episodes = extract_polyakov_episodes(classified)
    assert bool(episodes.iloc[0]["persistence_confirmed"])


def test_polyakov_incomplete_ten_day_window_is_not_calculated() -> None:
    daily = daily_frame(10)
    daily.loc[4, "accepted"] = False
    result = classify_polyakov_windows(daily, "2026-06-01")
    assert result.iloc[-1]["status"] == "INSUFFICIENT_DATA"
    assert pd.isna(result.iloc[-1]["critical"])


def test_polyakov_missing_phenophase_remains_not_active() -> None:
    result = classify_polyakov_windows(daily_frame(10), None)
    assert set(result["status"]) == {"NOT_EVALUABLE_MISSING_PHENOPHASE"}


def test_polyakov_precipitation_missing_blocks_daily_input() -> None:
    hourly = pd.DataFrame(
        {
            "time": pd.date_range("2026-06-01", periods=24, freq="h"),
            "temperature_2m": [16.0] * 24,
            "relative_humidity_2m": [80.0] * 24,
            "precipitation": [np.nan] * 24,
        }
    )
    daily = aggregate_polyakov_daily(hourly, "Europe/Moscow")
    assert not bool(daily.iloc[0]["accepted"])
    assert pd.isna(daily.iloc[0]["precipitation_sum_mm"])


def test_polyakov_requires_complete_local_day_by_default() -> None:
    hourly = pd.DataFrame(
        {
            "time": pd.date_range("2026-06-01", periods=24, freq="h"),
            "temperature_2m": [16.0] * 24,
            "relative_humidity_2m": [80.0] * 24,
            "precipitation": [0.0] * 20 + [np.nan] * 4,
        }
    )
    daily = aggregate_polyakov_daily(hourly, "Europe/Moscow")
    assert daily.iloc[0]["required_valid_hours"] == 24
    assert not bool(daily.iloc[0]["accepted"])


def test_polyakov_source_merge_keeps_land_temperature_and_era5_precipitation() -> None:
    times = pd.date_range("2026-06-01", periods=3, freq="h")
    land = pd.DataFrame(
        {
            "time": times,
            "temperature_2m": [15.0, 16.0, 17.0],
            "relative_humidity_2m": [80.0, 81.0, 82.0],
        }
    )
    era5 = pd.DataFrame(
        {
            "time": times,
            "temperature_2m": [99.0, 99.0, 99.0],
            "precipitation": [0.0, 1.2, 0.3],
        }
    )
    merged = merge_polyakov_weather_sources(land, era5)
    assert merged["temperature_2m"].tolist() == [15.0, 16.0, 17.0]
    assert merged["relative_humidity_2m"].tolist() == [80.0, 81.0, 82.0]
    assert merged["precipitation"].tolist() == [0.0, 1.2, 0.3]


def test_polyakov_source_merge_rejects_timestamp_mismatch() -> None:
    land = pd.DataFrame(
        {
            "time": ["2026-06-01T00:00"],
            "temperature_2m": [15.0],
            "relative_humidity_2m": [80.0],
        }
    )
    era5 = pd.DataFrame(
        {
            "time": ["2026-06-01T01:00"],
            "temperature_2m": [99.0],
            "precipitation": [0.0],
        }
    )
    with pytest.raises(ValueError, match="timestamp grids"):
        merge_polyakov_weather_sources(land, era5)


def test_polyakov_candidate_resets_when_criterion_breaks() -> None:
    daily = daily_frame(20)
    daily.loc[12, "relative_humidity_mean_pct"] = 10.0
    result = classify_polyakov_windows(daily, "2026-06-01")
    assert result.iloc[12]["status"] == "LOW_WEATHER_RISK"


def test_polyakov_threshold_boundaries_are_inclusive() -> None:
    daily = daily_frame(10)
    daily["temperature_mean_c"] = 13.0
    daily["relative_humidity_mean_pct"] = 75.0
    daily["precipitation_sum_mm"] = 2.0
    lower = classify_polyakov_windows(daily, "2026-06-01")
    assert bool(lower.iloc[-1]["critical"])
    daily["temperature_mean_c"] = 20.0
    upper = classify_polyakov_windows(daily, "2026-06-01")
    assert bool(upper.iloc[-1]["critical"])


def test_missing_required_weather_does_not_emit_false_high_risk() -> None:
    daily = daily_frame(10)
    daily["accepted"] = False
    classified = classify_polyakov_windows(daily, "2026-06-01")
    summary = summarize_polyakov_observation("obs", "2026-06-10", "detected", classified, "manual")
    assert not summary["polyakov_evaluable"]
    assert pd.isna(summary["high_weather_risk_on_observation_date"])


def test_unconfirmed_critical_window_is_not_a_manifestation_match() -> None:
    classified = classify_polyakov_windows(daily_frame(10), "2026-06-01")
    observed = classified.iloc[-1]["date"]
    classified.loc[classified.index[-1], "expected_manifestation_start"] = observed
    classified.loc[classified.index[-1], "expected_manifestation_end"] = observed
    summary = summarize_polyakov_observation(
        "obs",
        observed,
        "detected",
        classified,
        "manual",
        exclude_observation_day=False,
    )
    assert summary["polyakov_evaluable"]
    assert not summary["manifestation_window_match"]
    assert summary["pilot_association"] == "DETECTION_WITHOUT_MODEL_SIGNAL"


def test_incomplete_weather_at_observation_is_not_treated_as_no_signal() -> None:
    classified = classify_polyakov_windows(daily_frame(11), "2026-06-01")
    classified.loc[classified.index[-1], "accepted"] = False
    classified.loc[classified.index[-1], "status"] = "INSUFFICIENT_DATA"
    summary = summarize_polyakov_observation(
        "obs",
        "2026-06-11",
        "not_detected",
        classified,
        "manual",
        exclude_observation_day=False,
    )
    assert not summary["polyakov_evaluable"]
    assert summary["polyakov_reason"] == "incomplete_weather_at_cutoff"
    assert summary["pilot_association"] == "NOT_EVALUABLE_INCOMPLETE_WEATHER"


def test_observation_day_cannot_confirm_polyakov_signal_when_excluded() -> None:
    classified = classify_polyakov_windows(daily_frame(17), "2026-06-01")
    summary = summarize_polyakov_observation(
        "obs",
        "2026-06-16",
        "detected",
        classified,
        "manual",
    )
    assert summary["weather_cutoff_date"] == pd.Timestamp("2026-06-15").date()
    assert summary["polyakov_evaluable"]
    assert not summary["manifestation_window_match"]
    assert not summary["high_weather_risk_at_cutoff"]
    assert summary["pilot_association"] == "DETECTION_WITHOUT_MODEL_SIGNAL"


def test_full_activation_interval_is_aggregated_without_majority_voting() -> None:
    dates = pd.date_range("2026-06-15", "2026-06-30", freq="D")
    scenarios = pd.DataFrame(
        {
            "activation_rule": [
                f"author_confirmed_regional_phenophase_interval_{value:%Y-%m-%d}"
                for value in dates
            ],
            "polyakov_evaluable": [True] * len(dates),
            "polyakov_reason": [""] * len(dates),
            "manifestation_window_match": [False] * len(dates),
            "high_weather_risk_on_observation_date": [pd.NA] * len(dates),
            "weather_status_on_observation_date": ["LOW_WEATHER_RISK"] * len(dates),
            "weather_cutoff_date": [pd.Timestamp("2026-07-23").date()] * len(dates),
            "observation_day_excluded": [True] * len(dates),
            "weather_status_at_cutoff": ["LOW_WEATHER_RISK"] * len(dates),
            "high_weather_risk_at_cutoff": [False] * len(dates),
            "nearest_prior_manifestation_start": [pd.Timestamp("2026-07-12").date()] * len(dates),
            "nearest_prior_manifestation_end": [pd.Timestamp("2026-07-14").date()] * len(dates),
            "days_from_nearest_prior_manifestation_end": [10] * len(dates),
            "pilot_association": ["DETECTION_WITHOUT_MODEL_SIGNAL"] * len(dates),
        }
    )
    summary = summarize_polyakov_activation_interval(
        "obs",
        "2026-07-24",
        "detected",
        scenarios,
        "2026-06-15",
        "2026-06-30",
    )
    assert summary["activation_scenario_count"] == 16
    assert summary["polyakov_evaluable"]
    assert summary["activation_interval_status"] == "CONSISTENT_ACROSS_FULL_ACTIVATION_INTERVAL"
    assert (
        summary["activation_interval_result"]
        == "DETECTION_WITHOUT_MODEL_SIGNAL__ROBUST_TO_ACTIVATION_INTERVAL"
    )
