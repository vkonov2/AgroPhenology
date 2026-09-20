from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agro_phenology.polyakov_fixed_date_experiment import (
    DEFAULT_SNAPSHOT_LATE_BLIGHT,
    build_fixed_polyakov_scores,
)


def _weather(start: str = "2020-06-01", end: str = "2020-07-20") -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame(
        {
            "weather_cell": "57.00_24.00",
            "date": dates,
            "temperature_mean_c": 15.0,
            "relative_humidity_mean_pct": 80.0,
            "precipitation_sum_mm": 3.0,
            "accepted": True,
        }
    )


def _decisions(issue_dates: list[str]) -> pd.DataFrame:
    issue = pd.to_datetime(issue_dates)
    return pd.DataFrame(
        {
            "field_season": [f"field_{index}_2020" for index in range(len(issue))],
            "weather_cell": "57.00_24.00",
            "season": 2020,
            "issue_date": issue,
            "feature_cutoff_era_episode_date": issue - pd.Timedelta(days=2),
            "evaluation_field_day": True,
            "days_to_first_recorded_event": 5.0,
            "first_observed_bbch51_date": pd.to_datetime(
                ["2020-05-01"] * len(issue)
            ),
        }
    )


def test_fixed_july1_gate_warmup_and_earliest_positive_score():
    decisions = _decisions(
        ["2020-06-30", "2020-07-03", "2020-07-11", "2020-07-12", "2020-07-18"]
    )
    result, audit = build_fixed_polyakov_scores(
        decisions,
        _weather(),
        DEFAULT_SNAPSHOT_LATE_BLIGHT,
        source_label="synthetic_test",
    )
    assert result["polyakov_activation_date_fixed"].eq(pd.Timestamp("2020-07-01")).all()
    assert result["polyakov_score_fixed"].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert result["polyakov_status_fixed"].tolist()[:3] == [
        "not_active_before_fixed_july1",
        "deterministic_post_activation_warmup",
        "deterministic_post_activation_warmup",
    ]
    assert result.iloc[-1]["polyakov_status_fixed"] == "OUTBREAK_EXPECTED"
    assert audit["fixed_date_is_research_assumption_not_observed_phenology"]


def test_fixed_date_does_not_read_observed_phenology():
    decisions = _decisions(["2020-07-18", "2020-07-18"])
    decisions.loc[0, "first_observed_bbch51_date"] = pd.Timestamp("2020-04-10")
    decisions.loc[1, "first_observed_bbch51_date"] = pd.NaT
    result, _ = build_fixed_polyakov_scores(
        decisions,
        _weather(),
        DEFAULT_SNAPSHOT_LATE_BLIGHT,
        source_label="synthetic_test",
    )
    assert result["polyakov_score_fixed"].tolist() == [1.0, 1.0]
    assert result["polyakov_activation_policy"].eq(
        "fixed_calendar_date_july_1_known_in_advance"
    ).all()


def test_missing_weather_after_warmup_is_abstention_not_zero():
    decisions = _decisions(["2020-07-20"])
    weather = _weather(end="2020-07-10")
    result, audit = build_fixed_polyakov_scores(
        decisions,
        weather,
        DEFAULT_SNAPSHOT_LATE_BLIGHT,
        source_label="truncated_test",
    )
    assert np.isnan(result.loc[0, "polyakov_score_fixed"])
    assert result.loc[0, "polyakov_status_fixed"] == "missing_weather_after_warmup"
    assert audit["weather_abstention_service_days_after_warmup"] == 1


def test_weather_cutoff_later_than_issue_minus_two_is_rejected():
    decisions = _decisions(["2020-07-18"])
    decisions.loc[0, "feature_cutoff_era_episode_date"] = pd.Timestamp("2020-07-17")
    with pytest.raises(AssertionError, match="later than issue_date - 2"):
        build_fixed_polyakov_scores(
            decisions,
            _weather(),
            DEFAULT_SNAPSHOT_LATE_BLIGHT,
            source_label="leak_test",
        )


def test_duplicate_weather_keys_are_rejected():
    decisions = _decisions(["2020-07-18"])
    weather = _weather()
    weather = pd.concat([weather, weather.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate weather_cell/date"):
        build_fixed_polyakov_scores(
            decisions,
            weather,
            DEFAULT_SNAPSHOT_LATE_BLIGHT,
            source_label="duplicate_test",
        )
