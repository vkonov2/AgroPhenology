"""Methodological checks on synthetic dates; no VAAD or external weather reads."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agro_phenology.early_warning_core import (
    CALENDAR_FEATURES,
    EPISODE_FEATURES,
    ERA_COMMON_FEATURES,
    NASA_COMMON_FEATURES,
    _issue_timestamp,
    add_daily_features,
    build_daily_decisions,
    build_field_seasons,
)


def _visits(*observations: tuple[str, str], bbch51: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "visit_id": f"synthetic-{index}",
                "field_season": "synthetic-field-2024",
                "final_field_uid": "synthetic-field",
                "crop_code": 166,
                "crop_season": 2024,
                "observation_date": pd.Timestamp(day),
                "label_status": status,
                "observed_bbch51": day == bbch51,
                "final_coordinate_class": "A_direct",
                "final_latitude": 56.5,
                "final_longitude": 24.5,
                "weather_cell": "synthetic-era",
            }
            for index, (day, status) in enumerate(observations)
        ]
    )


def _seasons(visits: pd.DataFrame) -> pd.DataFrame:
    return build_field_seasons(visits, "2024-06-30")


@pytest.mark.parametrize(
    ("lead", "expected"),
    [(0, "imminent"), (2, "imminent"), (3, "actionable"), (10, "actionable"), (11, "no_record_in_horizon")],
)
def test_registration_target_uses_inclusive_three_to_ten_day_window(lead, expected):
    decisions = build_daily_decisions(
        _seasons(_visits(("2024-06-01", "unassessed"), ("2024-06-20", "positive")))
    ).set_index("issue_date")
    issue = pd.Timestamp("2024-06-20") - pd.Timedelta(days=lead)
    row = decisions.loc[issue]
    assert row.target_class == expected
    assert bool(row.target_observable)
    assert row.days_to_first_recorded_event == lead
    assert row.label_interval_start == issue + pd.Timedelta(days=3)
    assert row.label_interval_end == issue + pd.Timedelta(days=10)


def test_date_only_visits_and_phenophase_are_available_next_local_morning():
    seasons = _seasons(
        _visits(("2024-06-01", "unassessed"), ("2024-06-20", "positive"), bbch51="2024-06-01")
    )
    season = seasons.iloc[0]
    assert season.first_visit_available_date == pd.Timestamp("2024-06-02")
    assert season.first_observed_bbch51_available_date == pd.Timestamp("2024-06-02")
    assert season.first_recorded_event_available_date == pd.Timestamp("2024-06-21")
    decisions = build_daily_decisions(seasons).set_index("issue_date")
    assert decisions.index.min() == pd.Timestamp("2024-06-02")
    assert pd.Timestamp(decisions.iloc[0].issued_at) == pd.Timestamp("2024-06-02T08:00:00+03:00")
    assert bool(decisions.loc["2024-06-20", "service_active"])
    assert not bool(decisions.loc["2024-06-21", "service_active"])


def test_explicit_delayed_registry_availability_controls_service_stop():
    seasons = _seasons(_visits(("2024-06-01", "unassessed"), ("2024-06-20", "positive")))
    # This exercises the registry contract; ingestion of real publication logs is separate.
    seasons.loc[0, "first_recorded_event_available_date"] = pd.Timestamp("2024-06-23")
    decisions = build_daily_decisions(seasons).set_index("issue_date")
    assert bool(decisions.loc["2024-06-22", "service_active"])
    assert decisions.loc["2024-06-22", "target_class"] == "unknown"
    assert not bool(decisions.loc["2024-06-23", "service_active"])


@pytest.mark.parametrize(
    ("observations", "category", "warnable"),
    [
        ((("2024-06-01", "positive"),), "positive_known_at_entry", False),
        ((("2024-06-18", "unassessed"), ("2024-06-20", "positive")), "late_entry_first_event", False),
        ((("2024-06-16", "unassessed"), ("2024-06-20", "positive")), "warnable_first_event", True),
        ((("2024-06-01", "unassessed"),), "no_event_short_followup", False),
        ((("2024-06-01", "unassessed"), ("2024-06-12", "unassessed")), "no_event_with_observable_horizon", False),
    ],
)
def test_registry_keeps_first_positive_late_entry_and_no_event_categories(observations, category, warnable):
    seasons = _seasons(_visits(*observations))
    assert len(seasons) == 1
    assert seasons.iloc[0].entry_category == category
    assert bool(seasons.iloc[0].warnable_first_event) is warnable
    if category == "positive_known_at_entry":
        assert bool(seasons.iloc[0].positive_at_first_visit)
        assert not build_daily_decisions(seasons).service_active.any()


def test_repeated_positive_visit_does_not_move_or_duplicate_first_event():
    seasons = _seasons(
        _visits(("2024-06-01", "unassessed"), ("2024-06-20", "positive"), ("2024-06-25", "positive"))
    )
    assert len(seasons) == 1
    assert seasons.iloc[0].first_recorded_event_date == pd.Timestamp("2024-06-20")
    assert seasons.iloc[0].first_recorded_event_available_date == pd.Timestamp("2024-06-21")


def test_no_event_target_requires_complete_registration_horizon():
    decisions = build_daily_decisions(
        _seasons(_visits(("2024-06-01", "unassessed"), ("2024-06-12", "unassessed")))
    ).set_index("issue_date")
    assert decisions.loc["2024-06-02", "target_class"] == "no_record_in_horizon"
    assert bool(decisions.loc["2024-06-02", "target_observable"])
    assert decisions.loc["2024-06-03", "target_class"] == "unknown"
    assert not bool(decisions.loc["2024-06-03", "target_observable"])
    assert bool(decisions.loc["2024-06-30", "service_active"])


def test_future_visit_changes_outcome_mask_without_changing_decision_calendar():
    short = build_daily_decisions(_seasons(_visits(("2024-06-01", "unassessed"), ("2024-06-12", "unassessed"))))
    long = build_daily_decisions(_seasons(_visits(("2024-06-01", "unassessed"), ("2024-06-25", "unassessed"))))
    calendar = ["field_season", "issue_date", "issued_at", "service_active", "feature_cutoff_nasa_date", "feature_cutoff_era_episode_date"]
    pd.testing.assert_frame_equal(short[calendar], long[calendar])
    assert short.target_observable.sum() < long.target_observable.sum()
    assert short.issue_date.max() == pd.Timestamp("2024-06-30")


@pytest.mark.parametrize(
    ("before", "after", "hours", "utc_hour"),
    [("2024-03-30", "2024-03-31", 23, 5), ("2024-10-26", "2024-10-27", 25, 6)],
)
def test_riga_eight_am_uses_local_calendar_across_dst(before, after, hours, utc_hour):
    first = _issue_timestamp(pd.Timestamp(before), "Europe/Riga", "08:00:00")
    second = _issue_timestamp(pd.Timestamp(after), "Europe/Riga", "08:00:00")
    assert first.hour == second.hour == 8
    assert (second - first) == pd.Timedelta(hours=hours)
    assert second.tz_convert("UTC").hour == utc_hour
    assert (second.date() - first.date()).days == 1


def _weather_tables() -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2024-05-01", "2024-06-30")
    nasa = pd.DataFrame({"date": dates, "nasa_cell": "synthetic-nasa", "T2M": 16.0, "T2M_MIN": 12.0, "T2M_MAX": 20.0, "RH2M": 85.0, "PRECTOTCORR": 2.0})
    era = pd.DataFrame({"date": dates, "weather_cell": "synthetic-era", "temperature_mean_c": 16.0, "minimum_temperature_c": 12.0, "maximum_temperature_c": 20.0, "relative_humidity_mean_pct": 85.0, "precipitation_sum_mm": 2.0, "accepted": True, "day_status": "pass", "high_humidity_hours": 8.0})
    mapping = pd.DataFrame({"final_latitude": [56.5], "final_longitude": [24.5], "nasa_cell": ["synthetic-nasa"]})
    return {"nasa.parquet": nasa, "mapping.parquet": mapping, "era.parquet": era}


def _features_for_june20(monkeypatch, tables):
    monkeypatch.setattr(pd, "read_parquet", lambda path: tables[Path(path).name].copy(deep=True))
    decisions = build_daily_decisions(_seasons(_visits(("2024-06-01", "unassessed"))))
    decisions = decisions.loc[decisions.issue_date.eq(pd.Timestamp("2024-06-20"))]
    return add_daily_features(decisions, "nasa.parquet", "mapping.parquet", "era.parquet")[0]


def test_replacing_unavailable_weather_cannot_change_features_at_issue(monkeypatch):
    tables = _weather_tables()
    original = _features_for_june20(monkeypatch, tables)
    for name, cutoff in [("nasa.parquet", "2024-06-18"), ("era.parquet", "2024-06-18")]:
        future = tables[name].date.gt(pd.Timestamp(cutoff))
        numeric = tables[name].select_dtypes(include="number").columns
        tables[name].loc[future, numeric] = 999.0
    tables["era.parquet"].loc[tables["era.parquet"].date.gt(pd.Timestamp("2024-06-18")), "day_status"] = "fail"
    changed = _features_for_june20(monkeypatch, tables)
    features = CALENDAR_FEATURES + NASA_COMMON_FEATURES + ERA_COMMON_FEATURES + EPISODE_FEATURES
    pd.testing.assert_frame_equal(original[features], changed[features])
    assert bool(changed.iloc[0].common_weather_complete)
    assert bool(changed.iloc[0].episode_weather_complete)


@pytest.mark.parametrize("missing", ["calendar_day", "temperature_value"])
def test_missing_nasa_day_is_not_filled_with_future_or_zero_weather(monkeypatch, missing):
    tables = _weather_tables()
    affected = tables["nasa.parquet"].date.eq(pd.Timestamp("2024-06-15"))
    if missing == "calendar_day":
        tables["nasa.parquet"] = tables["nasa.parquet"].loc[~affected]
    else:
        tables["nasa.parquet"].loc[affected, "T2M"] = np.nan
    row = _features_for_june20(monkeypatch, tables).iloc[0]
    assert pd.isna(row.nasa_tmean_7d)
    assert pd.isna(row.nasa_active_temperature_sum_7d)
    assert not bool(row.nasa_common_complete)
    assert not bool(row.common_weather_complete)
    assert bool(row.era_common_complete)


def test_missing_era_calendar_day_makes_hutton_and_episode_inputs_unknown(monkeypatch):
    tables = _weather_tables()
    tables["era.parquet"] = tables["era.parquet"].loc[tables["era.parquet"].date.ne(pd.Timestamp("2024-06-18"))]
    row = _features_for_june20(monkeypatch, tables).iloc[0]
    assert not bool(row.era_common_complete)
    assert not bool(row.episode_weather_complete)
    assert pd.isna(row.hutton_score)
    assert pd.isna(row.era_polyakov_t10_c)
