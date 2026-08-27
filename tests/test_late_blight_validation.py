from datetime import date

import pandas as pd

from agro_phenology.late_blight_validation import summarize_hutton_lookbacks
from agro_phenology.late_blight_pilot import build_pilot_analysis_points


def test_fixed_hutton_lookbacks_are_reported_without_best_window_selection() -> None:
    periods = pd.DataFrame(
        {
            "period_end": [date(2026, 7, 10), date(2026, 7, 20)],
            "period_status": ["pass", "fail"],
        }
    )
    summary = summarize_hutton_lookbacks("obs-1", "2026-07-24", "detected", periods)
    assert summary["lookback_days"].tolist() == [7, 14, 21]
    assert not bool(summary.iloc[0]["hutton_signal_present"])
    assert bool(summary.iloc[1]["hutton_signal_present"])
    assert summary.iloc[1]["pilot_association"] == "TEMPORAL_CONCORDANCE"


def test_indeterminate_weather_is_not_silently_treated_as_no_signal() -> None:
    periods = pd.DataFrame(
        {"period_end": [date(2026, 8, 23)], "period_status": ["indeterminate"]}
    )
    summary = summarize_hutton_lookbacks("obs-2", "2026-08-24", "not_detected", periods)
    assert pd.isna(summary.iloc[0]["hutton_signal_present"])
    assert summary.iloc[0]["pilot_association"] == "NOT_EVALUABLE_INCOMPLETE_WEATHER"


def test_proxy_points_do_not_replace_missing_field_coordinates() -> None:
    observations = pd.DataFrame(
        {
            "observation_id": ["LB-PILOT-001", "LB-PILOT-002"],
            "latitude": [pd.NA, pd.NA],
            "longitude": [pd.NA, pd.NA],
        }
    )
    points = build_pilot_analysis_points(observations)
    assert len(points) == 6
    assert points["latitude"].isna().all()
    assert points["analysis_latitude"].notna().all()
    assert points["coordinate_rule"].str.contains("not_field_coordinate").all()


def test_supplied_village_coordinate_replaces_regional_proxy_grid() -> None:
    observations = pd.DataFrame(
        {
            "observation_id": ["LB-PILOT-001"],
            "latitude": [56.4008889],
            "longitude": [37.2483611],
            "location_precision": ["user_supplied_village_coordinate_not_confirmed_field"],
        }
    )
    points = build_pilot_analysis_points(observations)
    assert len(points) == 1
    assert points.iloc[0]["analysis_latitude"] == 56.4008889
    assert points.iloc[0]["analysis_longitude"] == 37.2483611
    assert points.iloc[0]["coordinate_rule"] == "user_supplied_village_coordinate_not_confirmed_field"
