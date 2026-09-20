from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from agro_phenology.early_warning_core import EPISODE_FEATURES
from agro_phenology.shadow_sources import (
    FROZEN_ERA5_ENDPOINT,
    FROZEN_ERA5_VARIABLES,
    SourceCompatibilityError,
    assert_source_available_as_of,
    build_episode_feature_snapshot,
    build_frozen_era5_archive_params,
    parse_frozen_era5_hourly,
    sanitised_request_contract,
    source_profile_contract,
    validate_frozen_era5_request_profile,
    validate_source_timing,
)


def _request(start: str, end: str) -> dict:
    return {
        "endpoint": FROZEN_ERA5_ENDPOINT,
        **build_frozen_era5_archive_params(56.95, 24.10, start, end),
    }


def _payload_for_local_days(start: str, end: str) -> dict:
    local_start = pd.Timestamp(start).tz_localize("Europe/Riga")
    local_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize(
        "Europe/Riga"
    )
    hours = pd.date_range(
        local_start.tz_convert("UTC"),
        local_end.tz_convert("UTC"),
        inclusive="left",
        freq="h",
    )
    n = len(hours)
    return {
        "latitude": 57.0,
        "longitude": 24.0,
        "elevation": 100.0,
        "timezone": "GMT",
        "timezone_abbreviation": "GMT",
        "utc_offset_seconds": 0,
        "hourly_units": {
            "time": "iso8601",
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "precipitation": "mm",
            "soil_temperature_0_to_7cm": "°C",
            "soil_moisture_0_to_7cm": "m³/m³",
            "shortwave_radiation": "W/m²",
            "wind_speed_10m": "km/h",
        },
        "hourly": {
            "time": [stamp.strftime("%Y-%m-%dT%H:%M") for stamp in hours],
            "temperature_2m": [15.0] * n,
            "relative_humidity_2m": [95.0] * n,
            "precipitation": [1.0] * n,
            "soil_temperature_0_to_7cm": [13.0] * n,
            "soil_moisture_0_to_7cm": [0.25] * n,
            "shortwave_radiation": [100.0] * n,
            "wind_speed_10m": [5.0] * n,
        },
    }


def test_exact_frozen_request_profile_and_contract() -> None:
    params = build_frozen_era5_archive_params(
        56.95, 24.10, "2026-06-01", "2026-06-23"
    )
    assert params == {
        "latitude": 56.95,
        "longitude": 24.1,
        "start_date": "2026-06-01",
        "end_date": "2026-06-23",
        "hourly": ",".join(FROZEN_ERA5_VARIABLES),
        "models": "era5",
        "timezone": "UTC",
        "cell_selection": "nearest",
        "elevation": "nan",
    }
    request = {"endpoint": FROZEN_ERA5_ENDPOINT, **params}
    attested = validate_frozen_era5_request_profile(request)
    assert attested["endpoint"] == FROZEN_ERA5_ENDPOINT
    assert attested["models"] == "era5"
    assert attested["hourly"] == list(FROZEN_ERA5_VARIABLES)
    assert attested["statistical_downscaling"] is False

    sanitised = sanitised_request_contract(
        start_date="2026-06-01",
        end_date="2026-06-23",
        location_ref="weather-cell-pseudonym",
    )
    assert "latitude" not in sanitised and "longitude" not in sanitised
    assert validate_frozen_era5_request_profile(sanitised) == attested


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("models", "best_match"),
        ("models", "era5_land"),
        ("models", "era5_seamless"),
        ("timezone", "auto"),
        ("cell_selection", "land"),
        ("elevation", 100.0),
        ("endpoint", "https://api.open-meteo.com/v1/forecast"),
        ("endpoint", None),
    ],
)
def test_incompatible_request_profiles_are_rejected(field: str, value: object) -> None:
    request = _request("2026-06-01", "2026-06-23")
    request[field] = value
    with pytest.raises(SourceCompatibilityError):
        validate_frozen_era5_request_profile(request)


def test_response_without_attested_request_cannot_claim_exact_era5() -> None:
    with pytest.raises(SourceCompatibilityError, match="response bytes do not attest"):
        parse_frozen_era5_hourly(
            _payload_for_local_days("2026-06-01", "2026-06-01"),
            weather_cell="cell-a",
        )


def test_exact_units_and_utc_response_are_enforced() -> None:
    payload = _payload_for_local_days("2026-06-01", "2026-06-01")
    daily, metadata = parse_frozen_era5_hourly(
        payload,
        weather_cell="cell-a",
        request_profile=_request("2026-06-01", "2026-06-01"),
    )
    assert len(daily) == 1
    assert daily.iloc[0]["accepted"]
    assert metadata["request_profile_attested"] is True
    assert metadata["product"] == "ERA5"
    assert metadata["returned_timezone"] == "GMT"
    assert metadata["local_day_timezone"] == "Europe/Riga"

    incompatible = deepcopy(payload)
    incompatible["hourly_units"]["wind_speed_10m"] = "m/s"
    with pytest.raises(SourceCompatibilityError, match="wind_speed_10m"):
        parse_frozen_era5_hourly(
            incompatible,
            weather_cell="cell-a",
            request_profile=_request("2026-06-01", "2026-06-01"),
        )

    incompatible = deepcopy(payload)
    incompatible["timezone"] = "Europe/Riga"
    with pytest.raises(SourceCompatibilityError, match="UTC response"):
        parse_frozen_era5_hourly(
            incompatible,
            weather_cell="cell-a",
            request_profile=_request("2026-06-01", "2026-06-01"),
        )


@pytest.mark.parametrize(
    ("local_day", "expected_hours"),
    [("2026-03-29", 23), ("2026-10-25", 25)],
)
def test_europe_riga_dst_days_have_23_or_25_hours(
    local_day: str, expected_hours: int
) -> None:
    daily, _ = parse_frozen_era5_hourly(
        _payload_for_local_days(local_day, local_day),
        weather_cell="cell-dst",
        request_profile=_request(local_day, local_day),
    )
    assert daily.iloc[0]["expected_hours"] == expected_hours
    assert daily.iloc[0]["temperature_valid_hours"] == expected_hours
    assert daily.iloc[0]["accepted"]


def test_episode_features_require_complete_t_minus_2_and_23_day_history() -> None:
    payload = _payload_for_local_days("2026-06-01", "2026-06-23")
    daily, _ = parse_frozen_era5_hourly(
        payload,
        weather_cell="cell-features",
        request_profile=_request("2026-06-01", "2026-06-23"),
    )
    snapshot = build_episode_feature_snapshot(
        daily,
        weather_cell="cell-features",
        issue_local_date="2026-06-25",
    )
    assert snapshot["required_last_local_date"] == "2026-06-23"
    assert snapshot["status"] == "available"
    assert snapshot["complete"] is True
    assert list(snapshot["features"]) == EPISODE_FEATURES
    assert all(np.isfinite(value) for value in snapshot["features"].values())

    only_22_days = build_episode_feature_snapshot(
        daily.iloc[:-1],
        weather_cell="cell-features",
        issue_local_date="2026-06-24",
    )
    assert only_22_days["required_last_local_date"] == "2026-06-22"
    assert only_22_days["status"] == "insufficient_complete_history"
    assert only_22_days["complete"] is False


def test_documented_five_day_lag_does_not_satisfy_unchanged_t_minus_2() -> None:
    daily, _ = parse_frozen_era5_hourly(
        _payload_for_local_days("2026-08-14", "2026-09-05"),
        weather_cell="cell-late",
        request_profile=_request("2026-08-14", "2026-09-05"),
    )
    snapshot = build_episode_feature_snapshot(
        daily,
        weather_cell="cell-late",
        issue_local_date="2026-09-10",
    )
    assert snapshot == {
        "status": "late_or_missing_required_cutoff",
        "required_last_local_date": "2026-09-08",
        "available_last_local_date": "2026-09-05",
        "features": None,
        "complete": False,
    }


def test_initialization_publication_and_actual_availability_are_distinct() -> None:
    timing = validate_source_timing(
        run_initialization_at="2026-09-05T00:00:00Z",
        provider_published_at=None,
        retrieval_started_at="2026-09-10T05:00:00Z",
        retrieval_completed_at="2026-09-10T05:00:02Z",
        first_seen_at="2026-09-10T05:00:02Z",
        ingested_at="2026-09-10T05:00:03Z",
    )
    assert timing["run_initialization_at"] == "2026-09-05T00:00:00+00:00"
    assert timing["provider_published_at"] is None
    assert source_profile_contract()["time_semantics"]["provider_published_at"].startswith(
        "confirmed publication"
    )

    with pytest.raises(SourceCompatibilityError, match="late bytes"):
        assert_source_available_as_of(
            retrieval_completed_at="2026-09-10T05:00:02Z",
            first_seen_at="2026-09-10T05:00:02Z",
            ingested_at="2026-09-10T05:00:03Z",
            decision_at="2026-09-10T05:00:01Z",
        )
