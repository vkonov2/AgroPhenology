import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from agro_phenology.open_meteo import (
    OpenMeteoClient,
    OpenMeteoError,
    build_archive_params,
    cache_key,
    parse_archive_response,
)

FIXTURE = Path(__file__).parent / "fixtures" / "open_meteo_response.json"


def test_parse_saved_open_meteo_response() -> None:
    frame, metadata = parse_archive_response(json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert frame["temperature_2m"].tolist() == [8.0, 10.0, 12.0]
    assert metadata["timezone"] == "Europe/Moscow"
    assert metadata["elevation"] == 156.0


def test_parser_rejects_inconsistent_arrays() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["hourly"]["temperature_2m"].pop()
    with pytest.raises(OpenMeteoError, match="inconsistent"):
        parse_archive_response(payload)


def test_parser_keeps_optional_late_blight_variables() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["hourly"]["relative_humidity_2m"] = [80.0, 90.0, 95.0]
    payload["hourly"]["precipitation"] = [0.0, 0.2, None]
    payload["hourly_units"]["relative_humidity_2m"] = "%"
    payload["hourly_units"]["precipitation"] = "mm"
    frame, _ = parse_archive_response(payload)
    assert frame["relative_humidity_2m"].tolist() == [80.0, 90.0, 95.0]
    assert frame["precipitation"].isna().sum() == 1


def test_archive_model_is_explicit_and_part_of_cache_identity() -> None:
    land = build_archive_params(
        55.7,
        37.6,
        "2026-06-01",
        "2026-06-02",
        ("temperature_2m", "relative_humidity_2m"),
    )
    era5 = build_archive_params(
        55.7,
        37.6,
        "2026-06-01",
        "2026-06-02",
        ("precipitation",),
        model="era5",
    )
    assert land["models"] == "era5_land"
    assert era5["models"] == "era5"
    assert era5["hourly"] == "precipitation"
    assert era5["precipitation_unit"] == "mm"
    assert "temperature_unit" not in era5
    assert cache_key(land) != cache_key(era5)
    with pytest.raises(ValueError, match="Unsupported archive model"):
        build_archive_params(55.7, 37.6, "2026-06-01", "2026-06-02", model="best_match")


def test_precipitation_only_response_is_valid_for_era5_profile() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    del payload["hourly"]["temperature_2m"]
    del payload["hourly_units"]["temperature_2m"]
    payload["hourly"]["precipitation"] = [0.0, 1.2, 0.0]
    payload["hourly_units"]["precipitation"] = "mm"
    frame, metadata = parse_archive_response(payload, ("precipitation",))
    assert frame["precipitation"].tolist() == [0.0, 1.2, 0.0]
    assert metadata["hourly_units"]["precipitation"] == "mm"


def test_era5_precipitation_fetch_has_separate_metadata_and_cache(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    del payload["hourly"]["temperature_2m"]
    del payload["hourly_units"]["temperature_2m"]
    payload["hourly"]["precipitation"] = [0.0, 1.2, 0.0]
    payload["hourly_units"]["precipitation"] = "mm"
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(payload).encode("utf-8")
    response.url = "https://archive-api.open-meteo.com/v1/archive?models=era5"
    session = Mock()
    session.get.return_value = response
    client = OpenMeteoClient(tmp_path, session=session)
    first = client.fetch_era5_precipitation_hourly(55.7, 37.6, "2026-06-01", "2026-06-02")
    second = client.fetch_era5_precipitation_hourly(55.7, 37.6, "2026-06-01", "2026-06-02")
    assert first.metadata["model"] == "era5"
    assert first.metadata["hourly_variables"] == ["precipitation"]
    assert not first.cache_hit and second.cache_hit
    assert session.get.call_count == 1


def test_network_error_retries_without_live_request(tmp_path: Path) -> None:
    session = Mock()
    session.get.side_effect = requests.ConnectionError("offline")
    client = OpenMeteoClient(tmp_path, max_attempts=3, backoff_seconds=0, session=session)
    with pytest.raises(OpenMeteoError, match="after 3 attempts"):
        client.fetch_hourly(55.7, 37.6, "2023-04-01", "2023-04-02")
    assert session.get.call_count == 3


def test_raw_response_and_request_metadata_are_cached(tmp_path: Path) -> None:
    raw = FIXTURE.read_bytes()
    response = requests.Response()
    response.status_code = 200
    response._content = raw
    response.url = "https://archive-api.open-meteo.com/v1/archive?example=1"
    session = Mock(); session.get.return_value = response
    client = OpenMeteoClient(tmp_path, session=session)
    first = client.fetch_hourly(55.7, 37.6, "2023-04-01", "2023-04-02")
    second = client.fetch_hourly(55.7, 37.6, "2023-04-01", "2023-04-02")
    assert first.raw_path.read_bytes() == raw
    assert not first.cache_hit and second.cache_hit
    assert session.get.call_count == 1
    assert second.metadata["request_parameters"]["models"] == "era5_land"


def test_stale_incomplete_cache_is_refreshed(tmp_path: Path) -> None:
    incomplete_payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    incomplete_payload["hourly"]["temperature_2m"][-1] = None
    complete_raw = FIXTURE.read_bytes()

    first_response = requests.Response()
    first_response.status_code = 200
    first_response._content = json.dumps(incomplete_payload).encode("utf-8")
    first_response.url = "https://archive-api.open-meteo.com/v1/archive?run=1"
    second_response = requests.Response()
    second_response.status_code = 200
    second_response._content = complete_raw
    second_response.url = "https://archive-api.open-meteo.com/v1/archive?run=2"
    session = Mock(); session.get.side_effect = [first_response, second_response]
    client = OpenMeteoClient(tmp_path, session=session)

    first = client.fetch_hourly(55.7, 37.6, "2023-04-01", "2023-04-02")
    metadata_path = first.raw_path.with_name(first.raw_path.stem + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["retrieved_at"] = "2000-01-01T00:00:00+00:00"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    refreshed = client.fetch_hourly(55.7, 37.6, "2023-04-01", "2023-04-02")
    assert not refreshed.cache_hit
    assert refreshed.hourly["temperature_2m"].notna().all()
    assert session.get.call_count == 2
