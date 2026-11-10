from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from agro_phenology.external_validation_io import DownloadError
from agro_phenology.external_validation_io import file_inventory, write_json
from agro_phenology.external_validation_weather import HOURLY_VARIABLES, _normalize_response, build_plan, check, fetch, main


def _cohort(root: Path, source: str, *, dates=("2020-06-01",), lat=50.01, lon=8.01, site="site1") -> None:
    folder = root / "normalized" / source
    folder.mkdir(parents=True)
    pd.DataFrame({"site_id": site, "observation_date": pd.to_datetime(list(dates))}).to_parquet(folder / "observations.parquet", index=False)
    pd.DataFrame({"site_id": [site], "latitude": [lat], "longitude": [lon]}).to_parquet(folder / "locations.parquet", index=False)


def _response(request: dict) -> dict:
    times = pd.date_range(request["start_date"], pd.Timestamp(request["end_date"]) + pd.Timedelta(hours=23), freq="h")
    return {"latitude": request["requested_grid_latitude"], "longitude": request["requested_grid_longitude"], "utc_offset_seconds": 0,
            "hourly": {"time": times.strftime("%Y-%m-%dT%H:%M").tolist(), **{variable: [1.0]*len(times) for variable in HOURLY_VARIABLES}},
            "hourly_units": {"temperature_2m": "°C", "relative_humidity_2m": "%"}}


class Client:
    def __init__(self, requests: list[dict], *, fail=False):
        self.requests = {r["url"]: r for r in requests}
        self.calls = []
        self.fail = fail

    def get_json(self, url, relative_path, max_bytes):
        self.calls.append(url)
        return {"error": True, "reason": "not available"} if self.fail else _response(self.requests[url])


def test_shared_grid_year_is_requested_once_but_both_source_sites_remain(tmp_path) -> None:
    _cohort(tmp_path, "zenodo_6352735", lat=50.01, lon=8.01)
    _cohort(tmp_path, "zenodo_6520484", lat=50.04, lon=8.02)
    plan = build_plan(tmp_path)
    assert len(plan["requests"]) == 1
    assert len(plan["site_request_mapping"]) == 2
    assert plan["requests"][0]["source_ids"] == ["zenodo_6352735", "zenodo_6520484"]
    assert {m["source_latitude"] for m in plan["site_request_mapping"]} == {50.01, 50.04}


def test_no_fake_date_from_year_and_no_weather_for_occurrence_or_unlocated_sources(tmp_path) -> None:
    _cohort(tmp_path, "zenodo_6352735", dates=(None,))
    _cohort(tmp_path, "gbif_potato_occurrences", dates=("2020-06-01",))
    _cohort(tmp_path, "zenodo_18351833", dates=("2022-01-01",))
    plan = build_plan(tmp_path)
    assert plan["requests"] == []
    assert plan["site_request_mapping"] == []


def test_full_calendar_year_with_winter_and_partial_2026_cutoff(tmp_path) -> None:
    _cohort(tmp_path, "zenodo_6520484", dates=("2026-07-01",))
    request = build_plan(tmp_path)["requests"][0]
    assert request["start_date"] == "2026-01-01"
    assert request["end_date"] == "2026-11-04"
    assert request["partial_year"]
    assert request["expected_hours"] == 308 * 24
    assert request["parameters"]["models"] == "era5"
    assert request["parameters"]["timezone"] == "UTC"
    assert "leaf_wetness_duration" not in request["parameters"]["hourly"]


def test_ambiguous_site_ids_are_rejected_instead_of_cartesian_weather_matching(tmp_path) -> None:
    _cohort(tmp_path, "zenodo_6352735")
    path = tmp_path / "normalized/zenodo_6352735/locations.parquet"
    pd.DataFrame({"site_id": ["site1", "site1"], "latitude": [50.0, 52.0], "longitude": [8.0, 9.0]}).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="multiple source coordinates"):
        build_plan(tmp_path)


def test_fetch_is_resumable_with_explicit_era5_and_integrity_check(tmp_path) -> None:
    collection, output = tmp_path / "collection", tmp_path / "weather"
    _cohort(collection, "zenodo_6520484", dates=("2020-06-01", "2021-06-01"))
    requests = build_plan(collection)["requests"]
    client = Client(requests)
    first = fetch(collection, output, max_requests=1, client=client)
    assert first["status"] == "partial"
    assert first["pending_requests"] == 1
    second = fetch(collection, output, max_requests=1, client=client)
    assert second["status"] == "complete"
    assert len(client.calls) == 2
    for url in client.calls:
        params = parse_qs(urlparse(url).query)
        assert params["models"] == ["era5"]
        assert params["timezone"] == ["UTC"]
        assert "best_match" not in url and "ifs" not in url
    assert check(output)["status"] == "passed"
    assert len(list((output / "normalized").glob("*.parquet"))) == 2
    metadata = json.loads(next((output / "request_metadata").glob("*.json")).read_text())
    assert metadata["publication_time"] is None
    assert metadata["initialization_time"] is None
    assert metadata["leaf_wetness"] == "not_observed_not_fabricated"


def test_failure_does_not_try_ifs_or_forecast_fallback(tmp_path) -> None:
    collection, output = tmp_path / "collection", tmp_path / "weather"
    _cohort(collection, "zenodo_6520484")
    client = Client(build_plan(collection)["requests"], fail=True)
    result = fetch(collection, output, max_requests=10, client=client)
    assert result["status"] == "complete_with_gaps"
    assert result["failed_requests"] == 1
    assert len(client.calls) == 1
    assert not (output / "normalized").exists()


def test_missing_hours_or_non_utc_response_are_not_filled(tmp_path) -> None:
    _cohort(tmp_path, "zenodo_6352735")
    request = build_plan(tmp_path)["requests"][0]
    response = _response(request)
    response["hourly"]["time"] = response["hourly"]["time"][:-1]
    with pytest.raises(ValueError, match="exactly cover"):
        _normalize_response(request, response)
    response = _response(request)
    response["utc_offset_seconds"] = 3600
    with pytest.raises(ValueError, match="not UTC"):
        _normalize_response(request, response)


@pytest.mark.parametrize("message,reason", [("429 Client Error: Too Many Requests", "provider_rate_limit"),
                                            ("HTTP 403: forbidden", "provider_authentication_or_access_block")])
def test_provider_global_limit_stops_batch_and_retry_failed_restores_pending(tmp_path, message, reason) -> None:
    collection, output = tmp_path / "collection", tmp_path / "weather"
    _cohort(collection, "zenodo_6520484", dates=("2020-06-01", "2021-06-01"))
    requests = build_plan(collection)["requests"]

    class BlockedClient(Client):
        def get_json(self, url, relative_path, max_bytes):
            self.calls.append(url)
            raise DownloadError(message)

    blocked = BlockedClient(requests)
    result = fetch(collection, output, max_requests=250, client=blocked)
    assert len(blocked.calls) == 1
    assert result["failed_requests"] == 1
    assert result["pending_requests"] == 1
    assert json.loads((output / "weather_manifest.json").read_text())["stop_reason"] == reason
    good = Client(requests)
    result = fetch(collection, output, max_requests=250, client=good, retry_failed=True)
    assert result["status"] == "complete"
    assert len(good.calls) == 2
    history = json.loads((output / "weather_manifest.json").read_text())["attempt_history"]
    assert [r["status"] for r in history] == ["failed", "complete", "complete"]


def test_completed_gaps_require_explicit_retry_failed(tmp_path) -> None:
    collection, output = tmp_path / "collection", tmp_path / "weather"
    _cohort(collection, "zenodo_6520484")
    requests = build_plan(collection)["requests"]
    assert fetch(collection, output, client=Client(requests, fail=True))["status"] == "complete_with_gaps"
    good = Client(requests)
    fetch(collection, output, client=good)
    assert good.calls == []
    assert fetch(collection, output, client=good, retry_failed=True)["status"] == "complete"
    assert len(good.calls) == 1


def test_check_cli_does_not_require_primary_collection_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("agro_phenology.external_validation_weather.check", lambda output_dir: {"status": "passed"})
    assert main(["check", "--output-dir", str(tmp_path)]) == 0


def test_completed_manifest_does_not_hide_changed_normalized_inputs(tmp_path) -> None:
    _cohort(tmp_path, "zenodo_6352735")
    write_json(tmp_path / "manifest.json", {"status": "complete", "file_inventory": file_inventory(tmp_path)})
    assert build_plan(tmp_path)["requests"]
    (tmp_path / "normalized/zenodo_6352735/locations.parquet").write_bytes(b"modified")
    with pytest.raises(ValueError, match="integrity check failed"):
        build_plan(tmp_path)
