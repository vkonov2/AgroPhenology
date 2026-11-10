from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from agro_phenology.external_validation_io import DownloadError
from agro_phenology.external_validation_io import file_inventory, write_json
from agro_phenology.external_validation_weather import HOURLY_VARIABLES, _normalize_response, build_plan, check, fetch, main, plan as save_plan


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


def test_request_selection_filters_requests_and_preserves_all_shared_source_mappings(tmp_path) -> None:
    _cohort(tmp_path, "zenodo_6352735", dates=("2020-06-01", "2021-06-01"))
    _cohort(tmp_path, "zenodo_6520484", dates=("2020-06-02",), lat=50.04, lon=8.02)
    _cohort(tmp_path, "zenodo_6351794", dates=("2022-06-01",), lat=53, lon=9)
    full = build_plan(tmp_path)
    selected_id = next(r["request_id"] for r in full["requests"] if r["year"] == 2020)
    selected = build_plan(tmp_path, request_ids={selected_id})
    assert [r["request_id"] for r in selected["requests"]] == [selected_id]
    assert selected["selected_request_ids"] == [selected_id]
    assert len(selected["site_request_mapping"]) == 2
    assert {m["source_id"] for m in selected["site_request_mapping"]} == {"zenodo_6352735", "zenodo_6520484"}
    assert all(m["request_id"] == selected_id for m in selected["site_request_mapping"])
    assert selected["input_hashes"] == full["input_hashes"]
    assert selected["requests"][0] == full["requests"][0]


def test_default_plan_adds_no_selection_key_and_explicit_none_is_identical(tmp_path) -> None:
    _cohort(tmp_path, "zenodo_6352735")
    implicit = build_plan(tmp_path)
    explicit_none = build_plan(tmp_path, request_ids=None)
    assert implicit == explicit_none
    assert "selected_request_ids" not in implicit
    output = tmp_path / "weather"
    assert save_plan(tmp_path, output) == implicit
    assert not (output / "selected_request_ids.json").exists()


def test_request_selection_is_sorted_and_hash_deterministic_regardless_of_set_order(tmp_path) -> None:
    _cohort(tmp_path, "zenodo_6352735", dates=("2020-06-01", "2021-06-01"))
    full = build_plan(tmp_path)
    ids = [r["request_id"] for r in full["requests"]]
    forward = build_plan(tmp_path, request_ids=set(ids))
    reverse = build_plan(tmp_path, request_ids=set(reversed(ids)))
    assert forward == reverse
    assert forward["selected_request_ids"] == sorted(ids)
    assert forward["requests"] == full["requests"]
    assert forward["plan_sha256"] != full["plan_sha256"]


def test_unknown_request_id_fails_before_creating_a_plan_directory(tmp_path) -> None:
    collection, output = tmp_path / "collection", tmp_path / "weather"
    _cohort(collection, "zenodo_6352735")
    with pytest.raises(ValueError, match="Unknown weather request IDs"):
        save_plan(collection, output, request_ids={"unknown_request"})
    assert not output.exists()


def test_empty_selection_is_explicit_and_never_becomes_the_full_plan(tmp_path) -> None:
    _cohort(tmp_path, "zenodo_6352735")
    selected = build_plan(tmp_path, request_ids=set())
    assert selected["selected_request_ids"] == []
    assert selected["requests"] == []
    assert selected["site_request_mapping"] == []


def test_selected_fetch_resumes_same_ids_and_rejects_changed_scope_without_mutation(tmp_path) -> None:
    collection, output = tmp_path / "collection", tmp_path / "weather"
    _cohort(collection, "zenodo_6352735", dates=("2019-06-01", "2020-06-01", "2021-06-01"))
    requests = build_plan(collection)["requests"]
    selection = {r["request_id"] for r in requests[1:]}
    client = Client(requests)
    first = fetch(collection, output, request_ids=selection, max_requests=1, client=client)
    assert first["status"] == "partial"
    assert first["total_requests"] == 2
    before = {name: (output/name).read_bytes() for name in ["weather_plan.json", "weather_manifest.json"]}
    with pytest.raises(ValueError, match="changed; choose a new output directory"):
        fetch(collection, output, client=client)
    with pytest.raises(ValueError, match="changed; choose a new output directory"):
        fetch(collection, output, request_ids={requests[0]["request_id"]}, client=client)
    assert all((output/name).read_bytes() == body for name, body in before.items())
    assert len(client.calls) == 1
    second = fetch(collection, output, request_ids=set(reversed(sorted(selection))), client=client)
    assert second["status"] == "complete"
    assert second["total_requests"] == 2
    assert set(client.calls) == {r["url"] for r in requests[1:]}
    assert check(output)["status"] == "passed"
    assert json.loads((output / "selected_request_ids.json").read_text()) == sorted(selection)
    report = (output / "README_ru.md").read_text()
    assert report.count(f"--request-ids-file {output.resolve() / 'selected_request_ids.json'}") == 2
    fetch(collection, output, request_ids=selection, client=client)
    assert len(client.calls) == 2


def test_plan_cli_loads_json_array_without_hashing_its_path_or_order(tmp_path) -> None:
    collection, output = tmp_path / "collection", tmp_path / "weather"
    _cohort(collection, "zenodo_6352735", dates=("2020-06-01", "2021-06-01"))
    full = build_plan(collection)
    ids = [r["request_id"] for r in reversed(full["requests"])]
    selection_file = tmp_path / "selection.json"
    write_json(selection_file, ids)
    args = ["plan", "--collection-dir", str(collection), "--output-dir", str(output), "--request-ids-file", str(selection_file)]
    assert main(args) == 0
    saved_bytes = (output / "weather_plan.json").read_bytes()
    saved = json.loads(saved_bytes)
    assert saved["selected_request_ids"] == sorted(ids)
    assert str(selection_file) not in saved["input_hashes"]
    write_json(selection_file, list(reversed(ids)))
    assert main(args) == 0
    assert (output / "weather_plan.json").read_bytes() == saved_bytes


@pytest.mark.parametrize("value", [{"request_ids": []}, [123], [""], None])
def test_selection_file_cli_requires_an_array_of_nonempty_strings(tmp_path, value) -> None:
    path = tmp_path / "selection.json"
    write_json(path, value)
    with pytest.raises(SystemExit) as error:
        main(["plan", "--collection-dir", str(tmp_path / "collection"), "--output-dir", str(tmp_path / "weather"), "--request-ids-file", str(path)])
    assert error.value.code == 2


def test_fetch_cli_forwards_selection_to_the_api(monkeypatch, tmp_path) -> None:
    path = tmp_path / "selection.json"
    write_json(path, ["chosen_request"])
    captured = {}

    def fake_fetch(collection_dir, output_dir, **kwargs):
        captured.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr("agro_phenology.external_validation_weather.fetch", fake_fetch)
    assert main(["fetch", "--collection-dir", str(tmp_path / "collection"), "--output-dir", str(tmp_path / "weather"), "--request-ids-file", str(path)]) == 0
    assert captured["request_ids"] == {"chosen_request"}


def test_saved_selection_copy_cannot_be_changed_when_resuming(tmp_path) -> None:
    collection, output = tmp_path / "collection", tmp_path / "weather"
    _cohort(collection, "zenodo_6352735", dates=("2020-06-01", "2021-06-01"))
    request_id = build_plan(collection)["requests"][0]["request_id"]
    save_plan(collection, output, request_ids={request_id})
    write_json(output / "selected_request_ids.json", ["changed_id"])
    before = (output / "weather_plan.json").read_bytes()
    with pytest.raises(ValueError, match="Saved request ID selection changed"):
        save_plan(collection, output, request_ids={request_id})
    assert (output / "weather_plan.json").read_bytes() == before
