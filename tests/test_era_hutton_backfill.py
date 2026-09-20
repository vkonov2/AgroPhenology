from __future__ import annotations
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from agro_phenology.early_warning_core import sha256_file
from agro_phenology.era_hutton_backfill import (
    DEFAULT_FROZEN_ERA,
    DEFAULT_FROZEN_META,
    DEFAULT_V3,
    FROZEN_ERA5_ENDPOINT,
    _canonical_sha256,
    _download_one,
    _ensure_run,
    _request_parameters,
    build_request_plan,
)
from agro_phenology.shadow_sources import (
    FROZEN_ERA5_PROFILE_ID,
    FROZEN_ERA5_VARIABLES,
    validate_frozen_era5_request_profile,
)


class FakeResponse:
    def __init__(self, payload: bytes, status_code: int = 200):
        self.content = payload
        self.status_code = status_code
        self.headers = {"Content-Type": "application/json"}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error


class FakeSession:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.calls = 0

    def get(self, *_args, **_kwargs):
        self.calls += 1
        return self.responses.pop(0)


@pytest.fixture(scope="module")
def exact_plan():
    decisions = pd.read_parquet(DEFAULT_V3 / "daily_decisions.parquet")
    frozen = pd.read_parquet(DEFAULT_FROZEN_ERA)
    return build_request_plan(decisions, frozen)


def test_frozen_inputs_have_expected_hashes():
    assert sha256_file(DEFAULT_V3 / "execution_manifest.json") == (
        "c7976317e6d94b8d1ca55b0bd618226c769914280bd7896ca36b5e0ca2cf3c80"
    )
    assert sha256_file(DEFAULT_V3 / "daily_decisions.parquet") == (
        "8e7cbc46805ee080b9d038927582aeb33e2713eaed4da48bc29ca0d19baff398"
    )
    assert sha256_file(DEFAULT_V3 / "field_seasons.parquet") == (
        "3846e8372c62a88574a0ce84bf94cba14779e3c27a7e4d028c58303d84188626"
    )
    assert sha256_file(DEFAULT_FROZEN_ERA) == (
        "112181fc0fc7215e63211ea8c2a8b0c5f6477ae04abe98b0ec14731c5500cc7a"
    )
    assert sha256_file(DEFAULT_FROZEN_META) == (
        "9960638b6153f34d7a38fca4ac57f4e1db591d9d578f221978dbfacbfbfef098"
    )


def test_exact_v3_missing_plan_counts_and_privacy(exact_plan):
    plan, audit = exact_plan
    assert audit == {
        "years": [2020, 2021, 2022, 2023, 2024, 2025],
        "service_field_days": 6924,
        "currently_computable_field_days": 4105,
        "missing_field_days": 2819,
        "unique_missing_weather_cell_dates": 2723,
        "weather_cells": 31,
        "request_ranges": 50,
        "target_days_in_ranges": 2723,
        "target_date_min": "2020-07-07",
        "target_date_max": "2025-09-28",
        "plan_contains_field_ids_or_outcomes": False,
        "privacy": "local_only_coarse_weather_cells_derived_from_private_field_selection",
    }
    yearly = plan.groupby(plan["target_start"].dt.year).agg(
        ranges=("request_id", "size"), days=("target_days", "sum")
    )
    assert yearly.to_dict(orient="index") == {
        2020: {"ranges": 11, "days": 645},
        2021: {"ranges": 11, "days": 576},
        2022: {"ranges": 6, "days": 333},
        2023: {"ranges": 12, "days": 626},
        2024: {"ranges": 8, "days": 414},
        2025: {"ranges": 2, "days": 129},
    }
    assert not {
        "field_uid",
        "field_season",
        "target_class",
        "first_recorded_event_date",
        "label_status",
    }.intersection(plan.columns)


def test_plan_uses_exact_frozen_request_profile_and_complete_overlap(exact_plan):
    plan, _ = exact_plan
    for row in plan.to_dict(orient="records"):
        params = _request_parameters(row)
        profile = {
            "endpoint": FROZEN_ERA5_ENDPOINT,
            **params,
            "profile_id": FROZEN_ERA5_PROFILE_ID,
            "statistical_downscaling": False,
        }
        validate_frozen_era5_request_profile(profile)
        assert tuple(params["hourly"].split(",")) == FROZEN_ERA5_VARIABLES
        assert pd.Timestamp(row["api_start"]) == pd.Timestamp(row["target_start"]) - pd.Timedelta(days=2)
        assert pd.Timestamp(row["overlap_date"]) == pd.Timestamp(row["target_start"]) - pd.Timedelta(days=1)
        assert row["request_id"] == _canonical_sha256(
            {"endpoint": FROZEN_ERA5_ENDPOINT, "parameters": params}
        )


def test_download_cache_is_content_verified_and_resumed_without_network(tmp_path: Path):
    row = {
        "request_id": "",
        "weather_cell": "57.25_21.50",
        "latitude": 57.25,
        "longitude": 21.5,
        "api_start": pd.Timestamp("2025-06-10"),
        "api_end": pd.Timestamp("2025-06-12"),
    }
    params = _request_parameters(row)
    row["request_id"] = _canonical_sha256(
        {"endpoint": FROZEN_ERA5_ENDPOINT, "parameters": params}
    )
    payload = json.dumps({"timezone": "GMT", "hourly": {}, "error": False}).encode()
    session = FakeSession([FakeResponse(payload)])
    status, receipt = _download_one(
        session, row, tmp_path, retries=1, timeout_seconds=1
    )
    assert status == "downloaded"
    assert receipt["content_sha256"] == hashlib.sha256(payload).hexdigest()
    assert session.calls == 1

    status, second = _download_one(
        session, row, tmp_path, retries=1, timeout_seconds=1
    )
    assert status == "cached"
    assert second["content_sha256"] == receipt["content_sha256"]
    assert session.calls == 1


def test_corrupt_cached_response_is_quarantined_and_refetched(tmp_path: Path):
    row = {
        "request_id": "",
        "weather_cell": "57.25_21.50",
        "latitude": 57.25,
        "longitude": 21.5,
        "api_start": pd.Timestamp("2025-06-10"),
        "api_end": pd.Timestamp("2025-06-12"),
    }
    params = _request_parameters(row)
    row["request_id"] = _canonical_sha256(
        {"endpoint": FROZEN_ERA5_ENDPOINT, "parameters": params}
    )
    first = json.dumps({"version": 1}).encode()
    second = json.dumps({"version": 2}).encode()
    initial_session = FakeSession([FakeResponse(first)])
    _download_one(initial_session, row, tmp_path, retries=1, timeout_seconds=1)
    folder = tmp_path / row["request_id"]
    (folder / "response.json.gz").write_bytes(b"corrupt")
    replacement_session = FakeSession([FakeResponse(second)])
    status, receipt = _download_one(
        replacement_session, row, tmp_path, retries=1, timeout_seconds=1
    )
    assert status == "downloaded"
    assert receipt["content_sha256"] == hashlib.sha256(second).hexdigest()
    assert replacement_session.calls == 1
    assert list(tmp_path.glob(f"{row['request_id']}.invalid.*"))


def test_unrecognised_nonempty_run_directory_is_never_overwritten(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "user-file.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _ensure_run(run)
    assert (run / "user-file.txt").read_text(encoding="utf-8") == "keep"
