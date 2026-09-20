"""Resumable ERA5 tail backfill and frozen-policy Hutton replay.

This module is deliberately separate from the first-cycle pipeline.  It never
rewrites the frozen Valeriy weather bundle or the v3 results, and it does not
fit or tune a model.  It only completes absent ERA5 days using the exact
historical request profile and replays the already fixed Hutton notification
policy sequentially.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import requests

from .early_warning_core import _episode_features, sha256_file
from .early_warning_cycle2_pipeline import verify_v3
from .early_warning_models import Policy, burden_metrics, event_metrics, simulate_policy
from .shadow_sources import (
    FROZEN_ERA5_ENDPOINT,
    FROZEN_ERA5_PROFILE_ID,
    build_frozen_era5_archive_params,
    parse_frozen_era5_hourly,
    validate_frozen_era5_request_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V3 = REPO_ROOT / "results/late_blight_early_warning/20260910_first_cycle_v3"
DEFAULT_FROZEN_ERA = (
    REPO_ROOT
    / "docs/extra/vaad_pipeline_repro_20260905/frozen_external/era5_potato_daily.parquet"
)
DEFAULT_FROZEN_META = (
    REPO_ROOT
    / "docs/extra/vaad_pipeline_repro_20260905/frozen_external/era5_metadata.csv"
)
DEFAULT_RUN = REPO_ROOT / "results/late_blight_early_warning/20260915_era5_hutton_backfill_v1"
EXPECTED_V3_MANIFEST_SHA256 = (
    "c7976317e6d94b8d1ca55b0bd618226c769914280bd7896ca36b5e0ca2cf3c80"
)
RUN_FORMAT = "agro_phenology_era5_hutton_backfill_v1"
TEST_YEARS = tuple(range(2020, 2026))
POLICY = Policy(0.5, 7, 15, "policy_v1_fixed_binary_rule")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, payload: str) -> None:
    _atomic_bytes(path, payload.encode("utf-8"))


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
    )


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _gzip_bytes(payload: bytes) -> bytes:
    # mtime=0 makes the cache artifact deterministic for the exact HTTP bytes.
    import io

    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as stream:
        stream.write(payload)
    return output.getvalue()


def _read_gzip(path: Path) -> bytes:
    with gzip.open(path, "rb") as stream:
        return stream.read()


def _normalise_dates(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_datetime(result[column]).dt.normalize()
    return result


def _weather_cell_coordinates(cell: str) -> tuple[float, float]:
    try:
        latitude, longitude = (float(value) for value in str(cell).split("_", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid weather_cell: {cell!r}") from exc
    return latitude, longitude


def _consecutive_ranges(days: Iterable[pd.Timestamp]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ordered = sorted({pd.Timestamp(day).normalize() for day in days})
    if not ordered:
        return []
    result: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = previous = ordered[0]
    for day in ordered[1:]:
        if (day - previous).days != 1 or day.year != previous.year:
            result.append((start, previous))
            start = day
        previous = day
    result.append((start, previous))
    return result


def build_request_plan(
    decisions: pd.DataFrame,
    frozen_era: pd.DataFrame,
    *,
    years: Iterable[int] = TEST_YEARS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build minimal missing ERA5 ranges from the saved v3 service calendar."""
    selected_years = tuple(sorted({int(year) for year in years}))
    required = {
        "weather_cell",
        "season",
        "evaluation_field_day",
        "hutton_score",
        "feature_cutoff_era_episode_date",
        "issue_date",
    }
    missing_columns = required.difference(decisions.columns)
    if missing_columns:
        raise ValueError(f"daily_decisions lacks {sorted(missing_columns)}")
    source = decisions.copy()
    source["feature_cutoff_era_episode_date"] = pd.to_datetime(
        source["feature_cutoff_era_episode_date"]
    ).dt.normalize()
    source["issue_date"] = pd.to_datetime(source["issue_date"]).dt.normalize()
    if (
        source["feature_cutoff_era_episode_date"]
        > source["issue_date"] - pd.Timedelta(days=2)
    ).any():
        raise AssertionError("A saved weather cutoff is later than issue_date - 2 days")
    service = source[
        source["evaluation_field_day"].astype(bool)
        & source["season"].isin(selected_years)
    ].copy()
    missing_field_days = service[service["hutton_score"].isna()].copy()
    targets = (
        missing_field_days[["weather_cell", "feature_cutoff_era_episode_date"]]
        .drop_duplicates()
        .rename(columns={"feature_cutoff_era_episode_date": "target_date"})
        .sort_values(["weather_cell", "target_date"])
        .reset_index(drop=True)
    )
    era = frozen_era[["weather_cell", "date", "day_status"]].copy()
    era["date"] = pd.to_datetime(era["date"]).dt.normalize()
    if era.duplicated(["weather_cell", "date"]).any():
        raise ValueError("Frozen ERA5 contains duplicate weather_cell/date keys")
    existing_keys = set(zip(era["weather_cell"], era["date"]))
    target_keys = list(zip(targets["weather_cell"], targets["target_date"]))
    already_present = [key for key in target_keys if key in existing_keys]
    if already_present:
        raise AssertionError(
            "Missing v3 Hutton scores include cutoff dates already present in frozen ERA5"
        )

    rows: list[dict[str, Any]] = []
    for cell, group in targets.groupby("weather_cell", sort=True):
        latitude, longitude = _weather_cell_coordinates(cell)
        for target_start, target_end in _consecutive_ranges(group["target_date"]):
            overlap_date = target_start - pd.Timedelta(days=1)
            if (cell, overlap_date) not in existing_keys:
                raise AssertionError(
                    f"Frozen predecessor is missing for range {cell}/{target_start.date()}"
                )
            # One local overlap day is fetched for compatibility control.  To
            # make it complete in Europe/Riga, the UTC API request starts one
            # calendar day earlier again.
            local_start = overlap_date
            api_start = local_start - pd.Timedelta(days=1)
            params = build_frozen_era5_archive_params(
                latitude, longitude, api_start.date(), target_end.date()
            )
            request_contract = {
                "endpoint": FROZEN_ERA5_ENDPOINT,
                **params,
                "statistical_downscaling": False,
                "profile_id": FROZEN_ERA5_PROFILE_ID,
            }
            validate_frozen_era5_request_profile(request_contract)
            request_id = _canonical_sha256(
                {"endpoint": FROZEN_ERA5_ENDPOINT, "parameters": params}
            )
            rows.append(
                {
                    "request_id": request_id,
                    "weather_cell": cell,
                    "latitude": latitude,
                    "longitude": longitude,
                    "target_start": target_start,
                    "target_end": target_end,
                    "overlap_date": overlap_date,
                    "api_start": api_start,
                    "api_end": target_end,
                    "target_days": int((target_end - target_start).days + 1),
                }
            )
    plan = pd.DataFrame(rows).sort_values(
        ["weather_cell", "target_start", "target_end"]
    ).reset_index(drop=True)
    if plan["request_id"].duplicated().any():
        raise AssertionError("Duplicate request ids in ERA5 backfill plan")
    audit = {
        "years": list(selected_years),
        "service_field_days": int(len(service)),
        "currently_computable_field_days": int(service["hutton_score"].notna().sum()),
        "missing_field_days": int(len(missing_field_days)),
        "unique_missing_weather_cell_dates": int(len(targets)),
        "weather_cells": int(targets["weather_cell"].nunique()),
        "request_ranges": int(len(plan)),
        "target_days_in_ranges": int(plan["target_days"].sum()),
        "target_date_min": str(targets["target_date"].min().date()) if len(targets) else None,
        "target_date_max": str(targets["target_date"].max().date()) if len(targets) else None,
        "plan_contains_field_ids_or_outcomes": bool(
            set(plan.columns)
            & {
                "field_uid",
                "field_season",
                "target_class",
                "first_recorded_event_date",
                "label_status",
            }
        ),
        "privacy": "local_only_coarse_weather_cells_derived_from_private_field_selection",
    }
    return plan, audit


def _request_parameters(row: Mapping[str, Any]) -> dict[str, Any]:
    return build_frozen_era5_archive_params(
        float(row["latitude"]),
        float(row["longitude"]),
        pd.Timestamp(row["api_start"]).date(),
        pd.Timestamp(row["api_end"]).date(),
    )


def _cache_paths(cache_root: Path, request_id: str) -> dict[str, Path]:
    folder = cache_root / request_id
    return {
        "folder": folder,
        "request": folder / "request.json",
        "response": folder / "response.json.gz",
        "receipt": folder / "receipt.json",
    }


def _validate_cached_response(
    paths: Mapping[str, Path], request_id: str, request_payload: Mapping[str, Any]
) -> tuple[bytes, dict[str, Any]]:
    if not all(paths[name].is_file() for name in ("request", "response", "receipt")):
        raise FileNotFoundError("cache entry is incomplete")
    stored_request = json.loads(paths["request"].read_text(encoding="utf-8"))
    if _canonical_sha256(stored_request) != _canonical_sha256(request_payload):
        raise ValueError("cached request contract differs from plan")
    if stored_request.get("request_id") != request_id:
        raise ValueError("cached request id differs from folder")
    raw = _read_gzip(paths["response"])
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    if hashlib.sha256(raw).hexdigest() != receipt.get("content_sha256"):
        raise ValueError("cached response SHA-256 mismatch")
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict) or parsed.get("error"):
        raise ValueError("cached response is not a successful JSON object")
    return raw, receipt


def _quarantine_cache(paths: Mapping[str, Path]) -> Path | None:
    folder = paths["folder"]
    if not folder.exists():
        return None
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = folder.with_name(f"{folder.name}.invalid.{suffix}")
    os.replace(folder, destination)
    return destination


def _retry_delay(response: requests.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(120.0, max(1.0, float(retry_after)))
            except ValueError:
                pass
    return min(60.0, (2 ** attempt) + random.random())


def _download_one(
    session: requests.Session,
    row: Mapping[str, Any],
    cache_root: Path,
    *,
    retries: int,
    timeout_seconds: float,
) -> tuple[str, dict[str, Any]]:
    request_id = str(row["request_id"])
    params = _request_parameters(row)
    request_profile = {
        "endpoint": FROZEN_ERA5_ENDPOINT,
        **params,
        "statistical_downscaling": False,
        "profile_id": FROZEN_ERA5_PROFILE_ID,
    }
    validate_frozen_era5_request_profile(request_profile)
    request_payload = {
        "format": "open_meteo_exact_request_v1",
        "request_id": request_id,
        "endpoint": FROZEN_ERA5_ENDPOINT,
        "parameters": params,
        "profile_id": FROZEN_ERA5_PROFILE_ID,
        "statistical_downscaling": False,
    }
    expected_id = _canonical_sha256(
        {"endpoint": FROZEN_ERA5_ENDPOINT, "parameters": params}
    )
    if expected_id != request_id:
        raise AssertionError("Request id no longer matches exact endpoint and parameters")
    paths = _cache_paths(cache_root, request_id)
    try:
        raw, receipt = _validate_cached_response(paths, request_id, request_payload)
        return "cached", {**receipt, "response_bytes": len(raw)}
    except FileNotFoundError:
        if paths["folder"].exists():
            _quarantine_cache(paths)
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        _quarantine_cache(paths)

    response: requests.Response | None = None
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                FROZEN_ERA5_ENDPOINT,
                params=params,
                timeout=(15, timeout_seconds),
                headers={"User-Agent": "agro-phenology-era5-backfill/1.0"},
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(
                    f"temporary HTTP {response.status_code}", response=response
                )
            response.raise_for_status()
            raw = response.content
            parsed = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed, dict) or parsed.get("error"):
                raise ValueError(f"provider returned an error payload: {parsed}")
            paths["folder"].mkdir(parents=True, exist_ok=False)
            receipt = {
                "format": "open_meteo_http_receipt_v1",
                "request_id": request_id,
                "retrieved_at_utc": _utc_now(),
                "http_status": int(response.status_code),
                "content_sha256": hashlib.sha256(raw).hexdigest(),
                "response_bytes": len(raw),
                "content_type": response.headers.get("Content-Type"),
                "provider_generation_time_ms": parsed.get("generationtime_ms"),
            }
            _atomic_json(paths["request"], request_payload)
            _atomic_bytes(paths["response"], _gzip_bytes(raw))
            _atomic_json(paths["receipt"], receipt)
            # Re-read from disk before treating the request as complete.
            raw_checked, receipt_checked = _validate_cached_response(
                paths, request_id, request_payload
            )
            return "downloaded", {**receipt_checked, "response_bytes": len(raw_checked)}
        except (
            requests.RequestException,
            json.JSONDecodeError,
            UnicodeDecodeError,
            ValueError,
            OSError,
        ) as exc:
            last_error = exc
            temporary = bool(
                isinstance(exc, requests.RequestException)
                and (
                    response is None
                    or response.status_code == 429
                    or response.status_code >= 500
                )
            )
            if attempt >= retries or not temporary:
                break
            time.sleep(_retry_delay(response, attempt))
    raise RuntimeError(
        f"ERA5 request {request_id[:12]} failed after {retries} attempts: {last_error}"
    ) from last_error


class StopController:
    def __init__(self) -> None:
        self.requested = False
        self.signal_name: str | None = None
        self._previous: dict[int, Any] = {}

    def _handle(self, signum: int, _frame: Any) -> None:
        self.requested = True
        self.signal_name = signal.Signals(signum).name
        print(
            f"\nПолучен {self.signal_name}; остановка после текущего сохранённого запроса...",
            flush=True,
        )

    def __enter__(self) -> "StopController":
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)
        return self

    def __exit__(self, *_args: Any) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)


@contextmanager
def exclusive_run_lock(run_dir: Path):
    lock_path = run_dir / ".download.lock"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        try:
            holder = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            holder = {"status": "unreadable_lock"}
        raise RuntimeError(f"Another backfill process holds {lock_path}: {holder}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "created_at_utc": _utc_now()}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def download_plan(
    plan: pd.DataFrame,
    run_dir: Path,
    *,
    retries: int = 6,
    timeout_seconds: float = 90.0,
    spacing_seconds: float = 0.4,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Download all requests with verified per-request cache and checkpoint."""
    cache_root = run_dir / "raw"
    checkpoint_path = run_dir / "download_checkpoint.json"
    own_session = session is None
    session = session or requests.Session()
    completed: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        with exclusive_run_lock(run_dir), StopController() as stop:
            for position, row in enumerate(plan.to_dict(orient="records"), 1):
                if stop.requested:
                    break
                status, receipt = _download_one(
                    session,
                    row,
                    cache_root,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                )
                # Rate-limit only actual network responses.  Verified cache
                # hits during resume should be immediate.
                if status == "downloaded" and spacing_seconds > 0:
                    time.sleep(spacing_seconds)
                completed.append(
                    {
                        "request_id": row["request_id"],
                        "status": status,
                        "content_sha256": receipt["content_sha256"],
                        "response_bytes": int(receipt["response_bytes"]),
                    }
                )
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = position / elapsed
                eta = (len(plan) - position) / rate if rate else np.nan
                print(
                    f"ERA5 [{position:02d}/{len(plan):02d} {100*position/len(plan):5.1f}%] "
                    f"{status}; ETA {eta:5.0f} c",
                    flush=True,
                )
                _atomic_json(
                    checkpoint_path,
                    {
                        "format": "era5_backfill_checkpoint_v1",
                        "updated_at_utc": _utc_now(),
                        "total_requests": int(len(plan)),
                        "completed_requests": int(position),
                        "stopped_by_signal": stop.signal_name,
                        "entries": completed,
                    },
                )
            if stop.requested:
                _atomic_json(
                    checkpoint_path,
                    {
                        "format": "era5_backfill_checkpoint_v1",
                        "updated_at_utc": _utc_now(),
                        "total_requests": int(len(plan)),
                        "completed_requests": int(len(completed)),
                        "stopped_by_signal": stop.signal_name,
                        "entries": completed,
                    },
                )
                raise KeyboardInterrupt(
                    f"Backfill stopped after {len(completed)}/{len(plan)} requests"
                )
    finally:
        if own_session:
            session.close()
    return {
        "total_requests": int(len(plan)),
        "completed_requests": int(len(completed)),
        "downloaded": int(sum(item["status"] == "downloaded" for item in completed)),
        "cached": int(sum(item["status"] == "cached" for item in completed)),
        "verified_raw_responses_total": int(len(completed)),
        "response_bytes": int(sum(item["response_bytes"] for item in completed)),
    }


def _frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
        return True
    except AssertionError:
        return False


def _write_parquet_idempotent(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        existing = pd.read_parquet(path)
        if not _frames_equal(existing, frame):
            raise FileExistsError(f"Refusing to replace different artifact: {path}")
        return
    _atomic_parquet(path, frame)


def assemble_weather_bundle(
    plan: pd.DataFrame,
    run_dir: Path,
    frozen_era_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Parse exact raw responses, validate overlaps, and append only new keys."""
    frozen_hash_before = sha256_file(frozen_era_path)
    frozen = pd.read_parquet(frozen_era_path)
    frozen["date"] = pd.to_datetime(frozen["date"]).dt.normalize()
    if frozen.duplicated(["weather_cell", "date"]).any():
        raise ValueError("Frozen ERA5 contains duplicate keys")
    frozen_index = frozen.set_index(["weather_cell", "date"], drop=False)
    new_tables: list[pd.DataFrame] = []
    overlap_records: list[dict[str, Any]] = []
    raw_receipts: list[dict[str, Any]] = []
    numeric_columns = [
        "minimum_temperature_c",
        "maximum_temperature_c",
        "temperature_mean_c",
        "relative_humidity_mean_pct",
        "precipitation_sum_mm",
        "soil_temperature_c",
        "soil_moisture",
        "radiation_mean_wm2",
        "wind_mean_kmh",
    ]
    exact_columns = [
        "expected_hours",
        "temperature_valid_hours",
        "humidity_valid_hours",
        "precipitation_valid_hours",
        "high_humidity_hours",
        "day_status",
        "accepted",
    ]
    for row in plan.to_dict(orient="records"):
        request_id = str(row["request_id"])
        paths = _cache_paths(run_dir / "raw", request_id)
        params = _request_parameters(row)
        request_payload = {
            "format": "open_meteo_exact_request_v1",
            "request_id": request_id,
            "endpoint": FROZEN_ERA5_ENDPOINT,
            "parameters": params,
            "profile_id": FROZEN_ERA5_PROFILE_ID,
            "statistical_downscaling": False,
        }
        raw, receipt = _validate_cached_response(paths, request_id, request_payload)
        profile = {
            "endpoint": FROZEN_ERA5_ENDPOINT,
            **params,
            "statistical_downscaling": False,
            "profile_id": FROZEN_ERA5_PROFILE_ID,
        }
        daily, parsed_metadata = parse_frozen_era5_hourly(
            raw, weather_cell=str(row["weather_cell"]), request_profile=profile
        )
        daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
        overlap_date = pd.Timestamp(row["overlap_date"]).normalize()
        target_start = pd.Timestamp(row["target_start"]).normalize()
        target_end = pd.Timestamp(row["target_end"]).normalize()
        relevant = daily[daily["date"].between(overlap_date, target_end)].copy()
        overlap = relevant[relevant["date"].eq(overlap_date)]
        if len(overlap) != 1:
            raise AssertionError(
                f"Request {request_id[:12]} does not contain one complete overlap day"
            )
        old = frozen_index.loc[(str(row["weather_cell"]), overlap_date)]
        new = overlap.iloc[0]
        exact_match = all(old[column] == new[column] for column in exact_columns)
        max_delta = 0.0
        for column in numeric_columns:
            old_value, new_value = old[column], new[column]
            if pd.isna(old_value) and pd.isna(new_value):
                delta = 0.0
            elif pd.isna(old_value) or pd.isna(new_value):
                delta = np.inf
            else:
                delta = abs(float(old_value) - float(new_value))
            max_delta = max(max_delta, delta)
        overlap_records.append(
            {
                "request_id": request_id,
                "exact_status_and_count_match": bool(exact_match),
                "max_numeric_absolute_delta": float(max_delta),
            }
        )
        if not exact_match or not np.isfinite(max_delta) or max_delta > 1e-12:
            _atomic_json(
                run_dir / "overlap_validation.json",
                {
                    "status": "source_revision_mismatch",
                    "checked_ranges": len(overlap_records),
                    "records": overlap_records,
                },
            )
            raise RuntimeError(
                "Current ERA5 response differs from frozen overlap; automatic merge stopped"
            )
        target = relevant[relevant["date"].between(target_start, target_end)].copy()
        expected_dates = set(pd.date_range(target_start, target_end, freq="D"))
        actual_dates = set(target["date"])
        if actual_dates != expected_dates:
            raise RuntimeError(
                f"Request {request_id[:12]} has incomplete target local-date coverage"
            )
        if not target["accepted"].all():
            raise RuntimeError(f"Request {request_id[:12]} contains incomplete target days")
        if target[
            ["soil_temperature_c", "soil_moisture", "radiation_mean_wm2", "wind_mean_kmh"]
        ].isna().any().any():
            raise RuntimeError(f"Request {request_id[:12]} lacks non-Hutton frozen variables")
        target["cache_key"] = request_id
        target = target[frozen.columns]
        new_tables.append(target)
        raw_receipts.append(
            {
                "request_id": request_id,
                "content_sha256": receipt["content_sha256"],
                "response_bytes": int(receipt["response_bytes"]),
                "retrieved_at_utc": receipt["retrieved_at_utc"],
                "parsed_metadata": parsed_metadata,
            }
        )
    backfill = pd.concat(new_tables, ignore_index=True).sort_values(
        ["weather_cell", "date"]
    ).reset_index(drop=True)
    if backfill.duplicated(["weather_cell", "date"]).any():
        raise AssertionError("Backfill contains duplicate weather keys")
    frozen_keys = set(zip(frozen["weather_cell"], frozen["date"]))
    overlap_keys = [
        key for key in zip(backfill["weather_cell"], backfill["date"]) if key in frozen_keys
    ]
    if overlap_keys:
        raise AssertionError("Target backfill would replace frozen rows")
    extended = pd.concat([frozen, backfill], ignore_index=True).sort_values(
        ["weather_cell", "date"]
    ).reset_index(drop=True)
    if extended.duplicated(["weather_cell", "date"]).any():
        raise AssertionError("Extended ERA5 contains duplicate weather keys")
    backfill_path = run_dir / "era5_backfill_daily.parquet"
    extended_path = run_dir / "era5_potato_daily_extended.parquet"
    _write_parquet_idempotent(backfill_path, backfill)
    _write_parquet_idempotent(extended_path, extended)
    overlap_summary = {
        "status": "passed",
        "checked_ranges": len(overlap_records),
        "exact_status_and_count_matches": int(
            sum(record["exact_status_and_count_match"] for record in overlap_records)
        ),
        "maximum_numeric_absolute_delta": float(
            max(record["max_numeric_absolute_delta"] for record in overlap_records)
        ),
    }
    _atomic_json(run_dir / "overlap_validation.json", overlap_summary)
    source_manifest = {
        "format": "era5_backfill_source_manifest_v1",
        "created_at_utc": _utc_now(),
        "endpoint": FROZEN_ERA5_ENDPOINT,
        "profile_id": FROZEN_ERA5_PROFILE_ID,
        "provider_product": "Open-Meteo Historical Weather API; models=era5",
        "historical_retrieval_semantics": (
            "retrospectively retrieved reanalysis; not proof of contemporaneous availability"
        ),
        "raw_responses": raw_receipts,
        "overlap_validation": overlap_summary,
    }
    _atomic_json(run_dir / "source_manifest.json", source_manifest)
    frozen_hash_after = sha256_file(frozen_era_path)
    if frozen_hash_after != frozen_hash_before:
        raise AssertionError("Frozen ERA5 input changed during assembly")
    audit = {
        "frozen_rows": int(len(frozen)),
        "backfill_rows": int(len(backfill)),
        "extended_rows": int(len(extended)),
        "frozen_sha256_before": frozen_hash_before,
        "frozen_sha256_after": frozen_hash_after,
        "backfill_sha256": sha256_file(backfill_path),
        "extended_sha256": sha256_file(extended_path),
        "overlap_validation": overlap_summary,
    }
    return extended_path, audit


def _fold_id(year: int) -> str:
    return "test_2026_partial" if year == 2026 else f"test_{year}"


def _compare_replay_identity(
    replay: pd.DataFrame, saved: pd.DataFrame, year: int
) -> None:
    keys = ["field_season", "issue_date"]
    columns = [
        "score",
        "score_status",
        "message_issued",
        "alarm_active",
        "suppressed_repeat",
        "action_reason",
    ]
    expected = saved[
        saved["model_code"].eq("hutton")
        & saved["evaluation_scope"].eq("service_calendar")
        & saved["fold_id"].eq(_fold_id(year))
    ][keys + columns].copy()
    actual = replay[keys + columns].copy()
    expected["issue_date"] = pd.to_datetime(expected["issue_date"])
    actual["issue_date"] = pd.to_datetime(actual["issue_date"])
    expected = expected.sort_values(keys).reset_index(drop=True)
    actual = actual.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)


def _summary_row(
    states: pd.DataFrame,
    seasons: pd.DataFrame,
    model_code: str,
    period: str,
    years: Iterable[int],
) -> tuple[dict[str, Any], pd.DataFrame]:
    years = tuple(int(year) for year in years)
    subset = states[states["season"].isin(years)].copy()
    registry = seasons[seasons["season"].isin(years)].copy()
    event, hits = event_metrics(
        subset,
        registry,
        model_code,
        period,
        "A_plus_B",
        "service_calendar",
    )
    burden = burden_metrics(
        subset, model_code, period, "A_plus_B", "service_calendar"
    )
    row = {
        **event,
        **burden,
        "period": period,
        "year_start": min(years),
        "year_end": max(years),
        "within_research_budget": bool(
            burden["messages_per_30_field_days"] <= 2.0
            and burden["active_alarm_fraction"] <= 0.5
        ),
    }
    return row, pd.DataFrame(hits)


def _paired_category(candidate: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, int]:
    left = candidate[candidate["warnable_event"]][
        ["field_season", "timely_hit"]
    ].rename(columns={"timely_hit": "candidate"})
    right = baseline[baseline["warnable_event"]][
        ["field_season", "timely_hit"]
    ].rename(columns={"timely_hit": "baseline"})
    paired = left.merge(right, on="field_season", how="outer", validate="one_to_one")
    if paired[["candidate", "baseline"]].isna().any().any():
        raise AssertionError("Comparison changed the warnable event population")
    return {
        "both": int((paired["candidate"] & paired["baseline"]).sum()),
        "candidate_only": int((paired["candidate"] & ~paired["baseline"]).sum()),
        "baseline_only": int((~paired["candidate"] & paired["baseline"]).sum()),
        "neither": int((~paired["candidate"] & ~paired["baseline"]).sum()),
        "events": int(len(paired)),
    }


def _bootstrap_comparison(
    candidate_states: pd.DataFrame,
    baseline_states: pd.DataFrame,
    candidate_hits: pd.DataFrame,
    baseline_hits: pd.DataFrame,
    *,
    baseline_code: str,
    draws: int = 20_000,
    seed: int = 20260915,
) -> dict[str, Any]:
    years = np.array(TEST_YEARS, dtype=int)
    rows: list[dict[str, float]] = []
    for year in years:
        c_states = candidate_states[candidate_states["season"].eq(year)]
        b_states = baseline_states[baseline_states["season"].eq(year)]
        c_hits = candidate_hits[
            candidate_hits["season"].eq(year) & candidate_hits["warnable_event"]
        ]
        b_hits = baseline_hits[
            baseline_hits["season"].eq(year) & baseline_hits["warnable_event"]
        ]
        rows.append(
            {
                "opportunities": float(len(c_hits)),
                "candidate_hits": float(c_hits["timely_hit"].sum()),
                "baseline_hits": float(b_hits["timely_hit"].sum()),
                "field_days": float(c_states["evaluation_scope_day"].sum()),
                "candidate_messages": float(c_states["message_issued"].sum()),
                "baseline_messages": float(b_states["message_issued"].sum()),
                "candidate_alarm_days": float(c_states["alarm_active"].sum()),
                "baseline_alarm_days": float(b_states["alarm_active"].sum()),
            }
        )
    matrix = pd.DataFrame(rows).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(years), size=(draws, len(years)))
    totals = matrix[samples].sum(axis=1)
    opportunity = totals[:, 0]
    field_days = totals[:, 3]
    deltas = {
        "delta_timely_recall": (totals[:, 1] - totals[:, 2]) / opportunity,
        "delta_messages_per_30_field_days": 30 * (totals[:, 4] - totals[:, 5]) / field_days,
        "delta_active_alarm_fraction": (totals[:, 6] - totals[:, 7]) / field_days,
    }
    result: dict[str, Any] = {
        "comparison": f"hutton_era5_completed_vs_{baseline_code}",
        "method": "paired_year_block_bootstrap_six_retrospective_years",
        "draws": draws,
        "seed": seed,
    }
    for name, values in deltas.items():
        result[f"{name}_estimate"] = float(
            (matrix[:, 1].sum() - matrix[:, 2].sum()) / matrix[:, 0].sum()
            if name == "delta_timely_recall"
            else 30 * (matrix[:, 4].sum() - matrix[:, 5].sum()) / matrix[:, 3].sum()
            if name == "delta_messages_per_30_field_days"
            else (matrix[:, 6].sum() - matrix[:, 7].sum()) / matrix[:, 3].sum()
        )
        result[f"{name}_p025"] = float(np.quantile(values, 0.025))
        result[f"{name}_p975"] = float(np.quantile(values, 0.975))
    return result


def replay_hutton(
    extended_era_path: Path,
    v3_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """Replace only Hutton scores and replay the fixed policy in full order."""
    decisions = pd.read_parquet(v3_dir / "daily_decisions.parquet")
    seasons = pd.read_parquet(v3_dir / "field_seasons.parquet")
    saved_states = pd.read_parquet(v3_dir / "alarm_states.parquet")
    decisions["issue_date"] = pd.to_datetime(decisions["issue_date"])
    decisions["feature_cutoff_era_episode_date"] = pd.to_datetime(
        decisions["feature_cutoff_era_episode_date"]
    )
    extended = pd.read_parquet(extended_era_path)
    episodes = _episode_features(extended)[
        ["weather_cell", "cutoff_date", "era_hutton_pair_now"]
    ].rename(columns={"era_hutton_pair_now": "era_hutton_pair_completed"})
    updated = decisions.merge(
        episodes,
        left_on=["weather_cell", "feature_cutoff_era_episode_date"],
        right_on=["weather_cell", "cutoff_date"],
        how="left",
        validate="many_to_one",
    ).drop(columns="cutoff_date")
    updated["hutton_score_completed"] = updated["era_hutton_pair_completed"].astype(float)
    preexisting = updated["hutton_score"].notna()
    if not np.array_equal(
        updated.loc[preexisting, "hutton_score"].to_numpy(),
        updated.loc[preexisting, "hutton_score_completed"].to_numpy(),
    ):
        raise AssertionError("Backfill changed at least one pre-existing Hutton score")

    old_replay_states: list[pd.DataFrame] = []
    new_states: list[pd.DataFrame] = []
    for year in TEST_YEARS:
        frame = updated[updated["season"].eq(year)].copy()
        old = simulate_policy(
            frame,
            frame["hutton_score"],
            POLICY,
            "service_calendar",
        )
        _compare_replay_identity(old, saved_states, year)
        old["model_code"] = "hutton_frozen_v3"
        old["fold_id"] = _fold_id(year)
        old_replay_states.append(old)
        new = simulate_policy(
            frame,
            frame["hutton_score_completed"],
            POLICY,
            "service_calendar",
        )
        new["model_code"] = "hutton_era5_completed"
        new["fold_id"] = _fold_id(year)
        new_states.append(new)
    old_states = pd.concat(old_replay_states, ignore_index=True)
    completed_states = pd.concat(new_states, ignore_index=True)
    calendar_states = saved_states[
        saved_states["model_code"].eq("calendar_window")
        & saved_states["evaluation_scope"].eq("service_calendar")
        & saved_states["season"].isin(TEST_YEARS)
    ].copy()

    metrics: list[dict[str, Any]] = []
    all_hits: list[pd.DataFrame] = []
    periods: list[tuple[str, tuple[int, ...]]] = [
        *((str(year), (year,)) for year in TEST_YEARS),
        ("2020_2025", TEST_YEARS),
    ]
    main_hits: dict[str, pd.DataFrame] = {}
    for code, states in (
        ("hutton_frozen_v3", old_states),
        ("hutton_era5_completed", completed_states),
        ("calendar_window", calendar_states),
    ):
        for period, years in periods:
            row, hits = _summary_row(states, seasons, code, period, years)
            metrics.append(row)
            hits["period"] = period
            all_hits.append(hits)
            if period == "2020_2025":
                main_hits[code] = hits
    metrics_frame = pd.DataFrame(metrics)
    hits_frame = pd.concat(all_hits, ignore_index=True)
    main_completed = metrics_frame[
        metrics_frame["model_code"].eq("hutton_era5_completed")
        & metrics_frame["period"].eq("2020_2025")
    ].iloc[0]
    if int(main_completed["field_days"]) != 6924:
        raise AssertionError("Main service denominator changed")
    if int(main_completed["events_with_warning_opportunity"]) != 87:
        raise AssertionError("Warnable first-event denominator changed")

    comparisons = []
    for baseline_code in ("hutton_frozen_v3", "calendar_window"):
        category = _paired_category(
            main_hits["hutton_era5_completed"], main_hits[baseline_code]
        )
        candidate_row = metrics_frame[
            metrics_frame["model_code"].eq("hutton_era5_completed")
            & metrics_frame["period"].eq("2020_2025")
        ].iloc[0]
        baseline_row = metrics_frame[
            metrics_frame["model_code"].eq(baseline_code)
            & metrics_frame["period"].eq("2020_2025")
        ].iloc[0]
        comparisons.append(
            {
                "candidate": "hutton_era5_completed",
                "baseline": baseline_code,
                **category,
                "delta_timely_hits": int(
                    candidate_row["timely_hits"] - baseline_row["timely_hits"]
                ),
                "delta_messages": int(candidate_row["messages"] - baseline_row["messages"]),
                "delta_active_alarm_days": int(
                    candidate_row["active_alarm_days"] - baseline_row["active_alarm_days"]
                ),
                "delta_computable_days": int(
                    candidate_row["computable_days"] - baseline_row["computable_days"]
                ),
            }
        )
    comparison_frame = pd.DataFrame(comparisons)
    bootstrap = pd.DataFrame(
        [
            _bootstrap_comparison(
                completed_states,
                baseline_states,
                main_hits["hutton_era5_completed"],
                main_hits[baseline_code],
                baseline_code=baseline_code,
            )
            for baseline_code, baseline_states in (
                ("hutton_frozen_v3", old_states),
                ("calendar_window", calendar_states),
            )
        ]
    )
    computability = []
    for year in TEST_YEARS:
        subset = updated[
            updated["season"].eq(year) & updated["evaluation_field_day"].astype(bool)
        ]
        computability.append(
            {
                "year": year,
                "service_field_days": int(len(subset)),
                "frozen_computable_days": int(subset["hutton_score"].notna().sum()),
                "completed_computable_days": int(
                    subset["hutton_score_completed"].notna().sum()
                ),
                "newly_computable_days": int(
                    (subset["hutton_score"].isna() & subset["hutton_score_completed"].notna()).sum()
                ),
                "remaining_abstention_days": int(
                    subset["hutton_score_completed"].isna().sum()
                ),
            }
        )
    computability_frame = pd.DataFrame(computability)

    predictions = updated[
        updated["season"].isin(TEST_YEARS)
        & updated["evaluation_field_day"].astype(bool)
    ][
        [
            "field_season",
            "season",
            "issue_date",
            "feature_cutoff_era_episode_date",
            "hutton_score",
            "hutton_score_completed",
            "target_class",
            "target_observable",
            "days_to_first_recorded_event",
        ]
    ].copy()
    predictions = predictions.rename(columns={"hutton_score": "hutton_score_frozen_v3"})
    _write_parquet_idempotent(run_dir / "predictions.parquet", predictions)
    _write_parquet_idempotent(run_dir / "alarm_states.parquet", completed_states)
    _write_parquet_idempotent(run_dir / "event_hits.parquet", hits_frame)
    _atomic_csv(run_dir / "metrics.csv", metrics_frame)
    _atomic_csv(run_dir / "comparisons.csv", comparison_frame)
    _atomic_csv(run_dir / "computability_by_year.csv", computability_frame)
    _atomic_csv(run_dir / "paired_year_bootstrap.csv", bootstrap)
    return {
        "identity_replay": "passed_exactly_for_2020_2025",
        "preexisting_scores_unchanged": True,
        "metrics": metrics_frame,
        "comparisons": comparison_frame,
        "computability": computability_frame,
        "bootstrap": bootstrap,
    }


def _format_pct(value: float) -> str:
    return f"{100 * float(value):.1f}%"


def write_report(run_dir: Path, replay: Mapping[str, Any], plan_audit: Mapping[str, Any]) -> None:
    metrics: pd.DataFrame = replay["metrics"]
    main = metrics[metrics["period"].eq("2020_2025")].set_index("model_code")
    new = main.loc["hutton_era5_completed"]
    old = main.loc["hutton_frozen_v3"]
    calendar = main.loc["calendar_window"]
    comparisons: pd.DataFrame = replay["comparisons"]
    vs_old = comparisons[comparisons["baseline"].eq("hutton_frozen_v3")].iloc[0]
    vs_calendar = comparisons[comparisons["baseline"].eq("calendar_window")].iloc[0]
    report = f"""# Дозагрузка ERA5 и повторная оценка правила Хаттона

Дата запуска: 2026-09-15. Это отдельный ретроспективный run; результаты первого цикла v3 и исходный frozen ERA5 не изменялись.

## Что было сделано

Для сервисных дат внешних лет 2020–2025 найдены {plan_audit['missing_field_days']:,} пропущенных поле-дней. Они соответствуют {plan_audit['unique_missing_weather_cell_dates']:,} уникальным датам в {plan_audit['weather_cells']} погодной ячейке и объединены в {plan_audit['request_ranges']} небольших диапазонов. Загружены реальные почасовые ответы Open-Meteo Historical Weather API с точным frozen-профилем `models=era5`, семью исходными переменными, `timezone=UTC`, `cell_selection=nearest`, `elevation=nan`. Каждый raw-ответ сохранён отдельно с хешем; 50 пограничных суток сравнены с исходным frozen-пакетом.

После дозагрузки изменён только вычисляемый score правила Хаттона. Порог 0,5, длительность тревоги 7 дней, cooldown 15 дней, события и окно успеха 3–10 дней оставлены прежними. Вся последовательность сообщений пересчитана заново, поэтому новые сообщения могли сдвигать последующие cooldown.

## Основной результат, 2020–2025

| Метод | Своевременные события | Сообщения | Сообщений / 30 дней | Тревожные дни | Доля тревожных дней | Вычислимые дни |
|---|---:|---:|---:|---:|---:|---:|
| Hutton, исходный frozen v3 | {int(old.timely_hits)}/{int(old.events_with_warning_opportunity)} | {int(old.messages)} | {old.messages_per_30_field_days:.3f} | {int(old.active_alarm_days)} | {_format_pct(old.active_alarm_fraction)} | {int(old.computable_days)}/{int(old.field_days)} |
| Hutton, ERA5 дополнен | {int(new.timely_hits)}/{int(new.events_with_warning_opportunity)} | {int(new.messages)} | {new.messages_per_30_field_days:.3f} | {int(new.active_alarm_days)} | {_format_pct(new.active_alarm_fraction)} | {int(new.computable_days)}/{int(new.field_days)} |
| Календарное окно v3 | {int(calendar.timely_hits)}/{int(calendar.events_with_warning_opportunity)} | {int(calendar.messages)} | {calendar.messages_per_30_field_days:.3f} | {int(calendar.active_alarm_days)} | {_format_pct(calendar.active_alarm_fraction)} | {int(calendar.computable_days)}/{int(calendar.field_days)} |

Вычислимость Hutton выросла на {int(new.computable_days-old.computable_days):,} дней: с {_format_pct(old.computable_fraction)} до {_format_pct(new.computable_fraction)}. Событийная вычислимость и раньше была полной: все 87 событий с возможностью предупреждения уже имели хотя бы один вычислимый день в окне 3–10 дней.

Относительно исходного Hutton: новых своевременно пойманных событий {int(vs_old.candidate_only)}, потерянных {int(vs_old.baseline_only)}, совместно пойманных {int(vs_old.both)}, не пойманных обеими версиями {int(vs_old.neither)}. Изменение нагрузки: {int(vs_old.delta_messages):+d} сообщений и {int(vs_old.delta_active_alarm_days):+d} тревожных дней.

Относительно календаря: только дополненный Hutton поймал {int(vs_calendar.candidate_only)} событий, только календарь — {int(vs_calendar.baseline_only)}, оба — {int(vs_calendar.both)}, никто — {int(vs_calendar.neither)}.

## Вывод

Дозагрузка устраняет техническое воздержание правила Хаттона на историческом сервисном календаре. Само по себе это не означает улучшения прогноза: главный критерий — своевременные первые регистрации при сопоставимой нагрузке — приведён в таблице без изменения знаменателей или политики. Эти годы уже исследовались и не являются новым независимым тестом.

Новые значения получены ретроспективно в 2026 году. Они не доказывают, что ERA5 был бы доступен на дату решения; для живого сервиса остаётся задержка публикации ERA5. Погодные ячейки связаны с приватной выборкой полей, поэтому raw, plan и погодный bundle предназначены только для локального использования.

## Проверки

- исходный replay Hutton точно воспроизвёл сохранённые сообщения v3;
- все ранее вычислимые Hutton-score остались неизменны;
- исходные v3 и frozen ERA5 проверены по SHA-256 до и после запуска;
- raw-ответы имеют отдельные request/content SHA-256 и используются повторно при перезапуске;
- источник проверен на 50 перекрывающихся сутках;
- численная неопределённость сравнения сохранена в `paired_year_bootstrap.csv` как блочный bootstrap по шести годам.
"""
    _atomic_text(run_dir / "report_ru.md", report.replace(",", " ", 0))
    reproduce = f"""# Воспроизведение

Из корня репозитория:

```bash
.venv/bin/python -m agro_phenology.era_hutton_backfill all \\
  --run-dir {run_dir.relative_to(REPO_ROOT)}
```

Повторный запуск не обращается к сети для уже проверенных raw-ответов. Если загрузка была прервана, команда продолжает с первого отсутствующего диапазона. Завершённый run только проверяется и не перезаписывается.

Файлы `request_plan.csv`, `raw/` и погодные parquet локальные: они содержат грубые погодные ячейки, выбранные по приватным полям, и не предназначены для публикации.
"""
    _atomic_text(run_dir / "REPRODUCE.md", reproduce)


def _ensure_run(run_dir: Path) -> None:
    marker = run_dir / ".backfill_run.json"
    if run_dir.exists() and any(run_dir.iterdir()) and not marker.exists():
        raise FileExistsError(
            f"Refusing to use non-empty unrecognised output directory: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        _atomic_json(
            marker,
            {
                "format": RUN_FORMAT,
                "created_at_utc": _utc_now(),
                "private_local_only": True,
            },
        )
    else:
        stored = json.loads(marker.read_text(encoding="utf-8"))
        if stored.get("format") != RUN_FORMAT:
            raise ValueError("Output directory belongs to a different run format")


def _write_source_snapshot(run_dir: Path) -> Path:
    """Persist exact research source text used to make this additive run."""
    sources = [
        REPO_ROOT / "src/agro_phenology/era_hutton_backfill.py",
        REPO_ROOT / "src/agro_phenology/shadow_sources.py",
        REPO_ROOT / "src/agro_phenology/early_warning_core.py",
        REPO_ROOT / "src/agro_phenology/early_warning_models.py",
        REPO_ROOT / "src/agro_phenology/early_warning_cycle2_pipeline.py",
        REPO_ROOT / "tests/test_era_hutton_backfill.py",
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Source snapshot is incomplete: {missing}")
    snapshot = {
        "format": "era5_hutton_backfill_source_snapshot_v1",
        "created_at_utc": _utc_now(),
        "files": [
            {
                "relative_path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
                "utf8_content": path.read_text(encoding="utf-8"),
            }
            for path in sources
        ],
    }
    path = run_dir / "source_snapshot.json"
    _atomic_json(path, snapshot)
    return path


def _write_or_verify_plan(run_dir: Path, plan: pd.DataFrame, audit: Mapping[str, Any]) -> None:
    path = run_dir / "request_plan.csv"
    serialised = plan.copy()
    for column in ("target_start", "target_end", "overlap_date", "api_start", "api_end"):
        serialised[column] = pd.to_datetime(serialised[column]).dt.strftime("%Y-%m-%d")
    if path.exists():
        existing = pd.read_csv(path, dtype={"request_id": str, "weather_cell": str})
        pd.testing.assert_frame_equal(existing, serialised, check_dtype=False, check_exact=True)
    else:
        _atomic_csv(path, serialised)
    _atomic_json(
        run_dir / "request_plan_manifest.json",
        {
            "format": "era5_backfill_request_plan_v1",
            "created_at_utc": _utc_now(),
            "request_plan_sha256": sha256_file(path),
            "audit": dict(audit),
            "profile_id": FROZEN_ERA5_PROFILE_ID,
            "endpoint": FROZEN_ERA5_ENDPOINT,
        },
    )


def _verify_completed_run(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "execution_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for relative, expected in manifest["output_hashes"].items():
        path = run_dir / relative
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            failures.append(relative)
    if failures:
        raise AssertionError(f"Completed backfill artifact hash failures: {failures}")
    for source, expected in manifest["input_hashes"].items():
        path = Path(source)
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            raise AssertionError(f"Completed backfill input changed: {source}")
    source_manifest = json.loads(
        (run_dir / "source_manifest.json").read_text(encoding="utf-8")
    )
    plan = pd.read_csv(
        run_dir / "request_plan.csv", dtype={"request_id": str, "weather_cell": str}
    ).set_index("request_id", drop=False)
    for receipt in source_manifest.get("raw_responses", []):
        request_id = receipt["request_id"]
        if request_id not in plan.index:
            raise AssertionError(f"Raw response is absent from request plan: {request_id}")
        row = plan.loc[request_id].to_dict()
        params = _request_parameters(row)
        request_payload = {
            "format": "open_meteo_exact_request_v1",
            "request_id": request_id,
            "endpoint": FROZEN_ERA5_ENDPOINT,
            "parameters": params,
            "profile_id": FROZEN_ERA5_PROFILE_ID,
            "statistical_downscaling": False,
        }
        paths = _cache_paths(run_dir / "raw", request_id)
        raw, stored_receipt = _validate_cached_response(
            paths, request_id, request_payload
        )
        if hashlib.sha256(raw).hexdigest() != receipt["content_sha256"]:
            raise AssertionError(f"Completed backfill raw response changed: {request_id}")
        if stored_receipt["content_sha256"] != receipt["content_sha256"]:
            raise AssertionError(f"Raw receipt differs from source manifest: {request_id}")
    if len(source_manifest.get("raw_responses", [])) != len(plan):
        raise AssertionError("Source manifest does not cover the full request plan")
    return manifest


def run_all(
    *,
    run_dir: Path = DEFAULT_RUN,
    v3_dir: Path = DEFAULT_V3,
    frozen_era_path: Path = DEFAULT_FROZEN_ERA,
    frozen_meta_path: Path = DEFAULT_FROZEN_META,
    retries: int = 6,
    timeout_seconds: float = 90.0,
    spacing_seconds: float = 0.4,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    v3_dir = v3_dir.resolve()
    frozen_era_path = frozen_era_path.resolve()
    frozen_meta_path = frozen_meta_path.resolve()
    if run_dir == v3_dir or v3_dir in run_dir.parents or frozen_era_path.parent == run_dir:
        raise ValueError("Output directory must be separate from frozen inputs")
    if (run_dir / "execution_manifest.json").is_file():
        manifest = _verify_completed_run(run_dir)
        print("Завершённый run проверен; сетевые запросы и перезапись не требуются.", flush=True)
        return manifest
    _ensure_run(run_dir)
    parent_verification = verify_v3(v3_dir, EXPECTED_V3_MANIFEST_SHA256)
    input_hashes = {
        str(v3_dir / "execution_manifest.json"): sha256_file(v3_dir / "execution_manifest.json"),
        str(v3_dir / "daily_decisions.parquet"): sha256_file(v3_dir / "daily_decisions.parquet"),
        str(v3_dir / "field_seasons.parquet"): sha256_file(v3_dir / "field_seasons.parquet"),
        str(v3_dir / "alarm_states.parquet"): sha256_file(v3_dir / "alarm_states.parquet"),
        str(frozen_era_path): sha256_file(frozen_era_path),
        str(frozen_meta_path): sha256_file(frozen_meta_path),
    }
    decisions = pd.read_parquet(v3_dir / "daily_decisions.parquet")
    frozen = pd.read_parquet(frozen_era_path)
    plan, plan_audit = build_request_plan(decisions, frozen, years=TEST_YEARS)
    _write_or_verify_plan(run_dir, plan, plan_audit)
    print(
        f"План: {len(plan)} запросов, {plan_audit['unique_missing_weather_cell_dates']} новых суток.",
        flush=True,
    )
    download_audit = download_plan(
        plan,
        run_dir,
        retries=retries,
        timeout_seconds=timeout_seconds,
        spacing_seconds=spacing_seconds,
    )
    extended_path, assembly_audit = assemble_weather_bundle(
        plan, run_dir, frozen_era_path
    )
    replay = replay_hutton(extended_path, v3_dir, run_dir)
    write_report(run_dir, replay, plan_audit)
    source_snapshot_path = _write_source_snapshot(run_dir)
    for source, expected in input_hashes.items():
        if sha256_file(source) != expected:
            raise AssertionError(f"Frozen input changed during run: {source}")
    outputs = [
        "request_plan.csv",
        "request_plan_manifest.json",
        "download_checkpoint.json",
        "era5_backfill_daily.parquet",
        "era5_potato_daily_extended.parquet",
        "overlap_validation.json",
        "source_manifest.json",
        "predictions.parquet",
        "alarm_states.parquet",
        "event_hits.parquet",
        "metrics.csv",
        "comparisons.csv",
        "computability_by_year.csv",
        "paired_year_bootstrap.csv",
        "report_ru.md",
        "REPRODUCE.md",
        source_snapshot_path.name,
    ]
    manifest = {
        "format": RUN_FORMAT,
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "years": list(TEST_YEARS),
        "policy": {
            "threshold": POLICY.threshold,
            "active_days": POLICY.active_days,
            "cooldown_days": POLICY.cooldown_days,
            "window_days": [3, 10],
        },
        "parent_v3_verification": parent_verification,
        "input_hashes": input_hashes,
        "plan_audit": plan_audit,
        "download_audit": download_audit,
        "assembly_audit": assembly_audit,
        "replay_checks": {
            "identity_replay": replay["identity_replay"],
            "preexisting_scores_unchanged": replay["preexisting_scores_unchanged"],
        },
        "output_hashes": {relative: sha256_file(run_dir / relative) for relative in outputs},
        "privacy": "local_only; do_not_publish_raw_plan_weather_or_field-linked_outputs",
    }
    _atomic_json(run_dir / "execution_manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("all", help="plan, resume download, assemble, and replay")
    run.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    run.add_argument("--v3-dir", type=Path, default=DEFAULT_V3)
    run.add_argument("--frozen-era", type=Path, default=DEFAULT_FROZEN_ERA)
    run.add_argument("--frozen-meta", type=Path, default=DEFAULT_FROZEN_META)
    run.add_argument("--retries", type=int, default=6)
    run.add_argument("--timeout-seconds", type=float, default=90.0)
    run.add_argument("--spacing-seconds", type=float, default=0.4)
    check = subparsers.add_parser("check", help="verify an already completed run without network")
    check.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "check":
            manifest = _verify_completed_run(args.run_dir.resolve())
            print(
                json.dumps(
                    {
                        "status": manifest["status"],
                        "run_dir": str(args.run_dir.resolve()),
                        "outputs_checked": len(manifest["output_hashes"]),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        manifest = run_all(
            run_dir=args.run_dir,
            v3_dir=args.v3_dir,
            frozen_era_path=args.frozen_era,
            frozen_meta_path=args.frozen_meta,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
            spacing_seconds=args.spacing_seconds,
        )
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "run_dir": str(args.run_dir.resolve()),
                    "requests": manifest.get("plan_audit", {}).get("request_ranges"),
                    "new_weather_days": manifest.get("assembly_audit", {}).get("backfill_rows"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except KeyboardInterrupt as exc:
        print(f"Остановлено безопасно: {exc}", file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
