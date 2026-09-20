"""Resumable builder for the retrospective GDEX GFS feature table.

The builder intentionally has three separate stages:

``plan``
    Read the private first-cycle decision calendar and freeze one archive plan
    per local issue date and publication-lag scenario.  The plan contains only
    public GFS grid identifiers; field identifiers and outcomes never enter it.
``fetch``
    Download only bounded NCSS subsets, retain their byte-level provenance in
    the :mod:`agro_phenology.gfs_archive` cache, and write one resumable feature
    checkpoint per issue date/scenario.
``assemble``
    Concatenate the checkpoints into the table consumed by the A/B/C model
    experiment and write an aggregate source manifest.

Historical GDEX metadata does not prove first-publication timestamps.  The
4-hour and 7-hour availability scenarios are therefore kept as explicit
assumptions, with ``publication_time_utc`` always null.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .gfs_archive import (
    DATASET_DOI,
    DATASET_ID,
    FORECAST_FEATURE_COLUMNS,
    GFSArchiveClient,
    GFSArchiveError,
    GFSBoundingBox,
    GFSCacheIntegrityError,
    GFSMetadataError,
    GFSSubsetArtifact,
    GFSSubsetRequest,
    GFSVariableMapping,
    RequestsGFSTransport,
    TEMPERATURE_VARIABLE,
    RELATIVE_HUMIDITY_VARIABLE,
    aggregate_forecast_features,
    build_experiment_requests,
    canonical_gfs_cell_id,
    load_subset_frame,
    select_assumed_available_cycle,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    REPO_ROOT
    / "docs/research/late_blight_early_warning/gfs_forecast_evaluation_contract.json"
)
DEFAULT_PARENT_DECISIONS = (
    REPO_ROOT
    / "results/late_blight_early_warning/20260910_first_cycle_v3/daily_decisions.parquet"
)
DEFAULT_BBOX = GFSBoundingBox(west=21.0, east=28.0, south=55.75, north=58.0)
PLAN_FILE_NAME = "gfs_request_plan.parquet"
PLAN_MANIFEST_FILE_NAME = "gfs_request_plan_manifest.json"
FEATURE_FILE_NAME = "gfs_forecast_features.parquet"
SOURCE_MANIFEST_FILE_NAME = "gfs_source_manifest.json"
PLAN_SCHEMA_VERSION = "gfs_feature_request_plan_v2"
CHECKPOINT_SCHEMA_VERSION = "gfs_feature_checkpoint_v1"
SOURCE_MANIFEST_SCHEMA_VERSION = "gfs_feature_source_manifest_v1"
DYNAMIC_BBOX_MODE = "dynamic_required_cells"
FIXED_BBOX_MODE = "fixed"
GFS_GRID_STEP_DEGREES = 0.25
DYNAMIC_BBOX_PADDING_DEGREES = 0.01
AUDITED_DDS_ACCESS_MODE = "audited_file_specific_dds_v1"
FAST_NCSS_ACCESS_MODE = "fast_ncss_candidate_validation_v1"
FAST_NCSS_PROVENANCE_SCHEMA = "gfs_gdex_fast_ncss_candidate_provenance_v1"
FAST_PRECIPITATION_CANDIDATES = (
    "Total_precipitation_surface_6_Hour_Accumulation",
    "Total_precipitation_surface_Mixed_intervals_Accumulation",
)
# GFS v15.1 changed the GRIB/NetCDF interval representation at the 12 UTC
# cycle on 2019-06-12.  Older d084001 files normally expose the dedicated
# six-hour name at every six-hour lead; newer files normally expose Mixed
# intervals after f006.  This only orders the two audited candidates.  Every
# returned file still has to pass the parser and exact interval-bound checks.
GFS_MIXED_INTERVALS_START_UTC = datetime(2019, 6, 12, 12, tzinfo=timezone.utc)

_WEATHER_CELL_RE = re.compile(
    r"^\s*(?P<latitude>[+-]?\d+(?:\.\d+)?)_(?P<longitude>[+-]?\d+(?:\.\d+)?)\s*$"
)
_GFS_CELL_ID_RE = re.compile(
    r"^gfs025_lat(?P<latitude>[+-]\d+(?:\.\d+)?)_lon(?P<longitude>[+-]\d+(?:\.\d+)?)$"
)
_PRIVATE_OR_TARGET_FRAGMENTS = (
    "field",
    "target",
    "event",
    "label",
    "visit",
    "outcome",
    "final_latitude",
    "final_longitude",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"unsupported JSON type: {type(value).__name__}")


def _write_json_atomic(path: Path, payload: Any) -> None:
    content = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    temp = Path(raw_temp)
    try:
        frame.to_parquet(temp, index=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _read_contract(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    contract = json.loads(source.read_text(encoding="utf-8"))
    sampling = contract.get("forecast_sampling", {})
    if int(sampling.get("selected_step_hours", -1)) != 6:
        raise ValueError("frozen contract must select six-hour forecast steps")
    bands = sampling.get("bands")
    expected_bands = [
        {
            "id": "d1_3",
            "lead_hours_after_decision_open": 0,
            "lead_hours_after_decision_closed": 72,
        },
        {
            "id": "d4_7",
            "lead_hours_after_decision_open": 72,
            "lead_hours_after_decision_closed": 168,
        },
    ]
    if bands != expected_bands:
        raise ValueError("frozen contract forecast bands differ from (0,72], (72,168]")
    features = tuple(contract.get("forecast_features", ()))
    if features != FORECAST_FEATURE_COLUMNS:
        raise ValueError("frozen contract feature list disagrees with gfs_archive")
    availability = contract.get("forecast_availability", {})
    main = float(availability.get("main_assumed_hours_after_initialization", -1))
    sensitivity = float(
        availability.get("delay_sensitivity_total_hours_after_initialization", -1)
    )
    if (main, sensitivity) != (4.0, 7.0):
        raise ValueError("frozen experiment requires total publication lags 4h and 7h")
    source_dataset = contract.get("source_dataset", {})
    if source_dataset.get("id") != DATASET_ID:
        raise ValueError("contract source dataset is not GDEX d084001")
    return contract


def _is_unknown_variable_http_400(error: Exception) -> bool:
    """Recognize only a provider response that explicitly rejects a variable.

    The normal transport includes the HTTP status, final URL and response-body
    preview in ``GFSArchiveError``.  Merely seeing status 400 is insufficient:
    invalid time/bbox requests must not be hidden by trying another variable.
    """

    message = str(error).lower()
    if re.match(r"^http 400\b", message) is None:
        return False
    return any(
        marker in message
        for marker in (
            "unknown variable",
            "variable not found",
            "no such variable",
            "invalid variable",
            "not contained in the requested dataset",
        )
    )


def _precipitation_candidate_order(
    request: GFSSubsetRequest,
) -> tuple[str, str]:
    """Prefer the archive schema most likely for the requested lead.

    Files before the operational 2019-06-12 12 UTC transition normally expose
    the dedicated six-hour accumulation at every selected lead.  In the newer
    archive f006 retains that name while later leads normally expose Mixed
    intervals.  This only changes query order: every individual response is
    still validated and the alternative remains restricted to explicit
    unknown-variable fallback.
    """

    six_hour, mixed = FAST_PRECIPITATION_CANDIDATES
    if request.init_time_utc < GFS_MIXED_INTERVALS_START_UTC:
        return (six_hour, mixed)
    return (six_hour, mixed) if request.lead_hours == 6 else (mixed, six_hour)


class FastNCSSCandidateClient:
    """NCSS-only extraction with explicit candidate validation provenance.

    This is an opt-in performance path.  It does not reuse a schema from a
    different archive file.  Each file is queried with at most the two frozen
    precipitation names.  The second is attempted only after an explicit HTTP
    400 unknown-variable response.  A successful response is parsed through
    ``load_subset_frame`` before it becomes a valid cache entry, confirming the
    returned variable, its 2 m levels and its six-hour precipitation bounds.
    """

    source_access_mode = FAST_NCSS_ACCESS_MODE

    def __init__(
        self,
        cache_root: str | Path,
        *,
        transport: Any | None = None,
        max_bbox_area_degrees_squared: float = 100.0,
        max_subset_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if max_bbox_area_degrees_squared <= 0:
            raise ValueError("max_bbox_area_degrees_squared must be positive")
        self.cache_root = Path(cache_root)
        self.transport = transport or RequestsGFSTransport()
        self.max_bbox_area_degrees_squared = max_bbox_area_degrees_squared
        self.max_subset_bytes = max_subset_bytes

    def _cache_dir(self, request: GFSSubsetRequest) -> Path:
        candidate_order = _precipitation_candidate_order(request)
        key_payload = {
            "request": request.canonical_payload(),
            "source_access_mode": FAST_NCSS_ACCESS_MODE,
            "precipitation_candidates": list(candidate_order),
        }
        key = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        return (
            self.cache_root
            / f"{DATASET_ID}_fast_ncss_candidates_v1"
            / f"{request.init_time_utc:%Y}"
            / f"{request.init_time_utc:%Y%m%d%H}"
            / f"f{request.lead_hours:03d}"
            / key
        )

    @staticmethod
    def _mapping(
        precipitation: str, candidate_order: Sequence[str]
    ) -> GFSVariableMapping:
        return GFSVariableMapping(
            temperature=TEMPERATURE_VARIABLE,
            relative_humidity=RELATIVE_HUMIDITY_VARIABLE,
            precipitation=precipitation,
            all_precipitation_candidates=tuple(candidate_order),
            # No DDS was retrieved. The empty marker is confined to this
            # in-memory compatibility object; provenance stores null explicitly.
            dds_sha256="",
        )

    def _load_cached(self, request: GFSSubsetRequest) -> GFSSubsetArtifact | None:
        directory = self._cache_dir(request)
        data_path = directory / "subset.nc"
        provenance_path = directory / "provenance.json"
        if not data_path.exists() and not provenance_path.exists():
            return None
        if not data_path.exists() or not provenance_path.exists():
            raise GFSCacheIntegrityError(f"incomplete fast-NCSS cache entry: {directory}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("schema_version") != FAST_NCSS_PROVENANCE_SCHEMA:
            raise GFSCacheIntegrityError(
                f"unexpected fast-NCSS provenance schema: {provenance_path}"
            )
        if provenance.get("source_access_mode") != FAST_NCSS_ACCESS_MODE:
            raise GFSCacheIntegrityError(f"fast-NCSS access mode mismatch: {provenance_path}")
        if provenance.get("dds_requested") is not False or provenance.get("dds_sha256") is not None:
            raise GFSCacheIntegrityError(
                f"fast-NCSS provenance unexpectedly claims DDS access: {provenance_path}"
            )
        if provenance.get("schema_confirmation") != (
            "successful_NCSS_NetCDF3_plus_parser_variable_2m_and_6h_bounds"
        ):
            raise GFSCacheIntegrityError(
                f"fast-NCSS parser validation marker is missing: {provenance_path}"
            )
        if provenance.get("request") != request.canonical_payload():
            raise GFSCacheIntegrityError(f"fast-NCSS request mismatch: {provenance_path}")
        actual_sha = sha256_file(data_path)
        if actual_sha != provenance.get("data_sha256"):
            raise GFSCacheIntegrityError(f"fast-NCSS checksum mismatch: {data_path}")
        precipitation = str(provenance.get("selected_precipitation_variable", ""))
        if precipitation not in FAST_PRECIPITATION_CANDIDATES:
            raise GFSCacheIntegrityError(
                f"unexpected cached precipitation variable: {precipitation!r}"
            )
        if provenance.get("publication_time_utc") is not None:
            raise GFSCacheIntegrityError("fast-NCSS cache claims a publication timestamp")
        candidate_order = _precipitation_candidate_order(request)
        if provenance.get("candidate_order") != list(candidate_order):
            raise GFSCacheIntegrityError(
                f"fast-NCSS candidate order mismatch: {provenance_path}"
            )
        retrieved = pd.Timestamp(provenance["retrieved_at_utc"])
        if retrieved.tzinfo is None:
            raise GFSCacheIntegrityError("retrieved_at_utc is timezone-naive")
        return GFSSubsetArtifact(
            data_path=data_path,
            provenance_path=provenance_path,
            data_sha256=actual_sha,
            data_bytes=data_path.stat().st_size,
            dds_sha256="",
            request=request,
            variable_mapping=self._mapping(precipitation, candidate_order),
            retrieved_at_utc=retrieved.to_pydatetime(),
            cached=True,
        )

    def fetch_subset(
        self,
        request: GFSSubsetRequest,
        *,
        retrieved_at_utc: datetime | None = None,
    ) -> GFSSubsetArtifact:
        cached = self._load_cached(request)
        if cached is not None:
            return cached
        if request.bbox.area_degrees_squared > self.max_bbox_area_degrees_squared:
            raise ValueError(
                "bbox is too large for guarded fast NCSS access: "
                f"{request.bbox.area_degrees_squared:.3f} deg2"
            )

        attempts: list[dict[str, Any]] = []
        selected_precipitation: str | None = None
        subset_payload: Any | None = None
        candidate_order = _precipitation_candidate_order(request)
        for candidate_index, candidate in enumerate(candidate_order):
            params = [
                ("var", TEMPERATURE_VARIABLE),
                ("var", RELATIVE_HUMIDITY_VARIABLE),
                ("var", candidate),
                *request.bbox.as_query(),
                ("time", pd.Timestamp(request.valid_time_utc).isoformat().replace("+00:00", "Z")),
                ("horizStride", "1"),
                ("accept", "netcdf3"),
            ]
            try:
                payload = self.transport.get(
                    request.ncss_url,
                    params=params,
                    max_bytes=self.max_subset_bytes,
                )
            except GFSArchiveError as error:
                if _is_unknown_variable_http_400(error):
                    attempts.append(
                        {
                            "precipitation_variable": candidate,
                            "result": "http_400_unknown_variable",
                            "error_sha256": hashlib.sha256(
                                str(error).encode("utf-8")
                            ).hexdigest(),
                        }
                    )
                    if candidate_index + 1 < len(candidate_order):
                        continue
                    raise GFSMetadataError(
                        "neither frozen precipitation candidate exists in archive file"
                    ) from error
                # Time, bbox, authentication, network and server failures retain
                # their original exception and never trigger schema fallback.
                raise
            if payload.status_code != 200:
                raise GFSArchiveError(
                    f"unexpected NCSS status {payload.status_code} for {payload.url}"
                )
            if not payload.content.startswith(b"CDF"):
                preview = payload.content[:200].decode("utf-8", errors="replace")
                raise GFSArchiveError(f"NCSS response is not NetCDF3: {preview}")
            selected_precipitation = candidate
            subset_payload = payload
            attempts.append(
                {
                    "precipitation_variable": candidate,
                    "result": "ncss_netcdf_received_pending_parser_validation",
                }
            )
            break
        if selected_precipitation is None or subset_payload is None:
            raise GFSMetadataError("no known precipitation candidate was selected")

        retrieved = pd.Timestamp(retrieved_at_utc or datetime.now(timezone.utc))
        if retrieved.tzinfo is None:
            raise ValueError("retrieved_at_utc must be timezone-aware")
        retrieved = retrieved.tz_convert("UTC")
        data_sha = hashlib.sha256(subset_payload.content).hexdigest()
        directory = self._cache_dir(request)
        directory.mkdir(parents=True, exist_ok=True)
        descriptor, raw_temp = tempfile.mkstemp(
            prefix=".candidate.", suffix=".nc", dir=directory
        )
        temp_data = Path(raw_temp)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(subset_payload.content)
                handle.flush()
                os.fsync(handle.fileno())
            validation_artifact = GFSSubsetArtifact(
                data_path=temp_data,
                provenance_path=directory / ".pending-provenance.json",
                data_sha256=data_sha,
                data_bytes=len(subset_payload.content),
                dds_sha256="",
                request=request,
                variable_mapping=self._mapping(selected_precipitation, candidate_order),
                retrieved_at_utc=retrieved.to_pydatetime(),
                cached=False,
            )
            # This validates returned names, 2 m axes and the exact six-hour
            # precipitation interval before cache provenance can claim success.
            load_subset_frame(validation_artifact)
        finally:
            temp_data.unlink(missing_ok=True)

        data_path = directory / "subset.nc"
        provenance_path = directory / "provenance.json"
        _write_bytes_atomic(data_path, subset_payload.content)
        attempts[-1]["result"] = "ncss_netcdf_parser_and_bounds_validated"
        provenance = {
            "schema_version": FAST_NCSS_PROVENANCE_SCHEMA,
            "source_access_mode": FAST_NCSS_ACCESS_MODE,
            "dataset_id": DATASET_ID,
            "dataset_doi": DATASET_DOI,
            "request": request.canonical_payload(),
            "archive_path": request.archive_path,
            "dds_requested": False,
            "dds_url": None,
            "dds_sha256": None,
            "publication_time_utc": None,
            "retrieved_at_utc": retrieved.isoformat().replace("+00:00", "Z"),
            "candidate_order": list(candidate_order),
            "candidate_attempts": attempts,
            "selected_precipitation_variable": selected_precipitation,
            "schema_confirmation": (
                "successful_NCSS_NetCDF3_plus_parser_variable_2m_and_6h_bounds"
            ),
            "ncss_url": subset_payload.url,
            "data_sha256": data_sha,
            "data_bytes": len(subset_payload.content),
            "response_headers": {
                key: value
                for key, value in subset_payload.headers.items()
                if key.lower()
                in {"content-type", "content-length", "last-modified", "etag"}
            },
        }
        _write_json_atomic(provenance_path, provenance)
        return GFSSubsetArtifact(
            data_path=data_path,
            provenance_path=provenance_path,
            data_sha256=data_sha,
            data_bytes=len(subset_payload.content),
            dds_sha256="",
            request=request,
            variable_mapping=self._mapping(selected_precipitation, candidate_order),
            retrieved_at_utc=retrieved.to_pydatetime(),
            cached=False,
        )


def weather_cell_to_gfs_cell_id(value: str) -> str:
    """Convert the existing private join key to a public-grid identifier."""

    match = _WEATHER_CELL_RE.fullmatch(str(value))
    if match is None:
        raise ValueError(f"invalid weather_cell value: {value!r}")
    latitude = float(match.group("latitude"))
    longitude = float(match.group("longitude"))
    return canonical_gfs_cell_id(latitude, longitude)


def gfs_cell_id_to_grid_center(value: str) -> tuple[float, float]:
    """Decode a public GFS cell ID and verify the quarter-degree lattice."""

    match = _GFS_CELL_ID_RE.fullmatch(str(value))
    if match is None:
        raise ValueError(f"invalid gfs_cell_id value: {value!r}")
    latitude = float(match.group("latitude"))
    longitude = float(match.group("longitude"))
    for coordinate, name in ((latitude, "latitude"), (longitude, "longitude")):
        if not np.isclose(
            coordinate / GFS_GRID_STEP_DEGREES,
            round(coordinate / GFS_GRID_STEP_DEGREES),
        ):
            raise ValueError(f"{name} is not on the 0.25-degree GFS grid: {coordinate}")
    if canonical_gfs_cell_id(latitude, longitude) != value:
        raise ValueError(f"gfs_cell_id is not canonical: {value!r}")
    return latitude, longitude


def minimal_bbox_for_gfs_cells(
    gfs_cell_ids: Sequence[str],
    *,
    padding_degrees: float = DYNAMIC_BBOX_PADDING_DEGREES,
) -> GFSBoundingBox:
    """Bound the extreme required grid centers without adding outer neighbors.

    Padding is positive so a one-cell request remains a valid non-zero-area
    bbox, and strictly less than half a GFS step so the immediately adjacent
    centers outside each extreme cannot enter through bbox rounding.
    """

    cells = sorted(set(str(value) for value in gfs_cell_ids))
    if not cells:
        raise ValueError("at least one GFS cell is required for a dynamic bbox")
    half_step = GFS_GRID_STEP_DEGREES / 2.0
    if not 0.0 < padding_degrees < half_step:
        raise ValueError(
            f"dynamic bbox padding must lie in (0, {half_step}) degrees"
        )
    centers = [gfs_cell_id_to_grid_center(cell) for cell in cells]
    latitudes = [center[0] for center in centers]
    longitudes = [center[1] for center in centers]
    bbox = GFSBoundingBox(
        west=min(longitudes) - padding_degrees,
        east=max(longitudes) + padding_degrees,
        south=min(latitudes) - padding_degrees,
        north=max(latitudes) + padding_degrees,
    )
    if not all(
        bbox.south <= latitude <= bbox.north
        and bbox.west <= longitude <= bbox.east
        for latitude, longitude in centers
    ):
        raise AssertionError("dynamic bbox excludes a required GFS grid center")
    if not (
        bbox.south > min(latitudes) - GFS_GRID_STEP_DEGREES
        and bbox.north < max(latitudes) + GFS_GRID_STEP_DEGREES
        and bbox.west > min(longitudes) - GFS_GRID_STEP_DEGREES
        and bbox.east < max(longitudes) + GFS_GRID_STEP_DEGREES
    ):
        raise AssertionError("dynamic bbox includes an outer neighboring grid center")
    return bbox


def _assumption_id(selection_rule_id: str, total_lag_hours: int) -> str:
    return f"{selection_rule_id}_total_lag_{total_lag_hours}h_unverified"


def build_request_plan(
    decisions: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    bbox_mode: str = DYNAMIC_BBOX_MODE,
    fixed_bbox: GFSBoundingBox = DEFAULT_BBOX,
    bbox_padding_degrees: float = DYNAMIC_BBOX_PADDING_DEGREES,
    start_year: int = 2015,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Freeze one bounded archive request bundle per issue date/scenario.

    Only service-active rows determine which public GFS cells are retained.
    The output deliberately contains no field-season, field ID, outcome,
    target, or raw coordinate column.
    """

    required = {"season", "issue_date", "issued_at", "service_active", "weather_cell"}
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"daily decisions miss columns: {missing}")
    work = decisions.loc[
        decisions["service_active"].fillna(False).astype(bool)
        & pd.to_numeric(decisions["season"], errors="coerce").between(start_year, end_year),
        list(required),
    ].copy()
    if work.empty:
        raise ValueError("no service-active decision rows in requested year range")
    work["issue_date"] = pd.to_datetime(work["issue_date"], errors="raise").dt.normalize()
    work["issued_at_utc"] = pd.to_datetime(work["issued_at"], errors="raise", utc=True)
    work["gfs_cell_id"] = work["weather_cell"].map(weather_cell_to_gfs_cell_id)
    issue_time_counts = work.groupby("issue_date")["issued_at_utc"].nunique()
    if not issue_time_counts.eq(1).all():
        bad = issue_time_counts[~issue_time_counts.eq(1)].index.astype(str).tolist()[:5]
        raise ValueError(f"issue date has multiple decision timestamps: {bad}")

    if bbox_mode not in {DYNAMIC_BBOX_MODE, FIXED_BBOX_MODE}:
        raise ValueError(f"unsupported bbox mode: {bbox_mode!r}")
    if not 0.0 < bbox_padding_degrees < GFS_GRID_STEP_DEGREES / 2.0:
        raise ValueError("bbox padding must be positive and smaller than half a GFS step")

    availability = contract["forecast_availability"]
    base_lag = int(availability["main_assumed_hours_after_initialization"])
    sensitivity_lag = int(
        availability["delay_sensitivity_total_hours_after_initialization"]
    )
    selection_rule_id = str(availability["selection_rule_id"])
    scenario_hours = (base_lag, sensitivity_lag)
    rows: list[dict[str, Any]] = []
    for issue_date, group in work.groupby("issue_date", sort=True):
        issue_time = group["issued_at_utc"].iloc[0].to_pydatetime()
        cells = sorted(group["gfs_cell_id"].unique().tolist())
        if not cells:
            raise AssertionError("service-active issue date has no GFS cells")
        request_bbox = (
            minimal_bbox_for_gfs_cells(
                cells, padding_degrees=bbox_padding_degrees
            )
            if bbox_mode == DYNAMIC_BBOX_MODE
            else fixed_bbox
        )
        for total_lag in scenario_hours:
            assumption_id = _assumption_id(selection_rule_id, total_lag)
            selection = select_assumed_available_cycle(
                issue_time,
                assumed_publication_lag=timedelta(hours=base_lag),
                additional_delay=timedelta(hours=total_lag - base_lag),
                assumption_id=assumption_id,
            )
            requests = build_experiment_requests(selection, request_bbox)
            offsets = [
                (request.valid_time_utc - selection.issue_time_utc).total_seconds() / 3600.0
                for request in requests
            ]
            d1_3_count = sum(0.0 < offset <= 72.0 for offset in offsets)
            d4_7_count = sum(72.0 < offset <= 168.0 for offset in offsets)
            if (d1_3_count, d4_7_count) != (12, 16):
                raise AssertionError(
                    f"dynamic plan does not yield 12/16 slots: {d1_3_count}/{d4_7_count}"
                )
            checkpoint_id = f"{pd.Timestamp(issue_date):%Y%m%d}_lag{total_lag:02d}h"
            rows.append(
                {
                    "plan_schema_version": PLAN_SCHEMA_VERSION,
                    "checkpoint_id": checkpoint_id,
                    "issue_date": pd.Timestamp(issue_date),
                    "issue_time_utc": pd.Timestamp(selection.issue_time_utc),
                    "issue_year": int(pd.Timestamp(issue_date).year),
                    "availability_scenario_hours": total_lag,
                    "selection_rule_id": assumption_id,
                    "selected_init_utc": pd.Timestamp(selection.init_time_utc),
                    "assumed_available_at_utc": pd.Timestamp(
                        selection.assumed_available_at_utc
                    ),
                    "publication_time_utc": pd.NaT,
                    "bbox_mode": bbox_mode,
                    "bbox_padding_degrees": (
                        bbox_padding_degrees
                        if bbox_mode == DYNAMIC_BBOX_MODE
                        else np.nan
                    ),
                    "bbox_west": request_bbox.west,
                    "bbox_east": request_bbox.east,
                    "bbox_south": request_bbox.south,
                    "bbox_north": request_bbox.north,
                    "archive_leads_hours_json": json.dumps(
                        [request.lead_hours for request in requests], separators=(",", ":")
                    ),
                    "valid_offsets_hours_json": json.dumps(offsets, separators=(",", ":")),
                    "required_gfs_cell_ids_json": json.dumps(cells, separators=(",", ":")),
                    "required_gfs_cell_count": len(cells),
                    "requested_snapshot_count": len(requests),
                    "expected_steps_1_3d": d1_3_count,
                    "expected_steps_4_7d": d4_7_count,
                    "source_dataset_id": DATASET_ID,
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["issue_date", "availability_scenario_hours"], kind="mergesort"
    ).reset_index(drop=True)
    if result["checkpoint_id"].duplicated().any():
        raise AssertionError("duplicate request-plan checkpoint ID")
    assert_no_private_or_target_columns(result)
    return result


def assert_no_private_or_target_columns(frame: pd.DataFrame) -> None:
    unsafe = [
        column
        for column in frame.columns
        if any(fragment in column.lower() for fragment in _PRIVATE_OR_TARGET_FRAGMENTS)
    ]
    if unsafe:
        raise ValueError(f"table contains private/evaluator columns: {sorted(unsafe)}")
    if {"latitude", "longitude"}.intersection(frame.columns):
        raise ValueError("feature table must not contain raw coordinate columns")


def _plan_bbox(row: Mapping[str, Any]) -> GFSBoundingBox:
    return GFSBoundingBox(
        west=float(row["bbox_west"]),
        east=float(row["bbox_east"]),
        south=float(row["bbox_south"]),
        north=float(row["bbox_north"]),
    )


def requests_for_plan_row(row: Mapping[str, Any]) -> list[GFSSubsetRequest]:
    issue_time = pd.Timestamp(row["issue_time_utc"])
    init_time = pd.Timestamp(row["selected_init_utc"])
    available_at = pd.Timestamp(row["assumed_available_at_utc"])
    for name, timestamp in (
        ("issue_time_utc", issue_time),
        ("selected_init_utc", init_time),
        ("assumed_available_at_utc", available_at),
    ):
        if timestamp.tzinfo is None:
            raise ValueError(f"plan {name} must be timezone-aware")
    leads = json.loads(str(row["archive_leads_hours_json"]))
    return [
        GFSSubsetRequest(
            issue_time_utc=issue_time.to_pydatetime(),
            init_time_utc=init_time.to_pydatetime(),
            lead_hours=int(lead),
            bbox=_plan_bbox(row),
            availability_assumption_id=str(row["selection_rule_id"]),
            assumed_available_at_utc=available_at.to_pydatetime(),
        )
        for lead in leads
    ]


def _missing_feature_row(
    row: Mapping[str, Any],
    gfs_cell_id: str,
    *,
    observed_steps_1_3d: int = 0,
    observed_steps_4_7d: int = 0,
    source_hashes: Sequence[str] = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "issue_date": pd.Timestamp(row["issue_date"]).normalize(),
        "issue_time_utc": pd.Timestamp(row["issue_time_utc"]),
        "gfs_cell_id": gfs_cell_id,
        "availability_scenario_hours": int(row["availability_scenario_hours"]),
        "selected_init_utc": pd.Timestamp(row["selected_init_utc"]),
        "assumed_available_at_utc": pd.Timestamp(row["assumed_available_at_utc"]),
        "publication_time_utc": pd.NaT,
        "selection_rule_id": str(row["selection_rule_id"]),
        "source_dataset_id": DATASET_ID,
        "source_hashes_json": json.dumps(sorted(set(source_hashes)), separators=(",", ":")),
        "first_lead_h": np.nan,
        "last_lead_h": np.nan,
        "native_step_hours": 6,
        "expected_steps_1_3d": 12,
        "observed_steps_1_3d": int(observed_steps_1_3d),
        "expected_steps_4_7d": 16,
        "observed_steps_4_7d": int(observed_steps_4_7d),
        "complete_1_3d": observed_steps_1_3d == 12,
        "complete_4_7d": observed_steps_4_7d == 16,
        "forecast_available": False,
    }
    payload.update({column: np.nan for column in FORECAST_FEATURE_COLUMNS})
    return payload


def aggregate_requested_cells(
    snapshots: pd.DataFrame,
    plan_row: Mapping[str, Any],
    *,
    failed_requests: Sequence[Mapping[str, Any]] = (),
    source_access_mode: str = AUDITED_DDS_ACCESS_MODE,
) -> pd.DataFrame:
    """Aggregate snapshots and retain only cells required on this issue date."""

    required_cells = json.loads(str(plan_row["required_gfs_cell_ids_json"]))
    if not isinstance(required_cells, list) or not required_cells:
        raise ValueError("plan row has no required GFS cells")
    if snapshots.empty:
        aggregated = pd.DataFrame()
    else:
        work = snapshots.copy()
        work["gfs_cell_id"] = [
            canonical_gfs_cell_id(latitude, longitude)
            for latitude, longitude in zip(
                work["latitude"], work["longitude"], strict=True
            )
        ]
        work = work[work["gfs_cell_id"].isin(required_cells)].drop(
            columns="gfs_cell_id"
        )
        aggregated = aggregate_forecast_features(work) if not work.empty else pd.DataFrame()

    if not aggregated.empty:
        aggregated = aggregated.drop(columns=["latitude", "longitude"], errors="ignore")
    existing = set() if aggregated.empty else set(aggregated["gfs_cell_id"])
    missing_cells = [cell for cell in required_cells if cell not in existing]
    if missing_cells:
        missing = pd.DataFrame(
            [_missing_feature_row(plan_row, cell) for cell in missing_cells]
        )
        aggregated = pd.concat([aggregated, missing], ignore_index=True, sort=False)

    if aggregated.empty:
        raise AssertionError("checkpoint aggregation produced no rows")
    failed_json = json.dumps(list(failed_requests), sort_keys=True, separators=(",", ":"))
    retrieved_count = int(plan_row["requested_snapshot_count"]) - len(failed_requests)
    aggregated["checkpoint_schema_version"] = CHECKPOINT_SCHEMA_VERSION
    aggregated["checkpoint_id"] = str(plan_row["checkpoint_id"])
    aggregated["source_access_mode"] = str(source_access_mode)
    aggregated["requested_snapshot_count"] = int(plan_row["requested_snapshot_count"])
    aggregated["retrieved_snapshot_count"] = retrieved_count
    aggregated["failed_snapshot_count"] = len(failed_requests)
    aggregated["failed_requests_json"] = failed_json
    aggregated["checkpoint_complete"] = (
        aggregated["forecast_available"].fillna(False).astype(bool)
        & (len(failed_requests) == 0)
    )
    aggregated["checkpoint_created_at_utc"] = pd.Timestamp.now(tz="UTC")
    aggregated = aggregated.sort_values("gfs_cell_id", kind="mergesort").reset_index(
        drop=True
    )
    assert_no_private_or_target_columns(aggregated)
    if set(aggregated["gfs_cell_id"]) != set(required_cells):
        raise AssertionError("checkpoint cells differ from the frozen plan")
    return aggregated


def _checkpoint_path(run_root: Path, row: Mapping[str, Any]) -> Path:
    return (
        run_root
        / "checkpoints"
        / f"year={int(row['issue_year'])}"
        / f"scenario_hours={int(row['availability_scenario_hours']):02d}"
        / f"{row['checkpoint_id']}.parquet"
    )


def _safe_fetch(
    client: Any, request: GFSSubsetRequest
) -> tuple[GFSSubsetRequest, GFSSubsetArtifact | None, dict[str, Any] | None]:
    try:
        return request, client.fetch_subset(request), None
    except Exception as error:  # exact type/message are retained for audit and retry
        return (
            request,
            None,
            {
                "lead_hours": request.lead_hours,
                "archive_path": request.archive_path,
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
            },
        )


def fetch_plan_checkpoint(
    plan_row: Mapping[str, Any],
    client: Any,
    *,
    max_workers: int = 2,
) -> pd.DataFrame:
    """Fetch and aggregate one independently resumable date/scenario bundle."""

    requests = requests_for_plan_row(plan_row)
    artifacts: list[GFSSubsetArtifact] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_safe_fetch, client, request) for request in requests]
        for future in as_completed(futures):
            _, artifact, failure = future.result()
            if artifact is not None:
                artifacts.append(artifact)
            if failure is not None:
                failures.append(failure)
    artifacts.sort(key=lambda artifact: artifact.request.lead_hours)
    failures.sort(key=lambda failure: int(failure["lead_hours"]))
    frames: list[pd.DataFrame] = []
    for artifact in artifacts:
        try:
            frames.append(load_subset_frame(artifact))
        except Exception as error:
            failures.append(
                {
                    "lead_hours": artifact.request.lead_hours,
                    "archive_path": artifact.request.archive_path,
                    "source_sha256": artifact.data_sha256,
                    "error_stage": "parse_and_interval_validation",
                    "error_type": type(error).__name__,
                    "error": str(error)[:1000],
                }
            )
    failures.sort(key=lambda failure: int(failure["lead_hours"]))
    snapshots = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    source_access_mode = getattr(client, "source_access_mode", AUDITED_DDS_ACCESS_MODE)
    return aggregate_requested_cells(
        snapshots,
        plan_row,
        failed_requests=failures,
        source_access_mode=source_access_mode,
    )


def write_request_plan(
    *,
    run_root: str | Path,
    parent_decisions_path: str | Path = DEFAULT_PARENT_DECISIONS,
    contract_path: str | Path = DEFAULT_CONTRACT,
    bbox_mode: str = DYNAMIC_BBOX_MODE,
    fixed_bbox: GFSBoundingBox = DEFAULT_BBOX,
    bbox_padding_degrees: float = DYNAMIC_BBOX_PADDING_DEGREES,
) -> dict[str, Any]:
    run = Path(run_root)
    parent_path = Path(parent_decisions_path)
    contract_source = Path(contract_path)
    contract = _read_contract(contract_source)
    decisions = pd.read_parquet(
        parent_path,
        columns=["season", "issue_date", "issued_at", "service_active", "weather_cell"],
    )
    plan = build_request_plan(
        decisions,
        contract,
        bbox_mode=bbox_mode,
        fixed_bbox=fixed_bbox,
        bbox_padding_degrees=bbox_padding_degrees,
    )
    plan_path = run / PLAN_FILE_NAME
    if plan_path.exists():
        existing = pd.read_parquet(plan_path)
        if not existing.equals(plan):
            raise FileExistsError(
                f"refusing to replace a different frozen request plan: {plan_path}"
            )
    else:
        _write_parquet_atomic(plan_path, plan)
    manifest = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(contract_source.resolve()),
        "contract_sha256": sha256_file(contract_source),
        "parent_daily_decisions_path": str(parent_path.resolve()),
        "parent_daily_decisions_sha256": sha256_file(parent_path),
        "request_plan_path": str(plan_path.resolve()),
        "request_plan_sha256": sha256_file(plan_path),
        "bbox_mode": bbox_mode,
        "bbox_formula": (
            "west=min(required_grid_longitude)-padding; "
            "east=max(required_grid_longitude)+padding; "
            "south=min(required_grid_latitude)-padding; "
            "north=max(required_grid_latitude)+padding"
            if bbox_mode == DYNAMIC_BBOX_MODE
            else "single_fixed_bbox_for_every_issue_date"
        ),
        "gfs_grid_step_degrees": GFS_GRID_STEP_DEGREES,
        "bbox_padding_degrees": (
            bbox_padding_degrees if bbox_mode == DYNAMIC_BBOX_MODE else None
        ),
        "bbox_padding_constraint": "0 < padding < gfs_grid_step_degrees / 2",
        "fixed_bbox": asdict(fixed_bbox) if bbox_mode == FIXED_BBOX_MODE else None,
        "planned_bbox_area_degrees_squared": {
            "minimum": float(
                (
                    (plan["bbox_east"] - plan["bbox_west"])
                    * (plan["bbox_north"] - plan["bbox_south"])
                ).min()
            ),
            "maximum": float(
                (
                    (plan["bbox_east"] - plan["bbox_west"])
                    * (plan["bbox_north"] - plan["bbox_south"])
                ).max()
            ),
        },
        "issue_dates": int(plan["issue_date"].nunique()),
        "scenario_hours": sorted(plan["availability_scenario_hours"].unique().tolist()),
        "checkpoints": int(len(plan)),
        "archive_subset_requests": int(plan["requested_snapshot_count"].sum()),
        "initial_http_request_upper_bound": int(
            2 * plan["requested_snapshot_count"].sum()
        ),
        "dds_strategy": (
            "file_specific_DDS_verified_once_and_cached_for_retry; "
            "no_cross_file_schema_substitution"
        ),
        "first_issue_date": plan["issue_date"].min().date().isoformat(),
        "last_issue_date": plan["issue_date"].max().date().isoformat(),
        "publication_time_status": "unknown_historical_assumed_only",
        "contains_field_ids": False,
        "contains_outcomes": False,
        "contains_coarse_grid_coordinates": True,
        "distribution": "local_private_do_not_publish",
    }
    manifest_path = run / PLAN_MANIFEST_FILE_NAME
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in manifest.items() if key != "created_at_utc"}
        old_comparable = {key: value for key, value in old.items() if key != "created_at_utc"}
        if old_comparable != comparable:
            raise FileExistsError(
                f"refusing to replace a different request-plan manifest: {manifest_path}"
            )
        manifest = old
    else:
        _write_json_atomic(manifest_path, manifest)
    return manifest


def _filter_plan(
    plan: pd.DataFrame,
    *,
    years: Sequence[int] = (),
    dates: Sequence[str] = (),
    scenario_hours: Sequence[int] = (),
) -> pd.DataFrame:
    selected = plan.copy()
    if years:
        selected = selected[selected["issue_year"].isin([int(value) for value in years])]
    if dates:
        normalized = pd.to_datetime(list(dates), errors="raise").normalize()
        selected = selected[selected["issue_date"].isin(normalized)]
    if scenario_hours:
        selected = selected[
            selected["availability_scenario_hours"].isin(
                [int(value) for value in scenario_hours]
            )
        ]
    return selected.sort_values(
        ["issue_date", "availability_scenario_hours"], kind="mergesort"
    )


def fetch_request_plan(
    *,
    run_root: str | Path,
    years: Sequence[int] = (),
    dates: Sequence[str] = (),
    scenario_hours: Sequence[int] = (),
    max_checkpoints: int | None = None,
    max_workers: int = 2,
    retry_incomplete: bool = True,
    fast_ncss_candidates: bool = False,
) -> dict[str, Any]:
    run = Path(run_root)
    plan_path = run / PLAN_FILE_NAME
    if not plan_path.is_file():
        raise FileNotFoundError(f"run plan first: {plan_path}")
    plan = pd.read_parquet(plan_path)
    selected = _filter_plan(
        plan, years=years, dates=dates, scenario_hours=scenario_hours
    )
    if max_checkpoints is not None:
        if max_checkpoints < 1:
            raise ValueError("max_checkpoints must be positive")
        selected = selected.head(max_checkpoints)
    if fast_ncss_candidates:
        client: Any = FastNCSSCandidateClient(run / "gfs_raw_cache")
        requested_access_mode = FAST_NCSS_ACCESS_MODE
    else:
        client = GFSArchiveClient(run / "gfs_raw_cache", max_workers=max_workers)
        requested_access_mode = AUDITED_DDS_ACCESS_MODE
    counts = {
        "selected": int(len(selected)),
        "selected_snapshot_files": int(selected["requested_snapshot_count"].sum()),
        "written": 0,
        "skipped_complete": 0,
        "skipped_incomplete": 0,
        "incomplete_written": 0,
        "rows_written": 0,
    }
    started = datetime.now(timezone.utc)
    for row in selected.to_dict(orient="records"):
        checkpoint_path = _checkpoint_path(run, row)
        if checkpoint_path.exists():
            existing = pd.read_parquet(checkpoint_path)
            existing_modes = set(
                existing.get(
                    "source_access_mode",
                    pd.Series([AUDITED_DDS_ACCESS_MODE] * len(existing)),
                ).astype(str)
            )
            if existing_modes != {requested_access_mode}:
                raise GFSCacheIntegrityError(
                    "checkpoint source access mode differs from requested mode: "
                    f"{checkpoint_path}: {sorted(existing_modes)} vs {requested_access_mode}"
                )
            is_complete = bool(existing["checkpoint_complete"].fillna(False).all())
            if is_complete:
                counts["skipped_complete"] += 1
                continue
            if not retry_incomplete:
                counts["skipped_incomplete"] += 1
                continue
        checkpoint = fetch_plan_checkpoint(row, client, max_workers=max_workers)
        _write_parquet_atomic(checkpoint_path, checkpoint)
        counts["written"] += 1
        counts["rows_written"] += int(len(checkpoint))
        if not bool(checkpoint["checkpoint_complete"].all()):
            counts["incomplete_written"] += 1
    finished = datetime.now(timezone.utc)
    uncached_file_http_lower = 1 if fast_ncss_candidates else 2
    uncached_file_http_upper = 2
    return {
        **counts,
        "uncached_http_requests_per_snapshot_file": {
            "lower": uncached_file_http_lower,
            "upper": uncached_file_http_upper,
        },
        "uncached_http_request_bounds_for_selection": {
            "lower": uncached_file_http_lower * counts["selected_snapshot_files"],
            "upper": uncached_file_http_upper * counts["selected_snapshot_files"],
        },
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "filters": {
            "years": list(years),
            "dates": list(dates),
            "scenario_hours": list(scenario_hours),
            "max_checkpoints": max_checkpoints,
            "retry_incomplete": retry_incomplete,
            "source_access_mode": requested_access_mode,
        },
    }


def assemble_feature_table(
    *,
    run_root: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    run = Path(run_root)
    plan_path = run / PLAN_FILE_NAME
    if not plan_path.is_file():
        raise FileNotFoundError(f"missing request plan: {plan_path}")
    plan = pd.read_parquet(plan_path)
    paths = sorted((run / "checkpoints").glob("year=*/scenario_hours=*/*.parquet"))
    if not paths:
        raise FileNotFoundError("no feature checkpoints have been fetched")
    frames = [pd.read_parquet(path) for path in paths]
    features = pd.concat(frames, ignore_index=True, sort=False)
    if "source_access_mode" not in features:
        raise ValueError("assembled checkpoints miss required source_access_mode provenance")
    source_modes = features["source_access_mode"]
    if source_modes.isna().any() or source_modes.astype(str).str.strip().eq("").any():
        raise ValueError("source_access_mode is absent for some assembled feature rows")
    unknown_source_modes = sorted(
        set(source_modes.astype(str)) - {AUDITED_DDS_ACCESS_MODE, FAST_NCSS_ACCESS_MODE}
    )
    if unknown_source_modes:
        raise ValueError(f"unknown source_access_mode values: {unknown_source_modes}")
    assert_no_private_or_target_columns(features)
    keys = ["issue_date", "gfs_cell_id", "availability_scenario_hours"]
    if features.duplicated(keys).any():
        duplicate = features.loc[features.duplicated(keys, keep=False), keys].head()
        raise ValueError(f"duplicate assembled feature rows:\n{duplicate}")
    if tuple(column for column in FORECAST_FEATURE_COLUMNS if column in features) != tuple(
        FORECAST_FEATURE_COLUMNS
    ):
        raise ValueError("assembled table does not contain all 12 frozen forecast features")
    features = features.sort_values(keys, kind="mergesort").reset_index(drop=True)
    annual_files: list[dict[str, Any]] = []
    annual_root = run / "annual"
    annual_frame = features.assign(
        _issue_year=pd.to_datetime(features["issue_date"]).dt.year
    )
    for (year, scenario), group in annual_frame.groupby(
        ["_issue_year", "availability_scenario_hours"], sort=True
    ):
        annual = group.drop(columns="_issue_year").reset_index(drop=True)
        annual_path = annual_root / f"forecast_features_{int(year)}_{int(scenario)}h.parquet"
        _write_parquet_atomic(annual_path, annual)
        annual_files.append(
            {
                "year": int(year),
                "availability_scenario_hours": int(scenario),
                "path": str(annual_path.resolve()),
                "rows": int(len(annual)),
                "sha256": sha256_file(annual_path),
            }
        )

    destination = Path(output_path) if output_path is not None else run / FEATURE_FILE_NAME
    _write_parquet_atomic(destination, features)

    planned_ids = set(plan["checkpoint_id"].astype(str))
    assembled_ids = set(features["checkpoint_id"].astype(str))
    missing_ids = sorted(planned_ids - assembled_ids)
    extra_ids = sorted(assembled_ids - planned_ids)
    if extra_ids:
        raise ValueError(f"checkpoints not present in frozen plan: {extra_ids[:5]}")
    source_hashes: set[str] = set()
    for value in features["source_hashes_json"].dropna().astype(str):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("source_hashes_json must encode a list")
        source_hashes.update(str(item) for item in parsed)
    complete = features["forecast_available"].fillna(False).astype(bool)
    per_year_scenario = []
    summary = features.assign(issue_year=pd.to_datetime(features["issue_date"]).dt.year)
    for (year, scenario), group in summary.groupby(
        ["issue_year", "availability_scenario_hours"], sort=True
    ):
        group_complete = group["forecast_available"].fillna(False).astype(bool)
        per_year_scenario.append(
            {
                "year": int(year),
                "availability_scenario_hours": int(scenario),
                "feature_rows": int(len(group)),
                "complete_feature_rows": int(group_complete.sum()),
                "complete_fraction": float(group_complete.mean()),
                "issue_dates": int(group["issue_date"].nunique()),
                "checkpoints": int(group["checkpoint_id"].nunique()),
            }
        )
    manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "assembled_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_plan_path": str(plan_path.resolve()),
        "request_plan_sha256": sha256_file(plan_path),
        "feature_table_path": str(destination.resolve()),
        "feature_table_sha256": sha256_file(destination),
        "feature_rows": int(len(features)),
        "complete_feature_rows": int(complete.sum()),
        "complete_feature_fraction": float(complete.mean()),
        "planned_checkpoints": int(len(plan)),
        "assembled_checkpoints": int(len(assembled_ids)),
        "missing_checkpoint_count": int(len(missing_ids)),
        "missing_checkpoint_ids": missing_ids,
        "raw_subset_hash_count": len(source_hashes),
        "raw_subset_sha256": sorted(source_hashes),
        "annual_files": annual_files,
        "source_access_modes": sorted(
            features["source_access_mode"].dropna().astype(str).unique().tolist()
        ),
        "forecast_features": list(FORECAST_FEATURE_COLUMNS),
        "expected_steps": {"d1_3": 12, "d4_7": 16},
        "publication_time_status": "unknown_historical_assumed_only",
        "contains_field_ids": False,
        "contains_outcomes": False,
        "contains_coarse_grid_coordinates": True,
        "distribution": "local_private_do_not_publish",
        "per_year_scenario": per_year_scenario,
    }
    _write_json_atomic(run / SOURCE_MANIFEST_FILE_NAME, manifest)
    return manifest


def _bbox_from_args(args: argparse.Namespace) -> GFSBoundingBox:
    return GFSBoundingBox(
        west=args.west, east=args.east, south=args.south, north=args.north
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a resumable historical GDEX GFS feature table"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="freeze requests from v3 decision dates")
    plan.add_argument("--run-root", type=Path, required=True)
    plan.add_argument("--parent-decisions", type=Path, default=DEFAULT_PARENT_DECISIONS)
    plan.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    plan.add_argument(
        "--bbox-mode",
        choices=(DYNAMIC_BBOX_MODE, FIXED_BBOX_MODE),
        default=DYNAMIC_BBOX_MODE,
    )
    plan.add_argument(
        "--bbox-padding-degrees",
        type=float,
        default=DYNAMIC_BBOX_PADDING_DEGREES,
    )
    plan.add_argument("--west", type=float, default=DEFAULT_BBOX.west)
    plan.add_argument("--east", type=float, default=DEFAULT_BBOX.east)
    plan.add_argument("--south", type=float, default=DEFAULT_BBOX.south)
    plan.add_argument("--north", type=float, default=DEFAULT_BBOX.north)

    fetch = subparsers.add_parser("fetch", help="fetch resumable date/scenario checkpoints")
    fetch.add_argument("--run-root", type=Path, required=True)
    fetch.add_argument("--year", action="append", type=int, default=[])
    fetch.add_argument("--date", action="append", default=[])
    fetch.add_argument("--scenario-hours", action="append", type=int, default=[])
    fetch.add_argument("--max-checkpoints", type=int)
    fetch.add_argument("--max-workers", type=int, default=2, choices=range(1, 9))
    fetch.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="do not retry checkpoints that previously recorded missing files",
    )
    fetch.add_argument(
        "--fast-ncss-candidates",
        action="store_true",
        help=(
            "skip per-file DDS and validate only the two frozen precipitation "
            "candidates from returned NCSS NetCDF"
        ),
    )

    assemble = subparsers.add_parser("assemble", help="assemble fetched checkpoints")
    assemble.add_argument("--run-root", type=Path, required=True)
    assemble.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        result = write_request_plan(
            run_root=args.run_root,
            parent_decisions_path=args.parent_decisions,
            contract_path=args.contract,
            bbox_mode=args.bbox_mode,
            fixed_bbox=_bbox_from_args(args),
            bbox_padding_degrees=args.bbox_padding_degrees,
        )
    elif args.command == "fetch":
        result = fetch_request_plan(
            run_root=args.run_root,
            years=args.year,
            dates=args.date,
            scenario_hours=args.scenario_hours,
            max_checkpoints=args.max_checkpoints,
            max_workers=args.max_workers,
            retry_incomplete=not args.skip_incomplete,
            fast_ncss_candidates=args.fast_ncss_candidates,
        )
    elif args.command == "assemble":
        result = assemble_feature_table(run_root=args.run_root, output_path=args.output)
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "AUDITED_DDS_ACCESS_MODE",
    "DEFAULT_BBOX",
    "DYNAMIC_BBOX_MODE",
    "DYNAMIC_BBOX_PADDING_DEGREES",
    "FEATURE_FILE_NAME",
    "FAST_NCSS_ACCESS_MODE",
    "FAST_NCSS_PROVENANCE_SCHEMA",
    "FAST_PRECIPITATION_CANDIDATES",
    "FIXED_BBOX_MODE",
    "FastNCSSCandidateClient",
    "PLAN_FILE_NAME",
    "PLAN_MANIFEST_FILE_NAME",
    "PLAN_SCHEMA_VERSION",
    "SOURCE_MANIFEST_FILE_NAME",
    "aggregate_requested_cells",
    "assemble_feature_table",
    "assert_no_private_or_target_columns",
    "build_request_plan",
    "fetch_plan_checkpoint",
    "fetch_request_plan",
    "main",
    "gfs_cell_id_to_grid_center",
    "minimal_bbox_for_gfs_cells",
    "requests_for_plan_row",
    "weather_cell_to_gfs_cell_id",
    "write_request_plan",
]
