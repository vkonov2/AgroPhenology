"""Small, auditable client for the NCAR GDEX NCEP GFS archive.

The archive stores one global GRIB2 file per model initialisation and lead.
This module never downloads those global files.  It discovers the variables in
the file's OPeNDAP DDS and asks THREDDS NCSS for a bounded regional NetCDF3
subset.  The resulting files are cached with a checksum and explicit time
provenance so that a retrospective experiment can be resumed safely.

The archive exposes the model reference time, but it does not expose a
documented first-publication timestamp.  ``select_assumed_available_cycle``
therefore requires an explicit, labelled lag assumption instead of silently
treating initialisation as publication.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import math
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DATASET_ID = "d084001"
DATASET_DOI = "10.5065/D65D8PWK"
TDS_ROOT = "https://tds.gdex.ucar.edu/thredds"
GFS_CYCLES_UTC = (0, 6, 12, 18)
SUPPORTED_LEADS_HOURS = tuple(range(3, 241, 3))
# The experiment uses six-hour synoptic samples.  Three-hour leads remain
# readable for source diagnostics, but are not the default modelling plan.
EXPERIMENT_LEADS_HOURS = tuple(range(6, 241, 6))
FORECAST_BANDS_HOURS = ((0.0, 72.0, "d1_3"), (72.0, 168.0, "d4_7"))
FORECAST_FEATURE_COLUMNS = tuple(
    f"fcst_{name}_{band}"
    for band in ("d1_3", "d4_7")
    for name in ("t_mean", "t_min", "t_max", "rh_mean", "rh_max", "precip_sum")
)

TEMPERATURE_VARIABLE = "Temperature_height_above_ground"
RELATIVE_HUMIDITY_VARIABLE = "Relative_humidity_height_above_ground"
PRECIPITATION_NAME_RE = re.compile(
    r"\b(Total_precipitation_surface_[A-Za-z0-9_]+_Accumulation)\b"
)


class GFSArchiveError(RuntimeError):
    """Base class for archive access and validation failures."""


class GFSMetadataError(GFSArchiveError):
    """The selected archive file does not expose the required variables."""


class GFSCacheIntegrityError(GFSArchiveError):
    """A cached file no longer matches its recorded checksum."""


class GFSDownloadTooLargeError(GFSArchiveError):
    """A subset exceeded the configured safety limit."""


def _utc(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return _utc(value, name="datetime").isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _utc(parsed, name="timestamp")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class GFSBoundingBox:
    """A mandatory regional subset in geographic degrees."""

    west: float
    east: float
    south: float
    north: float

    def __post_init__(self) -> None:
        values = (self.west, self.east, self.south, self.north)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("bbox coordinates must be finite")
        if not -180.0 <= self.west <= 360.0 or not -180.0 <= self.east <= 360.0:
            raise ValueError("bbox longitudes must lie in [-180, 360]")
        if not -90.0 <= self.south < self.north <= 90.0:
            raise ValueError("bbox must satisfy -90 <= south < north <= 90")
        if self.west >= self.east:
            raise ValueError("dateline-crossing boxes must be split explicitly")

    @property
    def area_degrees_squared(self) -> float:
        return (self.east - self.west) * (self.north - self.south)

    def as_query(self) -> list[tuple[str, str]]:
        return [
            ("north", format(self.north, ".8g")),
            ("south", format(self.south, ".8g")),
            ("west", format(self.west, ".8g")),
            ("east", format(self.east, ".8g")),
        ]


@dataclass(frozen=True)
class GFSCycleSelection:
    issue_time_utc: datetime
    init_time_utc: datetime
    publication_time_utc: None
    assumed_available_at_utc: datetime
    assumed_publication_lag_hours: float
    additional_delay_hours: float
    assumption_id: str


def select_assumed_available_cycle(
    issue_time: datetime,
    *,
    assumed_publication_lag: timedelta,
    additional_delay: timedelta = timedelta(0),
    assumption_id: str,
) -> GFSCycleSelection:
    """Select the latest GFS cycle under an explicit availability assumption.

    ``publication_time_utc`` deliberately remains null: neither ``reftime`` nor
    THREDDS' archival ``modified`` value establishes first public availability.
    Delay scenarios should call this function with the same base assumption and
    a positive ``additional_delay``.
    """

    issue_utc = _utc(issue_time, name="issue_time")
    if not assumption_id.strip():
        raise ValueError("assumption_id must be non-empty")
    if assumed_publication_lag < timedelta(0) or additional_delay < timedelta(0):
        raise ValueError("availability delays must be non-negative")
    total_lag = assumed_publication_lag + additional_delay
    midnight = issue_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    candidates = [
        midnight + timedelta(days=day_offset, hours=cycle)
        for day_offset in (-3, -2, -1, 0)
        for cycle in GFS_CYCLES_UTC
    ]
    eligible = [candidate for candidate in candidates if candidate + total_lag <= issue_utc]
    if not eligible:
        raise ValueError("no eligible cycle found")
    init_utc = max(eligible)
    return GFSCycleSelection(
        issue_time_utc=issue_utc,
        init_time_utc=init_utc,
        publication_time_utc=None,
        assumed_available_at_utc=init_utc + total_lag,
        assumed_publication_lag_hours=assumed_publication_lag.total_seconds() / 3600.0,
        additional_delay_hours=additional_delay.total_seconds() / 3600.0,
        assumption_id=assumption_id,
    )


@dataclass(frozen=True)
class GFSSubsetRequest:
    issue_time_utc: datetime
    init_time_utc: datetime
    lead_hours: int
    bbox: GFSBoundingBox
    availability_assumption_id: str
    assumed_available_at_utc: datetime
    publication_time_utc: None = None

    def __post_init__(self) -> None:
        issue_utc = _utc(self.issue_time_utc, name="issue_time_utc")
        init_utc = _utc(self.init_time_utc, name="init_time_utc")
        available_utc = _utc(self.assumed_available_at_utc, name="assumed_available_at_utc")
        object.__setattr__(self, "issue_time_utc", issue_utc)
        object.__setattr__(self, "init_time_utc", init_utc)
        object.__setattr__(self, "assumed_available_at_utc", available_utc)
        if init_utc.minute or init_utc.second or init_utc.microsecond:
            raise ValueError("GFS initialisation must be on an exact hour")
        if init_utc.hour not in GFS_CYCLES_UTC:
            raise ValueError("GFS initialisation hour must be 00, 06, 12, or 18 UTC")
        if self.lead_hours not in SUPPORTED_LEADS_HOURS:
            raise ValueError("supported lead must be 3..240 hours in 3-hour steps")
        if available_utc < init_utc:
            raise ValueError("assumed availability cannot precede initialisation")
        if self.publication_time_utc is not None:
            raise ValueError("GDEX d084001 does not provide a proven publication timestamp")
        if not self.availability_assumption_id.strip():
            raise ValueError("availability_assumption_id must be non-empty")

    @property
    def valid_time_utc(self) -> datetime:
        return self.init_time_utc + timedelta(hours=self.lead_hours)

    @property
    def file_name(self) -> str:
        return (
            f"gfs.0p25.{self.init_time_utc:%Y%m%d%H}."
            f"f{self.lead_hours:03d}.grib2"
        )

    @property
    def archive_path(self) -> str:
        return (
            f"files/g/{DATASET_ID}/{self.init_time_utc:%Y}/"
            f"{self.init_time_utc:%Y%m%d}/{self.file_name}"
        )

    @property
    def dds_url(self) -> str:
        return f"{TDS_ROOT}/dodsC/{self.archive_path}.dds"

    @property
    def ncss_url(self) -> str:
        return f"{TDS_ROOT}/ncss/grid/{self.archive_path}"

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "dataset_id": DATASET_ID,
            "issue_time_utc": _iso_utc(self.issue_time_utc),
            "init_time_utc": _iso_utc(self.init_time_utc),
            "lead_hours": self.lead_hours,
            "valid_time_utc": _iso_utc(self.valid_time_utc),
            "bbox": asdict(self.bbox),
            "availability_assumption_id": self.availability_assumption_id,
            "assumed_available_at_utc": _iso_utc(self.assumed_available_at_utc),
            "publication_time_utc": None,
        }


@dataclass(frozen=True)
class GFSVariableMapping:
    temperature: str
    relative_humidity: str
    precipitation: str
    all_precipitation_candidates: tuple[str, ...]
    dds_sha256: str


def discover_variable_mapping(dds: bytes | str) -> GFSVariableMapping:
    """Map stable 2 m fields and the version-dependent precipitation name."""

    content = dds.encode("utf-8") if isinstance(dds, str) else dds
    text = content.decode("utf-8", errors="strict")
    if not re.search(rf"\b{re.escape(TEMPERATURE_VARIABLE)}\b", text):
        raise GFSMetadataError(f"missing {TEMPERATURE_VARIABLE}")
    if not re.search(rf"\b{re.escape(RELATIVE_HUMIDITY_VARIABLE)}\b", text):
        raise GFSMetadataError(f"missing {RELATIVE_HUMIDITY_VARIABLE}")
    candidates = tuple(sorted(set(PRECIPITATION_NAME_RE.findall(text))))
    if not candidates:
        raise GFSMetadataError("missing total precipitation accumulation")
    if len(candidates) != 1:
        raise GFSMetadataError(
            "ambiguous total precipitation variables: " + ", ".join(candidates)
        )
    return GFSVariableMapping(
        temperature=TEMPERATURE_VARIABLE,
        relative_humidity=RELATIVE_HUMIDITY_VARIABLE,
        precipitation=candidates[0],
        all_precipitation_candidates=candidates,
        dds_sha256=_sha256_bytes(content),
    )


@dataclass(frozen=True)
class HTTPPayload:
    status_code: int
    url: str
    content: bytes
    headers: Mapping[str, str]


class GFSTransport(Protocol):
    def get(
        self,
        url: str,
        *,
        params: Sequence[tuple[str, str]] | None,
        max_bytes: int,
    ) -> HTTPPayload: ...


class RequestsGFSTransport:
    """HTTPS transport with bounded retries and a hard response-size limit."""

    def __init__(self, *, timeout_seconds: float = 60.0) -> None:
        self.timeout_seconds = timeout_seconds
        self._local = threading.local()

    @staticmethod
    def _new_session() -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.headers.update({"User-Agent": "agro-phenology-gfs-archive/1.0"})
        return session

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._new_session()
            self._local.session = session
        return session

    def get(
        self,
        url: str,
        *,
        params: Sequence[tuple[str, str]] | None,
        max_bytes: int,
    ) -> HTTPPayload:
        with self._session().get(
            url,
            params=params,
            timeout=self.timeout_seconds,
            stream=True,
        ) as response:
            if response.status_code != 200:
                preview = response.content[:500].decode("utf-8", errors="replace")
                raise GFSArchiveError(
                    f"HTTP {response.status_code} for {response.url}: {preview}"
                )
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > max_bytes:
                raise GFSDownloadTooLargeError(
                    f"declared response size {declared} exceeds {max_bytes} bytes"
                )
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                size += len(chunk)
                if size > max_bytes:
                    raise GFSDownloadTooLargeError(
                        f"response exceeded {max_bytes} bytes while streaming"
                    )
                chunks.append(chunk)
            return HTTPPayload(
                status_code=response.status_code,
                url=response.url,
                content=b"".join(chunks),
                headers=dict(response.headers),
            )


@dataclass(frozen=True)
class GFSSubsetArtifact:
    data_path: Path
    provenance_path: Path
    data_sha256: str
    data_bytes: int
    dds_sha256: str
    request: GFSSubsetRequest
    variable_mapping: GFSVariableMapping
    retrieved_at_utc: datetime
    cached: bool


class GFSArchiveClient:
    """Fetch regional archive subsets with cache integrity and bounded fanout."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        transport: GFSTransport | None = None,
        max_workers: int = 2,
        max_bbox_area_degrees_squared: float = 100.0,
        max_subset_bytes: int = 64 * 1024 * 1024,
        max_dds_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        if not 1 <= max_workers <= 8:
            raise ValueError("max_workers must be between 1 and 8")
        if max_bbox_area_degrees_squared <= 0:
            raise ValueError("max_bbox_area_degrees_squared must be positive")
        self.cache_root = Path(cache_root)
        self.transport = transport or RequestsGFSTransport()
        self.max_workers = max_workers
        self.max_bbox_area_degrees_squared = max_bbox_area_degrees_squared
        self.max_subset_bytes = max_subset_bytes
        self.max_dds_bytes = max_dds_bytes

    def _cache_dir(self, request: GFSSubsetRequest) -> Path:
        payload = json.dumps(
            request.canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        key = hashlib.sha256(payload).hexdigest()[:16]
        return (
            self.cache_root
            / DATASET_ID
            / f"{request.init_time_utc:%Y}"
            / f"{request.init_time_utc:%Y%m%d%H}"
            / f"f{request.lead_hours:03d}"
            / key
        )

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temp_path = Path(raw_temp)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _load_cached(self, request: GFSSubsetRequest) -> GFSSubsetArtifact | None:
        directory = self._cache_dir(request)
        data_path = directory / "subset.nc"
        provenance_path = directory / "provenance.json"
        if not data_path.exists() and not provenance_path.exists():
            return None
        if not data_path.exists() or not provenance_path.exists():
            raise GFSCacheIntegrityError(f"incomplete cache entry: {directory}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        actual_sha = _sha256_file(data_path)
        if actual_sha != provenance.get("data_sha256"):
            raise GFSCacheIntegrityError(f"checksum mismatch: {data_path}")
        if request.canonical_payload() != provenance.get("request"):
            raise GFSCacheIntegrityError(f"request mismatch: {provenance_path}")
        mapping_payload = provenance["variable_mapping"]
        mapping = GFSVariableMapping(
            temperature=mapping_payload["temperature"],
            relative_humidity=mapping_payload["relative_humidity"],
            precipitation=mapping_payload["precipitation"],
            all_precipitation_candidates=tuple(
                mapping_payload["all_precipitation_candidates"]
            ),
            dds_sha256=mapping_payload["dds_sha256"],
        )
        return GFSSubsetArtifact(
            data_path=data_path,
            provenance_path=provenance_path,
            data_sha256=actual_sha,
            data_bytes=data_path.stat().st_size,
            dds_sha256=provenance["dds_sha256"],
            request=request,
            variable_mapping=mapping,
            retrieved_at_utc=_parse_utc(provenance["retrieved_at_utc"]),
            cached=True,
        )

    def fetch_subset(
        self,
        request: GFSSubsetRequest,
        *,
        retrieved_at_utc: datetime | None = None,
    ) -> GFSSubsetArtifact:
        """Fetch one lead for one bounded region, or verify and reuse its cache."""

        cached = self._load_cached(request)
        if cached is not None:
            return cached
        if request.bbox.area_degrees_squared > self.max_bbox_area_degrees_squared:
            raise ValueError(
                "bbox is too large for guarded subset access: "
                f"{request.bbox.area_degrees_squared:.3f} deg2"
            )
        directory = self._cache_dir(request)
        dds_path = directory / "schema.dds"
        dds_sha_path = directory / "schema.sha256"
        if dds_path.exists() or dds_sha_path.exists():
            if not dds_path.exists() or not dds_sha_path.exists():
                raise GFSCacheIntegrityError(f"incomplete DDS cache: {directory}")
            dds_content = dds_path.read_bytes()
            dds_sha = _sha256_bytes(dds_content)
            if dds_sha != dds_sha_path.read_text(encoding="ascii").strip():
                raise GFSCacheIntegrityError(f"DDS checksum mismatch: {dds_path}")
            dds_url = request.dds_url
        else:
            dds_payload = self.transport.get(
                request.dds_url, params=None, max_bytes=self.max_dds_bytes
            )
            dds_content = dds_payload.content
            dds_sha = _sha256_bytes(dds_content)
            dds_url = dds_payload.url
            self._atomic_write(dds_path, dds_content)
            self._atomic_write(dds_sha_path, (dds_sha + "\n").encode("ascii"))
        mapping = discover_variable_mapping(dds_content)
        params = [
            ("var", mapping.temperature),
            ("var", mapping.relative_humidity),
            ("var", mapping.precipitation),
            *request.bbox.as_query(),
            ("time", _iso_utc(request.valid_time_utc)),
            ("horizStride", "1"),
            ("accept", "netcdf3"),
        ]
        subset_payload = self.transport.get(
            request.ncss_url, params=params, max_bytes=self.max_subset_bytes
        )
        if not subset_payload.content.startswith(b"CDF"):
            preview = subset_payload.content[:200].decode("utf-8", errors="replace")
            raise GFSArchiveError(f"NCSS response is not NetCDF3: {preview}")
        retrieved = _utc(
            retrieved_at_utc or datetime.now(timezone.utc), name="retrieved_at_utc"
        )
        data_path = directory / "subset.nc"
        provenance_path = directory / "provenance.json"
        data_sha = _sha256_bytes(subset_payload.content)
        self._atomic_write(data_path, subset_payload.content)
        provenance = {
            "schema_version": "gfs_gdex_subset_provenance_v1",
            "dataset_id": DATASET_ID,
            "dataset_doi": DATASET_DOI,
            "request": request.canonical_payload(),
            "archive_path": request.archive_path,
            "dds_url": dds_url,
            "ncss_url": subset_payload.url,
            "publication_time_utc": None,
            "retrieved_at_utc": _iso_utc(retrieved),
            "variable_mapping": asdict(mapping),
            "dds_sha256": mapping.dds_sha256,
            "data_sha256": data_sha,
            "data_bytes": len(subset_payload.content),
            "response_headers": {
                key: value
                for key, value in subset_payload.headers.items()
                if key.lower() in {"content-type", "content-length", "last-modified", "etag"}
            },
        }
        self._atomic_write(
            provenance_path,
            (json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
        return GFSSubsetArtifact(
            data_path=data_path,
            provenance_path=provenance_path,
            data_sha256=data_sha,
            data_bytes=len(subset_payload.content),
            dds_sha256=mapping.dds_sha256,
            request=request,
            variable_mapping=mapping,
            retrieved_at_utc=retrieved,
            cached=False,
        )

    def fetch_many(
        self, requests_to_fetch: Iterable[GFSSubsetRequest]
    ) -> list[GFSSubsetArtifact]:
        """Fetch a request sequence with deterministic output order and low fanout."""

        requests_list = list(requests_to_fetch)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(self.fetch_subset, requests_list))


def build_experiment_requests(
    selection: GFSCycleSelection,
    bbox: GFSBoundingBox,
    *,
    leads_hours: Sequence[int] | None = None,
) -> list[GFSSubsetRequest]:
    """Build six-hour valid slots in the fixed bands relative to issue time.

    The init-to-issue gap depends on the availability scenario.  Consequently
    the needed archive leads also change: an issue at 05Z with a 00Z cycle uses
    f006..f168, while the previous day's 18Z cycle uses f012..f174.
    """

    if leads_hours is None:
        candidates = EXPERIMENT_LEADS_HOURS
        leads_hours = tuple(
            lead
            for lead in candidates
            if 0.0
            < (
                selection.init_time_utc
                + timedelta(hours=lead)
                - selection.issue_time_utc
            ).total_seconds()
            / 3600.0
            <= 168.0
        )

    return [
        GFSSubsetRequest(
            issue_time_utc=selection.issue_time_utc,
            init_time_utc=selection.init_time_utc,
            lead_hours=lead,
            bbox=bbox,
            availability_assumption_id=selection.assumption_id,
            assumed_available_at_utc=selection.assumed_available_at_utc,
        )
        for lead in leads_hours
    ]


def load_subset_frame(artifact: GFSSubsetArtifact) -> pd.DataFrame:
    """Read one NCSS NetCDF3 response and select 2 m temperature/humidity.

    Precipitation interval bounds are retained.  A caller can therefore form a
    non-overlapping interval cover; it must not blindly sum overlapping 3- and
    6-hour accumulations.
    """

    try:
        from scipy.io import netcdf_file
    except ImportError as error:  # pragma: no cover - optional dependency guard
        raise RuntimeError("scipy is required to read GFS NetCDF3 subsets") from error

    with netcdf_file(artifact.data_path, "r", mmap=False) as dataset:
        variables = dataset.variables
        mapping = artifact.variable_mapping
        for name in (mapping.temperature, mapping.relative_humidity, mapping.precipitation):
            if name not in variables:
                raise GFSMetadataError(f"downloaded subset is missing {name}")
        latitudes = np.asarray(variables["latitude"].data, dtype=float).copy()
        longitudes = np.asarray(variables["longitude"].data, dtype=float).copy()

        def two_m_grid(variable_name: str) -> np.ndarray:
            variable = variables[variable_name]
            data = np.asarray(variable.data, dtype=float).copy()
            dimensions = list(variable.dimensions)
            height_positions = [
                index for index, name in enumerate(dimensions) if name.startswith("height_above_ground")
            ]
            if len(height_positions) != 1:
                raise GFSMetadataError(f"cannot identify height axis for {variable_name}")
            height_index = height_positions[0]
            heights = np.asarray(variables[dimensions[height_index]].data, dtype=float).copy()
            matches = np.flatnonzero(np.isclose(heights, 2.0))
            if len(matches) != 1:
                raise GFSMetadataError(f"cannot identify unique 2 m level for {variable_name}")
            data = np.take(data, int(matches[0]), axis=height_index)
            dimensions.pop(height_index)
            time_positions = [index for index, name in enumerate(dimensions) if name.startswith("time")]
            if len(time_positions) != 1 or data.shape[time_positions[0]] != 1:
                raise GFSMetadataError(f"expected one valid time for {variable_name}")
            data = np.take(data, 0, axis=time_positions[0])
            dimensions.pop(time_positions[0])
            if dimensions != ["latitude", "longitude"]:
                raise GFSMetadataError(
                    f"unexpected dimensions for {variable_name}: {dimensions}"
                )
            return data

        temperature = two_m_grid(mapping.temperature)
        humidity = two_m_grid(mapping.relative_humidity)
        precipitation_variable = variables[mapping.precipitation]
        precipitation = np.asarray(precipitation_variable.data, dtype=float).copy()
        precip_dimensions = list(precipitation_variable.dimensions)
        time_positions = [
            index for index, name in enumerate(precip_dimensions) if name.startswith("time")
        ]
        if len(time_positions) != 1:
            raise GFSMetadataError("cannot identify precipitation time axis")
        precip_time_name = precip_dimensions[time_positions[0]]
        precip_dimensions.pop(time_positions[0])
        if precip_dimensions != ["latitude", "longitude"]:
            raise GFSMetadataError(
                f"unexpected precipitation dimensions: {precip_dimensions}"
            )
        bounds_name = f"{precip_time_name}_bounds"
        if bounds_name not in variables:
            raise GFSMetadataError("precipitation time bounds are missing")
        bounds = np.asarray(variables[bounds_name].data, dtype=float).reshape(-1, 2)
        if len(bounds) != precipitation_variable.data.shape[time_positions[0]]:
            raise GFSMetadataError("precipitation values and interval bounds disagree")
        matching = np.flatnonzero(
            np.isclose(bounds[:, 1], artifact.request.lead_hours)
            & ((bounds[:, 1] - bounds[:, 0]) > 0.0)
        )
        if not len(matching):
            raise GFSMetadataError(
                f"no precipitation interval ends at lead {artifact.request.lead_hours}"
            )
        # Mixed-interval files may contain a cumulative record and a recent
        # interval with the same valid time.  The interval with the latest
        # start is the operational increment; cumulative records are excluded.
        selected_time_index = int(matching[np.argmax(bounds[matching, 0])])
        precipitation = np.take(
            np.asarray(precipitation_variable.data, dtype=float).copy(),
            selected_time_index,
            axis=time_positions[0],
        )
        precip_dimensions = list(precipitation_variable.dimensions)
        precip_dimensions.pop(time_positions[0])
        interval_start, interval_end = (
            float(bounds[selected_time_index, 0]),
            float(bounds[selected_time_index, 1]),
        )
        if not math.isclose(interval_end, artifact.request.lead_hours, abs_tol=1e-6):
            raise GFSMetadataError(
                f"precipitation interval ends at {interval_end}, expected {artifact.request.lead_hours}"
            )
        interval_hours = interval_end - interval_start
        if not 0.0 < interval_hours <= 24.0:
            raise GFSMetadataError(
                f"refusing long/cumulative precipitation interval [{interval_start}, {interval_end}]"
            )
        if artifact.request.lead_hours % 6 == 0 and not math.isclose(
            interval_hours, 6.0, abs_tol=1e-6
        ):
            raise GFSMetadataError(
                "six-hour experiment lead does not contain a six-hour precipitation "
                f"increment: [{interval_start}, {interval_end}]"
            )
        expected_shape = (len(latitudes), len(longitudes))
        for name, grid in (
            ("temperature", temperature),
            ("relative_humidity", humidity),
            ("precipitation", precipitation),
        ):
            if grid.shape != expected_shape:
                raise GFSMetadataError(
                    f"{name} grid shape {grid.shape} does not match {expected_shape}"
                )

    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    frame = pd.DataFrame(
        {
            "latitude": lat_grid.ravel(),
            "longitude": lon_grid.ravel(),
            "temperature_2m_k": temperature.ravel(),
            "relative_humidity_2m_pct": humidity.ravel(),
            "precipitation_kg_m2": precipitation.ravel(),
        }
    )
    frame["init_time_utc"] = pd.Timestamp(artifact.request.init_time_utc)
    frame["issue_time_utc"] = pd.Timestamp(artifact.request.issue_time_utc)
    frame["valid_time_utc"] = pd.Timestamp(artifact.request.valid_time_utc)
    frame["lead_hours"] = artifact.request.lead_hours
    frame["precip_interval_start_hours"] = interval_start
    frame["precip_interval_end_hours"] = interval_end
    frame["precip_interval_hours"] = interval_end - interval_start
    frame["publication_time_utc"] = pd.NaT
    frame["assumed_available_at_utc"] = pd.Timestamp(
        artifact.request.assumed_available_at_utc
    )
    frame["availability_assumption_id"] = artifact.request.availability_assumption_id
    frame["retrieved_at_utc"] = pd.Timestamp(artifact.retrieved_at_utc)
    frame["source_sha256"] = artifact.data_sha256
    return frame


def canonical_gfs_cell_id(latitude: float, longitude: float) -> str:
    """Return a stable public-grid identifier, independent of private field IDs."""

    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise ValueError("grid coordinates must be finite")
    lat = round(latitude * 4.0) / 4.0
    lon = ((round(longitude * 4.0) / 4.0 + 180.0) % 360.0) - 180.0
    if not -90.0 <= lat <= 90.0:
        raise ValueError("latitude is outside the GFS grid")
    return f"gfs025_lat{lat:+06.2f}_lon{lon:+07.2f}"


def aggregate_forecast_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate six-hour snapshots into the two frozen decision-time bands.

    The band assignment uses the valid time relative to the decision time:
    ``(0, 72]`` and ``(72, 168]`` hours.  Each six-hour precipitation amount is
    assigned by its interval end; intervals are not split fractionally at the
    band boundary.  Incomplete bands receive null model features.
    """

    required = {
        "issue_time_utc",
        "init_time_utc",
        "valid_time_utc",
        "latitude",
        "longitude",
        "temperature_2m_k",
        "relative_humidity_2m_pct",
        "precipitation_kg_m2",
        "precip_interval_hours",
        "availability_assumption_id",
        "assumed_available_at_utc",
        "source_sha256",
        "retrieved_at_utc",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing forecast snapshot columns: {sorted(missing)}")
    work = frame.copy()
    for column in (
        "issue_time_utc",
        "init_time_utc",
        "valid_time_utc",
        "assumed_available_at_utc",
        "retrieved_at_utc",
    ):
        work[column] = pd.to_datetime(work[column], utc=True, errors="raise")
    if not np.isclose(work["precip_interval_hours"].astype(float), 6.0).all():
        raise GFSMetadataError("primary feature table requires six-hour precipitation increments")
    work["valid_offset_hours"] = (
        work["valid_time_utc"] - work["issue_time_utc"]
    ).dt.total_seconds() / 3600.0
    work = work[
        (work["valid_offset_hours"] > 0.0)
        & (work["valid_offset_hours"] <= 168.0)
    ].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "issue_date",
                "issue_time_utc",
                "gfs_cell_id",
                "latitude",
                "longitude",
                "availability_scenario_hours",
                "selected_init_utc",
                "assumed_available_at_utc",
                "publication_time_utc",
                "selection_rule_id",
                "source_dataset_id",
                "source_hashes_json",
                "first_lead_h",
                "last_lead_h",
                "native_step_hours",
                "expected_steps_1_3d",
                "observed_steps_1_3d",
                "expected_steps_4_7d",
                "observed_steps_4_7d",
                "complete_1_3d",
                "complete_4_7d",
                "forecast_available",
                *FORECAST_FEATURE_COLUMNS,
            ]
        )
    work["gfs_cell_id"] = [
        canonical_gfs_cell_id(lat, lon)
        for lat, lon in zip(work["latitude"], work["longitude"], strict=True)
    ]
    assumption_lag = (
        work["assumed_available_at_utc"] - work["init_time_utc"]
    ).dt.total_seconds() / 3600.0
    if not np.isclose(assumption_lag, np.round(assumption_lag)).all():
        raise ValueError("availability scenario must use an integral number of hours")
    work["availability_scenario_hours"] = np.round(assumption_lag).astype(int)
    work["temperature_2m_c"] = work["temperature_2m_k"].astype(float) - 273.15

    group_columns = [
        "issue_time_utc",
        "gfs_cell_id",
        "latitude",
        "longitude",
        "availability_scenario_hours",
        "init_time_utc",
        "availability_assumption_id",
        "assumed_available_at_utc",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in work.groupby(group_columns, sort=True, dropna=False):
        row = dict(zip(group_columns, keys, strict=True))
        selected_init = row.pop("init_time_utc")
        selection_rule = row.pop("availability_assumption_id")
        row.update(
            {
                "issue_date": pd.Timestamp(row["issue_time_utc"]).date(),
                "selected_init_utc": selected_init,
                "selection_rule_id": selection_rule,
                "publication_time_utc": pd.NaT,
                "source_dataset_id": DATASET_ID,
                "source_hashes_json": json.dumps(
                    sorted(set(group["source_sha256"].astype(str))), separators=(",", ":")
                ),
                "first_lead_h": int(
                    ((group["valid_time_utc"] - pd.Timestamp(selected_init))
                     .dt.total_seconds() / 3600.0).min()
                ),
                "last_lead_h": int(
                    ((group["valid_time_utc"] - pd.Timestamp(selected_init))
                     .dt.total_seconds() / 3600.0).max()
                ),
                "native_step_hours": 6,
            }
        )
        all_complete = True
        for lower, upper, band in FORECAST_BANDS_HOURS:
            subset = group[
                (group["valid_offset_hours"] > lower)
                & (group["valid_offset_hours"] <= upper)
            ].sort_values("valid_time_utc")
            expected = int((upper - lower) / 6.0)
            observed = int(subset["valid_time_utc"].nunique())
            finite = subset[
                ["temperature_2m_c", "relative_humidity_2m_pct", "precipitation_kg_m2"]
            ].apply(np.isfinite).all(axis=None)
            complete = observed == expected and bool(finite)
            suffix = "1_3d" if band == "d1_3" else "4_7d"
            row[f"expected_steps_{suffix}"] = expected
            row[f"observed_steps_{suffix}"] = observed
            row[f"complete_{suffix}"] = complete
            all_complete &= complete
            features = {
                f"fcst_t_mean_{band}": subset["temperature_2m_c"].mean(),
                f"fcst_t_min_{band}": subset["temperature_2m_c"].min(),
                f"fcst_t_max_{band}": subset["temperature_2m_c"].max(),
                f"fcst_rh_mean_{band}": subset["relative_humidity_2m_pct"].mean(),
                f"fcst_rh_max_{band}": subset["relative_humidity_2m_pct"].max(),
                f"fcst_precip_sum_{band}": subset["precipitation_kg_m2"].sum(min_count=1),
            }
            row.update(features if complete else {name: np.nan for name in features})
        row["forecast_available"] = all_complete
        rows.append(row)
    result = pd.DataFrame(rows)
    ordered = [
        "issue_date",
        "issue_time_utc",
        "gfs_cell_id",
        "latitude",
        "longitude",
        "availability_scenario_hours",
        "selected_init_utc",
        "assumed_available_at_utc",
        "publication_time_utc",
        "selection_rule_id",
        "source_dataset_id",
        "source_hashes_json",
        "first_lead_h",
        "last_lead_h",
        "native_step_hours",
        "expected_steps_1_3d",
        "observed_steps_1_3d",
        "expected_steps_4_7d",
        "observed_steps_4_7d",
        "complete_1_3d",
        "complete_4_7d",
        "forecast_available",
        *FORECAST_FEATURE_COLUMNS,
    ]
    return result[ordered].sort_values(
        ["issue_date", "availability_scenario_hours", "gfs_cell_id"], kind="mergesort"
    ).reset_index(drop=True)


def choose_non_overlapping_precipitation_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    """Choose a gap-free accumulation cover for each init/grid cell.

    Some GFS files contain both cumulative and short accumulation records, and
    older products alternate 3- and 6-hour records.  The NCSS extraction keeps
    a short record where possible; this function additionally removes overlap
    across leads and raises on gaps instead of silently double counting rain.
    """

    required = {
        "init_time_utc",
        "latitude",
        "longitude",
        "precip_interval_start_hours",
        "precip_interval_end_hours",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing interval columns: {sorted(missing)}")
    chosen: list[pd.DataFrame] = []
    group_columns = ["init_time_utc", "latitude", "longitude"]
    for _, group in frame.groupby(group_columns, sort=False, dropna=False):
        ordered = group.sort_values(
            ["precip_interval_end_hours", "precip_interval_start_hours"],
            ascending=[True, False],
            kind="mergesort",
        )
        cursor = 0.0
        rows: list[Any] = []
        while cursor < float(ordered["precip_interval_end_hours"].max()) - 1e-9:
            candidates = ordered[
                np.isclose(ordered["precip_interval_start_hours"], cursor)
                & (ordered["precip_interval_end_hours"] > cursor)
            ]
            if candidates.empty:
                raise GFSMetadataError(f"precipitation interval gap after hour {cursor}")
            row = candidates.sort_values(
                "precip_interval_end_hours", ascending=False, kind="mergesort"
            ).iloc[0]
            rows.append(row.name)
            cursor = float(row["precip_interval_end_hours"])
        chosen.append(group.loc[rows])
    if not chosen:
        return frame.iloc[0:0].copy()
    return pd.concat(chosen, ignore_index=True).sort_values(
        group_columns + ["precip_interval_end_hours"], kind="mergesort"
    )


DEFAULT_DAILY_DECISIONS = Path(
    "results/late_blight_early_warning/20260910_first_cycle_v3/daily_decisions.parquet"
)


def load_historical_issue_times(
    daily_decisions_path: str | Path,
    *,
    years: Sequence[int] | None = None,
    issue_dates: Sequence[str] | None = None,
) -> list[datetime]:
    """Load unique service decision times without exporting private field rows."""

    path = Path(daily_decisions_path)
    decisions = pd.read_parquet(
        path, columns=["issue_date", "issued_at", "service_active"]
    )
    active = decisions[decisions["service_active"].fillna(False).astype(bool)].copy()
    active["issue_date"] = pd.to_datetime(active["issue_date"], errors="raise").dt.date
    active["issued_at_utc"] = pd.to_datetime(active["issued_at"], utc=True, errors="raise")
    per_date_counts = active.groupby("issue_date")["issued_at_utc"].nunique()
    if (per_date_counts > 1).any():
        bad = per_date_counts[per_date_counts > 1].index.astype(str).tolist()[:5]
        raise ValueError(f"multiple issue times for one decision date: {bad}")
    unique = active[["issue_date", "issued_at_utc"]].drop_duplicates()
    unique = unique[unique["issued_at_utc"] >= pd.Timestamp("2015-01-15T00:00:00Z")]
    if years is not None:
        wanted_years = {int(year) for year in years}
        unique = unique[unique["issue_date"].map(lambda value: value.year in wanted_years)]
    if issue_dates is not None:
        wanted_dates = {pd.Timestamp(value).date() for value in issue_dates}
        unique = unique[unique["issue_date"].isin(wanted_dates)]
    return [
        timestamp.to_pydatetime()
        for timestamp in unique.sort_values("issued_at_utc")["issued_at_utc"]
    ]


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
    os.close(fd)
    temp_path = Path(raw_temp)
    try:
        frame.to_parquet(temp_path, index=False)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def build_forecast_feature_checkpoints(
    *,
    issue_times: Sequence[datetime],
    bbox: GFSBoundingBox,
    cache_root: str | Path,
    output_dir: str | Path,
    availability_scenarios_hours: Sequence[int] = (4, 7),
    max_workers: int = 2,
    client: GFSArchiveClient | None = None,
) -> list[Path]:
    """Fetch and checkpoint annual public-grid feature tables.

    One issue date is committed to its annual Parquet checkpoint only after all
    28 required lead files have been fetched, parsed, and aggregated.  Existing
    dates are verified and skipped, so an interrupted run resumes without
    redownloading valid cache entries.  No private field identifier is written.
    """

    output_root = Path(output_dir)
    archive = client or GFSArchiveClient(
        cache_root, max_workers=max_workers
    )
    written: list[Path] = []
    normalized_issues = sorted({_utc(value, name="issue_time") for value in issue_times})
    for scenario_hours in availability_scenarios_hours:
        if int(scenario_hours) <= 0:
            raise ValueError("availability scenario hours must be positive")
        rule_id = f"gfs_full_run_plus_{int(scenario_hours)}h_unverified_v1"
        for year in sorted({value.year for value in normalized_issues}):
            annual_path = output_root / f"forecast_features_{year}_{int(scenario_hours)}h.parquet"
            if annual_path.exists():
                annual = pd.read_parquet(annual_path)
                missing = set(FORECAST_FEATURE_COLUMNS) - set(annual.columns)
                if missing:
                    raise GFSCacheIntegrityError(
                        f"annual checkpoint misses features {sorted(missing)}: {annual_path}"
                    )
                completed = set(pd.to_datetime(annual["issue_date"]).dt.date)
            else:
                annual = pd.DataFrame()
                completed = set()
            for issue_time in [value for value in normalized_issues if value.year == year]:
                if issue_time.date() in completed:
                    continue
                selection = select_assumed_available_cycle(
                    issue_time,
                    assumed_publication_lag=timedelta(hours=int(scenario_hours)),
                    assumption_id=rule_id,
                )
                request_plan = build_experiment_requests(selection, bbox)
                artifacts = archive.fetch_many(request_plan)
                snapshots = pd.concat(
                    [load_subset_frame(artifact) for artifact in artifacts],
                    ignore_index=True,
                )
                features = aggregate_forecast_features(snapshots)
                if features.empty or not features["forecast_available"].all():
                    raise GFSMetadataError(
                        f"incomplete feature horizon for {issue_time.date()} scenario {scenario_hours}h"
                    )
                annual = pd.concat([annual, features], ignore_index=True)
                duplicate_keys = annual.duplicated(
                    ["issue_date", "gfs_cell_id", "availability_scenario_hours"],
                    keep=False,
                )
                if duplicate_keys.any():
                    raise GFSCacheIntegrityError(
                        f"duplicate annual feature key in {annual_path}"
                    )
                annual = annual.sort_values(
                    ["issue_date", "gfs_cell_id"], kind="mergesort"
                ).reset_index(drop=True)
                _atomic_write_parquet(annual, annual_path)
                completed.add(issue_time.date())
            if annual_path.exists():
                written.append(annual_path)
    all_annual_paths = sorted(output_root.glob("forecast_features_[0-9][0-9][0-9][0-9]_*h.parquet"))
    if all_annual_paths:
        combined = pd.concat(
            [pd.read_parquet(path) for path in all_annual_paths], ignore_index=True
        )
        combined = combined.sort_values(
            ["issue_date", "availability_scenario_hours", "gfs_cell_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        _atomic_write_parquet(combined, output_root / "forecast_features.parquet")
        manifest = {
            "schema_version": "gfs_forecast_feature_build_v1",
            "source_dataset_id": DATASET_ID,
            "publication_time_semantics": "unknown_not_provided_by_archive",
            "availability_scenarios_hours": [
                int(value) for value in availability_scenarios_hours
            ],
            "issue_dates": len(normalized_issues),
            "annual_files": [
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in all_annual_paths
            ],
            "combined": {
                "path": "forecast_features.parquet",
                "bytes": (output_root / "forecast_features.parquet").stat().st_size,
                "sha256": _sha256_file(output_root / "forecast_features.parquet"),
            },
        }
        GFSArchiveClient._atomic_write(
            output_root / "forecast_features_manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    return written


def _bbox_from_cli(values: Sequence[float]) -> GFSBoundingBox:
    if len(values) != 4:
        raise ValueError("bbox requires WEST SOUTH EAST NORTH")
    west, south, east, north = (float(value) for value in values)
    return GFSBoundingBox(west=west, east=east, south=south, north=north)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded GDEX d084001 feature extraction")
    parser.add_argument("command", choices=("plan", "build"))
    parser.add_argument("--daily-decisions", type=Path, default=DEFAULT_DAILY_DECISIONS)
    parser.add_argument(
        "--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"), required=True
    )
    parser.add_argument("--years", nargs="*", type=int)
    parser.add_argument("--issue-dates", nargs="*")
    parser.add_argument("--scenario-hours", nargs="+", type=int, default=(4, 7))
    parser.add_argument("--limit-issue-dates", type=int)
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/gfs_d084001"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--execute-network", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    bbox = _bbox_from_cli(args.bbox)
    issue_times = load_historical_issue_times(
        args.daily_decisions, years=args.years or None, issue_dates=args.issue_dates or None
    )
    if args.limit_issue_dates is not None:
        if args.limit_issue_dates < 1:
            raise ValueError("--limit-issue-dates must be positive")
        issue_times = issue_times[: args.limit_issue_dates]
    request_count = len(issue_times) * len(args.scenario_hours) * 28
    plan = {
        "dataset_id": DATASET_ID,
        "issue_dates": len(issue_times),
        "first_issue_time_utc": _iso_utc(issue_times[0]) if issue_times else None,
        "last_issue_time_utc": _iso_utc(issue_times[-1]) if issue_times else None,
        "availability_scenarios_hours": args.scenario_hours,
        "requests": request_count,
        "bbox": asdict(bbox),
        "network_executed": False,
    }
    if args.command == "plan":
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if not args.execute_network:
        raise SystemExit("build requires --execute-network after reviewing the request plan")
    if args.output_dir is None:
        raise SystemExit("build requires --output-dir")
    paths = build_forecast_feature_checkpoints(
        issue_times=issue_times,
        bbox=bbox,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        availability_scenarios_hours=args.scenario_hours,
        max_workers=args.max_workers,
    )
    plan["network_executed"] = True
    plan["annual_outputs"] = [str(path) for path in paths]
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


__all__ = [
    "DATASET_DOI",
    "DATASET_ID",
    "EXPERIMENT_LEADS_HOURS",
    "FORECAST_BANDS_HOURS",
    "FORECAST_FEATURE_COLUMNS",
    "SUPPORTED_LEADS_HOURS",
    "GFSArchiveClient",
    "GFSArchiveError",
    "GFSBoundingBox",
    "GFSCacheIntegrityError",
    "GFSCycleSelection",
    "GFSDownloadTooLargeError",
    "GFSMetadataError",
    "GFSSubsetArtifact",
    "GFSSubsetRequest",
    "GFSVariableMapping",
    "HTTPPayload",
    "RequestsGFSTransport",
    "aggregate_forecast_features",
    "build_forecast_feature_checkpoints",
    "build_experiment_requests",
    "canonical_gfs_cell_id",
    "choose_non_overlapping_precipitation_intervals",
    "discover_variable_mapping",
    "load_historical_issue_times",
    "load_subset_frame",
    "select_assumed_available_cycle",
]


if __name__ == "__main__":  # pragma: no cover - exercised through CLI smoke tests
    raise SystemExit(main())
