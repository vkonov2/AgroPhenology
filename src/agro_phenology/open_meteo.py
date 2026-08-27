"""Resilient, cached client for Open-Meteo Historical Weather API."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_VARIABLES = ("temperature_2m",)
ALLOWED_ARCHIVE_MODELS = {"era5_land", "era5"}
ALLOWED_OPTIONAL_VARIABLES = {
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
}


class OpenMeteoError(RuntimeError):
    """Raised for request failures or malformed Open-Meteo responses."""


@dataclass(frozen=True)
class WeatherResponse:
    """Parsed hourly data and reproducibility metadata."""

    hourly: pd.DataFrame
    metadata: dict[str, Any]
    cache_hit: bool
    raw_path: Path


def build_archive_params(
    latitude: float,
    longitude: float,
    start_date: date | str,
    end_date: date | str,
    hourly_variables: tuple[str, ...] = DEFAULT_VARIABLES,
    model: str = "era5_land",
) -> dict[str, Any]:
    """Build an explicit single-model historical request contract."""

    variables = tuple(dict.fromkeys(hourly_variables))
    if model not in ALLOWED_ARCHIVE_MODELS:
        raise ValueError(
            f"Unsupported archive model: {model!r}; expected one of {sorted(ALLOWED_ARCHIVE_MODELS)}"
        )
    if model == "era5_land" and "temperature_2m" not in variables:
        raise ValueError("temperature_2m is required for the degree-day model")
    if model == "era5" and variables != ("precipitation",):
        raise ValueError("The ERA5 profile is restricted to precipitation only")
    unsupported = set(variables) - ({"temperature_2m"} | ALLOWED_OPTIONAL_VARIABLES)
    if unsupported:
        raise ValueError(f"Unsupported hourly variables: {sorted(unsupported)}")
    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": ",".join(variables),
        "models": model,
        "timezone": "auto",
        "cell_selection": "land",
    }
    if "temperature_2m" in variables:
        params["temperature_unit"] = "celsius"
    if "precipitation" in variables:
        params["precipitation_unit"] = "mm"
    return params


def cache_key(params: dict[str, Any]) -> str:
    """Create a stable key from all request parameters affecting the response."""

    canonical = json.dumps(params, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_archive_response(
    payload: dict[str, Any],
    required_variables: tuple[str, ...] = DEFAULT_VARIABLES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and parse a saved Open-Meteo response without network access."""

    required_top = {"latitude", "longitude", "timezone", "hourly"}
    missing = required_top - set(payload)
    if missing:
        raise OpenMeteoError(f"Open-Meteo response is missing keys: {sorted(missing)}")
    hourly = payload.get("hourly")
    units = payload.get("hourly_units")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise OpenMeteoError("Open-Meteo response has no hourly.time array")
    if not isinstance(units, dict):
        raise OpenMeteoError("Open-Meteo response has no hourly_units object")
    lengths = {name: len(values) for name, values in hourly.items() if isinstance(values, list)}
    if not lengths or len(set(lengths.values())) != 1 or set(lengths) != set(hourly):
        raise OpenMeteoError("Open-Meteo hourly arrays have inconsistent types or lengths")
    missing_variables = set(required_variables) - set(hourly)
    if missing_variables:
        raise OpenMeteoError(
            f"Open-Meteo response is missing hourly variables: {sorted(missing_variables)}"
        )
    missing_units = set(required_variables) - set(units)
    if missing_units:
        raise OpenMeteoError(
            f"Open-Meteo response is missing hourly units: {sorted(missing_units)}"
        )
    if "precipitation" in required_variables and units.get("precipitation") != "mm":
        raise OpenMeteoError("Open-Meteo precipitation unit must be mm")
    frame = pd.DataFrame(hourly)
    metadata = {
        "returned_latitude": payload.get("latitude"),
        "returned_longitude": payload.get("longitude"),
        "elevation": payload.get("elevation"),
        "timezone": payload.get("timezone"),
        "timezone_abbreviation": payload.get("timezone_abbreviation"),
        "utc_offset_seconds": payload.get("utc_offset_seconds"),
        "hourly_units": units,
    }
    return frame, metadata


class OpenMeteoClient:
    """Historical API client with retries, exponential backoff, and raw caching."""

    def __init__(
        self,
        cache_dir: str | Path = "data/cache/open_meteo",
        timeout_seconds: float = 30.0,
        max_attempts: int = 4,
        backoff_seconds: float = 1.0,
        incomplete_cache_ttl_hours: float = 24.0,
        session: requests.Session | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.incomplete_cache_ttl_hours = incomplete_cache_ttl_hours
        self.session = session or requests.Session()

    def _cached_response_is_usable(
        self,
        payload: dict[str, Any],
        metadata: dict[str, Any],
        requested_variables: tuple[str, ...],
    ) -> bool:
        """Keep complete history indefinitely and refresh stale incomplete responses."""

        hourly = payload.get("hourly") if isinstance(payload, dict) else None
        complete = bool(
            isinstance(hourly, dict)
            and all(
                isinstance(hourly.get(variable), list)
                and len(hourly[variable]) > 0
                and all(value is not None for value in hourly[variable])
                for variable in requested_variables
            )
        )
        if complete:
            return True
        try:
            retrieved_at = datetime.fromisoformat(str(metadata["retrieved_at"]).replace("Z", "+00:00"))
            if retrieved_at.tzinfo is None:
                retrieved_at = retrieved_at.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - retrieved_at.astimezone(timezone.utc)).total_seconds() / 3600
        except (KeyError, TypeError, ValueError):
            return False
        return age_hours <= self.incomplete_cache_ttl_hours

    def _fetch_hourly(
        self,
        latitude: float,
        longitude: float,
        start_date: date | str,
        end_date: date | str,
        hourly_variables: tuple[str, ...],
        model: str,
    ) -> WeatherResponse:
        """Fetch or load cached hourly data from one explicitly selected reanalysis."""

        params = build_archive_params(
            latitude,
            longitude,
            start_date,
            end_date,
            hourly_variables,
            model=model,
        )
        key = cache_key(params)
        raw_path = self.cache_dir / f"{key}.json"
        metadata_path = self.cache_dir / f"{key}.metadata.json"
        if raw_path.exists() and metadata_path.exists():
            try:
                payload = json.loads(raw_path.read_text(encoding="utf-8"))
                saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                frame, response_metadata = parse_archive_response(payload, hourly_variables)
            except (OSError, json.JSONDecodeError, OpenMeteoError) as exc:
                raise OpenMeteoError(f"Cached Open-Meteo response is invalid: {raw_path}: {exc}") from exc
            if self._cached_response_is_usable(payload, saved_metadata, hourly_variables):
                return WeatherResponse(frame, {**saved_metadata, **response_metadata}, True, raw_path)

        response: requests.Response | None = None
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.session.get(ARCHIVE_URL, params=params, timeout=self.timeout_seconds)
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(f"temporary HTTP {response.status_code}", response=response)
                response.raise_for_status()
                payload = response.json()
                frame, response_metadata = parse_archive_response(payload, hourly_variables)
                break
            except (requests.RequestException, ValueError, OpenMeteoError) as exc:
                last_error = exc
                if attempt == self.max_attempts:
                    raise OpenMeteoError(
                        f"Open-Meteo request failed after {self.max_attempts} attempts: {exc}"
                    ) from exc
                time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
        else:  # pragma: no cover - loop always raises or breaks
            raise OpenMeteoError(f"Open-Meteo request failed: {last_error}")

        assert response is not None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Store the exact response bytes unchanged; metadata lives in a sidecar.
        raw_path.write_bytes(response.content)
        metadata = {
            "requested_latitude": params["latitude"],
            "requested_longitude": params["longitude"],
            "model": params["models"],
            "period": {"start_date": params["start_date"], "end_date": params["end_date"]},
            "hourly_variables": params["hourly"].split(","),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "url": response.url,
            "request_parameters": params,
            **response_metadata,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return WeatherResponse(frame, metadata, False, raw_path)

    def fetch_hourly(
        self,
        latitude: float,
        longitude: float,
        start_date: date | str,
        end_date: date | str,
        hourly_variables: tuple[str, ...] = DEFAULT_VARIABLES,
    ) -> WeatherResponse:
        """Fetch or load cached ERA5-Land hourly data."""

        return self._fetch_hourly(
            latitude,
            longitude,
            start_date,
            end_date,
            hourly_variables,
            model="era5_land",
        )

    def fetch_era5_precipitation_hourly(
        self,
        latitude: float,
        longitude: float,
        start_date: date | str,
        end_date: date | str,
    ) -> WeatherResponse:
        """Fetch or load the separately authorized ERA5 precipitation series."""

        return self._fetch_hourly(
            latitude,
            longitude,
            start_date,
            end_date,
            ("precipitation",),
            model="era5",
        )
