"""Strict source adapters for the late-blight shadow-readiness contour.

The frozen episode models were built from one explicit ERA5 profile.  This
module describes and parses that profile without silently substituting a live
forecast, Best Match, ERA5-Land, downscaled values, or a shifted cutoff.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .early_warning_core import EPISODE_FEATURES, _episode_features


FROZEN_ERA5_PROFILE_ID = "open_meteo_archive_era5_025_nearest_no_downscale_v1"
FROZEN_ERA5_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
FROZEN_ERA5_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
    "shortwave_radiation",
    "wind_speed_10m",
)
SCORING_ERA5_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
)
LOCAL_DAY_TIMEZONE = "Europe/Riga"
WEATHER_CUTOFF_DAYS = 2
MINIMUM_EPISODE_HISTORY_DAYS = 23
FROZEN_ERA5_EXPECTED_UNITS = {
    "temperature_2m": ("°C", "C", "celsius"),
    "relative_humidity_2m": ("%",),
    "precipitation": ("mm",),
    "soil_temperature_0_to_7cm": ("°C", "C", "celsius"),
    "soil_moisture_0_to_7cm": ("m³/m³", "m^3/m^3"),
    "shortwave_radiation": ("W/m²", "W/m2"),
    "wind_speed_10m": ("km/h",),
}


class SourceCompatibilityError(ValueError):
    """Raised when bytes cannot represent the frozen training source."""


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _hourly_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(part) for part in value)
    raise SourceCompatibilityError("hourly request variables must be a list or CSV string")


def validate_frozen_era5_request_profile(
    request_profile: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Attest that a stored request used the exact frozen ERA5 source profile.

    Open-Meteo response JSON does not identify whether the caller requested
    ERA5, ERA5-Land, or Best Match.  Therefore response bytes alone are not
    sufficient provenance: the immutable request parameters must accompany
    them and pass this check.
    """
    if request_profile is None:
        raise SourceCompatibilityError(
            "stored request profile is required; response bytes do not attest models=era5"
        )
    profile = dict(request_profile)
    endpoint = profile.get("endpoint", profile.get("url"))
    if endpoint is None:
        raise SourceCompatibilityError("stored request endpoint/url is required")
    if endpoint != FROZEN_ERA5_ENDPOINT:
        raise SourceCompatibilityError(
            f"frozen profile requires archive endpoint, got {endpoint!r}"
        )
    if profile.get("models") != "era5":
        raise SourceCompatibilityError(
            f"frozen profile requires models='era5', got {profile.get('models')!r}"
        )
    variables = _hourly_names(profile.get("hourly"))
    if variables != FROZEN_ERA5_VARIABLES:
        raise SourceCompatibilityError(
            "frozen profile requires the original ordered hourly variable set"
        )
    if profile.get("timezone") != "UTC":
        raise SourceCompatibilityError(
            f"frozen profile requires timezone='UTC', got {profile.get('timezone')!r}"
        )
    if profile.get("cell_selection") != "nearest":
        raise SourceCompatibilityError(
            "frozen profile requires cell_selection='nearest'"
        )
    if str(profile.get("elevation")).lower() != "nan":
        raise SourceCompatibilityError(
            "frozen profile requires elevation='nan' (statistical downscaling disabled)"
        )
    if profile.get("statistical_downscaling") not in {None, False}:
        raise SourceCompatibilityError("statistical downscaling must be false")
    if profile.get("profile_id") not in {None, FROZEN_ERA5_PROFILE_ID}:
        raise SourceCompatibilityError("request profile_id does not match frozen ERA5")
    try:
        start_date = pd.Timestamp(profile["start_date"]).normalize()
        end_date = pd.Timestamp(profile["end_date"]).normalize()
    except (KeyError, TypeError, ValueError) as exc:
        raise SourceCompatibilityError(
            "stored request requires valid start_date and end_date"
        ) from exc
    if start_date > end_date:
        raise SourceCompatibilityError("request start_date is after end_date")
    return {
        "endpoint": FROZEN_ERA5_ENDPOINT,
        "models": "era5",
        "hourly": list(FROZEN_ERA5_VARIABLES),
        "timezone": "UTC",
        "cell_selection": "nearest",
        "elevation": "nan",
        "statistical_downscaling": False,
        "profile_id": FROZEN_ERA5_PROFILE_ID,
    }


def build_frozen_era5_archive_params(
    latitude: float,
    longitude: float,
    start_date: date | str,
    end_date: date | str,
) -> dict[str, Any]:
    """Return the exact request profile used to create frozen ERA5 weather.

    This function builds parameters only.  It deliberately performs no network
    request and does not claim that the t-2 data are available.
    """
    return {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": ",".join(FROZEN_ERA5_VARIABLES),
        "models": "era5",
        "timezone": "UTC",
        "cell_selection": "nearest",
        "elevation": "nan",
    }


def sanitised_request_contract(
    *,
    start_date: date | str,
    end_date: date | str,
    location_ref: str,
) -> dict[str, Any]:
    """Describe a request without putting protected coordinates in a log."""
    return {
        "endpoint": FROZEN_ERA5_ENDPOINT,
        "location_ref": str(location_ref),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": list(FROZEN_ERA5_VARIABLES),
        "models": "era5",
        "timezone": "UTC",
        "cell_selection": "nearest",
        "elevation": "nan",
        "statistical_downscaling": False,
        "profile_id": FROZEN_ERA5_PROFILE_ID,
    }


def _load_payload(payload: bytes | str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        result = dict(payload)
    else:
        try:
            raw = payload.decode("utf-8") if isinstance(payload, bytes) else payload
            result = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SourceCompatibilityError("weather response is not valid UTF-8 JSON") from exc
    if not isinstance(result, dict):
        raise SourceCompatibilityError("weather response root must be an object")
    return result


def _expected_local_hours(local_day: pd.Timestamp) -> int:
    day = pd.Timestamp(local_day).normalize()
    start = day.tz_localize(ZoneInfo(LOCAL_DAY_TIMEZONE)).tz_convert("UTC")
    end = (day + pd.Timedelta(days=1)).tz_localize(
        ZoneInfo(LOCAL_DAY_TIMEZONE)
    ).tz_convert("UTC")
    return int((end - start).total_seconds() // 3600)


def parse_frozen_era5_hourly(
    payload: bytes | str | Mapping[str, Any],
    *,
    weather_cell: str,
    request_profile: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse exact ERA5 hourly bytes into the frozen Europe/Riga daily schema."""
    attested_profile = validate_frozen_era5_request_profile(request_profile)
    source = _load_payload(payload)
    timezone_name = str(source.get("timezone") or "")
    if timezone_name not in {"UTC", "GMT"}:
        raise SourceCompatibilityError(
            f"frozen profile requires UTC response timestamps, got {timezone_name!r}"
        )
    if source.get("utc_offset_seconds") not in {None, 0}:
        raise SourceCompatibilityError("frozen profile requires zero UTC offset")
    if source.get("timezone_abbreviation") not in {None, "UTC", "GMT"}:
        raise SourceCompatibilityError("response timezone abbreviation is not UTC/GMT")
    hourly = source.get("hourly")
    units = source.get("hourly_units")
    if not isinstance(hourly, dict) or not isinstance(units, dict):
        raise SourceCompatibilityError("response must contain hourly and hourly_units objects")
    required = ("time", *FROZEN_ERA5_VARIABLES)
    missing = set(required).difference(hourly)
    if missing:
        raise SourceCompatibilityError(f"frozen ERA5 response is missing {sorted(missing)}")
    lengths = {key: len(hourly[key]) for key in required if isinstance(hourly[key], list)}
    if set(lengths) != set(required) or len(set(lengths.values())) != 1:
        raise SourceCompatibilityError("hourly arrays have incompatible types or lengths")
    for name, accepted_units in FROZEN_ERA5_EXPECTED_UNITS.items():
        actual = str(units.get(name))
        if actual not in accepted_units:
            raise SourceCompatibilityError(
                f"unexpected unit for {name}: {actual!r}; expected {accepted_units[0]}"
            )

    frame = pd.DataFrame({key: hourly[key] for key in required})
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    if frame["time"].isna().any() or frame["time"].duplicated().any():
        raise SourceCompatibilityError("hourly timestamps must be unique valid UTC instants")
    if not frame["time"].is_monotonic_increasing:
        raise SourceCompatibilityError("hourly timestamps must be sorted")
    for name in FROZEN_ERA5_VARIABLES:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["temperature_2m"] = frame["temperature_2m"].where(
        frame["temperature_2m"].between(-60, 60)
    )
    frame["relative_humidity_2m"] = frame["relative_humidity_2m"].where(
        frame["relative_humidity_2m"].between(0, 100)
    )
    frame["precipitation"] = frame["precipitation"].where(
        frame["precipitation"].ge(0)
    )
    frame["local_time"] = frame["time"].dt.tz_convert(LOCAL_DAY_TIMEZONE)
    frame["date"] = frame["local_time"].dt.tz_localize(None).dt.normalize()

    records: list[dict[str, Any]] = []
    for local_day, group in frame.groupby("date", sort=True):
        expected_hours = _expected_local_hours(local_day)
        t = group["temperature_2m"]
        rh = group["relative_humidity_2m"]
        rain = group["precipitation"]
        valid_t = int(t.notna().sum())
        valid_rh = int(rh.notna().sum())
        valid_rain = int(rain.notna().sum())
        high_humidity_hours = int(rh.ge(90).sum())
        temperature_status = (
            "fail"
            if t.min() < 10
            else "pass"
            if valid_t == expected_hours
            else "indeterminate"
        )
        humidity_status = (
            "pass"
            if high_humidity_hours >= 6
            else "fail"
            if high_humidity_hours + expected_hours - valid_rh < 6
            else "indeterminate"
        )
        statuses = {temperature_status, humidity_status}
        day_status = (
            "fail"
            if "fail" in statuses
            else "indeterminate"
            if "indeterminate" in statuses
            else "pass"
        )
        complete = min(valid_t, valid_rh, valid_rain) == expected_hours
        records.append(
            {
                "weather_cell": str(weather_cell),
                "date": pd.Timestamp(local_day),
                "expected_hours": expected_hours,
                "temperature_valid_hours": valid_t,
                "humidity_valid_hours": valid_rh,
                "precipitation_valid_hours": valid_rain,
                "minimum_temperature_c": float(t.min()) if valid_t else np.nan,
                "maximum_temperature_c": float(t.max()) if valid_t else np.nan,
                "high_humidity_hours": high_humidity_hours,
                "day_status": day_status,
                "temperature_mean_c": float(t.mean()) if complete else np.nan,
                "relative_humidity_mean_pct": float(rh.mean()) if complete else np.nan,
                "precipitation_sum_mm": float(rain.sum()) if complete else np.nan,
                "accepted": bool(complete),
                "soil_temperature_c": float(group["soil_temperature_0_to_7cm"].mean()),
                "soil_moisture": float(group["soil_moisture_0_to_7cm"].mean()),
                "radiation_mean_wm2": float(group["shortwave_radiation"].mean()),
                "wind_mean_kmh": float(group["wind_speed_10m"].mean()),
            }
        )
    daily = pd.DataFrame(records)
    metadata = {
        "profile_id": FROZEN_ERA5_PROFILE_ID,
        "request_profile_attested": True,
        "request_profile_sha256": _canonical_sha256(attested_profile),
        "provider": "Open-Meteo Historical Weather API",
        "product": "ERA5",
        "model_parameter": "era5",
        "returned_timezone": timezone_name,
        "local_day_timezone": LOCAL_DAY_TIMEZONE,
        "returned_latitude": source.get("latitude"),
        "returned_longitude": source.get("longitude"),
        "grid_elevation_m": source.get("elevation"),
        "statistical_downscaling": False,
        "hourly_units": {name: units.get(name) for name in FROZEN_ERA5_VARIABLES},
        "valid_time_min": frame["time"].min().isoformat() if len(frame) else None,
        "valid_time_max": frame["time"].max().isoformat() if len(frame) else None,
        "local_date_min": str(daily["date"].min().date()) if len(daily) else None,
        "local_date_max": str(daily["date"].max().date()) if len(daily) else None,
        "complete_local_days": int(daily["accepted"].sum()) if len(daily) else 0,
        "local_days": int(len(daily)),
    }
    return daily, metadata


def build_episode_feature_snapshot(
    daily: pd.DataFrame,
    *,
    weather_cell: str,
    issue_local_date: date | str | pd.Timestamp,
) -> dict[str, Any]:
    """Build one exact past-only feature row at the unchanged t-2 cutoff."""
    issue_date = pd.Timestamp(issue_local_date)
    if issue_date.tzinfo is not None:
        issue_date = issue_date.tz_convert(LOCAL_DAY_TIMEZONE).tz_localize(None)
    issue_date = issue_date.normalize()
    cutoff = issue_date - pd.Timedelta(days=WEATHER_CUTOFF_DAYS)
    required_columns = {
        "weather_cell",
        "date",
        "accepted",
        "day_status",
        "minimum_temperature_c",
        "high_humidity_hours",
        "temperature_mean_c",
        "relative_humidity_mean_pct",
        "precipitation_sum_mm",
    }
    missing_columns = required_columns.difference(daily.columns)
    if missing_columns:
        raise SourceCompatibilityError(
            f"daily ERA5 data are missing {sorted(missing_columns)}"
        )
    source = daily.loc[daily["weather_cell"].eq(str(weather_cell))].copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    if getattr(source["date"].dt, "tz", None) is not None:
        source["date"] = source["date"].dt.tz_convert(LOCAL_DAY_TIMEZONE).dt.tz_localize(None)
    if source["date"].isna().any():
        raise SourceCompatibilityError("daily ERA5 dates must be valid")
    if source["date"].duplicated().any():
        raise SourceCompatibilityError("daily ERA5 dates must be unique per weather cell")
    source = source.loc[source["date"].le(cutoff)].sort_values("date")
    if source.empty or pd.Timestamp(source["date"].max()) < cutoff:
        return {
            "status": "late_or_missing_required_cutoff",
            "required_last_local_date": str(cutoff.date()),
            "available_last_local_date": (
                str(pd.Timestamp(source["date"].max()).date()) if len(source) else None
            ),
            "features": None,
            "complete": False,
        }
    episodes = _episode_features(source)
    row = episodes.loc[
        episodes["weather_cell"].eq(str(weather_cell))
        & pd.to_datetime(episodes["cutoff_date"]).eq(cutoff)
    ]
    if len(row) != 1:
        raise SourceCompatibilityError("expected exactly one episode feature row at cutoff")
    feature_values = {
        name: (None if pd.isna(row.iloc[0][name]) else float(row.iloc[0][name]))
        for name in EPISODE_FEATURES
    }
    complete = all(value is not None for value in feature_values.values())
    return {
        "status": "available" if complete else "insufficient_complete_history",
        "required_last_local_date": str(cutoff.date()),
        "available_last_local_date": str(pd.Timestamp(source["date"].max()).date()),
        "minimum_history_days": MINIMUM_EPISODE_HISTORY_DAYS,
        "history_local_date_min": str(pd.Timestamp(source["date"].min()).date()),
        "history_local_date_max": str(pd.Timestamp(source["date"].max()).date()),
        "features": feature_values,
        "complete": complete,
    }


def assert_source_available_as_of(
    *,
    retrieval_completed_at: str,
    first_seen_at: str,
    ingested_at: str,
    decision_at: str,
) -> None:
    """Reject bytes that arrived after the actual decision time."""
    decision = pd.Timestamp(decision_at)
    if decision.tzinfo is None:
        raise SourceCompatibilityError("decision_at must be timezone-aware")
    for name, raw in {
        "retrieval_completed_at": retrieval_completed_at,
        "first_seen_at": first_seen_at,
        "ingested_at": ingested_at,
    }.items():
        value = pd.Timestamp(raw)
        if value.tzinfo is None:
            raise SourceCompatibilityError(f"{name} must be timezone-aware")
        if value > decision:
            raise SourceCompatibilityError(
                f"{name} is after actual decision time; late bytes cannot be used"
            )


def validate_source_timing(
    *,
    run_initialization_at: str | None,
    provider_published_at: str | None,
    retrieval_started_at: str,
    retrieval_completed_at: str,
    first_seen_at: str,
    ingested_at: str,
) -> dict[str, str | None]:
    """Validate and preserve distinct operational time meanings.

    In particular, a forecast run initialization is never copied into the
    provider publication field.  Unknown publication time remains null.
    """
    raw_values = {
        "run_initialization_at": run_initialization_at,
        "provider_published_at": provider_published_at,
        "retrieval_started_at": retrieval_started_at,
        "retrieval_completed_at": retrieval_completed_at,
        "first_seen_at": first_seen_at,
        "ingested_at": ingested_at,
    }
    parsed: dict[str, pd.Timestamp | None] = {}
    for name, raw in raw_values.items():
        if raw is None:
            parsed[name] = None
            continue
        value = pd.Timestamp(raw)
        if value.tzinfo is None:
            raise SourceCompatibilityError(f"{name} must be timezone-aware")
        parsed[name] = value.tz_convert("UTC")

    started = parsed["retrieval_started_at"]
    completed = parsed["retrieval_completed_at"]
    first_seen = parsed["first_seen_at"]
    ingested = parsed["ingested_at"]
    assert started is not None and completed is not None
    assert first_seen is not None and ingested is not None
    if started > completed:
        raise SourceCompatibilityError("retrieval_started_at is after retrieval_completed_at")
    if completed > first_seen:
        raise SourceCompatibilityError("first_seen_at is before retrieval completed")
    if first_seen > ingested:
        raise SourceCompatibilityError("ingested_at is before first_seen_at")
    published = parsed["provider_published_at"]
    if published is not None and published > completed:
        raise SourceCompatibilityError(
            "confirmed provider publication cannot be after retrieval completion"
        )
    return {
        name: value.isoformat() if value is not None else None
        for name, value in parsed.items()
    }


def source_profile_contract() -> dict[str, Any]:
    return {
        "profile_id": FROZEN_ERA5_PROFILE_ID,
        "provider": "Open-Meteo Historical Weather API",
        "endpoint": FROZEN_ERA5_ENDPOINT,
        "model_parameter": "era5",
        "product": "ERA5",
        "native_grid": "0.25 degree approximately 25 km",
        "cell_selection": "nearest",
        "elevation_parameter": "nan",
        "statistical_downscaling": False,
        "response_timezone": "UTC",
        "aggregation_timezone": LOCAL_DAY_TIMEZONE,
        "training_request_variables": list(FROZEN_ERA5_VARIABLES),
        "scoring_variables": list(SCORING_ERA5_VARIABLES),
        "scoring_units": {
            "temperature_2m": "degree_C",
            "relative_humidity_2m": "percent",
            "precipitation": "mm",
        },
        "temporal_resolution": "hourly",
        "required_last_local_date_rule": "issue_local_date_minus_2_calendar_days",
        "minimum_complete_local_history_days": MINIMUM_EPISODE_HISTORY_DAYS,
        "time_semantics": {
            "run_initialization_at": "forecast run initialization when known; not publication",
            "provider_published_at": "confirmed publication time or null; never inferred from initialization",
            "retrieval_completed_at": "actual end of byte retrieval",
            "first_seen_at": "first time these exact bytes were visible to the system",
            "ingested_at": "time these bytes entered immutable storage",
            "valid_time": "time represented by the weather value",
        },
        "substitutions_forbidden": [
            "best_match",
            "era5_seamless",
            "era5_land",
            "ecmwf_ifs",
            "forecast_api",
            "historical_forecast_api",
            "changed_cell_selection",
            "statistical_downscaling",
        ],
    }


__all__ = [
    "FROZEN_ERA5_ENDPOINT",
    "FROZEN_ERA5_EXPECTED_UNITS",
    "FROZEN_ERA5_PROFILE_ID",
    "FROZEN_ERA5_VARIABLES",
    "LOCAL_DAY_TIMEZONE",
    "MINIMUM_EPISODE_HISTORY_DAYS",
    "SCORING_ERA5_VARIABLES",
    "SourceCompatibilityError",
    "WEATHER_CUTOFF_DAYS",
    "assert_source_available_as_of",
    "build_episode_feature_snapshot",
    "build_frozen_era5_archive_params",
    "parse_frozen_era5_hourly",
    "sanitised_request_contract",
    "source_profile_contract",
    "validate_frozen_era5_request_profile",
    "validate_source_timing",
]
