"""Local-day aggregation of hourly Open-Meteo data."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def aggregate_hourly_temperature(
    hourly: pd.DataFrame,
    timezone: str,
    method: str = "hourly_mean",
    minimum_valid_hours: int = 20,
    allow_incomplete_days: bool = False,
) -> pd.DataFrame:
    """Aggregate timestamps into local days without assuming 24 hours per day."""

    if method not in {"hourly_mean", "min_max_mean"}:
        raise ValueError("method must be 'hourly_mean' or 'min_max_mean'")
    if not 1 <= minimum_valid_hours <= 25:
        raise ValueError("minimum_valid_hours must be between 1 and 25")
    if not {"time", "temperature_2m"}.issubset(hourly.columns):
        raise ValueError("Hourly data must contain time and temperature_2m")

    zone = ZoneInfo(timezone)
    frame = hourly[["time", "temperature_2m"]].copy()
    parsed = pd.to_datetime(frame["time"], errors="raise")
    if parsed.dt.tz is None:
        # Open-Meteo returns local wall-clock values when timezone=auto.
        localized = parsed.dt.tz_localize(zone, ambiguous="infer", nonexistent="shift_forward")
    else:
        localized = parsed.dt.tz_convert(zone)
    frame["local_time"] = localized
    frame["date"] = localized.dt.date
    frame["temperature_2m"] = pd.to_numeric(frame["temperature_2m"], errors="coerce")

    rows: list[dict[str, object]] = []
    for local_date, group in frame.groupby("date", sort=True):
        valid = group["temperature_2m"].dropna()
        local_midnight = pd.Timestamp(local_date).tz_localize(zone)
        next_midnight = (pd.Timestamp(local_date) + pd.Timedelta(days=1)).tz_localize(zone)
        expected_hours = int((next_midnight.tz_convert("UTC") - local_midnight.tz_convert("UTC")).total_seconds() / 3600)
        valid_hours = int(valid.size)
        coverage = valid_hours / expected_hours if expected_hours else 0.0
        sufficient = valid_hours >= minimum_valid_hours
        accepted = bool(valid_hours and (sufficient or allow_incomplete_days))
        if method == "hourly_mean":
            mean_value = float(valid.mean()) if accepted else np.nan
        else:
            mean_value = float((valid.min() + valid.max()) / 2.0) if accepted else np.nan
        rows.append(
            {
                "date": local_date,
                "temperature_mean_c": mean_value,
                "valid_hour_count": valid_hours,
                "expected_hour_count": expected_hours,
                "coverage_fraction": coverage,
                "accepted": accepted,
                "completeness_warning": "" if sufficient else "insufficient_valid_hours",
                "daily_temperature_method": method,
            }
        )
    return pd.DataFrame(rows)
