"""Past-only Colorado potato beetle egg-hatch predictions for external data.

The fixed UW parameters are 52 F lower / 86 F upper temperature thresholds
and 120 F degree-days after first observed eggs (66 2/3 C degree-days).
This adapter uses an explicit *hourly clipped* approximation, not the
published single-sine calculation: clip(T - lower, 0, upper - lower) / 24.

Weather through issue_date - lag_days is retrospectively treated as known.
Unavailable weather after that local date is never used. Its replacement is
one fixed rate: the mean DD of the last seven complete local calendar days.
This is a declared persistence forecast, not a historical weather forecast.
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


LOWER_THRESHOLD_C = (52.0 - 32.0) * 5.0 / 9.0
UPPER_THRESHOLD_C = (86.0 - 32.0) * 5.0 / 9.0
HATCH_THRESHOLD_C_DAYS = 120.0 * 5.0 / 9.0
MODEL_CODE = "UW_CPB_hourly_clipped_persistence"
MODEL_VERSION = "1"
_DD_TOLERANCE = 1e-10
_OUTPUT_COLUMNS = [
    "issue_date", "issued_at", "feature_cutoff_date", "known_cumulative_dd",
    "persistence_daily_dd", "predicted_event_date", "score", "score_status",
]


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _local_date(value, zone: ZoneInfo) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(zone).tz_localize(None)
    return timestamp.normalize()


def _issue_calendar(issue_dates, zone: ZoneInfo) -> pd.DatetimeIndex:
    issues = pd.DatetimeIndex([_local_date(value, zone) for value in issue_dates])
    if issues.hasnans:
        raise ValueError("issue_dates cannot contain missing dates")
    if issues.has_duplicates:
        raise ValueError("issue_dates must be unique local calendar dates")
    return issues


def _daily_degree_days(
    hourly: pd.DataFrame, zone: ZoneInfo, first_day: pd.Timestamp, last_day: pd.Timestamp,
) -> pd.Series:
    """Sum only complete eligible local days, including 23/25-hour DST days."""
    required = {"valid_time_utc", "temperature_2m"}
    if not required.issubset(hourly):
        raise ValueError(f"Missing hourly columns: {sorted(required - set(hourly))}")
    times = pd.to_datetime(hourly["valid_time_utc"], utc=True, errors="raise")
    if times.isna().any():
        raise ValueError("Hourly valid times cannot be missing")
    start = first_day.tz_localize(zone).tz_convert("UTC")
    end = (last_day + pd.Timedelta(days=1)).tz_localize(zone).tz_convert("UTC")
    eligible = times.ge(start) & times.lt(end)
    # Select dates before touching temperature values, so future values and
    # future grid metadata cannot affect the result or input validation.
    frame = hourly.loc[eligible].copy()
    frame["time"] = times.loc[eligible].to_numpy()
    if frame["time"].duplicated().any():
        raise ValueError("Duplicate hourly timestamps would change evidence")
    if (frame.time.dt.minute.ne(0) | frame.time.dt.second.ne(0) |
            frame.time.dt.microsecond.ne(0) | frame.time.dt.nanosecond.ne(0)).any():
        raise ValueError("CPB hourly input must be aligned to whole UTC hours")
    if {"provider_latitude", "provider_longitude"}.issubset(frame):
        if len(frame[["provider_latitude", "provider_longitude"]].drop_duplicates()) > 1:
            raise ValueError("One CPB call cannot combine multiple provider grid locations")
    temperatures = pd.to_numeric(frame["temperature_2m"], errors="coerce").astype(float)
    temperatures = temperatures.where(np.isfinite(temperatures))
    frame["hourly_dd"] = temperatures.sub(LOWER_THRESHOLD_C).clip(
        lower=0.0, upper=UPPER_THRESHOLD_C - LOWER_THRESHOLD_C,
    ) / 24.0
    frame["local_date"] = frame.time.dt.tz_convert(zone).dt.tz_localize(None).dt.normalize()
    calendar = pd.date_range(first_day, last_day, freq="D")
    grouped = frame.groupby("local_date")["hourly_dd"].agg(["sum", "count", "size"]).reindex(calendar)
    expected_hours = pd.Series([
        len(pd.date_range(
            day.tz_localize(zone).tz_convert("UTC").ceil("h"),
            (day + pd.Timedelta(days=1)).tz_localize(zone).tz_convert("UTC"),
            freq="h", inclusive="left",
        )) for day in calendar
    ], index=calendar)
    complete = grouped["count"].eq(expected_hours) & grouped["size"].eq(expected_hours)
    return grouped["sum"].where(complete).rename("daily_dd")


def build_cpb_predictions(
    hourly: pd.DataFrame,
    issue_dates,
    biofix_date,
    timezone: str = "Europe/Berlin",
    lag_days: int = 5,
    lookback_days: int = 7,
) -> pd.DataFrame:
    """Predict first egg hatch using only information available at each issue.

    Dates are local calendar dates; issuance is at 08:00. An observed first
    egg date (biofix) becomes usable the following day. DD accumulation
    includes the full biofix date and the full cutoff date. A missing hour
    makes its day's DD unknown, never zero. The persistence rate requires
    every day in cutoff-lookback_days+1 through cutoff to be complete; it
    never searches farther back to replace missing days.

    If the threshold is already reached in complete known weather, return
    its historical crossing date and score zero. Otherwise forecast all
    days after the cutoff, including the availability gap before issuance,
    at the fixed persistence rate. When cutoff precedes biofix, accumulation
    starts at zero on biofix. The predicted date is the first calendar day
    whose forecast cumulative DD reaches the threshold. Score is 1 exactly
    when that date lies in [issue_date + 3, issue_date + 10], else 0; unknown
    biofix, incomplete required weather or a zero forecast rate abstain.

    The lag is a protocol assumption, not a recovered ERA5 publication time.
    No outcomes, future visits or realized weather after a cutoff are read.
    """
    lag_days = _positive_integer(lag_days, "lag_days")
    lookback_days = _positive_integer(lookback_days, "lookback_days")
    zone = ZoneInfo(timezone)
    issues = _issue_calendar(issue_dates, zone)
    if issues.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    biofix = pd.NaT if biofix_date is None else _local_date(biofix_date, zone)
    cutoffs = issues - pd.Timedelta(days=lag_days)
    eligible_issues = issues[issues > biofix] if pd.notna(biofix) else pd.DatetimeIndex([])
    daily = None
    if not eligible_issues.empty:
        eligible_cutoffs = eligible_issues - pd.Timedelta(days=lag_days)
        first_day = min(biofix, eligible_cutoffs.min() - pd.Timedelta(days=lookback_days - 1))
        daily = _daily_degree_days(hourly, zone, first_day, eligible_cutoffs.max())
    rows = []
    for issue, cutoff in zip(issues, cutoffs):
        row = {
            "issue_date": issue,
            "issued_at": (issue + pd.Timedelta(hours=8)).tz_localize(zone).isoformat(),
            "feature_cutoff_date": cutoff,
            "known_cumulative_dd": np.nan,
            "persistence_daily_dd": np.nan,
            "predicted_event_date": pd.NaT,
            "score": np.nan,
            "score_status": "abstained_biofix_missing",
        }
        rows.append(row)
        if pd.isna(biofix):
            continue
        if issue <= biofix:
            row["score_status"] = "abstained_biofix_not_yet_available"
            continue
        history = daily.loc[biofix:cutoff] if cutoff >= biofix else pd.Series(dtype=float)
        rate_days = daily.loc[cutoff - pd.Timedelta(days=lookback_days - 1):cutoff]
        rate_complete = len(rate_days) == lookback_days and rate_days.notna().all()
        if rate_complete:
            row["persistence_daily_dd"] = float(rate_days.mean())
        if history.isna().any():
            row["score_status"] = "abstained_missing_cumulative_weather"
            continue
        known_dd = float(history.sum())
        row["known_cumulative_dd"] = known_dd
        crossed = history.cumsum().ge(HATCH_THRESHOLD_C_DAYS - _DD_TOLERANCE)
        if crossed.any():
            row.update(predicted_event_date=crossed.index[crossed][0], score=0.0, score_status="computed")
            continue
        if not rate_complete:
            row["score_status"] = "abstained_missing_persistence_weather"
            continue
        rate = row["persistence_daily_dd"]
        if rate <= 0:
            row["score_status"] = "abstained_zero_persistence_rate"
            continue
        first_forecast_day = max(biofix, cutoff + pd.Timedelta(days=1))
        forecast_days = int(np.ceil(max((HATCH_THRESHOLD_C_DAYS - known_dd) / rate - 1e-12, 0.0)))
        try:
            predicted = first_forecast_day + pd.Timedelta(days=max(forecast_days - 1, 0))
        except (OverflowError, pd.errors.OutOfBoundsDatetime, pd.errors.OutOfBoundsTimedelta):
            row["score_status"] = "abstained_prediction_out_of_range"
            continue
        row.update(
            predicted_event_date=predicted,
            score=float(issue + pd.Timedelta(days=3) <= predicted <= issue + pd.Timedelta(days=10)),
            score_status="computed",
        )
    return pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
