"""Past-only adaptation of the Luke/IPM Decisions carrot-fly flight model.

The published parameters are 260 Celsius degree-days above 5 C from March 1,
using daily air temperature. This implementation explicitly takes the mean of
all available hourly temperatures in a *complete local day*, then applies
max(mean_temperature - 5, 0). It is not hourly-clipped thermal excess and does
not add an upper threshold unsupported by the factsheet.

Source: https://www.ipmdecisions.net/media/ev1e2wz2/
ipm_factsheet-carrot-rust-fly_v0001_print.pdf

The formula forecasts the start of first-generation flight. A first adult
trap registration is an observation proxy, not an observed emergence date,
egg-laying date, population count or damage measurement.
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


BASE_TEMPERATURE_C = 5.0
FLIGHT_THRESHOLD_C_DAYS = 260.0
MODEL_CODE = "Luke_carrot_fly_DD260_persistence7"
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


def _daily_degree_days(
    hourly: pd.DataFrame,
    times: pd.Series,
    zone: ZoneInfo,
    first_day: pd.Timestamp,
    last_day: pd.Timestamp,
) -> pd.Series:
    """Use only this decision's eligible weather; require every local hour."""
    start = first_day.tz_localize(zone).tz_convert("UTC")
    end = (last_day + pd.Timedelta(days=1)).tz_localize(zone).tz_convert("UTC")
    eligible = times.ge(start) & times.lt(end)
    # Filter before temperature coercion or metadata validation. In particular,
    # metadata for a later issue cannot invalidate this earlier decision.
    frame = hourly.loc[eligible].copy()
    frame["time"] = times.loc[eligible].array
    if frame.time.duplicated().any():
        raise ValueError("Duplicate hourly timestamps")
    if (frame.time.dt.minute.ne(0) | frame.time.dt.second.ne(0)
            | frame.time.dt.microsecond.ne(0) | frame.time.dt.nanosecond.ne(0)).any():
        raise ValueError("Hourly timestamps must be aligned to whole UTC hours")
    if {"provider_latitude", "provider_longitude"}.issubset(frame):
        coordinates = frame[["provider_latitude", "provider_longitude"]]
        # A caller can append optional metadata only for unavailable future
        # rows. Entirely absent metadata in the selected history is equivalent
        # to not having these optional columns at all.
        if not coordinates.isna().all().all() and (
                coordinates.isna().any().any() or len(coordinates.drop_duplicates()) > 1):
            raise ValueError("Multiple or missing provider grid locations in known weather")
    temperatures = pd.to_numeric(frame.temperature_2m, errors="coerce").astype(float)
    frame["temperature"] = temperatures.where(np.isfinite(temperatures))
    frame["local_date"] = frame.time.dt.tz_convert(zone).dt.tz_localize(None).dt.normalize()
    calendar = pd.date_range(first_day, last_day, freq="D")
    grouped = frame.groupby("local_date").temperature.agg(["mean", "count", "size"]).reindex(calendar)
    # Use 23/25 actual local hours on DST changes, but still one mean/day:
    # constant temperature has the same degree-day value on all three lengths.
    boundaries = pd.date_range(first_day, last_day + pd.Timedelta(days=1), freq="D").tz_localize(zone).tz_convert("UTC")
    expected = pd.Series(np.diff(boundaries.asi8) / (3600 * 1e9), index=calendar)
    complete = grouped["count"].eq(expected) & grouped["size"].eq(expected)
    return grouped["mean"].sub(BASE_TEMPERATURE_C).clip(lower=0).where(complete).rename("daily_dd")


def build_carrot_predictions(
    hourly: pd.DataFrame,
    issue_dates,
    season: int,
    timezone: str = "Europe/Copenhagen",
    lag_days: int = 5,
    lookback_days: int = 7,
) -> pd.DataFrame:
    """Forecast the DD260 crossing from strictly past available temperatures.

    Each issue is at 08:00 local time and uses complete local days only through
    issue_date - lag_days inclusive. Availability lag is a declared assumption,
    not a recovered historical ERA5 publication timestamp. DD accumulation
    includes March 1 and the cutoff day; February temperatures do not contribute
    to cumulative DD (but can be used for persistence if the cutoff is early).

    After the cutoff, including the unavailable recent-weather gap, all daily
    DD values are forecast at one constant rate: the mean of the last seven
    consecutive available complete local days. No actual weather after the
    issue's cutoff is read. If the threshold is already crossed in complete
    known history, return that past date and score zero even when the recent
    rate is zero/missing. Otherwise missing cumulative/rate weather, a zero
    rate, or invalid known hourly structure cause explicit abstention.

    The score is 1 iff the predicted crossing date is in the inclusive window
    issue+3 through issue+10, otherwise 0; it is not a calibrated probability.
    No observation, future trap visit or observed outcome is an input.
    """
    lag_days = _positive_integer(lag_days, "lag_days")
    lookback_days = _positive_integer(lookback_days, "lookback_days")
    season = _positive_integer(season, "season")
    if not 1678 <= season <= 2260:
        raise ValueError("season must be within pandas' supported calendar years")
    zone = ZoneInfo(timezone)
    issues = pd.DatetimeIndex([_local_date(value, zone) for value in issue_dates])
    if issues.hasnans:
        raise ValueError("issue_dates cannot contain missing dates")
    if issues.has_duplicates:
        raise ValueError("issue_dates must be unique local calendar dates")
    if issues.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    if not (issues.year == season).all():
        raise ValueError("All issue dates must belong to the declared local season")
    required = {"valid_time_utc", "temperature_2m"}
    if not required.issubset(hourly):
        raise ValueError(f"Missing hourly columns: {sorted(required - set(hourly))}")
    times = pd.to_datetime(hourly.valid_time_utc, utc=True, errors="raise")
    if times.isna().any():
        raise ValueError("Hourly valid times cannot be missing")
    season_start = pd.Timestamp(year=season, month=3, day=1)
    rows = []
    for issue in issues:
        cutoff = issue - pd.Timedelta(days=lag_days)
        row = {
            "issue_date": issue,
            "issued_at": (issue + pd.Timedelta(hours=8)).tz_localize(zone).isoformat(),
            "feature_cutoff_date": cutoff,
            "known_cumulative_dd": np.nan,
            "persistence_daily_dd": np.nan,
            "predicted_event_date": pd.NaT,
            "score": np.nan,
            "score_status": "abstained_missing_cumulative_weather",
        }
        rows.append(row)
        first_day = min(season_start, cutoff - pd.Timedelta(days=lookback_days - 1))
        try:
            daily = _daily_degree_days(hourly, times, zone, first_day, cutoff)
        except ValueError:
            row["score_status"] = "abstained_invalid_hourly_input"
            continue
        history = daily.loc[season_start:cutoff] if cutoff >= season_start else pd.Series(dtype=float)
        rate_days = daily.loc[cutoff - pd.Timedelta(days=lookback_days - 1):cutoff]
        rate_complete = len(rate_days) == lookback_days and rate_days.notna().all()
        if rate_complete:
            row["persistence_daily_dd"] = float(rate_days.mean())
        if history.isna().any():
            continue
        known_dd = float(history.sum())
        row["known_cumulative_dd"] = known_dd
        crossed = history.cumsum().ge(FLIGHT_THRESHOLD_C_DAYS - _DD_TOLERANCE)
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
        first_forecast_day = max(season_start, cutoff + pd.Timedelta(days=1))
        forecast_days = np.ceil(max((FLIGHT_THRESHOLD_C_DAYS - known_dd) / rate - 1e-12, 0.0))
        try:
            predicted = first_forecast_day + pd.Timedelta(days=max(int(forecast_days) - 1, 0))
        except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime, pd.errors.OutOfBoundsTimedelta):
            row["score_status"] = "abstained_prediction_out_of_range"
            continue
        row.update(
            predicted_event_date=predicted,
            score=float(issue + pd.Timedelta(days=3) <= predicted <= issue + pd.Timedelta(days=10)),
            score_status="computed",
        )
    return pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
