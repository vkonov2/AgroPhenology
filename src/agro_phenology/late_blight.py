"""Weather-only late-blight indicators from the supplied Hutton and Polyakov specs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HuttonConfig:
    """Versioned thresholds for the Hutton Criteria weather indicator."""

    minimum_temperature_c: float = 10.0
    relative_humidity_threshold_pct: float = 90.0
    required_high_humidity_hours: float = 6.0
    interval_hours: float = 1.0
    minimum_physical_temperature_c: float = -60.0
    maximum_physical_temperature_c: float = 60.0
    and_indeterminate_resolution: str = "fail_dominates_indeterminate"
    model_version: str = "1.0.0"


@dataclass(frozen=True)
class PolyakovConfig:
    """Thresholds for the simplified 10-day Polyakov weather indicator."""

    window_days: int = 10
    minimum_window_temperature_c: float = 13.0
    maximum_window_temperature_c: float = 20.0
    minimum_window_relative_humidity_pct: float = 75.0
    minimum_window_precipitation_mm: float = 20.0
    persistence_days: int = 6
    manifestation_lag_days: tuple[int, int] = (6, 8)
    minimum_valid_hours_per_day: int = 20
    require_complete_daily_coverage: bool = True
    minimum_physical_temperature_c: float = -60.0
    maximum_physical_temperature_c: float = 60.0
    model_version: str = "1.1.0"


def _prepare_local_hourly(hourly: pd.DataFrame, timezone: str) -> pd.DataFrame:
    required = {"time", "temperature_2m", "relative_humidity_2m"}
    missing = required - set(hourly.columns)
    if missing:
        raise ValueError(f"Missing hourly columns: {sorted(missing)}")
    frame = hourly.copy()
    parsed = pd.to_datetime(frame["time"], errors="raise")
    zone = ZoneInfo(timezone)
    if parsed.dt.tz is None:
        localized = parsed.dt.tz_localize(zone, ambiguous="infer", nonexistent="shift_forward")
    else:
        localized = parsed.dt.tz_convert(zone)
    frame["local_time"] = localized
    frame["date"] = localized.dt.date
    for column in ("temperature_2m", "relative_humidity_2m", "precipitation"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("local_time").drop_duplicates("local_time", keep="last")
    return frame


def merge_polyakov_weather_sources(
    era5_land_hourly: pd.DataFrame,
    era5_hourly: pd.DataFrame,
) -> pd.DataFrame:
    """Combine ERA5-Land T/RH with ERA5 precipitation on an identical hourly grid.

    ERA5 temperature may be present because the shared API parser requires it, but
    it is deliberately discarded here.  A grid mismatch is an error rather than a
    reason to interpolate or silently shift either source.
    """

    land_required = {"time", "temperature_2m", "relative_humidity_2m"}
    era5_required = {"time", "precipitation"}
    missing_land = land_required - set(era5_land_hourly.columns)
    missing_era5 = era5_required - set(era5_hourly.columns)
    if missing_land:
        raise ValueError(f"ERA5-Land data is missing columns: {sorted(missing_land)}")
    if missing_era5:
        raise ValueError(f"ERA5 precipitation data is missing columns: {sorted(missing_era5)}")

    land = era5_land_hourly[["time", "temperature_2m", "relative_humidity_2m"]].copy()
    precipitation = era5_hourly[["time", "precipitation"]].copy()
    for frame, source_name in ((land, "ERA5-Land"), (precipitation, "ERA5")):
        frame["time"] = pd.to_datetime(frame["time"], errors="raise")
        if frame["time"].duplicated().any():
            raise ValueError(f"{source_name} hourly grid contains duplicate timestamps")
        frame.sort_values("time", inplace=True)
        frame.reset_index(drop=True, inplace=True)

    if not land["time"].equals(precipitation["time"]):
        raise ValueError("ERA5-Land and ERA5 hourly timestamp grids do not match exactly")

    return land.merge(precipitation, on="time", how="inner", validate="one_to_one")


def _expected_hours(local_date: date, timezone: str) -> int:
    zone = ZoneInfo(timezone)
    midnight = pd.Timestamp(local_date).tz_localize(zone).tz_convert("UTC")
    next_midnight = (pd.Timestamp(local_date) + pd.Timedelta(days=1)).tz_localize(zone).tz_convert("UTC")
    return int((next_midnight - midnight).total_seconds() / 3600)


def _date_range(frame: pd.DataFrame, start_date: date | str | None, end_date: date | str | None) -> list[date]:
    if frame.empty and (start_date is None or end_date is None):
        return []
    start = pd.Timestamp(start_date).date() if start_date is not None else min(frame["date"])
    end = pd.Timestamp(end_date).date() if end_date is not None else max(frame["date"])
    if end < start:
        raise ValueError("end_date must not be earlier than start_date")
    return [value.date() for value in pd.date_range(start, end, freq="D")]


def classify_hutton_days(
    hourly: pd.DataFrame,
    timezone: str,
    config: HuttonConfig = HuttonConfig(),
    start_date: date | str | None = None,
    end_date: date | str | None = None,
) -> pd.DataFrame:
    """Classify local days as pass, fail, or indeterminate under Hutton Criteria."""

    if config.and_indeterminate_resolution != "fail_dominates_indeterminate":
        raise ValueError("Unsupported Hutton AND/indeterminate resolution rule")
    frame = _prepare_local_hourly(hourly, timezone)
    rows: list[dict[str, object]] = []
    for local_date in _date_range(frame, start_date, end_date):
        group = frame[frame["date"] == local_date]
        expected = _expected_hours(local_date, timezone)
        temperatures = group["temperature_2m"].where(
            group["temperature_2m"].between(
                config.minimum_physical_temperature_c,
                config.maximum_physical_temperature_c,
                inclusive="both",
            )
        ).dropna()
        humidities = group["relative_humidity_2m"].where(
            group["relative_humidity_2m"].between(0.0, 100.0, inclusive="both")
        ).dropna()
        temperature_hours = min(float(len(temperatures)) * config.interval_hours, float(expected))
        humidity_hours = min(float(len(humidities)) * config.interval_hours, float(expected))
        high_humidity_hours = float(
            (humidities >= config.relative_humidity_threshold_pct).sum()
        ) * config.interval_hours
        missing_humidity_hours = max(float(expected) - humidity_hours, 0.0)
        minimum_temperature = float(temperatures.min()) if not temperatures.empty else np.nan

        if not temperatures.empty and minimum_temperature < config.minimum_temperature_c:
            temperature_status = "fail"
        elif temperature_hours < expected:
            temperature_status = "indeterminate"
        else:
            temperature_status = "pass"
        if high_humidity_hours >= config.required_high_humidity_hours:
            humidity_status = "pass"
        elif high_humidity_hours + missing_humidity_hours < config.required_high_humidity_hours:
            humidity_status = "fail"
        else:
            humidity_status = "indeterminate"
        if "fail" in {temperature_status, humidity_status}:
            day_status = "fail"
        elif "indeterminate" in {temperature_status, humidity_status}:
            day_status = "indeterminate"
        else:
            day_status = "pass"
        rows.append(
            {
                "date": local_date,
                "minimum_temperature_c": minimum_temperature,
                "high_humidity_hours": high_humidity_hours,
                "temperature_valid_hours": temperature_hours,
                "humidity_valid_hours": humidity_hours,
                "expected_hours": expected,
                "data_complete": temperature_hours >= expected and humidity_hours >= expected,
                "temperature_status": temperature_status,
                "humidity_status": humidity_status,
                "day_status": day_status,
                "model": "hutton_criteria",
                "model_version": config.model_version,
            }
        )
    return pd.DataFrame(rows)


def classify_hutton_periods(
    daily: pd.DataFrame,
    config: HuttonConfig = HuttonConfig(),
) -> pd.DataFrame:
    """Classify every pair of consecutive local dates and retain indeterminate pairs."""

    if config.and_indeterminate_resolution != "fail_dominates_indeterminate":
        raise ValueError("Unsupported Hutton AND/indeterminate resolution rule")
    if daily.empty:
        return pd.DataFrame()
    ordered = daily.sort_values("date").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for index in range(1, len(ordered)):
        previous = ordered.iloc[index - 1]
        current = ordered.iloc[index]
        if current["date"] - previous["date"] != timedelta(days=1):
            continue
        statuses = {previous["day_status"], current["day_status"]}
        if statuses == {"pass"}:
            period_status = "pass"
        elif "fail" in statuses:
            period_status = "fail"
        else:
            period_status = "indeterminate"
        rows.append(
            {
                "period_start": previous["date"],
                "period_end": current["date"],
                "first_day_status": previous["day_status"],
                "second_day_status": current["day_status"],
                "period_status": period_status,
            }
        )
    return pd.DataFrame(rows)


def extract_hutton_episodes(daily: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping Hutton pairs into episodes of consecutive PASS days."""

    pass_dates = sorted(daily.loc[daily["day_status"] == "pass", "date"].tolist())
    episodes: list[list[date]] = []
    for value in pass_dates:
        if not episodes or value - episodes[-1][-1] != timedelta(days=1):
            episodes.append([value])
        else:
            episodes[-1].append(value)
    return pd.DataFrame(
        [
            {
                "episode_start": values[0],
                "episode_end": values[-1],
                "qualified_day_count": len(values),
                "hutton_pair_count": len(values) - 1,
            }
            for values in episodes
            if len(values) >= 2
        ]
    )


def aggregate_polyakov_daily(
    hourly: pd.DataFrame,
    timezone: str,
    config: PolyakovConfig = PolyakovConfig(),
    start_date: date | str | None = None,
    end_date: date | str | None = None,
) -> pd.DataFrame:
    """Create complete local-day T/RH means and precipitation sums for Polyakov windows."""

    if "precipitation" not in hourly:
        raise ValueError("Missing hourly column: precipitation")
    frame = _prepare_local_hourly(hourly, timezone)
    rows: list[dict[str, object]] = []
    for local_date in _date_range(frame, start_date, end_date):
        group = frame[frame["date"] == local_date]
        temperatures = group["temperature_2m"].where(
            group["temperature_2m"].between(
                config.minimum_physical_temperature_c,
                config.maximum_physical_temperature_c,
                inclusive="both",
            )
        ).dropna()
        humidities = group["relative_humidity_2m"].where(
            group["relative_humidity_2m"].between(0.0, 100.0, inclusive="both")
        ).dropna()
        precipitation = group["precipitation"].where(group["precipitation"] >= 0.0).dropna()
        counts = (len(temperatures), len(humidities), len(precipitation))
        expected = _expected_hours(local_date, timezone)
        required_hours = (
            expected if config.require_complete_daily_coverage else config.minimum_valid_hours_per_day
        )
        accepted = all(value >= required_hours for value in counts)
        rows.append(
            {
                "date": local_date,
                "temperature_mean_c": float(temperatures.mean()) if accepted else np.nan,
                "relative_humidity_mean_pct": float(humidities.mean()) if accepted else np.nan,
                "precipitation_sum_mm": float(precipitation.sum()) if accepted else np.nan,
                "temperature_valid_hours": counts[0],
                "humidity_valid_hours": counts[1],
                "precipitation_valid_hours": counts[2],
                "expected_hours": expected,
                "required_valid_hours": required_hours,
                "accepted": accepted,
            }
        )
    return pd.DataFrame(rows)


def classify_polyakov_windows(
    daily: pd.DataFrame,
    activation_date: date | str | None,
    config: PolyakovConfig = PolyakovConfig(),
) -> pd.DataFrame:
    """Run the explicit continuous-window state machine from the supplied specification."""

    if activation_date is None:
        output = daily.copy()
        output["status"] = "NOT_EVALUABLE_MISSING_PHENOPHASE"
        output["critical"] = pd.NA
        output["activation_date"] = pd.NaT
        output["critical_start"] = pd.NaT
        output["expected_manifestation_start"] = pd.NaT
        output["expected_manifestation_end"] = pd.NaT
        return output
    activation = pd.Timestamp(activation_date).date()
    output = daily.sort_values("date").reset_index(drop=True).copy()
    output["t10_c"] = output["temperature_mean_c"].rolling(config.window_days).mean()
    output["rh10_pct"] = output["relative_humidity_mean_pct"].rolling(config.window_days).mean()
    output["p10_mm"] = output["precipitation_sum_mm"].rolling(config.window_days).sum()
    statuses: list[str] = []
    critical_values: list[object] = []
    critical_starts: list[object] = []
    expected_starts: list[object] = []
    expected_ends: list[object] = []
    critical_start: date | None = None
    for index, row in output.iterrows():
        current_date = row["date"]
        window = output.iloc[max(0, index - config.window_days + 1) : index + 1]
        enough_days = (
            len(window) == config.window_days
            and current_date >= activation + timedelta(days=config.window_days - 1)
            and window["date"].iloc[-1] - window["date"].iloc[0] == timedelta(days=config.window_days - 1)
        )
        if current_date < activation:
            status = "NOT_ACTIVE"
            critical: object = pd.NA
            critical_start = None
        elif not enough_days or not bool(window["accepted"].all()):
            status = "INSUFFICIENT_DATA"
            critical = pd.NA
            critical_start = None
        else:
            critical = bool(
                config.minimum_window_temperature_c <= row["t10_c"] <= config.maximum_window_temperature_c
                and row["rh10_pct"] >= config.minimum_window_relative_humidity_pct
                and row["p10_mm"] >= config.minimum_window_precipitation_mm
            )
            if not critical:
                status = "LOW_WEATHER_RISK"
                critical_start = None
            else:
                if critical_start is None:
                    critical_start = current_date
                age = (current_date - critical_start).days
                if age < config.persistence_days:
                    status = "CRITICAL_CONDITIONS"
                elif age <= config.manifestation_lag_days[1]:
                    status = "OUTBREAK_EXPECTED"
                else:
                    status = "PROLONGED_RISK"
        if critical_start is None:
            expected_start = pd.NaT
            expected_end = pd.NaT
        else:
            expected_start = critical_start + timedelta(days=config.manifestation_lag_days[0])
            expected_end = critical_start + timedelta(days=config.manifestation_lag_days[1])
        statuses.append(status)
        critical_values.append(critical)
        critical_starts.append(critical_start if critical_start is not None else pd.NaT)
        expected_starts.append(expected_start)
        expected_ends.append(expected_end)
    output["status"] = statuses
    output["critical"] = pd.array(critical_values, dtype="boolean")
    output["activation_date"] = activation
    output["critical_start"] = critical_starts
    output["expected_manifestation_start"] = expected_starts
    output["expected_manifestation_end"] = expected_ends
    output["model"] = "polyakov_late_blight_10d_v1"
    output["model_version"] = config.model_version
    return output


def extract_polyakov_episodes(
    classified: pd.DataFrame,
    config: PolyakovConfig = PolyakovConfig(),
) -> pd.DataFrame:
    """Return one row per continuous sequence of critical Polyakov windows."""

    critical_rows = classified[classified["critical"].fillna(False)].copy()
    episodes: list[list[pd.Series]] = []
    for _, row in critical_rows.iterrows():
        if not episodes or row["date"] - episodes[-1][-1]["date"] != timedelta(days=1):
            episodes.append([row])
        else:
            episodes[-1].append(row)
    rows: list[dict[str, object]] = []
    for values in episodes:
        start = values[0]["date"]
        end = values[-1]["date"]
        duration = (end - start).days + 1
        rows.append(
            {
                "episode_start": start,
                "episode_end": end,
                "critical_day_count": duration,
                "persistence_confirmed": duration >= config.persistence_days + 1,
                "expected_manifestation_start": start + timedelta(days=config.manifestation_lag_days[0]),
                "expected_manifestation_end": start + timedelta(days=config.manifestation_lag_days[1]),
            }
        )
    return pd.DataFrame(rows)
