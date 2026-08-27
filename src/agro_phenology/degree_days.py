"""Pure degree-day calculations."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import numpy as np
import pandas as pd


def daily_degree_days(mean_temperatures_c: Iterable[float], base_temperature_c: float) -> np.ndarray:
    """Return daily effective temperatures in °C·day."""

    values = np.asarray(list(mean_temperatures_c), dtype=float)
    return np.maximum(values - float(base_temperature_c), 0.0)


def cumulative_degree_days(daily_values: Iterable[float]) -> np.ndarray:
    """Return cumulative effective temperatures in °C·day."""

    return np.cumsum(np.asarray(list(daily_values), dtype=float))


def first_threshold_date(
    dates: Iterable[date | str | pd.Timestamp],
    cumulative_values: Iterable[float],
    threshold: float,
) -> date | None:
    """Return the first date whose cumulative value reaches the threshold."""

    date_values = pd.to_datetime(list(dates), errors="raise")
    cumulative = np.asarray(list(cumulative_values), dtype=float)
    if len(date_values) != len(cumulative):
        raise ValueError("dates and cumulative_values must have the same length")
    indexes = np.flatnonzero(np.isfinite(cumulative) & (cumulative >= threshold))
    return None if len(indexes) == 0 else date_values[indexes[0]].date()


def add_degree_days(
    daily_weather: pd.DataFrame,
    base_temperature_c: float,
    threshold: float,
) -> tuple[pd.DataFrame, date | None]:
    """Add daily and cumulative degree-days to an accepted daily-weather table."""

    required = {"date", "temperature_mean_c", "accepted"}
    missing = required - set(daily_weather.columns)
    if missing:
        raise ValueError(f"Missing daily weather columns: {sorted(missing)}")
    output = daily_weather.copy()
    accepted_mean = output["temperature_mean_c"].where(output["accepted"].astype(bool))
    output["daily_degree_days_c_day"] = np.maximum(accepted_mean - base_temperature_c, 0.0)
    output["cumulative_degree_days_c_day"] = output["daily_degree_days_c_day"].fillna(0.0).cumsum()
    predicted = first_threshold_date(output["date"], output["cumulative_degree_days_c_day"], threshold)
    return output, predicted
