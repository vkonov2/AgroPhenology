"""Pilot association summaries for late-blight weather indicators and field reports."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def summarize_hutton_lookbacks(
    observation_id: str,
    observation_date: date | str,
    observation_outcome: str,
    periods: pd.DataFrame,
    lookback_days: tuple[int, ...] = (7, 14, 21),
    exclude_observation_day: bool = True,
) -> pd.DataFrame:
    """Report fixed lookback windows without selecting the best one post hoc."""

    observed = pd.Timestamp(observation_date).date()
    cutoff = observed - timedelta(days=1) if exclude_observation_day else observed
    prior_passes = (
        periods[(periods["period_status"] == "pass") & (periods["period_end"] <= cutoff)]
        if not periods.empty
        else periods
    )
    nearest_prior = max(prior_passes["period_end"]) if not prior_passes.empty else pd.NaT
    nearest_prior_lag = (observed - nearest_prior).days if pd.notna(nearest_prior) else pd.NA
    rows: list[dict[str, object]] = []
    for days in lookback_days:
        start = observed - timedelta(days=days)
        if periods.empty:
            window = periods.copy()
        else:
            window = periods[(periods["period_end"] >= start) & (periods["period_end"] <= cutoff)]
        passes = window[window["period_status"] == "pass"] if not window.empty else window
        indeterminate_count = int((window["period_status"] == "indeterminate").sum()) if not window.empty else 0
        if not passes.empty:
            signal: bool | None = True
            last_signal = max(passes["period_end"])
            days_from_signal = (observed - last_signal).days
        elif indeterminate_count:
            signal = None
            last_signal = pd.NaT
            days_from_signal = pd.NA
        else:
            signal = False
            last_signal = pd.NaT
            days_from_signal = pd.NA
        if signal is None:
            association = "NOT_EVALUABLE_INCOMPLETE_WEATHER"
        elif observation_outcome == "detected" and signal:
            association = "TEMPORAL_CONCORDANCE"
        elif observation_outcome == "detected":
            association = "DETECTION_WITHOUT_MODEL_SIGNAL"
        elif observation_outcome == "not_detected" and signal:
            association = "SIGNAL_WITHOUT_DETECTION"
        else:
            association = "NO_SIGNAL_AND_SINGLE_NONDETECTION"
        rows.append(
            {
                "observation_id": observation_id,
                "observation_date": observed,
                "observation_outcome": observation_outcome,
                "lookback_days": days,
                "weather_cutoff_date": cutoff,
                "observation_day_excluded": exclude_observation_day,
                "hutton_signal_present": signal,
                "hutton_pass_pair_count": int(len(passes)),
                "hutton_indeterminate_pair_count": indeterminate_count,
                "last_hutton_period_end": last_signal,
                "days_from_last_hutton_period": days_from_signal,
                "nearest_prior_hutton_period_end": nearest_prior,
                "days_from_nearest_prior_hutton_period": nearest_prior_lag,
                "pilot_association": association,
            }
        )
    return pd.DataFrame(rows)


def summarize_polyakov_observation(
    observation_id: str,
    observation_date: date | str,
    observation_outcome: str,
    classified: pd.DataFrame,
    activation_rule: str,
    exclude_observation_day: bool = True,
) -> dict[str, object]:
    """Summarize manifestation-window overlap without calling it a confusion-matrix class."""

    observed = pd.Timestamp(observation_date).date()
    weather_cutoff = observed - timedelta(days=1) if exclude_observation_day else observed
    if classified.empty or classified["status"].eq("NOT_EVALUABLE_MISSING_PHENOPHASE").all():
        return {
            "observation_id": observation_id,
            "observation_date": observed,
            "observation_outcome": observation_outcome,
            "activation_rule": activation_rule,
            "weather_cutoff_date": weather_cutoff,
            "observation_day_excluded": exclude_observation_day,
            "polyakov_evaluable": False,
            "polyakov_reason": "missing_phenophase",
            "manifestation_window_match": pd.NA,
            "high_weather_risk_on_observation_date": pd.NA,
            "weather_status_on_observation_date": pd.NA,
            "weather_status_at_cutoff": pd.NA,
            "high_weather_risk_at_cutoff": pd.NA,
            "nearest_prior_manifestation_start": pd.NaT,
            "nearest_prior_manifestation_end": pd.NaT,
            "days_from_nearest_prior_manifestation_end": pd.NA,
            "pilot_association": "NOT_EVALUABLE",
        }
    confirmed_statuses = {"OUTBREAK_EXPECTED", "PROLONGED_RISK"}
    eligible = classified[
        pd.to_datetime(classified["date"]) <= pd.Timestamp(weather_cutoff)
    ]
    expected = eligible[
        eligible["status"].isin(confirmed_statuses)
        & eligible["expected_manifestation_start"].notna()
        & eligible["expected_manifestation_end"].notna()
    ]
    matches = expected[
        expected.apply(
            lambda row: row["expected_manifestation_start"] <= observed <= row["expected_manifestation_end"],
            axis=1,
        )
    ]
    confirmed_windows = expected[
        ["expected_manifestation_start", "expected_manifestation_end"]
    ].drop_duplicates().copy()
    if confirmed_windows.empty:
        nearest_prior_start = pd.NaT
        nearest_prior_end = pd.NaT
        nearest_prior_lag = pd.NA
    else:
        confirmed_windows["_end_timestamp"] = pd.to_datetime(
            confirmed_windows["expected_manifestation_end"]
        )
        prior_windows = confirmed_windows[
            confirmed_windows["_end_timestamp"] < pd.Timestamp(observed)
        ]
        if prior_windows.empty:
            nearest_prior_start = pd.NaT
            nearest_prior_end = pd.NaT
            nearest_prior_lag = pd.NA
        else:
            nearest_row = prior_windows.sort_values("_end_timestamp").iloc[-1]
            nearest_prior_start = pd.Timestamp(
                nearest_row["expected_manifestation_start"]
            ).date()
            nearest_prior_end = pd.Timestamp(nearest_row["expected_manifestation_end"]).date()
            nearest_prior_lag = (observed - nearest_prior_end).days
    on_date = classified[classified["date"] == observed]
    on_date_status = str(on_date.iloc[-1]["status"]) if not on_date.empty else "MISSING_DATE"
    at_cutoff = classified[classified["date"] == weather_cutoff]
    cutoff_status = str(at_cutoff.iloc[-1]["status"]) if not at_cutoff.empty else "MISSING_DATE"
    incomplete_at_cutoff = cutoff_status in {"INSUFFICIENT_DATA", "MISSING_DATE"}
    if not bool(classified["accepted"].any()):
        reason = "missing_required_weather_variables"
        evaluable = False
    elif incomplete_at_cutoff and matches.empty:
        reason = "incomplete_weather_at_cutoff"
        evaluable = False
    else:
        reason = ""
        evaluable = True
    high_risk_on_date = (
        bool(on_date["status"].isin(confirmed_statuses).any())
        if evaluable and not exclude_observation_day and not on_date.empty
        else pd.NA
    )
    high_risk_at_cutoff = (
        bool(at_cutoff["status"].isin(confirmed_statuses).any())
        if evaluable and not at_cutoff.empty
        else pd.NA
    )
    match = bool(not matches.empty) if evaluable else pd.NA
    if not evaluable:
        association = (
            "NOT_EVALUABLE_INCOMPLETE_WEATHER"
            if reason == "incomplete_weather_at_cutoff"
            else "NOT_EVALUABLE"
        )
    elif observation_outcome == "detected" and (match or high_risk_at_cutoff is True):
        association = "TEMPORAL_CONCORDANCE"
    elif observation_outcome == "detected":
        association = "DETECTION_WITHOUT_MODEL_SIGNAL"
    elif observation_outcome == "not_detected" and (match or high_risk_at_cutoff is True):
        association = "SIGNAL_WITHOUT_DETECTION"
    else:
        association = "NO_SIGNAL_AND_SINGLE_NONDETECTION"
    return {
        "observation_id": observation_id,
        "observation_date": observed,
        "observation_outcome": observation_outcome,
        "activation_rule": activation_rule,
        "weather_cutoff_date": weather_cutoff,
        "observation_day_excluded": exclude_observation_day,
        "polyakov_evaluable": evaluable,
        "polyakov_reason": reason,
        "manifestation_window_match": match,
        "high_weather_risk_on_observation_date": high_risk_on_date,
        "weather_status_on_observation_date": on_date_status,
        "weather_status_at_cutoff": cutoff_status,
        "high_weather_risk_at_cutoff": high_risk_at_cutoff,
        "nearest_prior_manifestation_start": nearest_prior_start,
        "nearest_prior_manifestation_end": nearest_prior_end,
        "days_from_nearest_prior_manifestation_end": nearest_prior_lag,
        "pilot_association": association,
    }


def summarize_polyakov_activation_interval(
    observation_id: str,
    observation_date: date | str,
    observation_outcome: str,
    scenario_summaries: pd.DataFrame,
    activation_date_start: date | str,
    activation_date_end: date | str,
    phenophase_status: str = "AUTHOR_CONFIRMED_REGIONAL_INTERVAL",
) -> dict[str, object]:
    """Aggregate every activation date in an interval without majority voting."""

    if scenario_summaries.empty:
        raise ValueError("Activation interval requires at least one scenario")
    start = pd.Timestamp(activation_date_start).date()
    end = pd.Timestamp(activation_date_end).date()
    expected_dates = [value.date() for value in pd.date_range(start, end, freq="D")]
    scenario_dates = sorted(
        pd.Timestamp(value.rsplit("_", 1)[-1]).date()
        for value in scenario_summaries["activation_rule"]
    )
    if scenario_dates != expected_dates:
        raise ValueError("Activation scenarios must cover every date in the interval exactly once")

    evaluable_count = int(scenario_summaries["polyakov_evaluable"].eq(True).sum())
    scenario_count = len(scenario_summaries)
    associations = sorted(set(scenario_summaries["pilot_association"].dropna()))
    reasons = sorted(
        {
            str(value)
            for value in scenario_summaries["polyakov_reason"].dropna()
            if str(value)
        }
    )
    if evaluable_count == scenario_count and len(associations) == 1:
        interval_status = "CONSISTENT_ACROSS_FULL_ACTIVATION_INTERVAL"
        interval_result = f"{associations[0]}__ROBUST_TO_ACTIVATION_INTERVAL"
        evaluable = True
        evaluability_status = "CONDITIONALLY_EVALUABLE_ACTIVATION_DATE_INTERVAL"
        reason = ""
        association = associations[0]
    elif evaluable_count == scenario_count:
        interval_status = "ACTIVATION_DATE_SENSITIVE"
        interval_result = "ACTIVATION_DATE_SENSITIVE"
        evaluable = True
        evaluability_status = "CONDITIONALLY_EVALUABLE_ACTIVATION_DATE_INTERVAL"
        reason = "activation_date_sensitive"
        association = "ACTIVATION_DATE_SENSITIVE"
    elif evaluable_count:
        interval_status = "PARTIALLY_EVALUABLE"
        interval_result = "PARTIALLY_EVALUABLE"
        evaluable = False
        evaluability_status = "PARTIALLY_EVALUABLE"
        reason = "partially_evaluable_activation_interval"
        association = "PARTIALLY_EVALUABLE"
    else:
        interval_status = "NOT_EVALUABLE"
        interval_result = associations[0] if len(associations) == 1 else "NOT_EVALUABLE"
        evaluable = False
        evaluability_status = interval_result
        reason = reasons[0] if len(reasons) == 1 else "multiple_non_evaluable_reasons"
        association = interval_result

    def consistent_value(column: str) -> object:
        values = scenario_summaries[column].dropna().drop_duplicates().tolist()
        return values[0] if len(values) == 1 else pd.NA

    lag_values = pd.to_numeric(
        scenario_summaries["days_from_nearest_prior_manifestation_end"],
        errors="coerce",
    ).dropna()
    start_values = pd.to_datetime(
        scenario_summaries["nearest_prior_manifestation_start"],
        errors="coerce",
    ).dropna()
    end_values = pd.to_datetime(
        scenario_summaries["nearest_prior_manifestation_end"],
        errors="coerce",
    ).dropna()
    return {
        "observation_id": observation_id,
        "observation_date": pd.Timestamp(observation_date).date(),
        "observation_outcome": observation_outcome,
        "activation_rule": f"author_confirmed_regional_phenophase_interval_{start}_{end}",
        "activation_date_start": start,
        "activation_date_end": end,
        "phenophase_status": phenophase_status,
        "field_specific_activation_date_known": False,
        "weather_cutoff_date": consistent_value("weather_cutoff_date"),
        "observation_day_excluded": consistent_value("observation_day_excluded"),
        "polyakov_evaluable": evaluable,
        "polyakov_evaluability_status": evaluability_status,
        "polyakov_reason": reason,
        "manifestation_window_match": consistent_value("manifestation_window_match"),
        "high_weather_risk_on_observation_date": pd.NA,
        "weather_status_on_observation_date": consistent_value(
            "weather_status_on_observation_date"
        ),
        "weather_status_at_cutoff": consistent_value("weather_status_at_cutoff"),
        "high_weather_risk_at_cutoff": consistent_value("high_weather_risk_at_cutoff"),
        "nearest_prior_manifestation_start": pd.NaT,
        "nearest_prior_manifestation_end": pd.NaT,
        "days_from_nearest_prior_manifestation_end": pd.NA,
        "nearest_prior_manifestation_start_min": (
            start_values.min().date() if not start_values.empty else pd.NaT
        ),
        "nearest_prior_manifestation_start_max": (
            start_values.max().date() if not start_values.empty else pd.NaT
        ),
        "nearest_prior_manifestation_end_min": (
            end_values.min().date() if not end_values.empty else pd.NaT
        ),
        "nearest_prior_manifestation_end_max": (
            end_values.max().date() if not end_values.empty else pd.NaT
        ),
        "days_from_nearest_prior_manifestation_end_min": (
            int(lag_values.min()) if not lag_values.empty else pd.NA
        ),
        "days_from_nearest_prior_manifestation_end_max": (
            int(lag_values.max()) if not lag_values.empty else pd.NA
        ),
        "activation_scenario_count": scenario_count,
        "evaluable_activation_scenario_count": evaluable_count,
        "activation_interval_status": interval_status,
        "activation_interval_result": interval_result,
        "possible_pilot_associations": "|".join(associations),
        "pilot_association": association,
    }
