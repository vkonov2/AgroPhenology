"""Explicit adaptations of public insect thermal-stage models.

The adapters implement published thermal milestones, not pesticide advice.
The underlying single-sine integrator has a numerical reference benchmark;
no biological external-validation result is implied for these adapters.
Daily weather must carry its actual or explicitly assumed availability time.
No default biofix, ground-thaw date, upper threshold or leaf-stage is invented.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
import pandas as pd

from .external_dd_benchmark import estimate_degree_days


@dataclass(frozen=True)
class Milestone:
    stage_id: str
    threshold_c_days: float
    published_value: float
    published_unit: str


@dataclass(frozen=True)
class ThermalModel:
    model_id: str
    organism: str
    host: str
    lower_c: float
    upper_c: float | None
    biofix_kind: str
    source_url: str
    calculation_source_url: str
    calculation_method: str
    implementation_status: str
    milestones: tuple[Milestone, ...]


NC_STATE: Final = "https://apples.ces.ncsu.edu/about-degree-day-models/"
UC_CODLING_TABLE: Final = "https://ipm.ucanr.edu/weather/ddtable/codling_moth.html"
UW_DELIA: Final = "https://vegento.russell.wisc.edu/pests/cabbage-maggot/"
UC_PLUTELLA: Final = "https://ipm.ucanr.edu/weather/phenology-models-description/diamondback-moth/"


def _f_days(value: float) -> float:
    return value * 5.0 / 9.0


MODEL_SPECS: Final = {
    "codling_ncstate_thresholds": ThermalModel(
        model_id="codling_ncstate_thresholds", organism="Cydia pomonella", host="apple",
        lower_c=10.0, upper_c=(88.0-32.0)*5.0/9.0,
        biofix_kind="observed_adult_trap_biofix",
        source_url=NC_STATE, calculation_source_url=UC_CODLING_TABLE,
        calculation_method="single_sine_horizontal",
        implementation_status="NC_State_milestones_with_UC_IPM_calculator_explicit_adaptation_not_exact_NC_reference_code",
        milestones=(
            Milestone("generation_1_egg_hatch_start", _f_days(350), 350, "F_degree_days"),
            Milestone("generation_1_egg_hatch_end", _f_days(1050), 1050, "F_degree_days"),
            Milestone("generation_2_egg_hatch_start", _f_days(1300), 1300, "F_degree_days"),
            Milestone("generation_2_egg_hatch_end", _f_days(2600), 2600, "F_degree_days"),
        ),
    ),
    "cabbage_maggot_uw": ThermalModel(
        model_id="cabbage_maggot_uw", organism="Delia radicum", host="cruciferous_crops",
        lower_c=(43.0-32.0)*5.0/9.0, upper_c=None,
        biofix_kind="observed_ground_thaw",
        source_url=UW_DELIA, calculation_source_url="https://ipm.ucanr.edu/weather/ddconcepts.html",
        calculation_method="single_sine_uncapped",
        implementation_status="published_thresholds_single_sine_adaptation_source_does_not_specify_calculation_method",
        milestones=(
            Milestone("generation_1_adult_appearance", _f_days(300), 300, "F_degree_days"),
            Milestone("generation_2_adult_appearance", _f_days(1476), 1476, "F_degree_days"),
            Milestone("generation_3_adult_appearance", _f_days(2652), 2652, "F_degree_days"),
        ),
    ),
    "diamondback_harcourt_cabbage": ThermalModel(
        model_id="diamondback_harcourt_cabbage", organism="Plutella xylostella", host="cabbage",
        lower_c=7.3, upper_c=None, biofix_kind="observed_egg_cohort",
        source_url=UC_PLUTELLA, calculation_source_url=UC_PLUTELLA,
        calculation_method="single_sine_uncapped",
        implementation_status="UC_IPM_recommended_single_sine_original_study_method_unspecified",
        milestones=(Milestone("egg_to_adult_generation", 283.0, 283.0, "C_degree_days"),),
    ),
    "diamondback_butts_brussels_sprouts": ThermalModel(
        model_id="diamondback_butts_brussels_sprouts", organism="Plutella xylostella", host="brussels_sprouts",
        lower_c=7.3, upper_c=None, biofix_kind="observed_egg_cohort",
        source_url=UC_PLUTELLA, calculation_source_url=UC_PLUTELLA,
        calculation_method="single_sine_uncapped",
        implementation_status="UC_IPM_recommended_single_sine_original_study_method_unspecified",
        milestones=(Milestone("egg_to_adult_generation", 293.0, 293.0, "C_degree_days"),),
    ),
}


def model_registry() -> list[dict]:
    """Serializable specification; status never masquerades as field quality."""
    return [dict(asdict(spec), biological_validation_status="blocked_appropriate_outcome_or_biofix_unavailable")
            for spec in MODEL_SPECS.values()]


def _aware(value, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware timestamp")
    return result


def _date(value, issue: pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result):
        return pd.NaT
    if result.tzinfo is not None:
        result = result.tz_convert(issue.tz).tz_localize(None)
    return result.normalize()


def daily_thermal_units(tmin_c, tmax_c, model_id: str) -> np.ndarray:
    """Use the verified integrator; no physical upper cutoff when unspecified.

    The numerical primitive requires a finite upper argument. For an uncapped
    model a bound above every finite input maximum makes its upper component
    exactly zero. This computational bound is not a biological parameter.
    """
    if model_id not in MODEL_SPECS:
        raise ValueError("Unknown thermal model")
    spec = MODEL_SPECS[model_id]
    low, high = np.broadcast_arrays(np.asarray(tmin_c, dtype=float), np.asarray(tmax_c, dtype=float))
    upper = spec.upper_c
    if upper is None:
        finite = high[np.isfinite(high)]
        upper = max(spec.lower_c, float(finite.max()) if len(finite) else spec.lower_c) + 1.0
    return estimate_degree_days(low, high, lower=spec.lower_c, upper=upper,
                                curve="sine", cutoff="horizontal")


def predict_stage_dates(
    daily_minmax: pd.DataFrame,
    model_id: str,
    biofix_date,
    issued_at,
    availability_column: str = "available_at",
    *,
    biofix_kind: str | None = None,
    biofix_available_at=None,
) -> pd.DataFrame:
    """One past-only decision for each published milestone of one model.

    Input columns are ``date`` (local calendar day), ``tmin_c``, ``tmax_c`` and
    an availability timestamp. Only records already available at issuance and
    complete local days strictly before issue_date can enter the calculation.
    Unknown availability is unavailable, never assumed contemporaneous.

    Biofix must have the model-specific observed meaning AND an explicit
    availability timestamp. Its date becomes usable no earlier than the next
    local day, and must not be inferred from a calendar date or future event.
    Accumulation includes the full biofix day: an explicit research convention.

    Missing dates inside required history or the last seven consecutive rate
    days abstain. The unavailable recent gap and future are forecast using the
    fixed mean DD of those seven available days, not realized future weather.
    This persistence extension is our declared research adaptation. It is not
    an archived operational forecast, probability, or field-validation result.
    """
    if model_id not in MODEL_SPECS:
        raise ValueError("Unknown thermal model")
    issue = _aware(issued_at, "issued_at")
    issue_day = issue.tz_localize(None).normalize()
    spec = MODEL_SPECS[model_id]
    rows = [dict(model_id=model_id, organism=spec.organism, host=spec.host,
                 stage_id=stage.stage_id, threshold_c_days=stage.threshold_c_days,
                 issued_at=issue.isoformat(), biofix_date=pd.NaT,
                 required_biofix_kind=spec.biofix_kind, feature_cutoff_date=pd.NaT,
                 known_cumulative_c_days=np.nan, forecast_daily_c_days=np.nan,
                 predicted_date=pd.NaT, score=np.nan, status="abstained_biofix_missing",
                 calculation_method=spec.calculation_method,
                 implementation_status=spec.implementation_status,
                 biological_validation_status="not_established") for stage in spec.milestones]

    def abstain(reason: str) -> pd.DataFrame:
        for row in rows:
            row["status"] = reason
        return pd.DataFrame(rows)

    anchor = _date(biofix_date, issue) if biofix_date is not None else pd.NaT
    if pd.isna(anchor):
        return abstain("abstained_biofix_missing")
    for row in rows:
        row["biofix_date"] = anchor
    if biofix_kind != spec.biofix_kind:
        return abstain("abstained_biofix_kind_missing_or_incompatible")
    if biofix_available_at is None:
        return abstain("abstained_biofix_availability_unknown")
    anchor_available = _aware(biofix_available_at, "biofix_available_at")
    if anchor_available.tz_convert(issue.tz).tz_localize(None).normalize() < anchor:
        return abstain("abstained_biofix_published_before_observation_date")
    if anchor >= issue_day or anchor_available > issue:
        return abstain("abstained_biofix_not_yet_available")
    required = {"date", "tmin_c", "tmax_c", availability_column}
    if not required.issubset(daily_minmax):
        raise ValueError(f"Missing daily weather columns: {sorted(required-set(daily_minmax))}")
    available = pd.to_datetime(daily_minmax[availability_column], utc=True, errors="coerce")
    frame = daily_minmax.loc[available.le(issue.tz_convert("UTC"))].copy()
    if frame.empty:
        return abstain("abstained_no_weather_available")
    try:
        frame["date"] = [_date(day, issue) for day in frame.date]
    except (ValueError, TypeError):
        return abstain("abstained_invalid_available_weather_dates")
    if frame.date.isna().any():
        return abstain("abstained_invalid_available_weather_dates")
    frame = frame[frame.date.lt(issue_day)]
    if frame.empty:
        return abstain("abstained_no_complete_past_weather_day")
    local_ends = pd.DatetimeIndex(frame.date + pd.Timedelta(days=1)).tz_localize(issue.tz).tz_convert("UTC")
    published = pd.to_datetime(frame[availability_column], utc=True, errors="raise")
    if (published.to_numpy() < local_ends.to_numpy()).any():
        return abstain("abstained_weather_published_before_day_complete")
    cutoff = frame.date.max()
    first_rate_day = cutoff-pd.Timedelta(days=6)
    frame = frame[frame.date.ge(min(anchor, first_rate_day))].sort_values("date")
    if frame.date.duplicated().any():
        return abstain("abstained_duplicate_available_weather_days")
    minimum = pd.to_numeric(frame.tmin_c, errors="coerce").astype(float)
    maximum = pd.to_numeric(frame.tmax_c, errors="coerce").astype(float)
    try:
        thermal = daily_thermal_units(minimum, maximum, model_id)
    except ValueError:
        return abstain("abstained_invalid_available_temperature_range")
    calendar = pd.date_range(min(anchor, first_rate_day), cutoff, freq="D")
    daily = pd.Series(thermal, index=pd.DatetimeIndex(frame.date)).reindex(calendar)
    history = daily.loc[anchor:cutoff] if cutoff >= anchor else pd.Series(dtype=float)
    for row in rows:
        row["feature_cutoff_date"] = cutoff
    if history.isna().any():
        return abstain("abstained_missing_cumulative_weather")
    rate_days = daily.loc[first_rate_day:cutoff]
    rate = float(rate_days.mean()) if len(rate_days) == 7 and rate_days.notna().all() else np.nan
    cumulative = history.cumsum()
    known = float(history.sum())
    for row in rows:
        row["known_cumulative_c_days"] = known
        row["forecast_daily_c_days"] = rate
        crossed = cumulative.ge(row["threshold_c_days"]-1e-10)
        if crossed.any():
            row.update(predicted_date=crossed.index[crossed][0], score=0.0, status="computed_past_crossing")
            continue
        if not np.isfinite(rate):
            row["status"] = "abstained_missing_persistence_weather"
            continue
        if rate <= 0:
            row["status"] = "abstained_zero_persistence_rate"
            continue
        first_forecast_day = max(anchor, cutoff+pd.Timedelta(days=1))
        days_needed = np.ceil(max((row["threshold_c_days"]-known)/rate-1e-12, 0.0))
        try:
            prediction = first_forecast_day+pd.Timedelta(days=max(int(days_needed)-1, 0))
        except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime, pd.errors.OutOfBoundsTimedelta):
            row["status"] = "abstained_prediction_out_of_range"
            continue
        row.update(predicted_date=prediction,
                   score=float(issue_day+pd.Timedelta(days=3) <= prediction <= issue_day+pd.Timedelta(days=10)),
                   status="computed_persistence_forecast")
    return pd.DataFrame(rows)
