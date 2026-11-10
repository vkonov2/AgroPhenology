"""Fixed first-registration timing models, with explicit right censoring.

This is a new statistical experiment, not a reproduction of AHDB coefficients
or an insect development model. Its target is the first registered capture or
detection after a measured-zero entry. Missing visits never become negative
biological observations. AFT labels use observed registration dates as points
and last measured follow-up as right-censor bounds.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


MODEL_VERSION = "external_event_aft_v1"
METADATA_KEY = "agro_event_aft"
STATION_FEATURES = ["entry_doy", "cluster"]
ENTRY_STAGE_FEATURES = ["entry_doy", "entry_egg_presence", "entry_adult_presence"]
PARAMETERS = {
    "loss_function": "SurvivalAft:dist=Normal;scale=1",
    "depth": 3,
    "iterations": 200,
    "learning_rate": .03,
    "l2_leaf_reg": 10,
    "random_seed": 20261110,
    "thread_count": 2,
    "bootstrap_type": "No",
    "allow_writing_files": False,
    "verbose": False,
}
CALENDAR_MIN_GROUP_EVENTS = 3


def _year(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer year")
    value = int(value)
    if not 1678 <= value <= 2260:
        raise ValueError(f"{name} is outside the supported calendar")
    return value


def _dates(values: pd.Series, name: str, *, allow_missing: bool = False) -> pd.Series:
    try:
        dates = pd.to_datetime(values, errors="raise")
        # Source assessments are local calendar dates. Preserve their local
        # dates, rather than inventing a common UTC timezone for US stations.
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)
        dates = dates.dt.normalize()
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError(f"Invalid {name}") from error
    if not allow_missing and dates.isna().any():
        raise ValueError(f"Missing {name}")
    return dates


def _scope(rows: pd.DataFrame) -> dict:
    result = {}
    for name in ("source_id", "target"):
        if name in rows:
            if rows[name].isna().any() or rows[name].nunique() != 1:
                raise ValueError(f"One model must contain exactly one {name}")
            result[name] = str(rows[name].iloc[0])
    return result


def _check_scope(rows: pd.DataFrame, metadata: dict) -> None:
    for name in ("source_id", "target"):
        if name in metadata and name in rows:
            if rows[name].isna().any() or not rows[name].astype(str).eq(metadata[name]).all():
                raise ValueError(f"Prediction rows do not match the model's {name}")


def feature_frame(rows: pd.DataFrame, *, use_station: bool) -> pd.DataFrame:
    """Allowlist entry-time inputs only; ignore outcome and follow-up columns."""
    if not isinstance(use_station, bool):
        raise ValueError("use_station must be boolean")
    if "entry_date" not in rows:
        raise ValueError("Missing entry_date")
    entry = _dates(rows.entry_date, "entry_date")
    features = pd.DataFrame({"entry_doy": entry.dt.dayofyear.astype(float)}, index=rows.index)
    if use_station:
        if "cluster" not in rows or rows.cluster.isna().any() or rows.cluster.astype(str).eq("").any():
            raise ValueError("Missing cluster at prediction entry")
        features["cluster"] = rows.cluster.astype(str)
    else:
        for name in ENTRY_STAGE_FEATURES[1:]:
            if name not in rows:
                raise ValueError(f"Missing entry-time feature {name}")
            try:
                values = pd.to_numeric(rows[name], errors="raise").astype(float)
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid {name}") from error
            if np.isinf(values.to_numpy()).any() or not values.dropna().isin([0., 1.]).all():
                raise ValueError(f"{name} must be measured presence 0/1 or unknown")
            features[name] = values
    return features


def _training_rows(registry: pd.DataFrame, outer_year: int) -> tuple[pd.DataFrame, dict]:
    outer_year = _year(outer_year, "outer_year")
    required = {"season", "entry_date", "first_recorded_event_date", "last_assessment_date"}
    if not required.issubset(registry):
        raise ValueError(f"Missing training columns: {sorted(required-set(registry))}")
    if registry.empty:
        raise ValueError("No training rows")
    scope = _scope(registry)
    years = pd.to_numeric(registry.season, errors="raise")
    if years.isna().any() or not np.isfinite(years).all() or not years.eq(np.floor(years)).all():
        raise ValueError("Invalid training season")
    if years.ge(outer_year).any():
        raise ValueError("Training must contain only strictly prior years")
    train = registry.copy()
    train["season"] = years.astype(int)
    received = len(train)
    if "cohort_included" in train:
        if train.cohort_included.isna().any() or not train.cohort_included.isin([True, False]).all():
            raise ValueError("Invalid cohort inclusion flags")
        train = train[train.cohort_included.astype(bool)].copy()
    if train.empty:
        raise ValueError("No included training rows")
    if "field_season" in train and train.field_season.duplicated().any():
        raise ValueError("Duplicate training field-season")
    entry = _dates(train.entry_date, "entry_date")
    event = _dates(train.first_recorded_event_date, "first_recorded_event_date", allow_missing=True)
    last = _dates(train.last_assessment_date, "last_assessment_date")
    if not entry.dt.year.eq(train.season).all():
        raise ValueError("Training entry dates disagree with season")
    observed = event.notna()
    if (event[observed] <= entry[observed]).any():
        raise ValueError("A measured-zero entry must precede its first event")
    if (last < entry).any() or (event[observed] > last[observed]).any():
        raise ValueError("Invalid measured follow-up chronology")
    endpoints = event.where(observed, last)
    if endpoints.dt.year.ge(outer_year).any():
        raise ValueError("Training outcomes or censoring extend into the external year")
    duration = (endpoints-entry).dt.days.astype(float)
    zero_followup = ~observed & duration.eq(0)
    train["entry_date"], train["first_recorded_event_date"], train["last_assessment_date"] = entry, event, last
    train["_duration"] = duration
    train["_observed"] = observed
    metadata = dict(version=MODEL_VERSION, outer_year=outer_year, received_rows=received,
                    noncohort_rows=received-len(train), cohort_rows=len(train),
                    excluded_zero_followup=int(zero_followup.sum()), **scope)
    train = train[~zero_followup].copy()
    if train.empty or not train._observed.any():
        raise ValueError("A timing model requires positive follow-up and an observed event")
    if not train._duration.gt(0).all():
        raise ValueError("AFT duration must be positive")
    order = ["season", "field_season"] if "field_season" in train else ["season", "entry_date"]
    train = train.sort_values(order, kind="stable")
    metadata.update(training_years=[int(v) for v in sorted(train.season.unique())],
                    training_max_year=int(train.season.max()), used_rows=len(train),
                    observed_events=int(train._observed.sum()), right_censored=int((~train._observed).sum()))
    return train, metadata


def aft_training_labels(train: pd.DataFrame) -> np.ndarray:
    """Expose the already-validated point/right-censored labels for audit."""
    lower = train._duration.to_numpy(dtype=float)
    upper = np.where(train._observed.to_numpy(dtype=bool), lower, -1.)
    return np.column_stack([lower, upper])


def fit_aft(train_registry: pd.DataFrame, outer_year: int, use_station: bool = True) -> CatBoostRegressor:
    train, metadata = _training_rows(train_registry, outer_year)
    features = feature_frame(train, use_station=use_station)
    if not any(features[name].nunique(dropna=False) > 1 for name in features):
        raise ValueError("Training entry features have no variation")
    metadata.update(model_kind="CatBoost_SurvivalAft", use_station=use_station,
                    features=features.columns.tolist(), parameters=dict(PARAMETERS),
                    duration_prediction="exp_RawFormulaVal_Normal_median_round_half_up_at_least_one_day",
                    label_semantics="point_first_registration_or_right_censored_last_measured_followup")
    model = CatBoostRegressor(**PARAMETERS, metadata={METADATA_KEY: json.dumps(metadata, sort_keys=True)})
    pool = Pool(features, label=aft_training_labels(train), cat_features=["cluster"] if use_station else [])
    model.fit(pool)
    return model


def aft_training_metadata(model: CatBoostRegressor) -> dict:
    try:
        return json.loads(model.get_metadata()[METADATA_KEY])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Model does not contain the external-event AFT contract") from error


def _prediction_entry(rows: pd.DataFrame, metadata: dict) -> pd.Series:
    if "season" not in rows:
        raise ValueError("Missing prediction season")
    seasons = pd.to_numeric(rows.season, errors="raise")
    if not seasons.eq(metadata["outer_year"]).all():
        raise ValueError("Prediction season does not match the declared external year")
    _check_scope(rows, metadata)
    entry = _dates(rows.entry_date, "entry_date")
    if not entry.dt.year.eq(seasons).all():
        raise ValueError("Prediction entry dates disagree with season")
    return entry


def _duration_dates(entry: pd.Series, log_durations) -> pd.Series:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        durations = np.exp(np.asarray(log_durations, dtype=float))
    if durations.shape != (len(entry),):
        raise ValueError("Model returned an unexpected prediction shape")
    dates = []
    for day, duration in zip(entry, durations):
        if not np.isfinite(duration):
            dates.append(pd.NaT)
            continue
        rounded = max(1, int(np.floor(duration+.5)))
        try:
            dates.append(day + pd.Timedelta(days=rounded))
        except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime, pd.errors.OutOfBoundsTimedelta):
            dates.append(pd.NaT)
    return pd.Series(dates, index=entry.index, dtype="datetime64[ns]", name="predicted_event_date")


def predict_aft_dates(model: CatBoostRegressor, rows: pd.DataFrame, use_station: bool | None = None) -> pd.Series:
    metadata = aft_training_metadata(model)
    saved_mode = metadata["use_station"]
    if use_station is not None and use_station != saved_mode:
        raise ValueError("Prediction feature mode differs from fitted AFT mode")
    entry = _prediction_entry(rows, metadata)
    if rows.empty:
        return pd.Series(index=rows.index, dtype="datetime64[ns]", name="predicted_event_date")
    features = feature_frame(rows, use_station=saved_mode)
    if features.columns.tolist() != model.feature_names_:
        raise ValueError("Model feature names do not match the saved input contract")
    # Explicit raw prediction avoids relying on loss-dependent defaults.
    raw = model.predict(features, prediction_type="RawFormulaVal")
    return _duration_dates(entry, raw)


def fit_calendar(train_registry: pd.DataFrame, outer_year: int, use_station: bool = True) -> dict:
    train, metadata = _training_rows(train_registry, outer_year)
    if not isinstance(use_station, bool):
        raise ValueError("use_station must be boolean")
    # The calendar is an event-timing comparator, not a survival likelihood.
    # Censored rows stay accounted for, but have no invented event date.
    observed = train[train._observed].copy()
    observed["_doy"] = observed.first_recorded_event_date.dt.dayofyear
    result = dict(metadata, model_kind="past_only_median_registration_DOY", use_station=use_station,
                  min_group_events=CALENDAR_MIN_GROUP_EVENTS,
                  global_median_doy=int(np.floor(observed._doy.median()+.5)), groups={})
    if use_station:
        # Validate the same known station input as the AFT candidate.
        feature_frame(train, use_station=True)
        for cluster, group in observed.groupby(observed.cluster.astype(str), sort=True):
            if len(group) >= CALENDAR_MIN_GROUP_EVENTS:
                result["groups"][cluster] = dict(events=len(group), median_doy=int(np.floor(group._doy.median()+.5)))
    return result


def predict_calendar_dates(calendar: dict, rows: pd.DataFrame) -> pd.Series:
    _prediction_entry(rows, calendar)
    if calendar["use_station"]:
        feature_frame(rows, use_station=True)
        days = [calendar["groups"].get(str(cluster), {}).get("median_doy", calendar["global_median_doy"])
                for cluster in rows.cluster]
    else:
        days = [calendar["global_median_doy"]] * len(rows)
    start = pd.Timestamp(year=int(calendar["outer_year"]), month=1, day=1)
    # Preserve the declared ordinal-day comparator; never clip dates to the
    # end of the observation window to make apparent success easier.
    return pd.Series([start+pd.Timedelta(days=int(d)-1) for d in days], index=rows.index,
                     dtype="datetime64[ns]", name="predicted_event_date")
