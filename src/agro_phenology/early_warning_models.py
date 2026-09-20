"""Models, notification policy, and event-level evaluation for early warning."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .early_warning_core import (
    CALENDAR_FEATURES,
    EPISODE_FEATURES,
    ERA_COMMON_FEATURES,
    NASA_COMMON_FEATURES,
)


TARGET_TO_INT = {"no_record_in_horizon": 0, "imminent": 1, "actionable": 2}
MODEL_SPECS = {
    "C0": {"kind": "logistic", "features": CALENDAR_FEATURES, "availability": None},
    "C1": {"kind": "catboost", "features": CALENDAR_FEATURES, "availability": None},
    "C2": {
        "kind": "catboost",
        "features": CALENDAR_FEATURES + NASA_COMMON_FEATURES,
        "availability": "nasa_common_complete",
    },
    "C3": {
        "kind": "catboost",
        "features": CALENDAR_FEATURES + ERA_COMMON_FEATURES,
        "availability": "era_common_complete",
    },
    "C4": {
        "kind": "catboost",
        "features": CALENDAR_FEATURES + EPISODE_FEATURES,
        "availability": "episode_weather_complete",
    },
    "C5": {
        "kind": "logistic",
        "features": CALENDAR_FEATURES + EPISODE_FEATURES,
        "availability": "episode_weather_complete",
    },
}
BASELINE_CODES = [
    "never",
    "constant_score_standard_policy",
    "always_on_alarm",
    "periodic_30d",
    "calendar_window",
    "hutton",
    "smith",
    "polyakov",
    "calendar_and_hutton",
]
ALWAYS_ON_ACTIVE_DAYS = 366
ALWAYS_ON_COOLDOWN_DAYS = 367
EVALUATION_SCOPES = {
    "service_calendar": None,
    "paired_candidate_days": "candidate_comparison_complete",
}


@dataclass(frozen=True)
class Policy:
    threshold: float
    active_days: int = 7
    cooldown_days: int = 15
    version: str = "policy_v1"


def _year_mask(frame: pd.DataFrame, bounds: list[int]) -> pd.Series:
    return frame["season"].between(int(bounds[0]), int(bounds[1]))


def _model_rows(frame: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Use one complete-case training population for the C0-C5 ablation."""
    complete = frame.get("candidate_comparison_complete", frame["common_weather_complete"])
    mask = (
        _year_mask(frame, years)
        & frame["target_observable"]
        & frame["service_active"]
        & complete
        & frame["target_class"].isin(TARGET_TO_INT)
    )
    return frame.loc[mask].copy()


def _make_model(kind: str, params: dict[str, Any], seed: int):
    if kind == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            loss_function="MultiClass",
            random_seed=seed,
            thread_count=2,
            verbose=False,
            allow_writing_files=False,
            **params,
        )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=float(params.get("C", 0.1)), max_iter=2000, random_state=seed),
    )


def _actionable_probability(model, values: pd.DataFrame) -> np.ndarray:
    probability = model.predict_proba(values)
    classes = [int(value) for value in model.classes_]
    if TARGET_TO_INT["actionable"] not in classes:
        return np.zeros(len(values), dtype=float)
    return probability[:, classes.index(TARGET_TO_INT["actionable"])]


def fit_model(
    frame: pd.DataFrame,
    features: list[str],
    kind: str,
    params: dict[str, Any],
    seed: int,
):
    y = frame["target_class"].map(TARGET_TO_INT).astype(int)
    if set(y.unique()) != set(TARGET_TO_INT.values()):
        raise ValueError(f"Training split does not contain all three target classes: {sorted(y.unique())}")
    model = _make_model(kind, params, seed)
    model.fit(frame[features], y)
    return model


def score_model(model, frame: pd.DataFrame, features: list[str], availability: str | None) -> pd.Series:
    score = pd.Series(np.nan, index=frame.index, dtype=float)
    mask = frame["service_active"].copy()
    if availability is not None:
        mask &= frame[availability]
    if mask.any():
        score.loc[mask] = _actionable_probability(model, frame.loc[mask, features])
    return score


def simulate_policy(
    frame: pd.DataFrame,
    score: pd.Series,
    policy: Policy,
    evaluation_scope: str = "service_calendar",
    evaluation_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Apply one stateful policy in chronological order for every field-season."""
    columns = [
        "field_season",
        "season",
        "issue_date",
        "issued_at",
        "service_active",
        "evaluation_field_day",
        "target_class",
        "target_observable",
        "days_to_first_recorded_event",
        "warnable_first_event",
        "coordinate_scope",
        "previous_visit_gap_days",
        "common_weather_complete",
    ]
    for optional in (
        "nasa_common_complete",
        "era_common_complete",
        "episode_weather_complete",
        "candidate_comparison_complete",
    ):
        if optional in frame and optional not in columns:
            columns.append(optional)
    result = frame[columns].copy()
    result["score"] = score.reindex(result.index)
    result["score_status"] = np.where(result["score"].notna(), "computed", "abstained")
    result["message_issued"] = False
    result["alarm_active"] = False
    result["active_from"] = pd.NaT
    result["active_through"] = pd.NaT
    result["forecast_window_start"] = result["issue_date"] + pd.Timedelta(days=3)
    result["forecast_window_end"] = result["issue_date"] + pd.Timedelta(days=10)
    result["action_reason"] = "below_threshold"
    result["suppressed_repeat"] = False
    # Keep chronological state transitions in Python, but avoid scalar pandas
    # reads/writes for every field-day and every candidate threshold. Positions
    # preserve the caller's index and row order; sorting matches the former loop.
    ordering = result[["field_season", "issue_date"]].reset_index(drop=True)
    days = [pd.Timestamp(value) for value in result["issue_date"]]
    service_active = result["service_active"].to_numpy()
    values = result["score"].to_numpy()
    messages = np.zeros(len(result), dtype=bool)
    alarms = np.zeros(len(result), dtype=bool)
    suppressed = np.zeros(len(result), dtype=bool)
    reasons = np.full(len(result), "below_threshold", dtype=object)
    starts = np.full(len(result), pd.NaT, dtype=object)
    ends = np.full(len(result), pd.NaT, dtype=object)
    active_duration = pd.Timedelta(days=policy.active_days - 1)
    for _, indexes in ordering.groupby("field_season", sort=False).groups.items():
        ordered = ordering.loc[indexes].sort_values("issue_date").index
        last_message: pd.Timestamp | None = None
        active_from: pd.Timestamp | None = None
        active_through: pd.Timestamp | None = None
        for index in ordered:
            day = days[index]
            if not bool(service_active[index]):
                active_from = None
                active_through = None
                reasons[index] = "stopped_after_record_available"
                continue
            value = values[index]
            if pd.isna(value):
                reasons[index] = "abstained_missing_input"
            elif float(value) >= policy.threshold:
                cooldown_ok = last_message is None or (day - last_message).days >= policy.cooldown_days
                if cooldown_ok:
                    messages[index] = True
                    reasons[index] = "issued_threshold_crossing_or_refresh"
                    last_message = day
                    active_from = day
                    active_through = day + active_duration
                else:
                    reasons[index] = "suppressed_cooldown"
                    suppressed[index] = True
            if active_through is not None and day <= active_through:
                alarms[index] = True
                starts[index] = active_from
                ends[index] = active_through
    result["message_issued"] = messages
    result["alarm_active"] = alarms
    result["action_reason"] = reasons
    result["suppressed_repeat"] = suppressed
    active_positions = np.flatnonzero(alarms)
    if len(active_positions):
        result.iloc[active_positions, result.columns.get_loc("active_from")] = starts[alarms].tolist()
        result.iloc[active_positions, result.columns.get_loc("active_through")] = ends[alarms].tolist()
    if evaluation_mask is None:
        evaluation_mask = result["service_active"]
    else:
        evaluation_mask = evaluation_mask.reindex(result.index).fillna(False) & result["service_active"]
    result["evaluation_scope"] = evaluation_scope
    result["evaluation_scope_day"] = evaluation_mask.astype(bool)
    result["policy_threshold"] = policy.threshold
    result["policy_active_days"] = policy.active_days
    result["policy_cooldown_days"] = policy.cooldown_days
    result["policy_version"] = policy.version
    return result


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return float(centre - half), float(centre + half)


def event_metrics(
    states: pd.DataFrame,
    seasons: pd.DataFrame,
    model_code: str,
    fold_id: str,
    slice_name: str = "A_plus_B",
    evaluation_scope: str = "service_calendar",
) -> tuple[dict, list[dict]]:
    if slice_name == "direct_A":
        registry = seasons[seasons["coordinate_scope"].eq("A_direct")]
    elif slice_name == "prior_gap_le14":
        registry = seasons[seasons["previous_visit_gap_days"].le(14)]
    elif slice_name == "prior_gap_le21":
        registry = seasons[seasons["previous_visit_gap_days"].le(21)]
    else:
        registry = seasons
    registry = registry[registry["first_recorded_event_date"].notna()]
    hits: list[dict] = []
    timely = 0
    opportunity = 0
    computable_events = 0
    lead_values: list[int] = []
    for event in registry.itertuples(index=False):
        event_states = states[states["field_season"].eq(event.field_season)]
        messages = event_states[event_states["message_issued"]]
        leads = [
            int((pd.Timestamp(event.first_recorded_event_date) - pd.Timestamp(day)).days)
            for day in messages["issue_date"]
        ]
        timely_leads = [lead for lead in leads if 3 <= lead <= 10]
        is_opportunity = bool(event.warnable_first_event)
        hit = bool(timely_leads) if is_opportunity else False
        actionable_rows = event_states[event_states["days_to_first_recorded_event"].between(3, 10)]
        computable = bool(actionable_rows["score"].notna().any()) if is_opportunity else False
        opportunity += int(is_opportunity)
        timely += int(hit)
        computable_events += int(computable)
        if timely_leads:
            lead_values.append(max(timely_leads))
        hits.append(
            {
                "model_code": model_code,
                "fold_id": fold_id,
                "evaluation_scope": evaluation_scope,
                "slice": slice_name,
                "field_season": event.field_season,
                "season": int(event.season),
                "warnable_event": is_opportunity,
                "positive_at_entry": bool(event.positive_at_first_visit),
                "computable_in_actionable_window": computable,
                "timely_hit": hit,
                "timely_message_count": len(timely_leads),
                "declared_window_contains_event": hit,
                "timely_best_lead_days": max(timely_leads) if timely_leads else np.nan,
                "messages_before_event": len([lead for lead in leads if lead >= 0]),
                "too_early_messages": len([lead for lead in leads if lead > 10]),
                "late_messages": len([lead for lead in leads if lead < 3]),
            }
        )
    lower, upper = wilson_interval(timely, opportunity)
    return (
        {
            "model_code": model_code,
            "fold_id": fold_id,
            "evaluation_scope": evaluation_scope,
            "slice": slice_name,
            "first_events": int(len(registry)),
            "events_with_warning_opportunity": int(opportunity),
            "timely_hits": int(timely),
            "timely_recall": timely / opportunity if opportunity else np.nan,
            "timely_recall_wilson_low": lower,
            "timely_recall_wilson_high": upper,
            "coverage_all_first_events": timely / len(registry) if len(registry) else np.nan,
            "computable_events": int(computable_events),
            "computable_event_fraction": computable_events / opportunity if opportunity else np.nan,
            "median_best_timely_lead_days": float(np.median(lead_values)) if lead_values else np.nan,
        },
        hits,
    )


def burden_metrics(
    states: pd.DataFrame,
    model_code: str,
    fold_id: str,
    slice_name: str = "A_plus_B",
    evaluation_scope: str = "service_calendar",
) -> dict:
    eligible = states.get("evaluation_scope_day", states["service_active"])
    subset = states[eligible]
    if slice_name == "direct_A":
        subset = subset[subset["coordinate_scope"].eq("A_direct")]
    field_days = int(len(subset))
    messages = int(subset["message_issued"].sum())
    alarm_days = int(subset["alarm_active"].sum())
    season_count = int(subset["field_season"].nunique())
    counts_per_season = subset.groupby("field_season")["message_issued"].sum() if field_days else pd.Series(dtype=float)
    return {
        "model_code": model_code,
        "fold_id": fold_id,
        "evaluation_scope": evaluation_scope,
        "slice": slice_name,
        "field_days": field_days,
        "field_seasons": season_count,
        "messages": messages,
        "messages_per_30_field_days": 30 * messages / field_days if field_days else np.nan,
        "messages_per_field_season": messages / season_count if season_count else np.nan,
        "active_alarm_days": alarm_days,
        "active_alarm_fraction": alarm_days / field_days if field_days else np.nan,
        "computable_days": int(subset["score"].notna().sum()),
        "computable_fraction": float(subset["score"].notna().mean()) if field_days else np.nan,
        "abstention_days": int(subset["score"].isna().sum()),
        "suppressed_repeats": int(subset["suppressed_repeat"].sum()),
        "messages_p95_per_field_season": float(counts_per_season.quantile(0.95)) if len(counts_per_season) else np.nan,
        "messages_max_per_field_season": int(counts_per_season.max()) if len(counts_per_season) else 0,
    }


def _policy_summary(states: pd.DataFrame, seasons: pd.DataFrame, evaluation_scope: str) -> tuple[dict, dict]:
    event, _ = event_metrics(states, seasons, "candidate", "validation", evaluation_scope=evaluation_scope)
    burden = burden_metrics(states, "candidate", "validation", evaluation_scope=evaluation_scope)
    return event, burden


def _pick_threshold_record(records: list[dict], max_messages_per_30: float, max_alarm_fraction: float) -> tuple[dict, bool]:
    annotated: list[dict] = []
    for source in records:
        record = dict(source)
        feasible = bool(
            record["messages_per_30_field_days"] <= max_messages_per_30
            and record["active_alarm_fraction"] <= max_alarm_fraction
        )
        record["feasible"] = feasible
        record["constraint_violation"] = max(
            0.0, record["messages_per_30_field_days"] - max_messages_per_30
        ) + max(0.0, record["active_alarm_fraction"] - max_alarm_fraction)
        annotated.append(record)
    feasible_records = [record for record in annotated if record["feasible"]]
    pool = feasible_records or annotated
    best = sorted(
        pool,
        key=lambda record: (
            -float(np.nan_to_num(record["timely_recall"], nan=-1)),
            float(record["messages_per_30_field_days"]),
            float(record["active_alarm_fraction"]),
            int(record["suppressed_repeats"]),
            -float(record["threshold"]),
        ),
    )[0]
    return best, bool(feasible_records)


def select_threshold(
    frame: pd.DataFrame,
    score: pd.Series,
    seasons: pd.DataFrame,
    active_days: int,
    cooldown_days: int,
    max_messages_per_30: float,
    max_alarm_fraction: float,
    evaluation_scope: str = "service_calendar",
    evaluation_mask: pd.Series | None = None,
) -> tuple[Policy, dict]:
    scoped_score = score.copy()
    if evaluation_mask is not None:
        scoped_score = scoped_score.where(evaluation_mask.reindex(scoped_score.index).fillna(False))
    finite = scoped_score[scoped_score.notna()]
    candidates = set(np.linspace(0.05, 0.95, 19).tolist() + [1.000001])
    if len(finite):
        candidates.update(float(value) for value in finite.quantile(np.linspace(0.05, 0.95, 19)).unique())
    records: list[dict] = []
    for threshold in sorted(candidates):
        policy = Policy(float(threshold), active_days, cooldown_days, f"policy_v1_{evaluation_scope}")
        states = simulate_policy(frame, scoped_score, policy, evaluation_scope, evaluation_mask)
        event, burden = _policy_summary(states, seasons, evaluation_scope)
        records.append({**event, **burden, "threshold": threshold})
    best, any_feasible = _pick_threshold_record(records, max_messages_per_30, max_alarm_fraction)
    return Policy(
        float(best["threshold"]), active_days, cooldown_days, f"policy_v1_{evaluation_scope}"
    ), {"selected": best, "candidates": records, "any_feasible": any_feasible}


def _fixed_catboost_params(contract: dict) -> dict:
    fixed = contract["catboost_fixed"]
    return {
        "iterations": int(fixed["iterations"]),
        "depth": int(fixed["depth"]),
        "learning_rate": float(fixed["learning_rate"]),
        "l2_leaf_reg": float(fixed["l2_leaf_reg"]),
    }


def _save_and_verify_model(model, kind: str, path: Path, sample: pd.DataFrame) -> tuple[float, str]:
    if kind == "catboost":
        from catboost import CatBoostClassifier

        model.save_model(path)
        loaded = CatBoostClassifier()
        loaded.load_model(path)
    else:
        joblib.dump(model, path)
        loaded = joblib.load(path)
    if sample.empty:
        return np.nan, "serialized_only_no_computable_sample"
    before = _actionable_probability(model, sample)
    after = _actionable_probability(loaded, sample)
    return float(np.max(np.abs(before - after))), "prediction_roundtrip_checked"


def _baseline_scores(frame: pd.DataFrame, seasons_fit: pd.DataFrame) -> dict[str, pd.Series]:
    first_events = seasons_fit.loc[
        seasons_fit["warnable_first_event"], "first_recorded_event_date"
    ].dropna()
    if len(first_events):
        doy = pd.to_datetime(first_events).dt.dayofyear
        lower, upper = int(doy.quantile(0.05)), int(doy.quantile(0.95))
    else:
        lower, upper = 182, 243
    issue_doy = frame["issue_date"].dt.dayofyear
    calendar = issue_doy.between(lower, upper).astype(float)
    periodic = (
        frame["issue_date"].dt.month.isin([6, 7, 8]) & frame["issue_date"].dt.day.eq(1)
    ).astype(float)
    hutton = frame["hutton_score"].astype(float)
    smith = frame["smith_score"].astype(float)
    return {
        "never": pd.Series(0.0, index=frame.index),
        "constant_score_standard_policy": pd.Series(1.0, index=frame.index),
        "always_on_alarm": pd.Series(1.0, index=frame.index),
        "periodic_30d": periodic,
        "calendar_window": calendar,
        "hutton": hutton,
        "smith": smith,
        "polyakov": frame["polyakov_score"].astype(float),
        "calendar_and_hutton": (calendar.eq(1) & hutton.eq(1)).astype(float).where(hutton.notna()),
    }


def _baseline_policy(model_code: str, active_days: int, cooldown_days: int) -> Policy:
    if model_code == "always_on_alarm":
        return Policy(
            0.5,
            ALWAYS_ON_ACTIVE_DAYS,
            ALWAYS_ON_COOLDOWN_DAYS,
            "policy_v1_always_on_alarm",
        )
    return Policy(0.5, active_days, cooldown_days, "policy_v1_fixed_binary_rule")


def _optuna_params(trial, contract: dict) -> dict:
    space = contract["optuna"]["search_space"]
    return {
        "depth": trial.suggest_int("depth", int(space["depth"][0]), int(space["depth"][1])),
        "iterations": trial.suggest_int(
            "iterations", int(space["iterations"][0]), int(space["iterations"][1]), step=20
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", float(space["learning_rate"][0]), float(space["learning_rate"][1]), log=True
        ),
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg", float(space["l2_leaf_reg"][0]), float(space["l2_leaf_reg"][1]), log=True
        ),
    }


def tune_c4(
    train: pd.DataFrame,
    validation_service: pd.DataFrame,
    validation_seasons: pd.DataFrame,
    contract: dict,
    storage: Path,
    study_name: str,
    target_trial_count: int,
    seed: int,
) -> tuple[dict, Any]:
    """Bring a persistent study to a fixed total trial count, never add blindly."""
    import optuna

    policy_settings = contract["notification_policy"]
    storage.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=f"sqlite:///{storage.resolve()}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    def objective(trial):
        params = _optuna_params(trial, contract)
        model = fit_model(train, MODEL_SPECS["C4"]["features"], "catboost", params, seed)
        score = score_model(
            model, validation_service, MODEL_SPECS["C4"]["features"], "episode_weather_complete"
        )
        policy, details = select_threshold(
            validation_service,
            score,
            validation_seasons,
            int(policy_settings["active_days_per_message"]),
            int(policy_settings["cooldown_days"]),
            float(policy_settings["research_budget"]["messages_per_30_field_days_max"]),
            float(policy_settings["research_budget"]["active_alarm_fraction_max"]),
        )
        selected = details["selected"]
        recall = float(np.nan_to_num(selected["timely_recall"], nan=0.0))
        utility = recall - 0.001 * float(selected["messages_per_30_field_days"]) - 0.0001 * float(
            selected["active_alarm_fraction"]
        )
        trial.set_user_attr("policy_threshold", policy.threshold)
        trial.set_user_attr("timely_recall", selected["timely_recall"])
        trial.set_user_attr("messages_per_30_field_days", selected["messages_per_30_field_days"])
        trial.set_user_attr("active_alarm_fraction", selected["active_alarm_fraction"])
        trial.set_user_attr("policy_feasible", details["any_feasible"])
        trial.set_user_attr("objective_utility", utility)
        return utility

    remaining = max(0, int(target_trial_count) - len(study.trials))
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, gc_after_trial=True)
    if not study.trials or study.best_trial.value is None:
        raise RuntimeError(f"Optuna study {study_name} has no completed trial")
    return dict(study.best_trial.params), study


def _trial_frame(study, fold_id: str) -> pd.DataFrame:
    rows = []
    for trial in study.trials:
        rows.append(
            {
                "fold_id": fold_id,
                "number": trial.number,
                "state": str(trial.state.name),
                "value": trial.value,
                "params": json.dumps(trial.params, sort_keys=True),
                "user_attrs": json.dumps(trial.user_attrs, sort_keys=True),
                "datetime_start": trial.datetime_start,
                "datetime_complete": trial.datetime_complete,
            }
        )
    return pd.DataFrame(rows)


def daily_diagnostic(frame: pd.DataFrame, model_code: str, fold_id: str) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    complete = frame.get("candidate_comparison_complete", frame["common_weather_complete"])
    subset = frame[
        frame["target_observable"]
        & complete
        & frame["score"].notna()
        & frame["service_active"]
    ]
    y = subset["target_class"].eq("actionable").astype(int)
    score = subset["score"].astype(float).clip(1e-9, 1 - 1e-9)
    result = {
        "model_code": model_code,
        "fold_id": fold_id,
        "rows": int(len(subset)),
        "field_seasons": int(subset["field_season"].nunique()),
        "actionable_rows": int(y.sum()),
        "average_precision": np.nan,
        "auroc": np.nan,
        "brier": np.nan,
        "binary_log_loss": np.nan,
        "warning": "dependent_daily_rows_diagnostic_only",
    }
    if len(subset):
        result["brier"] = float(brier_score_loss(y, score))
        result["binary_log_loss"] = float(
            log_loss(y, np.column_stack([1 - score, score]), labels=[0, 1])
        )
        if y.nunique() == 2:
            result["average_precision"] = float(average_precision_score(y, score))
            result["auroc"] = float(roc_auc_score(y, score))
    return result


def _append_evaluation(
    outputs: dict[str, list],
    test_service: pd.DataFrame,
    test_seasons: pd.DataFrame,
    score: pd.Series,
    policy: Policy,
    model_code: str,
    fold_id: str,
    model_version: str,
    feature_provenance: str,
    evaluation_scope: str,
    mask_column: str | None,
) -> pd.DataFrame:
    mask = test_service[mask_column] if mask_column else None
    scoped_score = score.where(mask) if mask is not None else score
    states = simulate_policy(test_service, scoped_score, policy, evaluation_scope, mask)
    states["model_code"] = model_code
    states["fold_id"] = fold_id
    states["model_version"] = model_version
    outputs["alarm_states"].append(states)
    prediction_columns = [
        "field_season",
        "season",
        "issue_date",
        "issued_at",
        "score",
        "score_status",
        "target_class",
        "target_observable",
        "evaluation_scope",
        "evaluation_scope_day",
        "common_weather_complete",
    ]
    prediction_columns.extend(
        column
        for column in (
            "nasa_common_complete",
            "era_common_complete",
            "episode_weather_complete",
            "candidate_comparison_complete",
        )
        if column in states
    )
    prediction = states[prediction_columns].copy()
    prediction["model_code"] = model_code
    prediction["fold_id"] = fold_id
    prediction["feature_cutoff_provenance"] = feature_provenance
    outputs["predictions"].append(prediction)
    for slice_name in ("A_plus_B", "direct_A", "prior_gap_le14", "prior_gap_le21"):
        event, hits = event_metrics(
            states, test_seasons, model_code, fold_id, slice_name, evaluation_scope
        )
        outputs["event_metrics"].append(event)
        outputs["event_hits"].extend(hits)
    for slice_name in ("A_plus_B", "direct_A"):
        outputs["burden_metrics"].append(
            burden_metrics(states, model_code, fold_id, slice_name, evaluation_scope)
        )
    return states


def _append_budget_grid(
    outputs: dict[str, list],
    details: dict,
    test_service: pd.DataFrame,
    test_seasons: pd.DataFrame,
    test_score: pd.Series,
    model_code: str,
    fold_id: str,
    active_days: int,
    cooldown_days: int,
    contract: dict,
) -> None:
    grid = contract["notification_policy"]["reported_budget_grid"]
    for message_budget in grid["messages_per_30_field_days"]:
        for alarm_budget in grid["active_alarm_fraction"]:
            selected, feasible = _pick_threshold_record(
                details["candidates"], float(message_budget), float(alarm_budget)
            )
            policy = Policy(
                float(selected["threshold"]), active_days, cooldown_days, "policy_v1_budget_grid"
            )
            states = simulate_policy(test_service, test_score, policy)
            event, _ = event_metrics(states, test_seasons, model_code, fold_id)
            burden = burden_metrics(states, model_code, fold_id)
            outputs["budget_grid_metrics"].append(
                {
                    "model_code": model_code,
                    "fold_id": fold_id,
                    "messages_budget_per_30": float(message_budget),
                    "alarm_fraction_budget": float(alarm_budget),
                    "threshold": policy.threshold,
                    "validation_feasible": feasible,
                    "validation_timely_recall": selected["timely_recall"],
                    "validation_messages_per_30": selected["messages_per_30_field_days"],
                    "validation_alarm_fraction": selected["active_alarm_fraction"],
                    "test_events": event["events_with_warning_opportunity"],
                    "test_timely_hits": event["timely_hits"],
                    "test_timely_recall": event["timely_recall"],
                    "test_field_days": burden["field_days"],
                    "test_messages": burden["messages"],
                    "test_messages_per_30": burden["messages_per_30_field_days"],
                    "test_alarm_fraction": burden["active_alarm_fraction"],
                }
            )


def run_experiments(
    decisions: pd.DataFrame,
    seasons: pd.DataFrame,
    contract: dict,
    run_dir: str | Path,
    smoke: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fit fixed C0-C5 first, then the bounded persistent C4 Optuna study."""
    run_path = Path(run_dir)
    model_dir = run_path / "models"
    optuna_dir = run_path / "optuna"
    model_dir.mkdir(parents=True, exist_ok=True)
    optuna_dir.mkdir(parents=True, exist_ok=True)
    settings = contract["notification_policy"]
    active_days = int(settings["active_days_per_message"])
    cooldown_days = int(settings["cooldown_days"])
    max_messages = float(settings["research_budget"]["messages_per_30_field_days_max"])
    max_alarm = float(settings["research_budget"]["active_alarm_fraction_max"])
    base_seed = int(contract["random_seed"])
    fixed_params = _fixed_catboost_params(contract)
    fixed_params["iterations"] = min(fixed_params["iterations"], 60) if smoke else fixed_params["iterations"]
    outputs: dict[str, list] = {
        "predictions": [],
        "alarm_states": [],
        "event_metrics": [],
        "event_hits": [],
        "burden_metrics": [],
        "daily_diagnostics": [],
        "policy_selection": [],
        "model_reload_verification": [],
        "optuna_trials": [],
        "optuna_seed_checks": [],
        "budget_grid_metrics": [],
    }

    folds = contract["rolling_origin_folds"][:1] if smoke else contract["rolling_origin_folds"]
    for fold_index, fold in enumerate(folds):
        fold_id = fold["id"]
        train = _model_rows(decisions, fold["train_years"])
        validation_service = decisions[_year_mask(decisions, fold["validation_years"])].copy()
        test_service = decisions[_year_mask(decisions, fold["test_years"])].copy()
        validation_seasons = seasons[_year_mask(seasons, fold["validation_years"])].copy()
        test_seasons = seasons[_year_mask(seasons, fold["test_years"])].copy()
        train_seasons = seasons[_year_mask(seasons, fold["train_years"])].copy()
        if train.empty or validation_service.empty or test_service.empty:
            raise ValueError(f"Fold {fold_id} has an empty train, validation, or test block")

        # Fixed C0-C5 complete before the first Optuna trial.
        for model_offset, (model_code, spec) in enumerate(MODEL_SPECS.items()):
            params = fixed_params if spec["kind"] == "catboost" else {"C": 0.1}
            model_seed = base_seed + fold_index * 20 + model_offset
            model = fit_model(train, spec["features"], spec["kind"], params, model_seed)
            validation_score = score_model(model, validation_service, spec["features"], spec["availability"])
            test_score = score_model(model, test_service, spec["features"], spec["availability"])
            suffix = ".cbm" if spec["kind"] == "catboost" else ".joblib"
            sample = test_service.loc[test_score.notna(), spec["features"]].head(50)
            if sample.empty:
                sample = train[spec["features"]].head(50)
            difference, verification_status = _save_and_verify_model(
                model, spec["kind"], model_dir / f"{fold_id}_{model_code}{suffix}", sample
            )
            outputs["model_reload_verification"].append(
                {
                    "model_code": model_code,
                    "fold_id": fold_id,
                    "status": verification_status,
                    "saved_reload_max_abs_score_difference": difference,
                }
            )
            outputs["daily_diagnostics"].append(
                daily_diagnostic(test_service.assign(score=test_score), model_code, fold_id)
            )
            for evaluation_scope, mask_column in EVALUATION_SCOPES.items():
                validation_mask = validation_service[mask_column] if mask_column else None
                policy, details = select_threshold(
                    validation_service,
                    validation_score,
                    validation_seasons,
                    active_days,
                    cooldown_days,
                    max_messages,
                    max_alarm,
                    evaluation_scope,
                    validation_mask,
                )
                _append_evaluation(
                    outputs,
                    test_service,
                    test_seasons,
                    test_score,
                    policy,
                    model_code,
                    fold_id,
                    "fixed_train_only_v1",
                    spec["availability"] or "calendar_only",
                    evaluation_scope,
                    mask_column,
                )
                selected = details["selected"]
                outputs["policy_selection"].append(
                    {
                        "model_code": model_code,
                        "fold_id": fold_id,
                        "evaluation_scope": evaluation_scope,
                        "threshold": policy.threshold,
                        "active_days": policy.active_days,
                        "cooldown_days": policy.cooldown_days,
                        "validation_policy_feasible": details["any_feasible"],
                        "validation_timely_recall": selected["timely_recall"],
                        "validation_messages_per_30": selected["messages_per_30_field_days"],
                        "validation_alarm_fraction": selected["active_alarm_fraction"],
                        "model_params": json.dumps(params, sort_keys=True),
                        "model_seed": model_seed,
                        "fit_population": "train_only_common_complete_cases",
                    }
                )
                if evaluation_scope == "service_calendar":
                    _append_budget_grid(
                        outputs,
                        details,
                        test_service,
                        test_seasons,
                        test_score,
                        model_code,
                        fold_id,
                        active_days,
                        cooldown_days,
                        contract,
                    )

        # Deterministic controls: validation feasibility is measured, not assumed.
        validation_baselines = _baseline_scores(validation_service, train_seasons)
        test_baselines = _baseline_scores(test_service, train_seasons)
        for model_code, test_score in test_baselines.items():
            validation_score = validation_baselines[model_code]
            policy = _baseline_policy(model_code, active_days, cooldown_days)
            for evaluation_scope, mask_column in EVALUATION_SCOPES.items():
                validation_mask = validation_service[mask_column] if mask_column else None
                scoped_validation = (
                    validation_score.where(validation_mask) if validation_mask is not None else validation_score
                )
                validation_states = simulate_policy(
                    validation_service, scoped_validation, policy, evaluation_scope, validation_mask
                )
                validation_event, validation_burden = _policy_summary(
                    validation_states, validation_seasons, evaluation_scope
                )
                validation_feasible = bool(
                    validation_burden["messages_per_30_field_days"] <= max_messages
                    and validation_burden["active_alarm_fraction"] <= max_alarm
                )
                _append_evaluation(
                    outputs,
                    test_service,
                    test_seasons,
                    test_score,
                    policy,
                    model_code,
                    fold_id,
                    "deterministic_baseline_v1",
                    "deterministic_rule",
                    evaluation_scope,
                    mask_column,
                )
                outputs["policy_selection"].append(
                    {
                        "model_code": model_code,
                        "fold_id": fold_id,
                        "evaluation_scope": evaluation_scope,
                        "threshold": policy.threshold,
                        "active_days": policy.active_days,
                        "cooldown_days": policy.cooldown_days,
                        "validation_policy_feasible": validation_feasible,
                        "validation_timely_recall": validation_event["timely_recall"],
                        "validation_messages_per_30": validation_burden["messages_per_30_field_days"],
                        "validation_alarm_fraction": validation_burden["active_alarm_fraction"],
                        "model_params": "{}",
                        "model_seed": np.nan,
                        "fit_population": "fixed_rule_calendar_bounds_from_train_only",
                    }
                )

        # Bounded C4 search starts only after every fixed candidate completed.
        trial_target = int(
            contract["optuna"]["smoke_trials_per_fold"] if smoke else contract["optuna"]["trials_per_fold"]
        )
        optuna_seed = base_seed + 1000 + fold_index
        best_params, study = tune_c4(
            train,
            validation_service,
            validation_seasons,
            contract,
            optuna_dir / f"{fold_id}.sqlite3",
            f"C4_{fold_id}",
            trial_target,
            optuna_seed,
        )
        outputs["optuna_trials"].append(_trial_frame(study, fold_id))
        tuned_models: dict[int, Any] = {}
        for seed_offset in range(3):
            sensitivity_seed = optuna_seed + seed_offset
            sensitivity_model = fit_model(
                train, MODEL_SPECS["C4"]["features"], "catboost", best_params, sensitivity_seed
            )
            tuned_models[sensitivity_seed] = sensitivity_model
            sensitivity_score = score_model(
                sensitivity_model,
                validation_service,
                MODEL_SPECS["C4"]["features"],
                MODEL_SPECS["C4"]["availability"],
            )
            sensitivity_policy, sensitivity_details = select_threshold(
                validation_service,
                sensitivity_score,
                validation_seasons,
                active_days,
                cooldown_days,
                max_messages,
                max_alarm,
            )
            chosen = sensitivity_details["selected"]
            outputs["optuna_seed_checks"].append(
                {
                    "fold_id": fold_id,
                    "seed": sensitivity_seed,
                    "primary_seed": sensitivity_seed == optuna_seed,
                    "threshold": sensitivity_policy.threshold,
                    "validation_feasible": sensitivity_details["any_feasible"],
                    "validation_timely_recall": chosen["timely_recall"],
                    "validation_messages_per_30": chosen["messages_per_30_field_days"],
                    "validation_alarm_fraction": chosen["active_alarm_fraction"],
                    "params": json.dumps(best_params, sort_keys=True),
                }
            )
        tuned = tuned_models[optuna_seed]
        tuned_validation_score = score_model(
            tuned,
            validation_service,
            MODEL_SPECS["C4"]["features"],
            MODEL_SPECS["C4"]["availability"],
        )
        tuned_test_score = score_model(
            tuned,
            test_service,
            MODEL_SPECS["C4"]["features"],
            MODEL_SPECS["C4"]["availability"],
        )
        sample = test_service.loc[tuned_test_score.notna(), MODEL_SPECS["C4"]["features"]].head(50)
        if sample.empty:
            sample = train[MODEL_SPECS["C4"]["features"]].head(50)
        difference, verification_status = _save_and_verify_model(
            tuned, "catboost", model_dir / f"{fold_id}_C4_optuna.cbm", sample
        )
        outputs["model_reload_verification"].append(
            {
                "model_code": "C4_optuna",
                "fold_id": fold_id,
                "status": verification_status,
                "saved_reload_max_abs_score_difference": difference,
            }
        )
        outputs["daily_diagnostics"].append(
            daily_diagnostic(test_service.assign(score=tuned_test_score), "C4_optuna", fold_id)
        )
        for evaluation_scope, mask_column in EVALUATION_SCOPES.items():
            validation_mask = validation_service[mask_column] if mask_column else None
            policy, details = select_threshold(
                validation_service,
                tuned_validation_score,
                validation_seasons,
                active_days,
                cooldown_days,
                max_messages,
                max_alarm,
                evaluation_scope,
                validation_mask,
            )
            _append_evaluation(
                outputs,
                test_service,
                test_seasons,
                tuned_test_score,
                policy,
                "C4_optuna",
                fold_id,
                "optuna_train_only_primary_seed_v1",
                "episode_weather_complete",
                evaluation_scope,
                mask_column,
            )
            selected = details["selected"]
            outputs["policy_selection"].append(
                {
                    "model_code": "C4_optuna",
                    "fold_id": fold_id,
                    "evaluation_scope": evaluation_scope,
                    "threshold": policy.threshold,
                    "active_days": policy.active_days,
                    "cooldown_days": policy.cooldown_days,
                    "validation_policy_feasible": details["any_feasible"],
                    "validation_timely_recall": selected["timely_recall"],
                    "validation_messages_per_30": selected["messages_per_30_field_days"],
                    "validation_alarm_fraction": selected["active_alarm_fraction"],
                    "model_params": json.dumps(best_params, sort_keys=True),
                    "model_seed": optuna_seed,
                    "fit_population": "train_only_common_complete_cases",
                }
            )
            if evaluation_scope == "service_calendar":
                _append_budget_grid(
                    outputs,
                    details,
                    test_service,
                    test_seasons,
                    tuned_test_score,
                    "C4_optuna",
                    fold_id,
                    active_days,
                    cooldown_days,
                    contract,
                )

    result: dict[str, pd.DataFrame] = {}
    for name, values in outputs.items():
        if name == "event_hits":
            result[name] = pd.DataFrame(values)
        elif name == "optuna_trials":
            result[name] = pd.concat(values, ignore_index=True) if values else pd.DataFrame()
        elif name in {"predictions", "alarm_states"}:
            result[name] = pd.concat(values, ignore_index=True) if values else pd.DataFrame()
        else:
            result[name] = pd.DataFrame(values)
    return result
