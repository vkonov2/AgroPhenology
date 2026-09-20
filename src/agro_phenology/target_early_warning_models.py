"""Model matrix and sequential replay for configurable orchard targets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .early_warning_models import (
    Policy,
    TARGET_TO_INT,
    burden_metrics,
    event_metrics,
    fit_model,
    score_model,
    simulate_policy,
)
from .target_early_warning_core import (
    CALENDAR_FEATURES,
    CODLING_FEATURES,
    COMMON_WEATHER_FEATURES,
    SCAB_FEATURES,
)


def model_specs(target_key: str) -> dict[str, dict[str, Any]]:
    if target_key == "codling_moth":
        target_features = CODLING_FEATURES
    elif target_key == "apple_scab":
        target_features = SCAB_FEATURES
    else:
        raise ValueError(f"Unsupported target key: {target_key}")
    return {
        "O0": {"kind": "logistic", "features": CALENDAR_FEATURES, "availability": None},
        "O1": {"kind": "catboost", "features": CALENDAR_FEATURES, "availability": None},
        "O2": {
            "kind": "catboost",
            "features": CALENDAR_FEATURES + COMMON_WEATHER_FEATURES,
            "availability": "common_weather_complete",
        },
        "O3": {
            "kind": "catboost",
            "features": CALENDAR_FEATURES + COMMON_WEATHER_FEATURES + target_features,
            "availability": "target_feature_weather_complete",
        },
        "O4": {
            "kind": "logistic",
            "features": CALENDAR_FEATURES + COMMON_WEATHER_FEATURES + target_features,
            "availability": "target_feature_weather_complete",
        },
    }


def _year_mask(frame: pd.DataFrame, bounds: list[int]) -> pd.Series:
    return frame["season"].between(int(bounds[0]), int(bounds[1]))


def _training_rows(frame: pd.DataFrame, bounds: list[int]) -> pd.DataFrame:
    return frame.loc[
        _year_mask(frame, bounds)
        & frame["target_observable"]
        & frame["service_active"]
        & frame["candidate_comparison_complete"]
        & frame["target_class"].isin(TARGET_TO_INT)
    ].copy()


def _save_and_verify(model, kind: str, path: Path, sample: pd.DataFrame) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "catboost":
        from catboost import CatBoostClassifier

        model.save_model(path)
        loaded = CatBoostClassifier()
        loaded.load_model(path)
    else:
        joblib.dump(model, path)
        loaded = joblib.load(path)
    if sample.empty:
        return float("nan")
    before = model.predict_proba(sample)
    after = loaded.predict_proba(sample)
    return float(np.max(np.abs(before - after)))


def _policy_summary(
    frame: pd.DataFrame,
    seasons: pd.DataFrame,
    score: pd.Series,
    policy: Policy,
) -> dict[str, Any]:
    states = simulate_policy(frame, score, policy)
    event, _ = event_metrics(states, seasons, "candidate", "validation")
    burden = burden_metrics(states, "candidate", "validation")
    return {**event, **burden, "threshold": policy.threshold}


def select_policy_fixed_grid(
    frame: pd.DataFrame,
    seasons: pd.DataFrame,
    score: pd.Series,
    contract: dict,
) -> tuple[Policy, dict[str, Any]]:
    """Choose a threshold only on internal validation under fixed load limits."""
    policy_cfg = contract["notification_policy"]
    candidates = [float(value) for value in policy_cfg["threshold_grid"]]
    if 1.000001 not in candidates:
        candidates.append(1.000001)
    rows: list[dict[str, Any]] = []
    for threshold in sorted(set(candidates)):
        policy = Policy(
            threshold,
            int(policy_cfg["active_days_per_message"]),
            int(policy_cfg["cooldown_days"]),
            "orchard_policy_v1_validation_grid",
        )
        rows.append(_policy_summary(frame, seasons, score, policy))
    max_messages = float(policy_cfg["research_budget"]["messages_per_30_field_days_max"])
    max_alarm = float(policy_cfg["research_budget"]["active_alarm_fraction_max"])
    for row in rows:
        row["feasible"] = bool(
            row["messages_per_30_field_days"] <= max_messages
            and row["active_alarm_fraction"] <= max_alarm
        )
        row["constraint_violation"] = max(
            0.0, row["messages_per_30_field_days"] - max_messages
        ) + max(0.0, row["active_alarm_fraction"] - max_alarm)
    feasible = [row for row in rows if row["feasible"]]
    pool = feasible or rows
    selected = sorted(
        pool,
        key=lambda row: (
            -float(np.nan_to_num(row["timely_recall"], nan=-1.0)),
            float(row["messages_per_30_field_days"]),
            float(row["active_alarm_fraction"]),
            -float(row["threshold"]),
        ),
    )[0]
    return (
        Policy(
            float(selected["threshold"]),
            int(policy_cfg["active_days_per_message"]),
            int(policy_cfg["cooldown_days"]),
            "orchard_policy_v1_validation_grid",
        ),
        {"selected": selected, "candidates": rows, "any_feasible": bool(feasible)},
    )


def _policy_grid_row(
    fold_id: str,
    model_code: str,
    candidate: dict[str, Any],
    selected_threshold: float,
) -> dict[str, Any]:
    """Attach the actual fitted model to an auditable threshold candidate."""
    row = dict(candidate)
    row["evaluation_fold_id"] = row.get("fold_id")
    row["evaluation_model_code"] = row.get("model_code")
    row["fold_id"] = fold_id
    row["model_code"] = model_code
    row["selected"] = bool(np.isclose(candidate["threshold"], selected_threshold))
    return row


def _calendar_scores(
    frame: pd.DataFrame, training_seasons: pd.DataFrame
) -> tuple[dict[str, pd.Series], dict[str, int]]:
    events = training_seasons.loc[
        training_seasons["warnable_first_event"], "first_recorded_event_date"
    ].dropna()
    if len(events):
        doy = pd.to_datetime(events).dt.dayofyear
        lower = int(np.floor(doy.quantile(0.05)))
        upper = int(np.ceil(doy.quantile(0.95)))
    else:
        lower, upper = 91, 304
    issue_doy = frame["issue_date"].dt.dayofyear
    return (
        {
            "calendar_window": issue_doy.between(lower, upper).astype(float),
            "periodic_30d": frame["issue_date"].dt.day.eq(1).astype(float),
            "never": pd.Series(0.0, index=frame.index),
        },
        {"calendar_window_start_doy": lower, "calendar_window_end_doy": upper},
    )


def _append_evaluation(
    outputs: dict[str, list],
    *,
    test: pd.DataFrame,
    test_seasons: pd.DataFrame,
    score: pd.Series,
    policy: Policy,
    model_code: str,
    fold_id: str,
    model_version: str,
    feature_provenance: str,
    evaluation_scope: str,
    evaluation_mask: pd.Series | None,
) -> None:
    scoped_score = score.where(evaluation_mask) if evaluation_mask is not None else score
    states = simulate_policy(
        test,
        scoped_score,
        policy,
        evaluation_scope=evaluation_scope,
        evaluation_mask=evaluation_mask,
    )
    states["model_code"] = model_code
    states["fold_id"] = fold_id
    states["model_version"] = model_version
    outputs["alarm_states"].append(states)

    predictions = states[
        [
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
            "candidate_comparison_complete",
        ]
    ].copy()
    predictions["model_code"] = model_code
    predictions["fold_id"] = fold_id
    predictions["feature_cutoff_provenance"] = feature_provenance
    outputs["predictions"].append(predictions)

    for slice_name in ("A_plus_B", "direct_A", "prior_gap_le14", "prior_gap_le21"):
        event, hits = event_metrics(
            states,
            test_seasons,
            model_code,
            fold_id,
            slice_name,
            evaluation_scope,
        )
        event["season"] = int(test_seasons["season"].iloc[0]) if len(test_seasons) else np.nan
        outputs["event_metrics"].append(event)
        for hit in hits:
            hit["season"] = int(hit["season"])
        outputs["event_hits"].extend(hits)
    for slice_name in ("A_plus_B", "direct_A"):
        burden = burden_metrics(states, model_code, fold_id, slice_name, evaluation_scope)
        burden["season"] = int(test_seasons["season"].iloc[0]) if len(test_seasons) else np.nan
        outputs["burden_metrics"].append(burden)


def run_experiments(
    decisions: pd.DataFrame,
    seasons: pd.DataFrame,
    contract: dict,
    run_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Fit the predeclared matrix and evaluate rolling external years."""
    target_key = contract["target_key"]
    specs = model_specs(target_key)
    params = {
        key: contract["catboost_fixed"][key]
        for key in ("iterations", "depth", "learning_rate", "l2_leaf_reg")
    }
    seed = int(contract["random_seed"])
    policy_cfg = contract["notification_policy"]
    outputs: dict[str, list] = {
        "predictions": [],
        "alarm_states": [],
        "event_metrics": [],
        "event_hits": [],
        "burden_metrics": [],
        "policy_selection": [],
        "policy_grid": [],
        "validation_predictions": [],
        "training_audit": [],
        "model_roundtrip": [],
    }
    for fold_index, fold in enumerate(contract["rolling_origin_folds"]):
        fold_id = str(fold["id"])
        train = _training_rows(decisions, fold["train_years"])
        validation = decisions.loc[_year_mask(decisions, fold["validation_years"])].copy()
        validation_seasons = seasons.loc[_year_mask(seasons, fold["validation_years"])].copy()
        test = decisions.loc[_year_mask(decisions, fold["test_years"])].copy()
        test_seasons = seasons.loc[_year_mask(seasons, fold["test_years"])].copy()
        if train.empty or validation.empty or test.empty:
            raise ValueError(f"Fold {fold_id} has an empty train, validation, or test partition")
        class_counts = train["target_class"].value_counts().to_dict()
        if set(class_counts) != set(TARGET_TO_INT):
            raise ValueError(f"Fold {fold_id} lacks target classes: {class_counts}")
        outputs["training_audit"].append(
            {
                "fold_id": fold_id,
                "train_rows": int(len(train)),
                "train_field_seasons": int(train["field_season"].nunique()),
                "validation_rows": int(len(validation)),
                "test_rows": int(len(test)),
                **{f"train_class_{key}": int(class_counts.get(key, 0)) for key in TARGET_TO_INT},
            }
        )

        for model_index, (model_code, spec) in enumerate(specs.items()):
            model = fit_model(
                train,
                spec["features"],
                spec["kind"],
                params if spec["kind"] == "catboost" else {"C": 0.1},
                seed + fold_index * 100 + model_index,
            )
            suffix = ".cbm" if spec["kind"] == "catboost" else ".joblib"
            model_path = run_dir / "models" / fold_id / f"{model_code}{suffix}"
            sample = train[spec["features"]].head(256)
            max_delta = _save_and_verify(model, spec["kind"], model_path, sample)
            outputs["model_roundtrip"].append(
                {
                    "fold_id": fold_id,
                    "model_code": model_code,
                    "path": str(model_path.relative_to(run_dir)),
                    "maximum_probability_delta": max_delta,
                    "features": json.dumps(spec["features"], ensure_ascii=False),
                }
            )
            validation_score = score_model(
                model, validation, spec["features"], spec["availability"]
            )
            policy, details = select_policy_fixed_grid(
                validation, validation_seasons, validation_score, contract
            )
            validation_prediction = validation[
                [
                    "field_season",
                    "season",
                    "issue_date",
                    "target_class",
                    "target_observable",
                    "service_active",
                    "common_weather_complete",
                    "candidate_comparison_complete",
                ]
            ].copy()
            validation_prediction["score"] = validation_score
            validation_prediction["model_code"] = model_code
            validation_prediction["fold_id"] = fold_id
            validation_prediction["feature_cutoff_provenance"] = (
                "frozen_NASA_through_issue_date_minus_2"
            )
            outputs["validation_predictions"].append(validation_prediction)
            for candidate in details["candidates"]:
                outputs["policy_grid"].append(
                    _policy_grid_row(fold_id, model_code, candidate, policy.threshold)
                )
            selected = details["selected"]
            outputs["policy_selection"].append(
                {
                    "fold_id": fold_id,
                    "model_code": model_code,
                    "threshold": policy.threshold,
                    "validation_timely_recall": selected["timely_recall"],
                    "validation_timely_hits": selected["timely_hits"],
                    "validation_events": selected["events_with_warning_opportunity"],
                    "validation_messages_per_30_field_days": selected[
                        "messages_per_30_field_days"
                    ],
                    "validation_active_alarm_fraction": selected["active_alarm_fraction"],
                    "validation_feasible": details["any_feasible"],
                    "candidate_count": len(details["candidates"]),
                }
            )
            test_score = score_model(model, test, spec["features"], spec["availability"])
            for scope, mask in (
                ("service_calendar", None),
                ("paired_candidate_days", test["candidate_comparison_complete"]),
            ):
                _append_evaluation(
                    outputs,
                    test=test,
                    test_seasons=test_seasons,
                    score=test_score,
                    policy=policy,
                    model_code=model_code,
                    fold_id=fold_id,
                    model_version=f"orchard_fixed_v1_{target_key}",
                    feature_provenance="frozen_NASA_through_issue_date_minus_2",
                    evaluation_scope=scope,
                    evaluation_mask=mask,
                )

        training_seasons = seasons.loc[_year_mask(seasons, fold["train_years"])].copy()
        baseline_scores, window = _calendar_scores(test, training_seasons)
        fixed_policy = Policy(
            0.5,
            int(policy_cfg["active_days_per_message"]),
            int(policy_cfg["cooldown_days"]),
            "orchard_policy_v1_fixed_binary_rule",
        )
        for baseline_code, score in baseline_scores.items():
            outputs["policy_selection"].append(
                {
                    "fold_id": fold_id,
                    "model_code": baseline_code,
                    "threshold": 0.5,
                    "validation_timely_recall": np.nan,
                    "validation_timely_hits": np.nan,
                    "validation_events": np.nan,
                    "validation_messages_per_30_field_days": np.nan,
                    "validation_active_alarm_fraction": np.nan,
                    "validation_feasible": np.nan,
                    "candidate_count": 1,
                    **window,
                }
            )
            for scope, mask in (
                ("service_calendar", None),
                ("paired_candidate_days", test["candidate_comparison_complete"]),
            ):
                _append_evaluation(
                    outputs,
                    test=test,
                    test_seasons=test_seasons,
                    score=score,
                    policy=fixed_policy,
                    model_code=baseline_code,
                    fold_id=fold_id,
                    model_version="deterministic_baseline_v1",
                    feature_provenance="training_year_event_calendar_only",
                    evaluation_scope=scope,
                    evaluation_mask=mask,
                )

    frames: dict[str, pd.DataFrame] = {}
    for name, values in outputs.items():
        frames[name] = pd.concat(values, ignore_index=True) if values and isinstance(values[0], pd.DataFrame) else pd.DataFrame(values)
    return frames
