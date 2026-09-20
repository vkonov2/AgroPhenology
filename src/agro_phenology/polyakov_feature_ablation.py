"""Bounded fixed-July-1 Polyakov feature ablation for early warning.

The experiment is deliberately separate from every completed research cycle.
It keeps the first-cycle target, rolling-origin folds, fixed shallow models and
notification policy, and adds exactly one engineered feature to C4/C5:
``polyakov_fixed_july1_score``.  The feature uses weather ending at the saved
``issue_date - 2`` cutoff and a calendar activation date known in advance.

The 2020--2025 years have already been inspected repeatedly.  Outputs from this
module therefore describe an exploratory retrospective ablation, not a new
independent test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd

from .early_warning_core import (
    CALENDAR_FEATURES,
    EPISODE_FEATURES,
    add_daily_features,
    sha256_file,
)
from .early_warning_cycle2_pipeline import verify_v3
from .early_warning_models import (
    TARGET_TO_INT,
    _actionable_probability,
    _append_evaluation,
    _fixed_catboost_params,
    daily_diagnostic,
    fit_model,
    score_model,
    select_threshold,
)
from .early_warning_reporting import aggregate_pooled_metrics, paired_year_bootstrap
from .era_hutton_backfill import (
    DEFAULT_RUN as DEFAULT_ERA_BACKFILL_RUN,
    DEFAULT_V3,
    EXPECTED_V3_MANIFEST_SHA256,
    _verify_completed_run as verify_era_backfill,
)
from .polyakov_fixed_date_experiment import (
    DEFAULT_SNAPSHOT_LATE_BLIGHT,
    build_fixed_polyakov_scores,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_SUBDIR = "polyakov_ablation"
RUN_FORMAT = "agro_phenology_polyakov_feature_ablation_v1"
TEST_YEARS = tuple(range(2020, 2026))
FIXED_FEATURE = "polyakov_fixed_july1_score"
BASE_FEATURES = tuple(CALENDAR_FEATURES + EPISODE_FEATURES)
PLUS_POLYAKOV_FEATURES = BASE_FEATURES + (FIXED_FEATURE,)
PRIMARY_SCOPE = "service_calendar"
PAIRED_SCOPE = "paired_candidate_days"
MODEL_SPECS: Mapping[str, Mapping[str, Any]] = {
    "C4": {"kind": "catboost", "features": BASE_FEATURES},
    "C5": {"kind": "logistic", "features": BASE_FEATURES},
    "C4-P": {"kind": "catboost", "features": PLUS_POLYAKOV_FEATURES},
    "C5-P": {"kind": "logistic", "features": PLUS_POLYAKOV_FEATURES},
}
CONTROL_CODES = ("calendar_window", "calendar_OR_polyakov_fixed")
COMPARISONS = (
    ("C4-P", "C4"),
    ("C5-P", "C5"),
    ("calendar_OR_polyakov_fixed", "calendar_window"),
    ("C4-P", "calendar_window"),
    ("C5-P", "calendar_window"),
)
SOURCE_FILES = (
    "src/agro_phenology/polyakov_feature_ablation.py",
    "tests/test_polyakov_feature_ablation.py",
    "src/agro_phenology/polyakov_fixed_date_experiment.py",
    "src/agro_phenology/early_warning_core.py",
    "src/agro_phenology/early_warning_cycle2_pipeline.py",
    "src/agro_phenology/early_warning_models.py",
    "src/agro_phenology/early_warning_reporting.py",
    "src/agro_phenology/era_hutton_backfill.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, payload: Any) -> None:
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=_json_default,
    )
    _atomic_bytes(path, (rendered + "\n").encode("utf-8"))


def _atomic_text(path: Path, text: str) -> None:
    _atomic_bytes(path, text.encode("utf-8"))


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_text(path, frame.to_csv(index=False))


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def output_dir(run_dir: str | Path) -> Path:
    """Return the required new subdirectory under a caller-owned run root."""
    return _resolve(run_dir) / OUTPUT_SUBDIR


def validate_causal_cutoffs(frame: pd.DataFrame) -> None:
    """Fail when an episode feature uses weather later than issue_date - 2."""
    required = {"issue_date", "feature_cutoff_era_episode_date"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Daily decisions lack {sorted(missing)}")
    issue = pd.to_datetime(frame["issue_date"], errors="raise").dt.normalize()
    cutoff = pd.to_datetime(
        frame["feature_cutoff_era_episode_date"], errors="raise"
    ).dt.normalize()
    if (cutoff > issue - pd.Timedelta(days=2)).any():
        raise AssertionError("Polyakov feature cutoff is later than issue_date - 2")


def shared_training_mask(frame: pd.DataFrame, year_bounds: Iterable[int]) -> pd.Series:
    """One observable complete-case mask shared by all four fitted models."""
    bounds = [int(value) for value in year_bounds]
    if len(bounds) != 2 or bounds[0] > bounds[1]:
        raise ValueError("year_bounds must contain an ordered start and end")
    required = {
        "season",
        "target_observable",
        "service_active",
        "target_class",
        "candidate_comparison_complete",
        "episode_weather_complete",
        FIXED_FEATURE,
        *BASE_FEATURES,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Ablation matrix lacks {sorted(missing)}")
    features_complete = frame[list(PLUS_POLYAKOV_FEATURES)].notna().all(axis=1)
    comparison_complete = frame.get(
        "frozen_candidate_comparison_complete",
        frame["candidate_comparison_complete"],
    ).astype(bool)
    return (
        frame["season"].between(bounds[0], bounds[1])
        & frame["target_observable"].astype(bool)
        & frame["service_active"].astype(bool)
        & comparison_complete
        & frame["episode_weather_complete"].astype(bool)
        & frame["target_class"].isin(TARGET_TO_INT)
        & features_complete
    )


def three_valued_or(calendar_score: pd.Series, polyakov_score: pd.Series) -> pd.Series:
    """Causal OR: a true calendar signal dominates, otherwise missing stays unknown."""
    calendar = pd.to_numeric(calendar_score, errors="coerce")
    polyakov = pd.to_numeric(polyakov_score, errors="coerce")
    result = pd.Series(np.nan, index=calendar.index, dtype=float)
    result.loc[calendar.eq(1) | polyakov.eq(1)] = 1.0
    result.loc[calendar.eq(0) & polyakov.eq(0)] = 0.0
    return result


def calendar_window_score(
    frame: pd.DataFrame, train_seasons: pd.DataFrame
) -> tuple[pd.Series, tuple[int, int]]:
    """Fit the existing 5--95% calendar bounds on train events only."""
    required = {"warnable_first_event", "first_recorded_event_date"}
    missing = required.difference(train_seasons.columns)
    if missing:
        raise ValueError(f"Field-season registry lacks {sorted(missing)}")
    dates = train_seasons.loc[
        train_seasons["warnable_first_event"].astype(bool), "first_recorded_event_date"
    ].dropna()
    if len(dates):
        doy = pd.to_datetime(dates, errors="raise").dt.dayofyear
        lower, upper = int(doy.quantile(0.05)), int(doy.quantile(0.95))
    else:
        lower, upper = 182, 243
    issue_doy = pd.to_datetime(frame["issue_date"], errors="raise").dt.dayofyear
    return issue_doy.between(lower, upper).astype(float), (lower, upper)


def complete_external_folds(contract: Mapping[str, Any], smoke: bool = False) -> list[dict]:
    """Use complete rolling folds through 2025 and reject temporal overlap."""
    selected: list[dict] = []
    for source in contract["rolling_origin_folds"]:
        fold = dict(source)
        test_start, test_end = map(int, fold["test_years"])
        if fold.get("incomplete") or test_start not in TEST_YEARS or test_end not in TEST_YEARS:
            continue
        train_end = int(fold["train_years"][1])
        validation_start, validation_end = map(int, fold["validation_years"])
        if not train_end < validation_start <= validation_end < test_start:
            raise AssertionError(f"Non-causal rolling fold: {fold['id']}")
        selected.append(fold)
    if not selected:
        raise ValueError("No complete 2020-2025 folds in the contract")
    return selected[:1] if smoke else selected


def _base_decision_frame(saved: pd.DataFrame) -> pd.DataFrame:
    if "outcome_semantics" not in saved.columns:
        raise ValueError("Saved v3 decisions lack outcome_semantics")
    end = saved.columns.get_loc("outcome_semantics")
    base = saved.iloc[:, : end + 1].copy()
    if base.duplicated(["field_season", "issue_date"]).any():
        raise ValueError("Saved decisions contain duplicate field-season dates")
    return base


def prepare_ablation_matrix(
    *,
    v3_dir: Path,
    extended_era_path: Path,
    snapshot_late_blight_path: Path,
    contract: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild all weather features from extended ERA and add one fixed score."""
    saved = pd.read_parquet(v3_dir / "daily_decisions.parquet")
    base = _base_decision_frame(saved)
    validate_causal_cutoffs(base)
    inputs = contract["inputs"]
    enriched, weather_audit = add_daily_features(
        base,
        _resolve(inputs["frozen_external_dir"]) / "nasa_daily.parquet",
        _resolve(inputs["frozen_external_dir"]) / "nasa_coordinate_mapping.parquet",
        extended_era_path,
    )
    era = pd.read_parquet(extended_era_path)
    years = sorted(pd.to_numeric(enriched["season"], errors="raise").astype(int).unique())
    enriched, polyakov_audit = build_fixed_polyakov_scores(
        enriched,
        era,
        snapshot_late_blight_path,
        years=years,
        source_label="extended_era_fixed_july1_feature_ablation",
    )
    enriched[FIXED_FEATURE] = enriched["polyakov_score_fixed"].astype(float)
    enriched["frozen_candidate_comparison_complete"] = saved[
        "candidate_comparison_complete"
    ].astype(bool).to_numpy()
    enriched["frozen_episode_weather_complete"] = saved[
        "episode_weather_complete"
    ].astype(bool).to_numpy()
    validate_causal_cutoffs(enriched)

    identity = ["field_season", "issue_date"]
    saved_identity = saved[identity].reset_index(drop=True).copy()
    enriched_identity = enriched[identity].reset_index(drop=True).copy()
    saved_identity["issue_date"] = pd.to_datetime(saved_identity["issue_date"])
    enriched_identity["issue_date"] = pd.to_datetime(enriched_identity["issue_date"])
    if not saved_identity.equals(enriched_identity):
        raise AssertionError("Feature reconstruction changed daily decision identity or order")

    complete = enriched["episode_weather_complete"].astype(bool)
    if enriched.loc[complete, FIXED_FEATURE].isna().any():
        raise AssertionError("Fixed Polyakov score is missing on complete episode rows")
    deterministic = (
        enriched.groupby(
            ["season", "weather_cell", "feature_cutoff_era_episode_date"],
            dropna=False,
        )[FIXED_FEATURE]
        .nunique(dropna=False)
        .max()
    )
    if int(deterministic) > 1:
        raise AssertionError("Fixed Polyakov score depends on field identity")

    old_common = saved["candidate_comparison_complete"].astype(bool)
    new_common = enriched["candidate_comparison_complete"].astype(bool)
    if (old_common & ~new_common).any():
        raise AssertionError("Extended ERA loses previously complete candidate rows")
    overlap = old_common & new_common
    maximum_overlap_feature_drift = 0.0
    if overlap.any():
        old_values = saved.loc[overlap, list(BASE_FEATURES)].astype(float).to_numpy()
        new_values = enriched.loc[overlap, list(BASE_FEATURES)].astype(float).to_numpy()
        maximum_overlap_feature_drift = float(np.nanmax(np.abs(old_values - new_values)))
        if not np.allclose(old_values, new_values, rtol=0.0, atol=1e-10, equal_nan=True):
            raise AssertionError("Extended ERA changes existing episode features")
        # Preserve the exact frozen matrix on its original complete rows.  The
        # backfill is additive: recomputed values are used only where v3 could
        # not score, avoiding even harmless rolling-point drift at old rows.
        enriched.loc[overlap, list(BASE_FEATURES)] = saved.loc[
            overlap, list(BASE_FEATURES)
        ].to_numpy()

    external = enriched["season"].isin(TEST_YEARS)
    service = external & enriched["evaluation_field_day"].astype(bool)
    observable = (
        enriched["target_observable"].astype(bool)
        & enriched["service_active"].astype(bool)
        & enriched["target_class"].isin(TARGET_TO_INT)
    )
    audit = {
        "weather": weather_audit,
        "polyakov": polyakov_audit,
        "rows": int(len(enriched)),
        "external_service_days": int(service.sum()),
        "old_external_candidate_complete_days": int((service & old_common).sum()),
        "new_external_candidate_complete_days": int((service & new_common).sum()),
        "new_external_episode_complete_days": int(
            (service & enriched["episode_weather_complete"].astype(bool)).sum()
        ),
        "new_complete_service_days": int((service & new_common & ~old_common).sum()),
        "old_observable_complete_rows": int((observable & old_common).sum()),
        "new_observable_complete_rows": int((observable & new_common).sum()),
        "overlap_features_unchanged": True,
        "maximum_recomputed_overlap_feature_drift_before_frozen_restore": (
            maximum_overlap_feature_drift
        ),
        "frozen_features_restored_on_original_complete_rows": True,
        "fixed_feature_name": FIXED_FEATURE,
        "fixed_feature_is_field_identity_independent": True,
        "feature_cutoff": "issue_date_minus_2_days",
        "historical_availability": "retrospective_past_only_assumption_not_publication_log",
    }
    return enriched.reset_index(drop=True), audit


def _fit_and_save(
    train: pd.DataFrame,
    features: list[str],
    kind: str,
    params: dict[str, Any],
    seed: int,
    path: Path,
) -> tuple[Any, float]:
    model = fit_model(train, features, kind, params, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if kind == "catboost":
        from catboost import CatBoostClassifier

        model.save_model(temporary)
        loaded = CatBoostClassifier()
        loaded.load_model(temporary)
    else:
        joblib.dump(model, temporary)
        loaded = joblib.load(temporary)
    before = _actionable_probability(model, train[features].head(100))
    after = _actionable_probability(loaded, train[features].head(100))
    difference = float(np.max(np.abs(before - after))) if len(before) else np.nan
    if not np.isfinite(difference) or difference > 1e-12:
        temporary.unlink(missing_ok=True)
        raise AssertionError(f"Model reload changed scores by {difference}")
    temporary.replace(path)
    return model, difference


def _empty_outputs() -> dict[str, list]:
    return {
        "predictions": [],
        "alarm_states": [],
        "event_metrics": [],
        "event_hits": [],
        "burden_metrics": [],
        "policy_selection": [],
        "daily_diagnostics": [],
        "model_registry": [],
        "baseline_reproduction": [],
        "fold_audit": [],
    }


def _as_frames(outputs: Mapping[str, list]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name, values in outputs.items():
        if values and isinstance(values[0], pd.DataFrame):
            frames[name] = pd.concat(values, ignore_index=True)
        else:
            frames[name] = pd.DataFrame(values)
    return frames


def _record_policy(
    outputs: dict[str, list],
    *,
    model_code: str,
    fold_id: str,
    scope: str,
    policy: Any,
    details: Mapping[str, Any],
    params: Mapping[str, Any],
    seed: int | None,
    fit_population: str,
) -> None:
    selected = details["selected"]
    outputs["policy_selection"].append(
        {
            "model_code": model_code,
            "fold_id": fold_id,
            "evaluation_scope": scope,
            "threshold": policy.threshold,
            "active_days": policy.active_days,
            "cooldown_days": policy.cooldown_days,
            "validation_policy_feasible": bool(details["any_feasible"]),
            "validation_timely_recall": selected["timely_recall"],
            "validation_messages_per_30": selected["messages_per_30_field_days"],
            "validation_alarm_fraction": selected["active_alarm_fraction"],
            "model_params": json.dumps(params, sort_keys=True),
            "model_seed": seed,
            "fit_population": fit_population,
            "selection_data": "validation_years_only",
        }
    )


def compare_saved_baseline_scores(
    saved_predictions: pd.DataFrame,
    *,
    model_code: str,
    fold_id: str,
    test_frame: pd.DataFrame,
    score: pd.Series,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Require C4/C5 to reproduce v3 wherever their old input was complete."""
    if model_code not in {"C4", "C5"}:
        raise ValueError("Only the no-P C4/C5 baselines have saved v3 counterparts")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    required = {
        "field_season",
        "issue_date",
        "model_code",
        "fold_id",
        "evaluation_scope",
        "score",
    }
    missing = required.difference(saved_predictions.columns)
    if missing:
        raise ValueError(f"Saved predictions lack {sorted(missing)}")
    old = saved_predictions.loc[
        saved_predictions["model_code"].eq(model_code)
        & saved_predictions["fold_id"].eq(fold_id)
        & saved_predictions["evaluation_scope"].eq(PRIMARY_SCOPE),
        ["field_season", "issue_date", "score"],
    ].copy()
    if old.empty:
        raise ValueError(f"Saved v3 predictions missing for {fold_id}/{model_code}")
    old["issue_date"] = pd.to_datetime(old["issue_date"], errors="raise")
    if old.duplicated(["field_season", "issue_date"]).any():
        raise ValueError(f"Duplicate saved v3 scores for {fold_id}/{model_code}")
    current = test_frame[["field_season", "issue_date"]].copy()
    current["issue_date"] = pd.to_datetime(current["issue_date"], errors="raise")
    current["current_score"] = score.to_numpy()
    current["expected_old_computed"] = (
        test_frame["service_active"].astype(bool)
        & test_frame["frozen_episode_weather_complete"].astype(bool)
    ).to_numpy()
    paired = old.merge(
        current,
        on=["field_season", "issue_date"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise AssertionError(f"Saved/current decision population differs for {fold_id}/{model_code}")
    old_computed = paired["score"].notna()
    if not old_computed.equals(paired["expected_old_computed"].astype(bool)):
        raise AssertionError(f"Saved v3 NaN mask differs from frozen input mask for {fold_id}/{model_code}")
    if paired.loc[old_computed, "current_score"].isna().any():
        raise AssertionError(f"Extended score is missing on an old complete row for {fold_id}/{model_code}")
    differences = (
        paired.loc[old_computed, "score"].astype(float)
        - paired.loc[old_computed, "current_score"].astype(float)
    ).abs()
    maximum = float(differences.max()) if len(differences) else 0.0
    if maximum > tolerance:
        raise AssertionError(
            f"{fold_id}/{model_code} does not reproduce v3: {maximum} > {tolerance}"
        )
    return {
        "fold_id": fold_id,
        "model_code": model_code,
        "saved_rows": int(len(paired)),
        "old_computed_rows": int(old_computed.sum()),
        "newly_computed_rows": int((~old_computed & paired["current_score"].notna()).sum()),
        "old_nan_mask_matches_frozen_scope": True,
        "max_abs_score_difference_on_old_complete_rows": maximum,
        "tolerance": tolerance,
        "status": "passed",
    }


def run_models(
    matrix: pd.DataFrame,
    seasons: pd.DataFrame,
    contract: Mapping[str, Any],
    destination: Path,
    *,
    saved_v3_predictions: pd.DataFrame,
    smoke: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fit fixed paired models and replay policies on complete external folds."""
    outputs = _empty_outputs()
    policy_settings = contract["notification_policy"]
    active_days = int(policy_settings["active_days_per_message"])
    cooldown_days = int(policy_settings["cooldown_days"])
    budget = policy_settings["research_budget"]
    max_messages = float(budget["messages_per_30_field_days_max"])
    max_alarm = float(budget["active_alarm_fraction_max"])
    base_seed = int(contract["random_seed"])
    catboost_params = _fixed_catboost_params(dict(contract))

    folds = complete_external_folds(contract, smoke=smoke)
    for fold_index, fold in enumerate(folds):
        fold_id = str(fold["id"])
        mask = shared_training_mask(matrix, fold["train_years"])
        train = matrix.loc[mask].copy().reset_index(drop=True)
        validation = matrix.loc[
            matrix["season"].between(*map(int, fold["validation_years"]))
        ].copy().reset_index(drop=True)
        test = matrix.loc[
            matrix["season"].between(*map(int, fold["test_years"]))
        ].copy().reset_index(drop=True)
        train_seasons = seasons.loc[
            seasons["season"].between(*map(int, fold["train_years"]))
        ].copy()
        validation_seasons = seasons.loc[
            seasons["season"].between(*map(int, fold["validation_years"]))
        ].copy()
        test_seasons = seasons.loc[
            seasons["season"].between(*map(int, fold["test_years"]))
        ].copy()
        if train.empty or validation.empty or test.empty:
            raise ValueError(f"Fold {fold_id} has an empty train, validation, or test block")
        if set(train["target_class"].map(TARGET_TO_INT).unique()) != set(TARGET_TO_INT.values()):
            raise ValueError(f"Fold {fold_id} lacks a target class")
        outputs["fold_audit"].append(
            {
                "fold_id": fold_id,
                "train_year_start": int(fold["train_years"][0]),
                "train_year_end": int(fold["train_years"][1]),
                "validation_year_start": int(fold["validation_years"][0]),
                "validation_year_end": int(fold["validation_years"][1]),
                "test_year": int(fold["test_years"][0]),
                "shared_train_rows": int(len(train)),
                "target_counts": json.dumps(
                    train["target_class"].value_counts().sort_index().to_dict(),
                    sort_keys=True,
                ),
                "same_mask_for_C4_C5_C4P_C5P": True,
            }
        )

        paired_mask_validation = validation["frozen_candidate_comparison_complete"].astype(bool)
        paired_mask_test = test["frozen_candidate_comparison_complete"].astype(bool)
        for model_code, spec in MODEL_SPECS.items():
            kind = str(spec["kind"])
            features = list(spec["features"])
            params = catboost_params if kind == "catboost" else {"C": 0.1}
            # Preserve the original fixed C4/C5 seeds while giving each paired
            # +P candidate exactly the same seed as its no-P counterpart.
            algorithm_offset = 4 if kind == "catboost" else 5
            seed = base_seed + fold_index * 20 + algorithm_offset
            suffix = ".cbm" if kind == "catboost" else ".joblib"
            safe_code = model_code.replace("-", "_")
            model, reload_difference = _fit_and_save(
                train,
                features,
                kind,
                dict(params),
                seed,
                destination / "models" / f"{fold_id}_{safe_code}{suffix}",
            )
            outputs["model_registry"].append(
                {
                    "model_code": model_code,
                    "fold_id": fold_id,
                    "kind": kind,
                    "features": json.dumps(features),
                    "feature_count": len(features),
                    "fixed_polyakov_feature_count": int(FIXED_FEATURE in features),
                    "seed": seed,
                    "params": json.dumps(params, sort_keys=True),
                    "shared_train_rows": int(len(train)),
                    "reload_max_abs_score_difference": reload_difference,
                }
            )
            validation_score = score_model(
                model, validation, features, "episode_weather_complete"
            )
            test_score = score_model(model, test, features, "episode_weather_complete")
            if model_code in {"C4", "C5"}:
                outputs["baseline_reproduction"].append(
                    compare_saved_baseline_scores(
                        saved_v3_predictions,
                        model_code=model_code,
                        fold_id=fold_id,
                        test_frame=test,
                        score=test_score,
                    )
                )
            outputs["daily_diagnostics"].append(
                daily_diagnostic(test.assign(score=test_score), model_code, fold_id)
            )
            for scope, validation_mask, test_mask in (
                (PRIMARY_SCOPE, None, None),
                (PAIRED_SCOPE, paired_mask_validation, paired_mask_test),
            ):
                policy, details = select_threshold(
                    validation,
                    validation_score,
                    validation_seasons,
                    active_days,
                    cooldown_days,
                    max_messages,
                    max_alarm,
                    scope,
                    validation_mask,
                )
                _append_evaluation(
                    outputs,
                    test,
                    test_seasons,
                    test_score,
                    policy,
                    model_code,
                    fold_id,
                    "fixed_train_only_polyakov_ablation_v1",
                    "extended_era_past_only_issue_minus_2",
                    scope,
                    "frozen_candidate_comparison_complete" if test_mask is not None else None,
                )
                _record_policy(
                    outputs,
                    model_code=model_code,
                    fold_id=fold_id,
                    scope=scope,
                    policy=policy,
                    details=details,
                    params=params,
                    seed=seed,
                    fit_population="shared_observable_candidate_complete_train_only",
                )

        calendar_validation, bounds = calendar_window_score(validation, train_seasons)
        calendar_test, test_bounds = calendar_window_score(test, train_seasons)
        if bounds != test_bounds:
            raise AssertionError("Calendar bounds differ within one fold")
        control_scores = {
            "calendar_window": (calendar_validation, calendar_test),
            "calendar_OR_polyakov_fixed": (
                three_valued_or(calendar_validation, validation[FIXED_FEATURE]),
                three_valued_or(calendar_test, test[FIXED_FEATURE]),
            ),
        }
        for model_code, (validation_score, test_score) in control_scores.items():
            for scope, validation_mask, test_mask in (
                (PRIMARY_SCOPE, None, None),
                (PAIRED_SCOPE, paired_mask_validation, paired_mask_test),
            ):
                policy, details = select_threshold(
                    validation,
                    validation_score,
                    validation_seasons,
                    active_days,
                    cooldown_days,
                    max_messages,
                    max_alarm,
                    scope,
                    validation_mask,
                )
                _append_evaluation(
                    outputs,
                    test,
                    test_seasons,
                    test_score,
                    policy,
                    model_code,
                    fold_id,
                    "train_calendar_bounds_unified_policy_v1",
                    "calendar_train_bounds_and_fixed_july1_extended_era",
                    scope,
                    "frozen_candidate_comparison_complete" if test_mask is not None else None,
                )
                _record_policy(
                    outputs,
                    model_code=model_code,
                    fold_id=fold_id,
                    scope=scope,
                    policy=policy,
                    details=details,
                    params={"calendar_doy_bounds": list(bounds)},
                    seed=None,
                    fit_population="calendar_bounds_from_train_events_only",
                )

    return _as_frames(outputs)


def _notification_log(states: pd.DataFrame) -> pd.DataFrame:
    selected = states.loc[
        states["message_issued"].astype(bool) | states["suppressed_repeat"].astype(bool)
    ].copy()
    columns = [
        "field_season",
        "season",
        "issued_at",
        "issue_date",
        "model_code",
        "model_version",
        "evaluation_scope",
        "score",
        "score_status",
        "policy_version",
        "policy_threshold",
        "message_issued",
        "action_reason",
        "active_from",
        "active_through",
        "forecast_window_start",
        "forecast_window_end",
        "suppressed_repeat",
    ]
    return selected[columns]


def _report_text(
    pooled: pd.DataFrame,
    bootstrap: pd.DataFrame,
    matrix_audit: Mapping[str, Any],
    smoke: bool,
) -> str:
    primary = pooled.loc[
        pooled["period"].eq("2020_2025")
        & pooled["evaluation_scope"].eq(PRIMARY_SCOPE)
        & pooled["slice"].eq("A_plus_B")
    ].copy()
    rows = [
        "# Абляция фиксированного признака Полякова",
        "",
        "Это разведочный ретроспективный эксперимент на уже изученных годах 2020–2025. "
        "Он не является новым независимым тестом.",
        "",
        "Использована прежняя цель первого зарегистрированного события, окно 3–10 дней, "
        "rolling-origin folds, фиксированный неглубокий CatBoost и прежняя последовательная "
        "политика. Optuna не запускалась. Порог каждого кандидата выбран только на validation-годах.",
        "",
        "C4-P/C5-P отличаются от C4/C5 ровно одним причинным признаком: "
        "`polyakov_fixed_july1_score`. Это готовая нелинейная комбинация уже существующих "
        "погодных компонентов и календарного фазового шлюза; новых физических измерений она не добавляет.",
        "",
        "## Основной полный сервис",
        "",
        "| Метод | Timely | Сообщения | Сообщений/30 дней | Тревожные дни | Вычислимость |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for code in (*MODEL_SPECS.keys(), *CONTROL_CODES):
        subset = primary.loc[primary["model_code"].eq(code)]
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append(
            f"| {code} | {int(row.timely_hits)}/{int(row.events_with_warning_opportunity)} "
            f"| {int(row.messages)} | {float(row.messages_per_30_field_days):.3f} "
            f"| {float(row.active_alarm_fraction):.1%} "
            f"| {int(row.computable_days)}/{int(row.field_days)} "
            f"({float(row.computable_fraction):.2%}) |"
        )
    rows.extend(
        [
            "",
            "## Доступность и интерпретация",
            "",
            f"Дополненный ERA дал {matrix_audit['new_external_candidate_complete_days']}/"
            f"{matrix_audit['external_service_days']} дней общей маски против "
            f"{matrix_audit['old_external_candidate_complete_days']} ранее. "
            f"Число observable complete-case строк изменилось с "
            f"{matrix_audit['old_observable_complete_rows']} до "
            f"{matrix_audit['new_observable_complete_rows']}.",
            "",
            "Задержка ERA5 остаётся ретроспективным допущением: наличие past-only строки в "
            "архиве не доказывает её оперативную доступность в историческую дату решения.",
            "",
            "Парные годовые интервалы и exact gained/lost доступны в `paired_year_bootstrap.csv` "
            "и `event_hits.parquet`. Интервалы основаны только на шести годовых блоках и не "
            "устраняют post-hoc характер проверки.",
        ]
    )
    if smoke:
        rows.extend(["", "Запуск выполнен в smoke-режиме и не предназначен для вывода о качестве."])
    if not bootstrap.empty:
        rows.extend(["", "Широкий подбор после результата автоматически не запускается."])
    return "\n".join(rows) + "\n"


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "catboost", "joblib", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def _hashes(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "execution_manifest.json":
            relative = str(path.relative_to(root))
            result[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def run_ablation(
    *,
    run_dir: Path,
    v3_dir: Path = DEFAULT_V3,
    era_backfill_run: Path = DEFAULT_ERA_BACKFILL_RUN,
    snapshot_late_blight_path: Path = DEFAULT_SNAPSHOT_LATE_BLIGHT,
    smoke: bool = False,
) -> Path:
    destination = output_dir(run_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Ablation directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "format": RUN_FORMAT,
        "status": "running",
        "started_at_utc": _utc_now(),
        "mode": "smoke" if smoke else "full",
        "exploratory_external_period": "2020-2025_already_studied_not_independent_test",
        "command": [sys.executable, *sys.argv],
        "environment": _environment(),
    }
    _atomic_json(destination / "execution_manifest.json", manifest)
    try:
        v3_dir = _resolve(v3_dir)
        era_backfill_run = _resolve(era_backfill_run)
        snapshot_late_blight_path = _resolve(snapshot_late_blight_path)
        v3_verification = verify_v3(v3_dir, EXPECTED_V3_MANIFEST_SHA256)
        backfill_verification = verify_era_backfill(era_backfill_run)
        extended_era = era_backfill_run / "era5_potato_daily_extended.parquet"
        contract_path = v3_dir / "evaluation_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        matrix, matrix_audit = prepare_ablation_matrix(
            v3_dir=v3_dir,
            extended_era_path=extended_era,
            snapshot_late_blight_path=snapshot_late_blight_path,
            contract=contract,
        )
        seasons = pd.read_parquet(v3_dir / "field_seasons.parquet")
        saved_v3_predictions = pd.read_parquet(v3_dir / "predictions.parquet")
        outputs = run_models(
            matrix,
            seasons,
            contract,
            destination,
            saved_v3_predictions=saved_v3_predictions,
            smoke=smoke,
        )
        periods = {"2020_2025": (2020, 2025)}
        pooled_parts = aggregate_pooled_metrics(
            outputs["event_metrics"],
            outputs["burden_metrics"],
            event_hits=outputs["event_hits"],
            alarm_states=outputs["alarm_states"],
            periods=periods,
        )
        bootstrap = paired_year_bootstrap(
            outputs["event_hits"],
            outputs["alarm_states"],
            seed=int(contract["random_seed"]),
            n_bootstrap=200 if smoke else 2000,
            periods=periods,
            comparisons=COMPARISONS,
            slices=("A_plus_B",),
        )
        outputs.update(pooled_parts)
        outputs["paired_year_bootstrap"] = bootstrap

        test_features = matrix.loc[
            matrix["season"].isin(TEST_YEARS),
            [
                "field_season",
                "season",
                "issue_date",
                "feature_cutoff_era_episode_date",
                FIXED_FEATURE,
                "polyakov_status_fixed",
                "candidate_comparison_complete",
                "episode_weather_complete",
                "frozen_candidate_comparison_complete",
                "frozen_episode_weather_complete",
            ],
        ].copy()
        _atomic_parquet(destination / "polyakov_fixed_scores.parquet", test_features)
        parquet_outputs = {"predictions", "alarm_states", "event_hits"}
        for name, frame in outputs.items():
            if name in parquet_outputs:
                _atomic_parquet(destination / f"{name}.parquet", frame)
            else:
                _atomic_csv(destination / f"{name}.csv", frame)
        _atomic_parquet(
            destination / "notification_log.parquet",
            _notification_log(outputs["alarm_states"]),
        )
        _atomic_json(destination / "matrix_audit.json", matrix_audit)
        contract_snapshot = {
            "source_contract_sha256": sha256_file(contract_path),
            "target": contract["daily_target"],
            "timeliness_window_days": contract["timeliness_window_days"],
            "rolling_origin_folds": complete_external_folds(contract, smoke=smoke),
            "notification_policy": contract["notification_policy"],
            "catboost_fixed": contract["catboost_fixed"],
            "fixed_feature": {
                "name": FIXED_FEATURE,
                "activation": "July 1 known at season start",
                "weather_cutoff": "issue_date - 2 days",
                "physical_information_added": False,
                "interpretation": "predefined nonlinear gate and conjunction",
            },
            "optuna": "disabled",
            "external_interpretation": "exploratory_already_studied_years",
            "evaluation_scopes": {
                PRIMARY_SCOPE: "full service replay scored from extended ERA when episode inputs are complete",
                PAIRED_SCOPE: "frozen v3 candidate-comparison mask, held fixed across all candidates",
            },
            "baseline_reproduction": (
                "C4/C5 must match saved v3 service scores on every frozen complete row "
                "within absolute tolerance 1e-10"
            ),
        }
        _atomic_json(destination / "ablation_contract.json", contract_snapshot)
        _atomic_text(
            destination / "report_ru.md",
            _report_text(
                outputs["pooled_summary"], bootstrap, matrix_audit, smoke=smoke
            ),
        )
        reproduce = (
            "# Воспроизведение\n\n"
            "Из корня репозитория задайте новый, ещё не существующий каталог:\n\n"
            "```bash\n"
            'new_run_dir="results/late_blight_early_warning/YYYYMMDD_post_backfill_models_new"\n'
            f"{sys.executable} -m agro_phenology.polyakov_feature_ablation run "
            '--run-dir "$new_run_dir"\n'
            f"{sys.executable} -m agro_phenology.polyakov_feature_ablation check "
            '--run-dir "$new_run_dir"\n'
            "```\n\n"
            "Команда `run` всегда создаёт новый подкаталог `polyakov_ablation` и "
            "отказывается перезаписывать непустой каталог.\n"
        )
        _atomic_text(destination / "REPRODUCE.md", reproduce)

        source_hashes = {
            relative: sha256_file(REPO_ROOT / relative) for relative in SOURCE_FILES
        }
        _atomic_json(
            destination / "source_snapshot.json",
            {
                "format": "polyakov_feature_ablation_source_snapshot_v1",
                "created_at_utc": _utc_now(),
                "files": [
                    {
                        "relative_path": relative,
                        "sha256": source_hashes[relative],
                        "utf8_content": (REPO_ROOT / relative).read_text(
                            encoding="utf-8"
                        ),
                    }
                    for relative in SOURCE_FILES
                ],
            },
        )
        input_paths = {
            "v3_execution_manifest": v3_dir / "execution_manifest.json",
            "v3_evaluation_contract": contract_path,
            "v3_daily_decisions": v3_dir / "daily_decisions.parquet",
            "v3_field_seasons": v3_dir / "field_seasons.parquet",
            "v3_predictions": v3_dir / "predictions.parquet",
            "extended_era": extended_era,
            "era_backfill_manifest": era_backfill_run / "execution_manifest.json",
            "snapshot_late_blight": snapshot_late_blight_path,
            "nasa_daily": _resolve(contract["inputs"]["frozen_external_dir"])
            / "nasa_daily.parquet",
            "nasa_mapping": _resolve(contract["inputs"]["frozen_external_dir"])
            / "nasa_coordinate_mapping.parquet",
        }
        manifest.update(
            {
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "v3_verification": v3_verification,
                "era_backfill_status": backfill_verification.get("status"),
                "input_hashes": {
                    str(path): sha256_file(path) for path in input_paths.values()
                },
                "source_hashes": source_hashes,
                "output_hashes": _hashes(destination),
                "matrix_audit": matrix_audit,
                "models": list(MODEL_SPECS) + list(CONTROL_CODES),
                "optuna_run": False,
            }
        )
        _atomic_json(destination / "execution_manifest.json", manifest)
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        _atomic_json(destination / "execution_manifest.json", manifest)
        raise
    return destination


def check_ablation(run_dir: Path) -> dict[str, Any]:
    destination = output_dir(run_dir)
    manifest_path = destination / "execution_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != RUN_FORMAT or manifest.get("status") != "complete":
        raise AssertionError("Ablation manifest is not a completed v1 run")
    failures: list[str] = []
    for relative, metadata in manifest["output_hashes"].items():
        path = destination / relative
        actual = sha256_file(path) if path.is_file() else None
        if actual != metadata["sha256"]:
            failures.append(relative)
    for path_text, expected in manifest["input_hashes"].items():
        path = Path(path_text)
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            failures.append(f"input:{path_text}")
    for relative, expected in manifest["source_hashes"].items():
        path = REPO_ROOT / relative
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            failures.append(f"source:{relative}")
    if failures:
        raise AssertionError(f"Ablation hash failures: {failures}")
    return {
        "status": "passed",
        "directory": str(destination),
        "outputs_checked": len(manifest["output_hashes"]),
        "inputs_checked": len(manifest["input_hashes"]),
        "sources_checked": len(manifest["source_hashes"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a new bounded ablation")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--v3-dir", type=Path, default=DEFAULT_V3)
    run.add_argument("--era-backfill-run", type=Path, default=DEFAULT_ERA_BACKFILL_RUN)
    run.add_argument(
        "--snapshot-late-blight",
        type=Path,
        default=DEFAULT_SNAPSHOT_LATE_BLIGHT,
    )
    run.add_argument("--smoke", action="store_true")
    check = subparsers.add_parser("check", help="verify a completed ablation")
    check.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        destination = run_ablation(
            run_dir=args.run_dir,
            v3_dir=args.v3_dir,
            era_backfill_run=args.era_backfill_run,
            snapshot_late_blight_path=args.snapshot_late_blight,
            smoke=args.smoke,
        )
        print(destination)
        return 0
    result = check_ablation(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
