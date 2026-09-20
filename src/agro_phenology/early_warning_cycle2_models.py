"""Leakage-safe C6 models for the second late-blight warning cycle.

C6 keeps the first-cycle calendar C0 logits as an offset and learns only a
small weather correction::

    p(y | x) = softmax(b_C0 + alpha * f_weather)

The helpers in this module deliberately keep the baseline, correction and
availability mask separate.  This makes the alpha=0 identity and the C0
fallback on days without weather directly testable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd

from .early_warning_core import CALENDAR_FEATURES, EPISODE_FEATURES
from .early_warning_models import Policy, TARGET_TO_INT, _model_rows, fit_model


CLASS_ORDER: tuple[int, int, int] = (0, 1, 2)
CLASS_NAMES: tuple[str, str, str] = (
    "no_record_in_horizon",
    "imminent",
    "actionable",
)
ALPHA_GRID: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)
SHALLOW_CATBOOST_PARAMS: dict[str, Any] = {
    "iterations": 200,
    "depth": 2,
    "learning_rate": 0.03,
    "l2_leaf_reg": 30.0,
}
RAW_LOGIT_COLUMNS: tuple[str, ...] = tuple(f"c0_raw_{value}" for value in CLASS_ORDER)
PROBABILITY_COLUMNS: tuple[str, ...] = tuple(f"c0_probability_{value}" for value in CLASS_ORDER)


def _two_dimensional(values: Any, *, rows: int | None = None, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim == 1:
        result = result.reshape(-1, 1)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if rows is not None and len(result) != rows:
        raise ValueError(f"{name} row count differs from the input row count")
    return result


def align_class_matrix(
    values: Any,
    source_classes: Iterable[int],
    class_order: Sequence[int] = CLASS_ORDER,
) -> np.ndarray:
    """Return a class-valued matrix in the declared, fixed class order."""
    matrix = _two_dimensional(values, name="class-valued output")
    source = tuple(int(value) for value in source_classes)
    target = tuple(int(value) for value in class_order)
    if len(source) != matrix.shape[1] or len(set(source)) != len(source):
        raise ValueError("Model classes do not match the number of output columns")
    if set(source) != set(target):
        raise ValueError(f"Expected model classes {list(target)}, got {list(source)}")
    return matrix[:, [source.index(value) for value in target]]


def stable_softmax(raw_logits: Any) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    logits = _two_dimensional(raw_logits, name="raw_logits")
    if not np.isfinite(logits).all():
        raise ValueError("raw_logits must be finite")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def aligned_probabilities(
    model: Any,
    values: Any,
    class_order: Sequence[int] = CLASS_ORDER,
) -> np.ndarray:
    """Predict probabilities and align them to ``class_order``."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("The baseline model must expose predict_proba")
    predicted = model.predict_proba(values)
    return align_class_matrix(predicted, model.classes_, class_order)


def c0_raw_logits(
    model: Any,
    values: Any,
    class_order: Sequence[int] = CLASS_ORDER,
) -> np.ndarray:
    """Extract C0 raw formula values and align their class columns.

    The first-cycle C0 is a scikit-learn pipeline and therefore exposes
    ``decision_function``.  The CatBoost branch is retained for audited bundle
    compatibility, but C6 training itself requires the calendar-logit C0.
    """
    if hasattr(model, "decision_function"):
        raw = model.decision_function(values)
    elif hasattr(model, "predict"):
        try:
            raw = model.predict(values, prediction_type="RawFormulaVal")
        except TypeError as error:
            raise TypeError("The baseline model does not expose raw multiclass logits") from error
    else:
        raise TypeError("The baseline model does not expose raw multiclass logits")
    matrix = _two_dimensional(raw, rows=len(values), name="C0 raw logits")
    return align_class_matrix(matrix, model.classes_, class_order)


def _validate_logits(matrix: Any, rows: int, name: str) -> np.ndarray:
    result = _two_dimensional(matrix, rows=rows, name=name)
    if result.shape[1] != len(CLASS_ORDER):
        raise ValueError(f"{name} must have {len(CLASS_ORDER)} columns")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if value not in ALPHA_GRID:
        raise ValueError(f"alpha must be one of {list(ALPHA_GRID)}")
    return value


def combine_c6_logits(base_logits: Any, correction_logits: Any, alpha: float) -> np.ndarray:
    """Apply the predeclared C6 formula exactly once."""
    value = _validate_alpha(alpha)
    base = _two_dimensional(base_logits, name="base_logits")
    base = _validate_logits(base, len(base), "base_logits")
    if value == 0.0:
        return base.copy()
    correction = _validate_logits(correction_logits, len(base), "correction_logits")
    return base + value * correction


def _catboost_parameters(seed: int) -> dict[str, Any]:
    return {
        "loss_function": "MultiClass",
        "random_seed": int(seed),
        "thread_count": 2,
        "verbose": False,
        "allow_writing_files": False,
        **SHALLOW_CATBOOST_PARAMS,
    }


def fit_catboost_correction(
    values: pd.DataFrame,
    target: Sequence[int],
    baseline_logits: Any,
    *,
    seed: int,
):
    """Fit the fixed shallow CatBoost correction with C0 raw logits as Pool baseline."""
    from catboost import CatBoostClassifier, Pool

    labels = np.asarray(target, dtype=int)
    if len(values) != len(labels):
        raise ValueError("Feature and target row counts differ")
    if set(labels.tolist()) != set(CLASS_ORDER):
        raise ValueError(f"Correction training needs all classes {list(CLASS_ORDER)}")
    baseline = _validate_logits(baseline_logits, len(values), "baseline_logits")
    if values.isna().any(axis=None):
        raise ValueError("Correction training features must be complete")
    pool = Pool(values, label=labels, baseline=baseline)
    model = CatBoostClassifier(**_catboost_parameters(seed))
    model.fit(pool)
    if tuple(int(value) for value in model.classes_) != CLASS_ORDER:
        raise ValueError(f"Unexpected CatBoost class order: {model.classes_}")
    return model


def catboost_correction_logits(model: Any, values: pd.DataFrame) -> np.ndarray:
    """Predict only the learned CatBoost correction, without supplying a baseline."""
    raw = model.predict(values, prediction_type="RawFormulaVal")
    return align_class_matrix(raw, model.classes_, CLASS_ORDER)


@dataclass(frozen=True)
class C6Prediction:
    base_logits: np.ndarray
    correction_logits: np.ndarray
    combined_logits: np.ndarray
    probabilities: np.ndarray
    correction_available: np.ndarray
    used_c0_fallback: np.ndarray

    @property
    def actionable_probability(self) -> np.ndarray:
        return self.probabilities[:, CLASS_ORDER.index(TARGET_TO_INT["actionable"])]

    def to_frame(self, index: pd.Index | None = None) -> pd.DataFrame:
        data: dict[str, Any] = {}
        for position, class_id in enumerate(CLASS_ORDER):
            data[f"c0_raw_{class_id}"] = self.base_logits[:, position]
            data[f"correction_raw_{class_id}"] = self.correction_logits[:, position]
            data[f"c6_raw_{class_id}"] = self.combined_logits[:, position]
            data[f"c6_probability_{class_id}"] = self.probabilities[:, position]
        data["correction_available"] = self.correction_available
        data["used_c0_fallback"] = self.used_c0_fallback
        data["actionable_probability"] = self.actionable_probability
        return pd.DataFrame(data, index=index)


def predict_c6_from_baseline(
    *,
    base_logits: Any,
    base_probabilities: Any,
    correction_model: Any,
    correction_values: pd.DataFrame,
    alpha: float,
    correction_available: Sequence[bool] | pd.Series | None = None,
) -> C6Prediction:
    """Combine an already-scored C0 baseline with a CatBoost correction.

    Availability is checked before CatBoost is called.  Missing rows receive a
    literal zero correction and their saved C0 probabilities, so there is no
    NaN multiplication and no second policy state.
    """
    value = _validate_alpha(alpha)
    rows = len(correction_values)
    base = _validate_logits(base_logits, rows, "base_logits")
    base_probability = _validate_logits(base_probabilities, rows, "base_probabilities")
    if correction_available is None:
        available = correction_values.notna().all(axis=1).to_numpy(dtype=bool)
    else:
        available = np.asarray(correction_available, dtype=bool)
        if available.shape != (rows,):
            raise ValueError("correction_available must contain one value per row")
        available &= correction_values.notna().all(axis=1).to_numpy(dtype=bool)

    correction = np.zeros_like(base)
    # Exact identity branch: do not call a correction model and do not
    # reconstruct C0 probabilities through another softmax implementation.
    if value == 0.0:
        return C6Prediction(
            base_logits=base,
            correction_logits=correction,
            combined_logits=base.copy(),
            probabilities=base_probability.copy(),
            correction_available=available,
            used_c0_fallback=~available,
        )

    if available.any():
        predicted = catboost_correction_logits(correction_model, correction_values.loc[available])
        correction[available] = predicted
    combined = combine_c6_logits(base, correction, value)
    probability = stable_softmax(combined)
    # Preserve the exact C0 output on fallback rows (including its floating
    # representation), rather than recomputing it from logits.
    probability[~available] = base_probability[~available]
    return C6Prediction(
        base_logits=base,
        correction_logits=correction,
        combined_logits=combined,
        probabilities=probability,
        correction_available=available,
        used_c0_fallback=~available,
    )


@dataclass
class MulticlassCalibrationCorrection:
    """Regularized multiclass calibration represented as a logit correction."""

    model: Any
    class_order: tuple[int, ...] = CLASS_ORDER

    def predict_correction(self, base_logits: Any) -> np.ndarray:
        base = _validate_logits(base_logits, len(base_logits), "base_logits")
        calibrated = align_class_matrix(
            self.model.decision_function(base), self.model.classes_, self.class_order
        )
        return calibrated - base


def fit_calibration_control(
    baseline_logits: Any,
    target: Sequence[int],
    *,
    seed: int,
    regularization_c: float = 0.1,
) -> MulticlassCalibrationCorrection:
    """Fit the predeclared regularized multiclass calibration-only control."""
    from sklearn.linear_model import LogisticRegression

    labels = np.asarray(target, dtype=int)
    baseline = _validate_logits(baseline_logits, len(labels), "baseline_logits")
    if set(labels.tolist()) != set(CLASS_ORDER):
        raise ValueError(f"Calibration training needs all classes {list(CLASS_ORDER)}")
    model = LogisticRegression(
        C=float(regularization_c),
        max_iter=2000,
        random_state=int(seed),
    )
    model.fit(baseline, labels)
    return MulticlassCalibrationCorrection(model=model)


def predict_calibration_control(
    baseline_logits: Any,
    baseline_probabilities: Any,
    correction: MulticlassCalibrationCorrection,
    *,
    alpha: float,
) -> C6Prediction:
    value = _validate_alpha(alpha)
    base = _validate_logits(baseline_logits, len(baseline_logits), "base_logits")
    probabilities = _validate_logits(baseline_probabilities, len(base), "base_probabilities")
    available = np.ones(len(base), dtype=bool)
    if value == 0.0:
        raw_correction = np.zeros_like(base)
        return C6Prediction(base, raw_correction, base.copy(), probabilities.copy(), available, ~available)
    raw_correction = correction.predict_correction(base)
    combined = combine_c6_logits(base, raw_correction, value)
    return C6Prediction(base, raw_correction, combined, stable_softmax(combined), available, ~available)


@dataclass
class TemporalOOFResult:
    predictions: pd.DataFrame
    provenance: pd.DataFrame


def _bounded_years(bounds: Sequence[int]) -> tuple[int, int]:
    if len(bounds) != 2:
        raise ValueError("train_years must be inclusive [start_year, end_year] bounds")
    start, end = int(bounds[0]), int(bounds[1])
    if end < start:
        raise ValueError("train_years end precedes start")
    return start, end


def build_expanding_c0_oof(
    decisions: pd.DataFrame,
    train_years: Sequence[int],
    *,
    seed: int,
    carry_columns: Sequence[str] | None = None,
) -> TemporalOOFResult:
    """Build expanding-year C0 predictions using only labels available earlier.

    Early forecast years whose preceding rows do not contain all three classes
    are recorded as ``skipped_missing_history_or_classes``.  The deterministic
    rule depends only on the past, so appending future years cannot change any
    already produced OOF prediction.
    """
    required = {
        "season",
        "issue_date",
        "target_class",
        "target_observable",
        "service_active",
        "candidate_comparison_complete",
        *CALENDAR_FEATURES,
    }
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"Missing columns for temporal OOF: {missing}")
    start_year, end_year = _bounded_years(train_years)
    working = decisions.copy()
    # ``_model_rows`` comes from cycle 1 and its fallback expression evaluates
    # ``common_weather_complete`` eagerly.  Mirror the declared candidate mask
    # for compact audit/test frames that only carry the cycle-2 column.
    if "common_weather_complete" not in working:
        working["common_weather_complete"] = working["candidate_comparison_complete"]
    working["_source_index"] = decisions.index
    default_carry = [
        "field_season",
        "season",
        "issue_date",
        "label_interval_end",
        "label_available_at",
        "target_class",
        *CALENDAR_FEATURES,
        *EPISODE_FEATURES,
        "episode_weather_complete",
        "candidate_comparison_complete",
    ]
    carry = list(dict.fromkeys(carry_columns or default_carry))
    carry = [column for column in carry if column in working.columns]
    prediction_parts: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []

    for forecast_year in range(start_year, end_year + 1):
        forecast_start = pd.Timestamp(year=forecast_year, month=1, day=1)
        history = _model_rows(working, [start_year, forecast_year - 1]) if forecast_year > start_year else working.iloc[0:0]
        availability_column = (
            "label_available_at" if "label_available_at" in history else "label_interval_end"
        )
        if availability_column in history:
            availability = pd.to_datetime(history[availability_column], errors="coerce")
            history = history.loc[availability.lt(forecast_start)].copy()
        forecast = _model_rows(working, [forecast_year, forecast_year])
        present_classes = sorted(forecast["target_class"].map(TARGET_TO_INT).dropna().astype(int).unique().tolist())
        history_classes = sorted(history["target_class"].map(TARGET_TO_INT).dropna().astype(int).unique().tolist())
        status = "predicted"
        if forecast.empty:
            status = "skipped_no_forecast_rows"
        elif set(history_classes) != set(CLASS_ORDER):
            status = "skipped_missing_history_or_classes"

        max_label_date = (
            pd.to_datetime(history[availability_column], errors="coerce").max()
            if availability_column in history and not history.empty
            else pd.NaT
        )
        record: dict[str, Any] = {
            "forecast_year": forecast_year,
            "status": status,
            "baseline_fit_start_year": start_year,
            "baseline_fit_end_year": forecast_year - 1,
            "baseline_train_rows": int(len(history)),
            "baseline_train_field_seasons": int(history["field_season"].nunique())
            if "field_season" in history
            else np.nan,
            "baseline_train_classes": json.dumps(history_classes),
            "forecast_rows": int(len(forecast)),
            "forecast_classes_for_evaluation": json.dumps(present_classes),
            "forecast_start_date": forecast_start,
            "baseline_max_issue_date": pd.to_datetime(history["issue_date"]).max()
            if not history.empty
            else pd.NaT,
            "baseline_max_label_availability_date": max_label_date,
            "label_availability_source": availability_column,
            "temporal_order_verified": bool(pd.isna(max_label_date) or max_label_date < forecast_start),
            "model_seed": int(seed) + forecast_year,
        }
        provenance.append(record)
        if status != "predicted":
            continue

        model = fit_model(
            history,
            list(CALENDAR_FEATURES),
            "logistic",
            {"C": 0.1},
            int(seed) + forecast_year,
        )
        features = forecast[list(CALENDAR_FEATURES)]
        raw = c0_raw_logits(model, features)
        probability = aligned_probabilities(model, features)
        output = forecast[["_source_index", *carry]].copy()
        output = output.rename(columns={"_source_index": "source_index"})
        output["forecast_year"] = forecast_year
        output["baseline_fit_end_year"] = forecast_year - 1
        output["baseline_max_label_availability_date"] = max_label_date
        output["target_int"] = forecast["target_class"].map(TARGET_TO_INT).astype(int).to_numpy()
        for position, class_id in enumerate(CLASS_ORDER):
            output[f"c0_raw_{class_id}"] = raw[:, position]
            output[f"c0_probability_{class_id}"] = probability[:, position]
        prediction_parts.append(output)

    if prediction_parts:
        predictions = pd.concat(prediction_parts, ignore_index=True)
        predictions = predictions.sort_values(["forecast_year", "issue_date", "source_index"]).reset_index(drop=True)
    else:
        predictions = pd.DataFrame()
    return TemporalOOFResult(predictions=predictions, provenance=pd.DataFrame(provenance))


def _oof_arrays(oof_predictions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    missing = [column for column in (*RAW_LOGIT_COLUMNS, "target_int") if column not in oof_predictions]
    if missing:
        raise ValueError(f"OOF predictions miss required columns: {missing}")
    baseline = oof_predictions[list(RAW_LOGIT_COLUMNS)].to_numpy(dtype=float)
    target = oof_predictions["target_int"].to_numpy(dtype=int)
    return baseline, target


def fit_weather_catboost_control(oof_predictions: pd.DataFrame, *, seed: int):
    """Fit the main C6 correction on reconstructable episode features."""
    baseline, target = _oof_arrays(oof_predictions)
    available = oof_predictions[list(EPISODE_FEATURES)].notna().all(axis=1).to_numpy()
    if "episode_weather_complete" in oof_predictions:
        available &= oof_predictions["episode_weather_complete"].astype(bool).to_numpy()
    return fit_catboost_correction(
        oof_predictions.loc[available, list(EPISODE_FEATURES)],
        target[available],
        baseline[available],
        seed=seed,
    )


def fit_calendar_catboost_control(oof_predictions: pd.DataFrame, *, seed: int):
    """Fit the calendar-only CatBoost correction on the same OOF population."""
    baseline, target = _oof_arrays(oof_predictions)
    available = oof_predictions[list(CALENDAR_FEATURES)].notna().all(axis=1).to_numpy()
    return fit_catboost_correction(
        oof_predictions.loc[available, list(CALENDAR_FEATURES)],
        target[available],
        baseline[available],
        seed=seed,
    )


def fit_oof_calibration_control(
    oof_predictions: pd.DataFrame,
    *,
    seed: int,
    regularization_c: float = 0.1,
) -> MulticlassCalibrationCorrection:
    baseline, target = _oof_arrays(oof_predictions)
    return fit_calibration_control(
        baseline,
        target,
        seed=seed,
        regularization_c=regularization_c,
    )


@dataclass
class C6Bundle:
    """Serializable C0 plus one predeclared correction and alpha."""

    base_model: Any
    correction_model: Any
    correction_kind: str
    correction_features: tuple[str, ...]
    alpha: float
    base_features: tuple[str, ...] = tuple(CALENDAR_FEATURES)
    availability_column: str | None = None
    policy: Policy | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.alpha = _validate_alpha(self.alpha)
        allowed = {"weather_catboost", "calendar_catboost", "calibration"}
        if self.correction_kind not in allowed:
            raise ValueError(f"Unknown correction kind: {self.correction_kind}")
        if self.correction_kind == "calibration" and self.correction_features:
            raise ValueError("Calibration correction does not consume frame features")

    def predict(self, frame: pd.DataFrame) -> C6Prediction:
        base_values = frame[list(self.base_features)]
        base_raw = c0_raw_logits(self.base_model, base_values)
        base_probability = aligned_probabilities(self.base_model, base_values)
        if self.correction_kind == "calibration":
            return predict_calibration_control(
                base_raw,
                base_probability,
                self.correction_model,
                alpha=self.alpha,
            )
        values = frame[list(self.correction_features)]
        availability: pd.Series | None = None
        if self.availability_column is not None:
            availability = frame[self.availability_column].astype(bool)
        return predict_c6_from_baseline(
            base_logits=base_raw,
            base_probabilities=base_probability,
            correction_model=self.correction_model,
            correction_values=values,
            alpha=self.alpha,
            correction_available=availability,
        )


def save_c6_bundle(bundle: C6Bundle, directory: str | Path) -> None:
    """Save a complete C6 bundle into a new, non-overwriting directory."""
    target = Path(directory)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty bundle directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.base_model, target / "c0.joblib")
    if bundle.correction_kind in {"weather_catboost", "calendar_catboost"}:
        bundle.correction_model.save_model(target / "correction.cbm")
        correction_file = "correction.cbm"
    else:
        joblib.dump(bundle.correction_model, target / "correction.joblib")
        correction_file = "correction.joblib"
    payload = {
        "format_version": 1,
        "class_order": list(CLASS_ORDER),
        "class_names": list(CLASS_NAMES),
        "alpha": bundle.alpha,
        "base_features": list(bundle.base_features),
        "correction_kind": bundle.correction_kind,
        "correction_features": list(bundle.correction_features),
        "availability_column": bundle.availability_column,
        "policy": asdict(bundle.policy) if bundle.policy is not None else None,
        "base_file": "c0.joblib",
        "correction_file": correction_file,
        "shallow_catboost_params": SHALLOW_CATBOOST_PARAMS
        if bundle.correction_kind.endswith("catboost")
        else None,
        "metadata": bundle.metadata,
    }
    (target / "bundle.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_c6_bundle(directory: str | Path) -> C6Bundle:
    """Load a saved bundle and reject incompatible class mappings."""
    source = Path(directory)
    payload = json.loads((source / "bundle.json").read_text(encoding="utf-8"))
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError("Unsupported C6 bundle format")
    if tuple(int(value) for value in payload["class_order"]) != CLASS_ORDER:
        raise ValueError("Saved C6 bundle uses an incompatible class order")
    if tuple(str(value) for value in payload.get("class_names", ())) != CLASS_NAMES:
        raise ValueError("Saved C6 bundle uses an incompatible class-name mapping")
    base_model = joblib.load(source / payload["base_file"])
    if payload["correction_kind"] in {"weather_catboost", "calendar_catboost"}:
        from catboost import CatBoostClassifier

        correction_model = CatBoostClassifier()
        correction_model.load_model(source / payload["correction_file"])
    else:
        correction_model = joblib.load(source / payload["correction_file"])
    return C6Bundle(
        base_model=base_model,
        correction_model=correction_model,
        correction_kind=payload["correction_kind"],
        correction_features=tuple(payload["correction_features"]),
        alpha=float(payload["alpha"]),
        base_features=tuple(payload["base_features"]),
        availability_column=payload.get("availability_column"),
        policy=Policy(**payload["policy"]) if payload.get("policy") is not None else None,
        metadata=payload.get("metadata", {}),
    )
