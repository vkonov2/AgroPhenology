"""Third late-blight cycle: notification policy replay on frozen scores only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from .early_warning_core import sha256_file
from .early_warning_models import (
    MODEL_SPECS,
    _baseline_scores,
    _year_mask,
    burden_metrics,
    event_metrics,
    score_model,
)
from .early_warning_reporting import aggregate_pooled_metrics, paired_year_bootstrap


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V3 = REPO_ROOT / "results/late_blight_early_warning/20260910_first_cycle_v3"
DEFAULT_V4 = REPO_ROOT / "results/late_blight_early_warning/20260910_second_cycle_v4"
SCORE_MODELS = (
    "calendar_window",
    "C0",
    "C1",
    "C4",
    "C5",
    "C6_weather",
    "C6_calibration_control",
    "C6_calendar_control",
)
SCOPES = {
    "service_calendar": None,
    "paired_candidate_days": "candidate_comparison_complete",
}
MAIN_YEARS = (2020, 2025)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any):
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
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _git_text(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else f"unavailable: {completed.stderr.strip()}"


def _environment_versions() -> dict[str, Any]:
    packages = ("numpy", "pandas", "scikit-learn", "pyarrow", "catboost", "joblib", "pytest")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
    }


def verify_frozen_run(run_dir: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    """Verify a complete immutable parent against every recorded output hash."""
    manifest_path = run_dir / "execution_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != expected_manifest_sha256:
        raise AssertionError(
            f"Unexpected parent manifest hash for {run_dir}: {manifest_hash}; "
            f"expected {expected_manifest_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise AssertionError(f"Frozen parent is not complete: {run_dir}")
    checked: list[dict[str, Any]] = []
    failures: list[str] = []
    for relative, metadata in sorted(manifest.get("output_hashes", {}).items()):
        expected = metadata["sha256"] if isinstance(metadata, dict) else str(metadata)
        path = run_dir / relative
        actual = sha256_file(path) if path.is_file() else None
        matches = actual == expected
        checked.append(
            {
                "relative_path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "bytes": path.stat().st_size if path.is_file() else None,
                "matches": matches,
            }
        )
        if not matches:
            failures.append(relative)
    if failures:
        raise AssertionError(f"Frozen parent output hash failures: {failures}")
    return {
        "status": "passed",
        "run_id": manifest.get("run_id"),
        "path": str(run_dir),
        "execution_manifest_sha256": manifest_hash,
        "outputs_checked": len(checked),
        "all_output_hashes_match": True,
        "files": checked,
    }


def _write_source_snapshot(path: Path) -> dict[str, Any]:
    candidates = [
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "requirements-early-warning.txt",
        REPO_ROOT / "docs/research/late_blight_early_warning/protocol.md",
        REPO_ROOT / "docs/research/late_blight_early_warning/evaluation_contract.json",
        REPO_ROOT / "docs/research/late_blight_early_warning/cycle2_evaluation_contract.json",
        REPO_ROOT / "docs/research/late_blight_early_warning/cycle3_protocol.md",
        REPO_ROOT / "docs/research/late_blight_early_warning/cycle3_evaluation_contract.json",
        REPO_ROOT / "src/agro_phenology/early_warning_core.py",
        REPO_ROOT / "src/agro_phenology/early_warning_models.py",
        REPO_ROOT / "src/agro_phenology/early_warning_reporting.py",
        REPO_ROOT / "src/agro_phenology/early_warning_cycle2_models.py",
    ]
    candidates += sorted((REPO_ROOT / "src/agro_phenology").glob("early_warning_cycle3*.py"))
    candidates += sorted((REPO_ROOT / "tests").glob("test_early_warning_cycle3*.py"))
    unique = list(dict.fromkeys(candidates))
    missing = [str(item.relative_to(REPO_ROOT)) for item in unique if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Cycle3 source snapshot is incomplete: {missing}")
    files = [
        {
            "relative_path": str(item.relative_to(REPO_ROOT)),
            "sha256": sha256_file(item),
            "utf8_content": item.read_text(encoding="utf-8"),
        }
        for item in unique
    ]
    payload = {
        "format": "agro_phenology_cycle3_source_snapshot_v1",
        "git_revision": _git_text("rev-parse", "HEAD"),
        "files": files,
    }
    _write_json(path, payload)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "files": [{"relative_path": x["relative_path"], "sha256": x["sha256"]} for x in files],
    }


def _verify_source_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    for recorded in snapshot["files"]:
        relative = str(recorded["relative_path"])
        path = REPO_ROOT / relative
        actual = sha256_file(path) if path.is_file() else None
        matches = actual == recorded["sha256"]
        checks.append(
            {
                "relative_path": relative,
                "expected_sha256": recorded["sha256"],
                "actual_sha256": actual,
                "matches": matches,
            }
        )
        if not matches:
            failures.append(relative)
    if failures:
        raise AssertionError(f"Cycle3 source changed during execution: {failures}")
    return {
        "status": "passed",
        "files_checked": len(checks),
        "all_source_hashes_match": True,
        "checks": checks,
    }


def _load_inputs(v3_dir: Path, v4_dir: Path) -> dict[str, Any]:
    frames: dict[str, Any] = {}
    for name in ("daily_decisions", "field_seasons", "events", "predictions", "alarm_states", "event_hits"):
        frames[f"v3_{name}"] = pd.read_parquet(v3_dir / f"{name}.parquet")
    frames["v3_policy"] = pd.read_csv(v3_dir / "policy_selection.csv")
    for name in ("c6_raw_predictions", "alarm_states", "event_hits", "gained_lost_events"):
        frames[f"v4_{name}"] = pd.read_parquet(v4_dir / f"{name}.parquet")
    frames["v4_policy"] = pd.read_csv(v4_dir / "policy_selection.csv")
    frames["v3_contract"] = json.loads((v3_dir / "evaluation_contract.json").read_text(encoding="utf-8"))
    for key in ("v3_daily_decisions", "v3_field_seasons", "v3_events"):
        for column in (
            "issue_date",
            "issued_at",
            "first_recorded_event_date",
            "first_recorded_event_available_date",
            "first_visit_available_date",
            "last_visit_date",
        ):
            if column in frames[key]:
                frames[key][column] = pd.to_datetime(frames[key][column])
    return frames


def _load_cycle1_model(v3_dir: Path, fold_id: str, model_code: str):
    spec = MODEL_SPECS[model_code]
    if spec["kind"] == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(v3_dir / "models" / f"{fold_id}_{model_code}.cbm")
        return model
    return joblib.load(v3_dir / "models" / f"{fold_id}_{model_code}.joblib")


def _align_saved_score(
    frame: pd.DataFrame,
    saved: pd.DataFrame,
    *,
    score_column: str = "score",
) -> pd.Series:
    keys = ["field_season", "season", "issue_date"]
    left = frame[keys].copy()
    left["_source_index"] = frame.index
    right = saved[keys + [score_column]].copy()
    if right.duplicated(keys).any():
        raise AssertionError("Saved score rows are not unique")
    merged = left.merge(right, on=keys, how="left", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise AssertionError("Saved score population does not match the decision frame")
    return pd.Series(
        merged[score_column].to_numpy(dtype=float), index=merged["_source_index"].to_numpy(), dtype=float
    ).reindex(frame.index)


def _series_equal(left: pd.Series, right: pd.Series, atol: float = 0.0) -> tuple[bool, float]:
    left = left.astype(float)
    right = right.reindex(left.index).astype(float)
    same_missing = left.isna().eq(right.isna()).all()
    finite = left.notna() & right.notna()
    maximum = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
    return bool(same_missing and maximum <= atol), maximum


def _alpha_tag(alpha: float) -> str:
    mapping = {0.0: "0p0", 0.1: "0p1", 0.25: "0p25", 0.5: "0p5", 1.0: "1p0"}
    try:
        return mapping[float(alpha)]
    except KeyError as exc:
        raise ValueError(f"Alpha is not in the frozen v4 grid: {alpha}") from exc


def _selected_c6_configuration(
    policy: pd.DataFrame, fold_id: str, model_code: str, scope: str
) -> pd.Series:
    rows = policy[
        policy["fold_id"].eq(fold_id)
        & policy["model_code"].eq(model_code)
        & policy["evaluation_scope"].eq(scope)
        & policy["policy_mode"].eq("validation_selected")
    ]
    if len(rows) != 1:
        raise AssertionError(f"Expected one frozen C6 configuration, got {len(rows)}")
    return rows.iloc[0]


def _saved_cycle1_policy(
    policy: pd.DataFrame, fold_id: str, model_code: str, scope: str
) -> dict[str, Any]:
    rows = policy[
        policy["fold_id"].eq(fold_id)
        & policy["model_code"].eq(model_code)
        & policy["evaluation_scope"].eq(scope)
    ]
    if len(rows) != 1:
        raise AssertionError(f"Expected one saved v3 policy, got {len(rows)}")
    row = rows.iloc[0]
    return {
        "threshold": float(row["threshold"]),
        "active_days": int(row["active_days"]),
        "cooldown_days": int(row["cooldown_days"]),
        "source": "cycle1_v3",
    }


def _c6_score_from_raw(
    raw: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    fold_id: str,
    model_code: str,
    split: str,
    alpha: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    selected = raw[
        raw["fold_id"].eq(fold_id)
        & raw["model_code"].eq(model_code)
        & raw["split"].eq(split)
    ].copy()
    column = f"c6_probability_2_alpha_{_alpha_tag(alpha)}"
    score = _align_saved_score(frame, selected, score_column=column)
    available = _align_saved_score(
        frame, selected.assign(_available=selected["correction_available"].astype(float)), score_column="_available"
    ).fillna(0).astype(bool)
    fallback = _align_saved_score(
        frame, selected.assign(_fallback=selected["used_c0_fallback"].astype(float)), score_column="_fallback"
    ).fillna(False).astype(bool)
    return score, available, fallback


def _score_hash(frame: pd.DataFrame) -> str:
    columns = [
        "field_season",
        "season",
        "issue_date",
        "score",
        "score_origin",
        "model_version",
        "alpha",
        "evaluation_scope_day",
    ]
    canonical = frame[columns].sort_values(["season", "field_season", "issue_date"], kind="stable")
    digest = hashlib.sha256()
    for row in canonical.itertuples(index=False, name=None):
        values: list[str] = []
        for value in row:
            if isinstance(value, (pd.Timestamp, datetime)):
                values.append(pd.Timestamp(value).isoformat())
            elif value is None or pd.isna(value):
                values.append("NA")
            elif isinstance(value, (float, np.floating)):
                values.append(format(float(value), ".17g"))
            else:
                values.append(str(value))
        digest.update(("\x1f".join(values) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _frozen_score_frame(
    source: pd.DataFrame,
    score: pd.Series,
    *,
    score_origin: pd.Series,
    fold_id: str,
    model_code: str,
    scope: str,
    split: str,
    model_version: str,
    alpha: float | None,
    correction_available: pd.Series,
    fallback_to_c0: pd.Series,
) -> pd.DataFrame:
    mask_column = SCOPES[scope]
    scope_day = source["service_active"].astype(bool)
    effective_score = score.reindex(source.index)
    if mask_column:
        scope_day &= source[mask_column].astype(bool)
        # Cycle 1 masked the score by the paired-comparison mask before replay,
        # while service_active remained a separate state-machine gate.  Keep
        # scores on inactive service days for byte-level semantic parity with
        # the saved predictions; those values can never issue a message.
        effective_score = effective_score.where(source[mask_column].astype(bool))
    effective_origin = score_origin.reindex(source.index).where(effective_score.notna())
    result = source[["field_season", "season", "issue_date", "issued_at", "service_active"]].copy()
    result["evaluation_scope_day"] = scope_day
    result["score"] = effective_score
    result["score_origin"] = effective_origin
    result["fold_id"] = fold_id
    result["score_model"] = model_code
    result["evaluation_scope"] = scope
    result["split"] = split
    result["model_version"] = model_version
    result["alpha"] = np.nan if alpha is None else float(alpha)
    result["correction_available"] = correction_available.reindex(source.index).fillna(False).astype(bool)
    result["fallback_to_c0"] = fallback_to_c0.reindex(source.index).fillna(False).astype(bool)
    return result


def build_frozen_scores(
    inputs: dict[str, Any], v3_dir: Path
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[str, str, str, str], dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    """Recover validation scores and freeze every score used by cycle3.

    No model is fitted here.  Saved v3 models are loaded only because v3 did not
    persist its outer-validation scores.  Their test predictions must exactly
    reproduce the already saved v3 predictions before any policy is evaluated.
    """
    decisions = inputs["v3_daily_decisions"]
    seasons = inputs["v3_field_seasons"]
    v3_predictions = inputs["v3_predictions"]
    v4_raw = inputs["v4_c6_raw_predictions"]
    v4_states = inputs["v4_alarm_states"]
    v3_policy = inputs["v3_policy"]
    v4_policy = inputs["v4_policy"]
    folds = inputs["v3_contract"]["rolling_origin_folds"]
    frozen_parts: list[pd.DataFrame] = []
    checks: list[dict[str, Any]] = []
    runtime: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    saved_policies: dict[tuple[str, str, str], dict[str, Any]] = {}

    for fold in folds:
        fold_id = str(fold["id"])
        validation = decisions[_year_mask(decisions, fold["validation_years"])].copy()
        test = decisions[_year_mask(decisions, fold["test_years"])].copy()
        train_seasons = seasons[_year_mask(seasons, fold["train_years"])].copy()
        split_frames = {"outer_validation": validation, "outer_test": test}

        cycle1_scores: dict[str, dict[str, pd.Series]] = {}
        for model_code in ("C0", "C1", "C4", "C5"):
            spec = MODEL_SPECS[model_code]
            model = _load_cycle1_model(v3_dir, fold_id, model_code)
            cycle1_scores[model_code] = {
                split: score_model(model, frame, spec["features"], spec["availability"])
                for split, frame in split_frames.items()
            }
        cycle1_scores["calendar_window"] = {
            split: _baseline_scores(frame, train_seasons)["calendar_window"]
            for split, frame in split_frames.items()
        }

        for model_code in ("calendar_window", "C0", "C1", "C4", "C5"):
            model_version = (
                "cycle1_deterministic_calendar_window_v1"
                if model_code == "calendar_window"
                else "cycle1_fixed_train_only_v1"
            )
            for scope, mask_column in SCOPES.items():
                saved_policies[(fold_id, model_code, scope)] = _saved_cycle1_policy(
                    v3_policy, fold_id, model_code, scope
                )
                for split, source in split_frames.items():
                    score = cycle1_scores[model_code][split]
                    origin = pd.Series(model_code, index=source.index, dtype=object)
                    available = score.notna()
                    fallback = pd.Series(False, index=source.index, dtype=bool)
                    frozen = _frozen_score_frame(
                        source,
                        score,
                        score_origin=origin,
                        fold_id=fold_id,
                        model_code=model_code,
                        scope=scope,
                        split=split,
                        model_version=model_version,
                        alpha=None,
                        correction_available=available,
                        fallback_to_c0=fallback,
                    )
                    frozen_parts.append(frozen)
                    runtime[(fold_id, model_code, scope, split)] = {
                        "source": source,
                        "score": pd.Series(frozen["score"].to_numpy(), index=source.index, dtype=float),
                        "origin": pd.Series(frozen["score_origin"].to_numpy(), index=source.index, dtype=object),
                        "correction_available": pd.Series(
                            frozen["correction_available"].to_numpy(), index=source.index, dtype=bool
                        ),
                        "fallback": pd.Series(frozen["fallback_to_c0"].to_numpy(), index=source.index, dtype=bool),
                        "alpha": None,
                        "model_version": model_version,
                    }
                    if split == "outer_test":
                        saved = v3_predictions[
                            v3_predictions["fold_id"].eq(fold_id)
                            & v3_predictions["model_code"].eq(model_code)
                            & v3_predictions["evaluation_scope"].eq(scope)
                        ]
                        expected = _align_saved_score(source, saved)
                        actual = runtime[(fold_id, model_code, scope, split)]["score"]
                        equal, maximum = _series_equal(actual, expected, atol=0.0)
                        checks.append(
                            {
                                "fold_id": fold_id,
                                "score_model": model_code,
                                "evaluation_scope": scope,
                                "split": split,
                                "source": "cycle1_saved_model_vs_v3_saved_prediction",
                                "rows": len(source),
                                "status": "passed" if equal else "failed",
                                "max_abs_score_difference": maximum,
                                "alpha": np.nan,
                            }
                        )
                        if not equal:
                            raise AssertionError(f"Frozen {model_code} score mismatch in {fold_id}/{scope}")

        for model_code in ("C6_weather", "C6_calibration_control", "C6_calendar_control"):
            for scope in SCOPES:
                selected = _selected_c6_configuration(v4_policy, fold_id, model_code, scope)
                alpha = float(selected["alpha"])
                saved_policies[(fold_id, model_code, scope)] = {
                    "threshold": float(selected["threshold"]),
                    "active_days": int(selected["active_days"]),
                    "cooldown_days": int(selected["cooldown_days"]),
                    "source": "cycle2_v4_validation_selected",
                    "alpha": alpha,
                }
                for split, source in split_frames.items():
                    score, correction_available, fallback = _c6_score_from_raw(
                        v4_raw,
                        source,
                        fold_id=fold_id,
                        model_code=model_code,
                        split=split,
                        alpha=alpha,
                    )
                    # Cycle 2 converted raw C6 predictions into operational
                    # scores by masking inactive service days before policy
                    # selection and replay.  Reproduce that frozen boundary.
                    score = score.where(source["service_active"].astype(bool))
                    if alpha == 0.0:
                        # No weather correction participates in the selected
                        # score, so missing weather is not a weather->C0
                        # fallback.  Keep correction_available separately as
                        # a coverage diagnostic.
                        fallback = pd.Series(False, index=source.index, dtype=bool)
                    if alpha == 0.0:
                        origin = pd.Series("C0", index=source.index, dtype=object)
                    else:
                        origin = pd.Series(
                            np.where(correction_available, model_code, "C0_fallback"),
                            index=source.index,
                            dtype=object,
                        )
                    model_version = "cycle2_nested_temporal_oof_v1"
                    frozen = _frozen_score_frame(
                        source,
                        score,
                        score_origin=origin,
                        fold_id=fold_id,
                        model_code=model_code,
                        scope=scope,
                        split=split,
                        model_version=model_version,
                        alpha=alpha,
                        correction_available=correction_available,
                        fallback_to_c0=fallback,
                    )
                    frozen_parts.append(frozen)
                    runtime[(fold_id, model_code, scope, split)] = {
                        "source": source,
                        "score": pd.Series(frozen["score"].to_numpy(), index=source.index, dtype=float),
                        "origin": pd.Series(frozen["score_origin"].to_numpy(), index=source.index, dtype=object),
                        "correction_available": pd.Series(
                            frozen["correction_available"].to_numpy(), index=source.index, dtype=bool
                        ),
                        "fallback": pd.Series(frozen["fallback_to_c0"].to_numpy(), index=source.index, dtype=bool),
                        "alpha": alpha,
                        "model_version": model_version,
                    }
                    if split == "outer_test":
                        prior = v4_states[
                            v4_states["fold_id"].eq(fold_id)
                            & v4_states["model_family"].eq(model_code)
                            & v4_states["policy_mode"].eq("validation_selected")
                            & v4_states["evaluation_scope"].eq(scope)
                        ]
                        expected = _align_saved_score(source, prior)
                        actual = runtime[(fold_id, model_code, scope, split)]["score"]
                        equal, maximum = _series_equal(actual, expected, atol=0.0)
                        checks.append(
                            {
                                "fold_id": fold_id,
                                "score_model": model_code,
                                "evaluation_scope": scope,
                                "split": split,
                                "source": "cycle2_raw_prediction_vs_v4_validation_selected_score",
                                "rows": len(source),
                                "status": "passed" if equal else "failed",
                                "max_abs_score_difference": maximum,
                                "alpha": alpha,
                            }
                        )
                        if not equal:
                            raise AssertionError(f"Frozen {model_code} score mismatch in {fold_id}/{scope}")

    frozen_scores = pd.concat(frozen_parts, ignore_index=True)
    group_columns = ["fold_id", "score_model", "evaluation_scope", "split"]
    hashes: list[dict[str, Any]] = []
    for key, group in frozen_scores.groupby(group_columns, sort=True, dropna=False):
        hashes.append(
            {
                **dict(zip(group_columns, key)),
                "rows": int(len(group)),
                "computed_rows": int(group["score"].notna().sum()),
                "score_sha256": _score_hash(group),
                "alpha": float(group["alpha"].dropna().iloc[0]) if group["alpha"].notna().any() else np.nan,
            }
        )
    return frozen_scores, pd.DataFrame(hashes), pd.DataFrame(checks), runtime, saved_policies


def threshold_grid(score: pd.Series, saved_threshold: float) -> list[float]:
    finite = score.dropna().astype(float)
    values = set(np.linspace(0.05, 0.95, 19).tolist() + [1.000001, float(saved_threshold)])
    if len(finite):
        values.update(float(value) for value in finite.quantile(np.linspace(0.05, 0.95, 19)).unique())
    return sorted(values)


def _constraint_fields(record: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    budget = contract["notification_policy"]["research_budget"]
    result = dict(record)
    violation = max(
        0.0,
        float(result["messages_per_30_field_days"])
        - float(budget["messages_per_30_field_days_max"]),
    ) + max(
        0.0,
        float(result["active_alarm_fraction"])
        - float(budget["active_alarm_fraction_max"]),
    )
    result["constraint_violation"] = violation
    result["feasible"] = bool(violation <= 1e-12)
    return result


def _selection_key(record: dict[str, Any], any_feasible: bool) -> tuple[Any, ...]:
    delta = record.get("growth_logit_delta")
    delta_rank = float(delta) if delta is not None and np.isfinite(delta) else float("inf")
    return (
        0.0 if any_feasible else float(record["constraint_violation"]),
        -float(np.nan_to_num(record["timely_recall"], nan=-1.0)),
        float(record["messages_per_30_field_days"]),
        float(record["active_alarm_fraction"]),
        int(bool(record.get("growth_override_enabled", False))),
        -delta_rank,
        -float(record["threshold"]),
    )


def select_policy_record(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("Policy candidate set is empty")
    feasible = [record for record in rows if bool(record["feasible"])]
    pool = feasible or rows
    selected = dict(sorted(pool, key=lambda record: _selection_key(record, bool(feasible)))[0])
    selected["selection_status"] = "feasible" if feasible else "no_feasible_policy"
    return selected


def _candidate_metrics(
    states: pd.DataFrame,
    seasons: pd.DataFrame,
    *,
    public_code: str,
    fold_id: str,
    scope: str,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    event, hits = event_metrics(
        states,
        seasons,
        public_code,
        fold_id,
        slice_name="A_plus_B",
        evaluation_scope=scope,
    )
    burden = burden_metrics(
        states,
        public_code,
        fold_id,
        slice_name="A_plus_B",
        evaluation_scope=scope,
    )
    selected = states[states["evaluation_scope_day"].astype(bool)]
    fallback_days = int(
        selected.get("fallback_to_c0", pd.Series(False, index=selected.index))
        .fillna(False)
        .astype(bool)
        .sum()
    )
    correction_unavailable_days = int(
        (~selected.get("correction_available", pd.Series(True, index=selected.index))
         .fillna(False)
         .astype(bool)).sum()
    )
    effective_c0_origin_days = int(
        selected.get("effective_c0_origin", pd.Series(False, index=selected.index))
        .fillna(False)
        .astype(bool)
        .sum()
    )
    record = {
        **event,
        **burden,
        "growth_messages": int(
            selected.get("growth_override_used", pd.Series(False, index=selected.index))
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "growth_reference_not_comparable_days": int(
            selected["action_reason"].eq("growth_reference_not_comparable").sum()
        ),
        "fallback_days": fallback_days,
        "fallback_day_fraction": fallback_days / len(selected) if len(selected) else np.nan,
        "correction_unavailable_days": correction_unavailable_days,
        "correction_unavailable_fraction": (
            correction_unavailable_days / len(selected) if len(selected) else np.nan
        ),
        "effective_c0_origin_days": effective_c0_origin_days,
        "effective_c0_origin_fraction": (
            effective_c0_origin_days / len(selected) if len(selected) else np.nan
        ),
        "external_budget_exceeded": bool(
            burden["messages_per_30_field_days"]
            > float(contract["notification_policy"]["research_budget"]["messages_per_30_field_days_max"])
            + 1e-12
            or burden["active_alarm_fraction"]
            > float(contract["notification_policy"]["research_budget"]["active_alarm_fraction_max"])
            + 1e-12
        ),
    }
    return record, hits


def _enrich_pooled_burden(
    pooled: pd.DataFrame, annual: pd.DataFrame
) -> pd.DataFrame:
    if pooled.empty:
        return pooled
    result = pooled.copy()
    for index, row in result.iterrows():
        subset = annual[
            annual["season"].between(int(row["year_start"]), int(row["year_end"]))
            & annual["model_code"].eq(row["model_code"])
            & annual["evaluation_scope"].eq(row["evaluation_scope"])
            & annual["slice"].eq(row["slice"])
        ]
        fallback = int(subset.get("fallback_days", pd.Series(dtype=float)).sum())
        growth = int(subset.get("growth_messages", pd.Series(dtype=float)).sum())
        incomparable = int(
            subset.get("growth_reference_not_comparable_days", pd.Series(dtype=float)).sum()
        )
        result.loc[index, "fallback_days"] = fallback
        result.loc[index, "fallback_day_fraction"] = (
            fallback / float(row["field_days"]) if row["field_days"] else np.nan
        )
        result.loc[index, "growth_messages"] = growth
        result.loc[index, "growth_reference_not_comparable_days"] = incomparable
        for count_column, fraction_column in (
            ("correction_unavailable_days", "correction_unavailable_fraction"),
            ("effective_c0_origin_days", "effective_c0_origin_fraction"),
        ):
            count = int(subset.get(count_column, pd.Series(dtype=float)).sum())
            result.loc[index, count_column] = count
            result.loc[index, fraction_column] = (
                count / float(row["field_days"]) if row["field_days"] else np.nan
            )
        result.loc[index, "external_years_over_budget"] = int(
            subset.get("external_budget_exceeded", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
        )
    return result


def _event_key(field_season: Any, season: int) -> str:
    raw = f"late_blight_cycle3_policy\x1f{int(season)}\x1f{field_season}".encode("utf-8")
    return "evt3_" + hashlib.sha256(raw).hexdigest()[:20]


def build_gained_lost(
    event_hits_frame: pd.DataFrame,
    states_frame: pd.DataFrame,
    seasons: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build eligible-only transitions and causal-path diagnostics."""
    policy_pairs = (
        ("P0_selected", "P0_saved"),
        ("P_growth_selected", "P0_saved"),
        ("P_short_selected", "P0_saved"),
        ("P_growth_selected", "P0_selected"),
        ("P_growth_selected", "P_short_selected"),
        ("P_growth_selected", "P0_growth_theta_control"),
    )
    event_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    hit_main = event_hits_frame[
        event_hits_frame["season"].between(*MAIN_YEARS)
        & event_hits_frame["slice"].eq("A_plus_B")
    ].copy()
    states_main = states_frame[states_frame["season"].between(*MAIN_YEARS)].copy()
    event_dates = seasons[["field_season", "season", "first_recorded_event_date"]].copy()
    event_dates["first_recorded_event_date"] = pd.to_datetime(event_dates["first_recorded_event_date"])

    for (score_model, scope), model_hits in hit_main.groupby(
        ["score_model", "evaluation_scope"], sort=True
    ):
        for candidate_family, baseline_family in policy_pairs:
            candidate = model_hits[model_hits["policy_family"].eq(candidate_family)].copy()
            baseline = model_hits[model_hits["policy_family"].eq(baseline_family)].copy()
            if candidate.empty or baseline.empty:
                continue
            keys = ["field_season", "season"]
            if baseline.duplicated(keys).any() or candidate.duplicated(keys).any():
                raise AssertionError("Duplicate event rows prevent gained/lost pairing")
            paired = candidate.merge(
                baseline,
                on=keys,
                how="outer",
                suffixes=("_candidate", "_baseline"),
                indicator=True,
                validate="one_to_one",
            )
            if not paired["_merge"].eq("both").all():
                raise AssertionError("Policy event populations do not match")
            if not paired["warnable_event_candidate"].eq(paired["warnable_event_baseline"]).all():
                raise AssertionError("Policy warning opportunities do not match")
            eligible = paired[paired["warnable_event_candidate"].astype(bool)].copy()
            candidate_hit = eligible["timely_hit_candidate"].astype(bool)
            baseline_hit = eligible["timely_hit_baseline"].astype(bool)
            eligible["transition"] = np.select(
                [candidate_hit & baseline_hit, candidate_hit & ~baseline_hit, ~candidate_hit & baseline_hit],
                ["both_hit", "gained", "lost"],
                default="neither_hit",
            )
            eligible = eligible.merge(event_dates, on=keys, how="left", validate="one_to_one")
            candidate_code = f"{score_model}__{candidate_family}"
            baseline_code = f"{score_model}__{baseline_family}"
            candidate_states = states_main[
                states_main["model_code"].eq(candidate_code)
                & states_main["evaluation_scope"].eq(scope)
            ]
            baseline_states = states_main[
                states_main["model_code"].eq(baseline_code)
                & states_main["evaluation_scope"].eq(scope)
            ]
            for row in eligible.itertuples(index=False):
                transition = str(row.transition)
                reason = "unchanged_event_status"
                event_date = pd.Timestamp(row.first_recorded_event_date)
                window_start = event_date - pd.Timedelta(days=10)
                window_end = event_date - pd.Timedelta(days=3)
                cdays = candidate_states[
                    candidate_states["field_season"].eq(row.field_season)
                    & candidate_states["issue_date"].between(window_start, window_end)
                ]
                bdays = baseline_states[
                    baseline_states["field_season"].eq(row.field_season)
                    & baseline_states["issue_date"].between(window_start, window_end)
                ]
                candidate_field = candidate_states[
                    candidate_states["field_season"].eq(row.field_season)
                ]
                candidate_issued = cdays[cdays["message_issued"].astype(bool)]
                baseline_issued = bdays[bdays["message_issued"].astype(bool)]
                candidate_growth_flag = candidate_field.get(
                    "growth_override_used",
                    candidate_field["action_reason"].eq("issued_growth_override"),
                ).fillna(False).astype(bool)
                candidate_growth = candidate_field[
                    candidate_field["message_issued"].astype(bool)
                    & candidate_growth_flag
                    & candidate_field["issue_date"].lt(event_date)
                ]
                same_theta_mechanism = bool(
                    candidate_family == "P_growth_selected"
                    and baseline_family == "P0_growth_theta_control"
                )
                causal_growth_dates: set[pd.Timestamp] = set()
                if transition == "gained":
                    timely_growth_flag = candidate_issued.get(
                        "growth_override_used",
                        candidate_issued["action_reason"].eq("issued_growth_override"),
                    ).fillna(False).astype(bool)
                    timely_growth = candidate_issued[timely_growth_flag]
                    if len(timely_growth) and same_theta_mechanism:
                        causal_growth_dates.update(pd.to_datetime(timely_growth["issue_date"]))
                        reason = "growth_override_created_timely_message_same_theta"
                    elif same_theta_mechanism:
                        causal_growth_dates.update(pd.to_datetime(candidate_growth["issue_date"]))
                        reason = "growth_sequence_shifted_ordinary_message_into_window_same_theta"
                    elif len(timely_growth):
                        reason = "timely_growth_message_with_threshold_and_schedule_differences"
                    elif candidate_family == "P_short_selected":
                        reason = "shorter_cooldown_or_retuned_threshold_created_timely_message"
                    else:
                        reason = "retuned_threshold_or_sequential_schedule_created_timely_message"
                elif transition == "lost":
                    baseline_dates = set(pd.to_datetime(baseline_issued["issue_date"]))
                    suppressed_on_old_message = cdays[
                        cdays["issue_date"].isin(baseline_dates)
                        & cdays["suppressed_repeat"].astype(bool)
                    ]
                    for suppressed_row in suppressed_on_old_message.itertuples(index=False):
                        previous_date = pd.to_datetime(
                            getattr(suppressed_row, "previous_message_date", pd.NaT)
                        )
                        if pd.isna(previous_date):
                            continue
                        previous = candidate_field[
                            candidate_field["issue_date"].eq(previous_date)
                            & candidate_field["message_issued"].astype(bool)
                            & candidate_growth_flag
                        ]
                        if len(previous):
                            causal_growth_dates.add(pd.Timestamp(previous_date))
                    if causal_growth_dates and same_theta_mechanism:
                        reason = "earlier_growth_message_caused_cooldown_on_baseline_timely_date_same_theta"
                    elif same_theta_mechanism:
                        reason = "growth_sequence_changed_schedule_without_direct_cooldown_match_same_theta"
                    else:
                        reason = "retuned_threshold_or_sequential_schedule_lost_timely_message"

                def _lead_days(days: Iterable[Any]) -> str:
                    leads = sorted(
                        {int((event_date - pd.Timestamp(day)).days) for day in days},
                        reverse=True,
                    )
                    return json.dumps(leads, separators=(",", ":"))

                event_rows.append(
                    {
                        "event_key": _event_key(row.field_season, int(row.season)),
                        "season": int(row.season),
                        "score_model": score_model,
                        "evaluation_scope": scope,
                        "candidate_policy": candidate_family,
                        "baseline_policy": baseline_family,
                        "transition": transition,
                        "reason": reason,
                        "same_theta_mechanism_comparison": same_theta_mechanism,
                        "candidate_timely_message_lead_days": _lead_days(
                            candidate_issued["issue_date"]
                        ),
                        "baseline_timely_message_lead_days": _lead_days(
                            baseline_issued["issue_date"]
                        ),
                        "candidate_growth_message_lead_days_before_event": _lead_days(
                            candidate_growth["issue_date"]
                        ),
                        "causal_growth_message_lead_days": _lead_days(causal_growth_dates),
                        "candidate_threshold": (
                            float(candidate_field["policy_threshold"].iloc[0])
                            if len(candidate_field)
                            else np.nan
                        ),
                        "baseline_threshold": (
                            float(
                                baseline_states.loc[
                                    baseline_states["field_season"].eq(row.field_season),
                                    "policy_threshold",
                                ].iloc[0]
                            )
                            if baseline_states["field_season"].eq(row.field_season).any()
                            else np.nan
                        ),
                    }
                )
            counts = eligible["transition"].value_counts()
            both = int(counts.get("both_hit", 0))
            gained = int(counts.get("gained", 0))
            lost = int(counts.get("lost", 0))
            neither = int(counts.get("neither_hit", 0))
            candidate_hits = int(candidate_hit.sum())
            baseline_hits = int(baseline_hit.sum())
            eligible_count = int(len(eligible))
            invariants = {
                "partition": both + gained + lost + neither == eligible_count,
                "candidate": both + gained == candidate_hits,
                "baseline": both + lost == baseline_hits,
                "delta": gained - lost == candidate_hits - baseline_hits,
            }
            if not all(invariants.values()):
                raise AssertionError(f"Gained/lost invariant failure: {invariants}")
            summaries.append(
                {
                    "score_model": score_model,
                    "evaluation_scope": scope,
                    "candidate_policy": candidate_family,
                    "baseline_policy": baseline_family,
                    "eligible_events": eligible_count,
                    "both": both,
                    "gained": gained,
                    "lost": lost,
                    "neither": neither,
                    "candidate_timely_events": candidate_hits,
                    "baseline_timely_events": baseline_hits,
                    "net_gain": gained - lost,
                    "all_invariants_pass": all(invariants.values()),
                }
            )

    registry = seasons[
        seasons["season"].between(*MAIN_YEARS)
        & seasons["first_recorded_event_date"].notna()
    ].copy()
    registry["event_key"] = [
        _event_key(field_season, int(season))
        for field_season, season in zip(registry["field_season"], registry["season"])
    ]
    registry["eligible_for_warning"] = registry["warnable_first_event"].astype(bool)
    registry["eligibility_reason"] = np.where(
        registry["eligible_for_warning"],
        "eligible_after_connection_with_actionable_window",
        "first_positive_known_at_entry",
    )
    population = registry[
        [
            "event_key",
            "season",
            "eligible_for_warning",
            "positive_at_first_visit",
            "eligibility_reason",
        ]
    ].copy()
    if len(population) != 127 or int(population["eligible_for_warning"].sum()) != 87:
        raise AssertionError("The frozen 2020-2025 event population is not 127/87")
    return pd.DataFrame(event_rows), pd.DataFrame(summaries), population


def build_leave_one_year_out(
    event_hits_frame: pd.DataFrame,
    states_frame: pd.DataFrame,
    comparisons: list[tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    events = event_hits_frame[
        event_hits_frame["season"].between(*MAIN_YEARS)
        & event_hits_frame["slice"].eq("A_plus_B")
    ]
    days = states_frame[states_frame["season"].between(*MAIN_YEARS)]
    for scope in SCOPES:
        for candidate, baseline in comparisons:
            ce = events[events["model_code"].eq(candidate) & events["evaluation_scope"].eq(scope)]
            be = events[events["model_code"].eq(baseline) & events["evaluation_scope"].eq(scope)]
            if ce.empty or be.empty:
                continue
            merged = ce[["season", "field_season", "warnable_event", "timely_hit"]].merge(
                be[["season", "field_season", "warnable_event", "timely_hit"]],
                on=["season", "field_season"],
                suffixes=("_candidate", "_baseline"),
                validate="one_to_one",
            )
            for excluded in range(MAIN_YEARS[0], MAIN_YEARS[1] + 1):
                keep = merged[merged["season"].ne(excluded)]
                eligible = keep["warnable_event_candidate"].astype(bool)
                opportunity = int(eligible.sum())
                candidate_hits = int((keep["timely_hit_candidate"].astype(bool) & eligible).sum())
                baseline_hits = int((keep["timely_hit_baseline"].astype(bool) & eligible).sum())
                selected_days = days[
                    days["evaluation_scope"].eq(scope)
                    & days["season"].ne(excluded)
                    & days["evaluation_scope_day"].astype(bool)
                ]
                cd = selected_days[selected_days["model_code"].eq(candidate)]
                bd = selected_days[selected_days["model_code"].eq(baseline)]
                if len(cd) != len(bd):
                    raise AssertionError("LOO comparison has nonmatching day populations")
                rows.append(
                    {
                        "evaluation_scope": scope,
                        "candidate": candidate,
                        "baseline": baseline,
                        "excluded_year": excluded,
                        "events": opportunity,
                        "candidate_hits": candidate_hits,
                        "baseline_hits": baseline_hits,
                        "delta_timely_recall": (candidate_hits - baseline_hits) / opportunity,
                        "candidate_messages_per_30": 30 * int(cd["message_issued"].sum()) / len(cd),
                        "baseline_messages_per_30": 30 * int(bd["message_issued"].sum()) / len(bd),
                        "candidate_alarm_fraction": float(cd["alarm_active"].mean()),
                        "baseline_alarm_fraction": float(bd["alarm_active"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def _output_hashes(run_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        relative = str(path.relative_to(run_dir))
        if relative == "execution_manifest.json":
            continue
        result[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def _write_frame(run_dir: Path, name: str, frame: pd.DataFrame, parquet: bool = False) -> None:
    if parquet:
        frame.to_parquet(run_dir / f"{name}.parquet", index=False)
    else:
        frame.to_csv(run_dir / f"{name}.csv", index=False)


def _run_test_commands() -> dict[str, Any]:
    commands = [
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_early_warning_cycle3_policy.py",
            "tests/test_early_warning_cycle3_audits.py",
            "tests/test_early_warning_cycle3_pipeline.py",
        ],
        [sys.executable, "-m", "pytest", "-q", "tests"],
    ]
    records: list[dict[str, Any]] = []
    for command in commands:
        completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
        records.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    return {
        "status": "passed" if all(record["returncode"] == 0 for record in records) else "failed",
        "commands": records,
    }


def _fast_validation_metrics(
    states: pd.DataFrame, seasons: pd.DataFrame, contract: dict[str, Any]
) -> dict[str, Any]:
    eligible_fields = set(
        seasons.loc[
            seasons["first_recorded_event_date"].notna()
            & seasons["warnable_first_event"].astype(bool),
            "field_season",
        ].astype(str)
    )
    timely = states[
        states["message_issued"].astype(bool)
        & states["days_to_first_recorded_event"].between(3, 10)
    ]
    hit_fields = set(timely["field_season"].astype(str)) & eligible_fields
    selected = states[states["evaluation_scope_day"].astype(bool)]
    field_days = int(len(selected))
    messages = int(selected["message_issued"].sum())
    alarm_days = int(selected["alarm_active"].sum())
    season_count = int(selected["field_season"].nunique())
    record = {
        "timely_hits": len(hit_fields),
        "events_with_warning_opportunity": len(eligible_fields),
        "timely_recall": len(hit_fields) / len(eligible_fields) if eligible_fields else np.nan,
        "field_days": field_days,
        "field_seasons": season_count,
        "messages": messages,
        "messages_per_30_field_days": 30 * messages / field_days if field_days else np.nan,
        "active_alarm_days": alarm_days,
        "active_alarm_fraction": alarm_days / field_days if field_days else np.nan,
        "suppressed_repeats": int(selected["suppressed_repeat"].sum()),
        "computable_days": int(selected["score"].notna().sum()),
        "computable_fraction": float(selected["score"].notna().mean()) if field_days else np.nan,
        "growth_messages": int(selected["growth_override_used"].sum()),
        "growth_reference_not_comparable_days": int(
            selected["action_reason"].eq("growth_reference_not_comparable").sum()
        ),
    }
    return _constraint_fields(record, contract)


def _make_growth_policy(
    *,
    threshold: float,
    active_days: int,
    cooldown_days: int,
    minimum_interval: int,
    enabled: bool,
    delta: float | None,
    epsilon: float,
    version: str,
):
    from .early_warning_cycle3_policy import GrowthPolicy

    return GrowthPolicy(
        threshold=float(threshold),
        active_days=int(active_days),
        cooldown_days=int(cooldown_days),
        minimum_repeat_interval_days=int(minimum_interval),
        growth_override_enabled=bool(enabled),
        growth_logit_delta=float(delta) if enabled and delta is not None else None,
        epsilon=float(epsilon),
        version=version,
    )


def _simulate_runtime(
    runtime: dict[str, Any],
    policy: Any,
    *,
    score_model: str,
    scope: str,
) -> pd.DataFrame:
    from .early_warning_cycle3_policy import simulate_growth_policy

    source = runtime["source"]
    mask_column = SCOPES[scope]
    evaluation_mask = source[mask_column] if mask_column else source["service_active"]
    alpha = runtime["alpha"]
    effective_c0 = bool(score_model.startswith("C6_") and alpha == 0.0)
    effective_model_id = "C0" if effective_c0 else score_model
    effective_model_version = (
        "cycle1_fixed_train_only_v1" if effective_c0 else runtime["model_version"]
    )
    states, _ = simulate_growth_policy(
        source,
        runtime["score"],
        policy,
        evaluation_scope=scope,
        evaluation_mask=evaluation_mask,
        score_origin=runtime["origin"],
        model_id=effective_model_id,
        model_version=effective_model_version,
    )
    states["correction_available"] = (
        runtime["correction_available"].reindex(states.index).fillna(False).astype(bool)
    )
    states["fallback_to_c0"] = (
        runtime["fallback"].reindex(states.index).fillna(False).astype(bool)
    )
    states["effective_c0_origin"] = states["score_origin"].isin(["C0", "C0_fallback"])
    return states


def _configuration_from_record(record: dict[str, Any], contract: dict[str, Any], version: str):
    settings = contract["notification_policy"]
    return _make_growth_policy(
        threshold=float(record["threshold"]),
        active_days=int(record.get("active_days", settings["active_days_per_message"])),
        cooldown_days=int(record["cooldown_days"]),
        minimum_interval=int(settings["minimum_growth_repeat_interval_days"]),
        enabled=bool(record.get("growth_override_enabled", False)),
        delta=record.get("growth_logit_delta"),
        epsilon=float(settings["logit_epsilon"]),
        version=version,
    )


def _verify_p0_saved_identity(
    new_states: pd.DataFrame,
    old_states: pd.DataFrame,
    *,
    fold_id: str,
    score_model: str,
    scope: str,
) -> dict[str, Any]:
    keys = ["field_season", "season", "issue_date"]
    columns = ["score", "message_issued", "alarm_active", "suppressed_repeat", "active_from", "active_through"]
    left = new_states[keys + columns].copy()
    right = old_states[keys + columns].copy()
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise AssertionError("P0 identity comparison contains duplicate days")
    paired = left.merge(right, on=keys, suffixes=("_new", "_old"), validate="one_to_one", indicator=True)
    population = len(paired) == len(left) == len(right) and paired["_merge"].eq("both").all()
    score_new = paired["score_new"].astype(float)
    score_old = paired["score_old"].astype(float)
    score_missing = score_new.isna().eq(score_old.isna()).all()
    finite = score_new.notna() & score_old.notna()
    max_score = float(np.max(np.abs(score_new[finite] - score_old[finite]))) if finite.any() else 0.0
    booleans = all(
        paired[f"{column}_new"].fillna(False).astype(bool).eq(
            paired[f"{column}_old"].fillna(False).astype(bool)
        ).all()
        for column in ("message_issued", "alarm_active", "suppressed_repeat")
    )
    dates = all(
        (
            pd.to_datetime(paired[f"{column}_new"]).eq(
                pd.to_datetime(paired[f"{column}_old"])
            )
            | (
                pd.to_datetime(paired[f"{column}_new"]).isna()
                & pd.to_datetime(paired[f"{column}_old"]).isna()
            )
        ).all()
        for column in ("active_from", "active_through")
    )
    passed = bool(population and score_missing and max_score == 0.0 and booleans and dates)
    return {
        "fold_id": fold_id,
        "score_model": score_model,
        "evaluation_scope": scope,
        "rows": len(left),
        "status": "passed" if passed else "failed",
        "same_population": population,
        "max_abs_score_difference": max_score,
        "same_messages_alarm_suppression": booleans,
        "same_alarm_boundaries": dates,
    }


def run_policy_experiments(
    inputs: dict[str, Any],
    runtime: dict[tuple[str, str, str, str], dict[str, Any]],
    saved_policies: dict[tuple[str, str, str], dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    seasons = inputs["v3_field_seasons"]
    settings = contract["notification_policy"]
    active_days = int(settings["active_days_per_message"])
    cooldown = int(settings["cooldown_days"])
    minimum = int(settings["minimum_growth_repeat_interval_days"])
    epsilon = float(settings["logit_epsilon"])
    growth_options = contract["notification_policy"]["growth_deltas"]
    folds = inputs["v3_contract"]["rolling_origin_folds"]
    validation_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    alarm_parts: list[pd.DataFrame] = []
    event_metric_rows: list[dict[str, Any]] = []
    event_hit_rows: list[dict[str, Any]] = []
    burden_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []

    for fold in folds:
        fold_id = str(fold["id"])
        validation_seasons = seasons[_year_mask(seasons, fold["validation_years"])].copy()
        test_seasons = seasons[_year_mask(seasons, fold["test_years"])].copy()
        for score_model in SCORE_MODELS:
            for scope in SCOPES:
                print(f"cycle3 selection {fold_id} {score_model} {scope}", flush=True)
                validation_runtime = runtime[(fold_id, score_model, scope, "outer_validation")]
                saved = saved_policies[(fold_id, score_model, scope)]
                threshold_population = validation_runtime["source"]["service_active"].astype(bool)
                if SCOPES[scope]:
                    threshold_population &= validation_runtime["source"][SCOPES[scope]].astype(bool)
                thresholds = threshold_grid(
                    validation_runtime["score"].where(threshold_population),
                    float(saved["threshold"]),
                )
                cache: dict[tuple[float, int, bool, float | None], tuple[pd.DataFrame, dict[str, Any]]] = {}

                def evaluate(threshold: float, candidate_cooldown: int, enabled: bool, delta: float | None):
                    key = (float(threshold), int(candidate_cooldown), bool(enabled), None if delta is None else float(delta))
                    if key not in cache:
                        policy = _make_growth_policy(
                            threshold=threshold,
                            active_days=active_days,
                            cooldown_days=candidate_cooldown,
                            minimum_interval=min(minimum, candidate_cooldown),
                            enabled=enabled,
                            delta=delta,
                            epsilon=epsilon,
                            version="cycle3_validation_candidate",
                        )
                        states = _simulate_runtime(
                            validation_runtime, policy, score_model=score_model, scope=scope
                        )
                        cache[key] = (states, _fast_validation_metrics(states, validation_seasons, contract))
                    return cache[key]

                family_rows: dict[str, list[dict[str, Any]]] = {
                    "P0_selected": [],
                    "P_growth_selected": [],
                    "P_short_selected": [],
                }
                for theta in thresholds:
                    _, p0_metrics = evaluate(theta, cooldown, False, None)
                    p0_row = {
                        **p0_metrics,
                        "candidate_family": "P0_selected",
                        "threshold": theta,
                        "active_days": active_days,
                        "cooldown_days": cooldown,
                        "minimum_repeat_interval_days": minimum,
                        "growth_override_enabled": False,
                        "growth_delta_id": "disabled",
                        "growth_logit_delta": np.nan,
                    }
                    family_rows["P0_selected"].append(p0_row)
                    validation_rows.append(
                        {**p0_row, "fold_id": fold_id, "score_model": score_model, "evaluation_scope": scope}
                    )
                    for option in growth_options:
                        enabled = bool(option["enabled"])
                        delta = option["logit_delta"] if enabled else None
                        _, metrics = evaluate(theta, cooldown, enabled, delta)
                        row = {
                            **metrics,
                            "candidate_family": "P_growth_selected",
                            "threshold": theta,
                            "active_days": active_days,
                            "cooldown_days": cooldown,
                            "minimum_repeat_interval_days": minimum,
                            "growth_override_enabled": enabled,
                            "growth_delta_id": option["id"],
                            "growth_logit_delta": np.nan if delta is None else float(delta),
                        }
                        family_rows["P_growth_selected"].append(row)
                        validation_rows.append(
                            {**row, "fold_id": fold_id, "score_model": score_model, "evaluation_scope": scope}
                        )
                    _, short_metrics = evaluate(theta, minimum, False, None)
                    short_row = {
                        **short_metrics,
                        "candidate_family": "P_short_selected",
                        "threshold": theta,
                        "active_days": active_days,
                        "cooldown_days": minimum,
                        "minimum_repeat_interval_days": minimum,
                        "growth_override_enabled": False,
                        "growth_delta_id": "disabled",
                        "growth_logit_delta": np.nan,
                    }
                    family_rows["P_short_selected"].append(short_row)
                    validation_rows.append(
                        {**short_row, "fold_id": fold_id, "score_model": score_model, "evaluation_scope": scope}
                    )

                selected_by_family = {
                    family: select_policy_record(rows) for family, rows in family_rows.items()
                }
                saved_policy = _make_growth_policy(
                    threshold=float(saved["threshold"]),
                    active_days=int(saved["active_days"]),
                    cooldown_days=int(saved["cooldown_days"]),
                    minimum_interval=minimum,
                    enabled=False,
                    delta=None,
                    epsilon=epsilon,
                    version="cycle3_P0_saved",
                )
                saved_validation_states = _simulate_runtime(
                    validation_runtime, saved_policy, score_model=score_model, scope=scope
                )
                saved_metrics = _fast_validation_metrics(saved_validation_states, validation_seasons, contract)
                saved_record = {
                    **saved_metrics,
                    "candidate_family": "P0_saved",
                    "threshold": saved_policy.threshold,
                    "active_days": saved_policy.active_days,
                    "cooldown_days": saved_policy.cooldown_days,
                    "minimum_repeat_interval_days": minimum,
                    "growth_override_enabled": False,
                    "growth_delta_id": "disabled",
                    "growth_logit_delta": np.nan,
                    "selection_status": "saved_parent_configuration",
                }
                selected_records = {"P0_saved": saved_record, **selected_by_family}
                growth_selected = selected_by_family["P_growth_selected"]
                _, mechanism_metrics = evaluate(
                    float(growth_selected["threshold"]), cooldown, False, None
                )
                mechanism_record = {
                    **mechanism_metrics,
                    "candidate_family": "P0_growth_theta_control",
                }
                mechanism_record.update(
                    {
                        "threshold": float(growth_selected["threshold"]),
                        "active_days": active_days,
                        "cooldown_days": cooldown,
                        "minimum_repeat_interval_days": minimum,
                        "growth_override_enabled": False,
                        "growth_delta_id": "disabled",
                        "growth_logit_delta": np.nan,
                        "selection_status": "diagnostic_same_theta_as_P_growth",
                    }
                )
                selected_records["P0_growth_theta_control"] = mechanism_record

                alpha = validation_runtime["alpha"]
                for family, record in selected_records.items():
                    selection_rows.append(
                        {
                            **record,
                            "fold_id": fold_id,
                            "score_model": score_model,
                            "evaluation_scope": scope,
                            "alpha": np.nan if alpha is None else float(alpha),
                            "threshold_candidates": len(thresholds),
                            "selection_population": "outer_validation_only",
                            "score_configuration": "frozen_before_cycle3_policy_selection",
                        }
                    )

                test_runtime = runtime[(fold_id, score_model, scope, "outer_test")]
                for family, selected_record in selected_records.items():
                    policy = _configuration_from_record(
                        selected_record, contract, f"cycle3_{family}_{scope}"
                    )
                    states = _simulate_runtime(
                        test_runtime, policy, score_model=score_model, scope=scope
                    )
                    public_code = f"{score_model}__{family}"
                    states["score_model"] = score_model
                    states["model_code"] = public_code
                    states["policy_family"] = family
                    states["fold_id"] = fold_id
                    states["alpha"] = np.nan if test_runtime["alpha"] is None else float(test_runtime["alpha"])
                    states["validation_selection_status"] = selected_record["selection_status"]
                    states["validation_feasible"] = bool(selected_record["feasible"])
                    alarm_parts.append(states)
                    metric, hits = _candidate_metrics(
                        states,
                        test_seasons,
                        public_code=public_code,
                        fold_id=fold_id,
                        scope=scope,
                        contract=contract,
                    )
                    metric.update(
                        score_model=score_model,
                        policy_family=family,
                        season=int(test_seasons["season"].iloc[0]),
                        alpha=states["alpha"].iloc[0],
                        threshold=policy.threshold,
                        cooldown_days=policy.cooldown_days,
                        growth_override_enabled=policy.growth_override_enabled,
                        growth_delta_id=selected_record["growth_delta_id"],
                        growth_logit_delta=selected_record.get("growth_logit_delta", np.nan),
                    )
                    event_metric_rows.append(
                        {key: value for key, value in metric.items() if key not in {
                            "field_days", "field_seasons", "messages", "messages_per_30_field_days",
                            "messages_per_field_season", "active_alarm_days", "active_alarm_fraction",
                            "computable_days", "computable_fraction", "abstention_days", "suppressed_repeats",
                            "messages_p95_per_field_season", "messages_max_per_field_season",
                            "growth_messages", "growth_reference_not_comparable_days",
                            "fallback_days", "fallback_day_fraction",
                            "correction_unavailable_days", "correction_unavailable_fraction",
                            "effective_c0_origin_days", "effective_c0_origin_fraction",
                            "external_budget_exceeded"
                        }}
                    )
                    burden_rows.append(
                        {
                            key: value
                            for key, value in metric.items()
                            if key
                            not in {
                                "first_events",
                                "events_with_warning_opportunity",
                                "timely_hits",
                                "timely_recall",
                                "timely_recall_wilson_low",
                                "timely_recall_wilson_high",
                                "coverage_all_first_events",
                                "computable_events",
                                "computable_event_fraction",
                                "median_best_timely_lead_days",
                            }
                        }
                    )
                    for hit in hits:
                        hit.update(
                            score_model=score_model,
                            policy_family=family,
                            alpha=states["alpha"].iloc[0],
                        )
                    event_hit_rows.extend(hits)

                    if family == "P0_saved":
                        if score_model.startswith("C6_"):
                            old = inputs["v4_alarm_states"]
                            old = old[
                                old["fold_id"].eq(fold_id)
                                & old["model_family"].eq(score_model)
                                & old["policy_mode"].eq("validation_selected")
                                & old["evaluation_scope"].eq(scope)
                            ]
                        else:
                            old = inputs["v3_alarm_states"]
                            old = old[
                                old["fold_id"].eq(fold_id)
                                & old["model_code"].eq(score_model)
                                & old["evaluation_scope"].eq(scope)
                            ]
                        identity = _verify_p0_saved_identity(
                            states,
                            old,
                            fold_id=fold_id,
                            score_model=score_model,
                            scope=scope,
                        )
                        identity_rows.append(identity)
                        if identity["status"] != "passed":
                            raise AssertionError(f"P0_saved identity failed: {identity}")

    alarm_states = pd.concat(alarm_parts, ignore_index=True)
    event_metrics_frame = pd.DataFrame(event_metric_rows)
    burden_metrics_frame = pd.DataFrame(burden_rows)
    event_hits_frame = pd.DataFrame(event_hit_rows)
    annual_keys = ["model_code", "fold_id", "evaluation_scope", "slice", "season"]
    burden_only = annual_keys + [
        column
        for column in burden_metrics_frame.columns
        if column not in event_metrics_frame.columns
    ]
    annual_metrics_frame = event_metrics_frame.merge(
        burden_metrics_frame[burden_only],
        on=annual_keys,
        how="outer",
        validate="one_to_one",
    )
    pooled = aggregate_pooled_metrics(
        event_metrics_frame,
        burden_metrics_frame,
        event_hits=event_hits_frame,
        alarm_states=alarm_states,
    )
    pooled["pooled_burden_metrics"] = _enrich_pooled_burden(
        pooled["pooled_burden_metrics"], burden_metrics_frame
    )
    join_keys = [
        "period",
        "year_start",
        "year_end",
        "test_years",
        "model_code",
        "evaluation_scope",
        "slice",
    ]
    pooled["pooled_summary"] = pooled["pooled_event_metrics"].merge(
        pooled["pooled_burden_metrics"], on=join_keys, how="outer", validate="one_to_one"
    )
    return {
        "validation_policy_candidates": pd.DataFrame(validation_rows),
        "policy_selection": pd.DataFrame(selection_rows),
        "alarm_states": alarm_states,
        "event_metrics": event_metrics_frame,
        "event_hits": event_hits_frame,
        "burden_metrics": burden_metrics_frame,
        "annual_metrics": annual_metrics_frame,
        "p0_saved_identity_checks": pd.DataFrame(identity_rows),
        **pooled,
    }


def build_required_audits(inputs: dict[str, Any]) -> dict[str, pd.DataFrame]:
    from .early_warning_cycle3_audits import audit_event_populations, audit_suppression_paths

    population_parts: dict[str, list[pd.DataFrame]] = {
        "population_audit_summary": [],
        "eligible_gained_lost_events": [],
        "all_first_event_diagnostics": [],
    }
    for candidate in (
        "C6_weather__c0_policy_replay",
        "C6_weather__validation_selected",
    ):
        for scope in SCOPES:
            result = audit_event_populations(
                inputs["v4_event_hits"],
                inputs["v3_event_hits"],
                candidate=candidate,
                baseline="C0",
                evaluation_scope=scope,
                event_registry=inputs["v3_events"],
                key_namespace="late_blight_cycle3_population",
            )
            for name, frame in result.items():
                population_parts[name].append(frame)

    suppression_parts: dict[str, list[pd.DataFrame]] = {
        "suppression_audit_summary": [],
        "suppression_audit_days": [],
        "suppression_episodes": [],
    }
    for candidate in (
        "C6_weather__c0_policy_replay",
        "C6_weather__validation_selected",
    ):
        for period, years in (("2020_2025", (2020, 2025)), ("2026_partial", (2026, 2026))):
            result = audit_suppression_paths(
                inputs["v4_alarm_states"],
                inputs["v3_alarm_states"],
                candidate=candidate,
                baseline="C0",
                evaluation_scope="service_calendar",
                years=years,
                key_namespace=f"late_blight_cycle3_suppression_{period}",
            )
            for name, frame in result.items():
                tagged = frame.copy()
                tagged["period"] = period
                suppression_parts[name].append(tagged)
    return {
        name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for name, parts in {**population_parts, **suppression_parts}.items()
    }


def _markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]], limit: int | None = None) -> str:
    selected = frame if limit is None else frame.head(limit)
    headers = [label for _, label in columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in selected.itertuples(index=False):
        source = row._asdict()
        values: list[str] = []
        for column, _ in columns:
            value = source.get(column)
            if value is None or pd.isna(value):
                values.append("—")
            elif isinstance(value, (bool, np.bool_)):
                values.append("да" if value else "нет")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_audit_markdown(run_dir: Path, audits: dict[str, pd.DataFrame]) -> None:
    population = audits["population_audit_summary"]
    population = population[population["aggregation"].eq("pooled")].copy()
    suppression = audits["suppression_audit_summary"]
    main = suppression[suppression["period"].eq("2020_2025")].copy()
    lines = [
        "# Аудит популяций и подавленных дней перед третьим циклом",
        "",
        "## 127 и 87",
        "",
        "Проверка выполнена по реальным `(season, field_season)` и затем обезличена. "
        "В 2020–2025 есть 127 первых регистраций: 87 eligible-событий после подключения "
        "и 40 положительных регистраций, уже известных при входе. В таблице v4 все 40 "
        "непригодных событий были отнесены к `neither_hit`, поэтому сумма категорий равнялась "
        "127. Основные числители, recall, дельты, интервалы и нагрузка рассчитаны по 87 и не изменились. "
        "Corrigendum третьего цикла показывает gained/lost только на eligible-популяции.",
        "",
        _markdown_table(
            population,
            [
                ("candidate", "C6 v4"),
                ("evaluation_scope", "Сценарий"),
                ("all_first_events", "Все первые"),
                ("eligible_events", "Eligible"),
                ("positive_at_entry_events", "При входе"),
                ("both_hit", "Оба"),
                ("gained_by_candidate", "Gained"),
                ("lost_by_candidate", "Lost"),
                ("neither_hit", "Никто"),
                ("all_invariants_pass", "Инварианты"),
            ],
        ),
        "",
        "## Alpha, score и фактические подавления в service",
        "",
        "Для `c0_policy_replay` погодная поправка фактически применялась только в 2022 году. "
        "Все 357 изменённых score разбиваются на 73 дня ниже обоих порогов, 29 дней реальной "
        "отправки обеими системами и 255 дней, когда обе системы были выше порога и обе имели "
        "фактическое подавление cooldown. Разных пороговых решений и разных дат сообщений не было. "
        "Среди 255 дней нет ни одного перехода снизу вверх; это 16 поле-сезонов и 26 последовательных "
        "эпизодов. Постфактум 68 дат пересекают окна 11 eligible-событий, но шесть из этих событий "
        "уже были своевременно покрыты. Эти количества не являются числом полезных упущенных сообщений.",
        "",
        _markdown_table(
            main[main["aggregation"].eq("year")],
            [
                ("candidate", "Режим v4"),
                ("season", "Год"),
                ("alpha", "Alpha"),
                ("matched_days", "Дни"),
                ("weather_correction_applied_days", "Поправка применена"),
                ("score_changed_days", "Score изменён"),
                ("different_threshold_decision_days", "Разное решение порога"),
                ("candidate_upward_crossings_comparable", "Переходы C6"),
                ("baseline_upward_crossings_comparable", "Переходы C0"),
                ("candidate_messages", "Сообщения C6"),
                ("baseline_messages", "Сообщения C0"),
                ("changed_score_suppressed_days", "Изменено и подавлено"),
                ("changed_score_suppressed_episodes", "Эпизоды"),
            ],
        ),
        "",
        "У `validation_selected` в 2022–2025 различались и score, и самостоятельно выбранные пороги, "
        "поэтому расхождения сообщений нельзя приписать только погодной поправке. На границах "
        "weather/fallback в 2023 году обнаружены три сырых перехода вверх; P_growth исключает их "
        "из сопоставимого reference и не сбрасывает cooldown.",
        "",
        "Замороженная погода доступна во всех 87 событийных окнах, но отсутствует на части полного "
        "сервисного календаря, главным образом после конца ERA-ряда. Это структурное ограничение снимка, "
        "а не случайный оперативный отказ.",
    ]
    (run_dir / "audit_population_and_suppression.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _gained_lost_annual(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["score_model", "evaluation_scope", "candidate_policy", "baseline_policy"]
    for group_key, group in events.groupby(keys, sort=True):
        for aggregation, season, selected in [
            ("pooled", pd.NA, group),
            *[("year", int(year), part) for year, part in group.groupby("season", sort=True)],
        ]:
            counts = selected["transition"].value_counts()
            both = int(counts.get("both_hit", 0))
            gained = int(counts.get("gained", 0))
            lost = int(counts.get("lost", 0))
            neither = int(counts.get("neither_hit", 0))
            total = int(len(selected))
            if both + gained + lost + neither != total:
                raise AssertionError("Annual gained/lost partition failed")
            rows.append(
                {
                    **dict(zip(keys, group_key)),
                    "aggregation": aggregation,
                    "season": season,
                    "eligible_events": total,
                    "both": both,
                    "gained": gained,
                    "lost": lost,
                    "neither": neither,
                    "candidate_timely_events": both + gained,
                    "baseline_timely_events": both + lost,
                    "net_gain": gained - lost,
                    "all_invariants_pass": True,
                }
            )
    return pd.DataFrame(rows)


def _policy_comparisons() -> list[tuple[str, str]]:
    comparisons: list[tuple[str, str]] = []
    for model in SCORE_MODELS:
        comparisons.extend(
            [
                (f"{model}__P0_selected", f"{model}__P0_saved"),
                (f"{model}__P_growth_selected", f"{model}__P0_saved"),
                (f"{model}__P_growth_selected", f"{model}__P0_selected"),
                (f"{model}__P_short_selected", f"{model}__P0_selected"),
                (f"{model}__P_growth_selected", f"{model}__P_short_selected"),
                (f"{model}__P_growth_selected", f"{model}__P0_growth_theta_control"),
            ]
        )
    for policy in ("P_growth_selected", "P_short_selected", "P0_selected"):
        weather = f"C6_weather__{policy}"
        for comparator in (
            "calendar_window",
            "C0",
            "C1",
            "C4",
            "C5",
            "C6_calibration_control",
            "C6_calendar_control",
        ):
            comparisons.append((weather, f"{comparator}__{policy}"))
    return list(dict.fromkeys(comparisons))


def _field_season_load(states: pd.DataFrame) -> pd.DataFrame:
    selected = states[states["evaluation_scope_day"].astype(bool)].copy()
    selected["growth_message"] = selected["growth_override_used"].astype(bool)
    return (
        selected.groupby(
            [
                "model_code",
                "score_model",
                "policy_family",
                "evaluation_scope",
                "fold_id",
                "season",
                "field_season",
            ],
            sort=True,
        )
        .agg(
            field_days=("issue_date", "size"),
            messages=("message_issued", "sum"),
            growth_messages=("growth_message", "sum"),
            alarm_days=("alarm_active", "sum"),
            suppressed_days=("suppressed_repeat", "sum"),
            incomparable_growth_candidates=(
                "action_reason",
                lambda values: int(pd.Series(values).eq("growth_reference_not_comparable").sum()),
            ),
        )
        .reset_index()
    )


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _main_row(summary: pd.DataFrame, model: str, policy: str, scope: str) -> pd.Series:
    rows = summary[
        summary["period"].eq("2020_2025")
        & summary["slice"].eq("A_plus_B")
        & summary["model_code"].eq(f"{model}__{policy}")
        & summary["evaluation_scope"].eq(scope)
    ]
    if len(rows) != 1:
        raise AssertionError(f"Expected one pooled row for {model}/{policy}/{scope}")
    return rows.iloc[0]


def write_report_ru(
    run_dir: Path,
    outputs: dict[str, pd.DataFrame],
    audits: dict[str, pd.DataFrame],
    fixed_references: pd.DataFrame,
    strong_effect_contract: dict[str, Any],
) -> None:
    from .early_warning_reporting import strong_effect_decision

    summary = outputs["pooled_summary"]
    selection = outputs["policy_selection"]
    gained = outputs["gained_lost_summary"]
    bootstrap = outputs["paired_year_bootstrap"]

    def period_row(period: str, model: str, policy: str, scope: str) -> pd.Series:
        rows = summary[
            summary["period"].eq(period)
            & summary["slice"].eq("A_plus_B")
            & summary["model_code"].eq(f"{model}__{policy}")
            & summary["evaluation_scope"].eq(scope)
        ]
        if len(rows) != 1:
            raise AssertionError(
                f"Expected one pooled row for {period}/{model}/{policy}/{scope}"
            )
        return rows.iloc[0]

    def result_table_row(
        period: str, model: str, policy: str, scope: str
    ) -> dict[str, Any]:
        row = period_row(period, model, policy, scope)
        return {
            "scenario": scope,
            "model": model,
            "policy": policy,
            "timely": f"{int(row['timely_hits'])}/{int(row['events_with_warning_opportunity'])}",
            "messages": float(row["messages_per_30_field_days"]),
            "alarm_pct": 100 * float(row["active_alarm_fraction"]),
            "computable_pct": 100 * float(row["computable_fraction"]),
            "fallback_pct": 100 * float(row.get("fallback_day_fraction", np.nan)),
            "correction_unavailable_pct": 100
            * float(row.get("correction_unavailable_fraction", np.nan)),
            "effective_c0_pct": 100
            * float(row.get("effective_c0_origin_fraction", np.nan)),
            "growth_messages": int(row.get("growth_messages", 0)),
            "budget_years": int(row.get("external_years_over_budget", 0)),
        }

    weather = _main_row(summary, "C6_weather", "P_growth_selected", "service_calendar")
    weather_old = _main_row(summary, "C6_weather", "P0_saved", "service_calendar")
    weather_short = _main_row(summary, "C6_weather", "P_short_selected", "service_calendar")
    weather_matched = _main_row(
        summary, "C6_weather", "P0_growth_theta_control", "service_calendar"
    )
    calendar_models = ("calendar_window", "C0", "C1", "C6_calibration_control", "C6_calendar_control")
    calendar_rows = [
        _main_row(summary, model, "P_growth_selected", "service_calendar")
        for model in calendar_models
    ]
    control_text = ", ".join(
        f"{row['model_code'].split('__')[0]} {int(row['timely_hits'])}/87, "
        f"{_fmt(row['messages_per_30_field_days'])} сообщ./30, "
        f"{_fmt(100 * row['active_alarm_fraction'], 1)}% тревоги"
        for row in calendar_rows
    )
    own_gain = int(weather["timely_hits"] - weather_old["timely_hits"])
    mechanism_gain = int(weather["timely_hits"] - weather_matched["timely_hits"])
    short_delta = int(weather["timely_hits"] - weather_short["timely_hits"])
    delta_recall = float(weather["timely_recall"] - weather_old["timely_recall"])
    delta_alarm = float(
        weather["active_alarm_fraction"] - weather_old["active_alarm_fraction"]
    )
    relative_message_reduction = (
        1.0
        - float(weather["messages_per_30_field_days"])
        / float(weather_old["messages_per_30_field_days"])
        if float(weather_old["messages_per_30_field_days"]) > 0
        else np.nan
    )
    strong_effect = strong_effect_decision(
        {
            "delta_timely_recall": delta_recall,
            "delta_messages_per_30_field_days": float(
                weather["messages_per_30_field_days"]
                - weather_old["messages_per_30_field_days"]
            ),
            "relative_message_reduction": relative_message_reduction,
            "delta_active_alarm_fraction": delta_alarm,
            "candidate_messages_per_30_field_days": float(
                weather["messages_per_30_field_days"]
            ),
            "candidate_active_alarm_fraction": float(
                weather["active_alarm_fraction"]
            ),
        },
        strong_effect_contract,
    )
    recall_gain_branch = strong_effect["recall_gain_branch"]
    efficiency_branch = strong_effect["efficiency_branch"]
    strong_policy_effect = strong_effect["strong_effect"]
    strong_effect_text = (
        "Ветка прироста покрытия (+0.15 при не большей нагрузке): "
        f"{'да' if recall_gain_branch else 'нет'}; ветка эффективности "
        "(-30% сообщений, потеря recall не более 0.05 и без роста тревоги): "
        f"{'да' if efficiency_branch else 'нет'}. "
        + (
            "Заранее заданный сильный эффект подтверждён."
            if strong_policy_effect
            else "Заранее заданный сильный эффект не подтверждён."
        )
    )
    next_priority = (
        "Начать проспективный теневой журнал реально доступных на дату выпуска прогнозов погоды, "
        "визитов и фактических причин пропуска. Это проверяет новую информационную гипотезу и качество "
        "наблюдений, вместо нового перебора политики на уже просмотренных годах."
    )

    main_rows: list[dict[str, Any]] = []
    partial_2026_rows: list[dict[str, Any]] = []
    for scope in ("service_calendar", "paired_candidate_days"):
        for model in SCORE_MODELS:
            for policy in ("P0_saved", "P0_selected", "P_growth_selected", "P_short_selected"):
                main_rows.append(result_table_row("2020_2025", model, policy, scope))
                partial_2026_rows.append(
                    result_table_row("2026_partial", model, policy, scope)
                )
    main_table = pd.DataFrame(main_rows)
    partial_2026_table = pd.DataFrame(partial_2026_rows)
    weather_control_table = pd.DataFrame(
        [
            result_table_row(
                "2020_2025", model, "P_growth_selected", "service_calendar"
            )
            for model in ("C6_weather", *calendar_models)
        ]
    )
    fixed_reference_table = fixed_references[
        fixed_references["period"].eq("2020_2025")
        & fixed_references["slice"].eq("A_plus_B")
        & fixed_references["model_code"].isin(
            ["calendar_window", "periodic_30d", "hutton", "smith", "polyakov"]
        )
    ].copy()
    fixed_reference_table["timely"] = (
        fixed_reference_table["timely_hits"].astype(int).astype(str)
        + "/"
        + fixed_reference_table["events_with_warning_opportunity"].astype(int).astype(str)
    )
    fixed_reference_table["alarm_pct"] = (
        100 * fixed_reference_table["active_alarm_fraction"]
    )
    fixed_reference_table["computable_pct"] = (
        100 * fixed_reference_table["computable_fraction"]
    )
    selected_growth = selection[
        selection["candidate_family"].eq("P_growth_selected")
        & ~selection["fold_id"].eq("test_2026_partial")
    ]
    selection_summary = (
        selected_growth.groupby(["score_model", "evaluation_scope", "growth_delta_id"], sort=True)
        .size()
        .rename("folds")
        .reset_index()
    )
    key_comparisons = bootstrap[
        bootstrap["period"].eq("2020_2025")
        & bootstrap["slice"].eq("A_plus_B")
        & bootstrap["evaluation_scope"].eq("service_calendar")
        & bootstrap["candidate"].eq("C6_weather__P_growth_selected")
        & bootstrap["baseline"].isin(
            [
                "C6_weather__P0_saved",
                "C6_weather__P_short_selected",
                "C6_weather__P0_growth_theta_control",
                "calendar_window__P_growth_selected",
                "C1__P_growth_selected",
                "C6_calibration_control__P_growth_selected",
            ]
        )
    ].copy()
    mechanism = gained[
        gained["score_model"].eq("C6_weather")
        & gained["evaluation_scope"].eq("service_calendar")
        & gained["candidate_policy"].eq("P_growth_selected")
        & gained["baseline_policy"].isin(
            ["P0_saved", "P0_selected", "P_short_selected", "P0_growth_theta_control"]
        )
        & gained["aggregation"].eq("pooled")
    ]

    lines = [
        "# Третий исследовательский цикл ранних предупреждений о фитофторозе картофеля",
        "",
        f"Запуск: `{run_dir.name}`.",
        "",
        "## Четыре ответа",
        "",
        "1. **127 против 87 и подавления.** Расхождение 127/87 было особенностью представления: "
        "gained/lost v4 включал 40 первых положительных регистраций, уже известных при подключении, "
        "и относил их к `neither`. Основные 45/87, другие числители, дельты и нагрузка не изменились. "
        "Из 357 изменённых score причинно чистого service replay все относятся к 2022: 73 ниже порога, "
        "29 совпали с сообщением обеих систем и 255 были фактически подавленными днями выше порога. "
        "Эти 255 образуют 26 эпизодов в 16 поле-сезонах и не содержат переходов снизу вверх.",
        "",
        f"2. **P_growth против старой политики и P_short.** Для C6_weather в полном сервисе P_growth "
        f"получила {int(weather['timely_hits'])}/87 против {int(weather_old['timely_hits'])}/87 у P0_saved "
        f"(дельта {own_gain:+d} событий) и {int(weather_short['timely_hits'])}/87 у P_short "
        f"(дельта {short_delta:+d}). Нагрузка: {_fmt(weather['messages_per_30_field_days'])} сообщения/30 "
        f"и {_fmt(100*weather['active_alarm_fraction'],1)}% тревожных дней; у старой политики "
        f"{_fmt(weather_old['messages_per_30_field_days'])} и {_fmt(100*weather_old['active_alarm_fraction'],1)}%, "
        f"у P_short {_fmt(weather_short['messages_per_30_field_days'])} и "
        f"{_fmt(100*weather_short['active_alarm_fraction'],1)}%. {strong_effect_text}",
        "",
        f"3. **Нужна ли погода.** С тем же семейством P_growth погодная C6 дала "
        f"{int(weather['timely_hits'])}/87. Все заданные календарные и калибровочные контроли: "
        f"{control_text}. Чистый эффект override при том же theta равен "
        f"{mechanism_gain:+d} событию(ям); смены fallback не считались ростом. Поэтому выигрыш политики "
        "сам по себе не приписывается погодной информации.",
        "",
        f"4. **Следующий шаг.** {next_priority}",
        "",
        "## Основные результаты 2020–2025",
        "",
        _markdown_table(
            main_table,
            [
                ("scenario", "Сценарий"),
                ("model", "Score"),
                ("policy", "Политика"),
                ("timely", "Timely"),
                ("messages", "Сообщ./30"),
                ("alarm_pct", "Тревожные дни, %"),
                ("computable_pct", "Вычислимость, %"),
                ("fallback_pct", "Fallback, %"),
                ("growth_messages", "Growth-сообщения"),
                ("budget_years", "Лет выше бюджета"),
            ],
        ),
        "",
        "В `service_calendar` C4/C5 сохраняют прежнее воздержание при отсутствии ERA, а C6 продолжает "
        "одну историю с C0 fallback. В `paired_candidate_days` используется прежняя общая маска без "
        "сжатия времени. `Fallback, %` теперь означает только фактический возврат из активной C6-поправки "
        "в C0; при alpha=0 он равен нулю. Сырая недоступность поправки и доля эффективного C0 показаны "
        "отдельно в таблице погодной специфичности. Внешнее превышение бюджета в отдельном году "
        "не исправлялось по его результату.",
        "",
        "## Погодная специфичность при одинаковой свободе политики",
        "",
        _markdown_table(
            weather_control_table,
            [
                ("model", "Score"),
                ("timely", "Timely"),
                ("messages", "Сообщ./30"),
                ("alarm_pct", "Тревожные дни, %"),
                ("computable_pct", "Вычислимость, %"),
                ("fallback_pct", "Fallback, %"),
                ("correction_unavailable_pct", "Нет поправки, %"),
                ("effective_c0_pct", "Эффективный C0, %"),
            ],
        ),
        "",
        "Все строки используют `P_growth_selected`, полный сервис и отдельный внутренний выбор для того "
        "же семейства политики. Поэтому рядом с каждым результатом покрытия приведены фактические сообщения "
        "и тревожные дни; равенство предельного бюджета не трактуется как равенство реальной нагрузки. "
        "`Нет поправки` — сырой показатель доступности погоды независимо от alpha; `Fallback` — только "
        "реальное использование C0 вместо активной положительной-alpha поправки; `Эффективный C0` включает "
        "как такие fallback-дни, так и все дни folds с alpha=0.",
        "",
        "## Что выбрала внутренняя validation",
        "",
        _markdown_table(
            selection_summary,
            [
                ("score_model", "Score"),
                ("evaluation_scope", "Сценарий"),
                ("growth_delta_id", "Выбранный delta"),
                ("folds", "Folds 2020–2025"),
            ],
        ),
        "",
        "Alpha для C6 не выбиралась заново: для каждого fold и сценария использована ровно конфигурация "
        "`validation_selected` из v4. Порог и политика выбирались только на прежнем outer-validation. "
        "Вариант `disabled` входил в P_growth и выигрывал ничью как более простой.",
        "",
        "## Последовательные gained/lost",
        "",
        _markdown_table(
            mechanism,
            [
                ("baseline_policy", "База"),
                ("eligible_events", "События"),
                ("both", "Оба"),
                ("gained", "Gained"),
                ("lost", "Lost"),
                ("neither", "Никто"),
                ("net_gain", "Net"),
                ("all_invariants_pass", "Инварианты"),
            ],
        ),
        "",
        "Каждый вариант переигран последовательно с начала календаря. Growth-сообщение обновляет reference "
        "и сдвигает следующий обычный refresh, поэтому таблица включает как приобретённые, так и потерянные "
        "события. Детальные причины сохранены отдельно; это не статическая подстановка сообщений в окна.",
        "",
        "## Неопределённость",
        "",
        _markdown_table(
            key_comparisons,
            [
                ("baseline", "База"),
                ("candidate_hits", "C6 growth hits"),
                ("baseline_hits", "Base hits"),
                ("delta_timely_recall", "Δ recall"),
                ("delta_timely_recall_low", "95% low"),
                ("delta_timely_recall_high", "95% high"),
                ("delta_messages_per_30_field_days", "Δ сообщ./30"),
                ("delta_active_alarm_fraction", "Δ тревоги"),
            ],
        ),
        "",
        "Bootstrap пересэмплирует шесть уже изученных внешних лет. Интервалы описательны: они не включают "
        "неопределённость выбора моделей, перекрывающихся обучающих периодов, даты биологического начала или "
        "процесса регистрации. Анализ исключения одного года сохранён в отдельной таблице.",
        "",
        "## Неполный 2026 год: описательная проверка",
        "",
        _markdown_table(
            partial_2026_table,
            [
                ("scenario", "Сценарий"),
                ("model", "Score"),
                ("policy", "Политика"),
                ("timely", "Timely"),
                ("messages", "Сообщ./30"),
                ("alarm_pct", "Тревожные дни, %"),
                ("computable_pct", "Вычислимость, %"),
                ("fallback_pct", "Fallback, %"),
                ("growth_messages", "Growth-сообщения"),
                ("budget_years", "Лет выше бюджета"),
            ],
        ),
        "",
        "2026 год неполный и не входил в основной агрегат 2020–2025, bootstrap или выбор политики. "
        "Эта таблица служит только описательной проверкой заранее зафиксированного replay.",
        "",
        "## Фиксированные референсы и вычислимость",
        "",
        _markdown_table(
            fixed_reference_table,
            [
                ("evaluation_scope", "Сценарий"),
                ("model_code", "Фиксированный референс v3"),
                ("timely", "Timely"),
                ("messages_per_30_field_days", "Сообщ./30"),
                ("alarm_pct", "Тревожные дни, %"),
                ("computable_pct", "Вычислимость, %"),
            ],
        ),
        "",
        "Calendar window получил P0, P_growth и P_short на тех же правилах выбора, что и погодные score. "
        "В таблице приведён фиксированный periodic_30d. Все фазы periodic_17d из v4 перенесены отдельными "
        "артефактами без выбора лучшей внешней фазы; Хаттон, Smith и Поляков не перенастраивались. "
        "Погода присутствует во всех 87 событийных окнах, но C4/C5 не вычислимы на всём сервисном календаре. "
        "Переход C0 fallback ↔ C6 weather разрывает сопоставимость growth-reference и не сбрасывает cooldown.",
        "",
        "## Ограничения",
        "",
        "- Результат относится к первой регистрации, а не к дате заражения или первых симптомов.",
        "- 2020–2025 уже многократно изучались; это не независимое подтверждение новой политики.",
        "- Ретроспективная погода не доказывает оперативную доступность прогноза на дату выпуска.",
        "- Нет таргет-специфических отрицательных осмотров; PPV и биологическая specificity не оценивались.",
        "- Численные delta отражают изменение odds модельного score, а не доказанный рост биологического риска.",
        "",
        "## Один следующий приоритет",
        "",
        next_priority,
        "",
        "## Воспроизведение",
        "",
        "См. `REPRODUCE.md`. Команда создаёт новый уникальный каталог и повторно проверяет оба родительских запуска.",
    ]
    (run_dir / "report_ru.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cycle(
    contract_path: Path,
    run_id: str,
    v3_dir: Path = DEFAULT_V3,
    v4_dir: Path = DEFAULT_V4,
) -> Path:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    output_root = Path(contract["output_root"])
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    run_dir = output_root / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty and will not be overwritten: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    expected_v3 = contract["parents"]["cycle1"]["execution_manifest_sha256"]
    expected_v4 = contract["parents"]["cycle2"]["execution_manifest_sha256"]
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "cycle": 3,
        "mode": "frozen_score_policy_only",
        "status": "running",
        "started_at_utc": _utc_now(),
        "command": [sys.executable, *sys.argv],
        "git": {
            "revision": _git_text("rev-parse", "HEAD"),
            "branch": _git_text("branch", "--show-current"),
            "status_short": _git_text("status", "--short").splitlines(),
        },
        "environment": _environment_versions(),
        "random_seed": int(contract["random_seed"]),
        "parent_runs": {"cycle1": str(v3_dir), "cycle2": str(v4_dir)},
        "scientific_status": "retrospective_registration_proxy_only_external_years_already_studied",
    }
    _write_json(run_dir / "execution_manifest.json", manifest)
    try:
        source_snapshot = _write_source_snapshot(run_dir / "source_snapshot.json")
        manifest["source_snapshot"] = source_snapshot
        _write_json(run_dir / "execution_manifest.json", manifest)

        parent_before = {
            "cycle1": verify_frozen_run(v3_dir, expected_v3),
            "cycle2": verify_frozen_run(v4_dir, expected_v4),
        }
        _write_json(run_dir / "parent_integrity_before.json", parent_before)
        shutil.copy2(contract_path, run_dir / "evaluation_contract.json")
        protocol_path = contract_path.with_name("cycle3_protocol.md")
        if not protocol_path.is_file():
            raise FileNotFoundError(protocol_path)
        shutil.copy2(protocol_path, run_dir / "protocol.md")

        inputs = _load_inputs(v3_dir, v4_dir)
        audits = build_required_audits(inputs)
        _write_frame(run_dir, "population_audit_summary", audits["population_audit_summary"])
        _write_frame(
            run_dir,
            "v4_eligible_gained_lost_events",
            audits["eligible_gained_lost_events"],
            parquet=True,
        )
        _write_frame(
            run_dir,
            "all_first_registration_diagnostics",
            audits["all_first_event_diagnostics"],
            parquet=True,
        )
        _write_frame(run_dir, "suppression_audit_summary", audits["suppression_audit_summary"])
        _write_frame(
            run_dir, "suppression_audit_days", audits["suppression_audit_days"], parquet=True
        )
        _write_frame(
            run_dir, "suppression_episodes", audits["suppression_episodes"], parquet=True
        )
        write_audit_markdown(run_dir, audits)

        frozen_scores, score_hashes, score_checks, runtime, saved_policies = build_frozen_scores(
            inputs, v3_dir
        )
        if not score_checks["status"].eq("passed").all():
            raise AssertionError("Not every frozen outer-test score was reproduced")
        _write_frame(run_dir, "frozen_scores", frozen_scores, parquet=True)
        _write_frame(run_dir, "frozen_score_hashes", score_hashes)
        _write_frame(run_dir, "score_reproduction_checks", score_checks)

        outputs = run_policy_experiments(inputs, runtime, saved_policies, contract)
        if not outputs["p0_saved_identity_checks"]["status"].eq("passed").all():
            raise AssertionError("P0_saved did not exactly reproduce a parent policy")

        gained_events, _, population = build_gained_lost(
            outputs["event_hits"], outputs["alarm_states"], inputs["v3_field_seasons"]
        )
        gained_summary = _gained_lost_annual(gained_events)
        outputs["gained_lost_events"] = gained_events
        outputs["gained_lost_summary"] = gained_summary
        outputs["cycle3_event_population"] = population

        comparisons = _policy_comparisons()
        outputs["paired_year_bootstrap"] = paired_year_bootstrap(
            outputs["event_hits"],
            outputs["alarm_states"],
            seed=int(contract["uncertainty"]["seed"]),
            n_bootstrap=int(contract["uncertainty"]["paired_year_bootstrap_repeats"]),
            comparisons=comparisons,
            slices=("A_plus_B",),
        )
        outputs["leave_one_year_out"] = build_leave_one_year_out(
            outputs["event_hits"], outputs["alarm_states"], comparisons
        )
        outputs["field_season_load"] = _field_season_load(outputs["alarm_states"])
        notification_columns = [
            "field_season",
            "season",
            "issue_date",
            "issued_at",
            "fold_id",
            "score_model",
            "model_code",
            "policy_family",
            "alpha",
            "evaluation_scope",
            "score",
            "score_status",
            "score_origin",
            "score_model_id",
            "score_model_version",
            "score_comparison_segment_id",
            "score_comparison_segment_started",
            "correction_available",
            "fallback_to_c0",
            "effective_c0_origin",
            "policy_threshold",
            "policy_active_days",
            "policy_cooldown_days",
            "policy_minimum_repeat_interval_days",
            "policy_growth_override_enabled",
            "policy_growth_logit_delta",
            "previous_message_date",
            "previous_message_score",
            "previous_message_origin",
            "previous_message_model_id",
            "previous_message_model_version",
            "previous_message_segment_id",
            "elapsed_calendar_days_since_message",
            "logit_growth_from_previous_message",
            "odds_multiplier_from_previous_message",
            "growth_reference_comparable",
            "growth_reference_status",
            "threshold_reached",
            "ordinary_cooldown_satisfied",
            "growth_minimum_interval_satisfied",
            "message_issued",
            "message_kind",
            "growth_override_used",
            "suppressed_repeat",
            "action_reason",
            "alarm_active",
            "active_from",
            "active_through",
            "cumulative_messages_field_season",
            "cumulative_alarm_days_field_season",
        ]
        notification = outputs["alarm_states"]
        notification = notification[
            notification["message_issued"].astype(bool)
            | notification["suppressed_repeat"].astype(bool)
        ].copy()
        outputs["notification_log"] = notification[
            [column for column in notification_columns if column in notification]
        ]
        prediction_columns = [
            "field_season",
            "season",
            "issue_date",
            "issued_at",
            "fold_id",
            "score_model",
            "model_code",
            "policy_family",
            "alpha",
            "evaluation_scope",
            "evaluation_scope_day",
            "score",
            "score_status",
            "score_origin",
            "score_model_id",
            "score_model_version",
            "correction_available",
            "fallback_to_c0",
            "effective_c0_origin",
        ]
        outputs["predictions"] = outputs["alarm_states"][prediction_columns].copy()

        v3_pooled = pd.read_csv(v3_dir / "pooled_summary.csv")
        fixed_references = v3_pooled[
            v3_pooled["model_code"].isin(
                ["calendar_window", "periodic_30d", "hutton", "smith", "polyakov"]
            )
        ].copy()
        _write_frame(run_dir, "fixed_reference_metrics_v3", fixed_references)
        for source_name, target_name in (
            ("v3_periodic_30d.csv", "periodic_30d_v4.csv"),
            ("v3_periodic_17d_all_phases.csv", "periodic_17d_all_phases_v4.csv"),
            ("v3_periodic_17d_phase_summary.csv", "periodic_17d_phase_summary_v4.csv"),
        ):
            shutil.copy2(v4_dir / source_name, run_dir / target_name)

        parquet_outputs = {
            "alarm_states",
            "predictions",
            "notification_log",
            "event_hits",
            "gained_lost_events",
            "cycle3_event_population",
            "field_season_load",
        }
        for name, frame in outputs.items():
            _write_frame(run_dir, name, frame, parquet=name in parquet_outputs)

        input_audit = {
            "status": "passed",
            "new_model_training": False,
            "new_feature_engineering": False,
            "new_alpha_selection": False,
            "parents": {
                "cycle1_manifest": {
                    "path": str(v3_dir / "execution_manifest.json"),
                    "sha256": sha256_file(v3_dir / "execution_manifest.json"),
                    "outputs_checked": parent_before["cycle1"]["outputs_checked"],
                },
                "cycle2_manifest": {
                    "path": str(v4_dir / "execution_manifest.json"),
                    "sha256": sha256_file(v4_dir / "execution_manifest.json"),
                    "outputs_checked": parent_before["cycle2"]["outputs_checked"],
                },
            },
            "direct_frozen_score_sources": {
                "cycle1_daily_decisions": sha256_file(v3_dir / "daily_decisions.parquet"),
                "cycle1_predictions": sha256_file(v3_dir / "predictions.parquet"),
                "cycle1_policy_selection": sha256_file(v3_dir / "policy_selection.csv"),
                "cycle2_raw_predictions": sha256_file(v4_dir / "c6_raw_predictions.parquet"),
                "cycle2_policy_selection": sha256_file(v4_dir / "policy_selection.csv"),
            },
            "score_reproduction_checks": int(len(score_checks)),
            "all_score_reproduction_checks_pass": bool(score_checks["status"].eq("passed").all()),
        }
        _write_json(run_dir / "input_audit.json", input_audit)

        write_report_ru(
            run_dir,
            outputs,
            audits,
            fixed_references,
            inputs["v3_contract"],
        )
        (run_dir / "REPRODUCE.md").write_text(
            "# Воспроизведение третьего цикла\n\n"
            "```bash\n"
            ".venv/bin/python -m agro_phenology.early_warning_cycle3_pipeline run \\\n"
            "  --contract docs/research/late_blight_early_warning/cycle3_evaluation_contract.json \\\n"
            "  --v3-run results/late_blight_early_warning/20260910_first_cycle_v3 \\\n"
            "  --v4-run results/late_blight_early_warning/20260910_second_cycle_v4 \\\n"
            "  --run-id <new_unique_cycle3_run_id>\n"
            "```\n\n"
            "Команда не обучает модели, не выбирает alpha и не изменяет родительские каталоги. "
            "Каталог нового запуска должен отсутствовать или быть пустым.\n",
            encoding="utf-8",
        )

        tests = _run_test_commands()
        _write_json(run_dir / "test_results.json", tests)
        if tests["status"] != "passed":
            raise RuntimeError("Cycle3 or repository tests failed")

        parent_after = {
            "cycle1": verify_frozen_run(v3_dir, expected_v3),
            "cycle2": verify_frozen_run(v4_dir, expected_v4),
        }
        _write_json(run_dir / "parent_integrity_after.json", parent_after)
        source_after = _verify_source_snapshot(source_snapshot)
        _write_json(run_dir / "source_integrity_after.json", source_after)
        manifest.update(
            {
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "source_snapshot": source_snapshot,
                "source_integrity_after": {
                    "status": source_after["status"],
                    "files_checked": source_after["files_checked"],
                    "artifact": "source_integrity_after.json",
                },
                "parent_integrity_before": "passed",
                "parent_integrity_after": "passed",
                "parent_outputs_checked": {
                    "cycle1": parent_after["cycle1"]["outputs_checked"],
                    "cycle2": parent_after["cycle2"]["outputs_checked"],
                },
                "fold_definitions": inputs["v3_contract"]["rolling_origin_folds"],
                "frozen_score_models": list(SCORE_MODELS),
                "policy_configuration": contract["notification_policy"],
                "tests_status": tests["status"],
                "output_hashes": _output_hashes(run_dir),
            }
        )
        _write_json(run_dir / "execution_manifest.json", manifest)
        return run_dir
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "output_hashes": _output_hashes(run_dir),
            }
        )
        _write_json(run_dir / "execution_manifest.json", manifest)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a new immutable third-cycle experiment")
    run.add_argument("--contract", required=True, type=Path)
    run.add_argument("--run-id", required=True)
    run.add_argument("--v3-run", type=Path, default=DEFAULT_V3)
    run.add_argument("--v4-run", type=Path, default=DEFAULT_V4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    path = run_cycle(
        args.contract.resolve(), args.run_id, args.v3_run.resolve(), args.v4_run.resolve()
    )
    print(json.dumps({"status": "complete", "run_dir": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
