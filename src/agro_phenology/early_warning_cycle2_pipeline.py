"""Second potato late-blight early-warning cycle: nested C6 corrections and diagnostics."""
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

from .early_warning_core import CALENDAR_FEATURES, EPISODE_FEATURES, sha256_file
from .early_warning_cycle2_models import (
    ALPHA_GRID,
    CLASS_ORDER,
    C6Bundle,
    aligned_probabilities,
    c0_raw_logits,
    catboost_correction_logits,
    fit_calendar_catboost_control,
    fit_oof_calibration_control,
    fit_weather_catboost_control,
    load_c6_bundle,
    predict_calibration_control,
    predict_c6_from_baseline,
    save_c6_bundle,
)
from .early_warning_cycle2_diagnostics import build_v3_cycle2_diagnostics
from .early_warning_models import (
    Policy,
    _model_rows,
    _year_mask,
    burden_metrics,
    event_metrics,
    select_threshold,
    simulate_policy,
)
from .early_warning_reporting import aggregate_pooled_metrics, paired_year_bootstrap


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V3 = REPO_ROOT / "results/late_blight_early_warning/20260910_first_cycle_v3"
MODEL_CODES = ("C6_weather", "C6_calibration_control", "C6_calendar_control")
POLICY_MODES = ("c0_policy_replay", "validation_selected")
SCOPES = {
    "service_calendar": None,
    "paired_candidate_days": "candidate_comparison_complete",
}


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
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
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


def verify_v3(v3_dir: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    """Verify every frozen output named by the completed v3 manifest."""
    manifest_path = v3_dir / "execution_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != expected_manifest_sha256:
        raise AssertionError(
            f"Unexpected v3 manifest hash: {manifest_hash}; expected {expected_manifest_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("mode") != "full":
        raise AssertionError("Frozen parent is not the completed full v3 run")
    checked: list[dict[str, Any]] = []
    failures: list[str] = []
    for relative, metadata in sorted(manifest.get("output_hashes", {}).items()):
        path = v3_dir / relative
        actual = sha256_file(path) if path.is_file() else None
        expected = metadata["sha256"]
        ok = actual == expected
        checked.append(
            {
                "relative_path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "bytes": path.stat().st_size if path.is_file() else None,
                "matches": ok,
            }
        )
        if not ok:
            failures.append(relative)
    if failures:
        raise AssertionError(f"Frozen v3 output hash failures: {failures}")
    return {
        "status": "passed",
        "v3_path": str(v3_dir),
        "execution_manifest_sha256": manifest_hash,
        "outputs_checked": len(checked),
        "all_output_hashes_match": True,
        "files": checked,
        "input_hashes_recorded_by_v3": manifest.get("input_hashes", {}),
        "source_snapshot": manifest.get("source_snapshot", {}),
    }


def _write_source_snapshot(path: Path) -> dict[str, Any]:
    candidates = [
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "requirements-early-warning.txt",
        REPO_ROOT / "docs/research/late_blight_early_warning/protocol.md",
        REPO_ROOT / "docs/research/late_blight_early_warning/evaluation_contract.json",
        REPO_ROOT
        / "docs/research/late_blight_early_warning"
        / "AGROPHENOLOGY_POTATO_LATE_BLIGHT_FULL_REPORT_2026-09-15.md",
        REPO_ROOT / "docs/research/late_blight_early_warning/cycle2_protocol.md",
        REPO_ROOT / "docs/research/late_blight_early_warning/cycle2_evaluation_contract.json",
        REPO_ROOT / "src/agro_phenology/early_warning_core.py",
        REPO_ROOT / "src/agro_phenology/early_warning_models.py",
        REPO_ROOT / "src/agro_phenology/early_warning_reporting.py",
    ]
    candidates += sorted((REPO_ROOT / "src/agro_phenology").glob("early_warning_cycle2*.py"))
    candidates += sorted((REPO_ROOT / "tests").glob("test_early_warning_cycle2*.py"))
    missing = [str(item.relative_to(REPO_ROOT)) for item in candidates if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Cycle2 source snapshot is incomplete: {missing}")
    files = [
        {
            "relative_path": str(item.relative_to(REPO_ROOT)),
            "sha256": sha256_file(item),
            "utf8_content": item.read_text(encoding="utf-8"),
        }
        for item in candidates
    ]
    payload = {
        "format": "agro_phenology_cycle2_source_snapshot_v2",
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
    """Prove that the source captured before fitting did not change during the run."""
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
        raise AssertionError(f"Cycle2 source changed during execution: {failures}")
    return {
        "status": "passed",
        "files_checked": len(checks),
        "all_source_hashes_match": True,
        "checks": checks,
    }


def _output_hashes(run_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        relative = str(path.relative_to(run_dir))
        if relative == "execution_manifest.json":
            continue
        result[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def _load_v3(v3_dir: Path) -> dict[str, Any]:
    frames: dict[str, pd.DataFrame] = {}
    for name in (
        "daily_decisions",
        "field_seasons",
        "events",
        "predictions",
        "notification_log",
        "alarm_states",
        "event_hits",
    ):
        frames[name] = pd.read_parquet(v3_dir / f"{name}.parquet")
    for name in (
        "event_metrics",
        "burden_metrics",
        "policy_selection",
        "pooled_summary",
        "paired_comparisons",
        "optuna_trials",
        "optuna_seed_checks",
    ):
        frames[name] = pd.read_csv(v3_dir / f"{name}.csv")
    for frame in frames.values():
        for column in (
            "issue_date",
            "issued_at",
            "first_recorded_event_date",
            "label_interval_end",
        ):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column])
    frames["contract"] = json.loads((v3_dir / "evaluation_contract.json").read_text(encoding="utf-8"))
    return frames


def _candidate_record(
    states: pd.DataFrame, seasons: pd.DataFrame, evaluation_scope: str
) -> dict[str, Any]:
    event, _ = event_metrics(
        states,
        seasons,
        "candidate",
        "validation",
        evaluation_scope=evaluation_scope,
    )
    burden = burden_metrics(
        states, "candidate", "validation", evaluation_scope=evaluation_scope
    )
    return {
        "timely_hits": event["timely_hits"],
        "events_with_warning_opportunity": event["events_with_warning_opportunity"],
        "timely_recall": event["timely_recall"],
        "messages": burden["messages"],
        "messages_per_30_field_days": burden["messages_per_30_field_days"],
        "active_alarm_fraction": burden["active_alarm_fraction"],
        "suppressed_repeats": burden["suppressed_repeats"],
        "computable_fraction": burden["computable_fraction"],
    }


def _annotate_feasibility(record: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    limits = contract["evaluation"]["selection_constraints"]
    result = dict(record)
    result["constraint_violation"] = max(
        0.0,
        float(result["messages_per_30_field_days"])
        - float(limits["messages_per_30_field_days_max"]),
    ) + max(
        0.0,
        float(result["active_alarm_fraction"]) - float(limits["active_alarm_fraction_max"]),
    )
    result["feasible"] = result["constraint_violation"] <= 1e-12
    return result


def _pick_alpha(records: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [record for record in records if bool(record["feasible"])]
    pool = feasible or records
    return sorted(
        pool,
        key=lambda r: (
            0.0 if feasible else float(r["constraint_violation"]),
            -float(np.nan_to_num(r["timely_recall"], nan=-1.0)),
            float(r["messages_per_30_field_days"]),
            float(r["active_alarm_fraction"]),
            int(r["suppressed_repeats"]),
            float(r["alpha"]),
            -float(r["threshold"]),
        ),
    )[0]


def _evaluate_one(
    frame: pd.DataFrame,
    seasons: pd.DataFrame,
    score: pd.Series,
    policy: Policy,
    scope: str,
    mask_column: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mask = frame[mask_column] if mask_column else None
    operational_score = score.reindex(frame.index).where(frame["service_active"].astype(bool))
    scoped_score = operational_score.where(mask) if mask is not None else operational_score
    states = simulate_policy(frame, scoped_score, policy, scope, mask)
    record = _candidate_record(states, seasons, scope)
    return states, record


def _select_alpha_and_policy(
    validation: pd.DataFrame,
    seasons: pd.DataFrame,
    score_by_alpha: dict[float, pd.Series],
    c0_policy: Policy,
    scope: str,
    mask_column: str | None,
    contract: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    limits = contract["evaluation"]["selection_constraints"]
    grid_rows: list[dict[str, Any]] = []
    replay_candidates: list[dict[str, Any]] = []
    selected_candidates: list[dict[str, Any]] = []
    for alpha, score in score_by_alpha.items():
        operational_score = score.reindex(validation.index).where(
            validation["service_active"].astype(bool)
        )
        replay_states, replay_metrics = _evaluate_one(
            validation, seasons, operational_score, c0_policy, scope, mask_column
        )
        replay = _annotate_feasibility(
            {
                **replay_metrics,
                "policy_mode": "c0_policy_replay",
                "alpha": alpha,
                "threshold": c0_policy.threshold,
                "active_days": c0_policy.active_days,
                "cooldown_days": c0_policy.cooldown_days,
                "policy_version": c0_policy.version,
            },
            contract,
        )
        replay_candidates.append(replay)
        grid_rows.append(replay)

        mask = validation[mask_column] if mask_column else None
        policy, details = select_threshold(
            validation,
            operational_score,
            seasons,
            int(limits["active_days_per_message"]),
            int(limits["cooldown_days"]),
            float(limits["messages_per_30_field_days_max"]),
            float(limits["active_alarm_fraction_max"]),
            scope,
            mask,
        )
        chosen = details["selected"]
        selected = _annotate_feasibility(
            {
                "policy_mode": "validation_selected",
                "alpha": alpha,
                "threshold": policy.threshold,
                "active_days": policy.active_days,
                "cooldown_days": policy.cooldown_days,
                "policy_version": policy.version,
                "timely_hits": chosen["timely_hits"],
                "events_with_warning_opportunity": chosen["events_with_warning_opportunity"],
                "timely_recall": chosen["timely_recall"],
                "messages": chosen["messages"],
                "messages_per_30_field_days": chosen["messages_per_30_field_days"],
                "active_alarm_fraction": chosen["active_alarm_fraction"],
                "suppressed_repeats": chosen["suppressed_repeats"],
                "computable_fraction": chosen["computable_fraction"],
                "threshold_candidates": len(details["candidates"]),
            },
            contract,
        )
        selected_candidates.append(selected)
        grid_rows.append(selected)
    return {
        "c0_policy_replay": _pick_alpha(replay_candidates),
        "validation_selected": _pick_alpha(selected_candidates),
    }, grid_rows


def _append_test_evaluation(
    outputs: dict[str, list[Any]],
    test: pd.DataFrame,
    test_seasons: pd.DataFrame,
    score: pd.Series,
    policy: Policy,
    model_code: str,
    fold_id: str,
    policy_mode: str,
    alpha: float,
    scope: str,
    mask_column: str | None,
    fallback: pd.Series,
) -> pd.DataFrame:
    states, _ = _evaluate_one(test, test_seasons, score, policy, scope, mask_column)
    states["model_code"] = f"{model_code}__{policy_mode}"
    states["model_family"] = model_code
    states["fold_id"] = fold_id
    states["model_version"] = "cycle2_nested_temporal_oof_v1"
    states["policy_mode"] = policy_mode
    states["alpha"] = alpha
    states["fallback_to_c0"] = fallback.reindex(states.index).fillna(False).astype(bool)
    outputs["alarm_states"].append(states)
    outputs["predictions"].append(
        states[
            [
                "field_season",
                "season",
                "issue_date",
                "issued_at",
                "score",
                "score_status",
                "evaluation_scope",
                "evaluation_scope_day",
                "message_issued",
                "alarm_active",
                "action_reason",
                "suppressed_repeat",
                "model_code",
                "model_family",
                "fold_id",
                "policy_mode",
                "alpha",
                "fallback_to_c0",
            ]
        ].copy()
    )
    public_code = f"{model_code}__{policy_mode}"
    for slice_name in ("A_plus_B", "direct_A", "prior_gap_le14", "prior_gap_le21"):
        event, hits = event_metrics(states, test_seasons, public_code, fold_id, slice_name, scope)
        event.update(model_family=model_code, policy_mode=policy_mode, alpha=alpha)
        outputs["event_metrics"].append(event)
        for hit in hits:
            hit.update(model_family=model_code, policy_mode=policy_mode, alpha=alpha)
        outputs["event_hits"].extend(hits)
    for slice_name in ("A_plus_B", "direct_A"):
        burden = burden_metrics(states, public_code, fold_id, slice_name, scope)
        selected = states[states["evaluation_scope_day"]]
        if slice_name == "direct_A":
            selected = selected[selected["coordinate_scope"].eq("A_direct")]
        burden.update(
            model_family=model_code,
            policy_mode=policy_mode,
            alpha=alpha,
            fallback_days=int(selected["fallback_to_c0"].sum()),
            fallback_day_fraction=float(selected["fallback_to_c0"].mean()) if len(selected) else np.nan,
        )
        outputs["burden_metrics"].append(burden)
    return states


def _c0_policy_table(v3: dict[str, Any]) -> dict[tuple[str, str], Policy]:
    table = v3["policy_selection"]
    table = table[table["model_code"].eq("C0")]
    policies: dict[tuple[str, str], Policy] = {}
    for row in table.itertuples(index=False):
        policies[(row.fold_id, row.evaluation_scope)] = Policy(
            threshold=float(row.threshold),
            active_days=int(row.active_days),
            cooldown_days=int(row.cooldown_days),
            version="cycle1_C0_policy_replay",
        )
    return policies


def _predict_grid(
    model_code: str,
    base_model: Any,
    correction_model: Any,
    frame: pd.DataFrame,
) -> dict[float, Any]:
    base_values = frame[list(CALENDAR_FEATURES)]
    base_raw = c0_raw_logits(base_model, base_values)
    base_probability = aligned_probabilities(base_model, base_values)
    result: dict[float, Any] = {}
    for alpha in ALPHA_GRID:
        if model_code == "C6_weather":
            result[alpha] = predict_c6_from_baseline(
                base_logits=base_raw,
                base_probabilities=base_probability,
                correction_model=correction_model,
                correction_values=frame[list(EPISODE_FEATURES)],
                alpha=alpha,
                correction_available=frame["episode_weather_complete"],
            )
        elif model_code == "C6_calendar_control":
            result[alpha] = predict_c6_from_baseline(
                base_logits=base_raw,
                base_probabilities=base_probability,
                correction_model=correction_model,
                correction_values=frame[list(CALENDAR_FEATURES)],
                alpha=alpha,
                correction_available=pd.Series(True, index=frame.index),
            )
        elif model_code == "C6_calibration_control":
            result[alpha] = predict_calibration_control(
                base_raw,
                base_probability,
                correction_model,
                alpha=alpha,
            )
        else:
            raise ValueError(f"Unknown cycle2 model: {model_code}")
    return result


def _raw_prediction_frame(
    source: pd.DataFrame,
    predictions: dict[float, Any],
    *,
    fold_id: str,
    model_code: str,
    split: str,
) -> pd.DataFrame:
    reference = predictions[1.0]
    columns = [
        column
        for column in (
            "field_season",
            "season",
            "issue_date",
            "issued_at",
            "target_class",
            "target_observable",
            "service_active",
            "episode_weather_complete",
            "candidate_comparison_complete",
        )
        if column in source
    ]
    result = source[columns].reset_index(drop=True).copy()
    result["fold_id"] = fold_id
    result["model_code"] = model_code
    result["split"] = split
    result["class_order"] = json.dumps(list(CLASS_ORDER))
    result["correction_available"] = reference.correction_available
    result["used_c0_fallback"] = reference.used_c0_fallback
    for position, class_id in enumerate(CLASS_ORDER):
        result[f"c0_raw_{class_id}"] = reference.base_logits[:, position]
        result[f"correction_raw_{class_id}"] = reference.correction_logits[:, position]
        for alpha, prediction in predictions.items():
            tag = str(alpha).replace(".", "p")
            result[f"c6_probability_{class_id}_alpha_{tag}"] = prediction.probabilities[:, position]
    return result


def _oof_component_frame(
    oof: pd.DataFrame,
    correction_model: Any,
    *,
    fold_id: str,
    model_code: str,
) -> pd.DataFrame:
    result = oof.copy()
    baseline = result[[f"c0_raw_{value}" for value in CLASS_ORDER]].to_numpy(dtype=float)
    base_probability = result[
        [f"c0_probability_{value}" for value in CLASS_ORDER]
    ].to_numpy(dtype=float)
    if model_code == "C6_weather":
        prediction = predict_c6_from_baseline(
            base_logits=baseline,
            base_probabilities=base_probability,
            correction_model=correction_model,
            correction_values=result[list(EPISODE_FEATURES)],
            alpha=1.0,
            correction_available=result["episode_weather_complete"],
        )
    elif model_code == "C6_calendar_control":
        prediction = predict_c6_from_baseline(
            base_logits=baseline,
            base_probabilities=base_probability,
            correction_model=correction_model,
            correction_values=result[list(CALENDAR_FEATURES)],
            alpha=1.0,
            correction_available=pd.Series(True, index=result.index),
        )
    elif model_code == "C6_calibration_control":
        prediction = predict_calibration_control(
            baseline, base_probability, correction_model, alpha=1.0
        )
    else:
        raise ValueError(model_code)
    result["outer_fold_id"] = fold_id
    result["model_code"] = model_code
    result["correction_available"] = prediction.correction_available
    for position, class_id in enumerate(CLASS_ORDER):
        result[f"correction_raw_{class_id}"] = prediction.correction_logits[:, position]
    return result


def _logit_distributions(frames: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouping = ["fold_id", "model_code", "split"]
    for keys, group in frames.groupby(grouping, sort=True):
        for component in ("c0_raw", "correction_raw"):
            for class_id in CLASS_ORDER:
                column = f"{component}_{class_id}"
                values = group[column].dropna().astype(float)
                rows.append(
                    {
                        **dict(zip(grouping, keys)),
                        "component": component,
                        "class_id": class_id,
                        "rows": len(values),
                        "mean": values.mean(),
                        "std": values.std(ddof=1),
                        "minimum": values.min(),
                        "q05": values.quantile(0.05),
                        "median": values.median(),
                        "q95": values.quantile(0.95),
                        "maximum": values.max(),
                    }
                )
    return pd.DataFrame(rows)


def _alpha_zero_identity(
    v3: dict[str, Any],
    test: pd.DataFrame,
    test_seasons: pd.DataFrame,
    prediction: Any,
    *,
    fold_id: str,
    scope: str,
    mask_column: str | None,
    policy: Policy,
) -> dict[str, Any]:
    score = pd.Series(prediction.actionable_probability, index=test.index)
    states, metrics = _evaluate_one(test, test_seasons, score, policy, scope, mask_column)
    old = v3["alarm_states"]
    old = old[
        old["model_code"].eq("C0")
        & old["fold_id"].eq(fold_id)
        & old["evaluation_scope"].eq(scope)
    ].copy()
    keys = ["field_season", "season", "issue_date"]
    columns = ["message_issued", "alarm_active", "suppressed_repeat", "action_reason"]
    left = states[keys + columns].sort_values(keys).reset_index(drop=True)
    right = old[keys + columns].sort_values(keys).reset_index(drop=True)
    same_population = left[keys].equals(right[keys])
    state_columns_equal = {
        column: bool(left[column].equals(right[column])) if same_population else False
        for column in columns
    }
    old_prediction = v3["predictions"]
    old_prediction = old_prediction[
        old_prediction["model_code"].eq("C0")
        & old_prediction["fold_id"].eq(fold_id)
        & old_prediction["evaluation_scope"].eq(scope)
    ][keys + ["score"]]
    new_score = states[keys + ["score"]]
    merged = new_score.merge(old_prediction, on=keys, suffixes=("_new", "_v3"), validate="one_to_one")
    score_mask = merged["score_v3"].notna()
    max_score_difference = (
        float((merged.loc[score_mask, "score_new"] - merged.loc[score_mask, "score_v3"]).abs().max())
        if score_mask.any()
        else np.nan
    )
    old_event = v3["event_metrics"]
    old_event = old_event[
        old_event["model_code"].eq("C0")
        & old_event["fold_id"].eq(fold_id)
        & old_event["evaluation_scope"].eq(scope)
        & old_event["slice"].eq("A_plus_B")
    ].iloc[0]
    old_burden = v3["burden_metrics"]
    old_burden = old_burden[
        old_burden["model_code"].eq("C0")
        & old_burden["fold_id"].eq(fold_id)
        & old_burden["evaluation_scope"].eq(scope)
        & old_burden["slice"].eq("A_plus_B")
    ].iloc[0]
    metric_equal = bool(
        int(metrics["timely_hits"]) == int(old_event["timely_hits"])
        and int(metrics["events_with_warning_opportunity"])
        == int(old_event["events_with_warning_opportunity"])
        and int(metrics["messages"]) == int(old_burden["messages"])
        and np.isclose(metrics["active_alarm_fraction"], old_burden["active_alarm_fraction"])
    )
    passed = bool(
        same_population
        and all(state_columns_equal.values())
        and max_score_difference == 0.0
        and metric_equal
    )
    return {
        "fold_id": fold_id,
        "evaluation_scope": scope,
        "alpha": 0.0,
        "threshold_reused": policy.threshold,
        "threshold_reselected": False,
        "same_daily_population": same_population,
        **{f"same_{key}": value for key, value in state_columns_equal.items()},
        "max_abs_c0_score_difference": max_score_difference,
        "same_event_and_burden_metrics": metric_equal,
        "status": "passed" if passed else "failed",
    }


def _save_bundle_and_verify(
    run_dir: Path,
    *,
    fold_id: str,
    model_code: str,
    policy_mode: str,
    base_model: Any,
    correction_model: Any,
    alpha: float,
    policy: Policy,
    sample: pd.DataFrame,
    oof_rows: int,
    parent_c0_sha256: str,
) -> dict[str, Any]:
    if model_code == "C6_weather":
        kind = "weather_catboost"
        features = tuple(EPISODE_FEATURES)
        availability = "episode_weather_complete"
    elif model_code == "C6_calendar_control":
        kind = "calendar_catboost"
        features = tuple(CALENDAR_FEATURES)
        availability = None
    elif model_code == "C6_calibration_control":
        kind = "calibration"
        features = tuple()
        availability = None
    else:
        raise ValueError(model_code)
    bundle = C6Bundle(
        base_model=base_model,
        correction_model=correction_model,
        correction_kind=kind,
        correction_features=features,
        alpha=alpha,
        availability_column=availability,
        policy=policy,
        metadata={
            "fold_id": fold_id,
            "model_code": model_code,
            "policy_mode": policy_mode,
            "oof_rows": oof_rows,
            "parent_c0_sha256": parent_c0_sha256,
            "policy": {
                "threshold": policy.threshold,
                "active_days": policy.active_days,
                "cooldown_days": policy.cooldown_days,
                "version": policy.version,
            },
        },
    )
    target = run_dir / "models" / fold_id / model_code / policy_mode
    save_c6_bundle(bundle, target)
    loaded = load_c6_bundle(target)
    before = bundle.predict(sample)
    after = loaded.predict(sample)
    differences = {
        "base_logits": float(np.max(np.abs(before.base_logits - after.base_logits))),
        "correction_logits": float(
            np.max(np.abs(before.correction_logits - after.correction_logits))
        ),
        "probabilities": float(np.max(np.abs(before.probabilities - after.probabilities))),
    }
    policy_equal = loaded.policy == policy
    fallback_equal = bool(
        np.array_equal(before.used_c0_fallback, after.used_c0_fallback)
    )
    return {
        "fold_id": fold_id,
        "model_code": model_code,
        "policy_mode": policy_mode,
        "alpha": alpha,
        "threshold": policy.threshold,
        "bundle_path": str(target.relative_to(run_dir)),
        "sample_rows": int(len(sample)),
        "fallback_rows_checked": int(np.asarray(before.used_c0_fallback, dtype=bool).sum()),
        **{f"max_abs_{key}_difference": value for key, value in differences.items()},
        "policy_roundtrip_equal": policy_equal,
        "fallback_mask_roundtrip_equal": fallback_equal,
        "status": "passed"
        if max(differences.values()) <= 1e-10 and policy_equal and fallback_equal
        else "failed",
    }


def _annual_summary(event_metrics_frame: pd.DataFrame, burden_metrics_frame: pd.DataFrame) -> pd.DataFrame:
    events = event_metrics_frame[event_metrics_frame["slice"].eq("A_plus_B")].copy()
    burdens = burden_metrics_frame[burden_metrics_frame["slice"].eq("A_plus_B")].copy()
    columns = ["model_code", "fold_id", "evaluation_scope"]
    result = events.merge(burdens, on=columns + ["slice"], suffixes=("_event", "_burden"))
    keep = [
        "model_code",
        "fold_id",
        "evaluation_scope",
        "model_family_event",
        "policy_mode_event",
        "alpha_event",
        "events_with_warning_opportunity",
        "timely_hits",
        "timely_recall",
        "computable_events",
        "computable_event_fraction",
        "field_days",
        "messages",
        "messages_per_30_field_days",
        "active_alarm_fraction",
        "computable_fraction",
        "fallback_days",
        "fallback_day_fraction",
    ]
    return result[[column for column in keep if column in result]].rename(
        columns={
            "model_family_event": "model_family",
            "policy_mode_event": "policy_mode",
            "alpha_event": "alpha",
        }
    )


def _add_pooled_fallback(pooled: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    result = pooled.copy()
    result["fallback_days"] = 0
    result["fallback_day_fraction"] = 0.0
    for index, row in result.iterrows():
        selected = states[
            states["model_code"].eq(row["model_code"])
            & states["evaluation_scope"].eq(row["evaluation_scope"])
            & states["season"].between(int(row["year_start"]), int(row["year_end"]))
            & states["evaluation_scope_day"]
        ]
        if row["slice"] == "direct_A":
            selected = selected[selected["coordinate_scope"].eq("A_direct")]
        fallback_days = int(selected.get("fallback_to_c0", False).sum()) if len(selected) else 0
        result.loc[index, "fallback_days"] = fallback_days
        result.loc[index, "fallback_day_fraction"] = (
            fallback_days / len(selected) if len(selected) else np.nan
        )
    return result


def _leave_one_year_out(
    event_hits_frame: pd.DataFrame,
    alarm_states_frame: pd.DataFrame,
    comparisons: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    events = event_hits_frame[
        event_hits_frame["slice"].eq("A_plus_B") & event_hits_frame["season"].between(2020, 2025)
    ]
    days = alarm_states_frame[alarm_states_frame["season"].between(2020, 2025)]
    for scope in SCOPES:
        scope_events = events[events["evaluation_scope"].eq(scope)]
        scope_days = days[days["evaluation_scope"].eq(scope) & days["evaluation_scope_day"]]
        for candidate, baseline in comparisons:
            for omitted in range(2020, 2026):
                ev = scope_events[scope_events["season"].ne(omitted)]
                dy = scope_days[scope_days["season"].ne(omitted)]
                left_e = ev[ev["model_code"].eq(candidate)]
                right_e = ev[ev["model_code"].eq(baseline)]
                left_d = dy[dy["model_code"].eq(candidate)]
                right_d = dy[dy["model_code"].eq(baseline)]
                if left_e.empty or right_e.empty or left_d.empty or right_d.empty:
                    continue
                opportunities = int(left_e["warnable_event"].sum())
                candidate_hits = int(left_e["timely_hit"].sum())
                baseline_hits = int(right_e["timely_hit"].sum())
                field_days = int(len(left_d))
                rows.append(
                    {
                        "evaluation_scope": scope,
                        "candidate": candidate,
                        "baseline": baseline,
                        "omitted_year": omitted,
                        "remaining_years": 5,
                        "opportunities": opportunities,
                        "candidate_hits": candidate_hits,
                        "baseline_hits": baseline_hits,
                        "delta_timely_recall": (candidate_hits - baseline_hits) / opportunities
                        if opportunities
                        else np.nan,
                        "delta_messages_per_30_field_days": 30
                        * (int(left_d["message_issued"].sum()) - int(right_d["message_issued"].sum()))
                        / field_days
                        if field_days
                        else np.nan,
                        "delta_active_alarm_fraction": (
                            int(left_d["alarm_active"].sum()) - int(right_d["alarm_active"].sum())
                        )
                        / field_days
                        if field_days
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _write_frames(run_dir: Path, outputs: dict[str, pd.DataFrame]) -> None:
    for name, frame in outputs.items():
        suffix = ".parquet" if name in {
            "oof_predictions",
            "c6_raw_predictions",
            "predictions",
            "alarm_states",
            "notification_log",
            "event_hits",
            "event_diagnostics",
            "event_pair_details",
            "gained_lost_events",
            "message_change_days",
        } else ".csv"
        if suffix == ".parquet":
            frame.to_parquet(run_dir / f"{name}{suffix}", index=False)
        else:
            frame.to_csv(run_dir / f"{name}{suffix}", index=False)


def _run_test_commands() -> dict[str, Any]:
    commands = [
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_early_warning_cycle2_models.py",
            "tests/test_early_warning_cycle2_diagnostics.py",
            "tests/test_early_warning_cycle2_reporting.py",
            "tests/test_early_warning_cycle2_pipeline.py",
        ],
        [sys.executable, "-m", "pytest", "-q", "tests"],
    ]
    records = []
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
    return {"status": "passed" if all(x["returncode"] == 0 for x in records) else "failed", "commands": records}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a new immutable second-cycle experiment")
    run.add_argument("--contract", required=True, type=Path)
    run.add_argument("--run-id", required=True)
    run.add_argument("--v3-run", type=Path, default=DEFAULT_V3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    path = run_cycle(args.contract.resolve(), args.run_id, args.v3_run.resolve())
    print(json.dumps({"status": "complete", "run_dir": str(path)}, ensure_ascii=False))
    return 0


def run_cycle(contract_path: Path, run_id: str, v3_dir: Path) -> Path:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    output_root = Path(contract["output_root"])
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    run_dir = output_root / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty and will not be overwritten: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    expected_v3_hash = contract["frozen_parent_cycle"]["execution_manifest_sha256"]
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "cycle": 2,
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
        "parent_run": str(v3_dir),
        "parent_manifest_expected_sha256": expected_v3_hash,
        "scientific_status": "retrospective_registration_proxy_only",
    }
    _write_json(run_dir / "execution_manifest.json", manifest)
    try:
        # Freeze the exact code and research documents before any fit starts.
        # The same hashes are checked again after all tests and reporting.
        source_snapshot = _write_source_snapshot(run_dir / "source_snapshot.json")
        manifest["source_snapshot"] = source_snapshot
        _write_json(run_dir / "execution_manifest.json", manifest)
        before = verify_v3(v3_dir, expected_v3_hash)
        _write_json(run_dir / "v3_integrity_before.json", before)
        shutil.copy2(contract_path, run_dir / "evaluation_contract.json")
        protocol_path = contract_path.with_name("cycle2_protocol.md")
        if not protocol_path.is_file():
            raise FileNotFoundError(protocol_path)
        shutil.copy2(protocol_path, run_dir / "protocol.md")

        # These diagnostics use only frozen v3 artefacts and are completed
        # before any new model is trained.
        diagnostic_outputs = build_v3_cycle2_diagnostics(v3_dir)
        _write_frames(run_dir, diagnostic_outputs)
        v3 = _load_v3(v3_dir)
        parent_contract = v3["contract"]
        era_relative = str(
            Path(parent_contract["inputs"]["frozen_external_dir"])
            / "era5_potato_daily.parquet"
        )
        era_path = Path(era_relative)
        if not era_path.is_absolute():
            era_path = REPO_ROOT / era_path
        era_expected = before["input_hashes_recorded_by_v3"].get(era_relative)
        era_actual = sha256_file(era_path) if era_path.is_file() else None
        direct_input_audit = {
            "status": "passed" if era_actual is not None and era_actual == era_expected else "failed",
            "inputs": [
                {
                    "role": "funnel_weather_boundary_diagnostic",
                    "relative_path": era_relative,
                    "expected_sha256_from_v3": era_expected,
                    "actual_sha256": era_actual,
                    "matches": era_actual == era_expected,
                }
            ],
            "v3_outputs_used": sorted(
                {
                    "daily_decisions.parquet",
                    "field_seasons.parquet",
                    "events.parquet",
                    "predictions.parquet",
                    "notification_log.parquet",
                    "alarm_states.parquet",
                    "event_hits.parquet",
                    "event_metrics.csv",
                    "burden_metrics.csv",
                    "policy_selection.csv",
                    "pooled_summary.csv",
                    "paired_comparisons.csv",
                    "optuna_trials.csv",
                    "optuna_seed_checks.csv",
                    "models/*_C0.joblib",
                }
            ),
        }
        _write_json(run_dir / "cycle2_input_audit.json", direct_input_audit)
        if direct_input_audit["status"] != "passed":
            raise AssertionError("Direct frozen ERA input does not match the v3 recorded hash")
        folds = parent_contract["rolling_origin_folds"]
        decisions = v3["daily_decisions"]
        seasons = v3["field_seasons"]
        label_availability = seasons[
            [
                "field_season",
                "first_recorded_event_available_date",
                "last_visit_date",
            ]
        ].copy()
        label_availability["label_available_at"] = pd.to_datetime(
            label_availability["first_recorded_event_available_date"], errors="coerce"
        ).fillna(
            pd.to_datetime(label_availability["last_visit_date"], errors="coerce")
            + pd.Timedelta(days=1)
        )
        decisions_for_oof = decisions.merge(
            label_availability[["field_season", "label_available_at"]],
            on="field_season",
            how="left",
            validate="many_to_one",
        )
        if decisions_for_oof["label_available_at"].isna().any():
            raise AssertionError("Missing causal label-availability date for OOF rows")
        c0_policies = _c0_policy_table(v3)
        seed = int(contract["random_seed"])
        model_outputs: dict[str, list[Any]] = {
            "oof_predictions": [],
            "oof_provenance": [],
            "c6_raw_predictions": [],
            "validation_grid": [],
            "policy_selection": [],
            "predictions": [],
            "alarm_states": [],
            "event_metrics": [],
            "event_hits": [],
            "burden_metrics": [],
            "alpha0_identity_checks": [],
            "model_reload_verification": [],
            "correction_training_summary": [],
        }

        from .early_warning_cycle2_models import build_expanding_c0_oof

        for fold_index, fold in enumerate(folds):
            fold_id = str(fold["id"])
            train_years = list(fold["train_years"])
            validation = decisions[_year_mask(decisions, fold["validation_years"])].copy()
            test = decisions[_year_mask(decisions, fold["test_years"])].copy()
            validation_seasons = seasons[_year_mask(seasons, fold["validation_years"])].copy()
            test_seasons = seasons[_year_mask(seasons, fold["test_years"])].copy()
            if validation.empty or test.empty:
                raise AssertionError(f"Empty validation or test block in {fold_id}")

            oof_result = build_expanding_c0_oof(decisions_for_oof, train_years, seed=seed)
            oof = oof_result.predictions
            provenance = oof_result.provenance.copy()
            predicted_provenance = provenance[provenance["status"].eq("predicted")]
            if oof.empty or not predicted_provenance["temporal_order_verified"].all():
                raise AssertionError(f"Invalid temporal OOF in {fold_id}")
            provenance["outer_fold_id"] = fold_id
            provenance["outer_train_year_start"] = train_years[0]
            provenance["outer_train_year_end"] = train_years[1]
            provenance["inner_forecast_year"] = provenance["forecast_year"]
            provenance["train_fold"] = provenance.apply(
                lambda row: f"{int(row['baseline_fit_start_year'])}_{int(row['baseline_fit_end_year'])}",
                axis=1,
            )
            provenance["forecast_fold"] = provenance["forecast_year"].map(
                lambda value: f"year_{int(value)}"
            )
            provenance["baseline_fit_year_start"] = provenance[
                "baseline_fit_start_year"
            ]
            provenance["baseline_fit_year_end"] = provenance["baseline_fit_end_year"]
            provenance["train_rows"] = provenance["baseline_train_rows"]
            provenance["train_field_seasons"] = provenance[
                "baseline_train_field_seasons"
            ]
            provenance["baseline_fit_end"] = provenance["baseline_max_issue_date"]
            provenance["max_label_available_at"] = provenance[
                "baseline_max_label_availability_date"
            ]
            for row_index, row in provenance.iterrows():
                forecast_year = int(row["forecast_year"])
                history = _model_rows(
                    decisions_for_oof, [train_years[0], forecast_year - 1]
                )
                history = history[
                    pd.to_datetime(history["label_available_at"]).lt(
                        pd.Timestamp(year=forecast_year, month=1, day=1)
                    )
                ]
                forecast = _model_rows(decisions_for_oof, [forecast_year, forecast_year])
                provenance.loc[row_index, "baseline_train_warnable_events"] = int(
                    history.loc[history["warnable_first_event"], "field_season"].nunique()
                )
                provenance.loc[row_index, "forecast_warnable_events"] = int(
                    forecast.loc[forecast["warnable_first_event"], "field_season"].nunique()
                )
                provenance.loc[row_index, "forecast_field_seasons"] = int(
                    forecast["field_season"].nunique()
                )
            provenance["train_warnable_events"] = provenance[
                "baseline_train_warnable_events"
            ]
            model_outputs["oof_provenance"].append(provenance)

            correction_models = {
                "C6_weather": fit_weather_catboost_control(oof, seed=seed),
                "C6_calibration_control": fit_oof_calibration_control(
                    oof,
                    seed=seed,
                    regularization_c=float(
                        contract["negative_controls"]["calibration_only"]["regularization_C"]
                    ),
                ),
                "C6_calendar_control": fit_calendar_catboost_control(oof, seed=seed),
            }
            outer_c0_train = _model_rows(decisions_for_oof, train_years)
            for summary_model in correction_models:
                model_outputs["correction_training_summary"].append(
                    {
                        "fold_id": fold_id,
                        "model_code": summary_model,
                        "rows": int(len(oof)),
                        "field_seasons": int(oof["field_season"].nunique()),
                        "warnable_events": int(
                            seasons.loc[
                                seasons["field_season"].isin(oof["field_season"])
                                & seasons["warnable_first_event"],
                                "field_season",
                            ].nunique()
                        ),
                        "forecast_year_start": int(oof["forecast_year"].min()),
                        "forecast_year_end": int(oof["forecast_year"].max()),
                        "class_0_rows": int(oof["target_int"].eq(0).sum()),
                        "class_1_rows": int(oof["target_int"].eq(1).sum()),
                        "class_2_rows": int(oof["target_int"].eq(2).sum()),
                        "validation_rows": int(len(validation)),
                        "validation_service_days": int(
                            validation["service_active"].astype(bool).sum()
                        ),
                        "validation_paired_days": int(
                            (
                                validation["service_active"].astype(bool)
                                & validation["candidate_comparison_complete"].astype(bool)
                            ).sum()
                        ),
                        "validation_field_seasons": int(validation["field_season"].nunique()),
                        "validation_warnable_events": int(
                            validation_seasons["warnable_first_event"].sum()
                        ),
                        "outer_c0_refit_rows": int(len(outer_c0_train)),
                        "outer_c0_refit_field_seasons": int(
                            outer_c0_train["field_season"].nunique()
                        ),
                        "outer_c0_refit_warnable_events": int(
                            seasons.loc[
                                seasons["field_season"].isin(outer_c0_train["field_season"])
                                & seasons["warnable_first_event"],
                                "field_season",
                            ].nunique()
                        ),
                    }
                )
            base_path = v3_dir / "models" / f"{fold_id}_C0.joblib"
            base_model = joblib.load(base_path)

            for model_code, correction_model in correction_models.items():
                oof_components = _oof_component_frame(
                    oof, correction_model, fold_id=fold_id, model_code=model_code
                )
                model_outputs["oof_predictions"].append(oof_components)
                validation_grid = _predict_grid(
                    model_code, base_model, correction_model, validation
                )
                test_grid = _predict_grid(model_code, base_model, correction_model, test)
                model_outputs["c6_raw_predictions"].append(
                    _raw_prediction_frame(
                        validation,
                        validation_grid,
                        fold_id=fold_id,
                        model_code=model_code,
                        split="outer_validation",
                    )
                )
                model_outputs["c6_raw_predictions"].append(
                    _raw_prediction_frame(
                        test,
                        test_grid,
                        fold_id=fold_id,
                        model_code=model_code,
                        split="outer_test",
                    )
                )

                for scope, mask_column in SCOPES.items():
                    c0_policy = c0_policies[(fold_id, scope)]
                    chosen, grid_rows = _select_alpha_and_policy(
                        validation,
                        validation_seasons,
                        {
                            alpha: pd.Series(pred.actionable_probability, index=validation.index)
                            for alpha, pred in validation_grid.items()
                        },
                        c0_policy,
                        scope,
                        mask_column,
                        contract,
                    )
                    for grid_row in grid_rows:
                        model_outputs["validation_grid"].append(
                            {
                                **grid_row,
                                "fold_id": fold_id,
                                "model_code": model_code,
                                "evaluation_scope": scope,
                            }
                        )
                    identity = _alpha_zero_identity(
                        v3,
                        test,
                        test_seasons,
                        test_grid[0.0],
                        fold_id=fold_id,
                        scope=scope,
                        mask_column=mask_column,
                        policy=c0_policy,
                    )
                    identity["model_code"] = model_code
                    model_outputs["alpha0_identity_checks"].append(identity)
                    if identity["status"] != "passed":
                        raise AssertionError(
                            f"alpha=0 failed to reproduce C0 in {fold_id}/{scope}/{model_code}"
                        )

                    for policy_mode in POLICY_MODES:
                        selected = chosen[policy_mode]
                        alpha = float(selected["alpha"])
                        policy = Policy(
                            threshold=float(selected["threshold"]),
                            active_days=int(selected["active_days"]),
                            cooldown_days=int(selected["cooldown_days"]),
                            version=f"cycle2_{policy_mode}_{scope}",
                        )
                        selected_row = {
                            **selected,
                            "validation_timely_recall": selected["timely_recall"],
                            "validation_messages_per_30": selected[
                                "messages_per_30_field_days"
                            ],
                            "validation_alarm_fraction": selected[
                                "active_alarm_fraction"
                            ],
                            "fold_id": fold_id,
                            "model_code": model_code,
                            "evaluation_scope": scope,
                            "selection_population": "outer_validation_only",
                            "correction_training_population": "inner_expanding_year_oof_within_outer_train",
                            "correction_training_rows": int(len(oof)),
                            "baseline_model_id": "C0",
                        }
                        model_outputs["policy_selection"].append(selected_row)
                        prediction = test_grid[alpha]
                        fallback = pd.Series(
                            prediction.used_c0_fallback, index=test.index, dtype=bool
                        )
                        _append_test_evaluation(
                            model_outputs,
                            test,
                            test_seasons,
                            pd.Series(prediction.actionable_probability, index=test.index),
                            policy,
                            model_code,
                            fold_id,
                            policy_mode,
                            alpha,
                            scope,
                            mask_column,
                            fallback,
                        )
                        if model_code == "C6_weather":
                            weather_available = test["episode_weather_complete"].astype(bool)
                            complete_sample = test.loc[weather_available].head(25)
                            fallback_sample = test.loc[~weather_available].head(25)
                            if not complete_sample.empty and not fallback_sample.empty:
                                sample = pd.concat([complete_sample, fallback_sample]).sort_index()
                            else:
                                sample = test.head(50)
                        else:
                            sample = test.head(50)
                        reload_record = _save_bundle_and_verify(
                            run_dir,
                            fold_id=fold_id,
                            model_code=model_code,
                            policy_mode=f"{policy_mode}__{scope}",
                            base_model=base_model,
                            correction_model=correction_model,
                            alpha=alpha,
                            policy=policy,
                            sample=sample,
                            oof_rows=len(oof),
                            parent_c0_sha256=sha256_file(base_path),
                        )
                        model_outputs["model_reload_verification"].append(reload_record)
                        if reload_record["status"] != "passed":
                            raise AssertionError(f"Bundle reload failed: {reload_record}")

        frames: dict[str, pd.DataFrame] = {}
        for name, parts in model_outputs.items():
            if name == "event_hits":
                frames[name] = pd.DataFrame(parts)
            elif parts and isinstance(parts[0], pd.DataFrame):
                frames[name] = pd.concat(parts, ignore_index=True)
            else:
                frames[name] = pd.DataFrame(parts)
        if not frames["alpha0_identity_checks"]["status"].eq("passed").all():
            raise AssertionError("Not all alpha=0 identity checks passed")
        if not frames["model_reload_verification"]["status"].eq("passed").all():
            raise AssertionError("Not all bundle roundtrip checks passed")

        frames["annual_metrics"] = _annual_summary(
            frames["event_metrics"], frames["burden_metrics"]
        )
        notification_columns = [
            "field_season",
            "season",
            "issued_at",
            "issue_date",
            "model_code",
            "model_family",
            "model_version",
            "policy_mode",
            "evaluation_scope",
            "alpha",
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
            "fallback_to_c0",
        ]
        notifications = frames["alarm_states"]
        notifications = notifications[
            notifications["message_issued"] | notifications["suppressed_repeat"]
        ].copy()
        frames["notification_log"] = notifications[
            [column for column in notification_columns if column in notifications]
        ]
        pooled = aggregate_pooled_metrics(
            frames["event_metrics"],
            frames["burden_metrics"],
            event_hits=frames["event_hits"],
            alarm_states=frames["alarm_states"],
        )
        pooled["pooled_burden_metrics"] = _add_pooled_fallback(
            pooled["pooled_burden_metrics"], frames["alarm_states"]
        )
        pooled["pooled_summary"] = pooled["pooled_event_metrics"].merge(
            pooled["pooled_burden_metrics"],
            on=[
                "period",
                "year_start",
                "year_end",
                "test_years",
                "model_code",
                "evaluation_scope",
                "slice",
            ],
            how="outer",
            validate="one_to_one",
        )
        frames.update(pooled)

        comparator_models = ["calendar_window", "C0", "C1", "C4", "C5"]
        old_hits = v3["event_hits"][v3["event_hits"]["model_code"].isin(comparator_models)]
        old_states = v3["alarm_states"][v3["alarm_states"]["model_code"].isin(comparator_models)]
        combined_hits = pd.concat([old_hits, frames["event_hits"]], ignore_index=True, sort=False)
        combined_states = pd.concat([old_states, frames["alarm_states"]], ignore_index=True, sort=False)
        comparisons: list[tuple[str, str]] = []
        for mode in POLICY_MODES:
            weather = f"C6_weather__{mode}"
            comparisons.extend((weather, baseline) for baseline in comparator_models)
            comparisons.append((weather, f"C6_calibration_control__{mode}"))
            comparisons.append((weather, f"C6_calendar_control__{mode}"))
        frames["paired_bootstrap"] = paired_year_bootstrap(
            combined_hits,
            combined_states,
            seed=int(contract["evaluation"]["uncertainty"]["seed"]),
            n_bootstrap=int(
                contract["evaluation"]["uncertainty"]["paired_year_bootstrap_repeats"]
            ),
            comparisons=comparisons,
        )
        frames["leave_one_year_out"] = _leave_one_year_out(
            combined_hits, combined_states, comparisons
        )
        from .early_warning_cycle2_reporting import build_cycle2_reporting_artifacts

        strict_reporting = build_cycle2_reporting_artifacts(
            frames["event_hits"],
            frames["alarm_states"],
            reference_event_hits=old_hits,
            reference_alarm_states=old_states,
            seed=int(contract["evaluation"]["uncertainty"]["seed"]),
            n_bootstrap=int(
                contract["evaluation"]["uncertainty"]["paired_year_bootstrap_repeats"]
            ),
        )
        # The strict reporting implementation checks exact event/day pairing.
        # Its LOO table supersedes the compact internal calculation above.
        frames.update(strict_reporting)
        oof_for_distribution = frames["oof_predictions"].rename(
            columns={"outer_fold_id": "fold_id"}
        ).copy()
        oof_for_distribution["split"] = "inner_temporal_oof"
        raw_for_distribution = pd.concat(
            [frames["c6_raw_predictions"], oof_for_distribution],
            ignore_index=True,
            sort=False,
        )
        frames["logit_distributions"] = _logit_distributions(raw_for_distribution)
        _write_frames(run_dir, frames)

        tests = _run_test_commands()
        _write_json(run_dir / "test_results.json", tests)
        if tests["status"] != "passed":
            raise RuntimeError("Cycle2 or repository tests failed")
        after = verify_v3(v3_dir, expected_v3_hash)
        _write_json(run_dir / "v3_integrity_after.json", after)

        # The report module consumes only aggregate tables and privacy-safe
        # pseudonymous details from this new run.
        from .early_warning_cycle2_reporting import write_cycle2_report_ru

        write_cycle2_report_ru(run_dir, v3_dir=v3_dir, contract=contract)
        (run_dir / "REPRODUCE.md").write_text(
            "# Воспроизведение второго цикла\n\n"
            "```bash\n"
            ".venv/bin/agro-late-blight-cycle2 run \\\n"
            "  --contract docs/research/late_blight_early_warning/cycle2_evaluation_contract.json \\\n"
            "  --v3-run results/late_blight_early_warning/20260910_first_cycle_v3 \\\n"
            "  --run-id <new_unique_cycle2_run_id>\n"
            "```\n\n"
            "Каталог запуска должен отсутствовать или быть пустым. Команда не изменяет v3.\n",
            encoding="utf-8",
        )
        source_integrity_after = _verify_source_snapshot(source_snapshot)
        _write_json(run_dir / "source_integrity_after.json", source_integrity_after)
        manifest.update(
            {
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "source_snapshot": source_snapshot,
                "source_integrity_after": {
                    "status": source_integrity_after["status"],
                    "files_checked": source_integrity_after["files_checked"],
                    "artifact": "source_integrity_after.json",
                    "sha256": sha256_file(run_dir / "source_integrity_after.json"),
                },
                "parent_integrity_before": before["status"],
                "parent_integrity_after": after["status"],
                "parent_outputs_checked": after["outputs_checked"],
                "fold_definitions": folds,
                "model_configuration": contract["c6"],
                "tests_status": tests["status"],
                "input_hashes": {
                    "cycle2_contract": {
                        "path": str(contract_path),
                        "sha256": sha256_file(contract_path),
                    },
                    "cycle2_protocol": {
                        "path": str(protocol_path),
                        "sha256": sha256_file(protocol_path),
                    },
                    "historical_research_report": {
                        "path": str(
                            REPO_ROOT
                            / "docs/research/late_blight_early_warning"
                            / "AGROPHENOLOGY_POTATO_LATE_BLIGHT_FULL_REPORT_2026-09-15.md"
                        ),
                        "sha256": sha256_file(
                            REPO_ROOT
                            / "docs/research/late_blight_early_warning"
                            / "AGROPHENOLOGY_POTATO_LATE_BLIGHT_FULL_REPORT_2026-09-15.md"
                        ),
                    },
                    "frozen_parent_execution_manifest": {
                        "path": str(v3_dir / "execution_manifest.json"),
                        "sha256": before["execution_manifest_sha256"],
                    },
                    "frozen_era_weather": {
                        "path": str(era_path),
                        "sha256": era_actual,
                    },
                    "audit_artifacts": {
                        name: {
                            "path": name,
                            "sha256": sha256_file(run_dir / name),
                        }
                        for name in (
                            "cycle2_input_audit.json",
                            "v3_integrity_before.json",
                            "v3_integrity_after.json",
                            "source_snapshot.json",
                            "source_integrity_after.json",
                        )
                    },
                },
            }
        )
        manifest["output_hashes"] = _output_hashes(run_dir)
        _write_json(run_dir / "execution_manifest.json", manifest)
        return run_dir
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "partial_output_hashes": _output_hashes(run_dir),
            }
        )
        _write_json(run_dir / "execution_manifest.json", manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
