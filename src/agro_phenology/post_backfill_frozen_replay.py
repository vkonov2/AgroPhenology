"""Frozen C3--C6 replay after the additive ERA5 tail backfill.

The command in this module is an inference-only audit.  It rebuilds the v3
decision calendar twice (with the original and extended ERA5 bundles), loads
the already saved models and notification policies, and replays each field
season chronologically.  It deliberately contains no fitting, threshold
search, alpha search, or model selection code.

Outputs are always written below ``<run-dir>/frozen_replay``.  Existing v3,
v4, cycle-3, backfill, and frozen-weather directories are read-only inputs.
The 2020--2025 comparison is retrospective and must not be described as a new
untouched external validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from .early_warning_core import (
    CALENDAR_FEATURES,
    EPISODE_FEATURES,
    ERA_COMMON_FEATURES,
    NASA_COMMON_FEATURES,
    add_daily_features,
    build_daily_decisions,
    sha256_file,
)
from .early_warning_cycle2_models import C6Bundle, load_c6_bundle
from .early_warning_cycle3_policy import GrowthPolicy, simulate_growth_policy
from .early_warning_models import (
    MODEL_SPECS,
    Policy,
    burden_metrics,
    event_metrics,
    score_model,
    simulate_policy,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results/late_blight_early_warning"
DEFAULT_V3 = RESULTS_ROOT / "20260910_first_cycle_v3"
DEFAULT_V4 = RESULTS_ROOT / "20260910_second_cycle_v4"
DEFAULT_CYCLE3 = RESULTS_ROOT / "20260910_third_cycle_v2"
DEFAULT_BACKFILL = RESULTS_ROOT / "20260915_era5_hutton_backfill_v1"
DEFAULT_FROZEN_ERA = (
    REPO_ROOT
    / "docs/extra/vaad_pipeline_repro_20260905/frozen_external/era5_potato_daily.parquet"
)
DEFAULT_EXTENDED_ERA = DEFAULT_BACKFILL / "era5_potato_daily_extended.parquet"
DEFAULT_NASA_DAILY = (
    REPO_ROOT / "docs/extra/vaad_pipeline_repro_20260905/frozen_external/nasa_daily.parquet"
)
DEFAULT_NASA_MAPPING = (
    REPO_ROOT
    / "docs/extra/vaad_pipeline_repro_20260905/frozen_external/nasa_coordinate_mapping.parquet"
)

EXPECTED_V3_MANIFEST_SHA256 = (
    "c7976317e6d94b8d1ca55b0bd618226c769914280bd7896ca36b5e0ca2cf3c80"
)
EXPECTED_V4_MANIFEST_SHA256 = (
    "1355bc9742f397a893774ed02edd7b4080364714d49f208ed75b190d8fd856c0"
)
EXPECTED_CYCLE3_MANIFEST_SHA256 = (
    "38f56c053d07fd390ced397bfd9e3566ba441581dc4feb4c8029fdad2920a833"
)
EXPECTED_BACKFILL_MANIFEST_SHA256 = (
    "44dc4b2462cb3ad400e1ddfb55a18c8fa22082472c0cae5a460ae840ee2de245"
)

RUN_FORMAT = "agro_phenology_post_backfill_frozen_replay_v1"
OUTPUT_SUBDIR = "frozen_replay"
TEST_YEARS = tuple(range(2020, 2026))
V3_MODEL_CODES = ("C3", "C4", "C5")
C6_POLICY_FAMILIES = ("P0_saved", "P_growth_selected")
EVALUATION_SCOPE = "service_calendar"
KEY_COLUMNS = ["field_season", "season", "issue_date"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NA or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, payload: str) -> None:
    _atomic_bytes(path, payload.encode("utf-8"))


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
    )


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def output_directory(run_dir: str | Path) -> Path:
    """Return the only directory this command is allowed to mutate."""

    return Path(run_dir).expanduser().resolve() / OUTPUT_SUBDIR


def _safe_relative_path(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe path in frozen manifest: {value}")
    return relative


def _manifest_expected_hash(metadata: Any) -> str:
    if isinstance(metadata, str):
        return metadata
    if isinstance(metadata, Mapping) and isinstance(metadata.get("sha256"), str):
        return str(metadata["sha256"])
    raise ValueError("Manifest output hash must be a SHA-256 string or mapping")


def verify_frozen_run(
    run_dir: str | Path,
    expected_manifest_sha256: str | None = None,
    *,
    expected_format: str | None = None,
) -> dict[str, Any]:
    """Verify a completed immutable parent run and all declared outputs."""

    source = Path(run_dir).expanduser().resolve()
    manifest_path = source / "execution_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    if expected_manifest_sha256 and manifest_sha256 != expected_manifest_sha256:
        raise AssertionError(
            f"Unexpected manifest hash for {source}: {manifest_sha256}; "
            f"expected {expected_manifest_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise AssertionError(f"Frozen run is not complete: {source}")
    if expected_format is not None and manifest.get("format") != expected_format:
        raise AssertionError(f"Unexpected frozen run format: {source}")

    checked: list[dict[str, Any]] = []
    failures: list[str] = []
    for raw_relative, metadata in sorted(manifest.get("output_hashes", {}).items()):
        relative = _safe_relative_path(str(raw_relative))
        path = source / relative
        expected = _manifest_expected_hash(metadata)
        actual = sha256_file(path) if path.is_file() else None
        matches = actual == expected
        checked.append(
            {
                "relative_path": str(relative),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": matches,
            }
        )
        if not matches:
            failures.append(str(relative))
    if failures:
        raise AssertionError(f"Frozen output hash failures in {source}: {failures}")
    return {
        "status": "passed",
        "path": str(source),
        "execution_manifest_sha256": manifest_sha256,
        "outputs_checked": len(checked),
        "all_output_hashes_match": True,
        "files": checked,
        "manifest": manifest,
    }


def _ensure_output_directory(run_dir: str | Path) -> Path:
    target = output_directory(run_dir)
    marker = target / ".frozen_replay_run.json"
    if target.exists() and any(target.iterdir()) and not marker.is_file():
        raise FileExistsError(
            f"Refusing non-empty unrecognised output directory: {target}"
        )
    target.mkdir(parents=True, exist_ok=True)
    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("format") != RUN_FORMAT:
            raise ValueError("Output directory belongs to another run format")
    else:
        _atomic_json(
            marker,
            {
                "format": RUN_FORMAT,
                "created_at_utc": _utc_now(),
                "private_local_only": True,
            },
        )
    return target


def _fold_id(year: int) -> str:
    if int(year) not in TEST_YEARS:
        raise ValueError(f"Unsupported replay year: {year}")
    return f"test_{int(year)}"


def _one_record(frame: pd.DataFrame, mask: pd.Series, description: str) -> dict[str, Any]:
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise AssertionError(f"Expected one {description}, found {len(selected)}")
    return selected.iloc[0].to_dict()


def read_v3_policy_records(
    policy_csv: str | Path,
    years: Iterable[int] = TEST_YEARS,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Read exact v3 service policies; no threshold candidates are evaluated."""

    frame = pd.read_csv(policy_csv)
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for year in sorted({int(value) for value in years}):
        fold_id = _fold_id(year)
        for model_code in V3_MODEL_CODES:
            record = _one_record(
                frame,
                frame["fold_id"].eq(fold_id)
                & frame["model_code"].eq(model_code)
                & frame["evaluation_scope"].eq(EVALUATION_SCOPE),
                f"v3 {fold_id}/{model_code}/{EVALUATION_SCOPE} policy",
            )
            records[(fold_id, model_code)] = record
    return records


def read_c6_policy_records(
    policy_csv: str | Path,
    years: Iterable[int] = TEST_YEARS,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Read only the two preselected final C6 weather policies by fold.

    The allow-list is intentional: this function never ranks rows and never
    reads ``validation_policy_candidates.csv``.
    """

    frame = pd.read_csv(policy_csv)
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for year in sorted({int(value) for value in years}):
        fold_id = _fold_id(year)
        for family in C6_POLICY_FAMILIES:
            record = _one_record(
                frame,
                frame["fold_id"].eq(fold_id)
                & frame["score_model"].eq("C6_weather")
                & frame["evaluation_scope"].eq(EVALUATION_SCOPE)
                & frame["candidate_family"].eq(family),
                f"cycle3 {fold_id}/C6_weather/{family} policy",
            )
            records[(fold_id, family)] = record
    return records


def _v4_policy_record(policy_csv: Path, fold_id: str) -> dict[str, Any]:
    frame = pd.read_csv(policy_csv)
    return _one_record(
        frame,
        frame["fold_id"].eq(fold_id)
        & frame["model_code"].eq("C6_weather")
        & frame["evaluation_scope"].eq(EVALUATION_SCOPE)
        & frame["policy_mode"].eq("validation_selected"),
        f"v4 {fold_id}/C6_weather/validation_selected policy",
    )


def _float_equal(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(float(left), float(right), atol=tolerance, rtol=0.0))


def load_frozen_c6_configurations(
    v4_dir: str | Path,
    cycle3_dir: str | Path,
    years: Iterable[int] = TEST_YEARS,
) -> dict[tuple[str, str], tuple[C6Bundle, dict[str, Any]]]:
    """Load frozen C6 bundles and validate both final policy families."""

    v4 = Path(v4_dir)
    cycle3 = Path(cycle3_dir)
    selected_years = tuple(sorted({int(value) for value in years}))
    records = read_c6_policy_records(cycle3 / "policy_selection.csv", selected_years)
    result: dict[tuple[str, str], tuple[C6Bundle, dict[str, Any]]] = {}
    for year in selected_years:
        fold_id = _fold_id(year)
        bundle_dir = (
            v4
            / "models"
            / fold_id
            / "C6_weather"
            / "validation_selected__service_calendar"
        )
        bundle = load_c6_bundle(bundle_dir)
        v4_record = _v4_policy_record(v4 / "policy_selection.csv", fold_id)
        if bundle.policy is None:
            raise AssertionError(f"C6 bundle lacks its saved policy: {bundle_dir}")
        if not _float_equal(bundle.alpha, v4_record["alpha"]):
            raise AssertionError(f"C6 bundle alpha differs from v4 policy in {fold_id}")
        if not _float_equal(bundle.policy.threshold, v4_record["threshold"]):
            raise AssertionError(f"C6 bundle threshold differs from v4 policy in {fold_id}")
        for family in C6_POLICY_FAMILIES:
            record = records[(fold_id, family)]
            if not _float_equal(record["alpha"], bundle.alpha):
                raise AssertionError(f"Cycle3 alpha differs from C6 bundle in {fold_id}/{family}")
            if int(record["active_days"]) != int(bundle.policy.active_days):
                raise AssertionError(f"Active days differ in {fold_id}/{family}")
            if family == "P0_saved":
                if not _float_equal(record["threshold"], bundle.policy.threshold):
                    raise AssertionError(f"P0_saved threshold differs from v4 in {fold_id}")
                if int(record["cooldown_days"]) != int(bundle.policy.cooldown_days):
                    raise AssertionError(f"P0_saved cooldown differs from v4 in {fold_id}")
                if bool(record["growth_override_enabled"]):
                    raise AssertionError("P0_saved unexpectedly enables growth")
            elif not bool(record["growth_override_enabled"]):
                raise AssertionError("P_growth_selected must enable the frozen growth rule")
            result[(fold_id, family)] = (bundle, record)
    return result


def _load_v3_model(v3_dir: Path, fold_id: str, model_code: str) -> Any:
    path = v3_dir / "models" / f"{fold_id}_{model_code}"
    if model_code in {"C3", "C4"}:
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(path.with_suffix(".cbm"))
        return model
    if model_code == "C5":
        return joblib.load(path.with_suffix(".joblib"))
    raise ValueError(f"Unsupported v3 model: {model_code}")


def _growth_policy(record: Mapping[str, Any], family: str) -> GrowthPolicy:
    enabled = bool(record["growth_override_enabled"])
    raw_delta = record.get("growth_logit_delta")
    delta = None if not enabled or pd.isna(raw_delta) else float(raw_delta)
    return GrowthPolicy(
        threshold=float(record["threshold"]),
        active_days=int(record["active_days"]),
        cooldown_days=int(record["cooldown_days"]),
        minimum_repeat_interval_days=int(record["minimum_repeat_interval_days"]),
        growth_override_enabled=enabled,
        growth_logit_delta=delta,
        epsilon=1e-6,
        version=f"cycle3_{family}_{EVALUATION_SCOPE}",
    )


def _c6_runtime(
    frame: pd.DataFrame,
    bundle: C6Bundle,
) -> dict[str, pd.Series | float]:
    prediction = bundle.predict(frame)
    score = pd.Series(prediction.actionable_probability, index=frame.index, dtype=float)
    score = score.where(frame["service_active"].astype(bool))
    available = pd.Series(prediction.correction_available, index=frame.index, dtype=bool)
    if bundle.alpha == 0.0:
        fallback = pd.Series(False, index=frame.index, dtype=bool)
        origin = pd.Series("C0", index=frame.index, dtype=object)
    else:
        fallback = ~available
        origin = pd.Series(
            np.where(available, "C6_weather", "C0_fallback"),
            index=frame.index,
            dtype=object,
        )
    origin = origin.where(score.notna())
    return {
        "score": score,
        "origin": origin,
        "correction_available": available,
        "fallback": fallback,
        "alpha": float(bundle.alpha),
    }


def replay_v3_model(
    frame: pd.DataFrame,
    model: Any,
    model_code: str,
    policy_record: Mapping[str, Any],
) -> pd.DataFrame:
    """Apply one saved v3 model and exact saved service policy."""

    spec = MODEL_SPECS[model_code]
    score = score_model(
        model,
        frame,
        list(spec["features"]),
        spec["availability"],
    )
    policy = Policy(
        threshold=float(policy_record["threshold"]),
        active_days=int(policy_record["active_days"]),
        cooldown_days=int(policy_record["cooldown_days"]),
        version=f"frozen_v3_{model_code}_{EVALUATION_SCOPE}",
    )
    states = simulate_policy(frame, score, policy, EVALUATION_SCOPE)
    availability = (
        frame[spec["availability"]].astype(bool)
        if spec["availability"] is not None
        else pd.Series(True, index=frame.index, dtype=bool)
    )
    states["score_origin"] = pd.Series(model_code, index=states.index).where(
        states["score"].notna()
    )
    states["score_model_id"] = model_code
    states["score_model_version"] = "cycle1_fixed_train_only_v1"
    states["correction_available"] = availability
    states["fallback_to_c0"] = False
    states["effective_c0_origin"] = False
    states["message_kind"] = np.where(states["message_issued"], "ordinary", "none")
    states["growth_override_used"] = False
    states["growth_reference_status"] = "not_applicable_v3_policy"
    return states


def replay_c6_policy(
    frame: pd.DataFrame,
    bundle: C6Bundle,
    policy_record: Mapping[str, Any],
    family: str,
) -> pd.DataFrame:
    """Apply a saved C6 score and one exact final cycle-3 policy."""

    runtime = _c6_runtime(frame, bundle)
    effective_c0 = float(runtime["alpha"]) == 0.0
    states, _ = simulate_growth_policy(
        frame,
        runtime["score"],
        _growth_policy(policy_record, family),
        evaluation_scope=EVALUATION_SCOPE,
        evaluation_mask=frame["service_active"],
        score_origin=runtime["origin"],
        model_id="C0" if effective_c0 else "C6_weather",
        model_version=(
            "cycle1_fixed_train_only_v1"
            if effective_c0
            else "cycle2_nested_temporal_oof_v1"
        ),
    )
    states["correction_available"] = runtime["correction_available"]
    states["fallback_to_c0"] = runtime["fallback"]
    states["effective_c0_origin"] = states["score_origin"].isin(
        ["C0", "C0_fallback"]
    )
    return states


def _normalise_key_dates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["issue_date"] = pd.to_datetime(result["issue_date"]).dt.tz_localize(None)
    return result


def _series_matches(left: pd.Series, right: pd.Series, tolerance: float) -> bool:
    if not left.isna().eq(right.isna()).all():
        return False
    present = left.notna()
    if not present.any():
        return True
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        return bool(
            np.allclose(
                left[present].astype(float),
                right[present].astype(float),
                atol=tolerance,
                rtol=0.0,
            )
        )
    left_values = left[present].astype(str).reset_index(drop=True)
    right_values = right[present].astype(str).reset_index(drop=True)
    return bool(left_values.equals(right_values))


def replay_identity_audit(
    actual: pd.DataFrame,
    saved: pd.DataFrame,
    *,
    model_code: str,
    fold_id: str,
    columns: Sequence[str] = (
        "score",
        "score_status",
        "message_issued",
        "alarm_active",
        "suppressed_repeat",
        "action_reason",
    ),
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Require the original-weather replay to reproduce a saved artifact."""

    left = _normalise_key_dates(actual)[KEY_COLUMNS + list(columns)]
    right = _normalise_key_dates(saved)[KEY_COLUMNS + list(columns)]
    if left.duplicated(KEY_COLUMNS).any() or right.duplicated(KEY_COLUMNS).any():
        raise AssertionError("Replay identity population contains duplicate day keys")
    paired = left.merge(
        right,
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("_actual", "_saved"),
        validate="one_to_one",
        indicator=True,
    ).sort_values(KEY_COLUMNS)
    same_population = bool(paired["_merge"].eq("both").all())
    matches: dict[str, bool] = {}
    if same_population:
        for column in columns:
            matches[column] = _series_matches(
                paired[f"{column}_actual"],
                paired[f"{column}_saved"],
                tolerance,
            )
    else:
        matches = {column: False for column in columns}
    passed = same_population and all(matches.values())
    audit = {
        "fold_id": fold_id,
        "model_code": model_code,
        "rows_actual": len(left),
        "rows_saved": len(right),
        "same_population": same_population,
        "column_matches": matches,
        "status": "passed" if passed else "failed",
    }
    if not passed:
        raise AssertionError(f"Frozen replay identity failed: {audit}")
    return audit


def _transition(old: pd.Series, new: pd.Series, positive: str) -> pd.Series:
    old_values = old.fillna(False).astype(bool)
    new_values = new.fillna(False).astype(bool)
    return pd.Series(
        np.select(
            [old_values & new_values, ~old_values & new_values, old_values & ~new_values],
            [f"stable_{positive}", f"gained_{positive}", f"lost_{positive}"],
            default=f"stable_no_{positive}",
        ),
        index=old.index,
        dtype=object,
    )


def pair_scenarios(
    old: pd.DataFrame,
    new: pd.DataFrame,
    *,
    keys: Sequence[str],
    value_columns: Sequence[str],
) -> pd.DataFrame:
    """Pair old/new rows exactly and add common replay transitions."""

    keys = list(keys)
    values = list(value_columns)
    if old.duplicated(keys).any() or new.duplicated(keys).any():
        raise AssertionError("Scenario pairing contains duplicate keys")
    paired = old[keys + values].merge(
        new[keys + values],
        on=keys,
        how="outer",
        suffixes=("_old", "_new"),
        validate="one_to_one",
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise AssertionError("Old/new scenario populations differ")
    paired = paired.drop(columns="_merge")
    if "score" in values:
        paired["score_delta"] = paired["score_new"] - paired["score_old"]
        paired["score_became_computable"] = (
            paired["score_old"].isna() & paired["score_new"].notna()
        )
    if "correction_available" in values:
        paired["availability_transition"] = _transition(
            paired["correction_available_old"],
            paired["correction_available_new"],
            "available",
        )
    if "message_issued" in values:
        paired["message_transition"] = _transition(
            paired["message_issued_old"], paired["message_issued_new"], "message"
        )
    if "alarm_active" in values:
        paired["alarm_transition"] = _transition(
            paired["alarm_active_old"], paired["alarm_active_new"], "alarm"
        )
    return paired.sort_values(keys, kind="stable").reset_index(drop=True)


def _feature_identity_audit(rebuilt: pd.DataFrame, saved: pd.DataFrame) -> dict[str, Any]:
    feature_columns = list(
        dict.fromkeys(CALENDAR_FEATURES + NASA_COMMON_FEATURES + ERA_COMMON_FEATURES + EPISODE_FEATURES)
    )
    flag_columns = [
        "nasa_common_complete",
        "era_common_complete",
        "common_weather_complete",
        "episode_weather_complete",
        "candidate_comparison_complete",
        "hutton_score",
        "smith_score",
    ]
    left = _normalise_key_dates(rebuilt)[KEY_COLUMNS + feature_columns + flag_columns]
    right = _normalise_key_dates(saved)[KEY_COLUMNS + feature_columns + flag_columns]
    paired = left.merge(
        right,
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("_rebuilt", "_saved"),
        validate="one_to_one",
        indicator=True,
    ).sort_values(KEY_COLUMNS)
    same_population = bool(paired["_merge"].eq("both").all())
    mismatches: list[str] = []
    maximum = 0.0
    if same_population:
        for column in feature_columns + flag_columns:
            left_values = paired[f"{column}_rebuilt"]
            right_values = paired[f"{column}_saved"]
            if not _series_matches(left_values, right_values, 1e-12):
                mismatches.append(column)
            if pd.api.types.is_numeric_dtype(left_values) and pd.api.types.is_numeric_dtype(
                right_values
            ):
                present = left_values.notna() & right_values.notna()
                if present.any():
                    maximum = max(
                        maximum,
                        float(
                            (left_values[present].astype(float) - right_values[present].astype(float))
                            .abs()
                            .max()
                        ),
                    )
    passed = same_population and not mismatches
    audit = {
        "status": "passed" if passed else "failed",
        "same_daily_population": same_population,
        "rows": len(left),
        "checked_columns": len(feature_columns) + len(flag_columns),
        "mismatched_columns": mismatches,
        "maximum_absolute_numeric_difference": maximum,
    }
    if not passed:
        raise AssertionError(f"Frozen feature reconstruction differs from v3: {audit}")
    return audit


def feature_coverage_audit(
    old: pd.DataFrame,
    new: pd.DataFrame,
    years: Iterable[int] = TEST_YEARS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year_value in [*sorted({int(value) for value in years}), "pooled_2020_2025"]:
        if isinstance(year_value, int):
            old_part = old[old["season"].eq(year_value)]
            new_part = new[new["season"].eq(year_value)]
            period = str(year_value)
        else:
            selected = tuple(sorted({int(value) for value in years}))
            old_part = old[old["season"].isin(selected)]
            new_part = new[new["season"].isin(selected)]
            period = year_value
        service_old = old_part["service_active"].astype(bool)
        service_new = new_part["service_active"].astype(bool)
        record: dict[str, Any] = {
            "period": period,
            "service_field_days_old": int(service_old.sum()),
            "service_field_days_new": int(service_new.sum()),
        }
        for column in (
            "era_common_complete",
            "episode_weather_complete",
            "candidate_comparison_complete",
        ):
            old_count = int((service_old & old_part[column].astype(bool)).sum())
            new_count = int((service_new & new_part[column].astype(bool)).sum())
            record[f"{column}_old"] = old_count
            record[f"{column}_new"] = new_count
            record[f"{column}_delta"] = new_count - old_count
        rows.append(record)
    return pd.DataFrame(rows)


def training_population_audit(
    old: pd.DataFrame,
    new: pd.DataFrame,
    contract: Mapping[str, Any],
    years: Iterable[int] = TEST_YEARS,
) -> pd.DataFrame:
    """Show whether the backfill changes any frozen train/validation/test rows."""

    selected_folds = {_fold_id(int(year)) for year in years}
    rows: list[dict[str, Any]] = []
    for fold in contract["rolling_origin_folds"]:
        fold_id = str(fold["id"])
        if fold_id not in selected_folds:
            continue
        for role in ("train", "validation", "test"):
            bounds = fold[f"{role}_years"]
            old_part = old[old["season"].between(int(bounds[0]), int(bounds[1]))]
            new_part = new[new["season"].between(int(bounds[0]), int(bounds[1]))]

            def eligible(part: pd.DataFrame) -> pd.Series:
                return (
                    part["target_observable"].astype(bool)
                    & part["service_active"].astype(bool)
                    & part["candidate_comparison_complete"].astype(bool)
                    & part["target_class"].isin(
                        ["no_record_in_horizon", "imminent", "actionable"]
                    )
                )

            old_eligible = eligible(old_part)
            new_eligible = eligible(new_part)
            old_keys = set(
                map(tuple, _normalise_key_dates(old_part.loc[old_eligible, KEY_COLUMNS]).to_numpy())
            )
            new_keys = set(
                map(tuple, _normalise_key_dates(new_part.loc[new_eligible, KEY_COLUMNS]).to_numpy())
            )
            rows.append(
                {
                    "fold_id": fold_id,
                    "role": role,
                    "year_start": int(bounds[0]),
                    "year_end": int(bounds[1]),
                    "eligible_rows_old": len(old_keys),
                    "eligible_rows_new": len(new_keys),
                    "eligible_rows_delta": len(new_keys) - len(old_keys),
                    "newly_eligible_rows": len(new_keys - old_keys),
                    "lost_eligible_rows": len(old_keys - new_keys),
                }
            )
    return pd.DataFrame(rows)


def _event_window_audit(old: pd.DataFrame, new: pd.DataFrame) -> dict[str, Any]:
    keys = KEY_COLUMNS
    left = _normalise_key_dates(old)[keys + ["service_active", "days_to_first_recorded_event", "episode_weather_complete"]]
    right = _normalise_key_dates(new)[keys + ["service_active", "days_to_first_recorded_event", "episode_weather_complete"]]
    paired = left.merge(right, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    event_window = (
        paired["service_active_old"].astype(bool)
        & paired["days_to_first_recorded_event_old"].between(3, 10)
    )
    gained = (
        event_window
        & ~paired["episode_weather_complete_old"].astype(bool)
        & paired["episode_weather_complete_new"].astype(bool)
    )
    return {
        "actionable_event_window_rows": int(event_window.sum()),
        "newly_episode_computable_event_window_rows": int(gained.sum()),
    }


def _prediction_view(states: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *KEY_COLUMNS,
        "issued_at",
        "service_active",
        "evaluation_scope_day",
        "score",
        "score_status",
        "score_origin",
        "correction_available",
        "fallback_to_c0",
        "effective_c0_origin",
    ]
    return states[columns].copy()


def _alarm_view(states: pd.DataFrame) -> pd.DataFrame:
    optional = [
        "message_kind",
        "growth_override_used",
        "growth_reference_status",
        "score_comparison_segment_id",
        "logit_growth_from_previous_message",
    ]
    columns = [
        *KEY_COLUMNS,
        "score",
        "score_status",
        "message_issued",
        "alarm_active",
        "active_from",
        "active_through",
        "action_reason",
        "suppressed_repeat",
        "policy_threshold",
        "policy_active_days",
        "policy_cooldown_days",
        "correction_available",
        "fallback_to_c0",
        "effective_c0_origin",
        "score_origin",
        *[column for column in optional if column in states.columns],
    ]
    return states[columns].copy()


def _scenario_metric(
    states: pd.DataFrame,
    seasons: pd.DataFrame,
    model_code: str,
    fold_id: str,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    event, hits = event_metrics(
        states,
        seasons,
        model_code,
        fold_id,
        slice_name="A_plus_B",
        evaluation_scope=EVALUATION_SCOPE,
    )
    burden = burden_metrics(
        states,
        model_code,
        fold_id,
        slice_name="A_plus_B",
        evaluation_scope=EVALUATION_SCOPE,
    )
    evaluated = states[states["evaluation_scope_day"].astype(bool)]
    burden.update(
        {
            "fallback_days": int(evaluated["fallback_to_c0"].astype(bool).sum()),
            "correction_unavailable_days": int(
                (~evaluated["correction_available"].astype(bool)).sum()
            ),
            "growth_messages": int(
                evaluated.get("growth_override_used", pd.Series(False, index=evaluated.index))
                .fillna(False)
                .astype(bool)
                .sum()
            ),
        }
    )
    return event, burden, pd.DataFrame(hits)


def _pair_metric_rows(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    identifiers: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(identifiers)
    shared = sorted(set(old).intersection(new) - {"model_code", "fold_id", "slice", "evaluation_scope"})
    for column in shared:
        result[f"{column}_old"] = old[column]
        result[f"{column}_new"] = new[column]
        left, right = old[column], new[column]
        if (
            isinstance(left, (int, float, np.integer, np.floating))
            and not isinstance(left, (bool, np.bool_))
            and isinstance(right, (int, float, np.integer, np.floating))
            and not isinstance(right, (bool, np.bool_))
        ):
            result[f"{column}_delta"] = float(right) - float(left)
    return result


def _pair_event_hits(
    old: pd.DataFrame,
    new: pd.DataFrame,
    identifiers: Mapping[str, Any],
) -> pd.DataFrame:
    keys = ["field_season", "season"]
    value_columns = [
        "warnable_event",
        "positive_at_entry",
        "computable_in_actionable_window",
        "timely_hit",
        "timely_message_count",
        "timely_best_lead_days",
        "messages_before_event",
        "too_early_messages",
        "late_messages",
    ]
    paired = pair_scenarios(old, new, keys=keys, value_columns=value_columns)
    paired["hit_transition"] = _transition(
        paired["timely_hit_old"], paired["timely_hit_new"], "hit"
    )
    for column, value in identifiers.items():
        paired[column] = value
    return paired


def _source_snapshot(output_dir: Path) -> Path:
    sources = [
        REPO_ROOT / "src/agro_phenology/post_backfill_frozen_replay.py",
        REPO_ROOT / "src/agro_phenology/early_warning_core.py",
        REPO_ROOT / "src/agro_phenology/early_warning_models.py",
        REPO_ROOT / "src/agro_phenology/early_warning_cycle2_models.py",
        REPO_ROOT / "src/agro_phenology/early_warning_cycle3_policy.py",
        REPO_ROOT / "tests/test_post_backfill_frozen_replay.py",
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Source snapshot is incomplete: {missing}")
    path = output_dir / "source_snapshot.json"
    _atomic_json(
        path,
        {
            "format": "post_backfill_frozen_replay_source_snapshot_v1",
            "created_at_utc": _utc_now(),
            "files": [
                {
                    "relative_path": str(source.relative_to(REPO_ROOT)),
                    "sha256": sha256_file(source),
                    "utf8_content": source.read_text(encoding="utf-8"),
                }
                for source in sources
            ],
        },
    )
    return path


def _reproduce_text(
    run_dir: Path,
    *,
    v3_dir: Path,
    v4_dir: Path,
    cycle3_dir: Path,
    frozen_era_path: Path,
    extended_era_path: Path,
    nasa_daily_path: Path,
    nasa_mapping_path: Path,
) -> str:
    arguments = [
        ("--v3-dir", v3_dir),
        ("--v4-dir", v4_dir),
        ("--cycle3-dir", cycle3_dir),
        ("--frozen-era", frozen_era_path),
        ("--extended-era", extended_era_path),
        ("--nasa-daily", nasa_daily_path),
        ("--nasa-mapping", nasa_mapping_path),
    ]
    suffix = " \\\n".join(f"  {name} {value}" for name, value in arguments)
    return f"""# Frozen replay после ERA5 backfill

Проверка этого уже завершённого результата:

```bash
.venv/bin/python -m agro_phenology.post_backfill_frozen_replay check --run-dir {run_dir}
```

Для полного повторного расчёта из корня репозитория задайте новый, ещё не существующий каталог:

```bash
new_run_dir="results/late_blight_early_warning/YYYYMMDD_post_backfill_models_new"
.venv/bin/python -m agro_phenology.post_backfill_frozen_replay run \\
  --run-dir "$new_run_dir" \\
{suffix}
.venv/bin/python -m agro_phenology.post_backfill_frozen_replay check --run-dir "$new_run_dir"
```

Команда не обучает модели и не выбирает новые alpha, пороги или параметры политик.
Результат является post-hoc replay на ранее изученных 2020--2025 годах.
"""


def _critical_input_files(
    *,
    v3_dir: Path,
    v4_dir: Path,
    cycle3_dir: Path,
    backfill_dir: Path,
    frozen_era_path: Path,
    extended_era_path: Path,
    nasa_daily_path: Path,
    nasa_mapping_path: Path,
) -> list[Path]:
    paths = [
        v3_dir / "execution_manifest.json",
        v3_dir / "evaluation_contract.json",
        v3_dir / "field_seasons.parquet",
        v3_dir / "daily_decisions.parquet",
        v3_dir / "policy_selection.csv",
        v3_dir / "predictions.parquet",
        v3_dir / "alarm_states.parquet",
        v4_dir / "execution_manifest.json",
        v4_dir / "policy_selection.csv",
        v4_dir / "alarm_states.parquet",
        cycle3_dir / "execution_manifest.json",
        cycle3_dir / "policy_selection.csv",
        cycle3_dir / "alarm_states.parquet",
        backfill_dir / "execution_manifest.json",
        frozen_era_path,
        extended_era_path,
        nasa_daily_path,
        nasa_mapping_path,
    ]
    for year in TEST_YEARS:
        fold_id = _fold_id(year)
        paths.extend(
            [
                v3_dir / "models" / f"{fold_id}_C3.cbm",
                v3_dir / "models" / f"{fold_id}_C4.cbm",
                v3_dir / "models" / f"{fold_id}_C5.joblib",
                v4_dir
                / "models"
                / fold_id
                / "C6_weather"
                / "validation_selected__service_calendar"
                / "bundle.json",
                v4_dir
                / "models"
                / fold_id
                / "C6_weather"
                / "validation_selected__service_calendar"
                / "c0.joblib",
                v4_dir
                / "models"
                / fold_id
                / "C6_weather"
                / "validation_selected__service_calendar"
                / "correction.cbm",
            ]
        )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen replay inputs are incomplete: {missing}")
    return paths


def _hash_files(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256_file(path) for path in paths}


def check_completed_replay(run_dir: str | Path) -> dict[str, Any]:
    target = output_directory(run_dir)
    marker = target / ".frozen_replay_run.json"
    if not marker.is_file():
        raise FileNotFoundError(marker)
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    if marker_payload.get("format") != RUN_FORMAT:
        raise AssertionError("Unrecognised frozen replay marker")
    return verify_frozen_run(target, expected_format=RUN_FORMAT)


def run_frozen_replay(
    *,
    run_dir: str | Path,
    v3_dir: str | Path = DEFAULT_V3,
    v4_dir: str | Path = DEFAULT_V4,
    cycle3_dir: str | Path = DEFAULT_CYCLE3,
    frozen_era_path: str | Path = DEFAULT_FROZEN_ERA,
    extended_era_path: str | Path = DEFAULT_EXTENDED_ERA,
    nasa_daily_path: str | Path = DEFAULT_NASA_DAILY,
    nasa_mapping_path: str | Path = DEFAULT_NASA_MAPPING,
) -> dict[str, Any]:
    """Run the complete paired 2020--2025 frozen inference replay."""

    run_dir = Path(run_dir).expanduser().resolve()
    output_dir = output_directory(run_dir)
    if (output_dir / "execution_manifest.json").is_file():
        return check_completed_replay(run_dir)["manifest"]

    v3_dir = Path(v3_dir).expanduser().resolve()
    v4_dir = Path(v4_dir).expanduser().resolve()
    cycle3_dir = Path(cycle3_dir).expanduser().resolve()
    frozen_era_path = Path(frozen_era_path).expanduser().resolve()
    extended_era_path = Path(extended_era_path).expanduser().resolve()
    nasa_daily_path = Path(nasa_daily_path).expanduser().resolve()
    nasa_mapping_path = Path(nasa_mapping_path).expanduser().resolve()
    backfill_dir = extended_era_path.parent
    protected = [v3_dir, v4_dir, cycle3_dir, backfill_dir, frozen_era_path.parent]
    if any(output_dir == path or path in output_dir.parents for path in protected):
        raise ValueError("frozen_replay output must be outside every frozen input")

    output_dir = _ensure_output_directory(run_dir)
    started_at = _utc_now()
    parent_before = {
        "v3": verify_frozen_run(v3_dir, EXPECTED_V3_MANIFEST_SHA256),
        "v4": verify_frozen_run(v4_dir, EXPECTED_V4_MANIFEST_SHA256),
        "cycle3": verify_frozen_run(cycle3_dir, EXPECTED_CYCLE3_MANIFEST_SHA256),
        "backfill": verify_frozen_run(
            backfill_dir,
            EXPECTED_BACKFILL_MANIFEST_SHA256,
            expected_format="agro_phenology_era5_hutton_backfill_v1",
        ),
    }
    critical_inputs = _critical_input_files(
        v3_dir=v3_dir,
        v4_dir=v4_dir,
        cycle3_dir=cycle3_dir,
        backfill_dir=backfill_dir,
        frozen_era_path=frozen_era_path,
        extended_era_path=extended_era_path,
        nasa_daily_path=nasa_daily_path,
        nasa_mapping_path=nasa_mapping_path,
    )
    input_hashes_before = _hash_files(critical_inputs)

    contract = json.loads((v3_dir / "evaluation_contract.json").read_text(encoding="utf-8"))
    target_window = contract["daily_target"]["actionable"]
    seasons = pd.read_parquet(v3_dir / "field_seasons.parquet")
    base = build_daily_decisions(
        seasons,
        timezone=str(contract["timezone"]),
        issue_time=str(contract["daily_issue_time"]),
        minimum_lead=int(target_window[0]),
        maximum_lead=int(target_window[1]),
    )
    old_features, old_feature_summary = add_daily_features(
        base,
        nasa_daily_path,
        nasa_mapping_path,
        frozen_era_path,
    )
    new_features, new_feature_summary = add_daily_features(
        base,
        nasa_daily_path,
        nasa_mapping_path,
        extended_era_path,
    )
    saved_features = pd.read_parquet(v3_dir / "daily_decisions.parquet")
    feature_identity = _feature_identity_audit(old_features, saved_features)
    coverage = feature_coverage_audit(old_features, new_features)
    training = training_population_audit(old_features, new_features, contract)
    event_window = _event_window_audit(
        old_features[old_features["season"].isin(TEST_YEARS)],
        new_features[new_features["season"].isin(TEST_YEARS)],
    )

    v3_policy = read_v3_policy_records(v3_dir / "policy_selection.csv")
    c6_configurations = load_frozen_c6_configurations(v4_dir, cycle3_dir)
    saved_v3_states = pd.read_parquet(v3_dir / "alarm_states.parquet")
    saved_v4_states = pd.read_parquet(v4_dir / "alarm_states.parquet")
    saved_cycle3_states = pd.read_parquet(cycle3_dir / "alarm_states.parquet")

    prediction_parts: list[pd.DataFrame] = []
    alarm_parts: list[pd.DataFrame] = []
    event_hit_parts: list[pd.DataFrame] = []
    event_rows: list[dict[str, Any]] = []
    burden_rows: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    states_by_configuration: dict[tuple[str, str], dict[str, list[pd.DataFrame]]] = {}

    for year in TEST_YEARS:
        fold_id = _fold_id(year)
        old_year = old_features[old_features["season"].eq(year)].copy()
        new_year = new_features[new_features["season"].eq(year)].copy()
        season_year = seasons[seasons["season"].eq(year)].copy()

        for model_code in V3_MODEL_CODES:
            model = _load_v3_model(v3_dir, fold_id, model_code)
            record = v3_policy[(fold_id, model_code)]
            old_states = replay_v3_model(old_year, model, model_code, record)
            new_states = replay_v3_model(new_year, model, model_code, record)
            saved = saved_v3_states[
                saved_v3_states["fold_id"].eq(fold_id)
                & saved_v3_states["model_code"].eq(model_code)
                & saved_v3_states["evaluation_scope"].eq(EVALUATION_SCOPE)
            ]
            identities.append(
                replay_identity_audit(
                    old_states,
                    saved,
                    model_code=model_code,
                    fold_id=fold_id,
                )
            )
            configuration = (model_code, "P0_saved")
            states_by_configuration.setdefault(configuration, {"old": [], "new": []})
            states_by_configuration[configuration]["old"].append(old_states)
            states_by_configuration[configuration]["new"].append(new_states)
            identifiers = {
                "model_code": model_code,
                "policy_family": "P0_saved",
                "fold_id": fold_id,
                "evaluation_scope": EVALUATION_SCOPE,
                "alpha": np.nan,
            }
            paired_predictions = pair_scenarios(
                _prediction_view(old_states),
                _prediction_view(new_states),
                keys=KEY_COLUMNS,
                value_columns=[
                    "issued_at",
                    "service_active",
                    "evaluation_scope_day",
                    "score",
                    "score_status",
                    "score_origin",
                    "correction_available",
                    "fallback_to_c0",
                    "effective_c0_origin",
                ],
            )
            paired_alarm = pair_scenarios(
                _alarm_view(old_states),
                _alarm_view(new_states),
                keys=KEY_COLUMNS,
                value_columns=[column for column in _alarm_view(old_states).columns if column not in KEY_COLUMNS],
            )
            for column, value in identifiers.items():
                paired_predictions[column] = value
                paired_alarm[column] = value
            prediction_parts.append(paired_predictions)
            alarm_parts.append(paired_alarm)

            old_event, old_burden, old_hits = _scenario_metric(
                old_states, season_year, model_code, fold_id
            )
            new_event, new_burden, new_hits = _scenario_metric(
                new_states, season_year, model_code, fold_id
            )
            event_rows.append(_pair_metric_rows(old_event, new_event, identifiers))
            burden_rows.append(_pair_metric_rows(old_burden, new_burden, identifiers))
            event_hit_parts.append(_pair_event_hits(old_hits, new_hits, identifiers))

        for family in C6_POLICY_FAMILIES:
            bundle, record = c6_configurations[(fold_id, family)]
            old_states = replay_c6_policy(old_year, bundle, record, family)
            new_states = replay_c6_policy(new_year, bundle, record, family)
            saved = saved_cycle3_states[
                saved_cycle3_states["fold_id"].eq(fold_id)
                & saved_cycle3_states["model_code"].eq(f"C6_weather__{family}")
                & saved_cycle3_states["evaluation_scope"].eq(EVALUATION_SCOPE)
            ]
            identities.append(
                replay_identity_audit(
                    old_states,
                    saved,
                    model_code=f"C6_weather__{family}",
                    fold_id=fold_id,
                )
            )
            if family == "P0_saved":
                saved_v4 = saved_v4_states[
                    saved_v4_states["fold_id"].eq(fold_id)
                    & saved_v4_states["model_code"].eq(
                        "C6_weather__validation_selected"
                    )
                    & saved_v4_states["evaluation_scope"].eq(EVALUATION_SCOPE)
                ]
                identities.append(
                    replay_identity_audit(
                        old_states,
                        saved_v4,
                        model_code="C6_weather__validation_selected",
                        fold_id=fold_id,
                    )
                )
            configuration = ("C6_weather", family)
            states_by_configuration.setdefault(configuration, {"old": [], "new": []})
            states_by_configuration[configuration]["old"].append(old_states)
            states_by_configuration[configuration]["new"].append(new_states)
            identifiers = {
                "model_code": "C6_weather",
                "policy_family": family,
                "fold_id": fold_id,
                "evaluation_scope": EVALUATION_SCOPE,
                "alpha": float(bundle.alpha),
            }
            paired_predictions = pair_scenarios(
                _prediction_view(old_states),
                _prediction_view(new_states),
                keys=KEY_COLUMNS,
                value_columns=[
                    "issued_at",
                    "service_active",
                    "evaluation_scope_day",
                    "score",
                    "score_status",
                    "score_origin",
                    "correction_available",
                    "fallback_to_c0",
                    "effective_c0_origin",
                ],
            )
            old_alarm_view = _alarm_view(old_states)
            new_alarm_view = _alarm_view(new_states)
            paired_alarm = pair_scenarios(
                old_alarm_view,
                new_alarm_view,
                keys=KEY_COLUMNS,
                value_columns=[column for column in old_alarm_view.columns if column not in KEY_COLUMNS],
            )
            for column, value in identifiers.items():
                paired_predictions[column] = value
                paired_alarm[column] = value
            prediction_parts.append(paired_predictions)
            alarm_parts.append(paired_alarm)

            old_event, old_burden, old_hits = _scenario_metric(
                old_states, season_year, "C6_weather", fold_id
            )
            new_event, new_burden, new_hits = _scenario_metric(
                new_states, season_year, "C6_weather", fold_id
            )
            event_rows.append(_pair_metric_rows(old_event, new_event, identifiers))
            burden_rows.append(_pair_metric_rows(old_burden, new_burden, identifiers))
            event_hit_parts.append(_pair_event_hits(old_hits, new_hits, identifiers))

    for (model_code, family), scenarios in sorted(states_by_configuration.items()):
        old_states = pd.concat(scenarios["old"], ignore_index=True)
        new_states = pd.concat(scenarios["new"], ignore_index=True)
        pooled_seasons = seasons[seasons["season"].isin(TEST_YEARS)]
        fold_id = "pooled_2020_2025"
        alpha_values = {
            float(value)
            for value in pd.concat(scenarios["old"])
            .get("alpha", pd.Series(dtype=float))
            .dropna()
            .unique()
        }
        identifiers = {
            "model_code": model_code,
            "policy_family": family,
            "fold_id": fold_id,
            "evaluation_scope": EVALUATION_SCOPE,
            "alpha": next(iter(alpha_values)) if len(alpha_values) == 1 else np.nan,
        }
        old_event, old_burden, _ = _scenario_metric(
            old_states, pooled_seasons, model_code, fold_id
        )
        new_event, new_burden, _ = _scenario_metric(
            new_states, pooled_seasons, model_code, fold_id
        )
        event_rows.append(_pair_metric_rows(old_event, new_event, identifiers))
        burden_rows.append(_pair_metric_rows(old_burden, new_burden, identifiers))

    predictions = pd.concat(prediction_parts, ignore_index=True)
    alarms = pd.concat(alarm_parts, ignore_index=True)
    event_hits_frame = pd.concat(event_hit_parts, ignore_index=True)
    event_metrics_frame = pd.DataFrame(event_rows)
    burden_metrics_frame = pd.DataFrame(burden_rows)

    _atomic_parquet(output_dir / "predictions.parquet", predictions)
    _atomic_parquet(output_dir / "alarm_states.parquet", alarms)
    _atomic_parquet(output_dir / "event_hits.parquet", event_hits_frame)
    _atomic_csv(output_dir / "event_metrics.csv", event_metrics_frame)
    _atomic_csv(output_dir / "burden_metrics.csv", burden_metrics_frame)
    _atomic_csv(output_dir / "feature_coverage.csv", coverage)
    _atomic_csv(output_dir / "training_population_audit.csv", training)

    replay_audit = {
        "format": "post_backfill_frozen_replay_audit_v1",
        "status": "passed",
        "years": list(TEST_YEARS),
        "evaluation_scope": EVALUATION_SCOPE,
        "feature_rebuild_identity": feature_identity,
        "old_feature_summary": old_feature_summary,
        "new_feature_summary": new_feature_summary,
        "event_window": event_window,
        "training_population_changed": bool(
            training[["newly_eligible_rows", "lost_eligible_rows"]].to_numpy().any()
        ),
        "old_replay_identity_checks": identities,
        "frozen_policies": {
            "v3_models": list(V3_MODEL_CODES),
            "c6_score_model": "C6_weather validation_selected service_calendar",
            "c6_policy_families": list(C6_POLICY_FAMILIES),
        },
        "no_fit": True,
        "no_retune": True,
        "new_threshold_selection": False,
        "new_alpha_selection": False,
        "scientific_status": (
            "post_hoc_retrospective_frozen_replay_on_previously_studied_2020_2025; "
            "not_an_untouched_external_validation"
        ),
    }
    _atomic_json(output_dir / "replay_audit.json", replay_audit)
    source_snapshot_path = _source_snapshot(output_dir)
    _atomic_text(
        output_dir / "REPRODUCE.md",
        _reproduce_text(
            run_dir,
            v3_dir=v3_dir,
            v4_dir=v4_dir,
            cycle3_dir=cycle3_dir,
            frozen_era_path=frozen_era_path,
            extended_era_path=extended_era_path,
            nasa_daily_path=nasa_daily_path,
            nasa_mapping_path=nasa_mapping_path,
        ),
    )

    input_hashes_after = _hash_files(critical_inputs)
    if input_hashes_after != input_hashes_before:
        changed = sorted(
            path
            for path in input_hashes_before
            if input_hashes_before[path] != input_hashes_after.get(path)
        )
        raise AssertionError(f"Frozen inputs changed during replay: {changed}")
    parent_after = {
        "v3": verify_frozen_run(v3_dir, EXPECTED_V3_MANIFEST_SHA256),
        "v4": verify_frozen_run(v4_dir, EXPECTED_V4_MANIFEST_SHA256),
        "cycle3": verify_frozen_run(cycle3_dir, EXPECTED_CYCLE3_MANIFEST_SHA256),
        "backfill": verify_frozen_run(
            backfill_dir,
            EXPECTED_BACKFILL_MANIFEST_SHA256,
            expected_format="agro_phenology_era5_hutton_backfill_v1",
        ),
    }
    for name in parent_before:
        if (
            parent_before[name]["execution_manifest_sha256"]
            != parent_after[name]["execution_manifest_sha256"]
        ):
            raise AssertionError(f"Frozen parent manifest changed during replay: {name}")

    output_names = [
        "predictions.parquet",
        "alarm_states.parquet",
        "event_hits.parquet",
        "event_metrics.csv",
        "burden_metrics.csv",
        "feature_coverage.csv",
        "training_population_audit.csv",
        "replay_audit.json",
        source_snapshot_path.name,
        "REPRODUCE.md",
    ]
    manifest = {
        "format": RUN_FORMAT,
        "status": "complete",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "output_subdirectory": OUTPUT_SUBDIR,
        "years": list(TEST_YEARS),
        "evaluation_scope": EVALUATION_SCOPE,
        "mode": "frozen_inference_and_policy_replay_only",
        "no_fit": True,
        "no_retune": True,
        "parent_integrity_before": {
            key: {
                "path": value["path"],
                "execution_manifest_sha256": value["execution_manifest_sha256"],
                "outputs_checked": value["outputs_checked"],
            }
            for key, value in parent_before.items()
        },
        "parent_integrity_after": {
            key: {
                "path": value["path"],
                "execution_manifest_sha256": value["execution_manifest_sha256"],
                "outputs_checked": value["outputs_checked"],
            }
            for key, value in parent_after.items()
        },
        "input_hashes": input_hashes_before,
        "output_hashes": {
            name: {
                "sha256": sha256_file(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
            }
            for name in output_names
        },
        "scientific_status": replay_audit["scientific_status"],
        "privacy": "local_only_field_linked_outputs_do_not_publish",
    }
    _atomic_json(output_dir / "execution_manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run frozen old/new replay")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--v3-dir", type=Path, default=DEFAULT_V3)
    run.add_argument("--v4-dir", type=Path, default=DEFAULT_V4)
    run.add_argument("--cycle3-dir", type=Path, default=DEFAULT_CYCLE3)
    run.add_argument("--frozen-era", type=Path, default=DEFAULT_FROZEN_ERA)
    run.add_argument("--extended-era", type=Path, default=DEFAULT_EXTENDED_ERA)
    run.add_argument("--nasa-daily", type=Path, default=DEFAULT_NASA_DAILY)
    run.add_argument("--nasa-mapping", type=Path, default=DEFAULT_NASA_MAPPING)
    check = subparsers.add_parser("check", help="verify completed frozen replay")
    check.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "check":
        result = check_completed_replay(args.run_dir)
        print(
            f"Frozen replay verified: {result['outputs_checked']} output files.",
            flush=True,
        )
        return 0
    manifest = run_frozen_replay(
        run_dir=args.run_dir,
        v3_dir=args.v3_dir,
        v4_dir=args.v4_dir,
        cycle3_dir=args.cycle3_dir,
        frozen_era_path=args.frozen_era,
        extended_era_path=args.extended_era,
        nasa_daily_path=args.nasa_daily,
        nasa_mapping_path=args.nasa_mapping,
    )
    print(
        f"Frozen replay complete: {output_directory(args.run_dir)} "
        f"({len(manifest['output_hashes'])} output files).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
