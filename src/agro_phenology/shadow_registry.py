"""Frozen model and policy registry for the late-blight shadow contour.

The registry is assembled only from the immutable cycle-1/2/3 outputs.  It
does not fit a model, select a policy, or inspect an external-test outcome.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .early_warning_core import CALENDAR_FEATURES, EPISODE_FEATURES


PARENT_RUNS = {
    "cycle1": {
        "run_id": "20260910_first_cycle_v3",
        "manifest_sha256": "c7976317e6d94b8d1ca55b0bd618226c769914280bd7896ca36b5e0ca2cf3c80",
    },
    "cycle2": {
        "run_id": "20260910_second_cycle_v4",
        "manifest_sha256": "1355bc9742f397a893774ed02edd7b4080364714d49f208ed75b190d8fd856c0",
    },
    "cycle3": {
        "run_id": "20260910_third_cycle_v2",
        "manifest_sha256": "38f56c053d07fd390ced397bfd9e3566ba441581dc4feb4c8029fdad2920a833",
    },
}

DEPLOYMENT_FOLD = "test_2026_partial"
DEPLOYMENT_SCOPE = "service_calendar"
CLASS_ORDER = [0, 1, 2]
CLASS_NAMES = ["no_record_in_horizon", "imminent", "actionable"]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(raw)


def _relative(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _parent_dir(project_root: Path, cycle: str) -> Path:
    return (
        project_root
        / "results"
        / "late_blight_early_warning"
        / PARENT_RUNS[cycle]["run_id"]
    )


def verify_parent_run(project_root: str | Path, cycle: str) -> dict[str, Any]:
    """Verify a parent manifest and every output recorded by it."""
    root = Path(project_root).resolve()
    run_dir = _parent_dir(root, cycle)
    manifest_path = run_dir / "execution_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    actual_manifest_hash = sha256_file(manifest_path)
    expected_manifest_hash = PARENT_RUNS[cycle]["manifest_sha256"]
    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError(
            f"{cycle} manifest hash mismatch: {actual_manifest_hash} != "
            f"{expected_manifest_hash}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[dict[str, str]] = []
    for relative_path, metadata in manifest.get("output_hashes", {}).items():
        path = run_dir / relative_path
        expected = str(metadata["sha256"])
        if not path.is_file():
            failures.append({"path": relative_path, "reason": "missing"})
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append(
                {
                    "path": relative_path,
                    "reason": "sha256_mismatch",
                    "expected": expected,
                    "actual": actual,
                }
            )
    if failures:
        raise ValueError(f"{cycle} parent integrity failed: {failures[:3]}")
    return {
        "cycle": cycle,
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "manifest_path": _relative(root, manifest_path),
        "manifest_sha256": actual_manifest_hash,
        "completed_at_utc": manifest.get("completed_at_utc"),
        "outputs_checked": len(manifest.get("output_hashes", {})),
        "all_outputs_match": True,
    }


def _artifact(project_root: Path, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "relative_path": _relative(project_root, path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _bundle_artifacts(project_root: Path, directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    artifacts = [_artifact(project_root, path) for path in sorted(directory.iterdir()) if path.is_file()]
    if not any(item["relative_path"].endswith("bundle.json") for item in artifacts):
        raise ValueError(f"No bundle.json in {directory}")
    return artifacts


def _normalise_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def select_latest_compatible_fold(
    project_root: str | Path,
    *,
    policy_selection: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Select by saved chronology and compatibility, without test outcomes.

    The pinned parent manifests are the complete candidate universe.  The sort
    key uses only validation and training year ends.  No predictions, event
    metrics, test-year scores, or external-test result tables are opened.
    """
    root = Path(project_root).resolve()
    manifests = {
        cycle: json.loads(
            (_parent_dir(root, cycle) / "execution_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        for cycle in PARENT_RUNS
    }
    reference_folds = manifests["cycle1"].get("fold_definitions", [])
    for cycle in ("cycle2", "cycle3"):
        if manifests[cycle].get("fold_definitions", []) != reference_folds:
            raise ValueError(f"{cycle} fold definitions differ from frozen cycle1")

    if policy_selection is None:
        policy_selection = pd.read_csv(
            _parent_dir(root, "cycle3") / "policy_selection.csv"
        )
    required_policies = {
        "calendar_window": {"P0_saved", "P_growth_selected"},
        "C0": {"P0_saved", "P_growth_selected"},
        "C1": {"P0_saved", "P_growth_selected"},
        "C4": {"P0_saved", "P_growth_selected"},
        "C5": {"P0_saved", "P_growth_selected"},
        "C6_weather": {"P0_saved", "P_growth_selected"},
        "C6_calibration_control": {"P_growth_selected"},
    }
    suffixes = {"C0": ".joblib", "C1": ".cbm", "C4": ".cbm", "C5": ".joblib"}
    candidates: list[dict[str, Any]] = []
    for definition in reference_folds:
        fold_id = str(definition["id"])
        reasons: list[str] = []
        for model_id, suffix in suffixes.items():
            path = (
                _parent_dir(root, "cycle1")
                / "models"
                / f"{fold_id}_{model_id}{suffix}"
            )
            if not path.is_file():
                reasons.append(f"missing_model:{model_id}")
        for model_id in ("C6_weather", "C6_calibration_control"):
            directory = (
                _parent_dir(root, "cycle2")
                / "models"
                / fold_id
                / model_id
                / "validation_selected__service_calendar"
            )
            bundle_path = directory / "bundle.json"
            if not bundle_path.is_file():
                reasons.append(f"missing_bundle:{model_id}")
                continue
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            if bundle.get("metadata", {}).get("fold_id") != fold_id:
                reasons.append(f"bundle_fold_mismatch:{model_id}")
            if bundle.get("class_order") != CLASS_ORDER or bundle.get("class_names") != CLASS_NAMES:
                reasons.append(f"bundle_class_schema_mismatch:{model_id}")
            for filename_key in ("base_file", "correction_file"):
                if not (directory / str(bundle.get(filename_key, ""))).is_file():
                    reasons.append(f"missing_bundle_artifact:{model_id}:{filename_key}")
        for model_id, families in required_policies.items():
            rows = policy_selection.loc[
                policy_selection["fold_id"].eq(fold_id)
                & policy_selection["evaluation_scope"].eq(DEPLOYMENT_SCOPE)
                & policy_selection["score_model"].eq(model_id)
                & policy_selection["candidate_family"].isin(families)
            ]
            found = set(rows["candidate_family"].astype(str))
            for missing in sorted(families.difference(found)):
                reasons.append(f"missing_policy:{model_id}:{missing}")
            if len(rows) != len(families):
                reasons.append(f"duplicate_policy:{model_id}")
            if len(rows) and not rows["selection_population"].eq("outer_validation_only").all():
                reasons.append(f"non_validation_selection:{model_id}")
        train_years = list(definition["train_years"])
        validation_years = list(definition["validation_years"])
        candidates.append(
            {
                "fold_id": fold_id,
                "train_years": train_years,
                "validation_years": validation_years,
                "compatible": not reasons,
                "incompatibility_reasons": reasons,
                "chronology_key": [int(validation_years[1]), int(train_years[1])],
            }
        )
    compatible = [item for item in candidates if item["compatible"]]
    if not compatible:
        raise ValueError("No chronologically compatible saved deployment fold")
    selected = max(
        compatible,
        key=lambda item: (
            item["chronology_key"][0],
            item["chronology_key"][1],
            item["fold_id"],
        ),
    )
    return {
        "selected_fold": selected["fold_id"],
        "train_years": selected["train_years"],
        "validation_years": selected["validation_years"],
        "sort_fields": ["validation_year_end", "train_year_end", "fold_id"],
        "outcome_or_external_metric_files_read": [],
        "external_test_performance_used": False,
        "candidates": candidates,
    }


def _selected_policy(
    cycle3_selection: pd.DataFrame,
    model_code: str,
    family: str,
    *,
    fold_id: str,
) -> dict[str, Any]:
    rows = cycle3_selection.loc[
        cycle3_selection["fold_id"].eq(fold_id)
        & cycle3_selection["evaluation_scope"].eq(DEPLOYMENT_SCOPE)
        & cycle3_selection["score_model"].eq(model_code)
        & cycle3_selection["candidate_family"].eq(family)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one saved policy for {model_code}/{family}, got {len(rows)}"
        )
    row = rows.iloc[0]
    if str(row["selection_population"]) != "outer_validation_only":
        raise ValueError(
            f"Policy {model_code}/{family} was not selected on outer validation only"
        )
    payload = {
        "policy_id": f"{model_code}__{family}",
        "family": family,
        "threshold": float(row["threshold"]),
        "active_days": int(row["active_days"]),
        "cooldown_days": int(row["cooldown_days"]),
        "minimum_repeat_interval_days": int(row["minimum_repeat_interval_days"]),
        "growth_override_enabled": bool(row["growth_override_enabled"]),
        "growth_delta_id": str(row["growth_delta_id"]),
        "growth_logit_delta": _normalise_scalar(row["growth_logit_delta"]),
        "logit_epsilon": 1e-6,
        "validation_selection_status": str(row["selection_status"]),
        "selection_population": str(row["selection_population"]),
        "source_fold": fold_id,
        "source_scope": DEPLOYMENT_SCOPE,
        "source_alpha": _normalise_scalar(row["alpha"]),
        "research_budget": {
            "messages_per_30_field_days_max": 2.0,
            "active_alarm_fraction_max": 0.5,
            "kind": "soft_validation_selection_constraint_not_operational_quota",
        },
    }
    payload["configuration_sha256"] = canonical_sha256(payload)
    return payload


def _model_entry(
    *,
    project_root: Path,
    model_id: str,
    role: str,
    kind: str,
    features: Iterable[str],
    artifacts: list[dict[str, Any]],
    policies: list[dict[str, Any]],
    fold_definition: dict[str, Any],
    alpha: float | None = None,
    availability: str = "calendar_only",
    status: str,
    notes: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_years = list(fold_definition["train_years"])
    validation_years = list(fold_definition["validation_years"])
    entry: dict[str, Any] = {
        "model_id": model_id,
        "role": role,
        "kind": kind,
        "class_order": CLASS_ORDER,
        "class_names": CLASS_NAMES,
        "actionable_class": 2,
        "features_in_order": list(features),
        "feature_pipeline": {
            "version": "early_warning_core_cycle1_v3",
            "artifact": _artifact(
                project_root, project_root / "src/agro_phenology/early_warning_core.py"
            ),
            "weather_cutoff_days_before_issue": 2,
            "timezone": "Europe/Riga",
        },
        "training": {
            "fold_id": str(fold_definition["id"]),
            "train_years": train_years,
            "validation_years": validation_years,
            "training_cutoff": f"{int(train_years[1])}-12-31",
            "selection_cutoff": f"{int(validation_years[1])}-12-31",
            "chosen_by": "latest_chronological_compatible_saved_fold_before_shadow_lock",
            "chosen_by_external_test_performance": False,
            "post_selection_refit": False,
        },
        "artifacts": artifacts,
        "artifact_set_sha256": canonical_sha256(artifacts),
        "policies": policies,
        "alpha": alpha,
        "input_availability": availability,
        "operational_status": status,
        "notes": notes or [],
    }
    if extra:
        entry.update(extra)
    entry["registry_entry_sha256"] = canonical_sha256(entry)
    return entry


def build_shadow_registry(
    project_root: str | Path,
    *,
    locked_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the deployment registry from the latest compatible frozen fold."""
    root = Path(project_root).resolve()
    if locked_at_utc is None:
        locked_at_utc = datetime.now(timezone.utc).isoformat()
    lock = datetime.fromisoformat(locked_at_utc.replace("Z", "+00:00"))
    if lock.tzinfo is None or lock.utcoffset() is None:
        raise ValueError("locked_at_utc must be timezone-aware")
    lock = lock.astimezone(timezone.utc)

    integrity = [verify_parent_run(root, cycle) for cycle in PARENT_RUNS]
    for record in integrity:
        completed = datetime.fromisoformat(
            str(record["completed_at_utc"]).replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        if completed > lock:
            raise ValueError(
                f"Parent {record['run_id']} completed after registry lock time"
            )

    v3 = _parent_dir(root, "cycle1")
    v4 = _parent_dir(root, "cycle2")
    v5 = _parent_dir(root, "cycle3")
    selection_path = v5 / "policy_selection.csv"
    selection = pd.read_csv(selection_path)
    policy_evidence = _artifact(root, selection_path)
    fold_selection = select_latest_compatible_fold(
        root, policy_selection=selection
    )
    selected_fold = str(fold_selection["selected_fold"])
    if selected_fold != DEPLOYMENT_FOLD:
        raise ValueError(
            f"Pinned deployment fold {DEPLOYMENT_FOLD} no longer matches chronological selection {selected_fold}"
        )
    cycle1_manifest = json.loads(
        (v3 / "execution_manifest.json").read_text(encoding="utf-8")
    )
    fold_definition = next(
        item
        for item in cycle1_manifest["fold_definitions"]
        if item["id"] == selected_fold
    )

    direct_specs = {
        "C0": ("primary", "sklearn_multinomial_logistic", CALENDAR_FEATURES, "ready_calendar_input"),
        "C1": ("diagnostic", "catboost_multiclass", CALENDAR_FEATURES, "ready_calendar_input"),
        "C4": (
            "diagnostic",
            "catboost_multiclass",
            list(CALENDAR_FEATURES) + list(EPISODE_FEATURES),
            "blocked_exact_era5_t_minus_2",
        ),
        "C5": (
            "diagnostic",
            "sklearn_multinomial_logistic",
            list(CALENDAR_FEATURES) + list(EPISODE_FEATURES),
            "blocked_exact_era5_t_minus_2",
        ),
    }
    models: list[dict[str, Any]] = []

    seasons_path = v3 / "field_seasons.parquet"
    seasons = pd.read_parquet(seasons_path)
    train = seasons.loc[
        seasons["season"].between(*fold_definition["train_years"])
        & seasons["warnable_first_event"].eq(True)
        & seasons["first_recorded_event_date"].notna()
    ]
    doy = pd.to_datetime(train["first_recorded_event_date"]).dt.dayofyear
    lower, upper = int(doy.quantile(0.05)), int(doy.quantile(0.95))
    models.append(
        _model_entry(
            project_root=root,
            model_id="calendar_window",
            role="primary",
            kind="deterministic_day_of_year_rule",
            features=["issue_date_local_day_of_year"],
            artifacts=[_artifact(root, seasons_path)],
            policies=[
                _selected_policy(
                    selection, "calendar_window", "P0_saved", fold_id=selected_fold
                ),
                _selected_policy(
                    selection,
                    "calendar_window",
                    "P_growth_selected",
                    fold_id=selected_fold,
                ),
            ],
            fold_definition=fold_definition,
            availability="calendar_only",
            status="ready_calendar_input",
            notes=[
                "The growth-selected policy is disabled for this frozen fold; it is still retained verbatim.",
            ],
            extra={
                "rule": {
                    "lower_day_of_year_inclusive": lower,
                    "upper_day_of_year_inclusive": upper,
                    "quantiles": [0.05, 0.95],
                    "training_warnable_events": int(len(train)),
                }
            },
        )
    )

    suffixes = {"C0": ".joblib", "C1": ".cbm", "C4": ".cbm", "C5": ".joblib"}
    for model_id, (role, kind, features, status) in direct_specs.items():
        model_path = v3 / "models" / f"{selected_fold}_{model_id}{suffixes[model_id]}"
        availability = "calendar_only" if model_id in {"C0", "C1"} else "exact_era5_episode_complete_or_abstain"
        models.append(
            _model_entry(
                project_root=root,
                model_id=model_id,
                role=role,
                kind=kind,
                features=features,
                artifacts=[_artifact(root, model_path)],
                policies=[
                    _selected_policy(
                        selection, model_id, "P0_saved", fold_id=selected_fold
                    ),
                    _selected_policy(
                        selection,
                        model_id,
                        "P_growth_selected",
                        fold_id=selected_fold,
                    ),
                ],
                fold_definition=fold_definition,
                availability=availability,
                status=status,
                notes=["No C0 fallback is allowed for this diagnostic model."]
                if model_id in {"C4", "C5"}
                else [],
            )
        )

    for model_id, role, status in [
        ("C6_weather", "primary", "weather_blocked_c0_fallback_ready"),
        ("C6_calibration_control", "primary_control", "ready_calendar_input"),
    ]:
        bundle_dir = (
            v4
            / "models"
            / selected_fold
            / model_id
            / "validation_selected__service_calendar"
        )
        bundle_payload = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
        if bundle_payload.get("class_order") != CLASS_ORDER:
            raise ValueError(f"Unexpected class order in {model_id} bundle")
        if bundle_payload.get("class_names") != CLASS_NAMES:
            raise ValueError(f"Unexpected class names in {model_id} bundle")
        if list(bundle_payload.get("base_features", [])) != list(CALENDAR_FEATURES):
            raise ValueError(f"Unexpected base feature order in {model_id} bundle")
        expected_correction = list(EPISODE_FEATURES) if model_id == "C6_weather" else []
        if list(bundle_payload.get("correction_features", [])) != expected_correction:
            raise ValueError(f"Unexpected correction feature order in {model_id} bundle")
        features = list(bundle_payload["base_features"]) + list(
            bundle_payload["correction_features"]
        )
        c6_policies = [
            _selected_policy(
                selection, model_id, "P_growth_selected", fold_id=selected_fold
            )
        ]
        if model_id == "C6_weather":
            c6_policies.insert(
                0,
                _selected_policy(
                    selection, model_id, "P0_saved", fold_id=selected_fold
                ),
            )
        bundle_alpha = float(bundle_payload["alpha"])
        if any(policy["source_alpha"] != bundle_alpha for policy in c6_policies):
            raise ValueError(f"Policy alpha does not match {model_id} bundle alpha")
        models.append(
            _model_entry(
                project_root=root,
                model_id=model_id,
                role=role,
                kind=f"c6_{bundle_payload['correction_kind']}",
                features=features,
                artifacts=_bundle_artifacts(root, bundle_dir),
                policies=c6_policies,
                fold_definition=fold_definition,
                alpha=bundle_alpha,
                availability=(
                    "exact_era5_episode_complete_else_exact_c0_fallback_single_history"
                    if model_id == "C6_weather"
                    else "calendar_only_alpha_zero_effective_c0"
                ),
                status=status,
                notes=(
                    [
                        "Weather and C0 fallback share one policy history.",
                        "A source transition breaks growth-score comparability without resetting cooldown.",
                    ]
                    if model_id == "C6_weather"
                    else [
                        "Frozen alpha is zero, so the effective score origin is C0 on every decision."
                    ]
                ),
                extra={
                    "bundle_format_version": int(bundle_payload["format_version"]),
                    "correction_kind": bundle_payload["correction_kind"],
                    "availability_column": bundle_payload.get("availability_column"),
                    "alpha_zero_exact_c0_recovery": model_id == "C6_calibration_control",
                },
            )
        )

    runtime_sources = [
        _artifact(root, root / "src/agro_phenology/early_warning_core.py"),
        _artifact(root, root / "src/agro_phenology/early_warning_models.py"),
        _artifact(root, root / "src/agro_phenology/early_warning_cycle2_models.py"),
        _artifact(root, root / "src/agro_phenology/early_warning_cycle3_policy.py"),
        _artifact(root, root / "src/agro_phenology/shadow_sources.py"),
        _artifact(root, root / "src/agro_phenology/shadow_storage.py"),
        _artifact(root, root / "src/agro_phenology/shadow_engine.py"),
        _artifact(root, root / "src/agro_phenology/shadow_cli.py"),
        _artifact(root, root / "src/agro_phenology/shadow_registry.py"),
    ]
    registry = {
        "schema_version": "1.0.0",
        "registry_id": "potato_late_blight_shadow_20260910_v1",
        "locked_at_utc": lock.isoformat(),
        "mode": "prospective_shadow_no_delivery",
        "research_target": "first_recorded_potato_late_blight_in_field_season",
        "timeliness_window_days": [3, 10],
        "decision_timezone": "Europe/Riga",
        "scheduled_local_time": "08:00:00",
        "delivery_mode": "shadow",
        "actually_sent": False,
        "selection_rule": {
            "description": "latest chronological compatible saved outer fold completed before registry lock",
            "selected_fold": selected_fold,
            "train_years": list(fold_definition["train_years"]),
            "validation_years": list(fold_definition["validation_years"]),
            "sort_fields": fold_selection["sort_fields"],
            "compatible_candidates": [
                item["fold_id"]
                for item in fold_selection["candidates"]
                if item["compatible"]
            ],
            "outcome_or_external_metric_files_read": fold_selection[
                "outcome_or_external_metric_files_read"
            ],
            "external_2026_results_used_for_selection": False,
            "scientific_superiority_claim": False,
        },
        "bundle_history": {
            "existing_unified_deployment_bundle_before_readiness": False,
            "registry_resolves_missing_bundle": True,
            "composition": "cycle1 direct models plus cycle2 validation-selected C6 bundles plus cycle3 validation-selected policies",
            "automatic_refit_or_retuning": False,
        },
        "class_mapping": dict(zip(CLASS_NAMES, CLASS_ORDER)),
        "parent_integrity": integrity,
        "policy_selection_evidence": policy_evidence,
        "runtime_sources": runtime_sources,
        "runtime_source_set_sha256": canonical_sha256(runtime_sources),
        "models": models,
        "operational_constraints": {
            "actual_active_field_registry_required": True,
            "historical_coordinates_must_not_be_promoted_to_active_fields": True,
            "weather_cutoff_days_before_issue": 2,
            "late_data_never_revises_prior_decisions": True,
            "future_forecasts_are_archive_only": True,
            "no_notifications": True,
        },
    }
    registry["registry_content_sha256"] = canonical_sha256(registry)
    return registry


def write_shadow_registry(
    destination: str | Path,
    project_root: str | Path,
    *,
    locked_at_utc: str | None = None,
) -> Path:
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite registry: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    registry = build_shadow_registry(project_root, locked_at_utc=locked_at_utc)
    target.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def verify_shadow_registry(
    registry_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = Path(registry_path)
    registry = json.loads(path.read_text(encoding="utf-8"))
    registry_without_hash = dict(registry)
    expected_registry_hash = registry_without_hash.pop("registry_content_sha256", None)
    actual_registry_hash = canonical_sha256(registry_without_hash)
    if expected_registry_hash != actual_registry_hash:
        raise ValueError("shadow registry content hash mismatch")
    selection_rule = registry["selection_rule"]
    if selection_rule.get("external_2026_results_used_for_selection") is not False:
        raise ValueError("registry permits external test performance selection")
    if selection_rule.get("outcome_or_external_metric_files_read") != []:
        raise ValueError("registry fold selection read outcome or external metric files")

    current_parent_integrity = {
        item["cycle"]: item for item in (verify_parent_run(root, cycle) for cycle in PARENT_RUNS)
    }
    for recorded in registry["parent_integrity"]:
        current = current_parent_integrity.get(recorded["cycle"])
        if current is None or current["manifest_sha256"] != recorded["manifest_sha256"]:
            raise ValueError(f"parent integrity mismatch for {recorded['cycle']}")

    def verify_artifact(artifact: dict[str, Any]) -> None:
        nonlocal artifact_count
        artifact_count += 1
        source = root / artifact["relative_path"]
        if not source.is_file():
            raise ValueError(f"missing artifact: {artifact['relative_path']}")
        if source.stat().st_size != int(artifact["bytes"]):
            raise ValueError(f"artifact byte size mismatch: {artifact['relative_path']}")
        if sha256_file(source) != artifact["sha256"]:
            raise ValueError(f"artifact hash mismatch: {artifact['relative_path']}")

    artifact_count = 0
    verify_artifact(registry["policy_selection_evidence"])
    if canonical_sha256(registry["runtime_sources"]) != registry["runtime_source_set_sha256"]:
        raise ValueError("runtime source artifact-set hash mismatch")
    for artifact in registry["runtime_sources"]:
        verify_artifact(artifact)
    for model in registry["models"]:
        entry = dict(model)
        expected_entry_hash = entry.pop("registry_entry_sha256")
        if canonical_sha256(entry) != expected_entry_hash:
            raise ValueError(f"registry entry hash mismatch for {model['model_id']}")
        if canonical_sha256(model["artifacts"]) != model["artifact_set_sha256"]:
            raise ValueError(f"artifact-set hash mismatch for {model['model_id']}")
        verify_artifact(model["feature_pipeline"]["artifact"])
        for artifact in model["artifacts"]:
            verify_artifact(artifact)
        for policy in model["policies"]:
            payload = dict(policy)
            expected_policy_hash = payload.pop("configuration_sha256")
            if canonical_sha256(payload) != expected_policy_hash:
                raise ValueError(
                    f"policy configuration hash mismatch: {policy['policy_id']}"
                )
    by_id = {model["model_id"]: model for model in registry["models"]}
    calibration = by_id["C6_calibration_control"]
    if calibration["alpha"] != 0.0 or not calibration["alpha_zero_exact_c0_recovery"]:
        raise ValueError("calibration control must be the frozen alpha-zero C0 control")
    weather = by_id["C6_weather"]
    if weather["alpha"] != 0.1:
        raise ValueError("unexpected frozen C6_weather alpha")
    return {
        "registry_path": str(path.resolve()),
        "registry_sha256": sha256_file(path),
        "registry_content_sha256": actual_registry_hash,
        "model_entries": len(registry["models"]),
        "artifacts_checked": artifact_count,
        "status": "passed",
    }


__all__ = [
    "CLASS_NAMES",
    "CLASS_ORDER",
    "DEPLOYMENT_FOLD",
    "DEPLOYMENT_SCOPE",
    "PARENT_RUNS",
    "build_shadow_registry",
    "canonical_sha256",
    "sha256_file",
    "select_latest_compatible_fold",
    "verify_parent_run",
    "verify_shadow_registry",
    "write_shadow_registry",
]
