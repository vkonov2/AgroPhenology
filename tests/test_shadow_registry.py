from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from agro_phenology.early_warning_core import CALENDAR_FEATURES, EPISODE_FEATURES
from agro_phenology.shadow_registry import (
    DEPLOYMENT_FOLD,
    PARENT_RUNS,
    build_shadow_registry,
    canonical_sha256,
    select_latest_compatible_fold,
    sha256_file,
    verify_parent_run,
    verify_shadow_registry,
    write_shadow_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCKED_AT = "2026-09-10T16:00:00Z"


@pytest.fixture(scope="module")
def registry() -> dict:
    return build_shadow_registry(PROJECT_ROOT, locked_at_utc=LOCKED_AT)


def _models(registry: dict) -> dict[str, dict]:
    return {item["model_id"]: item for item in registry["models"]}


def _policy(model: dict, family: str) -> dict:
    return next(item for item in model["policies"] if item["family"] == family)


def test_pinned_parent_manifests_and_all_recorded_outputs_match() -> None:
    expected_counts = {"cycle1": 87, "cycle2": 307, "cycle3": 43}
    for cycle, expected in PARENT_RUNS.items():
        check = verify_parent_run(PROJECT_ROOT, cycle)
        assert check["run_id"] == expected["run_id"]
        assert check["manifest_sha256"] == expected["manifest_sha256"]
        assert check["status"] == "complete"
        assert check["outputs_checked"] == expected_counts[cycle]
        assert check["all_outputs_match"] is True


def test_latest_compatible_fold_uses_chronology_without_external_results() -> None:
    selection = select_latest_compatible_fold(PROJECT_ROOT)
    assert selection["selected_fold"] == DEPLOYMENT_FOLD == "test_2026_partial"
    assert selection["train_years"] == [2010, 2023]
    assert selection["validation_years"] == [2024, 2025]
    assert selection["sort_fields"] == [
        "validation_year_end",
        "train_year_end",
        "fold_id",
    ]
    assert selection["outcome_or_external_metric_files_read"] == []
    assert selection["external_test_performance_used"] is False
    assert all(item["compatible"] for item in selection["candidates"])
    chronology = [item["chronology_key"] for item in selection["candidates"]]
    assert selection["candidates"][-1]["fold_id"] == DEPLOYMENT_FOLD
    assert selection["candidates"][-1]["chronology_key"] == max(chronology)


def test_registry_records_selection_history_and_frozen_target(registry: dict) -> None:
    assert registry["timeliness_window_days"] == [3, 10]
    assert registry["research_target"] == (
        "first_recorded_potato_late_blight_in_field_season"
    )
    assert registry["delivery_mode"] == "shadow"
    assert registry["actually_sent"] is False
    rule = registry["selection_rule"]
    assert rule["selected_fold"] == DEPLOYMENT_FOLD
    assert rule["external_2026_results_used_for_selection"] is False
    assert rule["outcome_or_external_metric_files_read"] == []
    assert rule["scientific_superiority_claim"] is False
    assert registry["bundle_history"] == {
        "existing_unified_deployment_bundle_before_readiness": False,
        "registry_resolves_missing_bundle": True,
        "composition": "cycle1 direct models plus cycle2 validation-selected C6 bundles plus cycle3 validation-selected policies",
        "automatic_refit_or_retuning": False,
    }
    assert registry["operational_constraints"]["no_notifications"] is True


def test_registry_model_roles_features_and_saved_policies(registry: dict) -> None:
    models = _models(registry)
    assert set(models) == {
        "calendar_window",
        "C0",
        "C1",
        "C4",
        "C5",
        "C6_weather",
        "C6_calibration_control",
    }
    assert models["calendar_window"]["role"] == "primary"
    assert models["C0"]["role"] == "primary"
    assert models["C6_weather"]["role"] == "primary"
    assert models["C6_calibration_control"]["role"] == "primary_control"
    assert all(models[key]["role"] == "diagnostic" for key in ("C1", "C4", "C5"))
    assert models["C0"]["features_in_order"] == CALENDAR_FEATURES
    assert models["C6_weather"]["features_in_order"] == (
        CALENDAR_FEATURES + EPISODE_FEATURES
    )
    assert models["C6_calibration_control"]["features_in_order"] == CALENDAR_FEATURES

    assert _policy(models["calendar_window"], "P0_saved")["threshold"] == 0.5
    calendar_growth = _policy(models["calendar_window"], "P_growth_selected")
    assert calendar_growth["growth_override_enabled"] is False
    assert calendar_growth["growth_delta_id"] == "disabled"
    assert _policy(models["C0"], "P0_saved")["threshold"] == 0.2
    c0_growth = _policy(models["C0"], "P_growth_selected")
    assert c0_growth["threshold"] == 0.15
    assert c0_growth["growth_logit_delta"] == pytest.approx(np.log(1.5))


def test_c6_uses_validation_selected_service_bundle_and_frozen_alpha(registry: dict) -> None:
    models = _models(registry)
    weather = models["C6_weather"]
    paths = {item["relative_path"]: item["sha256"] for item in weather["artifacts"]}
    assert all("validation_selected__service_calendar" in path for path in paths)
    assert not any("c0_policy_replay" in path for path in paths)
    assert next(value for path, value in paths.items() if path.endswith("bundle.json")) == (
        "36b879fb94e5b2857f8bb3286d10f4c35d6dc8e19de43c3c2b7d2957973eef8e"
    )
    assert next(value for path, value in paths.items() if path.endswith("c0.joblib")) == (
        "4c73c09bde11ac68187f3164d02d0b831b9de0b6dd70b696d0116a43d2231161"
    )
    assert next(value for path, value in paths.items() if path.endswith("correction.cbm")) == (
        "9d4dc2d9e2adf4d4575f4b14c29390ba65ab74f6ac74c692bcc86199fb1b6344"
    )
    assert weather["alpha"] == 0.1
    assert weather["availability_column"] == "episode_weather_complete"
    p0 = _policy(weather, "P0_saved")
    growth = _policy(weather, "P_growth_selected")
    assert p0["threshold"] == pytest.approx(0.2117858007450122)
    assert growth["threshold"] == pytest.approx(0.1500045112800455)
    assert growth["growth_logit_delta"] == pytest.approx(np.log(1.5))
    assert p0["source_alpha"] == growth["source_alpha"] == weather["alpha"]


def test_calibration_control_is_frozen_alpha_zero_and_exact_c0(registry: dict) -> None:
    models = _models(registry)
    control = models["C6_calibration_control"]
    assert control["alpha"] == 0.0
    assert control["alpha_zero_exact_c0_recovery"] is True
    assert control["correction_kind"] == "calibration"
    assert control["availability_column"] is None
    assert _policy(control, "P_growth_selected")["source_alpha"] == 0.0
    paths = {item["relative_path"]: item["sha256"] for item in control["artifacts"]}
    assert next(value for path, value in paths.items() if path.endswith("bundle.json")) == (
        "42313e40d07fbdc05e0d073d81357e2195c5eebbe0543256b26f0aa3cf56c4af"
    )
    assert next(value for path, value in paths.items() if path.endswith("correction.joblib")) == (
        "8a3b53f7899b23082354c7f5ff1813e0790f243c1777d022383f7c4f91bd4eed"
    )

    direct_c0_path = PROJECT_ROOT / models["C0"]["artifacts"][0]["relative_path"]
    bundled_c0_path = PROJECT_ROOT / next(
        path for path in paths if path.endswith("c0.joblib")
    )
    direct = joblib.load(direct_c0_path)
    bundled = joblib.load(bundled_c0_path)
    days = np.array([1, 100, 194, 236, 365], dtype=float)
    frame = pd.DataFrame(
        {
            f"doy_sin{harmonic}": np.sin(
                2 * np.pi * harmonic * days / 365.25
            )
            for harmonic in (1, 2)
        }
        | {
            f"doy_cos{harmonic}": np.cos(
                2 * np.pi * harmonic * days / 365.25
            )
            for harmonic in (1, 2)
        }
    )[CALENDAR_FEATURES]
    np.testing.assert_array_equal(direct.predict_proba(frame), bundled.predict_proba(frame))


def test_registry_hashes_cover_models_policies_runtime_and_verify(tmp_path: Path, registry: dict) -> None:
    assert registry["registry_content_sha256"] == canonical_sha256(
        {key: value for key, value in registry.items() if key != "registry_content_sha256"}
    )
    assert registry["runtime_source_set_sha256"] == canonical_sha256(
        registry["runtime_sources"]
    )
    for source in registry["runtime_sources"]:
        assert sha256_file(PROJECT_ROOT / source["relative_path"]) == source["sha256"]
    runtime_paths = {source["relative_path"] for source in registry["runtime_sources"]}
    assert {
        "src/agro_phenology/shadow_registry.py",
        "src/agro_phenology/shadow_sources.py",
        "src/agro_phenology/shadow_storage.py",
        "src/agro_phenology/shadow_engine.py",
        "src/agro_phenology/shadow_cli.py",
    }.issubset(runtime_paths)
    for model in registry["models"]:
        assert model["artifact_set_sha256"] == canonical_sha256(model["artifacts"])
        entry = {key: value for key, value in model.items() if key != "registry_entry_sha256"}
        assert model["registry_entry_sha256"] == canonical_sha256(entry)
        for policy in model["policies"]:
            payload = {
                key: value
                for key, value in policy.items()
                if key != "configuration_sha256"
            }
            assert policy["configuration_sha256"] == canonical_sha256(payload)

    path = write_shadow_registry(
        tmp_path / "shadow_registry.json",
        PROJECT_ROOT,
        locked_at_utc=LOCKED_AT,
    )
    result = verify_shadow_registry(path, PROJECT_ROOT)
    assert result["status"] == "passed"
    assert result["model_entries"] == 7
    assert result["artifacts_checked"] >= 20

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["models"][-1]["alpha"] = 0.5
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="registry content hash mismatch"):
        verify_shadow_registry(path, PROJECT_ROOT)
