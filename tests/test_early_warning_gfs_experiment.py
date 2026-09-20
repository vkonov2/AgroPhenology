from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agro_phenology.early_warning_core import (
    CALENDAR_FEATURES,
    NASA_COMMON_FEATURES,
    sha256_file,
)
from agro_phenology.early_warning_gfs_experiment import (
    GFS_FORECAST_FEATURES,
    MODEL_A,
    MODEL_A_MATCHED_C0,
    MODEL_A_STRONG,
    MODEL_B,
    MODEL_C,
    PRIMARY_SCOPE,
    SERVICE_SCOPE,
    _evaluate_model,
    _write_report_ru,
    availability_sensitivity_summary,
    assert_safe_feature_names,
    attach_forecast_features,
    calendar_window_bounds,
    contract_feature_columns,
    gfs_cell_id_from_weather_cell,
    model_training_rows,
    run_scenario_experiments,
    validate_contract,
    validate_forecast_input_bundle,
    validate_forecast_feature_table,
)
from agro_phenology.early_warning_models import Policy
from agro_phenology.gfs_feature_builder import (
    AUDITED_DDS_ACCESS_MODE,
    CHECKPOINT_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    SOURCE_MANIFEST_SCHEMA_VERSION,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    REPO_ROOT
    / "docs/research/late_blight_early_warning/gfs_forecast_evaluation_contract.json"
)


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _forecast_row(
    issue_date: str = "2020-06-01",
    *,
    lag: int = 4,
    complete: bool = True,
) -> dict:
    issue_time = pd.Timestamp(f"{issue_date}T05:00:00Z")
    selected_init = (issue_time - pd.Timedelta(hours=lag)).floor("6h")
    row = {
        "issue_date": issue_date,
        "issue_time_utc": issue_time,
        "gfs_cell_id": "gfs025_lat+57.00_lon+024.00",
        "latitude": 57.0,
        "longitude": 24.0,
        "availability_scenario_hours": lag,
        "selected_init_utc": selected_init,
        "assumed_available_at_utc": selected_init + pd.Timedelta(hours=lag),
        "publication_time_utc": pd.NaT,
        "selection_rule_id": f"latest_cycle_assumed_published_before_issue_v1_lag{lag}h",
        "source_dataset_id": "d084001",
        "source_hashes_json": '["' + "a" * 64 + '"]',
        "first_lead_h": 6 if lag == 4 else 12,
        "last_lead_h": 168 if lag == 4 else 174,
        "native_step_hours": 6,
        "expected_steps_1_3d": 12,
        "observed_steps_1_3d": 12 if complete else 11,
        "expected_steps_4_7d": 16,
        "observed_steps_4_7d": 16,
        "complete_1_3d": complete,
        "complete_4_7d": True,
        "forecast_available": complete,
    }
    for index, feature in enumerate(GFS_FORECAST_FEATURES):
        row[feature] = float(index + 1) if complete else np.nan
    return row


def _complete_bundle(tmp_path: Path) -> tuple[pd.DataFrame, dict[str, Path], Path]:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    contract_path = root / "contract.json"
    contract_path.write_text(
        CONTRACT_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    parent_path = root / "daily_decisions.parquet"
    parent_path.write_bytes(b"frozen-parent-test-bytes")

    feature_rows: list[dict] = []
    plan_rows: list[dict] = []
    for lag in (4, 7):
        feature = _forecast_row(lag=lag)
        checkpoint_id = f"20200601_lag{lag:02d}h"
        feature.update(
            {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "checkpoint_id": checkpoint_id,
                "source_access_mode": AUDITED_DDS_ACCESS_MODE,
                "requested_snapshot_count": 28,
                "retrieved_snapshot_count": 28,
                "failed_snapshot_count": 0,
                "failed_requests_json": "[]",
                "checkpoint_complete": True,
            }
        )
        feature_rows.append(feature)
        plan_rows.append(
            {
                "plan_schema_version": PLAN_SCHEMA_VERSION,
                "checkpoint_id": checkpoint_id,
                "issue_date": feature["issue_date"],
                "issue_time_utc": feature["issue_time_utc"],
                "availability_scenario_hours": lag,
                "selection_rule_id": feature["selection_rule_id"],
                "selected_init_utc": feature["selected_init_utc"],
                "assumed_available_at_utc": feature["assumed_available_at_utc"],
                "publication_time_utc": pd.NaT,
                "required_gfs_cell_ids_json": json.dumps([feature["gfs_cell_id"]]),
                "required_gfs_cell_count": 1,
                "requested_snapshot_count": 28,
                "expected_steps_1_3d": 12,
                "expected_steps_4_7d": 16,
                "source_dataset_id": "d084001",
            }
        )

    features = pd.DataFrame(feature_rows)
    plan = pd.DataFrame(plan_rows)
    feature_path = root / "gfs_forecast_features.parquet"
    plan_path = root / "gfs_request_plan.parquet"
    features.to_parquet(feature_path, index=False)
    plan.to_parquet(plan_path, index=False)
    privacy = {
        "contains_field_ids": False,
        "contains_outcomes": False,
        "contains_coarse_grid_coordinates": True,
        "distribution": "local_private_do_not_publish",
    }
    plan_manifest = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "contract_sha256": sha256_file(contract_path),
        "parent_daily_decisions_sha256": sha256_file(parent_path),
        "request_plan_sha256": sha256_file(plan_path),
        "checkpoints": 2,
        "scenario_hours": [4, 7],
        "archive_subset_requests": 56,
        **privacy,
    }
    source_manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "request_plan_sha256": sha256_file(plan_path),
        "feature_table_sha256": sha256_file(feature_path),
        "feature_rows": 2,
        "complete_feature_rows": 2,
        "complete_feature_fraction": 1.0,
        "planned_checkpoints": 2,
        "assembled_checkpoints": 2,
        "missing_checkpoint_count": 0,
        "missing_checkpoint_ids": [],
        "raw_subset_hash_count": 1,
        "raw_subset_sha256": ["a" * 64],
        "source_access_modes": [AUDITED_DDS_ACCESS_MODE],
        "forecast_features": GFS_FORECAST_FEATURES,
        "expected_steps": {"d1_3": 12, "d4_7": 16},
        **privacy,
    }
    source_manifest_path = root / "gfs_source_manifest.json"
    plan_manifest_path = root / "gfs_request_plan_manifest.json"
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    plan_manifest_path.write_text(json.dumps(plan_manifest), encoding="utf-8")
    return (
        features,
        {
            "forecast_features": feature_path,
            "source_manifest": source_manifest_path,
            "request_plan": plan_path,
            "request_plan_manifest": plan_manifest_path,
        },
        parent_path,
    )


def _decision_rows(dates: list[str]) -> pd.DataFrame:
    rows = []
    for day_number, date in enumerate(dates):
        row = {
            "field_season": "private_field_2020",
            "season": 2020,
            "issue_date": pd.Timestamp(date),
            "issued_at": f"{date}T08:00:00+03:00",
            "service_active": True,
            "evaluation_field_day": True,
            "target_class": "no_record_in_horizon",
            "target_observable": True,
            "days_to_first_recorded_event": np.nan,
            "warnable_first_event": False,
            "coordinate_scope": "A_direct",
            "previous_visit_gap_days": np.nan,
            "common_weather_complete": True,
            "nasa_common_complete": True,
            "weather_cell": "57.00_24.00",
        }
        angle = 2 * np.pi * (pd.Timestamp(date).dayofyear - 1) / 365.25
        row.update(
            {
                "doy_sin1": np.sin(angle),
                "doy_cos1": np.cos(angle),
                "doy_sin2": np.sin(2 * angle),
                "doy_cos2": np.cos(2 * angle),
            }
        )
        for index, feature in enumerate(NASA_COMMON_FEATURES):
            row[feature] = float(day_number + index + 1)
        rows.append(row)
    return pd.DataFrame(rows)


def test_frozen_contract_resolves_exact_twelve_forecast_features() -> None:
    contract = _contract()
    validate_contract(contract)
    past, forecast = contract_feature_columns(contract)
    assert past == list(NASA_COMMON_FEATURES)
    assert forecast == GFS_FORECAST_FEATURES
    assert len(forecast) == 12
    assert not any("step_count" in feature or "wet" in feature for feature in forecast)


def test_contract_rejects_changed_success_window_or_optuna() -> None:
    changed = _contract()
    changed["timeliness_window_days"] = {"minimum": 2, "maximum": 10}
    with pytest.raises(ValueError, match="3-10"):
        validate_contract(changed)
    changed = _contract()
    changed["optuna"]["enabled"] = True
    with pytest.raises(ValueError, match="Optuna"):
        validate_contract(changed)


@pytest.mark.parametrize(
    ("path", "changed_value"),
    [
        (("notification_policy", "active_days_per_message"), 6),
        (("notification_policy", "cooldown_days"), 14),
        (("notification_policy", "research_budget", "messages_per_30_field_days_max"), 2.1),
        (("notification_policy", "research_budget", "active_alarm_fraction_max"), 0.6),
        (("past_weather", "source"), "another_source"),
        (("past_weather", "features"), ["doy_sin1"]),
        (("past_weather", "cutoff_days_before_issue"), 1),
        (("catboost_fixed", "iterations"), 221),
        (("catboost_fixed", "loss_function"), "Logloss"),
        (("catboost_fixed", "random_seed"), 17),
        (("forecast_availability", "main_assumed_hours_after_initialization"), 5),
        (("forecast_availability", "delay_sensitivity_total_hours_after_initialization"), 8),
        (("source_dataset", "id"), "another_archive"),
        (("forecast_sampling", "native_archive_steps_hours_through_168"), 6),
        (("rolling_origin_folds",), []),
    ],
)
def test_contract_fails_closed_on_every_frozen_validity_setting(
    path: tuple[str, ...], changed_value: object
) -> None:
    changed = _contract()
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = changed_value
    with pytest.raises(ValueError):
        validate_contract(changed)


def test_contract_rejects_changed_forecast_band() -> None:
    changed = _contract()
    changed["forecast_sampling"]["bands"][0]["lead_hours_after_decision_closed"] = 71
    with pytest.raises(ValueError, match="bands"):
        validate_contract(changed)


def test_forecast_schema_accepts_real_4h_and_7h_total_lag_rows() -> None:
    validated = validate_forecast_feature_table(
        pd.DataFrame([_forecast_row(lag=4), _forecast_row(lag=7)])
    )
    assert validated["availability_scenario_hours"].tolist() == [4, 7]
    assert validated["expected_steps_1_3d"].eq(12).all()
    assert validated["expected_steps_4_7d"].eq(16).all()
    assert set(GFS_FORECAST_FEATURES).issubset(validated.columns)


def test_quality_input_bundle_requires_every_frozen_checkpoint_and_hash(
    tmp_path: Path,
) -> None:
    features, paths, parent_path = _complete_bundle(tmp_path)
    audit = validate_forecast_input_bundle(
        features,
        bundle_paths=paths,
        contract_path=paths["forecast_features"].parent / "contract.json",
        parent_decisions_path=parent_path,
        expected_scenarios=(4, 7),
    )
    assert audit["status"] == "complete_and_verified"
    assert audit["planned_checkpoints"] == audit["assembled_checkpoints"] == 2
    assert audit["planned_feature_rows"] == audit["assembled_feature_rows"] == 2


def test_quality_input_bundle_rejects_unfetched_checkpoint(tmp_path: Path) -> None:
    features, paths, parent_path = _complete_bundle(tmp_path)
    manifest = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    manifest["assembled_checkpoints"] = 1
    manifest["missing_checkpoint_count"] = 1
    manifest["missing_checkpoint_ids"] = ["20200601_lag07h"]
    paths["source_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="extraction is incomplete"):
        validate_forecast_input_bundle(
            features,
            bundle_paths=paths,
            contract_path=paths["forecast_features"].parent / "contract.json",
            parent_decisions_path=parent_path,
            expected_scenarios=(4, 7),
        )


def test_quality_input_bundle_rejects_hash_drift_and_missing_source_mode(
    tmp_path: Path,
) -> None:
    features, paths, parent_path = _complete_bundle(tmp_path)
    bad_manifest = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    bad_manifest["feature_table_sha256"] = "0" * 64
    paths["source_manifest"].write_text(json.dumps(bad_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="feature-table hash"):
        validate_forecast_input_bundle(
            features,
            bundle_paths=paths,
            contract_path=paths["forecast_features"].parent / "contract.json",
            parent_decisions_path=parent_path,
            expected_scenarios=(4, 7),
        )

    features, paths, parent_path = _complete_bundle(tmp_path / "second")
    features.loc[0, "source_access_mode"] = np.nan
    with pytest.raises(ValueError, match="provenance contains null"):
        validate_forecast_input_bundle(
            features,
            bundle_paths=paths,
            contract_path=paths["forecast_features"].parent / "contract.json",
            parent_decisions_path=parent_path,
            expected_scenarios=(4, 7),
        )


def test_forecast_schema_rejects_future_or_non_latest_cycle() -> None:
    wrong = _forecast_row()
    wrong["selected_init_utc"] = pd.Timestamp("2020-05-31T18:00:00Z")
    wrong["assumed_available_at_utc"] = pd.Timestamp("2020-05-31T22:00:00Z")
    with pytest.raises(ValueError, match="latest cycle"):
        validate_forecast_feature_table(pd.DataFrame([wrong]))


def test_forecast_schema_rejects_duplicates_and_private_identifiers() -> None:
    row = _forecast_row()
    with pytest.raises(ValueError, match="duplicate"):
        validate_forecast_feature_table(pd.DataFrame([row, row]))
    private = dict(row, field_uid="secret")
    with pytest.raises(ValueError, match="field/outcome"):
        validate_forecast_feature_table(pd.DataFrame([private]))


def test_forecast_schema_rejects_fake_complete_band_and_unproven_publication() -> None:
    inconsistent = _forecast_row()
    inconsistent["observed_steps_1_3d"] = 11
    with pytest.raises(ValueError, match="complete_1_3d"):
        validate_forecast_feature_table(pd.DataFrame([inconsistent]))
    published = _forecast_row()
    published["publication_time_utc"] = pd.Timestamp("2020-06-01T04:00:00Z")
    with pytest.raises(ValueError, match="no proven"):
        validate_forecast_feature_table(pd.DataFrame([published]))


def test_join_preserves_rows_and_constructs_exact_common_mask() -> None:
    decisions = _decision_rows(["2020-06-01", "2020-06-02", "2020-06-03"])
    decisions.loc[2, "nasa_common_complete"] = False
    forecasts = pd.DataFrame(
        [
            _forecast_row("2020-06-01"),
            _forecast_row("2020-06-03"),
        ]
    )
    joined = attach_forecast_features(decisions, forecasts, scenario_hours=4)
    assert joined.index.tolist() == decisions.index.tolist()
    assert joined["gfs_cell_id"].eq("gfs025_lat+57.00_lon+024.00").all()
    assert joined["gfs_feature_row_present"].tolist() == [True, False, True]
    assert joined["abc_complete"].tolist() == [True, False, False]
    assert joined["gfs_missing_reason"].tolist() == [
        "available",
        "no_forecast_feature_row",
        "available",
    ]


def test_availability_summary_separates_training_and_2020_2025_pool() -> None:
    years = [2010, 2015, 2019, 2020, 2025, 2026]
    decisions = pd.concat(
        [_decision_rows([f"{year}-06-01"]) for year in years], ignore_index=True
    )
    decisions["season"] = years
    decisions["field_season"] = [f"field_{year}" for year in years]
    forecasts = pd.DataFrame(
        [_forecast_row(f"{year}-06-01", lag=4) for year in years if 2015 <= year <= 2025]
    )
    summary = availability_sensitivity_summary(decisions, forecasts, (4,))
    annual = summary[summary["aggregation"].eq("year")]
    assert annual["season"].tolist() == [2015, 2019, 2020, 2025]
    pooled = summary[summary["aggregation"].eq("pooled_year_counts")]
    assert set(pooled["period"]) == {"training_support_2015_2019", "2020_2025"}
    counts = pooled.set_index("period")["service_field_days"].to_dict()
    assert counts == {"training_support_2015_2019": 2, "2020_2025": 2}


def test_join_rejects_issue_time_mismatch_instead_of_backdating() -> None:
    decisions = _decision_rows(["2020-06-01"])
    forecast = _forecast_row()
    forecast["issue_time_utc"] = pd.Timestamp("2020-06-01T06:00:00Z")
    forecast["issue_date"] = "2020-06-01"
    forecast["selected_init_utc"] = pd.Timestamp("2020-06-01T00:00:00Z")
    forecast["assumed_available_at_utc"] = pd.Timestamp("2020-06-01T04:00:00Z")
    with pytest.raises(ValueError, match="decision time"):
        attach_forecast_features(decisions, pd.DataFrame([forecast]), scenario_hours=4)


def test_private_weather_cell_mapping_uses_public_grid_identifier() -> None:
    mapped = gfs_cell_id_from_weather_cell(pd.Series(["57.00_24.00", "56.75_-1.25"]))
    assert mapped.tolist() == [
        "gfs025_lat+57.00_lon+024.00",
        "gfs025_lat+56.75_lon-001.25",
    ]


def test_model_training_rows_enforce_observability_and_exact_mask() -> None:
    frame = _decision_rows(["2020-06-01", "2020-06-02", "2020-06-03"])
    frame["abc_complete"] = [True, False, True]
    frame.loc[2, "target_observable"] = False
    selected = model_training_rows(frame, [2020, 2020])
    assert selected.index.tolist() == [0]


def test_feature_name_guard_blocks_outcomes_and_identifiers() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        assert_safe_feature_names(["doy_sin1", "days_to_first_recorded_event"])
    with pytest.raises(ValueError, match="forbidden"):
        assert_safe_feature_names(["field_uid_hash"])


def test_calendar_window_uses_training_events_and_common_coverage_only() -> None:
    seasons = pd.DataFrame(
        {
            "field_season": ["covered", "uncovered", "future"],
            "season": [2017, 2017, 2020],
            "warnable_first_event": [True, True, True],
            "first_recorded_event_date": pd.to_datetime(
                ["2017-06-10", "2017-09-20", "2020-01-01"]
            ),
        }
    )
    decisions = pd.DataFrame(
        {
            "field_season": ["covered", "uncovered", "future"],
            "season": [2017, 2017, 2020],
            "abc_complete": [True, False, True],
        }
    )
    lower, upper, count = calendar_window_bounds(seasons, decisions, [2015, 2017])
    assert count == 1
    assert lower == upper == pd.Timestamp("2017-06-10").dayofyear


def test_primary_replay_keeps_full_calendar_and_common_mask() -> None:
    decisions = _decision_rows(
        [f"2020-06-{day:02d}" for day in range(1, 21)]
    )
    decisions["abc_complete"] = False
    decisions.loc[[0, 19], "abc_complete"] = True
    decisions["gfs_forecast_complete"] = decisions["abc_complete"]
    score = pd.Series(1.0, index=decisions.index)
    result = _evaluate_model(
        frame=decisions,
        seasons=pd.DataFrame(
            columns=[
                "field_season",
                "season",
                "first_recorded_event_date",
                "coordinate_scope",
                "warnable_first_event",
                "positive_at_first_visit",
            ]
        ),
        raw_score=score,
        policy=Policy(0.5, 7, 15, "test_P0"),
        model_code=MODEL_A,
        fold_id="test_2020",
        scenario_hours=4,
        feature_provenance="test",
        evaluation_scope=PRIMARY_SCOPE,
    )["states"]
    assert len(result) == 20
    assert result["evaluation_scope_day"].sum() == 2
    assert result["message_issued"].sum() == 2
    assert not result.loc[~result["abc_complete"], "message_issued"].any()
    issued = result.loc[result["message_issued"], "issue_date"].tolist()
    assert issued == [pd.Timestamp("2020-06-01"), pd.Timestamp("2020-06-20")]


def _integration_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows: list[dict] = []
    season_rows: list[dict] = []
    years = [2010, 2015, 2016, 2017, 2018, 2019, 2020]
    classes = ["no_record_in_horizon", "imminent", "actionable"]
    for year in years:
        for class_index, target_class in enumerate(classes):
            field = f"f_{year}_{class_index}"
            event_date = pd.Timestamp(f"{year}-06-20") if target_class == "actionable" else pd.NaT
            season_rows.append(
                {
                    "field_season": field,
                    "season": year,
                    "first_recorded_event_date": event_date,
                    "warnable_first_event": target_class == "actionable",
                    "positive_at_first_visit": False,
                    "coordinate_scope": "A_direct",
                    "previous_visit_gap_days": 7.0,
                }
            )
            for day in range(1, 19):
                issue = pd.Timestamp(f"{year}-06-{day:02d}")
                angle = 2 * np.pi * (issue.dayofyear - 1) / 365.25
                row = {
                    "field_season": field,
                    "season": year,
                    "issue_date": issue,
                    "issued_at": f"{issue.date()}T08:00:00+03:00",
                    "service_active": True,
                    "evaluation_field_day": True,
                    "target_class": target_class,
                    "target_observable": True,
                    "days_to_first_recorded_event": (
                        float((event_date - issue).days) if pd.notna(event_date) else np.nan
                    ),
                    "warnable_first_event": target_class == "actionable",
                    "coordinate_scope": "A_direct",
                    "previous_visit_gap_days": 7.0,
                    "common_weather_complete": True,
                    "nasa_common_complete": True,
                    "abc_complete": True,
                    "gfs_forecast_complete": True,
                    "gfs_feature_row_present": True,
                    "gfs_missing_reason": "available",
                    "gfs_cell_id": "gfs025_lat+57.00_lon+024.00",
                    "issue_time_utc": pd.Timestamp(issue).tz_localize("Europe/Riga").replace(
                        hour=8
                    ).tz_convert("UTC"),
                    "selected_init_utc": pd.Timestamp(f"{issue.date()}T00:00:00Z"),
                    "assumed_available_at_utc": pd.Timestamp(f"{issue.date()}T04:00:00Z"),
                    "selection_rule_id": "latest_cycle_assumed_published_before_issue_v1_lag4h",
                    "doy_sin1": np.sin(angle),
                    "doy_cos1": np.cos(angle),
                    "doy_sin2": np.sin(2 * angle),
                    "doy_cos2": np.cos(2 * angle),
                    "nasa_test": float(day + class_index),
                    "fcst_test": float(day - class_index),
                }
                rows.append(row)
    contract = _contract()
    contract["rolling_origin_folds"] = [
        {
            "id": "test_2020",
            "train_years": [2015, 2017],
            "validation_years": [2018, 2019],
            "test_years": [2020, 2020],
        }
    ]
    contract["catboost_fixed"] = {
        "iterations": 5,
        "depth": 2,
        "learning_rate": 0.1,
        "l2_leaf_reg": 2.0,
        "loss_function": "MultiClass",
        "random_seed": 20260910,
    }
    contract["uncertainty"]["paired_year_bootstrap_repeats"] = 10
    return pd.DataFrame(rows), pd.DataFrame(season_rows), contract


def test_fold_pipeline_saves_matched_logit_and_primary_abc_outputs(tmp_path: Path) -> None:
    decisions, seasons, contract = _integration_data()
    outputs = run_scenario_experiments(
        decisions,
        seasons,
        contract,
        past_features=["nasa_test"],
        forecast_features=["fcst_test"],
        scenario_hours=4,
        model_root=tmp_path / "models",
    )
    registry = outputs["model_registry"]
    assert set(registry["model_code"]) == {
        MODEL_A,
        MODEL_A_MATCHED_C0,
        MODEL_A_STRONG,
        MODEL_B,
        MODEL_C,
    }
    matched = registry[registry["model_code"].eq(MODEL_A_MATCHED_C0)].iloc[0]
    assert matched["model_kind"] == "logistic"
    assert Path(matched["artifact"]).suffix == ".joblib"
    assert Path(matched["artifact"]).is_file()
    assert outputs["model_reload_verification"]["status"].eq(
        "prediction_roundtrip_checked"
    ).all()
    primary = outputs["alarm_states"]
    primary = primary[
        primary["evaluation_scope"].eq(PRIMARY_SCOPE)
        & primary["model_code"].isin([MODEL_A, MODEL_B, MODEL_C])
    ]
    counts = primary.groupby("model_code")["evaluation_scope_day"].sum()
    assert counts.nunique() == 1
    assert set(outputs["pooled_summary"]["availability_scenario_hours"]) == {4}
    assert set(outputs["alarm_states"]["evaluation_scope"]) == {
        PRIMARY_SCOPE,
        SERVICE_SCOPE,
    }


def test_b_and_c_use_same_seed_and_constant_forecast_gives_same_scores(
    tmp_path: Path,
) -> None:
    decisions, seasons, contract = _integration_data()
    decisions["fcst_test"] = 7.0
    outputs = run_scenario_experiments(
        decisions,
        seasons,
        contract,
        past_features=["nasa_test"],
        forecast_features=["fcst_test"],
        scenario_hours=4,
        model_root=tmp_path / "models",
    )
    registry = outputs["model_registry"].set_index("model_code")
    assert registry.loc[MODEL_B, "seed"] == registry.loc[MODEL_C, "seed"]
    assert registry.loc[MODEL_B, "params_json"] == registry.loc[MODEL_C, "params_json"]
    primary = outputs["alarm_states"]
    primary = primary[
        primary["evaluation_scope"].eq(PRIMARY_SCOPE)
        & primary["model_code"].isin((MODEL_B, MODEL_C))
    ]
    keys = ["field_season", "season", "issue_date"]
    b_scores = primary[primary["model_code"].eq(MODEL_B)].sort_values(keys)["score"]
    c_scores = primary[primary["model_code"].eq(MODEL_C)].sort_values(keys)["score"]
    np.testing.assert_allclose(
        b_scores.to_numpy(), c_scores.to_numpy(), atol=1e-12, rtol=0
    )


def test_report_distinguishes_pooled_and_annual_budget_exceedance(
    tmp_path: Path,
) -> None:
    pooled = pd.DataFrame(
        [
            {
                "availability_scenario_hours": 4,
                "model_code": code,
                "evaluation_scope": PRIMARY_SCOPE,
                "slice": "A_plus_B",
                "period": "2020_2025",
                "timely_hits": hits,
                "events_with_warning_opportunity": 87,
                "timely_recall": hits / 87,
                "messages": messages,
                "messages_per_30_field_days": rate,
                "active_alarm_days": alarm_days,
                "active_alarm_fraction": alarm_fraction,
                "computable_fraction": 1.0,
            }
            for code, hits, messages, rate, alarm_days, alarm_fraction in (
                (MODEL_A, 44, 288, 1.25, 1902, 0.275),
                (MODEL_B, 46, 378, 1.64, 2563, 0.370),
                (MODEL_C, 41, 387, 1.68, 2615, 0.378),
            )
        ]
    )
    annual = pd.DataFrame(
        [
            {
                "availability_scenario_hours": 4,
                "candidate": MODEL_C,
                "baseline": baseline,
                "evaluation_scope": PRIMARY_SCOPE,
                "season": 2020,
                "candidate_messages_per_30_field_days": 2.263,
                "baseline_messages_per_30_field_days": 1.5,
                "candidate_active_alarm_fraction": 0.5035,
                "baseline_active_alarm_fraction": 0.3,
            }
            for baseline in (MODEL_A, MODEL_B)
        ]
    )
    bootstrap = pd.DataFrame(
        columns=[
            "availability_scenario_hours",
            "evaluation_scope",
            "candidate",
            "baseline",
        ]
    )
    availability = pd.DataFrame(columns=["aggregation"])
    path = tmp_path / "report.md"

    _write_report_ru(
        path,
        pooled=pooled,
        annual=annual,
        bootstrap=bootstrap,
        availability=availability,
        primary_scenario=4,
        scenarios=(4,),
        contract=_contract(),
    )

    report = path.read_text(encoding="utf-8")
    assert "Совокупный бюджет 2020–2025 для C соблюдён" in report
    assert "В 1 из 1 внешних лет C превысила" in report
    assert "candidate_messages_per_30_field_days" in report
    assert "candidate_budget_status" in report
