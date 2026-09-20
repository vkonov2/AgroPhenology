from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from agro_phenology.shadow_audit import (
    build_source_compatibility_manifest,
    find_active_field_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_current_workspace_source_audit_keeps_documented_and_measured_freshness_separate() -> None:
    audit = build_source_compatibility_manifest(
        PROJECT_ROOT,
        issue_local_date=date(2026, 9, 10),
        audited_at_utc="2026-09-10T16:00:00+00:00",
    )

    assert audit["required_weather_last_local_date"] == "2026-09-08"
    assert audit["current_api_probe"]["performed"] is False
    assert audit["current_api_probe"]["measured_live_freshness"] is None
    assert audit["documented_availability"]["era5_era5t"][
        "compatible_with_required_t_minus_2"
    ] is False

    era = audit["local_snapshot_measurements"]["era5"]
    nasa = audit["local_snapshot_measurements"]["nasa_power"]
    assert era["date_max"] == "2026-08-20"
    assert era["latest_valid_date_gap_to_required_days"] == 19
    assert era["accepted_rows"] == era["rows"] == 39727
    assert nasa["date_max"] == "2026-08-31"
    assert nasa["latest_valid_date_gap_to_required_days"] == 8
    assert audit["training_raw_response_audit"]["exact_era5_raw_responses_found"] == 0
    assert audit["training_raw_response_audit"]["expected_era5_request_keys"] == 403
    assert audit["training_raw_response_audit"]["exact_nasa_raw_responses_found"] == 0
    assert audit["training_raw_response_audit"]["expected_nasa_request_keys"] == 41
    assert "not measured provider latency" in audit["local_snapshot_measurements"][
        "interpretation"
    ]


def test_source_audit_never_promotes_historical_fields_and_preserves_fallback_semantics() -> None:
    audit = build_source_compatibility_manifest(
        PROJECT_ROOT,
        issue_local_date="2026-09-10",
        audited_at_utc="2026-09-10T16:00:00Z",
    )
    models = {row["model_id"]: row for row in audit["participants"]}

    assert audit["active_field_registry"]["historical_tables_considered_active"] is False
    assert audit["active_field_registry"]["status"] == "missing"
    assert models["C4"]["status"] == "blocked_exact_era5_t_minus_2_abstain"
    assert models["C5"]["status"] == "blocked_exact_era5_t_minus_2_abstain"
    assert models["C6_weather"]["weather_correction_active_today"] is False
    assert models["C6_weather"]["effective_origin_without_weather"] == "C0_fallback"
    assert models["C6_calibration_control"]["alpha"] == 0.0
    assert audit["forecast_archive"]["automatic_use_in_frozen_past_only_features"] is False
    assert audit["forecast_archive"]["alternative_products_status"] == (
        "requires_separate_source_bridge_study"
    )


def test_only_allowlisted_or_explicit_registry_is_discovered(tmp_path) -> None:
    root = tmp_path
    historical = root / "results/old/field_registry.csv"
    historical.parent.mkdir(parents=True)
    pd.DataFrame({"field": ["historical"]}).to_csv(historical, index=False)
    assert find_active_field_registry(root)["status"] == "missing"

    active = root / "operator/active.json"
    active.parent.mkdir()
    active.write_text("{}", encoding="utf-8")
    found = find_active_field_registry(root, active)
    assert found["status"] == "available"
    assert found["discovery"] == "operator_supplied_path"
    assert found["matched_paths"] == ["operator/active.json"]

    csv_registry = root / "operator/active.csv"
    csv_registry.write_text("field_pseudo_id\nexample\n", encoding="utf-8")
    found_csv = find_active_field_registry(root, csv_registry)
    assert found_csv["status"] == "incompatible_schema_format"
    assert found_csv["accepted_formats"] == ["json"]
