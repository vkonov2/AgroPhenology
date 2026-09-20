from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from agro_phenology.shadow_reporting import generate_shadow_readiness_run


CREATED_AT = "2026-09-10T16:00:00.000000Z"


def _inputs() -> dict[str, object]:
    registry = {
        "registry_id": "shadow-registry-v1",
        "locked_at_utc": CREATED_AT,
        "selected_fold": "test_2026_partial",
        "models": [
            {
                "model_id": "C6_weather",
                "artifacts": [
                    {
                        "relative_path": "results/late_blight/models/frozen.bundle",
                        "sha256": "a" * 64,
                    }
                ],
                "local_debug_path": "/Users/konov/private/model.bundle",
            }
        ],
    }
    sources = {
        "weather_branch_executable_today": False,
        "sources": [
            {
                "source_id": "era5_frozen",
                "required": "ERA5, cutoff issue_date-2",
                "actual": "ERA5 with documented lag near five days",
                "freshness": "required 2026-09-08; expected about 2026-09-05",
                "status": "blocked_by_freshness",
                "evidence": "official_documentation_not_live_measurement",
                "location_ref": "secret-weather-cell-42",
            }
        ],
        "branches": [
            {
                "model_id": "C6_weather",
                "role": "primary",
                "required_inputs": ["calendar", "ERA5 episodes"],
                "status": "c0_fallback_only",
                "fallback": "C0 in one policy history",
                "reason": "ERA5 t-2 unavailable",
            }
        ],
    }
    demo = {
        "mode": "offline_technical_demo",
        "status": "passed",
        "live_run_executed": False,
        "real_notifications_sent": 0,
        "schedule_activated": False,
        "field_id": "synthetic-field-that-must-be-redacted",
        "decisions": 12,
        "expected_decisions": 12,
        "computed_fraction": 0.75,
        "abstentions_by_reason": {"weather_unavailable": 4},
        "weather_correction_fraction": 0.0,
        "effective_c0_fraction": 1.0,
        "virtual_messages": 2,
        "active_alarm_days": 7,
        "active_alarm_fraction": 0.25,
        "missed_slots": 0,
        "field_registry_status": "synthetic_only",
        "network_status": "not_called",
    }
    tests = {"status": "passed", "passed": 24, "failed": 0}
    blockers = {
        "blockers": [
            {
                "id": "active_field_registry_missing",
                "status": "blocked",
                "detail": "Нет актуального реестра полей",
                "required_action": "Передать обезличенный реестр подключения",
            }
        ]
    }
    return {
        "registry": registry,
        "source_compatibility_manifest": sources,
        "demo_summary": demo,
        "test_summary": tests,
        "blockers": blockers,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generator_creates_complete_versioned_run_and_verified_manifest(tmp_path) -> None:
    destination = tmp_path / "20260910_shadow_readiness_test_v1"
    database = tmp_path / "synthetic.sqlite"
    database.write_bytes(b"SQLite format 3\x00synthetic only")
    replay = tmp_path / "replay.json"
    replay.write_text('{"status":"identical"}\n')
    result = generate_shadow_readiness_run(
        destination,
        created_at_utc=CREATED_AT,
        technical_demo_artifacts={
            "synthetic_shadow.sqlite": database,
            "checks/replay.json": replay,
        },
        supplemental_json={
            "parent_integrity_before.json": {"all_match": True},
            "contracts/shadow_readiness_contract.json": {"version": "1.0.0"},
        },
        **_inputs(),
    )

    expected = {
        "shadow_registry.json",
        "source_compatibility_manifest.json",
        "operational_readiness_ru.md",
        "decision_record.md",
        "RUNBOOK.md",
        "field_observation_instructions_ru.md",
        "blockers.json",
        "readiness_summary.json",
        "technical_demo_summary.json",
        "test_results.json",
        "prospective_evaluation_plan.json",
        "REPRODUCE.md",
        "review_package.zip",
        "execution_manifest.json",
        "schemas/weather_retrieval.schema.json",
        "schemas/active_field_registry.schema.json",
        "schemas/observation.schema.json",
        "schemas/decision.schema.json",
        "technical_demo/synthetic_shadow.sqlite",
        "technical_demo/checks/replay.json",
        "parent_integrity_before.json",
        "contracts/shadow_readiness_contract.json",
    }
    actual = {
        path.relative_to(result).as_posix()
        for path in result.rglob("*")
        if path.is_file() and not path.is_relative_to(result / "review_package")
    }
    assert expected == actual

    manifest = json.loads((result / "execution_manifest.json").read_text())
    assert manifest["stage"] == "prospective_shadow_readiness"
    assert manifest["side_effects"] == {
        "models_trained_or_tuned": 0,
        "network_requests": 0,
        "real_notifications_sent": 0,
        "schedule_activated": False,
    }
    for relative, metadata in manifest["output_hashes"].items():
        path = result / relative
        assert path.is_file()
        assert metadata["sha256"] == _sha256(path)
        assert metadata["bytes"] == path.stat().st_size
    assert "execution_manifest.json" not in manifest["output_hashes"]

    report = (result / "operational_readiness_ru.md").read_text()
    assert "не является проспективным запуском" in report
    assert "не может** штатно использовать ERA5" in report
    assert "C0 fallback" in report
    assert "не проверяет погодную поправку C6" in report
    assert "Реальные уведомления не отправлялись" in report
    assert "расписание не установлено" in report
    assert "не оценивает чувствительность" in report
    assert "Агрегаты технической демонстрации" in report
    assert "Ожидаемые решения | 12" in report
    assert "Записанные решения | 12" in report
    assert "Воздержания по причинам | weather_unavailable=4" in report
    assert "Доля активной weather-поправки в C6 | 0.0" in report
    assert "Доля эффективного C0 в C6 | 1.0" in report
    assert "Реальный run-once выполнен | нет" in report
    assert "Статус сети | not_called" in report

    readiness = json.loads((result / "readiness_summary.json").read_text())
    assert readiness["weather_branch"]["decision"] == "blocked_by_freshness"
    assert readiness["weather_branch"]["c6_behavior"] == "c0_fallback_only"
    assert readiness["weather_branch"]["weather_effect_tested"] is False
    assert readiness["technical_demo"]["quality_claim_allowed"] is False
    assert readiness["prospective_live"]["executed"] is False
    evaluation = json.loads((result / "prospective_evaluation_plan.json").read_text())
    assert evaluation["primary_warning_window_calendar_days"] == {
        "maximum": 10,
        "minimum": 3,
    }
    assert evaluation["population"]["fixed_historical_denominator"] is None
    assert evaluation["outcome"]["missing_or_not_visited_is_negative"] is False
    assert evaluation["analysis_governance"]["repeated_winner_selection_on_accumulating_outcomes"] is False


def test_review_zip_contains_only_sanitised_review_documents(tmp_path) -> None:
    result = generate_shadow_readiness_run(
        tmp_path / "run",
        created_at_utc=CREATED_AT,
        **_inputs(),
    )
    with zipfile.ZipFile(result / "review_package.zip") as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert "README.md" in names
        assert "operational_readiness_ru.md" in names
        assert "blockers.json" in names
        assert "readiness_summary.json" in names
        assert not any(
            Path(name).suffix.lower()
            in {".joblib", ".cbm", ".parquet", ".sqlite", ".db", ".pkl", ".npy"}
            for name in names
        )
        combined = b"\n".join(archive.read(name) for name in names).decode("utf-8")
    assert "/Users/" not in combined
    assert "secret-weather-cell-42" not in combined
    assert "synthetic-field-that-must-be-redacted" not in combined
    assert "<обезличено>" in combined
    assert "Техническая демонстрация не является проспективным запуском" in combined


def test_generator_refuses_to_overwrite_existing_run(tmp_path) -> None:
    destination = tmp_path / "run"
    destination.mkdir()
    marker = destination / "user-file.txt"
    marker.write_text("keep")

    with pytest.raises(FileExistsError, match="already exists"):
        generate_shadow_readiness_run(
            destination,
            created_at_utc=CREATED_AT,
            **_inputs(),
        )
    assert marker.read_text() == "keep"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"live_run_executed": True}, "live run"),
        ({"real_notifications_sent": 1}, "must not send"),
        ({"schedule_activated": True}, "must not activate"),
        ({"mode": "prospective_live"}, "only an offline"),
    ],
)
def test_generator_rejects_claims_or_side_effects_outside_offline_demo(
    tmp_path, override, message
) -> None:
    inputs = _inputs()
    inputs["demo_summary"] = {**inputs["demo_summary"], **override}
    with pytest.raises(ValueError, match=message):
        generate_shadow_readiness_run(
            tmp_path / "run",
            created_at_utc=CREATED_AT,
            **inputs,
        )


def test_schemas_keep_time_roles_outcomes_and_location_reference_explicit(tmp_path) -> None:
    result = generate_shadow_readiness_run(
        tmp_path / "run",
        created_at_utc=CREATED_AT,
        **_inputs(),
    )
    weather = json.loads((result / "schemas/weather_retrieval.schema.json").read_text())
    assert {
        "run_initialization_at_utc",
        "provider_published_at_utc",
        "retrieval_started_at_utc",
        "retrieval_completed_at_utc",
        "first_seen_at_utc",
        "ingested_at_utc",
        "valid_start_utc",
    }.issubset(weather["properties"])
    assert weather["properties"]["availability_evidence"]["enum"] == [
        "actual_retrieval",
        "confirmed_publication",
        "assumption",
        "unknown",
    ]

    fields = json.loads((result / "schemas/active_field_registry.schema.json").read_text())
    item_properties = fields["properties"]["fields"]["items"]["properties"]
    assert "weather_location_ref" in item_properties
    assert "latitude" not in item_properties
    assert fields["properties"]["timezone"]["const"] == "Europe/Riga"
    assert fields["properties"]["historical_coordinates_used"]["const"] is False
    assert {"generated_at_utc", "available_at_utc", "ingested_at_utc"}.issubset(
        fields["required"]
    )
    assert item_properties["crop"]["const"] == "potato"
    assert "region_code" in item_properties
    assert {"crop", "region_code"}.issubset(
        fields["properties"]["fields"]["items"]["required"]
    )
    assert "irrigation_logging_capability" in item_properties

    observation = json.loads((result / "schemas/observation.schema.json").read_text())
    payload = observation["properties"]["payload"]
    outcomes = payload["properties"]["outcome_category"]["enum"]
    assert "target_specific_negative" in outcomes
    assert "generic_absent" in outcomes
    assert "unassessed" in outcomes
    assert "not_visited" in outcomes
    assert {
        "retrieval_started_at_utc",
        "retrieval_completed_at_utc",
    }.issubset(observation["properties"])
    assert {
        "visit_plan_created_at_utc",
        "visit_trigger",
        "planned_before_decision",
        "observer_role",
        "examined_extent",
        "result_available_at_utc",
        "first_symptom_date_precision",
        "censoring_status",
    }.issubset(payload["properties"])
    assert observation["properties"]["information_role"]["const"] == "outcome"

    decision = json.loads((result / "schemas/decision.schema.json").read_text())
    assert "outcome" not in decision["properties"]
    assert "next_visit_date" not in decision["properties"]
    assert decision["properties"]["input_hashes"]["type"] == "object"
    assert "exact_era5_episode" in decision["properties"]["score_origin"]["enum"]
    assert "not_in_service" in decision["properties"]["decision_action"]["enum"]
    assert "not_in_service" in decision["properties"]["score_status"]["enum"]
    assert decision["properties"]["transition"]["properties"][
        "score_comparison_segment_id"
    ]["type"] == ["integer", "string", "null"]


@pytest.mark.parametrize(
    "unsafe_name",
    ["../escape.json", "/absolute.json", "nested/../../escape.json", r"C:\\escape.json"],
)
def test_external_artifact_names_cannot_escape_run_directory(tmp_path, unsafe_name) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}")
    with pytest.raises(ValueError, match="unsafe"):
        generate_shadow_readiness_run(
            tmp_path / "run",
            created_at_utc=CREATED_AT,
            technical_demo_artifacts={unsafe_name: source},
            **_inputs(),
        )
    assert not (tmp_path / "run").exists()


def test_runtime_demo_shape_is_normalised_into_report_aggregates(tmp_path) -> None:
    inputs = _inputs()
    inputs["demo_summary"] = {
        "status": "passed",
        "demonstration_mode": "retrospective_replay",
        "prospective_live_run": False,
        "synthetic_inputs_only": True,
        "network_used": False,
        "notifications_sent": 0,
        "unique_decisions": 20,
        "exact_weather_decisions": 4,
        "c0_fallback_decisions": 6,
        "active_weather_fraction_in_c6": 0.25,
        "effective_c0_fraction_in_c6": 0.75,
        "abstentions": 5,
        "virtual_message_candidates": 3,
    }
    result = generate_shadow_readiness_run(
        tmp_path / "run",
        created_at_utc=CREATED_AT,
        **inputs,
    )
    demo = json.loads((result / "technical_demo_summary.json").read_text())
    assert demo["mode"] == "retrospective_replay"
    assert demo["computed_fraction"] == 0.75
    assert demo["weather_correction_fraction"] == 0.25
    assert demo["effective_c0_fraction"] == 0.75
    assert demo["field_registry_status"] == "synthetic_only"
    assert demo["network_status"] == "not_called"
    report = (result / "operational_readiness_ru.md").read_text()
    assert "Доля вычисленных решений | 0.75" in report
    assert "Доля активной weather-поправки в C6 | 0.25" in report
    assert "Виртуальные сообщения | 3" in report
