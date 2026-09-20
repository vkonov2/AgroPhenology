from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from agro_phenology.shadow_engine import (
    ACTIVE_REGISTRY_PURPOSE,
    FieldRegistryError,
    PROSPECTIVE_MODE,
    _synthetic_era5_payload,
    archive_exact_era5_observation,
    evaluate_one_policy_transition,
    load_active_field_registry,
    readiness,
    replay_from_log,
    run_once,
    technical_demo,
    verify_log,
)
from agro_phenology.shadow_registry import write_shadow_registry
from agro_phenology.shadow_storage import ShadowStorage


ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc


@pytest.fixture(scope="module")
def frozen_registry(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("shadow-registry") / "registry.json"
    write_shadow_registry(
        target, ROOT, locked_at_utc="2026-09-10T16:00:00Z"
    )
    return target


@pytest.fixture(scope="module")
def completed_demo(
    tmp_path_factory: pytest.TempPathFactory, frozen_registry: Path
) -> tuple[Path, dict[str, object]]:
    database = tmp_path_factory.mktemp("shadow-demo") / "shadow.sqlite"
    result = technical_demo(
        project_root=ROOT,
        registry_path=frozen_registry,
        database_path=database,
    )
    return database, result


def _policy(*, override: bool = True) -> dict[str, object]:
    return {
        "policy_id": "test__P_growth",
        "family": "P_growth_selected",
        "threshold": 0.1,
        "active_days": 7,
        "cooldown_days": 15,
        "minimum_repeat_interval_days": 7,
        "growth_override_enabled": override,
        "growth_logit_delta": 0.01 if override else None,
        "logit_epsilon": 1e-6,
        "configuration_sha256": "test-policy-v1",
        "research_budget": {
            "messages_per_30_field_days_max": 2.0,
            "active_alarm_fraction_max": 0.5,
        },
    }


def _field_registry(*, generated: str = "2026-09-10T15:00:00Z") -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "registry_id": "actual-current-fields-v1",
        "registry_purpose": ACTIVE_REGISTRY_PURPOSE,
        "generated_at_utc": generated,
        "available_at_utc": "2026-09-10T15:01:00Z",
        "ingested_at_utc": "2026-09-10T15:02:00Z",
        "timezone": "Europe/Riga",
        "historical_coordinates_used": False,
        "fields": [
            {
                "field_pseudo_id": "field-random-01",
                "season_id": "season-random-01",
                "field_season": "field-random-01/2026",
                "season": 2026,
                "crop": "potato",
                "region_code": "REGION-X",
                "enrolled_from": "2026-09-01",
                "enrolled_until": None,
                "status": "active",
                "weather_location_ref": "protected-weather-ref-01",
            }
        ],
    }


def test_field_registry_requires_current_pseudonymised_contract() -> None:
    valid = _field_registry()
    loaded = load_active_field_registry(
        valid, decision_at_utc="2026-09-10T16:10:00Z"
    )
    assert loaded["field_registry_content_sha256"]

    leaked = json.loads(json.dumps(valid))
    leaked["fields"][0]["latitude"] = 56.9
    with pytest.raises(FieldRegistryError, match="protected direct fields"):
        load_active_field_registry(
            leaked, decision_at_utc="2026-09-10T16:10:00Z"
        )

    historical = json.loads(json.dumps(valid))
    historical["historical_coordinates_used"] = True
    with pytest.raises(FieldRegistryError, match="historical_coordinates_used"):
        load_active_field_registry(
            historical, decision_at_utc="2026-09-10T16:10:00Z"
        )


def test_one_c6_history_breaks_reference_on_source_switch_without_resetting_cooldown() -> None:
    first, state = evaluate_one_policy_transition(
        field_season="field/2026",
        season=2026,
        issue_day=date(2026, 7, 15),
        decision_at_utc=datetime(2026, 7, 15, 5, tzinfo=UTC),
        service_active=True,
        score=0.2,
        score_origin="exact_era5_episode",
        model_id="C6_weather",
        model_version="frozen-v1",
        policy_payload=_policy(),
        state_before={},
    )
    assert first["shadow_message_would_be_issued"]

    fallback, state = evaluate_one_policy_transition(
        field_season="field/2026",
        season=2026,
        issue_day=date(2026, 7, 22),
        decision_at_utc=datetime(2026, 7, 22, 5, tzinfo=UTC),
        service_active=True,
        score=0.95,
        score_origin="c0_fallback_missing_exact_era5",
        model_id="C6_weather",
        model_version="frozen-v1",
        policy_payload=_policy(),
        state_before=state,
    )
    assert not fallback["shadow_message_would_be_issued"]
    assert fallback["growth_reference_status"] == "score_origin_or_model_not_comparable"
    assert fallback["previous_message_date"] == "2026-07-15"
    assert fallback["cumulative_messages_field_season"] == 1

    restored, state = evaluate_one_policy_transition(
        field_season="field/2026",
        season=2026,
        issue_day=date(2026, 7, 23),
        decision_at_utc=datetime(2026, 7, 23, 5, tzinfo=UTC),
        service_active=True,
        score=0.99,
        score_origin="exact_era5_episode",
        model_id="C6_weather",
        model_version="frozen-v1",
        policy_payload=_policy(),
        state_before=state,
    )
    assert not restored["shadow_message_would_be_issued"]
    assert restored["growth_reference_status"] == "not_same_continuous_score_segment"
    assert restored["previous_message_date"] == "2026-07-15"
    assert state["cumulative_messages"] == 1


def test_full_daily_demo_is_idempotent_sequential_and_never_delivers(
    completed_demo: tuple[Path, dict[str, object]],
) -> None:
    database, result = completed_demo
    assert result["status"] == "passed"
    assert result["prospective_live_run"] is False
    assert result["synthetic_inputs_only"] is True
    assert result["expected_daily_slots_per_field"] == 9
    assert result["executed_daily_slots_per_field"] == 9
    assert result["missed_daily_slots"] == 0
    assert result["unique_decisions"] == 234
    assert result["notifications_sent"] == 0
    assert result["source_switch_breaks_growth_comparability"] is True
    assert result["source_switch_preserves_cooldown_state"] is True
    retry = result["idempotent_retry"]
    assert retry["decisions_created"] == 0
    assert retry["decisions_already_present"] == 26
    assert retry["decision_at_utc"] != retry["scheduled_slot_utc"]

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 234
        assert connection.execute("SELECT COUNT(*) FROM policy_state").fetchone()[0] == 26
        assert connection.execute(
            "SELECT COUNT(DISTINCT decision_slot_utc) FROM decisions"
        ).fetchone()[0] == 9
        sent = [
            json.loads(row[0])["actually_sent"]
            for row in connection.execute("SELECT decision_payload_json FROM decisions")
        ]
        field_registry_timing = connection.execute(
            """
            SELECT retrieval_started_at_utc, retrieval_completed_at_utc,
                   first_seen_at_utc, ingested_at_utc
            FROM retrieval_records WHERE source='active_field_registry'
            """
        ).fetchone()
    assert not any(sent)
    assert field_registry_timing == (
        "2026-07-15T05:00:00.000000Z",
        "2026-07-15T05:00:00.000000Z",
        "2026-07-15T05:00:00.000000Z",
        "2026-07-15T05:00:00.000000Z",
    )


def test_missing_weather_abstains_c4_c5_and_c6_is_exact_c0_fallback(
    completed_demo: tuple[Path, dict[str, object]],
) -> None:
    database, _ = completed_demo
    with sqlite3.connect(database) as connection:
        payloads = [
            json.loads(row[0])
            for row in connection.execute(
                """
                SELECT decision_payload_json FROM decisions
                WHERE field_season='synthetic-field-fallback/2026'
                  AND decision_slot_utc='2026-07-15T05:00:00.000000Z'
                """
            )
        ]
    weather_diagnostics = [row for row in payloads if row["model_id"] in {"C4", "C5"}]
    assert len(weather_diagnostics) == 4
    assert all(row["decision_action"] == "abstain" for row in weather_diagnostics)
    c6 = [row for row in payloads if row["model_id"] == "C6_weather"]
    calibration = [
        row for row in payloads if row["model_id"] == "C6_calibration_control"
    ]
    assert len(c6) == 2 and len(calibration) == 1
    assert all(row["used_c0_fallback"] for row in c6)
    assert calibration[0]["alpha_zero_identity"] is True
    # Both frozen bundles contain the same C0 base and alpha=0 is an exact identity.
    assert all(row["score"] == calibration[0]["score"] for row in c6)


def test_raw_backed_replay_uses_linked_version_and_ignores_late_weather(
    completed_demo: tuple[Path, dict[str, object]], frozen_registry: Path
) -> None:
    database, _ = completed_demo
    store = ShadowStorage(database)
    body = json.loads(
        _synthetic_era5_payload(date(2026, 6, 15), date(2026, 7, 21)).decode(
            "utf-8"
        )
    )
    body["hourly"]["temperature_2m"] = [5.0] * len(body["hourly"]["time"])
    archive_exact_era5_observation(
        store,
        retrieval_id="late-corrected-era5-retrieval",
        observation_id="late-corrected-era5-observation",
        weather_cell="synthetic-cell-weather",
        body=json.dumps(body, separators=(",", ":")),
        retrieved_at_utc="2026-09-10T17:00:00Z",
        valid_from_utc="2026-06-14T21:00:00Z",
        valid_to_utc="2026-07-21T21:00:00Z",
        synthetic_fixture=True,
    )
    replay = replay_from_log(
        database_path=database,
        registry_path=frozen_registry,
        project_root=ROOT,
    )
    assert replay["status"] == "passed"
    assert replay["decisions_checked"] == 234
    assert replay["database_mutated"] is False
    assert verify_log(
        database_path=database,
        registry_path=frozen_registry,
        project_root=ROOT,
    )["status"] == "passed"


def test_replay_and_verification_are_byte_preserving(
    completed_demo: tuple[Path, dict[str, object]], frozen_registry: Path
) -> None:
    database, _ = completed_demo
    before = hashlib.sha256(database.read_bytes()).hexdigest()

    replay = replay_from_log(
        database_path=database,
        registry_path=frozen_registry,
        project_root=ROOT,
    )
    after_replay = hashlib.sha256(database.read_bytes()).hexdigest()
    assert replay["status"] == "passed"
    assert replay["database_mutated"] is False
    assert replay["database_sha256_before"] == before
    assert replay["database_sha256_after"] == before
    assert after_replay == before

    verification = verify_log(
        database_path=database,
        registry_path=frozen_registry,
        project_root=ROOT,
    )
    after_verification = hashlib.sha256(database.read_bytes()).hexdigest()
    assert verification["status"] == "passed"
    assert verification["database_mutated"] is False
    assert verification["database_sha256_before"] == before
    assert verification["database_sha256_after"] == before
    assert after_verification == before


def test_readiness_reports_missing_actual_registry(frozen_registry: Path) -> None:
    result = readiness(project_root=ROOT, registry_path=frozen_registry)
    assert result["status"] == "partially_ready"
    assert result["field_registry"] == {"status": "missing", "fields": 0}
    assert "missing_actual_active_field_registry" in result["blockers"]
    assert result["notifications_sent"] == 0


def test_prospective_guards_registry_lock_and_pre_scheduled_time(
    tmp_path: Path, frozen_registry: Path
) -> None:
    fields = _field_registry()
    with pytest.raises(ValueError, match="precedes the frozen registry lock"):
        run_once(
            project_root=ROOT,
            registry_path=frozen_registry,
            field_registry=fields,
            database_path=tmp_path / "before-lock.sqlite",
            decision_at_utc="2026-09-10T15:30:00Z",
            created_at_utc="2026-09-10T15:31:00Z",
            mode=PROSPECTIVE_MODE,
        )

    with pytest.raises(ValueError, match="scheduled slot precedes"):
        run_once(
            project_root=ROOT,
            registry_path=frozen_registry,
            field_registry=fields,
            database_path=tmp_path / "late-after-lock.sqlite",
            decision_at_utc="2026-09-10T16:10:00Z",
            created_at_utc="2026-09-10T16:11:00Z",
            mode=PROSPECTIVE_MODE,
        )

    next_day_fields = _field_registry()
    with pytest.raises(ValueError, match="before 08:00 Europe/Riga"):
        run_once(
            project_root=ROOT,
            registry_path=frozen_registry,
            field_registry=next_day_fields,
            database_path=tmp_path / "before-slot.sqlite",
            decision_at_utc="2026-09-11T04:59:00Z",
            created_at_utc="2026-09-11T04:59:30Z",
            mode=PROSPECTIVE_MODE,
        )
