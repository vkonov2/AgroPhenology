from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import sqlite3
import threading

import pytest

from agro_phenology.shadow_storage import (
    AsOfViolation,
    DecisionConflict,
    FutureOutcomeError,
    IdempotencyConflict,
    ShadowStorage,
    TimestampValidationError,
    local_day_utc_bounds,
)


UTC = timezone.utc


def _at(day: int, hour: int = 0) -> datetime:
    return datetime(2024, 6, day, hour, tzinfo=UTC)


def _observation(
    store: ShadowStorage,
    *,
    observation_id: str = "obs-1",
    observation_key: str = "weather/field-1/temperature",
    payload: dict[str, object] | None = None,
    role: str = "predictor",
    valid_from: datetime | None = None,
    day: int = 1,
    retrieval_id: str | None = None,
) -> dict[str, object]:
    return store.append_observation(
        observation_id=observation_id,
        observation_key=observation_key,
        payload=payload or {"value": 12.5},
        information_role=role,
        source_retrieval_id=retrieval_id,
        initialized_at_utc=_at(day, 0),
        published_at_utc=_at(day, 1),
        retrieved_at_utc=_at(day, 2),
        first_seen_at_utc=_at(day, 3),
        ingested_at_utc=_at(day, 4),
        valid_from_utc=valid_from or _at(day, 0),
        valid_to_utc=(valid_from or _at(day, 0)) + timedelta(hours=6),
    )


def _commit(
    store: ShadowStorage,
    *,
    slot: datetime = _at(2, 8),
    as_of: datetime = _at(2, 7),
    created: datetime = _at(2, 8),
    idempotency_key: str = "decision-1",
    expected_revision: int = 0,
    payload: dict[str, object] | None = None,
    state: dict[str, object] | None = None,
    observation_ids: tuple[str, ...] = (),
    policy_id: str = "P_growth",
    policy_version: str = "2024.1",
) -> dict[str, object]:
    return store.commit_decision(
        registry_version="registry-v1",
        policy_id=policy_id,
        policy_version=policy_version,
        field_season="field-1/2024",
        decision_slot_utc=slot,
        as_of_utc=as_of,
        created_at_utc=created,
        idempotency_key=idempotency_key,
        expected_state_revision=expected_revision,
        decision_payload=payload or {"score": 0.61, "would_notify": True},
        state_after=state or {"last_message_score": 0.61, "message_count": 1},
        observation_ids=observation_ids,
    )


def test_raw_archive_is_content_addressed_and_retrieval_is_idempotent(tmp_path) -> None:
    store = ShadowStorage(tmp_path / "shadow.sqlite")
    kwargs = {
        "retrieval_id": "retrieval-1",
        "source": "weather-provider",
        "request_key": "field-1/forecast",
        "body": b'{"temperature":12.5}',
        "media_type": "application/json",
        "initialized_at_utc": _at(1, 0),
        "published_at_utc": _at(1, 1),
        "retrieved_at_utc": _at(1, 2),
        "first_seen_at_utc": _at(1, 3),
        "ingested_at_utc": _at(1, 4),
        "valid_from_utc": _at(1, 5),
        "valid_to_utc": _at(1, 11),
        "metadata": {"status": 200},
    }
    first = store.record_retrieval(**kwargs)
    repeated = store.record_retrieval(**kwargs)
    second = store.record_retrieval(**{**kwargs, "retrieval_id": "retrieval-2"})

    assert first == repeated
    assert first["content_sha256"] == second["content_sha256"]
    assert store.read_raw_response(first["content_sha256"]) == kwargs["body"]
    assert len(store.retrievals_as_of(_at(1, 4))) == 2
    assert first["retrieval_started_at_utc"] == first["retrieved_at_utc"]
    assert first["retrieval_completed_at_utc"] == first["retrieved_at_utc"]
    assert len({
        first["initialized_at_utc"], first["published_at_utc"],
        first["first_seen_at_utc"], first["retrieved_at_utc"],
        first["ingested_at_utc"], first["valid_from_utc"],
    }) == 6

    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_responses").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError, match="raw responses are immutable"):
            connection.execute(
                "UPDATE raw_responses SET body = ? WHERE content_sha256 = ?",
                (b"changed", first["content_sha256"]),
            )
    with pytest.raises(IdempotencyConflict):
        store.record_retrieval(**{**kwargs, "body": b"different"})


def test_retrieval_start_completion_and_publication_are_distinct_fields(tmp_path) -> None:
    store = ShadowStorage(tmp_path / "shadow.sqlite")
    result = store.record_retrieval(
        retrieval_id="timed-retrieval",
        source="forecast-archive",
        request_key="synthetic-cell/run",
        body=b"{}",
        media_type="application/json",
        initialized_at_utc=_at(1, 0),
        published_at_utc=_at(1, 1),
        retrieval_started_at_utc=_at(1, 2),
        first_seen_at_utc=_at(1, 3),
        retrieval_completed_at_utc=_at(1, 3),
        retrieved_at_utc=_at(1, 3),
        ingested_at_utc=_at(1, 4),
        valid_from_utc=_at(2, 0),
        valid_to_utc=_at(2, 6),
        metadata={"run_initialization_is_publication": False},
    )
    assert result["initialized_at_utc"] != result["published_at_utc"]
    assert result["retrieval_started_at_utc"] != result["retrieval_completed_at_utc"]
    assert result["metadata"]["run_initialization_is_publication"] is False


def test_observation_log_is_versioned_and_late_safe_for_replay(tmp_path) -> None:
    store = ShadowStorage(tmp_path / "shadow.sqlite")
    first = _observation(store, payload={"value": 10.0}, valid_from=_at(1, 0), day=1)
    decision = _commit(store, observation_ids=("obs-1",))

    late = _observation(
        store,
        observation_id="obs-2",
        payload={"value": 11.0, "correction": True},
        valid_from=_at(1, 0),
        day=3,
    )
    assert (first["version"], late["version"]) == (1, 2)
    assert store.observations_as_of(_at(2, 7), valid_at_utc=_at(1, 1))[0]["payload"] == {
        "value": 10.0
    }
    assert store.observations_as_of(_at(4), valid_at_utc=_at(1, 1))[0]["payload"]["value"] == 11.0

    unchanged = store.get_decision(
        registry_version="registry-v1",
        policy_id="P_growth",
        policy_version="2024.1",
        field_season="field-1/2024",
        decision_slot_utc=_at(2, 8),
    )
    assert unchanged == decision
    bundle = store.replay_bundle(
        registry_version="registry-v1",
        policy_id="P_growth",
        policy_version="2024.1",
        field_season="field-1/2024",
        decision_slot_utc=_at(2, 8),
    )
    assert [row["observation_id"] for row in bundle["observations"]] == ["obs-1"]
    assert bundle["observations"][0]["payload"] == {"value": 10.0}
    with sqlite3.connect(store.database_path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="observations are append-only"):
            connection.execute(
                "UPDATE observations SET payload_json = '{}' WHERE observation_id = 'obs-1'"
            )


def test_decision_and_policy_state_are_atomic_and_idempotent(tmp_path) -> None:
    store = ShadowStorage(tmp_path / "shadow.sqlite")
    first = _commit(store)
    repeated = _commit(store)
    assert first == repeated
    state = store.get_policy_state(
        registry_version="registry-v1",
        policy_id="P_growth",
        policy_version="2024.1",
        field_season="field-1/2024",
    )
    assert state["revision"] == 1
    assert state["state"] == {"last_message_score": 0.61, "message_count": 1}

    with pytest.raises(DecisionConflict):
        _commit(store, idempotency_key="different-id", payload={"score": 0.2})
    assert store.get_policy_state(
        registry_version="registry-v1",
        policy_id="P_growth",
        policy_version="2024.1",
        field_season="field-1/2024",
    )["revision"] == 1


def test_scheduled_slot_can_record_later_actual_asof_without_backdating(tmp_path) -> None:
    store = ShadowStorage(tmp_path / "shadow.sqlite")
    decision = _commit(
        store,
        slot=_at(2, 5),
        as_of=_at(2, 7),
        created=_at(2, 8),
        idempotency_key="late-scheduled-slot",
        payload={
            "scheduled_for_utc": _at(2, 5).isoformat(),
            "actual_decision_at_utc": _at(2, 7).isoformat(),
            "late_run": True,
        },
    )
    assert decision["decision_slot_utc"].startswith("2024-06-02T05:00:00")
    assert decision["as_of_utc"].startswith("2024-06-02T07:00:00")
    assert decision["decision_payload"]["late_run"] is True


def test_failed_state_write_rolls_back_decision_and_retry_resumes(tmp_path) -> None:
    store = ShadowStorage(tmp_path / "shadow.sqlite")
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_policy_state BEFORE INSERT ON policy_state
            BEGIN SELECT RAISE(ABORT, 'simulated crash'); END
            """
        )
    with pytest.raises(sqlite3.IntegrityError, match="simulated crash"):
        _commit(store)

    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM policy_state").fetchone()[0] == 0
        connection.execute("DROP TRIGGER fail_policy_state")

    recovered = _commit(store)
    assert recovered["state_revision_after"] == 1


def test_concurrent_conflicting_attempts_commit_exactly_one_transition(tmp_path) -> None:
    database = tmp_path / "shadow.sqlite"
    ShadowStorage(database)
    barrier = threading.Barrier(2)

    def attempt(number: int) -> tuple[str, object]:
        store = ShadowStorage(database)
        barrier.wait()
        try:
            return "ok", _commit(
                store,
                idempotency_key=f"concurrent-{number}",
                payload={"score": 0.5 + 0.1 * number},
                state={"winner": number},
            )
        except Exception as exc:  # asserted by exact type below
            return "error", exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(attempt, (1, 2)))

    successes = [value for status, value in results if status == "ok"]
    errors = [value for status, value in results if status == "error"]
    assert len(successes) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], DecisionConflict)
    state = ShadowStorage(database).get_policy_state(
        registry_version="registry-v1",
        policy_id="P_growth",
        policy_version="2024.1",
        field_season="field-1/2024",
    )
    assert state["revision"] == 1


def test_policy_and_policy_version_have_independent_state_namespaces(tmp_path) -> None:
    store = ShadowStorage(tmp_path / "shadow.sqlite")
    _commit(store, policy_id="P0", policy_version="v1", idempotency_key="p0")
    _commit(
        store,
        policy_id="P_growth",
        policy_version="v2",
        idempotency_key="growth",
        state={"message_count": 0},
    )
    p0 = store.get_policy_state(
        registry_version="registry-v1",
        policy_id="P0",
        policy_version="v1",
        field_season="field-1/2024",
    )
    growth = store.get_policy_state(
        registry_version="registry-v1",
        policy_id="P_growth",
        policy_version="v2",
        field_season="field-1/2024",
    )
    assert p0["revision"] == growth["revision"] == 1
    assert p0["state"] != growth["state"]


def test_asof_guard_and_payload_guard_reject_future_information(tmp_path) -> None:
    store = ShadowStorage(tmp_path / "shadow.sqlite")
    _observation(store, observation_id="late-predictor", day=3)
    with pytest.raises(AsOfViolation, match="after as_of_utc"):
        _commit(store, observation_ids=("late-predictor",))

    _observation(
        store,
        observation_id="future-outcome",
        observation_key="disease/field-1/first-event",
        role="outcome",
        valid_from=_at(3),
        day=1,
    )
    with pytest.raises(FutureOutcomeError, match="future outcome observation"):
        _commit(store, idempotency_key="future-outcome-decision", observation_ids=("future-outcome",))
    with pytest.raises(FutureOutcomeError, match="days_to_first_recorded_event"):
        _commit(
            store,
            idempotency_key="embedded-outcome",
            payload={"features": {"days_to_first_recorded_event": 5}},
        )

    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0


def test_utc_validation_and_local_day_bounds_preserve_dst_23_and_25_hours(tmp_path) -> None:
    store = ShadowStorage(tmp_path / "shadow.sqlite")
    with pytest.raises(TimestampValidationError, match="timezone-aware UTC"):
        store.retrievals_as_of(datetime(2024, 6, 1, 12))
    with pytest.raises(TimestampValidationError, match="expressed in UTC"):
        store.retrievals_as_of(datetime.fromisoformat("2024-06-01T12:00:00+03:00"))

    spring_start, spring_end = local_day_utc_bounds("2024-03-31", "Europe/Berlin")
    autumn_start, autumn_end = local_day_utc_bounds("2024-10-27", "Europe/Berlin")
    assert spring_end - spring_start == timedelta(hours=23)
    assert autumn_end - autumn_start == timedelta(hours=25)
    assert spring_start.tzinfo is UTC and spring_end.tzinfo is UTC
