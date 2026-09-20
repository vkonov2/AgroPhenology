"""Durable local storage primitives for a shadow early-warning service.

The module deliberately has no network or notification code.  It persists the
inputs and decisions needed for deterministic replay while preserving the
information boundary that existed at each decision time.
"""
from __future__ import annotations

from contextlib import closing
from datetime import date, datetime, time, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo


SCHEMA_VERSION = 1


class ShadowStorageError(RuntimeError):
    """Base error raised by the shadow store."""


class TimestampValidationError(ShadowStorageError, ValueError):
    """A timestamp was naive, non-UTC, or temporally inconsistent."""


class IdempotencyConflict(ShadowStorageError):
    """An idempotency key was reused for different immutable content."""


class DecisionConflict(ShadowStorageError):
    """A decision slot already contains a different immutable decision."""


class ConcurrentStateUpdate(ShadowStorageError):
    """The policy-state revision changed before the decision was committed."""


class AsOfViolation(ShadowStorageError):
    """A decision attempted to consume information unavailable at its as-of."""


class FutureOutcomeError(AsOfViolation):
    """A decision attempted to persist a future outcome or label."""


class OutOfOrderDecision(ShadowStorageError):
    """A new policy-state transition precedes its current decision slot."""


_FORBIDDEN_DECISION_KEYS = {
    "outcome",
    "outcomes",
    "label",
    "labels",
    "target",
    "target_class",
    "target_observable",
    "first_recorded_event_date",
    "first_recorded_event_available_date",
    "days_to_first_recorded_event",
    "next_visit_date",
    "next_observation_date",
    "timely_hit",
}


def _utc_text(value: datetime | str, name: str) -> str:
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise TimestampValidationError(f"{name} is not an ISO timestamp") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TimestampValidationError(f"{name} must be datetime or ISO text")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TimestampValidationError(f"{name} must be timezone-aware UTC")
    if parsed.utcoffset().total_seconds() != 0:
        raise TimestampValidationError(f"{name} must be expressed in UTC")
    normalized = parsed.astimezone(timezone.utc)
    return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _optional_utc_text(value: datetime | str | None, name: str) -> str | None:
    return None if value is None else _utc_text(value, name)


def _json_text(value: Mapping[str, Any] | Sequence[Any]) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("payload must be finite JSON data") from exc


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_text(value).encode("utf-8")).hexdigest()


def _validate_availability_times(
    *,
    initialized_at_utc: str | None,
    published_at_utc: str | None,
    retrieval_started_at_utc: str,
    retrieval_completed_at_utc: str,
    retrieved_at_utc: str,
    first_seen_at_utc: str,
    ingested_at_utc: str,
    valid_from_utc: str,
    valid_to_utc: str | None,
) -> None:
    ordered = [
        (initialized_at_utc, published_at_utc, "initialized_at_utc", "published_at_utc"),
        (
            published_at_utc,
            retrieval_started_at_utc,
            "published_at_utc",
            "retrieval_started_at_utc",
        ),
        (
            retrieval_started_at_utc,
            retrieval_completed_at_utc,
            "retrieval_started_at_utc",
            "retrieval_completed_at_utc",
        ),
        (
            retrieval_completed_at_utc,
            first_seen_at_utc,
            "retrieval_completed_at_utc",
            "first_seen_at_utc",
        ),
        (
            retrieval_completed_at_utc,
            ingested_at_utc,
            "retrieval_completed_at_utc",
            "ingested_at_utc",
        ),
    ]
    for left, right, left_name, right_name in ordered:
        if left is not None and right is not None and left > right:
            raise TimestampValidationError(f"{left_name} must not be after {right_name}")
    if valid_to_utc is not None and valid_to_utc <= valid_from_utc:
        raise TimestampValidationError("valid_to_utc must be after valid_from_utc")


def _ensure_decision_payload_has_no_outcomes(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            if (
                key in _FORBIDDEN_DECISION_KEYS
                or key.startswith("future_")
                or key.endswith("_outcome")
                or key.endswith("_label")
            ):
                raise FutureOutcomeError(f"future/outcome field is forbidden in decision payload: {path}.{raw_key}")
            _ensure_decision_payload_has_no_outcomes(child, f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _ensure_decision_payload_has_no_outcomes(child, f"{path}[{index}]")


def local_day_utc_bounds(local_day: date | str, timezone_name: str) -> tuple[datetime, datetime]:
    """Return UTC bounds for a local calendar day, preserving 23/25-hour days."""

    parsed_day = date.fromisoformat(local_day) if isinstance(local_day, str) else local_day
    if not isinstance(parsed_day, date):
        raise TypeError("local_day must be a date or ISO date")
    zone = ZoneInfo(timezone_name)
    start_local = datetime.combine(parsed_day, time.min, tzinfo=zone)
    end_local = datetime.combine(parsed_day.fromordinal(parsed_day.toordinal() + 1), time.min, tzinfo=zone)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


class ShadowStorage:
    """Concurrency-safe SQLite storage for shadow decisions and their inputs."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        busy_timeout_seconds: float = 30.0,
        read_only: bool = False,
    ):
        self.database_path = Path(database_path)
        self.busy_timeout_seconds = float(busy_timeout_seconds)
        self.read_only = bool(read_only)
        if self.busy_timeout_seconds <= 0:
            raise ValueError("busy_timeout_seconds must be positive")
        if self.read_only:
            if not self.database_path.is_file():
                raise FileNotFoundError(self.database_path)
        else:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            uri = f"{self.database_path.resolve().as_uri()}?mode=ro"
            connection = sqlite3.connect(
                uri,
                timeout=self.busy_timeout_seconds,
                isolation_level=None,
                uri=True,
            )
            connection.execute("PRAGMA query_only = ON")
        else:
            connection = sqlite3.connect(
                self.database_path,
                timeout=self.busy_timeout_seconds,
                isolation_level=None,
            )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_seconds * 1000)}")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS raw_responses (
                    content_sha256 TEXT PRIMARY KEY,
                    body BLOB NOT NULL,
                    byte_length INTEGER NOT NULL,
                    archived_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS retrieval_records (
                    retrieval_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    request_key TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL REFERENCES raw_responses(content_sha256),
                    media_type TEXT NOT NULL,
                    initialized_at_utc TEXT,
                    published_at_utc TEXT,
                    retrieval_started_at_utc TEXT NOT NULL,
                    retrieval_completed_at_utc TEXT NOT NULL,
                    retrieved_at_utc TEXT NOT NULL,
                    first_seen_at_utc TEXT NOT NULL,
                    ingested_at_utc TEXT NOT NULL,
                    valid_from_utc TEXT NOT NULL,
                    valid_to_utc TEXT,
                    metadata_json TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS retrieval_asof_idx
                    ON retrieval_records(ingested_at_utc, source, request_key);

                CREATE TABLE IF NOT EXISTS observations (
                    observation_id TEXT PRIMARY KEY,
                    observation_key TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK(version > 0),
                    information_role TEXT NOT NULL CHECK(
                        information_role IN ('predictor', 'forecast', 'outcome', 'metadata')
                    ),
                    source_retrieval_id TEXT REFERENCES retrieval_records(retrieval_id),
                    payload_json TEXT NOT NULL,
                    initialized_at_utc TEXT,
                    published_at_utc TEXT,
                    retrieval_started_at_utc TEXT NOT NULL,
                    retrieval_completed_at_utc TEXT NOT NULL,
                    retrieved_at_utc TEXT NOT NULL,
                    first_seen_at_utc TEXT NOT NULL,
                    ingested_at_utc TEXT NOT NULL,
                    valid_from_utc TEXT NOT NULL,
                    valid_to_utc TEXT,
                    request_fingerprint TEXT NOT NULL,
                    UNIQUE(observation_key, version)
                );
                CREATE INDEX IF NOT EXISTS observation_asof_idx
                    ON observations(ingested_at_utc, observation_key, version);

                CREATE TABLE IF NOT EXISTS decisions (
                    registry_version TEXT NOT NULL,
                    policy_id TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    field_season TEXT NOT NULL,
                    decision_slot_utc TEXT NOT NULL,
                    as_of_utc TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    decision_payload_json TEXT NOT NULL,
                    state_before_json TEXT NOT NULL,
                    state_after_json TEXT NOT NULL,
                    state_revision_before INTEGER NOT NULL,
                    state_revision_after INTEGER NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    PRIMARY KEY(
                        registry_version, policy_id, policy_version,
                        field_season, decision_slot_utc
                    )
                );

                CREATE TABLE IF NOT EXISTS decision_observations (
                    registry_version TEXT NOT NULL,
                    policy_id TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    field_season TEXT NOT NULL,
                    decision_slot_utc TEXT NOT NULL,
                    observation_id TEXT NOT NULL REFERENCES observations(observation_id),
                    PRIMARY KEY(
                        registry_version, policy_id, policy_version,
                        field_season, decision_slot_utc, observation_id
                    ),
                    FOREIGN KEY(
                        registry_version, policy_id, policy_version,
                        field_season, decision_slot_utc
                    ) REFERENCES decisions(
                        registry_version, policy_id, policy_version,
                        field_season, decision_slot_utc
                    )
                );

                CREATE TABLE IF NOT EXISTS policy_state (
                    registry_version TEXT NOT NULL,
                    policy_id TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    field_season TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision >= 0),
                    state_json TEXT NOT NULL,
                    last_decision_slot_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    PRIMARY KEY(registry_version, policy_id, policy_version, field_season)
                );

                CREATE TRIGGER IF NOT EXISTS raw_responses_no_update
                BEFORE UPDATE ON raw_responses
                BEGIN SELECT RAISE(ABORT, 'raw responses are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS raw_responses_no_delete
                BEFORE DELETE ON raw_responses
                BEGIN SELECT RAISE(ABORT, 'raw responses are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS retrieval_records_no_update
                BEFORE UPDATE ON retrieval_records
                BEGIN SELECT RAISE(ABORT, 'retrieval records are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS retrieval_records_no_delete
                BEFORE DELETE ON retrieval_records
                BEGIN SELECT RAISE(ABORT, 'retrieval records are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS observations_no_update
                BEFORE UPDATE ON observations
                BEGIN SELECT RAISE(ABORT, 'observations are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS observations_no_delete
                BEFORE DELETE ON observations
                BEGIN SELECT RAISE(ABORT, 'observations are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS decisions_no_update
                BEFORE UPDATE ON decisions
                BEGIN SELECT RAISE(ABORT, 'decisions are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS decisions_no_delete
                BEFORE DELETE ON decisions
                BEGIN SELECT RAISE(ABORT, 'decisions are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS decision_observations_no_update
                BEFORE UPDATE ON decision_observations
                BEGIN SELECT RAISE(ABORT, 'decision inputs are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS decision_observations_no_delete
                BEFORE DELETE ON decision_observations
                BEGIN SELECT RAISE(ABORT, 'decision inputs are immutable'); END;
                """
            )
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in (0, SCHEMA_VERSION):
                raise ShadowStorageError(
                    f"unsupported shadow schema version {version}; expected {SCHEMA_VERSION}"
                )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @staticmethod
    def _decode_retrieval(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    @staticmethod
    def _decode_observation(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        return result

    @staticmethod
    def _decode_decision(row: sqlite3.Row, observation_ids: Sequence[str]) -> dict[str, Any]:
        result = dict(row)
        result["decision_payload"] = json.loads(result.pop("decision_payload_json"))
        result["state_before"] = json.loads(result.pop("state_before_json"))
        result["state_after"] = json.loads(result.pop("state_after_json"))
        result["observation_ids"] = list(observation_ids)
        return result

    def record_retrieval(
        self,
        *,
        retrieval_id: str,
        source: str,
        request_key: str,
        body: bytes | str,
        media_type: str,
        initialized_at_utc: datetime | str | None,
        published_at_utc: datetime | str | None,
        retrieved_at_utc: datetime | str,
        first_seen_at_utc: datetime | str,
        ingested_at_utc: datetime | str,
        valid_from_utc: datetime | str,
        valid_to_utc: datetime | str | None = None,
        metadata: Mapping[str, Any] | None = None,
        retrieval_started_at_utc: datetime | str | None = None,
        retrieval_completed_at_utc: datetime | str | None = None,
    ) -> dict[str, Any]:
        """Archive raw bytes by SHA-256 and append an immutable retrieval record."""

        raw = body.encode("utf-8") if isinstance(body, str) else bytes(body)
        digest = hashlib.sha256(raw).hexdigest()
        initialized = _optional_utc_text(initialized_at_utc, "initialized_at_utc")
        published = _optional_utc_text(published_at_utc, "published_at_utc")
        retrieved = _utc_text(retrieved_at_utc, "retrieved_at_utc")
        first_seen = _utc_text(first_seen_at_utc, "first_seen_at_utc")
        retrieval_started = _utc_text(
            retrieval_started_at_utc or retrieved,
            "retrieval_started_at_utc",
        )
        retrieval_completed = _utc_text(
            retrieval_completed_at_utc or retrieved,
            "retrieval_completed_at_utc",
        )
        if retrieval_completed != retrieved:
            raise TimestampValidationError(
                "retrieved_at_utc is the compatibility alias of retrieval_completed_at_utc"
            )
        ingested = _utc_text(ingested_at_utc, "ingested_at_utc")
        valid_from = _utc_text(valid_from_utc, "valid_from_utc")
        valid_to = _optional_utc_text(valid_to_utc, "valid_to_utc")
        _validate_availability_times(
            initialized_at_utc=initialized,
            published_at_utc=published,
            retrieval_started_at_utc=retrieval_started,
            retrieval_completed_at_utc=retrieval_completed,
            retrieved_at_utc=retrieved,
            first_seen_at_utc=first_seen,
            ingested_at_utc=ingested,
            valid_from_utc=valid_from,
            valid_to_utc=valid_to,
        )
        metadata_json = _json_text(metadata or {})
        immutable = {
            "retrieval_id": str(retrieval_id),
            "source": str(source),
            "request_key": str(request_key),
            "content_sha256": digest,
            "media_type": str(media_type),
            "initialized_at_utc": initialized,
            "published_at_utc": published,
            "retrieval_started_at_utc": retrieval_started,
            "retrieval_completed_at_utc": retrieval_completed,
            "retrieved_at_utc": retrieved,
            "first_seen_at_utc": first_seen,
            "ingested_at_utc": ingested,
            "valid_from_utc": valid_from,
            "valid_to_utc": valid_to,
            "metadata_json": metadata_json,
        }
        request_fingerprint = _fingerprint(immutable)
        archived = ingested
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO raw_responses(content_sha256, body, byte_length, archived_at_utc)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(content_sha256) DO NOTHING
                    """,
                    (digest, raw, len(raw), archived),
                )
                stored = connection.execute(
                    "SELECT body, byte_length FROM raw_responses WHERE content_sha256 = ?", (digest,)
                ).fetchone()
                if stored is None or bytes(stored["body"]) != raw or int(stored["byte_length"]) != len(raw):
                    raise ShadowStorageError("content-addressed raw archive integrity failure")
                existing = connection.execute(
                    "SELECT * FROM retrieval_records WHERE retrieval_id = ?", (str(retrieval_id),)
                ).fetchone()
                if existing is not None:
                    if existing["request_fingerprint"] != request_fingerprint:
                        raise IdempotencyConflict(
                            f"retrieval_id {retrieval_id!r} already identifies different content"
                        )
                    connection.commit()
                    return self._decode_retrieval(existing)
                connection.execute(
                    """
                    INSERT INTO retrieval_records(
                        retrieval_id, source, request_key, content_sha256, media_type,
                        initialized_at_utc, published_at_utc,
                        retrieval_started_at_utc, retrieval_completed_at_utc,
                        retrieved_at_utc,
                        first_seen_at_utc, ingested_at_utc, valid_from_utc,
                        valid_to_utc, metadata_json, request_fingerprint
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    tuple(immutable.values()) + (request_fingerprint,),
                )
                row = connection.execute(
                    "SELECT * FROM retrieval_records WHERE retrieval_id = ?", (str(retrieval_id),)
                ).fetchone()
                connection.commit()
                return self._decode_retrieval(row)
            except Exception:
                connection.rollback()
                raise

    def read_raw_response(self, content_sha256: str) -> bytes:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT body FROM raw_responses WHERE content_sha256 = ?", (content_sha256,)
            ).fetchone()
        if row is None:
            raise KeyError(content_sha256)
        body = bytes(row["body"])
        if hashlib.sha256(body).hexdigest() != content_sha256:
            raise ShadowStorageError("raw response digest mismatch")
        return body

    def retrievals_as_of(self, as_of_utc: datetime | str) -> list[dict[str, Any]]:
        as_of = _utc_text(as_of_utc, "as_of_utc")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM retrieval_records
                WHERE ingested_at_utc <= ? AND first_seen_at_utc <= ?
                  AND retrieved_at_utc <= ?
                  AND (published_at_utc IS NULL OR published_at_utc <= ?)
                  AND (initialized_at_utc IS NULL OR initialized_at_utc <= ?)
                ORDER BY ingested_at_utc, retrieval_id
                """,
                (as_of, as_of, as_of, as_of, as_of),
            ).fetchall()
        return [self._decode_retrieval(row) for row in rows]

    def append_observation(
        self,
        *,
        observation_id: str,
        observation_key: str,
        payload: Mapping[str, Any] | Sequence[Any],
        information_role: str,
        source_retrieval_id: str | None,
        initialized_at_utc: datetime | str | None,
        published_at_utc: datetime | str | None,
        retrieved_at_utc: datetime | str,
        first_seen_at_utc: datetime | str,
        ingested_at_utc: datetime | str,
        valid_from_utc: datetime | str,
        valid_to_utc: datetime | str | None = None,
        retrieval_started_at_utc: datetime | str | None = None,
        retrieval_completed_at_utc: datetime | str | None = None,
    ) -> dict[str, Any]:
        """Append an immutable observation version; retries return the same row."""

        role = str(information_role)
        if role not in {"predictor", "forecast", "outcome", "metadata"}:
            raise ValueError(f"unsupported information_role {role!r}")
        initialized = _optional_utc_text(initialized_at_utc, "initialized_at_utc")
        published = _optional_utc_text(published_at_utc, "published_at_utc")
        retrieved = _utc_text(retrieved_at_utc, "retrieved_at_utc")
        first_seen = _utc_text(first_seen_at_utc, "first_seen_at_utc")
        retrieval_started = _utc_text(
            retrieval_started_at_utc or retrieved,
            "retrieval_started_at_utc",
        )
        retrieval_completed = _utc_text(
            retrieval_completed_at_utc or retrieved,
            "retrieval_completed_at_utc",
        )
        if retrieval_completed != retrieved:
            raise TimestampValidationError(
                "retrieved_at_utc is the compatibility alias of retrieval_completed_at_utc"
            )
        ingested = _utc_text(ingested_at_utc, "ingested_at_utc")
        valid_from = _utc_text(valid_from_utc, "valid_from_utc")
        valid_to = _optional_utc_text(valid_to_utc, "valid_to_utc")
        _validate_availability_times(
            initialized_at_utc=initialized,
            published_at_utc=published,
            retrieval_started_at_utc=retrieval_started,
            retrieval_completed_at_utc=retrieval_completed,
            retrieved_at_utc=retrieved,
            first_seen_at_utc=first_seen,
            ingested_at_utc=ingested,
            valid_from_utc=valid_from,
            valid_to_utc=valid_to,
        )
        payload_json = _json_text(payload)
        immutable = {
            "observation_id": str(observation_id),
            "observation_key": str(observation_key),
            "information_role": role,
            "source_retrieval_id": source_retrieval_id,
            "payload_json": payload_json,
            "initialized_at_utc": initialized,
            "published_at_utc": published,
            "retrieval_started_at_utc": retrieval_started,
            "retrieval_completed_at_utc": retrieval_completed,
            "retrieved_at_utc": retrieved,
            "first_seen_at_utc": first_seen,
            "ingested_at_utc": ingested,
            "valid_from_utc": valid_from,
            "valid_to_utc": valid_to,
        }
        request_fingerprint = _fingerprint(immutable)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT * FROM observations WHERE observation_id = ?", (str(observation_id),)
                ).fetchone()
                if existing is not None:
                    if existing["request_fingerprint"] != request_fingerprint:
                        raise IdempotencyConflict(
                            f"observation_id {observation_id!r} already identifies another version"
                        )
                    connection.commit()
                    return self._decode_observation(existing)
                if source_retrieval_id is not None:
                    retrieval = connection.execute(
                        "SELECT retrieval_id FROM retrieval_records WHERE retrieval_id = ?",
                        (source_retrieval_id,),
                    ).fetchone()
                    if retrieval is None:
                        raise KeyError(f"unknown source retrieval {source_retrieval_id!r}")
                version = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(version), 0) + 1 FROM observations WHERE observation_key = ?",
                        (str(observation_key),),
                    ).fetchone()[0]
                )
                connection.execute(
                    """
                    INSERT INTO observations(
                        observation_id, observation_key, version, information_role,
                        source_retrieval_id, payload_json, initialized_at_utc,
                        published_at_utc, retrieval_started_at_utc,
                        retrieval_completed_at_utc, retrieved_at_utc, first_seen_at_utc,
                        ingested_at_utc, valid_from_utc, valid_to_utc, request_fingerprint
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(observation_id), str(observation_key), version, role,
                        source_retrieval_id, payload_json, initialized, published,
                        retrieval_started, retrieval_completed, retrieved,
                        first_seen, ingested, valid_from, valid_to,
                        request_fingerprint,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM observations WHERE observation_id = ?", (str(observation_id),)
                ).fetchone()
                connection.commit()
                return self._decode_observation(row)
            except Exception:
                connection.rollback()
                raise

    def observations_as_of(
        self,
        as_of_utc: datetime | str,
        *,
        valid_at_utc: datetime | str | None = None,
        latest_per_key: bool = True,
        include_outcomes: bool = True,
    ) -> list[dict[str, Any]]:
        """Read only versions that were observable by the requested as-of."""

        as_of = _utc_text(as_of_utc, "as_of_utc")
        valid_at = _optional_utc_text(valid_at_utc, "valid_at_utc")
        clauses = [
            "ingested_at_utc <= ?",
            "first_seen_at_utc <= ?",
            "retrieved_at_utc <= ?",
            "retrieval_completed_at_utc <= ?",
            "(published_at_utc IS NULL OR published_at_utc <= ?)",
            "(initialized_at_utc IS NULL OR initialized_at_utc <= ?)",
        ]
        parameters: list[Any] = [as_of, as_of, as_of, as_of, as_of, as_of]
        if valid_at is not None:
            clauses.extend(
                ["valid_from_utc <= ?", "(valid_to_utc IS NULL OR valid_to_utc > ?)"]
            )
            parameters.extend([valid_at, valid_at])
        if not include_outcomes:
            clauses.append("information_role != 'outcome'")
        query = (
            "SELECT * FROM observations WHERE "
            + " AND ".join(clauses)
            + " ORDER BY observation_key, version DESC"
        )
        with closing(self._connect()) as connection:
            rows = connection.execute(query, parameters).fetchall()
        decoded = [self._decode_observation(row) for row in rows]
        if not latest_per_key:
            return decoded
        latest: dict[str, dict[str, Any]] = {}
        for record in decoded:
            latest.setdefault(record["observation_key"], record)
        return [latest[key] for key in sorted(latest)]

    @staticmethod
    def _decision_key_values(
        registry_version: str,
        policy_id: str,
        policy_version: str,
        field_season: str,
        decision_slot_utc: str,
    ) -> tuple[str, str, str, str, str]:
        return (
            str(registry_version),
            str(policy_id),
            str(policy_version),
            str(field_season),
            decision_slot_utc,
        )

    def get_policy_state(
        self,
        *,
        registry_version: str,
        policy_id: str,
        policy_version: str,
        field_season: str,
    ) -> dict[str, Any]:
        key = (str(registry_version), str(policy_id), str(policy_version), str(field_season))
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT * FROM policy_state
                WHERE registry_version = ? AND policy_id = ? AND policy_version = ?
                  AND field_season = ?
                """,
                key,
            ).fetchone()
        if row is None:
            return {
                "registry_version": key[0],
                "policy_id": key[1],
                "policy_version": key[2],
                "field_season": key[3],
                "revision": 0,
                "state": {},
                "last_decision_slot_utc": None,
                "updated_at_utc": None,
            }
        result = dict(row)
        result["state"] = json.loads(result.pop("state_json"))
        return result

    def commit_decision(
        self,
        *,
        registry_version: str,
        policy_id: str,
        policy_version: str,
        field_season: str,
        decision_slot_utc: datetime | str,
        as_of_utc: datetime | str,
        created_at_utc: datetime | str,
        idempotency_key: str,
        expected_state_revision: int,
        decision_payload: Mapping[str, Any],
        state_after: Mapping[str, Any],
        observation_ids: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Atomically append an immutable decision and advance one policy state."""

        _ensure_decision_payload_has_no_outcomes(decision_payload)
        slot = _utc_text(decision_slot_utc, "decision_slot_utc")
        as_of = _utc_text(as_of_utc, "as_of_utc")
        created = _utc_text(created_at_utc, "created_at_utc")
        if created < as_of:
            raise TimestampValidationError("created_at_utc must not be before as_of_utc")
        if int(expected_state_revision) < 0:
            raise ValueError("expected_state_revision must be non-negative")
        decision_json = _json_text(decision_payload)
        state_after_json = _json_text(state_after)
        observation_keys = sorted(set(str(value) for value in observation_ids))
        key = self._decision_key_values(
            registry_version, policy_id, policy_version, field_season, slot
        )
        immutable_request = {
            "registry_version": key[0],
            "policy_id": key[1],
            "policy_version": key[2],
            "field_season": key[3],
            "decision_slot_utc": key[4],
            "as_of_utc": as_of,
            "idempotency_key": str(idempotency_key),
            "expected_state_revision": int(expected_state_revision),
            "decision_payload_json": decision_json,
            "state_after_json": state_after_json,
            "observation_ids": observation_keys,
        }
        request_fingerprint = _fingerprint(immutable_request)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing_by_idempotency = connection.execute(
                    "SELECT * FROM decisions WHERE idempotency_key = ?", (str(idempotency_key),)
                ).fetchone()
                existing_by_slot = connection.execute(
                    """
                    SELECT * FROM decisions
                    WHERE registry_version = ? AND policy_id = ? AND policy_version = ?
                      AND field_season = ? AND decision_slot_utc = ?
                    """,
                    key,
                ).fetchone()
                existing = existing_by_idempotency or existing_by_slot
                if existing is not None:
                    same_slot = tuple(existing[column] for column in (
                        "registry_version", "policy_id", "policy_version",
                        "field_season", "decision_slot_utc"
                    )) == key
                    if not same_slot or existing["request_fingerprint"] != request_fingerprint:
                        if existing_by_slot is not None:
                            raise DecisionConflict("decision slot already contains a different decision")
                        raise IdempotencyConflict(
                            f"idempotency_key {idempotency_key!r} already identifies another decision"
                        )
                    ids = [
                        row[0]
                        for row in connection.execute(
                            """
                            SELECT observation_id FROM decision_observations
                            WHERE registry_version = ? AND policy_id = ? AND policy_version = ?
                              AND field_season = ? AND decision_slot_utc = ?
                            ORDER BY observation_id
                            """,
                            key,
                        ).fetchall()
                    ]
                    connection.commit()
                    return self._decode_decision(existing, ids)

                state_key = key[:4]
                current = connection.execute(
                    """
                    SELECT * FROM policy_state
                    WHERE registry_version = ? AND policy_id = ? AND policy_version = ?
                      AND field_season = ?
                    """,
                    state_key,
                ).fetchone()
                current_revision = int(current["revision"]) if current is not None else 0
                if current_revision != int(expected_state_revision):
                    raise ConcurrentStateUpdate(
                        f"expected state revision {expected_state_revision}, found {current_revision}"
                    )
                if current is not None and current["last_decision_slot_utc"] >= slot:
                    raise OutOfOrderDecision(
                        "decision_slot_utc must be after the current policy-state slot"
                    )
                state_before_json = current["state_json"] if current is not None else "{}"
                if observation_keys:
                    placeholders = ",".join("?" for _ in observation_keys)
                    observations = connection.execute(
                        f"SELECT * FROM observations WHERE observation_id IN ({placeholders})",
                        observation_keys,
                    ).fetchall()
                    found = {row["observation_id"] for row in observations}
                    missing = sorted(set(observation_keys) - found)
                    if missing:
                        raise KeyError(f"unknown decision observations: {missing}")
                    for observation in observations:
                        for name in (
                            "initialized_at_utc", "published_at_utc",
                            "retrieval_started_at_utc", "retrieval_completed_at_utc",
                            "retrieved_at_utc",
                            "first_seen_at_utc", "ingested_at_utc",
                        ):
                            value = observation[name]
                            if value is not None and value > as_of:
                                raise AsOfViolation(
                                    f"observation {observation['observation_id']} has {name} after as_of_utc"
                                )
                        if (
                            observation["information_role"] == "outcome"
                            and observation["valid_from_utc"] > as_of
                        ):
                            raise FutureOutcomeError(
                                f"future outcome observation {observation['observation_id']} is forbidden"
                            )

                revision_after = current_revision + 1
                connection.execute(
                    """
                    INSERT INTO decisions(
                        registry_version, policy_id, policy_version, field_season,
                        decision_slot_utc, as_of_utc, created_at_utc, idempotency_key,
                        decision_payload_json, state_before_json, state_after_json,
                        state_revision_before, state_revision_after, request_fingerprint
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    key
                    + (
                        as_of, created, str(idempotency_key), decision_json,
                        state_before_json, state_after_json, current_revision,
                        revision_after, request_fingerprint,
                    ),
                )
                for observation_id in observation_keys:
                    connection.execute(
                        """
                        INSERT INTO decision_observations(
                            registry_version, policy_id, policy_version, field_season,
                            decision_slot_utc, observation_id
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        key + (observation_id,),
                    )
                connection.execute(
                    """
                    INSERT INTO policy_state(
                        registry_version, policy_id, policy_version, field_season,
                        revision, state_json, last_decision_slot_utc, updated_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(registry_version, policy_id, policy_version, field_season)
                    DO UPDATE SET revision = excluded.revision,
                                  state_json = excluded.state_json,
                                  last_decision_slot_utc = excluded.last_decision_slot_utc,
                                  updated_at_utc = excluded.updated_at_utc
                    """,
                    state_key + (revision_after, state_after_json, slot, created),
                )
                row = connection.execute(
                    """
                    SELECT * FROM decisions
                    WHERE registry_version = ? AND policy_id = ? AND policy_version = ?
                      AND field_season = ? AND decision_slot_utc = ?
                    """,
                    key,
                ).fetchone()
                connection.commit()
                return self._decode_decision(row, observation_keys)
            except Exception:
                connection.rollback()
                raise

    def get_decision(
        self,
        *,
        registry_version: str,
        policy_id: str,
        policy_version: str,
        field_season: str,
        decision_slot_utc: datetime | str,
    ) -> dict[str, Any]:
        slot = _utc_text(decision_slot_utc, "decision_slot_utc")
        key = self._decision_key_values(
            registry_version, policy_id, policy_version, field_season, slot
        )
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT * FROM decisions
                WHERE registry_version = ? AND policy_id = ? AND policy_version = ?
                  AND field_season = ? AND decision_slot_utc = ?
                """,
                key,
            ).fetchone()
            if row is None:
                raise KeyError(key)
            observation_ids = [
                item[0]
                for item in connection.execute(
                    """
                    SELECT observation_id FROM decision_observations
                    WHERE registry_version = ? AND policy_id = ? AND policy_version = ?
                      AND field_season = ? AND decision_slot_utc = ?
                    ORDER BY observation_id
                    """,
                    key,
                ).fetchall()
            ]
        return self._decode_decision(row, observation_ids)

    def replay_bundle(
        self,
        *,
        registry_version: str,
        policy_id: str,
        policy_version: str,
        field_season: str,
        decision_slot_utc: datetime | str,
    ) -> dict[str, Any]:
        """Return an immutable decision plus the exact input versions it used."""

        decision = self.get_decision(
            registry_version=registry_version,
            policy_id=policy_id,
            policy_version=policy_version,
            field_season=field_season,
            decision_slot_utc=decision_slot_utc,
        )
        observations: list[dict[str, Any]] = []
        retrievals: dict[str, dict[str, Any]] = {}
        with closing(self._connect()) as connection:
            for observation_id in decision["observation_ids"]:
                row = connection.execute(
                    "SELECT * FROM observations WHERE observation_id = ?", (observation_id,)
                ).fetchone()
                observation = self._decode_observation(row)
                observations.append(observation)
                retrieval_id = observation["source_retrieval_id"]
                if retrieval_id is not None and retrieval_id not in retrievals:
                    retrieval_row = connection.execute(
                        "SELECT * FROM retrieval_records WHERE retrieval_id = ?", (retrieval_id,)
                    ).fetchone()
                    retrievals[retrieval_id] = self._decode_retrieval(retrieval_row)
        return {
            "decision": decision,
            "observations": observations,
            "retrievals": [retrievals[key] for key in sorted(retrievals)],
        }
