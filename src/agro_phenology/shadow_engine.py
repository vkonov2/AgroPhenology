"""Offline, sequential runtime for the frozen late-blight shadow contour.

The runtime is intentionally small and conservative.  It has no HTTP client,
message-delivery adapter, scheduler, or model-training path.  A run consumes an
explicit current field registry plus immutable observations already present in
``ShadowStorage``.  Every virtual message is persisted with ``actually_sent``
set to ``False``.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from .early_warning_core import CALENDAR_FEATURES, EPISODE_FEATURES
from .early_warning_cycle2_models import load_c6_bundle
from .early_warning_cycle3_policy import (
    GrowthPolicy,
    simulate_growth_policy,
)
from .shadow_registry import canonical_sha256, verify_shadow_registry
from .shadow_sources import (
    FROZEN_ERA5_PROFILE_ID,
    LOCAL_DAY_TIMEZONE,
    assert_source_available_as_of,
    build_episode_feature_snapshot,
    parse_frozen_era5_hourly,
    sanitised_request_contract,
)
from .shadow_storage import ShadowStorage


FIELD_REGISTRY_SCHEMA_VERSION = "1.0.0"
DECISION_SCHEMA_VERSION = "1.0.0"
WEATHER_OBSERVATION_KIND = "frozen_era5_hourly_archive"
WEATHER_OBSERVATION_PREFIX = "weather/era5-hourly/"
EXACT_ERA5_SOURCE = "open_meteo_archive_era5"
PROSPECTIVE_MODE = "prospective_live"
REPLAY_MODE = "retrospective_replay"
DEMO_REGISTRY_PURPOSE = "technical_demo_synthetic"
ACTIVE_REGISTRY_PURPOSE = "prospective_active_fields"


class ShadowEngineError(RuntimeError):
    """Base error for the shadow runtime."""


class FieldRegistryError(ShadowEngineError, ValueError):
    """The supplied field registry is absent, stale, or structurally unsafe."""


class FrozenArtifactError(ShadowEngineError):
    """A frozen model cannot be loaded under its registered contract."""


@dataclass(frozen=True)
class ScoreSnapshot:
    model_id: str
    score: float | None
    probabilities: tuple[float, float, float] | None
    score_origin: str
    score_status: str
    input_status: str
    input_reason: str
    model_version: str
    alpha: float | None
    features: Mapping[str, float | None]
    observation_ids: tuple[str, ...] = ()
    used_c0_fallback: bool = False
    alpha_zero_identity: bool = False


def _read_json(value: str | Path | Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"{name} is missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} root must be an object")
    return payload


def _utc_datetime(value: str | datetime, *, name: str) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO timestamp") from exc
    else:
        parsed = value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def load_active_field_registry(
    source: str | Path | Mapping[str, Any],
    *,
    decision_at_utc: str | datetime,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    """Validate an explicit field registry without inferring historical fields."""

    registry = _read_json(source, name="field registry")
    declared_hash = registry.get("field_registry_content_sha256")
    unhashed_registry = {
        key: value
        for key, value in registry.items()
        if key != "field_registry_content_sha256"
    }
    actual_hash = canonical_sha256(unhashed_registry)
    if declared_hash is not None and declared_hash != actual_hash:
        raise FieldRegistryError("field registry content hash mismatch")
    if registry.get("schema_version") != FIELD_REGISTRY_SCHEMA_VERSION:
        raise FieldRegistryError(
            f"field registry schema must be {FIELD_REGISTRY_SCHEMA_VERSION}"
        )
    purpose = registry.get("registry_purpose")
    allowed = {ACTIVE_REGISTRY_PURPOSE}
    if allow_synthetic:
        allowed.add(DEMO_REGISTRY_PURPOSE)
    if purpose not in allowed:
        raise FieldRegistryError(
            "field registry must explicitly declare registry_purpose="
            f"{ACTIVE_REGISTRY_PURPOSE!r}"
        )
    if registry.get("historical_coordinates_used") is not False:
        raise FieldRegistryError(
            "historical_coordinates_used must be explicitly false"
        )
    if registry.get("timezone") != LOCAL_DAY_TIMEZONE:
        raise FieldRegistryError(
            f"field registry timezone must be {LOCAL_DAY_TIMEZONE}"
        )
    generated = _utc_datetime(registry.get("generated_at_utc", ""), name="generated_at_utc")
    available = _utc_datetime(
        registry.get("available_at_utc", ""), name="available_at_utc"
    )
    ingested = _utc_datetime(
        registry.get("ingested_at_utc", ""), name="ingested_at_utc"
    )
    decision = _utc_datetime(decision_at_utc, name="decision_at_utc")
    if not generated <= available <= ingested <= decision:
        raise FieldRegistryError(
            "field registry requires generated_at <= available_at <= ingested_at <= decision_at"
        )

    fields = registry.get("fields")
    if not isinstance(fields, list) or not fields:
        raise FieldRegistryError("field registry has no fields")
    forbidden_keys = {
        "latitude",
        "longitude",
        "address",
        "cadastre",
        "cadastral_number",
        "owner_name",
        "phone",
    }
    seen: set[str] = set()
    for index, field in enumerate(fields):
        if not isinstance(field, dict):
            raise FieldRegistryError(f"fields[{index}] must be an object")
        leaked = sorted(forbidden_keys.intersection(field))
        if leaked:
            raise FieldRegistryError(
                f"fields[{index}] contains protected direct fields: {leaked}"
            )
        required = {
            "field_pseudo_id",
            "season_id",
            "field_season",
            "season",
            "crop",
            "region_code",
            "enrolled_from",
            "enrolled_until",
            "status",
            "weather_location_ref",
        }
        missing = sorted(required.difference(field))
        if missing:
            raise FieldRegistryError(f"fields[{index}] misses {missing}")
        if field["crop"] != "potato":
            raise FieldRegistryError(f"fields[{index}] is not a potato field-season")
        if field["status"] not in {"active", "closed", "paused"}:
            raise FieldRegistryError(f"fields[{index}] has unsupported status")
        key = str(field["field_season"])
        if key in seen:
            raise FieldRegistryError(f"duplicate field_season {key!r}")
        seen.add(key)
        try:
            date.fromisoformat(str(field["enrolled_from"]))
            if field.get("enrolled_until") is not None:
                date.fromisoformat(str(field["enrolled_until"]))
            int(field["season"])
        except (TypeError, ValueError) as exc:
            raise FieldRegistryError(f"fields[{index}] has invalid season/service dates") from exc
    registry["field_registry_content_sha256"] = actual_hash
    return registry


def write_field_registry(
    destination: str | Path, payload: Mapping[str, Any]
) -> Path:
    """Write a registry only to a new path; mainly useful for controlled setup."""

    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite field registry: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def _field_in_service(field: Mapping[str, Any], issue_day: date) -> bool:
    if field.get("status") != "active":
        return False
    start = date.fromisoformat(str(field["enrolled_from"]))
    raw_end = field.get("enrolled_until")
    end = date.max if raw_end is None else date.fromisoformat(str(raw_end))
    return start <= issue_day <= end


def _calendar_features(issue_day: date) -> dict[str, float]:
    doy = issue_day.timetuple().tm_yday
    values: dict[str, float] = {}
    for harmonic in (1, 2):
        values[f"doy_sin{harmonic}"] = float(
            np.sin(2 * np.pi * harmonic * doy / 365.25)
        )
        values[f"doy_cos{harmonic}"] = float(
            np.cos(2 * np.pi * harmonic * doy / 365.25)
        )
    return values


def _actionable_probabilities(model: Any, features: pd.DataFrame) -> tuple[float, float, float]:
    values = np.asarray(model.predict_proba(features), dtype=float)
    if values.shape[0] != 1:
        raise FrozenArtifactError("frozen model returned an unexpected row count")
    classes = tuple(int(value) for value in model.classes_)
    if set(classes) != {0, 1, 2}:
        raise FrozenArtifactError(f"frozen model classes are incompatible: {classes}")
    aligned = values[0, [classes.index(value) for value in (0, 1, 2)]]
    if not np.isfinite(aligned).all() or not np.isclose(aligned.sum(), 1.0):
        raise FrozenArtifactError("frozen model returned invalid probabilities")
    return tuple(float(value) for value in aligned)  # type: ignore[return-value]


def _model_artifact_path(project_root: Path, entry: Mapping[str, Any]) -> Path:
    candidates = [
        project_root / item["relative_path"]
        for item in entry["artifacts"]
        if str(item["relative_path"]).endswith((".joblib", ".cbm"))
    ]
    if len(candidates) != 1:
        raise FrozenArtifactError(
            f"{entry['model_id']} must identify exactly one direct model artifact"
        )
    return candidates[0]


def _bundle_directory(project_root: Path, entry: Mapping[str, Any]) -> Path:
    candidates = [
        project_root / item["relative_path"]
        for item in entry["artifacts"]
        if str(item["relative_path"]).endswith("bundle.json")
    ]
    if len(candidates) != 1:
        raise FrozenArtifactError(
            f"{entry['model_id']} must identify exactly one C6 bundle"
        )
    return candidates[0].parent


class FrozenScorer:
    """Load and score only artifacts named by a verified frozen registry."""

    def __init__(self, project_root: str | Path, registry: Mapping[str, Any]):
        self.project_root = Path(project_root).resolve()
        self.registry = dict(registry)
        self.entries = {str(row["model_id"]): row for row in registry["models"]}
        self._models: dict[str, Any] = {}

    def _load(self, model_id: str) -> Any:
        if model_id in self._models:
            return self._models[model_id]
        entry = self.entries[model_id]
        if model_id.startswith("C6_"):
            model = load_c6_bundle(_bundle_directory(self.project_root, entry))
        else:
            path = _model_artifact_path(self.project_root, entry)
            if path.suffix == ".cbm":
                from catboost import CatBoostClassifier

                model = CatBoostClassifier()
                model.load_model(path)
            else:
                model = joblib.load(path)
        self._models[model_id] = model
        return model

    def verify_loadable(self) -> dict[str, Any]:
        loaded: list[str] = []
        for model_id in self.entries:
            if model_id == "calendar_window":
                loaded.append(model_id)
                continue
            model = self._load(model_id)
            if model_id.startswith("C6_"):
                classes = tuple(int(value) for value in model.base_model.classes_)
            else:
                classes = tuple(int(value) for value in model.classes_)
            if set(classes) != {0, 1, 2}:
                raise FrozenArtifactError(f"{model_id} has incompatible classes {classes}")
            loaded.append(model_id)
        return {"status": "passed", "models_loaded": loaded}

    def score(
        self,
        model_id: str,
        *,
        calendar: Mapping[str, float],
        episode: Mapping[str, float] | None,
        weather_status: str,
        weather_reason: str,
        observation_ids: Sequence[str],
        issue_day: date,
    ) -> ScoreSnapshot:
        entry = self.entries[model_id]
        version = str(entry["registry_entry_sha256"])
        if model_id == "calendar_window":
            lower = int(entry["rule"]["lower_day_of_year_inclusive"])
            upper = int(entry["rule"]["upper_day_of_year_inclusive"])
            score = float(lower <= issue_day.timetuple().tm_yday <= upper)
            probabilities = (1.0 - score, 0.0, score)
            return ScoreSnapshot(
                model_id, score, probabilities, "calendar_window", "computed",
                "available", "calendar_only", version, None, dict(calendar)
            )

        if model_id in {"C0", "C1"}:
            model = self._load(model_id)
            frame = pd.DataFrame([{name: calendar[name] for name in CALENDAR_FEATURES}])
            probabilities = _actionable_probabilities(model, frame)
            return ScoreSnapshot(
                model_id, probabilities[2], probabilities,
                "calendar_features", "computed", "available", "calendar_only",
                version, None, dict(calendar)
            )

        if model_id in {"C4", "C5"}:
            if episode is None:
                return ScoreSnapshot(
                    model_id, None, None, "unavailable_exact_era5_episode", "abstained",
                    weather_status, weather_reason, version, None,
                    {**calendar, **{name: None for name in EPISODE_FEATURES}},
                    tuple(observation_ids),
                )
            features = {**calendar, **episode}
            model = self._load(model_id)
            frame = pd.DataFrame([{name: features[name] for name in entry["features_in_order"]}])
            probabilities = _actionable_probabilities(model, frame)
            return ScoreSnapshot(
                model_id, probabilities[2], probabilities, "exact_era5_episode",
                "computed", "available", "exact_frozen_source_profile", version,
                None, features, tuple(observation_ids)
            )

        if model_id in {"C6_weather", "C6_calibration_control"}:
            bundle = self._load(model_id)
            features: dict[str, float | None] = dict(calendar)
            if model_id == "C6_weather":
                features.update(
                    episode if episode is not None else {name: None for name in EPISODE_FEATURES}
                )
                features["episode_weather_complete"] = episode is not None
            frame = pd.DataFrame([features])
            prediction = bundle.predict(frame)
            probabilities = tuple(float(value) for value in prediction.probabilities[0])
            fallback = bool(prediction.used_c0_fallback[0]) if model_id == "C6_weather" else False
            alpha_zero = float(entry["alpha"]) == 0.0
            origin = (
                "c0_fallback_missing_exact_era5"
                if fallback
                else "c0_alpha_zero_identity"
                if alpha_zero
                else "exact_era5_episode"
            )
            status = "computed_c0_fallback" if fallback else "computed"
            return ScoreSnapshot(
                model_id, probabilities[2], probabilities, origin, status,
                weather_status if fallback else "available",
                weather_reason if fallback else "exact_frozen_source_profile",
                version, float(entry["alpha"]), features,
                tuple(observation_ids) if model_id == "C6_weather" else (),
                fallback,
                alpha_zero,
            )
        raise KeyError(model_id)


def _retrieval_contract_compatible(record: Mapping[str, Any]) -> tuple[bool, str]:
    metadata = record.get("metadata", {})
    required = {
        "profile_id": FROZEN_ERA5_PROFILE_ID,
        "model_parameter": "era5",
        "product": "ERA5",
        "response_timezone": "UTC",
        "cell_selection": "nearest",
        "statistical_downscaling": False,
        "archive_only_forecast": False,
    }
    if record.get("source") != EXACT_ERA5_SOURCE:
        return False, "provider_or_product_not_exact_era5"
    for key, expected in required.items():
        if metadata.get(key) != expected:
            return False, f"retrieval_metadata_mismatch:{key}"
    if str(metadata.get("elevation_parameter")) != "nan":
        return False, "retrieval_metadata_mismatch:elevation_parameter"
    request_profile = metadata.get("request_profile")
    if not isinstance(request_profile, Mapping):
        return False, "missing_immutable_request_profile"
    return True, "compatible"


def resolve_episode_weather(
    store: ShadowStorage,
    *,
    weather_cell: str | None,
    issue_day: date,
    decision_at_utc: datetime,
) -> tuple[dict[str, float] | None, dict[str, Any], tuple[str, ...]]:
    """Resolve the newest compatible saved ERA5 observation available as-of."""

    if not weather_cell:
        return None, {
            "status": "missing_weather_cell_mapping",
            "reason": "active_field_registry_has_no_weather_cell",
            "required_last_local_date": str(issue_day - timedelta(days=2)),
        }, ()
    observations = store.observations_as_of(
        decision_at_utc, latest_per_key=False, include_outcomes=False
    )
    key = f"{WEATHER_OBSERVATION_PREFIX}{weather_cell}"
    candidates = [
        row
        for row in observations
        if row["observation_key"] == key
        and row["information_role"] == "predictor"
        and row["payload"].get("kind") == WEATHER_OBSERVATION_KIND
        and row["payload"].get("weather_cell") == weather_cell
        and row["payload"].get("archive_only_forecast") is False
    ]
    if not candidates:
        return None, {
            "status": "missing_saved_exact_era5",
            "reason": "no_predictor_observation_available_as_of_decision",
            "required_last_local_date": str(issue_day - timedelta(days=2)),
        }, ()
    retrievals = {
        row["retrieval_id"]: row for row in store.retrievals_as_of(decision_at_utc)
    }
    failures: list[str] = []
    for observation in candidates:
        retrieval_id = observation.get("source_retrieval_id")
        retrieval = retrievals.get(retrieval_id)
        if retrieval is None:
            failures.append("retrieval_not_available_as_of_decision")
            continue
        compatible, reason = _retrieval_contract_compatible(retrieval)
        if not compatible:
            failures.append(reason)
            continue
        if observation["payload"].get("profile_id") != FROZEN_ERA5_PROFILE_ID:
            failures.append("observation_profile_mismatch")
            continue
        try:
            assert_source_available_as_of(
                retrieval_completed_at=retrieval["retrieved_at_utc"],
                first_seen_at=retrieval["first_seen_at_utc"],
                ingested_at=retrieval["ingested_at_utc"],
                decision_at=_utc_text(decision_at_utc),
            )
            raw = store.read_raw_response(str(retrieval["content_sha256"]))
            daily, parsed_metadata = parse_frozen_era5_hourly(
                raw,
                weather_cell=weather_cell,
                request_profile=retrieval["metadata"]["request_profile"],
            )
            snapshot = build_episode_feature_snapshot(
                daily, weather_cell=weather_cell, issue_local_date=issue_day
            )
        except Exception as exc:
            failures.append(f"invalid_saved_response:{type(exc).__name__}:{exc}")
            continue
        if not snapshot["complete"]:
            failures.append(str(snapshot["status"]))
            continue
        status = {
            "status": "available",
            "reason": "exact_saved_era5_complete_at_t_minus_2",
            "profile_id": FROZEN_ERA5_PROFILE_ID,
            "retrieval_id": retrieval_id,
            "content_sha256": retrieval["content_sha256"],
            "required_last_local_date": snapshot["required_last_local_date"],
            "available_last_local_date": snapshot["available_last_local_date"],
            "first_seen_at_utc": retrieval["first_seen_at_utc"],
            "retrieved_at_utc": retrieval["retrieved_at_utc"],
            "ingested_at_utc": retrieval["ingested_at_utc"],
            "initialized_at_utc": retrieval["initialized_at_utc"],
            "published_at_utc": retrieval["published_at_utc"],
            "valid_time_min": parsed_metadata["valid_time_min"],
            "valid_time_max": parsed_metadata["valid_time_max"],
        }
        return (
            {name: float(snapshot["features"][name]) for name in EPISODE_FEATURES},
            status,
            (str(observation["observation_id"]),),
        )
    return None, {
        "status": "incompatible_or_incomplete_saved_era5",
        "reason": failures[0] if failures else "unknown_source_failure",
        "failures": failures,
        "required_last_local_date": str(issue_day - timedelta(days=2)),
    }, ()


def _growth_policy(payload: Mapping[str, Any]) -> GrowthPolicy:
    return GrowthPolicy(
        threshold=float(payload["threshold"]),
        active_days=int(payload["active_days"]),
        cooldown_days=int(payload["cooldown_days"]),
        minimum_repeat_interval_days=int(payload["minimum_repeat_interval_days"]),
        growth_override_enabled=bool(payload["growth_override_enabled"]),
        growth_logit_delta=(
            None
            if payload.get("growth_logit_delta") is None
            else float(payload["growth_logit_delta"])
        ),
        epsilon=float(payload["logit_epsilon"]),
        version=str(payload["configuration_sha256"]),
    )


def evaluate_one_policy_transition(
    *,
    field_season: str,
    season: int,
    issue_day: date,
    decision_at_utc: datetime,
    service_active: bool,
    score: float | None,
    score_origin: str,
    model_id: str,
    model_version: str,
    policy_payload: Mapping[str, Any],
    state_before: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run exactly one frozen policy transition from a persisted checkpoint."""

    frame = pd.DataFrame(
        [
            {
                "field_season": field_season,
                "season": int(season),
                "issue_date": pd.Timestamp(issue_day),
                "issued_at": _utc_text(decision_at_utc),
                "service_active": bool(service_active),
                "evaluation_field_day": bool(service_active),
                "target_class": "unknown",
                "target_observable": False,
                "days_to_first_recorded_event": np.nan,
                "warnable_first_event": False,
                "coordinate_scope": "prospective_pseudonymous_registry",
                "previous_visit_gap_days": np.nan,
                "common_weather_complete": False,
            }
        ]
    )
    initial = {field_season: dict(state_before or {})}
    result, states = simulate_growth_policy(
        frame,
        pd.Series([np.nan if score is None else float(score)]),
        _growth_policy(policy_payload),
        score_origin=score_origin,
        model_id=model_id,
        model_version=model_version,
        initial_states=initial,
    )
    row = result.iloc[0]
    transition = {
        "shadow_message_would_be_issued": bool(row["message_issued"]),
        "message_kind": str(row["message_kind"]),
        "alarm_active": bool(row["alarm_active"]),
        "action_reason": str(row["action_reason"]),
        "suppressed_repeat": bool(row["suppressed_repeat"]),
        "threshold_reached": bool(row["threshold_reached"]),
        "ordinary_cooldown_satisfied": bool(row["ordinary_cooldown_satisfied"]),
        "growth_minimum_interval_satisfied": bool(
            row["growth_minimum_interval_satisfied"]
        ),
        "growth_override_used": bool(row["growth_override_used"]),
        "growth_reference_comparable": bool(row["growth_reference_comparable"]),
        "growth_reference_status": str(row["growth_reference_status"]),
        "score_comparison_segment_id": (
            None if pd.isna(row["score_comparison_segment_id"]) else int(row["score_comparison_segment_id"])
        ),
        "score_comparison_segment_started": bool(row["score_comparison_segment_started"]),
        "previous_message_date": (
            None if pd.isna(row["previous_message_date"]) else str(pd.Timestamp(row["previous_message_date"]).date())
        ),
        "previous_message_score": (
            None if pd.isna(row["previous_message_score"]) else float(row["previous_message_score"])
        ),
        "previous_message_origin": (
            None if pd.isna(row["previous_message_origin"]) else str(row["previous_message_origin"])
        ),
        "elapsed_calendar_days_since_message": (
            None
            if pd.isna(row["elapsed_calendar_days_since_message"])
            else int(row["elapsed_calendar_days_since_message"])
        ),
        "logit_growth_from_previous_message": (
            None
            if pd.isna(row["logit_growth_from_previous_message"])
            else float(row["logit_growth_from_previous_message"])
        ),
        "cumulative_messages_field_season": int(row["cumulative_messages_field_season"]),
        "cumulative_alarm_days_field_season": int(row["cumulative_alarm_days_field_season"]),
        "active_from": None if pd.isna(row["active_from"]) else str(pd.Timestamp(row["active_from"]).date()),
        "active_through": None if pd.isna(row["active_through"]) else str(pd.Timestamp(row["active_through"]).date()),
    }
    return transition, states[field_season].to_json_dict()


def _decision_action(service_active: bool, score: ScoreSnapshot, transition: Mapping[str, Any]) -> str:
    if not service_active:
        return "not_in_service"
    if score.score is None:
        return "abstain"
    if transition["shadow_message_would_be_issued"]:
        return "message_candidate"
    if transition["suppressed_repeat"]:
        return "message_suppressed"
    return "no_message"


def _scheduled_time(issue_day: date, registry: Mapping[str, Any]) -> datetime:
    local_clock = time.fromisoformat(str(registry["scheduled_local_time"]))
    local = datetime.combine(issue_day, local_clock, tzinfo=ZoneInfo(LOCAL_DAY_TIMEZONE))
    return local.astimezone(timezone.utc)


def _archive_field_registry_input(
    store: ShadowStorage,
    registry: Mapping[str, Any],
    *,
    received_at_utc: datetime,
) -> tuple[str, str]:
    """Persist the exact pseudonymised registry bytes used by a decision."""

    content_hash = str(registry["field_registry_content_sha256"])
    source_payload = {
        key: value
        for key, value in registry.items()
        if key != "field_registry_content_sha256"
    }
    raw = json.dumps(
        source_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != content_hash:
        raise FieldRegistryError("canonical field registry bytes have an unexpected hash")
    short = content_hash[:24]
    retrieval_id = f"field-registry-retrieval-{short}"
    observation_id = f"field-registry-observation-{short}"
    received = _utc_text(received_at_utc)
    existing_retrievals = {
        row["retrieval_id"]: row for row in store.retrievals_as_of(received_at_utc)
    }
    retrieval = existing_retrievals.get(retrieval_id)
    if retrieval is None:
        retrieval = store.record_retrieval(
            retrieval_id=retrieval_id,
            source="active_field_registry",
            request_key=f"field-registry/{registry.get('registry_id')}",
            body=raw,
            media_type="application/json",
            initialized_at_utc=registry["generated_at_utc"],
            published_at_utc=None,
            retrieval_started_at_utc=received,
            retrieval_completed_at_utc=received,
            retrieved_at_utc=received,
            first_seen_at_utc=received,
            ingested_at_utc=received,
            valid_from_utc=registry["generated_at_utc"],
            valid_to_utc=None,
            metadata={
                "kind": "pseudonymised_active_field_registry",
                "registry_id": registry.get("registry_id"),
                "registry_purpose": registry["registry_purpose"],
                "field_registry_content_sha256": content_hash,
                "contains_exact_coordinates": False,
                "source_declared_available_at_utc": registry["available_at_utc"],
                "source_declared_ingested_at_utc": registry["ingested_at_utc"],
                "availability_evidence": "actual_run_once_read",
            },
        )
    existing_observations = {
        row["observation_id"]: row
        for row in store.observations_as_of(
            received_at_utc, latest_per_key=False, include_outcomes=False
        )
    }
    if observation_id not in existing_observations:
        observed_at = retrieval["retrieval_completed_at_utc"]
        store.append_observation(
            observation_id=observation_id,
            observation_key=f"field-registry/{registry.get('registry_id')}",
            payload={
                "kind": "pseudonymised_active_field_registry",
                "registry_id": registry.get("registry_id"),
                "field_registry_content_sha256": content_hash,
                "source_declared_available_at_utc": registry["available_at_utc"],
                "source_declared_ingested_at_utc": registry["ingested_at_utc"],
            },
            information_role="metadata",
            source_retrieval_id=retrieval_id,
            initialized_at_utc=registry["generated_at_utc"],
            published_at_utc=None,
            retrieval_started_at_utc=retrieval["retrieval_started_at_utc"],
            retrieval_completed_at_utc=observed_at,
            retrieved_at_utc=observed_at,
            first_seen_at_utc=retrieval["first_seen_at_utc"],
            ingested_at_utc=retrieval["ingested_at_utc"],
            valid_from_utc=registry["generated_at_utc"],
            valid_to_utc=None,
        )
    return observation_id, content_hash


def _recent_policy_burden(
    store: ShadowStorage,
    *,
    registry_version: str,
    policy_id: str,
    policy_version: str,
    field_season: str,
    issue_day: date,
    current_transition: Mapping[str, Any],
    current_service_active: bool,
    policy_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure the soft validation budget over up to 30 observed field-days.

    This is monitoring only.  It never suppresses a frozen-policy action.
    """

    with closing(sqlite3.connect(store.database_path)) as connection:
        rows = connection.execute(
            """
            SELECT decision_payload_json FROM decisions
            WHERE registry_version=? AND policy_id=? AND policy_version=?
              AND field_season=?
            ORDER BY decision_slot_utc DESC
            """,
            (registry_version, policy_id, policy_version, field_season),
        ).fetchall()
    lower = issue_day - timedelta(days=29)
    payloads = []
    for row in rows:
        payload = json.loads(row[0])
        prior_day = date.fromisoformat(payload["issue_local_date"])
        if lower <= prior_day < issue_day and payload.get("service_active"):
            payloads.append(payload)
    field_days = len(payloads) + int(current_service_active)
    messages = sum(
        bool(row["transition"]["shadow_message_would_be_issued"])
        for row in payloads
    ) + int(
        current_service_active
        and bool(current_transition["shadow_message_would_be_issued"])
    )
    alarm_days = sum(bool(row["transition"]["alarm_active"]) for row in payloads) + int(
        current_service_active and bool(current_transition["alarm_active"])
    )
    message_rate = 0.0 if field_days == 0 else float(messages * 30.0 / field_days)
    alarm_fraction = 0.0 if field_days == 0 else float(alarm_days / field_days)
    budget = policy_payload["research_budget"]
    return {
        "window_local_date_start": str(lower),
        "window_local_date_end": str(issue_day),
        "field_days": field_days,
        "virtual_messages": messages,
        "alarm_days": alarm_days,
        "messages_per_30_field_days": message_rate,
        "active_alarm_fraction": alarm_fraction,
        "messages_budget_exceeded": bool(
            message_rate > float(budget["messages_per_30_field_days_max"])
        ),
        "alarm_budget_exceeded": bool(
            alarm_fraction > float(budget["active_alarm_fraction_max"])
        ),
        "monitoring_only_no_hard_limiter": True,
    }


def run_once(
    *,
    project_root: str | Path,
    registry_path: str | Path,
    field_registry: str | Path | Mapping[str, Any],
    database_path: str | Path,
    decision_at_utc: str | datetime,
    mode: str = PROSPECTIVE_MODE,
    allow_synthetic: bool = False,
    created_at_utc: str | datetime | None = None,
) -> dict[str, Any]:
    """Execute one real-time slot from saved inputs, without network or delivery."""

    if mode not in {PROSPECTIVE_MODE, REPLAY_MODE}:
        raise ValueError(f"unsupported run mode {mode!r}")
    root = Path(project_root).resolve()
    verification = verify_shadow_registry(registry_path, root)
    registry = _read_json(registry_path, name="shadow registry")
    decision_at = _utc_datetime(decision_at_utc, name="decision_at_utc")
    created_at = (
        datetime.now(timezone.utc)
        if created_at_utc is None
        else _utc_datetime(created_at_utc, name="created_at_utc")
    )
    if created_at < decision_at:
        raise ValueError("created_at_utc cannot precede the actual decision time")
    issue_day = decision_at.astimezone(ZoneInfo(LOCAL_DAY_TIMEZONE)).date()
    scheduled_at = _scheduled_time(issue_day, registry)
    if mode == PROSPECTIVE_MODE:
        today = created_at.astimezone(ZoneInfo(LOCAL_DAY_TIMEZONE)).date()
        if issue_day != today:
            raise ValueError(
                "prospective_live cannot be backdated or future-dated; use retrospective_replay"
            )
        if allow_synthetic:
            raise ValueError("synthetic fields cannot be used in prospective_live")
        registry_locked = _utc_datetime(
            registry["locked_at_utc"], name="shadow_registry.locked_at_utc"
        )
        if decision_at < registry_locked:
            raise ValueError(
                "prospective_live decision precedes the frozen registry lock; "
                "the first honest slot must occur after locked_at_utc"
            )
        if scheduled_at < registry_locked:
            raise ValueError(
                "prospective_live scheduled slot precedes the frozen registry lock; "
                "a late run cannot turn that slot into a live decision"
            )
        if decision_at < scheduled_at:
            raise ValueError(
                "prospective_live cannot create today's scheduled slot before 08:00 Europe/Riga"
            )
    fields = load_active_field_registry(
        field_registry, decision_at_utc=decision_at, allow_synthetic=allow_synthetic
    )
    store = ShadowStorage(database_path)
    field_registry_observation_id, field_registry_raw_hash = _archive_field_registry_input(
        store, fields, received_at_utc=decision_at
    )
    scorer = FrozenScorer(root, registry)
    scorer.verify_loadable()
    calendar = _calendar_features(issue_day)
    registry_version = str(registry["registry_content_sha256"])
    field_registry_hash = str(fields["field_registry_content_sha256"])
    late_seconds = max(0.0, (decision_at - scheduled_at).total_seconds())

    decisions: list[dict[str, Any]] = []
    created_count = 0
    existing_count = 0
    for field in fields["fields"]:
        service_active = _field_in_service(field, issue_day)
        episode, weather, weather_observation_ids = resolve_episode_weather(
            store,
            weather_cell=field.get("weather_location_ref"),
            issue_day=issue_day,
            decision_at_utc=decision_at,
        )
        weather_status = str(weather["status"])
        weather_reason = str(weather["reason"])
        for entry in registry["models"]:
            model_id = str(entry["model_id"])
            score = scorer.score(
                model_id,
                calendar=calendar,
                episode=episode,
                weather_status=weather_status,
                weather_reason=weather_reason,
                observation_ids=weather_observation_ids,
                issue_day=issue_day,
            )
            for policy in entry["policies"]:
                policy_id = str(policy["policy_id"])
                policy_version = str(policy["configuration_sha256"])
                try:
                    existing = store.get_decision(
                        registry_version=registry_version,
                        policy_id=policy_id,
                        policy_version=policy_version,
                        field_season=str(field["field_season"]),
                        decision_slot_utc=scheduled_at,
                    )
                except KeyError:
                    existing = None
                if existing is not None:
                    existing_count += 1
                    decisions.append(existing)
                    continue

                state_record = store.get_policy_state(
                    registry_version=registry_version,
                    policy_id=policy_id,
                    policy_version=policy_version,
                    field_season=str(field["field_season"]),
                )
                transition, state_after = evaluate_one_policy_transition(
                    field_season=str(field["field_season"]),
                    season=int(field["season"]),
                    issue_day=issue_day,
                    decision_at_utc=decision_at,
                    service_active=service_active,
                    score=score.score if service_active else None,
                    score_origin=score.score_origin,
                    model_id=model_id,
                    model_version=score.model_version,
                    policy_payload=policy,
                    state_before=state_record["state"],
                )
                burden = _recent_policy_burden(
                    store,
                    registry_version=registry_version,
                    policy_id=policy_id,
                    policy_version=policy_version,
                    field_season=str(field["field_season"]),
                    issue_day=issue_day,
                    current_transition=transition,
                    current_service_active=service_active,
                    policy_payload=policy,
                )
                action = _decision_action(service_active, score, transition)
                feature_payload = {
                    key: (None if value is None else float(value))
                    for key, value in score.features.items()
                    if key != "episode_weather_complete"
                }
                feature_hash = canonical_sha256(feature_payload)
                message_id = (
                    hashlib.sha256(
                        f"{registry_version}|{policy_id}|{field['field_season']}|{_utc_text(scheduled_at)}".encode()
                    ).hexdigest()[:32]
                    if transition["shadow_message_would_be_issued"]
                    else None
                )
                decision_payload = {
                    "schema_version": DECISION_SCHEMA_VERSION,
                    "run_mode": mode,
                    "delivery_mode": "shadow",
                    "actually_sent": False,
                    "delivery_attempted": False,
                    "registry_id": registry["registry_id"],
                    "registry_content_sha256": registry_version,
                    "field_registry_id": fields.get("registry_id"),
                    "field_registry_content_sha256": field_registry_hash,
                    "field_pseudo_id": str(field["field_pseudo_id"]),
                    "season_id": str(field["season_id"]),
                    "region_code": str(field["region_code"]),
                    "season": int(field["season"]),
                    "issue_local_date": str(issue_day),
                    "data_timezone": LOCAL_DAY_TIMEZONE,
                    "scheduled_for_utc": _utc_text(scheduled_at),
                    "actual_decision_at_utc": _utc_text(decision_at),
                    "late_run": bool(decision_at > scheduled_at),
                    "late_by_seconds": float(late_seconds),
                    "service_active": bool(service_active),
                    "model_id": model_id,
                    "model_version": score.model_version,
                    "class_order": [0, 1, 2],
                    "actionable_class": 2,
                    "score": score.score if service_active else None,
                    "probabilities": (
                        None
                        if score.probabilities is None or not service_active
                        else list(score.probabilities)
                    ),
                    "score_origin": score.score_origin,
                    "score_status": score.score_status if service_active else "not_in_service",
                    "used_c0_fallback": score.used_c0_fallback,
                    "alpha_zero_identity": score.alpha_zero_identity,
                    "alpha": score.alpha,
                    "input_status": score.input_status,
                    "input_reason": score.input_reason,
                    "weather_availability": weather,
                    "input_observation_ids": [
                        field_registry_observation_id,
                        *score.observation_ids,
                    ],
                    "feature_snapshot": feature_payload,
                    "feature_snapshot_sha256": feature_hash,
                    "input_hashes": {
                        "shadow_registry": registry_version,
                        "field_registry": field_registry_hash,
                        "field_registry_raw_content": field_registry_raw_hash,
                        "feature_snapshot": feature_hash,
                        "model_artifact_set": entry["artifact_set_sha256"],
                        "raw_response_content": (
                            weather.get("content_sha256")
                            if score.observation_ids
                            else None
                        ),
                    },
                    "policy": {
                        key: policy[key]
                        for key in (
                            "policy_id",
                            "family",
                            "threshold",
                            "active_days",
                            "cooldown_days",
                            "minimum_repeat_interval_days",
                            "growth_override_enabled",
                            "growth_logit_delta",
                            "logit_epsilon",
                            "configuration_sha256",
                            "research_budget",
                        )
                    },
                    "decision_action": action,
                    "decision_reason": transition["action_reason"],
                    "threshold": float(policy["threshold"]),
                    "virtual_message_issued": bool(
                        transition["shadow_message_would_be_issued"]
                    ),
                    "message_id": message_id,
                    "transition": transition,
                    "rolling_soft_budget_monitor": burden,
                    "forecast_window_start_local_date": str(issue_day + timedelta(days=3)),
                    "forecast_window_end_local_date": str(issue_day + timedelta(days=10)),
                    "research_only": True,
                }
                idempotency_key = hashlib.sha256(
                    f"{registry_version}|{policy_id}|{policy_version}|{field['field_season']}|{_utc_text(scheduled_at)}".encode()
                ).hexdigest()
                committed = store.commit_decision(
                    registry_version=registry_version,
                    policy_id=policy_id,
                    policy_version=policy_version,
                    field_season=str(field["field_season"]),
                    decision_slot_utc=scheduled_at,
                    as_of_utc=decision_at,
                    created_at_utc=created_at,
                    idempotency_key=idempotency_key,
                    expected_state_revision=int(state_record["revision"]),
                    decision_payload=decision_payload,
                    state_after=state_after,
                    observation_ids=(field_registry_observation_id, *score.observation_ids),
                )
                created_count += 1
                decisions.append(committed)
    payloads = [row["decision_payload"] for row in decisions]
    return {
        "status": "completed",
        "run_mode": mode,
        "network_used": False,
        "notifications_sent": 0,
        "registry_verification": verification,
        "field_registry_id": fields.get("registry_id"),
        "field_registry_content_sha256": field_registry_hash,
        "decision_at_utc": _utc_text(decision_at),
        "scheduled_slot_utc": _utc_text(scheduled_at),
        "issue_local_date": str(issue_day),
        "fields": len(fields["fields"]),
        "decisions": len(decisions),
        "decisions_created": created_count,
        "decisions_already_present": existing_count,
        "computed_scores": sum(row["score"] is not None for row in payloads),
        "c0_fallback_scores": sum(bool(row["used_c0_fallback"]) for row in payloads),
        "abstentions": sum(row["decision_action"] == "abstain" for row in payloads),
        "virtual_message_candidates": sum(
            row["decision_action"] == "message_candidate" for row in payloads
        ),
        "weather_branch_decisions": sum(
            row["model_id"] in {"C4", "C5", "C6_weather"}
            and row["score_origin"] == "exact_era5_episode"
            and row["score"] is not None
            for row in payloads
        ),
    }


def _decision_rows(database_path: str | Path) -> list[dict[str, str]]:
    path = Path(database_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    store = ShadowStorage(path, read_only=True)
    with closing(store._connect()) as connection:
        rows = connection.execute(
            """
            SELECT registry_version, policy_id, policy_version, field_season,
                   decision_slot_utc
            FROM decisions
            ORDER BY registry_version, policy_id, policy_version,
                     field_season, decision_slot_utc
            """
        ).fetchall()
    return [dict(row) for row in rows]


def _replay_one(
    bundle: Mapping[str, Any],
    *,
    scorer: FrozenScorer | None = None,
    store: ShadowStorage | None = None,
) -> list[str]:
    decision = bundle["decision"]
    payload = decision["decision_payload"]
    errors: list[str] = []
    transition, state_after = evaluate_one_policy_transition(
        field_season=str(decision["field_season"]),
        season=int(payload["season"]),
        issue_day=date.fromisoformat(str(payload["issue_local_date"])),
        decision_at_utc=_utc_datetime(payload["actual_decision_at_utc"], name="actual_decision_at_utc"),
        service_active=bool(payload["service_active"]),
        score=(None if payload["score"] is None or not payload["service_active"] else float(payload["score"])),
        score_origin=str(payload["score_origin"]),
        model_id=str(payload["model_id"]),
        model_version=str(payload["model_version"]),
        policy_payload=payload["policy"],
        state_before=decision["state_before"],
    )
    if canonical_sha256(transition) != canonical_sha256(payload["transition"]):
        errors.append("transition_mismatch")
    if canonical_sha256(state_after) != canonical_sha256(decision["state_after"]):
        errors.append("state_after_mismatch")
    if payload.get("actually_sent") is not False or payload.get("delivery_attempted") is not False:
        errors.append("delivery_guard_failed")
    if canonical_sha256(payload["feature_snapshot"]) != payload["feature_snapshot_sha256"]:
        errors.append("feature_snapshot_hash_mismatch")
    if sorted(payload.get("input_observation_ids", [])) != sorted(
        decision["observation_ids"]
    ):
        errors.append("input_observation_links_mismatch")
    weather_observations = [
        row
        for row in bundle["observations"]
        if row["payload"].get("kind") == WEATHER_OBSERVATION_KIND
    ]
    field_registry_observations = [
        row
        for row in bundle["observations"]
        if row["payload"].get("kind") == "pseudonymised_active_field_registry"
    ]
    retrieval_by_id = {row["retrieval_id"]: row for row in bundle["retrievals"]}
    if store is not None:
        if len(field_registry_observations) != 1:
            errors.append("field_registry_replay_requires_one_linked_observation")
        else:
            registry_observation = field_registry_observations[0]
            registry_retrieval = retrieval_by_id.get(
                registry_observation["source_retrieval_id"]
            )
            try:
                if registry_retrieval is None:
                    raise ValueError("linked field registry retrieval missing")
                raw_registry = store.read_raw_response(
                    str(registry_retrieval["content_sha256"])
                )
                if (
                    registry_retrieval["content_sha256"]
                    != payload["input_hashes"]["field_registry_raw_content"]
                ):
                    raise ValueError("field registry raw hash mismatch")
                decoded_registry = json.loads(raw_registry.decode("utf-8"))
                if canonical_sha256(decoded_registry) != payload["field_registry_content_sha256"]:
                    raise ValueError("field registry canonical hash mismatch")
                matching_fields = [
                    row
                    for row in decoded_registry["fields"]
                    if str(row["field_season"]) == str(decision["field_season"])
                ]
                if len(matching_fields) != 1:
                    raise ValueError("decision field-season is absent from linked registry")
                if str(matching_fields[0]["field_pseudo_id"]) != str(payload["field_pseudo_id"]):
                    raise ValueError("decision field pseudonym differs from linked registry")
            except Exception as exc:
                errors.append(f"linked_field_registry_replay_failed:{type(exc).__name__}:{exc}")

    if weather_observations and store is not None:
        if len(weather_observations) != 1:
            errors.append("weather_replay_requires_one_linked_observation")
        else:
            observation = weather_observations[0]
            retrieval = retrieval_by_id.get(observation["source_retrieval_id"])
            try:
                if retrieval is None:
                    raise ValueError("linked weather retrieval missing")
                if payload["input_hashes"]["raw_response_content"] != retrieval["content_sha256"]:
                    raise ValueError("raw input hash mismatch")
                assert_source_available_as_of(
                    retrieval_completed_at=retrieval["retrieval_completed_at_utc"],
                    first_seen_at=retrieval["first_seen_at_utc"],
                    ingested_at=retrieval["ingested_at_utc"],
                    decision_at=payload["actual_decision_at_utc"],
                )
                raw = store.read_raw_response(str(retrieval["content_sha256"]))
                weather_cell = str(observation["payload"]["weather_cell"])
                daily, _ = parse_frozen_era5_hourly(
                    raw,
                    weather_cell=weather_cell,
                    request_profile=retrieval["metadata"]["request_profile"],
                )
                episode_snapshot = build_episode_feature_snapshot(
                    daily,
                    weather_cell=weather_cell,
                    issue_local_date=payload["issue_local_date"],
                )
                if not episode_snapshot["complete"]:
                    raise ValueError("linked weather no longer yields complete features")
                for name in EPISODE_FEATURES:
                    if not np.isclose(
                        float(episode_snapshot["features"][name]),
                        float(payload["feature_snapshot"][name]),
                        rtol=0.0,
                        atol=1e-12,
                    ):
                        raise ValueError(f"episode feature mismatch: {name}")
            except Exception as exc:
                errors.append(f"linked_raw_episode_replay_failed:{type(exc).__name__}:{exc}")
    elif payload.get("score_origin") == "exact_era5_episode":
        errors.append("exact_weather_score_has_no_linked_input")
    if scorer is not None and payload["service_active"]:
        feature_snapshot = payload["feature_snapshot"]
        calendar = {
            name: float(feature_snapshot[name]) for name in CALENDAR_FEATURES
        }
        complete_episode = all(
            feature_snapshot.get(name) is not None for name in EPISODE_FEATURES
        )
        episode = (
            {name: float(feature_snapshot[name]) for name in EPISODE_FEATURES}
            if complete_episode
            else None
        )
        recomputed = scorer.score(
            str(payload["model_id"]),
            calendar=calendar,
            episode=episode,
            weather_status=str(payload["weather_availability"]["status"]),
            weather_reason=str(payload["weather_availability"]["reason"]),
            observation_ids=tuple(
                str(row["observation_id"]) for row in weather_observations
            ),
            issue_day=date.fromisoformat(str(payload["issue_local_date"])),
        )
        saved_score = payload["score"]
        if (saved_score is None) != (recomputed.score is None):
            errors.append("recomputed_score_availability_mismatch")
        elif saved_score is not None and not np.isclose(
            float(saved_score), float(recomputed.score), rtol=0.0, atol=1e-12
        ):
            errors.append("recomputed_score_mismatch")
        if payload["score_origin"] != recomputed.score_origin:
            errors.append("recomputed_score_origin_mismatch")
        if bool(payload["used_c0_fallback"]) != recomputed.used_c0_fallback:
            errors.append("recomputed_fallback_mismatch")
    return errors


def replay_from_log(
    *, database_path: str | Path, registry_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    """Replay every saved policy transition without changing the database."""

    database_path = Path(database_path)
    database_sha256_before = hashlib.sha256(database_path.read_bytes()).hexdigest()
    registry = _read_json(registry_path, name="shadow registry")
    verify_shadow_registry(registry_path, project_root)
    registry_version = str(registry["registry_content_sha256"])
    store = ShadowStorage(database_path, read_only=True)
    scorer = FrozenScorer(project_root, registry)
    scorer.verify_loadable()
    failures: list[dict[str, Any]] = []
    checked = 0
    for key in _decision_rows(database_path):
        if key["registry_version"] != registry_version:
            failures.append({**key, "errors": ["unregistered_registry_version"]})
            continue
        bundle = store.replay_bundle(**key)
        errors = _replay_one(bundle, scorer=scorer, store=store)
        checked += 1
        if errors:
            failures.append({**key, "errors": errors})
    database_sha256_after = hashlib.sha256(database_path.read_bytes()).hexdigest()
    database_changed = database_sha256_before != database_sha256_after
    if database_changed:
        failures.append({"errors": ["database_changed_during_replay"]})
    return {
        "status": "passed" if not failures else "failed",
        "decisions_checked": checked,
        "failures": failures,
        "database_sha256_before": database_sha256_before,
        "database_sha256_after": database_sha256_after,
        "database_mutated": database_changed,
    }


def verify_log(
    *, database_path: str | Path, registry_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    """Verify SQLite integrity, raw hashes, state chains, and deterministic replay."""

    database_path = Path(database_path)
    database_sha256_before = hashlib.sha256(database_path.read_bytes()).hexdigest()
    replay = replay_from_log(
        database_path=database_path,
        registry_path=registry_path,
        project_root=project_root,
    )
    raw_failures: list[str] = []
    state_failures: list[str] = []
    store = ShadowStorage(database_path, read_only=True)
    with closing(store._connect()) as connection:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_keys = [tuple(row) for row in connection.execute("PRAGMA foreign_key_check")]
        for row in connection.execute(
            "SELECT content_sha256, body, byte_length FROM raw_responses"
        ):
            body = bytes(row["body"])
            if hashlib.sha256(body).hexdigest() != row["content_sha256"]:
                raw_failures.append(str(row["content_sha256"]))
            if len(body) != int(row["byte_length"]):
                raw_failures.append(f"{row['content_sha256']}:byte_length")
        groups = connection.execute(
            """
            SELECT DISTINCT registry_version, policy_id, policy_version, field_season
            FROM decisions
            """
        ).fetchall()
        for group in groups:
            key = tuple(group)
            rows = connection.execute(
                """
                SELECT state_before_json, state_after_json,
                       state_revision_before, state_revision_after
                FROM decisions
                WHERE registry_version=? AND policy_id=? AND policy_version=? AND field_season=?
                ORDER BY decision_slot_utc
                """,
                key,
            ).fetchall()
            previous = "{}"
            revision = 0
            for row in rows:
                if canonical_sha256(json.loads(row["state_before_json"])) != canonical_sha256(json.loads(previous)):
                    state_failures.append(f"{key}:state_chain")
                if int(row["state_revision_before"]) != revision or int(row["state_revision_after"]) != revision + 1:
                    state_failures.append(f"{key}:revision_chain")
                previous = row["state_after_json"]
                revision += 1
            current = connection.execute(
                """
                SELECT revision, state_json FROM policy_state
                WHERE registry_version=? AND policy_id=? AND policy_version=? AND field_season=?
                """,
                key,
            ).fetchone()
            if current is None or int(current["revision"]) != revision or canonical_sha256(json.loads(current["state_json"])) != canonical_sha256(json.loads(previous)):
                state_failures.append(f"{key}:checkpoint")
    database_sha256_after = hashlib.sha256(database_path.read_bytes()).hexdigest()
    database_changed = database_sha256_before != database_sha256_after
    passed = (
        integrity == "ok"
        and not foreign_keys
        and not raw_failures
        and not state_failures
        and replay["status"] == "passed"
        and not database_changed
    )
    return {
        "status": "passed" if passed else "failed",
        "sqlite_integrity": integrity,
        "foreign_key_failures": foreign_keys,
        "raw_hash_failures": raw_failures,
        "state_chain_failures": state_failures,
        "replay": replay,
        "database_sha256_before": database_sha256_before,
        "database_sha256_after": database_sha256_after,
        "database_mutated": database_changed,
        "trust_boundary": "local_sqlite_constraints_and_hashes_without_external_anchor",
    }


def readiness(
    *,
    project_root: str | Path,
    registry_path: str | Path,
    database_path: str | Path | None = None,
    field_registry: str | Path | Mapping[str, Any] | None = None,
    decision_at_utc: str | datetime | None = None,
) -> dict[str, Any]:
    """Return a machine-readable readiness assessment without network access."""

    root = Path(project_root).resolve()
    registry_verification = verify_shadow_registry(registry_path, root)
    registry = _read_json(registry_path, name="shadow registry")
    loadability = FrozenScorer(root, registry).verify_loadable()
    decision_at = (
        datetime.now(timezone.utc)
        if decision_at_utc is None
        else _utc_datetime(decision_at_utc, name="decision_at_utc")
    )
    blockers: list[str] = []
    field_summary: dict[str, Any]
    weather_summary: list[dict[str, Any]] = []
    database_status = "not_supplied"
    if field_registry is None:
        blockers.append("missing_actual_active_field_registry")
        field_summary = {"status": "missing", "fields": 0}
    else:
        try:
            fields = load_active_field_registry(
                field_registry, decision_at_utc=decision_at, allow_synthetic=False
            )
            field_summary = {"status": "available", "fields": len(fields["fields"])}
            if database_path is not None:
                if not Path(database_path).is_file():
                    database_status = "missing"
                    blockers.append("missing_shadow_database_for_saved_inputs")
                else:
                    database_status = "available"
                    store = ShadowStorage(database_path)
                    issue_day = decision_at.astimezone(ZoneInfo(LOCAL_DAY_TIMEZONE)).date()
                    for field in fields["fields"]:
                        _, weather, _ = resolve_episode_weather(
                            store,
                            weather_cell=field.get("weather_location_ref"),
                            issue_day=issue_day,
                            decision_at_utc=decision_at,
                        )
                        weather_summary.append(
                            {
                                "field_season": field["field_season"],
                                "status": weather["status"],
                                "reason": weather["reason"],
                            }
                        )
        except Exception as exc:
            blockers.append(f"invalid_actual_active_field_registry:{type(exc).__name__}:{exc}")
            field_summary = {"status": "invalid", "fields": 0}
    if field_summary["fields"] and database_path is None:
        blockers.append("missing_shadow_database_path_for_saved_inputs")
    if weather_summary and not any(row["status"] == "available" for row in weather_summary):
        blockers.append("no_exact_era5_t_minus_2_input_available_as_of_decision")
    return {
        "status": "ready_for_live_run_once" if not blockers else "partially_ready",
        "network_used": False,
        "notifications_configured": False,
        "notifications_sent": 0,
        "registry": registry_verification,
        "models": loadability,
        "field_registry": field_summary,
        "database_status": database_status,
        "weather_by_field": weather_summary,
        "blockers": blockers,
        "calendar_and_c0_runtime_ready": True,
        "c6_weather_runtime_ready_when_exact_saved_input_exists": True,
        "c6_uses_c0_fallback_when_weather_unavailable": True,
        "c4_c5_abstain_when_weather_unavailable": True,
    }


def archive_exact_era5_observation(
    store: ShadowStorage,
    *,
    retrieval_id: str,
    observation_id: str,
    weather_cell: str,
    body: bytes | str,
    retrieved_at_utc: str | datetime,
    valid_from_utc: str | datetime,
    valid_to_utc: str | datetime,
    synthetic_fixture: bool = False,
) -> dict[str, Any]:
    """Archive already obtained bytes under the exact frozen-source contract."""

    retrieved = _utc_datetime(retrieved_at_utc, name="retrieved_at_utc")
    valid_start = _utc_datetime(valid_from_utc, name="valid_from_utc")
    valid_end = _utc_datetime(valid_to_utc, name="valid_to_utc")
    request_profile = sanitised_request_contract(
        start_date=valid_start.astimezone(ZoneInfo(LOCAL_DAY_TIMEZONE)).date(),
        end_date=(valid_end - timedelta(microseconds=1))
        .astimezone(ZoneInfo(LOCAL_DAY_TIMEZONE))
        .date(),
        location_ref=weather_cell,
    )
    retrieval = store.record_retrieval(
        retrieval_id=retrieval_id,
        source=EXACT_ERA5_SOURCE,
        request_key=f"era5-hourly/{weather_cell}",
        body=body,
        media_type="application/json",
        initialized_at_utc=None,
        published_at_utc=None,
        retrieved_at_utc=retrieved,
        first_seen_at_utc=retrieved,
        ingested_at_utc=retrieved,
        valid_from_utc=valid_from_utc,
        valid_to_utc=valid_to_utc,
        metadata={
            "profile_id": FROZEN_ERA5_PROFILE_ID,
            "model_parameter": "era5",
            "product": "ERA5",
            "response_timezone": "UTC",
            "cell_selection": "nearest",
            "elevation_parameter": "nan",
            "statistical_downscaling": False,
            "archive_only_forecast": False,
            "availability_evidence": "technical_fixture" if synthetic_fixture else "actual_retrieval",
            "synthetic_fixture": bool(synthetic_fixture),
            "request_profile": request_profile,
        },
    )
    observation = store.append_observation(
        observation_id=observation_id,
        observation_key=f"{WEATHER_OBSERVATION_PREFIX}{weather_cell}",
        payload={
            "kind": WEATHER_OBSERVATION_KIND,
            "profile_id": FROZEN_ERA5_PROFILE_ID,
            "weather_cell": weather_cell,
            "archive_only_forecast": False,
            "synthetic_fixture": bool(synthetic_fixture),
        },
        information_role="predictor",
        source_retrieval_id=retrieval_id,
        initialized_at_utc=None,
        published_at_utc=None,
        retrieved_at_utc=retrieved,
        first_seen_at_utc=retrieved,
        ingested_at_utc=retrieved,
        valid_from_utc=valid_from_utc,
        valid_to_utc=valid_to_utc,
    )
    return {"retrieval": retrieval, "observation": observation}


def _synthetic_era5_payload(start_local_day: date, end_local_day: date) -> bytes:
    zone = ZoneInfo(LOCAL_DAY_TIMEZONE)
    start = datetime.combine(start_local_day, time.min, tzinfo=zone).astimezone(timezone.utc)
    end = datetime.combine(end_local_day + timedelta(days=1), time.min, tzinfo=zone).astimezone(timezone.utc)
    index = pd.date_range(start, end, freq="h", inclusive="left")
    count = len(index)
    hourly = {
        "time": [stamp.strftime("%Y-%m-%dT%H:%M") for stamp in index],
        "temperature_2m": [16.0] * count,
        "relative_humidity_2m": [92.0] * count,
        "precipitation": [0.12] * count,
        "soil_temperature_0_to_7cm": [15.0] * count,
        "soil_moisture_0_to_7cm": [0.32] * count,
        "shortwave_radiation": [120.0] * count,
        "wind_speed_10m": [7.0] * count,
    }
    payload = {
        "latitude": 0.0,
        "longitude": 0.0,
        "elevation": 0.0,
        "timezone": "UTC",
        "hourly_units": {
            "time": "iso8601",
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "precipitation": "mm",
            "soil_temperature_0_to_7cm": "°C",
            "soil_moisture_0_to_7cm": "m³/m³",
            "shortwave_radiation": "W/m²",
            "wind_speed_10m": "km/h",
        },
        "hourly": hourly,
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def technical_demo(
    *,
    project_root: str | Path,
    registry_path: str | Path,
    database_path: str | Path,
    created_at_utc: str | datetime | None = None,
) -> dict[str, Any]:
    """Run an offline fixture demonstration; this is never a prospective run."""

    database = Path(database_path)
    if database.exists():
        raise FileExistsError(f"Refusing to overwrite demo database: {database}")
    store = ShadowStorage(database)
    executed_at = (
        datetime.now(timezone.utc)
        if created_at_utc is None
        else _utc_datetime(created_at_utc, name="created_at_utc")
    )
    field_registry = {
        "schema_version": FIELD_REGISTRY_SCHEMA_VERSION,
        "registry_id": "synthetic-shadow-demo-v1",
        "registry_purpose": DEMO_REGISTRY_PURPOSE,
        "generated_at_utc": "2026-07-14T00:00:00Z",
        "available_at_utc": "2026-07-14T01:00:00Z",
        "ingested_at_utc": "2026-07-14T01:01:00Z",
        "timezone": LOCAL_DAY_TIMEZONE,
        "historical_coordinates_used": False,
        "fields": [
            {
                "field_pseudo_id": "synthetic-field-weather",
                "season_id": "synthetic-season-weather-2026",
                "field_season": "synthetic-field-weather/2026",
                "season": 2026,
                "crop": "potato",
                "region_code": "SYNTHETIC",
                "enrolled_from": "2026-07-01",
                "enrolled_until": None,
                "status": "active",
                "weather_location_ref": "synthetic-cell-weather",
            },
            {
                "field_pseudo_id": "synthetic-field-fallback",
                "season_id": "synthetic-season-fallback-2026",
                "field_season": "synthetic-field-fallback/2026",
                "season": 2026,
                "crop": "potato",
                "region_code": "SYNTHETIC",
                "enrolled_from": "2026-07-01",
                "enrolled_until": None,
                "status": "active",
                "weather_location_ref": "synthetic-cell-missing",
            },
        ],
    }
    first_body = _synthetic_era5_payload(date(2026, 6, 15), date(2026, 7, 13))
    first_start = datetime(2026, 6, 14, 21, tzinfo=timezone.utc)
    first_end = datetime(2026, 7, 13, 21, tzinfo=timezone.utc)
    archive_exact_era5_observation(
        store,
        retrieval_id="synthetic-era5-retrieval-1",
        observation_id="synthetic-era5-observation-1",
        weather_cell="synthetic-cell-weather",
        body=first_body,
        retrieved_at_utc="2026-07-15T04:00:00Z",
        valid_from_utc=first_start,
        valid_to_utc=first_end,
        synthetic_fixture=True,
    )
    first = run_once(
        project_root=project_root,
        registry_path=registry_path,
        field_registry=field_registry,
        database_path=database,
        decision_at_utc="2026-07-15T05:00:00Z",
        created_at_utc=executed_at,
        mode=REPLAY_MODE,
        allow_synthetic=True,
    )
    retry = run_once(
        project_root=project_root,
        registry_path=registry_path,
        field_registry=field_registry,
        database_path=database,
        decision_at_utc="2026-07-15T05:30:00Z",
        created_at_utc=executed_at,
        mode=REPLAY_MODE,
        allow_synthetic=True,
    )
    intermediate: list[dict[str, Any]] = []
    for day_number in range(16, 23):
        intermediate.append(
            run_once(
                project_root=project_root,
                registry_path=registry_path,
                field_registry=field_registry,
                database_path=database,
                decision_at_utc=f"2026-07-{day_number:02d}T05:00:00Z",
                created_at_utc=executed_at,
                mode=REPLAY_MODE,
                allow_synthetic=True,
            )
        )
    second_body = _synthetic_era5_payload(date(2026, 6, 15), date(2026, 7, 21))
    second_end = datetime(2026, 7, 21, 21, tzinfo=timezone.utc)
    archive_exact_era5_observation(
        store,
        retrieval_id="synthetic-era5-retrieval-2",
        observation_id="synthetic-era5-observation-2",
        weather_cell="synthetic-cell-weather",
        body=second_body,
        retrieved_at_utc="2026-07-23T04:00:00Z",
        valid_from_utc=first_start,
        valid_to_utc=second_end,
        synthetic_fixture=True,
    )
    final = run_once(
        project_root=project_root,
        registry_path=registry_path,
        field_registry=field_registry,
        database_path=database,
        decision_at_utc="2026-07-23T05:00:00Z",
        created_at_utc=executed_at,
        mode=REPLAY_MODE,
        allow_synthetic=True,
    )
    verification = verify_log(
        database_path=database,
        registry_path=registry_path,
        project_root=project_root,
    )
    with closing(sqlite3.connect(database)) as connection:
        connection.row_factory = sqlite3.Row
        payloads = [
            json.loads(row[0])
            for row in connection.execute("SELECT decision_payload_json FROM decisions")
        ]
    c6_transition_audit = [
        {
            "issue_local_date": row["issue_local_date"],
            "score_origin": row["score_origin"],
            "virtual_message_issued": row["virtual_message_issued"],
            "previous_message_date": row["transition"]["previous_message_date"],
            "elapsed_calendar_days_since_message": row["transition"][
                "elapsed_calendar_days_since_message"
            ],
            "growth_reference_status": row["transition"]["growth_reference_status"],
            "cumulative_messages_field_season": row["transition"][
                "cumulative_messages_field_season"
            ],
        }
        for row in sorted(payloads, key=lambda value: value["issue_local_date"])
        if row["model_id"] == "C6_weather"
        and row["policy"]["family"] == "P_growth_selected"
        and row["field_pseudo_id"] == "synthetic-field-weather"
    ]
    c6_rows = [row for row in payloads if row["model_id"] == "C6_weather"]
    exact_c6 = sum(
        row["score_origin"] == "exact_era5_episode" for row in c6_rows
    )
    fallback_c6 = sum(bool(row["used_c0_fallback"]) for row in c6_rows)
    alarm_days = sum(bool(row["transition"]["alarm_active"]) for row in payloads)
    abstentions = sum(row["decision_action"] == "abstain" for row in payloads)
    virtual_messages = sum(
        row["decision_action"] == "message_candidate" for row in payloads
    )
    return {
        "status": "passed" if verification["status"] == "passed" else "failed",
        "demonstration_mode": REPLAY_MODE,
        "prospective_live_run": False,
        "synthetic_inputs_only": True,
        "disease_quality_metrics_computed": False,
        "network_used": False,
        "network_status": "not_called",
        "notifications_sent": 0,
        "real_notifications_sent": 0,
        "schedule_activated": False,
        "live_run_executed": False,
        "field_registry_status": "synthetic_only",
        "executed_at_utc": _utc_text(executed_at),
        "first_slot": first,
        "idempotent_retry": retry,
        "intermediate_fallback_slots": intermediate,
        "middle_fallback_slot": intermediate[-1],
        "restored_weather_slot": final,
        "expected_daily_slots_per_field": 9,
        "executed_daily_slots_per_field": 9,
        "missed_daily_slots": 0,
        "unique_decisions": len(payloads),
        "exact_weather_decisions": sum(
            row["score_origin"] == "exact_era5_episode" and row["score"] is not None
            for row in payloads
        ),
        "c0_fallback_decisions": fallback_c6,
        "abstentions": abstentions,
        "abstention_fraction": abstentions / len(payloads),
        "virtual_message_candidates": virtual_messages,
        "virtual_messages": virtual_messages,
        "active_alarm_days": alarm_days,
        "active_alarm_fraction": alarm_days / len(payloads),
        "effective_c0_fraction_in_c6": fallback_c6 / len(c6_rows),
        "effective_c0_fraction": fallback_c6 / len(c6_rows),
        "c0_fallback_fraction": fallback_c6 / len(c6_rows),
        "active_weather_fraction_in_c6": exact_c6 / len(c6_rows),
        "weather_correction_fraction": exact_c6 / len(c6_rows),
        "compatible_weather_fraction": exact_c6 / len(c6_rows),
        "c6_source_transition_audit": c6_transition_audit,
        "source_switch_breaks_growth_comparability": bool(
            c6_transition_audit[1]["growth_reference_status"]
            == "score_origin_or_model_not_comparable"
            and c6_transition_audit[-1]["growth_reference_status"]
            == "not_same_continuous_score_segment"
        ),
        "source_switch_preserves_cooldown_state": bool(
            all(
                row["previous_message_date"] == "2026-07-15"
                and row["cumulative_messages_field_season"] == 1
                for row in c6_transition_audit[1:]
            )
        ),
        "verification": verification,
    }


__all__ = [
    "ACTIVE_REGISTRY_PURPOSE",
    "DEMO_REGISTRY_PURPOSE",
    "EXACT_ERA5_SOURCE",
    "FIELD_REGISTRY_SCHEMA_VERSION",
    "FieldRegistryError",
    "FrozenArtifactError",
    "FrozenScorer",
    "PROSPECTIVE_MODE",
    "REPLAY_MODE",
    "ScoreSnapshot",
    "ShadowEngineError",
    "WEATHER_OBSERVATION_KIND",
    "archive_exact_era5_observation",
    "evaluate_one_policy_transition",
    "load_active_field_registry",
    "readiness",
    "replay_from_log",
    "resolve_episode_weather",
    "run_once",
    "technical_demo",
    "verify_log",
    "write_field_registry",
]
