"""Sequential notification policy for the third late-blight research cycle.

The policy consumes already frozen scalar scores.  It never reads event dates,
labels, future visits, or weather values.  A growth repeat is compared with the
score saved at the last message that was actually issued.  Score-source changes
and missing-score gaps break comparability without resetting cooldown or alarm
state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GrowthPolicy:
    """Configuration of the frozen-score notification automaton."""

    threshold: float
    active_days: int = 7
    cooldown_days: int = 15
    minimum_repeat_interval_days: int = 7
    growth_override_enabled: bool = False
    growth_logit_delta: float | None = None
    epsilon: float = 1e-6
    version: str = "policy_v3_growth"

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if self.active_days < 1:
            raise ValueError("active_days must be positive")
        if self.cooldown_days < 1:
            raise ValueError("cooldown_days must be positive")
        if self.minimum_repeat_interval_days < 1:
            raise ValueError("minimum_repeat_interval_days must be positive")
        if self.minimum_repeat_interval_days > self.cooldown_days:
            raise ValueError("minimum_repeat_interval_days cannot exceed cooldown_days")
        if not 0 < self.epsilon < 0.5:
            raise ValueError("epsilon must be between zero and 0.5")
        if self.growth_override_enabled:
            if self.growth_logit_delta is None:
                raise ValueError("enabled growth override requires growth_logit_delta")
            if not np.isfinite(self.growth_logit_delta) or self.growth_logit_delta < 0:
                raise ValueError("growth_logit_delta must be finite and non-negative")


@dataclass
class GrowthRuntimeState:
    """Serializable state for one field-season replay."""

    last_message_date: pd.Timestamp | None = None
    last_message_score: float | None = None
    last_message_origin: str | None = None
    last_message_model_id: str | None = None
    last_message_model_version: str | None = None
    last_message_segment_id: int | None = None
    active_from: pd.Timestamp | None = None
    active_through: pd.Timestamp | None = None
    last_processed_date: pd.Timestamp | None = None
    last_score_was_valid: bool = False
    last_score_origin: str | None = None
    last_score_model_id: str | None = None
    last_score_model_version: str | None = None
    current_segment_id: int = 0
    cumulative_messages: int = 0
    cumulative_alarm_days: int = 0

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "last_message_date",
            "active_from",
            "active_through",
            "last_processed_date",
        ):
            value = payload[name]
            payload[name] = None if value is None else pd.Timestamp(value).isoformat()
        return payload

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "GrowthRuntimeState":
        values = dict(payload)
        for name in (
            "last_message_date",
            "active_from",
            "active_through",
            "last_processed_date",
        ):
            value = values.get(name)
            values[name] = None if value is None else pd.Timestamp(value)
        return cls(**values)


def serialize_runtime_states(
    states: Mapping[str, GrowthRuntimeState],
) -> dict[str, dict[str, Any]]:
    """Convert replay state to a payload accepted by :func:`json.dumps`."""

    return {str(key): state.to_json_dict() for key, state in sorted(states.items())}


def deserialize_runtime_states(
    payload: Mapping[str, Mapping[str, Any]],
) -> dict[str, GrowthRuntimeState]:
    """Restore replay state from :func:`serialize_runtime_states` output."""

    return {
        str(key): GrowthRuntimeState.from_json_dict(value)
        for key, value in payload.items()
    }


def dumps_runtime_states(states: Mapping[str, GrowthRuntimeState]) -> str:
    """Serialize replay state as deterministic JSON."""

    return json.dumps(
        serialize_runtime_states(states), ensure_ascii=False, sort_keys=True
    )


def loads_runtime_states(payload: str) -> dict[str, GrowthRuntimeState]:
    """Restore replay state from :func:`dumps_runtime_states`."""

    source = json.loads(payload)
    if not isinstance(source, dict):
        raise ValueError("runtime state JSON must contain an object")
    return deserialize_runtime_states(source)


# Descriptive aliases used by pipeline/reporting code.
serialize_policy_states = serialize_runtime_states
deserialize_policy_states = deserialize_runtime_states


def clipped_logit(value: float, epsilon: float = 1e-6) -> float:
    """Return the contract logit after clipping a scalar probability."""

    clipped = float(np.clip(float(value), epsilon, 1.0 - epsilon))
    return float(np.log(clipped / (1.0 - clipped)))


def _calendar_days(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Count local calendar dates, avoiding DST-dependent 23/25 hour days."""

    return (pd.Timestamp(end).date() - pd.Timestamp(start).date()).days


def _aligned_values(
    value: pd.Series | Sequence[Any] | Any,
    index: pd.Index,
    *,
    default: Any,
    name: str,
) -> pd.Series:
    if value is None:
        return pd.Series(default, index=index, name=name, dtype=object)
    if isinstance(value, pd.Series):
        return value.reindex(index).rename(name)
    if isinstance(value, str) or np.isscalar(value):
        return pd.Series(value, index=index, name=name)
    if len(value) != len(index):
        raise ValueError(f"{name} must contain one value per decision row")
    return pd.Series(list(value), index=index, name=name)


def _copy_initial_states(
    initial_states: Mapping[str, GrowthRuntimeState | Mapping[str, Any]] | None,
) -> dict[str, GrowthRuntimeState]:
    result: dict[str, GrowthRuntimeState] = {}
    for raw_key, raw_state in (initial_states or {}).items():
        key = str(raw_key)
        if isinstance(raw_state, GrowthRuntimeState):
            result[key] = replace(raw_state)
        elif isinstance(raw_state, Mapping):
            result[key] = GrowthRuntimeState.from_json_dict(raw_state)
        else:
            raise TypeError("initial_states values must be GrowthRuntimeState or mappings")
    return result


def _reference_status(
    state: GrowthRuntimeState,
    *,
    score_valid: bool,
    origin: str | None,
    model_id: str | None,
    model_version: str | None,
    segment_id: int | None,
) -> tuple[bool, str]:
    if state.last_message_date is None:
        return False, "no_previous_message"
    if not score_valid:
        return False, "current_score_missing"
    if (
        origin != state.last_message_origin
        or model_id != state.last_message_model_id
        or model_version != state.last_message_model_version
    ):
        return False, "score_origin_or_model_not_comparable"
    if segment_id != state.last_message_segment_id:
        return False, "not_same_continuous_score_segment"
    return True, "comparable"


def simulate_growth_policy(
    frame: pd.DataFrame,
    score: pd.Series | Sequence[float],
    policy: GrowthPolicy,
    evaluation_scope: str = "service_calendar",
    evaluation_mask: pd.Series | Sequence[bool] | None = None,
    score_origin: pd.Series | Sequence[str] | str | None = None,
    model_id: pd.Series | Sequence[str] | str = "frozen_score",
    model_version: pd.Series | Sequence[str] | str | None = None,
    initial_states: Mapping[str, GrowthRuntimeState | Mapping[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, GrowthRuntimeState]]:
    """Replay one policy chronologically and return daily rows plus checkpoint.

    ``evaluation_mask`` marks burden/evaluation days but deliberately does not
    alter the supplied score.  For the paired scenario the caller must pass the
    same score masked outside ``candidate_comparison_complete`` as in v3/v4.
    This mirrors :func:`early_warning_models.simulate_policy` and keeps the
    disabled-growth identity exact.
    """

    required = {
        "field_season",
        "season",
        "issue_date",
        "issued_at",
        "service_active",
        "evaluation_field_day",
        "target_class",
        "target_observable",
        "days_to_first_recorded_event",
        "warnable_first_event",
        "coordinate_scope",
        "previous_visit_gap_days",
        "common_weather_complete",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"decision frame is missing required columns: {sorted(missing)}")
    if frame["field_season"].isna().any():
        raise ValueError("field_season cannot be missing")

    columns = [
        "field_season",
        "season",
        "issue_date",
        "issued_at",
        "service_active",
        "evaluation_field_day",
        "target_class",
        "target_observable",
        "days_to_first_recorded_event",
        "warnable_first_event",
        "coordinate_scope",
        "previous_visit_gap_days",
        "common_weather_complete",
    ]
    for optional in (
        "nasa_common_complete",
        "era_common_complete",
        "episode_weather_complete",
        "candidate_comparison_complete",
    ):
        if optional in frame:
            columns.append(optional)
    result = frame[columns].copy()
    result["issue_date"] = pd.to_datetime(result["issue_date"])
    duplicate = result.duplicated(["field_season", "issue_date"], keep=False)
    if duplicate.any():
        raise ValueError("at most one decision row is allowed per field-season and date")

    result["score"] = _aligned_values(
        score, result.index, default=np.nan, name="score"
    ).astype(float)
    finite = result["score"].dropna()
    if (~np.isfinite(finite)).any() or (~finite.between(0.0, 1.0)).any():
        raise ValueError("computed scores must be finite probabilities in [0, 1]")
    origins = _aligned_values(
        score_origin, result.index, default="unspecified", name="score_origin"
    )
    model_ids = _aligned_values(
        model_id, result.index, default="frozen_score", name="score_model_id"
    )
    model_versions = _aligned_values(
        model_version,
        result.index,
        default="unspecified",
        name="score_model_version",
    )

    size = len(result)
    score_values = result["score"].to_numpy(dtype=float)
    service_active = result["service_active"].fillna(False).to_numpy(dtype=bool)
    days = [pd.Timestamp(value) for value in result["issue_date"]]
    origin_values = origins.to_numpy(dtype=object)
    model_id_values = model_ids.to_numpy(dtype=object)
    model_version_values = model_versions.to_numpy(dtype=object)

    messages = np.zeros(size, dtype=bool)
    alarms = np.zeros(size, dtype=bool)
    suppressed = np.zeros(size, dtype=bool)
    reasons = np.full(size, "below_threshold", dtype=object)
    message_kind = np.full(size, "none", dtype=object)
    starts = np.full(size, pd.NaT, dtype=object)
    ends = np.full(size, pd.NaT, dtype=object)
    prior_dates = np.full(size, pd.NaT, dtype=object)
    prior_scores = np.full(size, np.nan, dtype=float)
    prior_origins = np.full(size, None, dtype=object)
    prior_model_ids = np.full(size, None, dtype=object)
    prior_model_versions = np.full(size, None, dtype=object)
    prior_segments = np.full(size, np.nan, dtype=float)
    elapsed_days = np.full(size, np.nan, dtype=float)
    score_logits = np.full(size, np.nan, dtype=float)
    reference_logits = np.full(size, np.nan, dtype=float)
    logit_growth = np.full(size, np.nan, dtype=float)
    odds_growth = np.full(size, np.nan, dtype=float)
    reference_comparable = np.zeros(size, dtype=bool)
    reference_status = np.full(size, "no_previous_message", dtype=object)
    threshold_reached = np.zeros(size, dtype=bool)
    ordinary_cooldown_ok = np.zeros(size, dtype=bool)
    minimum_interval_ok = np.zeros(size, dtype=bool)
    growth_override_used = np.zeros(size, dtype=bool)
    segment_ids = np.full(size, np.nan, dtype=float)
    segment_started = np.zeros(size, dtype=bool)
    cumulative_messages = np.zeros(size, dtype=int)
    cumulative_alarm_days = np.zeros(size, dtype=int)

    states = _copy_initial_states(initial_states)
    ordering = result[["field_season", "issue_date"]].reset_index(drop=True)
    active_offset = pd.DateOffset(days=policy.active_days - 1)
    for raw_field, positions in ordering.groupby("field_season", sort=False).groups.items():
        field_key = str(raw_field)
        state = states.setdefault(field_key, GrowthRuntimeState())
        ordered = ordering.loc[positions].sort_values("issue_date").index
        for position in ordered:
            day = days[position]
            if state.last_processed_date is not None and _calendar_days(
                state.last_processed_date, day
            ) <= 0:
                raise ValueError(
                    f"replay for {field_key} must resume after its last processed date"
                )

            value = score_values[position]
            value_valid = bool(np.isfinite(value))
            origin = None if pd.isna(origin_values[position]) else str(origin_values[position])
            current_model_id = (
                None if pd.isna(model_id_values[position]) else str(model_id_values[position])
            )
            current_model_version = (
                None
                if pd.isna(model_version_values[position])
                else str(model_version_values[position])
            )

            if state.last_message_date is not None:
                prior_dates[position] = state.last_message_date
                prior_scores[position] = float(state.last_message_score)
                prior_origins[position] = state.last_message_origin
                prior_model_ids[position] = state.last_message_model_id
                prior_model_versions[position] = state.last_message_model_version
                if state.last_message_segment_id is not None:
                    prior_segments[position] = state.last_message_segment_id
                elapsed_days[position] = _calendar_days(state.last_message_date, day)
                reference_logits[position] = clipped_logit(
                    float(state.last_message_score), policy.epsilon
                )

            operational_score_valid = bool(service_active[position] and value_valid)
            current_segment: int | None = None
            if operational_score_valid:
                contiguous = bool(
                    state.last_processed_date is not None
                    and _calendar_days(state.last_processed_date, day) == 1
                    and state.last_score_was_valid
                    and origin == state.last_score_origin
                    and current_model_id == state.last_score_model_id
                    and current_model_version == state.last_score_model_version
                )
                if not contiguous:
                    state.current_segment_id += 1
                    segment_started[position] = True
                current_segment = state.current_segment_id
                segment_ids[position] = current_segment
                score_logits[position] = clipped_logit(value, policy.epsilon)

            comparable, comparison_status = _reference_status(
                state,
                score_valid=operational_score_valid,
                origin=origin,
                model_id=current_model_id,
                model_version=current_model_version,
                segment_id=current_segment,
            )
            reference_comparable[position] = comparable
            reference_status[position] = comparison_status
            if comparable:
                logit_growth[position] = (
                    score_logits[position] - reference_logits[position]
                )
                odds_growth[position] = float(np.exp(logit_growth[position]))

            if not service_active[position]:
                state.active_from = None
                state.active_through = None
                reasons[position] = "stopped_after_record_available"
            elif not value_valid:
                reasons[position] = "abstained_missing_input"
            elif value >= policy.threshold:
                threshold_reached[position] = True
                elapsed = elapsed_days[position]
                ordinary_ok = bool(
                    state.last_message_date is None
                    or elapsed >= policy.cooldown_days
                )
                ordinary_cooldown_ok[position] = ordinary_ok
                minimum_ok = bool(
                    state.last_message_date is not None
                    and elapsed >= policy.minimum_repeat_interval_days
                )
                minimum_interval_ok[position] = minimum_ok
                issue = False
                if ordinary_ok:
                    issue = True
                    message_kind[position] = "ordinary"
                    reasons[position] = "issued_threshold_crossing_or_refresh"
                elif not policy.growth_override_enabled:
                    suppressed[position] = True
                    reasons[position] = "suppressed_cooldown"
                elif not minimum_ok:
                    suppressed[position] = True
                    reasons[position] = "suppressed_growth_minimum_interval"
                elif not comparable:
                    suppressed[position] = True
                    reasons[position] = "growth_reference_not_comparable"
                elif logit_growth[position] >= float(policy.growth_logit_delta):
                    issue = True
                    growth_override_used[position] = True
                    message_kind[position] = "growth_override"
                    reasons[position] = "issued_growth_override"
                else:
                    suppressed[position] = True
                    reasons[position] = "suppressed_growth_below_delta"

                if issue:
                    messages[position] = True
                    state.last_message_date = day
                    state.last_message_score = float(value)
                    state.last_message_origin = origin
                    state.last_message_model_id = current_model_id
                    state.last_message_model_version = current_model_version
                    state.last_message_segment_id = current_segment
                    state.active_from = day
                    state.active_through = day + active_offset
                    state.cumulative_messages += 1

            if state.active_through is not None and day <= state.active_through:
                alarms[position] = True
                starts[position] = state.active_from
                ends[position] = state.active_through
                state.cumulative_alarm_days += 1
            cumulative_messages[position] = state.cumulative_messages
            cumulative_alarm_days[position] = state.cumulative_alarm_days

            state.last_processed_date = day
            state.last_score_was_valid = operational_score_valid
            state.last_score_origin = origin if operational_score_valid else None
            state.last_score_model_id = current_model_id if operational_score_valid else None
            state.last_score_model_version = (
                current_model_version if operational_score_valid else None
            )

    result["score_status"] = np.where(result["score"].notna(), "computed", "abstained")
    result["score_origin"] = origins
    result["score_model_id"] = model_ids
    result["score_model_version"] = model_versions
    result["score_comparison_segment_id"] = pd.array(segment_ids, dtype="Int64")
    result["score_comparison_segment_started"] = segment_started
    result["message_issued"] = messages
    result["message_kind"] = message_kind
    result["alarm_active"] = alarms
    result["active_from"] = pd.to_datetime(starts)
    result["active_through"] = pd.to_datetime(ends)
    result["forecast_window_start"] = result["issue_date"] + pd.Timedelta(days=3)
    result["forecast_window_end"] = result["issue_date"] + pd.Timedelta(days=10)
    result["action_reason"] = reasons
    result["suppressed_repeat"] = suppressed
    result["previous_message_date"] = pd.to_datetime(prior_dates)
    result["previous_message_score"] = prior_scores
    result["previous_message_origin"] = prior_origins
    result["previous_message_model_id"] = prior_model_ids
    result["previous_message_model_version"] = prior_model_versions
    result["previous_message_segment_id"] = pd.array(prior_segments, dtype="Int64")
    result["elapsed_calendar_days_since_message"] = pd.array(
        elapsed_days, dtype="Int64"
    )
    result["score_logit"] = score_logits
    result["previous_message_score_logit"] = reference_logits
    result["logit_growth_from_previous_message"] = logit_growth
    result["odds_multiplier_from_previous_message"] = odds_growth
    result["growth_reference_comparable"] = reference_comparable
    result["growth_reference_status"] = reference_status
    result["threshold_reached"] = threshold_reached
    result["ordinary_cooldown_satisfied"] = ordinary_cooldown_ok
    result["growth_minimum_interval_satisfied"] = minimum_interval_ok
    result["growth_override_used"] = growth_override_used
    result["cumulative_messages_field_season"] = cumulative_messages
    result["cumulative_alarm_days_field_season"] = cumulative_alarm_days

    if evaluation_mask is None:
        scope_mask = result["service_active"].fillna(False).astype(bool)
    else:
        scope_mask = _aligned_values(
            evaluation_mask,
            result.index,
            default=False,
            name="evaluation_scope_day",
        ).fillna(False).astype(bool)
        scope_mask &= result["service_active"].fillna(False).astype(bool)
    result["evaluation_scope"] = evaluation_scope
    result["evaluation_scope_day"] = scope_mask
    result["policy_threshold"] = policy.threshold
    result["policy_active_days"] = policy.active_days
    result["policy_cooldown_days"] = policy.cooldown_days
    result["policy_minimum_repeat_interval_days"] = (
        policy.minimum_repeat_interval_days
    )
    result["policy_growth_override_enabled"] = policy.growth_override_enabled
    result["policy_growth_logit_delta"] = policy.growth_logit_delta
    result["policy_logit_epsilon"] = policy.epsilon
    result["policy_version"] = policy.version
    return result, states


__all__ = [
    "GrowthPolicy",
    "GrowthRuntimeState",
    "clipped_logit",
    "deserialize_policy_states",
    "deserialize_runtime_states",
    "dumps_runtime_states",
    "loads_runtime_states",
    "serialize_policy_states",
    "serialize_runtime_states",
    "simulate_growth_policy",
]
