"""Audits required before the third late-blight warning cycle.

The functions in this module only compare saved event and policy artifacts.
They do not fit a model, select a threshold, or replay notification state.
Detailed outputs replace field identifiers with deterministic one-way keys.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
import hashlib
from typing import Any

import numpy as np
import pandas as pd


PRIMARY_YEARS = (2020, 2025)
EVENT_KEYS = ("season", "field_season")
DAY_KEYS = ("season", "field_season", "issue_date")


def _require(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _bool(values: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    converted = values.map(
        {
            True: True,
            False: False,
            "True": True,
            "False": False,
            "true": True,
            "false": False,
            1: True,
            0: False,
        }
    )
    if (values.notna() & converted.isna()).any():
        raise ValueError(f"{name} contains an invalid boolean value")
    return converted.fillna(False).astype(bool)


def _event_key(field_season: Any, season: int, namespace: str) -> str:
    payload = f"{namespace}\x1f{int(season)}\x1f{field_season}".encode()
    return "evt_" + hashlib.sha256(payload).hexdigest()[:20]


def _day_key(field_season: Any, season: int, issue_date: Any, namespace: str) -> str:
    day = pd.Timestamp(issue_date).date().isoformat()
    payload = f"{namespace}\x1f{int(season)}\x1f{field_season}\x1f{day}".encode()
    return "day_" + hashlib.sha256(payload).hexdigest()[:20]


def _episode_key(
    field_season: Any,
    season: int,
    first_date: Any,
    kind: str,
    namespace: str,
) -> str:
    day = pd.Timestamp(first_date).date().isoformat()
    payload = (
        f"{namespace}\x1f{kind}\x1f{int(season)}\x1f{field_season}\x1f{day}"
    ).encode()
    return "ep_" + hashlib.sha256(payload).hexdigest()[:20]


def _selected_events(
    frame: pd.DataFrame,
    *,
    model_code: str,
    evaluation_scope: str,
    slice_name: str,
    years: tuple[int, int],
    name: str,
) -> pd.DataFrame:
    _require(
        frame,
        (
            *EVENT_KEYS,
            "model_code",
            "evaluation_scope",
            "slice",
            "warnable_event",
            "positive_at_entry",
            "timely_hit",
        ),
        name,
    )
    selected = frame[
        frame["season"].between(*years)
        & frame["model_code"].eq(model_code)
        & frame["evaluation_scope"].eq(evaluation_scope)
        & frame["slice"].eq(slice_name)
    ].copy()
    selected["season"] = pd.to_numeric(selected["season"], errors="raise").astype(int)
    if selected.duplicated(list(EVENT_KEYS)).any():
        raise ValueError(f"{name} contains duplicate event identities")
    for column in ("warnable_event", "positive_at_entry", "timely_hit"):
        selected[column] = _bool(selected[column], name=f"{name}.{column}")
    return selected.sort_values(list(EVENT_KEYS)).reset_index(drop=True)


def audit_event_populations(
    candidate_event_hits: pd.DataFrame,
    baseline_event_hits: pd.DataFrame,
    *,
    candidate: str,
    baseline: str = "C0",
    evaluation_scope: str = "service_calendar",
    slice_name: str = "A_plus_B",
    years: tuple[int, int] = PRIMARY_YEARS,
    event_registry: pd.DataFrame | None = None,
    key_namespace: str = "late_blight_cycle3_population",
) -> dict[str, pd.DataFrame]:
    """Reconcile all first registrations with the eligible recall population.

    ``candidate_event_hits`` and ``baseline_event_hits`` may be separate v4 and
    v3 tables or two subsets of one combined table.  Pairing uses the real
    field-season identity internally and fails if either saved population or
    its eligibility flags differ.
    """
    candidate_rows = _selected_events(
        candidate_event_hits,
        model_code=candidate,
        evaluation_scope=evaluation_scope,
        slice_name=slice_name,
        years=years,
        name="candidate_event_hits",
    )
    baseline_rows = _selected_events(
        baseline_event_hits,
        model_code=baseline,
        evaluation_scope=evaluation_scope,
        slice_name=slice_name,
        years=years,
        name="baseline_event_hits",
    )
    candidate_keys = pd.MultiIndex.from_frame(candidate_rows[list(EVENT_KEYS)])
    baseline_keys = pd.MultiIndex.from_frame(baseline_rows[list(EVENT_KEYS)])
    if not candidate_keys.equals(baseline_keys):
        only_candidate = len(candidate_keys.difference(baseline_keys))
        only_baseline = len(baseline_keys.difference(candidate_keys))
        raise ValueError(
            "Non-matching first-event population: "
            f"candidate_only={only_candidate}, baseline_only={only_baseline}"
        )

    keep = [
        *EVENT_KEYS,
        "fold_id",
        "warnable_event",
        "positive_at_entry",
        "timely_hit",
    ]
    keep_candidate = [column for column in keep if column in candidate_rows]
    keep_baseline = [column for column in keep if column in baseline_rows]
    events = candidate_rows[keep_candidate].merge(
        baseline_rows[keep_baseline],
        on=list(EVENT_KEYS),
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    for column in ("warnable_event", "positive_at_entry"):
        left = events[f"{column}_candidate"]
        right = events[f"{column}_baseline"]
        if not left.equals(right):
            raise ValueError(f"Candidate and baseline disagree on {column}")
        events[column] = left
    if {"fold_id_candidate", "fold_id_baseline"}.issubset(events):
        if not events["fold_id_candidate"].astype(str).equals(
            events["fold_id_baseline"].astype(str)
        ):
            raise ValueError("Candidate and baseline disagree on outer fold")
        events["fold_id"] = events["fold_id_candidate"]

    registry_matches = True
    if event_registry is not None:
        _require(
            event_registry,
            ("season", "field_season", "first_recorded_event_date"),
            "event_registry",
        )
        registry = event_registry[event_registry["season"].between(*years)].copy()
        registry = registry[registry["first_recorded_event_date"].notna()]
        if registry.duplicated(list(EVENT_KEYS)).any():
            raise ValueError("event_registry contains duplicate first-event identities")
        registry_keys = pd.MultiIndex.from_frame(
            registry.sort_values(list(EVENT_KEYS))[list(EVENT_KEYS)]
        )
        registry_matches = candidate_keys.equals(registry_keys)
        if not registry_matches:
            raise ValueError("event_registry and event_hits contain different first events")
        enrich = [
            *EVENT_KEYS,
            "entry_category",
            "first_recorded_event_date",
            "first_recorded_event_available_date",
        ]
        enrich = [column for column in enrich if column in registry]
        events = events.merge(registry[enrich], on=list(EVENT_KEYS), validate="one_to_one")

    eligible = events["warnable_event"]
    candidate_hit = events["timely_hit_candidate"] & eligible
    baseline_hit = events["timely_hit_baseline"] & eligible
    events["transition"] = np.select(
        (
            eligible & candidate_hit & baseline_hit,
            eligible & candidate_hit & ~baseline_hit,
            eligible & ~candidate_hit & baseline_hit,
            eligible,
        ),
        ("both_hit", "gained_by_candidate", "lost_by_candidate", "neither_hit"),
        default="not_eligible",
    )
    events["eligibility_reason"] = np.select(
        (eligible, events["positive_at_entry"]),
        ("eligible_warnable_first_event", "positive_known_at_entry"),
        default="not_warnable_other",
    )
    events["event_key"] = [
        _event_key(field, season, key_namespace)
        for field, season in zip(events["field_season"], events["season"])
    ]

    summaries: list[dict[str, Any]] = []
    groupings: list[tuple[str, Any, pd.DataFrame]] = [("pooled", pd.NA, events)]
    groupings.extend(
        ("year", int(season), group)
        for season, group in events.groupby("season", sort=True)
    )
    for aggregation, season, group in groupings:
        eligible_group = group[group["warnable_event"]]
        counts = eligible_group["transition"].value_counts()
        both = int(counts.get("both_hit", 0))
        gained = int(counts.get("gained_by_candidate", 0))
        lost = int(counts.get("lost_by_candidate", 0))
        neither = int(counts.get("neither_hit", 0))
        candidate_timely = int(eligible_group["timely_hit_candidate"].sum())
        baseline_timely = int(eligible_group["timely_hit_baseline"].sum())
        eligible_events = int(len(eligible_group))
        invariants = {
            "invariant_partition": both + gained + lost + neither == eligible_events,
            "invariant_candidate_hits": both + gained == candidate_timely,
            "invariant_baseline_hits": both + lost == baseline_timely,
            "invariant_net_gain": gained - lost
            == candidate_timely - baseline_timely,
        }
        summaries.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "evaluation_scope": evaluation_scope,
                "aggregation": aggregation,
                "season": season,
                "all_first_events": int(len(group)),
                "eligible_events": eligible_events,
                "positive_at_entry_events": int(group["positive_at_entry"].sum()),
                "other_ineligible_events": int(
                    (~group["warnable_event"] & ~group["positive_at_entry"]).sum()
                ),
                "both_hit": both,
                "gained_by_candidate": gained,
                "lost_by_candidate": lost,
                "neither_hit": neither,
                "candidate_timely_events": candidate_timely,
                "baseline_timely_events": baseline_timely,
                "delta_timely_events": candidate_timely - baseline_timely,
                "registry_population_matches": bool(registry_matches),
                **invariants,
                "all_invariants_pass": bool(all(invariants.values())),
            }
        )
    summary = pd.DataFrame(summaries)
    if not summary["all_invariants_pass"].all():
        raise AssertionError("Eligible gained/lost invariants failed")

    eligible_detail = events[events["warnable_event"]].copy()
    for frame in (eligible_detail, events):
        frame["candidate"] = candidate
        frame["baseline"] = baseline
        frame["evaluation_scope"] = evaluation_scope
    detail_columns = [
        "candidate",
        "baseline",
        "evaluation_scope",
        "event_key",
        "season",
        "fold_id",
        "transition",
        "timely_hit_candidate",
        "timely_hit_baseline",
        "eligibility_reason",
    ]
    diagnostic_columns = [
        "candidate",
        "baseline",
        "evaluation_scope",
        "event_key",
        "season",
        "fold_id",
        "warnable_event",
        "positive_at_entry",
        "eligibility_reason",
        "transition",
        "timely_hit_candidate",
        "timely_hit_baseline",
        "entry_category",
        "first_recorded_event_date",
        "first_recorded_event_available_date",
    ]
    return {
        "population_audit_summary": summary,
        "eligible_gained_lost_events": eligible_detail[
            [column for column in detail_columns if column in eligible_detail]
        ].reset_index(drop=True),
        "all_first_event_diagnostics": events[
            [column for column in diagnostic_columns if column in events]
        ].reset_index(drop=True),
    }


def _selected_days(
    frame: pd.DataFrame,
    *,
    model_code: str,
    evaluation_scope: str,
    years: tuple[int, int],
    candidate: bool,
    name: str,
) -> pd.DataFrame:
    common = (
        *DAY_KEYS,
        "model_code",
        "evaluation_scope",
        "evaluation_scope_day",
        "score",
        "policy_threshold",
        "message_issued",
        "suppressed_repeat",
    )
    required = (
        *common,
        "alpha",
        "fallback_to_c0",
        "days_to_first_recorded_event",
        "warnable_first_event",
    ) if candidate else common
    _require(frame, required, name)
    selected = frame[
        frame["season"].between(*years)
        & frame["model_code"].eq(model_code)
        & frame["evaluation_scope"].eq(evaluation_scope)
        & _bool(frame["evaluation_scope_day"], name=f"{name}.evaluation_scope_day")
    ].copy()
    selected["season"] = pd.to_numeric(selected["season"], errors="raise").astype(int)
    selected["issue_date"] = pd.to_datetime(selected["issue_date"]).dt.normalize()
    if selected.duplicated(list(DAY_KEYS)).any():
        raise ValueError(f"{name} contains duplicate evaluation days")
    for column in ("message_issued", "suppressed_repeat"):
        selected[column] = _bool(selected[column], name=f"{name}.{column}")
    if candidate:
        selected["fallback_to_c0"] = _bool(
            selected["fallback_to_c0"], name=f"{name}.fallback_to_c0"
        )
        selected["warnable_first_event"] = _bool(
            selected["warnable_first_event"], name=f"{name}.warnable_first_event"
        )
    return selected.sort_values(list(DAY_KEYS)).reset_index(drop=True)


def _renamed(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    return frame.rename(
        columns={column: f"{column}_{suffix}" for column in frame if column not in DAY_KEYS}
    )


def _single_numeric(group: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(group[column], errors="raise").dropna().unique()
    if len(values) != 1:
        raise ValueError(f"Expected one {column} value per outer fold, got {len(values)}")
    return float(values[0])


def _episode_rows(
    days: pd.DataFrame,
    masks: Iterable[tuple[str, pd.Series]],
    *,
    namespace: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for kind, mask in masks:
        selected = days.loc[mask].sort_values(list(DAY_KEYS)).copy()
        if selected.empty:
            continue
        new_episode = (
            selected["field_season"].ne(selected["field_season"].shift())
            | selected["season"].ne(selected["season"].shift())
            | selected["issue_date"].diff().dt.days.ne(1)
        )
        selected["_episode_number"] = new_episode.cumsum()
        for _, group in selected.groupby("_episode_number", sort=False):
            first = group.iloc[0]
            records.append(
                {
                    "episode_key": _episode_key(
                        first["field_season"],
                        int(first["season"]),
                        first["issue_date"],
                        kind,
                        namespace,
                    ),
                    "episode_kind": kind,
                    "season": int(first["season"]),
                    "first_date": group["issue_date"].min(),
                    "last_date": group["issue_date"].max(),
                    "days": int(len(group)),
                    "posthoc_timely_days": int(group["posthoc_timely_day"].sum()),
                    "posthoc_touches_eligible_event": bool(
                        group["posthoc_timely_day"].any()
                    ),
                }
            )
    return pd.DataFrame(records)


def audit_suppression_paths(
    candidate_alarm_states: pd.DataFrame,
    baseline_alarm_states: pd.DataFrame,
    *,
    candidate: str,
    baseline: str = "C0",
    evaluation_scope: str = "service_calendar",
    years: tuple[int, int] = PRIMARY_YEARS,
    timely_window: tuple[int, int] = (3, 10),
    score_atol: float = 1e-12,
    key_namespace: str = "late_blight_cycle3_suppression",
) -> dict[str, pd.DataFrame]:
    """Decompose score changes, threshold states, crossings, and suppression.

    A weather correction is considered applied only when ``alpha > 0`` and the
    saved C6 row did not use its C0 fallback.  For ``alpha == 0`` the effective
    source is always C0, irrespective of the informational fallback flag.
    Temporal crossings require adjacent local calendar dates, finite scores,
    and an unchanged effective score source and model version.
    """
    candidate_rows = _selected_days(
        candidate_alarm_states,
        model_code=candidate,
        evaluation_scope=evaluation_scope,
        years=years,
        candidate=True,
        name="candidate_alarm_states",
    )
    baseline_rows = _selected_days(
        baseline_alarm_states,
        model_code=baseline,
        evaluation_scope=evaluation_scope,
        years=years,
        candidate=False,
        name="baseline_alarm_states",
    )
    candidate_keys = pd.MultiIndex.from_frame(candidate_rows[list(DAY_KEYS)])
    baseline_keys = pd.MultiIndex.from_frame(baseline_rows[list(DAY_KEYS)])
    if not candidate_keys.equals(baseline_keys):
        raise ValueError(
            "Non-matching evaluation-day population: "
            f"candidate_only={len(candidate_keys.difference(baseline_keys))}, "
            f"baseline_only={len(baseline_keys.difference(candidate_keys))}"
        )
    days = _renamed(candidate_rows, "candidate").merge(
        _renamed(baseline_rows, "baseline"),
        on=list(DAY_KEYS),
        validate="one_to_one",
    )
    days = days.sort_values(list(DAY_KEYS)).reset_index(drop=True)

    candidate_score = pd.to_numeric(days["score_candidate"], errors="coerce")
    baseline_score = pd.to_numeric(days["score_baseline"], errors="coerce")
    candidate_threshold = pd.to_numeric(
        days["policy_threshold_candidate"], errors="coerce"
    )
    baseline_threshold = pd.to_numeric(
        days["policy_threshold_baseline"], errors="coerce"
    )
    alpha = pd.to_numeric(days["alpha_candidate"], errors="raise")
    fallback = _bool(days["fallback_to_c0_candidate"], name="fallback_to_c0")
    days["weather_available"] = ~fallback
    days["weather_correction_applied"] = alpha.gt(0) & ~fallback
    days["effective_score_origin_candidate"] = np.select(
        (alpha.eq(0), fallback),
        ("C0", "C0_fallback"),
        default="C6_weather",
    )
    days["effective_score_origin_baseline"] = baseline

    both_missing = candidate_score.isna() & baseline_score.isna()
    both_equal = (
        candidate_score.notna()
        & baseline_score.notna()
        & np.isclose(candidate_score, baseline_score, atol=score_atol, rtol=0)
    )
    days["score_changed"] = ~(both_missing | both_equal)
    days["candidate_above_threshold"] = candidate_score.ge(candidate_threshold)
    days["baseline_above_threshold"] = baseline_score.ge(baseline_threshold)
    days["above_threshold_either"] = (
        days["candidate_above_threshold"] | days["baseline_above_threshold"]
    )
    days["different_threshold_decision"] = days["candidate_above_threshold"].ne(
        days["baseline_above_threshold"]
    )
    days["message_changed"] = days["message_issued_candidate"].ne(
        days["message_issued_baseline"]
    )
    days["factual_suppression_either"] = (
        days["suppressed_repeat_candidate"] | days["suppressed_repeat_baseline"]
    )
    days["changed_score_suppressed"] = (
        days["score_changed"]
        & days["above_threshold_either"]
        & days["factual_suppression_either"]
    )
    low, high = timely_window
    days["posthoc_timely_day"] = (
        pd.to_numeric(
            days["days_to_first_recorded_event_candidate"], errors="coerce"
        ).between(low, high)
        & days["warnable_first_event_candidate"]
    )

    same_field = days["field_season"].eq(days["field_season"].shift())
    adjacent_day = same_field & days["issue_date"].diff().dt.days.eq(1)
    candidate_present = candidate_score.notna() & candidate_score.shift().notna()
    baseline_present = baseline_score.notna() & baseline_score.shift().notna()
    same_candidate_origin = days["effective_score_origin_candidate"].eq(
        days["effective_score_origin_candidate"].shift()
    )
    same_baseline_origin = days["effective_score_origin_baseline"].eq(
        days["effective_score_origin_baseline"].shift()
    )
    if "model_version_candidate" in days:
        same_candidate_origin &= days["model_version_candidate"].astype(str).eq(
            days["model_version_candidate"].shift().astype(str)
        )
    if "model_version_baseline" in days:
        same_baseline_origin &= days["model_version_baseline"].astype(str).eq(
            days["model_version_baseline"].shift().astype(str)
        )
    days["score_source_transition"] = adjacent_day & ~same_candidate_origin
    days["candidate_upward_crossing_raw"] = (
        adjacent_day
        & candidate_present
        & days["candidate_above_threshold"]
        & ~days["candidate_above_threshold"].shift(fill_value=False)
    )
    days["candidate_upward_crossing_comparable"] = (
        days["candidate_upward_crossing_raw"] & same_candidate_origin
    )
    days["baseline_upward_crossing_comparable"] = (
        adjacent_day
        & baseline_present
        & same_baseline_origin
        & days["baseline_above_threshold"]
        & ~days["baseline_above_threshold"].shift(fill_value=False)
    )
    days["raw_crossing_on_source_transition"] = (
        days["candidate_upward_crossing_raw"] & days["score_source_transition"]
    )

    episode_masks = (
        ("candidate_suppressed", days["suppressed_repeat_candidate"]),
        ("baseline_suppressed", days["suppressed_repeat_baseline"]),
        ("changed_score_suppressed", days["changed_score_suppressed"]),
    )
    episodes = _episode_rows(days, episode_masks, namespace=key_namespace)
    episodes["candidate"] = candidate
    episodes["baseline"] = baseline
    episodes["evaluation_scope"] = evaluation_scope

    def summarize(
        group: pd.DataFrame, aggregation: str, season: Any
    ) -> dict[str, Any]:
        group_episodes = episodes if aggregation == "pooled" else episodes[
            episodes["season"].eq(int(season))
        ]
        diagnostic = group["changed_score_suppressed"]
        posthoc = diagnostic & group["posthoc_timely_day"]
        alpha_value = pd.NA if aggregation == "pooled" else _single_numeric(
            group, "alpha_candidate"
        )
        candidate_threshold_value = (
            pd.NA
            if aggregation == "pooled"
            else _single_numeric(group, "policy_threshold_candidate")
        )
        baseline_threshold_value = (
            pd.NA
            if aggregation == "pooled"
            else _single_numeric(group, "policy_threshold_baseline")
        )
        return {
            "candidate": candidate,
            "baseline": baseline,
            "evaluation_scope": evaluation_scope,
            "aggregation": aggregation,
            "season": season,
            "fold_id": (
                pd.NA
                if aggregation == "pooled" or "fold_id_candidate" not in group
                else group["fold_id_candidate"].astype(str).iloc[0]
            ),
            "alpha": alpha_value,
            "candidate_threshold": candidate_threshold_value,
            "baseline_threshold": baseline_threshold_value,
            "matched_days": int(len(group)),
            "weather_available_days": int(group["weather_available"].sum()),
            "fallback_days": int((~group["weather_available"]).sum()),
            "weather_correction_applied_days": int(
                group["weather_correction_applied"].sum()
            ),
            "score_changed_days": int(group["score_changed"].sum()),
            "above_threshold_either_days": int(
                group["above_threshold_either"].sum()
            ),
            "changed_score_above_threshold_days": int(
                (group["score_changed"] & group["above_threshold_either"]).sum()
            ),
            "different_threshold_decision_days": int(
                group["different_threshold_decision"].sum()
            ),
            "different_decision_on_applied_weather_days": int(
                (
                    group["different_threshold_decision"]
                    & group["weather_correction_applied"]
                ).sum()
            ),
            "different_decision_without_applied_weather_days": int(
                (
                    group["different_threshold_decision"]
                    & ~group["weather_correction_applied"]
                ).sum()
            ),
            "candidate_upward_crossings_raw": int(
                group["candidate_upward_crossing_raw"].sum()
            ),
            "candidate_upward_crossings_comparable": int(
                group["candidate_upward_crossing_comparable"].sum()
            ),
            "candidate_comparable_crossings_on_applied_weather_days": int(
                (
                    group["candidate_upward_crossing_comparable"]
                    & group["weather_correction_applied"]
                ).sum()
            ),
            "baseline_upward_crossings_comparable": int(
                group["baseline_upward_crossing_comparable"].sum()
            ),
            "baseline_crossings_on_applied_weather_days": int(
                (
                    group["baseline_upward_crossing_comparable"]
                    & group["weather_correction_applied"]
                ).sum()
            ),
            "score_source_transition_days": int(group["score_source_transition"].sum()),
            "raw_crossings_on_source_transition_days": int(
                group["raw_crossing_on_source_transition"].sum()
            ),
            "candidate_messages": int(group["message_issued_candidate"].sum()),
            "baseline_messages": int(group["message_issued_baseline"].sum()),
            "message_changed_days": int(group["message_changed"].sum()),
            "changed_score_any_message_days": int(
                (
                    group["score_changed"]
                    & (
                        group["message_issued_candidate"]
                        | group["message_issued_baseline"]
                    )
                ).sum()
            ),
            "candidate_suppressed_days": int(
                group["suppressed_repeat_candidate"].sum()
            ),
            "baseline_suppressed_days": int(
                group["suppressed_repeat_baseline"].sum()
            ),
            "factual_suppression_either_days": int(
                group["factual_suppression_either"].sum()
            ),
            "changed_score_suppressed_days": int(diagnostic.sum()),
            "changed_score_suppressed_field_seasons": int(
                group.loc[diagnostic, "field_season"].nunique()
            ),
            "changed_score_suppressed_episodes": int(
                group_episodes["episode_kind"].eq("changed_score_suppressed").sum()
            ),
            "candidate_suppressed_field_seasons": int(
                group.loc[group["suppressed_repeat_candidate"], "field_season"].nunique()
            ),
            "candidate_suppressed_episodes": int(
                group_episodes["episode_kind"].eq("candidate_suppressed").sum()
            ),
            "baseline_suppressed_field_seasons": int(
                group.loc[group["suppressed_repeat_baseline"], "field_season"].nunique()
            ),
            "baseline_suppressed_episodes": int(
                group_episodes["episode_kind"].eq("baseline_suppressed").sum()
            ),
            "posthoc_timely_suppressed_days": int(posthoc.sum()),
            "posthoc_timely_suppressed_events": int(
                group.loc[posthoc, "field_season"].nunique()
            ),
            "alpha_zero_score_mismatch_days": int(
                (pd.to_numeric(group["alpha_candidate"]).eq(0) & group["score_changed"]).sum()
            ),
        }

    summaries = [summarize(days, "pooled", pd.NA)]
    summaries.extend(
        summarize(group, "year", int(season))
        for season, group in days.groupby("season", sort=True)
    )
    summary = pd.DataFrame(summaries)

    days["day_key"] = [
        _day_key(field, season, day, key_namespace)
        for field, season, day in zip(
            days["field_season"], days["season"], days["issue_date"]
        )
    ]
    days["event_key"] = [
        _event_key(field, season, key_namespace)
        for field, season in zip(days["field_season"], days["season"])
    ]
    days["candidate"] = candidate
    days["baseline"] = baseline
    days["evaluation_scope"] = evaluation_scope
    safe_columns = [
        "candidate",
        "baseline",
        "evaluation_scope",
        "day_key",
        "event_key",
        "season",
        "issue_date",
        "alpha_candidate",
        "score_candidate",
        "score_baseline",
        "policy_threshold_candidate",
        "policy_threshold_baseline",
        "weather_available",
        "weather_correction_applied",
        "effective_score_origin_candidate",
        "effective_score_origin_baseline",
        "score_changed",
        "candidate_above_threshold",
        "baseline_above_threshold",
        "above_threshold_either",
        "different_threshold_decision",
        "candidate_upward_crossing_raw",
        "candidate_upward_crossing_comparable",
        "baseline_upward_crossing_comparable",
        "score_source_transition",
        "raw_crossing_on_source_transition",
        "message_issued_candidate",
        "message_issued_baseline",
        "message_changed",
        "suppressed_repeat_candidate",
        "suppressed_repeat_baseline",
        "factual_suppression_either",
        "changed_score_suppressed",
        "posthoc_timely_day",
    ]
    return {
        "suppression_audit_summary": summary,
        "suppression_audit_days": days[safe_columns].reset_index(drop=True),
        "suppression_episodes": episodes.reset_index(drop=True),
    }
