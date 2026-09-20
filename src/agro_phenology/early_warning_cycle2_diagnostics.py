"""Post-hoc diagnostics for the immutable first-cycle late-blight run.

The functions in this module read or transform saved first-cycle artefacts.  They
do not fit a model and never write into the supplied run directory.  Event dates
are used only by the evaluator; operational scores and notification states are
taken verbatim from the saved run.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .early_warning_models import Policy, burden_metrics, event_metrics, simulate_policy


PRIMARY_YEARS = (2020, 2025)
PRIMARY_SCOPE = "paired_candidate_days"
PRIMARY_SLICE = "A_plus_B"
V3_EVENT_PAIRS = (
    ("calendar_window", "C4"),
    ("calendar_window", "C5"),
    ("C1", "C4"),
    ("C0", "C4"),
)
V3_DIAGNOSTIC_MODELS = tuple(dict.fromkeys(model for pair in V3_EVENT_PAIRS for model in pair))
MISS_REASON_FLAGS = (
    "reason_no_actionable_date",
    "reason_weather_unavailable_all",
    "reason_weather_unavailable_partial",
    "reason_score_unavailable",
    "reason_score_below_threshold",
    "reason_cooldown_suppression",
    "reason_active_alarm_from_earlier_message",
    "reason_only_early_messages",
    "reason_only_late_messages",
    "reason_early_and_late_messages",
    "reason_no_message_at_any_time",
)
POLYAKOV_COMPUTABLE_STATUSES = {
    "LOW_WEATHER_RISK",
    "CRITICAL_CONDITIONS",
    "PROLONGED_RISK",
    "OUTBREAK_EXPECTED",
}
POLYAKOV_POSITIVE_STATUSES = {"PROLONGED_RISK", "OUTBREAK_EXPECTED"}


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _year_filter(frame: pd.DataFrame, years: tuple[int, int]) -> pd.Series:
    return pd.to_numeric(frame["season"], errors="coerce").between(int(years[0]), int(years[1]))


def _json_dates(values: Iterable[Any]) -> str:
    dates = sorted({pd.Timestamp(value).date().isoformat() for value in values if pd.notna(value)})
    return json.dumps(dates, ensure_ascii=False, separators=(",", ":"))


def _json_values(values: Iterable[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def stable_event_key(field_season: object, season: int, namespace: str = "late_blight_cycle2") -> str:
    """Return a stable, one-way event key without exposing the source identifier."""
    payload = f"{namespace}\x1f{int(season)}\x1f{field_season}".encode("utf-8")
    return "evt_" + hashlib.sha256(payload).hexdigest()[:20]


def _primary_event_hits(
    event_hits: pd.DataFrame,
    *,
    scope: str,
    slice_name: str,
    years: tuple[int, int],
    models: Sequence[str] | None = None,
) -> pd.DataFrame:
    _require_columns(
        event_hits,
        ("field_season", "season", "model_code", "evaluation_scope", "slice", "warnable_event", "timely_hit"),
        "event_hits",
    )
    selected = event_hits[
        event_hits["evaluation_scope"].eq(scope)
        & event_hits["slice"].eq(slice_name)
        & event_hits["warnable_event"].astype(bool)
        & _year_filter(event_hits, years)
    ].copy()
    if models is not None:
        selected = selected[selected["model_code"].isin(models)].copy()
    duplicated = selected.duplicated(["model_code", "field_season", "season"], keep=False)
    if duplicated.any():
        raise ValueError("event_hits contains duplicate model/event rows in the selected population")
    return selected


def event_intersections(
    event_hits: pd.DataFrame,
    *,
    pairs: Sequence[tuple[str, str]] = V3_EVENT_PAIRS,
    scope: str = PRIMARY_SCOPE,
    slice_name: str = PRIMARY_SLICE,
    years: tuple[int, int] = PRIMARY_YEARS,
    event_key_namespace: str = "20260910_first_cycle_v3",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare timely hits for fixed policy pairs, pooled and by external year.

    Returns a compact summary and privacy-safe event membership.  The oracle
    union is explicitly labelled as a post-hoc diagnostic, not an operational
    policy or an information ceiling.
    """
    models = tuple(dict.fromkeys(model for pair in pairs for model in pair))
    selected = _primary_event_hits(
        event_hits, scope=scope, slice_name=slice_name, years=years, models=models
    )
    summaries: list[dict[str, Any]] = []
    memberships: list[pd.DataFrame] = []
    category_order = ("both_hit", "baseline_only_hit", "candidate_only_hit", "neither_hit")

    for baseline, candidate in pairs:
        left = selected[selected["model_code"].eq(baseline)][
            ["field_season", "season", "fold_id", "timely_hit"]
        ].rename(columns={"fold_id": "baseline_fold_id", "timely_hit": "baseline_hit"})
        right = selected[selected["model_code"].eq(candidate)][
            ["field_season", "season", "fold_id", "timely_hit"]
        ].rename(columns={"fold_id": "candidate_fold_id", "timely_hit": "candidate_hit"})
        joined = left.merge(right, on=["field_season", "season"], how="outer", validate="one_to_one", indicator=True)
        if not joined["_merge"].eq("both").all():
            raise ValueError(f"Unpaired event population for {baseline} and {candidate}")
        joined["baseline_hit"] = joined["baseline_hit"].astype(bool)
        joined["candidate_hit"] = joined["candidate_hit"].astype(bool)
        joined["category"] = np.select(
            [
                joined["baseline_hit"] & joined["candidate_hit"],
                joined["baseline_hit"] & ~joined["candidate_hit"],
                ~joined["baseline_hit"] & joined["candidate_hit"],
            ],
            category_order[:3],
            default="neither_hit",
        )
        joined["event_key"] = [
            stable_event_key(key, season, event_key_namespace)
            for key, season in zip(joined["field_season"], joined["season"])
        ]
        joined["pair_id"] = f"{baseline}__vs__{candidate}"
        memberships.append(
            joined[
                [
                    "event_key",
                    "season",
                    "pair_id",
                    "baseline_fold_id",
                    "candidate_fold_id",
                    "baseline_hit",
                    "candidate_hit",
                    "category",
                ]
            ].copy()
        )
        for aggregation, season, group in [
            ("pooled", pd.NA, joined),
            *[("year", int(year), value) for year, value in joined.groupby("season", sort=True)],
        ]:
            counts = group["category"].value_counts().reindex(category_order, fill_value=0)
            baseline_hits = int(group["baseline_hit"].sum())
            candidate_hits = int(group["candidate_hit"].sum())
            union_hits = int((group["baseline_hit"] | group["candidate_hit"]).sum())
            summaries.append(
                {
                    "pair_id": f"{baseline}__vs__{candidate}",
                    "baseline_model": baseline,
                    "candidate_model": candidate,
                    "evaluation_scope": scope,
                    "slice": slice_name,
                    "aggregation": aggregation,
                    "season": season,
                    "events": int(len(group)),
                    **{name: int(counts[name]) for name in category_order},
                    "baseline_hits": baseline_hits,
                    "candidate_hits": candidate_hits,
                    "oracle_union_hits": union_hits,
                    "oracle_union_recall": union_hits / len(group) if len(group) else np.nan,
                    "oracle_interpretation": (
                        "post_hoc_union_of_two_fixed_policies_not_operational_policy_or_information_ceiling"
                    ),
                }
            )
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary["season"] = summary["season"].astype("Int64")
    membership = pd.concat(memberships, ignore_index=True) if memberships else pd.DataFrame()
    return summary, membership


def event_policy_details(
    event_hits: pd.DataFrame,
    alarm_states: pd.DataFrame,
    notification_log: pd.DataFrame,
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    models: Sequence[str] = V3_DIAGNOSTIC_MODELS,
    scope: str = PRIMARY_SCOPE,
    slice_name: str = PRIMARY_SLICE,
    years: tuple[int, int] = PRIMARY_YEARS,
    event_key_namespace: str = "20260910_first_cycle_v3",
    prediction_source_label: str = "predictions.parquet",
) -> pd.DataFrame:
    """Build a privacy-safe event/model trace linked to immutable prediction rows."""
    _require_columns(events, ("field_season", "season", "first_recorded_event_date"), "events")
    _require_columns(
        alarm_states,
        ("field_season", "season", "model_code", "evaluation_scope", "issue_date", "message_issued"),
        "alarm_states",
    )
    _require_columns(
        notification_log,
        ("field_season", "season", "model_code", "evaluation_scope", "issue_date", "message_issued"),
        "notification_log",
    )
    _require_columns(
        predictions,
        ("field_season", "season", "model_code", "evaluation_scope", "issue_date", "score"),
        "predictions",
    )
    selected = _primary_event_hits(
        event_hits, scope=scope, slice_name=slice_name, years=years, models=models
    )
    event_dates = events[["field_season", "season", "first_recorded_event_date"]].drop_duplicates()
    if event_dates.duplicated(["field_season", "season"]).any():
        raise ValueError("events contains duplicate event keys")
    selected = selected.merge(event_dates, on=["field_season", "season"], how="left", validate="many_to_one")
    if selected["first_recorded_event_date"].isna().any():
        raise ValueError("Missing first event date for selected event")

    state_groups = {
        key: group
        for key, group in alarm_states[
            alarm_states["evaluation_scope"].eq(scope)
            & alarm_states["model_code"].isin(models)
            & _year_filter(alarm_states, years)
        ].groupby(["field_season", "season", "model_code"], sort=False)
    }
    notification_groups = {
        key: group
        for key, group in notification_log[
            notification_log["evaluation_scope"].eq(scope)
            & notification_log["model_code"].isin(models)
            & _year_filter(notification_log, years)
        ].groupby(["field_season", "season", "model_code"], sort=False)
    }
    indexed_predictions = predictions.reset_index(drop=True).copy()
    indexed_predictions["_prediction_row"] = np.arange(len(indexed_predictions), dtype=np.int64)
    prediction_groups = {
        key: group
        for key, group in indexed_predictions[
            indexed_predictions["evaluation_scope"].eq(scope)
            & indexed_predictions["model_code"].isin(models)
            & _year_filter(indexed_predictions, years)
        ].groupby(["field_season", "season", "model_code"], sort=False)
    }

    rows: list[dict[str, Any]] = []
    for hit in selected.itertuples(index=False):
        key = (hit.field_season, int(hit.season), hit.model_code)
        states = state_groups.get(key, pd.DataFrame())
        notices = notification_groups.get(key, pd.DataFrame())
        prediction = prediction_groups.get(key, pd.DataFrame())
        event_date = pd.Timestamp(hit.first_recorded_event_date)
        issued_states = states[states["message_issued"].astype(bool)] if len(states) else states
        issued_notices = notices[notices["message_issued"].astype(bool)] if len(notices) else notices
        if _json_dates(issued_states.get("issue_date", [])) != _json_dates(issued_notices.get("issue_date", [])):
            raise ValueError(f"notification_log does not reproduce issued dates for {hit.model_code}")
        if len(prediction):
            issue_dates = pd.to_datetime(prediction["issue_date"])
            actionable = prediction[issue_dates.between(event_date - pd.Timedelta(days=10), event_date - pd.Timedelta(days=3))]
            references = [f"{prediction_source_label}#row={int(index)}" for index in actionable["_prediction_row"]]
        else:
            references = []
        issued_dates = pd.to_datetime(issued_notices.get("issue_date", pd.Series(dtype="datetime64[ns]")))
        leads = [(event_date - date).days for date in issued_dates]
        rows.append(
            {
                "event_key": stable_event_key(hit.field_season, hit.season, event_key_namespace),
                "season": int(hit.season),
                "fold_id": hit.fold_id,
                "evaluation_scope": scope,
                "model_code": hit.model_code,
                "first_recorded_event_date": event_date.date().isoformat(),
                "timely_hit": bool(hit.timely_hit),
                "timely_message_dates_json": _json_dates(date for date, lead in zip(issued_dates, leads) if 3 <= lead <= 10),
                "early_message_dates_json": _json_dates(date for date, lead in zip(issued_dates, leads) if lead > 10),
                "late_message_dates_json": _json_dates(date for date, lead in zip(issued_dates, leads) if lead < 3),
                "all_issued_message_dates_json": _json_dates(issued_dates),
                "daily_prediction_refs_json": _json_values(references),
                "prediction_locator": "zero_based_row_in_immutable_parquet",
            }
        )
    detail = pd.DataFrame(rows)
    forbidden = {"field_season", "field_uid", "final_latitude", "final_longitude", "weather_cell"}
    if forbidden & set(detail.columns):
        raise AssertionError("Privacy-sensitive identifier leaked into event detail")
    return detail.sort_values(["season", "event_key", "model_code"]).reset_index(drop=True)


def miss_reason_diagnostics(
    event_hits: pd.DataFrame,
    alarm_states: pd.DataFrame,
    events: pd.DataFrame,
    *,
    models: Sequence[str] = V3_DIAGNOSTIC_MODELS,
    pairs: Sequence[tuple[str, str]] = V3_EVENT_PAIRS,
    scope: str = PRIMARY_SCOPE,
    slice_name: str = PRIMARY_SLICE,
    years: tuple[int, int] = PRIMARY_YEARS,
    weather_availability_column: str = "candidate_comparison_complete",
    event_key_namespace: str = "20260910_first_cycle_v3",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify saved misses without changing scores, thresholds, or policy.

    Reason flags are deliberately multi-label.  ``primary_reason`` partitions
    misses by the first operational barrier, while early/late message flags are
    descriptive timing diagnostics and can overlap that partition.
    """
    _require_columns(events, ("field_season", "season", "first_recorded_event_date"), "events")
    _require_columns(
        alarm_states,
        (
            "field_season",
            "season",
            "model_code",
            "evaluation_scope",
            "issue_date",
            "service_active",
            "days_to_first_recorded_event",
            "score",
            "policy_threshold",
            "message_issued",
            "suppressed_repeat",
            "alarm_active",
            weather_availability_column,
        ),
        "alarm_states",
    )
    selected = _primary_event_hits(
        event_hits, scope=scope, slice_name=slice_name, years=years, models=models
    )
    event_dates = events[["field_season", "season", "first_recorded_event_date"]].drop_duplicates()
    selected = selected.merge(event_dates, on=["field_season", "season"], how="left", validate="many_to_one")
    states_selected = alarm_states[
        alarm_states["evaluation_scope"].eq(scope)
        & alarm_states["model_code"].isin(models)
        & _year_filter(alarm_states, years)
    ]
    groups = {
        key: group.sort_values("issue_date")
        for key, group in states_selected.groupby(["field_season", "season", "model_code"], sort=False)
    }
    rows: list[dict[str, Any]] = []
    for hit in selected.itertuples(index=False):
        key = (hit.field_season, int(hit.season), hit.model_code)
        states = groups.get(key)
        if states is None:
            raise ValueError(f"No alarm states for selected event/model: {hit.model_code}")
        event_date = pd.Timestamp(hit.first_recorded_event_date)
        active = states[states["service_active"].astype(bool)]
        lead = (event_date - pd.to_datetime(active["issue_date"])).dt.days
        actionable = active[lead.between(3, 10)]
        early = active[lead.gt(10) & active["message_issued"].astype(bool)]
        late = active[lead.lt(3) & active["message_issued"].astype(bool)]
        allowed_days = int(len(actionable))
        weather_days = int(actionable[weather_availability_column].fillna(False).astype(bool).sum())
        computed = actionable["score"].notna()
        computed_days = int(computed.sum())
        thresholds = states["policy_threshold"].dropna().unique()
        if len(thresholds) > 1:
            raise ValueError(f"Policy threshold changes inside one event window for {hit.model_code}")
        threshold = float(thresholds[0]) if len(thresholds) else np.nan
        maximum_score = float(actionable.loc[computed, "score"].max()) if computed_days else np.nan
        crossed = computed & actionable["score"].ge(actionable["policy_threshold"])
        crossing_days = int(crossed.sum())
        issued_days = int(actionable["message_issued"].astype(bool).sum())
        suppressed_days = int(actionable["suppressed_repeat"].astype(bool).sum())
        alarm_days = int(actionable["alarm_active"].astype(bool).sum())
        missed = not bool(hit.timely_hit)
        no_date = missed and allowed_days == 0
        no_weather = missed and allowed_days > 0 and weather_days == 0
        partial_weather = missed and 0 < weather_days < allowed_days
        score_unavailable = missed and computed_days == 0
        below = missed and computed_days > 0 and crossing_days == 0
        cooldown = missed and crossing_days > 0 and issued_days == 0 and suppressed_days > 0
        early_any, late_any = bool(len(early)), bool(len(late))
        flags = {
            "reason_no_actionable_date": no_date,
            "reason_weather_unavailable_all": no_weather,
            "reason_weather_unavailable_partial": partial_weather,
            "reason_score_unavailable": score_unavailable,
            "reason_score_below_threshold": below,
            "reason_cooldown_suppression": cooldown,
            "reason_active_alarm_from_earlier_message": missed and issued_days == 0 and alarm_days > 0,
            "reason_only_early_messages": missed and early_any and not late_any,
            "reason_only_late_messages": missed and late_any and not early_any,
            "reason_early_and_late_messages": missed and early_any and late_any,
            "reason_no_message_at_any_time": missed and not early_any and not late_any,
        }
        if not missed:
            primary_reason = "timely_hit"
        elif no_date:
            primary_reason = "no_actionable_date"
        elif score_unavailable:
            primary_reason = "score_unavailable"
        elif below:
            primary_reason = "score_below_threshold"
        elif cooldown:
            primary_reason = "cooldown_suppression"
        else:
            primary_reason = "unclassified_policy_path"
        rows.append(
            {
                "event_key": stable_event_key(hit.field_season, hit.season, event_key_namespace),
                "season": int(hit.season),
                "fold_id": hit.fold_id,
                "evaluation_scope": scope,
                "model_code": hit.model_code,
                "timely_hit": bool(hit.timely_hit),
                "actionable_dates": allowed_days,
                "weather_available_dates": weather_days,
                "computed_score_dates": computed_days,
                "maximum_actionable_score": maximum_score,
                "policy_threshold": threshold,
                "maximum_score_minus_threshold": (
                    maximum_score - threshold if computed_days and pd.notna(threshold) else np.nan
                ),
                "threshold_crossing_dates": crossing_days,
                "timely_message_dates": issued_days,
                "cooldown_suppressed_dates": suppressed_days,
                "active_alarm_dates_without_timely_message": alarm_days if missed and issued_days == 0 else 0,
                "early_message_count": int(len(early)),
                "late_message_count": int(len(late)),
                "primary_reason": primary_reason,
                "reason_semantics": "post_hoc_operational_diagnostic_not_biological_cause",
                **flags,
            }
        )
    detail = pd.DataFrame(rows)
    _, membership = event_intersections(
        event_hits,
        pairs=pairs,
        scope=scope,
        slice_name=slice_name,
        years=years,
        event_key_namespace=event_key_namespace,
    )
    summary_rows: list[dict[str, Any]] = []
    for pair_id, pair_events in membership.groupby("pair_id", sort=False):
        baseline, candidate = pair_id.split("__vs__", maxsplit=1)
        diagnostic_targets = []
        for row in pair_events.itertuples(index=False):
            if row.category == "baseline_only_hit":
                diagnostic_targets.append((row.event_key, row.season, row.category, candidate))
            elif row.category == "candidate_only_hit":
                diagnostic_targets.append((row.event_key, row.season, row.category, baseline))
            elif row.category == "neither_hit":
                diagnostic_targets.extend(
                    [(row.event_key, row.season, row.category, baseline), (row.event_key, row.season, row.category, candidate)]
                )
        targets = pd.DataFrame(diagnostic_targets, columns=["event_key", "season", "overlap_category", "missed_model"])
        if targets.empty:
            continue
        joined = targets.merge(
            detail,
            left_on=["event_key", "season", "missed_model"],
            right_on=["event_key", "season", "model_code"],
            how="left",
            validate="many_to_one",
        )
        if joined["model_code"].isna().any():
            raise ValueError(f"Missing event diagnostic for pair {pair_id}")
        groupings = [("pooled", pd.NA, joined)] + [
            ("year", int(year), group) for year, group in joined.groupby("season", sort=True)
        ]
        for aggregation, season, frame in groupings:
            for (category, missed_model), group in frame.groupby(["overlap_category", "missed_model"], sort=True):
                summary_rows.append(
                    {
                        "pair_id": pair_id,
                        "aggregation": aggregation,
                        "season": season,
                        "overlap_category": category,
                        "missed_model": missed_model,
                        "missed_events": int(len(group)),
                        **{flag: int(group[flag].astype(bool).sum()) for flag in MISS_REASON_FLAGS},
                    }
                )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["season"] = summary["season"].astype("Int64")
    return detail.sort_values(["season", "event_key", "model_code"]).reset_index(drop=True), summary


def yearly_funnel(
    field_seasons: pd.DataFrame,
    daily_decisions: pd.DataFrame,
    event_hits: pd.DataFrame,
    *,
    era_daily: pd.DataFrame | None = None,
    hit_models: Sequence[str] = ("calendar_window", "C0", "C1", "C4", "C5"),
    scope: str = PRIMARY_SCOPE,
    slice_name: str = PRIMARY_SLICE,
    years: tuple[int, int] = PRIMARY_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the external-year event/field-day funnel and named loss counts."""
    _require_columns(
        field_seasons,
        (
            "field_season",
            "season",
            "first_visit_available_date",
            "first_recorded_event_date",
            "warnable_first_event",
        ),
        "field_seasons",
    )
    _require_columns(
        daily_decisions,
        (
            "field_season",
            "season",
            "issue_date",
            "evaluation_field_day",
            "service_active",
            "days_to_first_recorded_event",
            "nasa_common_complete",
            "era_common_complete",
            "episode_weather_complete",
            "candidate_comparison_complete",
        ),
        "daily_decisions",
    )
    hits = _primary_event_hits(
        event_hits, scope=scope, slice_name=slice_name, years=years, models=hit_models
    )
    seasons = field_seasons[_year_filter(field_seasons, years)].copy()
    days = daily_decisions[_year_filter(daily_decisions, years) & daily_decisions["evaluation_field_day"].astype(bool)].copy()
    seasons["after_connection"] = seasons["first_recorded_event_date"].notna() & (
        pd.to_datetime(seasons["first_recorded_event_date"])
        >= pd.to_datetime(seasons["first_visit_available_date"])
    )

    event_weather_rows: list[dict[str, Any]] = []
    warnable = seasons[seasons["warnable_first_event"].astype(bool)]
    day_groups = {key: group for key, group in days.groupby("field_season", sort=False)}
    for event in warnable.itertuples(index=False):
        event_days = day_groups.get(event.field_season, pd.DataFrame())
        if len(event_days):
            window = event_days[event_days["days_to_first_recorded_event"].between(3, 10)]
        else:
            window = event_days
        availability = window["candidate_comparison_complete"].fillna(False).astype(bool) if len(window) else pd.Series(dtype=bool)
        event_weather_rows.append(
            {
                "field_season": event.field_season,
                "season": int(event.season),
                "actionable_dates": int(len(window)),
                "any_common_weather": bool(availability.any()),
                "all_common_weather": bool(len(window) and availability.all()),
            }
        )
    event_weather = pd.DataFrame(event_weather_rows)

    rows: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    groups = [("pooled", pd.NA, seasons, days)] + [
        ("year", int(year), seasons[seasons["season"].eq(year)], days[days["season"].eq(year)])
        for year in range(int(years[0]), int(years[1]) + 1)
    ]
    for aggregation, year, season_group, day_group in groups:
        event_group = season_group[season_group["first_recorded_event_date"].notna()]
        weather_group = event_weather if aggregation == "pooled" else event_weather[event_weather["season"].eq(year)]
        hit_group = hits if aggregation == "pooled" else hits[hits["season"].eq(year)]
        main_events = hit_group[hit_group["model_code"].eq(hit_models[0])]
        result: dict[str, Any] = {
            "aggregation": aggregation,
            "season": year,
            "field_seasons_total": int(len(season_group)),
            "field_seasons_with_service_days": int(day_group["field_season"].nunique()),
            "field_seasons_with_common_weather": int(
                day_group.loc[day_group["candidate_comparison_complete"].astype(bool), "field_season"].nunique()
            ),
            "all_first_events": int(len(event_group)),
            "events_after_connection": int(event_group["after_connection"].sum()),
            "events_with_actionable_time": int(event_group["warnable_first_event"].astype(bool).sum()),
            "main_registry_events": int(len(main_events)),
            "events_with_any_common_weather_in_window": int(weather_group["any_common_weather"].sum()) if len(weather_group) else 0,
            "events_with_all_common_weather_in_window": int(weather_group["all_common_weather"].sum()) if len(weather_group) else 0,
            "service_field_days": int(len(day_group)),
            "service_unique_calendar_dates": int(day_group["issue_date"].nunique()),
            "nasa_common_days": int(day_group["nasa_common_complete"].astype(bool).sum()),
            "era_common_days": int(day_group["era_common_complete"].astype(bool).sum()),
            "episode_weather_days": int(day_group["episode_weather_complete"].astype(bool).sum()),
            "common_candidate_days": int(day_group["candidate_comparison_complete"].astype(bool).sum()),
            "common_candidate_unique_calendar_dates": int(
                day_group.loc[day_group["candidate_comparison_complete"].astype(bool), "issue_date"].nunique()
            ),
        }
        for model in hit_models:
            model_hits = hit_group[hit_group["model_code"].eq(model)]
            result[f"{model}_timely_hits"] = int(model_hits["timely_hit"].astype(bool).sum())
        rows.append(result)
        loss_values = {
            "first_event_positive_known_at_entry": int(len(event_group) - event_group["after_connection"].sum()),
            "after_connection_without_actionable_lead": int(
                event_group["after_connection"].sum() - event_group["warnable_first_event"].astype(bool).sum()
            ),
            "actionable_event_outside_main_registry_population": int(
                event_group["warnable_first_event"].astype(bool).sum() - len(main_events)
            ),
            "main_event_without_any_common_weather": int(len(main_events) - weather_group["any_common_weather"].sum()),
            "main_event_with_partial_common_weather": int(
                weather_group["any_common_weather"].sum() - weather_group["all_common_weather"].sum()
            ),
            "field_season_without_service_day": int(len(season_group) - day_group["field_season"].nunique()),
            "service_day_missing_nasa_common": int(len(day_group) - day_group["nasa_common_complete"].astype(bool).sum()),
            "service_day_missing_era_common": int(len(day_group) - day_group["era_common_complete"].astype(bool).sum()),
            "service_day_missing_episode_weather": int(len(day_group) - day_group["episode_weather_complete"].astype(bool).sum()),
            "service_day_outside_common_candidate_mask": int(
                len(day_group) - day_group["candidate_comparison_complete"].astype(bool).sum()
            ),
        }
        losses.extend(
            {"aggregation": aggregation, "season": year, "loss_reason": reason, "count": value}
            for reason, value in loss_values.items()
        )

    if era_daily is not None and len(days):
        _require_columns(era_daily, ("weather_cell", "date"), "era_daily")
        _require_columns(days, ("weather_cell", "feature_cutoff_era_common_date"), "daily_decisions")
        era = era_daily.copy()
        era["season"] = pd.to_datetime(era["date"]).dt.year
        spans = era.groupby(["weather_cell", "season"], as_index=False).agg(
            era_first_date=("date", "min"), era_last_date=("date", "max")
        )
        classified = days.merge(spans, on=["weather_cell", "season"], how="left", validate="many_to_one")
        missing = classified[~classified["candidate_comparison_complete"].astype(bool)].copy()
        cutoff = pd.to_datetime(missing["feature_cutoff_era_common_date"])
        first_complete = pd.to_datetime(missing["era_first_date"]) + pd.Timedelta(days=29)
        last = pd.to_datetime(missing["era_last_date"])
        missing["candidate_missing_reason"] = "internal_or_cross_source_incomplete"
        missing.loc[missing["era_first_date"].isna(), "candidate_missing_reason"] = "no_matching_era_cell_season"
        missing.loc[missing["era_first_date"].notna() & cutoff.lt(first_complete), "candidate_missing_reason"] = "before_30_day_era_history"
        missing.loc[missing["era_last_date"].notna() & cutoff.gt(last), "candidate_missing_reason"] = "after_frozen_era_series_end"
        for aggregation, year, group in [("pooled", pd.NA, missing)] + [
            ("year", int(value), missing[missing["season"].eq(value)])
            for value in range(int(years[0]), int(years[1]) + 1)
        ]:
            counts = group["candidate_missing_reason"].value_counts()
            losses.extend(
                {
                    "aggregation": aggregation,
                    "season": year,
                    "loss_reason": f"common_candidate_day_{reason}",
                    "count": int(count),
                }
                for reason, count in counts.items()
            )
    funnel = pd.DataFrame(rows)
    loss_frame = pd.DataFrame(losses)
    if not funnel.empty:
        funnel["season"] = funnel["season"].astype("Int64")
    if not loss_frame.empty:
        loss_frame["season"] = loss_frame["season"].astype("Int64")
    return funnel, loss_frame


def _saved_policy_summary(
    event_hits: pd.DataFrame,
    alarm_states: pd.DataFrame,
    model_code: str,
    scope: str,
    slice_name: str,
    years: tuple[int, int],
) -> pd.DataFrame:
    hits = _primary_event_hits(
        event_hits, scope=scope, slice_name=slice_name, years=years, models=(model_code,)
    )
    states = alarm_states[
        alarm_states["model_code"].eq(model_code)
        & alarm_states["evaluation_scope"].eq(scope)
        & _year_filter(alarm_states, years)
        & alarm_states["evaluation_scope_day"].astype(bool)
    ].copy()
    rows: list[dict[str, Any]] = []
    for aggregation, year, hit_group, state_group in [("pooled", pd.NA, hits, states)] + [
        (
            "year",
            int(value),
            hits[hits["season"].eq(value)],
            states[states["season"].eq(value)],
        )
        for value in range(int(years[0]), int(years[1]) + 1)
    ]:
        opportunities = int(len(hit_group))
        timely = int(hit_group["timely_hit"].astype(bool).sum())
        field_days = int(len(state_group))
        messages = int(state_group["message_issued"].astype(bool).sum())
        rows.append(
            {
                "model_code": model_code,
                "evaluation_scope": scope,
                "slice": slice_name,
                "aggregation": aggregation,
                "season": year,
                "events_with_warning_opportunity": opportunities,
                "timely_hits": timely,
                "timely_recall": timely / opportunities if opportunities else np.nan,
                "field_days": field_days,
                "field_seasons": int(state_group["field_season"].nunique()),
                "messages": messages,
                "messages_per_30_field_days": 30 * messages / field_days if field_days else np.nan,
                "active_alarm_days": int(state_group["alarm_active"].astype(bool).sum()),
                "active_alarm_fraction": float(state_group["alarm_active"].astype(bool).mean()) if field_days else np.nan,
                "computable_days": int(state_group["score"].notna().sum()),
                "computable_fraction": float(state_group["score"].notna().mean()) if field_days else np.nan,
                "suppressed_repeats": int(state_group["suppressed_repeat"].astype(bool).sum()),
            }
        )
    result = pd.DataFrame(rows)
    result["season"] = result["season"].astype("Int64")
    return result


def extract_periodic_30d(
    event_hits: pd.DataFrame,
    alarm_states: pd.DataFrame,
    *,
    scopes: Sequence[str] = ("service_calendar", PRIMARY_SCOPE),
    slice_name: str = PRIMARY_SLICE,
    years: tuple[int, int] = PRIMARY_YEARS,
) -> pd.DataFrame:
    """Extract the saved control and expose its actual monthly-day-1 semantics."""
    frames = []
    expected_month_days = {(6, 1), (7, 1), (8, 1)}
    for scope in scopes:
        summary = _saved_policy_summary(event_hits, alarm_states, "periodic_30d", scope, slice_name, years)
        issued = alarm_states[
            alarm_states["model_code"].eq("periodic_30d")
            & alarm_states["evaluation_scope"].eq(scope)
            & _year_filter(alarm_states, years)
            & alarm_states["message_issued"].astype(bool)
        ]
        observed = {
            (pd.Timestamp(value).month, pd.Timestamp(value).day) for value in issued["issue_date"]
        }
        if not observed.issubset(expected_month_days):
            raise ValueError(f"Saved periodic_30d has unexpected issue dates in {scope}: {sorted(observed)}")
        summary["observed_message_month_days"] = "|".join(
            f"{month:02d}-{day:02d}" for month, day in sorted(observed)
        )
        frames.append(summary)
    result = pd.concat(frames, ignore_index=True)
    result["implementation"] = "June_1_July_1_August_1_when_field_is_active"
    result["nominal_name_warning"] = "periodic_30d_is_monthly_calendar_rule_not_exact_30_day_spacing"
    result["phase_selected_on_external_outcome"] = False
    return result


def periodic_k_all_phases(
    daily_decisions: pd.DataFrame,
    field_seasons: pd.DataFrame,
    *,
    period_days: int = 17,
    annual_anchor_month_day: str = "05-01",
    active_days: int = 7,
    cooldown_days: int = 15,
    scopes: Mapping[str, str | None] | None = None,
    years: tuple[int, int] = PRIMARY_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate every predeclared phase of a fixed-period diagnostic control."""
    if period_days < 1:
        raise ValueError("period_days must be positive")
    if scopes is None:
        scopes = {"service_calendar": None, PRIMARY_SCOPE: "candidate_comparison_complete"}
    service = daily_decisions[_year_filter(daily_decisions, years)].copy()
    seasons = field_seasons[_year_filter(field_seasons, years)].copy()
    _require_columns(service, ("issue_date", "service_active", "field_season", "season"), "daily_decisions")
    try:
        anchor_month, anchor_day = (int(value) for value in annual_anchor_month_day.split("-", maxsplit=1))
        anchors = pd.to_datetime(
            {
                "year": pd.to_numeric(service["season"], errors="raise").astype(int),
                "month": anchor_month,
                "day": anchor_day,
            }
        )
    except (TypeError, ValueError) as error:
        raise ValueError("annual_anchor_month_day must be a valid MM-DD value") from error
    offsets = (pd.to_datetime(service["issue_date"]).reset_index(drop=True) - anchors.reset_index(drop=True)).dt.days.mod(period_days)
    offsets.index = service.index
    rows: list[dict[str, Any]] = []
    policy = Policy(0.5, active_days, cooldown_days, f"diagnostic_periodic_{period_days}d_all_phases")
    for phase in range(period_days):
        base_score = offsets.eq(phase).astype(float)
        for scope, mask_column in scopes.items():
            if mask_column is None:
                mask = None
                score = base_score
            else:
                _require_columns(service, (mask_column,), "daily_decisions")
                mask = service[mask_column].fillna(False).astype(bool)
                score = base_score.where(mask)
            states = simulate_policy(service, score, policy, scope, mask)
            evaluation_groups = [("pooled", pd.NA, states, seasons)] + [
                (
                    "year",
                    int(year),
                    states[states["season"].eq(year)],
                    seasons[seasons["season"].eq(year)],
                )
                for year in range(int(years[0]), int(years[1]) + 1)
            ]
            for aggregation, season, state_group, season_group in evaluation_groups:
                event, _ = event_metrics(
                    state_group,
                    season_group,
                    f"periodic_{period_days}d_phase_{phase}",
                    "diagnostic",
                    PRIMARY_SLICE,
                    scope,
                )
                burden = burden_metrics(
                    state_group,
                    f"periodic_{period_days}d_phase_{phase}",
                    "diagnostic",
                    PRIMARY_SLICE,
                    scope,
                )
                rows.append(
                    {
                        "model_code": f"periodic_{period_days}d_all_phases",
                        "phase": phase,
                        "phase_origin": f"annual-{annual_anchor_month_day}",
                        "period_days": period_days,
                        "evaluation_scope": scope,
                        "aggregation": aggregation,
                        "season": season,
                        "events_with_warning_opportunity": event["events_with_warning_opportunity"],
                        "timely_hits": event["timely_hits"],
                        "timely_recall": event["timely_recall"],
                        "messages": burden["messages"],
                        "messages_per_30_field_days": burden["messages_per_30_field_days"],
                        "active_alarm_fraction": burden["active_alarm_fraction"],
                        "field_days": burden["field_days"],
                        "computable_fraction": burden["computable_fraction"],
                        "suppressed_repeats": burden["suppressed_repeats"],
                        "interpretation": "all_fixed_phases_reported_no_external_phase_selection",
                    }
                )
    phase_results = pd.DataFrame(rows)
    phase_results["season"] = phase_results["season"].astype("Int64")
    summary = (
        phase_results.groupby(
            ["model_code", "period_days", "phase_origin", "evaluation_scope", "aggregation", "season"],
            as_index=False,
            dropna=False,
        )
        .agg(
            phases=("phase", "size"),
            timely_hits_min=("timely_hits", "min"),
            timely_hits_median=("timely_hits", "median"),
            timely_hits_max=("timely_hits", "max"),
            timely_recall_mean=("timely_recall", "mean"),
            messages_per_30_min=("messages_per_30_field_days", "min"),
            messages_per_30_mean=("messages_per_30_field_days", "mean"),
            messages_per_30_median=("messages_per_30_field_days", "median"),
            messages_per_30_max=("messages_per_30_field_days", "max"),
            active_alarm_fraction_min=("active_alarm_fraction", "min"),
            active_alarm_fraction_median=("active_alarm_fraction", "median"),
            active_alarm_fraction_max=("active_alarm_fraction", "max"),
        )
    )
    summary["season"] = summary["season"].astype("Int64")
    summary["phase_selection"] = "none_all_phases_are_diagnostic"
    return phase_results, summary


def polyakov_computability(
    event_hits: pd.DataFrame,
    alarm_states: pd.DataFrame,
    daily_decisions: pd.DataFrame,
    field_seasons: pd.DataFrame,
    *,
    scopes: Sequence[str] = ("service_calendar", PRIMARY_SCOPE),
    years: tuple[int, int] = PRIMARY_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate Polyakov's overall coverage from quality when it is computable."""
    decisions = daily_decisions[
        _year_filter(daily_decisions, years) & daily_decisions["evaluation_field_day"].astype(bool)
    ].copy()
    _require_columns(
        decisions,
        (
            "field_season",
            "season",
            "issue_date",
            "days_to_first_recorded_event",
            "candidate_comparison_complete",
            "polyakov_status",
            "polyakov_score",
        ),
        "daily_decisions",
    )
    status_computable = decisions["polyakov_status"].isin(POLYAKOV_COMPUTABLE_STATUSES)
    if not status_computable.eq(decisions["polyakov_score"].notna()).all():
        raise ValueError("polyakov_score missingness is inconsistent with saved rule statuses")
    expected_positive = decisions["polyakov_status"].isin(POLYAKOV_POSITIVE_STATUSES)
    actual_positive = decisions["polyakov_score"].eq(1.0)
    if not expected_positive.eq(actual_positive).all():
        raise ValueError("polyakov_score values are inconsistent with saved positive statuses")

    _require_columns(
        field_seasons,
        (
            "field_season",
            "season",
            "warnable_first_event",
            "first_observed_bbch51_available_date",
        ),
        "field_seasons",
    )
    warnable_seasons = field_seasons[
        _year_filter(field_seasons, years) & field_seasons["warnable_first_event"].astype(bool)
    ].copy()
    decision_groups = {key: group for key, group in decisions.groupby("field_season", sort=False)}
    component_rows: list[dict[str, Any]] = []
    for event in warnable_seasons.itertuples(index=False):
        event_days = decision_groups.get(event.field_season, pd.DataFrame())
        window = (
            event_days[event_days["days_to_first_recorded_event"].between(3, 10)]
            if len(event_days)
            else event_days
        )
        bbch_date = (
            pd.Timestamp(event.first_observed_bbch51_available_date)
            if pd.notna(event.first_observed_bbch51_available_date)
            else pd.NaT
        )
        causal_bbch = bool(
            pd.notna(bbch_date)
            and len(window)
            and pd.to_datetime(window["issue_date"]).ge(bbch_date).any()
        )
        component_rows.append(
            {
                "season": int(event.season),
                "bbch_observed_ever": bool(pd.notna(bbch_date)),
                "bbch_causally_available_in_window": causal_bbch,
                "common_weather_in_window": bool(
                    len(window) and window["candidate_comparison_complete"].fillna(False).astype(bool).any()
                ),
                "polyakov_computable_in_window": bool(len(window) and window["polyakov_score"].notna().any()),
                "polyakov_positive_in_window": bool(len(window) and window["polyakov_score"].eq(1.0).any()),
            }
        )
    components = pd.DataFrame(component_rows)
    component_summary_rows: list[dict[str, Any]] = []
    for aggregation, year, group in [("pooled", pd.NA, components)] + [
        ("year", int(value), components[components["season"].eq(value)])
        for value in range(int(years[0]), int(years[1]) + 1)
    ]:
        component_summary_rows.append(
            {
                "aggregation": aggregation,
                "season": year,
                "bbch_observed_ever_events": int(group["bbch_observed_ever"].sum()),
                "bbch_causally_available_events": int(group["bbch_causally_available_in_window"].sum()),
                "common_weather_events": int(group["common_weather_in_window"].sum()),
                "positive_score_events": int(group["polyakov_positive_in_window"].sum()),
            }
        )
    component_summary = pd.DataFrame(component_summary_rows)
    component_summary["season"] = component_summary["season"].astype("Int64")

    summaries: list[pd.DataFrame] = []
    for scope in scopes:
        summary = _saved_policy_summary(event_hits, alarm_states, "polyakov", scope, PRIMARY_SLICE, years)
        hits = _primary_event_hits(
            event_hits, scope=scope, slice_name=PRIMARY_SLICE, years=years, models=("polyakov",)
        )
        conditional_rows = []
        for aggregation, year, group in [("pooled", pd.NA, hits)] + [
            ("year", int(value), hits[hits["season"].eq(value)])
            for value in range(int(years[0]), int(years[1]) + 1)
        ]:
            computable = int(group["computable_in_actionable_window"].astype(bool).sum())
            timely = int(group["timely_hit"].astype(bool).sum())
            if (group["timely_hit"].astype(bool) & ~group["computable_in_actionable_window"].astype(bool)).any():
                raise ValueError("Polyakov timely hit is marked on a non-computable event")
            conditional_rows.append(
                {
                    "aggregation": aggregation,
                    "season": year,
                    "computable_events": computable,
                    "conditional_timely_recall": timely / computable if computable else np.nan,
                }
            )
        conditional = pd.DataFrame(conditional_rows)
        conditional["season"] = conditional["season"].astype("Int64")
        summary = summary.merge(conditional, on=["aggregation", "season"], how="left", validate="one_to_one")
        summary = summary.merge(
            component_summary, on=["aggregation", "season"], how="left", validate="one_to_one"
        )
        summary["computable_event_fraction"] = summary["computable_events"] / summary["events_with_warning_opportunity"]
        summary["mask_definition"] = (
            "saved_polyakov_score_requires_observed_BBCH51_available_by_issue_date_and_computable_ERA_rule;_"
            + ("also_candidate_comparison_complete" if scope == PRIMARY_SCOPE else "service_calendar_no_common_mask")
        )
        summaries.append(summary)
    combined = pd.concat(summaries, ignore_index=True)

    statuses: list[dict[str, Any]] = []
    for aggregation, year, group in [("pooled", pd.NA, decisions)] + [
        ("year", int(value), decisions[decisions["season"].eq(value)])
        for value in range(int(years[0]), int(years[1]) + 1)
    ]:
        counts = group["polyakov_status"].fillna("missing_status").value_counts()
        bbch_seasons = field_seasons[
            _year_filter(field_seasons, years)
            if aggregation == "pooled"
            else field_seasons["season"].eq(year)
        ]
        for status, count in counts.items():
            statuses.append(
                {
                    "aggregation": aggregation,
                    "season": year,
                    "polyakov_status": str(status),
                    "field_days": int(count),
                    "field_seasons_with_observed_bbch51": int(
                        bbch_seasons["first_observed_bbch51_available_date"].notna().sum()
                    ),
                }
            )
    status_frame = pd.DataFrame(statuses)
    if not status_frame.empty:
        status_frame["season"] = status_frame["season"].astype("Int64")
    return combined, status_frame


def c4_optuna_diagnostics(
    event_hits: pd.DataFrame,
    alarm_states: pd.DataFrame,
    policy_selection: pd.DataFrame,
    optuna_trials: pd.DataFrame,
    optuna_seed_checks: pd.DataFrame,
    *,
    years: tuple[int, int] = PRIMARY_YEARS,
    event_key_namespace: str = "20260910_first_cycle_v3",
) -> dict[str, pd.DataFrame]:
    """Describe fixed C4 versus the saved, internally selected Optuna variant."""
    intersections: list[pd.DataFrame] = []
    memberships: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    for scope in ("service_calendar", PRIMARY_SCOPE):
        summary, membership = event_intersections(
            event_hits,
            pairs=(("C4", "C4_optuna"),),
            scope=scope,
            years=years,
            event_key_namespace=event_key_namespace,
        )
        intersections.append(summary)
        memberships.append(membership)
        metrics.extend(
            _saved_policy_summary(event_hits, alarm_states, model, scope, PRIMARY_SLICE, years)
            for model in ("C4", "C4_optuna")
        )
    policy = policy_selection[
        policy_selection["model_code"].isin(["C4", "C4_optuna"])
        & policy_selection["evaluation_scope"].isin(["service_calendar", PRIMARY_SCOPE])
    ].copy()
    forbidden = {"field_season", "field_uid", "final_latitude", "final_longitude", "weather_cell"}
    policy = policy.drop(columns=list(forbidden & set(policy.columns)))

    trial_rows: list[dict[str, Any]] = []
    if len(optuna_trials):
        _require_columns(optuna_trials, ("fold_id", "state", "value", "params", "user_attrs"), "optuna_trials")
        for fold_id, group in optuna_trials.groupby("fold_id", sort=True):
            complete = group[group["state"].eq("COMPLETE") & group["value"].notna()]
            best = complete.loc[complete["value"].idxmax()] if len(complete) else None
            trial_rows.append(
                {
                    "fold_id": fold_id,
                    "trials": int(len(group)),
                    "complete_trials": int(len(complete)),
                    "best_internal_value": float(best["value"]) if best is not None else np.nan,
                    "best_internal_params": best["params"] if best is not None else "{}",
                    "best_internal_user_attrs": best["user_attrs"] if best is not None else "{}",
                    "selection_warning": "internal_validation_only_do_not_reselect_trial_on_external_outcome",
                }
            )
    trial_summary = pd.DataFrame(trial_rows)

    seed_rows: list[dict[str, Any]] = []
    if len(optuna_seed_checks):
        _require_columns(
            optuna_seed_checks,
            ("fold_id", "validation_timely_recall", "validation_messages_per_30", "validation_alarm_fraction"),
            "optuna_seed_checks",
        )
        for fold_id, group in optuna_seed_checks.groupby("fold_id", sort=True):
            seed_rows.append(
                {
                    "fold_id": fold_id,
                    "seeds": int(len(group)),
                    "validation_timely_recall_min": float(group["validation_timely_recall"].min()),
                    "validation_timely_recall_max": float(group["validation_timely_recall"].max()),
                    "validation_messages_per_30_min": float(group["validation_messages_per_30"].min()),
                    "validation_messages_per_30_max": float(group["validation_messages_per_30"].max()),
                    "validation_alarm_fraction_min": float(group["validation_alarm_fraction"].min()),
                    "validation_alarm_fraction_max": float(group["validation_alarm_fraction"].max()),
                }
            )
    return {
        "c4_optuna_intersections": pd.concat(intersections, ignore_index=True),
        "c4_optuna_event_membership": pd.concat(memberships, ignore_index=True),
        "c4_optuna_external_metrics": pd.concat(metrics, ignore_index=True),
        "c4_optuna_policy_selection": policy.reset_index(drop=True),
        "c4_optuna_trial_summary": trial_summary,
        "c4_optuna_seed_summary": pd.DataFrame(seed_rows),
    }


def load_v3_artifacts(v3_dir: str | Path) -> dict[str, Any]:
    """Load only the saved artefacts needed by cycle-2 diagnostics."""
    path = Path(v3_dir)
    parquet = (
        "event_hits",
        "alarm_states",
        "notification_log",
        "predictions",
        "events",
        "field_seasons",
        "daily_decisions",
    )
    csv = ("policy_selection", "optuna_trials", "optuna_seed_checks")
    result: dict[str, Any] = {name: pd.read_parquet(path / f"{name}.parquet") for name in parquet}
    result.update({name: pd.read_csv(path / f"{name}.csv") for name in csv})
    contract_path = path / "evaluation_contract.json"
    result["evaluation_contract"] = json.loads(contract_path.read_text(encoding="utf-8"))
    return result


def build_v3_cycle2_diagnostics(v3_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Run all read-only diagnostics required before C6 fitting."""
    path = Path(v3_dir)
    artifacts = load_v3_artifacts(path)
    intersections, membership = event_intersections(artifacts["event_hits"])
    event_detail = event_policy_details(
        artifacts["event_hits"],
        artifacts["alarm_states"],
        artifacts["notification_log"],
        artifacts["predictions"],
        artifacts["events"],
        prediction_source_label=str(path / "predictions.parquet"),
    )
    miss_detail, miss_summary = miss_reason_diagnostics(
        artifacts["event_hits"], artifacts["alarm_states"], artifacts["events"]
    )

    contract = artifacts["evaluation_contract"]
    frozen_dir = Path(contract["inputs"]["frozen_external_dir"])
    if not frozen_dir.is_absolute():
        repository = path.parents[2]
        frozen_dir = repository / frozen_dir
    era_path = frozen_dir / "era5_potato_daily.parquet"
    era_daily = pd.read_parquet(era_path) if era_path.is_file() else None
    funnel, funnel_losses = yearly_funnel(
        artifacts["field_seasons"], artifacts["daily_decisions"], artifacts["event_hits"], era_daily=era_daily
    )
    periodic_saved = extract_periodic_30d(artifacts["event_hits"], artifacts["alarm_states"])
    periodic_phases, periodic_phase_summary = periodic_k_all_phases(
        artifacts["daily_decisions"], artifacts["field_seasons"]
    )
    polyakov, polyakov_status = polyakov_computability(
        artifacts["event_hits"],
        artifacts["alarm_states"],
        artifacts["daily_decisions"],
        artifacts["field_seasons"],
    )
    optuna = c4_optuna_diagnostics(
        artifacts["event_hits"],
        artifacts["alarm_states"],
        artifacts["policy_selection"],
        artifacts["optuna_trials"],
        artifacts["optuna_seed_checks"],
    )
    return {
        "v3_event_intersections": intersections,
        "v3_event_pair_membership": membership,
        "v3_event_policy_details": event_detail,
        "v3_miss_reason_details": miss_detail,
        "v3_miss_reason_summary": miss_summary,
        "v3_yearly_funnel": funnel,
        "v3_funnel_loss_reasons": funnel_losses,
        "v3_periodic_30d": periodic_saved,
        "v3_periodic_17d_all_phases": periodic_phases,
        "v3_periodic_17d_phase_summary": periodic_phase_summary,
        "v3_polyakov_computability": polyakov,
        "v3_polyakov_status_counts": polyakov_status,
        **optuna,
    }
