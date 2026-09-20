"""Reporting and paired uncertainty for the second late-blight cycle.

The functions in this module consume already saved event and daily policy
artifacts.  They do not fit models, choose thresholds, or alter notification
state.  Pairing is deliberately strict: a candidate and a comparator must have
the same event and evaluation-day populations before a delta is calculated.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CYCLE2_BASELINES = ("calendar_window", "C0", "C1", "C4", "C5")
CYCLE2_SCOPES = ("paired_candidate_days", "service_calendar")
PRIMARY_YEARS = (2020, 2025)
BOOTSTRAP_SEED = 20260910
BOOTSTRAP_REPETITIONS = 2000

_EVENT_KEYS = ("season", "field_season")
_DAY_KEYS = ("season", "field_season", "issue_date")
_COUNT_COLUMNS = (
    "first_events",
    "opportunities",
    "candidate_hits",
    "baseline_hits",
    "candidate_computable_events",
    "baseline_computable_events",
    "field_days",
    "candidate_messages",
    "baseline_messages",
    "candidate_alarm_days",
    "baseline_alarm_days",
    "candidate_computable_days",
    "baseline_computable_days",
    "candidate_fallback_days",
)
_RATE_COLUMNS = (
    "candidate_timely_recall",
    "baseline_timely_recall",
    "delta_timely_recall",
    "candidate_messages_per_30_field_days",
    "baseline_messages_per_30_field_days",
    "delta_messages_per_30_field_days",
    "candidate_active_alarm_fraction",
    "baseline_active_alarm_fraction",
    "delta_active_alarm_fraction",
    "candidate_computable_event_fraction",
    "baseline_computable_event_fraction",
    "delta_computable_event_fraction",
    "candidate_computable_fraction",
    "baseline_computable_fraction",
    "delta_computable_fraction",
    "candidate_fallback_fraction",
)


def _require(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _bool(values: pd.Series) -> pd.Series:
    """Read native booleans and lossless CSV representations."""
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
        raise ValueError("Invalid boolean value in a saved evaluation artifact")
    return converted.astype("boolean").fillna(False).astype(bool)


def _year(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "season" not in result:
        _require(result, ("fold_id",), "year-bearing artifact")
        result["season"] = result["fold_id"].astype(str).str.extract(
            r"^test_(\d{4})(?:_|$)", expand=False
        )
    result["season"] = pd.to_numeric(result["season"], errors="raise").astype(int)
    return result


def _ratio(numerator: Any, denominator: Any) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) <= 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _event_key(field_season: Any, season: int, namespace: str) -> str:
    payload = f"{namespace}\x1f{int(season)}\x1f{field_season}".encode("utf-8")
    return "evt_" + hashlib.sha256(payload).hexdigest()[:20]


def _day_key(field_season: Any, season: int, issue_date: Any, namespace: str) -> str:
    payload = (
        f"{namespace}\x1f{int(season)}\x1f{field_season}\x1f{pd.Timestamp(issue_date).date()}"
    ).encode("utf-8")
    return "day_" + hashlib.sha256(payload).hexdigest()[:20]


def _combine(primary: pd.DataFrame, reference: pd.DataFrame | None) -> pd.DataFrame:
    if reference is None or reference.empty:
        return primary.copy()
    if primary.empty:
        return reference.copy()
    return pd.concat([primary, reference], ignore_index=True, sort=False)


def discover_weather_c6_models(event_hits: pd.DataFrame) -> tuple[str, ...]:
    """Find main weather C6 variants while excluding both negative controls."""
    _require(event_hits, ("model_code",), "event_hits")
    if "model_family" in event_hits:
        selected = event_hits.loc[
            event_hits["model_family"].astype(str).eq("C6_weather"), "model_code"
        ]
    else:
        code = event_hits["model_code"].astype(str)
        selected = event_hits.loc[
            code.eq("C6_with_C0_fallback")
            | code.eq("C6_weather")
            | code.str.startswith("C6_weather__"),
            "model_code",
        ]
    return tuple(sorted(selected.astype(str).unique()))


def _selected_events(
    event_hits: pd.DataFrame,
    model: str,
    scope: str,
    slice_name: str,
    years: tuple[int, int],
) -> pd.DataFrame:
    required = (*_EVENT_KEYS, "model_code", "evaluation_scope", "slice", "warnable_event", "timely_hit")
    _require(event_hits, required, "event_hits")
    frame = _year(event_hits)
    frame = frame[
        frame["season"].between(*years)
        & frame["model_code"].eq(model)
        & frame["evaluation_scope"].eq(scope)
        & frame["slice"].eq(slice_name)
    ].copy()
    if frame.duplicated(list(_EVENT_KEYS)).any():
        raise ValueError(f"Duplicate event identities for {model}/{scope}/{slice_name}")
    frame["warnable_event"] = _bool(frame["warnable_event"])
    frame["timely_hit"] = _bool(frame["timely_hit"])
    if "computable_in_actionable_window" in frame:
        frame["computable_in_actionable_window"] = _bool(
            frame["computable_in_actionable_window"]
        )
    return frame


def _selected_days(
    alarm_states: pd.DataFrame,
    model: str,
    scope: str,
    years: tuple[int, int],
) -> pd.DataFrame:
    required = (
        *_DAY_KEYS,
        "model_code",
        "evaluation_scope",
        "message_issued",
        "alarm_active",
        "score",
    )
    _require(alarm_states, required, "alarm_states")
    frame = _year(alarm_states)
    day_flag = "evaluation_scope_day" if "evaluation_scope_day" in frame else "service_active"
    _require(frame, (day_flag,), "alarm_states")
    frame = frame[
        frame["season"].between(*years)
        & frame["model_code"].eq(model)
        & frame["evaluation_scope"].eq(scope)
        & _bool(frame[day_flag])
    ].copy()
    frame["issue_date"] = pd.to_datetime(frame["issue_date"])
    if frame.duplicated(list(_DAY_KEYS)).any():
        raise ValueError(f"Duplicate evaluation-day identities for {model}/{scope}")
    for column in ("message_issued", "alarm_active", "suppressed_repeat"):
        if column in frame:
            frame[column] = _bool(frame[column])
    if "fallback_to_c0" in frame:
        frame["fallback_to_c0"] = _bool(frame["fallback_to_c0"])
    return frame


def _paired_event_rows(
    event_hits: pd.DataFrame,
    candidate: str,
    baseline: str,
    scope: str,
    slice_name: str,
    years: tuple[int, int],
) -> pd.DataFrame:
    candidate_rows = _selected_events(event_hits, candidate, scope, slice_name, years)
    baseline_rows = _selected_events(event_hits, baseline, scope, slice_name, years)
    if candidate_rows.empty or baseline_rows.empty:
        raise ValueError(f"Missing event population for {candidate} versus {baseline} in {scope}")
    optional = ["computable_in_actionable_window"]
    candidate_columns = list(_EVENT_KEYS) + ["warnable_event", "timely_hit"] + [
        column for column in optional if column in candidate_rows
    ]
    baseline_columns = list(_EVENT_KEYS) + ["warnable_event", "timely_hit"] + [
        column for column in optional if column in baseline_rows
    ]
    paired = candidate_rows[candidate_columns].merge(
        baseline_rows[baseline_columns],
        on=list(_EVENT_KEYS),
        how="outer",
        suffixes=("_candidate", "_baseline"),
        indicator=True,
        validate="one_to_one",
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError(
            f"Non-matching event population for {candidate} versus {baseline} in {scope}"
        )
    if not paired["warnable_event_candidate"].eq(paired["warnable_event_baseline"]).all():
        raise ValueError("Candidate and baseline disagree on warning opportunities")
    return paired.drop(columns="_merge")


def _paired_day_rows(
    alarm_states: pd.DataFrame,
    candidate: str,
    baseline: str,
    scope: str,
    years: tuple[int, int],
) -> pd.DataFrame:
    candidate_rows = _selected_days(alarm_states, candidate, scope, years)
    baseline_rows = _selected_days(alarm_states, baseline, scope, years)
    if candidate_rows.empty or baseline_rows.empty:
        raise ValueError(f"Missing day population for {candidate} versus {baseline} in {scope}")
    optional = (
        "policy_threshold",
        "policy_active_days",
        "policy_cooldown_days",
        "suppressed_repeat",
        "action_reason",
        "days_to_first_recorded_event",
        "fallback_to_c0",
    )
    core = ("score", "message_issued", "alarm_active")
    candidate_columns = list(_DAY_KEYS) + list(core) + [
        column for column in optional if column in candidate_rows
    ]
    baseline_columns = list(_DAY_KEYS) + list(core) + [
        column for column in optional if column in baseline_rows
    ]
    paired = candidate_rows[candidate_columns].merge(
        baseline_rows[baseline_columns],
        on=list(_DAY_KEYS),
        how="outer",
        suffixes=("_candidate", "_baseline"),
        indicator=True,
        validate="one_to_one",
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError(
            f"Non-matching evaluation-day population for {candidate} versus {baseline} in {scope}"
        )
    if {
        "days_to_first_recorded_event_candidate",
        "days_to_first_recorded_event_baseline",
    }.issubset(paired):
        left = paired["days_to_first_recorded_event_candidate"]
        right = paired["days_to_first_recorded_event_baseline"]
        same = left.eq(right) | (left.isna() & right.isna())
        if not same.all():
            raise ValueError("Candidate and baseline disagree on event-relative dates")
    return paired.drop(columns="_merge")


def paired_annual_counts(
    event_hits: pd.DataFrame,
    alarm_states: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    scope: str,
    slice_name: str = "A_plus_B",
    years: tuple[int, int] = PRIMARY_YEARS,
) -> pd.DataFrame:
    """Return exact paired annual counts and rates for one comparison."""
    events = _paired_event_rows(event_hits, candidate, baseline, scope, slice_name, years)
    days = _paired_day_rows(alarm_states, candidate, baseline, scope, years)
    rows: list[dict[str, Any]] = []
    for season in sorted(events["season"].unique()):
        event_year = events[events["season"].eq(season)]
        day_year = days[days["season"].eq(season)]
        warnable = event_year["warnable_event_candidate"]
        row: dict[str, Any] = {
            "candidate": candidate,
            "baseline": baseline,
            "evaluation_scope": scope,
            "slice": slice_name,
            "season": int(season),
            "first_events": int(len(event_year)),
            "opportunities": int(warnable.sum()),
            "candidate_hits": int((warnable & event_year["timely_hit_candidate"]).sum()),
            "baseline_hits": int((warnable & event_year["timely_hit_baseline"]).sum()),
            "field_days": int(len(day_year)),
            "candidate_messages": int(day_year["message_issued_candidate"].sum()),
            "baseline_messages": int(day_year["message_issued_baseline"].sum()),
            "candidate_alarm_days": int(day_year["alarm_active_candidate"].sum()),
            "baseline_alarm_days": int(day_year["alarm_active_baseline"].sum()),
            "candidate_computable_days": int(day_year["score_candidate"].notna().sum()),
            "baseline_computable_days": int(day_year["score_baseline"].notna().sum()),
        }
        for side in ("candidate", "baseline"):
            column = f"computable_in_actionable_window_{side}"
            row[f"{side}_computable_events"] = (
                int((warnable & _bool(event_year[column])).sum()) if column in event_year else np.nan
            )
        fallback = "fallback_to_c0_candidate"
        row["candidate_fallback_days"] = (
            int(_bool(day_year[fallback]).sum()) if fallback in day_year else np.nan
        )
        row.update(_rates(row))
        rows.append(row)
    return pd.DataFrame(rows)


def _rates(counts: Mapping[str, Any]) -> dict[str, float]:
    opportunities = counts["opportunities"]
    days = counts["field_days"]
    candidate_recall = _ratio(counts["candidate_hits"], opportunities)
    baseline_recall = _ratio(counts["baseline_hits"], opportunities)
    candidate_messages = 30 * _ratio(counts["candidate_messages"], days)
    baseline_messages = 30 * _ratio(counts["baseline_messages"], days)
    candidate_alarm = _ratio(counts["candidate_alarm_days"], days)
    baseline_alarm = _ratio(counts["baseline_alarm_days"], days)
    candidate_event_compute = _ratio(counts["candidate_computable_events"], opportunities)
    baseline_event_compute = _ratio(counts["baseline_computable_events"], opportunities)
    candidate_day_compute = _ratio(counts["candidate_computable_days"], days)
    baseline_day_compute = _ratio(counts["baseline_computable_days"], days)
    return {
        "candidate_timely_recall": candidate_recall,
        "baseline_timely_recall": baseline_recall,
        "delta_timely_recall": candidate_recall - baseline_recall,
        "candidate_messages_per_30_field_days": candidate_messages,
        "baseline_messages_per_30_field_days": baseline_messages,
        "delta_messages_per_30_field_days": candidate_messages - baseline_messages,
        "candidate_active_alarm_fraction": candidate_alarm,
        "baseline_active_alarm_fraction": baseline_alarm,
        "delta_active_alarm_fraction": candidate_alarm - baseline_alarm,
        "candidate_computable_event_fraction": candidate_event_compute,
        "baseline_computable_event_fraction": baseline_event_compute,
        "delta_computable_event_fraction": candidate_event_compute - baseline_event_compute,
        "candidate_computable_fraction": candidate_day_compute,
        "baseline_computable_fraction": baseline_day_compute,
        "delta_computable_fraction": candidate_day_compute - baseline_day_compute,
        "candidate_fallback_fraction": _ratio(counts["candidate_fallback_days"], days),
    }


def _pooled_counts(annual: pd.DataFrame, weights: np.ndarray | None = None) -> dict[str, Any]:
    if weights is None:
        weights = np.ones(len(annual), dtype=int)
    result: dict[str, Any] = {}
    for column in _COUNT_COLUMNS:
        values = annual[column].to_numpy(dtype=float)
        result[column] = float(np.dot(weights, values)) if np.isfinite(values).all() else np.nan
    return result


def paired_year_bootstrap_cycle2(
    event_hits: pd.DataFrame,
    alarm_states: pd.DataFrame,
    *,
    candidates: Sequence[str] | None = None,
    baselines: Sequence[str] = CYCLE2_BASELINES,
    scopes: Sequence[str] = CYCLE2_SCOPES,
    slice_name: str = "A_plus_B",
    years: tuple[int, int] = PRIMARY_YEARS,
    seed: int = BOOTSTRAP_SEED,
    n_bootstrap: int = BOOTSTRAP_REPETITIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate annual paired values and a year-block percentile bootstrap.

    Whole external years are sampled with replacement.  Increasing the number
    of repetitions therefore does not create additional independent years and
    the interval excludes model-selection and registration-time uncertainty.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    candidates = discover_weather_c6_models(event_hits) if candidates is None else tuple(candidates)
    if not candidates:
        raise ValueError("No main weather C6 model was found")
    annual_frames: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for scope in scopes:
        for candidate in candidates:
            for baseline in baselines:
                annual = paired_annual_counts(
                    event_hits,
                    alarm_states,
                    candidate=candidate,
                    baseline=baseline,
                    scope=scope,
                    slice_name=slice_name,
                    years=years,
                )
                annual_frames.append(annual)
                totals = _pooled_counts(annual)
                row: dict[str, Any] = {
                    "candidate": candidate,
                    "baseline": baseline,
                    "evaluation_scope": scope,
                    "slice": slice_name,
                    "year_start": years[0],
                    "year_end": years[1],
                    "test_years": int(len(annual)),
                    "bootstrap_seed": int(seed),
                    "bootstrap_repetitions": int(n_bootstrap),
                    "block": "external_test_year",
                    **totals,
                    **_rates(totals),
                }
                if len(annual) >= 2:
                    weights = rng.multinomial(
                        len(annual), np.full(len(annual), 1 / len(annual)), size=n_bootstrap
                    )
                    samples = {name: [] for name in _RATE_COLUMNS}
                    for sample_weights in weights:
                        sample_rates = _rates(_pooled_counts(annual, sample_weights))
                        for name in samples:
                            samples[name].append(sample_rates[name])
                    for name, values in samples.items():
                        finite = np.asarray(values, dtype=float)
                        finite = finite[np.isfinite(finite)]
                        row[f"{name}_valid_repetitions"] = int(len(finite))
                        if len(finite):
                            low, high = np.quantile(finite, (0.025, 0.975))
                            row[f"{name}_low"] = float(low)
                            row[f"{name}_high"] = float(high)
                        else:
                            row[f"{name}_low"] = np.nan
                            row[f"{name}_high"] = np.nan
                    row["interval_status"] = "few_year_blocks_unadjusted_percentile_interval"
                else:
                    for name in _RATE_COLUMNS:
                        row[f"{name}_valid_repetitions"] = 0
                        row[f"{name}_low"] = np.nan
                        row[f"{name}_high"] = np.nan
                    row["interval_status"] = "not_estimable_fewer_than_two_years"
                bootstrap_rows.append(row)
    annual_result = pd.concat(annual_frames, ignore_index=True) if annual_frames else pd.DataFrame()
    return annual_result, pd.DataFrame(bootstrap_rows)


def leave_one_year_out_cycle2(annual_paired: pd.DataFrame) -> pd.DataFrame:
    """Re-pool exact paired counts after omitting each observed test year."""
    if annual_paired.empty:
        return pd.DataFrame()
    keys = ("candidate", "baseline", "evaluation_scope", "slice")
    _require(annual_paired, (*keys, "season", *_COUNT_COLUMNS), "annual_paired")
    rows: list[dict[str, Any]] = []
    for group_key, group in annual_paired.groupby(list(keys), sort=True, dropna=False):
        for omitted in sorted(group["season"].unique()):
            retained = group[~group["season"].eq(omitted)]
            totals = _pooled_counts(retained)
            rows.append(
                {
                    **dict(zip(keys, group_key)),
                    "omitted_year": int(omitted),
                    "remaining_years": int(len(retained)),
                    **totals,
                    **_rates(totals),
                    "interpretation": "year_influence_diagnostic_not_independent_test",
                }
            )
    return pd.DataFrame(rows)


def format_paired_bootstrap(bootstrap: pd.DataFrame) -> pd.DataFrame:
    """Return a compact Russian display table without changing numeric results."""
    if bootstrap.empty:
        return pd.DataFrame()
    _require(
        bootstrap,
        (
            "candidate",
            "baseline",
            "evaluation_scope",
            "candidate_hits",
            "baseline_hits",
            "opportunities",
            "delta_timely_recall",
            "delta_timely_recall_low",
            "delta_timely_recall_high",
            "candidate_messages_per_30_field_days",
            "baseline_messages_per_30_field_days",
            "candidate_active_alarm_fraction",
            "baseline_active_alarm_fraction",
        ),
        "paired bootstrap",
    )
    rows = []
    for row in bootstrap.itertuples(index=False):
        rows.append(
            {
                "Сценарий": "общая маска v3"
                if row.evaluation_scope == "paired_candidate_days"
                else "полный сервис с fallback",
                "C6": row.candidate,
                "Сравнение": row.baseline,
                "Попадания C6 / база": f"{int(row.candidate_hits)}/{int(row.opportunities)} / {int(row.baseline_hits)}/{int(row.opportunities)}",
                "Δ recall, п.п. [95%]": (
                    f"{100 * row.delta_timely_recall:+.1f}; 95%: "
                    f"[{100 * row.delta_timely_recall_low:+.1f}; {100 * row.delta_timely_recall_high:+.1f}]"
                ),
                "Сообщения / 30, C6 / база": (
                    f"{row.candidate_messages_per_30_field_days:.3f} / "
                    f"{row.baseline_messages_per_30_field_days:.3f}"
                ),
                "Тревожные дни, C6 / база": (
                    f"{100 * row.candidate_active_alarm_fraction:.1f}% / "
                    f"{100 * row.baseline_active_alarm_fraction:.1f}%"
                ),
                "Вычислимые дни C6": (
                    "—"
                    if not hasattr(row, "candidate_computable_fraction")
                    or pd.isna(row.candidate_computable_fraction)
                    else f"{100 * row.candidate_computable_fraction:.1f}%"
                ),
                "Fallback-дни C6": (
                    "—"
                    if not hasattr(row, "candidate_fallback_fraction")
                    or pd.isna(row.candidate_fallback_fraction)
                    else f"{100 * row.candidate_fallback_fraction:.1f}%"
                ),
            }
        )
    return pd.DataFrame(rows)


def classify_c6_vs_c0(
    event_hits: pd.DataFrame,
    alarm_states: pd.DataFrame,
    *,
    candidate: str,
    scope: str,
    slice_name: str = "A_plus_B",
    years: tuple[int, int] = PRIMARY_YEARS,
    baseline: str = "C0",
    score_atol: float = 1e-12,
    key_namespace: str = "late_blight_cycle2_c6_vs_c0",
) -> dict[str, pd.DataFrame]:
    """Classify gained/lost events and how score changes reached the policy.

    Detailed outputs contain only one-way event/day keys.  The interpretation
    is operational and post hoc; it is not a causal explanation of biology.
    """
    events = _paired_event_rows(event_hits, candidate, baseline, scope, slice_name, years)
    days = _paired_day_rows(alarm_states, candidate, baseline, scope, years)
    candidate_hit = events["timely_hit_candidate"] & events["warnable_event_candidate"]
    baseline_hit = events["timely_hit_baseline"] & events["warnable_event_candidate"]
    events["transition"] = np.select(
        (
            candidate_hit & baseline_hit,
            candidate_hit & ~baseline_hit,
            ~candidate_hit & baseline_hit,
        ),
        ("both_hit", "gained_by_c6", "lost_by_c6"),
        default="neither_hit",
    )

    candidate_score = pd.to_numeric(days["score_candidate"], errors="coerce")
    baseline_score = pd.to_numeric(days["score_baseline"], errors="coerce")
    both_missing = candidate_score.isna() & baseline_score.isna()
    both_present_equal = candidate_score.notna() & baseline_score.notna() & np.isclose(
        candidate_score, baseline_score, atol=score_atol, rtol=0
    )
    days["score_changed"] = ~(both_missing | both_present_equal)
    days["message_changed"] = days["message_issued_candidate"].ne(
        days["message_issued_baseline"]
    )
    days["message_transition"] = np.select(
        (
            days["message_issued_candidate"] & days["message_issued_baseline"],
            days["message_issued_candidate"] & ~days["message_issued_baseline"],
            ~days["message_issued_candidate"] & days["message_issued_baseline"],
        ),
        ("both_message", "candidate_only_message", "baseline_only_message"),
        default="neither_message",
    )

    thresholds_available = {
        "policy_threshold_candidate",
        "policy_threshold_baseline",
    }.issubset(days)
    if thresholds_available:
        candidate_threshold = pd.to_numeric(days["policy_threshold_candidate"], errors="coerce")
        baseline_threshold = pd.to_numeric(days["policy_threshold_baseline"], errors="coerce")
        days["threshold_changed"] = ~np.isclose(
            candidate_threshold, baseline_threshold, atol=score_atol, rtol=0, equal_nan=True
        )
        days["candidate_crossed_threshold"] = candidate_score.ge(candidate_threshold)
        days["baseline_crossed_threshold"] = baseline_score.ge(baseline_threshold)
    else:
        days["threshold_changed"] = pd.Series(pd.NA, index=days.index, dtype="boolean")
        days["candidate_crossed_threshold"] = pd.Series(pd.NA, index=days.index, dtype="boolean")
        days["baseline_crossed_threshold"] = pd.Series(pd.NA, index=days.index, dtype="boolean")

    candidate_suppressed = (
        days["suppressed_repeat_candidate"]
        if "suppressed_repeat_candidate" in days
        else pd.Series(False, index=days.index)
    )
    baseline_suppressed = (
        days["suppressed_repeat_baseline"]
        if "suppressed_repeat_baseline" in days
        else pd.Series(False, index=days.index)
    )
    factual_suppression = _bool(candidate_suppressed) | _bool(baseline_suppressed)
    no_message_change = ~days["message_changed"]
    neither_crossed = (
        ~(days["candidate_crossed_threshold"].fillna(False))
        & ~(days["baseline_crossed_threshold"].fillna(False))
    )
    either_crossed = ~neither_crossed
    days["score_policy_effect"] = np.select(
        (
            ~days["score_changed"] & days["message_changed"],
            days["score_changed"] & days["message_changed"],
            days["score_changed"] & no_message_change & neither_crossed,
            days["score_changed"]
            & no_message_change
            & either_crossed
            & factual_suppression,
            days["score_changed"] & no_message_change,
        ),
        (
            "message_changed_without_score_change",
            "score_change_altered_message",
            "score_change_hidden_below_threshold",
            # Keep the historical value/column name for artifact compatibility.
            # The category now requires an actual ``suppressed_repeat`` flag;
            # ``alarm_active`` alone is an outcome and never blocks a message.
            "score_change_hidden_by_cooldown_or_active_alarm",
            "score_change_without_message_change_other",
        ),
        default="no_score_or_message_change",
    )

    # Link event-relative days before source identifiers are discarded.
    lead_column = (
        "days_to_first_recorded_event_candidate"
        if "days_to_first_recorded_event_candidate" in days
        else None
    )
    detail_rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        relevant = days[
            days["season"].eq(event.season) & days["field_season"].eq(event.field_season)
        ]
        actionable = relevant[
            pd.to_numeric(relevant[lead_column], errors="coerce").between(3, 10)
        ] if lead_column else relevant.iloc[0:0]
        changed_messages = actionable["message_changed"]
        detail_rows.append(
            {
                "event_key": _event_key(event.field_season, event.season, key_namespace),
                "season": int(event.season),
                "candidate": candidate,
                "baseline": baseline,
                "evaluation_scope": scope,
                "transition": event.transition,
                "actionable_matched_days": int(len(actionable)),
                "candidate_timely_message_days": int(actionable["message_issued_candidate"].sum()),
                "baseline_timely_message_days": int(actionable["message_issued_baseline"].sum()),
                "score_changed_days_in_window": int(actionable["score_changed"].sum()),
                "message_changed_days_in_window": int(changed_messages.sum()),
                "score_change_altered_message_days": int(
                    actionable["score_policy_effect"].eq("score_change_altered_message").sum()
                ),
                "score_change_hidden_below_threshold_days": int(
                    actionable["score_policy_effect"].eq("score_change_hidden_below_threshold").sum()
                ),
                "score_change_hidden_by_cooldown_or_active_alarm_days": int(
                    actionable["score_policy_effect"]
                    .eq("score_change_hidden_by_cooldown_or_active_alarm")
                    .sum()
                ),
                "candidate_cooldown_suppressed_days": int(
                    actionable.get("suppressed_repeat_candidate", pd.Series(False, index=actionable.index)).sum()
                ),
                "baseline_cooldown_suppressed_days": int(
                    actionable.get("suppressed_repeat_baseline", pd.Series(False, index=actionable.index)).sum()
                ),
                "interpretation": "post_hoc_operational_path_not_biological_cause",
            }
        )
    event_detail = pd.DataFrame(detail_rows)

    event_summaries: list[dict[str, Any]] = []
    for aggregation, season, group in [
        ("pooled", pd.NA, event_detail),
        *[("year", int(year), value) for year, value in event_detail.groupby("season", sort=True)],
    ]:
        counts = group["transition"].value_counts()
        event_summaries.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "evaluation_scope": scope,
                "aggregation": aggregation,
                "season": season,
                "events": int(len(group)),
                "both_hit": int(counts.get("both_hit", 0)),
                "gained_by_c6": int(counts.get("gained_by_c6", 0)),
                "lost_by_c6": int(counts.get("lost_by_c6", 0)),
                "neither_hit": int(counts.get("neither_hit", 0)),
                "net_gain": int(counts.get("gained_by_c6", 0) - counts.get("lost_by_c6", 0)),
                "score_change_altered_message_days_in_event_windows": int(
                    group["score_change_altered_message_days"].sum()
                ),
                "score_change_hidden_below_threshold_days_in_event_windows": int(
                    group["score_change_hidden_below_threshold_days"].sum()
                ),
                "score_change_hidden_by_cooldown_or_active_alarm_days_in_event_windows": int(
                    group["score_change_hidden_by_cooldown_or_active_alarm_days"].sum()
                ),
            }
        )

    day_summaries: list[dict[str, Any]] = []
    effects = (
        "score_change_altered_message",
        "score_change_hidden_by_cooldown_or_active_alarm",
        "score_change_hidden_below_threshold",
        "score_change_without_message_change_other",
        "message_changed_without_score_change",
        "no_score_or_message_change",
    )
    for aggregation, season, group in [
        ("pooled", pd.NA, days),
        *[("year", int(year), value) for year, value in days.groupby("season", sort=True)],
    ]:
        counts = group["score_policy_effect"].value_counts()
        day_summaries.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "evaluation_scope": scope,
                "aggregation": aggregation,
                "season": season,
                "matched_days": int(len(group)),
                "score_changed_days": int(group["score_changed"].sum()),
                "message_changed_days": int(group["message_changed"].sum()),
                **{f"{effect}_days": int(counts.get(effect, 0)) for effect in effects},
            }
        )

    safe_days = days.copy()
    safe_days["day_key"] = [
        _day_key(field, season, date, key_namespace)
        for field, season, date in zip(
            safe_days["field_season"], safe_days["season"], safe_days["issue_date"]
        )
    ]
    safe_days = safe_days.drop(columns=["field_season"])
    safe_columns = [
        "day_key",
        "season",
        "issue_date",
        "score_candidate",
        "score_baseline",
        "message_issued_candidate",
        "message_issued_baseline",
        "alarm_active_candidate",
        "alarm_active_baseline",
        "score_changed",
        "message_changed",
        "message_transition",
        "threshold_changed",
        "candidate_crossed_threshold",
        "baseline_crossed_threshold",
        "score_policy_effect",
    ]
    return {
        "gained_lost_events": event_detail,
        "gained_lost_summary": pd.DataFrame(event_summaries),
        "message_change_days": safe_days[[column for column in safe_columns if column in safe_days]],
        "message_change_summary": pd.DataFrame(day_summaries),
    }


def build_cycle2_reporting_artifacts(
    event_hits: pd.DataFrame,
    alarm_states: pd.DataFrame,
    *,
    reference_event_hits: pd.DataFrame | None = None,
    reference_alarm_states: pd.DataFrame | None = None,
    candidates: Sequence[str] | None = None,
    baselines: Sequence[str] = CYCLE2_BASELINES,
    scopes: Sequence[str] = CYCLE2_SCOPES,
    slice_name: str = "A_plus_B",
    years: tuple[int, int] = PRIMARY_YEARS,
    seed: int = BOOTSTRAP_SEED,
    n_bootstrap: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, pd.DataFrame]:
    """Build all cycle-2 comparison artifacts from saved policy outputs."""
    combined_events = _combine(event_hits, reference_event_hits)
    combined_states = _combine(alarm_states, reference_alarm_states)
    selected_candidates = (
        discover_weather_c6_models(combined_events) if candidates is None else tuple(candidates)
    )
    annual, bootstrap = paired_year_bootstrap_cycle2(
        combined_events,
        combined_states,
        candidates=selected_candidates,
        baselines=baselines,
        scopes=scopes,
        slice_name=slice_name,
        years=years,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    result: dict[str, pd.DataFrame] = {
        "paired_annual_metrics": annual,
        "paired_year_bootstrap": bootstrap,
        "leave_one_year_out": leave_one_year_out_cycle2(annual),
    }
    transition_frames: dict[str, list[pd.DataFrame]] = {
        "gained_lost_events": [],
        "gained_lost_summary": [],
        "message_change_days": [],
        "message_change_summary": [],
    }
    if "C0" in baselines:
        for scope in scopes:
            for candidate in selected_candidates:
                classified = classify_c6_vs_c0(
                    combined_events,
                    combined_states,
                    candidate=candidate,
                    scope=scope,
                    slice_name=slice_name,
                    years=years,
                )
                for name, frame in classified.items():
                    transition_frames[name].append(frame)
    result.update(
        {
            name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            for name, frames in transition_frames.items()
        }
    )
    return result


def _fmt(value: Any, digits: int = 3, percent: bool = False) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{100 * float(value):.{digits}f}%" if percent else f"{float(value):.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: Sequence[tuple[str, str]], limit: int = 40) -> list[str]:
    available = [(column, label) for column, label in columns if column in frame]
    if frame.empty or not available:
        return ["Данные для этой таблицы не переданы в отчёт."]
    headers = [label.replace("|", "\\|").replace("\n", " ") for _, label in available]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, mapping in frame.head(limit).iterrows():
        cells = []
        for column, _ in available:
            value = mapping[column]
            if value is None or pd.isna(value):
                cell = "—"
            elif isinstance(value, (float, np.floating)):
                numeric = float(value)
                cell = str(int(numeric)) if np.isfinite(numeric) and numeric.is_integer() else f"{numeric:.3f}"
            else:
                cell = str(value)
            cells.append(cell.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    if len(frame) > limit:
        lines.append("")
        lines.append(f"Показаны первые {limit} из {len(frame)} агрегированных строк.")
    return lines


def _table(artifacts: Mapping[str, Any], *names: str) -> pd.DataFrame:
    first_empty: pd.DataFrame | None = None
    for name in names:
        value = artifacts.get(name)
        if isinstance(value, pd.DataFrame):
            if not value.empty:
                return value
            if first_empty is None:
                first_empty = value
    return pd.DataFrame() if first_empty is None else first_empty


def _pooled(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "aggregation" not in frame:
        return frame
    return frame[frame["aggregation"].eq("pooled")].copy()


def _derived_question_answers(
    artifacts: Mapping[str, Any], metadata: Mapping[str, Any]
) -> list[str]:
    explicit = metadata.get("question_answers")
    if isinstance(explicit, Mapping):
        keys = ("different_hits", "c6_addition", "weather_specificity", "next_experiment")
        return [str(explicit.get(key, "Ответ не зафиксирован.")) for key in keys]
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)):
        values = [str(value) for value in explicit]
        if len(values) == 4:
            return values

    intersections = _pooled(_table(artifacts, "v3_event_intersections"))
    main_intersection = intersections
    if not main_intersection.empty and "pair_id" in main_intersection:
        main_intersection = main_intersection[
            main_intersection["pair_id"].eq("calendar_window__vs__C4")
        ]
    if not main_intersection.empty:
        row = main_intersection.iloc[0]
        q1 = (
            f"Да. На общей маске v3 из {int(row['events'])} первых событий "
            f"{int(row['both_hit'])} поймали оба метода, "
            f"{int(row['baseline_only_hit'])} — только календарь, "
            f"{int(row['candidate_only_hit'])} — только C4, "
            f"{int(row['neither_hit'])} — никто. Постфактум объединение даёт "
            f"{int(row['oracle_union_hits'])}/{int(row['events'])}, но это не "
            "готовая политика при заданном бюджете сообщений."
        )
    else:
        q1 = "Таблица пересечений не передана; ответ по сохранённым политикам не вычислен."

    def pooled_pairs(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column, value in (("period", "2020_2025"), ("slice", "A_plus_B")):
            if column in result:
                result = result[result[column].eq(value)]
        if "aggregation" in result:
            result = result[result["aggregation"].eq("pooled")]
        return result

    def paired_row(
        frame: pd.DataFrame, candidate: str, baseline: str, scope: str
    ) -> pd.Series | None:
        if frame.empty:
            return None
        rows = frame[
            frame["candidate"].eq(candidate)
            & frame["baseline"].eq(baseline)
            & frame["evaluation_scope"].eq(scope)
        ]
        if len(rows) > 1:
            raise ValueError(
                f"Duplicate pooled comparison for {candidate} versus {baseline} in {scope}"
            )
        return None if rows.empty else rows.iloc[0]

    main_pairs = pooled_pairs(
        _table(artifacts, "paired_year_bootstrap", "cycle2_paired_bootstrap")
    )
    if main_pairs.empty:
        q2 = "Парный результат C6 не передан; добавка к конкурентам не оценена."
    else:
        baseline_labels = (
            ("calendar_window", "календарь"),
            ("C0", "C0"),
            ("C1", "C1"),
            ("C4", "C4"),
            ("C5", "C5"),
        )
        replay_code = "C6_weather__c0_policy_replay"
        selected_code = "C6_weather__validation_selected"
        replay_rows = [
            (
                label,
                paired_row(
                    main_pairs, replay_code, baseline, "paired_candidate_days"
                ),
            )
            for baseline, label in baseline_labels
        ]
        replay_rows = [(label, row) for label, row in replay_rows if row is not None]
        selected_c0 = paired_row(
            main_pairs, selected_code, "C0", "paired_candidate_days"
        )
        service_c0 = paired_row(main_pairs, replay_code, "C0", "service_calendar")
        if replay_rows:
            reference = replay_rows[0][1]
            assert reference is not None
            competitors = ", ".join(
                f"{label} {int(row['baseline_hits'])}"
                for label, row in replay_rows
                if row is not None
            )
            c0_replay = paired_row(
                main_pairs, replay_code, "C0", "paired_candidate_days"
            )
            interval = ""
            interval_columns = {
                "delta_timely_recall",
                "delta_timely_recall_low",
                "delta_timely_recall_high",
            }
            if c0_replay is not None and interval_columns.issubset(c0_replay.index):
                interval = (
                    f" Парная разница с C0: "
                    f"{100 * float(c0_replay['delta_timely_recall']):+.1f} п.п., "
                    f"интервал {100 * float(c0_replay['delta_timely_recall_low']):+.1f}…"
                    f"{100 * float(c0_replay['delta_timely_recall_high']):+.1f} п.п."
                )
            selected_text = ""
            if selected_c0 is not None:
                selected_text = (
                    f" С выбранной на временной валидации политикой C6 поймала "
                    f"{int(selected_c0['candidate_hits'])}/{int(selected_c0['opportunities'])}."
                )
            service_text = ""
            if service_c0 is not None:
                service_text = (
                    f" В полном сервисе replay даёт C6 "
                    f"{int(service_c0['candidate_hits'])} и C0 "
                    f"{int(service_c0['baseline_hits'])} попаданий."
                )
            q2 = (
                f"При неизменной политике C0 погодная C6 поймала "
                f"{int(reference['candidate_hits'])}/{int(reference['opportunities'])}; "
                f"конкуренты: {competitors}.{selected_text}{service_text}{interval} "
                "Сильное преимущество над набором заранее заданных конкурентов не установлено."
            )
        else:
            q2 = "Сопоставимый pooled-результат погодной C6 не найден."

    # The strict paired-year table intentionally contains the five main
    # competitors only.  Weather-specific negative controls are produced by the
    # broader pipeline comparison and must be read from ``paired_bootstrap``.
    control_pairs = pooled_pairs(_table(artifacts, "paired_bootstrap"))

    def control_text(mode: str, scope: str) -> str:
        candidate = f"C6_weather__{mode}"
        calibration = paired_row(
            control_pairs, candidate, f"C6_calibration_control__{mode}", scope
        )
        calendar = paired_row(
            control_pairs, candidate, f"C6_calendar_control__{mode}", scope
        )
        if calibration is None or calendar is None:
            return "нет полной пары контролей"
        return (
            f"погода {int(calibration['candidate_hits'])}, "
            f"калибровка {int(calibration['baseline_hits'])}, "
            f"календарный CatBoost {int(calendar['baseline_hits'])}"
        )

    if control_pairs.empty:
        q3 = (
            "Сопоставимые результаты погодной C6 и контролей не переданы; "
            "специфичность погодной добавки не установлена."
        )
    else:
        q3 = (
            "При replay на общей маске: "
            f"{control_text('c0_policy_replay', 'paired_candidate_days')}; "
            "в полном сервисе: "
            f"{control_text('c0_policy_replay', 'service_calendar')}. "
            "Для политики, выбранной на временной валидации, на общей маске: "
            f"{control_text('validation_selected', 'paired_candidate_days')}; "
            "в полном сервисе: "
            f"{control_text('validation_selected', 'service_calendar')}. "
            "Разница на ограниченной погодной маске сама по себе не доказывает "
            "погодную специфичность, особенно если она не сохраняется в сервисном сценарии."
        )
    q4 = str(
        metadata.get(
            "next_priority",
            "Проверить наблюдаемость и временную доступность погодных входов до расширения поиска моделей.",
        )
    )
    return [q1, q2, q3, q4]


def _privacy_values(artifacts: Mapping[str, Any]) -> set[str]:
    forbidden = {
        "field_season",
        "field_uid",
        "latitude",
        "longitude",
        "final_latitude",
        "final_longitude",
        "coordinates",
        "weather_cell",
    }
    values: set[str] = set()
    for value in artifacts.values():
        if not isinstance(value, pd.DataFrame):
            continue
        for column in value.columns:
            if str(column).lower() in forbidden:
                for item in value[column].dropna().unique():
                    rendered = str(item)
                    if len(rendered) >= 4:
                        values.add(rendered)
    return values


def _render_cycle2_report_ru(
    path: str | Path,
    artifacts: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write an aggregate Russian report from already saved cycle-2 tables.

    Unknown or row-level tables are never dumped wholesale.  Only named
    aggregate columns below can reach the report.  Values from source identifier
    and coordinate columns are checked after rendering as an additional guard.
    """
    metadata = {} if metadata is None else metadata
    answers = _derived_question_answers(artifacts, metadata)
    priority = str(metadata.get("next_priority", answers[3])).strip()
    if not priority:
        raise ValueError("One non-empty next priority is required")
    run_id = str(metadata.get("run_id", "cycle2"))
    lines: list[str] = [
        "# Второй исследовательский цикл ранних предупреждений о фитофторозе картофеля",
        "",
        f"Запуск: `{run_id}`.",
        "",
        "## Четыре ответа",
        "",
        f"1. **Различаются ли своевременные попадания календаря и погоды?** {answers[0]}",
        "",
        f"2. **Добавляет ли C6 сигнал к C0 и сильным календарным конкурентам?** {answers[1]}",
        "",
        f"3. **Связана ли добавка именно с погодой?** {answers[2]}",
        "",
        f"4. **Какой следующий эксперимент оправдан?** {answers[3]}",
        "",
        "## Зафиксированная задача и устройство C6",
        "",
        "Цель остаётся неизменной: сообщение за 3–10 календарных дней до первого зарегистрированного случая в поле-сезоне. Повторные положительные визиты не считаются новыми событиями, а отсутствие визита или записи не считается подтверждённым отсутствием болезни.",
        "",
        "C6 использует многоклассовые raw logits календарного C0 как baseline и добавляет к ним только масштабированную CatBoost-поправку. Alpha выбирается из фиксированной сетки 0, 0.1, 0.25, 0.5, 1 внутри временной валидации. Внешние годы не используются для выбора alpha, порога или политики.",
        "",
        "В отчёте раздельно показаны два сценария. «общая маска v3» изолирует сравнение модели на прежних погодных датах. «полный сервис с fallback» сохраняет календарную оценку C0 при недоступной погоде и единую непрерывную историю cooldown и тревоги. Эти знаменатели не смешиваются.",
        "",
        "## Пересечения попаданий сохранённых политик v3",
        "",
    ]
    intersections = _pooled(_table(artifacts, "v3_event_intersections"))
    lines.extend(
        _markdown_table(
            intersections,
            (
                ("pair_id", "Пара"),
                ("events", "События"),
                ("both_hit", "Оба"),
                ("baseline_only_hit", "Только база"),
                ("candidate_only_hit", "Только кандидат"),
                ("neither_hit", "Никто"),
                ("oracle_union_hits", "Post-hoc union"),
            ),
        )
    )
    intersection_years = _table(artifacts, "v3_event_intersections")
    if not intersection_years.empty and "aggregation" in intersection_years:
        intersection_years = intersection_years[
            intersection_years["aggregation"].eq("year")
        ]
    lines.extend(["", "### Те же пересечения по внешним годам", ""])
    lines.extend(
        _markdown_table(
            intersection_years,
            (
                ("season", "Год"),
                ("pair_id", "Пара"),
                ("events", "События"),
                ("both_hit", "Оба"),
                ("baseline_only_hit", "Только база"),
                ("candidate_only_hit", "Только кандидат"),
                ("neither_hit", "Никто"),
                ("oracle_union_hits", "Post-hoc union"),
            ),
            limit=max(30, len(intersection_years)),
        )
    )
    lines.extend(
        [
            "",
            "`oracle_union_of_fixed_policies` — постфактум объединение двух уже рассчитанных политик. Это не достижимая модель при том же бюджете и не информационный потолок задачи.",
            "",
            "### Диагностика промахов",
            "",
        ]
    )
    misses = _pooled(_table(artifacts, "v3_miss_reason_summary"))
    lines.extend(
        _markdown_table(
            misses,
            (
                ("pair_id", "Пара"),
                ("overlap_category", "Группа"),
                ("missed_model", "Пропустил"),
                ("missed_events", "Промахи"),
                ("reason_no_actionable_date", "Нет даты решения"),
                ("reason_weather_unavailable_all", "Нет погоды"),
                ("reason_score_below_threshold", "Ниже порога"),
                ("reason_cooldown_suppression", "Cooldown"),
                ("reason_only_early_messages", "Только рано"),
                ("reason_only_late_messages", "Только поздно"),
            ),
        )
    )

    lines.extend(["", "## Воронка данных и вычислимость", ""])
    funnel = _table(artifacts, "v3_yearly_funnel")
    lines.extend(
        _markdown_table(
            funnel,
            (
                ("aggregation", "Уровень"),
                ("season", "Год"),
                ("field_seasons_total", "Поле-сезоны"),
                ("all_first_events", "Все первые события"),
                ("events_after_connection", "После подключения"),
                ("events_with_actionable_time", "Есть окно 3–10"),
                ("main_registry_events", "Основной анализ"),
                ("events_with_any_common_weather_in_window", "Есть общая погода"),
                ("service_field_days", "Сервисные дни"),
                ("common_candidate_days", "Дни общей маски"),
                ("calendar_window_timely_hits", "Попадания календаря"),
                ("C0_timely_hits", "Попадания C0"),
                ("C1_timely_hits", "Попадания C1"),
                ("C4_timely_hits", "Попадания C4"),
                ("C5_timely_hits", "Попадания C5"),
            ),
        )
    )
    lines.extend(["", "### Причины потери строк в воронке", ""])
    funnel_losses = _pooled(_table(artifacts, "v3_funnel_loss_reasons"))
    lines.extend(
        _markdown_table(
            funnel_losses,
            (
                ("loss_reason", "Причина"),
                ("count", "Количество"),
            ),
        )
    )
    lines.extend(["", "### Поляков: сервис и условно вычислимые случаи", ""])
    polyakov = _table(artifacts, "v3_polyakov_computability")
    lines.extend(
        _markdown_table(
            polyakov,
            (
                ("aggregation", "Уровень"),
                ("season", "Год"),
                ("evaluation_scope", "Сценарий"),
                ("events_with_warning_opportunity", "События"),
                ("timely_hits", "Попадания"),
                ("computable_events", "Вычислимые события"),
                ("computable_event_fraction", "Доля вычислимых событий"),
                ("conditional_timely_recall", "Recall среди вычислимых"),
                ("computable_fraction", "Доля вычислимых дней"),
                ("bbch_causally_available_events", "BBCH доступна в окне"),
                ("common_weather_events", "Погода доступна в окне"),
                ("positive_score_events", "Положительный score в окне"),
            ),
        )
    )
    lines.extend(["", "Статусы вычислимости Полякова на сервисных днях:", ""])
    polyakov_status = _pooled(_table(artifacts, "v3_polyakov_status_counts"))
    lines.extend(
        _markdown_table(
            polyakov_status,
            (
                ("polyakov_status", "Статус"),
                ("field_days", "Поле-дни"),
                (
                    "field_seasons_with_observed_bbch51",
                    "Поле-сезоны с наблюдённой BBCH51",
                ),
            ),
        )
    )
    lines.extend(
        [
            "",
            "Для Полякова вычислимость требует доступной к дате выпуска BBCH51 и вычислимого погодного правила; в общей маске дополнительно действует общая погодная маска v3. Неизвестная BBCH не трактуется как низкий риск.",
            "",
            "## Периодические контроли",
            "",
            "Сохранённый `periodic_30d` — сообщения 1 июня, 1 июля и 1 августа при активном поле, а не точный шаг в 30 дней. Контроль k=17 приведён по всем заранее заданным фазам; лучшая внешняя фаза не выбирается.",
            "",
        ]
    )
    periodic = _table(artifacts, "v3_periodic_30d")
    lines.extend(
        _markdown_table(
            periodic,
            (
                ("aggregation", "Уровень"),
                ("season", "Год"),
                ("evaluation_scope", "Сценарий"),
                ("timely_hits", "Попадания"),
                ("events_with_warning_opportunity", "События"),
                ("messages_per_30_field_days", "Сообщения / 30"),
                ("active_alarm_fraction", "Тревожные дни"),
                ("computable_fraction", "Вычислимость"),
            ),
        )
    )
    lines.extend(["", "### k=17, все фазы", ""])
    phase_summary = _table(artifacts, "v3_periodic_17d_phase_summary")
    lines.extend(
        _markdown_table(
            phase_summary,
            (
                ("aggregation", "Уровень"),
                ("season", "Год"),
                ("evaluation_scope", "Сценарий"),
                ("phases", "Фазы"),
                ("timely_hits_min", "Попадания min"),
                ("timely_hits_median", "Попадания median"),
                ("timely_hits_max", "Попадания max"),
                ("messages_per_30_min", "Сообщения min"),
                ("messages_per_30_median", "Сообщения median"),
                ("messages_per_30_max", "Сообщения max"),
            ),
        )
    )

    lines.extend(["", "## OOF, alpha и политики", ""])
    oof = _table(artifacts, "oof_provenance", "oof_fold_manifest")
    lines.extend(
        _markdown_table(
            oof,
            (
                ("outer_fold_id", "Внешний fold"),
                ("inner_forecast_year", "OOF-год"),
                ("forecast_year", "OOF-год"),
                ("baseline_fit_year_end", "Конец обучения C0"),
                ("baseline_fit_end", "Последняя дата C0"),
                ("max_label_available_at", "Последняя доступная метка"),
                ("forecast_rows", "OOF-строки"),
                ("status", "Статус"),
            ),
            limit=30,
        )
    )
    lines.extend(["", "### Размеры обучения основы, поправки и validation", ""])
    training = _table(artifacts, "correction_training_summary")
    lines.extend(
        _markdown_table(
            training,
            (
                ("fold_id", "Fold"),
                ("model_code", "Поправка"),
                ("outer_c0_refit_rows", "Строки outer C0"),
                ("outer_c0_refit_field_seasons", "Поле-сезоны outer C0"),
                ("rows", "OOF-строки поправки"),
                ("field_seasons", "OOF поле-сезоны"),
                ("warnable_events", "OOF-события"),
                ("validation_rows", "Validation-строки"),
                ("validation_service_days", "Validation сервисные дни"),
                ("validation_paired_days", "Validation дни общей маски"),
                ("validation_field_seasons", "Validation поле-сезоны"),
                ("validation_warnable_events", "Validation-события"),
            ),
            limit=30,
        )
    )
    distributions = _table(artifacts, "logit_distributions")
    compact_distributions = pd.DataFrame()
    if not distributions.empty:
        compact_distributions = (
            distributions.groupby(
                ["model_code", "split", "component", "class_id"],
                sort=True,
                dropna=False,
            )
            .agg(
                folds=("fold_id", "nunique"),
                median_fold_mean=("mean", "median"),
                median_fold_std=("std", "median"),
                minimum_q05=("q05", "min"),
                maximum_q95=("q95", "max"),
            )
            .reset_index()
        )
    lines.extend(["", "### Сдвиг raw logits: temporal OOF и outer refit", ""])
    lines.extend(
        _markdown_table(
            compact_distributions,
            (
                ("model_code", "Модель"),
                ("split", "Блок"),
                ("component", "Компонента"),
                ("class_id", "Класс"),
                ("folds", "Folds"),
                ("median_fold_mean", "Медиана средних"),
                ("median_fold_std", "Медиана SD"),
                ("minimum_q05", "Min q05"),
                ("maximum_q95", "Max q95"),
            ),
            limit=60,
        )
    )
    lines.extend(["", "### Выбор alpha и политики на внутреннем прошлом", ""])
    policies = _table(artifacts, "alpha_policy_selection", "policy_selection", "c6_policy_selection")
    lines.extend(
        _markdown_table(
            policies,
            (
                ("fold_id", "Fold"),
                ("evaluation_scope", "Сценарий"),
                ("model_code", "Модель"),
                ("policy_mode", "Режим политики"),
                ("alpha", "Alpha"),
                ("selected_alpha", "Alpha"),
                ("threshold", "Порог"),
                ("active_days", "Дней тревоги"),
                ("cooldown_days", "Cooldown"),
                ("validation_timely_recall", "Validation recall"),
                ("validation_messages_per_30", "Validation сообщений / 30"),
                ("validation_alarm_fraction", "Validation тревожных дней"),
                ("feasible", "Бюджет соблюдён"),
                ("constraint_violation", "Превышение бюджета"),
            ),
            limit=max(120, len(policies)),
        )
    )
    verification_rows: list[dict[str, Any]] = []
    alpha0_checks = _table(artifacts, "alpha0_identity_checks")
    if not alpha0_checks.empty:
        score_difference = pd.to_numeric(
            alpha0_checks.get(
                "max_abs_c0_score_difference", pd.Series(dtype=float)
            ),
            errors="coerce",
        )
        threshold_reselected = (
            int(_bool(alpha0_checks["threshold_reselected"]).sum())
            if "threshold_reselected" in alpha0_checks
            else pd.NA
        )
        verification_rows.append(
            {
                "check": "alpha=0: C0 score, сообщения, тревога и метрики",
                "checks": int(len(alpha0_checks)),
                "passed": int(alpha0_checks["status"].astype(str).eq("passed").sum()),
                "max_abs_difference": score_difference.max(),
                "sample_rows": pd.NA,
                "fallback_rows_checked": pd.NA,
                "fallback_mask_checks_passed": pd.NA,
                "policy_invariant": (
                    f"перенастроено порогов: {threshold_reselected}"
                    if threshold_reselected is not pd.NA
                    else "нет поля threshold_reselected"
                ),
            }
        )
    reload_checks = _table(artifacts, "model_reload_verification")
    if not reload_checks.empty:
        difference_columns = [
            column
            for column in (
                "max_abs_base_logits_difference",
                "max_abs_correction_logits_difference",
                "max_abs_probabilities_difference",
            )
            if column in reload_checks
        ]
        maximum_difference = (
            pd.concat(
                [
                    pd.to_numeric(reload_checks[column], errors="coerce")
                    for column in difference_columns
                ],
                ignore_index=True,
            ).max()
            if difference_columns
            else np.nan
        )
        policy_equal = (
            int(_bool(reload_checks["policy_roundtrip_equal"]).sum())
            if "policy_roundtrip_equal" in reload_checks
            else pd.NA
        )
        sample_rows = (
            int(pd.to_numeric(reload_checks["sample_rows"], errors="coerce").sum())
            if "sample_rows" in reload_checks
            else pd.NA
        )
        fallback_rows = (
            int(
                pd.to_numeric(
                    reload_checks["fallback_rows_checked"], errors="coerce"
                ).sum()
            )
            if "fallback_rows_checked" in reload_checks
            else pd.NA
        )
        fallback_equal = (
            int(_bool(reload_checks["fallback_mask_roundtrip_equal"]).sum())
            if "fallback_mask_roundtrip_equal" in reload_checks
            else pd.NA
        )
        verification_rows.append(
            {
                "check": "сохранение и загрузка C0 + correction + alpha + policy",
                "checks": int(len(reload_checks)),
                "passed": int(reload_checks["status"].astype(str).eq("passed").sum()),
                "max_abs_difference": maximum_difference,
                "sample_rows": sample_rows,
                "fallback_rows_checked": fallback_rows,
                "fallback_mask_checks_passed": (
                    f"{fallback_equal}/{len(reload_checks)}"
                    if fallback_equal is not pd.NA
                    else pd.NA
                ),
                "policy_invariant": (
                    f"policy совпала: {policy_equal}/{len(reload_checks)}"
                    if policy_equal is not pd.NA
                    else "нет поля policy_roundtrip_equal"
                ),
            }
        )
    lines.extend(["", "### Контрольные проверки alpha=0 и сериализации", ""])
    lines.extend(
        _markdown_table(
            pd.DataFrame(verification_rows),
            (
                ("check", "Проверка"),
                ("checks", "Всего"),
                ("passed", "Успешно"),
                ("max_abs_difference", "Максимальное |Δ|"),
                ("sample_rows", "Строки roundtrip"),
                ("fallback_rows_checked", "Fallback-строки"),
                (
                    "fallback_mask_checks_passed",
                    "Совпала fallback-маска",
                ),
                ("policy_invariant", "Политика"),
            ),
        )
    )

    lines.extend(["", "### Результаты C6 по внешним годам", ""])
    annual_all = _table(artifacts, "annual_metrics", "annual_summary").copy()
    if not annual_all.empty and "model_family" in annual_all:
        annual_all = annual_all[annual_all["model_family"].eq("C6_weather")]
    annual = annual_all
    partial_2026 = pd.DataFrame(columns=annual_all.columns)
    if not annual_all.empty and "fold_id" in annual_all:
        fold_id = annual_all["fold_id"].astype(str)
        annual = annual_all[fold_id.str.fullmatch(r"test_202[0-5]")]
        partial_2026 = annual_all[fold_id.eq("test_2026_partial")]
    lines.extend(
        _markdown_table(
            annual,
            (
                ("fold_id", "Год/fold"),
                ("evaluation_scope", "Сценарий"),
                ("policy_mode", "Политика"),
                ("alpha", "Alpha"),
                ("timely_hits", "Попадания"),
                ("events_with_warning_opportunity", "События"),
                ("messages_per_30_field_days", "Сообщения / 30"),
                ("active_alarm_fraction", "Тревожные дни"),
                ("fallback_day_fraction", "Fallback"),
            ),
            limit=30,
        )
    )
    lines.extend(
        [
            "",
            "### Неполный 2026 год — только отдельная описательная проверка",
            "",
            "Этот блок не входит в агрегаты 2020–2025, bootstrap или выбор модели и политики.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            partial_2026,
            (
                ("fold_id", "Fold"),
                ("evaluation_scope", "Сценарий"),
                ("policy_mode", "Политика"),
                ("alpha", "Alpha"),
                ("timely_hits", "Попадания"),
                ("events_with_warning_opportunity", "События"),
                ("messages_per_30_field_days", "Сообщения / 30"),
                ("active_alarm_fraction", "Тревожные дни"),
                ("fallback_day_fraction", "Fallback"),
            ),
            limit=20,
        )
    )

    lines.extend(["", "## C6 против календарных и погодных конкурентов", ""])
    paired = _table(artifacts, "paired_year_bootstrap", "cycle2_paired_bootstrap")
    formatted = format_paired_bootstrap(paired) if not paired.empty else pd.DataFrame()
    lines.extend(
        _markdown_table(
            formatted,
            tuple((column, column) for column in formatted.columns),
            limit=30,
        )
    )
    lines.extend(["", "### Парные годовые значения 2020–2025", ""])
    paired_annual = _table(artifacts, "paired_annual_metrics")
    lines.extend(
        _markdown_table(
            paired_annual,
            (
                ("season", "Год"),
                ("evaluation_scope", "Сценарий"),
                ("candidate", "C6"),
                ("baseline", "База"),
                ("candidate_hits", "Попадания C6"),
                ("baseline_hits", "Попадания базы"),
                ("opportunities", "События"),
                ("delta_timely_recall", "Δ recall"),
                (
                    "candidate_messages_per_30_field_days",
                    "Сообщения / 30 C6",
                ),
                (
                    "baseline_messages_per_30_field_days",
                    "Сообщения / 30 базы",
                ),
                ("delta_messages_per_30_field_days", "Δ сообщений / 30"),
                ("candidate_active_alarm_fraction", "Тревожные дни C6"),
                ("baseline_active_alarm_fraction", "Тревожные дни базы"),
                ("candidate_fallback_fraction", "Fallback C6"),
            ),
            limit=max(140, len(paired_annual)),
        )
    )
    lines.extend(["", "### Gained/lost относительно C0 и прохождение score через политику", ""])
    gained = _pooled(_table(artifacts, "gained_lost_summary"))
    lines.extend(
        _markdown_table(
            gained,
            (
                ("candidate", "C6"),
                ("evaluation_scope", "Сценарий"),
                ("both_hit", "Оба"),
                ("gained_by_c6", "Gained"),
                ("lost_by_c6", "Lost"),
                ("neither_hit", "Никто"),
                ("net_gain", "Net"),
                ("score_change_altered_message_days_in_event_windows", "Score изменил сообщение"),
                ("score_change_hidden_below_threshold_days_in_event_windows", "Скрыто порогом"),
                (
                    "score_change_hidden_by_cooldown_or_active_alarm_days_in_event_windows",
                    "Скрыто cooldown (suppressed_repeat)",
                ),
            ),
        )
    )
    message_summary = _pooled(_table(artifacts, "message_change_summary"))
    lines.extend(["", "Изменения по всем совпадающим оценочным дням:", ""])
    lines.extend(
        _markdown_table(
            message_summary,
            (
                ("candidate", "C6"),
                ("evaluation_scope", "Сценарий"),
                ("matched_days", "Дни"),
                ("score_changed_days", "Изменился score"),
                ("message_changed_days", "Изменилось сообщение"),
                ("score_change_altered_message_days", "Score изменил сообщение"),
                ("score_change_hidden_below_threshold_days", "Скрыто порогом"),
                (
                    "score_change_hidden_by_cooldown_or_active_alarm_days",
                    "Скрыто cooldown (suppressed_repeat)",
                ),
            ),
        )
    )
    lines.extend(
        [
            "",
            "Категория «скрыто cooldown» требует фактический `suppressed_repeat` и пересечение порога хотя бы одной политикой. Сам по себе `alarm_active` является состоянием результата и не блокирует сообщение в симуляторе.",
        ]
    )

    lines.extend(["", "### Калибровочный и календарный контроли", ""])
    controls = _table(artifacts, "control_comparison", "pooled_summary", "annual_summary")
    if not controls.empty:
        if "period" in controls:
            controls = controls[controls["period"].eq("2020_2025")]
        if "slice" in controls:
            controls = controls[controls["slice"].eq("A_plus_B")]
        if "model_family" in controls:
            controls = controls[
                controls["model_family"].isin(
                    ("C6_weather", "C6_calibration_control", "C6_calendar_control")
                )
            ]
        elif "model_code" in controls:
            controls = controls[controls["model_code"].astype(str).str.startswith("C6_")]
    lines.extend(
        _markdown_table(
            controls,
            (
                ("model_code", "Модель"),
                ("model_family", "Семейство"),
                ("policy_mode", "Политика"),
                ("evaluation_scope", "Сценарий"),
                ("alpha", "Alpha"),
                ("timely_hits", "Попадания"),
                ("events_with_warning_opportunity", "События"),
                ("messages_per_30_field_days", "Сообщения / 30"),
                ("active_alarm_fraction", "Тревожные дни"),
                ("computable_fraction", "Вычислимость"),
                ("fallback_day_fraction", "Fallback"),
            ),
            limit=30,
        )
    )
    control_pairs = _table(artifacts, "paired_bootstrap")
    if not control_pairs.empty:
        control_pairs = control_pairs[
            control_pairs["baseline"].astype(str).str.startswith("C6_")
        ]
        if "period" in control_pairs:
            control_pairs = control_pairs[control_pairs["period"].eq("2020_2025")]
        if "slice" in control_pairs:
            control_pairs = control_pairs[control_pairs["slice"].eq("A_plus_B")]
    lines.extend(["", "Парные дельты погодной C6 против контролей:", ""])
    lines.extend(
        _markdown_table(
            control_pairs,
            (
                ("candidate", "Погодная C6"),
                ("baseline", "Контроль"),
                ("evaluation_scope", "Сценарий"),
                ("candidate_hits", "Попадания C6"),
                ("baseline_hits", "Попадания контроля"),
                ("delta_timely_recall", "Δ recall"),
                ("delta_timely_recall_low", "95% low"),
                ("delta_timely_recall_high", "95% high"),
                ("delta_messages_per_30_field_days", "Δ сообщений / 30"),
                ("delta_active_alarm_fraction", "Δ тревожных дней"),
            ),
            limit=20,
        )
    )

    lines.extend(["", "## Диагностика прежнего Optuna для C4", ""])
    optuna = _table(artifacts, "c4_optuna_external_metrics")
    lines.extend(
        _markdown_table(
            _pooled(optuna),
            (
                ("model_code", "Модель"),
                ("evaluation_scope", "Сценарий"),
                ("timely_hits", "Попадания"),
                ("events_with_warning_opportunity", "События"),
                ("messages_per_30_field_days", "Сообщения / 30"),
                ("active_alarm_fraction", "Тревожные дни"),
            ),
        )
    )
    lines.extend(["", "Пересечения событий фиксированного C4 и C4 Optuna:", ""])
    optuna_intersections = _pooled(_table(artifacts, "c4_optuna_intersections"))
    lines.extend(
        _markdown_table(
            optuna_intersections,
            (
                ("evaluation_scope", "Сценарий"),
                ("events", "События"),
                ("both_hit", "Оба"),
                ("baseline_only_hit", "Только C4"),
                ("candidate_only_hit", "Только C4 Optuna"),
                ("neither_hit", "Никто"),
                ("baseline_hits", "Попадания C4"),
                ("candidate_hits", "Попадания C4 Optuna"),
            ),
        )
    )
    lines.extend(["", "Порог, нагрузка validation и параметры сохранённых C4:", ""])
    optuna_policy = _table(artifacts, "c4_optuna_policy_selection")
    lines.extend(
        _markdown_table(
            optuna_policy,
            (
                ("fold_id", "Fold"),
                ("evaluation_scope", "Сценарий"),
                ("model_code", "Модель"),
                ("threshold", "Порог"),
                ("active_days", "Дней тревоги"),
                ("cooldown_days", "Cooldown"),
                ("validation_policy_feasible", "Бюджет соблюдён"),
                ("validation_timely_recall", "Validation recall"),
                ("validation_messages_per_30", "Validation сообщений / 30"),
                ("validation_alarm_fraction", "Validation тревожных дней"),
                ("model_params", "Параметры"),
                ("model_seed", "Seed"),
                ("fit_population", "Обучающая популяция"),
            ),
            limit=max(40, len(optuna_policy)),
        )
    )
    lines.extend(["", "Внутренние trials и проверка seed:", ""])
    trial_summary = _table(artifacts, "c4_optuna_trial_summary")
    lines.extend(
        _markdown_table(
            trial_summary,
            (
                ("fold_id", "Fold"),
                ("trials", "Trials"),
                ("complete_trials", "Завершены"),
                ("best_internal_value", "Лучший внутренний критерий"),
                ("best_internal_params", "Параметры внутреннего победителя"),
            ),
            limit=12,
        )
    )
    seed_summary = _table(artifacts, "c4_optuna_seed_summary")
    lines.extend(
        _markdown_table(
            seed_summary,
            (
                ("fold_id", "Fold"),
                ("seeds", "Seeds"),
                ("validation_timely_recall_min", "Recall min"),
                ("validation_timely_recall_max", "Recall max"),
                ("validation_messages_per_30_min", "Сообщения min"),
                ("validation_messages_per_30_max", "Сообщения max"),
            ),
            limit=12,
        )
    )
    lines.extend(
        [
            "",
            "Сохранённые trials описывают выбор по внутренней валидации. Их нельзя повторно ранжировать по внешним годам; ухудшение C4 Optuna само по себе не доказывает ошибку реализации или конкретный механизм переобучения.",
            "",
            "## Неопределённость и устойчивость к отдельным годам",
            "",
            f"Парный bootstrap пересэмплирует целые внешние годы, {BOOTSTRAP_REPETITIONS} повторов, seed {BOOTSTRAP_SEED}. Интервалы описательны для шести наблюдённых лет, не учитывают весь поиск модели, зависимость между перекрывающимися обучающими выборками и неопределённость даты регистрации. Наличие нуля в интервале не доказывает равенство методов.",
            "",
        ]
    )
    loo = _table(artifacts, "leave_one_year_out")
    loo_summary_rows: list[dict[str, Any]] = []
    if not loo.empty:
        for keys, group in loo.groupby(
            ["candidate", "baseline", "evaluation_scope"], sort=True, dropna=False
        ):
            loo_summary_rows.append(
                {
                    "candidate": keys[0],
                    "baseline": keys[1],
                    "evaluation_scope": keys[2],
                    "delta_recall_min": group["delta_timely_recall"].min(),
                    "delta_recall_max": group["delta_timely_recall"].max(),
                    "delta_messages_min": group["delta_messages_per_30_field_days"].min(),
                    "delta_messages_max": group["delta_messages_per_30_field_days"].max(),
                    "delta_alarm_min": group["delta_active_alarm_fraction"].min(),
                    "delta_alarm_max": group["delta_active_alarm_fraction"].max(),
                }
            )
    lines.extend(
        _markdown_table(
            pd.DataFrame(loo_summary_rows),
            (
                ("candidate", "C6"),
                ("baseline", "База"),
                ("evaluation_scope", "Сценарий"),
                ("delta_recall_min", "Δ recall min"),
                ("delta_recall_max", "Δ recall max"),
                ("delta_messages_min", "Δ сообщений min"),
                ("delta_messages_max", "Δ сообщений max"),
                ("delta_alarm_min", "Δ тревожных дней min"),
                ("delta_alarm_max", "Δ тревожных дней max"),
            ),
        )
    )

    limitations = metadata.get("limitations")
    if not isinstance(limitations, Sequence) or isinstance(limitations, (str, bytes)):
        limitations = (
            "Даты отражают первую регистрацию, а не доказанную дату заражения или начала симптомов.",
            "Замороженная погода ретроспективна; архивов реально доступных на дату выпуска прогнозов нет.",
            "Нет целевых отрицательных осмотров, поэтому биологические specificity и PPV не установлены.",
            "2020–2025 уже изучались и не являются новым независимым внешним тестом.",
        )
    lines.extend(["", "## Ограничения", ""])
    lines.extend(f"- {str(item)}" for item in limitations)
    lines.extend(["", "## Один следующий приоритет", "", priority])

    commands = metadata.get("reproduction_commands", ())
    if isinstance(commands, Sequence) and not isinstance(commands, (str, bytes)) and commands:
        lines.extend(["", "## Воспроизведение", "", "```bash"])
        lines.extend(str(command) for command in commands)
        lines.append("```")

    text = "\n".join(lines).rstrip() + "\n"
    leaked = sorted(value for value in _privacy_values(artifacts) if value in text)
    if leaked:
        raise AssertionError("A raw field identifier or coordinate value entered the public report")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def _read_saved_frame(directory: Path, name: str) -> pd.DataFrame:
    parquet = directory / f"{name}.parquet"
    csv = directory / f"{name}.csv"
    if parquet.is_file():
        return pd.read_parquet(parquet)
    if csv.is_file():
        return pd.read_csv(csv)
    return pd.DataFrame()


def _default_next_priority(artifacts: Mapping[str, Any]) -> str:
    changes = _pooled(_table(artifacts, "message_change_summary"))
    if not changes.empty:
        hidden = changes.get(
            "score_change_hidden_by_cooldown_or_active_alarm_days", pd.Series(dtype=float)
        ).sum()
        altered = changes.get("score_change_altered_message_days", pd.Series(dtype=float)).sum()
        if hidden > altered:
            return (
                "Проверить один заранее заданный вариант последовательной политики на внутренних "
                "временных данных: разрешить новое сообщение только при существенном росте score во "
                "время cooldown. Сохранить внешние годы, окно 3–10 дней и бюджеты без изменений; "
                "текущая диагностика показывает, что подтверждённое `suppressed_repeat` скрывает больше "
                "изменений score, чем превращается в новые сообщения."
            )
    policies = _table(artifacts, "policy_selection", "alpha_policy_selection")
    weather = policies
    if not policies.empty and "model_code" in policies:
        weather = policies[policies["model_code"].astype(str).eq("C6_weather")]
    if not weather.empty and "alpha" in weather:
        alpha = pd.to_numeric(weather["alpha"], errors="coerce").dropna()
        if len(alpha) and float(alpha.eq(0).mean()) >= 0.5:
            return (
                "Проверить наблюдаемость, временную доступность и стабильность погодных входов: "
                "в большинстве внутренних выборов погодная поправка отключалась через alpha=0."
            )
    return (
        "Проверить наблюдаемость и фактическую доступность погодных данных на дату выпуска "
        "до расширения подбора моделей или запуска Ranker."
    )


def _load_report_inputs(run_dir: Path) -> dict[str, pd.DataFrame]:
    names = (
        "v3_event_intersections",
        "v3_miss_reason_summary",
        "v3_yearly_funnel",
        "v3_funnel_loss_reasons",
        "v3_periodic_30d",
        "v3_periodic_17d_phase_summary",
        "v3_polyakov_computability",
        "v3_polyakov_status_counts",
        "c4_optuna_intersections",
        "c4_optuna_external_metrics",
        "c4_optuna_policy_selection",
        "c4_optuna_trial_summary",
        "c4_optuna_seed_summary",
        "oof_provenance",
        "correction_training_summary",
        "logit_distributions",
        "policy_selection",
        "annual_metrics",
        "pooled_summary",
        "paired_bootstrap",
        "paired_annual_metrics",
        "paired_year_bootstrap",
        "leave_one_year_out",
        "gained_lost_summary",
        "message_change_summary",
        "alpha0_identity_checks",
        "model_reload_verification",
    )
    return {name: _read_saved_frame(run_dir, name) for name in names}


def write_cycle2_report_ru(
    path: str | Path,
    artifacts: Mapping[str, Any] | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
    v3_dir: str | Path | None = None,
    contract: Mapping[str, Any] | None = None,
) -> Path:
    """Write the report either from frames or directly from a completed run.

    ``write_cycle2_report_ru(run_dir, v3_dir=..., contract=...)`` is the pipeline
    facade.  The ``v3_dir`` argument makes the immutable parent explicit; this
    function only reads it and never writes there.  Passing ``artifacts`` keeps
    the pure, in-memory API convenient for unit tests and downstream notebooks.
    """
    supplied_metadata = {} if metadata is None else dict(metadata)
    if artifacts is not None:
        supplied_metadata.setdefault("next_priority", _default_next_priority(artifacts))
        return _render_cycle2_report_ru(
            path, artifacts, metadata=supplied_metadata
        )

    run_dir = Path(path)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if v3_dir is None:
        raise ValueError("v3_dir is required when rendering directly from a run directory")
    parent = Path(v3_dir)
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    loaded = _load_report_inputs(run_dir)
    # The pipeline writes this table under the shorter historical name.
    if loaded["paired_year_bootstrap"].empty and not loaded["paired_bootstrap"].empty:
        paired = loaded["paired_bootstrap"].copy()
        if "period" in paired:
            paired = paired[paired["period"].eq("2020_2025")]
        if "slice" in paired:
            paired = paired[paired["slice"].eq("A_plus_B")]
        loaded["paired_year_bootstrap"] = paired.reset_index(drop=True)

    supplied_metadata.setdefault("run_id", run_dir.name)
    supplied_metadata.setdefault("next_priority", _default_next_priority(loaded))
    supplied_metadata.setdefault(
        "reproduction_commands",
        (
            ".venv/bin/agro-late-blight-cycle2 run \\",
            "  --contract docs/research/late_blight_early_warning/cycle2_evaluation_contract.json \\",
            "  --v3-run results/late_blight_early_warning/20260910_first_cycle_v3 \\",
            "  --run-id <new_unique_cycle2_run_id>",
        ),
    )
    if contract is not None:
        supplied_metadata.setdefault(
            "contract_version", str(contract.get("contract_version", "unknown"))
        )
    return _render_cycle2_report_ru(
        run_dir / "report_ru.md", loaded, metadata=supplied_metadata
    )
