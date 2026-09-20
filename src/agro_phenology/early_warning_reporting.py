"""Aggregate saved early-warning artifacts without fitting or changing models.

All public functions accept DataFrames, including frames reloaded from the run's
Parquet/CSV artifacts.  Counts are pooled before ratios are calculated; daily
rows are never treated as independent events.  Identifiers are used internally
to enforce pairing and never included in the generated Russian report.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


PERIODS = {"2020_2025": (2020, 2025), "2023_2025": (2023, 2025), "2026_partial": (2026, 2026)}
EVALUATION_SCOPES = ("service_calendar", "paired_candidate_days")
PAIRED_COMPARISONS = (
    ("C1", "C0"), ("C2", "C0"), ("C3", "C0"), ("C4", "C0"), ("C5", "C0"),
    ("C0", "calendar_window"), ("C1", "calendar_window"), ("C2", "calendar_window"),
    ("C3", "calendar_window"), ("C4", "calendar_window"), ("C5", "calendar_window"),
    ("C4_optuna", "calendar_window"),
    ("C4_optuna", "C4"), ("hutton", "calendar_window"), ("smith", "calendar_window"),
    ("polyakov", "calendar_window"), ("calendar_and_hutton", "calendar_window"),
)
EVENT_COUNTS = ("first_events", "events_with_warning_opportunity", "timely_hits", "computable_events")
BURDEN_COUNTS = (
    "field_days", "field_seasons", "messages", "active_alarm_days", "computable_days",
    "abstention_days", "suppressed_repeats",
)


def _require(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _years(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "season" not in result:
        _require(result, ["fold_id"], "year-bearing artifact")
        result["season"] = result["fold_id"].astype(str).str.extract(r"^test_(\d{4})(?:_|$)", expand=False)
    result["season"] = pd.to_numeric(result["season"], errors="raise")
    if result["season"].isna().any():
        raise ValueError("Every artifact row must have an unambiguous external test year")
    result["season"] = result["season"].astype(int)
    return result


def _bool(values: pd.Series) -> pd.Series:
    """Accept native booleans and their lossless CSV representations."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    converted = values.map({True: True, False: False, "True": True, "False": False, "true": True, "false": False})
    if (values.notna() & converted.isna()).any():
        raise ValueError("Invalid boolean in a saved evaluation artifact")
    return converted.fillna(False).astype(bool)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p = successes / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return float(centre - half), float(centre + half)


def _selected_days(frame: pd.DataFrame, slice_name: str) -> pd.DataFrame:
    day_column = "evaluation_scope_day" if "evaluation_scope_day" in frame else "service_active"
    result = frame.loc[_bool(frame[day_column])]
    if slice_name == "direct_A":
        result = result.loc[result["coordinate_scope"].eq("A_direct")]
    elif slice_name != "A_plus_B":
        raise ValueError("Burden slices require an outcome-independent population")
    return result


def aggregate_pooled_metrics(
    event_metrics: pd.DataFrame,
    burden_metrics: pd.DataFrame,
    *,
    event_hits: pd.DataFrame | None = None,
    alarm_states: pd.DataFrame | None = None,
    periods: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Pool fold counts for three prespecified periods and both evaluation scopes.

    Annual percentages, quantiles, and confidence bounds are never averaged.
    Optional per-event and per-day artifacts recover pooled lead medians and
    message-count quantiles.  Wilson intervals are descriptive only: they do not
    account for dependence within years or shared weather.
    """
    periods = PERIODS if periods is None else periods
    keys = ["model_code", "evaluation_scope", "slice"]
    events: list[dict] = []
    burdens: list[dict] = []
    eh = _years(event_hits) if event_hits is not None and not event_hits.empty else None
    states = _years(alarm_states) if alarm_states is not None and not alarm_states.empty else None
    for source, counts, output, is_event in (
        (event_metrics, EVENT_COUNTS, events, True), (burden_metrics, BURDEN_COUNTS, burdens, False)
    ):
        if source.empty:
            continue
        _require(source, keys + list(counts), "annual metrics")
        source = _years(source)
        if source.duplicated(keys + ["season"]).any():
            raise ValueError("Duplicate model/scope/slice/test-year metrics would double-count a fold")
        for period, (start, end) in periods.items():
            selected = source.loc[source["season"].between(start, end)]
            for group_key, group in selected.groupby(keys, sort=True, dropna=False):
                row = dict(zip(keys, group_key))
                row.update(period=period, year_start=start, year_end=end, test_years=int(group["season"].nunique()))
                row.update({name: int(group[name].sum()) for name in counts})
                if is_event:
                    n, hits = row["events_with_warning_opportunity"], row["timely_hits"]
                    if hits > n or n > row["first_events"]:
                        raise ValueError("Invalid first-event count hierarchy")
                    low, high = _wilson(hits, n)
                    row.update(
                        timely_recall=_ratio(hits, n), coverage_all_first_events=_ratio(hits, row["first_events"]),
                        opportunity_fraction=_ratio(n, row["first_events"]),
                        computable_event_fraction=_ratio(row["computable_events"], n),
                        timely_recall_wilson_low=low, timely_recall_wilson_high=high,
                        interval_basis="descriptive_event_Wilson_not_dependency_adjusted",
                        median_best_timely_lead_days=np.nan,
                    )
                    if eh is not None and "timely_best_lead_days" in eh:
                        mask = eh["season"].between(start, end)
                        for name, value in zip(keys, group_key):
                            mask &= eh[name].eq(value)
                        leads = eh.loc[mask & _bool(eh["timely_hit"]), "timely_best_lead_days"].dropna()
                        row["median_best_timely_lead_days"] = float(leads.median()) if len(leads) else np.nan
                else:
                    days = row["field_days"]
                    row.update(
                        messages_per_30_field_days=30 * _ratio(row["messages"], days),
                        messages_per_field_season=_ratio(row["messages"], row["field_seasons"]),
                        active_alarm_fraction=_ratio(row["active_alarm_days"], days),
                        computable_fraction=_ratio(row["computable_days"], days),
                        messages_p95_per_field_season=np.nan, messages_max_per_field_season=np.nan,
                    )
                    if states is not None:
                        mask = states["season"].between(start, end)
                        mask &= states["model_code"].eq(row["model_code"])
                        mask &= states["evaluation_scope"].eq(row["evaluation_scope"])
                        daily = _selected_days(states.loc[mask], row["slice"])
                        if len(daily) != days:
                            raise ValueError("Pooled burden denominator differs from saved alarm states")
                        messages = daily.assign(_message=_bool(daily["message_issued"]))
                        per_season = messages.groupby(["season", "field_season"])["_message"].sum()
                        if len(per_season):
                            row["messages_p95_per_field_season"] = float(per_season.quantile(0.95))
                            row["messages_max_per_field_season"] = int(per_season.max())
                output.append(row)
    pooled_events, pooled_burden = pd.DataFrame(events), pd.DataFrame(burdens)
    join_keys = ["period", "year_start", "year_end", "test_years"] + keys
    if pooled_events.empty:
        summary = pooled_burden.copy()
    elif pooled_burden.empty:
        summary = pooled_events.copy()
    else:
        summary = pooled_events.merge(pooled_burden, on=join_keys, how="outer", validate="one_to_one")
    return {"pooled_event_metrics": pooled_events, "pooled_burden_metrics": pooled_burden, "pooled_summary": summary}


def _paired_annual_counts(events: pd.DataFrame, days: pd.DataFrame, candidate: str, baseline: str) -> tuple[pd.DataFrame, str]:
    """Require exact event and evaluation-day pairing; never silently intersect."""
    event_keys = ["season", "field_season"]
    day_keys = event_keys + ["issue_date"]
    paired_frames: list[pd.DataFrame] = []
    for frame, keys, kind in ((events, event_keys, "event"), (days, day_keys, "day")):
        left = frame.loc[frame["model_code"].eq(candidate)].copy()
        right = frame.loc[frame["model_code"].eq(baseline)].copy()
        if left.empty or right.empty:
            return pd.DataFrame(), f"missing_{kind}_population"
        if left.duplicated(keys).any() or right.duplicated(keys).any():
            raise ValueError(f"Duplicate {kind} identities prevent a paired comparison")
        if kind == "event":
            cols = ["warnable_event", "timely_hit"]
            for column in cols:
                left[column] = _bool(left[column]); right[column] = _bool(right[column])
        else:
            cols = ["message_issued", "alarm_active", "computed"]
            for target in (left, right):
                target["message_issued"] = _bool(target["message_issued"])
                target["alarm_active"] = _bool(target["alarm_active"])
                target["computed"] = target["score"].notna()
        paired = left[keys + cols].merge(right[keys + cols], on=keys, how="outer", suffixes=("_candidate", "_baseline"), indicator=True)
        if not paired["_merge"].eq("both").all():
            return pd.DataFrame(), f"non_matching_{kind}_population"
        if kind == "event" and not paired["warnable_event_candidate"].eq(paired["warnable_event_baseline"]).all():
            return pd.DataFrame(), "non_matching_warning_opportunities"
        paired_frames.append(paired)
    events_paired, days_paired = paired_frames
    event_counts = events_paired.assign(
        opportunities=events_paired["warnable_event_candidate"].astype(int),
        candidate_hits=(events_paired["timely_hit_candidate"] & events_paired["warnable_event_candidate"]).astype(int),
        baseline_hits=(events_paired["timely_hit_baseline"] & events_paired["warnable_event_baseline"]).astype(int),
    ).groupby("season")[["opportunities", "candidate_hits", "baseline_hits"]].sum()
    day_counts = days_paired.assign(field_days=1).groupby("season")[[
        "field_days", "message_issued_candidate", "message_issued_baseline", "alarm_active_candidate",
        "alarm_active_baseline", "computed_candidate", "computed_baseline",
    ]].sum()
    return day_counts.join(event_counts, how="outer").fillna(0).sort_index(), "paired"


def _paired_rates(counts: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    values = {name: counts[..., index] for index, name in enumerate(names)}
    def divide(a, b):
        return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan), where=b > 0)
    return {
        "candidate_messages_per_30_field_days": 30 * divide(
            values["message_issued_candidate"], values["field_days"]
        ),
        "baseline_messages_per_30_field_days": 30 * divide(
            values["message_issued_baseline"], values["field_days"]
        ),
        "candidate_active_alarm_fraction": divide(
            values["alarm_active_candidate"], values["field_days"]
        ),
        "baseline_active_alarm_fraction": divide(
            values["alarm_active_baseline"], values["field_days"]
        ),
        "delta_timely_recall": divide(values["candidate_hits"] - values["baseline_hits"], values["opportunities"]),
        "delta_messages_per_30_field_days": 30 * divide(values["message_issued_candidate"] - values["message_issued_baseline"], values["field_days"]),
        "relative_message_reduction": 1 - divide(values["message_issued_candidate"], values["message_issued_baseline"]),
        "delta_active_alarm_fraction": divide(values["alarm_active_candidate"] - values["alarm_active_baseline"], values["field_days"]),
        "delta_computable_fraction": divide(values["computed_candidate"] - values["computed_baseline"], values["field_days"]),
    }


def strong_effect_decision(
    comparison: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    epsilon: float = 1e-12,
) -> dict[str, bool]:
    """Apply the predeclared strong-effect rule to one paired comparison.

    The candidate must satisfy the external research budget for either branch.
    The recall-gain branch additionally requires no increase in either burden
    measure relative to the paired calendar baseline.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    try:
        rule = contract["selection"]["strong_effect"]
        budget = contract["notification_policy"]["research_budget"]
        comparable = rule["comparable_burden"]
        efficiency = rule["efficiency_branch"]
        values = {
            name: float(comparison[name])
            for name in (
                "delta_timely_recall",
                "delta_messages_per_30_field_days",
                "relative_message_reduction",
                "delta_active_alarm_fraction",
                "candidate_messages_per_30_field_days",
                "candidate_active_alarm_fraction",
            )
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Incomplete strong-effect contract or paired comparison") from error
    if not all(np.isfinite(value) for value in values.values()):
        return {
            "within_external_budget": False,
            "comparable_burden": False,
            "recall_gain_branch": False,
            "efficiency_branch": False,
            "strong_effect": False,
        }
    within_budget = bool(
        values["candidate_messages_per_30_field_days"]
        <= float(budget["messages_per_30_field_days_max"]) + epsilon
        and values["candidate_active_alarm_fraction"]
        <= float(budget["active_alarm_fraction_max"]) + epsilon
    )
    budget_gate = bool(
        within_budget
        or not bool(rule.get("candidate_must_meet_external_research_budget", True))
    )
    comparable_burden = bool(
        values["delta_messages_per_30_field_days"]
        <= float(comparable["delta_messages_per_30_field_days_max"]) + epsilon
        and values["delta_active_alarm_fraction"]
        <= float(comparable["delta_active_alarm_fraction_max"]) + epsilon
    )
    recall_branch = bool(
        budget_gate
        and comparable_burden
        and values["delta_timely_recall"]
        >= float(rule["timely_recall_gain_min"]) - epsilon
    )
    efficiency_branch = bool(
        budget_gate
        and values["relative_message_reduction"]
        >= float(efficiency["relative_message_reduction_min"]) - epsilon
        and values["delta_timely_recall"]
        >= -float(efficiency["timely_recall_loss_max"]) - epsilon
        and values["delta_active_alarm_fraction"]
        <= float(efficiency["delta_active_alarm_fraction_max"]) + epsilon
    )
    return {
        "within_external_budget": within_budget,
        "comparable_burden": comparable_burden,
        "recall_gain_branch": recall_branch,
        "efficiency_branch": efficiency_branch,
        "strong_effect": recall_branch or efficiency_branch,
    }


def paired_year_bootstrap(
    event_hits: pd.DataFrame,
    alarm_states: pd.DataFrame,
    *,
    seed: int = 20260910,
    n_bootstrap: int = 2000,
    periods: Mapping[str, tuple[int, int]] | None = None,
    comparisons: Sequence[tuple[str, str]] = PAIRED_COMPARISONS,
    slices: Sequence[str] = ("A_plus_B", "direct_A"),
) -> pd.DataFrame:
    """Paired percentile bootstrap resampling whole external test years.

    The same year multiplicities weight both methods and all counts.  This
    preserves within-year dependence, including weather shared by fields; it
    does not model cross-year persistence of a field, training-set overlap, or
    uncertainty from model fitting.  A single test year yields point estimates
    but no interval.  Intervals are unadjusted for multiple comparisons.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if event_hits.empty or alarm_states.empty:
        return pd.DataFrame()
    periods = PERIODS if periods is None else periods
    events, states = _years(event_hits), _years(alarm_states)
    states["issue_date"] = pd.to_datetime(states["issue_date"])
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for period, (start, end) in periods.items():
        for scope in EVALUATION_SCOPES:
            event_period = events.loc[events["season"].between(start, end) & events["evaluation_scope"].eq(scope)]
            state_period = states.loc[states["season"].between(start, end) & states["evaluation_scope"].eq(scope)]
            if event_period.empty and state_period.empty:
                continue
            for slice_name in slices:
                event_slice = event_period.loc[event_period["slice"].eq(slice_name)]
                day_slice = _selected_days(state_period, slice_name)
                for candidate, baseline in comparisons:
                    row = dict(period=period, evaluation_scope=scope, slice=slice_name, candidate=candidate, baseline=baseline,
                               bootstrap_seed=seed, bootstrap_repetitions=n_bootstrap, block="external_test_year")
                    annual, status = _paired_annual_counts(event_slice, day_slice, candidate, baseline)
                    row["status"] = status
                    if annual.empty:
                        rows.append(row); continue
                    values = annual.to_numpy(dtype=float)
                    names = list(annual.columns)
                    totals = values.sum(axis=0)
                    row.update({name: int(value) for name, value in zip(names, totals)})
                    row["test_years"] = len(annual)
                    estimates = _paired_rates(totals, names)
                    row.update({name: float(value) for name, value in estimates.items()})
                    row["candidate_timely_recall"] = _ratio(row["candidate_hits"], row["opportunities"])
                    row["baseline_timely_recall"] = _ratio(row["baseline_hits"], row["opportunities"])
                    if len(annual) >= 2:
                        weights = rng.multinomial(len(annual), np.full(len(annual), 1 / len(annual)), size=n_bootstrap)
                        samples = _paired_rates(weights @ values, names)
                        for name, values_sample in samples.items():
                            finite = values_sample[np.isfinite(values_sample)]
                            row[f"{name}_valid_repetitions"] = len(finite)
                            low, high = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
                            row[f"{name}_low"], row[f"{name}_high"] = float(low), float(high)
                        row["interval_status"] = "few_year_blocks_unadjusted_percentile_interval"
                    else:
                        row["interval_status"] = "not_estimable_single_year"
                        for name in estimates:
                            row[f"{name}_low"] = row[f"{name}_high"] = np.nan
                            row[f"{name}_valid_repetitions"] = 0
                    rows.append(row)
    return pd.DataFrame(rows)


def registration_delay_sensitivity(
    alarm_states: pd.DataFrame,
    field_seasons: pd.DataFrame,
    *,
    delays: Sequence[int] = (0, 3, 7),
    periods: Mapping[str, tuple[int, int]] | None = None,
    minimum_lead: int = 3,
    maximum_lead: int = 10,
    slices: Sequence[str] = ("A_plus_B", "direct_A"),
) -> pd.DataFrame:
    """Shift event dates backwards by hypothetical registration delays.

    Saved messages and actual entry dates stay fixed.  Shifted warning
    opportunity is based on service entry, never on model computability.
    Report both the shifted-opportunity denominator and the original warnable
    cohort, so apparent gains from dropping late-entry events remain visible.
    This is a sensitivity scenario, not an estimate of biological onset.
    """
    if minimum_lead < 0 or maximum_lead < minimum_lead or any(int(d) != d or d < 0 for d in delays):
        raise ValueError("Invalid delay or lead window")
    if alarm_states.empty or field_seasons.empty:
        return pd.DataFrame()
    periods = PERIODS if periods is None else periods
    states, seasons = _years(alarm_states), _years(field_seasons)
    _require(seasons, ["field_season", "first_recorded_event_date", "first_visit_available_date", "warnable_first_event"], "field_seasons")
    if seasons.duplicated(["season", "field_season"]).any():
        raise ValueError("Field-season registry must contain one row per season")
    states["issue_date"] = pd.to_datetime(states["issue_date"])
    seasons["first_recorded_event_date"] = pd.to_datetime(seasons["first_recorded_event_date"])
    seasons["first_visit_available_date"] = pd.to_datetime(seasons["first_visit_available_date"])
    seasons["warnable_first_event"] = _bool(seasons["warnable_first_event"])
    rows: list[dict] = []
    for period, (start, end) in periods.items():
        subset = states.loc[states["season"].between(start, end)]
        for (model, scope), group in subset.groupby(["model_code", "evaluation_scope"], sort=True):
            registry = seasons.merge(group[["season", "field_season"]].drop_duplicates(), on=["season", "field_season"], how="inner", validate="one_to_one")
            registry = registry.loc[registry["first_recorded_event_date"].notna()]
            messages = group.loc[_bool(group["message_issued"]), ["season", "field_season", "issue_date"]]
            for slice_name in slices:
                event_slice = registry.loc[registry["coordinate_scope"].eq("A_direct")] if slice_name == "direct_A" else registry
                if slice_name not in {"A_plus_B", "direct_A"}:
                    raise ValueError("Unsupported registration-delay sensitivity slice")
                for delay in delays:
                    scenario = event_slice.copy()
                    scenario["shifted_event_date"] = scenario["first_recorded_event_date"] - pd.to_timedelta(int(delay), unit="D")
                    scenario["shifted_opportunity"] = scenario["first_visit_available_date"].le(scenario["shifted_event_date"] - pd.Timedelta(days=minimum_lead))
                    joined = messages.merge(scenario[["season", "field_season", "shifted_event_date"]], on=["season", "field_season"], how="inner", validate="many_to_one")
                    lead = (joined["shifted_event_date"] - joined["issue_date"]).dt.days
                    hit_keys = joined.loc[lead.between(minimum_lead, maximum_lead), ["season", "field_season"]].drop_duplicates().assign(_hit=True)
                    scenario = scenario.merge(hit_keys, on=["season", "field_season"], how="left", validate="one_to_one")
                    hit = scenario["_hit"].eq(True) & scenario["shifted_opportunity"]
                    total, opportunities, original = len(scenario), int(scenario["shifted_opportunity"].sum()), int(scenario["warnable_first_event"].sum())
                    hits = int(hit.sum())
                    fixed_hits = int((hit & scenario["warnable_first_event"]).sum())
                    rows.append(dict(
                        period=period, model_code=model, evaluation_scope=scope, slice=slice_name,
                        hypothetical_registration_delay_days=int(delay), first_events=total,
                        original_events_with_warning_opportunity=original, shifted_events_with_warning_opportunity=opportunities,
                        opportunities_lost_after_shift=original - opportunities, timely_hits=hits,
                        timely_recall_shifted_opportunities=_ratio(hits, opportunities),
                        timely_recall_original_opportunities=_ratio(fixed_hits, original),
                        coverage_all_first_events=_ratio(hits, total),
                        messages_and_policy="unchanged_saved_notifications",
                        interpretation="hypothetical_earlier_event_not_observed_biological_onset",
                    ))
    return pd.DataFrame(rows)


def aggregate_budget_grid_metrics(
    budget_grid_metrics: pd.DataFrame,
    *,
    periods: Mapping[str, tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Pool external-test counts for every prespecified policy-budget cell.

    The saved grid contains exact event, hit, message, and field-day counts but
    stores active alarm days only as a fraction.  The latter is losslessly
    recovered per fold before pooling.  Validation rates are deliberately not
    averaged because their count denominators are not present in the artifact.
    """
    if budget_grid_metrics.empty:
        return pd.DataFrame()
    periods = PERIODS if periods is None else periods
    required = [
        "model_code", "fold_id", "messages_budget_per_30", "alarm_fraction_budget",
        "validation_feasible", "test_events", "test_timely_hits", "test_field_days",
        "test_messages", "test_alarm_fraction",
    ]
    _require(budget_grid_metrics, required, "budget_grid_metrics")
    source = _years(budget_grid_metrics)
    keys = ["model_code", "messages_budget_per_30", "alarm_fraction_budget"]
    if source.duplicated(keys + ["season"]).any():
        raise ValueError("Duplicate model/budget/test-year cells would double-count the grid")
    active_exact = source["test_alarm_fraction"].astype(float) * source["test_field_days"].astype(float)
    active_rounded = np.rint(active_exact)
    if not np.allclose(active_exact, active_rounded, atol=1e-7, rtol=0):
        raise ValueError("Saved alarm fractions do not recover integer active-alarm-day counts")
    source = source.assign(
        _active_alarm_days=active_rounded.astype(int),
        _validation_feasible=_bool(source["validation_feasible"]).astype(int),
    )
    rows: list[dict[str, Any]] = []
    for period, (start, end) in periods.items():
        selected = source.loc[source["season"].between(start, end)]
        for group_key, group in selected.groupby(keys, sort=True, dropna=False):
            model, message_budget, alarm_budget = group_key
            events = int(group["test_events"].sum())
            hits = int(group["test_timely_hits"].sum())
            days = int(group["test_field_days"].sum())
            messages = int(group["test_messages"].sum())
            active_days = int(group["_active_alarm_days"].sum())
            if hits > events:
                raise ValueError("Budget-grid timely hits exceed warning opportunities")
            rows.append({
                "period": period,
                "year_start": start,
                "year_end": end,
                "test_years": int(group["season"].nunique()),
                "model_code": model,
                "messages_budget_per_30": float(message_budget),
                "alarm_fraction_budget": float(alarm_budget),
                "validation_feasible_folds": int(group["_validation_feasible"].sum()),
                "validation_total_folds": int(len(group)),
                "test_events": events,
                "test_timely_hits": hits,
                "test_field_days": days,
                "test_messages": messages,
                "test_active_alarm_days": active_days,
                "test_timely_recall": _ratio(hits, events),
                "test_messages_per_30": 30 * _ratio(messages, days),
                "test_alarm_fraction": _ratio(active_days, days),
                "aggregation": "pooled_external_test_counts",
            })
    return pd.DataFrame(rows)


def build_reporting_artifacts(
    artifacts: Mapping[str, pd.DataFrame], field_seasons: pd.DataFrame, *, seed: int = 20260910, n_bootstrap: int = 2000
) -> dict[str, pd.DataFrame]:
    """Build aggregate artifacts from saved experiment frames; no file I/O or ML."""
    required = ("event_metrics", "burden_metrics", "event_hits", "alarm_states")
    missing = set(required).difference(artifacts)
    if missing:
        raise ValueError(f"Missing experiment artifacts: {sorted(missing)}")
    result = aggregate_pooled_metrics(artifacts["event_metrics"], artifacts["burden_metrics"], event_hits=artifacts["event_hits"], alarm_states=artifacts["alarm_states"])
    result["paired_comparisons"] = paired_year_bootstrap(artifacts["event_hits"], artifacts["alarm_states"], seed=seed, n_bootstrap=n_bootstrap)
    result["registration_delay_sensitivity"] = registration_delay_sensitivity(artifacts["alarm_states"], field_seasons)
    result["budget_grid_pooled"] = aggregate_budget_grid_metrics(
        artifacts.get("budget_grid_metrics", pd.DataFrame())
    )
    return result


def _fmt(value: Any, digits: int = 3, percent: bool = False) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    return f"{100 * value:.1f}%" if percent else f"{value:.{digits}f}"


def _fmt_pp(value: Any) -> str:
    """Format an absolute probability difference in percentage points."""
    return "—" if value is None or pd.isna(value) else f"{100 * float(value):+.1f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(clean(value) for value in row) + " |" for row in rows]


def _audit_counts(audits: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Only explicit aggregate count fields may enter the public report."""
    allowed = {
        "source_rows", "source_columns", "invalid_organism_json", "potato_or_late_blight_source_rows",
        "usable_source_rows", "deduplicated_visits", "field_seasons", "fields", "decision_rows",
        "service_field_days", "observable_target_rows", "common_weather_observable_rows",
        "nasa_computable_service_days", "era_common_computable_service_days", "episode_computable_service_days",
        "candidate_comparison_service_days", "field_seasons_with_observed_bbch51", "service_days_computable",
        "warnable_events_with_any_computable_day", "visits", "positive_visits", "matches_current",
        "first_events", "warnable_events", "positive_at_first_visit", "files_checked", "hashes_matched",
    }
    result: list[tuple[str, Any]] = []
    def visit(node):
        if not isinstance(node, Mapping):
            return
        for key, value in node.items():
            if key in allowed and isinstance(value, (bool, int, float, np.integer, np.floating)):
                result.append((key, value))
            elif isinstance(value, Mapping):
                visit(value)
    visit(audits)
    return result


def _find_audit_value(audits: Mapping[str, Any], key: str) -> Any:
    """Find a named scalar in nested audit mappings without rendering raw paths."""
    if key in audits:
        return audits[key]
    for value in audits.values():
        if isinstance(value, Mapping):
            found = _find_audit_value(value, key)
            if found is not None:
                return found
    return None


def write_report_ru(
    path: str | Path,
    *,
    pooled_summary: pd.DataFrame,
    paired_comparisons: pd.DataFrame,
    delay_sensitivity: pd.DataFrame,
    audits: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
    daily_diagnostics: pd.DataFrame | None = None,
    policy_selection: pd.DataFrame | None = None,
    budget_grid_pooled: pd.DataFrame | None = None,
    optuna_trials: pd.DataFrame | None = None,
    optuna_seed_checks: pd.DataFrame | None = None,
) -> Path:
    """Write a Russian report using aggregates only, without field IDs or coordinates.

    The output is descriptive: no automatic winner, significance claim, or
    biological interpretation is inferred from point estimates.  Paths, raw
    audit strings, hashes, and row-level records are deliberately not rendered.
    """
    lines = [
        "# Ранние предупреждения о первой регистрации фитофтороза картофеля",
        "", "## Что именно оценено", "",
        "Новая задача — первое зарегистрированное обнаружение в поле-сезоне через 3–10 календарных дней после выпуска. "
        "Это регистрационная цель: дата заражения, первых симптомов и полезность обработки по этим данным не установлены. "
        "Дни за 0–2 дня до регистрации составляют отдельный класс imminent; неизвестный исход не считается отрицательным.",
        "", "Историческая задача Валерия предсказывала наличие записи болезни на отдельном визите с погодными признаками, "
        "сдвинутыми относительно визита. Её AP/AUROC нельзя напрямую сопоставлять с событийным timely recall новой задачи. "
        "Новые C0–C5 обучаются на новой цели; старые результаты не переименованы в раннее предупреждение.",
        "", "C0 — календарная логистическая регрессия; C1 — календарный CatBoost; C2 добавляет NASA-агрегаты; "
        "C3 использует общий набор агрегатов ERA5; C4 — восстанавливаемые погодные эпизоды; C5 — логистический контроль "
        "на тех же эпизодных признаках. C4_optuna обозначает отдельный ограниченный подбор. Правила Хаттона, Smith и "
        "Полякова и календарное окно проходят событийную оценку. Стресс-контроли различаются по смыслу: "
        "constant_score_standard_policy — постоянный высокий score при общей политике, always_on_alarm — одно сообщение "
        "и непрерывная тревога, когда этот отдельный контроль присутствует в артефактах; periodic_30d — календарные напоминания.",
        "", "## Популяции, время и знаменатели", "",
        "service_calendar — полный календарь доступного сервису поля, включая дни воздержания и сезоны без зарегистрированного события. "
        "paired_candidate_days — отдельное проигрывание политики на общей маске входов C0–C5: оно изолирует сравнение кандидатов "
        "от различий покрытия погоды. События с возможностью предупредить остаются в знаменателе при воздержании; это также "
        "отражается в вычислимости события. Этот общий погодный слой не гарантирует доступность наблюдённой BBCH51 для Полякова.",
        "", "Возможность предупреждения задаётся входом поля и календарём решений, а не успехом расчёта конкретной модели. "
        "Положительный первый визит не объявляется пропущенным предупреждением. Покрытие всех первых событий использует "
        "более широкий знаменатель, чем timely recall. Нагрузка включает сообщения и активные тревожные дни; это разные величины. "
        "Неподтверждённые уведомления не названы биологически ложными.",
        "", "Внешние годы оцениваются последовательно с обучением и выбором политики только по прошлому. "
        "2020–2025 — основной объединённый период, 2023–2025 — вложенный недавний срез, 2026 — неполный отдельный период. "
        "Вложенные периоды не являются независимыми повторениями. Уже изучавшиеся годы не называются нетронутым holdout. "
        "Погода — past-only реанализ; фактическая историческая оперативная доступность и архивные прогнозные выпуски не подтверждены.",
    ]
    if contract:
        notification = contract.get("notification_policy", {})
        active, cooldown = notification.get("active_days_per_message"), notification.get("cooldown_days")
        if isinstance(active, (int, float)) and isinstance(cooldown, (int, float)):
            lines += ["", f"По сохранённому контракту сообщение активно {int(active)} суток, минимальный интервал повторного сообщения — {int(cooldown)} суток. "
                      "Порог выбирается во внутренней валидации; превышение бюджета во внешнем году сохраняется как результат."]
    old_calendar_ap = _find_audit_value(audits, "old_calendar_test_AP_from_frozen_report")
    old_boosting_ap = _find_audit_value(audits, "old_boosting_test_AP_from_frozen_export")
    if old_calendar_ap is not None or old_boosting_ap is not None:
        lines += ["", "## Исторический результат Валерия — только контекст", ""]
        historical = []
        if old_calendar_ap is not None:
            historical.append(f"календарный контроль AP={float(old_calendar_ap):.4f}")
        if old_boosting_ap is not None:
            historical.append(f"выбранный бустинг AP={float(old_boosting_ap):.4f}")
        lines += [
            "; ".join(historical) + ". Эти значения относятся к старой визитной задаче с повторными положительными "
            "осмотрами и не являются качеством раннего предупреждения первого события. Их нельзя численно сравнивать "
            "с timely recall ниже."
        ]
    counts = _audit_counts(audits)
    if counts:
        lines += ["", "## Сохранённый аудит данных", ""] + _table(["Агрегат аудита", "Значение"], counts)
    lines += ["", "## Объединённые событийные результаты и нагрузка", "",
              "Все доли ниже вычислены из сумм числителей и знаменателей, а не усреднением годовых процентов. "
              "Вычислимость дней относится к знаменателю данного слоя; полнота покрытия событий показана отдельно."]
    if pooled_summary.empty:
        lines += ["", "Сохранённые объединённые метрики отсутствуют; оценка результата недоступна."]
    else:
        main = pooled_summary.loc[pooled_summary["slice"].eq("A_plus_B")]
        for (period, scope), group in main.groupby(["period", "evaluation_scope"], sort=False):
            lines += ["", f"### {period} · {scope}", ""]
            rows = []
            for row in group.sort_values("model_code").to_dict("records"):
                rows.append([row["model_code"], f"{_fmt(row.get('timely_hits'), 0)}/{_fmt(row.get('events_with_warning_opportunity'), 0)}",
                             _fmt(row.get("timely_recall"), percent=True), _fmt(row.get("coverage_all_first_events"), percent=True),
                             _fmt(row.get("messages_per_30_field_days")), _fmt(row.get("active_alarm_fraction"), percent=True),
                             _fmt(row.get("field_days"), 0), _fmt(row.get("computable_fraction"), percent=True),
                             _fmt(row.get("computable_event_fraction"), percent=True)])
            lines += _table(["Метод", "Hit / доступные события", "Timely", "Все первые события", "Сообщ./30 дней", "Дни тревоги", "Поле-дни", "Расчёт дней", "Расчёт событий"], rows)
        visit_sensitivity = pooled_summary.loc[
            pooled_summary["period"].eq("2020_2025")
            & pooled_summary["evaluation_scope"].eq("service_calendar")
            & pooled_summary["slice"].isin(["prior_gap_le14", "prior_gap_le21"])
            & pooled_summary["model_code"].isin(
                ["C0", "C1", "C2", "C3", "C4", "C5", "calendar_window", "hutton", "polyakov"]
            )
        ]
        if not visit_sensitivity.empty:
            lines += ["", "## Чувствительность к процессу визитов", "",
                      "Ниже приведены вложенные условные подмножества первых событий, для которых интервал от предыдущего визита "
                      "до регистрации не превышал 14 или 21 день. Это проверка чувствительности к наблюдаемости; такой отбор не "
                      "исправляет observation bias и не превращает дату регистрации в дату начала болезни. Нагрузка для этих "
                      "событийных срезов не переносится на всю сервисную популяцию."]
            rows = []
            for model in ["C0", "C1", "C2", "C3", "C4", "C5", "calendar_window", "hutton", "polyakov"]:
                cells = []
                for slice_name in ("prior_gap_le14", "prior_gap_le21"):
                    selected = visit_sensitivity.loc[
                        visit_sensitivity["model_code"].eq(model)
                        & visit_sensitivity["slice"].eq(slice_name)
                    ]
                    if selected.empty:
                        cells.append("—")
                    else:
                        row = selected.iloc[0]
                        cells.append(
                            f"{int(row['timely_hits'])}/{int(row['events_with_warning_opportunity'])}; "
                            f"{_fmt(row['timely_recall'], percent=True)}"
                        )
                rows.append([model] + cells)
            lines += [""] + _table(["Метод", "Интервал ≤14 дней", "Интервал ≤21 день"], rows)
        lines += ["", "Срез direct_A и ограничения интервала между предыдущим визитом и регистрацией сохранены в объединённых "
                  "таблицах, если присутствовали во входных артефактах. Для исходов с такими ограничениями нельзя автоматически "
                  "приписывать нагрузку всей популяции. Объединённые Wilson-интервалы являются описательными и не устраняют зависимость событий."]
    lines += ["", "## Парные сравнения и неопределённость", "",
              "Кандидат и базовый метод сопоставляются на одних событиях и одних оценочных поле-днях. "
              "При несовпадении популяций сравнение получает диагностический статус вместо скрытого пересечения выборок. "
              "Bootstrap пересэмплирует целые внешние test-годы с одинаковыми весами для обоих методов. "
              "Он сохраняет зависимость полей и общей погоды внутри года, но не учитывает повторение поля между годами, "
              "перекрытие обучающих выборок и повторную подгонку моделей. При шести, а тем более трёх годовых блоках "
              "интервалы нестабильны; поправка на множественные сравнения не применяется. Для одного года интервал не оценивается."]
    if not paired_comparisons.empty:
        repetitions = sorted(pd.to_numeric(paired_comparisons["bootstrap_repetitions"]).dropna().astype(int).unique())
        seeds = sorted(pd.to_numeric(paired_comparisons["bootstrap_seed"]).dropna().astype(int).unique())
        lines += ["", f"Сохранённые настройки: повторов {', '.join(map(str, repetitions))}; seed {', '.join(map(str, seeds))}."]
        for (period, scope), group in paired_comparisons.loc[paired_comparisons["slice"].eq("A_plus_B")].groupby(["period", "evaluation_scope"], sort=False):
            rows = []
            for row in group.to_dict("records"):
                interval = f"[{_fmt_pp(row.get('delta_timely_recall_low'))}; {_fmt_pp(row.get('delta_timely_recall_high'))}]"
                rows.append([f"{row['candidate']} / {row['baseline']}", _fmt_pp(row.get("delta_timely_recall")), interval,
                             _fmt(row.get("delta_messages_per_30_field_days")), _fmt_pp(row.get("delta_active_alarm_fraction")),
                             row.get("interval_status", row["status"])])
            lines += ["", f"### {period} · {scope}", ""] + _table(["Сравнение", "Δ timely, п.п.", "95% интервал Δ", "Δ сообщ./30", "Δ тревоги, п.п.", "Статус"], rows)
    else:
        lines += ["", "Парные сравнения отсутствуют; статистическое преимущество не оценено."]
    primary = paired_comparisons.loc[
        paired_comparisons.get("period", pd.Series(dtype=object)).eq("2020_2025")
        & paired_comparisons.get("evaluation_scope", pd.Series(dtype=object)).eq("paired_candidate_days")
        & paired_comparisons.get("slice", pd.Series(dtype=object)).eq("A_plus_B")
        & paired_comparisons.get("baseline", pd.Series(dtype=object)).eq("calendar_window")
        & paired_comparisons.get("candidate", pd.Series(dtype=object)).isin(
            ["C0", "C1", "C2", "C3", "C4", "C5", "C4_optuna"]
        )
        & paired_comparisons.get("status", pd.Series(dtype=object)).eq("paired")
    ] if not paired_comparisons.empty else pd.DataFrame()
    lines += ["", "## Вывод о превосходстве", ""]
    if primary.empty:
        lines += [
            "Сильное превосходство над календарным окном в первом цикле не подтверждено: для предусмотренного "
            "парного основного слоя недостаточно сохранённых сравнений."
        ]
    else:
        best = primary.sort_values("delta_timely_recall", ascending=False).iloc[0]
        strong_rule = (
            contract.get("selection", {}).get("strong_effect")
            if contract is not None else None
        )
        decisions = []
        if strong_rule is not None:
            decisions = [strong_effect_decision(row, contract) for row in primary.to_dict("records")]
        if any(item["strong_effect"] for item in decisions):
            conclusion = "Заранее заданный практический порог эффекта достигнут хотя бы одной конфигурацией; это требует отдельной проверки устойчивости."
        else:
            conclusion = "Сильное превосходство над календарным окном в первом цикле не подтверждено."
        lines += [
            f"{conclusion} Лучшее точечное изменение timely recall на общем слое 2020–2025: "
            f"{best['candidate']} относительно calendar_window, {_fmt_pp(best['delta_timely_recall'])} п.п.; "
            "заранее заданный ориентир составлял +15 п.п. либо снижение сообщений не менее чем на 30% без потери recall "
            "более 5 п.п. и без роста тревожных дней."
        ]
        tuned = paired_comparisons.loc[
            paired_comparisons["period"].eq("2020_2025")
            & paired_comparisons["evaluation_scope"].eq("paired_candidate_days")
            & paired_comparisons["slice"].eq("A_plus_B")
            & paired_comparisons["candidate"].eq("C4_optuna")
            & paired_comparisons["baseline"].eq("C4")
            & paired_comparisons["status"].eq("paired")
        ]
        if not tuned.empty:
            row = tuned.iloc[0]
            lines += [
                f"Ограниченный подбор C4_optuna изменил timely recall относительно фиксированного C4 на "
                f"{_fmt_pp(row['delta_timely_recall'])} п.п. при Δ сообщений {_fmt(row['delta_messages_per_30_field_days'])} "
                "на 30 поле-дней; это не даёт основания выбирать настроенную версию по внешнему результату."
            ]
    if budget_grid_pooled is not None and not budget_grid_pooled.empty:
        lines += ["", "## Сетка бюджетов уведомлений", "",
                  "Таблица показывает внешний результат 2020–2025 на полном сервисном календаре при бюджете тревожных дней 50%. "
                  "Каждая ячейка: timely hits/события; сообщений на 30 поле-дней; доля тревожных дней; число выполнимых "
                  "валидационных разбиений. Полная сетка 3×3 сохранена отдельным CSV. Все показатели объединены через суммы counts."]
        compact = budget_grid_pooled.loc[
            budget_grid_pooled["period"].eq("2020_2025")
            & np.isclose(budget_grid_pooled["alarm_fraction_budget"].astype(float), 0.5)
            & budget_grid_pooled["model_code"].isin(["C0", "C1", "C2", "C3", "C4", "C5", "C4_optuna"])
        ]
        rows = []
        for model, group in compact.groupby("model_code", sort=True):
            cells = []
            for budget in (1.0, 2.0, 3.0):
                selected = group.loc[np.isclose(group["messages_budget_per_30"].astype(float), budget)]
                if selected.empty:
                    cells.append("—")
                    continue
                row = selected.iloc[0]
                cells.append(
                    f"{int(row['test_timely_hits'])}/{int(row['test_events'])}; "
                    f"{row['test_messages_per_30']:.2f}; {100 * row['test_alarm_fraction']:.1f}%; "
                    f"{int(row['validation_feasible_folds'])}/{int(row['validation_total_folds'])}"
                )
            rows.append([model] + cells)
        lines += [""] + _table(["Модель", "Бюджет сообщ. 1", "Бюджет сообщ. 2", "Бюджет сообщ. 3"], rows)
    if optuna_trials is not None and not optuna_trials.empty:
        states = optuna_trials["state"].astype(str).str.upper()
        completed = int(states.eq("COMPLETE").sum())
        folds = int(optuna_trials["fold_id"].nunique()) if "fold_id" in optuna_trials else 0
        lines += ["", "## Ограниченная настройка Optuna", "",
                  f"Завершено {completed} из {len(optuna_trials)} сохранённых испытаний для {folds} внешних разбиений. "
                  "Настройка выполнялась только внутри временной валидации; внешний результат не использовался для выбора параметров."]
        if optuna_seed_checks is not None and not optuna_seed_checks.empty:
            checks = optuna_seed_checks.copy()
            checks["validation_timely_recall"] = pd.to_numeric(checks["validation_timely_recall"], errors="coerce")
            seed_count = int(checks["seed"].nunique()) if "seed" in checks else 0
            per_fold = checks.groupby("fold_id")["validation_timely_recall"].agg(["min", "max"])
            checks_per_fold = checks.groupby("fold_id").size()
            checks_description = (
                f"по {int(checks_per_fold.iloc[0])} seed на fold"
                if len(checks_per_fold) and checks_per_fold.nunique() == 1
                else "с неодинаковым числом seed по folds"
            )
            rows = [[fold, _fmt(row["min"], percent=True), _fmt(row["max"], percent=True)] for fold, row in per_fold.iterrows()]
            lines += [f"Выполнена {len(checks)} проверка с {seed_count} различными значениями seed, {checks_description}; "
                      "диапазоны validation timely recall:", ""]
            lines += _table(["Fold", "Минимум", "Максимум"], rows)
    lines += ["", "## Чувствительность к задержке регистрации", "",
              "Дата события гипотетически сдвинута на 0, 3 или 7 дней назад. Реальные сохранённые сообщения, политика и "
              "дата подключения поля остаются прежними. Это сценарий чувствительности, а не восстановленная дата заражения. "
              "Основная доля в таблице сохраняет исходный знаменатель событий с возможностью предупредить; число возможностей "
              "после сдвига показано отдельно, чтобы потеря доступных событий не создавала искусственного роста recall."]
    if not delay_sensitivity.empty:
        main = delay_sensitivity.loc[delay_sensitivity["period"].eq("2020_2025") & delay_sensitivity["slice"].eq("A_plus_B")]
        for scope, group in main.groupby("evaluation_scope", sort=False):
            rows = []
            for model, model_rows in group.groupby("model_code", sort=True):
                cells = []
                for delay in (0, 3, 7):
                    selected = model_rows.loc[model_rows["hypothetical_registration_delay_days"].eq(delay)]
                    if selected.empty:
                        cells.append("—"); continue
                    row = selected.iloc[0]
                    cells.append(f"{int(row['timely_hits'])}/{int(row['original_events_with_warning_opportunity'])}; доступно {int(row['shifted_events_with_warning_opportunity'])}")
                rows.append([model] + cells)
            lines += ["", f"### 2020–2025 · {scope}", ""] + _table(["Метод", "Сдвиг 0 дней", "Сдвиг 3 дня", "Сдвиг 7 дней"], rows)
        lines += ["", "Остальные периоды и direct_A присутствуют в отдельном агрегированном артефакте чувствительности."]
    if daily_diagnostics is not None and not daily_diagnostics.empty:
        lines += ["", "## Дневная диагностика", "",
                  f"Сохранено {len(daily_diagnostics)} строк диагностических результатов. AP, AUROC, Brier и log loss "
                  "относятся к дневной регистрационной цели и зависимым строкам, поэтому не заменяют событийное покрытие. "
                  "Они не усредняются по годам как объединённая метрика; для общего AP требуется пересчёт по сохранённым предсказаниям."]
    if policy_selection is not None and not policy_selection.empty and "validation_policy_feasible" in policy_selection:
        values = policy_selection["validation_policy_feasible"]
        known = values.notna()
        feasible = int(_bool(values.loc[known]).sum())
        lines += ["", f"В артефакте выбора политики {int(known.sum())} проверенных записей, из них {feasible} удовлетворяют "
                  "сохранённому внутреннему бюджету. Это не гарантия соблюдения бюджета во внешнем году."]
    lines += ["", "## Как интерпретировать результат", "",
              "Точечное улучшение не объявляется доказанным преимуществом. Его следует оценивать вместе с парным интервалом, "
              "годовыми counts, нагрузкой, вычислимостью, direct_A и чувствительностью к дате регистрации. "
              "Снижение числа сообщений на полном сервисном календаре может возникать из-за отсутствия замороженной погоды; "
              "поэтому вывод о новых признаках должен выдерживать общий слой paired_candidate_days. "
              "Даже прохождение заранее выбранного практического порога эффекта не заменяет оценку неопределённости и новую внешнюю проверку.",
              "", "Основные ограничения: неизвестные время публикации визита и фактическая задержка регистрации; "
              "неполное подтверждение отрицательных осмотров; редкая наблюдённая BBCH51; зависимость наблюдений и общей погоды; "
              "малое число лет; исторически изученные внешние периоды и неполный 2026 год. "
              "Эти результаты не подтверждают биологическую дату начала болезни и не обосновывают рекомендацию обработки.",
              "", "## Один следующий приоритет", "",
              "Следующий приоритет — C6: календарный logit как зафиксированная базовая вероятность плюс CatBoost-коррекция по погоде. "
              "Нулевая поправка должна точно воспроизводить календарный контроль, а добавочная ценность погоды оценивается по OOF-прогнозам "
              "в том же временном и парном контракте. Это прямо проверит, улучшает ли погода сильный календарный ориентир, не меняя цель, окно "
              "успеха или внешний тест по увиденному результату.", ""]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
