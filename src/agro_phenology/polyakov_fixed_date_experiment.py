"""Sensitivity replay of Polyakov with BBCH51 fixed to 1 July.

The fixed date is an explicit research convention, not an inferred biological
date.  Existing v3 scores and every previous result directory remain frozen.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .early_warning_core import load_snapshot_module, sha256_file
from .early_warning_cycle2_pipeline import verify_v3
from .early_warning_models import Policy, burden_metrics, event_metrics, simulate_policy
from .era_hutton_backfill import (
    DEFAULT_FROZEN_ERA,
    DEFAULT_V3,
    EXPECTED_V3_MANIFEST_SHA256,
    _atomic_csv,
    _atomic_json,
    _atomic_text,
    _verify_completed_run as verify_era_backfill,
    _write_parquet_idempotent,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ERA_BACKFILL_RUN = (
    REPO_ROOT / "results/late_blight_early_warning/20260915_era5_hutton_backfill_v1"
)
DEFAULT_EXTENDED_ERA = DEFAULT_ERA_BACKFILL_RUN / "era5_potato_daily_extended.parquet"
DEFAULT_SNAPSHOT_LATE_BLIGHT = (
    REPO_ROOT
    / "docs/extra/vaad_pipeline_repro_20260905/project/src/agro_phenology/late_blight.py"
)
DEFAULT_RUN = (
    REPO_ROOT / "results/late_blight_early_warning/20260915_polyakov_fixed_july1_v1"
)
RUN_FORMAT = "agro_phenology_polyakov_fixed_july1_v1"
TEST_YEARS = tuple(range(2020, 2026))
FIXED_MONTH = 7
FIXED_DAY = 1
POLICY = Policy(0.5, 7, 15, "policy_v1_fixed_binary_rule")

OBSERVED_CODE = "polyakov_observed_bbch51_v3"
FROZEN_CODE = "polyakov_fixed_july1_frozen_era"
EXTENDED_CODE = "polyakov_fixed_july1_extended_era"
CALENDAR_CODE = "calendar_window"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _activation_dates(seasons: pd.Series) -> pd.Series:
    return pd.to_datetime(
        seasons.astype(int).astype(str) + f"-{FIXED_MONTH:02d}-{FIXED_DAY:02d}"
    )


def build_fixed_polyakov_scores(
    decisions: pd.DataFrame,
    era_daily: pd.DataFrame,
    snapshot_late_blight: Path,
    *,
    years: Iterable[int] = TEST_YEARS,
    source_label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Set BBCH51 to 1 July and derive a causal binary Polyakov signal.

    The phase gate is deterministic and known from the start of the season.
    Before 1 July, and during the nine-day post-activation warm-up in which a
    ten-day window cannot yet exist, the rule is deterministically inactive
    (score zero).  Missing weather after warm-up remains an abstention.
    """
    required_decisions = {
        "field_season",
        "weather_cell",
        "season",
        "issue_date",
        "feature_cutoff_era_episode_date",
        "evaluation_field_day",
    }
    missing = required_decisions.difference(decisions.columns)
    if missing:
        raise ValueError(f"daily_decisions lacks {sorted(missing)}")
    required_weather = {
        "weather_cell",
        "date",
        "temperature_mean_c",
        "relative_humidity_mean_pct",
        "precipitation_sum_mm",
        "accepted",
    }
    missing = required_weather.difference(era_daily.columns)
    if missing:
        raise ValueError(f"ERA5 daily data lacks {sorted(missing)}")

    chosen_years = tuple(sorted({int(year) for year in years}))
    frame = decisions.copy().reset_index(drop=True)
    frame["_row_order"] = np.arange(len(frame))
    frame["issue_date"] = pd.to_datetime(frame["issue_date"]).dt.normalize()
    frame["feature_cutoff_era_episode_date"] = pd.to_datetime(
        frame["feature_cutoff_era_episode_date"]
    ).dt.normalize()
    if (
        frame["feature_cutoff_era_episode_date"]
        > frame["issue_date"] - pd.Timedelta(days=2)
    ).any():
        raise AssertionError("A weather cutoff is later than issue_date - 2 days")
    weather = era_daily.copy()
    weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()
    if weather.duplicated(["weather_cell", "date"]).any():
        raise ValueError("ERA5 daily data contains duplicate weather_cell/date keys")

    late_blight = load_snapshot_module(
        snapshot_late_blight,
        "agro_snapshot_late_blight_polyakov_fixed_july1",
    )
    config = late_blight.PolyakovConfig()
    lookup_tables: list[pd.DataFrame] = []
    needed = frame.loc[
        frame["season"].isin(chosen_years), ["weather_cell", "season"]
    ].drop_duplicates()
    for cell, year in needed.sort_values(["weather_cell", "season"]).itertuples(
        index=False
    ):
        subset = weather[
            weather["weather_cell"].eq(cell) & weather["date"].dt.year.eq(int(year))
        ][
            [
                "date",
                "temperature_mean_c",
                "relative_humidity_mean_pct",
                "precipitation_sum_mm",
                "accepted",
            ]
        ].copy()
        if subset.empty:
            continue
        subset["date"] = subset["date"].dt.date
        activation = pd.Timestamp(
            year=int(year), month=FIXED_MONTH, day=FIXED_DAY
        ).date()
        classified = late_blight.classify_polyakov_windows(
            subset, activation, config
        )
        classified["weather_cell"] = str(cell)
        classified["season"] = int(year)
        classified["cutoff_date"] = pd.to_datetime(classified["date"])
        lookup_tables.append(
            classified[
                [
                    "weather_cell",
                    "season",
                    "cutoff_date",
                    "status",
                    "critical",
                    "t10_c",
                    "rh10_pct",
                    "p10_mm",
                ]
            ].rename(
                columns={
                    "status": "polyakov_weather_status",
                    "critical": "polyakov_weather_critical",
                    "t10_c": "polyakov_t10_c",
                    "rh10_pct": "polyakov_rh10_pct",
                    "p10_mm": "polyakov_p10_mm",
                }
            )
        )
    lookup = pd.concat(lookup_tables, ignore_index=True) if lookup_tables else pd.DataFrame()
    if len(lookup) and lookup.duplicated(
        ["weather_cell", "season", "cutoff_date"]
    ).any():
        raise AssertionError("Polyakov daily lookup contains duplicate keys")
    result = frame.merge(
        lookup,
        left_on=["weather_cell", "season", "feature_cutoff_era_episode_date"],
        right_on=["weather_cell", "season", "cutoff_date"],
        how="left",
        validate="many_to_one",
    ).drop(columns="cutoff_date")
    result = result.sort_values("_row_order").drop(columns="_row_order").reset_index(
        drop=True
    )
    activation = _activation_dates(result["season"])
    cutoff = result["feature_cutoff_era_episode_date"]
    before_activation = cutoff.lt(activation)
    deterministic_warmup = cutoff.between(
        activation, activation + pd.Timedelta(days=config.window_days - 2)
    )
    result["polyakov_activation_date_fixed"] = activation
    result["polyakov_activation_policy"] = "fixed_calendar_date_july_1_known_in_advance"
    result["polyakov_score_fixed"] = np.nan
    result["polyakov_status_fixed"] = "missing_weather_after_warmup"
    result.loc[before_activation, "polyakov_score_fixed"] = 0.0
    result.loc[
        before_activation, "polyakov_status_fixed"
    ] = "not_active_before_fixed_july1"
    result.loc[deterministic_warmup, "polyakov_score_fixed"] = 0.0
    result.loc[
        deterministic_warmup, "polyakov_status_fixed"
    ] = "deterministic_post_activation_warmup"
    negative_statuses = {"LOW_WEATHER_RISK", "CRITICAL_CONDITIONS"}
    positive_statuses = {"OUTBREAK_EXPECTED", "PROLONGED_RISK"}
    negative = result["polyakov_weather_status"].isin(negative_statuses)
    positive = result["polyakov_weather_status"].isin(positive_statuses)
    result.loc[negative, "polyakov_score_fixed"] = 0.0
    result.loc[positive, "polyakov_score_fixed"] = 1.0
    result.loc[negative | positive, "polyakov_status_fixed"] = result.loc[
        negative | positive, "polyakov_weather_status"
    ]
    insufficient = result["polyakov_weather_status"].eq("INSUFFICIENT_DATA")
    after_warmup = cutoff.gt(activation + pd.Timedelta(days=config.window_days - 2))
    result.loc[
        insufficient & after_warmup, "polyakov_status_fixed"
    ] = "insufficient_weather_after_warmup"
    result["polyakov_weather_evaluable_fixed"] = (
        after_warmup
        & result["polyakov_weather_status"].isin(negative_statuses | positive_statuses)
    )

    values = set(result["polyakov_score_fixed"].dropna().unique())
    if not values.issubset({0.0, 1.0}):
        raise AssertionError(f"Polyakov score is not binary: {values}")
    service = result[
        result["season"].isin(chosen_years)
        & result["evaluation_field_day"].astype(bool)
    ]
    actionable = service["days_to_first_recorded_event"].between(3, 10)
    audit = {
        "source_label": source_label,
        "years": list(chosen_years),
        "fixed_activation_month_day": "07-01",
        "fixed_date_is_research_assumption_not_observed_phenology": True,
        "window_days": int(config.window_days),
        "persistence_days": int(config.persistence_days),
        "manifestation_lag_days": list(config.manifestation_lag_days),
        "service_field_days": int(len(service)),
        "policy_score_available_service_days": int(
            service["polyakov_score_fixed"].notna().sum()
        ),
        "policy_score_abstention_service_days": int(
            service["polyakov_score_fixed"].isna().sum()
        ),
        "structural_inactive_or_warmup_service_days": int(
            service["polyakov_status_fixed"].isin(
                {
                    "not_active_before_fixed_july1",
                    "deterministic_post_activation_warmup",
                }
            ).sum()
        ),
        "weather_evaluable_service_days_after_warmup": int(
            service["polyakov_weather_evaluable_fixed"].sum()
        ),
        "weather_abstention_service_days_after_warmup": int(
            (
                ~service["polyakov_weather_evaluable_fixed"]
                & ~service["polyakov_status_fixed"].isin(
                    {
                        "not_active_before_fixed_july1",
                        "deterministic_post_activation_warmup",
                    }
                )
            ).sum()
        ),
        "weather_evaluable_warnable_events": int(
            service.loc[
                actionable & service["polyakov_weather_evaluable_fixed"],
                "field_season",
            ].nunique()
        ),
        "positive_signal_service_days": int(service["polyakov_score_fixed"].eq(1).sum()),
        "status_counts": {
            str(key): int(value)
            for key, value in service["polyakov_status_fixed"].value_counts().items()
        },
    }
    return result, audit


def _fold_id(year: int) -> str:
    return f"test_{year}"


def _replay(
    frame: pd.DataFrame, score_column: str, model_code: str
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for year in TEST_YEARS:
        test = frame[frame["season"].eq(year)].copy()
        states = simulate_policy(
            test,
            test[score_column],
            POLICY,
            "service_calendar",
        )
        states["model_code"] = model_code
        states["fold_id"] = _fold_id(year)
        outputs.append(states)
    return pd.concat(outputs, ignore_index=True)


def _assert_saved_identity(
    replay: pd.DataFrame,
    saved_states: pd.DataFrame,
    saved_model_code: str,
) -> None:
    columns = [
        "field_season",
        "season",
        "issue_date",
        "score",
        "score_status",
        "message_issued",
        "alarm_active",
        "suppressed_repeat",
        "action_reason",
    ]
    expected = saved_states[
        saved_states["model_code"].eq(saved_model_code)
        & saved_states["evaluation_scope"].eq("service_calendar")
        & saved_states["season"].isin(TEST_YEARS)
    ][columns].copy()
    actual = replay[columns].copy()
    for item in (expected, actual):
        item["issue_date"] = pd.to_datetime(item["issue_date"])
        item.sort_values(["season", "field_season", "issue_date"], inplace=True)
        item.reset_index(drop=True, inplace=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)


def _summary(
    states: pd.DataFrame,
    seasons: pd.DataFrame,
    model_code: str,
    period: str,
    years: Iterable[int],
) -> tuple[dict[str, Any], pd.DataFrame]:
    years = tuple(int(year) for year in years)
    selected_states = states[states["season"].isin(years)]
    selected_seasons = seasons[seasons["season"].isin(years)]
    event, hits = event_metrics(
        selected_states,
        selected_seasons,
        model_code,
        period,
        "A_plus_B",
        "service_calendar",
    )
    burden = burden_metrics(
        selected_states, model_code, period, "A_plus_B", "service_calendar"
    )
    return (
        {
            **event,
            **burden,
            "period": period,
            "year_start": min(years),
            "year_end": max(years),
            "within_research_budget": bool(
                burden["messages_per_30_field_days"] <= 2.0
                and burden["active_alarm_fraction"] <= 0.5
            ),
        },
        pd.DataFrame(hits),
    )


def _paired_categories(candidate: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, int]:
    candidate = candidate[candidate["warnable_event"]][
        ["field_season", "timely_hit"]
    ].rename(columns={"timely_hit": "candidate_hit"})
    baseline = baseline[baseline["warnable_event"]][
        ["field_season", "timely_hit"]
    ].rename(columns={"timely_hit": "baseline_hit"})
    paired = candidate.merge(
        baseline, on="field_season", how="outer", validate="one_to_one"
    )
    if paired.isna().any().any():
        raise AssertionError("Comparison changed warnable event population")
    return {
        "events": int(len(paired)),
        "both": int((paired["candidate_hit"] & paired["baseline_hit"]).sum()),
        "candidate_only": int(
            (paired["candidate_hit"] & ~paired["baseline_hit"]).sum()
        ),
        "baseline_only": int(
            (~paired["candidate_hit"] & paired["baseline_hit"]).sum()
        ),
        "neither": int((~paired["candidate_hit"] & ~paired["baseline_hit"]).sum()),
    }


def _bootstrap(
    candidate_states: pd.DataFrame,
    baseline_states: pd.DataFrame,
    candidate_hits: pd.DataFrame,
    baseline_hits: pd.DataFrame,
    *,
    candidate_code: str,
    baseline_code: str,
    draws: int = 20_000,
    seed: int = 20260915,
) -> dict[str, Any]:
    rows: list[list[float]] = []
    for year in TEST_YEARS:
        candidate_year = candidate_states[candidate_states["season"].eq(year)]
        baseline_year = baseline_states[baseline_states["season"].eq(year)]
        candidate_event = candidate_hits[
            candidate_hits["season"].eq(year) & candidate_hits["warnable_event"]
        ]
        baseline_event = baseline_hits[
            baseline_hits["season"].eq(year) & baseline_hits["warnable_event"]
        ]
        if set(candidate_event["field_season"]) != set(baseline_event["field_season"]):
            raise AssertionError("Bootstrap candidates do not share event population")
        rows.append(
            [
                len(candidate_event),
                candidate_event["timely_hit"].sum(),
                baseline_event["timely_hit"].sum(),
                candidate_year["evaluation_scope_day"].sum(),
                candidate_year["message_issued"].sum(),
                baseline_year["message_issued"].sum(),
                candidate_year["alarm_active"].sum(),
                baseline_year["alarm_active"].sum(),
            ]
        )
    matrix = np.asarray(rows, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = matrix[
        rng.integers(0, len(TEST_YEARS), size=(draws, len(TEST_YEARS)))
    ].sum(axis=1)
    distributions = {
        "delta_timely_recall": (sampled[:, 1] - sampled[:, 2]) / sampled[:, 0],
        "delta_messages_per_30_field_days": 30
        * (sampled[:, 4] - sampled[:, 5])
        / sampled[:, 3],
        "delta_active_alarm_fraction": (sampled[:, 6] - sampled[:, 7])
        / sampled[:, 3],
    }
    estimates = {
        "delta_timely_recall": (matrix[:, 1].sum() - matrix[:, 2].sum())
        / matrix[:, 0].sum(),
        "delta_messages_per_30_field_days": 30
        * (matrix[:, 4].sum() - matrix[:, 5].sum())
        / matrix[:, 3].sum(),
        "delta_active_alarm_fraction": (matrix[:, 6].sum() - matrix[:, 7].sum())
        / matrix[:, 3].sum(),
    }
    result: dict[str, Any] = {
        "candidate": candidate_code,
        "baseline": baseline_code,
        "method": "paired_year_block_bootstrap_six_retrospective_years",
        "draws": draws,
        "seed": seed,
    }
    for name, values in distributions.items():
        result[f"{name}_estimate"] = float(estimates[name])
        result[f"{name}_p025"] = float(np.quantile(values, 0.025))
        result[f"{name}_p975"] = float(np.quantile(values, 0.975))
    return result


def _timing_diagnostics(seasons: pd.DataFrame) -> dict[str, Any]:
    # First 10-day window can end on 10 July.  Six additional persistence
    # days yield the earliest positive cutoff on 16 July; the unchanged t-2
    # rule makes 18 July the earliest possible issue date.
    window_days = 10
    persistence_days = 6
    registry = seasons[
        seasons["season"].isin(TEST_YEARS) & seasons["warnable_first_event"]
    ].copy()
    registry["event_date"] = pd.to_datetime(registry["first_recorded_event_date"])
    registry["activation_date"] = _activation_dates(registry["season"])
    registry["earliest_positive_cutoff"] = (
        registry["activation_date"]
        + pd.Timedelta(days=window_days - 1 + persistence_days)
    )
    registry["earliest_issue_date"] = registry["earliest_positive_cutoff"] + pd.Timedelta(
        days=2
    )
    registry["earliest_warnable_event_date"] = registry["earliest_issue_date"] + pd.Timedelta(
        days=3
    )
    return {
        "fixed_activation_month_day": "07-01",
        "earliest_positive_cutoff_month_day": "07-16",
        "earliest_issue_month_day": "07-18",
        "earliest_event_date_with_minimum_three_day_lead_month_day": "07-21",
        "warnable_events": int(len(registry)),
        "events_before_earliest_theoretical_warnable_date": int(
            (registry["event_date"] < registry["earliest_warnable_event_date"]).sum()
        ),
        "events_before_fixed_activation_date": int(
            (registry["event_date"] < registry["activation_date"]).sum()
        ),
        "note": "theoretical lower bound assumes every weather criterion passes immediately",
    }


def run_experiment(
    *,
    run_dir: Path = DEFAULT_RUN,
    v3_dir: Path = DEFAULT_V3,
    frozen_era_path: Path = DEFAULT_FROZEN_ERA,
    extended_era_path: Path = DEFAULT_EXTENDED_ERA,
    era_backfill_run: Path = DEFAULT_ERA_BACKFILL_RUN,
    snapshot_late_blight: Path = DEFAULT_SNAPSHOT_LATE_BLIGHT,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    v3_dir = v3_dir.resolve()
    frozen_era_path = frozen_era_path.resolve()
    extended_era_path = extended_era_path.resolve()
    era_backfill_run = era_backfill_run.resolve()
    snapshot_late_blight = snapshot_late_blight.resolve()
    if (run_dir / "execution_manifest.json").is_file():
        return verify_run(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not (
        run_dir / ".polyakov_fixed_run.json"
    ).is_file():
        raise FileExistsError(f"Refusing non-empty unrecognised run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / ".polyakov_fixed_run.json"
    if not marker.exists():
        _atomic_json(
            marker,
            {"format": RUN_FORMAT, "created_at_utc": _utc_now(), "private_local_only": True},
        )

    parent_v3 = verify_v3(v3_dir, EXPECTED_V3_MANIFEST_SHA256)
    parent_backfill = verify_era_backfill(era_backfill_run)
    input_paths = [
        v3_dir / "execution_manifest.json",
        v3_dir / "daily_decisions.parquet",
        v3_dir / "field_seasons.parquet",
        v3_dir / "alarm_states.parquet",
        frozen_era_path,
        extended_era_path,
        era_backfill_run / "execution_manifest.json",
        snapshot_late_blight,
    ]
    input_hashes = {str(path): sha256_file(path) for path in input_paths}

    decisions = pd.read_parquet(v3_dir / "daily_decisions.parquet")
    seasons = pd.read_parquet(v3_dir / "field_seasons.parquet")
    saved_states = pd.read_parquet(v3_dir / "alarm_states.parquet")
    frozen_frame, frozen_audit = build_fixed_polyakov_scores(
        decisions,
        pd.read_parquet(frozen_era_path),
        snapshot_late_blight,
        source_label="frozen_era5_v3",
    )
    extended_frame, extended_audit = build_fixed_polyakov_scores(
        decisions,
        pd.read_parquet(extended_era_path),
        snapshot_late_blight,
        source_label="era5_backfill_20260915",
    )
    key_columns = ["field_season", "season", "issue_date"]
    pd.testing.assert_frame_equal(
        frozen_frame[key_columns], extended_frame[key_columns], check_dtype=False
    )

    observed_states = _replay(decisions, "polyakov_score", OBSERVED_CODE)
    _assert_saved_identity(observed_states, saved_states, "polyakov")
    frozen_states = _replay(frozen_frame, "polyakov_score_fixed", FROZEN_CODE)
    extended_states = _replay(extended_frame, "polyakov_score_fixed", EXTENDED_CODE)
    calendar_states = saved_states[
        saved_states["model_code"].eq(CALENDAR_CODE)
        & saved_states["evaluation_scope"].eq("service_calendar")
        & saved_states["season"].isin(TEST_YEARS)
    ].copy()

    metrics: list[dict[str, Any]] = []
    hit_tables: list[pd.DataFrame] = []
    main_hits: dict[str, pd.DataFrame] = {}
    periods: list[tuple[str, tuple[int, ...]]] = [
        *((str(year), (year,)) for year in TEST_YEARS),
        ("2020_2025", TEST_YEARS),
    ]
    state_map = {
        OBSERVED_CODE: observed_states,
        FROZEN_CODE: frozen_states,
        EXTENDED_CODE: extended_states,
        CALENDAR_CODE: calendar_states,
    }
    for code, states in state_map.items():
        for period, years in periods:
            metric, hits = _summary(states, seasons, code, period, years)
            metrics.append(metric)
            hits["period"] = period
            hit_tables.append(hits)
            if period == "2020_2025":
                main_hits[code] = hits
    metrics_frame = pd.DataFrame(metrics)
    hits_frame = pd.concat(hit_tables, ignore_index=True)
    main = metrics_frame[metrics_frame["period"].eq("2020_2025")].set_index(
        "model_code"
    )
    if int(main.loc[EXTENDED_CODE, "events_with_warning_opportunity"]) != 87:
        raise AssertionError("Warnable event denominator changed")
    if int(main.loc[EXTENDED_CODE, "field_days"]) != 6924:
        raise AssertionError("Service field-day denominator changed")

    comparisons: list[dict[str, Any]] = []
    bootstrap: list[dict[str, Any]] = []
    for baseline_code in (OBSERVED_CODE, FROZEN_CODE, CALENDAR_CODE):
        categories = _paired_categories(
            main_hits[EXTENDED_CODE], main_hits[baseline_code]
        )
        candidate = main.loc[EXTENDED_CODE]
        baseline = main.loc[baseline_code]
        comparisons.append(
            {
                "candidate": EXTENDED_CODE,
                "baseline": baseline_code,
                **categories,
                "delta_timely_hits": int(
                    candidate["timely_hits"] - baseline["timely_hits"]
                ),
                "delta_messages": int(candidate["messages"] - baseline["messages"]),
                "delta_active_alarm_days": int(
                    candidate["active_alarm_days"] - baseline["active_alarm_days"]
                ),
                "delta_computable_days": int(
                    candidate["computable_days"] - baseline["computable_days"]
                ),
            }
        )
        bootstrap.append(
            _bootstrap(
                extended_states,
                state_map[baseline_code],
                main_hits[EXTENDED_CODE],
                main_hits[baseline_code],
                candidate_code=EXTENDED_CODE,
                baseline_code=baseline_code,
            )
        )
    comparisons_frame = pd.DataFrame(comparisons)
    bootstrap_frame = pd.DataFrame(bootstrap)

    predictions = extended_frame[
        extended_frame["season"].isin(TEST_YEARS)
        & extended_frame["evaluation_field_day"].astype(bool)
    ][
        [
            "field_season",
            "season",
            "issue_date",
            "feature_cutoff_era_episode_date",
            "polyakov_activation_date_fixed",
            "polyakov_activation_policy",
            "polyakov_score_fixed",
            "polyakov_status_fixed",
            "polyakov_weather_status",
            "polyakov_weather_evaluable_fixed",
            "polyakov_t10_c",
            "polyakov_rh10_pct",
            "polyakov_p10_mm",
            "target_class",
            "target_observable",
            "days_to_first_recorded_event",
        ]
    ].copy().reset_index(drop=True)
    frozen_scores = frozen_frame.loc[
        frozen_frame["season"].isin(TEST_YEARS)
        & frozen_frame["evaluation_field_day"].astype(bool),
        "polyakov_score_fixed",
    ].reset_index(drop=True)
    predictions["polyakov_score_fixed_frozen_era"] = frozen_scores
    predictions.rename(
        columns={"polyakov_score_fixed": "polyakov_score_fixed_extended_era"},
        inplace=True,
    )
    _write_parquet_idempotent(run_dir / "predictions.parquet", predictions)
    _write_parquet_idempotent(
        run_dir / "alarm_states.parquet",
        pd.concat([frozen_states, extended_states], ignore_index=True),
    )
    _write_parquet_idempotent(run_dir / "event_hits.parquet", hits_frame)
    _atomic_csv(run_dir / "metrics.csv", metrics_frame)
    _atomic_csv(run_dir / "comparisons.csv", comparisons_frame)
    _atomic_csv(run_dir / "paired_year_bootstrap.csv", bootstrap_frame)
    timing = _timing_diagnostics(seasons)
    _atomic_json(run_dir / "timing_diagnostics.json", timing)
    _atomic_json(
        run_dir / "score_audit.json",
        {"frozen_weather": frozen_audit, "extended_weather": extended_audit},
    )

    report = _report_text(
        metrics_frame,
        comparisons_frame,
        bootstrap_frame,
        timing,
        frozen_audit,
        extended_audit,
    )
    _atomic_text(run_dir / "report_ru.md", report)
    _atomic_text(
        run_dir / "REPRODUCE.md",
        "# Воспроизведение\n\n"
        "Из корня репозитория:\n\n"
        "```bash\n"
        ".venv/bin/python -m agro_phenology.polyakov_fixed_date_experiment all \\\n"
        f"  --run-dir {run_dir.relative_to(REPO_ROOT)}\n"
        "```\n\n"
        "Команда не переобучает модели и не обращается к сети. Завершённый run "
        "проверяется по хешам без перезаписи. Артефакты с погодными ячейками и "
        "полевыми псевдонимами предназначены только для локального использования.\n",
    )
    source_snapshot = _source_snapshot(run_dir, snapshot_late_blight)

    for path, expected in input_hashes.items():
        if sha256_file(path) != expected:
            raise AssertionError(f"Frozen input changed during experiment: {path}")
    outputs = [
        "predictions.parquet",
        "alarm_states.parquet",
        "event_hits.parquet",
        "metrics.csv",
        "comparisons.csv",
        "paired_year_bootstrap.csv",
        "timing_diagnostics.json",
        "score_audit.json",
        "report_ru.md",
        "REPRODUCE.md",
        source_snapshot.name,
    ]
    manifest = {
        "format": RUN_FORMAT,
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "research_assumption": {
            "bbch51_date": "07-01 of each field season",
            "known_in_advance": True,
            "not_a_validated_biological_norm": True,
        },
        "outcome_window_days": [3, 10],
        "policy": {
            "threshold": POLICY.threshold,
            "active_days": POLICY.active_days,
            "cooldown_days": POLICY.cooldown_days,
        },
        "parent_v3_verification": parent_v3,
        "parent_era_backfill_status": parent_backfill["status"],
        "input_hashes": input_hashes,
        "score_audits": {
            "frozen_weather": frozen_audit,
            "extended_weather": extended_audit,
        },
        "identity_replay": "observed_polyakov_v3_reproduced_exactly",
        "output_hashes": {
            relative: sha256_file(run_dir / relative) for relative in outputs
        },
        "privacy": "local_only; do_not_publish_row_level_or_weather_cell_artifacts",
    }
    _atomic_json(run_dir / "execution_manifest.json", manifest)
    return manifest


def _source_snapshot(run_dir: Path, snapshot_late_blight: Path) -> Path:
    sources = [
        REPO_ROOT / "src/agro_phenology/polyakov_fixed_date_experiment.py",
        REPO_ROOT / "src/agro_phenology/era_hutton_backfill.py",
        REPO_ROOT / "src/agro_phenology/early_warning_core.py",
        REPO_ROOT / "src/agro_phenology/early_warning_models.py",
        REPO_ROOT / "tests/test_polyakov_fixed_date_experiment.py",
        snapshot_late_blight,
    ]
    payload = {
        "format": "polyakov_fixed_july1_source_snapshot_v1",
        "created_at_utc": _utc_now(),
        "files": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
                "utf8_content": path.read_text(encoding="utf-8"),
            }
            for path in sources
        ],
    }
    path = run_dir / "source_snapshot.json"
    _atomic_json(path, payload)
    return path


def _report_text(
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    bootstrap: pd.DataFrame,
    timing: Mapping[str, Any],
    frozen_audit: Mapping[str, Any],
    extended_audit: Mapping[str, Any],
) -> str:
    main = metrics[metrics["period"].eq("2020_2025")].set_index("model_code")
    observed = main.loc[OBSERVED_CODE]
    frozen = main.loc[FROZEN_CODE]
    extended = main.loc[EXTENDED_CODE]
    calendar = main.loc[CALENDAR_CODE]
    vs_observed = comparisons[comparisons["baseline"].eq(OBSERVED_CODE)].iloc[0]
    vs_calendar = comparisons[comparisons["baseline"].eq(CALENDAR_CODE)].iloc[0]
    boot_calendar = bootstrap[bootstrap["baseline"].eq(CALENDAR_CODE)].iloc[0]
    return f"""# Поляков с фиксированной датой бутонизации 1 июля

Дата запуска: 2026-09-15. Сценарий является проверкой чувствительности. Дата 1 июля задана пользователем одинаково для каждого картофельного поле-сезона и считается известной заранее; это не доказанный биологический норматив и не восстановленная фактическая фенофаза.

## Реализация

До 1 июля фазовый шлюз детерминированно закрыт, поэтому score равен нулю. С 1 по 9 июля score также равен нулю: десятидневное погодное окно после активации ещё не может существовать. Начиная с 10 июля отсутствие требуемой погоды приводит к воздержанию; при полной погоде применена без изменений текущая формализация Полякова: средняя температура за 10 дней 13–20 °C, средняя влажность не ниже 75%, сумма осадков не ниже 20 мм и шесть дней сохранения критического состояния. Положительными считаются состояния `OUTBREAK_EXPECTED` и `PROLONGED_RISK`.

Погода по-прежнему заканчивается на `issue_date−2`. Окно успеха 3–10 дней, порог 0,5, тревога 7 дней и cooldown 15 дней не менялись. Выполнен полный последовательный replay, а не вставка отдельных сообщений в старый журнал.

## Результаты 2020–2025

| Сценарий | Своевременные события | Сообщения | Сообщений / 30 дней | Тревожные дни | Доля тревожных дней | Доступный policy-score |
|---|---:|---:|---:|---:|---:|---:|
| Поляков с наблюдаемым BBCH51, v3 | {int(observed.timely_hits)}/{int(observed.events_with_warning_opportunity)} | {int(observed.messages)} | {observed.messages_per_30_field_days:.3f} | {int(observed.active_alarm_days)} | {100*observed.active_alarm_fraction:.1f}% | {int(observed.computable_days)}/{int(observed.field_days)} |
| 1 июля, исходный frozen ERA5 | {int(frozen.timely_hits)}/{int(frozen.events_with_warning_opportunity)} | {int(frozen.messages)} | {frozen.messages_per_30_field_days:.3f} | {int(frozen.active_alarm_days)} | {100*frozen.active_alarm_fraction:.1f}% | {int(frozen.computable_days)}/{int(frozen.field_days)} |
| 1 июля, дополненный ERA5 | {int(extended.timely_hits)}/{int(extended.events_with_warning_opportunity)} | {int(extended.messages)} | {extended.messages_per_30_field_days:.3f} | {int(extended.active_alarm_days)} | {100*extended.active_alarm_fraction:.1f}% | {int(extended.computable_days)}/{int(extended.field_days)} |
| Календарное окно v3 | {int(calendar.timely_hits)}/{int(calendar.events_with_warning_opportunity)} | {int(calendar.messages)} | {calendar.messages_per_30_field_days:.3f} | {int(calendar.active_alarm_days)} | {100*calendar.active_alarm_fraction:.1f}% | {int(calendar.computable_days)}/{int(calendar.field_days)} |

Фиксированная дата увеличила своевременное покрытие относительно исходного Полякова с {int(observed.timely_hits)} до {int(extended.timely_hits)} событий. Парно приобретено {int(vs_observed.candidate_only)} событий, потеряно {int(vs_observed.baseline_only)}. Заполнение погодных хвостов не добавило событий: вариант с frozen и дополненным ERA5 имеет одинаковые {int(extended.timely_hits)}/87, но дополненный вариант увеличил нагрузку с {int(frozen.messages)} до {int(extended.messages)} сообщений и с {int(frozen.active_alarm_days)} до {int(extended.active_alarm_days)} тревожных дней.

Policy-score доступен на всех {int(extended.computable_days)} сервисных днях, поскольку до возможности построить погодное окно фазовый шлюз детерминированно возвращает ноль. Это не 100% вычислимость погодного критерия: {extended_audit['structural_inactive_or_warmup_service_days']} ранних дней являются структурно неактивными, а после разогрева погода реально вычислима на {extended_audit['weather_evaluable_service_days_after_warmup']}/5 721 днях. В окне событий погодная часть вычислима для {extended_audit['weather_evaluable_warnable_events']}/87 событий. На исходном frozen ERA5 после разогрева были вычислимы только {frozen_audit['weather_evaluable_service_days_after_warmup']} дней.

Календарь остаётся сильнее: {int(calendar.timely_hits)}/87 против {int(extended.timely_hits)}/87. Только Поляков своевременно поймал {int(vs_calendar.candidate_only)} событий, только календарь — {int(vs_calendar.baseline_only)}, оба — {int(vs_calendar.both)}, никто — {int(vs_calendar.neither)}. Разница timely recall Полякова против календаря равна {boot_calendar.delta_timely_recall_estimate:+.3f}; годовой парный bootstrap-интервал [{boot_calendar.delta_timely_recall_p025:+.3f}; {boot_calendar.delta_timely_recall_p975:+.3f}].

## Ограничение фиксированной даты

Даже при немедленном выполнении всех погодных условий первый положительный cutoff возможен только 16 июля, а сообщение с двухдневным погодным лагом — 18 июля. Поэтому событие раньше 21 июля нельзя предупредить с минимальным lead 3 дня. Таких событий в основной популяции {timing['events_before_earliest_theoretical_warnable_date']} из {timing['warnable_events']}.

## Вывод

Подстановка 1 июля решает большую часть проблемы отсутствующей фенофазы и заметно усиливает Полякова относительно его текущей реализации. Она не делает его лучше календаря: своевременно покрывается менее четверти событий, тогда как календарь покрывает более половины. Дополненная погода повышает вычислимость до 100%, но добавляет только нагрузку, а не своевременные события. Результат ретроспективный на уже изученных годах.
"""


def verify_run(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "execution_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != RUN_FORMAT or manifest.get("status") != "complete":
        raise AssertionError("Not a completed fixed-July Polyakov run")
    failures = []
    for relative, expected in manifest["output_hashes"].items():
        path = run_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            failures.append(relative)
    if failures:
        raise AssertionError(f"Polyakov run output hash failures: {failures}")
    for source, expected in manifest["input_hashes"].items():
        if not Path(source).is_file() or sha256_file(source) != expected:
            raise AssertionError(f"Polyakov run input changed: {source}")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("all")
    run.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    run.add_argument("--v3-dir", type=Path, default=DEFAULT_V3)
    run.add_argument("--frozen-era", type=Path, default=DEFAULT_FROZEN_ERA)
    run.add_argument("--extended-era", type=Path, default=DEFAULT_EXTENDED_ERA)
    run.add_argument("--era-backfill-run", type=Path, default=DEFAULT_ERA_BACKFILL_RUN)
    run.add_argument("--snapshot-late-blight", type=Path, default=DEFAULT_SNAPSHOT_LATE_BLIGHT)
    check = subparsers.add_parser("check")
    check.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "check":
        manifest = verify_run(args.run_dir.resolve())
    else:
        manifest = run_experiment(
            run_dir=args.run_dir,
            v3_dir=args.v3_dir,
            frozen_era_path=args.frozen_era,
            extended_era_path=args.extended_era,
            era_backfill_run=args.era_backfill_run,
            snapshot_late_blight=args.snapshot_late_blight,
        )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_dir": str(args.run_dir.resolve()),
                "assumption": manifest["research_assumption"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
