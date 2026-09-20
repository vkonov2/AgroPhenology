from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import agro_phenology.early_warning_cycle3_pipeline as cycle3
from agro_phenology.early_warning_core import sha256_file
from agro_phenology.early_warning_cycle3_pipeline import (
    _candidate_metrics,
    _simulate_runtime,
    build_frozen_scores,
    build_gained_lost,
    run_policy_experiments,
    select_policy_record,
    threshold_grid,
    verify_frozen_run,
)
from agro_phenology.early_warning_cycle3_policy import GrowthPolicy, simulate_growth_policy


def _policy_record(name: str, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "name": name,
        "feasible": True,
        "constraint_violation": 0.0,
        "timely_recall": 0.5,
        "messages_per_30_field_days": 1.5,
        "active_alarm_fraction": 0.3,
        "growth_override_enabled": False,
        "growth_logit_delta": np.nan,
        "threshold": 0.5,
    }
    record.update(overrides)
    return record


def test_threshold_grid_contains_frozen_grid_quantiles_and_saved_threshold() -> None:
    score = pd.Series([np.nan, 0.02, 0.41, 0.83, np.nan])
    saved_threshold = 0.333333333333

    actual = threshold_grid(score, saved_threshold)
    fixed = np.linspace(0.05, 0.95, 19)
    quantiles = score.dropna().quantile(np.linspace(0.05, 0.95, 19)).unique()

    assert actual == sorted(set(actual))
    assert not any(pd.isna(value) for value in actual)
    assert saved_threshold in actual
    assert 1.000001 in actual
    for expected in np.concatenate([fixed, quantiles]):
        assert any(value == pytest.approx(float(expected)) for value in actual)


@pytest.mark.parametrize(
    ("left", "right", "winner"),
    [
        (
            _policy_record("feasible", feasible=True, timely_recall=0.1),
            _policy_record(
                "infeasible",
                feasible=False,
                constraint_violation=0.01,
                timely_recall=1.0,
            ),
            "feasible",
        ),
        (
            _policy_record("higher-recall", timely_recall=0.6),
            _policy_record("lower-recall", timely_recall=0.5),
            "higher-recall",
        ),
        (
            _policy_record("fewer-messages", messages_per_30_field_days=1.4),
            _policy_record("more-messages", messages_per_30_field_days=1.5),
            "fewer-messages",
        ),
        (
            _policy_record("fewer-alarm-days", active_alarm_fraction=0.2),
            _policy_record("more-alarm-days", active_alarm_fraction=0.3),
            "fewer-alarm-days",
        ),
        (
            _policy_record("disabled", growth_override_enabled=False),
            _policy_record(
                "enabled",
                growth_override_enabled=True,
                growth_logit_delta=np.log(3.0),
            ),
            "disabled",
        ),
        (
            _policy_record(
                "larger-delta",
                growth_override_enabled=True,
                growth_logit_delta=np.log(3.0),
            ),
            _policy_record(
                "smaller-delta",
                growth_override_enabled=True,
                growth_logit_delta=np.log(1.5),
            ),
            "larger-delta",
        ),
        (
            _policy_record("higher-threshold", threshold=0.7),
            _policy_record("lower-threshold", threshold=0.5),
            "higher-threshold",
        ),
    ],
)
def test_policy_selection_uses_frozen_tie_break_independent_of_input_order(
    left: dict[str, object], right: dict[str, object], winner: str
) -> None:
    for ordered in ((left, right), (right, left)):
        selected = select_policy_record(ordered)
        assert selected["name"] == winner
        assert selected["selection_status"] == "feasible"


def test_no_feasible_policy_uses_minimum_violation_then_normal_tie_break() -> None:
    records = [
        _policy_record(
            "high-recall-large-violation",
            feasible=False,
            constraint_violation=0.2,
            timely_recall=1.0,
        ),
        _policy_record(
            "minimum-violation-lower-recall",
            feasible=False,
            constraint_violation=0.1,
            timely_recall=0.4,
        ),
        _policy_record(
            "minimum-violation-higher-recall",
            feasible=False,
            constraint_violation=0.1,
            timely_recall=0.6,
        ),
    ]

    for ordered in permutations(records):
        selected = select_policy_record(ordered)
        assert selected["name"] == "minimum-violation-higher-recall"
        assert selected["selection_status"] == "no_feasible_policy"


def _write_temp_manifest(run_dir: Path, *, status: str = "complete") -> str:
    artifact = run_dir / "artifact.bin"
    artifact.write_bytes(b"frozen parent payload\n")
    manifest = {
        "run_id": "temp_parent",
        "status": status,
        "output_hashes": {
            "artifact.bin": {
                "sha256": sha256_file(artifact),
                "bytes": artifact.stat().st_size,
            }
        },
    }
    manifest_path = run_dir / "execution_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sha256_file(manifest_path)


def test_frozen_parent_verifier_checks_manifest_status_hash_and_outputs(
    tmp_path: Path,
) -> None:
    expected_manifest_hash = _write_temp_manifest(tmp_path)
    result = verify_frozen_run(tmp_path, expected_manifest_hash)
    assert result["status"] == "passed"
    assert result["run_id"] == "temp_parent"
    assert result["outputs_checked"] == 1
    assert result["all_output_hashes_match"]

    (tmp_path / "artifact.bin").write_bytes(b"changed\n")
    with pytest.raises(AssertionError, match="output hash failures"):
        verify_frozen_run(tmp_path, expected_manifest_hash)

    incomplete_dir = tmp_path / "incomplete"
    incomplete_dir.mkdir()
    incomplete_hash = _write_temp_manifest(incomplete_dir, status="running")
    with pytest.raises(AssertionError, match="not complete"):
        verify_frozen_run(incomplete_dir, incomplete_hash)

    with pytest.raises(AssertionError, match="Unexpected parent manifest hash"):
        verify_frozen_run(incomplete_dir, "0" * 64)


def _synthetic_population() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fields = [f"field-{index:03d}" for index in range(127)]
    eligible = np.arange(127) < 87
    candidate_hits = np.zeros(127, dtype=bool)
    baseline_hits = np.zeros(127, dtype=bool)
    candidate_hits[[0, 1]] = True
    baseline_hits[[0, 2]] = True
    common = {
        "season": 2020,
        "fold_id": "test_2020",
        "evaluation_scope": "service_calendar",
        "slice": "A_plus_B",
        "score_model": "C0",
    }
    baseline = pd.DataFrame(
        {
            **common,
            "field_season": fields,
            "policy_family": "P0_saved",
            "model_code": "C0__P0_saved",
            "warnable_event": eligible,
            "timely_hit": baseline_hits,
        }
    )
    candidate = pd.DataFrame(
        {
            **common,
            "field_season": fields,
            "policy_family": "P_growth_selected",
            "model_code": "C0__P_growth_selected",
            "warnable_event": eligible,
            "timely_hit": candidate_hits,
        }
    )
    seasons = pd.DataFrame(
        {
            "field_season": fields,
            "season": 2020,
            "first_recorded_event_date": pd.Timestamp("2020-07-20"),
            "warnable_first_event": eligible,
            "positive_at_first_visit": ~eligible,
        }
    )
    states = pd.DataFrame(
        {
            "field_season": pd.Series(dtype=str),
            "season": pd.Series(dtype=int),
            "issue_date": pd.Series(dtype="datetime64[ns]"),
            "model_code": pd.Series(dtype=str),
            "evaluation_scope": pd.Series(dtype=str),
            "message_issued": pd.Series(dtype=bool),
            "action_reason": pd.Series(dtype=str),
            "suppressed_repeat": pd.Series(dtype=bool),
        }
    )
    return pd.concat([baseline, candidate], ignore_index=True), states, seasons


def test_gained_lost_uses_only_87_eligible_events_and_preserves_invariants() -> None:
    hits, states, seasons = _synthetic_population()
    detail, summary, population = build_gained_lost(hits, states, seasons)

    assert len(population) == 127
    assert int(population["eligible_for_warning"].sum()) == 87
    assert int(population["positive_at_first_visit"].sum()) == 40
    assert len(detail) == 87
    entry_keys = set(population.loc[~population["eligible_for_warning"], "event_key"])
    assert entry_keys.isdisjoint(detail["event_key"])

    row = summary.iloc[0]
    assert row["eligible_events"] == 87
    assert (row["both"], row["gained"], row["lost"], row["neither"]) == (
        1,
        1,
        1,
        84,
    )
    assert row["candidate_timely_events"] == 2
    assert row["baseline_timely_events"] == 2
    assert row["net_gain"] == 0
    assert bool(row["all_invariants_pass"])


def test_candidate_metrics_counts_actual_growth_override_column() -> None:
    dates = pd.date_range("2024-06-01", periods=16)
    event_date = pd.Timestamp("2024-06-15")
    frame = pd.DataFrame(
        {
            "field_season": "field-growth",
            "season": 2024,
            "issue_date": dates,
            "issued_at": dates + pd.Timedelta(hours=8),
            "service_active": True,
            "evaluation_field_day": True,
            "target_class": "unknown",
            "target_observable": False,
            "days_to_first_recorded_event": (event_date - dates).days,
            "warnable_first_event": True,
            "coordinate_scope": "A_direct",
            "previous_visit_gap_days": 7.0,
            "common_weather_complete": True,
        }
    )
    score = pd.Series(0.01, index=frame.index)
    score.iloc[0] = 0.2
    score.iloc[7] = 0.8
    states, _ = simulate_growth_policy(
        frame,
        score,
        GrowthPolicy(
            threshold=0.1,
            active_days=7,
            cooldown_days=15,
            minimum_repeat_interval_days=7,
            growth_override_enabled=True,
            growth_logit_delta=float(np.log(1.5)),
        ),
        score_origin="C0",
        model_id="C0",
        model_version="frozen",
    )
    seasons = pd.DataFrame(
        {
            "field_season": ["field-growth"],
            "season": [2024],
            "first_recorded_event_date": [event_date],
            "warnable_first_event": [True],
            "positive_at_first_visit": [False],
        }
    )
    contract = {
        "notification_policy": {
            "research_budget": {
                "messages_per_30_field_days_max": 10.0,
                "active_alarm_fraction_max": 1.0,
            }
        }
    }

    metrics, _ = _candidate_metrics(
        states,
        seasons,
        public_code="C0__P_growth_selected",
        fold_id="test_2024",
        scope="service_calendar",
        contract=contract,
    )
    assert int(states["growth_override_used"].sum()) == 1
    assert metrics["growth_messages"] == 1


def test_alpha_zero_c6_freeze_uses_c0_origin_without_fallback_and_keeps_missing_correction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    decisions = pd.DataFrame(
        {
            "field_season": ["field-validation", "field-test"],
            "season": [2020, 2021],
            "issue_date": pd.to_datetime(["2020-06-01", "2021-06-01"]),
            "issued_at": pd.to_datetime(["2020-06-01 08:00", "2021-06-01 08:00"]),
            "service_active": [True, True],
            "evaluation_field_day": [True, True],
            "candidate_comparison_complete": [True, True],
            "target_class": ["unknown", "unknown"],
            "target_observable": [False, False],
            "days_to_first_recorded_event": [np.nan, np.nan],
            "warnable_first_event": [False, False],
            "coordinate_scope": ["A_direct", "A_direct"],
            "previous_visit_gap_days": [7.0, 7.0],
            "common_weather_complete": [False, False],
        }
    )
    seasons = pd.DataFrame(
        {
            "field_season": ["field-validation", "field-test"],
            "season": [2020, 2021],
            "first_recorded_event_date": [pd.NaT, pd.NaT],
            "warnable_first_event": [False, False],
            "positive_at_first_visit": [False, False],
            "coordinate_scope": ["A_direct", "A_direct"],
            "previous_visit_gap_days": [7.0, 7.0],
        }
    )
    cycle1_models = ("calendar_window", "C0", "C1", "C4", "C5")
    c6_models = ("C6_weather", "C6_calibration_control", "C6_calendar_control")
    scopes = tuple(cycle3.SCOPES)
    v3_policy = pd.DataFrame(
        [
            {
                "fold_id": "test_2021",
                "model_code": model,
                "evaluation_scope": scope,
                "threshold": 0.5,
                "active_days": 7,
                "cooldown_days": 15,
            }
            for model in cycle1_models
            for scope in scopes
        ]
    )
    v4_policy = pd.DataFrame(
        [
            {
                "fold_id": "test_2021",
                "model_code": model,
                "evaluation_scope": scope,
                "policy_mode": "validation_selected",
                "threshold": 0.5,
                "active_days": 7,
                "cooldown_days": 15,
                "alpha": 0.0,
            }
            for model in c6_models
            for scope in scopes
        ]
    )
    saved_score_rows = {
        "field_season": "field-test",
        "season": 2021,
        "issue_date": pd.Timestamp("2021-06-01"),
        "score": 0.4,
        "fold_id": "test_2021",
    }
    v3_predictions = pd.DataFrame(
        [
            {**saved_score_rows, "model_code": model, "evaluation_scope": scope}
            for model in cycle1_models
            for scope in scopes
        ]
    )
    v4_states = pd.DataFrame(
        [
            {
                **saved_score_rows,
                "model_family": model,
                "policy_mode": "validation_selected",
                "evaluation_scope": scope,
            }
            for model in c6_models
            for scope in scopes
        ]
    )
    inputs = {
        "v3_daily_decisions": decisions,
        "v3_field_seasons": seasons,
        "v3_predictions": v3_predictions,
        "v4_c6_raw_predictions": pd.DataFrame(),
        "v4_alarm_states": v4_states,
        "v3_policy": v3_policy,
        "v4_policy": v4_policy,
        "v3_contract": {
            "rolling_origin_folds": [
                {
                    "id": "test_2021",
                    "train_years": [2019, 2019],
                    "validation_years": [2020, 2020],
                    "test_years": [2021, 2021],
                }
            ]
        },
    }

    monkeypatch.setattr(cycle3, "_load_cycle1_model", lambda *_: object())
    monkeypatch.setattr(
        cycle3,
        "score_model",
        lambda _model, frame, _features, _availability: pd.Series(0.4, index=frame.index),
    )
    monkeypatch.setattr(
        cycle3,
        "_baseline_scores",
        lambda frame, _seasons: {"calendar_window": pd.Series(0.4, index=frame.index)},
    )

    def unavailable_correction(
        _raw: pd.DataFrame,
        frame: pd.DataFrame,
        **_kwargs: object,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        return (
            pd.Series(0.4, index=frame.index),
            pd.Series(False, index=frame.index),
            pd.Series(True, index=frame.index),
        )

    monkeypatch.setattr(cycle3, "_c6_score_from_raw", unavailable_correction)

    frozen, _, _, runtime, _ = build_frozen_scores(inputs, tmp_path)
    alpha_zero = frozen[frozen["score_model"].isin(c6_models)]
    assert alpha_zero["score_origin"].eq("C0").all()
    assert not alpha_zero["fallback_to_c0"].any()
    assert not alpha_zero["correction_available"].any()

    validation_runtime = runtime[
        ("test_2021", "C6_weather", "service_calendar", "outer_validation")
    ]
    states = _simulate_runtime(
        validation_runtime,
        GrowthPolicy(threshold=0.5, growth_override_enabled=False),
        score_model="C6_weather",
        scope="service_calendar",
    )
    metrics, _ = _candidate_metrics(
        states,
        seasons[seasons["season"].eq(2020)],
        public_code="C6_weather__P0_saved",
        fold_id="test_2021",
        scope="service_calendar",
        contract={
            "notification_policy": {
                "research_budget": {
                    "messages_per_30_field_days_max": 2.0,
                    "active_alarm_fraction_max": 0.5,
                }
            }
        },
    )
    assert metrics["fallback_days"] == 0
    assert metrics["correction_unavailable_days"] == 1
    assert metrics["effective_c0_origin_days"] == 1


def test_policy_threshold_population_is_service_active_intersected_with_current_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = pd.DataFrame(
        {
            "field_season": ["field"] * 4,
            "season": [2020] * 4,
            "issue_date": pd.date_range("2020-06-01", periods=4),
            "issued_at": pd.date_range("2020-06-01 08:00", periods=4),
            "service_active": [True, False, True, True],
            "candidate_comparison_complete": [True, True, False, True],
        }
    )
    score = pd.Series([0.11, 0.22, 0.33, 0.44], index=source.index)
    runtime: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for scope in cycle3.SCOPES:
        for split in ("outer_validation", "outer_test"):
            runtime[("test_2021", "C0", scope, split)] = {
                "source": source,
                "score": score,
                "origin": pd.Series("C0", index=source.index),
                "correction_available": pd.Series(True, index=source.index),
                "fallback": pd.Series(False, index=source.index),
                "alpha": None,
                "model_version": "test",
            }
    saved_policies = {
        ("test_2021", "C0", scope): {
            "threshold": 0.5,
            "active_days": 7,
            "cooldown_days": 15,
        }
        for scope in cycle3.SCOPES
    }
    captured: list[pd.Series] = []

    def capture_threshold_grid(values: pd.Series, _saved: float) -> list[float]:
        captured.append(values.copy())
        return [0.5]

    def fake_states(
        current: dict[str, object], _policy: GrowthPolicy, *, score_model: str, scope: str
    ) -> pd.DataFrame:
        frame = current["source"].copy()
        evaluation = frame["service_active"].astype(bool)
        if cycle3.SCOPES[scope]:
            evaluation &= frame[cycle3.SCOPES[scope]].astype(bool)
        frame["evaluation_scope_day"] = evaluation
        frame["score"] = current["score"]
        frame["score_origin"] = "C0"
        frame["message_issued"] = False
        frame["alarm_active"] = False
        frame["suppressed_repeat"] = False
        frame["growth_override_used"] = False
        frame["action_reason"] = "below_threshold"
        frame["fallback_to_c0"] = False
        frame["correction_available"] = True
        frame["effective_c0_origin"] = True
        return frame

    base_metrics = {
        "timely_recall": np.nan,
        "messages_per_30_field_days": 0.0,
        "active_alarm_fraction": 0.0,
        "constraint_violation": 0.0,
        "feasible": True,
    }
    annual_metric = {
        "model_code": "placeholder",
        "fold_id": "test_2021",
        "evaluation_scope": "placeholder",
        "slice": "A_plus_B",
        "season": 2021,
        "first_events": 0,
        "events_with_warning_opportunity": 0,
        "timely_hits": 0,
        "timely_recall": np.nan,
        "field_days": 0,
        "field_seasons": 0,
        "messages": 0,
        "messages_per_30_field_days": np.nan,
        "active_alarm_days": 0,
        "active_alarm_fraction": np.nan,
    }
    pooled_keys = [
        "period",
        "year_start",
        "year_end",
        "test_years",
        "model_code",
        "evaluation_scope",
        "slice",
    ]

    monkeypatch.setattr(cycle3, "SCORE_MODELS", ("C0",))
    monkeypatch.setattr(cycle3, "threshold_grid", capture_threshold_grid)
    monkeypatch.setattr(cycle3, "_simulate_runtime", fake_states)
    monkeypatch.setattr(cycle3, "_fast_validation_metrics", lambda *_: dict(base_metrics))
    monkeypatch.setattr(
        cycle3,
        "_candidate_metrics",
        lambda _states, _seasons, **kwargs: (
            {
                **annual_metric,
                "model_code": kwargs["public_code"],
                "evaluation_scope": kwargs["scope"],
            },
            [],
        ),
    )
    monkeypatch.setattr(
        cycle3,
        "_verify_p0_saved_identity",
        lambda *_args, **kwargs: {
            "fold_id": kwargs["fold_id"],
            "score_model": kwargs["score_model"],
            "evaluation_scope": kwargs["scope"],
            "status": "passed",
        },
    )
    monkeypatch.setattr(
        cycle3,
        "aggregate_pooled_metrics",
        lambda *_args, **_kwargs: {
            "pooled_event_metrics": pd.DataFrame(columns=pooled_keys),
            "pooled_burden_metrics": pd.DataFrame(columns=pooled_keys),
        },
    )

    run_policy_experiments(
        {
            "v3_field_seasons": pd.DataFrame(
                {
                    "field_season": ["field", "field"],
                    "season": [2020, 2021],
                }
            ),
            "v3_contract": {
                "rolling_origin_folds": [
                    {
                        "id": "test_2021",
                        "validation_years": [2020, 2020],
                        "test_years": [2021, 2021],
                    }
                ]
            },
            "v3_alarm_states": pd.DataFrame(
                columns=["fold_id", "model_code", "evaluation_scope"]
            ),
            "v4_alarm_states": pd.DataFrame(
                columns=["fold_id", "model_family", "policy_mode", "evaluation_scope"]
            ),
        },
        runtime,
        saved_policies,
        {
            "notification_policy": {
                "active_days_per_message": 7,
                "cooldown_days": 15,
                "minimum_growth_repeat_interval_days": 7,
                "logit_epsilon": 1e-6,
                "growth_deltas": [{"id": "disabled", "enabled": False}],
            }
        },
    )

    assert len(captured) == 2
    assert captured[0].tolist() == pytest.approx([0.11, np.nan, 0.33, 0.44], nan_ok=True)
    assert captured[1].tolist() == pytest.approx([0.11, np.nan, np.nan, 0.44], nan_ok=True)
