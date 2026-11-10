"""Synthetic validation/replay tests; no real field records or metrics."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agro_phenology import external_native_validation as native
from agro_phenology.early_warning_models import burden_metrics, event_metrics, simulate_policy
from agro_phenology.external_native_cohorts import TARGET_SPECS, VISIT_COLUMNS, build_decisions, build_registry
from agro_phenology.external_validation_io import digest_file


def _visits(site, rows, *, cluster):
    return pd.DataFrame([
        dict(source_id=TARGET_SPECS["cutworm"]["source_id"], target="cutworm", site_id=site,
             season=pd.Timestamp(day).year, observation_date=pd.Timestamp(day), status=status,
             cluster=cluster, station_id=cluster, coordinate_scope="synthetic",
             entry_egg_presence=np.nan, entry_adult_presence=np.nan, n_reports=1,
             missing_reports=int(status == "unknown"), all_taxa_zero=False, zero_mode="reported")
        for day, status in rows
    ])[VISIT_COLUMNS]


@pytest.fixture
def replay_case():
    visits = pd.concat([
        _visits("synthetic_warnable", [("2020-05-01", "zero"), ("2020-05-04", "zero"), ("2020-05-08", "positive")], cluster="synthetic_a"),
        _visits("synthetic_short", [("2020-05-01", "zero"), ("2020-05-03", "positive")], cluster="synthetic_b"),
        _visits("synthetic_censored", [("2020-05-01", "zero"), ("2020-05-04", "zero"), ("2020-05-10", "unknown")], cluster="synthetic_c"),
    ], ignore_index=True)
    registry = build_registry(visits)
    decisions = build_decisions(registry)
    dates_by_site = {"synthetic_warnable": pd.Timestamp("2020-05-08"), "synthetic_short": pd.Timestamp("2020-05-08"),
                     "synthetic_censored": pd.Timestamp("2020-05-20")}
    point = registry[["field_season", "site_id"]].copy()
    point["predicted_date"] = point.site_id.map(dates_by_site)
    dates = decisions.field_season.map(point.set_index("field_season").predicted_date)
    score = (dates-decisions.issue_date).dt.days.between(3, 10).astype(float).where(dates.notna())
    states = simulate_policy(decisions, score, native.POLICY, evaluation_scope="full_service_calendar", evaluation_mask=decisions.service_active)
    return registry, decisions, point, states


def _unit_views(registry, point, states, model=native.CALENDAR):
    blocks = []
    for scope in native.SCOPES:
        unit = native._unit_metrics(states, registry, point, scope)
        unit["model_code"] = model
        blocks.append(unit)
    return pd.concat(blocks, ignore_index=True)


@pytest.mark.parametrize("scope", native.SCOPES)
def test_unit_aggregation_agrees_with_existing_event_and_burden_metrics(replay_case, scope):
    registry, _, point, states = replay_case
    assert native.simulate_policy is simulate_policy
    assert (native.POLICY.threshold, native.POLICY.active_days, native.POLICY.cooldown_days) == (.5, 7, 15)
    units = native._unit_metrics(states, registry, point, scope)
    units["model_code"] = native.CALENDAR
    actual = native.aggregate_metrics(units, "cutworm", "reported").query("fold_id == 'all_outer_years'").iloc[0]
    legacy = states.copy()
    legacy["evaluation_scope_day"] = legacy.evaluation_field_day if scope == "observed_followup" else legacy.service_active
    expected_event, _ = event_metrics(legacy, registry, native.CALENDAR, "synthetic", evaluation_scope=scope)
    expected_burden = burden_metrics(legacy, native.CALENDAR, "synthetic", evaluation_scope=scope)
    assert actual.first_events == expected_event["first_events"] == 2
    assert actual.warnable_events == expected_event["events_with_warning_opportunity"] == 1
    assert actual.timely_hits == expected_event["timely_hits"] == 1
    assert actual.timely_coverage == expected_event["timely_recall"] == 1
    assert actual.computable_events == expected_event["computable_events"] == 1
    for column in ["field_days", "messages", "active_alarm_days", "computable_days", "suppressed_repeats"]:
        assert actual[column] == expected_burden[column]
    assert actual.messages_per_30_field_days == expected_burden["messages_per_30_field_days"]
    assert actual.alarm_fraction == expected_burden["active_alarm_fraction"]
    assert actual.computability == expected_burden["computable_fraction"]
    short = units[units.field_season.eq(registry.loc[registry.site_id.eq("synthetic_short"), "field_season"].iloc[0])].iloc[0]
    assert not short.warnable_first_event and not short.timely_hit
    assert short.messages == 1 and short.late_messages == 1


def test_unknown_tail_counts_all_messages_in_full_service_without_inventing_outcomes(replay_case):
    registry, decisions, point, states = replay_case
    metrics = native.aggregate_metrics(_unit_views(registry, point, states), "cutworm", "reported")
    metrics = metrics[metrics.fold_id.eq("all_outer_years")].set_index("evaluation_scope")
    observed, full = metrics.loc["observed_followup"], metrics.loc["full_service_calendar"]
    assert observed.field_days == 12  # Seven to event, two short, three censored follow-up.
    assert full.field_days == 192
    assert observed.messages == 2 and full.messages == 3
    assert observed.active_alarm_days == 9 and full.active_alarm_days == 16
    assert observed.warnable_events == full.warnable_events == 1
    assert observed.first_events == full.first_events == 2
    assert observed.timely_hits == full.timely_hits == 1
    assert not decisions.target_observable.any() and decisions.target_class.eq("unknown").all()
    expected = simulate_policy(decisions, states.score, native.POLICY, evaluation_scope="full_service_calendar", evaluation_mask=decisions.service_active)
    pd.testing.assert_frame_equal(states, expected)


def test_windows_and_actual_message_leads_are_inclusive_three_through_ten(replay_case):
    registry, _, _, states = replay_case
    sent = states[states.message_issued]
    assert (sent.forecast_window_start-sent.issue_date).dt.days.eq(3).all()
    assert (sent.forecast_window_end-sent.issue_date).dt.days.eq(10).all()
    warnable = registry[registry.warnable_first_event].iloc[0]
    actual = sent[sent.field_season.eq(warnable.field_season)].iloc[0]
    assert (warnable.first_recorded_event_date-actual.issue_date).days == 6
    assert actual.forecast_window_start <= warnable.first_recorded_event_date <= actual.forecast_window_end
    assert not states.loc[~states.service_active, "message_issued"].any()


def test_missing_forecast_date_abstains_instead_of_becoming_a_low_risk_score(replay_case):
    registry, decisions, point, _ = replay_case
    point = point.copy()
    mask = point.site_id.eq("synthetic_censored")
    point.loc[mask, "predicted_date"] = pd.NaT
    dates = decisions.field_season.map(point.set_index("field_season").predicted_date)
    score = (dates-decisions.issue_date).dt.days.between(3, 10).astype(float).where(dates.notna())
    states = simulate_policy(decisions, score, native.POLICY, evaluation_scope="full_service_calendar", evaluation_mask=decisions.service_active)
    missing = states.field_season.eq(point.loc[mask, "field_season"].iloc[0])
    assert states.loc[missing, "score"].isna().all()
    assert states.loc[missing & states.service_active, "action_reason"].eq("abstained_missing_input").all()
    assert not states.loc[missing, "message_issued"].any()
    units = _unit_views(registry, point, states)
    subset = units[units.field_season.eq(point.loc[mask, "field_season"].iloc[0])]
    assert subset.computable_days.eq(0).all() and subset.messages.eq(0).all() and subset.predicted_date.isna().all()


@pytest.mark.parametrize("column,value", [("target", "kbs_myzus"), ("source_id", "another_dataset"), ("zero_mode", "all_taxa_zero_unknown")])
def test_aggregate_refuses_pooled_datasets_targets_and_zero_interpretations(replay_case, column, value):
    registry, _, point, states = replay_case
    units = _unit_views(registry, point, states)
    units.loc[0, column] = value
    with pytest.raises(ValueError, match="pool"):
        native.aggregate_metrics(units, "cutworm", "reported")


def _paired_case(replay_case):
    registry, _, point, states = replay_case
    base = _unit_views(registry, point, states)
    aft = base.copy()
    aft["model_code"] = native.AFT
    # The candidate loses the eligible hit and its actual message/alarm state.
    mask = aft.warnable_first_event
    aft.loc[mask, "timely_hit"] = False
    aft.loc[mask, "messages"] = 0
    aft.loc[mask, "active_alarm_days"] = 0
    return pd.concat([base, aft], ignore_index=True)


def test_paired_bootstrap_aligns_identity_and_excludes_undefined_zero_event_draws(replay_case):
    units = _paired_case(replay_case)
    ci = native.paired_uncertainty(units, "cutworm")
    # Reorder the candidate independently: pairing must be by identity, not
    # row position. Keep baseline block order fixed for identical Monte Carlo
    # draws, rather than changing the seed-to-block assignment.
    shuffled = pd.concat([
        units[units.model_code.eq(native.CALENDAR)],
        units[units.model_code.eq(native.AFT)].sample(frac=1, random_state=83),
    ], ignore_index=True)
    assert native.paired_uncertainty(shuffled, "cutworm") == ci
    for scope in native.SCOPES:
        block = ci[scope]["cluster"]
        coverage = block["coverage_difference"]
        assert block["clusters"] == 3
        assert 0 < coverage["valid_replicates"] < 2000
        assert coverage["p025"] == coverage["p975"] == -1
        assert ci[scope]["season"]["clusters"] == 1
        assert ci[scope]["season"]["coverage_difference"]["valid_replicates"] == 2000


def test_bootstrap_without_any_event_does_not_manufacture_zero_coverage(replay_case):
    units = _paired_case(replay_case)
    units["warnable_first_event"], units["timely_hit"] = False, False
    ci = native.paired_uncertainty(units, "cutworm")
    for scope in native.SCOPES:
        for block in ci[scope].values():
            assert block["coverage_difference"] == {"valid_replicates": 0, "p025": None, "p975": None}
            assert block["messages_per_30_days_difference"]["valid_replicates"] == 2000


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate_base", "duplicate_aft"])
def test_paired_uncertainty_rejects_population_mismatch_before_reindex(replay_case, mutation):
    units = _paired_case(replay_case)
    index = units.index[units.model_code.eq(native.AFT) & units.evaluation_scope.eq(native.SCOPES[0])][0]
    if mutation == "missing":
        units = units.drop(index)
    elif mutation == "extra":
        row = units.loc[[index]].copy()
        row["field_season"] = "synthetic_extra_identity"
        units = pd.concat([units, row], ignore_index=True)
    elif mutation == "duplicate_aft":
        units = pd.concat([units, units.loc[[index]]], ignore_index=True)
    else:
        index = units.index[units.model_code.eq(native.CALENDAR)][0]
        units = pd.concat([units, units.loc[[index]]], ignore_index=True)
    with pytest.raises(ValueError, match="Paired populations"):
        native.paired_uncertainty(units, "cutworm")


@pytest.mark.parametrize("column,value", [("season", 2021), ("cluster", "synthetic_mismatched_group"), ("warnable_first_event", False), ("field_days", 999)])
def test_paired_uncertainty_rejects_changed_event_or_exposure(replay_case, column, value):
    units = _paired_case(replay_case)
    index = units.index[units.model_code.eq(native.AFT) & units.warnable_first_event][0]
    units.loc[index, column] = value
    with pytest.raises(ValueError, match="Paired event definitions or exposure"):
        native.paired_uncertainty(units, "cutworm")


def test_wisconsin_uses_year_blocks_without_invented_physical_locations(replay_case):
    ci = native.paired_uncertainty(_paired_case(replay_case), "cpb_wisconsin")
    assert all(set(ci[scope]) == {"season"} for scope in native.SCOPES)


@pytest.fixture
def synthetic_collection(tmp_path, monkeypatch):
    collection = tmp_path/"collection"
    folder = collection/"normalized"/"zenodo_6351794"
    folder.mkdir(parents=True)
    rows = []
    for year in [1997, 1998, 1999, 2000]:
        for site, day, latitude, longitude, positive in [("synthetic_a", 1, 1., 2., True), ("synthetic_b", 3, 3., 4., False)]:
            start = pd.Timestamp(year=year, month=5, day=day)
            duration = 14+(year-1997)+(0 if positive else 7)
            for delay in [0, 7, duration]:
                rows.append(dict(source_id="zenodo_6351794", site_id=site, observation_date=start+pd.Timedelta(days=delay),
                    organism="Agrotis ipsilon", value=float(positive and delay == duration), unit="source_trap_catch_value",
                    source_latitude=latitude, source_longitude=longitude))
    pd.DataFrame(rows).to_parquet(folder/"observations.parquet", index=False)
    (collection/"manifest.json").write_text(json.dumps({"status": "synthetic_fixture"}), encoding="utf-8")
    monkeypatch.setattr(native, "target_config", lambda target: {"warmup_years": [1997, 1998, 1999], "outer_years": [2000]})
    monkeypatch.setattr(native, "check_collection", lambda root: {"status": "passed", "synthetic_only": True})
    return collection


def test_full_compute_fits_past_and_replays_existing_policy_with_censoring(synthetic_collection, tmp_path):
    tables = native._compute(synthetic_collection, "cutworm", "reported")
    registry = tables["event_registry"]
    outer = registry[registry.cohort_included & registry.season.eq(2000)]
    assert len(outer) == 2 and outer.warnable_first_event.sum() == 1
    assert tables["training_history"].training_max_year.tolist() == [1999]
    assert tables["training_history"].training_years.tolist() == ["1997,1998,1999"]
    metadata = json.loads(tables["training_history"].metadata_json.iloc[0])
    assert metadata["observed_events"] == 3 and metadata["right_censored"] == 3
    assert metadata["training_max_year"] < 2000
    assert metadata["parameters"]["depth"] == 3 and metadata["parameters"]["iterations"] == 200
    decisions, notifications = tables["decisions"], tables["notifications"]
    for model in native.MODELS:
        point = tables["predictions"][tables["predictions"].model_code.eq(model)]
        dates = decisions.field_season.map(point.set_index("field_season").predicted_date)
        score = (dates-decisions.issue_date).dt.days.between(3, 10).astype(float).where(dates.notna())
        replay = simulate_policy(decisions, score, native.POLICY, evaluation_scope="full_service_calendar", evaluation_mask=decisions.service_active)
        saved = notifications[notifications.model_code.eq(model)].reset_index(drop=True)
        pd.testing.assert_frame_equal(saved[replay.columns], replay.reset_index(drop=True))
        assert saved.issued_at_semantics.eq(decisions.issued_at_semantics.iloc[0]).all()
        assert not saved.loc[~saved.service_active, "message_issued"].any()
    totals = tables["metrics"][tables["metrics"].fold_id.eq("all_outer_years")]
    assert totals.warnable_events.eq(1).all() and totals.series.eq(2).all()
    assert set(totals.source_id) == {"zenodo_6351794"}
    assert len(tables["gained_lost"]) == 1
    # Parquet strips pivot's columns.name metadata. Persisted event comparison
    # must therefore canonicalize that metadata before sealing a run.
    path = tmp_path/"gained_lost.parquet"
    tables["gained_lost"].to_parquet(path, index=False)
    pd.testing.assert_frame_equal(tables["gained_lost"], pd.read_parquet(path), check_exact=True)


def test_existing_output_directory_is_not_overwritten(synthetic_collection, tmp_path):
    root = tmp_path/"existing"
    root.mkdir()
    sentinel = root/"user_file.txt"
    sentinel.write_text("preserve this", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable"):
        native.run(synthetic_collection, "cutworm", "reported", root)
    assert sentinel.read_text() == "preserve this" and list(root.iterdir()) == [sentinel]


def test_saved_run_recomputes_models_and_detects_output_and_input_tampering(synthetic_collection, tmp_path):
    root = tmp_path/"run"
    assert native.run(synthetic_collection, "cutworm", "reported", root)["status"] == "complete"
    checked = native.verify(root)
    assert checked["status"] == "passed"
    assert len(checked["recomputed_tables"]) == 9 and checked["checked_saved_models"] == 1
    assert checked["mismatches"] == checked["input_mismatches"] == []
    manifest_path = root/"manifest.json"
    original_manifest = manifest_path.read_bytes()
    metrics_path = root/"metrics.parquet"
    original_metrics = metrics_path.read_bytes()
    changed = pd.read_parquet(metrics_path)
    changed.loc[0, "messages"] += 1
    changed.to_parquet(metrics_path, index=False)
    rejected = native.verify(root)
    assert rejected["status"] == "failed" and rejected["mismatches"] == ["metrics.parquet"]
    manifest = json.loads(original_manifest)
    manifest["file_inventory"]["metrics.parquet"] = digest_file(metrics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    recomputed = native.verify(root)
    assert recomputed["status"] == "failed" and "recompute:metrics" in recomputed["mismatches"]
    metrics_path.write_bytes(original_metrics)
    manifest_path.write_bytes(original_manifest)
    source_path = synthetic_collection/"normalized"/"zenodo_6351794"/"observations.parquet"
    input_rows = pd.read_parquet(source_path)
    input_rows.loc[0, "value"] = 1
    input_rows.to_parquet(source_path, index=False)
    changed_input = native.verify(root)
    assert changed_input["status"] == "failed" and str(source_path.resolve()) in changed_input["input_mismatches"]
    assert changed_input["recomputed_tables"] == []
