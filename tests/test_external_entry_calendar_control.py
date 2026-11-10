from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agro_phenology import external_entry_calendar_control as entry
from agro_phenology import external_native_validation as native
from agro_phenology.early_warning_models import simulate_policy
from agro_phenology.external_native_cohorts import build_decisions, build_registry
from agro_phenology.external_validation_io import digest_file


def _train() -> pd.DataFrame:
    rows = []
    for i, (year, station, duration) in enumerate([(2017, "a", 5), (2018, "a", 7), (2019, "a", 9),
                                                (2017, "b", 20), (2018, "b", 30)]):
        start = pd.Timestamp(year=year, month=5, day=1)
        rows.append(dict(source_id="synthetic", target="synthetic_target", field_season=str(i), season=year,
                         entry_date=start, cluster=station, cohort_included=True,
                         first_recorded_event_date=start+pd.Timedelta(days=duration), last_assessment_date=start+pd.Timedelta(days=40)))
    rows.append(dict(source_id="synthetic", target="synthetic_target", field_season="censored", season=2019,
                     entry_date=pd.Timestamp("2019-05-01"), cluster="c", cohort_included=True,
                     first_recorded_event_date=pd.NaT, last_assessment_date=pd.Timestamp("2019-09-01")))
    return pd.DataFrame(rows)


def _test() -> pd.DataFrame:
    return pd.DataFrame(dict(source_id="synthetic", target="synthetic_target", season=2020,
                             entry_date=pd.to_datetime(["2020-10-30"]*3), cluster=["a", "b", "new"]), index=[9, 2, 8])


def test_entry_duration_median_is_global_or_sufficiently_local_and_censoring_is_not_event():
    calendar = entry.fit_entry_calendar(_train(), 2020)
    assert calendar["global_median_duration"] == 9
    assert calendar["groups"] == {"a": {"events": 3, "median_duration": 7}}
    assert calendar["right_censored"] == 1
    assert calendar["observed_events"] == 5
    predicted = entry.predict_entry_calendar(calendar, _test())
    expected = pd.Series(pd.to_datetime(["2020-11-06", "2020-11-08", "2020-11-08"]), index=[9, 2, 8], name="predicted_date")
    pd.testing.assert_series_equal(predicted, expected)


def test_wisconsin_global_control_has_no_fictitious_station_requirement():
    calendar = entry.fit_entry_calendar(_train().drop(columns="cluster"), 2020, use_station=False)
    assert calendar["groups"] == {}
    assert entry.predict_entry_calendar(calendar, _test().drop(columns="cluster")).dt.day.eq(8).all()


def test_short_events_are_included_half_up_is_fixed_and_future_outcomes_are_rejected():
    train = _train().iloc[:2].copy()
    train["first_recorded_event_date"] = train.entry_date+pd.to_timedelta([2, 3], unit="D")
    train["warnable_first_event"] = False
    calendar = entry.fit_entry_calendar(train, 2020, use_station=False)
    assert calendar["global_median_duration"] == 3
    train.loc[0, "season"] = 2020
    with pytest.raises(ValueError, match="prior years"):
        entry.fit_entry_calendar(train, 2020)


def test_current_future_outcomes_are_not_prediction_inputs():
    calendar = entry.fit_entry_calendar(_train(), 2020)
    before = entry.predict_entry_calendar(calendar, _test())
    rows = _test()
    rows["first_recorded_event_date"] = pd.Timestamp("2099-01-01")
    rows["last_assessment_date"] = pd.Timestamp("2100-01-01")
    rows["future_weather"] = -999
    rows["previous_visit_gap_days"] = 999
    pd.testing.assert_series_equal(before, entry.predict_entry_calendar(calendar, rows))


def _canonical_visits() -> pd.DataFrame:
    rows = []
    for year in [2017, 2018, 2019, 2020]:
        for site in ["observed", "censored"]:
            start = pd.Timestamp(year=year, month=5, day=1)
            for offset, status in [(0, "zero"), (20 if site == "observed" else 2, "positive" if site == "observed" else "zero")]:
                rows.append(dict(source_id="kbs_aphid_suction_2005_2013", target="kbs_myzus", site_id=site,
                    season=year, observation_date=start+pd.Timedelta(days=offset), status=status, cluster=site,
                    station_id=site, coordinate_scope="synthetic_station", entry_egg_presence=np.nan, entry_adult_presence=np.nan,
                    n_reports=2, missing_reports=0, all_taxa_zero=False, zero_mode="reported"))
    return pd.DataFrame(rows)


@pytest.fixture
def frozen_base(tmp_path) -> Path:
    base = tmp_path/"base"
    base.mkdir()
    registry = build_registry(_canonical_visits())
    outer = registry[registry.season.eq(2020)]
    decisions = build_decisions(outer)
    predictions, notifications, units = [], [], []
    for model, offset in [(native.AFT, 22), (native.CALENDAR, 30)]:
        p = outer[["field_season", "season"]].copy()
        p["model_code"], p["predicted_date"] = model, outer.entry_date+pd.Timedelta(days=offset)
        predictions.append(p)
        predicted = decisions.field_season.map(p.set_index("field_season").predicted_date)
        score = (predicted-decisions.issue_date).dt.days.between(3, 10).astype(float)
        state = simulate_policy(decisions, score, native.POLICY, evaluation_scope="full_service_calendar", evaluation_mask=decisions.service_active)
        state["model_code"] = model
        notifications.append(state)
        for scope in native.SCOPES:
            unit = native._unit_metrics(state, outer, p, scope)
            unit["model_code"] = model
            units.append(unit)
    units = pd.concat(units, ignore_index=True)
    tables = dict(event_registry=registry, decisions=decisions, predictions=pd.concat(predictions, ignore_index=True),
                  notifications=pd.concat(notifications, ignore_index=True), per_series_metrics=units,
                  metrics=native.aggregate_metrics(units, "kbs_myzus", "reported"))
    for name, frame in tables.items():
        frame.to_parquet(base/f"{name}.parquet", index=False)
    marker = tmp_path/"source.txt"
    marker.write_text("synthetic only")
    manifest = dict(status="complete", target="kbs_myzus", zero_mode="reported",
                    input_hashes={str(marker): digest_file(marker)},
                    file_inventory={p.name: digest_file(p) for p in base.glob("*.parquet")})
    (base/"manifest.json").write_text(json.dumps(manifest))
    return base


def test_supplement_uses_exact_frozen_populations_metrics_and_full_unknown_tail(frozen_base):
    tables = entry._compute(frozen_base)
    original = pd.read_parquet(frozen_base/"per_series_metrics.parquet")
    frozen_in_supplement = tables["per_series_metrics"][lambda x: x.model_code.isin(native.MODELS)].reset_index(drop=True)
    pd.testing.assert_frame_equal(original, frozen_in_supplement, check_exact=True)
    metrics = tables["metrics"]
    observed = metrics[(metrics.model_code==entry.ENTRY)&(metrics.fold_id=='all_outer_years')&(metrics.evaluation_scope=='observed_followup')].iloc[0]
    full = metrics[(metrics.model_code==entry.ENTRY)&(metrics.fold_id=='all_outer_years')&(metrics.evaluation_scope=='full_service_calendar')].iloc[0]
    assert observed.warnable_events == full.warnable_events == 1
    assert observed.timely_hits == full.timely_hits == 1
    assert observed.messages == 1
    assert full.messages == 2
    assert full.active_alarm_days > observed.active_alarm_days
    assert full.field_days > observed.field_days
    assert set(tables["gained_lost_vs_AFT"].evaluation_scope) == set(native.SCOPES)
    for frame in tables.values():
        assert frame.columns.name is None


def test_unchanged_policy_and_no_realignment_of_frozen_models(frozen_base):
    tables = entry._compute(frozen_base)
    state = tables["notifications"]
    assert state.policy_threshold.eq(.5).all()
    assert state.policy_active_days.eq(7).all()
    assert state.policy_cooldown_days.eq(15).all()
    assert (state.forecast_window_start-state.issue_date).dt.days.eq(3).all()
    assert (state.forecast_window_end-state.issue_date).dt.days.eq(10).all()
    assert not state.loc[~state.service_active, "message_issued"].any()
    assert set(state.model_code) == {entry.ENTRY}


def test_pair_uncertainty_reference_is_explicitly_entry_not_original_doy(frozen_base):
    tables = entry._compute(frozen_base)
    ci = entry.uncertainty(tables["per_series_metrics"], "kbs_myzus")
    assert ci["candidate_model"] == native.AFT
    assert ci["reference_model"] == entry.ENTRY
    assert ci["interpretation"] == "post_hoc_diagnostic_not_new_independent_test"
    assert set(ci["results"]) == set(native.SCOPES)


def test_run_is_immutable_and_full_recompute_passes_without_touching_base(frozen_base, tmp_path):
    root = tmp_path/"supplement"
    before = {p.name: digest_file(p) for p in frozen_base.iterdir()}
    assert entry.run(frozen_base, root)["status"] == "complete"
    checked = entry.verify(root)
    assert checked["status"] == "passed"
    assert len(checked["recomputed_tables"]) == 6
    assert {p.name: digest_file(p) for p in frozen_base.iterdir()} == before
    assert 'ПОСЛЕ просмотра' in (root/"REPORT_RU.md").read_text()
    with pytest.raises(FileExistsError):
        entry.run(frozen_base, root)


def test_verify_detects_changed_output_even_with_forged_hash(frozen_base, tmp_path):
    root = tmp_path/"supplement"
    entry.run(frozen_base, root)
    metrics = pd.read_parquet(root/"metrics.parquet")
    metrics.loc[0, "messages"] += 1
    metrics.to_parquet(root/"metrics.parquet", index=False)
    manifest = json.loads((root/"manifest.json").read_text())
    manifest["file_inventory"]["metrics.parquet"] = digest_file(root/"metrics.parquet")
    (root/"manifest.json").write_text(json.dumps(manifest))
    checked = entry.verify(root)
    assert checked["status"] == "failed"
    assert 'recompute:metrics' in checked['mismatches']


def test_base_tamper_blocks_computation_and_is_visible_in_verify(frozen_base, tmp_path):
    root = tmp_path/"supplement"
    entry.run(frozen_base, root)
    with (frozen_base/"notifications.parquet").open("ab") as handle:
        handle.write(b"tampered")
    assert entry.base_audit(frozen_base)["status"] == "failed"
    assert entry.verify(root)["status"] == "failed"
    with pytest.raises(ValueError, match="integrity"):
        entry.run(frozen_base, tmp_path/"another")
