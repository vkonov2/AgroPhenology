"""Synthetic-only checks for temporal selection, sequential burden and replay."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agro_phenology import external_carrot_validation as validation
from agro_phenology.early_warning_models import simulate_policy
from agro_phenology.external_carrot_cohort import SOURCE_ID, UNIT, build_decisions, build_registry
from agro_phenology.external_validation_io import digest_file


def _observations(site, year, visits, point=(1, 2)):
    """Locations and identifiers are synthetic, never copied from real fields."""
    return pd.DataFrame([
        dict(source_id=SOURCE_ID, site_id=site, organism="Psila rosae", unit=UNIT,
             observation_date=pd.Timestamp(f"{year}-{day}"), value=value,
             source_latitude=str(point[0]), source_longitude=str(point[1]))
        for day, value in visits
    ])


def _warmup():
    return pd.concat([
        _observations(f"warmup_{year}", year, [("05-31", 0), ("06-05", 1)], (5, 6))
        for year in [2009, 2010, 2011]
    ], ignore_index=True)


def test_calendar_reads_only_strictly_prior_eligible_outcomes(monkeypatch):
    monkeypatch.setattr(validation, "OUTER_YEARS", [2012, 2013])
    observations = pd.concat([
        _warmup(),
        _observations("outer_a", 2012, [("05-31", 0), ("06-07", 1)]),
        _observations("future", 2013, [("05-31", 0), ("06-14", 1)]),
        _observations("right_censored", 2010, [("05-31", 0), ("06-15", 0)]),
        _observations("late_entry", 2010, [("06-01", 0), ("06-08", 1)]),
        _observations("left_censored", 2010, [("05-31", 1), ("06-07", 0)]),
    ], ignore_index=True)
    registry = build_registry(observations)
    history = validation.calendar_history(registry).set_index("season")
    assert history.loc[2012, "training_years"] == "2009,2010,2011"
    assert history.loc[2012, "training_events"] == 3
    assert history.loc[2012, "training_max_year"] == 2011
    assert history.loc[2013, "training_max_year"] == 2012
    assert history.loc[2013, "training_events"] == 4
    changed = registry.copy()
    changed.loc[changed.season.ge(2012), "first_recorded_event_date"] += pd.Timedelta(days=20)
    again = validation.calendar_history(changed).set_index("season")
    pd.testing.assert_series_equal(history.loc[2012], again.loc[2012])
    # The same changed 2012 outcomes may legitimately affect the 2013 calendar.
    assert history.loc[2013, "training_events"] == again.loc[2013, "training_events"]
    future_only = registry.copy()
    future_only.loc[future_only.season.ge(2013), "first_recorded_event_date"] += pd.Timedelta(days=40)
    pd.testing.assert_frame_equal(validation.calendar_history(registry), validation.calendar_history(future_only))


def test_calendar_requires_three_past_years_not_three_repeated_series(monkeypatch):
    monkeypatch.setattr(validation, "OUTER_YEARS", [2012])
    observations = pd.concat([
        _observations(f"same_year_{i}", 2011, [("05-31", 0), ("06-07", 1)])
        for i in range(3)
    ], ignore_index=True)
    with pytest.raises(ValueError, match="three predeclared warmup years"):
        validation.calendar_history(build_registry(observations))


def test_calendar_rounds_half_up_and_never_uses_future_dates(monkeypatch):
    monkeypatch.setattr(validation, "OUTER_YEARS", [2012])
    observations = pd.concat([
        _observations("a", 2009, [("05-01", 0), ("05-24", 1)]),
        _observations("b", 2010, [("05-01", 0), ("05-25", 1)]),
        _observations("c", 2011, [("05-01", 0), ("05-26", 1)]),
        _observations("d", 2011, [("05-01", 0), ("05-27", 1)]),
    ], ignore_index=True)
    # 144,145,146,147 -> 145.5 -> 146, even in a leap-year prediction calendar.
    history = validation.calendar_history(build_registry(observations)).iloc[0]
    assert history.median_doy == 146
    assert history.predicted_date == pd.Timestamp("2012-01-01") + pd.Timedelta(days=145)


def _states(registry, *, score=None, common=None):
    decisions = build_decisions(registry)
    values = pd.Series(1., index=decisions.index) if score is None else score(decisions)
    mask = pd.Series(True, index=decisions.index) if common is None else common(decisions)
    decisions["common_weather_complete"] = mask
    one = simulate_policy(decisions, values, validation.POLICY,
                          evaluation_mask=decisions.evaluation_field_day)
    one["spatial_group"] = decisions.spatial_group
    one["source_id"] = SOURCE_ID
    blocks = []
    for model in validation.MODELS:
        block = one.copy()
        block["model_code"] = model
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def test_common_mask_filters_messages_and_burden_without_resetting_state(monkeypatch):
    monkeypatch.setattr(validation, "OUTER_YEARS", [2020])
    registry = build_registry(_observations("synthetic", 2020, [("05-18", 0), ("06-10", 1)]))
    states = _states(registry, common=lambda d: ~d.issue_date.eq("2020-05-19"))
    original = states.copy(deep=True)
    service, _, _ = validation.summarize_states(states, registry, "service_calendar")
    paired, _, _ = validation.summarize_states(states, registry, "paired_computable_days")
    full = service[service.fold_id.eq("all_2012_2019")]
    masked = paired[paired.fold_id.eq("all_2012_2019")]
    assert full.messages.eq(2).all()   # May19 and June3, real cooldown preserved.
    assert masked.messages.eq(1).all()
    assert full.active_alarm_days.eq(14).all()
    assert masked.active_alarm_days.eq(13).all()  # May20-25 stays active from off-mask May19.
    assert (full.field_days.to_numpy() - masked.field_days.to_numpy() == 1).all()
    assert full.timely_hits.eq(1).all() and masked.timely_hits.eq(1).all()
    pd.testing.assert_frame_equal(states, original)


def test_no_common_actionable_day_means_no_conditional_computability_or_hit(monkeypatch):
    monkeypatch.setattr(validation, "OUTER_YEARS", [2020])
    registry = build_registry(_observations("synthetic", 2020, [("05-18", 0), ("06-10", 1)]))
    states = _states(
        registry,
        score=lambda d: d.issue_date.eq("2020-06-03").astype(float),
        common=lambda d: ~d.days_to_first_recorded_event.between(3, 10),
    )
    original = states.copy(deep=True)
    service, _, _ = validation.summarize_states(states, registry, "service_calendar")
    paired, _, _ = validation.summarize_states(states, registry, "paired_computable_days")
    full = service[service.fold_id.eq("all_2012_2019")]
    masked = paired[paired.fold_id.eq("all_2012_2019")]
    assert full.events_with_warning_opportunity.eq(1).all()
    assert masked.events_with_warning_opportunity.eq(1).all()  # Denominator is not cherry-picked.
    assert full.computable_events.eq(1).all() and full.timely_hits.eq(1).all()
    assert masked.computable_events.eq(0).all() and masked.timely_hits.eq(0).all()
    assert masked.messages.eq(0).all()
    assert masked.active_alarm_days.eq(2).all()  # June8-9 retain the real off-mask alarm.
    pd.testing.assert_frame_equal(states, original)


def _bootstrap_series():
    rows = []
    for group, size in enumerate([9, 1, 1, 1]):
        for i in range(size):
            field = f"synthetic_{group}_{i}"
            for model in validation.MODELS:
                baseline = model == validation.CALENDAR
                hit = group == 1 if baseline else group == 0
                messages = 1 if baseline else 2 if group == 0 else 0 if group == 1 else 1
                alarms = 2 if baseline else 4 if group == 0 else 1 if group == 1 else 2
                rows.append(dict(evaluation_scope="service_calendar", model_code=model,
                                 field_season=field, spatial_group=f"group_{group}", season=2020,
                                 eligible_event=True, field_days=10, timely_hit=hit,
                                 messages=messages, active_alarm_days=alarms))
    return pd.DataFrame(rows)


def test_cluster_bootstrap_is_paired_weighted_and_invariant_to_other_model_row_order():
    series = _bootstrap_series()
    result = validation.uncertainty(series)
    pd.testing.assert_frame_equal(series, _bootstrap_series())
    shuffled = series[series.model_code.eq(validation.CALENDAR)]
    shuffled = pd.concat([shuffled, series[~series.model_code.eq(validation.CALENDAR)].sample(frac=1, random_state=7)])
    assert result == validation.uncertainty(shuffled)
    # Manual cluster totals: keep dominant cluster's nine events, do not average
    # four cluster-wise recall ratios as if each cluster had equal event counts.
    totals = np.array([[9, 90, 9, 9, 18], [1, 10, -1, -1, -1], [1, 10, 0, 0, 0], [1, 10, 0, 0, 0]], dtype=float)
    rng = np.random.default_rng(20261110)
    draws = totals[rng.integers(0, 4, size=(2000, 4))].sum(axis=1)
    values = {
        "coverage_difference": draws[:, 2] / draws[:, 0],
        "messages_per_30_days_difference": 30 * draws[:, 3] / draws[:, 1],
        "alarm_fraction_difference": draws[:, 4] / draws[:, 1],
    }
    for model in [validation.LUKE, validation.LAG7]:
        comparison = result[model + "_minus_calendar"]
        assert comparison["spatial_group"]["clusters"] == 4
        assert comparison["season"]["clusters"] == 1
        for metric, sample in values.items():
            assert comparison["spatial_group"][metric]["p025"] == pytest.approx(np.quantile(sample, .025))
            assert comparison["spatial_group"][metric]["p975"] == pytest.approx(np.quantile(sample, .975))
        assert comparison["season"]["coverage_difference"]["p025"] == pytest.approx(8 / 12)
        assert comparison["season"]["coverage_difference"]["p975"] == pytest.approx(8 / 12)


def test_identical_paired_models_have_zero_bootstrap_difference_and_ignore_other_scope():
    series = _bootstrap_series()
    calendar = series[series.model_code.eq(validation.CALENDAR)]
    copies = []
    for model in validation.MODELS:
        item = calendar.copy()
        item["model_code"] = model
        copies.append(item)
    same = pd.concat(copies, ignore_index=True)
    reference = validation.uncertainty(same)
    conditional = series.copy()
    conditional["evaluation_scope"] = "paired_computable_days"
    assert validation.uncertainty(pd.concat([same, conditional], ignore_index=True)) == reference
    for comparison in reference.values():
        for group in comparison.values():
            for metric in ["coverage_difference", "messages_per_30_days_difference", "alarm_fraction_difference"]:
                assert group[metric]["p025"] == group[metric]["p975"] == 0.
                assert group[metric]["valid_replicates"] == 2000


def test_bootstrap_draws_without_events_are_undefined_not_zero_coverage():
    series = _bootstrap_series()
    series = series[series.spatial_group.isin(["group_0", "group_1"])].copy()
    no_event = series.spatial_group.eq("group_1")
    series.loc[no_event, "eligible_event"] = False
    series.loc[no_event, "timely_hit"] = False
    result = validation.uncertainty(series)
    draws = np.random.default_rng(20261110).integers(0, 2, size=(2000, 2))
    valid = int((draws == 0).any(axis=1).sum())
    assert 0 < valid < 2000
    for comparison in result.values():
        coverage = comparison["spatial_group"]["coverage_difference"]
        assert coverage["valid_replicates"] == valid
        assert coverage["p025"] == coverage["p975"] == 1.
        assert comparison["spatial_group"]["messages_per_30_days_difference"]["valid_replicates"] == 2000
    series["eligible_event"] = False
    series["timely_hit"] = False
    no_events = validation.uncertainty(series)
    for comparison in no_events.values():
        for group in comparison.values():
            assert group["coverage_difference"] == {"valid_replicates": 0, "p025": None, "p975": None}


@pytest.fixture
def synthetic_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "OUTER_YEARS", [2012])
    collection, weather = tmp_path / "collection", tmp_path / "weather"
    normalized = collection / "normalized" / SOURCE_ID
    raw = collection / "raw" / SOURCE_ID
    normalized.mkdir(parents=True)
    raw.mkdir(parents=True)
    weather.mkdir()
    observations = pd.concat([
        _warmup(),
        _observations("outer_a", 2012, [("05-31", 0), ("06-07", 1)]),
        _observations("outer_b", 2012, [("05-31", 0), ("06-14", 1)], (3, 4)),
        _observations("outer_c", 2012, [("05-31", 0), ("06-20", 0)]),
    ], ignore_index=True)
    observations.to_parquet(normalized / "observations.parquet", index=False)
    (collection / "manifest.json").write_text("{}", encoding="utf-8")
    (raw / "metadata.json").write_text("{}", encoding="utf-8")
    (weather / "weather_plan.json").write_text("{}", encoding="utf-8")
    mapping = pd.DataFrame(dict(source_id=SOURCE_ID, site_id=["outer_a", "outer_b", "outer_c"],
                                year=2012, request_id="synthetic_weather_2012"))
    mapping.to_parquet(weather / "site_request_mapping.parquet", index=False)
    times = pd.date_range("2012-01-01", "2012-07-01", freq="h", tz="UTC", inclusive="left")
    pd.DataFrame(dict(valid_time_utc=times, temperature_2m=7.5)).to_parquet(weather / "hourly.parquet", index=False)
    (weather / "weather_manifest.json").write_text(json.dumps({"request_results": {
        "synthetic_weather_2012": {"status": "complete", "hourly_path": "hourly.parquet"},
    }}), encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text('{"synthetic_test_only": true}', encoding="utf-8")
    monkeypatch.setattr(validation, "CONTRACT", contract)
    # Integrity adapters are independently tested by the collection modules;
    # this fixture exercises actual weather/model/notification computations.
    monkeypatch.setattr(validation, "check_collection", lambda _: {"status": "passed"})
    monkeypatch.setattr(validation, "check_weather", lambda _: {"status": "passed"})
    return collection, weather


def test_end_to_end_compute_uses_same_events_dates_and_real_sequential_policy(synthetic_inputs):
    tables = validation._compute(*synthetic_inputs)
    outer = tables["event_registry"]
    outer = outer[outer.cohort_included & outer.season.eq(2012)]
    assert len(outer) == 3 and outer.warnable_first_event.sum() == 2
    notifications, predictions = tables["notifications"], tables["predictions"]
    key_columns = ["field_season", "season", "issue_date", "service_active", "evaluation_field_day"]
    reference = None
    for model in validation.MODELS:
        frame = predictions[predictions.model_code.eq(model)].reset_index(drop=True)
        if reference is None:
            reference = frame[key_columns]
        else:
            pd.testing.assert_frame_equal(reference, frame[key_columns])
        replay = simulate_policy(frame, frame.score, validation.POLICY,
                                 evaluation_mask=frame.evaluation_field_day)
        replay["source_id"], replay["model_code"] = SOURCE_ID, model
        replay["spatial_group"] = frame.spatial_group
        saved = notifications[notifications.model_code.eq(model)].reset_index(drop=True)
        pd.testing.assert_frame_equal(saved, replay)
        assert not saved.loc[~saved.service_active, "message_issued"].any()
        assert not saved.loc[~saved.service_active, "alarm_active"].any()
    overall = tables["metrics"]
    overall = overall[overall.fold_id.eq("all_2012_2019") & overall.evaluation_scope.eq("service_calendar")]
    assert overall.events_with_warning_opportunity.eq(2).all()
    assert overall.series.eq(3).all() and overall.right_censored.eq(1).all()
    assert overall.field_days.eq(7 + 14 + 20).all()
    right = outer[outer.first_recorded_event_date.isna()].field_season.iloc[0]
    censored_states = notifications[notifications.field_season.eq(right)]
    assert censored_states.service_active.all()
    assert not censored_states.loc[censored_states.issue_date.gt("2012-06-20"), "evaluation_scope_day"].any()
    assert censored_states.groupby("model_code").message_issued.sum().ge(1).all()


def test_run_is_immutable_and_verify_recomputes_tables_and_uncertainty(synthetic_inputs, tmp_path):
    root = tmp_path / "completed_run"
    result = validation.run(*synthetic_inputs, root)
    assert result["status"] == "complete"
    checked = validation.verify(root)
    assert checked["status"] == "passed"
    assert set(checked["recomputed_tables"]) == set(validation._compute(*synthetic_inputs))
    assert not checked["mismatches"] and not checked["input_mismatches"]
    before = digest_file(root / "manifest.json")
    with pytest.raises(FileExistsError, match="immutable"):
        validation.run(*synthetic_inputs, root)
    assert digest_file(root / "manifest.json") == before
    with (root / "metrics.csv").open("a") as stream:
        stream.write("changed_output\n")
    corrupted = validation.verify(root)
    assert corrupted["status"] == "failed" and corrupted["mismatches"] == ["metrics.csv"]
    assert corrupted["recomputed_tables"] == []


def test_verify_rejects_input_tamper_even_when_changed_weather_is_outside_all_cutoffs(synthetic_inputs, tmp_path):
    root = tmp_path / "completed_run"
    validation.run(*synthetic_inputs, root)
    path = synthetic_inputs[1] / "hourly.parquet"
    hourly = pd.read_parquet(path)
    hourly.loc[hourly.valid_time_utc.ge(pd.Timestamp("2012-06-29", tz="UTC")), "temperature_2m"] = 99.
    hourly.to_parquet(path, index=False)
    checked = validation.verify(root)
    assert checked["status"] == "failed"
    assert str(path.resolve()) in checked["input_mismatches"]
    assert checked["recomputed_tables"] == []


def test_verify_detects_forged_metric_after_output_hash_is_updated(synthetic_inputs, tmp_path):
    root = tmp_path / "completed_run"
    validation.run(*synthetic_inputs, root)
    path = root / "metrics.parquet"
    metrics = pd.read_parquet(path)
    metrics.loc[0, "timely_hits"] += 1
    metrics.to_parquet(path, index=False)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file_inventory"][path.name] = digest_file(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    checked = validation.verify(root)
    assert checked["status"] == "failed"
    assert "recompute:metrics" in checked["mismatches"]
    assert not checked["input_mismatches"]
