from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agro_phenology import external_cpb_validation as cpb
from agro_phenology.early_warning_models import simulate_policy
from agro_phenology.external_cpb_model import LOWER_THRESHOLD_C
from agro_phenology.external_validation_io import digest_file


def _visit(site, day, *, eggs=0.0, larvae=((0.0, 0.0), (0.0, 0.0)), treatment="control"):
    rows = []
    for plant, values in enumerate(larvae, start=1):
        for stage, value in [("egg", eggs), ("larva_1", values[0]), ("larva_2", values[1])]:
            rows.append({
                "source_id": cpb.DATASET, "site_id": site, "observation_date": pd.Timestamp(day),
                "unit": "individuals_per_plant", "treatment": treatment,
                "plot_id": "plot", "triplet_id": "triplet", "block": "block",
                "plant_id": plant, "stage": stage, "value": value,
            })
    return pd.DataFrame(rows)


def _six_series_observations():
    blocks = [
        _visit("eligible_a", "2020-06-01", eggs=1),
        _visit("eligible_a", "2020-06-04", larvae=((0, np.nan), (0, 0))),
        _visit("eligible_a", "2020-06-08", larvae=((1, np.nan), (0, 0))),
        _visit("eligible_a", "2020-06-16", larvae=((2, 0), (0, 0))),
        _visit("eligible_b", "2020-06-10", eggs=1),
        _visit("eligible_b", "2020-06-14", larvae=((0, 0), (0, 1))),
    ]
    for index in range(4):
        day = pd.Timestamp("2020-06-01") + pd.Timedelta(days=index)
        blocks.append(_visit(f"entry_{index}", day, eggs=1, larvae=((1, 0), (0, 0))))
        blocks.append(_visit(f"entry_{index}", day + pd.Timedelta(days=7), larvae=((2, 0), (0, 0))))
    # Treated positives cannot alter control biofix, event, completeness or plant count.
    blocks.append(_visit("eligible_a", "2020-05-25", eggs=3, larvae=((5, 5), (5, 5)), treatment="mulch"))
    return pd.concat(blocks, ignore_index=True)


@pytest.mark.parametrize("larvae,status,missing", [
    (((0, 0), (0, 0)), "zero", 0),
    (((0, np.nan), (0, 0)), "unknown", 1),
    (((np.nan, np.nan), (np.nan, np.nan)), "unknown", 4),
    (((1, np.nan), (0, 0)), "positive", 1),
])
def test_control_visits_distinguish_complete_zero_unknown_and_measured_positive(larvae, status, missing):
    visit = cpb.control_visits(_visit("site", "2020-06-01", larvae=larvae)).iloc[0]
    assert visit.early_larvae_status == status
    assert visit.early_values == visit.expected_values == 4
    assert visit.missing_early_values == missing


def test_zero_requires_both_stages_of_every_scheduled_plant_and_never_pools_treatments():
    complete = _visit("site", "2020-06-01", eggs=1)
    incomplete = _visit("site", "2020-06-02")
    incomplete = incomplete.loc[~(incomplete.plant_id.eq(2) & incomplete.stage.eq("larva_2"))]
    absent_plant = _visit("site", "2020-06-03").loc[lambda f: f.plant_id.eq(1)]
    treated = _visit("site", "2020-05-25", eggs=1, larvae=((1, 1), (1, 1)), treatment="mulch")
    treated["plant_id"] += 100
    visits = cpb.control_visits(pd.concat([complete, incomplete, absent_plant, treated], ignore_index=True))
    assert visits.early_larvae_status.tolist() == ["zero", "unknown", "unknown"]
    assert visits.expected_values.tolist() == [4, 4, 4]
    assert visits.early_values.tolist() == [4, 3, 2]
    assert visits.observation_date.min() == pd.Timestamp("2020-06-01")
    assert visits.treatment.eq("control").all()


def test_complete_visits_remain_zero_when_plant_identifiers_change_between_assessments():
    first = _visit("site", "2020-06-01", eggs=1)
    second = _visit("site", "2020-06-08")
    second["plant_id"] += 10
    visits = cpb.control_visits(pd.concat([first, second], ignore_index=True))
    assert visits.expected_values.tolist() == [4, 4]
    assert visits.early_values.tolist() == [4, 4]
    assert visits.early_larvae_status.tolist() == ["zero", "zero"]


def test_duplicate_plant_stage_mixed_dataset_wrong_units_and_negative_counts_are_rejected():
    observations = _visit("site", "2020-06-01")
    with pytest.raises(ValueError, match="Duplicate plant-stage"):
        cpb.control_visits(pd.concat([observations, observations.iloc[[1]]], ignore_index=True))
    mixed = observations.copy()
    mixed.loc[0, "source_id"] = "unrelated_dataset"
    with pytest.raises(ValueError, match="exactly the DaKS CPB dataset"):
        cpb.control_visits(mixed)
    wrong_unit = observations.copy()
    wrong_unit.loc[0, "unit"] = "individuals_per_plot"
    with pytest.raises(ValueError, match="measurement unit"):
        cpb.control_visits(wrong_unit)
    negative = observations.copy()
    negative.loc[1, "value"] = -1
    with pytest.raises(ValueError, match="Negative insect counts"):
        cpb.control_visits(negative)


def test_registry_has_two_independent_eligible_series_and_four_left_censored_entries():
    registry = cpb.event_registry(cpb.control_visits(_six_series_observations())).set_index("site_id")
    assert len(registry) == 6
    assert registry.warnable_first_event.sum() == 2
    assert registry.positive_at_first_visit.sum() == 4
    assert registry.eligibility_category.value_counts().to_dict() == {
        "positive_at_entry_left_censored": 4, "eligible": 2,
    }
    a = registry.loc["eligible_a"]
    assert a.biofix_date == pd.Timestamp("2020-06-01")
    assert a.last_complete_zero_date == pd.Timestamp("2020-06-01")
    assert a.first_recorded_event_date == pd.Timestamp("2020-06-08")
    assert a.last_assessment_date == pd.Timestamp("2020-06-16")
    assert a.previous_visit_gap_days == 7
    assert registry.field_season.nunique() == 6


@pytest.mark.parametrize("event_delay,eligible", [(3, False), (4, True)])
def test_registry_minimum_warning_opportunity_counts_entry_next_day(event_delay, eligible):
    biofix = pd.Timestamp("2020-06-01")
    observations = pd.concat([
        _visit("site", biofix, eggs=1),
        _visit("site", biofix + pd.Timedelta(days=event_delay), larvae=((1, 0), (0, 0))),
    ], ignore_index=True)
    event = cpb.event_registry(cpb.control_visits(observations)).iloc[0]
    assert bool(event.warnable_first_event) is eligible
    assert not event.positive_at_first_visit


def test_unknown_larvae_never_supply_a_required_pre_event_zero():
    observations = pd.concat([
        _visit("site", "2020-06-01", eggs=1, larvae=((0, np.nan), (0, 0))),
        _visit("site", "2020-06-08", larvae=((1, 0), (0, 0))),
    ], ignore_index=True)
    event = cpb.event_registry(cpb.control_visits(observations)).iloc[0]
    assert pd.isna(event.last_complete_zero_date)
    assert not event.warnable_first_event
    assert event.eligibility_category == "insufficient_warning_or_zero"


def test_registry_rejects_mixing_seasons_within_one_field_series():
    observations = pd.concat([
        _visit("site", "2020-06-01", eggs=1), _visit("site", "2021-06-01", eggs=1),
    ], ignore_index=True)
    with pytest.raises(ValueError, match="one season"):
        cpb.event_registry(cpb.control_visits(observations))


def test_daily_grid_is_independent_of_future_visits_and_stops_only_after_record_availability():
    registry = cpb.event_registry(cpb.control_visits(_six_series_observations()))
    original = cpb.decision_calendar(registry)
    changed_registry = registry.copy()
    changed_registry["last_assessment_date"] += pd.Timedelta(days=20)
    changed_registry["first_recorded_event_date"] += pd.Timedelta(days=15)
    changed_registry["previous_visit_gap_days"] = 100
    changed = cpb.decision_calendar(changed_registry)
    static_columns = ["site_id", "field_season", "issue_date", "issued_at", "biofix_date"]
    pd.testing.assert_frame_equal(original[static_columns], changed[static_columns])
    for event in registry.itertuples(index=False):
        sub = original[original.site_id.eq(event.site_id)].set_index("issue_date")
        assert sub.index.min() == event.biofix_date + pd.Timedelta(days=1)
        assert sub.index.max() == pd.Timestamp("2020-10-31")
        assert len(sub) == (pd.Timestamp("2020-10-31") - event.biofix_date).days
        assert sub.index.to_series().diff().dropna().eq(pd.Timedelta(days=1)).all()
        if event.first_recorded_event_date >= sub.index.min():
            assert sub.loc[event.first_recorded_event_date, "service_active"]
        assert not sub.loc[event.first_recorded_event_date + pd.Timedelta(days=1), "service_active"]
        assert not sub.loc[sub.index > event.first_recorded_event_date, "service_active"].any()
    assert original.target_class.eq("unknown").all()
    assert not original.target_observable.any()


@pytest.mark.parametrize("predicted,expected", [
    ("2020-06-01", 3), ("2020-06-03", 1), ("2020-06-04", 0),
    ("2020-06-06", 0), ("2020-06-08", 0), ("2020-06-09", 1),
])
def test_interval_distance_respects_exclusive_last_zero_and_inclusive_first_positive(predicted, expected):
    assert cpb.interval_distance(pd.Timestamp(predicted), pd.Timestamp("2020-06-03"), pd.Timestamp("2020-06-08")) == expected


@pytest.mark.parametrize("position", [0, 1, 2])
def test_interval_distance_is_unknown_when_any_date_is_unknown(position):
    dates = [pd.Timestamp("2020-06-04"), pd.Timestamp("2020-06-03"), pd.Timestamp("2020-06-08")]
    dates[position] = pd.NaT
    assert pd.isna(cpb.interval_distance(*dates))


@pytest.fixture
def synthetic_inputs(tmp_path):
    collection, weather = tmp_path / "collection", tmp_path / "weather"
    normalized = collection / "normalized" / cpb.DATASET
    normalized.mkdir(parents=True)
    weather.mkdir()
    observations = _six_series_observations()
    observations.to_parquet(normalized / "observations.parquet", index=False)
    sites = sorted(observations.site_id.unique())
    pd.DataFrame({"source_id": cpb.DATASET, "site_id": sites, "year": 2020, "request_id": "synthetic_cell_2020"}).to_parquet(weather / "site_request_mapping.parquet", index=False)
    times = pd.date_range(
        pd.Timestamp("2020-05-01", tz="Europe/Berlin").tz_convert("UTC"),
        pd.Timestamp("2020-10-27", tz="Europe/Berlin").tz_convert("UTC"),
        freq="h", inclusive="left",
    )
    pd.DataFrame({"valid_time_utc": times, "temperature_2m": LOWER_THRESHOLD_C + 5}).to_parquet(weather / "hourly.parquet", index=False)
    (weather / "weather_manifest.json").write_text(json.dumps({
        "request_results": {"synthetic_cell_2020": {"status": "complete", "hourly_path": "hourly.parquet"}},
    }), encoding="utf-8")
    return collection, weather


def test_compute_reuses_the_fixed_existing_simulator_and_preserves_message_windows(synthetic_inputs):
    tables = cpb._compute(*synthetic_inputs)
    assert cpb.simulate_policy is simulate_policy
    assert (cpb.POLICY.threshold, cpb.POLICY.active_days, cpb.POLICY.cooldown_days) == (.5, 7, 15)
    assert len(tables["metrics"]) == 2
    assert tables["metrics"].events_with_warning_opportunity.eq(2).all()
    assert tables["metrics"].first_events.eq(2).all()
    predictions, notifications = tables["predictions"], tables["notifications"]
    for model in cpb.MODELS:
        frame = predictions[predictions.model_code.eq(model)].reset_index(drop=True)
        replay = simulate_policy(frame, frame.score, cpb.POLICY, evaluation_mask=frame.evaluation_field_day)
        replay["model_code"], replay["source_id"] = model, cpb.DATASET
        saved = notifications[notifications.model_code.eq(model)].reset_index(drop=True)
        pd.testing.assert_frame_equal(saved, replay)
        messages = saved[saved.message_issued]
        assert len(messages) > 0
        assert (messages.forecast_window_start - messages.issue_date).dt.days.eq(3).all()
        assert (messages.forecast_window_end - messages.issue_date).dt.days.eq(10).all()
        assert messages.score.ge(.5).all()
        assert not saved.loc[~saved.service_active, "message_issued"].any()
        assert not saved.loc[~saved.service_active, "alarm_active"].any()
        assert saved.loc[~saved.service_active, "action_reason"].eq("stopped_after_record_available").all()
        leads = (frame.predicted_event_date - frame.issue_date).dt.days
        assert frame.loc[frame.score.eq(1), "issue_date"].index.equals(leads[leads.between(3, 10)].index)


def test_verify_recomputes_all_saved_tables_and_rejects_changed_artifacts(synthetic_inputs, tmp_path):
    tables = cpb._compute(*synthetic_inputs)
    root = tmp_path / "run"
    root.mkdir()
    for name, table in tables.items():
        table.to_parquet(root / f"{name}.parquet", index=False)
    inputs = [p for folder in synthetic_inputs for p in folder.rglob("*") if p.is_file()]
    manifest = {
        "status": "complete", "collection_dir": str(synthetic_inputs[0]), "weather_dir": str(synthetic_inputs[1]),
        "input_hashes": {str(p): digest_file(p) for p in inputs},
        "file_inventory": {p.name: digest_file(p) for p in root.iterdir()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    checked = cpb.verify(root)
    assert checked["status"] == "passed"
    assert set(checked["recomputed_tables"]) == set(tables)
    assert checked["mismatches"] == checked["input_mismatches"] == []
    changed = tables["metrics"].copy()
    changed.loc[0, "messages"] += 1
    changed.to_parquet(root / "metrics.parquet", index=False)
    rejected = cpb.verify(root)
    assert rejected["status"] == "failed"
    assert rejected["mismatches"] == ["metrics.parquet"]
