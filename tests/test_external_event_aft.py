from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor

from agro_phenology import external_event_aft as aft


def _registry() -> pd.DataFrame:
    rows = []
    for year in [2017, 2018, 2019]:
        for index in range(12):
            entry = pd.Timestamp(year=year, month=5, day=1) + pd.Timedelta(days=index)
            censored = index in {9, 10, 11}
            rows.append(dict(source_id="synthetic_source", target="synthetic_insect", field_season=f"{year}_{index}",
                             season=year, cluster="a" if index < 6 else "b", entry_date=entry,
                             entry_doy=-100, entry_egg_presence=float(index % 2),
                             entry_adult_presence=np.nan if index == 11 else float(index % 3 == 0),
                             first_recorded_event_date=pd.NaT if censored else entry+pd.Timedelta(days=10+index),
                             last_assessment_date=entry+pd.Timedelta(days=60), cohort_included=True))
    return pd.DataFrame(rows)


def _prediction_rows() -> pd.DataFrame:
    entry = pd.to_datetime(["2020-05-02", "2020-05-08", "2020-05-12"])
    return pd.DataFrame(dict(source_id="synthetic_source", target="synthetic_insect", season=2020,
                             entry_date=entry, cluster=["a", "b", "new_station"],
                             entry_egg_presence=[0., 1., np.nan], entry_adult_presence=[1., 0., np.nan]), index=[8, 3, 22])


@pytest.fixture(scope="module")
def fitted_aft():
    return aft.fit_aft(_registry(), outer_year=2020)


def test_fixed_model_parameters_and_point_censored_label_meaning():
    registry = _registry().iloc[:3].copy()
    registry["first_recorded_event_date"] = [registry.entry_date.iloc[0]+pd.Timedelta(days=2), pd.NaT, pd.NaT]
    registry["last_assessment_date"] = [registry.entry_date.iloc[0]+pd.Timedelta(days=10),
                                        registry.entry_date.iloc[1]+pd.Timedelta(days=20), registry.entry_date.iloc[2]]
    train, metadata = aft._training_rows(registry, 2020)
    assert aft.aft_training_labels(train).tolist() == [[2., 2.], [20., -1.]]
    assert metadata["excluded_zero_followup"] == 1
    assert metadata["observed_events"] == metadata["right_censored"] == 1
    assert metadata["used_rows"] == 2
    assert aft.PARAMETERS == {
        "loss_function": "SurvivalAft:dist=Normal;scale=1", "depth": 3, "iterations": 200,
        "learning_rate": .03, "l2_leaf_reg": 10, "random_seed": 20261110, "thread_count": 2,
        "bootstrap_type": "No", "allow_writing_files": False, "verbose": False,
    }


@pytest.mark.parametrize("change", ["season", "event", "censor"])
def test_current_or_future_training_information_is_rejected(change):
    registry = _registry()
    if change == "season":
        registry.loc[0, "season"] = 2020
    elif change == "event":
        registry.loc[0, "first_recorded_event_date"] = pd.Timestamp("2020-01-02")
        registry.loc[0, "last_assessment_date"] = pd.Timestamp("2020-02-01")
    else:
        registry.loc[11, "last_assessment_date"] = pd.Timestamp("2020-01-02")
    with pytest.raises(ValueError, match="prior years|external year"):
        aft.fit_aft(registry, 2020)
    with pytest.raises(ValueError, match="prior years|external year"):
        aft.fit_calendar(registry, 2020)


def test_future_row_is_rejected_even_when_marked_outside_the_cohort():
    registry = _registry()
    registry.loc[0, ["season", "cohort_included"]] = [2021, False]
    with pytest.raises(ValueError, match="prior years"):
        aft.fit_aft(registry, 2020)


@pytest.mark.parametrize("column", ["source_id", "target"])
def test_one_fitted_model_cannot_pool_datasets_or_species(column):
    registry = _registry()
    registry.loc[0, column] = "another"
    with pytest.raises(ValueError, match=f"one {column}"):
        aft.fit_aft(registry, 2020)


@pytest.mark.parametrize("change,match", [
    ("point_at_zero_entry", "precede"),
    ("followup_before_entry", "chronology"),
    ("event_after_last_assessment", "chronology"),
    ("missing_followup", "Missing last_assessment"),
    ("duplicate_series", "Duplicate"),
    ("entry_wrong_year", "disagree"),
])
def test_invalid_training_observation_chronology_is_not_silently_imputed(change, match):
    registry = _registry()
    if change == "point_at_zero_entry":
        registry.loc[0, "first_recorded_event_date"] = registry.entry_date.iloc[0]
    elif change == "followup_before_entry":
        registry.loc[0, "last_assessment_date"] = registry.entry_date.iloc[0]-pd.Timedelta(days=1)
    elif change == "event_after_last_assessment":
        registry.loc[0, "last_assessment_date"] = registry.entry_date.iloc[0]+pd.Timedelta(days=1)
    elif change == "missing_followup":
        registry.loc[0, "last_assessment_date"] = pd.NaT
    elif change == "duplicate_series":
        registry.loc[1, "field_season"] = registry.field_season.iloc[0]
    else:
        registry.loc[0, "entry_date"] = pd.Timestamp("2016-05-01")
    with pytest.raises(ValueError, match=match):
        aft.fit_aft(registry, 2020)


def test_unknown_censoring_is_not_an_observed_event_and_no_event_training_is_rejected():
    registry = _registry()
    registry["first_recorded_event_date"] = pd.NaT
    with pytest.raises(ValueError, match="observed event"):
        aft.fit_aft(registry, 2020)


def test_entry_day_is_recomputed_and_only_known_entry_features_are_used():
    rows = _registry()
    features = aft.feature_frame(rows, use_station=True)
    assert features.columns.tolist() == ["entry_doy", "cluster"]
    assert features.entry_doy.iloc[0] == 121
    assert not features.entry_doy.eq(-100).any()
    changed = rows.copy()
    changed["first_recorded_event_date"] = pd.Timestamp("2040-12-31")
    changed["last_assessment_date"] = pd.Timestamp("2050-01-01")
    changed["future_weather"] = 1e9
    changed["previous_visit_gap_days"] = 900
    changed["entry_doy"] = 999
    pd.testing.assert_frame_equal(features, aft.feature_frame(changed, use_station=True))
    stage_features = aft.feature_frame(rows, use_station=False)
    assert stage_features.columns.tolist() == ["entry_doy", "entry_egg_presence", "entry_adult_presence"]
    assert stage_features.entry_adult_presence.isna().sum() == 3


@pytest.mark.parametrize("bad", [-1, 2, np.inf, "unknown"])
def test_stage_presence_never_converts_missing_or_invalid_values_to_absence(bad):
    rows = _registry()
    if isinstance(bad, str):
        rows["entry_egg_presence"] = rows.entry_egg_presence.astype(object)
    rows.loc[0, "entry_egg_presence"] = bad
    with pytest.raises(ValueError, match="entry_egg_presence"):
        aft.feature_frame(rows, use_station=False)


def test_missing_station_is_not_assigned_an_arbitrary_location():
    rows = _registry()
    rows.loc[0, "cluster"] = None
    with pytest.raises(ValueError, match="Missing cluster"):
        aft.feature_frame(rows, use_station=True)


def test_log_median_to_calendar_date_half_up_and_overflow_are_explicit():
    entry = pd.Series(pd.to_datetime(["2020-10-30"]*5), index=[7, 2, 19, 3, 5])
    result = aft._duration_dates(entry, [np.log(2.5), np.log(.1), 1000, np.nan, np.log(5)])
    expected = pd.Series(pd.to_datetime(["2020-11-02", "2020-10-31", None, None, "2020-11-04"]),
                         index=entry.index, name="predicted_event_date")
    pd.testing.assert_series_equal(result, expected)
    # Dates past the service boundary are retained, not clipped to October 31.


def test_real_aft_prediction_uses_explicit_raw_log_time_and_preserves_index(fitted_aft):
    rows = _prediction_rows()
    result = aft.predict_aft_dates(fitted_aft, rows)
    raw = fitted_aft.predict(aft.feature_frame(rows, use_station=True), prediction_type="RawFormulaVal")
    expected_days = np.maximum(1, np.floor(np.exp(raw)+.5)).astype(int)
    assert ((result-rows.entry_date).dt.days.to_numpy() == expected_days).all()
    assert result.index.equals(rows.index)
    assert result.notna().all()
    assert not np.array_equal(expected_days, np.floor(raw+.5).astype(int))
    meta = aft.aft_training_metadata(fitted_aft)
    assert meta["observed_events"] == 27
    assert meta["right_censored"] == 9
    assert meta["training_years"] == [2017, 2018, 2019]
    assert meta["training_max_year"] == 2019


def test_predictions_do_not_read_external_outcomes_or_future_followup(fitted_aft):
    rows = _prediction_rows()
    original = aft.predict_aft_dates(fitted_aft, rows)
    changed = rows.copy()
    changed["first_recorded_event_date"] = pd.Timestamp("2099-01-01")
    changed["last_assessment_date"] = pd.Timestamp("2100-01-01")
    changed["cohort_included"] = False
    changed["previous_visit_gap_days"] = -1
    changed["weather_after_issue"] = -999
    pd.testing.assert_series_equal(original, aft.predict_aft_dates(fitted_aft, changed))


def test_native_model_serialization_keeps_predictions_and_training_provenance(fitted_aft, tmp_path):
    path = tmp_path/"model.cbm"
    fitted_aft.save_model(path)
    loaded = CatBoostRegressor()
    loaded.load_model(path)
    assert aft.aft_training_metadata(loaded) == aft.aft_training_metadata(fitted_aft)
    pd.testing.assert_series_equal(aft.predict_aft_dates(loaded, _prediction_rows()),
                                   aft.predict_aft_dates(fitted_aft, _prediction_rows()))


def test_shuffle_cannot_change_training_time_order_or_fitted_predictions(fitted_aft):
    shuffled = aft.fit_aft(_registry().sample(frac=1, random_state=17), 2020)
    pd.testing.assert_series_equal(aft.predict_aft_dates(shuffled, _prediction_rows()),
                                   aft.predict_aft_dates(fitted_aft, _prediction_rows()))


def test_stage_candidate_fits_without_station_or_future_stage_columns():
    model = aft.fit_aft(_registry(), 2020, use_station=False)
    assert model.feature_names_ == aft.ENTRY_STAGE_FEATURES
    rows = _prediction_rows().drop(columns="cluster")
    assert aft.predict_aft_dates(model, rows).notna().all()
    with pytest.raises(ValueError, match="differs"):
        aft.predict_aft_dates(model, rows, use_station=True)


def test_model_does_not_silently_apply_a_future_bundle_to_an_earlier_year(fitted_aft):
    rows = _prediction_rows()
    rows["season"] = 2019
    with pytest.raises(ValueError, match="external year"):
        aft.predict_aft_dates(fitted_aft, rows)
    rows = _prediction_rows()
    rows["target"] = "another_insect"
    with pytest.raises(ValueError, match="target"):
        aft.predict_aft_dates(fitted_aft, rows)


def test_calendar_uses_observed_dates_only_and_group_threshold_has_global_fallback():
    train = _registry()
    calendar = aft.fit_calendar(train, 2020)
    observed = train[train.first_recorded_event_date.notna()]
    assert calendar["global_median_doy"] == int(np.floor(observed.first_recorded_event_date.dt.dayofyear.median()+.5))
    assert calendar["groups"]["a"]["events"] == 18
    assert calendar["groups"]["b"]["events"] == 9
    rows = _prediction_rows()
    predicted = aft.predict_calendar_dates(calendar, rows)
    assert predicted.loc[22].dayofyear == calendar["global_median_doy"]
    assert predicted.loc[8].dayofyear == calendar["groups"]["a"]["median_doy"]
    assert calendar["right_censored"] == 9
    assert json.loads(json.dumps(calendar)) == calendar
    sparse = train.copy()
    sparse["cluster"] = "well_sampled"
    sparse.loc[sparse.index[:2], "cluster"] = "sparse"
    assert "sparse" not in aft.fit_calendar(sparse, 2020)["groups"]


def test_calendar_half_up_and_current_year_rejection():
    train = _registry().iloc[:2].copy()
    train["first_recorded_event_date"] = pd.to_datetime(["2017-05-10", "2017-05-11"])
    result = aft.fit_calendar(train, 2020, use_station=False)
    assert result["global_median_doy"] == pd.Timestamp("2017-05-11").dayofyear
    rows = _prediction_rows().drop(columns="cluster")
    assert aft.predict_calendar_dates(result, rows).dt.dayofyear.eq(result["global_median_doy"]).all()
    rows["season"] = 2021
    with pytest.raises(ValueError, match="external year"):
        aft.predict_calendar_dates(result, rows)


def test_global_calendar_has_no_fictitious_station_or_stage_requirement():
    train = _registry().drop(columns=["cluster", "entry_egg_presence", "entry_adult_presence"])
    result = aft.fit_calendar(train, 2020, use_station=False)
    rows = _prediction_rows().drop(columns=["cluster", "entry_egg_presence", "entry_adult_presence"])
    assert aft.predict_calendar_dates(result, rows).notna().all()
    assert result["groups"] == {}


def test_noncohort_rows_and_zero_followup_exclusions_are_audited():
    rows = _registry()
    rows.loc[0, "cohort_included"] = False
    rows.loc[11, "last_assessment_date"] = rows.entry_date.iloc[11]
    model = aft.fit_aft(rows, 2020)
    metadata = aft.aft_training_metadata(model)
    assert metadata["received_rows"] == 36
    assert metadata["noncohort_rows"] == 1
    assert metadata["cohort_rows"] == 35
    assert metadata["excluded_zero_followup"] == 1
    assert metadata["used_rows"] == 34
