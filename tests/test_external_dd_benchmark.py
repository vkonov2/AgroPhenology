from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agro_phenology import external_dd_benchmark as dd


@pytest.mark.parametrize("curve", ["sine", "triangle"])
@pytest.mark.parametrize("cutoff", ["horizontal", "vertical", "intermediate"])
def test_analytic_integral_matches_independent_dense_numerical_curve(curve, cutoff):
    # Fixed synthetic shapes cover every threshold relation and a high tail.
    minima = np.array([20, 40, 55, 60, 75, 30, 50, 70], dtype=float)
    maxima = np.array([40, 65, 65, 85, 100, 140, 50, 70], dtype=float)
    fraction = (np.arange(200_000)+.5)/200_000
    if curve == "sine":
        cycle = .5-.5*np.cos(2*np.pi*fraction)
    else:
        cycle = 1-np.abs(2*fraction-1)
    temperatures = minima[:, None] + (maxima-minima)[:, None]*cycle
    lower_area = np.maximum(temperatures-50, 0).mean(axis=1)
    upper_area = np.maximum(temperatures-70, 0).mean(axis=1)
    vertical_removal = 20*(temperatures>70).mean(axis=1)
    horizontal = lower_area-upper_area
    expected = (horizontal if cutoff == "horizontal" else horizontal-vertical_removal
                if cutoff == "vertical" else horizontal-np.minimum(upper_area, vertical_removal))
    actual = dd.estimate_degree_days(minima, maxima, curve=curve, cutoff=cutoff)
    np.testing.assert_allclose(actual, np.maximum(expected, 0), atol=.00015)


@pytest.mark.parametrize("curve", ["sine", "triangle"])
def test_degree_day_units_are_affine_temperature_invariant(curve):
    lo, hi = np.array([30., 55., 72.]), np.array([80., 85., 105.])
    fahrenheit = dd.estimate_degree_days(lo, hi, curve=curve)
    celsius = dd.estimate_degree_days((lo-32)*5/9, (hi-32)*5/9,
                                     lower=(50-32)*5/9, upper=(70-32)*5/9, curve=curve)
    np.testing.assert_allclose(celsius, fahrenheit*5/9, atol=1e-12)


def test_missing_input_is_unknown_and_constant_upper_boundary_is_not_above_upper():
    assert np.isnan(dd.estimate_degree_days([np.nan], [80])[0])
    assert np.isnan(dd.estimate_degree_days([50], [80], double=True)[0])
    assert dd.estimate_degree_days([70], [70], cutoff="vertical")[0] == 20
    assert dd.estimate_degree_days([80], [80], cutoff="vertical")[0] == 0
    assert dd.estimate_degree_days([45], [45])[0] == 0


@pytest.mark.parametrize("kwargs", [
    {"curve": "invalid"}, {"cutoff": "invalid"}, {"lower": 70, "upper": 50},
    {"lower": np.nan}, {"upper": np.inf},
])
def test_unknown_method_or_thresholds_are_rejected(kwargs):
    with pytest.raises(ValueError):
        dd.estimate_degree_days([40], [80], **kwargs)


def test_invalid_minimum_maximum_order_rejected_and_unknown_next_minimum_not_filled():
    with pytest.raises(ValueError, match="minimum exceeds maximum"):
        dd.estimate_degree_days([90], [80])
    with pytest.raises(ValueError, match="Next minimum"):
        dd.estimate_degree_days([40], [80], double=True, next_min=[90])
    actual = dd.estimate_degree_days([40, 42], [70, 80], double=True, next_min=[42, np.nan])
    assert np.isfinite(actual[0])
    assert np.isnan(actual[1])


@pytest.mark.parametrize("curve", ["sine", "triangle"])
@pytest.mark.parametrize("cutoff", ["horizontal", "vertical", "intermediate"])
def test_equal_next_minimum_reduces_double_exactly_to_single(curve, cutoff):
    lo, hi = np.array([30, 45, 72]), np.array([90, 60, 110])
    single = dd.estimate_degree_days(lo, hi, curve=curve, cutoff=cutoff)
    double = dd.estimate_degree_days(lo, hi, curve=curve, cutoff=cutoff, double=True, next_min=lo)
    np.testing.assert_allclose(single, double, atol=1e-12)


def test_double_uses_the_following_minimum_not_a_previous_or_repeated_minimum():
    lower = np.array([30, 40, 50.])
    upper = np.array([80, 80, 80.])
    following = np.array([40, 50, np.nan])
    predictions = dd.estimate_degree_days(lower, upper, curve="triangle", double=True, next_min=following)
    altered = dd.estimate_degree_days(lower, upper, curve="triangle", double=True, next_min=[60, 50, np.nan])
    assert altered[0] != predictions[0]
    assert altered[1] == predictions[1]
    assert np.isnan(predictions[2])


def test_intermediate_removal_is_bounded_by_vertical_removal_even_extreme_tail():
    vertical = dd.estimate_degree_days([20], [180], cutoff="vertical")[0]
    intermediate = dd.estimate_degree_days([20], [180], cutoff="intermediate")[0]
    assert vertical > 0
    assert intermediate == pytest.approx(vertical)


@pytest.fixture
def synthetic_reference(tmp_path):
    raw = tmp_path/"raw"
    raw.mkdir()
    dates = pd.date_range("2020-01-01", "2020-12-31")
    weather = pd.DataFrame({"station": "synthetic", "date": dates.strftime("%Y-%m-%d"), "tmin": 60, "tmax": 60})
    weather.to_csv(raw/"data__espartoa-weather-2020.csv", index=False)
    answers = pd.DataFrame({"date": dates.strftime("%m/%d/%Y"), "tmin": 60, "tmax": 60})
    # Constant temperature at 60 F gives ten degree-days for all formulas.
    for method in dd.METHODS:
        answers[method] = 10.0
    answers.to_csv(raw/"data__ucipm_results__ucipm_low50_high70_all.csv", index=False)
    (raw/"README.md").write_text("Synthetic reference for tests only")
    (raw/"LICENSE").write_text("Synthetic fixture")
    return raw


def test_reference_comparison_keeps_all_rows_and_abstains_on_only_missing_boundary(synthetic_reference):
    predictions, metrics = dd._compute(synthetic_reference)
    assert len(predictions) == 4392
    assert predictions.computable.sum() == 4386
    assert set(metrics.computable_days) == {365, 366}
    assert metrics.mae_f_days.eq(0).all()
    missing = predictions[~predictions.computable]
    assert len(missing) == 6
    assert missing.date.eq(pd.Timestamp("2020-12-31")).all()
    assert missing.method.str.startswith("dbl").all()
    assert missing.missing_reason.eq("next_day_minimum_not_in_input").all()


def test_run_immutable_and_verify_detects_mutation_or_forged_rehashed_metric(synthetic_reference, tmp_path):
    root = tmp_path/"run"
    assert dd.run(synthetic_reference, root)["status"] == "complete"
    assert dd.verify(root)["status"] == "passed"
    with pytest.raises(FileExistsError):
        dd.run(synthetic_reference, root)
    metrics = pd.read_parquet(root/"metrics.parquet")
    metrics.loc[0, "mae_f_days"] = .5
    metrics.to_parquet(root/"metrics.parquet", index=False)
    assert dd.verify(root)["mismatches"] == ["metrics.parquet"]
    manifest_path = root/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file_inventory"]["metrics.parquet"] = dd.digest_file(root/"metrics.parquet")
    dd.write_json(manifest_path, manifest)
    checked = dd.verify(root)
    assert checked["status"] == "failed"
    assert checked["mismatches"] == ["recompute:metrics"]


def test_reference_schema_misalignment_never_silently_matches_rows(synthetic_reference):
    path = synthetic_reference/"data__ucipm_results__ucipm_low50_high70_all.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "tmin"] = 59
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="temperatures differ"):
        dd._compute(synthetic_reference)
