from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agro_phenology.early_warning_core import CALENDAR_FEATURES, EPISODE_FEATURES
from agro_phenology.early_warning_cycle2_models import (
    ALPHA_GRID,
    CLASS_NAMES,
    CLASS_ORDER,
    RAW_LOGIT_COLUMNS,
    SHALLOW_CATBOOST_PARAMS,
    C6Bundle,
    align_class_matrix,
    build_expanding_c0_oof,
    catboost_correction_logits,
    combine_c6_logits,
    c0_raw_logits,
    fit_calendar_catboost_control,
    fit_calibration_control,
    fit_catboost_correction,
    fit_weather_catboost_control,
    load_c6_bundle,
    predict_c6_from_baseline,
    predict_calibration_control,
    save_c6_bundle,
    stable_softmax,
)
from agro_phenology.early_warning_models import Policy, _model_rows, fit_model, simulate_policy


class _CorrectionThatMustNotRun:
    classes_ = np.asarray(CLASS_ORDER)

    def predict(self, *_args, **_kwargs):
        raise AssertionError("alpha=0 must not evaluate the correction")


class _FixedCorrection:
    classes_ = np.asarray([1, 2, 0])

    def predict(self, values, prediction_type=None):
        assert prediction_type == "RawFormulaVal"
        # Source columns are class 1, class 2, class 0.
        strength = np.asarray(values.iloc[:, 0], dtype=float)
        return np.column_stack([np.zeros(len(values)), strength, np.zeros(len(values))])


class _ShuffledC0:
    classes_ = np.asarray([2, 0, 1])

    def decision_function(self, values):
        x = np.asarray(values.iloc[:, 0], dtype=float)
        # Source columns are class 2, class 0, class 1.
        return np.column_stack([x + 2.0, x, x + 1.0])


def _synthetic_decisions(years: range | list[int], repetitions: int = 3) -> pd.DataFrame:
    records = []
    target_names = ["no_record_in_horizon", "imminent", "actionable"]
    for year in years:
        for repetition in range(repetitions):
            for class_id, target_name in enumerate(target_names):
                issue_date = pd.Timestamp(year=year, month=6, day=2 + repetition * 4 + class_id)
                angle = 2 * np.pi * issue_date.dayofyear / 365.25
                records.append(
                    {
                        "field_season": f"f-{year}-{repetition}-{class_id}",
                        "season": year,
                        "issue_date": issue_date,
                        "label_interval_end": issue_date + pd.Timedelta(days=10),
                        "target_class": target_name,
                        "target_observable": True,
                        "service_active": True,
                        "candidate_comparison_complete": True,
                        "common_weather_complete": True,
                        "episode_weather_complete": True,
                        "doy_sin1": np.sin(angle),
                        "doy_cos1": np.cos(angle),
                        "doy_sin2": np.sin(2 * angle),
                        "doy_cos2": np.cos(2 * angle),
                        **{
                            feature: float(year - 2000) + repetition / 10 + class_id / 100
                            for feature in EPISODE_FEATURES
                        },
                    }
                )
    return pd.DataFrame(records)


def test_fixed_research_constants() -> None:
    assert CLASS_ORDER == (0, 1, 2)
    assert ALPHA_GRID == (0.0, 0.1, 0.25, 0.5, 1.0)
    assert SHALLOW_CATBOOST_PARAMS == {
        "iterations": 200,
        "depth": 2,
        "learning_rate": 0.03,
        "l2_leaf_reg": 30.0,
    }


def test_class_mapping_and_stable_softmax() -> None:
    source = np.asarray([[11.0, 22.0, 0.0], [1.0, 2.0, 3.0]])
    aligned = align_class_matrix(source, [1, 2, 0])
    np.testing.assert_array_equal(aligned, [[0.0, 11.0, 22.0], [3.0, 1.0, 2.0]])
    probability = stable_softmax([[10000.0, 10001.0, 9999.0]])
    assert np.isfinite(probability).all()
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="Expected model classes"):
        align_class_matrix(source[:, :2], [0, 2])
    raw = c0_raw_logits(_ShuffledC0(), pd.DataFrame({"x": [0.0, 4.0]}))
    np.testing.assert_array_equal(raw, [[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]])


def test_alpha_zero_is_bit_exact_c0_and_does_not_call_correction() -> None:
    base_logits = np.asarray([[0.2, -0.1, 0.7], [-0.4, 0.3, 0.1]])
    # Use non-softmax values to prove that the exact saved C0 probability
    # output is retained instead of reconstructed.
    base_probabilities = np.asarray([[0.123, 0.234, 0.643], [0.444, 0.333, 0.223]])
    features = pd.DataFrame({"weather": [2.0, np.nan]})
    result = predict_c6_from_baseline(
        base_logits=base_logits,
        base_probabilities=base_probabilities,
        correction_model=_CorrectionThatMustNotRun(),
        correction_values=features,
        alpha=0.0,
    )
    np.testing.assert_array_equal(result.base_logits, base_logits)
    np.testing.assert_array_equal(result.combined_logits, base_logits)
    np.testing.assert_array_equal(result.probabilities, base_probabilities)
    np.testing.assert_array_equal(result.correction_logits, np.zeros_like(base_logits))
    np.testing.assert_array_equal(result.used_c0_fallback, [False, True])
    np.testing.assert_array_equal(
        combine_c6_logits(base_logits, np.full_like(base_logits, np.nan), 0.0),
        base_logits,
    )
    days = pd.date_range("2024-06-01", periods=2, freq="D")
    policy_frame = pd.DataFrame(
        {
            "field_season": "one-field",
            "season": 2024,
            "issue_date": days,
            "issued_at": days,
            "service_active": True,
            "evaluation_field_day": True,
            "target_class": "no_record_in_horizon",
            "target_observable": True,
            "days_to_first_recorded_event": np.nan,
            "warnable_first_event": False,
            "coordinate_scope": "A_direct",
            "previous_visit_gap_days": 1.0,
            "common_weather_complete": [True, False],
            "episode_weather_complete": [True, False],
            "candidate_comparison_complete": [True, False],
        }
    )
    policy = Policy(0.5, 7, 15)
    c0_states = simulate_policy(policy_frame, pd.Series(base_probabilities[:, 2]), policy)
    alpha_zero_states = simulate_policy(
        policy_frame, pd.Series(result.actionable_probability), policy
    )
    columns = ["score", "message_issued", "alarm_active", "action_reason", "suppressed_repeat"]
    pd.testing.assert_frame_equal(c0_states[columns], alpha_zero_states[columns], check_exact=True)


def test_explicit_formula_and_missing_weather_fallback() -> None:
    base_logits = np.asarray([[2.0, 0.0, -2.0], [1.0, 0.0, -1.0], [0.5, 0.0, -0.5]])
    base_probabilities = stable_softmax(base_logits)
    values = pd.DataFrame({"correction_strength": [8.0, np.nan, 4.0]})
    result = predict_c6_from_baseline(
        base_logits=base_logits,
        base_probabilities=base_probabilities,
        correction_model=_FixedCorrection(),
        correction_values=values,
        alpha=0.5,
    )
    expected_correction = np.asarray([[0.0, 0.0, 8.0], [0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    expected_logits = base_logits + 0.5 * expected_correction
    np.testing.assert_allclose(result.correction_logits, expected_correction)
    np.testing.assert_allclose(result.combined_logits, expected_logits)
    np.testing.assert_allclose(result.probabilities[[0, 2]], stable_softmax(expected_logits[[0, 2]]))
    np.testing.assert_array_equal(result.probabilities[1], base_probabilities[1])
    np.testing.assert_array_equal(result.used_c0_fallback, [False, True, False])
    np.testing.assert_array_equal(combine_c6_logits(base_logits, expected_correction, 0.5), expected_logits)


def test_weather_drop_and_return_preserve_one_policy_state() -> None:
    days = pd.date_range("2024-06-01", periods=3, freq="D")
    base_logits = np.tile(np.asarray([[2.0, 0.0, -2.0]]), (3, 1))
    prediction = predict_c6_from_baseline(
        base_logits=base_logits,
        base_probabilities=stable_softmax(base_logits),
        correction_model=_FixedCorrection(),
        correction_values=pd.DataFrame({"weather": [8.0, np.nan, 8.0]}),
        alpha=1.0,
    )
    frame = pd.DataFrame(
        {
            "field_season": "one-field",
            "season": 2024,
            "issue_date": days,
            "issued_at": days,
            "service_active": True,
            "evaluation_field_day": True,
            "target_class": "no_record_in_horizon",
            "target_observable": True,
            "days_to_first_recorded_event": np.nan,
            "warnable_first_event": False,
            "coordinate_scope": "A_direct",
            "previous_visit_gap_days": 1.0,
            "common_weather_complete": [True, False, True],
            "episode_weather_complete": [True, False, True],
            "candidate_comparison_complete": [True, False, True],
        }
    )
    states = simulate_policy(frame, pd.Series(prediction.actionable_probability), Policy(0.5, 7, 15))
    np.testing.assert_array_equal(states["message_issued"], [True, False, False])
    assert states.loc[2, "action_reason"] == "suppressed_cooldown"
    assert bool(states.loc[2, "suppressed_repeat"])


def test_catboost_pool_baseline_is_added_once() -> None:
    rng = np.random.default_rng(20260910)
    features = pd.DataFrame({"episode_a": rng.normal(size=45), "episode_b": rng.normal(size=45)})
    target = np.tile(np.asarray(CLASS_ORDER), 15)
    baseline = rng.normal(scale=0.3, size=(len(features), len(CLASS_ORDER)))
    model = fit_catboost_correction(features, target, baseline, seed=73)
    correction = catboost_correction_logits(model, features)

    from catboost import Pool

    pool_with_baseline = Pool(features, baseline=baseline)
    raw_with_baseline = np.asarray(
        model.predict(pool_with_baseline, prediction_type="RawFormulaVal"), dtype=float
    )
    probability_with_baseline = np.asarray(model.predict_proba(pool_with_baseline), dtype=float)
    # CatBoost stores Pool baselines with float32 precision.
    np.testing.assert_allclose(raw_with_baseline, baseline + correction, atol=5e-8, rtol=0)
    np.testing.assert_allclose(
        probability_with_baseline,
        stable_softmax(baseline + correction),
        atol=5e-8,
        rtol=0,
    )
    params = model.get_params()
    for key, value in SHALLOW_CATBOOST_PARAMS.items():
        assert params[key] == value


def test_temporal_oof_has_no_future_labels_and_is_future_append_invariant() -> None:
    original = _synthetic_decisions(range(2010, 2014))
    first = build_expanding_c0_oof(original, [2010, 2013], seed=91)
    assert first.provenance.loc[0, "status"] == "skipped_missing_history_or_classes"
    predicted = first.provenance[first.provenance["status"].eq("predicted")]
    assert set(predicted["forecast_year"]) == {2011, 2012, 2013}
    assert predicted["temporal_order_verified"].all()
    assert (
        predicted["baseline_max_label_availability_date"]
        < predicted["forecast_start_date"]
    ).all()
    assert (first.predictions["baseline_fit_end_year"] < first.predictions["forecast_year"]).all()

    appended = pd.concat([original, _synthetic_decisions([2014])], ignore_index=True)
    second = build_expanding_c0_oof(appended, [2010, 2014], seed=91)
    comparable = second.predictions[second.predictions["forecast_year"].le(2013)]
    columns = ["source_index", "forecast_year", *RAW_LOGIT_COLUMNS]
    pd.testing.assert_frame_equal(
        first.predictions[columns].reset_index(drop=True),
        comparable[columns].reset_index(drop=True),
        check_exact=True,
    )


def test_calibration_control_has_identity_candidate() -> None:
    rng = np.random.default_rng(7)
    baseline = rng.normal(size=(60, 3))
    target = np.tile(np.asarray(CLASS_ORDER), 20)
    base_probability = stable_softmax(baseline)
    correction = fit_calibration_control(baseline, target, seed=8)
    identity = predict_calibration_control(
        baseline, base_probability, correction, alpha=0.0
    )
    np.testing.assert_array_equal(identity.probabilities, base_probability)
    adjusted = predict_calibration_control(
        baseline, base_probability, correction, alpha=0.25
    )
    assert np.isfinite(adjusted.correction_logits).all()
    np.testing.assert_allclose(adjusted.probabilities.sum(axis=1), 1.0)


def test_calendar_control_and_bundle_roundtrip(tmp_path) -> None:
    decisions = _synthetic_decisions(range(2010, 2014), repetitions=5)
    oof = build_expanding_c0_oof(decisions, [2010, 2013], seed=101)
    correction = fit_calendar_catboost_control(oof.predictions, seed=102)
    base_training = _model_rows(decisions, [2010, 2013])
    base_model = fit_model(
        base_training,
        list(CALENDAR_FEATURES),
        "logistic",
        {"C": 0.1},
        103,
    )
    bundle = C6Bundle(
        base_model=base_model,
        correction_model=correction,
        correction_kind="calendar_catboost",
        correction_features=tuple(CALENDAR_FEATURES),
        alpha=0.25,
        policy=Policy(0.42, 7, 15, "cycle2_test_policy"),
        metadata={"fold_id": "synthetic"},
    )
    sample = decisions.iloc[:12]
    before = bundle.predict(sample)
    bundle_dir = tmp_path / "bundle"
    save_c6_bundle(bundle, bundle_dir)
    loaded = load_c6_bundle(bundle_dir)
    after = loaded.predict(sample)
    np.testing.assert_allclose(after.base_logits, before.base_logits, atol=0, rtol=0)
    np.testing.assert_allclose(after.correction_logits, before.correction_logits, atol=1e-12, rtol=0)
    np.testing.assert_allclose(after.probabilities, before.probabilities, atol=1e-12, rtol=0)
    assert loaded.metadata == {"fold_id": "synthetic"}
    assert loaded.policy == Policy(0.42, 7, 15, "cycle2_test_policy")
    metadata = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    assert metadata["class_order"] == [0, 1, 2]
    assert metadata["class_names"] == list(CLASS_NAMES)
    assert metadata["alpha"] == 0.25
    assert metadata["policy"]["threshold"] == 0.42
    with pytest.raises(FileExistsError):
        save_c6_bundle(bundle, bundle_dir)

    metadata["class_names"] = ["wrong", *metadata["class_names"][1:]]
    (bundle_dir / "bundle.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="class-name mapping"):
        load_c6_bundle(bundle_dir)


def test_weather_bundle_roundtrip_preserves_mixed_availability_fallback(tmp_path) -> None:
    decisions = _synthetic_decisions(range(2010, 2014), repetitions=5)
    oof = build_expanding_c0_oof(decisions, [2010, 2013], seed=201)
    correction = fit_weather_catboost_control(oof.predictions, seed=202)
    base_training = _model_rows(decisions, [2010, 2013])
    base_model = fit_model(
        base_training,
        list(CALENDAR_FEATURES),
        "logistic",
        {"C": 0.1},
        203,
    )
    bundle = C6Bundle(
        base_model=base_model,
        correction_model=correction,
        correction_kind="weather_catboost",
        correction_features=tuple(EPISODE_FEATURES),
        alpha=0.5,
        availability_column="episode_weather_complete",
        policy=Policy(0.31, 7, 15, "cycle2_weather_roundtrip"),
    )
    sample = decisions.iloc[:12].copy()
    missing = sample.index[::2]
    sample.loc[missing, "episode_weather_complete"] = False
    sample.loc[missing, list(EPISODE_FEATURES)] = np.nan

    before = bundle.predict(sample)
    bundle_dir = tmp_path / "weather_bundle"
    save_c6_bundle(bundle, bundle_dir)
    loaded = load_c6_bundle(bundle_dir)
    after = loaded.predict(sample)

    assert before.used_c0_fallback.any()
    assert (~before.used_c0_fallback).any()
    np.testing.assert_array_equal(after.used_c0_fallback, before.used_c0_fallback)
    np.testing.assert_allclose(after.base_logits, before.base_logits, atol=0, rtol=0)
    np.testing.assert_allclose(after.correction_logits, before.correction_logits, atol=1e-12, rtol=0)
    np.testing.assert_allclose(after.probabilities, before.probabilities, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(
        after.probabilities[after.used_c0_fallback],
        before.probabilities[before.used_c0_fallback],
    )
    assert loaded.availability_column == "episode_weather_complete"
    assert loaded.policy == Policy(0.31, 7, 15, "cycle2_weather_roundtrip")
