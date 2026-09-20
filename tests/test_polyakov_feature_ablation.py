from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agro_phenology.polyakov_feature_ablation import (
    BASE_FEATURES,
    FIXED_FEATURE,
    MODEL_SPECS,
    PLUS_POLYAKOV_FEATURES,
    SOURCE_FILES,
    calendar_window_score,
    compare_saved_baseline_scores,
    complete_external_folds,
    output_dir,
    shared_training_mask,
    three_valued_or,
    validate_causal_cutoffs,
)


def test_source_snapshot_covers_direct_local_dependencies() -> None:
    assert "src/agro_phenology/early_warning_cycle2_pipeline.py" in SOURCE_FILES
    assert "src/agro_phenology/era_hutton_backfill.py" in SOURCE_FILES
    assert "src/agro_phenology/polyakov_feature_ablation.py" in SOURCE_FILES
    assert "tests/test_polyakov_feature_ablation.py" in SOURCE_FILES


def _training_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "season": [2017, 2017, 2017, 2018, 2017, 2017],
            "target_observable": [True, False, True, True, True, True],
            "service_active": [True, True, False, True, True, True],
            "target_class": [
                "actionable",
                "imminent",
                "no_record_in_horizon",
                "actionable",
                "unknown",
                "imminent",
            ],
            "candidate_comparison_complete": [True, True, True, True, True, False],
            "episode_weather_complete": [True, True, True, True, True, True],
            FIXED_FEATURE: [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    for index, feature in enumerate(BASE_FEATURES):
        frame[feature] = float(index + 1)
    return frame


def test_plus_polyakov_models_add_exactly_one_feature() -> None:
    assert tuple(MODEL_SPECS["C4"]["features"]) == BASE_FEATURES
    assert tuple(MODEL_SPECS["C5"]["features"]) == BASE_FEATURES
    assert tuple(MODEL_SPECS["C4-P"]["features"]) == PLUS_POLYAKOV_FEATURES
    assert tuple(MODEL_SPECS["C5-P"]["features"]) == PLUS_POLYAKOV_FEATURES
    assert PLUS_POLYAKOV_FEATURES[:-1] == BASE_FEATURES
    assert PLUS_POLYAKOV_FEATURES[-1] == FIXED_FEATURE
    assert MODEL_SPECS["C4"]["kind"] == MODEL_SPECS["C4-P"]["kind"]
    assert MODEL_SPECS["C5"]["kind"] == MODEL_SPECS["C5-P"]["kind"]


def test_shared_training_mask_is_observable_complete_case() -> None:
    frame = _training_frame()
    mask = shared_training_mask(frame, [2010, 2017])
    assert mask.tolist() == [True, False, False, False, False, False]


def test_shared_training_mask_rejects_missing_fixed_feature() -> None:
    frame = _training_frame().drop(columns=FIXED_FEATURE)
    with pytest.raises(ValueError, match="Ablation matrix lacks"):
        shared_training_mask(frame, [2010, 2017])


def test_shared_training_mask_prefers_frozen_comparison_scope() -> None:
    frame = _training_frame().iloc[[0]].copy()
    frame["candidate_comparison_complete"] = True
    frame["frozen_candidate_comparison_complete"] = False
    assert not shared_training_mask(frame, [2010, 2017]).any()


def test_three_valued_or_does_not_turn_missing_weather_into_negative() -> None:
    calendar = pd.Series([1.0, 0.0, 0.0, 1.0, np.nan, np.nan])
    polyakov = pd.Series([np.nan, 1.0, 0.0, 0.0, 1.0, np.nan])
    result = three_valued_or(calendar, polyakov)
    assert result.iloc[:5].tolist() == [1.0, 1.0, 0.0, 1.0, 1.0]
    assert np.isnan(result.iloc[5])


def test_calendar_window_uses_train_registry_only() -> None:
    train = pd.DataFrame(
        {
            "warnable_first_event": [True, True, False],
            "first_recorded_event_date": ["2018-07-10", "2019-07-10", "2019-01-01"],
        }
    )
    evaluation = pd.DataFrame(
        {"issue_date": pd.to_datetime(["2021-07-09", "2021-07-10", "2021-07-11"])}
    )
    score, bounds = calendar_window_score(evaluation, train)
    assert bounds == (191, 191)
    assert score.tolist() == [0.0, 1.0, 0.0]


def test_validate_causal_cutoffs_enforces_two_day_lag() -> None:
    valid = pd.DataFrame(
        {
            "issue_date": pd.to_datetime(["2020-07-10", "2020-07-11"]),
            "feature_cutoff_era_episode_date": pd.to_datetime(
                ["2020-07-08", "2020-07-08"]
            ),
        }
    )
    validate_causal_cutoffs(valid)
    invalid = valid.copy()
    invalid.loc[0, "feature_cutoff_era_episode_date"] = pd.Timestamp("2020-07-09")
    with pytest.raises(AssertionError, match="later than issue_date - 2"):
        validate_causal_cutoffs(invalid)


def test_complete_external_folds_excludes_partial_and_checks_order() -> None:
    contract = {
        "rolling_origin_folds": [
            {
                "id": "test_2020",
                "train_years": [2010, 2017],
                "validation_years": [2018, 2019],
                "test_years": [2020, 2020],
            },
            {
                "id": "test_2026_partial",
                "train_years": [2010, 2023],
                "validation_years": [2024, 2025],
                "test_years": [2026, 2026],
                "incomplete": True,
            },
        ]
    }
    assert [fold["id"] for fold in complete_external_folds(contract)] == ["test_2020"]
    bad = {
        "rolling_origin_folds": [
            {
                "id": "bad",
                "train_years": [2010, 2019],
                "validation_years": [2019, 2019],
                "test_years": [2020, 2020],
            }
        ]
    }
    with pytest.raises(AssertionError, match="Non-causal"):
        complete_external_folds(bad)


def test_output_is_always_a_new_named_subdirectory(tmp_path: Path) -> None:
    assert output_dir(tmp_path) == tmp_path / "polyakov_ablation"


def test_saved_baseline_reproduction_checks_score_and_nan_mask() -> None:
    test = pd.DataFrame(
        {
            "field_season": ["a", "a", "a"],
            "issue_date": pd.to_datetime(["2020-07-01", "2020-07-02", "2020-07-03"]),
            "service_active": [True, True, False],
            "frozen_episode_weather_complete": [True, False, True],
        }
    )
    saved = pd.DataFrame(
        {
            "field_season": ["a", "a", "a"],
            "issue_date": test["issue_date"],
            "model_code": ["C4", "C4", "C4"],
            "fold_id": ["test_2020", "test_2020", "test_2020"],
            "evaluation_scope": ["service_calendar"] * 3,
            "score": [0.25, np.nan, np.nan],
        }
    )
    audit = compare_saved_baseline_scores(
        saved,
        model_code="C4",
        fold_id="test_2020",
        test_frame=test,
        score=pd.Series([0.25 + 1e-12, 0.7, np.nan]),
    )
    assert audit["status"] == "passed"
    assert audit["old_computed_rows"] == 1
    assert audit["newly_computed_rows"] == 1
    with pytest.raises(AssertionError, match="does not reproduce v3"):
        compare_saved_baseline_scores(
            saved,
            model_code="C4",
            fold_id="test_2020",
            test_frame=test,
            score=pd.Series([0.3, 0.7, np.nan]),
        )
