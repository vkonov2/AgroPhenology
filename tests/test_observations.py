from datetime import date

import pandas as pd

from agro_phenology.observations import apply_column_mapping, validate_observations


def base_frame() -> pd.DataFrame:
    return apply_column_mapping(
        pd.DataFrame(
            [
                {"lat": 55.7, "lon": 37.6, "date": "2023-06-20", "pest": "яблонная плодожорка", "stage": "Массовый выход гусениц"},
                {"lat": 55.7, "lon": 37.6, "date": "2023-06-20", "pest": "яблонная плодожорка", "stage": "имаго"},
                {"lat": 95, "lon": 37.6, "date": "bad", "pest": "яблонная плодожорка", "stage": "отрождение гусениц"},
            ]
        ),
        {"lat": "latitude", "lon": "longitude", "date": "observation_date", "pest": "pest_name", "stage": "observed_stage"},
    )


def test_invalid_coordinates_dates_and_non_target_stage() -> None:
    report = validate_observations(
        base_frame(), "mass_larval_hatch", global_accumulation_start_date=date(2023, 4, 15)
    )
    assert len(report.accepted) == 1
    assert "known_non_target_stage" in report.excluded.iloc[0]["exclusion_reason"]
    assert "invalid_latitude" in report.errors.iloc[0]["error_reason"]
    assert "invalid_observation_date" in report.errors.iloc[0]["error_reason"]


def test_missing_start_date_is_explicitly_excluded() -> None:
    report = validate_observations(base_frame().iloc[[0]], "mass_larval_hatch")
    assert report.accepted.empty
    assert report.excluded.iloc[0]["exclusion_reason"] == "missing_accumulation_start_date"


def test_duplicate_rows_are_retained_and_marked() -> None:
    frame = pd.concat([base_frame().iloc[[0]], base_frame().iloc[[0]]], ignore_index=True)
    report = validate_observations(frame, "mass_larval_hatch", "2023-04-15")
    assert len(report.accepted) == 2
    assert report.accepted["duplicate_record"].all()


def test_invalid_supplied_start_is_not_replaced_by_global_date() -> None:
    frame = base_frame().iloc[[0]].copy()
    frame.loc[:, "accumulation_start_date"] = "not-a-date"
    report = validate_observations(frame, "mass_larval_hatch", "2023-04-15")
    assert report.accepted.empty
    assert "invalid_accumulation_start_date" in report.errors.iloc[0]["error_reason"]
    assert report.errors.iloc[0]["accumulation_start_rule"] == "invalid"


def test_different_pest_is_excluded() -> None:
    frame = base_frame().iloc[[0]].copy()
    frame.loc[:, "pest_name"] = "другой вредитель"
    report = validate_observations(
        frame, "mass_larval_hatch", "2023-04-15", target_pest_name="яблонная плодожорка"
    )
    assert report.accepted.empty
    assert report.excluded.iloc[0]["exclusion_reason"] == "different_pest"
