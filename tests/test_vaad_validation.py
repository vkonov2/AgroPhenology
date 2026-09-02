from __future__ import annotations

import json
from datetime import date

import pandas as pd

from agro_phenology.vaad_validation import (
    build_onset_events,
    build_polyakov_cases,
    build_polyakov_exact_interval_cases,
    build_polyakov_point_cases,
    hutton_date_permutation_test,
    polyakov_date_permutation_test,
    prepare_potato_visits,
    wilson_interval,
)


def _visit(
    day: str,
    outcome: bool,
    stage: int,
    *,
    field_uid: str = "field-1",
) -> dict[str, object]:
    return {
        "field_uid": field_uid,
        "observation_date": pd.Timestamp(day).date(),
        "season": int(day[:4]),
        "latitude": 56.5,
        "longitude": 24.5,
        "late_blight_detected": outcome,
        "explicit_no_harmful_organisms": False,
        "growth_stage_codes": (stage,),
        "growth_stage_labels": str(stage),
        "geolocation_stratum": "direct",
        "municipality": "municipality",
        "parish": "parish",
        "row_count": 1,
        "observation_ids": day,
    }


def test_polyakov_interval_uses_vegetative_branch_not_tuber_branch() -> None:
    visits = pd.DataFrame(
        [
            _visit("2024-06-01", False, 31),
            _visit("2024-06-08", False, 45),
            _visit("2024-06-15", False, 51),
            _visit("2024-06-22", True, 61),
        ]
    )
    events = build_onset_events(visits)
    cases = build_polyakov_cases(visits, events)
    assert len(cases) == 1
    assert cases.iloc[0]["phenophase_interval_start"] == date(2024, 6, 1)
    assert cases.iloc[0]["phenophase_interval_end"] == date(2024, 6, 15)
    exact_cases = build_polyakov_exact_interval_cases(visits, events)
    assert len(exact_cases) == 1
    assert exact_cases.iloc[0]["phenophase_interval_start"] == date(2024, 6, 1)


def test_bbch51_observed_on_detection_day_is_not_a_point_activation_case() -> None:
    visits = pd.DataFrame(
        [
            _visit("2024-06-01", False, 31),
            _visit("2024-06-15", True, 51),
        ]
    )
    events = build_onset_events(visits)
    assert build_polyakov_point_cases(visits, events).empty


def test_prepare_visits_excludes_zero_prevalence_late_blight(tmp_path) -> None:
    positive_zero = json.dumps(
        [
            {
                "organism_id": 640,
                "name": "Kartupeļu lakstu puve",
                "details_raw": "Izplatība: 0.00%, Attīstības pakāpe: 0%",
            }
        ],
        ensure_ascii=False,
    )
    rows = []
    for index, (day, detected, organisms) in enumerate(
        [
            ("2024-06-01", False, "[]"),
            ("2024-06-08", True, positive_zero),
        ],
        start=1,
    ):
        rows.append(
            {
                "observation_id": f"obs-{index}",
                "observation_date": day,
                "crop_code": 166,
                "crop_name": "Kartupeļi",
                "crop_stage_raw": "stage",
                "growth_stage_code": 31,
                "growth_stage": "stage",
                "detected_organisms": organisms,
                "organisms_raw": "",
                "late_blight_detected": detected,
                "explicit_no_harmful_organisms": False,
                "field_uid": "field-1",
                "latitude": 56.5,
                "longitude": 24.5,
                "geolocation_source": "lauksuid_direct",
                "municipality": "municipality",
                "parish": "parish",
            }
        )
    path = tmp_path / "observations.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    visits, qc = prepare_potato_visits(path, "direct")
    assert len(visits) == 1
    assert qc["excluded_zero_prevalence_positive"] == 1


def test_wilson_interval_is_bounded_for_extreme_rates() -> None:
    assert wilson_interval(0, 3)[0] == 0.0
    assert wilson_interval(3, 3)[1] == 1.0


def test_empty_permutation_results_are_strict_json() -> None:
    empty = pd.DataFrame()
    payload = {
        "hutton": hutton_date_permutation_test(
            empty,
            empty,
            lookback=14,
            repetitions=10,
        ),
        "polyakov": polyakov_date_permutation_test(empty, repetitions=10),
    }
    encoded = json.dumps(payload, allow_nan=False)
    assert '"observed_rate": null' in encoded
