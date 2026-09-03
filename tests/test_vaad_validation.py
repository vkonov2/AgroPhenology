from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd

from agro_phenology.vaad_validation import (
    _cohort,
    _target_explicit_absence_text,
    annotate_holdout_structure,
    build_negative_seasons,
    build_onset_events,
    build_polyakov_cases,
    build_polyakov_exact_interval_cases,
    build_polyakov_point_cases,
    calendar_baseline_summary,
    evaluate_polyakov_cases,
    evaluate_polyakov_negative_seasons,
    hutton_alarm_burden,
    hutton_date_permutation_test,
    paired_binary_summary,
    polyakov_date_permutation_test,
    prepare_potato_visits,
    score_interval_uncertainty,
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
        "explicit_late_blight_absent": False,
        "zero_prevalence_late_blight": False,
        "generic_explicit_absence_from_text": False,
        "target_explicit_absence_from_text": False,
        "positive_late_blight_from_text": False,
        "growth_stage_codes": (stage,),
        "growth_stage_labels": str(stage),
        "geolocation_stratum": "direct",
        "municipality": "municipality",
        "parish": "parish",
        "row_count": 1,
        "observation_ids": day,
    }


def test_polyakov_interval_uses_only_assumed_pre_bbch30_stage() -> None:
    visits = pd.DataFrame(
        [
            _visit("2024-06-01", False, 22),
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


def test_canopy_bbch31_is_not_a_pre_budding_lower_bound() -> None:
    visits = pd.DataFrame(
        [
            _visit("2024-06-01", False, 31),
            _visit("2024-06-15", False, 51),
            _visit("2024-06-22", True, 61),
        ]
    )
    events = build_onset_events(visits)
    assert build_polyakov_cases(visits, events).empty
    assert build_polyakov_exact_interval_cases(visits, events).empty


def test_bbch51_observed_on_detection_day_is_not_a_point_activation_case() -> None:
    visits = pd.DataFrame(
        [
            _visit("2024-06-01", False, 31),
            _visit("2024-06-15", True, 51),
        ]
    )
    events = build_onset_events(visits)
    assert build_polyakov_point_cases(visits, events).empty


def test_prepare_visits_reclassifies_zero_prevalence_as_explicit_negative(tmp_path) -> None:
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
    assert len(visits) == 2
    second = visits.iloc[1]
    assert second["late_blight_detected"] is False or not bool(
        second["late_blight_detected"]
    )
    assert bool(second["explicit_late_blight_absent"])
    assert qc["excluded_zero_prevalence_positive"] == 0
    assert qc["reclassified_zero_prevalence_as_explicit_target_negative"] == 1


def test_target_negative_text_accepts_named_late_blight_phrase() -> None:
    assert _target_explicit_absence_text(
        "Kartupeļu lakstu puve nav konstatēta."
    )
    assert _target_explicit_absence_text("Lakstu puves pazīmes nav.")


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


def test_any_prior_explicit_does_not_make_last_negative_explicit() -> None:
    first = _visit("2024-06-01", False, 22)
    first["explicit_late_blight_absent"] = True
    visits = pd.DataFrame(
        [
            first,
            _visit("2024-06-08", False, 31),
            _visit("2024-06-15", True, 51),
        ]
    )
    event = build_onset_events(visits).iloc[0]
    assert bool(event["explicit_negative_before_positive"])
    assert not bool(event["last_negative_is_explicit"])
    assert event["days_from_last_explicit_negative"] == 14


def test_negative_season_retains_location_for_holdout_annotation() -> None:
    generic = _visit("2024-06-01", False, 22)
    generic["explicit_no_harmful_organisms"] = True
    generic["explicit_late_blight_absent"] = True
    target_specific = _visit("2024-06-08", False, 31)
    target_specific["target_explicit_absence_from_text"] = True
    target_specific["explicit_late_blight_absent"] = True
    visits = pd.DataFrame([generic, target_specific])
    negative = build_negative_seasons(visits)
    annotated = annotate_holdout_structure(negative, visits)
    assert annotated.iloc[0]["municipality"] == "municipality"
    assert annotated.iloc[0]["parish"] == "parish"
    assert annotated.iloc[0]["explicit_negative_visit_count"] == 2
    assert annotated.iloc[0]["generic_explicit_negative_visit_count"] == 1
    assert annotated.iloc[0]["target_specific_negative_visit_count"] == 1
    assert bool(annotated.iloc[0]["last_negative_is_target_specific"])
    assert "distance_to_nearest_development_field_km" in annotated


def test_late_only_dense_followup_is_not_full_risk_season_coverage() -> None:
    visits = pd.DataFrame(
        [
            _visit("2024-07-20", False, 51),
            _visit("2024-08-01", False, 61),
            _visit("2024-08-15", False, 69),
        ]
    )
    negative = build_negative_seasons(visits).iloc[0]
    assert bool(negative["dense_late_season_followup"])
    assert not bool(negative["full_risk_season_coverage"])
    assert not bool(negative["strict_coverage"])


def test_regular_june_to_august_followup_is_full_risk_season_coverage() -> None:
    dates = list(pd.date_range("2024-06-15", "2024-08-10", freq="7D"))
    dates.append(pd.Timestamp("2024-08-15"))
    visits = pd.DataFrame(
        [_visit(day.strftime("%Y-%m-%d"), False, 51) for day in dates]
    )
    negative = build_negative_seasons(visits).iloc[0]
    assert bool(negative["dense_late_season_followup"])
    assert bool(negative["full_risk_season_coverage"])
    assert bool(negative["strict_coverage"])


def test_missing_admin_name_is_not_a_new_admin_area() -> None:
    development = _visit("2022-07-01", False, 51, field_uid="development")
    holdout = _visit("2024-07-01", False, 51, field_uid="holdout")
    holdout["municipality"] = ""
    holdout["parish"] = ""
    visits = pd.DataFrame([development, holdout])
    frame = pd.DataFrame(
        [
            {
                "field_uid": "holdout",
                "season": 2024,
                "latitude": 56.6,
                "longitude": 24.6,
                "municipality": "",
                "parish": "",
            }
        ]
    )
    annotated = annotate_holdout_structure(frame, visits)
    assert not bool(annotated.iloc[0]["municipality_known"])
    assert not bool(annotated.iloc[0]["parish_known"])
    assert _cohort(annotated, "holdout_2023_2025_new_municipality").empty
    assert _cohort(annotated, "holdout_2023_2025_new_parish").empty


def test_interval_uncertainty_scores_both_axes() -> None:
    d1 = date(2024, 7, 1)
    d2 = d1 + timedelta(days=1)
    scores = score_interval_uncertainty(
        [{d1, d2}, {d2}], d1 - timedelta(days=1), d2
    )
    assert scores == {
        "activation_any__onset_any": True,
        "activation_all__onset_any": True,
        "activation_any__onset_all": True,
        "activation_all__onset_all": False,
    }


def test_polyakov_permutation_never_crosses_seasons() -> None:
    rows = []
    for season in (2020, 2021):
        start = date(season, 7, 1)
        end = date(season, 7, 2)
        payload = [
            {
                "activation_date": f"{season}-06-01",
                "windows": [
                    {
                        "start": end.isoformat(),
                        "end": end.isoformat(),
                        "issued_on": start.isoformat(),
                    }
                ],
            }
        ]
        rows.append(
            {
                "season": season,
                "last_negative_date": start,
                "first_positive_date": end,
                "full_season_prediction_scenarios_json": json.dumps(payload),
            }
        )
    result = polyakov_date_permutation_test(
        pd.DataFrame(rows), repetitions=100, seed=7
    )
    assert result["observed_rate"] == 1.0
    assert result["null_mean_rate"] == 1.0
    assert result["permutation_p_one_sided"] == 1.0


def test_polyakov_permutation_does_not_score_issue_day_as_actionable() -> None:
    issue_day = date(2024, 7, 2)
    payload = [
        {
            "activation_date": "2024-06-01",
            "windows": [
                {
                    "start": issue_day.isoformat(),
                    "end": issue_day.isoformat(),
                    "issued_on": issue_day.isoformat(),
                }
            ],
        }
    ]
    frame = pd.DataFrame(
        [
            {
                "season": 2024,
                "last_negative_date": issue_day - timedelta(days=1),
                "first_positive_date": issue_day + timedelta(days=1),
                "full_season_prediction_scenarios_json": json.dumps(payload),
            }
        ]
    )
    result = polyakov_date_permutation_test(frame, repetitions=10, seed=7)
    assert result["observed_rate"] == 0.0


def test_hutton_burden_keeps_may_carry_in() -> None:
    dates = [value.date() for value in pd.date_range("2024-05-24", "2024-06-01")]
    weather = pd.DataFrame(
        {
            "field_uid": ["field-1"] * len(dates),
            "season": [2024] * len(dates),
            "date": dates,
            "hutton_period_status": ["fail"] * 7 + ["pass", "fail"],
        }
    )
    results = pd.DataFrame({"field_uid": ["field-1"], "season": [2024]})
    burden = hutton_alarm_burden(results, weather, 7)
    assert burden["field_days"] == 1
    assert burden["alarm_day_fraction"] == 1.0


def test_calendar_baseline_distinguishes_fitted_and_fixed_rules() -> None:
    development = pd.DataFrame(
        {
            "first_positive_date": [date(2021, 7, 10), date(2022, 8, 20)],
        }
    )
    evaluation = pd.DataFrame(
        {
            "season": [2024],
            "first_observation_date": [date(2024, 6, 20)],
            "first_positive_date": [date(2024, 7, 20)],
        }
    )
    summary = calendar_baseline_summary(development, evaluation)
    fixed = summary["fixed_july_august"]
    fitted = summary["development_min_max_detection_window"]
    assert bool(fixed["prespecified_fixed_rule"])
    assert not bool(fixed["trained_on_development_only"])
    assert bool(fitted["trained_on_development_only"])
    assert not bool(fitted["prespecified_fixed_rule"])
    assert "pre_detection_alarm_day_fraction_june1" in fixed
    assert "pre_detection_alarm_day_fraction_first_visit_proxy" in fixed


def test_hutton_partial_window_is_not_scorable_without_known_pass() -> None:
    dates = [value.date() for value in pd.date_range("2024-05-29", "2024-06-01")]
    weather = pd.DataFrame(
        {
            "field_uid": ["field-1"] * len(dates),
            "season": [2024] * len(dates),
            "date": dates,
            "hutton_period_status": ["fail"] * len(dates),
        }
    )
    results = pd.DataFrame({"field_uid": ["field-1"], "season": [2024]})
    burden = hutton_alarm_burden(results, weather, 7)
    assert burden["scorable_field_days"] == 0
    assert burden["indeterminate_field_days"] == 1


def _polyakov_weather(days: int = 18) -> pd.DataFrame:
    dates = [value.date() for value in pd.date_range("2024-06-01", periods=days)]
    return pd.DataFrame(
        {
            "field_uid": ["field-1"] * days,
            "season": [2024] * days,
            "date": dates,
            "temperature_mean_c": [16.0] * days,
            "relative_humidity_mean_pct": [80.0] * days,
            "precipitation_sum_mm": [2.0] * days,
            "accepted": [True] * days,
        }
    )


def _polyakov_case(first_positive: date) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "field_uid": "field-1",
                "season": 2024,
                "phenophase_interval_start": date(2024, 5, 31),
                "phenophase_interval_end": date(2024, 6, 1),
                "last_negative_date": first_positive - timedelta(days=1),
                "first_positive_date": first_positive,
            }
        ]
    )


def test_polyakov_observation_day_cannot_confirm_same_day_signal() -> None:
    same_day = evaluate_polyakov_cases(
        _polyakov_case(date(2024, 6, 16)), _polyakov_weather()
    ).iloc[0]
    next_day = evaluate_polyakov_cases(
        _polyakov_case(date(2024, 6, 17)), _polyakov_weather()
    ).iloc[0]
    assert not bool(same_day["polyakov_activation_any__onset_any"])
    assert bool(next_day["polyakov_activation_any__onset_any"])
    operational_days = {
        date.fromisoformat(value)
        for value in next_day["operational_alarm_dates_before_detection"].split("|")
        if value
    }
    assert all(value < date(2024, 6, 17) for value in operational_days)
    assert next_day["operational_alarm_day_count_before_detection"] <= next_day[
        "phenology_to_detection_at_risk_day_count"
    ]
    assert 0.0 <= next_day["operational_alarm_fraction_before_detection"] <= 1.0


def test_polyakov_negative_burden_excludes_issue_and_censor_days() -> None:
    visits = pd.DataFrame(
        [
            _visit("2024-06-01", False, 51),
            _visit("2024-06-18", False, 61),
        ]
    )
    negative = pd.DataFrame(
        [
            {
                "field_uid": "field-1",
                "season": 2024,
                "last_negative_date": date(2024, 6, 18),
                "dense_late_season_followup": False,
                "strict_coverage": False,
                "full_risk_season_coverage": False,
                "strict_explicit_coverage": False,
            }
        ]
    )
    result = evaluate_polyakov_negative_seasons(
        negative, visits, _polyakov_weather()
    ).iloc[0]
    assert bool(result["polyakov_point_evaluable"])
    assert result["polyakov_manifestation_alarm_day_count"] == 1
    assert result["polyakov_at_risk_day_count"] == 17


def test_paired_binary_summary_reports_model_vs_baseline() -> None:
    frame = pd.DataFrame(
        {
            "model": [True, True, True, True, False, False],
            "baseline": [True, False, False, False, True, False],
        }
    )
    result = paired_binary_summary(frame, "model", "baseline")
    assert result["both_hit"] == 1
    assert result["model_only"] == 3
    assert result["baseline_only"] == 1
    assert result["neither"] == 1
    assert result["mcnemar_p_one_sided_model_better"] == 0.3125
