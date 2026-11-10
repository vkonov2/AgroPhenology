"""Synthetic native counts only; no real coordinates or field identifiers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agro_phenology.early_warning_models import Policy, simulate_policy
from agro_phenology.external_native_cohorts import (
    ISSUED_AT_SEMANTICS, TARGET_SPECS, VISIT_COLUMNS, build_decisions, build_registry, load_visits,
)


def _save(root: Path, source: str, rows: list[dict]) -> None:
    folder = root/"normalized"/source
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    for column in ("source_latitude", "source_longitude"):
        if column in frame:
            frame[column] = frame[column].astype("string")
    frame.to_parquet(folder/"observations.parquet", index=False)


def _aphids(day, *, values=(0, 0), site="synthetic_station_a", species="Myzus persicae"):
    return [dict(source_id="kbs_aphid_suction_2005_2013", site_id=site,
                 observation_date=pd.Timestamp(day), organism=species, value=value,
                 sex=sex, unit="individuals_per_collection") for sex, value in zip(["F", "M"], values)]


def _cutworm(day, value=0, *, site="synthetic_unit_a", latitude=1, longitude=2):
    return dict(source_id="zenodo_6351794", site_id=site, observation_date=pd.Timestamp(day),
                organism="Agrotis ipsilon", unit="source_trap_catch_value", value=value,
                source_latitude=latitude, source_longitude=longitude)


def _wisconsin(day, *, site="synthetic_plot_a", larva=0, egg=0, adult=0):
    return [dict(source_id="zenodo_18664442", site_id=site, observation_date=pd.Timestamp(day),
                 organism="Leptinotarsa decemlineata", value=value, stage=stage,
                 unit="egg_masses_per_plant" if stage == "egg_mass" else "mean_individuals_per_25_sweeps")
            for stage, value in [("larva_unspecified", larva), ("egg_mass", egg), ("adult", adult)]]


def _canonical(rows, *, site="synthetic_unit", target="cutworm", cluster="synthetic_cluster"):
    return pd.DataFrame([
        dict(source_id=TARGET_SPECS[target]["source_id"], target=target, site_id=site,
             season=pd.Timestamp(day).year, observation_date=pd.Timestamp(day), status=status,
             cluster=cluster, station_id="synthetic_station", coordinate_scope="synthetic",
             entry_egg_presence=np.nan, entry_adult_presence=np.nan,
             n_reports=1, missing_reports=int(status == "unknown"), all_taxa_zero=False, zero_mode="reported")
        for day, status in rows
    ])[VISIT_COLUMNS]


@pytest.mark.parametrize("target,species", [
    ("kbs_myzus", "Myzus persicae"), ("kbs_macrosiphum", "Macrosiphum euphorbiae"),
    ("kbs_gossypii", "Aphis gossypii"),
])
def test_kbs_combines_sexes_into_one_visit_and_keeps_species_separate(tmp_path, target, species):
    rows = _aphids("2020-05-01", species=species) + _aphids("2020-05-08", values=(0, .5), species=species)
    rows += _aphids("2020-05-01", values=(9, 4), species="synthetic_unrelated_species")
    _save(tmp_path, "kbs_aphid_suction_2005_2013", rows)
    visits = load_visits(tmp_path, target)
    assert len(visits) == 2
    assert visits.status.tolist() == ["zero", "positive"]
    registry = build_registry(visits)
    assert len(registry) == 1 and registry.warnable_first_event.all()
    assert registry.source_visit_count.tolist() == [2]


def test_missing_sex_is_unknown_but_one_positive_measurement_confirms_presence(tmp_path):
    rows = _aphids("2020-05-01", values=(0, np.nan)) + _aphids("2020-05-08", values=(1, np.nan))
    _save(tmp_path, "kbs_aphid_suction_2005_2013", rows)
    visits = load_visits(tmp_path, "kbs_myzus")
    assert visits.status.tolist() == ["unknown", "positive"]
    registry = build_registry(visits)
    assert registry.eligibility_category.tolist() == ["unknown_at_entry"]
    assert not registry.cohort_included.any()


def test_completely_absent_target_on_first_collection_does_not_move_entry(tmp_path):
    rows = _aphids("2020-05-01", species="synthetic_other_species", values=(1, 0))
    rows += _aphids("2020-05-08") + _aphids("2020-05-15", values=(1, 0))
    _save(tmp_path, "kbs_aphid_suction_2005_2013", rows)
    visits = load_visits(tmp_path, "kbs_myzus")
    assert visits.status.tolist() == ["unknown", "zero", "positive"]
    registry = build_registry(visits)
    assert registry.entry_date.iloc[0] == pd.Timestamp("2020-05-01")
    assert not registry.cohort_included.any()


def test_absent_one_sex_cannot_make_a_complete_zero(tmp_path):
    rows = _aphids("2020-05-01")[:1] + _aphids("2020-05-08", values=(1, 0))
    _save(tmp_path, "kbs_aphid_suction_2005_2013", rows)
    assert load_visits(tmp_path, "kbs_myzus").status.tolist() == ["unknown", "positive"]


def test_all_taxa_zero_sensitivity_is_predeclared_unknown_not_a_later_convenient_entry(tmp_path):
    rows = []
    for day, target_counts, other_counts in [
        ("2020-05-01", (0, 0), (0, 0)),
        ("2020-05-08", (0, 0), (1, 0)),
        ("2020-05-15", (1, 0), (0, 0)),
    ]:
        rows += _aphids(day, values=target_counts)
        rows += _aphids(day, values=other_counts, species="synthetic_other_species")
    _save(tmp_path, "kbs_aphid_suction_2005_2013", rows)
    primary = load_visits(tmp_path, "kbs_myzus")
    sensitivity = load_visits(tmp_path, "kbs_myzus", "all_taxa_zero_unknown")
    assert primary.status.tolist() == ["zero", "zero", "positive"]
    assert primary.all_taxa_zero.tolist() == [True, False, False]
    assert sensitivity.status.tolist() == ["unknown", "zero", "positive"]
    p, s = build_registry(primary), build_registry(sensitivity)
    assert p.cohort_included.all() and not s.cohort_included.any()
    assert p.entry_date.equals(s.entry_date)
    assert p.field_season.equals(s.field_season)


def test_missing_non_target_values_cannot_establish_an_all_taxa_zero(tmp_path):
    rows = _aphids("2020-05-01") + _aphids("2020-05-01", values=(0, np.nan), species="synthetic_other_species")
    _save(tmp_path, "kbs_aphid_suction_2005_2013", rows)
    visits = load_visits(tmp_path, "kbs_myzus", "all_taxa_zero_unknown")
    assert visits.status.tolist() == ["zero"]
    assert not visits.all_taxa_zero.any()


def test_duplicate_kbs_species_sex_is_not_treated_as_a_second_sex(tmp_path):
    rows = _aphids("2020-05-01")
    _save(tmp_path, "kbs_aphid_suction_2005_2013", rows+[rows[0]])
    with pytest.raises(ValueError, match="Duplicate KBS"):
        load_visits(tmp_path, "kbs_myzus")


def test_cutworm_fractional_presence_and_numeric_coordinate_grouping(tmp_path):
    rows = [_cutworm("2020-05-01"), _cutworm("2020-05-08", .25),
            _cutworm("2021-05-01", site="synthetic_unit_b", latitude="1.000", longitude="2.0"),
            _cutworm("2021-05-08", 1, site="synthetic_unit_b", latitude="1.000", longitude="2.0")]
    _save(tmp_path, "zenodo_6351794", rows)
    visits = load_visits(tmp_path, "cutworm")
    assert visits.status.tolist() == ["zero", "positive", "zero", "positive"]
    assert visits.cluster.nunique() == 1
    assert not any("latitude" in c or "longitude" in c for c in visits.columns)
    registry = build_registry(visits)
    assert len(registry) == registry.field_season.nunique() == 2
    assert registry.warnable_first_event.all()


def test_cutworm_other_species_cannot_supply_a_target_visit(tmp_path):
    rows = [_cutworm("2020-05-08"), _cutworm("2020-05-15", 1)]
    other = _cutworm("2020-05-01", 1)
    other["organism"] = "Psila rosae"
    _save(tmp_path, "zenodo_6351794", [other, *rows])
    visits = load_visits(tmp_path, "cutworm")
    assert visits.observation_date.min() == pd.Timestamp("2020-05-08")


def test_wisconsin_conflicting_duplicates_use_presence_not_mean_or_replicate_events(tmp_path):
    rows = _wisconsin("2020-05-01", adult=1)
    rows += _wisconsin("2020-05-08", larva=0, egg=1)
    rows += [dict(rows[3], value=1), dict(rows[3], value=np.nan)]
    rows += _wisconsin("2020-05-15", larva=2)
    _save(tmp_path, "zenodo_18664442", rows)
    visits = load_visits(tmp_path, "cpb_wisconsin")
    assert visits.status.tolist() == ["zero", "positive", "positive"]
    assert visits.n_reports.tolist() == [1, 3, 1]
    registry = build_registry(visits)
    assert len(registry) == 1 and registry.warnable_first_event.all()
    assert registry.first_recorded_event_date.iloc[0] == pd.Timestamp("2020-05-08")
    assert registry.entry_egg_presence.tolist() == [0]
    assert registry.entry_adult_presence.tolist() == [1]
    assert registry.station_id.tolist() == ["geography_not_provided"]
    assert registry.cluster.tolist() == ["year_2020"]


def test_wisconsin_missing_report_prevents_zero_and_cannot_be_rescued_by_later_zero(tmp_path):
    rows = _wisconsin("2020-05-01")
    rows += [dict(rows[0], value=np.nan)]
    rows += _wisconsin("2020-05-08") + _wisconsin("2020-05-15", larva=1)
    _save(tmp_path, "zenodo_18664442", rows)
    visits = load_visits(tmp_path, "cpb_wisconsin")
    assert visits.status.tolist() == ["unknown", "zero", "positive"]
    assert build_registry(visits).eligibility_category.tolist() == ["unknown_at_entry"]


def test_wisconsin_absent_larval_stage_preserves_original_visit_as_unknown(tmp_path):
    rows = _wisconsin("2020-05-01")[1:]
    rows += _wisconsin("2020-05-08") + _wisconsin("2020-05-15", larva=1)
    _save(tmp_path, "zenodo_18664442", rows)
    visits = load_visits(tmp_path, "cpb_wisconsin")
    assert visits.status.tolist() == ["unknown", "zero", "positive"]
    assert visits.n_reports.iloc[0] == 0
    assert not build_registry(visits).cohort_included.any()


def test_entry_features_never_use_future_stage_observations(tmp_path):
    original = _wisconsin("2020-05-01", egg=np.nan, adult=0)+_wisconsin("2020-05-08", larva=1, egg=1, adult=1)
    _save(tmp_path, "zenodo_18664442", original)
    before = build_registry(load_visits(tmp_path, "cpb_wisconsin"))
    original[-2]["value"], original[-1]["value"] = 0, np.nan
    _save(tmp_path, "zenodo_18664442", original)
    after = build_registry(load_visits(tmp_path, "cpb_wisconsin"))
    columns = ["entry_date", "entry_doy", "entry_egg_presence", "entry_adult_presence"]
    pd.testing.assert_frame_equal(before[columns], after[columns])
    assert pd.isna(before.entry_egg_presence.iloc[0])


@pytest.mark.parametrize("value", [-1, np.inf, -np.inf, "invalid"])
def test_nonfinite_or_negative_native_counts_are_rejected(tmp_path, value):
    _save(tmp_path, "zenodo_6351794", [_cutworm("2020-05-01", value)])
    with pytest.raises(ValueError, match="Counts|counts"):
        load_visits(tmp_path, "cutworm")


def test_wrong_units_or_mixed_dataset_are_rejected(tmp_path):
    row = _cutworm("2020-05-01")
    _save(tmp_path, "zenodo_6351794", [dict(row, unit="invented_individuals_per_day")])
    with pytest.raises(ValueError, match="measurements"):
        load_visits(tmp_path, "cutworm")
    _save(tmp_path, "zenodo_6351794", [dict(row, source_id="another_dataset")])
    with pytest.raises(ValueError, match="declared dataset"):
        load_visits(tmp_path, "cutworm")


def test_unknown_target_zero_mode_or_coordinates_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown native"):
        load_visits(tmp_path, "invented")
    with pytest.raises(ValueError, match="Unknown zero"):
        load_visits(tmp_path, "cutworm", "invented")
    with pytest.raises(ValueError, match="KBS only"):
        load_visits(tmp_path, "cutworm", "all_taxa_zero_unknown")
    _save(tmp_path, "zenodo_6351794", [_cutworm("2020-05-01", latitude=np.nan)])
    with pytest.raises(ValueError, match="coordinates"):
        load_visits(tmp_path, "cutworm")


@pytest.mark.parametrize("delay,warnable", [(1, False), (3, False), (4, True), (7, True)])
def test_short_first_events_remain_in_service_cohort_but_not_warning_denominator(delay, warnable):
    entry = pd.Timestamp("2020-05-01")
    visits = _canonical([(entry, "zero"), (entry+pd.Timedelta(days=delay), "positive")])
    registry = build_registry(visits)
    assert registry.cohort_included.all()
    assert bool(registry.warnable_first_event.iloc[0]) is warnable
    assert registry.training_informative.all()
    assert registry.eligibility_category.iloc[0] == ("eligible" if warnable else "short_warning")
    assert registry.first_recorded_event_date.iloc[0] == entry+pd.Timedelta(days=delay)


def test_first_positive_is_left_censored_even_after_later_zero():
    visits = _canonical([("2020-05-01", "positive"), ("2020-05-08", "zero"), ("2020-05-15", "positive")])
    registry = build_registry(visits)
    assert registry.positive_at_first_visit.all()
    assert not registry.cohort_included.any()
    assert registry.first_recorded_event_date.iloc[0] == pd.Timestamp("2020-05-01")
    assert len(build_decisions(registry)) == 0


def test_right_censoring_does_not_extend_through_unknown_measurements_or_november_positive():
    visits = _canonical([("2020-09-30", "zero"), ("2020-10-15", "zero"),
                         ("2020-10-22", "unknown"), ("2020-11-01", "positive")])
    registry = build_registry(visits)
    row = registry.iloc[0]
    assert row.eligibility_category == "right_censored"
    assert row.last_assessment_date == row.last_complete_zero_date == pd.Timestamp("2020-10-15")
    assert pd.isna(row.first_recorded_event_date)
    decisions = build_decisions(registry)
    assert decisions.issue_date.max() == pd.Timestamp("2020-10-31")
    assert decisions.service_active.all()
    assert decisions.evaluation_field_day.sum() == 15


def test_no_followup_zero_is_not_positive_or_informative_training_label():
    registry = build_registry(_canonical([("2020-05-01", "zero"), ("2020-05-08", "unknown")]))
    assert registry.cohort_included.all()
    assert registry.followup_days.tolist() == [0]
    assert not registry.training_informative.any()
    assert not build_decisions(registry).evaluation_field_day.any()


def test_calendar_boundary_preserves_late_entries_with_reasons():
    a = _canonical([("2020-10-30", "zero")], site="synthetic_a")
    b = _canonical([("2020-10-31", "zero")], site="synthetic_b")
    registry = build_registry(pd.concat([a, b], ignore_index=True)).set_index("site_id")
    assert registry.loc["synthetic_a", "cohort_included"]
    assert registry.loc["synthetic_b", "eligibility_category"] == "outside_season_entry"
    assert len(build_decisions(registry.reset_index())) == 1


def test_decision_grid_and_entry_features_are_independent_of_future_visit_dates():
    registry = build_registry(_canonical([("2020-05-01", "zero"), ("2020-05-08", "positive")]))
    original = build_decisions(registry)
    changed_registry = registry.copy()
    changed_registry["first_recorded_event_date"] += pd.Timedelta(days=14)
    changed_registry["last_assessment_date"] += pd.Timedelta(days=21)
    changed_registry["previous_visit_gap_days"] = 99
    changed = build_decisions(changed_registry)
    static = ["field_season", "issue_date", "issued_at", "entry_date", "entry_doy",
              "entry_egg_presence", "entry_adult_presence", "station_id"]
    pd.testing.assert_frame_equal(original[static], changed[static])
    assert original.issue_date.min() == pd.Timestamp("2020-05-02")
    assert original.issue_date.max() == pd.Timestamp("2020-10-31")
    assert pd.to_datetime(original.issued_at).dt.tz is None
    assert original.issued_at_semantics.eq(ISSUED_AT_SEMANTICS).all()
    assert not original.target_observable.any() and original.target_class.eq("unknown").all()


def test_service_stops_only_when_first_positive_record_becomes_available_next_day():
    registry = build_registry(_canonical([("2020-05-01", "zero"), ("2020-05-08", "positive")]))
    decisions = build_decisions(registry)
    on_event = decisions.issue_date.eq(pd.Timestamp("2020-05-08"))
    after_event = decisions.issue_date.gt(pd.Timestamp("2020-05-08"))
    assert decisions.loc[on_event, "service_active"].all()
    assert not decisions.loc[after_event, "service_active"].any()
    state = simulate_policy(decisions, pd.Series(1., index=decisions.index), Policy(.5))
    assert state.message_issued.sum() == 1
    assert not state.loc[after_event, "message_issued"].any()
    assert not state.loc[after_event, "alarm_active"].any()


def test_registry_rejects_mixed_targets_zero_interpretations_duplicates_and_wrong_dates():
    frame = _canonical([("2020-05-01", "zero"), ("2020-05-08", "positive")])
    with pytest.raises(ValueError, match="Duplicate canonical"):
        build_registry(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
    mixed = frame.copy()
    mixed.loc[1, "target"] = "kbs_myzus"
    with pytest.raises(ValueError, match="cannot pool"):
        build_registry(mixed)
    mixed = frame.copy()
    mixed.loc[1, "zero_mode"] = "all_taxa_zero_unknown"
    with pytest.raises(ValueError, match="cannot pool"):
        build_registry(mixed)
    with pytest.raises(ValueError, match="disagrees"):
        build_registry(frame.assign(season=2021))


def test_source_date_not_legacy_day_counter_determines_features(tmp_path):
    rows = _wisconsin("2020-05-01")+_wisconsin("2020-05-08", larva=1)
    for row in rows:
        row["Day"], row["Week"], row["cum_dd"], row["future_weather"] = 999, 999, 99999, -999
    _save(tmp_path, "zenodo_18664442", rows)
    registry = build_registry(load_visits(tmp_path, "cpb_wisconsin"))
    assert registry.entry_doy.tolist() == [122]
    assert "cum_dd" not in registry
    assert registry.entry_date.iloc[0] == pd.Timestamp("2020-05-01")
