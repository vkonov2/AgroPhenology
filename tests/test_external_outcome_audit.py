from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agro_phenology import external_outcome_audit as audit
from agro_phenology.external_validation_io import digest_file


APPLE, GRAPE = audit.SOURCES


def _row(trial, day, value, *, measurement="measure_a", site="synthetic_site", source=APPLE,
         unit="percent", treatment="1", control=True, valid=True):
    return dict(source_id=source, trial_id=trial, site_id=site, measurement=measurement,
                unit=unit, treatment=treatment, untreated_control=control,
                observation_date=pd.Timestamp(day), value=value, value_domain_valid=valid)


def _tables(rows, source=APPLE):
    return audit.audit_tables({source: pd.DataFrame(rows)})


def test_conflicting_zero_and_positive_duplicates_never_create_a_negative_entry():
    tables = _tables([_row("trial", "2020-05-01", 0), _row("trial", "2020-05-01", 5),
                      _row("trial", "2020-05-08", 0), _row("trial", "2020-05-15", 10)])
    registry = tables["series"].iloc[0]
    assert registry.category == "positive_at_entry"
    assert not registry.eligible and not registry.cohort_included
    assert registry.first_recorded_event_date == pd.Timestamp("2020-05-01")
    assert registry.mixed_zero_positive_groups == registry.conflicting_groups == 1
    assert tables["visits"].status.tolist() == ["positive", "zero", "positive"]
    assert tables["source_summary"].eligible_measurement_series.iloc[0] == 0


def test_measurement_overlap_is_counted_as_shared_technical_trials_not_independent_events():
    rows = []
    for measure in ["fruit_frequency", "fruit_intensity"]:
        rows += [_row("trial", "2020-05-01", 0, measurement=measure),
                 _row("trial", "2020-05-08", 7, measurement=measure)]
    tables = _tables(rows)
    summary = tables["source_summary"].iloc[0]
    assert summary.eligible_measurement_series == 2
    assert summary.unique_eligible_technical_trial_ids == 1
    assert len(tables["measure_summary"]) == 2
    assert tables["overlaps"].shared_eligible_technical_trial_ids.tolist() == [1]


def test_measurement_names_and_unverified_classes_are_not_merged_or_assumed_percent():
    rows = []
    for measure in ["LEAF_FREQ", "BEFHKT_LE"]:
        rows += [_row("trial", "2020-05-01", 0, measurement=measure),
                 _row("trial", "2020-05-08", 1, measurement=measure)]
    rows += [_row("trial", "2020-05-01", 0, measurement="CLASSES", unit="source_units_unverified", valid=pd.NA)]
    tables = _tables(rows)
    assert set(tables["measure_summary"].measurement) == {"LEAF_FREQ", "BEFHKT_LE"}
    excluded = tables["excluded_units"].iloc[0]
    assert excluded.measurement == "CLASSES"
    assert excluded.rows == excluded.technical_trial_ids == 1
    assert not tables["series"].unit.ne("percent").any()


@pytest.mark.parametrize("value,valid", [(np.nan, True), (0, pd.NA), (-1, False), (101, True), (np.inf, True)])
def test_unknown_first_visit_is_not_rescued_by_a_later_zero(value, valid):
    tables = _tables([_row("trial", "2020-05-01", value, valid=valid),
                      _row("trial", "2020-05-08", 0), _row("trial", "2020-05-15", 10)])
    row = tables["series"].iloc[0]
    assert row.category == "unknown_at_entry"
    assert not row.eligible and not row.cohort_included
    assert tables["visits"].status.tolist() == ["unknown", "zero", "positive"]


def test_any_confirmed_positive_survives_other_unknown_values_but_partial_zero_does_not():
    rows = [_row("a", "2020-05-01", 3), _row("a", "2020-05-01", np.nan),
            _row("b", "2020-05-01", 0), _row("b", "2020-05-01", np.nan)]
    visits = _tables(rows)["visits"].set_index("trial_id")
    assert visits.loc["a", "status"] == "positive"
    assert visits.loc["b", "status"] == "unknown"


@pytest.mark.parametrize("delay,category", [(3, "short_warning"), (4, "eligible")])
def test_opportunity_boundary_accounts_for_first_issue_on_next_day(delay, category):
    entry = pd.Timestamp("2020-05-01")
    row = _tables([_row("trial", entry, 0), _row("trial", entry+pd.Timedelta(days=delay), 2)])["series"].iloc[0]
    assert row.category == category
    assert row.cohort_included
    assert bool(row.eligible) is (category == "eligible")


def test_season_cutoff_does_not_turn_postseason_positive_or_unvisited_days_into_negatives():
    tables = _tables([_row("trial", "2020-10-01", 0), _row("trial", "2020-10-10", 0),
                      _row("trial", "2020-11-02", 4)])
    row = tables["series"].iloc[0]
    assert row.category == "right_censored"
    assert pd.isna(row.first_recorded_event_date)
    assert row.last_assessment_date == pd.Timestamp("2020-10-10")
    assert tables["visits"].before_or_on_season_end.tolist() == [True, True, False]


def test_too_late_entry_and_duplicate_zero_reports_do_not_inflate_events_or_visits():
    rows = [_row("late", "2020-10-31", 0),
            _row("normal", "2020-05-01", 0), _row("normal", "2020-05-01", 0),
            _row("normal", "2020-05-08", 2)]
    tables = _tables(rows)
    registry = tables["series"].set_index("trial_id")
    assert registry.loc["late", "category"] == "outside_season_entry"
    assert registry.loc["normal", "assessment_dates"] == 2
    assert registry.loc["normal", "source_rows"] == 3
    assert registry.loc["normal", "duplicate_groups"] == 1
    assert registry.loc["normal", "conflicting_groups"] == 0
    assert tables["source_summary"].eligible_measurement_series.iloc[0] == 1


def test_treatment_groups_and_datasets_remain_separate_and_invalid_treated_values_do_not_enter_controls():
    apple = pd.DataFrame([_row("trial", "2020-05-01", 0), _row("trial", "2020-05-08", 2),
                          _row("trial", "2020-05-01", "unusable", control=False, treatment="2")])
    grape = pd.DataFrame([_row("trial", "2020-05-01", 0, source=GRAPE),
                          _row("trial", "2020-05-08", 1, source=GRAPE)])
    tables = audit.audit_tables({APPLE: apple, GRAPE: grape})
    assert len(tables["source_summary"]) == 2
    assert tables["source_summary"].eligible_measurement_series.tolist() == [1, 1]
    assert tables["overlaps"].empty  # Same technical text in another dataset is not overlap.
    assert tables["visits"].treatment.eq("1").all()


@pytest.mark.parametrize("mutation,reason", [
    ("different_source", "Exactly one"), ("multiple_sites", "one site"),
    ("multiple_years", "one site"), ("missing_dates", "known local"),
    ("intraday_date", "silently rounded"), ("unknown_control", "control flag"),
])
def test_invalid_or_ambiguous_identity_is_not_silently_aggregated(mutation, reason):
    frame = pd.DataFrame([_row("trial", "2020-05-01", 0), _row("trial", "2020-05-08", 1)])
    if mutation == "different_source":
        frame.loc[0, "source_id"] = GRAPE
    elif mutation == "multiple_sites":
        frame.loc[1, "site_id"] = "other_synthetic_site"
    elif mutation == "multiple_years":
        frame.loc[1, "observation_date"] = pd.Timestamp("2021-05-08")
    elif mutation == "missing_dates":
        frame.loc[0, "observation_date"] = pd.NaT
    elif mutation == "intraday_date":
        frame.loc[0, "observation_date"] += pd.Timedelta(hours=12)
    else:
        frame["untreated_control"] = frame.untreated_control.astype("boolean")
        frame.loc[0, "untreated_control"] = pd.NA
    with pytest.raises(ValueError, match=reason):
        audit.audit_tables({APPLE: frame})


def _collection(tmp_path):
    collection = tmp_path/"collection"
    files = []
    for source in audit.SOURCES:
        normalized = collection/f"normalized/{source}"
        raw = collection/f"raw/{source}"
        normalized.mkdir(parents=True)
        raw.mkdir(parents=True)
        observations = pd.DataFrame([_row("synthetic_trial", "2020-05-01", 0, source=source),
                                     _row("synthetic_trial", "2020-05-08", 3, source=source)])
        path = normalized/"observations.parquet"
        observations.to_parquet(path, index=False)
        metadata = raw/"metadata.json"
        metadata.write_text('{}', encoding="utf-8")
        files += [path, metadata]
    manifest = dict(status="complete", file_inventory={str(p.relative_to(collection)): {
        "bytes": p.stat().st_size, "sha256": digest_file(p)} for p in files})
    (collection/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return collection


def test_run_is_immutable_and_verify_recomputes_and_detects_output_and_input_tamper(tmp_path):
    collection = _collection(tmp_path)
    root = tmp_path/"run"
    assert audit.run(collection, root)["status"] == "complete"
    checked = audit.verify(root)
    assert checked["status"] == "passed"
    assert len(checked["recomputed_tables"]) == 7
    with pytest.raises(FileExistsError, match="immutable"):
        audit.run(collection, root)
    path = root/"measure_summary.parquet"
    original = path.read_bytes()
    table = pd.read_parquet(path)
    table.loc[0, "eligible"] += 1
    table.to_parquet(path, index=False)
    assert audit.verify(root)["mismatches"] == ["measure_summary.parquet"]
    # Updating a forged output hash is insufficient: recomputation still fails.
    manifest_path = root/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file_inventory"][path.name] = digest_file(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert audit.verify(root)["mismatches"] == ["recompute:measure_summary"]
    path.write_bytes(original)
    input_path = collection/f"normalized/{APPLE}/observations.parquet"
    input_path.write_bytes(input_path.read_bytes()+b"changed")
    assert str(input_path.resolve()) in audit.verify(root)["input_mismatches"]
