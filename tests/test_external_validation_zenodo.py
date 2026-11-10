from __future__ import annotations

import io
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from agro_phenology.external_validation_zenodo import RECORDS, discover, normalize

# All coordinate fixtures below are deliberately synthetic, unrelated to fields.


def _source(record_id: str, filename: str) -> dict:
    return {
        "source_id": f"zenodo_{record_id}", "access_status": "open",
        "normalizer": RECORDS[record_id]["normalizer"], "limitations": [],
        "files": [{"name": filename}],
    }


def _raw(tmp_path: Path, filename: str, content: str) -> Path:
    raw = tmp_path / "raw"
    (raw / "files").mkdir(parents=True)
    (raw / "files" / filename).write_text(content, encoding="utf-8")
    return raw


def test_potato_preserves_control_plots_and_unknown_units(tmp_path) -> None:
    raw = _raw(tmp_path, "potato.txt", "Country\tLatitude\tLongitude\tDate\tCropStage\tTrial\tPlot\tTreatment\tUntreated\tValue\nUK\t0.125\t-20.25\t2013-08-01\t65-71\t1\t103\t2\t0\t0\nUK\t0.125\t-20.25\t2013-08-07\t65-73\t1\t104\t1\t1\tNA\n")
    result = normalize(_source("6352735", "potato.txt"), raw, tmp_path / "out")
    obs = pd.read_parquet(tmp_path / "out/observations.parquet")
    assert result["rows"] == 2
    assert obs.untreated_control.tolist() == [False, True]
    assert obs.explicit_zero.iloc[0]
    assert pd.isna(obs.explicit_zero.iloc[1])
    assert pd.isna(obs.value.iloc[1])
    assert obs.unit.eq("source_units_unverified").all()
    assert obs.source_plot.tolist() == ["103", "104"]
    assert obs.plot_id.str.endswith((":plot:103", ":plot:104")).all()
    assert obs.source_row_number.tolist() == [2, 3]


def test_trap_absence_between_visits_is_not_created_and_fractional_catches_remain(tmp_path) -> None:
    raw = _raw(tmp_path, "trap.txt", "Species\tFieldId\tLatitude\tLongitude\tDate\tValue\nCarrot fly\t1\t0.375\t-21.25\t2007-06-08\t0\nCarrot fly\t1\t0.375\t-21.25\t2007-06-14\t9.3\n")
    normalize(_source("6351794", "trap.txt"), raw, tmp_path / "out")
    obs = pd.read_parquet(tmp_path / "out/observations.parquet")
    assert len(obs) == 2
    assert obs.value.tolist() == [0, 9.3]
    assert obs.observation_date.dt.day.tolist() == [8, 14]
    assert obs.crop.isna().all()
    assert obs.organism.eq("Psila rosae").all()


def test_grape_arms_are_separate_with_aligned_source_coordinates(tmp_path) -> None:
    raw = _raw(tmp_path, "grape.txt", "TrialId\tCountry\tLatitude\tLongitude\tDate\tGsFrom\tGsTo\tMethod\tPlantPart\tUntreated\tTreated\n2\tFrance\t0.25\t-22.25\t2012-06-13\t68\t71\tP%FREQ\tRACEME\t30.5\t0\n3\tGreece\t0.75\t-22.75\t2012-06-28\t75\t75\tP%INF\tLEAF\t54.18\t14.5\n")
    result = normalize(_source("6520937", "grape.txt"), raw, tmp_path / "out")
    obs = pd.read_parquet(tmp_path / "out/observations.parquet")
    loc = pd.read_parquet(tmp_path / "out/locations.parquet")
    assert result["rows"] == 4
    assert obs.untreated_control.tolist() == [True, True, False, False]
    assert obs.value.tolist() == [30.5, 54.18, 0, 14.5]
    assert obs.record_id.nunique() == 4
    assert len(loc) == 2
    assert set(loc.latitude) == {0.25, 0.75}
    assert not loc.site_id.duplicated().any()
    assert obs.trial_id.nunique() == 2


def test_spatial_year_does_not_become_an_invented_assessment_date(tmp_path) -> None:
    filename = "lateblight_2022_infection_severities_plusXY.csv"
    raw = _raw(tmp_path, filename, "mean_sev_wp,treatm,cell,x,y\n0,0,164,100000.25,1000000.75\n3,1,164,100000.25,1000000.75\n")
    result = normalize(_source("18351833", filename), raw, tmp_path / "out")
    obs = pd.read_parquet(tmp_path / "out/observations.parquet")
    loc = pd.read_parquet(tmp_path / "out/locations.parquet")
    assert result["date_range"] is None
    assert obs.observation_date.isna().all()
    assert obs.observation_year.eq(2022).all()
    assert loc.crs.eq("EPSG:26917").all()
    assert "latitude" not in loc.columns


def test_only_original_zip_members_are_normalized(tmp_path) -> None:
    raw = tmp_path / "raw"
    (raw / "files").mkdir(parents=True)
    data = "\tdate\tair_hum\tair_temp\tpluvm\tdown_mildew\tpowdery_mildew\tbotrytis\terysiphe\n0\t2/26/2020\t53\t9.5\t0.1\t0\t1\t0\t1\n"
    with zipfile.ZipFile(raw / "files/environmental_data.zip", "w") as z:
        z.writestr("env/Vine_original_2020_form.csv", data)
        z.writestr("env/Vine_2020_2024_synth_form.csv", data)
        z.writestr("env/Vine_noised_0_form.csv", data)
    result = normalize(_source("14989522", "environmental_data.zip"), raw, tmp_path / "out")
    obs = pd.read_parquet(tmp_path / "out/observations.parquet")
    assert result["rows"] == 4
    assert result["synthetic_rows_normalized"] == 0
    assert len(result["excluded_archive_members"]) == 2
    assert not obs.synthetic.any()
    assert obs.label_provenance.eq("unverified_source_label").all()
    assert obs.observation_date.eq(pd.Timestamp("2020-02-26")).all()
    assert len(pd.read_parquet(tmp_path / "out/original_daily_weather.parquet")) == 1


def test_nonbinary_control_flag_is_rejected(tmp_path) -> None:
    raw = _raw(tmp_path, "apple.txt", "Country\tLatitude\tLongitude\tDate\tTrial\tTreatment\tUntreated\tMeasure\tValue\nBelgium\t0.5\t-23.25\t2011-06-09\t1\t1\t2\tLEAF_FREQ\t44.25\n")
    with pytest.raises(ValueError, match="not binary"):
        normalize(_source("6520484", "apple.txt"), raw, tmp_path / "out")


def test_apple_raw_classes_do_not_inherit_percent_units(tmp_path) -> None:
    raw = _raw(tmp_path, "apple.txt", "Country\tLatitude\tLongitude\tDate\tTrial\tTreatment\tUntreated\tMeasure\tValue\nBelgium\t0.5\t-23.25\t2011-06-09\t1\t1\t1\tLEAF_FREQ\t44.25\nBelgium\t0.5\t-23.25\t2011-06-09\t1\t1\t1\tCLASSES\t2\n")
    result = normalize(_source("6520484", "apple.txt"), raw, tmp_path / "out")
    obs = pd.read_parquet(tmp_path / "out/observations.parquet")
    assert obs.unit.tolist() == ["percent", "source_units_unverified"]
    assert obs.value.tolist() == [44.25, 2]
    assert result["units_unverified_records"] == 1


def test_invalid_percentage_is_preserved_and_flagged_without_clipping(tmp_path) -> None:
    raw = _raw(tmp_path, "apple.txt", "Country\tLatitude\tLongitude\tDate\tTrial\tTreatment\tUntreated\tMeasure\tValue\nBelgium\t0.5\t-23.25\t2011-06-09\t1\t1\t1\tLEAF_FREQ\t-2\nBelgium\t0.5\t-23.25\t2011-06-09\t1\t1\t1\tLEAF_FREQ\t0\nBelgium\t0.5\t-23.25\t2011-06-09\t1\t1\t1\tCLASSES\t-1\n")
    result = normalize(_source("6520484", "apple.txt"), raw, tmp_path / "out")
    obs = pd.read_parquet(tmp_path / "out/observations.parquet")
    assert obs.value.tolist() == [-2, 0, -1]
    assert obs.value_domain_valid.iloc[0] == False
    assert obs.value_domain_valid.iloc[1] == True
    assert pd.isna(obs.value_domain_valid.iloc[2])
    assert result["invalid_percent_records"] == 1


def test_repeated_trial_counter_across_locations_and_years_does_not_merge_trials(tmp_path) -> None:
    raw = _raw(tmp_path, "potato.txt", "Country\tLatitude\tLongitude\tDate\tCropStage\tTrial\tPlot\tTreatment\tUntreated\tValue\nUK\t0.125\t-20.25\t2013-08-01\t65\t1\t103\t1\t1\t0\nUK\t0.625\t-20.75\t2014-08-01\t65\t1\t103\t1\t1\t10\nUK\t0.125\t-20.25\t2015-08-01\t65\t1\t103\t1\t1\t20\n")
    normalize(_source("6352735", "potato.txt"), raw, tmp_path / "out")
    obs = pd.read_parquet(tmp_path / "out/observations.parquet")
    locations = pd.read_parquet(tmp_path / "out/locations.parquet")
    assert obs.site_id.nunique() == 2
    assert obs.trial_id.nunique() == 3
    assert obs.plot_id.nunique() == 3
    assert obs.source_trial.tolist() == ["1", "1", "1"]
    assert len(locations) == 2
    assert not locations.site_id.duplicated().any()


def test_reused_field_id_at_distinct_coordinates_is_not_one_location(tmp_path) -> None:
    raw = _raw(tmp_path, "trap.txt", "Species\tFieldId\tLatitude\tLongitude\tDate\tValue\nCarrot fly\t1\t0.375\t-21.25\t2007-06-08\t0\nCarrot fly\t1\t0.875\t-21.75\t2008-06-08\t1\n")
    normalize(_source("6351794", "trap.txt"), raw, tmp_path / "out")
    obs = pd.read_parquet(tmp_path / "out/observations.parquet")
    assert obs.site_id.nunique() == 2


def test_discovery_links_closed_older_records_to_distinct_open_records() -> None:
    class Client:
        def get_json(self, url, relative_path, max_bytes):
            record_id = url.rsplit("/", 1)[-1]
            restricted = record_id in {"6361745", "6353533"}
            return {"metadata": {"title": record_id, "access_right": "restricted" if restricted else "open", "license": None if restricted else {"id": "cc-by-4.0"}},
                    "files": [] if restricted else [{"key": "example.txt", "size": 4, "checksum": "md5:abc", "links": {"self": url + "/files/example.txt/content"}}]}
    sources = {x["source_id"]: x for x in discover(Client())}
    assert sources["zenodo_6361745"]["access_status"] == "restricted"
    assert sources["zenodo_6361745"]["files"] == []
    assert sources["zenodo_6361745"]["newer_open_record"] == "zenodo_6520484"
    assert sources["zenodo_6520484"]["access_status"] == "open"
    assert sources["zenodo_6520484"]["earlier_restricted_record"] == "zenodo_6361745"
    assert sources["zenodo_6361745"]["license"]["id"] is None
