from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from agro_phenology.external_validation_insects import (
    _daks_field_coordinates,
    _discover_ris_annual,
    normalize,
)


def _source(source_id: str) -> dict:
    return {"source_id": source_id, "limitations": ["Test uses synthetic rows only."]}


def _raw(tmp_path: Path, *, daks_metadata: bool = False) -> Path:
    raw = tmp_path / "raw"
    (raw / "files").mkdir(parents=True)
    if daks_metadata:
        # Invented coordinates and descriptions; no observed field data.
        description = (
            'Small-scale field trials were set up at a synthetic farm (10°10\'10"N 20°20\'20"E). '
            'Farm-scale trials were conducted in 2021b (11°11\'11"N, 21°21\'21"E) '
            'and 2022 (12°12\'12"N, 22°22\'22"E). '
            'A synthetic trial was conducted in 2021a at a synthetic site (13°13\'13"N, 23°23\'23"E).'
        )
        (raw / "metadata.json").write_text(json.dumps({"metadata": {"dc.description": [{"value": description}]}}), encoding="utf-8")
    return raw


def test_daks_preserves_life_stages_treatments_and_missing_values(tmp_path: Path) -> None:
    raw = _raw(tmp_path, daks_metadata=True)
    (raw / "files/CPB_small-scale_trial_2019.csv").write_text(
        "DATE;PLOT;BLOCK;TREATMENT;PLANT;LEPDEC_I;LEPDEC_E;LEPDEC_ED;LEPDEC_L1\n"
        "2019-06-18;2;I;control;1;0;25;3;0\n"
        "2019-06-25;6;I;mulch;2;;0;0;4\n"
    )
    result = normalize(_source("daks_cpb_2019_2022"), raw, tmp_path / "out")
    rows = pd.read_parquet(tmp_path / "out/observations.parquet")
    assert result["rows"] == 8
    assert set(rows.stage) == {"adult", "egg", "damaged_egg", "larva_1"}
    assert rows.loc[rows.stage.eq("damaged_egg"), "value"].tolist() == [3, 0]
    missing = rows[rows.stage.eq("adult") & rows.treatment.eq("mulch")].iloc[0]
    assert pd.isna(missing.value)
    assert not missing.explicit_zero
    assert set(rows.treatment) == {"control", "mulch"}
    assert not rows.record_id.duplicated().any()
    assert pd.read_parquet(tmp_path / "out/locations.parquet").coordinate_scope.eq("trial_field_not_individual_plant").all()


def test_cpb_does_not_promote_preaggregated_weather_to_safe_features(tmp_path: Path) -> None:
    raw = _raw(tmp_path)
    pd.DataFrame({
        "Date": ["6/3/2008", "6/9/2008"], "plot_ID": ["40_2008"] * 2,
        "adults": [0., .8], "larvae": [0., 0.], "eggs": [float("nan"), .2],
        "Summer.Adults_maxtempMed": [79.16] * 2, "cum_dd": [267.69, 362.35],
    }).to_csv(raw / "files/WI_CPB_preformatted.csv", index=False)
    result = normalize(_source("zenodo_18664442"), raw, tmp_path / "out")
    rows = pd.read_parquet(tmp_path / "out/observations.parquet")
    assert len(rows) == 6
    assert .8 in rows.value.tolist()
    assert "Summer.Adults_maxtempMed" in result["excluded_predictor_columns"]
    assert "Summer.Adults_maxtempMed" not in rows.columns
    assert "cum_dd" not in rows.columns
    assert rows[rows.value.isna()].explicit_zero.eq(False).all()
    assert result["date_range"] == {"minimum": "2008-06-03", "maximum": "2008-06-09"}
    assert rows.loc[rows.source_column.eq("eggs"), "stage"].eq("egg_mass").all()
    assert rows.loc[rows.source_column.eq("eggs"), "unit"].eq("egg_masses_per_plant").all()
    assert rows.loc[rows.source_column.isin(["adults", "larvae"]), "unit"].eq("mean_individuals_per_25_sweeps").all()
    assert rows.loc[rows.source_column.eq("eggs"), "value"].iloc[1] == .2


@pytest.mark.parametrize("trial,extra_header,extra_value", [
    ("CPB_farm-scale_trial_2021a", ";TRIPLETT", ";2"),
    ("CPB_small-scale_trial_2020", "", ""),
])
def test_daks_trial_schemas_do_not_invent_missing_plot_ids(tmp_path: Path, trial, extra_header, extra_value) -> None:
    raw = _raw(tmp_path, daks_metadata=True)
    (raw / "files" / f"{trial}.csv").write_text(
        f"DATE;BLOCK;TREATMENT{extra_header};PLANT;LEPDEC_I;LEPDEC_E\n"
        f"2020-06-17;I;control{extra_value};1;0;3\n"
    )
    normalize(_source("daks_cpb_2019_2022"), raw, tmp_path / "out")
    rows = pd.read_parquet(tmp_path / "out/observations.parquet")
    assert rows.plot_id.isna().all()
    if extra_header:
        assert rows.triplet_id.eq("2").all()
    else:
        assert rows.triplet_id.isna().all()


def test_daks_coordinates_are_derived_only_from_the_local_archived_metadata(tmp_path: Path) -> None:
    raw = _raw(tmp_path, daks_metadata=True)
    coordinates = _daks_field_coordinates(raw / "metadata.json")
    assert len(coordinates) == 6
    for year in (2019, 2020, 2021):
        assert coordinates[f"CPB_small-scale_trial_{year}"] == (10 + 10 / 60 + 10 / 3600, 20 + 20 / 60 + 20 / 3600)
    metadata = json.loads((raw / "metadata.json").read_text())
    metadata["metadata"]["dc.description"][0]["value"] = metadata["metadata"]["dc.description"][0]["value"].replace('10°10\'10"N', '14°10\'10"S')
    (raw / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    changed = _daks_field_coordinates(raw / "metadata.json")
    assert changed["CPB_small-scale_trial_2019"][0] == -(14 + 10 / 60 + 10 / 3600)
    assert changed["CPB_farm-scale_trial_2021a"] == coordinates["CPB_farm-scale_trial_2021a"]


@pytest.mark.parametrize("modification", ["missing_description", "ambiguous_description", "invalid_dms"])
def test_daks_missing_ambiguous_or_invalid_source_coordinates_fail_closed(tmp_path: Path, modification: str) -> None:
    raw = _raw(tmp_path, daks_metadata=True)
    metadata = json.loads((raw / "metadata.json").read_text())
    if modification == "missing_description":
        metadata["metadata"] = {}
    elif modification == "ambiguous_description":
        metadata["metadata"]["dc.description"] *= 2
    else:
        entry = metadata["metadata"]["dc.description"][0]
        entry["value"] = entry["value"].replace('10°10\'10"N', '10°60\'10"N')
    (raw / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="DaKS"):
        _daks_field_coordinates(raw / "metadata.json")


def test_daks_metadata_is_required_no_hardcoded_location_fallback(tmp_path: Path) -> None:
    raw = _raw(tmp_path)
    with pytest.raises(FileNotFoundError):
        _daks_field_coordinates(raw / "metadata.json")


def test_ris_annual_never_fabricates_first_flight_or_zero_when_trap_off(tmp_path: Path) -> None:
    raw = _raw(tmp_path)
    frame = pd.DataFrame({"Year": [1969, 1970], "Site": ["Example"] * 2, "Count": [0, float("nan")], "Latitude": [52., float("nan")], "Longitude": [0., float("nan")]})
    for name in ["aphidtidy.csv", "mothtidy.csv"]:
        frame.to_csv(raw / "files" / name, index=False)
    result = normalize(_source("ris_annual_1969_2016"), raw, tmp_path / "out")
    rows = pd.read_parquet(tmp_path / "out/observations.parquet")
    assert rows.observation_date.isna().all()
    assert set(rows.observation_year) == {1969, 1970}
    assert rows.loc[rows.observation_year.eq(1970), "value"].isna().all()
    assert rows.loc[rows.observation_year.eq(1970), "explicit_zero"].eq(False).all()
    assert rows.loc[rows.observation_year.eq(1969), "explicit_zero"].eq(True).all()
    assert result["year_range"] == {"minimum": 1969, "maximum": 1970}


def test_kbs_comment_preamble_and_explicit_zero_are_preserved(tmp_path: Path) -> None:
    raw = _raw(tmp_path)
    (raw / "files/aphid_suction_2005_2013.csv").write_text(
        "# These data require publication permission\n"
        "sample_date,state,station,species,sex,count\n"
        "#,,,,,number\n"
        "2013-10-25,IA,Ames,Myzus.persicae,F,0\n"
        "2013-10-25,IA,Ames,Myzus.persicae,M,2\n"
    )
    result = normalize(_source("kbs_aphid_suction_2005_2013"), raw, tmp_path / "out")
    rows = pd.read_parquet(tmp_path / "out/observations.parquet")
    assert result["rows"] == 2
    assert rows.organism.tolist() == ["Myzus persicae"] * 2
    assert rows.sex.tolist() == ["F", "M"]
    assert rows.site_id.eq("IA:Ames").all()
    assert rows.explicit_zero.tolist() == [True, False]
    assert result["organism_rows"] == {"Myzus persicae": 2}


def test_degree_day_reference_is_not_biological_groundtruth(tmp_path: Path) -> None:
    raw = _raw(tmp_path)
    pd.DataFrame({"station": ["Esparto.A"], "date": ["2020-01-01"], "tmin": [38], "tmax": [55]}).to_csv(raw / "files/data__espartoa-weather-2020.csv", index=False)
    pd.DataFrame({"date": ["1/1/2020"], "tmin": [38], "tmax": [55], "sngsine_horiz": [1.19], "sngtri_horiz": [.74]}).to_csv(raw / "files/data__ucipm_results__ucipm_low50_high70_all.csv", index=False)
    result = normalize(_source("ucanr_degree_day_challenge"), raw, tmp_path / "out")
    assert result["rows"] == 0
    assert result["weather_rows"] == 1
    assert result["numerical_reference_rows"] == 2
    ref = pd.read_parquet(tmp_path / "out/numerical_reference.parquet")
    assert ref.unit.eq("degF_day").all()
    assert ref.lower_threshold_f.eq(50).all()
    assert not (tmp_path / "out/observations.parquet").exists()


def test_ris_metadata_keeps_dataset_specific_license() -> None:
    class Client:
        def get_json(self, url, path):
            return {"result": {"title": "RIS annual", "license": "Creative Commons Attribution Licence (CC BY 4.0)", "license_id": None, "resources": [{"url": "https://data.rothamsted.ac.uk/aphidtidy.csv", "size": 1}]}}

    result = _discover_ris_annual(Client())
    assert result["license"]["id"] == "CC-BY-4.0"
    assert "license_id=None" in result["license"]["notes"]
    assert result["files"][0]["role"] == "raw_annual_counts"


def test_unexpected_cpb_schema_fails_without_creating_fake_rows(tmp_path: Path) -> None:
    raw = _raw(tmp_path)
    (raw / "files/WI_CPB_preformatted.csv").write_text("Date,plot_ID\n6/3/2008,a\n")
    with pytest.raises(ValueError, match="schema"):
        normalize(_source("zenodo_18664442"), raw, tmp_path / "out")
