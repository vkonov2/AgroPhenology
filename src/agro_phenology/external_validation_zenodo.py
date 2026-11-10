"""Discover public Zenodo validation records and conservatively normalize them.

Only source observations are represented here.  An unrecorded date is never
inserted, treatment arms are kept separate, and a recorded numeric zero does
not imply absence between visits.  Publisher files remain the source of truth.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any
import zipfile

import pandas as pd


RECORDS: dict[str, dict[str, Any]] = {
    "6352735": {"crops": ["potato"], "organisms": ["Phytophthora infestans"], "evidence_kind": "field_trial_disease_assessments", "normalizer": "potato_trials"},
    "6351794": {"crops": ["vegetables_unspecified_per_record"], "organisms": ["Psila rosae", "Agrotis ipsilon"], "evidence_kind": "field_trap_catches", "normalizer": "vegetable_traps"},
    "6352394": {"crops": ["winter_wheat", "spring_barley"], "organisms": ["cereal_aphids_unspecified_species"], "evidence_kind": "field_infestation_assessments", "normalizer": "cereal_aphids", "scope": "secondary_cross_crop_benchmark"},
    "6361745": {"crops": ["apple"], "organisms": ["Venturia inaequalis"], "evidence_kind": "field_trial_disease_assessments", "normalizer": None, "newer_open_record": "6520484"},
    "6353533": {"crops": ["grapevine"], "organisms": ["Plasmopara viticola"], "evidence_kind": "field_trial_disease_assessments", "normalizer": None, "newer_open_record": "6520937"},
    "6520484": {"crops": ["apple"], "organisms": ["Venturia inaequalis"], "evidence_kind": "field_trial_disease_assessments", "normalizer": "apple_trials", "earlier_restricted_record": "6361745"},
    "6520937": {"crops": ["grapevine"], "organisms": ["Plasmopara viticola"], "evidence_kind": "field_trial_disease_assessments", "normalizer": "grape_trials", "earlier_restricted_record": "6353533"},
    "6521175": {"crops": ["wheat"], "organisms": ["wheat_diseases_source_codes"], "evidence_kind": "field_trial_disease_assessments", "normalizer": "basf_wheat", "scope": "secondary_cross_crop_benchmark"},
    "6390211": {"crops": ["wheat"], "organisms": ["wheat_diseases_source_codes"], "evidence_kind": "field_trial_disease_assessments", "normalizer": "corteva_wheat", "scope": "secondary_cross_crop_benchmark"},
    "18351833": {"crops": ["potato"], "organisms": ["Phytophthora infestans"], "evidence_kind": "spatial_disease_severity_without_assessment_dates", "normalizer": "spatial_late_blight"},
    "14989522": {"crops": ["grapevine"], "organisms": ["down_mildew_source_label", "powdery_mildew_source_label", "botrytis_source_label", "erysiphe_source_label"], "evidence_kind": "daily_sensor_data_with_unverified_disease_labels", "normalizer": "agridatavalue"},
}

# The source describes these frequency/intensity measures as percentages.
# Raw CLASSES assessments have no verified unit and must not inherit one.
APPLE_PERCENT_MEASURES = {
    "LEAF_FREQ", "LEAF_INTENSITY", "FRUIT_FREQ", "FRUIT_INTENSITY",
    "BEFHKT_LE", "BEFHKT_FR", "BEFWER_LE", "BEFWER_FR",
    "VENTIN_LEAF_FREQ", "VENTIN_FRUIT_FREQ", "VENTIN_FRUIT_INF", "VENTIN_LEAF_INF",
    "P%INF_LEAVES", "P%FREQ_LEAVES", "P%FREQ_FRUIT", "P%INF_FRUIT",
}


def discover(client: Any) -> list[dict[str, Any]]:
    """Archive record metadata and list only useful public data/support files."""
    sources = []
    for record_id, spec in RECORDS.items():
        source_id = f"zenodo_{record_id}"
        source: dict[str, Any] = {
            "source_id": source_id, "provider": "Zenodo",
            "title": f"Zenodo record {record_id}",
            "landing_url": f"https://zenodo.org/records/{record_id}",
            "metadata_url": f"https://zenodo.org/api/records/{record_id}",
            "access_status": "unavailable", "license": {"id": None, "url": None, "notes": "Not yet retrieved"},
            "evidence_kind": spec["evidence_kind"], "crops": spec["crops"],
            "organisms": spec["organisms"], "limitations": [], "files": [],
            "normalizer": spec["normalizer"], "scope": spec.get("scope", "garden_crop_validation"),
        }
        for relation in ("newer_open_record", "earlier_restricted_record"):
            if relation in spec:
                source[relation] = f"zenodo_{spec[relation]}"
        try:
            record = client.get_json(source["metadata_url"], f"raw/{source_id}/metadata.json", max_bytes=4_000_000)
        except (OSError, ValueError) as exc:
            source["limitations"].append(f"Metadata retrieval failed: {exc}")
            sources.append(source)
            continue
        metadata = record["metadata"]
        source["title"] = metadata.get("title", source["title"])
        source["doi"] = metadata.get("doi")
        source["record_parent_url"] = record.get("links", {}).get("parent")
        source["record_version"] = metadata.get("version")
        license_id = (metadata.get("license") or {}).get("id")
        source["license"] = {
            "id": license_id,
            "url": "https://creativecommons.org/licenses/by/4.0/" if license_id == "cc-by-4.0" else None,
            "notes": "As stated in the archived Zenodo record; individual file notices also retained.",
        }
        access = metadata.get("access_right", "unknown")
        source["access_status"] = "open" if access == "open" else "restricted" if access in {"restricted", "closed", "embargoed"} else "metadata_only"
        for item in record.get("files", []):
            name = item["key"]
            if not name.lower().endswith((".txt", ".csv", ".tsv", ".r", ".rdata", ".zip", ".md", ".pdf")):
                continue
            source["files"].append({
                "name": name, "url": item["links"]["self"],
                "role": "data" if name.lower().endswith((".txt", ".csv", ".tsv", ".zip")) else "supporting_code_or_documentation",
                "expected_checksum": item.get("checksum"), "size_bytes": int(item["size"]),
            })
        source["limitations"].extend(_limitations(record_id))
        if source["access_status"] != "open":
            source["files"] = []
            source["limitations"].append("Restricted historical record; no access request is submitted automatically.")
        sources.append(source)
    return sources


def _limitations(record_id: str) -> list[str]:
    if record_id in {"6352735", "6520484", "6520937", "6521175", "6390211"}:
        result = [
            "Treatment arms must remain separate; untreated does not establish absence of artificial inoculation.",
            "Coordinates in this collection are rounded; they do not identify an exact weather station.",
            "No hourly weather is included. Missing visits/dates are not negative disease observations.",
            "Split validation by trial/location/year, not randomly by repeated assessments or treatment arms.",
        ]
        if record_id == "6352735":
            result.append("Value units are not stated in the TSV; preserved as source units until an assessment-unit audit.")
        if record_id == "6520937":
            result.append("Wide untreated/treated group assessments are reshaped without inventing individual plots or replicates.")
        if record_id == "6520484":
            result.append("CLASSES has unverified source units; never pool raw class scores with percentage severity/frequency.")
        return result
    if record_id == "6351794":
        return ["Trap catches indicate recorded adult captures, not larval damage or disease.", "Sampling interval, number of traps and mean/sum convention require audit; fractional catches are preserved.", "Crop is not identified per source row; no host crop is invented."]
    if record_id == "6352394":
        return ["Percentage of infested cereal tillers, not species-specific aphid trap counts.", "Cereal hosts are a secondary benchmark, not direct validation on garden crops.", "Duplicate source rows are retained; never invent replicate identities."]
    if record_id == "18351833":
        return ["Assessment dates are absent; file year is retained separately and is not converted to January 1.", "Native coordinates use EPSG:26917 metres, not longitude/latitude degrees.", "Useful for spatial severity and treatment comparison; not ready for first-event weather validation."]
    if record_id == "14989522":
        return ["Only original members are normalized; synthetic and noised members remain raw and excluded from ground truth.", "Disease-label derivation and inspection/treatment provenance are unverified; zeros are source labels only.", "Weather is daily aggregated, not hourly measured leaf wetness; original weather units are not guessed.", "No site coordinates are specified in the archived files; unsuitable for coordinate-based weather matching without clarification."]
    return []


def _read_tsv(path_or_buffer: Any) -> pd.DataFrame:
    return pd.read_csv(path_or_buffer, sep="\t", keep_default_na=True, dtype=str)


def _base(source: dict, frame: pd.DataFrame, *, date_column: str | None, file_name: str) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    result["source_id"] = source["source_id"]
    result["record_id"] = [f"{source['source_id']}:{file_name}:{i+2}" for i in range(len(frame))]
    result["source_file"] = file_name
    result["source_row_number"] = pd.Series(range(2, len(frame) + 2), index=frame.index, dtype="int64")
    result["site_id"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    result["observation_date"] = pd.to_datetime(frame[date_column], errors="raise") if date_column else pd.NaT
    result["organism"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    result["crop"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    result["measurement"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    result["value"] = float("nan")
    result["unit"] = "source_units_unverified"
    result["stage"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    result["treatment"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in frame:
        result["source_" + column.lower()] = frame[column]
    return result


def _zero(frame: pd.DataFrame) -> None:
    frame["explicit_zero"] = frame["value"].eq(0).astype("boolean").where(frame["value"].notna(), pd.NA)


def _coordinate_token(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    number = float(value)
    if not math.isfinite(number):
        return "unknown"
    return "0" if number == 0 else repr(number)


def _spatial_site_ids(source: dict, frame: pd.DataFrame) -> pd.Series:
    """Source trial counters repeat across years and locations; never use alone."""
    countries = frame.Country.fillna("unspecified_country").astype(str).str.strip().str.casefold() if "Country" in frame else pd.Series("unspecified_country", index=frame.index)
    return (source["source_id"] + ":country:" + countries + ":lat:" + frame.Latitude.map(_coordinate_token)
            + ":lon:" + frame.Longitude.map(_coordinate_token))


def _trial_ids(source: dict, frame: pd.DataFrame, observations: pd.DataFrame, trial_column: str) -> pd.Series:
    years = observations.observation_date.dt.year.astype("Int64").astype("string")
    return (source["source_id"] + ":year:" + years + ":trial:" + frame[trial_column].astype("string")
            + ":site:" + observations.site_id)


def _locations(source: dict, frame: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    if not {"Latitude", "Longitude"}.issubset(frame):
        return pd.DataFrame(columns=["source_id", "site_id", "latitude", "longitude", "coordinate_notes"])
    locations = pd.DataFrame({
        "source_id": source["source_id"], "site_id": observations["site_id"],
        "latitude": pd.to_numeric(frame["Latitude"], errors="raise"),
        "longitude": pd.to_numeric(frame["Longitude"], errors="raise"),
        "coordinate_notes": "Public source coordinates; rounding retained without assigning field precision.",
    })
    if not (locations.latitude.dropna().between(-90, 90).all() and locations.longitude.dropna().between(-180, 180).all()):
        raise ValueError("Source latitude/longitude outside geographic range")
    locations = locations.drop_duplicates().reset_index(drop=True)
    if locations.site_id.duplicated().any():
        raise ValueError("A normalized site_id refers to multiple coordinate pairs")
    return locations


def normalize(source: dict, raw_dir: Path, output_dir: Path) -> dict:
    """Normalize locally downloaded files; no network calls or target fitting."""
    kind = source.get("normalizer")
    if source.get("access_status") != "open" or not kind:
        return {"rows": 0, "schema": {}, "date_range": None, "endpoints": [], "limitations": source["limitations"], "output_files": []}
    files_dir = Path(raw_dir) / "files"
    if kind == "agridatavalue":
        return _normalize_agridatavalue(source, files_dir / "environmental_data.zip", Path(output_dir))
    if kind == "spatial_late_blight":
        return _normalize_spatial(source, files_dir, Path(output_dir))
    candidates = [f for f in source["files"] if f["name"].lower().endswith(".txt")]
    if len(candidates) != 1:
        raise ValueError(f"Expected one source TSV for {source['source_id']}")
    filename = candidates[0]["name"]
    frame = _read_tsv(files_dir / filename)
    observations = _base(source, frame, date_column="Date", file_name=filename)
    if kind == "vegetable_traps":
        observations["site_id"] = _spatial_site_ids(source, frame) + ":field:" + frame.FieldId
        observations["organism"] = frame.Species.map({"Carrot fly": "Psila rosae", "Cutworm": "Agrotis ipsilon"}).fillna("source:" + frame.Species)
        observations["measurement"] = "trap_catch"
        observations["stage"] = "adult_trap_catch"
        observations["unit"] = "source_trap_catch_value"
        observations["value"] = pd.to_numeric(frame.Value, errors="raise")
        endpoints = ["first_recorded_adult_capture_with_sampling_interval_censoring", "seasonal_trap_catch_phenology"]
    elif kind == "cereal_aphids":
        observations["site_id"] = _spatial_site_ids(source, frame) + ":field:" + frame.FieldId
        observations["organism"] = "cereal_aphids_unspecified_species"
        observations["crop"] = frame.Crop.map({"WinterWheat": "winter_wheat", "SpringBarley": "spring_barley"}).fillna("source:" + frame.Crop)
        observations["measurement"] = "infested_tillers_percent"
        observations["unit"] = "percent"
        observations["stage"] = "source_life_stage_unspecified"
        observations["value"] = pd.to_numeric(frame.Value, errors="raise")
        endpoints = ["first_recorded_tiller_infestation", "cereal_aphid_infestation_progression"]
    elif kind == "potato_trials":
        observations["site_id"] = _spatial_site_ids(source, frame)
        observations["trial_id"] = _trial_ids(source, frame, observations, "Trial")
        observations["organism"] = "Phytophthora infestans"
        observations["crop"] = "potato"
        observations["measurement"] = "late_blight_assessment_value"
        observations["treatment"] = frame.Treatment
        observations["untreated_control"] = _untreated_flag(frame.Untreated)
        observations["plot_id"] = observations.trial_id + ":plot:" + frame.Plot
        observations["crop_growth_stage"] = frame.CropStage
        observations["value"] = pd.to_numeric(frame.Value, errors="raise")
        endpoints = ["first_positive_assessment_on_untreated_plot_after_inoculation_audit", "disease_assessment_progression"]
    elif kind == "apple_trials":
        observations["site_id"] = _spatial_site_ids(source, frame)
        observations["trial_id"] = _trial_ids(source, frame, observations, "Trial")
        observations["organism"] = "Venturia inaequalis"
        observations["crop"] = "apple"
        observations["measurement"] = "apple_scab_" + frame.Measure.str.lower()
        observations["unit"] = frame.Measure.map({code: "percent" for code in APPLE_PERCENT_MEASURES}).fillna("source_units_unverified")
        observations["treatment"] = frame.Treatment
        observations["untreated_control"] = _untreated_flag(frame.Untreated)
        observations["value"] = pd.to_numeric(frame.Value, errors="raise")
        endpoints = ["first_positive_source_measure_assessment_on_untreated_trial_group", "scab_frequency_or_severity_progression_by_measure"]
    elif kind == "grape_trials":
        observations["site_id"] = _spatial_site_ids(source, frame)
        observations["trial_id"] = _trial_ids(source, frame, observations, "TrialId")
        observations["organism"] = "Plasmopara viticola"
        observations["crop"] = "grapevine"
        observations["measurement"] = "grape_downy_mildew_" + frame.PlantPart.str.lower() + "_" + frame.Method
        observations["unit"] = "percent"
        arms = []
        for arm in ("Untreated", "Treated"):
            copy = observations.copy()
            copy["record_id"] += ":" + arm.lower()
            copy["value"] = pd.to_numeric(frame[arm], errors="raise")
            copy["treatment"] = arm.lower() + "_group"
            copy["untreated_control"] = arm == "Untreated"
            arms.append(copy)
        observations = pd.concat(arms, ignore_index=True)
        endpoints = ["first_positive_source_measure_assessment_on_untreated_trial_group", "downy_mildew_assessment_progression_by_plant_part_and_method"]
    elif kind in {"basf_wheat", "corteva_wheat"}:
        observations["site_id"] = _spatial_site_ids(source, frame)
        observations["trial_id"] = _trial_ids(source, frame, observations, "TrialId" if kind == "basf_wheat" else "Field")
        observations["organism"] = "source_eppo:" + (frame.Organism if kind == "basf_wheat" else frame.PestCode)
        observations["crop"] = "wheat"
        observations["measurement"] = "source:" + (frame.Parameter if kind == "basf_wheat" else frame.EvaluationType)
        observations["treatment"] = frame.Treatment
        observations["untreated_control"] = frame.Treatment.eq("Untreated").astype("boolean").where(frame.Treatment.notna(), pd.NA)
        observations["value"] = pd.to_numeric(frame.Value, errors="raise")
        endpoints = ["secondary_cross_crop_assessment_benchmark_after_measure_unit_audit"]
    else:
        raise ValueError(f"Unsupported normalizer: {kind}")
    _zero(observations)
    # Grape arm expansion changes row positions; source coordinates must expand
    # by the same raw-row mapping rather than being silently misaligned.
    location_frame = frame if kind != "grape_trials" else pd.concat([frame, frame], ignore_index=True)
    locations = _locations(source, location_frame, observations)
    return _write(source, Path(output_dir), observations, locations, endpoints)


def _untreated_flag(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if not numeric.dropna().isin([0, 1]).all():
        raise ValueError("Untreated source flag is not binary")
    return numeric.eq(1).astype("boolean").where(numeric.notna(), pd.NA)


def _write(source: dict, output_dir: Path, observations: pd.DataFrame, locations: pd.DataFrame,
           endpoints: list[str], extras: dict[str, pd.DataFrame] | None = None,
           extra_summary: dict | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    observations["value_domain_valid"] = pd.Series(pd.NA, index=observations.index, dtype="boolean")
    percent = observations.unit.eq("percent") & observations.value.notna()
    observations.loc[percent, "value_domain_valid"] = observations.loc[percent, "value"].between(0, 100)
    tables = {"observations.parquet": observations, "locations.parquet": locations, **(extras or {})}
    for name, table in tables.items():
        table.to_parquet(output_dir / name, index=False)
    valid_dates = observations.observation_date.dropna()
    result = {
        "rows": int(len(observations)), "schema": {c: str(t) for c, t in observations.dtypes.items()},
        "date_range": [str(valid_dates.min().date()), str(valid_dates.max().date())] if len(valid_dates) else None,
        "endpoints": endpoints, "limitations": source["limitations"], "output_files": list(tables),
        "sites": int(observations.site_id.nunique()), "location_rows": int(len(locations)),
        "explicit_zero_records": int(observations.explicit_zero.fillna(False).sum()),
        "units_unverified_records": int(observations.unit.eq("source_units_unverified").sum()),
        "invalid_percent_records": int(observations.value_domain_valid.eq(False).fillna(False).sum()),
        "source_rows_preserved": True,
        **(extra_summary or {}),
    }
    (output_dir / "normalization_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["output_files"].append("normalization_summary.json")
    return result


def _normalize_spatial(source: dict, files_dir: Path, output_dir: Path) -> dict:
    parts, places = [], []
    for item in source["files"]:
        if not item["name"].endswith(".csv"):
            continue
        frame = pd.read_csv(files_dir / item["name"], dtype=str)
        observations = _base(source, frame, date_column=None, file_name=item["name"])
        year = next(y for y in ("2022", "2023") if y in item["name"])
        observations["observation_year"] = int(year)
        observations["site_id"] = source["source_id"] + ":year:" + year + ":cell:" + frame.cell
        observations["organism"] = "Phytophthora infestans"
        observations["crop"] = "potato"
        observations["measurement"] = "mean_cell_infection_severity_source_units"
        observations["value"] = pd.to_numeric(frame.mean_sev_wp, errors="raise")
        observations["treatment"] = frame.treatm.map({"0": "untreated_control", "1": "sprayed"})
        observations["untreated_control"] = frame.treatm.eq("0").astype("boolean").where(frame.treatm.notna(), pd.NA)
        _zero(observations)
        parts.append(observations)
        places.append(pd.DataFrame({"source_id": source["source_id"], "site_id": observations.site_id,
                                    "x": pd.to_numeric(frame.x, errors="raise"), "y": pd.to_numeric(frame.y, errors="raise"),
                                    "crs": "EPSG:26917", "coordinate_unit": "metre"}))
    return _write(source, output_dir, pd.concat(parts, ignore_index=True), pd.concat(places, ignore_index=True).drop_duplicates(),
                  ["spatial_severity_and_treatment_association_only_no_temporal_validation"])


def _normalize_agridatavalue(source: dict, archive_path: Path, output_dir: Path) -> dict:
    parts, weather_parts, excluded = [], [], []
    with zipfile.ZipFile(archive_path) as archive:
        if sum(f.file_size for f in archive.infolist()) > 100_000_000:
            raise ValueError("Unexpected oversized AgriDataValue archive")
        for member in archive.infolist():
            if member.is_dir():
                continue
            name = Path(member.filename).name
            if "_original_" not in name.lower() or not name.lower().endswith(".csv"):
                excluded.append(member.filename)
                continue
            frame = _read_tsv(io.BytesIO(archive.read(member)))
            frame["parsed_date"] = pd.to_datetime(frame.date, format="%m/%d/%Y", errors="raise")
            weather = frame.copy()
            weather["source_id"] = source["source_id"]
            weather["site_id"] = source["source_id"] + ":unlocated_vineyard"
            weather["observation_date"] = frame.parsed_date
            weather["source_archive_member"] = member.filename
            weather_parts.append(weather)
            base = _base(source, frame, date_column="parsed_date", file_name=name)
            base["site_id"] = source["source_id"] + ":unlocated_vineyard"
            base["crop"] = "grapevine"
            base["unit"] = "binary_source_label_not_verified_field_diagnosis"
            base["label_provenance"] = "unverified_source_label"
            base["synthetic"] = False
            base["source_archive_member"] = member.filename
            for label in ("down_mildew", "powdery_mildew", "botrytis", "erysiphe"):
                obs = base.copy()
                obs["record_id"] += ":" + label
                obs["organism"] = label + "_source_label"
                obs["measurement"] = "unverified_source_binary_disease_label"
                obs["value"] = pd.to_numeric(frame[label], errors="raise")
                if not obs.value.dropna().isin([0, 1]).all():
                    raise ValueError(f"Nonbinary source disease label: {label}")
                _zero(obs)
                parts.append(obs)
    if not parts:
        raise ValueError("No original AgriDataValue source members")
    return _write(source, output_dir, pd.concat(parts, ignore_index=True),
                  pd.DataFrame(columns=["source_id", "site_id", "latitude", "longitude"]),
                  ["auxiliary_daily_sensor_label_associations_only_pending_label_provenance_audit"],
                  {"original_daily_weather.parquet": pd.concat(weather_parts, ignore_index=True)},
                  {"excluded_archive_members": excluded, "synthetic_rows_normalized": 0,
                   "validation_readiness": "auxiliary_only_pending_label_audit"})
