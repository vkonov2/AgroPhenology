"""Public insect observations and numerical degree-day reference inputs.

Acquisition only: no fitting, no daily outcome imputation, and no inference of
unobserved zeros. Discovery archives source metadata using the common client.
Normalizers keep specimen counts, life stages and annual sums as distinct
measurements. Weather aggregates supplied with CPB data are not model inputs.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd


DAKS_ITEM = "aeb5b8aa-ece2-4bfc-a915-087cd29f1dbf"
RIS_PACKAGE = "df18fc21-ab5e-4c9c-bb9f-731f6e07a73f"
GITHUB_REPO = "UCANR-IGIS/degree-day-challenge"
STAGES = {
    "LEPDEC_I": "adult",
    "LEPDEC_E": "egg",
    "LEPDEC_ED": "damaged_egg",
    "LEPDEC_L1": "larva_1",
    "LEPDEC_L2": "larva_2",
    "LEPDEC_L3": "larva_3",
    "LEPDEC_L4": "larva_4",
}


def _spec(source_id: str, **kwargs: Any) -> dict:
    return {"source_id": source_id, "files": [], "limitations": [], **kwargs}


def _discover_daks(client: Any) -> dict:
    source_id = "daks_cpb_2019_2022"
    root = "https://daks.uni-kassel.de/server/api/core"
    item = client.get_json(f"{root}/items/{DAKS_ITEM}", f"raw/{source_id}/metadata.json")
    bundles = client.get_json(f"{root}/items/{DAKS_ITEM}/bundles", f"raw/{source_id}/bundles.json")
    original = next(b for b in bundles["_embedded"]["bundles"] if b["name"] == "ORIGINAL")
    streams = client.get_json(original["_links"]["bitstreams"]["href"], f"raw/{source_id}/bitstreams.json")
    files = []
    for stream in streams["_embedded"]["bitstreams"]:
        if not stream["name"].lower().endswith((".csv", ".txt")):
            continue
        checksum = stream.get("checkSum", {})
        files.append({
            "name": stream["name"], "url": stream["_links"]["content"]["href"],
            "role": "raw_observations" if stream["name"].endswith(".csv") else "data_dictionary",
            "size_bytes": stream["sizeBytes"],
            "expected_checksum": f"{checksum['checkSumAlgorithm'].lower()}:{checksum['value']}",
        })
    return _spec(
        source_id, provider="DaKS Universität Kassel", title=item["name"],
        landing_url="https://doi.org/10.48662/daks-28", access_status="open",
        license={"id": "CC-BY-NC-SA-4.0", "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/", "notes": "Non-commercial/share-alike conditions; private research collection only."},
        evidence_kind="repeated_plant_stage_counts", crops=["potato"], organisms=["Leptinotarsa decemlineata"], files=files,
        limitations=["Monitoring starts when first eggs are observed; cannot validate pre-biofix first arrival.", "Mulch treatments change populations: control must be analyzed separately.", "Multiple plants and stages are not independent field-season events."],
    )


def _discover_cpb_wisconsin(client: Any) -> dict:
    source_id = "zenodo_18664442"
    data = client.get_json("https://zenodo.org/api/records/18664442", f"raw/{source_id}/metadata.json")
    return _spec(
        source_id, provider="Zenodo", title=data["metadata"]["title"], landing_url="https://zenodo.org/records/18664442", access_status="open",
        license={"id": data["metadata"].get("license", {}).get("id"), "url": "https://creativecommons.org/licenses/by/4.0/", "notes": "Dataset-specific CC BY 4.0 metadata."},
        evidence_kind="repeated_anonymous_plot_stage_abundance", crops=["potato"], organisms=["Leptinotarsa decemlineata"],
        measurement_reference="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0345180",
        limitations=["Geographic coordinates deliberately removed by authors.", "Weather/landscape features aggregate biological periods and may contain future information; not approved past-only predictors.", "Adults and larvae are mean individuals per 25 sweep-net sweeps; the eggs column is egg masses per plant, not individual eggs. Do not pool with DaKS individual-egg counts.", "Commercial fields were conventionally managed; within-season management histories are not supplied.", "Fractional values and source NA are retained; the authors' R preprocessing that replaces NA by zero is not applied."],
        files=[{"name": f["key"], "url": f["links"]["self"], "size_bytes": f["size"], "expected_checksum": f["checksum"], "role": "raw_observations" if f["key"].endswith(".csv") else "analysis_code_reference"} for f in data.get("files", [])],
    )


def _discover_ris_annual(client: Any) -> dict:
    source_id = "ris_annual_1969_2016"
    data = client.get_json(f"https://data.rothamsted.ac.uk/api/3/action/package_show?id={RIS_PACKAGE}", f"raw/{source_id}/metadata.json")["result"]
    files = []
    for resource in data["resources"]:
        name = Path(urlparse(resource["url"]).path).name
        files.append({"name": name, "url": resource["url"], "size_bytes": resource.get("size"), "role": "raw_annual_counts" if name.lower() in {"mothtidy.csv", "aphidtidy.csv"} else "derived_trend_not_observation"})
    return _spec(
        source_id, provider="Rothamsted e-RA CKAN", title=data["title"], landing_url=f"https://data.rothamsted.ac.uk/dataset/{RIS_PACKAGE}", access_status="open",
        license={"id": "CC-BY-4.0", "url": "https://creativecommons.org/licenses/by/4.0/", "notes": f"Specific CKAN license text: {data.get('license')}; license_id={data.get('license_id')}. Dataset-specific terms preserved, not substituted by site-wide terms."},
        evidence_kind="annual_all_species_trap_sums", crops=[], organisms=["Aphididae aggregate", "moths aggregate"], files=files,
        limitations=["Annual sums across species cannot validate first flight dates or species-specific alerts.", "NA indicates trap not operating; never convert NA to zero.", "Six trend files are derived predictions, not additional ground-truth observations."],
    )


def _discover_dd_challenge(client: Any) -> dict:
    source_id = "ucanr_degree_day_challenge"
    api = f"https://api.github.com/repos/{GITHUB_REPO}"
    repo = client.get_json(api, f"raw/{source_id}/metadata.json")
    commit = client.get_json(f"{api}/commits/{repo['default_branch']}", f"raw/{source_id}/commit.json")
    sha = commit["sha"]
    tree = client.get_json(f"{api}/git/trees/{sha}?recursive=1", f"raw/{source_id}/tree.json")
    files = []
    for item in tree["tree"]:
        path = item["path"]
        if not (path in {"LICENSE", "README.md"} or (path.startswith("data/") and path.endswith(".csv"))):
            continue
        files.append({"name": path.replace("/", "__"), "url": f"https://raw.githubusercontent.com/{GITHUB_REPO}/{sha}/{path}", "size_bytes": item.get("size"), "role": "numerical_reference" if "ucipm_results" in path else "weather_reference" if path.endswith(".csv") else "documentation"})
    return _spec(
        source_id, provider="UC ANR IGIS GitHub", title="Degree Days Validation Challenge", landing_url="https://ucanr-igis.github.io/degree-day-challenge/", access_status="open",
        license={"id": (repo.get("license") or {}).get("spdx_id"), "url": f"https://github.com/{GITHUB_REPO}/blob/{sha}/LICENSE", "notes": "Pinned repository commit and license are archived."},
        evidence_kind="numerical_implementation_benchmark", crops=[], organisms=[], files=files, commit=sha,
        limitations=["Numerical formula verification only, not biological early-warning validation.", "Temperature and thresholds are Fahrenheit; preserve units and convert explicitly."],
    )


def _discover_kbs(client: Any) -> dict:
    source_id = "kbs_aphid_suction_2005_2013"
    client.get_text("https://lter.kbs.msu.edu/datatables/122", f"raw/{source_id}/metadata.html")
    client.get_text("https://lter.kbs.msu.edu/datasets/52.eml", f"raw/{source_id}/metadata.eml", max_bytes=5_000_000)
    client.get_text("https://lter.kbs.msu.edu/data/terms-of-use/", f"raw/{source_id}/data_use_policy.html")
    return _spec(
        source_id, provider="KBS LTER Michigan State", title="Aphid Colonization via Suction Traps", landing_url="https://lter.kbs.msu.edu/datatables/122", access_status="open",
        license={"id": "KBS-standard-data-use-policy", "url": "https://lter.kbs.msu.edu/data/terms-of-use/", "notes": "Open download URL and public read in EML. Source explicitly requires written permission before publishing core data. Collection is local; no redistribution/publication permission inferred."},
        evidence_kind="dated_species_sex_trap_counts", crops=["soybean_and_other_crops"], organisms=["Aphididae"],
        files=[{"name": "aphid_suction_2005_2013.csv", "url": "https://lter.kbs.msu.edu/datatables/122.csv", "role": "raw_observations", "max_bytes": 200_000_000}],
        limitations=["Core data publication requires permission; keep raw data local.", "Species relevance and station coordinate precision must be checked.", "Values are counts per collection, predominantly weekly; collection date is not proof of daily exposure. No implicit daily rows or missing zeros are added.", "Taxon labels include genus-level/aggregate identifications, not only resolved species."],
    )


def _discover_reference_sites(client: Any) -> list[dict]:
    specs = []
    source_id = "pennstate_frec_trap_graphs"
    url = "https://agsci.psu.edu/research/centers-facilities/extension/frec/growing-season-information/pheromone-traps"
    client.get_text(url, f"raw/{source_id}/metadata.html")
    specs.append(_spec(source_id, provider="Penn State FREC", title="Weekly fruit pest trap graphs", landing_url=url, access_status="metadata_only", license={"id": "not_established", "notes": "Web page and graphs are public; no tabular historical download was found."}, evidence_kind="graph_only_not_numeric_groundtruth", crops=["apple"], organisms=["Cydia pomonella", "Grapholita molesta", "leafrollers"], limitations=["Only graph images confirmed; no numeric values are inferred or digitized."]))
    source_id = "ris_daily_data_warehouse"
    url = "https://insectsurvey.com/get-data/"
    client.get_text(url, f"raw/{source_id}/metadata.html")
    client.get_text("https://insectsurvey.com/legal-notice/", f"raw/{source_id}/license.html")
    specs.append(_spec(source_id, provider="Rothamsted Insect Survey", title="RIS daily and weekly data warehouse", landing_url=url, access_status="permission_required", license={"id": "CC-BY-NC-4.0-with-additional-terms", "url": "https://insectsurvey.com/legal-notice/", "notes": "Separate from the Bell annual CC BY dataset."}, evidence_kind="daily_species_trap_counts_access_not_established", crops=[], organisms=["Aphididae", "moths"], limitations=["Warehouse routes to login; no authorized bulk daily export available in this collection."]))
    return specs


def _discover_dryad_biofix(client: Any) -> dict:
    source_id = "dryad_montana_cydia_biofix"
    base = "https://datadryad.org"
    data = client.get_json(f"{base}/api/v2/datasets/doi%3A10.5061%2Fdryad.h18931zjc", f"raw/{source_id}/metadata.json")
    version = data["_links"]["stash:version"]["href"]
    files = client.get_json(urljoin(base, version + "/files"), f"raw/{source_id}/files_metadata.json")
    candidates = []
    for file in files.get("_embedded", {}).get("stash:files", []):
        file_id = file["_links"]["self"]["href"].rsplit("/", 1)[-1]
        candidates.append({"name": file["path"], "url": f"{base}/downloads/file_stream/{file_id}", "size_bytes": file["size"], "expected_checksum": f"sha256:{file['digest']}"})
    return _spec(
        source_id, provider="Dryad", title=data["title"], landing_url="https://doi.org/10.5061/dryad.h18931zjc", access_status="metadata_only",
        license={"id": "CC0-1.0", "url": data.get("license"), "notes": "Public metadata declares CC0; XLSX has not been downloaded."},
        evidence_kind="reported_observed_biofix_and_temperature_model_comparison", crops=["apple"], organisms=["Cydia pomonella"], candidate_files=candidates,
        limitations=["Montana 2018–2020 biofix comparison: not necessarily a complete seasonal catch series.", "Official website file_stream returned HTTP 403 and API download returned HTTP 401 in access preflight; no access-control bypass attempted.", "Metadata and exact missing file/checksum saved; no numerical ground-truth rows claimed."],
    )


def _discover_peerj_cydia(client: Any) -> dict:
    source_id = "peerj_20226_cydia_trap_counts"
    data = client.get_json("https://api.crossref.org/works/10.7717/peerj.20226", f"raw/{source_id}/metadata.json")["message"]
    return _spec(
        source_id, provider="PeerJ", title=data["title"][0], landing_url="https://doi.org/10.7717/peerj.20226", access_status="metadata_only",
        license={"id": "CC-BY-4.0", "url": "https://creativecommons.org/licenses/by/4.0/", "notes": "Open-access article identifies supplementary raw counts; raw XLSX files not retrieved."},
        evidence_kind="reported_weekly_cydia_trap_counts_and_fruit_infestation", crops=["apple"], organisms=["Cydia pomonella"],
        candidate_files=[{"name": f"peerj-13-20226-s00{i}.xlsx", "url": f"https://doi.org/10.7717/peerj.20226/supp-{i}"} for i in (1, 2)],
        limitations=["2023/2024 mating-disruption and control orchards; intervention groups must be kept separate.", "Publisher landing page returned HTTP 403/Cloudflare and PMC article returned browser challenge during preflight; no challenge bypass attempted.", "Only Crossref metadata and exact supplementary DOI references collected; no numeric observations or coordinates extracted."],
    )


def discover(client: Any) -> list[dict]:
    """Return source specifications; an unavailable host does not hide others."""
    results = []
    funcs = [
        ("daks_cpb_2019_2022", _discover_daks),
        ("zenodo_18664442", _discover_cpb_wisconsin),
        ("ris_annual_1969_2016", _discover_ris_annual),
        ("ucanr_degree_day_challenge", _discover_dd_challenge),
        ("kbs_aphid_suction_2005_2013", _discover_kbs),
        ("dryad_montana_cydia_biofix", _discover_dryad_biofix),
        ("peerj_20226_cydia_trap_counts", _discover_peerj_cydia),
    ]
    for source_id, function in funcs:
        try:
            results.append(function(client))
        except (OSError, ValueError, KeyError, StopIteration) as exc:
            results.append(_spec(source_id, provider="external_insect_sources", title=source_id, landing_url="", access_status="unavailable", license={"id": "not_established"}, evidence_kind="source_discovery_failed", limitations=[f"{type(exc).__name__}: {exc}"]))
    try:
        results.extend(_discover_reference_sites(client))
    except (OSError, ValueError, KeyError) as exc:
        results.append(_spec("insect_reference_sites", provider="external_insect_sources", title="Reference-site access", landing_url="", access_status="unavailable", license={"id": "not_established"}, evidence_kind="reference_site_failed", limitations=[str(exc)]))
    return results


def _date_range(dates: pd.Series) -> dict:
    dates = pd.to_datetime(dates.dropna())
    return {"minimum": dates.min().date().isoformat() if len(dates) else None, "maximum": dates.max().date().isoformat() if len(dates) else None}


def _write_tables(output_dir: Path, tables: dict[str, pd.DataFrame], *, limitations: list[str], endpoints: list[str]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, table in tables.items():
        path = output_dir / f"{name}.parquet"
        table.to_parquet(path, index=False)
        outputs.append(path.name)
    observations = tables.get("observations", pd.DataFrame())
    return {
        "rows": len(observations), "table_rows": {name: len(table) for name, table in tables.items()},
        "schema": {name: str(dtype) for name, dtype in observations.dtypes.items()},
        "date_range": _date_range(observations["observation_date"]) if "observation_date" in observations else {},
        "endpoints": endpoints, "limitations": limitations, "output_files": outputs,
    }


def _daks_field_coordinates(metadata_path: Path) -> dict[str, tuple[float, float]]:
    """Read locations locally from the archived source, never embed field data.

    The source describes one shared small-scale site and three labelled
    farm-scale trials. Require one unambiguous DMS pair for each description;
    a changed or missing source description must not silently assign a site.
    """
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    descriptions = metadata.get("metadata", {}).get("dc.description", [])
    text = "\n".join(entry.get("value", "") for entry in descriptions)
    number = r"(\d+(?:\.\d+)?)"
    dms = number + r"\s*[°º]\s*" + number + r"\s*['′]\s*" + number + r'\s*["″]\s*([NSEW])'
    pair = dms + r"\s*,?\s*" + dms

    def extract(prefix: str) -> tuple[float, float]:
        matches = list(re.finditer(prefix + r"\s*\(" + pair + r"\)", text, re.DOTALL))
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous DaKS trial coordinates in local metadata")
        values = matches[0].groups()
        coordinates = []
        for offset, hemispheres, limit in [(0, {"N", "S"}, 90), (4, {"E", "W"}, 180)]:
            degrees, minutes, seconds = map(float, values[offset:offset + 3])
            direction = values[offset + 3]
            value = degrees + minutes / 60 + seconds / 3600
            if direction not in hemispheres or minutes >= 60 or seconds >= 60 or value > limit:
                raise ValueError("Invalid DaKS DMS coordinates in local metadata")
            coordinates.append(-value if direction in {"S", "W"} else value)
        return tuple(coordinates)

    small = extract(r"Small-scale field trials[^()]*")
    locations = {f"CPB_small-scale_trial_{year}": small for year in (2019, 2020, 2021)}
    locations["CPB_farm-scale_trial_2021a"] = extract(r"conducted in 2021a\b[^()]*")
    for trial in ("2021b", "2022"):
        locations[f"CPB_farm-scale_trial_{trial}"] = extract(r"\b" + trial)
    return locations


def _normalise_daks(source: dict, raw_dir: Path, output_dir: Path) -> dict:
    observations, locations = [], []
    field_coordinates = _daks_field_coordinates(raw_dir / "metadata.json")
    for path in sorted((raw_dir / "files").glob("*.csv")):
        frame = pd.read_csv(path, sep=";", encoding="utf-8-sig")
        required = {"DATE", "BLOCK", "TREATMENT", "PLANT"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Unexpected DaKS schema in {path.name}")
        # 2019 has PLOT, farm-scale files have TRIPLETT, and the other
        # small-scale trials have neither. Preserve this source hierarchy
        # instead of inventing plot numbers from block/treatment.
        for optional in ("PLOT", "TRIPLETT"):
            if optional not in frame:
                frame[optional] = pd.NA
        frame["_row"] = np.arange(len(frame))
        stages = [name for name in STAGES if name in frame]
        long = frame.melt(id_vars=list(required) + ["PLOT", "TRIPLETT", "_row"], value_vars=stages, var_name="source_stage", value_name="value")
        long["source_id"] = source["source_id"]
        long["record_id"] = path.name + ":" + long["_row"].astype(str) + ":" + long.source_stage
        long["site_id"] = path.stem
        long["observation_date"] = pd.to_datetime(long.DATE, format="%Y-%m-%d", errors="raise")
        long["organism"] = "Leptinotarsa decemlineata"
        long["crop"] = "potato"
        long["measurement"] = "whole_plant_stage_count"
        long["value"] = pd.to_numeric(long.value, errors="raise")
        long["unit"] = "individuals_per_plant"
        long["stage"] = long.source_stage.map(STAGES)
        long["treatment"] = long.TREATMENT.astype("string")
        long["explicit_zero"] = long.value.notna() & long.value.eq(0)
        long["plot_id"] = long.PLOT.astype("string")
        long["triplet_id"] = long.TRIPLETT.astype("string")
        long["block"] = long.BLOCK.astype("string")
        long["plant_id"] = long.PLANT.astype("string")
        long["source_file"] = path.name
        observations.append(long[["source_id", "record_id", "site_id", "observation_date", "organism", "crop", "measurement", "value", "unit", "stage", "treatment", "explicit_zero", "plot_id", "triplet_id", "block", "plant_id", "source_file"]])
        latitude, longitude = field_coordinates[path.stem]
        locations.append({"source_id": source["source_id"], "site_id": path.stem, "latitude": latitude, "longitude": longitude, "coordinate_source": "public_DaKS_dataset_description_DMS", "coordinate_scope": "trial_field_not_individual_plant"})
    if not observations:
        raise ValueError("No DaKS observation CSV files downloaded")
    return _write_tables(output_dir, {"observations": pd.concat(observations, ignore_index=True), "locations": pd.DataFrame(locations)}, limitations=source["limitations"], endpoints=["post_first_egg_stage_progression", "larval_hatch_timing", "control_vs_mulch_population_dynamics"])


def _normalise_cpb(source: dict, raw_dir: Path, output_dir: Path) -> dict:
    frame = pd.read_csv(raw_dir / "files/WI_CPB_preformatted.csv")
    required = {"Date", "plot_ID", "adults", "larvae", "eggs"}
    if not required.issubset(frame):
        raise ValueError("Unexpected Wisconsin CPB schema")
    frame["_row"] = np.arange(len(frame))
    long = frame.melt(id_vars=["Date", "plot_ID", "_row"], value_vars=["adults", "larvae", "eggs"], var_name="stage", value_name="value")
    long["source_id"] = source["source_id"]
    long["record_id"] = long["_row"].astype(str) + ":" + long.stage
    long["site_id"] = long.plot_ID.astype(str)
    long["observation_date"] = pd.to_datetime(long.Date, format="%m/%d/%Y", errors="raise")
    long["organism"] = "Leptinotarsa decemlineata"
    long["crop"] = "potato"
    long["source_column"] = long.stage
    long["measurement"] = long.source_column.map({"adults": "sweep_net_mean_abundance", "larvae": "sweep_net_mean_abundance", "eggs": "visual_egg_mass_mean_abundance"})
    long["value"] = pd.to_numeric(long.value, errors="raise")
    long["unit"] = long.source_column.map({"adults": "mean_individuals_per_25_sweeps", "larvae": "mean_individuals_per_25_sweeps", "eggs": "egg_masses_per_plant"})
    long["stage"] = long.source_column.map({"adults": "adult", "larvae": "larva_unspecified", "eggs": "egg_mass"})
    long["treatment"] = "not_provided"
    long["explicit_zero"] = long.value.notna() & long.value.eq(0)
    long["source_row"] = long["_row"]
    columns = ["source_id", "record_id", "site_id", "observation_date", "organism", "crop", "measurement", "value", "unit", "stage", "treatment", "explicit_zero", "source_row", "source_column"]
    result = _write_tables(output_dir, {"observations": long[columns]}, limitations=source["limitations"], endpoints=["anonymous_plot_stage_population_dynamics", "biological_timing_only_if_compatible_weather_and_biofix_available"])
    result["original_rows"] = len(frame)
    result["excluded_predictor_columns"] = sorted(set(frame.columns) - required - {"_row", "Year", "Week", "Day"})
    return result


def _normalise_ris(source: dict, raw_dir: Path, output_dir: Path) -> dict:
    records, locations = [], []
    for name, organism in [("aphidtidy.csv", "Aphididae aggregate"), ("mothtidy.csv", "moths aggregate")]:
        frame = pd.read_csv(raw_dir / "files" / name)
        frame = frame.rename(columns={c: c.lower() for c in frame})
        if not {"year", "site", "count", "latitude", "longitude"}.issubset(frame):
            raise ValueError(f"Unexpected RIS annual schema: {name}")
        out = pd.DataFrame({
            "source_id": source["source_id"], "record_id": [f"{name}:{i}" for i in range(len(frame))],
            "site_id": frame.site.astype(str), "observation_date": pd.NaT,
            "observation_year": pd.to_numeric(frame.year, errors="raise").astype(int),
            "organism": organism, "crop": "not_crop_specific", "measurement": "annual_all_species_trap_sum",
            "value": pd.to_numeric(frame["count"], errors="raise"), "unit": "individuals_per_site_year",
            "stage": "flying_trapped_aggregate", "treatment": "not_provided", "source_file": name,
        })
        out["explicit_zero"] = out.value.notna() & out.value.eq(0)
        out["trap_operating"] = out.value.notna()
        records.append(out)
        loc = frame[["site", "latitude", "longitude"]].dropna().drop_duplicates().rename(columns={"site": "site_id"})
        loc["source_id"] = source["source_id"]
        loc["coordinate_source"] = name
        locations.append(loc)
    result = _write_tables(output_dir, {"observations": pd.concat(records, ignore_index=True), "locations": pd.concat(locations, ignore_index=True).drop_duplicates()}, limitations=source["limitations"], endpoints=["interannual_abundance_weather_association_only"])
    result["year_range"] = {"minimum": int(pd.concat(records).observation_year.min()), "maximum": int(pd.concat(records).observation_year.max())}
    return result


def _normalise_dd(source: dict, raw_dir: Path, output_dir: Path) -> dict:
    files = raw_dir / "files"
    weather = pd.read_csv(files / "data__espartoa-weather-2020.csv")
    weather["date"] = pd.to_datetime(weather.date, format="%Y-%m-%d", errors="raise")
    weather["source_id"] = source["source_id"]
    weather["temperature_unit"] = "degF"
    answers = pd.read_csv(files / "data__ucipm_results__ucipm_low50_high70_all.csv")
    answers["date"] = pd.to_datetime(answers.date, format="%m/%d/%Y", errors="raise")
    methods = [c for c in answers.columns if c not in {"date", "tmin", "tmax"}]
    long = answers.melt(id_vars=["date", "tmin", "tmax"], value_vars=methods, var_name="method", value_name="value")
    long["source_id"] = source["source_id"]
    long["lower_threshold_f"] = 50.
    long["upper_threshold_f"] = 70.
    long["unit"] = "degF_day"
    result = _write_tables(output_dir, {"weather": weather, "numerical_reference": long}, limitations=source["limitations"], endpoints=["degree_day_numerical_formula_equivalence"])
    result["numerical_reference_rows"] = len(long)
    result["weather_rows"] = len(weather)
    result["date_range"] = _date_range(weather.date)
    return result


def _normalise_kbs(source: dict, raw_dir: Path, output_dir: Path) -> dict:
    # The public table is ~149 MB and several million rows. Do not expand it
    # into an in-memory concatenation shared with all the other sources.
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "observations.parquet"
    required = {"sample_date", "state", "station", "species", "sex", "count"}
    count = 0
    earliest, latest = None, None
    organisms: dict[str, int] = {}
    writer = None
    schema = {}
    try:
        reader = pd.read_csv(raw_dir / "files/aphid_suction_2005_2013.csv", comment="#", chunksize=100_000, dtype={"count": "float64"})
        for frame in reader:
            if not required.issubset(frame):
                raise ValueError("Unexpected KBS aphid schema")
            out = pd.DataFrame({
                "source_id": source["source_id"], "record_id": np.arange(count, count + len(frame)).astype(str),
                "site_id": frame.state.astype("string") + ":" + frame.station.astype("string"),
                "observation_date": pd.to_datetime(frame.sample_date, format="%Y-%m-%d", errors="raise"),
                "organism": frame.species.astype("string").str.replace(".", " ", regex=False),
                "crop": "not_crop_specific", "measurement": "suction_trap_species_sex_count",
                "value": frame["count"].astype(float), "unit": "individuals_per_collection",
                "stage": "flying_trapped", "sex": frame.sex.astype("string"), "treatment": "not_provided",
            })
            out["explicit_zero"] = out.value.notna() & out.value.eq(0)
            arrow = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, arrow.schema, compression="zstd")
                schema = {name: str(dtype) for name, dtype in out.dtypes.items()}
            writer.write_table(arrow)
            count += len(out)
            lower, upper = out.observation_date.min(), out.observation_date.max()
            earliest = lower if earliest is None else min(earliest, lower)
            latest = upper if latest is None else max(latest, upper)
            for organism, rows in out.organism.value_counts().items():
                organisms[str(organism)] = organisms.get(str(organism), 0) + int(rows)
    finally:
        if writer is not None:
            writer.close()
    if not count:
        raise ValueError("KBS aphid download has no observation rows")
    return {
        "rows": count, "table_rows": {"observations": count}, "schema": schema,
        "date_range": {"minimum": earliest.date().isoformat(), "maximum": latest.date().isoformat()},
        "endpoints": ["species_specific_collection_timing", "trap_abundance_with_sampling_interval_audit"],
        "limitations": source["limitations"], "output_files": [path.name],
        "organism_rows": dict(sorted(organisms.items())),
        "normalization_mode": "chunked_100000_rows_no_implicit_daily_fill",
    }


def normalize(source: dict, raw_dir: Path, output_dir: Path) -> dict:
    """Normalize downloaded observations without changing outcome semantics."""
    functions = {
        "daks_cpb_2019_2022": _normalise_daks,
        "zenodo_18664442": _normalise_cpb,
        "ris_annual_1969_2016": _normalise_ris,
        "ucanr_degree_day_challenge": _normalise_dd,
        "kbs_aphid_suction_2005_2013": _normalise_kbs,
    }
    function = functions.get(source["source_id"])
    if function is None:
        return {"rows": 0, "output_files": [], "limitations": source.get("limitations", []), "endpoints": []}
    return function(source, Path(raw_dir), Path(output_dir))
