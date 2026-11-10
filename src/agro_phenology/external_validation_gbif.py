"""Public Russian GBIF occurrences as an auxiliary presence-only collection.

Search pages are archived, not represented as a DOI-stable complete download.
No private project locations or observations are sent to the service.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from urllib.parse import urlencode

import pandas as pd


TAXA = [
    "Leptinotarsa decemlineata", "Cydia pomonella", "Chamaepsila rosae",
    "Delia radicum", "Plutella xylostella", "Tetranychus urticae",
    "Phytophthora infestans", "Venturia inaequalis", "Venturia pyrina",
    "Peronospora destructor", "Botrytis cinerea", "Alternaria solani",
    "Pseudoperonospora cubensis", "Podosphaera xanthii", "Podosphaera mors-uvae",
    "Myzus persicae", "Brevicoryne brassicae", "Macrosiphum euphorbiae",
]
SOURCE_ID = "gbif_ru_garden_occurrences"


def discover(client, *, max_per_taxon: int = 10000) -> list[dict]:
    files, inventory, errors = [], [], []
    for scientific_name in TAXA:
        slug = scientific_name.lower().replace(" ", "_")
        print(f"  GBIF: {scientific_name}", flush=True)
        try:
            match_url = "https://api.gbif.org/v1/species/match?" + urlencode({"name": scientific_name, "strict": "true"})
            match = client.get_json(match_url, f"raw/{SOURCE_ID}/taxonomy/{slug}.json")
            if match.get("matchType") != "EXACT":
                errors.append({"taxon": scientific_name, "reason": "exact_taxonomy_match_unavailable", "matchType": match.get("matchType")})
                continue
            taxon_key = match.get("acceptedUsageKey") or match["usageKey"]
            count, retrieved = None, 0
            for offset in range(0, max_per_taxon, 300):
                url = "https://api.gbif.org/v1/occurrence/search?" + urlencode({"taxonKey": taxon_key, "country": "RU", "year": "2000,2026", "limit": min(300, max_per_taxon - offset), "offset": offset})
                name = f"{slug}__{offset:06d}.json"
                data = client.get_json(url, f"raw/{SOURCE_ID}/files/{name}")
                files.append({"name": name, "url": url, "role": "occurrence_search_page", "max_bytes": 8_000_000})
                count = data.get("count", count)
                retrieved += len(data.get("results", []))
                if data.get("endOfRecords") or not data.get("results"):
                    break
            inventory.append({"requested_taxon": scientific_name, "accepted_key": taxon_key, "reported_count": count,
                              "downloaded_records_before_deduplication": retrieved, "limit": max_per_taxon,
                              "truncated_by_limit": count is not None and count > retrieved})
        except OSError as error:
            errors.append({"taxon": scientific_name, "reason": str(error)})
    return [{"source_id": SOURCE_ID, "provider": "gbif", "title": "GBIF: garden pests and pathogens, Russia, 2000–2026",
             "landing_url": "https://techdocs.gbif.org/en/openapi/v1/occurrence", "access_status": "open" if files else "unavailable",
             "license": {"id": "per_record", "url": "https://www.gbif.org/terms", "notes": "Retain individual record licenses and originating dataset keys; no photographs downloaded."},
             "evidence_kind": "auxiliary_occurrences_not_systematic_surveillance", "crops": [], "organisms": TAXA,
             "limitations": ["Country RU and years 2000–2026; search pages are not an exhaustive DOI-stable GBIF download.",
                             "Absence of a record is not absence of an organism. Sampling effort, host, stage and coordinate privacy vary.",
                             "Not an independent dataset from its publishers: iNaturalist/other provider records must not be counted twice.",
                             "Partial or interval event dates remain unresolved, never replaced with January 1."],
             "query_inventory": inventory, "discovery_errors": errors, "files": files}]


def exact_event_date(record: dict):
    """Accept an actual day, but never fill missing month/day components."""
    if record.get("year") and record.get("month") and record.get("day"):
        try:
            return pd.Timestamp(year=int(record["year"]), month=int(record["month"]), day=int(record["day"]))
        except ValueError:
            return pd.NaT
    raw = record.get("eventDate") or ""
    if re.match(r"^\d{4}-\d{2}-\d{2}(?:T.*)?$", raw):
        return pd.to_datetime(raw[:10], errors="coerce")
    return pd.NaT


def normalize(source: dict, raw_dir: Path, output_dir: Path) -> dict:
    records = {}
    before = 0
    for spec in source["files"]:
        path = raw_dir / "files" / spec["name"]
        if not path.is_file():
            continue
        page = json.loads(path.read_text())
        for record in page.get("results", []):
            before += 1
            records[str(record["key"])] = record
    rows = []
    for key, record in records.items():
        status = record.get("occurrenceStatus")
        rows.append({"source_id": source["source_id"], "record_id": key, "site_id": None,
                     "observation_date": exact_event_date(record), "event_date_raw": record.get("eventDate"),
                     "event_year": record.get("year"), "event_month": record.get("month"), "event_day": record.get("day"),
                     "organism": record.get("species") or record.get("scientificName"), "crop": None,
                     "measurement": "occurrence_status", "value": 1.0 if status == "PRESENT" else 0.0 if status == "ABSENT" else None,
                     "unit": "presence_record", "stage": record.get("lifeStage"), "treatment": None,
                     "explicit_zero": status == "ABSENT", "occurrence_status": status,
                     "individual_count": record.get("individualCount"), "basis_of_record": record.get("basisOfRecord"),
                     "dataset_key": record.get("datasetKey"), "occurrence_id": record.get("occurrenceID"),
                     "recorded_by": record.get("recordedBy"), "country": record.get("countryCode"),
                     "latitude": record.get("decimalLatitude"), "longitude": record.get("decimalLongitude"),
                     "coordinate_uncertainty_m": record.get("coordinateUncertaintyInMeters"),
                     "license": record.get("license"), "references": record.get("references"),
                     "gbif_issues": json.dumps(record.get("issues", []))})
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output_dir / "occurrences.parquet", index=False)
    return {"rows": len(frame), "duplicates_removed": before - len(frame), "schema": {c: str(t) for c, t in frame.dtypes.items()},
            "exact_date_rows": int(frame.observation_date.notna().sum()) if len(frame) else 0,
            "date_range": [str(frame.observation_date.min()), str(frame.observation_date.max())] if len(frame) else [],
            "endpoints": ["auxiliary_presence_and_phenology_only"], "output_files": ["occurrences.parquet"],
            "limitations": source["limitations"], "query_inventory": source["query_inventory"], "discovery_errors": source["discovery_errors"]}
