"""Observation loading, column mapping, normalization, and validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_STAGE_ALIASES, KNOWN_NON_TARGET_STAGES

CANONICAL_COLUMNS = (
    "observation_id",
    "latitude",
    "longitude",
    "observation_date",
    "pest_name",
    "observed_stage",
    "accumulation_start_date",
    "crop_name",
    "region",
    "source",
)
REQUIRED_COLUMNS = {"latitude", "longitude", "observation_date", "pest_name", "observed_stage"}


@dataclass(frozen=True)
class ObservationReport:
    """Accepted, excluded, and erroneous input rows."""

    accepted: pd.DataFrame
    excluded: pd.DataFrame
    errors: pd.DataFrame
    summary: pd.DataFrame


def load_observations(path: str | Path) -> pd.DataFrame:
    """Load a CSV or XLSX observation file."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix == ".xlsx":
        return pd.read_excel(source)
    raise ValueError("Observation file must be CSV or XLSX")


def apply_column_mapping(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Apply explicit source-to-canonical mapping and reject ambiguous collisions."""

    unknown_sources = set(mapping) - set(frame.columns)
    unknown_targets = set(mapping.values()) - set(CANONICAL_COLUMNS)
    if unknown_sources:
        raise ValueError(f"Mapping refers to missing source columns: {sorted(unknown_sources)}")
    if unknown_targets:
        raise ValueError(f"Mapping has unknown canonical columns: {sorted(unknown_targets)}")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Multiple source columns map to the same canonical column")
    renamed = frame.rename(columns=mapping).copy()
    missing = REQUIRED_COLUMNS - set(renamed.columns)
    if missing:
        raise ValueError(f"Missing required canonical columns: {sorted(missing)}")
    for column in CANONICAL_COLUMNS:
        if column not in renamed:
            renamed[column] = pd.NA
    return renamed[list(CANONICAL_COLUMNS)].copy()


def normalize_text(value: Any) -> str:
    """Normalize observation labels for editable exact-match dictionaries."""

    return " ".join(str(value).strip().lower().replace("ё", "е").split()) if pd.notna(value) else ""


def validate_observations(
    frame: pd.DataFrame,
    target_stage_code: str,
    global_accumulation_start_date: date | str | None = None,
    stage_aliases: dict[str, str] | None = None,
    target_pest_name: str | None = None,
) -> ObservationReport:
    """Validate rows and explicitly separate target, non-target, and erroneous records."""

    aliases = {normalize_text(k): v for k, v in (stage_aliases or DEFAULT_STAGE_ALIASES).items()}
    non_targets = {normalize_text(item) for item in KNOWN_NON_TARGET_STAGES}
    work = frame.copy().reset_index(drop=True)
    work["input_row_number"] = work.index + 2
    work["observation_id"] = work["observation_id"].fillna("").astype(str)
    empty_ids = work["observation_id"].str.strip().eq("")
    work.loc[empty_ids, "observation_id"] = work.loc[empty_ids, "input_row_number"].map(lambda n: f"row-{n}")
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    raw_start_dates = work["accumulation_start_date"].copy()
    start_date_supplied = raw_start_dates.map(lambda value: pd.notna(value) and bool(str(value).strip()))
    observation_dates = pd.to_datetime(work["observation_date"], errors="coerce")
    start_dates = pd.to_datetime(raw_start_dates, errors="coerce")
    # Explicit object dtype keeps Python dates assignable even when a whole input
    # column is empty (pandas 3 otherwise retains datetime64 for an all-NaT column).
    work["observation_date"] = pd.Series(
        [value.date() if pd.notna(value) else pd.NaT for value in observation_dates], dtype=object
    )
    work["accumulation_start_date"] = pd.Series(
        [value.date() if pd.notna(value) else pd.NaT for value in start_dates], dtype=object
    )
    work["invalid_accumulation_start_date"] = start_date_supplied & work["accumulation_start_date"].isna()
    global_date = pd.to_datetime(global_accumulation_start_date).date() if global_accumulation_start_date else None
    work["accumulation_start_rule"] = "missing"
    work.loc[start_date_supplied, "accumulation_start_rule"] = "observation_record"
    work.loc[work["invalid_accumulation_start_date"], "accumulation_start_rule"] = "invalid"
    if global_date is not None:
        missing_start = ~start_date_supplied
        work.loc[missing_start, "accumulation_start_date"] = global_date
        work.loc[missing_start, "accumulation_start_rule"] = "manual_global_date"

    work["normalized_stage"] = work["observed_stage"].map(normalize_text)
    work["stage_code"] = work["normalized_stage"].map(aliases)
    work["duplicate_record"] = work.duplicated(
        subset=["latitude", "longitude", "observation_date", "pest_name", "observed_stage"], keep=False
    )

    error_reasons: list[str] = []
    exclusion_reasons: list[str] = []
    statuses: list[str] = []
    for row in work.itertuples(index=False):
        errors: list[str] = []
        excluded: list[str] = []
        if pd.isna(row.latitude) or not -90 <= row.latitude <= 90:
            errors.append("invalid_latitude")
        if pd.isna(row.longitude) or not -180 <= row.longitude <= 180:
            errors.append("invalid_longitude")
        if pd.isna(row.observation_date):
            errors.append("invalid_observation_date")
        if not normalize_text(row.observed_stage):
            errors.append("missing_observed_stage")
        if not normalize_text(row.pest_name):
            errors.append("missing_pest_name")
        elif target_pest_name and normalize_text(row.pest_name) != normalize_text(target_pest_name):
            excluded.append("different_pest")
        if row.invalid_accumulation_start_date:
            errors.append("invalid_accumulation_start_date")
        if pd.notna(row.accumulation_start_date) and pd.notna(row.observation_date):
            if row.accumulation_start_date > row.observation_date:
                errors.append("accumulation_start_after_observation")
        if pd.isna(row.accumulation_start_date) and not row.invalid_accumulation_start_date:
            excluded.append("missing_accumulation_start_date")
        if row.stage_code != target_stage_code:
            if row.normalized_stage in non_targets:
                excluded.append("known_non_target_stage")
            elif not row.stage_code:
                excluded.append("unmapped_or_ambiguous_stage")
            else:
                excluded.append("different_canonical_stage")
        if errors:
            statuses.append("error")
        elif excluded:
            statuses.append("excluded")
        else:
            statuses.append("accepted")
        error_reasons.append(";".join(errors))
        exclusion_reasons.append(";".join(excluded))
    work["record_status"] = statuses
    work["error_reason"] = error_reasons
    work["exclusion_reason"] = exclusion_reasons

    summary = (
        work.groupby("record_status", dropna=False).size().rename("record_count").reset_index()
    )
    return ObservationReport(
        accepted=work[work["record_status"] == "accepted"].copy(),
        excluded=work[work["record_status"] == "excluded"].copy(),
        errors=work[work["record_status"] == "error"].copy(),
        summary=summary,
    )
