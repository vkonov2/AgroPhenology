"""Conservative-v2 labels for apple codling moth and apple scab.

The VAAD source stores several observations for one organism in a JSON list.
For codling moth, larval field observations and adult trap counts answer two
different questions and must never be collapsed into one target.  The primary
early-warning outcome in this module is a positive larval field record
(``Kāpurs`` with ``Izplatība > 0``); adult ``Imago`` pheromone-trap counts are retained as
a separate, causal diagnostic/biofix input.

An organism omitted from a visit is unassessed.  A generic statement that no
harmful organisms were found is retained as ``generic_absent`` and is not
upgraded to a target-specific zero.  When duplicate source rows disagree, the
visit status follows the predeclared order positive > explicit target absence
> generic absence > unassessed, while a separate conflict flag preserves the
disagreement for audit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


APPLE_CROP_CODE = 41
CODLING_MOTH_ID = 1341
APPLE_SCAB_ID = 1213
SUPPORTED_TARGETS = frozenset({CODLING_MOTH_ID, APPLE_SCAB_ID})

LABEL_PRIORITY = {
    "unassessed": 0,
    "generic_absent": 1,
    "explicit_target_absent": 2,
    "positive": 3,
}

_GENERIC_ABSENCE_RE = re.compile(
    r"(?:kaitīgie\s+organismi|kaitīgo\s+organismu\s+klātbūtne|slimības)"
    r"\s+(?:nav|netika)\s+konstat",
    re.IGNORECASE,
)
_PREVALENCE_RE = re.compile(
    r"Izplatība:\s*([0-9]+(?:[.,][0-9]+)?)\s*%",
    re.IGNORECASE,
)
_LARVA_RE = re.compile(r"(?:^|,\s*)Kāpurs(?:\s*,|$)", re.IGNORECASE)
_ADULT_RE = re.compile(r"(?:^|,\s*)Imago(?:\s*,|$)", re.IGNORECASE)
_ADULT_TRAP_RE = re.compile(
    r"Invāzijas\s+pakāpe:\s*([0-9]+(?:[.,][0-9]+)?)"
    r"(?:\s*\([^)]*\))?\s*"
    r"(feromonu\s+slazdā)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TargetLabel:
    """One field-observation label with explicit provenance."""

    status: str
    listed_in_source: bool
    value: float = math.nan
    source_conflict: bool = False


@dataclass(frozen=True)
class CodlingMothLabels:
    """Independent primary-larva and adult-trap labels for one source row."""

    primary: TargetLabel
    adult_trap: TargetLabel
    organism_listed_in_source: bool
    adult_trap_instruments: tuple[str, ...] = ()


def truth(value: Any) -> bool:
    """Parse conservative-v2 truth-like values without treating NaN as true."""

    return str(value).strip().lower() in {"true", "1", "yes"}


def generic_absence(explicit_value: Any, organisms_raw: Any) -> bool:
    """Return the source-level generic absence flag.

    This flag is deliberately kept separate from a target-specific zero.
    """

    raw_text = "" if pd.isna(organisms_raw) else str(organisms_raw)
    return truth(explicit_value) or bool(_GENERIC_ABSENCE_RE.search(raw_text))


def parse_detected_organisms(value: Any) -> tuple[list[dict[str, Any]], bool]:
    """Parse a conservative-v2 organism list and report JSON validity.

    A missing or malformed value returns an empty list with ``valid=False``;
    callers can therefore keep the row unassessed without silently pretending
    that the target was omitted from a valid source record.
    """

    if isinstance(value, list):
        parsed = value
    else:
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return [], False
    if not isinstance(parsed, list) or any(not isinstance(item, dict) for item in parsed):
        return [], False
    return parsed, True


def _organism_id(entry: Mapping[str, Any]) -> int | None:
    value = entry.get("organism_id")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not numeric.is_integer():
        return None
    return int(numeric)


def _numbers(pattern: re.Pattern[str], text: str) -> list[float]:
    return [float(match.replace(",", ".")) for match in pattern.findall(text)]


def _label_from_values(
    values: Sequence[float],
    *,
    listed: bool,
    generic_absent: bool,
) -> TargetLabel:
    positive = any(value > 0 for value in values)
    zero = bool(values) and any(value == 0 for value in values)
    conflict = (positive and zero) or (positive and generic_absent)
    if positive:
        status = "positive"
    elif zero:
        status = "explicit_target_absent"
    elif generic_absent:
        status = "generic_absent"
    else:
        status = "unassessed"
    return TargetLabel(
        status=status,
        listed_in_source=listed,
        value=max(values) if values else math.nan,
        source_conflict=conflict,
    )


def label_codling_moth(
    organisms: Sequence[Mapping[str, Any]],
    *,
    generic_absent_flag: bool = False,
) -> CodlingMothLabels:
    """Label codling-moth larvae and adult traps independently.

    Only ``Kāpurs`` entries with an explicit ``Izplatība`` value contribute to
    the primary outcome.  Only ``Imago`` entries with a count tied to a
    pheromone trap or glue shield contribute to the adult-trap diagnostic.
    Thus ``Imago, ... 0 feromonu slazdā`` is an adult-trap zero and never a
    positive larval event.
    """

    entries = [entry for entry in organisms if _organism_id(entry) == CODLING_MOTH_ID]
    larval_entries: list[Mapping[str, Any]] = []
    adult_entries: list[Mapping[str, Any]] = []
    larval_values: list[float] = []
    adult_values: list[float] = []
    instruments: set[str] = set()

    for entry in entries:
        detail = str(entry.get("details_raw", ""))
        if _LARVA_RE.search(detail):
            larval_entries.append(entry)
            larval_values.extend(_numbers(_PREVALENCE_RE, detail))
        if _ADULT_RE.search(detail):
            trap_matches = _ADULT_TRAP_RE.findall(detail)
            if trap_matches:
                adult_entries.append(entry)
                for count, instrument in trap_matches:
                    adult_values.append(float(count.replace(",", ".")))
                    instruments.add(" ".join(instrument.lower().split()))

    primary = _label_from_values(
        larval_values,
        listed=bool(larval_entries),
        generic_absent=generic_absent_flag,
    )
    # A general scouting absence does not prove that an adult trap was checked.
    adult_trap = _label_from_values(
        adult_values,
        listed=bool(adult_entries),
        generic_absent=False,
    )
    return CodlingMothLabels(
        primary=primary,
        adult_trap=adult_trap,
        organism_listed_in_source=bool(entries),
        adult_trap_instruments=tuple(sorted(instruments)),
    )


def label_apple_scab(
    organisms: Sequence[Mapping[str, Any]],
    *,
    generic_absent_flag: bool = False,
) -> TargetLabel:
    """Label apple scab using explicit target prevalence only."""

    entries = [entry for entry in organisms if _organism_id(entry) == APPLE_SCAB_ID]
    values: list[float] = []
    for entry in entries:
        values.extend(_numbers(_PREVALENCE_RE, str(entry.get("details_raw", ""))))
    return _label_from_values(values, listed=bool(entries), generic_absent=generic_absent_flag)


def label_apple_rows(frame: pd.DataFrame, target_id: int) -> pd.DataFrame:
    """Filter apple rows and attach target-specific conservative-v2 labels."""

    if target_id not in SUPPORTED_TARGETS:
        raise ValueError(f"Unsupported apple target: {target_id}")
    required = {"crop_code", "detected_organisms"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    crop_codes = pd.to_numeric(frame["crop_code"], errors="coerce")
    work = frame.loc[crop_codes.eq(APPLE_CROP_CODE)].copy()
    parsed = work["detected_organisms"].map(parse_detected_organisms)
    work["organism_json_valid"] = parsed.map(lambda item: item[1])
    work["generic_absent_flag"] = [
        generic_absence(explicit, raw)
        for explicit, raw in zip(
            work.get("explicit_no_harmful_organisms", pd.Series(False, index=work.index)),
            work.get("organisms_raw", pd.Series("", index=work.index)),
        )
    ]

    if target_id == CODLING_MOTH_ID:
        labels = [
            label_codling_moth(items, generic_absent_flag=bool(generic))
            if valid
            else CodlingMothLabels(
                primary=TargetLabel("generic_absent" if generic else "unassessed", False),
                adult_trap=TargetLabel("unassessed", False),
                organism_listed_in_source=False,
            )
            for (items, valid), generic in zip(parsed, work["generic_absent_flag"])
        ]
        work["label_status"] = [label.primary.status for label in labels]
        work["listed_in_source"] = [label.primary.listed_in_source for label in labels]
        work["organism_listed_in_source"] = [label.organism_listed_in_source for label in labels]
        work["prevalence_pct"] = [label.primary.value for label in labels]
        work["source_conflict"] = [label.primary.source_conflict for label in labels]
        work["adult_trap_status"] = [label.adult_trap.status for label in labels]
        work["adult_trap_listed_in_source"] = [label.adult_trap.listed_in_source for label in labels]
        work["adult_trap_count"] = [label.adult_trap.value for label in labels]
        work["adult_trap_source_conflict"] = [label.adult_trap.source_conflict for label in labels]
        work["adult_trap_instruments"] = [";".join(label.adult_trap_instruments) for label in labels]
    else:
        labels = [
            label_apple_scab(items, generic_absent_flag=bool(generic))
            if valid
            else TargetLabel("generic_absent" if generic else "unassessed", False)
            for (items, valid), generic in zip(parsed, work["generic_absent_flag"])
        ]
        work["label_status"] = [label.status for label in labels]
        work["listed_in_source"] = [label.listed_in_source for label in labels]
        work["organism_listed_in_source"] = work["listed_in_source"]
        work["prevalence_pct"] = [label.value for label in labels]
        work["source_conflict"] = [label.source_conflict for label in labels]
    return work


def _highest_priority_status(values: Iterable[Any]) -> str:
    statuses = [str(value) for value in values]
    unknown = sorted(set(statuses) - set(LABEL_PRIORITY))
    if unknown:
        raise ValueError(f"Unknown label statuses: {unknown}")
    return max(statuses, key=LABEL_PRIORITY.__getitem__) if statuses else "unassessed"


def _max_or_nan(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else math.nan


def deduplicate_apple_visits(labeled_rows: pd.DataFrame, target_id: int) -> pd.DataFrame:
    """Collapse labeled source rows to one apple field-date visit.

    Known fields are grouped by ``final_field_uid`` and date.  Rows without a
    field id are kept separate using ``observation_id`` so unrelated records do
    not collapse into one artificial visit.  Codling-moth adult-trap fields are
    aggregated independently and cannot change the primary larval status.
    """

    if target_id not in SUPPORTED_TARGETS:
        raise ValueError(f"Unsupported apple target: {target_id}")
    required = {"final_field_uid", "observation_date", "label_status"}
    if target_id == CODLING_MOTH_ID:
        required.add("adult_trap_status")
    missing = required - set(labeled_rows.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = labeled_rows.copy().reset_index(drop=True)
    work["observation_date"] = pd.to_datetime(work["observation_date"], errors="coerce")
    if work["observation_date"].isna().any():
        raise ValueError("Cannot deduplicate rows with invalid observation_date")
    if "crop_code" in work:
        crop_codes = pd.to_numeric(work["crop_code"], errors="coerce")
        if not crop_codes.eq(APPLE_CROP_CODE).all():
            raise ValueError("deduplicate_apple_visits accepts crop_code=41 rows only")

    observation_ids = work.get("observation_id", pd.Series(work.index.astype(str), index=work.index)).astype(str)
    work["_field_key"] = work["final_field_uid"].astype("string")
    missing_field = work["_field_key"].isna() | work["_field_key"].str.strip().eq("")
    work.loc[missing_field, "_field_key"] = "observation:" + observation_ids.loc[missing_field]

    visits: list[dict[str, Any]] = []
    for (field_key, day), group in work.groupby(["_field_key", "observation_date"], sort=False, dropna=False):
        primary_status = _highest_priority_status(group["label_status"])
        primary_statuses = set(group["label_status"].astype(str))
        primary_conflict = (
            "positive" in primary_statuses
            and bool(primary_statuses.intersection({"explicit_target_absent", "generic_absent"}))
        ) or bool(
            group.get("source_conflict", pd.Series(False, index=group.index)).fillna(False).astype(bool).any()
        )
        winner = group.loc[group["label_status"].map(LABEL_PRIORITY).idxmax()].to_dict()
        winner.update(
            {
                "label_status": primary_status,
                "listed_in_source": bool(
                    group.get("listed_in_source", pd.Series(False, index=group.index)).fillna(False).astype(bool).any()
                ),
                "organism_listed_in_source": bool(
                    group.get("organism_listed_in_source", pd.Series(False, index=group.index))
                    .fillna(False)
                    .astype(bool)
                    .any()
                ),
                "prevalence_pct": _max_or_nan(
                    group.get("prevalence_pct", pd.Series(math.nan, index=group.index))
                ),
                "source_conflict": primary_conflict,
                "source_row_count": int(len(group)),
                "visit_id": hashlib.sha256(f"{field_key}|{day.date()}|{target_id}".encode()).hexdigest()[:20],
            }
        )

        if target_id == CODLING_MOTH_ID:
            adult_status = _highest_priority_status(group["adult_trap_status"])
            adult_statuses = set(group["adult_trap_status"].astype(str))
            adult_conflict = (
                "positive" in adult_statuses
                and "explicit_target_absent" in adult_statuses
            ) or bool(
                group.get("adult_trap_source_conflict", pd.Series(False, index=group.index))
                .fillna(False)
                .astype(bool)
                .any()
            )
            instruments: set[str] = set()
            for value in group.get("adult_trap_instruments", pd.Series("", index=group.index)).fillna(""):
                instruments.update(item for item in str(value).split(";") if item)
            winner.update(
                {
                    "adult_trap_status": adult_status,
                    "adult_trap_listed_in_source": bool(
                        group.get("adult_trap_listed_in_source", pd.Series(False, index=group.index))
                        .fillna(False)
                        .astype(bool)
                        .any()
                    ),
                    "adult_trap_count": _max_or_nan(
                        group.get("adult_trap_count", pd.Series(math.nan, index=group.index))
                    ),
                    "adult_trap_source_conflict": adult_conflict,
                    "adult_trap_instruments": ";".join(sorted(instruments)),
                }
            )
        winner.pop("_field_key", None)
        visits.append(winner)

    if not visits:
        return work.drop(columns="_field_key").iloc[0:0].copy()
    result = pd.DataFrame(visits)
    return result.sort_values(["observation_date", "visit_id"]).reset_index(drop=True)


def prepare_apple_target_visits(frame: pd.DataFrame, target_id: int) -> pd.DataFrame:
    """Convenience wrapper: filter, label, and deduplicate conservative-v2 rows."""

    return deduplicate_apple_visits(label_apple_rows(frame, target_id), target_id)
