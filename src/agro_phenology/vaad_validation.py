"""Retrospective field-season validation of late-blight weather indicators on VAAD data.

The module deliberately separates event-timing evidence from claims about
specificity.  A VAAD row without late blight in the structured organism list is
not automatically a confirmed clean inspection, and crop protection treatments
are unavailable.  Negative visits are therefore used only in paired and
sensitivity analyses, with explicit provenance in the outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import requests

from .late_blight import (
    PolyakovConfig,
    aggregate_polyakov_daily,
    classify_hutton_days,
    classify_hutton_periods,
    classify_polyakov_windows,
    extract_polyakov_episodes,
    merge_polyakov_weather_sources,
)
from .open_meteo import ARCHIVE_URL, OpenMeteoError, parse_archive_response


DIRECT_SOURCE = "lauksuid_direct"
RECOVERED_SOURCE = "daily_date_crop_unique"
ALLOWED_GEOLOCATION_SOURCES = {DIRECT_SOURCE, RECOVERED_SOURCE}
POTATO_CROP_CODE = 166
LATE_BLIGHT_ORGANISM_ID = 640
DEFAULT_LOOKBACKS = (7, 14, 21)
DEFAULT_SEED = 20260902


def _bool_value(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _late_blight_has_zero_prevalence(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        organisms = json.loads(value)
    except json.JSONDecodeError:
        return False
    if not isinstance(organisms, list):
        return False
    for organism in organisms:
        if not isinstance(organism, dict):
            continue
        is_late_blight = (
            organism.get("organism_id") == LATE_BLIGHT_ORGANISM_ID
            or organism.get("name") == "Kartupeļu lakstu puve"
        )
        if not is_late_blight:
            continue
        match = re.search(
            r"Izplatība:\s*([0-9]+(?:[.,][0-9]+)?)\s*%",
            str(organism.get("details_raw", "")),
            flags=re.IGNORECASE,
        )
        if match and float(match.group(1).replace(",", ".")) == 0.0:
            return True
    return False


def _generic_explicit_absence_text(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return bool(
        re.search(
            r"(?:kaitīgie\s+organismi|kaitīgo\s+organismu\s+klātbūtne|slimības)"
            r"\s+(?:nav|netika)\s+konstat",
            value,
            flags=re.IGNORECASE,
        )
    )


def _target_explicit_absence_text(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return bool(
        re.search(
            r"(?:lakstu\s+puves\s+pazīmes|(?:kartupeļu\s+)?lakstu\s+puve)"
            r"\s+(?:nav|netika)(?:\s+konstat\w*)?",
            value,
            flags=re.IGNORECASE,
        )
    )


def _late_blight_positive_text(value: object) -> bool:
    """Recognize only unambiguous free-text late-blight detections."""

    if not isinstance(value, str):
        return False
    return bool(
        re.search(
            r"(?:100\s*%\s*lakstu\s+puve|lakstu\s+puves\s+izplat|"
            r"lakstu\s+puve\s+izplat)",
            value,
            flags=re.IGNORECASE,
        )
    )


def _source_label(values: Iterable[object]) -> str:
    sources = {str(value) for value in values if pd.notna(value)}
    if sources == {DIRECT_SOURCE}:
        return "direct"
    if sources == {RECOVERED_SOURCE}:
        return "recovered"
    return "mixed"


def prepare_potato_visits(
    csv_path: str | Path,
    source_mode: str = "expanded",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load, filter, and deduplicate VAAD potato observations by field and date.

    ``direct`` uses only original ``lauksuid`` links. ``expanded`` additionally
    admits the conservative one-to-one date/crop recovery stratum.
    """

    if source_mode not in {"direct", "expanded"}:
        raise ValueError("source_mode must be 'direct' or 'expanded'")
    path = Path(csv_path)
    raw = pd.read_csv(path, low_memory=False)
    required = {
        "observation_id",
        "observation_date",
        "crop_code",
        "crop_name",
        "growth_stage_code",
        "growth_stage",
        "detected_organisms",
        "late_blight_detected",
        "explicit_no_harmful_organisms",
        "field_uid",
        "latitude",
        "longitude",
        "geolocation_source",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"VAAD file is missing columns: {sorted(missing)}")

    raw["observation_date"] = pd.to_datetime(raw["observation_date"], errors="coerce")
    raw["crop_code"] = pd.to_numeric(raw["crop_code"], errors="coerce")
    raw["growth_stage_code"] = pd.to_numeric(raw["growth_stage_code"], errors="coerce")
    raw["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
    raw["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")

    potato = raw[
        raw["crop_code"].eq(POTATO_CROP_CODE)
        | raw["crop_name"].astype("string").str.startswith("Kartupeļi", na=False)
    ].copy()
    initial_potato_rows = len(potato)

    permitted_sources = {DIRECT_SOURCE} if source_mode == "direct" else ALLOWED_GEOLOCATION_SOURCES
    valid_geo = (
        potato["geolocation_source"].isin(permitted_sources)
        & potato["field_uid"].notna()
        & potato["latitude"].between(-90, 90, inclusive="both")
        & potato["longitude"].between(-180, 180, inclusive="both")
    )
    excluded_missing_geo = int((~valid_geo).sum())
    potato = potato[valid_geo].copy()

    in_season = potato["observation_date"].dt.month.between(5, 9)
    excluded_off_season = int((~in_season).sum())
    potato = potato[in_season].copy()

    stage_99 = potato["growth_stage_code"].eq(99)
    excluded_stage_99 = int(stage_99.sum())
    potato = potato[~stage_99].copy()

    text = (
        potato.get("crop_stage_raw", pd.Series("", index=potato.index)).fillna("").astype(str)
        + " "
        + potato.get("organisms_raw", pd.Series("", index=potato.index)).fillna("").astype(str)
    )
    storage = text.str.contains(
        r"noliktav|pēc\s+ražas\s+novākšanas|uzglabāšanas\s+gatavība",
        case=False,
        regex=True,
    )
    excluded_storage = int(storage.sum())
    potato = potato[~storage].copy()

    raw_organism_text = potato.get(
        "organisms_raw", pd.Series("", index=potato.index)
    ).fillna("")
    potato["_generic_explicit_absence_text"] = raw_organism_text.map(
        _generic_explicit_absence_text
    )
    potato["_target_explicit_absence_text"] = raw_organism_text.map(
        _target_explicit_absence_text
    )
    potato["_late_blight_positive_text"] = raw_organism_text.map(
        _late_blight_positive_text
    )
    # The source occasionally lists late blight with an explicit prevalence of
    # 0 %.  That is evidence of target-specific absence, not a positive case and
    # not a row to discard.  Preserve its provenance for sensitivity analyses.
    potato["_late_blight_zero_prevalence"] = potato["detected_organisms"].map(
        _late_blight_has_zero_prevalence
    )
    reclassified_zero_prevalence = int(
        potato["_late_blight_zero_prevalence"].sum()
    )

    rows: list[dict[str, object]] = []
    for (field_uid, observation_date), group in potato.groupby(
        ["field_uid", "observation_date"], sort=True, dropna=False
    ):
        coordinates = group[["latitude", "longitude"]].drop_duplicates()
        if len(coordinates) != 1:
            raise ValueError(
                f"Field/date {field_uid}/{observation_date.date()} has inconsistent coordinates"
            )
        outcomes = [
            True
            if bool(text_positive)
            else False
            if bool(zero_prevalence)
            else _bool_value(value)
            for value, zero_prevalence, text_positive in zip(
                group["late_blight_detected"],
                group["_late_blight_zero_prevalence"],
                group["_late_blight_positive_text"],
                strict=True,
            )
        ]
        if True in outcomes:
            outcome: bool | None = True
        elif False in outcomes:
            outcome = False
        else:
            outcome = None
        stages = tuple(
            sorted(set(group["growth_stage_code"].dropna().astype(int).tolist()))
        )
        rows.append(
            {
                "field_uid": str(field_uid),
                "observation_date": pd.Timestamp(observation_date).date(),
                "season": int(pd.Timestamp(observation_date).year),
                "latitude": float(coordinates.iloc[0]["latitude"]),
                "longitude": float(coordinates.iloc[0]["longitude"]),
                "late_blight_detected": outcome,
                "explicit_no_harmful_organisms": any(
                    _bool_value(value) is True
                    for value in group["explicit_no_harmful_organisms"]
                ),
                "explicit_late_blight_absent": bool(
                    outcome is False
                    and (
                        group["_late_blight_zero_prevalence"].any()
                        or group["_generic_explicit_absence_text"].any()
                        or group["_target_explicit_absence_text"].any()
                        or any(
                            _bool_value(value) is True
                            for value in group["explicit_no_harmful_organisms"]
                        )
                    )
                ),
                "zero_prevalence_late_blight": bool(
                    group["_late_blight_zero_prevalence"].any()
                ),
                "generic_explicit_absence_from_text": bool(
                    group["_generic_explicit_absence_text"].any()
                ),
                "target_explicit_absence_from_text": bool(
                    group["_target_explicit_absence_text"].any()
                ),
                "positive_late_blight_from_text": bool(
                    group["_late_blight_positive_text"].any()
                ),
                "growth_stage_codes": stages,
                "growth_stage_labels": " | ".join(
                    sorted(set(group["growth_stage"].dropna().astype(str)))
                ),
                "geolocation_stratum": _source_label(group["geolocation_source"]),
                "municipality": next(
                    (str(value) for value in group.get("municipality", []) if pd.notna(value)),
                    "",
                ),
                "parish": next(
                    (str(value) for value in group.get("parish", []) if pd.notna(value)),
                    "",
                ),
                "row_count": int(len(group)),
                "observation_ids": "|".join(group["observation_id"].astype(str)),
            }
        )

    visits = pd.DataFrame(rows).sort_values(
        ["field_uid", "season", "observation_date"]
    ).reset_index(drop=True)
    qc = {
        "input_rows": int(len(raw)),
        "potato_rows": int(initial_potato_rows),
        "excluded_missing_or_unapproved_geolocation": excluded_missing_geo,
        "excluded_off_season": excluded_off_season,
        "excluded_stage_99": excluded_stage_99,
        "excluded_storage_text": excluded_storage,
        "excluded_zero_prevalence_positive": 0,
        "reclassified_zero_prevalence_as_explicit_target_negative": (
            reclassified_zero_prevalence
        ),
        "generic_explicit_absence_text_rows": int(
            potato["_generic_explicit_absence_text"].sum()
        ),
        "target_explicit_absence_text_rows": int(
            potato["_target_explicit_absence_text"].sum()
        ),
        "reclassified_positive_free_text_rows": int(
            (
                potato["_late_blight_positive_text"]
                & ~potato["late_blight_detected"].map(
                    lambda value: _bool_value(value) is True
                )
            ).sum()
        ),
        "deduplicated_visit_rows": int(visits["row_count"].sum() - len(visits)),
        "analysis_visits": int(len(visits)),
        "analysis_field_seasons": int(
            visits[["field_uid", "season"]].drop_duplicates().shape[0]
        ),
    }
    return visits, qc


def build_onset_events(visits: pd.DataFrame) -> pd.DataFrame:
    """Create interval-censored first-detection events ``(last N, first P]``."""

    rows: list[dict[str, object]] = []
    for (field_uid, season), group in visits.groupby(["field_uid", "season"], sort=True):
        group = group.sort_values("observation_date")
        positives = group[group["late_blight_detected"].eq(True)]
        if positives.empty:
            continue
        first_positive = positives.iloc[0]
        prior_negatives = group[
            (group["observation_date"] < first_positive["observation_date"])
            & group["late_blight_detected"].eq(False)
        ]
        if prior_negatives.empty:
            continue
        last_negative = prior_negatives.iloc[-1]
        explicit_prior = prior_negatives[
            prior_negatives["explicit_late_blight_absent"].eq(True)
        ]
        target_specific_prior = prior_negatives[
            prior_negatives["zero_prevalence_late_blight"].eq(True)
            | prior_negatives["target_explicit_absence_from_text"].eq(True)
        ]
        strata = set(group["geolocation_stratum"])
        rows.append(
            {
                "event_id": f"{field_uid}__{season}",
                "field_uid": field_uid,
                "season": int(season),
                "latitude": float(group.iloc[0]["latitude"]),
                "longitude": float(group.iloc[0]["longitude"]),
                "last_negative_date": last_negative["observation_date"],
                "first_observation_date": group.iloc[0]["observation_date"],
                "first_positive_date": first_positive["observation_date"],
                "onset_interval_days": int(
                    (first_positive["observation_date"] - last_negative["observation_date"]).days
                ),
                "explicit_negative_before_positive": bool(not explicit_prior.empty),
                "target_specific_negative_before_positive": bool(
                    not target_specific_prior.empty
                ),
                "last_negative_is_explicit": bool(
                    last_negative["explicit_late_blight_absent"]
                ),
                "last_negative_is_target_specific": bool(
                    last_negative["zero_prevalence_late_blight"]
                    or last_negative["target_explicit_absence_from_text"]
                ),
                "last_explicit_negative_date": (
                    explicit_prior.iloc[-1]["observation_date"]
                    if not explicit_prior.empty
                    else pd.NaT
                ),
                "days_from_last_explicit_negative": (
                    int(
                        (
                            first_positive["observation_date"]
                            - explicit_prior.iloc[-1]["observation_date"]
                        ).days
                    )
                    if not explicit_prior.empty
                    else pd.NA
                ),
                "event_geolocation_stratum": (
                    "direct_supported"
                    if "direct" in strata or "mixed" in strata
                    else "recovered_only"
                ),
                "municipality": first_positive["municipality"],
                "parish": first_positive["parish"],
            }
        )
    return pd.DataFrame(rows)


def build_negative_seasons(visits: pd.DataFrame) -> pd.DataFrame:
    """Summarize negative-only field seasons without declaring them true negatives."""

    rows: list[dict[str, object]] = []
    for (field_uid, season), group in visits.groupby(["field_uid", "season"], sort=True):
        known = group[group["late_blight_detected"].notna()].sort_values("observation_date")
        if known.empty or known["late_blight_detected"].eq(True).any():
            continue
        negatives = known[known["late_blight_detected"].eq(False)]
        if negatives.empty:
            continue
        first_date = negatives.iloc[0]["observation_date"]
        last_date = negatives.iloc[-1]["observation_date"]
        ordered_dates = negatives["observation_date"].tolist()
        gaps = [
            int((current - previous).days)
            for previous, current in zip(ordered_dates, ordered_dates[1:])
        ]
        maximum_gap = max(gaps) if gaps else pd.NA
        dense_late_season_followup = bool(
            len(negatives) >= 3
            and last_date >= date(int(season), 8, 15)
            and pd.notna(maximum_gap)
            and int(maximum_gap) <= 14
        )
        full_risk_season_coverage = bool(
            dense_late_season_followup
            and first_date <= date(int(season), 6, 15)
        )
        rows.append(
            {
                "field_uid": field_uid,
                "season": int(season),
                "latitude": float(group.iloc[0]["latitude"]),
                "longitude": float(group.iloc[0]["longitude"]),
                "municipality": negatives.iloc[-1]["municipality"],
                "parish": negatives.iloc[-1]["parish"],
                "negative_visit_count": int(len(negatives)),
                "explicit_negative_visit_count": int(
                    negatives["explicit_late_blight_absent"].eq(True).sum()
                ),
                "target_specific_negative_visit_count": int(
                    (
                        negatives["zero_prevalence_late_blight"].eq(True)
                        | negatives["target_explicit_absence_from_text"].eq(True)
                    ).sum()
                ),
                "generic_explicit_negative_visit_count": int(
                    (
                        negatives["explicit_no_harmful_organisms"].eq(True)
                        | negatives["generic_explicit_absence_from_text"].eq(True)
                    ).sum()
                ),
                "last_negative_is_explicit": bool(
                    negatives.iloc[-1]["explicit_late_blight_absent"]
                ),
                "last_negative_is_target_specific": bool(
                    negatives.iloc[-1]["zero_prevalence_late_blight"]
                    or negatives.iloc[-1]["target_explicit_absence_from_text"]
                ),
                "first_negative_date": first_date,
                "last_negative_date": last_date,
                "followup_span_days": int((last_date - first_date).days),
                "maximum_visit_gap_days": maximum_gap,
                "lenient_coverage": bool(len(negatives) >= 3 and last_date.month >= 8),
                "dense_late_season_followup": dense_late_season_followup,
                "strict_coverage": full_risk_season_coverage,
                "full_risk_season_coverage": full_risk_season_coverage,
                "strict_explicit_coverage": _strict_explicit_negative_coverage(
                    negatives, int(season)
                ),
            }
        )
    return pd.DataFrame(rows)


def _strict_explicit_negative_coverage(
    negatives: pd.DataFrame, season: int
) -> bool:
    """Require repeated explicit negatives from mid-June through mid-August.

    This remains a surveillance-quality sensitivity definition because crop
    protection treatments and the inspection protocol are unavailable.
    """

    explicit = negatives[negatives["explicit_late_blight_absent"].eq(True)].sort_values(
        "observation_date"
    )
    if (
        len(explicit) < 3
        or explicit.iloc[0]["observation_date"] > date(season, 6, 15)
        or explicit.iloc[-1]["observation_date"] < date(season, 8, 15)
    ):
        return False
    gaps = explicit["observation_date"].diff().dropna().map(lambda value: value.days)
    return bool(not gaps.empty and int(gaps.max()) <= 14)


def _is_assumed_early_stage(stages: object) -> bool:
    """Return the deliberately weak BBCH 0--29 sensitivity assumption.

    Potato principal stages can overlap.  In particular BBCH 31--39 (canopy
    cover) does not prove that reproductive BBCH 5x has not started, so those
    codes must not form a lower budding bound.
    """

    values = tuple(stages) if isinstance(stages, (tuple, list)) else ()
    return bool(values) and all(0 <= int(value) <= 29 for value in values)


def _has_reproductive_stage(stages: object) -> bool:
    values = tuple(stages) if isinstance(stages, (tuple, list)) else ()
    return any(51 <= int(value) <= 89 for value in values)


def build_polyakov_cases(visits: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Build an exploratory early-stage-to-reproductive activation interval.

    Only BBCH 0--29 is admitted as a weak sensitivity assumption.  The result is
    not a validated budding bracket because the source has no explicit
    ``flower buds absent`` observation.
    """

    rows: list[dict[str, object]] = []
    if events.empty:
        return pd.DataFrame()
    for event in events.itertuples(index=False):
        group = visits[
            visits["field_uid"].eq(event.field_uid)
            & visits["season"].eq(event.season)
            & (visits["observation_date"] < event.first_positive_date)
        ].sort_values("observation_date")
        reproductive = group[group["growth_stage_codes"].map(_has_reproductive_stage)]
        if reproductive.empty:
            continue
        first_reproductive = reproductive.iloc[0]
        definite_pre = group[
            (group["observation_date"] < first_reproductive["observation_date"])
            & group["growth_stage_codes"].map(_is_assumed_early_stage)
        ]
        if definite_pre.empty:
            continue
        last_pre = definite_pre.iloc[-1]
        width = int(
            (first_reproductive["observation_date"] - last_pre["observation_date"]).days
        )
        if width <= 0:
            continue
        rows.append(
            {
                **event._asdict(),
                "phenophase_interval_start": last_pre["observation_date"],
                "phenophase_interval_end": first_reproductive["observation_date"],
                "phenophase_interval_days": width,
                "first_reproductive_stage_codes": first_reproductive["growth_stage_codes"],
                "exact_bbch51_at_interval_end": bool(
                    51 in first_reproductive["growth_stage_codes"]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_polyakov_point_cases(visits: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Use the first observed BBCH 51 date as a point activation sensitivity.

    This is less rigorous than a bracketed field transition: the visit date is
    an upper bound on the unobserved onset of BBCH 51.  It is retained because
    it provides the largest transparent exact-stage sample and a direct check
    against the simple ``activation + 15..17 days`` phenology-only baseline.
    """

    rows: list[dict[str, object]] = []
    if events.empty:
        return pd.DataFrame()
    for event in events.itertuples(index=False):
        group = visits[
            visits["field_uid"].eq(event.field_uid)
            & visits["season"].eq(event.season)
            & (visits["observation_date"] < event.first_positive_date)
        ].sort_values("observation_date")
        exact = group[
            group["growth_stage_codes"].map(
                lambda values: 51 in values if isinstance(values, (tuple, list)) else False
            )
        ]
        if exact.empty:
            continue
        activation_date = exact.iloc[0]["observation_date"]
        rows.append(
            {
                **event._asdict(),
                "phenophase_interval_start": activation_date - timedelta(days=1),
                "phenophase_interval_end": activation_date,
                "phenophase_interval_days": 1,
                "first_reproductive_stage_codes": (51,),
                "exact_bbch51_at_interval_end": True,
            }
        )
    return pd.DataFrame(rows)


def build_polyakov_exact_interval_cases(
    visits: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """Build a weak BBCH 0--29 to observed-BBCH51 sensitivity interval."""

    rows: list[dict[str, object]] = []
    if events.empty:
        return pd.DataFrame()
    for event in events.itertuples(index=False):
        group = visits[
            visits["field_uid"].eq(event.field_uid)
            & visits["season"].eq(event.season)
            & (visits["observation_date"] < event.first_positive_date)
        ].sort_values("observation_date")
        exact = group[
            group["growth_stage_codes"].map(
                lambda values: 51 in values if isinstance(values, (tuple, list)) else False
            )
        ]
        if exact.empty:
            continue
        observed_bbch51 = exact.iloc[0]
        definite_pre = group[
            (group["observation_date"] < observed_bbch51["observation_date"])
            & group["growth_stage_codes"].map(_is_assumed_early_stage)
        ]
        if definite_pre.empty:
            continue
        last_pre = definite_pre.iloc[-1]
        width = int(
            (observed_bbch51["observation_date"] - last_pre["observation_date"]).days
        )
        if width <= 0:
            continue
        rows.append(
            {
                **event._asdict(),
                "phenophase_interval_start": last_pre["observation_date"],
                "phenophase_interval_end": observed_bbch51["observation_date"],
                "phenophase_interval_days": width,
                "first_reproductive_stage_codes": (51,),
                "exact_bbch51_at_interval_end": True,
            }
        )
    return pd.DataFrame(rows)


def _canonical_cache_key(params: dict[str, Any]) -> str:
    encoded = json.dumps(params, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _fetch_batch_payload(
    session: requests.Session,
    cache_dir: Path,
    coordinates: pd.DataFrame,
    start_date: str,
    end_date: str,
    model: str,
    variables: tuple[str, ...],
    timeout_seconds: float = 180.0,
    max_attempts: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params: dict[str, Any] = {
        "latitude": ",".join(f"{value:.8f}" for value in coordinates["latitude"]),
        "longitude": ",".join(f"{value:.8f}" for value in coordinates["longitude"]),
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "models": model,
        "timezone": "auto",
        "cell_selection": "land",
    }
    if "temperature_2m" in variables:
        params["temperature_unit"] = "celsius"
    if "precipitation" in variables:
        params["precipitation_unit"] = "mm"
    key = _canonical_cache_key(params)
    raw_path = cache_dir / f"{key}.json"
    metadata_path = cache_dir / f"{key}.metadata.json"
    if raw_path.exists() and metadata_path.exists():
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        payloads = payload if isinstance(payload, list) else [payload]
        if len(payloads) != len(coordinates):
            raise OpenMeteoError(f"Cached batch has unexpected location count: {raw_path}")
        return payloads, {**metadata, "cache_hit": True, "cache_path": str(raw_path)}

    response: requests.Response | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(ARCHIVE_URL, params=params, timeout=timeout_seconds)
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(
                    f"temporary HTTP {response.status_code}", response=response
                )
            response.raise_for_status()
            payload = response.json()
            payloads = payload if isinstance(payload, list) else [payload]
            if len(payloads) != len(coordinates):
                raise OpenMeteoError(
                    f"Open-Meteo returned {len(payloads)} locations for {len(coordinates)} requests"
                )
            break
        except (requests.RequestException, ValueError, OpenMeteoError):
            if attempt == max_attempts:
                raise
            time.sleep(2 ** (attempt - 1))
    else:  # pragma: no cover
        raise OpenMeteoError("Open-Meteo batch failed")

    assert response is not None
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(response.content)
    metadata = {
        "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "request_parameters": params,
        "url": response.url,
        "location_count": len(coordinates),
        "cache_hit": False,
        "cache_path": str(raw_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payloads, metadata


def fetch_weather_daily(
    visits: pd.DataFrame,
    cache_dir: str | Path = "data/cache/vaad_open_meteo",
    batch_size: int = 20,
    maximum_season: int = 2025,
    progress: Callable[[str], None] | None = print,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download/cache April--September weather and return field-day features."""

    field_keys = visits[visits["season"].le(maximum_season)][
        ["field_uid", "season", "latitude", "longitude"]
    ].drop_duplicates(["field_uid", "season"])
    if field_keys.empty:
        return pd.DataFrame(), pd.DataFrame()
    cache_path = Path(cache_dir)
    signature_source = field_keys.sort_values(["field_uid", "season"]).to_csv(
        index=False
    ) + "\nvaad_weather_daily_v2_hutton_1.0.1_polyakov_1.1.0"
    processed_key = hashlib.sha256(signature_source.encode("utf-8")).hexdigest()
    processed_weather_path = cache_path / f"{processed_key}.daily.pkl"
    processed_metadata_path = cache_path / f"{processed_key}.daily_metadata.pkl"
    if processed_weather_path.exists() and processed_metadata_path.exists():
        if progress:
            progress("weather daily cache hit")
        return (
            pd.read_pickle(processed_weather_path),
            pd.read_pickle(processed_metadata_path),
        )
    duplicate_coords = field_keys.duplicated(
        ["season", "latitude", "longitude"], keep=False
    )
    unique_coords = field_keys.drop_duplicates(
        ["season", "latitude", "longitude"]
    ).sort_values(["season", "latitude", "longitude"])

    session = requests.Session()
    daily_tables: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []
    for season, season_coords in unique_coords.groupby("season", sort=True):
        season_coords = season_coords.reset_index(drop=True)
        start = f"{int(season):04d}-04-01"
        end = f"{int(season):04d}-09-30"
        for offset in range(0, len(season_coords), batch_size):
            chunk = season_coords.iloc[offset : offset + batch_size].reset_index(drop=True)
            if progress:
                progress(
                    f"weather {season}: locations {offset + 1}-{offset + len(chunk)} "
                    f"of {len(season_coords)}"
                )
            land_payloads, land_meta = _fetch_batch_payload(
                session,
                cache_path,
                chunk,
                start,
                end,
                "era5_land",
                ("temperature_2m", "relative_humidity_2m"),
            )
            rain_payloads, rain_meta = _fetch_batch_payload(
                session,
                cache_path,
                chunk,
                start,
                end,
                "era5",
                ("precipitation",),
            )
            for index, coordinate in chunk.iterrows():
                land_frame, land_response_meta = parse_archive_response(
                    land_payloads[index], ("temperature_2m", "relative_humidity_2m")
                )
                rain_frame, rain_response_meta = parse_archive_response(
                    rain_payloads[index], ("precipitation",)
                )
                land_timezone = str(land_response_meta.get("timezone"))
                rain_timezone = str(rain_response_meta.get("timezone"))
                if land_timezone != rain_timezone:
                    raise OpenMeteoError(
                        f"Weather source timezone mismatch: {land_timezone} != {rain_timezone}"
                    )
                merged = merge_polyakov_weather_sources(land_frame, rain_frame)
                hutton_daily = classify_hutton_days(
                    land_frame, land_timezone, start_date=start, end_date=end
                )
                hutton_periods = classify_hutton_periods(hutton_daily)
                hutton_period_status_by_end = dict(
                    zip(
                        hutton_periods["period_end"],
                        hutton_periods["period_status"],
                        strict=True,
                    )
                )
                polyakov_daily = aggregate_polyakov_daily(
                    merged, land_timezone, start_date=start, end_date=end
                )
                combined = hutton_daily[
                    [
                        "date",
                        "minimum_temperature_c",
                        "high_humidity_hours",
                        "data_complete",
                        "day_status",
                    ]
                ].merge(polyakov_daily, on="date", how="inner", validate="one_to_one")
                combined["hutton_period_status"] = combined["date"].map(
                    hutton_period_status_by_end
                )
                combined["season"] = int(season)
                combined["latitude"] = float(coordinate["latitude"])
                combined["longitude"] = float(coordinate["longitude"])
                daily_tables.append(combined)
                metadata_rows.append(
                    {
                        "season": int(season),
                        "requested_latitude": float(coordinate["latitude"]),
                        "requested_longitude": float(coordinate["longitude"]),
                        "timezone": land_timezone,
                        "era5_land_returned_latitude": land_response_meta.get(
                            "returned_latitude"
                        ),
                        "era5_land_returned_longitude": land_response_meta.get(
                            "returned_longitude"
                        ),
                        "era5_returned_latitude": rain_response_meta.get(
                            "returned_latitude"
                        ),
                        "era5_returned_longitude": rain_response_meta.get(
                            "returned_longitude"
                        ),
                        "era5_land_cache_hit": land_meta["cache_hit"],
                        "era5_cache_hit": rain_meta["cache_hit"],
                        "era5_land_retrieved_at": land_meta.get("retrieved_at"),
                        "era5_retrieved_at": rain_meta.get("retrieved_at"),
                        "era5_land_cache_path": land_meta["cache_path"],
                        "era5_cache_path": rain_meta["cache_path"],
                    }
                )

    coordinate_daily = pd.concat(daily_tables, ignore_index=True)
    weather = field_keys.merge(
        coordinate_daily,
        on=["season", "latitude", "longitude"],
        how="left",
        validate="many_to_many" if bool(duplicate_coords.any()) else "one_to_many",
    )
    metadata = pd.DataFrame(metadata_rows)
    cache_path.mkdir(parents=True, exist_ok=True)
    weather.to_pickle(processed_weather_path)
    metadata.to_pickle(processed_metadata_path)
    return weather, metadata


def _tri_any(values: Iterable[object]) -> object:
    normalized = [None if pd.isna(value) else bool(value) for value in values]
    if any(value is True for value in normalized):
        return True
    if any(value is None for value in normalized):
        return pd.NA
    return False


def _tri_all(values: Iterable[object]) -> object:
    normalized = [None if pd.isna(value) else bool(value) for value in values]
    if any(value is False for value in normalized):
        return False
    if any(value is None for value in normalized):
        return pd.NA
    return bool(normalized)


def _recent_hutton_signal(
    weather: pd.DataFrame, target: date, lookback_days: int
) -> object:
    window = weather[
        (weather["date"] >= target - timedelta(days=lookback_days))
        & (weather["date"] <= target - timedelta(days=1))
    ]
    if window.empty:
        return pd.NA
    statuses = window["hutton_period_status"]
    if statuses.eq("pass").any():
        return True
    expected_dates = {
        target - timedelta(days=offset) for offset in range(1, lookback_days + 1)
    }
    observed_dates = set(window["date"])
    if (
        observed_dates != expected_dates
        or statuses.eq("indeterminate").any()
        or statuses.isna().any()
    ):
        return pd.NA
    return False


def evaluate_hutton_events(
    events: pd.DataFrame,
    weather: pd.DataFrame,
    lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        field_weather = weather[
            weather["field_uid"].eq(event.field_uid)
            & weather["season"].eq(event.season)
        ].sort_values("date")
        result = event._asdict()
        if field_weather.empty:
            result["nearest_prior_hutton_period_end"] = pd.NaT
            result["days_from_nearest_hutton_period"] = pd.NA
            for lookback in lookbacks:
                for suffix in (
                    "first_positive",
                    "last_negative",
                    "interval_possible",
                    "interval_robust",
                    "interval_hit_fraction",
                ):
                    result[f"hutton_{lookback}d_{suffix}"] = pd.NA
                result[f"hutton_{lookback}d_interval_not_scorable_days"] = int(
                    event.onset_interval_days
                )
            rows.append(result)
            continue
        pass_dates = field_weather.loc[
            field_weather["hutton_period_status"].eq("pass"), "date"
        ]
        prior_passes = pass_dates[pass_dates < event.first_positive_date]
        result["nearest_prior_hutton_period_end"] = (
            max(prior_passes) if not prior_passes.empty else pd.NaT
        )
        result["days_from_nearest_hutton_period"] = (
            int((event.first_positive_date - max(prior_passes)).days)
            if not prior_passes.empty
            else pd.NA
        )
        candidate_onsets = [
            value.date()
            for value in pd.date_range(
                event.last_negative_date + timedelta(days=1),
                event.first_positive_date,
                freq="D",
            )
        ]
        for lookback in lookbacks:
            candidate_hits = [
                _recent_hutton_signal(field_weather, value, lookback)
                for value in candidate_onsets
            ]
            result[f"hutton_{lookback}d_first_positive"] = _recent_hutton_signal(
                field_weather, event.first_positive_date, lookback
            )
            result[f"hutton_{lookback}d_last_negative"] = _recent_hutton_signal(
                field_weather, event.last_negative_date, lookback
            )
            result[f"hutton_{lookback}d_interval_possible"] = _tri_any(
                candidate_hits
            )
            result[f"hutton_{lookback}d_interval_robust"] = _tri_all(
                candidate_hits
            )
            evaluable_hits = [
                bool(value) for value in candidate_hits if not pd.isna(value)
            ]
            result[f"hutton_{lookback}d_interval_hit_fraction"] = (
                float(np.mean(evaluable_hits)) if evaluable_hits else pd.NA
            )
            result[f"hutton_{lookback}d_interval_not_scorable_days"] = int(
                sum(pd.isna(value) for value in candidate_hits)
            )
        rows.append(result)
    return pd.DataFrame(rows)


def evaluate_negative_seasons(
    negative_seasons: pd.DataFrame,
    visits: pd.DataFrame,
    weather: pd.DataFrame,
    lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season_row in negative_seasons.itertuples(index=False):
        field_visits = visits[
            visits["field_uid"].eq(season_row.field_uid)
            & visits["season"].eq(season_row.season)
            & visits["late_blight_detected"].eq(False)
        ]
        field_weather = weather[
            weather["field_uid"].eq(season_row.field_uid)
            & weather["season"].eq(season_row.season)
        ]
        if field_weather.empty:
            continue
        result = season_row._asdict()
        for lookback in lookbacks:
            visit_hits = [
                _recent_hutton_signal(field_weather, value, lookback)
                for value in field_visits["observation_date"]
            ]
            result[f"hutton_{lookback}d_any_negative_visit"] = _tri_any(
                visit_hits
            )
            evaluable_hits = [
                bool(value) for value in visit_hits if not pd.isna(value)
            ]
            result[f"hutton_{lookback}d_negative_visit_fraction"] = (
                float(np.mean(evaluable_hits)) if evaluable_hits else pd.NA
            )
            result[f"hutton_{lookback}d_last_negative"] = _recent_hutton_signal(
                field_weather, season_row.last_negative_date, lookback
            )
        rows.append(result)
    return pd.DataFrame(rows)


def _date_interval_overlaps(
    predicted_start: date,
    predicted_end: date,
    observed_start_exclusive: date,
    observed_end_inclusive: date,
) -> bool:
    observed_start = observed_start_exclusive + timedelta(days=1)
    return predicted_start <= observed_end_inclusive and predicted_end >= observed_start


def score_interval_uncertainty(
    scenario_prediction_days: list[set[date]],
    observed_start_exclusive: date,
    observed_end_inclusive: date,
) -> dict[str, bool]:
    """Score both activation-date and disease-onset interval uncertainty."""

    onset_days = {
        value.date()
        for value in pd.date_range(
            observed_start_exclusive + timedelta(days=1),
            observed_end_inclusive,
            freq="D",
        )
    }
    overlaps = [bool(days & onset_days) for days in scenario_prediction_days]
    covers = [onset_days.issubset(days) for days in scenario_prediction_days]
    return {
        "activation_any__onset_any": bool(any(overlaps)),
        "activation_all__onset_any": bool(overlaps and all(overlaps)),
        "activation_any__onset_all": bool(any(covers)),
        "activation_all__onset_all": bool(covers and all(covers)),
    }


def _window_date_set(start: date, end: date) -> set[date]:
    return {value.date() for value in pd.date_range(start, end, freq="D")}


def evaluate_polyakov_cases(
    cases: pd.DataFrame,
    weather: pd.DataFrame,
    config: PolyakovConfig = PolyakovConfig(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case in cases.itertuples(index=False):
        field_weather = weather[
            weather["field_uid"].eq(case.field_uid)
            & weather["season"].eq(case.season)
        ].sort_values("date")
        if field_weather.empty:
            continue
        daily = field_weather[
            [
                "date",
                "temperature_mean_c",
                "relative_humidity_mean_pct",
                "precipitation_sum_mm",
                "accepted",
            ]
        ].copy()
        activation_dates = [
            value.date()
            for value in pd.date_range(
                case.phenophase_interval_start + timedelta(days=1),
                case.phenophase_interval_end,
                freq="D",
            )
        ]
        relevant_weather = daily[
            (daily["date"] >= min(activation_dates))
            & (daily["date"] < case.first_positive_date)
        ]
        scenario_prediction_sets: list[set[date]] = []
        baseline_prediction_sets: list[set[date]] = []
        nominal_predicted_union: set[date] = set()
        nominal_baseline_union: set[date] = set()
        operational_predicted_union: set[date] = set()
        operational_baseline_union: set[date] = set()
        full_scenario_payload: list[dict[str, object]] = []
        signal_issue_dates: list[date] = []
        for activation_date in activation_dates:
            classified = classify_polyakov_windows(daily, activation_date, config)
            episodes = extract_polyakov_episodes(classified, config)
            confirmed = episodes[episodes["persistence_confirmed"].eq(True)]
            payload_windows: list[dict[str, str]] = []
            operational_prediction_days: set[date] = set()
            for row in confirmed.itertuples(index=False):
                issued_on = row.episode_start + timedelta(days=config.persistence_days)
                start = row.expected_manifestation_start
                end = row.expected_manifestation_end
                payload_windows.append(
                    {
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                        "issued_on": issued_on.isoformat(),
                    }
                )
                # The observation day's reanalysis is not allowed to create a
                # retrospective signal for that same observation.  Because the
                # daily inputs are known only after day close, the issue day is
                # not counted as an actionable alarm day either.
                if issued_on < case.first_positive_date:
                    nominal_days = _window_date_set(start, end)
                    nominal_predicted_union.update(nominal_days)
                    actionable_days = {
                        value
                        for value in nominal_days
                        if issued_on < value <= case.first_positive_date
                    }
                    operational_prediction_days.update(actionable_days)
                    operational_predicted_union.update(
                        value
                        for value in actionable_days
                        if value < case.first_positive_date
                    )
                    signal_issue_dates.append(issued_on)
            scenario_prediction_sets.append(operational_prediction_days)
            full_scenario_payload.append(
                {
                    "activation_date": activation_date.isoformat(),
                    "windows": payload_windows,
                }
            )
            baseline_start = activation_date + timedelta(
                days=config.window_days - 1 + config.manifestation_lag_days[0]
            )
            baseline_end = activation_date + timedelta(
                days=config.window_days - 1 + config.manifestation_lag_days[1]
            )
            baseline_days = _window_date_set(baseline_start, baseline_end)
            nominal_baseline_union.update(baseline_days)
            operational_baseline_days = {
                value for value in baseline_days if value <= case.first_positive_date
            }
            baseline_prediction_sets.append(operational_baseline_days)
            operational_baseline_union.update(
                value
                for value in operational_baseline_days
                if value < case.first_positive_date
            )
        polyakov_scores = score_interval_uncertainty(
            scenario_prediction_sets,
            case.last_negative_date,
            case.first_positive_date,
        )
        baseline_scores = score_interval_uncertainty(
            baseline_prediction_sets,
            case.last_negative_date,
            case.first_positive_date,
        )
        result = case._asdict()
        result["activation_scenario_count"] = len(activation_dates)
        for key, value in polyakov_scores.items():
            result[f"polyakov_{key}"] = value
        for key, value in baseline_scores.items():
            result[f"phenology_only_{key}"] = value
        scenario_overlaps = [
            bool(
                days
                & {
                    value.date()
                    for value in pd.date_range(
                        case.last_negative_date + timedelta(days=1),
                        case.first_positive_date,
                        freq="D",
                    )
                }
            )
            for days in scenario_prediction_sets
        ]
        baseline_overlaps = [
            bool(
                days
                & {
                    value.date()
                    for value in pd.date_range(
                        case.last_negative_date + timedelta(days=1),
                        case.first_positive_date,
                        freq="D",
                    )
                }
            )
            for days in baseline_prediction_sets
        ]
        # Backward-compatible aliases, retained so old notebooks fail visibly
        # only on changed interpretation rather than missing columns.
        result["polyakov_interval_possible"] = polyakov_scores[
            "activation_any__onset_any"
        ]
        result["polyakov_interval_robust"] = polyakov_scores[
            "activation_all__onset_any"
        ]
        result["polyakov_scenario_hit_fraction"] = float(np.mean(scenario_overlaps))
        result["polyakov_bbch_upper_bound_hit"] = bool(scenario_overlaps[-1])
        result["phenology_only_interval_possible"] = baseline_scores[
            "activation_any__onset_any"
        ]
        result["phenology_only_interval_robust"] = baseline_scores[
            "activation_all__onset_any"
        ]
        result["phenology_only_bbch_upper_bound_hit"] = bool(
            baseline_overlaps[-1]
        )
        result["predicted_manifestation_dates_union"] = "|".join(
            value.isoformat() for value in sorted(nominal_predicted_union)
        )
        result["predicted_manifestation_day_count"] = len(nominal_predicted_union)
        result["phenology_only_predicted_dates_union"] = "|".join(
            value.isoformat() for value in sorted(nominal_baseline_union)
        )
        result["phenology_only_predicted_day_count"] = len(nominal_baseline_union)
        result["operational_alarm_dates_before_detection"] = "|".join(
            value.isoformat() for value in sorted(operational_predicted_union)
        )
        result["operational_alarm_day_count_before_detection"] = len(
            operational_predicted_union
        )
        result["phenology_only_alarm_dates_before_detection"] = "|".join(
            value.isoformat() for value in sorted(operational_baseline_union)
        )
        result["phenology_only_alarm_day_count_before_detection"] = len(
            operational_baseline_union
        )
        at_risk_days = _window_date_set(
            min(activation_dates), case.first_positive_date - timedelta(days=1)
        )
        result["phenology_to_detection_at_risk_day_count"] = len(at_risk_days)
        result["operational_alarm_fraction_before_detection"] = (
            len(operational_predicted_union) / len(at_risk_days)
        )
        result["phenology_only_alarm_fraction_before_detection"] = (
            len(operational_baseline_union) / len(at_risk_days)
        )
        result["full_season_prediction_scenarios_json"] = json.dumps(
            full_scenario_payload, separators=(",", ":")
        )
        result["polyakov_weather_complete"] = bool(
            not relevant_weather.empty and relevant_weather["accepted"].all()
        )
        result["polyakov_weather_days_before_detection"] = int(len(relevant_weather))
        result["polyakov_signal_issue_count_before_detection"] = int(
            len(set(signal_issue_dates))
        )
        result["nearest_operational_signal_lead_days"] = (
            min((case.first_positive_date - value).days for value in signal_issue_dates)
            if signal_issue_dates
            else pd.NA
        )
        result["nearest_actionable_alarm_lead_days"] = (
            min(
                (case.first_positive_date - (value + timedelta(days=1))).days
                for value in signal_issue_dates
            )
            if signal_issue_dates
            else pd.NA
        )
        rows.append(result)
    return pd.DataFrame(rows)


def evaluate_polyakov_negative_seasons(
    negative_results: pd.DataFrame,
    visits: pd.DataFrame,
    weather: pd.DataFrame,
    config: PolyakovConfig = PolyakovConfig(),
) -> pd.DataFrame:
    """Estimate Polyakov alert burden in negative-only surveillance seasons.

    These rows are not true negatives.  The function only asks how often the
    model would have emitted an alert before the last registered negative visit
    when an observed BBCH51 activation point is available.
    """

    rows: list[dict[str, object]] = []
    for season_row in negative_results.itertuples(index=False):
        result = season_row._asdict()
        field_visits = visits[
            visits["field_uid"].eq(season_row.field_uid)
            & visits["season"].eq(season_row.season)
            & (visits["observation_date"] <= season_row.last_negative_date)
        ].sort_values("observation_date")
        exact = field_visits[
            field_visits["growth_stage_codes"].map(
                lambda values: 51 in values if isinstance(values, (tuple, list)) else False
            )
        ]
        if exact.empty:
            result.update(
                {
                    "polyakov_point_evaluable": False,
                    "polyakov_point_reason": "missing_observed_bbch51_before_censor",
                    "polyakov_activation_date": pd.NaT,
                    "polyakov_any_alert_before_censor": pd.NA,
                    "polyakov_alert_episode_count": pd.NA,
                    "polyakov_manifestation_alarm_day_count": pd.NA,
                    "polyakov_at_risk_day_count": pd.NA,
                    "polyakov_alarm_day_fraction": pd.NA,
                    "phenology_only_any_alert_before_censor": pd.NA,
                }
            )
            rows.append(result)
            continue
        activation = exact.iloc[0]["observation_date"]
        if activation >= season_row.last_negative_date:
            result.update(
                {
                    "polyakov_point_evaluable": False,
                    "polyakov_point_reason": "bbch51_not_before_censor",
                    "polyakov_activation_date": activation,
                    "polyakov_any_alert_before_censor": pd.NA,
                    "polyakov_alert_episode_count": pd.NA,
                    "polyakov_manifestation_alarm_day_count": pd.NA,
                    "polyakov_at_risk_day_count": pd.NA,
                    "polyakov_alarm_day_fraction": pd.NA,
                    "phenology_only_any_alert_before_censor": pd.NA,
                }
            )
            rows.append(result)
            continue
        field_weather = weather[
            weather["field_uid"].eq(season_row.field_uid)
            & weather["season"].eq(season_row.season)
            & (weather["date"] < season_row.last_negative_date)
        ].sort_values("date")
        daily = field_weather[
            [
                "date",
                "temperature_mean_c",
                "relative_humidity_mean_pct",
                "precipitation_sum_mm",
                "accepted",
            ]
        ].copy()
        relevant = daily[daily["date"] >= activation]
        if relevant.empty or not bool(relevant["accepted"].all()):
            result.update(
                {
                    "polyakov_point_evaluable": False,
                    "polyakov_point_reason": "incomplete_weather_before_censor",
                    "polyakov_activation_date": activation,
                    "polyakov_any_alert_before_censor": pd.NA,
                    "polyakov_alert_episode_count": pd.NA,
                    "polyakov_manifestation_alarm_day_count": pd.NA,
                    "polyakov_at_risk_day_count": int(
                        (season_row.last_negative_date - activation).days
                    ),
                    "polyakov_alarm_day_fraction": pd.NA,
                    "phenology_only_any_alert_before_censor": pd.NA,
                }
            )
            rows.append(result)
            continue
        classified = classify_polyakov_windows(daily, activation, config)
        episodes = extract_polyakov_episodes(classified, config)
        confirmed = episodes[episodes["persistence_confirmed"].eq(True)]
        at_risk_days = {
            value.date()
            for value in pd.date_range(
                activation,
                season_row.last_negative_date,
                inclusive="left",
                freq="D",
            )
        }
        predicted_days: set[date] = set()
        episode_count = 0
        for episode in confirmed.itertuples(index=False):
            issued_on = episode.episode_start + timedelta(days=config.persistence_days)
            if issued_on >= season_row.last_negative_date:
                continue
            actionable_days = {
                value
                for value in _window_date_set(
                    episode.expected_manifestation_start,
                    episode.expected_manifestation_end,
                )
                if value > issued_on and value in at_risk_days
            }
            if actionable_days:
                episode_count += 1
                predicted_days.update(actionable_days)
        alarm_days = predicted_days & at_risk_days
        baseline_start = activation + timedelta(
            days=config.window_days - 1 + config.manifestation_lag_days[0]
        )
        baseline_end = activation + timedelta(
            days=config.window_days - 1 + config.manifestation_lag_days[1]
        )
        baseline_days = _window_date_set(baseline_start, baseline_end) & at_risk_days
        result.update(
            {
                "polyakov_point_evaluable": True,
                "polyakov_point_reason": "",
                "polyakov_activation_date": activation,
                "polyakov_any_alert_before_censor": bool(alarm_days),
                "polyakov_alert_episode_count": int(episode_count),
                "polyakov_manifestation_alarm_day_count": int(len(alarm_days)),
                "polyakov_at_risk_day_count": int(len(at_risk_days)),
                "polyakov_alarm_day_fraction": (
                    len(alarm_days) / len(at_risk_days) if at_risk_days else pd.NA
                ),
                "phenology_only_any_alert_before_censor": bool(baseline_days),
            }
        )
        rows.append(result)
    return pd.DataFrame(rows)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    low = 0.0 if successes == 0 else max(0.0, centre - margin)
    high = 1.0 if successes == total else min(1.0, centre + margin)
    return low, high


def paired_one_sided_binomial_p(supportive: int, contradictory: int) -> float:
    discordant = supportive + contradictory
    if discordant == 0:
        return 1.0
    return float(
        sum(math.comb(discordant, value) for value in range(supportive, discordant + 1))
        / (2**discordant)
    )


def paired_binary_summary(
    frame: pd.DataFrame, model_column: str, baseline_column: str
) -> dict[str, object]:
    paired = frame[[model_column, baseline_column]].dropna()
    both = int((paired[model_column].eq(True) & paired[baseline_column].eq(True)).sum())
    model_only = int(
        (paired[model_column].eq(True) & paired[baseline_column].eq(False)).sum()
    )
    baseline_only = int(
        (paired[model_column].eq(False) & paired[baseline_column].eq(True)).sum()
    )
    neither = int(
        (paired[model_column].eq(False) & paired[baseline_column].eq(False)).sum()
    )
    total = int(len(paired))
    model_rate = (both + model_only) / total if total else None
    baseline_rate = (both + baseline_only) / total if total else None
    one_sided = paired_one_sided_binomial_p(model_only, baseline_only)
    reverse = paired_one_sided_binomial_p(baseline_only, model_only)
    return {
        "n_total": int(len(frame)),
        "n_paired": total,
        "n_not_scorable": int(len(frame) - total),
        "both_hit": both,
        "model_only": model_only,
        "baseline_only": baseline_only,
        "neither": neither,
        "model_rate": model_rate,
        "baseline_rate": baseline_rate,
        "absolute_lift": (
            model_rate - baseline_rate if total else None
        ),
        "mcnemar_p_one_sided_model_better": one_sided if total else None,
        "mcnemar_p_two_sided": min(1.0, 2.0 * min(one_sided, reverse)) if total else None,
    }


def _holm_adjust(p_values: list[float | None]) -> list[float | None]:
    indexed = [(index, value) for index, value in enumerate(p_values) if value is not None]
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    total = len(indexed)
    for rank, (index, value) in enumerate(sorted(indexed, key=lambda item: item[1])):
        running = max(running, min(1.0, float(value) * (total - rank)))
        adjusted[index] = running
    return adjusted


def hutton_date_permutation_test(
    results: pd.DataFrame,
    weather: pd.DataFrame,
    lookback: int,
    repetitions: int = 2000,
    seed: int = DEFAULT_SEED,
    strata: tuple[str, ...] = ("season",),
) -> dict[str, float | int | None]:
    """Shuffle first-detection dates within predeclared temporal/spatial strata."""

    if results.empty:
        return {
            "observed_rate": None,
            "null_mean_rate": None,
            "null_p025_rate": None,
            "null_p975_rate": None,
            "lift": None,
            "permutation_p_one_sided": None,
            "repetitions": repetitions,
        }
    lookup = {
        (field_uid, int(season)): frame
        for (field_uid, season), frame in weather.groupby(["field_uid", "season"])
    }
    observed_values = results[f"hutton_{lookback}d_first_positive"].dropna()
    if observed_values.empty:
        return {
            "observed_rate": None,
            "null_mean_rate": None,
            "null_p025_rate": None,
            "null_p975_rate": None,
            "lift": None,
            "permutation_p_one_sided": None,
            "repetitions": repetitions,
            "strata": list(strata),
            "reason": "no_scorable_observed_events",
        }
    observed = float(observed_values.astype(bool).mean())
    rng = np.random.default_rng(seed + lookback)
    grouped = [group.copy() for _, group in results.groupby(list(strata), sort=True)]
    signal_matrices: list[np.ndarray] = []
    for group in grouped:
        assigned_dates = group["first_positive_date"].tolist()
        group_rows = list(group.itertuples(index=False))
        matrix = np.zeros((len(group_rows), len(assigned_dates)), dtype=np.int8)
        for row_index, row in enumerate(group_rows):
            field_weather = lookup[(row.field_uid, int(row.season))]
            for date_index, assigned_date in enumerate(assigned_dates):
                value = _recent_hutton_signal(field_weather, assigned_date, lookback)
                if pd.isna(value):
                    return {
                        "observed_rate": observed,
                        "null_mean_rate": None,
                        "null_p025_rate": None,
                        "null_p975_rate": None,
                        "lift": None,
                        "permutation_p_one_sided": None,
                        "repetitions": repetitions,
                        "strata": list(strata),
                        "reason": "permuted_assignment_not_scorable",
                    }
                matrix[row_index, date_index] = int(bool(value))
        signal_matrices.append(matrix)
    null_hit_counts = np.zeros(repetitions, dtype=float)
    for matrix in signal_matrices:
        random_order = np.argsort(
            rng.random((repetitions, matrix.shape[1])), axis=1
        )
        row_indices = np.arange(matrix.shape[0])[None, :]
        null_hit_counts += matrix[row_indices, random_order].sum(axis=1)
    values = null_hit_counts / len(results)
    return {
        "observed_rate": observed,
        "null_mean_rate": float(values.mean()),
        "null_p025_rate": float(np.quantile(values, 0.025)),
        "null_p975_rate": float(np.quantile(values, 0.975)),
        "lift": float(observed - values.mean()),
        "permutation_p_one_sided": float((1 + np.sum(values >= observed)) / (repetitions + 1)),
        "repetitions": repetitions,
        "strata": list(strata),
        "stratum_count": int(len(grouped)),
        "singleton_strata": int(sum(len(group) == 1 for group in grouped)),
    }


def hutton_alarm_burden(
    results: pd.DataFrame,
    weather: pd.DataFrame,
    lookback: int,
    *,
    start_column: str | None = None,
    end_column: str | None = None,
    end_inclusive: bool = True,
) -> dict[str, float | int | None]:
    """Summarize June--August alert coverage with optional observation censoring.

    Rolling alert state is calculated on the complete April--September weather
    sequence before applying the seasonal display window.  This preserves Hutton
    periods ending in May that remain active in an early-June lookback.
    """

    if results.empty:
        return {
            "field_seasons": 0,
            "field_days": 0,
            "scorable_field_days": 0,
            "indeterminate_field_days": 0,
            "alarm_day_fraction": None,
            "median_field_alarm_fraction": None,
            "field_seasons_with_alarm": 0,
            "field_season_alarm_rate": None,
            "alarm_episode_count": 0,
            "median_alarm_episodes_per_field_season": None,
        }
    key_columns = ["field_uid", "season"]
    for column in (start_column, end_column):
        if column is not None and column not in key_columns:
            key_columns.append(column)
    keys = results[key_columns].drop_duplicates(["field_uid", "season"])
    frame = weather.merge(
        keys, on=["field_uid", "season"], how="inner", validate="many_to_one"
    ).copy()
    frame = frame.sort_values(["field_uid", "season", "date"])
    alarm_parts: list[pd.Series] = []
    for _, group in frame.groupby(["field_uid", "season"], sort=False):
        prior_pass = (
            group["hutton_period_status"]
            .eq("pass")
            .shift(1, fill_value=False)
            .rolling(lookback, min_periods=1)
            .max()
            .astype(bool)
        )
        prior_indeterminate = (
            (
                group["hutton_period_status"].eq("indeterminate")
                | group["hutton_period_status"].isna()
            )
            .shift(1, fill_value=False)
            .rolling(lookback, min_periods=1)
            .max()
            .astype(bool)
        )
        dates = pd.to_datetime(group["date"])
        complete_window = (
            dates - dates.shift(lookback)
        ).dt.days.eq(lookback)
        alarm = pd.Series(
            pd.array(
                [
                    True
                    if passed
                    else pd.NA
                    if indeterminate or not complete
                    else False
                    for passed, indeterminate, complete in zip(
                        prior_pass,
                        prior_indeterminate,
                        complete_window,
                        strict=True,
                    )
                ],
                dtype="boolean",
            ),
            index=group.index,
        )
        alarm_parts.append(alarm)
    frame["alarm"] = pd.concat(alarm_parts).sort_index()

    dates = pd.to_datetime(frame["date"])
    eligible = dates.dt.month.between(6, 8)
    if start_column is not None:
        eligible &= frame.apply(
            lambda row: row["date"] >= row[start_column], axis=1
        )
    if end_column is not None:
        if end_inclusive:
            eligible &= frame.apply(
                lambda row: row["date"] <= row[end_column], axis=1
            )
        else:
            eligible &= frame.apply(
                lambda row: row["date"] < row[end_column], axis=1
            )
    frame = frame[eligible].copy()
    if frame.empty:
        return {
            "field_seasons": 0,
            "field_days": 0,
            "scorable_field_days": 0,
            "indeterminate_field_days": 0,
            "alarm_day_fraction": None,
            "median_field_alarm_fraction": None,
            "field_seasons_with_alarm": 0,
            "field_season_alarm_rate": None,
            "alarm_episode_count": 0,
            "median_alarm_episodes_per_field_season": None,
        }

    scorable = frame[frame["alarm"].notna()].copy()
    by_field = scorable.groupby(["field_uid", "season"])["alarm"].mean()
    field_any = scorable.groupby(["field_uid", "season"])["alarm"].any()
    episode_counts: list[int] = []
    for _, group in scorable.groupby(["field_uid", "season"], sort=False):
        states = group.sort_values("date")["alarm"].astype(bool)
        episode_counts.append(int((states & ~states.shift(1, fill_value=False)).sum()))
    return {
        "field_seasons": int(
            frame[["field_uid", "season"]].drop_duplicates().shape[0]
        ),
        "field_days": int(len(frame)),
        "scorable_field_days": int(len(scorable)),
        "indeterminate_field_days": int(frame["alarm"].isna().sum()),
        "alarm_day_fraction": (
            float(scorable["alarm"].mean()) if not scorable.empty else None
        ),
        "median_field_alarm_fraction": (
            float(by_field.median()) if not by_field.empty else None
        ),
        "field_seasons_with_alarm": int(field_any.sum()) if not field_any.empty else 0,
        "field_season_alarm_rate": (
            float(field_any.mean()) if not field_any.empty else None
        ),
        "alarm_episode_count": int(sum(episode_counts)),
        "median_alarm_episodes_per_field_season": (
            float(np.median(episode_counts)) if episode_counts else None
        ),
    }


def polyakov_date_permutation_test(
    results: pd.DataFrame,
    repetitions: int = 2000,
    seed: int = DEFAULT_SEED,
    endpoint: str = "activation_any__onset_any",
) -> dict[str, object]:
    """Shuffle complete disease intervals only within their season year.

    Full-season weather windows are retained for each field, but an assigned
    event may only use windows whose ``issued_on`` precedes that assigned first
    detection.  This avoids both cross-year nulls and observation-day leakage.
    """

    if results.empty:
        return {
            "observed_rate": None,
            "null_mean_rate": None,
            "null_p025_rate": None,
            "null_p975_rate": None,
            "lift": None,
            "permutation_p_one_sided": None,
            "repetitions": repetitions,
        }
    def parse_scenarios(value: object) -> list[dict[str, object]]:
        if value is None or pd.isna(value):
            return []
        try:
            payload = json.loads(str(value))
        except (json.JSONDecodeError, TypeError):
            return []
        return payload if isinstance(payload, list) else []

    def assigned_hit(
        scenarios: list[dict[str, object]],
        start: date,
        end: date,
    ) -> bool:
        scenario_sets: list[set[date]] = []
        for scenario in scenarios:
            predicted_days: set[date] = set()
            for window in scenario.get("windows", []):
                issued_on = date.fromisoformat(str(window["issued_on"]))
                if issued_on >= end:
                    continue
                predicted_days.update(
                    value
                    for value in _window_date_set(
                        date.fromisoformat(str(window["start"])),
                        date.fromisoformat(str(window["end"])),
                    )
                    if value > issued_on
                )
            scenario_sets.append(predicted_days)
        return score_interval_uncertainty(scenario_sets, start, end).get(
            endpoint, False
        )

    rng = np.random.default_rng(seed + len(results))
    grouped = [group.copy() for _, group in results.groupby("season", sort=True)]
    matrices: list[np.ndarray] = []
    observed_hits = 0
    for group in grouped:
        scenarios = [
            parse_scenarios(value)
            for value in group["full_season_prediction_scenarios_json"]
        ]
        intervals = list(
            zip(
                group["last_negative_date"].tolist(),
                group["first_positive_date"].tolist(),
                strict=True,
            )
        )
        matrix = np.asarray(
            [
                [assigned_hit(payload, start, end) for start, end in intervals]
                for payload in scenarios
            ],
            dtype=np.int8,
        )
        matrices.append(matrix)
        observed_hits += int(np.diag(matrix).sum())
    observed = observed_hits / len(results)
    null_hit_counts = np.zeros(repetitions, dtype=float)
    for matrix in matrices:
        random_order = np.argsort(
            rng.random((repetitions, matrix.shape[1])), axis=1
        )
        row_indices = np.arange(matrix.shape[0])[None, :]
        null_hit_counts += matrix[row_indices, random_order].sum(axis=1)
    values = null_hit_counts / len(results)
    return {
        "observed_rate": observed,
        "null_mean_rate": float(values.mean()),
        "null_p025_rate": float(np.quantile(values, 0.025)),
        "null_p975_rate": float(np.quantile(values, 0.975)),
        "lift": float(observed - values.mean()),
        "permutation_p_one_sided": float((1 + np.sum(values >= observed)) / (repetitions + 1)),
        "repetitions": repetitions,
        "endpoint": endpoint,
        "strata": ["season"],
        "season_count": int(len(grouped)),
        "singleton_seasons": int(sum(len(group) == 1 for group in grouped)),
    }


def _rate_summary(frame: pd.DataFrame, column: str) -> dict[str, object]:
    total = int(len(frame))
    values = frame[column].dropna() if total else pd.Series(dtype="boolean")
    evaluable = int(len(values))
    successes = int(values.eq(True).sum()) if evaluable else 0
    low, high = wilson_interval(successes, evaluable)
    return {
        "n": evaluable,
        "n_total": total,
        "n_not_scorable": total - evaluable,
        "successes": successes,
        "rate": successes / evaluable if evaluable else None,
        "wilson_95_low": low if evaluable else None,
        "wilson_95_high": high if evaluable else None,
    }


def _haversine_km(
    latitude: float,
    longitude: float,
    other_latitudes: np.ndarray,
    other_longitudes: np.ndarray,
) -> np.ndarray:
    radius_km = 6371.0088
    lat1 = np.radians(latitude)
    lon1 = np.radians(longitude)
    lat2 = np.radians(other_latitudes)
    lon2 = np.radians(other_longitudes)
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    haversine = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * radius_km * np.arcsin(np.sqrt(haversine))


def annotate_holdout_structure(
    frame: pd.DataFrame, visits: pd.DataFrame
) -> pd.DataFrame:
    """Add exact-field, administrative, and distance-to-development flags."""

    output = frame.copy()
    development = visits[visits["season"].between(2015, 2022)]
    development_fields = set(development["field_uid"])
    development_municipalities = {
        value for value in development["municipality"].astype(str) if value
    }
    development_parishes = {
        value for value in development["parish"].astype(str) if value
    }
    coordinates = development[["latitude", "longitude"]].drop_duplicates()
    latitudes = coordinates["latitude"].to_numpy(dtype=float)
    longitudes = coordinates["longitude"].to_numpy(dtype=float)
    output["field_seen_in_development"] = output["field_uid"].isin(
        development_fields
    )
    municipality = output["municipality"].fillna("").astype(str).str.strip()
    parish = output["parish"].fillna("").astype(str).str.strip()
    output["municipality_known"] = municipality.ne("")
    output["parish_known"] = parish.ne("")
    output["municipality_seen_in_development"] = (
        output["municipality_known"] & municipality.isin(development_municipalities)
    )
    output["parish_seen_in_development"] = (
        output["parish_known"] & parish.isin(development_parishes)
    )
    if coordinates.empty:
        output["distance_to_nearest_development_field_km"] = np.nan
    else:
        output["distance_to_nearest_development_field_km"] = [
            float(
                _haversine_km(
                    float(row.latitude),
                    float(row.longitude),
                    latitudes,
                    longitudes,
                ).min()
            )
            for row in output.itertuples(index=False)
        ]
    return output


def _month_day(value: date) -> tuple[int, int]:
    return value.month, value.day


def _date_for_season(season: int, month_day: tuple[int, int]) -> date:
    return date(int(season), month_day[0], month_day[1])


def _calendar_window_metrics(
    events: pd.DataFrame,
    start_month_day: tuple[int, int],
    end_month_day: tuple[int, int],
) -> dict[str, object]:
    if events.empty:
        return {
            "n": 0,
            "hits": 0,
            "hit_rate": None,
            "june_august_alarm_day_fraction": None,
            "pre_detection_alarm_day_fraction_first_visit_proxy": None,
            "pre_detection_field_days_first_visit_proxy": 0,
            "pre_detection_alarm_day_fraction_june1": None,
            "pre_detection_field_days_june1": 0,
        }
    hits = 0
    fixed_alarm_days = 0
    fixed_total_days = 0
    pre_alarm_days = 0
    pre_total_days = 0
    june1_pre_alarm_days = 0
    june1_pre_total_days = 0
    for row in events.itertuples(index=False):
        start = _date_for_season(row.season, start_month_day)
        end = _date_for_season(row.season, end_month_day)
        hits += int(start <= row.first_positive_date <= end)
        june_start = date(int(row.season), 6, 1)
        august_end = date(int(row.season), 8, 31)
        fixed_days = {
            value.date()
            for value in pd.date_range(june_start, august_end, freq="D")
        }
        alarm_days = {
            value.date() for value in pd.date_range(start, end, freq="D")
        }
        fixed_alarm_days += len(fixed_days & alarm_days)
        fixed_total_days += len(fixed_days)
        observed_start = max(row.first_observation_date, june_start)
        observed_end = min(
            row.first_positive_date - timedelta(days=1), august_end
        )
        if observed_end >= observed_start:
            observed_days = {
                value.date()
                for value in pd.date_range(observed_start, observed_end, freq="D")
            }
            pre_alarm_days += len(observed_days & alarm_days)
            pre_total_days += len(observed_days)
        june1_pre_end = min(
            row.first_positive_date - timedelta(days=1), august_end
        )
        if june1_pre_end >= june_start:
            june1_pre_days = {
                value.date()
                for value in pd.date_range(june_start, june1_pre_end, freq="D")
            }
            june1_pre_alarm_days += len(june1_pre_days & alarm_days)
            june1_pre_total_days += len(june1_pre_days)
    return {
        "n": int(len(events)),
        "hits": int(hits),
        "hit_rate": float(hits / len(events)),
        "june_august_alarm_day_fraction": (
            fixed_alarm_days / fixed_total_days if fixed_total_days else None
        ),
        "pre_detection_alarm_day_fraction_first_visit_proxy": (
            pre_alarm_days / pre_total_days if pre_total_days else None
        ),
        "pre_detection_field_days_first_visit_proxy": int(pre_total_days),
        "pre_detection_alarm_day_fraction_june1": (
            june1_pre_alarm_days / june1_pre_total_days
            if june1_pre_total_days
            else None
        ),
        "pre_detection_field_days_june1": int(june1_pre_total_days),
    }


def calendar_baseline_summary(
    development_events: pd.DataFrame, evaluation_events: pd.DataFrame
) -> dict[str, object]:
    """Fit simple day-of-year windows on development and score without refitting."""

    if development_events.empty:
        return {}
    detection_dates = sorted(development_events["first_positive_date"])
    reference_year = 2001
    detection_doys = np.asarray(
        [date(reference_year, value.month, value.day).timetuple().tm_yday for value in detection_dates],
        dtype=int,
    )
    central_start_doy = int(np.quantile(detection_doys, 0.05, method="nearest"))
    central_end_doy = int(np.quantile(detection_doys, 0.95, method="nearest"))
    minimum_date = date(reference_year, 1, 1) + timedelta(
        days=int(detection_doys.min()) - 1
    )
    maximum_date = date(reference_year, 1, 1) + timedelta(
        days=int(detection_doys.max()) - 1
    )
    central_start = date(reference_year, 1, 1) + timedelta(
        days=central_start_doy - 1
    )
    central_end = date(reference_year, 1, 1) + timedelta(days=central_end_doy - 1)
    definitions = {
        "development_min_max_detection_window": (
            _month_day(minimum_date),
            _month_day(maximum_date),
            True,
            False,
        ),
        "fixed_july_august": ((7, 1), (8, 31), False, True),
        "development_central_90pct_detection_window": (
            _month_day(central_start),
            _month_day(central_end),
            True,
            False,
        ),
    }
    output: dict[str, object] = {}
    for name, (start, end, trained, prespecified) in definitions.items():
        output[name] = {
            "trained_on_development_only": trained,
            "prespecified_fixed_rule": prespecified,
            "start_month_day": f"{start[0]:02d}-{start[1]:02d}",
            "end_month_day": f"{end[0]:02d}-{end[1]:02d}",
            **_calendar_window_metrics(evaluation_events, start, end),
        }
    return output


def _by_season_rate(frame: pd.DataFrame, column: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for season, group in frame.groupby("season", sort=True):
        rows.append({"season": int(season), **_rate_summary(group, column)})
    return rows


def _cohort(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "all_2010_2025":
        return frame[frame["season"].between(2010, 2025)]
    if name == "development_2015_2022":
        return frame[frame["season"].between(2015, 2022)]
    if name == "holdout_2023_2025":
        return frame[frame["season"].between(2023, 2025)]
    if name == "holdout_2023_2025_unseen_fields":
        return frame[
            frame["season"].between(2023, 2025)
            & frame["field_seen_in_development"].eq(False)
        ]
    if name.startswith("holdout_2023_2025_spatial_gt_"):
        threshold = float(name.rsplit("_", 1)[-1].removesuffix("km"))
        return frame[
            frame["season"].between(2023, 2025)
            & frame["field_seen_in_development"].eq(False)
            & frame["distance_to_nearest_development_field_km"].gt(threshold)
        ]
    if name == "holdout_2023_2025_new_municipality":
        return frame[
            frame["season"].between(2023, 2025)
            & frame["municipality_known"].eq(True)
            & frame["municipality_seen_in_development"].eq(False)
        ]
    if name == "holdout_2023_2025_new_parish":
        return frame[
            frame["season"].between(2023, 2025)
            & frame["parish_known"].eq(True)
            & frame["parish_seen_in_development"].eq(False)
        ]
    if name == "early_recovered_2010_2014":
        return frame[frame["season"].between(2010, 2014)]
    raise ValueError(f"Unknown cohort {name}")


def build_summary(
    qc: dict[str, dict[str, object]],
    hutton_results: pd.DataFrame,
    negative_results: pd.DataFrame,
    polyakov_results: pd.DataFrame,
    weather: pd.DataFrame,
    visits_results: pd.DataFrame,
    repetitions: int = 2000,
) -> dict[str, object]:
    cohorts = (
        "all_2010_2025",
        "development_2015_2022",
        "holdout_2023_2025",
        "holdout_2023_2025_unseen_fields",
        "holdout_2023_2025_spatial_gt_1km",
        "holdout_2023_2025_spatial_gt_5km",
        "holdout_2023_2025_spatial_gt_10km",
        "holdout_2023_2025_new_municipality",
        "holdout_2023_2025_new_parish",
        "early_recovered_2010_2014",
    )
    summary: dict[str, object] = {
        "analysis_version": "vaad_late_blight_validation_v3",
        "study_status": "exploratory_retrospective_temporal_association_not_operational_validation",
        "inference_boundary": {
            "can_estimate": [
                "association_with_first_registered_detection",
                "alarm_coverage_during_observed_followup",
                "increment_over_prespecified_simple_baselines",
            ],
            "cannot_estimate_reliably": [
                "biological_onset_sensitivity",
                "specificity",
                "precision_or_ppv",
                "treatment_adjusted_effect",
                "live_forecast_accuracy",
                "causal_or_farm_utility",
            ],
            "holdout_label": "temporal_test_slice_not_untouched_external_holdout",
        },
        "data_qc": qc,
        "weather": {
            "field_seasons": int(
                weather[["field_uid", "season"]].drop_duplicates().shape[0]
            ),
            "daily_rows": int(len(weather)),
            "date_min": min(weather["date"]).isoformat() if not weather.empty else None,
            "date_max": max(weather["date"]).isoformat() if not weather.empty else None,
            "hutton_indeterminate_days": int(
                weather["day_status"].eq("indeterminate").sum()
            ),
            "polyakov_incomplete_days": int((~weather["accepted"]).sum()),
        },
        "hutton": {},
        "polyakov": {},
        "negative_only_season_sensitivity_not_specificity": {},
        "observation_completeness": {},
        "privacy": {
            "raw_and_event_level_outputs_contain_sensitive_field_locations": True,
            "share_only_aggregate_summary": True,
        },
        "data_limitations": {
            "no_structured_treatment_fields": True,
            "no_standardized_target_assessed_absent_not_assessed_state": True,
            "no_explicit_flower_buds_absent_field": True,
            "negative_only_seasons_are_not_confirmed_true_negatives": True,
        },
    }

    completeness: dict[str, object] = {}
    for mode in ("direct", "expanded"):
        visits = visits_results[visits_results["source_mode"].eq(mode)]
        field_seasons = visits.groupby(["field_uid", "season"], sort=False)
        counts = field_seasons.size()
        spans = field_seasons["observation_date"].agg(
            lambda values: (max(values) - min(values)).days
        )
        positive_fs = field_seasons["late_blight_detected"].agg(
            lambda values: bool(values.eq(True).any())
        )
        bbch51_fs = field_seasons["growth_stage_codes"].agg(
            lambda values: any(
                51 in item if isinstance(item, (tuple, list)) else False
                for item in values
            )
        )
        completeness[mode] = {
            "visits": int(len(visits)),
            "field_seasons": int(len(counts)),
            "positive_field_seasons": int(positive_fs.sum()),
            "nonpositive_field_seasons": int((~positive_fs).sum()),
            "field_seasons_with_at_least_2_visits": int((counts >= 2).sum()),
            "field_seasons_with_at_least_3_visits": int((counts >= 3).sum()),
            "median_visits_per_field_season": float(counts.median()),
            "median_observed_span_days": float(spans.median()),
            "field_seasons_with_observed_bbch51": int(bbch51_fs.sum()),
            "visits_with_explicit_absence_evidence": int(
                visits["explicit_late_blight_absent"].eq(True).sum()
            ),
            "visits_with_target_specific_absence_evidence": int(
                (
                    visits["zero_prevalence_late_blight"].eq(True)
                    | visits["target_explicit_absence_from_text"].eq(True)
                ).sum()
            ),
            "visits_reclassified_from_zero_prevalence": int(
                visits["zero_prevalence_late_blight"].eq(True).sum()
            ),
            "visits_with_generic_absence_evidence": int(
                (
                    visits["explicit_no_harmful_organisms"].eq(True)
                    | visits["generic_explicit_absence_from_text"].eq(True)
                ).sum()
            ),
            "positive_from_free_text_visits": int(
                visits["positive_late_blight_from_text"].eq(True).sum()
            ),
        }
    summary["observation_completeness"] = completeness

    hutton_summary: dict[str, object] = {}
    for mode in ("direct", "expanded"):
        mode_frame = hutton_results[hutton_results["source_mode"].eq(mode)]
        mode_summary: dict[str, object] = {}
        development_localized = mode_frame[
            mode_frame["season"].between(2015, 2022)
            & mode_frame["onset_interval_days"].le(21)
        ]
        for cohort_name in cohorts:
            cohort_frame = _cohort(mode_frame, cohort_name)
            localized = cohort_frame[cohort_frame["onset_interval_days"].le(21)]
            cohort_summary: dict[str, object] = {
                "all_events": int(len(cohort_frame)),
                "localized_events_le_21d": int(len(localized)),
                "events_with_any_prior_explicit_negative": int(
                    localized["explicit_negative_before_positive"].eq(True).sum()
                ),
                "events_with_any_prior_target_specific_negative": int(
                    localized["target_specific_negative_before_positive"]
                    .eq(True)
                    .sum()
                ),
                "events_with_explicit_interval_lower_bound": int(
                    localized["last_negative_is_explicit"].eq(True).sum()
                ),
                "events_with_target_specific_interval_lower_bound": int(
                    localized["last_negative_is_target_specific"].eq(True).sum()
                ),
                "unique_fields": int(localized["field_uid"].nunique()),
                "unique_municipalities": int(
                    localized["municipality"].astype(str).replace("", np.nan).nunique()
                ),
                "unique_parishes": int(
                    localized["parish"].astype(str).replace("", np.nan).nunique()
                ),
            }
            paired_p_values: list[float | None] = []
            permutation_p_values: list[float | None] = []
            for lookback in DEFAULT_LOOKBACKS:
                first_column = f"hutton_{lookback}d_first_positive"
                last_column = f"hutton_{lookback}d_last_negative"
                paired = localized[[first_column, last_column]].dropna()
                supportive = int(
                    (
                        paired[first_column].eq(True)
                        & paired[last_column].eq(False)
                    ).sum()
                )
                contradictory = int(
                    (
                        paired[first_column].eq(False)
                        & paired[last_column].eq(True)
                    ).sum()
                )
                paired_p = paired_one_sided_binomial_p(supportive, contradictory)
                permutation = hutton_date_permutation_test(
                    localized,
                    weather,
                    lookback,
                    repetitions=repetitions,
                )
                region_permutation = hutton_date_permutation_test(
                    localized,
                    weather,
                    lookback,
                    repetitions=repetitions,
                    strata=("season", "municipality"),
                )
                metric = {
                    "first_detection": _rate_summary(
                        localized, first_column
                    ),
                    "interval_possible": _rate_summary(
                        localized, f"hutton_{lookback}d_interval_possible"
                    ),
                    "interval_robust": _rate_summary(
                        localized, f"hutton_{lookback}d_interval_robust"
                    ),
                    "paired_supportive": supportive,
                    "paired_contradictory": contradictory,
                    "paired_scorable": int(len(paired)),
                    "paired_not_scorable": int(len(localized) - len(paired)),
                    "paired_one_sided_p": paired_p,
                    "within_season_date_permutation_null": permutation,
                    "within_municipality_season_date_permutation_null": (
                        region_permutation
                    ),
                    "alarm_burden_june_august": hutton_alarm_burden(
                        localized, weather, lookback
                    ),
                    "pre_detection_alarm_burden_june1": hutton_alarm_burden(
                        localized,
                        weather,
                        lookback,
                        end_column="first_positive_date",
                        end_inclusive=False,
                    ),
                    "pre_detection_alarm_burden_first_visit_proxy": (
                        hutton_alarm_burden(
                            localized,
                            weather,
                            lookback,
                            start_column="first_observation_date",
                            end_column="first_positive_date",
                            end_inclusive=False,
                        )
                    ),
                    "by_season_first_detection": _by_season_rate(
                        localized, first_column
                    ),
                }
                cohort_summary[f"lookback_{lookback}d"] = metric
                paired_p_values.append(paired_p)
                permutation_p_values.append(
                    permutation.get("permutation_p_one_sided")
                )
            for lookback, paired_adjusted, permutation_adjusted in zip(
                DEFAULT_LOOKBACKS,
                _holm_adjust(paired_p_values),
                _holm_adjust(permutation_p_values),
                strict=True,
            ):
                cohort_summary[f"lookback_{lookback}d"][
                    "paired_holm_adjusted_p_across_3_horizons"
                ] = paired_adjusted
                cohort_summary[f"lookback_{lookback}d"][
                    "permutation_holm_adjusted_p_across_3_horizons"
                ] = permutation_adjusted
            if cohort_name.startswith("holdout_2023_2025"):
                cohort_summary["development_fitted_calendar_baselines"] = (
                    calendar_baseline_summary(development_localized, localized)
                )
            mode_summary[cohort_name] = cohort_summary
        hutton_summary[mode] = mode_summary
    summary["hutton"] = hutton_summary

    polyakov_summary: dict[str, object] = {}
    for mode in ("direct", "expanded"):
        mode_frame = polyakov_results[polyakov_results["source_mode"].eq(mode)]
        mode_summary = {}
        for cohort_name in cohorts:
            cohort_frame = _cohort(mode_frame, cohort_name)
            interval_frame = cohort_frame[
                cohort_frame["polyakov_design"].eq(
                    "assumed_pre_bbch30_to_first_reproductive_sensitivity"
                )
            ]
            broad_sensitivity = interval_frame[
                interval_frame["phenophase_interval_days"].le(21)
                & interval_frame["onset_interval_days"].le(21)
            ]
            bbch51_interval_sensitivity = cohort_frame[
                cohort_frame["polyakov_design"].eq(
                    "assumed_pre_bbch30_to_bbch51_sensitivity"
                )
                & cohort_frame["phenophase_interval_days"].le(21)
                & cohort_frame["onset_interval_days"].le(21)
            ]
            point = cohort_frame[
                cohort_frame["polyakov_design"].eq("observed_bbch51_point_assumption")
                & cohort_frame["onset_interval_days"].le(21)
            ]
            cohort_summary = {}
            for case_name, cases in (
                ("observed_bbch51_operational_point_primary", point),
                (
                    "assumed_pre_bbch30_to_bbch51_interval_sensitivity",
                    bbch51_interval_sensitivity,
                ),
                (
                    "assumed_pre_bbch30_to_first_reproductive_sensitivity",
                    broad_sensitivity,
                ),
            ):
                endpoints: dict[str, object] = {}
                for endpoint in (
                    "activation_any__onset_any",
                    "activation_all__onset_any",
                    "activation_any__onset_all",
                    "activation_all__onset_all",
                ):
                    model_column = f"polyakov_{endpoint}"
                    baseline_column = f"phenology_only_{endpoint}"
                    endpoints[endpoint] = {
                        "polyakov": _rate_summary(cases, model_column),
                        "phenology_only": _rate_summary(cases, baseline_column),
                        "paired_vs_phenology_only": paired_binary_summary(
                            cases, model_column, baseline_column
                        ),
                    }
                cohort_summary[case_name] = {
                    "role": (
                        "primary_available_data_analysis"
                        if case_name == "observed_bbch51_operational_point_primary"
                        else "exploratory_sensitivity_not_valid_budding_bracket"
                    ),
                    "endpoints": endpoints,
                    "within_season_date_permutation_any_overlap": (
                        polyakov_date_permutation_test(
                            cases,
                            repetitions=repetitions,
                            endpoint="activation_any__onset_any",
                        )
                    ),
                    "within_season_date_permutation_full_onset_coverage": (
                        polyakov_date_permutation_test(
                            cases,
                            repetitions=repetitions,
                            endpoint="activation_all__onset_all",
                        )
                    ),
                    "mean_nominal_polyakov_window_days_issued_before_detection": (
                        float(cases["predicted_manifestation_day_count"].mean())
                        if not cases.empty
                        else None
                    ),
                    "mean_nominal_phenology_only_window_days": (
                        float(cases["phenology_only_predicted_day_count"].mean())
                        if not cases.empty
                        else None
                    ),
                    "operational_alarm_burden_before_detection": {
                        "at_risk_field_days": int(
                            cases["phenology_to_detection_at_risk_day_count"].sum()
                        ),
                        "polyakov_actionable_alarm_days": int(
                            cases[
                                "operational_alarm_day_count_before_detection"
                            ].sum()
                        ),
                        "phenology_only_alarm_days": int(
                            cases[
                                "phenology_only_alarm_day_count_before_detection"
                            ].sum()
                        ),
                        "polyakov_actionable_alarm_fraction": (
                            float(
                                cases[
                                    "operational_alarm_day_count_before_detection"
                                ].sum()
                                / cases[
                                    "phenology_to_detection_at_risk_day_count"
                                ].sum()
                            )
                            if not cases.empty
                            and cases[
                                "phenology_to_detection_at_risk_day_count"
                            ].sum()
                            else None
                        ),
                        "phenology_only_alarm_fraction": (
                            float(
                                cases[
                                    "phenology_only_alarm_day_count_before_detection"
                                ].sum()
                                / cases[
                                    "phenology_to_detection_at_risk_day_count"
                                ].sum()
                            )
                            if not cases.empty
                            and cases[
                                "phenology_to_detection_at_risk_day_count"
                            ].sum()
                            else None
                        ),
                        "issue_day_excluded_from_polyakov_actionable_days": True,
                        "detection_day_excluded_from_burden": True,
                    },
                    "weather_complete": _rate_summary(
                        cases, "polyakov_weather_complete"
                    ),
                    "operational_signal_lead_days": {
                        "n": int(
                            cases["nearest_operational_signal_lead_days"].notna().sum()
                        ),
                        "median": (
                            float(
                                cases["nearest_operational_signal_lead_days"]
                                .dropna()
                                .median()
                            )
                            if cases["nearest_operational_signal_lead_days"].notna().any()
                            else None
                        ),
                    },
                    "actionable_alarm_lead_days": {
                        "n": int(
                            cases["nearest_actionable_alarm_lead_days"]
                            .notna()
                            .sum()
                        ),
                        "median": (
                            float(
                                cases["nearest_actionable_alarm_lead_days"]
                                .dropna()
                                .median()
                            )
                            if cases["nearest_actionable_alarm_lead_days"]
                            .notna()
                            .any()
                            else None
                        ),
                    },
                }
            cohort_summary["strict_explicit_budding_absence_brackets"] = {
                "n": 0,
                "reason": "source_has_no_explicit_flower_buds_absent_field",
            }
            mode_summary[cohort_name] = cohort_summary
        polyakov_summary[mode] = mode_summary
    summary["polyakov"] = polyakov_summary

    negative_summary: dict[str, object] = {}
    for mode in ("direct", "expanded"):
        mode_frame = negative_results[negative_results["source_mode"].eq(mode)]
        mode_summary = {}
        for cohort_name in cohorts:
            cohort_frame = _cohort(mode_frame, cohort_name)
            cohort_payload: dict[str, object] = {
                "all_negative_only_seasons": int(len(cohort_frame)),
                "lenient_coverage_seasons": int(cohort_frame["lenient_coverage"].sum()),
                "dense_late_season_followup_seasons": int(
                    cohort_frame["dense_late_season_followup"].sum()
                ),
                "strict_full_risk_season_coverage_seasons": int(
                    cohort_frame["full_risk_season_coverage"].sum()
                ),
                "seasons_with_any_explicit_absence_evidence": int(
                    cohort_frame["explicit_negative_visit_count"].gt(0).sum()
                ),
                "seasons_with_any_target_specific_absence_evidence": int(
                    cohort_frame["target_specific_negative_visit_count"].gt(0).sum()
                ),
                "seasons_with_any_generic_absence_evidence": int(
                    cohort_frame["generic_explicit_negative_visit_count"].gt(0).sum()
                ),
                "strict_with_any_explicit_absence_evidence": int(
                    (
                        cohort_frame["strict_coverage"]
                        & cohort_frame["explicit_negative_visit_count"].gt(0)
                    ).sum()
                ),
                "strict_with_any_target_specific_absence_evidence": int(
                    (
                        cohort_frame["strict_coverage"]
                        & cohort_frame["target_specific_negative_visit_count"].gt(0)
                    ).sum()
                ),
                "strict_with_explicit_last_visit": int(
                    (
                        cohort_frame["strict_coverage"]
                        & cohort_frame["last_negative_is_explicit"]
                    ).sum()
                ),
                "strict_with_target_specific_last_visit": int(
                    (
                        cohort_frame["strict_coverage"]
                        & cohort_frame["last_negative_is_target_specific"]
                    ).sum()
                ),
                "strict_explicit_coverage_seasons": int(
                    cohort_frame["strict_explicit_coverage"].sum()
                ),
            }
            for lookback in DEFAULT_LOOKBACKS:
                cohort_payload[f"hutton_{lookback}d"] = {
                    "last_visit_all": _rate_summary(
                        cohort_frame, f"hutton_{lookback}d_last_negative"
                    ),
                    "last_visit_lenient": _rate_summary(
                        cohort_frame[cohort_frame["lenient_coverage"]],
                        f"hutton_{lookback}d_last_negative",
                    ),
                    "last_visit_dense_late_season_followup": _rate_summary(
                        cohort_frame[cohort_frame["dense_late_season_followup"]],
                        f"hutton_{lookback}d_last_negative",
                    ),
                    "last_visit_strict_full_risk_season": _rate_summary(
                        cohort_frame[cohort_frame["full_risk_season_coverage"]],
                        f"hutton_{lookback}d_last_negative",
                    ),
                    "observed_followup_alarm_burden": hutton_alarm_burden(
                        cohort_frame,
                        weather,
                        lookback,
                        start_column="first_negative_date",
                        end_column="last_negative_date",
                        end_inclusive=True,
                    ),
                }
            poly_evaluable = cohort_frame[
                cohort_frame["polyakov_point_evaluable"].eq(True)
            ]
            cohort_payload["polyakov_observed_bbch51_point"] = {
                "evaluable_seasons": int(len(poly_evaluable)),
                "not_evaluable_seasons": int(len(cohort_frame) - len(poly_evaluable)),
                "any_alert_before_censor": _rate_summary(
                    poly_evaluable, "polyakov_any_alert_before_censor"
                ),
                "phenology_only_any_alert_before_censor": _rate_summary(
                    poly_evaluable, "phenology_only_any_alert_before_censor"
                ),
                "median_alarm_day_fraction": (
                    float(poly_evaluable["polyakov_alarm_day_fraction"].median())
                    if not poly_evaluable.empty
                    else None
                ),
                "dense_late_season_followup": _rate_summary(
                    poly_evaluable[
                        poly_evaluable["dense_late_season_followup"]
                    ],
                    "polyakov_any_alert_before_censor",
                ),
                "strict_full_risk_season_coverage": _rate_summary(
                    poly_evaluable[poly_evaluable["full_risk_season_coverage"]],
                    "polyakov_any_alert_before_censor",
                ),
                "strict_explicit_coverage": _rate_summary(
                    poly_evaluable[poly_evaluable["strict_explicit_coverage"]],
                    "polyakov_any_alert_before_censor",
                ),
            }
            mode_summary[cohort_name] = cohort_payload
        negative_summary[mode] = mode_summary
    summary["negative_only_season_sensitivity_not_specificity"] = negative_summary
    return summary


def render_validation_report_ru(summary: dict[str, object]) -> str:
    """Render a location-free, aggregate report suitable for sharing."""

    def percentage(value: object, digits: int = 1) -> str:
        if value is None:
            return "—"
        return f"{100 * float(value):.{digits}f}%"

    def probability(value: object) -> str:
        if value is None:
            return "—"
        return f"{float(value):.3f}"

    def display_month_day(value: str) -> str:
        month, day = value.split("-", maxsplit=1)
        return f"{day}.{month}"

    direct_complete = summary["observation_completeness"]["direct"]
    expanded_complete = summary["observation_completeness"]["expanded"]
    hutton = summary["hutton"]["direct"][
        "holdout_2023_2025_unseen_fields"
    ]
    expanded_hutton_test = summary["hutton"]["expanded"][
        "holdout_2023_2025_unseen_fields"
    ]
    calendar = hutton["development_fitted_calendar_baselines"]
    calendar_minmax = calendar["development_min_max_detection_window"]
    calendar_fixed = calendar["fixed_july_august"]
    polyakov = summary["polyakov"]["direct"][
        "holdout_2023_2025_unseen_fields"
    ]["observed_bbch51_operational_point_primary"]
    polyakov_endpoint = polyakov["endpoints"]["activation_any__onset_any"]
    model_rate = polyakov_endpoint["polyakov"]
    baseline_rate = polyakov_endpoint["phenology_only"]
    paired = polyakov_endpoint["paired_vs_phenology_only"]
    permutation = polyakov["within_season_date_permutation_any_overlap"]
    polyakov_all = summary["polyakov"]["direct"]["all_2010_2025"][
        "observed_bbch51_operational_point_primary"
    ]
    polyakov_all_endpoint = polyakov_all["endpoints"][
        "activation_any__onset_any"
    ]
    polyakov_bbch51_interval = summary["polyakov"]["direct"]["all_2010_2025"][
        "assumed_pre_bbch30_to_bbch51_interval_sensitivity"
    ]["endpoints"]["activation_any__onset_any"]
    polyakov_first_reproductive_interval = summary["polyakov"]["direct"][
        "all_2010_2025"
    ]["assumed_pre_bbch30_to_first_reproductive_sensitivity"]["endpoints"][
        "activation_any__onset_any"
    ]
    negatives = summary["negative_only_season_sensitivity_not_specificity"]
    direct_negative = negatives["direct"]["holdout_2023_2025_unseen_fields"]
    expanded_negative_all = negatives["expanded"]["all_2010_2025"]
    direct_all = summary["hutton"]["direct"]["all_2010_2025"]
    expanded_all = summary["hutton"]["expanded"]["all_2010_2025"]
    weather = summary["weather"]
    annual_7d_rates = [
        value["rate"]
        for value in direct_all["lookback_7d"]["by_season_first_detection"]
        if value["rate"] is not None
    ]

    lines = [
        "# Максимально полная ретроспективная проверка Hutton и Полякова на VAAD",
        "",
        "## Главный вывод",
        "",
        (
            "На имеющихся данных Hutton часто предшествует первому "
            "зарегистрированному обнаружению фитофтороза, но не показывает "
            "убедимого выигрыша перед простым сезонным календарём. Высокая доля "
            "попаданий достигается ценой тревоги в большую часть сезона."
        ),
        (
            "Для модели Полякова пригодных наблюдений BBCH51 слишком мало. В "
            f"наиболее защищаемом срезе 2023–2025 осталось только {model_rate['n']} "
            f"наблюдения; погодная модель попала в {model_rate['successes']} из них, "
            "но такой размер выборки не "
            "позволяет ни подтвердить, ни надёжно отвергнуть модель."
        ),
        (
            "Это проверка временной ассоциации с первым зарегистрированным "
            "обнаружением, а не оценка биологической чувствительности, "
            "специфичности или точности живого прогноза."
        ),
        "",
        "## Данные и граница анализа",
        "",
        f"- Исходных строк: {summary['data_qc']['expanded']['input_rows']:,}.",
        (
            f"- Прямые геопривязки: {direct_complete['visits']} визитов, "
            f"{direct_complete['field_seasons']} поле-сезона; расширенный срез: "
            f"{expanded_complete['visits']} визита, "
            f"{expanded_complete['field_seasons']} поле-сезона."
        ),
        (
            f"- Для Hutton в срезе прямых геопривязок: "
            f"{direct_all['all_events']} событий с предшествующей "
            f"неположительной записью, из них {direct_all['localized_events_le_21d']} "
            "с интервалом до 21 дня."
        ),
        (
            f"- В расширенном срезе только "
            f"{expanded_all['events_with_explicit_interval_lower_bound']} из "
            f"{expanded_all['localized_events_le_21d']} интервалов имеет явно "
            "отрицательную последнюю границу; целевых явно отрицательных границ — "
            f"{expanded_all['events_with_target_specific_interval_lower_bound']}."
        ),
        (
            f"- Погода: {weather['daily_rows']:,} поле-дней, неопределённых дней "
            f"Hutton — {weather['hutton_indeterminate_days']}, неполных дней "
            f"Полякова — {weather['polyakov_incomplete_days']}."
        ),
        (
            f"- После консервативной проверки текста исправлены "
            f"{expanded_complete['positive_from_free_text_visits']} "
            "явно положительных записей и "
            f"{expanded_complete['visits_reclassified_from_zero_prevalence']} "
            "записи с распространённостью фитофтороза 0%. В расширенном наборе "
            f"{expanded_complete['visits_with_explicit_absence_evidence']} визитов "
            "с явным отрицательным свидетельством, но только "
            f"{expanded_complete['visits_with_target_specific_absence_evidence']} "
            "из них целевые; остальные формулируют общее отсутствие вредных организмов."
        ),
        (
            "- Структурированных полей о фунгицидных обработках в источнике нет."
        ),
        (
            f"- Сезоны после {summary['input']['maximum_complete_season']} года "
            "исключены как незавершённые для этого запуска."
        ),
        "",
        "## Hutton: временной тест 2023–2025 на полях, не встречавшихся в 2015–2022",
        "",
        (
            "Попадание здесь означает хотя бы один завершённый двухсуточный "
            "период Hutton среди предыдущих 7/14/21 календарных дней; погода дня "
            "осмотра не используется. Нагрузка — доля дней, для которых такое "
            "скользящее предупреждение активно."
        ),
        "",
        "| Окно | Попадания | 95% Wilson CI | Среднее календарного null | Разница | p permutation | Holm p | Весь июнь–август | 1 июня → обнаружение | Первый визит → обнаружение |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lookback in DEFAULT_LOOKBACKS:
        metric = hutton[f"lookback_{lookback}d"]
        hit = metric["first_detection"]
        null = metric["within_season_date_permutation_null"]
        burden = metric["alarm_burden_june_august"]
        june1_burden = metric["pre_detection_alarm_burden_june1"]
        first_visit_burden = metric[
            "pre_detection_alarm_burden_first_visit_proxy"
        ]
        lines.append(
            f"| {lookback} дней | {hit['successes']}/{hit['n']} "
            f"({percentage(hit['rate'])}) | "
            f"{percentage(hit['wilson_95_low'])}–{percentage(hit['wilson_95_high'])} | "
            f"{percentage(null['null_mean_rate'])} | "
            f"{percentage(null['lift'])} | "
            f"{probability(null['permutation_p_one_sided'])} | "
            f"{probability(metric['permutation_holm_adjusted_p_across_3_horizons'])} | "
            f"{percentage(burden['alarm_day_fraction'])} | "
            f"{percentage(june1_burden['alarm_day_fraction'])} | "
            f"{percentage(first_visit_burden['alarm_day_fraction'])} |"
        )
    lines.extend(
        [
            "",
            (
                "Ни один из трёх горизонтов не даёт статистически убедимого "
                "выигрыша после поправки на множественный выбор окна. "
                "Муниципалитет-сезонная перестановка почти неинформативна: "
                f"{hutton['lookback_7d']['within_municipality_season_date_permutation_null']['singleton_strata']} "
                f"из {hutton['lookback_7d']['within_municipality_season_date_permutation_null']['stratum_count']} "
                "страт содержат только одно событие."
            ),
            (
                "На всём прямом срезе 2015–2025 попадания составили "
                f"{direct_all['lookback_7d']['first_detection']['successes']}/"
                f"{direct_all['lookback_7d']['first_detection']['n']}, "
                f"{direct_all['lookback_14d']['first_detection']['successes']}/"
                f"{direct_all['lookback_14d']['first_detection']['n']} и "
                f"{direct_all['lookback_21d']['first_detection']['successes']}/"
                f"{direct_all['lookback_21d']['first_detection']['n']}. Для "
                f"7-дневного окна годовые доли менялись от "
                f"{percentage(min(annual_7d_rates))} до "
                f"{percentage(max(annual_7d_rates))}, то есть результат заметно "
                "зависит от сезона."
            ),
            (
                "Добавление однозначно восстановленных геолокаций не изменило "
                "временной тест ("
                + ", ".join(
                    f"{expanded_hutton_test[f'lookback_{lookback}d']['first_detection']['successes']}/"
                    f"{expanded_hutton_test[f'lookback_{lookback}d']['first_detection']['n']}"
                    for lookback in DEFAULT_LOOKBACKS
                )
                + "); на полном срезе "
                f"получено {expanded_all['localized_events_le_21d']} вместо "
                f"{direct_all['localized_events_le_21d']} событий."
            ),
            "",
            "### Простые календарные правила",
            "",
            "| Правило | Попадания | Весь июнь–август | 1 июня → обнаружение | Первый визит → обнаружение |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    calendar_labels = {
        "development_min_max_detection_window": "min–max дат development",
        "fixed_july_august": "фиксированное окно",
        "development_central_90pct_detection_window": "центральные 90% дат development",
    }
    for key, label in calendar_labels.items():
        metric = calendar[key]
        dated_label = (
            f"{label}: {display_month_day(metric['start_month_day'])}–"
            f"{display_month_day(metric['end_month_day'])}"
        )
        lines.append(
            f"| {dated_label} | {metric['hits']}/{metric['n']} "
            f"({percentage(metric['hit_rate'])}) | "
            f"{percentage(metric['june_august_alarm_day_fraction'])} | "
            f"{percentage(metric['pre_detection_alarm_day_fraction_june1'])} | "
            f"{percentage(metric['pre_detection_alarm_day_fraction_first_visit_proxy'])} |"
        )

    lines.extend(
        [
            "",
            (
                f"Календарь {display_month_day(calendar_minmax['start_month_day'])}–"
                f"{display_month_day(calendar_minmax['end_month_day'])} дал "
                f"{calendar_minmax['hits']}/"
                f"{calendar_minmax['n']} попаданий при "
                f"{percentage(calendar_minmax['june_august_alarm_day_fraction'])} "
                "тревожных дней; для 7-дневного Hutton соответствующие значения — "
                f"{hutton['lookback_7d']['first_detection']['successes']}/"
                f"{hutton['lookback_7d']['first_detection']['n']} и "
                f"{percentage(hutton['lookback_7d']['alarm_burden_june_august']['alarm_day_fraction'])}. "
                f"Фиксированное окно "
                f"{display_month_day(calendar_fixed['start_month_day'])}–"
                f"{display_month_day(calendar_fixed['end_month_day'])} дало "
                f"{calendar_fixed['hits']}/{calendar_fixed['n']} при "
                f"{percentage(calendar_fixed['june_august_alarm_day_fraction'])} "
                "тревожных дней; Hutton 14/21 дней дал "
                f"{hutton['lookback_14d']['first_detection']['successes']}/"
                f"{hutton['lookback_14d']['first_detection']['n']} и "
                f"{hutton['lookback_21d']['first_detection']['successes']}/"
                f"{hutton['lookback_21d']['first_detection']['n']} при нагрузке "
                f"{percentage(hutton['lookback_14d']['alarm_burden_june_august']['alarm_day_fraction'])} "
                "и "
                f"{percentage(hutton['lookback_21d']['alarm_burden_june_august']['alarm_day_fraction'])}."
            ),
            "",
            "### Пространственная устойчивость",
            "",
        ]
    )
    spatial_labels = (
        ("holdout_2023_2025_spatial_gt_1km", ">1 км"),
        ("holdout_2023_2025_spatial_gt_5km", ">5 км"),
        ("holdout_2023_2025_spatial_gt_10km", ">10 км"),
        ("holdout_2023_2025_new_municipality", "новый муниципалитет"),
        ("holdout_2023_2025_new_parish", "новый приход"),
    )
    spatial_items = []
    for cohort_name, label in spatial_labels:
        cohort = summary["hutton"]["direct"][cohort_name]
        spatial_items.append(f"{label}: n={cohort['localized_events_le_21d']}")
    lines.append("Пригодные размеры срезов — " + "; ".join(spatial_items) + ".")
    lines.append(
        "Поэтому высокие проценты в удалённых срезах нельзя считать внешней пространственной валидацией."
    )

    lines.extend(
        [
            "",
            "## Поляков: наблюдавшаяся BBCH51 как точка активации",
            "",
            (
                "Phenology-only baseline — заранее фиксированное трёхдневное "
                "окно BBCH51+15…17 без погодного фильтра; оно не подбиралось по "
                "исходам и служит ablation-сравнением."
            ),
            (
                f"В основном временном тесте доступно только {model_rate['n']} "
                f"случая: Поляков — {model_rate['successes']}/{model_rate['n']}, "
                f"фенологический baseline без погоды — "
                f"{baseline_rate['successes']}/{baseline_rate['n']}."
            ),
            (
                f"Парная разница: {percentage(paired['absolute_lift'])}; "
                f"односторонний McNemar p={probability(paired['mcnemar_p_one_sided_model_better'])}. "
                f"Within-season null: {percentage(permutation['null_mean_rate'])}, "
                f"p={probability(permutation['permutation_p_one_sided'])}."
            ),
            (
                f"95% Wilson-интервал для Полякова в тесте: "
                f"{percentage(model_rate['wilson_95_low'])}–"
                f"{percentage(model_rate['wilson_95_high'])}. Три случая лежат "
                f"только в {permutation['season_count']} сезонах, из которых "
                f"{permutation['singleton_seasons']} singleton; поскольку модель "
                "не сформировала ни одной тревоги, permutation null вырожден и "
                "p=1 здесь не является свидетельством эквивалентности."
            ),
            (
                f"На всех {polyakov_all_endpoint['polyakov']['n']} пригодных прямых случаях "
                f"2015–2025 Поляков дал {polyakov_all_endpoint['polyakov']['successes']}/"
                f"{polyakov_all_endpoint['polyakov']['n']} попаданий против "
                f"{polyakov_all_endpoint['phenology_only']['successes']}/"
                f"{polyakov_all_endpoint['phenology_only']['n']} у фенологического baseline; "
                f"95% Wilson CI: {percentage(polyakov_all_endpoint['polyakov']['wilson_95_low'])}–"
                f"{percentage(polyakov_all_endpoint['polyakov']['wilson_95_high'])} и "
                f"{percentage(polyakov_all_endpoint['phenology_only']['wilson_95_low'])}–"
                f"{percentage(polyakov_all_endpoint['phenology_only']['wilson_95_high'])} "
                "соответственно. Это не "
                "подтверждает добавочную ценность погодного фильтра."
            ),
            (
                "Expanded-геолокации не добавили ни одного нового пригодного "
                "случая BBCH51: primary-срез остался теми же 19 наблюдениями."
            ),
            (
                "До дня обнаружения погодный фильтр был уже: "
                f"{polyakov_all['operational_alarm_burden_before_detection']['polyakov_actionable_alarm_days']}/"
                f"{polyakov_all['operational_alarm_burden_before_detection']['at_risk_field_days']} "
                "доступных тревожных поле-дней против "
                f"{polyakov_all['operational_alarm_burden_before_detection']['phenology_only_alarm_days']}/"
                f"{polyakov_all['operational_alarm_burden_before_detection']['at_risk_field_days']} "
                "у baseline. День выпуска после закрытия суток и день обнаружения "
                "в нагрузку не включены. Без надёжных отрицательных сезонов нельзя "
                "определить, компенсирует ли меньшая тревожная нагрузка пропущенные события."
            ),
            (
                f"Среди {polyakov_all['actionable_alarm_lead_days']['n']} случаев, "
                "где погодная тревога стала доступна до обнаружения, медиана расстояния до "
                f"первого зарегистрированного обнаружения составила "
                f"{polyakov_all['actionable_alarm_lead_days']['median']:.1f} дня. "
                "Это условная ретроспективная величина по ERA5, не проверка live-прогноза."
            ),
            (
                "При требовании покрыть весь неопределённый интервал onset, а не "
                "хотя бы один его день, получено "
                f"{polyakov_all['endpoints']['activation_all__onset_all']['polyakov']['successes']}/"
                f"{polyakov_all['endpoints']['activation_all__onset_all']['polyakov']['n']}. "
                "Этот консервативный endpoint главным образом показывает грубость "
                "интервального цензурирования."
            ),
            (
                "Строго установить интервал начала бутонизации нельзя: в VAAD "
                "нет отдельного признака «бутоны отсутствуют». BBCH31–39 не "
                "использовались как такая граница; варианты с BBCH0–29 оставлены "
                "только как слабый анализ чувствительности."
            ),
            (
                "В этих слабых interval-срезах результаты также не лучше baseline: "
                f"до BBCH51 — {polyakov_bbch51_interval['polyakov']['successes']}/"
                f"{polyakov_bbch51_interval['polyakov']['n']} против "
                f"{polyakov_bbch51_interval['phenology_only']['successes']}/"
                f"{polyakov_bbch51_interval['phenology_only']['n']}; до первой "
                "репродуктивной стадии — "
                f"{polyakov_first_reproductive_interval['polyakov']['successes']}/"
                f"{polyakov_first_reproductive_interval['polyakov']['n']} против "
                f"{polyakov_first_reproductive_interval['phenology_only']['successes']}/"
                f"{polyakov_first_reproductive_interval['phenology_only']['n']}."
            ),
            (
                "Температура/влажность и осадки для Полякова взяты из двух "
                "реанализов. Расстояние между возвращёнными центрами их сеток: "
                f"медиана {weather['era5_land_to_era5_grid_distance_km']['median']:.1f} км, "
                f"p95 {weather['era5_land_to_era5_grid_distance_km']['p95']:.1f} км, "
                f"максимум {weather['era5_land_to_era5_grid_distance_km']['maximum']:.1f} км; "
                f"p95 означает, что у 95% из "
                f"{weather['source_request_metadata_rows']} сопоставленных "
                "запросов «координата–сезон» расстояние "
                "не превышает это значение; это не доверительный интервал и не "
                "ошибка координаты поля. Это дополнительная пространственная "
                "неопределённость."
            ),
            "",
            "## Что дают записи без зарегистрированного фитофтороза",
            "",
            (
                f"В расширенном наборе есть {expanded_negative_all['all_negative_only_seasons']} "
                "negative-only сезонов. Плотное позднесезонное наблюдение есть у "
                f"{expanded_negative_all['dense_late_season_followup_seasons']}, "
                "но полное покрытие риска с середины июня — у "
                f"{expanded_negative_all['strict_full_risk_season_coverage_seasons']}; "
                "повторных полносезонных последовательностей явных отрицательных "
                "подтверждений также нет."
            ),
            (
                f"Во временном тесте с прямой геопривязкой есть "
                f"{direct_negative['all_negative_only_seasons']} таких сезонов, "
                "с плотным поздним наблюдением — "
                f"{direct_negative['dense_late_season_followup_seasons']}, с полным "
                "покрытием риска — "
                f"{direct_negative['strict_full_risk_season_coverage_seasons']}; "
                "с любым целевым явно отрицательным свидетельством — "
                f"{direct_negative['seasons_with_any_target_specific_absence_evidence']}."
            ),
            (
                f"Даже в этих сезонах Hutton срабатывал к последнему визиту в "
                f"{direct_negative['hutton_7d']['last_visit_all']['successes']}/"
                f"{direct_negative['hutton_7d']['last_visit_all']['n']}, "
                f"{direct_negative['hutton_14d']['last_visit_all']['successes']}/"
                f"{direct_negative['hutton_14d']['last_visit_all']['n']} и "
                f"{direct_negative['hutton_21d']['last_visit_all']['successes']}/"
                f"{direct_negative['hutton_21d']['last_visit_all']['n']} сезонов "
                "для окон 7/14/21 день. Это показывает низкую селективность "
                "сигнала, но не является оценкой false-positive rate: болезнь "
                "могли не искать, не записать или подавить обработкой."
            ),
            (
                "Для Полякова число negative-only сезонов с наблюдавшейся "
                "BBCH51 и достаточным последующим отрезком — "
                f"{expanded_negative_all['polyakov_observed_bbch51_point']['evaluable_seasons']}; "
                "во временном "
                "тесте таких сезонов — "
                f"{direct_negative['polyakov_observed_bbch51_point']['evaluable_seasons']}. "
                "Поэтому даже анализ чувствительности к "
                "specificity для него практически отсутствует."
            ),
            "",
            "## Что подтверждено и что не подтверждено",
            "",
            "Подтверждено:",
            "",
            "- расчётный контур воспроизводимо связывает визиты, локальные календарные дни и погодные правила;",
            "- Hutton часто активен перед зарегистрированными обнаружениями;",
            "- результат устойчив к добавлению однозначно восстановленных геолокаций;",
            "- техническая полнота реанализа достаточна для расчёта текущих индикаторов.",
            "",
            "Не подтверждено:",
            "",
            "- что Hutton информативнее простого календаря;",
            "- что Поляков добавляет ценность к одной только фенофазе;",
            "- specificity, PPV/precision, NPV и false-positive rate;",
            "- переносимость на новые регионы и реальные оперативные прогнозы;",
            "- эффект без учёта фунгицидов, сорта, орошения и источника инфекции.",
            "",
            "## Что в идеале нужно собрать для максимально точной проверки",
            "",
            "1. Проспективный, заранее зарегистрированный протокол с замороженными правилами и отдельным нетронутым внешним тестом.",
            "2. Одни и те же поля в нескольких регионах минимум 3 полных сезона, лучше 5, с устойчивыми ID и полигонами полей.",
            "3. Плановые осмотры каждые 2–3 дня около ожидаемого начала болезни и не реже раза в 3–7 дней остальную часть риска.",
            "4. На каждом визите отдельная целевая метка: обнаружен / специально искали и не обнаружен / не оценивался; интенсивность поражения и лабораторное подтверждение хотя бы подвыборки.",
            "5. Полная фенология: явное «бутоны отсутствуют», первая BBCH51, посадка, всходы и последующие BBCH, а не вывод об отсутствии бутонов из BBCH31–39.",
            "6. Все фунгицидные обработки, сорт и устойчивость, орошение, дата посадки/всходов, удаление ботвы, уборка и сведения об источнике инфекции.",
            "7. Полевые или ближайшие проверенные почасовые датчики температуры, влажности, осадков и желательно смачивания листа; единый пространственный источник переменных.",
            "8. Для проверки именно прогноза — архивные прогнозные выпуски с issued_at, а не только ERA5-реанализ, доступный задним числом.",
            "9. Предварительный ориентир: порядка 150 положительных и 300–400 полноценно прослеженных отрицательных/censored поле-сезонов (450–550 до запаса на потери; практически целиться в 500–700). Это не готовый расчёт мощности: effect size, кластерный design effect и число независимых region-year кластеров нужно зафиксировать после пилота.",
            "",
            "## Воспроизводимость и приватность",
            "",
            f"- SHA-256 входного CSV: `{summary['input']['sha256']}`.",
            f"- Перестановок: {summary['execution']['permutation_repetitions']:,}; seed: {summary['execution']['random_seed']}.",
            "- Этот Markdown и summary.json агрегированы. Файлы hutton_events.csv, polyakov_events.csv, negative_seasons.csv и weather_metadata.csv содержат точные координаты и/или идентификаторы полей и не предназначены для публикации.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_vaad_validation(
    csv_path: str | Path,
    output_dir: str | Path = "results/vaad_late_blight",
    cache_dir: str | Path = "data/cache/vaad_open_meteo",
    maximum_season: int = 2025,
    batch_size: int = 20,
    repetitions: int = 2000,
    progress: Callable[[str], None] | None = print,
) -> dict[str, object]:
    direct_visits, direct_qc = prepare_potato_visits(csv_path, "direct")
    expanded_visits, expanded_qc = prepare_potato_visits(csv_path, "expanded")
    direct_visits = direct_visits[direct_visits["season"].le(maximum_season)].copy()
    expanded_visits = expanded_visits[
        expanded_visits["season"].le(maximum_season)
    ].copy()
    for visits, qc_payload in (
        (direct_visits, direct_qc),
        (expanded_visits, expanded_qc),
    ):
        pre_cutoff_visits = int(qc_payload["analysis_visits"])
        qc_payload["excluded_visits_after_maximum_season"] = int(
            pre_cutoff_visits - len(visits)
        )
        qc_payload["label_corrections_before_maximum_season_filter"] = {
            "zero_prevalence_as_target_absence": int(
                qc_payload.pop(
                    "reclassified_zero_prevalence_as_explicit_target_negative"
                )
            ),
            "generic_absence_text": int(
                qc_payload.pop("generic_explicit_absence_text_rows")
            ),
            "target_specific_absence_text": int(
                qc_payload.pop("target_explicit_absence_text_rows")
            ),
            "positive_free_text": int(
                qc_payload.pop("reclassified_positive_free_text_rows")
            ),
        }
        qc_payload["label_corrections_in_analysis_visits"] = {
            "zero_prevalence_as_target_absence": int(
                visits["zero_prevalence_late_blight"].eq(True).sum()
            ),
            "generic_absence_text": int(
                visits["generic_explicit_absence_from_text"].eq(True).sum()
            ),
            "target_specific_absence_text": int(
                visits["target_explicit_absence_from_text"].eq(True).sum()
            ),
            "positive_free_text": int(
                visits["positive_late_blight_from_text"].eq(True).sum()
            ),
        }
        qc_payload["maximum_complete_season"] = int(maximum_season)
        qc_payload["analysis_visits"] = int(len(visits))
        qc_payload["analysis_field_seasons"] = int(
            visits[["field_uid", "season"]].drop_duplicates().shape[0]
        )
    weather, metadata = fetch_weather_daily(
        expanded_visits,
        cache_dir=cache_dir,
        batch_size=batch_size,
        maximum_season=maximum_season,
        progress=progress,
    )

    hutton_tables: list[pd.DataFrame] = []
    negative_tables: list[pd.DataFrame] = []
    polyakov_tables: list[pd.DataFrame] = []
    visit_tables: list[pd.DataFrame] = []
    for mode, visits in (("direct", direct_visits), ("expanded", expanded_visits)):
        events = build_onset_events(visits)
        negative_seasons = build_negative_seasons(visits)
        polyakov_cases = build_polyakov_cases(visits, events)
        polyakov_exact_interval_cases = build_polyakov_exact_interval_cases(
            visits, events
        )
        polyakov_point_cases = build_polyakov_point_cases(visits, events)
        hutton = evaluate_hutton_events(events, weather).assign(source_mode=mode)
        negative = evaluate_negative_seasons(
            negative_seasons, visits, weather
        )
        negative = evaluate_polyakov_negative_seasons(
            negative, visits, weather
        ).assign(source_mode=mode)
        polyakov_interval = evaluate_polyakov_cases(
            polyakov_cases, weather
        ).assign(
            source_mode=mode,
            polyakov_design="assumed_pre_bbch30_to_first_reproductive_sensitivity",
        )
        polyakov_exact_interval = evaluate_polyakov_cases(
            polyakov_exact_interval_cases, weather
        ).assign(
            source_mode=mode,
            polyakov_design="assumed_pre_bbch30_to_bbch51_sensitivity",
        )
        polyakov_point = evaluate_polyakov_cases(
            polyakov_point_cases, weather
        ).assign(
            source_mode=mode,
            polyakov_design="observed_bbch51_point_assumption",
        )
        polyakov = pd.concat(
            [polyakov_interval, polyakov_exact_interval, polyakov_point],
            ignore_index=True,
        )
        hutton = annotate_holdout_structure(hutton, visits)
        negative = annotate_holdout_structure(negative, visits)
        polyakov = annotate_holdout_structure(polyakov, visits)
        hutton_tables.append(hutton)
        negative_tables.append(negative)
        polyakov_tables.append(polyakov)
        visit_tables.append(visits.assign(source_mode=mode))

    hutton_results = pd.concat(hutton_tables, ignore_index=True)
    negative_results = pd.concat(negative_tables, ignore_index=True)
    polyakov_results = pd.concat(polyakov_tables, ignore_index=True)
    visits_results = pd.concat(visit_tables, ignore_index=True)
    summary = build_summary(
        {"direct": direct_qc, "expanded": expanded_qc},
        hutton_results,
        negative_results,
        polyakov_results,
        weather,
        visits_results,
        repetitions=repetitions,
    )
    input_path = Path(csv_path)
    digest = hashlib.sha256()
    with input_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    summary["input"] = {
        "filename": input_path.name,
        "sha256": digest.hexdigest(),
        "maximum_complete_season": maximum_season,
    }
    source_dir = Path(__file__).parent
    summary["execution"] = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "vaad_validation_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "analysis_source_sha256": {
            filename: hashlib.sha256((source_dir / filename).read_bytes()).hexdigest()
            for filename in (
                "vaad_validation.py",
                "late_blight.py",
                "open_meteo.py",
            )
        },
        "permutation_repetitions": int(repetitions),
        "random_seed": DEFAULT_SEED,
    }
    summary["model_contract"] = {
        "hutton": {
            "daily_rule": "Tmin >= 10 C and at least 6 hourly intervals with RH >= 90%",
            "period_rule": "two consecutive local calendar days",
            "observation_day_excluded": True,
            "lookback_days": list(DEFAULT_LOOKBACKS),
            "weather_source": "Open-Meteo ERA5-Land",
            "model_version": "1.0.1",
            "period_missing_data_rule": "indeterminate_dominates_fail",
        },
        "polyakov": {
            "window_days": 10,
            "temperature_mean_range_c": [13, 20],
            "relative_humidity_mean_min_pct": 75,
            "precipitation_sum_min_mm": 20,
            "continuous_recalculations_through_t0_plus_days": 6,
            "manifestation_lag_days": [6, 8],
            "weather_sources": {
                "temperature_relative_humidity": "Open-Meteo ERA5-Land",
                "precipitation": "Open-Meteo ERA5",
            },
            "interpolation": False,
            "model_version": PolyakovConfig().model_version,
            "operational_signal_available_on": "t0_plus_6_after_daily_data_close",
            "first_actionable_full_day": "t0_plus_7",
            "alarm_burden_detection_day_excluded": True,
            "phenology_only_baseline": {
                "rule": "fixed BBCH51 plus 15 through 17 days",
                "weather_filter": False,
                "prespecified_ablation": True,
            },
        },
    }

    if not metadata.empty:
        summary["weather"]["source_request_metadata_rows"] = int(len(metadata))
        grid_distances = [
            float(
                _haversine_km(
                    float(row.era5_land_returned_latitude),
                    float(row.era5_land_returned_longitude),
                    np.asarray([float(row.era5_returned_latitude)]),
                    np.asarray([float(row.era5_returned_longitude)]),
                )[0]
            )
            for row in metadata.itertuples(index=False)
        ]
        summary["weather"]["era5_land_to_era5_grid_distance_km"] = {
            "median": float(np.median(grid_distances)),
            "p95": float(np.quantile(grid_distances, 0.95)),
            "maximum": float(np.max(grid_distances)),
        }
        retrieval_times = sorted(
            {
                str(value)
                for column in ("era5_land_retrieved_at", "era5_retrieved_at")
                for value in metadata[column].dropna()
            }
        )
        summary["weather"]["source_retrieval_times_utc"] = retrieval_times

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if metadata.empty:
        summary["weather"]["source_request_metadata_rows"] = 0
    hutton_results.to_csv(output / "hutton_events.csv", index=False)
    polyakov_results.to_csv(output / "polyakov_events.csv", index=False)
    negative_results.to_csv(output / "negative_seasons.csv", index=False)
    metadata.to_csv(output / "weather_metadata.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            default=str,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (output / "report_ru.md").write_text(
        render_validation_report_ru(summary), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--output-dir", default="results/vaad_late_blight")
    parser.add_argument("--cache-dir", default="data/cache/vaad_open_meteo")
    parser.add_argument("--maximum-season", type=int, default=2025)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args()
    summary = run_vaad_validation(
        args.csv_path,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        maximum_season=args.maximum_season,
        batch_size=args.batch_size,
        repetitions=args.permutations,
    )
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            default=str,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
