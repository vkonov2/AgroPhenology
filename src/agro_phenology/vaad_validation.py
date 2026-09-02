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
) -> tuple[pd.DataFrame, dict[str, int]]:
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

    zero_prevalence = potato["detected_organisms"].map(_late_blight_has_zero_prevalence)
    excluded_zero_prevalence = int(zero_prevalence.sum())
    potato = potato[~zero_prevalence].copy()

    rows: list[dict[str, object]] = []
    for (field_uid, observation_date), group in potato.groupby(
        ["field_uid", "observation_date"], sort=True, dropna=False
    ):
        coordinates = group[["latitude", "longitude"]].drop_duplicates()
        if len(coordinates) != 1:
            raise ValueError(
                f"Field/date {field_uid}/{observation_date.date()} has inconsistent coordinates"
            )
        outcomes = [_bool_value(value) for value in group["late_blight_detected"]]
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
        "excluded_zero_prevalence_positive": excluded_zero_prevalence,
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
            prior_negatives["explicit_no_harmful_organisms"].eq(True)
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
                "first_positive_date": first_positive["observation_date"],
                "onset_interval_days": int(
                    (first_positive["observation_date"] - last_negative["observation_date"]).days
                ),
                "explicit_negative_before_positive": bool(not explicit_prior.empty),
                "last_explicit_negative_date": (
                    explicit_prior.iloc[-1]["observation_date"]
                    if not explicit_prior.empty
                    else pd.NaT
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
        rows.append(
            {
                "field_uid": field_uid,
                "season": int(season),
                "latitude": float(group.iloc[0]["latitude"]),
                "longitude": float(group.iloc[0]["longitude"]),
                "negative_visit_count": int(len(negatives)),
                "explicit_negative_visit_count": int(
                    negatives["explicit_no_harmful_organisms"].eq(True).sum()
                ),
                "first_negative_date": first_date,
                "last_negative_date": last_date,
                "followup_span_days": int((last_date - first_date).days),
                "maximum_visit_gap_days": maximum_gap,
                "lenient_coverage": bool(len(negatives) >= 3 and last_date.month >= 8),
                "strict_coverage": bool(
                    len(negatives) >= 3
                    and last_date >= date(int(season), 8, 15)
                    and pd.notna(maximum_gap)
                    and int(maximum_gap) <= 14
                ),
            }
        )
    return pd.DataFrame(rows)


def _is_definite_pre_budding(stages: object) -> bool:
    values = tuple(stages) if isinstance(stages, (tuple, list)) else ()
    return bool(values) and all(0 <= int(value) <= 39 for value in values)


def _has_reproductive_stage(stages: object) -> bool:
    values = tuple(stages) if isinstance(stages, (tuple, list)) else ()
    return any(51 <= int(value) <= 89 for value in values)


def build_polyakov_cases(visits: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Pair disease-onset intervals with defensible field phenophase intervals.

    Codes 41--49 are a parallel potato tuber-development branch and are not used
    as evidence that budding had not yet started.
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
            & group["growth_stage_codes"].map(_is_definite_pre_budding)
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
    """Bracket an observed BBCH 51 visit by the prior definite vegetative visit."""

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
            & group["growth_stage_codes"].map(_is_definite_pre_budding)
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
    ) + "\nvaad_weather_daily_v1"
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
                pass_periods = set(
                    hutton_periods.loc[
                        hutton_periods["period_status"].eq("pass"), "period_end"
                    ]
                )
                indeterminate_periods = set(
                    hutton_periods.loc[
                        hutton_periods["period_status"].eq("indeterminate"), "period_end"
                    ]
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
                    lambda value: (
                        "pass"
                        if value in pass_periods
                        else "indeterminate"
                        if value in indeterminate_periods
                        else "fail"
                    )
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


def _recent_hutton_signal(weather: pd.DataFrame, target: date, lookback_days: int) -> bool:
    return bool(
        weather[
            (weather["date"] >= target - timedelta(days=lookback_days))
            & (weather["date"] <= target - timedelta(days=1))
        ]["hutton_period_status"].eq("pass").any()
    )


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
        if field_weather.empty:
            continue
        result = event._asdict()
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
            result[f"hutton_{lookback}d_interval_possible"] = any(candidate_hits)
            result[f"hutton_{lookback}d_interval_robust"] = all(candidate_hits)
            result[f"hutton_{lookback}d_interval_hit_fraction"] = float(
                np.mean(candidate_hits)
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
            result[f"hutton_{lookback}d_any_negative_visit"] = any(visit_hits)
            result[f"hutton_{lookback}d_negative_visit_fraction"] = float(
                np.mean(visit_hits)
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
        scenario_hits: list[bool] = []
        phenology_baseline_hits: list[bool] = []
        predicted_union: set[date] = set()
        scenario_windows: list[list[tuple[date, date]]] = []
        for activation_date in activation_dates:
            classified = classify_polyakov_windows(daily, activation_date, config)
            episodes = extract_polyakov_episodes(classified, config)
            confirmed = episodes[episodes["persistence_confirmed"].eq(True)]
            windows = [
                (
                    row.expected_manifestation_start,
                    row.expected_manifestation_end,
                )
                for row in confirmed.itertuples(index=False)
            ]
            scenario_windows.append(windows)
            for start, end in windows:
                predicted_union.update(
                    value.date() for value in pd.date_range(start, end, freq="D")
                )
            scenario_hits.append(
                any(
                    _date_interval_overlaps(
                        start,
                        end,
                        case.last_negative_date,
                        case.first_positive_date,
                    )
                    for start, end in windows
                )
            )
            baseline_start = activation_date + timedelta(
                days=config.window_days - 1 + config.manifestation_lag_days[0]
            )
            baseline_end = activation_date + timedelta(
                days=config.window_days - 1 + config.manifestation_lag_days[1]
            )
            phenology_baseline_hits.append(
                _date_interval_overlaps(
                    baseline_start,
                    baseline_end,
                    case.last_negative_date,
                    case.first_positive_date,
                )
            )
        result = case._asdict()
        result["activation_scenario_count"] = len(activation_dates)
        result["polyakov_interval_possible"] = bool(any(scenario_hits))
        result["polyakov_interval_robust"] = bool(all(scenario_hits))
        result["polyakov_scenario_hit_fraction"] = float(np.mean(scenario_hits))
        result["polyakov_bbch_upper_bound_hit"] = bool(scenario_hits[-1])
        result["phenology_only_interval_possible"] = bool(any(phenology_baseline_hits))
        result["phenology_only_interval_robust"] = bool(all(phenology_baseline_hits))
        result["phenology_only_bbch_upper_bound_hit"] = bool(
            phenology_baseline_hits[-1]
        )
        result["predicted_manifestation_dates_union"] = "|".join(
            value.isoformat() for value in sorted(predicted_union)
        )
        result["predicted_manifestation_day_count"] = len(predicted_union)
        result["polyakov_weather_complete"] = bool(daily["accepted"].all())
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


def hutton_date_permutation_test(
    results: pd.DataFrame,
    weather: pd.DataFrame,
    lookback: int,
    repetitions: int = 2000,
    seed: int = DEFAULT_SEED,
) -> dict[str, float | int | None]:
    """Shuffle observed first-detection dates among fields within each season year."""

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
    observed = float(results[f"hutton_{lookback}d_first_positive"].mean())
    rng = np.random.default_rng(seed + lookback)
    grouped = [group.copy() for _, group in results.groupby("season", sort=True)]
    signal_matrices: list[np.ndarray] = []
    for group in grouped:
        assigned_dates = group["first_positive_date"].tolist()
        group_rows = list(group.itertuples(index=False))
        matrix = np.zeros((len(group_rows), len(assigned_dates)), dtype=np.int8)
        for row_index, row in enumerate(group_rows):
            field_weather = lookup[(row.field_uid, int(row.season))]
            for date_index, assigned_date in enumerate(assigned_dates):
                matrix[row_index, date_index] = int(
                    _recent_hutton_signal(field_weather, assigned_date, lookback)
                )
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
    }


def hutton_alarm_burden(
    results: pd.DataFrame,
    weather: pd.DataFrame,
    lookback: int,
) -> dict[str, float | int | None]:
    """Fraction of June--August field-days covered by a recent Hutton period."""

    if results.empty:
        return {
            "field_seasons": 0,
            "field_days": 0,
            "alarm_day_fraction": None,
            "median_field_alarm_fraction": None,
        }
    keys = results[["field_uid", "season"]].drop_duplicates()
    frame = weather.merge(
        keys, on=["field_uid", "season"], how="inner", validate="many_to_many"
    ).copy()
    timestamps = pd.to_datetime(frame["date"])
    frame = frame[timestamps.dt.month.between(6, 8)].sort_values(
        ["field_uid", "season", "date"]
    )
    frame["alarm"] = frame.groupby(["field_uid", "season"], sort=False)[
        "hutton_period_status"
    ].transform(
        lambda values: values.eq("pass")
        .shift(1)
        .rolling(lookback, min_periods=1)
        .max()
        .fillna(0)
        .astype(bool)
    )
    by_field = frame.groupby(["field_uid", "season"])["alarm"].mean()
    return {
        "field_seasons": int(len(by_field)),
        "field_days": int(len(frame)),
        "alarm_day_fraction": float(frame["alarm"].mean()),
        "median_field_alarm_fraction": float(by_field.median()),
    }


def polyakov_date_permutation_test(
    results: pd.DataFrame,
    repetitions: int = 2000,
    seed: int = DEFAULT_SEED,
) -> dict[str, float | int | None]:
    """Shuffle complete disease intervals among cases in the supplied cohort."""

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
    predicted = [
        {date.fromisoformat(value) for value in str(text).split("|") if value}
        for text in results["predicted_manifestation_dates_union"]
    ]
    intervals = list(
        zip(
            results["last_negative_date"].tolist(),
            results["first_positive_date"].tolist(),
            strict=True,
        )
    )

    def hit(days: set[date], interval: tuple[date, date]) -> bool:
        start, end = interval
        return any(start < value <= end for value in days)

    hit_matrix = np.asarray(
        [[hit(days, interval) for interval in intervals] for days in predicted],
        dtype=np.int8,
    )
    observed = float(np.diag(hit_matrix).mean())
    rng = np.random.default_rng(seed + len(results))
    random_order = np.argsort(rng.random((repetitions, len(intervals))), axis=1)
    row_indices = np.arange(len(intervals))[None, :]
    values = hit_matrix[row_indices, random_order].mean(axis=1)
    return {
        "observed_rate": observed,
        "null_mean_rate": float(values.mean()),
        "null_p025_rate": float(np.quantile(values, 0.025)),
        "null_p975_rate": float(np.quantile(values, 0.975)),
        "lift": float(observed - values.mean()),
        "permutation_p_one_sided": float((1 + np.sum(values >= observed)) / (repetitions + 1)),
        "repetitions": repetitions,
    }


def _rate_summary(frame: pd.DataFrame, column: str) -> dict[str, object]:
    total = int(len(frame))
    successes = int(frame[column].eq(True).sum()) if total else 0
    low, high = wilson_interval(successes, total)
    return {
        "n": total,
        "successes": successes,
        "rate": successes / total if total else None,
        "wilson_95_low": low if total else None,
        "wilson_95_high": high if total else None,
    }


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
    if name == "early_recovered_2010_2014":
        return frame[frame["season"].between(2010, 2014)]
    raise ValueError(f"Unknown cohort {name}")


def build_summary(
    qc: dict[str, dict[str, int]],
    hutton_results: pd.DataFrame,
    negative_results: pd.DataFrame,
    polyakov_results: pd.DataFrame,
    weather: pd.DataFrame,
    repetitions: int = 2000,
) -> dict[str, object]:
    cohorts = (
        "all_2010_2025",
        "development_2015_2022",
        "holdout_2023_2025",
        "holdout_2023_2025_unseen_fields",
        "early_recovered_2010_2014",
    )
    summary: dict[str, object] = {
        "study_status": "retrospective_temporal_association_not_causal_accuracy",
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
        "negative_season_sensitivity": {},
    }
    hutton_summary: dict[str, object] = {}
    for mode in ("direct", "expanded"):
        mode_frame = hutton_results[hutton_results["source_mode"].eq(mode)]
        mode_summary: dict[str, object] = {}
        for cohort_name in cohorts:
            cohort_frame = _cohort(mode_frame, cohort_name)
            localized = cohort_frame[cohort_frame["onset_interval_days"].le(21)]
            cohort_summary: dict[str, object] = {
                "all_events": int(len(cohort_frame)),
                "localized_events_le_21d": int(len(localized)),
                "explicit_negative_events": int(
                    localized["explicit_negative_before_positive"].eq(True).sum()
                ),
            }
            for lookback in DEFAULT_LOOKBACKS:
                supportive = int(
                    (
                        localized[f"hutton_{lookback}d_first_positive"].eq(True)
                        & localized[f"hutton_{lookback}d_last_negative"].eq(False)
                    ).sum()
                )
                contradictory = int(
                    (
                        localized[f"hutton_{lookback}d_first_positive"].eq(False)
                        & localized[f"hutton_{lookback}d_last_negative"].eq(True)
                    ).sum()
                )
                cohort_summary[f"lookback_{lookback}d"] = {
                    "first_detection": _rate_summary(
                        localized, f"hutton_{lookback}d_first_positive"
                    ),
                    "interval_possible": _rate_summary(
                        localized, f"hutton_{lookback}d_interval_possible"
                    ),
                    "interval_robust": _rate_summary(
                        localized, f"hutton_{lookback}d_interval_robust"
                    ),
                    "paired_supportive": supportive,
                    "paired_contradictory": contradictory,
                    "paired_one_sided_p": paired_one_sided_binomial_p(
                        supportive, contradictory
                    ),
                    "date_permutation": hutton_date_permutation_test(
                        localized,
                        weather,
                        lookback,
                        repetitions=repetitions,
                    ),
                    "alarm_burden_june_august": hutton_alarm_burden(
                        localized, weather, lookback
                    ),
                }
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
                cohort_frame["polyakov_design"].eq("observed_phenology_interval")
            ]
            expanded = interval_frame[
                interval_frame["phenophase_interval_days"].le(21)
                & interval_frame["onset_interval_days"].le(21)
            ]
            exact = cohort_frame[
                cohort_frame["polyakov_design"].eq("observed_bbch51_interval")
                & cohort_frame["phenophase_interval_days"].le(21)
                & cohort_frame["onset_interval_days"].le(21)
            ]
            point = cohort_frame[
                cohort_frame["polyakov_design"].eq("observed_bbch51_point_assumption")
            ]
            cohort_summary = {}
            for case_name, cases in (
                ("phenology_and_onset_le_21d", expanded),
                ("exact_bbch51_interval_le_21d", exact),
                ("observed_bbch51_point_assumption", point),
            ):
                cohort_summary[case_name] = {
                    "possible": _rate_summary(cases, "polyakov_interval_possible"),
                    "robust": _rate_summary(cases, "polyakov_interval_robust"),
                    "bbch_upper_bound": _rate_summary(
                        cases, "polyakov_bbch_upper_bound_hit"
                    ),
                    "phenology_only_possible": _rate_summary(
                        cases, "phenology_only_interval_possible"
                    ),
                    "phenology_only_robust": _rate_summary(
                        cases, "phenology_only_interval_robust"
                    ),
                    "phenology_only_bbch_upper_bound": _rate_summary(
                        cases, "phenology_only_bbch_upper_bound_hit"
                    ),
                    "date_permutation": polyakov_date_permutation_test(
                        cases, repetitions=repetitions
                    ),
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
            mode_summary[cohort_name] = {
                "all_negative_only_seasons": int(len(cohort_frame)),
                "lenient_coverage_seasons": int(cohort_frame["lenient_coverage"].sum()),
                "strict_coverage_seasons": int(cohort_frame["strict_coverage"].sum()),
                "strict_with_explicit_clean_visit": int(
                    (
                        cohort_frame["strict_coverage"]
                        & cohort_frame["explicit_negative_visit_count"].gt(0)
                    ).sum()
                ),
                "strict_last_visit_hutton_21d": _rate_summary(
                    cohort_frame[cohort_frame["strict_coverage"]],
                    "hutton_21d_last_negative",
                ),
            }
        negative_summary[mode] = mode_summary
    summary["negative_season_sensitivity"] = negative_summary
    return summary


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
        visits = visits[visits["season"].le(maximum_season)].copy()
        development_fields = set(
            visits.loc[visits["season"].between(2015, 2022), "field_uid"]
        )
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
        ).assign(source_mode=mode)
        polyakov_interval = evaluate_polyakov_cases(
            polyakov_cases, weather
        ).assign(
            source_mode=mode,
            polyakov_design="observed_phenology_interval",
        )
        polyakov_exact_interval = evaluate_polyakov_cases(
            polyakov_exact_interval_cases, weather
        ).assign(
            source_mode=mode,
            polyakov_design="observed_bbch51_interval",
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
        for table in (hutton, negative, polyakov):
            table["field_seen_in_development"] = table["field_uid"].isin(
                development_fields
            )
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
        repetitions=repetitions,
    )
    input_path = Path(csv_path)
    digest = hashlib.sha256()
    with input_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    summary["input"] = {
        "path": str(input_path.resolve()),
        "sha256": digest.hexdigest(),
        "maximum_complete_season": maximum_season,
    }
    summary["model_contract"] = {
        "hutton": {
            "daily_rule": "Tmin >= 10 C and at least 6 hourly intervals with RH >= 90%",
            "period_rule": "two consecutive local calendar days",
            "observation_day_excluded": True,
            "lookback_days": list(DEFAULT_LOOKBACKS),
            "weather_source": "Open-Meteo ERA5-Land",
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
        },
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
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
