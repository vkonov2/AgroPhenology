"""Audited spring carrot-fly observation cohort; no weather, fitting or scores.

Rows represent monitoring series, not confirmed independent physical fields.
The source's rounded locations define spatial bootstrap groups only. Unknown
counts remain unknown; a sampled zero is not proof of biological absence.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd


SOURCE_ID = "zenodo_6351794"
ORGANISM = "Psila rosae"
UNIT = "source_trap_catch_value"
TIMEZONE = "Europe/Copenhagen"
COORDINATE_SCOPE = "public_rounded_field"

REGISTRY_COLUMNS = [
    "source_id", "site_id", "field_season", "season", "entry_date",
    "first_recorded_event_date", "last_complete_zero_date", "last_assessment_date",
    "cohort_included", "eligibility_category", "warnable_first_event",
    "positive_at_first_visit", "spatial_group", "coordinate_scope",
    "previous_visit_gap_days", "spring_observation_rows", "spring_measured_rows",
    "spring_missing_count_rows", "series_observation_rows", "series_missing_count_rows",
]
DECISION_COLUMNS = [
    "source_id", "site_id", "field_season", "season", "entry_date", "spatial_group",
    "issue_date", "issued_at", "service_active", "evaluation_field_day",
    "target_class", "target_observable", "days_to_first_recorded_event",
    "warnable_first_event", "coordinate_scope", "previous_visit_gap_days",
    "common_weather_complete",
]


def _stable_key(kind: str, values: list) -> str:
    encoded = json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return kind + "_" + hashlib.sha256(encoded).hexdigest()[:20]


def _prepare(observations: pd.DataFrame) -> pd.DataFrame:
    required = {"source_id", "site_id", "observation_date", "organism", "value", "unit",
                "source_latitude", "source_longitude"}
    if not required.issubset(observations):
        raise ValueError(f"Missing observation columns: {sorted(required - set(observations))}")
    if observations.source_id.isna().any() or set(observations.source_id) != {SOURCE_ID}:
        raise ValueError("Only the zenodo_6351794 dataset is accepted")
    frame = observations[observations.organism.eq(ORGANISM)].copy()
    if frame.empty:
        raise ValueError("No Psila rosae observations")
    if frame.site_id.isna().any() or frame.site_id.astype(str).eq("").any():
        raise ValueError("Unknown site identifier")
    if frame.unit.isna().any() or set(frame.unit) != {UNIT}:
        raise ValueError("Unexpected catch measurement unit")
    try:
        dates = pd.to_datetime(frame.observation_date, errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid observation dates") from error
    if dates.isna().any():
        raise ValueError("Missing observation dates")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    frame["observation_date"] = dates.dt.normalize()
    frame["season"] = frame.observation_date.dt.year.astype(int)
    try:
        frame["value"] = pd.to_numeric(frame.value, errors="raise").astype("Float64")
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid catch count") from error
    values = frame.value.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any() or np.any(values[np.isfinite(values)] < 0):
        raise ValueError("Counts must be nonnegative finite measurements or missing")
    if frame.duplicated(["site_id", "season", "observation_date"]).any():
        raise ValueError("Duplicate site-season observation dates")
    for name, low, high in (("source_latitude", -90, 90), ("source_longitude", -180, 180)):
        try:
            coordinate = pd.to_numeric(frame[name], errors="raise").astype(float)
        except (ValueError, TypeError) as error:
            raise ValueError("Unknown or invalid public source coordinates") from error
        if not np.isfinite(coordinate).all() or not coordinate.between(low, high).all():
            raise ValueError("Unknown or invalid public source coordinates")
        # Float hex canonically merges textual representations such as '1.0'
        # and '1.000' without coarsening the source's original precision.
        frame[name] = coordinate.where(coordinate.ne(0), 0.0)
    return frame.sort_values(["site_id", "season", "observation_date"], kind="stable")


def build_registry(observations: pd.DataFrame) -> pd.DataFrame:
    """Retain all Psila series, explicitly identifying the fixed spring cohort.

    Entry is the FIRST source observation, never a later convenient zero.
    Include iff that first observation is measured zero and no later than
    May 31. Follow spring observations through the fixed June 30 boundary.
    A first positive at entry is left censored, never a new warnable event.
    Three calendar days' warning from entry+1 requires event >= entry+4.
    Missing counts neither establish a zero nor extend measured follow-up.
    """
    frame = _prepare(observations)
    rows = []
    for (site, year), group in frame.groupby(["site_id", "season"], sort=True):
        locations = group[["source_latitude", "source_longitude"]].drop_duplicates()
        if len(locations) != 1:
            raise ValueError("One monitoring series has inconsistent source coordinates")
        location = locations.iloc[0]
        spatial = _stable_key("spatial", [float(location.source_latitude).hex(),
                                          float(location.source_longitude).hex()])
        first = group.iloc[0]
        entry = first.observation_date
        spring = group[group.observation_date.le(pd.Timestamp(year=int(year), month=6, day=30))]
        positives = spring.loc[spring.value.gt(0).fillna(False), "observation_date"]
        event = positives.min() if len(positives) else pd.NaT
        zeros = spring[spring.value.eq(0).fillna(False)]
        if pd.notna(event):
            zeros = zeros[zeros.observation_date.lt(event)]
        last_zero = zeros.observation_date.max() if len(zeros) else pd.NaT
        measured = spring[spring.value.notna()]
        last_assessment = measured.observation_date.max() if len(measured) else pd.NaT
        first_known = pd.notna(first.value)
        entry_positive = bool(first_known and first.value > 0)
        early_entry = entry <= pd.Timestamp(year=int(year), month=5, day=31)
        included = bool(early_entry and first_known and first.value == 0)
        eligible = bool(included and pd.notna(event) and pd.notna(last_zero)
                        and event >= entry + pd.Timedelta(days=4))
        if not early_entry:
            category = "outside_spring_entry"
        elif not first_known:
            category = "missing_at_entry"
        elif entry_positive:
            category = "positive_at_entry_left_censored"
        elif pd.isna(event):
            category = "right_censored"
        elif eligible:
            category = "eligible"
        else:
            category = "insufficient_warning_opportunity"
        rows.append(dict(
            source_id=SOURCE_ID, site_id=site,
            field_season=_stable_key("carrot", [str(site), int(year)]), season=int(year),
            entry_date=entry, first_recorded_event_date=event,
            last_complete_zero_date=last_zero, last_assessment_date=last_assessment,
            cohort_included=included, eligibility_category=category,
            warnable_first_event=eligible, positive_at_first_visit=entry_positive,
            spatial_group=spatial, coordinate_scope=COORDINATE_SCOPE,
            previous_visit_gap_days=(event - last_zero).days
            if pd.notna(event) and pd.notna(last_zero) else np.nan,
            spring_observation_rows=len(spring), spring_measured_rows=len(measured),
            spring_missing_count_rows=int(spring.value.isna().sum()),
            series_observation_rows=len(group), series_missing_count_rows=int(group.value.isna().sum()),
        ))
    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


def build_decisions(registry: pd.DataFrame) -> pd.DataFrame:
    """Fixed entry+1..June30 grid; future visits never create issue dates.

    An outcome record becomes available on the following local day. Evaluator
    censoring uses last measured assessment; neither it nor future outcomes
    are model inputs here. There are no inferred daily biological negatives.
    """
    required = set(REGISTRY_COLUMNS[:15])
    if not required.issubset(registry):
        raise ValueError(f"Missing registry columns: {sorted(required - set(registry))}")
    if registry.field_season.duplicated().any():
        raise ValueError("Duplicate registry field-season keys")
    if registry.source_id.isna().any() or set(registry.source_id) != {SOURCE_ID}:
        raise ValueError("Unexpected registry dataset")
    blocks = []
    for row in registry[registry.cohort_included].itertuples(index=False):
        entry = pd.Timestamp(row.entry_date).normalize()
        end = pd.Timestamp(year=int(row.season), month=6, day=30)
        if pd.isna(entry) or entry.year != row.season or entry > end:
            raise ValueError("Invalid cohort entry date")
        days = pd.date_range(entry + pd.Timedelta(days=1), end, freq="D")
        block = pd.DataFrame({"issue_date": days})
        block["issued_at"] = [(day + pd.Timedelta(hours=8)).tz_localize(TIMEZONE).isoformat() for day in days]
        for name in ("source_id", "site_id", "field_season", "season", "entry_date", "spatial_group",
                     "warnable_first_event", "coordinate_scope", "previous_visit_gap_days"):
            block[name] = getattr(row, name)
        event = pd.Timestamp(row.first_recorded_event_date)
        block["service_active"] = True if pd.isna(event) else days <= event
        last = pd.Timestamp(row.last_assessment_date)
        block["evaluation_field_day"] = block.service_active & (days <= last)
        block["days_to_first_recorded_event"] = (event - days).days if pd.notna(event) else np.nan
        block["target_class"] = "unknown"
        block["target_observable"] = False
        block["common_weather_complete"] = False
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)[DECISION_COLUMNS] if blocks else pd.DataFrame(columns=DECISION_COLUMNS)
