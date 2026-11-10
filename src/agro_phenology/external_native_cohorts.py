"""Native first-registration cohorts for external insect observations.

Each load contains one dataset and one endpoint. Reported zeros, confirmed
presence and unknown measurements remain distinct. Rows are monitoring series,
not an assertion of independent physical fields. No weather, model fitting or
future visit schedule is used to construct the issue calendar.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_SPECS = {
    "kbs_myzus": {"source_id": "kbs_aphid_suction_2005_2013", "organism": "Myzus persicae"},
    "kbs_macrosiphum": {"source_id": "kbs_aphid_suction_2005_2013", "organism": "Macrosiphum euphorbiae"},
    "kbs_gossypii": {"source_id": "kbs_aphid_suction_2005_2013", "organism": "Aphis gossypii"},
    "cutworm": {"source_id": "zenodo_6351794", "organism": "Agrotis ipsilon"},
    "cpb_wisconsin": {"source_id": "zenodo_18664442", "organism": "Leptinotarsa decemlineata"},
}
STATUSES = {"positive", "zero", "unknown"}
VISIT_COLUMNS = [
    "source_id", "target", "site_id", "season", "observation_date", "status", "cluster",
    "station_id", "coordinate_scope", "entry_egg_presence", "entry_adult_presence",
    "n_reports", "missing_reports", "all_taxa_zero", "zero_mode",
]
REGISTRY_COLUMNS = [
    "source_id", "target", "site_id", "field_season", "season", "entry_date",
    "first_recorded_event_date", "last_complete_zero_date", "last_assessment_date",
    "cohort_included", "eligibility_category", "warnable_first_event", "positive_at_first_visit",
    "cluster", "station_id", "coordinate_scope", "previous_visit_gap_days", "entry_doy",
    "entry_egg_presence", "entry_adult_presence", "zero_mode", "source_visit_count",
    "unknown_visit_count", "all_taxa_zero_visits", "followup_days", "training_informative",
]
DECISION_COLUMNS = [
    "source_id", "target", "site_id", "field_season", "season", "entry_date", "cluster",
    "station_id", "issue_date", "issued_at", "issued_at_semantics", "service_active",
    "evaluation_field_day", "target_class", "target_observable", "days_to_first_recorded_event",
    "warnable_first_event", "coordinate_scope", "previous_visit_gap_days", "common_weather_complete",
    "entry_doy", "entry_egg_presence", "entry_adult_presence", "zero_mode",
]
ISSUED_AT_SEMANTICS = "assumed_08_local_calendar_time_source_timezone_unknown"


def _key(kind: str, values: list) -> str:
    body = json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return kind + "_" + hashlib.sha256(body).hexdigest()[:20]


def _base(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    required = {"source_id", "site_id", "observation_date", "organism", "value", "unit"}
    if not required.issubset(frame):
        raise ValueError(f"Missing observation columns: {sorted(required-set(frame))}")
    if frame.source_id.isna().any() or set(frame.source_id) != {source}:
        raise ValueError("A native load must contain exactly its declared dataset")
    if frame.site_id.isna().any() or frame.site_id.astype(str).str.len().eq(0).any():
        raise ValueError("Unknown monitoring series identifier")
    frame = frame.copy()
    try:
        dates = pd.to_datetime(frame.observation_date, errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid observation dates") from error
    if dates.isna().any():
        raise ValueError("Missing observation date")
    if dates.dt.tz is not None:
        raise ValueError("Timezone-aware observations need a known source timezone")
    frame["observation_date"] = dates.dt.normalize()
    frame["season"] = dates.dt.year.astype(int)
    try:
        frame["value"] = pd.to_numeric(frame.value, errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid insect counts") from error
    if np.isinf(frame.value).any() or frame.value.dropna().lt(0).any():
        raise ValueError("Counts must be nonnegative finite measurements or missing")
    return frame


def _status_table(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Presence is robust to repeated reports; zero requires every report."""
    group = frame.assign(_positive=frame.value.gt(0), _missing=frame.value.isna()).groupby(keys, sort=True, observed=True)
    output = group.agg(n_reports=("value", "size"), missing_reports=("_missing", "sum"), positive=("_positive", "any"))
    output["status"] = np.where(output.positive, "positive", np.where(output.missing_reports.eq(0), "zero", "unknown"))
    return output.drop(columns="positive").reset_index()


def _presence(status: pd.Series) -> pd.Series:
    return status.map({"positive": 1.0, "zero": 0.0, "unknown": np.nan}).astype(float)


def _kbs(frame: pd.DataFrame, target: str, zero_mode: str) -> pd.DataFrame:
    if set(frame.unit) != {"individuals_per_collection"}:
        raise ValueError("Unexpected KBS count unit")
    if "sex" not in frame or frame.sex.isna().any() or not set(frame.sex).issubset({"F", "M"}):
        raise ValueError("KBS observations require known F/M sex categories")
    keys = ["site_id", "season", "observation_date"]
    if frame.duplicated(keys + ["organism", "sex"]).any():
        raise ValueError("Duplicate KBS species-sex reports")
    collections = frame[keys].drop_duplicates()
    # This is an explicit source-wide sensitivity, not a claim of trap failure.
    # Fully reported zeros are required even for flagging an all-taxa zero.
    all_status = _status_table(frame, keys)
    pairs = frame[["organism", "sex"]].drop_duplicates()
    all_status["all_taxa_zero"] = all_status.status.eq("zero") & all_status.n_reports.eq(len(pairs))
    selected = frame[frame.organism.eq(TARGET_SPECS[target]["organism"])]
    if selected.empty:
        raise ValueError("No observations for the requested aphid species")
    status = _status_table(selected, keys)
    sexes = selected.groupby(keys, observed=True).sex.nunique().rename("sex_categories").reset_index()
    status = status.merge(sexes, on=keys, validate="one_to_one")
    incomplete_zero = status.status.eq("zero") & (~status.n_reports.eq(2) | ~status.sex_categories.eq(2))
    status.loc[incomplete_zero, "status"] = "unknown"
    output = collections.merge(status, on=keys, how="left", validate="one_to_one")
    output["status"] = output.status.fillna("unknown")
    output[["n_reports", "missing_reports"]] = output[["n_reports", "missing_reports"]].fillna(0).astype(int)
    output = output.merge(all_status[keys+["all_taxa_zero"]], on=keys, validate="one_to_one")
    if zero_mode == "all_taxa_zero_unknown":
        output.loc[output.all_taxa_zero, "status"] = "unknown"
    output["cluster"] = output.site_id.astype(str).map(lambda x: _key("station", [x]))
    output["station_id"] = output.cluster
    output["coordinate_scope"] = "station_label_only_geographic_coordinates_not_provided"
    output["entry_egg_presence"], output["entry_adult_presence"] = np.nan, np.nan
    return output


def _cutworm(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[frame.organism.eq(TARGET_SPECS["cutworm"]["organism"])].copy()
    if selected.empty or set(selected.unit) != {"source_trap_catch_value"}:
        raise ValueError("Unexpected or absent cutworm trap measurements")
    keys = ["site_id", "season", "observation_date"]
    if selected.duplicated(keys).any():
        raise ValueError("Duplicate cutworm observations")
    for name, lo, hi in [("source_latitude", -90, 90), ("source_longitude", -180, 180)]:
        if name not in selected:
            raise ValueError("Cutworm source coordinates missing")
        selected[name] = pd.to_numeric(selected[name], errors="raise").astype(float)
        if not np.isfinite(selected[name]).all() or not selected[name].between(lo, hi).all():
            raise ValueError("Invalid source coordinates")
        selected[name] = selected[name].where(selected[name].ne(0), 0.0)
    locations = selected[["site_id", "source_latitude", "source_longitude"]].drop_duplicates()
    if locations.site_id.duplicated().any():
        raise ValueError("One cutworm series maps to multiple source locations")
    locations["cluster"] = [_key("location", [float(a).hex(), float(b).hex()])
                            for a, b in zip(locations.source_latitude, locations.source_longitude)]
    output = _status_table(selected, keys).merge(locations[["site_id", "cluster"]], on="site_id", validate="many_to_one")
    output["station_id"] = output.cluster
    output["coordinate_scope"] = "public_rounded_source_location"
    output["entry_egg_presence"], output["entry_adult_presence"] = np.nan, np.nan
    output["all_taxa_zero"] = False
    return output


def _wisconsin(frame: pd.DataFrame) -> pd.DataFrame:
    if set(frame.organism) != {TARGET_SPECS["cpb_wisconsin"]["organism"]}:
        raise ValueError("Unexpected Wisconsin organism")
    if "stage" not in frame or not set(frame.stage).issubset({"adult", "larva_unspecified", "egg_mass"}):
        raise ValueError("Unexpected Wisconsin stages")
    units = {"adult": "mean_individuals_per_25_sweeps", "larva_unspecified": "mean_individuals_per_25_sweeps",
             "egg_mass": "egg_masses_per_plant"}
    if any(set(group.unit) != {units[stage]} for stage, group in frame.groupby("stage", observed=True)):
        raise ValueError("Unexpected Wisconsin stage measurement units")
    keys = ["site_id", "season", "observation_date"]
    # Preserve every source date even if the larval stage is entirely missing.
    output = frame[keys].drop_duplicates()
    status = _status_table(frame, keys+["stage"])
    larvae = status[status.stage.eq("larva_unspecified")].drop(columns="stage")
    output = output.merge(larvae, on=keys, how="left", validate="one_to_one")
    output["status"] = output.status.fillna("unknown")
    output[["n_reports", "missing_reports"]] = output[["n_reports", "missing_reports"]].fillna(0).astype(int)
    for stage, name in [("egg_mass", "entry_egg_presence"), ("adult", "entry_adult_presence")]:
        observed = status[status.stage.eq(stage)][keys+["status"]].copy()
        observed[name] = _presence(observed.pop("status"))
        output = output.merge(observed, on=keys, how="left", validate="one_to_one")
    # Authors withheld geography. Reusing a plot prefix across years does not
    # establish a persistent location. The uncertainty unit is a YEAR block.
    output["cluster"] = "year_" + output.season.astype(str)
    output["station_id"] = "geography_not_provided"
    output["coordinate_scope"] = "anonymous_plot_geography_withheld_year_block_uncertainty"
    output["all_taxa_zero"] = False
    return output


def load_visits(collection: Path, target: str, zero_mode: str = "reported") -> pd.DataFrame:
    """Load one dataset/endpoint, preserving native observation dates and zeros.

    The KBS sensitivity masks source-wide all-taxa zero collections as unknown;
    it does not assert those collections were failures. No coordinates are
    inferred for KBS or for anonymized Wisconsin plots.
    """
    if target not in TARGET_SPECS:
        raise ValueError(f"Unknown native insect target: {target}")
    if zero_mode not in {"reported", "all_taxa_zero_unknown"}:
        raise ValueError("Unknown zero interpretation")
    if zero_mode != "reported" and not target.startswith("kbs_"):
        raise ValueError("All-taxa zero sensitivity is defined for KBS only")
    source = TARGET_SPECS[target]["source_id"]
    path = Path(collection)/"normalized"/source/"observations.parquet"
    columns = ["source_id", "site_id", "observation_date", "organism", "value", "unit"]
    columns += ["sex"] if target.startswith("kbs_") else ["stage"] if target == "cpb_wisconsin" else ["source_latitude", "source_longitude"]
    frame = _base(pd.read_parquet(path, columns=columns), source)
    if target.startswith("kbs_"):
        output = _kbs(frame, target, zero_mode)
    elif target == "cutworm":
        output = _cutworm(frame)
    else:
        output = _wisconsin(frame)
    output["source_id"], output["target"], output["zero_mode"] = source, target, zero_mode
    return output.sort_values(["site_id", "season", "observation_date"], kind="stable").reset_index(drop=True)[VISIT_COLUMNS]


def build_registry(visits: pd.DataFrame) -> pd.DataFrame:
    """Use the FIRST source visit; missing/positive entry is never rescued.

    All first source series are retained with an explicit reason. A first
    positive >=entry+4 has a three-day warning opportunity from entry+1.
    Short events remain in the service cohort, only outside that denominator.
    Records after October 31 cannot turn an unobserved tail into negatives.
    """
    required = set(VISIT_COLUMNS)
    if not required.issubset(visits):
        raise ValueError(f"Missing canonical visit columns: {sorted(required-set(visits))}")
    if visits.empty:
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    if visits.target.nunique() != 1 or visits.source_id.nunique() != 1 or visits.zero_mode.nunique() != 1:
        raise ValueError("A registry cannot pool targets, datasets or zero interpretations")
    target = visits.target.iloc[0]
    if target not in TARGET_SPECS or set(visits.source_id) != {TARGET_SPECS[target]["source_id"]}:
        raise ValueError("Target and dataset do not match")
    frame = visits.copy()
    dates = pd.to_datetime(frame.observation_date, errors="raise")
    if dates.isna().any() or dates.dt.tz is not None:
        raise ValueError("Missing or timezone-ambiguous local observation dates")
    frame["observation_date"] = dates.dt.normalize()
    if not frame.season.eq(frame.observation_date.dt.year).all():
        raise ValueError("Observation date disagrees with season")
    if frame.status.isna().any() or not set(frame.status).issubset(STATUSES):
        raise ValueError("Unknown canonical observation status")
    if frame.duplicated(["site_id", "season", "observation_date"]).any():
        raise ValueError("Duplicate canonical visits")
    rows = []
    for (site, season), group in frame.groupby(["site_id", "season"], sort=True, observed=True):
        group = group.sort_values("observation_date", kind="stable")
        if group.cluster.isna().any() or group.cluster.nunique() != 1 or group.station_id.nunique() != 1:
            raise ValueError("A native series has missing or inconsistent grouping")
        first = group.iloc[0]
        entry = first.observation_date
        end = pd.Timestamp(year=int(season), month=10, day=31)
        observed = group[group.observation_date.le(end)]
        positive = observed[observed.status.eq("positive")]
        event = positive.observation_date.min() if len(positive) else pd.NaT
        zeros = observed[observed.status.eq("zero")]
        if pd.notna(event):
            zeros = zeros[zeros.observation_date.lt(event)]
        last_zero = zeros.observation_date.max() if len(zeros) else pd.NaT
        measured = observed[~observed.status.eq("unknown")]
        last = measured.observation_date.max() if len(measured) else pd.NaT
        included = bool(first.status == "zero" and entry < end)
        warnable = bool(included and pd.notna(event) and event >= entry+pd.Timedelta(days=4))
        if entry >= end:
            category = "outside_season_entry"
        elif first.status == "unknown":
            category = "unknown_at_entry"
        elif first.status == "positive":
            category = "positive_at_entry_left_censored"
        elif pd.isna(event):
            category = "right_censored"
        elif warnable:
            category = "eligible"
        else:
            category = "short_warning"
        followup = int((last-entry).days) if pd.notna(last) else 0
        rows.append(dict(
            source_id=first.source_id, target=target, site_id=site,
            field_season=_key("native", [first.source_id, target, str(site), int(season)]),
            season=int(season), entry_date=entry, first_recorded_event_date=event,
            last_complete_zero_date=last_zero, last_assessment_date=last,
            cohort_included=included, eligibility_category=category, warnable_first_event=warnable,
            positive_at_first_visit=bool(first.status == "positive"), cluster=first.cluster,
            station_id=first.station_id, coordinate_scope=first.coordinate_scope,
            previous_visit_gap_days=(event-last_zero).days if pd.notna(event) and pd.notna(last_zero) else np.nan,
            entry_doy=int(entry.dayofyear), entry_egg_presence=first.entry_egg_presence,
            entry_adult_presence=first.entry_adult_presence, zero_mode=first.zero_mode,
            source_visit_count=len(group), unknown_visit_count=int(group.status.eq("unknown").sum()),
            all_taxa_zero_visits=int(group.all_taxa_zero.sum()), followup_days=followup,
            training_informative=bool(included and followup > 0),
        ))
    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


def build_decisions(registry: pd.DataFrame) -> pd.DataFrame:
    """Fixed daily grid entry+1..October31, never tied to the next visit.

    Time of day is an explicit local-calendar research assumption. The source
    lacks timezone/receipt timestamps; no UTC timestamp is fabricated.
    """
    required = set(REGISTRY_COLUMNS)
    if not required.issubset(registry):
        raise ValueError(f"Missing registry columns: {sorted(required-set(registry))}")
    if registry.field_season.duplicated().any():
        raise ValueError("Duplicate registry series")
    if registry.target.nunique() > 1 or registry.zero_mode.nunique() > 1:
        raise ValueError("Decisions cannot pool targets or zero interpretations")
    blocks = []
    for row in registry[registry.cohort_included].itertuples(index=False):
        entry = pd.Timestamp(row.entry_date)
        end = pd.Timestamp(year=int(row.season), month=10, day=31)
        if pd.isna(entry) or entry.year != row.season or entry >= end:
            raise ValueError("Invalid native cohort entry")
        days = pd.date_range(entry+pd.Timedelta(days=1), end, freq="D")
        block = pd.DataFrame({"issue_date": days})
        block["issued_at"] = (days+pd.Timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%S")
        block["issued_at_semantics"] = ISSUED_AT_SEMANTICS
        for name in ("source_id", "target", "site_id", "field_season", "season", "entry_date", "cluster", "station_id",
                     "warnable_first_event", "coordinate_scope", "previous_visit_gap_days", "entry_doy",
                     "entry_egg_presence", "entry_adult_presence", "zero_mode"):
            block[name] = getattr(row, name)
        event = pd.Timestamp(row.first_recorded_event_date)
        last = pd.Timestamp(row.last_assessment_date)
        block["service_active"] = True if pd.isna(event) else days <= event
        block["evaluation_field_day"] = block.service_active & (days <= last)
        block["target_class"] = "unknown"
        block["target_observable"] = False
        block["days_to_first_recorded_event"] = (event-days).days if pd.notna(event) else np.nan
        block["common_weather_complete"] = False
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)[DECISION_COLUMNS] if blocks else pd.DataFrame(columns=DECISION_COLUMNS)
