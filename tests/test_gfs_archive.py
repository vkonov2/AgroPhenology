from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import netcdf_file

from agro_phenology.gfs_archive import (
    EXPERIMENT_LEADS_HOURS,
    FORECAST_FEATURE_COLUMNS,
    GFSArchiveClient,
    GFSBoundingBox,
    GFSCacheIntegrityError,
    GFSMetadataError,
    GFSSubsetArtifact,
    GFSSubsetRequest,
    GFSVariableMapping,
    HTTPPayload,
    aggregate_forecast_features,
    build_experiment_requests,
    canonical_gfs_cell_id,
    choose_non_overlapping_precipitation_intervals,
    discover_variable_mapping,
    load_historical_issue_times,
    load_subset_frame,
    select_assumed_available_cycle,
)


UTC = timezone.utc


def _dds(precipitation_name: str) -> bytes:
    return f"""
Dataset {{
  Grid {{ Float32 Temperature_height_above_ground[time=1][height=3][lat=2][lon=2]; }}
    Temperature_height_above_ground;
  Grid {{ Float32 Relative_humidity_height_above_ground[time=1][height=1][lat=2][lon=2]; }}
    Relative_humidity_height_above_ground;
  Grid {{ Float32 {precipitation_name}[time1=1][lat=2][lon=2]; }}
    {precipitation_name};
}} gfs;
""".encode()


@pytest.mark.parametrize(
    "precipitation_name",
    [
        "Total_precipitation_surface_3_Hour_Accumulation",
        "Total_precipitation_surface_6_Hour_Accumulation",
        "Total_precipitation_surface_Mixed_intervals_Accumulation",
    ],
)
def test_discovers_version_dependent_precipitation_name(precipitation_name: str) -> None:
    mapping = discover_variable_mapping(_dds(precipitation_name))
    assert mapping.temperature == "Temperature_height_above_ground"
    assert mapping.relative_humidity == "Relative_humidity_height_above_ground"
    assert mapping.precipitation == precipitation_name
    assert mapping.dds_sha256 == hashlib.sha256(_dds(precipitation_name)).hexdigest()


def test_rejects_missing_or_ambiguous_required_variables() -> None:
    with pytest.raises(GFSMetadataError, match="missing Temperature"):
        discover_variable_mapping(b"Dataset { Float32 x; } x;")
    ambiguous = _dds("Total_precipitation_surface_3_Hour_Accumulation") + (
        b" Float32 Total_precipitation_surface_6_Hour_Accumulation;"
    )
    with pytest.raises(GFSMetadataError, match="ambiguous"):
        discover_variable_mapping(ambiguous)


def test_cycle_selection_keeps_publication_unknown_and_applies_delay() -> None:
    issue = datetime(2024, 6, 1, 5, tzinfo=UTC)
    base = select_assumed_available_cycle(
        issue,
        assumed_publication_lag=timedelta(hours=4),
        assumption_id="historical_full_run_plus_4h_unverified",
    )
    delayed = select_assumed_available_cycle(
        issue,
        assumed_publication_lag=timedelta(hours=4),
        additional_delay=timedelta(hours=3),
        assumption_id="historical_full_run_plus_7h_sensitivity",
    )
    assert base.init_time_utc == datetime(2024, 6, 1, 0, tzinfo=UTC)
    assert base.assumed_available_at_utc == datetime(2024, 6, 1, 4, tzinfo=UTC)
    assert base.publication_time_utc is None
    assert delayed.init_time_utc == datetime(2024, 5, 31, 18, tzinfo=UTC)
    assert delayed.publication_time_utc is None


def test_fixed_experiment_plan_is_6_hourly_through_day_7() -> None:
    selection = select_assumed_available_cycle(
        datetime(2024, 6, 1, 5, tzinfo=UTC),
        assumed_publication_lag=timedelta(hours=4),
        assumption_id="lag_assumption",
    )
    requests = build_experiment_requests(
        selection, GFSBoundingBox(west=23.8, east=24.3, south=56.8, north=57.2)
    )
    assert tuple(request.lead_hours for request in requests) == tuple(range(6, 169, 6))
    assert len(requests) == 28
    assert requests[0].valid_time_utc == selection.init_time_utc + timedelta(hours=6)
    assert requests[-1].valid_time_utc == selection.init_time_utc + timedelta(hours=168)

    delayed = select_assumed_available_cycle(
        datetime(2024, 6, 1, 5, tzinfo=UTC),
        assumed_publication_lag=timedelta(hours=7),
        assumption_id="lag_7h_sensitivity",
    )
    delayed_requests = build_experiment_requests(
        delayed, GFSBoundingBox(west=23.8, east=24.3, south=56.8, north=57.2)
    )
    assert tuple(request.lead_hours for request in delayed_requests) == tuple(
        range(12, 175, 6)
    )
    assert len(delayed_requests) == 28


class _FakeTransport:
    def __init__(self, dds: bytes, subset: bytes) -> None:
        self.dds = dds
        self.subset = subset
        self.calls: list[tuple[str, tuple[tuple[str, str], ...]]] = []

    def get(self, url: str, *, params, max_bytes: int) -> HTTPPayload:
        canonical_params = tuple(params or ())
        self.calls.append((url, canonical_params))
        content = self.dds if url.endswith(".dds") else self.subset
        assert len(content) <= max_bytes
        return HTTPPayload(
            status_code=200,
            url=url + ("?fake=1" if params else ""),
            content=content,
            headers={"Content-Type": "application/x-netcdf"},
        )


def _request() -> GFSSubsetRequest:
    return GFSSubsetRequest(
        issue_time_utc=datetime(2024, 6, 1, 5, tzinfo=UTC),
        init_time_utc=datetime(2024, 6, 1, 0, tzinfo=UTC),
        lead_hours=3,
        bbox=GFSBoundingBox(west=23.8, east=24.3, south=56.8, north=57.2),
        availability_assumption_id="lag_assumption",
        assumed_available_at_utc=datetime(2024, 6, 1, 4, tzinfo=UTC),
    )


def test_fetches_bounded_subset_and_resumes_from_verified_cache(tmp_path: Path) -> None:
    transport = _FakeTransport(
        _dds("Total_precipitation_surface_3_Hour_Accumulation"),
        b"CDF\x01small-netcdf-payload",
    )
    client = GFSArchiveClient(tmp_path, transport=transport, max_workers=2)
    retrieved = datetime(2026, 9, 10, 18, tzinfo=UTC)
    first = client.fetch_subset(_request(), retrieved_at_utc=retrieved)
    second = client.fetch_subset(_request())

    assert not first.cached
    assert second.cached
    assert first.data_sha256 == second.data_sha256
    assert len(transport.calls) == 2
    subset_params = transport.calls[1][1]
    assert ("north", "57.2") in subset_params
    assert ("south", "56.8") in subset_params
    assert ("time", "2024-06-01T03:00:00Z") in subset_params
    assert ("accept", "netcdf3") in subset_params
    provenance = json.loads(first.provenance_path.read_text())
    assert provenance["publication_time_utc"] is None
    assert provenance["request"]["init_time_utc"] == "2024-06-01T00:00:00Z"
    assert provenance["request"]["valid_time_utc"] == "2024-06-01T03:00:00Z"
    assert provenance["retrieved_at_utc"] == "2026-09-10T18:00:00Z"


def test_cache_corruption_is_detected_instead_of_overwritten(tmp_path: Path) -> None:
    transport = _FakeTransport(
        _dds("Total_precipitation_surface_3_Hour_Accumulation"), b"CDF\x01payload"
    )
    client = GFSArchiveClient(tmp_path, transport=transport)
    artifact = client.fetch_subset(_request())
    artifact.data_path.write_bytes(b"changed")
    with pytest.raises(GFSCacheIntegrityError, match="checksum mismatch"):
        client.fetch_subset(_request())


def _write_subset(
    path: Path, *, intervals: tuple[tuple[float, float], ...] = ((0.0, 6.0),)
) -> None:
    with netcdf_file(path, "w") as dataset:
        dataset.createDimension("time", 1)
        dataset.createDimension("time1", len(intervals))
        dataset.createDimension("bounds_dim", 2)
        dataset.createDimension("height_above_ground3", 3)
        dataset.createDimension("height_above_ground4", 1)
        dataset.createDimension("latitude", 2)
        dataset.createDimension("longitude", 2)

        latitude = dataset.createVariable("latitude", "f", ("latitude",))
        longitude = dataset.createVariable("longitude", "f", ("longitude",))
        height3 = dataset.createVariable(
            "height_above_ground3", "f", ("height_above_ground3",)
        )
        height4 = dataset.createVariable(
            "height_above_ground4", "f", ("height_above_ground4",)
        )
        latitude[:] = [57.0, 56.75]
        longitude[:] = [24.0, 24.25]
        height3[:] = [2.0, 80.0, 100.0]
        height4[:] = [2.0]

        temperature = dataset.createVariable(
            "Temperature_height_above_ground",
            "f",
            ("time", "height_above_ground3", "latitude", "longitude"),
        )
        humidity = dataset.createVariable(
            "Relative_humidity_height_above_ground",
            "f",
            ("time", "height_above_ground4", "latitude", "longitude"),
        )
        precipitation = dataset.createVariable(
            "Total_precipitation_surface_6_Hour_Accumulation",
            "f",
            ("time1", "latitude", "longitude"),
        )
        bounds = dataset.createVariable("time1_bounds", "f", ("time1", "bounds_dim"))
        temperature[:] = np.array(
            [[[[280.0, 281.0], [282.0, 283.0]], [[290.0] * 2] * 2, [[300.0] * 2] * 2]]
        )
        humidity[:] = np.array([[[[80.0, 81.0], [82.0, 83.0]]]])
        precipitation[:] = np.stack(
            [np.array([[1.0, 2.0], [3.0, 4.0]]) + 10.0 * index for index in range(len(intervals))]
        )
        bounds[:] = np.asarray(intervals)


def _artifact(path: Path, *, lead_hours: int = 6) -> GFSSubsetArtifact:
    request = GFSSubsetRequest(
        issue_time_utc=datetime(2024, 6, 1, 5, tzinfo=UTC),
        init_time_utc=datetime(2024, 6, 1, 0, tzinfo=UTC),
        lead_hours=lead_hours,
        bbox=GFSBoundingBox(west=23.8, east=24.3, south=56.7, north=57.1),
        availability_assumption_id="lag_assumption",
        assumed_available_at_utc=datetime(2024, 6, 1, 4, tzinfo=UTC),
    )
    mapping = GFSVariableMapping(
        temperature="Temperature_height_above_ground",
        relative_humidity="Relative_humidity_height_above_ground",
        precipitation="Total_precipitation_surface_6_Hour_Accumulation",
        all_precipitation_candidates=("Total_precipitation_surface_6_Hour_Accumulation",),
        dds_sha256="d" * 64,
    )
    return GFSSubsetArtifact(
        data_path=path,
        provenance_path=path.with_suffix(".json"),
        data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        data_bytes=path.stat().st_size,
        dds_sha256=mapping.dds_sha256,
        request=request,
        variable_mapping=mapping,
        retrieved_at_utc=datetime(2026, 9, 10, 18, tzinfo=UTC),
        cached=False,
    )


def test_load_subset_selects_2m_and_retains_precipitation_bounds(tmp_path: Path) -> None:
    path = tmp_path / "subset.nc"
    _write_subset(path)
    frame = load_subset_frame(_artifact(path))
    assert len(frame) == 4
    assert frame["temperature_2m_k"].tolist() == [280.0, 281.0, 282.0, 283.0]
    assert frame["relative_humidity_2m_pct"].tolist() == [80.0, 81.0, 82.0, 83.0]
    assert frame["precipitation_kg_m2"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert frame["precip_interval_start_hours"].unique().tolist() == [0.0]
    assert frame["precip_interval_end_hours"].unique().tolist() == [6.0]
    assert frame["precip_interval_hours"].unique().tolist() == [6.0]
    assert frame["publication_time_utc"].isna().all()


def test_load_subset_rejects_cumulative_or_wrong_end_precipitation(tmp_path: Path) -> None:
    cumulative = tmp_path / "cumulative.nc"
    _write_subset(cumulative, intervals=((0.0, 30.0),))
    with pytest.raises(GFSMetadataError, match="ends at"):
        load_subset_frame(_artifact(cumulative, lead_hours=6))

    long_at_expected_end = tmp_path / "long.nc"
    _write_subset(long_at_expected_end, intervals=((0.0, 30.0),))
    artifact = _artifact(long_at_expected_end, lead_hours=30)
    with pytest.raises(GFSMetadataError, match="long/cumulative"):
        load_subset_frame(artifact)


def test_load_subset_chooses_latest_mixed_precipitation_interval(tmp_path: Path) -> None:
    path = tmp_path / "mixed.nc"
    _write_subset(path, intervals=((0.0, 168.0), (162.0, 168.0)))
    frame = load_subset_frame(_artifact(path, lead_hours=168))
    assert frame["precipitation_kg_m2"].tolist() == [11.0, 12.0, 13.0, 14.0]
    assert frame["precip_interval_start_hours"].unique().tolist() == [162.0]
    assert frame["precip_interval_end_hours"].unique().tolist() == [168.0]


def test_precipitation_intervals_are_not_double_counted() -> None:
    common = {
        "init_time_utc": pd.Timestamp("2024-06-01T00:00:00Z"),
        "latitude": 57.0,
        "longitude": 24.0,
    }
    frame = pd.DataFrame(
        [
            {**common, "precip_interval_start_hours": 0.0, "precip_interval_end_hours": 3.0},
            {**common, "precip_interval_start_hours": 0.0, "precip_interval_end_hours": 6.0},
            {**common, "precip_interval_start_hours": 6.0, "precip_interval_end_hours": 9.0},
            {**common, "precip_interval_start_hours": 6.0, "precip_interval_end_hours": 12.0},
        ]
    )
    chosen = choose_non_overlapping_precipitation_intervals(frame)
    assert list(
        zip(
            chosen["precip_interval_start_hours"],
            chosen["precip_interval_end_hours"],
            strict=True,
        )
    ) == [(0.0, 6.0), (6.0, 12.0)]


def test_precipitation_interval_gap_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "init_time_utc": [pd.Timestamp("2024-06-01T00:00:00Z")] * 2,
            "latitude": [57.0] * 2,
            "longitude": [24.0] * 2,
            "precip_interval_start_hours": [0.0, 9.0],
            "precip_interval_end_hours": [6.0, 12.0],
        }
    )
    with pytest.raises(GFSMetadataError, match="gap after hour 6"):
        choose_non_overlapping_precipitation_intervals(frame)


def test_canonical_grid_cell_id_normalizes_longitude() -> None:
    assert canonical_gfs_cell_id(57.0, 24.0) == "gfs025_lat+57.00_lon+024.00"
    assert canonical_gfs_cell_id(57.0, 384.0) == "gfs025_lat+57.00_lon+024.00"


def _complete_snapshot_frame() -> pd.DataFrame:
    issue = pd.Timestamp("2024-06-01T05:00:00Z")
    init = pd.Timestamp("2024-06-01T00:00:00Z")
    rows = []
    for index, lead in enumerate(range(6, 169, 6), start=1):
        rows.append(
            {
                "issue_time_utc": issue,
                "init_time_utc": init,
                "valid_time_utc": init + pd.Timedelta(hours=lead),
                "latitude": 57.0,
                "longitude": 24.0,
                "temperature_2m_k": 273.15 + index,
                "relative_humidity_2m_pct": 70.0 + index,
                "precipitation_kg_m2": 1.0,
                "precip_interval_hours": 6.0,
                "availability_assumption_id": "gfs_full_run_plus_4h_unverified",
                "assumed_available_at_utc": init + pd.Timedelta(hours=4),
                "retrieved_at_utc": pd.Timestamp("2026-09-10T18:00:00Z"),
                "source_sha256": f"{index:064x}",
            }
        )
    return pd.DataFrame(rows)


def test_aggregate_forecast_features_uses_decision_relative_bands() -> None:
    result = aggregate_forecast_features(_complete_snapshot_frame())
    assert len(result) == 1
    row = result.iloc[0]
    assert row["gfs_cell_id"] == "gfs025_lat+57.00_lon+024.00"
    assert row["availability_scenario_hours"] == 4
    assert row["first_lead_h"] == 6
    assert row["last_lead_h"] == 168
    assert row["expected_steps_1_3d"] == row["observed_steps_1_3d"] == 12
    assert row["expected_steps_4_7d"] == row["observed_steps_4_7d"] == 16
    assert bool(row["complete_1_3d"])
    assert bool(row["complete_4_7d"])
    assert bool(row["forecast_available"])
    assert row["fcst_t_mean_d1_3"] == pytest.approx(6.5)
    assert row["fcst_t_min_d1_3"] == pytest.approx(1.0)
    assert row["fcst_t_max_d1_3"] == pytest.approx(12.0)
    assert row["fcst_rh_mean_d1_3"] == pytest.approx(76.5)
    assert row["fcst_rh_max_d1_3"] == pytest.approx(82.0)
    assert row["fcst_precip_sum_d1_3"] == pytest.approx(12.0)
    assert row["fcst_t_mean_d4_7"] == pytest.approx(20.5)
    assert row["fcst_precip_sum_d4_7"] == pytest.approx(16.0)
    assert set(FORECAST_FEATURE_COLUMNS).issubset(result.columns)


def test_incomplete_band_is_flagged_and_features_are_null() -> None:
    snapshots = _complete_snapshot_frame().iloc[1:].copy()
    result = aggregate_forecast_features(snapshots)
    row = result.iloc[0]
    assert row["observed_steps_1_3d"] == 11
    assert not bool(row["complete_1_3d"])
    assert not bool(row["forecast_available"])
    assert pd.isna(row["fcst_t_mean_d1_3"])
    assert not pd.isna(row["fcst_t_mean_d4_7"])


def test_historical_issue_loader_filters_pre_archive_and_inactive_rows(tmp_path: Path) -> None:
    path = tmp_path / "daily.parquet"
    pd.DataFrame(
        {
            "issue_date": ["2014-06-01", "2015-06-01", "2015-06-01", "2015-06-02"],
            "issued_at": [
                "2014-06-01T08:00:00+03:00",
                "2015-06-01T08:00:00+03:00",
                "2015-06-01T08:00:00+03:00",
                "2015-06-02T08:00:00+03:00",
            ],
            "service_active": [True, True, True, False],
        }
    ).to_parquet(path, index=False)
    result = load_historical_issue_times(path)
    assert result == [datetime(2015, 6, 1, 5, tzinfo=UTC)]
