from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
import json

import pandas as pd
import pytest
from scipy.io import netcdf_file

from agro_phenology.gfs_archive import (
    FORECAST_FEATURE_COLUMNS,
    GFSArchiveError,
    HTTPPayload,
)
from agro_phenology.early_warning_gfs_experiment import (
    validate_forecast_feature_table,
)
from agro_phenology.gfs_feature_builder import (
    AUDITED_DDS_ACCESS_MODE,
    DEFAULT_CONTRACT,
    DYNAMIC_BBOX_MODE,
    DYNAMIC_BBOX_PADDING_DEGREES,
    FAST_NCSS_ACCESS_MODE,
    FAST_NCSS_PROVENANCE_SCHEMA,
    FAST_PRECIPITATION_CANDIDATES,
    FastNCSSCandidateClient,
    FIXED_BBOX_MODE,
    aggregate_requested_cells,
    assemble_feature_table,
    assert_no_private_or_target_columns,
    build_request_plan,
    fetch_plan_checkpoint,
    gfs_cell_id_to_grid_center,
    minimal_bbox_for_gfs_cells,
    requests_for_plan_row,
    weather_cell_to_gfs_cell_id,
    write_request_plan,
)


def _contract() -> dict:
    return {
        "source_dataset": {"id": "d084001"},
        "forecast_availability": {
            "selection_rule_id": "latest_cycle_assumed_published_before_issue_v1",
            "main_assumed_hours_after_initialization": 4,
            "delay_sensitivity_total_hours_after_initialization": 7,
        },
    }


def _decisions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "issue_date": pd.to_datetime(
                ["2024-06-01", "2024-06-01", "2024-06-02", "2024-06-02"]
            ),
            "issued_at": [
                "2024-06-01T08:00:00+03:00",
                "2024-06-01T08:00:00+03:00",
                "2024-06-02T08:00:00+03:00",
                "2024-06-02T08:00:00+03:00",
            ],
            "service_active": [True, True, True, False],
            "weather_cell": ["57.00_24.00", "56.75_24.25", "57.00_24.00", "25_25"],
            # These evaluator columns must never be copied to the request plan.
            "field_uid": ["secret-a", "secret-b", "secret-a", "secret-c"],
            "target_class": ["unknown", "none", "actionable", "none"],
        }
    )


def test_plan_uses_dynamic_decision_relative_slots_and_both_lags() -> None:
    plan = build_request_plan(_decisions(), _contract())
    assert len(plan) == 4
    assert plan["availability_scenario_hours"].tolist() == [4, 7, 4, 7]
    assert plan["expected_steps_1_3d"].eq(12).all()
    assert plan["expected_steps_4_7d"].eq(16).all()
    assert plan["requested_snapshot_count"].eq(28).all()

    base = plan.iloc[0]
    delayed = plan.iloc[1]
    assert base["selected_init_utc"] == pd.Timestamp("2024-06-01T00:00:00Z")
    assert delayed["selected_init_utc"] == pd.Timestamp("2024-05-31T18:00:00Z")
    base_requests = requests_for_plan_row(base)
    delayed_requests = requests_for_plan_row(delayed)
    assert [request.lead_hours for request in base_requests] == list(range(6, 169, 6))
    assert [request.lead_hours for request in delayed_requests] == list(range(12, 175, 6))
    for requests in (base_requests, delayed_requests):
        offsets = [
            (request.valid_time_utc - request.issue_time_utc).total_seconds() / 3600
            for request in requests
        ]
        assert sum(0 < offset <= 72 for offset in offsets) == 12
        assert sum(72 < offset <= 168 for offset in offsets) == 16


def test_plan_contains_no_field_outcome_or_coordinate_mapping() -> None:
    plan = build_request_plan(_decisions(), _contract())
    assert_no_private_or_target_columns(plan)
    assert "field_uid" not in plan
    assert "target_class" not in plan
    assert "weather_cell" not in plan
    assert "latitude" not in plan
    assert "longitude" not in plan
    assert weather_cell_to_gfs_cell_id("57.00_24.00") == (
        "gfs025_lat+57.00_lon+024.00"
    )


def test_dynamic_bbox_contains_required_extremes_without_outer_grid_neighbors() -> None:
    plan = build_request_plan(_decisions(), _contract())
    first_date = plan.iloc[0]
    assert first_date["bbox_mode"] == DYNAMIC_BBOX_MODE
    assert first_date["bbox_padding_degrees"] == DYNAMIC_BBOX_PADDING_DEGREES
    cells = json.loads(first_date["required_gfs_cell_ids_json"])
    centers = [gfs_cell_id_to_grid_center(cell) for cell in cells]
    latitudes = [item[0] for item in centers]
    longitudes = [item[1] for item in centers]
    assert all(
        first_date["bbox_south"] <= latitude <= first_date["bbox_north"]
        and first_date["bbox_west"] <= longitude <= first_date["bbox_east"]
        for latitude, longitude in centers
    )
    assert first_date["bbox_south"] > min(latitudes) - 0.25
    assert first_date["bbox_north"] < max(latitudes) + 0.25
    assert first_date["bbox_west"] > min(longitudes) - 0.25
    assert first_date["bbox_east"] < max(longitudes) + 0.25

    single_cell = plan[plan["issue_date"].eq(pd.Timestamp("2024-06-02"))].iloc[0]
    assert single_cell["required_gfs_cell_count"] == 1
    assert single_cell["bbox_east"] - single_cell["bbox_west"] == pytest.approx(
        2 * DYNAMIC_BBOX_PADDING_DEGREES
    )
    assert single_cell["bbox_north"] - single_cell["bbox_south"] == pytest.approx(
        2 * DYNAMIC_BBOX_PADDING_DEGREES
    )


def test_fixed_bbox_mode_remains_explicit_and_constant() -> None:
    from agro_phenology.gfs_archive import GFSBoundingBox

    fixed = GFSBoundingBox(west=21.0, east=28.0, south=55.75, north=58.0)
    plan = build_request_plan(
        _decisions(), _contract(), bbox_mode=FIXED_BBOX_MODE, fixed_bbox=fixed
    )
    assert plan["bbox_mode"].eq(FIXED_BBOX_MODE).all()
    assert plan["bbox_padding_degrees"].isna().all()
    assert plan["bbox_west"].eq(21.0).all()
    assert plan["bbox_east"].eq(28.0).all()
    assert plan["bbox_south"].eq(55.75).all()
    assert plan["bbox_north"].eq(58.0).all()


def test_dynamic_bbox_formula_is_written_to_v2_plan_manifest(tmp_path: Path) -> None:
    decisions_path = tmp_path / "daily.parquet"
    _decisions().to_parquet(decisions_path, index=False)
    run = tmp_path / "v2"
    manifest = write_request_plan(
        run_root=run,
        parent_decisions_path=decisions_path,
        contract_path=DEFAULT_CONTRACT,
    )
    assert manifest["schema_version"] == "gfs_feature_request_plan_v2"
    assert manifest["bbox_mode"] == DYNAMIC_BBOX_MODE
    assert "min(required_grid_longitude)-padding" in manifest["bbox_formula"]
    assert manifest["gfs_grid_step_degrees"] == 0.25
    assert manifest["bbox_padding_degrees"] == DYNAMIC_BBOX_PADDING_DEGREES
    assert manifest["fixed_bbox"] is None
    assert manifest["contains_field_ids"] is False
    assert manifest["contains_outcomes"] is False
    assert manifest["contains_coarse_grid_coordinates"] is True
    assert manifest["distribution"] == "local_private_do_not_publish"
    assert "contains_field_ids_or_outcomes" not in manifest
    written = pd.read_parquet(run / "gfs_request_plan.parquet")
    assert written["bbox_mode"].eq(DYNAMIC_BBOX_MODE).all()
    assert_no_private_or_target_columns(written)


def test_dynamic_bbox_rejects_half_step_padding() -> None:
    cells = ["gfs025_lat+57.00_lon+024.00"]
    with pytest.raises(ValueError, match="padding"):
        minimal_bbox_for_gfs_cells(cells, padding_degrees=0.125)


def _snapshot_frame(plan_row: pd.Series) -> pd.DataFrame:
    rows = []
    requests = requests_for_plan_row(plan_row)
    cells = [(57.0, 24.0), (56.75, 24.25), (55.75, 21.0)]
    for index, request in enumerate(requests, start=1):
        for latitude, longitude in cells:
            rows.append(
                {
                    "issue_time_utc": pd.Timestamp(request.issue_time_utc),
                    "init_time_utc": pd.Timestamp(request.init_time_utc),
                    "valid_time_utc": pd.Timestamp(request.valid_time_utc),
                    "latitude": latitude,
                    "longitude": longitude,
                    "temperature_2m_k": 273.15 + index,
                    "relative_humidity_2m_pct": 60.0 + index,
                    "precipitation_kg_m2": 1.0,
                    "precip_interval_hours": 6.0,
                    "availability_assumption_id": request.availability_assumption_id,
                    "assumed_available_at_utc": pd.Timestamp(
                        request.assumed_available_at_utc
                    ),
                    "retrieved_at_utc": pd.Timestamp("2026-09-10T20:00:00Z"),
                    "source_sha256": f"{index:064x}",
                }
            )
    return pd.DataFrame(rows)


def test_aggregate_keeps_only_project_grid_ids_and_twelve_features() -> None:
    plan_row = build_request_plan(_decisions(), _contract()).iloc[0]
    result = aggregate_requested_cells(_snapshot_frame(plan_row), plan_row)
    assert len(result) == 2
    assert set(result["gfs_cell_id"]) == {
        "gfs025_lat+57.00_lon+024.00",
        "gfs025_lat+56.75_lon+024.25",
    }
    assert result["expected_steps_1_3d"].eq(12).all()
    assert result["observed_steps_1_3d"].eq(12).all()
    assert result["expected_steps_4_7d"].eq(16).all()
    assert result["observed_steps_4_7d"].eq(16).all()
    assert result["forecast_available"].all()
    assert result["checkpoint_complete"].all()
    assert set(FORECAST_FEATURE_COLUMNS).issubset(result.columns)
    assert "latitude" not in result
    assert "longitude" not in result
    assert result["fcst_precip_sum_d1_3"].eq(12.0).all()
    assert result["fcst_precip_sum_d4_7"].eq(16.0).all()
    validated = validate_forecast_feature_table(result)
    assert validated["forecast_available"].all()


def test_incomplete_snapshot_bundle_is_retained_but_not_model_available() -> None:
    plan_row = build_request_plan(_decisions(), _contract()).iloc[0]
    snapshots = _snapshot_frame(plan_row)
    missing_valid = snapshots["valid_time_utc"].min()
    snapshots = snapshots[snapshots["valid_time_utc"].ne(missing_valid)]
    result = aggregate_requested_cells(
        snapshots,
        plan_row,
        failed_requests=[{"lead_hours": 6, "error_type": "GFSArchiveError"}],
    )
    assert result["observed_steps_1_3d"].eq(11).all()
    assert not result["forecast_available"].any()
    assert not result["checkpoint_complete"].any()
    assert result["failed_snapshot_count"].eq(1).all()
    assert result[list(FORECAST_FEATURE_COLUMNS)[:6]].isna().all().all()
    assert result[list(FORECAST_FEATURE_COLUMNS)[6:]].notna().all().all()


def test_fetch_checkpoint_records_one_missing_file_without_aborting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_row = build_request_plan(_decisions(), _contract()).iloc[0]

    class FakeClient:
        def fetch_subset(self, request):
            if request.lead_hours == 6:
                raise RuntimeError("archive file is missing")
            return SimpleNamespace(request=request, data_sha256=f"{request.lead_hours:064x}")

    def fake_load(artifact) -> pd.DataFrame:
        request = artifact.request
        return pd.DataFrame(
            [
                {
                    "issue_time_utc": pd.Timestamp(request.issue_time_utc),
                    "init_time_utc": pd.Timestamp(request.init_time_utc),
                    "valid_time_utc": pd.Timestamp(request.valid_time_utc),
                    "latitude": latitude,
                    "longitude": longitude,
                    "temperature_2m_k": 280.0,
                    "relative_humidity_2m_pct": 80.0,
                    "precipitation_kg_m2": 1.0,
                    "precip_interval_hours": 6.0,
                    "availability_assumption_id": request.availability_assumption_id,
                    "assumed_available_at_utc": pd.Timestamp(
                        request.assumed_available_at_utc
                    ),
                    "retrieved_at_utc": pd.Timestamp("2026-09-10T20:00:00Z"),
                    "source_sha256": artifact.data_sha256,
                }
                for latitude, longitude in ((57.0, 24.0), (56.75, 24.25))
            ]
        )

    monkeypatch.setattr(
        "agro_phenology.gfs_feature_builder.load_subset_frame", fake_load
    )
    result = fetch_plan_checkpoint(plan_row, FakeClient(), max_workers=2)
    assert len(result) == 2
    assert result["retrieved_snapshot_count"].eq(27).all()
    assert result["failed_snapshot_count"].eq(1).all()
    assert result["observed_steps_1_3d"].eq(11).all()
    assert not result["forecast_available"].any()
    assert "archive file is missing" in result["failed_requests_json"].iloc[0]


def _fast_subset_bytes(
    tmp_path: Path, precipitation_name: str, *, lead_hours: int = 6
) -> bytes:
    path = tmp_path / f"{precipitation_name}_{lead_hours}.nc"
    with netcdf_file(path, "w") as dataset:
        dataset.createDimension("time", 1)
        dataset.createDimension("time1", 1)
        dataset.createDimension("bounds_dim", 2)
        dataset.createDimension("height_above_ground3", 3)
        dataset.createDimension("height_above_ground4", 1)
        dataset.createDimension("latitude", 1)
        dataset.createDimension("longitude", 1)
        dataset.createVariable("latitude", "f", ("latitude",))[:] = [57.0]
        dataset.createVariable("longitude", "f", ("longitude",))[:] = [24.0]
        dataset.createVariable(
            "height_above_ground3", "f", ("height_above_ground3",)
        )[:] = [2.0, 80.0, 100.0]
        dataset.createVariable(
            "height_above_ground4", "f", ("height_above_ground4",)
        )[:] = [2.0]
        dataset.createVariable(
            "Temperature_height_above_ground",
            "f",
            ("time", "height_above_ground3", "latitude", "longitude"),
        )[:] = [[[[280.0]], [[290.0]], [[300.0]]]]
        dataset.createVariable(
            "Relative_humidity_height_above_ground",
            "f",
            ("time", "height_above_ground4", "latitude", "longitude"),
        )[:] = [[[[80.0]]]]
        dataset.createVariable(
            precipitation_name,
            "f",
            ("time1", "latitude", "longitude"),
        )[:] = [[[2.0]]]
        dataset.createVariable(
            "time1_bounds", "f", ("time1", "bounds_dim")
        )[:] = [[float(lead_hours - 6), float(lead_hours)]]
    return path.read_bytes()


class _FastCandidateTransport:
    def __init__(
        self,
        *,
        available_precipitation: str,
        payload: bytes,
        first_error: Exception | None = None,
    ) -> None:
        self.available_precipitation = available_precipitation
        self.payload = payload
        self.first_error = first_error
        self.calls: list[str] = []

    def get(self, url: str, *, params, max_bytes: int) -> HTTPPayload:
        precipitation = next(
            value
            for key, value in params
            if key == "var" and value.startswith("Total_precipitation_surface_")
        )
        self.calls.append(precipitation)
        if self.first_error is not None and len(self.calls) == 1:
            raise self.first_error
        if precipitation != self.available_precipitation:
            raise GFSArchiveError(
                f"HTTP 400 for {url}: Variable {precipitation} is not contained "
                "in the requested dataset"
            )
        return HTTPPayload(
            status_code=200,
            url=f"{url}?selected={precipitation}",
            content=self.payload,
            headers={"Content-Type": "application/x-netcdf"},
        )


@pytest.mark.parametrize(
    ("precipitation_name", "expected_calls"),
    [
        (FAST_PRECIPITATION_CANDIDATES[0], 1),
        (FAST_PRECIPITATION_CANDIDATES[1], 2),
    ],
)
def test_fast_ncss_validates_both_known_schemas_and_records_separate_provenance(
    tmp_path: Path,
    precipitation_name: str,
    expected_calls: int,
) -> None:
    request = requests_for_plan_row(build_request_plan(_decisions(), _contract()).iloc[0])[0]
    transport = _FastCandidateTransport(
        available_precipitation=precipitation_name,
        payload=_fast_subset_bytes(tmp_path, precipitation_name),
    )
    client = FastNCSSCandidateClient(tmp_path / "cache", transport=transport)
    artifact = client.fetch_subset(
        request, retrieved_at_utc=datetime(2026, 9, 10, 20, tzinfo=timezone.utc)
    )
    cached = client.fetch_subset(request)

    assert len(transport.calls) == expected_calls
    assert cached.cached
    assert artifact.variable_mapping.precipitation == precipitation_name
    assert "d084001_fast_ncss_candidates_v1" in artifact.data_path.parts
    provenance = json.loads(artifact.provenance_path.read_text(encoding="utf-8"))
    assert provenance["schema_version"] == FAST_NCSS_PROVENANCE_SCHEMA
    assert provenance["source_access_mode"] == FAST_NCSS_ACCESS_MODE
    assert provenance["dds_requested"] is False
    assert provenance["dds_sha256"] is None
    assert provenance["publication_time_utc"] is None
    assert provenance["selected_precipitation_variable"] == precipitation_name
    assert provenance["candidate_attempts"][-1]["result"] == (
        "ncss_netcdf_parser_and_bounds_validated"
    )


@pytest.mark.parametrize(
    ("precipitation_name", "expected_calls"),
    [
        (FAST_PRECIPITATION_CANDIDATES[1], [FAST_PRECIPITATION_CANDIDATES[1]]),
        (
            FAST_PRECIPITATION_CANDIDATES[0],
            [FAST_PRECIPITATION_CANDIDATES[1], FAST_PRECIPITATION_CANDIDATES[0]],
        ),
    ],
)
def test_fast_ncss_prefers_mixed_after_f006_and_records_actual_order(
    tmp_path: Path,
    precipitation_name: str,
    expected_calls: list[str],
) -> None:
    request = requests_for_plan_row(build_request_plan(_decisions(), _contract()).iloc[0])[1]
    assert request.lead_hours == 12
    transport = _FastCandidateTransport(
        available_precipitation=precipitation_name,
        payload=_fast_subset_bytes(
            tmp_path, precipitation_name, lead_hours=request.lead_hours
        ),
    )
    client = FastNCSSCandidateClient(tmp_path / "cache", transport=transport)
    artifact = client.fetch_subset(request)
    provenance = json.loads(artifact.provenance_path.read_text(encoding="utf-8"))

    assert transport.calls == expected_calls
    assert provenance["candidate_order"] == [
        FAST_PRECIPITATION_CANDIDATES[1],
        FAST_PRECIPITATION_CANDIDATES[0],
    ]
    assert artifact.variable_mapping.all_precipitation_candidates == tuple(
        provenance["candidate_order"]
    )


def test_fast_ncss_prefers_dedicated_six_hour_name_before_2019_transition(
    tmp_path: Path,
) -> None:
    current = requests_for_plan_row(
        build_request_plan(_decisions(), _contract()).iloc[0]
    )[1]
    request = replace(
        current,
        init_time_utc=datetime(2018, 6, 1, tzinfo=timezone.utc),
        assumed_available_at_utc=datetime(2018, 6, 1, 4, tzinfo=timezone.utc),
    )
    transport = _FastCandidateTransport(
        available_precipitation=FAST_PRECIPITATION_CANDIDATES[0],
        payload=_fast_subset_bytes(
            tmp_path,
            FAST_PRECIPITATION_CANDIDATES[0],
            lead_hours=request.lead_hours,
        ),
    )
    client = FastNCSSCandidateClient(tmp_path / "cache", transport=transport)
    artifact = client.fetch_subset(request)
    provenance = json.loads(artifact.provenance_path.read_text(encoding="utf-8"))

    assert transport.calls == [FAST_PRECIPITATION_CANDIDATES[0]]
    assert provenance["candidate_order"] == [
        FAST_PRECIPITATION_CANDIDATES[0],
        FAST_PRECIPITATION_CANDIDATES[1],
    ]


@pytest.mark.parametrize(
    ("request_index", "error", "expected_first_candidate"),
    [
        (
            0,
            GFSArchiveError("HTTP 500 for archive: server failure"),
            FAST_PRECIPITATION_CANDIDATES[0],
        ),
        (
            1,
            GFSArchiveError("HTTP 400 for archive: invalid requested time"),
            FAST_PRECIPITATION_CANDIDATES[1],
        ),
    ],
)
def test_fast_ncss_does_not_mask_non_variable_errors(
    tmp_path: Path,
    request_index: int,
    error: Exception,
    expected_first_candidate: str,
) -> None:
    request = requests_for_plan_row(build_request_plan(_decisions(), _contract()).iloc[0])[
        request_index
    ]
    transport = _FastCandidateTransport(
        available_precipitation=FAST_PRECIPITATION_CANDIDATES[1],
        payload=_fast_subset_bytes(
            tmp_path,
            FAST_PRECIPITATION_CANDIDATES[1],
            lead_hours=request.lead_hours,
        ),
        first_error=error,
    )
    client = FastNCSSCandidateClient(tmp_path / "cache", transport=transport)
    with pytest.raises(GFSArchiveError, match=str(error).split(":", 1)[-1].strip()):
        client.fetch_subset(request)
    assert transport.calls == [expected_first_candidate]


def test_fast_ncss_does_not_fallback_after_successful_but_invalid_netcdf(
    tmp_path: Path,
) -> None:
    request = requests_for_plan_row(build_request_plan(_decisions(), _contract()).iloc[0])[0]
    transport = _FastCandidateTransport(
        available_precipitation=FAST_PRECIPITATION_CANDIDATES[0],
        payload=b"CDF\x01not-a-valid-netcdf-file",
    )
    client = FastNCSSCandidateClient(tmp_path / "cache", transport=transport)
    with pytest.raises(Exception):
        client.fetch_subset(request)
    assert transport.calls == [FAST_PRECIPITATION_CANDIDATES[0]]
    assert not list((tmp_path / "cache").rglob("provenance.json"))


def test_assemble_rejects_private_checkpoint_column(tmp_path: Path) -> None:
    run = tmp_path / "run"
    plan = build_request_plan(_decisions(), _contract()).iloc[[0]].copy()
    run.mkdir()
    plan.to_parquet(run / "gfs_request_plan.parquet", index=False)
    checkpoint = aggregate_requested_cells(_snapshot_frame(plan.iloc[0]), plan.iloc[0])
    checkpoint["field_uid"] = "secret"
    checkpoint_dir = run / "checkpoints/year=2024/scenario_hours=04"
    checkpoint_dir.mkdir(parents=True)
    checkpoint.to_parquet(checkpoint_dir / "bad.parquet", index=False)
    with pytest.raises(ValueError, match="private/evaluator"):
        assemble_feature_table(run_root=run)


def test_assemble_merges_date_checkpoints_and_writes_annual_files(tmp_path: Path) -> None:
    run = tmp_path / "run"
    plan = build_request_plan(_decisions(), _contract())
    run.mkdir()
    plan.to_parquet(run / "gfs_request_plan.parquet", index=False)
    for _, row in plan.iterrows():
        checkpoint = aggregate_requested_cells(_snapshot_frame(row), row)
        path = (
            run
            / "checkpoints"
            / f"year={int(row['issue_year'])}"
            / f"scenario_hours={int(row['availability_scenario_hours']):02d}"
            / f"{row['checkpoint_id']}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.to_parquet(path, index=False)

    manifest = assemble_feature_table(run_root=run)
    combined = pd.read_parquet(run / "gfs_forecast_features.parquet")
    assert len(combined) == 6
    assert len(manifest["annual_files"]) == 2
    assert manifest["missing_checkpoint_count"] == 0
    assert {item["availability_scenario_hours"] for item in manifest["annual_files"]} == {
        4,
        7,
    }
    assert all(Path(item["path"]).is_file() for item in manifest["annual_files"])
    assert manifest["contains_field_ids"] is False
    assert manifest["contains_outcomes"] is False
    assert manifest["contains_coarse_grid_coordinates"] is True
    assert manifest["distribution"] == "local_private_do_not_publish"
    assert "contains_field_ids_coordinates_or_outcomes" not in manifest
    assert_no_private_or_target_columns(combined)


@pytest.mark.parametrize("mode_value", [None, float("nan"), ""])
def test_assemble_fails_closed_for_missing_source_access_mode_rows(
    tmp_path: Path, mode_value: object
) -> None:
    run = tmp_path / "run"
    plan = build_request_plan(_decisions(), _contract()).iloc[[0]].copy()
    run.mkdir()
    plan.to_parquet(run / "gfs_request_plan.parquet", index=False)
    checkpoint = aggregate_requested_cells(_snapshot_frame(plan.iloc[0]), plan.iloc[0])
    if mode_value is None:
        checkpoint = checkpoint.drop(columns="source_access_mode")
    else:
        checkpoint.loc[checkpoint.index[0], "source_access_mode"] = mode_value
    checkpoint_dir = run / "checkpoints/year=2024/scenario_hours=04"
    checkpoint_dir.mkdir(parents=True)
    checkpoint.to_parquet(checkpoint_dir / "missing-provenance.parquet", index=False)
    with pytest.raises(ValueError, match="source_access_mode"):
        assemble_feature_table(run_root=run)
