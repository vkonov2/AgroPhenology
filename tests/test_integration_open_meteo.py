import os

import pytest

from agro_phenology.open_meteo import OpenMeteoClient


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_OPEN_METEO_INTEGRATION") != "1", reason="set RUN_OPEN_METEO_INTEGRATION=1")
def test_live_era5_land_request(tmp_path) -> None:
    response = OpenMeteoClient(tmp_path).fetch_hourly(55.7558, 37.6173, "2023-04-01", "2023-04-02")
    assert not response.hourly.empty
    assert response.metadata["model"] == "era5_land"


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_OPEN_METEO_INTEGRATION") != "1", reason="set RUN_OPEN_METEO_INTEGRATION=1")
def test_live_era5_precipitation_request(tmp_path) -> None:
    response = OpenMeteoClient(tmp_path).fetch_era5_precipitation_hourly(
        55.7558,
        37.6173,
        "2023-04-01",
        "2023-04-02",
    )
    assert response.hourly["precipitation"].notna().all()
    assert response.metadata["model"] == "era5"
