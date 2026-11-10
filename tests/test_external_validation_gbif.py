import pandas as pd

from agro_phenology.external_validation_gbif import exact_event_date


def test_partial_event_date_is_not_imputed():
    assert pd.isna(exact_event_date({"year": 2020, "month": 6, "eventDate": "2020-06"}))
    assert pd.isna(exact_event_date({"eventDate": "2020-06-01/2020-06-20"}))
    assert exact_event_date({"eventDate": "2020-06-12T18:00:00Z"}) == pd.Timestamp("2020-06-12")


def test_complete_components_require_valid_calendar_date():
    assert pd.isna(exact_event_date({"year": 2020, "month": 2, "day": 30}))
    assert exact_event_date({"year": 2020, "month": 2, "day": 29}) == pd.Timestamp("2020-02-29")
