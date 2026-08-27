from datetime import date

from agro_phenology.degree_days import first_threshold_date


def test_first_exact_threshold_date() -> None:
    dates = [date(2023, 4, day) for day in range(1, 5)]
    assert first_threshold_date(dates, [0, 2, 7, 9], threshold=7) == date(2023, 4, 3)


def test_threshold_not_reached() -> None:
    assert first_threshold_date(["2023-04-01", "2023-04-02"], [1, 2], threshold=3) is None
