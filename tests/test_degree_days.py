import numpy as np

from agro_phenology.degree_days import cumulative_degree_days, daily_degree_days


def test_degree_day_example() -> None:
    daily = daily_degree_days([8, 10, 12, 15], base_temperature_c=10)
    cumulative = cumulative_degree_days(daily)
    np.testing.assert_allclose(daily, [0, 0, 2, 5])
    np.testing.assert_allclose(cumulative, [0, 0, 2, 7])
