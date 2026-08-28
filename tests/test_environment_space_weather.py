from datetime import date

import pytest
from astropy.time import Time

from ssapy_toolkit.environment import SpaceEnvironment
from ssapy_toolkit.environment_space_weather import (
    SpaceWeatherRecord,
    SpaceWeatherTable,
)


def test_space_weather_properties_lookup_and_msis_inputs():
    records = tuple(
        SpaceWeatherRecord(
            date(2025, 1, day), day * 86_400.0, 100.0 + day, 110.0 + day,
            90.0 + day, 95.0 + day, 4.0 + day, tuple(float(i) for i in range(1, 9)),
            "PRD" if day == 4 else "OBS",
        )
        for day in range(1, 5)
    )
    table = SpaceWeatherTable(records, source="test")
    assert table.start_gps == 86_400.0
    assert table.end_gps == 5 * 86_400.0
    assert not table.records[0].predicted
    query = 4 * 86_400.0 + 43_200.0
    with pytest.raises(ValueError, match="predicted"):
        table.at(query)
    assert table.at(query, allow_predicted=True).predicted
    f107, f107a, ap = table.msis_inputs(query, allow_predicted=True)
    assert f107 == pytest.approx(113.0)
    assert f107a == pytest.approx(99.0)
    assert ap.shape == (7,)
    assert ap[0] == pytest.approx(8.0)


def test_nrlmsise00_density_uses_packaged_eop_and_space_weather():
    pytest.importorskip("pymsis")
    environment = SpaceEnvironment(atmosphere_density_model="nrlmsise00")
    density = environment.density(
        400_000.0,
        Time("2024-01-15T12:00:00", scale="utc").gps,
        [6_778_137.0, 0.0, 0.0],
        [0.0, 7_668.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        None,
    )

    assert density == pytest.approx(6.060943413821462e-12, rel=2.0e-10)
