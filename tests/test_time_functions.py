import re

import numpy as np
import pytest
from astropy.time import Time

from ssapy_toolkit.time_functions import (
    dd_to_dms,
    dd_to_hms,
    dms_to_dd,
    hms_to_dd,
    now,
    to_gps,
)
from ssapy_toolkit.time_functions.convert_gps_to_TT import _gpsToTT


def test_dms_and_hms_decimal_conversions_round_trip_common_values():
    assert dms_to_dd("12:30:0") == pytest.approx(12.5)
    assert dms_to_dd("-12:30:0") == pytest.approx(-12.5)
    np.testing.assert_allclose(
        dms_to_dd(["0:0:0", "1:30:0"]),
        [0.0, 1.5],
    )

    assert hms_to_dd("1:0:0") == pytest.approx(15.0)
    np.testing.assert_allclose(
        hms_to_dd(["0:0:0", "2:30:0"]),
        [0.0, 37.5],
    )

    assert dd_to_dms(12.5) == "12:30:0"
    assert dd_to_hms(15.0) == "1:0:0"
    assert dd_to_hms("15:0:0") == "1:0:0"
    with pytest.warns(UserWarning, match="cannot be negative"):
        assert dd_to_hms(-15.0) == "1:0:0"


def test_gps_helpers_and_now_format():
    scalar_time = Time(10.0, format="gps")
    vector_time = Time([10.0, 20.0], format="gps")
    list_time = [Time(10.0, format="gps"), Time(20.0, format="gps")]

    assert to_gps(scalar_time) == pytest.approx(10.0)
    np.testing.assert_allclose(to_gps(vector_time), [10.0, 20.0])
    np.testing.assert_allclose(to_gps(list_time), [10.0, 20.0])
    assert to_gps(123.0) == 123.0

    assert _gpsToTT(0.0) == pytest.approx(44244.0 + 51.184 / 86400.0)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", now())
