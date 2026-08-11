import re

import numpy as np
import pytest
from astropy.time import Time

from ssapy_toolkit.time_functions import (
    dd_to_dms,
    dd_to_hms,
    dms_to_dd,
    get_times,
    hms_to_dd,
    now,
    time_abs_to_rel,
    time_rel_to_abs,
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


@pytest.mark.parametrize(
    ("value", "expected_dms", "expected_hms"),
    [
        (12.99999999, "13:0:0", "0:52:0"),
        (-12.99999999, "-13:0:0", "0:52:0"),
        (14.99999999, "15:0:0", "1:0:0"),
        (359.99999999, "360:0:0", "24:0:0"),
    ],
)

def test_decimal_angle_conversions_carry_rounded_seconds(value, expected_dms, expected_hms):
    assert dd_to_dms(value) == expected_dms
    assert dd_to_hms(value) == expected_hms


def test_hms_to_dd_rejects_negative_values():
    with pytest.raises(ValueError, match="cannot be negative"):
        hms_to_dd("-1:0:0")
    with pytest.raises(ValueError, match="cannot be negative"):
        hms_to_dd(["0:0:0", "-1:0:0"])


def test_relative_and_absolute_time_helpers_round_trip_start_and_end_anchors():
    relative = np.array([10.0, 12.5, 20.0])
    start_ref = Time("2025-01-01 00:00:00", scale="utc")
    end_ref = Time("2025-01-01 00:10:00", scale="utc")

    absolute_start = time_rel_to_abs(relative, start_ref, anchor="start")
    np.testing.assert_allclose(
        time_abs_to_rel(absolute_start, anchor="start"),
        [0.0, 2.5, 10.0],
    )

    absolute_end = time_rel_to_abs(relative, end_ref, anchor="end")
    np.testing.assert_allclose(
        time_abs_to_rel(absolute_end, anchor="end"),
        [-10.0, -7.5, 0.0],
    )
    assert absolute_end[-1].gps == pytest.approx(end_ref.gps)

    with pytest.raises(ValueError, match="anchor"):
        time_rel_to_abs(relative, start_ref, anchor="middle")
    with pytest.raises(ValueError, match="anchor"):
        time_abs_to_rel(absolute_start, anchor="middle")
    assert len(time_abs_to_rel([])) == 0


def test_get_times_start_end_middle_and_validation(capsys):
    start = get_times((10, "s"), freq=(5, "s"), t0=Time(0, format="gps"))
    np.testing.assert_allclose(start.gps - start[0].gps, [0.0, 5.0, 10.0])

    end_ref = Time(100.0, format="gps")
    end = get_times(10, freq=5, tf=end_ref)
    np.testing.assert_allclose(end.gps, [90.0, 95.0, 100.0])

    middle_ref = Time(100.0, format="gps")
    middle = get_times(10, freq=4, tm=middle_ref)
    assert middle[1].gps == pytest.approx(middle_ref.gps)
    np.testing.assert_allclose(middle.gps - middle[1].gps, [-5.0, 0.0, 5.0])
    assert "adjusted frequency" in capsys.readouterr().out

    assert get_times(0, t0=middle_ref)[0].gps == pytest.approx(middle_ref.gps)
    assert get_times(0, tf=middle_ref)[0].gps == pytest.approx(middle_ref.gps)
    assert get_times(0, tm=middle_ref)[0].gps == pytest.approx(middle_ref.gps)

    with pytest.raises(ValueError, match="duration"):
        get_times(-1, t0=middle_ref)
    with pytest.raises(ValueError, match="freq"):
        get_times(1, freq=0, t0=middle_ref)
    with pytest.raises(ValueError, match="valid time unit"):
        get_times((1, "fortnight"), t0=middle_ref)
    with pytest.raises(ValueError, match="At least one"):
        get_times(1, t0=None, tf=None, tm=None)


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
