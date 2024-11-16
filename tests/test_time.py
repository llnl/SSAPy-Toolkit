import pytest
import numpy as np
from astropy.time import Time
from yeager_utils.time import now, _gpsToTT, dms_to_dd, dd_to_dms, hms_to_dd, dd_to_hms, get_times


def test_now():
    """Test the now() function returns the current time as a string."""
    current_time = now()
    assert isinstance(current_time, str)
    assert len(current_time) == 16  # 'YYYY-MM-DD HH:MM' format


def test_gpsToTT():
    """Test the _gpsToTT() function converts GPS time to TT."""
    gps_time = 1000000.0  # Example GPS time in seconds
    tt_time = _gpsToTT(gps_time)
    assert isinstance(tt_time, float)
    assert np.isclose(tt_time, 44244.0 + (gps_time + 51.184) / 86400)


@pytest.mark.parametrize(
    "dms, expected_dd",
    [
        ("10:15:30", 10 + 15 / 60 + 30 / 3600),
        (["10:15:30", "20:30:45"], [10 + 15 / 60 + 30 / 3600, 20 + 30 / 60 + 45 / 3600]),
    ],
)
def test_dms_to_dd(dms, expected_dd):
    """Test the dms_to_dd() function converts DMS to decimal degrees."""
    dd = dms_to_dd(dms)
    if isinstance(dd, list):
        assert isinstance(dd, list)
        assert len(dd) == len(expected_dd)
        assert all(np.isclose(a, b) for a, b in zip(dd, expected_dd))
    else:
        assert isinstance(dd, float)
        assert np.isclose(dd, expected_dd)


def test_dd_to_dms():
    """Test the dd_to_dms() function converts decimal degrees to DMS."""
    dd = 10.25833333333333333333333
    dms = dd_to_dms(dd)
    assert isinstance(dms, str)
    assert dms == "10:15:30"


@pytest.mark.parametrize(
    "hms, expected_dd",
    [
        ("10:15:30", 10 * 15 + 15 / 4 + 30 / 240),
        (["10:15:30", "20:30:45"], [10 * 15 + 15 / 4 + 30 / 240, 20 * 15 + 30 / 4 + 45 / 240]),
    ],
)
def test_hms_to_dd(hms, expected_dd):
    """Test the hms_to_dd() function converts HMS to decimal degrees."""
    dd = hms_to_dd(hms)
    if isinstance(dd, list):
        assert isinstance(dd, list)
        assert len(dd) == len(expected_dd)
        assert all(np.isclose(a, b) for a, b in zip(dd, expected_dd))
    else:
        assert isinstance(dd, float)
        assert np.isclose(dd, expected_dd)


def test_dd_to_hms():
    """Test the dd_to_hms() function converts decimal degrees to HMS."""
    dd = 10.2583333
    hms = dd_to_hms(dd)
    assert isinstance(hms, str)
    assert hms == "0:41:2"


def test_get_times():
    """Test the get_times() function generates the correct time steps."""
    duration = (30, "day")
    freq = (1, "hr")
    t = Time("2025-01-01", scale="utc")
    times = get_times(duration, freq, t)

    assert isinstance(times.decimalyear, np.ndarray)
    assert len(times) == 30 * 24 + 1  # 30 days at 1 hour interval
    assert times[0] == t

    # Test with different frequency
    freq = (6, "hr")
    times = get_times(duration, freq, t)
    assert len(times) == 30 * 4 + 1  # 30 days at 6 hour interval


def test_invalid_units():
    """Test that invalid time units raise ValueError in get_times."""
    with pytest.raises(ValueError):
        get_times((30, "invalid_unit"), (1, "hr"), Time("2025-01-01", scale="utc"))

    with pytest.raises(ValueError):
        get_times((30, "day"), (1, "invalid_unit"), Time("2025-01-01", scale="utc"))


if __name__ == "__main__":
    import os
    import pytest

    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the script's name
    script_name = os.path.basename(__file__)
    
    # Construct the path dynamically
    test_dir = os.path.join(current_dir, script_name)
    
    # Run pytest
    pytest.main([test_dir])