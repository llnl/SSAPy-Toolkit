import pytest
import numpy as np
from astropy.time import Time
from yeager_utils.coordinates import *  # Replace with the actual module name

class TestCoordinateFunctions:
    
    @pytest.mark.parametrize(
        "input_coords, expected",
        [
            ("30d", np.pi / 6),             # 30 degrees
            ("0d0m30s", 30 * np.pi / (180 * 60 * 60)),       # 30 seconds
            ("0d30m", 30 * np.pi / (180 * 60)),           # 30 minutes
            ("1d0m0s", np.pi / 180),        # 1 degree
            ("0d1m0s", np.pi / (180 * 60))       # 1 minute
        ],
    )
    def test_dms_to_rad(self, input_coords, expected):
        result = dms_to_rad(input_coords)
        assert pytest.approx(result, rel=1e-5) == expected

    @pytest.mark.parametrize(
        "input_coords, expected",
        [
            ("30d", 30),             # 30 degrees
            ("0d0m30s", 30 / 3600),       # 30 seconds
            ("0d30m", 30 / 60),           # 30 minutes
            ("1d0m0s", 1),        # 1 degree
            ("0d1m0s", 1 / 60)       # 1 minute
        ],
    )
    def test_dms_to_deg(self, input_coords, expected):
        result = dms_to_deg(input_coords)
        assert pytest.approx(result, rel=1e-5) == expected

    @pytest.mark.parametrize(
        "input_angle, expected",
        [
            (-np.pi, np.pi),  # -pi -> pi
            (np.pi, np.pi),  # pi -> pi
            (-3 * np.pi, np.pi),  # -3pi -> pi
        ],
    )
    def test_rad0to2pi(self, input_angle, expected):
        result = rad0to2pi(input_angle)
        assert pytest.approx(result, rel=1e-5) == expected

    @pytest.mark.parametrize(
        "input_angle, expected",
        [
            (450, 90),   # 450 -> 90
            (-45, 315),  # -45 -> 315
            (720, 0),    # 720 -> 0
        ],
    )
    def test_deg0to360(self, input_angle, expected):
        result = deg0to360(input_angle)
        assert result == expected

    def test_lonlat_distance(self):
        lat1 = 0
        lat2 = 0
        lon1 = 0
        lon2 = np.pi / 2
        expected = 10018754.171394622  # approximate distance between these points on Earth
        result = lonlat_distance(lat1, lat2, lon1, lon2)
        assert pytest.approx(result, rel=1e-3) == expected

    @pytest.mark.parametrize(
        "zenith_angle, expected",
        [
            (0, 90),   # zenith angle 0 -> altitude 90
            (1, 89),  # zenith angle 90 -> altitude 0
            (90, 0),
            (45, 45),
        ],
    )
    def test_zenithangle2altitude(self, zenith_angle, expected):
        result = zenithangle2altitude(zenith_angle)
        assert pytest.approx(result, rel=1e-5) == expected

    def test_rightasension2hourangle(self):
        ra = "10:30:00"
        local_time = "12:00:00"
        expected = "22:30:0"  # expected hour angle
        result = rightasension2hourangle(ra, local_time)
        assert result == expected

    def test_equatorial_to_horizontal(self):
        observer_latitude = 40.7128  # degrees
        declination = 30  # degrees
        right_ascension = 10  # degrees
        local_time = "12:00:00"
        expected_azimuth = 104.27924211554033  # expected azimuth value
        expected_altitude = 63.460348442579196  # expected altitude value

        azimuth, altitude = equatorial_to_horizontal(
            observer_latitude,
            declination,
            right_ascension=right_ascension,
            local_time=local_time,
        )

        assert pytest.approx(azimuth, rel=1e-2) == expected_azimuth
        assert pytest.approx(altitude, rel=1e-2) == expected_altitude

    def test_horizontal_to_equatorial(self):
        observer_latitude = 40.7128  # degrees
        azimuth = 104.27924211554033  # degrees
        altitude = 63.460348442579196  # degrees

        expected_hour_angle = 30  # expected hour angle value
        expected_declination = 30  # expected declination value

        hour_angle, declination = horizontal_to_equatorial(
            observer_latitude, azimuth, altitude
        )

        assert pytest.approx(hour_angle, rel=1e-2) == expected_hour_angle
        assert pytest.approx(declination, rel=1e-2) == expected_declination

    def test_xyz_to_ecliptic(self):
        xc, yc, zc = 1, 1, 1  # Example Cartesian coordinates
        expected_longitude = 45  # Expected longitude in degrees
        expected_latitude = 35.26438968275466  # Expected latitude

        longitude, latitude = xyz_to_ecliptic(xc, yc, zc, degrees=True)

        assert pytest.approx(longitude, rel=1e-4) == expected_longitude
        assert pytest.approx(latitude, rel=1e-4) == expected_latitude

    def test_ecliptic_xyz_to_equatorial(self):
        xc, yc, zc = 1, 1, 1  # Example ecliptic XYZ coordinates
        expected_ra = 27.46113319993068  # Expected right ascension
        expected_dec = 49.408267454957425  # Expected declination

        ra, dec = ecliptic_xyz_to_equatorial(xc, yc, zc, degrees=True)

        assert pytest.approx(ra, rel=1e-4) == expected_ra
        assert pytest.approx(dec, rel=1e-4) == expected_dec


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
