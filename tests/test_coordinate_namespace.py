import importlib

import ssapy_toolkit.coordinates as coordinates


def test_canonical_coordinate_modules_import():
    canonical = {
        "angle_units": "rad0to2pi",
        "attitude": "attitude_quaternion_from_frame",
        "cartesian": "cart_to_cyl",
        "earth_fixed": "gcrf_to_itrf",
        "equatorial_ecliptic": "equatorial_to_ecliptic",
        "geodetic": "gcrf_to_llh",
        "inertial": "j2000_to_gcrf",
        "local_equatorial": "equatorial_to_horizontal",
        "lunar": "gcrf_to_lunar",
        "rotating_frames": "inert2rot",
        "satellite_frames": "ntw_to_gcrf",
        "sky": "ra_dec",
        "velocity": "v_from_r",
    }
    for module_name, attr in canonical.items():
        module = importlib.import_module(f"ssapy_toolkit.coordinates.{module_name}")
        assert hasattr(module, attr)
        assert hasattr(coordinates, attr)
