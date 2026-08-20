"""Coordinate transformation utilities (GCRF, ITRF, LLH, and satellite frames)."""

from importlib import import_module
import sys

from ssapy_toolkit._namespace import import_public_modules

import_public_modules(__name__, __file__, globals())

_LEGACY_MODULE_ALIASES = {
    "cartesian_to_cylindrical": "cartesian",
    "cartesian_to_spherical": "cartesian",
    "earth_trojan_sim": "rotating_frames",
    "equatorial_and_ecliptic": "equatorial_ecliptic",
    "equitorial_and_ecliptic": "equatorial_ecliptic",
    "gcrf_to_itrf": "earth_fixed",
    "itrf_to_gcrf": "earth_fixed",
    "gcrf_to_llh": "geodetic",
    "gcrf_to_lonlat": "geodetic",
    "llh_to_gcrf": "geodetic",
    "lon_lat_bbox": "geodetic",
    "lonlat_perigee": "geodetic",
    "on_sky_distance": "geodetic",
    "surface_rv": "geodetic",
    "gcrf_to_lunar": "lunar",
    "lunar_position": "lunar",
    "gcrf_to_ntw": "satellite_frames",
    "ntw_to_gcrf": "satellite_frames",
    "j2000_to_gcrf": "inertial",
    "local_and_equatorial": "local_equatorial",
    "local_and_equitorial": "local_equatorial",
    "sky_angles": "sky",
    "unit_conversions": "angle_units",
    "v_from_r": "velocity",
}

for _legacy_name, _canonical_name in _LEGACY_MODULE_ALIASES.items():
    sys.modules[f"{__name__}.{_legacy_name}"] = import_module(
        f".{_canonical_name}",
        __name__,
    )
