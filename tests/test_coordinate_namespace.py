import importlib

import ssapy_toolkit.coordinates as coordinates


def test_canonical_coordinate_modules_and_legacy_aliases_import():
    canonical = {
        "angle_units": "rad0to2pi",
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
    legacy = {
        "cartesian_to_cylindrical": "cart_to_cyl",
        "cartesian_to_spherical": "cart2sph_deg",
        "earth_trojan_sim": "inert2rot",
        "equatorial_and_ecliptic": "equatorial_to_ecliptic",
        "equitorial_and_ecliptic": "equatorial_to_ecliptic",
        "gcrf_to_itrf": "gcrf_to_itrf",
        "itrf_to_gcrf": "itrf_to_gcrf",
        "gcrf_to_llh": "gcrf_to_llh",
        "gcrf_to_lonlat": "gcrf_to_lonlat",
        "llh_to_gcrf": "llh_to_gcrf",
        "lon_lat_bbox": "bbox_min",
        "lonlat_perigee": "lonlat_perigee",
        "on_sky_distance": "lonlat_distance",
        "surface_rv": "surface_rv",
        "gcrf_to_lunar": "gcrf_to_lunar",
        "lunar_position": "get_lunar_rv",
        "gcrf_to_ntw": "gcrf_to_ntw",
        "ntw_to_gcrf": "ntw_to_gcrf",
        "j2000_to_gcrf": "j2000_to_gcrf",
        "local_and_equatorial": "equatorial_to_horizontal",
        "local_and_equitorial": "equatorial_to_horizontal",
        "sky_angles": "ra_dec",
        "unit_conversions": "rad0to2pi",
        "v_from_r": "v_from_r",
    }

    for module_name, attr in canonical.items():
        module = importlib.import_module(f"ssapy_toolkit.coordinates.{module_name}")
        assert hasattr(module, attr)
        assert hasattr(coordinates, attr)

    for module_name, attr in legacy.items():
        module = importlib.import_module(f"ssapy_toolkit.coordinates.{module_name}")
        assert hasattr(module, attr)
