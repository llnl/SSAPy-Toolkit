import ssapy_toolkit as ssatk
from ssapy import constants as ssapy_constants
from ssapy_toolkit import constants as ssatk_constants


def test_shared_constants_are_sourced_from_ssapy():
    for name in (
        "EARTH_MU",
        "EARTH_RADIUS",
        "MOON_RADIUS",
        "RGEO",
        "WGS84_EARTH_MU",
    ):
        assert getattr(ssatk_constants, name) == getattr(ssapy_constants, name)


def test_constants_available_from_toolkit_entrypoint():
    assert ssatk.EARTH_MU == ssapy_constants.EARTH_MU
    assert ssatk.constants.RGEO == ssapy_constants.RGEO


def test_toolkit_specific_constants_are_preserved():
    assert ssatk_constants.au_to_m == 149597870700
    assert ssatk_constants.SUN_RADIUS == 696340000.0
    assert ssatk_constants.STANDARD_GRAVITY == 9.80665
    assert ssatk_constants.G0 == ssatk_constants.STANDARD_GRAVITY
    assert ssatk_constants.SOLAR_CONSTANT == ssatk_constants.SOLAR_FLUX_1_AU
    assert ssatk_constants.EARTH_GEOMAGNETIC_REFERENCE_RADIUS_KM == 6371.2
    assert "EARTH_MU" in ssatk_constants.__all__
    assert "au_to_m" in ssatk_constants.__all__


def test_solar_system_constants_are_available():
    assert ssatk_constants.AU == ssatk_constants.au_to_m
    assert ssatk_constants.AU_KM == ssatk_constants.au_to_m / 1000
    assert ssatk_constants.SUN_RADIUS_KM == ssatk_constants.SUN_RADIUS / 1000
    assert 0.52 < ssatk_constants.SUN_ANGULAR_DIAMETER_DEG < 0.54
    for planet in ssatk_constants.PLANET_NAMES:
        prefix = planet.upper()
        assert getattr(ssatk_constants, f"{prefix}_SEMI_MAJOR_AXIS_AU") > 0
        assert getattr(ssatk_constants, f"{prefix}_SEMI_MAJOR_AXIS_KM") > 0
        assert getattr(ssatk_constants, f"{prefix}_RADIUS_KM") > 0
        assert planet in ssatk_constants.PLANET_SEMI_MAJOR_AXIS_AU
        assert planet in ssatk_constants.PLANET_RADIUS_KM
    assert ssatk.SUN_RADIUS_KM == ssatk_constants.SUN_RADIUS_KM
    assert ssatk.MARS_SEMI_MAJOR_AXIS_AU == ssatk_constants.MARS_SEMI_MAJOR_AXIS_AU
    assert ssatk.SUN_NOMINAL_RADIUS_KM == ssatk_constants.SUN_NOMINAL_RADIUS_KM
