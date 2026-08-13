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
    assert "EARTH_MU" in ssatk_constants.__all__
    assert "au_to_m" in ssatk_constants.__all__
