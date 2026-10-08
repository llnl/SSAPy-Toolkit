from ssapy_toolkit import constants as ssatk_constants


def test_toolkit_specific_constants_are_preserved():
    assert ssatk_constants.au_to_m == 149597870700
    assert ssatk_constants.SUN_RADIUS == 696340000.0
    assert ssatk_constants.STANDARD_GRAVITY == 9.80665
    assert ssatk_constants.G0 == ssatk_constants.STANDARD_GRAVITY
    assert ssatk_constants.SOLAR_CONSTANT == ssatk_constants.SOLAR_FLUX_1_AU
    assert ssatk_constants.EARTH_GEOMAGNETIC_REFERENCE_RADIUS_KM == 6371.2
    assert ssatk_constants.EARTH_DIPOLE_EQUATOR_FIELD == 3.12e-5
    assert ssatk_constants.EARTH_DIPOLE_POLE_FIELD == 2.0 * ssatk_constants.EARTH_DIPOLE_EQUATOR_FIELD
    assert "EARTH_MU" in ssatk_constants.__all__
    assert "au_to_m" in ssatk_constants.__all__
    assert "EARTH_DIPOLE_EQUATOR_FIELD" in ssatk_constants.__all__
