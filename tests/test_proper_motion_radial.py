"""Angular speed remains accurate when radial speed dominates."""

import numpy as np
import pytest

from ssapy_toolkit.compute.proper_motions import proper_motion


@pytest.mark.parametrize("radial_speed", [-7500.0, 7500.0])
@pytest.mark.parametrize("transverse_speed", [1e-6, 1e-4, 1.0])
def test_nearly_radial_motion_retains_transverse_component(radial_speed, transverse_speed):
    distance = 7.0e6
    actual = proper_motion(distance, 0.0, 0.0, radial_speed, transverse_speed, 0.0)
    expected = transverse_speed / distance * 206265
    assert actual == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("input_unit, divisor", [("si", 1.0), ("rebound", 31557600 * 2 * np.pi)])
def test_radial_motion_with_observer_offsets_and_units(input_unit, divisor):
    actual = proper_motion(
        7.0e6 + 100.0, 200.0, 300.0, 7500.0 + 40.0, 1e-4 - 50.0, 60.0,
        xe=100.0, ye=200.0, ze=300.0, vxe=40.0, vye=-50.0, vze=60.0,
        input_unit=input_unit,
    )
    # Account only for rounding already present in the supplied velocity.
    transverse = (1e-4 - 50.0) - (-50.0)
    assert actual == pytest.approx(transverse / 7.0e6 * 206265 / divisor, rel=1e-12, abs=0.0)


def test_general_motion_matches_independent_tangent_components():
    rng = np.random.default_rng(21)
    for _ in range(100):
        ra, dec = rng.uniform(-np.pi, np.pi), rng.uniform(-1.4, 1.4)
        radial = np.array([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])
        east = np.array([-np.sin(ra), np.cos(ra), 0.0])
        north = np.cross(radial, east)
        vr, ve, vn = rng.normal(size=3) * 1000.0
        distance = rng.uniform(7.0e6, 4.2e7)
        velocity = vr * radial + ve * east + vn * north
        actual = proper_motion(*(distance * radial), *velocity)
        assert actual == pytest.approx(np.hypot(ve, vn) / distance * 206265, rel=1e-12)


def test_zero_transverse_speed_is_zero():
    assert proper_motion(7e6, 0.0, 0.0, 7500.0, 0.0, 0.0) == 0.0
