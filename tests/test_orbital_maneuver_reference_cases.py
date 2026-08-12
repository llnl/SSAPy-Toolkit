"""Reference checks for orbital maneuver solvers.

These tests use closed-form two-body maneuver equations from standard
astrodynamics texts (Hohmann transfer, bi-elliptic transfer, simple plane
change, and Tsiolkovsky rocket equation). The formulas are repeated here rather
than imported from SSATK helpers so the tests verify solver outputs against an
independent implementation.

Benchmark provenance
--------------------
The numeric anchors are recomputed from the cited closed-form equations using
SSAPy/SSATK's Earth constants (``EARTH_MU``, ``EARTH_RADIUS``, ``RGEO``, and
``VGEO``), not copied from a table. Sources for the formulas:

* Hohmann and bi-elliptic transfers: Curtis, ``Orbital Mechanics for
  Engineering Students``, 4th ed., Secs. 6.3--6.4; Bate, Mueller, and White,
  ``Fundamentals of Astrodynamics``, Dover, 1971, Ch. 6.
* Plane-change impulse: Curtis, Sec. 6.6, with ``Delta-v = 2 v sin(Delta i/2)``.
* Rocket equation / propellant mass: Tsiolkovsky ideal rocket equation,
  commonly stated as ``Delta-v = Isp g0 ln(m0/mf)``; see Sutton and Biblarz,
  ``Rocket Propulsion Elements``, 9th ed., Ch. 2.
"""

import numpy as np
import pytest

from ssapy_toolkit.constants import EARTH_MU, EARTH_RADIUS, RGEO, VGEO
from ssapy_toolkit.orbital_mechanics._transfer_result import G0, maneuver_burn
from ssapy_toolkit.orbital_mechanics.misc import plane_change_delta_v
from ssapy_toolkit.orbital_mechanics.transfer_bielliptic import transfer_bielliptic
from ssapy_toolkit.orbital_mechanics.transfer_hohmann import transfer_hohmann


def _circular_speed(radius, mu=EARTH_MU):
    return np.sqrt(mu / radius)


def _vis_viva_speed(radius, semi_major_axis, mu=EARTH_MU):
    return np.sqrt(mu * (2.0 / radius - 1.0 / semi_major_axis))


def _reference_hohmann(radius1, radius2, mu=EARTH_MU):
    semi_major_axis = 0.5 * (radius1 + radius2)
    dv_depart = abs(_vis_viva_speed(radius1, semi_major_axis, mu) - _circular_speed(radius1, mu))
    dv_arrive = abs(_circular_speed(radius2, mu) - _vis_viva_speed(radius2, semi_major_axis, mu))
    tof = np.pi * np.sqrt(semi_major_axis**3 / mu)
    return dv_depart, dv_arrive, dv_depart + dv_arrive, tof


def _reference_bielliptic(radius1, radius2, intermediate_radius, mu=EARTH_MU):
    semi_major_axis1 = 0.5 * (radius1 + intermediate_radius)
    semi_major_axis2 = 0.5 * (radius2 + intermediate_radius)
    dv1 = abs(_vis_viva_speed(radius1, semi_major_axis1, mu) - _circular_speed(radius1, mu))
    dv2 = abs(
        _vis_viva_speed(intermediate_radius, semi_major_axis2, mu)
        - _vis_viva_speed(intermediate_radius, semi_major_axis1, mu)
    )
    dv3 = abs(_circular_speed(radius2, mu) - _vis_viva_speed(radius2, semi_major_axis2, mu))
    tof = np.pi * np.sqrt(semi_major_axis1**3 / mu) + np.pi * np.sqrt(semi_major_axis2**3 / mu)
    return dv1, dv2, dv3, dv1 + dv2 + dv3, tof


def test_300km_leo_to_geo_hohmann_matches_textbook_reference():
    """Classic 300 km LEO -> GEO Hohmann: about 3.893 km/s and 5.275 h."""
    # Textbook Hohmann benchmark: use the closed-form two-impulse transfer
    # equations cited in the module docstring, with SSAPy Earth's mu/radii.
    radius_leo = EARTH_RADIUS + 300e3
    radius_geo = RGEO
    expected_dv1, expected_dv2, expected_total, expected_tof = _reference_hohmann(radius_leo, radius_geo)

    result = transfer_hohmann(radius_leo, radius_geo, samples=16)

    np.testing.assert_allclose(result["delta_v_magnitudes"], [expected_dv1, expected_dv2], rtol=1e-12)
    assert result["delta_v_total"] == pytest.approx(expected_total, rel=1e-12)
    assert result["tof"] == pytest.approx(expected_tof, rel=1e-12)

    assert expected_dv1 / 1e3 == pytest.approx(2.426, rel=5e-4)
    assert expected_dv2 / 1e3 == pytest.approx(1.467, rel=5e-4)
    assert expected_total / 1e3 == pytest.approx(3.893, rel=5e-4)
    assert expected_tof / 3600.0 == pytest.approx(5.275, rel=5e-4)

    assert result["diagnostics"]["radius1"] == pytest.approx(radius_leo)
    assert result["diagnostics"]["radius2"] == pytest.approx(radius_geo)
    np.testing.assert_allclose(result["trajectory"]["r"][0], result["initial"]["r"])
    np.testing.assert_allclose(result["trajectory"]["r"][-1], result["final"]["r"], atol=1e-6)


def test_bielliptic_known_high_radius_ratio_beats_hohmann():
    """For radius ratios above ~11.94, a high-apogee bi-elliptic can win."""
    # Curtis Sec. 6.4 gives the classical result that bi-elliptic transfers can
    # beat Hohmann transfers once r2/r1 exceeds about 11.94 for sufficiently
    # large intermediate apoapsis. This test uses r2/r1 = 16 and rb/r1 = 1000.
    radius1 = 7000e3
    radius2 = 16.0 * radius1
    intermediate_radius = 1000.0 * radius1
    expected = _reference_bielliptic(radius1, radius2, intermediate_radius)
    hohmann_total = _reference_hohmann(radius1, radius2)[2]

    result = transfer_bielliptic(
        radius1,
        radius2,
        intermediate_radius=intermediate_radius,
        samples_per_arc=8,
    )

    np.testing.assert_allclose(result["delta_v_magnitudes"], expected[:3], rtol=1e-12)
    assert result["delta_v_total"] == pytest.approx(expected[3], rel=1e-12)
    assert result["tof"] == pytest.approx(expected[4], rel=1e-12)
    assert result["delta_v_total"] < hohmann_total
    assert (hohmann_total - result["delta_v_total"]) == pytest.approx(134.13, rel=5e-3)
    assert result["delta_v_total"] / 1e3 == pytest.approx(3.912, rel=5e-4)


def test_geo_plane_change_matches_closed_form_reference():
    """A 28.5 deg GEO plane change costs about 1.514 km/s at GEO speed."""
    # Closed-form impulsive plane change at constant speed: Delta-v =
    # 2 v sin(Delta i / 2). 28.5 deg is the standard Cape Canaveral latitude
    # inclination benchmark often used for GEO insertion plane-change examples.
    delta_i = np.deg2rad(28.5)
    expected = 2.0 * VGEO * np.sin(delta_i / 2.0)

    assert plane_change_delta_v(VGEO, 0.0, delta_i) == pytest.approx(expected, rel=1e-12)
    assert expected / 1e3 == pytest.approx(1.514, rel=5e-4)


def test_maneuver_burn_matches_tsiolkovsky_and_constant_thrust_reference():
    """A 100 m/s, 300 s Isp burn from 1000 kg consumes about 33.4 kg."""
    # Independent rocket-equation check: mf = m0 exp(-Delta-v / (Isp g0)), so
    # propellant = m0 - mf. Duration follows constant acceleration F/m.
    state = {
        "r": np.array([7000e3, 0.0, 0.0]),
        "v": np.array([0.0, _circular_speed(7000e3), 0.0]),
        "t": 10.0,
    }
    burn = maneuver_burn(
        name="reference_burn",
        state=state,
        delta_v=np.array([0.0, 100.0, 0.0]),
        thrust=200.0,
        mass=1000.0,
        isp=300.0,
    )

    expected_duration = 100.0 / (200.0 / 1000.0)
    expected_propellant = 1000.0 * (1.0 - np.exp(-100.0 / (300.0 * G0)))

    assert burn["delta_v_mag"] == pytest.approx(100.0)
    assert burn["acceleration_mag"] == pytest.approx(0.2)
    assert burn["duration"] == pytest.approx(expected_duration)
    assert burn["propellant_mass"] == pytest.approx(expected_propellant, rel=1e-12)
    assert burn["propellant_mass"] == pytest.approx(33.42, rel=5e-4)
