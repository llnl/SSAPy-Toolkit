import importlib

import numpy as np
import pytest

from ssapy_toolkit.compute.lyapunov_exponent import lyapunov_exponent_from_statevectors
from ssapy_toolkit.compute.proper_motions import proper_motion, proper_motion_ra_dec
from ssapy_toolkit.compute.segment_intersection import segment_intersects_sphere
from ssapy_toolkit.constants import EARTH_MU
from ssapy_toolkit.coordinates import equatorial_and_ecliptic
from ssapy_toolkit.coordinates.cartesian_to_cylindrical import cart_to_cyl
from ssapy_toolkit.coordinates.cartesian_to_spherical import cart2sph_deg
from ssapy_toolkit.coordinates.gcrf_to_ntw import gcrf_to_ntw
from ssapy_toolkit.coordinates.ntw_to_gcrf import ntw_to_gcrf, ntw_to_gcrf_matrix
from ssapy_toolkit.coordinates.unit_conversions import (
    deg0to360,
    deg0to360array,
    deg90to90,
    deg90to90array,
    dms_to_deg,
    dms_to_rad,
    rad0to2pi,
)
from ssapy_toolkit.propagators import int_utils
from ssapy_toolkit.orbital_mechanics import misc

rk4_module = importlib.import_module("ssapy_toolkit.propagators.rk4")
leapfrog_module = importlib.import_module("ssapy_toolkit.propagators.leap_frog")
eqecl = equatorial_and_ecliptic


def test_orbital_misc_formula_helpers():
    mu = EARTH_MU
    r = 7000e3
    v_circ = np.sqrt(mu / r)
    assert np.isclose(misc.escape_velocity(mu, r), np.sqrt(2) * v_circ)
    assert np.isclose(misc.circular_velocity(mu, r), v_circ)
    assert np.isclose(misc.vis_viva(mu, r, r), v_circ)
    assert np.isclose(misc.specific_orbital_energy(mu, r, v_circ), -mu / (2 * r))
    np.testing.assert_allclose(misc.specific_angular_momentum([r, 0, 0], [0, v_circ, 0]), [0, 0, r * v_circ])
    np.testing.assert_allclose(misc.eccentricity_vector(np.array([r, 0, 0]), np.array([0, v_circ, 0]), mu), [0, 0, 0], atol=1e-12)

    a, e, inc, raan, argp, nu, M = misc.orbital_elements_from_state(np.array([r, 0, 0]), np.array([0, v_circ, 0]), mu)
    assert np.isclose(a, r)
    assert e < 1e-12
    assert inc == 0.0
    assert raan == argp == nu == 0.0
    assert M is not None

    E = misc.kepler_E_from_M(0.5, 0.1)
    assert np.isclose(E - 0.1 * np.sin(E), 0.5)
    assert np.isclose(misc.kepler_E_from_M_from_nu(0.0, 0.1), 0.0)
    assert np.isclose(misc.orbital_period(r, mu), 2 * np.pi * np.sqrt(r**3 / mu))
    assert misc.orbital_period(-1.0, mu) is None
    dv1, dv2, total = misc.hohmann_transfer_delta_v(r, 2 * r, mu)
    assert np.isclose(total, dv1 + dv2)
    vals = misc.bi_elliptic_transfer_delta_v(r, 2 * r, 4 * r, mu)
    assert len(vals) == 4
    assert np.isclose(vals[-1], sum(vals[:3]))
    assert np.isclose(misc.plane_change_delta_v(10.0, 0.0, np.pi / 3), 10.0)
    assert np.isclose(misc.sphere_of_influence_radius(1.0, 1.0, 32.0), 1.0 * (1.0 / 32.0) ** (2.0 / 5.0))


def test_coordinate_conversion_helpers():
    radius, theta, z = cart_to_cyl(3.0, 4.0, 5.0)
    assert radius == 5.0
    assert np.isclose(theta, np.arctan2(4.0, 3.0))
    assert z == 5.0
    az, el, r = cart2sph_deg(0.0, 1.0, 1.0)
    assert np.isclose(az, 90.0)
    assert np.isclose(el, 45.0)
    assert np.isclose(r, np.sqrt(2.0))

    assert np.isclose(dms_to_rad("180d"), np.pi)
    assert dms_to_deg(["0d", "90d"]) == [0.0, 90.0]
    np.testing.assert_allclose(rad0to2pi([-np.pi, 3 * np.pi]), [np.pi, np.pi])
    assert deg0to360(-90) == 270.0
    assert deg0to360array([-1, 360]) == [359.0, 0.0]
    assert deg90to90(100) == -80.0
    assert deg90to90array([100, -100]) == [-80.0, 80.0]

    r_vec = np.array([1.0, 0.0, 0.0])
    v_vec = np.array([0.0, 1.0, 0.0])
    matrix = ntw_to_gcrf_matrix(r_vec, v_vec)
    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(ntw_to_gcrf([1, 2, 3], r_vec, v_vec), [1, 2, 3], atol=1e-12)
    np.testing.assert_allclose(gcrf_to_ntw([1, 2, 3], r_vec, v_vec), [1, 2, 3], atol=1e-12)

    assert equatorial_and_ecliptic.equatorial_to_ecliptic is eqecl.equatorial_to_ecliptic

    xq, yq, zq = eqecl.ecliptic_xyz_to_equatorial_xyz(1.0, 2.0, 3.0)
    xc, yc, zc = eqecl.equatorial_xyz_to_ecliptic_xyz(xq, yq, zq)
    np.testing.assert_allclose([xc, yc, zc], [1.0, 2.0, 3.0])
    lon, lat = eqecl.xyz_to_ecliptic(1.0, 0.0, 0.0, degrees=True)
    assert np.isclose(lon, 0.0)
    assert np.isclose(lat, 0.0)
    ra, dec = eqecl.xyz_to_equatorial(1.0, 0.0, 0.0, degrees=True)
    assert np.isclose(ra, 0.0)
    assert np.isclose(dec, 0.0)
    ra2, dec2 = eqecl.ecliptic_to_equatorial(*eqecl.equatorial_to_ecliptic(ra, dec, degrees=True), degrees=True)
    assert np.isfinite(ra2)
    assert np.isfinite(dec2)


def test_propagator_profile_and_simple_motion(monkeypatch):
    t = np.arange(5.0)
    np.testing.assert_array_equal(int_utils.build_profile(None, t), np.zeros(5))
    np.testing.assert_array_equal(int_utils.build_profile(2.0, t), np.full(5, 2.0))
    np.testing.assert_array_equal(int_utils.build_profile([1, 2, 3, 4, 5], t), [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(int_utils.build_profile({"start": 1, "end": 3, "thrust": 4}, t), [0, 4, 4, 0, 0])
    np.testing.assert_array_equal(int_utils.build_profile([(1, 3, 2), (3, 1)], t), [0, 2, 2, 1, 1])
    with pytest.raises(ValueError):
        int_utils.build_profile([(1, 2, 3, 4)], t)
    with pytest.raises(TypeError):
        int_utils.build_profile(object(), t)

    class FakeBody:
        def position(self, times):
            times = np.asarray(times, dtype=float)
            return np.vstack((times, 2 * times, 3 * times))

    monkeypatch.setattr("ssapy.get_body", lambda name: FakeBody())
    interp = int_utils.precompute_third_body_positions(np.array([0.0, 1.0, 2.0, 3.0]), "moon")
    np.testing.assert_allclose(interp(np.array([0.5, 1.5])), [[0.5, 1.0, 1.5], [1.5, 3.0, 4.5]])

    monkeypatch.setattr(rk4_module, "accel_point_moon", lambda r, t: np.zeros(3))
    monkeypatch.setattr(rk4_module, "accel_point_sun", lambda r, t: np.zeros(3))
    r_hist, v_hist = rk4_module.rk4([0, 0, 0], [1, 0, 0], np.array([0.0, 1.0, 2.0]), accel_gravity=lambda r: np.zeros(3))
    np.testing.assert_allclose(r_hist[:, 0], [0, 1, 2])
    np.testing.assert_allclose(v_hist[:, 0], [1, 1, 1])

    monkeypatch.setattr(leapfrog_module, "accel_point_earth", lambda r: np.zeros(3))
    r_hist, v_hist = leapfrog_module.leapfrog([0, 0, 0], [1, 0, 0], np.array([0.0, 1.0, 2.0]), stop_altitude_m=-1e9)
    np.testing.assert_allclose(r_hist[:, 0], [0, 1, 2])
    np.testing.assert_allclose(v_hist[:, 0], [1, 1, 1])
    with pytest.raises(ValueError, match="at least 2"):
        leapfrog_module.leapfrog([0, 0, 0], [1, 0, 0], np.array([0.0]))
    with pytest.raises(ValueError, match="Non-uniform"):
        leapfrog_module.leapfrog([0, 0, 0], [1, 0, 0], np.array([0.0, 1.0, 3.0]), stop_altitude_m=-1e9)


def test_proper_motion_segment_intersection_and_lyapunov():
    assert np.isclose(proper_motion(1.0, 0.0, 0.0, 0.0, 1.0, 0.0), 206265.0)
    assert np.isnan(proper_motion(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    with pytest.warns(UserWarning, match="input_unit"):
        assert proper_motion(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, input_unit="bad") is None

    r = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    v = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    pmra, pmdec = proper_motion_ra_dec(r=r, v=v)
    assert pmra.shape == pmdec.shape == (2,)
    pmra_rebound, pmdec_rebound = proper_motion_ra_dec(r=r, v=v, input_unit="rebound")
    assert np.all(np.abs(pmra_rebound) < np.abs(pmra))
    with pytest.raises(ValueError):
        proper_motion_ra_dec(x=1, y=2)
    with pytest.warns(UserWarning, match="input_unit"):
        assert proper_motion_ra_dec(r=r, v=v, input_unit="bad") is None

    assert segment_intersects_sphere([-2, 0, 0], [2, 0, 0], radius=1.0)
    assert not segment_intersects_sphere([2, 0, 0], [3, 0, 0], radius=1.0)
    assert segment_intersects_sphere([1, 0, 0], [2, 0, 0], radius=1.0, atol=0.0)

    n = 12
    times = np.arange(n, dtype=float)
    r_series = np.column_stack((times, np.sin(times), np.cos(times)))
    v_series = np.column_stack((np.ones(n), np.cos(times), -np.sin(times)))
    lle, t_curve, mean_log, diag = lyapunov_exponent_from_statevectors(r_series, v_series, dt=0.5, theiler_window=1, max_horizon=4, trim_percentile=90)
    assert np.isfinite(lle)
    assert t_curve.shape == mean_log.shape
    assert diag["K"] <= 4
    lle2, _, _ = lyapunov_exponent_from_statevectors(r_series, v_series, dt=1.0, theiler_window=0, return_diagnostics=False)
    assert np.isfinite(lle2)

    with pytest.raises(ValueError, match="shaped"):
        lyapunov_exponent_from_statevectors(np.zeros((3, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="same length"):
        lyapunov_exponent_from_statevectors(np.zeros((3, 3)), np.zeros((4, 3)))
    with pytest.raises(ValueError, match="positive"):
        lyapunov_exponent_from_statevectors(np.zeros((3, 3)), np.zeros((3, 3)), dt=0)
    with pytest.raises(ValueError, match=">= 0"):
        lyapunov_exponent_from_statevectors(np.zeros((3, 3)), np.zeros((3, 3)), theiler_window=-1)
    with pytest.raises(ValueError, match="at least 3"):
        lyapunov_exponent_from_statevectors(np.zeros((2, 3)), np.zeros((2, 3)))
