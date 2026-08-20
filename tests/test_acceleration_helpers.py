import numpy as np
import pytest

from ssapy_toolkit.accelerations_6dof.accel_add import accel_add
from ssapy_toolkit.accelerations_6dof.accel_circularize import accel_to_circular, reset_orbit_status
from ssapy_toolkit.accelerations_6dof.accel_earth_harmonics import (
    accel_J2,
    accel_J3,
    accel_J4,
    accel_J5,
    accel_J6,
    accel_J7,
    accel_J8,
    accel_earth_harmonics,
)
from ssapy_toolkit.accelerations_6dof.accel_equatorial import accel_equatorial
from ssapy_toolkit.accelerations_6dof.accel_inclination import accel_inclination
from ssapy_toolkit.accelerations_6dof.accel_plane import accel_plane
from ssapy_toolkit.accelerations_6dof.accel_point_earth import accel_point_earth
from ssapy_toolkit.accelerations_6dof.accel_radial import accel_radial
from ssapy_toolkit.accelerations_6dof.accel_uniform_earth import accel_uniform_earth
from ssapy_toolkit.accelerations_6dof.accel_velocity import accel_velocity
from ssapy_toolkit.constants import EARTH_MU, EARTH_RADIUS, RGEO
from ssapy_toolkit.dynamics import Spacecraft


def test_accel_add_supports_time_and_position_only_functions():
    def with_time(r, t):
        return np.array([t, r[0], 0.0])

    def without_time(r):
        return np.array([0.0, r[1], 1.0])

    combined = accel_add(with_time, without_time)

    np.testing.assert_allclose(combined(np.array([2.0, 3.0, 4.0]), t=5.0), [5.0, 5.0, 1.0])


def test_directional_acceleration_helpers_and_zero_cases():
    r = np.array([3.0, 4.0, 0.0])
    v = np.array([-4.0, 3.0, 0.0])

    np.testing.assert_allclose(accel_radial(r, 2.0), [1.2, 1.6, 0.0])
    np.testing.assert_allclose(accel_radial(np.zeros(3), 2.0), np.zeros(3))

    np.testing.assert_allclose(accel_velocity(v, 10.0), [-8.0, 6.0, 0.0])
    np.testing.assert_allclose(accel_velocity(np.zeros(3), 10.0), np.zeros(3))

    np.testing.assert_allclose(accel_plane(r, v, 0.5), v / np.linalg.norm(v) * 0.5)
    np.testing.assert_allclose(accel_plane(r, r, 0.5), np.zeros(3))

    np.testing.assert_allclose(accel_equatorial([1.0, 0.0, 0.0], v, 0.25), [0.0, 0.25, 0.0])
    np.testing.assert_allclose(accel_equatorial([0.0, 0.0, 1.0], v, 0.25), np.zeros(3))

    np.testing.assert_allclose(accel_inclination([1.0, 0.0, 0.0], v, 0.75), [0.0, 0.0, 0.75])
    np.testing.assert_allclose(accel_inclination([0.0, 0.0, 1.0], v, 0.75), np.zeros(3))


def test_legacy_acceleration_helpers_accept_spacecraft_state():
    sat = Spacecraft(
        r=[3.0, 4.0, 0.0],
        v=[-4.0, 3.0, 0.0],
        inertia=np.eye(3),
        mass=100.0,
    )

    np.testing.assert_allclose(accel_radial(sat, 2.0), [1.2, 1.6, 0.0])
    np.testing.assert_allclose(accel_velocity(sat, 10.0), [-8.0, 6.0, 0.0])
    np.testing.assert_allclose(accel_plane(sat, 0.5), sat.v / np.linalg.norm(sat.v) * 0.5)
    np.testing.assert_allclose(accel_equatorial(sat, 0.25), [-0.2, 0.15, 0.0])
    np.testing.assert_allclose(accel_inclination(sat, 0.75), [0.0, 0.0, 0.75])

    expected_gravity = -EARTH_MU * sat.r / np.linalg.norm(sat.r) ** 3
    np.testing.assert_allclose(accel_point_earth(sat), expected_gravity)
    np.testing.assert_allclose(
        accel_add(accel_point_earth, lambda r, v, t: accel_radial(r, 0.0))(sat, t=0.0),
        expected_gravity,
    )


def test_uniform_earth_gravity_inside_and_outside():
    outside = np.array([2.0 * EARTH_RADIUS, 0.0, 0.0])
    inside = np.array([0.5 * EARTH_RADIUS, 0.0, 0.0])

    np.testing.assert_allclose(accel_uniform_earth(outside), [-EARTH_MU / outside[0] ** 2, 0.0, 0.0])
    np.testing.assert_allclose(accel_uniform_earth(inside), [-EARTH_MU * inside[0] / EARTH_RADIUS**3, 0.0, 0.0])


def test_accel_to_circular_latches_after_tolerance_and_can_reset():
    reset_orbit_status()
    r = np.array([RGEO, 0.0, 0.0])
    circular_v = np.array([0.0, np.sqrt(EARTH_MU / RGEO), 0.0])
    command = accel_to_circular(r, circular_v + np.array([0.0, 100.0, 0.0]), thrust=0.01, tol=1.0)
    assert np.isclose(np.linalg.norm(command), 0.01)
    assert command[1] < 0.0

    np.testing.assert_allclose(accel_to_circular(r, circular_v, thrust=0.01, tol=1.0), np.zeros(3))
    np.testing.assert_allclose(accel_to_circular(r, circular_v + np.array([0.0, 100.0, 0.0]), thrust=0.01, tol=1.0), np.zeros(3))

    reset_orbit_status()
    np.testing.assert_allclose(accel_to_circular(np.zeros(3), circular_v, thrust=0.01), np.zeros(3))
    np.testing.assert_allclose(accel_to_circular([EARTH_RADIUS, 0.0, 0.0], circular_v, thrust=0.01), np.zeros(3))

    sat = Spacecraft(r=RGEO * np.array([1.0, 0.0, 0.0]), v=circular_v + [0.0, 100.0, 0.0], inertia=np.eye(3))
    command = accel_to_circular(sat, thrust=0.01, tol=1.0)
    assert np.isclose(np.linalg.norm(command), 0.01)


def test_earth_harmonics_are_finite_and_reject_subsurface_positions():
    r = np.array([RGEO, 0.1 * RGEO, 0.25 * RGEO])
    harmonics = [accel_J2, accel_J3, accel_J4, accel_J5, accel_J6, accel_J7, accel_J8]
    components = [func(r) for func in harmonics]

    for component in components:
        assert component.shape == (3,)
        assert np.all(np.isfinite(component))

    expected = -EARTH_MU * r / np.linalg.norm(r) ** 3 + np.sum(components, axis=0)
    np.testing.assert_allclose(accel_earth_harmonics(r), expected)
    np.testing.assert_allclose(accel_earth_harmonics(Spacecraft(r=r, v=[0, 0, 0], inertia=np.eye(3))), expected)

    with pytest.raises(ValueError, match="below Earth's surface"):
        accel_J2([EARTH_RADIUS / 2.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="below Earth's surface"):
        accel_earth_harmonics([EARTH_RADIUS / 2.0, 0.0, 0.0])
