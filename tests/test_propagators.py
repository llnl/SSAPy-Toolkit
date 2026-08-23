from types import SimpleNamespace

import numpy as np
import pytest

from ssapy_toolkit.accelerations_6dof import SpacecraftAccelConstInertial
from ssapy_toolkit.constants import EARTH_MU
from ssapy_toolkit.propagators import propagate_orbit_state
from ssapy_toolkit.propagators_6dof import propagate_6dof_high_accuracy


def test_high_accuracy_orbit_propagator_returns_near_circular_state_after_period():
    radius = 7_000_000.0
    speed = np.sqrt(EARTH_MU / radius)
    period = 2.0 * np.pi * np.sqrt(radius**3 / EARTH_MU)

    trajectory = propagate_orbit_state(
        r0=[radius, 0.0, 0.0],
        v0=[0.0, speed, 0.0],
        times=np.linspace(0.0, period, 16),
    )

    assert trajectory.r.shape == trajectory.v.shape == (16, 3)
    assert trajectory.nfev > 0
    np.testing.assert_allclose(trajectory.r[-1], trajectory.r[0], atol=30.0)
    np.testing.assert_allclose(trajectory.v[-1], trajectory.v[0], atol=0.05)


def test_high_accuracy_orbit_propagator_accepts_orbit_like_and_accel_models():
    orbit = SimpleNamespace(
        r=np.array([0.0, 0.0, 0.0]),
        v=np.array([1.0, 0.0, 0.0]),
        t=10.0,
    )

    trajectory = propagate_orbit_state(
        orbit0=orbit,
        times=[10.0, 11.0, 12.0],
        mu=0.0,
        acceleration=[
            lambda r, v, t: np.array([0.0, 1.0, 0.0]),
            SpacecraftAccelConstInertial([0.0, 0.0, 2.0]),
        ],
    )

    np.testing.assert_allclose(trajectory.r[:, 0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(trajectory.r[:, 1], [0.0, 0.5, 2.0])
    np.testing.assert_allclose(trajectory.r[:, 2], [0.0, 1.0, 4.0])
    np.testing.assert_allclose(trajectory.v[-1], [1.0, 2.0, 4.0])


def test_high_accuracy_orbit_propagator_validates_inputs():
    with pytest.raises(ValueError, match="at least two"):
        propagate_orbit_state(r0=[1, 0, 0], v0=[0, 1, 0], times=[0.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        propagate_orbit_state(r0=[1, 0, 0], v0=[0, 1, 0], times=[0.0, 0.0])
    with pytest.raises(ValueError, match="r0 and v0"):
        propagate_orbit_state(times=[0.0, 1.0])
    with pytest.raises(ValueError, match="either orbit0 or r0/v0"):
        propagate_orbit_state(
            orbit0=SimpleNamespace(r=[1, 0, 0], v=[0, 1, 0]),
            r0=[1, 0, 0],
            v0=[0, 1, 0],
            times=[0.0, 1.0],
        )


def test_high_accuracy_6dof_wrapper_sets_solve_ivp_defaults():
    trajectory = propagate_6dof_high_accuracy(
        r0=[0.0, 0.0, 0.0],
        v0=[1.0, 0.0, 0.0],
        times=[0.0, 1.0],
        inertia=np.eye(3),
        mu=0.0,
    )

    np.testing.assert_allclose(trajectory.r[:, 0], [0.0, 1.0])
