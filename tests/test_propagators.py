from types import SimpleNamespace

import numpy as np
import pytest

from ssapy_toolkit.accelerations_6dof import (
    SpacecraftAccelConstInertial,
    SpacecraftManeuverAccel,
    SpacecraftReactionWheelTorque,
    SpacecraftThrusterAccel,
)
from ssapy_toolkit.constants import AU, EARTH_MU, STANDARD_GRAVITY
from ssapy_toolkit.propagators_6dof import Spacecraft
from ssapy_toolkit.environment import SpaceEnvironment
from ssapy_toolkit.propagators_orbit import (
    propagate_orbit_state,
    propagate_orbit_state_with_stm,
)
from ssapy_toolkit.propagators_6dof import (
    propagate_6dof_high_accuracy,
    propagate_spacecraft_high_accuracy,
    propagate_spacecraft_segments,
)
from ssapy_toolkit.satellites import SpacecraftBody, Thruster, reaction_wheel_triplet


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


def test_orbit_stm_matches_finite_difference_of_propagated_state():
    radius = 7_000_000.0
    speed = np.sqrt(EARTH_MU / radius)
    times = np.linspace(0.0, 900.0, 5)
    state = propagate_orbit_state_with_stm(
        r0=[radius, 0.0, 0.0],
        v0=[0.0, speed, 0.0],
        times=times,
    )
    delta = np.array([0.2, -0.1, 0.05, 1.0e-4, -2.0e-4, 3.0e-4])
    plus = propagate_orbit_state(
        r0=np.array([radius, 0.0, 0.0]) + delta[:3],
        v0=np.array([0.0, speed, 0.0]) + delta[3:],
        times=times,
    )
    minus = propagate_orbit_state(
        r0=np.array([radius, 0.0, 0.0]) - delta[:3],
        v0=np.array([0.0, speed, 0.0]) - delta[3:],
        times=times,
    )
    predicted = np.einsum("tij,j->ti", state.stm, delta)
    actual = 0.5 * np.column_stack((plus.r - minus.r, plus.v - minus.v))
    np.testing.assert_allclose(actual, predicted, rtol=3.0e-4, atol=2.0e-7)


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
    assert trajectory.nfev > 0
    assert isinstance(trajectory.message, str)


def test_high_accuracy_spacecraft_propagator_assembles_models_and_mass_flow():
    body = SpacecraftBody.box(name="bus", mass=20.0, size=(1.0, 1.0, 1.0)).with_thrusters(
        Thruster(thrust=2.0, direction_body=[1.0, 0.0, 0.0], isp=200.0),
        append=False,
    )
    spacecraft = Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        q=[1.0, 0.0, 0.0, 0.0],
        omega=[0.0, 0.0, 0.0],
        body=body,
    )

    trajectory = propagate_spacecraft_high_accuracy(
        spacecraft,
        times=[0.0, 5.0],
        models=[SpacecraftThrusterAccel()],
        mu=0.0,
    )

    expected_mass = spacecraft.mass - 2.0 / (200.0 * STANDARD_GRAVITY) * 5.0
    assert trajectory.mass[-1] == pytest.approx(expected_mass)
    assert trajectory.v[-1, 0] > 0.0


def test_high_accuracy_spacecraft_propagator_accepts_environment_models():
    body = SpacecraftBody.box(name="plate", mass=10.0, size=(1.0, 1.0, 1.0))
    spacecraft = Spacecraft(
        r=[7_000_000.0, 0.0, 0.0],
        v=[0.0, 7_500.0, 0.0],
        body=body,
    )
    environment = SpaceEnvironment(
        sun_position_model=[-AU, 0.0, 0.0],
        atmosphere_density_model=0.0,
        eclipse_model=None,
    )

    trajectory = propagate_spacecraft_high_accuracy(
        spacecraft,
        times=[0.0, 1.0],
        environment=environment,
        environment_models={"drag": True, "solar_radiation": True},
        mu=0.0,
    )

    assert trajectory.r.shape == (2, 3)
    assert trajectory.v[-1, 0] > 0.0


def test_high_accuracy_spacecraft_propagator_accepts_environment_preset_string():
    body = SpacecraftBody.box(name="plate", mass=10.0, size=(1.0, 1.0, 1.0))
    spacecraft = Spacecraft(
        r=[7_000_000.0, 0.0, 0.0],
        v=[0.0, 7_500.0, 0.0],
        body=body,
    )
    environment = SpaceEnvironment(
        sun_position_model=[-AU, 0.0, 0.0],
        moon_position_model=[384_400_000.0, 0.0, 0.0],
        atmosphere_density_model=0.0,
        eclipse_model=None,
    )

    trajectory = propagate_spacecraft_high_accuracy(
        spacecraft,
        times=[0.0, 1.0],
        environment=environment,
        environment_models="earth_orbit",
        mu=0.0,
    )

    assert trajectory.r.shape == (2, 3)
    assert trajectory.nfev > 0


def test_high_accuracy_environment_preset_avoids_ssapy_force_double_counting(monkeypatch):
    captured = {}

    def fake_ssapy_stack(**options):
        captured.update(options)
        return SpacecraftAccelConstInertial([0.0, 0.0, 0.0])

    monkeypatch.setattr(
        "ssapy_toolkit.accelerations_6dof.make_ssapy_perturbation_acceleration",
        fake_ssapy_stack,
    )
    spacecraft = Spacecraft(
        r=[7_000_000.0, 0.0, 0.0],
        v=[0.0, 7_500.0, 0.0],
        body=SpacecraftBody.box(name="bus", mass=10.0, size=(1.0, 1.0, 1.0)),
    )
    environment = SpaceEnvironment(
        sun_position_model=[-AU, 0.0, 0.0],
        moon_position_model=[384_400_000.0, 0.0, 0.0],
        atmosphere_density_model=0.0,
        magnetic_field_model="zero",
        eclipse_model=None,
    )

    trajectory = propagate_spacecraft_high_accuracy(
        spacecraft,
        times=[0.0, 1.0],
        environment=environment,
        environment_models="leo",
        ssapy_perturbations=True,
        mu=0.0,
    )

    assert trajectory.r.shape == (2, 3)
    assert captured["include_drag"] is False
    assert captured["include_solar_radiation"] is False


def test_high_accuracy_spacecraft_segments_chain_state_and_mass():
    import ssapy_toolkit as ssatk

    assert ssatk.propagate_spacecraft_segments is propagate_spacecraft_segments
    spacecraft = Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        inertia=np.eye(3),
        mass=100.0,
    )
    burn = SpacecraftManeuverAccel(
        1.0,
        frame="gcrf",
        direction=[1.0, 0.0, 0.0],
        isp=100.0,
        start=1.0,
        stop=2.0,
    )

    trajectory = propagate_spacecraft_segments(
        spacecraft,
        [
            {"times": [0.0, 1.0], "mu": 0.0},
            {"times": [1.0, 2.0], "models": [burn], "mu": 0.0},
        ],
    )

    np.testing.assert_allclose(trajectory.t, [0.0, 1.0, 2.0])
    assert trajectory.v[-1, 0] > 0.0
    assert trajectory.mass is not None
    assert trajectory.mass[-1] < trajectory.mass[0]
    assert trajectory.nfev > 0


def test_high_accuracy_spacecraft_segments_preserve_wheel_momentum():
    body = SpacecraftBody.cubesat(1, mass=10.0).with_reaction_wheels(
        *reaction_wheel_triplet(max_torque=0.1)
    )
    spacecraft = Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        body=body,
    )

    trajectory = propagate_spacecraft_segments(
        spacecraft,
        [
            {"times": [0.0, 1.0], "mu": 0.0},
            {
                "times": [1.0, 2.0],
                "models": [SpacecraftReactionWheelTorque([0.0, 0.0, 0.03])],
                "mu": 0.0,
            },
        ],
    )

    assert trajectory.wheel_momentum.shape == (3, 3)
    np.testing.assert_allclose(trajectory.wheel_momentum[0], [0.0, 0.0, 0.0])
    assert trajectory.wheel_momentum[-1, 2] < 0.0


def test_high_accuracy_spacecraft_segments_validate_continuity():
    spacecraft = Spacecraft(r=[0.0, 0.0, 0.0], v=[0.0, 0.0, 0.0], inertia=np.eye(3))

    with pytest.raises(ValueError, match="current spacecraft epoch"):
        propagate_spacecraft_segments(
            spacecraft,
            [
                {"times": [0.0, 1.0], "mu": 0.0},
                {"times": [2.0, 3.0], "mu": 0.0},
            ],
        )
