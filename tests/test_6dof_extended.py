import numpy as np
import pytest

from ssapy_toolkit.propagators_6dof import (
    FlexibleMode,
    HingedAppendage,
    SloshMode,
    propagate_6dof_extended,
)


def test_extended_modes_propagate_and_couple_to_rigid_body():
    result = propagate_6dof_extended(
        times=np.linspace(0, 0.2, 5), inertia=np.eye(3), mu=0,
        r0=[7e6, 0, 0], v0=[0, 7500, 0], q0=[1, 0, 0, 0],
        hinge=HingedAppendage([0, 0, 1], 2, stiffness=4, angle0=0.1),
        flexible=FlexibleMode([1, 0, 0], 1, 3, displacement0=0.1),
        slosh=SloshMode([0, 1, 0], 1, 2, lever_arm_body=[1, 0, 0], displacement0=0.1),
        bus_mass=10,
    )
    assert result.trajectory.r.shape == (5, 3)
    assert result.hinge.shape == result.flexible.shape == result.slosh.shape == (5, 2)
    assert not np.allclose(result.trajectory.omega[0], result.trajectory.omega[-1])
    assert np.linalg.norm(result.trajectory.v[0] - result.trajectory.v[-1]) > 1e-6
    np.testing.assert_allclose(np.linalg.norm(result.trajectory.q, axis=1), 1.0, atol=1e-12)


def test_extended_modes_accept_multiple_modes_per_category():
    result = propagate_6dof_extended(
        times=np.linspace(0, 0.1, 3), inertia=np.eye(3), mu=0,
        r0=[7e6, 0, 0], v0=[0, 7500, 0], q0=[1, 0, 0, 0], bus_mass=10,
        hinge=(
            HingedAppendage([0, 0, 1], 1, stiffness=2, angle0=0.1),
            HingedAppendage([0, 1, 0], 2, stiffness=3, angle0=-0.2),
        ),
        flexible=[
            FlexibleMode([1, 0, 0], 1, 3, displacement0=0.1),
            FlexibleMode([0, 1, 0], 2, 4, displacement0=-0.1),
        ],
        slosh=(SloshMode([1, 0, 0], 1, 2, displacement0=0.1),),
    )

    assert result.hinge.shape == (3, 2, 2)
    assert result.flexible.shape == (3, 2, 2)
    assert result.slosh.shape == (3, 2)


def test_extended_modes_reject_invalid_coupling():
    with pytest.raises(ValueError):
        HingedAppendage([0, 0, 0], 1)
    with pytest.raises(ValueError):
        propagate_6dof_extended(times=[0, 1], inertia=np.eye(3), r0=[1, 0, 0], v0=[0, 1, 0],
                                 flexible=FlexibleMode([1, 0, 0], 1, 1), slosh=SloshMode([1, 0, 0], 1, 1))


def test_slosh_force_follows_attitude():
    result = propagate_6dof_extended(
        times=[0, 1e-3], inertia=np.eye(3), mu=0,
        r0=[1, 0, 0], v0=[0, 0, 0], q0=[np.sqrt(0.5), 0, 0, np.sqrt(0.5)],
        slosh=SloshMode([1, 0, 0], 1, 1, displacement0=1), bus_mass=1,
    )

    np.testing.assert_allclose(result.trajectory.v[-1], [0, 1e-3, 0], atol=1e-9)


def test_hinge_cubic_stiffness_matches_initial_torque_response():
    kwargs = {
        "times": np.linspace(0, 1e-4, 3), "inertia": np.eye(3), "mu": 0,
        "r0": [1, 0, 0], "v0": [0, 0, 0], "q0": [1, 0, 0, 0],
    }
    nonlinear = propagate_6dof_extended(
        hinge=HingedAppendage([0, 0, 1], 1, angle0=0.5, cubic_stiffness=8), **kwargs)

    np.testing.assert_allclose(nonlinear.hinge[-1, 1], -1e-4, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(nonlinear.trajectory.omega[-1, 2], 1e-4, rtol=1e-6, atol=1e-12)


def test_extended_propagation_honors_initial_epoch_before_first_sample():
    result = propagate_6dof_extended(
        times=[10.0, 10.1], t0=0.0, inertia=np.eye(3), mu=0.0,
        r0=[1.0, 0.0, 0.0], v0=[0.0, 0.0, 0.0],
        acceleration=lambda t, r, v, q, omega: [t, 0.0, 0.0],
        hinge=HingedAppendage([0.0, 0.0, 1.0], 1.0),
    )

    np.testing.assert_allclose(result.trajectory.v[0], [50.0, 0.0, 0.0], atol=1e-10)
