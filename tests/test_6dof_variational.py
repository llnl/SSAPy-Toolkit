import numpy as np

import ssapy_toolkit as ssatk
from ssapy_toolkit.propagators_6dof import (
    propagate_6dof,
    propagate_6dof_covariance,
    propagate_6dof_variational,
)


def test_variational_stm_matches_terminal_finite_difference_response():
    assert ssatk.propagate_6dof_variational is propagate_6dof_variational
    kwargs = {
        "times": np.array([0.0, 0.2]),
        "r0": np.array([7_000_000.0, 0.0, 0.0]),
        "v0": np.array([0.0, 7_500.0, 0.0]),
        "q0": np.array([1.0, 0.0, 0.0, 0.0]),
        "omega0": np.array([0.0, 0.0, 0.01]),
        "inertia": np.diag([2.0, 3.0, 4.0]),
        "rtol": 1e-10,
        "atol": 1e-12,
    }
    result = propagate_6dof_variational(**kwargs, jacobian_step=1e-6)
    delta = np.zeros(13)
    delta[0] = 0.1
    delta[10] = 1e-7
    plus = propagate_6dof(**{**kwargs, "r0": kwargs["r0"] + delta[:3], "omega0": kwargs["omega0"] + delta[10:13]})
    minus = propagate_6dof(**{**kwargs, "r0": kwargs["r0"] - delta[:3], "omega0": kwargs["omega0"] - delta[10:13]})
    finite_difference = np.concatenate((
        plus.r[-1] - minus.r[-1],
        plus.v[-1] - minus.v[-1],
        plus.q[-1] - minus.q[-1],
        plus.omega[-1] - minus.omega[-1],
    )) / 2.0
    predicted = result.stm[-1] @ delta

    assert result.stm.shape == (2, 13, 13)
    np.testing.assert_allclose(predicted, finite_difference, rtol=2e-4, atol=1e-12)


def test_variational_mass_state_is_included_and_custom_stm_is_honored():
    stm0 = np.eye(14) * 2.0
    result = propagate_6dof_variational(
        times=[0.0, 0.01], inertia=np.eye(3), r0=[7e6, 0, 0], v0=[0, 7500, 0],
        mass0=100.0, mass_flow_rate=lambda t, r, v, q, omega: 1.0, stm0=stm0,
        rtol=1e-6,
        atol=1e-9,
    )
    assert result.stm.shape == (2, 14, 14)
    np.testing.assert_allclose(result.stm[0], stm0)
    np.testing.assert_allclose(result.trajectory.mass, [100.0, 99.99], rtol=1e-8)


def test_covariance_transform_uses_stm_and_process_noise():
    result = propagate_6dof_variational(
        times=[0.0, 0.01],
        inertia=np.eye(3),
        r0=[7e6, 0, 0],
        v0=[0, 7500, 0],
        rtol=1e-6,
        atol=1e-9,
    )
    covariance0 = np.eye(13)
    process_noise = np.broadcast_to(np.eye(13) * 1e-6, result.stm.shape).copy()
    covariance = propagate_6dof_covariance(result, covariance0, process_noise)
    expected = result.stm[-1] @ covariance0 @ result.stm[-1].T + process_noise[-1]

    assert ssatk.propagate_6dof_covariance is propagate_6dof_covariance
    np.testing.assert_allclose(covariance[-1], expected, atol=1e-12)
    np.testing.assert_allclose(covariance, np.swapaxes(covariance, -1, -2), atol=1e-12)
