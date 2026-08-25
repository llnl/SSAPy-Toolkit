import numpy as np
import pytest

import ssapy_toolkit as ssatk


def test_solve_6dof_target_reaches_terminal_velocity():
    spacecraft = ssatk.Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        inertia=np.eye(3),
    )

    result = ssatk.solve_6dof_target(
        spacecraft,
        times=np.linspace(0.0, 10.0, 11),
        target_v=[2.0, 0.0, 0.0],
        control_scale=[1.0, 1.0, 1.0],
        propagation_kwargs={"mu": 0.0},
    )

    assert result.success
    assert result.nfev <= 20
    np.testing.assert_allclose(result.control, [0.2, 0.0, 0.0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.trajectory.v[-1], [2.0, 0.0, 0.0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.residual, 0.0, atol=1e-12)


def test_solve_6dof_target_requires_a_terminal_target():
    spacecraft = ssatk.Spacecraft(r=[0.0, 0.0, 0.0], v=[0.0, 0.0, 0.0], inertia=np.eye(3))

    with pytest.raises(ValueError, match="target_r or target_v is required"):
        ssatk.solve_6dof_target(spacecraft, times=[0.0, 1.0])
