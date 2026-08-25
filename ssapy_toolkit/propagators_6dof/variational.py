"""Coupled 6-DoF state-transition propagation.

The state-transition matrix is integrated with the same ``sixdof_rhs`` used
by the nominal propagator.  Its Jacobian is computed by central differences,
which keeps every existing force, torque, mass, and wheel model in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..constants import EARTH_MU
from ..coordinates.attitude import normalize_quaternion
from .sixdof import (
    ArrayLike,
    SixDOFTrajectory,
    _inertia_matrix,
    _initial_state,
    _times,
    _validate_time_direction,
    _wheel_axes_from_body,
    _wheel_axes_matrix,
    _wheel_capacity_from_body,
    _wheel_capacity_vector,
    sixdof_rhs,
)

__all__ = [
    "SixDOFVariationalTrajectory",
    "propagate_6dof_covariance",
    "propagate_6dof_variational",
]


@dataclass(frozen=True)
class SixDOFVariationalTrajectory:
    """Nominal trajectory and state-transition matrices.

    ``stm[k]`` maps infinitesimal perturbations of the solver state at
    ``t[0]`` to perturbations at ``t[k]``.  The state ordering is the same as
    ``sixdof_rhs``: ``r, v, q, omega, [mass], [wheel_momentum]``.
    """

    trajectory: SixDOFTrajectory
    stm: np.ndarray

    @property
    def t(self):
        return self.trajectory.t


def propagate_6dof_covariance(variational, covariance0, process_noise=None):
    """Map an initial covariance through a 6-DoF STM.

    ``process_noise`` is either one covariance contribution applied at every
    sample or an array with one contribution per sample. Contributions are in
    the propagated state coordinates and are added after the STM transform.
    """

    stm = np.asarray(getattr(variational, "stm", variational), dtype=float)
    if stm.ndim != 3 or stm.shape[1] != stm.shape[2]:
        raise ValueError("variational must contain an STM array with shape (samples, n, n).")
    covariance0 = np.asarray(covariance0, dtype=float)
    if covariance0.shape != stm.shape[1:] or not np.all(np.isfinite(covariance0)):
        raise ValueError(f"covariance0 must be finite with shape {stm.shape[1:]}.")
    covariance = np.einsum("...ij,jk,...lk->...il", stm, covariance0, stm)
    if process_noise is not None:
        process_noise = np.asarray(process_noise, dtype=float)
        if process_noise.shape == stm.shape[1:] or process_noise.shape == stm.shape:
            if not np.all(np.isfinite(process_noise)):
                raise ValueError("process_noise must be finite.")
            covariance = covariance + process_noise
        else:
            raise ValueError(f"process_noise must have shape {stm.shape[1:]} or {stm.shape}.")
    return 0.5 * (covariance + np.swapaxes(covariance, -1, -2))


def propagate_6dof_variational(
    *,
    times: ArrayLike,
    inertia: ArrayLike,
    orbit0=None,
    r0: ArrayLike | None = None,
    v0: ArrayLike | None = None,
    t0: float | None = None,
    q0: ArrayLike | None = None,
    omega0: ArrayLike | None = None,
    mu: float = EARTH_MU,
    acceleration=None,
    ntw_acceleration=None,
    body_acceleration=None,
    torque=None,
    gravity_gradient: bool = False,
    mass0: float | None = None,
    mass_flow_rate=None,
    wheel_momentum0: ArrayLike | None = None,
    wheel_axes_body: ArrayLike | None = None,
    wheel_torque=None,
    wheel_momentum_capacity: ArrayLike | None = None,
    stm0: ArrayLike | None = None,
    jacobian_step: float = 1e-7,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    method: str = "DOP853",
    max_step: float = np.inf,
    first_step: float | None = None,
) -> SixDOFVariationalTrajectory:
    """Propagate a 6-DoF state and its coupled state-transition matrix.

    ``jacobian_step`` is a relative central-difference perturbation.  The
    result uses the solver's quaternion coordinates; quaternion normalization
    in ``sixdof_rhs`` therefore makes this a local STM, not a globally valid
    six-parameter attitude error state.
    """
    times = _times(times)
    if wheel_axes_body is None and orbit0 is not None:
        wheel_axes_body = _wheel_axes_from_body(getattr(orbit0, "body", None))
    if wheel_momentum_capacity is None and orbit0 is not None:
        wheel_momentum_capacity = _wheel_capacity_from_body(getattr(orbit0, "body", None))
    state = _initial_state(
        orbit0=orbit0, r0=r0, v0=v0, t0=t0, q0=q0, omega0=omega0,
        mass0=mass0, wheel_momentum0=wheel_momentum0,
    )
    _validate_time_direction(times, state.t)
    if mass_flow_rate is not None and state.mass is None:
        raise ValueError("mass0 or orbit0.mass is required when mass_flow_rate is provided.")
    wheel_axes = _wheel_axes_matrix(wheel_axes_body)
    if wheel_torque is not None and wheel_axes is None:
        raise ValueError("wheel_axes_body or orbit0.body.reaction_wheels is required when wheel_torque is provided.")
    wheel_capacity = _wheel_capacity_vector(
        wheel_momentum_capacity, 0 if wheel_axes is None else wheel_axes.shape[1]
    )
    wheel_momentum = state.wheel_momentum
    if wheel_axes is not None:
        if wheel_momentum is None:
            wheel_momentum = np.zeros(wheel_axes.shape[1])
        if wheel_momentum.shape != (wheel_axes.shape[1],):
            raise ValueError("wheel_momentum0 must match the number of wheel axes.")
    y_parts = [state.r, state.v, state.q, state.omega]
    if state.mass is not None:
        y_parts.append([state.mass])
    if wheel_momentum is not None:
        y_parts.append(wheel_momentum)
    y0 = np.concatenate(y_parts)
    n = y0.size
    if jacobian_step <= 0.0 or not np.isfinite(jacobian_step):
        raise ValueError("jacobian_step must be finite and positive.")
    if stm0 is None:
        phi0 = np.eye(n)
    else:
        phi0 = np.asarray(stm0, dtype=float)
        if phi0.shape != (n, n):
            raise ValueError(f"stm0 must have shape {(n, n)}.")

    if callable(inertia):
        inertia_arg = inertia
        inv_inertia = None
    else:
        inertia_arg = _inertia_matrix(inertia)
        inv_inertia = np.linalg.inv(inertia_arg)

    rhs_kwargs = {
        "inertia": inertia_arg, "mu": mu, "acceleration": acceleration,
        "ntw_acceleration": ntw_acceleration, "body_acceleration": body_acceleration,
        "torque": torque, "gravity_gradient": gravity_gradient,
        "mass_flow_rate": mass_flow_rate, "wheel_axes_body": wheel_axes,
        "wheel_torque": wheel_torque, "wheel_momentum_capacity": wheel_capacity,
        "inv_inertia": inv_inertia, "mass_state": state.mass is not None,
    }

    def nominal_rhs(t, y):
        return sixdof_rhs(t, y, **rhs_kwargs)

    def combined_rhs(t, combined):
        y = combined[:n]
        phi = combined[n:].reshape(n, n)
        f = nominal_rhs(t, y)
        jac = np.empty((n, n))
        for column in range(n):
            step = jacobian_step * max(1.0, abs(y[column]))
            delta = np.zeros(n)
            delta[column] = step
            jac[:, column] = (nominal_rhs(t, y + delta) - nominal_rhs(t, y - delta)) / (2.0 * step)
        return np.concatenate((f, (jac @ phi).ravel()))

    sol = solve_ivp(
        combined_rhs, (state.t, float(times[-1])), np.concatenate((y0, phi0.ravel())),
        t_eval=times, rtol=rtol, atol=atol, method=method,
        max_step=max_step, first_step=first_step,
    )
    if not sol.success:
        raise RuntimeError(f"6-DoF variational propagation failed: {sol.message}")

    y = sol.y[:n].T
    q = np.array([normalize_quaternion(item) for item in y[:, 6:10]])
    mass = y[:, 13] if state.mass is not None else None
    wheel_start = 14 if state.mass is not None else 13
    wheels = None if wheel_momentum is None else y[:, wheel_start:]
    trajectory = SixDOFTrajectory(sol.t, y[:, :3], y[:, 3:6], q, y[:, 10:13], mass, wheels, int(sol.nfev), str(sol.message), int(sol.status), solution=sol.sol)
    return SixDOFVariationalTrajectory(trajectory, sol.y[n:].T.reshape((-1, n, n)))
