"""Coupled 6-DoF state-transition propagation.

The state-transition matrix is integrated with the same ``sixdof_rhs`` used
by the nominal propagator. The central-gravity, fixed-inertia rigid-body case
uses analytic Jacobians, including gravity-gradient torque partials.
Force and torque models that expose a ``state_jacobian`` (such as constant
NTW, inertial, body-frame, and body-torque models) are included analytically;
other model combinations retain central differences so every existing force,
torque, mass, and wheel model stays in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..constants import EARTH_MU
from ..coordinates.attitude import (
    normalize_quaternion,
    quaternion_conjugate,
    rotate_vector,
)
from ..propagators_orbit.high_accuracy import _kepler_jacobian
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
    "attitude_error_stm",
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

    @property
    def attitude_error_stm(self):
        """Return the STM in ``[r, v, δθ_body, omega, ...]`` coordinates."""

        return attitude_error_stm(self)


def attitude_error_stm(variational) -> np.ndarray:
    """Convert a quaternion-coordinate STM to a multiplicative attitude-error STM.

    The returned state ordering is ``[r, v, δθ_body, omega, [mass],
    [wheel_momentum]]``.  ``δθ_body`` is the three-parameter local error in the
    body frame, defined by ``q_perturbed = q_nominal ⊗ δq_body``.  This removes
    the redundant quaternion scalar direction while retaining the original
    quaternion STM for workflows that need it.
    """
    trajectory = getattr(variational, "trajectory", None)
    stm = np.asarray(getattr(variational, "stm", variational), dtype=float)
    if trajectory is None or stm.ndim != 3 or stm.shape[1] != stm.shape[2]:
        raise ValueError("variational must contain a trajectory and square STM array.")
    n = stm.shape[1]
    if n < 13 or trajectory.q.shape[0] != stm.shape[0]:
        raise ValueError("variational must contain a quaternion STM matching its trajectory.")

    initial_q = normalize_quaternion(trajectory.q[0])
    input_map = np.zeros((n, n - 1))
    input_map[:6, :6] = np.eye(6)
    input_map[6:10, 6:9] = _quaternion_error_input_map(initial_q)
    input_map[10:13, 9:12] = np.eye(3)
    input_map[13:, 12:] = np.eye(n - 13)

    result = np.empty((stm.shape[0], n - 1, n - 1))
    for index, quaternion in enumerate(trajectory.q):
        output_map = np.zeros((n - 1, n))
        output_map[:6, :6] = np.eye(6)
        output_map[6:9, 6:10] = _quaternion_error_output_map(normalize_quaternion(quaternion))
        output_map[9:12, 10:13] = np.eye(3)
        output_map[12:, 13:] = np.eye(n - 13)
        result[index] = output_map @ stm[index] @ input_map
    return result


def _quaternion_error_input_map(q: np.ndarray) -> np.ndarray:
    return 0.5 * np.vstack((-q[1:], q[0] * np.eye(3) + _skew(q[1:])))


def _quaternion_error_output_map(q: np.ndarray) -> np.ndarray:
    return 2.0 * np.hstack((-q[1:].reshape(3, 1), q[0] * np.eye(3) - _skew(q[1:])))


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
    analytic_jacobian = (
        (acceleration is None or callable(getattr(acceleration, "state_jacobian", None)))
        and ntw_acceleration is None
        and (body_acceleration is None or hasattr(body_acceleration, "attitude_jacobian"))
        and (torque is None or callable(getattr(torque, "state_jacobian", None)))
        and mass_flow_rate is None
        and wheel_torque is None
        and not callable(inertia)
    )

    def nominal_rhs(t, y):
        return sixdof_rhs(t, y, **rhs_kwargs)

    def combined_rhs(t, combined):
        y = combined[:n]
        phi = combined[n:].reshape(n, n)
        f = nominal_rhs(t, y)
        if analytic_jacobian:
            jac = _free_rigid_body_jacobian(
                y,
                mu=mu,
                inertia=inertia_arg,
                gravity_gradient=gravity_gradient,
                wheel_axes=wheel_axes,
                wheel_start=14 if state.mass is not None else 13,
            )
            if body_acceleration is not None:
                jac[3:6, 6:10] = _body_acceleration_jacobian(
                    body_acceleration, t, y, q_raw=y[6:10]
                )
            if acceleration is not None:
                jac[3:6, :13] += _model_state_jacobian(acceleration, t, y)
            if torque is not None:
                jac[10:13, :13] += np.linalg.solve(
                    inertia_arg, _model_state_jacobian(torque, t, y)
                )
        else:
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


def _free_rigid_body_jacobian(
    y,
    *,
    mu: float,
    inertia: np.ndarray,
    gravity_gradient: bool = False,
    wheel_axes: np.ndarray | None = None,
    wheel_start: int = 13,
) -> np.ndarray:
    n = y.size
    jacobian = np.zeros((n, n), dtype=float)
    jacobian[:6, :6] = _kepler_jacobian(y[:3], mu)

    q_raw = y[6:10]
    q = normalize_quaternion(q_raw)
    q_norm = np.linalg.norm(q_raw)
    omega = y[10:13]
    omega_skew = _skew(omega)
    q_vector = q[1:]
    jacobian[6:10, 10:13] = 0.5 * np.vstack((-q_vector, q[0] * np.eye(3) + _skew(q_vector)))
    q_jacobian = 0.5 * np.block(
        [
            [np.zeros((1, 1)), -omega.reshape(1, 3)],
            [omega.reshape(3, 1), -omega_skew],
        ]
    )
    jacobian[6:10, 6:10] = q_jacobian @ (np.eye(4) - np.outer(q, q)) / q_norm

    angular_momentum = inertia @ omega
    wheel_momentum = None if wheel_axes is None else y[wheel_start:]
    if wheel_momentum is not None:
        angular_momentum = angular_momentum + wheel_axes @ wheel_momentum
    jacobian[10:13, 10:13] = np.linalg.solve(
        inertia,
        _skew(angular_momentum) - omega_skew @ inertia,
    )
    if wheel_momentum is not None:
        jacobian[10:13, wheel_start:] = -np.linalg.solve(
            inertia, omega_skew @ wheel_axes
        )
    if gravity_gradient:
        torque_dr, torque_dq = _gravity_gradient_jacobian(
            y[:3], q_raw, inertia, mu
        )
        jacobian[10:13, :3] = np.linalg.solve(inertia, torque_dr)
        jacobian[10:13, 6:10] = np.linalg.solve(inertia, torque_dq)
    return jacobian


def _gravity_gradient_jacobian(
    r: np.ndarray, q_raw: np.ndarray, inertia: np.ndarray, mu: float
) -> tuple[np.ndarray, np.ndarray]:
    radius = np.linalg.norm(r)
    if radius == 0.0 or mu == 0.0:
        return np.zeros((3, 3)), np.zeros((3, 4))
    q = normalize_quaternion(q_raw)
    q_norm = np.linalg.norm(q_raw)
    radial = r / radius
    body_axes = np.column_stack(
        [rotate_vector(quaternion_conjugate(q), np.eye(3)[:, index]) for index in range(3)]
    )
    body_radial = body_axes @ radial
    angular_momentum = inertia @ body_radial
    force_jacobian = _skew(body_radial) @ inertia - _skew(angular_momentum)
    scale = 3.0 * mu / radius**3
    radial_jacobian = body_axes @ (np.eye(3) - np.outer(radial, radial)) / radius
    dscale_dr = -9.0 * mu * r / radius**5
    torque_direction = np.cross(body_radial, angular_momentum)
    torque_dr = np.outer(torque_direction, dscale_dr) + scale * force_jacobian @ radial_jacobian
    body_error = _quaternion_error_output_map(q) / q_norm
    torque_dq = scale * force_jacobian @ _skew(body_radial) @ body_error
    return torque_dr, torque_dq


def _body_acceleration_jacobian(model, t, y, *, q_raw):
    q = normalize_quaternion(q_raw)
    body_acceleration = np.asarray(model(t, y[:3], y[3:6], q, y[10:13]), dtype=float)
    if body_acceleration.shape != (3,):
        raise ValueError("body_acceleration must return a 3-vector.")
    body_jacobian = np.asarray(model.attitude_jacobian(q), dtype=float)
    if body_jacobian.shape != (3, 4):
        raise ValueError("attitude_jacobian must return a (3, 4) array.")
    normalization_jacobian = (
        np.eye(4) - np.outer(q, q)
    ) / np.linalg.norm(q_raw)
    return (
        _rotate_vector_jacobian(q_raw, body_acceleration)
        + rotate_vector_matrix(q) @ body_jacobian @ normalization_jacobian
    )


def _model_state_jacobian(model, t, y) -> np.ndarray:
    q = normalize_quaternion(y[6:10])
    jacobian = np.asarray(
        model.state_jacobian(
            t=t, r=y[:3], v=y[3:6], q=q, omega=y[10:13]
        ),
        dtype=float,
    )
    if jacobian.shape != (3, 13):
        raise ValueError("state_jacobian must return a (3, 13) array.")
    normalization_jacobian = (
        np.eye(4) - np.outer(q, q)
    ) / np.linalg.norm(y[6:10])
    jacobian = jacobian.copy()
    jacobian[:, 6:10] = jacobian[:, 6:10] @ normalization_jacobian
    return jacobian


def rotate_vector_matrix(q: np.ndarray) -> np.ndarray:
    w = q[0]
    vector = q[1:]
    return (w * w - vector @ vector) * np.eye(3) + 2.0 * np.outer(vector, vector) + 2.0 * w * _skew(vector)


def _rotate_vector_jacobian(q_raw: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q_raw = np.asarray(q_raw, dtype=float)
    q = normalize_quaternion(q_raw)
    w = q[0]
    u = q[1:]
    derivative = np.empty((3, 4))
    derivative[:, 0] = 2.0 * (w * np.eye(3) + _skew(u)) @ vector
    for index in range(3):
        basis = np.eye(3)[:, index]
        derivative[:, index + 1] = (
            -2.0 * u[index] * np.eye(3)
            + 2.0 * np.outer(basis, u)
            + 2.0 * np.outer(u, basis)
            + 2.0 * w * _skew(basis)
        ) @ vector
    return derivative @ (np.eye(4) - np.outer(q, q)) / np.linalg.norm(q_raw)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
