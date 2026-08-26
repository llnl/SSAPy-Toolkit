"""Small, dependency-light extended Kalman filter primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray


def _matrix(value, shape, name):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite matrix with shape {shape}.")
    return matrix


def _vector(value, size, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},).")
    return vector


@dataclass(frozen=True)
class EKFState:
    """State estimate and covariance at one filter epoch."""

    x: Array
    covariance: Array
    time: float = 0.0

    def __post_init__(self):
        x = _vector(self.x, np.asarray(self.x).size, "x")
        covariance = _matrix(self.covariance, (x.size, x.size), "covariance")
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
            raise ValueError("covariance must be symmetric.")
        if np.any(np.linalg.eigvalsh(covariance) < -1e-12):
            raise ValueError("covariance must be positive semidefinite.")
        object.__setattr__(self, "x", x.copy())
        object.__setattr__(self, "covariance", covariance.copy())
        object.__setattr__(self, "time", float(self.time))


class ExtendedKalmanFilter:
    """Discrete extended Kalman filter with user-supplied dynamics Jacobians.

    The dynamics callback returns the predicted state. Its Jacobian callback
    returns the state-transition matrix for the same step. Measurement models
    return ``(predicted_measurement, measurement_jacobian)``.
    """

    def __init__(self, state: EKFState):
        self.state = state

    def predict(
        self,
        dynamics: Callable[[Array, float], Array],
        jacobian: Array,
        process_noise: Array,
        *,
        time: float | None = None,
    ) -> EKFState:
        n = self.state.x.size
        transition = _matrix(jacobian, (n, n), "jacobian")
        noise = _matrix(process_noise, (n, n), "process_noise")
        predicted = _vector(dynamics(self.state.x.copy(), self.state.time), n, "predicted state")
        covariance = transition @ self.state.covariance @ transition.T + noise
        covariance = 0.5 * (covariance + covariance.T)
        self.state = EKFState(predicted, covariance, self.state.time if time is None else time)
        return self.state

    def update(
        self,
        measurement,
        model: Callable[[Array], tuple[Array, Array]],
        noise: Array,
        *,
        time: float | None = None,
    ) -> EKFState:
        prediction, jacobian = model(self.state.x.copy())
        prediction = np.asarray(prediction, dtype=float)
        measurement = np.asarray(measurement, dtype=float)
        if prediction.ndim != 1 or measurement.shape != prediction.shape:
            raise ValueError("measurement and model output must be one-dimensional and equal-sized.")
        m, n = prediction.size, self.state.x.size
        jacobian = _matrix(jacobian, (m, n), "measurement jacobian")
        noise = _matrix(noise, (m, m), "measurement noise")
        innovation = measurement - prediction
        innovation_covariance = jacobian @ self.state.covariance @ jacobian.T + noise
        try:
            gain = np.linalg.solve(innovation_covariance, jacobian @ self.state.covariance).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("measurement covariance is singular.") from exc
        updated = self.state.x + gain @ innovation
        identity = np.eye(n)
        residual = identity - gain @ jacobian
        covariance = (
            residual @ self.state.covariance @ residual.T
            + gain @ noise @ gain.T
        )
        covariance = 0.5 * (covariance + covariance.T)
        self.state = EKFState(updated, covariance, self.state.time if time is None else time)
        return self.state


class CartesianMeasurement:
    """Direct Cartesian position/velocity measurement model."""

    def __init__(self, indices=(0, 1, 2, 3, 4, 5)):
        self.indices = tuple(int(index) for index in indices)

    def __call__(self, state: Array) -> tuple[Array, Array]:
        state = np.asarray(state, dtype=float)
        if state.ndim != 1 or any(index < 0 or index >= state.size for index in self.indices):
            raise ValueError("measurement indices must address a one-dimensional state.")
        jacobian = np.zeros((len(self.indices), state.size))
        jacobian[np.arange(len(self.indices)), self.indices] = 1.0
        return state[list(self.indices)], jacobian


__all__ = ["CartesianMeasurement", "EKFState", "ExtendedKalmanFilter"]
