"""Deterministic, testable navigation sensor primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorMeasurement:
    """One sensor sample, including validity and reported covariance."""

    time: float
    value: np.ndarray | None
    covariance: np.ndarray
    valid: bool = True


class CartesianSensor:
    """Cartesian sensor with fixed bias, Gaussian noise, and dropouts."""

    def __init__(
        self,
        covariance,
        *,
        bias=None,
        rng=None,
        dropout_probability: float = 0.0,
        valid_interval: tuple[float, float] | None = None,
    ):
        covariance = np.asarray(covariance, dtype=float)
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be square.")
        if not np.all(np.isfinite(covariance)) or np.any(np.linalg.eigvalsh(covariance) < 0.0):
            raise ValueError("covariance must be finite and positive semidefinite.")
        self.covariance = covariance
        self.bias = np.zeros(covariance.shape[0]) if bias is None else np.asarray(bias, dtype=float)
        if self.bias.shape != (covariance.shape[0],) or not np.all(np.isfinite(self.bias)):
            raise ValueError("bias must match covariance dimension and be finite.")
        if not 0.0 <= float(dropout_probability) <= 1.0:
            raise ValueError("dropout_probability must be between 0 and 1.")
        if valid_interval is not None and valid_interval[1] < valid_interval[0]:
            raise ValueError("valid_interval stop must be >= start.")
        self.rng = np.random.default_rng() if rng is None else rng
        self.dropout_probability = float(dropout_probability)
        self.valid_interval = valid_interval

    def measure(self, truth, time: float) -> SensorMeasurement:
        truth = np.asarray(truth, dtype=float)
        if truth.shape != self.bias.shape or not np.all(np.isfinite(truth)):
            raise ValueError("truth must match the sensor dimension and be finite.")
        time = float(time)
        in_interval = self.valid_interval is None or self.valid_interval[0] <= time <= self.valid_interval[1]
        valid = in_interval and float(self.rng.random()) >= self.dropout_probability
        if not valid:
            return SensorMeasurement(time, None, self.covariance.copy(), False)
        noise = self.rng.multivariate_normal(np.zeros(truth.size), self.covariance)
        return SensorMeasurement(time, truth + self.bias + noise, self.covariance.copy(), True)


__all__ = ["CartesianSensor", "SensorMeasurement"]
