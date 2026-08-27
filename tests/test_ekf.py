import numpy as np
import pytest

from ssapy_toolkit.navigation import (
    CartesianMeasurement,
    CartesianOrbitEKF,
    EKFState,
    ExtendedKalmanFilter,
    wrap_angle_residual,
)


def test_ekf_predict_update_reduces_position_error():
    state = EKFState(np.array([0.0, 1.0]), np.eye(2), time=0.0)
    ekf = ExtendedKalmanFilter(state)
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    ekf.predict(lambda x, _t: transition @ x, transition, np.eye(2) * 1e-6, time=1.0)
    updated = ekf.update([1.2], lambda x: (x[[0]], np.array([[1.0, 0.0]])), np.array([[0.01]]))
    assert updated.x[0] == pytest.approx(1.199, abs=0.01)
    assert updated.covariance[0, 0] < 0.01
    np.testing.assert_allclose(updated.covariance, updated.covariance.T)


def test_cartesian_measurement_builds_selection_matrix():
    measurement, jacobian = CartesianMeasurement((0, 2))(np.arange(4.0))
    np.testing.assert_allclose(measurement, [0.0, 2.0])
    np.testing.assert_allclose(jacobian, [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])


def test_wrap_angle_residual_handles_circular_innovation():
    np.testing.assert_allclose(
        wrap_angle_residual([2.0 * np.pi - 1.0e-3], (0,)), [-1.0e-3], atol=1.0e-12
    )


def test_ekf_update_wraps_requested_angle_components():
    ekf = ExtendedKalmanFilter(EKFState([0.0], [[1.0]]))
    updated = ekf.update(
        [2.0 * np.pi - 0.1],
        lambda state: (np.array([state[0]]), np.ones((1, 1))),
        [[0.01]],
        angle_indices=(0,),
    )
    assert updated.x[0] == pytest.approx(-0.099, abs=0.002)


def test_ekf_rejects_invalid_covariance():
    with pytest.raises(ValueError, match="positive semidefinite"):
        EKFState([0.0], [[-1.0]])


def test_cartesian_orbit_ekf_predicts_and_updates():
    mu = 3.986004418e14
    state = EKFState(
        [7.0e6, 0.0, 0.0, 0.0, np.sqrt(mu / 7.0e6), 0.0],
        np.eye(6),
    )
    ekf = CartesianOrbitEKF(state)
    predicted = ekf.predict_orbit(time=60.0, mu=mu, max_step=10.0)
    assert predicted.time == pytest.approx(60.0)
    assert predicted.covariance.shape == (6, 6)
    measurement = predicted.x[:3] + [10.0, 0.0, 0.0]
    updated = ekf.update(
        measurement,
        CartesianMeasurement((0, 1, 2)),
        np.eye(3) * 1.0e-4,
        time=60.0,
    )
    assert abs(updated.x[0] - measurement[0]) < 1.0
    assert updated.covariance[0, 0] < predicted.covariance[0, 0]
