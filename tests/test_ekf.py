import numpy as np
import pytest

from ssapy_toolkit.navigation import CartesianMeasurement, EKFState, ExtendedKalmanFilter


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


def test_ekf_rejects_invalid_covariance():
    with pytest.raises(ValueError, match="positive semidefinite"):
        EKFState([0.0], [[-1.0]])
