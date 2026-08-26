import numpy as np
import pytest

from ssapy_toolkit.navigation import CartesianSensor


def test_cartesian_sensor_is_reproducible_and_reports_covariance():
    sensor_a = CartesianSensor(np.eye(2) * 0.01, bias=[1.0, -2.0], rng=np.random.default_rng(4))
    sensor_b = CartesianSensor(np.eye(2) * 0.01, bias=[1.0, -2.0], rng=np.random.default_rng(4))
    sample_a = sensor_a.measure([10.0, 20.0], 5.0)
    sample_b = sensor_b.measure([10.0, 20.0], 5.0)
    np.testing.assert_allclose(sample_a.value, sample_b.value)
    np.testing.assert_allclose(sample_a.covariance, np.eye(2) * 0.01)
    assert sample_a.valid


def test_cartesian_sensor_invalid_interval_returns_missing_measurement():
    sensor = CartesianSensor(np.eye(1), valid_interval=(1.0, 2.0))
    sample = sensor.measure([0.0], 3.0)
    assert not sample.valid
    assert sample.value is None


def test_cartesian_sensor_validates_dropout_probability():
    with pytest.raises(ValueError, match="dropout_probability"):
        CartesianSensor(np.eye(1), dropout_probability=1.1)
