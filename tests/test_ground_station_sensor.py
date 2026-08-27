from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from ssapy_toolkit.navigation import (
    GroundStation,
    GroundStationSensor,
    StationPrediction,
)


@pytest.fixture
def station(monkeypatch):
    def predict(self, truth, time, measurement):
        values = {"range": 10.0, "range_rate": 2.0, "az_el": np.array([1.0, 0.2]), "ra_dec": np.array([2.0, 0.3])}
        rows = 1 if measurement in {"range", "range_rate"} else 2
        return StationPrediction(float(time), measurement, values[measurement], np.zeros((rows, 6)), True, 0.2)

    monkeypatch.setattr(GroundStation, "predict", predict)
    return GroundStation(0.0, 0.0, min_elevation_rad=0.0)


def test_seeded_noise_and_bias_are_reproducible(station):
    first = GroundStationSensor(station, 1e-12, measurement="range", bias=1.5, rng=123)
    second = GroundStationSensor(station, 1e-12, measurement="range", bias=1.5, rng=123)
    one = first.measure(np.ones(6), 0.0)
    two = second.measure(np.ones(6), 0.0)
    assert one.value == pytest.approx(two.value)
    assert one.value == pytest.approx(11.5, abs=1e-5)
    assert one.valid and one.visible and one.kind == "range"


def test_vector_bias_noise_and_metadata_are_immutable(station):
    sensor = GroundStationSensor(
        station,
        [[1e-12, 0.0], [0.0, 2e-12]],
        kind="az_el",
        bias=[0.1, -0.2],
        rng=7,
    )
    observation = sensor.measure(np.ones(6), 0.0)
    assert observation.value.shape == (2,)
    assert observation.covariance.shape == (2, 2)
    assert not observation.value.flags.writeable
    assert not observation.covariance.flags.writeable
    assert not observation.bias.flags.writeable
    with pytest.raises(FrozenInstanceError):
        observation.valid = False


def test_covariance_shape_and_positive_semidefinite_validation(station):
    zero = GroundStationSensor(station, 0.0)
    assert zero.measure(np.ones(6), 0.0).value == pytest.approx(10.0)
    with pytest.raises(ValueError, match="positive semidefinite"):
        GroundStationSensor(station, [[1.0, 2.0], [2.0, 1.0]], measurement="az_el")
    with pytest.raises(ValueError, match="shape"):
        GroundStationSensor(station, np.eye(2))
    with pytest.raises(ValueError, match="shape"):
        GroundStationSensor(station, 1.0, measurement="az_el")


def test_dropout_and_visibility_mask_preserve_metadata(station, monkeypatch):
    dropout = GroundStationSensor(station, 1.0, dropout_probability=1.0, rng=1)
    dropped = dropout.measure(np.ones(6), 0.0)
    assert dropped.value is None and not dropped.valid
    assert dropped.covariance.shape == (1, 1)

    original_min = station.min_elevation_rad
    monkeypatch.setattr(
        GroundStation,
        "predict",
        lambda self, truth, time, measurement: StationPrediction(
            0.0, measurement, 10.0 if measurement == "range" else np.array([1.0, 0.2]),
            np.zeros((1 if measurement == "range" else 2, 6)), True, -0.1,
        ),
    )
    masked = GroundStationSensor(station, 1.0, min_elevation_rad=0.0, rng=1).measure(np.ones(6), 0.0)
    allowed = GroundStationSensor(station, 1.0, min_elevation_rad=-0.2, rng=1).measure(np.ones(6), 0.0)
    assert masked.value is None and not masked.valid and not masked.visible
    assert allowed.valid and allowed.visible
    assert station.min_elevation_rad == original_min


def test_visibility_mask_does_not_consume_rng_and_override_allows_generation(station, monkeypatch):
    monkeypatch.setattr(
        GroundStation,
        "predict",
        lambda self, truth, time, measurement: StationPrediction(
            0.0, measurement, 10.0 if measurement == "range" else np.array([1.0, 0.2]),
            np.zeros((1 if measurement == "range" else 2, 6)), True, -0.1,
        ),
    )
    generator = np.random.default_rng(19)
    before = generator.bit_generator.state
    masked = GroundStationSensor(station, 0.0, min_elevation_rad=0.0, rng=generator).measure(
        np.ones(6), 0.0
    )
    after = generator.bit_generator.state
    assert masked.value is None and not masked.valid and not masked.visible
    assert before == after

    forced = GroundStationSensor(
        station, 0.0, min_elevation_rad=0.0, visibility_override=True, rng=19
    ).measure(np.ones(6), 0.0)
    assert forced.valid and not forced.visible and forced.value == pytest.approx(10.0)


def test_angular_noise_wraps_azimuth_and_retains_prediction(station, monkeypatch):
    monkeypatch.setattr(
        GroundStation,
        "predict",
        lambda self, truth, time, measurement: StationPrediction(
            0.0, measurement, np.array([2.0 * np.pi - 1e-3, 0.2]), np.ones((2, 6)), True, 0.2
        ),
    )
    observation = GroundStationSensor(station, np.zeros((2, 2)), measurement="az_el").measure(
        np.ones(6), 0.0
    )
    assert 0.0 <= observation.value[0] < 2.0 * np.pi
    np.testing.assert_allclose(observation.jacobian, np.ones((2, 6)))


def test_rng_type_and_parameter_validation(station):
    with pytest.raises(TypeError, match="Generator or integer"):
        GroundStationSensor(station, 1.0, rng=object())
    with pytest.raises(ValueError, match="dropout"):
        GroundStationSensor(station, 1.0, dropout_probability=1.1)
    with pytest.raises(ValueError, match="bias"):
        GroundStationSensor(station, 1.0, bias=[1.0])
