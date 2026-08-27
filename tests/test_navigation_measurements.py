import numpy as np
import pytest
from astropy.time import Time

from ssapy_toolkit.navigation import GroundStation, GroundStationMeasurement


@pytest.fixture
def station():
    return GroundStation(0.0, 0.0, fast=True)


def test_range_and_range_rate_use_relative_si_geometry(station, monkeypatch):
    monkeypatch.setattr(GroundStation, "_observer", lambda self: type("Observer", (), {
        "getRV": lambda self, time: (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        "itrs": np.array([1.0, 0.0, 0.0]),
    })())
    state = np.array([4.0, 4.0, 0.0, 2.0, 1.0, 0.0])
    assert station.predict(state, 0.0, "range").value == pytest.approx(5.0)
    prediction = station.predict(state, 0.0, "range_rate")
    assert prediction.value == pytest.approx(1.2)
    np.testing.assert_allclose(prediction.jacobian, [[0.256, -0.192, 0, 0.6, 0.8, 0]])


def test_az_el_and_ra_dec_wrap_and_visibility(station, monkeypatch):
    observer = type("Observer", (), {
        "getRV": lambda self, time: (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        "itrs": np.array([1.0, 0.0, 0.0]),
    })()
    monkeypatch.setattr(GroundStation, "_observer", lambda self: observer)
    monkeypatch.setattr(
        GroundStation,
        "_local_basis",
        lambda self, time: (
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
        ),
    )
    state = np.array([1.0, -1.0, 1.0, 0.0, 0.0, 0.0])
    az_el = station.predict(state, 0.0, "az_el")
    ra_dec = station.predict(state, 0.0, "ra_dec")
    np.testing.assert_allclose(az_el.value, [7 * np.pi / 4, 0.0])
    np.testing.assert_allclose(ra_dec.value, [3 * np.pi / 2, np.pi / 4])
    assert az_el.visible
    assert np.all(az_el.jacobian[:, 3:] == 0.0)


def test_analytic_jacobians_match_finite_difference(station, monkeypatch):
    observer = type("Observer", (), {
        "getRV": lambda self, time: (np.array([1.0, 2.0, 3.0]), np.array([-0.4, 0.8, 0.1])),
        "itrs": np.array([1.0, 0.0, 0.0]),
    })()
    monkeypatch.setattr(GroundStation, "_observer", lambda self: observer)
    state = np.array([8.0, -4.0, 10.0, 1.0, 2.0, -0.5])
    step = 1e-5
    for measurement in ("range", "range_rate", "az_el", "ra_dec"):
        prediction = station.predict(state, 0.0, measurement)
        finite = []
        for index in range(6):
            plus = state.copy(); plus[index] += step
            minus = state.copy(); minus[index] -= step
            delta = station.predict(plus, 0.0, measurement).value - station.predict(minus, 0.0, measurement).value
            if measurement in {"az_el", "ra_dec"} and index < 3:
                delta[0] = (delta[0] + np.pi) % (2 * np.pi) - np.pi
            finite.append(np.atleast_1d(delta / (2 * step)))
        np.testing.assert_allclose(prediction.jacobian, np.asarray(finite).T, rtol=2e-5, atol=2e-7)

    assert prediction.kind == "ra_dec"


def test_measurement_validation(station):
    with pytest.raises(ValueError, match="measurement"):
        station.predict(np.ones(6), 0.0, "doppler")
    with pytest.raises(ValueError, match="six-element"):
        station.predict(np.ones(5), 0.0)
    with pytest.raises(TypeError, match="numeric GPS"):
        station.predict(np.ones(6), "now")
    with pytest.raises(ValueError, match="scalar GPS"):
        station.predict(np.ones(6), [1.0, 2.0])
    with pytest.raises(ValueError, match="scalar GPS"):
        station.predict(np.ones(6), Time([1.0, 2.0], format="gps"))


def test_real_earth_observer_frame_is_orthonormal():
    station = GroundStation(0.0, 0.0, fast=True)
    observer = station._observer()
    east, north, up = station._local_basis(0.0)
    basis = np.array([east, north, up])
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=2e-12)
    target = observer.getRV(0.0)[0] + 1_000_000.0 * up + 1_000.0 * east
    prediction = station.predict(np.r_[target, [0.0, 0.0, 0.0]], 0.0, "az_el")
    assert prediction.visible
    assert np.all(np.isfinite(prediction.value))


def test_zenith_and_nadir_keep_non_azimuth_measurements_valid(station, monkeypatch):
    observer = type("Observer", (), {
        "getRV": lambda self, time: (np.zeros(3), np.zeros(3)),
        "itrs": np.array([1.0, 0.0, 0.0]),
    })()
    monkeypatch.setattr(GroundStation, "_observer", lambda self: observer)
    monkeypatch.setattr(
        GroundStation,
        "_local_basis",
        lambda self, time: (
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
        ),
    )
    for sign in (1.0, -1.0):
        state = np.r_[sign * np.array([100.0, 0.0, 0.0]), np.zeros(3)]
        assert station.predict(state, 0.0, "range").value == pytest.approx(100.0)
        assert station.predict(state, 0.0, "range_rate").value == pytest.approx(0.0)
        assert station.predict(state, 0.0, "ra_dec").value[0] == pytest.approx(0.0 if sign > 0 else np.pi)
        assert station.predict(state, 0.0, "range").elevation_rad == pytest.approx(sign * np.pi / 2)
        with pytest.raises(ValueError, match="azimuth is singular"):
            station.predict(state, 0.0, "az_el")


def test_scalar_astropy_time_matches_numeric_time(station, monkeypatch):
    observer = type("Observer", (), {
        "getRV": lambda self, time: (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        "itrs": np.array([1.0, 0.0, 0.0]),
    })()
    monkeypatch.setattr(GroundStation, "_observer", lambda self: observer)
    monkeypatch.setattr(GroundStation, "_local_basis", lambda self, time: (
        np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    ))
    state = np.array([4.0, 4.0, 0.0, 2.0, 1.0, 0.0])
    numeric = station.predict(state, 123.0, "range")
    scalar_time = station.predict(state, Time(123.0, format="gps"), "range")
    assert scalar_time.time == pytest.approx(numeric.time)
    assert scalar_time.value == pytest.approx(numeric.value)


def test_ground_station_measurement_adapts_prediction_to_ekf(station, monkeypatch):
    monkeypatch.setattr(GroundStation, "_observer", lambda self: type("Observer", (), {
        "getRV": lambda self, time: (np.zeros(3), np.zeros(3)),
    })())
    model = GroundStationMeasurement(station, 0.0, "ra_dec")
    value, jacobian = model(np.array([4.0, 3.0, 2.0, 0.0, 0.0, 0.0]))
    assert value.shape == (2,)
    assert jacobian.shape == (2, 6)
    assert model.angle_indices == (0,)
