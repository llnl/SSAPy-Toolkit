import warnings

import numpy as np
import pytest
from astropy.time import Time

from ssapy_toolkit.constants import EARTH_MU
from ssapy_toolkit.orbital_mechanics import transfer_ssapy_function as tsf


def _circular_state(radius, theta=0.0):
    r = radius * np.array([np.cos(theta), np.sin(theta), 0.0])
    speed = np.sqrt(EARTH_MU / radius)
    v = speed * np.array([-np.sin(theta), np.cos(theta), 0.0])
    return r, v


def test_stumpff_and_lambert_solver_branches():
    assert np.isclose(tsf._stumpff_C(0.0), 0.5)
    assert np.isclose(tsf._stumpff_S(0.0), 1.0 / 6.0)
    assert tsf._stumpff_C(1.0) > 0.0
    assert tsf._stumpff_S(1.0) > 0.0
    assert tsf._stumpff_C(-1.0) > 0.0
    assert tsf._stumpff_S(-1.0) > 0.0

    r1, _ = _circular_state(7000e3, 0.0)
    r2, _ = _circular_state(7000e3, np.deg2rad(60.0))
    v1, v2 = tsf.solve_lambert(r1, r2, 1000.0, mu=EARTH_MU, prograde=True)
    assert v1.shape == v2.shape == (3,)
    v1_retro, _ = tsf.solve_lambert(r1, r2, 5000.0, mu=EARTH_MU, prograde=False)
    assert v1_retro.shape == (3,)

    with pytest.raises(ValueError, match="positive"):
        tsf.solve_lambert(r1, r2, 0.0)
    with pytest.raises(RuntimeError, match="singular"):
        tsf.solve_lambert(r1, -r1, 1000.0)
    with pytest.raises(RuntimeError, match="converge"):
        tsf.solve_lambert(r1, r2, 1000.0, max_iter=0)


def test_burn_transfer_result_and_engine_info_summary():
    burn = tsf.Burn(1.0, 3.0, np.array([3.0, 4.0, 0.0]), np.array([0.0, 5.0, 0.0]))
    assert burn.dv_mag == 5.0
    np.testing.assert_allclose(burn.direction_ntw, [0.0, 1.0, 0.0])
    assert "|dv|=5.000" in repr(burn)
    zero = tsf.Burn(0.0, 1.0, np.zeros(3), np.zeros(3))
    np.testing.assert_array_equal(zero.direction_ntw, np.zeros(3))

    transfer = tsf.TransferResult([burn, zero], dv_budget=4.0, transfer_orbit=object())
    assert transfer.dv_total == 5.0
    assert transfer.within_budget is False
    transfer.arrival_error = 12.3
    summary = transfer.summary()
    assert "EXCEEDS" in summary
    assert "arrival position error" in summary

    tsf._attach_engine_info([burn], thrust=10.0, mass=100.0, isp=300.0)
    assert burn.thrust == 10.0
    assert burn.duration == 2.0
    assert burn.propellant_mass > 0.0

    inertial = tsf._ntw_to_inertial(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(inertial, [1.0, 2.0, 3.0])
    assert np.isclose(tsf._to_gps_seconds(Time(100.0, format="gps")), 100.0)
    assert tsf._to_gps_seconds(5) == 5.0


def test_transfer_ssapy_nonpropagating_modes_and_validation():
    radius = 7000e3
    r1, v1 = _circular_state(radius, 0.0)
    r2, v2 = _circular_state(radius, np.deg2rad(60.0))
    departure = (r1, v1, 0.0)
    arrival = (r2, v2, 1000.0)

    result = tsf.transfer_ssapy(departure, arrival, propagate=False, refine=False, burn_duration=1.0, dv_budget=1e9)
    assert result["schema_version"] == "ssatk.transfer.v2"
    assert len(result["burns"]) == 2
    assert result["diagnostics"]["within_budget"] is True
    assert result["trajectory"] is None
    assert result["transfer_orbits"][0].r.shape == (3,)

    mapped = tsf.transfer_ssapy(
        initial={"r": r1, "v": v1, "t": 10.0},
        target={"r": r2, "v": v2, "t": 1010.0},
        propagate=False,
        refine=False,
        burn_duration=1.0,
        thrust=1e7,
        mass=1000.0,
        isp=300.0,
    )
    assert mapped["tof"] == pytest.approx(1000.0)
    assert mapped["hardware"]["thrust"] == pytest.approx(1e7)
    assert mapped["hardware"]["mass"] == pytest.approx(1000.0)
    assert mapped["burns"][0]["propellant_mass"] > 0.0

    intercept = tsf.transfer_ssapy(departure, arrival, propagate=False, refine=False, burn_duration=1.0, arrival_burn=False, burn_accel=100.0)
    assert len(intercept["burns"]) == 1

    with pytest.raises(ValueError, match="after departure"):
        tsf.transfer_ssapy(arrival, departure, propagate=False, refine=False)
    with pytest.raises(ValueError, match="thrust and mass"):
        tsf.transfer_ssapy(departure, arrival, propagate=False, refine=False, thrust=1.0)
    with pytest.raises(ValueError, match="either burn_accel"):
        tsf.transfer_ssapy(departure, arrival, propagate=False, refine=False, thrust=1.0, mass=1.0, burn_accel=1.0)
    with pytest.raises(ValueError, match="isp requires"):
        tsf.transfer_ssapy(departure, arrival, propagate=False, refine=False, isp=300.0)
    with pytest.raises(ValueError, match="exceed a third"):
        tsf.transfer_ssapy(departure, arrival, propagate=False, refine=False, burn_duration=400.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = tsf.transfer_ssapy(departure, arrival, propagate=False, refine=False, burn_duration=1.0, dv_budget=1.0)
    assert result["diagnostics"]["within_budget"] is False
    assert any("exceeding" in str(w.message) for w in caught)
    with pytest.raises(ValueError, match="exceeding"):
        tsf.transfer_ssapy(departure, arrival, propagate=False, refine=False, burn_duration=1.0, dv_budget=1.0, raise_on_budget=True)


class FakeAccel:
    def __init__(self, name="accel", mu=EARTH_MU):
        self.name = name
        self.mu = mu

    def __add__(self, other):
        return FakeAccel(f"{self.name}+{getattr(other, 'name', other)}", mu=self.mu)


class FakeOrbit:
    def __init__(self, r, v, t=0.0, mu=EARTH_MU):
        self.r = np.asarray(r, dtype=float)
        self.v = np.asarray(v, dtype=float)
        self.t = float(t)
        self.mu = mu


class FakePropagator:
    def __init__(self, accel, h=None):
        self.accel = accel
        self.h = h


def test_transfer_ssapy_propagation_path_with_fakes(monkeypatch):
    import ssapy.compute as compute

    monkeypatch.setattr(tsf, "Orbit", FakeOrbit)
    monkeypatch.setattr(tsf, "AccelKepler", lambda: FakeAccel("kepler"))
    monkeypatch.setattr(tsf, "AccelConstNTW", lambda vec, time_breakpoints=None: FakeAccel(f"burn:{np.linalg.norm(vec):.3f}"))
    monkeypatch.setattr(tsf, "RK78Propagator", FakePropagator)
    monkeypatch.setattr(tsf, "rv_to_ntw", lambda r, v, x: np.asarray(x, dtype=float))
    monkeypatch.setattr(tsf, "normed", lambda x: np.asarray(x, dtype=float) / np.linalg.norm(x))

    r1, v1 = _circular_state(7000e3, 0.0)
    r2, v2 = _circular_state(7100e3, np.deg2rad(20.0))
    monkeypatch.setattr(
        tsf,
        "solve_lambert",
        lambda r1_in, r2_in, tof, mu=EARTH_MU, prograde=True: (
            np.asarray(v1, dtype=float) + np.array([1.0, 0.0, 0.0]),
            np.asarray(v2, dtype=float) - np.array([0.0, 2.0, 0.0]),
        ),
    )

    def fake_rv(orbit, times, propagator=None):
        times = np.asarray(times, dtype=float)
        dt = times - float(orbit.t)
        drift = np.column_stack((dt, 0.5 * dt, np.zeros_like(dt)))
        return (
            np.asarray(orbit.r, dtype=float)[None, :] + drift,
            np.repeat(np.asarray(orbit.v, dtype=float)[None, :], len(times), axis=0),
        )

    monkeypatch.setattr(compute, "rv", fake_rv)
    departure = FakeOrbit(r1, v1, 0.0)
    arrival = FakeOrbit(r2, v2, 100.0)

    result = tsf.transfer_ssapy(
        departure,
        arrival,
        accel=[FakeAccel("a"), FakeAccel("b")],
        thrust=10.0,
        mass=100.0,
        isp=300.0,
        propagate=True,
        refine=False,
        n_samples=5,
        propagator=FakePropagator,
        rk_step=3.0,
    )
    assert result["trajectory"]["r"].shape[1] == 3
    assert result["diagnostics"]["arrival_error"] is not None
    assert isinstance(result["transfer_orbits"][0], FakeOrbit)
    assert all(burn["propellant_mass"] is not None for burn in result["burns"])

    intercept = tsf.transfer_ssapy(
        (r1, v1, 0.0),
        (r2, v2, 100.0),
        burn_accel=1.0,
        arrival_burn=False,
        propagate=True,
        refine=False,
        n_samples=4,
    )
    assert len(intercept["burns"]) == 1
    assert intercept["trajectory"]["t"].size >= 2


def test_transfer_ssapy_propagation_error_branches(monkeypatch):
    import ssapy.compute as compute

    monkeypatch.setattr(tsf, "Orbit", FakeOrbit)
    monkeypatch.setattr(tsf, "AccelKepler", lambda: FakeAccel("kepler"))
    monkeypatch.setattr(tsf, "AccelConstNTW", lambda vec, time_breakpoints=None: FakeAccel("burn"))
    monkeypatch.setattr(tsf, "rv_to_ntw", lambda r, v, x: np.asarray(x, dtype=float))
    r1, v1 = _circular_state(7000e3, 0.0)
    r2, v2 = _circular_state(7000e3, np.deg2rad(10.0))
    monkeypatch.setattr(
        tsf,
        "solve_lambert",
        lambda r1_in, r2_in, tof, mu=EARTH_MU, prograde=True: (
            np.asarray(v1, dtype=float) + np.array([1.0, 0.0, 0.0]),
            np.asarray(v2, dtype=float),
        ),
    )

    def fake_rv(orbit, times, propagator=None):
        times = np.asarray(times, dtype=float)
        return (
            np.repeat(np.asarray(orbit.r)[None, :], len(times), axis=0),
            np.repeat(np.asarray(orbit.v)[None, :], len(times), axis=0),
        )

    monkeypatch.setattr(compute, "rv", fake_rv)
    with pytest.raises(TypeError, match="propagator must"):
        tsf.transfer_ssapy((r1, v1, 0.0), (r2, v2, 100.0), propagate=True, refine=False, propagator=object())

    def short_rv(orbit, times, propagator=None):
        return np.asarray(orbit.r).reshape(1, 3), np.asarray(orbit.v).reshape(1, 3)

    monkeypatch.setattr(compute, "rv", short_rv)
    with pytest.raises(RuntimeError, match="terminated early"):
        tsf.transfer_ssapy((r1, v1, 0.0), (r2, v2, 100.0), propagate=True, refine=False, n_samples=4)
