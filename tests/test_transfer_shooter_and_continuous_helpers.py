import importlib
from types import SimpleNamespace

import numpy as np
import pytest


class FakeTime:
    def __init__(self, value=0.0, *args, **kwargs):
        if hasattr(value, "gps"):
            self.gps = float(value.gps)
        elif isinstance(value, str):
            self.gps = 0.0
        else:
            self.gps = float(value)

    def __add__(self, other):
        return FakeTime(self.gps + float(other))

    __radd__ = __add__


class FakeOrbit:
    def __init__(self, r=None, v=None, t=0.0, period=1.0, **kwargs):
        self.r = np.asarray(r if r is not None else [0.0, 0.0, 0.0], dtype=float)
        self.v = np.asarray(v if v is not None else [0.0, 0.0, 0.0], dtype=float)
        self.t = t
        self.period = float(period)


def _fake_times(*args, **kwargs):
    return [FakeTime(0.0), FakeTime(1.0)]


def _linear_rv(orbit, time=None, times=None, propagator=None):
    sample_times = time if time is not None else times
    count = len(sample_times)
    positions = np.vstack((orbit.r, orbit.r + orbit.v))[:count]
    velocities = np.repeat(orbit.v[None, :], count, axis=0)
    return positions, velocities


def test_transfer_shooter_converges_with_fake_linear_propagation(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_shooter")
    monkeypatch.setattr(module, "Orbit", FakeOrbit)
    monkeypatch.setattr(module, "Time", FakeTime)
    monkeypatch.setattr(module, "get_times", _fake_times)
    monkeypatch.setattr(module, "rv", _linear_rv)

    plots = importlib.import_module("ssapy_toolkit.plots")
    monkeypatch.setattr(plots, "transfer_plot", lambda *args, **kwargs: "fake-fig")

    r1 = np.zeros(3)
    v1 = np.zeros(3)
    r2 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([0.5, 0.5, 0.5])
    result = module.transfer_shooter(r1, v1, r2, v2, tol=1e-9, max_iter=4, plot=True, status=True)

    np.testing.assert_allclose(result["delta_v1"], r2, rtol=0, atol=1e-6)
    np.testing.assert_allclose(result["r_transfer"][-1], r2, rtol=0, atol=1e-6)
    assert result["tof"] == 1.0
    assert result["fig"] == "fake-fig"
    assert result["|delta_v2|"] == pytest.approx(np.linalg.norm(v2 - r2))


def test_transfer_shooter_validation_and_fallback_branches(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_shooter")
    monkeypatch.setattr(module, "Orbit", FakeOrbit)
    monkeypatch.setattr(module, "Time", FakeTime)
    monkeypatch.setattr(module, "get_times", _fake_times)
    monkeypatch.setattr(module, "rv", _linear_rv)

    with pytest.raises(ValueError, match="Positional arguments"):
        module.transfer_shooter(1, 2, 3, 4, 5)
    with pytest.raises(ValueError, match="Two positional"):
        module.transfer_shooter(object(), object())
    with pytest.raises(ValueError, match="orbit1"):
        module.transfer_shooter(orbit1=object(), orbit2=FakeOrbit())
    with pytest.raises(ValueError, match="orbit2"):
        module.transfer_shooter(orbit1=FakeOrbit(), orbit2=object())
    with pytest.raises(ValueError, match="both r1 and v1"):
        module.transfer_shooter(r1=np.zeros(3), r2=np.ones(3))
    with pytest.raises(ValueError, match="z.axis"):
        module.transfer_shooter(np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 2.0]))

    result = module.transfer_shooter(FakeOrbit(r=[0, 0, 0], v=[0, 0, 0]), FakeOrbit(r=[1, 0, 0], v=[0, 1, 0]), max_iter=1)
    assert result["final"].v.shape == (3,)


def test_transfer_shooter_runtime_fallback_and_singular_jacobian(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_shooter")
    monkeypatch.setattr(module, "Orbit", FakeOrbit)
    monkeypatch.setattr(module, "Time", FakeTime)
    monkeypatch.setattr(module, "get_times", _fake_times)
    monkeypatch.setattr(module, "SciPyPropagator", lambda accel: ("propagator", accel))

    calls = []

    def flaky_rv(orbit, time=None, times=None, propagator=None):
        calls.append(propagator)
        if propagator is None:
            raise RuntimeError("analytic propagator failed")
        return _linear_rv(orbit, time=time, times=times, propagator=propagator)

    monkeypatch.setattr(module, "rv", flaky_rv)
    result = module.transfer_shooter(np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0]), np.zeros(3), max_iter=1)
    assert calls[0] is None
    assert calls[1][0] == "propagator"
    assert result["r_transfer"].shape[1] == 3

    monkeypatch.setattr(module, "rv", lambda orbit, time=None, times=None, propagator=None: (np.zeros((2, 3)), np.zeros((2, 3))))
    result = module.transfer_shooter(np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0]), np.zeros(3), max_iter=2, status=True)
    assert result["error"] == pytest.approx(1.0)


class FakeSolution:
    def __init__(self, y0, t_span=(0.0, 1.0), status=1, event=True, final_offset=None):
        self.t = np.asarray([float(t_span[0]), float(t_span[1])])
        y0 = np.asarray(y0, dtype=float)
        offset = np.zeros_like(y0) if final_offset is None else np.asarray(final_offset, dtype=float)
        self.y = np.vstack((y0, y0 + offset)).T
        self.status = status
        self.t_events = [np.asarray([self.t[-1]])] if event else [np.asarray([])]

    def sol(self, t_values):
        raw_t = np.asarray(t_values, dtype=float)
        scalar_input = raw_t.ndim == 0
        t_values = np.atleast_1d(raw_t)
        span = self.t[-1] - self.t[0]
        frac = np.zeros_like(t_values) if span == 0 else (t_values - self.t[0]) / span
        values = self.y[:, :1] + (self.y[:, -1:] - self.y[:, :1]) * frac[None, :]
        return values[:, 0] if scalar_input else values


def _patched_solver(status=1, event=True, offset_scale=1.0):
    def fake_solve_ivp(fun=None, t_span=(0.0, 1.0), y0=None, **kwargs):
        y0 = np.asarray(y0, dtype=float)
        if fun is not None:
            try:
                np.asarray(fun(t_span[0], y0), dtype=float)
            except Exception:
                pass
        events = kwargs.get("events")
        if events is not None:
            event_list = events if isinstance(events, (list, tuple)) else [events]
            for event_func in event_list:
                try:
                    event_func(t_span[0], y0)
                except Exception:
                    pass
        offset = np.ones_like(y0) * offset_scale
        if y0.size >= 6:
            offset[:3] = np.array([10.0, 0.0, 0.0])
            offset[3:6] = np.array([0.0, 1.0, 0.0])
        if y0.size >= 7:
            offset[6:] = 1.0
        return FakeSolution(y0, t_span=t_span, status=status, event=event, final_offset=offset)

    return fake_solve_ivp


def test_continuous_transfer_main_paths_and_failures(monkeypatch):
    r0 = np.array([7_000_000.0, 0.0, 0.0])
    v0 = np.array([0.0, 7_500.0, 0.0])

    vel = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_velocity_continuous")
    monkeypatch.setattr(vel, "solve_ivp", _patched_solver())
    with pytest.warns(RuntimeWarning, match="Final orbit is unbound"):
        r, v, t = vel.transfer_velocity_continuous(r0, v0, v_target=-2.0, a_thrust=0.1, mu=1e12, max_time=2.0)
    assert r.shape == (1000, 3)
    assert v.shape == (1000, 3)
    assert t[-1] == pytest.approx(2.0)
    with pytest.raises(TypeError, match="unexpected keyword"):
        vel.transfer_velocity_continuous(r0, v0, bogus=True)
    monkeypatch.setattr(vel, "solve_ivp", _patched_solver(status=0, event=False))
    with pytest.raises(ValueError, match="Target delta-v"):
        vel.transfer_velocity_continuous(r0, v0, v_target=1.0)

    inc = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_inclination_continuous")
    monkeypatch.setattr(inc, "solve_ivp", _patched_solver())
    r, v, t = inc.transfer_inclination_continuous(r0, v0, delta_v=-1.0, a_thrust=0.1, mu=1e12)
    assert r.shape == (2, 3)
    assert v.shape == (2, 3)
    assert t.shape == (2,)
    with pytest.raises(ValueError, match="exactly one"):
        inc.transfer_inclination_continuous(r0, v0)
    with pytest.raises(TypeError, match="unexpected keyword"):
        inc.transfer_inclination_continuous(r0, v0, delta_v=1.0, savefig="x.png", extra=True)
    monkeypatch.setattr(inc, "solve_ivp", _patched_solver(status=0, event=False))
    with pytest.raises(ValueError, match="Condition"):
        inc.transfer_inclination_continuous(r0, v0, i_target=0.1)


def test_continuous_transfer_plot_branches(monkeypatch, tmp_path):
    r0 = np.array([7_000_000.0, 0.0, 0.0])
    v0 = np.array([0.0, 7_500.0, 0.0])
    mu = 4.0e14

    vel = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_velocity_continuous")
    monkeypatch.setattr(vel, "solve_ivp", _patched_solver(offset_scale=0.01))
    monkeypatch.setattr(vel.plt, "show", lambda: None)
    saved = []
    monkeypatch.setattr(vel, "save_plot", lambda fig, path: saved.append(path))
    r, v, t = vel.transfer_velocity_continuous(r0, v0, v_target=None, a_thrust=0.01, mu=mu, max_time=1.0, plot=True, save_path=tmp_path / "velocity.png")
    assert r.shape == (1000, 3)
    assert saved == [tmp_path / "velocity.png"]

    inc = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_inclination_continuous")
    monkeypatch.setattr(inc, "solve_ivp", _patched_solver(offset_scale=0.01))
    monkeypatch.setattr(inc.plt, "show", lambda: None)
    saved.clear()
    monkeypatch.setattr(inc, "save_plot", lambda fig, path: saved.append(path))
    r, v, t = inc.transfer_inclination_continuous(r0, v0, i_target=0.1, a_thrust=0.01, mu=mu, plot=True, save_path=tmp_path / "inclination.png")
    assert r.shape == (2, 3)
    assert saved == [tmp_path / "inclination.png"]

    combo = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_velocity_and_inclination_continuous")
    monkeypatch.setattr(combo, "solve_ivp", _patched_solver(offset_scale=0.01))
    monkeypatch.setattr(combo.plt, "show", lambda: None)
    r, v, t = combo.transfer_velocity_and_inclination_continuous(r0, v0, i_target=0.1, a_thrust=0.01, mu=mu, plot=True)
    assert r.shape[1] == 3


def test_coplanar_and_two_phase_continuous_transfers(monkeypatch):
    r0 = np.array([7_000_000.0, 0.0, 0.0])
    v0 = np.array([0.0, 7_500.0, 0.0])
    r2 = np.array([7_000_010.0, 0.0, 0.0])

    cop = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_coplanar_continuous")
    monkeypatch.setattr(cop, "Time", FakeTime)
    monkeypatch.setattr(cop, "solve_ivp", _patched_solver())
    plots = importlib.import_module("ssapy_toolkit.plots")
    monkeypatch.setattr(plots, "transfer_plot", lambda *args, **kwargs: "coplanar-fig")
    result = cop.transfer_coplanar_continuous(r0, v0, r2, a_thrust=0.1, mu=1e12, plot=True)
    assert result["t_final"] == pytest.approx(7200.0)
    assert result["delta_v1"] > 0.0
    assert result["fig"] == "coplanar-fig"
    monkeypatch.setattr(cop, "solve_ivp", _patched_solver(status=0, event=False))
    with pytest.raises(ValueError, match="rendezvous"):
        cop.transfer_coplanar_continuous(r0, v0, r2)

    combo = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_velocity_and_inclination_continuous")
    monkeypatch.setattr(combo, "solve_ivp", _patched_solver())
    with pytest.warns(RuntimeWarning, match="Final orbit is unbound"):
        r, v, t = combo.transfer_velocity_and_inclination_continuous(r0, v0, i_target=0.1, a_thrust=0.1, mu=1e12)
    assert r.shape[1] == 3
    assert v.shape[1] == 3
    assert t[0] == 0.0
    monkeypatch.setattr(combo, "solve_ivp", _patched_solver(status=0, event=False))
    with pytest.raises(ValueError, match="Target inclination"):
        combo.transfer_velocity_and_inclination_continuous(r0, v0, i_target=0.1, a_thrust=0.1)
