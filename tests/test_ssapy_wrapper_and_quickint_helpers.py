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

    def __repr__(self):
        return f"FakeTime({self.gps})"


class FakeOrbit:
    def __init__(self, r=None, v=None, t=0.0, propkw=None, **kwargs):
        self.r = np.asarray(r if r is not None else [7000e3, 0.0, 0.0], dtype=float)
        self.v = np.asarray(v if v is not None else [0.0, 7500.0, 0.0], dtype=float)
        self.t = t
        self.propkw = propkw
        self.period = kwargs.get("period", 10.0)

    @classmethod
    def fromKeplerianElements(cls, *args, **kwargs):
        return cls([8000e3, 0.0, 0.0], [0.0, 7100.0, 0.0], kwargs.get("t", 0.0))


class FakeAccel:
    def __init__(self, name="accel", *args, **kwargs):
        self.name = name

    def __add__(self, other):
        return FakeAccel(f"{self.name}+{getattr(other, 'name', other)}")

    def __repr__(self):
        return self.name


def test_ssapy_props_build_fake_propagators(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.ssapy_wrappers.ssapy_props")
    monkeypatch.setattr(module, "AccelKepler", lambda: FakeAccel("Kep"))
    monkeypatch.setattr(module, "AccelSolRad", lambda **kwargs: FakeAccel("SRP"))
    monkeypatch.setattr(module, "AccelEarthRad", lambda **kwargs: FakeAccel("ERad"))
    monkeypatch.setattr(module, "AccelDrag", lambda **kwargs: FakeAccel("Drag"))
    monkeypatch.setattr(module, "AccelHarmonic", lambda body, degree, order: FakeAccel(f"H{body}:{degree}:{order}"))
    monkeypatch.setattr(module, "AccelThirdBody", lambda body: FakeAccel(f"TB{body}"))
    monkeypatch.setattr(module, "get_body", lambda name, **kwargs: name)
    monkeypatch.setattr(module, "SciPyPropagator", lambda accel, ode_kwargs=None: SimpleNamespace(accel=accel, ode_kwargs=ode_kwargs))
    monkeypatch.setattr(module, "KeplerianPropagator", lambda: SimpleNamespace(kind="keplerian"))
    module._accel_3_cache = None
    module._accel_4_cache = None
    module._accel_best_cache = None
    module._accel_best_gravity_cache = None

    assert module.ssapy_kwargs(mass="2", area="3", CD="4", CR="5") == {"mass": 2.0, "area": 3.0, "CD": 4.0, "CR": 5.0}
    assert module.keplerian_prop().kind == "keplerian"
    assert repr(module.keplerian_numerical_prop().accel) == "Kep"
    assert "TBmoon" in repr(module.threebody_prop(ode_kwargs={"x": 1}).accel)
    assert module.threebody_prop().accel is module._accel_3_cache
    assert "TBSun" in repr(module.fourbody_prop().accel)
    assert "Drag" in repr(module.best_prop(kwargs={"mass": 1.0}).accel)
    assert "ERad" in repr(module.best_gravity_prop(kwargs={"mass": 1.0}).accel)
    assert "Hmoon:20:20" in repr(module.ssapy_prop(propkw={"mass": 1.0}).accel)


def test_accel_ladder_with_fake_ssapy_modules(monkeypatch):
    import ssapy.accel as accel
    import ssapy.body as body
    import ssapy.gravity as gravity

    monkeypatch.setattr(accel, "AccelKepler", lambda: FakeAccel("Kep"))
    monkeypatch.setattr(accel, "AccelSolRad", lambda **kwargs: FakeAccel("SRP"))
    monkeypatch.setattr(accel, "AccelEarthRad", lambda **kwargs: FakeAccel("ERad"))
    monkeypatch.setattr(body, "get_body", lambda name, **kwargs: name)
    monkeypatch.setattr(gravity, "AccelHarmonic", lambda body, degree, order: FakeAccel(f"H{body}:{degree}:{order}"))
    monkeypatch.setattr(gravity, "AccelThirdBody", lambda body: FakeAccel(f"TB{body}"))

    module = importlib.import_module("ssapy_toolkit.ssapy_wrappers.accel_ladder")
    ladder = module.ssapy_accel_ladder(area=2, mass=10, CR=1.1, CD=2.2)
    assert list(ladder) == [
        "Kep",
        "Kep+Moon",
        "Kep+Moon+J2",
        "Kep+Moon+J2+Sun",
        "Kep+Moon+J2+Sun+Pln",
        "EH140+Moon+Sun+Pln",
        "EH140+MH20+Sun+Pln",
        "EH140+MH20+Sun+Pln+SRP",
        "EH140+MH20+Sun+Pln+SRP+ERad",
    ]
    assert "ERad" in repr(ladder["EH140+MH20+Sun+Pln+SRP+ERad"])


def test_ssapy_orbit_wrappers_success_and_errors(monkeypatch, capsys):
    module = importlib.import_module("ssapy_toolkit.ssapy_wrappers.ssapy_orbits")
    monkeypatch.setattr(module, "Time", FakeTime)
    monkeypatch.setattr(module, "Orbit", FakeOrbit)
    monkeypatch.setattr(module, "get_times", lambda *args, **kwargs: np.array([FakeTime(0), FakeTime(1), FakeTime(2)], dtype=object))
    monkeypatch.setattr(module, "ssapy_kwargs", lambda **kwargs: {"kw": kwargs})
    monkeypatch.setattr(module, "ssapy_prop", lambda **kwargs: SimpleNamespace(prop=kwargs))

    def fake_rv(*args, **kwargs):
        orbit = kwargs.get("orbit", args[0] if args else None)
        time = kwargs.get("time", args[1] if len(args) > 1 else None)
        n = len(time) if isinstance(time, (list, tuple, np.ndarray)) else 1
        return np.repeat(np.asarray(orbit.r).reshape(1, 3), n, axis=0), np.repeat(np.asarray(orbit.v).reshape(1, 3), n, axis=0)

    monkeypatch.setattr(module, "rv", fake_rv)
    times = np.array([FakeTime(10), FakeTime(11)], dtype=object)
    r, v, t = module.ssapy_orbit(r=[1, 2, 3], v=[4, 5, 6], t=times, prop=object())
    assert r.shape == v.shape == (2, 3)
    assert t is times

    r, v, t = module.ssapy_orbit(a=7000e3, e=0.1, integration_timestep=5.0)
    assert r.shape == (3, 3)
    assert "Keplerian elements" in capsys.readouterr().out

    orbit = FakeOrbit([7, 8, 9], [1, 0, 0], t=5.0)
    r, v, t = module.ssapy_orbit(orbit=orbit, t=times, prop=object())
    assert r[0, 0] == 7

    with pytest.raises(ValueError, match="Provide either"):
        module.ssapy_orbit(t=times, prop=object())

    def raising_rv(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "rv", raising_rv)
    r, v, _ = module.ssapy_orbit(r=[1, 2, 3], v=[4, 5, 6], t=times, prop=object())
    assert np.isnan(r).all() and np.isnan(v).all()


def test_ssapy_orbit_incremented_and_similar_orbits(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.ssapy_wrappers.ssapy_orbits")
    monkeypatch.setattr(module, "Time", FakeTime)
    monkeypatch.setattr(module, "Orbit", FakeOrbit)
    monkeypatch.setattr(module, "get_times", lambda *args, **kwargs: np.array([FakeTime(0), FakeTime(1), FakeTime(2)], dtype=object))
    monkeypatch.setattr(module, "ssapy_kwargs", lambda **kwargs: {"kw": kwargs})
    monkeypatch.setattr(module, "ssapy_prop", lambda **kwargs: SimpleNamespace(prop=kwargs))

    def fake_rv(*args, **kwargs):
        orbit = kwargs.get("orbit", args[0] if args else None)
        return np.asarray(orbit.r, dtype=float) + 1.0, np.asarray(orbit.v, dtype=float) + 1.0

    monkeypatch.setattr(module, "rv", fake_rv)
    r_hist, v_hist, t_out = module.ssapy_orbit_incremented(r=[1, 2, 3], v=[4, 5, 6], integration_timestep=2.0)
    assert r_hist.shape == v_hist.shape == (3, 3)
    np.testing.assert_allclose(r_hist[-1], [3, 4, 5])

    with pytest.raises(ValueError, match="Either an Orbit"):
        module.ssapy_orbit_incremented(t=np.array([FakeTime(0)], dtype=object), prop=object())

    def raising_rv(*args, **kwargs):
        raise RuntimeError("stop")

    monkeypatch.setattr(module, "rv", raising_rv)
    short_r, short_v, short_t = module.ssapy_orbit_incremented(r=[1, 2, 3], v=[4, 5, 6])
    assert short_r.shape == short_v.shape == (1, 3)
    assert len(short_t) == 1

    monkeypatch.setattr(module, "points_on_circle", lambda r0, v0, rad, num_points: [r0.reshape(3), r0.reshape(3) + 1.0])
    monkeypatch.setattr(module, "ssapy_orbit", lambda **kwargs: (np.ones((2, 3)), np.zeros((2, 3)), np.array([FakeTime(0), FakeTime(1)], dtype=object)))
    trajectories, times = module.get_similar_orbits([1, 2, 3], [4, 5, 6], num_orbits=2, area=None)
    assert trajectories.shape == (2, 6, 2)
    assert len(times) == 2


def test_quickint_modes_and_period_validation(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.ssapy_wrappers.quick_int")
    monkeypatch.setattr(module, "Time", FakeTime)
    monkeypatch.setattr(module, "Orbit", FakeOrbit)
    monkeypatch.setattr(module, "get_times", lambda *args, **kwargs: np.array([FakeTime(0), FakeTime(1)], dtype=object))
    monkeypatch.setattr(module, "leapfrog", lambda r0, v0, t: (np.repeat(np.asarray(r0).reshape(1, 3), len(t), axis=0), np.repeat(np.asarray(v0).reshape(1, 3), len(t), axis=0)))

    assert module._validate_period(5.0) == 5.0
    for bad in (None, np.inf, np.nan, 99 * 86400.0, 0.0, -1.0):
        with pytest.warns(UserWarning, match="Orbit period"):
            assert module._validate_period(bad, max_period_days=1) == 86400
    assert module._is_position_like([1, 2, 3]) is True
    assert module._is_position_like(object()) is False

    orbit = FakeOrbit([1, 0, 0], [0, 1, 0], t=2.0)
    r, v, t = module.quickint(orbit=orbit)
    assert r.shape == v.shape == (2, 3)
    r, v, t = module.quickint([1, 0, 0], [0, 1, 0])
    assert r.shape == (2, 3)
    r, v, t = module.quickint(r0=[0, 0, 1])
    assert np.isfinite(v).all()
    custom_t = np.array([FakeTime(0), FakeTime(1)], dtype=object)
    r, v, t = module.quickint(r0=[1, 0, 0], v0=[0, 1, 0], t=custom_t)
    assert t is custom_t

    with pytest.raises(ValueError, match="either orbit"):
        module.quickint(orbit=orbit, r0=[1, 0, 0])
    with pytest.raises(ValueError, match="Too many"):
        module.quickint([1, 0, 0], [0, 1, 0], v0=[0, 0, 1])
    with pytest.raises(ValueError, match="either orbit or r0"):
        module.quickint(orbit=object(), r0=[0, 1, 0])
    with pytest.raises(ValueError, match="zero vector"):
        module.quickint(r0=[0, 0, 0])
    with pytest.raises(ValueError, match="Must provide"):
        module.quickint()
