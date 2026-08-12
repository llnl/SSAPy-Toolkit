import importlib

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

    def __sub__(self, other):
        return self.gps - FakeTime(other).gps

    def __repr__(self):
        return f"FakeTime({self.gps})"


class FakeOrbit:
    def __init__(self, r=None, v=None, t=0.0, mu=None, **kwargs):
        self.r = np.asarray(r if r is not None else [7000e3, 0.0, 0.0], dtype=float)
        self.v = np.asarray(v if v is not None else [0.0, 7500.0, 0.0], dtype=float)
        self.t = t
        self.mu = mu
        radius = np.linalg.norm(self.r) or 1.0
        self.a = float(kwargs.get("a", radius))
        self.e = float(kwargs.get("e", 0.0))
        self.i = float(kwargs.get("i", 0.0))
        self.pa = float(kwargs.get("pa", 0.1))
        self.raan = float(kwargs.get("raan", 0.0))
        self.meanAnomaly = float(kwargs.get("meanAnomaly", 0.0))
        self.period = float(kwargs.get("period", 100.0))
        self.periapsis = np.array([self.a * (1.0 - self.e), 0.0, 0.0])
        self.apoapsis = np.array([self.a * (1.0 + self.e), 0.0, 0.0])

    @classmethod
    def fromKeplerianElements(cls, a, e, i, pa, raan, true_anomaly, t=0.0, mu=None):
        speed = np.sqrt((mu or 3.986004418e14) / max(float(a), 1.0))
        radius = a * (1.0 + e)
        direction = np.array([np.cos(true_anomaly), np.sin(true_anomaly), 0.0])
        tangent = np.array([-np.sin(true_anomaly), np.cos(true_anomaly), 0.0])
        return cls(
            r=radius * direction,
            v=speed * tangent,
            t=t,
            mu=mu,
            a=a,
            e=e,
            i=i,
            pa=pa,
            raan=raan,
            meanAnomaly=true_anomaly,
        )

    def at(self, t):
        return FakeOrbit(
            self.r,
            self.v,
            t,
            self.mu,
            a=self.a,
            e=self.e,
            i=self.i,
            pa=self.pa,
            raan=self.raan,
            meanAnomaly=self.meanAnomaly,
            period=self.period,
        )


def _state(radius=7000e3, theta=0.0):
    r = radius * np.array([np.cos(theta), np.sin(theta), 0.0])
    v = np.sqrt(3.986004418e14 / radius) * np.array([-np.sin(theta), np.cos(theta), 0.0])
    return r, v


def test_hohmann_fake_orbit_modes_and_equal_radius(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_hohmann")
    monkeypatch.setattr(module, "Orbit", FakeOrbit)
    monkeypatch.setattr(module, "to_gps", lambda value: 0.0)
    monkeypatch.setattr(module, "ssapy_orbit", lambda **kwargs: (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), None, [0, 1]))
    monkeypatch.setattr(module.plt, "show", lambda: None)

    ntw = module.velocity_to_ntw(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(ntw, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(module._tangential_direction(np.array([0.0, 0.0, 1.0])), [1.0, 0.0, 0.0])

    outward = module.transfer_hohmann([7000e3, 0.1, 0.0, 0.1, 0.0, 0.0], [9000e3, 0.1, 0.2, 0.2, 0.0, 0.0], plot=True)
    assert outward["tof"] > 0.0
    assert "fig" in outward
    assert outward["delta_ntw1"].shape == (3,)

    inward = module.transfer_hohmann([9000e3, 0.1, 0.0, 0.1, 0.0, np.pi], [7000e3, 0.1, 0.0, 0.2, 0.0, 0.0])
    assert inward["tof"] > 0.0

    equal = module.transfer_hohmann([7000e3, 0.0, 0.0, 0.1, 0.0, 0.0], [7000e3, 0.0, 0.0, 0.2, 0.0, 0.0])
    assert equal["tof"] == 0.0
    assert equal["|delta_v2|"] == 0.0

    r1, v1 = _state(7000e3)
    r2, v2 = _state(8000e3, np.pi / 2)
    state_result = module.transfer_hohmann(r1, v1, r2, v2)
    assert state_result["initial"].r.shape == (3,)


def test_hohmann_validation_branches(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_hohmann")
    monkeypatch.setattr(module, "Orbit", FakeOrbit)
    monkeypatch.setattr(module, "to_gps", lambda value: 0.0)

    with pytest.raises(ValueError, match="Positional arguments"):
        module.transfer_hohmann(1, 2, 3, 4, 5)
    with pytest.raises(ValueError, match="Two positional"):
        module.transfer_hohmann(object(), object())
    with pytest.raises(ValueError, match="orbit1"):
        module.transfer_hohmann(orbit1=object(), orbit2=FakeOrbit())
    with pytest.raises(ValueError, match="orbit2"):
        module.transfer_hohmann(orbit1=FakeOrbit(), orbit2=object())
    with pytest.raises(ValueError, match="elements1"):
        module.transfer_hohmann([1, 2, 3], [1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError, match="elements2"):
        module.transfer_hohmann([1, 2, 3, 4, 5, 6], [1, 2, 3])
    with pytest.raises(TypeError, match="elements1"):
        module.transfer_hohmann(elements1=object(), elements2=[1, 2, 3, 4, 5, 6])
    with pytest.raises(TypeError, match="elements2"):
        module.transfer_hohmann(elements1=[1, 2, 3, 4, 5, 6], elements2=object())


def test_lambertian_find_intersection_and_fake_transfer(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_lambertian")

    def fake_leapfrog(r_start, v_start, t):
        t = np.asarray(t, dtype=float)
        r = np.column_stack((t, 2.0 * t, np.zeros_like(t)))
        v = np.ones_like(r)
        return r, v

    monkeypatch.setattr(module, "leapfrog", fake_leapfrog)
    tof, r_transfer, v_transfer = module.find_intersection_time(np.zeros(3), np.ones(3), np.array([2.0, 4.0, 0.0]), 5)
    assert tof == 2
    assert r_transfer.shape == v_transfer.shape == (3, 3)

    monkeypatch.setattr(module, "Orbit", FakeOrbit)
    monkeypatch.setattr(module, "Time", FakeTime)

    def fake_intersection(r1, v1_t, r2, t_max):
        return 3.0, np.vstack([r1, 0.5 * (r1 + r2), r2]), np.vstack([v1_t, v1_t, v1_t])

    monkeypatch.setattr(module, "find_intersection_time", fake_intersection)
    r1, v1 = _state(7000e3)
    r2, v2 = _state(8000e3, np.pi / 2)
    result = module.transfer_lambertian(r1, v1, r2, v2, MIN_PERIGEE=1.0)
    assert result["orbit_type"] in {"ellipse", "hyperbola"}
    assert result["r_transfer"].shape == (3, 3)

    elements_result = module.transfer_lambertian([7000e3, 0.0, 0.0, 0.0, 0.0, 0.0], [8000e3, 0.0, 0.0, 0.0, 0.0, 1.0], MIN_PERIGEE=1.0)
    assert elements_result["initial"].r.shape == (3,)


def test_lambertian_validation_and_plot(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_lambertian")
    plots = importlib.import_module("ssapy_toolkit.plots")
    monkeypatch.setattr(module, "Orbit", FakeOrbit)
    monkeypatch.setattr(module, "Time", FakeTime)
    monkeypatch.setattr(module, "find_intersection_time", lambda r1, v1_t, r2, t_max: (1.0, np.vstack([r1, r2]), np.vstack([v1_t, v1_t])))
    monkeypatch.setattr(plots, "transfer_plot", lambda *args, **kwargs: "fake-fig")

    r1, v1 = _state(7000e3)
    r2, v2 = _state(8000e3, np.pi / 2)
    plotted = module.transfer_lambertian(r1, v1, r2, v2, MIN_PERIGEE=1.0, plot=True)
    assert plotted["fig"] == "fake-fig"

    with pytest.raises(ValueError, match="Positional arguments"):
        module.transfer_lambertian(1, 2, 3, 4, 5)
    with pytest.raises(ValueError, match="Two positional"):
        module.transfer_lambertian(object(), object())
    with pytest.raises(ValueError, match="elements1"):
        module.transfer_lambertian([1, 2], [1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError, match="elements2"):
        module.transfer_lambertian([1, 2, 3, 4, 5, 6], [1, 2])
    with pytest.raises(ValueError, match="too low"):
        module.transfer_lambertian(np.array([1.0, 0.0, 0.0]), v1, r2, v2)


def test_coplanar_transfer_fake_propagation_and_validation(monkeypatch, capsys):
    module = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_coplanar")
    plots = importlib.import_module("ssapy_toolkit.plots")
    monkeypatch.setattr(module, "Orbit", FakeOrbit)
    monkeypatch.setattr(module, "Time", FakeTime)
    monkeypatch.setattr(module, "get_times", lambda *args, **kwargs: np.array([FakeTime(0.0), FakeTime(1.0), FakeTime(2.0)], dtype=object))
    monkeypatch.setattr(plots, "transfer_plot", lambda *args, **kwargs: "fake-fig")

    def fake_rv(orbit, time, propagator=None):
        n = len(time) if isinstance(time, (list, tuple, np.ndarray)) else 1
        return np.repeat(np.asarray(orbit.r, dtype=float).reshape(1, 3), n, axis=0), np.repeat(np.asarray(orbit.v, dtype=float).reshape(1, 3), n, axis=0)

    monkeypatch.setattr(module, "rv", fake_rv)
    r1, v1 = _state(7000e3)
    result = module.transfer_coplanar(r1, v1, r1, v1, tol=1.0, plot=True, status=True)
    assert result["fig"] == "fake-fig"
    assert result["error"] == pytest.approx(0.0)
    assert "Done" in capsys.readouterr().out

    singular = module.transfer_coplanar(r1, v1, r1 + np.array([1.0, 0.0, 0.0]), v1, tol=0.0, max_iter=1, status=True)
    assert singular["error"] > 0.0
    assert "Singular J" in capsys.readouterr().out

    with pytest.raises(ValueError, match="Positional args"):
        module.transfer_coplanar(1, 2)
    with pytest.raises(ValueError, match="both r1"):
        module.transfer_coplanar(r1=None, v1=v1, r2=r1)
    with pytest.raises(ValueError, match="orbit2 or r2"):
        module.transfer_coplanar(r1=r1, v1=v1)
