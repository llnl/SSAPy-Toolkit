import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.time import Time

import ssapy_toolkit.orbital_mechanics.orbital_accel_model_comparisons as module


class FakeOrbit:
    def __init__(self, r, v, t):
        self.r = np.asarray(r, dtype=float)
        self.v = np.asarray(v, dtype=float)
        self.t = t


class FakePropagator:
    def __init__(self, accel, ode_kwargs=None):
        self.accel = accel
        self.ode_kwargs = ode_kwargs


class FakeAccel:
    def __init__(self, name):
        self.name = name

    def __add__(self, other):
        return FakeAccel(f"{self.name}+{other.name}")


def _install_fake_ssapy(monkeypatch, fake_rv):
    fake_ssapy = types.ModuleType("ssapy")
    fake_ssapy.Orbit = FakeOrbit
    fake_compute = types.ModuleType("ssapy.compute")
    fake_compute.rv = fake_rv
    fake_propagator = types.ModuleType("ssapy.propagator")
    fake_propagator.SciPyPropagator = FakePropagator
    monkeypatch.setitem(sys.modules, "ssapy", fake_ssapy)
    monkeypatch.setitem(sys.modules, "ssapy.compute", fake_compute)
    monkeypatch.setitem(sys.modules, "ssapy.propagator", fake_propagator)


def test_time_coercion_and_keplerian_edge_branches():
    astropy_time = Time([0.0, 10.0], format="gps", scale="utc")
    assert module._orbit_epoch_gps(FakeOrbit([1, 0, 0], [0, 1, 0], astropy_time[0])) == pytest.approx(astropy_time[0].gps)
    with pytest.raises(ValueError, match="epoch"):
        module._orbit_epoch_gps(types.SimpleNamespace(r=np.zeros(3), v=np.zeros(3)))

    assert module._coerce_times_for_ssapy(astropy_time, astropy_time[0].gps) is astropy_time
    converted = module._coerce_times_for_ssapy(np.array(["2026-01-01"], dtype="datetime64[D]"), astropy_time[0].gps)
    assert isinstance(converted, Time)
    object_astropy = module._coerce_times_for_ssapy(np.array([astropy_time[0], astropy_time[1]], dtype=object), astropy_time[0].gps)
    assert isinstance(object_astropy, Time)
    object_datetime = module._coerce_times_for_ssapy(np.array([np.datetime64("2026-01-01")], dtype=object), astropy_time[0].gps)
    assert isinstance(object_datetime, Time)
    np.testing.assert_allclose(module._coerce_times_for_ssapy([1.0, 2.0], 1.2e9, assume="gps"), [1.0, 2.0])
    np.testing.assert_allclose(module._coerce_times_for_ssapy([1.0, 2.0], 1.2e9, assume="offset"), [1.2e9 + 1.0, 1.2e9 + 2.0])
    np.testing.assert_allclose(module._coerce_times_for_ssapy([1.0, 2.0], 1.2e9, assume="auto"), [1.2e9 + 1.0, 1.2e9 + 2.0])
    np.testing.assert_allclose(module._times_to_relative_seconds(astropy_time), [0.0, 10.0])

    mu = 3.986004418e14
    r = np.array([7_000_000.0, 0.0, 0.0])
    parabolic = module._keplerian_elements_from_rv(r, [0.0, np.sqrt(2.0 * mu / np.linalg.norm(r)), 0.0], mu)
    assert np.isinf(parabolic["a_m"])
    assert "T = n/a" in module._format_oe_text(parabolic)
    radial = module._keplerian_elements_from_rv(r, [1.0, 0.0, 0.0], mu)
    assert radial["i_deg"] == pytest.approx(0.0)
    retro = module._keplerian_elements_from_rv([0.0, -7_000_000.0, -10.0], [-7_500.0, 0.0, -100.0], mu)
    assert retro["nu_deg"] >= 0.0


def test_small_ladder_color_branches_and_calculation(monkeypatch):
    fake_accel_mod = types.ModuleType("ssapy.accel")
    fake_accel_mod.AccelKepler = lambda: FakeAccel("kepler")
    fake_body_mod = types.ModuleType("ssapy.body")
    fake_body_mod.get_body = lambda *args, **kwargs: types.SimpleNamespace(args=args, kwargs=kwargs)
    fake_gravity_mod = types.ModuleType("ssapy.gravity")
    fake_gravity_mod.AccelThirdBody = lambda body: FakeAccel("third")
    fake_gravity_mod.AccelHarmonic = lambda body, degree, order: FakeAccel(f"harmonic{degree}{order}")
    monkeypatch.setitem(sys.modules, "ssapy.accel", fake_accel_mod)
    monkeypatch.setitem(sys.modules, "ssapy.body", fake_body_mod)
    monkeypatch.setitem(sys.modules, "ssapy.gravity", fake_gravity_mod)

    assert list(module._small_accel_ladder(1)) == ["Kep"]
    assert list(module._small_accel_ladder(2)) == ["Kep", "Kep+Moon"]
    assert list(module._small_accel_ladder(3)) == ["Kep", "Kep+Moon", "Kep+Moon+J2"]
    assert len(module._nice_vivid_colors(15)) == 15
    assert len(module._nice_vivid_colors(25)) == 25

    def fake_rv(orbit, times, prop):
        t = np.asarray(times, dtype=float)
        rel = t - t[0]
        offset = float(prop.accel)
        r_hist = orbit.r + rel[:, None] * orbit.v + offset
        v_hist = np.repeat(orbit.v.reshape(1, 3), len(t), axis=0) + offset
        if offset == 1.0:
            r_hist = r_hist[:2]
            v_hist = v_hist[:2]
        if offset == 2.0:
            r_hist = r_hist.reshape(1, *r_hist.shape)
            v_hist = v_hist.reshape(1, *v_hist.shape)
        return r_hist, v_hist

    _install_fake_ssapy(monkeypatch, fake_rv)
    monkeypatch.setattr(module, "_small_accel_ladder", lambda max_rungs: {"kep": 0.0, "moon": 1.0, "j2": 2.0})
    result = module.calculate_accel_comparisons(
        r=[7_000_000.0, 0.0, 0.0],
        v=[0.0, 7_500.0, 0.0],
        t0=1.2e9,
        times=[0.0, 1.0, 2.0],
        max_rungs=3,
        reference=1,
        ode_kwargs={"rtol": 1e-9},
    )
    assert result["common_len"] == 2
    assert result["stop_idx"].tolist() == [2, 1, 2]
    assert result["r_list"][1].shape == (2, 3)
    assert result["worst_idx"] in {0, 2}

    fake_ladder = types.ModuleType("ssapy_toolkit.ssapy_wrappers.accel_ladder")
    fake_ladder.ssapy_accel_ladder = lambda: {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}
    monkeypatch.setitem(sys.modules, "ssapy_toolkit.ssapy_wrappers.accel_ladder", fake_ladder)
    four = module.calculate_accel_comparisons(
        orbit=FakeOrbit([7_000_000.0, 0.0, 0.0], [0.0, 7_500.0, 0.0], 1.2e9),
        times=np.array([1.2e9, 1.2e9 + 1.0]),
        assume_times="gps",
        max_rungs=4,
    )
    assert four["labels"] == ["a", "b", "c", "d"]


def test_accel_comparison_validation_and_dashboard_wrapper(monkeypatch):
    with pytest.raises(ValueError, match="times"):
        module.calculate_accel_comparisons(orbit=FakeOrbit([1, 0, 0], [0, 1, 0], 0.0))

    def fake_rv_empty(orbit, times, prop):
        return np.empty((0, 3)), np.empty((0, 3))

    _install_fake_ssapy(monkeypatch, fake_rv_empty)
    monkeypatch.setattr(module, "_small_accel_ladder", lambda max_rungs: {"kep": 0.0, "moon": 1.0})
    with pytest.raises(ValueError, match="Provide either"):
        module.calculate_accel_comparisons(times=[0.0, 1.0], max_rungs=2)
    with pytest.raises(ValueError, match="missing `.r`"):
        module.calculate_accel_comparisons(orbit=types.SimpleNamespace(v=np.ones(3), t=0.0), times=[0.0, 1.0], max_rungs=2)
    with pytest.raises(ValueError, match="max_rungs"):
        module.calculate_accel_comparisons(orbit=FakeOrbit([1, 0, 0], [0, 1, 0], 0.0), times=[0.0, 1.0], max_rungs=1)
    with pytest.raises(ValueError, match="reference"):
        module.calculate_accel_comparisons(orbit=FakeOrbit([1, 0, 0], [0, 1, 0], 0.0), times=[0.0, 1.0], max_rungs=2, reference=5)
    with pytest.raises(ValueError, match="empty histories"):
        module.calculate_accel_comparisons(orbit=FakeOrbit([1, 0, 0], [0, 1, 0], 0.0), times=[0.0, 1.0], max_rungs=2)

    calc = {
        "labels": ["a", "b"],
        "reference": 1,
        "t_rel_s": np.array([0.0, 1.0, 2.0]),
        "drn_vs_ref": np.array([[0.2, 0.4, 0.8], [0.0, 0.0, 0.0]]),
        "drn_inc": np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]),
        "worst_idx": 0,
        "ntw_worst": np.ones((3, 3)),
        "final_drn_vs_ref": np.array([0.8, 0.0]),
        "final_drn_inc": np.array([0.0, 0.3]),
        "orbit_elements_text": "OE text",
    }
    figs = module.make_accel_ladder_dashboard_figures(calc=calc, show_legend=False)
    assert len(figs["figures"]) == 2
    for fig in figs["figures"]:
        plt.close(fig)
