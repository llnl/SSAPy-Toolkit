from types import SimpleNamespace

import numpy as np
import pytest
from astropy.time import Time

from ssapy_toolkit.constants import EARTH_MU
from ssapy_toolkit.orbital_mechanics import transfer_optimal_function as tof


def _state(radius=7000e3, theta=0.0, t=0.0):
    r = radius * np.array([np.cos(theta), np.sin(theta), 0.0])
    v = np.sqrt(EARTH_MU / radius) * np.array([-np.sin(theta), np.cos(theta), 0.0])
    return r, v, t


def test_optimal_transfer_result_summary_and_helpers(monkeypatch):
    transfer = SimpleNamespace(summary=lambda: "transfer summary")
    result = tof.OptimalTransferResult(
        transfer=transfer,
        t_depart=10.0,
        t_arrive=110.0,
        tof=100.0,
        dv_total=42.0,
        prograde=False,
        arrival_phase=12.0,
        objective="min_dv",
        rendezvous=False,
        arrival_burn=False,
        perigee_altitude=500e3,
        grid={"t_dep": np.array([0.0, 10.0]), "feasible_fraction": 0.5},
        pareto={"tof": np.array([100.0]), "dv": np.array([42.0])},
    )
    summary = result.summary()
    assert "insertion" in summary
    assert "retrograde" in summary
    assert "Arrival phase" in summary
    assert "transfer summary" in summary

    assert tof._to_gps_seconds(Time(5.0, format="gps")) == pytest.approx(5.0)
    orbit = tof._as_orbit(_state(t=Time(0.0, format="gps")), EARTH_MU)
    assert orbit.r.shape == (3,)
    assert tof._period(orbit, EARTH_MU) > 0.0
    with pytest.raises(ValueError, match="closed"):
        tof._period(SimpleNamespace(a=-1.0), EARTH_MU)

    r = np.array([7000e3, 0.0, 0.0])
    v = np.array([0.0, np.sqrt(EARTH_MU / 7000e3), 0.0])
    assert tof._conic_perigee(r, v, EARTH_MU) == pytest.approx(7000e3, rel=1e-6)

    def fake_rv(_orbit, times, propagator=None):
        times = np.asarray(times, dtype=float)
        return np.column_stack((times, times + 1, times + 2)), np.column_stack((times + 3, times + 4, times + 5))

    monkeypatch.setattr(tof, "rv", fake_rv)
    rr, vv = tof._ephemeris(orbit, np.array([2.0, 0.0, 1.0]))
    np.testing.assert_allclose(rr[:, 0], [2.0, 0.0, 1.0])
    np.testing.assert_allclose(vv[:, 0], [5.0, 3.0, 4.0])


def test_transfer_optimal_early_validation_branches():
    dep = _state(t=0.0)
    arr = _state(theta=0.1, t=10.0)
    with pytest.raises(ValueError, match="objective"):
        tof.transfer_optimal(dep, arr, objective="bad", n_grid=(2, 2), polish=False)
    with pytest.raises(ValueError, match="requires dv_budget"):
        tof.transfer_optimal(dep, arr, objective="min_time", n_grid=(2, 2), polish=False)
    with pytest.raises(ValueError, match="thrust and mass"):
        tof.transfer_optimal(dep, arr, thrust=1.0, n_grid=(2, 2), polish=False)
    with pytest.raises(ValueError, match="either burn_accel"):
        tof.transfer_optimal(dep, arr, thrust=1.0, mass=1.0, burn_accel=1.0, n_grid=(2, 2), polish=False)


class FakeOrbit:
    def __init__(self, r, v, t=0.0, mu=EARTH_MU):
        self.r = np.asarray(r, dtype=float)
        self.v = np.asarray(v, dtype=float)
        self.t = float(t)
        self.mu = mu
        self.a = max(np.linalg.norm(self.r), 1.0)


class FakeTransfer(SimpleNamespace):
    def summary(self):
        return "fake transfer summary"


def _install_fast_optimal_fakes(monkeypatch, dv=10.0):
    monkeypatch.setattr(tof, "Orbit", FakeOrbit)
    monkeypatch.setattr(tof, "_conic_perigee", lambda r, v, mu: tof.EARTH_RADIUS + 1e6)

    def fake_rv(orbit, times, propagator=None):
        times = np.asarray(times, dtype=float)
        dt = times - float(orbit.t)
        r = np.asarray(orbit.r, dtype=float)[None, :] + dt[:, None] * np.asarray(orbit.v, dtype=float)[None, :] * 0.01
        v = np.zeros_like(r)
        return r, v

    def fake_solve_lambert(r1, r2, tof_seconds, mu=EARTH_MU, prograde=True, max_iter=60, tol=1e-6):
        if not prograde:
            raise RuntimeError("fake retrograde miss")
        return np.asarray(r1, dtype=float) * 0.0 + np.array([dv, 0.0, 0.0]), np.asarray(r2, dtype=float) * 0.0

    def fake_transfer(departure, arrival, **kwargs):
        r1, v1, t1 = departure
        transfer_orbit = FakeOrbit(r1, np.array([1.0, 0.0, 0.0]), t1)
        return FakeTransfer(dv_total=float(dv), transfer_orbit=transfer_orbit)

    monkeypatch.setattr(tof, "rv", fake_rv)
    monkeypatch.setattr(tof, "solve_lambert", fake_solve_lambert)
    monkeypatch.setattr(tof, "transfer_ssapy", fake_transfer)


def test_transfer_optimal_rendezvous_and_visualization(monkeypatch, tmp_path):
    _install_fast_optimal_fakes(monkeypatch, dv=12.0)
    plot_module = __import__("ssapy_toolkit.plots.transfer_designer_curves_plot", fromlist=["transfer_designer_curves_plot"])
    plotted = []
    monkeypatch.setattr(plot_module, "transfer_designer_curves_plot", lambda result, save_path: plotted.append(save_path))

    dep = _state(radius=7000e3, theta=0.0, t=0.0)
    arr = _state(radius=7100e3, theta=0.2, t=20.0)
    result = tof.transfer_optimal(
        dep,
        arr,
        objective="min_dv",
        rendezvous=True,
        arrival_burn=True,
        t_window=(0.0, 20.0),
        tof_range=(20.0, 40.0),
        n_grid=(2, 3),
        polish=False,
        visualize=True,
        fig_prefix=str(tmp_path / "optimal"),
        refine=False,
    )
    assert result.rendezvous is True
    assert result.arrival_phase is None
    assert result.grid["cost"].shape == (2, 3)
    assert result.pareto["dv"].shape == (3,)
    assert plotted == [str(tmp_path / "optimal") + "_designer_curves.jpg"]
    assert "fake transfer summary" in result.summary()


def test_transfer_optimal_insertion_min_time_and_polish(monkeypatch):
    _install_fast_optimal_fakes(monkeypatch, dv=8.0)
    dep = _state(radius=7000e3, theta=0.0, t=0.0)
    r2, v2, t2 = _state(radius=7200e3, theta=0.3, t=15.0)
    arr = (r2, -v2, t2)
    result = tof.transfer_optimal(
        dep,
        arr,
        objective="min_time",
        dv_budget=20.0,
        rendezvous=False,
        arrival_burn=False,
        t_window=(0.0, 10.0),
        tof_range=(10.0, 30.0),
        n_grid=(2, 2),
        n_phase=3,
        polish=True,
        refine=False,
    )
    assert result.objective == "min_time"
    assert result.arrival_phase is not None
    assert result.arrival_burn is False
    assert result.grid["cost"].shape == (2, 2)


def test_transfer_optimal_no_feasible_and_budget_branches(monkeypatch):
    monkeypatch.setattr(tof, "Orbit", FakeOrbit)
    monkeypatch.setattr(tof, "rv", lambda orbit, times, propagator=None: (np.repeat(np.asarray(orbit.r)[None, :], len(times), axis=0), np.repeat(np.asarray(orbit.v)[None, :], len(times), axis=0)))
    monkeypatch.setattr(tof, "solve_lambert", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no solution")))
    dep = _state(radius=7000e3, theta=0.0, t=0.0)
    arr = _state(radius=7100e3, theta=0.2, t=20.0)
    with pytest.raises(RuntimeError, match="No feasible"):
        tof.transfer_optimal(dep, arr, t_window=(0.0, 10.0), tof_range=(10.0, 20.0), n_grid=(2, 2), polish=False)

    _install_fast_optimal_fakes(monkeypatch, dv=100.0)
    with pytest.raises(ValueError, match="No transfer"):
        tof.transfer_optimal(dep, arr, objective="min_time", dv_budget=1.0, t_window=(0.0, 10.0), tof_range=(10.0, 20.0), n_grid=(2, 2), polish=False)
