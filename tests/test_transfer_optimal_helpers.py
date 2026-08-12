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


def test_optimal_time_and_orbit_helpers(monkeypatch):
    assert tof._to_gps_seconds(Time(5.0, format="gps")) == pytest.approx(5.0)
    orbit = tof._as_orbit(_state(t=Time(0.0, format="gps")), EARTH_MU)
    assert orbit.r.shape == (3,)
    mapped_orbit = tof._as_orbit({"r": orbit.r, "v": orbit.v, "t": 12.0}, EARTH_MU)
    assert mapped_orbit.t == pytest.approx(12.0)
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
    with pytest.raises(ValueError, match="delta_v_mode"):
        tof.transfer_optimal(dep, arr, delta_v_mode="bad", n_grid=(2, 2), polish=False)
    with pytest.raises(ValueError, match="requires arrival_burn"):
        tof.transfer_optimal(dep, arr, delta_v_mode="last", arrival_burn=False, n_grid=(2, 2), polish=False)


class FakeOrbit:
    def __init__(self, r, v, t=0.0, mu=EARTH_MU):
        self.r = np.asarray(r, dtype=float)
        self.v = np.asarray(v, dtype=float)
        self.t = float(t)
        self.mu = mu
        self.a = max(np.linalg.norm(self.r), 1.0)


def _install_fast_optimal_fakes(monkeypatch, dv=10.0, dv1=None, dv2=0.0):
    dv1 = float(dv if dv1 is None else dv1)
    dv2 = float(dv2)
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
        return (
            np.asarray(r1, dtype=float) * 0.0 + np.array([dv1, 0.0, 0.0]),
            np.asarray(r2, dtype=float) * 0.0 + np.array([0.0, dv2, 0.0]),
        )

    def fake_transfer(departure, arrival, **kwargs):
        r1, v1, t1 = departure
        r2, v2, t2 = arrival
        transfer_orbit = FakeOrbit(r1, np.array([1.0, 0.0, 0.0]), t1)
        arrival_burn = kwargs.get("arrival_burn", True)
        magnitudes = [dv1] + ([dv2] if arrival_burn else [])
        burns = [
            {"delta_v_mag": value, "delta_v": np.array([value, 0.0, 0.0])}
            for value in magnitudes
        ]
        return {
            "schema_version": "ssatk.transfer.v2",
            "method": "transfer_ssapy",
            "initial": {"r": np.asarray(r1), "v": np.asarray(v1), "t": float(t1)},
            "target": {"r": np.asarray(r2), "v": np.asarray(v2), "t": float(t2)},
            "final": {"r": np.asarray(r2), "v": np.asarray(v2), "t": float(t2)},
            "tof": float(t2) - float(t1),
            "burns": burns,
            "delta_v_total": float(sum(magnitudes)),
            "delta_v_vectors": [burn["delta_v"] for burn in burns],
            "delta_v_ntw_vectors": [burn["delta_v"] for burn in burns],
            "delta_v_magnitudes": magnitudes,
            "trajectory": None,
            "transfer_orbits": [transfer_orbit],
            "hardware": {},
            "success": True,
            "assumptions": ["fake transfer"],
            "diagnostics": {"arrival_error": 0.0},
        }

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
    assert result["method"] == "transfer_optimal"
    assert result["diagnostics"]["rendezvous"] is True
    assert result["diagnostics"]["delta_v_mode"] == "total"
    assert result["diagnostics"]["objective_delta_v"] == pytest.approx(12.0)
    assert result["diagnostics"]["arrival_phase"] is None
    assert result["diagnostics"]["grid"]["cost"].shape == (2, 3)
    assert result["diagnostics"]["pareto"]["dv"].shape == (3,)
    assert plotted == [str(tmp_path / "optimal") + "_designer_curves.jpg"]


def test_transfer_optimal_orbit_workflow_burn_cost_modes(monkeypatch):
    _install_fast_optimal_fakes(monkeypatch, dv1=30.0, dv2=7.0)
    dep = tof.Orbit(*_state(radius=7000e3, theta=0.0, t=0.0))
    arr = tof.Orbit(*_state(radius=7200e3, theta=0.2, t=5.0))

    both = tof.transfer_optimal(
        dep,
        arr,
        objective="delta_v",
        delta_v_mode="both",
        t_window=(0.0, 10.0),
        tof_range=(20.0, 30.0),
        n_grid=(2, 2),
        polish=False,
        refine=False,
    )
    first = tof.transfer_optimal(
        dep,
        arr,
        delta_v_mode="departure",
        arrival_burn=False,
        t_window=(0.0, 10.0),
        tof_range=(20.0, 30.0),
        n_grid=(2, 2),
        polish=False,
        refine=False,
    )
    last = tof.transfer_optimal(
        dep,
        arr,
        delta_v_mode="arrival",
        t_window=(0.0, 10.0),
        tof_range=(20.0, 30.0),
        n_grid=(2, 2),
        polish=False,
        refine=False,
    )

    assert both["diagnostics"]["delta_v_mode"] == "total"
    assert both["diagnostics"]["objective_delta_v"] == pytest.approx(37.0)
    assert both["diagnostics"]["grid"]["cost"][0, 0] == pytest.approx(37.0)

    assert first["diagnostics"]["delta_v_mode"] == "first"
    assert first["diagnostics"]["arrival_burn"] is False
    assert first["diagnostics"]["objective_delta_v"] == pytest.approx(30.0)
    assert len(first["burns"]) == 1

    assert last["diagnostics"]["delta_v_mode"] == "last"
    assert last["diagnostics"]["objective_delta_v"] == pytest.approx(7.0)
    assert last["diagnostics"]["grid"]["cost"][0, 0] == pytest.approx(7.0)


def test_transfer_optimal_min_time_budget_uses_selected_delta_v_mode(monkeypatch):
    _install_fast_optimal_fakes(monkeypatch, dv1=100.0, dv2=8.0)
    dep = _state(radius=7000e3, theta=0.0, t=0.0)
    arr = _state(radius=7200e3, theta=0.3, t=15.0)

    result = tof.transfer_optimal(
        dep,
        arr,
        objective="time",
        delta_v_mode="last",
        dv_budget=9.0,
        t_window=(0.0, 10.0),
        tof_range=(20.0, 40.0),
        n_grid=(2, 2),
        polish=False,
        refine=False,
    )
    assert result["diagnostics"]["objective"] == "min_time"
    assert result["diagnostics"]["delta_v_mode"] == "last"
    assert result["diagnostics"]["objective_delta_v"] == pytest.approx(8.0)
    assert result["diagnostics"]["within_delta_v_budget"] is True
    assert result["delta_v_total"] == pytest.approx(108.0)

    with pytest.raises(ValueError, match="No transfer"):
        tof.transfer_optimal(
            dep,
            arr,
            objective="time",
            delta_v_mode="last",
            dv_budget=7.0,
            t_window=(0.0, 10.0),
            tof_range=(20.0, 40.0),
            n_grid=(2, 2),
            polish=False,
            refine=False,
        )


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
    assert result["diagnostics"]["objective"] == "min_time"
    assert result["diagnostics"]["arrival_phase"] is not None
    assert result["diagnostics"]["arrival_burn"] is False
    assert result["diagnostics"]["grid"]["cost"].shape == (2, 2)


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
