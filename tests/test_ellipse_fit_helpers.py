from types import SimpleNamespace
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pytest

ellipse_module = importlib.import_module("ssapy_toolkit.orbital_mechanics.ellipse_fit")
from ssapy_toolkit.orbital_mechanics.ellipse_fit import delta_v_transfer, ellipse_fit


P1 = np.array([7000e3, 0.0, 0.0])
P2 = np.array([0.0, 8000e3, 0.0])


def test_ellipse_fit_input_modes_and_outputs(tmp_path):
    for kwargs in [
        {},
        {"a_m": 10000e3},
        {"e": 0.25},
        {"F2_m": np.array([1000e3, 1000e3, 0.0])},
        {"inc_deg": 10.0, "v_pref_m_s": np.array([0.0, 7000.0, 500.0])},
    ]:
        fit = ellipse_fit(P1, P2, n_pts=16, plot=False, **kwargs)
        assert fit["r"].shape == fit["v"].shape == (16, 3)
        np.testing.assert_allclose(fit["r"][0], P1, rtol=0, atol=2e3)
        assert np.all(np.isfinite(fit["r"][-1]))
        assert fit["a"] > 0
        assert 0 <= fit["e"] < 1
        assert fit["period"] > 0
        assert fit["rot_dir"] in {-1, 1}

    fit = ellipse_fit(P1, P2, n_pts=12, time_of_departure=100.0)
    assert fit["t_abs"] is not None
    assert fit["t_abs"][0] == pytest.approx(100.0)

    arrival_fit = ellipse_fit(P1, P2, n_pts=12, time_of_arrival=200.0)
    assert arrival_fit["t_abs"][-1] == pytest.approx(200.0)


def test_ellipse_fit_polygon_fallback_for_degenerate_kepler_basis():
    fit = ellipse_fit(
        P1,
        P2,
        F2_m=np.array([0.0, 0.0, 1_000_000.0]),
        n_pts=24,
        v_pref_m_s=np.array([0.0, 7_500.0, 0.0]),
        plot=False,
    )

    assert fit["r"].shape[1] == 3
    assert fit["v"].shape == fit["r"].shape
    assert fit["r"].shape[0] >= 2
    assert np.all(np.isfinite(fit["r"]))
    assert np.all(np.isfinite(fit["v"]))
    assert np.all(np.diff(fit["t_rel"]) >= 0.0)
    assert 0.0 < fit["e"] < 1.0
    assert fit["rot_dir"] in {-1, 1}


def test_ellipse_fit_plot_save_path_with_fake_ssapy_overlay(monkeypatch, tmp_path):
    ssapy_orbits = __import__("ssapy_toolkit.ssapy_wrappers.ssapy_orbits", fromlist=["ssapy_orbit"])

    def fake_ssapy_orbit(r, v, t):
        t = np.asarray(t, dtype=float)
        r0 = np.asarray(r, dtype=float)
        v0 = np.asarray(v, dtype=float)
        return r0[None, :] + 0.01 * t[:, None] * v0[None, :], np.repeat(v0[None, :], len(t), axis=0), t

    saved = []
    monkeypatch.setattr(ssapy_orbits, "ssapy_orbit", fake_ssapy_orbit)
    monkeypatch.setattr(ellipse_module, "save_plot", lambda fig, path: saved.append(path))
    monkeypatch.setattr(plt, "show", lambda: None)

    save_path = tmp_path / "ellipse.png"
    fit = ellipse_fit(P1, P2, n_pts=10, plot=True, save_path=save_path)
    assert fit["r"].shape == (10, 3)
    assert saved == [save_path]


def test_ellipse_fit_validation_branches():
    with pytest.raises(ValueError, match="non-zero"):
        ellipse_fit(np.zeros(3), P2, n_pts=4)
    with pytest.raises(ValueError, match="Specify only one"):
        ellipse_fit(P1, P2, a_m=9000e3, F2_m=np.array([1.0, 0.0, 0.0]))
    with pytest.raises(RuntimeError, match="supplied"):
        ellipse_fit(P1, P2, a_m=100.0)
    with pytest.raises(ValueError, match="0 < e < 1"):
        ellipse_fit(P1, P2, e=0.0)
    with pytest.raises(ValueError, match="0 < e < 1"):
        ellipse_fit(P1, P2, e=2.0)
    with pytest.raises(ValueError, match="either time_of_departure"):
        ellipse_fit(P1, P2, time_of_departure=100.0, time_of_arrival=200.0)
    with pytest.raises(TypeError, match="unexpected keyword"):
        ellipse_fit(P1, P2, bad=True)


def test_delta_v_transfer_vectors_orbit_like_and_errors():
    fit = ellipse_fit(P1, P2, n_pts=8, plot=False)
    depart_v = fit["v"][0] - np.array([1.0, 2.0, 3.0])
    arrive_v = fit["v"][-1] + np.array([4.0, 5.0, 6.0])
    dv = delta_v_transfer(fit, depart_v, arrive_v)
    np.testing.assert_allclose(dv["dv1"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(dv["dv2"], [4.0, 5.0, 6.0])
    assert dv["dv_total"] == pytest.approx(dv["dv1_mag"] + dv["dv2_mag"])
    assert dv["pos_residual_p1"] is None

    depart = SimpleNamespace(r=fit["r"][0], v=depart_v)
    arrive = SimpleNamespace(r=fit["r"][-1], v=arrive_v)
    dv = delta_v_transfer(fit, depart, arrive, check_positions=True, pos_tol_m=10.0)
    assert dv["pos_residual_p1"] == pytest.approx(0.0)
    assert dv["pos_residual_p2"] == pytest.approx(0.0)

    bad_fit = {"v": np.zeros((1, 3)), "r": np.zeros((1, 3))}
    with pytest.raises(ValueError, match=r"fit\['v'\]"):
        delta_v_transfer(bad_fit, np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="expected"):
        delta_v_transfer(fit, np.zeros(2), np.zeros(3))
    bad_depart = SimpleNamespace(r=fit["r"][0] + np.array([1e6, 0, 0]), v=depart_v)
    with pytest.raises(ValueError, match="Departure state"):
        delta_v_transfer(fit, bad_depart, arrive, pos_tol_m=1.0)
    bad_arrive = SimpleNamespace(r=fit["r"][-1] + np.array([1e6, 0, 0]), v=arrive_v)
    with pytest.raises(ValueError, match="Arrival state"):
        delta_v_transfer(fit, depart, bad_arrive, pos_tol_m=1.0)
    bad_shape = SimpleNamespace(r=np.zeros((2, 3)), v=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="single 3-vector"):
        delta_v_transfer(fit, bad_shape, arrive)
