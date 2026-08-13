import importlib

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ssapy_toolkit.constants import EARTH_RADIUS


dash_module = importlib.import_module("ssapy_toolkit.plots.groundtrack_dashboard")
gamma_dash_module = importlib.import_module("ssapy_toolkit.plots.groundtrack_dashboard_gamma_heading")
gamma_module = importlib.import_module("ssapy_toolkit.orbital_mechanics.gamma_and_heading")


def _track_data():
    r1 = np.array([
        [EARTH_RADIUS + 400e3, 0.0, 0.0],
        [0.0, EARTH_RADIUS + 410e3, 0.0],
        [-EARTH_RADIUS - 420e3, 0.0, 100e3],
        [0.0, -EARTH_RADIUS - 430e3, 0.0],
    ])
    r2 = r1 * np.array([1.01, 0.99, 1.0])
    t = np.array([0.0, 60.0, 120.0, 180.0])
    return [r1, r2], [t, t + 10.0]


def _fake_ground_track(xyz, t, format="geodetic"):
    xyz = np.asarray(xyz, dtype=float)
    n = len(xyz)
    if format == "cartesian":
        return xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lon = np.linspace(-np.pi, np.pi, n)
    lat = np.linspace(-0.25 * np.pi, 0.25 * np.pi, n)
    height = np.linalg.norm(xyz, axis=1) - EARTH_RADIUS
    return lon, lat, height


def test_groundtrack_dashboard_smoke_and_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(dash_module, "groundTrack", _fake_ground_track)
    r, t = _track_data()
    fig = dash_module.groundtrack_dashboard(r, t, show=False, show_legend=True, labels=["A"], fontsize=8, limit=8000)
    assert len(fig.axes) >= 4
    plt.close(fig)

    save_path = tmp_path / "dashboard.png"
    fig = dash_module.groundtrack_dashboard(r[0], t[0], show=False, show_legend=False, fontsize=8, save_path=save_path)
    assert save_path.exists()
    plt.close(fig)

    with pytest.raises(TypeError, match="unexpected keyword"):
        dash_module.groundtrack_dashboard(r, t, bad=True)


def test_groundtrack_dashboard_gamma_heading_smoke(monkeypatch, tmp_path):
    monkeypatch.setattr("ssapy.groundTrack", _fake_ground_track)
    monkeypatch.setattr(gamma_module, "calc_gamma_and_heading", lambda xyz, t: (np.linspace(-10, 10, len(xyz)), np.linspace(0, 180, len(xyz))))
    r, t = _track_data()
    fig = gamma_dash_module.groundtrack_dashboard_gamma_heading(r, t, show=False, show_legend=True, fontsize=8, limit=8000)
    assert len(fig.axes) >= 6
    plt.close(fig)

    save_path = tmp_path / "gamma_dashboard.png"
    fig = gamma_dash_module.groundtrack_dashboard_gamma_heading(r[0], t[0], show=False, show_legend=False, fontsize=8, save_path=save_path)
    assert save_path.exists()
    plt.close(fig)

    def bad_gamma(xyz, t):
        raise RuntimeError("boom")

    monkeypatch.setattr(gamma_module, "calc_gamma_and_heading", bad_gamma)
    fig = gamma_dash_module.groundtrack_dashboard_gamma_heading(r[0], t[0], show=False, show_legend=False, fontsize=8, limit=8000)
    assert len(fig.axes) >= 6
    plt.close(fig)

    with pytest.raises(TypeError, match="unexpected keyword"):
        gamma_dash_module.groundtrack_dashboard_gamma_heading(r, t, bad=True)
