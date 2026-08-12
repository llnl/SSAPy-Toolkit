from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.time import Time

from ssapy_toolkit.constants import RGEO
from ssapy_toolkit.plots import misc_plotting


def test_koe_plot_with_monkeypatched_elements(monkeypatch, tmp_path):
    import ssapy_toolkit.orbital_mechanics as orbital_mechanics

    def fake_elements(r, v, mu_barycenter):
        n = len(r)
        return {
            "a": np.linspace(RGEO, RGEO * 1.1, n),
            "e": np.linspace(0.01, 0.02, n),
            "i": np.linspace(0.1, 0.2, n),
        }

    monkeypatch.setattr(orbital_mechanics, "calculate_orbital_elements", fake_elements, raising=False)
    r = np.zeros((3, 3))
    v = np.ones((3, 3))
    fig, ax = misc_plotting.koe_plot(r, v, elements=["e", "i", "a"], body="Earth")
    assert ax.get_xlabel() == "Index"
    assert len(fig.axes) == 2
    plt.close(fig)

    save_path = tmp_path / "koe.png"
    fig, ax = misc_plotting.koe_plot(r, v, t=Time([0.0, 1.0, 2.0], format="gps"), elements=["a"], body="Moon", save_path=save_path)
    assert save_path.exists()
    plt.close(fig)
    with pytest.raises(TypeError, match="unexpected keyword"):
        misc_plotting.koe_plot(r, v, bad=True)


def test_koe_2dhist_validates_and_plots(tmp_path):
    data = SimpleNamespace(
        a=np.linspace(RGEO, 2 * RGEO, 10),
        e=np.linspace(0.1, 0.8, 10),
        i=np.linspace(0.0, np.pi / 2, 10),
        ta=np.linspace(0.0, 2 * np.pi, 10),
    )
    fig = misc_plotting.koe_2dhist(data, bins=4, logscale=False)
    assert len(fig.axes) >= 9
    plt.close(fig)
    fig = misc_plotting.koe_2dhist(data, bins=4, logscale="log", save_path=tmp_path / "hist.png")
    assert (tmp_path / "hist.png").exists()
    plt.close(fig)

    bad_i = SimpleNamespace(a=data.a, e=data.e, i=np.array([-1.0]), ta=np.array([0.0]))
    with pytest.raises(ValueError, match="Inclination"):
        misc_plotting.koe_2dhist(bad_i)
    bad_ta = SimpleNamespace(a=data.a, e=data.e, i=np.array([0.0]), ta=np.array([3 * np.pi]))
    with pytest.raises(ValueError, match="True Anomaly"):
        misc_plotting.koe_2dhist(bad_ta)


def test_scatter_helpers_and_orbit_divergence(monkeypatch, tmp_path, capsys):
    misc_plotting.scatter2d([1, 2], [3, 4], [0.1, 0.2], colorscale="linear", save_path=tmp_path / "scatter2d.png")
    assert (tmp_path / "scatter2d.png").exists()
    misc_plotting.scatter2d([1, 2], [3, 4], [1.0, 2.0], colorscale="log")
    with pytest.raises(TypeError, match="unexpected keyword"):
        misc_plotting.scatter2d([1], [2], [3], bad=True)

    fig, ax = misc_plotting.scatter3d(np.array([[1, 2, 3], [4, 5, 6]]), cs=[0.0, 1.0], save_path=tmp_path / "scatter3d.png")
    assert (tmp_path / "scatter3d.png").exists()
    ax = ax[0] if isinstance(ax, tuple) else ax
    assert hasattr(ax, "get_zlim")
    plt.close(fig)
    fig, ax = misc_plotting.scatter3d([1, 2], [3, 4], [5, 6], cs=None)
    plt.close(fig)
    with pytest.raises(TypeError, match="unexpected keyword"):
        misc_plotting.scatter3d([1], [2], [3], bad=True)

    assert len(misc_plotting.dotcolors_scaled(3)) == 3

    rs = np.zeros((3, 3, 2))
    rs[:, 0, 0] = [RGEO, 1.1 * RGEO, 1.2 * RGEO]
    rs[:, 1, 0] = [0, 0.1 * RGEO, 0.2 * RGEO]
    rs[:, 2, 1] = [RGEO, 1.1 * RGEO, 1.2 * RGEO]
    r_moon = np.tile([2 * RGEO, 0.0, 0.0], (3, 1))
    fig = misc_plotting.orbit_divergence_plot(rs, r_moon=r_moon, limits=None, show=False)
    assert "limits" in capsys.readouterr().out
    plt.close(fig)
    fig = misc_plotting.orbit_divergence_plot(rs, r_moon=r_moon.T, limits=3, show=False)
    plt.close(fig)
    assert misc_plotting.orbit_divergence_plot(rs, r_moon=r_moon, limits=3, show=False, save_path=tmp_path / "div.png") is None
    assert (tmp_path / "div.png").exists()

    with pytest.raises(IndexError, match="2 dimensions"):
        misc_plotting.orbit_divergence_plot(rs, r_moon=np.zeros(3), show=False)
    with pytest.raises(IndexError, match="expected"):
        misc_plotting.orbit_divergence_plot(rs, r_moon=np.zeros((2, 2)), show=False)

    class FakeMoon:
        def position(self, t):
            return r_moon

    monkeypatch.setattr(misc_plotting, "get_body", lambda name: FakeMoon())
    fig = misc_plotting.orbit_divergence_plot(rs, r_moon=None, t=np.arange(3), limits=3, show=False)
    plt.close(fig)
