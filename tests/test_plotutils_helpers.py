from pathlib import Path
import importlib
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.time import Time

from ssapy_toolkit.plots import plotutils
from ssapy_toolkit.plots.plotutils import VarType

figpath_module = importlib.import_module("ssapy_toolkit.plots.figpath")


def test_check_type_classifies_plot_inputs():
    assert plotutils.check_type(None) is VarType.NONE
    assert plotutils.check_type(Time(0.0, format="gps")) is VarType.TIME
    assert plotutils.check_type(np.zeros((2, 3))) is VarType.ARRAY
    assert plotutils.check_type([np.zeros((1, 3)), np.ones((1, 3))]) is VarType.LIST_ARRAYS
    assert plotutils.check_type([[1, 2], [3, 4]]) is VarType.LIST_LISTS
    assert plotutils.check_type([Time(0.0, format="gps"), Time(1.0, format="gps")]) is VarType.TIME
    assert plotutils.check_type([np.zeros(3), "mixed"]) is VarType.MIXED_LIST
    assert plotutils.check_type([]) is VarType.OTHER
    assert plotutils.check_type("x") is VarType.OTHER


def test_valid_orbits_normalizes_single_batched_and_per_track_times(capsys):
    single_r, single_t = plotutils.valid_orbits(np.array([1.0, 2.0, 3.0]), None, warn=False)
    assert len(single_r) == len(single_t) == 1
    assert single_r[0].shape == (1, 3)
    np.testing.assert_allclose(single_t[0].gps, [0.0])

    batched = np.arange(18.0).reshape(2, 3, 3)
    times = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    r_list, t_list = plotutils.valid_orbits(batched, times, warn=False)
    assert [track.shape for track in r_list] == [(3, 3), (3, 3)]
    np.testing.assert_allclose(t_list[0].gps, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(t_list[1].gps, [4.0, 5.0, 6.0])

    time_array = Time([10.0, 11.0, 12.0], format="gps")
    r_list, t_list = plotutils.valid_orbits([batched[0], batched[1]], [time_array, 20.0], warn=False)
    np.testing.assert_allclose(t_list[0].gps, time_array.gps)
    np.testing.assert_allclose(t_list[1].gps, [20.0, 20.0, 20.0])

    empty_r, empty_t = plotutils.valid_orbits([np.empty((0, 3))], [np.array([])], warn=True)
    assert empty_r == []
    assert empty_t == []
    assert "all orbit tracks were empty" in capsys.readouterr().out

    col_r, col_t = plotutils.valid_orbits(np.array([[1.0], [2.0], [3.0]]), np.array(30.0), warn=False)
    assert col_r[0].shape == (1, 3)
    np.testing.assert_allclose(col_t[0].gps, [30.0])

    scalar_time = Time(40.0, format="gps")
    scalar_r, scalar_t = plotutils.valid_orbits([np.ones((2, 3)), np.ones((2, 3)) * 2], scalar_time, warn=False)
    assert len(scalar_r) == 2
    np.testing.assert_allclose(scalar_t[0].gps, [40.0, 40.0])

    shared_time = Time([50.0, 51.0], format="gps")
    _, shared_t = plotutils.valid_orbits([np.ones((2, 3)), np.ones((2, 3)) * 2], shared_time, warn=False)
    np.testing.assert_allclose(shared_t[1].gps, [50.0, 51.0])

    broadcast_list_r, broadcast_list_t = plotutils.valid_orbits([np.ones((1, 3)), np.ones((1, 3)) * 2], [60.0], warn=False)
    assert len(broadcast_list_r) == 2
    np.testing.assert_allclose(broadcast_list_t[0].gps, [60.0])

    kept_empty, kept_time = plotutils.valid_orbits([np.empty((0, 3)), np.ones((1, 3))], [np.array([]), np.array([1.0])], drop_empty=False, warn=False)
    assert kept_empty[0].shape == (0, 3)
    assert len(kept_time[0]) == 0


def test_valid_orbits_rejects_shape_and_time_mismatches():
    with pytest.raises(ValueError, match="cannot interpret r"):
        plotutils.valid_orbits(np.ones((2, 2)), None)
    with pytest.raises(ValueError, match="t length"):
        plotutils.valid_orbits(np.ones((2, 3)), np.array([1.0]))
    with pytest.raises(TypeError, match="ndarray t must be numeric"):
        plotutils.valid_orbits(np.ones((1, 3)), np.array(["bad"]))
    with pytest.raises(ValueError, match="same number of tracks"):
        plotutils.valid_orbits(np.ones((2, 2, 3)), np.ones((1, 2)))
    with pytest.raises(TypeError, match="unsupported type"):
        plotutils.valid_orbits(np.ones((1, 3)), object())
    with pytest.raises(ValueError, match="Time length"):
        plotutils.valid_orbits(np.ones((2, 3)), Time([1.0], format="gps"))
    with pytest.raises(ValueError, match="single Time array"):
        plotutils.valid_orbits([np.ones((2, 3)), np.ones((3, 3))], Time([1.0, 2.0], format="gps"))
    with pytest.raises(ValueError, match="single t-array"):
        plotutils.valid_orbits([np.ones((2, 3)), np.ones((3, 3))], np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="number of t entries"):
        plotutils.valid_orbits([np.ones((1, 3)), np.ones((1, 3))], [1.0, 2.0, 3.0])


def test_save_alias_resolution_and_figure_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(figpath_module, "HOME_FIG_DIR", tmp_path / "figs")

    assert plotutils._figure_save_path(False) is None
    assert Path(plotutils._figure_save_path(None, default_name="default_name")) == tmp_path / "figs" / "default_name"

    absolute = tmp_path / "explicit" / "plot.png"
    assert Path(plotutils._figure_save_path(absolute)) == absolute
    assert absolute.parent.exists()

    save_path, remaining = plotutils._pop_save_path_aliases({"save": "same", "save_fig": "same", "other": 1})
    assert save_path == "same"
    assert remaining == {"other": 1}
    assert plotutils._pop_save_path_aliases({}, save_path=False) == (False, {})
    with pytest.raises(TypeError, match="Conflicting"):
        plotutils._pop_save_path_aliases({"save": "one", "save_path": "two"})
    with pytest.raises(TypeError, match="unexpected keyword"):
        plotutils._raise_unrecognized_kwargs({"bad": 1}, "helper")


def test_figsave_save_plot_display_and_theme_helpers(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(figpath_module, "HOME_FIG_DIR", tmp_path / "figs")

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    saved = Path(plotutils.figsave(fig, save=True, default_name="quicklook"))
    assert saved == tmp_path / "figs" / "quicklook.jpg"
    assert saved.exists()

    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    saved_pdf = Path(plotutils.figsave(fig, save_path=tmp_path / "plot.pdf"))
    assert saved_pdf.exists()

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    appended_pdf = Path(plotutils.figsave(fig, save_path=tmp_path / "plot.pdf"))
    assert appended_pdf.exists()

    assert plotutils.figsave(plt.figure(), save_path=False) is None

    class BadFigure:
        def savefig(self, *args, **kwargs):
            raise RuntimeError("cannot save")

    assert plotutils.figsave(BadFigure(), save_path=tmp_path / "bad.png") is None
    assert "Error occurred" in capsys.readouterr().out

    plotutils.display_figure(str(tmp_path / "missing_image"), display="PIL")
    assert "No image file found" in capsys.readouterr().out

    image_path = tmp_path / "shown.png"
    plt.imsave(image_path, np.zeros((2, 2, 3)))
    shown = []

    class FakePILImage:
        def show(self):
            shown.append(True)

    monkeypatch.setattr(plotutils.PILImage, "open", lambda filename: FakePILImage())
    plotutils.display_figure(str(image_path.with_suffix("")), display="PIL")
    assert shown == [True]

    displayed = []
    fake_ipython_display = SimpleNamespace(
        Image=lambda filename: ("image", filename),
        display=lambda image: displayed.append(image),
    )
    monkeypatch.setitem(sys.modules, "IPython.display", fake_ipython_display)
    plotutils.display_figure(str(image_path), display="IPython")
    assert displayed == [("image", str(image_path))]

    with pytest.raises(ValueError, match="Invalid display"):
        plotutils.display_figure(str(image_path), display="bad")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plotutils.make_black(fig, ax)
    assert fig.get_facecolor()[:3] == (0.0, 0.0, 0.0)
    plotutils.make_white(fig, ax)
    assert fig.get_facecolor()[:3] == (1.0, 1.0, 1.0)
    plt.close(fig)


def test_auto_log_scale_uses_meaningful_lower_bound():
    fig, ax = plt.subplots()
    values = [
        np.array([0.0, 0.2, 10.0, 1000.0]),
        np.array([0.0, 0.6, 20.0, 2000.0]),
    ]

    assert plotutils.should_use_log_scale(values)
    np.testing.assert_allclose(
        plotutils.log_safe_values([0.0, -1.0, 2.0]),
        [np.nan, np.nan, 2.0],
        equal_nan=True,
    )
    assert plotutils.apply_auto_log_scale(ax, values)
    assert ax.get_yscale() == "log"
    assert ax.get_ylim()[0] == pytest.approx(0.2)
    plt.close(fig)


def test_auto_log_scale_skips_small_dynamic_range():
    fig, ax = plt.subplots()
    assert not plotutils.apply_auto_log_scale(ax, [np.array([1.0, 2.0, 3.0])])
    assert ax.get_yscale() == "linear"
    plt.close(fig)


def test_draw_earth_and_moon_with_fake_ipyvolume(monkeypatch):
    calls = []

    def fake_plot_mesh(*args, **kwargs):
        calls.append((args, kwargs))
        return "mesh"

    monkeypatch.setitem(sys.modules, "ipyvolume", SimpleNamespace(plot_mesh=fake_plot_mesh))
    monkeypatch.setattr(plotutils, "load_earth_file", lambda: "earth-texture")
    monkeypatch.setattr(plotutils, "load_moon_file", lambda: "moon-texture")

    assert plotutils.drawEarth(0.0, ngrid=4, R=1.0, rfactor=2.0) == "mesh"
    assert plotutils.drawMoon(Time(0.0, format="gps"), ngrid=4, R=1.0, rfactor=3.0) == "mesh"
    assert calls[0][1]["texture"] == "earth-texture"
    assert calls[1][1]["texture"] == "moon-texture"
    assert calls[0][0][0].shape == (4, 4)


def test_plot_geometry_and_color_helpers():
    sphere = plotutils.create_sphere(1.0, 2.0, 3.0, 4.0, resolution=5)
    assert sphere.shape == (3, 10, 5)
    assert np.isclose(np.nanmax(np.linalg.norm((sphere - np.array([1.0, 2.0, 3.0])[:, None, None]).reshape(3, -1), axis=0)), 4.0)

    x, y, z = plotutils.drawSphere(0.0, 0.0, 0.0, 1.0, res=4, flatten=True)
    assert x.ndim == y.ndim == z.ndim == 1
    x_grid, y_grid, z_grid = plotutils.drawSphere(0.0, 0.0, 0.0, 1.0, res=4, flatten=False)
    assert x_grid.shape == y_grid.shape == z_grid.shape

    assert plotutils.rgb(0.0, 10.0, 0.0) == (0, 0, 255)
    assert plotutils.rgb(0.0, 10.0, 10.0) == (255, 0, 0)
    assert len(plotutils.generate_rainbow_colors(4)) == 4
    darkened = plotutils.darken("red", amount=[0.0, 0.5, 2.0])
    assert len(darkened) == 3
    assert all(len(rgb) == 3 for rgb in darkened)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plotutils.draw_dashed_circle(ax, np.array([0.0, 0.0, 1.0]), radius=1.0, dashes=4, dash_length=0.05)
    assert len(ax.lines) == 4
    plt.close(fig)
