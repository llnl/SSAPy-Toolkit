from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


def _sample_track():
    return np.array(
        [
            [7_000_000.0, 0.0, 0.0],
            [0.0, 7_000_000.0, 1_000_000.0],
            [-7_000_000.0, 0.0, -1_000_000.0],
        ]
    )


class _FakeBody:
    def position(self, t):
        count = 1 if getattr(t, "isscalar", False) else len(t)
        return np.zeros((3, count))


def test_orbit_plot_delegates_to_core(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.plots.orbit_plot")
    calls = []

    def fake_core(r, **kwargs):
        calls.append((r, kwargs))
        return "fig", list(kwargs["views"])

    monkeypatch.setattr(module, "_orbit_plot_core", fake_core)
    result = module.orbit_plot(
        _sample_track(),
        t=np.arange(3.0),
        title="demo",
        view="xyxz",
        savefig="plots/alias.jpg",
        frame="gcrf",
        show=False,
        c="black",
        pad=2,
    )

    assert result == (
        "fig",
        ["xy", "xz"],
    )
    assert calls[0][1]["views"] == ("xy", "xz")
    assert calls[0][1]["save_path"] == "plots/alias.jpg"
    assert calls[0][1]["lunar_transform"] == "standard"


def test_orbit_plot_entry_point_accepts_save_aliases(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.plots.orbit_plot")
    calls = []

    def fake_core(r, **kwargs):
        calls.append(kwargs)
        return "fig", kwargs["views"]

    monkeypatch.setattr(module, "_orbit_plot_core", fake_core)

    assert module.orbit_plot(_sample_track(), t=np.arange(3.0), view="xy", save_path="same.jpg", save="same.jpg") == (
        "fig",
        ("xy",),
    )
    assert calls[-1]["save_path"] == "same.jpg"

    with pytest.raises(TypeError, match="Conflicting figure save aliases"):
        module.orbit_plot(_sample_track(), t=np.arange(3.0), view="xy", save="one.jpg", save_fig="two.jpg")


def test_orbit_plot_routes_gif_and_mp4_saves_to_animation(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.plots.orbit_plot")
    calls = []

    def fake_animation(r, **kwargs):
        calls.append(kwargs)
        return kwargs["save_path"]

    def fail_core(r, **kwargs):
        raise AssertionError("animated save paths should not call static orbit core")

    monkeypatch.setattr(module, "_orbit_animation_core", fake_animation)
    monkeypatch.setattr(module, "_orbit_plot_core", fail_core)

    assert module.orbit_plot(_sample_track(), t=np.arange(3.0), view="xy", save_path="orbit.mp4", fps=12, tail=5) == "orbit.mp4"
    assert calls[-1]["views"] == ("xy",)
    assert calls[-1]["fps"] == 12
    assert calls[-1]["tail"] == 5

    assert module.orbit_plot(_sample_track(), t=np.arange(3.0), view="lunar_xz", savefig="orbit.gif") == "orbit.gif"
    assert calls[-1]["views"] == ("xz",)
    assert calls[-1]["frame"] == "lunar_fixed"
    assert calls[-1]["lunar_transform"] == "fixed"


def test_orbit_plot_routes_keyword_views_to_orbit_core(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.plots.orbit_plot")
    calls = []

    def fake_core(r, **kwargs):
        calls.append(kwargs)
        return "fig", list(kwargs["views"])

    monkeypatch.setattr(module, "_orbit_plot_core", fake_core)

    assert module.orbit_plot(_sample_track(), t=np.arange(3.0), view="xz") == ("fig", ["xz"])
    assert calls[-1]["views"] == ("xz",)

    assert module.orbit_plot(_sample_track(), t=np.arange(3.0), view="xyxz") == ("fig", ["xy", "xz"])
    assert calls[-1]["views"] == ("xy", "xz")

    assert module.orbit_plot(_sample_track(), t=np.arange(3.0), view=("xy", "yz", "3d")) == ("fig", ["xy", "yz", "3d"])
    assert calls[-1]["views"] == ("xy", "yz", "3d")

    assert module.orbit_plot(_sample_track(), t=np.arange(3.0), view="xy+3d", frame="lunar_fixed") == ("fig", ["xy", "3d"])
    assert calls[-1]["views"] == ("xy", "3d")
    assert calls[-1]["lunar_transform"] == "fixed"


def test_orbit_plot_routes_map_view_aliases_to_orbit_core(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.plots.orbit_plot")
    calls = []

    def fake_core(r, **kwargs):
        calls.append(kwargs)
        return "fig", list(kwargs["views"])

    monkeypatch.setattr(module, "_orbit_plot_core", fake_core)

    assert module.orbit_plot(_sample_track(), t=np.arange(3.0), view="Ground Track") == ("fig", ["groundtrack"])
    assert calls[-1]["views"] == ("groundtrack",)

    assert module.orbit_plot(_sample_track(), t=np.arange(3.0), view="ground_track") == ("fig", ["groundtrack"])
    assert calls[-1]["views"] == ("groundtrack",)

    assert module.orbit_plot(_sample_track(), t=np.arange(3.0), view="GLOBE Plot", el=25.0) == ("fig", ["globe"])
    assert calls[-1]["views"] == ("globe",)
    assert calls[-1]["special_plot_kwargs"]["el"] == 25.0

    result = module.orbit_plot(_sample_track(), t=np.arange(3.0), view=("xy", "ground track", "globeplot"))
    assert result == ("fig", ["xy", "groundtrack", "globe"])
    assert calls[-1]["views"] == ("xy", "groundtrack", "globe")


def test_orbit_plot_routes_dashboard_aliases_to_orbit_core(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.plots.orbit_plot")
    calls = []

    def fake_core(r, **kwargs):
        calls.append(kwargs)
        return "fig", list(kwargs["views"])

    monkeypatch.setattr(module, "_orbit_plot_core", fake_core)

    expected_views = ["groundtrack", "globe", "xy", "xz", "yz", "3d"]
    assert module.orbit_plot(_sample_track(), t=np.arange(3.0), view="Dashboard") == ("fig", expected_views)
    assert calls[-1]["views"] == tuple(expected_views)
    assert calls[-1]["figsize"] == (16, 12)

    assert module.orbit_plot(_sample_track(), t=np.arange(3.0), view="orbit dashboard") == ("fig", expected_views)

    module.orbit_plot(_sample_track(), t=np.arange(3.0), view="dashboard", figsize=(8, 8))
    assert calls[-1]["figsize"] == (8, 8)


def test_orbit_plot_lunar_views_default_to_lunar_fixed(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.plots.orbit_plot")
    calls = []

    def fake_core(r, **kwargs):
        calls.append(kwargs)
        return "fig", list(kwargs["views"])

    monkeypatch.setattr(module, "_orbit_plot_core", fake_core)

    assert module.orbit_plot(_sample_track(), t=np.arange(3.0), view="lunar_yz") == ("fig", ["yz"])
    assert calls[-1]["frame"] == "lunar_fixed"
    assert calls[-1]["lunar_transform"] == "fixed"

    assert module.orbit_plot(_sample_track(), t=np.arange(3.0), view=("lunar_xy", "lunar_xz", "lunar_3d")) == ("fig", ["xy", "xz", "3d"])
    assert calls[-1]["frame"] == "lunar_fixed"
    assert calls[-1]["views"] == ("xy", "xz", "3d")


def test_orbit_plot_coordinate_aliases_override_lunar_view_defaults(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.plots.orbit_plot")
    calls = []

    def fake_core(r, **kwargs):
        calls.append(kwargs)
        return "fig", list(kwargs["views"])

    monkeypatch.setattr(module, "_orbit_plot_core", fake_core)

    module.orbit_plot(_sample_track(), t=np.arange(3.0), view="lunar_xy", coordinate="gcrf")
    assert calls[-1]["frame"] == "gcrf"
    assert calls[-1]["lunar_transform"] == "standard"

    module.orbit_plot(_sample_track(), t=np.arange(3.0), view="xy", coordinates="itrf")
    assert calls[-1]["frame"] == "itrf"

    module.orbit_plot(_sample_track(), t=np.arange(3.0), view="xy", frame="itrf")
    assert calls[-1]["frame"] == "itrf"


def test_orbit_plot_routes_cislunar_keywords_to_cislunar_core(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.plots.orbit_plot")
    calls = []

    def fail_orbit_core(r, **kwargs):
        raise AssertionError("cislunar selectors should not call the standard orbit core")

    def fake_cislunar_core(r, **kwargs):
        calls.append(kwargs)
        return "fig", kwargs["mode"]

    monkeypatch.setattr(module, "_orbit_plot_core", fail_orbit_core)
    monkeypatch.setattr(module, "_cislunar_plot_core", fake_cislunar_core)

    result = module.orbit_plot(
        _sample_track(),
        t=np.arange(3.0),
        view="cislunar_3d",
        title="demo",
        save_path="plots/cislunar.jpg",
        fontsize=16,
        legend=False,
    )

    assert result == ("fig", "3d")
    assert calls[-1]["mode"] == "3d"
    assert calls[-1]["title"] == "demo"
    assert calls[-1]["save_path"] == "plots/cislunar.jpg"
    assert calls[-1]["fontsize"] == 16
    assert calls[-1]["legend"] is False

    assert module.orbit_plot(_sample_track(), t=np.arange(3.0), view="cislunar_xy") == ("fig", "xy")
    assert calls[-1]["mode"] == "xy"

    assert module.orbit_plot(_sample_track(), t=np.arange(3.0), view="cislunar dashboard") == ("fig", "combined")
    assert calls[-1]["mode"] == "combined"
    assert calls[-1]["figsize"] == (12, 7)


def test_orbit_plot_routes_transfer_keywords(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.plots.orbit_plot")
    transfer_trajectory = importlib.import_module("ssapy_toolkit.plots.transfer_trajectory_plot")
    transfer_burn = importlib.import_module("ssapy_toolkit.plots.transfer_burn_profile_plot")
    transfer_designer = importlib.import_module("ssapy_toolkit.plots.transfer_designer_curves_plot")
    calls = []

    class FakeTransfer:
        burns = []
        trajectory = {"t": np.arange(3.0), "r": _sample_track(), "v": _sample_track()}
        transfer_orbit = object()

    class FakeOptimal:
        transfer = FakeTransfer()

    def fake_trajectory(result, **kwargs):
        calls.append(("trajectory", result, kwargs))
        return "trajectory_axes"

    def fake_burn(result, **kwargs):
        calls.append(("burn", result, kwargs))
        return "burn_fig"

    def fake_designer(result, **kwargs):
        calls.append(("designer", result, kwargs))
        return "designer_fig"

    monkeypatch.setattr(transfer_trajectory, "transfer_trajectory_plot", fake_trajectory)
    monkeypatch.setattr(transfer_burn, "transfer_burn_profile_plot", fake_burn)
    monkeypatch.setattr(transfer_designer, "transfer_designer_curves_plot", fake_designer)

    result = FakeOptimal()
    assert module.orbit_plot(result, view="transfer", save_path="transfer.png") == "trajectory_axes"
    assert calls[-1][0] == "trajectory"
    assert calls[-1][2]["save_path"] == "transfer.png"
    assert calls[-1][2]["three_d"] is False

    assert module.orbit_plot(result, view="transfer_trajectory_3d") == "trajectory_axes"
    assert calls[-1][2]["three_d"] is True

    assert module.orbit_plot(result, view="transfer_burn_profile", title="Burns") == "burn_fig"
    assert calls[-1] == ("burn", result, {"title": "Burns", "save_path": False})

    assert module.orbit_plot(result, view="transfer_designer", title="Designer") == "designer_fig"
    assert calls[-1] == ("designer", result, {"title": "Designer", "save_path": False})

    states = tuple(np.array([7_000_000.0, 0.0, 0.0]) for _ in range(6))
    with pytest.raises(TypeError, match="transfer result"):
        module.orbit_plot(states, view="transfer_plot")


def test_orbit_plot_routes_divergence_keywords(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.plots.orbit_plot")
    divergence = importlib.import_module("ssapy_toolkit.plots.divergence_plot")
    misc = importlib.import_module("ssapy_toolkit.plots.misc_plotting")
    calls = []

    def fake_divergence(r, **kwargs):
        calls.append(("divergence", r, kwargs))
        return "divergence_fig"

    def fake_orbit_divergence(r, **kwargs):
        calls.append(("orbit_divergence", r, kwargs))
        return "orbit_divergence_fig"

    monkeypatch.setattr(divergence, "divergence_plot", fake_divergence)
    monkeypatch.setattr(misc, "orbit_divergence_plot", fake_orbit_divergence)

    assert module.orbit_plot(_sample_track(), view="divergence", v_center=np.array([0.0, 1.0, 0.0])) == "divergence_fig"
    assert calls[-1][0] == "divergence"
    assert calls[-1][2]["show"] is False
    np.testing.assert_allclose(calls[-1][2]["v_center"], [0.0, 1.0, 0.0])

    assert module.orbit_plot(_sample_track(), t=np.arange(3.0), view="orbit_divergence") == "orbit_divergence_fig"
    assert calls[-1][0] == "orbit_divergence"
    assert calls[-1][2]["show"] is False


def test_orbit_plot_routes_cislunar_dashboard_to_core(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.plots.orbit_plot")
    calls = []

    def fake_core(r, **kwargs):
        calls.append((r, kwargs))
        return "fig", kwargs["mode"]

    monkeypatch.setattr(module, "_cislunar_plot_core", fake_core)

    result = module.orbit_plot(
        _sample_track(),
        t=np.arange(3.0),
        view="cislunar_dashboard",
        title="demo",
        save="plots/cislunar_dashboard.jpg",
        legend=False,
    )

    assert result == ("fig", "combined")
    assert calls[0][1]["mode"] == "combined"
    assert calls[0][1]["save_path"] == "plots/cislunar_dashboard.jpg"
    assert calls[0][1]["legend"] is False


def test_orbit_plot_rejects_unknown_view():
    module = importlib.import_module("ssapy_toolkit.plots.orbit_plot")

    with pytest.raises(ValueError, match="Unknown orbit_plot view"):
        module.orbit_plot(_sample_track(), view="not_a_plot")

    with pytest.raises(ValueError, match="multiple view entries"):
        module.orbit_plot(_sample_track(), view=("xy", "not_a_view"))

    with pytest.raises(ValueError, match="Use only one of frame"):
        module.orbit_plot(_sample_track(), view="xy", frame="gcrf", coordinate="itrf")


def test_orbit_plot_core_forwards_save_path(monkeypatch, tmp_path):
    core = importlib.import_module("ssapy_toolkit.plots._orbit_plot_core")
    saved = {}

    monkeypatch.setattr(core, "_get_body", lambda name: _FakeBody())
    monkeypatch.setattr(core, "_lagrange_points_lunar_frame", lambda: {})
    monkeypatch.setattr(core, "_figsave", lambda fig, path: saved.update(fig=fig, path=Path(path)))

    output_path = tmp_path / "orbit.png"
    fig, axes = core._orbit_plot_core(
        _sample_track(),
        t=np.arange(3.0),
        views=("xy",),
        save_path=output_path,
        show=False,
    )

    assert len(axes) == 1
    assert saved["fig"] is fig
    assert saved["path"] == output_path


def test_orbit_plot_core_uses_auto_grid_for_multiple_views(monkeypatch):
    core = importlib.import_module("ssapy_toolkit.plots._orbit_plot_core")

    monkeypatch.setattr(core, "_get_body", lambda name: _FakeBody())
    monkeypatch.setattr(core, "_lagrange_points_lunar_frame", lambda: {})

    fig, axes = core._orbit_plot_core(
        _sample_track(),
        t=np.arange(3.0),
        views=("xy", "xz"),
        show=False,
    )

    grid = axes[0].get_subplotspec().get_gridspec()
    assert len(axes) == 2
    assert grid.nrows == 2
    assert grid.ncols == 2

    assert core._auto_grid_shape(1) == (1, 1)
    assert core._auto_grid_shape(4) == (2, 2)
    assert core._auto_grid_shape(5) == (2, 3)
    assert core._auto_grid_shape(7) == (3, 3)
    fig.clear()


def test_orbit_plot_core_embeds_groundtrack_and_globe_views(monkeypatch):
    core = importlib.import_module("ssapy_toolkit.plots._orbit_plot_core")
    calls = []

    def fake_groundtrack(r, t, *, ax=None, save_path=None, title=None, central_longitude=None, **kwargs):
        kwargs = {"central_longitude": central_longitude, **kwargs}
        calls.append(("groundtrack", ax, save_path, title, kwargs))
        ax.set_title(title)
        return ax.figure

    def fake_globe(r, t=None, *, ax=None, save_path=None, title=None, el=None, **kwargs):
        kwargs = {"el": el, **kwargs}
        calls.append(("globe", ax, save_path, title, kwargs))
        ax.set_title(title)
        return ax.figure, ax

    monkeypatch.setattr(core, "_groundtrack_plot", fake_groundtrack)
    monkeypatch.setattr(core, "_globe_plot", fake_globe)

    fig, axes = core._orbit_plot_core(
        _sample_track(),
        t=np.arange(3.0),
        views=("groundtrack",),
        title="Map View",
        special_plot_kwargs={"central_longitude": 180.0, "el": 20.0},
        show=False,
    )

    grid = axes[0].get_subplotspec().get_gridspec()
    assert len(axes) == 1
    assert grid.nrows == 1
    assert grid.ncols == 2
    assert axes[0].get_subplotspec().colspan.start == 0
    assert axes[0].get_subplotspec().colspan.stop == 2
    assert calls[0][0] == "groundtrack"
    assert calls[0][2] is None
    assert calls[0][3] == "Map View"
    assert calls[0][4]["central_longitude"] == 180.0
    fig.clear()

    calls.clear()
    fig, axes = core._orbit_plot_core(
        _sample_track(),
        t=np.arange(3.0),
        views=("groundtrack", "globe"),
        title="Map + Globe",
        special_plot_kwargs={"central_longitude": 180.0, "el": 20.0},
        show=False,
    )

    assert [call[0] for call in calls] == ["groundtrack", "globe"]
    assert axes[0].name == "rectilinear"
    assert axes[1].name == "3d"
    assert calls[1][4]["el"] == 20.0
    fig.clear()


def test_orbit_plot_core_packs_wide_views_without_avoidable_gaps():
    import matplotlib.pyplot as plt

    core = importlib.import_module("ssapy_toolkit.plots._orbit_plot_core")

    assert core._pack_views(("xy", "xz", "groundtrack", "globe"), ncols=3) == [
        ("xy", 0, 0, 1),
        ("xz", 0, 1, 1),
        ("globe", 0, 2, 1),
        ("groundtrack", 1, 0, 2),
    ]

    fig = plt.figure()
    axes = core._create_orbit_axes(fig, ("xy", "xz", "groundtrack", "globe"))

    assert axes["xy"].get_subplotspec().rowspan.start == 0
    assert axes["xy"].get_subplotspec().colspan.start == 0
    assert axes["xz"].get_subplotspec().rowspan.start == 0
    assert axes["xz"].get_subplotspec().colspan.start == 1
    assert axes["globe"].get_subplotspec().rowspan.start == 0
    assert axes["globe"].get_subplotspec().colspan.start == 2
    assert axes["groundtrack"].get_subplotspec().rowspan.start == 1
    assert axes["groundtrack"].get_subplotspec().colspan.start == 0
    assert axes["groundtrack"].get_subplotspec().colspan.stop == 2
    fig.clear()


def test_globe_plot_returns_axis_after_theming(monkeypatch):
    from PIL import Image

    module = importlib.import_module("ssapy_toolkit.plots.globe_plot")

    monkeypatch.setattr(module, "find_file", lambda *args, **kwargs: "earth.png")
    monkeypatch.setattr(
        module.PILImage,
        "open",
        lambda path: Image.fromarray(np.zeros((8, 16, 3), dtype=np.uint8)),
    )

    fig, ax = module.globe_plot(
        _sample_track(),
        t=np.arange(3.0),
        c="black",
        scale=1000,
        show_legend=False,
    )

    assert ax.name == "3d"
    fig.clear()


def test_globe_plot_validation_labels_limits_and_save(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt
    from astropy.time import Time
    from PIL import Image

    module = importlib.import_module("ssapy_toolkit.plots.globe_plot")
    monkeypatch.setattr(module, "find_file", lambda *args, **kwargs: "earth.png")
    monkeypatch.setattr(module.PILImage, "open", lambda path: Image.fromarray(np.ones((6, 12, 3), dtype=np.uint8) * 255))
    monkeypatch.setattr(module, "make_white", lambda fig, ax: (fig, [ax]))
    saved = []
    monkeypatch.setattr(module, "figsave", lambda fig, save_path: saved.append(Path(save_path)))

    assert np.isfinite(module._earth_lon0_from_time(0.0))
    assert np.isfinite(module._earth_lon0_from_time(Time(0.0, format="gps", scale="utc")))
    with pytest.raises(ValueError, match="shape"):
        module._earth_occultation_mask(np.ones((3, 2)), 1.0, 0.0, 0.0)

    track = _sample_track()
    with pytest.raises(ValueError, match="labels"):
        module.globe_plot([track, track], labels=["one"], scale=2000)
    with pytest.raises(ValueError, match="orbit_colors"):
        module.globe_plot([track, track], orbit_colors=["red"], scale=2000)
    with pytest.raises(ValueError, match="limits"):
        module.globe_plot(track, limits=[[1, 2]], scale=2000)

    fig2d, ax2d = plt.subplots()
    with pytest.raises(ValueError, match="3D"):
        module.globe_plot(track, ax=ax2d, scale=2000)
    plt.close(fig2d)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    returned_fig, returned_ax = module.globe_plot(
        [track, track * 1.05],
        t=[np.arange(3.0), np.arange(3.0)],
        ax=ax,
        c="white",
        labels=["a", "b"],
        orbit_colors=["red", "blue"],
        legend_kwargs={"loc": "upper left"},
        limits=[[-2, 2], [-3, 3], [-4, 4]],
        title="Globe",
        globe_time=0.0,
        scale=2000,
        savefig=tmp_path / "globe.png",
    )
    assert returned_fig is fig
    assert returned_ax is ax
    assert saved == [tmp_path / "globe.png"]
    assert ax.get_xlim() == pytest.approx((-2, 2))
    plt.close(fig)

    fig, ax = module.globe_plot(track, c="other", orbit_colors=["green"], limits=1_000_000.0, scale=2000, show_legend=False)
    assert ax.get_xlim()[1] < 1.0
    plt.close(fig)


def test_cislunar_plot_core_forwards_save_path(monkeypatch, tmp_path):
    core = importlib.import_module("ssapy_toolkit.plots._cislunar_plot_core")
    saved = {}

    monkeypatch.setattr(core, "_get_body", lambda name: _FakeBody())
    monkeypatch.setattr(core, "_gcrf_to_lunar_fixed", lambda xyz, t: xyz)
    monkeypatch.setattr(core, "_lagrange_points_lunar_fixed_frame", lambda: {})
    monkeypatch.setattr(core, "_sphere_mesh", lambda radius: tuple(np.zeros((2, 2)) for _ in range(3)))
    monkeypatch.setattr(core, "_figsave", lambda fig, path: saved.update(fig=fig, path=Path(path)))

    output_path = tmp_path / "cislunar.png"
    fig, ax = core._cislunar_plot_core(
        _sample_track(),
        t=np.arange(3.0),
        mode="3d",
        save_path=output_path,
        show=False,
    )

    assert ax.name == "3d"
    assert saved["fig"] is fig
    assert saved["path"] == output_path


def test_cislunar_plot_core_combined_and_xy_modes(monkeypatch):
    core = importlib.import_module("ssapy_toolkit.plots._cislunar_plot_core")
    monkeypatch.setattr(core, "_get_body", lambda name: _FakeBody())
    monkeypatch.setattr(core, "_gcrf_to_lunar_fixed", lambda xyz, t: np.asarray(xyz) + 1000.0)
    monkeypatch.setattr(
        core,
        "_lagrange_points_lunar_fixed_frame",
        lambda: {"L1": np.array([1000.0, 1000.0, 1000.0]), "L2": np.array([1e9, 1e9, 1e9])},
    )

    tracks = [_sample_track(), _sample_track() * 1.05]
    times = [np.arange(3.0), np.arange(3.0)]
    fig, axes = core._cislunar_plot_core(tracks, t=times, mode="combined", c="black", legend=True, title="Demo")
    assert len(axes) == 2
    assert axes[0].name == "3d"
    assert axes[1].name == "3d"
    fig.clear()

    fig, axes = core._cislunar_plot_core(_sample_track(), t=np.arange(3.0), mode="xy", c="white", legend=False)
    assert axes[0].name == "rectilinear"
    assert axes[1].name == "rectilinear"
    fig.clear()

    with pytest.raises(ValueError, match="mode"):
        core._cislunar_plot_core(_sample_track(), t=np.arange(3.0), mode="bad")
