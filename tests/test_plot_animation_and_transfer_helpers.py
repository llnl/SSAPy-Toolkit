import importlib
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ssapy_toolkit.plots import divergence_plot as divergence_plot_func
from ssapy_toolkit.plots import rendezvous_plot as rendezvous_plot_func
from ssapy_toolkit.plots.transfer_burn_profile_plot import transfer_burn_profile_plot
from ssapy_toolkit.plots.transfer_designer_curves_plot import transfer_designer_curves_plot
from ssapy_toolkit.plots.transfer_trajectory_plot import _burn_label, transfer_trajectory_plot


gifify_module = importlib.import_module("ssapy_toolkit.plots.gifify")
orbit_animation = importlib.import_module("ssapy_toolkit.plots.orbit_animation")


class FakeWriter:
    def __init__(self):
        self.frames = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def append_data(self, data):
        self.frames.append(np.asarray(data))


def _line_plot(x, y, *, ax=None, limit=None, show=False):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(x, y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if limit is not None:
        ax.set_xlim(-limit, limit)
    return fig


def _line_plot_ax_first(ax, x, y, *, limit=None, show=False):
    return _line_plot(x, y, ax=ax, limit=limit, show=show)


def test_gifify_modes_validation_and_fake_writer(tmp_path, monkeypatch):
    writers = []

    def fake_get_writer(*_args, **_kwargs):
        writer = FakeWriter()
        writers.append(writer)
        return writer

    monkeypatch.setattr(gifify_module.imageio, "get_writer", fake_get_writer)
    monkeypatch.setattr(gifify_module.imageio, "imread", lambda _path: np.zeros((2, 2, 3), dtype=np.uint8))

    x = np.arange(5.0)
    y = x**2
    result = gifify_module.gifify(_line_plot, x, y, save_path=tmp_path / "a.gif", mode="cumulative", step=2, fps=5, verbose=True)
    assert result["frames"] == 3
    assert result["range"] == (0, 5)
    assert len(writers[-1].frames) == 3

    result = gifify_module.gifify(
        _line_plot_ax_first,
        save_path=tmp_path / "b.gif",
        array_kw_keys=("x", "y"),
        x=x,
        y=y,
        mode="chunks",
        chunk_size=2,
        step=2,
        inject_ax=True,
        ax_kw_key="ax",
        fixed_limits=False,
    )
    assert result["frames"] == 3

    result = gifify_module.gifify(
        _line_plot_ax_first,
        x,
        y,
        save_path=tmp_path / "c.gif",
        mode="sliding",
        chunk_size=3,
        step=2,
        inject_ax=True,
        ax_arg_index=0,
        fixed_limits=False,
    )
    assert result["mode"] == "sliding"

    with pytest.raises(ValueError, match="array_kw_keys"):
        gifify_module.gifify(_line_plot, x, y, array_kw_keys=("x",))
    with pytest.raises(ValueError, match="array_arg_indices"):
        gifify_module.gifify(_line_plot, x, y, array_arg_indices=(0, 1, 2))
    with pytest.raises(ValueError, match="mode"):
        gifify_module.gifify(_line_plot, x, y, mode="bad")
    with pytest.raises(ValueError, match="Invalid start/end"):
        gifify_module.gifify(_line_plot, x, y, start=4, end=4)
    with pytest.raises(ValueError, match="chunk_size"):
        gifify_module.gifify(_line_plot, x, y, mode="chunks")
    with pytest.raises(ValueError, match="step"):
        gifify_module.gifify(_line_plot, x, y, step=0)


def test_orbit_animation_private_helpers(monkeypatch, tmp_path):
    assert orbit_animation._animation_save_path(tmp_path / "movie.gif").endswith("movie.gif")
    with pytest.raises(ValueError, match="requires a save path"):
        orbit_animation._animation_save_path(False)
    assert orbit_animation._animation_writer(".gif", fps=2, bitrate=100).__class__.__name__ == "PillowWriter"
    monkeypatch.setattr(orbit_animation, "_ensure_ffmpeg_path", lambda: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        orbit_animation._animation_writer(".mp4", fps=2, bitrate=100)
    monkeypatch.setattr(orbit_animation, "_ensure_ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    assert orbit_animation._animation_writer(".mp4", fps=2, bitrate=100).__class__.__name__ == "FFMpegWriter"

    r_tracks = [np.arange(30.0).reshape(10, 3), np.arange(30.0, 60.0).reshape(10, 3)]
    t_tracks = [np.arange(10.0), np.arange(10.0)]
    r_down, t_down = orbit_animation._downsample_tracks(r_tracks, t_tracks, max_frames=4)
    assert r_down[0].shape == (4, 3)
    assert t_down[0].shape == (4,)
    same_r, same_t = orbit_animation._downsample_tracks(r_tracks, t_tracks, max_frames=20)
    assert same_r is r_tracks
    assert same_t is t_tracks

    monkeypatch.setattr(orbit_animation, "_ground_track", lambda r, t, format="geodetic": (np.deg2rad([0, 200, -170]), np.deg2rad([0, 10, -10]), np.zeros(3)))
    lonlat = orbit_animation._groundtrack_degrees(np.zeros((3, 3)), np.arange(3))
    np.testing.assert_allclose(lonlat[:, 0], [0, -160, -170])

    class FakeMoon:
        def position(self, t):
            return np.tile([[1.0], [2.0], [3.0]], (1, len(t)))

    monkeypatch.setattr(orbit_animation, "_get_body", lambda name: FakeMoon())
    prepared = orbit_animation._prepare_animation_tracks([np.eye(3) * 1000.0], [np.arange(3.0)], frame="gcrf", lunar_transform="standard", pad=0.1)
    assert prepared["tracks"][0]["xyz"].shape == (3, 3)
    assert prepared["unit_label"] == "km"
    with pytest.raises(ValueError, match="Unknown plot type"):
        orbit_animation._prepare_animation_tracks([np.eye(3)], [np.arange(3.0)], frame="bad", lunar_transform="standard", pad=0)

    fig = plt.figure()
    axes = orbit_animation._create_orbit_axes(fig, ("xy", "groundtrack", "3d"), layout="auto")
    orbit_animation._draw_animation_backgrounds(axes, ("xy", "groundtrack", "3d"), prepared, "Title", "black")
    artists = orbit_animation._init_animation_artists(axes, ("xy", "groundtrack", "3d"), prepared["tracks"], [np.array([1.0, 0.0, 0.0, 1.0])])
    frame_artists = []
    orbit_animation._update_secondary_markers(axes, ("xy", "groundtrack", "3d"), artists, prepared, 2, frame_artists)
    assert frame_artists
    tail2d = orbit_animation._add_2d_tail(axes["xy"], np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([1, 0, 0, 1]))
    tail3d = orbit_animation._add_3d_tail(axes["3d"], np.array([[0.0, 0.0, 0.0]]), np.array([1, 0, 0, 1]))
    ground_tail = orbit_animation._add_groundtrack_tail(axes["groundtrack"], np.array([[170.0, 0.0], [-170.0, 1.0], [-160.0, 2.0]]), np.array([1, 0, 0, 1]))
    assert orbit_animation._tail_colors([1.0, 0.0, 0.0], 2).shape == (2, 4)
    orbit_animation._remove_collection(tail2d)
    orbit_animation._remove_collection(tail3d)
    orbit_animation._remove_collection(ground_tail)
    orbit_animation._set_3d_limits(axes["3d"], prepared["bounds"])
    plt.close(fig)


def test_orbit_animation_public_gif_path_runs_update_loop(monkeypatch, tmp_path):
    saved = []

    class FakeAnimation:
        def __init__(self, fig, update, frames, interval, blit, repeat):
            self.fig = fig
            self.update = update
            self.frames = frames
            for frame_index in range(frames):
                update(frame_index)

        def save(self, save_path, writer=None, dpi=None):
            saved.append((save_path, writer, dpi, self.frames))
            Path(save_path).write_text("animation")

    class FakeMoon:
        def position(self, t):
            n = len(t)
            return np.vstack((np.ones(n), np.ones(n) * 2.0, np.ones(n) * 3.0)) * 1_000.0

    monkeypatch.setattr(orbit_animation, "FuncAnimation", FakeAnimation)
    monkeypatch.setattr(orbit_animation, "_animation_writer", lambda suffix, fps, bitrate: {"suffix": suffix, "fps": fps, "bitrate": bitrate})
    monkeypatch.setattr(orbit_animation, "_get_body", lambda name: FakeMoon())
    monkeypatch.setattr(
        orbit_animation,
        "_ground_track",
        lambda r, t, format="geodetic": (np.linspace(0.0, 0.2, len(r)), np.linspace(-0.1, 0.1, len(r)), np.zeros(len(r))),
    )
    monkeypatch.setattr(orbit_animation._plt, "show", lambda: None)

    r = np.array(
        [
            [7_000_000.0, 0.0, 0.0],
            [0.0, 7_100_000.0, 0.0],
            [-7_000_000.0, 0.0, 0.0],
            [0.0, -7_100_000.0, 0.0],
        ]
    )
    out = tmp_path / "orbit.gif"
    returned = orbit_animation.orbit_animation(
        r,
        t=np.arange(4.0),
        save_path=out,
        views=("xy", "groundtrack", "3d", "globe"),
        max_frames=3,
        tail=2,
        fps=4,
        c="white",
        show=True,
    )

    assert Path(returned) == out
    assert out.exists()
    assert saved[0][1]["suffix"] == ".gif"
    assert saved[0][1]["fps"] == 4
    assert saved[0][3] == 3

    with pytest.raises(ValueError, match="unsupported"):
        orbit_animation.orbit_animation(r, t=np.arange(4.0), save_path=tmp_path / "bad.gif", views=("bad",))
    with pytest.raises(ValueError, match="must end"):
        orbit_animation.orbit_animation(r, t=np.arange(4.0), save_path=tmp_path / "bad.png")
    with pytest.raises(ValueError, match="Empty trajectory"):
        orbit_animation.orbit_animation([], t=[], save_path=tmp_path / "empty.gif")


def _transfer_fixture():
    burn1 = SimpleNamespace(t_start=0.0, t_end=10.0, dv_mag=20.0, thrust=5.0, propellant_mass=1.5)
    burn2 = SimpleNamespace(t_start=30.0, t_end=50.0, dv_mag=10.0, thrust=None, propellant_mass=None)
    t = np.array([0.0, 25.0, 50.0])
    r = np.array([[7000e3, 0.0, 0.0], [0.0, 7100e3, 0.0], [-7000e3, 0.0, 1000.0]])
    v = np.array([[0.0, 7500.0, 0.0], [-7500.0, 0.0, 0.0], [0.0, -7500.0, 0.0]])
    transfer = SimpleNamespace(
        burns=[burn1, burn2],
        trajectory={"t": t, "r": r, "v": v},
        dv_total=30.0,
        arrival_error=12.0,
    )
    return SimpleNamespace(transfer=transfer)


def test_transfer_plot_helpers_and_variants(tmp_path):
    result = _transfer_fixture()
    assert "prop" in _burn_label(1, result.transfer.burns[0])

    fig = transfer_burn_profile_plot(result)
    assert len(fig.axes) == 2
    plt.close(fig)

    grid = {
        "t_dep": np.array([0.0, 10.0, 20.0]),
        "tof": np.array([100.0, 200.0, 300.0]),
        "cost": np.array([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0], [12.0, 22.0, 32.0]]),
    }
    designer = SimpleNamespace(
        grid=grid,
        pareto={"dv": np.array([30.0, 20.0, 25.0]), "dv1": np.array([10.0, 8.0, 9.0]), "dv2": np.array([20.0, 12.0, 16.0])},
        t_depart=10.0,
        tof=200.0,
        dv_total=20.0,
        dv_budget=40.0,
        arrival_burn=True,
        rendezvous=True,
        objective="min_dv",
    )
    fig = transfer_designer_curves_plot(designer)
    assert len(fig.axes) >= 2
    plt.close(fig)

    ax = transfer_trajectory_plot(result, show_orbits=False)
    assert ax.get_title()
    plt.close(ax.figure)
    ax = transfer_trajectory_plot(result, show_orbits=False, three_d=True, annotate_burns=False)
    assert hasattr(ax, "get_zlim")
    plt.close(ax.figure)
    with pytest.raises(ValueError, match="no trajectory"):
        transfer_trajectory_plot(SimpleNamespace(transfer=SimpleNamespace(trajectory=None)))


def test_divergence_and_rendezvous_plots(monkeypatch, capsys):
    r_vectors = np.array([[7000e3, 0.0, 0.0], [7000e3 + 10.0, 5.0, 2.0], [7000e3 - 8.0, -3.0, 4.0]])
    fig = divergence_plot_func(r_vectors, r_center=r_vectors[0], v_center=np.array([0.0, 7500.0, 0.0]), show=False)
    assert "Projected position offsets" in capsys.readouterr().out
    plt.close(fig)
    with pytest.raises(ValueError, match="v_center"):
        divergence_plot_func(r_vectors, show=False)

    rendezvous_module = importlib.import_module("ssapy_toolkit.plots.rendezvous_plot")

    class FakeOrbit:
        def __init__(self, r, v, t=0):
            self.r = r
            self.v = v
            self.t = t
            self.period = 3

    def fake_rv(_orbit, time):
        time = np.asarray(time, dtype=float)
        return np.column_stack((time + 1.0, time + 2.0, time + 3.0)) * 1000.0, None

    monkeypatch.setattr(rendezvous_module, "Orbit", FakeOrbit)
    monkeypatch.setattr(rendezvous_module, "rv", fake_rv)
    r1 = np.array([7000e3, 0.0, 0.0])
    v1 = np.array([0.0, 7500.0, 0.0])
    rtransfer = np.array([[7000e3, 0.0, 0.0], [0.0, 7000e3, 0.0]])
    fig = rendezvous_plot_func(r1, v1, rtransfer, np.zeros_like(rtransfer), r2=r1, v2=v1, title="Rendezvous")
    assert fig.axes[0].get_title() == "Rendezvous"
    plt.close(fig)


def test_gifify_modes_injection_and_validation(monkeypatch, tmp_path, capsys):
    gifify_module = importlib.import_module("ssapy_toolkit.plots.gifify")
    appended = []

    class FakeWriter:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def append_data(self, arr):
            appended.append(np.asarray(arr).shape)

    monkeypatch.setattr(gifify_module.imageio, "get_writer", lambda *args, **kwargs: FakeWriter())

    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [1_000.0, 2_000.0, 3_000.0],
            [2_000.0, 3_000.0, 4_000.0],
            [3_000.0, 4_000.0, 5_000.0],
        ]
    )
    other = xyz + 100.0

    def plot_with_kw(a, b, *, ax=None, limit=None, show=False, save_path=None):
        ax = ax or plt.figure().add_subplot(111, projection="3d")
        ax.plot(a[:, 0], a[:, 1], a[:, 2])
        ax.set_title("kw")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if limit is not None:
            ax.set_xlim(-limit, limit)
        return ax

    result = gifify_module.gifify(
        plot_with_kw,
        xyz,
        other,
        save_path=tmp_path / "kw.gif",
        inject_ax=True,
        ax_kw_key="ax",
        verbose=True,
        dpi=50,
        step=2,
    )
    assert result["frames"] == 3
    assert appended
    assert "Precomputed 3D limits" in capsys.readouterr().out

    def plot_with_arg(ax, *, first, second, **kwargs):
        ax.plot(first[:, 0], second[:, 0])
        return ax.figure

    chunked = gifify_module.gifify(
        plot_with_arg,
        save_path=tmp_path / "chunks.gif",
        array_kw_keys=("first", "second"),
        first=xyz,
        second=other,
        mode="chunks",
        chunk_size=2,
        inject_ax=True,
        ax_arg_index=0,
        fixed_limits=True,
    )
    assert chunked["mode"] == "chunks"
    assert chunked["frames"] == 2

    def plot_none(a, b, **kwargs):
        plt.plot(a[:, 0], b[:, 0])

    sliding = gifify_module.gifify(
        plot_none,
        xyz,
        other,
        savefig=tmp_path / "sliding.gif",
        mode="sliding",
        chunk_size=2,
        step=1,
        fixed_limits=False,
    )
    assert sliding["mode"] == "sliding"
    assert sliding["frames"] == 4

    def plot_ignore_arrays(a, b, **kwargs):
        return plt.figure()

    object_xyz = np.array([["x", "y", "z"], ["u", "v", "w"]], dtype=object)
    gifify_module.gifify(
        plot_ignore_arrays,
        object_xyz,
        object_xyz,
        save_path=tmp_path / "object.gif",
        verbose=True,
    )
    assert "Could not precompute 3D limits" in capsys.readouterr().out

    probe_calls = {"count": 0}

    def plot_probe_fails_once(a, b, **kwargs):
        if probe_calls["count"] == 0:
            probe_calls["count"] += 1
            raise RuntimeError("probe only")
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        return fig

    gifify_module.gifify(
        plot_probe_fails_once,
        xyz,
        other,
        save_path=tmp_path / "probe.gif",
        verbose=True,
        step=3,
    )
    assert "Probe failed" in capsys.readouterr().out

    with pytest.raises(ValueError, match="array_kw_keys"):
        gifify_module.gifify(plot_none, xyz, other, array_kw_keys=("a",), save_path=tmp_path / "bad.gif")
    with pytest.raises(ValueError, match="array_arg_indices"):
        gifify_module.gifify(plot_none, xyz, other, array_arg_indices=(0,), save_path=tmp_path / "bad.gif")
    with pytest.raises(ValueError, match="mode"):
        gifify_module.gifify(plot_none, xyz, other, mode="bad", save_path=tmp_path / "bad.gif")
    with pytest.raises(ValueError, match="start/end"):
        gifify_module.gifify(plot_none, xyz, other, start=3, end=2, save_path=tmp_path / "bad.gif")
    with pytest.raises(ValueError, match="chunk_size"):
        gifify_module.gifify(plot_none, xyz, other, mode="chunks", save_path=tmp_path / "bad.gif")
    with pytest.raises(ValueError, match="step"):
        gifify_module.gifify(plot_none, xyz, other, step=0, save_path=tmp_path / "bad.gif")
