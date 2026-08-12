from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from ssapy_toolkit._paths import safe_relative_parts
from ssapy_toolkit._sorting import natural_key
from ssapy_toolkit.io.datapath import datapath, dpath
from ssapy_toolkit.io.h5cache import h5cache, h5load
from ssapy_toolkit.io.ssatk_cache import ssatk_cache, ssatk_load
from ssapy_toolkit.io.ssatk_data import ssatk_data
from ssapy_toolkit.orbital_mechanics._two_body import _keplerian_two_body_rhs
from ssapy_toolkit.plots.figpath import figpath, fpath, ssatk_path
from ssapy_toolkit.plots.plotutils import figsave, fsave, ssatk_fig

data_path_module = importlib.import_module("ssapy_toolkit.io.datapath")
fig_path_module = importlib.import_module("ssapy_toolkit.plots.figpath")
top_level_launch_pads = importlib.import_module("ssapy_toolkit.launch_pads")
orbital_launch_pads = importlib.import_module("ssapy_toolkit.orbital_mechanics.launch_pads")


def test_natural_key_sorts_embedded_numbers():
    names = ["frame_10.png", "frame_2.png", "frame_1.png"]

    assert sorted(names, key=natural_key) == [
        "frame_1.png",
        "frame_2.png",
        "frame_10.png",
    ]


def test_safe_relative_parts_strips_roots_and_collapses_parent_segments():
    assert safe_relative_parts("/absolute/path/../figure.png") == [
        "absolute",
        "figure.png",
    ]
    assert safe_relative_parts("../../data/catalog.csv") == ["data", "catalog.csv"]
    assert safe_relative_parts("./nested/./file.txt") == ["nested", "file.txt"]


def test_keplerian_two_body_rhs_returns_velocity_and_gravity():
    mu = 4.0
    state = np.array([2.0, 0.0, 0.0, 0.0, 3.0, 0.0])

    rhs = _keplerian_two_body_rhs(123.0, state, mu)

    np.testing.assert_allclose(rhs, [0.0, 3.0, 0.0, -1.0, 0.0, 0.0])


def test_ssatk_short_helper_aliases_are_primary_exports():
    assert ssatk_path is figpath
    assert fpath is figpath
    assert ssatk_fig is figsave
    assert fsave is figsave
    assert dpath is datapath
    assert callable(ssatk_data)
    assert callable(h5cache)
    assert callable(h5load)
    assert callable(ssatk_cache)
    assert callable(ssatk_load)


def test_launch_pad_metadata_uses_one_canonical_dataset():
    assert orbital_launch_pads.launch_pads is top_level_launch_pads.launch_pads
    assert orbital_launch_pads.landing_pads is top_level_launch_pads.landing_pads


def test_figpath_roots_relative_paths_under_home_output_dir(tmp_path, monkeypatch):
    home_figs = tmp_path / "home_figs"
    monkeypatch.setattr(fig_path_module, "HOME_FIG_DIR", home_figs)

    path = Path(fig_path_module.figpath("demo/../plots/example"))

    assert path == home_figs / "plots" / "example"
    assert path.parent.exists()


def test_figpath_can_use_explicit_env_root(tmp_path, monkeypatch):
    env_figs = tmp_path / "env_figs"
    monkeypatch.setenv("SSATK_FIGURES_DIR", str(env_figs))

    path = Path(fig_path_module.figpath("demo_gallery/index.html"))

    assert path == env_figs / "demo_gallery" / "index.html"
    assert path.parent.exists()


def test_figpath_does_not_fall_back_to_cwd(tmp_path, monkeypatch):
    blocked_home = tmp_path / "blocked_home"
    blocked_home.write_text("not a directory", encoding="utf-8")
    fallback = tmp_path / "fallback_figs"
    monkeypatch.setattr(fig_path_module, "HOME_FIG_DIR", blocked_home)
    monkeypatch.setattr(fig_path_module, "FALLBACK_DIR", fallback)

    with pytest.raises(RuntimeError, match="SSATK_FIGURES_DIR"):
        fig_path_module.figpath("plots/example")

    assert not fallback.exists()


def test_figsave_defaults_to_figpath_and_adds_jpg_extension(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(fig_path_module, "HOME_FIG_DIR", tmp_path / "figs")

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    saved = Path(figsave(fig, "quicklook/test_plot"))

    assert saved == tmp_path / "figs" / "quicklook" / "test_plot.jpg"
    assert saved.exists()


def test_figsave_without_path_uses_home_figure_default(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(fig_path_module, "HOME_FIG_DIR", tmp_path / "figs")

    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])

    saved = Path(figsave(fig))

    assert saved == tmp_path / "figs" / "figure.jpg"
    assert saved.exists()


def test_figsave_accepts_relative_save_aliases(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(fig_path_module, "HOME_FIG_DIR", tmp_path / "figs")

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    saved = Path(ssatk_fig(fig, save="aliases/relative_plot"))

    assert saved == tmp_path / "figs" / "aliases" / "relative_plot.jpg"
    assert saved.exists()


def test_figsave_honors_absolute_save_aliases(tmp_path):
    import matplotlib.pyplot as plt

    output_path = tmp_path / "absolute" / "plot.png"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])

    saved = Path(figsave(fig, save_fig=output_path))

    assert saved == output_path
    assert saved.exists()


def test_figsave_rejects_conflicting_save_aliases():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])

    with pytest.raises(TypeError, match="Conflicting figure save aliases"):
        figsave(fig, save="one", save_figure="two")

    plt.close(fig)


def test_datapath_uses_ssatk_data_dir_and_custom_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(data_path_module, "HOME_DATA_DIR", tmp_path / "home_data")

    default_path = Path(data_path_module.datapath("catalogs/sample.txt"))
    custom_path = Path(data_path_module.datapath("cache/sample.npy", dirs=[tmp_path / "custom_data"]))

    assert default_path == tmp_path / "home_data" / "catalogs" / "sample.txt"
    assert default_path.parent.exists()
    assert custom_path == tmp_path / "custom_data" / "cache" / "sample.npy"
    assert custom_path.parent.exists()


def test_datapath_can_use_explicit_env_root(tmp_path, monkeypatch):
    env_data = tmp_path / "env_data"
    monkeypatch.setenv("SSATK_DATA_DIR", str(env_data))

    path = Path(data_path_module.datapath("catalogs/sample.txt"))

    assert path == env_data / "catalogs" / "sample.txt"
    assert path.parent.exists()


def test_datapath_does_not_fall_back_to_cwd(tmp_path, monkeypatch):
    blocked_home = tmp_path / "blocked_home"
    blocked_home.write_text("not a directory", encoding="utf-8")
    fallback = tmp_path / "fallback_data"
    monkeypatch.setattr(data_path_module, "HOME_DATA_DIR", blocked_home)
    monkeypatch.setattr(data_path_module, "FALLBACK_DATA_DIR", fallback)

    with pytest.raises(RuntimeError, match="SSATK_DATA_DIR"):
        data_path_module.datapath("catalogs/sample.txt")

    assert not fallback.exists()
