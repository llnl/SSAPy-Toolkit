from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib
import numpy as np

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


def test_orbit_plot_wrappers_delegate_to_core(monkeypatch):
    cases = [
        ("ssapy_toolkit.plots.orbit_plot", "orbit_plot", ("xy", "xz", "yz", "3d"), "standard", False),
        ("ssapy_toolkit.plots.orbit_plot_xy", "orbit_plot_xy", ("xy",), "fixed", True),
        ("ssapy_toolkit.plots.orbit_plot_xyxz", "orbit_plot_xyxz", ("xy", "xz"), "fixed", False),
    ]

    for module_name, function_name, expected_views, expected_lunar_transform, expected_xy_title in cases:
        module = importlib.import_module(module_name)
        calls = []

        def fake_core(r, **kwargs):
            calls.append((r, kwargs))
            return "fig", list(kwargs["views"])

        monkeypatch.setattr(module, "_orbit_plot_core", fake_core)
        result = getattr(module, function_name)(
            _sample_track(),
            t=np.arange(3.0),
            title="demo",
            save_path="plots/orbit.jpg",
            frame="gcrf",
            show=False,
            c="black",
            pad=2,
        )

        assert result == ("fig", list(expected_views))
        assert calls[0][1]["views"] == expected_views
        assert calls[0][1]["save_path"] == "plots/orbit.jpg"
        assert calls[0][1]["lunar_transform"] == expected_lunar_transform
        assert calls[0][1].get("xy_title_includes_title", False) is expected_xy_title


def test_cislunar_plot_wrappers_delegate_to_core(monkeypatch):
    cases = [
        ("ssapy_toolkit.plots.cislunar_plot", "cislunar_plot", "combined", True),
        ("ssapy_toolkit.plots.cislunar_plot_3d", "cislunar_plot_3d", "3d", False),
        ("ssapy_toolkit.plots.cislunar_plot_xy", "cislunar_plot_xy", "xy", True),
    ]

    for module_name, function_name, expected_mode, legend_value in cases:
        module = importlib.import_module(module_name)
        calls = []

        def fake_core(r, **kwargs):
            calls.append((r, kwargs))
            return "fig", kwargs["mode"]

        monkeypatch.setattr(module, "_cislunar_plot_core", fake_core)
        kwargs = {
            "t": np.arange(3.0),
            "title": "demo",
            "save_path": "plots/cislunar.jpg",
            "show": False,
        }
        if function_name == "cislunar_plot_3d":
            kwargs["legend"] = legend_value

        result = getattr(module, function_name)(_sample_track(), **kwargs)

        assert result == ("fig", expected_mode)
        assert calls[0][1]["mode"] == expected_mode
        assert calls[0][1]["save_path"] == "plots/cislunar.jpg"
        assert calls[0][1]["legend"] is legend_value


def test_orbit_plot_core_forwards_save_path(monkeypatch, tmp_path):
    core = importlib.import_module("ssapy_toolkit.plots._orbit_plot_core")
    saved = {}

    monkeypatch.setattr(core, "_get_body", lambda name: _FakeBody())
    monkeypatch.setattr(core, "_lagrange_points_lunar_frame", lambda: {})
    monkeypatch.setattr(core, "_save_plot", lambda fig, path: saved.update(fig=fig, path=Path(path)))

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


def test_cislunar_plot_core_forwards_save_path(monkeypatch, tmp_path):
    core = importlib.import_module("ssapy_toolkit.plots._cislunar_plot_core")
    saved = {}

    monkeypatch.setattr(core, "_get_body", lambda name: _FakeBody())
    monkeypatch.setattr(core, "_gcrf_to_lunar_fixed", lambda xyz, t: xyz)
    monkeypatch.setattr(core, "_lagrange_points_lunar_fixed_frame", lambda: {})
    monkeypatch.setattr(core, "_sphere_mesh", lambda radius: tuple(np.zeros((2, 2)) for _ in range(3)))
    monkeypatch.setattr(core, "_save_plot", lambda fig, path: saved.update(fig=fig, path=Path(path)))

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
