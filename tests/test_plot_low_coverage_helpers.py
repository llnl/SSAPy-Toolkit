import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image


class FakeOrbit:
    def __init__(self, r, v, t=0.0, mu=None, **kwargs):
        self.r = np.asarray(r, dtype=float)
        self.v = np.asarray(v, dtype=float)
        self.t = t
        self.mu = mu
        self.period = kwargs.get("period", 3.0)
        self.a = kwargs.get("a", np.linalg.norm(self.r) or 1.0)
        self.e = kwargs.get("e", 0.1)
        self.i = kwargs.get("i", np.deg2rad(5.0))


def _fake_leapfrog(r, v, t):
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        t = np.array([0.0])
    r0 = np.asarray(r, dtype=float)
    v0 = np.asarray(v, dtype=float)
    rr = r0[None, :] + 0.001 * t[:, None] * v0[None, :]
    vv = np.repeat(v0[None, :], len(t), axis=0)
    return rr, vv


def test_transfer_plot_and_intersection_helpers(monkeypatch, tmp_path):
    module = importlib.import_module("ssapy_toolkit.plots.transfer_plot")
    monkeypatch.setattr(module, "Orbit", FakeOrbit)
    monkeypatch.setattr(module, "leapfrog", _fake_leapfrog)
    saved = []
    monkeypatch.setattr(module, "figsave", lambda fig, path: saved.append(Path(path)))

    idx = module.find_intersection_time(np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([0.002, 0.0, 0.0]), 4)
    assert idx == 2

    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7500.0, 0.0])
    rf = np.array([0.0, 8000e3, 0.0])
    vf = np.array([-7000.0, 0.0, 0.0])
    r_transfer = np.vstack([r0, 0.5 * (r0 + rf), rf])
    v_transfer = np.vstack([v0, 0.5 * (v0 + vf), vf])

    fig = module.transfer_plot(r0, v0, r_transfer, v_transfer, rf, vf, c="white", savefig=tmp_path / "transfer.png", show=False)
    assert fig.axes[0].get_title() == ""
    assert saved == [tmp_path / "transfer.png"]
    plt.close(fig)

    fig = module.transfer_plot(r0, v0, r0, v0, rf, vf, c="not-a-theme", show=False)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_orbit_plot_rv_branches(monkeypatch, tmp_path):
    module = importlib.import_module("ssapy_toolkit.plots.orbit_plot_rv")
    monkeypatch.setattr(module, "Orbit", FakeOrbit)
    monkeypatch.setattr(module, "leapfrog", _fake_leapfrog)
    saved = []
    monkeypatch.setattr(module, "figsave", lambda fig, path: saved.append(Path(path)))

    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7500.0, 0.0])
    module.orbit_plot_rv((r0, v0), show=False, c="black", savepath=tmp_path / "orbit.png", title="Orbit")
    assert saved == [tmp_path / "orbit.png"]

    module.orbit_plot_rv([(r0, v0), (r0 * 1.1, v0 * 0.9)], show=False, c="bad-theme")
    with pytest.raises(ValueError, match="state_vectors"):
        module.orbit_plot_rv([r0, v0], show=False)
    with pytest.raises(TypeError, match="unexpected keyword"):
        module.orbit_plot_rv((r0, v0), show=False, bad=True)


def test_tracking_plot_with_fake_groundtrack_and_image(monkeypatch, tmp_path):
    module = importlib.import_module("ssapy_toolkit.plots.tracking_plot")
    image = Image.new("RGB", (16, 8), "blue")
    monkeypatch.setattr(module.PILImage, "open", lambda path: image.copy())
    monkeypatch.setattr(module, "find_file", lambda *args, **kwargs: "earth.png")
    monkeypatch.setattr(module, "figsave", lambda fig, path: Path(path).write_text("saved"))

    def fake_groundtrack(r, t):
        n = len(r)
        lon = np.linspace(0.0, 2.0 * np.pi, n)
        lat = np.linspace(-0.2, 0.2, n)
        return lon, lat, np.zeros(n)

    monkeypatch.setattr(module, "groundTrack", fake_groundtrack)
    r = np.array([[7000e3, 0.0, 0.0], [0.0, 7000e3, 0.0], [-7000e3, 0.0, 1000.0]])
    t = np.array([0.0, 1.0, 2.0])
    fig = module.tracking_plot([r, r * 1.01], [t, t], ground_stations=[(35.0, -106.0)], limits=None, save=tmp_path / "track.txt", scale=100)
    assert (tmp_path / "track.txt").read_text() == "saved"
    assert len(fig.axes) == 5
    plt.close(fig)

    fig = module.tracking_plot(r, t, limits=2.0, scale=100)
    assert len(fig.axes) == 5
    plt.close(fig)


def test_groundtrack_video_helpers_and_fake_writer(monkeypatch, tmp_path, capsys):
    module = importlib.import_module("ssapy_toolkit.plots.groundtrack_video")
    assert module._as_list(1) == [1]
    assert module._broadcast_time_list([1, 2], ["a", "b"]) == ["a", "b"]
    with pytest.raises(ValueError, match="length"):
        module._broadcast_time_list([1, 2], ["a"])
    np.testing.assert_array_equal(module._ensure_Nx3(np.ones((3, 2))), np.ones((2, 3)))
    with pytest.raises(ValueError, match="2D"):
        module._ensure_Nx3(np.ones(3))
    lon, lat = module._clean_lonlat_wrap(np.array([170.0, -170.0]), np.array([1.0, 2.0]))
    assert np.isnan(lon[1]) and np.isnan(lat[1])

    class DummyWriter:
        def __init__(self, *args, **kwargs):
            self.frames = 0

        class _Saving:
            def __enter__(self):
                return None

            def __exit__(self, *exc):
                return False

        def saving(self, fig, save_path, dpi):
            return self._Saving()

        def grab_frame(self):
            self.frames += 1

    monkeypatch.setattr(module, "_ensure_ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(module, "FFMpegWriter", DummyWriter)
    monkeypatch.setattr(module, "_try_load_earth", lambda: np.zeros((2, 2, 3)))

    def fake_groundtrack(r, t, format="geodetic"):
        n = len(r)
        return np.linspace(0.0, 0.1, n), np.linspace(0.2, 0.3, n), np.zeros(n)

    monkeypatch.setattr(module, "groundTrack", fake_groundtrack)
    r = np.array([[7000e3, 0.0, 0.0], [0.0, 7000e3, 0.0], [-7000e3, 0.0, 0.0], [0.0, -7000e3, 0.0]])
    out = module.groundtrack_video([r, r * 1.01], [np.arange(4.0), np.arange(4.0)], ground_stations=[(35.0, -106.0)], save_path=tmp_path / "gt.mp4", max_frames=2, progress=True)
    assert out == tmp_path / "gt.mp4"
    assert "Rendering MP4" in capsys.readouterr().out

    with pytest.raises(ValueError, match=".mp4"):
        module.groundtrack_video(r, np.arange(4.0), save_path=tmp_path / "bad.gif")
    monkeypatch.setattr(module, "_ensure_ffmpeg_path", lambda: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        module.groundtrack_video(r, np.arange(4.0), save_path=tmp_path / "missing.mp4")


def test_groundtrack_video_ffmpeg_fallback_handles_imageio_errors(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.plots.groundtrack_video")
    monkeypatch.setattr(module.shutil, "which", lambda name: None)

    class BrokenImageioFFmpeg:
        @staticmethod
        def get_ffmpeg_exe():
            raise RuntimeError("ffmpeg unavailable")

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", BrokenImageioFFmpeg)

    assert module._ensure_ffmpeg_path() is None


def test_groundtrack_video_earth_texture_fallback(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.plots.groundtrack_video")
    monkeypatch.setattr("ssapy_toolkit.plots.plotutils.load_earth_file", lambda: (_ for _ in ()).throw(FileNotFoundError("earth")))

    assert module._try_load_earth() is None


def test_new_plot_sun_direction_eci_uses_equatorial_obliquity():
    from ssapy_toolkit.plots.eclipse_brightness_plot import sun_direction_eci as eclipse_sun
    from ssapy_toolkit.plots.globe_orbit_daynight_plotly import sun_direction_eci as globe_sun

    t_s = np.array([0.0, 91.0 * 86400.0, 182.0 * 86400.0])
    eclipse_vectors = eclipse_sun(t_s)
    globe_vectors = globe_sun(t_s)

    assert np.allclose(eclipse_vectors, globe_vectors)
    assert np.allclose(np.linalg.norm(eclipse_vectors, axis=1), 1.0)
    assert np.max(np.abs(eclipse_vectors[:, 2])) > 0.1



def test_satellite_viewer_scene_uses_physical_sun_moon_defaults():
    scene = Path("ssapy_toolkit/plots/satellite_viewer_scene.js").read_text()

    assert "const SUN_RENDER_DISTANCE_KM = AU_KM;" in scene
    assert "const SUN_RENDER_RADIUS_KM = R_SUN_KM;" in scene
    assert "const MOON_RENDER_DISTANCE_KM = MOON_MEAN_DISTANCE_KM;" in scene
    assert "const MOON_RENDER_RADIUS_KM = R_MOON_KM;" in scene
    assert "logarithmicDepthBuffer: true" in scene
    default_keys = "const DEFAULT_DEMO_SATELLITES = ['iss', 'gps', 'weather_goes', 'cislunar_demo'];"
    assert default_keys in scene
    assert "name: 'Cislunar orbit demo'" in scene
    assert "const SATELLITE_GLYPH_SIZE_KM = 900;" in scene
    assert "const modelSize = SATELLITE_GLYPH_SIZE_KM;" in scene
    assert "framingR * 0.11" not in scene


def test_satellite_viewer_texture_fallback_without_packaged_assets(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.plots.build_satellite_viewer")

    def missing_resource(filename):
        raise module.DataResourceNotFoundError(filename)

    class NoSiblingData:
        @staticmethod
        def find_data_file(filename):
            return None

    monkeypatch.setattr(module, "read_data_binary", missing_resource)
    monkeypatch.setitem(sys.modules, "ssapy_toolkit.plots.starfield", NoSiblingData)

    textures = module.load_textures()

    assert set(textures) == {"day", "night", "specular", "clouds"}
    assert all(isinstance(value, str) and len(value) > 20 for value in textures.values())
