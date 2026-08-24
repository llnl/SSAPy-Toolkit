from io import BytesIO
from types import SimpleNamespace
import logging

import numpy as np
import pytest


def test_html_to_gif_crop_and_fake_browser(monkeypatch, tmp_path):
    module = pytest.importorskip("ssapy_toolkit.io.html_to_gif")
    Image = module.Image

    blank = Image.new("RGB", (5, 5), "white")
    assert module._auto_crop_bbox(blank) is None
    marked = blank.copy()
    marked.putpixel((2, 3), (255, 0, 0))
    assert module._auto_crop_bbox(marked) == (2, 3, 3, 4)

    png_buffer = BytesIO()
    marked.save(png_buffer, format="PNG")
    png_bytes = png_buffer.getvalue()

    class FakeOptions:
        def __init__(self):
            self.arguments = []
            self.binary_location = None

        def add_argument(self, value):
            self.arguments.append(value)

    captured = {}

    class FakeDriver:
        def __init__(self, options=None):
            self.options = options
            self.quit_called = False
            captured["options"] = options

        def get(self, url):
            self.url = url

        def set_window_size(self, width, height):
            self.window_size = (width, height)

        def get_screenshot_as_png(self):
            return png_bytes

        def quit(self):
            self.quit_called = True

    monkeypatch.setattr(module, "Options", FakeOptions)
    monkeypatch.setattr(module.webdriver, "Chrome", FakeDriver)
    monkeypatch.setenv("SSATK_CHROME_BINARY", "/opt/chrome")
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    ticks = iter([0.0, 10.0])
    monkeypatch.setattr(module.time, "time", lambda: next(ticks, 10.0))

    html_path = tmp_path / "demo.html"
    html_path.write_text("<html></html>")
    out_gif = tmp_path / "nested" / "demo.gif"
    module.html_to_gif(html_path=str(html_path), out_gif=str(out_gif), duration_s=0.1, fps=2, wait_after_load_s=0.0)
    assert out_gif.exists()
    assert captured["options"].binary_location == "/opt/chrome"

    module.main([str(html_path), str(tmp_path / "main.gif"), "--duration-s", "0.1", "--fps", "2"])
    assert (tmp_path / "main.gif").exists()

    with pytest.raises(FileNotFoundError):
        module.html_to_gif(html_path=str(tmp_path / "missing.html"), out_gif=str(out_gif))


def test_divergence_gif_generates_frames_and_invokes_writer(monkeypatch, tmp_path):
    import inspect

    module = pytest.importorskip("ssapy_toolkit.plots.divergence_gif")
    write_module = pytest.importorskip("ssapy_toolkit.plots.write_gifs")
    calls = []

    def fake_write_gif(**kwargs):
        calls.append(kwargs)
        open(kwargs["gif_name"], "wb").write(b"GIF89a")

    monkeypatch.setattr(write_module, "write_gif", fake_write_gif)

    times = np.array([0.0, 10.0, 20.0])
    nominal = np.array([[7_000_000.0, 0.0, 0.0], [7_000_010.0, 0.0, 0.0], [7_000_020.0, 0.0, 0.0]])
    histories = np.stack(
        [
            nominal + np.array([0.0, 1.0, 0.0]),
            nominal + np.array([0.0, -2.0, 1.0]),
            nominal + np.array([0.0, 3.0, -1.0]),
        ]
    )
    velocities = np.repeat([[1.0, 7_500.0, 10.0]], len(times), axis=0)

    out = tmp_path / "divergence.gif"
    returned = module.divergence_gif(histories, times, str(out), r_nominal_hist=nominal, v_nominal_hist=velocities, duration=0.2)

    assert returned == str(out)
    assert out.exists()
    assert calls[0]["gif_name"] == str(out)
    assert len(calls[0]["frames"]) == len(times)
    assert calls[0]["duration"] == 0.2
    assert inspect.signature(module.divergence_gif).parameters["fps"].default == 1


def test_hpc_partition_helpers_cover_remainder_and_empty_slices():
    from ssapy_toolkit import hpc

    assert hpc.get_unique_id(rank=2, run_number=3, cpus_per_node=8) == 26
    assert hpc.distribute_array_no_mpi(unique_id=0, total_jobs=3, array_size=10) == (0, 4)
    assert hpc.distribute_array_no_mpi(unique_id=1, total_jobs=3, array_size=10) == (4, 7)
    assert hpc.distribute_array_no_mpi(unique_id=2, total_jobs=3, array_size=10) == (7, 10)
    assert hpc.distribute_array_no_mpi(unique_id=9, total_jobs=10, array_size=3) == (None, None)


def test_ssatk_logging_builds_file_and_log_print(tmp_path, capsys):
    module = pytest.importorskip("ssapy_toolkit.io.ssatk_logging")
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    for handler in original_handlers:
        root.removeHandler(handler)
    try:
        module.build_logging("abc", str(tmp_path))
        module.log_print("visible message", show_prints=True)
        module.log_print("hidden message", show_prints=False)
        for handler in root.handlers:
            handler.flush()
        log_path = tmp_path / "id_abc.log"
        assert log_path.exists()
        text = log_path.read_text(encoding="utf-8")
        assert "[ID: abc]" in text
        assert "visible message" in text
        assert "hidden message" in text
        out = capsys.readouterr().out
        assert "Logging initialized" in out
        assert "visible message" in out
        assert "hidden message" not in out

        module.build_logging("abc", str(tmp_path))
        assert log_path.exists()
    finally:
        for handler in root.handlers[:]:
            handler.close()
            root.removeHandler(handler)
        for handler in original_handlers:
            root.addHandler(handler)


def test_orbit_initializer_uses_moon_state(monkeypatch):
    import ssapy_toolkit.orbit_initializer as module

    class FakeMoon:
        def position(self, t):
            gps = getattr(t, "gps", t)
            return np.array([10.0 + float(gps), 0.0, 0.0])

    class FakeOrbit:
        def __init__(self, r, v, t):
            self.r = np.asarray(r)
            self.v = np.asarray(v)
            self.t = t

    t = SimpleNamespace(gps=5.0)
    monkeypatch.setattr(module, "get_body", lambda name: FakeMoon())
    monkeypatch.setattr(module, "Orbit", FakeOrbit)

    dro = module.OrbitInitialize.DRO(t, delta_r=2.0, delta_v=3.0)
    l4 = module.OrbitInitialize.Lunar_L4(t, delta_r=4.0, delta_v=5.0)

    np.testing.assert_allclose(dro.r, [13.0, 0.0, 0.0])
    np.testing.assert_allclose(dro.v, [4.0, 0.0, 0.0])
    np.testing.assert_allclose(l4.r, [11.0, 0.0, 0.0])
    np.testing.assert_allclose(l4.v, [6.0, 0.0, 0.0])
