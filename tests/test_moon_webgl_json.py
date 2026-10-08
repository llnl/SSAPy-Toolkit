"""Regression tests for Moon WebGL orbit JSON input."""

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path


_MODULE_PATH = Path(__file__).parents[1] / "ssapy_toolkit" / "plots" / "moon_webgl.py"
_SPEC = importlib.util.spec_from_file_location("moon_webgl_test_module", _MODULE_PATH)
MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(MODULE)

_BAKER_PATH = Path(__file__).parents[1] / "ssapy_toolkit" / "io" / "moon_maps.py"
_BAKER_SPEC = importlib.util.spec_from_file_location("moon_maps_test_module", _BAKER_PATH)
BAKER = importlib.util.module_from_spec(_BAKER_SPEC)
assert _BAKER_SPEC.loader is not None
_BAKER_SPEC.loader.exec_module(BAKER)


def _gdo_payload():
    return {
        "source": "https://gdo-cislunar.llnl.gov/",
        "frame": "moon_centered",
        "units": "km",
        "orbits": [
            {"oid": 546929, "resident": True,
             "xyz": [[1800.0, 0.0, 0.0], [0.0, 1800.0, 50.0], [-1800.0, 0.0, 0.0]]},
            {"oid": 776801, "resident": True,
             "xyz": [[2200.0, 0.0, 200.0], [0.0, -2200.0, 0.0], [-2200.0, 0.0, -200.0]]},
        ],
    }


def test_cache_and_server_helpers_work_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("SSAPY_TOOLKIT_CACHE", str(tmp_path))
    assert MODULE.cache_dir() == str(tmp_path)

    for name in MODULE._THREE_FILES:
        (tmp_path / name).write_bytes(b"runtime")
    assert MODULE.ensure_three() == {
        name: f"/cache/{name}" for name in MODULE._THREE_FILES
    }

    meta = {"n_az": 16}
    (tmp_path / "moon_albedo.jpg").write_bytes(b"albedo")
    (tmp_path / "moon_normal.png").write_bytes(b"normal")
    (tmp_path / "moon_horizon_meta.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )
    for index in range(meta["n_az"] // 4):
        (tmp_path / f"moon_horizon_{index}.png").write_bytes(b"horizon")
    assert MODULE.find_moon_cache() == (str(tmp_path), meta)

    html_path = tmp_path / "moon.html"
    html_path.write_text("moon", encoding="utf-8")
    calls = {}

    class Server:
        def __init__(self, address, handler):
            calls["address"] = address
            self.server_address = (address[0], 4321)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def serve_forever(self):
            calls["served"] = True
            raise KeyboardInterrupt

    monkeypatch.setattr(MODULE.socketserver, "TCPServer", Server)
    MODULE.show(html_path, open_browser=False)

    assert calls == {"address": ("127.0.0.1", 0), "served": True}


def test_server_only_serves_page_cache_and_runtime(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    page = tmp_path / "index.html"
    page.write_text("page", encoding="utf-8")
    (cache / "asset.txt").write_text("asset", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")

    server = HTTPServer(("127.0.0.1", 0), MODULE._handler(str(page), str(cache)))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        root = f"http://127.0.0.1:{server.server_port}"
        assert urllib.request.urlopen(root + "/").read() == b"page"
        assert urllib.request.urlopen(root + "/cache/asset.txt").read() == b"asset"
        for path in ("/secret.txt", "/cache/../secret.txt"):
            try:
                urllib.request.urlopen(root + path)
            except urllib.error.HTTPError as exc:
                assert exc.code == 404
            else:
                raise AssertionError(f"server exposed {path}")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_missing_cache_raises_file_not_found(tmp_path):
    try:
        MODULE.find_moon_cache(tmp_path)
    except FileNotFoundError as exc:
        assert "missing baked textures" in str(exc)
    else:
        raise AssertionError("missing assets should fail")


def test_default_bake_produces_complete_cache(monkeypatch, tmp_path):
    import numpy as np
    from PIL import Image

    dem_path = tmp_path / "dem.npz"
    albedo_path = tmp_path / "albedo.png"
    np.savez(dem_path, elev_km=np.zeros((4, 8), dtype=np.float32))
    Image.new("RGB", (8, 4), "gray").save(albedo_path)

    def source(url, directory):
        return str(dem_path if url == BAKER.DEM_URL else albedo_path)

    monkeypatch.setattr(BAKER, "download_source", source)
    BAKER.main([
        "--outdir", str(tmp_path), "--n-az", "4", "--n-step", "1",
        "--horizon-width", "8", "--albedo-max", "8",
    ])

    cache, meta = MODULE.find_moon_cache(tmp_path)
    assert cache == str(tmp_path)
    assert meta["n_az"] == 4


def test_baker_cache_dir_honours_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SSAPY_TOOLKIT_CACHE", str(tmp_path))
    assert BAKER.cache_dir() == str(tmp_path)

    monkeypatch.delenv("SSAPY_TOOLKIT_CACHE")
    assert Path(BAKER.cache_dir()).parts[-2:] == (".ssapy_toolkit", "moon")


def test_download_source_fetches_once_and_cleans_up_failures(monkeypatch, tmp_path):
    calls = []

    def retrieve(url, filename):
        calls.append(url)
        Path(filename).write_bytes(b"source bytes")

    monkeypatch.setattr(BAKER.urllib.request, "urlretrieve", retrieve)
    url = "https://example.invalid/moon/ldem_16_uint.tif"
    path = BAKER.download_source(url, str(tmp_path))

    assert Path(path) == tmp_path / "ldem_16_uint.tif"
    assert Path(path).read_bytes() == b"source bytes"
    assert BAKER.download_source(url, str(tmp_path)) == path
    assert calls == [url]

    def interrupted(url, filename):
        Path(filename).write_bytes(b"partial")
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(BAKER.urllib.request, "urlretrieve", interrupted)
    try:
        BAKER.download_source("https://example.invalid/moon/other.tif", str(tmp_path))
    except urllib.error.URLError:
        pass
    else:
        raise AssertionError("a failed download should raise")
    assert not (tmp_path / "other.tif").exists()
    assert not (tmp_path / "other.tif.part").exists()


def test_bake_normals_tilt_downslope_in_east_north_frame():
    import numpy as np

    # Same grid as the LOLA source: row 0 at the north pole, columns eastward
    # from -180 degrees.  The shader reads red as east and green as north.
    height, width = 180, 360
    latitude = np.radians(np.linspace(90.0, -90.0, height))[:, None]
    longitude = np.radians(np.linspace(-180.0, 180.0, width, endpoint=False))[None, :]
    slope = 0.01
    rises_north = slope * BAKER.R_MOON_KM * latitude * np.ones((1, width))
    rises_east = slope * BAKER.R_MOON_KM * np.cos(latitude) * longitude

    mid_latitudes = slice(30, 151)
    north = BAKER.bake_normals(rises_north)[mid_latitudes]
    east = BAKER.bake_normals(rises_east)[mid_latitudes]

    # Ground rising northward faces south; ground rising eastward faces west.
    assert np.all(north[..., 1] < 0.0)
    assert np.all(east[..., 0] < 0.0)
    np.testing.assert_allclose(north[..., 1], -slope, rtol=0.02)
    np.testing.assert_allclose(east[..., 0], -slope, rtol=0.02)
    np.testing.assert_allclose(north[..., 0], 0.0, atol=1e-12)


def test_load_gdo_orbit_set_defaults_xyz_to_moon_centered_km(tmp_path):
    path = tmp_path / "gdo.json"
    path.write_text(json.dumps(_gdo_payload()), encoding="utf-8")

    loaded = MODULE.load_orbit_json(path)

    assert [track["name"] for track in loaded["tracks"]] == ["546929", "776801"]
    assert all(track["r_frame"] == "moon_centered" for track in loaded["tracks"])
    assert all(track["units"] == "km" for track in loaded["tracks"])
    tracks = MODULE._resolve_orbit_positions(
        None, None, "gcrf", None, None, 360, 1.0, orbit_json=path
    )
    assert len(tracks) == 2
    assert all(len(track["xyz"]) == 9 for track in tracks)


def test_load_gdo_output_frame_and_stride_metadata():
    payload = _gdo_payload()
    payload["frame"] = "moon_centered_earth_moon_rotating"
    payload["stride_hours"] = 6

    loaded = MODULE.load_orbit_json(payload)
    tracks = MODULE._resolve_orbit_positions(
        None, None, "gcrf", None, None, 360, 1.0, orbit_json=payload
    )

    assert all(track["r_frame"] == "moon_centered" for track in loaded["tracks"])
    assert all(track["step_seconds"] == 6 * 3600 for track in loaded["tracks"])
    assert all(track["durationSeconds"] == 12 * 3600 for track in tracks)


def test_orbit_set_inherits_root_units_and_keeps_per_track_override():
    payload = _gdo_payload()
    payload["units"] = "m"
    payload["orbits"][0]["xyz"] = [
        [1_800_000.0, 0.0, 0.0],
        [0.0, 1_800_000.0, 0.0],
    ]
    payload["orbits"][1]["units"] = "km"

    loaded = MODULE.load_orbit_json(payload)
    tracks = MODULE._resolve_orbit_positions(
        None, None, "gcrf", None, None, 360, 1.0, orbit_json=payload
    )

    assert [track["units"] for track in loaded["tracks"]] == ["m", "km"]
    assert tracks[0]["xyz"] == [1800.0, 0.0, 0.0, 0.0, 1800.0, 0.0]
    assert tracks[1]["xyz"][:3] == [2200.0, 0.0, 200.0]


def test_unitless_non_gdo_positions_keep_auto_detection():
    payload = {
        "frame": "moon_centered",
        "orbits": [{"r": [[1_800_000.0, 0.0, 0.0], [0.0, 1_800_000.0, 0.0]]}],
    }

    loaded = MODULE.load_orbit_json(payload)
    track = MODULE._resolve_orbit_positions(
        None, None, "gcrf", None, None, 360, 1.0, orbit_json=payload
    )[0]

    assert loaded["tracks"][0]["units"] == "auto"
    assert track["xyz"] == [1800.0, 0.0, 0.0, 0.0, 1800.0, 0.0]


def test_keplerian_json_defaults_to_moon_gravity_and_frame():
    from ssapy_toolkit.constants import MOON_MU

    payload = {
        "elements": {"a": 2_737_400.0, "e": 0.01, "i": 0.2},
    }
    loaded = MODULE.load_orbit_json(payload)
    track = MODULE._resolve_orbit_positions(
        None, None, "gcrf", None, None, 8, 1.0, orbit_json=payload
    )[0]

    assert loaded["orbit"].mu == MOON_MU
    assert loaded["r_frame"] == "moon_centered"
    assert max(abs(value) for value in track["xyz"]) < 3_000.0


def test_moon_webgl_embeds_multiple_named_tracks(monkeypatch, tmp_path):
    meta = {"n_az": 16, "angle_lo_deg": 0.0, "angle_hi_deg": 90.0}
    monkeypatch.setattr(MODULE, "find_moon_cache", lambda path=None: (str(tmp_path), meta))
    monkeypatch.setattr(
        MODULE,
        "ensure_three",
        lambda directory=None: {"three.min.js": "three.js", "OrbitControls.js": "controls.js"},
    )
    monkeypatch.setattr(
        MODULE,
        "_starfield_payload",
        lambda epoch=None: {"p": [0.0, 0.0, 4.0e6], "c": [1.0, 1.0, 1.0], "s": [4.0]},
    )

    output = MODULE.moon_webgl(
        orbit_json=_gdo_payload(),
        save_path=tmp_path / "moon.html",
    )
    html = (tmp_path / "moon.html").read_text(encoding="utf-8")

    assert output == str(tmp_path / "moon.html")
    assert "const ORBITS" in html
    assert "const STARS" in html
    assert "new THREE.Points(starGeometry" in html
    assert html.count('"name": "546929"') == 1
    assert html.count('"name": "776801"') == 1
    assert 'type="range"' not in html
    assert 'id="panel"' not in html
    assert 'id="playback-toggle"' in html
    assert 'id="orbit-toolbar"' in html
    assert 'id="orbit-file-input"' in html
    assert 'id="orbit-list"' in html
    assert 'id="time-step-mode"' in html
    assert 'id="time-step-value"' in html
    assert 'id="step-back"' in html
    assert 'id="step-forward"' in html
    assert "new FileReader()" in html
    assert "normaliseOrbitJson" in html
    assert "moon_centered_earth_moon_rotating" in html
    assert "stride_hours" in html
    assert "addOrbitTrack(track, file.name)" in html
    assert "removeOrbit(visual)" in html
    assert "focusOrbit(visual)" in html
    assert "stepTimeline(direction)" in html
    assert "setTimelineMode(mode)" in html
    assert "periodInput.disabled = timelineMode !== 'day'" in html
    assert "timelineMode === 'orbit'" in html
    assert "timelineValue * 86400 / visual.durationSeconds" in html
    assert "Math.max(0, Math.min(1, rawPhase))" in html
    assert (
        "if (target >= lastOffset) {\n"
        "      i0 = visual.pointCount - 2;\n"
        "      fraction = 1;"
    ) in html
    assert "requestAnimationFrame(loop)" in html
    assert "updateOrbitAnimation(now)" in html
    assert "new THREE.Sprite" in html
    assert "function fitDistance(radius)" in html
    assert "const viewDistance = fitDistance(orbitRadius)" in html
    assert "let framedRadius = orbitRadius" in html
    assert "Math.sin(Math.min(verticalHalfFov, horizontalHalfFov))" in html
    assert "if (added) focusRadius(orbitRadius)" in html
    assert "const previousFitDistance = fitDistance(framedRadius)" in html
    assert "fitDistance(framedRadius) / previousFitDistance" in html
    assert "MAX_ANIMATED_MARKERS = 12" in html
    assert "opacity: orbitLineOpacity" in html
    assert "depthTest: true, depthWrite: false" in html


def test_orbit_tracks_preserve_independent_timing_metadata():
    payload = _gdo_payload()
    payload["orbits"][0]["times"] = [1000.0, 1600.0, 2200.0]
    payload["orbits"][1]["period_days"] = 2.0

    tracks = MODULE._resolve_orbit_positions(
        None, None, "gcrf", None, None, 360, 1.0, orbit_json=payload
    )

    assert tracks[0]["times"] == [0.0, 600.0, 1200.0]
    assert tracks[0]["durationSeconds"] == 1200.0
    assert tracks[1]["durationSeconds"] == 2.0 * 86400.0


def test_orbit_tracks_inherit_shared_root_times():
    payload = _gdo_payload()
    payload["times"] = [1000.0, 1600.0, 2200.0]

    tracks = MODULE._resolve_orbit_positions(
        None, None, "gcrf", None, None, 360, 1.0, orbit_json=payload
    )

    assert [track["times"] for track in tracks] == [
        [0.0, 600.0, 1200.0],
        [0.0, 600.0, 1200.0],
    ]


def test_orbit_json_rejects_invalid_timing_metadata():
    payload = _gdo_payload()
    payload["orbits"][0]["period_days"] = 0

    try:
        MODULE.load_orbit_json(payload)
    except ValueError as exc:
        assert "positive finite" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-positive orbit timing should be rejected")


def test_orbit_timing_aliases_inherit_and_override_per_track():
    payload = _gdo_payload()
    payload["periodDays"] = 3.0
    payload["orbits"][0]["stepDays"] = 0.25

    tracks = MODULE._resolve_orbit_positions(
        None, None, "gcrf", None, None, 360, 1.0, orbit_json=payload
    )

    assert tracks[0]["durationSeconds"] == 0.5 * 86400.0
    assert tracks[1]["durationSeconds"] == 3.0 * 86400.0


def test_explicit_orbit_duration_overrides_timestamp_span():
    payload = _gdo_payload()
    payload["orbits"] = [payload["orbits"][0]]
    payload["orbits"][0]["times"] = [1000.0, 1600.0, 2200.0]
    payload["orbits"][0]["durationSeconds"] = 7200.0

    track = MODULE._resolve_orbit_positions(
        None, None, "gcrf", None, None, 360, 1.0, orbit_json=payload
    )[0]

    assert track["times"] == [0.0, 600.0, 1200.0]
    assert track["durationSeconds"] == 7200.0


def test_moon_webgl_rejects_invalid_animation_duration(monkeypatch, tmp_path):
    meta = {"n_az": 16, "angle_lo_deg": 0.0, "angle_hi_deg": 90.0}
    monkeypatch.setattr(MODULE, "find_moon_cache", lambda path=None: (str(tmp_path), meta))

    try:
        MODULE.moon_webgl(animation_seconds=0, save_path=tmp_path / "moon.html")
    except ValueError as exc:
        assert "positive finite" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("zero animation duration should be rejected")


def test_moon_webgl_portable_export_inlines_runtime_and_textures(monkeypatch, tmp_path):
    meta = {"n_az": 16, "angle_lo_deg": 0.0, "angle_hi_deg": 90.0}
    monkeypatch.setattr(MODULE, "find_moon_cache", lambda path=None: (str(tmp_path), meta))
    monkeypatch.setattr(
        MODULE,
        "ensure_three",
        lambda directory=None: {
            "three.min.js": "/cache/three.min.js",
            "OrbitControls.js": "/cache/OrbitControls.js",
        },
    )
    epochs = []
    monkeypatch.setattr(MODULE, "_starfield_payload", lambda epoch=None: epochs.append(epoch))
    (tmp_path / "three.min.js").write_text("window.THREE_FAKE=1;", encoding="utf-8")
    (tmp_path / "OrbitControls.js").write_text(
        "window.ORBIT_CONTROLS_FAKE=1;", encoding="utf-8"
    )
    for name in (
        "moon_albedo.jpg",
        "moon_normal.png",
        "moon_horizon_0.png",
        "moon_horizon_1.png",
        "moon_horizon_2.png",
        "moon_horizon_3.png",
    ):
        (tmp_path / name).write_bytes(b"image")

    MODULE.moon_webgl(
        r=[[1800.0, 0.0, 0.0], [0.0, 1800.0, 0.0]],
        t=[0.0, 1.0],
        r_frame="moon_centered",
        embed_assets=True,
        save_path=tmp_path / "portable.html",
    )
    html = (tmp_path / "portable.html").read_text(encoding="utf-8")

    assert "window.THREE_FAKE=1" in html
    assert "window.ORBIT_CONTROLS_FAKE=1" in html
    assert "data:image/jpeg;base64,aW1hZ2U=" in html
    assert "data:image/png;base64,aW1hZ2U=" in html
    assert "<script src=" not in html
    assert epochs == [0.0]


def test_orbit_json_rejects_empty_orbit_sets():
    try:
        MODULE.load_orbit_json({"orbits": []})
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("empty orbit set should be rejected")


def test_orbit_json_loader_keeps_plot_compatibility():
    assert MODULE.load_orbit_json({"orbits": [{"xyz": [[1800, 0, 0], [0, 1800, 0]]}]})[
        "tracks"
    ][0]["units"] == "km"


def test_shared_starfield_gracefully_handles_missing_catalogue(monkeypatch):
    from ssapy_toolkit.plots import starfield

    monkeypatch.setattr(starfield, "star_directions", lambda **kwargs: None)

    assert starfield.moon_fixed_webgl_stars() is None
