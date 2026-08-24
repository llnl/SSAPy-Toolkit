from __future__ import annotations

import sys
import types
import importlib
from datetime import datetime
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pytest
from astropy.time import Time


def test_demo_timeout_private_handler_raises():
    from ssapy_toolkit.demo_gallery import DemoTimeoutError, demo_timeout

    timeout = demo_timeout(2.0)
    with pytest.raises(DemoTimeoutError, match="2"):
        timeout._raise_timeout(None, None)


def test_geomagnetic_private_helpers_with_synthetic_fields(monkeypatch, tmp_path):
    from ssapy_toolkit import geomagnetics as gm

    date = datetime(2025, 1, 1)
    monkeypatch.setattr(gm, "_texture_cache_dir", lambda: tmp_path)

    monkeypatch.setattr(
        gm,
        "_bfield_batch",
        lambda positions, _date: np.tile([1.0, 2.0, 2.0], (len(positions), 1)),
    )
    positions = np.array([[gm.EARTH_RADIUS_KM + 500.0, 0.0, 0.0], [0.0, gm.EARTH_RADIUS_KM + 600.0, 0.0]])
    unit = gm._bunit_batch(positions, date)
    np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1.0)
    np.testing.assert_allclose(gm._field_magnitude_along(positions, date, chunk=1), [3.0, 3.0])
    np.testing.assert_allclose(
        gm._enu_to_cartesian_batch(
            Be=np.array([1.0]),
            Bn=np.array([2.0]),
            Bu=np.array([3.0]),
            lons_deg=np.array([0.0]),
            lats_deg=np.array([0.0]),
        ),
        [[3.0, 1.0, 2.0]],
    )

    resampled = gm._resample_curve(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]), 3)
    np.testing.assert_allclose(resampled[:, 0], [0.0, 5.0, 10.0])
    np.testing.assert_allclose(gm._resample_curve(np.array([[2.0, 0.0, 0.0]]), 2), [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    def fake_trace_batch(seeds, _date, direction=1, **_kw):
        return [
            np.array([seed, seed + direction * np.array([1.0, 0.0, 0.0])])
            for seed in np.asarray(seeds, dtype=float)
        ]

    monkeypatch.setattr(gm, "_trace_batch_rk4", fake_trace_batch)
    closed = gm._trace_all_closed([np.array([1.0, 0.0, 0.0])], date)
    assert closed[0].shape == (3, 3)

    monkeypatch.setattr(
        gm,
        "_bfield_batch",
        lambda positions, _date: np.linalg.norm(positions, axis=1, keepdims=True) * np.array([[1.0, 0.0, 0.0]]),
    )
    equator, b0 = gm._true_magnetic_equator([np.array([2.0, 0.0, 0.0])], date, iters=1)
    assert equator.shape == (1, 3)
    assert b0.shape == (1,)

    def fake_true_equator(guess, _date, iters=2):
        eq = np.asarray(guess, dtype=float)
        shell_l = np.linalg.norm(eq, axis=1) / gm.EARTH_RADIUS_KM
        return eq, gm._M_DIPOLE_NT_RE3 / shell_l**3

    monkeypatch.setattr(gm, "_true_magnetic_equator", fake_true_equator)
    seeds, shell_l = gm._make_seeds_lshell([2.0, 3.0], date, n_lons=3)
    assert len(seeds) == 6
    np.testing.assert_allclose(np.sort(shell_l), np.repeat([2.0, 3.0], 3), rtol=1e-12)

    gm._AEP8_TABLE = None
    assert gm._load_aep8_table(allow_build=False) is None
    np.savez_compressed(
        gm._aep8_table_path(),
        L=np.array([1.0, 2.0]),
        B=np.array([1.0, 4.0]),
        p=np.array([[10.0, 20.0], [30.0, 40.0]]),
        e=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    table = gm._load_aep8_table(allow_build=False)
    assert set(table) == {"L", "B", "p", "e"}
    lookup = gm._aep8_lookup("p", np.array([1.5]), np.array([2.0]))
    assert lookup.shape == (1,)
    assert 10.0 < lookup[0] < 40.0

    monkeypatch.setattr(gm, "_aep8_lookup", lambda species, L, ratio: np.asarray(L) + (1 if species == "p" else 2))
    monkeypatch.setattr(
        gm,
        "_true_magnetic_equator",
        lambda seeds, _date, iters=2: (np.asarray(seeds, dtype=float), np.full(len(seeds), gm._M_DIPOLE_NT_RE3 / 8.0)),
    )
    monkeypatch.setattr(
        gm,
        "_trace_batch_rk4",
        lambda seeds, _date, direction=1, **_kw: [
            np.array([seed, seed + direction * np.array([0.0, 0.0, 10.0])])
            for seed in np.asarray(seeds, dtype=float)
        ],
    )
    monkeypatch.setattr(
        gm,
        "_bfield_batch",
        lambda positions, _date: np.tile([gm._M_DIPOLE_NT_RE3 / 8.0, 0.0, 0.0], (len(positions), 1)),
    )
    belt = gm._belt_flux_samples(date, np.array([0.0, 0.0, 1.0]), L_min=2.0, L_max=2.0, n_L=1, n_azim=1, n_pts=4, eq_iters=1)
    assert belt is not None
    belt_positions, proton_flux, electron_flux = belt
    assert belt_positions.shape == (4, 3)
    assert proton_flux.shape == electron_flux.shape == (4,)
    assert np.all(electron_flux > proton_flux)

    re = gm.EARTH_RADIUS_KM
    assert gm._classify_line(np.array([[re, 0.0, 0.0], [-re, 0.0, 0.0]])) == "closed"
    assert gm._classify_line(np.array([[re, 0.0, 0.0], [3.0 * re, 0.0, 0.0]])) == "open"
    assert gm._classify_line(np.array([[3.0 * re, 0.0, 0.0], [4.0 * re, 0.0, 0.0]])) == "detached"

    assert gm._apply_solar_wind(date, 2, -5.0, 1.5, use_omni=False) == (2, -5.0, 1.5, "nominal (user-specified)")
    monkeypatch.setattr(gm, "get_solar_wind", lambda _date: None)
    assert "unavailable" in gm._apply_solar_wind(date, 2, -5.0, 1.5)[3]
    monkeypatch.setattr(
        gm,
        "get_solar_wind",
        lambda _date: {"dp_nPa": 2.5, "bz_nT": -3.0, "kp": 4.2, "speed": 410.0, "density": 6.0},
    )
    kp, bz, dp, source = gm._apply_solar_wind(date, 2, -5.0, 1.5)
    assert (kp, bz, dp) == (4, -3.0, 2.5)
    assert "OMNI" in source


def test_geomagnetic_optional_grid_helpers_with_fake_geopack(monkeypatch, tmp_path):
    from ssapy_toolkit import geomagnetics as gm

    class FakeGp:
        @staticmethod
        def recalc(ut):
            return 0.25

        @staticmethod
        def geogsm(x, y, z, _direction):
            return x, y, z

    class FakeT89:
        @staticmethod
        def t89(iopt, ps, x, y, z):
            return np.array([x + iopt, y + ps, z])

    class FakeT96:
        @staticmethod
        def t96(parmod, ps, x, y, z):
            return np.array([parmod[0] + x, parmod[1] + y, parmod[2] + z])

    monkeypatch.setattr(gm, "_HAS_GEOPACK", True)
    monkeypatch.setattr(gm, "_gp", FakeGp, raising=False)
    monkeypatch.setattr(gm, "_t89", FakeT89, raising=False)
    monkeypatch.setattr(gm, "_t96", FakeT96, raising=False)
    monkeypatch.setattr(gm, "_texture_cache_dir", lambda: tmp_path)
    gm._T89_CACHE.clear()

    matrix = gm._geo_to_gsm_matrix(0.0)
    np.testing.assert_allclose(matrix, np.eye(3))

    grid = gm._T89Grid(0.0, kp=2, x_min=2.0, x_max=3.0, half_yz=0.5, step=0.5)
    field = grid(np.array([[2.5 * 6371.2, 0.0, 0.0]]))
    assert field.shape == (1, 3)
    np.testing.assert_allclose(field[0], [5.5, 0.25, 0.0], atol=1e-6)

    class TinyGrid:
        def __init__(self, ut, kp=2, step=0.5, model="t89", parmod=None):
            self.ps = 0.0
            self.iopt = 1
            self.kp = kp
            self.M = np.eye(3)
            self.x = self.y = self.z = np.array([0.0, 1.0])
            self.B = np.zeros((2, 2, 2, 3))
            self.step = step
            self.model = model
            self.parmod = parmod

        def __call__(self, positions):
            return np.zeros_like(np.asarray(positions, dtype=float))

    monkeypatch.setattr(gm, "_T89Grid", TinyGrid)
    date = datetime(2025, 1, 1)
    external = gm._get_external(date, kp=3, step=2.0, model="t96", solar_wind={"dp_nPa": 2.0, "dst": -5.0, "by_nT": 1.0, "bz_nT": -2.0})
    assert isinstance(external, TinyGrid)
    np.testing.assert_allclose(external.parmod[:4], [2.0, -5.0, 1.0, -2.0])
    assert gm._get_external(date, kp=3, step=2.0, model="t96", solar_wind={"dp_nPa": 2.0, "dst": -5.0, "by_nT": 1.0, "bz_nT": -2.0}) is external
    assert isinstance(gm._get_external(date, kp=1, step=2.0, model="t89"), TinyGrid)


def test_geomagnetic_aep8_builder_with_fake_spacepy(monkeypatch, tmp_path):
    from ssapy_toolkit import geomagnetics as gm

    package = types.ModuleType("spacepy")
    irbempy = types.ModuleType("spacepy.irbempy")
    irbempy.get_AEP8 = lambda energy, values, model="max", fluxtype="int", particles="p": energy + values[0] + values[1] + (10.0 if particles == "p" else 20.0)
    package.irbempy = irbempy
    monkeypatch.setitem(sys.modules, "spacepy", package)
    monkeypatch.setitem(sys.modules, "spacepy.irbempy", irbempy)
    monkeypatch.setattr(gm, "_texture_cache_dir", lambda: tmp_path)
    gm._AEP8_TABLE = None

    table = gm._build_aep8_table(L_step=7.0, n_bb=2, energies=(10.0, 1.0), model="max")
    assert table is not None
    assert table["p"].shape == (2, 2)
    assert table["e"].shape == (2, 2)
    assert gm._aep8_table_path().exists()


def test_plot_layer_private_helpers_with_synthetic_geometry(monkeypatch):
    from ssapy_toolkit.plots import layers

    precession = layers._precession_matrix(2_451_545.0)
    np.testing.assert_allclose(precession, np.eye(3), atol=1e-12)

    class DummyLayer(layers.BaseLayer):
        def add_to_mpl(self, ax, orbit_state, traj=None, satellite=None, **kw):
            return []

        def add_to_plotly(self, fig, orbit_state, traj=None, satellite=None, **kw):
            return None

    dummy = DummyLayer("audit", "Audit")
    assert "audit" in repr(dummy)
    dummy.enabled = False
    assert "off" in repr(dummy)

    moon_layer = layers.MoonLayer(n_pts=6)
    moon_pos = moon_layer._moon_position_km(0.0)
    assert moon_pos.shape == (3,)
    assert np.linalg.norm(moon_pos) > 100_000.0

    sun_layer = layers.OrbitSunLayer()
    sun_pos = sun_layer._sun_pos(0.0, scene_r=10_000.0)
    assert sun_pos.shape == (3,)
    assert np.linalg.norm(sun_pos) > 10_000.0

    import ssapy.compute

    monkeypatch.setattr(ssapy.compute, "sunPos", lambda t: np.tile([[1.0], [0.0], [0.0]], (1, len(np.atleast_1d(t)))))
    monkeypatch.setattr(ssapy.compute, "earthShadowCoords", lambda r, sun: np.array([[-1.0, 1.0], [0.0, layers.RE_KM * 2e3]]))
    traj = SimpleNamespace(r=np.array([[-7000.0, 0.0, 0.0], [7000.0, 0.0, 0.0]]), t=np.array([0.0, 60.0]))
    mask = layers.EclipseLayer()._eclipse_mask(traj, orbit_state=None)
    np.testing.assert_array_equal(mask, [True, False])

    axis = layers.MagfieldLayer()._dipole_axis()
    assert np.linalg.norm(axis) == pytest.approx(1.0)

    class FakePpigrf:
        @staticmethod
        def igrf(lon, lat, alt_km, date):
            return 0.0, 0.0, 1.0

    monkeypatch.setattr(layers, "_HAS_PPIGRF", True)
    monkeypatch.setattr(layers, "ppigrf", FakePpigrf, raising=False)
    lines = layers.MagfieldLayer(seed_lats=[80], max_r_re=1.021)._trace_lines()
    assert isinstance(lines, list)


def test_lagrange_plot_helpers_and_orbit_3d_core(monkeypatch):
    core = importlib.import_module("ssapy_toolkit.plots._orbit_plot_core")
    cislunar_core = importlib.import_module("ssapy_toolkit.plots._cislunar_plot_core")

    monkeypatch.setattr(core, "_get_body", lambda name: SimpleNamespace(position=lambda t: np.zeros((3, len(np.atleast_1d(t))))))

    lagrange = core._lagrange_points_lunar_frame()
    lagrange_fixed = cislunar_core._lagrange_points_lunar_fixed_frame()
    assert {"L1", "L2", "L3", "L4", "L5"} <= set(lagrange)
    assert {"L1", "L2", "L3", "L4", "L5"} <= set(lagrange_fixed)
    assert all(np.asarray(value).shape == (3,) for value in lagrange.values())
    assert all(np.asarray(value).shape == (3,) for value in lagrange_fixed.values())

    track = np.array(
        [
            [7_000_000.0, 0.0, 0.0],
            [0.0, 7_000_000.0, 1_000_000.0],
            [-7_000_000.0, 0.0, -1_000_000.0],
        ]
    )
    fig, axes = core._orbit_plot_core(track, t=np.arange(3.0), views=("3d",), show=False)
    assert len(axes) == 1
    assert axes[0].name == "3d"
    plt.close(fig)


def test_base_plot_iers_threads_and_animation_update(monkeypatch):
    from astropy.utils import iers

    from ssapy_toolkit.plots import base_plot
    from ssapy_toolkit.plots.orbit_state import OrbitalState

    monkeypatch.setattr(iers.IERS_B, "open", lambda *args, **kwargs: object())

    state = OrbitalState(a_km=7000.0, e=0.001, inc_deg=5.0, epoch="2025-01-01T00:00:00+00:00")
    plot = base_plot.BasePlot3D(state)
    if plot._iers_thread is not None:
        plot._iers_thread.join(timeout=5.0)
    assert plot.fidelity in {"high", "fast"}

    seen = []
    plot.on_fidelity_change(seen.append)
    plot._start_iers_thread()
    plot._iers_thread.join(timeout=5.0)
    assert not plot._iers_thread.is_alive()
    assert seen[0] == "loading"
    assert seen[-1] in {"high", "fast"}

    animation = plot.animate(n_orbits=0.001, dt_s=10.0, interval_ms=1, trail_pts=2)
    artists = animation._func(0)
    assert len(artists) == 2
    plt.close(plot.fig)

    scene = base_plot.PlotlyScene(state)
    if scene._iers_thread is not None:
        scene._iers_thread.join(timeout=5.0)
    scene_seen = []
    scene.on_fidelity_change(scene_seen.append)
    scene._start_iers_thread()
    scene._iers_thread.join(timeout=5.0)
    assert not scene._iers_thread.is_alive()
    assert scene_seen[0] == "loading"
    assert scene_seen[-1] == "high"


def test_find_and_plot_eclipse_uses_flat_zone_separation(monkeypatch):
    espace = importlib.import_module("ssapy_toolkit.plots.eclipse_space_view_plotly")

    t_s = np.linspace(-4.0, 4.0, 17) * 3600.0
    r_moon = np.tile([384_400.0, 0.0, 0.0], (t_s.size, 1))
    r_moon[:, 1] = np.linspace(-1200.0, 1200.0, t_s.size)
    sun_hat = np.tile([-1.0, 0.0, 0.0], (t_s.size, 1))
    illum = np.array(
        [1.0, 0.99, 0.85, 0.55, 0.25, 0.05, 0.0, 0.0, 0.0, 0.0, 0.05, 0.25, 0.55, 0.85, 0.99, 1.0, 1.0]
    )
    window = {
        "t_s": t_s,
        "r_moon": r_moon,
        "sun_hat": sun_hat,
        "illum": illum,
        "peak_idx": int(np.argmin(np.abs(t_s))),
        "event_key": "synthetic-total-lunar",
        "event_label": "Synthetic total lunar eclipse",
        "event_source": "unit-test",
        "peak_utc": "2025-01-01T00:00:00",
    }

    monkeypatch.setattr(espace, "_eclipse_window", lambda *args, **kwargs: window)
    monkeypatch.setattr(espace, "render_lunar_panel", lambda offset: np.zeros((6, 6, 4), dtype=float))

    fig, stats = espace.find_and_plot_eclipse(mode="lunar", verbose=False)
    plt.close(fig)

    assert stats["event_key"] == "synthetic-total-lunar"
    assert stats["min_illum"] == pytest.approx(0.0)
    assert stats["eclipse_type"] == "Total"

    solar_window = {
        **window,
        "sun_hat": np.tile([1.0, 0.0, 0.0], (t_s.size, 1)),
        "event_key": "synthetic-total-solar",
        "event_label": "Synthetic total solar eclipse",
    }
    monkeypatch.setattr(espace, "_eclipse_window", lambda *args, **kwargs: solar_window)
    monkeypatch.setattr(espace, "render_solar_panel", lambda offset, corona=False: np.zeros((6, 6, 4), dtype=float))

    fig, stats = espace.find_and_plot_eclipse(mode="solar", verbose=False)
    plt.close(fig)

    assert stats["event_key"] == "synthetic-total-solar"
    assert stats["min_illum"] == pytest.approx(0.0)
    assert stats["eclipse_type"] == "Total"


def test_plot_private_mesh_and_math_helpers(monkeypatch, tmp_path):
    from PIL import Image

    from ssapy_toolkit.constants import EARTH_MU
    from ssapy_toolkit.plots import (
        earth_sun_plot,
        globe_orbit_daynight_plotly,
        groundtrack_enhanced,
        magnetosphere_core,
        moon_render,
        sensor_fov_plot,
        solar_bodies,
        sun_mpl,
        sun_view,
        van_allen_plot_3d,
    )
    moon_plot_3d = importlib.import_module("ssapy_toolkit.plots.moon_plot_3d")
    transfer_trajectory_plot = importlib.import_module("ssapy_toolkit.plots.transfer_trajectory_plot")

    fig = SimpleNamespace(
        data=[{"x": np.arange(10, dtype=np.float64), "label": "keep"}],
        frames=[SimpleNamespace(data=[{"x": np.arange(12, dtype=np.float64)}])],
    )
    shrunk = earth_sun_plot._shrink_floats(fig)
    assert shrunk.data[0]["x"].dtype == np.float32
    assert shrunk.frames[0].data[0]["x"].dtype == np.float32
    assert shrunk.data[0]["label"] == "keep"

    continents = globe_orbit_daynight_plotly._procedural_continents(8, 16, seed=2)
    assert continents.shape == (8, 16, 3)
    assert continents.dtype == np.uint8
    assert np.all((0 <= continents) & (continents <= 255))

    monkeypatch.setattr(groundtrack_enhanced, "_HAS_SSAPY_ASSETS", False)
    groundtrack_enhanced._earth_texture_cache = None
    assert groundtrack_enhanced._load_earth_texture() is None
    monkeypatch.setattr(groundtrack_enhanced, "_load_earth_texture", lambda: np.zeros((4, 8, 3), dtype=np.uint8))
    mpl_fig, ax = plt.subplots()
    assert groundtrack_enhanced._draw_continents(ax) == "ssapy earth.png"
    plt.close(mpl_fig)

    monkeypatch.setattr(magnetosphere_core, "starfield_traces", lambda sky_radius, when=None, frame="ecef", mag_limit=6.5: [go.Scatter3d(x=[sky_radius], y=[0], z=[0], name=frame)])
    star_traces = magnetosphere_core._build_starfield_trace(100.0, date=datetime(2025, 1, 1), mag_limit=5.0)
    assert star_traces[0].name == "ecef"
    texture = tmp_path / "earth_texture.jpg"
    texture.write_bytes(b"0" * 60_000)
    monkeypatch.setattr(magnetosphere_core, "_texture_cache_dir", lambda: tmp_path)
    assert magnetosphere_core._download_earth_texture() == texture
    shell = magnetosphere_core._ellipsoid_shell(10.0, n_lon=5, n_lat=4)
    assert len(shell[0]) == 20
    atmosphere = magnetosphere_core._atmosphere_traces(n_shells=2, top_km=10.0)
    assert len(atmosphere) == 2
    camera = magnetosphere_core._mpl_to_plotly_camera(0.0, 0.0, dist=2.0)
    assert camera == {"x": 2.0, "y": 0.0, "z": 0.0}
    rho, z, core = magnetosphere_core._lshell_cross_section(1.2, 2.0, 30.0, 5)
    assert rho.shape == z.shape == core.shape == (10,)

    assert moon_plot_3d._ssapy_texture("definitely_missing_texture_name") is None
    image_path = tmp_path / "earth.png"
    Image.new("RGB", (8, 4), "blue").save(image_path)
    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111, projection="3d")
    assert moon_plot_3d._textured_earth(ax, 0.0, 0.0, 0.0, 1.0, image_path, n=4, sun_hat=[1.0, 0.0, 0.0])
    assert ax.collections
    plt.close(mpl_fig)

    lat = np.linspace(-90.0, 90.0, 6)[:, None] * np.ones((6, 8))
    lon = np.ones((6, 1)) * np.linspace(-180.0, 180.0, 8)[None, :]
    albedo = moon_render._procedural_moon_albedo(lat, lon, seed=1)
    assert albedo.shape == lat.shape
    assert np.all((0.30 <= albedo) & (albedo <= 1.0))

    marker = sensor_fov_plot._satellite_marker(np.array([1.0, 2.0, 3.0]))
    assert marker.name == "Satellite"
    assert np.linalg.norm(sensor_fov_plot._default_sun_direction_gcrf()) == pytest.approx(1.0)

    earth_blob, colorscale = solar_bodies._color_earth_blobs(8)
    assert earth_blob.shape == (8, 8)
    assert colorscale

    sun_pos = sun_mpl._sun_position_fallback(Time("2025-01-01", scale="utc"))
    assert sun_pos.shape == (3,)
    assert 1.4e11 < np.linalg.norm(sun_pos) < 1.6e11

    x = np.array([[1.0, -1.0]])
    y = np.zeros_like(x)
    z_arr = np.zeros_like(x)
    shadow = sun_view._shadow_surface_color(x, y, z_arr, np.array([1.0, 0.0, 0.0]), R=1.0)
    assert shadow[0, 0] < shadow[0, 1]

    r = np.array([7000e3, 0.0, 0.0])
    v = np.array([0.0, np.sqrt(EARTH_MU / np.linalg.norm(r)), 0.0])
    ring = transfer_trajectory_plot._orbit_ring(r, v, n=8)
    assert ring.shape == (8, 3)
    np.testing.assert_allclose(np.linalg.norm(ring, axis=1), np.linalg.norm(r), rtol=5e-3)

    ti, tj, tk = van_allen_plot_3d._revolve_faces(3, 4)
    assert len(ti) == len(tj) == len(tk) == 24
    backend = van_allen_plot_3d._igrf_belt_backend()
    assert backend is None or callable(backend)


def test_io_orbit_and_transfer_private_helpers(tmp_path, monkeypatch):
    import h5py

    from ssapy_toolkit.orbit_initializer import OrbitInitialize
    from ssapy_toolkit.plots.orbit_state import OrbitalState, PropagatorConfig
    ssatk_save = importlib.import_module("ssapy_toolkit.io.ssatk_save")
    transfer_bielliptic = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_bielliptic")
    transfer_optimal_function = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_optimal_function")
    tvic = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_velocity_and_inclination_continuous")

    h5_path = tmp_path / "audit.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("group/value", data=[1, 2, 3])
    assert ssatk_save._hdf5_key_exists(h5_path, ["group", "value"])
    ssatk_save._delete_hdf5_key(h5_path, ["group", "value"])
    assert not ssatk_save._hdf5_key_exists(h5_path, ["group", "value"])

    assert ssatk_save._is_scalar(np.float64(1.25))
    assert ssatk_save._json_default(np.array([1, 2])) == [1, 2]
    assert ssatk_save._json_default(np.int64(3)) == 3
    with pytest.raises(TypeError, match="not JSON serializable"):
        ssatk_save._json_default({object()})

    assert isinstance(OrbitInitialize(), OrbitInitialize)

    tangent = transfer_bielliptic._default_tangent(np.array([1.0, 0.0, 0.0]))
    assert np.linalg.norm(tangent) == pytest.approx(1.0)
    np.testing.assert_allclose(np.dot(tangent, [1.0, 0.0, 0.0]), 0.0, atol=1e-12)
    pole_tangent = transfer_bielliptic._default_tangent(np.array([0.0, 0.0, 1.0]))
    assert np.linalg.norm(pole_tangent) == pytest.approx(1.0)

    initial = {"r": np.array([7000e3, 0.0, 0.0])}
    target = {"r": np.array([9000e3, 0.0, 0.0])}
    radii = transfer_optimal_function._default_stage_radii(initial, target)
    assert radii == sorted(radii)
    assert all(radius > 0.0 for radius in radii)
    assert any(7000e3 < radius < 9000e3 for radius in radii)

    state = OrbitalState(
        a_km=7000.0,
        e=0.001,
        inc_deg=10.0,
        epoch="2025-01-01T00:00:00+00:00",
        config=PropagatorConfig(propagator="keplerian", gravity="point_mass"),
    )
    propagator, accels = state._build_propagator()
    assert propagator is not None
    assert accels == []
    assert "OrbitalState" in repr(state)

    monkeypatch.setattr(tvic.plt, "show", lambda: None)
    mu = 3.986004418e14
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(mu / np.linalg.norm(r0)), 0.0])
    r_full = np.array([r0, [0.0, 7000e3, 0.0]])
    v_full = np.array([v0, [-np.sqrt(mu / np.linalg.norm(r0)), 0.0, 0.0]])
    tvic._plot_transfer(r0, v0, r_full, v_full, np.array([0.0, 60.0]), 0.0, 60.0, mu, 6378e3)
    plt.close("all")


def test_magfield_plot_private_helpers_with_synthetic_models(monkeypatch, tmp_path):
    from ssapy_toolkit.plots import magfield_plot_3d as mf

    date = datetime(2025, 1, 1)
    axis = np.array([0.0, 0.0, 1.0])

    monkeypatch.setattr(
        mf,
        "_bfield_batch",
        lambda positions, _date: np.tile([100.0, 0.0, 0.0], (len(positions), 1)),
    )
    monkeypatch.setattr(
        mf,
        "_trace_batch_rk4",
        lambda seeds, _date, direction=1, **_kw: [
            np.array([
                seed - direction * np.array([0.0, 0.0, 10.0]),
                seed,
                seed + direction * np.array([0.0, 0.0, 10.0]),
            ])
            for seed in np.asarray(seeds, dtype=float)
        ],
    )
    boundary = mf._igrf_lshell_boundary(2.0, date, axis, n_azim=3, n_pts=4)
    assert boundary.shape == (3, 4, 3)

    mesh = mf._igrf_belt_mesh(1.5, 2.0, date, axis, base_rgb=(0.8, 0.2, 0.1), n_azim=3, n_pts=4)
    xs, ys, zs, ti, tj, tk, colors = mesh
    assert len(xs) == len(ys) == len(zs) == len(colors) == 3 * 8
    assert len(ti) == len(tj) == len(tk)

    monkeypatch.setattr(mf, "_HAS_PPIGRF", True)
    monkeypatch.setattr(mf, "_load_aep8_table", lambda: {"available": True})
    monkeypatch.setattr(mf, "_texture_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(mf, "_physics_fingerprint", lambda: "audit")
    monkeypatch.setattr(mf._geo, "get_external_model", lambda: None)
    monkeypatch.setattr(
        mf,
        "_belt_flux_samples",
        lambda _date, _axis, **_kw: (
            np.array([[0.0, 0.0, 0.0]]),
            np.array([10.0]),
            np.array([20.0]),
        ),
    )
    flux_traces = mf._flux_isosurfaces(date, axis, grid_n=3, extent_re=0.01, levels=(0.5,), cache=False)
    assert len(flux_traces) == 2
    assert all(trace.type == "isosurface" for trace in flux_traces)

    assert len(mf._orbit_rings(date, n=8, show_labels=True)) == 6
    assert len(mf._orbit_rings(date, n=8, show_labels=False)) == 3
    monkeypatch.setattr(mf, "_sun_direction_geo", lambda _date: np.array([1.0, 0.0, 0.0]))
    sun_traces = mf._sun_marker(date, length_km=1000.0)
    assert len(sun_traces) == 2
    assert sun_traces[0].x[-1] == pytest.approx(1000.0)

    monkeypatch.setattr(mf, "_pp", SimpleNamespace(shc_fn="/tmp/igrf_2025.shc"), raising=False)
    monkeypatch.setattr(mf, "_HAS_GEOPACK", True)
    provenance = mf._model_provenance(
        date,
        external_field="t96",
        kp=4,
        belt_style="flux",
        sw_bz_nT=-2.5,
        sw_dp_nPa=3.0,
        sw_source="synthetic",
    )
    assert "IGRF 2025" in provenance
    assert "T96" in provenance
    assert "AE-8" in provenance


def test_eclipse_space_private_helpers_and_nested_animation_paths(monkeypatch):
    from ssapy_toolkit.plots import eclipse_space_view_plotly as espace

    monkeypatch.setattr(
        espace,
        "propagate_eci",
        lambda **kwargs: (
            np.linspace(0.0, 3600.0, max(3, int(kwargs.get("n_steps", 3)))),
            np.column_stack([
                np.linspace(espace.D_MOON_A_KM, espace.D_MOON_A_KM + 10.0, max(3, int(kwargs.get("n_steps", 3)))),
                np.zeros(max(3, int(kwargs.get("n_steps", 3)))),
                np.zeros(max(3, int(kwargs.get("n_steps", 3)))),
            ]),
            1.0,
        ),
    )
    monkeypatch.setattr(espace, "sun_direction_eci", lambda t: np.tile([-1.0, 0.0, 0.0], (len(np.atleast_1d(t)), 1)))
    monkeypatch.setattr(espace, "illumination_fraction", lambda r, sun, **kwargs: np.linspace(1.0, 0.2, len(r)))
    window = espace._synthetic_eclipse_window("lunar", search_days=1.0, n_steps=5, verbose=False)
    assert window["mode"] == "lunar"
    assert window["peak_idx"] == len(window["illum"]) - 1

    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111, projection="3d")
    espace._sun_sphere_mpl(ax, np.array([100.0, 0.0, 0.0]), 5.0)
    espace._earth_sphere_mpl(ax, np.zeros(3), 5.0, np.array([1.0, 0.0, 0.0]))
    espace._moon_sphere_mpl(ax, np.array([20.0, 0.0, 0.0]), 2.0, np.array([0.8, 0.7, 0.6]))
    espace._plot_space_view_unified(ax, np.array([50.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 0.5, "lunar")
    assert ax.collections
    plt.close(mpl_fig)

    ray_traces = espace._light_ray_traces(
        np.array([-100.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        10.0,
        np.array([-50.0, 0.0, 0.0]),
        5.0,
    )
    assert len(ray_traces) == 2
    with pytest.raises(NotImplementedError, match="moon_render"):
        espace._moon_mesh_plotly_REMOVED_use_moon_render_instead()

    footprint = espace._shadow_footprint_traces(
        np.zeros(3),
        np.array([espace.RE_KM, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        100.0,
        500.0,
    )
    assert len(footprint) == 2
    path = espace._ground_latlon_path_to_eci([(0.0, 0.0), (None, 20.0), (10.0, 20.0)], Time("2025-01-01"))
    assert path.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(path, axis=1), espace.RE_KM, rtol=1e-12)

    def mesh(name):
        return go.Mesh3d(x=[0.0], y=[0.0], z=[0.0], i=[], j=[], k=[], name=name)

    times = Time(["2024-04-08T17:30:00", "2024-04-08T17:31:00"], scale="utc")
    monkeypatch.setattr(espace, "_time_grid_utc", lambda *args, **kwargs: times)
    monkeypatch.setattr(espace, "_real_solar_eclipse_geometry", lambda times, ephemeris="builtin": (
        np.tile([384_400.0, 0.0, 0.0], (len(times), 1)),
        np.tile([-1.0, 0.0, 0.0], (len(times), 1)),
    ))
    hit_calls = iter([None, np.array([espace.RE_KM, 0.0, 0.0])])
    monkeypatch.setattr(espace, "_shadow_ground_point", lambda *args, **kwargs: next(hit_calls))
    monkeypatch.setattr(espace, "_eci_surface_to_latlon", lambda point, time: (30.0, -98.0))
    monkeypatch.setattr(espace, "_latlon_to_eci_surface", lambda lat, lon, time, radius_scale=1.0, radius_km=espace.RE_KM: np.array([radius_km * radius_scale, 0.0, 0.0]))
    monkeypatch.setattr(espace, "_ground_latlon_path_to_eci", lambda latlon, time, radius_scale=1.0, radius_km=espace.RE_KM: np.tile([radius_km * radius_scale, 0.0, 0.0], (len(latlon), 1)))
    monkeypatch.setattr(espace, "_earth_mesh", lambda *args, **kwargs: mesh("Earth"))
    monkeypatch.setattr(espace, "_earth_atmosphere_trace", lambda *args, **kwargs: mesh("Atmosphere"))
    monkeypatch.setattr(espace, "moon_mesh_plotly", lambda *args, **kwargs: mesh("Moon"))
    monkeypatch.setattr(espace, "_shadow_cone_trace", lambda *args, **kwargs: mesh("Shadow"))
    monkeypatch.setattr(espace, "_sun_direction_arrow", lambda *args, **kwargs: [go.Scatter3d(x=[0, 1], y=[0, 0], z=[0, 0], name="Sun direction")])
    monkeypatch.setattr(espace, "_shadow_footprint_traces", lambda *args, **kwargs: [mesh("Penumbra"), mesh("Umbra")])
    monkeypatch.setattr(espace, "_starfield_trace", lambda *args, **kwargs: go.Scatter3d(x=[], y=[], z=[], name="Stars"))
    monkeypatch.setattr(espace, "_lunar_or_solar_camera_eye", lambda *args, **kwargs: dict(x=1.0, y=1.0, z=0.5))
    monkeypatch.setattr(espace, "earth_rotation_deg_from_time", lambda *args, **kwargs: 0.0)

    fig_2024, stats_2024 = espace.plot_2024_solar_eclipse_animated(
        n_frames=2,
        n_lat=4,
        n_lon=4,
        show_stars=False,
        verbose=False,
    )
    assert len(fig_2024.frames) == 2
    assert stats_2024["event"] == "2024-04-08 total solar eclipse"

    solar_window = {
        "mode": "solar",
        "event_label": "Synthetic solar eclipse",
        "t_s": np.array([-1000.0, -500.0, 0.0, 500.0, 1000.0]),
        "times": Time(
            [
                "2024-04-08T17:28:00",
                "2024-04-08T17:29:00",
                "2024-04-08T17:30:00",
                "2024-04-08T17:31:00",
                "2024-04-08T17:32:00",
            ],
            scale="utc",
        ),
        "r_moon": np.tile([384_400.0, 0.0, 0.0], (5, 1)),
        "sun_hat": np.tile([-1.0, 0.0, 0.0], (5, 1)),
        "illum": np.array([1.0, 0.8, 0.1, 0.8, 1.0]),
        "peak_idx": 2,
        "epoch_jd": 2_460_408.0,
    }
    monkeypatch.setattr(espace, "_eclipse_window", lambda *args, **kwargs: solar_window)
    monkeypatch.setattr(espace, "_shadow_ground_point", lambda *args, **kwargs: np.array([espace.RE_KM, 0.0, 0.0]))
    fig_anim = espace.plot_space_view_animated(
        mode="solar",
        n_frames=2,
        n_lat=4,
        n_lon=4,
        verbose=False,
        show_eclipse_path=True,
    )
    assert len(fig_anim.frames) == 2
    assert "Synthetic solar eclipse" in fig_anim.layout.title.text
