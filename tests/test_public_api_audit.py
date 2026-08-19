from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pytest
from astropy.time import Time


def test_equatorial_ecliptic_round_trips_radians_and_degrees():
    from ssapy_toolkit.coordinates import equatorial_and_ecliptic as eqecl

    ra_rad = np.deg2rad(132.5)
    dec_rad = np.deg2rad(-18.25)
    lon_rad, lat_rad = eqecl.equatorial_to_ecliptic(ra_rad, dec_rad)
    round_ra_rad, round_dec_rad = eqecl.ecliptic_to_equatorial(lon_rad, lat_rad)

    assert round_ra_rad == pytest.approx(ra_rad)
    assert round_dec_rad == pytest.approx(dec_rad)

    lon_deg, lat_deg = eqecl.equatorial_to_ecliptic(132.5, -18.25, degrees=True)
    round_ra_deg, round_dec_deg = eqecl.ecliptic_to_equatorial(lon_deg, lat_deg, degrees=True)

    assert round_ra_deg == pytest.approx(132.5)
    assert round_dec_deg == pytest.approx(-18.25)

    ecl_x, ecl_y, ecl_z = eqecl.equatorial_xyz_to_ecliptic_xyz(1.0, 2.0, 3.0)
    ra_from_xyz, dec_from_xyz = eqecl.ecliptic_xyz_to_equatorial(ecl_x, ecl_y, ecl_z)
    ra_direct, dec_direct = eqecl.xyz_to_equatorial(1.0, 2.0, 3.0)
    assert ra_from_xyz == pytest.approx(ra_direct)
    assert dec_from_xyz == pytest.approx(dec_direct)


def test_gcrf_to_itrf_astropy_is_geocentric_and_norm_preserving():
    from ssapy_toolkit.coordinates.gcrf_to_itrf import gcrf_to_itrf_astropy

    times = Time(["2025-01-01T00:00:00", "2025-01-01T00:10:00"], scale="utc")
    positions = np.array([[0.0, 0.0, 0.0], [6_378_137.0, 0.0, 0.0]])

    transformed = gcrf_to_itrf_astropy(positions, times)

    assert transformed.shape == (2, 3)
    np.testing.assert_allclose(transformed[0], [0.0, 0.0, 0.0], atol=1e-6)
    assert np.linalg.norm(transformed[1]) == pytest.approx(np.linalg.norm(positions[1]), rel=0, abs=1e-3)
    with pytest.raises(ValueError, match="shape"):
        gcrf_to_itrf_astropy(np.ones(3), times[0])


def test_frame_transform_conventions_match_ssapy_ntw_order():
    from ssapy_toolkit.coordinates.ntw_to_gcrf import ntw_to_gcrf_matrix
    from ssapy_toolkit.plots.frames import (
        Frame,
        FrameTransform,
        eci_to_ecf_matrix,
        eci_to_lon_lat,
        lvlh_axes,
        lvlh_matrix,
        ntw_axes,
        ntw_matrix,
    )

    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 7.5, 0.0])

    assert "velocity" in Frame.NTW.label
    rotation = eci_to_ecf_matrix(0.0)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)

    lvlh = lvlh_matrix(r, v)
    ntw = ntw_matrix(r, v)
    np.testing.assert_allclose(lvlh @ lvlh.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(ntw @ ntw.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(ntw, ntw_to_gcrf_matrix(r, v).T)

    T_hat, N_hat, W_hat = ntw_axes(r, v)
    R_hat, S_hat, W_lvlh = lvlh_axes(r, v)
    np.testing.assert_allclose(T_hat, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(N_hat, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(W_hat, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(R_hat, N_hat)
    np.testing.assert_allclose(S_hat, T_hat)
    np.testing.assert_allclose(W_lvlh, W_hat)

    ntw_tf = FrameTransform(Frame.NTW)
    np.testing.assert_allclose(ntw_tf.transform_point(r, v), [7000.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(ntw_tf.transform_vector(v, r, v), [0.0, 7.5, 0.0], atol=1e-12)

    r_series = np.array([r, [0.0, 7000.0, 0.0]])
    v_series = np.array([v, [-7.5, 0.0, 0.0]])
    transformed = ntw_tf.transform_trajectory(r_series, v_series)
    np.testing.assert_allclose(transformed[:, 0], [7000.0, 7000.0], atol=1e-12)
    relative = ntw_tf.relative_trajectory(r_series, v_series)
    np.testing.assert_allclose(relative[0], [0.0, 0.0, 0.0], atol=1e-12)

    with pytest.raises(ValueError, match="t_gps required"):
        FrameTransform(Frame.ECF).transform_trajectory(r_series, v_series)
    lon, lat = eci_to_lon_lat(np.array([[7000.0, 0.0, 0.0]]), np.array([0.0]))
    assert lon.shape == lat.shape == (1,)
    assert np.isfinite(lon[0]) and np.isfinite(lat[0])


def test_satellite_burns_use_canonical_ntw_components(tmp_path):
    from ssapy_toolkit.plots.orbit_state import OrbitalState, Trajectory
    from ssapy_toolkit.plots.satellite import BurnEvent, Satellite3D

    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 7.5, 0.0])
    burn = BurnEvent(epoch_offset_s=10.0, dv_ntw_km_s=[0.0, 0.02, 0.0])

    np.testing.assert_allclose(burn.dv_eci(r, v), [0.0, 0.02, 0.0], atol=1e-12)
    assert burn.dv_mag_m_s == pytest.approx(20.0)
    assert burn.dv_mag_km_s == pytest.approx(0.02)
    assert burn.burn_duration_s() == 0.0

    finite = BurnEvent(
        epoch_offset_s=20.0,
        dv_ntw_km_s=[0.0, 0.01, 0.0],
        mode="finite",
        thrust_N=10.0,
        isp_s=300.0,
        mass_kg=100.0,
    )
    assert finite.burn_duration_s() > 0.0

    sat = Satellite3D(mass_kg=100.0)
    sat.add_burn(finite)
    sat.add_burn(burn)
    assert [item.epoch_offset_s for item in sat.burns] == [10.0, 20.0]
    assert sat.total_delta_v_m_s() == pytest.approx(30.0)

    T, N, W = sat.ntw_vectors(r, v, scale=2.0)
    R, S, W_lvlh = sat.lvlh_vectors(r, v, scale=3.0)
    np.testing.assert_allclose(T, [0.0, 2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(N, [2.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(W, [0.0, 0.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(R, [3.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(S, [0.0, 3.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(W_lvlh, [0.0, 0.0, 3.0], atol=1e-12)
    assert np.linalg.norm(sat.burn_vector_eci(burn, r, v)) == pytest.approx(sat.ntw_scale * 0.2)
    sat.remove_burn(0)
    assert sat.burns == [finite]
    sat.add_burn(burn)

    obj_path = tmp_path / "cube.obj"
    obj_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    obj_sat = Satellite3D(model_path=obj_path)
    assert obj_sat.load_obj()
    assert obj_sat.faces == [[0, 1, 2]]
    vertices = obj_sat.model_vertices_eci(r, v, scale_km=1.0)
    assert vertices.shape == (3, 3)

    empty_sat = Satellite3D()
    assert empty_sat.model_vertices_eci(r, v, scale_km=1.0).shape == (8, 3)
    assert len(empty_sat.faces) == 6
    assert not Satellite3D(model_path=tmp_path / "missing.obj").load_obj()

    trajectory = Trajectory(
        r=np.array([r, r + np.array([0.0, 75.0, 0.0])]),
        v=np.array([v, v]),
        t=np.array([0.0, 10.0]),
    )
    state = OrbitalState.from_rv(r, v, epoch="2025-01-01T00:00:00+00:00", name="audit")
    results = sat.apply_burns_to_trajectory(trajectory, state)
    assert len(results) == 2
    assert results[0][1] == 1
    assert results[0][2] is burn


def test_orbital_state_public_quantities_have_physical_oracles():
    from ssapy_toolkit.plots.orbit_state import MU, OrbitalState, PropagatorConfig

    state = OrbitalState(
        a_km=7000.0,
        e=0.001,
        inc_deg=45.0,
        raan_deg=10.0,
        argp_deg=20.0,
        nu_deg=30.0,
        epoch="2025-01-01T00:00:00+00:00",
        name="audit",
    )

    assert PropagatorConfig(propagator="rk4", gravity="j2", third_body="moon", non_grav="drag").label() == "RK4 + j2 + moon + drag"
    assert state.period_s == pytest.approx(2.0 * np.pi * np.sqrt(state.a_km**3 / MU))
    assert state.v_p > state.v_a
    assert state.v_circ == pytest.approx(np.sqrt(MU / state.a_km))
    assert state.specific_angular_momentum == pytest.approx(np.sqrt(MU * state.a_km * (1.0 - state.e**2)))
    assert state.regime == "LEO"
    assert state.j2_raan_drift_deg_day < 0.0
    assert np.isfinite(state.j2_argp_drift_deg_day)
    assert state.warnings == []

    r, v = state.to_rv()
    roundtrip = OrbitalState.from_rv(r, v, epoch=state.epoch)
    assert roundtrip.a_km == pytest.approx(state.a_km)
    assert roundtrip.e == pytest.approx(state.e)
    assert roundtrip.inc_deg == pytest.approx(state.inc_deg)
    assert state.osculating_ellipse(n_pts=12).shape == (12, 3)

    clone = state.clone(e=0.01, name="clone")
    assert clone.e == pytest.approx(0.01)
    assert clone.name == "clone"

    state.set_elements(a_km=7100.0)
    assert state.a_km == pytest.approx(7100.0)

    bad = OrbitalState(a_km=6000.0, e=-0.1, inc_deg=190.0)
    assert any("Negative eccentricity" in warning for warning in bad.warnings)
    assert any("Inclination" in warning for warning in bad.warnings)
    assert not bad.propagate(n_orbits=0.01, dt_s=10.0).ok

    callback_results = []
    traj = clone.propagate(n_orbits=0.01, dt_s=10.0, callback=callback_results.append)
    assert traj.ok
    assert callback_results == [traj]
    done = []
    thread, stop = clone.propagate_async(n_orbits=0.001, dt_s=10.0, on_done=done.append)
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert not stop.is_set()
    assert done and done[0].ok

    with pytest.warns(UserWarning, match="legacy alias"):
        alias = OrbitalState.from_preset("Cislunar L1 Halo")
    assert alias.name == "Orbit"
    with pytest.raises(KeyError):
        OrbitalState.from_preset("not a preset")


def test_orbital_state_tle_and_ssapy_roundtrip():
    pytest.importorskip("ssapy")
    from ssapy_toolkit.plots.orbit_state import OrbitalState

    tle = """ISS (ZARYA)
1 25544U 98067A   25001.00000000  .00016717  00000+0  10270-3 0  9000
2 25544  51.6400 120.0000 0007000  90.0000  10.0000 15.50000000 00001
"""
    state = OrbitalState.from_tle(tle, name="ISS audit")
    assert state.name == "ISS audit"
    assert 6500.0 < state.a_km < 7000.0
    assert state.e == pytest.approx(0.0007)
    assert state.inc_deg == pytest.approx(51.64)

    ssapy_orbit = state.to_ssapy()
    converted = OrbitalState.from_ssapy(ssapy_orbit, name="converted")
    assert converted.name == "converted"
    assert converted.a_km == pytest.approx(state.a_km, rel=1e-6)
    assert converted.e == pytest.approx(state.e, abs=1e-8)


def test_sun_geometry_helpers_have_expected_directions_and_scaling():
    from ssapy_toolkit.constants import SUN_EARTH_AVERAGE_DISTANCE_KM, SUN_RADIUS_KM
    from ssapy_toolkit.plots import sun_mpl, sun_render, sun_view

    PHI, THETA = np.meshgrid(np.linspace(0.0, np.pi, 5), np.linspace(0.0, 2.0 * np.pi, 7))
    lit = sun_mpl.shade_texture(PHI, THETA, [0.0, 0.0, 1.0], ambient=0.2, diffuse=0.6)
    assert lit.min() >= 0.2
    assert lit.max() <= 0.8
    assert lit[0, 0] == pytest.approx(0.8)

    image = np.ones((*PHI.shape, 3))
    rows, cols = np.indices(PHI.shape)
    shaded = sun_mpl.apply_shading(image, rows, cols, PHI, THETA, [0.0, 0.0, 1.0])
    assert shaded.shape == image.shape
    assert np.all((0.0 <= shaded) & (shaded <= 1.0))

    assert sun_mpl.auto_sun_distance(100.0) == pytest.approx(42.0)
    assert sun_mpl.auto_sun_radius(100.0) == pytest.approx(4.5)
    np.testing.assert_allclose(
        sun_render.light_direction_from_positions([2.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        [1.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        sun_render.light_direction_from_positions([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        [1.0, 0.0, 0.0],
    )
    sun_pos = sun_render.background_sun_position([0.0, 1.0, 0.0], 100.0, distance_factor=3.0)
    np.testing.assert_allclose(sun_pos, [0.0, 300.0, 0.0])
    assert sun_render.background_sun_radius(100.0, distance_factor=3.0) == pytest.approx(
        max(100.0 * 3.0 * SUN_RADIUS_KM / SUN_EARTH_AVERAGE_DISTANCE_KM, 1.0)
    )
    assert sun_render.background_sun_radius(100.0, size_factor=0.2) == pytest.approx(20.0)
    assert sun_view.auto_sun_position(np.array([1.0, 0.0, 0.0]), 100.0).shape == (3,)
    assert sun_view.auto_sun_radius(100.0) >= 1.0

    import datetime

    assert sun_view.jd_from_datetime(datetime.datetime(2000, 1, 1, 12, 0, 0)) == pytest.approx(2_451_545.0)
    with pytest.raises(TypeError, match="datetime"):
        sun_view.jd_from_datetime("2000-01-01")

    sun_position = sun_mpl.get_sun_position(Time([0.0], format="gps"))
    assert np.atleast_2d(sun_position).shape[-1] == 3
    sun_hat = sun_mpl.sun_direction_in_frame(
        Time([0.0, 60.0], format="gps"),
        transform_func=lambda pos, t: pos,
    )
    assert np.linalg.norm(sun_hat) == pytest.approx(1.0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sun_mpl.draw_sun(ax, [10.0, 0.0, 0.0], radius=1.0, n=8)
    sun_render.render_sun(ax, [0.0, 10.0, 0.0], radius=1.0, n=8, corona_layers_n=2, label="")
    assert ax.collections
    plt.close(fig)


def test_eclipse_and_solar_body_lightweight_helpers():
    from ssapy_toolkit.plots import eclipse_brightness_plot as ebp
    from ssapy_toolkit.plots import eclipse_space_view_plotly as espace
    from ssapy_toolkit.plots.solar_bodies import make_moon_trace

    times, positions, period = ebp.propagate_eci(
        a_km=7000.0,
        e=0.001,
        inc_deg=10.0,
        raan_deg=20.0,
        argp_deg=30.0,
        nu0_deg=40.0,
        n_orbits=0.05,
        n_steps=16,
    )
    assert times.shape == (16,)
    assert positions.shape == (16, 3)
    assert period > 0.0
    assert np.all(np.isfinite(positions))

    full = espace.moon_color(1.0)
    eclipsed = espace.moon_color(0.0)
    assert full.shape == eclipsed.shape == (3,)
    assert np.mean(eclipsed) < np.mean(full)
    assert espace.moon_brightness(0.0) == pytest.approx(0.12)
    assert espace.moon_brightness(1.0) == pytest.approx(1.0)
    np.testing.assert_allclose(espace.moon_red_bias(1.0), np.ones(3))
    assert np.all(espace.moon_red_bias(0.0) > 0.0)

    moon = make_moon_trace((1.0, 0.0, 0.0), t_jd=2_451_545.0)
    assert moon.name == "Moon"
    assert len(moon.x) == len(moon.y) == len(moon.z) == 1


def test_eclipse_space_view_public_entrypoints_with_lightweight_geometry(monkeypatch):
    from ssapy_toolkit.plots import eclipse_space_view_plotly as espace

    def mesh(name):
        return go.Mesh3d(x=[0.0], y=[0.0], z=[0.0], i=[], j=[], k=[], name=name)

    monkeypatch.setattr(espace, "_real_solar_eclipse_geometry", lambda times, ephemeris="builtin": (
        np.tile([384_400.0, 0.0, 0.0], (len(times), 1)),
        np.tile([-1.0, 0.0, 0.0], (len(times), 1)),
    ))
    monkeypatch.setattr(espace, "_shadow_ground_point", lambda *args, **kwargs: np.array([espace.RE_KM, 0.0, 0.0]))
    monkeypatch.setattr(espace, "_eci_surface_to_latlon", lambda point, time: (30.0, -98.0))
    monkeypatch.setattr(espace, "_latlon_to_eci_surface", lambda *args, **kwargs: np.array([espace.RE_KM, 0.0, 0.0]))
    monkeypatch.setattr(espace, "_ground_latlon_path_to_eci", lambda latlon, time, radius_scale=1.0, radius_km=None: np.array([[espace.RE_KM * radius_scale, 0.0, 0.0]] * len(latlon)))
    monkeypatch.setattr(espace, "_earth_mesh", lambda *args, **kwargs: mesh("Earth"))
    monkeypatch.setattr(espace, "_earth_atmosphere_trace", lambda *args, **kwargs: mesh("Atmosphere"))
    monkeypatch.setattr(espace, "moon_mesh_plotly", lambda *args, **kwargs: mesh("Moon"))
    monkeypatch.setattr(espace, "_shadow_cone_trace", lambda *args, **kwargs: mesh("Shadow cone"))
    monkeypatch.setattr(espace, "_shadow_footprint_traces", lambda *args, **kwargs: [mesh("Umbra"), mesh("Penumbra")])
    monkeypatch.setattr(espace, "_sun_direction_arrow", lambda *args, **kwargs: [go.Scatter3d(x=[0, 1], y=[0, 0], z=[0, 0], name="Sun direction")])
    monkeypatch.setattr(espace, "_starfield_trace", lambda *args, **kwargs: go.Scatter3d(x=[], y=[], z=[], name="Stars"))
    monkeypatch.setattr(espace, "_lunar_or_solar_camera_eye", lambda *args, **kwargs: dict(x=1.0, y=1.0, z=0.5))
    monkeypatch.setattr(espace, "earth_rotation_deg_from_time", lambda *args, **kwargs: 0.0)

    fig, stats = espace.plot_2024_solar_eclipse_animated(n_frames=2, n_lat=4, n_lon=4, show_stars=False, verbose=False)
    assert isinstance(fig, go.Figure)
    assert len(fig.frames) == 2
    assert stats["event"] == "2024-04-08 total solar eclipse"

    times = Time([0.0, 60.0], format="gps")
    window = {
        "t_s": np.array([0.0, 60.0]),
        "r_moon": np.tile([384_400.0, 0.0, 0.0], (2, 1)),
        "sun_hat": np.tile([-1.0, 0.0, 0.0], (2, 1)),
        "illum": np.array([1.0, 0.0]),
        "peak_idx": 1,
        "times": times,
    }
    monkeypatch.setattr(espace, "find_and_plot_eclipse", lambda **kwargs: (None, {"angle_at_peak_deg": 0.1}))
    monkeypatch.setattr(espace, "_eclipse_window", lambda *args, **kwargs: window)
    fig2, stats2 = espace.plot_space_view_plotly(mode="solar", verbose=False)
    assert isinstance(fig2, go.Figure)
    assert stats2["angle_at_peak_deg"] == pytest.approx(0.1)


def test_starfield_matplotlib_entrypoint_with_synthetic_catalog(monkeypatch):
    from ssapy_toolkit.plots import starfield

    stars = {
        "v": np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]),
        "mag": np.array([1.0, 2.0, 3.0]),
        "sizes": np.array([4.0, 3.0, 2.0]),
        "rgb": np.array([[1.0, 1.0, 1.0], [0.8, 0.8, 1.0], [1.0, 0.8, 0.6]]),
    }
    monkeypatch.setattr(starfield, "_load_stars", lambda **kwargs: stars)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    starfield.add_starfield(
        ax,
        plot_range=10.0,
        elev=0.0,
        azim=180.0,
        fov=360.0,
        show_milky_way=True,
        depth_variation=True,
    )
    assert ax.collections
    assert ax.lines
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    monkeypatch.setattr(starfield, "_load_stars", lambda **kwargs: None)
    assert starfield.add_starfield(ax, plot_range=10.0) is None
    plt.close(fig)


def test_accel_thrust_and_geomagnetic_state_factories():
    from ssapy_toolkit import geomagnetics
    from ssapy_toolkit.plots.accel_thrust import G0, AccelConstantThrust, burn_from_dv

    accel = AccelConstantThrust([2.0, 0.0, 0.0], thrust_n=20.0, isp_s=300.0, wet_mass_kg=100.0, t_start_gps=10.0, t_end_gps=20.0)
    np.testing.assert_allclose(accel(None, None, 5.0), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(accel(None, None, 10.0), [0.2, 0.0, 0.0])
    assert np.linalg.norm(accel(None, None, 20.0)) > 0.2
    assert accel == AccelConstantThrust([1.0, 0.0, 0.0], 20.0, 300.0, 100.0, 10.0, 20.0)
    assert accel != AccelConstantThrust([1.0, 0.0, 0.0], 30.0, 300.0, 100.0, 10.0, 20.0)
    assert isinstance(hash(accel), int)
    with pytest.raises(ValueError, match="non-zero"):
        AccelConstantThrust([0.0, 0.0, 0.0], 20.0, 300.0, 100.0, 10.0, 20.0)

    burn, duration, propellant = burn_from_dv(np.array([0.1, 0.0, 0.0]), 100.0, 20.0, 300.0, 10.0)
    assert burn is not None
    assert duration == pytest.approx(propellant / (20.0 / (300.0 * G0)))
    assert burn.time_breakpoints[1] == pytest.approx(10.0 + duration)
    assert burn_from_dv(np.zeros(3), 100.0, 20.0, 300.0, 10.0) == (None, 0.0, 0.0)

    previous = geomagnetics.set_external_model({"model": "audit"})
    assert geomagnetics.get_external_model() == {"model": "audit"}
    geomagnetics.set_external_model(previous)


def test_benchmark_dashboard_helpers_write_expected_artifacts(monkeypatch, tmp_path):
    from ssapy_toolkit import benchmark
    from ssapy_toolkit.benchmark import BenchmarkContext, BenchmarkResult

    monkeypatch.setattr(benchmark, "figpath", lambda path: str(tmp_path / path))
    assert benchmark.default_output_dir().is_dir()

    context = BenchmarkContext(output_dir=tmp_path, rng_seed=10)
    np.testing.assert_allclose(context.rng(5).normal(size=3), np.random.default_rng(15).normal(size=3))

    result = BenchmarkResult(
        name="audit.fast",
        group="audit",
        description="synthetic audit benchmark",
        tags=("unit",),
        success=True,
        repeats=3,
        warmups=1,
        loops_per_repeat=2,
        total_sample_time_s=0.006,
        mean_s=0.001,
        median_s=0.001,
        stdev_s=0.0001,
        min_s=0.0008,
        max_s=0.0012,
        p05_s=0.0008,
        p25_s=0.0009,
        p75_s=0.0011,
        p95_s=0.0012,
        p99_s=0.0012,
        iqr_s=0.0002,
        cv=0.1,
        hz=1000.0,
        peak_memory_bytes=2048,
    )
    failed = BenchmarkResult(
        name="audit.fail",
        group="audit",
        description="synthetic failed benchmark",
        tags=(),
        success=False,
        repeats=1,
        warmups=0,
        loops_per_repeat=1,
        total_sample_time_s=0.0,
        error="boom",
        traceback="trace",
    )

    charts = benchmark.write_charts([result, failed], tmp_path)
    assert {path.name for path in charts} == {"benchmark_timing_summary.png", "benchmark_variability.png"}

    csv_path = benchmark.write_csv([result, failed], tmp_path / "benchmark_results.csv")
    json_path = benchmark.write_json([result, failed], {"generated_utc": "now", "ssapy_toolkit_version": "test", "platform": "linux"}, tmp_path / "benchmark_results.json")
    dashboard = benchmark.write_dashboard(
        [result, failed],
        {"generated_utc": "now", "ssapy_toolkit_version": "test", "platform": "linux"},
        tmp_path,
        csv_path=csv_path,
        json_path=json_path,
        chart_paths=charts,
    )

    assert dashboard.exists()
    text = dashboard.read_text(encoding="utf-8")
    assert "audit.fast" in text
    assert "audit.fail" in text
    assert "benchmark_timing_summary.png" in text
    assert "Peak Memory" in text

    cases_text = benchmark.list_cases(benchmark.build_benchmark_cases(include_io=False, include_plots=False, include_slow=False))
    assert "Available SSATK benchmark cases" in cases_text
    assert "vectors" in cases_text
    state_case = next(
        case
        for case in benchmark.build_benchmark_cases(include_io=False, include_plots=False, include_slow=False)
        if case.name == "orbital.state_to_kepler"
    )
    state_result = state_case.factory(context)()
    assert len(state_result) >= 5


def test_plot_utility_and_scene_primitive_wrappers(monkeypatch, tmp_path):
    from PIL import Image

    from ssapy_toolkit.plots import plotutils, scene_primitives

    moon_image = Image.new("RGB", (10, 5), "gray")
    monkeypatch.setattr(plotutils, "find_file", lambda *args, **kwargs: str(tmp_path / "moon.png"))
    monkeypatch.setattr(plotutils.PILImage, "open", lambda path: moon_image.copy())
    loaded = plotutils.load_moon_file()
    assert loaded.size == (1080, 540)

    fig = go.Figure()
    html_path = plotutils.save_plotly_figure(fig, save_path=tmp_path / "plotly_audit")
    assert html_path == str(tmp_path / "plotly_audit.html")
    assert (tmp_path / "plotly_audit.html").exists()
    assert plotutils.save_plotly_figure(fig, save_path=False) is None
    with pytest.raises(TypeError, match="unexpected"):
        plotutils.save_plotly_figure(fig, save_path=tmp_path / "bad.html", invalid=True)

    trace = plotutils.plotly_orbit_trace([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name="audit", color="#123456", width=4)
    assert trace.name == "audit"
    assert trace.line.color == "#123456"
    assert list(trace.x) == [1.0, 4.0]

    belts = scene_primitives.van_allen_traces(show_inner=True, show_outer=False, n_pts=6)
    assert len(belts) == 1
    fig = go.Figure()
    added_belts = scene_primitives.add_van_allen(fig, show_inner=False, show_outer=True, n_pts=6)
    assert len(added_belts) == 1
    assert len(fig.data) == 1

    class FakeMagfieldLayer:
        def __init__(self, *args, **kwargs):
            pass

        def add_to_plotly(self, fig, orbit_state=None):
            fig.add_trace(go.Scatter3d(x=[1.0], y=[2.0], z=[3.0], name="mag"))

    monkeypatch.setattr("ssapy_toolkit.plots.layers.MagfieldLayer", FakeMagfieldLayer)
    mag = scene_primitives.magfield_traces(seed_lats=[30], max_r_re=2.0)
    assert len(mag) == 1
    fig = go.Figure()
    added_mag = scene_primitives.add_magfield(fig)
    assert len(added_mag) == 1
    assert len(fig.data) == 1


def test_satellite_viewer_builder_helpers(monkeypatch, tmp_path):
    from ssapy_toolkit.plots import build_satellite_viewer as builder

    monkeypatch.setattr(builder, "SEARCH_DIRS", [str(tmp_path)])
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")
    assert builder.find_input("sample.txt") == str(sample)
    assert builder.text("sample.txt") == "hello"
    with pytest.raises(FileNotFoundError, match="could not find"):
        builder.find_input("missing.txt")

    monkeypatch.setattr(builder, "load_textures", lambda: {key: f"{key}_b64" for key in ("day", "night", "specular", "clouds")})
    monkeypatch.setattr(builder, "text", lambda filename: f"/* {filename} */")
    out = tmp_path / "viewer.html"
    assert builder.build(out_path=out, verbose=False) == out
    built = out.read_text(encoding="utf-8")
    assert "__SCENE_JS__" not in built
    assert "day_b64" in built


def test_groundtrack_enhanced_core_helpers(monkeypatch, tmp_path):
    from ssapy_toolkit.plots import groundtrack_enhanced as gt

    times = Time(["2025-01-01T00:00:00", "2025-01-01T00:05:00", "2025-01-01T00:10:00"], scale="utc")
    r_eci = np.array([[7000.0, 0.0, 0.0], [0.0, 7000.0, 0.0], [-7000.0, 0.0, 0.0]])

    ecef = gt.gcrf_to_itrf(r_eci, times)
    assert ecef.shape == r_eci.shape
    lat, lon, alt = gt.ecef_to_geodetic(ecef)
    assert lat.shape == lon.shape == alt.shape == (3,)
    ss_lat, ss_lon = gt.subsolar_point(times)
    assert ss_lat.shape == ss_lon.shape == (3,)
    illumination = gt.compute_eclipse(r_eci, times)
    assert illumination.shape == (3,)
    assert np.all((0.0 <= illumination) & (illumination <= 1.0))

    monkeypatch.setattr(gt, "_draw_continents", lambda ax: None)
    monkeypatch.setattr(gt, "gcrf_to_itrf", lambda r, t: np.asarray(r, dtype=float))
    monkeypatch.setattr(gt, "subsolar_point", lambda t: (np.zeros(len(t)), np.zeros(len(t))))
    monkeypatch.setattr(gt, "compute_eclipse", lambda r, t: np.ones(len(r)))
    fig, ax, meta = gt.plot_enhanced_groundtrack(
        r_eci,
        times,
        site_lat=0.0,
        site_lon=0.0,
        sat_name="audit",
        save_path=tmp_path / "groundtrack.png",
    )
    assert (tmp_path / "groundtrack.png").exists()
    assert 0.0 <= meta["site_visibility_pct"] <= 100.0
    plt.close(fig)


def test_sensor_fov_plot_public_entrypoints(tmp_path):
    from ssapy_toolkit.plots import sensor_fov_plot as sfp

    cfg = sfp.DEFAULT_CFG.copy()
    cfg.update(
        n_orbits=0.001,
        dt_s=120.0,
        show_stars=False,
        show_moon=False,
        show_sun=False,
        show_sensor=True,
        fov_animate=False,
        earth_n_lat=6,
        earth_n_lon=12,
        axis_range_km=10_000.0,
    )
    r_km, v_kms = sfp.propagate_orbit(cfg)
    assert r_km.shape == v_kms.shape
    assert r_km.shape[1] == 3
    assert len(r_km) >= 100

    fig = sfp.plot_sensor_fov(
        r=r_km[:4],
        v=v_kms[:4],
        t=Time([0.0, 60.0, 120.0, 180.0], format="gps"),
        cfg=cfg,
        save_path=tmp_path / "sensor.html",
    )
    assert (tmp_path / "sensor.html").exists()
    assert isinstance(fig, go.Figure)


def test_moon_plot_3d_minimal_public_entrypoint(monkeypatch):
    import importlib

    moon_module = importlib.import_module("ssapy_toolkit.plots.moon_plot_3d")

    monkeypatch.setattr(moon_module, "_ssapy_texture", lambda name: None)
    fig, ax = moon_module.moon_plot_3d(
        show_stars=False,
        show_sun=False,
        show_lagrange=False,
        show_earth=False,
        figsize=(4, 4),
    )
    assert fig is not None
    assert ax.name == "3d"
    plt.close(fig)


def test_magfield_public_entrypoints_controlled_paths(monkeypatch):
    import importlib

    magfield = importlib.import_module("ssapy_toolkit.plots.magfield_plot_3d")

    calls = []
    original_plot_magfield_3d = magfield.plot_magfield_3d

    def fake_plot_magfield_3d(**kwargs):
        calls.append(kwargs)
        return go.Figure()

    monkeypatch.setattr(magfield, "latest_driven_epoch", lambda: 2025.0)
    monkeypatch.setattr(magfield, "plot_magfield_3d", fake_plot_magfield_3d)
    fig = magfield.quick_figure("draft", save_path=False, max_r_re=2.0)
    assert isinstance(fig, go.Figure)
    assert calls[0]["epoch"] == 2025.0
    assert calls[0]["max_r_re"] == 2.0
    with pytest.raises(ValueError, match="preset"):
        magfield.quick_figure("not-a-preset")

    monkeypatch.setattr(magfield, "plot_magfield_3d", original_plot_magfield_3d)
    monkeypatch.setattr(magfield, "_HAS_PPIGRF", False)
    with pytest.raises(ImportError, match="ppigrf"):
        magfield.plot_magfield_3d(fidelity="draft", show=False)


def test_continuous_transfer_plot_helpers_execute_with_small_cases(monkeypatch):
    import importlib

    tic_module = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_inclination_continuous")
    tvc_module = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_velocity_continuous")

    monkeypatch.setattr(tic_module.plt, "show", lambda: None)
    monkeypatch.setattr(tvc_module.plt, "show", lambda: None)
    monkeypatch.setattr(tic_module, "save_plot", lambda fig, path: path)
    monkeypatch.setattr(tvc_module, "save_plot", lambda fig, path: path)

    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7546.0, 0.0])

    velocity_result = tvc_module.transfer_velocity_continuous(
        r0,
        v0,
        v_target=1.0,
        a_thrust=10.0,
        max_time=2.0,
        plot=True,
        save_path=False,
    )
    assert velocity_result["tof"] == pytest.approx(0.1)
    assert velocity_result["delta_v_total"] == pytest.approx(1.0)

    inclination_result = tic_module.transfer_inclination_continuous(
        r0,
        v0,
        i_target=0.0,
        a_thrust=10.0,
        max_time=2.0,
        plot=True,
        save_path=False,
    )
    assert inclination_result["tof"] == pytest.approx(0.0)
    assert inclination_result["delta_v_total"] == pytest.approx(0.0)
    plt.close("all")


def test_plot_layer_and_scene_public_methods_smoke(monkeypatch, tmp_path):
    from ssapy_toolkit.plots import base_plot
    from ssapy_toolkit.plots.layers import (
        BaseLayer,
        BurnLayer,
        EarthLayer,
        EclipseLayer,
        GroundTrackLayer,
        LagrangeLayer,
        MagfieldLayer,
        MoonLayer,
        NTWLayer,
        OrbitSunLayer,
        SensorFOVLayer,
        StarfieldLayer,
        TerminatorLayer,
        VanAllenLayer,
        available_layers,
        create_layer,
    )
    from ssapy_toolkit.plots.orbit_state import OrbitalState, Trajectory
    from ssapy_toolkit.plots.satellite import BurnEvent, Satellite3D

    monkeypatch.setattr(base_plot.BasePlot3D, "_start_iers_thread", lambda self: None)
    monkeypatch.setattr(base_plot.PlotlyScene, "_start_iers_thread", lambda self: None)
    monkeypatch.setattr(MoonLayer, "_moon_position_km", lambda self, t_gps: np.array([384_400.0, 0.0, 0.0]))
    monkeypatch.setattr(OrbitSunLayer, "_sun_pos", lambda self, t_gps, scene_r: np.array([scene_r, 0.0, 0.0]))
    monkeypatch.setattr(EclipseLayer, "_eclipse_mask", lambda self, traj, orbit_state: np.array([False, True, False]))
    monkeypatch.setattr(MagfieldLayer, "_trace_lines", lambda self: [np.array([[6500.0, 0.0, 0.0], [6600.0, 10.0, 0.0]])])

    class DummyLayer(BaseLayer):
        def add_to_mpl(self, ax, orbit_state, traj=None, satellite=None, **kw):
            return []

        def add_to_plotly(self, fig, orbit_state, traj=None, satellite=None, **kw):
            fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], name="dummy"))

    layer = DummyLayer("dummy", "Dummy")
    artist = SimpleNamespace(removed=False)
    artist.remove = lambda: setattr(artist, "removed", True)
    layer._artists_mpl = [artist]
    layer.remove_from_mpl()
    assert artist.removed
    assert "earth" in available_layers()
    assert isinstance(create_layer("earth"), EarthLayer)
    with pytest.raises(KeyError, match="Unknown layer"):
        create_layer("missing")

    state = OrbitalState(a_km=7000.0, e=0.001, inc_deg=10.0, epoch="2025-01-01T00:00:00+00:00", name="audit")
    r0, v0 = state.to_rv()
    traj = Trajectory(
        r=np.array([r0, r0 + np.array([0.0, 10.0, 0.0]), r0 + np.array([0.0, 20.0, 0.0])]),
        v=np.array([v0, v0, v0]),
        t=np.array([0.0, 10.0, 20.0]),
    )
    sat = Satellite3D(name="audit-sat")
    sat.add_burn(BurnEvent(epoch_offset_s=10.0, dv_ntw_km_s=[0.0, 0.01, 0.0]))

    csv_path = tmp_path / "stars.csv"
    csv_path.write_text("ra,dec,mag,sptype\n0,0,1.0,G\n6,30,2.0,B\n", encoding="utf-8")
    layers = [
        StarfieldLayer(csv_path, sky_radius_factor=2.0),
        EarthLayer(n_lat=6, n_lon=12),
        MoonLayer(n_pts=8),
        OrbitSunLayer(),
        GroundTrackLayer(n_pts=5),
        TerminatorLayer(),
        EclipseLayer(),
        VanAllenLayer(n_pts=6),
        MagfieldLayer(seed_lats=[30], max_r_re=2.0),
        LagrangeLayer(),
        NTWLayer(satellite=sat),
        BurnLayer(satellite=sat),
        SensorFOVLayer(
            traj.r,
            traj.v,
            half_angle_deg=5.0,
            cone_length_km=500.0,
            n_sides=8,
            sun_direction_gcrf=[1.0, 0.0, 0.0],
        ),
    ]

    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111, projection="3d")
    for item in layers:
        artists = item.add_to_mpl(ax, state, traj=traj, satellite=sat, scene_radius_km=10_000.0)
        assert artists is not None
    plt.close(mpl_fig)

    plotly_fig = go.Figure()
    for item in layers:
        before = len(plotly_fig.data)
        item.add_to_plotly(plotly_fig, state, traj=traj, satellite=sat, scene_radius_km=10_000.0)
        assert len(plotly_fig.data) >= before
    assert any(getattr(trace, "name", "") == "Stars" for trace in plotly_fig.data)
    assert len(SensorFOVLayer(traj.r, traj.v, n_sides=8, sun_direction_gcrf=[1, 0, 0]).build_traces()) == 5
    with pytest.raises(ValueError, match="pointing_mode"):
        SensorFOVLayer(traj.r, traj.v, pointing_mode="bad")
    with pytest.raises(ValueError, match="v_gcrf_kms required"):
        SensorFOVLayer(traj.r, pointing_mode="velocity")

    plot = base_plot.BasePlot3D(state, satellite=sat)
    assert plot.add_layer(DummyLayer("dummy", "Dummy")) is plot
    assert plot.add_layer("earth") is plot
    assert plot.toggle_layer("earth", False).layers["earth"].enabled is False
    assert plot.remove_layer("earth") is plot
    assert plot.set_frame("ECI") is plot
    assert plot.fidelity == "fast"
    assert plot.on_fidelity_change(lambda status: None) is plot
    plot.invalidate_trajectory()
    line = plot.draw_osculating()
    assert line is not None
    fig = plot.render(n_orbits=0.001, dt_s=10.0)
    assert fig is plot.fig
    assert plot.ax.name == "3d"
    animation = plot.animate(n_orbits=0.001, dt_s=10.0, interval_ms=1, trail_pts=2)
    assert animation is plot._anim
    save_path = plot.save(tmp_path / "plot.png")
    assert save_path.exists()
    with pytest.raises(RuntimeError, match="animate"):
        base_plot.BasePlot3D(state).save_animation(tmp_path / "missing.mp4")
    plot._anim = SimpleNamespace(save=lambda path, fps=25, **kwargs: tmp_path.joinpath("anim.txt").write_text(f"{path}:{fps}"))
    assert plot.save_animation(tmp_path / "anim.mp4", fps=12) == tmp_path / "anim.mp4"
    plt.close(plot.fig)

    scene = base_plot.PlotlyScene(state, satellite=sat)
    assert scene.add_layer(DummyLayer("dummy", "Dummy")) is scene
    assert scene.add_layer("earth") is scene
    assert scene.toggle_layer("earth", False).layers["earth"].enabled is False
    assert scene.remove_layer("earth") is scene
    assert scene.set_frame("ECI") is scene
    assert scene.fidelity == "fast"
    assert scene.on_fidelity_change(lambda status: None) is scene
    scene.invalidate_trajectory()
    built = scene.build(n_orbits=0.001, dt_s=10.0, show_osculating=True)
    assert isinstance(built, go.Figure)
    fast = scene.build_fast()
    assert isinstance(fast, go.Figure)
    comparison = base_plot.PlotlyScene.compare([state, state.clone(name="audit2")], n_orbits=0.001, dt_s=10.0)
    assert isinstance(comparison, go.Figure)
