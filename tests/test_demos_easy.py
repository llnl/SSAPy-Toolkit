from importlib import import_module
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEMO_MODULES = {
    "demo_all_orbit_quantities": "demos.orbital_mechanics.demo_all_orbit_quantities",
    "demo_build_dashboard": "demos.analysis_dashboards.demo_build_dashboard",
    "demo_compare_models": "demos.orbital_mechanics.demo_compare_models",
    "demo_orbital_maneuvers": "demos.orbital_mechanics.demo_orbital_maneuvers",
    "demo_coordinate_frames": "demos.coordinates.demo_coordinate_frames",
    "demo_data_package_access": "demos.getting_started.demo_data_package_access",
    "demo_earth_sun_plot": "demos.space_environment.demo_earth_sun_plot",
    "demo_eclipse_views": "demos.eclipse.demo_eclipse_views",
    "demo_ellipse_ae_for_arrival_rv": "demos.orbital_mechanics.demo_ellipse_ae_for_arrival_rv",
    "demo_ellipses": "demos.orbital_mechanics.demo_ellipses",
    "demo_first_user_workflow": "demos.getting_started.demo_first_user_workflow",
    "demo_gcrs_to_itrs_astropy": "demos.coordinates.demo_gcrs_to_itrs_astropy",
    "demo_globe_orbit_daynight": "demos.orbit_visualization.demo_globe_orbit_daynight",
    "demo_globe_plot": "demos.orbit_visualization.demo_globe_plot",
    "demo_groundtrack_accuracy": "demos.orbital_mechanics.demo_groundtrack_accuracy",
    "demo_groundtrack_plot": "demos.orbit_visualization.demo_groundtrack_plot",
    "demo_kepler_vs_harmonics": "demos.orbital_mechanics.demo_kepler_vs_harmonics",
    "demo_magfield_plot": "demos.space_environment.demo_magfield_plot",
    "demo_moon_plot": "demos.orbit_visualization.demo_moon_plot",
    "demo_orbital_stats_dashboard": "demos.analysis_dashboards.demo_orbital_stats_dashboard",
    "demo_parsing_3le": "demos.getting_started.demo_parsing_3le",
    "demo_photometry_application": "demos.photometry.demo_photometry_application",
    "demo_sampling": "demos.data_io.demo_sampling",
    "demo_sphere_generation": "demos.data_io.demo_sphere_generation",
    "demo_ssapy_ground_lambertian_reflectance": "demos.photometry.demo_ssapy_ground_lambertian_reflectance",
    "demo_satellite_viewer": "demos.sensor_coverage.demo_satellite_viewer",
    "demo_sensor_fov_plot": "demos.sensor_coverage.demo_sensor_fov_plot",
    "demo_solar_view_plot": "demos.space_environment.demo_solar_view_plot",
    "demo_sun_view": "demos.space_environment.demo_sun_view",
    "demo_van_allen": "demos.space_environment.demo_van_allen",
}


def demo_main(module_name):
    return import_module(DEMO_MODULES[module_name]).main


def test_demo_all_orbit_quantities():
    demo_all_orbit_quantities = demo_main("demo_all_orbit_quantities")
    out = demo_all_orbit_quantities(verbose=False)
    assert "rv_case" in out
    assert "rpra_case" in out
    assert "ae_ma_case" in out


def test_demo_build_dashboard():
    demo_build_dashboard = demo_main("demo_build_dashboard")
    out = demo_build_dashboard(make_figures=False, fast=True)
    assert "fig" in out


def test_demo_compare_models():
    demo_compare_models = demo_main("demo_compare_models")
    out = demo_compare_models(make_figures=False, fast=True)
    assert "out" in out
    assert "figure_time_domain" in out
    assert "figure_rung_summary" in out


def test_demo_orbital_maneuvers():
    demo_orbital_maneuvers = demo_main("demo_orbital_maneuvers")
    out = demo_orbital_maneuvers(make_figures=False, fast=True)
    assert set(out["results"]) == {"impulsive", "fixed_time", "continuous", "optimal", "staged_optimal", "split_plane_change", "elliptical_two_burn", "burn_conversion"}
    assert len(out["summary_delta_v"]) >= 24
    assert "burn_to_deltav" in out["results"]["burn_conversion"]
    assert "deltav_to_burn" in out["results"]["burn_conversion"]
    staged = out["results"]["staged_optimal"]
    assert staged["Immediate one-stop"]["diagnostics"]["stage_timing"] == "immediate"
    assert staged["Timed one-stop"]["diagnostics"]["stage_stop_count"] == 1
    assert staged["Timed two-stop min-time"]["diagnostics"]["stage_stop_count"] == 2
    assert staged["Timed one-stop"]["delta_v_total"] < staged["Direct leave-now"]["delta_v_total"]
    assert staged["Timed two-stop min-time"]["delta_v_total"] < staged["Direct leave-now"]["delta_v_total"]
    timed_waits = [
        b["diagnostics"]["t_depart"] - a["diagnostics"]["t_arrive"]
        for a, b in zip(staged["Timed two-stop min-time"]["stage_legs"][:-1], staged["Timed two-stop min-time"]["stage_legs"][1:])
    ]
    assert any(wait > 0 for wait in timed_waits)
    elliptical = out["results"]["elliptical_two_burn"]
    for name in [key.removesuffix(" direct") for key in elliptical if key.endswith(" direct")]:
        assert elliptical[f"{name} direct"]["delta_v_total"] < elliptical[f"{name} best staged"]["delta_v_total"]
        assert "e₀=" in elliptical[f"{name} direct"]["case_description"]
    split = out["results"]["split_plane_change"]
    for name in [key.removesuffix(" split") for key in split if key.endswith(" split")]:
        assert split[f"{name} split"]["delta_v_total"] < split[f"{name} all departure"]["delta_v_total"]
        assert split[f"{name} split"]["delta_v_total"] < split[f"{name} all arrival"]["delta_v_total"]
        assert 0.0 < split[f"{name} split"]["diagnostics"]["split_inclination"] < split[f"{name} split"]["diagnostics"]["inclination_change"]


def test_demo_coordinate_frames():
    demo_coordinate_frames = demo_main("demo_coordinate_frames")
    out = demo_coordinate_frames(make_figures=False, fast=True)
    assert "roundtrip_error_m" in out
    assert out["roundtrip_error_m"].shape[0] > 0
    assert out["ntw_error"] < 1e-9


def test_demo_data_package_access():
    demo_data_package_access = demo_main("demo_data_package_access")
    out = demo_data_package_access(verbose=False)
    assert out["path_exists"]
    assert "DEMO-1" in out["text"]
    assert out["missing_package_available"] is False
    assert "is not installed" in out["missing_error"]


def test_demo_earth_sun_plot():
    demo_earth_sun_plot = demo_main("demo_earth_sun_plot")
    out = demo_earth_sun_plot(make_figures=False, fast=True)
    assert out["n_frames"] >= 180
    assert out["n_traces"] > 0


def test_demo_eclipse_views():
    demo_eclipse_views = demo_main("demo_eclipse_views")
    out = demo_eclipse_views(make_figures=False, fast=True)
    assert out["solar_strip_fig"] is not None
    assert len(out["solar_strip_fig"].axes) == 5
    assert out["lunar_strip_fig"] is out["lunar_search_fig"]
    assert len(out["lunar_strip_fig"].axes) == 2
    assert out["search_fig"] is not None
    assert out["lunar_search_fig"] is out["search_fig"]
    assert out["stats"]["mode"] == "lunar"
    assert out["stats"]["event_key"] == "2014-04-15-total-lunar-wisconsin"
    assert out["stats"]["eclipse_type"] == "Total"
    assert out["stats"]["observer_name"] == "Madison, Wisconsin"
    assert out["stats"]["observer_moon_alt_deg"] > 25.0
    assert out["stats"]["observer_sun_alt_deg"] < -20.0
    assert out["solar_2024_fig"] is None


def test_demo_ellipse_ae_for_arrival_rv():
    demo_ellipse_ae_for_arrival_rv = demo_main("demo_ellipse_ae_for_arrival_rv")
    out = demo_ellipse_ae_for_arrival_rv(make_figures=False, verbose=False)
    assert "result" in out


def test_demo_ellipses():
    demo_ellipses = demo_main("demo_ellipses")
    out = demo_ellipses(make_figures=False, fast=True)
    assert "trajectories" in out
    assert len(out["trajectories"]) > 0


def test_demo_first_user_workflow():
    demo_first_user_workflow = demo_main("demo_first_user_workflow")
    out = demo_first_user_workflow(make_figures=False, fast=True)
    assert "orbit" in out
    assert out["r"].shape[1] == 3


def test_demo_gcrs_to_itrs_astropy():
    demo_gcrs_to_itrs_astropy = demo_main("demo_gcrs_to_itrs_astropy")
    out = demo_gcrs_to_itrs_astropy(verbose=False)
    assert "itrs_coords" in out
    assert "itrs_position" in out


def test_demo_globe_orbit_daynight():
    demo_globe_orbit_daynight = demo_main("demo_globe_orbit_daynight")
    out = demo_globe_orbit_daynight(make_figures=False, fast=True)
    assert out["n_traces"] > 0
    assert out["html"] is None


def test_demo_globe_plot():
    demo_globe_plot = demo_main("demo_globe_plot")
    out = demo_globe_plot(make_figures=False, make_video=False, fast=True)
    assert "r1" in out
    assert "r2" in out


def test_demo_groundtrack_accuracy():
    demo_groundtrack_accuracy = demo_main("demo_groundtrack_accuracy")
    out = demo_groundtrack_accuracy(make_figures=False, make_video=False, fast=True)
    assert "dr_norm_km" in out
    assert len(out["dr_norm_km"]) > 0


def test_demo_groundtrack_plot():
    demo_groundtrack_plot = demo_main("demo_groundtrack_plot")
    out = demo_groundtrack_plot(make_figures=False, fast=True)
    assert "tracks" in out
    assert len(out["tracks"]) == 3


def test_demo_kepler_vs_harmonics():
    demo_kepler_vs_harmonics = demo_main("demo_kepler_vs_harmonics")
    out = demo_kepler_vs_harmonics(fast=True)
    assert "errs_km" in out
    assert len(out["errs_km"]) > 0


def test_demo_magfield_plot():
    demo_magfield_plot = demo_main("demo_magfield_plot")
    out = demo_magfield_plot(make_figures=False)
    assert out["skipped"] is True


def test_demo_moon_plot():
    demo_moon_plot = demo_main("demo_moon_plot")
    out = demo_moon_plot(make_figures=False, fast=True)
    assert out["r"].shape[1] == 3
    assert out["figures_created"] is False


def test_demo_orbital_stats_dashboard():
    demo_orbital_stats_dashboard = demo_main("demo_orbital_stats_dashboard")
    out = demo_orbital_stats_dashboard(make_figures=False, fast=True)
    assert "population" in out or "figure" in out


def test_demo_parsing_3le():
    demo_parsing_3le = demo_main("demo_parsing_3le")
    out = demo_parsing_3le(verbose=False, fast=True)
    assert "data" in out
    if out.get("skipped"):
        assert out["reason"] == "missing_data_file"
    else:
        assert out["data"] is not None


def test_demo_photometry_application():
    demo_photometry_application = demo_main("demo_photometry_application")
    out = demo_photometry_application(make_figures=False, fast=True)
    assert out["ranges_km"].shape[0] > 0
    assert set(out["topocentric"]) == {"V", "SWIR", "LWIR"}
    assert out["topocentric"]["V"].shape == out["ranges_km"].shape


def test_demo_sampling():
    demo_sampling = demo_main("demo_sampling")
    out = demo_sampling(make_figures=False, fast=True, verbose=False)
    assert "Uniform ball" in out
    assert "Gaussian" in out


def test_demo_sphere_generation():
    demo_sphere_generation = demo_main("demo_sphere_generation")
    out = demo_sphere_generation(make_figures=False, fast=True)
    assert "uniform" in out
    assert "random" in out


def test_demo_ssapy_ground_lambertian_reflectance():
    demo_ssapy_ground_lambertian_reflectance = demo_main("demo_ssapy_ground_lambertian_reflectance")
    out = demo_ssapy_ground_lambertian_reflectance(make_figures=False, fast=True)
    assert "mv" in out
    assert len(out["mv"]) > 0


def test_demo_satellite_viewer():
    demo_satellite_viewer = demo_main("demo_satellite_viewer")
    out = demo_satellite_viewer(make_figures=False, fast=True)
    assert out["skipped"] is True


def test_demo_sensor_fov_plot():
    from ssapy_toolkit.constants import RGEO

    demo_sensor_fov_plot = demo_main("demo_sensor_fov_plot")
    out = demo_sensor_fov_plot(make_figures=False, fast=True)
    assert out["r_km"].shape[1] == 3
    assert out["v_kms"].shape == out["r_km"].shape
    assert len(out["orbits"]) == 3
    assert len(out["satellites"]) == 18

    trace_names = {getattr(trace, "name", "") for trace in out["figure"].data}
    assert "Earth" in trace_names
    assert "Stars" in trace_names
    assert "Moon" in trace_names
    assert "Subsatellite ground track" in trace_names

    legend_names = [
        getattr(trace, "name", "")
        for trace in out["figure"].data
        if getattr(trace, "showlegend", None) is True
    ]
    assert legend_names == [orbit["name"] for orbit in out["orbits"]]
    symbol_guides = [annotation.text for annotation in out["figure"].layout.annotations]
    assert any("spacecraft" in text for text in symbol_guides)
    assert any("instantaneous ground coverage" in text for text in symbol_guides)
    assert not any("ground intercept" in text for text in symbol_guides)
    assert max(annotation.y for annotation in out["figure"].layout.annotations if "spacecraft" in annotation.text) < out["figure"].layout.title.y

    coverage_shading = [
        trace for trace in out["figure"].data
        if str(getattr(trace, "name", "")).endswith(" coverage shading")
    ]
    coverage_boundaries = [
        trace for trace in out["figure"].data
        if str(getattr(trace, "name", "")).endswith(" coverage boundary")
    ]
    map_track_traces = [name for name in trace_names if "map ground track" in str(name)]
    assert len(coverage_shading) == 3
    assert len(coverage_boundaries) == 18
    assert len(map_track_traces) == 0
    polar_shading = [
        trace for trace in coverage_shading
        if str(getattr(trace, "name", "")).startswith(("PXZ", "PYZ"))
    ]
    assert len(polar_shading) == 2
    for trace in polar_shading:
        mask = np.asarray(trace.z, dtype=float)
        assert np.isfinite(mask[0]).all()
        assert np.isfinite(mask[-1]).all()
    assert out["figure"].layout.scene.domain.y[0] > out["figure"].layout.yaxis.domain[1]
    assert out["figure"].layout.height <= 760
    assert 0.90 < out["figure"].layout.legend.y < out["figure"].layout.title.y

    sat_positions = np.vstack([sat["r_km"] for sat in out["satellites"]])
    assert np.allclose(np.linalg.norm(sat_positions, axis=1), RGEO / 1000.0)
    assert np.max(np.abs(sat_positions[:, 0])) > 0.95 * RGEO / 1000.0
    assert np.max(np.abs(sat_positions[:, 1])) > 0.95 * RGEO / 1000.0
    assert np.max(np.abs(sat_positions[:, 2])) > 0.95 * RGEO / 1000.0

    for prefix in ("EQ-", "PXZ-", "PYZ-"):
        assert any(str(name).startswith(prefix) and "FOV" in str(name) for name in trace_names)
        assert any(str(name).startswith(prefix) and "Footprint" in str(name) for name in trace_names)
        assert not any(str(name).startswith(prefix) and "ground intercept" in str(name) for name in trace_names)


def test_demo_solar_view_plot():
    demo_solar_view_plot = demo_main("demo_solar_view_plot")
    out = demo_solar_view_plot(make_figures=False, fast=True)
    assert out["n_traces"] > 0
    assert out["html"] is None
    trace_names = {getattr(trace, "name", "") for trace in out["figure"].data}
    assert "Stars" in trace_names
    assert "Uranus" in trace_names
    assert "Neptune" in trace_names
    assert "Mars view" in out["figure"].layout.title.text
    assert "builtin ephemeris" in out["figure"].layout.title.text
    assert out["figure"].layout.scene.camera.projection.type == "perspective"

    camera_eye = out["figure"].layout.scene.camera.eye
    horizontal_eye = np.hypot(camera_eye.x, camera_eye.y)
    assert camera_eye.z / horizontal_eye < 0.12
    assert out["figure"].layout.scene.camera.up.z == 1

    stars = next(
        trace for trace in out["figure"].data
        if getattr(trace, "name", "") == "Stars"
    )
    star_radius = np.nanmedian(np.sqrt(
        np.asarray(stars.x) ** 2 + np.asarray(stars.y) ** 2 + np.asarray(stars.z) ** 2
    ))

    x_range = out["figure"].layout.scene.xaxis.range
    y_range = out["figure"].layout.scene.yaxis.range
    z_range = out["figure"].layout.scene.zaxis.range
    assert x_range == y_range == z_range
    view_radius = float(x_range[1])
    assert 1.75 < view_radius < 1.95
    assert np.isclose(star_radius, 0.98 * view_radius, rtol=0.02)


def test_demo_sun_view():
    demo_sun_view = demo_main("demo_sun_view")
    out = demo_sun_view(make_figures=False, fast=True)
    assert "sun_hat" in out
    assert out["n_traces"] > 0


def test_demo_van_allen():
    demo_van_allen = demo_main("demo_van_allen")
    out = demo_van_allen(make_figures=False)
    assert out["skipped"] is True
