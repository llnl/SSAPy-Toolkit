import json

import numpy as np
import pytest

from ssapy_toolkit.constants import EARTH_MU
from ssapy_toolkit.benchmark import (
    BenchmarkContext,
    build_benchmark_cases,
    filter_cases,
    main,
    run_benchmarks,
    write_json,
)


def test_benchmark_case_filter_and_timing(tmp_path):
    cases = build_benchmark_cases(include_io=False, include_plots=False, include_slow=False)
    selected = filter_cases(cases, groups={"vectors"}, pattern="normed_matrix")

    assert [case.name for case in selected] == ["vectors.normed_matrix"]

    results = run_benchmarks(
        selected,
        BenchmarkContext(output_dir=tmp_path),
        repeats=2,
        warmups=0,
        min_sample_time=0.0,
        max_loops=1,
    )

    assert len(results) == 1
    assert results[0].success
    assert results[0].loops_per_repeat == 1
    assert results[0].median_s is not None
    assert results[0].median_s >= 0.0


def test_benchmark_json_output(tmp_path):
    cases = filter_cases(
        build_benchmark_cases(include_io=False, include_plots=False, include_slow=False),
        pattern="cart2sph",
    )
    results = run_benchmarks(
        cases,
        BenchmarkContext(output_dir=tmp_path),
        repeats=1,
        warmups=0,
        min_sample_time=0.0,
        max_loops=1,
    )
    path = write_json(results, {"unit_test": True}, tmp_path / "benchmark_results.json")

    text = path.read_text(encoding="utf-8")
    assert "coordinates.cart2sph_deg_scalar" in text
    assert "unit_test" in text


def test_benchmark_registry_includes_6dof_propagation_speed_cases():
    cases = build_benchmark_cases(include_io=False, include_plots=False, include_slow=False)
    selected = filter_cases(cases, groups={"propagators_6dof"})
    names = {case.name for case in selected}

    assert names == {
        "propagators_6dof.point_mass_32_steps",
        "propagators_6dof.thruster_mass_32_steps",
        "propagators_6dof.reaction_wheel_32_steps",
        "propagators_6dof.environment_facet_32_steps",
        "propagators_6dof.articulated_facet_32_steps",
    }
    assert all("6dof" in case.tags for case in selected)


def test_6dof_benchmark_json_contains_validation_metrics(tmp_path):
    cases = filter_cases(
        build_benchmark_cases(include_io=False, include_plots=False, include_slow=False),
        pattern="point_mass_32_steps",
    )
    results = run_benchmarks(
        cases,
        BenchmarkContext(output_dir=tmp_path),
        repeats=1,
        warmups=0,
        min_sample_time=0.0,
        max_loops=1,
    )
    path = write_json(results, {}, tmp_path / "benchmark_results.json")
    result = json.loads(path.read_text(encoding="utf-8"))["results"][0]

    assert result["success"]
    assert result["median_s"] >= 0.0
    validation = result["validation"]
    assert validation["nfev"] > 0
    assert validation["finite_state_residual"] == 0.0
    assert np.isfinite(validation["quaternion_norm_residual"])


def test_benchmark_cli_quick_no_dashboard(tmp_path):
    exit_code = main([
        "--profile",
        "core",
        "--groups",
        "time",
        "--quick",
        "--no-dashboard",
        "--output-dir",
        str(tmp_path),
    ])

    assert exit_code == 0
    assert (tmp_path / "benchmark_results.csv").exists()
    assert (tmp_path / "benchmark_results.json").exists()
    assert not (tmp_path / "benchmark_dashboard.html").exists()


def test_benchmark_private_wrappers_have_expected_outputs(monkeypatch, tmp_path):
    from ssapy_toolkit import benchmark
    from ssapy_toolkit.orbital_mechanics import transfer_optimal_function

    array = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 12.0]])
    np.testing.assert_allclose(benchmark._vectors_norm(array), [5.0, 12.0])
    np.testing.assert_allclose(benchmark._vectors_einsum_norm(array), [5.0, 12.0])
    rotated = benchmark._vectors_rotate_points_3d(array)
    assert rotated.shape == array.shape
    np.testing.assert_allclose(np.linalg.norm(rotated, axis=1), np.linalg.norm(array, axis=1))

    angles = np.array([-np.pi, 0.0, 3.0 * np.pi])
    wrapped = benchmark._coordinates_rad0to2pi(angles)
    assert np.all((0.0 <= wrapped) & (wrapped < 2.0 * np.pi))
    deg = np.asarray(benchmark._coordinates_deg90to90array(np.array([-120.0, -45.0, 120.0])))
    assert np.all((-90.0 <= deg) & (deg <= 90.0))
    cyl = benchmark._coordinates_cart_to_cyl(3.0, 4.0, 5.0)
    assert cyl[0] == pytest.approx(5.0)
    assert cyl[2] == pytest.approx(5.0)

    sphere = benchmark._compute_generate_sphere_vectors(8, 2.5, seed=4)
    assert sphere.shape == (8, 3)
    np.testing.assert_allclose(np.linalg.norm(sphere, axis=1), 2.5)
    assert benchmark._compute_lambert_sphere_phase(0.0) == pytest.approx(2.0 / 3.0)
    assert benchmark._compute_airmass_kasten_young(0.0) == pytest.approx(1.0, rel=1e-3)

    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(EARTH_MU / np.linalg.norm(r0)), 0.0])
    gravity = benchmark._accel_point_earth(r0)
    np.testing.assert_allclose(gravity[1:], [0.0, 0.0])
    assert gravity[0] < 0.0

    times = np.array([0.0, 5.0, 10.0])
    lf_r, lf_v = benchmark._propagator_leapfrog(r0, v0, times)
    rk_r, rk_v = benchmark._propagator_rk4(r0, v0, times)
    assert lf_r.shape == lf_v.shape == rk_r.shape == rk_v.shape == (3, 3)
    assert np.all(np.isfinite(lf_r)) and np.all(np.isfinite(rk_r))

    sixdof = benchmark._propagator_6dof_point_mass(r0, v0, times)
    assert sixdof.r.shape == sixdof.v.shape == (3, 3)
    assert sixdof.q.shape == (3, 4)
    assert sixdof.mass is None

    thruster = benchmark._propagator_6dof_thruster_mass(times)
    assert thruster.r.shape == thruster.v.shape == (3, 3)
    assert thruster.mass is not None
    assert thruster.mass[-1] < thruster.mass[0]

    wheel = benchmark._propagator_6dof_reaction_wheel(times)
    assert wheel.r.shape == wheel.v.shape == (3, 3)
    assert wheel.wheel_momentum.shape == (3, 3)
    assert wheel.wheel_momentum[-1, 2] < 0.0

    environment = benchmark._propagator_6dof_environment(times)
    assert environment.r.shape == environment.v.shape == (3, 3)
    assert np.all(np.isfinite(environment.r))

    articulated = benchmark._propagator_6dof_articulated_facet(times)
    assert articulated.r.shape == articulated.v.shape == (3, 3)
    assert articulated.q.shape == (3, 4)
    assert np.all(np.isfinite(articulated.r))

    r2 = np.array([9000e3, 0.0, 0.0])
    v2 = np.array([0.0, np.sqrt(EARTH_MU / np.linalg.norm(r2)), 0.0])
    hohmann = benchmark._orbital_transfer_hohmann(r0, v0, r2, v2, plot=False)
    assert hohmann["success"]
    assert hohmann["delta_v_total"] > 0.0
    bielliptic = benchmark._orbital_transfer_bielliptic(r0, v0, r2, v2, rb=12_000e3, plot=False)
    assert bielliptic["success"]
    assert len(bielliptic["burns"]) == 3

    monkeypatch.setattr(
        transfer_optimal_function,
        "transfer_optimal",
        lambda *args, **kwargs: {"success": True, "args": args, "kwargs": kwargs},
    )
    optimal = benchmark._orbital_transfer_optimal("initial", "target", n_grid=(1, 1))
    assert optimal["success"]
    assert optimal["args"] == ("initial", "target")
    assert optimal["kwargs"]["n_grid"] == (1, 1)

    data = {"array": np.arange(3), "value": 4}
    saved = benchmark._io_ssatk_save_json(data, tmp_path)
    assert saved.exists()
    loaded = benchmark._io_load_json_case(data, tmp_path)()
    np.testing.assert_array_equal(loaded["array"], data["array"])
    assert loaded["value"] == 4

    positions = np.column_stack([
        7000e3 * np.cos(np.linspace(0.0, 0.2, 5)),
        7000e3 * np.sin(np.linspace(0.0, 0.2, 5)),
        np.zeros(5),
    ])
    plot_result = benchmark._plot_orbit_xy(positions)
    assert plot_result is not None
    assert benchmark._run_once(lambda: "visible", quiet=False) == "visible"
    assert benchmark._run_once(lambda: "hidden", quiet=True) == "hidden"
    assert benchmark._measure_peak_memory(lambda: [0] * 16, quiet=True) >= 0
    benchmark._close_matplotlib_figures()
