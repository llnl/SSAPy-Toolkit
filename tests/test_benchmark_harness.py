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
