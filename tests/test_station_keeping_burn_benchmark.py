from demos.benchmarks.demo_station_keeping_burn import run


def test_station_keeping_burn_benchmark(tmp_path):
    result = run(output_dir=tmp_path)
    assert result["burn_count"] == 35
    assert result["uncontrolled_semimajor_axis_change_m"] < -50.0
    assert abs(result["controlled_semimajor_axis_change_m"]) < 5.0
    assert (tmp_path / "station_keeping_burn_benchmark.json").is_file()
