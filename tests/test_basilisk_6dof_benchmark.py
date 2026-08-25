import pytest

from demos.benchmarks.demo_basilisk_6dof import _basilisk_available, run


@pytest.mark.skipif(not _basilisk_available(), reason="Basilisk package unavailable")
def test_basilisk_6dof_benchmark(tmp_path):
    result = run(output_dir=tmp_path)
    assert not result["skipped"]
    assert result["max_position_error_m"] < 1e-5
    assert result["max_velocity_error_m_s"] < 1e-7
    assert result["max_quaternion_error"] < 1e-8
    assert result["max_body_rate_error_rad_s"] < 1e-9
