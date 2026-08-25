import json
from types import SimpleNamespace

import numpy as np
import pytest

from ssapy_toolkit.io import (
    ReferenceCaseFiles,
    ReferenceCase,
    compare_reference_case,
    read_reference_case,
    write_reference_case,
)


def test_compare_reference_case_exact_match():
    trajectory = SimpleNamespace(t=np.array([0.0, 1.0]), r=np.array([[1., 2., 3.], [2., 3., 4.]]), v=np.ones((2, 3)))
    result = compare_reference_case(trajectory, ReferenceCase(trajectory.t, trajectory.r, trajectory.v, {}))
    assert result["sample_count"] == 2
    assert result["max_position_m"] == result["max_velocity_m_s"] == 0.0


def test_compare_reference_case_interpolates():
    actual = SimpleNamespace(t=np.array([0.0, 2.0]), r=np.array([[0., 0., 0.], [2., 4., 6.]]), v=np.array([[1., 2., 3.], [3., 4., 5.]]))
    reference = ReferenceCase(np.array([1.0]), np.array([[1., 2., 3.]]), np.array([[2., 3., 4.]]), {})
    result = compare_reference_case(actual, reference)
    assert result["sample_count"] == 1
    assert result["max_position_m"] == result["max_velocity_m_s"] == 0.0


def test_compare_reference_case_rejects_span_and_metadata_mismatch():
    actual = SimpleNamespace(t=np.array([0., 1.]), r=np.zeros((2, 3)), v=np.zeros((2, 3)), metadata={"reference_frame": "GCRF"})
    outside = ReferenceCase(np.array([2.]), np.zeros((1, 3)), np.zeros((1, 3)), {})
    with pytest.raises(ValueError, match="time span"):
        compare_reference_case(actual, outside)
    mismatch = ReferenceCase(actual.t, actual.r, actual.v, {"reference_frame": "ITRF"})
    with pytest.raises(ValueError, match="reference_frame"):
        compare_reference_case(actual, mismatch)


def test_write_reference_case_emits_reproducible_oem_and_metadata(tmp_path):
    trajectory = SimpleNamespace(
        t=np.array([10.0, 12.5]),
        r=np.array([[7_000_000.0, 0.0, 0.0], [7_000_001.0, 10.0, 20.0]]),
        v=np.array([[0.0, 7_500.0, 0.0], [-1.0, 7_500.5, 2.0]]),
    )

    files = write_reference_case(
        trajectory,
        tmp_path,
        epoch="2025-01-01T00:00:00Z",
        case_name="leo_truth",
        force_models=["point_mass_earth", "J2"],
        constants={"mu_m3_s2": 3.986004418e14, "J2": 1.08262668e-3},
        integrator={"method": "DOP853", "rtol": 1e-12, "atol": 1e-9},
    )

    assert isinstance(files, ReferenceCaseFiles)
    metadata = json.loads(files.metadata_path.read_text())
    ephemeris = files.ephemeris_path.read_text()
    assert metadata["reference_frame"] == "GCRF"
    assert metadata["time_origin_s"] == 10.0
    assert metadata["force_models"] == ["point_mass_earth", "J2"]
    assert metadata["files"]["ephemeris"] == "leo_truth.oem"
    assert "REF_FRAME = GCRF" in ephemeris
    assert "2025-01-01T00:00:02.500000Z" in ephemeris
    assert "7.00000000000000000e+03" in ephemeris


def test_write_reference_case_rejects_ambiguous_epochs_and_repeated_outputs(tmp_path):
    trajectory = SimpleNamespace(t=[0.0], r=[[1.0, 2.0, 3.0]], v=[[4.0, 5.0, 6.0]])
    with pytest.raises(ValueError, match="timezone"):
        write_reference_case(trajectory, tmp_path, epoch="2025-01-01T00:00:00")

    write_reference_case(trajectory, tmp_path, epoch="2025-01-01T00:00:00Z")
    with pytest.raises(FileExistsError):
        write_reference_case(trajectory, tmp_path, epoch="2025-01-01T00:00:00Z")


def test_reference_case_oem_round_trip_reads_si_states(tmp_path):
    trajectory = SimpleNamespace(
        t=np.array([10.0, 12.5]),
        r=np.array([[7_000_000.0, 0.0, 0.0], [7_000_001.0, 10.0, 20.0]]),
        v=np.array([[0.0, 7_500.0, 0.0], [-1.0, 7_500.5, 2.0]]),
    )
    files = write_reference_case(trajectory, tmp_path, epoch="2025-01-01T00:00:00Z", case_name="round_trip")

    loaded = read_reference_case(files.metadata_path)
    assert loaded.metadata["case_name"] == "round_trip"
    np.testing.assert_allclose(loaded.t, trajectory.t, atol=1e-12)
    np.testing.assert_allclose(loaded.r, trajectory.r, atol=1e-9)
    np.testing.assert_allclose(loaded.v, trajectory.v, atol=1e-9)
    residuals = compare_reference_case(trajectory, files.ephemeris_path)
    assert residuals["max_position_m"] < 1e-8
    assert residuals["max_velocity_m_s"] < 1e-8
    np.testing.assert_allclose(read_reference_case(files.ephemeris_path).r, trajectory.r, atol=1e-9)
