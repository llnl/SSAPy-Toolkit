import numpy as np
import pytest

from ssapy_toolkit.constants import EARTH_MU
from ssapy_toolkit.orbital_mechanics.misc import hohmann_transfer_delta_v
from ssapy_toolkit.orbital_mechanics.transfer_hohmann import transfer_hohmann
from ssapy_toolkit.orbital_mechanics.transfer_ssapy_function import transfer_ssapy


def _circular_state(radius, theta=0.0):
    r = radius * np.array([np.cos(theta), np.sin(theta), 0.0])
    speed = np.sqrt(EARTH_MU / radius)
    v = speed * np.array([-np.sin(theta), np.cos(theta), 0.0])
    return r, v


def _assert_standard_transfer(result, method):
    assert result["schema_version"] == "ssatk.transfer.v2"
    assert result["method"] == method
    assert result["units"]["delta_v"] == "m/s"
    assert result["initial"]["r"].shape == (3,)
    assert result["target"]["v"].shape == (3,)
    assert all(burn["delta_v"].shape == (3,) for burn in result["burns"])
    assert result["delta_v_total"] == pytest.approx(sum(burn["delta_v_mag"] for burn in result["burns"]))


def test_hohmann_matches_closed_form_outward_and_inward():
    radius1 = 7000e3
    radius2 = 9000e3
    outward = transfer_hohmann(radius1, radius2, samples=12, burn_accel=0.5)
    expected = hohmann_transfer_delta_v(radius1, radius2, EARTH_MU)
    _assert_standard_transfer(outward, "transfer_hohmann")
    assert outward["diagnostics"]["arrival_mode"] == "insertion"
    assert outward["diagnostics"]["arrival_burn"] is True
    np.testing.assert_allclose(outward["delta_v_magnitudes"], expected[:2], rtol=1e-12)
    assert outward["delta_v_total"] == pytest.approx(expected[-1])
    assert outward["trajectory"]["r"].shape == (12, 3)
    np.testing.assert_allclose(outward["target"]["r"], outward["final"]["r"])
    np.testing.assert_allclose(outward["trajectory"]["r"][0], outward["initial"]["r"])
    np.testing.assert_allclose(outward["trajectory"]["r"][-1], outward["final"]["r"], atol=1e-7)
    transfer_energy = (
        0.5 * np.sum(outward["trajectory"]["v"]**2, axis=1)
        - EARTH_MU / np.linalg.norm(outward["trajectory"]["r"], axis=1)
    )
    expected_energy = -EARTH_MU / (radius1 + radius2)
    np.testing.assert_allclose(transfer_energy, expected_energy, rtol=1e-12)
    assert all(burn["duration"] == pytest.approx(burn["delta_v_mag"] / 0.5) for burn in outward["burns"])

    inward = transfer_hohmann(radius2, radius1, samples=12)
    expected_inward = hohmann_transfer_delta_v(radius2, radius1, EARTH_MU)
    np.testing.assert_allclose(inward["delta_v_magnitudes"], expected_inward[:2], rtol=1e-12)
    assert inward["delta_v_total"] == pytest.approx(expected_inward[-1])


def test_hohmann_rejects_non_circular_state_and_bad_samples():
    radius1 = 7000e3
    radius2 = 9000e3
    r1, v1 = _circular_state(radius1)
    r2, v2 = _circular_state(radius2)
    with pytest.raises(ValueError, match="circular boundary"):
        transfer_hohmann((r1, v1 + np.array([100.0, 0.0, 0.0]), 0.0), (r2, v2, 0.0))
    with pytest.raises(ValueError, match="samples"):
        transfer_hohmann(radius1, radius2, samples=1)


def test_transfer_ssapy_fixed_time_returns_standard_schema():
    radius = 7000e3
    r1, v1 = _circular_state(radius, 0.0)
    r2, v2 = _circular_state(radius, np.deg2rad(60.0))
    initial = (r1, v1, 0.0)
    target = (r2, v2, 1000.0)

    result = transfer_ssapy(initial, target, propagate=False, refine=False, burn_duration=1.0)

    _assert_standard_transfer(result, "transfer_ssapy")
    w_components = [burn["delta_v_ntw"][2] for burn in result["burns"]]
    assert max(abs(float(value)) for value in w_components) == pytest.approx(0.0, abs=1e-9)


def test_transfer_ssapy_requires_epoch_or_tof():
    radius = 7000e3
    r1, v1 = _circular_state(radius, 0.0)
    r2, v2 = _circular_state(radius, np.deg2rad(60.0))
    with pytest.raises(ValueError, match="tof"):
        transfer_ssapy((r1, v1), (r2, v2), propagate=False)
    result = transfer_ssapy((r1, v1), (r2, v2), tof=1000.0, propagate=False, refine=False)
    assert result["tof"] == pytest.approx(1000.0)
