import numpy as np
import pytest

from ssapy_toolkit.constants import EARTH_MU
from ssapy_toolkit.orbital_mechanics.ellipse_inject import (
    ellipse_inject,
    ellipse_insertion,
    ellipse_intercept,
)


def _circular_state(radius=7000e3, theta=0.0, t=0.0):
    r = radius * np.array([np.cos(theta), np.sin(theta), 0.0])
    v = np.sqrt(EARTH_MU / radius) * np.array([-np.sin(theta), np.cos(theta), 0.0])
    return r, v, t


def test_ellipse_inject_rendezvous_returns_canonical_transfer():
    r1, v1, _ = _circular_state(7000e3)
    r2, v2, _ = _circular_state(8000e3, theta=np.pi / 2.0)
    free_time = ellipse_insertion(r1, v1, r2, v2, n_pts=32)["tof"]

    result = ellipse_inject(r1, v1, r2, v2, tof=free_time, n_pts=32, arrival_mode="rendezvous")

    assert result["method"] == "ellipse_inject"
    assert result["schema_version"] == "ssatk.transfer.v2"
    assert result["tof"] > 0.0
    assert len(result["burns"]) == 2
    assert result["delta_v_total"] == pytest.approx(sum(result["delta_v_magnitudes"]))
    assert result["trajectory"]["r"].shape == (32, 3)
    np.testing.assert_allclose(result["trajectory"]["r"][0], r1, atol=2e-3, rtol=0)
    np.testing.assert_allclose(result["trajectory"]["r"][-1], r2, atol=2e-3, rtol=0)
    np.testing.assert_allclose(result["initial"]["v"], v1)
    np.testing.assert_allclose(result["final"]["v"], v2)
    assert result["diagnostics"]["arrival_mode"] == "rendezvous"
    assert result["diagnostics"]["timing_constraint"] == "fixed"
    assert result["diagnostics"]["arrival_burn"] is True
    assert result["diagnostics"]["ellipse"]["e"] < 1.0
    assert result["fit"] is result["ellipse_fit"]


def test_ellipse_intercept_wrapper_is_position_only_transfer():
    r1, v1, _ = _circular_state(7000e3)
    r2, _v2, _ = _circular_state(9000e3, theta=0.7)
    free_time = ellipse_inject(r1, v1, r2, n_pts=16, arrival_mode="inject")["tof"]

    result = ellipse_intercept(r1, v1, r2, tof=free_time, n_pts=16)

    assert result["method"] == "ellipse_intercept"
    assert len(result["burns"]) == 1
    assert result["diagnostics"]["arrival_mode"] == "intercept"
    assert result["diagnostics"]["timing_constraint"] == "fixed"
    assert result["diagnostics"]["arrival_burn"] is False
    assert result["diagnostics"]["target_velocity_inferred"] is True
    np.testing.assert_allclose(result["final"]["r"], r2, atol=2e-3, rtol=0)
    np.testing.assert_allclose(result["final"]["v"], result["trajectory"]["v"][-1])
