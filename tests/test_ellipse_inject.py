import numpy as np
import pytest

from ssapy_toolkit.constants import EARTH_MU
from ssapy_toolkit.orbital_mechanics.ellipse_inject import (
    ellipse_inject,
    ellipse_insertion,
    ellipse_intercept,
    ellipse_rendezvous,
    transfer_to_endpoint,
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


def test_ellipse_inject_intercept_accepts_position_only_target():
    r1, v1, _ = _circular_state(7000e3)
    r2, _v2, _ = _circular_state(9000e3, theta=0.7)

    result = ellipse_inject(r1, v1, r2, n_pts=16, arrival_mode="inject")

    assert len(result["burns"]) == 1
    assert result["diagnostics"]["arrival_mode"] == "inject"
    assert result["diagnostics"]["timing_constraint"] == "free"
    assert result["diagnostics"]["arrival_burn"] is False
    assert result["diagnostics"]["target_velocity_inferred"] is True
    np.testing.assert_allclose(result["final"]["r"], r2, atol=2e-3, rtol=0)
    np.testing.assert_allclose(result["final"]["v"], result["trajectory"]["v"][-1])

    keyword_result = ellipse_inject(initial=(r1, v1, 50.0), target=r2, n_pts=12, arrival_mode="inject")
    assert keyword_result["initial"]["t"] == pytest.approx(50.0)
    assert keyword_result["diagnostics"]["target_velocity_inferred"] is True


def test_ellipse_insertion_wrapper_matches_arrival_velocity_free_time():
    r1, v1, _ = _circular_state(7000e3)
    r2, v2, _ = _circular_state(8000e3, theta=np.pi / 2.0)

    result = ellipse_insertion(r1, v1, r2, v2, n_pts=32)

    assert result["method"] == "ellipse_insertion"
    assert len(result["burns"]) == 2
    assert result["diagnostics"]["arrival_mode"] == "insertion"
    assert result["diagnostics"]["timing_constraint"] == "free"
    assert result["diagnostics"]["arrival_burn"] is True
    np.testing.assert_allclose(result["final"]["r"], r2, atol=2e-3, rtol=0)
    np.testing.assert_allclose(result["final"]["v"], v2)


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


def test_ellipse_rendezvous_wrapper_matches_arrival_velocity():
    r1, v1, _ = _circular_state(7000e3)
    r2, v2, _ = _circular_state(8000e3, theta=np.pi / 2.0)
    free_time = ellipse_insertion(r1, v1, r2, v2, n_pts=32)["tof"]

    result = ellipse_rendezvous(r1, v1, r2, v2, tof=free_time, n_pts=32)

    assert result["method"] == "ellipse_rendezvous"
    assert len(result["burns"]) == 2
    assert result["diagnostics"]["arrival_mode"] == "rendezvous"
    assert result["diagnostics"]["timing_constraint"] == "fixed"
    assert result["diagnostics"]["arrival_burn"] is True
    assert result["diagnostics"]["target_velocity_inferred"] is False
    np.testing.assert_allclose(result["final"]["r"], r2, atol=2e-3, rtol=0)
    np.testing.assert_allclose(result["final"]["v"], v2)


def test_ellipse_wrappers_reject_conflicting_modes():
    r1, v1, _ = _circular_state(7000e3)
    r2, v2, _ = _circular_state(8000e3, theta=0.4)

    with pytest.raises(ValueError, match="ellipse_intercept fixes arrival_mode"):
        ellipse_intercept(r1, v1, r2, v2, arrival_mode="rendezvous")

    with pytest.raises(ValueError, match="ellipse_rendezvous fixes match_arrival_velocity"):
        ellipse_rendezvous(r1, v1, r2, v2, match_arrival_velocity=False)

    with pytest.raises(ValueError, match="requires an explicit arrival time"):
        ellipse_intercept(r1, v1, r2)


def test_ellipse_inject_preserves_tuple_times_and_velocity_detection():
    r1, v1, _ = _circular_state(7000e3)
    r2, v2, _ = _circular_state(8000e3, theta=0.5)

    result = ellipse_inject((r1, v1, 123.0), (r2, v2, 456.0), n_pts=14)

    assert result["initial"]["t"] == pytest.approx(123.0)
    assert result["target"]["t"] == pytest.approx(123.0 + result["tof"])
    assert result["diagnostics"]["arrival_mode"] == "inject"
    assert result["diagnostics"]["target_velocity_inferred"] is False

    with pytest.raises(ValueError, match="initial requires both position and velocity"):
        ellipse_inject(initial=r1, target=r2, arrival_mode="intercept")


def test_ellipse_inject_rejects_enforced_incompatible_tof():
    r1, v1, _ = _circular_state(7000e3)
    r2, v2, _ = _circular_state(8000e3, theta=np.pi / 2.0)

    with pytest.raises(ValueError, match="ellipse_fit time of flight"):
        ellipse_inject(r1, v1, r2, v2, tof=10.0, enforce_tof=True, n_pts=16)


def test_transfer_to_endpoint_dispatches_canonical_methods():
    r1, v1, _ = _circular_state(7000e3)
    r2, v2, _ = _circular_state(7000e3, theta=0.2)

    ellipse_result = transfer_to_endpoint("ellipse", r1, v1, r2, v2, n_pts=12)
    intercept_result = transfer_to_endpoint("ellipse_intercept", r1, v1, r2, tof=ellipse_result["tof"], n_pts=12)
    insertion_result = transfer_to_endpoint("ellipse_insertion", r1, v1, r2, v2, n_pts=12)
    rendezvous_result = transfer_to_endpoint("ellipse_rendezvous", r1, v1, r2, v2, tof=insertion_result["tof"], n_pts=12)
    lambert_result = transfer_to_endpoint(
        "lambert",
        r1,
        v1,
        r2,
        v2,
        tof=1000.0,
        propagate=False,
        refine=False,
        burn_duration=1.0,
    )

    assert ellipse_result["method"] == "ellipse_inject"
    assert ellipse_result["diagnostics"]["arrival_mode"] == "inject"
    assert intercept_result["diagnostics"]["arrival_mode"] == "intercept"
    assert len(intercept_result["burns"]) == 1
    assert insertion_result["diagnostics"]["arrival_mode"] == "insertion"
    assert len(insertion_result["burns"]) == 2
    assert rendezvous_result["diagnostics"]["arrival_mode"] == "rendezvous"
    assert len(rendezvous_result["burns"]) == 2
    assert lambert_result["method"] == "transfer_ssapy"
    assert lambert_result["tof"] == pytest.approx(1000.0)

    with pytest.raises(ValueError, match="Unsupported method"):
        transfer_to_endpoint("not_a_method", r1, v1, r2, v2)
