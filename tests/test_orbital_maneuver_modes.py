"""End-to-end maneuver mode coverage.

Benchmark provenance:
* Hohmann and bi-elliptic checks use the closed-form two-body equations in
  Bate, Mueller, and White, *Fundamentals of Astrodynamics*, Ch. 3, and Curtis,
  *Orbital Mechanics for Engineering Students*, Sec. 6.3--6.4.
* Fixed-time transfer checks compare against the standard Lambert
  two-point boundary-value solution described in
  Vallado, *Fundamentals of Astrodynamics and Applications*, Ch. 7.
* Continuous-burn checks use constant-acceleration identities, delta-v = a t,
  with inclination/velocity direction conventions verified through the canonical
  SSATK transfer result schema.
* Engine sizing and propellant checks are covered in
  ``test_orbital_maneuver_reference_cases.py`` against the Tsiolkovsky rocket
  equation.
"""

from __future__ import annotations

import numpy as np
import pytest
from ssapy import Orbit

from ssapy_toolkit.constants import EARTH_MU
from ssapy_toolkit.orbital_mechanics.burn_to_deltav import burn_to_deltav
from ssapy_toolkit.orbital_mechanics.deltav_to_burn import deltav_to_burn
from ssapy_toolkit.orbital_mechanics.misc import bi_elliptic_transfer_delta_v, hohmann_transfer_delta_v
from ssapy_toolkit.orbital_mechanics.transfer_bielliptic import transfer_bielliptic
from ssapy_toolkit.orbital_mechanics.transfer_coplanar_continuous import transfer_coplanar_continuous
from ssapy_toolkit.orbital_mechanics.transfer_hohmann import transfer_hohmann
from ssapy_toolkit.orbital_mechanics.transfer_inclination_continuous import transfer_inclination_continuous
from ssapy_toolkit.orbital_mechanics.transfer_optimal_function import transfer_optimal
from ssapy_toolkit.orbital_mechanics.transfer_ssapy_function import transfer_ssapy
from ssapy_toolkit.orbital_mechanics.transfer_velocity_and_inclination_continuous import (
    transfer_velocity_and_inclination_continuous,
)
from ssapy_toolkit.orbital_mechanics.transfer_velocity_continuous import transfer_velocity_continuous


def _state(radius=7000e3, theta=0.0, inclination=0.0, t=0.0):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_inc = np.cos(inclination)
    sin_inc = np.sin(inclination)
    r = radius * np.array([cos_theta, sin_theta * cos_inc, sin_theta * sin_inc])
    v = np.sqrt(EARTH_MU / radius) * np.array([-sin_theta, cos_theta * cos_inc, cos_theta * sin_inc])
    return r, v, t


def _assert_standard_transfer(result, method, min_burns=1):
    assert result["schema_version"] == "ssatk.transfer.v2"
    assert result["method"] == method
    assert result["units"]["delta_v"] == "m/s"
    assert len(result["burns"]) >= min_burns
    assert result["delta_v_total"] == pytest.approx(sum(burn["delta_v_mag"] for burn in result["burns"]))
    assert result["initial"]["r"].shape == (3,)
    assert result["target"]["v"].shape == (3,)


@pytest.mark.parametrize("radius1,radius2", [(7000e3, 9000e3), (9000e3, 7000e3)])
def test_hohmann_outward_and_inward_cases_match_closed_form(radius1, radius2):
    result = transfer_hohmann(radius1, radius2, samples=10, burn_accel=0.25)
    expected = hohmann_transfer_delta_v(radius1, radius2, EARTH_MU)

    _assert_standard_transfer(result, "transfer_hohmann", min_burns=2)
    np.testing.assert_allclose(result["delta_v_magnitudes"], expected[:2], rtol=1e-12)
    assert result["delta_v_total"] == pytest.approx(expected[-1], rel=1e-12)
    assert result["tof"] > 0.0
    assert result["trajectory"]["r"].shape == (10, 3)
    assert all(burn["duration"] == pytest.approx(burn["delta_v_mag"] / 0.25) for burn in result["burns"])


@pytest.mark.parametrize("radius1,radius2", [(7000e3, 9000e3), (9000e3, 7000e3)])
def test_bielliptic_outward_and_inward_cases_match_closed_form(radius1, radius2):
    intermediate_radius = 20_000e3
    result = transfer_bielliptic(radius1, radius2, intermediate_radius=intermediate_radius, samples_per_arc=6)
    expected = bi_elliptic_transfer_delta_v(radius1, radius2, intermediate_radius, EARTH_MU)

    _assert_standard_transfer(result, "transfer_bielliptic", min_burns=3)
    np.testing.assert_allclose(result["delta_v_magnitudes"], expected[:3], rtol=1e-12)
    assert result["delta_v_total"] == pytest.approx(expected[-1], rel=1e-12)
    assert result["trajectory"]["r"].shape == (11, 3)


def test_transfer_ssapy_fixed_time_matches_reference_solution():
    departure = _state(theta=0.0, t=0.0)
    arrival = _state(theta=0.2, t=1000.0)
    result = transfer_ssapy(departure, arrival, propagate=False, refine=False, burn_duration=1.0)

    _assert_standard_transfer(result, "transfer_ssapy", min_burns=2)
    assert result["delta_v_total"] == pytest.approx(13624.643379536796, rel=1e-9)


def test_transfer_ssapy_accepts_raw_state_vectors():
    r0, v0, _ = _state(theta=0.0)
    r1, v1, _ = _state(theta=0.2)

    vector_result = transfer_ssapy(r0, v0, r1, v1, tof=1000.0, propagate=False, refine=False, burn_duration=1.0)
    tuple_result = transfer_ssapy((r0, v0, 0.0), (r1, v1, 1000.0), propagate=False, refine=False, burn_duration=1.0)

    _assert_standard_transfer(vector_result, "transfer_ssapy", min_burns=2)
    assert vector_result["delta_v_total"] == pytest.approx(tuple_result["delta_v_total"], rel=1e-12)
    assert vector_result["tof"] == pytest.approx(1000.0)


def test_fixed_time_transfers_accept_orbit_keyword_aliases():
    r0, v0, _ = _state(theta=0.0)
    r1, v1, _ = _state(theta=0.2)
    orbit1 = Orbit(r0, v0, t=0.0)
    orbit2 = Orbit(r1, v1, t=1000.0)

    result = transfer_ssapy(orbit1=orbit1, orbit2=orbit2, propagate=False, refine=False, burn_duration=1.0)

    _assert_standard_transfer(result, "transfer_ssapy", min_burns=2)
    assert result["tof"] == pytest.approx(1000.0)


def test_fixed_time_raw_state_vectors_require_time_of_flight():
    r0, v0, _ = _state(theta=0.0)
    r1, v1, _ = _state(theta=0.2)

    with pytest.raises(ValueError, match="tof/t2"):
        transfer_ssapy(r0, v0, r1, v1, propagate=False, refine=False)


def test_transfer_ssapy_catalog_cases_cover_raise_phasing_and_plane_change():
    radius1 = 7000e3
    radius2 = 9000e3
    period = 2.0 * np.pi * np.sqrt(radius1**3 / EARTH_MU)
    hohmann_tof = np.pi * np.sqrt((0.5 * (radius1 + radius2)) ** 3 / EARTH_MU)
    cases = {
        "coplanar raise": (_state(radius1), _state(radius2, np.deg2rad(150.0), t=(150.0 / 180.0) * hohmann_tof), 800.0, 1300.0),
        "co-orbital phasing": (_state(radius1), _state(radius1, np.deg2rad(334.0), t=0.9 * period), 1.0, 400.0),
        "raise plus plane change": (
            _state(radius1),
            _state(15_000e3, np.deg2rad(130.0), inclination=np.deg2rad(15.0), t=(130.0 / 180.0) * np.pi * np.sqrt(((radius1 + 15_000e3) / 2.0) ** 3 / EARTH_MU)),
            2000.0,
            4500.0,
        ),
    }

    for label, (departure, arrival, lower, upper) in cases.items():
        result = transfer_ssapy(departure, arrival, propagate=False, refine=False, burn_duration=1.0)
        _assert_standard_transfer(result, "transfer_ssapy", min_burns=2)
        assert lower <= result["delta_v_total"] <= upper, (label, result["delta_v_total"])


@pytest.mark.parametrize("target_delta_v", [10.0, -5.0])
def test_velocity_continuous_positive_and_negative_delta_v_cases(target_delta_v):
    r0, v0, _ = _state()
    result = transfer_velocity_continuous(r0, v0, v_target=target_delta_v, a_thrust=1.0, max_time=30.0)

    _assert_standard_transfer(result, "transfer_velocity_continuous")
    assert result["delta_v_total"] == pytest.approx(abs(target_delta_v), rel=1e-12)
    assert result["burns"][0]["duration"] == pytest.approx(abs(target_delta_v), rel=1e-12)


@pytest.mark.parametrize("delta_v", [5.0, -2.0])
def test_inclination_continuous_positive_and_negative_plane_change_cases(delta_v):
    r0, v0, _ = _state()
    result = transfer_inclination_continuous(r0, v0, delta_v=delta_v, a_thrust=1.0, max_time=30.0)

    _assert_standard_transfer(result, "transfer_inclination_continuous")
    assert result["delta_v_total"] == pytest.approx(abs(delta_v), rel=1e-12)
    assert result["trajectory"]["r"].shape[1] == 3


def test_two_phase_velocity_and_inclination_continuous_case():
    r0, v0, _ = _state()
    result = transfer_velocity_and_inclination_continuous(
        r0,
        v0,
        i_target=np.deg2rad(0.01),
        a_thrust=1.0,
        max_time1=2.0,
        max_time2=200.0,
    )

    _assert_standard_transfer(result, "transfer_velocity_and_inclination_continuous", min_burns=2)
    assert result["delta_v_total"] == pytest.approx(3.3173839266663556, rel=1e-9)
    assert result["trajectory"]["r"].shape[1] == 3


def test_coplanar_continuous_reports_unreached_rendezvous_cleanly():
    r0, v0, _ = _state()
    r1, v1, _ = _state(theta=0.01)

    with pytest.raises(ValueError, match="Failed to rendezvous"):
        transfer_coplanar_continuous(r0, v0, r1, v1, a_thrust=1.0, max_time=50.0)


@pytest.mark.parametrize(
    "delta_v_mode,arrival_burn,expected_burns,expected_objective",
    [
        ("total", True, 2, 2984.9944303932793),
        ("first", False, 1, 1492.49721519664),
        ("last", True, 2, 1492.4972151966394),
    ],
)
def test_transfer_optimal_delta_v_modes(delta_v_mode, arrival_burn, expected_burns, expected_objective):
    r0, v0, _ = _state()
    r1, v1, _ = _state(theta=0.2)
    result = transfer_optimal(
        (r0, v0, 0.0),
        (r1, v1, 0.0),
        delta_v_mode=delta_v_mode,
        arrival_burn=arrival_burn,
        t_window=(0.0, 100.0),
        tof_range=(500.0, 1000.0),
        n_grid=(2, 2),
        polish=False,
        propagate=False,
        refine=False,
        burn_duration=1.0,
    )

    _assert_standard_transfer(result, "transfer_optimal", min_burns=expected_burns)
    assert len(result["burns"]) == expected_burns
    assert result["diagnostics"]["delta_v_mode"] == delta_v_mode
    assert result["diagnostics"]["objective_delta_v"] == pytest.approx(expected_objective, rel=1e-9)


def test_transfer_optimal_time_budget_and_rendezvous_cases():
    r0, v0, _ = _state()
    r1, v1, _ = _state(theta=0.2)
    kwargs = {
        "t_window": (0.0, 100.0),
        "tof_range": (500.0, 1000.0),
        "n_grid": (2, 2),
        "polish": False,
        "propagate": False,
        "refine": False,
        "burn_duration": 1.0,
    }

    min_time = transfer_optimal((r0, v0, 0.0), (r1, v1, 0.0), objective="time", dv_budget=5000.0, **kwargs)
    rendezvous = transfer_optimal((r0, v0, 0.0), (r1, v1, 0.0), arrival_mode="rendezvous", **kwargs)

    _assert_standard_transfer(min_time, "transfer_optimal", min_burns=2)
    _assert_standard_transfer(rendezvous, "transfer_optimal", min_burns=2)
    assert min_time["diagnostics"]["objective"] == "min_time"
    assert min_time["tof"] == pytest.approx(1000.0)
    assert rendezvous["delta_v_total"] == pytest.approx(min_time["delta_v_total"], rel=1e-9)


def test_transfer_optimal_accepts_raw_state_vectors_for_all_objectives():
    r0, v0, _ = _state()
    r1, v1, _ = _state(theta=0.2)
    kwargs = {
        "t_window": (0.0, 100.0),
        "tof_range": (500.0, 1000.0),
        "n_grid": (2, 2),
        "polish": False,
        "propagate": False,
        "refine": False,
        "burn_duration": 1.0,
    }

    total = transfer_optimal(r0, v0, r1, v1, delta_v_mode="both", **kwargs)
    first = transfer_optimal(r0, v0, r1, v1, delta_v_mode="first", arrival_burn=False, **kwargs)
    last = transfer_optimal(r0, v0, r1, v1, delta_v_mode="last", **kwargs)
    fastest = transfer_optimal(r0, v0, r1, v1, objective="time", dv_budget=5000.0, **kwargs)
    rendezvous = transfer_optimal(r0, v0, r1, v1, arrival_mode="rendezvous", **kwargs)

    assert total["diagnostics"]["delta_v_mode"] == "total"
    assert first["diagnostics"]["delta_v_mode"] == "first"
    assert last["diagnostics"]["delta_v_mode"] == "last"
    assert fastest["diagnostics"]["objective"] == "min_time"
    assert rendezvous["method"] == "transfer_optimal"
    assert len(first["burns"]) == 1
    assert len(total["burns"]) == len(last["burns"]) == len(rendezvous["burns"]) == 2


def test_transfer_optimal_departure_mode_now_vs_optimize_for_state_vectors():
    r0, v0, _ = _state()
    r1, v1, _ = _state(radius=9000e3, theta=0.4)
    kwargs = {
        "tof_range": (1000.0, 6000.0),
        "n_grid": (4, 4),
        "polish": False,
        "propagate": False,
        "refine": False,
        "burn_duration": 1.0,
    }

    leave_now = transfer_optimal(r0, v0, r1, v1, departure_mode="leave_now", **kwargs)
    leave_whenever = transfer_optimal(r0, v0, r1, v1, departure_mode="leave_whenever", t_window=(0.0, 1000.0), **kwargs)

    assert leave_now["diagnostics"]["departure_mode"] == "now"
    assert leave_now["diagnostics"]["t_depart"] == pytest.approx(0.0)
    assert leave_now["diagnostics"]["grid"]["t_dep"].shape == (1,)
    assert leave_whenever["diagnostics"]["departure_mode"] == "optimize"
    assert leave_whenever["diagnostics"]["grid"]["t_dep"].shape == (4,)


def test_transfer_optimal_explicit_staged_modes_for_state_vectors():
    r0, v0, _ = _state()
    r1, v1, _ = _state(radius=9000e3, theta=0.4)
    kwargs = {
        "departure_mode": "now",
        "tof_range": (1000.0, 6000.0),
        "n_grid": (2, 2),
        "polish": False,
        "propagate": False,
        "refine": False,
        "burn_duration": 1.0,
        "stage_radii": [8000e3],
        "stage_plane_fractions": [0.0],
        "n_stage_phase": 2,
    }

    immediate = transfer_optimal(r0, v0, r1, v1, stage_mode="immediate", **kwargs)
    timed = transfer_optimal(r0, v0, r1, v1, stage_mode="timed", stage_timing="leave_whenever", **kwargs)
    fastest = transfer_optimal(r0, v0, r1, v1, stage_mode="timed", objective="time", dv_budget=1e6, **kwargs)
    best = transfer_optimal(r0, v0, r1, v1, stage_mode="best", **kwargs)

    for result, mode in [(immediate, "immediate"), (timed, "timed")]:
        _assert_standard_transfer(result, "transfer_optimal_staged", min_burns=4)
        assert result["diagnostics"]["stage_mode"] == mode
        assert result["diagnostics"]["stage_count"] == 2
        assert len(result["diagnostics"]["legs"]) == 2
    assert immediate["diagnostics"]["stage_timing"] == "immediate"
    assert timed["diagnostics"]["stage_timing"] == "timed"
    assert fastest["diagnostics"]["objective"] == "min_time"
    assert best["diagnostics"]["stage_mode"] == "best"
    assert best["diagnostics"].get("selected_stage_mode", "direct") in {"direct", "staged"}


def test_transfer_optimal_staged_keyword_orbits_and_multiple_stops():
    r0, v0, _ = _state()
    r1, v1, _ = _state(radius=9000e3, theta=0.4)
    orbit0 = Orbit(r0, v0, t=0.0)
    orbit1 = Orbit(r1, v1, t=0.0)
    kwargs = {
        "departure_mode": "now",
        "tof_range": (1000.0, 6000.0),
        "n_grid": (2, 2),
        "polish": False,
        "propagate": False,
        "refine": False,
        "burn_duration": 1.0,
        "stage_plane_fractions": [0.0],
        "n_stage_phase": 1,
    }

    keyword_orbits = transfer_optimal(
        orbit1=orbit0,
        orbit2=orbit1,
        stage_mode="immediate",
        stage_radii=[8000e3],
        **kwargs,
    )
    staged = transfer_optimal(
        r0,
        v0,
        r1,
        v1,
        stage_mode="timed",
        stage_timing="appropriately timed",
        stage_radii=[7600e3, 8200e3],
        n_stage_stops=2,
        stage_beam_width=2,
        **kwargs,
    )

    _assert_standard_transfer(keyword_orbits, "transfer_optimal_staged", min_burns=4)
    assert keyword_orbits["diagnostics"]["stage_timing"] == "immediate"
    assert keyword_orbits["diagnostics"]["stage_stop_count"] == 1

    _assert_standard_transfer(staged, "transfer_optimal_staged", min_burns=6)
    assert staged["diagnostics"]["stage_timing"] == "timed"
    assert staged["diagnostics"]["stage_stop_count"] == 2
    assert staged["diagnostics"]["leg_count"] == 3
    assert len(staged["stage_legs"]) == 3
    assert staged["diagnostics"]["stage"]["n_stage_stops"] == 2


def test_burn_impulse_conversions_preserve_time_input_and_return_consistent_shapes():
    r0, v0, _ = _state()
    orbit = Orbit(r0, v0, t=0.0)
    times = np.arange(0.0, 20.0, 1.0)
    original_times = times.copy()

    burn = burn_to_deltav(orbit, times, np.array([0.01, 0.02, 0.0]))
    impulse = deltav_to_burn(orbit, times, np.array([0.1, 0.2, 0.0]))

    np.testing.assert_array_equal(times, original_times)
    for result in (burn, impulse):
        assert result["r_continuous"].shape == result["r_instantaneous"].shape == (times.size, 3)
        assert result["v_continuous"].shape == result["v_instantaneous"].shape == (times.size, 3)
        assert result["delta_v_ntw"].shape == (3,)
        assert result["delta_v_gcrf"].shape == (3,)
