import importlib

import numpy as np
import pytest

from ssapy_toolkit.constants import EARTH_MU
from ssapy_toolkit.orbital_mechanics.transfer_coplanar_continuous import transfer_coplanar_continuous
from ssapy_toolkit.orbital_mechanics.transfer_inclination_continuous import transfer_inclination_continuous
from ssapy_toolkit.orbital_mechanics.transfer_optimal_function import transfer_optimal
from ssapy_toolkit.orbital_mechanics.transfer_ssapy_function import transfer_ssapy
from ssapy_toolkit.orbital_mechanics.transfer_velocity_and_inclination_continuous import (
    transfer_velocity_and_inclination_continuous,
)
from ssapy_toolkit.orbital_mechanics.transfer_velocity_continuous import transfer_velocity_continuous


def _state(radius=7000e3, theta=0.0, t=0.0):
    r = radius * np.array([np.cos(theta), np.sin(theta), 0.0])
    v = np.sqrt(EARTH_MU / radius) * np.array([-np.sin(theta), np.cos(theta), 0.0])
    return r, v, t


def _assert_standard(result, method, min_burns=1):
    assert result["schema_version"] == "ssatk.transfer.v2"
    assert result["method"] == method
    assert len(result["burns"]) >= min_burns
    assert result["trajectory"] is None or result["trajectory"]["r"].shape[1] == 3
    assert result["delta_v_total"] == pytest.approx(sum(burn["delta_v_mag"] for burn in result["burns"]))


def test_transfer_ssapy_uses_standard_schema():
    dep = _state(theta=0.0, t=0.0)
    arr = _state(theta=np.deg2rad(45.0), t=900.0)
    result = transfer_ssapy(dep, arr, propagate=False, refine=False, burn_duration=1.0)
    _assert_standard(result, "transfer_ssapy", min_burns=2)


def test_transfer_arrival_modes_use_single_entry_point():
    kwargs = {"n_grid": (2, 2), "polish": False, "propagate": False, "refine": False}

    assert transfer_optimal(_state(), _state(theta=0.1), arrival_mode="inject", **kwargs)["diagnostics"]["arrival_mode"] == "inject"
    assert transfer_optimal(_state(), _state(theta=0.1), arrival_mode="intercept", **kwargs)["diagnostics"]["arrival_mode"] == "intercept"
    assert transfer_optimal(_state(), _state(theta=0.1), arrival_mode="rendezvous", **kwargs)["diagnostics"]["arrival_mode"] == "rendezvous"
    assert transfer_optimal(_state(), _state(theta=0.1), arrival_mode="insertion", **kwargs)["diagnostics"]["arrival_mode"] == "insertion"


def test_velocity_continuous_schema():
    r0, v0, _ = _state()
    result = transfer_velocity_continuous(r0, v0, v_target=5.0, a_thrust=0.5, max_time=20.0)
    _assert_standard(result, "transfer_velocity_continuous")
    assert result["burns"][0]["duration"] == pytest.approx(10.0)
    assert result["delta_v_total"] == pytest.approx(5.0)


def test_inclination_continuous_schema_and_validation():
    r0, v0, _ = _state()
    result = transfer_inclination_continuous(r0, v0, delta_v=2.0, a_thrust=0.5, max_time=10.0)
    _assert_standard(result, "transfer_inclination_continuous")
    assert result["delta_v_total"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="exactly one"):
        transfer_inclination_continuous(r0, v0, a_thrust=0.5)


def test_two_phase_continuous_schema(monkeypatch):
    combo_module = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_velocity_and_inclination_continuous")

    class FakeSolution1:
        status = 0
        t = np.array([0.0, 2.0])
        y = np.array(
            [
                [7000e3, 7000e3 + 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [7500.0, 7500.0],
                [0.0, 0.0],
            ]
        )

    class FakeSolution2:
        status = 1
        t_events = [np.array([5.0])]
        t = np.array([2.0, 5.0])

        def sol(self, times):
            times = np.asarray(times, dtype=float)
            rows = np.vstack((
                7000e3 + times,
                times,
                0.1 * times,
                np.zeros_like(times),
                np.full_like(times, 7500.0),
                np.zeros_like(times),
            ))
            return rows[:, 0] if rows.shape[1] == 1 else rows

    calls = []

    def fake_solve_ivp(**kwargs):
        calls.append(kwargs)
        return FakeSolution1() if len(calls) == 1 else FakeSolution2()

    monkeypatch.setattr(combo_module, "solve_ivp", fake_solve_ivp)
    r0, v0, _ = _state()
    result = transfer_velocity_and_inclination_continuous(
        r0,
        v0,
        i_target=0.01,
        a_thrust=1e-3,
        max_time1=2.0,
        max_time2=100.0,
    )
    _assert_standard(result, "transfer_velocity_and_inclination_continuous", min_burns=2)
    assert result["diagnostics"]["phase_split_time"] >= 2.0


def test_coplanar_continuous_schema_with_fake_solve_ivp(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.orbital_mechanics.transfer_coplanar_continuous")

    class FakeSolution:
        status = 1
        t_events = [np.array([2.0])]
        t = np.array([0.0, 1.0, 2.0])

        def sol(self, times):
            times = np.asarray(times, dtype=float)
            rows = np.vstack((
                7000e3 + times,
                times,
                np.zeros_like(times),
                np.zeros_like(times),
                np.full_like(times, 7500.0),
                np.zeros_like(times),
                np.ones_like(times),
                np.zeros_like(times),
                np.zeros_like(times),
                np.zeros_like(times),
            ))
            return rows[:, 0] if rows.shape[1] == 1 else rows

    monkeypatch.setattr(module, "solve_ivp", lambda **kwargs: FakeSolution())
    r0, v0, _ = _state()
    target = np.array([7000e3 + 2.0, 2.0, 0.0])
    result = transfer_coplanar_continuous(r0, v0, target, a_thrust=0.1)
    _assert_standard(result, "transfer_coplanar_continuous")
    assert result["tof"] == pytest.approx(2.0)
