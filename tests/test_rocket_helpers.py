import numpy as np
import pytest
from astropy.time import Time
from ssapy.accel import AccelConstNTW

from ssapy_toolkit.rockets import rescale_burn as exported_rescale_burn
from ssapy_toolkit.rockets.fuel import (
    G0,
    _finite_on_intervals,
    _to_gps_seconds,
    estimate_fuel_for_accel_ntw_burn,
    mass_profile_for_accel_ntw_burn,
)
from ssapy_toolkit.rockets.rescale_burn import rescale_burn


def test_finite_on_intervals_handles_clipping_odd_lengths_and_sorting():
    assert _finite_on_intervals([]) == []
    assert _finite_on_intervals([0.0, 10.0, 20.0], t_min=5.0, t_max=25.0) == [(5.0, 10.0), (20.0, 25.0)]

    with pytest.raises(ValueError, match="sorted"):
        _finite_on_intervals([10.0, 0.0])

    time = Time("2025-01-01T00:00:00", scale="utc")
    assert np.isclose(_to_gps_seconds(time), time.gps)
    assert _to_gps_seconds(123.0) == 123.0


def test_estimate_fuel_for_accel_ntw_burn_modes_and_errors():
    burn = AccelConstNTW(accelntw=[0.0, 1.0e-4, 0.0], time_breakpoints=[100.0, 220.0, 300.0])

    constant_accel = estimate_fuel_for_accel_ntw_burn(burn, m0_kg=250.0, isp_s=220.0, t_min=100.0, t_max=250.0)
    expected_final = 250.0 * np.exp(-(1.0e-4 * 120.0) / (220.0 * G0))
    assert constant_accel["burn_time_s"] == 120.0
    assert np.isclose(constant_accel["a_int_mps"], 0.012)
    assert np.isclose(constant_accel["m_final_kg"], expected_final)
    assert np.isclose(constant_accel["m_prop_kg"], 250.0 - expected_final)

    constant_thrust = estimate_fuel_for_accel_ntw_burn(burn, m0_kg=250.0, isp_s=220.0, mode="constant_thrust")
    assert np.isclose(constant_thrust["thrust_N"], 0.025)
    assert np.isclose(constant_thrust["mdot_kgps"], 0.025 / (220.0 * G0))
    assert np.isclose(constant_thrust["m_prop_kg"], constant_thrust["mdot_kgps"] * 120.0)

    with pytest.raises(ValueError, match="mode"):
        estimate_fuel_for_accel_ntw_burn(burn, m0_kg=250.0, isp_s=220.0, mode="bad")


def test_mass_profile_for_accel_ntw_burn_tracks_on_intervals():
    burn = AccelConstNTW(accelntw=[0.0, 1.0e-4, 0.0], time_breakpoints=[100.0, 220.0, 300.0, 360.0])
    t_grid = np.array([100.0, 160.0, 220.0, 280.0, 340.0, 400.0])

    mass = mass_profile_for_accel_ntw_burn(burn, t_grid, m0_kg=250.0, isp_s=220.0)

    assert mass.shape == t_grid.shape
    assert np.all(np.diff(mass) <= 0.0)
    np.testing.assert_allclose(mass[0], 250.0)
    np.testing.assert_allclose(mass[2], 250.0 * np.exp(-(1.0e-4 * 120.0) / (220.0 * G0)))
    np.testing.assert_allclose(mass[3], mass[2])
    np.testing.assert_allclose(mass[-1], 250.0 * np.exp(-(1.0e-4 * 180.0) / (220.0 * G0)))

    with pytest.raises(ValueError, match="1D array"):
        mass_profile_for_accel_ntw_burn(burn, np.array([[100.0, 101.0]]), m0_kg=250.0, isp_s=220.0)


def test_rescale_burn_modes_and_package_export():
    assert exported_rescale_burn is rescale_burn

    a, t, dv, thrust, impulse = rescale_burn(a0=0.2, m0=100.0, t0=10.0, m=200.0, t=5.0)
    assert np.isclose(a, 0.1)
    assert np.isclose(t, 5.0)
    assert np.isclose(dv, 0.5)
    assert np.isclose(thrust, 20.0)
    assert np.isclose(impulse, 100.0)

    a_imp, t_imp, dv_imp, thrust_imp, impulse_imp = rescale_burn(
        a0=np.array([0.2, 0.4]),
        m0=100.0,
        t0=10.0,
        m=200.0,
        t=5.0,
        mode="constant_impulse",
    )
    np.testing.assert_allclose(a_imp, [0.2, 0.4])
    np.testing.assert_allclose(t_imp, 5.0)
    np.testing.assert_allclose(dv_imp, [1.0, 2.0])
    np.testing.assert_allclose(thrust_imp, [40.0, 80.0])
    np.testing.assert_allclose(impulse_imp, [200.0, 400.0])

    with pytest.raises(ValueError, match="mode"):
        rescale_burn(0.2, 100.0, 10.0, mode="bad")
