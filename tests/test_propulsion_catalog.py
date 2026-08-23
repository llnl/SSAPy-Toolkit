import numpy as np
import pytest

from ssapy_toolkit.constants import G0
from ssapy_toolkit.propulsion import (
    ThrusterSpec,
    available_throttle_maps,
    available_thruster_families,
    available_thruster_scales,
    available_thruster_specs,
    build_thruster,
    load_throttle_map,
    make_thruster_acceleration,
    make_thruster_profile,
    mass_flow_rate,
    propellant_mass_for_delta_v,
    thruster_catalog_dict,
    thruster_spec,
)


def test_propulsion_catalog_covers_common_satellite_and_delivery_families():
    families = set(available_thruster_families())
    assert {
        "arcjet",
        "bipropellant",
        "cold_gas",
        "dual_mode",
        "electrospray",
        "gridded_ion",
        "hall_effect",
        "liquid",
        "monopropellant",
        "resistojet",
        "solid",
    } <= families

    assert "small" in available_thruster_scales("hall_effect")
    assert "solid_kick_motor_small" in available_thruster_specs("solid")
    assert "hall_effect_small" in available_thruster_specs("hall_effect", scale="small")


def test_thruster_spec_aliases_scaled_selection_and_builders():
    hall_by_alias = thruster_spec("SPT-100")
    hall_by_family_scale = thruster_spec("hall_effect", scale="small")
    assert hall_by_alias is hall_by_family_scale
    assert isinstance(hall_by_alias, ThrusterSpec)
    assert hall_by_alias.nominal_thrust_n == pytest.approx(0.083)
    assert hall_by_alias.nominal_isp_s == pytest.approx(1604.0)
    assert hall_by_alias.nominal_power_w is not None

    thruster = build_thruster(
        "hall_effect",
        scale="small",
        direction_body=[0.0, 1.0, 0.0],
        position_body=[0.2, 0.0, 0.0],
    )
    assert thruster.thrust == pytest.approx(hall_by_alias.nominal_thrust_n)
    assert thruster.isp == pytest.approx(hall_by_alias.nominal_isp_s)
    np.testing.assert_allclose(thruster.direction_body, [0.0, 1.0, 0.0])

    profile = make_thruster_profile("monopropellant", scale="small", start=10.0, burn_time=20.0, rise_time=5.0)
    assert profile(0.0) == 0.0
    assert profile(20.0) > 0.0
    assert profile(40.0) == 0.0

    accel = make_thruster_acceleration("cold_gas_micro", mass=10.0, start=0.0, stop=1.0)
    assert accel.isp == pytest.approx(thruster_spec("cold_gas_micro").nominal_isp_s)


def test_solid_motor_constraints_and_rocket_equation_helpers():
    solid = thruster_spec("solid", scale="small")
    with pytest.raises(ValueError, match="not throttleable"):
        solid.thrust_profile(start=0.0, burn_time=10.0, throttle=0.5)
    with pytest.raises(ValueError, match="finite"):
        solid.thrust_profile(start=0.0)

    profile = solid.thrust_profile(start=0.0, burn_time=10.0)
    assert profile(5.0) == pytest.approx(solid.nominal_thrust_n)
    assert solid.acceleration_for_mass(100.0) == pytest.approx(solid.nominal_thrust_n / 100.0)
    assert solid.mass_flow_rate() == pytest.approx(solid.nominal_thrust_n / (solid.nominal_isp_s * G0))
    assert mass_flow_rate(10.0, 250.0) == pytest.approx(10.0 / (250.0 * G0))
    assert propellant_mass_for_delta_v(100.0, wet_mass_kg=1000.0, isp_s=300.0) > 0.0


def test_propulsion_catalog_is_available_from_top_level_rockets_and_engines():
    import ssapy_toolkit as ssatk
    from ssapy_toolkit.engines import thruster_specs, thrusters
    from ssapy_toolkit.rockets import thruster_spec as rocket_thruster_spec

    assert ssatk.thruster_spec("AEPS").family == "hall_effect"
    assert ssatk.available_throttle_maps is available_throttle_maps
    assert ssatk.load_throttle_map is load_throttle_map
    assert rocket_thruster_spec("AEPS") is ssatk.thruster_spec("AEPS")
    assert "Mira" in thrusters
    assert "hall_effect_high_power" in thrusters
    assert "hall_effect_high_power" in thruster_specs
    assert thruster_catalog_dict(legacy=True)["hall_effect_high_power"]["ISP"] == pytest.approx(2800.0)


def test_packaged_electric_throttle_maps_load_from_ssapy_data():
    assert {"aeps_etu2", "spt140", "next_tt10", "hermes_tdu3"} <= set(available_throttle_maps())

    aeps = load_throttle_map("AEPS ETU2")
    assert len(aeps) == 12
    assert any(row["dataset"] == "ppe_aeps_rfc" and row["average_thrust_mn"] == pytest.approx(594.0) for row in aeps)

    spt140 = load_throttle_map("spt140")
    assert len(spt140) == 26
    assert max(row["thrust_mn"] for row in spt140) == pytest.approx(287.0)
