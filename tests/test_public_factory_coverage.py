"""Smoke tests for the public factory and packaged-data entry points."""

import numpy as np
from astropy.time import Time

from ssapy_toolkit.accelerations_6dof import (
    SpacecraftAccel,
    SpacecraftAccelJ2,
    SpacecraftAccelSum,
    SpacecraftAccelThirdBody,
    SpacecraftAttitudePD,
    SpacecraftFacetDrag,
    SpacecraftFacetSolRad,
    SpacecraftFlatPlateDrag,
    SpacecraftFlatPlateSolRad,
    SpacecraftGravityGradientTorque,
    SpacecraftMagneticTorque,
    SpacecraftReactionWheelTorque,
    SpacecraftThrusterAccel,
    SpacecraftTorqueSum,
    make_attitude_pd,
    make_drag_acceleration,
    make_facet_drag,
    make_facet_srp,
    make_flat_plate_drag,
    make_flat_plate_srp,
    make_gravity_gradient_torque,
    make_j2_acceleration,
    make_magnetic_torque,
    make_reaction_wheel_torque,
    make_srp_acceleration,
    make_ssapy_drag,
    make_ssapy_earth_harmonics,
    make_ssapy_earth_radiation,
    make_ssapy_perturbation_acceleration,
    make_ssapy_solar_radiation,
    make_ssapy_third_body,
    make_third_body_acceleration,
    make_thruster_acceleration,
    sum_accelerations,
    sum_torques,
)
from ssapy_toolkit.environment import SpaceEnvironment
from ssapy_toolkit.environment_eop import load_packaged_eop
from ssapy_toolkit.environment_space_weather import load_packaged_space_weather


def test_public_spacecraft_factories_construct_expected_models():
    sun = np.array([1.5e11, 0.0, 0.0])
    models = [
        (make_j2_acceleration(), SpacecraftAccelJ2),
        (make_third_body_acceleration([3.8e8, 0.0, 0.0], 4.9e12), SpacecraftAccelThirdBody),
        (make_drag_acceleration(density=1.0e-12), SpacecraftAccel),
        (make_srp_acceleration(sun), SpacecraftAccel),
        (make_flat_plate_drag(density=1.0e-12), SpacecraftFlatPlateDrag),
        (make_flat_plate_srp(sun), SpacecraftFlatPlateSolRad),
        (make_facet_drag(density=1.0e-12), SpacecraftFacetDrag),
        (make_facet_srp(sun), SpacecraftFacetSolRad),
        (make_thruster_acceleration(), SpacecraftThrusterAccel),
        (make_magnetic_torque([0.0, 0.0, 1.0e-5]), SpacecraftMagneticTorque),
        (make_gravity_gradient_torque(), SpacecraftGravityGradientTorque),
        (make_reaction_wheel_torque(0.0), SpacecraftReactionWheelTorque),
        (make_attitude_pd(), SpacecraftAttitudePD),
    ]

    assert all(isinstance(model, expected) for model, expected in models)
    assert isinstance(sum_accelerations(models[0][0]), SpacecraftAccelSum)
    assert isinstance(sum_torques(models[12][0]), SpacecraftTorqueSum)


def test_public_ssapy_factories_construct_adapters():
    assert make_ssapy_third_body().accel is not None
    assert make_ssapy_drag().accel is not None
    assert make_ssapy_solar_radiation().accel is not None
    assert make_ssapy_earth_radiation().accel is not None
    assert make_ssapy_earth_harmonics(degree=2, order=0).accel is not None

    stack = make_ssapy_perturbation_acceleration(
        earth_degree=2,
        earth_order=0,
        include_planets=False,
    )
    assert len(stack.accels) == 5


def test_packaged_environment_readers_and_space_environment_accessors():
    eop = load_packaged_eop()
    weather = load_packaged_space_weather()
    assert eop.records
    assert weather.records

    environment = SpaceEnvironment(
        earth_orientation_model=lambda time: time,
        space_weather_model=lambda time: time,
    )
    query = Time("2024-01-15T00:00:00", scale="utc").gps
    assert environment.earth_orientation(query) == query
    assert environment.space_weather(query) == query
