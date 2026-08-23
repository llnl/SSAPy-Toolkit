from types import SimpleNamespace

import numpy as np
import pytest

from ssapy_toolkit.accelerations_6dof import (
    SpacecraftAccelConstBody,
    SpacecraftAccelConstInertial,
    SpacecraftAccelConstNTW,
    SpacecraftAccelDrag,
    SpacecraftAccelJ2,
    SpacecraftAccelKepler,
    SpacecraftAccelSolRad,
    SpacecraftAccelSum,
    SpacecraftAccelThirdBody,
    SpacecraftAttitudePD,
    SpacecraftFacetDrag,
    SpacecraftFacetSolRad,
    SpacecraftFlatPlateDrag,
    SpacecraftFlatPlateSolRad,
    SpacecraftMagneticTorque,
    SpacecraftManeuverAccel,
    SpacecraftReactionWheelTorque,
    SpacecraftThrusterAccel,
    ThrustCurve,
    attitude_error_quaternion,
    constant_body_thrust,
    constant_body_torque,
    drag_acceleration,
    exponential_density_model,
    facet_drag_acceleration_torque,
    facet_srp_acceleration_torque,
    flat_plate_drag_acceleration_torque,
    flat_plate_srp_acceleration_torque,
    integrated_thrust_impulse,
    j2_acceleration,
    load_thrust_curve_csv,
    load_thrust_curve_data,
    magnetic_dipole_torque,
    reaction_wheel_torque,
    srp_acceleration,
    third_body_acceleration,
    thrust_profile_constant,
    thrust_profile_exponential,
    thrust_profile_pulsed,
    thrust_profile_smoothstep,
    thrust_profile_trapezoid,
    thruster_force_torque,
    thruster_mass_flow_rate,
)
from ssapy_toolkit.constants import (
    AU,
    EARTH_MU,
    EARTH_RADIUS,
    SOLAR_FLUX_1_AU,
    STANDARD_GRAVITY,
    WGS84_EARTH_OMEGA,
    J2_wgs,
    c,
)
from ssapy_toolkit.dynamics import (
    Spacecraft,
    attitude_quaternion_from_frame,
    gravity_gradient_torque,
    normalize_quaternion,
    propagate_6dof,
    quaternion_from_matrix,
    rotate_vector,
    sixdof_rhs,
)
from ssapy_toolkit.satellites import (
    Component,
    Facet,
    MagneticDipole,
    ReactionWheel,
    SpacecraftBody,
    Tank,
    Thruster,
    available_satellite_designs,
    cislunar_probe,
    debris_panel,
    earth_observation_sat,
    gnss_sat,
    load_obj_facets,
    mesh_facets,
    point_mass_inertia,
    reaction_wheel_triplet,
    satellite_design,
)


def test_spacecraft_wraps_orbit_like_state_and_propagates():
    orbit = SimpleNamespace(
        r=np.array([7_000_000.0, 0.0, 0.0]),
        v=np.array([0.0, 7_500.0, 0.0]),
        t=5.0,
    )
    inertia = np.diag([10.0, 12.0, 8.0])
    spacecraft = Spacecraft.from_orbit(
        orbit,
        q=[2.0, 0.0, 0.0, 0.0],
        omega=[0.0, 0.0, 0.001],
        inertia=inertia,
        mass=100.0,
    )

    assert spacecraft.orbit is orbit
    assert spacecraft.t == 5.0
    np.testing.assert_allclose(spacecraft.q, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(spacecraft.inertia, inertia)

    traj = spacecraft.propagate(times=[5.0, 6.0], mu=0.0)
    np.testing.assert_allclose(traj.r[0], orbit.r)
    np.testing.assert_allclose(traj.v[0], orbit.v)
    np.testing.assert_allclose(traj.omega[0], spacecraft.omega)


def test_spacecraft_raw_state_and_top_level_alias():
    import ssapy_toolkit as ssatk

    spacecraft = ssatk.Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[1.0, 0.0, 0.0],
        q=[1.0, 0.0, 0.0, 0.0],
        omega=[0.0, 0.0, 0.0],
    )
    state = spacecraft.state()

    np.testing.assert_allclose(state.r, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(state.v, [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="inertia is required"):
        spacecraft.propagate(times=[0.0, 1.0])
    with pytest.raises(ValueError, match="mass must be positive"):
        Spacecraft(r=[0, 0, 0], v=[0, 0, 0], mass=0.0)
    with pytest.raises(ValueError, match="area must be positive"):
        Spacecraft(r=[0, 0, 0], v=[0, 0, 0], area=0.0)


def test_spacecraft_physical_properties_and_trajectory_sample():
    spacecraft = Spacecraft(
        r=[1.0, 2.0, 3.0],
        v=[4.0, 5.0, 6.0],
        inertia=np.eye(3),
        mass=10.0,
        area=2.0,
        cd=2.2,
        cr=1.3,
        center_of_pressure=[0.0, 1.0, 0.0],
    )
    assert spacecraft.mass == 10.0
    assert spacecraft.area == 2.0
    np.testing.assert_allclose(spacecraft.center_of_pressure, [0.0, 1.0, 0.0])

    traj = propagate_6dof(
        orbit0=spacecraft,
        times=[0.0, 1.0],
        inertia=spacecraft.inertia,
        mu=0.0,
    )
    sampled = traj.spacecraft(
        inertia=spacecraft.inertia,
        mass=spacecraft.mass,
        area=spacecraft.area,
        cd=spacecraft.cd,
        cr=spacecraft.cr,
        center_of_pressure=spacecraft.center_of_pressure,
    )
    np.testing.assert_allclose(sampled.r, [5.0, 7.0, 9.0])
    assert sampled.mass == spacecraft.mass


def test_spacecraft_body_presets_attach_design_properties():
    body = SpacecraftBody.box_wing(
        name="test_bus",
        mass=100.0,
        bus_size=(1.0, 1.0, 1.0),
        solar_array_area=4.0,
    ).with_tanks(Tank(propellant_mass=5.0, dry_mass=1.0, name="main"))

    assert body.name == "test_bus"
    assert len(body.facets) == 8
    assert body.current_mass == pytest.approx(106.0)
    assert body.area == pytest.approx(4.0)
    assert "cubesat_3u" in available_satellite_designs()
    assert satellite_design("3U").name == "3u_cubesat"

    spacecraft = Spacecraft(r=[0, 0, 0], v=[0, 0, 0], body=body)
    np.testing.assert_allclose(spacecraft.inertia, body.inertia)
    assert spacecraft.mass == pytest.approx(body.current_mass)
    assert spacecraft.area == pytest.approx(body.area)


def test_spacecraft_body_components_update_mass_center_and_inertia():
    body = SpacecraftBody.box(name="bus", mass=10.0, size=(1.0, 1.0, 1.0)).with_components(
        Component(mass=2.0, position_body=[1.0, 0.0, 0.0], name="payload")
    ).with_tanks(Tank(propellant_mass=3.0, dry_mass=1.0, position_body=[0.0, 2.0, 0.0], name="tank"))

    expected_mass = 16.0
    expected_center = np.array([2.0, 8.0, 0.0]) / expected_mass

    assert body.current_mass == pytest.approx(expected_mass)
    np.testing.assert_allclose(body.current_center_of_mass, expected_center)
    np.testing.assert_allclose(
        point_mass_inertia(2.0, [1.0, 0.0, 0.0]),
        np.diag([0.0, 2.0, 2.0]),
    )
    assert np.min(np.linalg.eigvalsh(body.current_inertia)) > 0.0

    spacecraft = Spacecraft(r=[0, 0, 0], v=[0, 0, 0], body=body)
    assert spacecraft.mass == pytest.approx(expected_mass)
    np.testing.assert_allclose(spacecraft.inertia, body.current_inertia)


def test_satellite_design_library_supports_common_presets_and_overrides():
    designs = available_satellite_designs()
    assert "earth_observation_sat" in designs
    assert "gnss_sat" in designs
    assert "cislunar_probe" in designs
    assert "debris_panel" in designs

    eo = satellite_design("eo", mass=500.0, solar_array_axis="x")
    assert eo.name == "earth_observation_sat"
    assert eo.mass == pytest.approx(500.0)
    assert eo.current_mass > eo.mass
    assert len(eo.components) == 2

    smallsat_area = satellite_design("smallsat").area
    assert earth_observation_sat().area > smallsat_area
    assert gnss_sat().current_mass > cislunar_probe().current_mass
    assert debris_panel().facets

    import ssapy_toolkit as ssatk

    assert ssatk.satellite_design("gnss").name == "gnss_sat"
    assert ssatk.SpacecraftFacetDrag is SpacecraftFacetDrag
    assert ssatk.SpacecraftMagneticTorque is SpacecraftMagneticTorque
    assert ssatk.SpacecraftReactionWheelTorque is SpacecraftReactionWheelTorque
    assert ssatk.SpacecraftAttitudePD is SpacecraftAttitudePD
    assert ssatk.SpacecraftManeuverAccel is SpacecraftManeuverAccel
    assert ssatk.ThrustCurve is ThrustCurve
    assert ssatk.mesh_facets is mesh_facets
    assert ssatk.load_obj_facets is load_obj_facets
    assert ssatk.load_thrust_curve_data is load_thrust_curve_data
    assert ssatk.attitude_quaternion_from_frame is attitude_quaternion_from_frame
    assert ssatk.thrust_profile_trapezoid(1.0, burn_time=1.0)(0.5) == pytest.approx(1.0)
    assert ssatk.constant_body_thrust([1.0, 0.0, 0.0], 2.0)(0.0, np.zeros(3), np.zeros(3), [1, 0, 0, 0], [0, 0, 0])[0] == pytest.approx(0.5)


def test_quaternion_helpers_rotate_body_to_inertial():
    q_z90 = normalize_quaternion([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    np.testing.assert_allclose(
        rotate_vector(q_z90, [1.0, 0.0, 0.0]),
        [0.0, 1.0, 0.0],
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="non-zero"):
        normalize_quaternion([0.0, 0.0, 0.0, 0.0])


def test_attitude_quaternion_helpers_use_satellite_frame_matrices():
    z90_matrix = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    q_z90 = quaternion_from_matrix(z90_matrix)
    np.testing.assert_allclose(rotate_vector(q_z90, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)

    r = np.array([7_000_000.0, 0.0, 0.0])
    v = np.array([0.0, 7_500.0, 0.0])
    q_ntw = attitude_quaternion_from_frame("ntw", r=r, v=v)
    np.testing.assert_allclose(q_ntw, [1.0, 0.0, 0.0, 0.0])

    q_nadir = attitude_quaternion_from_frame("nadir_velocity", r=r, v=v)
    np.testing.assert_allclose(rotate_vector(q_nadir, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rotate_vector(q_nadir, [0.0, 0.0, 1.0]), [-1.0, 0.0, 0.0], atol=1e-12)

    with pytest.raises(ValueError, match="orthonormal"):
        quaternion_from_matrix(np.diag([1.0, 2.0, 1.0]))


def test_torque_free_principal_axis_spin_preserves_rate_and_norm():
    traj = propagate_6dof(
        r0=[7_000_000.0, 0.0, 0.0],
        v0=[0.0, 0.0, 0.0],
        times=np.linspace(0.0, 20.0, 5),
        mu=0.0,
        inertia=np.diag([10.0, 12.0, 8.0]),
        omega0=[0.0, 0.0, 0.01],
    )

    np.testing.assert_allclose(
        traj.omega,
        np.tile([0.0, 0.0, 0.01], (5, 1)),
        atol=1e-12,
    )
    np.testing.assert_allclose(np.linalg.norm(traj.q, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(traj.r, np.tile([7_000_000.0, 0.0, 0.0], (5, 1)), atol=1e-10)


def test_gravity_gradient_torque_matches_rigid_body_formula():
    inertia = np.diag([10.0, 20.0, 30.0])
    assert np.linalg.norm(
        gravity_gradient_torque(
            [7_000_000.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            inertia,
        )
    ) == pytest.approx(0.0)

    r = np.array([7_000_000.0, 7_000_000.0, 0.0])
    r_hat = r / np.linalg.norm(r)
    expected = (
        3.0
        * EARTH_MU
        / np.linalg.norm(r) ** 3
        * np.cross(r_hat, inertia @ r_hat)
    )
    np.testing.assert_allclose(
        gravity_gradient_torque(r, [1.0, 0.0, 0.0, 0.0], inertia),
        expected,
    )


def test_j2_acceleration_matches_standard_equatorial_formula():
    radius = 7_000_000.0
    r = np.array([radius, 0.0, 0.0])
    expected = np.array([
        -1.5 * J2_wgs * EARTH_MU * EARTH_RADIUS**2 / radius**4,
        0.0,
        0.0,
    ])

    np.testing.assert_allclose(j2_acceleration(r), expected)


def test_third_body_acceleration_is_earth_centered_perturbation():
    assert np.linalg.norm(third_body_acceleration([0, 0, 0], [10, 0, 0], 1.0)) == 0.0
    expected = np.array([1.0 / 81.0 - 1.0 / 100.0, 0.0, 0.0])

    np.testing.assert_allclose(
        third_body_acceleration([1, 0, 0], [10, 0, 0], 1.0),
        expected,
    )


def test_drag_and_srp_acceleration_have_expected_directions_and_magnitudes():
    r = np.array([EARTH_RADIUS + 400_000.0, 0.0, 0.0])
    atmosphere_velocity = np.cross([0.0, 0.0, WGS84_EARTH_OMEGA], r)
    np.testing.assert_allclose(
        drag_acceleration(r, atmosphere_velocity, density=1e-12, area=2.0, mass=100.0),
        0.0,
        atol=1e-20,
    )

    v = atmosphere_velocity + np.array([0.0, 100.0, 0.0])
    drag = drag_acceleration(r, v, density=1e-12, area=2.0, mass=100.0)
    assert np.dot(drag, v - atmosphere_velocity) < 0.0

    srp = srp_acceleration([0, 0, 0], [AU, 0, 0], area=2.0, mass=100.0, cr=1.5)
    expected_srp = np.array([-1361.0 / c * 1.5 * 2.0 / 100.0, 0.0, 0.0])
    np.testing.assert_allclose(srp, expected_srp)


def test_flat_plate_drag_acceleration_torque_and_attitude_shadowing():
    r = np.array([7_000_000.0, 0.0, 0.0])
    v = np.array([3.0, 0.0, 0.0])
    q = [1.0, 0.0, 0.0, 0.0]

    acceleration, torque = flat_plate_drag_acceleration_torque(
        r,
        v,
        q,
        density=1.0,
        area=2.0,
        mass=10.0,
        cd=2.0,
        normal_body=[1.0, 0.0, 0.0],
        center_of_pressure=[0.0, 1.0, 0.0],
        earth_radius=0.0,
        earth_rotation_rate=0.0,
    )

    np.testing.assert_allclose(acceleration, [-1.8, 0.0, 0.0])
    np.testing.assert_allclose(torque, [0.0, 0.0, 18.0])

    hidden_accel, hidden_torque = flat_plate_drag_acceleration_torque(
        r,
        v,
        q,
        density=1.0,
        area=2.0,
        mass=10.0,
        cd=2.0,
        normal_body=[-1.0, 0.0, 0.0],
        earth_radius=0.0,
        earth_rotation_rate=0.0,
    )
    np.testing.assert_allclose(hidden_accel, 0.0)
    np.testing.assert_allclose(hidden_torque, 0.0)

    facet_accel, facet_torque = facet_drag_acceleration_torque(
        r,
        v,
        q,
        [Facet(area=2.0, normal_body=[1.0, 0.0, 0.0], center_of_pressure=[0.0, 1.0, 0.0], cd=2.0)],
        density=1.0,
        mass=10.0,
        earth_radius=0.0,
        earth_rotation_rate=0.0,
    )
    np.testing.assert_allclose(facet_accel, acceleration)
    np.testing.assert_allclose(facet_torque, torque)


def test_flat_plate_srp_acceleration_torque_and_attitude_shadowing():
    acceleration, torque = flat_plate_srp_acceleration_torque(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [AU, 0.0, 0.0],
        area=2.0,
        mass=10.0,
        cr=1.0,
        normal_body=[1.0, 0.0, 0.0],
        center_of_pressure=[0.0, 1.0, 0.0],
    )

    assert acceleration[0] < 0.0
    np.testing.assert_allclose(acceleration[1:], 0.0)
    assert torque[2] > 0.0

    hidden_accel, hidden_torque = flat_plate_srp_acceleration_torque(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [AU, 0.0, 0.0],
        area=2.0,
        mass=10.0,
        cr=1.0,
        normal_body=[-1.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(hidden_accel, 0.0)
    np.testing.assert_allclose(hidden_torque, 0.0)

    facet_accel, facet_torque = facet_srp_acceleration_torque(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [AU, 0.0, 0.0],
        [Facet(area=2.0, normal_body=[1.0, 0.0, 0.0], center_of_pressure=[0.0, 1.0, 0.0], cr=1.0)],
        mass=10.0,
    )
    np.testing.assert_allclose(facet_accel, acceleration)
    np.testing.assert_allclose(facet_torque, torque)


def test_optical_srp_coefficients_match_flat_plate_limits():
    pressure = SOLAR_FLUX_1_AU / c

    absorber, _ = flat_plate_srp_acceleration_torque(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [AU, 0.0, 0.0],
        area=2.0,
        mass=10.0,
        cr=9.0,
        specular_reflectivity=0.0,
        diffuse_reflectivity=0.0,
        normal_body=[1.0, 0.0, 0.0],
    )
    specular, _ = flat_plate_srp_acceleration_torque(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [AU, 0.0, 0.0],
        area=2.0,
        mass=10.0,
        specular_reflectivity=1.0,
        diffuse_reflectivity=0.0,
        normal_body=[1.0, 0.0, 0.0],
    )
    diffuse, _ = flat_plate_srp_acceleration_torque(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [AU, 0.0, 0.0],
        area=2.0,
        mass=10.0,
        specular_reflectivity=0.0,
        diffuse_reflectivity=1.0,
        normal_body=[1.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(absorber, [-pressure * 2.0 / 10.0, 0.0, 0.0])
    np.testing.assert_allclose(specular, 2.0 * absorber)
    np.testing.assert_allclose(diffuse, (5.0 / 3.0) * absorber)
    with pytest.raises(ValueError, match="must be <= 1"):
        flat_plate_srp_acceleration_torque(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [AU, 0.0, 0.0],
            area=1.0,
            mass=1.0,
            specular_reflectivity=0.8,
            diffuse_reflectivity=0.4,
        )


def test_facet_srp_self_shadowing_and_mesh_facets(tmp_path):
    vertices = np.array([
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5],
    ])
    blocker = mesh_facets(vertices, [(0, 1, 2, 3)], specular_reflectivity=0.0, diffuse_reflectivity=0.0)[0]
    shaded = Facet(
        area=1.0,
        normal_body=[1.0, 0.0, 0.0],
        center_of_pressure=[0.0, 0.0, 0.0],
        specular_reflectivity=0.0,
        diffuse_reflectivity=0.0,
        vertices_body=[[-0.01, -0.25, -0.25], [-0.01, 0.25, -0.25], [-0.01, 0.25, 0.25], [-0.01, -0.25, 0.25]],
    )

    no_shadow, _ = facet_srp_acceleration_torque(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [AU, 0.0, 0.0],
        [shaded, blocker],
        mass=10.0,
    )
    with_shadow, _ = facet_srp_acceleration_torque(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [AU, 0.0, 0.0],
        [shaded, blocker],
        mass=10.0,
        self_shadowing=True,
    )
    assert np.linalg.norm(with_shadow) < np.linalg.norm(no_shadow)
    np.testing.assert_allclose(np.linalg.norm(with_shadow), 0.5 * np.linalg.norm(no_shadow), rtol=1e-12)

    obj_path = tmp_path / "plate.obj"
    obj_path.write_text("v 0 -0.5 -0.5\nv 0 0.5 -0.5\nv 0 0.5 0.5\nv 0 -0.5 0.5\nf 1 2 3 4\n")
    facets = load_obj_facets(obj_path)
    assert len(facets) == 1
    assert facets[0].area == pytest.approx(1.0)
    assert facets[0].vertices_body is not None


def test_facet_srp_torque_spins_spacecraft_when_center_of_pressure_is_offset():
    body = SpacecraftBody(
        name="offset_plate",
        mass=10.0,
        inertia=np.eye(3),
        facets=(
            Facet(
                area=2.0,
                normal_body=[1.0, 0.0, 0.0],
                center_of_pressure=[0.0, 1.0, 0.0],
                specular_reflectivity=0.0,
                diffuse_reflectivity=0.0,
            ),
        ),
    )
    spacecraft = Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        q=[1.0, 0.0, 0.0, 0.0],
        omega=[0.0, 0.0, 0.0],
        body=body,
    )
    srp = SpacecraftFacetSolRad([AU, 0.0, 0.0])

    traj = spacecraft.propagate(times=[0.0, 1.0], mu=0.0, models=[srp])

    assert traj.v[-1, 0] < 0.0
    assert traj.omega[-1, 2] > 0.0


def test_facet_and_thruster_models_use_spacecraft_body():
    body = SpacecraftBody.box(
        name="single_plate_body",
        mass=10.0,
        size=(1.0, 1.0, 1.0),
    ).with_facets(
        Facet(area=2.0, normal_body=[1.0, 0.0, 0.0], center_of_pressure=[0.0, 1.0, 0.0], cd=2.0, cr=1.0),
        append=False,
    ).with_thrusters(
        Thruster(thrust=1.0, direction_body=[1.0, 0.0, 0.0], position_body=[0.0, 1.0, 0.0], isp=300.0),
    )
    spacecraft = Spacecraft(
        r=[7_000_000.0, 0.0, 0.0],
        v=[3.0, 0.0, 0.0],
        q=[1.0, 0.0, 0.0, 0.0],
        omega=[0.0, 0.0, 0.0],
        body=body,
    )

    drag = SpacecraftFacetDrag(density=1.0, earth_radius=0.0, earth_rotation_rate=0.0)
    srp = SpacecraftFacetSolRad([AU, 0.0, 0.0])
    thrust = SpacecraftThrusterAccel()

    np.testing.assert_allclose(drag(spacecraft), [-1.8, 0.0, 0.0])
    assert drag.torque(spacecraft)[2] > 0.0
    assert srp(spacecraft)[0] < 0.0
    np.testing.assert_allclose(thrust(spacecraft), [0.1, 0.0, 0.0])
    np.testing.assert_allclose(thrust.torque(spacecraft), [0.0, 0.0, -1.0])
    assert thrust.mass_flow_rate(spacecraft) == pytest.approx(body.thrusters[0].mass_flow_rate())

    force_body, torque_body = thruster_force_torque(body.thrusters)
    np.testing.assert_allclose(force_body, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(torque_body, [0.0, 0.0, -1.0])
    assert thruster_mass_flow_rate(body.thrusters, throttle=0.5) == pytest.approx(
        body.thrusters[0].mass_flow_rate(throttle=0.5)
    )


def test_magnetic_dipole_torque_uses_body_frame_field():
    dipole = MagneticDipole(moment_body=[1.0, 0.0, 0.0], name="x_rod")
    body = SpacecraftBody.cubesat(1, mass=10.0).with_magnetic_dipoles(dipole)
    spacecraft = Spacecraft(
        r=[7_000_000.0, 0.0, 0.0],
        v=[0.0, 7_500.0, 0.0],
        q=[1.0, 0.0, 0.0, 0.0],
        omega=[0.0, 0.0, 0.0],
        body=body,
    )

    np.testing.assert_allclose(
        magnetic_dipole_torque([dipole], [0.0, 2.0e-5, 0.0]),
        [0.0, 0.0, 2.0e-5],
    )
    torque = SpacecraftMagneticTorque([0.0, 2.0e-5, 0.0])
    np.testing.assert_allclose(torque(spacecraft), [0.0, 0.0, 2.0e-5])
    np.testing.assert_allclose(
        SpacecraftMagneticTorque([0.0, 2.0e-5, 0.0], dipole_names=["missing"])(spacecraft),
        [0.0, 0.0, 0.0],
    )


def test_reaction_wheel_torque_allocates_and_saturates_body_torque():
    wheels = reaction_wheel_triplet(max_torque=0.02, name_prefix="wheel")
    body = SpacecraftBody.cubesat(1, mass=10.0).with_reaction_wheels(*wheels)

    assert all(isinstance(wheel, ReactionWheel) for wheel in wheels)
    assert len(body.reaction_wheels) == 3
    np.testing.assert_allclose(
        reaction_wheel_torque(body.reaction_wheels, [0.01, -0.03, 0.0]),
        [0.01, -0.02, 0.0],
    )
    np.testing.assert_allclose(
        reaction_wheel_torque(body.reaction_wheels, {"wheel_z": 0.05}),
        [0.0, 0.0, 0.02],
    )
    np.testing.assert_allclose(
        SpacecraftReactionWheelTorque([0.0, 0.0, -0.05])(Spacecraft(r=[0, 0, 0], v=[0, 0, 0], body=body)),
        [0.0, 0.0, -0.02],
    )
    with pytest.raises(ValueError, match="reaction-wheel command"):
        reaction_wheel_torque(body.reaction_wheels, [1.0, 2.0])


def test_attitude_pd_torque_uses_shortest_quaternion_error():
    q_z_error = normalize_quaternion([np.cos(0.05), 0.0, 0.0, np.sin(0.05)])
    controller = SpacecraftAttitudePD(kp=2.0, kd=0.5, max_torque=0.05)
    torque = controller(
        t=0.0,
        r=np.zeros(3),
        v=np.zeros(3),
        q=q_z_error,
        omega=np.array([0.0, 0.0, 0.02]),
    )

    assert attitude_error_quaternion(q_z_error)[3] > 0.0
    assert torque[2] < 0.0
    assert np.linalg.norm(torque) <= 0.05
    np.testing.assert_allclose(
        attitude_error_quaternion(-q_z_error),
        attitude_error_quaternion(q_z_error),
    )


def test_physics_callback_factories_compose_with_propagation():
    density = exponential_density_model(
        reference_density=1e-12,
        reference_altitude=400_000.0,
        scale_height=50_000.0,
    )
    acceleration = SpacecraftAccelSum(
        [
            SpacecraftAccelJ2(),
            SpacecraftAccelDrag(density=density, area=2.0, mass=100.0),
            SpacecraftAccelSolRad([AU, 0, 0], area=2.0, mass=100.0),
            SpacecraftAccelThirdBody([384_400_000.0, 0.0, 0.0], 4.9048695e12),
            SpacecraftAccelConstInertial([1e-5, 0.0, 0.0]),
        ]
    )
    r = np.array([EARTH_RADIUS + 400_000.0, 0.0, 0.0])
    v = np.array([0.0, 7_700.0, 0.0])
    value = acceleration(0.0, r, v, [1, 0, 0, 0], [0, 0, 0])
    assert value.shape == (3,)
    assert np.all(np.isfinite(value))

    traj = propagate_6dof(
        r0=r,
        v0=v,
        times=[0.0, 1.0],
        inertia=np.eye(3),
        acceleration=acceleration,
        body_acceleration=constant_body_thrust([0.0, 1e-3, 0.0], mass=100.0),
        torque=constant_body_torque([0.0, 0.0, 1e-6]),
    )
    assert traj.r.shape == (2, 3)
    assert traj.q.shape == (2, 4)


def test_thrust_profiles_and_csv_curve(tmp_path):
    constant = thrust_profile_constant(4.0, start=1.0, stop=3.0)
    assert constant(0.0) == pytest.approx(0.0)
    assert constant(2.0) == pytest.approx(4.0)

    trapezoid = thrust_profile_trapezoid(10.0, burn_time=10.0, rise_time=2.0, fall_time=3.0)
    assert trapezoid(1.0) == pytest.approx(5.0)
    assert trapezoid(5.0) == pytest.approx(10.0)
    assert trapezoid(8.5) == pytest.approx(5.0)
    assert integrated_thrust_impulse(trapezoid, 0.0, 10.0, samples=5001) == pytest.approx(75.0, rel=1e-4)

    smooth = thrust_profile_smoothstep(10.0, burn_time=4.0, rise_time=2.0, fall_time=2.0)
    assert smooth(0.0) == pytest.approx(0.0)
    assert smooth(1.0) == pytest.approx(5.0)
    assert smooth(2.0) == pytest.approx(10.0)

    exponential = thrust_profile_exponential(10.0, start=0.0, stop=10.0, rise_tau=2.0, decay_tau=1.0)
    assert 0.0 < exponential(1.0) < exponential(4.0) < 10.0
    assert 0.0 < exponential(11.0) < exponential(10.0)

    pulsed = thrust_profile_pulsed(2.0, period=10.0, duty_cycle=0.25, start=0.0, stop=30.0)
    assert pulsed(1.0) == pytest.approx(2.0)
    assert pulsed(3.0) == pytest.approx(0.0)

    curve = ThrustCurve([0.0, 1.0, 2.0], [0.0, 10.0, 0.0])
    assert curve(0.5) == pytest.approx(5.0)
    assert curve.total_impulse == pytest.approx(10.0)

    csv_path = tmp_path / "thrust.csv"
    csv_path.write_text("time_s,thrust_n\n0,0\n1,3\n2,0\n")
    loaded = load_thrust_curve_csv(csv_path)
    assert loaded(0.5) == pytest.approx(1.5)


def test_spacecraft_maneuver_accel_supports_operational_frames():
    spacecraft = Spacecraft(
        r=[7_000_000.0, 0.0, 0.0],
        v=[0.0, 7_500.0, 0.0],
        q=[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        omega=[0.0, 0.0, 0.0],
        inertia=np.eye(3),
        mass=100.0,
    )

    np.testing.assert_allclose(
        SpacecraftManeuverAccel(10.0, frame="rtn")(spacecraft),
        [0.0, 0.1, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        SpacecraftManeuverAccel(10.0, frame="ntw")(spacecraft),
        [0.0, 0.1, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        SpacecraftManeuverAccel(10.0, frame="body")(spacecraft),
        [0.0, 0.1, 0.0],
        atol=1e-12,
    )
    assert SpacecraftManeuverAccel(10.0, frame="rtn", isp=200.0).mass_flow_rate(spacecraft) == pytest.approx(
        10.0 / (200.0 * STANDARD_GRAVITY)
    )


def test_spacecraft_maneuver_accel_propagates_variable_finite_burn():
    spacecraft = Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        inertia=np.eye(3),
        mass=100.0,
    )
    burn = SpacecraftManeuverAccel(
        thrust_profile_constant(2.0, start=0.0, stop=10.0),
        frame="gcrf",
        direction=[1.0, 0.0, 0.0],
    )

    traj = spacecraft.propagate(times=[0.0, 10.0], mu=0.0, acceleration=burn)

    np.testing.assert_allclose(traj.v[-1], [0.2, 0.0, 0.0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(traj.r[-1], [1.0, 0.0, 0.0], rtol=1e-10, atol=1e-12)


def test_spacecraft_accel_classes_accept_spacecraft_and_ssapy_style_calls():
    spacecraft = Spacecraft(
        r=[7_000_000.0, 0.0, 0.0],
        v=[0.0, 7_500.0, 0.0],
        q=[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        omega=[0.0, 0.0, 0.0],
        inertia=np.eye(3),
        mass=100.0,
    )
    object.__setattr__(spacecraft, "area", 2.0)

    kepler = SpacecraftAccelKepler()
    np.testing.assert_allclose(kepler(spacecraft), kepler(spacecraft.r, spacecraft.v, spacecraft.t))
    np.testing.assert_allclose(SpacecraftAccelJ2()(spacecraft), j2_acceleration(spacecraft.r))
    np.testing.assert_allclose(
        SpacecraftAccelConstNTW([1.0, 2.0, 3.0])(spacecraft),
        [1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(
        SpacecraftAccelConstBody([1.0, 0.0, 0.0])(spacecraft),
        [0.0, 1.0, 0.0],
        atol=1e-12,
    )
    assert np.all(_is_finite_vector(SpacecraftAccelDrag(density=1e-12)(spacecraft)))
    assert np.all(_is_finite_vector(SpacecraftAccelSolRad([AU, 0.0, 0.0])(spacecraft)))
    assert np.all(_is_finite_vector(SpacecraftFlatPlateDrag(density=1e-12)(spacecraft)))
    assert np.all(_is_finite_vector(SpacecraftFlatPlateSolRad([AU, 0.0, 0.0])(spacecraft)))


def test_spacecraft_propagate_binds_spacecraft_acceleration_models():
    sat = Spacecraft(
        r=[EARTH_RADIUS + 400_000.0, 0.0, 0.0],
        v=[0.0, 7_700.0, 0.0],
        inertia=np.eye(3),
        mass=100.0,
    )
    object.__setattr__(sat, "area", 2.0)

    traj = sat.propagate(
        times=[0.0, 1.0],
        acceleration=SpacecraftAccelSum([
            SpacecraftAccelDrag(density=1e-12),
            SpacecraftAccelSolRad([AU, 0.0, 0.0]),
        ]),
    )
    assert traj.r.shape == (2, 3)


def test_spacecraft_propagate_binds_flat_plate_acceleration_and_torque():
    spacecraft = Spacecraft(
        r=[7_000_000.0, 0.0, 0.0],
        v=[3.0, 0.0, 0.0],
        inertia=np.eye(3),
        mass=10.0,
        area=2.0,
        cd=2.0,
        center_of_pressure=[0.0, 1.0, 0.0],
    )
    drag = SpacecraftFlatPlateDrag(density=1.0, earth_radius=0.0, earth_rotation_rate=0.0)

    traj = spacecraft.propagate(
        times=[0.0, 0.1],
        mu=0.0,
        acceleration=drag,
        torque=drag,
    )

    assert traj.v[-1, 0] < spacecraft.v[0]
    assert traj.omega[-1, 2] > 0.0


def test_spacecraft_propagate_accepts_body_and_model_list():
    body = SpacecraftBody.cubesat(1, mass=10.0).with_thrusters(
        Thruster(thrust=1.0, direction_body=[1.0, 0.0, 0.0], position_body=[0.0, 1.0, 0.0])
    )
    spacecraft = Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        q=[1.0, 0.0, 0.0, 0.0],
        omega=[0.0, 0.0, 0.0],
    )

    traj = spacecraft.propagate(
        times=[0.0, 1.0],
        body=body,
        mu=0.0,
        models=[SpacecraftThrusterAccel()],
    )

    assert traj.v[-1, 0] > 0.0
    assert traj.omega[-1, 2] < 0.0


def test_spacecraft_propagate_accepts_magnetic_torque_model():
    body = SpacecraftBody.cubesat(1, mass=10.0).with_magnetic_dipoles(
        MagneticDipole(moment_body=[1.0, 0.0, 0.0])
    )
    spacecraft = Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        q=[1.0, 0.0, 0.0, 0.0],
        omega=[0.0, 0.0, 0.0],
        body=body,
    )

    traj = spacecraft.propagate(
        times=[0.0, 1.0],
        mu=0.0,
        models=[SpacecraftMagneticTorque([0.0, 1.0e-5, 0.0])],
    )

    assert traj.omega[-1, 2] > 0.0


def test_spacecraft_propagate_accepts_reaction_wheels_and_pd_controller():
    body = SpacecraftBody.cubesat(1, mass=10.0).with_reaction_wheels(
        *reaction_wheel_triplet(max_torque=0.1)
    )
    spacecraft = Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        q=[1.0, 0.0, 0.0, 0.0],
        omega=[0.0, 0.0, 0.0],
        body=body,
    )

    wheel_traj = spacecraft.propagate(
        times=[0.0, 1.0],
        mu=0.0,
        models=[SpacecraftReactionWheelTorque([0.0, 0.0, 0.03])],
    )
    assert wheel_traj.omega[-1, 2] > 0.0

    perturbed_spacecraft = Spacecraft(
        r=spacecraft.r,
        v=spacecraft.v,
        q=normalize_quaternion([np.cos(0.02), 0.0, 0.0, np.sin(0.02)]),
        omega=[0.0, 0.0, 0.01],
        body=body,
    )
    perturbed = perturbed_spacecraft.propagate(
        times=[0.0, 1.0],
        mu=0.0,
        torque=SpacecraftAttitudePD(kp=0.2, kd=0.1),
    )
    assert perturbed.omega[-1, 2] < 0.01


def _is_finite_vector(value):
    value = np.asarray(value, dtype=float)
    return value.shape == (3,) and np.all(np.isfinite(value))


def test_rhs_central_gravity_matches_newtonian_acceleration():
    r = np.array([7_000_000.0, 1_000_000.0, 0.0])
    v = np.array([100.0, 7_400.0, 10.0])
    y = np.concatenate([r, v, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    dy = sixdof_rhs(0.0, y, inertia=np.eye(3))

    np.testing.assert_allclose(dy[0:3], v)
    np.testing.assert_allclose(dy[3:6], -EARTH_MU * r / np.linalg.norm(r) ** 3)
    np.testing.assert_allclose(dy[10:13], 0.0)


def test_coupled_acceleration_can_change_translation_and_accepts_orbit_like():
    acceleration = lambda t, r, v, q, omega: rotate_vector(q, [2.0e-6, 0.0, 0.0])
    orbit = SimpleNamespace(r=np.zeros(3), v=np.zeros(3), t=0.0)
    times = np.array([0.0, 10.0, 20.0])

    traj = propagate_6dof(
        orbit0=orbit,
        times=times,
        mu=0.0,
        inertia=np.eye(3),
        acceleration=acceleration,
    )

    np.testing.assert_allclose(traj.v[:, 0], 2.0e-6 * times, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(
        traj.r[:, 0],
        0.5 * 2.0e-6 * times**2,
        rtol=1e-10,
        atol=1e-14,
    )
    np.testing.assert_allclose(traj.r[:, 1:], 0.0, atol=1e-14)


def test_body_acceleration_rotates_through_current_attitude():
    q_z90 = normalize_quaternion([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    times = np.array([0.0, 10.0, 20.0])

    traj = propagate_6dof(
        r0=[0.0, 0.0, 0.0],
        v0=[0.0, 0.0, 0.0],
        q0=q_z90,
        times=times,
        mu=0.0,
        inertia=np.eye(3),
        body_acceleration=lambda t, r, v, q, omega: [2.0e-6, 0.0, 0.0],
    )

    np.testing.assert_allclose(traj.r[:, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(traj.v[:, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(traj.v[:, 1], 2.0e-6 * times, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(
        traj.r[:, 1],
        0.5 * 2.0e-6 * times**2,
        rtol=1e-10,
        atol=1e-14,
    )


def test_ntw_acceleration_uses_ssapy_component_order():
    y = np.concatenate(
        [
            [7_000_000.0, 0.0, 0.0],
            [0.0, 7_500.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    dy = sixdof_rhs(
        0.0,
        y,
        mu=0.0,
        inertia=np.eye(3),
        ntw_acceleration=lambda t, r, v, q, omega: [1.0e-6, 2.0e-6, 3.0e-6],
    )

    np.testing.assert_allclose(dy[3:6], [1.0e-6, 2.0e-6, 3.0e-6])


def test_constant_torque_changes_principal_axis_spin():
    times = np.array([0.0, 2.0, 4.0])

    traj = propagate_6dof(
        r0=[0.0, 0.0, 0.0],
        v0=[0.0, 0.0, 0.0],
        times=times,
        mu=0.0,
        inertia=np.diag([2.0, 3.0, 4.0]),
        torque=lambda t, r, v, q, omega: [2.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(traj.omega[:, 0], times, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(traj.omega[:, 1:], 0.0, atol=1e-12)


def test_propagate_6dof_forwards_solve_ivp_step_controls(monkeypatch):
    import ssapy_toolkit.dynamics.sixdof as sixdof_module

    captured = {}

    class FakeSolution:
        success = True
        message = "ok"
        t = np.array([0.0, 1.0])
        y = np.array(
            [
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

    def fake_solve_ivp(*args, **kwargs):
        captured.update(kwargs)
        return FakeSolution()

    monkeypatch.setattr(sixdof_module, "solve_ivp", fake_solve_ivp)

    propagate_6dof(
        r0=[0.0, 0.0, 0.0],
        v0=[1.0, 0.0, 0.0],
        times=[0.0, 1.0],
        inertia=np.eye(3),
        mu=0.0,
        max_step=0.25,
        first_step=0.1,
    )

    assert captured["max_step"] == 0.25
    assert captured["first_step"] == 0.1


def test_propagate_6dof_rejects_bad_inputs():
    with pytest.raises(ValueError, match="times"):
        propagate_6dof(r0=[0, 0, 0], v0=[0, 0, 0], times=[0.0], inertia=np.eye(3))
    with pytest.raises(ValueError, match="positive definite"):
        propagate_6dof(
            r0=[0, 0, 0],
            v0=[0, 0, 0],
            times=[0.0, 1.0],
            inertia=np.zeros((3, 3)),
        )
    with pytest.raises(ValueError, match="either orbit0"):
        propagate_6dof(
            orbit0=SimpleNamespace(r=np.zeros(3), v=np.zeros(3), t=0.0),
            r0=[0, 0, 0],
            times=[0.0, 1.0],
            inertia=np.eye(3),
        )
    with pytest.raises(ValueError, match="body_acceleration"):
        propagate_6dof(
            r0=[0, 0, 0],
            v0=[0, 0, 0],
            times=[0.0, 1.0],
            inertia=np.eye(3),
            body_acceleration=lambda t, r, v, q, omega: [1.0, 0.0],
        )
    with pytest.raises(ValueError, match="ntw_acceleration"):
        propagate_6dof(
            r0=[1.0, 0.0, 0.0],
            v0=[0.0, 1.0, 0.0],
            times=[0.0, 1.0],
            inertia=np.eye(3),
            ntw_acceleration=lambda t, r, v, q, omega: [1.0, 0.0],
        )
