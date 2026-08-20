from types import SimpleNamespace

import numpy as np
import pytest

from ssapy_toolkit.constants import AU, EARTH_MU, EARTH_RADIUS, J2_wgs, WGS84_EARTH_OMEGA, c
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
    constant_body_thrust,
    constant_body_torque,
    drag_acceleration,
    exponential_density_model,
    j2_acceleration,
    srp_acceleration,
    third_body_acceleration,
)
from ssapy_toolkit.dynamics import (
    Spacecraft,
    gravity_gradient_torque,
    normalize_quaternion,
    propagate_6dof,
    rotate_vector,
    sixdof_rhs,
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


def test_quaternion_helpers_rotate_body_to_inertial():
    q_z90 = normalize_quaternion([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    np.testing.assert_allclose(
        rotate_vector(q_z90, [1.0, 0.0, 0.0]),
        [0.0, 1.0, 0.0],
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="non-zero"):
        normalize_quaternion([0.0, 0.0, 0.0, 0.0])


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
