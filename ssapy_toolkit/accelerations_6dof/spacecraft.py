"""Spacecraft acceleration models for 6-DoF propagation.

These classes intentionally mirror SSAPy's acceleration-call style while also
accepting a :class:`ssapy_toolkit.dynamics.Spacecraft`-like object.
"""

from __future__ import annotations

from collections.abc import Callable as _Callable

import numpy as _np

from ..constants import AU, EARTH_MU, EARTH_RADIUS, SOLAR_FLUX_1_AU, WGS84_EARTH_OMEGA, J2_wgs, c
from ..coordinates.satellite_frames import frame_to_gcrf_matrix

ArrayLike = _np.ndarray | list[float] | tuple[float, ...]


class SpacecraftAccel:
    """Base class for SSATK spacecraft accelerations.

    Calls accept any of these forms:

    - ``accel(spacecraft)``
    - ``accel(r, v, t)`` matching SSAPy's ``Accel`` convention
    - ``accel(t, r, v, q, omega)`` matching ``propagate_6dof``
    """

    spacecraft_acceleration_model = True

    def __call__(self, *args, **kwargs) -> _np.ndarray:
        spacecraft, t, r, v, q, omega = _parse_state_args(args, kwargs)
        return _as_vector3(
            self.acceleration(t=t, r=r, v=v, q=q, omega=omega, spacecraft=spacecraft),
            self.__class__.__name__,
        )

    def acceleration(self, *, t, r, v, q, omega, spacecraft=None) -> _np.ndarray:
        raise NotImplementedError


class SpacecraftAccelKepler(SpacecraftAccel):
    """Point-mass central gravity, equivalent in scope to SSAPy ``AccelKepler``."""

    def __init__(self, mu: float = EARTH_MU):
        self.mu = float(mu)

    def acceleration(self, *, t, r, v, q, omega, spacecraft=None) -> _np.ndarray:
        radius = _np.linalg.norm(r)
        if self.mu == 0.0 or radius == 0.0:
            return _np.zeros(3)
        return -self.mu * r / radius**3


class SpacecraftAccelJ2(SpacecraftAccel):
    """Earth J2 perturbing acceleration."""

    def __init__(
        self,
        *,
        mu: float = EARTH_MU,
        radius: float = EARTH_RADIUS,
        j2: float = J2_wgs,
    ):
        self.mu = float(mu)
        self.radius = float(radius)
        self.j2 = float(j2)

    def acceleration(self, *, t, r, v, q, omega, spacecraft=None) -> _np.ndarray:
        return j2_acceleration(r, mu=self.mu, radius=self.radius, j2=self.j2)


class SpacecraftAccelThirdBody(SpacecraftAccel):
    """Point-mass third-body perturbation in an Earth-centered frame."""

    def __init__(self, body_position, body_mu: float):
        self.body_position = body_position
        self.body_mu = float(body_mu)

    def acceleration(self, *, t, r, v, q, omega, spacecraft=None) -> _np.ndarray:
        return third_body_acceleration(
            r,
            _vector_or_model(self.body_position, t, r, v, q, omega, spacecraft, "body_position"),
            self.body_mu,
        )


class SpacecraftAccelDrag(SpacecraftAccel):
    """Cannonball atmospheric drag with optional mass/area from ``Spacecraft``."""

    def __init__(
        self,
        *,
        density,
        area: float | None = None,
        mass: float | None = None,
        cd: float = 2.2,
        earth_radius: float = EARTH_RADIUS,
        earth_rotation_rate: float = WGS84_EARTH_OMEGA,
    ):
        self.density = density
        self.area = None if area is None else _validate_positive(area, "area")
        self.mass = None if mass is None else _validate_positive(mass, "mass")
        self.cd = _validate_positive(cd, "cd")
        self.earth_radius = float(earth_radius)
        self.earth_rotation_rate = float(earth_rotation_rate)

    def acceleration(self, *, t, r, v, q, omega, spacecraft=None) -> _np.ndarray:
        altitude = _np.linalg.norm(r) - self.earth_radius
        density = _call_density(self.density, altitude, t, r, v, q, omega, spacecraft)
        return drag_acceleration(
            r,
            v,
            density=density,
            area=_value_or_spacecraft(self.area, spacecraft, "area"),
            mass=_value_or_spacecraft(self.mass, spacecraft, "mass"),
            cd=self.cd,
            earth_radius=self.earth_radius,
            earth_rotation_rate=self.earth_rotation_rate,
        )


class SpacecraftAccelSolRad(SpacecraftAccel):
    """Cannonball solar-radiation pressure, equivalent in scope to SSAPy ``AccelSolRad``."""

    def __init__(
        self,
        sun_position,
        *,
        area: float | None = None,
        mass: float | None = None,
        cr: float = 1.3,
        solar_flux_1au: float = SOLAR_FLUX_1_AU,
        eclipse=1.0,
    ):
        self.sun_position = sun_position
        self.area = None if area is None else _validate_positive(area, "area")
        self.mass = None if mass is None else _validate_positive(mass, "mass")
        self.cr = _validate_positive(cr, "cr")
        self.solar_flux_1au = float(solar_flux_1au)
        self.eclipse = eclipse

    def acceleration(self, *, t, r, v, q, omega, spacecraft=None) -> _np.ndarray:
        eclipse = _call_optional(self.eclipse, t, r, v, q, omega, spacecraft)
        return srp_acceleration(
            r,
            _vector_or_model(self.sun_position, t, r, v, q, omega, spacecraft, "sun_position"),
            area=_value_or_spacecraft(self.area, spacecraft, "area"),
            mass=_value_or_spacecraft(self.mass, spacecraft, "mass"),
            cr=self.cr,
            solar_flux_1au=self.solar_flux_1au,
            eclipse=eclipse,
        )


class SpacecraftAccelConstInertial(SpacecraftAccel):
    """Constant inertial acceleration."""

    def __init__(self, acceleration: ArrayLike):
        self.value = _as_vector3(acceleration, "acceleration")

    def acceleration(self, *, t, r, v, q, omega, spacecraft=None) -> _np.ndarray:
        return self.value


class SpacecraftAccelConstNTW(SpacecraftAccel):
    """Constant SSAPy-order ``[N, T, W]`` acceleration rotated into GCRF."""

    def __init__(self, accelntw: ArrayLike):
        self.accelntw = _as_vector3(accelntw, "accelntw")

    def acceleration(self, *, t, r, v, q, omega, spacecraft=None) -> _np.ndarray:
        return frame_to_gcrf_matrix("ntw", r=r, v=v) @ self.accelntw


class SpacecraftAccelConstBody(SpacecraftAccel):
    """Constant body-frame acceleration rotated by the spacecraft attitude."""

    def __init__(self, acceleration: ArrayLike):
        self.value = _as_vector3(acceleration, "acceleration")

    def acceleration(self, *, t, r, v, q, omega, spacecraft=None) -> _np.ndarray:
        return _rotate_vector(q, self.value)


class SpacecraftAccelSum(SpacecraftAccel):
    """Sum multiple spacecraft acceleration models."""

    def __init__(self, accels):
        self.accels = tuple(accel for accel in accels if accel is not None)

    def acceleration(self, *, t, r, v, q, omega, spacecraft=None) -> _np.ndarray:
        total = _np.zeros(3)
        for accel in self.accels:
            total = total + _as_vector3(
                accel(spacecraft=spacecraft, t=t, r=r, v=v, q=q, omega=omega)
                if isinstance(accel, SpacecraftAccel)
                else accel(t, r, v, q, omega),
                "acceleration",
            )
        return total


class SpacecraftTorqueSum:
    """Sum multiple body-frame torque models."""

    spacecraft_torque_model = True

    def __init__(self, torques):
        self.torques = tuple(torque for torque in torques if torque is not None)

    def __call__(self, *args, **kwargs) -> _np.ndarray:
        return self.torque(*args, **kwargs)

    def torque(self, *args, **kwargs) -> _np.ndarray:
        spacecraft, t, r, v, q, omega = _parse_state_args(args, kwargs)
        total = _np.zeros(3)
        for torque in self.torques:
            total = total + _as_vector3(
                torque.torque(spacecraft=spacecraft, t=t, r=r, v=v, q=q, omega=omega)
                if getattr(torque, "spacecraft_torque_model", False)
                else torque(t, r, v, q, omega),
                "torque",
            )
        return total


class SpacecraftFlatPlateDrag(SpacecraftAccel):
    """Attitude-dependent flat-plate atmospheric drag with optional torque."""

    spacecraft_torque_model = True

    def __init__(
        self,
        *,
        density,
        area: float | None = None,
        mass: float | None = None,
        cd: float | None = 2.2,
        normal_body: ArrayLike = (1.0, 0.0, 0.0),
        center_of_pressure: ArrayLike | None = None,
        earth_radius: float = EARTH_RADIUS,
        earth_rotation_rate: float = WGS84_EARTH_OMEGA,
    ):
        self.density = density
        self.area = None if area is None else _validate_positive(area, "area")
        self.mass = None if mass is None else _validate_positive(mass, "mass")
        self.cd = None if cd is None else _validate_positive(cd, "cd")
        self.normal_body = _unit_vector(normal_body, "normal_body")
        self.center_of_pressure = None if center_of_pressure is None else _as_vector3(center_of_pressure, "center_of_pressure")
        self.earth_radius = float(earth_radius)
        self.earth_rotation_rate = float(earth_rotation_rate)

    def acceleration(self, *, t, r, v, q, omega, spacecraft=None) -> _np.ndarray:
        acceleration, _ = self._acceleration_torque(t, r, v, q, omega, spacecraft)
        return acceleration

    def torque(self, *args, **kwargs) -> _np.ndarray:
        spacecraft, t, r, v, q, omega = _parse_state_args(args, kwargs)
        _, torque = self._acceleration_torque(t, r, v, q, omega, spacecraft)
        return torque

    def _acceleration_torque(self, t, r, v, q, omega, spacecraft):
        return flat_plate_drag_acceleration_torque(
            r,
            v,
            q,
            density=_call_density(self.density, _np.linalg.norm(r) - self.earth_radius, t, r, v, q, omega, spacecraft),
            area=_value_or_spacecraft(self.area, spacecraft, "area"),
            mass=_value_or_spacecraft(self.mass, spacecraft, "mass"),
            cd=_value_or_spacecraft(self.cd, spacecraft, "cd"),
            normal_body=self.normal_body,
            center_of_pressure=_center_of_pressure(self.center_of_pressure, spacecraft),
            earth_radius=self.earth_radius,
            earth_rotation_rate=self.earth_rotation_rate,
        )


class SpacecraftFlatPlateSolRad(SpacecraftAccel):
    """Attitude-dependent flat-plate solar-radiation pressure with optional torque."""

    spacecraft_torque_model = True

    def __init__(
        self,
        sun_position,
        *,
        area: float | None = None,
        mass: float | None = None,
        cr: float | None = 1.3,
        specular_reflectivity: float | None = None,
        diffuse_reflectivity: float | None = None,
        thermal_reemission: float = 0.0,
        normal_body: ArrayLike = (1.0, 0.0, 0.0),
        center_of_pressure: ArrayLike | None = None,
        solar_flux_1au: float = SOLAR_FLUX_1_AU,
        eclipse=1.0,
    ):
        self.sun_position = sun_position
        self.area = None if area is None else _validate_positive(area, "area")
        self.mass = None if mass is None else _validate_positive(mass, "mass")
        self.cr = None if cr is None else _validate_positive(cr, "cr")
        self.specular_reflectivity = None if specular_reflectivity is None else _unit_interval(specular_reflectivity, "specular_reflectivity")
        self.diffuse_reflectivity = None if diffuse_reflectivity is None else _unit_interval(diffuse_reflectivity, "diffuse_reflectivity")
        self.thermal_reemission = _unit_interval(thermal_reemission, "thermal_reemission")
        _validate_optical_coefficients(self.specular_reflectivity, self.diffuse_reflectivity)
        self.normal_body = _unit_vector(normal_body, "normal_body")
        self.center_of_pressure = None if center_of_pressure is None else _as_vector3(center_of_pressure, "center_of_pressure")
        self.solar_flux_1au = float(solar_flux_1au)
        self.eclipse = eclipse

    def acceleration(self, *, t, r, v, q, omega, spacecraft=None) -> _np.ndarray:
        acceleration, _ = self._acceleration_torque(t, r, v, q, omega, spacecraft)
        return acceleration

    def torque(self, *args, **kwargs) -> _np.ndarray:
        spacecraft, t, r, v, q, omega = _parse_state_args(args, kwargs)
        _, torque = self._acceleration_torque(t, r, v, q, omega, spacecraft)
        return torque

    def _acceleration_torque(self, t, r, v, q, omega, spacecraft):
        return flat_plate_srp_acceleration_torque(
            r,
            q,
            _vector_or_model(self.sun_position, t, r, v, q, omega, spacecraft, "sun_position"),
            area=_value_or_spacecraft(self.area, spacecraft, "area"),
            mass=_value_or_spacecraft(self.mass, spacecraft, "mass"),
            cr=_value_or_spacecraft(self.cr, spacecraft, "cr"),
            specular_reflectivity=self.specular_reflectivity,
            diffuse_reflectivity=self.diffuse_reflectivity,
            thermal_reemission=self.thermal_reemission,
            normal_body=self.normal_body,
            center_of_pressure=_center_of_pressure(self.center_of_pressure, spacecraft),
            solar_flux_1au=self.solar_flux_1au,
            eclipse=_call_optional(self.eclipse, t, r, v, q, omega, spacecraft),
        )


class SpacecraftFacetDrag(SpacecraftAccel):
    """Attitude-dependent drag over all facets on a spacecraft body."""

    spacecraft_torque_model = True

    def __init__(
        self,
        *,
        density,
        body=None,
        mass: float | None = None,
        earth_radius: float = EARTH_RADIUS,
        earth_rotation_rate: float = WGS84_EARTH_OMEGA,
    ):
        self.density = density
        self.body = body
        self.mass = None if mass is None else _validate_positive(mass, "mass")
        self.earth_radius = float(earth_radius)
        self.earth_rotation_rate = float(earth_rotation_rate)

    def acceleration(self, *, t, r, v, q, omega, spacecraft=None) -> _np.ndarray:
        acceleration, _ = self._acceleration_torque(t, r, v, q, omega, spacecraft)
        return acceleration

    def torque(self, *args, **kwargs) -> _np.ndarray:
        spacecraft, t, r, v, q, omega = _parse_state_args(args, kwargs)
        _, torque = self._acceleration_torque(t, r, v, q, omega, spacecraft)
        return torque

    def _acceleration_torque(self, t, r, v, q, omega, spacecraft):
        body = _body_or_spacecraft(self.body, spacecraft)
        return facet_drag_acceleration_torque(
            r,
            v,
            q,
            _facets(body),
            density=_call_density(
                self.density,
                _np.linalg.norm(r) - self.earth_radius,
                t,
                r,
                v,
                q,
                omega,
                spacecraft,
            ),
            mass=_mass_from(self.mass, spacecraft, body),
            center_of_mass=_center_of_mass(body),
            earth_radius=self.earth_radius,
            earth_rotation_rate=self.earth_rotation_rate,
        )


class SpacecraftFacetSolRad(SpacecraftAccel):
    """Attitude-dependent solar-radiation pressure over all body facets."""

    spacecraft_torque_model = True

    def __init__(
        self,
        sun_position,
        *,
        body=None,
        mass: float | None = None,
        solar_flux_1au: float = SOLAR_FLUX_1_AU,
        eclipse=1.0,
        self_shadowing: bool = False,
    ):
        self.sun_position = sun_position
        self.body = body
        self.mass = None if mass is None else _validate_positive(mass, "mass")
        self.solar_flux_1au = float(solar_flux_1au)
        self.eclipse = eclipse
        self.self_shadowing = bool(self_shadowing)

    def acceleration(self, *, t, r, v, q, omega, spacecraft=None) -> _np.ndarray:
        acceleration, _ = self._acceleration_torque(t, r, v, q, omega, spacecraft)
        return acceleration

    def torque(self, *args, **kwargs) -> _np.ndarray:
        spacecraft, t, r, v, q, omega = _parse_state_args(args, kwargs)
        _, torque = self._acceleration_torque(t, r, v, q, omega, spacecraft)
        return torque

    def _acceleration_torque(self, t, r, v, q, omega, spacecraft):
        body = _body_or_spacecraft(self.body, spacecraft)
        return facet_srp_acceleration_torque(
            r,
            q,
            _vector_or_model(self.sun_position, t, r, v, q, omega, spacecraft, "sun_position"),
            _facets(body),
            mass=_mass_from(self.mass, spacecraft, body),
            center_of_mass=_center_of_mass(body),
            solar_flux_1au=self.solar_flux_1au,
            eclipse=_call_optional(self.eclipse, t, r, v, q, omega, spacecraft),
            self_shadowing=self.self_shadowing,
        )


class SpacecraftThrusterAccel(SpacecraftAccel):
    """Acceleration and torque from body-mounted thrusters."""

    spacecraft_torque_model = True

    def __init__(self, *, body=None, throttle=1.0, thruster_names=None, mass: float | None = None):
        self.body = body
        self.throttle = throttle
        self.thruster_names = None if thruster_names is None else set(thruster_names)
        self.mass = None if mass is None else _validate_positive(mass, "mass")

    def acceleration(self, *, t, r, v, q, omega, spacecraft=None) -> _np.ndarray:
        body = _body_or_spacecraft(self.body, spacecraft)
        force_body, _ = thruster_force_torque(
            _thrusters(body, self.thruster_names),
            throttle=_call_optional(self.throttle, t, r, v, q, omega, spacecraft),
            center_of_mass=_center_of_mass(body),
        )
        return _rotate_vector(q, force_body) / _mass_from(self.mass, spacecraft, body)

    def torque(self, *args, **kwargs) -> _np.ndarray:
        spacecraft, t, r, v, q, omega = _parse_state_args(args, kwargs)
        body = _body_or_spacecraft(self.body, spacecraft)
        _, torque = thruster_force_torque(
            _thrusters(body, self.thruster_names),
            throttle=_call_optional(self.throttle, t, r, v, q, omega, spacecraft),
            center_of_mass=_center_of_mass(body),
        )
        return torque

    def mass_flow_rate(self, *args, **kwargs) -> float:
        """Return positive propellant mass flow in kg/s for selected thrusters."""

        spacecraft, t, r, v, q, omega = _parse_state_args(args, kwargs)
        body = _body_or_spacecraft(self.body, spacecraft)
        return thruster_mass_flow_rate(
            _thrusters(body, self.thruster_names),
            throttle=_call_optional(self.throttle, t, r, v, q, omega, spacecraft),
        )


class SpacecraftMagneticTorque:
    """Body-frame torque from magnetic dipoles in an inertial magnetic field."""

    spacecraft_torque_model = True

    def __init__(self, magnetic_field, *, body=None, dipole_names=None):
        self.magnetic_field = magnetic_field
        self.body = body
        self.dipole_names = None if dipole_names is None else set(dipole_names)

    def __call__(self, *args, **kwargs) -> _np.ndarray:
        return self.torque(*args, **kwargs)

    def torque(self, *args, **kwargs) -> _np.ndarray:
        spacecraft, t, r, v, q, omega = _parse_state_args(args, kwargs)
        body = _body_or_spacecraft(self.body, spacecraft)
        field_inertial = _vector_or_model(self.magnetic_field, t, r, v, q, omega, spacecraft, "magnetic_field")
        field_body = _rotate_vector(_quaternion_conjugate(q), field_inertial)
        return magnetic_dipole_torque(_magnetic_dipoles(body, self.dipole_names), field_body)


class SpacecraftReactionWheelTorque:
    """Body-frame torque from reaction wheels on a ``SpacecraftBody``.

    ``command`` may be a desired 3-vector body torque, a scalar command applied
    to every selected wheel, a ``{wheel_name: torque}`` mapping, or a callable
    returning one of those forms. Wheel commands are saturated by each wheel's
    ``max_torque``.
    """

    spacecraft_torque_model = True

    def __init__(self, command, *, body=None, wheel_names=None):
        self.command = command
        self.body = body
        self.wheel_names = None if wheel_names is None else set(wheel_names)

    def __call__(self, *args, **kwargs) -> _np.ndarray:
        return self.torque(*args, **kwargs)

    def torque(self, *args, **kwargs) -> _np.ndarray:
        spacecraft, t, r, v, q, omega = _parse_state_args(args, kwargs)
        body = _body_or_spacecraft(self.body, spacecraft)
        command = _call_optional(self.command, t, r, v, q, omega, spacecraft)
        return reaction_wheel_torque(_reaction_wheels(body, self.wheel_names), command)


class SpacecraftAttitudePD:
    """Basic body-frame quaternion PD attitude controller.

    The controller assumes SSATK's body-to-inertial quaternion convention
    ``[w, x, y, z]`` and returns body-frame torque in N m. It is intentionally
    a simple stabilizing controller, not a full flight-software/GNC stack.
    """

    spacecraft_torque_model = True

    def __init__(
        self,
        *,
        q_target: ArrayLike = (1.0, 0.0, 0.0, 0.0),
        omega_target: ArrayLike = (0.0, 0.0, 0.0),
        kp: float | ArrayLike = 1.0,
        kd: float | ArrayLike = 0.0,
        max_torque: float | ArrayLike | None = None,
    ):
        self.q_target = q_target
        self.omega_target = omega_target
        self.kp = _nonnegative_gain(kp, "kp")
        self.kd = _nonnegative_gain(kd, "kd")
        self.max_torque = None if max_torque is None else _torque_limit(max_torque)

    def __call__(self, *args, **kwargs) -> _np.ndarray:
        return self.torque(*args, **kwargs)

    def torque(self, *args, **kwargs) -> _np.ndarray:
        spacecraft, t, r, v, q, omega = _parse_state_args(args, kwargs)
        q_target = _normalize_quaternion(_call_optional(self.q_target, t, r, v, q, omega, spacecraft))
        omega_target = _vector_or_model(self.omega_target, t, r, v, q, omega, spacecraft, "omega_target")
        error = attitude_error_quaternion(q_current=q, q_target=q_target)
        torque = -self.kp * error[1:] - self.kd * (omega - omega_target)
        return _apply_torque_limit(torque, self.max_torque)


def j2_acceleration(
    r_inertial: ArrayLike,
    *,
    mu: float = EARTH_MU,
    radius: float = EARTH_RADIUS,
    j2: float = J2_wgs,
) -> _np.ndarray:
    """Earth J2 perturbing acceleration in an inertial frame aligned to Earth z."""

    r = _as_vector3(r_inertial, "r_inertial")
    r_norm = _np.linalg.norm(r)
    if r_norm == 0.0:
        return _np.zeros(3)
    z_over_r = r[2] / r_norm
    factor = 1.5 * j2 * mu * radius**2 / r_norm**5
    return factor * _np.array(
        [
            r[0] * (5.0 * z_over_r**2 - 1.0),
            r[1] * (5.0 * z_over_r**2 - 1.0),
            r[2] * (5.0 * z_over_r**2 - 3.0),
        ],
        dtype=float,
    )


def third_body_acceleration(
    r_inertial: ArrayLike,
    body_position: ArrayLike,
    body_mu: float,
) -> _np.ndarray:
    """Perturbing point-mass third-body acceleration in an Earth-centered frame."""

    r = _as_vector3(r_inertial, "r_inertial")
    r_body = _as_vector3(body_position, "body_position")
    body_radius = _np.linalg.norm(r_body)
    spacecraft_to_body = r_body - r
    spacecraft_body_distance = _np.linalg.norm(spacecraft_to_body)
    if body_radius == 0.0 or spacecraft_body_distance == 0.0:
        return _np.zeros(3)
    return body_mu * (
        spacecraft_to_body / spacecraft_body_distance**3
        - r_body / body_radius**3
    )


def drag_acceleration(
    r_inertial: ArrayLike,
    v_inertial: ArrayLike,
    *,
    density: float,
    area: float,
    mass: float,
    cd: float = 2.2,
    earth_radius: float = EARTH_RADIUS,
    earth_rotation_rate: float = WGS84_EARTH_OMEGA,
) -> _np.ndarray:
    """Cannonball atmospheric drag acceleration in m/s²."""

    r = _as_vector3(r_inertial, "r_inertial")
    v = _as_vector3(v_inertial, "v_inertial")
    density = max(float(density), 0.0)
    _validate_positive(area, "area")
    _validate_positive(mass, "mass")
    _validate_positive(cd, "cd")
    if density == 0.0:
        return _np.zeros(3)

    atmosphere_velocity = _np.cross([0.0, 0.0, earth_rotation_rate], r)
    relative_velocity = v - atmosphere_velocity
    relative_speed = _np.linalg.norm(relative_velocity)
    if relative_speed == 0.0 or _np.linalg.norm(r) < earth_radius:
        return _np.zeros(3)
    return -0.5 * density * cd * area / mass * relative_speed * relative_velocity


def srp_acceleration(
    r_inertial: ArrayLike,
    sun_position: ArrayLike,
    *,
    area: float,
    mass: float,
    cr: float = 1.3,
    solar_flux_1au: float = SOLAR_FLUX_1_AU,
    eclipse: float = 1.0,
) -> _np.ndarray:
    """Cannonball solar-radiation-pressure acceleration in m/s²."""

    r = _as_vector3(r_inertial, "r_inertial")
    sun = _as_vector3(sun_position, "sun_position")
    _validate_positive(area, "area")
    _validate_positive(mass, "mass")
    _validate_positive(cr, "cr")
    distance_vector = r - sun
    distance = _np.linalg.norm(distance_vector)
    if distance == 0.0:
        return _np.zeros(3)
    pressure = solar_flux_1au / c * (AU / distance) ** 2
    illumination = float(_np.clip(eclipse, 0.0, 1.0))
    return illumination * pressure * cr * area / mass * distance_vector / distance


def flat_plate_drag_acceleration_torque(
    r_inertial: ArrayLike,
    v_inertial: ArrayLike,
    q: ArrayLike,
    *,
    density: float,
    area: float,
    mass: float,
    cd: float = 2.2,
    normal_body: ArrayLike = (1.0, 0.0, 0.0),
    center_of_pressure: ArrayLike = (0.0, 0.0, 0.0),
    earth_radius: float = EARTH_RADIUS,
    earth_rotation_rate: float = WGS84_EARTH_OMEGA,
) -> tuple[_np.ndarray, _np.ndarray]:
    """Flat-plate drag acceleration and body-frame torque."""

    r = _as_vector3(r_inertial, "r_inertial")
    v = _as_vector3(v_inertial, "v_inertial")
    q = _normalize_quaternion(q)
    density = max(float(density), 0.0)
    area = _validate_positive(area, "area")
    mass = _validate_positive(mass, "mass")
    cd = _validate_positive(cd, "cd")
    normal_body = _unit_vector(normal_body, "normal_body")
    center_of_pressure = _as_vector3(center_of_pressure, "center_of_pressure")
    if density == 0.0 or _np.linalg.norm(r) < earth_radius:
        return _np.zeros(3), _np.zeros(3)

    atmosphere_velocity = _np.cross([0.0, 0.0, earth_rotation_rate], r)
    relative_velocity = v - atmosphere_velocity
    relative_speed = _np.linalg.norm(relative_velocity)
    if relative_speed == 0.0:
        return _np.zeros(3), _np.zeros(3)

    relative_hat = relative_velocity / relative_speed
    normal_inertial = _rotate_vector(q, normal_body)
    projected = max(0.0, float(_np.dot(normal_inertial, relative_hat)))
    if projected == 0.0:
        return _np.zeros(3), _np.zeros(3)

    force_inertial = -0.5 * density * cd * area * projected * relative_speed**2 * relative_hat
    acceleration = force_inertial / mass
    force_body = _rotate_vector(_quaternion_conjugate(q), force_inertial)
    return acceleration, _np.cross(center_of_pressure, force_body)


def flat_plate_srp_acceleration_torque(
    r_inertial: ArrayLike,
    q: ArrayLike,
    sun_position: ArrayLike,
    *,
    area: float,
    mass: float,
    cr: float = 1.3,
    specular_reflectivity: float | None = None,
    diffuse_reflectivity: float | None = None,
    thermal_reemission: float = 0.0,
    normal_body: ArrayLike = (1.0, 0.0, 0.0),
    center_of_pressure: ArrayLike = (0.0, 0.0, 0.0),
    solar_flux_1au: float = SOLAR_FLUX_1_AU,
    eclipse: float = 1.0,
) -> tuple[_np.ndarray, _np.ndarray]:
    """Flat-plate solar-radiation-pressure acceleration and body-frame torque."""

    r = _as_vector3(r_inertial, "r_inertial")
    q = _normalize_quaternion(q)
    sun = _as_vector3(sun_position, "sun_position")
    area = _validate_positive(area, "area")
    mass = _validate_positive(mass, "mass")
    cr = _validate_positive(cr, "cr")
    _validate_optical_coefficients(specular_reflectivity, diffuse_reflectivity)
    thermal_reemission = _unit_interval(thermal_reemission, "thermal_reemission")
    normal_body = _unit_vector(normal_body, "normal_body")
    center_of_pressure = _as_vector3(center_of_pressure, "center_of_pressure")

    photon_direction = r - sun
    distance = _np.linalg.norm(photon_direction)
    if distance == 0.0:
        return _np.zeros(3), _np.zeros(3)
    photon_direction = photon_direction / distance
    normal_inertial = _rotate_vector(q, normal_body)
    projected = max(0.0, float(_np.dot(normal_inertial, -photon_direction)))
    if projected == 0.0:
        return _np.zeros(3), _np.zeros(3)

    pressure = solar_flux_1au / c * (AU / distance) ** 2
    illumination = float(_np.clip(eclipse, 0.0, 1.0))
    force_inertial = _srp_force_inertial(
        pressure=pressure,
        illumination=illumination,
        area=area,
        projected=projected,
        photon_direction=photon_direction,
        normal_inertial=normal_inertial,
        cr=cr,
        specular_reflectivity=specular_reflectivity,
        diffuse_reflectivity=diffuse_reflectivity,
        thermal_reemission=thermal_reemission,
    )
    acceleration = force_inertial / mass
    force_body = _rotate_vector(_quaternion_conjugate(q), force_inertial)
    return acceleration, _np.cross(center_of_pressure, force_body)


def facet_drag_acceleration_torque(
    r_inertial: ArrayLike,
    v_inertial: ArrayLike,
    q: ArrayLike,
    facets,
    *,
    density: float,
    mass: float,
    center_of_mass: ArrayLike = (0.0, 0.0, 0.0),
    earth_radius: float = EARTH_RADIUS,
    earth_rotation_rate: float = WGS84_EARTH_OMEGA,
) -> tuple[_np.ndarray, _np.ndarray]:
    """Facet drag acceleration and body-frame torque."""

    r = _as_vector3(r_inertial, "r_inertial")
    v = _as_vector3(v_inertial, "v_inertial")
    q = _normalize_quaternion(q)
    density = max(float(density), 0.0)
    mass = _validate_positive(mass, "mass")
    center_of_mass = _as_vector3(center_of_mass, "center_of_mass")
    if density == 0.0 or _np.linalg.norm(r) < earth_radius:
        return _np.zeros(3), _np.zeros(3)

    atmosphere_velocity = _np.cross([0.0, 0.0, earth_rotation_rate], r)
    relative_velocity = v - atmosphere_velocity
    relative_speed = _np.linalg.norm(relative_velocity)
    if relative_speed == 0.0:
        return _np.zeros(3), _np.zeros(3)

    relative_hat = relative_velocity / relative_speed
    total_force_inertial = _np.zeros(3)
    total_torque_body = _np.zeros(3)
    q_conj = _quaternion_conjugate(q)
    for facet in facets:
        normal_body = _unit_vector(facet.normal_body, "facet.normal_body")
        normal_inertial = _rotate_vector(q, normal_body)
        projected = max(0.0, float(_np.dot(normal_inertial, relative_hat)))
        if projected == 0.0:
            continue
        force_inertial = (
            -0.5
            * density
            * _validate_positive(facet.cd, "facet.cd")
            * _validate_positive(facet.area, "facet.area")
            * projected
            * relative_speed**2
            * relative_hat
        )
        total_force_inertial = total_force_inertial + force_inertial
        force_body = _rotate_vector(q_conj, force_inertial)
        arm_body = _as_vector3(facet.center_of_pressure, "facet.center_of_pressure") - center_of_mass
        total_torque_body = total_torque_body + _np.cross(arm_body, force_body)
    return total_force_inertial / mass, total_torque_body


def facet_srp_acceleration_torque(
    r_inertial: ArrayLike,
    q: ArrayLike,
    sun_position: ArrayLike,
    facets,
    *,
    mass: float,
    center_of_mass: ArrayLike = (0.0, 0.0, 0.0),
    solar_flux_1au: float = SOLAR_FLUX_1_AU,
    eclipse: float = 1.0,
    self_shadowing: bool = False,
    shadow_epsilon: float = 1e-9,
) -> tuple[_np.ndarray, _np.ndarray]:
    """Facet solar-radiation-pressure acceleration and body-frame torque."""

    r = _as_vector3(r_inertial, "r_inertial")
    q = _normalize_quaternion(q)
    sun = _as_vector3(sun_position, "sun_position")
    mass = _validate_positive(mass, "mass")
    center_of_mass = _as_vector3(center_of_mass, "center_of_mass")

    photon_direction = r - sun
    distance = _np.linalg.norm(photon_direction)
    if distance == 0.0:
        return _np.zeros(3), _np.zeros(3)
    photon_direction = photon_direction / distance
    pressure = solar_flux_1au / c * (AU / distance) ** 2
    illumination = float(_np.clip(eclipse, 0.0, 1.0))
    if illumination == 0.0:
        return _np.zeros(3), _np.zeros(3)

    facets = tuple(facets)
    total_force_inertial = _np.zeros(3)
    total_torque_body = _np.zeros(3)
    q_conj = _quaternion_conjugate(q)
    sun_unit_body = _rotate_vector(q_conj, -photon_direction)
    for index, facet in enumerate(facets):
        normal_body = _unit_vector(facet.normal_body, "facet.normal_body")
        normal_inertial = _rotate_vector(q, normal_body)
        projected = max(0.0, float(_np.dot(normal_inertial, -photon_direction)))
        if projected == 0.0:
            continue
        if self_shadowing and _facet_is_shadowed(index, facet, facets, sun_unit_body, shadow_epsilon):
            continue
        force_inertial = _srp_force_inertial(
            pressure=pressure,
            illumination=illumination,
            area=_validate_positive(facet.area, "facet.area"),
            projected=projected,
            photon_direction=photon_direction,
            normal_inertial=normal_inertial,
            cr=_validate_positive(facet.cr, "facet.cr"),
            specular_reflectivity=getattr(facet, "specular_reflectivity", None),
            diffuse_reflectivity=getattr(facet, "diffuse_reflectivity", None),
            thermal_reemission=getattr(facet, "thermal_reemission", 0.0),
        )
        total_force_inertial = total_force_inertial + force_inertial
        force_body = _rotate_vector(q_conj, force_inertial)
        arm_body = _as_vector3(facet.center_of_pressure, "facet.center_of_pressure") - center_of_mass
        total_torque_body = total_torque_body + _np.cross(arm_body, force_body)
    return total_force_inertial / mass, total_torque_body


def _srp_force_inertial(
    *,
    pressure: float,
    illumination: float,
    area: float,
    projected: float,
    photon_direction: _np.ndarray,
    normal_inertial: _np.ndarray,
    cr: float,
    specular_reflectivity: float | None,
    diffuse_reflectivity: float | None,
    thermal_reemission: float,
) -> _np.ndarray:
    if illumination == 0.0:
        return _np.zeros(3)
    if specular_reflectivity is None and diffuse_reflectivity is None and thermal_reemission == 0.0:
        return illumination * pressure * cr * area * projected * photon_direction

    _validate_optical_coefficients(specular_reflectivity, diffuse_reflectivity)
    thermal_reemission = _unit_interval(thermal_reemission, "thermal_reemission")
    specular = 0.0 if specular_reflectivity is None else float(specular_reflectivity)
    diffuse = 0.0 if diffuse_reflectivity is None else float(diffuse_reflectivity)
    absorbed = max(0.0, 1.0 - specular - diffuse)
    sun_unit = -photon_direction
    normal_term = 2.0 * specular * projected + (2.0 / 3.0) * diffuse + (2.0 / 3.0) * thermal_reemission * absorbed
    return -illumination * pressure * area * projected * ((1.0 - specular) * sun_unit + normal_term * normal_inertial)


def _validate_optical_coefficients(specular_reflectivity, diffuse_reflectivity) -> None:
    specular = 0.0 if specular_reflectivity is None else _unit_interval(specular_reflectivity, "specular_reflectivity")
    diffuse = 0.0 if diffuse_reflectivity is None else _unit_interval(diffuse_reflectivity, "diffuse_reflectivity")
    if specular + diffuse > 1.0:
        raise ValueError("specular_reflectivity + diffuse_reflectivity must be <= 1.")


def _facet_is_shadowed(index: int, facet, facets: tuple, sun_unit_body: _np.ndarray, epsilon: float) -> bool:
    vertices = _facet_vertices(facet)
    if vertices is None:
        return False
    origin = _as_vector3(facet.center_of_pressure, "facet.center_of_pressure") + float(epsilon) * sun_unit_body
    for blocker_index, blocker in enumerate(facets):
        if blocker_index == index:
            continue
        blocker_vertices = _facet_vertices(blocker)
        if blocker_vertices is None:
            continue
        if _np.dot(_as_vector3(blocker.center_of_pressure, "blocker.center_of_pressure") - origin, sun_unit_body) <= 0.0:
            continue
        for i in range(1, len(blocker_vertices) - 1):
            if _ray_intersects_triangle(origin, sun_unit_body, blocker_vertices[0], blocker_vertices[i], blocker_vertices[i + 1], epsilon):
                return True
    return False


def _facet_vertices(facet):
    vertices = getattr(facet, "vertices_body", None)
    if vertices is None:
        return None
    vertices = _np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
        return None
    return vertices


def _ray_intersects_triangle(origin, direction, v0, v1, v2, epsilon) -> bool:
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = _np.cross(direction, edge2)
    det = float(_np.dot(edge1, h))
    if abs(det) <= epsilon:
        return False
    inv_det = 1.0 / det
    s = origin - v0
    u = inv_det * float(_np.dot(s, h))
    if u < -epsilon or u > 1.0 + epsilon:
        return False
    q = _np.cross(s, edge1)
    v = inv_det * float(_np.dot(direction, q))
    if v < -epsilon or u + v > 1.0 + epsilon:
        return False
    t = inv_det * float(_np.dot(edge2, q))
    return t > epsilon


def thruster_force_torque(thrusters, *, throttle=1.0, center_of_mass: ArrayLike = (0.0, 0.0, 0.0)):
    """Return total body-frame force and torque from thrusters."""

    center_of_mass = _as_vector3(center_of_mass, "center_of_mass")
    total_force_body = _np.zeros(3)
    total_torque_body = _np.zeros(3)
    for thruster in thrusters:
        force_body = thruster.force_body(throttle)
        total_force_body = total_force_body + force_body
        total_torque_body = total_torque_body + _np.cross(
            _as_vector3(thruster.position_body, "thruster.position_body") - center_of_mass,
            force_body,
        )
    return total_force_body, total_torque_body


def thruster_mass_flow_rate(thrusters, *, throttle=1.0) -> float:
    """Return summed positive propellant mass flow in kg/s."""

    return float(sum(thruster.mass_flow_rate(throttle) for thruster in thrusters))


def magnetic_dipole_torque(dipoles, magnetic_field_body: ArrayLike):
    """Return total body-frame torque from magnetic dipoles and a body-frame field."""

    field = _as_vector3(magnetic_field_body, "magnetic_field_body")
    total_torque_body = _np.zeros(3)
    for dipole in dipoles:
        total_torque_body = total_torque_body + _np.cross(_as_vector3(dipole.moment_body, "dipole.moment_body"), field)
    return total_torque_body


def reaction_wheel_torque(wheels, command):
    """Return total saturated body-frame torque from reaction wheels."""

    wheels = tuple(wheels)
    if not wheels:
        return _np.zeros(3)
    if isinstance(command, dict):
        total = _np.zeros(3)
        for wheel in wheels:
            total = total + wheel.torque_body(command.get(getattr(wheel, "name", ""), 0.0))
        return total

    command = _np.asarray(command, dtype=float)
    if command.shape == ():
        return sum((wheel.torque_body(float(command)) for wheel in wheels), start=_np.zeros(3))
    if command.shape == (3,):
        axes = _np.column_stack([_as_vector3(wheel.axis_body, "wheel.axis_body") for wheel in wheels])
        wheel_commands = _np.linalg.lstsq(axes, command, rcond=None)[0]
        return sum((wheel.torque_body(value) for wheel, value in zip(wheels, wheel_commands)), start=_np.zeros(3))
    if command.shape == (len(wheels),):
        return sum((wheel.torque_body(value) for wheel, value in zip(wheels, command)), start=_np.zeros(3))
    raise ValueError("reaction-wheel command must be a scalar, 3-vector, per-wheel vector, or name mapping.")


def attitude_error_quaternion(q_current: ArrayLike, q_target: ArrayLike = (1.0, 0.0, 0.0, 0.0)) -> _np.ndarray:
    """Return the shortest current-relative-to-target quaternion error."""

    q_error = _quaternion_multiply(_quaternion_conjugate(q_target), _normalize_quaternion(q_current))
    if q_error[0] < 0.0:
        q_error = -q_error
    return q_error


def exponential_density_model(
    *,
    reference_density: float,
    reference_altitude: float,
    scale_height: float,
) -> _Callable[[float], float]:
    """Return ``rho(altitude_m)`` for a simple exponential atmosphere."""

    _validate_positive(reference_density, "reference_density")
    _validate_positive(scale_height, "scale_height")

    def density(altitude_m: float) -> float:
        return float(reference_density * _np.exp(-(float(altitude_m) - reference_altitude) / scale_height))

    return density


def make_j2_acceleration(**kwargs):
    return SpacecraftAccelJ2(**kwargs)


def make_third_body_acceleration(body_position, body_mu: float):
    return SpacecraftAccelThirdBody(body_position, body_mu)


def make_drag_acceleration(**kwargs):
    return SpacecraftAccelDrag(**kwargs)


def make_srp_acceleration(sun_position, **kwargs):
    return SpacecraftAccelSolRad(sun_position, **kwargs)


def make_flat_plate_drag(**kwargs):
    return SpacecraftFlatPlateDrag(**kwargs)


def make_flat_plate_srp(sun_position, **kwargs):
    return SpacecraftFlatPlateSolRad(sun_position, **kwargs)


def make_facet_drag(**kwargs):
    return SpacecraftFacetDrag(**kwargs)


def make_facet_srp(sun_position, **kwargs):
    return SpacecraftFacetSolRad(sun_position, **kwargs)


def make_thruster_acceleration(**kwargs):
    return SpacecraftThrusterAccel(**kwargs)


def make_magnetic_torque(magnetic_field, **kwargs):
    return SpacecraftMagneticTorque(magnetic_field, **kwargs)


def make_reaction_wheel_torque(command, **kwargs):
    return SpacecraftReactionWheelTorque(command, **kwargs)


def make_attitude_pd(**kwargs):
    return SpacecraftAttitudePD(**kwargs)


def constant_inertial_thrust(thrust: ArrayLike, mass: float):
    return SpacecraftAccelConstInertial(_as_vector3(thrust, "thrust") / _validate_positive(mass, "mass"))


def constant_ntw_thrust(thrust: ArrayLike, mass: float):
    acceleration = _as_vector3(thrust, "thrust") / _validate_positive(mass, "mass")
    return lambda t, r, v, q, omega: acceleration


def constant_body_thrust(thrust: ArrayLike, mass: float):
    acceleration = _as_vector3(thrust, "thrust") / _validate_positive(mass, "mass")
    return lambda t, r, v, q, omega: acceleration


def constant_body_torque(torque: ArrayLike):
    torque = _as_vector3(torque, "torque")
    return lambda t, r, v, q, omega: torque


def sum_accelerations(*models):
    return SpacecraftAccelSum(models)


def sum_torques(*models):
    return SpacecraftTorqueSum(models)


def _parse_state_args(args, kwargs):
    kwargs = dict(kwargs)
    spacecraft = kwargs.pop("spacecraft", None)
    q = kwargs.pop("q", None)
    omega = kwargs.pop("omega", None)

    if len(args) == 1 and _has_state(args[0]):
        spacecraft = args[0]
        r = kwargs.pop("r", spacecraft.r)
        v = kwargs.pop("v", spacecraft.v)
        t = kwargs.pop("t", getattr(spacecraft, "t", 0.0))
        q = getattr(spacecraft, "q", (1.0, 0.0, 0.0, 0.0)) if q is None else q
        omega = getattr(spacecraft, "omega", (0.0, 0.0, 0.0)) if omega is None else omega
    elif len(args) == 3:
        r, v, t = args
    elif len(args) == 5:
        t, r, v, q, omega = args
    elif len(args) == 0:
        r = kwargs.pop("r")
        v = kwargs.pop("v")
        t = kwargs.pop("t", 0.0)
    else:
        raise TypeError("expected accel(spacecraft), accel(r, v, t), or accel(t, r, v, q, omega)")

    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected keyword argument(s): {unexpected}")
    if q is None:
        q = (1.0, 0.0, 0.0, 0.0)
    if omega is None:
        omega = (0.0, 0.0, 0.0)
    return spacecraft, float(t), _as_vector3(r, "r"), _as_vector3(v, "v"), _normalize_quaternion(q), _as_vector3(omega, "omega")


def _has_state(value) -> bool:
    return hasattr(value, "r") and hasattr(value, "v")


def _as_vector3(value: ArrayLike, name: str) -> _np.ndarray:
    vector = _np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector.")
    return vector


def _normalize_quaternion(q: ArrayLike) -> _np.ndarray:
    q = _np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("q must be a 4-vector [w, x, y, z].")
    norm = _np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("q must be non-zero.")
    return q / norm


def _quaternion_multiply(q1, q2) -> _np.ndarray:
    w1, x1, y1, z1 = _np.asarray(q1, dtype=float)
    w2, x2, y2, z2 = _np.asarray(q2, dtype=float)
    return _np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def _quaternion_conjugate(q) -> _np.ndarray:
    q = _normalize_quaternion(q)
    return _np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _rotate_vector(q, vector) -> _np.ndarray:
    q = _normalize_quaternion(q)
    q_conj = _np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)
    rotated = _quaternion_multiply(_quaternion_multiply(q, [0.0, *_as_vector3(vector, "vector")]), q_conj)
    return rotated[1:]


def _unit_vector(value: ArrayLike, name: str) -> _np.ndarray:
    vector = _as_vector3(value, name)
    norm = _np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero.")
    return vector / norm


def _validate_positive(value: float, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def _unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return value


def _value_or_spacecraft(value, spacecraft, name: str) -> float:
    if value is not None:
        return value
    if spacecraft is not None and getattr(spacecraft, name, None) is not None:
        return _validate_positive(getattr(spacecraft, name), name)
    raise ValueError(f"{name} must be provided by the model or Spacecraft.")


def _center_of_pressure(value, spacecraft) -> _np.ndarray:
    if value is not None:
        return value
    if spacecraft is not None and getattr(spacecraft, "center_of_pressure", None) is not None:
        return _as_vector3(spacecraft.center_of_pressure, "center_of_pressure")
    return _np.zeros(3)


def _body_or_spacecraft(body, spacecraft):
    if body is not None:
        return body
    if spacecraft is not None and getattr(spacecraft, "body", None) is not None:
        return spacecraft.body
    raise ValueError("body must be provided by the model or Spacecraft.")


def _facets(body):
    facets = tuple(getattr(body, "facets", ()))
    if not facets:
        raise ValueError("body must define at least one facet.")
    return facets


def _thrusters(body, names=None):
    thrusters = tuple(getattr(body, "thrusters", ()))
    if names is None:
        return thrusters
    return tuple(thruster for thruster in thrusters if getattr(thruster, "name", None) in names)


def _magnetic_dipoles(body, names=None):
    dipoles = tuple(getattr(body, "magnetic_dipoles", ()))
    if names is None:
        return dipoles
    return tuple(dipole for dipole in dipoles if getattr(dipole, "name", None) in names)


def _reaction_wheels(body, names=None):
    wheels = tuple(getattr(body, "reaction_wheels", ()))
    if names is None:
        return wheels
    return tuple(wheel for wheel in wheels if getattr(wheel, "name", None) in names)


def _center_of_mass(body):
    center = getattr(body, "current_center_of_mass", getattr(body, "center_of_mass", (0.0, 0.0, 0.0)))
    return _as_vector3(center, "center_of_mass")


def _mass_from(value, spacecraft, body) -> float:
    if value is not None:
        return value
    if spacecraft is not None and getattr(spacecraft, "mass", None) is not None:
        return _validate_positive(spacecraft.mass, "mass")
    if hasattr(body, "current_mass"):
        return _validate_positive(body.current_mass, "mass")
    return _validate_positive(body.mass, "mass")


def _call_optional(value, t, r, v, q, omega, spacecraft):
    if not callable(value):
        return value
    try:
        return value(t, r, v, q, omega, spacecraft)
    except TypeError:
        return value(t, r, v, q, omega)


def _call_density(value, altitude, t, r, v, q, omega, spacecraft) -> float:
    if not callable(value):
        return value
    try:
        return value(altitude)
    except TypeError:
        return _call_optional(value, t, r, v, q, omega, spacecraft)


def _vector_or_model(value, t, r, v, q, omega, spacecraft, name: str) -> _np.ndarray:
    value = _call_optional(value, t, r, v, q, omega, spacecraft) if callable(value) else value
    return _as_vector3(value, name)


def _nonnegative_gain(value, name: str):
    gain = _np.asarray(value, dtype=float)
    if gain.shape not in ((), (3,)):
        raise ValueError(f"{name} must be a scalar or 3-vector.")
    if _np.any(gain < 0.0):
        raise ValueError(f"{name} must be non-negative.")
    return float(gain) if gain.shape == () else gain


def _torque_limit(value):
    limit = _np.asarray(value, dtype=float)
    if limit.shape not in ((), (3,)):
        raise ValueError("max_torque must be a scalar or 3-vector.")
    if _np.any(limit <= 0.0):
        raise ValueError("max_torque must be positive.")
    return float(limit) if limit.shape == () else limit


def _apply_torque_limit(torque, limit):
    torque = _as_vector3(torque, "torque")
    if limit is None:
        return torque
    if _np.isscalar(limit):
        norm = _np.linalg.norm(torque)
        return torque if norm <= limit or norm == 0.0 else torque * (limit / norm)
    return _np.clip(torque, -limit, limit)
