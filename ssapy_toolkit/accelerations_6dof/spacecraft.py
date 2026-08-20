"""Spacecraft acceleration models for 6-DoF propagation.

These classes intentionally mirror SSAPy's acceleration-call style while also
accepting a :class:`ssapy_toolkit.dynamics.Spacecraft`-like object.
"""

from __future__ import annotations

from collections.abc import Callable as _Callable

import numpy as _np

from ..constants import AU, EARTH_MU, EARTH_RADIUS, J2_wgs, WGS84_EARTH_OMEGA, c
from ..coordinates.satellite_frames import frame_to_gcrf_matrix

ArrayLike = _np.ndarray | list[float] | tuple[float, ...]
SOLAR_FLUX_1_AU = 1361.0


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


def _rotate_vector(q, vector) -> _np.ndarray:
    q = _normalize_quaternion(q)
    q_conj = _np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)
    rotated = _quaternion_multiply(_quaternion_multiply(q, [0.0, *_as_vector3(vector, "vector")]), q_conj)
    return rotated[1:]


def _validate_positive(value: float, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def _value_or_spacecraft(value, spacecraft, name: str) -> float:
    if value is not None:
        return value
    if spacecraft is not None and getattr(spacecraft, name, None) is not None:
        return _validate_positive(getattr(spacecraft, name), name)
    raise ValueError(f"{name} must be provided by the model or Spacecraft.")


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
