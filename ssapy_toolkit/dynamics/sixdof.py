from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..constants import EARTH_MU
from ..coordinates.satellite_frames import frame_to_gcrf_matrix

ArrayLike = np.ndarray | list[float] | tuple[float, ...]
AccelerationModel = Callable[
    [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], ArrayLike
]
BodyAccelerationModel = Callable[
    [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], ArrayLike
]
NTWAccelerationModel = Callable[
    [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], ArrayLike
]
TorqueModel = Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], ArrayLike]


@dataclass(frozen=True)
class SixDOFTrajectory:
    """Integrated 6-DoF trajectory.

    Quaternion convention is ``[w, x, y, z]`` and rotates body-frame vectors
    into the inertial frame. Angular rates are body-frame rad/s.
    """

    t: np.ndarray
    r: np.ndarray
    v: np.ndarray
    q: np.ndarray
    omega: np.ndarray

    def spacecraft(
        self,
        index: int = -1,
        *,
        inertia: ArrayLike | None = None,
        mass: float | None = None,
        area: float | None = None,
        cd: float | None = None,
        cr: float | None = None,
        center_of_pressure: ArrayLike | None = None,
        body: object | None = None,
    ) -> Spacecraft:
        """Return one trajectory sample as a :class:`Spacecraft` state."""

        return Spacecraft(
            r=self.r[index],
            v=self.v[index],
            t=float(self.t[index]),
            q=self.q[index],
            omega=self.omega[index],
            inertia=inertia,
            mass=mass,
            area=area,
            cd=cd,
            cr=cr,
            center_of_pressure=center_of_pressure,
            body=body,
        )


@dataclass(frozen=True)
class SixDOFState:
    """Single 6-DoF state using the same convention as ``SixDOFTrajectory``."""

    r: np.ndarray
    v: np.ndarray
    q: np.ndarray
    omega: np.ndarray
    t: float = 0.0


@dataclass(frozen=True)
class Spacecraft:
    """Orbit-like spacecraft state with attitude and optional physical properties.

    ``r`` and ``v`` are inertial/GCRF position and velocity. ``q`` is a
    body-to-inertial quaternion ``[w, x, y, z]``. ``omega`` is the body-frame
    angular-rate vector in rad/s.
    """

    r: ArrayLike
    v: ArrayLike
    t: float = 0.0
    q: ArrayLike = (1.0, 0.0, 0.0, 0.0)
    omega: ArrayLike = (0.0, 0.0, 0.0)
    inertia: ArrayLike | None = None
    mass: float | None = None
    area: float | None = None
    cd: float | None = None
    cr: float | None = None
    center_of_pressure: ArrayLike | None = None
    body: object | None = None
    orbit: object | None = None

    def __post_init__(self) -> None:
        body = self.body
        object.__setattr__(self, "r", _as_vector3(self.r, "r"))
        object.__setattr__(self, "v", _as_vector3(self.v, "v"))
        object.__setattr__(self, "q", normalize_quaternion(self.q))
        object.__setattr__(self, "omega", _as_vector3(self.omega, "omega"))
        object.__setattr__(self, "t", float(self.t))
        if body is not None:
            if self.mass is None:
                object.__setattr__(self, "mass", _body_value(body, "current_mass", "mass"))
            if self.inertia is None:
                inertia = _body_value(body, "current_inertia", "inertia", default=None)
                if inertia is not None:
                    object.__setattr__(self, "inertia", inertia)
            if self.area is None:
                area = _body_value(body, "area", "reference_area", default=None)
                if area is not None:
                    object.__setattr__(self, "area", area)
        if self.inertia is not None:
            object.__setattr__(self, "inertia", _inertia_matrix(self.inertia))
        if self.mass is not None and self.mass <= 0.0:
            raise ValueError("mass must be positive when provided.")
        for name in ("area", "cd", "cr"):
            value = getattr(self, name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive when provided.")
        if self.center_of_pressure is not None:
            object.__setattr__(
                self,
                "center_of_pressure",
                _as_vector3(self.center_of_pressure, "center_of_pressure"),
            )

    @classmethod
    def from_orbit(
        cls,
        orbit,
        *,
        q: ArrayLike = (1.0, 0.0, 0.0, 0.0),
        omega: ArrayLike = (0.0, 0.0, 0.0),
        inertia: ArrayLike | None = None,
        mass: float | None = None,
        area: float | None = None,
        cd: float | None = None,
        cr: float | None = None,
        center_of_pressure: ArrayLike | None = None,
        body: object | None = None,
        t: float | None = None,
    ) -> Spacecraft:
        """Build a spacecraft state from an SSAPy-style object with ``r/v/t``."""

        return cls(
            r=orbit.r,
            v=orbit.v,
            t=getattr(orbit, "t", 0.0) if t is None else t,
            q=q,
            omega=omega,
            inertia=inertia,
            mass=mass,
            area=area,
            cd=cd,
            cr=cr,
            center_of_pressure=center_of_pressure,
            body=body,
            orbit=orbit,
        )

    def state(self) -> SixDOFState:
        """Return the numerical 6-DoF state."""

        return SixDOFState(
            r=self.r.copy(),
            v=self.v.copy(),
            q=self.q.copy(),
            omega=self.omega.copy(),
            t=self.t,
        )

    def with_body(self, body: object) -> Spacecraft:
        """Return a copy with a spacecraft body design attached."""

        return Spacecraft(
            r=self.r,
            v=self.v,
            t=self.t,
            q=self.q,
            omega=self.omega,
            inertia=self.inertia,
            mass=self.mass,
            area=self.area,
            cd=self.cd,
            cr=self.cr,
            center_of_pressure=self.center_of_pressure,
            body=body,
            orbit=self.orbit,
        )

    def propagate(
        self,
        *,
        times: ArrayLike,
        inertia: ArrayLike | None = None,
        body: object | None = None,
        models=None,
        **kwargs,
    ) -> SixDOFTrajectory:
        """Propagate this spacecraft using :func:`propagate_6dof`."""

        spacecraft = self.with_body(body) if body is not None else self
        if models is not None:
            if "acceleration" not in kwargs:
                kwargs["acceleration"] = _sum_model_accelerations(models)
            if "torque" not in kwargs and _any_torque_model(models):
                kwargs["torque"] = _sum_model_torques(models)
        inertia = spacecraft.inertia if inertia is None else inertia
        if inertia is None:
            raise ValueError("inertia is required to propagate a Spacecraft.")
        if "acceleration" in kwargs:
            kwargs["acceleration"] = _bind_spacecraft_acceleration(kwargs["acceleration"], spacecraft)
        if "torque" in kwargs:
            kwargs["torque"] = _bind_spacecraft_torque(kwargs["torque"], spacecraft)
        return propagate_6dof(
            orbit0=spacecraft,
            times=times,
            inertia=inertia,
            q0=spacecraft.q,
            omega0=spacecraft.omega,
            **kwargs,
        )


def normalize_quaternion(q: ArrayLike) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("q must be a 4-vector [w, x, y, z].")
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("q must be non-zero.")
    return q / norm


def quaternion_multiply(q1: ArrayLike, q2: ArrayLike) -> np.ndarray:
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    if q1.shape != (4,) or q2.shape != (4,):
        raise ValueError("quaternion operands must be 4-vectors.")
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quaternion_conjugate(q: ArrayLike) -> np.ndarray:
    q = normalize_quaternion(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def rotate_vector(q: ArrayLike, vector: ArrayLike) -> np.ndarray:
    q = normalize_quaternion(q)
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,):
        raise ValueError("vector must be a 3-vector.")
    rotated = quaternion_multiply(
        quaternion_multiply(q, [0.0, *vector]),
        quaternion_conjugate(q),
    )
    return rotated[1:]


def quaternion_from_matrix(matrix: ArrayLike) -> np.ndarray:
    """Return a body-to-inertial quaternion from a 3x3 direction-cosine matrix."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("matrix must be 3x3.")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-8):
        raise ValueError("matrix must be orthonormal.")
    if np.linalg.det(matrix) <= 0.0:
        raise ValueError("matrix must be right-handed.")

    trace = np.trace(matrix)
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            q = np.array([
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ])
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            q = np.array([
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ])
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            q = np.array([
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            ])
    return normalize_quaternion(q)


def attitude_quaternion_from_frame(frame: str, **kwargs) -> np.ndarray:
    """Return a body-to-GCRF quaternion whose body axes match ``frame`` axes."""

    return quaternion_from_matrix(frame_to_gcrf_matrix(frame, **kwargs))


def gravity_gradient_torque(
    r_inertial: ArrayLike,
    q: ArrayLike,
    inertia: ArrayLike,
    mu: float = EARTH_MU,
) -> np.ndarray:
    r_inertial = np.asarray(r_inertial, dtype=float)
    if r_inertial.shape != (3,):
        raise ValueError("r_inertial must be a 3-vector.")
    radius = np.linalg.norm(r_inertial)
    if radius == 0.0:
        return np.zeros(3)
    inertia = _inertia_matrix(inertia)
    r_hat_body = rotate_vector(quaternion_conjugate(q), r_inertial / radius)
    return 3.0 * mu / radius**3 * np.cross(r_hat_body, inertia @ r_hat_body)



def sixdof_rhs(
    t: float,
    y: ArrayLike,
    *,
    inertia: ArrayLike,
    mu: float = EARTH_MU,
    acceleration: AccelerationModel | None = None,
    ntw_acceleration: NTWAccelerationModel | None = None,
    body_acceleration: BodyAccelerationModel | None = None,
    torque: TorqueModel | None = None,
    gravity_gradient: bool = False,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.shape != (13,):
        raise ValueError("y must have 13 elements: r, v, q, omega.")
    inertia = _inertia_matrix(inertia)
    inv_inertia = np.linalg.inv(inertia)

    r = y[0:3]
    v = y[3:6]
    q = normalize_quaternion(y[6:10])
    omega = y[10:13]

    radius = np.linalg.norm(r)
    a = np.zeros(3) if mu == 0.0 or radius == 0.0 else -mu * r / radius**3
    if acceleration is not None:
        a = a + _as_vector3(acceleration(t, r, v, q, omega), "acceleration")
    if ntw_acceleration is not None:
        a = a + frame_to_gcrf_matrix("ntw", r=r, v=v) @ _as_vector3(
            ntw_acceleration(t, r, v, q, omega),
            "ntw_acceleration",
        )
    if body_acceleration is not None:
        a = a + rotate_vector(
            q,
            _as_vector3(body_acceleration(t, r, v, q, omega), "body_acceleration"),
        )

    q_dot = 0.5 * quaternion_multiply(q, [0.0, *omega])
    torque_body = np.zeros(3)
    if gravity_gradient:
        torque_body = torque_body + gravity_gradient_torque(r, q, inertia, mu=mu)
    if torque is not None:
        torque_body = torque_body + _evaluate_torque(torque, t, r, v, q, omega)
    omega_dot = inv_inertia @ (torque_body - np.cross(omega, inertia @ omega))

    return np.concatenate([v, a, q_dot, omega_dot])


def propagate_6dof(
    *,
    times: ArrayLike,
    inertia: ArrayLike,
    orbit0=None,
    r0: ArrayLike | None = None,
    v0: ArrayLike | None = None,
    t0: float | None = None,
    q0: ArrayLike = (1.0, 0.0, 0.0, 0.0),
    omega0: ArrayLike = (0.0, 0.0, 0.0),
    mu: float = EARTH_MU,
    acceleration: AccelerationModel | None = None,
    ntw_acceleration: NTWAccelerationModel | None = None,
    body_acceleration: BodyAccelerationModel | None = None,
    torque: TorqueModel | None = None,
    gravity_gradient: bool = False,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    method: str = "DOP853",
    max_step: float = np.inf,
    first_step: float | None = None,
) -> SixDOFTrajectory:
    """Propagate a coupled translational and rigid-body attitude state.

    ``acceleration`` returns inertial/GCRF m/s². ``ntw_acceleration`` returns
    SSAPy-style ``[N, T, W]`` m/s², where ``T`` follows velocity and ``W``
    follows ``r × v``. ``body_acceleration`` returns body-frame m/s² and is
    rotated into the inertial frame by the current attitude. ``torque`` returns
    body-frame N m. Use ``gravity_gradient=True`` to add the standard
    rigid-body gravity-gradient torque. Without attitude-dependent
    acceleration, attitude does not affect the orbit trajectory.
    """

    times = _times(times)
    state = _initial_state(
        orbit0=orbit0,
        r0=r0,
        v0=v0,
        t0=t0,
        q0=q0,
        omega0=omega0,
    )
    _validate_time_direction(times, state.t)
    y0 = np.concatenate([state.r, state.v, state.q, state.omega])

    sol = solve_ivp(
        lambda t, y: sixdof_rhs(
            t,
            y,
            inertia=inertia,
            mu=mu,
            acceleration=acceleration,
            ntw_acceleration=ntw_acceleration,
            body_acceleration=body_acceleration,
            torque=torque,
            gravity_gradient=gravity_gradient,
        ),
        (state.t, float(times[-1])),
        y0,
        t_eval=times,
        rtol=rtol,
        atol=atol,
        method=method,
        max_step=max_step,
        first_step=first_step,
    )
    if not sol.success:
        raise RuntimeError(f"6-DoF propagation failed: {sol.message}")

    y = sol.y.T
    q = np.array([normalize_quaternion(item) for item in y[:, 6:10]])
    return SixDOFTrajectory(t=sol.t, r=y[:, 0:3], v=y[:, 3:6], q=q, omega=y[:, 10:13])


def _initial_state(*, orbit0, r0, v0, t0, q0, omega0) -> SixDOFState:
    if orbit0 is not None:
        if r0 is not None or v0 is not None:
            raise ValueError("Provide either orbit0 or r0/v0, not both.")
        r0 = orbit0.r
        v0 = orbit0.v
        t0 = getattr(orbit0, "t", 0.0) if t0 is None else t0
    if r0 is None or v0 is None:
        raise ValueError("r0 and v0 are required when orbit0 is not provided.")
    return SixDOFState(
        r=_as_vector3(r0, "r0"),
        v=_as_vector3(v0, "v0"),
        q=normalize_quaternion(q0),
        omega=_as_vector3(omega0, "omega0"),
        t=0.0 if t0 is None else float(t0),
    )


def _times(times: ArrayLike) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a 1-D array with at least two entries.")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing.")
    return times


def _validate_time_direction(times: np.ndarray, t0: float) -> None:
    if times[0] < t0 or times[-1] < t0:
        raise ValueError("times must be at or after the initial epoch t0.")


def _as_vector3(value: ArrayLike, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector.")
    return vector


def _bind_spacecraft_acceleration(model, spacecraft):
    if getattr(model, "spacecraft_acceleration_model", False):
        return lambda t, r, v, q, omega: model(
            spacecraft=spacecraft,
            t=t,
            r=r,
            v=v,
            q=q,
            omega=omega,
        )
    return model


def _bind_spacecraft_torque(model, spacecraft):
    if getattr(model, "spacecraft_torque_model", False):
        evaluator = model.torque if hasattr(model, "torque") else model
        return lambda t, r, v, q, omega: model.torque(
            spacecraft=spacecraft,
            t=t,
            r=r,
            v=v,
            q=q,
            omega=omega,
        ) if hasattr(model, "torque") else evaluator(
            spacecraft=spacecraft,
            t=t,
            r=r,
            v=v,
            q=q,
            omega=omega,
        )
    return model


def _evaluate_torque(model, t, r, v, q, omega):
    if getattr(model, "spacecraft_torque_model", False):
        evaluator = model.torque if hasattr(model, "torque") else model
        return _as_vector3(
            evaluator(t=t, r=r, v=v, q=q, omega=omega),
            "torque",
        )
    return _as_vector3(model(t, r, v, q, omega), "torque")


def _body_value(body, *names: str, default=None):
    for name in names:
        if hasattr(body, name):
            return getattr(body, name)
    return default


def _sum_model_accelerations(models):
    models = tuple(models)

    def acceleration(*args, **kwargs):
        spacecraft, t, r, v, q, omega = _model_call_state(args, kwargs)
        total = np.zeros(3)
        for model in models:
            if getattr(model, "spacecraft_acceleration_model", False):
                total = total + _as_vector3(
                    model(spacecraft=spacecraft, t=t, r=r, v=v, q=q, omega=omega),
                    "acceleration",
                )
            elif callable(model) and not getattr(model, "spacecraft_torque_model", False):
                total = total + _as_vector3(model(t, r, v, q, omega), "acceleration")
        return total

    acceleration.spacecraft_acceleration_model = True
    return acceleration


def _sum_model_torques(models):
    models = tuple(models)

    def torque(*args, **kwargs):
        spacecraft, t, r, v, q, omega = _model_call_state(args, kwargs)
        total = np.zeros(3)
        for model in models:
            if getattr(model, "spacecraft_torque_model", False):
                evaluator = model.torque if hasattr(model, "torque") else model
                total = total + _as_vector3(
                    evaluator(spacecraft=spacecraft, t=t, r=r, v=v, q=q, omega=omega),
                    "torque",
                )
        return total

    torque.spacecraft_torque_model = True
    torque.torque = torque
    return torque


def _any_torque_model(models) -> bool:
    return any(getattr(model, "spacecraft_torque_model", False) for model in models)


def _model_call_state(args, kwargs):
    kwargs = dict(kwargs)
    spacecraft = kwargs.pop("spacecraft", None)
    if args:
        if len(args) != 5:
            raise TypeError("model calls require t, r, v, q, omega")
        t, r, v, q, omega = args
    else:
        t = kwargs.pop("t")
        r = kwargs.pop("r")
        v = kwargs.pop("v")
        q = kwargs.pop("q")
        omega = kwargs.pop("omega")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected keyword argument(s): {unexpected}")
    return spacecraft, t, r, v, q, omega


def _inertia_matrix(inertia: ArrayLike) -> np.ndarray:
    inertia = np.asarray(inertia, dtype=float)
    if inertia.shape != (3, 3):
        raise ValueError("inertia must be a 3x3 matrix.")
    if not np.allclose(inertia, inertia.T):
        raise ValueError("inertia must be symmetric.")
    if np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
        raise ValueError("inertia must be positive definite.")
    return inertia
