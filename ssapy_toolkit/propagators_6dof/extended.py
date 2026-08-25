"""Small linear appendage, flexible-mode, and propellant-slosh extension.

The appendage and modes are linearized about a nominal spacecraft: hinge angle
and modal displacement are generalized coordinates, while coupling loads are
reported in the spacecraft body frame.  This is not a finite-element or CFD
model; use a higher-fidelity structural model when that approximation matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..constants import EARTH_MU
from .sixdof import (
    SixDOFTrajectory,
    _initial_state,
    _times,
    _validate_time_direction,
    rotate_vector,
    sixdof_rhs,
)

__all__ = [
    "Extended6DOFTrajectory",
    "FlexibleMode",
    "HingedAppendage",
    "SloshMode",
    "propagate_6dof_extended",
]


def _vec(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 3-vector.")
    return value


@dataclass(frozen=True)
class HingedAppendage:
    """Reduced-order hinged appendage with linear and optional cubic stiffness."""

    axis_body: np.ndarray
    inertia: float
    stiffness: float = 0.0
    damping: float = 0.0
    angle0: float = 0.0
    rate0: float = 0.0
    cubic_stiffness: float = 0.0

    def __post_init__(self):
        axis = _vec(self.axis_body, "axis_body")
        inertia = float(self.inertia)
        stiffness = float(self.stiffness)
        damping = float(self.damping)
        cubic_stiffness = float(self.cubic_stiffness)
        if (np.linalg.norm(axis) == 0 or not np.isfinite(inertia) or inertia <= 0
                or not np.isfinite(stiffness) or stiffness < 0
                or not np.isfinite(damping) or damping < 0
                or not np.isfinite(cubic_stiffness) or cubic_stiffness < 0):
            raise ValueError("hinge axis must be nonzero; inertia must be positive; stiffness/damping/cubic_stiffness nonnegative.")
        object.__setattr__(self, "axis_body", axis / np.linalg.norm(axis))
        object.__setattr__(self, "inertia", inertia)
        object.__setattr__(self, "stiffness", stiffness)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "cubic_stiffness", cubic_stiffness)


@dataclass(frozen=True)
class FlexibleMode:
    axis_body: np.ndarray
    effective_mass: float
    natural_frequency: float
    damping_ratio: float = 0.0
    displacement0: float = 0.0
    velocity0: float = 0.0

    def __post_init__(self):
        axis = _vec(self.axis_body, "axis_body")
        effective_mass = float(self.effective_mass)
        natural_frequency = float(self.natural_frequency)
        damping_ratio = float(self.damping_ratio)
        if np.linalg.norm(axis) == 0 or not np.isfinite(effective_mass) or effective_mass <= 0 or not np.isfinite(natural_frequency) or natural_frequency <= 0 or not np.isfinite(damping_ratio) or damping_ratio < 0:
            raise ValueError("flexible mode requires nonzero axis, positive mass/frequency, and nonnegative damping.")
        object.__setattr__(self, "axis_body", axis / np.linalg.norm(axis))
        object.__setattr__(self, "effective_mass", effective_mass)
        object.__setattr__(self, "natural_frequency", natural_frequency)
        object.__setattr__(self, "damping_ratio", damping_ratio)


@dataclass(frozen=True)
class SloshMode:
    axis_body: np.ndarray
    mass: float
    natural_frequency: float
    damping_ratio: float = 0.0
    lever_arm_body: np.ndarray = (0.0, 0.0, 0.0)
    displacement0: float = 0.0
    velocity0: float = 0.0

    def __post_init__(self):
        axis = _vec(self.axis_body, "axis_body")
        mass = float(self.mass)
        natural_frequency = float(self.natural_frequency)
        damping_ratio = float(self.damping_ratio)
        if np.linalg.norm(axis) == 0 or not np.isfinite(mass) or mass <= 0 or not np.isfinite(natural_frequency) or natural_frequency <= 0 or not np.isfinite(damping_ratio) or damping_ratio < 0:
            raise ValueError("slosh mode requires nonzero axis, positive mass/frequency, and nonnegative damping.")
        object.__setattr__(self, "axis_body", axis / np.linalg.norm(axis))
        object.__setattr__(self, "lever_arm_body", _vec(self.lever_arm_body, "lever_arm_body"))
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "natural_frequency", natural_frequency)
        object.__setattr__(self, "damping_ratio", damping_ratio)


@dataclass(frozen=True)
class Extended6DOFTrajectory:
    trajectory: SixDOFTrajectory
    hinge: np.ndarray | None = None
    flexible: np.ndarray | None = None
    slosh: np.ndarray | None = None


def propagate_6dof_extended(*, times, inertia, hinge=None, flexible=None, slosh=None,
                            bus_mass=None, r0=None, v0=None, t0=None, q0=None, omega0=None,
                            mass0=None, mu=EARTH_MU, acceleration=None, torque=None,
                            gravity_gradient=False, rtol=1e-9, atol=1e-12, method="DOP853",
                            max_step=np.inf, first_step=None):
    """Propagate rigid-body state plus one each of hinge, flexible, and slosh modes.

    Extended arrays have columns ``[coordinate, rate]`` and are optional. Slosh
    coupling is the linear restoring force ``m*w²*x`` at ``lever_arm_body``.
    ``HingedAppendage.cubic_stiffness`` adds the restoring torque
    ``cubic_stiffness * angle**3`` in N m, and defaults to zero.
    """
    times = _times(times)
    if sum(x is not None for x in (hinge, flexible, slosh)) == 0:
        raise ValueError("at least one extended mode is required.")
    if slosh is not None:
        bus_mass = float(bus_mass) if bus_mass is not None else 0.0
        if not np.isfinite(bus_mass) or bus_mass <= 0:
            raise ValueError("positive bus_mass is required for slosh force coupling.")
    state = _initial_state(orbit0=None, r0=r0, v0=v0, t0=t0, q0=q0, omega0=omega0, mass0=mass0)
    _validate_time_direction(times, state.t)
    ext0 = []
    if hinge is not None: ext0 += [hinge.angle0, hinge.rate0]
    if flexible is not None: ext0 += [flexible.displacement0, flexible.velocity0]
    if slosh is not None: ext0 += [slosh.displacement0, slosh.velocity0]
    y0 = np.concatenate(([ *state.r, *state.v, *state.q, *state.omega] + ([] if mass0 is None else [mass0]), ext0))
    rigid_n = 14 if mass0 is not None else 13

    def rhs(t, y):
        rigid, z = y[:rigid_n], y[rigid_n:]
        i = 0; loads_t = np.zeros(3); loads_a = np.zeros(3)
        if hinge is not None:
            angle, rate = z[i:i+2]; i += 2
            loads_t += hinge.axis_body * (hinge.stiffness * angle + hinge.cubic_stiffness * angle**3 + hinge.damping * rate)
        if flexible is not None:
            disp, rate = z[i:i+2]; i += 2
            loads_t += flexible.axis_body * (flexible.effective_mass * flexible.natural_frequency**2 * disp + 2 * flexible.damping_ratio * flexible.natural_frequency * flexible.effective_mass * rate)
        if slosh is not None:
            disp, rate = z[i:i+2]
            force = slosh.axis_body * (
                slosh.mass * slosh.natural_frequency**2 * disp
                + 2 * slosh.damping_ratio * slosh.natural_frequency * slosh.mass * rate
            )
            loads_a = rotate_vector(rigid[6:10], force) / bus_mass
            loads_t = loads_t + np.cross(slosh.lever_arm_body, force)
        a = lambda tt, r, v, q, om: np.asarray(acceleration(tt, r, v, q, om) if acceleration else 0.0) + loads_a
        trq = lambda tt, r, v, q, om: np.asarray(torque(tt, r, v, q, om) if torque else 0.0) + loads_t
        base = sixdof_rhs(t, rigid, inertia=inertia, mu=mu, acceleration=a, torque=trq, gravity_gradient=gravity_gradient, mass_state=mass0 is not None)
        dz = []
        if hinge is not None: dz += [rate, -(hinge.stiffness * angle + hinge.cubic_stiffness * angle**3 + hinge.damping * rate) / hinge.inertia]
        if flexible is not None: dz += [rate, -flexible.natural_frequency**2 * disp - 2 * flexible.damping_ratio * flexible.natural_frequency * rate]
        if slosh is not None: dz += [rate, -slosh.natural_frequency**2 * disp - 2 * slosh.damping_ratio * slosh.natural_frequency * rate]
        return np.concatenate((base, dz))

    sol = solve_ivp(rhs, (times[0], times[-1]), y0, t_eval=times, rtol=rtol, atol=atol, method=method, max_step=max_step, first_step=first_step)
    if not sol.success: raise RuntimeError(sol.message)
    y = sol.y.T
    q = np.array([item / np.linalg.norm(item) for item in y[:, 6:10]])
    trajectory = SixDOFTrajectory(sol.t, y[:, :3], y[:, 3:6], q, y[:, 10:13], None if mass0 is None else y[:, 13], nfev=sol.nfev, message=sol.message, status=sol.status)
    j = rigid_n
    arrays = []
    for mode in (hinge, flexible, slosh):
        arrays.append(None if mode is None else y[:, j:j + 2])
        if mode is not None:
            j += 2
    return Extended6DOFTrajectory(trajectory, *arrays)
