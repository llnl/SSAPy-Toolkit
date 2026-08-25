"""Single-shooting finite-burn targeting for 6-DoF trajectories."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ..coordinates.satellite_frames import frame_to_gcrf_matrix
from .high_accuracy import propagate_spacecraft_segments
from .sixdof import SixDOFTrajectory

__all__ = [
    "SixDOFMultiSegmentTargetResult",
    "SixDOFTargetResult",
    "solve_6dof_multi_segment_target",
    "solve_6dof_target",
]


@dataclass(frozen=True)
class SixDOFTargetResult:
    """Result of a finite-burn 6-DoF single-shooting solve."""

    control: np.ndarray
    residual: np.ndarray
    trajectory: SixDOFTrajectory
    success: bool
    message: str
    nfev: int
    cost: float


@dataclass(frozen=True)
class SixDOFMultiSegmentTargetResult:
    """Result of a bounded multi-segment 6-DoF single-shooting solve."""

    controls: np.ndarray
    residual: np.ndarray
    trajectory: SixDOFTrajectory
    success: bool
    message: str
    nfev: int
    cost: float


class _ControlAcceleration:
    spacecraft_acceleration_model = True

    def __init__(self, control, frame: str, start: float, stop: float):
        self.control = np.asarray(control, dtype=float)
        self.frame = frame
        self.start = float(start)
        self.stop = float(stop)

    def __call__(self, *, t, r, v, q, omega, spacecraft=None):
        if not self.start <= float(t) <= self.stop:
            return np.zeros(3)
        return frame_to_gcrf_matrix(self.frame, r=r, v=v, q=q) @ self.control


def solve_6dof_target(
    spacecraft,
    *,
    times,
    target_r=None,
    target_v=None,
    control0=(0.0, 0.0, 0.0),
    control_scale=(1.0e-4, 1.0e-4, 1.0e-4),
    frame: str = "inertial",
    start: float | None = None,
    stop: float | None = None,
    models: Iterable = (),
    position_scale: float = 1.0e3,
    velocity_scale: float = 1.0,
    bounds=None,
    max_nfev: int = 100,
    propagation_kwargs: Mapping[str, object] | None = None,
) -> SixDOFTargetResult:
    """Solve a finite-burn acceleration control for a terminal state target.

    ``control0`` and the returned ``control`` are three-component accelerations
    in m/s² expressed in ``frame``. The burn is active from ``start`` through
    ``stop`` and is combined with ``models`` during each shooting propagation.
    ``target_r`` and ``target_v`` are inertial Cartesian terminal targets; at
    least one must be supplied. ``bounds`` contains lower and upper acceleration
    vectors and is passed to SciPy's bounded least-squares solver.
    """

    times = _times(times)
    target_r = None if target_r is None else _vector3(target_r, "target_r")
    target_v = None if target_v is None else _vector3(target_v, "target_v")
    if target_r is None and target_v is None:
        raise ValueError("target_r or target_v is required.")
    position_scale = _positive(position_scale, "position_scale")
    velocity_scale = _positive(velocity_scale, "velocity_scale")
    control0 = _vector3(control0, "control0")
    control_scale = _vector3(control_scale, "control_scale")
    if np.any(control_scale <= 0.0):
        raise ValueError("control_scale values must be positive.")
    start = float(times[0] if start is None else start)
    stop = float(times[-1] if stop is None else stop)
    if stop < start:
        raise ValueError("stop must be greater than or equal to start.")
    if max_nfev < 1:
        raise ValueError("max_nfev must be positive.")
    lower, upper = _bounds(bounds, control_scale)
    scaled_control0 = control0 / control_scale
    if np.any(scaled_control0 < lower) or np.any(scaled_control0 > upper):
        raise ValueError("control0 must lie within bounds.")

    base_models = tuple(models or ())
    propagation_options = dict(propagation_kwargs or {})

    def evaluate(scaled_control):
        control = np.asarray(scaled_control, dtype=float) * control_scale
        burn = _ControlAcceleration(control, frame, start, stop)
        trajectory = spacecraft.propagate(
            times=times,
            models=(*base_models, burn),
            **propagation_options,
        )
        residual = []
        if target_r is not None:
            residual.extend((trajectory.r[-1] - target_r) / position_scale)
        if target_v is not None:
            residual.extend((trajectory.v[-1] - target_v) / velocity_scale)
        return np.asarray(residual, dtype=float), trajectory

    def residual(scaled_control):
        return evaluate(scaled_control)[0]

    result = least_squares(
        residual,
        scaled_control0,
        bounds=(lower, upper),
        x_scale=1.0,
        max_nfev=int(max_nfev),
    )
    scaled_residual, trajectory = evaluate(result.x)
    physical_residual = []
    if target_r is not None:
        physical_residual.extend(trajectory.r[-1] - target_r)
    if target_v is not None:
        physical_residual.extend(trajectory.v[-1] - target_v)
    return SixDOFTargetResult(
        control=np.asarray(result.x, dtype=float) * control_scale,
        residual=np.asarray(physical_residual, dtype=float),
        trajectory=trajectory,
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
        cost=float(0.5 * np.dot(scaled_residual, scaled_residual)),
    )


def solve_6dof_multi_segment_target(
    spacecraft,
    *,
    segments,
    target_r=None,
    target_v=None,
    control0=(0.0, 0.0, 0.0),
    control_scale=(1.0e-4, 1.0e-4, 1.0e-4),
    frame: str = "inertial",
    bounds=None,
    position_scale: float = 1.0e3,
    velocity_scale: float = 1.0,
    constraints=(),
    residual_hook=None,
    max_nfev: int = 100,
    propagation_kwargs: Mapping[str, object] | None = None,
) -> SixDOFMultiSegmentTargetResult:
    """Target a terminal state with bounded controls over several segments.

    Each segment is a mapping with ``times`` plus optional overrides for
    ``models``, ``control0``, ``control_scale``, ``frame``, ``bounds``,
    ``start``, and ``stop``. Remaining keys are passed to
    :func:`propagate_spacecraft_segments`. ``constraints`` may be one callable
    or an iterable of callables; each returns normalized residuals that are
    zero when feasible. ``residual_hook(trajectory, controls)`` can append
    normalized objective residuals.
    """

    target_r = None if target_r is None else _vector3(target_r, "target_r")
    target_v = None if target_v is None else _vector3(target_v, "target_v")
    if target_r is None and target_v is None:
        raise ValueError("target_r or target_v is required.")
    position_scale = _positive(position_scale, "position_scale")
    velocity_scale = _positive(velocity_scale, "velocity_scale")
    max_nfev = int(max_nfev)
    if max_nfev < 1:
        raise ValueError("max_nfev must be positive.")
    specs = _segment_specs(segments, propagation_kwargs)
    controls0 = []
    scales = []
    lower = []
    upper = []
    for spec in specs:
        segment_control0 = _vector3(spec.pop("control0", control0), "control0")
        segment_scale = _vector3(spec.pop("control_scale", control_scale), "control_scale")
        if np.any(segment_scale <= 0.0):
            raise ValueError("control_scale values must be positive.")
        segment_bounds = spec.pop("bounds", bounds)
        segment_lower, segment_upper = _bounds(segment_bounds, segment_scale)
        scaled_control0 = segment_control0 / segment_scale
        if np.any(scaled_control0 < segment_lower) or np.any(scaled_control0 > segment_upper):
            raise ValueError("control0 must lie within bounds.")
        controls0.append(scaled_control0)
        scales.append(segment_scale)
        lower.append(segment_lower)
        upper.append(segment_upper)
    scaled_control0 = np.concatenate(controls0)
    lower = np.concatenate(lower)
    upper = np.concatenate(upper)
    constraint_hooks = (constraints,) if callable(constraints) else tuple(constraints or ())

    def evaluate(scaled_control):
        scaled_control = np.asarray(scaled_control, dtype=float).reshape(len(specs), 3)
        segment_options = []
        for index, (spec, scale) in enumerate(zip(specs, scales)):
            options = dict(spec)
            segment_frame = options.pop("frame", frame)
            segment_start = options.pop("start", None)
            segment_stop = options.pop("stop", None)
            times = np.asarray(options["times"], dtype=float)
            burn = _ControlAcceleration(
                scaled_control[index] * scale,
                segment_frame,
                times[0] if segment_start is None else segment_start,
                times[-1] if segment_stop is None else segment_stop,
            )
            options["models"] = (*tuple(options.get("models") or ()), burn)
            segment_options.append(options)
        trajectory = propagate_spacecraft_segments(spacecraft, segment_options)
        residual = []
        if target_r is not None:
            residual.extend((trajectory.r[-1] - target_r) / position_scale)
        if target_v is not None:
            residual.extend((trajectory.v[-1] - target_v) / velocity_scale)
        for hook in constraint_hooks:
            residual.extend(_hook_residual(hook(trajectory), "constraint"))
        if residual_hook is not None:
            residual.extend(_hook_residual(residual_hook(trajectory, scaled_control * np.asarray(scales)), "residual_hook"))
        return np.asarray(residual, dtype=float), trajectory

    result = least_squares(
        lambda value: evaluate(value)[0],
        scaled_control0,
        bounds=(lower, upper),
        x_scale=1.0,
        max_nfev=max_nfev,
    )
    scaled_residual, trajectory = evaluate(result.x)
    physical_residual = []
    if target_r is not None:
        physical_residual.extend(trajectory.r[-1] - target_r)
    if target_v is not None:
        physical_residual.extend(trajectory.v[-1] - target_v)
    return SixDOFMultiSegmentTargetResult(
        controls=np.asarray(result.x, dtype=float).reshape(len(specs), 3)
        * np.asarray(scales),
        residual=np.asarray(physical_residual, dtype=float),
        trajectory=trajectory,
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
        cost=float(0.5 * np.dot(scaled_residual, scaled_residual)),
    )


def _segment_specs(segments, propagation_kwargs):
    specs = [dict(propagation_kwargs or {}, **dict(segment)) for segment in segments]
    if not specs:
        raise ValueError("segments must contain at least one segment.")
    for spec in specs:
        if "times" not in spec:
            raise ValueError("each segment must define times.")
        spec["times"] = _times(spec["times"])
    return specs


def _hook_residual(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim == 0:
        value = value.reshape(1)
    else:
        value = value.ravel()
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must return finite residuals.")
    return value


def _times(times):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a 1-D array with at least two entries.")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing.")
    return times


def _vector3(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 3-vector.")
    return value


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def _bounds(bounds, control_scale):
    if bounds is None:
        return np.full(3, -np.inf), np.full(3, np.inf)
    if len(bounds) != 2:
        raise ValueError("bounds must be a (lower, upper) pair.")
    lower = _vector3(bounds[0], "lower bound") / control_scale
    upper = _vector3(bounds[1], "upper bound") / control_scale
    if np.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds.")
    return lower, upper
