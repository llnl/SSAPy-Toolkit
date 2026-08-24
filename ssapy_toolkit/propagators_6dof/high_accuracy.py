"""High-accuracy 6-DoF propagation helpers."""

import numpy as np

from .sixdof import SixDOFTrajectory
from .sixdof import propagate_6dof as _propagate_6dof

__all__ = [
    "propagate_6dof_high_accuracy",
    "propagate_spacecraft_high_accuracy",
    "propagate_spacecraft_segments",
]


def propagate_6dof_high_accuracy(**kwargs):
    """Call :func:`ssapy_toolkit.propagators_6dof.propagate_6dof` with high-accuracy defaults."""

    kwargs.setdefault("method", "DOP853")
    kwargs.setdefault("rtol", 1e-10)
    kwargs.setdefault("atol", 1e-9)
    return _propagate_6dof(**kwargs)


def propagate_spacecraft_high_accuracy(
    spacecraft,
    *,
    times,
    models=(),
    environment=None,
    environment_models: bool | str | dict = False,
    ssapy_perturbations: bool | dict = False,
    gravity_gradient: bool = False,
    **kwargs,
):
    """Propagate a :class:`~ssapy_toolkit.propagators_6dof.Spacecraft` with high-accuracy defaults.

    ``models`` may contain any SSATK acceleration/torque/mass-flow models. Pass
    a ``SpaceEnvironment`` and ``environment_models=True`` to add environment
    backed facet drag and solar-radiation pressure, or pass a preset string
    such as ``"leo"``, ``"earth_orbit"``, ``"cislunar"``, or ``"all"``. Set
    ``ssapy_perturbations=True`` to prepend SSAPy's mature translational
    perturbation stack. Pass dictionaries to customize either option.
    """

    from ..accelerations_6dof import make_ssapy_perturbation_acceleration
    from ..environment import SpaceEnvironment

    model_list = list(models or ())
    environment_flags = {}
    if environment_models:
        if environment is None:
            environment = SpaceEnvironment()
        environment_options = _environment_model_options(environment_models)
        environment_options.setdefault("body", getattr(spacecraft, "body", None))
        environment_model_list = environment.force_models(**environment_options)
        environment_flags = _environment_model_flags(environment_model_list)
        model_list[:0] = environment_model_list
    if ssapy_perturbations:
        options = {} if ssapy_perturbations is True else dict(ssapy_perturbations)
        options.setdefault("spacecraft_kwargs", _spacecraft_physical_kwargs(spacecraft))
        if environment_flags.get("solar_radiation"):
            options.setdefault("include_solar_radiation", False)
        if environment_flags.get("drag"):
            options.setdefault("include_drag", False)
        model_list.insert(0, make_ssapy_perturbation_acceleration(**options))
    kwargs.setdefault("method", "DOP853")
    kwargs.setdefault("rtol", 1e-10)
    kwargs.setdefault("atol", 1e-9)
    return spacecraft.propagate(
        times=times,
        models=model_list,
        gravity_gradient=gravity_gradient,
        **kwargs,
    )


def propagate_spacecraft_segments(spacecraft, segments, **defaults):
    """Propagate consecutive high-accuracy spacecraft segments.

    Each segment is a mapping with ``times`` plus any
    :func:`propagate_spacecraft_high_accuracy` keyword. Segment values override
    ``defaults``. The first time in each segment must equal the current
    spacecraft epoch so gaps are explicit.
    """

    segments = [dict(defaults, **dict(segment)) for segment in segments]
    tracks_mass = spacecraft.mass is not None and any(
        "mass_flow_rate" in segment
        or any(hasattr(model, "mass_flow_rate") for model in (segment.get("models") or ()))
        for segment in segments
    )
    trajectories = []
    current = spacecraft
    for segment in segments:
        options = dict(segment)
        if "times" not in options:
            raise ValueError("each segment must define times.")
        times = np.asarray(options["times"], dtype=float)
        if times.ndim != 1 or times.size < 2:
            raise ValueError("each segment times must be a 1-D array with at least two entries.")
        if not np.isclose(times[0], current.t):
            raise ValueError("each segment must start at the current spacecraft epoch.")
        if tracks_mass:
            options.setdefault("mass0", current.mass)
        trajectory = propagate_spacecraft_high_accuracy(current, **options)
        trajectories.append(trajectory)
        current = trajectory.spacecraft(
            inertia=current.inertia,
            mass=current.mass if trajectory.mass is None else None,
            area=current.area,
            cd=current.cd,
            cr=current.cr,
            center_of_pressure=current.center_of_pressure,
            body=current.body,
        )
        if trajectory.status == 1:
            break
    if not trajectories:
        raise ValueError("segments must contain at least one segment.")
    return _combine_trajectories(trajectories)


def _spacecraft_physical_kwargs(spacecraft) -> dict:
    mapping = {"mass": "mass", "area": "area", "cd": "CD", "cr": "CR"}
    return {
        key: float(getattr(spacecraft, attr))
        for attr, key in mapping.items()
        if getattr(spacecraft, attr, None) is not None
    }


def _environment_model_options(environment_models) -> dict:
    if environment_models is True:
        return {"drag": True, "solar_radiation": True}
    if isinstance(environment_models, str):
        return {"preset": environment_models}
    return dict(environment_models)


def _environment_model_flags(models) -> dict[str, bool]:
    from ..accelerations_6dof import (
        SpacecraftAccelDrag,
        SpacecraftAccelSolRad,
        SpacecraftFacetDrag,
        SpacecraftFacetSolRad,
    )

    return {
        "drag": any(isinstance(model, (SpacecraftAccelDrag, SpacecraftFacetDrag)) for model in models),
        "solar_radiation": any(isinstance(model, (SpacecraftAccelSolRad, SpacecraftFacetSolRad)) for model in models),
    }


def _combine_trajectories(trajectories) -> SixDOFTrajectory:
    slices = [slice(None)]
    slices.extend(slice(1, None) for _ in trajectories[1:])
    t = np.concatenate([trajectory.t[index] for trajectory, index in zip(trajectories, slices)])
    r = np.vstack([trajectory.r[index] for trajectory, index in zip(trajectories, slices)])
    v = np.vstack([trajectory.v[index] for trajectory, index in zip(trajectories, slices)])
    q = np.vstack([trajectory.q[index] for trajectory, index in zip(trajectories, slices)])
    omega = np.vstack([trajectory.omega[index] for trajectory, index in zip(trajectories, slices)])
    mass = (
        np.concatenate([trajectory.mass[index] for trajectory, index in zip(trajectories, slices)])
        if all(trajectory.mass is not None for trajectory in trajectories)
        else None
    )
    wheel_momentum = (
        np.vstack([
            trajectory.wheel_momentum[index]
            for trajectory, index in zip(trajectories, slices)
        ])
        if all(trajectory.wheel_momentum is not None for trajectory in trajectories)
        else None
    )
    t_events = tuple(event for trajectory in trajectories for event in (trajectory.t_events or ()))
    y_events = tuple(event for trajectory in trajectories for event in (trajectory.y_events or ()))
    return SixDOFTrajectory(
        t=t,
        r=r,
        v=v,
        q=q,
        omega=omega,
        mass=mass,
        wheel_momentum=wheel_momentum,
        nfev=sum(trajectory.nfev for trajectory in trajectories),
        message="; ".join(trajectory.message for trajectory in trajectories if trajectory.message),
        status=trajectories[-1].status,
        t_events=t_events or None,
        y_events=y_events or None,
    )
