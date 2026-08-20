"""6-DoF translational and rigid-body attitude dynamics."""

from .sixdof import (
    BodyAccelerationModel,
    NTWAccelerationModel,
    Spacecraft,
    SixDOFState,
    SixDOFTrajectory,
    gravity_gradient_torque,
    normalize_quaternion,
    propagate_6dof,
    quaternion_conjugate,
    quaternion_multiply,
    rotate_vector,
    sixdof_rhs,
)

__all__ = [
    "BodyAccelerationModel",
    "NTWAccelerationModel",
    "Spacecraft",
    "SixDOFState",
    "SixDOFTrajectory",
    "gravity_gradient_torque",
    "normalize_quaternion",
    "propagate_6dof",
    "quaternion_conjugate",
    "quaternion_multiply",
    "rotate_vector",
    "sixdof_rhs",
]
