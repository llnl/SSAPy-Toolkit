"""6-DoF translational and rigid-body attitude dynamics."""

from .sixdof import (
    BodyAccelerationModel,
    NTWAccelerationModel,
    SixDOFState,
    SixDOFTrajectory,
    Spacecraft,
    attitude_quaternion_from_frame,
    gravity_gradient_torque,
    normalize_quaternion,
    propagate_6dof,
    quaternion_conjugate,
    quaternion_from_matrix,
    quaternion_multiply,
    rotate_vector,
    sixdof_rhs,
)

__all__ = [
    "BodyAccelerationModel",
    "NTWAccelerationModel",
    "SixDOFState",
    "SixDOFTrajectory",
    "Spacecraft",
    "attitude_quaternion_from_frame",
    "gravity_gradient_torque",
    "normalize_quaternion",
    "propagate_6dof",
    "quaternion_conjugate",
    "quaternion_from_matrix",
    "quaternion_multiply",
    "rotate_vector",
    "sixdof_rhs",
]
