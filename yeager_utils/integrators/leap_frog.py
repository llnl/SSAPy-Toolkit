import numpy as np
from ..accelerations import accel_point_earth
from .accel_radial import accel_radial
from .accel_velocity import accel_velocity
from .accel_perp import accel_perp
from .accel_plane import accel_plane  # Import the accel_plane function
from ..time import to_gps


def leapfrog(r0, v0, t, accel=accel_point_earth, radial_thrust=0, velocity_thrust=0, perp_thrust=0, 
             radial_active=None, velocity_active=None, perp_active=None, plane_thrust=0, plane_active=None):
    """
    Integrate equations of motion using the velocity Verlet (Leapfrog) method
    with optional velocity-directed thrust, radial acceleration, perpendicular acceleration,
    and acceleration in the orbital plane.

    Parameters
    ----------
    r0 : array_like
        Initial position [x0, y0, z0] in meters.
    v0 : array_like
        Initial velocity [vx0, vy0, vz0] in meters per second.
    t : array_like or astropy.time.Time
        Times at which to compute the solution (s or Time).
    accel : callable, optional
        Central acceleration function, defaults to accel_point_earth.
        Should accept position array and return acceleration array (m/s^2).
    radial_thrust : float or array_like, optional
        Magnitude of the radial thrust acceleration. Scalar or array of length n_steps.
    velocity_thrust : float or array_like, optional
        Magnitude of the velocity-directed thrust acceleration. Scalar or array of length n_steps.
    perp_thrust : float or array_like, optional
        Magnitude of the perpendicular thrust acceleration. Scalar or array of length n_steps.
    plane_thrust : float or array_like, optional
        Magnitude of the thrust in the orbital plane. Scalar or array of length n_steps.
    radial_active : array_like of bool, optional
        Boolean mask indicating when radial thrust is active. Array of length n_steps.
    velocity_active : array_like of bool, optional
        Boolean mask indicating when velocity-directed thrust is active. Array of length n_steps.
    perp_active : array_like of bool, optional
        Boolean mask indicating when perpendicular thrust is active. Array of length n_steps.
    plane_active : array_like of bool, optional
        Boolean mask indicating when plane-directed thrust is active. Array of length n_steps.

    Returns
    -------
    r : ndarray, shape (n_steps, 3)
        Position history (m).
    v : ndarray, shape (n_steps, 3)
        Velocity history (m/s).

    Author: Travis Yeager
    """
    # Convert time to GPS seconds and ensure array
    t_arr = to_gps(t)
    t_arr = np.asarray(t_arr, dtype=float)
    n_steps = len(t_arr)

    # Prepare active masks
    if radial_active is None:
        radial_active = np.zeros(n_steps, dtype=bool)
    if velocity_active is None:
        velocity_active = np.zeros(n_steps, dtype=bool)
    if perp_active is None:
        perp_active = np.zeros(n_steps, dtype=bool)
    if plane_active is None:
        plane_active = np.zeros(n_steps, dtype=bool)

    # Prepare thrust magnitudes
    if np.isscalar(radial_thrust):
        radial_thrust_mags = np.full(n_steps, float(radial_thrust), dtype=float)
    else:
        radial_thrust_mags = np.asarray(radial_thrust, dtype=float)
        if radial_thrust_mags.shape != (n_steps,):
            raise ValueError("`radial_thrust` must be a scalar or array of length equal to `t`")

    if np.isscalar(velocity_thrust):
        velocity_thrust_mags = np.full(n_steps, float(velocity_thrust), dtype=float)
    else:
        velocity_thrust_mags = np.asarray(velocity_thrust, dtype=float)
        if velocity_thrust_mags.shape != (n_steps,):
            raise ValueError("`velocity_thrust` must be a scalar or array of length equal to `t`")

    if np.isscalar(perp_thrust):
        perp_thrust_mags = np.full(n_steps, float(perp_thrust), dtype=float)
    else:
        perp_thrust_mags = np.asarray(perp_thrust, dtype=float)
        if perp_thrust_mags.shape != (n_steps,):
            raise ValueError("`perp_thrust` must be a scalar or array of length equal to `t`")

    if np.isscalar(plane_thrust):
        plane_thrust_mags = np.full(n_steps, float(plane_thrust), dtype=float)
    else:
        plane_thrust_mags = np.asarray(plane_thrust, dtype=float)
        if plane_thrust_mags.shape != (n_steps,):
            raise ValueError("`plane_thrust` must be a scalar or array of length equal to `t`")

    # Apply mask to thrust magnitudes
    radial_thrust_mags *= radial_active
    velocity_thrust_mags *= velocity_active
    perp_thrust_mags *= perp_active
    plane_thrust_mags *= plane_active

    # Time step (assumes uniform spacing)
    dt_vals = np.diff(t_arr)
    if not np.allclose(dt_vals, dt_vals[0]):
        raise ValueError("Time steps must be uniform for this implementation.")
    dt = dt_vals[0]

    # Allocate arrays
    r = np.zeros((n_steps, 3), dtype=float)
    v = np.zeros((n_steps, 3), dtype=float)

    r[0] = np.asarray(r0, dtype=float)
    v[0] = np.asarray(v0, dtype=float)

    # Leapfrog integration loop
    for i in range(n_steps - 1):
        # Current acceleration (gravity + velocity-directed thrust + radial thrust + perpendicular thrust + plane thrust)
        a_curr = (
            accel(r[i]) +
            accel_velocity(v[i], velocity_thrust_mags[i]) +
            accel_radial(r[i], radial_thrust_mags[i]) +
            accel_perp(v[i], r[i], perp_thrust_mags[i]) +
            accel_plane(r[i], v[i], plane_thrust_mags[i])  # Added accel_plane
        )
        v_half = v[i] + 0.5 * dt * a_curr
        r[i + 1] = r[i] + dt * v_half

        # Next acceleration and full-step velocity update
        a_next = (
            accel(r[i + 1]) +
            accel_velocity(v_half, velocity_thrust_mags[i + 1]) +
            accel_radial(r[i + 1], radial_thrust_mags[i + 1]) +
            accel_perp(v_half, r[i + 1], perp_thrust_mags[i + 1]) +
            accel_plane(r[i + 1], v_half, plane_thrust_mags[i + 1])  # Added accel_plane
        )
        v[i + 1] = v_half + 0.5 * dt * a_next

    return r, v
