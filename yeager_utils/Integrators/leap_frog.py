import numpy as np
from ..constants import EARTH_RADIUS
from ..Time_Functions import to_gps
from .int_utils import (
    accel_point_earth,   # GM / r² gravity
    build_profile,       # thrust-profile helper
    accel_radial,
    accel_velocity,
    accel_inclination,
)

def leapfrog(
    r0,
    v0,
    t,
    radial=None,
    velocity=None,
    inclination=None,
):
    """
    Symplectic leap-frog integrator with point-mass Earth gravity.

    Parameters
    ----------
    r0, v0 : array-like (3,)
        Initial position [m] and velocity [m s⁻¹] in an inertial frame.
    t : array-like
        Time grid (anything `to_gps` can handle). Must be evenly spaced.
    radial, velocity, inclination : profile spec or None
        Thrust-acceleration profiles (m s⁻²).  See `build_profile`.

    Returns
    -------
    r, v : ndarray (n,3)
        State history up to (and including) the first impact step, or full
        length if no impact occurs.
    """
    # ---- time array (seconds since t[0]) -----------------------------------
    t_arr = to_gps(t).astype(float)
    t_arr -= t_arr[0]
    n_steps = len(t_arr)

    dt_vals = np.diff(t_arr)
    if not np.allclose(dt_vals, dt_vals[0]):
        raise ValueError("Non-uniform Δt not supported")
    dt = dt_vals[0]

    # ---- burn profiles -----------------------------------------------------
    r_th = build_profile(radial,      t_arr)
    v_th = build_profile(velocity,    t_arr)
    i_th = build_profile(inclination, t_arr)

    # ---- state arrays ------------------------------------------------------
    r = np.empty((n_steps, 3), dtype=float)
    v = np.empty((n_steps, 3), dtype=float)
    r[0] = np.asarray(r0, float)
    v[0] = np.asarray(v0, float)

    # ---- leap-frog loop ----------------------------------------------------
    for i in range(n_steps - 1):
        # stop on impact
        if np.linalg.norm(r[i]) < EARTH_RADIUS + 100e3:
            print(f"Impact at step {i}, t = {t_arr[i]:.2f} s")
            return r[: i + 1], v[: i + 1]

        # first half-kick
        a0 = (
            accel_point_earth(r[i])
            + accel_radial(r[i],            r_th[i])
            + accel_velocity(v[i],          v_th[i])
            + accel_inclination(r[i], v[i], i_th[i])
        )
        v_half = v[i] + 0.5 * dt * a0

        # drift
        r[i + 1] = r[i] + dt * v_half

        # second half-kick
        a1 = (
            accel_point_earth(r[i + 1])
            + accel_radial(r[i + 1],            r_th[i + 1])
            + accel_velocity(v_half,            v_th[i + 1])
            + accel_inclination(r[i + 1], v_half, i_th[i + 1])
        )
        v[i + 1] = v_half + 0.5 * dt * a1

    return r, v
