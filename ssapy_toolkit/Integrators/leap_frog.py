# ssapy_toolkit/Integrators/leap_frog.py

import numpy as np

from ..constants import EARTH_RADIUS
from ..Time_Functions import to_gps

from .int_utils import build_profile  # thrust-profile helper [104]

from ..Accelerations.accel_point_earth import accel_point_earth  # [64]
from ..Accelerations.accel_radial import accel_radial            # [65]
from ..Accelerations.accel_velocity import accel_velocity        # [68]
from ..Accelerations.accel_inclination import accel_inclination  # [61]


def leapfrog(
    r0,
    v0,
    t,
    radial=None,
    velocity=None,
    inclination=None,
    *,
    accels=None,
    stop_altitude_m=100e3,
    verbose=False,
):
    """
    Symplectic leap-frog integrator with point-mass Earth gravity + optional extra accelerations.

    Parameters
    ----------
    r0, v0 : array-like (3,)
        Initial position [m] and velocity [m/s] in an inertial frame.
    t : array-like
        Time grid (anything `to_gps` can handle). Must be evenly spaced.
    radial, velocity, inclination : profile spec or None
        Thrust-acceleration profiles (m/s^2). See `build_profile` [104].
        - radial      uses accel_radial(r, magnitude) [65]
        - velocity    uses accel_velocity(v, thrust_mag) [68]
        - inclination uses accel_inclination(r, v, magnitude) [61]
    accels : callable or list[callable] or None
        Optional additional acceleration models to add each step.
        Each function may have signature f(r), f(r,t), f(r,v), or f(r,v,t)
        and must return a (3,) acceleration vector [m/s^2].
    stop_altitude_m : float
        Stop integration if ||r|| < EARTH_RADIUS + stop_altitude_m [104].
    verbose : bool
        Print impact message.

    Returns
    -------
    r, v : ndarray (n,3)
        State history up to (and including) the first impact step, or full length.
    """
    # ---- time array (seconds since t[0]) ----
    t_arr = np.asarray(to_gps(t), dtype=float)
    t_arr -= t_arr[0]
    n_steps = len(t_arr)

    if n_steps < 2:
        raise ValueError("t must contain at least 2 time samples")

    dt_vals = np.diff(t_arr)
    if not np.allclose(dt_vals, dt_vals[0]):
        raise ValueError("Non-uniform Δt not supported")
    dt = float(dt_vals[0])

    # ---- burn profiles ----
    r_th = build_profile(radial,      t_arr)
    v_th = build_profile(velocity,    t_arr)
    i_th = build_profile(inclination, t_arr)

    # ---- normalize accels -> list ----
    if accels is None:
        accel_list = []
    elif callable(accels):
        accel_list = [accels]
    else:
        accel_list = list(accels)

    def _eval_extra_accels(r_i, v_i, t_i):
        """Sum extra accelerations, supporting several common call signatures."""
        if not accel_list:
            return np.zeros(3, dtype=float)

        a = np.zeros(3, dtype=float)
        for f in accel_list:
            # Try (r, v, t) then (r, t) then (r, v) then (r)
            try:
                a += np.asarray(f(r_i, v_i, t_i), dtype=float).reshape(3)
                continue
            except TypeError:
                pass
            try:
                a += np.asarray(f(r_i, t_i), dtype=float).reshape(3)
                continue
            except TypeError:
                pass
            try:
                a += np.asarray(f(r_i, v_i), dtype=float).reshape(3)
                continue
            except TypeError:
                pass

            a += np.asarray(f(r_i), dtype=float).reshape(3)

        return a

    # ---- state arrays ----
    r = np.empty((n_steps, 3), dtype=float)
    v = np.empty((n_steps, 3), dtype=float)
    r[0] = np.asarray(r0, dtype=float).reshape(3)
    v[0] = np.asarray(v0, dtype=float).reshape(3)

    # ---- leap-frog loop ----
    r_stop = float(EARTH_RADIUS + stop_altitude_m)

    for i in range(n_steps - 1):
        if np.linalg.norm(r[i]) < r_stop:
            if verbose:
                print(f"Impact at step {i}, t = {t_arr[i]:.2f} s")
            return r[: i + 1], v[: i + 1]

        # first half-kick
        a0 = (
            accel_point_earth(r[i])                          # [64]
            + accel_radial(r[i],            r_th[i])         # [65]
            + accel_velocity(v[i],          v_th[i])         # [68]
            + accel_inclination(r[i], v[i], i_th[i])         # [61]
            + _eval_extra_accels(r[i], v[i], t_arr[i])
        )
        v_half = v[i] + 0.5 * dt * a0  # [104]

        # drift
        r[i + 1] = r[i] + dt * v_half

        # second half-kick
        a1 = (
            accel_point_earth(r[i + 1])                          # [64]
            + accel_radial(r[i + 1],            r_th[i + 1])     # [65]
            + accel_velocity(v_half,            v_th[i + 1])     # [68]
            + accel_inclination(r[i + 1], v_half, i_th[i + 1])   # [61]
            + _eval_extra_accels(r[i + 1], v_half, t_arr[i + 1])
        )
        v[i + 1] = v_half + 0.5 * dt * a1  # [104]

    return r, v
