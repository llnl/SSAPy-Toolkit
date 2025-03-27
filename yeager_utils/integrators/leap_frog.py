import numpy as np
from ..accelerations import accel_point_earth
from ..time import to_gps


def leapfrog(r0, v0, t, accel=accel_point_earth, accel_const=None):
    """
    Integrate equations of motion using Leapfrog (velocity Verlet) method with optional constant acceleration.

    Parameters
    ----------
    r0 : array_like
        Initial position [x0, y0, z0] in meters (m).
    v0 : array_like
        Initial velocity [vx0, vy0, vz0] in meters per second (m/s).
    t : array_like
        Time array in seconds (s).
    accel : callable, optional
        Acceleration function, defaults to accel_point_earth. Takes position (m) and returns acceleration (m/s^2).
    accel_const : array_like, optional
        Constant acceleration vector [ax, ay, az] in meters per second squared (m/s^2). Defaults to None (zero).

    Returns
    -------
    r : ndarray
        Position array over time [x, y, z], shape (n_steps, 3) in meters.
    v : ndarray
        Velocity array over time [vx, vy, vz], shape (n_steps, 3) in meters per second.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    t = to_gps(t)
    dt = t[1] - t[0]

    r0 = np.asarray(r0)
    v0 = np.asarray(v0)

    if accel_const is None:
        accel_const = np.zeros(3)
    else:
        accel_const = np.asarray(accel_const)
        if accel_const.shape != (3,):
            raise ValueError("accel_const must be a 3-element array.")

    n_steps = len(t)
    r = np.zeros((n_steps, 3))
    v = np.zeros((n_steps, 3))
    r[0] = r0
    v[0] = v0

    for i in range(n_steps - 1):
        a_current = accel(r[i]) + accel_const
        v_half = v[i] + 0.5 * dt * a_current

        r[i + 1] = r[i] + dt * v_half

        a_next = accel(r[i + 1]) + accel_const
        v[i + 1] = v_half + 0.5 * dt * a_next

    return r, v
