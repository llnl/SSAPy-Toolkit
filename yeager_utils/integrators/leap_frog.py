import numpy as np
from ..accelerations import accel_point_earth
from ..time import to_gps


def leapfrog(r0, v0, t, accel=accel_point_earth):
    """
    Integrate equations of motion using Leapfrog (velocity Verlet) method.

    Parameters:
    - r0: Initial position [x0, y0, z0] (m)
    - v0: Initial velocity [vx0, vy0, vz0] (m/s)
    - t: Time array (s)
    - dt: Time step (s)

    Returns:
    - r: Position array over time [x, y, z]
    - v: Velocity array over time [vx, vy, vz]

    Author: Travis Yeager (yeager7@llnl.gov)
    """

    t = to_gps(t)

    dt = t[1] - t[0]

    n_steps = len(t)
    r = np.zeros((n_steps, 3))
    v = np.zeros((n_steps, 3))
    r[0] = r0
    v[0] = v0

    for i in range(n_steps - 1):
        # Half-step velocity
        v_half = v[i] + 0.5 * dt * accel(r[i])
        # Full-step position
        r[i + 1] = r[i] + dt * v_half
        # Full-step velocity
        v[i + 1] = v_half + 0.5 * dt * accel(r[i + 1])

    return r, v
