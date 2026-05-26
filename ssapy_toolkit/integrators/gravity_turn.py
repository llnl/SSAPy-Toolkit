import numpy as np
from ..constants import EARTH_MU  # gravitational parameter

def accel_gravity_turn(r, t_idx, t_array, thrust_mags, turn_time, launch_az=0.0):
    """
    r : 3‑vector position (m)
    t_idx : integer index of current step
    t_array : full time array (s)
    thrust_mags : array of thrust magnitudes (m/s²)
    turn_time : time over which pitch goes from vertical (90°) to horizontal (0°)
    launch_az : heading around Earth’s axis (rad; 0 = +x, toward +y)
    """

    # gravity
    r_norm = np.linalg.norm(r)
    a_grav = -EARTH_MU * r / r_norm**3

    # compute instantaneous pitch angle (linear schedule)
    t = t_array[t_idx] - t_array[0]
    pitch = np.clip(np.pi/2 * (1 - t/turn_time), 0, np.pi/2)

    # thrust direction in local launch plane
    # start vertical: (0,0,1), end horizontal: (cos(launch_az), sin(launch_az), 0)
    vert = np.array([0, 0, 1])
    horz = np.array([np.cos(launch_az), np.sin(launch_az), 0])
    thrust_dir = np.sin(pitch)*vert + np.cos(pitch)*horz
    thrust_dir /= np.linalg.norm(thrust_dir)

    return a_grav + thrust_mags[t_idx] * thrust_dir
