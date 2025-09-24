import numpy as np
from ..constants import EARTH_MU, EARTH_RADIUS

_orbit_achieved = False  # global variable to track circularization

def reset_orbit_status():
    global _orbit_achieved
    _orbit_achieved = False

def accel_to_circular(r, v, thrust, tol=10, min_altitude=100e3):
    """
    Compute a fixed‐magnitude acceleration vector that steers (r, v)
    optimally toward a circular orbit at radius |r|.

    The direction is chosen along the velocity‐error vector
    v_target – v. Once within tol or below min_altitude, returns zero.

    Parameters
    ----------
    r : array_like, shape (3,)
        Position vector (m).
    v : array_like, shape (3,)
        Velocity vector (m/s).
    thrust : float
        Magnitude of the available acceleration (m/s²).
    tol : float, optional
        Convergence tolerance on ||v_target – v|| (default 10 m/s).
    min_altitude : float, optional
        Minimum altitude above Earth's surface for control (m).

    Returns
    -------
    a : ndarray, shape (3,)
        Acceleration command (m/s²), or zero if converged or too low.

    Author: Travis Yeager
    """
    global _orbit_achieved
    if _orbit_achieved:
        return np.zeros(3)

    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    r_norm = np.linalg.norm(r)
    if r_norm == 0 or thrust == 0:
        return np.zeros(3)

    if r_norm < EARTH_RADIUS + min_altitude:
        return np.zeros(3)

    r_hat = r / r_norm
    v_r = np.dot(v, r_hat)
    v_t_vec = v - v_r * r_hat
    v_t = np.linalg.norm(v_t_vec)

    if v_t > 0:
        t_hat = v_t_vec / v_t
    else:
        t_hat = np.cross(np.array([0., 0., 1.]), r_hat)
        n = np.linalg.norm(t_hat)
        if n == 0:
            return np.zeros(3)
        t_hat /= n

    v_circ = np.sqrt(EARTH_MU / r_norm)
    v_target = v_circ * t_hat
    dv = v_target - v
    dv_norm = np.linalg.norm(dv)

    if dv_norm < tol:
        _orbit_achieved = True
        print("Orbit achieved, thrust off.")
        return np.zeros(3)

    return thrust * (dv / dv_norm)
