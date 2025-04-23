import numpy as np
from ..constants import EARTH_MU, EARTH_RADIUS


def accel_to_circular(r, v, thrust, tol=1e-6, min_altitude=100e3):
    """
    Compute a fixed‐magnitude acceleration vector that steers (r, v)
    optimally toward a circular orbit at radius |r|.

    The direction is chosen along the velocity‐error vector
    v_target – v, where v_target is the purely tangential
    circular speed at the current radius.  Once within tol
    in velocity error or below min_altitude, returns zero.

    Parameters
    ----------
    r : array_like, shape (3,)
        Position vector (m).
    v : array_like, shape (3,)
        Velocity vector (m/s).
    thrust : float
        Magnitude of the available acceleration (m/s²).
    tol : float, optional
        Convergence tolerance on ||v_target – v|| (default 1e-6 m/s).
    min_altitude : float, optional
        Minimum altitude above Earth's surface for control (m).

    Returns
    -------
    a : ndarray, shape (3,)
        Acceleration command (m/s²), of magnitude thrust in the
        optimal direction, or zero if converged or too low.
    
    Author: Travis Yeager
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    r_norm = np.linalg.norm(r)
    if r_norm == 0 or thrust == 0:
        return np.zeros(3)

    # altitude check
    if r_norm < EARTH_RADIUS + min_altitude:
        return np.zeros(3)

    # radial unit
    r_hat = r / r_norm

    # radial & tangential components of current velocity
    v_r = np.dot(v, r_hat)
    v_t_vec = v - v_r * r_hat
    v_t = np.linalg.norm(v_t_vec)

    # tangential unit (handle zero case)
    if v_t > 0:
        t_hat = v_t_vec / v_t
    else:
        t_hat = np.cross(np.array([0., 0., 1.]), r_hat)
        n = np.linalg.norm(t_hat)
        if n == 0:
            return np.zeros(3)
        t_hat /= n

    # desired circular speed
    v_circ = np.sqrt(EARTH_MU / r_norm)
    v_target = v_circ * t_hat

    # velocity error
    dv = v_target - v
    dv_norm = np.linalg.norm(dv)
    if dv_norm < tol:
        print('Not applying thrust')
        return np.zeros(3)

    # fixed‐magnitude thrust along error direction
    return thrust * (dv / dv_norm)
