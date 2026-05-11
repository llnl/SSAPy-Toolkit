import numpy as np


def accel_deltav_vector(dv, duration):
    """
    Compute the constant acceleration vector needed to achieve a given Δv in a set time.

    Parameters
    ----------
    dv : array_like, shape (3,)
        Desired change in velocity vector (m/s).
    duration : float
        Time over which to apply the acceleration (s).

    Returns
    -------
    thrust : float
        Magnitude of the required acceleration (m/s²).
    direction : ndarray, shape (3,)
        Unit vector indicating acceleration direction.
    a_vec : ndarray, shape (3,)
        Acceleration vector (m/s²) that, when applied constantly over `duration`, yields `dv`.

    Raises
    ------
    ValueError
        If `duration` is zero or negative.
    """
    dv = np.asarray(dv, dtype=float)
    if duration <= 0:
        raise ValueError("Duration must be positive and non-zero.")
    dv_norm = np.linalg.norm(dv)
    if dv_norm == 0:
        return 0.0, np.zeros(3), np.zeros(3)

    thrust = dv_norm / duration
    direction = dv / dv_norm
    a_vec = thrust * direction
    return thrust, direction, a_vec
