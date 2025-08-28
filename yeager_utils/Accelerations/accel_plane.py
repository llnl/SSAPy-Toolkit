import numpy as np


def accel_plane(r, v, magnitude):
    """Tangential acceleration in the orbital plane, perpendicular to r̂.

    Parameters
    ----------
    r : array_like, shape (3,)
        Position vector in meters.
    v : array_like, shape (3,)
        Velocity vector in meters per second.
    magnitude : float
        Desired acceleration magnitude in m/s².
        Positive → along the direction of motion;
        negative → opposite direction.

    Returns
    -------
    a : ndarray, shape (3,)
        Tangential acceleration vector in m/s².

    Author: Travis Yeager
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    if magnitude == 0:
        return np.zeros(3)

    norm_r = np.linalg.norm(r)
    if norm_r == 0:
        return np.zeros(3)

    # Unit radial vector
    r_hat = r / norm_r

    # Remove radial component of v to get purely tangential direction
    tangential = v - np.dot(v, r_hat) * r_hat
    norm_tan = np.linalg.norm(tangential)
    if norm_tan == 0:
        return np.zeros(3)

    # Scale to desired magnitude
    return magnitude * tangential / norm_tan
