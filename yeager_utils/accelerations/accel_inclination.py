import numpy as np

def accel_inclination(r, v, magnitude):
    """
    Acceleration vector always in the local north/south direction,
    i.e. perpendicular to the radial direction but aimed toward
    the Earth’s North pole (or South if magnitude<0).

    Parameters
    ----------
    r : array_like, shape (3,)
        Position (radial) vector in meters.
    v : array_like, shape (3,)
        (Unused; kept for signature consistency.)
    magnitude : float
        Desired magnitude of the inclination-change acceleration in m/s^2.
        Positive → northward; negative → southward.

    Returns
    -------
    a : ndarray, shape (3,)
        Inclination-change acceleration vector in m/s^2.
    """
    r = np.asarray(r, dtype=float)
    norm_r = np.linalg.norm(r)
    if norm_r == 0 or magnitude == 0:
        return np.zeros(3)

    # Unit radial vector
    r_hat = r / norm_r

    # Earth's spin axis
    z_hat = np.array([0.0, 0.0, 1.0])

    # Project z_hat onto plane perpendicular to r_hat
    north_dir = z_hat - np.dot(z_hat, r_hat) * r_hat
    norm_nd = np.linalg.norm(north_dir)
    if norm_nd == 0:
        return np.zeros(3)

    # Scale to desired magnitude
    return magnitude * north_dir / norm_nd
