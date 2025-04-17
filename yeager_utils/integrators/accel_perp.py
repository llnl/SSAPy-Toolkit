import numpy as np

def accel_perp(r, v, magnitude):
    """
    Acceleration vector always perpendicular to both the velocity and radial vectors.

    Parameters
    ----------
    r : array_like, shape (3,)
        Position (radial) vector in meters.
    v : array_like, shape (3,)
        Velocity vector in m/s.
    magnitude : float
        Magnitude of the perpendicular acceleration in m/s^2.

    Returns
    -------
    a : ndarray, shape (3,)
        Perpendicular acceleration vector in m/s^2.

    Author: Travis Yeager
    """
    v = np.asarray(v)
    r = np.asarray(r)

    # Compute the cross product of velocity and radial direction to get perpendicular vector
    perp_dir = np.cross(v, r)

    # Normalize the direction
    norm_perp = np.linalg.norm(perp_dir)
    if norm_perp == 0:
        return np.zeros(3)
    
    # Scale the direction to the desired magnitude
    perp_accel = magnitude * perp_dir / norm_perp
    return perp_accel
