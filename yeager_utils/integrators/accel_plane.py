import numpy as np

def accel_plane(r, v, magnitude):
    """
    Perpendicular acceleration in the direction tangential to the orbit,
    restricted to the plane defined by the inclination of the radial vector.

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
    r = np.asarray(r)
    v = np.asarray(v)

    # Compute the cross product of velocity and radial direction to get perpendicular vector
    perp_dir = np.cross(r, v)

    # Normalize the direction
    norm_perp = np.linalg.norm(perp_dir)
    if norm_perp == 0:
        return np.zeros(3)
    
    # Scale the direction to the desired magnitude
    perp_accel = magnitude * perp_dir / norm_perp
    
    # Inclination calculation
    r_norm = np.linalg.norm(r)
    inclination = np.arccos(r[2] / r_norm)  # Angle between r and the z-axis

    # If inclination is close to 0 (near equatorial), no projection needed
    # Otherwise, we project the perpendicular acceleration into the plane defined by the inclination
    if inclination != 0:
        # The unit vector normal to the orbital plane
        orbital_plane_normal = np.cross(r, v)  # Cross product of r and v gives the normal to the plane
        orbital_plane_normal /= np.linalg.norm(orbital_plane_normal)

        # Project the perpendicular acceleration onto the orbital plane
        perp_accel = perp_accel - np.dot(perp_accel, orbital_plane_normal) * orbital_plane_normal
    
    return perp_accel
