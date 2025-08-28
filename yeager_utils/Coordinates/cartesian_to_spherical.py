from typing import Tuple
import numpy as np


def cart2sph_deg(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (azimuth, elevation, radius) in degrees.

    Parameters:
    - x (float): The x-coordinate in Cartesian space.
    - y (float): The y-coordinate in Cartesian space.
    - z (float): The z-coordinate in Cartesian space.

    Returns:
    - Tuple[float, float, float]: Azimuth, elevation, and radius in degrees.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy) * (180 / np.pi)
    az = (np.arctan2(y, x)) * (180 / np.pi)
    return az, el, r
