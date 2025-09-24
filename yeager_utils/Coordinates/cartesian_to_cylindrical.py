import numpy as np


def cart_to_cyl(x: float, y: float, z: float) -> float:
    """
    Convert Cartesian coordinates (x, y, z) to cylindrical coordinates (radius, angle, z).

    Parameters:
    - x (float): The x-coordinate in Cartesian space.
    - y (float): The y-coordinate in Cartesian space.
    - z (float): The z-coordinate in Cartesian space.

    Returns:
    - float: Radius, angle, and z in cylindrical coordinates.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    r = np.linalg.norm([x, y])
    theta = np.arctan2(y, x)
    return r, theta, z
