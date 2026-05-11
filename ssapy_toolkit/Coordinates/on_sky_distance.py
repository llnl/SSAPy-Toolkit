import numpy as np
from ..constants import EARTH_RADIUS


def lonlat_distance(lat1: float, lat2: float, lon1: float, lon2: float) -> float:
    """
    Calculate the distance between two points on the Earth's surface using the Haversine formula.

    Parameters:
    - lat1, lat2 (float): Latitude of the two points in radians.
    - lon1, lon2 (float): Longitude of the two points in radians.

    Returns:
    - float: Distance between the two points in kilometers.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    # Radius of Earth in kilometers
    return c * EARTH_RADIUS
