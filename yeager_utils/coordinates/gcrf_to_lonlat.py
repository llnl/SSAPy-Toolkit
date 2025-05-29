import numpy as np
from typing import Union, Tuple, Optional
from ssapy import groundTrack
from .v_from_r import v_from_r
from ..time import Time, to_gps


def gcrf_to_lonlat(r_gcrf: np.ndarray, t: np.ndarray):
    """
    Convert GCRF coordinates to ITRF coordinates.

    Parameters:
    - r_gcrf (np.ndarray): 3D position vector in GCRF coordinates (meters).
    - t (np.ndarray): Time array for conversion.
    - v (Optional[np.ndarray]): Velocity vector in GCRF coordinates (meters per second). Optional.

    Returns:
    - lon, lat, height in degrees and meters.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    t = np.atleast_1d(t)
    r_gcrf = np.atleast_2d(r_gcrf)
    t = to_gps(t)
    lon, lat, height = groundTrack(r_gcrf, t, format='geodetic')
    return np.degrees(lon), np.degrees(lat), height

