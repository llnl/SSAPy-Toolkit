import numpy as np
from ..Time_Functions import Time
from ssapy import get_body
from .v_from_r import v_from_r


def get_lunar_rv(t):
    """
    Calculate the position and velocity of the Moon at a given time or times.

    Parameters:
    t (Time or array-like): A `Time` object or an array of time points for which to calculate the position and velocity. 
                            If a single time is passed, the position and velocity are calculated at that time. 
                            If multiple times are passed, the position and velocity are calculated for each time.

    Returns:
    tuple: A tuple containing two elements:
        - r (ndarray): The position of the Moon(s) at the given time(s), in kilometers.
        - v (ndarray): The velocity of the Moon(s) at the given time(s), in kilometers per second.

    Author:
    Travis Yeager (yeager7@llnl.gov)
    """
    if np.size(t) > 1:
        if isinstance(t[0], Time):
            t = t.gps
    else:
        if isinstance(t, Time):
            t = t.gps
    r = get_body("moon").position(t).T
    if np.size(t) > 1:
        v = v_from_r(r, t)
    else:
        v = (r - get_body("moon").position(t + 1).T) / 2
    return np.atleast_2d(r), np.atleast_2d(v)
