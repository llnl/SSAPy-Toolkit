# yeager_utils/Accelerations/accel_moon.py

import numpy as np
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import get_body, GCRS, solar_system_ephemeris

from ..constants import MOON_MU
from ..Time_Functions import to_gps


def accel_point_moon(r: np.ndarray, time) -> np.ndarray:
    """
    Point-mass lunar gravity in GCRF.

    Parameters
    ----------
    r : array_like, shape (3,)
        Satellite position vector in GCRF (m), Earth-centered.
    time : float | datetime-like | astropy.time.Time | array_like
        Time corresponding to the state (anything supported by to_gps()).

    Returns
    -------
    a_moon : ndarray, shape (3,)
        Acceleration from the Moon (m/s^2).
    """
    r = np.asarray(r, dtype=float).reshape(3)

    # 1) Convert time to GPS seconds since 1980-01-06
    time_gps = to_gps(time)  # [176]
    t = Time(time_gps, format="gps", scale="utc")

    # 2) Get Moon position in GCRS at this time using JPL ephemeris
    with solar_system_ephemeris.set("jpl"):
        moon_gcrs = get_body("moon", t).transform_to(GCRS(obstime=t))

    r_moon = moon_gcrs.cartesian.xyz.to(u.m).value  # (3,)

    # 3) Compute delta and gravitational acceleration
    delta = r_moon - r
    d = np.linalg.norm(delta)
    if d == 0.0:
        return np.zeros(3, dtype=float)

    return MOON_MU * delta / d**3
