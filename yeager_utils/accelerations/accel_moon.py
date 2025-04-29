import numpy as np
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import get_body, GCRS, solar_system_ephemeris

from ..constants import MOON_MU
from ..time import to_gps


def accel_point_moon(r: np.ndarray, time) -> np.ndarray:
    """
    Point-mass lunar gravity in GCRF, given raw_time (same input as leapfrog)
    and the GPS-seconds of the initial epoch.

    Parameters
    ----------
    r : array_like, shape (3,)
        Satellite position vector in GCRF (m), Earth-centered.
    time : float or datetime-like or array_like
        The time array element from leapfrog (same type you passed into leapfrog).

    Returns
    -------
    a_moon : ndarray, shape (3,)
        Acceleration from the Moon (m/s²).
    """
    # 1) Convert time to GPS seconds since 1980-01-06
    time_gps = to_gps(time)
    t = Time(time_gps, format="gps", scale="utc")

    # 2) Get Moon position in GCRS at this time using JPL ephemeris
    with solar_system_ephemeris.set('jpl'):
        moon_gcrs = get_body("moon", t).transform_to(GCRS(obstime=t))

    r_moon = moon_gcrs.cartesian.xyz.to(u.m).value

    # 3) Compute delta and gravitational acceleration
    delta = r_moon - r
    return MOON_MU * delta / np.linalg.norm(delta) ** 3
