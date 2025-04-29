import numpy as np
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import get_body, GCRS, solar_system_ephemeris

from ..constants import SUN_MU
from ..time import to_gps


def accel_point_sun(r: np.ndarray, time) -> np.ndarray:
    """
    Point-mass solar gravity in GCRF, given raw_time (same input as leapfrog).

    Parameters
    ----------
    r : array_like, shape (3,)
        Satellite position vector in GCRF (m), Earth-centered.
    time : float or datetime-like or array_like
        The time array element from leapfrog (same type you passed into leapfrog).

    Returns
    -------
    a_sun : ndarray, shape (3,)
        Acceleration from the Sun (m/s²).
    """
    # 1) Convert time to GPS seconds
    time_gps = to_gps(time)
    t = Time(time_gps, format="gps", scale="utc")

    # 2) Get Sun position in GCRS at this time using JPL ephemeris
    with solar_system_ephemeris.set('jpl'):
        sun_gcrs = get_body("sun", t).transform_to(GCRS(obstime=t))

    r_sun = sun_gcrs.cartesian.xyz.to(u.m).value

    # 3) Compute delta and gravitational acceleration
    delta = r_sun - r
    return SUN_MU * delta / np.linalg.norm(delta) ** 3
