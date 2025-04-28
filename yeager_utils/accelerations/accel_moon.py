import numpy as np
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import get_moon, GCRS

from ..constants import MOON_MU
from ..time import to_gps


def accel_point_moon(r: np.ndarray, time) -> np.ndarray:
    """
    Point‐mass lunar gravity in GCRF, given raw_time (same input as leapfrog)
    and the GPS‐seconds of the initial epoch.

    Parameters
    ----------
    r : array_like, shape (3,)
        Satellite position vector in GCRF (m), Earth‐centered.
    raw_time : float or datetime-like or array_like
        The time array element from leapfrog (same type you passed into leapfrog).

    Returns
    -------
    a_moon : ndarray, shape (3,)
        Acceleration from the Moon (m/s²).
    """
    # 1) Convert whatever raw_time is into GPS seconds since 1980-01-06
    time_gps = to_gps(time)  # scalar or array element
    # 2) Build an Astropy Time in GPS scale
    t = Time(time_gps, format="gps", scale="utc")

    # 3) Get the Moon’s GCRS position at that GPS‐time
    moon_gcrs = get_moon(t).transform_to(GCRS(obstime=t))
    r_moon = moon_gcrs.cartesian.xyz.to(u.m).value

    # 4) Compute δ = r_moon − r_sat and return μ δ / |δ|^3
    delta = r_moon - r
    return MOON_MU * delta / np.linalg.norm(delta) ** 3
