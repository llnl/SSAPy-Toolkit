from ssapy.orbit import EarthObserver
from astropy.time import Time
from ..Yastropy import astropy_surface_rv
from ..Time_Functions import to_gps


def surface_rv(lon, lat, elevation=0.0, t=Time(0, format="gps", scale="utc")):
    """
    Get GCRF position and velocity of a surface observer at a specific location and time.

    Parameters
    ----------
    lat : float
        Geodetic latitude in degrees.
    lon : float
        Geodetic longitude in degrees.
    elevation : float, optional
        Elevation above sea level in meters (default is 0).
    t : Time
        Astropy Time object.

    Returns
    -------
    r : ndarray, shape (3,)
        Position vector in GCRF coordinates (m).
    v : ndarray, shape (3,)
        Velocity vector in GCRF coordinates (m/s).
    """

    return astropy_surface_rv(lon=lon, lat=lat, elevation=elevation, t=t)


def surface_rv_ssapy(lon, lat, elevation=0.0, t=Time(0, format="gps", scale="utc"), fast=False):
    """
    Get GCRF position and velocity of a surface observer at a specific location and time.

    Parameters
    ----------
    lat : float
        Geodetic latitude in degrees.
    lon : float
        Geodetic longitude in degrees.
    elevation : float, optional
        Elevation above sea level in meters (default is 0).
    t : float or array_like, optional
        GPS seconds since 1980-01-06 00:00:00 UTC. If None, uses current time.
    fast : bool, optional
        Use fast computation with approximate Earth orientation parameters.

    Returns
    -------
    r : ndarray, shape (3,)
        Position vector in GCRF coordinates (m).
    v : ndarray, shape (3,)
        Velocity vector in GCRF coordinates (m/s).
    """
    t = to_gps(t)

    observer = EarthObserver(lon=lon, lat=lat, elevation=elevation, fast=fast)
    return observer.getRV(t)
