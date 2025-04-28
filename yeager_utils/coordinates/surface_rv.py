from ssapy.orbit import EarthObserver
from astropy.time import Time
from ..time import to_gps


def surface_rv(lon, lat, elevation=0.0, time=Time("1980-01-06 00:00:00", format="iso", scale="utc"), fast=False):
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
    time_gps : float or array_like, optional
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
    time = to_gps(time)

    observer = EarthObserver(lon=lon, lat=lat, elevation=elevation, fast=fast)
    return observer.getRV(time)
