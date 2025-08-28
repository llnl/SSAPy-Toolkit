from astropy.time import Time
from astropy import units as u
from astropy.coordinates import EarthLocation, GCRS


def llh_to_gcrf(lon: float,
                lat: float,
                t: Time,
                height: float = 0.0):
    """
    Convert geodetic (lon, lat, height) at a given time to GCRF (GCRS) position & velocity.

    Parameters
    ----------
    lon : float
        Geodetic longitude in degrees (east-positive).
    lat : float
        Geodetic latitude in degrees (north-positive).
    t : astropy.time.Time
        The observation time (UTC or TAI, etc.).
    height : float, optional
        Height above the reference ellipsoid in meters.

    Returns
    -------
    r_gcrf : ndarray, shape (3,)
        GCRF (GCRS) Cartesian position [x, y, z] in meters.
    v_gcrf : ndarray, shape (3,)
        GCRF (GCRS) Cartesian velocity [vx, vy, vz] in m/s.
    """
    # Earth-fixed location
    loc = EarthLocation(lon=lon * u.deg,
                        lat=lat * u.deg,
                        height=height * u.m)

    # Get GCRS coordinate (includes velocity differential)
    gcrs = loc.get_gcrs(obstime=t)

    # Position in meters
    r_gcrf = gcrs.cartesian.xyz.to(u.m).value

    # Velocity in m/s: differential 's' holds d(x,y,z)/dt
    v_gcrf = gcrs.cartesian.differentials['s'].d_xyz.to(u.m/u.s).value

    return r_gcrf, v_gcrf
