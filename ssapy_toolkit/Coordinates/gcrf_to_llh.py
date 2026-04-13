from astropy.coordinates import GCRS, ITRS, EarthLocation, CartesianRepresentation
from astropy import units as u
import numpy as np


def gcrf_to_llh(r_gcrf, t):
    """
    Invert GCRF position to geodetic lon (deg), lat (deg), height (m).

    Parameters
    ----------
    r_gcrf : array_like (3,) or (n,3)
        GCRF Cartesian position(s) in meters.
    t : astropy.time.Time
        Observation time.

    Returns
    -------
    lon : float or ndarray
    lat : float or ndarray
    height : float or ndarray
    """
    # Make sure we have an array of shape (N,3)
    r = np.atleast_2d(r_gcrf)
    if r.shape[1] != 3:
        raise ValueError("r_gcrf must be shape (3,) or (N,3)")

    # Build a CartesianRepresentation (with units)
    cart = CartesianRepresentation(x=r[:,0]*u.m,
                                   y=r[:,1]*u.m,
                                   z=r[:,2]*u.m)

    # Create a GCRS frame at time t
    gcrs = GCRS(cart, obstime=t)

    # Transform into ITRS (Earth-fixed)
    itrs = gcrs.transform_to(ITRS(obstime=t))

    # Extract cartesian coordinates in meters
    xyz = itrs.cartesian.xyz.to(u.m).value.T  # back to shape (N,3)

    # Now use EarthLocation to recover lon, lat, height
    loc = EarthLocation(x=xyz[...,0]*u.m,
                        y=xyz[...,1]*u.m,
                        z=xyz[...,2]*u.m)

    lon = loc.lon.to_value(u.deg)
    lat = loc.lat.to_value(u.deg)
    height = loc.height.to_value(u.m)

    # If input was 1D, squeeze outputs back to scalars
    if r_gcrf.ndim == 1:
        return lon.item(), lat.item(), height.item()

    return lon, lat, height
