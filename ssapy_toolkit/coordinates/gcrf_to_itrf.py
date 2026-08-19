import numpy as np
from ssapy import groundTrack
from .v_from_r import v_from_r
from ..time_functions import Time, to_gps


def gcrf_to_itrf(r_gcrf, t, v=None):
    """
    Convert GCRF coordinates to ITRF coordinates.

    Parameters:
    - r_gcrf (np.ndarray): 3D position vector in GCRF coordinates (meters).
    - t (np.ndarray): Time array for conversion.
    - v (np.ndarray, optional): Velocity vector in GCRF coordinates (m/s). Optional.

    Returns:
    - np.ndarray: Position in ITRF coordinates,
      or (position, velocity) in ITRF coordinates if velocity is provided.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    t = to_gps(t)
    x, y, z = groundTrack(r_gcrf, t, format="cartesian")
    pos = np.array([x, y, z]).T
    if v is None:
        return pos
    else:
        return pos, v_from_r(pos, t)


def gcrf_to_itrf_astropy(state_vectors, t):
    """
    Convert GCRF positions to geocentric ITRF using Astropy.

    Parameters:
    - state_vectors (np.ndarray): Position vectors in GCRF coordinates (meters),
      shape (N, 3).
    - t (Time): Time of conversion.

    Returns:
    - np.ndarray: Position vectors in ITRF coordinates (meters), shape (N, 3).

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    import astropy.units as u
    from astropy.coordinates import GCRS, ITRS, SkyCoord

    state_vectors = np.asarray(state_vectors, dtype=float)
    if state_vectors.ndim != 2 or state_vectors.shape[1] != 3:
        raise ValueError("state_vectors must have shape (N, 3)")

    sc = SkyCoord(
        x=state_vectors[:, 0] * u.m,
        y=state_vectors[:, 1] * u.m,
        z=state_vectors[:, 2] * u.m,
        representation_type="cartesian",
        frame=GCRS(obstime=t),
    )
    sc_itrs = sc.transform_to(ITRS(obstime=t))

    return np.array([
        sc_itrs.cartesian.x.to_value(u.m),
        sc_itrs.cartesian.y.to_value(u.m),
        sc_itrs.cartesian.z.to_value(u.m),
    ]).T
