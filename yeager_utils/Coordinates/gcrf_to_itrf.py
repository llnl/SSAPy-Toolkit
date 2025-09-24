import numpy as np
from ssapy import groundTrack
from .v_from_r import v_from_r
from ..Time_Functions import Time, to_gps


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
    Convert GCRF state vectors to ITRF using Astropy.

    Parameters:
    - state_vectors (np.ndarray): Position and velocity vectors in GCRF coordinates (meters).
    - t (Time): Time of conversion.

    Returns:
    - np.ndarray: Position and velocity vectors in ITRF coordinates (meters).

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    import astropy.units as u
    from astropy.coordinates import GCRS, ITRS, SkyCoord, get_body_barycentric, solar_system_ephemeris, ICRS

    sc = SkyCoord(
        x=state_vectors[:, 0] * u.m,
        y=state_vectors[:, 1] * u.m,
        z=state_vectors[:, 2] * u.m,
        representation_type="cartesian",
        frame=GCRS(obstime=t),
    )
    sc_itrs = sc.transform_to(ITRS(obstime=t))

    with solar_system_ephemeris.set("de430"):
        earth = get_body_barycentric("earth", t)

    earth_center_itrs = SkyCoord(
        earth.x,
        earth.y,
        earth.z,
        representation_type="cartesian",
        frame=ICRS(),
    ).transform_to(ITRS(obstime=t))

    itrs_coords = SkyCoord(
        sc_itrs.x.value - earth_center_itrs.x.to_value(u.m),
        sc_itrs.y.value - earth_center_itrs.y.to_value(u.m),
        sc_itrs.z.value - earth_center_itrs.z.to_value(u.m),
        representation_type="cartesian",
        frame=ITRS(obstime=t),
    )

    return np.array([itrs_coords.x, itrs_coords.y, itrs_coords.z]).T
