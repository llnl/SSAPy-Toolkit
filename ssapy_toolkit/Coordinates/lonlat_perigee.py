import numpy as np
from ssapy.orbit import Orbit
from ..Yastropy import astropy_surface_rv
from ..constants import EARTH_MU, EARTH_RADIUS
from ..Time_Functions import to_gps, Time


def lonlat_perigee(lon, lat, t, alt=1000e3, e=0, i=None, EARTH_MU=EARTH_MU):
    """
    Create an Orbit whose perigee is exactly over the fixed geodetic
    longitude/latitude (lon, lat) at epoch t.

    Parameters
    ----------
    lon : float
        Geodetic longitude in degrees.
    lat : float
        Geodetic latitude in degrees.
    t : str or Time
        Epoch (ISO string or Astropy Time).
    alt : float
        Altitude above Earth surface in meters.
    e : float
        Eccentricity of the orbit.
    i : float
        Inclination in degrees. Defaults to the same as latitude if None.
    EARTH_MU : float
        Earth's gravitational parameter [m³/s²].

    Returns
    -------
    Orbit
        Orbit object with perigee over (lon, lat) at time t.
    """
    if i is None:
        i = lat

    # Get surface position vector in GCRF
    r_gcrf, _ = astropy_surface_rv(lon, lat, t=t)

    # Set perigee radius
    rp = EARTH_RADIUS + alt
    r_hat = r_gcrf / np.linalg.norm(r_gcrf)
    r_peri = rp * r_hat

    # Use Rodrigues' rotation formula to tilt the angular momentum vector
    def rodrigues(u, k, theta):
        return (
            u * np.cos(theta)
            + np.cross(k, u) * np.sin(theta)
            + k * (np.dot(k, u)) * (1 - np.cos(theta))
        )

    # Define unit vector for Earth's rotation axis
    k_hat = np.array([0.0, 0.0, 1.0])
    i_rad = np.deg2rad(i)

    # Get angular momentum direction
    h_hat = rodrigues(k_hat, r_hat, i_rad)
    h_hat /= np.linalg.norm(h_hat)

    # Velocity direction is perpendicular to both h_hat and r_hat
    v_hat = np.cross(h_hat, r_hat)
    v_hat /= np.linalg.norm(v_hat)

    # Perigee speed for elliptical orbit
    v_mag = np.sqrt(EARTH_MU * (1 + e) / rp)
    v_peri = v_hat * v_mag

    return Orbit(r=r_peri, v=v_peri, t=t, mu=EARTH_MU)
