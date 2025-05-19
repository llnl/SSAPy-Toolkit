import numpy as np
from astropy.time import Time
from ssapy.orbit import Orbit
from ..constants import EARTH_MU
from ..coordinates import surface_rv
from ..time import to_gps

def lonlat_perigee(lon, lat, t0, a, e, i, mu=EARTH_MU):
    """
    Create an Orbit whose perigee is exactly over the fixed geodetic
    longitude/latitude (lon, lat) at epoch t0.

    Parameters
    ----------
    lon : float
        Geodetic longitude in degrees.
    lat : float
        Geodetic latitude in degrees.
    t0 : str or Time
        Epoch (ISO string or Astropy Time).
    a : float
        Semimajor axis [m].
    e : float
        Eccentricity.
    i : float
        Inclination in degrees.
    mu : float, optional
        Gravitational parameter [m³/s²], by default EARTH_MU.

    Returns
    -------
    Orbit
        Orbit object with perigee over (lon, lat) at time t0.
    """
    t0 = to_gps(t0)

    r_surf, _ = surface_rv(lon, lat, elevation=0.0, fast=False)
    rp = a * (1 - e)
    r_peri = rp * (r_surf / np.linalg.norm(r_surf))
    r_hat = r_peri / np.linalg.norm(r_peri)

    def rodrigues(u, k, theta):
        return (
            u * np.cos(theta)
            + np.cross(k, u) * np.sin(theta)
            + k * (np.dot(k, u)) * (1 - np.cos(theta))
        )

    # Rotate Earth's spin axis by inclination about the periapsis direction
    k_hat = np.array([0.0, 0.0, 1.0])
    i_rad = np.deg2rad(i)
    h_hat = rodrigues(k_hat, r_hat, i_rad)
    h_hat /= np.linalg.norm(h_hat)

    # Velocity direction perpendicular to r_hat in the orbital plane
    v_hat = np.cross(h_hat, r_hat)
    v_hat /= np.linalg.norm(v_hat)

    # Perigee speed
    v_peri = v_hat * np.sqrt(mu * (1 + e) / (a * (1 - e)))

    # Construct orbit from state vectors
    return Orbit(r=r_peri, v=v_peri, t=t0, mu=mu)
