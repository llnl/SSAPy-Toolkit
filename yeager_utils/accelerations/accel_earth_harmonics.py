import numpy as np
from ..constants import EARTH_MU, EARTH_RADIUS


def accel_earth_harmonics(r: np.ndarray) -> np.ndarray:
    """
    Zonal-harmonic gravity (J2–J5) computed directly in GCRF.

    Parameters
    ----------
    r : ndarray, shape (3,)
        Satellite position vector in GCRF (m), Earth-centered.

    Returns
    -------
    a : ndarray, shape (3,)
        Total gravitational acceleration including J2–J5 (m/s²).
    """
    # Earth's zonal coefficients
    J2 = 1.08262668e-3
    J3 = -2.5324105e-6
    J4 = -1.6198976e-6
    J5 = -2.272960828e-7

    # Earth spin axis in GCRF (unit Z)
    k = np.array([0.0, 0.0, 1.0])

    # Precompute norms and dot products
    r2 = np.dot(r, r)
    r_mag = np.sqrt(r2)
    inv_r = 1.0 / r_mag
    inv_r2 = inv_r * inv_r
    inv_r3 = inv_r * inv_r2
    u = np.dot(r, k) * inv_r     # cos(theta) = (r·k)/r

    # Central term
    a = -EARTH_MU * inv_r3 * r

    # Helper: repeated factors
    mu = EARTH_MU
    Re = EARTH_RADIUS

    # Legendre polynomials Pn(u) and derivatives Pn'(u)
    P2 = 0.5*(3*u*u - 1)
    dP2 = 3*u

    P3 = 0.5*(5*u**3 - 3*u)
    dP3 = 0.5*(15*u*u - 3)

    P4 = (1/8)*(35*u**4 - 30*u*u + 3)
    dP4 = (1/8)*(140*u**3 - 60*u)

    P5 = (1/8)*(63*u**5 - 70*u**3 + 15*u)
    dP5 = (1/8)*(315*u**4 - 210*u**2 + 15)

    # Generic nth term contribution: 
    #   a_n = mu * Jn * (Re^n) / r^(n+2) * [ (n+1) Pn(u) r  -  dPn(u) r * (r·k) * k/r  ]
    # but simplified below for each n:

    # J2 term
    factor2 = mu * J2 * Re**2 * inv_r**4
    a += factor2 * (3*P2 * r - dP2 * r_mag * u * k)

    # J3 term
    factor3 = mu * J3 * Re**3 * inv_r**5
    a += factor3 * (4*P3 * r - dP3 * r_mag * u * k)

    # J4 term
    factor4 = mu * J4 * Re**4 * inv_r**6
    a += factor4 * (5*P4 * r - dP4 * r_mag * u * k)

    # J5 term
    factor5 = mu * J5 * Re**5 * inv_r**7
    a += factor5 * (6*P5 * r - dP5 * r_mag * u * k)

    return a
