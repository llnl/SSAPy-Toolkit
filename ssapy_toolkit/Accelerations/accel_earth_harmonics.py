# ssapy_toolkit/Accelerations/accel_earth_harmonics.py

import numpy as np
from ..constants import EARTH_MU, EARTH_RADIUS


def _check_r_safe(r2: float) -> None:
    """Raise error if r is below Earth's surface."""
    r_mag = float(np.sqrt(r2))
    if r_mag < EARTH_RADIUS:
        raise ValueError(f"r magnitude ({r_mag:.2f} m) is below Earth's surface.")


def accel_J2(r: np.ndarray) -> np.ndarray:
    J2 = 1.08262668e-3
    mu = EARTH_MU
    Re = EARTH_RADIUS

    r = np.asarray(r, dtype=float).reshape(3)
    r2 = float(np.dot(r, r))
    _check_r_safe(r2)
    r_mag = np.sqrt(r2)

    inv_r5 = 1.0 / (r2 * r_mag**3)  # = 1/r^5
    z = r[2]
    u = z / r_mag
    k2 = 5.0 * u**2 - 1.0

    factor2 = 1.5 * J2 * mu * Re**2 * inv_r5
    return factor2 * np.array([r[0] * k2, r[1] * k2, r[2] * (5.0 * u**2 - 3.0)])


def accel_J3(r: np.ndarray) -> np.ndarray:
    J3 = -2.5324105e-6
    mu = EARTH_MU
    Re = EARTH_RADIUS

    r = np.asarray(r, dtype=float).reshape(3)
    r2 = float(np.dot(r, r))
    _check_r_safe(r2)
    r_mag = np.sqrt(r2)

    inv_r6 = 1.0 / (r2**3)  # = 1/r^6
    z = r[2]
    u = z / r_mag

    P3 = 0.5 * (5.0 * u**3 - 3.0 * u)
    dP3 = 0.5 * (15.0 * u**2 - 3.0)
    factor3 = -mu * J3 * Re**3 * inv_r6
    return factor3 * (4.0 * P3 * r - dP3 * r * u * np.array([0.0, 0.0, 1.0]))


def accel_J4(r: np.ndarray) -> np.ndarray:
    J4 = -1.6198976e-6
    mu = EARTH_MU
    Re = EARTH_RADIUS

    r = np.asarray(r, dtype=float).reshape(3)
    r2 = float(np.dot(r, r))
    _check_r_safe(r2)
    r_mag = np.sqrt(r2)

    inv_r7 = 1.0 / (r2**2 * r_mag**3)  # = 1/r^7
    z = r[2]
    u = z / r_mag

    P4 = (1.0 / 8.0) * (35.0 * u**4 - 30.0 * u**2 + 3.0)
    dP4 = (1.0 / 8.0) * (140.0 * u**3 - 60.0 * u)
    factor4 = -mu * J4 * Re**4 * inv_r7
    return factor4 * (5.0 * P4 * r - dP4 * r * u * np.array([0.0, 0.0, 1.0]))


def accel_J5(r: np.ndarray) -> np.ndarray:
    J5 = -2.272960828e-7
    mu = EARTH_MU
    Re = EARTH_RADIUS

    r = np.asarray(r, dtype=float).reshape(3)
    r2 = float(np.dot(r, r))
    _check_r_safe(r2)
    r_mag = np.sqrt(r2)

    inv_r9 = 1.0 / (r2**4 * r_mag)  # = 1/r^9
    z = r[2]
    u = z / r_mag

    P5 = (1.0 / 8.0) * (63.0 * u**5 - 70.0 * u**3 + 15.0 * u)
    dP5 = (1.0 / 8.0) * (315.0 * u**4 - 210.0 * u**2 + 15.0)
    factor5 = -mu * J5 * Re**5 * inv_r9
    return factor5 * (6.0 * P5 * r - dP5 * r * u * np.array([0.0, 0.0, 1.0]))


def accel_J6(r: np.ndarray) -> np.ndarray:
    J6 = 5.406812391e-7
    mu = EARTH_MU
    Re = EARTH_RADIUS

    r = np.asarray(r, dtype=float).reshape(3)
    r2 = float(np.dot(r, r))
    _check_r_safe(r2)
    r_mag = np.sqrt(r2)

    inv_r10 = 1.0 / (r2**4 * r_mag**2)  # = 1/r^10
    z = r[2]
    u = z / r_mag

    P6 = (1.0 / 16.0) * (231.0 * u**6 - 315.0 * u**4 + 105.0 * u**2 - 5.0)
    dP6 = (1.0 / 16.0) * (1386.0 * u**5 - 1260.0 * u**3 + 210.0 * u)
    factor6 = -mu * J6 * Re**6 * inv_r10
    return factor6 * (7.0 * P6 * r - dP6 * r * u * np.array([0.0, 0.0, 1.0]))


def accel_J7(r: np.ndarray) -> np.ndarray:
    J7 = -3.529000898e-7
    mu = EARTH_MU
    Re = EARTH_RADIUS

    r = np.asarray(r, dtype=float).reshape(3)
    r2 = float(np.dot(r, r))
    _check_r_safe(r2)
    r_mag = np.sqrt(r2)

    inv_r11 = 1.0 / (r2**4 * r_mag**3)  # = 1/r^11
    z = r[2]
    u = z / r_mag

    P7 = (1.0 / 16.0) * (429.0 * u**7 - 693.0 * u**5 + 315.0 * u**3 - 35.0 * u)
    dP7 = (1.0 / 16.0) * (3003.0 * u**6 - 3465.0 * u**4 + 945.0 * u**2 - 35.0)
    factor7 = -mu * J7 * Re**7 * inv_r11
    return factor7 * (8.0 * P7 * r - dP7 * r * u * np.array([0.0, 0.0, 1.0]))


def accel_J8(r: np.ndarray) -> np.ndarray:
    J8 = 2.532000898e-7
    mu = EARTH_MU
    Re = EARTH_RADIUS

    r = np.asarray(r, dtype=float).reshape(3)
    r2 = float(np.dot(r, r))
    _check_r_safe(r2)
    r_mag = np.sqrt(r2)

    inv_r12 = 1.0 / (r2**5 * r_mag**2)  # = 1/r^12
    z = r[2]
    u = z / r_mag

    P8 = (1.0 / 128.0) * (6435.0 * u**8 - 12012.0 * u**6 + 6930.0 * u**4 - 1260.0 * u**2 + 35.0)
    dP8 = (1.0 / 128.0) * (51480.0 * u**7 - 72072.0 * u**5 + 27720.0 * u**3 - 2520.0 * u)
    factor8 = -mu * J8 * Re**8 * inv_r12
    return factor8 * (9.0 * P8 * r - dP8 * r * u * np.array([0.0, 0.0, 1.0]))


def accel_earth_harmonics(r: np.ndarray) -> np.ndarray:
    """
    Central gravity + J2..J8 zonal harmonics.
    """
    r = np.asarray(r, dtype=float).reshape(3)
    r2 = float(np.dot(r, r))
    _check_r_safe(r2)
    r_mag = np.sqrt(r2)

    a_central = -EARTH_MU * r / (r_mag**3)  # [59]
    return (
        a_central
        + accel_J2(r)
        + accel_J3(r)
        + accel_J4(r)
        + accel_J5(r)
        + accel_J6(r)
        + accel_J7(r)
        + accel_J8(r)
    )
