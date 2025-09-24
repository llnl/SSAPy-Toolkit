import numpy as np
from ..constants import EARTH_MU, EARTH_RADIUS

def _check_r_safe(r2: float) -> None:
    """Raise error if r is below Earth's surface."""
    if np.sqrt(r2) < EARTH_RADIUS:
        raise ValueError(f"r magnitude ({np.sqrt(r2):.2f} m) is below Earth's surface.")

def accel_J2(r: np.ndarray) -> np.ndarray:
    J2 = 1.08262668e-3
    mu = EARTH_MU
    Re = EARTH_RADIUS

    r2 = np.dot(r, r)
    _check_r_safe(r2)
    r_mag = np.sqrt(r2)
    inv_r5 = 1.0 / (r2 * r_mag**3)
    z = r[2]
    u = z / r_mag
    k2 = 5 * u**2 - 1

    factor2 = 1.5 * J2 * mu * Re**2 * inv_r5
    return factor2 * np.array([r[0] * k2, r[1] * k2, r[2] * (5 * u**2 - 3)])

def accel_J3(r: np.ndarray) -> np.ndarray:
    J3 = -2.5324105e-6
    mu = EARTH_MU
    Re = EARTH_RADIUS

    r2 = np.dot(r, r)
    _check_r_safe(r2)
    r_mag = np.sqrt(r2)
    inv_r6 = 1.0 / (r2 * r2 * r2)  # r2**3 = r_mag**6
    z = r[2]
    u = z / r_mag

    P3 = 0.5 * (5 * u**3 - 3 * u)
    dP3 = 0.5 * (15 * u**2 - 3)
    factor3 = -mu * J3 * Re**3 * inv_r6
    return factor3 * (4 * P3 * r - dP3 * r * u * np.array([0, 0, 1]))

def accel_J4(r: np.ndarray) -> np.ndarray:
    J4 = -1.6198976e-6
    mu = EARTH_MU
    Re = EARTH_RADIUS

    r2 = np.dot(r, r)
    _check_r_safe(r2)
    r_mag = np.sqrt(r2)
    inv_r7 = 1.0 / (r2 * r2 * r_mag**3)
    z = r[2]
    u = z / r_mag

    P4 = (1 / 8) * (35 * u**4 - 30 * u**2 + 3)
    dP4 = (1 / 8) * (140 * u**3 - 60 * u)
    factor4 = -mu * J4 * Re**4 * inv_r7
    return factor4 * (5 * P4 * r - dP4 * r * u * np.array([0, 0, 1]))

def accel_J5(r: np.ndarray) -> np.ndarray:
    J5 = -2.272960828e-7
    mu = EARTH_MU
    Re = EARTH_RADIUS

    r2 = np.dot(r, r)
    _check_r_safe(r2)
    r_mag = np.sqrt(r2)
    inv_r9 = 1.0 / (r2 * r2 * r2 * r2 * r_mag)
    z = r[2]
    u = z / r_mag

    P5 = (1 / 8) * (63 * u**5 - 70 * u**3 + 15 * u)
    dP5 = (1 / 8) * (315 * u**4 - 210 * u**2 + 15)
    factor5 = -mu * J5 * Re**5 * inv_r9
    return factor5 * (6 * P5 * r - dP5 * r * u * np.array([0, 0, 1]))

def accel_J6(r: np.ndarray) -> np.ndarray:
    J6 = 5.406812391e-7
    mu = EARTH_MU
    Re = EARTH_RADIUS

    r2 = np.dot(r, r)
    _check_r_safe(r2)
    r_mag = np.sqrt(r2)
    inv_r10 = 1.0 / (r2 * r2 * r2 * r2 * r_mag**2)
    z = r[2]
    u = z / r_mag

    P6 = (1 / 16) * (231 * u**6 - 315 * u**4 + 105 * u**2 - 5)
    dP6 = (1 / 16) * (1386 * u**5 - 1260 * u**3 + 210 * u)
    factor6 = -mu * J6 * Re**6 * inv_r10
    return factor6 * (7 * P6 * r - dP6 * r * u * np.array([0, 0, 1]))

def accel_J7(r: np.ndarray) -> np.ndarray:
    J7 = -3.529000898e-7  # Corrected from 1.977000898e-7
    mu = EARTH_MU
    Re = EARTH_RADIUS

    r2 = np.dot(r, r)
    _check_r_safe(r2)
    r_mag = np.sqrt(r2)
    inv_r11 = 1.0 / (r2 * r2 * r2 * r2 * r_mag**3)
    z = r[2]
    u = z / r_mag

    P7 = (1 / 16) * (429 * u**7 - 693 * u**5 + 315 * u**3 - 35 * u)
    dP7 = (1 / 16) * (3003 * u**6 - 3465 * u**4 + 945 * u**2 - 35)
    factor7 = -mu * J7 * Re**7 * inv_r11
    return factor7 * (8 * P7 * r - dP7 * r * u * np.array([0, 0, 1]))

def accel_J8(r: np.ndarray) -> np.ndarray:
    J8 = 2.532000898e-7  # Corrected from 5.406812391e-8
    mu = EARTH_MU
    Re = EARTH_RADIUS

    r2 = np.dot(r, r)
    _check_r_safe(r2)
    r_mag = np.sqrt(r2)
    inv_r12 = 1.0 / (r2 * r2 * r2 * r2 * r2 * r_mag**2)
    z = r[2]
    u = z / r_mag

    P8 = (1 / 128) * (6435 * u**8 - 12012 * u**6 + 6930 * u**4 - 1260 * u**2 + 35)
    dP8 = (1 / 128) * (51480 * u**7 - 72072 * u**5 + 27720 * u**3 - 2520 * u)
    factor8 = -mu * J8 * Re**8 * inv_r12
    return factor8 * (9 * P8 * r - dP8 * r * u * np.array([0, 0, 1]))

def accel_earth_harmonics(r: np.ndarray) -> np.ndarray:
    r2 = np.dot(r, r)
    _check_r_safe(r2)
    r_mag = np.sqrt(r2)
    a_central = -EARTH_MU * r / (r_mag**3)

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