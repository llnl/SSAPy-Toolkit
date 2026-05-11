import numpy as np
from ssapy import get_body
from ..vectors import rotation_matrix_from_vectors
from ..constants import EARTH_MU, MOON_MU
from ..Coordinates import gcrf_to_lunar, gcrf_to_lunar_fixed
from ..Time_Functions import Time


def moon_normal_vector(t):
    """
    Calculate the normal vector to the Moon's orbital plane at a given time.

    Parameters
    ----------
    t : Time or list
        The time at which to calculate the Moon's orbital plane normal vector. Can be:
        - A single `Time` object (from astropy)
        - A list of `Time` objects
        - A list of GPS times (float)
        - A single GPS time (float)

    Returns
    -------
    np.ndarray
        The normal vector to the Moon's orbital plane, normalized to unit length.

    Notes
    -----
    - The normal vector is calculated as the cross product of the Moon's position
      vector at time `t` and its position vector one week later (`t + 604800` seconds).
    - The result is normalized to ensure it has unit length.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    if isinstance(t, list):
        t = [item.gps if isinstance(item, Time) else item for item in t]
    elif isinstance(t, Time):
        t = t.gps
    r = get_body("moon").position(t).T
    r_random = get_body("moon").position(t.gps + 604800).T
    return np.cross(r, r_random) / np.linalg.norm(r, axis=-1)


def lunar_lagrange_points(t):
    """
    Calculate the positions of the lunar Lagrange points (L1, L2, L3, L4, L5)
    in the Earth-Moon system.

    Parameters
    ----------
    t : Time or list
        The time at which to calculate the Lagrange points. Can be:
        - A single `Time` object (from astropy)
        - A list of `Time` objects
        - A list of GPS times (float)
        - A single GPS time (float)

    Returns
    -------
    dict
        A dictionary containing the positions of the Lagrange points:
        - "L1": Position of L1 from Earth in the Moon's direction (np.ndarray or None if discriminant < 0)
        - "L2": Position of L2 from Earth in the Moon's direction (np.ndarray or None if discriminant < 0)
        - "L3": Position of L3 (opposite to the Moon, on the far side of Earth)
        - "L4": Position of L4 (60 degrees ahead of the Moon in its orbit)
        - "L5": Position of L5 (60 degrees behind the Moon in its orbit)

    Notes
    -----
    - L1 and L2 are calculated by solving a quadratic equation.
    - L4 and L5 are approximated by shifting the Moon's position forward
      or backward by 1/6 of its orbital period.
    - L3 is simply the position opposite the Moon.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    if isinstance(t, list):
        t = [item.gps if isinstance(item, Time) else item for item in t]
    elif isinstance(t, Time):
        t = t.gps
    r = get_body("moon").position(t).T
    d = np.linalg.norm(r)  # Distance between Earth and Moon
    unit_vector_moon = r / np.linalg.norm(r, axis=-1)
    # plane_vector = np.cross(r, r_random)
    lunar_period_seconds = 2.3605915968e6

    # Coefficients of the quadratic equation
    a = EARTH_MU - MOON_MU
    b = 2 * MOON_MU * d
    c = -MOON_MU * d**2

    # Solve the quadratic equation
    discriminant = b**2 - 4 * a * c

    if discriminant >= 0:
        L1_from_moon = (-b - np.sqrt(discriminant)) / (2 * a) * unit_vector_moon
        L2_from_moon = (-b + np.sqrt(discriminant)) / (2 * a) * unit_vector_moon
    else:
        print("Discriminate is less than 0! THAT'S WEIRD FIX IT.")
        L1_from_moon = None
        L2_from_moon = None

    return {
        "L1": L1_from_moon + r,
        "L2": L2_from_moon + r,
        "L3": -r,
        "L4": get_body("moon").position(t + lunar_period_seconds / 6).T,
        "L5": get_body("moon").position(t - lunar_period_seconds / 6).T
    }


def lunar_lagrange_points_circular(t):
    """
    Calculate the positions of the lunar Lagrange points (L1, L2, L3, L4, L5)
    in a circular restricted three-body problem.

    Parameters
    ----------
    t : Time or list
        The time at which to calculate the Lagrange points. Can be:
        - A single `Time` object (from astropy)
        - A list of `Time` objects
        - A list of GPS times (float)
        - A single GPS time (float)

    Returns
    -------
    dict
        A dictionary containing the positions of the Lagrange points:
        - "L1": Position of L1 from Earth in the Moon's direction (np.ndarray or None if discriminant < 0)
        - "L2": Position of L2 from Earth in the Moon's direction (np.ndarray or None if discriminant < 0)
        - "L3": Position of L3 (opposite to the Moon, on the far side of Earth)
        - "L4": Position of L4 (60 degrees ahead of the Moon in its orbit)
        - "L5": Position of L5 (60 degrees behind the Moon in its orbit)

    Notes
    -----
    - L1 and L2 are calculated by solving a quadratic equation.
    - L4 and L5 are calculated by rotating the Moon's position by ±60 degrees.
    - L3 is simply the position opposite the Moon.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    if isinstance(t, list):
        t = [item.gps if isinstance(item, Time) else item for item in t]
    elif isinstance(t, Time):
        t = t.gps
    r = get_body("moon").position(t).T
    d = np.linalg.norm(r)  # Distance between Earth and Moon
    unit_vector_moon = r / np.linalg.norm(r, axis=-1)

    # Coefficients of the quadratic equation
    a = EARTH_MU - MOON_MU
    b = 2 * MOON_MU * d
    c = -MOON_MU * d**2

    # Solve the quadratic equation
    discriminant = b**2 - 4 * a * c

    if discriminant >= 0:
        L1_from_moon = (-b - np.sqrt(discriminant)) / (2 * a) * unit_vector_moon
        L2_from_moon = (-b + np.sqrt(discriminant)) / (2 * a) * unit_vector_moon
    else:
        print("Discriminate is less than 0! THAT'S WEIRD FIX IT.")
        L1_from_moon = None
        L2_from_moon = None

    # L45
    # Create the rotation matrix to align z-axis with the normal vector
    normal_vector = moon_normal_vector(t)
    rotation_matrix = rotation_matrix_from_vectors(np.array([0, 0, 1]), normal_vector)
    theta = np.radians(60) + np.arctan2(unit_vector_moon[1], unit_vector_moon[0])
    L4 = np.vstack((d * np.cos(theta), d * np.sin(theta), np.zeros_like(theta))).T
    L4 = np.squeeze(L4 @ rotation_matrix.T)
    theta = -np.radians(60) + np.arctan2(unit_vector_moon[1], unit_vector_moon[0])
    L5 = np.vstack((d * np.cos(theta), d * np.sin(theta), np.zeros_like(theta))).T
    L5 = np.squeeze(L5 @ rotation_matrix.T)

    return {
        "L1": L1_from_moon + r,
        "L2": L2_from_moon + r,
        "L3": -r,
        "L4": L4,
        "L5": L5
    }


def lagrange_points_lunar_frame():
    t = Time(["2025-1-1"], scale='utc').gps
    L = lunar_lagrange_points(t)
    return {
        "L1": np.squeeze(gcrf_to_lunar(L["L1"], t)),
        "L2": np.squeeze(gcrf_to_lunar(L["L2"], t)),
        "L3": np.squeeze(gcrf_to_lunar(L["L3"], t)),
        "L4": np.squeeze(gcrf_to_lunar(L["L4"], t)),
        "L5": np.squeeze(gcrf_to_lunar(L["L5"], t)),
    }


def lagrange_points_lunar_fixed_frame():
    t = Time(["2025-1-1"], scale='utc').gps
    L = lunar_lagrange_points(t)
    return {
        "L1": np.squeeze(gcrf_to_lunar_fixed(L["L1"], t)),
        "L2": np.squeeze(gcrf_to_lunar_fixed(L["L2"], t)),
        "L3": np.squeeze(gcrf_to_lunar_fixed(L["L3"], t)),
        "L4": np.squeeze(gcrf_to_lunar_fixed(L["L4"], t)),
        "L5": np.squeeze(gcrf_to_lunar_fixed(L["L5"], t)),
    }
