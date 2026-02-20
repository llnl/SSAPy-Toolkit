"""
Keplerian orbital element conversions and utilities.

This module provides functions for converting between Keplerian orbital
elements and Cartesian state vectors, as well as computing various
orbital parameters.
"""

import numpy as np
import rebound
from rebound import hash as h
from astropy.time import Time

from ..constants import au_to_m, EARTH_MU, MOON_RADIUS
from ..Coordinates import deg90to90, deg0to360
from ..utils import nby3shape


def hkoe(kElements_or_a, e=None, i=None, ap=None, raan=None, nu=None):
    """
    Convert human-readable Keplerian Orbital Elements (KOE) to SSAPy format.

    Parameters
    ----------
    kElements_or_a : list, array, or float
        If a 6-element iterable [a, e, i, ap, raan, nu], it's treated as the
        full set of elements. If a float, it's treated as the semimajor axis
        (a), followed by 5 more positional arguments.
        Elements are:
        - a: semimajor axis (meters)
        - e: eccentricity (unitless)
        - i: inclination (degrees)
        - ap: argument of periapsis (degrees)
        - raan: right ascension of ascending node (degrees)
        - nu: true anomaly (degrees)
    e, i, ap, raan, nu : float, optional
        Remaining elements if 6 positional arguments are provided (degrees).

    Returns
    -------
    numpy.ndarray
        Array of [a, e, i_rad, ap_rad, raan_rad, nu_rad] with angles in
        radians. Can be unpacked into individual variables or used as array.

    Raises
    ------
    ValueError
        If arguments don't match expected patterns (6-element iterable or
        6 positional args).

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    # Case 1: Single 6-element iterable provided
    if hasattr(kElements_or_a, '__iter__') and not isinstance(
            kElements_or_a, (str, bytes)):
        if len(kElements_or_a) != 6:
            raise ValueError("kElements must be a 6-element iterable")
        a, e, i, ap, raan, nu = kElements_or_a
    # Case 2: 6 positional arguments provided
    elif (e is not None and i is not None and ap is not None and
          raan is not None and nu is not None):
        a = kElements_or_a  # First arg is a
    # Invalid case
    else:
        raise ValueError(
            "Must provide either a 6-element iterable or 6 positional "
            "arguments (a, e, i, ap, raan, nu)")

    # Create array with converted angles
    result = np.array([
        a, e,
        np.radians(i),
        np.radians(ap),
        np.radians(raan),
        np.radians(nu)
    ])
    return result


def get_chance_radius(v, time_step):
    """
    Calculate the chance radius based on the velocity vector.

    Parameters
    ----------
    v : np.ndarray
        The velocity vector of the object at the final time step.
    time_step : float
        The time step between checks in seconds.

    Returns
    -------
    float
        The calculated chance radius in meters.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    return np.linalg.norm(v[-1]) * time_step * 4 + 2 * MOON_RADIUS


def period(a, mu_barycenter=EARTH_MU):
    """
    Calculate the orbital period from the semi-major axis using Kepler's law.

    This function computes the orbital period for a satellite orbiting a
    central body, based on the semi-major axis of the orbit and the
    gravitational parameter of the body (default is Earth).

    Parameters
    ----------
    a : float or np.ndarray
        Semi-major axis (in meters) of the orbit.
    mu_barycenter : float, optional
        Gravitational parameter of the central body (default is Earth's
        gravitational parameter in m^3/s^2).

    Returns
    -------
    float or np.ndarray
        Orbital period(s) in seconds.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    period_seconds = np.sqrt(4 * np.pi**2 / mu_barycenter * a**3)
    return period_seconds


def mean_longitude(longitude_of_ascending_node, argument_of_periapsis,
                   mean_anomaly):
    """
    Calculate mean longitude from orbital angles.

    Parameters
    ----------
    longitude_of_ascending_node : float
        Longitude of ascending node (radians).
    argument_of_periapsis : float
        Argument of periapsis (radians).
    mean_anomaly : float
        Mean anomaly (radians).

    Returns
    -------
    float
        Mean longitude (radians).

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    return (longitude_of_ascending_node + argument_of_periapsis +
            mean_anomaly)


def true_anomaly(eccentricity=None, eccentric_anomaly=None, mean_anomaly=None,
                 true_longitude=None, argument_of_periapsis=None,
                 longitude_of_ascending_node=None):
    """
    Calculate true anomaly from various orbital parameters.

    Parameters
    ----------
    eccentricity : float, optional
        Orbital eccentricity.
    eccentric_anomaly : float, optional
        Eccentric anomaly (radians).
    mean_anomaly : float, optional
        Mean anomaly (radians).
    true_longitude : float, optional
        True longitude (radians).
    argument_of_periapsis : float, optional
        Argument of periapsis (radians).
    longitude_of_ascending_node : float, optional
        Longitude of ascending node (radians).

    Returns
    -------
    float
        True anomaly (radians).

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    if eccentricity is not None and eccentric_anomaly is not None:
        beta = eccentricity / (1 + np.sqrt(1 - eccentricity**2))
        ta = eccentric_anomaly + 2 * np.arctan2(
            beta * np.sin(eccentric_anomaly),
            1 - beta * np.cos(eccentric_anomaly)
        )
    elif eccentricity is not None and mean_anomaly is not None:
        ta = (mean_anomaly +
              (2 * eccentricity - 1 / 4 * eccentricity**3) *
              np.sin(mean_anomaly) +
              5 / 4 * eccentricity**2 * np.sin(2 * mean_anomaly) +
              13 / 12 * eccentricity**3 * np.sin(3 * mean_anomaly))
    elif (true_longitude is not None and
          longitude_of_ascending_node is not None and
          argument_of_periapsis is not None):
        ta = (true_longitude - longitude_of_ascending_node -
              argument_of_periapsis)
    else:
        print('Not enough information provided to calculate true anomaly.')
        return None
    return ta


def rebound_orbital_elements(planet='earth', time="2000-1-1", format='utc'):
    """
    Get orbital elements for planets using REBOUND.

    Parameters
    ----------
    planet : str, optional
        Planet name or "all" for all planets (default: 'earth').
    time : str, optional
        Time string (default: "2000-1-1").
    format : str, optional
        Time format (default: 'utc').

    Returns
    -------
    dict
        Dictionary of orbital elements.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    if not isinstance(time, Time):
        time = Time(time, format=format)
    jd = time.jd

    try:
        if isinstance(jd, (float, int)):
            jd = str(jd)
        if jd[0:2] == 'JD':
            pass
        else:
            jd = f'JD{jd}'
    except (IndexError, TypeError):
        print('Error with the date provided. Give a year/month/day or JD.')
        return None

    sim = rebound.Simulation()
    sim.add("sun", date=jd, hash=0)
    sim.add("mercury", date=jd, hash=1)
    sim.add("venus", date=jd, hash=2)
    sim.add("earth", date=jd, hash=3)
    sim.add("mars", date=jd, hash=4)
    sim.add("jupiter", date=jd, hash=5)
    sim.add("saturn", date=jd, hash=6)
    sim.add("uranus", date=jd, hash=7)
    sim.add("neptune", date=jd, hash=8)
    sim.move_to_com()

    if planet.lower() == "all":
        orbitals = {}
        for i, planet_str in enumerate(['mercury', 'venus', 'earth', 'mars',
                                        'jupiter', 'saturn', 'uranus',
                                        'neptune']):
            p = sim.particles[h(i + 1)]
            orbitals[planet_str] = {
                'a': p.a,
                'e': p.e,
                'i': p.inc,
                'true_longitude': p.theta,
                'argument_of_pericenter': p.omega,
                'longitude_of_ascending_node': p.Omega,
                'true_anomaly': p.f,
                'longitude_of_pericenter': p.pomega,
                'mean_longitude': p.l,
                'mean_anomaly': p.M,
                'eccentric_anomaly': rebound.M_to_E(p.e, p.M)
            }
        return orbitals

    # Map planet names to indices
    planet_indices = {
        "sun": 0, "mercury": 1, "venus": 2, "earth": 3,
        "mars": 4, "jupiter": 5, "saturn": 6, "uranus": 7, "neptune": 8
    }

    index = planet_indices.get(planet.lower())
    if index is None:
        raise ValueError(f"Unknown planet: {planet}")

    p = sim.particles[h(index)]
    return {
        'a': p.a,
        'e': p.e,
        'i': p.inc,
        'true_longitude': p.theta,
        'argument_of_pericenter': p.omega,
        'longitude_of_ascending_node': p.Omega,
        'true_anomaly': p.f,
        'longitude_of_pericenter': p.pomega,
        'mean_longitude': p.l,
        'mean_anomaly': p.M,
        'eccentric_anomaly': rebound.M_to_E(p.e, p.M)
    }


def j2000_orbitals(planet='earth', Teph=2451545.0):
    """
    Calculate JPL orbital elements using J2000 approximations.

    Reference: https://ssd.jpl.nasa.gov/planets/approx_pos.html
    Valid for time interval 1800 AD - 2050 AD.

    Parameters
    ----------
    planet : str, optional
        Planet name (default: 'earth').
    Teph : float, optional
        Epoch in Julian Date (default: 2451545.0 = J2000.0).

    Returns
    -------
    dict
        Dictionary with keys: 'a', 'e', 'i', 'mean_longitude',
        'longitude_of_perihelion', 'longitude_of_the_ascending_node'.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    # Orbital element coefficients (table 1 from JPL)
    elements = {
        'mercury': (0.38709927, 0.20563593, 7.00497902, 252.25032350,
                    77.45779628, 48.33076593, 0.00000037, 0.00001906,
                    -0.00594749, 149472.67411175, 0.16047689, -0.12534081),
        'venus': (0.72333566, 0.00677672, 3.39467605, 181.97909950,
                  131.60246718, 76.67984255, 0.00000390, -0.00004107,
                  -0.00078890, 58517.81538729, 0.00268329, -0.27769418),
        'earth': (1.00000261, 0.01671123, -0.00001531, 100.46457166,
                  102.93768193, 0.0, 0.00000562, -0.00004392, -0.01294668,
                  35999.37244981, 0.32327364, 0.0),
        'mars': (1.52371034, 0.09339410, 1.84969142, -4.55343205,
                 -23.94362959, 49.55953891, 0.00001847, 0.00007882,
                 -0.00813131, 19140.30268499, 0.44441088, -0.29257343),
        'jupiter': (5.20288700, 0.04838624, 1.30439695, 34.39644051,
                    14.72847983, 100.47390909, -0.00011607, -0.00013253,
                    -0.00183714, 3034.74612775, 0.21252668, 0.20469106),
        'saturn': (9.53667594, 0.05386179, 2.48599187, 49.95424423,
                   92.59887831, 113.66242448, -0.00125060, -0.00050991,
                   0.00193609, 1222.49362201, -0.41897216, -0.28867794),
        'uranus': (19.18916464, 0.04725744, 0.77263783, 313.23810451,
                   170.95427630, 74.01692503, -0.00196176, -0.00004397,
                   -0.00242939, 428.48202785, 0.40805281, 0.04240589),
        'neptune': (30.06992276, 0.00859048, 1.77004347, -55.12002969,
                    44.96476227, 131.78422574, 0.00026291, 0.00005105,
                    0.00035372, 218.45945325, -0.32241464, -0.00508664)
    }

    if planet.lower() not in elements:
        raise ValueError(f"Unknown planet: {planet}")

    (anaut, enaut, inaut, Lnaut, onaut, Onaut,
     arate, erate, irate, Lrate, orate, Orate) = elements[planet.lower()]

    # Number of centuries after J2000
    T = (Teph - 2451545.0) / 36525

    a = anaut + arate * T
    e = enaut + erate * T
    i = inaut + irate * T
    mean_longitude_val = Lnaut + Lrate * T
    longitude_of_perihelion = onaut + orate * T
    longitude_of_the_ascending_node = Onaut + Orate * T

    return {
        'a': a,
        'e': e,
        'i': deg90to90(i),
        'mean_longitude': deg0to360(mean_longitude_val),
        'longitude_of_perihelion': deg0to360(longitude_of_perihelion),
        'longitude_of_the_ascending_node': deg0to360(
            longitude_of_the_ascending_node)
    }


def kepler_to_state(a=1, e=0, i=0, ap=0, raan=0, nu=0, mu=EARTH_MU):
    """
    Convert Keplerian orbital elements to state vectors using broadcasting.

    If any inputs are invalid (e.g., NaN or out-of-range values), the
    corresponding output is replaced with the average of the surrounding
    valid outputs.

    Parameters
    ----------
    a : float or array-like
        Semi-major axis (m). Can be a single value or an array of values.
    e : float or array-like
        Eccentricity (dimensionless). Must be in [0, 1).
    i : float or array-like
        Inclination (rad). Must be in [0, π].
    raan : float or array-like
        Right ascension of the ascending node (rad). Must be in [0, 2π].
    ap : float or array-like
        Argument of perigee (rad). Must be in [0, 2π].
    nu : float or array-like
        True anomaly (rad). Must be in [0, 2π].
    mu : float, optional
        Gravitational parameter (m^3/s^2). Defaults to EARTH_MU.

    Returns
    -------
    tuple
        A tuple containing:
        - r : ndarray
            Position vector(s) in inertial frame (m). Shape is (3,) for
            single set of orbital elements or (N, 3) for multiple sets.
        - v : ndarray
            Velocity vector(s) in inertial frame (m/s). Shape is (3,) for
            single set of orbital elements or (N, 3) for multiple sets.

    Notes
    -----
    - This function supports both single sets and arrays of orbital elements.
    - Invalid inputs (NaN or out-of-range) are handled by replacing with
      average of surrounding valid outputs.
    - Assumes all input values are in the same inertial reference frame.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    # Ensure inputs are arrays
    a = np.atleast_1d(a)
    e = np.atleast_1d(e)
    i = np.atleast_1d(i)
    raan = np.atleast_1d(raan)
    ap = np.atleast_1d(ap)
    nu = np.atleast_1d(nu)

    # Validate inputs
    valid_mask = (a > 0) & (e >= 0) & (e < 1)
    if not np.all(valid_mask):
        print("Warning: Some inputs are invalid. Their outputs will be "
              "averaged from valid neighbors.")

    # Compute position and velocity in the perifocal frame
    cos_nu = np.cos(nu)
    sin_nu = np.sin(nu)
    r_pf = ((a * (1 - e**2) / (1 + e * cos_nu))[:, None] *
            np.stack([cos_nu, sin_nu, np.zeros_like(cos_nu)], axis=-1))
    v_pf = ((np.sqrt(mu / a)[:, None]) *
            np.stack([-sin_nu, e + cos_nu, np.zeros_like(sin_nu)], axis=-1))

    # Precompute trigonometric terms
    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)
    cos_w = np.cos(ap)
    sin_w = np.sin(ap)
    cos_i = np.cos(i)
    sin_i = np.sin(i)

    # Rotation matrix components
    R11 = cos_raan * cos_w - sin_raan * cos_i * sin_w
    R12 = -cos_raan * sin_w - sin_raan * cos_i * cos_w
    R13 = sin_raan * sin_i
    R21 = sin_raan * cos_w + cos_raan * cos_i * sin_w
    R22 = -sin_raan * sin_w + cos_raan * cos_i * cos_w
    R23 = -cos_raan * sin_i
    R31 = sin_i * sin_w
    R32 = sin_i * cos_w
    R33 = cos_i

    # Full rotation matrix (3x3xN for N sets of elements)
    R_pf_i = np.stack([np.stack([R11, R12, R13], axis=-1),
                       np.stack([R21, R22, R23], axis=-1),
                       np.stack([R31, R32, R33], axis=-1)], axis=-2)

    # Transform position and velocity to inertial frame
    r = np.einsum('ijk,ik->ij', R_pf_i, r_pf)
    v = np.einsum('ijk,ik->ij', R_pf_i, v_pf)

    # Handle invalid inputs by averaging surrounding valid outputs
    if not np.all(valid_mask):
        for idx, valid in enumerate(valid_mask):
            if not valid:
                # Find surrounding valid indices
                left = idx - 1
                right = idx + 1
                while left >= 0 and not valid_mask[left]:
                    left -= 1
                while right < len(valid_mask) and not valid_mask[right]:
                    right += 1

                # Average valid neighbors
                if left >= 0 and right < len(valid_mask):
                    r[idx] = (r[left] + r[right]) / 2
                    v[idx] = (v[left] + v[right]) / 2
                elif left >= 0:
                    r[idx] = r[left]
                    v[idx] = v[left]
                elif right < len(valid_mask):
                    r[idx] = r[right]
                    v[idx] = v[right]

    # Return single output if input was single
    if r.shape[0] == 1:
        return r[0], v[0]

    return r, v


def kepler_to_state_loop(a=1, e=0, i=0, ap=0, raan=0, nu=0, mu=EARTH_MU):
    """
    Convert Keplerian orbital elements to state vectors using loop.

    Parameters
    ----------
    a : float or array-like
        Semi-major axis (m).
    e : float or array-like
        Eccentricity (dimensionless).
    i : float or array-like
        Inclination (rad).
    raan : float or array-like
        Right ascension of the ascending node (rad).
    ap : float or array-like
        Argument of perigee (rad).
    nu : float or array-like
        True anomaly (rad).
    mu : float, optional
        Gravitational parameter (m^3/s^2). Defaults to EARTH_MU.

    Returns
    -------
    tuple
        A tuple containing:
        - r : ndarray
            Position vector(s) in inertial frame (m).
        - v : ndarray
            Velocity vector(s) in inertial frame (m/s).

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    # Ensure inputs are arrays for consistency
    single_input = False
    if not isinstance(a, np.ndarray):
        single_input = True
        a = np.array([a])
        e = np.array([e])
        i = np.array([i])
        raan = np.array([raan])
        ap = np.array([ap])
        nu = np.array([nu])

    # Check validity of inputs
    if np.any(a <= 0):
        raise ValueError("Semi-major axis 'a' must be positive.")
    if np.any((e < 0) | (e >= 1)):
        raise ValueError("Eccentricity 'e' must be in the range [0, 1).")

    # Initialize arrays for results
    r_list = []
    v_list = []

    for ai, ei, ii, raani, wi, nui in zip(a, e, i, raan, ap, nu):
        # Compute position vector in perifocal frame
        r_pf = (ai * (1 - ei**2) / (1 + ei * np.cos(nui)) *
                np.array([np.cos(nui), np.sin(nui), 0]))

        # Compute velocity vector in perifocal frame
        v_pf = (np.sqrt(mu / ai) *
                np.array([-np.sin(nui), ei + np.cos(nui), 0]))

        # Create rotation matrix from perifocal to inertial frame
        R_pf_i = np.array([
            [np.cos(raani) * np.cos(wi) - np.sin(raani) * np.cos(ii) *
             np.sin(wi),
             -np.cos(raani) * np.sin(wi) - np.sin(raani) * np.cos(ii) *
             np.cos(wi),
             np.sin(raani) * np.sin(ii)],
            [np.sin(raani) * np.cos(wi) + np.cos(raani) * np.cos(ii) *
             np.sin(wi),
             -np.sin(raani) * np.sin(wi) + np.cos(raani) * np.cos(ii) *
             np.cos(wi),
             -np.cos(raani) * np.sin(ii)],
            [np.sin(ii) * np.sin(wi),
             np.sin(ii) * np.cos(wi),
             np.cos(ii)]
        ])

        # Transform position and velocity to inertial frame
        r = np.dot(R_pf_i, r_pf)
        v = np.dot(R_pf_i, v_pf)

        r_list.append(r)
        v_list.append(v)

    # Convert results to arrays
    r_array = np.array(r_list)
    v_array = np.array(v_list)

    # Return single output if input was single
    if single_input:
        return r_array[0], v_array[0]

    return r_array, v_array


def state_to_kepler(r, v, mu=EARTH_MU):
    """
    Convert state vector (position and velocity) to Keplerian elements.

    Parameters
    ----------
    r : array-like
        Position vector in inertial frame (m). Can be single vector (3,)
        or array of vectors (N, 3).
    v : array-like
        Velocity vector in inertial frame (m/s). Can be single vector (3,)
        or array of vectors (N, 3).
    mu : float, optional
        Gravitational parameter (m^3/s^2). Defaults to EARTH_MU.

    Returns
    -------
    tuple
        A tuple containing Keplerian orbital elements:
        - a : float or ndarray - Semi-major axis (m).
        - e : float or ndarray - Eccentricity (dimensionless).
        - i : float or ndarray - Inclination (rad).
        - ap : float or ndarray - Argument of perigee (rad).
        - raan : float or ndarray - RAAN (rad).
        - nu : float or ndarray - True anomaly (rad).

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    # Compute angular momentum vector
    h = np.cross(r, v)

    # Compute eccentricity vector
    e_vec = (np.cross(v, h) / mu) - r / np.linalg.norm(r)

    # Compute semi-major axis
    a = 1 / (2 / np.linalg.norm(r) - np.linalg.norm(v)**2 / mu)

    # Compute eccentricity
    e = np.linalg.norm(e_vec)

    # Compute inclination
    i = np.arccos(h[2] / np.linalg.norm(h))

    # Compute right ascension of the ascending node
    h_xy_norm = np.linalg.norm(h[:2])
    if h_xy_norm < 1e-10:  # Tolerance for equatorial orbit
        raan = 0  # RAAN is undefined, set to 0
    else:
        if h[0] >= 0:
            raan = np.arccos(h[0] / h_xy_norm)
        else:
            raan = 2 * np.pi - np.arccos(h[0] / h_xy_norm)

    # Compute argument of perigee
    if e_vec[2] >= 0:
        ap = np.arccos(np.dot(h, e_vec) / (np.linalg.norm(h) * e))
    else:
        ap = 2 * np.pi - np.arccos(np.dot(h, e_vec) / (np.linalg.norm(h) * e))

    # Compute true anomaly
    if np.dot(r, v) >= 0:
        nu = np.arccos(np.dot(e_vec, r) / (e * np.linalg.norm(r)))
    else:
        nu = 2 * np.pi - np.arccos(np.dot(e_vec, r) / (e * np.linalg.norm(r)))

    return a, e, i, ap, raan, nu


def kepler_to_parametric(a, e, i, omega, ap, theta):
    """
    Convert Keplerian elements to parametric coordinates.

    Parameters
    ----------
    a : float
        Semi-major axis.
    e : float
        Eccentricity.
    i : float
        Inclination (degrees).
    omega : float
        Longitude of ascending node (degrees).
    ap : float
        Argument of periapsis (degrees).
    theta : float
        True anomaly (degrees).

    Returns
    -------
    tuple
        (x_final, y_final, z_final) parametric coordinates.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    # Convert to radians
    i = np.radians(i)
    omega = np.radians(omega)
    ap = np.radians(ap)
    theta = np.radians(theta)

    # Compute the semi-major and semi-minor axes
    b = a * np.sqrt(1 - e**2)

    # Compute the parametric coefficients
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    z = 0

    # Rotate the ellipse about the x-axis
    x_prime = x
    y_prime = y * np.cos(i) - z * np.sin(i)
    z_prime = y * np.sin(i) + z * np.cos(i)

    # Rotate the ellipse about the z-axis
    x_prime_prime = x_prime * np.cos(omega) - y_prime * np.sin(omega)
    y_prime_prime = x_prime * np.sin(omega) + y_prime * np.cos(omega)
    z_prime_prime = z_prime

    # Translate the ellipse
    x_final = x_prime_prime + ap
    y_final = y_prime_prime
    z_final = z_prime_prime

    return x_final, y_final, z_final


def calculate_orbital_elements(r_, v_, mu_barycenter=EARTH_MU):
    """
    Calculate classical orbital elements from position and velocity.

    Parameters
    ----------
    r_ : array-like
        Position vector(s) in Cartesian coordinates (m).
        Can be single vector or array of vectors.
    v_ : array-like
        Velocity vector(s) in Cartesian coordinates (m/s).
        Can be single vector or array of vectors.
    mu_barycenter : float, optional
        Gravitational parameter (m^3/s^2). Defaults to EARTH_MU.

    Returns
    -------
    dict
        Dictionary containing the following orbital elements:
        - 'a' : ndarray - Semi-major axis (m).
        - 'e' : ndarray - Eccentricity (dimensionless).
        - 'i' : ndarray - Inclination (rad).
        - 'tl' : ndarray - True longitude (rad).
        - 'ap' : ndarray - Argument of periapsis (rad).
        - 'raan' : ndarray - Longitude of ascending node (rad).
        - 'ta' : ndarray - True anomaly (rad).
        - 'L' : ndarray - Specific angular momentum magnitude (m^2/s).

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    mu_ = mu_barycenter
    rarr = nby3shape(r_)
    varr = nby3shape(v_)

    aarr = []
    earr = []
    incarr = []
    true_longitudearr = []
    argument_of_periapsisarr = []
    longitude_of_ascending_nodearr = []
    true_anomalyarr = []
    hmagarr = []

    for r, v in zip(rarr, varr):
        r = np.array(r)
        v = np.array(v)

        rmag = np.sqrt(r.dot(r))
        vmag = np.sqrt(v.dot(v))

        h = np.cross(r, v)
        hmag = np.sqrt(h.dot(h))
        n = np.cross(np.array([0, 0, 1]), h)

        a = 1 / ((2 / rmag) - (vmag**2) / mu_)

        evector = np.cross(v, h) / (mu_) - r / rmag
        e = np.sqrt(evector.dot(evector))

        inc = np.arccos(h[2] / hmag)

        if np.dot(r, v) > 0:
            true_anomaly_val = np.arccos(np.dot(evector, r) / (e * rmag))
        else:
            true_anomaly_val = (2 * np.pi - np.arccos(np.dot(evector, r) / (e * rmag)))

        if evector[2] >= 0:
            argument_of_periapsis = np.arccos(
                np.dot(n, evector) / (e * np.sqrt(n.dot(n))))
        else:
            argument_of_periapsis = (2 * np.pi - np.arccos(np.dot(n, evector) / (e * np.sqrt(n.dot(n)))))

        if n[1] >= 0:
            longitude_of_ascending_node = np.arccos(n[0] / np.sqrt(n.dot(n)))
        else:
            longitude_of_ascending_node = (2 * np.pi - np.arccos(n[0] / np.sqrt(n.dot(n))))

        true_longitude_val = (true_anomaly_val + argument_of_periapsis +
                              longitude_of_ascending_node)

        aarr.append(a)
        earr.append(e)
        incarr.append(inc)
        true_longitudearr.append(true_longitude_val)
        argument_of_periapsisarr.append(argument_of_periapsis)
        longitude_of_ascending_nodearr.append(longitude_of_ascending_node)
        true_anomalyarr.append(true_anomaly_val)
        hmagarr.append(hmag)

    return {
        'a': np.array(aarr),
        'e': np.array(earr),
        'i': np.array(incarr),
        'tl': np.array(true_longitudearr),
        'ap': np.array(argument_of_periapsisarr),
        'raan': np.array(longitude_of_ascending_nodearr),
        'ta': np.array(true_anomalyarr),
        'L': np.array(hmagarr)
    }


def a_from_periap(rp, ra):
    """
    Compute semi-major axis (a) from perigee (rp) and apogee (ra) distances.

    Parameters
    ----------
    rp : float
        Perigee distance (m), measured from center of the body.
    ra : float
        Apogee distance (m), measured from center of the body.

    Returns
    -------
    float
        Semi-major axis (a) in meters.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    if rp <= 0 or ra <= 0:
        raise ValueError("Perigee and apogee distances must be positive.")
    if ra < rp:
        raise ValueError("Apogee distance must be >= perigee distance.")
    return (rp + ra) / 2.0


def e_from_periap(rp, ra):
    """
    Compute eccentricity (e) from perigee (rp) and apogee (ra) distances.

    Parameters
    ----------
    rp : float
        Perigee distance (m), measured from center of the body.
    ra : float
        Apogee distance (m), measured from center of the body.

    Returns
    -------
    float
        Eccentricity (e), unitless (0 <= e < 1 for an ellipse).

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    if rp <= 0 or ra <= 0:
        raise ValueError("Perigee and apogee distances must be positive.")
    if ra < rp:
        raise ValueError("Apogee distance must be >= perigee distance.")
    return (ra - rp) / (ra + rp)


def ae_from_periap(rp, ra):
    """
    Compute both semi-major axis and eccentricity from apsides.

    Parameters
    ----------
    rp : float
        Perigee distance (m).
    ra : float
        Apogee distance (m).

    Returns
    -------
    tuple
        (a, e) - semi-major axis and eccentricity.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    return a_from_periap(rp, ra), e_from_periap(rp, ra)


def periapsis(a, e):
    """
    Calculate periapsis distance from semi-major axis and eccentricity.

    Parameters
    ----------
    a : float
        Semi-major axis (m).
    e : float
        Eccentricity.

    Returns
    -------
    float
        Periapsis distance (m).

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    return (1 - e) * a


def apoapsis(a, e):
    """
    Calculate apoapsis distance from semi-major axis and eccentricity.

    Parameters
    ----------
    a : float
        Semi-major axis (m).
    e : float
        Eccentricity.

    Returns
    -------
    float
        Apoapsis distance (m).

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    return (1 + e) * a


def peri_apo_from_rv(perigee, apogee):
    """
    Compute semi-major axis and eccentricity from apsides.

    Parameters
    ----------
    perigee : float
        Perigee distance (m).
    apogee : float
        Apogee distance (m).

    Returns
    -------
    dict
        Dictionary with keys 'a' and 'e'.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    # Semi-major axis
    a = (perigee + apogee) / 2
    # Eccentricity
    e = (apogee - perigee) / (apogee + perigee)
    return {"a": a, "e": e}


def peri_apo_apsis_from_rv(r, v):
    """
    Compute periapsis and apoapsis from state vectors.

    Parameters
    ----------
    r : array-like
        Position vector (m).
    v : array-like
        Velocity vector (m/s).

    Returns
    -------
    dict
        Dictionary with keys 'periapsis' and 'apoapsis'.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    temp = calculate_orbital_elements(r, v, EARTH_MU)
    return {
        "periapsis": periapsis(temp['a'], temp['e']),
        "apoapsis": apoapsis(temp['a'], temp['e'])
    }


def vcircular(r=au_to_m, mu_=1.32712440018e20 + 2.2032e13 + 3.24859e14):
    """
    Calculate circular orbital velocity.

    Parameters
    ----------
    r : float, optional
        Orbital radius (m). Defaults to 1 AU.
    mu_ : float, optional
        Gravitational parameter (m^3/s^2). Defaults to solar system total.

    Returns
    -------
    float
        Circular orbital velocity (m/s).

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    return np.sqrt(mu_ / r)


def vis_viva(a, r, mu):
    """
    Calculate orbital velocity using vis-viva equation.

    Parameters
    ----------
    a : float
        Semi-major axis (m).
    r : float
        Current orbital radius (m).
    mu : float
        Gravitational parameter (m^3/s^2).

    Returns
    -------
    float
        Orbital velocity (m/s).

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    return np.sqrt(mu * (2.0 / r - 1.0 / a))


def v_periapsis(a, rp, mu):
    """
    Calculate velocity at periapsis.

    Parameters
    ----------
    a : float
        Semi-major axis (m).
    rp : float
        Periapsis distance (m).
    mu : float
        Gravitational parameter (m^3/s^2).

    Returns
    -------
    float
        Velocity at periapsis (m/s).

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    return np.sqrt(mu * (2.0 / rp - 1.0 / a))


def apapsis_from_a_rp(a, rp):
    """
    Calculate apoapsis from semi-major axis and periapsis.

    Parameters
    ----------
    a : float
        Semi-major axis (m).
    rp : float
        Periapsis distance (m).

    Returns
    -------
    float
        Apoapsis distance (m).

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    return 2.0 * a - rp
