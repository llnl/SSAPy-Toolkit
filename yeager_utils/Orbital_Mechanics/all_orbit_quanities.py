import numpy as np
from astropy.time import Time
from ssapy import Orbit
from yeager_utils.constants import EARTH_MU


def all_orbital_quantities(
    r=None,
    v=None,
    a=None,
    ecc=None,
    inc=None,
    raan=None,
    aop=None,
    nu=None,
    M=None,
    periapsis=None,
    apoapsis=None,
    t=None,
    mu=EARTH_MU
):
    """
    Compute all orbital elements from any combination of orbit-defining parameters.
    
    Missing angular elements (inc, raan, aop, nu) default to 0 radians.
    Periapsis and apoapsis are measured from the center of the central body.
    All inputs and outputs are in SI units.
    
    Parameters
    ----------
    r : array-like, optional
        Position vector [x, y, z] in meters
    v : array-like, optional
        Velocity vector [vx, vy, vz] in m/s
    a : float, optional
        Semi-major axis in meters
    ecc : float, optional
        Eccentricity (dimensionless)
    inc : float, optional
        Inclination in radians (defaults to 0)
    raan : float, optional
        Right ascension of ascending node in radians (defaults to 0)
    aop : float, optional
        Argument of periapsis in radians (defaults to 0)
    nu : float, optional
        True anomaly in radians (defaults to 0)
    M : float, optional
        Mean anomaly in radians
    periapsis : float, optional
        Periapsis distance from center of central body in meters
    apoapsis : float, optional
        Apoapsis distance from center of central body in meters
    t : float, optional
        Time (GPS seconds). Defaults to current time.
    mu : float, optional
        Gravitational parameter (m^3/s^2). Defaults to Earth's GM.
    
    Returns
    -------
    dict
        Dictionary containing all computed orbital elements (all in SI units):
        - 'r': Position vector [m]
        - 'v': Velocity vector [m/s]
        - 'a': Semi-major axis [m]
        - 'ecc': Eccentricity
        - 'inc': Inclination [rad]
        - 'raan': Right ascension of ascending node [rad]
        - 'aop': Argument of periapsis [rad]
        - 'nu': True anomaly [rad]
        - 'M': Mean anomaly [rad]
        - 'E': Eccentric anomaly [rad]
        - 'period': Orbital period [s]
        - 'n': Mean motion [rad/s]
        - 'periapsis': Periapsis distance from center [m]
        - 'apoapsis': Apoapsis distance from center [m]
        - 'periapsis_vector': Periapsis position vector [m]
        - 'apoapsis_vector': Apoapsis position vector [m]
    
    Notes
    -----
    - Handles circular orbits (ecc=0) without issues
    - For parabolic (ecc=1) and hyperbolic (ecc>1) orbits, some quantities
      (period, apoapsis) may be infinite or undefined
    - If conflicting inputs are provided, priority order is:
      1. Position/velocity vectors (r, v)
      2. Periapsis/apoapsis
      3. Semi-major axis/eccentricity (a, ecc)
    
    Examples
    --------
    # Minimal input with periapsis and apoapsis (all angles default to 0)
    >>> result = all_orbital_quantities(
    ...     periapsis=6571000,
    ...     apoapsis=42164000
    ... )
    
    # From Cartesian state vectors
    >>> result = all_orbital_quantities(
    ...     r=[42164000, 0, 0],
    ...     v=[0, 3074.66, 0]
    ... )
    
    # From Keplerian elements (missing angles default to 0)
    >>> result = all_orbital_quantities(
    ...     a=42164000,
    ...     ecc=0.001,
    ...     inc=np.radians(45)
    ... )
    
    # Moon orbit (pass Moon's mu)
    >>> result = all_orbital_quantities(
    ...     r=r_vec,
    ...     v=v_vec,
    ...     mu=4.9028e12
    ... )
    """
    # Set default time if not provided
    if t is None:
        t = Time.now().gps
    
    # Case 1: Position and velocity vectors provided (highest priority)
    if r is not None and v is not None:
        r = np.array(r)
        v = np.array(v)
        
        # Create orbit from state vectors
        orbit = Orbit(r=r, v=v, t=t)
        orbit._setKeplerian()
        
        result = _extract_all_elements(orbit)
        
    # Case 2: Periapsis and apoapsis provided (second priority)
    elif periapsis is not None and apoapsis is not None:
        # Calculate semi-major axis and eccentricity
        a = (periapsis + apoapsis) / 2.0
        ecc = (apoapsis - periapsis) / (apoapsis + periapsis)
        
        # Use default angles if not provided
        if inc is None:
            inc = 0.0
        if raan is None:
            raan = 0.0
        if aop is None:
            aop = 0.0
        if nu is None and M is None:
            nu = 0.0
        
        # Create orbit
        if nu is not None:
            orbit = Orbit.fromKeplerianElements(
                a, ecc, inc, raan, aop, nu, t=t
            )
        else:  # M is not None
            # Convert mean anomaly to true anomaly
            nu = _mean_to_true_anomaly(M, ecc)
            orbit = Orbit.fromKeplerianElements(
                a, ecc, inc, raan, aop, nu, t=t
            )
        
        orbit._setKeplerian()
        result = _extract_all_elements(orbit)
    
    # Case 3: Semi-major axis and eccentricity provided (third priority)
    elif a is not None and ecc is not None:
        # Use default angles if not provided
        if inc is None:
            inc = 0.0
        if raan is None:
            raan = 0.0
        if aop is None:
            aop = 0.0
        if nu is None and M is None:
            nu = 0.0
        
        # Create orbit
        if nu is not None:
            orbit = Orbit.fromKeplerianElements(
                a, ecc, inc, raan, aop, nu, t=t
            )
        else:  # M is not None
            # Convert mean anomaly to true anomaly
            nu = _mean_to_true_anomaly(M, ecc)
            orbit = Orbit.fromKeplerianElements(
                a, ecc, inc, raan, aop, nu, t=t
            )
        
        orbit._setKeplerian()
        result = _extract_all_elements(orbit)
        
    else:
        raise ValueError(
            "Insufficient parameters provided. Need either:\n"
            "  1. Position (r) and velocity (v) vectors, or\n"
            "  2. Periapsis and apoapsis distances, or\n"
            "  3. Semi-major axis (a) and eccentricity (ecc)\n"
            "Angular elements (inc, raan, aop, nu) default to 0 if not provided."
        )
    
    return result


def _solve_kepler_equation(M, ecc, tol=1e-8, max_iter=100):
    """
    Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E.
    
    Uses Newton-Raphson iteration [1].
    
    Parameters
    ----------
    M : float
        Mean anomaly in radians
    ecc : float
        Eccentricity
    tol : float, optional
        Convergence tolerance (default: 1e-8)
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    
    Returns
    -------
    float
        Eccentric anomaly E in radians
    """
    # Initial guess for E
    E = M if ecc < 0.8 else np.pi
    
    # Newton-Raphson iteration
    for _ in range(max_iter):
        f = E - ecc * np.sin(E) - M
        df = 1 - ecc * np.cos(E)
        dE = -f / df
        E += dE
        
        if np.abs(dE) < tol:
            break
    
    return E


def _eccentric_to_true_anomaly(E, ecc):
    """
    Convert eccentric anomaly to true anomaly [1].
    
    Parameters
    ----------
    E : float
        Eccentric anomaly in radians
    ecc : float
        Eccentricity
    
    Returns
    -------
    float
        True anomaly in radians
    """
    nu = 2 * np.arctan2(
        np.sqrt(1 + ecc) * np.sin(E / 2),
        np.sqrt(1 - ecc) * np.cos(E / 2)
    )
    
    return nu


def _mean_to_true_anomaly(M, ecc):
    """
    Convert mean anomaly to true anomaly [1].
    
    Parameters
    ----------
    M : float
        Mean anomaly in radians
    ecc : float
        Eccentricity
    
    Returns
    -------
    float
        True anomaly in radians
    """
    # First solve Kepler's equation for E
    E = _solve_kepler_equation(M, ecc)
    
    # Then convert E to true anomaly
    nu = _eccentric_to_true_anomaly(E, ecc)
    
    return nu


def _extract_all_elements(orbit):
    """
    Helper function to extract all orbital elements from an Orbit object.
    
    Uses SSAPy's internal Orbit properties [3].
    
    Parameters
    ----------
    orbit : ssapy.Orbit
        Orbit object with computed elements
    
    Returns
    -------
    dict
        Dictionary containing all orbital elements in SI units
    """
    # Calculate periapsis and apoapsis distances (from center of body)
    periapsis_dist = np.linalg.norm(orbit.periapsis)
    
    # Apoapsis can be infinite for hyperbolic orbits
    apoapsis_vec = orbit.apoapsis
    if np.any(np.isinf(apoapsis_vec)):
        apoapsis_dist = np.inf
    else:
        apoapsis_dist = np.linalg.norm(apoapsis_vec)
    
    result = {
        'r': orbit.r,
        'v': orbit.v,
        'a': orbit.a,
        'ecc': orbit.ecc,
        'inc': orbit.inc,
        'raan': orbit.raan,
        'aop': orbit.aop,
        'nu': orbit.nu,
        'M': orbit.M,
        'E': orbit.E,
        'period': orbit.period,
        'n': orbit.n,
        'periapsis': periapsis_dist,
        'apoapsis': apoapsis_dist,
        'periapsis_vector': orbit.periapsis,
        'apoapsis_vector': orbit.apoapsis
    }
    
    return result