import numpy as np
from astropy.time import Time
from ssapy import Orbit
from ssapy.utils import meanAnomaly2trueAnomaly
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
    
    Parameters
    ----------
    r : array-like, optional
        Position vector [x, y, z] in meters
    v : array-like, optional
        Velocity vector [vx, vy, vz] in m/s
    a : float, optional
        Semi-major axis in meters [1]
    ecc : float, optional
        Eccentricity (dimensionless) [1]
    inc : float, optional
        Inclination in radians (defaults to 0) [1]
    raan : float, optional
        Right ascension of ascending node in radians (defaults to 0) [1]
    aop : float, optional
        Argument of periapsis in radians (defaults to 0) [2]
    nu : float, optional
        True anomaly in radians (defaults to 0) [2]
    M : float, optional
        Mean anomaly in radians
    periapsis : float, optional
        Periapsis distance from center of central body in meters
    apoapsis : float, optional
        Apoapsis distance from center of central body in meters
    t : float, optional
        Time (GPS seconds). Defaults to current time.
    mu : float, optional
        Gravitational parameter (m^3/s^2). Defaults to Earth's GM [2][3]
    
    Returns
    -------
    dict
        Dictionary containing all computed orbital elements [1][2][3]:
        - 'r': Position vector [m]
        - 'v': Velocity vector [m/s]
        - 'a': Semi-major axis [m] [1]
        - 'ecc': Eccentricity [1]
        - 'inc': Inclination [rad] [1]
        - 'raan': Right ascension of ascending node [rad] [2]
        - 'aop': Argument of periapsis [rad] [2]
        - 'nu': True anomaly [rad] [2]
        - 'M': Mean anomaly [rad]
        - 'E': Eccentric anomaly [rad]
        - 'period': Orbital period [s]
        - 'n': Mean motion [rad/s]
        - 'periapsis': Periapsis distance from center [m]
        - 'apoapsis': Apoapsis distance from center [m]
        - 'periapsis_vector': Periapsis position vector [m]
        - 'apoapsis_vector': Apoapsis position vector [m]
    
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
    
    # Case 1: Position and velocity vectors provided
    if r is not None and v is not None:
        r = np.array(r)
        v = np.array(v)
        
        # Create orbit from state vectors
        orbit = Orbit(r=r, v=v, t=t)
        orbit._setKeplerian()
        
        result = _extract_all_elements(orbit)
        
    # Case 2: Periapsis and apoapsis provided
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
            nu = meanAnomaly2trueAnomaly(M, ecc)
            orbit = Orbit.fromKeplerianElements(
                a, ecc, inc, raan, aop, nu, t=t
            )
        
        orbit._setKeplerian()
        result = _extract_all_elements(orbit)
    
    # Case 3: Semi-major axis and eccentricity provided
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
            nu = meanAnomaly2trueAnomaly(M, ecc)
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


def _extract_all_elements(orbit):
    """
    Helper function to extract all orbital elements from an Orbit object.
    
    Parameters
    ----------
    orbit : ssapy.Orbit
        Orbit object with computed elements
    
    Returns
    -------
    dict
        Dictionary containing all orbital elements
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