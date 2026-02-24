import numpy as np
from astropy.time import Time
from ssapy import Orbit
from yeager_utils.constants import EARTH_MU


def all_orbital_quantities(
    r=None,
    v=None,
    a=None,
    e=None,
    i=None,
    pa=None,
    raan=None,
    ta=None,
    ma=None,
    periapsis=None,
    apoapsis=None,
    t=None,
    mu=EARTH_MU
):
    """
    Compute all orbital elements from any combination of orbit-defining parameters.
    
    Missing angular elements (i, raan, pa, ta) default to 0 radians.
    Periapsis and apoapsis are measured from the center of the central body.
    All inputs and outputs are in SI units.
    
    Parameter names match SSAPy conventions [17][19]:
    - a: semi-major axis
    - e: eccentricity  
    - i: inclination
    - pa: argument of periapsis (perigee argument)
    - raan: right ascension of ascending node
    - ta: true anomaly
    - ma: mean anomaly
    
    Parameters
    ----------
    r : array-like, optional
        Position vector [x, y, z] in meters
    v : array-like, optional
        Velocity vector [vx, vy, vz] in m/s
    a : float, optional
        Semi-major axis in meters
    e : float, optional
        Eccentricity (dimensionless)
    i : float, optional
        Inclination in radians (defaults to 0)
    raan : float, optional
        Right ascension of ascending node in radians (defaults to 0)
    pa : float, optional
        Argument of periapsis in radians (defaults to 0)
    ta : float, optional
        True anomaly in radians (defaults to 0)
    ma : float, optional
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
        - 'e': Eccentricity
        - 'i': Inclination [rad]
        - 'pa': Argument of periapsis [rad]
        - 'raan': Right ascension of ascending node [rad]
        - 'ta': True anomaly [rad]
        - 'ma': Mean anomaly [rad]
        - 'ea': Eccentric anomaly [rad]
        - 'period': Orbital period [s]
        - 'n': Mean motion [rad/s]
        - 'periapsis': Periapsis distance from center [m]
        - 'apoapsis': Apoapsis distance from center [m]
    
    Notes
    -----
    - Handles circular orbits (e=0) without issues
    - For parabolic (e=1) and hyperbolic (e>1) orbits, some quantities
      (period, apoapsis) may be infinite or undefined
    - If conflicting inputs are provided, priority order is:
      1. Position/velocity vectors (r, v)
      2. Periapsis/apoapsis
      3. Semi-major axis/eccentricity (a, e)
    
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
    ...     e=0.001,
    ...     i=np.radians(45)
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
        
        # Create orbit from state vectors [19]
        orbit = Orbit(r=r, v=v, t=t, mu=mu)
        
        result = _extract_all_elements(orbit)
        
    # Case 2: Periapsis and apoapsis provided (second priority)
    elif periapsis is not None and apoapsis is not None:
        # Calculate semi-major axis and eccentricity
        a = (periapsis + apoapsis) / 2.0
        e = (apoapsis - periapsis) / (apoapsis + periapsis)
        
        # Use default angles if not provided
        if i is None:
            i = 0.0
        if raan is None:
            raan = 0.0
        if pa is None:
            pa = 0.0
        if ta is None and ma is None:
            ta = 0.0
        
        # Create orbit using Keplerian elements [17][19]
        if ta is not None:
            orbit = Orbit.fromKeplerianElements(
                a, e, i, raan, pa, ta, t=t, mu=mu
            )
        else:  # ma is not None
            # Convert mean anomaly to true anomaly
            ta = _mean_to_true_anomaly(ma, e)
            orbit = Orbit.fromKeplerianElements(
                a, e, i, raan, pa, ta, t=t, mu=mu
            )
        
        result = _extract_all_elements(orbit)
    
    # Case 3: Semi-major axis and eccentricity provided (third priority)
    elif a is not None and e is not None:
        # Use default angles if not provided
        if i is None:
            i = 0.0
        if raan is None:
            raan = 0.0
        if pa is None:
            pa = 0.0
        if ta is None and ma is None:
            ta = 0.0
        
        # Create orbit using Keplerian elements [17][19]
        if ta is not None:
            orbit = Orbit.fromKeplerianElements(
                a, e, i, raan, pa, ta, t=t, mu=mu
            )
        else:  # ma is not None
            # Convert mean anomaly to true anomaly
            ta = _mean_to_true_anomaly(ma, e)
            orbit = Orbit.fromKeplerianElements(
                a, e, i, raan, pa, ta, t=t, mu=mu
            )
        
        result = _extract_all_elements(orbit)
        
    else:
        raise ValueError(
            "Insufficient parameters provided. Need either:\n"
            "  1. Position (r) and velocity (v) vectors, or\n"
            "  2. Periapsis and apoapsis distances, or\n"
            "  3. Semi-major axis (a) and eccentricity (e)\n"
            "Angular elements (i, raan, pa, ta) default to 0 if not provided."
        )
    
    return result


def _solve_kepler_equation(M, e, tol=1e-8, max_iter=100):
    """
    Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E.
    
    Uses Newton-Raphson iteration [11].
    
    Parameters
    ----------
    M : float
        Mean anomaly in radians
    e : float
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
    E = M if e < 0.8 else np.pi
    
    # Newton-Raphson iteration
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        df = 1 - e * np.cos(E)
        dE = -f / df
        E += dE
        
        if np.abs(dE) < tol:
            break
    
    return E


def _eccentric_to_true_anomaly(E, e):
    """
    Convert eccentric anomaly to true anomaly [11].
    
    Parameters
    ----------
    E : float
        Eccentric anomaly in radians
    e : float
        Eccentricity
    
    Returns
    -------
    float
        True anomaly in radians
    """
    nu = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )
    
    return nu


def _mean_to_true_anomaly(M, e):
    """
    Convert mean anomaly to true anomaly [11].
    
    Parameters
    ----------
    M : float
        Mean anomaly in radians
    e : float
        Eccentricity
    
    Returns
    -------
    float
        True anomaly in radians
    """
    # First solve Kepler's equation for E
    E = _solve_kepler_equation(M, e)
    
    # Then convert E to true anomaly
    nu = _eccentric_to_true_anomaly(E, e)
    
    return nu


def _extract_all_elements(orbit):
    """
    Helper function to extract all orbital elements from an Orbit object.
    
    Uses SSAPy's Orbit.keplerianElements property which returns (a, e, i, pa, raan, ta) [17].
    
    Parameters
    ----------
    orbit : ssapy.Orbit
        Orbit object with computed elements
    
    Returns
    -------
    dict
        Dictionary containing all orbital elements in SI units
    """
    # Get Keplerian elements tuple: (a, e, i, pa, raan, ta) [17]
    a_val, e_val, i_val, pa_val, raan_val, ta_val = orbit.keplerianElements
    
    # Calculate orbital period: T = 2π√(a³/μ)
    if a_val > 0 and e_val < 1.0:  # Elliptical orbit
        period = 2 * np.pi * np.sqrt(a_val**3 / orbit.mu)
        n = 2 * np.pi / period  # Mean motion
    else:
        period = np.inf
        n = 0.0
    
    # Calculate periapsis and apoapsis
    if e_val < 1.0:  # Elliptical orbit
        periapsis_dist = a_val * (1 - e_val)
        apoapsis_dist = a_val * (1 + e_val)
    elif e_val == 1.0:  # Parabolic
        periapsis_dist = a_val * (1 - e_val) if a_val > 0 else np.inf
        apoapsis_dist = np.inf
    else:  # Hyperbolic
        periapsis_dist = abs(a_val) * (1 - e_val)
        apoapsis_dist = np.inf
    
    # Calculate eccentric and mean anomalies for elliptical orbits
    if e_val < 1.0:  # Elliptical
        # Eccentric anomaly from true anomaly [11]
        ea_val = 2 * np.arctan2(
            np.sqrt(1 - e_val) * np.sin(ta_val / 2),
            np.sqrt(1 + e_val) * np.cos(ta_val / 2)
        )
        # Mean anomaly from eccentric anomaly
        ma_val = ea_val - e_val * np.sin(ea_val)
    else:
        ea_val = np.nan
        ma_val = np.nan
    
    result = {
        'r': orbit.r,
        'v': orbit.v,
        'a': a_val,
        'e': e_val,
        'i': i_val,
        'pa': pa_val,
        'raan': raan_val,
        'ta': ta_val,
        'ma': ma_val,
        'ea': ea_val,
        'period': period,
        'n': n,
        'periapsis': periapsis_dist,
        'apoapsis': apoapsis_dist,
    }
    
    return result
