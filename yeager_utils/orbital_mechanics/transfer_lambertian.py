import numpy as np
from ..constants import EARTH_MU, EARTH_RADIUS
from ..integrators import leapfrog
from ..time import Time
from ssapy import Orbit


def state_vector_to_conic(r1, r2, v1_t, mu=EARTH_MU):
    """Convert transfer orbit to conic coefficients (ellipse or hyperbola) in 2D.

    Args:
        r1: Initial position vector [x1, y1] (m), 2D
        r2: Final position vector [x2, y2] (m), 2D
        v1_t: Transfer velocity at r1 [vx, vy] (m/s), 2D
        mu: Gravitational parameter (m^3/s^2), default Earth

    Returns:
        List of [A, B, C, D, E, F] coefficients for the conic equation
    """
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)

    h = r1[0] * v1_t[1] - r1[1] * v1_t[0]  # Angular momentum (z-component)
    p = h**2 / mu  # Semi-latus rectum

    v1_t_mag = np.linalg.norm(v1_t)
    epsilon = v1_t_mag**2 / 2 - mu / r1_mag  # Specific energy

    a = -mu / (2 * epsilon)  # Semi-major axis (negative for hyperbola)
    e = np.sqrt(1 - p / a) if epsilon < 0 else np.sqrt(1 + p / abs(a))  # Eccentricity

    cos_theta1 = r1[0] / r1_mag
    sin_theta1 = r1[1] / r1_mag
    center_x = -a * e * cos_theta1
    center_y = -a * e * sin_theta1

    if epsilon < 0:  # Ellipse
        b = a * np.sqrt(1 - e**2)
    else:  # Hyperbola
        b = abs(a) * np.sqrt(e**2 - 1)

    A = b**2
    B = 0  # No tilt (2D assumption)
    C = a**2
    D = -2 * A * center_x
    E = -2 * C * center_y
    F = A * center_x**2 + C * center_y**2 - a**2 * b**2

    F_adjusted = -(A * 0**2 + B * 0 * 0 + C * 0**2 + D * 0 + E * 0)
    coeffs = [A, B, C, D, E, F_adjusted]

    norm_factor = abs(F_adjusted) if F_adjusted != 0 else 1
    return [c / norm_factor for c in coeffs]


def find_intersection_time(r_start, v_start, r_ref, t_max):
    """Find time when orbit from (r_start, v_start) nearly intersects r_ref."""
    t_array = np.arange(0, t_max, 1)
    r_orbit, _ = leapfrog(r_start, v_start, t=t_array)
    distances = np.linalg.norm(r_orbit - r_ref, axis=1)
    idx = np.argmin(distances)
    return t_array[idx]  # Time in seconds


def transfer_lambertian(*args, r1=None, v1=None, r2=None, v2=None, elements1=None, elements2=None, orbit1=None, orbit2=None, MIN_PERIGEE=EARTH_RADIUS + 100000, mu=EARTH_MU, plot=False):
    """Find transfer conic (ellipse or hyperbola) connecting r1 and r2 with perigee >= 6478 km.

    Can be called with positional arguments:
    - transfer_lambertian(orbit1, orbit2)  # Two Orbit objects
    - transfer_lambertian(r1, v1, r2, v2)  # Four state vectors
    - transfer_lambertian(elements1, elements2)  # Two sets of Keplerian elements

    Args:
        *args : tuple
            Positional arguments: (orbit1, orbit2), (r1, v1, r2, v2), or (elements1, elements2).
        r1, v1 : array_like, optional
            Initial position and velocity vectors (m, m/s), 2D or 3D.
        r2, v2 : array_like, optional
            Final position and velocity vectors (m, m/s), 2D or 3D.
        elements1, elements2 : tuple/list, optional
            Keplerian elements (a, e, i, ap, raan, trueAnomaly).
        orbit1, orbit2 : ssapy.Orbit, optional
            Initial and target orbit objects.
        MIN_PERIGEE : float, optional
            Minimum perigee altitude (m), default 6478 km.
        mu : float, optional
            Gravitational parameter (m^3/s^2), default Earth.
        plot : bool, optional
            If True, generate a plot (default: False).

    Returns:
        dict: Result containing:
            - 'initial': Orbit object at r1, v1
            - 'final': Orbit object at r2, v2
            - 'transfer': Orbit object for transfer orbit
            - '|delta_v1|': Magnitude of departure delta-V (m/s)
            - '|delta_v2|': Magnitude of arrival delta-V (m/s)
            - 'delta_v1': Departure delta-V vector (m/s)
            - 'delta_v2': Arrival delta-V vector (m/s)
            - 'tof': Time of flight from r1 to r2 (s)
            - 't_to_transfer': Time to transfer start (s, 0 here)
            - 'orbit_type': 'ellipse' or 'hyperbola'

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    # Handle positional arguments
    if args:
        if len(args) == 2:
            arg1, arg2 = args
            if isinstance(arg1, Orbit) and isinstance(arg2, Orbit):
                orbit1, orbit2 = arg1, arg2
            elif isinstance(arg1, (list, tuple)) and isinstance(arg2, (list, tuple)):
                elements1, elements2 = arg1, arg2
            else:
                raise ValueError("Two positional arguments must be either (orbit1, orbit2) or (elements1, elements2)")
        elif len(args) == 4:
            r1, v1, r2, v2 = args
        else:
            raise ValueError("Positional arguments must be 2 (orbits or elements) or 4 (state vectors)")

    # Default time
    t0 = Time("2025-1-1")

    # Extract state vectors from orbit objects if provided
    if orbit1 is not None:
        if not isinstance(orbit1, Orbit):
            raise ValueError("orbit1 must be an ssapy.Orbit object")
        r1 = orbit1.r
        v1 = orbit1.v
        t0 = orbit1.t
    if orbit2 is not None:
        if not isinstance(orbit2, Orbit):
            raise ValueError("orbit2 must be an ssapy.Orbit object")
        r2 = orbit2.r
        v2 = orbit2.v

    # Determine input mode and create Orbit objects
    if r1 is not None and v1 is not None and r2 is not None and v2 is not None:
        orbit1 = Orbit(r=r1, v=v1, t=t0, mu=mu)
        orbit2 = Orbit(r=r2, v=v2, t=t0, mu=mu)
    elif elements1 is not None and elements2 is not None:
        if isinstance(elements1, (list, tuple)):
            if len(elements1) != 6:
                raise ValueError("elements1 must contain exactly 6 Keplerian elements.")
            a1, e1, i1, ap1, raan1, trueAnomaly1 = elements1
            orbit1 = Orbit.fromKeplerianElements(a1, e1, i1, ap1, raan1, trueAnomaly1, t=t0, mu=mu)
        else:
            raise TypeError("elements1 must be a list/tuple of 6 elements.")

        if isinstance(elements2, (list, tuple)):
            if len(elements2) != 6:
                raise ValueError("elements2 must contain exactly 6 Keplerian elements.")
            a2, e2, i2, ap2, raan2, trueAnomaly2 = elements2
            orbit2 = Orbit.fromKeplerianElements(a2, e2, i2, ap2, raan2, trueAnomaly2, t=t0, mu=mu)
        else:
            raise TypeError("elements2 must be a list/tuple of 6 elements.")
    else:
        raise ValueError("Must provide (orbit1, orbit2), (r1, v1, r2, v2), or (elements1, elements2) via args or kwargs")

    r1 = orbit1.r
    v1 = orbit1.v
    r2 = orbit2.r
    v2 = orbit2.v

    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)

    if r1_mag < MIN_PERIGEE or r2_mag < MIN_PERIGEE:
        raise ValueError(f"Perigee too low: r1 = {r1_mag/1000:.0f} km, r2 = {r2_mag/1000:.0f} km")

    cos_dnu = np.dot(r1, r2) / (r1_mag * r2_mag)
    dnu = np.arccos(np.clip(cos_dnu, -1.0, 1.0))
    cross_r = np.cross(r1, r2)
    if len(r1) == 3 and cross_r[2] < 0:
        dnu = 2 * np.pi - dnu

    s = r1_mag + r2_mag + np.sqrt(np.sum((r1 - r2)**2))
    a_min = s / 4
    tof_min = np.pi * np.sqrt(a_min**3 / mu)

    def lambert_velocity(tof):
        a = (mu * tof**2 / (dnu**2))**(1 / 3)
        p = r1_mag * r2_mag * np.sin(dnu)**2
        p /= (r1_mag + r2_mag - 2 * np.sqrt(r1_mag * r2_mag) * np.cos(dnu))
        e = np.sqrt(1 - p / a) if p < a else np.sqrt(1 + p / a)
        f = 1 - r2_mag * (1 - np.cos(dnu)) / p
        g = r1_mag * r2_mag * np.sin(dnu) / np.sqrt(mu * p)
        v1_t = (r2 - f * r1) / g
        v2_t = (-r1 + f * r2) / g
        return v1_t, v2_t

    tof = tof_min
    v1_t, v2_t = lambert_velocity(tof)
    h = r1[0] * v1_t[1] - r1[1] * v1_t[0] if len(r1) == 2 else np.linalg.norm(np.cross(r1, v1_t))
    p = h**2 / mu
    v1_t_mag = np.linalg.norm(v1_t)
    epsilon = v1_t_mag**2 / 2 - mu / r1_mag
    a = -mu / (2 * epsilon)
    e = np.sqrt(1 - p / a) if epsilon < 0 else np.sqrt(1 + p / abs(a))
    r_p = a * (1 - e) if epsilon < 0 else abs(a) * (e - 1)

    if r_p < MIN_PERIGEE:
        print(f"Initial perigee {r_p/1000:.0f} km < {MIN_PERIGEE/1000:.0f} km, adjusting TOF...")
        tof_values = [tof_min * (1 + 0.1 * i) for i in range(20)] + [tof_min * i for i in [0.5, 2, 5, 10, 20]]
        for tof in sorted(tof_values):
            v1_t, v2_t = lambert_velocity(tof)
            h = np.linalg.norm(np.cross(r1, v1_t)) if len(r1) == 3 else r1[0] * v1_t[1] - r1[1] * v1_t[0]
            p = h**2 / mu
            v1_t_mag = np.linalg.norm(v1_t)
            epsilon = v1_t_mag**2 / 2 - mu / r1_mag
            a = -mu / (2 * epsilon)
            e = np.sqrt(1 - p / a) if epsilon < 0 else np.sqrt(1 + p / abs(a))
            r_p = a * (1 - e) if epsilon < 0 else abs(a) * (e - 1)
            if r_p >= MIN_PERIGEE:
                print(f"Adjusted to perigee {r_p/1000:.0f} km with TOF {tof/60:.0f} min")
                break
        else:
            raise ValueError(f"No transfer orbit found with perigee >= {MIN_PERIGEE/1000:.0f} km")

    # Use adjusted TOF for final calculation
    v1_t, v2_t = lambert_velocity(tof)
    orbit_type = 'ellipse' if epsilon < 0 else 'hyperbola'
    t_max = 2 * np.pi * np.sqrt(abs(a)**3 / mu) if epsilon < 0 else 2 * tof
    tof = find_intersection_time(r1, v1_t, r2, t_max)

    delta_v1 = v1_t - v1
    delta_v2 = v2 - v2_t
    result = {
        'initial': Orbit(r=r1, v=v1, t=t0),
        'final': Orbit(r=r2, v=v2, t=t0 + tof),
        'transfer': Orbit(r=r1, v=v1_t, t=t0),
        '|delta_v1|': np.linalg.norm(delta_v1),
        '|delta_v2|': np.linalg.norm(delta_v2),
        'delta_v1': delta_v1,
        'delta_v2': delta_v2,
        'tof': tof,
        't_to_transfer': 0,
        'orbit_type': orbit_type
    }

    if plot:
        from ..plots import transfer_plot
        result['fig'] = transfer_plot(r1, v1, r1, v1_t, r2, v2, show=True, c='black', title=f"Transfer time: {result['tof'] / 60:.0f} min ({result['orbit_type']})")

    return result
