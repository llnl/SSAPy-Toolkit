import numpy as np
from ..constants import EARTH_MU, EARTH_RADIUS
from ..integrators import leapfrog
from ..time import Time
from ssapy import Orbit


def find_intersection_time(r_start, v_start, r_ref, t_max):
    """Find trajectory and validate TOF for orbit from (r_start, v_start) to r_ref."""
    t_array = np.arange(0, t_max, 1)
    r_orbit, v_orbit = leapfrog(r_start, v_start, t=t_array)
    distances = np.linalg.norm(r_orbit - r_ref, axis=1)
    idx = np.argmin(distances)
    tof = t_array[idx]
    r_transfer = r_orbit[:idx + 1]  # Trajectory from start to intersection
    v_transfer = v_orbit[:idx + 1]
    return tof, r_transfer, v_transfer


def transfer_lambertian(*args, r1=None, v1=None, r2=None, v2=None, elements1=None, elements2=None, orbit1=None, orbit2=None, MIN_PERIGEE=EARTH_RADIUS + 100000, mu=EARTH_MU, plot=False):
    """Find transfer conic connecting r1 and r2, ensuring transfer arc stays above MIN_PERIGEE.

    Args:
        *args : tuple
            Positional arguments: (orbit1, orbit2), (r1, v1, r2, v2), or (elements1, elements2).
        r1, v1 : array_like, optional
            Initial position and velocity vectors (m, m/s).
        r2 : array_like, optional
            Target position vector (m).
        v2 : array_like, optional
            Target velocity vector (m/s). If None, assumes circular orbit at r2.
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
        dict: Result containing transfer orbit details (see original docstring).

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
        elif len(args) == 3:
            r1, v1, r2 = args
        elif len(args) == 4:
            r1, v1, r2, v2 = args
        else:
            raise ValueError("Positional arguments must be 2 (orbits or elements), 3 (r1, v1, r2), or 4 (state vectors)")

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
    if r1 is not None and v1 is not None and r2 is not None:
        if v2 is None:
            r2 = np.asarray(r2)
            r2_norm = np.linalg.norm(r2)
            v_circ = np.sqrt(mu / r2_norm)
            v2 = np.cross(r2 / r2_norm, [0, 0, 1]) * v_circ
            if np.allclose(v2, 0):
                v2 = np.cross(r2 / r2_norm, [0, 1, 0]) * v_circ
        orbit1 = Orbit(r=r1, v=v1, t=t0, mu=mu)
        orbit2 = Orbit(r=r2, v=v2, t=t0, mu=mu)
    elif elements1 is not None and elements2 is not None:
        if isinstance(elements1, (list, tuple)) and len(elements1) == 6:
            a1, e1, i1, ap1, raan1, trueAnomaly1 = elements1
            orbit1 = Orbit.fromKeplerianElements(a1, e1, i1, ap1, raan1, trueAnomaly1, t=t0, mu=mu)
        else:
            raise ValueError("elements1 must be a list/tuple of 6 elements.")
        if isinstance(elements2, (list, tuple)) and len(elements2) == 6:
            a2, e2, i2, ap2, raan2, trueAnomaly2 = elements2
            orbit2 = Orbit.fromKeplerianElements(a2, e2, i2, ap2, raan2, trueAnomaly2, t=t0, mu=mu)
        else:
            raise ValueError("elements2 must be a list/tuple of 6 elements.")
    else:
        raise ValueError("Must provide (orbit1, orbit2), (r1, v1, r2, v2), or (elements1, elements2)")

    r1 = np.asarray(orbit1.r)
    v1 = np.asarray(orbit1.v)
    r2 = np.asarray(orbit2.r)
    v2 = np.asarray(orbit2.v)

    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)

    if r1_mag < MIN_PERIGEE or r2_mag < MIN_PERIGEE:
        raise ValueError(f"Initial or final position too low: r1 = {r1_mag/1000:.0f} km, r2 = {r2_mag/1000:.0f} km")

    cos_dnu = np.dot(r1, r2) / (r1_mag * r2_mag)
    dnu = np.arccos(np.clip(cos_dnu, -1.0, 1.0))
    cross_r = np.cross(r1, r2)
    if len(r1) == 3 and cross_r[2] < 0:
        dnu = 2 * np.pi - dnu  # Long way around for retrograde

    s = r1_mag + r2_mag + np.sqrt(np.sum((r1 - r2)**2))
    a_min = s / 4
    tof_min = np.pi * np.sqrt(a_min**3 / mu)

    def lambert_velocity(tof):
        """Compute Lambert transfer velocities and exact TOF."""
        a = (mu * tof**2 / (dnu**2))**(1 / 3)
        p = r1_mag * r2_mag * np.sin(dnu)**2 / (r1_mag + r2_mag - 2 * np.sqrt(r1_mag * r2_mag) * np.cos(dnu))
        e = np.sqrt(1 - p / a) if p < a else np.sqrt(1 + p / a)
        f = 1 - r2_mag * (1 - np.cos(dnu)) / p
        g = r1_mag * r2_mag * np.sin(dnu) / np.sqrt(mu * p)
        v1_t = (r2 - f * r1) / g
        v2_t = (-r1 + f * r2) / g
        
        # Compute true anomalies and exact TOF
        h = np.linalg.norm(np.cross(r1, v1_t)) if len(r1) == 3 else r1[0] * v1_t[1] - r1[1] * v1_t[0]
        nu1 = np.arccos(np.clip((p / r1_mag - 1) / e, -1.0, 1.0))
        nu2 = np.arccos(np.clip((p / r2_mag - 1) / e, -1.0, 1.0))
        if np.dot(r1, v1_t) < 0:
            nu1 = 2 * np.pi - nu1
        if np.dot(r2, v2_t) < 0:
            nu2 = 2 * np.pi - nu2
        if cross_r[2] < 0 and nu2 > nu1:
            nu2 -= 2 * np.pi
        elif cross_r[2] >= 0 and nu2 < nu1:
            nu2 += 2 * np.pi
        dnu_transfer = nu2 - nu1
        
        if e < 1:  # Ellipse
            E1 = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu1 / 2))
            E2 = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu2 / 2))
            if E2 < E1:
                E2 += 2 * np.pi
            tof_exact = np.sqrt(a**3 / mu) * (E2 - E1 - e * (np.sin(E2) - np.sin(E1)))
        else:  # Hyperbola
            F1 = 2 * np.arctanh(np.sqrt((e - 1) / (e + 1)) * np.tan(nu1 / 2))
            F2 = 2 * np.arctanh(np.sqrt((e - 1) / (e + 1)) * np.tan(nu2 / 2))
            tof_exact = np.sqrt((-a)**3 / mu) * (e * (np.sinh(F2) - np.sinh(F1)) - (F2 - F1))
        
        return v1_t, v2_t, a, e, tof_exact

    # Initial guess
    tof = tof_min
    v1_t, v2_t, a, e, tof = lambert_velocity(tof)
    orbit_type = 'ellipse' if e < 1 else 'hyperbola'
    t_max = tof * 1.5  # Slightly larger than exact TOF for safety
    _, r_transfer, v_transfer = find_intersection_time(r1, v1_t, r2, t_max)

    # Check minimum radius along the transfer arc
    r_mags = np.linalg.norm(r_transfer, axis=1)
    min_radius = np.min(r_mags)
    if min_radius < MIN_PERIGEE:
        print(f"Transfer arc dips to {min_radius/1000:.0f} km < {MIN_PERIGEE/1000:.0f} km, adjusting TOF...")
        tof_values = [tof_min * (1 + 0.1 * i) for i in range(20)] + [tof_min * i for i in [0.5, 2, 5, 10, 20]]
        for tof_guess in sorted(tof_values):
            v1_t, v2_t, a, e, tof = lambert_velocity(tof_guess)
            t_max = tof * 1.5
            _, r_transfer, v_transfer = find_intersection_time(r1, v1_t, r2, t_max)
            r_mags = np.linalg.norm(r_transfer, axis=1)
            min_radius = np.min(r_mags)
            if min_radius >= MIN_PERIGEE:
                print(f"Adjusted to min radius {min_radius/1000:.0f} km with TOF {tof/60:.0f} min")
                break
        else:
            raise ValueError(f"No transfer orbit found with arc radius >= {MIN_PERIGEE/1000:.0f} km")

    # Final calculation with adjusted TOF
    v1_t, v2_t, a, e, tof = lambert_velocity(tof)
    orbit_type = 'ellipse' if e < 1 else 'hyperbola'
    t_max = tof * 1.5
    _, r_transfer, v_transfer = find_intersection_time(r1, v1_t, r2, t_max)

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
        'r_transfer': r_transfer,
        'v_transfer': v_transfer,
        'tof': tof,
        't_to_transfer': 0,
        'orbit_type': orbit_type,
    }

    if plot:
        from ..plots import transfer_plot
        result['fig'] = transfer_plot(r1, v1, r1, v1_t, r2, v2, show=True, c='black', title=f"Transfer time: {result['tof'] / 60:.0f} min ({result['orbit_type']})\n|Δv₁| {np.linalg.norm(delta_v1) / 1e3:.3f}, |Δv₂| {np.linalg.norm(delta_v2) / 1e3:.3f} km/s")

    return result
