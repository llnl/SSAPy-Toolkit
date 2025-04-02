import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import warnings
from ssapy import Orbit
from ..constants import EARTH_MU, EARTH_RADIUS
from ..time import Time, to_gps
from ..ssapy_wrapper import ssapy_orbit


def velocity_to_ntw(r, v, v_target):
    """Convert a velocity vector to NTW coordinates using r and v to define the frame."""
    t_hat = v / np.linalg.norm(v)
    w_hat = np.cross(r, v) / np.linalg.norm(np.cross(r, v))
    n_hat = np.cross(v, np.cross(r, v)) / np.linalg.norm(np.cross(v, np.cross(r, v)))
    n = np.dot(n_hat, v_target)
    t = np.dot(t_hat, v_target)
    w = np.dot(w_hat, v_target)
    return np.array([n, t, w])


def transfer_hohmann(*args, r1=None, v1=None, r2=None, v2=None, elements1=None, elements2=None, orbit1=None, orbit2=None, t0=Time("2025-01-01"), mu=EARTH_MU, plot=False):
    """
    Compute a Hohmann transfer between two orbits and return orbital parameters and delta-V.

    Can be called with positional arguments:
    - transfer_hohmann(orbit1, orbit2)  # Two Orbit objects
    - transfer_hohmann(r1, v1, r2, v2)  # Four state vectors
    - transfer_hohmann(elements1, elements2)  # Two sets of Keplerian elements

    Or with keyword arguments as before.

    Parameters
    ----------
    *args : tuple
        Positional arguments: either (orbit1, orbit2), (r1, v1, r2, v2), or (elements1, elements2).
    r1, v1 : array_like, optional
        Initial position and velocity vectors (m, m/s).
    r2 : array_like, optional
        Target position vector (m).
    v2 : array_like, optional
        Target velocity vector (m/s). If not provided and r2 is given, assumes a circular orbit at r2.
    orbit1, orbit2 : ssapy.Orbit, optional
        Initial and target orbit objects.
    elements1, elements2 : tuple/list or Orbit, optional
        Keplerian elements (a, e, i, ap, raan, trueAnomaly) or Orbit objects.
    t0 : Time or float, optional
        Initial time (default: "2025-01-01").
    mu : float, optional
        Gravitational parameter (default: EARTH_MU).
    plot : bool, optional
        If True, generate a plot (default: False).

    Returns
    -------
    dict
        Dictionary with transfer details (initial, final, transfer orbits, delta-Vs, etc.).

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
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
            raise ValueError("Positional arguments must be 2 (orbits or elements) or 4 (state vectors)")

    t0 = to_gps(t0)

    # Extract state vectors from orbit objects if provided
    if orbit1 is not None:
        if not isinstance(orbit1, Orbit):
            raise ValueError("orbit1 must be an ssapy.Orbit object")
        r1 = orbit1.r
        v1 = orbit1.v
        t0 = orbit1.t  # Use orbit1's time
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
            v_circ = np.sqrt(EARTH_MU / r2_norm)
            v2 = np.cross(r2 / r2_norm, [0, 0, 1]) * v_circ
            if np.allclose(v2, 0):
                v2 = np.cross(r2 / r2_norm, [0, 1, 0]) * v_circ
        orbit1 = Orbit(r=r1, v=v1, t=t0, mu=mu)
        orbit2 = Orbit(r=r2, v=v2, t=t0, mu=mu)
    elif elements1 is not None and elements2 is not None:
        if isinstance(elements1, (list, tuple)):
            if len(elements1) != 6:
                raise ValueError("elements1 must contain exactly 6 Keplerian elements.")
            a1, e1, i1, ap1, raan1, trueAnomaly1 = elements1
            orbit1 = Orbit.fromKeplerianElements(*[a1, e1, i1, ap1, raan1, trueAnomaly1], t=t0, mu=mu)
        elif isinstance(elements1, Orbit):
            orbit1 = elements1
            if not np.isclose(orbit1.t, t0, atol=1e-6):
                orbit1 = orbit1.at(t0)
        else:
            raise TypeError("elements1 must be an ssapy.Orbit object or a list/tuple of 6 elements.")

        if isinstance(elements2, (list, tuple)):
            if len(elements2) != 6:
                raise ValueError("elements2 must contain exactly 6 Keplerian elements.")
            a2, e2, i2, ap2, raan2, trueAnomaly2 = elements2
            orbit2 = Orbit.fromKeplerianElements(*[a2, e2, i2, ap2, raan2, trueAnomaly2], t=t0, mu=mu)
        elif isinstance(elements2, Orbit):
            orbit2 = elements2
            if not np.isclose(orbit2.t, t0, atol=1e-6):
                orbit2 = orbit2.at(t0)
        else:
            raise TypeError("elements2 must be an ssapy.Orbit object or a list/tuple of 6 elements.")
    else:
        raise ValueError("Must provide (orbit1, orbit2), (r1, v1, r2, v2), or (elements1, elements2) via args or kwargs")

    if not np.isclose(orbit2.i, orbit1.i, atol=1e-6):
        warnings.warn(
            f"Inclination of orbit2 ({orbit2.i:.2f} rad) differs from orbit1 ({orbit1.i:.2f} rad). "
            "Computing transfer in the plane of orbit1."
        )

    if orbit1.a < orbit2.a:
        ap2_adjusted = orbit1.pa + np.pi
        trueAnomaly2_adjusted = 0.0
    else:
        ap2_adjusted = orbit1.pa
        trueAnomaly2_adjusted = np.pi

    orbit2 = Orbit.fromKeplerianElements(
        *[orbit2.a,
        orbit2.e,
        orbit1.i,
        ap2_adjusted,
        orbit2.raan,
        trueAnomaly2_adjusted],
        t=t0,
        mu=mu
    )

    if orbit1.a > orbit2.a:
        r1, r2 = orbit1.periapsis, orbit2.apoapsis
    else:
        r1, r2 = orbit1.apoapsis, orbit2.periapsis
    r1_mag, r2_mag = np.linalg.norm(r1), np.linalg.norm(r2)

    if np.isclose(r1_mag, r2_mag, rtol=1e-6):
        v1_initial = np.sqrt(mu * (2.0 / r1_mag - 1.0 / orbit1.a))
        v2_final = np.sqrt(mu * (2.0 / r2_mag - 1.0 / orbit2.a))
        delta_v1 = abs(v1_initial - v2_final)
        delta_v2 = 0.0
        tof = 0.0
        t_to_transfer = 0.0
        transfer_orbit = Orbit(r=r1, v=orbit2.v, t=orbit1.t, mu=mu)
    else:
        a_transfer = (r1_mag + r2_mag) / 2.0
        v1_initial = np.sqrt(mu * (2.0 / r1_mag - 1.0 / orbit1.a))
        v2_final = np.sqrt(mu * (2.0 / r2_mag - 1.0 / orbit2.a))
        v1_trans = np.sqrt(mu * (2.0 / r1_mag - 1.0 / a_transfer))
        v2_trans = np.sqrt(mu * (2.0 / r2_mag - 1.0 / a_transfer))
        delta_v1 = v1_trans - v1_initial
        delta_v2 = v2_final - v2_trans
        v1_direction = np.cross([0, 0, 1], r1) / np.linalg.norm(np.cross([0, 0, 1], r1))
        v1_trans_vector = v1_trans * v1_direction
        tof = np.pi * np.sqrt(a_transfer**3 / mu)
        M1 = orbit1.meanAnomaly
        T1 = orbit1.period
        if orbit1.a > orbit2.a:
            t_to_transfer = (M1 / (2 * np.pi)) * T1 if M1 >= 0 else ((2 * np.pi + M1) / (2 * np.pi)) * T1
        else:
            t_to_transfer = ((M1 - np.pi) / (2 * np.pi)) * T1 if M1 <= np.pi else ((2 * np.pi + M1 - np.pi) / (2 * np.pi)) * T1
        t_to_transfer = max(t_to_transfer, 0)
        transfer_orbit = Orbit(r=r1, v=v1_trans_vector, t=orbit1.t + t_to_transfer, mu=mu)

    # Compute velocity vectors and delta-V in inertial frame
    t_start = orbit1.t + t_to_transfer
    t_end = t_start + tof
    v1_initial_vector = v1_initial * v1_direction  # Assuming initial velocity direction aligns with transfer
    v2_trans_vector = v2_trans * np.cross([0, 0, 1], r2) / np.linalg.norm(np.cross([0, 0, 1], r2))
    delta_v1_vector = transfer_orbit.at(t_start).v - v1_initial_vector
    delta_v2_vector = v2_final * v1_direction - transfer_orbit.at(t_end).v  # Adjusted direction assumption

    # Compute NTW delta-V
    delta_ntw1 = velocity_to_ntw(r1, v1_initial_vector, delta_v1_vector)
    delta_ntw2 = velocity_to_ntw(r2, v2_trans_vector, delta_v2_vector)

    # Prepare result dictionary
    result = {
        'initial': orbit1,
        'final': orbit2,
        'transfer': transfer_orbit,
        '|delta_v1|': delta_v1,
        '|delta_v2|': delta_v2,
        'delta_v1': delta_v1_vector,
        'delta_v2': delta_v2_vector,
        'delta_ntw1': delta_ntw1,
        'delta_ntw2': delta_ntw2,
        'tof': tof,
        't_to_transfer': t_to_transfer
    }

    if plot:
        r_traj1, _, times1 = ssapy_orbit(orbit=orbit1, duration=(orbit1.period, 's'), t0=Time(orbit1.t, format='gps'))
        r_traj2, _, times2 = ssapy_orbit(orbit=orbit2, duration=(orbit2.period, 's'), t0=Time(orbit2.t, format='gps'))
        if np.isclose(r1_mag, r2_mag, rtol=1e-6):
            r_traj_transfer = np.array([r1, r1])
        else:
            r_traj_transfer, _, times_transfer = ssapy_orbit(
                r=transfer_orbit.r,
                v=transfer_orbit.v,
                duration=(transfer_orbit.period / 2, 's'),
                t0=Time(transfer_orbit.t, format='gps')
            )

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(r_traj1[:, 0], r_traj1[:, 1], label="Initial Orbit", linestyle="dashed", c='LightBlue')
        ax.plot(r_traj2[:, 0], r_traj2[:, 1], label="Final Orbit", linestyle="dotted", c="Orange")
        if np.isclose(r1_mag, r2_mag, rtol=1e-6):
            ax.scatter(r1[0], r1[1], s=6, color='Green', label="Transfer Point")
        else:
            ax.plot(r_traj_transfer[:, 0], r_traj_transfer[:, 1], label="Transfer Orbit", c="Green")
        ax.add_patch(Circle((0, 0), radius=EARTH_RADIUS, color='Blue', alpha=0.5, label="Earth"))
        ax.scatter(r1[0], r1[1], color='LightBlue', marker='o', label="Departure Point")
        ax.scatter(r2[0], r2[1], color='Orange', marker='o', label="Arrival Point")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title(f"Hohmann Transfer\nTOF {tof / 60:.0f} min\n|Δv₁| {delta_v1 / 1000:.3f} km/s, |Δv₂| {delta_v2 / 1000:.3f} km/s")
        ax.legend(loc='upper left')
        plt.axis('equal')
        plt.show()

        result['fig'] = fig

    return result
