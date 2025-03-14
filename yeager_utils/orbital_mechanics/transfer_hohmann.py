import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import warnings
from ssapy import Orbit
from ..constants import EARTH_MU, EARTH_RADIUS
from ..time import Time, to_gps
from ..ssapy_wrapper import ssapy_orbit


def hohmann_transfer(elements1, elements2, t0=Time("2025-01-01"), mu=EARTH_MU, plot=False):
    """
    Computes the Hohmann transfer between two orbits and returns a dictionary of results.

    Parameters
    ----------
    elements1 : ssapy.Orbit or tuple/list
        Initial orbit, either as an ssapy.Orbit object or six Keplerian elements:
        (a, e, i, ap, raan, trueAnomaly) in (m, -, rad, rad, rad, rad).
    elements2 : ssapy.Orbit or tuple/list
        Final orbit, either as an ssapy.Orbit object or six Keplerian elements:
        (a, e, i, ap, raan, trueAnomaly), where only a and e are used for radius calculations,
        others are adjusted for the transfer.
    t0 : float or Time, optional
        Initial time (e.g., GPS seconds or Time object).
    mu : float, optional
        Gravitational parameter of the central body (default is Earth’s in m^3/s^2).
    plot : bool, optional
        If True, plots the initial, final, and transfer orbits (default is False).

    Returns
    -------
    dict
        Dictionary containing:
        - 'starting_orbit': Orbit object for the initial orbit.
        - 'final_orbit': Orbit object for the final orbit.
        - 'transfer_orbit': Orbit object for the transfer orbit.
        - 'delta_v1': Delta-V required at the start (m/s).
        - 'delta_v2': Delta-V required at the end (m/s).
        - 'tof': Time of flight for the transfer (s).
        - 't_to_transfer': Time to wait on orbit1 until transfer window (s).
    fig : matplotlib.figure.Figure, optional
        The figure object if plot=True (returned as second value in result dict).

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    t0 = to_gps(t0)

    # Convert elements1 to Orbit object if it’s a list/tuple
    if isinstance(elements1, (list, tuple)):
        if len(elements1) != 6:
            raise ValueError("elements1 must contain exactly 6 Keplerian elements.")
        a1, e1, i1, ap1, raan1, trueAnomaly1 = elements1
        orbit1 = Orbit.fromKeplerianElements(a1, e1, i1, ap1, raan1, trueAnomaly1, t=t0, mu=mu)
    elif isinstance(elements1, Orbit):
        orbit1 = elements1
        if not np.isclose(orbit1.t, t0, atol=1e-6):
            orbit1 = orbit1.at(t0)
    else:
        raise TypeError("elements1 must be an ssapy.Orbit object or a list/tuple of 6 elements.")

    # Convert elements2 to Orbit object if it’s a list/tuple
    if isinstance(elements2, (list, tuple)):
        if len(elements2) != 6:
            raise ValueError("elements2 must contain exactly 6 Keplerian elements.")
        a2, e2, i2, ap2, raan2, trueAnomaly2 = elements2
        orbit2 = Orbit.fromKeplerianElements(a2, e2, i2, ap2, raan2, trueAnomaly2, t=t0, mu=mu)
    elif isinstance(elements2, Orbit):
        orbit2 = elements2
        if not np.isclose(orbit2.t, t0, atol=1e-6):
            orbit2 = orbit2.at(t0)
    else:
        raise TypeError("elements2 must be an ssapy.Orbit object or a list/tuple of 6 elements.")

    # Warn if inclinations differ, but use orbit1’s plane for the transfer
    if not np.isclose(orbit2.i, orbit1.i, atol=1e-6):
        warnings.warn(
            f"Inclination of orbit2 ({orbit2.i:.2f} rad) differs from orbit1 ({orbit1.i:.2f} rad). "
            "Computing transfer in the plane of orbit1."
        )

    # Adjust orbit2 for Hohmann transfer alignment
    if orbit1.a < orbit2.a:
        ap2_adjusted = orbit1.pa + np.pi  # Outward: periapsis of orbit2 aligns with apoapsis
        trueAnomaly2_adjusted = 0.0
    else:
        ap2_adjusted = orbit1.pa  # Inward: apoapsis of orbit2 aligns with periapsis
        trueAnomaly2_adjusted = np.pi

    orbit2 = Orbit.fromKeplerianElements(
        orbit2.a,
        orbit2.e,
        orbit1.i,
        ap2_adjusted,
        orbit2.raan,
        trueAnomaly2_adjusted,
        t=t0,
        mu=mu
    )

    # Determine transfer start and end points
    if orbit1.a > orbit2.a:
        r1, r2 = orbit1.periapsis, orbit2.apoapsis  # Inward: start at periapsis, end at apoapsis
    else:
        r1, r2 = orbit1.apoapsis, orbit2.periapsis  # Outward: start at apoapsis, end at periapsis
    r1_mag, r2_mag = np.linalg.norm(r1), np.linalg.norm(r2)

    # Check for instant transfer case
    if np.isclose(r1_mag, r2_mag, rtol=1e-6):
        # Instant transfer: r1 == r2
        v1_initial = np.sqrt(mu * (2.0 / r1_mag - 1.0 / orbit1.a))  # Initial orbit velocity at r1
        v2_final = np.sqrt(mu * (2.0 / r2_mag - 1.0 / orbit2.a))          # Final orbit velocity at r2
        delta_v1 = abs(v1_initial - v2_final)
        delta_v2 = 0.0
        tof = 0.0
        t_to_transfer = 0.0  # No wait time for instant transfer
        transfer_orbit = Orbit(
            r=r1,
            v=orbit2.v,  # Match orbit2's velocity at r1
            t=orbit1.t,
            mu=mu
        )
    else:
        # Standard Hohmann transfer
        a_transfer = (r1_mag + r2_mag) / 2.0
        v1_initial = np.sqrt(mu * (2.0 / r1_mag - 1.0 / orbit1.a))
        v2_final = np.sqrt(mu * (2.0 / r2_mag - 1.0 / orbit2.a))
        v1_trans = np.sqrt(mu * (2.0 / r1_mag - 1.0 / a_transfer))
        v2_trans = np.sqrt(mu * (2.0 / r2_mag - 1.0 / a_transfer))
        delta_v1 = abs(v1_trans - v1_initial)
        delta_v2 = abs(v2_final - v2_trans)
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
        transfer_orbit = Orbit(
            r=r1,
            v=v1_trans_vector,
            t=orbit1.t + t_to_transfer,
            mu=mu
        )

    # Prepare result dictionary
    result = {
        'initial': orbit1,
        'final': orbit2,
        'transfer': transfer_orbit,
        'delta_v1': delta_v1,
        'delta_v2': delta_v2,
        'tof': tof,
        't_to_transfer': t_to_transfer
    }

    # Optional plotting
    if plot:
        r_traj1, _, times1 = ssapy_orbit(
            orbit=orbit1,
            duration=(orbit1.period, 's'),
            t0=Time(orbit1.t, format='gps')
        )
        r_traj2, _, times2 = ssapy_orbit(
            orbit=orbit2,
            duration=(orbit2.period, 's'),
            t0=Time(orbit2.t, format='gps')
        )
        if np.isclose(r1_mag, r2_mag, rtol=1e-6):
            # Instant transfer: plot a single point
            r_traj_transfer = np.array([r1, r1])  # Dummy array with start point
        else:
            # Standard transfer: plot the transfer orbit trajectory
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
        ax.set_title(f"Hohmann Transfer (TOF = {tof / 60:.0f} min\nΔv₁ = {delta_v1 / 1000:.3f} km/s, Δv₂ = {delta_v2 / 1000:.3f} km/s)")
        ax.legend(loc='upper left')
        plt.axis('equal')
        plt.show()

        result['fig'] = fig

    return result
