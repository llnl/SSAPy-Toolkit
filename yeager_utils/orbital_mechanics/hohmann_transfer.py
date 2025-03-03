import numpy as np
import matplotlib.pyplot as plt
import warnings
from ssapy import Orbit
from ..constants import EARTH_MU
from ..time import Time
from ..ssapy_wrapper import ssapy_orbit

def hohmann_transfer(elements1, elements2, t0, mu=EARTH_MU, plot=False):
    """
    Computes the Hohmann transfer between two orbits and returns a dictionary of results.

    Parameters
    ----------
    elements1 : tuple or list
        Six Keplerian elements of the initial orbit: (a, e, i, Omega, omega, nu) in
        meters, dimensionless, radians, radians, radians, radians.
    elements2 : tuple or list
        Six Keplerian elements of the final orbit: (a, e, i, Omega, omega, nu), where
        only a and e are used, others are adjusted for the transfer.
    t0 : float or Time
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
        The figure object if plot=True (returned as second value).

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    # Unpack Keplerian elements
    a1, e1, i1, Omega1, omega1, nu1 = elements1
    a2, e2, i2, Omega2, omega2, nu2 = elements2

    # Warn if inclinations differ, but use i1 for the transfer plane
    if not np.isclose(i2, i1, atol=1e-6):
        warnings.warn(
            f"Inclination of orbit2 ({i2:.2f} rad) differs from orbit1 ({i1:.2f} rad). "
            "Computing transfer in the plane of orbit1."
        )

    orbit1 = Orbit.fromKeplerianElements(a1, e1, i1, Omega1, omega1, nu1, t=t0, mu=mu)

    # Adjust orbit2 for proper alignment
    if a1 < a2:
        omega2_adjusted = omega1 + np.pi  # Outward: align periapsis of orbit2 with apoapsis of transfer
        nu2_adjusted = 0.0
    else:
        omega2_adjusted = omega1  # Inward: align apoapsis of orbit2 with periapsis of transfer
        nu2_adjusted = np.pi

    orbit2 = Orbit.fromKeplerianElements(
        a2,
        e2,
        i1,           # Use orbit1's inclination
        Omega1,       # Use orbit1's RAAN
        omega2_adjusted,
        nu2_adjusted,
        t=t0,
        mu=mu
    )

    # Determine transfer start and end points
    if a1 > a2:
        r1, r2 = orbit1.periapsis, orbit2.apoapsis  # Inward: start at periapsis, end at apoapsis
    else:
        r1, r2 = orbit1.apoapsis, orbit2.periapsis  # Outward: start at apoapsis, end at periapsis
    r1_mag, r2_mag = np.linalg.norm(r1), np.linalg.norm(r2)

    # Check for instant transfer case
    if np.isclose(r1_mag, r2_mag, rtol=1e-6):
        # Instant transfer: r1 == r2
        v1_initial = np.sqrt(mu * (2.0 / r1_mag - 1.0 / a1))  # Initial orbit velocity at r1
        v2_final = np.sqrt(mu * (2.0 / r2_mag - 1.0 / a2))    # Final orbit velocity at r2
        delta_v1 = abs(v1_initial - v2_final)
        delta_v2 = 0.0
        tof = 0.0
        t_to_transfer = 0.0  # No wait time for instant transfer
        transfer_orbit = Orbit(
            r=r1,
            v=orbit2.getVelocityAtRadius(r1_mag),  # Match orbit2's velocity at r1
            t=orbit1.t,
            mu=mu
        )
    else:
        # Standard Hohmann transfer
        a_transfer = (r1_mag + r2_mag) / 2.0
        v1_initial = np.sqrt(mu * (2.0 / r1_mag - 1.0 / a1))
        v2_final = np.sqrt(mu * (2.0 / r2_mag - 1.0 / a2))
        v1_trans = np.sqrt(mu * (2.0 / r1_mag - 1.0 / a_transfer))
        v2_trans = np.sqrt(mu * (2.0 / r2_mag - 1.0 / a_transfer))
        delta_v1 = abs(v1_trans - v1_initial)
        delta_v2 = abs(v2_final - v2_trans)
        v1_direction = np.cross([0, 0, 1], r1) / np.linalg.norm(np.cross([0, 0, 1], r1))
        v1_trans_vector = v1_trans * v1_direction
        tof = np.pi * np.sqrt(a_transfer**3 / mu)
        M1 = orbit1.meanAnomaly
        T1 = orbit1.period
        if a1 > a2:
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

    # Prepare result dictionary with starting and final orbits
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

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(r_traj1[:, 0], r_traj1[:, 1], label="Initial Orbit", linestyle="dashed")
        ax.plot(r_traj2[:, 0], r_traj2[:, 1], label="Final Orbit", linestyle="dotted")
        if np.isclose(r1_mag, r2_mag, rtol=1e-6):
            ax.scatter(r1[0], r1[1], color='red', label="Transfer Point")
        else:
            ax.plot(r_traj_transfer[:, 0], r_traj_transfer[:, 1], label="Transfer Orbit")
        ax.scatter([0], [0], color='blue', marker='o', label="Earth")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Hohmann Transfer")
        ax.legend(loc='upper left')
        ax.set_aspect('equal')
        plt.show()

        result['fig'] = fig

    return result
