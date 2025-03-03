from ..constants import EARTH_MU
import numpy as np
from poliastro.iod import izzo
import astropy.units as u
from ssapy import Orbit, ssapy_orbit
import matplotlib.pyplot as plt
from astropy.time import Time


def lambertian_transfer(elements1, elements2, t0, tof, mu=EARTH_MU, plot=False):
    """
    Computes a general Lambertian transfer between two orbits given a departure time and time of flight.

    Parameters
    ----------
    elements1 : tuple or list
        Six Keplerian elements of the initial orbit: (a, e, i, Omega, omega, nu) in (m, -, rad, rad, rad, rad).
    elements2 : tuple or list
        Six Keplerian elements of the final orbit: (a, e, i, Omega, omega, nu) in (m, -, rad, rad, rad, rad).
    t0 : float or Time
        Departure time (s if float, or astropy Time object).
    tof : float
        Time of flight (s).
    mu : float, optional
        Gravitational parameter (m^3/s^2), defaults to EARTH_MU.
    plot : bool, optional
        If True, plots the initial orbit, final orbit, and transfer trajectory.

    Returns
    -------
    dict
        Dictionary containing:
            - 'transfer_orbit': Orbit object for the transfer orbit.
            - 'delta_v1': Delta-V required at departure (m/s).
            - 'delta_v2': Delta-V required at arrival (m/s).
            - 'tof': Time of flight (s).
            - 't_to_transfer': Time to wait until transfer (s, 0 for immediate departure).
            - 'fig': Matplotlib figure object (if plot=True).

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    # Convert t0 to float (seconds) if it's an astropy Time object
    if isinstance(t0, Time):
        t0 = t0.to_value('s', subfmt='float')

    # Create initial and final orbits at t0
    orbit1 = Orbit.fromKeplerianElements(*elements1, t=t0, mu=mu)
    orbit2 = Orbit.fromKeplerianElements(*elements2, t=t0, mu=mu)

    # Compute state at departure (t0) for orbit1
    r1 = orbit1.getPositionAtTime(t0)  # Position at t0
    v1_orbit1 = orbit1.getVelocityAtTime(t0)  # Velocity at t0

    # Compute state at arrival (t0 + tof) for orbit2
    t_arrive = t0 + tof
    r2 = orbit2.getPositionAtTime(t_arrive)  # Position at t0 + tof
    v2_orbit2 = orbit2.getVelocityAtTime(t_arrive)  # Velocity at t0 + tof

    # Solve Lambert's problem for transfer velocities
    v1_trans, v2_trans = izzo.lambert(
        mu * u.m**3 / u.s**2,
        r1 * u.m,
        r2 * u.m,
        tof * u.s
    )
    v1_trans = v1_trans.to_value(u.m / u.s)  # Convert to numpy array in m/s
    v2_trans = v2_trans.to_value(u.m / u.s)

    # Compute required delta-v
    delta_v1 = np.linalg.norm(v1_trans - v1_orbit1)
    delta_v2 = np.linalg.norm(v2_orbit2 - v2_trans)

    # Create transfer orbit starting at r1, v1_trans
    transfer_orbit = Orbit(r=r1, v=v1_trans, t=t0, mu=mu)

    # Prepare result dictionary
    result = {
        'initial': orbit1,
        'final': orbit2,
        'transfer': transfer_orbit,
        'delta_v1': delta_v1,
        'delta_v2': delta_v2,
        'tof': tof,
        't_to_transfer': 0  # Immediate departure at t0
    }

    # Optional plotting
    if plot:
        # Propagate orbits for one period (initial and final) and transfer duration
        r_traj1, _, times1 = ssapy_orbit(orbit=orbit1, duration=(orbit1.period, 's'), t0=t0)
        r_traj2, _, times2 = ssapy_orbit(orbit=orbit2, duration=(orbit2.period, 's'), t0=t0)
        r_traj_transfer, _, times_transfer = ssapy_orbit(
            r=transfer_orbit.r,
            v=transfer_orbit.v,
            duration=(tof, 's'),
            t0=t0
        )

        # Create 2D plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(r_traj1[:, 0], r_traj1[:, 1], label="Initial Orbit", linestyle="dashed")
        ax.plot(r_traj2[:, 0], r_traj2[:, 1], label="Final Orbit", linestyle="dotted")
        ax.plot(r_traj_transfer[:, 0], r_traj_transfer[:, 1], label="Transfer Orbit")
        ax.scatter([0], [0], color='blue', marker='o', label="Central Body")
        ax.scatter(r1[0], r1[1], color='green', marker='o', label="Departure Point")
        ax.scatter(r2[0], r2[1], color='red', marker='o', label="Arrival Point")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title(f"Lambertian Transfer (TOF = {tof:.2f} s)")
        ax.legend(loc='upper left')
        ax.set_aspect('equal')
        plt.grid(True)
        plt.show()

        result['fig'] = fig

    return result
