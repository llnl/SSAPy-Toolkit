from ..constants import EARTH_MU, EARTH_RADIUS
from ..ssapy_wrapper import ssapy_orbit
from ..Time_Functions import Time, to_gps
import numpy as np
from poliastro.iod import izzo
import astropy.units as u
from ssapy import Orbit
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def transfer_lambertian_poliastro(elements1, elements2, t0=Time("2025-1-1", scale='utc'), mu=EARTH_MU, plot=False):
    """
    Computes a general Lambertian transfer between two orbits with the shortest time of flight.

    Parameters
    ----------
    elements1 : tuple or list
        Six Keplerian elements of the initial orbit: (a, e, i, Omega, omega, nu) in (m, -, rad, rad, rad, rad).
    elements2 : tuple or list
        Six Keplerian elements of the final orbit: (a, e, i, Omega, omega, nu) in (m, -, rad, rad, rad, rad).
    t0 : float or Time, optional
        Initial time (e.g., GPS seconds or Time object).
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
            - 'tof': Shortest time of flight (s).
            - 't_to_transfer': Time to wait until transfer (s, 0 for immediate departure).
            - 'fig': Matplotlib figure object (if plot=True).

    Author: Travis Yeager (yeager7@llnl.gov)
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

    r1 = orbit1.r
    v1_orbit1 = orbit1.v
    r2_initial = orbit2.r
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2_initial)
    c = np.linalg.norm(r2_initial - r1)
    s = (r1_norm + r2_norm + c) / 2.0
    a_min = s / 2.0
    tof = np.pi * np.sqrt(a_min**3 / mu)

    t_arrive = t0 + tof

    orbit2_at_arrive = orbit2.at(t_arrive)
    r2 = orbit2_at_arrive.r
    v2_orbit2 = orbit2_at_arrive.v

    v1_trans, v2_trans = izzo.lambert(
        mu * u.m**3 / u.s**2,
        r1 * u.m,
        r2 * u.m,
        tof * u.s
    )
    v1_trans = v1_trans.to_value(u.m / u.s)
    v2_trans = v2_trans.to_value(u.m / u.s)

    delta_v1 = np.linalg.norm(v1_trans - v1_orbit1)
    delta_v2 = np.linalg.norm(v2_orbit2 - v2_trans)

    transfer_orbit = Orbit(r=r1, v=v1_trans, t=t0, mu=mu)

    result = {
        'initial': orbit1,
        'final': orbit2,
        'transfer': transfer_orbit,
        'delta_v1': delta_v1,
        'delta_v2': delta_v2,
        'v_dep': v1_trans,
        'v_arr': v2_trans,
        'tof': tof,
        't_to_transfer': 0
    }

    if plot:
        r_traj1, _, times1 = ssapy_orbit(orbit=orbit1, duration=(orbit1.period, 's'), t0=Time(orbit1.t, format='gps'))
        r_traj2, _, times2 = ssapy_orbit(orbit=orbit2, duration=(orbit2.period, 's'), t0=Time(orbit2.t, format='gps'))
        r_traj_transfer, _, times_transfer = ssapy_orbit(
            r=transfer_orbit.r,
            v=transfer_orbit.v,
            duration=(tof, 's'),
            t0=Time(transfer_orbit.t, format='gps')
        )

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(r_traj1[:, 0], r_traj1[:, 1], label="Initial Orbit", linestyle="dashed", c='LightBlue')
        ax.plot(r_traj2[:, 0], r_traj2[:, 1], label="Final Orbit", linestyle="dotted", c='Orange')
        if np.isclose(r1_norm, r2_norm, rtol=1e-6):
            ax.scatter(r1[0], r1[1], s=6, color='Green', label="Transfer Point")
        else:
            ax.plot(r_traj_transfer[:, 0], r_traj_transfer[:, 1], label="Transfer Orbit", c="Green")
        ax.add_patch(Circle((0, 0), radius=EARTH_RADIUS, color='Blue', alpha=0.5, label="Earth"))
        ax.scatter(r1[0], r1[1], color='LightBlue', marker='o', label="Departure Point")
        ax.scatter(r2[0], r2[1], color='Orange', marker='o', label="Arrival Point")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title(f"Lambertian Transfer (TOF = {tof / 60:.0f} min\nΔv₁ = {delta_v1 / 1000:.3f} km/s, Δv₂ = {delta_v2 / 1000:.3f} km/s)")
        ax.legend(loc='upper left')
        plt.axis('equal')
        plt.show()

        result['fig'] = fig

    return result
