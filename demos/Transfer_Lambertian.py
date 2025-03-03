#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script computes a transfer orbit between two state vectors.
If the position vectors are collinear (as in a Hohmann transfer between circular orbits),
it computes a Hohmann transfer. Otherwise, it uses the Lambert solver.
The orbits are integrated and plotted in different colors.

Author: Travis Yeager (yeager7@llnl.gov)
"""

import numpy as np
import matplotlib.pyplot as plt
from ssapy import get_body, rv, Orbit, SciPyPropagator, AccelKepler, AccelThirdBody, AccelHarmonic
from yeager_utils import compute_transfer, get_times, RGEO

from astropy.time import Time
from astropy import units as u

from poliastro.bodies import Earth


def build_orbit_from_input(orbit_data, t0):
    """
    Builds an orbit from either six Keplerian elements or a state vector (r, v).

    Parameters
    ----------
    orbit_data : list, tuple, or dict
        - If list/tuple of length 6, they are interpreted as
          [a, e, i, pa, raan, ta] (semi-major axis, eccentricity, inclination,
          argument of periapsis, RAAN, true anomaly).
        - If list/tuple of length 2, they are interpreted as (r, v) state vectors.
        - If dict, then either keys "r" and "v" or keys "a", "e", "i", "pa", "raan", "ta"
          must be provided.
    t0 : Time
        The epoch for the orbit.

    Returns
    -------
    Orbit
        The resulting orbit.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if isinstance(orbit_data, (list, tuple)):
        if len(orbit_data) == 6:
            # Assume Keplerian elements.
            return Orbit.fromKeplerianElements(*orbit_data, t=t0)
        elif len(orbit_data) == 2:
            # Assume state vectors.
            r, v = orbit_data
            return Orbit(r=r, v=v, t=t0)
        else:
            raise ValueError("List/tuple must have length 6 (Keplerian elements) or 2 (state vectors).")
    elif isinstance(orbit_data, dict):
        if "r" in orbit_data and "v" in orbit_data:
            return Orbit(r=orbit_data["r"], v=orbit_data["v"], t=t0)
        elif all(key in orbit_data for key in ["a", "e", "i", "pa", "raan", "ta"]):
            return Orbit.fromKeplerianElements(
                orbit_data["a"], orbit_data["e"], orbit_data["i"],
                orbit_data["pa"], orbit_data["raan"], orbit_data["ta"], t=t0)
        else:
            raise ValueError("Dictionary keys do not match expected orbit data.")
    else:
        raise ValueError("Orbit data must be a list, tuple, or dictionary.")


def build_target_orbit(a_target, e_target, t0):
    """
    Builds the target (outer) orbit at periapsis using Keplerian elements.

    Parameters
    ----------
    a_target : float
        Semi-major axis of the target orbit (m).
    e_target : float
        Eccentricity of the target orbit.
    t0 : Time
        Initial epoch.

    Returns
    -------
    Orbit
        The target orbit with true anomaly set to 0 (periapsis).

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    # If using Keplerian elements, TA is set to 0.
    return Orbit.fromKeplerianElements(a_target, e_target, 0, 0, 0, 0, t=t0)


def adjust_inner_orbit(target_orbit, R_inner, t0):
    """
    Adjusts the inner (departure) orbit so that its departure state is diametrically opposite
    to the target orbit's periapsis.

    Parameters
    ----------
    target_orbit : Orbit
        The target orbit (assumed to be at periapsis).
    R_inner : float
        Radius of the inner (circular) orbit (m).
    t0 : Time
        Initial epoch.

    Returns
    -------
    Orbit
        The adjusted inner orbit with its state vector chosen for proper departure phase.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    r_target_peri = target_orbit.r
    u_target = r_target_peri / np.linalg.norm(r_target_peri)
    r_inner = R_inner * (-u_target)
    mu = Earth.k.to(u.m**3 / u.s**2).value
    v_circ = np.sqrt(mu / R_inner)
    # Assume the orbits lie in the xy-plane; a prograde tangent is (-y, x, 0).
    x, y, z = r_inner
    if np.allclose([x, y], [0, 0]):
        tangent = np.array([0, 1, 0])
    else:
        tangent = np.array([-y, x, 0])
        tangent /= np.linalg.norm(tangent)
    v_inner = v_circ * tangent
    return Orbit(r=r_inner, v=v_inner, t=t0)


def integrate_orbit(orbit, duration, t0):
    """
    Propagates an orbit over a specified duration and returns position vectors.

    Parameters
    ----------
    orbit : poliastro.twobody.orbit.Orbit
        The orbit to be propagated.
    duration : float
        Total duration over which to propagate in seconds.

    Returns
    -------
    np.ndarray
        Array of shape (num_points, 3) containing the position vectors in meters.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    times = get_times(duration=(duration, 's'), freq=(1, "min"), t0=t0)
    earth = get_body("earth", model="egm2008")
    moon = get_body("moon")
    sun = get_body("sun")
    aEarth = AccelKepler() + AccelHarmonic(earth, 180, 180)
    aMoon = AccelThirdBody(moon) + AccelHarmonic(moon)
    aSun = AccelThirdBody(sun)
    accel = aEarth + aMoon + aSun
    propagator = SciPyPropagator(accel)
    r, v = rv(orbit, time=times, propagator=propagator)
    return r, v, times


def plot_orbits(inner_positions, transfer_positions, target_positions, method):
    """
    Plots the inner, transfer, and target orbit positions in the xy, xz, and yz planes.

    Parameters
    ----------
    inner_positions : np.ndarray
        Propagated positions from the inner orbit.
    transfer_positions : np.ndarray
        Propagated positions from the transfer orbit.
    target_positions : np.ndarray
        Propagated positions from the target orbit.
    method : str
        Transfer method used (e.g., "Hohmann" or "Lambert").
    RGEO : float
        Radius of GEO (m), used to normalize positions.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    # Normalize positions by RGEO to convert to GEO units
    inner_positions_geo = inner_positions / RGEO
    transfer_positions_geo = transfer_positions / RGEO
    target_positions_geo = target_positions / RGEO

    # Create a 1x3 subplot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    planes = ['xy', 'xz', 'yz']
    labels = [("x [GEO]", "y [GEO]"), ("x [GEO]", "z [GEO]"), ("y [GEO]", "z [GEO]")]

    # Plot each plane
    for idx, (ax, plane, label) in enumerate(zip(axes, planes, labels)):
        if plane == 'xy':
            ax.plot(inner_positions_geo[:, 0], inner_positions_geo[:, 1],
                    label="Inner Orbit (Adjusted Departure)", color="blue", linewidth=2)
            ax.plot(transfer_positions_geo[:, 0], transfer_positions_geo[:, 1],
                    label=f"Transfer Orbit ({method})", color="red", linestyle="--", linewidth=2)
            ax.plot(target_positions_geo[:, 0], target_positions_geo[:, 1],
                    label="Target Orbit (Periapsis)", color="green", linewidth=2)
        elif plane == 'xz':
            ax.plot(inner_positions_geo[:, 0], inner_positions_geo[:, 2],
                    label="Inner Orbit (Adjusted Departure)", color="blue", linewidth=2)
            ax.plot(transfer_positions_geo[:, 0], transfer_positions_geo[:, 2],
                    label=f"Transfer Orbit ({method})", color="red", linestyle="--", linewidth=2)
            ax.plot(target_positions_geo[:, 0], target_positions_geo[:, 2],
                    label="Target Orbit (Periapsis)", color="green", linewidth=2)
        elif plane == 'yz':
            ax.plot(inner_positions_geo[:, 1], inner_positions_geo[:, 2],
                    label="Inner Orbit (Adjusted Departure)", color="blue", linewidth=2)
            ax.plot(transfer_positions_geo[:, 1], transfer_positions_geo[:, 2],
                    label=f"Transfer Orbit ({method})", color="red", linestyle="--", linewidth=2)
            ax.plot(target_positions_geo[:, 1], target_positions_geo[:, 2],
                    label="Target Orbit (Periapsis)", color="green", linewidth=2)

        # Set axis labels and grid
        ax.set_xlabel(label[0])
        ax.set_ylabel(label[1])
        ax.grid(True)
        ax.axis("equal")

        # Add legend to the first subplot only
        if idx == 0:
            ax.legend(loc="upper right")

    # Set the overall title for the figure
    fig.suptitle(f"Orbit Visualization in {method} Transfer", fontsize=16)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function that sets up the orbits, computes the transfer, and plots the results.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    # Common parameters.
    R_inner = RGEO  # (m) for the inner (departure) orbit
    t0 = Time("2025-1-1T12:00:00.000", scale="utc")

    initial_elements = [R_inner, 0.0, 90, 0, 0, 0]
    initial_elements = initial_elements[:2] + list(np.radians(initial_elements[2:]))
    initial_orbit = build_orbit_from_input(initial_elements, t0)
    target_elements = [2 * R_inner, 0.3, 0, 60, 0, np.pi / 4]
    target_elements = target_elements[:2] + list(np.radians(target_elements[2:]))
    target_orbit = build_orbit_from_input(target_elements, t0)

    r1 = initial_orbit.r
    v1 = initial_orbit.v
    r2 = target_orbit.r
    v2 = target_orbit.v

    delta_v1, delta_v2, v1_trans, v2_trans, tof_used, method = compute_transfer(r1, v1, r2, v2)
    print("Transfer Method:", method)
    print("Initial Delta-V: {:.2f} m/s".format(delta_v1))
    print("Final Delta-V: {:.2f} m/s".format(delta_v2))
    print("Time-of-Flight used: {:.2f} s".format(tof_used))

    print(r1, v1_trans)
    transfer_orbit = Orbit(r=r1, v=v1_trans, t=t0)

    inner_positions, v, inner_times = integrate_orbit(initial_orbit, duration=initial_orbit.period, t0=t0)
    transfer_positions, v, transfer_times = integrate_orbit(transfer_orbit, duration=tof_used, t0=t0)
    target_positions, v, target_times = integrate_orbit(target_orbit, duration=target_orbit.period, t0=t0)

    plot_orbits(inner_positions, transfer_positions, target_positions, method)


if __name__ == "__main__":
    main()
