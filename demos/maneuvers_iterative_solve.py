# flake8: noqa: E501
#!/usr/bin/env python3
"""
Transfer Orbit Propagation using SSApy with Lambert/Hohmann Handling

This script computes a transfer orbit between two orbits defined by Keplerian elements.
If the position vectors are collinear (as in a Hohmann transfer between circular orbits),
it computes a Hohmann transfer. Otherwise, it uses the Lambert solver.
The orbits are propagated using SSApy and plotted.

Author: Travis Yeager (yeager7@llnl.gov)
"""

import numpy as np
import matplotlib.pyplot as plt

# SSApy and related modules
import ssapy
from ssapy.accel import AccelKepler, AccelEarthRad, AccelSolRad, AccelDrag, AccelConstNTW
from ssapy.gravity import AccelHarmonic, AccelThirdBody
from ssapy.body import get_body
from ssapy.constants import RGEO, EARTH_MU
from ssapy.propagator import SciPyPropagator

# Yeager utilities
from yeager_utils import calc_gamma_and_heading

# For Lambert solver in non-collinear cases
from astropy import units as u
from poliastro.iod import izzo


def compute_transfer(r1, v1, r2, v2, tof):
    """
    Computes transfer orbit parameters between two state vectors.
    For collinear position vectors, computes a Hohmann transfer.
    Otherwise, uses the Lambert solver.

    Parameters
    ----------
    r1 : np.ndarray
        Initial position vector in meters.
    v1 : np.ndarray
        Initial velocity vector in m/s.
    r2 : np.ndarray
        Final position vector in meters.
    v2 : np.ndarray
        Final velocity vector in m/s.
    tof : float
        Time of flight in seconds for Lambert transfer.
        For a Hohmann transfer, the computed value is used.

    Returns
    -------
    delta_v1 : float
        Required delta-v at the initial position (m/s).
    delta_v2 : float
        Required delta-v at the final position (m/s).
    v1_trans : np.ndarray or astropy.units.quantity.Quantity
        Transfer orbit initial velocity vector (m/s).
    v2_trans : np.ndarray or astropy.units.quantity.Quantity
        Transfer orbit final velocity vector (m/s).
    tof_used : float
        Time-of-flight used in seconds.
    method : str
        'Hohmann' if a Hohmann transfer was computed, or 'Lambert' otherwise.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    # Check collinearity using the cross product
    cross_norm = np.linalg.norm(np.cross(r1, r2))
    if cross_norm < 1e-6 * (np.linalg.norm(r1) * np.linalg.norm(r2)):
        # Compute Hohmann transfer parameters
        r1_mag = np.linalg.norm(r1)
        r2_mag = np.linalg.norm(r2)
        a_transfer = (r1_mag + r2_mag) / 2.0
        # Velocities along the transfer ellipse
        v_perigee = np.sqrt(EARTH_MU * (2.0 / r1_mag - 1.0 / a_transfer))
        v_apogee = np.sqrt(EARTH_MU * (2.0 / r2_mag - 1.0 / a_transfer))
        # Circular orbit velocities
        v_circ1 = np.sqrt(EARTH_MU / r1_mag)
        v_circ2 = np.sqrt(EARTH_MU / r2_mag)
        delta_v1 = abs(v_perigee - v_circ1)
        delta_v2 = abs(v_circ2 - v_apogee)
        # Apply the maneuver tangentially along the current velocity directions
        tangential_direction1 = v1 / np.linalg.norm(v1)
        tangential_direction2 = v2 / np.linalg.norm(v2)
        v1_trans = v_perigee * tangential_direction1
        v2_trans = v_apogee * tangential_direction2
        # Hohmann time-of-flight is half the period of the transfer ellipse
        tof_hohmann = np.pi * np.sqrt(a_transfer**3 / EARTH_MU)
        if abs(tof - tof_hohmann) > 1e-3:
            print(
                f"Warning: Provided tof ({tof} s) does not match computed Hohmann tof ({tof_hohmann} s). Using computed value."
            )
        return delta_v1, delta_v2, v1_trans, v2_trans, tof_hohmann, "Hohmann"
    else:
        # Use the Lambert solver for non-collinear transfers
        r1_q = r1 * u.m
        v1_q = v1 * u.m / u.s
        r2_q = r2 * u.m
        v2_q = v2 * u.m / u.s
        tof_q = tof * u.s
        v1_trans, v2_trans = izzo.lambert(EARTH_MU, r1_q, r2_q, tof_q)
        delta_v1 = np.linalg.norm((v1_trans - v1_q).to(u.m / u.s).value)
        delta_v2 = np.linalg.norm((v2_q - v2_trans).to(u.m / u.s).value)
        return delta_v1, delta_v2, v1_trans, v2_trans, tof, "Lambert"


def integrate_orbit(orbit, duration, propagator, num_points=500):
    """
    Propagates an orbit over a specified duration using SSApy's propagator.

    Parameters
    ----------
    orbit : ssapy.Orbit
        The orbit to propagate.
    duration : float
        Propagation duration in seconds.
    propagator : ssapy.propagator.SciPyPropagator
        The propagator to use for orbit integration.
    num_points : int, optional
        Number of points to compute, by default 500.

    Returns
    -------
    np.ndarray
        Array of shape (num_points, 3) containing position vectors in meters.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    t_array = np.linspace(0, duration, num_points)
    r0 = orbit.r  # initial position (m)
    v0 = orbit.v  # initial velocity (m/s)
    r, _ = ssapy.rv(orbit, t=t_array, propagator=propagator)
    return r


if __name__ == "__main__":
    # Set the initial epoch
    t0 = ssapy.Time("2025-1-1T12:00:00.000", scale="utc")
    
    # Define propagation parameters for the spacecraft
    kwargs = dict(
        mass=250,   # [kg]
        area=0.25,  # [m^2]
        CD=2.3,     # Drag coefficient
        CR=1.3,     # Radiation pressure coefficient
    )
    
    # Set up celestial bodies and acceleration models
    earth = get_body("earth", model="egm2008")
    moon = get_body("moon")
    sun = get_body("sun")
    
    aEarth = AccelKepler() + AccelHarmonic(earth, 180, 180)
    aMoon = AccelThirdBody(moon) + AccelHarmonic(moon)
    aSun = AccelThirdBody(sun)
    aSolRad = AccelSolRad()
    aEarthRad = AccelEarthRad()
    aDrag = AccelDrag()
    accel = aEarth + aMoon + aSun + aSolRad + aEarthRad + aDrag
    
    propagator = SciPyPropagator(accel)
    
    # Define initial orbital elements for circular orbits
    # Initial orbit (GEO)
    a1 = RGEO
    e1 = 0.0
    i1 = 0.0
    pa1 = 0.0
    raan1 = 0.0
    ta1 = 0.0
    
    # Target orbit (2×GEO)
    a2 = 2 * RGEO
    e2 = 0.0
    i2 = 0.0
    pa2 = 0.0
    raan2 = 0.0
    ta2 = 0.0
    
    # Initialize the orbits using SSApy's Keplerian element initializer
    initial_orbit = ssapy.Orbit.fromKeplerianElements(a1, e1, i1, pa1, raan1, ta1, t0, propkw=kwargs)
    target_orbit = ssapy.Orbit.fromKeplerianElements(a2, e2, i2, pa2, raan2, ta2, t0, propkw=kwargs)
    
    # Extract state vectors from the orbits
    r1 = initial_orbit.r  # [m]
    v1 = initial_orbit.v  # [m/s]
    r2 = target_orbit.r   # [m]
    v2 = target_orbit.v   # [m/s]
    
    # Set a nominal time-of-flight (e.g., 12 hours in seconds)
    tof_input = 12 * 3600
    
    # Compute transfer orbit parameters (this will select Hohmann since the vectors are collinear)
    delta_v1, delta_v2, v1_trans, v2_trans, tof_used, method = compute_transfer(r1, v1, r2, v2, tof_input)
    
    print(f"Transfer Method: {method}")
    print(f"Initial Delta-V: {delta_v1:.2f} m/s")
    print(f"Final Delta-V: {delta_v2:.2f} m/s")
    print(f"Time-of-Flight used: {tof_used:.2f} s")
    
    # Create the transfer orbit by updating the initial velocity with the computed maneuver.
    # Attempt to use fromRV; if not available, create a simple orbit-like object.
    try:
        transfer_orbit = ssapy.Orbit.fromRV(r1, v1_trans, t0, propkw=kwargs)
    except AttributeError:
        class SimpleOrbit:
            pass
        transfer_orbit = SimpleOrbit()
        transfer_orbit.r = r1
        transfer_orbit.v = v1_trans
        transfer_orbit.t0 = t0
        transfer_orbit.propkw = kwargs
    
    # Obtain orbital periods directly from the Orbit objects.
    T_initial = initial_orbit.period  # period in seconds from the Orbit object
    T_target = target_orbit.period
    
    # Propagate each orbit
    initial_positions = integrate_orbit(initial_orbit, T_initial, propagator)
    transfer_positions = integrate_orbit(transfer_orbit, tof_used, propagator)
    target_positions = integrate_orbit(target_orbit, T_target, propagator)
    
    # Optionally, compute gamma for the transfer orbit using Yeager's utility
    t_array = np.linspace(0, tof_used, transfer_positions.shape[0])
    gamma, heading = calc_gamma_and_heading(transfer_positions, t_array)
    print(f"Final gamma for transfer orbit: {gamma[-1]:.2f} (units as defined by calc_gamma_and_heading)")
    
    # Plot the orbits in the x-y plane
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(initial_positions[:, 0], initial_positions[:, 1], label="Initial Orbit (GEO)", color="blue", linewidth=2)
    ax.plot(transfer_positions[:, 0], transfer_positions[:, 1], label=f"Transfer Orbit ({method})", color="red", linestyle="--", linewidth=2)
    ax.plot(target_positions[:, 0], target_positions[:, 1], label="Target Orbit (2×GEO)", color="green", linewidth=2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Transfer Orbit using SSApy and Yeager Utils")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")
    plt.show()
