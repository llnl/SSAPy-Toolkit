from ssapy import Orbit, AccelKepler, AccelConstNTW
from yeager_utils.Coordinates import ntw_to_gcrf
from yeager_utils.Integrators import leapfrog
import numpy as np


def burn_to_deltav(orbit, times, burn):
    """
    Compute continuous and instantaneous trajectories for an orbital maneuver with a constant NTW burn using Leapfrog.

    This function models a spacecraft maneuver by applying a constant acceleration in the NTW (Normal, Tangential,
    W-Normal) frame over the given time period, then approximates it as an instantaneous delta-v at the midpoint.
    It returns both the continuous trajectory (with thrust applied over time) and the instantaneous trajectory
    (with an equivalent delta-v applied at the center time), along with the delta-v in NTW and GCRF frames.

    Parameters
    ----------
    orbit : ssapy.Orbit
        Initial orbit object defining the spacecraft's state (position and velocity) at the starting time.
    times : numpy.ndarray
        Array of time points (in seconds) for burn propagation, assumed to be monotonically increasing.
    burn : numpy.ndarray
        Constant acceleration vector in the NTW frame (m/s^2), shape (3,) for [Normal, Tangential, W-Normal].

    Returns
    -------
    dict
        Dictionary containing:
        - 'r_continuous' : numpy.ndarray
            Position vectors (m) from continuous propagation over all `times`, shape (len(times), 3).
        - 'r_instantaneous' : numpy.ndarray
            Position vectors (m) from instantaneous delta-v approximation, shape (len(times), 3).
        - 'delta_v_ntw' : numpy.ndarray
            Delta-v vector in NTW frame (m/s), shape (3,).
        - 'delta_v_gcrf' : numpy.ndarray
            Delta-v vector in GCRF frame (m/s), shape (3,).
        - 't_center' : float
            Time (s) at the midpoint of the burn where the instantaneous delta-v is applied.

    Author
    ------
    Travis Yeager yeager7@llnl.gov
    """
    # Extract initial conditions from orbit
    r0 = orbit.r  # Initial position (m)
    v0 = orbit.v  # Initial velocity (m/s)
    t0 = orbit.t  # Initial time (s)

    # Adjust times to be relative to t0 if necessary
    times = np.asarray(times)
    if not np.isclose(times[0], t0, atol=1e-6):
        times = times - times[0] + t0  # Normalize to start at t0

    # Define acceleration functions
    accel_kepler = AccelKepler()  # Keplerian two-body acceleration
    accel_thrust = AccelConstNTW(accelntw=burn, time_breakpoints=[times[0], times[-1]])

    def accel_combined(r, t=0):  # Leapfrog expects accel as a function of position
        """Combined Keplerian and constant NTW acceleration."""
        v = leapfrog_velocity(r, t)  # Approximate velocity at this point (needed for NTW)
        return accel_kepler(r) + accel_thrust(r, v, t)

    def accel_only_kepler(r, t=0):
        """Keplerian acceleration only."""
        return accel_kepler(r)

    # Helper to approximate velocity during continuous propagation (for NTW frame)
    last_v = v0  # Store last velocity for approximation
    def leapfrog_velocity(r, t):
        nonlocal last_v
        return last_v  # Use last computed velocity (updated in loop)

    # Continuous trajectory with burn
    r_continuous, v_continuous = leapfrog(r0, v0, times, accel=accel_combined)
    # Update last_v during integration (Leapfrog doesn’t provide intermediate velocities directly)
    for i in range(len(times) - 1):
        last_v = v_continuous[i + 1]

    # Midpoint for instantaneous approximation
    center_index = len(times) // 2
    center_time = (times[-1] + times[0]) / 2
    times_impulse = times.copy()
    times_impulse[center_index] = center_time

    # Pre-burn trajectory (Keplerian only)
    r_preburn, v_preburn = leapfrog(r0, v0, times_impulse[:center_index + 1], accel=accel_only_kepler)
    r_center = r_preburn[-1]
    v_center = v_preburn[-1]

    # Calculate delta-v
    delta_t = times[-1] - times[0]
    delta_v_ntw = burn * delta_t
    delta_v_gcrf = ntw_to_gcrf(delta_v_ntw, r_center, v_center)

    # Post-burn trajectory (instantaneous delta-v applied)
    v_afterburn = v_center + delta_v_gcrf
    orbit_afterburn = Orbit(r=r_center, v=v_afterburn, t=center_time)
    r_afterburn, v_afterburn = leapfrog(r_center, v_afterburn, times_impulse[center_index:], accel=accel_only_kepler)

    # Combine instantaneous trajectory
    r_instantaneous = np.vstack((r_preburn[:-1], r_afterburn))

    return {
        'r_continuous': r_continuous,
        'r_instantaneous': r_instantaneous,
        'delta_v_ntw': delta_v_ntw,
        'delta_v_gcrf': delta_v_gcrf,
        't_center': center_time
    }
