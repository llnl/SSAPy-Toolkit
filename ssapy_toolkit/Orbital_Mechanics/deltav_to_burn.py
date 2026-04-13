# ssapy_toolkit/Orbital_Mechanics/deltav_to_burn.py

import numpy as np
from ssapy import Orbit
from ssapy_toolkit.Coordinates import ntw_to_gcrf
from ssapy_toolkit.Integrators import leapfrog

def deltav_to_burn(orbit, times, delta_v_ntw):
    """
    Use a constant NTW acceleration across `times` such that a*duration = delta_v_ntw,
    and compare continuous vs impulsive results.

    Parameters
    ----------
    orbit : ssapy.Orbit
    times : 1D array of seconds (monotonic)
    delta_v_ntw : (3,) NTW delta-v [m/s],
                  interpreted as [Normal(out-of-plane), Tangential, Radial].

    Returns
    -------
    dict: keys r_continuous, r_instantaneous, delta_v_ntw, delta_v_gcrf, t_center, a_ntw
    """
    r0 = np.asarray(orbit.r, float)
    v0 = np.asarray(orbit.v, float)
    t0 = float(orbit.t)

    t = np.asarray(times, float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("times must be a 1D array with at least 2 samples")

    # Align start time with orbit.t if needed
    if not np.isclose(t[0], t0, atol=1e-6):
        t = t - (t[0] - t0)

    duration = float(t[-1] - t[0])
    if duration <= 0.0:
        raise ValueError("times must span a positive duration")

    delta_v_ntw = np.asarray(delta_v_ntw, float)
    a_ntw = delta_v_ntw / duration

    # Map NTW -> leapfrog profiles
    a_incl = a_ntw[0]   # out-of-plane
    a_tan  = a_ntw[1]   # along-track
    a_rad  = a_ntw[2]   # radial

    radial_prof      = a_rad
    velocity_prof    = a_tan
    inclination_prof = a_incl

    # Continuous thrust
    r_cont, v_cont = leapfrog(r0, v0, t,
                              radial=radial_prof,
                              velocity=velocity_prof,
                              inclination=inclination_prof)

    # Impulsive approximation at mid-time
    k = t.size // 2
    t_center = 0.5 * (t[0] + t[-1])

    r_pre, v_pre = leapfrog(r0, v0, t[:k+1],
                            radial=None, velocity=None, inclination=None)
    r_c = r_pre[-1]
    v_c = v_pre[-1]

    delta_v_gcrf = ntw_to_gcrf(delta_v_ntw, r_c, v_c)

    v_after = v_c + delta_v_gcrf
    r_post, v_post = leapfrog(r_c, v_after, t[k:],
                              radial=None, velocity=None, inclination=None)

    r_inst = np.vstack([r_pre[:-1], r_post])

    return {
        "r_continuous":     r_cont,
        "r_instantaneous":  r_inst,
        "delta_v_ntw":      delta_v_ntw,
        "delta_v_gcrf":     delta_v_gcrf,
        "t_center":         t_center,
        "a_ntw":            a_ntw,
    }
