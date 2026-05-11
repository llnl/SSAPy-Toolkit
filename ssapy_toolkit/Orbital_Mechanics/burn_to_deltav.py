# ssapy_toolkit/Orbital_Mechanics/burn_to_deltav.py

import numpy as np
from ssapy import Orbit
from ssapy_toolkit.Coordinates import ntw_to_gcrf
from ssapy_toolkit.Integrators import leapfrog

def burn_to_deltav(orbit, times, burn_ntw):
    """
    Compare continuous finite-duration NTW acceleration vs an instantaneous impulse
    applied at the mid-time of `times`.

    Parameters
    ----------
    orbit : ssapy.Orbit
    times : 1D array of seconds (monotonic)
    burn_ntw : (3,) constant NTW acceleration [m/s^2],
               interpreted as [Normal(out-of-plane), Tangential, Radial].

    Returns
    -------
    dict: keys r_continuous, r_instantaneous, delta_v_ntw, delta_v_gcrf, t_center
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

    # Map NTW -> leapfrog profiles
    burn_ntw = np.asarray(burn_ntw, float)
    a_incl = burn_ntw[0]   # along +h_hat  (out-of-plane)   <-- swap if your NTW order differs
    a_tan  = burn_ntw[1]   # along +v_hat  (tangential)
    a_rad  = burn_ntw[2]   # along +r_hat  (radial)

    # Constant profiles across the window
    radial_prof      = a_rad
    velocity_prof    = a_tan
    inclination_prof = a_incl

    # Continuous thrust propagation over the window
    r_cont, v_cont = leapfrog(r0, v0, t,
                              radial=radial_prof,
                              velocity=velocity_prof,
                              inclination=inclination_prof)

    # Impulsive approximation at mid-time:
    # 1) Kepler-only to center
    k = t.size // 2
    t_center = 0.5 * (t[0] + t[-1])
    r_pre, v_pre = leapfrog(r0, v0, t[:k+1],
                            radial=None, velocity=None, inclination=None)
    r_c = r_pre[-1]
    v_c = v_pre[-1]

    # 2) Equivalent delta-v = a * duration, convert NTW->GCRF at center
    duration = t[-1] - t[0]
    delta_v_ntw = burn_ntw * duration
    delta_v_gcrf = ntw_to_gcrf(delta_v_ntw, r_c, v_c)

    # 3) Apply impulse and Kepler-only forward from center
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
    }
