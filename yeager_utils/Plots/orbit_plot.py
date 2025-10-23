#!/usr/bin/env python3
"""
Test: ellipse_fit time array + SSAPy forward/backward propagation (Time-normalized)

Fix for VarType.MIXED_LIST:
- Ensure that the 't' argument passed to orbit_plot is a list of the SAME type.
- Here we standardize all times to yeager_utils.Time arrays.

Outputs:
- tests/testing_ellipse_fit_against_ssapy_gcrf.png
- tests/testing_ellipse_fit_against_ssapy_lunar.png
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
import numpy as np
from ssapy.simple import ssapy_orbit
from ssapy.plotUtils import orbit_plot

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def times_as_Time(duration_s, n_samples):
    """
    Build a Time array of length n_samples spanning 0..duration_s.
    This avoids mixed types when plotting multiple trajectories.
    """
    out = get_times(duration=(float(duration_s), "s"), freq=(int(n_samples), "s"))
    # get_times may return (TimeArray, step_seconds, etc.) -> take the first item
    return out[0] if isinstance(out, (list, tuple)) else out

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Two ECI points (meters)
    P1 = np.array([RGEO, 0.0, 0.0])
    P2 = np.array([-0.0 * RGEO, -1.1 * RGEO, 0.1 * RGEO])

    # --------------------------------------------------------------
    # Run ellipse_fit (choose ccw=True by default)
    # --------------------------------------------------------------
    from yeager_utils import RGEO, get_times, pprint, figpath, ellipse_fit
    res = ellipse_fit(
        P1,
        P2,
        n_pts=400,
        plot=False,
        inc=0.0,
        ccw=True,
    )
    pprint(res)

    # Unpack
    r_arc   = res["r"]           # (N,3)
    v_arc   = res["v"]           # (N,3)
    t_rel   = np.asarray(res["t_rel"], dtype=float)  # seconds, increasing
    r0, v0  = res["r0"], res["v0"]
    r2, v2  = r_arc[-1], v_arc[-1]

    T_flight = float(t_rel[-1])
    N_samps  = int(t_rel.shape[0])

    # Build a Time array for the ellipse_fit arc: [t0 .. t0+T_flight]
    t_fit = times_as_Time(T_flight, N_samps)  # Time array

    # --------------------------------------------------------------
    # SSAPy forward: start from (r0,v0), propagate for T_flight
    # --------------------------------------------------------------
    r_fwd, v_fwd, t_fwd_raw = ssapy_orbit(
        r=r0,
        v=v0,
        duration=(T_flight, "s"),
        freq=(N_samps, "s"),
    )
    # Force times for plotting to be Time arrays (match t_fit exactly)
    t_fwd = t_fit

    # --------------------------------------------------------------
    # SSAPy "backward": start from (r2,v2), integrate forward and reverse
    # --------------------------------------------------------------
    r_b_fwd, v_b_fwd, t_b_fwd_raw = ssapy_orbit(
        r=r2,
        v=v2,
        duration=(T_flight, "s"),
        freq=(N_samps, "s"),
    )
    r_back = r_b_fwd[::-1].copy()
    # Descending Time array for the backward run (explicitly requested)
    t_back = t_fit[::-1].copy()

    # --------------------------------------------------------------
    # Plot in GCRF and Lunar frames; ensure t is a list of Time arrays
    # --------------------------------------------------------------
    r_list = [r_arc, r_fwd, r_back]
    t_list = [t_fit, t_fwd, t_back]  # all Time, no mixing

    save_gcrf = figpath("tests/testing_ellipse_fit_against_ssapy_gcrf.png")
    orbit_plot(
        r_list,
        t=t_list,
        title="Ellipse-fit arc vs SSAPy forward/backward (GCRF)",
        frame="gcrf",
        show=False,
        save_path=save_gcrf,
        c="black",
        pad=1,
    )
    print(f"[saved] {save_gcrf}")

    save_lunar = figpath("tests/testing_ellipse_fit_against_ssapy_lunar.png")
    orbit_plot(
        r_list,
        t=t_list,
        title="Ellipse-fit arc vs SSAPy forward/backward (Lunar)",
        frame="lunar",
        show=False,
        save_path=save_lunar,
        c="black",
        pad=1,
    )
    print(f"[saved] {save_lunar}")

    # --------------------------------------------------------------
    # Endpoint sanity checks
    # --------------------------------------------------------------
    err_fwd_start = np.linalg.norm(r_fwd[0] - r0)
    err_fwd_end   = np.linalg.norm(r_fwd[-1] - r2)
    err_back_start = np.linalg.norm(r_back[0] - r2)
    err_back_end   = np.linalg.norm(r_back[-1] - r0)

    print("\nEndpoint checks [m]:")
    print(f"  Forward start vs P1: {err_fwd_start:.6e}")
    print(f"  Forward end   vs P2: {err_fwd_end:.6e}")
    print(f"  Backward start vs P2: {err_back_start:.6e}")
    print(f"  Backward end   vs P1: {err_back_end:.6e}")

    print("\nDONE!")
