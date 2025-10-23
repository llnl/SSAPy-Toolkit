#!/usr/bin/env python3
"""
Test: ellipse_fit time array + SSAPy forward/backward propagation

Fixes:
- Ensure each Time array length matches its corresponding r-array length.
- Use a single cadence for all trajectories: dt = T_flight / (N_samps - 1).

Goal
-----
1) Run ellipse_fit(P1, P2, ...) and use a consistent Time array.
2) Propagate with SSAPy from P1->P2 using that Time array (ascending).
3) Propagate "backwards" from P2->P1 using the same Time array reversed (descending).

Outputs
-------
- Saves a sanity plot to figpath("tests/testing_ellipse_fit_vs_ssapy.png")
- Prints endpoint errors.

Notes
-----
- Uses numpy only (no math, no typing).
- Uses yeager_utils.figpath for saving; no manual mkdirs.
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
import numpy as np
from yeager_utils import (
    RGEO,
    ellipse_fit,
    figpath,
    pprint,
    orbit_plot,
    groundtrack_dashboard,
    Time,
    get_times,
)  # type: ignore
from ssapy.simple import ssapy_orbit  # type: ignore

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define two points (meters) in ECI-like frame
    P1 = np.array([RGEO, 0.0, 0.0])
    P2 = np.array([-0.0 * RGEO, -1.1 * RGEO, 0.1 * RGEO])

    # ------------------------------------------------------------------
    # Run ellipse_fit once and extract arc + relative times (seconds)
    # ------------------------------------------------------------------
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
    r_arc   = res["r"]                       # (N, 3)
    v_arc   = res["v"]                       # (N, 3)
    t_rel   = np.asarray(res["t_rel"])       # (N,) seconds, ascending
    r0, v0  = res["r0"], res["v0"]
    r2, v2  = r_arc[-1], v_arc[-1]

    T_flight = float(t_rel[-1])
    N_samps  = int(t_rel.shape[0])

    # ------------------------------------------------------------------
    # Build a canonical Time array of the SAME length as r arrays
    # ------------------------------------------------------------------
    # Use dt = T/(N-1) so that we get exactly N samples from 0..T inclusive.
    dt_s = T_flight / max(N_samps - 1, 1)
    _times = get_times(duration=(T_flight, "s"), freq=(dt_s, "s"))
    t_fit  = _times[0] if isinstance(_times, (list, tuple)) else _times       # ascending Time array, length N
    t_fit_desc = t_fit[::-1].copy()                                            # descending Time array, length N

    # ------------------------------------------------------------------
    # 1) Forward SSAPy propagation: P1 -> P2 using same cadence
    # ------------------------------------------------------------------
    r_fwd, v_fwd, _ = ssapy_orbit(
        r=r0,
        v=v0,
        duration=(T_flight, "s"),
        freq=(dt_s, "s"),
    )
    # Ensure length match (guard slight off-by-one inside integrator)
    if len(r_fwd) != N_samps:
        if len(r_fwd) > N_samps:
            r_fwd = r_fwd[:N_samps]
        else:
            # pad last sample if short (rare)
            pad = np.repeat(r_fwd[-1][None, :], N_samps - len(r_fwd), axis=0)
            r_fwd = np.vstack([r_fwd, pad])

    # ------------------------------------------------------------------
    # 2) "Backward" propagation: integrate from P2 forward, then reverse
    # ------------------------------------------------------------------
    r_b_fwd, v_b_fwd, _ = ssapy_orbit(
        r=r2,
        v=v2,
        duration=(T_flight, "s"),
        freq=(dt_s, "s"),
    )
    if len(r_b_fwd) != N_samps:
        if len(r_b_fwd) > N_samps:
            r_b_fwd = r_b_fwd[:N_samps]
        else:
            pad = np.repeat(r_b_fwd[-1][None, :], N_samps - len(r_b_fwd), axis=0)
            r_b_fwd = np.vstack([r_b_fwd, pad])

    r_back = r_b_fwd[::-1].copy()

    # ------------------------------------------------------------------
    # Endpoint checks (meters)
    # ------------------------------------------------------------------
    err_fwd_start = np.linalg.norm(r_fwd[0] - r0)
    err_fwd_end   = np.linalg.norm(r_fwd[-1] - r2)
    err_back_start = np.linalg.norm(r_back[0] - r2)
    err_back_end   = np.linalg.norm(r_back[-1] - r0)

    print("\nEndpoint checks (meters):")
    print(f"  Forward start vs P1: {err_fwd_start:.6e}")
    print(f"  Forward end   vs P2: {err_fwd_end:.6e}")
    print(f"  Backward start vs P2: {err_back_start:.6e}")
    print(f"  Backward end   vs P1: {err_back_end:.6e}")

    # ------------------------------------------------------------------
    # Plot: ellipse_fit arc, SSAPy forward, SSAPy backward
    # ------------------------------------------------------------------
    r_list = [r_arc, r_fwd, r_back]
    t_list = [t_fit, t_fit, t_fit_desc]   # each Time array length == matching r length

    save_path = figpath("tests/testing_ellipse_fit_vs_ssapy.png")
    orbit_plot(
        r_list,
        t_list,
        title="Ellipse-fit vs SSAPy forward/backward propagation",
        save_path=save_path,
        frame="gcrf",
    )
    print(f"\n[saved] {save_path}")

    # Optional dashboard
    groundtrack_dashboard(r_list, t_list, save_path=figpath("tests/testing_ellipse_fit_vs_ssapy_dashboard.png"))

    print("\nDONE!")
