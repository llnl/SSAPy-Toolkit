#!/usr/bin/env python3
"""
Test: ellipse_fit vs multiple SSAPy propagations using yeager_utils props (no ssapy.simple)

What this script does
---------------------
1) Builds the ellipse_fit arc between P1->P2 and resamples it onto a uniform time grid.
2) Runs SSAPy propagations from the same initial state (r0, v0) using yeager_utils props:
   - keplerian_prop(): 2-body Kepler-only
   - ssapy_prop():     nominal multi-accel model (Earth EGM2008 harm., Sun/Moon, SRP, Earth rad)
   - best_prop():      heavier model with planets + non-conservative forces
3) Optional backward propagation using time-reversal (r2, -v2) to demonstrate correctness.
4) Puts ALL versions on the existing plots:
   - orbit_plot
   - groundtrack_dashboard
5) KEEPS the distance-vs-time plot between each propagation and the ellipse_fit reference,
   and gives each line a distinct dash style so they do not visually overlap.

Notes
-----
- Uses numpy for math (no math, no typing).
- Time grid is canonical: dt = T/(N-1), N samples on [0..T].
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
import numpy as np
import inspect
import matplotlib.pyplot as plt

from yeager_utils import (
    RGEO,
    ellipse_fit,
    figpath,
    pprint,
    orbit_plot,
    groundtrack_dashboard,
    get_times,
    # use these props from yeager_utils (ignore ssapy.simple)
    ssapy_orbit,        # wrapper that takes a propagator via `prop=...`
    keplerian_prop,
    ssapy_prop,
    best_prop,
)  # type: ignore


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def resample_cartesian(time_src_s, r_src, time_dst_s):
    """
    Resample a 3D Cartesian trajectory r_src(t) from time_src_s onto time_dst_s
    using per-component linear interpolation.
    """
    time_src_s = np.asarray(time_src_s, dtype=float)
    r_src = np.asarray(r_src, dtype=float)
    time_dst_s = np.asarray(time_dst_s, dtype=float)

    if time_src_s.ndim != 1 or time_dst_s.ndim != 1:
        raise ValueError("time arrays must be 1-D")
    if r_src.ndim != 2 or r_src.shape[1] != 3 or r_src.shape[0] != time_src_s.shape[0]:
        raise ValueError("r_src must be shape (N,3) matching time_src_s length")

    if np.any(np.diff(time_src_s) < 0.0):
        raise ValueError("time_src_s must be ascending")

    r_dst = np.empty((time_dst_s.shape[0], 3), dtype=float)
    for k in range(3):
        r_dst[:, k] = np.interp(time_dst_s, time_src_s, r_src[:, k])
    return r_dst


def trim_or_pad_to(arr, n_samples):
    """Ensure arr has exactly n_samples rows by trimming or padding last sample."""
    arr = np.asarray(arr, dtype=float)
    if arr.shape[0] == n_samples:
        return arr
    if arr.shape[0] > n_samples:
        return arr[:n_samples]
    pad = np.repeat(arr[-1][None, :], n_samples - arr.shape[0], axis=0)
    return np.vstack([arr, pad])


def _supports_kwarg(func, kw):
    """Return True if callable `func` accepts keyword arg `kw` or **kwargs."""
    try:
        sig = inspect.signature(func)
        if kw in sig.parameters:
            return True
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return True
    except Exception:
        pass
    return False


def _call_with_optional_labels(func, *args, labels=None, **kwargs):
    """Call func and include labels only if the function supports it."""
    if labels is not None and _supports_kwarg(func, "labels"):
        return func(*args, labels=labels, **kwargs)
    return func(*args, **kwargs)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Two points (meters) in an inertial frame
    P1 = np.array([RGEO, 0.0, 0.0], dtype=float)
    P2 = np.array([-0.0 * RGEO, -1.1 * RGEO, 0.1 * RGEO], dtype=float)

    # Ellipse fit arc
    res = ellipse_fit(P1, P2, n_pts=400, plot=False, inc=0.0, ccw=True)
    pprint(res)

    r_arc = np.asarray(res["r"], dtype=float)     # (N,3)
    v_arc = np.asarray(res["v"], dtype=float)     # (N,3)
    t_rel = np.asarray(res["t_rel"], dtype=float) # (N,)

    r0, v0 = np.asarray(res["r0"], float), np.asarray(res["v0"], float)
    r2, v2 = r_arc[-1].copy(), v_arc[-1].copy()

    # Sanity and normalization
    if t_rel.ndim != 1 or r_arc.shape[0] != t_rel.shape[0]:
        raise RuntimeError("ellipse_fit returned inconsistent lengths for r and t_rel")
    if t_rel[0] != 0.0:
        t_rel = t_rel - t_rel[0]

    T_flight = float(t_rel[-1])
    N_native = int(t_rel.shape[0])

    # Canonical time grid and plotting Time arrays
    if N_native > 1:
        t_grid = np.linspace(0.0, T_flight, N_native, dtype=float)
        dt_s = T_flight / float(N_native - 1)
    else:
        t_grid = np.array([0.0], dtype=float)
        dt_s = 0.0

    _times = get_times(duration=(T_flight, "s"), freq=((dt_s if dt_s > 0.0 else 0.0), "s"))
    t_fit = _times[0] if isinstance(_times, (list, tuple)) else _times
    t_fit_desc = t_fit[::-1].copy()

    # Resample ellipse-fit arc onto the uniform grid
    r_efit = resample_cartesian(t_rel, r_arc, t_grid)

    # ------------------------------------------------------------------
    # Build propagators from yeager_utils (no ssapy.simple anywhere)
    # ------------------------------------------------------------------
    prop_kep   = keplerian_prop()
    prop_nom   = ssapy_prop()     # nominal multi-accel
    prop_best  = best_prop()      # heavier model

    # ------------------------------------------------------------------
    # Forward propagations from identical initial state (r0, v0)
    # ------------------------------------------------------------------
    def run_with_prop(prop):
        r, v, t = ssapy_orbit(
            r=r0, v=v0,
            duration=(T_flight, "s"),
            freq=((dt_s, "s") if N_native > 1 else (0.0, "s")),
            prop=prop,
        )
        return trim_or_pad_to(r, N_native)

    r_kep  = run_with_prop(prop_kep)
    r_nom  = run_with_prop(prop_nom)
    r_best = run_with_prop(prop_best)

    # ------------------------------------------------------------------
    # Optional: time-reversed sanity (start at r2, -v2)
    # ------------------------------------------------------------------
    r_back, v_unused, t_unused = ssapy_orbit(
        r=r2, v=-v2,
        duration=(T_flight, "s"),
        freq=((dt_s, "s") if N_native > 1 else (0.0, "s")),
        prop=prop_nom,  # pick one model for backward check
    )
    r_back = trim_or_pad_to(r_back, N_native)  # pairs naturally with reversed efit

    # ------------------------------------------------------------------
    # Metrics (printed only)
    # ------------------------------------------------------------------
    def rms_max_delta(a, b):
        d = a - b
        rms = np.sqrt(np.mean(np.sum(d * d, axis=1)))
        mx = np.max(np.linalg.norm(d, axis=1))
        return rms, mx

    print("\nArc deltas vs ellipse_fit (meters):")
    rms_kep,  max_kep  = rms_max_delta(r_kep,  r_efit)
    rms_nom,  max_nom  = rms_max_delta(r_nom,  r_efit)
    rms_best, max_best = rms_max_delta(r_best, r_efit)
    print(f"  keplerian_prop  RMS: {rms_kep:.6e}  MAX: {max_kep:.6e}")
    print(f"  ssapy_prop      RMS: {rms_nom:.6e}  MAX: {max_nom:.6e}")
    print(f"  best_prop       RMS: {rms_best:.6e} MAX: {max_best:.6e}")

    # Backward vs reversed ellipse
    rms_back, max_back = rms_max_delta(r_back, r_efit[::-1])
    print(f"  backward (ssapy_prop) vs reverse(efit)  RMS: {rms_back:.6e}  MAX: {max_back:.6e}")

    # ------------------------------------------------------------------
    # Build lists for existing plots
    #   - Keep backward paired with descending time for clarity
    # ------------------------------------------------------------------
    r_list = [r_efit, r_kep, r_nom, r_best, r_back]
    t_list = [t_fit,  t_fit, t_fit, t_fit,  t_fit_desc]
    labels = ["ellipse_fit", "keplerian_prop", "ssapy_prop", "best_prop", "backward(ssapy_prop)"]

    # Existing plots only: orbit_plot and dashboard
    save_path = figpath("tests/testing_ellipse_fit_vs_ssapy.jpg")
    _call_with_optional_labels(
        orbit_plot,
        r_list,
        t_list,
        title="Ellipse-fit vs SSAPy (keplerian, ssapy_prop, best_prop, backward)",
        save_path=save_path,
        frame="gcrf",
        labels=labels,
    )
    print(f"\n[saved] {save_path}")

    save_dash = figpath("tests/testing_ellipse_fit_vs_ssapy_dashboard.jpg")
    _call_with_optional_labels(
        groundtrack_dashboard,
        r_list,
        t_list,
        save_path=save_dash,
        labels=labels,
    )
    print(f"[saved] {save_dash}")

    # ------------------------------------------------------------------
    # KEEP: Distance-to-ellipse plot (km) for each propagation
    #   - forward methods vs same-direction ellipse_fit
    #   - backward vs reversed ellipse_fit
    #   - each line uses a distinct dash pattern
    # ------------------------------------------------------------------
    t_minutes = t_grid / 60.0
    curves = [
        np.linalg.norm(r_kep  - r_efit,       axis=1) / 1e3,
        np.linalg.norm(r_nom  - r_efit,       axis=1) / 1e3,
        np.linalg.norm(r_best - r_efit,       axis=1) / 1e3,
        np.linalg.norm(r_back - r_efit[::-1], axis=1) / 1e3,
    ]
    names = [
        "keplerian_prop vs ellipse_fit",
        "ssapy_prop vs ellipse_fit",
        "best_prop vs ellipse_fit",
        "backward(ssapy_prop) vs reverse(efit)",
    ]

    # Define a variety of dash patterns (offset, on_off_seq)
    dash_styles = [
        (0, (1, 1)),                # very dotted
        (0, (4, 2)),                # short dash
        (0, (8, 2)),                # long dash
        (0, (4, 1, 1, 1)),          # dash-dot
        (0, (10, 3, 1, 3)),         # long-short-dot
        (0, (3, 1, 1, 1, 1, 1)),    # dash-dot-dot
        (0, (2, 3)),                # sparse dots
        (0, (5, 5)),                # even dash
    ]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    for i, (d, n) in enumerate(zip(curves, names)):
        ls = dash_styles[i % len(dash_styles)]
        ax.plot(t_minutes, d, linewidth=2.5, linestyle=ls, label=n)
    ax.set_xlabel("Time since start (minutes)")
    ax.set_ylabel("Distance to reference (km)")
    ax.set_title("Distance to ellipse_fit (or reversed) vs time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_dist = figpath("tests/testing_ellipse_fit_distance.jpg")
    fig.savefig(save_dist, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_dist}")

    print("\nDONE!")
