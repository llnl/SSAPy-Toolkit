"""
Demo: run compare_models (SSAPy accel ladder) and save the two dashboard figures via yufig.

Assumes:
    from ssapy_toolkit import compare_models

and compare_models(...) returns:
    out["dashboard"]["figure_time_domain"]
    out["dashboard"]["figure_rung_summary"]

Figures are saved into:
    "/tests"
(note: yufig normalizes under ~/yu_figures)
"""

import numpy as np


def main():
    from astropy.time import Time
    import ssapy

    from ssapy_toolkit import compare_models
    from ssapy_toolkit.Plots.plotutils import yufig

    # Demo orbit (replace with real values as needed)
    t0 = Time("2026-01-21T00:00:00", scale="utc")

    r0_m = np.array([7000e3, 0.0, 0.0], dtype=float)
    v0_mps = np.array([0.0, 7.5e3, 1.0e3], dtype=float)

    orbit = ssapy.Orbit(r=r0_m, v=v0_mps, t=t0)

    # Time grid: 1 hour, 60-second cadence (offset seconds from epoch)
    dt_s = 60.0
    duration_s = 1 * 3600.0
    times = np.arange(0.0, duration_s + dt_s, dt_s, dtype=float)

    # Run ladder + divergence dashboard
    # NOTE: compare_models always plots now; do not pass make_plots.
    out = compare_models(
        orbit=orbit,
        times=times,
        assume_times="offset",
        ode_kwargs={"rtol": 1e-10, "atol": 1e-12},
        reference=None,
        plot_title="Demo: SSAPy accel ladder divergences",
        show_legend=True,
        epsilon_m=1e-3,
    )

    dash = out["dashboard"]

    fig_time = None
    fig_rung = None
    if isinstance(dash, dict):
        fig_time = dash.get("figure_time_domain", None)
        fig_rung = dash.get("figure_rung_summary", None)

        # fallback if only "figures" list exists
        if (fig_time is None or fig_rung is None) and isinstance(dash.get("figures", None), list):
            figs = dash["figures"]
            if len(figs) >= 2:
                fig_time = figs[0]
                fig_rung = figs[1]

    if fig_time is None or fig_rung is None:
        raise RuntimeError("Expected dashboard figures not found in output.")

    # Save using yufig. Provide directory-like prefix; yufig will create the path under ~/yu_figures.
    out_dir = "/tests"
    yufig(fig_time, out_dir + "/accel_ladder_time_domain.jpg", dpi=200)
    yufig(fig_rung, out_dir + "/accel_ladder_rung_summary.jpg", dpi=200)

    print("Saved (yufig root-normalized):")
    print("  " + out_dir + "/accel_ladder_time_domain.jpg")
    print("  " + out_dir + "/accel_ladder_rung_summary.jpg")


if __name__ == "__main__":
    main()
