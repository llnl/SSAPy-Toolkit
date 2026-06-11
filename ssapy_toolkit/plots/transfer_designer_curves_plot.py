"""Mission-designer curves for an optimal-transfer search.

Left: the porkchop contour (departure time x time of flight -> objective
delta-v, log color scale) with infeasible candidates greyed out and the
chosen transfer starred.  Right: the delta-v versus time-of-flight trade
(Pareto front) broken out per burn -- total, departure burn, and arrival
burn -- with the chosen transfer and the delta-v budget line.

Takes the ``OptimalTransferResult`` returned by ``transfer_optimal``;
all curves are recreated from the stored search grid, so this plot can
be regenerated at any time from the result object alone.
"""

import numpy as np


def transfer_designer_curves_plot(result, save_path=None):
    """Plot porkchop + per-burn Pareto curves from a transfer_optimal
    result.

    Parameters
    ----------
    result : OptimalTransferResult
    save_path : str, optional
        If given, save via ``ssapy_toolkit.plots.yufig`` and close;
        otherwise the figure is returned.
    """
    import matplotlib
    if save_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    g = result.grid
    dep_h = (g["t_dep"] - g["t_dep"][0]) / 3600.0
    tof_h = g["tof"] / 3600.0
    dv_budget = getattr(result, "dv_budget", None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))

    cost = np.ma.masked_invalid(g["cost"])
    pc = ax1.pcolormesh(dep_h, tof_h, cost.T, shading="nearest",
                        norm=LogNorm(vmin=max(cost.min(), 1e-1),
                                     vmax=cost.max()), cmap="viridis")
    fig.colorbar(pc, ax=ax1, label="objective delta-v [m/s]")
    ax1.plot((result.t_depart - g["t_dep"][0]) / 3600.0,
             result.tof / 3600.0, "r*", ms=16, mec="w",
             label=f"chosen: {result.dv_total:.1f} m/s")
    ax1.set_xlabel("departure time into window [h]")
    ax1.set_ylabel("time of flight [h]")
    ax1.set_title("Porkchop (grey = infeasible: no 0-rev solution,\n"
                  "perigee below margin, or burns don't fit)")
    ax1.set_facecolor("0.85")
    ax1.legend(loc="upper right", fontsize=9)

    p = result.pareto
    ax2.plot(tof_h, p["dv"], "k.-", lw=2, label="total (best per TOF)")
    if "dv1" in p:
        ax2.plot(tof_h, p["dv1"], "C0.--", lw=1.2,
                 label="burn 1 (departure)")
    if "dv2" in p and result.arrival_burn:
        ax2.plot(tof_h, p["dv2"], "C1.--", lw=1.2,
                 label="burn 2 (arrival)")
    ax2.plot(result.tof / 3600.0, result.dv_total, "r*", ms=16, mec="w",
             label="chosen transfer")
    if dv_budget is not None:
        ax2.axhline(dv_budget, color="k", ls="--", lw=1,
                    label=f"delta-v budget ({dv_budget:.0f} m/s)")
    ax2.set_yscale("log")
    ax2.set_xlabel("time of flight [h]")
    ax2.set_ylabel("delta-v [m/s]")
    ax2.set_title("Delta-v vs time-of-flight trade, per burn")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)

    mode = "rendezvous" if result.rendezvous else "insertion"
    burns = "both burns" if result.arrival_burn else "first burn only"
    fig.suptitle(f"transfer_optimal: {result.objective}, {mode}, {burns}",
                 fontsize=12)
    fig.tight_layout()

    if save_path is not None:
        from ssapy_toolkit.plots import yufig
        yufig(fig, save_path)
        plt.close(fig)
        return None
    return fig
