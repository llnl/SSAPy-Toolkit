"""Plot the burn timeline of a transfer: where each burn occurs in time
and how strong it is.

Top panel: commanded acceleration magnitude versus time -- a step
profile that is ``|dv|/duration`` inside each burn window and zero while
coasting -- with each burn block annotated by its delta-v, duration,
acceleration, and (when an engine model was used) thrust and propellant
estimate.  Bottom panel: cumulative delta-v expended along the transfer.

Works with a ``TransferResult`` (from ``transfer_ssapy``) or an
``OptimalTransferResult`` (from ``transfer_optimal``).
"""

import numpy as np


def transfer_burn_profile_plot(result, title=None, save_path=None):
    """Plot acceleration-vs-time and cumulative delta-v for all burns.

    Parameters
    ----------
    result : TransferResult or OptimalTransferResult
    title : str, optional
    save_path : str, optional
        If given, save via ``ssapy_toolkit.plots.yufig`` and close;
        otherwise the figure is returned.
    """
    transfer = getattr(result, "transfer", result)
    burns = transfer.burns
    if transfer.trajectory is not None:
        t0 = float(transfer.trajectory["t"][0])
        t1 = float(transfer.trajectory["t"][-1])
    else:
        t0 = burns[0].t_start
        t1 = burns[-1].t_end

    import matplotlib
    if save_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True,
        gridspec_kw=dict(height_ratios=[2, 1]))

    th = lambda t: (t - t0) / 3600.0
    a_max = 0.0
    for i, b in enumerate(burns, 1):
        dur = b.t_end - b.t_start
        a = b.dv_mag / dur
        a_max = max(a_max, a)
        ax1.fill_between([th(b.t_start), th(b.t_end)], 0, a,
                         color=f"C{i - 1}", alpha=0.75, step="pre")
        label = (f"burn {i}: {b.dv_mag:.1f} m/s\n"
                 f"{a:.3f} m/s$^2$ x {dur:.0f} s")
        if getattr(b, "thrust", None) is not None:
            label += f"\nF = {b.thrust:.0f} N"
        if getattr(b, "propellant_mass", None) is not None:
            label += f"\nprop ~{b.propellant_mass:.1f} kg"
        ax1.annotate(label,
                     (th(0.5 * (b.t_start + b.t_end)), a),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8)
    ax1.set_ylim(0, a_max * 1.45 if a_max > 0 else 1)
    ax1.set_xlim(th(t0), th(t1))
    ax1.set_ylabel("commanded acceleration [m/s$^2$]")
    ax1.grid(alpha=0.3)
    ax1.set_title(title or
                  f"Burn timeline: total dv {transfer.dv_total:.1f} m/s "
                  f"across {len(burns)} burn(s)")

    # Cumulative delta-v: piecewise-linear ramps inside burn windows.
    ts = [t0]
    dvs = [0.0]
    total = 0.0
    for b in burns:
        ts += [b.t_start, b.t_end]
        dvs += [total, total + b.dv_mag]
        total += b.dv_mag
    ts.append(t1)
    dvs.append(total)
    ax2.plot([th(t) for t in ts], dvs, "C3-", lw=2)
    ax2.set_xlabel("time since departure [h]")
    ax2.set_ylabel("cumulative dv [m/s]")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        from ssapy_toolkit.plots import yufig
        yufig(fig, save_path)
        plt.close(fig)
        return None
    return fig
