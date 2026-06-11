"""Plot a propagated transfer trajectory with annotated burns.

Draws the transfer arc, the departure/arrival orbits (reconstructed from
the trajectory's boundary states), the Earth, and a marker at each burn
location annotated with its strength: delta-v magnitude, duration, the
acceleration flown, and -- when an engine model was used -- the thrust
and propellant estimate.

Works with a ``TransferResult`` (from ``transfer_ssapy``) or an
``OptimalTransferResult`` (from ``transfer_optimal``); the result must
have been produced with ``propagate=True`` so a trajectory exists.
"""

import numpy as np

from ssapy.orbit import Orbit
from ssapy.propagator import KeplerianPropagator
from ssapy.compute import rv
from ssapy.constants import EARTH_MU, EARTH_RADIUS


def _burn_label(i, b):
    a = b.dv_mag / (b.t_end - b.t_start)
    label = (f"burn {i}: {b.dv_mag:.1f} m/s\n"
             f"{a:.3f} m/s$^2$ x {b.t_end - b.t_start:.0f} s")
    if getattr(b, "thrust", None) is not None:
        label += f"\nF = {b.thrust:.0f} N"
    if getattr(b, "propellant_mass", None) is not None:
        label += f", prop ~{b.propellant_mass:.1f} kg"
    return label


def _orbit_ring(r, v, n=361):
    orb = Orbit(np.asarray(r, float), np.asarray(v, float), t=0.0)
    period = 2 * np.pi * np.sqrt(abs(orb.a) ** 3 / EARTH_MU)
    rr, _ = rv(orb, np.linspace(0.0, period, n),
               propagator=KeplerianPropagator())
    return rr


def transfer_trajectory_plot(result, ax=None, three_d=False,
                             show_orbits=True, show_earth=True,
                             annotate_burns=True, title=None,
                             save_path=None):
    """Plot a transfer trajectory with burn locations and strengths.

    Parameters
    ----------
    result : TransferResult or OptimalTransferResult
        A propagated transfer (``propagate=True``).
    ax : matplotlib axes, optional
        Draw onto existing axes (e.g. a gallery panel); otherwise a new
        figure is created.  Must be a 3-D axes when ``three_d=True``.
    three_d : bool
        Render in 3-D (useful for plane-change transfers).
    show_orbits, show_earth, annotate_burns : bool
        Toggle the orbit rings, the Earth disk (2-D only), and the
        per-burn strength annotations.
    title : str, optional
        Axes title; a default with total delta-v and arrival error is
        used when omitted.
    save_path : str, optional
        If given, save the figure via ``ssapy_toolkit.plots.yufig`` and
        close it; otherwise the axes are returned for further styling.

    Returns
    -------
    matplotlib axes (when ``save_path`` is None)
    """
    transfer = getattr(result, "transfer", result)
    if transfer.trajectory is None:
        raise ValueError("result has no trajectory; rerun the transfer "
                         "with propagate=True.")
    import matplotlib
    if save_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        if three_d:
            fig = plt.figure(figsize=(9, 8))
            ax = fig.add_subplot(projection="3d")
        else:
            fig, ax = plt.subplots(figsize=(7.5, 7.5))
    else:
        fig = ax.get_figure()

    tt = transfer.trajectory["t"]
    tr = transfer.trajectory["r"] / 1e3
    tv = transfer.trajectory["v"]

    def plot(xyz, *a, **kw):
        cols = (xyz[:, 0], xyz[:, 1], xyz[:, 2]) if three_d \
            else (xyz[:, 0], xyz[:, 1])
        return ax.plot(*cols, *a, **kw)

    if show_orbits:
        plot(_orbit_ring(tr[0] * 1e3, tv[0]) / 1e3, "C0--", lw=0.9,
             label="departure orbit")
        plot(_orbit_ring(tr[-1] * 1e3, tv[-1]) / 1e3, "C2--", lw=0.9,
             label="arrival orbit")
    if show_earth and not three_d:
        ang = np.linspace(0, 2 * np.pi, 181)
        ax.fill(EARTH_RADIUS / 1e3 * np.cos(ang),
                EARTH_RADIUS / 1e3 * np.sin(ang), color="0.85")
    plot(tr, "C3-", lw=2, label="transfer")

    for i, b in enumerate(transfer.burns, 1):
        # Burn location: trajectory position at the burn start.
        rb = np.array([np.interp(b.t_start, tt, tr[:, k])
                       for k in range(3)])
        if three_d:
            ax.scatter(*rb, color="k", marker="*", s=120, zorder=5)
            if annotate_burns:
                ax.text(*rb, "  " + _burn_label(i, b), fontsize=8)
        else:
            ax.plot(rb[0], rb[1], "k*", ms=13, zorder=5)
            if annotate_burns:
                ax.annotate(_burn_label(i, b), rb[:2],
                            textcoords="offset points", xytext=(10, 8),
                            fontsize=8)

    if not three_d:
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    if three_d:
        ax.set_zlabel("z [km]")
        lim = np.max(np.abs(tr)) * 1.1
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
    if title is None:
        title = (f"dv {transfer.dv_total:.1f} m/s | "
                 f"arrival err {transfer.arrival_error:.1f} m")
    ax.set_title(title, fontsize=10)

    if save_path is not None:
        from ssapy_toolkit.plots import yufig
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        yufig(fig, save_path)
        plt.close(fig)
        return None
    return ax
