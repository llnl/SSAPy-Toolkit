import numpy as np
import matplotlib.pyplot as plt

from ssapy import groundTrack
from ..Time_Functions import Time
from .plotutils import load_earth_file, save_plot


def _as_list(x):
    """Ensure input is a list (without copying big arrays unnecessarily)."""
    return x if isinstance(x, (list, tuple)) else [x]


def _broadcast_time_list(r_list, t):
    """Return a time list matching r_list length."""
    if isinstance(t, (list, tuple)):
        if len(t) != len(r_list):
            raise ValueError("When passing a list of times, its length must match the number of orbits.")
        return list(t)
    # single time array -> reuse for all
    return [t for _ in r_list]


def _clean_lonlat_wrap(lon_deg, lat_deg, threshold=179.0):
    """Insert NaNs at 180° crossings so lines do not jump across the map."""
    jumps = np.where(np.abs(np.diff(lon_deg)) > threshold)[0]
    if jumps.size == 0:
        return lon_deg, lat_deg
    lon_out = np.insert(lon_deg, jumps + 1, np.nan)
    lat_out = np.insert(lat_deg, jumps + 1, np.nan)
    return lon_out, lat_out


def groundtrack_plot(
    r,
    t,
    ground_stations=None,
    save_path=None,
    title="Ground Track",
    show_legend=True,
    fontsize=18,
    start_end_markers=True,
    labels=None,          # NEW: per-orbit legend labels
    orbit_colors=None,    # NEW: per-orbit colors
    legend_kwargs=None,   # NEW: kwargs passed to ax.legend()
):
    """
    Pretty ground-track plot (styled like the dashboard subplot).

    Parameters
    ----------
    r : (n,3) array_like or list of (n,3)
        GCRF positions [m]. Single orbit or a list of orbits.
    t : (n,) array_like or list of (n,)
        Absolute times matching r.
        If r is a list, either pass a matching list of time arrays or a single array reused for all.
    ground_stations : (k,2) array_like, optional
        (lat_deg, lon_deg) rows.
    save_path : str, optional
        If provided, saves the figure via `save_plot(fig, save_path)`.
    title : str, optional
        Title at the top of the plot.
    show_legend : bool, default True
        Show legend for track + markers.
    fontsize : int, default 18
        Base font size for labels/ticks.
    start_end_markers : bool, default True
        Draw star at start and 'x' at end for each orbit.
    labels : list of str, optional
        Per-orbit labels; must have same length as number of orbits if provided.
    orbit_colors : list or array-like, optional
        Per-orbit colors. If provided, length must match number of orbits.
        Each can be any Matplotlib color spec.
    legend_kwargs : dict, optional
        Extra kwargs forwarded to ax.legend().

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # normalize inputs
    r_list = _as_list(r)
    t_list = _broadcast_time_list(r_list, t)
    n_tracks = len(r_list)

    # sanity checks for labels/colors
    if labels is not None and len(labels) != n_tracks:
        raise ValueError("labels must have same length as number of orbits (tracks)")
    if orbit_colors is not None and len(orbit_colors) != n_tracks:
        raise ValueError("orbit_colors must have same length as number of orbits (tracks)")

    # figure
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)

    # background earth image (if available)
    try:
        ax.imshow(load_earth_file(), extent=[-180, 180, -90, 90], aspect='auto', zorder=-1)
    except Exception:
        pass

    # axes styling
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (deg)", fontsize=fontsize)
    ax.set_ylabel("Latitude (deg)", fontsize=fontsize)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=fontsize-2)
    ax.set_title(title, fontsize=fontsize+4)

    # colors
    if orbit_colors is not None:
        colors = list(orbit_colors)
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, n_tracks)))

    # keep handles if you want legend entries per orbit
    line_handles = []

    # plot each orbit
    for i, (ri, ti) in enumerate(zip(r_list, t_list)):
        # get ground track (radians)
        lon, lat, _h = groundTrack(np.asarray(ri), ti, format='geodetic')
        lon_deg = np.degrees(lon)
        lat_deg = np.degrees(lat)

        # clean wrap
        lon_plot, lat_plot = _clean_lonlat_wrap(lon_deg, lat_deg, threshold=179.0)

        color = colors[i % len(colors)]
        label = labels[i] if labels is not None else f"Orbit {i+1}"

        # line + start/end markers
        line, = ax.plot(
            lon_plot,
            lat_plot,
            color=color,
            linewidth=2.5,
            label=label,
        )
        line_handles.append(line)

        if start_end_markers and len(lon_deg) > 0:
            ax.plot(
                lon_deg[0], lat_deg[0],
                marker='*', color=color,
                markersize=12, linestyle='None',
            )
            ax.plot(
                lon_deg[-1], lat_deg[-1],
                marker='x', color=color,
                markersize=9, linestyle='None',
            )

    # ground stations (lat, lon in deg)
    gs_handle = None
    if ground_stations is not None:
        gs = np.asarray(ground_stations, dtype=float)
        if gs.ndim == 2 and gs.shape[1] == 2:
            gs_handle = ax.scatter(
                gs[:, 1], gs[:, 0],
                s=50, color='red', label="Ground Station",
            )

    # legend
    if show_legend:
        handles = list(line_handles)
        if gs_handle is not None:
            handles.append(gs_handle)

        kw = dict(loc='lower left', fontsize=fontsize-2)
        if legend_kwargs:
            kw.update(legend_kwargs)
        ax.legend(handles=handles, **kw)

    # finalize
    if save_path:
        save_plot(fig, save_path)
    return fig