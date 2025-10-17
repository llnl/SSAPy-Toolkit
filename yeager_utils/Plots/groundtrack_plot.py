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
):
    """
    Pretty ground-track plot (styled like the dashboard subplot).

    Parameters
    ----------
    r : (n,3) array_like or list of (n,3)
        GCRF positions [m]. Single orbit or a list of orbits.
    t : (n,) array_like or list of (n,)
        Absolute times matching r (e.g., gpstime/epoch seconds or Time objects supported by ssapy.groundTrack).
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

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # normalize inputs
    r_list = _as_list(r)
    t_list = _broadcast_time_list(r_list, t)

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
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(r_list))))

    # plot each orbit
    for i, (ri, ti) in enumerate(zip(r_list, t_list)):
        # get ground track (radians)
        lon, lat, _h = groundTrack(np.asarray(ri), ti, format='geodetic')
        lon_deg = np.degrees(lon)
        lat_deg = np.degrees(lat)

        # clean wrap
        lon_plot, lat_plot = _clean_lonlat_wrap(lon_deg, lat_deg, threshold=179.0)

        # line + start/end markers
        label = f"Orbit {i+1}"
        ax.plot(lon_plot, lat_plot, color=colors[i % len(colors)], linewidth=2.5, label=label)
        if start_end_markers and len(lon_deg) > 0:
            ax.plot(lon_deg[0],  lat_deg[0],  marker='*', color=colors[i % len(colors)], markersize=12, linestyle='None')
            ax.plot(lon_deg[-1], lat_deg[-1], marker='x', color=colors[i % len(colors)], markersize=9,  linestyle='None')

    # ground stations (lat, lon in deg)
    if ground_stations is not None:
        gs = np.asarray(ground_stations, dtype=float)
        if gs.ndim == 2 and gs.shape[1] == 2:
            ax.scatter(gs[:, 1], gs[:, 0], s=50, color='red', label="Ground Station")

    # legend
    if show_legend:
        # custom entries to match dashboard semantics
        from matplotlib.lines import Line2D
        base = [
            Line2D([0], [0], color='black', linewidth=2.5, label='Orbit Track'),
            Line2D([0], [0], marker='*', color='black', linestyle='None', markersize=12, label='Orbit Start'),
            Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=10, label='Orbit End')
        ]
        # add ground station handle only if provided
        if ground_stations is not None:
            base.append(Line2D([0], [0], marker='o', color='red', linestyle='None', markersize=8, label='Ground Station'))
        ax.legend(handles=base, loc='lower left', fontsize=fontsize-2)

    # finalize
    if save_path:
        save_plot(fig, save_path)
    return fig
