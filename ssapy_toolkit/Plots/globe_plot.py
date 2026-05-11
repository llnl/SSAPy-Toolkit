import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image as PILImage
import numpy as np

from astropy.time import Time
from erfa import gst94

from ssapy.utils import find_file
from .plotutils import make_black, make_white, save_plot, valid_orbits
from ..constants import RGEO, EARTH_RADIUS


def _earth_lon0_from_time(t):
    """
    Compute a z-axis rotation angle (degrees) for the globe texture
    using the same GPS→TT→GST mapping as drawEarth [1].

    Parameters
    ----------
    t : astropy.time.Time or float
        Time at which to compute Earth's rotation. If float, interpreted
        as GPS seconds since 1980-01-06 00:00:00 UTC.

    Returns
    -------
    lon0 : float
        Rotation angle in degrees.
    """
    if isinstance(t, Time):
        t_gps = t.gps
    else:
        t_gps = float(t)

    # Same mapping as drawEarth: GPS seconds -> TT MJD [1]
    mjd_tt = 44244.0 + (t_gps + 51.184) / 86400.0
    gst = gst94(2400000.5, mjd_tt)  # radians
    return np.degrees(gst)


def globe_plot(
    r,
    t=None,
    limits=None,
    title="",
    c="black",
    figsize=(7, 8),
    save_path=None,
    el=5.0,
    az=5.0,
    scale=5.0,
    fontsize=18,
    labels=None,          # legend labels, one per orbit
    orbit_colors=None,    # per-orbit color(s)
    show_legend=True,     # toggle legend
    legend_kwargs=None,   # kwargs passed to ax.legend()
    lon0=0.0,             # manual rotation of globe about z-axis in degrees
    globe_time=None,      # optional time to orient globe based on Earth rotation
):
    """
    Plot a textured Earth and scatter satellite positions in 3D.

    Parameters
    ----------
    r, t : passed to valid_orbits to normalize [1].
    limits : None, scalar, or [[x1,x2],[y1,y2],[z1,z2]]
        None -> auto; scalar -> cube [-L,L] for all axes;
        3x2-like -> explicit per-axis limits.
    labels : list of str, optional
        Per-orbit labels for legend; must match number of tracks if provided.
    orbit_colors : list or array-like, optional
        Per-orbit colors. If provided, length must match number of tracks.
        Each entry can be any Matplotlib color (name, hex, RGB tuple, etc.).
    show_legend : bool
        If True and labels are provided, draw a legend.
    legend_kwargs : dict
        Extra kwargs forwarded to ax.legend().
    lon0 : float
        Rotation of the globe texture about the z-axis in degrees. Positive
        values rotate longitudes eastward.
    globe_time : astropy.time.Time or float, optional
        If provided, overrides lon0 by computing the Earth-rotation angle
        from this time using gst94 (same as drawEarth) [1].

    Returns
    -------
    fig, ax : Matplotlib Figure and 3D Axes.
    """

    # ---- Normalize/validate inputs ----
    r_list, t_list = valid_orbits(r, t)  # t_list kept for consistency [1]
    n_tracks = len(r_list)

    # --- Labels sanity ---
    if labels is not None and len(labels) != n_tracks:
        raise ValueError("labels must have same length as number of orbits (tracks)")

    # --- Orbit colors sanity ---
    if orbit_colors is not None and len(orbit_colors) != n_tracks:
        raise ValueError("orbit_colors must have same length as number of orbits (tracks)")

    # --- If globe_time is provided, override lon0 based on Earth rotation ---
    if globe_time is not None:
        lon0 = _earth_lon0_from_time(globe_time)

    # ---------- Theme ----------
    if c in ("black", "b"):
        plotcolor, textcolor = "black", "white"
    elif c in ("white", "w"):
        plotcolor, textcolor = "white", "black"
    else:
        plotcolor, textcolor = "white", "black"

    # ---------- Earth texture ----------
    scale = 1.0 if (not np.isfinite(scale) or scale <= 0) else float(scale)
    tex_w = int(round(5400 / scale))
    tex_h = int(round(2700 / scale))
    earth_png = PILImage.open(find_file("earth", ext=".png")).resize(
        (tex_w, tex_h), resample=PILImage.BILINEAR
    )
    bm = np.asarray(earth_png, dtype=float) / 255.0  # (H, W, C)

    # ---------- Globe mesh (match texture resolution) ----------
    lons = np.radians(np.linspace(-180.0, 180.0, bm.shape[1]))
    lats = np.radians(np.linspace(-90.0, 90.0, bm.shape[0])[::-1])

    # Apply longitude offset (rotate globe about z-axis)
    lon0_rad = np.radians(float(lon0))
    lons = lons + lon0_rad

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    scale_fac = EARTH_RADIUS / RGEO
    mesh_x = np.cos(lat_grid) * np.cos(lon_grid) * scale_fac
    mesh_y = np.cos(lat_grid) * np.sin(lon_grid) * scale_fac
    mesh_z = np.sin(lat_grid) * scale_fac

    # ---------- Figure and axes ----------
    fig = plt.figure(dpi=100, figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor(plotcolor)
    ax.set_facecolor(plotcolor)
    ax.view_init(elev=float(el), azim=float(az))
    ax.tick_params(axis="both", colors=textcolor)
    ax.grid(True, color="grey", linestyle="--", linewidth=0.5)

    ax.plot_surface(mesh_x, mesh_y, mesh_z, rstride=4, cstride=4, facecolors=bm, shade=False)

    # ---------- Colors for satellites ----------
    if orbit_colors is not None:
        colors = list(orbit_colors)
    else:
        # 1 orbit -> colormap over points; >1 -> one color per orbit
        if n_tracks == 1:
            colors = None  # handled in loop
        else:
            colors = [plt.cm.rainbow(v) for v in np.linspace(0.0, 1.0, n_tracks)]

    # ---------- Scatter satellites ----------
    max_extent = 0.0

    for i, ri in enumerate(r_list):
        ri_scaled = ri / RGEO
        if ri_scaled.size:
            max_extent = max(max_extent, float(np.nanmax(np.abs(ri_scaled))))

        if n_tracks == 1:
            # Single orbit:
            if orbit_colors is not None:
                color = orbit_colors[0]
                ax.scatter(
                    ri_scaled[:, 0], ri_scaled[:, 1], ri_scaled[:, 2],
                    color=color, s=1,
                )
            else:
                # original: color by point
                point_colors = plt.cm.rainbow(np.linspace(0.0, 1.0, ri_scaled.shape[0]))
                ax.scatter(
                    ri_scaled[:, 0], ri_scaled[:, 1], ri_scaled[:, 2],
                    color=point_colors, s=1,
                )
        else:
            # Multiple orbits: one color per orbit
            color = colors[i] if colors is not None else None
            ax.scatter(
                ri_scaled[:, 0], ri_scaled[:, 1], ri_scaled[:, 2],
                color=color, s=1,
            )

    # ---------- Axis limits: scalar or explicit ----------
    # limits can be:
    #   None -> auto
    #   scalar -> cube [-L,L]
    #   [[x1,x2],[y1,x2],[z1,z2]] -> explicit
    if limits is None:
        L = (max_extent if max_extent > 0 else scale_fac) * 1.2
        xlim = ylim = zlim = (-L, L)
        limits_is_scalar_or_none = True
    elif np.isscalar(limits):
        L = float(limits)
        xlim = ylim = zlim = (-L, L)
        limits_is_scalar_or_none = True
    else:
        limits = np.asarray(limits, dtype=float)
        if limits.shape != (3, 2):
            raise ValueError("limits must be scalar or array-like of shape (3,2): [[x1,x2],[y1,y2],[z1,z2]]")
        xlim = tuple(limits[0])
        ylim = tuple(limits[1])
        zlim = tuple(limits[2])
        limits_is_scalar_or_none = False

    # Backward compatibility with old large values (e.g., meters)
    if limits_is_scalar_or_none and max(abs(xlim[0]), abs(xlim[1])) > 1e5:
        xlim = (xlim[0] / RGEO, xlim[1] / RGEO)
        ylim = (ylim[0] / RGEO, ylim[1] / RGEO)
        zlim = (zlim[0] / RGEO, zlim[1] / RGEO)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    # ---------- Ticks/labels ----------
    # For symmetric cubes use integer ticks around center; for general limits, keep it simple
    if xlim[0] == -xlim[1] and ylim[0] == -ylim[1] and zlim[0] == -zlim[1]:
        L_tick = int(xlim[1])
        ax.set_xticks([-L_tick, 0, L_tick])
        ax.set_yticks([-L_tick, 0, L_tick])
        ax.set_zticks([-L_tick, 0, L_tick])

    # hide tick labels as in your original
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_xlabel("x [GEO]", color=textcolor, fontsize=fontsize)
    ax.set_ylabel("y [GEO]", color=textcolor, fontsize=fontsize)
    ax.set_zlabel("z [GEO]", color=textcolor, fontsize=fontsize)
    if title:
        ax.set_title(title, color=textcolor, fontsize=fontsize)

    ax.tick_params(axis="x", colors=textcolor, labelsize=fontsize)
    ax.tick_params(axis="y", colors=textcolor, labelsize=fontsize)
    ax.tick_params(axis="z", colors=textcolor, labelsize=fontsize)

    # ---------- Legend (lines instead of dots) ----------
    if show_legend and labels is not None:
        handles = []
        for i, label in enumerate(labels):
            if orbit_colors is not None:
                color = orbit_colors[i]
            else:
                if n_tracks == 1:
                    color = plt.cm.rainbow(0.5)
                else:
                    color = colors[i]
            h = Line2D(
                [0], [0],
                color=color,
                linewidth=2,
                label=label,
            )
            handles.append(h)

        kw = dict(loc="best")
        if legend_kwargs:
            kw.update(legend_kwargs)
        ax.legend(handles=handles, **kw)

    # Apply theme helpers
    if c in ("black", "b"):
        fig, ax = make_black(fig, ax)
    elif c in ("white", "w"):
        fig, ax = make_white(fig, ax)

    if save_path:
        save_plot(fig, save_path)

    return fig, ax