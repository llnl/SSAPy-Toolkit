import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image as PILImage
import numpy as np

from astropy.time import Time
from erfa import gst94

from ssapy.utils import find_file
from .plotutils import make_black, make_white, save_plot, valid_orbits
from ..constants import RGEO, EARTH_RADIUS
from .starfield import add_starfield

# NEW: real Sun ephemeris + texture shading + Sun sphere renderer
from core.sun_mpl import (
    sun_direction_in_frame,
    shade_texture,
    draw_sun,
    auto_sun_distance,
    auto_sun_radius,
)


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



def _view_unit_vector(el_deg, az_deg):
    el = np.radians(float(el_deg))
    az = np.radians(float(az_deg))
    v = np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ], dtype=float)
    n = np.linalg.norm(v)
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v / n


def _earth_occultation_mask(r_scaled, earth_radius_scaled, el_deg, az_deg):
    """
    Return boolean mask where True means point is visible (not hidden behind Earth)
    for the current camera direction defined by (el, az).
    """
    r_scaled = np.asarray(r_scaled, dtype=float)
    if r_scaled.ndim != 2 or r_scaled.shape[1] != 3:
        raise ValueError("r_scaled must be shape (N, 3)")

    # Unit vector pointing from origin toward the viewer
    v_hat = _view_unit_vector(el_deg, az_deg)

    # Signed depth along viewing axis
    depth = r_scaled @ v_hat

    # Perpendicular distance from viewing axis
    perp = r_scaled - np.outer(depth, v_hat)
    rho = np.linalg.norm(perp, axis=1)

    # Points are hidden if they are behind the planet center and project inside the Earth disk
    hidden = (depth < 0.0) & (rho < float(earth_radius_scaled))
    return ~hidden


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
    show_sun=True,        # NEW: shade Earth by real solar ephemeris + draw Sun
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
    show_sun : bool
        If True, compute the real Sun direction at globe_time (or the first
        orbit's epoch, or "now" if neither is given), shade the Earth
        texture so the night side is dark, and draw a Sun sphere placed
        closer than the starfield.

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

    # --- Reference epoch for Sun direction & Earth rotation ---
    t_for_sun = globe_time
    if t_for_sun is None and len(t_list) > 0 and t_list[0] is not None:
        t_for_sun = t_list[0]

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
    tex_w = int(round(1080 / scale))
    tex_h = int(round(540 / scale))
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

    # ---------- Sun direction (real ephemeris, GCRF == this plot's frame) ──
    sun_hat = None
    if show_sun:
        try:
            ref_time = t_for_sun
            if ref_time is None:
                ref_time = Time.now()
            sun_hat = sun_direction_in_frame(ref_time, transform_func=None)
        except Exception as e:
            print(f"[globe_plot] Sun ephemeris failed, rendering unshaded: {e}")
            sun_hat = None

    # ---------- Apply shading to texture (if Sun available) ----------
    if sun_hat is not None:
        # lat/lon grid gives us surface normals directly: the globe mesh
        # IS the unit-sphere normal scaled by scale_fac, so reuse it.
        nx = np.cos(lat_grid) * np.cos(lon_grid)
        ny = np.cos(lat_grid) * np.sin(lon_grid)
        nz = np.sin(lat_grid)
        dot = nx * sun_hat[0] + ny * sun_hat[1] + nz * sun_hat[2]
        lit = 0.22 + 0.78 * np.clip(dot, 0.0, 1.0)
        bm_shaded = np.clip(bm[..., :3] * lit[..., None], 0.0, 1.0)
        if bm.shape[-1] == 4:
            bm_render = np.concatenate([bm_shaded, bm[..., 3:4]], axis=-1)
        else:
            bm_render = bm_shaded
    else:
        bm_render = bm

    # ---------- Figure and axes ----------
    fig = plt.figure(dpi=100, figsize=figsize, facecolor=plotcolor)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor(plotcolor)
    ax.set_facecolor(plotcolor)
    ax.view_init(elev=float(el), azim=float(az))
    ax.tick_params(axis="both", colors=textcolor)
    ax.grid(True, color="grey", linestyle="--", linewidth=0.5)
    if c in ("black", "b"):
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_alpha(0)
            pane.set_edgecolor('none')
        ax.grid(False)
    ax.plot_surface(mesh_x, mesh_y, mesh_z, rstride=2, cstride=2, facecolors=bm_render, shade=False)

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

        visible = _earth_occultation_mask(
            ri_scaled,
            earth_radius_scaled=scale_fac,
            el_deg=el,
            az_deg=az,
        )
        ri_vis = ri_scaled[visible]

        if ri_vis.size == 0:
            continue

        if n_tracks == 1:
            # Single orbit:
            if orbit_colors is not None:
                color = orbit_colors[0]
                ax.scatter(
                    ri_vis[:, 0], ri_vis[:, 1], ri_vis[:, 2],
                    color=color, s=1,
                )
            else:
                # original: color by point
                point_colors = plt.cm.rainbow(np.linspace(0.0, 1.0, ri_scaled.shape[0]))
                point_colors = point_colors[visible]
                ax.scatter(
                    ri_vis[:, 0], ri_vis[:, 1], ri_vis[:, 2],
                    color=point_colors, s=1,
                )
        else:
            # Multiple orbits: one color per orbit
            color = colors[i] if colors is not None else None
            ax.scatter(
                ri_vis[:, 0], ri_vis[:, 1], ri_vis[:, 2],
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
    # ── Star background (dark theme only) ────────────────────────────────────
    if c in ("black", "b"):
        plot_range = max(abs(xlim[0]), abs(xlim[1]),
                         abs(ylim[0]), abs(ylim[1]),
                         abs(zlim[0]), abs(zlim[1])) * 2
        print(f"DEBUG globe starfield: plot_range={plot_range:.3f} el={el} az={az}")
        add_starfield(ax, plot_range, elev=float(el), azim=float(az),
                      epoch=t_for_sun)

        # ── Sun sphere ────────────────────────────────────────────────────
        # auto_sun_distance()/auto_sun_radius() expect a real km scale
        # (they have km-scale floor constants, e.g. a minimum visual sun
        # radius of tens of thousands of km) — but `plot_range` here is in
        # this plot's GEO-normalized units (1.0 == RGEO == 42,164 km), so
        # a typical LEO-scale plot_range of ~0.3-2 was being read as
        # "0.3-2 km", which is far below those floors. The floor value
        # (tens of thousands of km) then got plotted directly as if it
        # were 0.3-2 GEO-units, making the sun sphere enormous relative to
        # the whole scene — filling (and extending past) the entire frame
        # with flat colour. Convert to km for the function calls, then
        # back to GEO-normalized units for plotting.
        if sun_hat is not None:
            plot_range_km = plot_range * RGEO
            sun_dist_km   = auto_sun_distance(plot_range_km)
            sun_radius_km = auto_sun_radius(plot_range_km)
            sun_dist      = sun_dist_km / RGEO
            sun_radius    = sun_radius_km / RGEO
            print(f"[globe_plot] plot_range={plot_range:.3f} GEO-units "
                  f"({plot_range_km:.0f} km) -> sun_dist={sun_dist:.3f} GEO-units "
                  f"({sun_dist_km:.0f} km), sun_radius={sun_radius:.4f} GEO-units "
                  f"({sun_radius_km:.0f} km)")
            draw_sun(ax, sun_hat * sun_dist, radius=sun_radius)

    # Apply theme helpers
    if c in ("white", "w"):
        fig, ax = make_white(fig, ax)

    if save_path:
        save_plot(fig, save_path)

    return fig, ax

if __name__ == "__main__":
    # ── GUI / standalone entry point ─────────────────────────────────────────
    # Uses the satellites already fetched into ~/ssapy_satellites.json
    # (via tle_updater.py) instead of re-fetching from Space-Track every
    # time — faster, doesn't burn API rate limit, and lets you pick a
    # handful to actually look at instead of always plotting everything.
    #
    # Select which satellites to plot via GUI_CONFIG:
    #   select = ["ISS", "Hubble", "GOES"]   # case-insensitive name match
    #   norad_ids = [25544, 20580]           # or by exact NORAD ID
    # With neither set, defaults to a small friendly set below.
    import os
    import sys as _sys

    for _search in [
        os.path.join(os.path.dirname(__file__), "..", ".."),
        os.path.expanduser("~"),
        os.path.join(os.path.expanduser("~"), "SSAPy-Toolkit"),
    ]:
        _search = os.path.abspath(_search)
        if _search not in _sys.path and os.path.isdir(_search):
            _sys.path.insert(0, _search)

    cfg = {}
    cfg_path = os.environ.get("GUI_CONFIG")
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            exec(f.read(), cfg)
    else:
        print("[globe_plot] No GUI_CONFIG found — running with defaults.")

    output_dir = cfg.get(
        "output_dir",
        os.path.join(os.path.expanduser("~"), "yu_figures", "demo_gallery", "figures"),
    )
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "globe_plot.jpg")

    try:
        from tle_updater import load_satellites, _norad_from_tle
        _all_sats = load_satellites()
    except Exception as ex:
        print(f"[globe_plot] Could not load satellites from tle_updater ({ex}) — "
              f"run tle_updater.py --add-group <group> first.")
        _all_sats = []
        _norad_from_tle = lambda l1: None

    _select_names = cfg.get("select", ["ISS", "Hubble", "GOES"])
    _select_norad = cfg.get("norad_ids", None)

    def _matches(sat):
        if _select_norad is not None:
            _nid = _norad_from_tle(sat.get("line1", ""))
            return _nid is not None and int(_nid) in [int(n) for n in _select_norad]
        name = sat.get("name", "")
        return any(s.lower() in name.lower() for s in _select_names)

    _chosen = [s for s in _all_sats if s.get("type") == "tle" and _matches(s)]

    if not _chosen:
        print(f"[globe_plot] No satellites matched select={_select_names!r} "
              f"norad_ids={_select_norad!r} out of {len(_all_sats)} loaded. "
              f"Nothing to plot — check tle_updater's saved names/NORAD IDs.")
        _sys.exit(1)

    print(f"[globe_plot] Selected {len(_chosen)}/{len(_all_sats)} satellites: "
          f"{', '.join(s['name'] for s in _chosen)}")

    _COLOR_CYCLE = ["#00e5ff", "#ffd400", "#7cfc00", "#ff5f5f", "#c77dff", "#ff9e3f"]

    r_list, t_list, label_list, color_list = [], [], [], []
    try:
        from core import OrbitalState

        for _i, sat in enumerate(_chosen):
            _name = sat.get("name", f"Sat {_i}")
            try:
                state = OrbitalState.from_tle(f"{_name}\n{sat['line1']}\n{sat['line2']}")
                n_orbits = cfg.get("n_orbits", 1.0 if state.a_km > 20_000 else 3.0)
                traj = state.propagate(n_orbits=n_orbits, dt_s=cfg.get("dt_s", 60.0))
                if traj.ok:
                    r_list.append(traj.r * 1e3)   # km -> m (globe_plot expects meters)
                    t_list.append(traj.t)
                    label_list.append(_name)
                    color_list.append(_COLOR_CYCLE[_i % len(_COLOR_CYCLE)])
                    print(f"[globe_plot] {_name}: OK, a={state.a_km:.0f} km "
                          f"e={state.e:.4f} i={state.inc_deg:.1f}°")
                else:
                    print(f"[globe_plot] {_name}: propagation failed ({traj.msg}) — skipping.")
            except Exception as ex:
                print(f"[globe_plot] {_name}: {ex} — skipping.")
    except Exception as ex:
        print(f"[globe_plot] core.OrbitalState unavailable ({ex}) — cannot propagate.")

    if not r_list:
        print("[globe_plot] All selected satellites failed to propagate — "
              "nothing to plot. See errors above (this is very likely the "
              "orbit_state.py `_solve_kepler.__func__` bug — see chat for the fix).")
        _sys.exit(1)

    # Single reference epoch for sun-direction shading (first satellite's
    # first timestep) — must be a real astropy Time, not a raw GPS-seconds
    # float or array.
    _ref_epoch = Time(float(t_list[0][0]), format="gps")

    fig, ax = globe_plot(
        r=r_list, t=t_list,
        labels=label_list, orbit_colors=color_list, show_legend=True,
        title=cfg.get("title", f"{len(r_list)} satellite(s)"),
        c=cfg.get("c", "black"),
        show_sun=cfg.get("show_sun", True),
        globe_time=_ref_epoch,
        save_path=out_path,
    )
    print(f"[globe_plot] Plotted {len(r_list)} satellite(s): {', '.join(label_list)}")
    print(f"[globe_plot] Saved -> {out_path}")