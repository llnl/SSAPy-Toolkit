"""
Enhanced orbit_plot_xy.py — 3D camera with star field
======================================================
Uses matplotlib 3D axes with proper rendering order:
1. Stars rendered on far sphere first
2. Milky Way second  
3. Orbit track third
4. Earth/Moon last (on top)

This ensures correct visual layering.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

from ..constants import RGEO, EARTH_RADIUS, MOON_RADIUS
from ..coordinates import gcrf_to_itrf, gcrf_to_lunar, gcrf_to_lunar_fixed
from .plotutils import valid_orbits, save_plot
from ..compute import find_smallest_bounding_cube
from ssapy import get_body

_STAR_CACHE = None
_HYG_PATHS  = [
    os.path.expanduser("~/bright_stars.csv"),
    os.path.expanduser("~/SSAPy/ssapy/data/bright_stars.csv"),
    os.path.join(os.path.dirname(__file__), "bright_stars.csv"),
]

_SPECT_COLORS = {
    'O': [0.61, 0.69, 1.00],
    'B': [0.67, 0.75, 1.00],
    'A': [0.79, 0.85, 1.00],
    'F': [0.97, 0.97, 1.00],
    'G': [1.00, 0.96, 0.92],
    'K': [1.00, 0.82, 0.63],
    'M': [1.00, 0.80, 0.44],
}


def _load_stars(mag_limit=6.5):
    global _STAR_CACHE
    if _STAR_CACHE is not None and _STAR_CACHE.get('mag_limit') == mag_limit:
        return _STAR_CACHE
    csv_path = None
    for p in _HYG_PATHS:
        if os.path.exists(p):
            csv_path = p
            break
    if csv_path is None:
        return None
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        df = df[(df['mag'] < mag_limit) & (df['mag'] > -10)].copy()
        df = df.dropna(subset=['ra', 'dec', 'mag'])
        ra_rad  = np.radians(df['ra'].values * 15.0)
        dec_rad = np.radians(df['dec'].values)
        mag     = df['mag'].values
        cx = np.cos(dec_rad) * np.cos(ra_rad)
        cy = np.cos(dec_rad) * np.sin(ra_rad)
        cz = np.sin(dec_rad)
        sizes  = np.clip(0.8 * (mag_limit - mag) ** 1.2, 0.1, 8.0)
        spect  = df['spect'].fillna('G').str[:1].values
        colors = np.array([_SPECT_COLORS.get(s, _SPECT_COLORS['G']) for s in spect])
        names  = df['proper'].fillna('').values
        _STAR_CACHE = {
            'cx': cx, 'cy': cy, 'cz': cz,
            'mag': mag, 'sizes': sizes,
            'colors': colors, 'names': names,
            'n': len(mag), 'mag_limit': mag_limit,
        }
        print(f"[orbit_plot_xy] Loaded {len(mag):,} stars")
        return _STAR_CACHE
    except Exception as e:
        print(f"[orbit_plot_xy] Star catalog error: {e}")
        return None


def _camera_direction(elev_deg, azim_deg):
    e = np.radians(elev_deg)
    a = np.radians(azim_deg)
    cam = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    look = -cam
    return look / np.linalg.norm(look)


def _filter_stars_fov(stars, elev_deg, azim_deg, fov_deg, sky_radius):
    """
    Place stars on the celestial sphere with depth variation.
    Brighter stars placed closer, fainter stars further away,
    giving realistic 3D depth to the star field.
    """
    look      = _camera_direction(elev_deg, azim_deg)
    star_dirs = np.stack([stars['cx'], stars['cy'], stars['cz']], axis=1)
    dots      = star_dirs @ (-look)
    half_cos  = np.cos(np.radians(fov_deg / 2.0))
    mask      = dots > half_cos

    # Give each star a depth based on magnitude
    # Brighter (lower mag) = closer, fainter = further
    mag    = stars['mag'][mask]
    mag_min = mag.min()
    mag_max = mag.max()
    # Normalize magnitude to depth range 0.5 to 1.0 of sky_radius
    depth = 0.5 + 0.5 * (mag - mag_min) / (mag_max - mag_min + 1e-6)

    x = stars['cx'][mask] * sky_radius * depth
    y = stars['cy'][mask] * sky_radius * depth
    z = stars['cz'][mask] * sky_radius * depth

    return x, y, z, stars['sizes'][mask], stars['colors'][mask], dots[mask]

def _textured_sphere(ax, cx, cy, cz, radius, img_path, n=48):
    try:
        img     = Image.open(img_path).convert("RGB")
        img     = img.resize((256, 128), Image.LANCZOS)
        img_arr = np.array(img) / 255.0
        phi   = np.linspace(0,    np.pi,  n)
        theta = np.linspace(0, 2*np.pi, 2*n)
        PHI, THETA = np.meshgrid(phi, theta)
        x = cx + radius * np.sin(PHI) * np.cos(THETA)
        y = cy + radius * np.sin(PHI) * np.sin(THETA)
        z = cz + radius * np.cos(PHI)
        row = np.clip((PHI   / np.pi    * (img_arr.shape[0]-1)).astype(int), 0, img_arr.shape[0]-1)
        col = np.clip((THETA / (2*np.pi) * (img_arr.shape[1]-1)).astype(int), 0, img_arr.shape[1]-1)
        ax.plot_surface(x, y, z, facecolors=img_arr[row, col],
                        rstride=1, cstride=1, linewidth=0,
                        antialiased=True, shade=False)
        return True
    except Exception as e:
        print(f"[texture] {e}")
        return False


def _atmosphere(ax, r, n=48):
    for scale, color, alpha in [(1.04,'#1166cc',0.04),(1.02,'#44aaff',0.03),(1.01,'#66bbff',0.02)]:
        phi   = np.linspace(0,    np.pi,  n)
        theta = np.linspace(0, 2*np.pi, 2*n)
        P, T  = np.meshgrid(phi, theta)
        rr = r * scale
        ax.plot_surface(rr*np.sin(P)*np.cos(T), rr*np.sin(P)*np.sin(T), rr*np.cos(P),
                        color=color, alpha=alpha, linewidth=0, shade=False)


def _milky_way(ax, sky_r):
    gnp = np.array([np.cos(np.radians(27.13))*np.cos(np.radians(192.85)),
                    np.cos(np.radians(27.13))*np.sin(np.radians(192.85)),
                    np.sin(np.radians(27.13))])
    arb = np.array([0., 1., 0.])
    v1  = np.cross(gnp, arb); v1 /= np.linalg.norm(v1)
    theta = np.linspace(0, 2*np.pi, 360)
    for w, a in [(0.,0.08),(0.1,0.05),(0.2,0.03),(-0.1,0.05),(-0.2,0.03)]:
        n_ = gnp + w*v1; n_ /= np.linalg.norm(n_)
        b1 = np.cross(n_, arb)
        if np.linalg.norm(b1) < 1e-6: b1 = np.cross(n_, np.array([1,0,0]))
        b1 /= np.linalg.norm(b1)
        b2 = np.cross(n_, b1); b2 /= np.linalg.norm(b2)
        pts = (np.outer(np.cos(theta), b1) + np.outer(np.sin(theta), b2)) * sky_r
        ax.plot(pts[:,0], pts[:,1], pts[:,2], color='#8899dd', alpha=a, linewidth=0.5)


def orbit_plot_xy(r, t=None, title='', figsize=(12, 10), save_path=False,
                  frame="gcrf", show=False, c='black', pad=1,
                  show_stars=True, mag_limit=7.0,
                  show_milky_way=True, label_stars=False,
                  label_mag_limit=1.5, texture_bodies=True,
                  elev=60, azim=45, fov=360):
    from ..orbital_mechanics import (lagrange_points_lunar_frame,
                                     lagrange_points_lunar_fixed_frame)
    r, t = valid_orbits(r, t)

    dark_bg   = 'w' not in c
    textcolor = 'white' if dark_bg else 'black'
    plotcolor = '#000008' if dark_bg else 'white'
    show_stars = show_stars and dark_bg

    ssapy_data = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'SSAPy', 'ssapy', 'data'))
    earth_img = os.path.join(ssapy_data, 'earth.png')
    moon_img  = os.path.join(ssapy_data, 'moon.png')

    fig = plt.figure(dpi=120, figsize=figsize, facecolor=plotcolor)
    ax  = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_facecolor(plotcolor)
    fig.patch.set_facecolor(plotcolor)
    ax.view_init(elev=elev, azim=azim)
    ax.set_clip_on(False)
    # Hide box panes completely
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.set_axis_off()

    if any(np.max(np.linalg.norm(xyz, axis=-1)) >= 0.95 * RGEO for xyz in r):
        unit_conversion = RGEO
        unit_label = 'GEO'
    else:
        unit_conversion = 1e3
        unit_label = 'km'

    earth_r = EARTH_RADIUS / unit_conversion
    moon_r  = MOON_RADIUS  / unit_conversion
    mx, my, mz = 0.0, 0.0, 0.0
    title2  = "GCRF"
    fk      = "gcrf"
    xyz_list = []

    # ── Pass 1: compute bounds and collect orbit data ─────────────────────────
    raw_lower = np.array([ np.inf,  np.inf,  np.inf])
    raw_upper = np.array([-np.inf, -np.inf, -np.inf])

    for orbit_index in range(len(r)):
        xyz       = r[orbit_index]
        t_current = t[orbit_index]
        r_moon    = get_body("moon").position(t_current).T

        def get_main_category(fs):
            mapping = {
                "gcrf": "gcrf", "gcrs": "gcrf",
                "itrf": "itrf", "itrs": "itrf",
                "lunar": "lunar",
                "lunar_fixed": "lunar", "lunar fixed": "lunar",
                "lunar_centered": "lunar", "lunar centered": "lunar",
                "lunarearthfixed": "lunar axis",
                "lunarearth": "lunar axis",
                "lunar axis": "lunar axis",
                "lunar_axis": "lunar axis",
                "lunaraxis": "lunar axis",
            }
            return mapping.get(fs.lower())

        frame_transforms = {
            "gcrf":       ("GCRF",                None),
            "itrf":       ("ITRF",                gcrf_to_itrf),
            "lunar":      ("Lunar Frame",          gcrf_to_lunar_fixed),
            "lunar axis": ("Moon on x-axis Frame", gcrf_to_lunar),
        }

        fk = get_main_category(frame)
        if fk in frame_transforms:
            title2, tfunc = frame_transforms[fk]
            if tfunc:
                xyz    = tfunc(xyz,    t_current)
                r_moon = tfunc(r_moon, t_current)
        else:
            raise ValueError("Unknown frame. Options: gcrf, itrf, lunar, lunar fixed")

        xyz    = xyz    / unit_conversion
        r_moon = r_moon / unit_conversion

        raw_lower = np.minimum(raw_lower, xyz.min(axis=0))
        raw_upper = np.maximum(raw_upper, xyz.max(axis=0))

        moon_pos = np.atleast_1d(r_moon[:, 0])
        mx = float(moon_pos[-1]) if len(moon_pos) > 1 else float(r_moon[0, 0])
        moon_pos = np.atleast_1d(r_moon[:, 1])
        my = float(moon_pos[-1]) if len(moon_pos) > 1 else float(r_moon[0, 1])
        moon_pos = np.atleast_1d(r_moon[:, 2])
        mz = float(moon_pos[-1]) if len(moon_pos) > 1 else float(r_moon[0, 2])

        if orbit_index == 0:
            xyz_collected = xyz.copy()
        else:
            xyz_collected = np.vstack([xyz_collected, xyz])

    # Store the full collected orbit as one entry
        # Lagrange points
        if 'lunar' in fk:
            lpts = lagrange_points_lunar_frame().items()
            if 'fixed' in fk:
                lpts = lagrange_points_lunar_fixed_frame().items()
            for pt, pos in lpts:
                pos = pos / unit_conversion
                ax.scatter([pos[0]], [pos[1]], [pos[2]], color=textcolor, s=10)
                ax.text(pos[0], pos[1], pos[2], pt, color=textcolor, fontsize=7)


    # Store the full collected orbit as one entry
    xyz_list.append((0, xyz_collected))

    # ── Compute plot range ────────────────────────────────────────────────────
    plot_range = max(
        raw_upper[0] - raw_lower[0],
        raw_upper[1] - raw_lower[1],
        raw_upper[2] - raw_lower[2],
        earth_r * 4,
    )
    cx_ = (raw_lower[0] + raw_upper[0]) / 2
    cy_ = (raw_lower[1] + raw_upper[1]) / 2
    cz_ = (raw_lower[2] + raw_upper[2]) / 2

    ax.set_xlim(cx_ - plot_range/2, cx_ + plot_range/2)
    ax.set_ylim(cy_ - plot_range/2, cy_ + plot_range/2)
    ax.set_zlim(cz_ - plot_range/2, cz_ + plot_range/2)

    sky_radius = plot_range * 5.0
    visible_earth_r = max(earth_r, plot_range * 0.08)

    # ── RENDER ORDER: stars first, then orbit, then Earth ────────────────────

    # 1. Stars on celestial sphere (rendered first = furthest back)
    if show_stars:
        stars = _load_stars(mag_limit=mag_limit)
        if stars is not None:
            sx, sy, sz, ssizes, scolors, dots = \
                _filter_stars_fov(stars, elev, azim, fov, sky_radius)
            if len(sx) > 0:
                # Outer glow
                ax.scatter(sx, sy, sz, s=ssizes*5, c=scolors,
                           alpha=0.06, depthshade=False, linewidths=0, clip_on=False)

                ax.scatter(sx, sy, sz, s=ssizes*2, c=scolors,
                           alpha=0.15, depthshade=False, linewidths=0, clip_on=False)
                #bright core
                ax.scatter(sx, sy, sz, s=ssizes*0.6, c=scolors,
                           alpha=0.85, depthshade=False, linewidths=0, clip_on=False)
                print(f"[orbit_plot_xy] {len(sx)} stars rendered")

    # 2. Milky Way
    if show_stars and show_milky_way:
        _milky_way(ax, sky_radius)

    # 3. Orbit tracks
    for orbit_index, xyz in xyz_list:
        if len(r) == 1:
            colors_track = cm.plasma(np.linspace(0, 1, len(xyz)))
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                       c=colors_track, s=8, depthshade=False,
                       alpha=1.0, clip_on=False, zorder=100)
        else:
            tc = cm.rainbow(np.linspace(0, 1, len(r)))[orbit_index]
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    color=tc, linewidth=2.0, alpha=1.0, 
                    clip_on=False)

    # Orbital plane reference ring
    theta   = np.linspace(0, 2*np.pi, 200)
    orbit_r = np.mean([np.linalg.norm(xyz, axis=1).mean() for _, xyz in xyz_list])
    ax.plot(orbit_r*np.cos(theta), orbit_r*np.sin(theta), np.zeros(200),
            color='#334455', linewidth=0.5, alpha=0.4, clip_on=False)
    # 4. Earth (rendered last = on top)
    if texture_bodies:
        ok = _textured_sphere(ax, 0, 0, 0, visible_earth_r, earth_img)
        if not ok:
            u = np.linspace(0, 2*np.pi, 30)
            v = np.linspace(0, np.pi,   30)
            ax.plot_surface(
                visible_earth_r * np.outer(np.cos(u), np.sin(v)),
                visible_earth_r * np.outer(np.sin(u), np.sin(v)),
                visible_earth_r * np.outer(np.ones(30), np.cos(v)),
                color='steelblue', alpha=0.8)
        if fk in ("gcrf", "itrf"):
            _atmosphere(ax, visible_earth_r)

    # 5. Moon
    if texture_bodies:
        _textured_sphere(ax, mx, my, mz, moon_r, moon_img, n=32)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(f'{title}\nFrame: {title2}', color=textcolor,
                 fontsize=11, pad=15)

    if save_path:
        save_plot(fig, save_path)
    if show:
        plt.show()
    plt.close()
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT — called by the GUI via  python -m ssapy_toolkit.plots.orbit_plot_xy
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys, os, datetime
    import numpy as np
    from pathlib import Path

    # ── 1. Load GUI config ────────────────────────────────────────────────────
    cfg = {}
    cfg_path = os.environ.get("GUI_CONFIG", "")
    if cfg_path and Path(cfg_path).exists():
        exec(open(cfg_path, encoding="utf-8").read(), cfg)
        print(f"[orbit_plot_xy] Config loaded from {cfg_path}")
    else:
        # Sensible defaults for standalone testing
        cfg = dict(
            a_km=6928.0, e=0.001, inc_deg=51.6,
            raan_deg=0.0, argp_deg=0.0, nu_deg=0.0,
            epoch=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            n_orbits=3.0, dt_s=60.0,
            output_dir=str(Path.home() / "SSAPy-Toolkit" / "output"),
            show_stars=True, show_earth=True, show_moon=True,
        )
        print("[orbit_plot_xy] No GUI_CONFIG — using defaults")

    a_km      = float(cfg.get("a_km",      6928.0))
    e         = float(cfg.get("e",         0.001))
    inc_deg   = float(cfg.get("inc_deg",   51.6))
    raan_deg  = float(cfg.get("raan_deg",  0.0))
    argp_deg  = float(cfg.get("argp_deg",  0.0))
    nu_deg    = float(cfg.get("nu_deg",    0.0))
    epoch_str = str(cfg.get("epoch",       "2024-01-01 00:00:00"))
    n_orbits  = float(cfg.get("n_orbits",  3.0))
    dt_s      = float(cfg.get("dt_s",      60.0))
    output_dir = str(cfg.get("output_dir", "."))
    show_stars  = bool(cfg.get("show_stars",  True))
    show_earth  = bool(cfg.get("show_earth",  True))
    show_moon   = bool(cfg.get("show_moon",   True))

    os.makedirs(output_dir, exist_ok=True)

    # ── 2. Propagate ──────────────────────────────────────────────────────────
    try:
        import ssapy
        import astropy.time

        MU_M3S2 = 3.986004418e14   # m^3/s^2
        a_m     = a_km * 1e3
        T_s     = 2.0 * np.pi * np.sqrt(a_m**3 / MU_M3S2)
        n_steps = max(100, int(round(n_orbits * T_s / dt_s)))

        t0      = astropy.time.Time(epoch_str.strip(), format="iso", scale="utc")
        t_gps   = t0.gps + np.arange(n_steps) * dt_s

        orbit = ssapy.Orbit.fromKeplerianElements(
            a_m,
            e,
            np.radians(inc_deg),
            np.radians(argp_deg),
            np.radians(raan_deg),
            np.radians(nu_deg),
            t0.gps,
        )

        print(f"[orbit_plot_xy] Propagating {n_steps} steps "
              f"({n_orbits:.1f} orbits, T={T_s/60:.1f} min, dt={dt_s:.0f}s)…")

        orbits = orbit.at(t_gps)
        r_arr  = np.array([o.r for o in orbits])   # (N, 3) metres GCRF
        v_arr  = np.array([o.v for o in orbits])   # (N, 3) m/s
        t_arr    = astropy.time.Time(t_gps, format="gps", scale="utc")

        print(f"[orbit_plot_xy] Propagation complete — "
              f"{len(r_arr):,} points, "
              f"alt range {(np.linalg.norm(r_arr,axis=1).min()/1e3 - 6378.137):.0f}–"
              f"{(np.linalg.norm(r_arr,axis=1).max()/1e3 - 6378.137):.0f} km")

    except Exception as prop_err:
        print(f"[orbit_plot_xy] Propagation failed: {prop_err}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # ── 3. Render ─────────────────────────────────────────────────────────────
    save_path = str(Path(output_dir) / "orbit_plot_xy.jpg")
    try:
        orbit_plot_xy(
            r=[r_arr],
            t=[t_arr],
            title=f"a={a_km:.0f} km  e={e:.4f}  i={inc_deg:.1f}°",
            save_path=save_path,
            show_stars=show_stars,
            texture_bodies=show_earth,
            show=False,
        )
        print(f"[orbit_plot_xy] Saved → {save_path}")
    except Exception as plot_err:
        print(f"[orbit_plot_xy] Render failed: {plot_err}")
        import traceback; traceback.print_exc()
        sys.exit(1)