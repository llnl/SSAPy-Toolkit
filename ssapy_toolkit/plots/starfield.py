"""
starfield.py — Shared star background module for SSAPy-Toolkit 3D plots
========================================================================
Drop into: ~/SSAPy-Toolkit/ssapy_toolkit/plots/starfield.py

Usage in any 3D plot file:
    
"""

import os
import numpy as np

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


def _precession_matrix(epoch_jd: float) -> np.ndarray:
    """IAU 1976 precession rotation matrix from J2000.0 to the given Julian date.

    Shifts J2000.0 catalog positions (ICRS) to the mean equator-of-date
    so the star directions are consistent with the plot's GCRF-aligned
    coordinate frame at the actual mission epoch.

    Shift magnitude: ~50 arcseconds/year.  By 2022 (Artemis I) that is
    ~1143" (~3 matplotlib pixels), and by 2026 (Artemis II) ~1395" --
    large enough to be visible in the rendered star sphere, especially for
    Polaris which should sit exactly on the +Z axis but drifts ~450-550"
    off it without this correction.

    Parameters
    ----------
    epoch_jd : float
        Target Julian date.

    Returns
    -------
    P : ndarray, shape (3, 3)
        Rotation matrix to apply to J2000 unit vectors.
    """
    T = (epoch_jd - 2_451_545.0) / 36_525.0   # Julian centuries from J2000
    # IAU 1976 precession angles, arcseconds
    zeta  = ((2306.2181 + 1.39656*T - 0.000139*T**2)*T
             + (0.30188 - 0.000344*T)*T**2 + 0.017998*T**3)
    z     = ((2306.2181 + 1.39656*T - 0.000139*T**2)*T
             + (1.09468 + 0.000066*T)*T**2 + 0.018203*T**3)
    theta = ((2004.3109 - 0.85330*T - 0.000217*T**2)*T
             - (0.42665 + 0.000217*T)*T**2 - 0.041775*T**3)

    zeta_r  = np.radians(zeta  / 3600.0)
    z_r     = np.radians(z     / 3600.0)
    theta_r = np.radians(theta / 3600.0)

    Rz_zeta  = np.array([[np.cos(zeta_r),  np.sin(zeta_r), 0.0],
                          [-np.sin(zeta_r), np.cos(zeta_r), 0.0],
                          [0.0, 0.0, 1.0]])
    Ry_theta = np.array([[np.cos(theta_r),  0.0, -np.sin(theta_r)],
                          [0.0,              1.0,  0.0],
                          [np.sin(theta_r),  0.0,  np.cos(theta_r)]])
    Rz_z     = np.array([[np.cos(z_r),  np.sin(z_r), 0.0],
                          [-np.sin(z_r), np.cos(z_r), 0.0],
                          [0.0, 0.0, 1.0]])

    return Rz_z @ Ry_theta @ Rz_zeta


def _load_stars(mag_limit=7.0, epoch_jd: float = None):
    """Load and cache the HYG star catalog, optionally precessed to epoch_jd."""
    global _STAR_CACHE
    cache_key = (mag_limit, epoch_jd)
    if _STAR_CACHE is not None and _STAR_CACHE.get('_key') == cache_key:
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

        # Unit vectors on celestial sphere (GCRF-aligned, J2000.0)
        cx = np.cos(dec_rad) * np.cos(ra_rad)
        cy = np.cos(dec_rad) * np.sin(ra_rad)
        cz = np.sin(dec_rad)

        # Apply IAU 1976 precession if an epoch was supplied.
        # Shifts catalog J2000 positions to the mean equator of the
        # mission date so the stars land in the correct GCRF directions
        # for that specific day (~3px shift at 2022, ~3.5px at 2026).
        if epoch_jd is not None:
            P = _precession_matrix(epoch_jd)
            vecs = P @ np.stack([cx, cy, cz])   # (3, N)
            cx, cy, cz = vecs[0], vecs[1], vecs[2]

        sizes  = np.clip(0.5 * (mag_limit - mag) ** 1.1, 0.05, 4.0)
        spect  = df['spect'].fillna('G').str[:1].values
        colors = np.array([_SPECT_COLORS.get(s, _SPECT_COLORS['G']) for s in spect])

        _STAR_CACHE = {
            'cx': cx, 'cy': cy, 'cz': cz,
            'mag': mag, 'sizes': sizes,
            'colors': colors,
            'n': len(mag), 'mag_limit': mag_limit,
            '_key': cache_key,
        }
        return _STAR_CACHE
    except Exception as e:
        print(f"[starfield] Could not load catalog: {e}")
        return None


def _camera_direction(elev_deg, azim_deg):
    """Unit vector from camera toward origin."""
    e = np.radians(elev_deg)
    a = np.radians(azim_deg)
    cam = np.array([np.cos(e)*np.cos(a),
                    np.cos(e)*np.sin(a),
                    np.sin(e)])
    look = -cam
    return look / np.linalg.norm(look)


def add_starfield(ax, plot_range, elev=30, azim=45,
                  fov=360, mag_limit=6.5,
                  show_milky_way=True,
                  epoch=None):
    """
    Add a realistic star background to a matplotlib 3D axes.

    Parameters
    ----------
    ax         : Axes3D  — the 3D axes to draw on
    plot_range : float   — size of the scene (sets star sphere radius)
    elev       : float   — camera elevation in degrees
    azim       : float   — camera azimuth in degrees
    fov        : float   — field of view in degrees (360 = all stars)
    mag_limit  : float   — magnitude cutoff (7.0 = deeper than naked eye)
    show_milky_way : bool — draw Milky Way band
    epoch      : float, astropy.time.Time, or None
        If provided, star J2000 catalog positions are precessed (IAU 1976)
        to the given epoch so they land in the correct GCRF directions for
        that specific mission date.

        Accepted formats:
          - float  : GPS seconds since 1980-01-06 (same convention used
                     elsewhere in this toolkit, e.g. OEM t_gps arrays)
          - astropy.time.Time : any time object
          - None   : J2000.0 positions used as-is (original behavior)

    Returns
    -------
    None
    """
    # Convert epoch to Julian Date for the precession calculation
    epoch_jd = None
    if epoch is not None:
        try:
            # astropy Time object
            epoch_jd = float(epoch.jd)
        except AttributeError:
            # GPS seconds — may be a plain float, 0-d array, or 1-element array
            GPS_JD_EPOCH = 2_444_244.5
            epoch_jd = GPS_JD_EPOCH + (float(np.asarray(epoch).flat[0]) - 18.0) / 86400.0

    stars = _load_stars(mag_limit=mag_limit, epoch_jd=epoch_jd)
    if stars is None:
        return

    sky_radius = plot_range * 4.0

    # Filter stars by FOV
    look      = _camera_direction(elev, azim)
    star_dirs = np.stack([stars['cx'], stars['cy'], stars['cz']], axis=1)
    dots      = star_dirs @ (-look)
    half_cos  = np.cos(np.radians(fov / 2.0))
    mask      = dots > half_cos

    # Depth variation — brighter stars closer, fainter further
    mag     = stars['mag'][mask]
    mag_min = mag.min() if len(mag) > 0 else 0
    mag_max = mag.max() if len(mag) > 0 else 1
    depth   = 0.5 + 0.5 * (mag - mag_min) / (mag_max - mag_min + 1e-6)

    x = stars['cx'][mask] * sky_radius * depth
    y = stars['cy'][mask] * sky_radius * depth
    z = stars['cz'][mask] * sky_radius * depth

    sizes  = stars['sizes'][mask]
    colors = stars['colors'][mask]

    if len(x) == 0:
        return

    # Airy disk rendering — outer glow, mid glow, bright core
    ax.scatter(x, y, z, s=sizes*2.0, c=colors,
               alpha=0.75, depthshade=False,
               linewidths=0)

    # Milky Way band on celestial sphere
    if show_milky_way:
        _add_milky_way(ax, sky_radius)


def _add_milky_way(ax, sky_radius):
    """Draw Milky Way as great circle arcs on the celestial sphere."""
    gnp = np.array([
        np.cos(np.radians(27.13)) * np.cos(np.radians(192.85)),
        np.cos(np.radians(27.13)) * np.sin(np.radians(192.85)),
        np.sin(np.radians(27.13))
    ])
    arb = np.array([0., 1., 0.])
    v1  = np.cross(gnp, arb); v1 /= np.linalg.norm(v1)
    theta = np.linspace(0, 2*np.pi, 60)

    for w, a in [(0., 0.08), (0.1, 0.05), (0.2, 0.03),
                 (-0.1, 0.05), (-0.2, 0.03)]:
        n_  = gnp + w * v1
        n_ /= np.linalg.norm(n_)
        b1  = np.cross(n_, arb)
        if np.linalg.norm(b1) < 1e-6:
            b1 = np.cross(n_, np.array([1, 0, 0]))
        b1 /= np.linalg.norm(b1)
        b2  = np.cross(n_, b1); b2 /= np.linalg.norm(b2)
        pts = (np.outer(np.cos(theta), b1) +
               np.outer(np.sin(theta), b2)) * sky_radius
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color='#8899dd', alpha=a,
                linewidth=0.5)