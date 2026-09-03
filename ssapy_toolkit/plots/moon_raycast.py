"""
moon_raycast.py — high-fidelity Moon figures from real lunar data
======================================================================
Drop into:
  ~/SSAPy-Toolkit/ssapy_toolkit/plots/moon_raycast.py

Data, both living in SSAPy-Data and found through the toolkit's own
discovery (starfield.find_data_file):
  moon.png       — lunar photomosaic, already there
  moon_dem.npz   — LOLA topography, built once by
                   SSAPy-Toolkit/scripts/fetch_lola_dem.py

Why this exists alongside moon_render.py
----------------------------------------
Plotly colours a 3D surface per VERTEX, so visible detail is capped by the
mesh rather than the data. At 540x1080 that is one colour sample per ~10 km
of lunar surface, and both the 13.7 MB albedo map and the LOLA topography
are thrown away to meet it. For a still figure that limit is unnecessary:
this ray-casts the sphere per PIXEL, so every pixel gets its own surface
point, albedo sample, slope and shadow test from full-resolution data.

The trade is interactivity — this makes an image, not a scene you can spin.
moon_render.py stays for the interactive viewer.

WHAT IS PHYSICAL HERE
  * topography and slopes: LOLA, differentiated at its own resolution
  * cast shadows: ray-marched against that topography along the true Sun
    direction, in 3D so lunar curvature is exact. Shadowed points get no
    direct light and no ambient fill, because the Moon has no atmosphere
  * albedo: the real photomosaic, oriented by physical checks rather than an
    assumed file convention
  * photometry: Lommel-Seeliger with an opposition surge, computed from the
    real incidence, emission and phase angles

WHAT IS NOT
  * slope_gain defaults to 1.0, meaning true slopes. Anything higher
    exaggerates every shadow by that factor: decoration, not data
  * exposure is a display gain; nothing is calibrated in radiance
  * detail below the DEM's pixel scale is absent (~15 km/px at 8
    pixels/degree, ~7 km/px if fetched with --full)
  * Lommel-Seeliger is first-order for regolith. Full Hapke would add
    single-scattering albedo, a phase function and macroscopic roughness
    (~20-25 deg for the Moon), which matters most at large phase
  * the opposition surge uses representative parameters (B0 = 1, h = 0.06)
  * no multiple scattering, so shadowed floors go black rather than very dark

Usage
-----
    from ssapy_toolkit.plots.moon_raycast import moon_figure

    moon_figure(save_path="~/ssatk_figures/moon.png")

    moon_figure(orbit_alt_km=100, orbit_inc_deg=90,
                save_path="~/ssatk_figures/moon_llo.png")
"""

from __future__ import annotations

import os
import time

import numpy as np

R_MOON_KM = 1737.4


# ----------------------------------------------------------------------
# data discovery and orientation
# ----------------------------------------------------------------------
def _find(name, ext):
    """
    Locate a data asset.

    starfield.py already owns SSAPy-Data discovery for this package -- it
    searches $SSAPY_DATA, sibling SSAPy-Data checkouts (both the repo root and
    the packaged src/ssapy_data/data layout) and the installed ssapy_data
    package. Use it, so assets in SSAPy-Data are found the same way here as
    everywhere else in the toolkit.

    ssapy.utils.find_file is tried afterwards, but note it searches SSAPy's
    OWN bundled data directory, not SSAPy-Data -- which is why moon_dem.npz
    placed in SSAPy-Data is invisible to it.
    """
    filename = name + ext
    try:
        try:
            from .starfield import find_data_file
        except ImportError:
            from ssapy_toolkit.plots.starfield import find_data_file
        hit = find_data_file(filename)
        if hit is not None:
            return str(hit)
    except Exception:
        pass

    try:
        from ssapy.utils import find_file
        p = find_file(name, ext=ext)
        if p and os.path.exists(p):
            return p
    except Exception:
        pass

    if "dem" in name:
        for env in ("SSAPY_MOON_DEM", "MOON_DEM"):
            p = os.environ.get(env)
            if p and os.path.exists(p):
                return p
    return None


def _orient_lon(arr, is_dem, label):
    """
    Lunar maps ship with 0 deg longitude either at the image centre or at its
    edge. Guessing wrong rotates the Moon 180 deg and silently renders the
    farside, so check against physics instead of assuming a convention.

    Albedo: the Moon is tidally locked, sub-Earth point 0N 0E, and the
    nearside maria make that hemisphere markedly darker than the farside.

    Topography: the centre of figure is offset about 1.9 km toward Earth, so
    the farside crust stands higher.

    Arrays here run lon -180 -> +180 across the columns, so lon 0 is the
    MIDDLE and lon 180 sits at both edges.
    """
    W = arr.shape[1]
    f = arr if arr.ndim == 2 else arr.mean(axis=2)
    near = f[:, W // 4: 3 * W // 4].mean()
    far = (f[:, : W // 4].mean() + f[:, -W // 4:].mean()) / 2.0
    if near > far:
        what = "stood higher" if is_dem else "was brighter"
        print(f"[moon_raycast] {label}: rolling 180 deg — the lon 0 hemisphere "
              f"{what} ({near:.3f}) than the lon 180 one ({far:.3f}), which is "
              f"the farside")
        return np.roll(arr, W // 2, axis=1)
    print(f"[moon_raycast] {label}: orientation OK ({near:.3f} vs {far:.3f})")
    return arr


def load_moon_data(dem_path=None, albedo_path=None):
    """Return (dem_km, albedo_rgb), both oriented."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    dem_path = dem_path or _find("moon_dem", ".npz")
    if dem_path is None:
        raise FileNotFoundError(
            "moon_dem.npz not found. Build it once with:\n"
            "    python scripts/fetch_lola_dem.py "
            "--out ../SSAPy-Data/src/ssapy_data/data/moon_dem.npz")
    with np.load(os.path.expanduser(dem_path)) as z:
        key = "elev_km" if "elev_km" in z else z.files[0]
        dem = np.asarray(z[key], dtype=np.float64)
    print(f"[moon_raycast] topography {dem.shape[1]}x{dem.shape[0]}, "
          f"{dem.min():.1f} to {dem.max():.1f} km")

    albedo_path = albedo_path or _find("moon", ".png")
    if albedo_path is None:
        raise FileNotFoundError("moon.png not found in SSAPy-Data")
    alb = np.asarray(Image.open(os.path.expanduser(albedo_path)).convert("RGB"),
                     dtype=np.float64) / 255.0
    print(f"[moon_raycast] albedo {alb.shape[1]}x{alb.shape[0]}")

    return _orient_lon(dem, True, "topography"), _orient_lon(alb, False, "albedo")


# ----------------------------------------------------------------------
# renderer
# ----------------------------------------------------------------------
def render(dem, albedo, sun, view, px=1500, up_hint=(0, 0, 1.0),
           shadows=True, max_km=260.0, step_km=1.5, exposure=1.9,
           slope_gain=1.0, supersample=2):
    """Orthographic per-pixel render of the lunar disc. Returns (img, camera)."""
    t0 = time.time()
    n = px * supersample
    w = np.asarray(view, float); w /= np.linalg.norm(w)
    up = np.asarray(up_hint, float)
    right = np.cross(up, w)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(np.array([0, 1.0, 0]), w)
    right /= np.linalg.norm(right)
    up = np.cross(w, right)

    lim = R_MOON_KM * 1.02
    xs = np.linspace(-lim, lim, n)
    U, V = np.meshgrid(xs, xs)
    rho2 = U ** 2 + V ** 2
    hit = rho2 <= R_MOON_KM ** 2

    depth = np.zeros_like(U)
    depth[hit] = np.sqrt(R_MOON_KM ** 2 - rho2[hit])
    P = U[..., None] * right + V[..., None] * up + depth[..., None] * w
    r = np.linalg.norm(P, axis=-1)
    good = hit & (r > 0)
    Pn = np.zeros_like(P)
    Pn[good] = P[good] / r[good][..., None]

    lat = np.degrees(np.arcsin(np.clip(Pn[..., 2], -1, 1)))
    lon = np.degrees(np.arctan2(Pn[..., 1], Pn[..., 0]))

    H, W = dem.shape
    row = np.clip(((90.0 - lat) / 180.0 * (H - 1)).astype(np.int32), 0, H - 1)
    col = (((lon + 180.0) / 360.0 * W).astype(np.int32)) % W

    latn = np.radians(np.linspace(90, -90, H))[:, None]
    s_lat_f = np.gradient(dem, axis=0) / (np.pi / H) / R_MOON_KM
    s_lon_f = (np.gradient(dem, axis=1) / (2 * np.pi / W)
               / (R_MOON_KM * np.maximum(np.cos(latn), 1e-3)))
    s_lat = s_lat_f[row, col] * slope_gain
    s_lon = s_lon_f[row, col] * slope_gain

    latr, lonr = np.radians(lat), np.radians(lon)
    e_lat = np.stack([-np.sin(latr) * np.cos(lonr),
                      -np.sin(latr) * np.sin(lonr), np.cos(latr)], -1)
    e_lon = np.stack([-np.sin(lonr), np.cos(lonr), np.zeros_like(lonr)], -1)
    N = Pn - s_lat[..., None] * e_lat - s_lon[..., None] * e_lon
    nn = np.linalg.norm(N, axis=-1, keepdims=True); nn[nn == 0] = 1
    N /= nn

    s = np.asarray(sun, float); s /= np.linalg.norm(s)
    mu0 = np.clip((N * s).sum(-1), 0, 1)
    mu = np.clip((N * w).sum(-1), 0, 1)

    if shadows:
        surf = Pn * (R_MOON_KM + dem[row, col])[..., None]
        lit = good & (mu0 > 1e-4)
        idx = np.argwhere(lit)
        pts = surf[lit]
        ts = np.arange(step_km, max_km + step_km, step_km)
        blocked = np.zeros(len(pts), dtype=bool)
        for a in range(0, len(pts), 30000):
            b = min(a + 30000, len(pts))
            Q = pts[a:b, None, :] + s[None, None, :] * ts[None, :, None]
            rq = np.linalg.norm(Q, axis=2)
            la = np.degrees(np.arcsin(np.clip(Q[:, :, 2] / rq, -1, 1)))
            lo = np.degrees(np.arctan2(Q[:, :, 1], Q[:, :, 0]))
            rr = np.clip(((90.0 - la) / 180.0 * (H - 1)).astype(np.int32), 0, H - 1)
            cc = (((lo + 180.0) / 360.0 * W).astype(np.int32)) % W
            blocked[a:b] = np.any(rq < (R_MOON_KM + dem[rr, cc]) - 0.05, axis=1)
        sh = np.zeros_like(mu0, dtype=bool)
        sh[idx[:, 0], idx[:, 1]] = blocked
        mu0 = np.where(sh, 0.0, mu0)
        print(f"[moon_raycast] cast shadows on {100.0 * blocked.mean():.1f}% "
              f"of the sunlit face")

    ls = np.where(mu0 + mu > 0, mu0 / (mu0 + mu + 1e-9), 0.0)
    g = float(np.arccos(np.clip(np.dot(s, w), -1, 1)))
    B = 1.0 / (1.0 + np.tan(min(g, np.pi / 2 - 1e-3) / 2) / 0.06)
    shade = ls * (1 + B) / (1 + B * 0.5)

    ha, wa = albedo.shape[:2]
    ar = np.clip(((90.0 - lat) / 180.0 * (ha - 1)).astype(np.int32), 0, ha - 1)
    ac = (((lon + 180.0) / 360.0 * wa).astype(np.int32)) % wa
    alb = albedo[ar, ac]
    if alb.ndim == 2:
        alb = alb[..., None] * np.array([1.0, 0.99, 0.97])

    img = np.clip(alb * shade[..., None] * exposure, 0, 1)
    img[~good] = 0.0
    if supersample > 1:
        img = img.reshape(px, supersample, px, supersample, 3).mean(axis=(1, 3))
    print(f"[moon_raycast] {px}x{px} at phase {np.degrees(g):.0f} deg, "
          f"{time.time() - t0:.0f}s")
    return img, dict(right=right, up=up, forward=w, px=px, lim=lim)


def _draw_orbit(img, cam, alt_km, inc_deg, raan_deg,
                colour=(1.0, 0.72, 0.20), n=6000):
    """Overlay a circular orbit, hidden where it passes behind the Moon."""
    a = R_MOON_KM + alt_km
    th = np.linspace(0, 2 * np.pi, n)
    inc, raan = np.radians(inc_deg), np.radians(raan_deg)
    xo = a * np.cos(th)
    yo = a * np.sin(th) * np.cos(inc)
    zo = a * np.sin(th) * np.sin(inc)
    Q = np.stack([xo * np.cos(raan) - yo * np.sin(raan),
                  xo * np.sin(raan) + yo * np.cos(raan), zo], -1)
    u, v, d = Q @ cam["right"], Q @ cam["up"], Q @ cam["forward"]
    rho = np.hypot(u, v)
    front = np.sqrt(np.clip(R_MOON_KM ** 2 - rho ** 2, 0, None))
    vis = (rho > R_MOON_KM) | (d > front)      # occluded behind the disc
    px, lim = cam["px"], cam["lim"]
    iu = ((u + lim) / (2 * lim) * (px - 1)).astype(int)
    iv = ((lim - v) / (2 * lim) * (px - 1)).astype(int)
    ok = vis & (iu >= 0) & (iu < px) & (iv >= 0) & (iv < px)
    c = np.asarray(colour, float)
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            uu = np.clip(iu[ok] + du, 0, px - 1)
            vv = np.clip(iv[ok] + dv, 0, px - 1)
            wgt = 1.0 if (du == 0 and dv == 0) else 0.35
            img[vv, uu] = np.clip(img[vv, uu] * (1 - wgt) + c * wgt, 0, 1)
    return img


def moon_figure(phase_deg=55.0, view=(1.0, 0.0, 0.12), px=1500, supersample=2,
                orbit_alt_km=None, orbit_inc_deg=90.0, orbit_raan_deg=15.0,
                dem_path=None, albedo_path=None, save_path=None, **kw):
    """
    One call: find the data, derive the Sun direction from a phase angle,
    render, optionally overlay an orbit, optionally save.

    phase_deg : Sun-Moon-observer angle. 0 is full and looks flat; 55 gives a
                waxing gibbous with the terminator down the limb, which shows
                relief best; past about 85 there is little disc left.
    """
    dem, alb = load_moon_data(dem_path, albedo_path)
    v = np.asarray(view, float); v /= np.linalg.norm(v)
    a = np.radians(phase_deg)
    sun = np.array([np.cos(a) * v[0] - np.sin(a) * v[1],
                    np.sin(a) * v[0] + np.cos(a) * v[1], v[2]])
    sun /= np.linalg.norm(sun)

    img, cam = render(dem, alb, sun, v, px=px, supersample=supersample, **kw)
    if orbit_alt_km is not None:
        img = _draw_orbit(img, cam, orbit_alt_km, orbit_inc_deg, orbit_raan_deg)

    if save_path:
        from PIL import Image
        save_path = os.path.expanduser(save_path)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        Image.fromarray((img * 255).astype(np.uint8)).save(save_path)
        print(f"[moon_raycast] saved -> {save_path}")
    return img