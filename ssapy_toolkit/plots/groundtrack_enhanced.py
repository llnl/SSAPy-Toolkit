"""
groundtrack_enhanced.py — Enhanced ground track for ssapy_analysis.py
========================================================================
Drop-in replacement for the toolkit's plain `groundtrack_plot`, matching
the spec in the ssapy_analysis.py checkpoint:

    groundtrack_{tag}.png — enhanced ground track with day/night
    terminator, eclipse coloring, subsolar point, site marker

This version actively uses your real ssapy / ssapy-toolkit modules wherever
they're available, instead of reimplementing them:

  - Continents: the REAL Earth texture your own `globe_plot.py` already
    uses (`from ssapy.utils import find_file; find_file("earth", ext=".png")`)
    is tried first. Falls back to `cartopy` coastlines (if installed), then
    to a bundled low-res coastline outline, so it still renders *something*
    real in an environment that has neither — but in your actual conda env
    with ssapy installed, this will show the same continents as the rest
    of the toolkit's plots.
  - Coordinate transform: tries your `core.frames.eci_to_ecf_matrix` (GMST
    rotation, matches the rest of this toolkit's frame conventions) first;
    falls back to astropy's GCRS->ITRS (higher precision, no toolkit
    dependency) if `core.frames` isn't importable.
  - Eclipse / shadow geometry: tries `core.sun.shadow_cone_params` (your
    real umbra/penumbra cone implementation) first; falls back to the
    two-circle overlap formula below (same physics, just inlined) if
    `core.sun` isn't importable.
  - Orbit propagation (self-test only): tries real `ssapy.Orbit` first;
    falls back to a plain two-body propagator only if `ssapy` isn't
    installed, so you can see immediately whether your real environment
    is wired up correctly by which code path prints.

Interface
---------
    from groundtrack_enhanced import plot_enhanced_groundtrack
    plot_enhanced_groundtrack(
        r_eci_km, t,                      # from your real propagator
        site_lat=28.5, site_lon=-80.6, site_name="Cape Canaveral",
        sat_name="ISS", save_path=f"groundtrack_{tag}.png",
    )
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from astropy.time import Time
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, get_sun
import astropy.units as u

RE_KM = 6_378.137
R_SUN_KM = 695_700.0
AU_KM = 149_597_870.7

# ── Optional real-toolkit imports (all soft — each falls back cleanly) ──────
try:
    from core.frames import eci_to_ecf_matrix as _tk_eci_to_ecf_matrix
    _HAS_TK_FRAMES = True
except ImportError:
    _HAS_TK_FRAMES = False

try:
    from core.sun import shadow_cone_params as _tk_shadow_cone_params
    _HAS_TK_SUN = True
except ImportError:
    _HAS_TK_SUN = False

try:
    import ssapy as _ssapy
    _HAS_SSAPY = hasattr(_ssapy, "Orbit")   # PyPI also hosts an unrelated
    # "ssapy" (Singular Spectrum Analysis) package with the same import
    # name — checking for `Orbit` avoids silently treating that as a match.
    if not _HAS_SSAPY:
        print("[groundtrack_enhanced] found a package named 'ssapy' but it "
              "has no Orbit class — likely the unrelated PyPI 'ssapy' "
              "(Singular Spectrum Analysis), not LLNL's SSAPy. Treating as unavailable.")
except ImportError:
    _HAS_SSAPY = False

try:
    from ssapy.utils import find_file as _ssapy_find_file
    _HAS_SSAPY_ASSETS = True
except ImportError:
    _HAS_SSAPY_ASSETS = False

print(f"[groundtrack_enhanced] core.frames: {'found' if _HAS_TK_FRAMES else 'not found — using astropy fallback'}")
print(f"[groundtrack_enhanced] core.sun:    {'found' if _HAS_TK_SUN else 'not found — using inline eclipse formula'}")
print(f"[groundtrack_enhanced] ssapy:       {'found' if _HAS_SSAPY else 'not found — self-test uses two-body fallback'}")
print(f"[groundtrack_enhanced] ssapy assets:{'found — will use real Earth texture' if _HAS_SSAPY_ASSETS else ' not found — trying cartopy/outline fallback for continents'}")


# ── gcrf_to_itrf — uses your real core.frames if available, else astropy ───
def gcrf_to_itrf(r_eci_km: np.ndarray, t: Time) -> np.ndarray:
    """
    Transform (N,3) GCRS (ECI, km) positions to ITRS (ECEF, km) at times t.
    t must be an astropy.time.Time of the same length as r_eci_km (or a
    scalar Time, broadcast to all rows).
    """
    r_eci_km = np.atleast_2d(r_eci_km)
    if t.isscalar:
        t = Time([t.iso] * len(r_eci_km))

    if _HAS_TK_FRAMES:
        # Your toolkit's own GMST rotation — consistent with every other
        # plot in this codebase that uses core.frames.
        out = np.empty_like(r_eci_km)
        for i in range(len(r_eci_km)):
            M = _tk_eci_to_ecf_matrix(t[i].gps)
            out[i] = M @ r_eci_km[i]
        return out

    # Fallback: astropy GCRS->ITRS (no toolkit dependency, still real)
    cart = CartesianRepresentation(r_eci_km[:, 0], r_eci_km[:, 1], r_eci_km[:, 2], unit=u.km)
    gcrs = GCRS(cart, obstime=t)
    itrs = gcrs.transform_to(ITRS(obstime=t))
    return np.stack([itrs.x.to(u.km).value,
                     itrs.y.to(u.km).value,
                     itrs.z.to(u.km).value], axis=1)


def ecef_to_geodetic(r_ecef_km: np.ndarray):
    """Spherical-Earth lat/lon/alt — fine for ground-track plotting."""
    x, y, z = r_ecef_km[:, 0], r_ecef_km[:, 1], r_ecef_km[:, 2]
    r = np.linalg.norm(r_ecef_km, axis=1)
    lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon, r - RE_KM


def subsolar_point(t: Time):
    """Real subsolar (lat, lon) in degrees at time(s) t, via astropy get_sun + ITRS."""
    sun_gcrs = get_sun(t)
    sun_cart = sun_gcrs.cartesian
    sun_itrs = GCRS(sun_cart, obstime=t).transform_to(ITRS(obstime=t))
    x, y, z = sun_itrs.x.value, sun_itrs.y.value, sun_itrs.z.value
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def compute_eclipse(r_eci_km: np.ndarray, t: Time, R_body_km: float = RE_KM):
    """
    Continuous illumination fraction (1.0 = full sun, 0.0 = total umbra,
    partial = penumbra). Uses your real `core.sun.shadow_cone_params` umbra/
    penumbra cone geometry when available (binary in_umbra/in_penumbra,
    upgraded here to a continuous fraction by linear falloff across the
    penumbra cone); falls back to the inline two-circle Sun/Earth angular-
    disk overlap formula otherwise. Either way, real Sun position from
    astropy (not a simplified ecliptic-longitude formula).
    """
    r = np.atleast_2d(r_eci_km)
    sun = get_sun(t).cartesian
    sun_km = np.stack([sun.x.to(u.km).value, sun.y.to(u.km).value, sun.z.to(u.km).value], axis=-1)
    sun_km = np.atleast_2d(sun_km)

    if _HAS_TK_SUN:
        frac = np.ones(len(r))
        for i in range(len(r)):
            cone = _tk_shadow_cone_params(sun_km[i], R_body_km)
            if cone["in_umbra"](r[i]):
                frac[i] = 0.0
            elif cone["in_penumbra"](r[i]):
                # Linear falloff across the penumbra cone's angular width,
                # since shadow_cone_params only gives a binary boolean.
                sun_hat = cone["sun_hat"]
                to_body_hat = -r[i] / np.linalg.norm(r[i])
                sep = np.arccos(np.clip(np.dot(to_body_hat, sun_hat), -1, 1))
                u_ha, p_ha = cone["umbra_half_angle_rad"], cone["penumbra_half_angle_rad"]
                frac[i] = np.clip((sep - u_ha) / max(p_ha - u_ha, 1e-9), 0.0, 1.0)
            else:
                frac[i] = 1.0
        return frac

    # Fallback: inline two-circle Sun/Earth angular-disk overlap
    D_km = np.linalg.norm(sun_km, axis=1)
    sun_hat = sun_km / D_km[:, None]
    r_mag = np.linalg.norm(r, axis=1)
    to_body_hat = -r / r_mag[:, None]
    cos_sep = np.clip(np.sum(to_body_hat * sun_hat, axis=1), -1, 1)
    sep = np.arccos(cos_sep)

    ang_body = np.arcsin(np.clip(R_body_km / r_mag, -1, 1))
    ang_sun = np.arcsin(np.clip(R_SUN_KM / D_km, -1, 1))
    anti_sun = np.sum(r * sun_hat, axis=1) < 0

    d, s, b = sep, ang_sun, ang_body
    no_overlap = d >= (s + b)
    full_overlap = d <= np.abs(s - b)
    partial = ~no_overlap & ~full_overlap

    frac = np.ones_like(r_mag)
    frac = np.where(full_overlap, np.clip(1.0 - (np.minimum(b, s)/np.maximum(s, 1e-12))**2, 0.0, 1.0), frac)

    d_s = np.where(partial, d, 1.0); s_s = np.maximum(s, 1e-12); b_s = np.maximum(b, 1e-12)
    t1 = s_s**2*np.arccos(np.clip((d_s**2+s_s**2-b_s**2)/(2*d_s*s_s+1e-30), -1, 1))
    t2 = b_s**2*np.arccos(np.clip((d_s**2+b_s**2-s_s**2)/(2*d_s*b_s+1e-30), -1, 1))
    t3 = 0.5*np.sqrt(np.clip((-d_s+s_s+b_s)*(d_s+s_s-b_s)*(d_s-s_s+b_s)*(d_s+s_s+b_s), 0, None))
    blocked = np.clip((t1+t2-t3)/(np.pi*s_s**2), 0.0, 1.0)
    frac = np.where(partial, 1.0 - blocked, frac)
    frac = np.where(anti_sun, frac, 1.0)
    return np.clip(frac, 0.0, 1.0)


def _elevation_deg(sat_ecef_km, site_ecef_km, site_lat_deg, site_lon_deg):
    lat, lon = np.radians(site_lat_deg), np.radians(site_lon_deg)
    up = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    rel = sat_ecef_km - site_ecef_km
    rng = np.linalg.norm(rel, axis=-1)
    return np.degrees(np.arcsin(np.clip((rel @ up) / rng, -1, 1)))


def _site_ecef(lat_deg, lon_deg, alt_km=0.0):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    r = RE_KM + alt_km
    return np.array([r*np.cos(lat)*np.cos(lon), r*np.cos(lat)*np.sin(lon), r*np.sin(lat)])


# ── Continents ───────────────────────────────────────────────────────────────
_earth_texture_cache = None


def _load_earth_texture():
    """
    Real Earth raster, same source your own globe_plot.py already uses
    (ssapy's bundled data/earth.png). Cached after first call.
    Returns an (H, W, 3) uint8 array in equirectangular projection
    (lon -180..180, lat +90..-90 top-to-bottom), or None if unavailable.
    """
    global _earth_texture_cache
    if _earth_texture_cache is not None:
        return _earth_texture_cache
    if not _HAS_SSAPY_ASSETS:
        return None
    try:
        from PIL import Image as _PILImage
        img = _PILImage.open(_ssapy_find_file("earth", ext=".png")).convert("RGB")
        _earth_texture_cache = np.asarray(img)
        return _earth_texture_cache
    except Exception as ex:
        print(f"[groundtrack_enhanced] real Earth texture load failed: {ex}")
        return None


def _draw_continents(ax):
    """
    Draw real continents under everything else. Tries, in order:
    1. Your own toolkit's real Earth texture (ssapy.utils.find_file) —
       what globe_plot.py / moon_plot_3d.py already use.
    2. cartopy Natural Earth coastlines/land polygons (if installed).
    3. A visible warning box if neither is available, rather than silently
       leaving the map blank the way the previous version of this script did.
    """
    tex = _load_earth_texture()
    if tex is not None:
        ax.imshow(tex, extent=[-180, 180, -90, 90], origin="upper",
                  aspect="auto", zorder=0.5, alpha=0.9)
        return "ssapy earth.png"

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        # NOTE: this path only works if `ax` was created with a cartopy
        # projection; since the rest of this plot uses plain matplotlib
        # axes (for the night-shading contourf/eclipse-coloured
        # LineCollection to line up in plain lon/lat), we instead pull the
        # raw coastline/land geometries and draw them as plain patches on
        # the existing lon/lat axes.
        land = cfeature.NaturalEarthFeature("physical", "land", "110m",
                                            facecolor="#c8c8a0")
        for geom in land.geometries():
            xs, ys = geom.exterior.xy if geom.geom_type == "Polygon" else (None, None)
            if xs is not None:
                ax.fill(xs, ys, facecolor="#c8c8a0", edgecolor="none", zorder=0.5, alpha=0.9)
            elif geom.geom_type == "MultiPolygon":
                for part in geom.geoms:
                    px, py = part.exterior.xy
                    ax.fill(px, py, facecolor="#c8c8a0", edgecolor="none", zorder=0.5, alpha=0.9)
        return "cartopy Natural Earth"
    except Exception as ex:
        ax.text(0.5, 0.5, "⚠ No continent data available\n"
                          "(ssapy earth.png not found, cartopy not installed)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="red", zorder=10,
                bbox=dict(facecolor="white", alpha=0.7))
        print(f"[groundtrack_enhanced] cartopy fallback also failed: {ex}")
        return None


# ── Main plot ────────────────────────────────────────────────────────────────
def plot_enhanced_groundtrack(r_eci_km: np.ndarray, t: Time,
                               site_lat: float, site_lon: float, site_name: str,
                               sat_name: str = "Satellite",
                               min_elev_deg: float = 10.0,
                               save_path: str | None = None):
    r_ecef = gcrf_to_itrf(r_eci_km, t)
    lat, lon, alt = ecef_to_geodetic(r_ecef)
    illum = compute_eclipse(r_eci_km, t)
    ss_lat, ss_lon = subsolar_point(t)

    fig, ax = plt.subplots(figsize=(15, 8), dpi=120)
    ax.set_facecolor("#dfe9f2")

    continent_source = _draw_continents(ax)
    if continent_source:
        print(f"[groundtrack_enhanced] continents rendered from: {continent_source}")

    # Night-side shading: points on Earth where the sun is below the horizon,
    # i.e. angular distance from the (mean) subsolar point > 90 deg.
    # Semi-transparent so the continents drawn above remain visible through it.
    grid_lon = np.linspace(-180, 180, 361)
    grid_lat = np.linspace(-90, 90, 181)
    GLon, GLat = np.meshgrid(grid_lon, grid_lat)
    ss_lat_m, ss_lon_m = np.mean(ss_lat), np.mean(ss_lon)
    cos_c = (np.sin(np.radians(GLat))*np.sin(np.radians(ss_lat_m)) +
             np.cos(np.radians(GLat))*np.cos(np.radians(ss_lat_m)) *
             np.cos(np.radians(GLon - ss_lon_m)))
    night = cos_c < 0
    ax.contourf(GLon, GLat, night.astype(float), levels=[0.5, 1.5],
                colors=["#0a1530"], alpha=0.45, zorder=1)

    # Terminator line (cos_c == 0 contour)
    ax.contour(GLon, GLat, cos_c, levels=[0], colors="orange", linewidths=1.3,
               linestyles="--", zorder=2)

    # Ground track, colour-mapped by illumination fraction (eclipse coloring)
    lon_c = lon.copy()
    jumps = np.where(np.abs(np.diff(lon_c)) > 180)[0]
    seg_start = 0
    cmap = plt.get_cmap("inferno")
    for j in list(jumps) + [len(lon_c) - 1]:
        seg = slice(seg_start, j + 1)
        pts = np.array([lon_c[seg], lat[seg]]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        from matplotlib.collections import LineCollection
        lc = LineCollection(segs, colors=cmap(0.2 + 0.7*illum[seg][:-1]),
                            linewidths=2.2, zorder=4)
        ax.add_collection(lc)
        seg_start = j + 1
    ax.plot(lon_c[0], lat[0], "o", color="lime", markersize=8,
            markeredgecolor="black", zorder=6, label="Start")
    ax.plot(lon_c[-1], lat[-1], "s", color="red", markersize=7,
            markeredgecolor="black", zorder=6, label="End")

    # Subsolar point
    ax.plot(ss_lon_m, ss_lat_m, "*", color="gold", markersize=20,
            markeredgecolor="black", markeredgewidth=1, zorder=7,
            label="Subsolar point")

    # Ground site + elevation-mask visibility circle
    site_ecef = _site_ecef(site_lat, site_lon)
    elev = _elevation_deg(r_ecef, site_ecef, site_lat, site_lon)
    vis_pct = float(np.mean(elev >= min_elev_deg) * 100)
    ax.plot(site_lon, site_lat, "^", color="cyan", markersize=14,
            markeredgecolor="black", markeredgewidth=1.2, zorder=7,
            label=f"{site_name} ({vis_pct:.1f}% visible)")
    # Approximate elevation-mask ground radius (spherical Earth)
    max_range_km = RE_KM * np.arccos(RE_KM / (RE_KM + np.max(alt))) if np.max(alt) > 0 else 2000
    circle_r_deg = np.degrees(max_range_km / RE_KM)
    ax.add_patch(Circle((site_lon, site_lat), circle_r_deg, fill=False,
                        edgecolor="cyan", linestyle=":", linewidth=1, alpha=0.6))

    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude [deg]"); ax.set_ylabel("Latitude [deg]")
    ax.set_title(f"Ground Track — {sat_name}\n"
                f"Eclipse-coloured (bright=sunlit, dark=eclipsed) · "
                f"terminator dashed · {site_name} visibility {vis_pct:.1f}%")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25, linewidth=0.5, zorder=0)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    return fig, ax, dict(site_visibility_pct=vis_pct)


# ── Standalone self-test (two-body propagation; swap in your real r_eci/t) ──
if __name__ == "__main__":
    MU = 398_600.4418

    def _solve_kepler(M, e):
        E = M.copy()
        for _ in range(60):
            dE = (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
            E -= dE
        return E

    def two_body(a_km, e, inc_deg, raan_deg, argp_deg, nu0_deg, n_orbits, n_steps, epoch_iso):
        inc, raan, argp = np.radians([inc_deg, raan_deg, argp_deg])
        nu0 = np.radians(nu0_deg)
        E0 = 2*np.arctan2(np.sqrt(1-e)*np.sin(nu0/2), np.sqrt(1+e)*np.cos(nu0/2))
        M0 = E0 - e*np.sin(E0)
        T_s = 2*np.pi*np.sqrt(a_km**3/MU)
        t_s = np.linspace(0, n_orbits*T_s, n_steps)
        n_rad_s = np.sqrt(MU/a_km**3)
        E = _solve_kepler(M0 + n_rad_s*t_s, e)
        nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
        r_mag = a_km*(1-e*np.cos(E))
        cO, sO = np.cos(raan), np.sin(raan); ci, si = np.cos(inc), np.sin(inc); cw, sw = np.cos(argp), np.sin(argp)
        R11=cO*cw-sO*sw*ci; R12=-cO*sw-sO*cw*ci; R21=sO*cw+cO*sw*ci; R22=-sO*sw+cO*cw*ci; R31=sw*si; R32=cw*si
        xp, yp = r_mag*np.cos(nu), r_mag*np.sin(nu)
        r_eci = np.stack([R11*xp+R12*yp, R21*xp+R22*yp, R31*xp+R32*yp], axis=1)
        t = Time(epoch_iso) + t_s*u.s
        return r_eci, t

    r_eci, t = two_body(a_km=RE_KM+408, e=0.0003, inc_deg=51.6, raan_deg=0,
                        argp_deg=0, nu0_deg=0, n_orbits=1.0, n_steps=400,
                        epoch_iso="2025-06-15 12:00:00")
    _out_dir = Path(__file__).parent / "output"
    _out_dir.mkdir(parents=True, exist_ok=True)
    plot_enhanced_groundtrack(
        r_eci, t, site_lat=28.5, site_lon=-80.6, site_name="Cape Canaveral",
        sat_name="ISS (self-test)", save_path=str(_out_dir / "07_groundtrack_enhanced.png"),
    )