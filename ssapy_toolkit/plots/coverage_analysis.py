"""
ssapy_toolkit.plots.coverage_analysis
=====================================
Ground-site coverage and pass structure for one or more satellites.

Answers the question a ground station actually has: not "is this satellite
overhead" but "how often, for how long, and with what gaps". Two figures per
satellite:

  site_analysis_<sat>.png
      Four views of one propagation window from a specific site: elevation and
      azimuth against time, the satellite's own illumination state, and which
      windows are genuinely observable -- the logical AND of above-mask, sunlit,
      and site-dark, rather than a single collapsed percentage.

  coverage_metrics_<sat>.png
      The pass structure behind a coverage percentage, computed at every point
      on Earth: mean pass duration, longest gap between contacts, contacts per
      day, and total contact minutes per day. "% of time visible" cannot
      distinguish one clean usable pass from a dozen useless slivers at the same
      duty cycle; these four can. All are masked to zero outside the orbit's
      real inclination band rather than filled with a misleading low value.

The coverage grid uses a fixed 3-day, 30-second baseline regardless of the
satellite's own period. A shorter window undersamples longitude -- Earth has
not rotated enough to give each meridian a fair sample -- which produced patchy
contouring that looked like coverage dropping over oceans and was purely a
sampling artefact.

Satellite list resolution, in order: the local JSON store written by
tle_updater (~/ssatk_data), the copy shipped in llnl-ssapy-data, then the
list embedded below. Nothing here touches the network at import time; refresh
elements explicitly with tle_updater.update_satellites_auto().

Run interactively:
  python -m ssapy_toolkit.plots.coverage_analysis
  python -m ssapy_toolkit.plots.coverage_analysis --lat 28.39 --lon -80.60 --name "Cape Canaveral"
"""

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import astropy.units as u
import matplotlib.pyplot as plt

# No sys.path manipulation: this is package code, reached through normal
# imports. It previously prepended ~/SSAPy and ~/SSAPy-Toolkit, which only
# worked if the checkout happened to sit at those exact paths in the current
# user's home directory.
import ssapy
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.time import Time
from ssapy import compute
from ssapy.body import SunPosition

from ..environment import solar_disk_visible_fraction
from ..io.tle_updater import load_satellites, update_satellites_auto


# Real continent rendering (same source as globe_plot.py/moon_plot_3d.py's
# textures where available) — reused from groundtrack_enhanced.py so the
# coverage plots below don't need their own separate implementation.
def _fallback_draw_continents(ax):
    """Self-contained fallback if groundtrack_enhanced.py isn't importable."""
    try:
        from PIL import Image as _PILImage
        from ssapy.utils import find_file
        tex = np.asarray(_PILImage.open(find_file("earth", ext=".png")).convert("RGB"))
        ax.imshow(tex, extent=[-180, 180, -90, 90], origin="upper",
                  aspect="auto", zorder=0.5, alpha=0.9)
        return "ssapy earth.png"
    except Exception:  # noqa: BLE001, S110 - try the next optional map source.
        pass
    try:
        import cartopy.feature as cfeature
        land = cfeature.NaturalEarthFeature("physical", "land", "110m",
                                            facecolor="#c8c8a0")
        for geom in land.geometries():
            geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
            for g in geoms:
                xs, ys = g.exterior.xy
                ax.fill(xs, ys, facecolor="#c8c8a0", edgecolor="none",
                        zorder=0.5, alpha=0.9)
        return "cartopy Natural Earth"
    except Exception:  # noqa: BLE001 - map rendering is optional.
        return None


try:
    from ssapy_toolkit.plots.groundtrack_enhanced import _draw_continents
except ImportError:
    try:
        from groundtrack_enhanced import _draw_continents
    except ImportError:
        _draw_continents = _fallback_draw_continents

R_EARTH = 6.3781e6  # m


# ═════════════════════════════════════════════════════════════════════════════
# SATELLITES LIST — add or remove satellites here
# ═════════════════════════════════════════════════════════════════════════════

# Satellite defaults are intentionally not bundled. Space-Track data may only be
# redistributed with prior approval; users must populate the local cache through
# the authenticated Space-Track updater.
SATELLITES = []
_SATELLITES_SOURCE = "local Space-Track cache"


# ═════════════════════════════════════════════════════════════════════════════
# SATELLITE SELECTION — asked before the ground-site prompt below
# ═════════════════════════════════════════════════════════════════════════════

def select_satellites_from_args():
    parser = argparse.ArgumentParser(description="SSAPy point analysis", add_help=False)
    parser.add_argument("--sats", type=str, default=None,
                        help="Comma-separated satellite names or indices, e.g. "
                             "--sats ISS,Hubble  or  --sats 0,3,7")
    parser.add_argument("--all-sats", action="store_true",
                        help="Analyse every satellite in the pool (skip the prompt)")
    args, _ = parser.parse_known_args()
    return args


def prompt_for_satellites(pool):
    print("\n-- Select satellites ------------------------------------")
    if not pool:
        print("  No satellites in the pool -- run:")
        print("    python tle_updater.py --add-group all_sample")
        print("  then re-run this script.\n")
        return []

    # Only enumerate a pool small enough to read. Once the active catalogue
    # has been pulled in (fetch_active_catalog gets ~30,000 non-decayed
    # objects in one request), printing every name floods the terminal and
    # makes the numeric indices useless anyway -- nobody is going to scroll
    # to entry 24,817. Name matching below already handles that case, so for
    # a large pool just say how big it is and take a search term.
    _LIST_LIMIT = 200
    print(f"  {len(pool)} satellite(s) available.")
    if len(pool) <= _LIST_LIMIT:
        for i, sat in enumerate(pool):
            print(f"    [{i:>3}] {sat.get('name', '?')}")
    else:
        for i, sat in enumerate(pool[:20]):
            print(f"    [{i:>3}] {sat.get('name', '?')}")
        print(f"    ... and {len(pool) - 20:,} more (too many to list).")
        print("    Type part of a name instead, e.g. 'starlink' or 'noaa'.")
    print()
    while True:
        raw = input("  Enter numbers (e.g. 0,3,7), names (e.g. ISS,Hubble), "
                     "'all', or press Enter for all: ").strip()
        if not raw or raw.lower() == "all":
            return pool
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if all(p.isdigit() for p in parts):
            idx = [int(p) for p in parts]
            bad = [i for i in idx if not (0 <= i < len(pool))]
            if bad:
                print(f"  x Index out of range: {bad}. Valid range: 0-{len(pool)-1}.\n")
                continue
            return [pool[i] for i in idx]
        else:
            chosen = [s for s in pool if any(p.lower() in s.get("name", "").lower() for p in parts)]
            if not chosen:
                print(f"  x No satellites matched {parts!r}. Try again.\n")
                continue
            print(f"  Matched: {', '.join(s['name'] for s in chosen)}")
            return chosen


def resolve_satellites():
    """
    Decide which satellites to analyse — asked FIRST, before
    resolve_point() asks for a ground site. The selection order is:

    * ``--all-sats`` flag -> everything in the pool.
    * ``--sats "ISS,Hubble"`` flag -> matched non-interactively.
    * Otherwise -> interactive prompt.

    The pool comes from the user-local Space-Track cache. No satellite TLEs
    are bundled with the package.
    """
    try:
        pool = load_satellites()
    except Exception as e:  # noqa: BLE001 - report missing user cache below.
        print(f"  Note: could not load the local Space-Track cache ({e}).")
        pool = []

    args = select_satellites_from_args()

    if args.all_sats:
        print(f"\n  --all-sats: using all {len(pool)} satellite(s) in the pool.")
        return pool

    if args.sats:
        parts = [p.strip() for p in args.sats.split(",") if p.strip()]
        if all(p.isdigit() for p in parts):
            idx = [int(p) for p in parts]
            chosen = [pool[i] for i in idx if 0 <= i < len(pool)]
        else:
            chosen = [s for s in pool if any(p.lower() in s.get("name", "").lower() for p in parts)]
        print(f"\n  --sats: matched {len(chosen)} satellite(s): "
              f"{', '.join(s['name'] for s in chosen)}")
        return chosen

    return prompt_for_satellites(pool)


# ═════════════════════════════════════════════════════════════════════════════
# INPUT (ground site)
# ═════════════════════════════════════════════════════════════════════════════

def get_point_from_args():
    parser = argparse.ArgumentParser(description="SSAPy point analysis")
    parser.add_argument("--lat",  type=float, default=None)
    parser.add_argument("--lon",  type=float, default=None)
    parser.add_argument("--name", type=str,   default=None)
    parser.add_argument("--alt",  type=float, default=0.0)
    args, _ = parser.parse_known_args()
    return args


def prompt_for_point():
    print("\n-- Enter a point on Earth -------------------------------")
    print("   Latitude  : -90  to  90   (negative = South)")
    print("   Longitude : -180 to 180   (negative = West)")
    print()
    while True:
        try:
            name = input("  Site name (press Enter to skip): ").strip()
            if not name:
                name = "My Site"
            lat_str = input("  Latitude  [-90 to 90]:  ").strip()
            lat = float(lat_str)
            if not -90 <= lat <= 90:
                print("  x Latitude must be between -90 and 90.\n")
                continue
            lon_str = input("  Longitude [-180 to 180]: ").strip()
            lon = float(lon_str)
            if not -180 <= lon <= 180:
                print("  x Longitude must be between -180 and 180.\n")
                continue
            alt_str = input("  Altitude [m, default 0]: ").strip()
            alt = float(alt_str) if alt_str else 0.0
            print(f"\n  OK: {name}  ({lat}, {lon}, {alt}m)\n")
            return name, lat, lon, alt
        except ValueError:
            print("  x Please enter a valid number.\n")


def resolve_point():
    args = get_point_from_args()
    if args.lat is not None and args.lon is not None:
        name = args.name if args.name else f"({args.lat}, {args.lon})"
        return name, args.lat, args.lon, args.alt
    return prompt_for_point()


# ═════════════════════════════════════════════════════════════════════════════
# ORBIT BUILDERS
# ═════════════════════════════════════════════════════════════════════════════

def make_orbit_from_keplerian(a_km=6778.0, e=0.0, i_deg=51.6,
                               raan=0.0, argp=0.0, nu=0.0,
                               year=2025, month=6, day=1, **kwargs):
    t0 = Time(f"{year}-{month:02d}-{day:02d}", scale='utc').gps
    return ssapy.Orbit.fromKeplerianElements(
        a           = a_km * 1e3,
        e           = e,
        i           = np.radians(i_deg),
        pa          = np.radians(argp),
        raan        = np.radians(raan),
        trueAnomaly = np.radians(nu),
        t           = t0,
    )


def make_orbit_from_tle(name, line1, line2, **kwargs):
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tle', delete=False) as f:
        f.write(f"{name}\n{line1}\n{line2}\n")
        fname = f.name
    orbit = ssapy.Orbit.fromTLE(name, fname)
    import os
    os.unlink(fname)
    return orbit

def build_orbit(sat_dict):
    if sat_dict["type"] == "tle":
        return make_orbit_from_tle(
            name  = sat_dict["name"],
            line1 = sat_dict["line1"],
            line2 = sat_dict["line2"],
        )
    else:
        return make_orbit_from_keplerian(**sat_dict)


def propagate(orbit, n_orbits=15, n_points=3000):
    period = 2 * np.pi * np.sqrt(orbit.a**3 / ssapy.constants.WGS84_EARTH_MU)
    t = np.linspace(orbit.t, orbit.t + n_orbits * period, n_points)
    r, v = compute.rv(orbit, t)
    return r, v, t


# ═════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def gcrf_to_itrf(r_gcrf, t_gps):
    times = Time(t_gps, format='gps', scale='utc')
    r_m   = r_gcrf * u.m
    gcrs  = GCRS(CartesianRepresentation(
                     r_m[:, 0], r_m[:, 1], r_m[:, 2]),
                 obstime=times)
    itrs  = gcrs.transform_to(ITRS(obstime=times))
    return itrs.cartesian.xyz.to(u.m).value.T


def site_ecef(lat_deg, lon_deg, alt_m=0.0):
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    r   = R_EARTH + alt_m
    return np.array([
        r * np.cos(phi) * np.cos(lam),
        r * np.cos(phi) * np.sin(lam),
        r * np.sin(phi),
    ])


def elevation_from_site(r_ecef_sat, se):
    phi = np.arcsin(se[2] / np.linalg.norm(se))
    lam = np.arctan2(se[1], se[0])
    sp, cp = np.sin(phi), np.cos(phi)
    sl, cl = np.sin(lam), np.cos(lam)
    ENU = np.array([
        [-sl,       cl,      0  ],
        [-sp * cl, -sp * sl, cp ],
        [ cp * cl,  cp * sl, sp ],
    ])
    rho     = r_ecef_sat - se
    rho_enu = (ENU @ rho.T).T
    dist    = np.linalg.norm(rho_enu, axis=1)
    el      = np.degrees(np.arcsin(rho_enu[:, 2] / dist))
    az      = np.degrees(np.arctan2(rho_enu[:, 0], rho_enu[:, 1])) % 360
    return el, az, dist


def compute_eclipse(r, r_sun):
    return np.array([
        solar_disk_visible_fraction(position, sun_position) < 1.0
        for position, sun_position in zip(r, r_sun)
    ], dtype=bool)


def compute_sun_positions(t):
    sun = SunPosition()
    return np.array([sun(ti) for ti in t])


# ═════════════════════════════════════════════════════════════════════════════
# PER-SATELLITE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def _pass_statistics(vis_mask, dt_s, total_s):
    """
    Extract the actual pass structure from a boolean above-mask time
    series — this is the fix for the core problem with a bare "% of
    time visible" number: identical coverage fractions can come from one
    clean long pass or a dozen useless slivers, and the percentage alone
    can't tell them apart. Returns the set of metrics that actually
    answer a planner's question:

      mean_pass_min   — mean contact-window length (literal "how long is
                        line of sight" per contact)
      max_pass_min    — longest single contact window
      max_gap_min     — longest gap between contacts (drives onboard
                        storage sizing, command latency, autonomy
                        requirements — often the number planners care
                        about most)
      n_passes        — contact opportunities in this window
      passes_per_day  — same, normalized to a daily rate
      cum_contact_min_per_day — total contact minutes/day (budgetable
                        against a downlink data rate, unlike a bare %)
      time_to_first_min — responsiveness for tasking: minutes from the
                        start of this window until the first contact
    """
    n = len(vis_mask)
    total_days = total_s / 86400.0
    if not vis_mask.any():
        return {
            "mean_pass_min": 0.0,
            "max_pass_min": 0.0,
            "max_gap_min": total_s / 60.0,
            "n_passes": 0,
            "passes_per_day": 0.0,
            "cum_contact_min_per_day": 0.0,
            "time_to_first_min": np.inf,
        }

    # Rising/falling edges of the boolean mask -> pass start/end indices
    padded = np.concatenate([[False], vis_mask, [False]])
    edges = np.diff(padded.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]  # exclusive
    pass_lengths_s = (ends - starts) * dt_s
    gap_lengths_s = np.concatenate((
        [starts[0] * dt_s],
        starts[1:] * dt_s - ends[:-1] * dt_s,
        [(n - ends[-1]) * dt_s],
    ))

    mean_pass_min = pass_lengths_s.mean() / 60.0
    max_pass_min = pass_lengths_s.max() / 60.0
    max_gap_min = (gap_lengths_s.max() / 60.0) if len(gap_lengths_s) else 0.0
    n_passes = len(starts)
    time_to_first_min = starts[0] * dt_s / 60.0

    return {
        "mean_pass_min": mean_pass_min,
        "max_pass_min": max_pass_min,
        "max_gap_min": max_gap_min,
        "n_passes": n_passes,
        "passes_per_day": n_passes / total_days,
        "cum_contact_min_per_day": (pass_lengths_s.sum() / 60.0) / total_days,
        "time_to_first_min": time_to_first_min,
    }


def _smooth_masked(grid, valid_mask, sigma=1.1):
    """
    Smooth a grid for DISPLAY only, without blurring across the real
    physical boundary (an orbit's inclination limit) into regions that
    are genuinely, exactly zero — a naive Gaussian blur would leak
    fractional coverage across that edge, which isn't smoothing out
    noise, it's fabricating coverage that can't exist. Standard
    masked-smoothing trick: smooth the valid data and the validity mask
    separately, then divide, so cells near the boundary are only
    averaged with their valid neighbours, not the zeroed-out invalid
    ones on the other side.

    This is a display-only transform — global_mean, site_cov, and every
    other printed statistic are computed from the raw, unsmoothed grids,
    never this.
    """
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return grid  # smoothing unavailable -> just show the raw grid
    grid = np.asarray(grid, dtype=float)
    valid = valid_mask.astype(float)
    # Per-axis boundary mode: longitude (axis 1) genuinely wraps at +/-180,
    # but latitude (axis 0) does NOT — the North and South poles are not
    # adjacent to each other. A single 'wrap' mode for both axes (an
    # earlier version of this used exactly that) silently wrapped latitude
    # too, letting a valid band near one pole bleed values across the
    # wraparound to near the OTHER pole — verified this directly: it
    # produced nonzero "coverage" beyond the real cutoff, which is the
    # exact bug this function exists to prevent, just introduced a
    # different way.
    axis_modes = ("nearest", "wrap")
    num = gaussian_filter(grid * valid, sigma=sigma, mode=axis_modes)
    den = gaussian_filter(valid, sigma=sigma, mode=axis_modes)
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = np.where(den > 1e-6, num / den, 0.0)
    return smoothed


def analyse_satellite(sat_name, orbit, site_name, lat, lon, alt_m,
                       min_el_sat=10.0, sun_el_dark=-6.0, min_el_cov=5.0,
                       n_lat=36, n_lon=72, out_dir="."):
    tag = sat_name.lower().replace(" ", "_").replace("/", "_")
    print(f"\n  Satellite: {sat_name}")
    print(f"  Orbit: {orbit.a/1e3:.0f} km  i={np.degrees(orbit.i):.1f} deg")

    r, _v, t = propagate(orbit, n_orbits=5, n_points=3000)
    print(f"  Propagated {len(t)} steps over {(t[-1]-t[0])/3600:.1f} hours")

    r_ecef = gcrf_to_itrf(r, t)
    t_hr   = (t - t[0]) / 3600.0

    r_sun      = compute_sun_positions(t)
    eclipse    = compute_eclipse(r, r_sun)
    sunlit     = ~eclipse
    r_sun_ecef = gcrf_to_itrf(r_sun, t)

    r_norm = r     / np.linalg.norm(r,     axis=1, keepdims=True)
    s_norm = r_sun / np.linalg.norm(r_sun, axis=1, keepdims=True)
    phase  = np.degrees(np.arccos(
                 np.clip(np.einsum('ij,ij->i', r_norm, s_norm), -1, 1)))

    frac_sunlit = sunlit.sum() / len(t)
    print(f"  Sunlit fraction: {frac_sunlit:.1%}")

    se             = site_ecef(lat, lon, alt_m)
    el, az, dist   = elevation_from_site(r_ecef, se)

    rho_sun  = r_sun_ecef - se
    dist_sun = np.linalg.norm(rho_sun, axis=1)
    up       = se / np.linalg.norm(se)
    el_sun   = np.degrees(np.arcsin(
                   np.einsum('ij,j->i', rho_sun, up) / dist_sun))

    sat_up     = el     > min_el_sat
    site_dark  = el_sun < sun_el_dark
    observable = sat_up & sunlit & site_dark

    dt      = np.mean(np.diff(t))
    vis_min = sat_up.sum()    * dt / 60.0
    obs_min = observable.sum() * dt / 60.0
    print(f"  Visible from site: {vis_min:.1f} min")
    print(f"  Observable windows: {obs_min:.1f} min")

    # Site analysis plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(
        f"{sat_name} | Site: {site_name} ({lat}N, {lon}E)",
        fontsize=14, fontweight="bold"
    )

    ax = axes[0]
    ax.plot(t_hr, el, color="steelblue", linewidth=0.8)
    ax.fill_between(t_hr, el, 0, where=sat_up,
                    color="steelblue", alpha=0.3, label=f"Above {min_el_sat} deg")
    ax.axhline(min_el_sat, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Elevation [deg]")
    ax.set_ylim(bottom=-10)
    ax.set_title(f"Elevation — visible {vis_min:.1f} min")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    ax.scatter(t_hr[sat_up], az[sat_up], s=1.5, color="darkorange")
    ax.set_ylabel("Azimuth [deg]")
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_yticklabels(["N", "E", "S", "W", "N"])
    ax.set_title("Azimuth (compass bearing from site)")

    ax = axes[2]
    ax.fill_between(t_hr, 0, 1, where=sunlit,
                    transform=ax.get_xaxis_transform(),
                    color="gold", alpha=0.35, label="Sunlit")
    ax.fill_between(t_hr, 0, 1, where=eclipse,
                    transform=ax.get_xaxis_transform(),
                    color="midnightblue", alpha=0.35, label="Eclipse")
    ax2 = ax.twinx()
    ax2.plot(t_hr, phase, color="darkorange", linewidth=0.8, alpha=0.8)
    ax2.set_ylabel("Phase angle [deg]", color="darkorange")
    ax2.tick_params(axis='y', labelcolor="darkorange")
    ax.set_ylabel("Illumination")
    ax.set_yticks([])
    ax.set_title(f"Illumination — sunlit {frac_sunlit:.1%}")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[3]
    ax.fill_between(t_hr, el, 0, where=sat_up,
                    color="steelblue", alpha=0.15, label="Above horizon")
    ax.fill_between(t_hr, el, 0, where=observable,
                    color="limegreen", alpha=0.7,
                    label=f"Observable ({obs_min:.1f} min)")
    ax.axhline(min_el_sat, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Elevation [deg]")
    ax.set_xlabel("Time from epoch [hours]")
    ax.set_ylim(bottom=0)
    ax.set_title("Observable windows (sat up + sunlit + site dark)")
    ax.legend(loc="upper right", fontsize=9)

    fig.text(0.5, -0.01,
             "What this is: four views of one specific propagation window from this site — elevation and azimuth vs time, "
             "the satellite's own illumination state, and which windows are actually observable (above the horizon mask, "
             "sunlit, and the site itself dark). How it's made: elevation/azimuth from a real topocentric ENU transform at "
             "the site; eclipse/illumination from a two-circle Sun-Earth angular-overlap check; \"observable\" is the logical "
             "AND of all three real-time conditions, not a single collapsed percentage.",
             ha="center", va="top", fontsize=8.5, wrap=True, transform=fig.transFigure)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"site_analysis_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> site_analysis_{tag}.png")

    # Coverage grid needs a MUCH longer baseline than the per-site plots
    # above. Those use 5 orbits (~8 hours for a LEO like HST) which is
    # fine for showing one specific pass window, but far too short to
    # estimate "% of time visible" at every longitude on the globe: in
    # 8 hours, Earth hasn't rotated nearly enough to give each longitude
    # a fair, representative sample of the ground track passing near it.
    # Verified directly: at a fixed latitude, the TRUE long-run coverage
    # is essentially flat across every longitude (Earth's rotation should
    # expose all of them equally over time) — but the old short window
    # produced wildly different values (0% to 10%+) purely from sparse
    # sampling, not real geography. That's the actual cause of the patchy,
    # wavy contouring that looked like coverage mysteriously dropping over
    # oceans — it doesn't know or care what's underneath it, it was purely
    # an artifact of not sampling long enough.
    #
    # Fixed time span (not tied to this satellite's own orbital period) so
    # it works consistently from LEO up through GEO: 3 days at 30s
    # resolution is short enough to stay fast, long enough that Earth's
    # rotation has fully swept every longitude under the orbit's
    # inclination band multiple times over, and fine enough not to miss
    # any real pass (typical LEO passes last several minutes).
    t_cov = np.arange(orbit.t, orbit.t + 3*86400.0, 30.0)
    r_cov, _ = compute.rv(orbit, t_cov)
    r_ecef_cov = gcrf_to_itrf(r_cov, t_cov)
    print(f"  Coverage grid baseline: {len(t_cov)} samples over "
         f"{(t_cov[-1]-t_cov[0])/86400:.1f} days")

    # Coverage grid
    lat_grid = np.linspace(-90,  90,  n_lat)
    lon_grid = np.linspace(-180, 180, n_lon)
    frac              = np.zeros((n_lat, n_lon))
    mean_pass_grid    = np.zeros((n_lat, n_lon))
    max_pass_grid     = np.zeros((n_lat, n_lon))
    max_gap_grid      = np.zeros((n_lat, n_lon))
    passes_day_grid   = np.zeros((n_lat, n_lon))
    cum_contact_grid  = np.zeros((n_lat, n_lon))
    dt_s = 30.0
    total_s = t_cov[-1] - t_cov[0]

    for i, la in enumerate(lat_grid):
        phi = np.radians(la)
        sp, cp = np.sin(phi), np.cos(phi)
        for j, lo in enumerate(lon_grid):
            lam  = np.radians(lo)
            gp   = R_EARTH * np.array([cp*np.cos(lam), cp*np.sin(lam), sp])
            rho  = r_ecef_cov - gp
            dist = np.linalg.norm(rho, axis=1)
            up_  = gp / R_EARTH
            el_  = np.degrees(np.arcsin(np.einsum('ij,j->i', rho, up_) / dist))
            vis_ = el_ > min_el_cov
            frac[i, j] = np.mean(vis_)
            stats = _pass_statistics(vis_, dt_s, total_s)
            mean_pass_grid[i, j]   = stats["mean_pass_min"]
            max_pass_grid[i, j]    = stats["max_pass_min"]
            max_gap_grid[i, j]     = stats["max_gap_min"]
            passes_day_grid[i, j]  = stats["passes_per_day"]
            cum_contact_grid[i, j] = stats["cum_contact_min_per_day"]

    global_mean = frac.mean()
    # Smoothed DISPLAY-ONLY copies — global_mean/site_cov below and every
    # other printed statistic still use the raw, unsmoothed grids.
    _valid = frac > 0.0
    frac_smooth        = _smooth_masked(frac, _valid)
    mean_pass_smooth   = _smooth_masked(mean_pass_grid, _valid)
    max_gap_smooth     = _smooth_masked(max_gap_grid, _valid)
    passes_day_smooth  = _smooth_masked(passes_day_grid, _valid)
    cum_contact_smooth = _smooth_masked(cum_contact_grid, _valid)
    # Recomputed from the SAME long-baseline r_ecef_cov used for the grid
    # above, not the original short 5-orbit `el` — those were two
    # different time windows, so the site marker's printed percentage
    # and the heatmap colour directly underneath it could disagree
    # (e.g. a site sitting right at an orbit's inclination limit, where
    # short-window sampling noise is largest, showing a misleadingly
    # high number that the surrounding map colours don't support).
    el_cov_site = elevation_from_site(r_ecef_cov, se)[0]
    site_cov    = np.mean(el_cov_site > min_el_cov)
    print(f"  Global mean coverage: {global_mean:.1%}")
    print(f"  Site coverage: {site_cov:.1%}")

    # ── Pass-structure metrics (mean/max pass duration, max gap, passes/day,
    # cumulative contact/day) ────────────────────────────────────────────────
    # "% of time visible" is a duty cycle — it tells you how much of the
    # window a site is in contact, but it collapses all the temporal
    # structure that actually drives mission design. The same percentage
    # can be one clean usable pass or a dozen useless slivers, and the
    # number alone can't tell you which. These four panels are the metrics
    # that actually answer a planner's question, using the exact same
    # per-cell visibility time series the percentage above was collapsed
    # from — no extra propagation needed, just not throwing the structure
    # away this time.
    metric_panels = [
        (mean_pass_smooth, "Mean pass duration [min]", "viridis"),
        (max_gap_smooth,   "Max coverage gap [min] (lower=better)", "magma_r"),
        (passes_day_smooth, "Passes per day", "cividis"),
        (cum_contact_smooth, "Cumulative contact [min/day]", "plasma"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"Pass-structure metrics — {sat_name}", fontsize=14, fontweight="bold")
    for ax, (grid_vals, title, cmap) in zip(axes.ravel(), metric_panels):
        _draw_continents(ax)
        # Same zero-masking fix as the percentage heatmap — cells with no
        # coverage at all (guaranteed outside the inclination band) should
        # show bare basemap, not a "0" colour that reads as data.
        masked = np.ma.masked_where(frac <= 0.0, grid_vals)
        im = ax.contourf(lon_grid, lat_grid, masked, levels=21, cmap=cmap,
                         alpha=0.75, zorder=1, extend="neither")
        fig.colorbar(im, ax=ax, label=title)
        ax.plot(lon, lat, marker="*", markersize=14, color="cyan",
                markeredgecolor="black", markeredgewidth=0.8, zorder=5)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude [deg]"); ax.set_ylabel("Latitude [deg]")
        ax.grid(True, linewidth=0.3, color="gray", alpha=0.4)
    fig.text(0.5, -0.02,
             "What this is: the pass-structure behind the coverage percentage above, since \"% of time visible\" alone can't "
             "tell one clean usable pass from a dozen useless slivers at the same duty cycle. How it's made: each grid cell's "
             "real elevation time series is split into individual contacts by rising/falling edges of the above-mask crossing, "
             "then reduced to mean/max pass length, longest gap between contacts, contact count per day, and total contact "
             "minutes per day — smoothed for display only, all four masked to zero outside the orbit's real inclination band.",
             ha="center", va="top", fontsize=8.5, wrap=True, transform=fig.transFigure)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"coverage_metrics_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> coverage_metrics_{tag}.png")

    # Site-specific pass statistics (time to first access included here,
    # since it's only meaningful as a single number from a specific
    # epoch/site, not something that makes sense as a global heatmap).
    site_vis_mask = el_cov_site > min_el_cov
    site_stats = _pass_statistics(site_vis_mask, dt_s, total_s)
    print(f"  {site_name} pass structure: mean {site_stats['mean_pass_min']:.1f} min, "
         f"max {site_stats['max_pass_min']:.1f} min, "
         f"max gap {site_stats['max_gap_min']:.1f} min, "
         f"{site_stats['passes_per_day']:.1f} passes/day, "
         f"{site_stats['cum_contact_min_per_day']:.1f} min/day contact, "
         f"first access in {site_stats['time_to_first_min']:.1f} min")

    return {
        "name":        sat_name,
        "tag":         tag,
        "frac":        frac,
        "frac_smooth": frac_smooth,
        "lat_grid":    lat_grid,
        "lon_grid":    lon_grid,
        "site_cov":    site_cov,
        "global_mean": global_mean,
        "vis_min":     vis_min,
        "obs_min":     obs_min,
        "frac_sunlit": frac_sunlit,
        "el":          el,
        "t_hr":        t_hr,
        "sat_up":      sat_up,
        "observable":  observable,
        "mean_pass_grid":   mean_pass_grid,
        "max_pass_grid":    max_pass_grid,
        "max_gap_grid":     max_gap_grid,
        "passes_day_grid":  passes_day_grid,
        "cum_contact_grid": cum_contact_grid,
        "site_pass_stats":  site_stats,
    }


# ═════════════════════════════════════════════════════════════════════════════
# COMPARISON PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def _analysis_output_dir(site_name, lat, lon):
    """Directory for one site's analysis figures, created if needed.

    Uses the toolkit's figure path (~/ssatk_figures, or SSATK_FIGURES_DIR)
    rather than a private ~/ssapy_outputs tree, so these land beside every
    other generated figure. Falls back to the old location only if the
    toolkit's figpath module is unavailable.
    """
    folder = _site_folder_name(site_name, lat, lon)
    try:
        from .figpath import ssatk_path
        base = os.path.dirname(ssatk_path(f"demo_gallery/figures/{folder}/_placeholder"))
    except Exception:  # noqa: BLE001 - preserve the historical output fallback.
        base = os.path.join(os.path.expanduser("~"), "ssatk_figures",
                            "demo_gallery", "figures", folder)
    os.makedirs(base, exist_ok=True)
    return base


def _site_folder_name(site_name, lat, lon):
    """
    Build a unique output folder name per actual location.

    Previously this was just site_name.lower().replace(" ", "_") — any run
    where the site name was left blank fell back to the literal string
    "My Site" every time, so *every* unnamed location collapsed into the
    same ~/ssapy_outputs/my_site/ folder, silently overwriting results
    from a completely different lat/lon on the next run. Appending the
    coordinates (to 2 decimal places, ~1km resolution) guarantees each
    distinct location gets its own folder regardless of whether — or how
    — it was named.
    """
    base = site_name.lower().strip()
    base = "".join(c if (c.isalnum() or c in "-_") else "_" for c in base.replace(" ", "_"))
    base = base.strip("_") or "site"
    lat_tag = f"{lat:+.2f}".replace("+", "p").replace("-", "m")
    lon_tag = f"{lon:+.2f}".replace("+", "p").replace("-", "m")
    return f"{base}_{lat_tag}_{lon_tag}"


def main():
    print("\n-- SSAPy Analysis -- Multiple Satellites, Any Point -----")

    # Satellites asked first now — was after resolve_point() before.
    selected = resolve_satellites()
    if not selected:
        print("\nNo satellites selected -- nothing to do.")
        return

    site_name, lat, lon, alt_m = resolve_point()
    selected = update_satellites_auto(selected)
    # Per-site folder under the toolkit's figure directory. The site subfolder
    # is deliberate: coverage depends entirely on where you are standing, so
    # two runs for different sites are different results, not one overwriting
    # the other. It used to be ~/ssapy_outputs, which predates the toolkit's
    # ssatk_figures convention and left these outputs somewhere nothing else
    # looked.
    out_dir = _analysis_output_dir(site_name, lat, lon)
    print(f"  Saving outputs to: {out_dir}")

    print(f"\nRunning analysis for {len(selected)} satellite(s)...")

    results = []
    for sat_dict in selected:
        try:
            orbit = build_orbit(sat_dict)
            res   = analyse_satellite(
                sat_name    = sat_dict["name"],
                orbit       = orbit,
                site_name   = site_name,
                lat         = lat,
                lon         = lon,
                alt_m       = alt_m,
                min_el_sat  = 10.0,
                sun_el_dark = -6.0,
                min_el_cov  = 5.0,
                out_dir     = out_dir,
            )
            results.append(res)
        except Exception as e:  # noqa: BLE001 - report and continue other satellites.
            print(f"  ERROR with {sat_dict['name']}: {e}")

    print("\n-- All output files ------------------------------------")
    for res in results:
        tag = res["tag"]
        for f in (f"site_analysis_{tag}.png", f"coverage_metrics_{tag}.png"):
            print(f"  {f}")
    print()


if __name__ == "__main__":
    main()
