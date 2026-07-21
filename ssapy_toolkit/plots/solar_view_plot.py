"""
ssapy_toolkit/plots/solar_view_plot.py
========================================
Standalone solar system visualization script.
Renders properly 3D-shaded planets using solar_bodies.py.

Run via GUI "Export Plots" tab or directly:

    python -m ssapy_toolkit.plots.solar_view_plot

Reads GUI_CONFIG from environment or falls back to defaults.

Output
------
  <output_dir>/solar_view_plot.html   — interactive Plotly HTML
  <output_dir>/solar_view_plot.jpg    — static snapshot (requires kaleido)
"""

from __future__ import annotations
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # ssapy_toolkit/plots/
_ROOT = _HERE.parent.parent                      # SSAPy-Toolkit/
for _p in [str(_ROOT), str(_ROOT / "ssapy_toolkit"), str(_ROOT / "core")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Project imports ───────────────────────────────────────────────────────────
from ssapy_toolkit.plots.solar_bodies import (
    make_planet_traces,
    make_saturn_ring_traces,
    make_sun_traces,
    make_moon_traces,
    _R_AU,
)

# Always define these locally — importing toolkit_gui.py here is unsafe:
# it's a Streamlit *app* script, not a library, and merely importing it
# executes its entire top-level code (page config, CSS, sidebar, all 8
# tabs) outside of a real Streamlit session. That doesn't just fail
# quietly — it floods the terminal with one "missing ScriptRunContext"
# warning per Streamlit call in that file (thousands of lines).
_PLANETS = {
    "Mercury": dict(a=0.38710, e=0.20563, i=7.005,  Om=48.331,  w=29.125,  M0=174.795),
    "Venus":   dict(a=0.72333, e=0.00677, i=3.395,  Om=76.680,  w=54.884,  M0=50.416),
    "Earth":   dict(a=1.00000, e=0.01671, i=0.000,  Om=0.000,   w=102.937, M0=357.527),
    "Mars":    dict(a=1.52366, e=0.09341, i=1.850,  Om=49.558,  w=286.502, M0=19.393),
    "Jupiter": dict(a=5.20336, e=0.04839, i=1.303,  Om=100.464, w=273.867, M0=20.065),
    "Saturn":  dict(a=9.53707, e=0.05415, i=2.485,  Om=113.666, w=339.391, M0=317.020),
    "Uranus":  dict(a=19.1913, e=0.04717, i=0.773,  Om=74.230,  w=96.998,  M0=142.827),
    "Neptune": dict(a=30.0690, e=0.00859, i=1.770,  Om=131.722, w=272.847, M0=259.780),
}

def _solve_kepler(M: float, e: float) -> float:
    E = M
    for _ in range(60):
        dE = (M - E + e * math.sin(E)) / (1.0 - e * math.cos(E))
        E += dE
        if abs(dE) < 1e-12:
            break
    return E

def _planet_pos_au(p: dict, t_jd: float):
    J2000 = 2_451_545.0
    T_days = t_jd - J2000
    n_deg_day = 360.0 / (365.25 * math.sqrt(p["a"] ** 3))
    M = math.radians((p["M0"] + n_deg_day * T_days) % 360.0)
    E = _solve_kepler(M, p["e"])
    nu = 2.0 * math.atan2(math.sqrt(1 + p["e"]) * math.sin(E / 2),
                           math.sqrt(1 - p["e"]) * math.cos(E / 2))
    r = p["a"] * (1.0 - p["e"] * math.cos(E))
    xo, yo = r * math.cos(nu), r * math.sin(nu)
    w  = math.radians(p["w"])
    i  = math.radians(p["i"])
    Om = math.radians(p["Om"])
    cw, sw = math.cos(w), math.sin(w)
    ci, si = math.cos(i), math.sin(i)
    cO, sO = math.cos(Om), math.sin(Om)
    x = cO*(cw*xo - sw*yo) - sO*(sw*xo + cw*yo)*ci
    y = sO*(cw*xo - sw*yo) + cO*(sw*xo + cw*yo)*ci
    z = (sw*xo + cw*yo)*si
    return x, y, z

def _orbit_trail_au(p: dict, t_jd: float, n_pts: int = 360):
    T_orbit_days = 365.25 * math.sqrt(p["a"] ** 3)
    xs, ys, zs = [], [], []
    for k in range(n_pts + 1):
        tj = t_jd - T_orbit_days + k * T_orbit_days / n_pts
        x, y, z = _planet_pos_au(p, tj)
        xs.append(x); ys.append(y); zs.append(z)
    return xs, ys, zs


# ── Catalog helper ────────────────────────────────────────────────────────────
def _load_stars(catalog_path: str, mag_limit: float = 5.5):
    """Load HYG catalog and return GCRF unit-sphere arrays for starfield."""
    try:
        import pandas as pd
        df = pd.read_csv(catalog_path, encoding="utf-8")
    except UnicodeDecodeError:
        import pandas as pd
        df = pd.read_csv(catalog_path, encoding="latin-1")
    except Exception:
        return None
    df = df[(df["mag"] < mag_limit) & (df["mag"] > -10)].dropna(subset=["ra","dec","mag"])
    ra_r  = np.radians(df["ra"].values * 15.0)
    dec_r = np.radians(df["dec"].values)
    cx = np.cos(dec_r)*np.cos(ra_r)
    cy = np.cos(dec_r)*np.sin(ra_r)
    cz = np.sin(dec_r)
    return cx, cy, cz, df["mag"].values, df["spect"].fillna("G").str[:1].values


_SPECT_COLORS = {
    "O":"rgb(155,175,255)","B":"rgb(171,191,255)","A":"rgb(201,217,255)",
    "F":"rgb(247,247,255)","G":"rgb(255,245,235)","K":"rgb(255,209,160)","M":"rgb(255,204,112)",
}


# ── Figure builder ────────────────────────────────────────────────────────────

def build_figure(cfg: dict) -> go.Figure:
    import datetime as _dt

    # Resolve JD
    yr    = int(cfg.get("sol_year",  2025))
    mo    = int(cfg.get("sol_month", 1))
    try:
        d = _dt.date(yr, mo, 1)
    except ValueError:
        d = _dt.date(2025, 1, 1)
    t_jd = 2_451_545.0 + (d - _dt.date(2000, 1, 1)).days

    show_planets = {
        name: bool(cfg.get(f"sol_show_{name.lower()}", True))
        for name in _PLANETS
    }
    show_trails   = bool(cfg.get("sol_show_trails",   True))
    show_stars    = bool(cfg.get("sol_show_stars",    True))
    show_ecliptic = bool(cfg.get("sol_show_ecliptic", True))
    show_labels   = bool(cfg.get("sol_show_labels",   True))
    show_moon     = bool(cfg.get("sol_show_moon",     True))
    scale_au      = float(cfg.get("planet_scale",     1.0))
    catalog_path  = cfg.get("star_catalog",
                            str(Path.home() / "bright_stars.csv"))
    sphere_res    = int(cfg.get("sphere_resolution",  50))
    bg            = cfg.get("bg_color", "#060810")

    fig = go.Figure()

    # ── Starfield ─────────────────────────────────────────────────────────────
    # Was silently skipped whenever `catalog_path` didn't exist (default:
    # ~/bright_stars.csv) — no error, just no stars. Now tries a few common
    # locations first, matching van_allen_plot_3d.py's approach, and falls
    # back to a synthetic random starfield so stars always render.
    if show_stars:
        _star_paths = [
            catalog_path,
            str(Path.home() / "SSAPy" / "ssapy" / "data" / "bright_stars.csv"),
            os.path.join(os.path.dirname(__file__), "bright_stars.csv"),
        ]
        try:
            from ssapy.utils import find_file as _ssapy_find_file
            _star_paths.insert(0, _ssapy_find_file("bright_stars", ext=".csv"))
        except Exception:
            pass

        res = None
        _found_path = None
        for _sp in _star_paths:
            if _sp and Path(_sp).exists():
                res = _load_stars(_sp, mag_limit=5.5)
                if res is not None:
                    _found_path = _sp
                    break

        R_star = 45.0
        if res is not None:
            cx, cy, cz, mags, spects = res
            sizes  = np.clip(0.6 * (5.5 - mags)**1.1, 0.3, 3.5)
            colors = [_SPECT_COLORS.get(s, _SPECT_COLORS["G"]) for s in spects]
            print(f"[solar_view_plot] Loaded {len(mags)} stars from {_found_path}")
        else:
            print(f"[solar_view_plot] No star catalog found (tried "
                  f"{[p for p in _star_paths if p]}) — using a synthetic "
                  f"starfield instead of a real catalog.")
            _rng = np.random.default_rng(7)
            _n_syn = 1500
            _th = _rng.uniform(0, 2*np.pi, _n_syn)
            _ph = np.arccos(_rng.uniform(-1, 1, _n_syn))
            cx, cy, cz = np.sin(_ph)*np.cos(_th), np.sin(_ph)*np.sin(_th), np.cos(_ph)
            sizes = _rng.uniform(0.4, 1.8, _n_syn)
            colors = "white"
        fig.add_trace(go.Scatter3d(
            x=cx*R_star, y=cy*R_star, z=cz*R_star,
            mode="markers",
            marker=dict(size=sizes, color=colors, opacity=0.75),
            hoverinfo="skip", name="Stars", showlegend=True,
        ))

    # ── Ecliptic grid — faint teal reference circles (radii 1/5/10/20/30 AU)
    # + 12 radial spokes every 30°, all centred on the Sun. Purely a visual
    # scale reference, not orbits — labelled here (showlegend on the first
    # circle) so it doesn't look like unexplained stray lines.
    if show_ecliptic:
        th = np.linspace(0, 2*np.pi, 200)
        outer_a = max((p["a"] for n, p in _PLANETS.items() if show_planets.get(n)), default=1.5)
        _first_ring = True
        for _r in [1, 5, 10, 20, 30]:
            if _r > outer_a * 1.1:
                continue
            fig.add_trace(go.Scatter3d(
                x=np.cos(th)*_r, y=np.sin(th)*_r, z=np.zeros(200),
                mode="lines", line=dict(color="rgba(0,255,156,0.07)", width=1),
                hoverinfo="skip", showlegend=_first_ring,
                name="Ecliptic reference grid" if _first_ring else None,
                legendgroup="ecliptic_grid",
            ))
            _first_ring = False
        for ang in range(0, 360, 30):
            a = math.radians(ang)
            fig.add_trace(go.Scatter3d(
                x=[0, math.cos(a)*outer_a*1.1],
                y=[0, math.sin(a)*outer_a*1.1],
                z=[0, 0],
                mode="lines", line=dict(color="rgba(0,255,156,0.04)", width=1),
                hoverinfo="skip", showlegend=False,
                legendgroup="ecliptic_grid",
            ))

    # ── Sun ───────────────────────────────────────────────────────────────────
    for t in make_sun_traces(r_display_au=_R_AU["Sun"] * scale_au):
        fig.add_trace(t)

    # ── Planets ───────────────────────────────────────────────────────────────
    earth_pos = None
    trail_colors = {
        "Mercury": "rgba(170,170,170,0.35)", "Venus":   "rgba(255,204,68,0.35)",
        "Earth":   "rgba(26,143,209,0.40)",  "Mars":    "rgba(212,90,42,0.38)",
        "Jupiter": "rgba(200,164,110,0.30)", "Saturn":  "rgba(232,217,160,0.28)",
        "Uranus":  "rgba(125,232,232,0.28)", "Neptune": "rgba(63,84,186,0.28)",
    }

    for name, p in _PLANETS.items():
        if not show_planets.get(name, False):
            continue
        x, y, z = _planet_pos_au(p, t_jd)
        pos = (x, y, z)
        if name == "Earth":
            earth_pos = pos

        # Orbit trail
        if show_trails:
            tx, ty, tz = _orbit_trail_au(p, t_jd)
            fig.add_trace(go.Scatter3d(
                x=tx, y=ty, z=tz, mode="lines",
                line=dict(color=trail_colors.get(name, "rgba(200,200,200,0.3)"), width=1),
                hoverinfo="skip", showlegend=False,
            ))

        # Planet sphere(s)
        for t in make_planet_traces(name, pos, scale_au=scale_au,
                                    show_label=show_labels, n=sphere_res):
            fig.add_trace(t)

        # Saturn rings
        if name == "Saturn":
            for t in make_saturn_ring_traces(pos, scale_au=scale_au):
                fig.add_trace(t)

    # ── Moon ──────────────────────────────────────────────────────────────────
    # Was a fixed offset at a constant angle regardless of date (and smaller
    # than Earth's own display radius, so it rendered inside the Earth
    # sphere). Now a real position for t_jd, shaded like the other bodies.
    if show_moon and earth_pos is not None:
        for t in make_moon_traces(earth_pos, t_jd, show_label=show_labels):
            fig.add_trace(t)

    # ── Layout ────────────────────────────────────────────────────────────────
    T_yr = (t_jd - 2_451_545.0) / 365.25
    outer_a = max((p["a"] for n, p in _PLANETS.items() if show_planets.get(n)), default=1.5)
    rng = outer_a * 1.25

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-rng, rng], showbackground=False,
                       showgrid=False, zeroline=False, title="X (AU)"),
            yaxis=dict(range=[-rng, rng], showbackground=False,
                       showgrid=False, zeroline=False, title="Y (AU)"),
            zaxis=dict(range=[-rng*0.25, rng*0.25], showbackground=False,
                       showgrid=False, zeroline=False, title="Z (AU)"),
            bgcolor=bg,
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.25),
            camera=dict(eye=dict(x=0, y=0, z=2.4),
                        up=dict(x=0, y=1, z=0)),
        ),
        paper_bgcolor=bg,
        font=dict(color="#C8D8E8"),
        title=dict(
            text=f"Heliocentric Solar System — {2000+T_yr:.3f}",
            x=0.5, font=dict(color="#00FF9C", size=14),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#333",
                    borderwidth=1, font=dict(size=10)),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────
DEFAULT_CFG = dict(
    sol_year=2025, sol_month=6,
    sol_show_mercury=True, sol_show_venus=True, sol_show_earth=True,
    sol_show_mars=True,    sol_show_jupiter=True, sol_show_saturn=True,
    sol_show_uranus=False, sol_show_neptune=False,
    sol_show_moon=True,    sol_show_trails=True,
    sol_show_stars=True,   sol_show_ecliptic=True, sol_show_labels=True,
    planet_scale=1.0,
    sphere_resolution=50,
    bg_color="#060810",
    star_catalog=str(Path.home() / "bright_stars.csv"),
    output_dir=str(Path.home() / "yu_figures" / "demo_gallery" / "figures"),
)

if __name__ == "__main__":
    cfg = DEFAULT_CFG.copy()
    env_cfg = os.environ.get("GUI_CONFIG", "")
    if env_cfg:
        try:
            cfg.update(json.loads(env_cfg))
            print("[solar_view_plot] Loaded GUI_CONFIG from environment.")
        except json.JSONDecodeError as e:
            print(f"[solar_view_plot] Warning: bad GUI_CONFIG ({e}); using defaults.")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[solar_view_plot] Building scene for "
          f"{cfg['sol_year']}-{cfg['sol_month']:02d} ...")
    fig = build_figure(cfg)

    html_path = output_dir / "solar_view_plot.html"
    fig.write_html(str(html_path))
    print(f"[solar_view_plot] Saved → {html_path}")

    jpg_path = output_dir / "solar_view_plot.jpg"
    try:
        fig.write_image(str(jpg_path), width=1920, height=1080, scale=2)
        print(f"[solar_view_plot] Saved → {jpg_path}")
    except Exception as e:
        print(f"[solar_view_plot] write_image failed: {e} — install kaleido")

    print("[solar_view_plot] Done.")