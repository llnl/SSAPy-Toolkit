"""
SSAPy Toolkit GUI  —  Mission Command Centre
Run with:  streamlit run toolkit_gui.py
"""

import json
import os
import subprocess
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st

# Astropy/ERFA solar-system ephemeris functions (used by Sun/Moon/Eclipse
# layers) warn loudly whenever a date falls outside 1900-2100 — this is
# expected and harmless for the far-future Star Map epoch slider (up to
# year 14000) and was previously amplified by a bug that fed an
# accidentally-inflated orbit period into propagation (now fixed). Silencing
# just this one warning category so it doesn't spam the terminal; genuine
# errors still raise/print normally.
try:
    from erfa import ErfaWarning
    warnings.filterwarnings("ignore", category=ErfaWarning)
except ImportError:
    pass

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="SSAPy Toolkit",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── try to import the core library ───────────────────────────────────────────
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))

_CORE_OK      = False
_core_err_msg = ""

# Step 1 — critical: OrbitalState + PlotlyScene
try:
    from core import OrbitalState, PlotlyScene
    _core_step1 = True
except Exception as _e:
    _core_step1   = False
    _core_err_msg = "core/__init__.py: " + str(_e)

# Step 2 — layers (individual imports so one missing class won't block the rest)
EarthLayer = StarfieldLayer = GroundTrackLayer = MoonLayer = SunLayer = None
TerminatorLayer = EclipseLayer = VanAllenLayer = LagrangeLayer = None
NTWLayer = BurnLayer = SensorFOVLayer = None
_layer_warns = []

if _core_step1:
    try:
        import importlib as _il
        _lmod = _il.import_module("core.layers")
        for _cls in ["EarthLayer","StarfieldLayer","GroundTrackLayer","MoonLayer",
                     "SunLayer","TerminatorLayer","EclipseLayer","VanAllenLayer",
                     "LagrangeLayer","NTWLayer","BurnLayer","SensorFOVLayer"]:
            if hasattr(_lmod, _cls):
                globals()[_cls] = getattr(_lmod, _cls)
            else:
                _layer_warns.append(_cls)
    except Exception as _e:
        _core_err_msg = "core/layers.py: " + str(_e)
        _core_step1 = False

# Step 3 — satellite + frames
Satellite3D = BurnEvent = Frame = None
if _core_step1:
    try:
        from core.satellite import Satellite3D, BurnEvent
        from core.frames import Frame
    except Exception as _e:
        _layer_warns.append("satellite/frames: " + str(_e))

if _core_step1:
    _CORE_OK = True
    if _layer_warns:
        _core_err_msg = "partial — missing: " + ", ".join(_layer_warns)

# ── tle_updater (optional — Space-Track / Celestrak live TLE fetch) ──────────
# Search the toolkit root, home dir, and a few other common spots so the user
# doesn't have to worry about where they saved the file.
_here = Path(__file__).parent
for _search in [
    _here,                                    # C:\Users\diamond10\SSAPy-Toolkit\
    _here / "ssapy_toolkit",                  # …\SSAPy-Toolkit\ssapy_toolkit\
    Path.home(),                              # C:\Users\diamond10\
    Path.home() / "SSAPy-Toolkit",            # in case gui is run from elsewhere
    Path.home() / "Downloads",               # common drop location
]:
    if str(_search) not in _sys.path and _search.exists():
        _sys.path.insert(0, str(_search))

try:
    from tle_updater import fetch_tle_celestrak, fetch_tle_spacetrack, ST_USER, ST_PASSWORD
    _TLE_UPDATER_OK = True
except ImportError:
    _TLE_UPDATER_OK = False
    fetch_tle_celestrak = fetch_tle_spacetrack = None
    ST_USER = ST_PASSWORD = ""

# ── sun direction / rendering ─────────────────────────────────────────────────
# `core/sun.py` (SunLayer + sun_position_eci) no longer exists — it was split
# into core/sun_mpl.py (ephemeris + diffuse texture shading, matplotlib) and
# core/sun_render.py (limb-darkened Sun rendering, matplotlib). Neither of
# those replaces the Plotly-corona `SunLayer` this file used to import, so
# the two capabilities are tracked separately below:
#   _SUN_PY_OK      — sun direction lookup works (backs the FOV sun-shading
#                      checkboxes and cone shading further down)
#   _SUN_CORONA_OK  — a Plotly SunLayer with corona/build_traces() is
#                      available for the solar-view scene. Currently always
#                      False; the solar-view Sun falls back to the inline
#                      plain-sphere renderer already in this file.
_SunLayer = None
_SUN_CORONA_OK = False

try:
    from core.sun_mpl import get_sun_position as _get_sun_position_eci

    def _sun_position_eci(t_jd):
        """Compatibility shim for the old core/sun.py sun_position_eci(t_jd)
        API. Returns (unit_vector_gcrf, distance_km) from the current
        core/sun_mpl.py ephemeris."""
        from astropy.time import Time
        t = Time(t_jd, format="jd", scale="utc")
        pos_m = np.atleast_2d(_get_sun_position_eci(t))[0]
        dist_km = float(np.linalg.norm(pos_m)) / 1000.0
        if dist_km == 0:
            return np.array([1.0, 0.0, 0.0]), 0.0
        return pos_m / (dist_km * 1000.0), dist_km

    _SUN_PY_OK = True
except ImportError:
    _SUN_PY_OK = False
    _sun_position_eci = None

# ── constants ─────────────────────────────────────────────────────────────────
RE_KM     = 6_378.137
MU        = 398_600.4418
MOON_A_KM = 384_400.0

def _estimate_belt_residence(a_km: float, e: float, n_orbits: float = 1.0) -> dict:
    """
    Analytic estimate of how long a satellite spends inside the Van Allen
    radiation belts per orbit and over the configured propagation window.

    This is a geometric approximation (not a full IGRF/AP-8/AE-8 flux model):
    it treats the belts as fixed spherical shells around Earth —
    inner belt ≈ 1.2–2.5 Rₑ, outer belt ≈ 3–10 Rₑ (Rₑ = 6378.137 km) — and
    computes the true-anomaly arc of the orbit's radius-vs-time curve that
    falls inside each shell, then converts that arc to time via Kepler's
    equation. Good for quick mission-design screening; not a dose model.

    Returns dict with keys (all times in seconds):
      t_inner_s, t_outer_s, t_total_s   — per orbit
      frac_inner, frac_outer            — fraction of one orbit
      t_inner_total_s, t_outer_total_s  — scaled by n_orbits
    """
    Re = RE_KM
    r_p = a_km * (1 - e)   # perigee radius
    r_a = a_km * (1 + e)   # apogee radius
    T_s = 2*np.pi*np.sqrt(a_km**3 / MU)

    def _time_in_shell(r_lo_re, r_hi_re):
        r_lo, r_hi = r_lo_re*Re, r_hi_re*Re
        # No overlap with orbit's radius range at all
        if r_hi <= r_p or r_lo >= r_a:
            return 0.0
        lo = max(r_lo, r_p); hi = min(r_hi, r_a)
        # Sample true anomaly finely and integrate dt where r(nu) in [lo,hi]
        nu = np.linspace(0, 2*np.pi, 2000)
        r_nu = a_km*(1-e**2) / (1 + e*np.cos(nu))
        mask = (r_nu >= lo) & (r_nu <= hi)
        if not mask.any():
            return 0.0
        # Mean anomaly from true anomaly (standard Kepler relation)
        E = 2*np.arctan2(np.sqrt(1-e)*np.sin(nu/2), np.sqrt(1+e)*np.cos(nu/2))
        M = E - e*np.sin(E)
        M = np.unwrap(M)
        t = (M / (2*np.pi)) * T_s
        t = t - t.min()
        # Fraction of samples in-shell scaled to arc time (coarse but robust
        # to the multi-crossing geometry of eccentric orbits)
        frac = mask.sum() / len(nu)
        return frac * T_s

    t_inner = _time_in_shell(1.2, 2.5)
    t_outer = _time_in_shell(3.0, 10.0)
    return dict(
        t_inner_s=float(t_inner), t_outer_s=float(t_outer), t_total_s=float(t_inner+t_outer),
        frac_inner=float(t_inner/T_s), frac_outer=float(t_outer/T_s),
        t_inner_total_s=float(t_inner*n_orbits), t_outer_total_s=float(t_outer*n_orbits),
        period_s=float(T_s),
    )



# ── Preset orbits (used by sidebar; does not require core library) ─────────────
PRESETS = {
    # ── LEO — Crewed ──────────────────────────────────────────────────────────
    "ISS (LEO)":               dict(a_km=RE_KM+408,     e=0.0003, inc_deg=51.60, raan_deg=0, argp_deg=0,   nu_deg=0),
    "Tiangong CSS (LEO)":      dict(a_km=RE_KM+370,     e=0.001,  inc_deg=41.50, raan_deg=0, argp_deg=0,   nu_deg=0),
    # ── LEO — Earth Observation ───────────────────────────────────────────────
    "Hubble (LEO)":            dict(a_km=RE_KM+540,     e=0.001,  inc_deg=28.47, raan_deg=0, argp_deg=0,   nu_deg=0),
    "Landsat 9 (SSO)":         dict(a_km=RE_KM+705,     e=0.001,  inc_deg=98.20, raan_deg=0, argp_deg=0,   nu_deg=0),
    "Sentinel-2A (SSO)":       dict(a_km=RE_KM+786,     e=0.001,  inc_deg=98.57, raan_deg=0, argp_deg=0,   nu_deg=0),
    "Terra / EOS AM (SSO)":    dict(a_km=RE_KM+705,     e=0.001,  inc_deg=98.20, raan_deg=0, argp_deg=0,   nu_deg=0),
    "ICESat-2 (SSO)":          dict(a_km=RE_KM+496,     e=0.001,  inc_deg=92.00, raan_deg=0, argp_deg=0,   nu_deg=0),
    "WorldView-3 (SSO)":       dict(a_km=RE_KM+617,     e=0.001,  inc_deg=97.90, raan_deg=0, argp_deg=0,   nu_deg=0),
    "JPSS-2 / NOAA-21 (SSO)":  dict(a_km=RE_KM+824,     e=0.001,  inc_deg=98.70, raan_deg=0, argp_deg=0,   nu_deg=0),
    # ── LEO — Communications ──────────────────────────────────────────────────
    "Starlink (LEO)":          dict(a_km=RE_KM+550,     e=0.001,  inc_deg=53.00, raan_deg=0, argp_deg=0,   nu_deg=0),
    "Iridium NEXT (LEO)":      dict(a_km=RE_KM+780,     e=0.001,  inc_deg=86.40, raan_deg=0, argp_deg=0,   nu_deg=0),
    # ── LEO — Science ─────────────────────────────────────────────────────────
    "Fermi / GLAST (LEO)":     dict(a_km=RE_KM+550,     e=0.001,  inc_deg=25.58, raan_deg=0, argp_deg=0,   nu_deg=0),
    "Swift (LEO)":             dict(a_km=RE_KM+600,     e=0.001,  inc_deg=20.56, raan_deg=0, argp_deg=0,   nu_deg=0),
    # ── SSO generic ───────────────────────────────────────────────────────────
    "Sun-Sync (SSO)":          dict(a_km=RE_KM+700,     e=0.001,  inc_deg=98.20, raan_deg=0, argp_deg=0,   nu_deg=0),
    # ── MEO ───────────────────────────────────────────────────────────────────
    "GPS Block III (MEO)":     dict(a_km=26_560.0,      e=0.010,  inc_deg=55.00, raan_deg=0, argp_deg=0,   nu_deg=0),
    "Galileo (MEO)":           dict(a_km=29_600.0,      e=0.001,  inc_deg=56.00, raan_deg=0, argp_deg=0,   nu_deg=0),
    "GLONASS (MEO)":           dict(a_km=25_508.0,      e=0.001,  inc_deg=64.80, raan_deg=0, argp_deg=0,   nu_deg=0),
    "BeiDou MEO":              dict(a_km=27_906.0,      e=0.001,  inc_deg=55.00, raan_deg=0, argp_deg=0,   nu_deg=0),
    # ── GEO ───────────────────────────────────────────────────────────────────
    "GEO (Generic)":           dict(a_km=42_164.0,      e=0.0001, inc_deg=0.05,  raan_deg=0, argp_deg=0,   nu_deg=0),
    "GOES-16 (GEO)":           dict(a_km=42_165.2,      e=0.0001, inc_deg=0.05,  raan_deg=0, argp_deg=0,   nu_deg=0),
    "GOES-17 (GEO)":           dict(a_km=42_165.0,      e=0.0001, inc_deg=0.04,  raan_deg=0, argp_deg=0,   nu_deg=0),
    # ── HEO ───────────────────────────────────────────────────────────────────
    "Molniya (HEO)":           dict(a_km=26_560.0,      e=0.720,  inc_deg=63.40, raan_deg=0, argp_deg=270, nu_deg=0),
    "Tundra (HEO)":            dict(a_km=42_164.0,      e=0.270,  inc_deg=63.40, raan_deg=0, argp_deg=270, nu_deg=0),
    "Chandra X-ray (HEO)":     dict(a_km=77_878.0,      e=0.790,  inc_deg=28.46, raan_deg=0, argp_deg=120, nu_deg=0),
    "TESS (HEO)":              dict(a_km=247_878.0,     e=0.540,  inc_deg=37.00, raan_deg=0, argp_deg=90,  nu_deg=0),
    # ── Cislunar / Lunar ──────────────────────────────────────────────────────
    "Cislunar Test Orbit":     dict(a_km=MOON_A_KM*0.85, e=0.03,  inc_deg=5.10,  raan_deg=0, argp_deg=0,   nu_deg=0),
    "Lunar Gateway (NRHO)":    dict(a_km=350_000.0,     e=0.950,  inc_deg=90.00, raan_deg=0, argp_deg=270, nu_deg=0),
    "Artemis lunar orbit":     dict(a_km=RE_KM+400_000, e=0.970,  inc_deg=90.00, raan_deg=0, argp_deg=180, nu_deg=0),
}

# NORAD catalog IDs — used to fetch live TLEs from Celestrak / Space-Track.
# Sourced from tle_updater.SATELLITE_GROUPS where available.
PRESET_NORAD: dict[str, int] = {
    "ISS (LEO)":               25544,
    "Tiangong CSS (LEO)":      54216,
    "Hubble (LEO)":            20580,
    "Landsat 9 (SSO)":         49260,
    "Sentinel-2A (SSO)":       40697,
    "Terra / EOS AM (SSO)":    25994,
    "ICESat-2 (SSO)":          43613,
    "Starlink (LEO)":          44235,
    "Fermi / GLAST (LEO)":     33053,
    "Swift (LEO)":             28485,
    "GPS Block III (MEO)":     44506,
    "Galileo (MEO)":           37846,
    "GLONASS (MEO)":           32276,
    "BeiDou MEO":              43706,
    "GOES-16 (GEO)":           41866,
    "GOES-17 (GEO)":           43226,
    "Chandra X-ray (HEO)":     25867,
    "TESS (HEO)":              43435,
}

# Human-readable info shown in the sidebar card when a preset is selected.
PRESET_INFO: dict[str, dict] = {
    "ISS (LEO)":               dict(regime="LEO",      alt="408 km",      desc="International Space Station — crewed, 51.6° inc"),
    "Tiangong CSS (LEO)":      dict(regime="LEO",      alt="370 km",      desc="Chinese Space Station — crewed, 41.5° inc"),
    "Hubble (LEO)":            dict(regime="LEO",      alt="540 km",       desc="Hubble Space Telescope — optical, 28.5° inc"),
    "Landsat 9 (SSO)":         dict(regime="SSO",      alt="705 km",       desc="USGS/NASA land imaging — 16-day repeat cycle"),
    "Sentinel-2A (SSO)":       dict(regime="SSO",      alt="786 km",       desc="ESA multispectral — 10 m resolution, 5-day revisit"),
    "Terra / EOS AM (SSO)":    dict(regime="SSO",      alt="705 km",       desc="NASA Earth Observing System — MODIS, ASTER, MOPITT"),
    "ICESat-2 (SSO)":          dict(regime="SSO",      alt="496 km",       desc="NASA laser altimetry — ice sheets, sea level"),
    "WorldView-3 (SSO)":       dict(regime="SSO",      alt="617 km",       desc="Maxar commercial imaging — 31 cm resolution"),
    "JPSS-2 / NOAA-21 (SSO)":  dict(regime="SSO",      alt="824 km",       desc="NOAA weather — VIIRS, CrIS, ATMS instruments"),
    "Starlink (LEO)":          dict(regime="LEO",      alt="550 km",       desc="SpaceX broadband constellation — 53° inc shell"),
    "Iridium NEXT (LEO)":      dict(regime="LEO",      alt="780 km",       desc="Global voice/data — 86.4° near-polar, 66 planes"),
    "Fermi / GLAST (LEO)":     dict(regime="LEO",      alt="550 km",       desc="NASA gamma-ray telescope — 25.6° inc, dark matter search"),
    "Swift (LEO)":             dict(regime="LEO",      alt="600 km",       desc="NASA GRB observatory — 20.6° inc, multi-wavelength"),
    "Sun-Sync (SSO)":          dict(regime="SSO",      alt="700 km",       desc="Generic sun-synchronous — dawn/dusk repeating groundtrack"),
    "GPS Block III (MEO)":     dict(regime="MEO",      alt="20 200 km",    desc="US Global Positioning System — 24+ satellites, 6 planes"),
    "Galileo (MEO)":           dict(regime="MEO",      alt="23 222 km",    desc="EU GNSS — 30 satellites, 56° inc, 3 orbital planes"),
    "GLONASS (MEO)":           dict(regime="MEO",      alt="19 130 km",    desc="Russian GNSS — 24 satellites, 64.8° inc, 3 planes"),
    "BeiDou MEO":              dict(regime="MEO",      alt="21 528 km",    desc="Chinese GNSS MEO component — 55° inc"),
    "GEO (Generic)":           dict(regime="GEO",      alt="35 786 km",    desc="Geostationary — zero inc, fixed longitude"),
    "GOES-16 (GEO)":           dict(regime="GEO",      alt="35 786 km",    desc="NOAA weather — Eastern US, 75.2°W, ABI imager"),
    "GOES-17 (GEO)":           dict(regime="GEO",      alt="35 786 km",    desc="NOAA weather — Western US, 137.2°W, ABI imager"),
    "Molniya (HEO)":           dict(regime="HEO",      alt="1 000–39 700 km", desc="Russian comms pattern — 12 hr, 63.4° inc, high Arctic dwell"),
    "Tundra (HEO)":            dict(regime="HEO",      alt="17 900–71 000 km", desc="24 hr Tundra — GEO SMA, 63.4° inc, one apogee per day"),
    "Chandra X-ray (HEO)":     dict(regime="HEO",      alt="10 000–133 000 km", desc="NASA X-ray observatory — 64 hr period, 28.5° inc"),
    "TESS (HEO)":              dict(regime="HEO",      alt="108 000–375 000 km", desc="NASA exoplanet survey — 2:1 lunar resonance, 37° inc"),
    "Cislunar Test Orbit":     dict(regime="Cislunar", alt="~327 000 km",   desc="Earth–Moon space test — 0.85× lunar distance"),
    "Lunar Gateway (NRHO)":    dict(regime="Cislunar", alt="~3 000–70 000 km (lunar)", desc="Near Rectilinear Halo Orbit — 7-day period around Moon"),
    "Artemis lunar orbit":     dict(regime="Cislunar", alt="~400 000 km",   desc="Artemis program — highly elliptical Earth–Moon transfer"),
}

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Space+Grotesk:wght@300;400;600;700&display=swap');

:root {
  --bg:     #12141C;  --panel:  #182C42;  --panel2: #1D3450;
  --green:  #2BFFB0;  --amber:  #FFC454;  --violet: #9B5CFF;
  --star:   #E4EEFA;  --dim:    #7290B0;  --danger: #FF7088;
  --border: rgba(43,255,176,0.24);
}
.stApp { background: var(--bg) !important; }
html, body, [class*="css"] { font-family:'Space Grotesk',sans-serif; color:var(--star); }

[data-testid="stSidebar"] { background:var(--panel) !important; border-right:1px solid var(--border); }
[data-testid="stSidebar"] * { color:var(--star) !important; }
[data-testid="stSidebar"] label { color:var(--dim) !important; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; }
[data-testid="stSidebar"] [data-testid="stToggle"] label,
[data-testid="stSidebar"] [data-testid="stToggle"] p {
  color:#FFFFFF !important; font-size:0.85rem !important;
  text-transform:none !important; letter-spacing:0 !important; font-weight:500 !important; }

/* Toggle/checkbox labels anywhere outside the sidebar (e.g. the "Preview
   layers" list, Sensor FOV options) otherwise fall back to Streamlit's
   default dark-grey label color, which reads as dim/illegible against this
   theme — force them bright like the rest of the UI. */
[data-testid="stToggle"] label, [data-testid="stToggle"] p,
[data-testid="stCheckbox"] label, [data-testid="stCheckbox"] p {
  color:var(--star) !important; font-weight:500 !important; }

.sidebar-sec { border-top:1px solid var(--border); padding-top:0.5rem; margin-top:0.4rem;
  font-size:0.62rem; text-transform:uppercase; letter-spacing:0.18em;
  color:var(--green) !important; font-family:'JetBrains Mono',monospace; }

.readout { background:#0B1626; border:1px solid var(--border); border-left:3px solid var(--green);
  border-radius:6px; padding:0.8rem 1rem; font-family:'JetBrains Mono',monospace;
  font-size:0.71rem; line-height:1.85; color:var(--green); margin-bottom:0.8rem; }
.r-label{color:var(--dim)} .r-val{color:var(--green)} .r-unit{color:var(--amber);font-size:0.65rem}
.r-warn{color:var(--amber)} .r-err{color:var(--danger)} .r-head{color:var(--star);letter-spacing:0.12em;border-bottom:1px solid var(--border);padding-bottom:0.3rem;margin-bottom:0.4rem}
.r-ok{color:var(--green)}

.plot-card { background:var(--panel); border:1px solid var(--border); border-radius:10px;
  padding:1rem 1.1rem; margin-bottom:0.8rem; }
.plot-card.active { border-color:var(--green); }
.plot-card .card-title { font-size:0.83rem; font-weight:700; letter-spacing:0.06em;
  color:var(--star); text-transform:uppercase; margin-bottom:0.2rem; }
.plot-card.active .card-title { color:var(--green); }
.plot-card .card-desc { font-size:0.72rem; color:var(--dim); margin-bottom:0.5rem; line-height:1.5; }

.badge { display:inline-block; font-family:'JetBrains Mono',monospace; font-size:0.59rem;
  padding:0.12rem 0.4rem; border-radius:20px; margin-right:0.3rem; margin-bottom:0.4rem; letter-spacing:0.06em; }
.b-plotly{background:rgba(155,92,255,0.2);color:var(--violet);border:1px solid var(--violet)}
.b-mpl{background:rgba(255,196,84,0.12);color:var(--amber);border:1px solid var(--amber)}
.b-html{background:rgba(43,255,176,0.1);color:var(--green);border:1px solid var(--green)}
.b-png{background:rgba(228,238,250,0.1);color:var(--star);border:1px solid var(--dim)}

.fidelity-fast{color:var(--amber);font-family:'JetBrains Mono',monospace;font-size:0.72rem}
.fidelity-hi{color:var(--green);font-family:'JetBrains Mono',monospace;font-size:0.72rem}
.fidelity-load{color:#00BFFF;font-family:'JetBrains Mono',monospace;font-size:0.72rem}

.console { background:#0B1626; border:1px solid var(--border); border-radius:6px;
  padding:1rem; font-family:'JetBrains Mono',monospace; font-size:0.71rem;
  color:var(--green); min-height:100px; max-height:280px; overflow-y:auto;
  line-height:1.7; white-space:pre-wrap; }
.c-ok{color:var(--green)} .c-err{color:var(--danger)} .c-warn{color:var(--amber)}
.c-info{color:var(--star)} .c-dim{color:var(--dim)}

.page-title { font-family:'JetBrains Mono',monospace; font-size:1.35rem; font-weight:700;
  color:var(--green); letter-spacing:0.15em; text-transform:uppercase; }
.page-sub { font-size:0.71rem; color:var(--dim); letter-spacing:0.1em;
  margin-bottom:1rem; font-family:'JetBrains Mono',monospace; }

.warn-box { background:rgba(255,196,84,0.07); border:1px solid rgba(255,196,84,0.3);
  border-left:3px solid var(--amber); border-radius:5px; padding:0.55rem 0.8rem;
  font-size:0.73rem; color:var(--amber); font-family:'JetBrains Mono',monospace; margin:0.3rem 0; }

.mission-card { background:var(--panel2); border:1px solid var(--border);
  border-radius:8px; padding:0.9rem 1rem; margin-bottom:0.7rem; }
.mission-card h4 { font-size:0.8rem; text-transform:uppercase; letter-spacing:0.1em;
  color:var(--green); margin:0 0 0.5rem; font-family:'JetBrains Mono',monospace; }

.burn-card { background:rgba(255,196,84,0.05); border:1px solid rgba(255,196,84,0.2);
  border-radius:6px; padding:0.7rem 0.9rem; margin:0.4rem 0; }

hr { border-color:var(--border) !important; }

.stNumberInput input,.stTextInput input,.stTextArea textarea {
  background:#0B1626 !important; border:1px solid var(--border) !important;
  border-radius:5px !important; color:var(--green) !important;
  font-family:'JetBrains Mono',monospace !important; font-size:0.81rem !important; }
.stSelectbox>div>div { background:#0B1626 !important; border:1px solid var(--border) !important; }
.stSlider>div>div>div>div { background:var(--green) !important; }
.stExpander { border:1px solid var(--border) !important; border-radius:6px !important; background:var(--panel2) !important; }
.stButton>button { background:transparent !important; border:1px solid var(--green) !important;
  color:var(--green) !important; font-family:'JetBrains Mono',monospace !important;
  font-size:0.77rem !important; letter-spacing:0.1em !important; border-radius:5px !important;
  padding:0.4rem 1.1rem !important; transition:background 0.15s !important; }
.stButton>button:hover { background:rgba(43,255,176,0.08) !important; }
.stTabs [data-baseweb="tab-list"] { background:transparent !important; border-bottom:1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] { color:var(--dim) !important; font-family:'JetBrains Mono',monospace !important;
  font-size:0.73rem !important; letter-spacing:0.1em !important; text-transform:uppercase !important; }
.stTabs [aria-selected="true"] { color:var(--green) !important; border-bottom:2px solid var(--green) !important; }

.run-btn>button { background:var(--green) !important; color:#12141C !important;
  font-weight:700 !important; font-size:0.84rem !important; width:100%; border:none !important; }
.run-btn>button:hover { background:#17FFC2 !important; }

/* ── Bordered containers (st.container(border=True)) ───────────────────────
   Streamlit's default border-only styling renders too faint against this
   dark theme and gives no background fill, so cards using it (Mission
   Planner steps, Burns, Transfer Calculator, Save tab) looked "boxless"
   next to older markdown-div cards that did have a visible fill. Forcing
   the same navy background + border here makes every card look identical
   regardless of which mechanism built it. */
div[data-testid="stVerticalBlockBorderWrapper"] > div,
div[data-testid="stContainer"] {
  background: var(--panel2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
  margin-bottom: 0.7rem !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
  padding: 0.9rem 1rem !important;
}

/* ── Responsive / compact sidebar — fits any screen resolution ────────────── */
[data-testid="stSidebar"] { min-width: 260px !important; max-width: 340px !important; }
[data-testid="stSidebar"] .block-container { padding-top: 0.6rem !important; }
[data-testid="stSidebar"] label { font-size: 0.66rem !important; margin-bottom: 0.1rem !important; }
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea { font-size: 0.72rem !important; padding: 0.3rem 0.5rem !important; }
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] { font-size: 0.72rem !important; min-height: 2rem !important; }
[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] { gap: 0.25rem !important; }
[data-testid="stSidebar"] .stExpander summary { padding: 0.35rem 0.6rem !important; font-size: 0.74rem !important; }
[data-testid="stSidebar"] [data-testid="stToggle"] { transform: scale(0.85); transform-origin: left center; }
[data-testid="stSidebar"] hr { margin: 0.35rem 0 !important; }
[data-testid="stSidebar"] .sidebar-sec { margin-top: 0.3rem !important; padding-top: 0.35rem !important; }

/* Narrow viewport (small laptops / split screens): shrink further and let the
   sidebar scroll instead of overflowing the page. */
@media (max-width: 1100px) {
  [data-testid="stSidebar"] { min-width: 220px !important; max-width: 280px !important; }
  [data-testid="stSidebar"] label { font-size: 0.6rem !important; }
  .page-title { font-size: 1.05rem !important; }
  .stTabs [data-baseweb="tab"] { font-size: 0.62rem !important; padding: 0.3rem 0.5rem !important; }
}
@media (max-width: 768px) {
  [data-testid="stSidebar"] { min-width: 180px !important; max-width: 100% !important; }
}
/* Sidebar itself scrolls independently so tall panels never blow out the
   viewport on smaller displays. */
[data-testid="stSidebar"] > div:first-child {
  max-height: 100vh; overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
def init_state():
    defs = dict(
        # input mode
        input_mode="Keplerian", preset="Custom",
        # elements
        a_km=RE_KM+550, e=0.001, inc_deg=51.6, raan_deg=0.0, argp_deg=0.0, nu_deg=0.0,
        tle_text="",
        # propagation config
        propagator="keplerian", gravity="j2", third_body="none", non_grav="none",
        cd=2.2, area_m2=10.0, mass_kg=500.0, cr=1.3,
        n_orbits=3.0, dt_s=60.0,
        epoch=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        # display
        frame="ECI", show_osculating=False,
        # live preview layers
        lyr_earth=True, lyr_stars=False, lyr_moon=False, lyr_groundtrack=False,
        lyr_terminator=False, lyr_eclipse=False, lyr_van_allen=False,
        lyr_lagrange=False, lyr_sun=False,
        lyr_ntw=False,
        # paths
        toolkit_dir=str(Path.home() / "SSAPy-Toolkit"),
        ssapy_dir=str(Path.home() / "SSAPy"),
        star_catalog=str(Path.home() / "bright_stars.csv"),
        earth_texture=(str(_here / "earth.png") if (_here / "earth.png").exists()
                       else str(Path.home() / "SSAPy" / "ssapy" / "data" / "earth.png")),
        output_dir=str(Path.home() / "SSAPy-Toolkit" / "output"),
        conda_env="myenv",
        # plot module enables
        mod_orbit_xy=False, mod_orbit_full=False, mod_globe=False,
        mod_cislunar_3d=False, mod_cislunar_combo=False, mod_moon_3d=False,
        mod_magfield=False, mod_van_allen=False,
        # plot module settings
        ps_orbit_xy=dict(show_stars=True, show_earth=True, show_moon=True, n_frames=48),
        ps_orbit_full=dict(show_stars=True, dark_bg=True),
        ps_globe=dict(show_stars=True, show_earth=True),
        ps_cislunar_3d=dict(show_stars=True, dark_bg=True, show_lagrange=True),
        ps_cislunar_combo=dict(show_stars=True, dark_bg=True),
        ps_moon_3d=dict(show_surface=True, show_orbit=True, show_polar=True),
        ps_magfield=dict(show_van_allen=True, show_dipole_axis=True, show_stars=True,
                         max_r_re=15.0, seed_lats=[20,30,40,55,65,75],
                         show_belt_residence=True),
        ps_van_allen=dict(show_inner=True, show_outer=True, views=["oblique","equatorial","polar"]),
        # mission planner
        mp_targets=[],
        mp_objective="min_dv", mp_optimizer="greedy",
        mp_dv_budget=2000.0, mp_tof_min=0.5, mp_tof_max=24.0,
        mp_geometry="insertion", mp_results=None,
        # burns
        burns=[],  # list of dicts
        # console
        console_log=[],
        # scene cache key
        _scene_key=0,
        # starfield epoch
        starfield_epoch_mode="auto",    # "auto" = use orbit epoch, "custom" = date picker, "far_future" = year slider
        starfield_epoch_date=None,      # datetime.date when mode == "custom"
        sm_far_year=2000,               # year when mode == "far_future"
        # solar view
        sol_year=2025,
        sol_month=1,
        sol_show_mercury=False,
        sol_show_venus=True,
        sol_show_earth=True,
        sol_show_mars=True,
        sol_show_jupiter=True,
        sol_show_saturn=True,
        sol_show_uranus=False,
        sol_show_neptune=False,
        sol_show_moon=True,
        sol_show_trails=True,
        sol_show_stars=True,
        sol_show_ecliptic=True,
        sol_show_labels=True,
        # sensor FOV
        fov_enabled=False,
        fov_pointing_mode="nadir",
        fov_half_angle_deg=15.0,
        fov_cone_length_km=8_000.0,
        fov_color="#00CED1",
        fov_opacity=0.35,
        fov_show_boresight=True,
        fov_time_index=0,
        fov_custom_x=1.0,
        fov_custom_y=0.0,
        fov_custom_z=0.0,
        fov_animate=False,
        fov_anim_step=15,
        fov_show_sun_shading=True,
        fov_show_footprint=True,
        # scene scale
        preview_scale="Auto",
        preview_scale_custom_km=20_000.0,
        # multi-satellite
        extra_satellites=[],
        # persistent "full propagation" mode so extra sats / FOV anim don't
        # disappear on the next rerun (fixes multi-sat display bug)
        _preview_full_mode=False,
        # unified 3D-view plot switcher (3D PREVIEW tab)
        preview_plot_view="🛰 Orbit Scene",
        # save / export tab
        save_name="ssapy_export",
        save_dir=str(Path.home() / "yu_figures" / "demo_gallery" / "figures"),
        save_fmt_json=True, save_fmt_csv=False, save_fmt_hdf5=False,
        save_log=[],
        # mission planner — transfer calculator
        mp_xfer_target_idx=0,
        mp_xfer_type="Hohmann",
    )
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
ss = st.session_state


# ── Starfield epoch → Julian Date ─────────────────────────────────────────────
def _get_starfield_epoch_jd() -> float | None:
    """Return Julian Date for starfield precession, or None for J2000.0.

    Modes:
      "auto"       — parse the orbit epoch string
      "custom"     — user-selected date (up to year 9999)
      "far_future" — year slider (2000–14 000)
    """
    try:
        import datetime as _dt
        mode = ss.starfield_epoch_mode

        if mode == "far_future":
            yr = float(ss.get("sm_far_year", 2000))
            # JD = J2000 + (year - 2000) * 365.25
            return 2_451_545.0 + (yr - 2000.0) * 365.25

        if mode == "custom" and ss.starfield_epoch_date is not None:
            d   = ss.starfield_epoch_date
            ref = _dt.date(2000, 1, 1)
            return 2_451_545.0 + (d - ref).days

        # Auto — parse orbit epoch string
        epoch_str = ss.epoch.strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                dt  = _dt.datetime.strptime(epoch_str, fmt)
                ref = _dt.datetime(2000, 1, 1, 12, 0, 0)
                return 2_451_545.0 + (dt - ref).total_seconds() / 86400.0
            except ValueError:
                continue
    except Exception:
        pass
    return None   # fall back to J2000


# ── Precession matrix (standalone, no core dependency) ────────────────────────
def _prec_matrix(epoch_jd: float):
    """IAU 1976 precession from J2000.0 to epoch_jd."""
    import numpy as _np
    T = (epoch_jd - 2_451_545.0) / 36_525.0
    zeta  = ((2306.2181 + 1.39656*T)*T + 0.30188*T**2 + 0.017998*T**3)
    z     = ((2306.2181 + 1.39656*T)*T + 1.09468*T**2 + 0.018203*T**3)
    theta = ((2004.3109 - 0.85330*T)*T - 0.42665*T**2 - 0.041775*T**3)
    zr, zeta_r, th_r = (_np.radians(v/3600) for v in (z, zeta, theta))
    # R_z(-z_A)
    Rz_z    = _np.array([[ _np.cos(zr),  _np.sin(zr), 0],
                          [-_np.sin(zr),  _np.cos(zr), 0], [0, 0, 1]])
    # R_y(+theta_A)  — note: +sin in [0,2], -sin in [2,0]
    Ry_th   = _np.array([[ _np.cos(th_r), 0,  _np.sin(th_r)],
                          [0,              1,  0             ],
                          [-_np.sin(th_r), 0,  _np.cos(th_r)]])
    # R_z(-zeta_A)
    Rz_zeta = _np.array([[ _np.cos(zeta_r),  _np.sin(zeta_r), 0],
                          [-_np.sin(zeta_r),  _np.cos(zeta_r), 0], [0, 0, 1]])
    return Rz_z @ Ry_th @ Rz_zeta


_REFERENCE_STARS = [
    ("Sirius",      6.7525,  -16.716, -1.46, "#ff2244"),
    ("Canopus",     6.3992,  -52.696, -0.72, "#ff9900"),
    ("Arcturus",   14.2610,  +19.182, -0.04, "#00ffcc"),
    ("Vega",       18.6156,  +38.784, +0.03, "#ffff00"),
    ("Capella",     5.2780,  +45.998, +0.08, "#00aaff"),
    ("Rigel",       5.2423,   -8.202, +0.12, "#ff66ff"),
    ("Betelgeuse",  5.9194,   +7.407, +0.42, "#aaffaa"),
    ("Polaris",     2.5303,  +89.264, +1.97, "#ffffff"),
]

# ── Star catalog search ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_star_catalog_df(catalog_path: str):
    """Load and cache HYG catalog as a DataFrame. Result is reused across reruns."""
    import pandas as pd
    from pathlib import Path as _P
    if not _P(catalog_path).exists():
        return None
    try:
        df = pd.read_csv(catalog_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(catalog_path, encoding="latin-1")
    for _col in ("proper", "bayer", "bf", "spect", "con", "var"):
        if _col in df.columns:
            df[_col] = df[_col].fillna("")
    for _col in ("hr", "hd", "hip"):
        if _col in df.columns:
            df[_col] = pd.to_numeric(df[_col], errors="coerce")
    return df


def _search_star_catalog(query: str, catalog_path: str, max_results: int = 12):
    """
    Search HYG catalog by proper name, Bayer designation, constellation,
    or catalog number (HR / HD / HIP).

    Returns a list of dicts sorted by magnitude (brightest first).
    """
    import math
    df = _load_star_catalog_df(catalog_path)
    if df is None or len(df) == 0:
        return []

    q = query.strip()
    q_up = q.upper()
    mask = None

    # ── Catalog-number prefix searches (exact) ─────────────────────────────────
    for prefix, col in (("HR ", "hr"), ("HD ", "hd"), ("HIP ", "hip"), ("GL ", "gl")):
        if q_up.startswith(prefix) and col in df.columns:
            num_str = q_up[len(prefix):].strip()
            try:
                num = float(num_str)
                mask = df[col] == num
            except ValueError:
                pass
            break

    # ── Fuzzy name / constellation search ─────────────────────────────────────
    if mask is None:
        q_lo = q.lower()
        parts = []
        for col in ("proper", "bayer", "bf", "con"):
            if col in df.columns:
                parts.append(df[col].str.lower().str.contains(q_lo, na=False, regex=False))
        if parts:
            import functools
            mask = functools.reduce(lambda a, b: a | b, parts)

    if mask is None or mask.sum() == 0:
        return []

    hits = df[mask].copy()
    if "mag" in hits.columns:
        hits = hits.sort_values("mag").head(max_results)
    else:
        hits = hits.head(max_results)

    results = []
    for _, row in hits.iterrows():
        # Best display name
        name = (row.get("proper", "") or
                row.get("bayer", "") or
                row.get("bf", "") or
                (f"HD {int(row['hd'])}" if "hd" in row and not math.isnan(row.get("hd", float("nan"))) else "") or
                (f"HIP {int(row['hip'])}" if "hip" in row and not math.isnan(row.get("hip", float("nan"))) else "?"))
        dist_pc = row.get("dist", None)
        dist_ly = float(dist_pc) * 3.2616 if dist_pc and float(dist_pc) > 0 else None
        results.append(dict(
            name    = name,
            ra_h    = float(row.get("ra",  0)),
            dec_deg = float(row.get("dec", 0)),
            mag     = float(row.get("mag", 99)),
            spect   = str(row.get("spect", "?"))[:3] or "?",
            dist_ly = dist_ly,
            con     = str(row.get("con", "")),
            bayer   = str(row.get("bayer", "")),
            hr      = int(row["hr"])  if "hr"  in row and not math.isnan(row.get("hr",  float("nan"))) else None,
            hd      = int(row["hd"])  if "hd"  in row and not math.isnan(row.get("hd",  float("nan"))) else None,
            hip     = int(row["hip"]) if "hip" in row and not math.isnan(row.get("hip", float("nan"))) else None,
        ))
    return results


def _add_search_highlights(fig, stars: list, epoch_jd):
    """
    Overlay yellow highlight rings + precession arrows on an existing star map fig
    for each star dict in *stars*.  Returns the figure (modified in place).
    """
    import plotly.graph_objects as go

    P   = _prec_matrix(epoch_jd) if epoch_jd is not None else None
    R   = 1.0
    HL  = "#FFFF00"   # highlight colour

    for star in stars:
        ra_h    = star["ra_h"]
        dec_deg = star["dec_deg"]
        name    = star["name"]
        v0      = _ra_dec_gcrf(ra_h, dec_deg)

        hover_j2000 = (
            f"★ {name}<br>"
            f"RA {ra_h:.4f} h  Dec {dec_deg:+.3f}°<br>"
            f"mag {star['mag']:.2f}  spec {star['spect']}"
            + (f"<br>dist {star['dist_ly']:.1f} ly" if star.get("dist_ly") else "")
            + (f"<br>HR {star['hr']}" if star.get("hr") else "")
            + (f"  HD {star['hd']}" if star.get("hd") else "")
            + (f"  HIP {star['hip']}" if star.get("hip") else "")
        )

        # J2000 — large yellow ring
        fig.add_trace(go.Scatter3d(
            x=[v0[0]*R], y=[v0[1]*R], z=[v0[2]*R],
            mode="markers+text",
            marker=dict(size=14, color="rgba(0,0,0,0)",
                        line=dict(color=HL, width=3)),
            text=[f"★ {name}"], textposition="top center",
            textfont=dict(color=HL, size=11, family="JetBrains Mono,monospace"),
            hovertext=hover_j2000, hoverinfo="text",
            name=f"★ {name}", showlegend=True,
            legendgroup="search_hl",
            legendgrouptitle_text="Search results" if stars.index(star) == 0 else None,
        ))

        if P is not None:
            vp = P @ v0
            dot = float(np.clip(np.dot(v0, vp), -1.0, 1.0))
            shift_as = np.degrees(np.arccos(dot)) * 3600.0

            # Precessed — filled yellow dot
            fig.add_trace(go.Scatter3d(
                x=[vp[0]*R], y=[vp[1]*R], z=[vp[2]*R],
                mode="markers",
                marker=dict(size=10, color=HL, opacity=0.95),
                hovertext=f"★ {name} (precessed)<br>shift {shift_as:.1f}\"",
                hoverinfo="text",
                name=f"★ {name} precessed", showlegend=False,
            ))

            # Shift line
            fig.add_trace(go.Scatter3d(
                x=[v0[0], vp[0]], y=[v0[1], vp[1]], z=[v0[2], vp[2]],
                mode="lines",
                line=dict(color=HL, width=4),
                hoverinfo="skip", showlegend=False,
            ))

    return fig

_SPECT_COLORS_PLOTLY = {
    "O":"rgb(155,175,255)","B":"rgb(171,191,255)","A":"rgb(201,217,255)",
    "F":"rgb(247,247,255)","G":"rgb(255,245,235)","K":"rgb(255,209,160)","M":"rgb(255,204,112)",
}


def _ra_dec_gcrf(ra_h, dec_deg):
    ra_r  = np.radians(ra_h * 15.0)
    dec_r = np.radians(dec_deg)
    return np.array([np.cos(dec_r)*np.cos(ra_r),
                     np.cos(dec_r)*np.sin(ra_r),
                     np.sin(dec_r)])


def build_star_accuracy_fig(epoch_jd, catalog_path, mag_limit=6.5):
    """Return (fig, shift_rows) where shift_rows is a list of
    (name, shift_arcsec) for the readout table."""
    import plotly.graph_objects as go

    R   = 1.0
    P   = _prec_matrix(epoch_jd) if epoch_jd is not None else None
    fig = go.Figure()

    # ── Catalog stars ──────────────────────────────────────────────────────────
    cat_loaded = False
    try:
        import pandas as pd
        from pathlib import Path as _Path
        if _Path(catalog_path).exists():
            df = pd.read_csv(catalog_path)
            df = df[(df["mag"] < mag_limit) & (df["mag"] > -10)].dropna(
                subset=["ra","dec","mag"])
            ra_r  = np.radians(df["ra"].values * 15.0)
            dec_r = np.radians(df["dec"].values)
            cx0 = np.cos(dec_r)*np.cos(ra_r)
            cy0 = np.cos(dec_r)*np.sin(ra_r)
            cz0 = np.sin(dec_r)
            mag    = df["mag"].values
            spect  = df["spect"].fillna("G").str[:1].values
            names_col = (df["proper"].fillna("") if "proper" in df.columns
                         else pd.Series([""] * len(df))).values
            sizes  = np.clip(0.5*(mag_limit - mag)**1.1, 0.3, 5.0)
            colors = [_SPECT_COLORS_PLOTLY.get(s, _SPECT_COLORS_PLOTLY["G"]) for s in spect]
            hover  = [f"{n if n else '—'}<br>mag={m:.2f}"
                      for n,m in zip(names_col, mag)]

            # J2000 catalog (dim reference layer)
            fig.add_trace(go.Scatter3d(
                x=cx0*R, y=cy0*R, z=cz0*R,
                mode="markers",
                marker=dict(size=sizes*0.5, color=colors, opacity=0.25),
                text=hover, hoverinfo="text",
                name="Catalog — J2000.0",
                legendgroup="cat_j2000",
            ))

            # Precessed catalog (if epoch set)
            if P is not None:
                vp = P @ np.stack([cx0,cy0,cz0])
                fig.add_trace(go.Scatter3d(
                    x=vp[0]*R, y=vp[1]*R, z=vp[2]*R,
                    mode="markers",
                    marker=dict(size=sizes*0.9, color=colors, opacity=0.85),
                    text=hover, hoverinfo="text",
                    name="Catalog — epoch precessed",
                    legendgroup="cat_prec",
                ))
            cat_loaded = True
    except Exception as _ce:
        pass  # catalog missing — still show reference stars

    # ── Reference stars ────────────────────────────────────────────────────────
    shift_rows = []
    for name, ra_h, dec_deg, mag_v, color in _REFERENCE_STARS:
        v0 = _ra_dec_gcrf(ra_h, dec_deg)   # J2000

        # J2000 marker (open circle)
        fig.add_trace(go.Scatter3d(
            x=[v0[0]], y=[v0[1]], z=[v0[2]],
            mode="markers+text",
            marker=dict(size=9, color="rgba(0,0,0,0)",
                        line=dict(color=color, width=2), opacity=0.8),
            text=[f"{name} J2000"], textposition="top center",
            textfont=dict(color=color, size=9),
            hovertext=f"{name}  J2000<br>RA {ra_h:.4f}h  Dec {dec_deg:+.3f}°<br>mag {mag_v:.2f}",
            hoverinfo="text",
            name=f"{name}",
            legendgroup="refs",
            legendgrouptitle_text="Reference stars" if name == "Sirius" else None,
        ))

        shift_arcsec = 0.0
        if P is not None:
            vp = P @ v0                    # precessed
            dot = float(np.clip(np.dot(v0, vp), -1.0, 1.0))
            shift_arcsec = np.degrees(np.arccos(dot)) * 3600.0

            # Precessed marker (filled)
            fig.add_trace(go.Scatter3d(
                x=[vp[0]], y=[vp[1]], z=[vp[2]],
                mode="markers",
                marker=dict(size=9, color=color, opacity=1.0),
                hovertext=(f"{name}  epoch-precessed<br>"
                           f"RA {ra_h:.4f}h  Dec {dec_deg:+.3f}°<br>"
                           f"Shift: {shift_arcsec:.1f}\u201d"),
                hoverinfo="text",
                name=f"{name} (precessed)",
                legendgroup="refs_prec",
                showlegend=(name == "Sirius"),
            ))

            # Shift arrow J2000 → precessed
            fig.add_trace(go.Scatter3d(
                x=[v0[0], vp[0]], y=[v0[1], vp[1]], z=[v0[2], vp[2]],
                mode="lines",
                line=dict(color=color, width=3),
                hoverinfo="skip", showlegend=False,
            ))

        # Depth-range dotted line
        fig.add_trace(go.Scatter3d(
            x=[v0[0]*0.48, v0[0]*1.02], y=[v0[1]*0.48, v0[1]*1.02],
            z=[v0[2]*0.48, v0[2]*1.02],
            mode="lines",
            line=dict(color=color, width=1, dash="dot"),
            hoverinfo="skip", showlegend=False,
        ))

        shift_rows.append((name, ra_h, dec_deg, mag_v, shift_arcsec, color))

    # ── Principal axes ─────────────────────────────────────────────────────────
    for axis, label, col in [
        ([1,0,0], "X  RA=0h  (vernal equinox)", "#ff4444"),
        ([0,1,0], "Y  RA=6h", "#44ff44"),
        ([0,0,1], "Z  NCP  (Polaris direction)", "#4444ff"),
    ]:
        fig.add_trace(go.Scatter3d(
            x=[0, axis[0]*1.15], y=[0, axis[1]*1.15], z=[0, axis[2]*1.15],
            mode="lines+text",
            line=dict(color=col, width=3),
            text=["", label], textposition="top center",
            textfont=dict(color=col, size=9),
            hoverinfo="skip", name=label,
        ))

    # ── Galactic equator ───────────────────────────────────────────────────────
    gnp = _ra_dec_gcrf(192.85/15.0, 27.13)
    b1  = np.cross(gnp, [0,1,0]); b1 /= np.linalg.norm(b1)
    b2  = np.cross(gnp, b1);      b2 /= np.linalg.norm(b2)
    th  = np.linspace(0, 2*np.pi, 200)
    mw  = np.outer(np.cos(th), b1) + np.outer(np.sin(th), b2)
    fig.add_trace(go.Scatter3d(
        x=mw[:,0]*1.01, y=mw[:,1]*1.01, z=mw[:,2]*1.01,
        mode="lines", line=dict(color="#8899dd", width=1),
        opacity=0.35, name="Galactic equator", hoverinfo="skip",
    ))

    # ── Layout ─────────────────────────────────────────────────────────────────
    if epoch_jd is not None:
        T_yr = (epoch_jd - 2_451_545.0) / 365.25
        epoch_label = f"Epoch {T_yr:+.2f} yr from J2000"
    else:
        epoch_label = "J2000.0 (no precession)"

    subtitle = (
        "Open circle = J2000 position  |  Filled circle = epoch position  |  "
        "Line = precession shift<br>"
        "Catalog speckles should align with the filled circles."
        if P is not None else
        "No epoch set — showing J2000 positions only."
    )

    fig.update_layout(
        title=dict(
            text=f"Star accuracy — {epoch_label}<br><sup>{subtitle}</sup>",
            font=dict(color="white", size=13), x=0.5,
        ),
        paper_bgcolor="#101F32",
        scene=dict(
            bgcolor="#101F32",
            xaxis=dict(range=[-1.2,1.2], showbackground=False,
                       gridcolor="rgba(255,255,255,0.05)",
                       color="rgba(255,255,255,0.3)", title="X (GCRF)"),
            yaxis=dict(range=[-1.2,1.2], showbackground=False,
                       gridcolor="rgba(255,255,255,0.05)",
                       color="rgba(255,255,255,0.3)", title="Y (GCRF)"),
            zaxis=dict(range=[-1.2,1.2], showbackground=False,
                       gridcolor="rgba(255,255,255,0.05)",
                       color="rgba(255,255,255,0.3)", title="Z (GCRF)"),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=-1.2, z=0.7),
                        up=dict(x=0, y=0, z=1)),
        ),
        legend=dict(
            font=dict(color="white", size=9),
            bgcolor="rgba(12,16,28,0.85)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1, x=0.01, y=0.99,
        ),
        margin=dict(l=0, r=0, t=90, b=0),
    )
    return fig, shift_rows



# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def compute_derived():
    a, e = ss.a_km, ss.e
    r_a = a*(1+e); r_p = a*(1-e)
    T_s = 2*np.pi*np.sqrt(a**3/MU) if a > 0 else 0
    v_p = np.sqrt(MU*(2/r_p - 1/a)) if r_p > 0 and e < 1 else 0
    v_a = np.sqrt(MU*(2/r_a - 1/a)) if r_a > 0 and e < 1 else 0
    ha, hp = r_a-RE_KM, r_p-RE_KM
    if ha < 2000:               regime = "LEO"
    elif ha < 35000:            regime = "MEO"
    elif abs(ha-35786) < 750:   regime = "GEO"
    elif a > MOON_A_KM*0.4:    regime = "CISLUNAR"
    else:                       regime = "HEO"
    warns = []
    if hp < 0:     warns.append("✖ Periapsis below Earth surface")
    elif hp < 150: warns.append("⚠ Periapsis <150 km — re-entry likely")
    if e >= 1:     warns.append("✖ e ≥ 1.0 — hyperbolic trajectory")
    # J2 drifts
    n = 2*np.pi/T_s if T_s > 0 else 0
    p = a*(1-e**2)
    cos_i = np.cos(np.radians(ss.inc_deg))
    sin2_i = np.sin(np.radians(ss.inc_deg))**2
    J2 = 1.08262668e-3
    raan_drift = np.degrees(-1.5*n*J2*(RE_KM/p)**2*cos_i)*86400 if p > 0 else 0
    argp_drift = np.degrees(0.75*n*J2*(RE_KM/p)**2*(5*sin2_i-4))*86400 if p > 0 else 0
    return dict(T_s=T_s, T_min=T_s/60, T_hr=T_s/3600,
                h_a=ha, h_p=hp, v_p=v_p, v_a=v_a,
                regime=regime, warns=warns,
                raan_drift=raan_drift, argp_drift=argp_drift)


def orbit_readout_html(d):
    fi = ss.get("_fidelity","fast")
    fbadge = {
        "fast":    '<span class="fidelity-fast">⚡ Fast preview</span>',
        "loading": '<span class="fidelity-load">⟳ Loading hi-fidelity…</span>',
        "high":    '<span class="fidelity-hi">✓ Hi-fidelity (IERS)</span>',
    }.get(fi, "")
    T_str = f"{d['T_hr']:.3f} hr ({d['T_min']:.1f} min)"
    warns = "".join(f'<div class="r-warn">{w}</div>' for w in d["warns"])
    return f"""
<div class="readout">
  <div class="r-head">▸ ORBIT READOUT  {fbadge}</div>
  <span class="r-label">REGIME   </span><span class="r-val">{d['regime']}</span><br>
  <span class="r-label">SMA      </span><span class="r-val">{ss.a_km:,.1f}</span><span class="r-unit"> km ({ss.a_km/RE_KM:.4f} Rₑ)</span><br>
  <span class="r-label">ECC      </span><span class="r-val">{ss.e:.6f}</span><br>
  <span class="r-label">INC      </span><span class="r-val">{ss.inc_deg:.3f}</span><span class="r-unit"> deg</span><br>
  <span class="r-label">RAAN     </span><span class="r-val">{ss.raan_deg:.3f}</span><span class="r-unit"> deg</span><br>
  <span class="r-label">ARGP     </span><span class="r-val">{ss.argp_deg:.3f}</span><span class="r-unit"> deg</span><br>
  <span class="r-label">NU       </span><span class="r-val">{ss.nu_deg:.3f}</span><span class="r-unit"> deg</span><br>
  <hr style="border-color:rgba(43,255,176,0.1);margin:5px 0">
  <span class="r-label">PERIOD   </span><span class="r-val">{T_str}</span><br>
  <span class="r-label">ALT_P    </span><span class="r-val">{d['h_p']:,.1f}</span><span class="r-unit"> km</span><br>
  <span class="r-label">ALT_A    </span><span class="r-val">{d['h_a']:,.1f}</span><span class="r-unit"> km</span><br>
  <span class="r-label">V_PERI   </span><span class="r-val">{d['v_p']:.3f}</span><span class="r-unit"> km/s</span><br>
  <span class="r-label">V_APO    </span><span class="r-val">{d['v_a']:.3f}</span><span class="r-unit"> km/s</span><br>
  <span class="r-label">RAAN_DOT </span><span class="r-val">{d['raan_drift']:.4f}</span><span class="r-unit"> deg/day (J2)</span><br>
  <span class="r-label">ARGP_DOT </span><span class="r-val">{d['argp_drift']:.4f}</span><span class="r-unit"> deg/day (J2)</span><br>
  {warns}
</div>"""


def _add_fov_animation(fig, state):
    """Append satellite + FOV cone animation frames to an existing Plotly figure.

    Adds 5 animated traces (satellite, cone, boresight, footprint day/night)
    then builds go.Frame objects for every *fov_anim_step* timestep.
    The existing static traces (Earth, orbit, stars …) are untouched.

    Parameters
    ----------
    fig   : go.Figure — already built by scene.build()
    state : st.session_state
    """
    import plotly.graph_objects as go

    r_km  = state.get("orbit_r_km")
    v_kms = state.get("orbit_v_kms")
    if r_km is None or v_kms is None:
        return

    N    = len(r_km)
    step = max(1, int(state.get("fov_anim_step", 15)))
    step = max(step, N // 300)          # cap at ~300 frames for performance
    indices = list(range(0, N, step))

    # Sun direction
    sun_hat = None
    if _SUN_PY_OK and state.get("fov_show_sun_shading", True):
        try:
            import astropy.time as _at
            _t0_jd = _at.Time(state.epoch, format="iso", scale="utc").jd
            sun_hat, _ = _sun_position_eci(_t0_jd)
        except Exception:
            pass

    # Build FOV layer (used to generate per-frame traces)
    fov = SensorFOVLayer(
        r_gcrf_km        = r_km,
        v_gcrf_kms       = v_kms,
        time_index       = indices[0],
        half_angle_deg   = float(state.fov_half_angle_deg),
        cone_length_km   = float(state.fov_cone_length_km),
        pointing_mode    = state.fov_pointing_mode,
        custom_direction = [state.fov_custom_x, state.fov_custom_y, state.fov_custom_z],
        color            = state.fov_color,
        opacity          = float(state.fov_opacity),
        show_boresight   = bool(state.fov_show_boresight),
        sun_direction_gcrf = sun_hat,
        show_sun_shading   = bool(state.get("fov_show_sun_shading", True)),
        show_footprint     = bool(state.get("fov_show_footprint", True)),
    )

    # Record static trace count, add initial animated traces
    n_static = len(fig.data)
    for t in fov.build_traces():              # always exactly 5 traces
        fig.add_trace(t)
    animated_idx = list(range(n_static, n_static + 5))

    # Build frames
    frames = []
    for i in indices:
        fov.time_index = i
        frame_traces = fov.build_traces()     # always exactly 5
        frames.append(go.Frame(
            data=frame_traces,
            traces=animated_idx,
            name=str(i),
        ))
    fig.frames = frames

    # Preserve any existing updatemenus, add play/pause
    existing_menus = list(fig.layout.updatemenus or [])
    fig.update_layout(
        updatemenus=existing_menus + [dict(
            type="buttons", showactive=False,
            y=0.01, x=0.50, xanchor="center", yanchor="bottom",
            bgcolor="rgba(16,31,50,0.85)", bordercolor="#4A5F78",
            font=dict(color="#E4EEFA", size=11),
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=80, redraw=True),
                                       fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                         mode="immediate")]),
            ],
        )],
        sliders=[dict(
            steps=[dict(method="animate",
                        args=[[str(i)],
                               dict(mode="immediate",
                                    frame=dict(duration=0, redraw=True))],
                        label=str(i)) for i in indices],
            transition=dict(duration=0),
            x=0.04, len=0.92, y=0.0,
            currentvalue=dict(
                prefix="Step: ", visible=True,
                xanchor="center",
                font=dict(color="#E4EEFA", size=11),
            ),
            pad=dict(t=55),
            bgcolor="rgba(16,31,50,0.7)",
            bordercolor="#4A5F78",
            tickcolor="#4A5F78",
            font=dict(color="#7290B0", size=9),
        )],
    )


# ── Multi-satellite helpers ───────────────────────────────────────────────────

_SAT_PALETTE = [
    "#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF",
    "#FF922B", "#CC5DE8", "#20C997", "#F06595",
    "#74C0FC", "#A9E34B", "#FFA8A8", "#E599F7",
]


def _parse_tle_to_kepler(tle_text: str) -> dict:
    """Parse a 2-line or 3-line TLE into Keplerian element dict."""
    import math as _m, datetime as _dt
    lines = [l.strip() for l in tle_text.strip().splitlines() if l.strip()]
    l1 = next((l for l in lines if l.startswith("1 ")), None)
    l2 = next((l for l in lines if l.startswith("2 ")), None)
    if l1 is None or l2 is None:
        raise ValueError("Could not find TLE line 1 and line 2 — paste all 3 lines.")

    # Epoch from line 1 (cols 18-32: YYDDD.DDDDDDDD)
    ep = l1[18:32].strip()
    yy = int(ep[:2]); yr = 2000 + yy if yy < 57 else 1900 + yy
    doy = float(ep[2:])
    epoch_dt = _dt.datetime(yr, 1, 1, tzinfo=_dt.timezone.utc) + _dt.timedelta(days=doy - 1)
    epoch_iso = epoch_dt.strftime("%Y-%m-%d %H:%M:%S")

    # Line 2 fields
    inc_deg  = float(l2[8:16].strip())
    raan_deg = float(l2[17:25].strip())
    e        = float("0." + l2[26:33].strip())
    argp_deg = float(l2[34:42].strip())
    M_deg    = float(l2[43:51].strip())
    n_rev_d  = float(l2[52:63].strip())   # mean motion (rev/day)

    # Semi-major axis
    MU_KM3S2 = 398_600.4418
    n_rad_s  = n_rev_d * 2.0 * _m.pi / 86_400.0
    a_km     = (MU_KM3S2 / n_rad_s**2) ** (1.0 / 3.0)

    # Mean anomaly → true anomaly via Kepler's equation
    M = _m.radians(M_deg)
    E = M
    for _ in range(60):
        dE = (M - E + e * _m.sin(E)) / (1.0 - e * _m.cos(E))
        E += dE
        if abs(dE) < 1e-12:
            break
    nu_deg = _m.degrees(2.0 * _m.atan2(
        _m.sqrt(1 + e) * _m.sin(E / 2),
        _m.sqrt(1 - e) * _m.cos(E / 2),
    )) % 360.0

    return dict(a_km=a_km, e=e, inc_deg=inc_deg, raan_deg=raan_deg,
                argp_deg=argp_deg, nu_deg=nu_deg, epoch=epoch_iso)


def _propagate_extra_sat(sat: dict):
    """Propagate an extra satellite. Returns (r_km, v_kms) as (N,3) arrays."""
    import ssapy as _ssapy
    import astropy.time as _at
    import math as _m
    _MU = 3.986004418e14

    # Parse TLE if in TLE mode
    mode = sat.get("input_mode", "Keplerian")
    if mode == "TLE":
        tle = sat.get("tle_text", "").strip()
        if not tle:
            raise ValueError("No TLE text — paste or fetch one first.")
        kepler = _parse_tle_to_kepler(tle)
        sat.update(kepler)   # merge parsed elements into sat dict

    a_m = float(sat["a_km"]) * 1e3
    T_s = 2.0 * _m.pi * _m.sqrt(a_m**3 / _MU)
    n_steps = max(100, int(round(float(sat.get("n_orbits", 2.0)) * T_s
                                  / float(sat.get("dt_s", 10.0)))))
    t0    = _at.Time(sat.get("epoch", "2025-01-01 00:00:00"), format="iso", scale="utc")
    t_gps = t0.gps + np.arange(n_steps) * float(sat.get("dt_s", 10.0))
    orbit = _ssapy.Orbit.fromKeplerianElements(
        a_m, float(sat["e"]),
        _m.radians(float(sat["inc_deg"])),
        _m.radians(float(sat["argp_deg"])),
        _m.radians(float(sat["raan_deg"])),
        _m.radians(float(sat["nu_deg"])),
        t0.gps,
    )
    _orbits = orbit.at(t_gps)
    r_km  = np.array([o.r for o in _orbits]) / 1e3
    v_kms = np.array([o.v for o in _orbits]) / 1e3
    return r_km, v_kms


def _add_extra_sat_traces(fig, sat: dict, r_km: np.ndarray, v_kms: np.ndarray | None = None):
    """Append orbit trail, position marker, and optional FOV cone to fig."""
    import plotly.graph_objects as go
    color = sat.get("color", "#FF6B6B")
    name  = sat.get("name",  "Satellite")
    ti    = max(0, min(int(sat.get("time_index", 0)), len(r_km) - 1))

    fig.add_trace(go.Scatter3d(
        x=r_km[:, 0], y=r_km[:, 1], z=r_km[:, 2],
        mode="lines", line=dict(color=color, width=2),
        name=f"{name} (orbit)", showlegend=True,
    ))
    pos = r_km[ti]
    fig.add_trace(go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]], mode="markers",
        marker=dict(size=7, color=color, symbol="circle",
                    line=dict(color="#FFFFFF", width=1)),
        name=name, showlegend=True,
        hovertext=(f"<b>{name}</b><br>"
                   f"r = {float(np.linalg.norm(pos)):.0f} km<br>"
                   f"alt = {float(np.linalg.norm(pos))-6378.137:.0f} km"),
        hoverinfo="text",
    ))
    if sat.get("show_fov") and SensorFOVLayer is not None and v_kms is not None:
        try:
            _fov = SensorFOVLayer(
                r_gcrf_km=r_km, v_gcrf_kms=v_kms, time_index=ti,
                half_angle_deg=float(sat.get("fov_half_angle_deg", 15.0)),
                cone_length_km=float(sat.get("fov_cone_length_km", 8_000.0)),
                pointing_mode=sat.get("fov_pointing_mode", "nadir"),
                color=color, opacity=0.30, show_boresight=True,
                show_sun_shading=False, show_footprint=False,
            )
            for t in _fov.build_traces()[1:3]:
                fig.add_trace(t)
        except Exception:
            pass


def _render_multi_sat_sidebar():
    """Render the multi-satellite panel inside the 3D Preview sidebar."""
    with st.expander("🛰 Multi-Satellite", expanded=bool(ss.extra_satellites)):

        if st.button("＋ Add satellite", key="msat_add"):
            _idx = len(ss.extra_satellites)
            _default_preset = "ISS (LEO)"
            _p = PRESETS.get(_default_preset, {})
            ss.extra_satellites.append(dict(
                id=f"sat_{_idx+2:03d}",
                name=f"Satellite {_idx + 2}",
                color=_SAT_PALETTE[_idx % len(_SAT_PALETTE)],
                active=True,
                input_mode="Preset",
                preset=_default_preset,
                a_km=float(_p.get("a_km", RE_KM + 408)),
                e=float(_p.get("e", 0.0003)),
                inc_deg=float(_p.get("inc_deg", 51.6)),
                raan_deg=float(_p.get("raan_deg", 0.0)),
                argp_deg=float(_p.get("argp_deg", 0.0)),
                nu_deg=float(_p.get("nu_deg", 0.0)),
                epoch=ss.epoch,
                n_orbits=float(ss.n_orbits),
                dt_s=float(ss.dt_s),
                time_index=0,
                show_fov=False,
                fov_half_angle_deg=15.0,
                fov_cone_length_km=8_000.0,
                fov_pointing_mode="nadir",
                tle_text="",
                norad_id="",
            ))
            st.rerun()

        to_delete = None
        for i, sat in enumerate(ss.extra_satellites):
            _sid = sat["id"]
            st.markdown(
                f'<div style="background:#0B1626;border:1px solid #4A5F78;'
                f'border-left:3px solid {sat["color"]};border-radius:4px;'
                f'padding:0.35rem 0.6rem;margin:0.3rem 0;'
                f'font-family:JetBrains Mono,monospace;font-size:0.72rem;">'
                f'{sat["name"]}</div>',
                unsafe_allow_html=True,
            )
            with st.expander(f"⚙ {sat['name']}", expanded=False):
                _ca, _cb = st.columns([3, 1])
                with _ca:
                    sat["name"] = st.text_input("Name", sat["name"], key=f"ms_nm_{_sid}")
                with _cb:
                    sat["color"] = st.color_picker("Colour", sat["color"],
                                                   key=f"ms_col_{_sid}",
                                                   label_visibility="collapsed")
                sat["active"] = st.toggle("Show orbit", sat.get("active", True),
                                          key=f"ms_act_{_sid}")

                # ── Input mode ────────────────────────────────────────────────
                _MODES = ["Preset", "Keplerian", "TLE"]
                _cur_mode = sat.get("input_mode", "Preset")
                sat["input_mode"] = st.radio(
                    "Source", _MODES,
                    index=_MODES.index(_cur_mode) if _cur_mode in _MODES else 0,
                    horizontal=True, key=f"ms_mode_{_sid}",
                )

                # ── Preset mode ───────────────────────────────────────────────
                if sat["input_mode"] == "Preset":
                    _plist = list(PRESETS.keys())
                    _cur_p = sat.get("preset", _plist[0])
                    _new_p = st.selectbox(
                        "Preset", _plist,
                        index=_plist.index(_cur_p) if _cur_p in _plist else 0,
                        key=f"ms_preset_{_sid}",
                    )
                    if _new_p != _cur_p or sat.get("a_km") is None:
                        sat["preset"] = _new_p
                        _pd = PRESETS[_new_p]
                        sat.update({k: float(v) for k, v in _pd.items()})
                        sat["epoch"] = ss.epoch

                    # Live TLE fetch if NORAD ID known for this preset
                    _norad = PRESET_NORAD.get(_new_p)
                    if _norad and _TLE_UPDATER_OK:
                        _cached = ss.get(f"_tle_cache_{_norad}")
                        if _cached:
                            st.caption(f"✔ TLE cached (NORAD {_norad})")
                            if st.button("Use cached TLE", key=f"ms_use_tle_{_sid}"):
                                sat["tle_text"] = (f"{_new_p}\n"
                                                   f"{_cached[0]}\n{_cached[1]}")
                                sat["input_mode"] = "TLE"
                                st.rerun()
                        else:
                            if st.button(f"📡 Fetch TLE (NORAD {_norad})",
                                         key=f"ms_fetch_p_{_sid}"):
                                with st.spinner("Fetching from Space-Track…"):
                                    try:
                                        import urllib.parse, urllib.request
                                        _l1, _l2, _ = fetch_tle_spacetrack(_norad)
                                        if _l1 is None:
                                            _l1, _l2 = fetch_tle_celestrak(_norad)
                                        if _l1 and _l2:
                                            ss[f"_tle_cache_{_norad}"] = (_l1, _l2)
                                            sat["tle_text"] = (f"{_new_p}\n"
                                                               f"{_l1}\n{_l2}")
                                            sat["input_mode"] = "TLE"
                                            st.rerun()
                                        else:
                                            st.error("Fetch failed.")
                                    except Exception as _fe:
                                        st.error(str(_fe))

                    # Show derived elements as read-only info
                    st.caption(f"a = {sat.get('a_km',0):.0f} km  "
                               f"e = {sat.get('e',0):.4f}  "
                               f"i = {sat.get('inc_deg',0):.1f}°")

                # ── TLE mode ──────────────────────────────────────────────────
                elif sat["input_mode"] == "TLE":
                    _nc, _fc = st.columns([2, 1])
                    with _nc:
                        sat["norad_id"] = st.text_input(
                            "NORAD ID", sat.get("norad_id", ""),
                            placeholder="e.g. 25544", key=f"ms_norad_{_sid}")
                    with _fc:
                        st.write("")
                        if st.button("📡 Fetch", key=f"ms_fetch_{_sid}"):
                            _nid = sat.get("norad_id", "").strip()
                            if _nid and _TLE_UPDATER_OK:
                                with st.spinner("Fetching…"):
                                    try:
                                        import urllib.parse, urllib.request
                                        _l1, _l2, _ = fetch_tle_spacetrack(int(_nid))
                                        if _l1 is None:
                                            _l1, _l2 = fetch_tle_celestrak(int(_nid))
                                        if _l1 and _l2:
                                            sat["tle_text"] = (f"NORAD {_nid}\n"
                                                               f"{_l1}\n{_l2}")
                                            st.rerun()
                                        else:
                                            st.error("Fetch failed.")
                                    except Exception as _fe:
                                        st.error(str(_fe))
                            elif not _TLE_UPDATER_OK:
                                st.error("tle_updater.py not found.")
                            else:
                                st.warning("Enter a NORAD ID first.")

                    sat["tle_text"] = st.text_area(
                        "TLE (3 lines)", sat.get("tle_text", ""),
                        height=95, key=f"ms_tle_{_sid}",
                        placeholder="Object name\n1 NNNNN...\n2 NNNNN...",
                    )
                    if sat.get("tle_text"):
                        try:
                            _kp = _parse_tle_to_kepler(sat["tle_text"])
                            st.caption(
                                f"a = {_kp['a_km']:.0f} km  "
                                f"e = {_kp['e']:.4f}  "
                                f"i = {_kp['inc_deg']:.1f}°  "
                                f"epoch {_kp['epoch'][:10]}"
                            )
                        except Exception as _pe:
                            st.caption(f"⚠ {_pe}")

                # ── Keplerian mode ────────────────────────────────────────────
                else:
                    _c1, _c2 = st.columns(2)
                    with _c1:
                        sat["a_km"]    = st.number_input("SMA (km)", value=float(sat.get("a_km", RE_KM+408)),  step=10.0,  key=f"ms_a_{_sid}")
                        sat["e"]       = st.number_input("e",         value=float(sat.get("e",    0.001)),      step=0.001, format="%.4f", key=f"ms_e_{_sid}")
                        sat["inc_deg"] = st.number_input("i (°)",     value=float(sat.get("inc_deg", 51.6)),    step=0.1,   key=f"ms_i_{_sid}")
                    with _c2:
                        sat["raan_deg"] = st.number_input("RAAN (°)", value=float(sat.get("raan_deg", 0.0)),   step=1.0,   key=f"ms_raan_{_sid}")
                        sat["argp_deg"] = st.number_input("ω (°)",    value=float(sat.get("argp_deg", 0.0)),   step=1.0,   key=f"ms_w_{_sid}")
                        sat["nu_deg"]   = st.number_input("ν (°)",    value=float(sat.get("nu_deg",   0.0)),   step=1.0,   key=f"ms_nu_{_sid}")

                # ── Common controls ───────────────────────────────────────────
                st.markdown("---")
                sat["n_orbits"] = st.slider("Orbits", 0.5, 10.0,
                                            float(sat.get("n_orbits", 2.0)), 0.5,
                                            key=f"ms_norb_{_sid}")
                _N = int(ss.get("orbit_n_steps", 1000))
                sat["time_index"] = st.slider(
                    "Marker step", 0, max(0, _N-1),
                    min(int(sat.get("time_index", 0)), max(0, _N-1)),
                    key=f"ms_ti_{_sid}")
                sat["show_fov"] = st.checkbox("Show FOV cone",
                                              sat.get("show_fov", False),
                                              key=f"ms_fov_{_sid}")
                if sat["show_fov"]:
                    sat["fov_half_angle_deg"] = st.slider(
                        "Half-angle °", 1.0, 60.0,
                        float(sat.get("fov_half_angle_deg", 15.0)), 0.5,
                        key=f"ms_fovha_{_sid}")
                    sat["fov_cone_length_km"] = st.number_input(
                        "Cone length (km)", min_value=100.0, max_value=100_000.0,
                        value=float(sat.get("fov_cone_length_km", 8_000.0)),
                        step=500.0, key=f"ms_fovl_{_sid}")
                    _PM = ["nadir", "anti-nadir", "velocity", "anti-velocity"]
                    sat["fov_pointing_mode"] = st.selectbox(
                        "Pointing", _PM,
                        index=_PM.index(sat.get("fov_pointing_mode", "nadir")),
                        key=f"ms_fovpm_{_sid}")

                if st.button("🗑 Remove", key=f"ms_del_{_sid}"):
                    to_delete = i

        if to_delete is not None:
            ss.extra_satellites.pop(to_delete)
            st.rerun()

        if ss.extra_satellites:
            st.caption(f"{len(ss.extra_satellites)} extra satellite(s) "
                       "· ▶ Full propagation to update")


def build_orbital_state() -> "OrbitalState":
    if not _CORE_OK:
        return None
    from core.orbit_state import PropagatorConfig
    cfg = PropagatorConfig(
        propagator=ss.propagator, gravity=ss.gravity,
        third_body=None if ss.third_body=="none" else ss.third_body,
        non_grav=None if ss.non_grav=="none" else ss.non_grav,
        cd=ss.cd, area_m2=ss.area_m2, mass_kg=ss.mass_kg, cr=ss.cr,
    )
    return OrbitalState(
        a_km=ss.a_km, e=ss.e, inc_deg=ss.inc_deg,
        raan_deg=ss.raan_deg, argp_deg=ss.argp_deg, nu_deg=ss.nu_deg,
        config=cfg, name=ss.get("preset","Orbit"),
    )


def build_plotly_scene(state) -> "PlotlyScene":
    # ── Scene scale override ──────────────────────────────────────────────────
    _orig_a_km    = ss.a_km
    _scale_choice = ss.get("preview_scale", "Auto")

    if _scale_choice == "Auto":
        _override = None
    elif _scale_choice == "2× orbit":
        _override = ss.a_km * 2.0
    elif _scale_choice == "5× orbit":
        _override = ss.a_km * 5.0
    elif _scale_choice == "10× orbit":
        _override = ss.a_km * 10.0
    elif _scale_choice == "GEO belt (50 kkm)":
        _override = 50_000.0
    elif _scale_choice == "Earth-Moon (500 kkm)":
        _override = MOON_A_KM * 1.30
    elif _scale_choice == "Sun-Earth L1/L2 (2M km)":
        _override = 2_000_000.0
    elif _scale_choice == "Custom":
        _override = float(ss.get("preview_scale_custom_km", ss.a_km * 2.0))
    else:
        _override = None

    # Auto-widen when Moon, Sun, or L-points are on and the orbit is too small
    # to show them at their true distance (was Lagrange-only; Moon/Sun/L-points
    # sit near lunar distance, so a LEO/GEO-sized Auto scene clips them out of
    # view entirely — this was the root cause of "toggle does nothing").
    # Only Moon and L-points sit near TRUE lunar distance (~384,400 km) and
    # need the scene widened to be visible at all. Sun is drawn schematically
    # relative to whatever the CURRENT scene radius already is (see
    # SunLayer._sun_pos) — it was mistakenly included here before, which
    # ballooned the scene to lunar scale just for turning Sun on, shrinking
    # Earth and the orbit down to an invisible speck.
    _needs_lunar_scale = ss.lyr_lagrange or ss.lyr_moon
    if _override is None and _needs_lunar_scale and ss.a_km < MOON_A_KM * 0.8:
        _override = MOON_A_KM * 1.30
    if _override is not None:
        ss.a_km = max(ss.a_km, _override)
    ss["_scene_orig_a_km"] = _orig_a_km   # tab will restore after scene.build()

    scene = PlotlyScene(state, frame=Frame(ss.frame))

    # When the view is widened toward lunar distance (Moon/Sun/L-points),
    # Earth and Moon become nearly invisible at true 1:1 scale — boost their
    # rendered size (position unchanged, purely visual) so they stay
    # identifiable. Mirrors the "planet_scale" exaggeration already used in
    # the Solar View tab.
    _lunar_scale_active = _needs_lunar_scale and _orig_a_km < MOON_A_KM * 0.8
    _earth_radius_scale = 15.0 if _lunar_scale_active else 1.0
    _moon_radius_scale  = 8.0  if _lunar_scale_active else 1.0

    if ss.lyr_earth:
        tp = ss.earth_texture if Path(ss.earth_texture).exists() else None
        scene.add_layer(EarthLayer(texture_path=tp, radius_scale=_earth_radius_scale))
    if ss.lyr_stars and Path(ss.star_catalog).exists():
        scene.add_layer(StarfieldLayer(catalog_path=ss.star_catalog,
                                       epoch_jd=_get_starfield_epoch_jd()))
    if ss.lyr_moon:     scene.add_layer(MoonLayer(radius_scale=_moon_radius_scale))
    if ss.lyr_sun:      scene.add_layer(SunLayer())
    if ss.lyr_groundtrack: scene.add_layer(GroundTrackLayer())
    if ss.lyr_terminator:  scene.add_layer(TerminatorLayer())
    if ss.lyr_eclipse:     scene.add_layer(EclipseLayer())
    if ss.lyr_van_allen:   scene.add_layer(VanAllenLayer())
    if ss.lyr_lagrange:    scene.add_layer(LagrangeLayer())
    # satellite object — needed for NTW vectors AND burns, independently.
    # (Previously the Satellite3D/NTWLayer were only created inside
    # `if ss.burns:`, so toggling "NTW vectors" with zero burns configured
    # silently did nothing — fixed by building `sat` whenever either is on.)
    if ss.burns or ss.lyr_ntw:
        sat = Satellite3D(mass_kg=ss.mass_kg)
        sat.show_ntw = ss.lyr_ntw
        for b in ss.burns:
            sat.add_burn(BurnEvent(
                epoch_offset_s=b["t_s"],
                dv_ntw_km_s=np.array([b["dv_t"], b["dv_n"], b["dv_w"]]),
                mode=b["mode"],
            ))
        scene.satellite = sat
        if ss.burns:
            scene.add_layer(BurnLayer(satellite=sat))
        if ss.lyr_ntw:
            scene.add_layer(NTWLayer(satellite=sat))
    # sensor FOV — propagate a quick orbit to get r/v arrays, then add the layer
    if ss.fov_enabled and SensorFOVLayer is not None:
        try:
            import math as _math
            _MU = 398_600.4418
            # Use the TRUE orbit semi-major axis, not `ss.a_km` — which may
            # have been temporarily widened above (Moon/Sun/L-points scale
            # override) purely for view-range purposes. Using the widened
            # value here inflated the orbital period by orders of magnitude,
            # which propagated over spans of years-to-decades, pushing dates
            # outside ERFA's 1900-2100 validity window (source of the
            # "epv00 ... date outside range" warning spam).
            _a_m = _orig_a_km * 1e3
            _T_s = 2.0 * _math.pi * _math.sqrt(_a_m**3 / (_MU * 1e9))
            _n_steps = max(200, int(round(ss.n_orbits * _T_s / ss.dt_s)))
            import astropy.time as _at
            import ssapy as _ssapy
            _t0 = _at.Time(ss.epoch, format="iso", scale="utc")
            _t_gps = _t0.gps + np.arange(_n_steps) * ss.dt_s
            _orbit = _ssapy.Orbit.fromKeplerianElements(
                _a_m, ss.e,
                _math.radians(ss.inc_deg), _math.radians(ss.argp_deg),
                _math.radians(ss.raan_deg), _math.radians(ss.nu_deg),
                _t0.gps,
            )
            _orbits = _orbit.at(_t_gps)
            _r_km  = np.array([o.r for o in _orbits]) / 1e3
            _v_kms = np.array([o.v for o in _orbits]) / 1e3
            # cache for sidebar live readout and slider max
            ss["orbit_r_km"]    = _r_km
            ss["orbit_v_kms"]   = _v_kms
            ss["orbit_n_steps"] = _n_steps
            _alt_km = float(np.linalg.norm(_r_km[0])) - 6_378.137
            ss["fov_cone_length_km_suggestion"] = round(_alt_km / 500) * 500
            _ti = max(0, min(int(ss.fov_time_index), _n_steps - 1))
            # Compute sun direction from orbit epoch
            _sun_hat = None
            if _SUN_PY_OK and ss.get("fov_show_sun_shading", True):
                try:
                    import astropy.time as _at2
                    _t0_jd = _at.Time(ss.epoch, format="iso", scale="utc").jd
                    _sun_hat, _ = _sun_position_eci(_t0_jd)
                except Exception:
                    _sun_hat = None
            scene.add_layer(SensorFOVLayer(
                r_gcrf_km      = _r_km,
                v_gcrf_kms     = _v_kms,
                time_index     = _ti,
                half_angle_deg = ss.fov_half_angle_deg,
                cone_length_km = ss.fov_cone_length_km,
                pointing_mode  = ss.fov_pointing_mode,
                custom_direction=[ss.fov_custom_x, ss.fov_custom_y, ss.fov_custom_z],
                color          = ss.fov_color,
                opacity        = ss.fov_opacity,
                show_boresight = ss.fov_show_boresight,
                sun_direction_gcrf = _sun_hat,
                show_sun_shading   = bool(ss.get("fov_show_sun_shading", True)),
                show_footprint     = bool(ss.get("fov_show_footprint", True)),
            ))
        except Exception as _fov_err:
            import streamlit as _st
            _st.warning(f"Sensor FOV error: {_fov_err}")
    return scene


PLOT_MODULES = [
    dict(key="orbit_xy",      title="Orbit XY / 3D",         script="orbit_plot_xy.py",
         engine="matplotlib", output="multi-png", category="Orbit views",
         desc="Textured Earth + Moon, orbit track, ground trace. LEO–cislunar.",
         units="Axes in km (ECI). Ground track lat/lon in degrees."),
    dict(key="orbit_full",    title="Full Orbit Panel",       script="orbit_plot.py",
         engine="matplotlib", output="png", category="Orbit views",
         desc="4-panel: 3D orbit, ground track, altitude vs time, velocity vs time.",
         units="Position: km. Altitude: km. Velocity: km/s. Time: seconds from epoch."),
    dict(key="globe",         title="Globe / Ground Track",   script="globe_plot.py",
         engine="matplotlib", output="png", category="Orbit views",
         desc="Equirectangular Earth map with ground track overlay.",
         units="Latitude/longitude in degrees, equirectangular (Plate Carrée) projection."),
    dict(key="cislunar_3d",   title="Cislunar 3D",            script="cislunar_plot_3d.py",
         engine="matplotlib", output="png", category="Cislunar / Lunar",
         desc="Standalone cislunar 3D: Earth, Moon, orbit, L-points.",
         units="Axes in km (ECI). L-points: L1–L5, Earth-Moon two-body system."),
    dict(key="cislunar_combo",title="Cislunar Combined",      script="cislunar_plot.py",
         engine="matplotlib", output="png", category="Cislunar / Lunar",
         desc="Side-by-side 3D + 2D ecliptic plane views.",
         units="Axes in km (ECI); 2D panel projected onto the ecliptic plane."),
    dict(key="moon_3d",       title="Moon 3D",                script="moon_plot_3d.py",
         engine="matplotlib", output="multi-png", category="Cislunar / Lunar",
         desc="Textured Moon with orbit, surface markers, polar view.",
         units="Axes in km, Moon-centred. Surface markers in selenographic lat/lon (°)."),
    dict(key="magfield",      title="Magnetic Field Lines",   script="magfield_plot_3d.py",
         engine="plotly",     output="html", category="Space environment",
         desc="IGRF 2025 field lines + Van Allen belts. Fully interactive HTML.",
         units="Distance in Earth radii (Rₑ = 6378.137 km). Field lines are dipole-model IGRF traces. "
               "Belt residence time (min/orbit) is an analytic estimate — see Settings tab."),
    dict(key="van_allen",     title="Van Allen Belts",        script="van_allen_plot_3d.py",
         engine="plotly",     output="multi-html", category="Space environment",
         desc="Inner (1–2 Rₑ) + outer (3–6 Rₑ) belt tori — 3 interactive views.",
         units="Distance in Earth radii (Rₑ). Belt boundaries are nominal shells, not flux contours."),
    dict(key="sensor_fov",   title="Sensor FOV",             script="sensor_fov_plot.py",
         engine="plotly",     output="html", category="Sensor / FOV",
         desc="Interactive 3D FOV cone along the propagated orbit. Nadir/velocity/custom pointing.",
         units="Half-angle in degrees; cone length and footprint radius in km. "
               "Colour ramp = solar illumination (bright=sunlit, dark blue=eclipse)."),
]
PLOT_CATEGORIES = sorted(set(m["category"] for m in PLOT_MODULES))


def run_script(mod: dict, log: list) -> bool:
    script = Path(ss.toolkit_dir) / "ssapy_toolkit" / "plots" / mod["script"]
    if not script.exists():
        log.append(f"✖ NOT FOUND: {script}"); return False
    cfg = {
        "a_km": ss.a_km, "e": ss.e, "inc_deg": ss.inc_deg,
        "raan_deg": ss.raan_deg, "argp_deg": ss.argp_deg, "nu_deg": ss.nu_deg,
        "epoch": ss.epoch, "n_orbits": ss.n_orbits, "dt_s": ss.dt_s,
        "output_dir": ss.output_dir,
        "starfield_epoch_jd": _get_starfield_epoch_jd(),  # None = J2000, float = precessed
        "fov_time_index":        int(ss.fov_time_index),
        "fov_half_angle_deg":    float(ss.fov_half_angle_deg),
        "fov_cone_length_km":    float(ss.fov_cone_length_km),
        "fov_pointing_mode":     ss.fov_pointing_mode,
        "fov_custom_direction":  [float(ss.fov_custom_x), float(ss.fov_custom_y), float(ss.fov_custom_z)],
        "fov_color":             ss.fov_color,
        "fov_opacity":           float(ss.fov_opacity),
        "fov_show_boresight":    bool(ss.fov_show_boresight),
        "fov_animate":           bool(ss.fov_animate),
        "fov_anim_step":         int(ss.fov_anim_step),
        # Time-in-orbit-in-field estimate, for the Magnetic Field / Van Allen
        # plots — computed analytically here so it's available even before
        # magfield_plot_3d.py implements its own flux-based version.
        "belt_residence":       _estimate_belt_residence(ss.a_km, ss.e, ss.n_orbits),
        **ss.get(f"ps_{mod['key']}", {}),
    }
    cfg_path = Path(ss.toolkit_dir) / f"_gui_cfg_{mod['key']}.py"
    cfg_path.write_text("\n".join(
        # This file is imported as Python (see GUI_CONFIG usage below), not
        # parsed as JSON — repr() produces valid Python literals (True/
        # False/None) for every value here, whereas json.dumps() used to
        # write JSON's lowercase true/false/null straight into the .py
        # file, which is invalid Python and crashed on import.
        f"{k} = {v!r}"
        for k, v in cfg.items()
    ))
    env = os.environ.copy()
    env.update({"GUI_CONFIG": str(cfg_path), "SSAPY_DIR": ss.ssapy_dir,
                 "OUTPUT_DIR": ss.output_dir})
    log.append(f"▸ {script.name}  [{mod['engine'].upper()}]")
    try:
        module_name = f"ssapy_toolkit.plots.{Path(mod['script']).stem}"
        result = subprocess.run(
            f"conda run -n {ss.conda_env} python -m {module_name}",
            shell=True, capture_output=True, text=True, env=env,
            cwd=ss.toolkit_dir, timeout=900,
        )
        if result.returncode == 0:
            log.append(f"✔ {mod['title']} — complete")
            for ln in (result.stdout or "").strip().splitlines():
                log.append(f"  {ln}")
            return True
        else:
            log.append(f"✖ {mod['title']} — exit {result.returncode}")
            for ln in (result.stderr or result.stdout or "").strip().splitlines():
                log.append(f"  {ln}")
            return False
    except subprocess.TimeoutExpired:
        log.append(f"⏱ {mod['title']} — TIMEOUT"); return False
    except Exception as ex:
        log.append(f"✖ {mod['title']} — {ex}"); return False


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="font-family:JetBrains Mono,monospace;font-size:0.82rem;'
        'color:#2BFFB0;letter-spacing:0.18em;padding:0.2rem 0 0.8rem">SSAPy TOOLKIT</div>',
        unsafe_allow_html=True,
    )

    # ── INPUT MODE ────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-sec">INPUT MODE</div>', unsafe_allow_html=True)
    ss.input_mode = st.selectbox("Mode", ["Keplerian","TLE"], label_visibility="collapsed",
                                  index=["Keplerian","TLE"].index(ss.input_mode))

    if ss.input_mode == "Keplerian":
        st.markdown('<div class="sidebar-sec">PRESET</div>', unsafe_allow_html=True)
        preset_list = ["Custom"] + list(PRESETS.keys())
        prev_preset = ss.preset
        ss.preset = st.selectbox("Preset", preset_list, label_visibility="collapsed",
                                  index=preset_list.index(ss.preset) if ss.preset in preset_list else 0)
        if ss.preset != "Custom" and ss.preset != prev_preset and _CORE_OK:
            p = PRESETS[ss.preset]
            for k, v in p.items():
                ss[k] = v
            ss._scene_key += 1

        # ── Preset info card ─────────────────────────────────────────────────
        if ss.preset != "Custom" and ss.preset in PRESET_INFO:
            info = PRESET_INFO[ss.preset]
            norad = PRESET_NORAD.get(ss.preset)
            norad_badge = (f'<span style="color:var(--amber);font-size:0.65rem">'
                           f'NORAD {norad}</span>' if norad else
                           '<span style="color:var(--dim);font-size:0.65rem">no NORAD ID</span>')
            st.markdown(
                f'<div style="background:#0B1626;border:1px solid var(--border);'
                f'border-left:3px solid var(--amber);border-radius:5px;'
                f'padding:0.55rem 0.75rem;margin:0.4rem 0 0.2rem;'
                f'font-family:JetBrains Mono,monospace;font-size:0.68rem;line-height:1.7;">'
                f'<span style="color:var(--green)">{info["regime"]}</span>'
                f'<span style="color:var(--dim)"> · </span>'
                f'<span style="color:var(--star)">{info["alt"]}</span>'
                f'<span style="color:var(--dim)"> · </span>{norad_badge}<br>'
                f'<span style="color:var(--dim)">{info["desc"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Live TLE fetch ────────────────────────────────────────────────
            if norad:
                with st.expander("📡 Live TLE", expanded=False):
                    cached_tle = ss.get(f"_tle_cache_{norad}")
                    if cached_tle:
                        st.code(f"{ss.preset}\n{cached_tle[0]}\n{cached_tle[1]}", language="text")
                        if st.button("⬆ Load into TLE tab", key=f"load_tle_{norad}"):
                            ss.tle_text = f"{ss.preset}\n{cached_tle[0]}\n{cached_tle[1]}"
                            ss.input_mode = "TLE"
                            st.rerun()
                    else:
                        st.caption(f"NORAD {norad} · via Space-Track")
                        if st.button("Fetch TLE", key=f"fetch_{norad}"):
                            with st.spinner("Fetching from Space-Track…"):
                                try:
                                    if not _TLE_UPDATER_OK:
                                        st.error("tle_updater.py not found — place it in the toolkit root.")
                                    elif not ST_USER:
                                        st.error("ST_USER not set in tle_updater.py.")
                                    else:
                                        import urllib.parse  # required by fetch_tle_spacetrack
                                        import urllib.request
                                        l1, l2 = None, None
                                        try:
                                            # Primary: Space-Track (credentials from tle_updater.py)
                                            l1, l2, _ = fetch_tle_spacetrack(norad)
                                        except Exception as _st_err:
                                            st.warning(f"Space-Track error: {_st_err} — trying Celestrak…")
                                        # Fallback: Celestrak
                                        if l1 is None:
                                            try:
                                                l1, l2 = fetch_tle_celestrak(norad)
                                            except Exception as _ct_err:
                                                st.error(f"Celestrak error: {_ct_err}")
                                        if l1 and l2:
                                            ss[f"_tle_cache_{norad}"] = (l1, l2)
                                            st.rerun()
                                        else:
                                            st.error("Fetch failed — check network or NORAD ID.")
                                except Exception as _fe:
                                    st.error(f"Fetch error: {type(_fe).__name__}: {_fe}")

        st.markdown('<div class="sidebar-sec">KEPLERIAN ELEMENTS</div>', unsafe_allow_html=True)
        prev = (ss.a_km, ss.e, ss.inc_deg, ss.raan_deg, ss.argp_deg, ss.nu_deg)

        ss.a_km     = st.number_input("SMA (km)",    value=float(ss.a_km),    min_value=RE_KM+100, step=10.0, format="%.2f")
        ss.e        = st.number_input("Eccentricity",value=float(ss.e),        min_value=0.0, max_value=0.9999, step=0.001, format="%.6f")
        ss.inc_deg  = st.number_input("Inclination °",value=float(ss.inc_deg), min_value=0.0, max_value=180.0, step=0.1, format="%.3f")
        ss.raan_deg = st.number_input("RAAN °",       value=float(ss.raan_deg),min_value=0.0, max_value=360.0, step=1.0, format="%.3f")
        ss.argp_deg = st.number_input("Arg Perigee °",value=float(ss.argp_deg),min_value=0.0, max_value=360.0, step=1.0, format="%.3f")
        ss.nu_deg   = st.number_input("True Anomaly °",value=float(ss.nu_deg), min_value=0.0, max_value=360.0, step=1.0, format="%.3f")

        if (ss.a_km, ss.e, ss.inc_deg, ss.raan_deg, ss.argp_deg, ss.nu_deg) != prev:
            ss._scene_key += 1

    else:  # TLE
        st.markdown('<div class="sidebar-sec">TLE</div>', unsafe_allow_html=True)
        tle = st.text_area("Paste TLE", value=ss.tle_text, height=120,
                            label_visibility="collapsed",
                            placeholder="ISS (ZARYA)\n1 25544U ...\n2 25544 ...")
        ss.tle_text = tle
        if tle.strip() and _CORE_OK:
            try:
                parsed = OrbitalState.from_tle(tle)
                ss.a_km=parsed.a_km; ss.e=parsed.e; ss.inc_deg=parsed.inc_deg
                ss.raan_deg=parsed.raan_deg; ss.argp_deg=parsed.argp_deg; ss.nu_deg=parsed.nu_deg
                st.markdown('<span style="color:#2BFFB0;font-size:0.7rem;font-family:JetBrains Mono,monospace">✔ TLE parsed</span>', unsafe_allow_html=True)
                ss._scene_key += 1
            except Exception as ex:
                st.markdown(f'<span style="color:#FF7088;font-size:0.7rem;font-family:JetBrains Mono,monospace">✖ {ex}</span>', unsafe_allow_html=True)

    # ── PROPAGATION CONFIG ────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-sec">PROPAGATOR</div>', unsafe_allow_html=True)
    ss.propagator = st.selectbox("Integrator", ["keplerian","scipy","rk4","rk78"],
                                  index=["keplerian","scipy","rk4","rk78"].index(ss.propagator))
    ss.gravity    = st.selectbox("Gravity model", ["point_mass","j2","4x4","8x8"],
                                  index=["point_mass","j2","4x4","8x8"].index(ss.gravity))
    ss.third_body = st.selectbox("Third body", ["none","moon","sun","both"],
                                  index=["none","moon","sun","both"].index(ss.third_body))
    ss.non_grav   = st.selectbox("Non-grav", ["none","drag","srp","both"],
                                  index=["none","drag","srp","both"].index(ss.non_grav))
    if ss.non_grav in ("drag","both"):
        ss.cd      = st.number_input("Cd",     value=float(ss.cd),     min_value=0.1, step=0.1, format="%.2f")
        ss.area_m2 = st.number_input("Area m²",value=float(ss.area_m2),min_value=0.1, step=1.0, format="%.1f")
        ss.mass_kg = st.number_input("Mass kg",value=float(ss.mass_kg),min_value=1.0, step=10.0, format="%.1f")

    # ── TIME ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-sec">PROPAGATION TIME</div>', unsafe_allow_html=True)
    ss.epoch    = st.text_input("Epoch (UTC)", value=ss.epoch)
    ss.n_orbits = st.number_input("Orbits", value=float(ss.n_orbits), min_value=0.5, max_value=50.0, step=0.5)
    ss.dt_s     = st.number_input("Time step (s)", value=float(ss.dt_s), min_value=1.0, max_value=3600.0, step=10.0)

    # ── FRAME ────────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-sec">REFERENCE FRAME</div>', unsafe_allow_html=True)
    ss.frame = st.selectbox("Frame", ["ECI","ECF","LVLH","RTN","NTW"],
                             index=["ECI","ECF","LVLH","RTN","NTW"].index(ss.frame))

    # ── PATHS ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-sec">PATHS</div>', unsafe_allow_html=True)
    ss.toolkit_dir   = st.text_input("Toolkit dir",   value=ss.toolkit_dir)
    ss.ssapy_dir     = st.text_input("SSAPy dir",     value=ss.ssapy_dir)
    ss.star_catalog  = st.text_input("Star catalog",  value=ss.star_catalog)
    ss.earth_texture = st.text_input("Earth texture", value=ss.earth_texture)
    ss.output_dir    = st.text_input("Output dir",    value=ss.output_dir)
    ss.conda_env     = st.text_input("Conda env",     value=ss.conda_env)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="page-title">▸ ORBITAL MISSION PLANNER</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">SSAPy Toolkit · Mission Command Centre</div>', unsafe_allow_html=True)

# Core error notice
if not _CORE_OK:
    st.markdown(f'<div class="warn-box">⚠ core library unavailable: {_core_err_msg} — '
                f'live preview and propagation disabled. Export scripts still work.</div>',
                unsafe_allow_html=True)

# Derived quantities + readout
d = compute_derived()
st.markdown(orbit_readout_html(d), unsafe_allow_html=True)
for w in d["warns"]:
    st.markdown(f'<div class="warn-box">{w}</div>', unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_preview, tab_plots, tab_mission, tab_settings, tab_starmap, tab_solar, tab_save = st.tabs([
    "3D PREVIEW", "EXPORT PLOTS", "MISSION PLANNER", "SETTINGS", "STAR MAP", "☀ SOLAR VIEW", "💾 SAVE",
])


# ══════════════════════════════════════════════════════
# Solar-system helpers (moved above TAB 1 so the 3D PREVIEW tab can render
# the heliocentric view too — lets one "Plot view" dropdown swap between it
# and the orbit scene without needing a separate tab).
# ══════════════════════════════════════════════════════
# ── Orbital mechanics (Keplerian elements + propagation) ──────────────────────
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
    import math
    E = M
    for _ in range(60):
        dE = (M - E + e * math.sin(E)) / (1.0 - e * math.cos(E))
        E += dE
        if abs(dE) < 1e-12:
            break
    return E

def _planet_pos_au(p: dict, t_jd: float):
    import math as _m
    J2000 = 2_451_545.0
    T_days = t_jd - J2000
    n_deg_day = 360.0 / (365.25 * _m.sqrt(p["a"] ** 3))
    M = _m.radians((p["M0"] + n_deg_day * T_days) % 360.0)
    E = _solve_kepler(M, p["e"])
    nu = 2.0 * _m.atan2(_m.sqrt(1 + p["e"]) * _m.sin(E / 2),
                         _m.sqrt(1 - p["e"]) * _m.cos(E / 2))
    r = p["a"] * (1.0 - p["e"] * _m.cos(E))
    xo, yo = r * _m.cos(nu), r * _m.sin(nu)
    w  = _m.radians(p["w"]);  i  = _m.radians(p["i"]);  Om = _m.radians(p["Om"])
    cw, sw = _m.cos(w), _m.sin(w)
    ci, si = _m.cos(i), _m.sin(i)
    cO, sO = _m.cos(Om), _m.sin(Om)
    x = cO*(cw*xo - sw*yo) - sO*(sw*xo + cw*yo)*ci
    y = sO*(cw*xo - sw*yo) + cO*(sw*xo + cw*yo)*ci
    z = (sw*xo + cw*yo)*si
    return x, y, z

def _orbit_trail_au(p: dict, t_jd: float, n_pts: int = 360):
    import math as _m
    T_orbit_days = 365.25 * _m.sqrt(p["a"] ** 3)
    xs, ys, zs = [], [], []
    for k in range(n_pts + 1):
        tj = t_jd - T_orbit_days + k * T_orbit_days / n_pts
        x, y, z = _planet_pos_au(p, tj)
        xs.append(x); ys.append(y); zs.append(z)
    return xs, ys, zs

# ── solar_bodies import (3D sphere rendering) ─────────────────────────────────
try:
    import sys as _sys
    for _sp in [str(Path(__file__).parent / "ssapy_toolkit" / "plots"),
                str(Path(__file__).parent)]:
        if _sp not in _sys.path: _sys.path.insert(0, _sp)
    from solar_bodies import (make_planet_traces, make_saturn_ring_traces,
                               make_sun_traces, make_moon_trace, _R_AU as _SB_R_AU)
    _SOLAR_BODIES_OK = True
except ImportError as _sbe:
    _SOLAR_BODIES_OK = False
    _SOLAR_BODIES_ERR = str(_sbe)

# (_SunLayer, _sun_position_eci, _SUN_PY_OK, _SUN_CORONA_OK already set at top of file)

_TRAIL_COLORS = {
    "Mercury":"rgba(170,170,170,0.35)","Venus":"rgba(255,204,68,0.35)",
    "Earth":"rgba(26,143,209,0.40)","Mars":"rgba(212,90,42,0.38)",
    "Jupiter":"rgba(200,164,110,0.30)","Saturn":"rgba(232,217,160,0.28)",
    "Uranus":"rgba(125,232,232,0.28)","Neptune":"rgba(63,84,186,0.28)",
}
_PLANET_COLORS = {
    "Mercury":"#AAAAAA","Venus":"#FFCC44","Earth":"#1a8fd1","Mars":"#d45a2a",
    "Jupiter":"#c8a46e","Saturn":"#e8d9a0","Uranus":"#7de8e8","Neptune":"#3f54ba",
}
_PLANET_SIZES = {
    "Mercury":4,"Venus":6,"Earth":7,"Mars":5,
    "Jupiter":12,"Saturn":10,"Uranus":8,"Neptune":8,
}

def _build_solar_view_fig(t_jd, controls, catalog_path):
    import plotly.graph_objects as go
    fig = go.Figure()
    bg  = "#101F32"

    # Scene range — compute early so starfield radius can use it
    import math as _msv2
    outer_a = max((p["a"] for n, p in _PLANETS.items() if controls.get(n, False)), default=1.5)
    rng = outer_a * 1.25
    # Star sphere must stay INSIDE the scene bounds to avoid clipping gaps.
    # Any point on a sphere of radius R satisfies |x|,|y|,|z| ≤ R, so
    # keeping R < rng guarantees every star is within the [-rng, rng] box.
    _R_star = rng * 0.95

    # Starfield
    if controls.get("stars") and Path(catalog_path).exists():
        try:
            _sdf = _load_star_catalog_df(catalog_path)
            if _sdf is not None:
                _mag_lim = 6.0
                _sdf = _sdf[(_sdf["mag"] < _mag_lim) & (_sdf["mag"] > -10)].dropna(subset=["ra","dec","mag"])
                _ra_r  = np.radians(_sdf["ra"].values * 15.0)
                _dec_r = np.radians(_sdf["dec"].values)
                _cx = np.cos(_dec_r)*np.cos(_ra_r)*_R_star
                _cy = np.cos(_dec_r)*np.sin(_ra_r)*_R_star
                _cz = np.sin(_dec_r)*_R_star
                _mag   = _sdf["mag"].values
                _sizes = np.clip(0.6*(_mag_lim-_mag)**1.1, 0.3, 3.5)
                _spect = _sdf["spect"].fillna("G").str[:1].values
                _cols  = [_SPECT_COLORS_PLOTLY.get(s,_SPECT_COLORS_PLOTLY["G"]) for s in _spect]
                fig.add_trace(go.Scatter3d(x=_cx,y=_cy,z=_cz,mode="markers",
                    marker=dict(size=_sizes,color=_cols,opacity=0.7),
                    hoverinfo="skip",name="Stars",showlegend=True))
        except Exception:
            pass

    # Ecliptic grid
    if controls.get("ecliptic"):
        _th = np.linspace(0,2*np.pi,200)
        for _r in [1,5,10,20,30]:
            fig.add_trace(go.Scatter3d(x=np.cos(_th)*_r,y=np.sin(_th)*_r,z=np.zeros(200),
                mode="lines",line=dict(color="rgba(43,255,176,0.07)",width=1),
                hoverinfo="skip",showlegend=False))

    # Sun — proper corona rendering needs the old Plotly SunLayer, which no
    # longer exists (see the note near the top of this file); falls back to
    # the inline plain-sphere renderer below.
    _r_sun = 0.045 * float(controls.get("planet_scale", 1.0))
    if _SUN_CORONA_OK:
        # SunLayer uses whatever units you pass; scene is in AU so radius is in AU
        _sun_layer = _SunLayer(
            sun_pos_eci=[0.0, 0.0, 0.0],
            radius_km=_r_sun,          # "km" arg is in AU here — units consistent with scene
            show_corona=True,
            show_label=True,
        )
        for _t in _sun_layer.build_traces():
            fig.add_trace(_t)
    else:
        # Inline fallback (no sun.py) — plain sphere, no corona
        _n_s = 50
        _u_s = np.linspace(0, 2*np.pi, _n_s)
        _v_s = np.linspace(0, np.pi,   _n_s)
        _Us, _Vs = np.meshgrid(_u_s, _v_s)
        _xs = np.sin(_Vs)*np.cos(_Us); _ys = np.sin(_Vs)*np.sin(_Us); _zs = np.cos(_Vs)
        _bright = np.clip(0.6 + 0.4*_zs, 0, 1)
        fig.add_trace(go.Surface(
            x=_r_sun*_xs, y=_r_sun*_ys, z=_r_sun*_zs,
            surfacecolor=_bright,
            colorscale=[[0,"rgb(255,180,20)"],[0.5,"rgb(255,220,80)"],
                        [0.8,"rgb(255,245,180)"],[1,"rgb(255,255,240)"]],
            cmin=0, cmax=1, showscale=False, name="Sun",
            hovertext="☀ Sun", hoverinfo="text",
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0),
        ))

    # Planets
    earth_pos = None
    for name, p in _PLANETS.items():
        if not controls.get(name, False):
            continue
        x, y, z = _planet_pos_au(p, t_jd)
        pos = (x, y, z)
        if name == "Earth":
            earth_pos = pos

        if controls.get("trails"):
            tx, ty, tz = _orbit_trail_au(p, t_jd)
            fig.add_trace(go.Scatter3d(x=tx,y=ty,z=tz,mode="lines",
                line=dict(color=_TRAIL_COLORS.get(name,"rgba(200,200,200,0.3)"),width=1),
                hoverinfo="skip",showlegend=False))

        if _SOLAR_BODIES_OK:
            _scale = float(controls.get("planet_scale", 1.0))
            for t in make_planet_traces(name, pos, scale_au=_scale,
                                        show_label=controls.get("labels",True)):
                fig.add_trace(t)
            if name == "Saturn":
                for t in make_saturn_ring_traces(pos, scale_au=_scale):
                    fig.add_trace(t)
        else:
            fig.add_trace(go.Scatter3d(x=[x],y=[y],z=[z],mode="markers",
                marker=dict(size=_PLANET_SIZES.get(name,5),
                            color=_PLANET_COLORS.get(name,"#CCCCCC")),
                name=name,hovertext=name,hoverinfo="text"))

    if controls.get("moon") and earth_pos is not None:
        if _SOLAR_BODIES_OK:
            fig.add_trace(make_moon_trace(earth_pos))
        else:
            import math as _m
            mx = earth_pos[0]+0.012*_m.cos(0.5)
            my = earth_pos[1]+0.012*_m.sin(0.5)
            fig.add_trace(go.Scatter3d(x=[mx],y=[my],z=[earth_pos[2]],
                mode="markers",marker=dict(size=3,color="#DDDDDD"),
                name="Moon",hovertext="Moon",hoverinfo="text"))

    import math as _m
    T_yr = (t_jd-2_451_545.0)/365.25
    epoch_label = f"Year {2000+T_yr:.2f}"
    # outer_a and rng already computed at top of function for starfield radius

    # Camera preset eye positions (normalised to rng so they scale with zoom)
    _cam_top  = dict(eye=dict(x=0,   y=0,    z=1.5), up=dict(x=0,y=1,z=0),
                     projection=dict(type="perspective"))
    _cam_side = dict(eye=dict(x=1.5, y=0,    z=0.08), up=dict(x=0,y=0,z=1),
                     projection=dict(type="perspective"))
    _cam_persp= dict(eye=dict(x=0.75, y=-0.75, z=0.55), up=dict(x=0,y=0,z=1),
                     projection=dict(type="perspective"))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-rng,rng], showbackground=False,
                       showgrid=False, zeroline=False, title="X (AU)"),
            yaxis=dict(range=[-rng,rng], showbackground=False,
                       showgrid=False, zeroline=False, title="Y (AU)"),
            zaxis=dict(range=[-rng,rng], showbackground=False,
                       showgrid=False, zeroline=False, title="Z (AU)"),
            bgcolor=bg,
            aspectmode="cube",           # true 3D — enables full freecam rotation
            camera=_cam_persp,
        ),
        updatemenus=[dict(
            type="buttons", showactive=True,
            x=0.01, y=0.99, xanchor="left", yanchor="top",
            bgcolor="rgba(16,31,50,0.85)", bordercolor="#4A5F78",
            font=dict(color="#E4EEFA", size=10, family="JetBrains Mono,monospace"),
            buttons=[
                dict(label="☀ Top",
                     method="relayout",
                     args=[{"scene.camera": _cam_top}]),
                dict(label="↔ Side",
                     method="relayout",
                     args=[{"scene.camera": _cam_side}]),
                dict(label="⟳ Persp",
                     method="relayout",
                     args=[{"scene.camera": _cam_persp}]),
            ],
        )],
        paper_bgcolor=bg, font=dict(color="#E4EEFA"),
        title=dict(text=f"Heliocentric Solar System — {epoch_label}",
                   x=0.5, font=dict(color="#2BFFB0", size=13)),
        legend=dict(bgcolor="rgba(0,0,0,0.45)", bordercolor="#333",
                    borderwidth=1, font=dict(size=10)),
        margin=dict(l=0, r=0, t=45, b=30),
        # Constant uirevision — as long as this string doesn't change,
        # Plotly preserves the user's camera/zoom across reruns, so
        # stepping the date slider no longer snaps the view back to default.
        uirevision="solar_view_camera",
    )
    return fig

# ══════════════════════════════════════════════════════
# TAB 1 — 3D PREVIEW
# ══════════════════════════════════════════════════════
with tab_preview:
    pc1, pc2 = st.columns([3, 1])

    with pc2:
        # ── Unified plot-view switcher ───────────────────────────────────────
        # One dropdown swaps the entire main panel between the geocentric orbit
        # scene and the heliocentric solar system view — no separate tab needed.
        _VIEW_OPTS = ["🛰 Orbit Scene", "☀ Solar System"]
        ss.preview_plot_view = st.selectbox(
            "Plot view", _VIEW_OPTS,
            index=_VIEW_OPTS.index(ss.preview_plot_view) if ss.preview_plot_view in _VIEW_OPTS else 0,
            key="preview_view_sel",
            help="Swap the main panel's plot without leaving this tab.",
        )
        st.markdown("---")
        fast_btn = full_btn = False
        if ss.preview_plot_view == "🛰 Orbit Scene":
            st.markdown("**Preview layers**")
            ss.lyr_earth      = st.toggle("Earth",        value=ss.lyr_earth,      key="l_earth")
            ss.lyr_stars      = st.toggle("Stars",         value=ss.lyr_stars,      key="l_stars")
            ss.lyr_moon       = st.toggle("Moon",          value=ss.lyr_moon,       key="l_moon",
                                          help="Moon sits ~384,400 km away — auto-widens Scene scale "
                                               "to Earth-Moon distance so it's actually visible.")
            ss.lyr_sun        = st.toggle("Sun",           value=ss.lyr_sun,        key="l_sun",
                                          help="Same auto-widening as Moon — the Sun marker is placed "
                                               "just outside the visible scene radius.")
            ss.lyr_groundtrack= st.toggle("Ground track",  value=ss.lyr_groundtrack,key="l_gt")
            ss.lyr_terminator = st.toggle("Terminator",    value=ss.lyr_terminator, key="l_term")
            ss.lyr_eclipse    = st.toggle("Eclipse",       value=ss.lyr_eclipse,    key="l_ecl",
                                          help="Only shows markers on orbit segments actually inside "
                                               "Earth's shadow. Needs ▶ Full propagation (not Instant "
                                               "preview) and enough Orbits to pass through the shadow — "
                                               "if the plane rarely crosses it, you may see nothing.")
            ss.lyr_van_allen  = st.toggle("Van Allen",     value=ss.lyr_van_allen,  key="l_va")
            ss.lyr_lagrange   = st.toggle("L-points",      value=ss.lyr_lagrange,   key="l_lag",
                                          help="L1–L5 sit near lunar distance — auto-widens Scene scale "
                                               "to Earth-Moon so the points land inside the visible box.")
            ss.lyr_ntw        = st.toggle("NTW vectors",   value=ss.lyr_ntw,        key="l_ntw",
                                          help="Draws T/N/W axes at the satellite's current position — "
                                               "works with or without burns configured.")
            st.markdown("---")
            # ── Multi-Satellite ──────────────────────────────────────────────────
            _render_multi_sat_sidebar()
            st.markdown("---")
            # ── Sensor FOV ──────────────────────────────────────────────────────
            ss.fov_enabled = st.toggle("🔭 Sensor FOV", value=ss.fov_enabled, key="l_fov")
            if ss.fov_enabled:
                POINTING_MODES = ["nadir","anti-nadir","velocity","anti-velocity","custom"]
                ss.fov_pointing_mode = st.selectbox(
                    "Pointing", POINTING_MODES,
                    index=POINTING_MODES.index(ss.fov_pointing_mode), key="fov_pm",
                    help="nadir=toward Earth, anti-nadir=away from Earth, velocity=ram direction",
                )
                if ss.fov_pointing_mode == "custom":
                    _fc1, _fc2, _fc3 = st.columns(3)
                    with _fc1: ss.fov_custom_x = st.number_input("X", value=float(ss.fov_custom_x), step=0.1, format="%.2f", key="fov_cx")
                    with _fc2: ss.fov_custom_y = st.number_input("Y", value=float(ss.fov_custom_y), step=0.1, format="%.2f", key="fov_cy")
                    with _fc3: ss.fov_custom_z = st.number_input("Z", value=float(ss.fov_custom_z), step=0.1, format="%.2f", key="fov_cz")
                ss.fov_half_angle_deg = st.slider(
                    "Half-angle °", 1.0, 90.0, float(ss.fov_half_angle_deg), 0.5, key="fov_ha",
                )
                ss.fov_cone_length_km = st.number_input(
                    "Cone length (km)", min_value=100.0, max_value=500_000.0,
                    value=float(ss.get("fov_cone_length_km_suggestion", ss.fov_cone_length_km)),
                    step=500.0, key="fov_cl",
                )
                _N = int(ss.get("orbit_n_steps", 1000))
                ss.fov_time_index = st.slider(
                    "Time step", 0, max(0, _N-1), min(int(ss.fov_time_index), max(0,_N-1)), key="fov_ti",
                    help="Slide to move the satellite (and cone) along the orbit.",
                )
                _fov_col, _fov_opc = st.columns(2)
                with _fov_col: ss.fov_color   = st.color_picker("Colour", ss.fov_color, key="fov_col")
                with _fov_opc: ss.fov_opacity = st.slider("Opacity", 0.05, 1.0, float(ss.fov_opacity), 0.05, key="fov_op")
                ss.fov_show_boresight = st.checkbox("Boresight line", value=ss.fov_show_boresight, key="fov_bs")
                if _SUN_PY_OK:
                    ss["fov_show_sun_shading"] = st.checkbox(
                        "☀ Sun shading", value=ss.get("fov_show_sun_shading", True), key="fov_sun",
                        help="Shade the cone by solar illumination — lit side bright, shadow side dark blue.",
                    )
                    ss["fov_show_footprint"] = st.checkbox(
                        "Ground footprint", value=ss.get("fov_show_footprint", True), key="fov_fp",
                        help="Draw the footprint circle on Earth, split into day (bright) and night (dark) arcs.",
                    )
                else:
                    st.caption("☀ Sun shading requires core/sun_mpl.py")

                st.markdown("---")
                ss.fov_animate = st.toggle(
                    "🛰 Animate along orbit", value=ss.fov_animate, key="fov_anim_tog",
                    help="Adds play/pause controls and a time slider. "
                         "Satellite and cone move along the propagated orbit.",
                )
                if ss.fov_animate:
                    _N_orbit = int(ss.get("orbit_n_steps", 1000))
                    _max_step = max(1, _N_orbit // 20)
                    ss.fov_anim_step = st.slider(
                        "Frame step", 1, max(2, _N_orbit // 50),
                        int(ss.fov_anim_step), key="fov_anim_step_sl",
                        help="Larger step = fewer frames = faster playback. "
                             f"~{max(1, _N_orbit // max(1,int(ss.fov_anim_step)))} frames total.",
                    )
                    st.caption(
                        f"~{max(1, _N_orbit // max(1,int(ss.fov_anim_step)))} frames  "
                        f"· Hit ▶ Full propagation first"
                    )
                if ss.get("orbit_r_km") is not None:
                    _r_now = ss["orbit_r_km"][min(int(ss.fov_time_index), len(ss["orbit_r_km"])-1)]
                    _alt   = float(np.linalg.norm(_r_now)) - 6_378.137
                    import math as _m
                    _fp = ss.fov_cone_length_km * abs(_m.tan(_m.radians(ss.fov_half_angle_deg)))
                    st.caption(f"Alt: {_alt:,.0f} km  |  Footprint r ≈ {_fp:,.0f} km")
            st.markdown("---")
            ss.show_osculating= st.toggle("Osculating ellipse", value=ss.show_osculating, key="l_osc")

            # ── Scene scale ──────────────────────────────────────────────────────
            st.markdown("---")
            _SCALE_OPTS = [
                "Auto",
                "2× orbit",
                "5× orbit",
                "10× orbit",
                "GEO belt (50 kkm)",
                "Earth-Moon (500 kkm)",
                "Sun-Earth L1/L2 (2M km)",
                "Custom",
            ]
            _prev_scale = ss.get("preview_scale", "Auto")
            ss.preview_scale = st.selectbox(
                "🔭 Scene scale",
                _SCALE_OPTS,
                index=_SCALE_OPTS.index(_prev_scale) if _prev_scale in _SCALE_OPTS else 0,
                key="prev_scale",
                help=(
                    "**Auto** — sized to the propagated orbit.\n"
                    "**2×/5×/10× orbit** — multiples of the current semi-major axis.\n"
                    "**GEO belt** — 50 000 km; shows GEO ring + Van Allen belts.\n"
                    "**Earth-Moon** — ~500 000 km; all 5 E-M Lagrange points.\n"
                    "**Sun-Earth L1/L2** — 2 000 000 km.\n"
                    "**Custom** — enter any range in km."
                ),
            )
            if ss.preview_scale == "Custom":
                ss["preview_scale_custom_km"] = st.number_input(
                    "Range (km)", min_value=1_000.0, max_value=5_000_000.0,
                    value=float(ss.get("preview_scale_custom_km", max(ss.a_km * 2.0, 20_000.0))),
                    step=1_000.0, format="%.0f", key="prev_scale_custom",
                )
            # Remind the user when L-points are on but scale is too small
            if ss.lyr_lagrange and ss.get("preview_scale", "Auto") == "Auto" and ss.a_km < MOON_A_KM * 0.8:
                st.caption("⚠ Auto-widening to Earth-Moon scale — orbit is smaller than L-point distance.")
            st.markdown("---")
            fast_btn = st.button("⚡ Instant preview", key="fast_prev")
            full_btn = st.button("▶ Full propagation", key="full_prev")
        else:
            st.markdown("**☀ Solar System controls**")
            import datetime as _dtp
            _sv_year_p  = st.number_input("Year",  min_value=1800, max_value=2200,
                                          value=int(ss.sol_year), step=1, key="prev_sv_year")
            _sv_month_p = st.slider("Month", 1, 12, int(ss.sol_month), key="prev_sv_month")
            ss.sol_year, ss.sol_month = int(_sv_year_p), int(_sv_month_p)
            _spc1, _spc2, _spc3 = st.columns(3)
            with _spc1:
                if st.button("◀ −1mo", key="prev_sv_back"):
                    ss.sol_month -= 1
                    if ss.sol_month < 1: ss.sol_month = 12; ss.sol_year -= 1
                    st.rerun()
            with _spc2:
                if st.button("▶ +1mo", key="prev_sv_fwd"):
                    ss.sol_month += 1
                    if ss.sol_month > 12: ss.sol_month = 1; ss.sol_year += 1
                    st.rerun()
            with _spc3:
                if st.button("▶▶ +1yr", key="prev_sv_fwd_yr"):
                    ss.sol_year += 1; st.rerun()
            st.caption("Camera position is preserved when you change the date — "
                       "drag/scroll to explore, it won't snap back.")
            st.markdown("**Planets**")
            ss.sol_show_mercury = st.toggle("Mercury", value=ss.sol_show_mercury, key="prev_sv_merc")
            ss.sol_show_venus   = st.toggle("Venus",   value=ss.sol_show_venus,   key="prev_sv_ven")
            ss.sol_show_earth   = st.toggle("Earth",   value=ss.sol_show_earth,   key="prev_sv_ear")
            ss.sol_show_mars    = st.toggle("Mars",    value=ss.sol_show_mars,    key="prev_sv_mars")
            ss.sol_show_jupiter = st.toggle("Jupiter", value=ss.sol_show_jupiter, key="prev_sv_jup")
            ss.sol_show_saturn  = st.toggle("Saturn",  value=ss.sol_show_saturn,  key="prev_sv_sat")
            ss.sol_show_uranus  = st.toggle("Uranus",  value=ss.sol_show_uranus,  key="prev_sv_ura")
            ss.sol_show_neptune = st.toggle("Neptune", value=ss.sol_show_neptune, key="prev_sv_nep")
            ss.sol_show_moon     = st.toggle("Moon",          value=ss.sol_show_moon,     key="prev_sv_moon")
            ss.sol_show_trails   = st.toggle("Orbit trails",  value=ss.sol_show_trails,   key="prev_sv_trail")
            ss.sol_show_stars    = st.toggle("Starfield",     value=ss.sol_show_stars,    key="prev_sv_stars")
            ss.sol_show_ecliptic = st.toggle("Ecliptic grid", value=ss.sol_show_ecliptic, key="prev_sv_ecl")
            ss["sol_planet_scale"] = st.slider(
                "Planet size ×", 0.3, 3.0, float(ss.get("sol_planet_scale", 1.0)), 0.1, key="prev_sv_scale")


    with pc1:
        if ss.preview_plot_view == "☀ Solar System":
            # ── Heliocentric solar system view (reuses the same panel) ───────
            try:
                import datetime as _dtp2
                try:
                    _prev_sv_date = _dtp2.date(int(ss.sol_year), int(ss.sol_month), 1)
                except Exception:
                    _prev_sv_date = _dtp2.date(2025, 1, 1)
                _prev_sv_jd = 2_451_545.0 + (_prev_sv_date - _dtp2.date(2000, 1, 1)).days
                _prev_sv_controls = dict(
                    Mercury=ss.sol_show_mercury, Venus=ss.sol_show_venus,
                    Earth=ss.sol_show_earth,     Mars=ss.sol_show_mars,
                    Jupiter=ss.sol_show_jupiter, Saturn=ss.sol_show_saturn,
                    Uranus=ss.sol_show_uranus,   Neptune=ss.sol_show_neptune,
                    moon=ss.sol_show_moon,       trails=ss.sol_show_trails,
                    stars=ss.sol_show_stars,     ecliptic=ss.sol_show_ecliptic,
                    labels=True,                 planet_scale=float(ss.get("sol_planet_scale", 1.0)),
                )
                with st.spinner("Building solar view…"):
                    _prev_sv_fig = _build_solar_view_fig(_prev_sv_jd, _prev_sv_controls, ss.star_catalog)
                st.plotly_chart(_prev_sv_fig, width='stretch')
            except Exception as _psve:
                st.error(f"Solar view error: {_psve}")

        elif _CORE_OK:
            state = build_orbital_state()
            if state:
                try:
                    scene = build_plotly_scene(state)
                    # `is_full` captures whether we're in "fully propagated" mode.
                    # It stays True across reruns (any widget edit) once the user
                    # has clicked Full propagation, until they click Instant —
                    # this is also what keeps extra satellites / FOV animation
                    # from vanishing on the very next rerun (previous bug: those
                    # were gated on `full_btn` alone, which is only True for the
                    # single rerun where the button was physically clicked).
                    is_full = full_btn or (not fast_btn and (ss._scene_key > 0 or ss._preview_full_mode))
                    if full_btn:
                        ss._preview_full_mode = True
                    if fast_btn:
                        ss._preview_full_mode = False
                    if is_full:
                        with st.spinner("Propagating…"):
                            fig = scene.build(
                                n_orbits=ss.n_orbits, dt_s=ss.dt_s,
                                show_osculating=ss.show_osculating,
                            )
                    else:
                        fig = scene.build_fast()
                    # Restore a_km now that the scene is built
                    ss.a_km = ss.get("_scene_orig_a_km", ss.a_km)
                    # Extra satellites — propagate and paint onto the same figure.
                    # Fixed: now keyed off `is_full` (persistent) instead of the
                    # one-shot `full_btn`, so satellites stay visible after any
                    # later rerun (slider tweak, toggle, etc).
                    if ss.extra_satellites and is_full:
                        for _esat in ss.extra_satellites:
                            if not _esat.get("active", True):
                                continue
                            try:
                                _er_km, _ev_kms = _propagate_extra_sat(_esat)
                                _add_extra_sat_traces(fig, _esat, _er_km, _ev_kms)
                            except Exception as _ese:
                                st.warning(f"Could not plot {_esat.get('name','?')}: {_ese}")
                    elif ss.extra_satellites and not is_full:
                        st.caption(f"🛰 {len(ss.extra_satellites)} extra satellite(s) added — "
                                   "hit ▶ Full propagation to render them.")
                    # FOV animation — appends frames to the existing figure
                    if ss.fov_enabled and ss.fov_animate and ss.get("orbit_r_km") is not None:
                        with st.spinner("Building FOV animation frames…"):
                            _add_fov_animation(fig, ss)
                    # Closer default camera — starts zoomed toward the scene
                    # centre; all normal Plotly scroll/drag zoom-out still works.
                    # uirevision is keyed off the *effective scale* (scene-scale
                    # choice + whether Moon/Sun/L-points are forcing a lunar-
                    # distance view) so the camera resets to a sane default
                    # whenever that scale actually changes — e.g. turning Moon
                    # off after it widened the view no longer leaves Earth
                    # stranded tiny inside an unchanged, stale zoomed-out
                    # camera. Pure cosmetic toggles (Earth/Stars/etc, which
                    # don't affect scale) still preserve the user's own
                    # pan/zoom across reruns.
                    _scale_sig = (ss.get("preview_scale", "Auto"),
                                  bool(ss.lyr_moon or ss.lyr_sun or ss.lyr_lagrange))
                    fig.update_layout(
                        scene_camera=dict(eye=dict(x=0.9, y=0.9, z=0.6)),
                        uirevision=f"orbit_scene_camera_{_scale_sig}",
                    )
                    st.plotly_chart(fig, width='stretch')
                except Exception as ex:
                    st.error(f"Preview error: {ex}")
        else:
            err_detail = _core_err_msg or 'unknown import error'
            st.markdown(
                f'<div class="readout" style="min-height:350px;">'
                f'<div class="r-head">▸ CORE LIBRARY UNAVAILABLE</div>'
                f'<div class="r-err" style="margin:1rem 0;font-size:0.78rem;word-break:break-all">'
                f'{err_detail}</div>'
                f'<div class="r-label">Export plots (Tab 2) still work without the core library.<br>'
                f'Fix the error above, then restart Streamlit.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════
# TAB 2 — EXPORT PLOTS
# ══════════════════════════════════════════════════════
with tab_plots:
    st.markdown('<p style="font-size:0.71rem;color:#7290B0;font-family:JetBrains Mono,monospace;'
                'margin-bottom:1rem">Toggle modules, configure, then GENERATE.</p>',
                unsafe_allow_html=True)

    # ── Dropdown plot browser: category → plot → sub-options ────────────────
    st.markdown("**🔎 Plot browser**")
    _pb1, _pb2 = st.columns(2)
    with _pb1:
        _pb_cat = st.selectbox("Category", PLOT_CATEGORIES, key="pb_category")
    _cat_mods = [m for m in PLOT_MODULES if m["category"] == _pb_cat]
    with _pb2:
        _pb_mod_title = st.selectbox(
            "Plot", [m["title"] for m in _cat_mods], key="pb_plot")
    _pb_mod = next(m for m in _cat_mods if m["title"] == _pb_mod_title)
    _pb_badge_e = f'<span class="badge b-{"plotly" if _pb_mod["engine"]=="plotly" else "mpl"}">{_pb_mod["engine"].upper()}</span>'
    _pb_badge_o = f'<span class="badge b-{"html" if "html" in _pb_mod["output"] else "png"}">{_pb_mod["output"].upper()}</span>'
    st.markdown(
        f'<div class="plot-card active"><div class="card-title">{_pb_mod["title"]}</div>'
        f'<div class="card-desc">{_pb_mod["desc"]}</div>'
        f'{_pb_badge_e}{_pb_badge_o}<br><br>'
        f'<span style="font-size:0.68rem;color:var(--dim);line-height:1.6">'
        f'📏 <b>Units / legend:</b> {_pb_mod["units"]}</span></div>',
        unsafe_allow_html=True,
    )
    _pb_key = f"mod_{_pb_mod['key']}"
    _pb_on  = st.toggle("Enable this plot", value=ss.get(_pb_key, False), key=f"pb_tog_{_pb_mod['key']}")
    ss[_pb_key] = _pb_on
    _pb_ps_key = f"ps_{_pb_mod['key']}"
    _pb_ps = ss.get(_pb_ps_key, {})
    with st.expander("⚙ Sub-options for this plot", expanded=_pb_on):
        if "show_stars"    in _pb_ps: _pb_ps["show_stars"]    = st.checkbox("Stars",          value=_pb_ps["show_stars"],    key=f"pb_{_pb_mod['key']}_s")
        if "show_earth"    in _pb_ps: _pb_ps["show_earth"]    = st.checkbox("Earth texture",  value=_pb_ps["show_earth"],    key=f"pb_{_pb_mod['key']}_e")
        if "show_moon"     in _pb_ps: _pb_ps["show_moon"]     = st.checkbox("Moon texture",   value=_pb_ps["show_moon"],     key=f"pb_{_pb_mod['key']}_m")
        if "dark_bg"       in _pb_ps: _pb_ps["dark_bg"]       = st.checkbox("Dark bg",        value=_pb_ps["dark_bg"],       key=f"pb_{_pb_mod['key']}_d")
        if "show_lagrange" in _pb_ps: _pb_ps["show_lagrange"] = st.checkbox("L-points",       value=_pb_ps["show_lagrange"], key=f"pb_{_pb_mod['key']}_l")
        if "show_van_allen" in _pb_ps: _pb_ps["show_van_allen"]= st.checkbox("Van Allen",     value=_pb_ps["show_van_allen"],key=f"pb_{_pb_mod['key']}_va")
        if "show_dipole_axis" in _pb_ps: _pb_ps["show_dipole_axis"]= st.checkbox("Dipole axis",value=_pb_ps["show_dipole_axis"],key=f"pb_{_pb_mod['key']}_da")
        if "show_belt_residence" in _pb_ps:
            _pb_ps["show_belt_residence"] = st.checkbox(
                "☢ Show belt residence time", value=_pb_ps["show_belt_residence"], key=f"pb_{_pb_mod['key']}_br",
                help="Annotates the plot with minutes/orbit spent in the inner/outer Van Allen belts "
                     "(analytic estimate — see Settings tab).")
            if _pb_ps["show_belt_residence"]:
                _b = ss.get("_belt_residence") or _estimate_belt_residence(ss.a_km, ss.e, ss.n_orbits)
                st.caption(f"Inner {_b['t_inner_s']/60:.2f} min/orbit · Outer {_b['t_outer_s']/60:.2f} min/orbit")
        if "max_r_re"      in _pb_ps: _pb_ps["max_r_re"]      = st.slider("Max r (Rₑ)",5.0,20.0,float(_pb_ps["max_r_re"]),0.5,key=f"pb_{_pb_mod['key']}_mr")
        if "n_frames"      in _pb_ps: _pb_ps["n_frames"]      = st.number_input("Frames",10,100,int(_pb_ps["n_frames"]),key=f"pb_{_pb_mod['key']}_fr")
        if "show_surface"  in _pb_ps: _pb_ps["show_surface"]  = st.checkbox("Surface view",  value=_pb_ps["show_surface"],  key=f"pb_{_pb_mod['key']}_sv")
        if "show_orbit"    in _pb_ps: _pb_ps["show_orbit"]    = st.checkbox("Orbit view",    value=_pb_ps["show_orbit"],    key=f"pb_{_pb_mod['key']}_ov")
        if "show_polar"    in _pb_ps: _pb_ps["show_polar"]    = st.checkbox("Polar view",    value=_pb_ps["show_polar"],    key=f"pb_{_pb_mod['key']}_pv")
        if "views"         in _pb_ps:
            _pb_ps["views"] = st.multiselect("Views",["oblique","equatorial","polar"],default=_pb_ps["views"],key=f"pb_{_pb_mod['key']}_vw") or ["oblique"]
    ss[_pb_ps_key] = _pb_ps
    st.caption("Changing the plot above also toggles it in the full module grid below — "
               "one setting swaps what gets generated.")

    st.markdown("---")
    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        if st.button("✦ Select All"):
            for m in PLOT_MODULES: ss[f"mod_{m['key']}"] = True
    with qc2:
        if st.button("✦ Clear All"):
            for m in PLOT_MODULES: ss[f"mod_{m['key']}"] = False
    with qc3:
        if st.button("✦ Matplotlib"):
            for m in PLOT_MODULES: ss[f"mod_{m['key']}"] = (m["engine"]=="matplotlib")
    with qc4:
        if st.button("✦ Plotly"):
            for m in PLOT_MODULES: ss[f"mod_{m['key']}"] = (m["engine"]=="plotly")

    st.markdown("<br>", unsafe_allow_html=True)

    for i in range(0, len(PLOT_MODULES), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i+j >= len(PLOT_MODULES): break
            m   = PLOT_MODULES[i+j]
            key = f"mod_{m['key']}"
            on  = ss.get(key, False)
            badge_e = f'<span class="badge b-{"plotly" if m["engine"]=="plotly" else "mpl"}">{m["engine"].upper()}</span>'
            badge_o = f'<span class="badge b-{"html" if "html" in m["output"] else "png"}">{m["output"].upper()}</span>'
            with col:
                st.markdown(
                    f'<div class="plot-card {"active" if on else ""}">'
                    f'<div class="card-title">{m["title"]}</div>'
                    f'<div class="card-desc">{m["desc"]}</div>'
                    f'{badge_e}{badge_o}<br>'
                    f'<span style="font-size:0.63rem;color:var(--dim);">📏 {m["units"]}</span></div>',
                    unsafe_allow_html=True,
                )
                ss[key] = st.toggle(f"Enable", value=on, key=f"tog_{m['key']}")
                ps_key  = f"ps_{m['key']}"
                ps      = ss.get(ps_key, {})
                with st.expander("⚙ Settings"):
                    if "show_stars"    in ps: ps["show_stars"]    = st.checkbox("Stars",          value=ps["show_stars"],    key=f"{m['key']}_s")
                    if "show_earth"    in ps: ps["show_earth"]    = st.checkbox("Earth texture",  value=ps["show_earth"],    key=f"{m['key']}_e")
                    if "show_moon"     in ps: ps["show_moon"]     = st.checkbox("Moon texture",   value=ps["show_moon"],     key=f"{m['key']}_m")
                    if "dark_bg"       in ps: ps["dark_bg"]       = st.checkbox("Dark bg",        value=ps["dark_bg"],       key=f"{m['key']}_d")
                    if "show_lagrange" in ps: ps["show_lagrange"] = st.checkbox("L-points",       value=ps["show_lagrange"], key=f"{m['key']}_l")
                    if "show_van_allen" in ps: ps["show_van_allen"]= st.checkbox("Van Allen",     value=ps["show_van_allen"],key=f"{m['key']}_va")
                    if "show_dipole_axis" in ps: ps["show_dipole_axis"]= st.checkbox("Dipole axis",value=ps["show_dipole_axis"],key=f"{m['key']}_da")
                    if "show_belt_residence" in ps: ps["show_belt_residence"]= st.checkbox("☢ Belt residence time",value=ps["show_belt_residence"],key=f"{m['key']}_br")
                    if "max_r_re"      in ps: ps["max_r_re"]      = st.slider("Max r (Rₑ)",5.0,20.0,float(ps["max_r_re"]),0.5,key=f"{m['key']}_mr")
                    if "n_frames"      in ps: ps["n_frames"]      = st.number_input("Frames",10,100,int(ps["n_frames"]),key=f"{m['key']}_fr")
                    if "show_surface"  in ps: ps["show_surface"]  = st.checkbox("Surface view",  value=ps["show_surface"],  key=f"{m['key']}_sv")
                    if "show_orbit"    in ps: ps["show_orbit"]    = st.checkbox("Orbit view",    value=ps["show_orbit"],    key=f"{m['key']}_ov")
                    if "show_polar"    in ps: ps["show_polar"]    = st.checkbox("Polar view",    value=ps["show_polar"],    key=f"{m['key']}_pv")
                    if "seed_lats"     in ps:
                        raw = st.text_input("Seed lats (°)", value=",".join(str(x) for x in ps["seed_lats"]), key=f"{m['key']}_sl")
                        try: ps["seed_lats"] = [float(x.strip()) for x in raw.split(",") if x.strip()]
                        except ValueError: pass
                    if "views"         in ps:
                        ps["views"] = st.multiselect("Views",["oblique","equatorial","polar"],default=ps["views"],key=f"{m['key']}_vw") or ["oblique"]
                ss[ps_key] = ps

    st.markdown("---")
    selected = [m for m in PLOT_MODULES if ss.get(f"mod_{m['key']}", False)]
    n = len(selected)
    label = f"▶ GENERATE  {n} PLOT{'S' if n!=1 else ''}" if n else "— SELECT AT LEAST ONE MODULE —"
    st.markdown('<div class="run-btn">', unsafe_allow_html=True)
    if st.button(label, disabled=(n==0), key="gen_btn"):
        log = [f"═══ Run — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ═══",
               f"  a={ss.a_km:.0f}km e={ss.e:.5f} i={ss.inc_deg:.2f}° [{d['regime']}]",
               f"  Propagator: {ss.propagator} / {ss.gravity} / {ss.third_body} / {ss.non_grav}",
               "───"]
        Path(ss.output_dir).mkdir(parents=True, exist_ok=True)
        ok = 0
        with st.spinner("Running…"):
            for m in selected:
                if run_script(m, log): ok += 1
                log.append("───")
        log.append(f"✔ {ok}/{n} succeeded → {ss.output_dir}")
        ss.console_log = log
        if ok == n: st.success(f"All {n} plots done → {ss.output_dir}")
        elif ok:    st.warning(f"{ok}/{n} succeeded — see run log below")
        else:       st.error("All failed — see run log below")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Run log (former standalone CONSOLE tab, folded in here) ──────────────
    if ss.console_log:
        with st.expander("🖥 Run log", expanded=False):
            _log_html = '<div class="console">'
            for _line in ss.console_log:
                _cls = ("c-ok" if _line.startswith("✔") else
                        "c-err" if _line.startswith("✖") or "Error" in _line.lower() else
                        "c-warn" if _line.startswith(("⚠","⏱")) else
                        "c-dim" if _line.startswith(("═","─")) else "c-info")
                _log_html += f'<span class="{_cls}">{_line}</span>\n'
            _log_html += '</div>'
            st.markdown(_log_html, unsafe_allow_html=True)
            _lc1, _lc2 = st.columns([1, 4])
            with _lc1:
                if st.button("Clear", key="clear_console_log"):
                    ss.console_log = []; st.rerun()
            with _lc2:
                st.download_button("Download log", data="\n".join(ss.console_log),
                                    file_name=f"ssapy_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain", key="dl_console_log")


# ══════════════════════════════════════════════════════
# TAB 3 — MISSION PLANNER
# ══════════════════════════════════════════════════════
with tab_mission:
    st.markdown(
        '<p style="font-size:0.71rem;color:#7290B0;font-family:JetBrains Mono,monospace;'
        'margin-bottom:0.8rem">Four steps: check your origin orbit → add targets → '
        'set fuel/time limits → plan the transfer.</p>',
        unsafe_allow_html=True,
    )

    mc1, mc2 = st.columns([1, 1])

    with mc1:
        with st.container(border=True):
            st.markdown('<h4 style="color:var(--green);font-family:JetBrains Mono,monospace;text-transform:uppercase;letter-spacing:0.1em;font-size:0.8rem;margin:0 0 0.6rem">① Origin orbit</h4>', unsafe_allow_html=True)
            st.markdown(
                f'<span class="r-label">Current: </span>'
                f'<span class="r-val">{d["regime"]}  a={ss.a_km:.0f}km  e={ss.e:.4f}  i={ss.inc_deg:.2f}°</span>'
                f'<br><span style="font-size:0.68rem;color:var(--dim)">Edit this in the sidebar '
                f'(Keplerian elements / TLE / Preset) — it\'s the same orbit shown in 3D PREVIEW.</span>',
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown('<h4 style="color:var(--green);font-family:JetBrains Mono,monospace;text-transform:uppercase;letter-spacing:0.1em;font-size:0.8rem;margin:0 0 0.6rem">② Target orbits</h4>', unsafe_allow_html=True)
            targets = ss.mp_targets

            if not targets:
                st.caption("No targets yet — add one below.")
            for idx, tgt in enumerate(targets):
                with st.expander(f"🎯 Target {idx+1}: {tgt.get('name','—')} — "
                                 f"a={tgt.get('a_km',0):,.0f} km, e={tgt.get('e',0):.3f}, "
                                 f"i={tgt.get('inc_deg',0):.1f}°"):
                    tgt["name"]     = st.text_input("Name",      value=tgt.get("name",""),    key=f"tn_{idx}")
                    tgt["a_km"]     = st.number_input("Semi-major axis (km)",value=float(tgt.get("a_km",RE_KM+600)),min_value=RE_KM+100,step=10.0,key=f"ta_{idx}",
                                                       help="Roughly the average of the orbit's highest and lowest altitude above Earth's centre.")
                    tgt["e"]        = st.number_input("Eccentricity",     value=float(tgt.get("e",0.001)),min_value=0.0,max_value=0.9999,step=0.001,key=f"te_{idx}",
                                                       help="0 = perfectly circular. Closer to 1 = more elongated ellipse.")
                    tgt["inc_deg"]  = st.number_input("Inclination (°)",   value=float(tgt.get("inc_deg",51.6)),key=f"ti_{idx}",
                                                       help="Tilt of the orbit plane relative to Earth's equator. 0°=equatorial, 90°=polar.")
                    tgt["raan_deg"] = st.number_input("RAAN (°)",  value=float(tgt.get("raan_deg",0.0)),key=f"tr_{idx}",
                                                       help="Right Ascension of Ascending Node — where the orbit crosses the equator heading north.")
                    tgt["argp_deg"] = st.number_input("Argument of perigee (°)",  value=float(tgt.get("argp_deg",0.0)),key=f"tap_{idx}",
                                                       help="Orientation of the orbit's closest-approach point within its plane.")
                    if st.button(f"🗑 Remove target {idx+1}", key=f"rm_{idx}"):
                        targets.pop(idx); ss.mp_targets = targets; st.rerun()

            st.markdown(
                '<div style="background:#0B1626;border:1px dashed var(--border);border-radius:6px;'
                'padding:0.6rem 0.8rem;margin-top:0.5rem;">'
                '<span style="font-size:0.72rem;color:var(--star);font-weight:600;">+ Add a target orbit</span>'
                '</div>', unsafe_allow_html=True,
            )
            _tgt_add_mode = st.radio("How", ["From preset", "Manual entry"], horizontal=True,
                                      key="mp_tgt_add_mode", label_visibility="collapsed")
            if _tgt_add_mode == "From preset":
                _pa1, _pa2 = st.columns([3, 1])
                with _pa1:
                    _tgt_preset = st.selectbox("Preset orbit", list(PRESETS.keys()),
                                                key="mp_tgt_preset_sel", label_visibility="collapsed")
                with _pa2:
                    if st.button("+ Add", key="mp_add_preset_tgt", type="primary"):
                        _pp = PRESETS[_tgt_preset]
                        targets.append(dict(name=_tgt_preset, a_km=_pp["a_km"], e=_pp["e"],
                                            inc_deg=_pp["inc_deg"], raan_deg=_pp["raan_deg"],
                                            argp_deg=_pp["argp_deg"]))
                        ss.mp_targets = targets; st.rerun()
            else:
                if st.button("+ Add blank target orbit (edit it above once added)", key="mp_add_blank_tgt"):
                    targets.append(dict(name=f"Target {len(targets)+1}",
                                        a_km=RE_KM+600, e=0.001, inc_deg=51.6, raan_deg=0, argp_deg=0))
                    ss.mp_targets = targets; st.rerun()

    with mc2:
        with st.container(border=True):
            st.markdown('<h4 style="color:var(--green);font-family:JetBrains Mono,monospace;text-transform:uppercase;letter-spacing:0.1em;font-size:0.8rem;margin:0 0 0.6rem">③ Fuel &amp; time limits</h4>', unsafe_allow_html=True)
            _OBJ = {"min_dv": "Minimize fuel use (Δv)", "min_time": "Minimize time (TOF)"}
            ss.mp_objective  = st.selectbox("Goal", list(_OBJ.keys()), format_func=lambda k: _OBJ[k],
                                             index=list(_OBJ.keys()).index(ss.mp_objective),
                                             help="What the planner prioritizes when a trade-off exists.")
            ss.mp_dv_budget  = st.number_input("Fuel budget — Δv (m/s)", value=float(ss.mp_dv_budget),
                                                min_value=0.0, step=100.0,
                                                help="Total velocity change available. Plans exceeding this get flagged.")
            _tc1, _tc2 = st.columns(2)
            with _tc1:
                ss.mp_tof_min = st.number_input("Min time of flight (hr)", value=float(ss.mp_tof_min), min_value=0.1, step=0.5)
            with _tc2:
                ss.mp_tof_max = st.number_input("Max time of flight (hr)", value=float(ss.mp_tof_max), min_value=0.1, step=1.0)

            with st.expander("⚙ Advanced (method, mission type)"):
                _OPT = {"greedy": "Greedy — plan each leg one at a time (fast)",
                        "joint":  "Joint — optimize all legs together (slower; UI only for now, still runs greedy)"}
                ss.mp_optimizer  = st.selectbox("Method", list(_OPT.keys()), format_func=lambda k: _OPT[k],
                                                 index=list(_OPT.keys()).index(ss.mp_optimizer))
                _GEO = {"insertion": "Insertion — arrive and stay (standard orbit change)",
                        "rendezvous": "Rendezvous — match position AND velocity with a target",
                        "intercept": "Intercept — reach the target's position only, timing matters"}
                ss.mp_geometry   = st.selectbox("Mission type", list(_GEO.keys()), format_func=lambda k: _GEO[k],
                                                 index=list(_GEO.keys()).index(ss.mp_geometry))

    with st.container(border=True):
        st.markdown('<h4 style="color:var(--green);font-family:JetBrains Mono,monospace;text-transform:uppercase;letter-spacing:0.1em;font-size:0.8rem;margin:0 0 0.6rem">④ Plan the mission</h4>', unsafe_allow_html=True)
        if not ss.mp_targets:
            st.info("Add at least one target orbit (above) before planning a mission.")
        elif not _CORE_OK:
            st.warning("Core library unavailable — mission planning needs it. See the banner at the top of the page.")
        else:
            st.caption(f"Ready to plan {len(ss.mp_targets)} leg(s) from the current orbit "
                       f"({d['regime']}, a={ss.a_km:,.0f} km) through: " +
                       " → ".join(t.get("name","Target") for t in ss.mp_targets))
            if st.button("▶ PLAN MISSION", key="plan_btn", type="primary"):
                    with st.spinner("Planning transfer sequence…"):
                        try:
                            from core.orbit_state import PropagatorConfig
                            origin = build_orbital_state()
                            legs = []
                            current = origin
                            for tgt in ss.mp_targets:
                                target = OrbitalState(
                                    a_km=tgt["a_km"], e=tgt["e"],
                                    inc_deg=tgt["inc_deg"], raan_deg=tgt["raan_deg"],
                                    argp_deg=tgt["argp_deg"], nu_deg=0,
                                    name=tgt.get("name","Target"),
                                )
                                # Hohmann-ish Δv estimate (coplanar)
                                r1 = current.a_km; r2 = target.a_km
                                v1 = np.sqrt(MU/r1); v2 = np.sqrt(MU/r2)
                                a_t = (r1+r2)/2
                                dv1 = abs(np.sqrt(MU*(2/r1-1/a_t)) - v1)
                                dv2 = abs(v2 - np.sqrt(MU*(2/r2-1/a_t)))
                                tof_hr = np.pi*np.sqrt(a_t**3/MU)/3600
                                legs.append(dict(
                                    origin_name=current.name,
                                    target_name=target.name,
                                    dv1_ms=dv1*1000, dv2_ms=dv2*1000,
                                    total_dv_ms=(dv1+dv2)*1000,
                                    tof_hr=tof_hr,
                                    origin=current, target=target,
                                ))
                                current = target
                            ss.mp_results = legs
                        except Exception as ex:
                            st.error(f"Planning error: {ex}")

            if ss.mp_results:
                with st.container(border=True):
                    st.markdown('<h4 style="color:var(--green);font-family:JetBrains Mono,monospace;text-transform:uppercase;letter-spacing:0.1em;font-size:0.8rem;margin:0 0 0.6rem">Transfer plan</h4>', unsafe_allow_html=True)
                    total_dv = sum(l["total_dv_ms"] for l in ss.mp_results)
                    total_tof= sum(l["tof_hr"] for l in ss.mp_results)
                    st.markdown(
                        f'<span class="r-label">TOTAL Δv  </span><span class="r-val">{total_dv:.1f}</span><span class="r-unit"> m/s</span><br>'
                        f'<span class="r-label">TOTAL TOF </span><span class="r-val">{total_tof:.2f}</span><span class="r-unit"> hr</span><br>',
                        unsafe_allow_html=True,
                    )
                    for i, leg in enumerate(ss.mp_results):
                        st.markdown(
                            f'<div class="burn-card">'
                            f'<span class="r-label">Leg {i+1}  </span>'
                            f'<span class="r-val">{leg["origin_name"]} → {leg["target_name"]}</span><br>'
                            f'<span class="r-label">Δv₁  </span><span class="r-val">{leg["dv1_ms"]:.1f}</span><span class="r-unit"> m/s</span>  '
                            f'<span class="r-label">Δv₂  </span><span class="r-val">{leg["dv2_ms"]:.1f}</span><span class="r-unit"> m/s</span>  '
                            f'<span class="r-label">TOF  </span><span class="r-val">{leg["tof_hr"]:.2f}</span><span class="r-unit"> hr</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        # add burns to satellite burn list
                        if st.button(f"Add leg {i+1} burns to satellite", key=f"add_burns_{i}"):
                            T_s = 2*np.pi*np.sqrt(leg["origin"].a_km**3/MU)
                            t_offset = sum(l["tof_hr"] for l in ss.mp_results[:i]) * 3600
                            ss.burns.append(dict(
                                label=f"Leg {i+1} Δv₁",
                                t_s=t_offset,
                                dv_t=leg["dv1_ms"]/1000, dv_n=0.0, dv_w=0.0,
                                mode="impulsive",
                            ))
                            ss.burns.append(dict(
                                label=f"Leg {i+1} Δv₂",
                                t_s=t_offset + leg["tof_hr"]*3600,
                                dv_t=leg["dv2_ms"]/1000, dv_n=0.0, dv_w=0.0,
                                mode="impulsive",
                            ))
                            st.success(f"Added leg {i+1} burns")

                # multi-orbit preview
                if _CORE_OK and st.button("🔭 Preview all orbits", key="prev_all"):
                    try:
                        states = [build_orbital_state()] + [
                            OrbitalState(a_km=l["target"].a_km, e=l["target"].e,
                                         inc_deg=l["target"].inc_deg, name=l["target_name"])
                            for l in ss.mp_results
                        ]
                        tp = ss.earth_texture if Path(ss.earth_texture).exists() else None
                        multi_fig = PlotlyScene.compare(states, n_orbits=2, texture_path=tp)
                        st.plotly_chart(multi_fig, width='stretch')
                    except Exception as ex:
                        st.error(f"Preview error: {ex}")

    # ── Transfer calculator ────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown('<h4 style="color:var(--green);font-family:JetBrains Mono,monospace;text-transform:uppercase;letter-spacing:0.1em;font-size:0.8rem;margin:0 0 0.6rem">🚀 Orbital transfer calculator</h4>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.71rem;color:#7290B0;font-family:JetBrains Mono,monospace;">'
            'Compares the current origin orbit against a chosen target using three '
            'classic coplanar transfer strategies. Units: Δv in m/s, TOF in hours, '
            'r in km.</p>', unsafe_allow_html=True,
        )
        if ss.mp_targets:
            _xt_names = [t.get("name", f"Target {i+1}") for i, t in enumerate(ss.mp_targets)]
            ss.mp_xfer_target_idx = st.selectbox(
                "Target orbit", list(range(len(_xt_names))),
                format_func=lambda i: _xt_names[i],
                index=min(ss.mp_xfer_target_idx, len(_xt_names)-1), key="mp_xfer_sel",
            )
            _xt = ss.mp_targets[ss.mp_xfer_target_idx]
            r1_km = float(ss.a_km); r2_km = float(_xt.get("a_km", RE_KM+600))
            v1 = np.sqrt(MU/r1_km); v2 = np.sqrt(MU/r2_km)

            # 1) Hohmann transfer (2 burns, coplanar)
            a_h = (r1_km + r2_km) / 2.0
            dv1_h = abs(np.sqrt(MU*(2/r1_km - 1/a_h)) - v1)
            dv2_h = abs(v2 - np.sqrt(MU*(2/r2_km - 1/a_h)))
            dv_h  = (dv1_h + dv2_h) * 1000
            tof_h = np.pi * np.sqrt(a_h**3 / MU) / 3600.0

            # 2) Bi-elliptic transfer (3 burns, via an intermediate high apogee r_b)
            r_b = max(r1_km, r2_km) * 2.5
            a_t1 = (r1_km + r_b) / 2.0
            a_t2 = (r_b + r2_km) / 2.0
            dv1_be = abs(np.sqrt(MU*(2/r1_km - 1/a_t1)) - v1)
            dv2_be = abs(np.sqrt(MU*(2/r_b - 1/a_t2)) - np.sqrt(MU*(2/r_b - 1/a_t1)))
            dv3_be = abs(v2 - np.sqrt(MU*(2/r2_km - 1/a_t2)))
            dv_be  = (dv1_be + dv2_be + dv3_be) * 1000
            tof_be = (np.pi*np.sqrt(a_t1**3/MU) + np.pi*np.sqrt(a_t2**3/MU)) / 3600.0

            # 3) Hohmann + inclination/plane change combined at apogee burn
            di_deg  = abs(float(_xt.get("inc_deg", ss.inc_deg)) - ss.inc_deg)
            v_apog  = np.sqrt(MU*(2/r2_km - 1/a_h))
            dv_plane = 2 * v_apog * np.sin(np.radians(di_deg)/2.0)
            dv_combined = dv_h + dv_plane * 1000

            _xfer_rows = [
                dict(name="Hohmann (2-burn)",           dv_ms=dv_h,        tof_hr=tof_h),
                dict(name="Bi-elliptic (3-burn)",       dv_ms=dv_be,       tof_hr=tof_be),
                dict(name=f"Hohmann + plane change ({di_deg:.1f}°)", dv_ms=dv_combined, tof_hr=tof_h),
            ]
            _XT = ["Hohmann","Bi-elliptic","Hohmann + plane change"]
            ss.mp_xfer_type = st.radio("Strategy", _XT,
                                        index=_XT.index(ss.mp_xfer_type) if ss.mp_xfer_type in _XT else 0,
                                        horizontal=True, key="mp_xfer_type_radio")

            for row in _xfer_rows:
                _hl = "border-left:3px solid var(--green);" if row["name"].startswith(ss.mp_xfer_type) else ""
                st.markdown(
                    f'<div class="burn-card" style="{_hl}">'
                    f'<span class="r-label">{row["name"]}</span><br>'
                    f'<span class="r-label">Δv   </span><span class="r-val">{row["dv_ms"]:.1f}</span><span class="r-unit"> m/s</span>  '
                    f'<span class="r-label">TOF  </span><span class="r-val">{row["tof_hr"]:.2f}</span><span class="r-unit"> hr</span>'
                    f'</div>', unsafe_allow_html=True,
                )
            st.caption(
                f"r₁={r1_km:,.0f} km · r₂={r2_km:,.0f} km · Δi={di_deg:.2f}° "
                "· bi-elliptic is cheaper than Hohmann only when r₂/r₁ is large "
                "(classically ≳11.9×); otherwise Hohmann wins on both Δv and time."
            )
        else:
            st.caption("Add at least one target orbit above to compare transfer strategies.")

    # ── Burns / Manoeuvres (merged from the former standalone BURNS tab) ────
    with st.container(border=True):
        st.markdown(
            '<h4 style="color:var(--green);font-family:JetBrains Mono,monospace;'
            'text-transform:uppercase;letter-spacing:0.1em;font-size:0.8rem;'
            'margin:0 0 0.6rem">🔥 Burns / Manoeuvres</h4>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="font-size:0.71rem;color:#7290B0;font-family:JetBrains Mono,monospace;'
            'margin-bottom:0.8rem">Manoeuvre events in the NTW frame '
            '(T=along-track, N=cross-track, W=orbit-normal). Units: Δv in km/s per '
            'component (readouts convert to m/s); epoch offset in seconds from the orbit epoch.</p>',
            unsafe_allow_html=True,
        )

        # Action row (buttons + live totals) sits above the list, in a single
        # horizontal row, instead of a narrow side column that looked stranded
        # and empty whenever there were no burns yet.
        _total_dv = sum(np.sqrt(b["dv_t"]**2+b["dv_n"]**2+b["dv_w"]**2)*1000 for b in ss.burns)
        _ba1, _ba2, _ba3 = st.columns([1, 1, 2])
        with _ba1:
            if st.button("+ Add burn", key="add_burn_btn"):
                ss.burns.append(dict(label=f"Burn {len(ss.burns)+1}",
                                      t_s=ss.n_orbits/2*2*np.pi*np.sqrt(ss.a_km**3/MU),
                                      dv_t=0.0, dv_n=0.0, dv_w=0.0, mode="impulsive"))
                st.rerun()
        with _ba2:
            if st.button("Clear all burns", key="clear_burns_btn", disabled=not ss.burns):
                ss.burns = []
                st.rerun()
        with _ba3:
            st.markdown(
                f'<div class="readout" style="margin:0;padding:0.4rem 0.8rem;">'
                f'<span class="r-label">BURNS   </span><span class="r-val">{len(ss.burns)}</span>'
                f'&nbsp;&nbsp;<span class="r-label">TOTAL Δv</span>'
                f'<span class="r-val">{_total_dv:.1f}</span><span class="r-unit"> m/s</span>'
                f'</div>', unsafe_allow_html=True,
            )

        if not ss.burns:
            st.caption("No burns yet — add one above to define a manoeuvre.")
        for idx, burn in enumerate(ss.burns):
            with st.expander(f"🔥 {burn.get('label','Burn')}  —  "
                             f"Δv={np.sqrt(burn['dv_t']**2+burn['dv_n']**2+burn['dv_w']**2)*1000:.1f} m/s  "
                             f"t+{burn['t_s']:.0f}s"):
                burn["label"] = st.text_input("Label",  value=burn["label"], key=f"bl_{idx}")
                burn["t_s"]   = st.number_input("Epoch offset (s)", value=float(burn["t_s"]),
                                                 min_value=0.0, step=60.0, key=f"bt_{idx}")
                bc1a, bc1b, bc1c = st.columns(3)
                with bc1a: burn["dv_t"] = st.number_input("ΔvT km/s (along-track)",
                                                            value=float(burn["dv_t"]),step=0.001,format="%.4f",key=f"bdt_{idx}")
                with bc1b: burn["dv_n"] = st.number_input("ΔvN km/s (cross-track)",
                                                            value=float(burn["dv_n"]),step=0.001,format="%.4f",key=f"bdn_{idx}")
                with bc1c: burn["dv_w"] = st.number_input("ΔvW km/s (orbit-normal)",
                                                            value=float(burn["dv_w"]),step=0.001,format="%.4f",key=f"bdw_{idx}")
                burn["mode"] = st.radio("Mode", ["impulsive","finite"],
                                         index=["impulsive","finite"].index(burn["mode"]),
                                         horizontal=True, key=f"bm_{idx}")
                if st.button(f"🗑 Remove burn {idx+1}", key=f"rb_{idx}"):
                    ss.burns.pop(idx); st.rerun()
            ss.burns[idx] = burn


# ══════════════════════════════════════════════════════
# TAB 4 — SETTINGS

# ══════════════════════════════════════════════════════
with tab_settings:
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**Global visual defaults**")
        g_stars  = st.toggle("Stars on all exports",      value=True, key="gs")
        g_dark   = st.toggle("Dark background",           value=True, key="gd")
        g_earth  = st.toggle("Earth texture",             value=True, key="ge")
        g_moon   = st.toggle("Moon texture",              value=True, key="gm")
        g_lag    = st.toggle("Lagrange points",           value=True, key="gl")
        g_va     = st.toggle("Van Allen (magfield plot)", value=True, key="gva")
        if st.button("Apply to all export modules"):
            for m in PLOT_MODULES:
                ps = ss.get(f"ps_{m['key']}", {})
                if "show_stars"    in ps: ps["show_stars"]    = g_stars
                if "dark_bg"       in ps: ps["dark_bg"]       = g_dark
                if "show_earth"    in ps: ps["show_earth"]    = g_earth
                if "show_moon"     in ps: ps["show_moon"]     = g_moon
                if "show_lagrange" in ps: ps["show_lagrange"] = g_lag
                if "show_van_allen" in ps: ps["show_van_allen"]= g_va
                ss[f"ps_{m['key']}"] = ps
            st.success("Applied.")

        st.markdown("---")
        st.markdown("**Starfield epoch (star precession)**")
        st.markdown(
            '<p style="font-size:0.69rem;color:#7290B0;font-family:JetBrains Mono,monospace;'
            'margin:0 0 0.5rem">IAU 1976 precession shifts J2000 catalog positions to the '
            'actual mission date (~50 arcsec/yr, ~3 px at 2022).</p>',
            unsafe_allow_html=True,
        )

        epoch_mode = st.radio(
            "Epoch source",
            ["Auto (= orbit epoch)", "Custom date"],
            index=0 if ss.starfield_epoch_mode == "auto" else 1,
            horizontal=True, key="sf_mode_radio",
        )
        ss.starfield_epoch_mode = "auto" if epoch_mode == "Auto (= orbit epoch)" else "custom"

        if ss.starfield_epoch_mode == "custom":
            import datetime as _dt
            default_date = (ss.starfield_epoch_date
                            if isinstance(ss.starfield_epoch_date, _dt.date)
                            else _dt.date.today())
            picked = st.date_input(
                "Observation date",
                value=default_date,
                min_value=_dt.date(1950, 1, 1),
                max_value=_dt.date(2100, 1, 1),
                key="sf_date_picker",
            )
            ss.starfield_epoch_date = picked

        # Show live precession info
        epoch_jd = _get_starfield_epoch_jd()
        if epoch_jd is not None:
            T_yr = (epoch_jd - 2_451_545.0) / 365.25
            shift_arcsec = abs(50.3 * T_yr)
            sign = "after" if T_yr >= 0 else "before"
            label = f"~{shift_arcsec:.0f}\" shift ({abs(T_yr):.1f} yr {sign} J2000)"
            color = "#2BFFB0" if shift_arcsec > 200 else "#7290B0"
            st.markdown(
                f'<div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;'
                f'color:{color};margin-top:0.3rem">⟳ {label}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;'
                'color:#7290B0;margin-top:0.3rem">⟳ J2000.0 (no precession)</div>',
                unsafe_allow_html=True,
            )
        st.markdown("---")
        st.markdown("**☢ Magnetic field / Van Allen belt residence**")
        st.markdown(
            '<p style="font-size:0.69rem;color:#7290B0;font-family:JetBrains Mono,monospace;'
            'margin:0 0 0.5rem">Analytic per-orbit estimate of time spent inside the '
            'inner (1.2–2.5 Rₑ) and outer (3–10 Rₑ) belts, feeding the Magnetic Field '
            'plot config. Geometric approximation, not a flux/dose model.</p>',
            unsafe_allow_html=True,
        )
        _belt = _estimate_belt_residence(ss.a_km, ss.e, ss.n_orbits)
        ss["_belt_residence"] = _belt   # cached for run_script() cfg + export
        st.markdown(
            f'<div class="readout">'
            f'<span class="r-label">PERIOD       </span><span class="r-val">{_belt["period_s"]/60:.1f}</span><span class="r-unit"> min</span><br>'
            f'<span class="r-label">INNER BELT   </span><span class="r-val">{_belt["t_inner_s"]/60:.2f}</span><span class="r-unit"> min/orbit</span> '
            f'<span class="r-unit">({_belt["frac_inner"]*100:.1f}%)</span><br>'
            f'<span class="r-label">OUTER BELT   </span><span class="r-val">{_belt["t_outer_s"]/60:.2f}</span><span class="r-unit"> min/orbit</span> '
            f'<span class="r-unit">({_belt["frac_outer"]*100:.1f}%)</span><br>'
            f'<span class="r-label">OVER {ss.n_orbits:g} ORBITS</span><span class="r-val"> {(_belt["t_inner_total_s"]+_belt["t_outer_total_s"])/60:.1f}</span><span class="r-unit"> min total in belts</span>'
            f'</div>', unsafe_allow_html=True,
        )
    with sc2:
        st.markdown("**Current config snapshot**")
        st.markdown(
            f'<div class="readout">'
            f'<span class="r-label">PROPAGATOR </span><span class="r-val">{ss.propagator}</span><br>'
            f'<span class="r-label">GRAVITY    </span><span class="r-val">{ss.gravity}</span><br>'
            f'<span class="r-label">3RD BODY   </span><span class="r-val">{ss.third_body}</span><br>'
            f'<span class="r-label">NON-GRAV   </span><span class="r-val">{ss.non_grav}</span><br>'
            f'<span class="r-label">FRAME      </span><span class="r-val">{ss.frame}</span><br>'
            f'<span class="r-label">EPOCH      </span><span class="r-val">{ss.epoch}</span><br>'
            f'<span class="r-label">N_ORBITS   </span><span class="r-val">{ss.n_orbits}</span><br>'
            f'<span class="r-label">DT_S       </span><span class="r-val">{ss.dt_s}</span><br>'
            f'<span class="r-label">BURNS      </span><span class="r-val">{len(ss.burns)}</span><br>'
            f'<span class="r-label">CORE       </span>'
            f'<span class="{"r-ok" if _CORE_OK else "r-err"}">'
            f'{"✔ OK" if _CORE_OK else "✖ unavailable"}</span>'
            f'</div>', unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════
# TAB 6 — STAR MAP
# ══════════════════════════════════════════════════════
with tab_starmap:
    sm1, sm2 = st.columns([3, 1])

    with sm2:
        st.markdown("**Observation epoch**")
        import datetime as _dt

        sm_mode = st.radio(
            "Epoch source",
            ["Auto (orbit epoch)", "Custom date", "Far future (year slider)"],
            index={"auto": 0, "custom": 1, "far_future": 2}.get(ss.starfield_epoch_mode, 0),
            horizontal=False, key="sm_epoch_mode",
        )
        ss.starfield_epoch_mode = {"Auto (orbit epoch)": "auto",
                                    "Custom date": "custom",
                                    "Far future (year slider)": "far_future"}[sm_mode]

        if ss.starfield_epoch_mode == "custom":
            default_date = (ss.starfield_epoch_date
                            if isinstance(ss.starfield_epoch_date, _dt.date)
                            else _dt.date.today())
            picked = st.date_input(
                "Date",
                value=default_date,
                min_value=_dt.date(1, 1, 1),
                max_value=_dt.date(9999, 12, 31),
                key="sm_date_picker",
            )
            ss.starfield_epoch_date = picked

        elif ss.starfield_epoch_mode == "far_future":
            _MILESTONES = {
                2000:  "J2000 (baseline)",
                2500:  "~2 500 — Vega drifts ~2°",
                5000:  "~5 000 — clear naked-eye shift",
                10000: "~10 000 — North Star → Deneb",
                14000: "~14 000 — Vega becomes pole star",
            }
            ss.sm_far_year = int(st.slider(
                "Year", min_value=2000, max_value=14000,
                value=int(ss.get("sm_far_year", 2000)),
                step=100, key="sm_year_slider",
            ))
            # Milestone hint
            _closest = min(_MILESTONES, key=lambda y: abs(y - ss.sm_far_year))
            if abs(_closest - ss.sm_far_year) < 300:
                st.caption(_MILESTONES[_closest])

            # Step buttons
            _ba, _bb = st.columns(2)
            with _ba:
                if st.button("◀ −100 yr", key="sm_step_back"):
                    ss.sm_far_year = max(2000, int(ss.sm_far_year) - 100)
                    st.rerun()
            with _bb:
                if st.button("▶ +100 yr", key="sm_step_fwd"):
                    ss.sm_far_year = min(14000, int(ss.sm_far_year) + 100)
                    st.rerun()

            # Jump buttons for landmark years
            st.caption("Jump to:")
            _j1, _j2 = st.columns(2)
            with _j1:
                if st.button("5 000", key="sm_j5k"):
                    ss.sm_far_year = 5000; st.rerun()
                if st.button("10 000", key="sm_j10k"):
                    ss.sm_far_year = 10000; st.rerun()
            with _j2:
                if st.button("3 000", key="sm_j3k"):
                    ss.sm_far_year = 3000; st.rerun()
                if st.button("14 000", key="sm_j14k"):
                    ss.sm_far_year = 14000; st.rerun()

        epoch_jd = _get_starfield_epoch_jd()
        if epoch_jd is not None:
            T_yr = (epoch_jd - 2_451_545.0) / 365.25
            shift_approx = abs(50.3 * T_yr)
            # Color-code by how visible the shift is
            if shift_approx > 50_000:   sh_color = "#FF7088"
            elif shift_approx > 10_000: sh_color = "#FF9900"
            elif shift_approx > 1_000:  sh_color = "#2BFFB0"
            else:                        sh_color = "#FFC454"
            shift_str = (f"{shift_approx/3600:.1f}°" if shift_approx > 3600
                         else f"{shift_approx:.0f}\"")
            yr_label = (f"{2000 + T_yr:.0f}" if ss.starfield_epoch_mode == "far_future"
                        else f"{T_yr:+.2f} yr")
            st.markdown(
                f'<div class="readout">'
                f'<span class="r-label">EPOCH </span>'
                f'<span class="r-val">{yr_label}</span><br>'
                f'<span class="r-label">JD    </span>'
                f'<span class="r-val">{epoch_jd:.1f}</span><br>'
                f'<span class="r-label">SHIFT </span>'
                f'<span style="color:{sh_color};font-family:JetBrains Mono,monospace">'
                f'{shift_str}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="readout"><span class="r-dim">J2000.0<br>no precession</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("**Catalog**")
        sm_mag = st.slider("Mag limit", 3.0, 8.0, 6.5, 0.5, key="sm_mag")
        cat_ok = Path(ss.star_catalog).exists()
        st.markdown(
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;'
            f'color:{"#2BFFB0" if cat_ok else "#FF7088"};">'
            f'{"✔ catalog found" if cat_ok else "✖ catalog not found"}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        gen_btn = st.button("▶ Generate star map", key="sm_gen",
                            help="Rebuild the plot with the current date")

        st.markdown("**Legend**")
        st.markdown(
            '<div style="font-family:JetBrains Mono,monospace;font-size:0.67rem;'
            'color:#7290B0;line-height:1.9">'
            '○ open = J2000<br>'
            '● filled = epoch<br>'
            '— line = shift<br>'
            '· dotted = depth range<br>'
            '· dim = catalog J2000<br>'
            '· bright = catalog epoch</div>',
            unsafe_allow_html=True,
        )

        # ── Star search ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**Star search**")
        sm_query = st.text_input(
            "Search",
            placeholder="Sirius · HR 2491 · HIP 32349 · HD 48915 · Ori",
            key="sm_search_query",
            label_visibility="collapsed",
            help="Search by proper name, Bayer designation, constellation abbreviation, "
                 "or catalog number (HR / HD / HIP prefix required for numbers).",
        )

        _cat_ok_search = Path(ss.star_catalog).exists()
        if sm_query and _cat_ok_search:
            _sm_results = _search_star_catalog(sm_query, ss.star_catalog)
            if _sm_results:
                # Radio selector — show up to 10 hits
                _labels = []
                for _r in _sm_results:
                    _cat = " · ".join(filter(None, [
                        f"HR {_r['hr']}" if _r.get("hr") else None,
                        f"HD {_r['hd']}" if _r.get("hd") else None,
                    ]))
                    _labels.append(f"{_r['name']}  mag {_r['mag']:.1f}  {_r['spect']}"
                                   + (f"  [{_cat}]" if _cat else ""))
                _sel_label = st.radio(
                    "Results", _labels, key="sm_search_sel",
                    label_visibility="collapsed",
                )
                _sel_idx = _labels.index(_sel_label) if _sel_label in _labels else 0
                _star = _sm_results[_sel_idx]

                # Detail card
                _dist_str = (f"{_star['dist_ly']:.1f} ly"
                             if _star.get("dist_ly") else "—")
                _ids = "  ".join(filter(None, [
                    f"HR {_star['hr']}"  if _star.get("hr")  else None,
                    f"HD {_star['hd']}"  if _star.get("hd")  else None,
                    f"HIP {_star['hip']}" if _star.get("hip") else None,
                ]))
                st.markdown(
                    f'<div style="background:#0B1626;border:1px solid #FFC454;'
                    f'border-left:3px solid #FFD700;border-radius:5px;'
                    f'padding:0.5rem 0.7rem;margin-top:0.4rem;'
                    f'font-family:JetBrains Mono,monospace;font-size:0.67rem;line-height:1.85;">'
                    f'<span style="color:#FFD700;font-size:0.8rem">{_star["name"]}</span><br>'
                    f'<span style="color:#7290B0">RA  </span>'
                    f'<span style="color:#E4EEFA">{_star["ra_h"]:.4f} h</span><br>'
                    f'<span style="color:#7290B0">Dec </span>'
                    f'<span style="color:#E4EEFA">{_star["dec_deg"]:+.3f}°</span><br>'
                    f'<span style="color:#7290B0">mag </span>'
                    f'<span style="color:#2BFFB0">{_star["mag"]:.2f}</span>'
                    f'<span style="color:#7290B0">  spec </span>'
                    f'<span style="color:#E4EEFA">{_star["spect"]}</span><br>'
                    f'<span style="color:#7290B0">dist </span>'
                    f'<span style="color:#E4EEFA">{_dist_str}</span><br>'
                    + (f'<span style="color:#7290B0">con  </span>'
                       f'<span style="color:#E4EEFA">{_star["con"]}</span><br>'
                       if _star.get("con") else "")
                    + (f'<span style="color:#7290B0">{_ids}</span>'
                       if _ids else "")
                    + '</div>',
                    unsafe_allow_html=True,
                )

                # Store for highlight overlay
                ss["_sm_search_stars"] = [_star]

                if st.button("Clear highlight", key="sm_clear_hl"):
                    ss["_sm_search_stars"] = []
                    st.rerun()
            else:
                ss["_sm_search_stars"] = []
                st.caption("No matches found.")
        elif not sm_query:
            ss["_sm_search_stars"] = []
        elif not _cat_ok_search:
            st.caption("Catalog not found — set path in Settings.")

    with sm1:
        _current_epoch_jd = _get_starfield_epoch_jd()
        _built_epoch_jd   = ss.get("_sm_epoch_jd_built")
        _built_mag        = ss.get("_sm_mag_built")

        # Auto-rebuild when epoch or mag limit has changed since last render
        _needs_rebuild = (
            ss.get("_sm_fig") is not None and (
                _built_epoch_jd != _current_epoch_jd or
                _built_mag != sm_mag
            )
        )

        if gen_btn or _needs_rebuild or ss.get("_sm_fig") is not None:
            if gen_btn or _needs_rebuild:
                _reason = "date changed" if _needs_rebuild else "button clicked"
                with st.spinner(f"Computing precession ({_reason})…"):
                    try:
                        fig_sm, shift_rows = build_star_accuracy_fig(
                            epoch_jd=_current_epoch_jd,
                            catalog_path=ss.star_catalog,
                            mag_limit=sm_mag,
                        )
                        ss["_sm_fig"]           = fig_sm
                        ss["_sm_shifts"]        = shift_rows
                        ss["_sm_mag_built"]     = sm_mag
                        ss["_sm_epoch_jd_built"] = _current_epoch_jd
                    except Exception as _sme:
                        st.error(f"Star map error: {_sme}")
                        ss["_sm_fig"] = None

            if ss.get("_sm_fig") is not None:
                import copy
                _fig_display = copy.deepcopy(ss["_sm_fig"])
                _hl_stars = ss.get("_sm_search_stars", [])
                if _hl_stars:
                    _add_search_highlights(_fig_display, _hl_stars,
                                           _get_starfield_epoch_jd())
                st.plotly_chart(_fig_display, width="stretch")

                # Per-star shift table
                if ss.get("_sm_shifts") and _get_starfield_epoch_jd() is not None:
                    st.markdown("---")
                    rows_html = ""
                    for name, ra_h, dec_deg, mag_v, shift_arcsec, color in ss["_sm_shifts"]:
                        bar_w = min(int(shift_arcsec / 20), 120)
                        rows_html += (
                            f'<tr>'
                            f'<td style="color:{color};padding:0.15rem 0.5rem">{name}</td>'
                            f'<td style="color:#7290B0;padding:0.15rem 0.5rem">'
                            f'RA {ra_h:.4f}h  Dec {dec_deg:+.1f}°</td>'
                            f'<td style="color:#FFC454;padding:0.15rem 0.5rem;text-align:right">'
                            f'{shift_arcsec:.1f}"</td>'
                            f'<td style="padding:0.15rem 0.5rem">'
                            f'<div style="background:{color};height:6px;'
                            f'width:{bar_w}px;border-radius:3px;opacity:0.7"></div></td>'
                            f'</tr>'
                        )
                    st.markdown(
                        f'<div class="readout"><div class="r-head">▸ PRECESSION SHIFT PER STAR</div>'
                        f'<table style="width:100%;border-collapse:collapse;font-size:0.7rem">'
                        f'<tr style="color:#7290B0;font-size:0.62rem">'
                        f'<th align="left">STAR</th><th align="left">J2000 COORDS</th>'
                        f'<th align="right">SHIFT</th><th>BAR</th></tr>'
                        f'{rows_html}</table></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                '<div class="readout" style="min-height:420px;display:flex;'
                'align-items:center;justify-content:center;flex-direction:column;gap:1rem">'
                '<div class="r-head">▸ STAR ACCURACY MAP</div>'
                '<div class="r-dim" style="text-align:center;line-height:2">'
                'Verifies that HYG catalog stars land at correct GCRF positions.<br>'
                'Open circles = J2000 reference positions (SIMBAD verified).<br>'
                'Filled circles = IAU 1976 precessed positions at your epoch.<br>'
                'Lines show the precession shift between the two.<br><br>'
                'Set an epoch in the <b>Settings</b> tab, then click<br>'
                '<b>▶ Generate star map</b> to render.</div>'
                '</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════
# TAB 7 — SOLAR VIEW
# ══════════════════════════════════════════════════════


with tab_solar:
    sv1, sv2 = st.columns([3, 1])
    import datetime as _dt2

    with sv2:
        st.markdown("**Date**")
        _sv_year  = st.number_input("Year",  min_value=1800, max_value=2200,
                                    value=int(ss.sol_year),  step=1, key="sv_year")
        _sv_month = st.slider("Month", 1, 12, int(ss.sol_month), key="sv_month")
        ss.sol_year  = int(_sv_year)
        ss.sol_month = int(_sv_month)
        try:
            _sv_date = _dt2.date(int(_sv_year), int(_sv_month), 1)
            _sv_jd   = 2_451_545.0 + (_sv_date - _dt2.date(2000, 1, 1)).days
        except Exception:
            _sv_jd = 2_451_545.0

        _sc1, _sc2, _sc3 = st.columns(3)
        with _sc1:
            if st.button("◀ −1 mo", key="sv_back"):
                ss.sol_month -= 1
                if ss.sol_month < 1: ss.sol_month = 12; ss.sol_year -= 1
                st.rerun()
        with _sc2:
            if st.button("▶ +1 mo", key="sv_fwd"):
                ss.sol_month += 1
                if ss.sol_month > 12: ss.sol_month = 1; ss.sol_year += 1
                st.rerun()
        with _sc3:
            if st.button("▶▶ +1 yr", key="sv_fwd_yr"):
                ss.sol_year += 1; st.rerun()

        import math as _msv
        _ex, _ey, _ez = _planet_pos_au(_PLANETS["Earth"], _sv_jd)
        _earth_r   = (_ex**2 + _ey**2 + _ez**2)**0.5
        _earth_lon = _msv.degrees(_msv.atan2(_ey, _ex)) % 360
        T_yr_sv    = (_sv_jd - 2_451_545.0) / 365.25
        st.markdown(
            f'<div class="readout">'
            f'<span class="r-label">JD     </span><span class="r-val">{_sv_jd:.1f}</span><br>'
            f'<span class="r-label">EPOCH  </span><span class="r-val">{2000+T_yr_sv:.3f}</span><br>'
            f'<span class="r-label">EARTH r</span><span class="r-val">{_earth_r:.4f}</span>'
            f'<span class="r-unit"> AU</span><br>'
            f'<span class="r-label">EARTH λ</span><span class="r-val">{_earth_lon:.1f}</span>'
            f'<span class="r-unit"> °ecl</span>'
            f'</div>', unsafe_allow_html=True,
        )

        st.markdown("**Planets**")
        ss.sol_show_mercury = st.toggle("Mercury", value=ss.sol_show_mercury, key="sv_merc")
        ss.sol_show_venus   = st.toggle("Venus",   value=ss.sol_show_venus,   key="sv_ven")
        ss.sol_show_earth   = st.toggle("Earth",   value=ss.sol_show_earth,   key="sv_ear")
        ss.sol_show_mars    = st.toggle("Mars",    value=ss.sol_show_mars,    key="sv_mars")
        ss.sol_show_jupiter = st.toggle("Jupiter", value=ss.sol_show_jupiter, key="sv_jup")
        ss.sol_show_saturn  = st.toggle("Saturn",  value=ss.sol_show_saturn,  key="sv_sat")
        ss.sol_show_uranus  = st.toggle("Uranus",  value=ss.sol_show_uranus,  key="sv_ura")
        ss.sol_show_neptune = st.toggle("Neptune", value=ss.sol_show_neptune, key="sv_nep")

        st.markdown("**Display**")
        ss.sol_show_moon     = st.toggle("Moon",          value=ss.sol_show_moon,     key="sv_moon")
        ss.sol_show_trails   = st.toggle("Orbit trails",  value=ss.sol_show_trails,   key="sv_trail")
        ss.sol_show_stars    = st.toggle("Starfield",     value=ss.sol_show_stars,    key="sv_stars")
        ss.sol_show_ecliptic = st.toggle("Ecliptic grid", value=ss.sol_show_ecliptic, key="sv_ecl")
        ss.sol_show_labels   = st.toggle("Labels",        value=ss.sol_show_labels,   key="sv_lbl")

        st.markdown("**Planet scale**")
        ss["sol_planet_scale"] = st.slider(
            "Size multiplier", 0.3, 3.0,
            float(ss.get("sol_planet_scale", 1.0)), 0.1, key="sv_scale",
            help="1.0 = default (proportional, ~500× real sizes). "
                 "Increase to make planets easier to see at outer system scale.",
        )
        if _SOLAR_BODIES_OK:
            st.caption("✔ 3D spheres active")
        else:
            st.caption(f"⚠ Fallback: dot markers\n({_SOLAR_BODIES_ERR})")

    with sv1:
        if not _SOLAR_BODIES_OK:
            st.warning("solar_bodies.py not found — planets shown as dots. "
                       "Drop solar_bodies.py into ssapy_toolkit/plots/")
        _sv_controls = dict(
            Mercury=ss.sol_show_mercury, Venus=ss.sol_show_venus,
            Earth=ss.sol_show_earth,     Mars=ss.sol_show_mars,
            Jupiter=ss.sol_show_jupiter, Saturn=ss.sol_show_saturn,
            Uranus=ss.sol_show_uranus,   Neptune=ss.sol_show_neptune,
            moon=ss.sol_show_moon,       trails=ss.sol_show_trails,
            stars=ss.sol_show_stars,     ecliptic=ss.sol_show_ecliptic,
            labels=ss.sol_show_labels,   planet_scale=float(ss.get("sol_planet_scale",1.0)),
        )
        with st.spinner("Building solar view…"):
            try:
                _sv_fig = _build_solar_view_fig(_sv_jd, _sv_controls, ss.star_catalog)
                st.plotly_chart(_sv_fig, width="stretch")
            except Exception as _sve:
                import traceback as _tb
                st.error(f"Solar view error: {_sve}")
                st.code(_tb.format_exc())


# ══════════════════════════════════════════════════════
# TAB — SAVE / EXPORT
# ══════════════════════════════════════════════════════
with tab_save:
    st.markdown(
        '<p style="font-size:0.71rem;color:#7290B0;font-family:JetBrains Mono,monospace;'
        'margin-bottom:0.8rem">Save 3D-view data and Mission-Planner data separately — '
        'each has its own name, output folder (defaults into the demo gallery), and '
        'format checkboxes.</p>', unsafe_allow_html=True,
    )

    def _write_payload(payload: dict, out_dir: Path, name: str,
                        fmt_json: bool, fmt_csv: bool, fmt_hdf5: bool,
                        csv_rows_fn=None) -> list:
        """Shared writer for both save sections. csv_rows_fn(payload) -> (header, rows) or None."""
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []

        if fmt_json:
            p = out_dir / f"{name}.json"
            p.write_text(json.dumps(payload, indent=2, default=str))
            written.append(str(p))

        if fmt_csv:
            import csv as _csv
            p = out_dir / f"{name}.csv"
            with open(p, "w", newline="") as f:
                w = _csv.writer(f)
                rows_out = csv_rows_fn(payload) if csv_rows_fn else None
                if rows_out:
                    header, rows = rows_out
                    w.writerow(header)
                    for row in rows:
                        w.writerow(row)
                else:
                    w.writerow(["field", "value"])
                    for k, val in payload.items():
                        if isinstance(val, (dict, list)):
                            val = json.dumps(val, default=str)
                        w.writerow([k, val])
            written.append(str(p))

        if fmt_hdf5:
            try:
                import h5py
                p = out_dir / f"{name}.h5"
                with h5py.File(p, "w") as hf:
                    for k, val in payload.items():
                        if isinstance(val, (int, float, str)):
                            hf.attrs[k] = val
                        elif isinstance(val, np.ndarray):
                            hf.create_dataset(k, data=val)
                        elif isinstance(val, list) and val and isinstance(val[0], (list, tuple)):
                            hf.create_dataset(k, data=np.asarray(val))
                        else:
                            hf.attrs[f"{k}_json"] = json.dumps(val, default=str)
                written.append(str(p))
            except ImportError:
                st.error("h5py is not installed — run `pip install h5py` to enable HDF5 export. "
                         "JSON/CSV were still written if selected.")
        return written

    # ══════════════════════════════════════════════════════
    # SECTION 1 — 3D VIEW (orbit, extra satellites, belt residence)
    # ══════════════════════════════════════════════════════
    with st.container(border=True):
        st.markdown('<h4 style="color:var(--green);font-family:JetBrains Mono,monospace;text-transform:uppercase;letter-spacing:0.1em;font-size:0.8rem;margin:0 0 0.6rem">🛰 3D View data</h4>', unsafe_allow_html=True)
        v_c1, v_c2 = st.columns([2, 1])
        with v_c1:
            ss.save_name_view = st.text_input("Save name", value=ss.get("save_name_view", "ssapy_3dview"),
                                               key="save_name_view_in")
            ss.save_dir_view = st.text_input("Output directory", value=ss.get("save_dir_view", ss.save_dir),
                                              key="save_dir_view_in",
                                              help="Defaults to yu_figures/demo_gallery/figures.")
            if st.button("↺ Reset to default demo-gallery path", key="save_dir_view_reset"):
                ss.save_dir_view = str(Path.home() / "yu_figures" / "demo_gallery" / "figures")
                st.rerun()
        with v_c2:
            st.markdown("**Formats**")
            ss.save_fmt_json_view = st.checkbox("JSON", value=ss.get("save_fmt_json_view", True),  key="save_fmt_json_view_cb")
            ss.save_fmt_csv_view  = st.checkbox("CSV",  value=ss.get("save_fmt_csv_view", False),   key="save_fmt_csv_view_cb")
            ss.save_fmt_hdf5_view = st.checkbox("HDF5", value=ss.get("save_fmt_hdf5_view", False),  key="save_fmt_hdf5_view_cb")
        st.caption("Includes: orbital elements + propagation config, propagated r/v time series "
                   "(if a Full propagation has been run), extra satellites, Van Allen belt residence.")

        if st.button("💾 Save 3D View", key="save_view_btn"):
            if not (ss.save_fmt_json_view or ss.save_fmt_csv_view or ss.save_fmt_hdf5_view):
                st.warning("Select at least one format.")
            else:
                try:
                    payload = dict(
                        saved_at=datetime.now(timezone.utc).isoformat(),
                        orbit=dict(a_km=ss.a_km, e=ss.e, inc_deg=ss.inc_deg, raan_deg=ss.raan_deg,
                                   argp_deg=ss.argp_deg, nu_deg=ss.nu_deg, epoch=ss.epoch,
                                   frame=ss.frame, n_orbits=ss.n_orbits, dt_s=ss.dt_s,
                                   propagator=ss.propagator, gravity=ss.gravity,
                                   third_body=ss.third_body, non_grav=ss.non_grav),
                        extra_satellites=[{k: v for k, v in s.items() if k != "tle_text"}
                                           for s in ss.extra_satellites],
                        belt_residence=ss.get("_belt_residence", _estimate_belt_residence(ss.a_km, ss.e, ss.n_orbits)),
                    )
                    if ss.get("orbit_r_km") is not None:
                        payload["orbit_r_km"] = np.asarray(ss["orbit_r_km"]).tolist()
                    if ss.get("orbit_v_kms") is not None:
                        payload["orbit_v_kms"] = np.asarray(ss["orbit_v_kms"]).tolist()

                    def _view_csv_rows(p):
                        if "orbit_r_km" not in p:
                            return None
                        header = ["t_index", "x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms"]
                        r, v = p["orbit_r_km"], p.get("orbit_v_kms")
                        rows = [[i, *r[i], *(v[i] if v and i < len(v) else [None, None, None])]
                                for i in range(len(r))]
                        return header, rows

                    written = _write_payload(payload, Path(ss.save_dir_view), ss.save_name_view,
                                              ss.save_fmt_json_view, ss.save_fmt_csv_view, ss.save_fmt_hdf5_view,
                                              csv_rows_fn=_view_csv_rows)
                    if written:
                        ss.save_log = [f"✔ 3D View → {ss.save_dir_view}"] + written + ss.save_log
                        st.success(f"Saved {len(written)} file(s)")
                        for w in written: st.code(w)
                except Exception as ex:
                    st.error(f"Save failed: {ex}")

    # ══════════════════════════════════════════════════════
    # SECTION 2 — MISSION PLANNER (targets, transfer legs, burns)
    # ══════════════════════════════════════════════════════
    with st.container(border=True):
        st.markdown('<h4 style="color:var(--green);font-family:JetBrains Mono,monospace;text-transform:uppercase;letter-spacing:0.1em;font-size:0.8rem;margin:0 0 0.6rem">🚀 Mission Planner data</h4>', unsafe_allow_html=True)
        m_c1, m_c2 = st.columns([2, 1])
        with m_c1:
            ss.save_name_mission = st.text_input("Save name", value=ss.get("save_name_mission", "ssapy_mission_plan"),
                                                  key="save_name_mission_in")
            ss.save_dir_mission = st.text_input("Output directory", value=ss.get("save_dir_mission", ss.save_dir),
                                                 key="save_dir_mission_in",
                                                 help="Defaults to yu_figures/demo_gallery/figures.")
            if st.button("↺ Reset to default demo-gallery path", key="save_dir_mission_reset"):
                ss.save_dir_mission = str(Path.home() / "yu_figures" / "demo_gallery" / "figures")
                st.rerun()
        with m_c2:
            st.markdown("**Formats**")
            ss.save_fmt_json_mission = st.checkbox("JSON", value=ss.get("save_fmt_json_mission", True),  key="save_fmt_json_mission_cb")
            ss.save_fmt_csv_mission  = st.checkbox("CSV",  value=ss.get("save_fmt_csv_mission", False),   key="save_fmt_csv_mission_cb")
            ss.save_fmt_hdf5_mission = st.checkbox("HDF5", value=ss.get("save_fmt_hdf5_mission", False),  key="save_fmt_hdf5_mission_cb")
        st.caption("Includes: target orbits, transfer-strategy comparison, burns/manoeuvres.")

        if st.button("💾 Save Mission Plan", key="save_mission_btn"):
            if not (ss.save_fmt_json_mission or ss.save_fmt_csv_mission or ss.save_fmt_hdf5_mission):
                st.warning("Select at least one format.")
            else:
                try:
                    payload = dict(
                        saved_at=datetime.now(timezone.utc).isoformat(),
                        targets=list(ss.mp_targets),
                        objective=ss.mp_objective, optimizer=ss.mp_optimizer,
                        geometry=ss.mp_geometry, dv_budget_ms=ss.mp_dv_budget,
                        tof_min_hr=ss.mp_tof_min, tof_max_hr=ss.mp_tof_max,
                        transfer_legs=[{k: v for k, v in leg.items() if k not in ("origin", "target")}
                                       for leg in (ss.mp_results or [])],
                        burns=list(ss.burns),
                    )

                    def _mission_csv_rows(p):
                        if not p["burns"]:
                            return None
                        header = ["label", "t_s", "dv_t_kms", "dv_n_kms", "dv_w_kms", "mode"]
                        rows = [[b.get("label",""), b.get("t_s",0), b.get("dv_t",0),
                                 b.get("dv_n",0), b.get("dv_w",0), b.get("mode","")] for b in p["burns"]]
                        return header, rows

                    written = _write_payload(payload, Path(ss.save_dir_mission), ss.save_name_mission,
                                              ss.save_fmt_json_mission, ss.save_fmt_csv_mission, ss.save_fmt_hdf5_mission,
                                              csv_rows_fn=_mission_csv_rows)
                    if written:
                        ss.save_log = [f"✔ Mission Plan → {ss.save_dir_mission}"] + written + ss.save_log
                        st.success(f"Saved {len(written)} file(s)")
                        for w in written: st.code(w)
                except Exception as ex:
                    st.error(f"Save failed: {ex}")

    if ss.save_log:
        st.markdown("---")
        st.markdown("**Recent saves**")
        st.markdown(
            '<div class="console">' +
            "\n".join(f'<span class="c-ok">{l}</span>' for l in ss.save_log[:20]) +
            '</div>', unsafe_allow_html=True,
        )
        if st.button("Clear save log", key="clear_save_log"):
            ss.save_log = []; st.rerun()