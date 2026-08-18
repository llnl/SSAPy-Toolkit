"""
test_magfield_physics.py — physics regression suite for the SSAPy-Toolkit
magnetosphere plots.

Every check here corresponds to a property that was verified by hand during
development.  They exist because several real bugs in these modules were the
kind that produce a plausible-looking figure:

  * geocentric coordinates passed to a geodetic IGRF entry point
  * a star catalogue plotted in equatorial axes inside an Earth-fixed scene
  * the GEO->GSM rotation applied transposed

The last one survived a round of testing because the test derived "nightside"
from the same transposed matrix — self-consistently wrong.  The frame tests
below therefore check against *independent* references (geopack's own inverse
call, and a solar-position calculation that does not use the matrix at all).

Run:
    pytest -q -m "not slow"          47 checks, ~50 s
    pytest -q -m slow                 6 cold-cache / subprocess checks, ~4 min
    python test_magfield_physics.py   same, without pytest installed (--slow to include)

The `slow` marker is registered in tests/conftest.py.
"""

from __future__ import annotations

import math
import os
import warnings
from datetime import datetime

import numpy as np
import pytest

warnings.filterwarnings("ignore")

# Package imports. These were bare flat imports (`import magfield_plot_3d`),
# which only resolved when the interpreter happened to be started from inside
# ssapy_toolkit/plots/ -- so `pytest tests/test_magfield_physics.py` from the
# repo root failed at collection with ModuleNotFoundError. The fallback keeps
# the "run this file directly" mode advertised in the docstring working.
try:
    from ssapy_toolkit.plots import magfield_plot_3d as mf
    from ssapy_toolkit.plots import magnetosphere_core as core
    from ssapy_toolkit.plots import starfield as sf
except ImportError:
    import magfield_plot_3d as mf
    try:
        from ssapy_toolkit.plots import magnetosphere_core as core
    except ImportError:
        import magnetosphere_core as core
    import starfield as sf

DATE = datetime(2025, 7, 2)
RE = core.EARTH_RADIUS_KM

_HAS_GEOPACK = getattr(mf, "_HAS_GEOPACK", False)
_HAS_PPIGRF = getattr(mf, "_HAS_PPIGRF", False)
try:
    import spacepy.irbempy as _ib
    _HAS_SPACEPY = True
except Exception:
    _HAS_SPACEPY = False

needs_geopack = pytest.mark.skipif(not _HAS_GEOPACK, reason="geopack not installed")
needs_ppigrf = pytest.mark.skipif(not _HAS_PPIGRF, reason="ppigrf not installed")
needs_spacepy = pytest.mark.skipif(not _HAS_SPACEPY, reason="spacepy/IRBEM not installed")


# ---------------------------------------------------------------------------
# Reference frames  (the class of bug that hurt most)
# ---------------------------------------------------------------------------

@needs_geopack
def test_gsm_matrix_is_a_rotation():
    M = mf._gsm_frame(DATE)
    assert np.allclose(M @ M.T, np.eye(3), atol=1e-9), "GEO->GSM must be orthonormal"
    assert abs(np.linalg.det(M) - 1.0) < 1e-9, "must be a proper rotation, not a reflection"


@needs_geopack
def test_gsm_matrix_orientation_against_geopack():
    """M must satisfy v_gsm = M @ v_geo, checked against geopack's own call."""
    from geopack import geopack as gp
    ut = (DATE - datetime(1970, 1, 1)).total_seconds()
    gp.recalc(ut)
    M = mf._gsm_frame(DATE)
    v = np.array([0.3, -0.5, 0.81]); v /= np.linalg.norm(v)
    truth = np.array(gp.geogsm(*v, 1))
    assert np.allclose(M @ v, truth, atol=1e-9), "M @ v_geo must equal geogsm(v_geo)"
    # and the transpose must NOT satisfy it — guards against re-introducing the bug
    assert not np.allclose(M.T @ v, truth, atol=1e-6)


@needs_geopack
def test_gsm_round_trip():
    M = mf._gsm_frame(DATE)
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(50, 3)) * 1e4
    back = (rows @ M.T) @ M          # GEO -> GSM -> GEO for row vectors
    assert np.allclose(rows, back, atol=1e-6)


def test_sun_direction_matches_independent_solar_position():
    """
    The sunward axis in GEO must agree with the subsolar point computed from
    an ephemeris that never touches the rotation matrix.  This is the check
    that exposed the transposed frame (it was 1.74 deg out).
    """
    sun = mf._sun_direction_geo(DATE)
    dec, lon = core._subsolar_point(DATE)
    ref = np.array([math.cos(math.radians(dec)) * math.cos(math.radians(lon)),
                    math.cos(math.radians(dec)) * math.sin(math.radians(lon)),
                    math.sin(math.radians(dec))])
    sep = math.degrees(math.acos(float(np.clip(sun @ ref, -1, 1))))
    assert sep < 0.05, f"sunward axis is {sep:.3f} deg from the subsolar point"


@needs_geopack
def test_t89_wrapper_evaluates_at_the_right_place():
    """
    Decisive frame test: take a point defined in GSM, express it in GEO, and
    push it through the interpolating wrapper.  The wrapper converts back to
    GSM internally, so the answer must match t89 called directly on the
    original GSM point.  A transposed rotation cannot pass this.
    """
    from geopack import t89
    grid = mf._get_t89(DATE, kp=3)
    M = grid.M
    for gsm in ([6.0, 1.0, 2.0], [-9.0, -2.0, 1.5], [4.0, -3.0, -2.0]):
        gsm = np.array(gsm, dtype=float)
        geo_km = (gsm @ M) * 6371.2                      # GSM -> GEO (row vector)
        got = grid(geo_km[None, :])[0]                   # wrapper: GEO -> GSM -> field -> GEO
        want_gsm = np.array(t89.t89(grid.iopt, grid.ps, *gsm))
        want_geo = want_gsm @ M                          # GSM -> GEO
        err = np.linalg.norm(got - want_geo)
        assert err < 0.05 * max(np.linalg.norm(want_geo), 1.0), (
            f"T89 wrapper off by {err:.3f} nT at GSM {gsm}")


@needs_geopack
def test_magnetopause_nose_points_at_the_sun():
    sun = mf._sun_direction_geo(DATE)
    pts, _, _, _, r0, _ = mf._shue_magnetopause(DATE, n_theta=10, n_phi=12)
    nose = pts[np.argmax(pts @ sun)]
    ang = math.degrees(math.acos(float(np.clip(
        (nose / np.linalg.norm(nose)) @ sun, -1, 1))))
    assert ang < 1.0, f"magnetopause nose is {ang:.2f} deg off the Sun line"
    assert abs(np.linalg.norm(nose) / RE - r0) < 0.05


def test_shue_standoff_and_its_response():
    """Nominal standoff ~10-11 RE; southward Bz and higher pressure compress it."""
    r_nom = mf._shue_magnetopause(DATE, bz_nT=0.0, dp_nPa=2.0, n_theta=4, n_phi=4)[4]
    r_south = mf._shue_magnetopause(DATE, bz_nT=-5.0, dp_nPa=2.0, n_theta=4, n_phi=4)[4]
    r_press = mf._shue_magnetopause(DATE, bz_nT=0.0, dp_nPa=8.0, n_theta=4, n_phi=4)[4]
    assert 10.0 < r_nom < 11.0, f"nominal standoff {r_nom:.2f} RE outside 10-11"
    assert r_south < r_nom, "southward Bz must erode the dayside"
    assert r_press < r_nom, "higher dynamic pressure must compress the boundary"


# ---------------------------------------------------------------------------
# Internal field
# ---------------------------------------------------------------------------

@needs_ppigrf
def test_bfield_matches_geocentric_reference():
    """_bfield_batch must equal ppigrf's geocentric synthesis exactly."""
    import ppigrf.ppigrf as pp
    rng = np.random.default_rng(0)
    P = rng.normal(size=(200, 3))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    P *= rng.uniform(RE + 300, 6 * RE, 200)[:, None]
    saved = mf.set_external_model(None)
    try:
        got = mf._bfield_batch(P, DATE)
    finally:
        mf.set_external_model(saved)
    r = np.linalg.norm(P, axis=1)
    gclat = np.degrees(np.arcsin(P[:, 2] / r))
    lon = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
    Br, Bt, Bp = [np.asarray(x).flatten() for x in pp.igrf_gc(r, 90 - gclat, lon, DATE)]
    th, ph = np.radians(90 - gclat), np.radians(lon)
    want = np.stack([Br*np.sin(th)*np.cos(ph) + Bt*np.cos(th)*np.cos(ph) - Bp*np.sin(ph),
                     Br*np.sin(th)*np.sin(ph) + Bt*np.cos(th)*np.sin(ph) + Bp*np.cos(ph),
                     Br*np.cos(th) - Bt*np.sin(th)], axis=1)
    assert np.abs(got - want).max() < 1e-6


@needs_ppigrf
def test_south_atlantic_anomaly_location():
    """|B| minimum at 500 km must fall in the South Atlantic."""
    lon = np.linspace(-180, 180, 121)
    lat = np.linspace(-70, 70, 57)
    LO, LA = np.meshgrid(lon, lat)
    r = RE + 500.0
    P = np.stack([r*np.cos(np.radians(LA))*np.cos(np.radians(LO)),
                  r*np.cos(np.radians(LA))*np.sin(np.radians(LO)),
                  r*np.sin(np.radians(LA))], axis=-1).reshape(-1, 3)
    saved = mf.set_external_model(None)
    try:
        B = np.linalg.norm(mf._bfield_batch(P, DATE), axis=1).reshape(LA.shape)
    finally:
        mf.set_external_model(saved)
    k = np.unravel_index(np.argmin(B), B.shape)
    assert -75 < LO[k] < -30, f"SAA longitude {LO[k]:.1f} outside the South Atlantic"
    assert -40 < LA[k] < -5, f"SAA latitude {LA[k]:.1f} outside the South Atlantic"
    assert 15e3 < B[k] < 24e3, f"SAA |B| {B[k]:.0f} nT implausible at 500 km"


# ---------------------------------------------------------------------------
# Geometry and integration
# ---------------------------------------------------------------------------

def test_wgs84_flattening():
    mesh = core._build_earth_mesh("none", n_lon=90, n_lat=45)
    r = np.sqrt(np.asarray(mesh.x)**2 + np.asarray(mesh.y)**2 + np.asarray(mesh.z)**2)
    f = 1.0 - r.min() / r.max()
    assert abs(f - 1/298.257223563) < 2e-5, f"flattening {f:.6f}"
    assert abs(r.min() - core.WGS84_B_KM) < 1e-3



def test_presampled_earth_shading_is_self_lit():
    mesh = core._build_earth_mesh(
        "none",
        n_lon=24,
        n_lat=12,
        sun_shading=True,
        date=DATE,
    )
    assert mesh.lighting.ambient == 1.0
    assert mesh.lighting.diffuse == 0.0


def test_unshaded_earth_mesh_keeps_plotly_lighting():
    mesh = core._build_earth_mesh("none", n_lon=24, n_lat=12, sun_shading=False)
    assert mesh.lighting.diffuse > 0.0

def test_surface_radius_is_the_ellipsoid():
    eq = mf._surface_radius_km(np.array([[RE, 0.0, 0.0]]))[0]
    pole = mf._surface_radius_km(np.array([[0.0, 0.0, RE]]))[0]
    assert abs(eq - core.WGS84_A_KM) < 1e-6
    assert abs(pole - core.WGS84_B_KM) < 1e-6


def test_tracer_converges_under_step_halving():
    saved = mf.set_external_model(None)
    try:
        seed = mf._make_seeds_magnetic([45.0], n_lons=1)
        coarse = mf._trace_batch_rk4(seed, DATE, direction=+1, step_min=8, step_max=220,
                                     max_steps=9000)[0]
        fine = mf._trace_batch_rk4(seed, DATE, direction=+1, step_min=4, step_max=110,
                                   max_steps=20000)[0]
    finally:
        mf.set_external_model(saved)
    assert np.linalg.norm(coarse[-1] - fine[-1]) < 1.0, "RK4 endpoints must agree within 1 km"


@needs_ppigrf
def test_trace_terminates_on_the_ellipsoid():
    saved = mf.set_external_model(None)
    try:
        seed = mf._make_seeds_magnetic([50.0], n_lons=2)
        lines = mf._trace_batch_rk4(seed, DATE, direction=+1, step_min=8, step_max=220,
                                    max_steps=9000)
    finally:
        mf.set_external_model(saved)
    for L in lines:
        end = L[-1:]
        assert abs(np.linalg.norm(end) - mf._surface_radius_km(end)[0]) < 0.01


# ---------------------------------------------------------------------------
# Trapped particles
# ---------------------------------------------------------------------------

def test_flux_ratio_falls_monotonically_from_the_equator():
    b = np.array([1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0])
    f = core._omnidirectional_flux_ratio(b, n=2.0)
    assert abs(f[0] - 1.0) < 1e-9, "density must be 1 at the equator"
    assert np.all(np.diff(f) < 0), "density must fall as |B| rises"
    assert f[-1] < 1e-3


def test_flux_ratio_matches_closed_form():
    """
    The trapped-density profile is my own derivation, so check it against the
    cases that integrate analytically.  For j(a_eq) ~ sin^n(a_eq) the surviving
    fraction is int_0^amax sin^(n+1) / int_0^(pi/2) sin^(n+1), with
    sin(amax) = sqrt(Beq/B):

        n = 0 :  1 - cos(amax)              = 1 - sqrt(1 - Beq/B)
        n = 1 :  (amax - sin amax cos amax) / (pi/2)
    """
    b = np.array([1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0])
    amax = np.arcsin(np.sqrt(1.0 / b))
    exact0 = 1.0 - np.sqrt(1.0 - 1.0 / b)
    exact1 = (amax - np.sin(amax) * np.cos(amax)) / (np.pi / 2)
    assert np.abs(core._omnidirectional_flux_ratio(b, n=0.0) - exact0).max() < 1e-5
    assert np.abs(core._omnidirectional_flux_ratio(b, n=1.0) - exact1).max() < 1e-5


@pytest.mark.slow
@needs_ppigrf
def test_gui_entrypoint_config_path(tmp_path):
    """
    Exercise the __main__ block the Streamlit GUI actually drives.  It reads a
    Python config file via GUI_CONFIG and writes into OUTPUT_DIR; nothing in
    the suite had ever run it, so a syntax or key error there would only
    surface in the app.
    """
    import subprocess, sys, os
    cfg = tmp_path / "cfg.py"
    cfg.write_text(
        "epoch = 2025.5\nfidelity = 'draft'\nmax_r_re = 5.0\n"
        "show_van_allen = False\ntexture_path = 'none'\nshow_stars = False\n"
        "show_magnetopause = False\nexternal_field = None\nuse_omni = False\n"
        "width = 400\nheight = 320\nlons = [0, 120, 240]\n"
        "L_values = [2.0]\npolar_seeds = False\nnonsense_key = 1\n")
    env = dict(os.environ, GUI_CONFIG=str(cfg), OUTPUT_DIR=str(tmp_path))
    r = subprocess.run([sys.executable, mf.__file__], env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"GUI entrypoint failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    assert (tmp_path / "magfield_plot_3d.html").exists(), "no HTML written"
    # keys that are real parameters must be honoured, unknown ones reported
    assert "ignoring unknown config keys: ['nonsense_key']" in r.stdout


@needs_ppigrf
def test_mirror_latitude_tracks_dipole_theory():
    """Median IGRF mirror latitude must sit near the analytic dipole value."""
    from scipy.optimize import brentq
    saved = mf.set_external_model(None)
    try:
        axis = core._dipole_axis()
        for alpha, tol in ((25.0, 4.0), (40.0, 4.0)):
            theory = math.degrees(brentq(
                lambda lam: math.sqrt(1 + 3*math.sin(lam)**2)/math.cos(lam)**6
                            - 1/math.sin(math.radians(alpha))**2,
                0.01, math.radians(75)))
            b = mf._igrf_lshell_boundary(2.0, DATE, axis, n_azim=12, n_pts=40,
                                         pitch_angle_deg=alpha)
            ends = np.concatenate([b[:, 0, :], b[:, -1, :]])
            ml = np.abs(np.degrees(np.arcsin(np.clip(
                (ends @ axis) / np.linalg.norm(ends, axis=1), -1, 1))))
            assert abs(np.median(ml) - theory) < tol, (
                f"pitch {alpha}: median {np.median(ml):.1f} vs theory {theory:.1f}")
    finally:
        mf.set_external_model(saved)


@needs_spacepy
def test_aep8_reproduces_the_belt_structure():
    tab = mf._load_aep8_table(allow_build=False)
    if tab is None:
        pytest.skip("AE8/AP8 table not cached")
    L, Fp, Fe = tab['L'], tab['p'], tab['e']
    assert 1.4 <= L[np.argmax(Fp[:, 0])] <= 2.0, "AP-8 proton peak outside L 1.4-2.0"
    assert 3.8 <= L[np.argmax(Fe[:, 0])] <= 4.8, "AE-8 electron peak outside L 3.8-4.8"
    slot = (L > 1.9) & (L < 3.4)
    assert Fe[slot, 0].min() < 0.2 * Fe[:, 0].max(), "slot region not resolved"
    row = Fe[np.argmin(abs(L - 4.4))]
    assert np.all(np.diff(row) <= 0), "flux must fall with B/B0 along a field line"


@needs_spacepy
def test_mcilwain_L_matches_irbem():
    """
    Compared at 2024-07-02, not the plot epoch: IRBEM's bundled IGRF ends at
    2025 and silently clamps to the nearest year ("out of valid range ...
    Using nearest"), which would make this a comparison against a clamped
    reference rather than a real one.
    """
    import spacepy.time as spt, spacepy.coordinates as spc
    date = datetime(2024, 7, 2)
    saved = mf.set_external_model(None)
    try:
        axis = core._dipole_axis()
        _, e1, e2 = core._mag_basis(axis)
        seeds = [L*RE*(math.cos(p)*e1 + math.sin(p)*e2)
                 for L in (2.0, 3.0, 4.0) for p in (0.0, math.pi)]
        eq, B0 = mf._true_magnetic_equator(seeds, date)
    finally:
        mf.set_external_model(saved)
    mine = (mf._M_DIPOLE_NT_RE3 / B0) ** (1/3)
    t = spt.Ticktock([date]*len(eq), 'UTC')
    c = spc.Coords((eq/RE).tolist(), 'GEO', 'car', use_irbem=True); c.ticks = t
    Lm = np.abs(np.array(_ib.get_Lm(t, c, [90.0], extMag='0', intMag='IGRF')['Lm']).flatten())
    rel = np.abs(mine - Lm) / Lm
    assert rel.max() < 0.02, f"L differs from IRBEM by {100*rel.max():.2f}%"


# ---------------------------------------------------------------------------
# Astrometry
# ---------------------------------------------------------------------------

def test_julian_date_and_gmst():
    assert abs(core._julian_date(datetime(2000, 1, 1, 12)) - 2451545.0) < 1e-6
    g = math.degrees(core._gmst_rad(datetime(2000, 1, 1, 12))) / 15.0
    assert abs(g - 18.697375) < 1e-4, f"GMST at J2000 = {g:.6f} h"


def test_precession_rate():
    """Pole motion must match theta = 2004.31 arcsec/century."""
    d = datetime(2026, 7, 2)
    T = (core._julian_date(d) - 2451545.0) / 36525.0
    pole = np.array([0.0, 0.0, 1.0]) @ core._precession_matrix(d).T
    moved = math.degrees(math.acos(float(np.clip(pole @ [0, 0, 1], -1, 1)))) * 3600
    assert abs(moved - 2004.31 * T) < 1.0


def test_star_at_ra_equal_to_gmst_sits_over_greenwich():
    d = datetime(2026, 7, 2)
    g_h = math.degrees(core._gmst_rad(d)) / 15.0
    v = core._stars_to_ecef(np.array([g_h]), np.array([0.0]), np.array([0.0]),
                          np.array([0.0]), d, apply_pm=False, apply_prec=False)[0]
    assert abs(math.degrees(math.atan2(v[1], v[0]))) < 1e-4


def test_celestial_pole_maps_to_earth_rotation_axis():
    """Polaris must sit within ~1 deg of +Z in the Earth-fixed frame."""
    d = datetime(2026, 7, 2)
    v = core._stars_to_ecef(np.array([2.5303]), np.array([89.264]),
                          np.array([44.5]), np.array([-11.9]), d)[0]
    assert math.degrees(math.acos(float(np.clip(v[2], -1, 1)))) < 1.0


def test_star_directions_match_astropy():
    """
    Independent check of the whole J2000 -> Earth-fixed chain against astropy's
    ITRS transform, which shares no code with this implementation.  Residuals
    of order 10-20 arcsec are expected: nutation is deliberately omitted.
    """
    pytest.importorskip("astropy")
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, ITRS
    import astropy.units as u
    when = datetime(2026, 7, 8)
    t = Time("2026-07-08T00:00:00", scale="utc")
    assert abs(math.degrees(core._gmst_rad(when))/15
               - t.sidereal_time("mean", "greenwich").hour) * 3600 < 0.5
    for ra, dec, pmra, pmdec in ((6.7525, -16.716, -546.0, -1223.0),
                                 (18.6156, 38.784, 201.0, 287.0),
                                 (2.5303, 89.264, 44.5, -11.9),
                                 (14.2610, 19.182, -1093.4, -1999.4)):
        mine = core._stars_to_ecef(np.array([ra]), np.array([dec]),
                                   np.array([pmra]), np.array([pmdec]), when)[0]
        c = SkyCoord(ra=ra*15*u.deg, dec=dec*u.deg, distance=1e6*u.pc,
                     pm_ra_cosdec=pmra*u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr,
                     frame="icrs", obstime=Time("J2000")).apply_space_motion(new_obstime=t)
        ref = c.transform_to(ITRS(obstime=t)).cartesian.xyz.value
        ref = ref / np.linalg.norm(ref)
        sep = math.degrees(math.acos(float(np.clip(mine @ ref, -1, 1)))) * 3600
        assert sep < 60.0, f"star direction {sep:.0f} arcsec from astropy"


@pytest.mark.parametrize("frame", ["j2000", "gcrf", "ecef"])
def test_starfield_frames_are_consistent(frame):
    """All three frames must return unit vectors for the same star set."""
    out = sf.star_directions(when=datetime(2026, 7, 8), frame=frame)
    if out is None:
        pytest.skip("star catalogue not installed")
    v, mag, rgb = out
    assert len(v) == len(mag) == len(rgb)
    assert np.abs(np.linalg.norm(v, axis=1) - 1).max() < 1e-9


def test_gmst_rotation_only_changes_longitude():
    """
    GCRF -> ECEF is a rotation about the spin axis, so declination must be
    untouched and longitude must move.  Getting this backwards is how the sky
    ended up misaligned by 68 deg before.
    """
    g = sf.star_directions(when=datetime(2026, 7, 8), frame="gcrf")
    e = sf.star_directions(when=datetime(2026, 7, 8), frame="ecef")
    if g is None:
        pytest.skip("star catalogue not installed")
    assert np.abs(g[0][:, 2] - e[0][:, 2]).max() < 1e-12, "declination changed"
    shift = np.abs(np.arctan2(g[0][:, 1], g[0][:, 0])
                   - np.arctan2(e[0][:, 1], e[0][:, 0]))
    assert math.degrees(np.median(shift)) > 1.0, "longitude did not change"


def test_starfield_accepts_toolkit_epoch_formats():
    """datetime, decimal year and GPS seconds are all used in this toolkit."""
    assert sf._to_datetime(datetime(2026, 7, 8)).year == 2026
    assert sf._to_datetime(2026.52).year == 2026
    assert 2000 < sf._to_datetime(1.4e9).year < 2030
    assert sf._to_datetime(None) is None


def test_core_does_not_redefine_sky_helpers():
    """starfield.py owns the sky code; core must re-export, not copy."""
    import ast, pathlib
    tree = ast.parse(pathlib.Path(core.__file__).read_text())
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    owned = {"_julian_date", "_gmst_rad", "_precession_matrix", "_stars_to_ecef",
             "_bv_to_teff", "_teff_to_srgb", "_cie_xyz_bar"}
    assert not (defined & owned), f"core redefines starfield code: {defined & owned}"
    for name in owned:
        assert getattr(core, name) is getattr(sf, name), f"core rebinds {name}"


def test_star_vectors_are_unit_length():
    rng = np.random.default_rng(3)
    n = 200
    v = core._stars_to_ecef(rng.uniform(0, 24, n), rng.uniform(-89, 89, n),
                          rng.uniform(-500, 500, n), rng.uniform(-500, 500, n), DATE)
    assert np.abs(np.linalg.norm(v, axis=1) - 1).max() < 1e-9


def test_solar_colour_temperature():
    """B-V = 0.65 must give roughly the solar effective temperature."""
    T = float(core._bv_to_teff(0.65))
    assert 5600 < T < 5950, f"Sun-like B-V gives {T:.0f} K"
    rgb = core._teff_to_srgb([T])[0]
    assert rgb.max() > 0.9 and rgb.min() > 0.7, "solar colour should be near-white"
    hot, cool = core._teff_to_srgb([core._bv_to_teff(0.0)])[0], core._teff_to_srgb([core._bv_to_teff(1.8)])[0]
    assert hot[2] > hot[0], "hot star must be blue-weighted"
    assert cool[0] > cool[2], "cool star must be red-weighted"


# ---------------------------------------------------------------------------
# Cross-module parity and van_allen coverage
# ---------------------------------------------------------------------------

def _shared_sources():
    """AST-extracted source of every top-level name defined in both modules."""
    import ast
    try:
        from ssapy_toolkit.plots import van_allen_plot_3d as va
    except ImportError:
        import van_allen_plot_3d as va
    out = {}
    for mod in (mf, va):
        src = open(mod.__file__).read()
        tree = ast.parse(src)
        out[mod.__name__] = {n.name: ast.get_source_segment(src, n)
                             for n in tree.body
                             if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    # Key on each module's actual __name__ rather than a hardcoded bare name:
    # these are now imported as ssapy_toolkit.plots.*, so "magfield_plot_3d"
    # is no longer a key in `out`.
    return out[mf.__name__], out[va.__name__]


def _render_with_blocked_imports(module_name, blocked, **kwargs):
    """Import a plot module with certain packages unavailable and render."""
    import builtins, contextlib, importlib, io, os, sys, tempfile
    real = builtins.__import__

    def guard(name, *a, **k):
        if any(name == b or name.startswith(b + ".") for b in blocked):
            raise ImportError(f"blocked {name}")
        return real(name, *a, **k)

    # Modules are imported by their package path; the bare names only ever
    # resolved when the interpreter was started from inside plots/.
    qual = f"ssapy_toolkit.plots.{module_name}"
    prefixes = (module_name, qual)

    os.environ["SSAPY_TOOLKIT_CACHE"] = tempfile.mkdtemp()   # cold cache
    for k in [m for m in sys.modules if m.startswith(prefixes)]:
        del sys.modules[k]
    builtins.__import__ = guard
    try:
        try:
            mod = importlib.import_module(qual)
        except ImportError:
            mod = importlib.import_module(module_name)
        fn = (mod.plot_van_allen_3d if module_name.startswith("van")
              else mod.plot_magfield_3d)
        with contextlib.redirect_stdout(io.StringIO()):
            r = fn(**kwargs)
        return r[0] if isinstance(r, tuple) else r
    finally:
        builtins.__import__ = real
        os.environ.pop("SSAPY_TOOLKIT_CACHE", None)
        for k in [m for m in sys.modules if m.startswith(prefixes)]:
            del sys.modules[k]


@pytest.mark.slow
@pytest.mark.parametrize("blocked", [("ppigrf",), ("spacepy",), ("geopack",),
                                     ("ppigrf", "spacepy", "geopack")])
def test_van_allen_degrades_instead_of_crashing(blocked):
    """
    van_allen documents itself as dependency-light.  With a cold cache and
    ppigrf missing, belt_style='flux' used to reach the field tracer anyway and
    die with NameError: _fast_igrf_for — the guard was on a different code
    path from the one the flux belts take.  A warm cache hid it, because the
    traced samples were read from disk and the tracer was never called.
    """
    fig = _render_with_blocked_imports(
        "van_allen_plot_3d", blocked, epoch=2025.5, fidelity="draft",
        width=380, height=300, texture_path="none", show_stars=False,
        belt_style="flux")
    assert len(fig.data) >= 3


@pytest.mark.slow
@needs_ppigrf
def test_magfield_degrades_without_geopack():
    fig = _render_with_blocked_imports(
        "magfield_plot_3d", ("geopack",), epoch=2025.5, fidelity="draft",
        width=380, height=300, texture_path="none", show_stars=False,
        show_van_allen=False, external_field="t89", use_omni=False)
    assert len(fig.data) >= 3


def _find_pyproject():
    import pathlib
    here = pathlib.Path(core.__file__).resolve()
    for parent in here.parents:
        cand = parent / "pyproject.toml"
        if cand.exists():
            return cand
    return None


def _load_toml(path):
    try:
        import tomllib
    except ImportError:
        tomllib = pytest.importorskip("tomli")
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def test_pyproject_declares_runtime_dependencies():
    """
    Guards a TOML ordering trap.  A `dependencies = [...]` line placed *below*
    a `[project.optional-dependencies]` header is parsed as an extra named
    "dependencies" rather than the real requirement list, so `pip install -e .`
    silently installs nothing and the failure only shows up as a missing module
    at import time on someone else's machine.
    """
    path = _find_pyproject()
    if path is None:
        pytest.skip("no pyproject.toml in this layout")
    data = _load_toml(path)
    project = data.get("project")
    if project is None:
        pytest.skip("not a PEP 621 pyproject (Poetry or setup.py layout)")

    extras = project.get("optional-dependencies", {})
    assert "dependencies" not in extras, (
        "'dependencies' appears as an EXTRA — the `dependencies = [...]` line "
        "is below the [project.optional-dependencies] header and must move "
        "above it")

    declared = " ".join(project.get("dependencies", []))
    for pkg in ("numpy", "plotly", "pandas", "pillow"):
        assert pkg in declared.lower(), (
            f"{pkg} missing from [project] dependencies (found: {declared!r})")


def test_pyproject_optional_extras_are_named_as_documented():
    """The plot modules' optional pieces should be reachable as extras."""
    path = _find_pyproject()
    if path is None:
        pytest.skip("no pyproject.toml in this layout")
    project = _load_toml(path).get("project")
    if project is None:
        pytest.skip("not a PEP 621 pyproject")
    extras = project.get("optional-dependencies", {})
    if not extras:
        pytest.skip("no extras declared yet")
    joined = " ".join(" ".join(v) for v in extras.values()).lower()
    declared = " ".join(project.get("dependencies", [])).lower()
    for pkg in ("ppigrf", "geopack"):
        assert pkg in declared or pkg in joined, (
            f"{pkg} is declared nowhere; the field lines need it")


def test_data_assets_resolve_from_a_sibling_data_repo(tmp_path):
    """
    Assets live in ssapy-data, not in the toolkit repo.  They must resolve
    with no environment variable set, from a checkout sitting beside the
    toolkit — the layout most people will actually have.
    """
    import importlib, os, sys
    repo = tmp_path / "SSAPy-Toolkit" / "ssapy_toolkit" / "plots"
    data = tmp_path / "ssapy-data"
    repo.mkdir(parents=True); data.mkdir()
    for pkg in (repo.parent, repo):
        (pkg / "__init__.py").write_text("")
    import shutil, pathlib
    for name in ("starfield.py", "magnetosphere_core.py", "magfield_plot_3d.py"):
        shutil.copy(pathlib.Path(sf.__file__).parent / name, repo / name)
    (data / "bright_stars.csv").write_text("ra,dec,mag\n0,0,1\n")
    (data / "aep8_table.npz").write_bytes(b"x" * 32)

    saved_env = os.environ.pop("SSAPY_DATA", None)
    sys.path.insert(0, str(tmp_path / "SSAPy-Toolkit"))
    # Snapshot rather than discard. An earlier version deleted every
    # ssapy_toolkit entry from sys.modules and never put them back, so every
    # test after this one re-imported fresh module objects while the
    # module-level `mf`/`sf` bindings still pointed at the originals. That
    # made test_both_modules_use_the_same_core_objects fail with two distinct
    # _atmosphere_traces functions -- a pure ordering artifact that looked
    # exactly like real duplication.
    saved_mods = {k: v for k, v in sys.modules.items()
                  if k.startswith("ssapy_toolkit")}
    for k in saved_mods:
        del sys.modules[k]
    try:
        mod = importlib.import_module("ssapy_toolkit.plots.starfield")
        found = mod.find_data_file("bright_stars.csv")
        assert found is not None, f"sibling repo not searched: {mod.ssapy_data_dirs()}"
        assert found.parent == data
    finally:
        sys.path.remove(str(tmp_path / "SSAPy-Toolkit"))
        if saved_env is not None:
            os.environ["SSAPY_DATA"] = saved_env
        for k in [m for m in sys.modules if m.startswith("ssapy_toolkit")]:
            del sys.modules[k]
        sys.modules.update(saved_mods)


def test_data_assets_are_gitignored():
    """The toolkit repo must not carry the assets it reads from ssapy-data."""
    import pathlib
    gi = pathlib.Path(sf.__file__).parent / ".gitignore"
    if not gi.exists():
        pytest.skip(".gitignore not alongside the modules in this layout")
    patterns = gi.read_text()
    for name in ("earth_texture.jpg", "bright_stars.csv", "*.npz"):
        assert name in patterns, f"{name} is not gitignored"


@pytest.mark.parametrize("preset", ["draft", "poster", "public", "measure"])
def test_presets_are_valid_keyword_sets(preset):
    """Every preset must name only real parameters."""
    import inspect
    valid = set(inspect.signature(mf.plot_magfield_3d).parameters)
    bad = sorted(set(mf.PRESETS[preset]) - valid)
    assert not bad, f"preset '{preset}' sets unknown parameters: {bad}"


def test_latest_driven_epoch_is_real_and_not_the_future():
    """
    The OMNI final record lags real time, so the newest drivable epoch must be
    in the past — and must actually have drivers, not fall back to nominal.
    """
    if os.environ.get("SSATK_RUN_NETWORK_TESTS") != "1":
        pytest.skip("set SSATK_RUN_NETWORK_TESTS=1 to query live OMNI data")
    when = mf.latest_driven_epoch()
    assert when < datetime.now(), "drivable epoch cannot be in the future"
    sw = mf.get_solar_wind(when)
    assert sw and sw.get("dp_nPa") is not None, "returned epoch has no real drivers"


def test_cli_check_runs_and_reports():
    """`ssapy-magnetosphere --check` must work before anything else does."""
    import contextlib, io
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = mf.main(["--check"])
    text = out.getvalue()
    assert rc == 0
    for expected in ("data files", "optional packages", "solar-wind drivers",
                     "bright_stars.csv", "ppigrf", "geopack"):
        assert expected in text, f"--check output missing {expected!r}"


def test_cli_argument_surface_is_valid():
    """Every --preset choice must be a real preset."""
    import contextlib, io
    with contextlib.redirect_stderr(io.StringIO()):
        with pytest.raises(SystemExit):
            mf.main(["--preset", "not-a-preset"])
    assert set(mf.PRESETS) >= {"draft", "poster", "public", "measure"}


@needs_ppigrf
def test_quick_figure_builds():
    fig, _ = mf.quick_figure("draft", show_van_allen=False, texture_path="none",
                             width=360, height=280, show_stars=False)
    assert len(fig.data) >= 3


def test_no_undefined_names_or_dead_imports():
    """
    Static gate.  van_allen once carried a _true_magnetic_equator that called
    _trace_batch_rk4 and _bfield_batch — neither defined in that module — and a
    starfield default referencing an unimported `datetime`.  Both were
    unreachable in the normal code path, so every runtime test passed while the
    functions were guaranteed to raise NameError if anyone called them.  Only
    static analysis finds that class of bug.

    Uses ruff (the linter this repository already installs) and falls back to
    pyflakes; skips if neither is present. The check is intentionally limited
    to undefined names because magfield_plot_3d deliberately re-exports private
    helpers for backwards compatibility, which pyflakes reports as unused.
    """
    import pathlib
    import shutil
    import subprocess
    try:
        from ssapy_toolkit.plots import van_allen_plot_3d as va
    except ImportError:
        import van_allen_plot_3d as va

    paths = [str(pathlib.Path(m.__file__)) for m in (mf, va, core, sf)]

    if shutil.which("ruff"):
        r = subprocess.run(["ruff", "check", "--select", "F821",
                            "--no-cache", "--output-format", "concise", *paths],
                           capture_output=True, text=True)
        problems = [ln for ln in r.stdout.splitlines()
                    if "F821" in ln]
        assert not problems, "ruff found:\n  " + "\n  ".join(problems)
        return

    try:
        from pyflakes.api import check
        from pyflakes.reporter import Reporter
    except ImportError:
        pytest.skip("neither ruff nor pyflakes installed")
    import io
    problems = []
    for path in paths:
        out, err = io.StringIO(), io.StringIO()
        check(pathlib.Path(path).read_text(), path, Reporter(out, err))
        problems += [ln for ln in out.getvalue().splitlines()
                     if "undefined name" in ln]
    assert not problems, "static analysis found:\n  " + "\n  ".join(problems)


def test_van_allen_helpers_are_self_contained():
    """Every name a van_allen helper calls must exist in van_allen."""
    import ast
    try:
        from ssapy_toolkit.plots import van_allen_plot_3d as va
    except ImportError:
        import van_allen_plot_3d as va
    src = open(va.__file__).read()
    tree = ast.parse(src)
    defined = {n.name for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    defined |= {t.id for n in tree.body if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
    def local_names(fn):
        names = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        for sub in ast.walk(fn):
            if isinstance(sub, (ast.Import, ast.ImportFrom)):
                names |= {(al.asname or al.name).split(".")[0] for al in sub.names}
            elif isinstance(sub, ast.Assign):
                names |= {t.id for t in sub.targets if isinstance(t, ast.Name)}
            elif isinstance(sub, ast.FunctionDef):
                names.add(sub.name)
        return names

    missing = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        allowed = defined | local_names(node)
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                    and sub.func.id.startswith("_")
                    and sub.func.id not in allowed
                    and sub.func.id not in dir(va)):
                missing.append(f"{node.name} calls undefined {sub.func.id}")
    assert not missing, "; ".join(sorted(set(missing)))


def test_no_duplicated_definitions_between_modules():
    """
    The shared helpers now live in magnetosphere_core, so neither plot module
    should define the same name as the other.  Duplication drifted three
    separate times before it was extracted — a crude subsolar point, a
    cosmetic belt gradient, and a quadrature with a 2e-3 error all survived in
    van_allen after magfield was fixed.  Re-copying a helper into both modules
    fails here.
    """
    A, B = _shared_sources()
    dupes = sorted(set(A) & set(B))
    assert not dupes, ("defined in BOTH plot modules — move to "
                       f"magnetosphere_core instead: {dupes}")


def test_core_is_importable_without_field_model_packages():
    """magnetosphere_core must not pull in ppigrf, geopack or spacepy."""
    import ast, pathlib
    try:
        from ssapy_toolkit.plots import magnetosphere_core as core
    except ImportError:
        import magnetosphere_core as core
    tree = ast.parse(pathlib.Path(core.__file__).read_text())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    heavy = imported & {"ppigrf", "geopack", "spacepy", "scipy"}
    assert not heavy, f"core should stay dependency-light, found: {sorted(heavy)}"


def test_both_modules_use_the_same_core_objects():
    """Both plot modules must bind the identical function objects, not copies."""
    try:
        from ssapy_toolkit.plots import van_allen_plot_3d as va
    except ImportError:
        import van_allen_plot_3d as va
    try:
        from ssapy_toolkit.plots import magnetosphere_core as core
    except ImportError:
        import magnetosphere_core as core
    for mod in (mf, va):
        for name in dir(mod):
            if name in dir(core) and not name.startswith("__"):
                a, b = getattr(mod, name), getattr(core, name)
                if callable(b):
                    assert a is b, f"{mod.__name__} rebinds {name} instead of using core"


def test_physics_fingerprint_tracks_the_code():
    """The belt cache key must change when the physics routines change.

    Patches the module where the physics actually lives. _physics_fingerprint
    resolves the routine names from its own module globals, and the physics
    moved to ssapy_toolkit.geomagnetics -- magfield_plot_3d only re-exports
    them. Re-exported names are separate bindings, so patching the plot
    module's copy leaves the fingerprint's view untouched (the same trap that
    made attribute assignment to _EXTERNAL_MODEL a silent no-op).
    """
    phys = mf._physics_fingerprint.__module__
    import importlib
    src_mod = importlib.import_module(phys)

    before = src_mod._physics_fingerprint()
    orig = src_mod._adaptive_step_km
    try:
        src_mod._adaptive_step_km = lambda r, a, b: r * 0 + 1.0
        assert src_mod._physics_fingerprint() != before, (
            "cache fingerprint ignored a change to a physics routine")
    finally:
        src_mod._adaptive_step_km = orig
    assert src_mod._physics_fingerprint() == before


@pytest.mark.parametrize("belt_style", ["lshell", "torus"])
def test_van_allen_builds(belt_style):
    try:
        from ssapy_toolkit.plots import van_allen_plot_3d as va
    except ImportError:
        import van_allen_plot_3d as va
    fig = va.plot_van_allen_3d(epoch=2025.5, fidelity="draft", width=400, height=320,
                               belt_style=belt_style, external_field=None,
                               texture_path="none", show_stars=False)
    assert len(fig.data) >= 3


if __name__ == "__main__":
    import sys
    run_slow = "--slow" in sys.argv
    print("running without pytest; add --slow for the cold-cache checks "
          "(~4 min).  'pytest -q -m \"not slow\"' gives better output.\n")
    fails = skipped = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        marks = list(getattr(fn, "pytestmark", []))
        skip = False
        cases = [()]
        for mk in marks:
            if mk.name == "skipif" and mk.args and mk.args[0]:
                skip = True
            if mk.name == "slow" and not run_slow:
                skip = True
            if mk.name == "parametrize":
                vals = mk.args[1]
                cases = [(v,) for v in vals]
        if skip:
            skipped += 1
            print(f"SKIP  {name}")
            continue
        for args in cases:
            tag = f"{name}{list(args) if args else ''}"
            try:
                fn(*args)
                print(f"PASS  {tag}")
            except Exception as e:
                fails += 1
                print(f"FAIL  {tag}: {type(e).__name__}: {e}")
    print(f"\n{fails} failure(s), {skipped} skipped")
    sys.exit(1 if fails else 0)
