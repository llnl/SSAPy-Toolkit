"""Orbital transfer catalog: two-burn Lambert/NTW planning across regimes.

Demo + pytest benchmarks for ``transfer_ssapy`` / ``solve_lambert``,
exercising a catalog of transfer types: coplanar orbit raising, LEO to
GEO, co-orbital phasing (rendezvous), GEO station repositioning, HEO
apogee injection, and a combined raise + plane change, plus prograde vs
retrograde geometry and alternate SSAPy propagators.  Also benchmarks
``transfer_optimal`` (orbit-to-orbit search): Hohmann rediscovery,
insertion vs rendezvous, first-burn-only intercepts, min-time under a
delta-v budget, autonomous GEO drift-window discovery, and the engine
model (thrust/mass-sized burn durations, propellant estimates, and
engine-honest feasibility), with mission-designer porkchop/Pareto
curves in the figure gallery.

Run modes
---------
* ``pytest demo_transfer_ssapy.py`` -- runs the numerical benchmarks
  only; no figures are produced.
* ``python demo_transfer_ssapy.py`` -- runs the same benchmarks, then
  renders gallery figures (transfer-geometry gallery, 3-D plane-change
  view, delta-v budget by transfer type, Hohmann-optimum benchmark, and
  finite-burn refinement benchmark), saving each via
  ``ssapy_toolkit.plots.yufig`` under ``demo_gallery/figures/``.
"""

import numpy as np

from ssapy.orbit import Orbit
from ssapy.propagator import KeplerianPropagator, RK4Propagator
from ssapy.compute import rv
from ssapy.constants import EARTH_MU, EARTH_RADIUS, RGEO

from ssapy_toolkit.orbital_mechanics.transfer_ssapy_function import (
    transfer_ssapy, solve_lambert)
from ssapy_toolkit.orbital_mechanics.transfer_optimal_function import (
    transfer_optimal)
from ssapy_toolkit.plots.transfer_trajectory_plot import (
    transfer_trajectory_plot)
from ssapy_toolkit.plots.transfer_burn_profile_plot import (
    transfer_burn_profile_plot)
_SOURCE = "ssapy_toolkit.orbital_mechanics (submodules)"

MU = EARTH_MU
FIGDIR = "demo_gallery/figures"


# ======================================================================
# Shared geometry helpers
# ======================================================================

def _circular_state(radius, angle, inclination=0.0):
    """Position/velocity on a circular orbit at the given phase angle."""
    ca, sa = np.cos(angle), np.sin(angle)
    ci, si = np.cos(inclination), np.sin(inclination)
    r = radius * np.array([ca, sa * ci, sa * si])
    v = np.sqrt(MU / radius) * np.array([-sa, ca * ci, ca * si])
    return r, v


def _apsis_state(radius, angle, speed, inclination=0.0):
    """State at an apsis: position at ``angle``, tangential ``speed``."""
    ca, sa = np.cos(angle), np.sin(angle)
    ci, si = np.cos(inclination), np.sin(inclination)
    r = radius * np.array([ca, sa * ci, sa * si])
    v = speed * np.array([-sa, ca * ci, ca * si])
    return r, v


def _hohmann_dv(r1, r2):
    """Analytic two-impulse Hohmann delta-vs between circular radii."""
    dv1 = np.sqrt(MU / r1) * (np.sqrt(2 * r2 / (r1 + r2)) - 1)
    dv2 = np.sqrt(MU / r2) * (1 - np.sqrt(2 * r1 / (r1 + r2)))
    return dv1, dv2


def _hohmann_tof(r1, r2):
    return np.pi * np.sqrt((0.5 * (r1 + r2)) ** 3 / MU)


R1, R2 = 7000e3, 9000e3
TOF_HOHMANN = _hohmann_tof(R1, R2)


# ======================================================================
# Transfer catalog
# ======================================================================
# Each entry builds (departure, arrival, transfer_ssapy kwargs) plus
# bounds used by the pytest benchmarks: dv_max is a generous sanity
# ceiling (analytic where available), dv_min catches degenerate zeros.

def _case_leo_raise():
    """Coplanar circular orbit raise, 7000 -> 9000 km, 150 deg sweep."""
    dep = _circular_state(R1, 0.0)
    arr = _circular_state(R2, np.deg2rad(150.0))
    tof = (150.0 / 180.0) * TOF_HOHMANN
    return dict(dep=dep, arr=arr, tof=tof, kwargs={},
                dv_min=sum(_hohmann_dv(R1, R2)),
                dv_max=1.4 * sum(_hohmann_dv(R1, R2)))


def _case_leo_to_geo():
    """LEO (500 km) to GEO, 160 deg sweep -- the classic big raise."""
    r_leo = EARTH_RADIUS + 500e3
    dep = _circular_state(r_leo, 0.0)
    arr = _circular_state(RGEO, np.deg2rad(160.0))
    tof = (160.0 / 180.0) * _hohmann_tof(r_leo, RGEO)
    return dict(dep=dep, arr=arr, tof=tof, kwargs=dict(rk_step=30.0),
                dv_min=sum(_hohmann_dv(r_leo, RGEO)),
                dv_max=1.25 * sum(_hohmann_dv(r_leo, RGEO)))


def _case_phasing():
    """Co-orbital rendezvous: catch a target 10 deg ahead in 0.9 rev."""
    period = 2 * np.pi * np.sqrt(R1 ** 3 / MU)
    tof = 0.9 * period
    sweep = 0.9 * 360.0 + 10.0          # chaser flies 334 deg
    dep = _circular_state(R1, 0.0)
    arr = _circular_state(R1, np.deg2rad(sweep))
    return dict(dep=dep, arr=arr, tof=tof, kwargs={},
                dv_min=1.0, dv_max=400.0)


def _case_geo_repositioning():
    """GEO station shift: drift +3 deg of longitude in 0.95 sidereal day.

    Same-radius transfers are only tangential (cheap) when the sweep
    nears a full revolution, so realistic repositioning drifts over
    nearly whole orbits: at this window the cost is ~18 m/s, versus
    >100 m/s for a half-day window whose Lambert arc forces large
    radial delta-v at both burns.
    """
    omega = np.sqrt(MU / RGEO ** 3)
    tof = 0.95 * 86164.0
    sweep = omega * tof + np.deg2rad(3.0)
    dep = _circular_state(RGEO, 0.0)
    arr = _circular_state(RGEO, sweep)
    return dict(dep=dep, arr=arr, tof=tof, kwargs=dict(rk_step=60.0),
                dv_min=0.5, dv_max=60.0)


def _case_heo_injection():
    """LEO to the apogee of an HEO (30000 km, 10 deg inclined).

    Departure radius matches the HEO perigee and the sweep stays near
    apsis-to-apsis so the transfer conic remains near-tangential at both
    ends -- its perigee keeps a healthy ~570 km margin above the Earth
    (steeper inclined-apogee geometries skim the surface and trip
    SSAPy's impact event).
    """
    r_leo = 7000e3
    r_apo = 30000e3
    a_heo = 0.5 * (r_leo + r_apo)
    v_apo = np.sqrt(MU * (2 / r_apo - 1 / a_heo))
    dep = _circular_state(r_leo, 0.0)
    arr = _apsis_state(r_apo, np.deg2rad(165.0), v_apo,
                       inclination=np.deg2rad(10.0))
    tof = (165.0 / 180.0) * _hohmann_tof(r_leo, r_apo)
    return dict(dep=dep, arr=arr, tof=tof, kwargs=dict(rk_step=30.0),
                dv_min=sum(_hohmann_dv(r_leo, r_apo)) * 0.5,
                dv_max=4500.0)


def _case_plane_change():
    """Combined raise + 15 deg plane change to MEO, 130 deg sweep."""
    r_meo = 15000e3
    dep = _circular_state(R1, 0.0)
    arr = _circular_state(r_meo, np.deg2rad(130.0),
                          inclination=np.deg2rad(15.0))
    tof = (130.0 / 180.0) * _hohmann_tof(R1, r_meo)
    # Coplanar Hohmann floor plus full plane change at arrival ceiling.
    dv_h = sum(_hohmann_dv(R1, r_meo))
    dv_pc = 2 * np.sqrt(MU / r_meo) * np.sin(np.deg2rad(15.0) / 2)
    return dict(dep=dep, arr=arr, tof=tof, kwargs={},
                dv_min=dv_h, dv_max=1.6 * (dv_h + dv_pc))


CATALOG = {
    "LEO raise (coplanar)": _case_leo_raise,
    "LEO -> GEO": _case_leo_to_geo,
    "Phasing rendezvous": _case_phasing,
    "GEO repositioning": _case_geo_repositioning,
    "HEO apogee injection": _case_heo_injection,
    "Raise + 15 deg plane change": _case_plane_change,
}

_RESULTS = {}  # cache: case name -> (case dict, TransferResult)


def _run_case(name):
    """Run a catalog transfer once (cached) and assert its benchmarks."""
    if name in _RESULTS:
        return _RESULTS[name]
    case = CATALOG[name]()
    (r1, v1), (r2, v2), tof = case["dep"], case["arr"], case["tof"]
    res = transfer_ssapy((r1, v1, 0.0), (r2, v2, tof), **case["kwargs"])
    # Refined finite burns must reach the target state...
    assert res.arrival_error < 10.0, (name, res.arrival_error)
    vel_err = np.linalg.norm(res.trajectory["v"][-1] - v2)
    assert vel_err < 0.01, (name, vel_err)
    # ...for a physically sensible delta-v.
    assert case["dv_min"] - 1e-6 <= res.dv_total <= case["dv_max"], (
        name, res.dv_total, case["dv_min"], case["dv_max"])
    _RESULTS[name] = (case, res)
    return _RESULTS[name]


# ======================================================================
# Pytest benchmarks (no graphics)
# ======================================================================

def test_lambert_reproduces_keplerian_arc():
    """Lambert between two points of one Keplerian arc returns its v's."""
    r0 = np.array([7000e3, 1000e3, 500e3])
    v0 = np.array([-1.2e3, 7.0e3, 1.5e3])
    tof = 2500.0
    (r_arc, v_arc) = rv(Orbit(r0, v0, t=0.0), np.array([0.0, tof]),
                        propagator=KeplerianPropagator())
    v1, v2 = solve_lambert(r_arc[0], r_arc[1], tof, mu=MU)
    assert np.linalg.norm(v1 - v_arc[0]) < 1e-3
    assert np.linalg.norm(v2 - v_arc[1]) < 1e-3


def test_same_orbit_transfer_needs_no_delta_v():
    """Boundary states on the same orbit -> near-zero total delta-v."""
    r0 = np.array([8000e3, 0, 0])
    v0 = np.array([0, np.sqrt(MU / 8000e3), 0])
    tof = 2000.0
    (r_arc, v_arc) = rv(Orbit(r0, v0, t=0.0), np.array([0.0, tof]),
                        propagator=KeplerianPropagator())
    res = transfer_ssapy((r_arc[0], v_arc[0], 0.0),
                         (r_arc[1], v_arc[1], tof), propagate=False)
    assert res.dv_total < 1e-2


def test_near_hohmann_approaches_analytic_minimum():
    """At a wide transfer angle, total dv nears the Hohmann optimum."""
    theta = np.deg2rad(177.0)
    r1, v1 = _circular_state(R1, 0.0)
    r2, v2 = _circular_state(R2, theta)
    tof = (theta / np.pi) * TOF_HOHMANN
    res = transfer_ssapy((r1, v1, 0.0), (r2, v2, tof), propagate=False)
    dv_h = sum(_hohmann_dv(R1, R2))
    assert res.dv_total > dv_h - 1e-6          # Hohmann is the floor
    assert res.dv_total < dv_h * 1.02          # and we are within 2 %


def test_dv_budget_warning_and_raise():
    import warnings
    r1, v1 = _circular_state(R1, 0.0)
    r2, v2 = _circular_state(R2, np.deg2rad(150.0))
    tof = 0.85 * TOF_HOHMANN
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = transfer_ssapy((r1, v1, 0.0), (r2, v2, tof),
                             dv_budget=100.0, propagate=False)
    assert res.within_budget is False and len(w) == 1
    try:
        transfer_ssapy((r1, v1, 0.0), (r2, v2, tof), dv_budget=100.0,
                       raise_on_budget=True, propagate=False)
    except ValueError:
        pass
    else:
        raise AssertionError("budget overrun did not raise")


def test_lambert_180_degree_geometry_raises():
    try:
        solve_lambert([7000e3, 0, 0], [-9000e3, 0, 0], 3000.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("singular geometry did not raise")


# ---- catalog: one benchmark per transfer type ------------------------

def test_transfer_leo_raise():
    _run_case("LEO raise (coplanar)")


def test_transfer_leo_to_geo():
    _run_case("LEO -> GEO")


def test_transfer_phasing_rendezvous():
    """Same orbit, 10 deg phase catch-up: dv exists, both burns small."""
    case, res = _run_case("Phasing rendezvous")
    assert all(b.dv_mag < 250.0 for b in res.burns)


def test_transfer_geo_repositioning():
    """A 3 deg GEO drift over ~1 rev costs only tens of m/s."""
    case, res = _run_case("GEO repositioning")
    assert res.dv_total < 30.0


def test_transfer_heo_apogee_injection():
    """Inclined HEO target: burns include cross-track (W) components."""
    case, res = _run_case("HEO apogee injection")
    w_total = sum(abs(b.dv_ntw[2]) for b in res.burns)
    assert w_total > 100.0


def test_transfer_combined_plane_change():
    """Raise + 15 deg inclination: dv exceeds the coplanar Hohmann."""
    case, res = _run_case("Raise + 15 deg plane change")
    assert res.dv_total > case["dv_min"] + 100.0


def test_retrograde_geometry_costs_far_more():
    """Solving the same boundary states retrograde is vastly costlier."""
    r1, v1 = _circular_state(R1, 0.0)
    r2, v2 = _circular_state(R2, np.deg2rad(150.0))
    tof = (150.0 / 180.0) * TOF_HOHMANN
    pro = transfer_ssapy((r1, v1, 0.0), (r2, v2, tof), propagate=False)
    retro = transfer_ssapy((r1, v1, 0.0), (r2, v2, tof),
                           prograde=False, propagate=False)
    assert retro.dv_total > 5.0 * pro.dv_total


def test_alternate_propagator_matches_default():
    """RK4 fixed-step reproduces the default RK78 plan."""
    r1, v1 = _circular_state(R1, 0.0)
    r2, v2 = _circular_state(R2, np.deg2rad(150.0))
    tof = (150.0 / 180.0) * TOF_HOHMANN
    res = transfer_ssapy((r1, v1, 0.0), (r2, v2, tof),
                         propagator=RK4Propagator, rk_step=5.0)
    _, ref = _run_case("LEO raise (coplanar)")
    assert res.arrival_error < 10.0
    assert abs(res.dv_total - ref.dv_total) < 1.0


# ---- transfer_optimal: orbit-to-orbit search benchmarks ---------------

_OPT_RESULTS = {}


def _opt_boundaries():
    dep = _circular_state(R1, 0.0)
    arr = _circular_state(R2, np.deg2rad(40.0))
    return (dep[0], dep[1], 0.0), (arr[0], arr[1], 0.0)


def test_optimal_rediscovers_hohmann():
    """Free departure/TOF search lands on the Hohmann optimum."""
    o1, o2 = _opt_boundaries()
    res = transfer_optimal(o1, o2)
    _OPT_RESULTS["min_dv"] = res
    assert res.transfer.arrival_error < 10.0
    assert res.dv_total < 1.05 * sum(_hohmann_dv(R1, R2))


def test_optimal_insertion_no_worse_than_rendezvous():
    """Freeing the arrival phase can only help."""
    o1, o2 = _opt_boundaries()
    res = transfer_optimal(o1, o2, rendezvous=False,
                           n_grid=(16, 16), n_phase=12)
    ref = _OPT_RESULTS.get("min_dv") or transfer_optimal(o1, o2)
    assert res.dv_total <= ref.dv_total + 1.0


def test_optimal_first_burn_only_intercept():
    """arrival_burn=False plans a one-burn flyby through the target."""
    o1, o2 = _opt_boundaries()
    res = transfer_optimal(o1, o2, arrival_burn=False)
    ref = _OPT_RESULTS.get("min_dv") or transfer_optimal(o1, o2)
    assert len(res.transfer.burns) == 1
    assert res.transfer.arrival_error < 10.0
    assert res.dv_total < ref.dv_total          # one burn beats two
    # ...and the cheapest intercept is the first Hohmann burn.
    assert abs(res.dv_total - _hohmann_dv(R1, R2)[0]) < 25.0


def test_optimal_min_time_respects_budget():
    """min_time returns the fastest transfer fitting the budget."""
    o1, o2 = _opt_boundaries()
    ref = _OPT_RESULTS.get("min_dv") or transfer_optimal(o1, o2)
    fast = transfer_optimal(o1, o2, objective="min_time",
                            dv_budget=1500.0)
    assert fast.tof < ref.tof
    assert fast.dv_total <= 1500.0 * 1.02   # impulsive-vs-finite wiggle
    try:
        transfer_optimal(o1, o2, objective="min_time", dv_budget=100.0)
    except ValueError:
        pass
    else:
        raise AssertionError("infeasible budget did not raise")


def test_optimal_finds_cheap_geo_drift_window():
    """The optimizer discovers the near-full-rev GEO drift on its own."""
    dep = _circular_state(RGEO, 0.0)
    arr = _circular_state(RGEO, np.deg2rad(3.0))
    res = transfer_optimal((dep[0], dep[1], 0.0), (arr[0], arr[1], 0.0),
                           tof_range=(3600.0, 1.3 * 86164.0),
                           n_grid=(24, 48), rk_step=60.0)
    _OPT_RESULTS["geo"] = res
    assert res.dv_total < 25.0
    assert res.transfer.arrival_error < 10.0


# ---- engine model (thrust / mass / isp) -------------------------------

def test_engine_sized_burns_match_hardware():
    """thrust/mass size each burn so implied thrust matches the spec,
    and isp attaches Tsiolkovsky propellant estimates."""
    r1, v1 = _circular_state(R1, 0.0)
    r2, v2 = _circular_state(R2, np.deg2rad(150.0))
    tof = (150.0 / 180.0) * TOF_HOHMANN
    res = transfer_ssapy((r1, v1, 0.0), (r2, v2, tof),
                         thrust=1000.0, mass=500.0, isp=320.0)
    assert res.arrival_error < 10.0
    for b in res.burns:
        implied = 500.0 * b.dv_mag / b.duration
        assert abs(implied - 1000.0) < 30.0
        assert b.propellant_mass is not None and b.propellant_mass > 0


def test_engine_too_weak_raises():
    """Burns that cannot fit a third of the TOF fail loudly, and the
    engine parameters validate together."""
    r1, v1 = _circular_state(R1, 0.0)
    r2, v2 = _circular_state(R2, np.deg2rad(150.0))
    tof = (150.0 / 180.0) * TOF_HOHMANN
    for kw in (dict(thrust=220.0, mass=500.0),   # ~0.44 m/s^2: too weak
               dict(thrust=220.0), dict(mass=500.0), dict(isp=320.0)):
        try:
            transfer_ssapy((r1, v1, 0.0), (r2, v2, tof), **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kw} did not raise")


def test_burn_accel_shortcut_matches_thrust_mass():
    """burn_accel [m/s^2] alone reproduces the thrust/mass plan and is
    mutually exclusive with it."""
    r1, v1 = _circular_state(R1, 0.0)
    r2, v2 = _circular_state(R2, np.deg2rad(150.0))
    tof = (150.0 / 180.0) * TOF_HOHMANN
    ra = transfer_ssapy((r1, v1, 0.0), (r2, v2, tof), burn_accel=2.0)
    rt = transfer_ssapy((r1, v1, 0.0), (r2, v2, tof),
                        thrust=1000.0, mass=500.0)
    assert ra.arrival_error < 10.0
    for ba, bt in zip(ra.burns, rt.burns):
        assert abs(ba.duration - bt.duration) < 1e-6
    for kw in (dict(burn_accel=2.0, thrust=1000.0, mass=500.0),
               dict(burn_accel=2.0, isp=320.0)):
        try:
            transfer_ssapy((r1, v1, 0.0), (r2, v2, tof), **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kw} did not raise")


def test_optimal_min_time_is_engine_honest():
    """A weaker engine forces min_time onto a slower transfer, and the
    delta-v budget holds on the *refined* finite-burn plan."""
    o1, o2 = _opt_boundaries()
    strong = transfer_optimal(o1, o2, objective="min_time",
                              dv_budget=1500.0, thrust=2000.0, mass=500.0)
    weak = transfer_optimal(o1, o2, objective="min_time",
                            dv_budget=1500.0, thrust=800.0, mass=500.0)
    assert weak.tof > strong.tof
    assert strong.dv_total <= 1500.0 and weak.dv_total <= 1500.0
    assert (strong.transfer.arrival_error < 10.0
            and weak.transfer.arrival_error < 10.0)


# ======================================================================
# Demo mode: figures (only when run as a script, never under pytest)
# ======================================================================

def _orbit_ring(state, n=361):
    """Sample one full revolution of the orbit through ``state``."""
    r, v = state
    orb = Orbit(np.asarray(r, float), np.asarray(v, float), t=0.0)
    period = 2 * np.pi * np.sqrt(orb.a ** 3 / MU)
    ts = np.linspace(0.0, period, n)
    rr, _ = rv(orb, ts, propagator=KeplerianPropagator())
    return rr


def _demo_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ssapy_toolkit.plots import yufig

    for name in CATALOG:
        _run_case(name)

    # ---- Figure 1: gallery of transfer geometries ---------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, name in zip(axes.ravel(), CATALOG):
        case, res = _RESULTS[name]
        transfer_trajectory_plot(
            res, ax=ax, annotate_burns=False,
            title=(f"{name}\ndv {res.dv_total:.1f} m/s | "
                   f"arrival err {res.arrival_error:.1f} m"))
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("transfer_ssapy catalog: propagated two-burn transfers "
                 "(x-y projection, km)", fontsize=13)
    fig.tight_layout()
    yufig(fig, f"{FIGDIR}/demo_transfer_catalog_gallery.jpg")
    plt.close(fig)

    # ---- Figure 2: 3-D view of the combined plane change --------------
    case, res = _RESULTS["Raise + 15 deg plane change"]
    transfer_trajectory_plot(
        res, three_d=True,
        title="Raise + plane change: cross-track (W) delta-v carries "
              "the inclination",
        save_path=f"{FIGDIR}/demo_transfer_plane_change_3d.jpg")

    # ---- Figure 3: delta-v budget by transfer type ---------------------
    names = list(CATALOG)
    b1 = [_RESULTS[n][1].burns[0].dv_mag for n in names]
    b2 = [_RESULTS[n][1].burns[1].dv_mag for n in names]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    xpos = np.arange(len(names))
    ax.bar(xpos, b1, 0.6, label="burn 1 (departure)", color="C0")
    ax.bar(xpos, b2, 0.6, bottom=b1, label="burn 2 (arrival)", color="C1")
    for i, n in enumerate(names):
        total = _RESULTS[n][1].dv_total
        ax.text(i, total, f"{total:.0f}", ha="center", va="bottom",
                fontsize=9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("delta-v [m/s] (log scale)")
    ax.set_title("Total delta-v by transfer type "
                 "(stacked departure/arrival burns)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    yufig(fig, f"{FIGDIR}/demo_transfer_dv_by_type.jpg")
    plt.close(fig)

    # ---- Figure 4: benchmark vs the analytic Hohmann minimum ----------
    r1, v1 = _circular_state(R1, 0.0)
    thetas = np.deg2rad(np.linspace(60, 178, 40))
    totals = []
    for th in thetas:
        ra, va = _circular_state(R2, th)
        r = transfer_ssapy((r1, v1, 0.0), (ra, va, th / np.pi * TOF_HOHMANN),
                           propagate=False)
        totals.append(r.dv_total)
    dv_h = sum(_hohmann_dv(R1, R2))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.rad2deg(thetas), np.array(totals), "C0.-",
            label="transfer_ssapy (Lambert, impulsive)")
    ax.axhline(dv_h, color="k", lw=3, alpha=0.35,
               label=f"analytic Hohmann minimum ({dv_h:.1f} m/s)")
    ax.set_xlabel("transfer sweep angle [deg]")
    ax.set_ylabel("total delta-v [m/s]")
    ax.set_title(
        f"{R1/1e3:.0f} km -> {R2/1e3:.0f} km circular transfer:\n"
        "total dv approaches the Hohmann optimum as sweep -> 180 deg")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    yufig(fig, f"{FIGDIR}/demo_transfer_hohmann_benchmark.jpg")
    plt.close(fig)

    # ---- Figure 5: finite-burn arrival error, raw vs refined ----------
    r2c, v2c = _circular_state(R2, np.deg2rad(150.0))
    tof = (150.0 / 180.0) * TOF_HOHMANN
    durations = np.array([1.0, 5.0, 15.0, 60.0, 180.0])
    err_raw, err_ref = [], []
    for d in durations:
        kw = dict(burn_duration=float(d))
        err_raw.append(transfer_ssapy(
            (r1, v1, 0.0), (r2c, v2c, tof), refine=False, **kw).arrival_error)
        err_ref.append(transfer_ssapy(
            (r1, v1, 0.0), (r2c, v2c, tof), **kw).arrival_error)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(durations, err_raw, "C3o-",
              label="impulsive Lambert command, no correction")
    ax.loglog(durations, err_ref, "C0s-",
              label="with differential correction (default)")
    ax.axhline(10.0, color="k", ls=":", lw=1,
               label="10 m refinement tolerance")
    ax.set_xlabel("finite burn duration [s]")
    ax.set_ylabel("propagated arrival position error [m]")
    ax.set_title("Why the shooting refinement matters:\n"
                 "the NTW frame rotates during a burn, so raw Lambert "
                 "commands miss")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    yufig(fig, f"{FIGDIR}/demo_transfer_refinement.jpg")
    plt.close(fig)

    # ---- Figure 6: burn timeline with strengths (engine model) --------
    r1d, v1d = _circular_state(R1, 0.0)
    r2d, v2d = _circular_state(R2, np.deg2rad(150.0))
    eng = transfer_ssapy((r1d, v1d, 0.0),
                         (r2d, v2d, (150.0 / 180.0) * TOF_HOHMANN),
                         thrust=1000.0, mass=500.0, isp=320.0)
    transfer_burn_profile_plot(
        eng, title="Burn timeline: 500 kg / 1 kN / Isp 320 s, "
                   f"total dv {eng.dv_total:.1f} m/s",
        save_path=f"{FIGDIR}/demo_transfer_burn_profile.jpg")

    # ---- Figures 7-8: transfer_optimal mission-designer curves --------
    o1, o2 = _opt_boundaries()
    transfer_optimal(o1, o2, visualize=True,
                     fig_prefix=f"{FIGDIR}/demo_transfer_optimal_leo")
    dep = _circular_state(RGEO, 0.0)
    arr = _circular_state(RGEO, np.deg2rad(3.0))
    transfer_optimal((dep[0], dep[1], 0.0), (arr[0], arr[1], 0.0),
                     tof_range=(3600.0, 1.3 * 86164.0),
                     n_grid=(24, 48), rk_step=60.0, visualize=True,
                     fig_prefix=f"{FIGDIR}/demo_transfer_optimal_geo")

    for fname in ["demo_transfer_catalog_gallery",
                  "demo_transfer_plane_change_3d",
                  "demo_transfer_dv_by_type",
                  "demo_transfer_hohmann_benchmark",
                  "demo_transfer_refinement",
                  "demo_transfer_burn_profile",
                  "demo_transfer_optimal_leo_designer_curves",
                  "demo_transfer_optimal_geo_designer_curves"]:
        print(f"Saved via yufig: {FIGDIR}/{fname}.jpg")


if __name__ == "__main__":
    print(f"Transfer planner imported from: {_SOURCE}")
    for fn in [test_lambert_reproduces_keplerian_arc,
               test_same_orbit_transfer_needs_no_delta_v,
               test_near_hohmann_approaches_analytic_minimum,
               test_dv_budget_warning_and_raise,
               test_lambert_180_degree_geometry_raises,
               test_transfer_leo_raise,
               test_transfer_leo_to_geo,
               test_transfer_phasing_rendezvous,
               test_transfer_geo_repositioning,
               test_transfer_heo_apogee_injection,
               test_transfer_combined_plane_change,
               test_retrograde_geometry_costs_far_more,
               test_alternate_propagator_matches_default,
               test_optimal_rediscovers_hohmann,
               test_optimal_insertion_no_worse_than_rendezvous,
               test_optimal_first_burn_only_intercept,
               test_optimal_min_time_respects_budget,
               test_optimal_finds_cheap_geo_drift_window,
               test_engine_sized_burns_match_hardware,
               test_engine_too_weak_raises,
               test_burn_accel_shortcut_matches_thrust_mass,
               test_optimal_min_time_is_engine_honest]:
        fn()
        print(f"PASS  {fn.__name__}")
    _demo_figures()
