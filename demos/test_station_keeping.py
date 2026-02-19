"""
Station-keeping simulation — SSAPy based.

Architecture
------------
* Reference orbit  : KeplerianPropagator (pure two-body).
  This is the ideal ellipse the spacecraft must track.

* Spacecraft orbit : RK4Propagator with high-fidelity force model:
      AccelKepler
    + AccelHarmonic(earth, 140, 140)   Earth gravity 140x140
    + AccelHarmonic(moon,  10,  10)    Lunar harmonics
    + AccelThirdBody(sun)              Solar point-mass gravity
    + AccelThirdBody(moon)             Lunar point-mass gravity
    + AccelDrag                        Atmospheric drag (propkw: CD, area, mass)
    + AccelSolRad                      Solar radiation pressure (propkw: CR, area, mass)
    + AccelEarthRad                    Earth albedo radiation (propkw: CR, area, mass)

Control loop  (every check_interval_s seconds)
----------------------------------------------
1. Propagate both orbits forward check_interval_s seconds.
2. Compare final states in the NTW frame.
3. If |dr| > deadband_m, compute an impulsive CW-feedback dV in NTW,
   cap it at a_max * check_interval_s, apply instantaneously, and continue.

Note: spacecraft physical properties (CD, CR, area, mass) are passed via
Orbit(propkw=...) so SSAPy's stateless Accel objects can read them at
integration time.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from astropy.time import Time

import ssapy.compute as compute
from ssapy.orbit import Orbit
from ssapy.propagator import KeplerianPropagator, RK4Propagator
from ssapy.accel import AccelKepler, AccelSolRad, AccelEarthRad, AccelDrag, AccelConstNTW
from ssapy.gravity import AccelHarmonic, AccelThirdBody
from ssapy.body import get_body

# ── constants ─────────────────────────────────────────────────────────────────
MU_EARTH = 3.986004418e14   # m³ s⁻²
RE        = 6.3781e6        # m


# ── NTW frame helpers ─────────────────────────────────────────────────────────

def ntw_frame(r, v):
    n = v / np.linalg.norm(v)
    w = np.cross(r, v); w /= np.linalg.norm(w)
    t = np.cross(n, w)
    return n, t, w


def eci_to_ntw(dr, dv, r_ref, v_ref):
    n, t, w = ntw_frame(r_ref, v_ref)
    return (np.array([dr@n, dr@t, dr@w]),
            np.array([dv@n, dv@t, dv@w]))


def ntw_to_eci(a_ntw, r_ref, v_ref):
    n, t, w = ntw_frame(r_ref, v_ref)
    return a_ntw[0]*n + a_ntw[1]*t + a_ntw[2]*w


# ── CW impulsive ΔV in NTW ────────────────────────────────────────────────────

def compute_burn(r_sc, v_sc, r_ref, v_ref, a_thrust, max_burn_s):
    """
    Compute a finite burn: constant acceleration a_thrust [m/s²] in the
    direction that nulls the velocity mismatch, for a duration that exactly
    delivers the required ΔV — capped at max_burn_s.

    Returns (burn_dir_eci, burn_duration_s) or (None, 0) if no burn needed.
    """
    dv_needed = v_ref - v_sc          # velocity we need to add to match reference

    # Small position feedback: nudge direction slightly toward reference
    dr     = r_ref - r_sc
    dr_mag = np.linalg.norm(dr)
    dv_mag = np.linalg.norm(dv_needed)

    if dv_mag < 1e-9 and dr_mag < 1.0:
        return None, 0.0

    # Blend velocity null + position pull (position term tapers off once close)
    alpha = 0.05
    direction = dv_needed + alpha * (dr / max(dr_mag, 1.0))
    direction /= np.linalg.norm(direction)

    # Duration to deliver dv_needed, capped
    dv_total  = dv_mag
    duration  = min(dv_total / a_thrust, max_burn_s)

    return direction, duration


# ── propagation helper ────────────────────────────────────────────────────────

def prop_step(orbit, dt_s, propagator):
    """Propagate orbit by dt_s seconds; return (r, v) at the end."""
    t_end = np.array([orbit.t + dt_s])
    r, v  = compute.rv(orbit, t_end, propagator)
    return r[0], v[0]


# ── main simulation ───────────────────────────────────────────────────────────

def run_simulation(
    # reference orbit
    a_km             = 6778.0,
    ecc              = 1e-4,
    inc_deg          = 51.6,
    raan_deg         = 0.0,
    aop_deg          = 0.0,
    nu0_deg          = 0.0,
    t0_str           = "2026-01-01T00:00:00",
    # simulation
    t_span_h         = 24.0,
            check_interval_s = 300.0,
    # controller
    deadband_m       = 100.0,
    a_max_mps2       = 5e-5,
    # spacecraft properties (passed as propkw to SSAPy accels)
    sc_mass_kg       = 100.0,
    sc_area_m2       = 1.0,
    sc_Cd            = 2.2,
    sc_Cr            = 1.3,
    # initial offset from reference
    dr0_m            = np.array([50.0, 0.0, 0.0]),
    dv0_mps          = np.array([0.0,  0.0, 0.0]),
    # RK4 internal step
    rk4_h_s          = 30.0,
    # output
    figure_dir       = os.path.expanduser("~/yu_figures/tests"),
    verbose          = True,
):
    t0_gps   = Time(t0_str, scale="utc").gps
    dv_max   = a_max_mps2 * check_interval_s
    n_checks = int(t_span_h * 3600.0 / check_interval_s)

    print(f"[setup] a={a_km} km  inc={inc_deg}°  span={t_span_h}h  "
          f"interval={check_interval_s}s  a_max={a_max_mps2:.1e} m/s²  "
          f"deadband={deadband_m} m  dv_max/burn={dv_max*1e3:.2f} mm/s  "
          f"Cd={sc_Cd}  Cr={sc_Cr}  A={sc_area_m2} m²  m={sc_mass_kg} kg")

    # ── reference orbit (Keplerian, evaluated analytically) ──────────────────
    ref_orbit = Orbit.fromKeplerianElements(
        a_km*1e3, ecc,
        np.radians(inc_deg), np.radians(raan_deg),
        np.radians(aop_deg), np.radians(nu0_deg),
        t=t0_gps
    )
    kep_prop = KeplerianPropagator()

    # ── spacecraft initial state ──────────────────────────────────────────────
    r0_ref, v0_ref = compute.rv(ref_orbit, np.array([t0_gps]), kep_prop)
    r0_ref, v0_ref = r0_ref[0], v0_ref[0]

    r_sc = r0_ref + dr0_m
    v_sc = v0_ref + dv0_mps
    t_sc = t0_gps

    # spacecraft physical properties injected into every Orbit via propkw
    sc_propkw  = dict(CD=sc_Cd, CR=sc_Cr, area=sc_area_m2, mass=sc_mass_kg)
    max_burn_s = check_interval_s  # burn can be up to one full interval

    # ── spacecraft force model ────────────────────────────────────────────────
    earth = get_body("earth")
    moon  = get_body("moon")
    sun   = get_body("sun")

    # Thrust accel object — direction updated at each burn
    ntw_thrust   = AccelConstNTW()
    sc_accel_base = (
        AccelKepler(earth.mu)
        + AccelHarmonic(earth, 140, 140)
        + AccelHarmonic(moon,  10,  10)
        + AccelThirdBody(sun)
        + AccelThirdBody(moon)
        + AccelDrag()
        + AccelSolRad()
        + AccelEarthRad()
    )
    sc_accel_thrust = sc_accel_base + ntw_thrust
    rk4_prop_coast  = RK4Propagator(sc_accel_base,   h=rk4_h_s)
    rk4_prop_thrust = RK4Propagator(sc_accel_thrust, h=rk4_h_s)

    # ── storage ───────────────────────────────────────────────────────────────
    times_s  = np.zeros(n_checks + 1)
    sep      = np.zeros(n_checks + 1)
    sep_ntw  = np.zeros((n_checks + 1, 3))
    dv_hist  = np.zeros(n_checks + 1)
    total_dv = 0.0
    n_burns  = 0

    dr0        = r_sc - r0_ref
    sep[0]     = np.linalg.norm(dr0)
    n, t, w    = ntw_frame(r0_ref, v0_ref)
    sep_ntw[0] = [dr0@n, dr0@t, dr0@w]

    # ── main loop ─────────────────────────────────────────────────────────────
    for k in range(n_checks):
        t_next = t_sc + check_interval_s

        # propagate reference analytically (Keplerian, from t0)
        r_ref_next, v_ref_next = compute.rv(
            ref_orbit, np.array([t_next]), kep_prop
        )
        r_ref_next, v_ref_next = r_ref_next[0], v_ref_next[0]

        # propagate spacecraft: burn phase then coast phase
        pos_err_pre = np.linalg.norm(r_sc - r_ref_now)
        burn_dir, burn_dur = compute_burn(
            r_sc, v_sc, r_ref_now, v_ref_now, a_thrust_mps2, max_burn_s
        )

        dv_applied = 0.0
        if burn_dir is not None and burn_dur > 0.0 and pos_err_pre > deadband_m:
            # Set constant thrust direction on the AccelConstNTW object
            ntw_thrust.aN = burn_dir[0] * a_thrust_mps2  # ECI components directly
            ntw_thrust.aT = burn_dir[1] * a_thrust_mps2
            ntw_thrust.aW = burn_dir[2] * a_thrust_mps2
            # Override __call__ to use raw ECI direction instead of NTW frame
            ntw_thrust._eci_dir = burn_dir * a_thrust_mps2

            coast_dur = check_interval_s - burn_dur

            # Burn phase
            sc_orbit  = Orbit(r=r_sc, v=v_sc, t=t_sc, propkw=sc_propkw)
            r_sc, v_sc = prop_step(sc_orbit, burn_dur, rk4_prop_thrust)
            dv_applied = a_thrust_mps2 * burn_dur

            # Coast phase (if time remains)
            if coast_dur > 1.0:
                sc_orbit = Orbit(r=r_sc, v=v_sc,
                                 t=t_sc + burn_dur, propkw=sc_propkw)
                r_sc, v_sc = prop_step(sc_orbit, coast_dur, rk4_prop_coast)
        else:
            # Full coast, no burn
            sc_orbit  = Orbit(r=r_sc, v=v_sc, t=t_sc, propkw=sc_propkw)
            r_sc, v_sc = prop_step(sc_orbit, check_interval_s, rk4_prop_coast)

        if not np.all(np.isfinite(r_sc_next)):
            print(f"[ERROR] NaN at k={k} — orbit lost")
            sep[k+1:] = np.nan
            break

        # error
        pos_err = np.linalg.norm(r_sc_next - r_ref_next)

        # correction
        dv_applied = 0.0
        if pos_err > deadband_m:
            dv_cmd = cw_dv_ntw(r_sc_next, v_sc_next,
                               r_ref_next, v_ref_next, dv_max)
            v_sc_next += ntw_to_eci(dv_cmd, r_ref_next, v_ref_next)
            dv_applied = np.linalg.norm(dv_cmd)
            total_dv  += dv_applied
            n_burns   += 1
            if verbose:
                print(f"[burn] k={k:5d}  t={t_next-t0_gps:8.1f}s  "
                      f"err={pos_err:9.2f} m  |dV|={dv_applied*1e3:.3f} mm/s  "
                      f"NTW=[{dv_cmd[0]*1e3:.3f}, "
                      f"{dv_cmd[1]*1e3:.3f}, "
                      f"{dv_cmd[2]*1e3:.3f}] mm/s")

        # advance
        r_sc, v_sc, t_sc = r_sc_next, v_sc_next, t_next

        # record
        times_s[k+1]  = t_sc - t0_gps
        dr_now        = r_sc - r_ref_next
        sep[k+1]      = np.linalg.norm(dr_now)
        n, t, w       = ntw_frame(r_ref_next, v_ref_next)
        sep_ntw[k+1]  = [dr_now@n, dr_now@t, dr_now@w]
        dv_hist[k+1]  = dv_applied

        if verbose and (k+1) % max(1, int(3600/check_interval_s)) == 0:
            print(f"[status]  t={times_s[k+1]/3600:.1f}h  "
                  f"err={sep[k+1]:.2f} m  burns={n_burns}  "
                  f"total_dv={total_dv*1e3:.3f} mm/s")

    # ── summary ───────────────────────────────────────────────────────────────
    valid = sep[np.isfinite(sep)]
    print(f"\n--- Summary ---")
    print(f"burns        : {n_burns}")
    print(f"total dV     : {total_dv*1e3:.3f} mm/s")
    if len(valid):
        print(f"final error  : {valid[-1]:.2f} m")
        print(f"max error    : {np.nanmax(sep):.2f} m")
        print(f"mean error   : {np.nanmean(sep):.2f} m")

    # ── plots ─────────────────────────────────────────────────────────────────
    os.makedirs(figure_dir, exist_ok=True)
    t_h = times_s / 3600.0

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t_h, sep, lw=0.9, color="steelblue", label="3-D separation")
    ax.axhline(deadband_m, color="orange", ls="--", lw=1.2,
               label=f"deadband ({deadband_m:.0f} m)")
    ax.set_xlabel("Time [h]");  ax.set_ylabel("Position error [m]")
    ax.set_title("Station-keeping: spacecraft vs. Keplerian reference")
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.35)
    fig.tight_layout()
    p = os.path.join(figure_dir, "sk_separation.png")
    fig.savefig(p, dpi=150);  plt.close(fig);  print(f"Figure saved: {p}")

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for j, (lbl, col) in enumerate(zip(
            ["Along-track N [m]", "Radial T [m]", "Normal W [m]"],
            ["tab:blue", "tab:orange", "tab:green"])):
        axes[j].plot(t_h, sep_ntw[:, j], lw=0.7, color=col)
        axes[j].axhline(0, color="k", lw=0.5)
        axes[j].set_ylabel(lbl);  axes[j].grid(True, alpha=0.35)
    axes[-1].set_xlabel("Time [h]")
    fig.suptitle("NTW position error components")
    fig.tight_layout()
    p = os.path.join(figure_dir, "sk_ntw_error.png")
    fig.savefig(p, dpi=150);  plt.close(fig);  print(f"Figure saved: {p}")

    mask = dv_hist > 0
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.stem(t_h[mask], dv_hist[mask]*1e3,
            linefmt="crimson", markerfmt="ro", basefmt=" ")
    ax.set_xlabel("Time [h]");  ax.set_ylabel("|dV| [mm/s]")
    ax.set_title(f"Burn history — {n_burns} burns, "
                 f"total dV = {total_dv*1e3:.2f} mm/s")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    p = os.path.join(figure_dir, "sk_burns.png")
    fig.savefig(p, dpi=150);  plt.close(fig);  print(f"Figure saved: {p}")

    return times_s, sep, sep_ntw, total_dv, n_burns


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_simulation(
        a_km             = 6778.0,
        inc_deg          = 51.6,
        t_span_h         = 24.0,
        check_interval_s = 300.0,
        deadband_m       = 100.0,
        a_max_mps2       = 5e-5,
        sc_mass_kg       = 100.0,
        sc_area_m2       = 1.0,
        sc_Cd            = 2.2,
        sc_Cr            = 1.3,
        dr0_m            = np.array([50.0, 0.0, 0.0]),
        dv0_mps          = np.array([0.0,  0.0, 0.0]),
        rk4_h_s          = 30.0,
        verbose          = True,
    )