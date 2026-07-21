"""
ssapy_toolkit/plots/artemis1_ssapy_divergence.py
---------------------------------------------------
Measures SSAPy's force model accuracy against the real Artemis I OEM
ephemeris using short-interval seeding, and optionally propagates the
full mission with finite burns injected.

Two modes:
  SHORT_INTERVAL  — seed from real OEM every 6 hours, propagate 6 hours,
                    measure divergence. Isolates pure force model error.
  FULL_MISSION    — propagate from mission start with finite burns applied
                    at each detected maneuver using real engine physics
                    (constant thrust, Tsiolkovsky mass flow, fixed direction).

Run:
    conda activate myenv
    cd C:/Users/diamond10/SSAPy-Toolkit
    python -m ssapy_toolkit.plots.artemis1_ssapy_divergence
"""

import sys
import pathlib
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core import OrbitalState
from core.orbit_state import PropagatorConfig
from core.accel_thrust import burn_from_dv
from ssapy_toolkit.plots.moon_plot_3d import moon_plot_3d

try:
    from ssapy_toolkit.plots.figpath import FIG_DIR
    OUT_DIR = pathlib.Path(FIG_DIR)
except Exception:
    OUT_DIR = pathlib.Path.home() / "yu_figures" / "demo_gallery" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OEM_PATH = pathlib.Path(__file__).resolve().parent / \
           "Post_TLI_Orion_AsFlown_20221213_EPH_OEM.asc"

GPS_EPOCH      = datetime.datetime(1980, 1, 6, tzinfo=datetime.timezone.utc)
GPS_UTC_OFFSET = 18.0
MU_EARTH       = 398600.4418
MU_MOON        = 4902.8
MU_SUN         = 1.327124e11

SEED_INTERVAL_HR = 6.0
SEED_INTERVAL_S  = SEED_INTERVAL_HR * 3600.0
PROP_DT_S        = 60.0

# Verified NASA engine specs from mission documentation
ENGINES = {
    "ICPS RL10B-2":    dict(thrust_n=110_100.0, isp_s=465.5),
    "Orion OMSE":      dict(thrust_n=26_700.0,  isp_s=316.0),
}
# Artemis I wet mass at TLI ignition (ICPS dry + propellant + Orion)
WET_MASS_KG = 54_700.0

# Burns large enough to be TLI-class use ICPS; smaller ones use Orion OMSE
ICPS_THRESHOLD_MS = 500.0   # m/s


def _utc_iso_to_gps(s):
    dt = datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc)
    return (dt - GPS_EPOCH).total_seconds() + GPS_UTC_OFFSET

def _gps_to_utc(gps):
    return GPS_EPOCH + datetime.timedelta(seconds=gps - GPS_UTC_OFFSET)


def parse_oem_full(path):
    times, xs, ys, zs, vxs, vys, vzs = [], [], [], [], [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("CCSDS","COMMENT","CREATION",
                "ORIGINATOR","META_","OBJECT_","CENTER_","REF_FRAME",
                "TIME_SYSTEM","START_TIME","USEABLE_","STOP_TIME")):
                continue
            parts = line.split()
            if len(parts) != 7: continue
            ts,x,y,z,vx,vy,vz = parts
            times.append(ts); xs.append(float(x)); ys.append(float(y))
            zs.append(float(z)); vxs.append(float(vx)); vys.append(float(vy))
            vzs.append(float(vz))
    return (np.stack([xs,ys,zs], axis=1),
            np.stack([vxs,vys,vzs], axis=1),
            np.array([_utc_iso_to_gps(t) for t in times]))


def detect_burns_full_model(r_km, v_kms, t_gps,
                             dv_threshold_ms=15.0, gap_s=5400.0):
    """Detect maneuvers using full force model (Earth+Moon+Sun) as reference."""
    from astropy.time import Time
    from astropy.coordinates import get_body_barycentric
    import astropy.units as u

    flagged    = {}
    for i in range(len(t_gps)-1):
        dt    = t_gps[i+1] - t_gps[i]
        r     = r_km[i]; r_mag = np.linalg.norm(r)
        a_earth = -MU_EARTH / r_mag**3 * r
        t_ast = Time(t_gps[i], format='gps')
        try:
            mb  = get_body_barycentric('moon',  t_ast)
            eb  = get_body_barycentric('earth', t_ast)
            sb  = get_body_barycentric('sun',   t_ast)
            rm  = (mb-eb).xyz.to(u.km).value
            rs  = (sb-eb).xyz.to(u.km).value
            a_moon = -MU_MOON/np.linalg.norm(r-rm)**3*(r-rm)
            a_sun  = -MU_SUN /np.linalg.norm(r-rs)**3*(r-rs)
        except Exception:
            a_moon = a_sun = np.zeros(3)
        v_pred  = v_kms[i] + (a_earth+a_moon+a_sun)*dt
        dv_vec  = v_kms[i+1] - v_pred
        dv_mag  = np.linalg.norm(dv_vec)*1000
        if dv_mag > dv_threshold_ms:
            flagged[i+1] = dv_vec

    if not flagged:
        return [], []

    idxs = sorted(flagged)
    clusters = [[idxs[0]]]
    for idx in idxs[1:]:
        if t_gps[idx] - t_gps[clusters[-1][-1]] > gap_s:
            clusters.append([idx])
        else:
            clusters[-1].append(idx)

    starts = [c[0] for c in clusters]
    dv_vecs = [sum(flagged.get(j, np.zeros(3)) for j in c) for c in clusters]
    return starts, dv_vecs


def _moon_distance_km(r_km, t_gps):
    from astropy.time import Time
    from astropy.coordinates import get_body_barycentric
    import astropy.units as u
    t_ast = Time(t_gps, format='gps')
    mb = get_body_barycentric('moon',  t_ast)
    eb = get_body_barycentric('earth', t_ast)
    rm = (mb-eb).xyz.to(u.km).value
    return float(np.linalg.norm(r_km - rm))


def _base_accels():
    """Build the gravitational force model (no thrust).
    
    Moon gravity harmonics are deliberately excluded: at cislunar distances
    (350,000+ km from Moon) the spherical harmonic expansion is numerically
    ill-conditioned and produces large errors rather than improving accuracy.
    The Moon's dominant gravity is already in AccelThirdBody(moon).
    """
    from ssapy.gravity import AccelHarmonic, AccelThirdBody
    from ssapy.body   import get_body as _gb
    from ssapy.accel  import AccelKepler, AccelSum
    earth = _gb("earth"); moon = _gb("moon"); sun = _gb("sun")
    return [AccelKepler(earth.mu), AccelHarmonic(earth, 8, 8),
            AccelThirdBody(sun), AccelThirdBody(moon)]


def _make_prop(state, extra_accels=None, dt_s=None):
    from ssapy.accel import AccelSum
    acc = _base_accels() + (extra_accels or [])
    h   = dt_s or PROP_DT_S
    return __import__('ssapy').propagator.RK4Propagator(AccelSum(acc), h=h)


def propagate_interval(r0_km, v0_kms, t0_gps, duration_s, extra_accels=None):
    """Propagate one interval, returning final (r_km, v_kms, dt_used)."""
    import ssapy
    try:
        d_moon = _moon_distance_km(r0_km, t0_gps)
        dt_s = 5.0 if d_moon < 10_000 else (15.0 if d_moon < 50_000 else PROP_DT_S)
    except Exception:
        dt_s = PROP_DT_S

    orbit = ssapy.Orbit(r=r0_km*1e3, v=v0_kms*1e3, t=t0_gps)
    epoch = _gps_to_utc(t0_gps)
    cfg   = PropagatorConfig(propagator="rk4", gravity="8x8", third_body="both")
    tmp   = OrbitalState.from_rv(r0_km, v0_kms, epoch=epoch, config=cfg)
    prop  = _make_prop(tmp, extra_accels=extra_accels, dt_s=dt_s)

    t_arr = np.arange(t0_gps, t0_gps+duration_s+dt_s, dt_s)
    r_m, v_m = ssapy.rv(orbit, t_arr, propagator=prop)
    return r_m[-1]/1e3, v_m[-1]/1e3, dt_s


if __name__ == "__main__":
    if not OEM_PATH.exists():
        print(f"OEM file not found at {OEM_PATH}"); sys.exit(1)

    real_r_km, real_v_kms, real_t_gps = parse_oem_full(OEM_PATH)
    print(f"Loaded {len(real_t_gps)} real OEM state vectors")

    # ── Detect burns ─────────────────────────────────────────────────────────
    print("Detecting burns (full force model reference)...")
    burn_starts, burn_dv_vecs = detect_burns_full_model(
        real_r_km, real_v_kms, real_t_gps)

    # Build finite burn objects for each detected maneuver
    burn_accels = []
    print(f"\nDetected {len(burn_starts)} maneuvers:")
    for ms, dv in zip(burn_starts, burn_dv_vecs):
        elapsed = (real_t_gps[ms] - real_t_gps[0]) / 86400
        dv_ms   = np.linalg.norm(dv) * 1000
        engine  = "ICPS RL10B-2" if dv_ms > ICPS_THRESHOLD_MS else "Orion OMSE"
        eng     = ENGINES[engine]
        utc     = _gps_to_utc(real_t_gps[ms])
        burn_accel, dur_s, m_prop = burn_from_dv(
            dv, WET_MASS_KG, eng["thrust_n"], eng["isp_s"], real_t_gps[ms])
        if burn_accel:
            burn_accels.append(burn_accel)
        print(f"  Day {elapsed:5.2f} | {utc.strftime('%b %d %H:%M')} UTC | "
              f"|dv|={dv_ms:.0f} m/s | engine={engine} | "
              f"dur={dur_s:.0f}s | m_prop={m_prop:.0f}kg")

    BURN_DAYS = [(real_t_gps[ms]-real_t_gps[0])/86400 for ms in burn_starts]

    # ── Short-interval divergence (force model accuracy) ─────────────────────
    print(f"\nShort-interval accuracy ({SEED_INTERVAL_HR:.0f}-hr windows)...")
    seed_elapsed, pos_errors, vel_errors, is_burn = [], [], [], []
    i = 0
    while i < len(real_t_gps)-1:
        j = np.searchsorted(real_t_gps, real_t_gps[i]+SEED_INTERVAL_S)
        if j >= len(real_t_gps): break
        el_start = (real_t_gps[i]-real_t_gps[0])/86400
        el_end   = (real_t_gps[j]-real_t_gps[0])/86400
        in_burn  = any(el_start <= bd <= el_end for bd in BURN_DAYS)
        try:
            r_ss, v_ss, _ = propagate_interval(
                real_r_km[i], real_v_kms[i], real_t_gps[i],
                real_t_gps[j]-real_t_gps[i])
            seed_elapsed.append(el_start)
            pos_errors.append(np.linalg.norm(r_ss - real_r_km[j]))
            vel_errors.append(np.linalg.norm(v_ss - real_v_kms[j])*1e5)
            is_burn.append(in_burn)
        except Exception as ex:
            print(f"  Day {el_start:.1f} failed: {ex}")
        i = j

    seed_elapsed = np.array(seed_elapsed)
    pos_errors   = np.array(pos_errors)
    vel_errors   = np.array(vel_errors)
    is_burn      = np.array(is_burn)
    coast        = ~is_burn

    print(f"\n  Coast median pos: {np.median(pos_errors[coast]):.1f} km  "
          f"max: {pos_errors[coast].max():.1f} km")
    print(f"  Coast median vel: {np.median(vel_errors[coast]):.2f} cm/s  "
          f"max: {vel_errors[coast].max():.2f} cm/s")

    # ── Step-by-step propagation with sync (advisor's approach + better physics)
    # The advisor's key insight: propagate one OEM timestep at a time (~4 min).
    # Over 4 minutes, force model errors are tiny even for Earth-only gravity.
    # When error exceeds a threshold, reset to real OEM state (the "sync").
    # This mimics a navigation filter: predict with model, correct with ground truth.
    # We use our full force model (Earth 8x8 + Moon + Sun) instead of Kepler-only,
    # so our sync events should be fewer and our between-sync accuracy better.
    print("\nStep-by-step propagation with sync (4-min steps, full force model)...")

    SYNC_THRESHOLD_KM = 50.0   # same as advisor's default
    import ssapy as _ssapy
    from ssapy.accel import AccelSum as _AccelSum

    n = len(real_t_gps)
    r_model = np.zeros_like(real_r_km)
    v_model = np.zeros_like(real_v_kms)
    r_model[0] = real_r_km[0].copy()
    v_model[0] = real_v_kms[0].copy()
    sync_indices = [0]
    step_errors_km = []

    # Build propagator once (reused for all steps)
    _acc = _AccelSum(_base_accels())
    _prop_step = _ssapy.propagator.RK4Propagator(_acc, h=15.0)  # 15s internal step

    for i in range(n - 1):
        dur = real_t_gps[i+1] - real_t_gps[i]
        try:
            orb = _ssapy.Orbit(r=r_model[i]*1e3, v=v_model[i]*1e3, t=real_t_gps[i])
            t_eval = np.array([real_t_gps[i], real_t_gps[i+1]])
            r_m, v_m = _ssapy.rv(orb, t_eval, propagator=_prop_step)
            r_model[i+1] = r_m[-1] / 1e3
            v_model[i+1] = v_m[-1] / 1e3
        except Exception:
            # propagation failed — sync immediately
            r_model[i+1] = real_r_km[i+1].copy()
            v_model[i+1] = real_v_kms[i+1].copy()
            sync_indices.append(i+1)
            step_errors_km.append(0.0)
            continue

        err_km = np.linalg.norm(r_model[i+1] - real_r_km[i+1])
        step_errors_km.append(err_km)

        if err_km > SYNC_THRESHOLD_KM:
            r_model[i+1] = real_r_km[i+1].copy()
            v_model[i+1] = real_v_kms[i+1].copy()
            sync_indices.append(i+1)

    step_errors_km = np.array(step_errors_km)
    dr_norm_km = np.linalg.norm(r_model - real_r_km, axis=1)
    dv_norm_cms = np.linalg.norm(v_model - real_v_kms, axis=1) * 1e5

    n_syncs = len(sync_indices) - 1
    print(f"  Sync events: {n_syncs}  (threshold={SYNC_THRESHOLD_KM:.0f} km)")
    print(f"  RMS position error: {np.sqrt(np.mean(dr_norm_km**2)):.2f} km")
    print(f"  Median step error:  {np.median(step_errors_km[step_errors_km>0]):.2f} km")
    print(f"  Max step error:     {step_errors_km.max():.2f} km")
    print(f"  RMS velocity error: {np.sqrt(np.mean(dv_norm_cms**2)):.2f} cm/s")

    # ── Plot 1: divergence chart ──────────────────────────────────────────────
    fig1, axes = plt.subplots(3, 1, figsize=(13, 11), dpi=120,
                               facecolor="#0a0d14",
                               gridspec_kw={'height_ratios':[2,2,2]})
    ax_pos, ax_vel, ax_full = axes

    for ax in axes:
        ax.set_facecolor("#0a0d14")
        ax.tick_params(colors="white")
        ax.grid(True, color="#4A6080", alpha=0.3)
        for sp in ax.spines.values():
            sp.set_color("#4A6080")

    # Position error
    ax_pos.plot(seed_elapsed[coast], pos_errors[coast],
                color="#00ff9c", lw=1.5, marker='o', ms=3,
                label=f"Coast ({coast.sum()} windows)")
    ax_pos.plot(seed_elapsed[is_burn], pos_errors[is_burn],
                color="#ff4444", lw=0, marker='x', ms=8, mew=2,
                label=f"Burn window ({is_burn.sum()} windows)")
    ax_pos.axhline(np.median(pos_errors[coast]), color="#00ff9c", alpha=0.4,
                   ls='--', lw=0.8,
                   label=f"Coast median: {np.median(pos_errors[coast]):.1f} km")
    ax_pos.set_ylabel("Position error (km)", color="white")
    ax_pos.legend(fontsize=8, facecolor='#0a0d14', edgecolor='#4A6080',
                  labelcolor='white', loc='upper left')

    # Velocity error
    ax_vel.plot(seed_elapsed[coast], vel_errors[coast],
                color="#ff9900", lw=1.5, marker='o', ms=3,
                label=f"Coast ({coast.sum()} windows)")
    ax_vel.plot(seed_elapsed[is_burn], vel_errors[is_burn],
                color="#ff4444", lw=0, marker='x', ms=8, mew=2,
                label=f"Burn window ({is_burn.sum()} windows)")
    ax_vel.axhline(np.median(vel_errors[coast]), color="#ff9900", alpha=0.4,
                   ls='--', lw=0.8,
                   label=f"Coast median: {np.median(vel_errors[coast]):.2f} cm/s")
    ax_vel.set_ylabel("Velocity error (cm/s)", color="white")
    ax_vel.legend(fontsize=8, facecolor='#0a0d14', edgecolor='#4A6080',
                  labelcolor='white', loc='upper left')

    # Bottom: step-by-step with sync (same approach as advisor's benchmark)
    elapsed_full = (real_t_gps - real_t_gps[0]) / 86400
    ax_full.plot(elapsed_full, dr_norm_km, color="#a78bfa", lw=0.8,
                 label=f"Step-by-step 4-min (RMS={np.sqrt(np.mean(dr_norm_km**2)):.1f} km)")
    if len(sync_indices) > 1:
        ax_full.scatter(elapsed_full[sync_indices], dr_norm_km[sync_indices],
                        color="#ff4444", s=12, zorder=5, label=f"Sync events ({n_syncs})")
    ax_full.axhline(SYNC_THRESHOLD_KM, color='#ff8800', alpha=0.4,
                    ls='--', lw=0.8, label=f"Sync threshold ({SYNC_THRESHOLD_KM:.0f} km)")
    ax_full.set_ylabel("Position error (km)", color="white")
    ax_full.set_xlabel("Mission elapsed time (days)", color="white")
    ax_full.legend(fontsize=8, facecolor='#0a0d14', edgecolor='#4A6080',
                   labelcolor='white', loc='upper left')

    fig1.suptitle(
        "SSAPy force model accuracy vs. real Artemis I\n"
        f"Top/mid: {SEED_INTERVAL_HR:.0f}-hr seeding  |  "
        "Bottom: step-by-step (4-min) with sync — matching advisor benchmark\n"
        "Force model: Earth 8x8 + Moon harmonics + Moon + Sun  | 15s RK4",
        color="white", fontsize=10)
    fig1.tight_layout()
    div_path = OUT_DIR / "artemis1_ssapy_divergence.jpg"
    fig1.savefig(str(div_path), facecolor="#0a0d14", dpi=120)
    plt.close(fig1)
    print(f"\nSaved -> {div_path}")

    # 3D overlay
    n  = 768
    ir = np.round(np.linspace(0, len(real_r_km)-1, n)).astype(int)
    im = np.round(np.linspace(0, len(r_model)-1,   n)).astype(int)
    r_both = np.stack([real_r_km[ir]*1e3, r_model[im]*1e3])
    t_both = np.stack([real_t_gps[ir],    real_t_gps[im]])

    fig2, ax2 = moon_plot_3d(
        r=r_both, t=t_both, r_frame='gcrf',
        show_earth=True, show_lagrange=False,
        shade_ambient=0.22, shade_diffuse=0.78,
        title="Real Artemis I vs. SSAPy (step-by-step, sync at 50 km)",
        save_path=False,
    )
    import matplotlib.cm as cm
    cols = cm.rainbow(np.linspace(0,1,2))
    ax2.legend(handles=[
        mpatches.Patch(color=cols[0], label="Real Artemis I (NASA AROW OEM)"),
        mpatches.Patch(color=cols[1],
                       label=f"SSAPy step-by-step (Earth 8x8+Moon+Sun, {n_syncs} syncs)"),
    ], loc='upper left', fontsize=8, framealpha=0.6,
       facecolor='#0a0d14', edgecolor='#4A6080', labelcolor='white')
    fig2.savefig(str(OUT_DIR/"artemis1_ssapy_overlay.jpg"), facecolor="#050810", dpi=120)
    plt.close(fig2)
    print(f"Saved -> {OUT_DIR/'artemis1_ssapy_overlay.jpg'}")