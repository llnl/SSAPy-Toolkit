#!/usr/bin/env python3
"""
demos/test_station_keeping_ntw.py

Standalone demo/test harness for a future yeager_utils.station_keeping().

Adds progress/timing prints around SSAPy propagation so you can see whether
best_gravity_prop() is just slow (it is a heavy model: EGM2008 140x140 + Moon harm
+ Sun + planets + SRP + EarthRad + Drag) [187].
"""

import time as _time
import numpy as np
import matplotlib.pyplot as plt

from ssapy import Orbit, rv
from yeager_utils import Time, RGEO
from yeager_utils.SSAPy_wrappers import keplerian_prop, best_gravity_prop  # [187]
from yeager_utils.Plots.plotutils import yufig  # [165]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _norm(x, axis=None):
    return np.linalg.norm(x, axis=axis)


def _as_1d_float_time_grid(times):
    try:
        return np.asarray(times.gps, dtype=float)
    except Exception:
        return np.asarray(times, dtype=float)


def _ntw_frame_from_rv(r, v):
    r = np.asarray(r, float).reshape(3)
    v = np.asarray(v, float).reshape(3)

    vmag = _norm(v)
    h = np.cross(r, v)
    hmag = _norm(h)

    if vmag == 0.0 or hmag == 0.0:
        return (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )

    e_T = v / vmag
    e_W = h / hmag
    e_N = np.cross(e_W, e_T)
    nmag = _norm(e_N)
    if nmag == 0.0:
        return np.array([1.0, 0.0, 0.0]), e_T, e_W

    e_N = e_N / nmag
    return e_N, e_T, e_W


def _timed(name, fn, *args, **kwargs):
    t0 = _time.time()
    print(f"[timing] start {name} ...", flush=True)
    out = fn(*args, **kwargs)
    dt = _time.time() - t0
    print(f"[timing] done  {name} in {dt:.3f} s", flush=True)
    return out, dt


def _rv_chunked_progress(*, orbit, times, propagator, label="rv", chunk_size=200, heartbeat_s=10.0):
    """
    Call ssapy.rv(...) in chunks so we can print progress during very slow propagations.

    Assumes `times` is an array-like of absolute times accepted by ssapy.rv.
    Returns full (r, v) stacked over all chunks.
    """
    times = np.asarray(times)
    N = times.shape[0]
    if N <= chunk_size:
        (out, _dt) = _timed(label, rv, orbit=orbit, time=times, propagator=propagator)
        return out[0], out[1]

    r_out = np.empty((N, 3), float)
    v_out = np.empty((N, 3), float)

    t0 = _time.time()
    last_hb = t0
    print(f"[progress] {label}: N={N}, chunk_size={chunk_size}", flush=True)

    i0 = 0
    while i0 < N:
        i1 = min(i0 + chunk_size, N)
        # propagate this chunk
        r_blk, v_blk = rv(orbit=orbit, time=times[i0:i1], propagator=propagator)
        r_out[i0:i1] = r_blk
        v_out[i0:i1] = v_blk

        now = _time.time()
        if (now - last_hb) >= heartbeat_s or i1 == N:
            frac = i1 / N
            elapsed = now - t0
            rate = i1 / max(elapsed, 1e-12)
            eta = (N - i1) / max(rate, 1e-12)
            print(
                f"[progress] {label}: {i1}/{N} ({100*frac:.1f}%) "
                f"elapsed={elapsed:.1f}s rate={rate:.2f} pts/s eta~{eta:.1f}s",
                flush=True,
            )
            last_hb = now

        i0 = i1

    total = _time.time() - t0
    print(f"[timing] done  {label} total {total:.3f} s", flush=True)
    return r_out, v_out


# -----------------------------------------------------------------------------
# Core: station keeping (demo version)
# -----------------------------------------------------------------------------
def station_keeping(
    r_ref,
    v_ref,
    t_ref,
    r_truth,
    v_truth,
    t_truth,
    *,
    mode="distance",
    dist_thresh_m=10e3,
    update_period_s=600.0,
    kp=1.0e-4,
    kd=2.0e-4,
    a_max_mps2=5e-5,
    use_truth_frame=True,
):
    r_ref = np.asarray(r_ref, float)
    v_ref = np.asarray(v_ref, float)
    r_truth = np.asarray(r_truth, float)
    v_truth = np.asarray(v_truth, float)

    t_ref_s = _as_1d_float_time_grid(t_ref)
    t_truth_s = _as_1d_float_time_grid(t_truth)

    if r_ref.shape != r_truth.shape or v_ref.shape != v_truth.shape:
        raise ValueError("Reference and truth states must have matching shapes for this demo.")
    if t_ref_s.shape != t_truth_s.shape or not np.allclose(t_ref_s, t_truth_s):
        raise ValueError("Reference and truth time grids must match for this demo.")

    N = r_truth.shape[0]
    a_gcrf_cmd = np.zeros((N, 3), float)
    a_ntw_cmd = np.zeros((N, 3), float)

    events = []
    last_update_t = float(t_truth_s[0])

    def _should_update(i):
        nonlocal last_update_t
        if i == 0:
            return True

        dt = float(t_truth_s[i] - last_update_t)
        dist = float(_norm(r_truth[i] - r_ref[i]))

        trig_dist = dist >= float(dist_thresh_m)
        trig_time = dt >= float(update_period_s)

        if mode == "distance":
            return trig_dist
        if mode == "time":
            return trig_time
        if mode == "both":
            return trig_dist or trig_time
        raise ValueError("mode must be 'distance', 'time', or 'both'")

    for i in range(N):
        if _should_update(i):
            dist = float(_norm(r_truth[i] - r_ref[i]))
            events.append(
                dict(
                    idx=i,
                    t=float(t_truth_s[i]),
                    dist_m=dist,
                    dt_since_last=float(t_truth_s[i] - last_update_t),
                )
            )
            last_update_t = float(t_truth_s[i])

        dr = (r_ref[i] - r_truth[i])
        dv = (v_ref[i] - v_truth[i])

        # FIXED: correct multiplication
        a = float(kp) * dr + float(kd) * dv

        amag = float(_norm(a))
        if amag > float(a_max_mps2) and amag > 0.0:
            a = (float(a_max_mps2) / amag) * a

        a_gcrf_cmd[i] = a

        if use_truth_frame:
            n_hat, t_hat, w_hat = _ntw_frame_from_rv(r_truth[i], v_truth[i])
        else:
            n_hat, t_hat, w_hat = _ntw_frame_from_rv(r_ref[i], v_ref[i])

        a_ntw_cmd[i, 0] = float(np.dot(n_hat, a))
        a_ntw_cmd[i, 1] = float(np.dot(t_hat, a))
        a_ntw_cmd[i, 2] = float(np.dot(w_hat, a))

    return {"a_ntw_cmd": a_ntw_cmd, "a_gcrf_cmd": a_gcrf_cmd, "events": events}


# -----------------------------------------------------------------------------
# Demo runner
# -----------------------------------------------------------------------------
def main():
    print("=== test_station_keeping_ntw.py ===", flush=True)
    print("Plots will be saved via yufig(...) under ~/yu_figures/tests/ [153][165].", flush=True)

    # Ellipse from r_p ~ 1*RGEO to r_a ~ 4*RGEO
    rp = 1.0 * RGEO
    ra = 4.0 * RGEO
    a = 0.5 * (rp + ra)
    e = (ra - rp) / (ra + rp)

    inc = np.deg2rad(15.0)
    raan = np.deg2rad(20.0)
    pa = np.deg2rad(40.0)
    ta0 = np.deg2rad(0.0)

    t0 = Time("2026-01-01T00:00:00", scale="utc")

    orbit0 = Orbit.fromKeplerianElements(a, e, inc, pa, raan, ta0, t=t0)
    period = float(orbit0.period)

    dt = 60.0
    times = t0 + np.arange(0.0, period + dt, dt)
    print(f"[info] samples={len(times)} dt={dt:.1f}s period={period/3600:.2f} hr", flush=True)

    prop_kep = keplerian_prop()         # [187]
    prop_best = best_gravity_prop()     # [187]

    # Reference propagation (fast)
    (out_ref, _dt_ref) = _timed("rv keplerian_prop", rv, orbit=orbit0, time=times, propagator=prop_kep)
    r_ref, v_ref = out_ref

    # Truth propagation (may be slow) -> run chunked with progress
    r_truth, v_truth = _rv_chunked_progress(
        orbit=orbit0,
        times=times,
        propagator=prop_best,
        label="rv best_gravity_prop (chunked)",
        chunk_size=200,
        heartbeat_s=10.0,
    )

    # Compute station-keeping command (not applied, just computed)
    out = station_keeping(
        r_ref=r_ref, v_ref=v_ref, t_ref=times,
        r_truth=r_truth, v_truth=v_truth, t_truth=times,
        mode="both",
        dist_thresh_m=25e3,
        update_period_s=900.0,
        kp=1.0e-4,
        kd=2.0e-4,
        a_max_mps2=2e-5,
        use_truth_frame=True,
    )

    a_ntw = out["a_ntw_cmd"]
    events = out["events"]

    pos_err = _norm(r_truth - r_ref, axis=1)
    a_mag = _norm(a_ntw, axis=1)
    t_sec = times.gps - times.gps[0]

    print(f"Max |truth-ref|:  {float(np.max(pos_err)):.3e} m", flush=True)
    print(f"Mean |truth-ref|: {float(np.mean(pos_err)):.3e} m", flush=True)
    print(f"Max |a_ntw|:      {float(np.max(a_mag)):.3e} m/s^2", flush=True)
    print(f"Num update events: {len(events)}", flush=True)

    # ---------------------------
    # ALWAYS SAVE PLOTS
    # ---------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_sec / 3600.0, pos_err, lw=2)
    ax.set_xlabel("t since start [hr]")
    ax.set_ylabel("|r_truth - r_ref| [m]")
    ax.set_title("High-fidelity vs Kepler divergence (no control applied)")
    ax.grid(True)
    fig.tight_layout()
    yufig(fig, "tests/station_keeping_divergence.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_sec / 3600.0, a_mag, lw=2)
    ax.set_xlabel("t since start [hr]")
    ax.set_ylabel("|a_ntw_cmd| [m/s^2]")
    ax.set_title("Station-keeping command magnitude (computed)")
    ax.grid(True)
    fig.tight_layout()
    yufig(fig, "tests/station_keeping_a_ntw_mag.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_sec / 3600.0, pos_err, lw=2, label="|truth-ref|")
    if events:
        ev_t = np.array([ev["t"] for ev in events], float)
        ev_t_hr = (ev_t - ev_t[0]) / 3600.0
        ev_d = np.array([ev["dist_m"] for ev in events], float)
        ax.scatter(ev_t_hr, ev_d, s=20, label="update events")
    ax.set_xlabel("t since start [hr]")
    ax.set_ylabel("distance [m]")
    ax.set_title("Update events vs position error")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    yufig(fig, "tests/station_keeping_update_events.png")
    plt.close(fig)

    print("Saved plots under ~/yu_figures/tests/ via yufig [153][165].", flush=True)


if __name__ == "__main__":
    main()