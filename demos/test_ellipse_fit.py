#!/usr/bin/env python3
"""
demos/test_ellipse_fit.py

Baseline:
  - Runs ellipse_fit twice (A/B) using v_pref to pick direction
  - Validates shapes, finiteness, required keys
  - Checks endpoints: r[0]≈P1, r[-1]≈P2
  - Checks invariants from returned r,v:
      |h| const, specific energy const, |e| const
  - Reconstructs with SSAPy (SV + Kepler) and checks internal consistency
  - Forces SSAPy propagation with KeplerianPropagator

Inclination sweep (NO inc kwarg):
  - Start: equatorial GEO position at +X, radius = 1*RGEO
  - End: on a sphere at radius = 2*RGEO with latitude = inc (0..90 step 15)
  - For each inc, try several candidate longitudes until ellipse_fit converges
  - Computes A/B arcs per inc
  - Compares each run vs SSAPy KeplerianPropagator USING STATE VECTORS (r0,v0)
    to avoid equatorial-element singularities at inc=0.

Plots saved:
  1) tests/testing_ellipse_fit_incsweep_all_transfers.png
     - All transfer orbits (A and B for all inclinations)
  2) tests/testing_ellipse_fit_compared_ssapy.png
     - |r_fit - r_ssapy(Kepler)| vs time for ALL inclination runs (A and B)
       with a COMPLETE legend (all curves) and DISTINCT line styles.
       (No transparency.)

Run:
  python3 demos/test_ellipse_fit.py
"""

from ssapy_toolkit import RGEO, ellipse_fit, ssapy_orbit, yufig
import numpy as np

from ssapy.propagator import KeplerianPropagator


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _norm(x, axis=None):
    return np.linalg.norm(x, axis=axis)


def _unit(x, eps=1e-15):
    x = np.asarray(x, float)
    n = float(np.linalg.norm(x))
    if n < eps:
        return None
    return x / n


def _finite(x, name):
    x = np.asarray(x)
    if not np.all(np.isfinite(x)):
        bad = np.where(~np.isfinite(x))
        raise AssertionError(f"Non-finite values in {name} at indices {bad}")


def _assert_keys(d, keys):
    missing = [k for k in keys if k not in d]
    if missing:
        raise AssertionError(f"Missing keys from ellipse_fit result: {missing}")


def _assert_close(a, b, atol, msg):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    err = float(np.linalg.norm(a - b))
    if err > float(atol):
        raise AssertionError(f"{msg} |err|={err:.6e} > atol={float(atol):.6e}")
    return err


def _rel_std(x, eps=1e-30):
    x = np.asarray(x, float)
    return float(np.std(x) / max(float(np.mean(np.abs(x))), eps))


def _invariants_from_rv(mu, r_m, v_m_s):
    r_m = np.asarray(r_m, float)
    v_m_s = np.asarray(v_m_s, float)

    h_vec = np.cross(r_m, v_m_s)
    h_mag = _norm(h_vec, axis=1)

    rmag = _norm(r_m, axis=1)
    vmag = _norm(v_m_s, axis=1)

    energy = 0.5 * (vmag**2) - float(mu) / rmag

    e_vec = (np.cross(v_m_s, h_vec) / float(mu)) - (r_m / rmag[:, None])
    e_mag = _norm(e_vec, axis=1)

    return {
        "h_rel_std": _rel_std(h_mag),
        "E_rel_std": _rel_std(energy),
        "e_rel_std": _rel_std(e_mag),
    }


def _cos_sim(a, b, eps=1e-15):
    ah = _unit(a, eps=eps)
    bh = _unit(b, eps=eps)
    if ah is None or bh is None:
        return -np.inf
    return float(np.dot(ah, bh))


def _compute_t_hat(P1_m, P2_m):
    """
    Approximate tangential direction at P1 using:
      w_hat ~ unit(P1 x P2)
      t_hat = unit(w_hat x r_hat) at P1
    Robust fallback if nearly colinear.
    """
    P1_m = np.asarray(P1_m, float)
    P2_m = np.asarray(P2_m, float)

    r_hat = _unit(P1_m)
    if r_hat is None:
        raise AssertionError("P1 has near-zero norm; cannot form tangential direction.")

    w_hat = _unit(np.cross(P1_m, P2_m))
    if w_hat is None:
        cand = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(r_hat, cand))) > 0.99:
            cand = np.array([1.0, 0.0, 0.0])
        w_hat = _unit(np.cross(r_hat, cand))
        if w_hat is None:
            raise AssertionError("Could not form stable plane normal for tangential direction.")

    t_hat = _unit(np.cross(w_hat, r_hat))
    if t_hat is None:
        raise AssertionError("Could not form tangential unit vector at P1.")
    return t_hat


def _p2_on_sphere_lat_lon(R, lat_rad, lon_rad):
    """
    Spherical coords:
      lat = angle above equatorial plane (0..+pi/2),
      lon in xy-plane from +x toward +y.
    """
    cl = float(np.cos(lat_rad))
    sl = float(np.sin(lat_rad))
    co = float(np.cos(lon_rad))
    so = float(np.sin(lon_rad))
    return np.array([R * cl * co, R * cl * so, R * sl], dtype=float)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
REQUIRED_KEYS = [
    "r", "v", "t_rel", "t_abs",
    "a", "e", "i", "raan", "pa", "ta", "L",
    "rp", "ra", "rp_alt", "ra_alt", "b", "p",
    "mean_motion", "eta", "period",
    "h_vec", "h", "Energy", "e_vec", "n_vec",
    "r0", "v0", "F2",
    "plane_basis", "rot_dir",
    "mu",
]

ENDPOINT_ATOL_M = 5.0e-3
TIME_MONO_TOL_S = 1.0e-9

INV_REL_STD_H = 2.0e-9
INV_REL_STD_E = 2.0e-9
INV_REL_STD_e = 2.0e-9

SSAPY_INTERNAL_RMS_M = 1.0e-3
SSAPY_INTERNAL_MAX_M = 1.0e-2

DIR_COS_MIN = 0.90

SAVE_PLOTS = True

INC_SWEEP_DEG = np.arange(0.0, 90.0 + 1e-9, 15.0)
R1_SWEEP = 1.0 * RGEO
R2_SWEEP = 2.0 * RGEO

LON_TRY_DEG = [90.0, 75.0, 105.0, 60.0, 120.0, 45.0, 135.0, 30.0, 150.0]

KEP_PROP = KeplerianPropagator()


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def _plot_incsweep_all_transfers_3d(sweep_store, P1_m):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    labeledA = False
    labeledB = False

    for item in sweep_store:
        rA = np.asarray(item["A"]["r"], float)
        rB = np.asarray(item["B"]["r"], float)

        if not labeledA:
            ax.plot(rA[:, 0], rA[:, 1], rA[:, 2], linewidth=1.0, label="A (all inc)")
            labeledA = True
        else:
            ax.plot(rA[:, 0], rA[:, 1], rA[:, 2], linewidth=1.0)

        if not labeledB:
            ax.plot(rB[:, 0], rB[:, 1], rB[:, 2], linewidth=1.0, label="B (all inc)")
            labeledB = True
        else:
            ax.plot(rB[:, 0], rB[:, 1], rB[:, 2], linewidth=1.0)

    ax.scatter([P1_m[0]], [P1_m[1]], [P1_m[2]], s=50, label="P1 (start)")
    P2s = np.array([it["P2"] for it in sweep_store], float)
    ax.scatter(P2s[:, 0], P2s[:, 1], P2s[:, 2], s=22, label="P2 (ends)")

    ax.set_title("Inclination sweep transfers: 1*RGEO (+X) → 2*RGEO sphere\nlat = 0°..90° in 15° steps")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    fig.tight_layout()
    yufig(fig, "tests/testing_ellipse_fit_incsweep_all_transfers.png")
    plt.close(fig)


def _plot_incsweep_compared_ssapy_time_series(err_series, inc_values_deg):
    """
    Plot ALL runs with:
      - complete legend (one entry per line)
      - no transparency
      - distinct line styles (and different linewidths) so curves remain visible
    """
    import matplotlib.pyplot as plt

    # Line styles: use inclination to pick style; use A/B to pick linewidth.
    # (Keeps everything monochrome-friendly and doesn't require transparency.)
    style_cycle = ["-", "--", ":", "-."]

    # Deterministic mapping from inc -> style
    inc_list = [float(x) for x in inc_values_deg]
    inc_to_style = {}
    for i, inc in enumerate(inc_list):
        inc_to_style[inc] = style_cycle[i % len(style_cycle)]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Sort by dir then inc for a stable legend ordering
    def _key(item):
        d = item["dir"]
        inc = float(item["inc_deg"])
        return (0 if d == "A" else 1, inc)

    for item in sorted(err_series, key=_key):
        t = np.asarray(item["t"], float)
        e = np.asarray(item["err_m"], float)
        inc = float(item["inc_deg"])
        d = item["dir"]

        ls = inc_to_style.get(inc, "-")
        lw = 1.6 if d == "A" else 1.0

        label = f"inc={inc:.0f}° ({d})"
        ax.plot(t, e, linestyle=ls, linewidth=lw, label=label)

    ax.set_xlabel("t_rel [s]")
    ax.set_ylabel("|r_fit - r_ssapy(Kepler)| [m]")
    ax.set_title(
        "Inclination sweep: ellipse_fit vs SSAPy KeplerianPropagator (all runs)\n"
        "(SSAPy propagated from r0,v0)"
    )
    ax.grid(True)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    yufig(fig, "tests/testing_ellipse_fit_compared_ssapy.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Core runner (NO inc kwarg)
# -----------------------------------------------------------------------------
def _run_one_case(P1_m, P2_m, v_pref_m_s, tag, dir_cos_min=DIR_COS_MIN, compare_ssapy=True):
    res = ellipse_fit(
        P1_m,
        P2_m,
        n_pts=400,
        plot=False,
        v_pref_m_s=v_pref_m_s,
    )

    _assert_keys(res, REQUIRED_KEYS)

    r = np.asarray(res["r"], float)
    v = np.asarray(res["v"], float)
    t_rel = np.asarray(res["t_rel"], float)

    if r.ndim != 2 or r.shape[1] != 3:
        raise AssertionError(f"{tag}: r must be (N,3), got {r.shape}")
    if v.shape != r.shape:
        raise AssertionError(f"{tag}: v must match r shape, got {v.shape}")
    if t_rel.ndim != 1 or t_rel.shape[0] != r.shape[0]:
        raise AssertionError(f"{tag}: t_rel must be (N,), got {t_rel.shape}")

    _finite(r, f"r {tag}")
    _finite(v, f"v {tag}")
    _finite(t_rel, f"t_rel {tag}")

    if np.any(np.diff(t_rel) < -TIME_MONO_TOL_S):
        raise AssertionError(f"{tag}: t_rel is not non-decreasing.")

    d0 = _assert_close(r[0], P1_m, ENDPOINT_ATOL_M, f"{tag}: r[0] != P1")
    d1 = _assert_close(r[-1], P2_m, ENDPOINT_ATOL_M, f"{tag}: r[-1] != P2")
    print(f"{tag}: endpoints |r[0]-P1|={d0:.3e} m, |r[-1]-P2|={d1:.3e} m")

    inv = _invariants_from_rv(float(res["mu"]), r, v)
    print(
        f"{tag}: invariants "
        f"h_rel_std={inv['h_rel_std']:.3e}, "
        f"E_rel_std={inv['E_rel_std']:.3e}, "
        f"e_rel_std={inv['e_rel_std']:.3e}"
    )
    if inv["h_rel_std"] > INV_REL_STD_H:
        raise AssertionError(f"{tag}: |h| rel_std too large: {inv['h_rel_std']:.3e}")
    if inv["E_rel_std"] > INV_REL_STD_E:
        raise AssertionError(f"{tag}: Energy rel_std too large: {inv['E_rel_std']:.3e}")
    if inv["e_rel_std"] > INV_REL_STD_e:
        raise AssertionError(f"{tag}: |e| rel_std too large: {inv['e_rel_std']:.3e}")

    cs = _cos_sim(v[0], v_pref_m_s)
    print(f"{tag}: v_pref cos_sim(v0, v_pref)={cs:.3f}")
    if dir_cos_min is not None and cs < float(dir_cos_min):
        raise AssertionError(f"{tag}: v_pref not respected: cos_sim={cs:.3f} < {float(dir_cos_min):.3f}")

    if compare_ssapy:
        r_sv, v_sv, t_sv = ssapy_orbit(
            r=np.asarray(res["r0"], float),
            v=np.asarray(res["v0"], float),
            t=t_rel,
            prop=KEP_PROP,
        )
        r_ke, v_ke, t_ke = ssapy_orbit(
            a=float(res["a"]),
            e=float(res["e"]),
            i=float(res["i"]),
            raan=float(res["raan"]),
            pa=float(res["pa"]),
            ta=float(res["ta"]),
            t=t_rel,
            prop=KEP_PROP,
        )

        err_int = _norm(r_sv - r_ke, axis=1)
        rms_int = float(np.sqrt(np.mean(err_int**2)))
        mx_int = float(np.max(err_int))
        print(f"{tag} SSAPY SV vs KE: RMS={rms_int:.3e} m, max={mx_int:.3e} m")
        if rms_int > SSAPY_INTERNAL_RMS_M or mx_int > SSAPY_INTERNAL_MAX_M:
            raise AssertionError(
                f"{tag}: SSAPy internal mismatch too large (RMS={rms_int:.3e}, max={mx_int:.3e})"
            )

    return res


def _run_sweep_case_with_retries(P1_m, lat_rad, tag_prefix):
    last_err = None

    for lon_deg in LON_TRY_DEG:
        lon_rad = np.deg2rad(float(lon_deg))
        P2_m = _p2_on_sphere_lat_lon(R2_SWEEP, lat_rad, lon_rad)

        t_hat = _compute_t_hat(P1_m, P2_m)
        vA = 1.0 * t_hat
        vB = -1.0 * t_hat

        try:
            print(f"{tag_prefix}: trying lon={lon_deg:+.1f} deg")
            resA = _run_one_case(
                P1_m, P2_m, vA,
                tag=f"{tag_prefix} A (lon={lon_deg:+.1f})",
                dir_cos_min=None,
                compare_ssapy=False,
            )
            resB = _run_one_case(
                P1_m, P2_m, vB,
                tag=f"{tag_prefix} B (lon={lon_deg:+.1f})",
                dir_cos_min=None,
                compare_ssapy=False,
            )
            return P2_m, resA, resB, float(lon_deg)
        except RuntimeError as e:
            last_err = e
            print(f"{tag_prefix}: ellipse_fit failed at lon={lon_deg:+.1f} deg: {e}")

    raise RuntimeError(f"{tag_prefix}: all longitude retries failed. Last error: {last_err}")


def _ssapy_kepler_err_series_from_state(res):
    """
    Compare ellipse_fit r(t) to SSAPy propagated from returned STATE (r0,v0)
    using KeplerianPropagator. Returns (t_rel, err_m).
    """
    t = np.asarray(res["t_rel"], float)
    r_fit = np.asarray(res["r"], float)

    r_ss, v_ss, t_ss = ssapy_orbit(
        r=np.asarray(res["r0"], float),
        v=np.asarray(res["v0"], float),
        t=t,
        prop=KEP_PROP,
    )

    err = _norm(r_fit - np.asarray(r_ss, float), axis=1)
    return t, err


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=== demos/test_ellipse_fit.py ===")
    print(f"SAVE_PLOTS={SAVE_PLOTS}")

    # -----------------------------
    # Baseline
    # -----------------------------
    P1_m = np.array([RGEO, 0.0, 0.0], dtype=float)
    P2_m = np.array([0.0 * RGEO, -1.1 * RGEO, 0.1 * RGEO], dtype=float)

    t_hat = _compute_t_hat(P1_m, P2_m)
    v_pref_A_m_s = 1.0 * t_hat
    v_pref_B_m_s = -1.0 * t_hat

    print("\n----- RUN A (baseline) -----")
    _run_one_case(P1_m, P2_m, v_pref_A_m_s, tag="RUN A", dir_cos_min=DIR_COS_MIN, compare_ssapy=True)

    print("\n----- RUN B (baseline) -----")
    _run_one_case(P1_m, P2_m, v_pref_B_m_s, tag="RUN B", dir_cos_min=DIR_COS_MIN, compare_ssapy=True)

    # -----------------------------
    # Sweep
    # -----------------------------
    print("\n=== Inclination sweep via endpoint selection (NO inc kwarg) ===")
    print("Start: equatorial GEO at +X (1*RGEO)")
    print("End: 2*RGEO sphere, latitude=inc, longitude tried around +90°")

    P1s_m = np.array([R1_SWEEP, 0.0, 0.0], dtype=float)

    sweep_store = []
    err_series = []

    for inc_deg in INC_SWEEP_DEG:
        lat_rad = np.deg2rad(float(inc_deg))
        tagp = f"SWEEP lat={inc_deg:+.0f}deg"

        print(f"\n----- {tagp} -----")
        P2s_m, resA, resB, used_lon = _run_sweep_case_with_retries(P1s_m, lat_rad, tagp)

        sweep_store.append({
            "inc_deg": float(inc_deg),
            "lon_deg": float(used_lon),
            "P2": P2s_m,
            "A": resA,
            "B": resB,
        })

        tA, eA = _ssapy_kepler_err_series_from_state(resA)
        err_series.append({"inc_deg": float(inc_deg), "lon_deg": float(used_lon), "dir": "A", "t": tA, "err_m": eA})

        tB, eB = _ssapy_kepler_err_series_from_state(resB)
        err_series.append({"inc_deg": float(inc_deg), "lon_deg": float(used_lon), "dir": "B", "t": tB, "err_m": eB})

        print(f"{tagp}: SSAPy(Kepler, from state) err A: RMS={float(np.sqrt(np.mean(eA**2))):.3e} m, max={float(np.max(eA)):.3e} m")
        print(f"{tagp}: SSAPy(Kepler, from state) err B: RMS={float(np.sqrt(np.mean(eB**2))):.3e} m, max={float(np.max(eB)):.3e} m")

    if SAVE_PLOTS:
        _plot_incsweep_all_transfers_3d(sweep_store, P1s_m)
        _plot_incsweep_compared_ssapy_time_series(err_series, INC_SWEEP_DEG)

    print("\nDONE! (baseline checks passed; sweep transfers + SSAPy comparison plotted)")


if __name__ == "__main__":
    main()
