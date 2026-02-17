#!/usr/bin/env python3
"""
demos/test_ellipse_fit.py

Rigorous validation driver for yeager_utils.ellipse_fit (demo location, test behavior).

This script:
  1) Runs ellipse_fit twice using velocity preference to pick direction (A/B)
  2) Validates shapes, finiteness, required keys
  3) Checks endpoints: r[0]≈P1, r[-1]≈P2
  4) Checks approximate invariants from returned r,v:
       - |h| const, specific energy const, |e| const
  5) Reconstructs with SSAPy:
       - from state vectors (r0,v0) at t = t_rel
       - from returned Keplerian elements at t = t_rel
  6) Optionally saves regression plots via yufig(fig, figname):
       - arcs
       - SV recon overlay
       - Kepler recon overlay
       - position error vs time (SV + Kepler)
       - radius vs time (fit vs ssapy)
       - speed vs time (fit vs ssapy)
       - A vs B separation vs time

Run:
  python3 demos/test_ellipse_fit.py

Optional:
  SAVE_PLOTS=1 python3 demos/test_ellipse_fit.py
"""

from yeager_utils import RGEO, ellipse_fit, ssapy_orbit, orbit_plot, yufig
import numpy as np
import os


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

    energy = 0.5 * vmag**2 - float(mu) / rmag

    e_vec = (np.cross(v_m_s, h_vec) / float(mu)) - (r_m / rmag[:, None])
    e_mag = _norm(e_vec, axis=1)

    return {
        "h_mag": h_mag,
        "energy": energy,
        "e_mag": e_mag,
        "h_rel_std": _rel_std(h_mag),
        "E_rel_std": _rel_std(energy),
        "e_rel_std": _rel_std(e_mag),
        "h_mean": float(np.mean(h_mag)),
        "E_mean": float(np.mean(energy)),
        "e_mean": float(np.mean(e_mag)),
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


def _ssapy_pos_err(r_ref_m, r_test_m, label=""):
    r_ref_m = np.asarray(r_ref_m, float)
    r_test_m = np.asarray(r_test_m, float)
    if r_ref_m.shape != r_test_m.shape:
        raise AssertionError(f"{label}shape mismatch: {r_ref_m.shape} vs {r_test_m.shape}")
    err = _norm(r_ref_m - r_test_m, axis=1)
    rms = float(np.sqrt(np.mean(err**2)))
    mx = float(np.max(err))
    print(f"{label}SSAPy position error: RMS={rms:.6e} m, max={mx:.6e} m")
    return rms, mx, err


# -----------------------------------------------------------------------------
# Config (tune to taste)
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

SSAPY_RMS_M = 5.0e-3
SSAPY_MAX_M = 5.0e-2

DIR_COS_MIN = 0.90

SAVE_PLOTS = (os.environ.get("SAVE_PLOTS", "0").strip() == "1")


# -----------------------------------------------------------------------------
# Plot helpers (saved via yufig)
# -----------------------------------------------------------------------------
def _plot_error_vs_time(results, recon_cache):
    import matplotlib.pyplot as plt

    # historical filename (jpg) for Kepler error
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name in ["A", "B"]:
        t = np.asarray(results[name]["t_rel"], float)
        r_fit = np.asarray(results[name]["r"], float)
        r_ke = np.asarray(recon_cache[name]["ke"][0], float)
        ax.plot(t, _norm(r_fit - r_ke, axis=1), label=f"Kepler err {name}")
    ax.set_xlabel("t_rel [s]")
    ax.set_ylabel("|r_fit - r_ssapy| [m]")
    ax.set_title("ellipse_fit vs SSAPy position error (Kepler reconstruction)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    yufig(fig, "tests/testing_ellipse_fit_distance.jpg")
    plt.close(fig)

    # SV error (png)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name in ["A", "B"]:
        t = np.asarray(results[name]["t_rel"], float)
        r_fit = np.asarray(results[name]["r"], float)
        r_sv = np.asarray(recon_cache[name]["sv"][0], float)
        ax.plot(t, _norm(r_fit - r_sv, axis=1), label=f"SV err {name}")
    ax.set_xlabel("t_rel [s]")
    ax.set_ylabel("|r_fit - r_ssapy| [m]")
    ax.set_title("ellipse_fit vs SSAPy position error (SV reconstruction)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    yufig(fig, "tests/testing_ellipse_fit_error_sv.png")
    plt.close(fig)

    # Kepler error (png)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name in ["A", "B"]:
        t = np.asarray(results[name]["t_rel"], float)
        r_fit = np.asarray(results[name]["r"], float)
        r_ke = np.asarray(recon_cache[name]["ke"][0], float)
        ax.plot(t, _norm(r_fit - r_ke, axis=1), label=f"Kepler err {name}")
    ax.set_xlabel("t_rel [s]")
    ax.set_ylabel("|r_fit - r_ssapy| [m]")
    ax.set_title("ellipse_fit vs SSAPy position error (Kepler reconstruction)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    yufig(fig, "tests/testing_ellipse_fit_error_kepler.png")
    plt.close(fig)

    # Combined SV + Kepler
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name in ["A", "B"]:
        t = np.asarray(results[name]["t_rel"], float)
        r_fit = np.asarray(results[name]["r"], float)
        r_sv = np.asarray(recon_cache[name]["sv"][0], float)
        r_ke = np.asarray(recon_cache[name]["ke"][0], float)
        ax.plot(t, _norm(r_fit - r_sv, axis=1), label=f"SV err {name}")
        ax.plot(t, _norm(r_fit - r_ke, axis=1), label=f"Kepler err {name}")
    ax.set_xlabel("t_rel [s]")
    ax.set_ylabel("position error [m]")
    ax.set_title("ellipse_fit vs SSAPy position error (SV & Kepler)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    yufig(fig, "tests/testing_ellipse_fit_error_both.png")
    plt.close(fig)


def _plot_radius_and_speed(results, recon_cache):
    import matplotlib.pyplot as plt

    # radius
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name in ["A", "B"]:
        t = np.asarray(results[name]["t_rel"], float)
        r_fit = np.asarray(results[name]["r"], float)
        r_ke = np.asarray(recon_cache[name]["ke"][0], float)
        ax.plot(t, _norm(r_fit, axis=1), label=f"|r| fit {name}")
        ax.plot(t, _norm(r_ke, axis=1), label=f"|r| ssapy {name}")
    ax.set_xlabel("t_rel [s]")
    ax.set_ylabel("|r| [m]")
    ax.set_title("Radius vs time (ellipse_fit vs SSAPy)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    yufig(fig, "tests/testing_ellipse_fit_radius.png")
    plt.close(fig)

    # speed
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name in ["A", "B"]:
        t = np.asarray(results[name]["t_rel"], float)
        v_fit = np.asarray(results[name]["v"], float)
        v_ke = np.asarray(recon_cache[name]["ke"][1], float)
        ax.plot(t, _norm(v_fit, axis=1), label=f"|v| fit {name}")
        ax.plot(t, _norm(v_ke, axis=1), label=f"|v| ssapy {name}")
    ax.set_xlabel("t_rel [s]")
    ax.set_ylabel("|v| [m/s]")
    ax.set_title("Speed vs time (ellipse_fit vs SSAPy)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    yufig(fig, "tests/testing_ellipse_fit_speed.png")
    plt.close(fig)


def _plot_arcs_3d(results, P1_m, P2_m):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for name in ["A", "B"]:
        r = np.asarray(results[name]["r"], float)
        ax.plot(r[:, 0], r[:, 1], r[:, 2], label=f"fit {name}")
    ax.scatter([P1_m[0]], [P1_m[1]], [P1_m[2]], s=40, label="P1")
    ax.scatter([P2_m[0]], [P2_m[1]], [P2_m[2]], s=40, label="P2")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("ellipse_fit arcs (direction picked by v_pref)")
    ax.legend()
    fig.tight_layout()
    yufig(fig, "tests/testing_ellipse_fit_arcs.png")
    plt.close(fig)


def _plot_sep_AB(results):
    import matplotlib.pyplot as plt

    rA = np.asarray(results["A"]["r"], float)
    rB = np.asarray(results["B"]["r"], float)
    if rA.shape != rB.shape:
        print(f"A vs B: different sample counts (A={rA.shape[0]}, B={rB.shape[0]}), skipping separation plot.")
        return

    sep = _norm(rA - rB, axis=1)
    t = np.asarray(results["A"]["t_rel"], float)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, sep)
    ax.set_xlabel("t_rel [s]")
    ax.set_ylabel("|r_A - r_B| [m]")
    ax.set_title("A vs B separation vs time")
    ax.grid(True)
    fig.tight_layout()
    yufig(fig, "tests/testing_ellipse_fit_separation_AB.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main harness
# -----------------------------------------------------------------------------
def main():
    print("=== demos/test_ellipse_fit.py (rigorous harness) ===")
    print(f"SAVE_PLOTS={SAVE_PLOTS}")

    # Scenario setup (keep close to your original)
    P1_m = np.array([RGEO, 0.0, 0.0], dtype=float)
    P2_m = np.array([-0.0 * RGEO, -1.1 * RGEO, 0.1 * RGEO], dtype=float)

    # Preferred velocity directions at P1
    t_hat = _compute_t_hat(P1_m, P2_m)
    v_pref_A_m_s = 1.0 * t_hat
    v_pref_B_m_s = -1.0 * t_hat

    results = {}
    recon_cache = {}

    for name, v_pref in [("A", v_pref_A_m_s), ("B", v_pref_B_m_s)]:
        print(f"\n----- RUN {name} -----")
        res = ellipse_fit(
            P1_m,
            P2_m,
            n_pts=400,
            plot=False,
            inc=0.0,
            v_pref_m_s=v_pref,
        )

        _assert_keys(res, REQUIRED_KEYS)

        r = np.asarray(res["r"], float)
        v = np.asarray(res["v"], float)
        t_rel = np.asarray(res["t_rel"], float)

        # schema / shape / finiteness / time
        if r.ndim != 2 or r.shape[1] != 3:
            raise AssertionError(f"RUN {name}: r must be (N,3), got {r.shape}")
        if v.shape != r.shape:
            raise AssertionError(f"RUN {name}: v must match r shape, got {v.shape}")
        if t_rel.ndim != 1 or t_rel.shape[0] != r.shape[0]:
            raise AssertionError(f"RUN {name}: t_rel must be (N,), got {t_rel.shape}")

        _finite(r, f"r[{name}]")
        _finite(v, f"v[{name}]")
        _finite(t_rel, f"t_rel[{name}]")

        if np.any(np.diff(t_rel) < -TIME_MONO_TOL_S):
            raise AssertionError(f"RUN {name}: t_rel is not non-decreasing.")

        d0 = _assert_close(r[0], P1_m, ENDPOINT_ATOL_M, f"RUN {name}: r[0] != P1")
        d1 = _assert_close(r[-1], P2_m, ENDPOINT_ATOL_M, f"RUN {name}: r[-1] != P2")
        print(f"RUN {name}: endpoints |r[0]-P1|={d0:.3e} m, |r[-1]-P2|={d1:.3e} m")

        # invariants
        inv = _invariants_from_rv(float(res["mu"]), r, v)
        print(
            f"RUN {name}: invariants "
            f"h_rel_std={inv['h_rel_std']:.3e}, "
            f"E_rel_std={inv['E_rel_std']:.3e}, "
            f"e_rel_std={inv['e_rel_std']:.3e}"
        )
        if inv["h_rel_std"] > INV_REL_STD_H:
            raise AssertionError(f"RUN {name}: |h| rel_std too large: {inv['h_rel_std']:.3e}")
        if inv["E_rel_std"] > INV_REL_STD_E:
            raise AssertionError(f"RUN {name}: Energy rel_std too large: {inv['E_rel_std']:.3e}")
        if inv["e_rel_std"] > INV_REL_STD_e:
            raise AssertionError(f"RUN {name}: |e| rel_std too large: {inv['e_rel_std']:.3e}")

        # direction preference
        cs = _cos_sim(v[0], v_pref)
        print(f"RUN {name}: v_pref cos_sim(v0, v_pref)={cs:.3f}")
        if cs < DIR_COS_MIN:
            raise AssertionError(f"RUN {name}: v_pref not respected: cos_sim={cs:.3f} < {DIR_COS_MIN:.3f}")

        results[name] = res

        # SSAPy recon: state vectors
        r_sv, v_sv, t_sv = ssapy_orbit(
            r=np.asarray(res["r0"], float),
            v=np.asarray(res["v0"], float),
            t=t_rel,
        )
        _ssapy_pos_err(r, r_sv, label=f"SV RECON {name}: ")

        # SSAPy recon: Kepler elements
        r_ke, v_ke, t_ke = ssapy_orbit(
            a=float(res["a"]),
            e=float(res["e"]),
            i=float(res["i"]),
            raan=float(res["raan"]),
            pa=float(res["pa"]),
            ta=float(res["ta"]),
            t=t_rel,
        )
        _ssapy_pos_err(r, r_ke, label=f"KEPLER RECON {name}: ")

        # enforce SSAPy thresholds
        rms_sv, mx_sv, _ = _ssapy_pos_err(r, r_sv, label=f"SV CHECK {name}: ")
        rms_ke, mx_ke, _ = _ssapy_pos_err(r, r_ke, label=f"KE CHECK {name}: ")

        if rms_sv > SSAPY_RMS_M or mx_sv > SSAPY_MAX_M:
            raise AssertionError(f"RUN {name}: SV recon too far (RMS={rms_sv:.3e}, max={mx_sv:.3e})")
        if rms_ke > SSAPY_RMS_M or mx_ke > SSAPY_MAX_M:
            raise AssertionError(f"RUN {name}: Kepler recon too far (RMS={rms_ke:.3e}, max={mx_ke:.3e})")

        recon_cache[name] = {"sv": (r_sv, v_sv, t_sv), "ke": (r_ke, v_ke, t_ke)}

    # A vs B sanity
    rA = np.asarray(results["A"]["r"], float)
    rB = np.asarray(results["B"]["r"], float)
    if rA.shape == rB.shape:
        sep = _norm(rA - rB, axis=1)
        print(f"\nA vs B separation: mean={float(np.mean(sep)):.6e} m, max={float(np.max(sep)):.6e} m")
        if float(np.max(sep)) < 1e-12:
            raise AssertionError("A vs B are identical; direction selection did not produce distinct solutions.")
    else:
        print(f"\nA vs B: different sample counts (A={rA.shape[0]}, B={rB.shape[0]}), skipping pointwise separation.")

    # Plots (optional)
    if SAVE_PLOTS:
        _plot_arcs_3d(results, P1_m, P2_m)

        # If you still want orbit_plot overlays (it may have its own saving),
        # we keep it but don't rely on it for saving.
        try:
            orbit_plot(
                [results["A"]["r"], results["B"]["r"]],
                t=[results["A"]["t_rel"], results["B"]["t_rel"]],
                title="ellipse_fit arcs (direction picked by v_pref)",
                show=False,
                save_path=None,
            )
        except Exception:
            pass

        # Recon overlays (use simple 3D plots via yufig)
        import matplotlib.pyplot as plt

        # SV overlay
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for name in ["A", "B"]:
            r_fit = np.asarray(results[name]["r"], float)
            r_sv = np.asarray(recon_cache[name]["sv"][0], float)
            ax.plot(r_fit[:, 0], r_fit[:, 1], r_fit[:, 2], label=f"fit {name}")
            ax.plot(r_sv[:, 0], r_sv[:, 1], r_sv[:, 2], label=f"sv {name}", alpha=0.7)
        ax.set_title("SSAPy reconstruction via state vectors (t=t_rel)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.legend()
        fig.tight_layout()
        yufig(fig, "tests/testing_ellipse_fit_recons_sv.png")
        plt.close(fig)

        # Kepler overlay
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for name in ["A", "B"]:
            r_fit = np.asarray(results[name]["r"], float)
            r_ke = np.asarray(recon_cache[name]["ke"][0], float)
            ax.plot(r_fit[:, 0], r_fit[:, 1], r_fit[:, 2], label=f"fit {name}")
            ax.plot(r_ke[:, 0], r_ke[:, 1], r_ke[:, 2], label=f"kepler {name}", alpha=0.7)
        ax.set_title("SSAPy reconstruction via Keplerian elements (t=t_rel)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.legend()
        fig.tight_layout()
        yufig(fig, "tests/testing_ellipse_fit_recons_kepler.png")
        plt.close(fig)

        _plot_error_vs_time(results, recon_cache)
        _plot_radius_and_speed(results, recon_cache)
        _plot_sep_AB(results)

    print("\nDONE! (ellipse_fit rigorous harness passed)")


if __name__ == "__main__":
    main()
