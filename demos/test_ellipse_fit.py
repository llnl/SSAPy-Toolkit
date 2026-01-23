#!/usr/bin/env python3
"""
Full evaluation driver for the velocity-directed `ellipse_fit`.

This script:
  1) Runs ellipse_fit twice using velocity preference to pick direction
  2) Validates shapes, finiteness, required keys
  3) Checks endpoints: r[0]≈P1, r[-1]≈P2
  4) Checks approximate invariants from returned r,v:
       - |h| const, specific energy const, |e| const (numerical tolerance)
  5) Reconstructs with SSAPy:
       - from state vectors (r0,v0) at t = t_rel
       - from returned Keplerian elements at t = t_rel
  6) Saves plots into tests/ via your figpath helper, including:
       - arcs
       - SV recon overlay
       - Kepler recon overlay
       - position error vs time (SV + Kepler)
       - radius vs time (fit vs ssapy)
       - speed vs time (fit vs ssapy)

No math module, no typing module.
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
from yeager_utils import RGEO, ellipse_fit, pprint, figpath, ssapy_orbit
from ssapy.plotUtils import orbit_plot
import numpy as np


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _norm(x, axis=None):
    return np.linalg.norm(x, axis=axis)


def _unit(x, eps=1e-15):
    x = np.asarray(x, float)
    n = np.linalg.norm(x)
    if n < eps:
        return None
    return x / n


def _finite(x, name):
    x = np.asarray(x)
    if not np.all(np.isfinite(x)):
        bad = np.where(~np.isfinite(x))
        raise ValueError(f"Non-finite values in {name} at indices {bad}")


def _assert_keys(d, keys):
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"Missing keys from ellipse_fit result: {missing}")


def _endpoint_report(res, P1_m, P2_m, label=""):
    r_m = np.asarray(res["r"], float)
    d0_m = float(_norm(r_m[0] - P1_m))
    d1_m = float(_norm(r_m[-1] - P2_m))
    print(f"{label}endpoint deltas: |r[0]-P1|={d0_m:.6e} m, |r[-1]-P2|={d1_m:.6e} m")
    return d0_m, d1_m


def _invariant_report(res, label=""):
    mu_m3_s2 = float(res["mu"])
    r_m = np.asarray(res["r"], float)
    v_m_s = np.asarray(res["v"], float)

    h_vec_m2_s = np.cross(r_m, v_m_s)
    h_mag_m2_s = _norm(h_vec_m2_s, axis=1)

    energy_J_kg = 0.5 * _norm(v_m_s, axis=1) ** 2 - mu_m3_s2 / _norm(r_m, axis=1)

    e_vec = (np.cross(v_m_s, h_vec_m2_s) / mu_m3_s2) - (r_m / _norm(r_m, axis=1)[:, None])
    e_mag = _norm(e_vec, axis=1)

    def _rel_std(x, eps=1e-30):
        return float(np.std(x) / max(np.mean(np.abs(x)), eps))

    metrics = {
        "h_rel_std": _rel_std(h_mag_m2_s),
        "E_rel_std": _rel_std(energy_J_kg),
        "e_rel_std": _rel_std(e_mag),
        "h_mean": float(np.mean(h_mag_m2_s)),
        "E_mean": float(np.mean(energy_J_kg)),
        "e_mean": float(np.mean(e_mag)),
    }

    print(f"{label}invariants:")
    print(f"{label}  |h| mean={metrics['h_mean']:.6e}, rel std={metrics['h_rel_std']:.3e}")
    print(f"{label}  E   mean={metrics['E_mean']:.6e}, rel std={metrics['E_rel_std']:.3e}")
    print(f"{label}  e   mean={metrics['e_mean']:.6e}, rel std={metrics['e_rel_std']:.3e}")

    return metrics


def _ssapy_pos_err(r_ref_m, r_test_m, label=""):
    r_ref_m = np.asarray(r_ref_m, float)
    r_test_m = np.asarray(r_test_m, float)
    if r_ref_m.shape != r_test_m.shape:
        raise ValueError(f"{label} shape mismatch: {r_ref_m.shape} vs {r_test_m.shape}")
    err_m = _norm(r_ref_m - r_test_m, axis=1)
    rms_m = float(np.sqrt(np.mean(err_m**2)))
    max_m = float(np.max(err_m))
    print(f"{label}SSAPy position error: RMS={rms_m:.6e} m, max={max_m:.6e} m")
    return rms_m, max_m, err_m


def _compute_t_hat(P1_m, P2_m):
    """
    Approximate plane normal from P1×P2 and compute t_hat = (w_hat × r_hat) at P1.
    This yields a reasonable tangential direction for building v_pref.
    """
    r_hat = _unit(P1_m)
    if r_hat is None:
        raise ValueError("P1 has near-zero norm.")

    w_hat = _unit(np.cross(P1_m, P2_m))
    if w_hat is None:
        # colinear fallback
        cand = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(r_hat, cand)) > 0.99:
            cand = np.array([1.0, 0.0, 0.0])
        w_hat = _unit(np.cross(r_hat, cand))
        if w_hat is None:
            raise ValueError("Could not form a stable plane normal.")

    t_hat = _unit(np.cross(w_hat, r_hat))
    if t_hat is None:
        raise ValueError("Could not form tangential unit vector at P1.")
    return t_hat


def _plot_error_vs_time(results, recon_cache):
    """
    recon_cache[name] must contain:
      - "sv": (r_sv_m, v_sv_m_s, t_sv_s)
      - "ke": (r_ke_m, v_ke_m_s, t_ke_s)
    Saves:
      - testing_ellipse_fit_distance.jpg (Kepler error)
      - testing_ellipse_fit_error_sv.png
      - testing_ellipse_fit_error_kepler.png
      - testing_ellipse_fit_error_both.png
    """
    import matplotlib.pyplot as plt

    # --- Kepler error plot (historical filename requested) ---
    save_path_dist = figpath("tests/testing_ellipse_fit_distance.jpg")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name in ["A", "B"]:
        res = results[name]
        t_rel_s = np.asarray(res["t_rel"], float)
        r_ke_m = np.asarray(recon_cache[name]["ke"][0], float)
        err_m = _norm(np.asarray(res["r"], float) - r_ke_m, axis=1)
        ax.plot(t_rel_s, err_m, label=f"Kepler err {name}")
    ax.set_xlabel("t_rel [s]")
    ax.set_ylabel("|r_fit - r_ssapy| [m]")
    ax.set_title("ellipse_fit vs SSAPy position error (Kepler reconstruction)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path_dist, dpi=200)
    plt.close(fig)
    print(f"[saved] {save_path_dist}")

    # --- SV error plot ---
    save_path_sv = figpath("tests/testing_ellipse_fit_error_sv.png")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name in ["A", "B"]:
        res = results[name]
        t_rel_s = np.asarray(res["t_rel"], float)
        r_sv_m = np.asarray(recon_cache[name]["sv"][0], float)
        err_m = _norm(np.asarray(res["r"], float) - r_sv_m, axis=1)
        ax.plot(t_rel_s, err_m, label=f"SV err {name}")
    ax.set_xlabel("t_rel [s]")
    ax.set_ylabel("|r_fit - r_ssapy| [m]")
    ax.set_title("ellipse_fit vs SSAPy position error (SV reconstruction)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path_sv, dpi=200)
    plt.close(fig)
    print(f"[saved] {save_path_sv}")

    # --- Kepler error plot (png) ---
    save_path_ke = figpath("tests/testing_ellipse_fit_error_kepler.png")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name in ["A", "B"]:
        res = results[name]
        t_rel_s = np.asarray(res["t_rel"], float)
        r_ke_m = np.asarray(recon_cache[name]["ke"][0], float)
        err_m = _norm(np.asarray(res["r"], float) - r_ke_m, axis=1)
        ax.plot(t_rel_s, err_m, label=f"Kepler err {name}")
    ax.set_xlabel("t_rel [s]")
    ax.set_ylabel("|r_fit - r_ssapy| [m]")
    ax.set_title("ellipse_fit vs SSAPy position error (Kepler reconstruction)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path_ke, dpi=200)
    plt.close(fig)
    print(f"[saved] {save_path_ke}")

    # --- Combined error plot (SV + Kepler) ---
    save_path_both = figpath("tests/testing_ellipse_fit_error_both.png")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name in ["A", "B"]:
        res = results[name]
        t_rel_s = np.asarray(res["t_rel"], float)
        r_sv_m = np.asarray(recon_cache[name]["sv"][0], float)
        r_ke_m = np.asarray(recon_cache[name]["ke"][0], float)
        err_sv_m = _norm(np.asarray(res["r"], float) - r_sv_m, axis=1)
        err_ke_m = _norm(np.asarray(res["r"], float) - r_ke_m, axis=1)
        ax.plot(t_rel_s, err_sv_m, label=f"SV err {name}")
        ax.plot(t_rel_s, err_ke_m, label=f"Kepler err {name}")
    ax.set_xlabel("t_rel [s]")
    ax.set_ylabel("|r_fit - r_ssapy| [m]")
    ax.set_title("ellipse_fit vs SSAPy position error (SV & Kepler)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path_both, dpi=200)
    plt.close(fig)
    print(f"[saved] {save_path_both}")


def _plot_radius_and_speed(results, recon_cache):
    """
    Saves:
      - testing_ellipse_fit_radius.png
      - testing_ellipse_fit_speed.png
    Each compares ellipse_fit vs SSAPy (Kepler recon) for A and B.
    """
    import matplotlib.pyplot as plt

    # radius
    save_path_r = figpath("tests/testing_ellipse_fit_radius.png")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name in ["A", "B"]:
        res = results[name]
        t_rel_s = np.asarray(res["t_rel"], float)
        r_fit_m = np.asarray(res["r"], float)
        r_ke_m = np.asarray(recon_cache[name]["ke"][0], float)
        ax.plot(t_rel_s, _norm(r_fit_m, axis=1), label=f"|r| fit {name}")
        ax.plot(t_rel_s, _norm(r_ke_m, axis=1), label=f"|r| ssapy {name}")
    ax.set_xlabel("t_rel [s]")
    ax.set_ylabel("|r| [m]")
    ax.set_title("Radius vs time (ellipse_fit vs SSAPy)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path_r, dpi=200)
    plt.close(fig)
    print(f"[saved] {save_path_r}")

    # speed
    save_path_v = figpath("tests/testing_ellipse_fit_speed.png")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name in ["A", "B"]:
        res = results[name]
        t_rel_s = np.asarray(res["t_rel"], float)
        v_fit_m_s = np.asarray(res["v"], float)
        v_ke_m_s = np.asarray(recon_cache[name]["ke"][1], float)
        ax.plot(t_rel_s, _norm(v_fit_m_s, axis=1), label=f"|v| fit {name}")
        ax.plot(t_rel_s, _norm(v_ke_m_s, axis=1), label=f"|v| ssapy {name}")
    ax.set_xlabel("t_rel [s]")
    ax.set_ylabel("|v| [m/s]")
    ax.set_title("Speed vs time (ellipse_fit vs SSAPy)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path_v, dpi=200)
    plt.close(fig)
    print(f"[saved] {save_path_v}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Scenario setup
    # ------------------------------------------------------------------
    P1_m = np.array([RGEO, 0.0, 0.0], dtype=float)
    P2_m = np.array([-0.0 * RGEO, -1.1 * RGEO, 0.1 * RGEO], dtype=float)

    # Tangential directions used to define two opposite "preferred" velocities
    t_hat = _compute_t_hat(P1_m, P2_m)
    v_pref_A_m_s = 1.0 * t_hat
    v_pref_B_m_s = -1.0 * t_hat

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

    # ------------------------------------------------------------------
    # Run ellipse_fit twice and collect results
    # ------------------------------------------------------------------
    results = {}
    for name, v_pref_m_s in [("A", v_pref_A_m_s), ("B", v_pref_B_m_s)]:
        print(f"\n===== RUN {name} (v_pref along {'+t_hat' if name=='A' else '-t_hat'}) =====")

        res = ellipse_fit(
            P1_m,
            P2_m,
            n_pts=400,
            plot=False,
            inc=0.0,                # radians
            v_pref_m_s=v_pref_m_s,
        )

        _assert_keys(res, REQUIRED_KEYS)

        r_m = np.asarray(res["r"], float)
        v_m_s = np.asarray(res["v"], float)
        t_rel_s = np.asarray(res["t_rel"], float)

        # shape checks
        if r_m.ndim != 2 or r_m.shape[1] != 3:
            raise ValueError(f"RUN {name}: r must be (N,3), got {r_m.shape}")
        if v_m_s.shape != r_m.shape:
            raise ValueError(f"RUN {name}: v must match r shape, got {v_m_s.shape}")
        if t_rel_s.ndim != 1 or t_rel_s.shape[0] != r_m.shape[0]:
            raise ValueError(f"RUN {name}: t_rel must be (N,), got {t_rel_s.shape}")

        _finite(r_m, f"r[{name}]")
        _finite(v_m_s, f"v[{name}]")
        _finite(t_rel_s, f"t_rel[{name}]")

        if np.any(np.diff(t_rel_s) < -1e-9):
            raise ValueError(f"RUN {name}: t_rel is not non-decreasing.")

        _endpoint_report(res, P1_m, P2_m, label=f"RUN {name}: ")
        _invariant_report(res, label=f"RUN {name}: ")

        pprint(res)
        results[name] = res

    # ------------------------------------------------------------------
    # Save figure: arcs from ellipse_fit
    # ------------------------------------------------------------------
    save_path_arcs = figpath("tests/testing_ellipse_fit_arcs.png")
    orbit_plot(
        [results["A"]["r"], results["B"]["r"]],
        t=[results["A"]["t_rel"], results["B"]["t_rel"]],
        title="ellipse_fit arcs (direction picked by v_pref)",
        show=False,
        save_path=save_path_arcs,
    )
    print(f"[saved] {save_path_arcs}")

    # ------------------------------------------------------------------
    # 1️⃣  Reconstruct using initial state vectors r0,v0 AT THE SAME t_rel GRID
    # ------------------------------------------------------------------
    recon_cache = {"A": {}, "B": {}}
    sv_recons = []
    for name, res in results.items():
        r0_m = np.asarray(res["r0"], float)
        v0_m_s = np.asarray(res["v0"], float)
        t_rel_s = np.asarray(res["t_rel"], float)

        r_sv_m, v_sv_m_s, t_sv_s = ssapy_orbit(
            r=r0_m,
            v=v0_m_s,
            t=t_rel_s,
        )

        _ssapy_pos_err(res["r"], r_sv_m, label=f"SV RECON {name}: ")
        recon_cache[name]["sv"] = (r_sv_m, v_sv_m_s, t_sv_s)
        sv_recons.append((name, r_sv_m, t_sv_s))

    save_path_sv = figpath("tests/testing_ellipse_fit_recons_sv.png")
    orbit_plot(
        [r for _, r, _ in sv_recons],
        t=[t for _, _, t in sv_recons],
        title="SSAPy reconstructions via state vectors (t=t_rel)",
        show=False,
        save_path=save_path_sv,
    )
    print(f"[saved] {save_path_sv}")

    # ------------------------------------------------------------------
    # 2️⃣  Reconstruct using Keplerian elements AT THE SAME t_rel GRID
    # ------------------------------------------------------------------
    ke_recons = []
    for name, res in results.items():
        t_rel_s = np.asarray(res["t_rel"], float)

        r_ke_m, v_ke_m_s, t_ke_s = ssapy_orbit(
            a=float(res["a"]),
            e=float(res["e"]),
            i=float(res["i"]),
            raan=float(res["raan"]),
            pa=float(res["pa"]),
            ta=float(res["ta"]),
            t=t_rel_s,
        )

        _ssapy_pos_err(res["r"], r_ke_m, label=f"KEPLER RECON {name}: ")
        recon_cache[name]["ke"] = (r_ke_m, v_ke_m_s, t_ke_s)
        ke_recons.append((name, r_ke_m, t_ke_s))

    save_path_ke = figpath("tests/testing_ellipse_fit_recons_kepler.png")
    orbit_plot(
        [r for _, r, _ in ke_recons],
        t=[t for _, _, t in ke_recons],
        title="SSAPy reconstructions via Keplerian elements (t=t_rel)",
        show=False,
        save_path=save_path_ke,
    )
    print(f"[saved] {save_path_ke}")

    # ------------------------------------------------------------------
    # 3️⃣  Error/time + radius + speed plots (regression artifacts)
    # ------------------------------------------------------------------
    _plot_error_vs_time(results, recon_cache)
    _plot_radius_and_speed(results, recon_cache)

    # ------------------------------------------------------------------
    # 4️⃣  Compare the two solutions against each other (sanity)
    # ------------------------------------------------------------------
    rA_m = np.asarray(results["A"]["r"], float)
    rB_m = np.asarray(results["B"]["r"], float)
    if rA_m.shape == rB_m.shape:
        sep_m = _norm(rA_m - rB_m, axis=1)
        print(f"\nA vs B separation: mean={float(np.mean(sep_m)):.6e} m, max={float(np.max(sep_m)):.6e} m")

        # Optional plot: separation vs time
        import matplotlib.pyplot as plt
        save_path_sep = figpath("tests/testing_ellipse_fit_separation_AB.png")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.asarray(results["A"]["t_rel"], float), sep_m)
        ax.set_xlabel("t_rel [s]")
        ax.set_ylabel("|r_A - r_B| [m]")
        ax.set_title("A vs B separation vs time")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(save_path_sep, dpi=200)
        plt.close(fig)
        print(f"[saved] {save_path_sep}")
    else:
        print(f"\nA vs B: different sample counts (A={rA_m.shape[0]}, B={rB_m.shape[0]}), skipping pointwise separation.")

    print("\nDONE!")
