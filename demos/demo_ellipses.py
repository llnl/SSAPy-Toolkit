#!/usr/bin/env python3
"""Example driver for the *new* `ellipse_arc` that now returns a single dict."""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
from yeager_utils import RGEO, ellipse_arc, get_times           # type: ignore
from ssapy.simple import ssapy_orbit                           # type: ignore
from ssapy.plotUtils import orbit_plot                         # type: ignore
from ssapy import Orbit                                        # type: ignore
import numpy as np


# ----------------------------------------------------------------------
# Pretty printer for a handful of arc samples
# ----------------------------------------------------------------------
def print_samples(label: str, rs, vs, ts):
    print(f"--- {label} ---")
    for i, (r, v_i, t_i) in enumerate(zip(rs, vs, ts), start=1):
        print(f" Sample {i:1d}:")
        print(f"   r = {r}")
        print(f"   v = {v_i}")
        print(f"   t = {t_i:.6f} s")
    print()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    P1 = np.array([RGEO, 0.0, 0.0])
    P2 = np.array([-0.0 * RGEO, -1.0 * RGEO, 0.0 * RGEO])

    # ------------------------------------------------------------------
    # Run ellipse_arc twice (CW and CCW) and collect results
    # ------------------------------------------------------------------
    results = {}
    for ccw in [False, True]:
        print(f"\n===== CCW = {ccw} =====")

        res = ellipse_arc(
            P1,
            P2,
            n_pts=400,
            plot=True,
            inc=0.0,
            ccw=ccw,
            debug=True,
        )

        # Split out convenience aliases
        arc = res["arc3d"]
        vel = res["vel3d"]
        times = res["t_rel"]

        # Quick sanity print‑out
        print(f"Total flight time: {times[-1]:.3f} s")
        print("Keplerian elements (rad):")
        print(
            f"  a = {res['a']:.3f} m  |  e = {res['e']:.6f}\n"
            f"  i = {res['i']:.6f}\n"
            f"  raan = {res['raan']:.6f}\n"
            f"  ω    = {res['pa']:.6f}\n"
            f"  ν0   = {res['ta']:.6f}\n"
        )

        # Uncomment to inspect the first / last samples
        # print_samples("First sample", arc[:1], vel[:1], times[:1])
        # print_samples("Last  sample", arc[-1:], vel[-1:], times[-1:])

        # Stash for later reconstruction tests
        results[ccw] = res

    # ------------------------------------------------------------------
    # 1️⃣  Reconstruct using the initial state vector r0, v0
    # ------------------------------------------------------------------
    sv_recons = []
    for ccw, res in results.items():
        r0, v0 = res["r0"], res["v0"]
        T_flight = res["t_rel"][-1]

        # Build an evenly‑spaced epoch vector the same length as t_rel
        epochs = get_times(duration=(T_flight, "s"), freq=(len(res["t_rel"]), "s"))[0]

        r_sv, v_sv, t_sv = ssapy_orbit(
            r=r0,
            v=v0,
            duration=(T_flight, "s"),
            freq=(len(res["t_rel"]), "s"),
        )

        sv_recons.append((ccw, r_sv, t_sv))

    orbit_plot(
        [r for _, r, _ in sv_recons],
        t=[t for _, _, t in sv_recons],
        title="SSAPy reconstructions via state vectors",
        show=True,
    )

    # ------------------------------------------------------------------
    # 2️⃣  Reconstruct using the *Keplerian* elements returned by ellipse_arc
    # ------------------------------------------------------------------
    ke_recons = []
    for ccw, res in results.items():
        T_flight = res["t_rel"][-1]

        r_ke, v_ke, t_ke = ssapy_orbit(
            a=res["a"],
            e=res["e"],
            i=res["i"],
            raan=res["raan"],
            pa=res["pa"],
            ta=res["ta"],
            duration=(T_flight, "s"),
            freq=(len(res["t_rel"]), "s"),
        )

        ke_recons.append((ccw, r_ke, t_ke))

    orbit_plot(
        [r for _, r, _ in ke_recons],
        t=[t for _, _, t in ke_recons],
        title="SSAPy reconstructions via Keplerian elements",
        show=True,
    )

print("DONE!")
