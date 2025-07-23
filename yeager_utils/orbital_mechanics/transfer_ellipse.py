import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..constants import EARTH_RADIUS
from .ellipse_3d import ellipse_arc


def _orbital_inclination(r, v):
    """Return inclination [deg] and ccw flag from r & v."""
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    if h_norm < 1e-9:
        raise ValueError("Position and velocity are (near‑)colinear; "
                         "cannot determine orbital plane.")
    incl = np.degrees(np.arccos(h[2] / h_norm))  # 0…180°
    ccw   = h[2] >= 0.0                           # +z angular momentum ⇒ CCW in xy‑plane
    return incl, ccw


def transfer_ellipse(
        P1, P2, v, *,
        inc_span=10.0,         # ± span [deg] around reference inclination
        n_inclinations=9,
        try_both_dirs=False,   # DEFAULT is False as requested
        n_trials=30,
        top_n=5,
        plot=False,
        debug=False):

    atm_limit = EARTH_RADIUS + 100e3

    # --- 1. Reference orbit plane ------------------------------------------------
    i_ref, ccw_ref = _orbital_inclination(P1, v)

    # Build inclination grid around i_ref
    raw_inclinations = np.linspace(i_ref - inc_span, i_ref + inc_span, n_inclinations)
    inclinations = np.clip(raw_inclinations, 0.0, 180.0)
    inclinations = np.unique(np.round(inclinations, decimals=6))  # remove exact duplicates

    directions = [ccw_ref] if not try_both_dirs else [ccw_ref, not ccw_ref]

    if debug:
        print(f"\nReference inclination: {i_ref:.3f}°,  ccw_ref = {ccw_ref}")
        print(f"Testing {len(inclinations)} unique inclination(s): {inclinations}")
        print(f"Testing directions: {directions}")

    # --- 2. Search over candidate planes ----------------------------------------
    candidates = []
    tested = set()

    for inc in inclinations:
        for ccw in directions:
            key = (round(inc, 6), ccw)
            if key in tested:
                continue
            tested.add(key)

            try:
                arc3d, vel3d, t_rel, info = ellipse_arc(
                    P1, P2,
                    inc=inc,
                    ccw=ccw,
                    n_pts=1000,
                    plot=False,
                    debug=debug
                )

                r_norms = np.linalg.norm(arc3d, axis=1)
                if np.any(r_norms < atm_limit):
                    if debug:
                        print(f"Rejected inc={inc:.2f}°, ccw={ccw}: "
                              f"dips into atmosphere (rₘᵢₙ={r_norms.min()/1e3:.2f} km)")
                    continue

                Δv = np.linalg.norm(vel3d[0] - v)
                candidates.append({
                    'delta_v': Δv,
                    'inc': inc,
                    'ccw': ccw,
                    'periapsis_km': r_norms.min() / 1e3,
                    'apoapsis_km' : r_norms.max() / 1e3,
                    'duration_s'  : t_rel[-1],
                    'a'           : info['a'],
                    'e'           : info['e'],
                    'F2'          : info['F2'],
                    'arc3d'       : arc3d,
                    'vel3d'       : vel3d,
                    't_rel'       : t_rel,
                })

            except Exception as e:
                if debug:
                    print(f"Failed inc={inc:.2f}°, ccw={ccw}: {e}")
                continue

    if not candidates:
        raise RuntimeError("No valid transfer solutions found.")

    # --- 3. Rank & report --------------------------------------------------------
    candidates.sort(key=lambda c: c['delta_v'])

    df = pd.DataFrame([{
        'Δv (m/s)'     : c['delta_v'],
        'Inc (°)'      : np.degrees(info['i']),
        'CCW'          : c['ccw'],
        'Periapsis (km)': c['periapsis_km'],
        'Apoapsis (km)' : c['apoapsis_km'],
        'Duration (s)'  : c['duration_s'],
        'a (m)'         : info['a'],
        'e'             : info['e'],
    } for c in candidates])

    print("\nTop candidate solutions:\n")
    print(df.head(top_n).to_string(index=True))

    # --- 4. Optional plotting ----------------------------------------------------
    if plot:
        for i, c in enumerate(candidates[:top_n]):
            arc = c['arc3d'] / 1e3
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(projection='3d')
            ax.plot3D(arc[:, 0], arc[:, 1], arc[:, 2],
                      label=f"Candidate {i+1}")
            ax.scatter([0], [0], [0], color='k', label='Earth')
            ax.set_title(f"Transfer Arc #{i+1}  |  Δv = {c['delta_v']:.2f} m/s")
            ax.set_xlabel("X [km]"); ax.set_ylabel("Y [km]"); ax.set_zlabel("Z [km]")
            ax.legend()
            plt.tight_layout()
            plt.show()

    return candidates[:top_n]
