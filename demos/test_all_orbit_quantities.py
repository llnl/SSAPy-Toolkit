# demo_all_orbital_quantities.py
"""
Demo for ssapy_toolkit.Orbital_Mechanics.all_orbit_quantities.all_orbital_quantities

Shows three usage modes:
  1) From Cartesian state (r, v)
  2) From (periapsis, apoapsis)
  3) From (a, e) plus mean anomaly (ma) -> true anomaly via keplerian.true_anomaly [2]
"""

import numpy as np
from astropy.time import Time

from ssapy_toolkit.constants import EARTH_MU
from ssapy_toolkit import all_orbital_quantities


def print_summary(tag, out):
    print(f"\n=== {tag} ===")

    # Print all scalar-ish fields first (sorted)
    for k in sorted(out.keys()):
        v = out[k]
        if isinstance(v, np.ndarray):
            continue
        print(f"{k:>16s}: {v}")

    # Then print arrays (r, v, vectors, etc.)
    for k in sorted(out.keys()):
        v = out[k]
        if isinstance(v, np.ndarray):
            v_arr = np.asarray(v)
            if v_arr.ndim == 1 and v_arr.size == 3:
                print(f"{k:>16s}: {v_arr}   |{k}|={np.linalg.norm(v_arr):.6g}")
            else:
                print(f"{k:>16s}: shape={v_arr.shape}\n{v_arr}")

    # Keep these convenience magnitudes too (since you were using them)
    if "r" in out:
        print(f"{'|r|':>16s}: {np.linalg.norm(out['r']):.3f} m")
    if "v" in out:
        print(f"{'|v|':>16s}: {np.linalg.norm(out['v']):.6f} m/s")


def main():
    t = Time.now().gps
    mu = EARTH_MU

    # -----------------------------
    # Case 1: Cartesian state (LEO-ish)
    # -----------------------------
    r = np.array([7000e3, 0.0, 0.0])       # m
    v = np.array([0.0, 7.546e3, 1.0e2])    # m/s (slight out-of-plane component)
    out_rv = all_orbital_quantities(r=r, v=v, t=t, mu=mu)
    print_summary("From (r, v)", out_rv)

    # -----------------------------
    # Case 2: From periapsis/apoapsis (elliptic)
    # NOTE: periapsis/apoapsis are distances from Earth's center (not altitude).
    # -----------------------------
    periapsis = 7000e3  # m
    apoapsis = 12000e3  # m
    out_rpra = all_orbital_quantities(
        periapsis=periapsis,
        apoapsis=apoapsis,
        i=np.deg2rad(28.5),
        raan=np.deg2rad(40.0),
        pa=np.deg2rad(15.0),
        ta=np.deg2rad(10.0),
        t=t,
        mu=mu,
    )
    print_summary("From (periapsis, apoapsis)", out_rpra)

    # -----------------------------
    # Case 3: From (a, e) and mean anomaly (ma) (ta omitted -> ma->ta)
    # -----------------------------
    a = 9000e3
    e = 0.1
    ma = np.deg2rad(25.0)
    out_ae_ma = all_orbital_quantities(
        a=a,
        e=e,
        i=np.deg2rad(10.0),
        raan=np.deg2rad(80.0),
        pa=np.deg2rad(5.0),
        ma=ma,          # ma provided, ta omitted -> uses ssapy_toolkit.keplerian.true_anomaly [2]
        t=t,
        mu=mu,
    )
    print_summary("From (a, e) with mean anomaly (ma)", out_ae_ma)


if __name__ == "__main__":
    main()
