import numpy as np

from ssapy_toolkit.orbital_mechanics.transfer_inclination_continuous import transfer_inclination_continuous
from ssapy_toolkit.constants import RGEO, VGEO
from ssapy_toolkit.plots.figpath import figpath  # [36]


def run_case(name, r0, v0, a_thrust, i_target):
    rf, vf, tf = transfer_inclination_continuous(r0=r0, v0=v0, a_thrust=a_thrust, i_target=i_target)
    np.set_printoptions(precision=6, suppress=True)
    print(f"\n=== {name} ===")
    print(f"Final position (m): {rf}")
    print(f"Final velocity (m/s): {vf}")
    tf_scalar = float(np.asarray(tf).reshape(-1)[-1])
    print(f"Time to reach inclination: {tf_scalar:.1f} s")
    return rf, vf, tf


def main():
    r0 = np.array([RGEO, 0.0, 0.0])
    v0 = np.array([0.0, VGEO, 0.0])
    a_thrust = 1.0

    try:
        return run_case(
            name="demo_gallery/figures/transters_demo_inclination_negative_target",
            r0=r0,
            v0=v0,
            a_thrust=a_thrust,
            i_target=np.radians(-45.0),
        )
    except ValueError as err:
        print(f"Error: {err}")
        return None


if __name__ == "__main__":
    main()