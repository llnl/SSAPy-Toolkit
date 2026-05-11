import numpy as np

from ssapy_toolkit.Orbital_Mechanics.transfer_inclination_continuous import transfer_inclination_continuous
from ssapy_toolkit.constants import RGEO, VGEO
from ssapy_toolkit.Plots.figpath import figpath  # [36]


def run_case(name, r0, v0, a_thrust, i_target):
    rf, vf, tf = transfer_inclination_continuous(r0=r0, v0=v0, a_thrust=a_thrust, i_target=i_target)
    np.set_printoptions(precision=6, suppress=True)
    print(f"\n=== {name} ===")
    print(f"Final position (m): {rf}")
    print(f"Final velocity (m/s): {vf}")
    print(f"Time to reach inclination: {tf:.1f} s")
    return rf, vf, tf


def main():
    r0 = np.array([RGEO, 0.0, 0.0])
    v0 = np.array([0.0, VGEO, 0.0])
    a_thrust = 1.0

    try:
        return run_case(
            name="tests/transters_demo_inclination_negative_target",
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