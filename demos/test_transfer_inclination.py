import numpy as np
from ssapy_toolkit import transfer_inclination_continuous, RGEO, VGEO, figpath


def final_state(x):
    """
    Return the final state vector from a history or pass through a single state.
    - If x is (N,3) -> returns x[-1] (shape (3,))
    - If x is (3,)   -> returns x
    - If x is scalar -> returns scalar
    - If x is (N,)   -> returns x[-1]
    """
    a = np.asarray(x)
    if a.ndim == 0:
        return a
    if a.ndim == 1:
        return a[-1]
    return a[-1]


def final_time(t):
    """
    Return a float for the final time, regardless of whether t is a scalar or array.
    """
    a = np.asarray(t)
    if a.ndim == 0:
        return float(a)
    return float(np.ravel(a)[-1])


def run_case(name, r0, v0, a_thrust, **kwargs):
    """
    Run a single case of transfer_inclination_continuous and print final results.
    """
    r_f, v_f, t_f = transfer_inclination_continuous(
        r0,
        v0,
        a_thrust=a_thrust,
        plot=True,
        save_path=figpath(name),
        **kwargs
    )

    rf = final_state(r_f)
    vf = final_state(v_f)
    tf = final_time(t_f)

    # Tidy printing
    np.set_printoptions(precision=6, suppress=True)

    print(f"\n=== {name} ===")
    print(f"Final position (m): {rf}")
    print(f"Final velocity (m/s): {vf}")
    print(f"Time to reach inclination: {tf:.1f} s")

    return rf, vf, tf


# Example usage
if __name__ == "__main__":
    r0 = np.array([RGEO, 0.0, 0.0])  # m
    v0 = np.array([0.0, VGEO, 0.0])  # m/s
    a_thrust = 1.0                   # m/s^2

    try:
        # Target inclination: +45 deg
        run_case(
            name="tests/transters_demo_inclination_target_burn",
            r0=r0,
            v0=v0,
            a_thrust=a_thrust,
            i_target=np.radians(45.0)
        )

        # Allocate +500 m/s for maneuver
        run_case(
            name="tests/transters_demo_inclination_target_dv",
            r0=r0,
            v0=v0,
            a_thrust=a_thrust,
            delta_v=500.0
        )

        # Allocate -500 m/s (should still run; function may interpret as opposite sense)
        run_case(
            name="tests/transters_demo_inclination_negative_dv",
            r0=r0,
            v0=v0,
            a_thrust=a_thrust,
            delta_v=-500.0
        )

        # Target inclination: -45 deg
        run_case(
            name="tests/transters_demo_inclination_negative_target",
            r0=r0,
            v0=v0,
            a_thrust=a_thrust,
            i_target=np.radians(-45.0)
        )

    except ValueError as err:
        print(f"Error: {err}")
