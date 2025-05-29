import numpy as np
from yeager_utils import transfer_inclination_continuous, RGEO, VGEO, figpath


# Example usage
if __name__ == "__main__":
    r0 = np.array([RGEO, 0.0, 0.0])   # m
    v0 = np.array([0.0, VGEO, 0.0])        # m/s
    i_target = np.radians(45)               # Target inclination
    a_thrust = 1.0                            # m/s^2

    try:
        r_f, v_f, t_f = transfer_inclination_continuous(
            r0, v0, i_target=i_target, a_thrust=a_thrust, plot=True, save_path=figpath('demo_inclination_target_burn')
        )
        print(f"Final position: {r_f} m")
        print(f"Final velocity: {v_f} m/s")
        print(f"Time to reach inclination: {t_f:.1f} s")

        r_f, v_f, t_f = transfer_inclination_continuous(
            r0, v0, delta_v=500, a_thrust=a_thrust, plot=True, save_path=figpath('demo_inclination_target_dv')
        )
        print(f"Final position: {r_f} m")
        print(f"Final velocity: {v_f} m/s")
        print(f"Time to reach inclination: {t_f:.1f} s")

        r_f, v_f, t_f = transfer_inclination_continuous(
            r0, v0, delta_v=-500, a_thrust=a_thrust, plot=True, save_path=figpath('demo_inclination_negative_dv')
        )
        print(f"Final position: {r_f} m")
        print(f"Final velocity: {v_f} m/s")
        print(f"Time to reach inclination: {t_f:.1f} s")

        r_f, v_f, t_f = transfer_inclination_continuous(
            r0, v0, i_target=np.radians(-45), a_thrust=a_thrust, plot=True, save_path=figpath('demo_inclination_negative_target')
        )
        print(f"Final position: {r_f} m")
        print(f"Final velocity: {v_f} m/s")
        print(f"Time to reach inclination: {t_f:.1f} s")

    except ValueError as err:
        print(f"Error: {err}")
