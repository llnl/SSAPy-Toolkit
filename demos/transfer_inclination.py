import numpy as np
from yeager_utils import transfer_inclination_continuous


# Example usage
if __name__ == "__main__":
    r0 = np.array([42_000_000.0, 0.0, 0.0])   # m
    v0 = np.array([0.0, 4000.0, 0.0])        # m/s
    i_target = np.radians(89)               # Target inclination
    a_thrust = 1.0                            # m/s^2

    try:
        r_f, v_f, t_f = transfer_inclination_continuous(
            r0, v0, i_target, a_thrust, plot=True
        )
        print(f"Final position: {r_f} m")
        print(f"Final velocity: {v_f} m/s")
        print(f"Time to reach inclination: {t_f:.1f} s")
    except ValueError as err:
        print(f"Error: {err}")
