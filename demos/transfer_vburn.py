import numpy as np
from yeager_utils import transfer_velocity_continuous, EARTH_RADIUS, RGEO, VGEO, figpath


if __name__ == "__main__":
    # Sample values for demo
    r0 = np.array([RGEO, 0.0, 0.0])  # 500 km altitude
    v0 = np.array([0.0, VGEO, 0.0])  # Roughly circular LEO speed
    v_target = 400
    a_thrust = 1.0  # Low thrust acceleration in m/s²

    # print("Plotting v target")
    # transfer_velocity_continuous(r0, v0, v_target, a_thrust, plot=True, save_path=figpath("demo_velocity_burn_positive_v.jpg"))

    # print("Plotting no v target")
    # transfer_velocity_continuous(r0, v0, max_time=100, plot=True, save_path=figpath("demo_velocity_burn_no_v.jpg"))

    print("Plotting negative v target")
    v_target = -400
    transfer_velocity_continuous(r0, v0, v_target=v_target, plot=True, save_path=figpath("demo_velocity_burn_negative_v.jpg"))

    print("Plotting negative v target lower accel")
    v_target = -400
    a_thrust = 0.05
    transfer_velocity_continuous(r0, v0, v_target=v_target, a_thrust=a_thrust, max_time=3*3600, plot=True, save_path=figpath("demo_velocity_burn_negative_v_small_accel.jpg"))
