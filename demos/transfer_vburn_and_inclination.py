import numpy as np
from yeager_utils import transfer_velocity_and_inclination_continuous, EARTH_RADIUS

if __name__ == "__main__":
    # Sample values for demo
    r0 = np.array([EARTH_RADIUS + 500e3, 0.0, 0.0])  # 500 km altitude
    v0 = np.array([0.0, 7.6e3, 0.0])  # Roughly circular LEO speed
    i_target = np.radians(15.0)  # Target inclination in radians
    a_thrust = 1  # Low thrust acceleration in m/s²

    transfer_velocity_and_inclination_continuous(r0, v0, i_target, a_thrust, plot=True)