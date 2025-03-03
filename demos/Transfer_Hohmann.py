import numpy as np
import matplotlib.pyplot as plt
from yeager_utils import hohmann_transfer, ssapy_orbit, EARTH_MU, RGEO, Time

# Define initial and final orbit Keplerian elements
t0 = Time("2025-01-01")
elements1 = [2 * RGEO, 0.6, 0.0, np.pi / 6, 0.0, np.pi / 4]  # [a, e, i, Omega, omega, nu]
elements2 = [RGEO, 0.4, 0.0, 0, 0.0, 0.0]

# Compute Hohmann transfer using the function with plot=False
result = hohmann_transfer(elements1, elements2, t0, mu=EARTH_MU, plot=False)

# Extract results from the dictionary
orbit1 = result['initial']         # Initial orbit as computed by the function
orbit2 = result['final']            # Final orbit, adjusted for the transfer
transfer_orbit = result['transfer']  # Transfer orbit
delta_v1 = result['delta_v1']             # First delta-V maneuver
delta_v2 = result['delta_v2']             # Second delta-V maneuver
tof = result['tof']                       # Time of flight
t_to_transfer = result['t_to_transfer']   # Time to transfer window

# Integrate the orbits for plotting
r_traj1, v_traj1, times1 = ssapy_orbit(orbit=orbit1, duration=(orbit1.period, 's'), t0=t0)
r_traj2, v_traj2, times2 = ssapy_orbit(orbit=orbit2, duration=(orbit2.period, 's'), t0=t0)
r_traj_transfer, v_traj_transfer, times_transfer = ssapy_orbit(
    r=transfer_orbit.r,
    v=transfer_orbit.v,
    duration=(transfer_orbit.period / 2, 's'),
    t0=Time(transfer_orbit.t, format='gps')
)

# Plotting
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(r_traj1[:, 0], r_traj1[:, 1], label="Initial Orbit", linestyle="dashed")
ax.plot(r_traj2[:, 0], r_traj2[:, 1], label="Final Orbit", linestyle="dotted")
ax.plot(r_traj_transfer[:, 0], r_traj_transfer[:, 1], label="Transfer Orbit")
ax.scatter([0], [0], color='Blue', marker='o', label="Earth")
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_title("Hohmann Transfer")
ax.legend(loc='upper left')
ax.set_aspect('equal')
plt.show()

# Print results
print(f"Delta-V1: {delta_v1:.2f} m/s")
print(f"Delta-V2: {delta_v2:.2f} m/s")
print(f"Time of Flight: {tof:.2f} s")
print(f"Time to Transfer Window: {t_to_transfer:.2f} s")
