import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ssapy_toolkit.Coordinates import ntw_to_gcrf  # Adjust import based on your module structure

# Define a simple circular orbit in GCRF (e.g., equatorial orbit)
R_earth = 6378e3  # Earth radius in meters
mu = 3.986e14     # Earth's gravitational parameter (m^3/s^2)
r_magnitude = R_earth + 500e3  # 500 km altitude

# Position and velocity for a circular orbit in the equatorial plane (x-y plane)
r_center = np.array([r_magnitude, 0.0, 0.0])  # Along x-axis (m)
v_magnitude = np.sqrt(mu / r_magnitude)       # Circular orbit speed
v_center = np.array([0.0, v_magnitude, 0.0])  # Along y-axis (m/s)

# Define a sample delta-v in NTW frame (m/s)
# Example: 100 m/s tangential burn
delta_v_ntw = np.array([0.0, 100.0, 0.0])  # [Normal, Tangential, W-Normal]

# Convert delta-v from NTW to GCRF
delta_v_gcrf = ntw_to_gcrf(delta_v_ntw, r_center, v_center)

# Print results for verification
print("Position (GCRF, m):", r_center)
print("Velocity (GCRF, m/s):", v_center)
print("Delta-v (NTW, m/s):", delta_v_ntw)
print("Delta-v (GCRF, m/s):", delta_v_gcrf)

# Set up 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot origin (Earth center)
ax.scatter([0], [0], [0], color='blue', label='Earth Center', s=100)

# Plot position vector (r_center)
ax.quiver(0, 0, 0, r_center[0], r_center[1], r_center[2], color='green', label='Position (r)', arrow_length_ratio=0.1)

# Plot velocity vector (v_center)
ax.quiver(r_center[0], r_center[1], r_center[2], v_center[0], v_center[1], v_center[2],
          color='red', label='Velocity (v)', arrow_length_ratio=0.1)

# Plot delta-v vector in GCRF (from r_center)
ax.quiver(r_center[0], r_center[1], r_center[2], delta_v_gcrf[0], delta_v_gcrf[1], delta_v_gcrf[2],
          color='purple', label='Delta-v (GCRF)', arrow_length_ratio=0.1)

# Set plot limits (scale down for visibility, since vectors are in meters and m/s)
scale = r_magnitude * 1.5
ax.set_xlim([-scale, scale])
ax.set_ylim([-scale, scale])
ax.set_zlim([-scale, scale])

# Labels and title
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('NTW to GCRF Delta-v Transformation')
ax.legend()

# Equal aspect ratio
ax.set_box_aspect([1, 1, 1])  # Ensures 3D plot is not distorted
