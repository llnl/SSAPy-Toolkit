import numpy as np
from ssapy import Orbit
from yeager_utils import transfer_rendezvous, Time, EARTH_MU, EARTH_RADIUS, RGEO

# Define initial and target orbit radii
r1_mag = RGEO                    # GEO radius
r2_mag = 2 * RGEO                # 2×GEO radius

# Position vectors (circular, equatorial)
r1 = np.array([r1_mag, 0, 0])
r2 = np.array([0, r2_mag, 0])

# Circular velocities
v1_mag = np.sqrt(EARTH_MU / r1_mag)
v2_mag = np.sqrt(EARTH_MU / r2_mag)
v1 = np.array([0, v1_mag, 0])
v2 = np.array([-v2_mag, 0, 0])

# Set same epoch for both orbits
t = Time(0, format='gps')

# Build Orbit objects
orbit1 = Orbit(r=r1, v=v1, t=t)
orbit2 = Orbit(r=r2, v=v2, t=t)

# Run rendezvous calculation
result = transfer_rendezvous(orbit1, orbit2, tol=1.0, status=True, plot=True)

# Output results
print(f"Initial Δv magnitude: {result['|delta_v1|']:.3f} m/s")
print(f"Final Δv magnitude: {result['|delta_v2|']:.3f} m/s")
print(f"Time of flight: {result['tof'] / 60:.2f} minutes")
print(f"Final position error: {result['error']:.3f} m")

# Show plot if available
if 'fig' in result:
    result['fig'].show()
