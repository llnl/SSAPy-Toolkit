import numpy as np
from ssapy import Orbit
from yeager_utils import transfer_rendezvous, transfer_rendezvous_gradientfree, Time, EARTH_MU, EARTH_RADIUS, RGEO

# Set same epoch for both orbits
t = Time(0, format='gps')

# Build Orbit objects
orbit1 = Orbit.fromKeplerianElements(a=RGEO, e=0.5, i=np.radians(0), pa=0, raan=0, trueAnomaly=0, t=t)
orbit2 = Orbit.fromKeplerianElements(a=2 * RGEO, e=0, i=np.radians(80), pa=0, raan=0, trueAnomaly=np.radians(0), t=t)

result = transfer_rendezvous(orbit1, orbit2, status=True, plot=True)

# Output results
print(f"Initial Δv magnitude: {result['|delta_v1|']:.3f} m/s")
print(f"Final Δv magnitude: {result['|delta_v2|']:.3f} m/s")
print(f"Time of flight: {result['tof'] / 60:.2f} minutes")
print(f"Final position error: {result['error']:.3f} m")

# Show plot if available
if 'fig' in result:
    result['fig'].show()

# result = transfer_rendezvous(orbit1, orbit2, status=True, plot=True)
result = transfer_rendezvous_gradientfree(orbit1, orbit2, status=True, plot=True)

# Output results
print(f"Initial Δv magnitude: {result['|delta_v1|']:.3f} m/s")
print(f"Final Δv magnitude: {result['|delta_v2|']:.3f} m/s")
print(f"Time of flight: {result['tof'] / 60:.2f} minutes")
print(f"Final position error: {result['error']:.3f} m")

# Show plot if available
if 'fig' in result:
    result['fig'].show()
