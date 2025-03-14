import matplotlib.pyplot as plt
from yeager_utils import hohmann_transfer, lambertian_transfer, RGEO, Time, EARTH_RADIUS, ae_from_periap, np
from ssapy import Orbit

# Define initial and final orbit Keplerian elements
t0 = Time("2025-01-01")

a, e = ae_from_periap(EARTH_RADIUS + 1000e3, EARTH_RADIUS + 1000e3)
orbit1 = Orbit.fromKeplerianElements(*[a, e, 0.0, 0, 0.0, np.pi], t=t0)
a, e = ae_from_periap(RGEO, RGEO)
orbit2 = Orbit.fromKeplerianElements(*[a, e, 0.0, 0, 0.0, 0], t=t0)

# Compute Hohmann transfer using the function with plot=False
result = hohmann_transfer(orbit1, orbit2, plot=True)
fig = result['fig']
plt.show()

result = lambertian_transfer(orbit1, orbit2, plot=True)
fig = result['fig']
plt.show()
