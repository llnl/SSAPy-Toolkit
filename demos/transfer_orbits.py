import matplotlib.pyplot as plt
from yeager_utils import hohmann_transfer, lambertian_transfer, RGEO, Time, EARTH_RADIUS, ae_from_periap, np, orbit_plot_rv
from ssapy import Orbit

# Define initial and final orbit Keplerian elements
t0 = Time("2025-01-01").gps

orbit1 = Orbit.fromKeplerianElements(*[RGEO, 0.1, 0.0, 0, 0.0, 0], t=t0)
orbit2 = Orbit.fromKeplerianElements(*[0.9 * RGEO, 0.1, 0.0, 0.0, 0.0, 0.0], t=t0 + 6000)

# # Compute Hohmann transfer using the function with plot=False
# result = hohmann_transfer(orbit1, orbit2, plot=True)
# fig = result['fig']
# plt.show()

result = lambertian_transfer(orbit1, orbit2, plot=True)
fig = result['fig']
plt.show()
