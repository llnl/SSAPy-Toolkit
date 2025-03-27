import matplotlib.pyplot as plt
from yeager_utils import transfer_shooter, transfer_hohmann, transfer_lambertian, RGEO, Time
from ssapy import Orbit

# Define initial and final orbit Keplerian elements
t0 = Time("2025-01-01").gps

orbit1 = Orbit.fromKeplerianElements(*[1 * RGEO, 0.0, 0.0, 0, 0.0, 0], t=t0)
orbit2 = Orbit.fromKeplerianElements(*[0.9 * RGEO, 0.0, 0.0, 0.0, 0.0, 10], t=t0)

# Compute Hohmann transfer using the function with plot=False
result = transfer_hohmann(orbit1, orbit2, plot=True)
fig = result['fig']
plt.show()

try:
    result = transfer_lambertian(orbit1, orbit2, plot=True)
    fig = result['fig']
    plt.show()
except Exception:
    pass

result = transfer_shooter(orbit1, orbit2, plot=True)
fig = result['fig']
plt.show()
