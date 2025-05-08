import matplotlib.pyplot as plt
from yeager_utils import EARTH_RADIUS, transfer_shooter, transfer_shooter2, transfer_hohmann, transfer_lambertian, transfer_coplanar, transfer_coplanar, RGEO, Time, hkoe
from ssapy import Orbit, rv

# Define initial and final orbit Keplerian elements
t0 = Time("2025-01-01").gps

orbit1 = Orbit.fromKeplerianElements(*hkoe([10000e3 + EARTH_RADIUS, 0.1, 0.0, 0.0, 0.0, 0.0]), t=t0)
orbit2 = Orbit.fromKeplerianElements(*hkoe([20000e3 + EARTH_RADIUS, 0.0, 90.0, 0, 0.0, 90.0]), t=t0)

# Compute Hohmann transfer using the function with plot=False
print("Running Hohmann")
result = transfer_hohmann(orbit1, orbit2, plot=True)
fig = result['fig']
plt.show()

print("Running Hohmann r1 v1 r2")
result = transfer_hohmann(orbit1.r, orbit1.v, orbit2.r, plot=True)
fig = result['fig']
plt.show()

print("Running Lambertian")
try:
    result = transfer_lambertian(orbit1, orbit2, plot=True)
    fig = result['fig']
    plt.show()
except Exception as err:
    print(err)
    pass

print("Running Lambertian r1 v1 r2")
try:
    result = transfer_lambertian(orbit1.r, orbit1.v, orbit2.r, plot=True)
    fig = result['fig']
    plt.show()
except Exception as err:
    print(err)
    pass


print("Running shooter")
result = transfer_shooter(orbit1, orbit2, plot=True, status=True)
fig = result['fig']
plt.show()

print("Running shooter r1 v1 r2")
result = transfer_shooter(orbit1.r, orbit1.v, orbit2.r, plot=True, status=True)
fig = result['fig']
plt.show()


print("Running coplanar")
result = transfer_coplanar(orbit1, orbit2, plot=True, status=True)
fig = result['fig']
plt.show()
