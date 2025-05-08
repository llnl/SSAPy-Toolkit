import matplotlib.pyplot as plt
from yeager_utils import transfer_coplanar_continuous, EARTH_RADIUS, RGEO, Time, hkoe
from ssapy import Orbit, rv

# Define initial and final orbit Keplerian elements
t0 = Time("2025-01-01").gps

orbit1 = Orbit.fromKeplerianElements(*hkoe([10000e3 + EARTH_RADIUS, 0.1, 0.0, 0.0, 0.0, 0.0]), t=t0)
orbit2 = Orbit.fromKeplerianElements(*hkoe([20000e3 + EARTH_RADIUS, 0.0, 0.0, 0, 0.0, 90.0]), t=t0)
print("Running coplanar")
transfer_coplanar_continuous(r1=orbit1.r, v1=orbit1.v, r2=orbit2.r, plot=True)
