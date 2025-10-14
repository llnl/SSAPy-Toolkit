from ssapy import Orbit
from ssapy.constants import RGEO
from ssapy.utils import get_times
from ssapy import Time
from yeager_utils import to_gps, orbit_plot, burn_to_deltav, deltav_to_burn
import numpy as np
import matplotlib.pyplot as plt
print("Modules imported.")

burn_vector = np.array([50.0, 0.0, 0.0])
t0 = Time("2025-1-1T12:00:00.000", scale='utc')
times = to_gps(get_times(duration=(12, 'hour'), freq=(1, 's'), t0=t0))
t1 = 30000
t2 = t1 + 100

a = RGEO
e = 0
i = 0
pa = 0
raan = 0
trueAnomaly = 0

kElements = [a, e, i, pa, raan, trueAnomaly]

orbit = Orbit.fromKeplerianElements(*kElements, t0)

burn_times = times[t1:t2]
result = burn_to_deltav(orbit, burn_times, burn_vector)
print(result)

orbit_plot([result['r_continuous'], result['r_instantaneous']], times, show=True)

plt.figure()
plt.plot(result['r_continuous'][:, 0] / 1e3, result['r_continuous'][:, 1] / 1e3, label='Burn')
plt.plot(result['r_instantaneous'][:, 0] / 1e3, result['r_instantaneous'][:, 1] / 1e3, label='Impulse')
plt.legend()

plt.figure()
plt.plot(burn_times - burn_times[0], np.linalg.norm(result['r_continuous'] - result['r_instantaneous'], axis=-1) / 1e3)
plt.ylabel("Distance between trajectories [km]")

result = deltav_to_burn(orbit, burn_times, burn_vector)
print(result)

orbit_plot([result['r_continuous'], result['r_instantaneous']], times, show=True)

plt.figure()
plt.plot(result['r_continuous'][:, 0] / 1e3, result['r_continuous'][:, 1] / 1e3, label='Burn')
plt.plot(result['r_instantaneous'][:, 0] / 1e3, result['r_instantaneous'][:, 1] / 1e3, label='Impulse')
plt.legend()

plt.figure()
plt.plot(burn_times - burn_times[0], np.linalg.norm(result['r_continuous'] - result['r_instantaneous'], axis=-1) / 1e3)
plt.ylabel("Distance between trajectories [km]")
