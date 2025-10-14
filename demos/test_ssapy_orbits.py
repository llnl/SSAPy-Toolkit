from yeager_utils import *
import numpy as np
from tqdm import tqdm

print("True Anomalies")
t0 = Time(0, format='gps')
rs = []
for trueAnomaly in np.arange(0, 360, 5):
    orbit = Orbit.fromKeplerianElements(a=RGEO, e=.4, i=0, pa=0, raan=0, trueAnomaly=np.radians(trueAnomaly), t=t0)
    # r, v, t = quickint(orbit)

    rs.append(orbit.r)
orbit_plot_xy(rs, show=True)

print("Second part")
rs = []
for t in tqdm(np.arange(0, orbit.period, 600)):
    orbit_new = orbit.at(t)

    rs.append(orbit_new.r)
orbit_plot(rs, show=True)


print("Full orbit sampled.")
rs, v = rv(orbit, np.arange(0, orbit.period, 600))
orbit_plot(rs, show=True)


print("Complete.")