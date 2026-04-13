from ssapy_toolkit import figpath, Orbit, RGEO, Time, orbit_plot_xy, orbit_plot, rv
import numpy as np
from tqdm import tqdm

print("True Anomalies")
t0 = Time(0, format='gps')
rs = []
for trueAnomaly in np.arange(0, 360, 5):
    orbit = Orbit.fromKeplerianElements(a=RGEO, e=.4, i=0, pa=0, raan=0, trueAnomaly=np.radians(trueAnomaly), t=t0)
    # r, v, t = quickint(orbit)

    rs.append(orbit.r)
orbit_plot_xy(rs, show=False, save_path=figpath("tests/ssapy_orbit_sampling_trueAnomaly"))

print("Time sampling")
rs = []
for t in tqdm(np.arange(0, orbit.period, 600)):
    orbit_new = orbit.at(t)

    rs.append(orbit_new.r)

orbit_plot(np.array(rs), show=False, save_path=figpath("tests/ssapy_orbit_sampling_time"))


print("Full orbit sampled.")
rs, v = rv(orbit, np.arange(0, orbit.period, 600))
orbit_plot(rs, show=False, save_path=figpath("tests/ssapy_orbit_object"))


print("Complete.")
