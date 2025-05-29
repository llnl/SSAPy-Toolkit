import numpy as np
import astropy.units as u
from yeager_utils import leapfrog, groundtrack_dashboard, surface_rv, launch_pads

pad = launch_pads["Kennedy Space Center LC-39A"]
r0, v0 = surface_rv(lon=pad["longitude"], lat=pad["latitude"])

t_max, dt = 180 * 60, 1.0
t = np.arange(0, t_max + dt, dt) * u.s  # attach units here

r, v, fuel = leapfrog(
    r0,
    v0,
    t,
    radial=[30, 0 * u.min, 4 * u.min],
    velocity={"thrust": 20, "start": 4 * u.min, "end": 5 * u.min},
    plane=[0, 5 * u.min, 10 * u.min],
    inclination={"thrust": 0, "start": 60 * u.min, "end": 65 * u.min},
    circular={"thrust": 5, "start": 10 * u.min},
    fuel=False
)
print(fuel)
groundtrack_dashboard(r, t.value, show=True, save_path="images/groundtrack_dashboard.jpg")
