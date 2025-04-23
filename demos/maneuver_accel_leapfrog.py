import numpy as np
import astropy.units as u
from yeager_utils import leapfrog, groundtrack_dashboard, surface_rv, launch_pads, RGEO, VGEO, EARTH_MU

r0 = [42164e3, 0, 0]  # GEO radius in meters
v0 = [0, np.sqrt(EARTH_MU / 42164e3), 0]  # Circular velocity

t = np.arange(0, 3600 * 24 * 10, 10)  # 6 hours, 10s steps
r, v = leapfrog(
    r0,
    v0,
    t,
    velocity=[10, 0, 100]
)
groundtrack_dashboard(r[:, 0], r[:, 1], r[:, 2], t, save_path="images/testing_leapfrog_maneuvers.jpg")
