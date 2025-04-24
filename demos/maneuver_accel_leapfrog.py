import numpy as np
import astropy.units as u
from yeager_utils import leapfrog, accel_deltav, groundtrack_dashboard, surface_rv, launch_pads, RGEO, VGEO, EARTH_MU, hkoe, kepler_to_state

# r0 = [42164e3, 0, 0]  # GEO radius in meters
# v0 = [0, np.sqrt(EARTH_MU / 42164e3), 0]  # Circular velocity
a, e, i, pa, raan, ta = hkoe(RGEO, 0.3, 40, 0, 0, 0)
print(a, e, i, pa, raan, ta)
r0, v0 = kepler_to_state(a, e, i, pa, raan, ta)
print(r0, v0)
t = np.arange(0, 3600 * 24 * 1, 10)  # 6 hours, 10s steps

profile1 = accel_deltav(dv=-200.0, thrust_accel=0.5, center_idx=100)
profile2 = accel_deltav(dv=-1000.0, thrust_accel=0.1, center_idx=200*60)

profiles = [profile1, profile2]
r, v = leapfrog(
    r0,
    v0,
    t,
    velocity=profiles,
    circular=[10, 200*60]
)
groundtrack_dashboard(r[:, 0], r[:, 1], r[:, 2], t, save_path="/home/yeager7/yeager_utils/demos/images/testing_leapfrog_maneuvers.jpg")
