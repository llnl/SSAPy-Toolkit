import numpy as np
import astropy.units as u
from yeager_utils import leapfrog, accel_point_earth, accel_earth_harmonics, accel_deltav, groundtrack_dashboard, surface_rv, launch_pads, RGEO, VGEO, EARTH_MU, hkoe, kepler_to_state

# r0 = [42164e3, 0, 0]  # GEO radius in meters
# v0 = [0, np.sqrt(EARTH_MU / 42164e3), 0]  # Circular velocity
a, e, i, pa, raan, ta = hkoe(1 * RGEO, 0.0, 0, 0, 0, 0)
r0, v0 = kepler_to_state(a, e, i, pa, raan, ta)
t = np.arange(0, 3600 * 24 * 1, 10)  # 6 hours, 10s steps

profile1 = accel_deltav(dv=-200.0, thrust_accel=0.5, center_idx=100)
profile2 = accel_deltav(dv=-1000.0, thrust_accel=1.0, center_idx=int(150*60))

profiles = [profile1, profile2]
r, v = leapfrog(
    r0,
    v0,
    t,
    accel=accel_point_earth,
    velocity=profiles,
    circular=[0.5, 600*60]
)

print(r[-1], v[-1])

groundtrack_dashboard(r, t, save_path="/home/yeager7/yeager_utils/demos/images/testing_leapfrog_maneuvers.jpg")
