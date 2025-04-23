import numpy as np
from yeager_utils import leapfrog, groundtrack_dashboard, surface_rv

r0, v0 = surface_rv(lon=-90, lat=0)

t_max = 3000.0
dt = 1.0
t = np.arange(0, t_max + dt, dt)
n_steps = len(t)

radial_active = np.zeros(n_steps, dtype=bool)
velocity_active = np.zeros(n_steps, dtype=bool)
plane_active = np.zeros(n_steps, dtype=bool)
perp_active = np.zeros(n_steps, dtype=bool)

radial_active[:60 * 10] = True

radial_thrust = 30
velocity_thrust = 10
perp_thrust = 0
plane_thrust = 30

r, v = leapfrog(r0, v0, t, radial_thrust=radial_thrust, radial_active=radial_active,
                velocity_thrust=velocity_thrust, velocity_active=velocity_active, 
                plane_thrust=plane_thrust, plane_active=plane_active,
                perp_thrust=perp_thrust, perp_active=perp_active)

groundtrack_dashboard(r[:, 0], r[:, 1], r[:, 2], t)