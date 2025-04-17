import numpy as np
from yeager_utils import leapfrog, groundtrack_dashboard, surface_rv

# --- Initial Conditions ---
r0, v0 = surface_rv(lon=-90, lat=0)  # Radial speed magnitude (m/s)

t_max = 3000.0  # Total duration (s)
dt = 1.0        # Time step (s)
t = np.arange(0, t_max + dt, dt)  # Time array
n_steps = len(t)

# --- Acceleration Inputs ---
# Create active masks for different directions
radial_active = np.zeros(n_steps, dtype=bool)
velocity_active = np.zeros(n_steps, dtype=bool)
perp_active = np.zeros(n_steps, dtype=bool)  # Not used for now

# Example: Activate radial thrust for the first 600 seconds
radial_active[:60 * 10] = True

# --- Integrate the trajectory ---
radial_thrust = 20  # Radial thrust magnitude
velocity_thrust = 10  # Example: Velocity thrust magnitude
# perp_thrust = 5  # Perpendicular thrust magnitude (commented out for now)

r, v = leapfrog(r0, v0, t, radial_thrust=radial_thrust, 
                velocity_thrust=velocity_thrust, 
                radial_active=radial_active, velocity_active=velocity_active, 
                # perp_thrust=perp_thrust,  # Commented out
                perp_active=perp_active)  # Not used

# --- Plotting ---
groundtrack_dashboard(r[:, 0], r[:, 1], r[:, 2], t)
