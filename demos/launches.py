import numpy as np
import matplotlib.pyplot as plt
from yeager_utils import leapfrog, groundtrack_dashboard, surface_rv

# --- Initial Conditions ---
# Launch from 0° lat, 90° W (Cape Canaveral‑ish)
r0, v0 = surface_rv(lon=-90, lat=0)

# --- Time Array ---
t_max = 15 * 60    # simulate ~13 minutes total
dt = 1.0         # 1 s step
t = np.arange(0, t_max + dt, dt)
n_steps = len(t)

# --- LEO‑Reaching Thrust Profile ---
# 1) Vertical radial boost longer: 200 s at ~30 m/s² net
radial_thrust = 30.0
radial_active = t < 200.0

# 2) Gravity‑turn & horizontal burn: from 200 s to 600 s at ~15 m/s²
velocity_thrust = 15.0
velocity_active = (t >= 200.0) & (t < 600.0)

# No perpendicular thrust for now
perp_thrust = 0.0
perp_active = np.zeros(n_steps, dtype=bool)

# --- Integrate Trajectory ---
r, v = leapfrog(
    r0, v0, t,
    radial_thrust=radial_thrust,
    velocity_thrust=velocity_thrust,
    perp_thrust=perp_thrust,
    radial_active=radial_active,
    velocity_active=velocity_active,
    perp_active=perp_active
)

# --- Plot Ground Track & Altitude ---
groundtrack_dashboard(r[:, 0], r[:, 1], r[:, 2], t)
