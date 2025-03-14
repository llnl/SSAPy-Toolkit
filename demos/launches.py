import numpy as np
import matplotlib.pyplot as plt
from yeager_utils import EARTH_RADIUS, EARTH_MU, launch_pads, Time, TimeDelta
from ssapy.plotUtils import find_file
from PIL import Image

# Simulation start time and launch pad selection.
time_of_launch = Time("2025-1-1")
pad_name = "Kennedy Space Center LC-39A"
G0 = 9.81

# Falcon 9–like first stage parameters.
ROCKET = {
    "m0": 549054.0,      # Initial mass at liftoff (kg)
    "m_fuel": 518800.0,  # Fuel mass (kg) [m0 minus dry mass]
    "thrust": 7.6e6,     # Sea–level thrust (N)
    "Isp": 282.0,        # Sea–level specific impulse (s)
    "burn_time": 280.0,  # Burn time of the first stage (s)
    "dt": 0.1,           # Time step for integration (s)
}
ROCKET["ve"] = ROCKET["Isp"] * G0
ROCKET["mdot"] = ROCKET["thrust"] / ROCKET["ve"]
ROCKET["t_max"] = min(ROCKET["burn_time"], ROCKET["m_fuel"] / ROCKET["mdot"])


def launchpad_positions():
    positions = []
    for pad in launch_pads.values():
        lat, lon = np.deg2rad(pad["latitude"]), np.deg2rad(pad["longitude"])
        x = EARTH_RADIUS * np.cos(lat) * np.cos(lon)
        y = EARTH_RADIUS * np.cos(lat) * np.sin(lon)
        z = EARTH_RADIUS * np.sin(lat)
        positions.append([x, y, z])
    return np.array(positions)


def derivatives(state, t):
    """Compute state derivatives using a simple gravity turn."""
    x, y, z, vx, vy, vz, m = state
    r_vec = np.array([x, y, z])
    r = np.linalg.norm(r_vec)
    v_vec = np.array([vx, vy, vz])
    g = EARTH_MU / r**2
    gx, gy, gz = -g * (x / r), -g * (y / r), -g * (z / r)

    # Apply thrust only if fuel remains and t is within burn time.
    if t < ROCKET["t_max"] and m > (ROCKET["m0"] - ROCKET["m_fuel"]):
        # Define pitch-over: for the first few seconds, thrust points vertically.
        pitch_over_time = 10.0  # seconds after liftoff when pitch-over begins
        vertical_dir = r_vec / r  # local vertical direction
        # Use a safe fallback if speed is extremely low.
        if np.linalg.norm(v_vec) < 1e-3:
            horizontal_dir = vertical_dir
        else:
            horizontal_dir = v_vec / np.linalg.norm(v_vec)
        if t < pitch_over_time:
            thrust_dir = vertical_dir
        else:
            # Gradually rotate thrust from vertical to aligning with the velocity.
            weight = min((t - pitch_over_time) / (ROCKET["t_max"] - pitch_over_time),
                         1.0)
            thrust_dir = (1 - weight) * vertical_dir + weight * horizontal_dir
            thrust_dir /= np.linalg.norm(thrust_dir)
        thrust_acc = ROCKET["thrust"] / m
        ax_t, ay_t, az_t = thrust_acc * thrust_dir
        dm_dt = -ROCKET["mdot"]
    else:
        ax_t, ay_t, az_t = 0.0, 0.0, 0.0
        dm_dt = 0.0
    return np.array([vx, vy, vz, gx + ax_t, gy + ay_t, gz + az_t, dm_dt])


def rk4_step(state, t, dt):
    k1 = derivatives(state, t)
    k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = derivatives(state + dt * k3, t + dt)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_trajectory(state0, time_of_launch, max_time=1000.0):
    t = 0.0
    states = [state0]
    times = [time_of_launch]
    dt = ROCKET["dt"]
    dt_astropy = TimeDelta(dt, format="sec")
    while t < max_time:
        state_new = rk4_step(states[-1], t, dt)
        # Stop simulation if the rocket would hit Earth.
        if np.linalg.norm(state_new[:3]) < EARTH_RADIUS:
            break
        states.append(state_new)
        times.append(times[-1] + dt_astropy)
        t += dt
    states = np.array(states)
    r = states[:, :3]
    v = states[:, 3:6]
    t_arr = np.array(times)
    return r, v, t_arr


# Compute initial conditions at the chosen launch pad.
lat, lon = np.deg2rad(launch_pads[pad_name]["latitude"]), np.deg2rad(
    launch_pads[pad_name]["longitude"]
)
r0 = EARTH_RADIUS * np.array(
    [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
)
omega = 7.2921159e-5
v0 = np.array([-omega * r0[1], omega * r0[0], 0])
state0 = np.concatenate((r0, v0, [ROCKET["m0"]]))

launch_positions = launchpad_positions()
r, v, t_arr = simulate_trajectory(state0, time_of_launch, max_time=1000.0)
altitude = np.linalg.norm(r, axis=1) - EARTH_RADIUS

# Plot altitude vs. time.
plt.figure(figsize=(7, 6))
plt.plot(np.arange(len(r)) * ROCKET["dt"], altitude / 1000, label="Altitude")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (km)")
plt.title("Altitude vs Time")
plt.legend()
plt.grid(True)
plt.show()

# 3D plot of the trajectory.
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
earth_png = Image.open(find_file("earth", ext=".png"))
earth_png = earth_png.resize((5400 // 5, 2700 // 5))
bm = np.array(earth_png) / 256.0
lons = np.linspace(-180, 180, bm.shape[1]) * np.pi / 180
lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi / 180
shrink = 1.0
mesh_x = np.outer(np.cos(lons), np.cos(lats)).T * EARTH_RADIUS / 1e3 * shrink
mesh_y = np.outer(np.sin(lons), np.cos(lats)).T * EARTH_RADIUS / 1e3 * shrink
mesh_z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T * EARTH_RADIUS / 1e3 * shrink
ax.plot_surface(mesh_x, mesh_y, mesh_z, rstride=4, cstride=4,
                facecolors=bm, shade=False)

# Plot launch pad positions.
launch_positions_km = launch_positions / 1e3
for i, pos in enumerate(launch_positions_km):
    ax.scatter(pos[0], pos[1], pos[2], c="red", s=100,
               label="Launch Pads" if i == 0 else "")
# Plot the rocket trajectory.
ax.scatter(r[:, 0] / 1e3, r[:, 1] / 1e3, r[:, 2] / 1e3, c="yellow", s=10,
           label="Rocket Path")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title("Rocket Trajectory in ITRF")
ax.legend()
ax.set_box_aspect([1, 1, 1])  # For an equal aspect ratio

# Compute final orbit parameters.
r_final, v_final = r[-1], v[-1]
r_mag, v_mag = np.linalg.norm(r_final), np.linalg.norm(v_final)
energy = v_mag**2 / 2 - EARTH_MU / r_mag
a = -EARTH_MU / (2 * energy)
h_vec = np.cross(r_final, v_final)
h = np.linalg.norm(h_vec)
e = np.sqrt(1 + (2 * energy * h**2) / EARTH_MU**2)
perigee = a * (1 - e)
perigee_altitude = perigee - EARTH_RADIUS

print(f"Final Altitude: {altitude[-1] / 1000:.2f} km")
print(f"Final Speed: {v_mag / 1000:.2f} km/s")
print(f"Semi-major axis: {a / 1000:.2f} km")
print(f"Eccentricity: {e:.3f}")
print(f"Perigee Altitude: {perigee_altitude / 1000:.2f} km")
