#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

GM_EARTH = 3.986004418e14
R_EARTH = 6.371e6
OMEGA_EARTH = 7.2921159e-5

def geocentric_position(lat_deg, lon_deg, altitude=0):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    r = R_EARTH + altitude
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z])

def initial_velocity_due_to_earth_rotation(lat_deg, lon_deg):
    r = geocentric_position(lat_deg, lon_deg)
    omega_vec = np.array([0, 0, OMEGA_EARTH])
    return np.cross(omega_vec, r)

def earth_gravity_acceleration(r):
    norm_r = np.linalg.norm(r)
    return -GM_EARTH / norm_r**3 * r

def thrust_acceleration(stage, t, state, stage_params):
    mass = state['mass']
    if mass < 1e-6:
        return np.zeros(3)
    thrust = stage_params['thrust']
    a_mag = thrust / mass
    direction = stage_params['direction_func'](t, state)
    return a_mag * np.array(direction)

def default_direction_vertical(t, state):
    r = state['r']
    return r / np.linalg.norm(r)

def gravity_turn_direction(t, state):
    T = 120.0
    alpha = min(t / T, 1.0)
    vertical = state['r'] / np.linalg.norm(state['r'])
    lat = np.deg2rad(state.get('launch_lat', 28.3922))
    lon = np.deg2rad(state.get('launch_lon', -80.6077))
    east = np.array([-np.sin(lon), np.cos(lon), 0])
    east /= np.linalg.norm(east)
    direction = (1 - alpha) * vertical + alpha * east
    return direction / np.linalg.norm(direction)

def leo_insertion_direction(t, state):
    v = state['v']
    if np.linalg.norm(v) > 0:
        return v / np.linalg.norm(v)
    else:
        return np.array([1, 0, 0])

def circularization_direction(t, state):
    r_vec = state['r']
    v_vec = state['v']
    r_norm = np.linalg.norm(r_vec)
    rhat = r_vec / r_norm
    v_rad = np.dot(v_vec, rhat)
    v_rad_vec = v_rad * rhat
    v_tan = v_vec - v_rad_vec
    v_tan_norm = np.linalg.norm(v_tan)
    v_circ = np.sqrt(GM_EARTH / r_norm)
    if v_tan_norm > 1e-6:
        tan_dir = v_tan / v_tan_norm
    else:
        tan_dir = np.cross(rhat, np.array([0, 0, 1]))
        if np.linalg.norm(tan_dir) < 1e-6:
            tan_dir = np.array([1, 0, 0])
        tan_dir /= np.linalg.norm(tan_dir)
    u = -v_rad_vec + (v_circ - v_tan_norm) * tan_dir
    if np.linalg.norm(u) < 1e-6:
        u = tan_dir
    return u / np.linalg.norm(u)

def leapfrog_integrator(acc_func, state0, t0, tf, dt, stage_params):
    times = np.arange(t0, tf + dt, dt)
    N = len(times)
    traj = np.zeros((N, 3))
    vels = np.zeros((N, 3))
    masses = np.zeros(N)
    r = state0['r']
    v = state0['v']
    m = state0['mass']
    dry_mass = state0.get('dry_mass', 100000)
    traj[0] = r
    vels[0] = v
    masses[0] = m
    a = total_acceleration(r, v, m, t0, stage_params)
    v_half = v + 0.5 * dt * a
    for i in range(1, N):
        t = times[i-1]
        r = r + dt * v_half
        burn_rate = stage_params.get('burn_rate', 0)
        m = max(m - burn_rate * dt, dry_mass)
        a = total_acceleration(r, v_half, m, times[i], stage_params)
        v_half = v_half + dt * a
        traj[i] = r
        vels[i] = v_half - 0.5 * dt * a
        masses[i] = m
    return times, traj, vels, masses

def total_acceleration(r, v, m, t, stage_params):
    state = {'r': r, 'v': v, 'mass': m}
    for key in ['launch_lat', 'launch_lon']:
        if key in stage_params:
            state[key] = stage_params[key]
    grav = earth_gravity_acceleration(r)
    thrust = thrust_acceleration(stage_params['stage_name'], t, state, stage_params)
    return grav + thrust

def state_to_latlon(r):
    x, y, z = r
    r_norm = np.linalg.norm(r)
    lat = np.arcsin(z / r_norm)
    lon = np.arctan2(y, x)
    return np.rad2deg(lat), np.rad2deg(lon)

def simulate_stage(stage_params, state0, dt):
    t0 = 0
    tf = stage_params['duration']
    times, traj, vels, masses = leapfrog_integrator(total_acceleration, state0, t0, tf, dt, stage_params)
    final_state = {'r': traj[-1], 'v': vels[-1], 'mass': masses[-1]}
    print(f"\n{stage_params}\nFinal state: {final_state}")
    return times, traj, vels, masses, final_state

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)
    x_center = np.mean(x_limits)
    y_center = np.mean(y_limits)
    z_center = np.mean(z_limits)
    ax.set_xlim3d([x_center - max_range/2, x_center + max_range/2])
    ax.set_ylim3d([y_center - max_range/2, y_center + max_range/2])
    ax.set_zlim3d([z_center - max_range/2, z_center + max_range/2])

def plot_3d_trajectory(traj_total):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj_total[:, 0] * 1e-3, traj_total[:, 1] * 1e-3, traj_total[:, 2] * 1e-3, label='Trajectory')
    ax.set_title('3D Launch Trajectory (km)', fontsize=20)
    ax.set_xlabel('X (km)', fontsize=16)
    ax.set_ylabel('Y (km)', fontsize=16)
    ax.set_zlabel('Z (km)', fontsize=16)
    ax.legend(fontsize=14)
    set_axes_equal(ax)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    ax.set_xticks([xlim[0], 0, xlim[1]])
    ax.set_yticks([ylim[0], 0, ylim[1]])
    ax.set_zticks([zlim[0], 0, zlim[1]])

def plot_ground_track(traj_total):
    lats = []
    lons = []
    for pos in traj_total:
        lat, lon = state_to_latlon(pos)
        lats.append(lat)
        lons.append(lon)
    lons = np.unwrap(np.radians(lons))
    lons = np.degrees(lons)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True)
    ax.plot(lons, lats, color='red', marker='o', markersize=4,
            transform=ccrs.PlateCarree(), label='Ground Track')
    ax.set_title("Ground Track over Earth's Map")
    ax.legend()

def plot_speed_and_mass(times_total, vels_total, masses_total, traj_total):
    speed = np.linalg.norm(vels_total, axis=1)
    altitude = np.linalg.norm(traj_total, axis=1) - R_EARTH
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    ax1.plot(times_total, altitude * 1e-3)
    ax1.set_ylabel('Altitude (km)')
    ax1.set_title('Altitude vs Time')
    ax2.plot(times_total, speed)
    ax2.set_ylabel('Speed (m/s)')
    ax2.set_title('Flight Speed vs Time')
    ax3.plot(times_total, masses_total)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Mass (kg)')
    ax3.set_title('Fuel/Mass vs Time')
    plt.tight_layout()

def calculate_required_fuel(stages, dry_mass, dt, tol=1e-3, max_iter=50):
    F_low = 0.0
    F_high = 10 * dry_mass
    for _ in range(max_iter):
        F_candidate = (F_low + F_high) / 2
        initial_mass = dry_mass + F_candidate
        state = {'r': initial_state_r, 'v': initial_state_v, 'mass': initial_mass, 'dry_mass': dry_mass}
        success = True
        for stage in stages:
            times, traj, vels, masses, state = simulate_stage(stage, state, dt)
            if np.abs(state['mass'] - dry_mass) < 1e-2:
                success = False
                break
        if success:
            F_high = F_candidate
        else:
            F_low = F_candidate
        if np.abs(F_high - F_low) < tol:
            break
    return F_high

initial_state_r = geocentric_position(28.3922, -80.6077)
initial_state_v = initial_velocity_due_to_earth_rotation(28.3922, -80.6077)

def main():
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'legend.fontsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16
    })
    dt = 0.1
    launch_lat = 28.3922
    launch_lon = -80.6077

    stage1_params = {
        'duration': 60.0,
        'thrust': 8.0e6,
        'burn_rate': 2000,
        'direction_func': default_direction_vertical,
        'stage_name': 'vertical',
        'launch_lat': launch_lat,
        'launch_lon': launch_lon
    }
    stage2_params = {
        'duration': 120.0,
        'thrust': 7.0e6,
        'burn_rate': 2000,
        'direction_func': gravity_turn_direction,
        'stage_name': 'gravity_turn',
        'launch_lat': launch_lat,
        'launch_lon': launch_lon
    }
    stage3_params = {
        'duration': 60.0,
        'thrust': 5.0e6,
        'burn_rate': 1500,
        'direction_func': leo_insertion_direction,
        'stage_name': 'leo_insertion'
    }
    circularization_params = {
        'duration': 40.0,
        'thrust': 4.0e6,
        'burn_rate': 1000,
        'direction_func': circularization_direction,
        'stage_name': 'circularization',
        'launch_lat': launch_lat,
        'launch_lon': launch_lon
    }
    stages = [stage1_params, stage2_params, stage3_params, circularization_params]
    dry_mass = 100000
    required_fuel = calculate_required_fuel(stages, dry_mass, dt)
    initial_mass = dry_mass + required_fuel
    print("Required fuel (kg):", required_fuel)
    r0 = geocentric_position(launch_lat, launch_lon)
    v0 = initial_velocity_due_to_earth_rotation(launch_lat, launch_lon)
    state = {'r': r0, 'v': v0, 'mass': initial_mass, 'dry_mass': dry_mass}
    times1, traj1, vels1, masses1, state = simulate_stage(stage1_params, state, dt)
    times2, traj2, vels2, masses2, state = simulate_stage(stage2_params, state, dt)
    times3, traj3, vels3, masses3, state = simulate_stage(stage3_params, state, dt)
    times4, traj4, vels4, masses4, state = simulate_stage(circularization_params, state, dt)
    T_orbit = 2 * np.pi * np.sqrt(np.linalg.norm(state['r'])**3 / GM_EARTH)
    orbit_params = {
        'duration': T_orbit,
        'thrust': 0,
        'burn_rate': 0,
        'direction_func': lambda t, state: np.array([0, 0, 0]),
        'stage_name': 'orbit'
    }
    times5, traj5, vels5, masses5, _ = simulate_stage(orbit_params, state, dt)
    traj_total = np.vstack((traj1, traj2, traj3, traj4, traj5))
    vels_total = np.vstack((vels1, vels2, vels3, vels4, vels5))
    masses_total = np.concatenate((masses1, masses2, masses3, masses4, masses5))
    times_total = np.concatenate((times1, times1[-1] + times2, times1[-1] + times2[-1] + times3,
                                   times1[-1] + times2[-1] + times3[-1] + times4,
                                   times1[-1] + times2[-1] + times3[-1] + times4[-1] + times5))
    plot_3d_trajectory(traj_total)
    plot_ground_track(traj_total)
    plot_speed_and_mass(times_total, vels_total, masses_total, traj_total)
    plt.show()

if __name__ == '__main__':
    main()
