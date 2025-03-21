import numpy as np
import matplotlib.pyplot as plt
from yeager_utils import EARTH_RADIUS, launch_pads, Time, get_times, load_earth_file, save_plot, groundTrack, mkdir
import ssapy
from matplotlib.patches import Circle

time_of_launch = Time("2025-3-21")
pad_name = "Kennedy Space Center LC-39A"


def groundTrackPlot(r, time, launch_pads, save_path=False):
    if isinstance(r, np.ndarray):
        r = [r]
    if isinstance(time, np.ndarray):
        if isinstance(time, list):
            time_list = time
        else:
            time_list = []
            for _ in r:
                time_list.append(time)
    else:
        time_list = [time]

    fig = plt.figure(figsize=(15, 12))
    plt.imshow(load_earth_file(), extent=[-180, 180, -90, 90])

    standard_colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    for i, orbit in enumerate(r):
        if len(r) <= len(standard_colors):
            print(len(r), len(standard_colors))
            color = standard_colors[i]
        else:
            color = plt.cm.rainbow(i / len(r))

        lon, lat, height = groundTrack(orbit, time_list[i])

        lon_deg = np.degrees(lon)
        lat_deg = np.degrees(lat)
        discont_indices = np.where(np.abs(np.diff(lon_deg)) > 179)[0]

        segments = np.split(np.arange(len(lon_deg)), discont_indices + 1)

        for j, segment in enumerate(segments):
            plt.plot(lon_deg[segment], lat_deg[segment], linewidth=8, color=color, label=f"Launch Stage {i+1}" if j == 0 else "")

    for launch_pad in launch_pads:
        plt.scatter(launch_pad[1], launch_pad[0], s=50, color='Red', label=pad_name)

    plt.ylim(-90, 90)
    plt.xlim(-180, 180)
    plt.xlabel("Longitude [deg]", fontsize=24)
    plt.ylabel("Latitude [deg]", fontsize=24)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(fontsize=14)
    plt.show()

    if save_path:
        save_plot(fig, save_path)  # Save ground track plot


def launchpad_positions():
    positions = []
    for pad in launch_pads.values():
        lat, lon = np.deg2rad(pad["latitude"]), np.deg2rad(pad["longitude"])
        x = EARTH_RADIUS * np.cos(lat) * np.cos(lon)
        y = EARTH_RADIUS * np.cos(lat) * np.sin(lon)
        z = EARTH_RADIUS * np.sin(lat)
        positions.append([x, y, z])
    return np.array(positions)


def simulate_launch(pad_name="Kennedy Space Center LC-39A", time_of_launch=Time("2025-1-1"), save_path="/g/g16/yeager7/workdir/yeager_utils/demos/images/launches/"):
    mkdir(save_path)
    ROCKET = {
        "name": 'Falcon 9',         # Name of the rocket
        "m0": 549054.0,            # Initial mass at liftoff (kg)
        "m_fuel": 518800.0,        # Fuel mass (kg) [m0 minus dry mass]
        "thrust": 7.6e6,           # Sea-level thrust (N)
        "Isp": 282.0,              # Sea-level specific impulse (s)
        "burn_time": 1200,         # Burn time of the first stage (s)
        "dt": 1.0,                 # Time step for integration (s)
    }
    ROCKET["ve"] = ROCKET["Isp"] * 9.81
    ROCKET["mdot"] = ROCKET["thrust"] / ROCKET["ve"]
    ROCKET["t_max"] = ROCKET["burn_time"]

    times = get_times(duration=(ROCKET["burn_time"], 's'), freq=(ROCKET["dt"], 's'), t0=time_of_launch).gps

    lat, lon = launch_pads[pad_name]["latitude"], launch_pads[pad_name]["longitude"]

    observer = ssapy.EarthObserver(lon=lon, lat=lat, fast=False)
    r_launch, v_launch = observer.getRV(times[0])

    earth = ssapy.get_body("earth", model='egm2008')
    aEarth = ssapy.AccelKepler() + ssapy.AccelHarmonic(earth, 140, 140)

    m_f = ROCKET["m0"] - ROCKET["m_fuel"]
    delta_v = ROCKET["ve"] * np.log(ROCKET["m0"] / m_f)
    a_avg = delta_v / ROCKET["t_max"] + 10

    print(f'Burn time {ROCKET["t_max"]}, delta V: {delta_v}, a_avg: {a_avg}')
    burn1 = ssapy.AccelConstNTW(
        accelntw=[40, 0, 0.0],
        time_breakpoints=[times[0], times[0] + 600]
    )
    burn2 = ssapy.AccelConstNTW(
        accelntw=[20, 20, 0.0],
        time_breakpoints=[times[0] + 600, times[0] + 1200]
    )
    accel = aEarth + burn1 + burn2

    r, v = ssapy.rv(ssapy.Orbit(r=r_launch, v=v_launch, t=times[0]), time=times, propagator=ssapy.RK8Propagator(accel, h=1))

    if save_path is not False:
        groundTrackPlot(r, Time(times, format='gps'), launch_pads=[(lat, lon),], save_path=f"{save_path}ground_track.png")

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(r[:, 0] / 1000, r[:, 1] / 1000, label="Rocket Path", c="Green", linewidth=10)
        ax.add_patch(Circle((0, 0), radius=EARTH_RADIUS / 1000, color='Blue', alpha=0.5, label="Earth"))
        ax.set_xlabel("X Position (km)")
        ax.set_ylabel("Y Position (km)")
        ax.set_title("Rocket Trajectory")
        ax.legend(loc='upper left')
        plt.axis('equal')
        plt.show()
        save_plot(fig, f"{save_path}rocket_trajectory.png")  # Save trajectory plot

        fig, ax1 = plt.subplots(figsize=(7, 6))
        ax1.plot(times - times[0], (np.linalg.norm(r, axis=-1) - np.linalg.norm(r_launch, axis=-1)) / 1000, label="Altitude", color="blue")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Altitude (km)", color="blue")
        ax1.tick_params(axis='y', labelcolor="blue")
        ax1.grid(True)
        plt.show()
        save_plot(fig, f"{save_path}altitude_vs_time.png")  # Save altitude plot

    return {'r': r,
            'v': v,
            'altitude': (np.linalg.norm(r, axis=-1) - np.linalg.norm(r_launch, axis=-1)),
            'time': Time(times, format='gps'),
            'seconds': times - times[0]
            }


info = simulate_launch(pad_name, time_of_launch)
