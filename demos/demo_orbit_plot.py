from yeager_utils import ssapy_orbit, orbit_plot, groundtrack_dashboard, cislunar_plot_3d, cislunar_plot, globe_plot, RGEO, get_lunar_rv, Time
import os
import numpy as np


def imname(filename):
    # List of directories to try
    directories = [
        "/g/g16/yeager7/workdir/yeager_utils/demos/images/",
        "/home/yeager7/yeager_utils/demos/images/"
    ]
    
    for save_dir in directories:
        try:
            # Attempt to create directory and return path
            os.makedirs(save_dir, exist_ok=True)
            return f"{save_dir}{filename}.png"
        except (OSError, PermissionError):
            # Skip to next directory if there's an error
            continue
    
    # If all directories fail, raise an exception
    raise Exception("Could not create or access any of the specified directories")


times = Time("2024-1-1").gps
print(times)
r_moon, v_moon = get_lunar_rv(times)
print(r_moon, v_moon)

r0 = r_moon[0] + (1000e3 * r_moon[0] / np.linalg.norm(r_moon[0]))
v0 = v_moon[0] + 100
print(r0, v0)

# single orbit
print("\nCalculating orbit.")
r, v, t = ssapy_orbit(r=r0, v=v0, duration=(1, 'month'))
# print(t, type(t), type(np.array(t)), type([]), type([t, t]))
print(f"Plotting orbit. {np.shape(r)} {np.shape(t)}")
orbit_plot(r=r, t=t, save_path=imname("demo_orbit_plot"))
cislunar_plot(r=r, t=t, save_path=imname("demo_cislunar_plot"))
cislunar_plot_3d(r=r, t=t, save_path=imname("demo_cislunar_plot_3d"))
globe_plot(r=r, t=t, save_path=imname("demo_globe_plot_black"), scale=5)
globe_plot(r=r, t=t, save_path=imname("demo_globe_plot_white"), scale=5, c='white')

# two same length orbit
print("\nCalculating 2 orbit.")
r2, v2, t2 = ssapy_orbit(a=9 * RGEO, e=0.5, i=.25, pa=np.pi / 2, duration=(1, 'month'))
print(f"Plotting two orbits same length. {np.shape(r)} {np.shape(r2)} {np.shape(t)}")
orbit_plot(r=[r, r2], t=t, save_path=imname("demo_orbit_plot_two_orbits"))
cislunar_plot(r=[r, r2], t=t, save_path=imname("demo_cislunar_plot_two_orbits"))
globe_plot(r=[r, r2], t=t, save_path=imname("demo_globe_two_orbits"), scale=5, c='black')

# two orbit different lengths
print("\nCalculating 2 different orbit.")
r3, v3, t3 = ssapy_orbit(a=5 * RGEO, e=0.5, i=.75, duration=(7, 'day'), t0="2024-1-1")
print("Plotting two orbits different lengths.")
orbit_plot(r=[r, r3], t=[t, t3], save_path=imname("demo_orbit_plot_two_different_length_orbits"))
orbit_plot(r=[r, r3], t=[t, t3], save_path=imname("demo_orbit_plot_two_different_length_orbits_itrf"), frame='itrf')
cislunar_plot(r=[r, r3], t=[t, t3], save_path=imname("demo_cislunar_plot_two_different_length_orbits"))
globe_plot(r=[r, r3], t=[t, t3], save_path=imname("demo_globe_two_different_length_orbits"), scale=5, c='black')


r, v, t = ssapy_orbit(a=RGEO, e=0.2, duration=(1, 'day'))
# Call the dashboard
fig = groundtrack_dashboard(r, t, show=True, save_path=imname("demo_ground_dashboard_test"))
fig = groundtrack_dashboard(r=[r, r3], t=[t, t3], show=True, save_path=imname("demo_ground_dashboard_two_different_length_orbits"))

print("PLOT DEMO DONE.")