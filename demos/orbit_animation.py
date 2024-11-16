import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from PIL import Image  # Import Image from PIL (Pillow)
from yeager_utils import find_file, RGEO, get_times, ssapy_orbit, rv_gcrf_to_itrf, gcrf_to_lunar_fixed, OrbitInitialize
from ssapy import get_body

plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 18,
    'figure.titlesize': 22
})

def get_earth_mesh(scale=20):
    from ssapy.constants import EARTH_RADIUS, RGEO
    png = Image.open(find_file("earth", ext=".png"))
    png = png.resize((5400 // scale, 2700 // scale))
    bm = np.array(png.resize([int(d) for d in png.size])) / 256.
    lons = np.linspace(-180, 180, bm.shape[1]) * np.pi / 180
    lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi / 180
    mesh_x = np.outer(np.cos(lons), np.cos(lats)).T * EARTH_RADIUS / RGEO
    mesh_y = np.outer(np.sin(lons), np.cos(lats)).T * EARTH_RADIUS / RGEO
    mesh_z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T * EARTH_RADIUS / RGEO
    return mesh_x, mesh_y, mesh_z, bm


def get_moon_mesh(scale=20):
    from ssapy.constants import MOON_RADIUS, RGEO
    png = Image.open(find_file("moon", ext=".png"))
    png = png.resize((5400 // scale, 2700 // scale))
    bm = np.array(png.resize([int(d) for d in png.size])) / 256.
    lons = np.linspace(-180, 180, bm.shape[1]) * np.pi / 180
    lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi / 180
    mesh_x = np.outer(np.cos(lons), np.cos(lats)).T * MOON_RADIUS / RGEO
    mesh_y = np.outer(np.sin(lons), np.cos(lats)).T * MOON_RADIUS / RGEO
    mesh_z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T * MOON_RADIUS / RGEO
    return mesh_x, mesh_y, mesh_z, bm

# Get times and moon position
t = get_times(duration=(1, 'month'), freq=(1, 'min'), t="2025-01-01")
r_moon = get_body("moon").position(t).T
r_moon_itrf, _ = rv_gcrf_to_itrf(r_moon, t)
r_moon_lunar = gcrf_to_lunar_fixed(r_moon, t)
print(np.shape(r_moon), np.shape(r_moon_itrf), np.shape(r_moon_lunar))
# Initialize DRO orbit
dro_orbit = OrbitInitialize.DRO(t=t[0])
r, v = ssapy_orbit(orbit=dro_orbit, times=t)

r_itrf, v_itrf = rv_gcrf_to_itrf(r, t)

# MAKE PLOT
fig = plt.figure(figsize=(27, 9))  # Adjusted figsize for 1x3 layout

# 3D scatter plot for r_itrf
ax1 = fig.add_subplot(131, projection='3d')
sc1 = ax1.scatter(r_itrf[:, 0] / RGEO, r_itrf[:, 1] / RGEO, r_itrf[:, 2] / RGEO, color=plt.cm.rainbow(np.linspace(0, 1, len(r_itrf))))
point1, = ax1.plot([], [], [], 'wo', zorder=10)  # White dot with higher zorder
moon_point1, = ax1.plot([], [], [], 'go', zorder=5)  # Grey dot for the moon
ax1.set_xlabel('X [GEO]')
ax1.set_ylabel('Y [GEO]')
ax1.set_zlabel('Z [GEO]')
max_dist = np.max(np.linalg.norm(r_itrf, axis=-1)) / RGEO
print(max_dist)
ax1.set_xlim((-max_dist, max_dist))
ax1.set_ylim((-max_dist, max_dist))
ax1.set_zlim((-max_dist, max_dist))
ax1.set_title('Fixed to Earth Surface (ITRS)')

# ADD Earth Mesh
mesh_x, mesh_y, mesh_z, bm = get_earth_mesh()
ax1.plot_surface(mesh_x, mesh_y, mesh_z, rstride=4, cstride=4, facecolors=bm, shade=False)


# 3D scatter plot for r_gcrf
ax2 = fig.add_subplot(132, projection='3d')
sc2 = ax2.scatter(r[:, 0] / RGEO, r[:, 1] / RGEO, r[:, 2] / RGEO, color=plt.cm.rainbow(np.linspace(0, 1, len(r))))
point2, = ax2.plot([], [], [], 'wo', zorder=10)  # White dot with higher zorder
moon_point2, = ax2.plot([], [], [], 'go', zorder=5)  # Grey dot for the moon
ax2.set_xlabel('X [GEO]')
ax2.set_ylabel('Y [GEO]')
ax2.set_zlabel('Z [GEO]')
max_dist = np.max(np.linalg.norm(r, axis=-1)) / RGEO
print(max_dist)
ax2.set_xlim((-max_dist, max_dist))
ax2.set_ylim((-max_dist, max_dist))
ax2.set_zlim((-max_dist, max_dist))
ax2.set_title('Fixed to Stars (GCRS)')

# ADD Earth Mesh
ax2.plot_surface(mesh_x, mesh_y, mesh_z, rstride=4, cstride=4, facecolors=bm, shade=False)


# 3D scatter plot for r_gcrf (copy of subplot 2)
ax3 = fig.add_subplot(133, projection='3d')
r_lunar = gcrf_to_lunar_fixed(r, t)
sc3 = ax3.scatter(r_lunar[:, 0] / RGEO, r_lunar[:, 1] / RGEO, r_lunar[:, 2] / RGEO, color=plt.cm.rainbow(np.linspace(0, 1, len(r))))
point3, = ax3.plot([], [], [], 'wo', zorder=10)  # White dot with higher zorder
moon_point3, = ax3.plot([], [], [], 'go', zorder=5)  # Grey dot for the moon
ax3.set_xlabel('X [GEO]')
ax3.set_ylabel('Y [GEO]')
ax3.set_zlabel('Z [GEO]')
max_dist = np.max(np.linalg.norm(r_lunar, axis=-1)) / RGEO
print(max_dist)
ax3.set_xlim((-max_dist, max_dist))
ax3.set_ylim((-max_dist, max_dist))
ax3.set_zlim((-max_dist, max_dist))
ax3.set_title('Fixed to the Moon')
# ADD Moon Mesh
mesh_x, mesh_y, mesh_z, bm = get_moon_mesh()
ax3.plot_surface(mesh_x, mesh_y, mesh_z, rstride=4, cstride=4, facecolors=bm, shade=False)


def update(num):
    # Update 3D plot for r_itrf
    idx = frame_indices[num]
    x1 = r_itrf[idx, 0] / RGEO
    y1 = r_itrf[idx, 1] / RGEO
    z1 = r_itrf[idx, 2] / RGEO
    point1.set_data([x1], [y1])
    point1.set_3d_properties([z1])
    
    # Update 3D plot for r_gcrf
    x2 = r[idx, 0] / RGEO
    y2 = r[idx, 1] / RGEO
    z2 = r[idx, 2] / RGEO
    point2.set_data([x2], [y2])
    point2.set_3d_properties([z2])
    
    # Update 3D plot for r_lunar
    x3 = r_lunar[idx, 0] / RGEO
    y3 = r_lunar[idx, 1] / RGEO
    z3 = r_lunar[idx, 2] / RGEO
    point3.set_data([x3], [y3])
    point3.set_3d_properties([z3])

    # Update 3D plot for r_moon_itrf
    xm1 = r_moon_itrf[idx, 0] / RGEO
    ym1 = r_moon_itrf[idx, 1] / RGEO
    zm1 = r_moon_itrf[idx, 2] / RGEO
    moon_point1.set_data([xm1], [ym1])
    moon_point1.set_3d_properties([zm1])
    
    # Update 3D plot for r_moon
    xm2 = r_moon[idx, 0] / RGEO
    ym2 = r_moon[idx, 1] / RGEO
    zm2 = r_moon[idx, 2] / RGEO
    moon_point2.set_data([xm2], [ym2])
    moon_point2.set_3d_properties([zm2])
    
    # Update 3D plot for r_moon_lunar
    xm3 = r_moon_lunar[idx, 0] / RGEO
    ym3 = r_moon_lunar[idx, 1] / RGEO
    zm3 = r_moon_lunar[idx, 2] / RGEO
    moon_point3.set_data([xm3], [ym3])
    moon_point3.set_3d_properties([zm3])
    
    return point1, point2, point3, moon_point1, moon_point2, moon_point3

# Generate 100 equally spaced frame indices
frame_indices = np.linspace(0, len(t) - 1, 200).astype(int)

ani = animation.FuncAnimation(fig, update, frames=range(200), blit=True, interval=50)

# Save animation as a GIF in the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, 'orbit_animation_DRO.gif')
ani.save(output_path, writer='pillow', fps=5)
print(f"Saved gif to {output_path}\nComplete.\n")

plt.tight_layout()
plt.show()
