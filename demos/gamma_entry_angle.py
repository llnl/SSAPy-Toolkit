import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from PIL import Image  # Import Image from PIL (Pillow)
from yeager_utils import find_file, EARTH_RADIUS, RGEO, ssapy_orbit, rv_gcrf_to_itrf, angle_between_vectors, calc_gamma

# Example usage:
r, v, t = ssapy_orbit(a=RGEO, e=0.75, i=0, pa=0, raan=0, ta=np.pi * 2, 
                      duration=(1, 'day'), freq=(1, 'min'), start_date="2025-01-01")

r_itrf, v_itrf = rv_gcrf_to_itrf(r, t)
gamma = calc_gamma(r, t)

# MAKE PLOT
fig = plt.figure(figsize=(18, 6))  # Increased figsize to accommodate three subplots

# 2D plot for gamma vs time
ax1 = fig.add_subplot(131)
line1, = ax1.plot((t.decimalyear - t[0].decimalyear) * 365.25, gamma)
point1, = ax1.plot([], [], 'ko', zorder=10)  # Black dot with higher zorder
ax1.set_xlabel('Time [days]')
ax1.set_ylabel('Gamma [degrees]')
ax1.set_title('Gamma vs Time')

# 3D scatter plot for r_itrf
ax2 = fig.add_subplot(132, projection='3d')
sc = ax2.scatter(r_itrf[:, 0] / RGEO, r_itrf[:, 1] / RGEO, r_itrf[:, 2] / RGEO, color=plt.cm.rainbow(np.linspace(0, 1, len(r_itrf))))
point2, = ax2.plot([], [], [], 'ko', zorder=10)  # Black dot with higher zorder
ax2.set_xlabel('X [GEO]')
ax2.set_ylabel('Y [GEO]')
ax2.set_zlabel('Z [GEO]')
ax2.set_xlim((-1.5, 1.5))
ax2.set_ylim((-1.5, 1.5))
ax2.set_zlim((-1.5, 1.5))
ax2.set_title('Fixed to Earth Surface')

# ADD Earth Mesh
scale = 20
earth_png = Image.open(find_file("earth", ext=".png"))
earth_png = earth_png.resize((5400 // scale, 2700 // scale))
bm = np.array(earth_png.resize([int(d) for d in earth_png.size])) / 256.
lons = np.linspace(-180, 180, bm.shape[1]) * np.pi / 180
lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi / 180
mesh_x = np.outer(np.cos(lons), np.cos(lats)).T * EARTH_RADIUS / RGEO
mesh_y = np.outer(np.sin(lons), np.cos(lats)).T * EARTH_RADIUS / RGEO
mesh_z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T * EARTH_RADIUS / RGEO
ax2.plot_surface(mesh_x, mesh_y, mesh_z, rstride=4, cstride=4, facecolors=bm, shade=False)

# 3D scatter plot for r_gcrf
ax3 = fig.add_subplot(133, projection='3d')
sc = ax3.scatter(r[:, 0] / RGEO, r[:, 1] / RGEO, r[:, 2] / RGEO, color=plt.cm.rainbow(np.linspace(0, 1, len(r))))
point3, = ax3.plot([], [], [], 'ko', zorder=10)  # Black dot with higher zorder
ax3.set_xlabel('X [GEO]')
ax3.set_ylabel('Y [GEO]')
ax3.set_zlabel('Z [GEO]')
ax3.set_xlim((-1.5, 1.5))
ax3.set_ylim((-1.5, 1.5))
ax3.set_zlim((-1.5, 1.5))
ax3.set_title('Fixed to Stars')
# ADD Earth Mesh
ax3.plot_surface(mesh_x, mesh_y, mesh_z, rstride=4, cstride=4, facecolors=bm, shade=False)

def update(num):
    # Update 2D plot
    idx = frame_indices[num]
    x = (t.decimalyear[idx] - t[0].decimalyear) * 365.25
    y = gamma[idx]
    point1.set_data(x, y)
    point1.set_xdata(x)
    point1.set_ydata(y)
    
    # Update 3D plots
    x2 = r_itrf[idx, 0] / RGEO
    y2 = r_itrf[idx, 1] / RGEO
    z2 = r_itrf[idx, 2] / RGEO
    point2.set_data([x2], [y2])
    point2.set_3d_properties([z2])
    point2.set_xdata([x2])
    point2.set_ydata([y2])
    point2.set_3d_properties([z2])
    
    x3 = r[idx, 0] / RGEO
    y3 = r[idx, 1] / RGEO
    z3 = r[idx, 2] / RGEO
    point3.set_data([x3], [y3])
    point3.set_3d_properties([z3])
    point3.set_xdata([x3])
    point3.set_ydata([y3])
    point3.set_3d_properties([z3])

    return point1, point2, point3

# Generate 100 equally spaced frame indices
frame_indices = np.linspace(0, len(t) - 1, 100).astype(int)

ani = animation.FuncAnimation(fig, update, frames=range(100), blit=True, interval=50)

# Save animation as a GIF
ani.save('gamma_animation.gif', writer='pillow', fps=5)

plt.tight_layout()
plt.show()
