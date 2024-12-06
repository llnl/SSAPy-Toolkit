import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from yeager_utils import RGEO, ssapy_orbit, gcrf_to_itrf, calc_gamma, calc_heading, get_script_dir, EARTH_RADIUS, find_file

# Example usage:
inc = 0
r, v, t = ssapy_orbit(a=RGEO, e=0.75, i=np.radians(inc), pa=0, raan=0, ta=np.pi * 2, duration=(1, 'day'), freq=(1, 'min'), t0="2025-01-01")

r_itrf, v_itrf = gcrf_to_itrf(r, t, v=True)
gamma = calc_gamma(r, t)
heading = calc_heading(r, t)

# Load Earth image and mesh
earth_png = Image.open(find_file("earth", ext=".png"))
earth_png = earth_png.resize((5400 // 20, 2700 // 20))  # Resize to match Earth mesh scale
bm = np.array(earth_png.resize([int(d) for d in earth_png.size])) / 256.
lons = np.linspace(-180, 180, bm.shape[1]) * np.pi / 180
lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi / 180
mesh_x = np.outer(np.cos(lons), np.cos(lats)).T * EARTH_RADIUS / RGEO
mesh_y = np.outer(np.sin(lons), np.cos(lats)).T * EARTH_RADIUS / RGEO
mesh_z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T * EARTH_RADIUS / RGEO

# Create figure and axes
fig = plt.figure(figsize=(14, 10))  # Increased figsize to accommodate three subplots

# Top-left: Orbits in ITRF frame
ax_itrf = fig.add_subplot(221, projection='3d')
ax_itrf.scatter(r_itrf[:, 0] / RGEO, r_itrf[:, 1] / RGEO, r_itrf[:, 2] / RGEO,
                c=plt.cm.rainbow(np.linspace(0, 1, len(r_itrf))))


point_itrf, = ax_itrf.plot([], [], [], 'ko')  # Black dot for animation
ax_itrf.set_xlim((-1.5, 1.5))
ax_itrf.set_ylim((-1.5, 1.5))
ax_itrf.set_zlim((-1.5, 1.5))
ax_itrf.set_title('ITRF Orbit')
ax_itrf.set_xlabel('X [GEO]')
ax_itrf.set_ylabel('Y [GEO]')
ax_itrf.set_zlabel('Z [GEO]')

# Add Earth mesh to ITRF plot
ax_itrf.plot_surface(mesh_x, mesh_y, mesh_z, rstride=4, cstride=4, facecolors=bm, shade=False)

# Top-right: Orbits in GCRF frame
ax_gcrf = fig.add_subplot(222, projection='3d')
ax_gcrf.scatter(r[:, 0] / RGEO, r[:, 1] / RGEO, r[:, 2] / RGEO,
                c=plt.cm.rainbow(np.linspace(0, 1, len(r))))


point_gcrf, = ax_gcrf.plot([], [], [], 'ko')  # Black dot for animation
ax_gcrf.set_xlim((-1.5, 1.5))
ax_gcrf.set_ylim((-1.5, 1.5))
ax_gcrf.set_zlim((-1.5, 1.5))
ax_gcrf.set_title('GCRF Orbit')
ax_gcrf.set_xlabel('X [GEO]')
ax_gcrf.set_ylabel('Y [GEO]')
ax_gcrf.set_zlabel('Z [GEO]')

# Add Earth mesh to GCRF plot
ax_gcrf.plot_surface(mesh_x, mesh_y, mesh_z, rstride=4, cstride=4, facecolors=bm, shade=False)

# Bottom-left: Gamma over time
ax_gamma = fig.add_subplot(223)
line_gamma, = ax_gamma.plot((t.decimalyear - t[0].decimalyear) * 365.25, gamma)
point_gamma, = ax_gamma.plot([], [], 'ko')  # Black dot for animation
ax_gamma.set_title('Gamma vs Time')
ax_gamma.set_xlabel('Time [days]')
ax_gamma.set_ylabel('Gamma [degrees]')

# Bottom-left: Gamma over time
ax_heading = fig.add_subplot(224)
line_heading, = ax_heading.plot((t.decimalyear - t[0].decimalyear) * 365.25, heading)
point_heading, = ax_heading.plot([], [], 'ko')  # Black dot for animation
ax_heading.set_title('Heading vs Time')
ax_heading.set_xlabel('Time [days]')
ax_heading.set_ylabel('Heading [degrees]')

# Generate 100 equally spaced frame indices
frame_indices = np.linspace(0, len(t) - 1, 100).astype(int)


# Update function for animation
def update(num):
    idx = frame_indices[num]

    # Update gamma plot
    x = (t.decimalyear[idx] - t[0].decimalyear) * 365.25
    y = gamma[idx]
    point_gamma.set_data([x], [y])

    y = heading[idx]
    point_heading.set_data([x], [y])

    # Update ITRF orbit
    x_itrf = r_itrf[idx, 0] / RGEO
    y_itrf = r_itrf[idx, 1] / RGEO
    z_itrf = r_itrf[idx, 2] / RGEO
    point_itrf.set_data([x_itrf], [y_itrf])
    point_itrf.set_3d_properties([z_itrf])

    # Update GCRF orbit
    x_gcrf = r[idx, 0] / RGEO
    y_gcrf = r[idx, 1] / RGEO
    z_gcrf = r[idx, 2] / RGEO
    point_gcrf.set_data([x_gcrf], [y_gcrf])
    point_gcrf.set_3d_properties([z_gcrf])

    return point_gamma, point_heading, point_itrf, point_gcrf


# Create the animation
ani = animation.FuncAnimation(
    fig, update, frames=len(frame_indices), blit=False, interval=50
)

# Save animation as a GIF
script_dir = get_script_dir()
gif_path = os.path.join(script_dir, f"orbit_gamma_heading_incl_{inc}.gif")
print(f"Saving gif to {gif_path}")
ani.save(gif_path, writer="pillow", fps=5)

plt.tight_layout()
plt.show()
