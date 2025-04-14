import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

R_EARTH = 6.371e6
GM = 3.986e14
omega = 7.292e-5

launch_lat = 0
launch_lon = -90

v0 = 8e3
azimuth_from_north = 135  # Example: 90 degrees clockwise from North is East
altitude_angle = 45       # Example: 45 degrees from the local plane

lat_rad = np.radians(launch_lat)
lon_rad = np.radians(launch_lon)
x0 = R_EARTH * np.cos(lat_rad) * np.cos(lon_rad)
y0 = R_EARTH * np.cos(lat_rad) * np.sin(lon_rad)
z0 = R_EARTH * np.sin(lat_rad)

def get_initial_state(lat, lon, v0, az, alt):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x0 = R_EARTH * np.cos(lat_rad) * np.cos(lon_rad)
    y0 = R_EARTH * np.cos(lat_rad) * np.sin(lon_rad)
    z0 = R_EARTH * np.sin(lat_rad)
    east_x = -np.sin(lon_rad)
    east_y = np.cos(lon_rad)
    east_z = 0
    north_x = -np.sin(lat_rad) * np.cos(lon_rad)
    north_y = -np.sin(lat_rad) * np.sin(lon_rad)
    north_z = np.cos(lat_rad)
    up_x = np.cos(lat_rad) * np.cos(lon_rad)
    up_y = np.cos(lat_rad) * np.sin(lon_rad)
    up_z = np.sin(lat_rad)

    az_rad = np.radians(az)
    alt_rad = np.radians(alt)

    # Velocity components in local East, North, Up frame
    v_east = v0 * np.cos(alt_rad) * np.sin(az_rad)
    v_north = v0 * np.cos(alt_rad) * np.cos(az_rad)
    v_up = v0 * np.sin(alt_rad)

    # Convert local frame velocity components to global Cartesian frame
    vx0 = v_east * east_x + v_north * north_x + v_up * up_x
    vy0 = v_east * east_y + v_north * north_y + v_up * up_y
    vz0 = v_east * east_z + v_north * north_z + v_up * up_z

    return [x0, y0, z0, vx0, vy0, vz0]

state0 = get_initial_state(launch_lat, launch_lon, v0, azimuth_from_north, altitude_angle)

def derivatives(t, s):
    x, y, z, vx, vy, vz = s
    r = np.sqrt(x**2 + y**2 + z**2)
    ax_g = -GM / r**3 * x
    ay_g = -GM / r**3 * y
    az_g = -GM / r**3 * z

    ax_cen = omega**2 * x
    ay_cen = omega**2 * y
    az_cen = 0

    ax_cor = 2 * omega * vy
    ay_cor = -2 * omega * vx
    az_cor = 0

    ax = ax_g + ax_cen + ax_cor
    ay = ay_g + ay_cen + ay_cor
    az = az_g + az_cen + az_cor

    return [vx, vy, vz, ax, ay, az]

def hit_surface(t, s):
    r = np.sqrt(s[0]**2 + s[1]**2 + s[2]**2)
    return r - R_EARTH

hit_surface.terminal = True
hit_surface.direction = -1

sol = solve_ivp(derivatives, (0, 1e5), state0,
                    events=hit_surface, max_step=1.0, rtol=1e-8, atol=1e-8)

x, y, z = sol.y[0], sol.y[1], sol.y[2]
landing = [x[-1], y[-1], z[-1]]
print(f"Landing position: ({landing[0]:.2f}, {landing[1]:.2f}, {landing[2]:.2f}) m")

lon = np.degrees(np.arctan2(y, x))
lat = np.degrees(np.arcsin(z / (np.sqrt(x**2 + y**2 + z**2))))

altitude = np.sqrt(x**2 + y**2 + z**2) - R_EARTH
t = sol.t

phi_earth = np.linspace(0, np.pi, 50)
theta_earth = np.linspace(0, 2 * np.pi, 50)
phi_earth, theta_earth = np.meshgrid(phi_earth, theta_earth)
earth_x = R_EARTH * np.sin(phi_earth) * np.cos(theta_earth)
earth_y = R_EARTH * np.sin(phi_earth) * np.sin(theta_earth)
earth_z = R_EARTH * np.cos(phi_earth)

fig = plt.figure(figsize=(20, 20))
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

ax1 = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
ax1.set_global()
ax1.add_feature(cfeature.LAND, edgecolor='black')
ax1.add_feature(cfeature.OCEAN)
ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linestyle=':')
ax1.add_feature(cfeature.LAKES, alpha=0.5)
ax1.add_feature(cfeature.RIVERS)
ax1.stock_img()
ax1.gridlines(draw_labels=True)
ax1.coastlines()
ax1.gridlines(draw_labels=True)

ax1.plot(lon, lat, color='red', label='Ground Track', transform=ccrs.Geodetic())
ax1.plot(launch_lon, launch_lat, 'g*', markersize=10, label='Launch Site', transform=ccrs.Geodetic())
ax1.plot(lon[-1], lat[-1], 'kx', markersize=8, label='Landing Site', transform=ccrs.Geodetic())
ax1.legend(loc='lower left')

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t / 60, altitude / 1000, color='blue')
ax2.set_xlabel('Time (minutes)')
ax2.set_ylabel('Altitude (km)')
ax2.set_title('Altitude vs Time')
ax2.grid(True)

ax3 = fig.add_subplot(gs[1, 1], projection='3d')
ax3.plot_surface(earth_x / 1e3 , earth_y / 1e3, earth_z / 1e3, color='blue', alpha=0.5, linewidth=0)
ax3.plot(x / 1e3, y / 1e3, z / 1e3, color='red', label='Trajectory')
ax3.scatter(x[0] / 1e3, y[0] / 1e3, z[0] / 1e3, color='green', marker='*', s=100, label='Launch Site')
ax3.scatter(x[-1] / 1e3, y[-1] / 1e3, z[-1] / 1e3, color='black', marker='x', s=80, label='Landing Site')
ax3.set_xlabel('X (km)')
ax3.set_ylabel('Y (km)')
ax3.set_zlabel('Z (km)')
ax3.set_title('3D Earth and Rocket Trajectory')
ax3.set_xlim([-10e3, 10e3])
ax3.set_ylim([-10e3, 10e3])
ax3.set_zlim([-10e3, 10e3])
ax3.set_xticks([-10, 10])
ax3.set_yticks([-10, 10])
ax3.set_zticks([-10, 10])
ax3.legend()

plt.tight_layout()
plt.show()