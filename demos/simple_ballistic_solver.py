import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Constants
R_EARTH = 6.371e6     # Earth radius, m
GM = 3.986e14         # Gravitational parameter, m^3/s^2
omega = 7.292e-5      # Earth's angular velocity, rad/s

# Launch location (in degrees)
launch_lat = 0.0
launch_lon = -90.0

# Initial guess for launch parameters
v0 = 8e3              # initial speed, m/s
azimuth_from_north = 135.0  # initial azimuth (clockwise from North), degrees
altitude_angle = 45.0       # initial altitude angle (angle above local horizontal), degrees

# Desired target landing site (latitude and longitude in degrees)
target_lat = 20.0
target_lon = -150.0

# Functions

def get_initial_state(lat, lon, v0, az, alt):
    """Compute initial global position and velocity from local launch parameters."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x0 = R_EARTH * np.cos(lat_rad) * np.cos(lon_rad)
    y0 = R_EARTH * np.cos(lat_rad) * np.sin(lon_rad)
    z0 = R_EARTH * np.sin(lat_rad)
    # Define local east, north, up unit vectors.
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

def derivatives(t, s):
    """Compute time derivatives including gravity, centrifugal and Coriolis accelerations."""
    x, y, z, vx, vy, vz = s
    r = np.sqrt(x**2 + y**2 + z**2)
    # Gravitational acceleration
    ax_g = -GM * x / r**3
    ay_g = -GM * y / r**3
    az_g = -GM * z / r**3
    # Centrifugal acceleration (approximate)
    ax_cen = omega**2 * x
    ay_cen = omega**2 * y
    az_cen = 0
    # Coriolis acceleration
    ax_cor = 2 * omega * vy
    ay_cor = -2 * omega * vx
    az_cor = 0
    ax = ax_g + ax_cen + ax_cor
    ay = ay_g + ay_cen + ay_cor
    az = az_g + az_cen + az_cor
    return [vx, vy, vz, ax, ay, az]

def hit_surface(t, s):
    """Event function: stop integration when the projectile hits the Earth’s surface."""
    r = np.sqrt(s[0]**2 + s[1]**2 + s[2]**2)
    return r - R_EARTH

hit_surface.terminal = True
hit_surface.direction = -1

def great_circle_distance(lat1, lon1, lat2, lon2):
    """Compute the great-circle distance between two points (in meters) using the haversine formula."""
    R = R_EARTH
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def compute_bearing(lat1, lon1, lat2, lon2):
    """Compute the initial bearing (in degrees) from point 1 to point 2."""
    dlon = np.radians(lon2 - lon1)
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    x = np.sin(dlon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360

# Iterative loop to adjust initial parameters
max_iter = 20
tol_distance = 1e3  # tolerance in meters (for the haversine error)
# Gains
k_v0 = 0.001      # v0 update gain (m/s per meter of range error)
k_az = 0.1        # azimuth update gain (deg per deg of bearing error)
# Use separate gain for altitude angle:
k_alt = 0.5       # update from range error (deg per relative range error)
k_lat = 0.5       # update from latitude error (deg per deg)

for iteration in range(max_iter):
    # Generate initial state using current parameters.
    state0 = get_initial_state(launch_lat, launch_lon, v0,
                               azimuth_from_north, altitude_angle)
    # Solve the trajectory
    sol = solve_ivp(derivatives, (0, 1e5), state0, events=hit_surface,
                    max_step=1.0, rtol=1e-8, atol=1e-8)
    # Extract landing state from the last integration step.
    landing_x, landing_y, landing_z = sol.y[0, -1], sol.y[1, -1], sol.y[2, -1]
    landing_r = np.sqrt(landing_x**2 + landing_y**2 + landing_z**2)
    landing_lat = np.degrees(np.arcsin(landing_z / landing_r))
    landing_lon = np.degrees(np.arctan2(landing_y, landing_x))
    
    # Compute the overall error (haversine distance between landing and target).
    total_error = great_circle_distance(landing_lat, landing_lon, target_lat, target_lon)
    
    # Also compute the range error from the launch point.
    desired_range = great_circle_distance(launch_lat, launch_lon, target_lat, target_lon)
    simulated_range = great_circle_distance(launch_lat, launch_lon, landing_lat, landing_lon)
    range_error = desired_range - simulated_range

    # Compute the bearing from launch to target and launch to landing.
    desired_bearing = compute_bearing(launch_lat, launch_lon, target_lat, target_lon)
    simulated_bearing = compute_bearing(launch_lat, launch_lon, landing_lat, landing_lon)
    bearing_error = (desired_bearing - simulated_bearing + 540) % 360 - 180  # shortest angle

    # Also compute landing latitude error
    lat_error = target_lat - landing_lat

    print(f"Iteration {iteration+1}:")
    print(f"  Landing lat, lon: ({landing_lat:.4f}, {landing_lon:.4f}) deg")
    print(f"  Total haversine error: {total_error:.2f} m")
    print(f"  Simulated range: {simulated_range/1e3:.2f} km, Desired range: {desired_range/1e3:.2f} km, Range error: {range_error:.2f} m")
    print(f"  Desired bearing: {desired_bearing:.2f} deg, Simulated bearing: {simulated_bearing:.2f} deg, Bearing error: {bearing_error:.2f} deg")
    print(f"  Latitude error: {lat_error:.2f} deg")
    print(f"  Current v0: {v0:.2f} m/s, azimuth: {azimuth_from_north:.2f} deg, altitude angle: {altitude_angle:.2f} deg\n")

    # Check convergence based on the overall (haversine) error.
    if total_error < tol_distance:
        print("Convergence achieved!\n")
        break

    # Update v0 to adjust the range.
    v0 += k_v0 * range_error

    # Update azimuth based on the difference between the launch-to-target and launch-to-landing bearings.
    azimuth_from_north += k_az * bearing_error

    # Update altitude angle using both range error and landing latitude error.
    altitude_angle -= k_alt * (range_error / desired_range) + k_lat * lat_error

# Final simulation with converged parameters.
state0 = get_initial_state(launch_lat, launch_lon, v0,
                           azimuth_from_north, altitude_angle)
sol = solve_ivp(derivatives, (0, 1e5), state0, events=hit_surface,
                max_step=1.0, rtol=1e-8, atol=1e-8)

x, y, z = sol.y[0], sol.y[1], sol.y[2]
landing = [x[-1], y[-1], z[-1]]
print(f"Final landing position: ({landing[0]:.2f}, {landing[1]:.2f}, {landing[2]:.2f}) m")

# Convert landing coordinates to lat-lon for plotting.
lon_arr = np.degrees(np.arctan2(y, x))
lat_arr = np.degrees(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))
altitude = np.sqrt(x**2 + y**2 + z**2) - R_EARTH
t = sol.t

# Create Earth surface grid for 3D plot.
phi_earth = np.linspace(0, np.pi, 50)
theta_earth = np.linspace(0, 2 * np.pi, 50)
phi_earth, theta_earth = np.meshgrid(phi_earth, theta_earth)
earth_x = R_EARTH * np.sin(phi_earth) * np.cos(theta_earth)
earth_y = R_EARTH * np.sin(phi_earth) * np.sin(theta_earth)
earth_z = R_EARTH * np.cos(phi_earth)

# Plotting.
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
ax1.plot(lon_arr, lat_arr, color='red', label='Ground Track', transform=ccrs.Geodetic())
ax1.plot(launch_lon, launch_lat, 'g*', markersize=10, label='Launch Site', transform=ccrs.Geodetic())
ax1.plot(lon_arr[-1], lat_arr[-1], 'kx', markersize=8, label='Landing Site', transform=ccrs.Geodetic())
ax1.legend(loc='lower left')

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t / 60, altitude / 1000, color='blue')
ax2.set_xlabel('Time (minutes)')
ax2.set_ylabel('Altitude (km)')
ax2.set_title('Altitude vs Time')
ax2.grid(True)

ax3 = fig.add_subplot(gs[1, 1], projection='3d')
ax3.plot_surface(earth_x / 1e3, earth_y / 1e3, earth_z / 1e3,
                 color='blue', alpha=0.5, linewidth=0)
ax3.plot(x / 1e3, y / 1e3, z / 1e3, color='red', label='Trajectory')
ax3.scatter(x[0] / 1e3, y[0] / 1e3, z[0] / 1e3,
            color='green', marker='*', s=100, label='Launch Site')
ax3.scatter(x[-1] / 1e3, y[-1] / 1e3, z[-1] / 1e3,
            color='black', marker='x', s=80, label='Landing Site')
ax3.set_xlabel('X (km)')
ax3.set_ylabel('Y (km)')
ax3.set_zlabel('Z (km)')
ax3.set_title('3D Earth and Rocket Trajectory')
ax3.set_xlim([-10e3, 10e3])
ax3.set_ylim([-10e3, 10e3])
ax3.set_zlim([-10e3, 10e3])
ax3.legend()

plt.tight_layout()
plt.show()
