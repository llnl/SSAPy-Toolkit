import numpy as np
from scipy.integrate import solve_ivp
from yeager_utils import groundtrack_dashboard

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
target_lat = 0.0
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
tol_distance = 1e3  # tolerance in meters for landing error
k_v0 = 0.001      # proportional gain for v0 update (m/s per meter of range error)
k_az = 0.1        # proportional gain for azimuth update (deg per deg of bearing error)
k_alt = 0.5       # proportional gain for altitude angle update (deg per relative range error)
k_lat = 0.5       # additional gain to directly correct latitude error (deg per deg)

for iteration in range(max_iter):
    # Generate initial state using current parameters.
    state0 = get_initial_state(launch_lat, launch_lon, v0,
                               azimuth_from_north, altitude_angle)
    # Solve the trajectory
    sol = solve_ivp(derivatives, (0, 1e5), state0, events=hit_surface,
                    max_step=1.0, rtol=1e-8, atol=1e-8)
    # Extract landing state
    landing_x, landing_y, landing_z = sol.y[0, -1], sol.y[1, -1], sol.y[2, -1]
    landing_r = np.sqrt(landing_x**2 + landing_y**2 + landing_z**2)
    landing_lat = np.degrees(np.arcsin(landing_z / landing_r))
    landing_lon = np.degrees(np.arctan2(landing_y, landing_x))
    
    # Compute error in range (great-circle distance from launch to landing)
    sim_range = great_circle_distance(launch_lat, launch_lon, landing_lat, landing_lon)
    desired_range = great_circle_distance(launch_lat, launch_lon, target_lat, target_lon)
    range_error = desired_range - sim_range

    # Compute bearing differences
    desired_bearing = compute_bearing(launch_lat, launch_lon, target_lat, target_lon)
    sim_bearing = compute_bearing(launch_lat, launch_lon, landing_lat, landing_lon)
    bearing_error = (desired_bearing - sim_bearing + 540) % 360 - 180  # shortest angle

    # Compute the landing latitude error (target - landing).
    lat_error = target_lat - landing_lat

    print(f"Iteration {iteration+1}:")
    print(f"  Landing lat, lon: ({landing_lat:.4f}, {landing_lon:.4f}) deg")
    print(f"  Simulated range: {sim_range/1e3:.2f} km, "
          f"Desired range: {desired_range/1e3:.2f} km, "
          f"Range error: {range_error:.2f} m")
    print(f"  Desired bearing: {desired_bearing:.2f} deg, "
          f"Simulated bearing: {sim_bearing:.2f} deg, "
          f"Bearing error: {bearing_error:.2f} deg")
    print(f"  Latitude error: {lat_error:.2f} deg")
    print(f"  Current v0: {v0:.2f} m/s, azimuth: {azimuth_from_north:.2f} deg, "
          f"altitude angle: {altitude_angle:.2f} deg\n")

    # Check convergence (if range error is less than tolerance)
    if abs(range_error) < tol_distance and abs(lat_error) < 0.1:
        print("Convergence achieved!\n")
        break

    # Update v0 to correct range error
    v0 += k_v0 * range_error

    # Update azimuth based on bearing error and direct latitude error correction.
    azimuth_from_north += k_az * bearing_error + k_lat * lat_error

    # Update altitude angle to help adjust the range (if needed)
    altitude_angle -= k_alt * (range_error / desired_range)

# Final simulation with converged parameters
state0 = get_initial_state(launch_lat, launch_lon, v0,
                           azimuth_from_north, altitude_angle)
sol = solve_ivp(derivatives, (0, 1e5), state0, events=hit_surface,
                max_step=1.0, rtol=1e-8, atol=1e-8)

x, y, z = sol.y[0], sol.y[1], sol.y[2]
landing = [x[-1], y[-1], z[-1]]
print(f"Final landing position: ({landing[0]:.2f}, {landing[1]:.2f}, {landing[2]:.2f}) m")

fig = groundtrack_dashboard(x, y, z, sol.t, save_path="images/simple_arc_solver.jpg")