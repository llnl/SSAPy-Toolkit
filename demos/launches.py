import numpy as np
import matplotlib.pyplot as plt
from ssapy import Orbit, Propagator, EarthGravityModel, Plotter

# Constants
R_EARTH = 6371e3  # Earth's radius in meters
ALTITUDE = 200e3  # Orbit altitude in meters
A = R_EARTH + ALTITUDE  # Semi-major axis for circular orbit in meters

# Launch locations: (name, latitude in degrees, longitude in degrees)
launch_locations = [
    ("Cape Canaveral", 28.5, -80.5),
    ("Vandenberg", 34.7, -120.6),
    ("Kourou", 5.2, -52.7),
    ("Baikonur", 45.6, 63.3),
]

# Time parameters
propagation_time = 86400  # 24 hours in seconds
time_step = 60  # Time step in seconds

# Set up the gravity model
gravity_model = EarthGravityModel.EGM96  # Using EGM96 for realistic gravity perturbations

# Initialize plotter for visualization
plotter = Plotter()

# Simulate launch and orbit propagation for each location
for name, lat, lon in launch_locations:
    # Set inclination equal to latitude (due east launch)
    inclination = lat  # in degrees
    
    # Define Keplerian orbital elements for a circular orbit
    elements = {
        'a': A,              # Semi-major axis in meters
        'e': 0.0,            # Eccentricity (circular orbit)
        'i': np.deg2rad(inclination),  # Inclination in radians
        'Omega': 0.0,        # RAAN in radians
        'omega': 0.0,        # Argument of perigee in radians (undefined for e=0, set to 0)
        'nu': 0.0,           # True anomaly in radians (starting position)
    }
    
    # Create an Orbit object from Keplerian elements
    orbit = Orbit.from_keplerian(elements)
    
    # Set up the propagator with a numerical integrator
    propagator = Propagator.RK8(orbit, gravity_model=gravity_model)
    
    # Propagate the orbit over time
    times = np.arange(0, propagation_time, time_step)
    states = propagator.propagate(times)
    
    # Extract latitude and longitude for ground track
    lats, lons = [], []
    for state in states:
        lat, lon = state.to_lat_lon()  # Convert state to geodetic coordinates
        lats.append(lat)
        lons.append(lon)
    
    # Plot the ground track for this launch location
    plotter.plot_ground_track(lats, lons, label=name)

# Display the plot with all ground tracks
plotter.show()