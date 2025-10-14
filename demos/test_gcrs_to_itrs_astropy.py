from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, CartesianDifferential
from astropy.time import Time
import astropy.units as u
import numpy as np

# Define your GCRS coordinates and velocities (example values)
gcrs_x = 1.0 * u.m
gcrs_y = 2.0 * u.m
gcrs_z = 3.0 * u.m

gcrs_vx = 0.1 * u.m / u.s
gcrs_vy = 0.2 * u.m / u.s
gcrs_vz = 0.3 * u.m / u.s

# Create a CartesianRepresentation for positions
position = CartesianRepresentation(gcrs_x, gcrs_y, gcrs_z)

# Create a CartesianDifferential for velocities
velocity = CartesianDifferential(gcrs_vx, gcrs_vy, gcrs_vz)

# Combine position and velocity into one representation
gcrs_rep = position.with_differentials(velocity)

# Create a GCRS coordinate object with the combined representation
obstime = Time('2024-06-10T00:00:00')
gcrs_coords = GCRS(gcrs_rep, obstime=obstime)

# Convert to ITRS
itrs_coords = gcrs_coords.transform_to(ITRS(obstime=obstime))

# Extract positions and velocities in ITRS
itrs_position = itrs_coords.cartesian
itrs_velocity = itrs_coords.velocity

# Print the ITRS coordinates and velocities
print(f"GCRS Position (x, y, z): {gcrs_x}, {gcrs_y}, {gcrs_z}")
print(f"GCRS Velocity (vx, vy, vz): {gcrs_vx}, {gcrs_vy}, {gcrs_vz}")

# Print the ITRS coordinates and velocities
print(f"ITRS Position (x, y, z): {itrs_position.x}, {itrs_position.y}, {itrs_position.z}")
print(f"ITRS Velocity (vx, vy, vz): {itrs_velocity.d_x}, {itrs_velocity.d_y}, {itrs_velocity.d_z}")

print(f"Magnitudes: {np.linalg.norm([gcrs_x.value, gcrs_y.value, gcrs_z.value])}, {np.linalg.norm([itrs_position.x.value, itrs_position.y.value, itrs_position.z.value])}")