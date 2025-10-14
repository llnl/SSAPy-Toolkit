from yeager_utils import RGEO, ssapy_orbit
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, CartesianDifferential
from astropy.time import Time
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

# def gcrs_to_itrs(positions, velocities, obstimes):
#     """
#     Convert GCRS coordinates and velocities to ITRS.

#     Parameters:
#     obstimes (astropy.time.Time): Array of observation times.
#     positions (numpy.ndarray): Array of shape (n, 3) containing GCRS positions.
#     velocities (numpy.ndarray): Array of shape (n, 3) containing GCRS velocities.

#     Returns:
#     itrs_positions (numpy.ndarray): Array of shape (n, 3) containing ITRS positions in meters.
#     itrs_velocities (numpy.ndarray): Array of shape (n, 3) containing ITRS velocities in meters per second.
#     """
#     # Ensure positions and velocities are quantities
#     positions = positions * u.m
#     velocities = velocities * (u.m / u.s)

#     # Create CartesianRepresentations for positions and velocities
#     positions_rep = CartesianRepresentation(positions[:, 0], positions[:, 1], positions[:, 2])
#     velocities_diff = CartesianDifferential(velocities[:, 0], velocities[:, 1], velocities[:, 2])

#     # Combine into one representation with differentials
#     gcrs_rep = positions_rep.with_differentials(velocities_diff)

#     # Create GCRS coordinate objects
#     gcrs_coords = GCRS(gcrs_rep, obstime=obstimes[:, np.newaxis])

#     # Convert to ITRS
#     itrs_coords = gcrs_coords.transform_to(ITRS(obstime=obstimes[:, np.newaxis]))

#     # Extract positions and velocities in ITRS and ensure they are in SI units
#     itrs_positions = np.stack((itrs_coords.cartesian.x.to(u.m).value,
#                                itrs_coords.cartesian.y.to(u.m).value,
#                                itrs_coords.cartesian.z.to(u.m).value), axis=-1)
    
#     itrs_velocities = np.stack((itrs_coords.velocity.d_x.to(u.m / u.s).value,
#                                 itrs_coords.velocity.d_y.to(u.m / u.s).value,
#                                 itrs_coords.velocity.d_z.to(u.m / u.s).value), axis=-1)
#     return itrs_positions, itrs_velocities

# Example usage:
r, v, t = ssapy_orbit(a=RGEO, e=0, i=np.pi / 4, pa=0, raan=0, ta=0, 
                      duration=(30, 'day'), freq=(1, 'min'), start_date="2025-01-01")

# itrs_positions, itrs_velocities = gcrs_to_itrs(r, v, t)

# print("ITRS Positions:", itrs_positions, np.shape(itrs_positions))
# print("ITRS Velocities:", itrs_velocities, np.shape(itrs_velocities))


from yeager_utils import RGEO, rv_gcrf_to_itrf
r_itrf, v_itrf = rv_gcrf_to_itrf(r, t)

fig = plt.figure(figsize=(12, 6))

# 3D scatter plot for r
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(r[:, 0] / RGEO, r[:, 1] / RGEO, r[:, 2] / RGEO)

ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax1.set_zlabel('Z Label')

# Setting equal axis lengths
ax1.set_xlim((-1, 1))
ax1.set_ylim((-1, 1))
ax1.set_zlim((-1, 1))

# Subplot for r_itrf
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(r_itrf[:, 0] / RGEO, r_itrf[:, 1] / RGEO, r_itrf[:, 2] / RGEO)

ax2.set_xlabel('X Label')
ax2.set_ylabel('Y Label')
ax2.set_zlabel('Z Label')

# Setting equal axis lengths
ax2.set_xlim((-1, 1))
ax2.set_ylim((-1, 1))
ax2.set_zlim((-1, 1))

plt.tight_layout()
plt.show()

print(v_itrf)