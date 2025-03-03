import rebound
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define physical constants in SI units
G = 6.67430e-11      # Gravitational constant (m^3 kg^-1 s^-2)
M_earth = 5.972e24   # Mass of Earth (kg)
R_earth = 6.371e6    # Radius of Earth (m)

# Initialize the REBOUND simulation
sim = rebound.Simulation()

# Add the Earth as a central point mass at the origin
sim.add(m=M_earth, x=0, y=0, z=0)

# Add the object (mass negligible, so gravitational effect on Earth is ignored)
# Initial position at Earth's surface along x-axis, zero velocity
sim.add(m=1e-20, x=R_earth, y=0, z=0, vx=0, vy=8000, vz=0)  # Small mass to avoid affecting Earth

# Set the gravitational constant (SI units)
sim.G = G

# Set the integrator to IAS15 (adaptive, high-precision)
sim.integrator = "ias15"

# Define time array for integration (0 to 5000 seconds, ~period of oscillation)
times = np.linspace(0, 15000, 1000)

# Store positions over time
positions = []

# Integrate the simulation and collect positions
for t in times:
    print(t)
    sim.integrate(t)
    particle = sim.particles[1]  # Index 1 is the object (index 0 is Earth)
    positions.append([particle.x, particle.y, particle.z])

# Convert positions to a NumPy array for easier handling
positions = np.array(positions)

# Create a 3D plot of the path
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Path of Object', color='blue')

# Plot Earth's surface as a wireframe sphere for reference
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
x = R_earth * np.outer(np.cos(u), np.sin(v))
y = R_earth * np.outer(np.sin(u), np.sin(v))
z = R_earth * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x, y, z, color='gray', alpha=0.3, label='Earth Surface')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Path of Object in Free Fall Through Earth (Point Source)')
plt.axis('equal')
ax.legend()
plt.show()
