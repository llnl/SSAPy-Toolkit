"""
Demo script for divergence_plot - visualizing position error distributions
"""

import numpy as np
from yeager_utils import yufig
from yeager_utils.Plots.divergence_plot import divergence_plot

# Generate sample data: 25 position vectors with random errors
n_samples = 25

# Define a nominal position (e.g., GEO orbit position in meters)
r_nominal = np.array([42164000.0, 0.0, 0.0])

# Define a nominal velocity (e.g., circular orbit velocity in m/s)
v_nominal = np.array([0.0, 3074.66, 0.0])

# Generate position vectors with random errors
pos_radius = 100.0  # Position error radius in meters
r_vectors = np.zeros((n_samples, 3))

for i in range(n_samples):
    # Generate random position error (uniformly inside a 3D ball)
    pos_direction = np.random.randn(3)
    pos_direction /= np.linalg.norm(pos_direction)
    pos_magnitude = pos_radius * np.cbrt(np.random.uniform(0, 1))
    pos_error = pos_direction * pos_magnitude
    r_vectors[i] = r_nominal + pos_error

# Create divergence plot using median as center
print("Example 1: Using median center and provided velocity")
fig1 = divergence_plot(r_vectors, v_center=v_nominal)
yufig(fig1, "tests/divergence_plot_median")

# Create divergence plot with explicit center
print("\nExample 2: Using explicit center position and velocity")
fig2 = divergence_plot(r_vectors, r_center=r_nominal, v_center=v_nominal)
yufig(fig2, "tests/divergence_plot_explicit")

print("\nPlots saved to ~/yu_figures/tests/")