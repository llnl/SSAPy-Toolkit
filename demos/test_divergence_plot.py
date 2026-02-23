"""
Demo script for divergence_plot - visualizing position error distributions
Uses SSAPy to propagate orbits forward in time and creates an animated GIF
"""

import numpy as np
from astropy.time import Time
from yeager_utils import yufig
from yeager_utils.Plots.divergence_plot import divergence_plot

# Import SSAPy components
from ssapy import Orbit, compute
from ssapy.propagator import SciPyPropagator
from ssapy.accel import AccelKepler
from ssapy.body import get_body

import matplotlib.pyplot as plt
from PIL import Image
import os
import tempfile

# Generate sample data: 100 initial position vectors with random errors
n_samples = 100

# Define a nominal initial position (e.g., GEO orbit position in meters)
r_nominal = np.array([42164000.0, 0.0, 0.0])

# Define a nominal initial velocity (e.g., circular orbit velocity in m/s)
v_nominal = np.array([0.0, 3074.66, 0.0])

# Initial time
t0 = Time("2026-01-01T00:00:00", scale='utc')
t0_gps = t0.gps

# Propagation duration: 24 hours with snapshots every 30 minutes
duration_hours = 24.0
snapshot_interval_minutes = 30
n_snapshots = int((duration_hours * 60) / snapshot_interval_minutes) + 1

# Generate time array for snapshots
times_gps = np.linspace(t0_gps, t0_gps + duration_hours * 3600.0, n_snapshots)

# Generate initial position vectors with random errors
pos_radius = 100.0  # Position error radius in meters
vel_radius = 1.0    # Velocity error radius in m/s

r_initial = []
v_initial = []

for i in range(n_samples):
    # Generate random position error (uniformly inside a 3D ball)
    pos_direction = np.random.randn(3)
    pos_direction /= np.linalg.norm(pos_direction)
    pos_magnitude = pos_radius * np.cbrt(np.random.uniform(0, 1))
    pos_error = pos_direction * pos_magnitude
    
    # Generate random velocity error (uniformly inside a 3D ball)
    vel_direction = np.random.randn(3)
    vel_direction /= np.linalg.norm(vel_direction)
    vel_magnitude = vel_radius * np.cbrt(np.random.uniform(0, 1))
    vel_error = vel_direction * vel_magnitude
    
    r_initial.append(r_nominal + pos_error)
    v_initial.append(v_nominal + vel_error)

# Set up propagator with Keplerian acceleration
earth = get_body("earth")
accel = AccelKepler(earth.mu)
propagator = SciPyPropagator(accel)

# Propagate nominal orbit for all snapshot times
orbit_nominal = Orbit(r=r_nominal, v=v_nominal, t=t0_gps)
r_nominal_hist, v_nominal_hist = compute.rv(orbit_nominal, times_gps, propagator)

# Propagate each orbit and collect states at all snapshot times
print(f"Propagating {n_samples} orbits for {duration_hours} hours with {n_snapshots} snapshots...")

r_histories = []
v_histories = []

for i in range(n_samples):
    # Create orbit object
    orbit = Orbit(r=r_initial[i], v=v_initial[i], t=t0_gps)
    
    # Propagate to all snapshot times
    r_hist, v_hist = compute.rv(orbit, times_gps, propagator)
    
    r_histories.append(r_hist)
    v_histories.append(v_hist)

print(f"Propagation complete. Creating animated GIF...")

# Convert to numpy arrays (n_samples, n_snapshots, 3)
r_histories = np.array(r_histories)
v_histories = np.array(v_histories)

# Calculate global max error for consistent axis limits
# Need to compute max of projected N-W errors, not full 3D errors
print("Computing global axis limits from projected errors...")
max_nw_errors = []

for snap_idx in range(n_snapshots):
    r_center = r_nominal_hist[snap_idx]
    v_center = v_nominal_hist[snap_idx]
    errors = r_histories[:, snap_idx, :] - r_center
    
    # Compute NTW coordinate system
    t_hat = v_center / np.linalg.norm(v_center)
    w_hat = np.cross(r_center, v_center)
    w_hat = w_hat / np.linalg.norm(w_hat)
    n_hat = np.cross(w_hat, t_hat)
    
    # Project errors
    errors_projected = errors - np.outer(np.dot(errors, t_hat), t_hat)
    
    # Convert to 2D
    x_2d = np.dot(errors_projected, n_hat)
    y_2d = np.dot(errors_projected, w_hat)
    
    # Track max in N-W plane
    max_nw_errors.append(max(np.max(np.abs(x_2d)), np.max(np.abs(y_2d))))

all_max = max(max_nw_errors)
print(f"Global max N-W error: {all_max:.2f} m")

# Create temporary directory for frame images
temp_dir = tempfile.mkdtemp()
frame_files = []

# Generate frames for each snapshot
for snap_idx in range(n_snapshots):
    time_hours = (times_gps[snap_idx] - t0_gps) / 3600.0
    
    # Get positions and velocities at this snapshot
    r_snap = r_histories[:, snap_idx, :]
    v_snap = v_histories[:, snap_idx, :]
    
    # Get nominal state at this snapshot
    r_center = r_nominal_hist[snap_idx]
    v_center = v_nominal_hist[snap_idx]
    
    # Create divergence plot (without showing it)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate errors
    errors = r_snap - r_center
    
    # Compute NTW coordinate system
    t_hat = v_center / np.linalg.norm(v_center)
    w_hat = np.cross(r_center, v_center)
    w_hat = w_hat / np.linalg.norm(w_hat)
    n_hat = np.cross(w_hat, t_hat)
    
    # Project errors
    errors_projected = errors - np.outer(np.dot(errors, t_hat), t_hat)
    t_component = np.dot(errors, t_hat)
    
    # Convert to timing difference
    v_magnitude = np.linalg.norm(v_center)
    timing_difference = t_component / v_magnitude
    
    # Convert to 2D
    x_2d = np.dot(errors_projected, n_hat)
    y_2d = np.dot(errors_projected, w_hat)
    
    # Plot
    scatter = ax.scatter(x_2d, y_2d, c=timing_difference, s=80, 
                        cmap='RdYlBu_r', zorder=100, 
                        edgecolors='black', linewidths=0.8,
                        alpha=0.8, vmin=-50, vmax=50)
    
    cbar = fig.colorbar(scatter, ax=ax, label='Timing difference [seconds]', 
                       shrink=0.8, pad=0.02)
    
    ax.scatter(0, 0, c='red', s=150, marker='x', zorder=101, 
              linewidths=3, label='Reference center')
    
    # Add reference circles
    max_error_snap = max(np.max(np.abs(x_2d)), np.max(np.abs(y_2d)))
    if max_error_snap > 0:
        circle_radii = np.linspace(max_error_snap/4, max_error_snap, 4)
        theta = np.linspace(0, 2 * np.pi, 100)
        for radius in circle_radii:
            x_circle = radius * np.cos(theta)
            y_circle = radius * np.sin(theta)
            ax.plot(x_circle, y_circle, 'k--', alpha=0.2, linewidth=0.8)
    
    # Set consistent axis limits across all frames based on projected errors
    ax.set_xlim(-all_max * 1.1, all_max * 1.1)
    ax.set_ylim(-all_max * 1.1, all_max * 1.1)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlabel('Normal (N) error [m]', fontsize=12)
    ax.set_ylabel('Cross-track (W) error [m]', fontsize=12)
    ax.set_title(f'Position Errors at T+{time_hours:.1f} hours', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save frame
    frame_path = os.path.join(temp_dir, f'frame_{snap_idx:03d}.png')
    fig.savefig(frame_path, dpi=100, bbox_inches='tight')
    frame_files.append(frame_path)
    plt.close(fig)
    
    if snap_idx % 10 == 0:
        print(f"  Generated frame {snap_idx+1}/{n_snapshots}")

# Create GIF from frames
print("Creating GIF from frames...")
frames = [Image.open(frame_file) for frame_file in frame_files]

# Save as GIF
gif_path = os.path.expanduser("~/yu_figures/tests/divergence_animation.gif")
os.makedirs(os.path.dirname(gif_path), exist_ok=True)

frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=200,  # milliseconds per frame
    loop=0
)

print(f"GIF saved to: {gif_path}")

# Clean up temporary files
for frame_file in frame_files:
    os.remove(frame_file)
os.rmdir(temp_dir)

# Also create final snapshot plots using divergence_plot function
print("\nCreating final snapshot plots...")

# Final state
r_final = r_histories[:, -1, :]
v_final = v_histories[:, -1, :]
v_center = v_nominal_hist[-1]

fig1 = divergence_plot(r_final, v_center=v_center, 
                       title=f'Position Errors at T+{duration_hours:.0f} hours (Median Center)')
yufig(fig1, "tests/divergence_plot_final_median")

fig2 = divergence_plot(r_final, r_center=r_nominal_hist[-1], v_center=v_nominal_hist[-1],
                       title=f'Position Errors at T+{duration_hours:.0f} hours (Nominal Center)')
yufig(fig2, "tests/divergence_plot_final_explicit")

print("\nPlots and animation saved to ~/yu_figures/tests/")
print(f"Final position spread: {np.std(np.linalg.norm(r_final, axis=1)):.2f} m")



print("\n" + "="*70)
print("TESTING NEW divergence_gif FUNCTION")
print("="*70)

from yeager_utils.Plots.divergence_gif import divergence_gif

# Test 1: Using the function with explicit nominal trajectories
print("\nTest 1: Creating GIF with explicit nominal trajectories...")
gif_path_1 = divergence_gif(
    r_histories=r_histories,
    times_gps=times_gps,
    output_path="~/yu_figures/tests/divergence_test_explicit.gif",
    r_nominal_hist=r_nominal_hist,
    v_nominal_hist=v_nominal_hist,
    duration=0.2,  # 200ms per frame
    vmin=-50,
    vmax=50
)
print(f"Test 1 complete: {gif_path_1}")

# Test 2: Using the function with automatic median calculation
print("\nTest 2: Creating GIF with automatic median center...")
gif_path_2 = divergence_gif(
    r_histories=r_histories,
    times_gps=times_gps,
    output_path="~/yu_figures/tests/divergence_test_median.gif",
    fps=5  # Use fps instead of duration
)
print(f"Test 2 complete: {gif_path_2}")

# Test 3: Custom colorbar limits
print("\nTest 3: Creating GIF with custom timing colorbar limits...")
gif_path_3 = divergence_gif(
    r_histories=r_histories,
    times_gps=times_gps,
    output_path="~/yu_figures/tests/divergence_test_custom_colors.gif",
    r_nominal_hist=r_nominal_hist,
    v_nominal_hist=v_nominal_hist,
    duration=0.15,  # Faster animation
    vmin=-100,  # Wider color range
    vmax=100
)
print(f"Test 3 complete: {gif_path_3}")

print("\n" + "="*70)
print("All divergence_gif tests completed successfully!")
print("="*70)