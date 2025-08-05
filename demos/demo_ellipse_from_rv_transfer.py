#!/usr/bin/env python3
# feasible_departure_ellipses_simple.py
# ---------------------------------------------------------------
# Find (a,e) pairs that can reach target state with |Δv| ≤ dv_max
# Simplified approach focusing on energy and angular momentum matching

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

MU = 3.986004418e14  # [m^3 s^-2]

# Target state
r_t = np.array([70_000e3, 0.0, 0.0])  # 70,000 km
u_t = np.array([2.0, 1.0, 0.0])       # velocity direction
r_t_mag = np.linalg.norm(r_t)
dv_max = 50.0  # [m/s]

def ellipse_state(a, e, f):
    """Get position and velocity at true anomaly f."""
    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(f))
    
    # Position in perifocal frame
    x = r * np.cos(f)
    y = r * np.sin(f)
    r_vec = np.array([x, y, 0.0])
    
    # Velocity in perifocal frame
    h = np.sqrt(MU * p)
    vx = -h/p * np.sin(f)
    vy = h/p * (e + np.cos(f))
    v_vec = np.array([vx, vy, 0.0])
    
    return r_vec, v_vec

def can_reach_target(a, e, n_points=60):
    """
    Check if ellipse (a,e) has any departure point that can reach target.
    
    For each point on the ellipse:
    1. Calculate current state (r1, v1)
    2. For various arrival speeds s2, calculate the orbit that would
       connect r1 to (r_t, s2*u_t)
    3. Check if the required Δv at departure is within budget
    """
    # Sample points around the ellipse
    true_anomalies = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    # Reasonable arrival speed range (based on circular orbit at target)
    v_circ_target = np.sqrt(MU / r_t_mag)
    arrival_speeds = np.linspace(0.7 * v_circ_target, 1.5 * v_circ_target, 20)
    
    for f in true_anomalies:
        r1, v1 = ellipse_state(a, e, f)
        r1_mag = np.linalg.norm(r1)
        
        # Skip if we're at periapsis/apoapsis of a very eccentric orbit
        # (numerical issues)
        if r1_mag < 1000e3 or r1_mag > 500_000e3:
            continue
        
        for s2 in arrival_speeds:
            v2 = s2 * u_t
            
            # Conservation of energy
            E = 0.5 * s2**2 - MU / r_t_mag
            
            # Required speed at departure
            v1_required_mag2 = 2 * (E + MU / r1_mag)
            if v1_required_mag2 < 0:
                continue
            v1_required_mag = np.sqrt(v1_required_mag2)
            
            # Conservation of angular momentum
            h2 = np.cross(r_t, v2)
            h2_mag = np.linalg.norm(h2)
            
            if h2_mag < 1e-10:  # Nearly radial arrival
                continue
            
            # The departure velocity must produce the same angular momentum
            # h1 = r1 × v1_required = h2
            
            # For coplanar motion, h is perpendicular to the x-y plane
            # |h1| = |r1| * |v1| * sin(angle)
            # where angle is between r1 and v1
            
            # Maximum possible |h1| with given |v1_required|
            h1_max = r1_mag * v1_required_mag
            
            if h2_mag > h1_max:
                continue  # Can't achieve required angular momentum
            
            # Find the flight path angle γ that gives the right h
            sin_gamma = h2_mag / (r1_mag * v1_required_mag)
            cos_gamma = np.sqrt(1 - sin_gamma**2)
            
            # Departure velocity components (in local radial/transverse frame)
            # Two possible solutions (prograde/retrograde)
            r1_unit = r1 / r1_mag
            
            # Transverse direction (perpendicular to r1 in x-y plane)
            t1_unit = np.array([-r1_unit[1], r1_unit[0], 0])
            
            # Try both flight path angle signs
            for sign in [1, -1]:
                v1_radial = v1_required_mag * sign * sin_gamma
                v1_transverse = v1_required_mag * cos_gamma
                
                # Check if this gives the right angular momentum direction
                v1_required = v1_radial * r1_unit + v1_transverse * t1_unit
                h1_test = np.cross(r1, v1_required)
                
                # Check if angular momenta match (magnitude and direction)
                if np.allclose(h1_test, h2, rtol=1e-3):
                    # Calculate required Δv
                    dv = v1_required - v1
                    dv_mag = np.linalg.norm(dv)
                    
                    if dv_mag <= dv_max:
                        return True
    
    return False

# Main search
print(f"Searching for feasible departure ellipses...")
print(f"Target: r = {r_t_mag/1e6:.0f} Mm, velocity direction = {u_t}")
print(f"Max Δv: {dv_max} m/s\n")

# Define search grid
# Focus on ellipses that might plausibly reach the target
a_min = 0.3 * r_t_mag  # Don't go too small
a_max = 2.0 * r_t_mag  # Don't go too large
a_vals = np.linspace(a_min, a_max, 35)

e_vals = np.linspace(0.0, 0.85, 35)  # Avoid very high eccentricity

feasible_pairs = []

# Progress bar
with tqdm(total=len(a_vals)*len(e_vals), desc="Checking ellipses") as pbar:
    for a in a_vals:
        for e in e_vals:
            if can_reach_target(a, e):
                feasible_pairs.append((a, e))
            pbar.update(1)

print(f"\nFound {len(feasible_pairs)} feasible (a,e) pairs")

# Visualization
if feasible_pairs:
    pairs = np.array(feasible_pairs)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Scatter plot of feasible pairs
    ax1.scatter(pairs[:, 0]/1e6, pairs[:, 1], s=40, alpha=0.6, c='blue', 
                edgecolors='darkblue', linewidth=0.5)
    ax1.set_xlabel('Semi-major axis a [Mm]')
    ax1.set_ylabel('Eccentricity e')
    ax1.set_title(f'Feasible Departure Ellipses (Δv ≤ {dv_max} m/s)')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add reference lines
    ax1.axvline(r_t_mag/1e6, color='red', linestyle='--', alpha=0.5,
                label=f'Target radius')
    ax1.legend()
    
    # Plot 2: Density plot
    from scipy.stats import gaussian_kde
    
    if len(pairs) > 10:
        # Create density plot
        x = pairs[:, 0] / 1e6
        y = pairs[:, 1]
        
        # Create grid for density calculation
        xi = np.linspace(a_min/1e6, a_max/1e6, 100)
        yi = np.linspace(0, 0.85, 100)
        xi, yi = np.meshgrid(xi, yi)
        
        # Calculate density
        positions = np.vstack([xi.ravel(), yi.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        zi = kernel(positions).reshape(xi.shape)
        
        # Plot density
        contour = ax2.contourf(xi, yi, zi, levels=20, cmap='viridis', alpha=0.7)
        ax2.contour(xi, yi, zi, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        plt.colorbar(contour, ax=ax2, label='Density')
        
        ax2.set_xlabel('Semi-major axis a [Mm]')
        ax2.set_ylabel('Eccentricity e')
        ax2.set_title('Density of Feasible Orbits')
        ax2.axvline(r_t_mag/1e6, color='red', linestyle='--', alpha=0.5)
    else:
        ax2.text(0.5, 0.5, 'Too few points for density plot', 
                ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig('feasible_departure_ellipses.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    # Statistics
    a_feas = pairs[:, 0] / 1e6
    e_feas = pairs[:, 1]
    print(f"\nStatistics:")
    print(f"  a range: {a_feas.min():.1f} - {a_feas.max():.1f} Mm")
    print(f"  e range: {e_feas.min():.3f} - {e_feas.max():.3f}")
    print(f"  Most common a ≈ {np.median(a_feas):.1f} Mm")
    print(f"  Most common e ≈ {np.median(e_feas):.3f}")
else:
    print("\nNo feasible orbits found!")
    print("Suggestions:")
    print("  - Increase dv_max")
    print("  - Expand search domain")
    print("  - Check target parameters")