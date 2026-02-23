import numpy as np
import matplotlib.pyplot as plt

def divergence_plot(r_vectors, r_center=None, v_center=None):
    """
    Plot position errors projected onto a plane defined by velocity vector.
    
    Parameters
    ----------
    r_vectors : np.ndarray
        Array of shape (N, 3) containing position vectors
    r_center : np.ndarray, optional
        Reference position vector (3,). If None, uses median of r_vectors.
    v_center : np.ndarray, optional
        Reference velocity vector (3,) that defines the plane normal.
        If None, raises an error.
    """
    # Use median if r_center not provided
    if r_center is None:
        r_center = np.median(r_vectors, axis=0)
    
    # Calculate errors from center
    errors = r_vectors - r_center
    
    # Check that v_center is provided
    if v_center is None:
        raise ValueError("v_center must be provided")
    
    # Normalize the plane normal (velocity vector)
    normal = v_center / np.linalg.norm(v_center)
    
    # Project errors onto the plane (remove component along normal)
    errors_projected = errors - np.outer(np.dot(errors, normal), normal)
    
    # Get two perpendicular basis vectors in the plane
    if abs(normal[0]) < 0.9:
        u = np.cross(normal, [1, 0, 0])
    else:
        u = np.cross(normal, [0, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    
    # Convert 3D projected points to 2D coordinates in the plane
    x_2d = np.dot(errors_projected, u)
    y_2d = np.dot(errors_projected, v)
    
    # Calculate radial norms for statistics
    r_norm = np.sqrt(x_2d**2 + y_2d**2)
    print(f"\nProjected position offsets | "
          f"mean radius = {np.mean(r_norm):.3f}, "
          f"std radius  = {np.std(r_norm):.3f}, "
          f"max radius  = {np.max(r_norm):.3f}")
    
    # Create bullseye plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Define radii for concentric circles
    max_error = max(np.max(np.abs(x_2d)), np.max(np.abs(y_2d)))
    if max_error > 0:
        radii = np.linspace(max_error/5, max_error, 5)
    else:
        radii = [1, 2, 3, 4, 5]  # Default if all points are at center
    colors = ['red', 'white', 'blue', 'white', 'red']
    
    # Draw bullseye rings
    theta = np.linspace(0, 2 * np.pi, 1000)
    for i, radius in enumerate(radii):
        x_circle = radius * np.cos(theta)
        y_circle = radius * np.sin(theta)
        ax.fill(x_circle, y_circle, color=colors[i], alpha=0.7, zorder=len(radii)-i)
    
    # Plot error points
    ax.scatter(x_2d, y_2d, c='black', s=50, zorder=100, edgecolors='white', linewidths=0.5)
    
    # Plot center point
    ax.scatter(0, 0, c='yellow', s=100, marker='x', zorder=101, linewidths=2)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('In-plane error (m)')
    ax.set_ylabel('In-plane error (m)')
    ax.set_title('Position Errors Projected onto Velocity Plane')
    plt.tight_layout()
    plt.show()
    
    return fig

# Usage examples:
# divergence_plot(r_vectors, v_center=velocity_vector)  # Uses median r_center
# divergence_plot(r_vectors, r_center=my_r, v_center=my_v)  # Uses provided centers