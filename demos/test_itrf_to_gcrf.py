import numpy as np
from yeager_utils import gcrf_to_itrf, itrf_to_gcrf, get_times, ssapy_orbit, RGEO
import matplotlib.pyplot as plt


def test_coordinate_transforms():
    t = get_times(duration=(1, 'days'), freq=(1, 'min'))
    r_gcrf_orig, v = ssapy_orbit(a=2 * RGEO, e=0.3, t=t)

    r_itrf = gcrf_to_itrf(r_gcrf_orig, t)
    print("Original GCRF:\n", r_gcrf_orig)
    print("Converted ITRF:\n", r_itrf)

    r_gcrf_back = itrf_to_gcrf(r_itrf, t)
    print("Converted back to GCRF:\n", r_gcrf_back)

    tolerance = 1e-8 * np.max(r_gcrf_orig)
    difference = np.max(np.abs(r_gcrf_orig - r_gcrf_back))
    print(f"Maximum difference: {difference}, tolerance: {tolerance}")
    if difference < tolerance:
        print("Test passed: GCRF -> ITRF -> GCRF transformation is consistent.")
    else:
        print("Test failed: Transformations are not inverses within tolerance.")
        print("Differences:\n", r_gcrf_orig - r_gcrf_back)

    # Plotting all "orbits" in a single 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define labels for each trajectory
    labels = ['Point 1', 'Point 2', 'Point 3', 'Point 4', 'Point 5']

    # Plot original GCRF positions
    ax.scatter(r_gcrf_orig[:, 0] / RGEO, r_gcrf_orig[:, 1] / RGEO, r_gcrf_orig[:, 2] / RGEO, s=100,
                c='b', marker='o', label=f'GCRF')

    # Plot ITRF positions
    ax.scatter(r_itrf[:, 0] / RGEO, r_itrf[:, 1] / RGEO, r_itrf[:, 2] / RGEO, s=300,
                c='r', marker='^', label=f'ITRF')

    # Plot converted-back GCRF positions
    ax.scatter(r_gcrf_back[:, 0] / RGEO, r_gcrf_back[:, 1] / RGEO, r_gcrf_back[:, 2] / RGEO, s=20,
                c='g', marker='s', label=f'GCRF Back')

    # Labeling axes
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('GCRF and ITRF Coordinate Transformations')

    # Add legend (only showing one entry per type to avoid clutter)
    ax.legend()

    # Adjust view for better visibility
    ax.view_init(elev=20, azim=45)

    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    test_coordinate_transforms()
