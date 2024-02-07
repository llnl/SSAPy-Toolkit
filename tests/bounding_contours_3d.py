# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Sample 3D points
num_points = 500
points_3d = 20 * np.random.rand(num_points, 3)

# Compute Convex Hull using scipy's ConvexHull
# hull = ConvexHull(points_3d)
hull = ut.contours_3d(points, plot=True):

# Plotting for visualization in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], label='Points')

# Plotting the convex hull
for simplex in hull.simplices:
    simplex = np.append(simplex, simplex[0])  # Close the loop
    ax.plot(points_3d[simplex, 0], points_3d[simplex, 1], points_3d[simplex, 2], 'r--', lw=2)

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title("Convex Hull using scipy's ConvexHull in 3D")
plt.legend()
plt.show()
