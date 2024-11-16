# %%
# flake8: noqa: E501
import numpy as np
import matplotlib.pyplot as plt
import yeager_utils as ut

# def graham_scan(points):
#     def orientation(p, q, r):
#         val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
#         if val == 0:
#             return 0
#         return 1 if val > 0 else 2

#     # Select the anchor point with minimum x and minimum y values
#     anchor_point_index = np.lexsort((points[:, 1], points[:, 0]))[0]
#     anchor_point = points[anchor_point_index]

#     # Sort the points based on polar angle and distance from anchor point
#     sorted_points = sorted(points, key=lambda p: (np.arctan2(p[1] - anchor_point[1], p[0] - anchor_point[0]), np.linalg.norm(p - anchor_point)))

#     convex_hull = [anchor_point, sorted_points[0], sorted_points[1]]

#     for i in range(2, len(sorted_points)):
#         while len(convex_hull) > 1 and orientation(convex_hull[-2], convex_hull[-1], sorted_points[i]) != 2:
#             convex_hull.pop()
#         convex_hull.append(sorted_points[i])

#     return np.array(convex_hull)

# Sample 2D points
num_points = 25
points = 20 * np.random.rand(num_points, 2)

# Calculate Convex Hull using Graham's scan
hull_vertices = ut.contours_2d(points, plot=True)

# # Plotting for visualization
# plt.scatter(points[:, 0], points[:, 1], label='Points')
# plt.plot(np.append(hull_vertices[:, 0], hull_vertices[0, 0]),
#          np.append(hull_vertices[:, 1], hull_vertices[0, 1]), 'r--', lw=2)

# plt.fill(hull_vertices[:, 0], hull_vertices[:, 1], alpha=0.2, color='blue', label='Convex Hull Area')
# plt.legend(loc='upper left')
# plt.xlim((0, 20))
# plt.ylim((0, 20))
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title("Bounding Contour using Graham's scan")
# plt.show()
