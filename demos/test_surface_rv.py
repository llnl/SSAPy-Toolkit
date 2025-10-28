from yeager_utils import surface_rv, Time, get_times, np

# ----------------------------------------------------------------------
# Original single-site trajectory (2D plot, over 1 day)
# ----------------------------------------------------------------------
t0 = Time("2025-5-1")
times = get_times(duration=(1, "day"), freq=(1, "min"))

rs = []
for t in times:
    r, v = surface_rv(lat=0, lon=0, t=t)
    rs.append(r)
rs = np.array(rs)

print(rs[:, 0])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

plt.figure()
plt.plot(rs[:, 0], rs[:, 1], linewidth=1.5)
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Single site trajectory (lat=0, lon=0)")

# ----------------------------------------------------------------------
# 3D plot at a single time (t0), for many lat/lon pairs
# ----------------------------------------------------------------------
lats = np.arange(-90, 91, 10)     # degrees
lons = np.arange(-180, 180, 10)  # degrees

points = []
for lat in lats:
    for lon in lons:
        r, v = surface_rv(lat=lat, lon=lon, t=t0)
        points.append(r)

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Scatter the positions
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=20, c="r")

# Equal aspect ratio for x, y, z
try:
    ax.set_box_aspect((1, 1, 1))
except Exception:
    pass

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title(f"Surface positions at t0 = {t0.isot}")

plt.tight_layout()

