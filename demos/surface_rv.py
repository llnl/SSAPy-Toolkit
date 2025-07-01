from yeager_utils import surface_rv, Time, get_times, np

t0 = Time("2025-5-1")
times = get_times(duration=(1, 'day'), freq=(1, 'min'))
rs = []
for t in times:
    r, v = surface_rv(lat=0, lon=0, t=t)
    rs.append(r)

rs = np.array(rs)
print(rs[:, 0])

import matplotlib.pyplot as plt

plt.plot(rs[:, 0], rs[:, 1])
plt.axis('equal')