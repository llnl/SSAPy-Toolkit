import numpy as np
from ..constants import EARTH_MU


def accel_point_earth(r):
    x, y, z = r
    r_mag = np.sqrt(x**2 + y**2 + z**2)
    factor = -EARTH_MU / (r_mag**3)
    return np.array([factor * x, factor * y, factor * z])
