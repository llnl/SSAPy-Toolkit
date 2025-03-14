import numpy as np
from ..time import Time

def v_from_r(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Calculate velocity from position using numerical differentiation.

    Parameters:
    - r (np.ndarray): 3D position array (meters).
    - t (np.ndarray): Time array corresponding to the position data.

    Returns:
    - 3D velocity array (meters per second).
    """
    if isinstance(t[0], Time):
        t = t.gps
    delta_r = np.diff(r, axis=0)
    delta_t = np.diff(t)
    v = delta_r / delta_t[:, np.newaxis]
    v = np.vstack((v, v[-1]))
    return v
