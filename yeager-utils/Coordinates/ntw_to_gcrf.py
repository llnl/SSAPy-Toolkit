import numpy as np


def ntw_to_gcrf_matrix(r, v):
    """
    Compute the rotation matrix from NTW to GCRF given position and velocity in GCRF.

    Parameters
    ----------
    r : array_like
        Position vector in GCRF (m).
    v : array_like
        Velocity vector in GCRF (m/s).

    Returns
    -------
    R_ntw_to_gcrf : ndarray
        3x3 rotation matrix such that R @ vector_ntw = vector_gcrf

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    r = np.asarray(r)
    v = np.asarray(v)
    e_T = v / np.linalg.norm(v)          # Tangential (along velocity)
    h = np.cross(r, v)                   # Angular momentum
    e_W = h / np.linalg.norm(h)          # Out-of-plane
    e_N = np.cross(e_W, e_T)             # Normal
    return np.column_stack((e_N, e_T, e_W))


def ntw_to_gcrf(delta_v_ntw, r_center, v_center):
    R_ntw_to_gcrf = ntw_to_gcrf_matrix(r_center, v_center)
    return R_ntw_to_gcrf @ delta_v_ntw
