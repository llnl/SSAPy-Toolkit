from .ntw_to_gcrf import ntw_to_gcrf_matrix


def gcrf_to_ntw(delta_v_gcrf, r_center, v_center):
    """
    Transform a vector from GCRF to NTW coordinates.

    Parameters
    ----------
    delta_v_gcrf : array_like
        Vector in GCRF coordinates.
    r_center : array_like
        Position vector in GCRF (m).
    v_center : array_like
        Velocity vector in GCRF (m/s).

    Returns
    -------
    delta_v_ntw : ndarray
        Vector in NTW coordinates.
    """
    R_ntw_to_gcrf = ntw_to_gcrf_matrix(r_center, v_center)
    return R_ntw_to_gcrf.T @ delta_v_gcrf
