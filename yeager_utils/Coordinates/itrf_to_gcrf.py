import numpy as np
import erfa
from ssapy.utils import _gpsToTT, iers_interp
from ..Time_Functions import to_gps


def itrf_to_gcrf(r_itrf: np.ndarray, time: np.ndarray) -> np.ndarray:
    """
    Convert positions from ITRF to GCRF, undoing the transformation of groundTrack.

    Parameters
    ----------
    r_itrf : array_like, shape (n, 3)
        Positions in ITRF (meters).
    time : array_like, shape (n,)
        GPS seconds since 1980-01-06 00:00:00 UTC.

    Returns
    -------
    r_gcrf : ndarray, shape (n, 3)
        Positions in GCRF (meters).
    """
    time = to_gps(time)

    # Ensure inputs are numpy arrays with correct shapes
    r_itrf = np.asarray(r_itrf)
    time = np.asarray(time)
    if r_itrf.shape[0] != time.shape[0]:
        raise ValueError("Number of positions must match number of times.")
    if r_itrf.shape[-1] != 3 or time.ndim != 1:
        raise ValueError("r_itrf must be (n, 3) and time must be (n,)")

    n = len(time)

    # Compute time-dependent transformation matrices
    mjd_tt = _gpsToTT(time)
    d_ut1_tt_mjd, pmx, pmy = iers_interp(time)
    pn = erfa.pnm80(2400000.5, mjd_tt)  # Shape (n, 3, 3)
    gst = erfa.gst94(2400000.5, mjd_tt + d_ut1_tt_mjd)
    cg, sg = np.cos(gst), np.sin(gst)

    # Greenwich Sidereal Time rotation matrix
    gstMat = np.zeros((n, 3, 3), dtype=float)
    gstMat[:, 0, 0] = cg
    gstMat[:, 0, 1] = sg
    gstMat[:, 1, 0] = -sg
    gstMat[:, 1, 1] = cg
    gstMat[:, 2, 2] = 1.0

    # Polar motion matrix (approximation)
    polar = np.eye(3, dtype=float)[np.newaxis, :, :].repeat(n, axis=0)
    polar[:, 0, 2] = pmx
    polar[:, 1, 2] = -pmy
    polar[:, 2, 0] = -pmx
    polar[:, 2, 1] = pmy

    # Combined transformation matrix T = polar @ U, where U = gstMat @ pn
    U = np.einsum("tij,tjk->tik", gstMat, pn)  # (n, 3, 3)
    T = np.einsum("tij,tjk->tik", polar, U)    # (n, 3, 3)

    # Inverse transformation: T_inv = T.transpose(0, 2, 1)
    T_inv = T.transpose(0, 2, 1)

    # Apply inverse transformation to each position
    r_gcrf = np.einsum("tij,tj->ti", T_inv, r_itrf)

    return r_gcrf
