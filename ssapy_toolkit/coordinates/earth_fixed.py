import erfa
import numpy as np
from ssapy import groundTrack
from ssapy.utils import _gpsToTT, iers_interp

from ..time_functions import to_gps
from .velocity import v_from_r


def gcrf_to_itrf(r_gcrf, t, v=None):
    """
    Convert GCRF coordinates to ITRF coordinates.

    Parameters:
    - r_gcrf (np.ndarray): 3D position vector in GCRF coordinates (meters).
    - t (np.ndarray): Time array for conversion.
    - v (np.ndarray, optional): Velocity vector in GCRF coordinates (m/s). Optional.

    Returns:
    - np.ndarray: Position in ITRF coordinates,
      or (position, velocity) in ITRF coordinates if velocity is provided.
    """
    t = to_gps(t)
    x, y, z = groundTrack(r_gcrf, t, format="cartesian")
    pos = np.array([x, y, z]).T
    if v is None:
        return pos
    return pos, v_from_r(pos, t)


def gcrf_to_itrf_astropy(state_vectors, t):
    """Convert GCRF positions to geocentric ITRF using Astropy."""
    import astropy.units as u
    from astropy.coordinates import GCRS, ITRS, SkyCoord

    state_vectors = np.asarray(state_vectors, dtype=float)
    if state_vectors.ndim != 2 or state_vectors.shape[1] != 3:
        raise ValueError("state_vectors must have shape (N, 3)")

    sc = SkyCoord(
        x=state_vectors[:, 0] * u.m,
        y=state_vectors[:, 1] * u.m,
        z=state_vectors[:, 2] * u.m,
        representation_type="cartesian",
        frame=GCRS(obstime=t),
    )
    sc_itrs = sc.transform_to(ITRS(obstime=t))

    return np.array([
        sc_itrs.cartesian.x.to_value(u.m),
        sc_itrs.cartesian.y.to_value(u.m),
        sc_itrs.cartesian.z.to_value(u.m),
    ]).T


def itrf_to_gcrf(r_itrf: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Convert ITRF positions to GCRF, undoing the SSAPy ``groundTrack`` transform."""
    time = to_gps(time)

    r_itrf = np.asarray(r_itrf)
    time = np.asarray(time)
    if r_itrf.shape[0] != time.shape[0]:
        raise ValueError("Number of positions must match number of times.")
    if r_itrf.shape[-1] != 3 or time.ndim != 1:
        raise ValueError("r_itrf must be (n, 3) and time must be (n,)")

    n = len(time)
    mjd_tt = _gpsToTT(time)
    d_ut1_tt_mjd, pmx, pmy = iers_interp(time)
    pn = erfa.pnm80(2400000.5, mjd_tt)
    gst = erfa.gst94(2400000.5, mjd_tt + d_ut1_tt_mjd)
    cg, sg = np.cos(gst), np.sin(gst)

    gst_mat = np.zeros((n, 3, 3), dtype=float)
    gst_mat[:, 0, 0] = cg
    gst_mat[:, 0, 1] = sg
    gst_mat[:, 1, 0] = -sg
    gst_mat[:, 1, 1] = cg
    gst_mat[:, 2, 2] = 1.0

    polar = np.eye(3, dtype=float)[np.newaxis, :, :].repeat(n, axis=0)
    polar[:, 0, 2] = pmx
    polar[:, 1, 2] = -pmy
    polar[:, 2, 0] = -pmx
    polar[:, 2, 1] = pmy

    u = np.einsum("tij,tjk->tik", gst_mat, pn)
    transform = np.einsum("tij,tjk->tik", polar, u)
    return np.einsum("tij,tj->ti", transform.transpose(0, 2, 1), r_itrf)
