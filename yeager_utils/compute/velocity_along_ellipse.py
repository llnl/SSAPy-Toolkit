# ──────────────────────────────────────────────────────────────────────────────
#   Orbital-velocity helper
# ──────────────────────────────────────────────────────────────────────────────
from ..constants import EARTH_MU


def velocity_along_ellipse(arc3d, *, a, e, F2, mu=EARTH_MU):
    """
    Compute the 3-D inertial velocity at every point of an ellipse arc.

    Parameters
    ----------
    arc3d : (N, 3) ndarray      positions returned by `ellipse_arc`
    a     : float               semi-major axis (from ellipse_arc info)
    e     : float               eccentricity    (from ellipse_arc info)
    F2    : (3,)  array-like    2nd focus vector (from ellipse_arc info)
    mu    : float               GM of the central body (defaults to Earth)

    Returns
    -------
    v3d   : (N, 3) ndarray      inertial velocity vectors [m s⁻¹]
    """
    arc3d = np.asarray(arc3d, float)
    if arc3d.ndim != 2 or arc3d.shape[1] != 3:
        raise ValueError("arc3d must be shape-(N, 3)")

    # orbital plane triad  (u, v) came from _plane_basis; rebuild it
    F2 = np.asarray(F2, float)
    u = arc3d[0] / np.linalg.norm(arc3d[0])          # same 'u' as before
    v = np.cross(F2, u);  v -= np.dot(v, u)*u        # any perp in the plane
    v /= np.linalg.norm(v)
    w = np.cross(u, v)                               # angular-momentum direction

    # specific angular-momentum magnitude
    h = np.sqrt(mu * a * (1.0 - e*e))

    # eccentric-vector direction (points to periapsis)
    e_hat = F2 / np.linalg.norm(F2)

    v_out = np.empty_like(arc3d)
    for i, r_vec in enumerate(arc3d):
        r = np.linalg.norm(r_vec)
        r_hat = r_vec / r

        # true anomaly f  via polar-angle formula (robust sign handling)
        cos_f = np.dot(e_hat, r_hat)
        sin_f = np.dot(w, np.cross(e_hat, r_hat))
        f = np.arctan2(sin_f, cos_f)

        # radial / transverse components in the perifocal frame
        v_r = (mu / h) * e * np.sin(f)
        v_t = (mu / h) * (1.0 + e * np.cos(f))

        t_hat = np.cross(w, r_hat)                   # tangent in-plane unit
        v_out[i] = v_r * r_hat + v_t * t_hat

    return v_out
