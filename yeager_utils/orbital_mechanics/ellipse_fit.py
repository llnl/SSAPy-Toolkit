import numpy as np

def ellipse_from_state(
    r,
    v,
    *,
    mu=3.986004418e14,          # Earth GM  [m^3 s^-2]
    R_body=6_378_136.0          # Earth mean radius [m]
):
    """
    Derive full ellipse information from a single Cartesian state vector,
    returning the same key names used in ellipse_arc.py.
    """
    # ensure ndarray
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    # magnitudes
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    # angular momentum & node vectors
    h_vec = np.cross(r, v)
    h     = np.linalg.norm(h_vec)
    k_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_hat, h_vec)
    n     = np.linalg.norm(n_vec)

    # eccentricity vector
    e_vec = (np.cross(v, h_vec) / mu) - r / r_mag
    e     = np.linalg.norm(e_vec)

    # specific energy
    Energy = v_mag**2 / 2.0 - mu / r_mag

    # semi-major axis (elliptic if Energy < 0)
    a = -mu / (2.0 * Energy) if Energy < 0 else np.inf

    # plane basis (u in-plane toward r̂, v in-plane 90°, w = ĥ)
    u_hat = r / r_mag
    w_hat = h_vec / h
    v_hat = np.cross(w_hat, u_hat)
    v_hat /= np.linalg.norm(v_hat)
    plane_basis = (u_hat, v_hat, w_hat)

    # inclination
    i = np.arccos(h_vec[2] / h)

    # RAAN
    if n != 0.0:
        raan = np.arccos(n_vec[0] / n)
        if n_vec[1] < 0:
            raan = 2.0 * np.pi - raan
    else:
        raan = 0.0

    # argument of periapsis
    if n != 0.0 and e > 0.0:
        pa = np.arccos(np.dot(n_vec, e_vec) / (n * e))
        if e_vec[2] < 0:
            pa = 2.0 * np.pi - pa
    else:
        pa = 0.0

    # true anomaly
    if e > 0.0:
        ta = np.arccos(np.dot(e_vec, r) / (e * r_mag))
        if np.dot(r, v) < 0:
            ta = 2.0 * np.pi - ta
    else:
        ta = 0.0

    # mean longitude L = raan + pa + M
    if Energy < 0.0:
        cosE = (e + np.cos(ta)) / (1.0 + e * np.cos(ta))
        sinE = (np.sin(ta) * np.sqrt(1.0 - e**2)) / (1.0 + e * np.cos(ta))
        E_anom = np.arctan2(sinE, cosE)
        if E_anom < 0.0:
            E_anom += 2.0 * np.pi
        M = E_anom - e * np.sin(E_anom)
        L = (raan + pa + M) % (2.0 * np.pi)
    else:
        L = np.nan

    # ellipse geometry
    p   = h**2 / mu                    # semi-latus rectum
    eta = np.sqrt(max(0.0, 1.0 - e**2))
    b   = a * eta if Energy < 0 else np.nan
    rp  = a * (1.0 - e) if Energy < 0 else p / (1.0 + e)
    ra  = a * (1.0 + e) if Energy < 0 else np.inf
    rp_alt = rp - R_body
    ra_alt = ra - R_body

    # motion
    mean_motion = np.sqrt(mu / a**3) if Energy < 0 else np.nan
    period      = 2.0 * np.pi / mean_motion if Energy < 0 else np.nan

    # second focus (first focus at origin)
    F2 = 2.0 * a * e_vec               # |F2| = 2ae

    # rotation sense (+1 CCW when viewed along +w_hat)
    rot_dir = 1                        # always +1 with this definition

    # assemble dict with original key names
    return {
        # trajectory placeholders (single sample)
        "r"           : np.atleast_2d(r),       # shape (1,3)
        "v"           : np.atleast_2d(v),       # shape (1,3)
        "t_rel"       : np.array([0.0]),
        "t_abs"       : None,

        # classical elements
        "a"           : a,
        "e"           : e,
        "i"           : i,
        "raan"        : raan,
        "pa"          : pa,
        "ta"          : ta,
        "L"           : L,

        # handy scalars
        "rp"          : rp,
        "ra"          : ra,
        "rp_alt"      : rp_alt,
        "ra_alt"      : ra_alt,
        "b"           : b,
        "p"           : p,
        "mean_motion" : mean_motion,
        "eta"         : eta,
        "period"      : period,

        # vectors / invariants
        "h_vec"       : h_vec,
        "h"           : h,
        "Energy"      : Energy,
        "e_vec"       : e_vec,
        "n_vec"       : n_vec,

        # convenience state
        "r0"          : r,
        "v0"          : v,
        "F2"          : F2,

        # plane / rotation
        "plane_basis" : plane_basis,
        "rot_dir"     : rot_dir,

        # constant
        "mu"          : mu,
    }

# quick self-test
if __name__ == "__main__":
    r0 = [7000e3, 0.0, 0.0]
    v0 = [0.0, 7.5e3, 1.0e3]
    info = ellipse_from_state(r0, v0)
    print("semi-major axis a [km]:", info["a"] / 1e3)
    print("eccentricity e        :", info["e"])
