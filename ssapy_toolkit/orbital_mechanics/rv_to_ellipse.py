import numpy as np
from ..constants import EARTH_MU, EARTH_RADIUS

try:
    from astropy.time import Time
except ImportError:
    Time = None


def rv_to_ellipse(
    r,
    v,
    *,
    t0=None,                    # epoch echoed as first element of 't_abs'
    num: int | None = None,     # total # samples (≥1)
    mu: float = EARTH_MU,       # GM [m³ s⁻²]
    R_body: float = EARTH_RADIUS
):
    """
    Return the ellipse-arc dictionary used by ellipse_arc.py.
    When `num` ≥ 2, generate that many samples starting with (r0, v0):

        • Elliptic   – advance uniformly in true anomaly through 2π.
        • Non-elliptic – advance until r = 2·|r0|.

    The first sample is always the exact input state.
    """
    # ── convert inputs ───────────────────────────────────────────────
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    # ── orbital invariants ───────────────────────────────────────────
    h_vec = np.cross(r, v);  h = np.linalg.norm(h_vec)
    k_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_hat, h_vec);  n = np.linalg.norm(n_vec)
    e_vec = (np.cross(v, h_vec) / mu) - r / r_mag
    e = np.linalg.norm(e_vec)
    Energy = v_mag**2 / 2.0 - mu / r_mag
    a = -mu / (2.0 * Energy) if Energy < 0 else np.inf
    p = h**2 / mu
    eta = np.sqrt(max(0.0, 1.0 - e**2))
    b = a * eta if Energy < 0 else np.nan

    # ── fundamental angles ──────────────────────────────────────────
    i   = np.arccos(h_vec[2] / h)
    raan = np.arccos(n_vec[0] / n) if n else 0.0
    if n and n_vec[1] < 0: raan = 2*np.pi - raan
    pa = np.arccos(np.clip(np.dot(n_vec, e_vec)/(n*e), -1, 1)) if n and e else 0.0
    if n and e and e_vec[2] < 0: pa = 2*np.pi - pa
    ta = np.arccos(np.clip(np.dot(e_vec, r)/(e*r_mag), -1, 1)) if e else 0.0
    if e and np.dot(r, v) < 0: ta = 2*np.pi - ta

    # mean longitude (elliptic only, needed for period timing)
    if Energy < 0:
        cosE = (e + np.cos(ta)) / (1 + e*np.cos(ta))
        sinE = (np.sin(ta) * eta) / (1 + e*np.cos(ta))
        E0 = np.arctan2(sinE, cosE) % (2*np.pi)
        M0 = E0 - e*np.sin(E0)
        L = (raan + pa + M0) % (2*np.pi)
    else:
        M0 = 0.0
        L = np.nan

    # ── geometric / timing scalars ───────────────────────────────────
    rp = a*(1-e) if Energy < 0 else p/(1+e)
    ra = a*(1+e) if Energy < 0 else np.inf
    rp_alt, ra_alt = rp - R_body, ra - R_body
    mean_motion = np.sqrt(mu/a**3) if Energy < 0 else np.nan
    period = 2*np.pi/mean_motion if Energy < 0 else np.nan

    # in-plane frame
    u_hat = r / r_mag
    w_hat = h_vec / h
    v_hat = np.cross(w_hat, u_hat);  v_hat /= np.linalg.norm(v_hat)
    plane_basis = (u_hat, v_hat, w_hat)
    F2 = 2*a*e_vec

    # ── sampling setup ───────────────────────────────────────────────
    total = max(1, num or 1)          # at least one sample
    elliptical = Energy < 0
    r_samples = np.empty((total, 3))
    v_samples = np.empty((total, 3))
    t_rel = np.zeros(total)

    # first entry is exactly the input state
    r_samples[0] = r
    v_samples[0] = v

    if total > 1:
        if elliptical:
            # uniform Δf through 0→2π, excluding the initial point
            f_steps = np.linspace(0, 2*np.pi, total, endpoint=False)[1:]
        else:
            # determine Δf that brings r to 2·|r0|
            cos_f_lim = (p/(2*r_mag) - 1)/e
            cos_f_lim = np.clip(cos_f_lim, -1.0, 1.0)
            f_lim = np.arccos(cos_f_lim)          # positive angle
            f_steps = np.linspace(0,  f_lim, total)[1:]

        # generate the remaining samples
        for k, df in enumerate(f_steps, start=1):
            f = ta + df
            r_k = p / (1 + e*np.cos(f))
            r_vec = r_k*(np.cos(f)*u_hat + np.sin(f)*v_hat)
            r_samples[k] = r_vec

            # velocity
            v_r = (mu/h)*e*np.sin(f)
            v_t = (mu/h)*(1 + e*np.cos(f))
            r_hat = r_vec / r_k
            t_hat = -np.sin(f)*u_hat + np.cos(f)*v_hat
            v_samples[k] = v_r*r_hat + v_t*t_hat

        # relative times via area law
        delta_f = np.diff(np.concatenate(([0.0], f_steps)))
        r_norm = np.linalg.norm(r_samples, axis=1)
        dt_mid = 0.5*(r_norm[:-1]**2 + r_norm[1:]**2)/h * delta_f
        t_rel[1:] = np.cumsum(dt_mid)

    # ── absolute times array ─────────────────────────────────────────
    if t0 is not None:
        if Time is not None and isinstance(t0, Time):
            t_abs = (t0 + t_rel * t0.unit).astype(object)
        else:
            t_abs = t0 + t_rel
    else:
        t_abs = None

    # ── assemble final dictionary ────────────────────────────────────
    return {
        "r": r_samples,
        "v": v_samples,
        "t_rel": t_rel,
        "t_abs": t_abs,

        "a": a, "e": e, "i": i, "raan": raan, "pa": pa, "ta": ta, "L": L,
        "rp": rp, "ra": ra, "rp_alt": rp_alt, "ra_alt": ra_alt,
        "b": b, "p": p, "mean_motion": mean_motion, "eta": eta, "period": period,
        "h_vec": h_vec, "h": h, "Energy": Energy, "e_vec": e_vec, "n_vec": n_vec,
        "r0": r, "v0": v, "F2": F2,
        "plane_basis": plane_basis, "rot_dir": 1,
        "mu": mu,
    }


# ── quick demo ───────────────────────────────────────────────────────
if __name__ == "__main__":
    r0 = [7000e3, 0.0, 0.0]
    v0 = [0.0, 7.5e3, 1.0e3]
    info = rv_to_ellipse(r0, v0, num=5)
    print("First r sample equals r0?", np.allclose(info["r"][0], r0))
    print("t_rel:", info["t_rel"])
