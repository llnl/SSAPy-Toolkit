import numpy as np
from scipy.interpolate import interp1d
from ..constants import EARTH_MU, MOON_MU, SUN_MU
from ..time import to_gps


def accel_point_earth(r):
    x, y, z = r
    r_mag = np.sqrt(x**2 + y**2 + z**2)
    factor = -EARTH_MU / r_mag**3
    return np.array([factor * x, factor * y, factor * z])


def precompute_third_body_positions(t, body_name):
    from ssapy import get_body
    body = get_body(body_name)
    r_body = body.position(t).T  # shape (n, 3)
    t_gps = to_gps(t)

    interp_funcs = [
        interp1d(t_gps, r_body[:, i], kind="cubic", fill_value="extrapolate")
        for i in range(3)
    ]

    def interpolated_position(t_query):
        tq = to_gps(t_query)
        return np.stack([f(tq) for f in interp_funcs], axis=-1)

    return interpolated_position


def accel_moon(r, t, interp_pos_moon):
    r_moon = interp_pos_moon(t)
    r_sb = r_moon - r
    norm_sb = np.linalg.norm(r_sb, axis=1)[:, np.newaxis]
    norm_b = np.linalg.norm(r_moon, axis=1)[:, np.newaxis]
    a = MOON_MU * (r_sb / norm_sb**3 - r_moon / norm_b**3)
    return a


def accel_sun(r, t, interp_pos_sun):
    r_sun = interp_pos_sun(t)
    r_sb = r_sun - r
    norm_sb = np.linalg.norm(r_sb, axis=1)[:, np.newaxis]
    norm_b = np.linalg.norm(r_sun, axis=1)[:, np.newaxis]
    a = SUN_MU * (r_sb / norm_sb**3 - r_sun / norm_b**3)
    return a


def accel_velocity(v, thrust_mag):
    v = np.asarray(v)
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return np.zeros(3)
    return thrust_mag * v / norm


def accel_radial(r, magnitude):
    r = np.asarray(r)
    norm = np.linalg.norm(r)
    if norm == 0.0:
        return np.zeros(3)
    return magnitude * r / norm


def accel_inclination(r, v, magnitude):
    r = np.asarray(r, float)
    norm_r = np.linalg.norm(r)
    if norm_r == 0.0 or magnitude == 0.0:
        return np.zeros(3)

    r_hat = r / norm_r
    z_hat = np.array([0.0, 0.0, 1.0])
    north_dir = z_hat - np.dot(z_hat, r_hat) * r_hat
    norm_nd = np.linalg.norm(north_dir)
    if norm_nd == 0.0:
        return np.zeros(3)
    return magnitude * north_dir / norm_nd


def build_profile(profile, t_arr):
    n = len(t_arr)
    out = np.zeros(n, float)

    if profile is None:
        return out

    if np.isscalar(profile):
        out[:] = float(profile)
        return out

    if isinstance(profile, (list, tuple, np.ndarray)) and len(profile) == n:
        return np.asarray(profile, float)

    if isinstance(profile, tuple) and (len(profile) == 2 or len(profile) == 3):
        segments = [profile]
    elif isinstance(profile, (list, tuple)):
        segments = profile
    else:
        raise TypeError("Unsupported profile format")

    for seg in segments:
        if len(seg) == 2:
            start, thrust = seg
            end = None
        elif len(seg) == 3:
            start, end, thrust = seg
        else:
            raise ValueError("Segment must be (start, thrust) or (start, end, thrust)")

        start_idx = (
            int(start)
            if isinstance(start, (int, np.integer))
            else int(np.searchsorted(t_arr, start))
        )
        end_idx = (
            n
            if end is None
            else (
                int(end)
                if isinstance(end, (int, np.integer))
                else int(np.searchsorted(t_arr, end))
            )
        )
        if start_idx >= n:
            continue
        out[start_idx:end_idx] += float(thrust)

    return out