# ssapy_toolkit/propagators_orbit/int_utils.py

import numpy as np
from scipy.interpolate import interp1d

from ..time_functions import to_gps


def precompute_third_body_positions(t, body_name):
    """
    Precompute an interpolated position function for a third body (Moon/Sun/etc.).
    Returns a callable pos(t_query) -> (N,3) position array.
    Use at least two strictly increasing epochs. Interpolation is cubic for
    four or more samples, quadratic for three and linear for two.
    """
    from ssapy import get_body

    t_gps = np.asarray(to_gps(t), dtype=float)
    if (t_gps.ndim != 1 or t_gps.size < 2 or not np.all(np.isfinite(t_gps))
            or np.any(np.diff(t_gps) <= 0)):
        raise ValueError("ephemeris grid must contain at least two finite increasing epochs")
    body = get_body(body_name)
    r_body = np.asarray(body.position(t_gps), dtype=float).T
    if r_body.shape != (len(t_gps), 3) or not np.all(np.isfinite(r_body)):
        raise ValueError("body.position must return finite positions with shape (3, n)")

    interp_funcs = [
        interp1d(t_gps, r_body[:, i], kind=min(3, len(t_gps) - 1), fill_value="extrapolate")
        for i in range(3)
    ]

    def interpolated_position(t_query):
        tq = np.asarray(to_gps(t_query), dtype=float)
        return np.stack([f(tq) for f in interp_funcs], axis=-1)

    return interpolated_position


def build_profile(profile, t_arr):
    """
    Build an (n,) acceleration-magnitude profile aligned to t_arr.

    Supported profiles include ``None`` (zeros), a scalar (constant), an
    array-like of length ``n`` (pass-through), dictionaries or lists of
    dictionary segments with ``start``, ``end``, and ``thrust``/``accel``
    keys, and tuple segments ``(start, thrust)`` or ``(start, end, thrust)``.
    Segment bounds may be indices or times searched in ``t_arr``.
    """
    n = len(t_arr)
    out = np.zeros(n, float)

    if profile is None:
        return out

    if np.isscalar(profile):
        out[:] = float(profile)
        return out

    if isinstance(profile, (list, tuple, np.ndarray)) and len(profile) == n:
        return np.asarray(profile, float)

    # Handle single dictionary
    if isinstance(profile, dict):
        profile = [profile]  # wrap in list for uniform handling [103]

    # Handle list of dicts
    if isinstance(profile, (list, tuple)) and all(isinstance(p, dict) for p in profile):
        for seg in profile:
            start = seg.get("start", 0)
            end = seg.get("end", None)
            thrust = seg.get("thrust", seg.get("accel", 0))

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

    # Handle tuple-based segment(s)
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
