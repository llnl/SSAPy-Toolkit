# yeager_utils/Integrators/int_utils.py

import numpy as np
from scipy.interpolate import interp1d

from ..Time_Functions import to_gps

# Canonical accel functions live in yeager_utils/Accelerations/
from ..Accelerations.accel_point_earth import accel_point_earth  # [64]
from ..Accelerations.accel_velocity import accel_velocity        # [68]
from ..Accelerations.accel_radial import accel_radial            # [65]
from ..Accelerations.accel_inclination import accel_inclination  # [61]

# You said you will move these into Accelerations as individual scripts.
# This file now only re-exports them for backward compatibility with integrators
# that import from .int_utils (e.g., rk4.py) [106].
from ..Accelerations.accel_moon import accel_moon                # (new file you’ll add)
from ..Accelerations.accel_sun import accel_sun                  # (new file you’ll add)


def precompute_third_body_positions(t, body_name):
    """
    Precompute an interpolated position function for a third body (Moon/Sun/etc.).
    Returns a callable pos(t_query) -> (N,3) position array.
    """
    from ssapy import get_body

    body = get_body(body_name)
    r_body = body.position(t).T  # (n,3)
    t_gps = np.asarray(to_gps(t), dtype=float)

    interp_funcs = [
        interp1d(t_gps, r_body[:, i], kind="cubic", fill_value="extrapolate")
        for i in range(3)
    ]

    def interpolated_position(t_query):
        tq = np.asarray(to_gps(t_query), dtype=float)
        return np.stack([f(tq) for f in interp_funcs], axis=-1)

    return interpolated_position


def build_profile(profile, t_arr):
    """
    Build an (n,) acceleration-magnitude profile aligned to t_arr.

    Supports:
    - None -> zeros
    - scalar -> constant
    - array-like length n -> pass-through
    - dict or list of dict segments with keys:
        start, end, thrust (or accel)
      where start/end can be indices or times (searched in t_arr).
    - tuple segments: (start, thrust) or (start, end, thrust)
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