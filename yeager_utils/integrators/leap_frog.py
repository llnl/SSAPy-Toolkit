import numpy as np
from ..accelerations import (
    accel_point_earth,
    accel_radial,
    accel_velocity,
    accel_inclination,
    accel_plane,
    accel_to_circular,
)
from ..time import to_gps
from .fuel import estimate_fuel_usage


def ensure_list_of_lists(x):
    if isinstance(x, (list, tuple, np.ndarray)) and all(isinstance(i, (list, tuple, dict, np.ndarray)) for i in x):
        return x  # Already good
    else:
        if not isinstance(x, (list, tuple, np.ndarray)):
            x = [x]
        if not isinstance(x[0], (list, tuple, dict, np.ndarray)):
            x = [x]
        return x


def leapfrog(
    r0,
    v0,
    t,
    accel=accel_point_earth,
    radial=None,
    velocity=None,
    inclination=None,
    plane=None,
    circular=None,
    fuel=False,
):
    """
    Propagate position and velocity using Leapfrog integration with optional accelerations.

    Parameters
    ----------
    r0 : array_like
        Initial position vector [m].
    v0 : array_like
        Initial velocity vector [m/s].
    t : array_like
        Array of times [s] as floats or datetime-like objects.
    accel : function
        Function to compute natural accelerations.
    radial : dict or list, optional
        Thrust profile as {'thrust': value, 'start': start_time/index, 'end': end_time/index}.
    velocity : dict or list, optional
        Thrust profile for along-track acceleration.
    inclination : dict or list, optional
        Thrust profile for changing orbital inclination.
    plane : dict or list, optional
        Thrust profile for changing orbital plane orientation.
    circular : dict or list, optional
        Thrust profile for circularizing orbit.
    fuel : bool, optional
        Whether to estimate fuel usage.

    Returns
    -------
    r : ndarray
        Position array (n_steps, 3) [m].
    v : ndarray
        Velocity array (n_steps, 3) [m/s].
    fuels : dict (optional)
        Estimated fuel usage per maneuver type.
    
    Author: Travis Yeager
    """

    def get_mask(n_steps, start, end=None, time_array=None):
        if isinstance(start, (float, int)) and time_array is not None:
            start = int(np.searchsorted(time_array, start))
        if end is not None and isinstance(end, (float, int)) and time_array is not None:
            end = int(np.searchsorted(time_array, end))
        if isinstance(start, int) and end is None:
            end = n_steps
        mask = np.zeros(n_steps, dtype=bool)
        mask[start:end] = True
        return mask

    def prep_thrust(n_steps, t_arr, profiles, continuous=False):
        """
        Build a thrust-time array from profiles that may be:
        - None
        - dict with keys 'thrust', 'start', optionally 'end'
        - list/tuple/ndarray [thrust, start, end?]  (or [thrust, start] if continuous)
        - list of any of the above

        Overlaps sum naturally.
        """
        total = np.zeros(n_steps, float)
        if profiles is None:
            return total

        # Normalize to list
        profiles = ensure_list_of_lists(profiles)

        for prof in profiles:
            # Case A: dict
            if isinstance(prof, dict):
                thrust = float(prof["thrust"])
                start = int(prof["start"])
                end   = int(prof.get("end", -1))

            # Case B: sequence
            elif isinstance(prof, (list, tuple, np.ndarray)):
                arr = list(prof)
                thrust = float(arr[0])
                start  = arr[1]
                if continuous:
                    end = None
                else:
                    end = arr[2] if len(arr) > 2 else None

            else:
                raise TypeError(f"Unsupported profile type: {type(prof)}")

            # Build mask
            if isinstance(start, (float, int)) and t_arr is not None:
                start_idx = int(np.searchsorted(t_arr, start))
            else:
                start_idx = start
            if end is None:
                end_idx = n_steps
            else:
                if isinstance(end, (float, int)) and t_arr is not None:
                    end_idx = int(np.searchsorted(t_arr, end))
                else:
                    end_idx = end

            mask = np.zeros(n_steps, bool)
            mask[start_idx:end_idx] = True
            if continuous and mask.any():
                mask[mask.argmax():] = True

            total += thrust * mask

        return total

    # convert time to GPS‐seconds
    t_arr = to_gps(t)
    t_arr = t_arr - t_arr[0]
    n_steps = len(t_arr)

    # now each can be a dict or list of dicts
    r_th = prep_thrust(n_steps, t_arr, radial)
    v_th = prep_thrust(n_steps, t_arr, velocity)
    i_th = prep_thrust(n_steps, t_arr, inclination)
    p_th = prep_thrust(n_steps, t_arr, plane)
    c_th = prep_thrust(n_steps, t_arr, circular, continuous=True)

    dt_vals = np.diff(t_arr)
    if not np.allclose(dt_vals, dt_vals[0]):
        raise ValueError("non-uniform dt not supported")
    dt = dt_vals[0]

    r = np.zeros((n_steps, 3))
    v = np.zeros((n_steps, 3))
    r[0], v[0] = np.array(r0), np.array(v0)

    for i in range(n_steps - 1):
        a0 = (
            accel(r[i])
            + accel_radial(r[i], r_th[i])
            + accel_velocity(v[i], v_th[i])
            + accel_inclination(r[i], v[i], i_th[i])
            + accel_plane(r[i], v[i], p_th[i])
            + accel_to_circular(r[i], v[i], c_th[i])
        )
        v_half = v[i] + 0.5 * dt * a0
        r[i + 1] = r[i] + dt * v_half
        
        a1 = (
            accel(r[i + 1])
            + accel_radial(r[i + 1], r_th[i + 1])
            + accel_velocity(v_half, v_th[i + 1])
            + accel_inclination(r[i + 1], v_half, i_th[i + 1])
            + accel_plane(r[i + 1], v_half, p_th[i + 1])
            + accel_to_circular(r[i + 1], v_half, c_th[i + 1])
        )
        v[i + 1] = v_half + 0.5 * dt * a1

    if not fuel:
        return r, v

    fuels = {
        "radial": estimate_fuel_usage(np.abs(r_th), dt, r, engine="Mira"),
        "velocity": estimate_fuel_usage(np.abs(v_th), dt, r, engine="Mira"),
        "inclination": estimate_fuel_usage(np.abs(i_th), dt, r, engine="Mira"),
        "plane": estimate_fuel_usage(np.abs(p_th), dt, r, engine="Mira"),
        "circular": estimate_fuel_usage(np.abs(c_th), dt, r, engine="Mira"),
    }
    return r, v, fuels
