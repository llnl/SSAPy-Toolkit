import numpy as np
from astropy import units as u
from astropy.units import Quantity
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
        Initial position vector (3-element).
    v0 : array_like
        Initial velocity vector (3-element).
    t : array_like or Quantity
        Array of times. Assumed to be in seconds unless a Quantity with time units is passed.
    accel : function
        Function to compute natural accelerations, defaults to accel_point_earth.
    radial : dict or list, optional
        Thrust profile as {'thrust': value, 'start': start_time/index, 'end': end_time/index}.
    velocity : dict or list, optional
        Thrust profile for along-track acceleration.
    inclination : dict or list, optional
        Thrust profile for changing orbital inclination.
    plane : dict or list, optional
        Thrust profile for changing orbital plane orientation.
    circular : dict or list, optional
        Thrust profile for circularizing orbit. If 'end' not given, thrust continues from 'start'.
    fuel : bool, optional
        Whether to estimate fuel usage for each active acceleration component.
    mass0 : float, optional
        Initial spacecraft mass in kg. Used if fuel=True.
    isp : float, optional
        Specific impulse in seconds. Used if fuel=True.

    Returns
    -------
    r : ndarray
        Position array of shape (n_steps, 3).
    v : ndarray
        Velocity array of shape (n_steps, 3).

    Notes
    -----
    Author: Travis Yeager
    """

    def get_mask(n_steps, start, end=None, time_array=None):
        if isinstance(start, Quantity):
            start = int(np.searchsorted(time_array, start.to(u.s).value))
        if end is not None and isinstance(end, Quantity):
            end = int(np.searchsorted(time_array, end.to(u.s).value))
        if end is None:
            end = n_steps
        mask = np.zeros(n_steps, dtype=bool)
        mask[start:end] = True
        return mask

    def prep_thrust(n_steps, t_arr, profile, continuous=False):
        if profile is None:
            return np.zeros(n_steps, float)
        if isinstance(profile, list):
            profile = dict(thrust=profile[0], start=profile[1], end=(profile[2] if len(profile) > 2 else None))

        thrust = float(profile["thrust"])
        start = profile["start"]
        end = profile.get("end", None)
        mask = get_mask(n_steps, start, end, t_arr)
        if continuous and np.any(mask):
            mask[np.argmax(mask):] = True
        return np.full(n_steps, thrust) * mask

    t_arr = to_gps(t)
    t_arr = t_arr.to_value(u.s) if isinstance(t, Quantity) else np.asarray(t, dtype=float)
    n_steps = len(t_arr)

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
        "radial": estimate_fuel_usage(
            np.abs(r_th), dt, r, engine="Mira"
        ),  # Fuel for radial maneuvers
        "velocity": estimate_fuel_usage(
            np.abs(v_th), dt, r, engine="Mira"
        ),  # Fuel for velocity (in-track) maneuvers
        "inclination": estimate_fuel_usage(
            np.abs(i_th), dt, r, engine="Mira"
        ),  # Fuel for inclination changes
        "plane": estimate_fuel_usage(
            np.abs(p_th), dt, r, engine="Mira"
        ),  # Fuel for plane (out-of-plane) maneuvers
        "circular": estimate_fuel_usage(
            np.abs(c_th), dt, r, engine="Mira"
        ),  # Fuel for circularization maneuvers
    }
    return r, v, fuels
