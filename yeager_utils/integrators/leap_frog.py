import numpy as np
from scipy.interpolate import interp1d
from ..constants import EARTH_MU, MOON_MU, SUN_MU
from ..time import to_gps
from .int_utils import (
    accel_point_earth, 
    build_profile, 
    precompute_third_body_positions,
    accel_moon, 
    accel_sun, 
    accel_radial, 
    accel_velocity, 
    accel_inclination
)


def leapfrog(
    r0,
    v0,
    t,
    radial=None,
    velocity=None,
    inclination=None,
    accel_gravity=accel_point_earth,
):
    t_arr = to_gps(t)
    t_arr = t_arr - t_arr[0]
    n_steps = len(t_arr)

    dt_vals = np.diff(t_arr)
    if not np.allclose(dt_vals, dt_vals[0]):
        raise ValueError("Non-uniform Δt not supported")
    dt = dt_vals[0]

    r_th = build_profile(radial,       t_arr)
    v_th = build_profile(velocity,     t_arr)
    i_th = build_profile(inclination,  t_arr)

    r = np.empty((n_steps, 3))
    v = np.empty((n_steps, 3))
    r[0] = np.asarray(r0, float)
    v[0] = np.asarray(v0, float)

    # Precompute interpolators for moon and sun
    interp_moon = precompute_third_body_positions(t, "moon")
    interp_sun  = precompute_third_body_positions(t, "sun")

    # Precompute third-body accelerations at r[0] positions (initial guess)
    a_moon_all = accel_moon(r, t, interp_moon)
    a_sun_all  = accel_sun(r, t, interp_sun)

    for i in range(n_steps - 1):
        a0 = (
            accel_gravity(r[i])
            + a_moon_all[i]
            + a_sun_all[i]
            + accel_radial(r[i],            r_th[i])
            + accel_velocity(v[i],          v_th[i])
            + accel_inclination(r[i], v[i], i_th[i])
        )

        v_half = v[i] + 0.5 * dt * a0
        r[i + 1] = r[i] + dt * v_half

        a1 = (
            accel_gravity(r[i + 1])
            + a_moon_all[i + 1]
            + a_sun_all[i + 1]
            + accel_radial(r[i + 1],            r_th[i + 1])
            + accel_velocity(v_half,            v_th[i + 1])
            + accel_inclination(r[i + 1], v_half, i_th[i + 1])
        )

        v[i + 1] = v_half + 0.5 * dt * a1

    return r, v
