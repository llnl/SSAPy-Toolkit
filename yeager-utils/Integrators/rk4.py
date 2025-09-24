import numpy as np
from ..constants import EARTH_MU
from ..Time_Functions import to_gps
from scipy.interpolate import interp1d
from .int_utils import (
    build_profile, 
    precompute_third_body_positions,
    accel_moon, 
    accel_sun, 
    accel_radial, 
    accel_velocity, 
    accel_inclination
)


def rk4(
    r0,
    v0,
    t,
    radial=None,
    velocity=None,
    inclination=None,
    accel_gravity=lambda r: -EARTH_MU * r / np.linalg.norm(r)**3,
):
    t_arr = to_gps(t)
    t_arr = t_arr - t_arr[0]
    n_steps = len(t_arr)

    r_th = build_profile(radial,       t_arr)
    v_th = build_profile(velocity,     t_arr)
    i_th = build_profile(inclination,  t_arr)

    r = np.empty((n_steps, 3))
    v = np.empty((n_steps, 3))
    r[0] = np.asarray(r0, float)
    v[0] = np.asarray(v0, float)

    # Precompute third-body interpolators
    interp_moon = precompute_third_body_positions(t, "moon")
    interp_sun  = precompute_third_body_positions(t, "sun")

    for i in range(n_steps - 1):
        dt = t_arr[i + 1] - t_arr[i]

        def a_total(r_i, v_i, t_i, i_thrust):
            a = accel_gravity(r_i)
            a += accel_moon(np.array([r_i]), np.array([t_i]), interp_moon)[0]
            a += accel_sun(np.array([r_i]), np.array([t_i]), interp_sun)[0]
            a += accel_radial(r_i, r_th[i_thrust])
            a += accel_velocity(v_i, v_th[i_thrust])
            a += accel_inclination(r_i, v_i, i_th[i_thrust])
            return a

        # RK4 steps
        k1_v = a_total(r[i], v[i], t_arr[i], i)
        k1_r = v[i]

        k2_v = a_total(r[i] + 0.5*dt*k1_r, v[i] + 0.5*dt*k1_v, t_arr[i] + 0.5*dt, i)
        k2_r = v[i] + 0.5*dt*k1_v

        k3_v = a_total(r[i] + 0.5*dt*k2_r, v[i] + 0.5*dt*k2_v, t_arr[i] + 0.5*dt, i)
        k3_r = v[i] + 0.5*dt*k2_v

        k4_v = a_total(r[i] + dt*k3_r, v[i] + dt*k3_v, t_arr[i] + dt, i)
        k4_r = v[i] + dt*k3_v

        r[i + 1] = r[i] + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
        v[i + 1] = v[i] + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    return r, v
