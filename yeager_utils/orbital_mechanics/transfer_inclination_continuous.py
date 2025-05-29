import numpy as np
from scipy.integrate import solve_ivp
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ..plots import set_axes_equal, save_plot
from ..constants import EARTH_MU, EARTH_RADIUS


def transfer_inclination_continuous(r0,
                         v0,
                         i_target=None,
                         delta_v=None,
                         a_thrust=1,
                         mu=EARTH_MU,
                         t0=0.0,
                         max_time=1e6,
                         body_radius=EARTH_RADIUS,
                         plot=False,
                         save_path=False):
    """
    Change orbital inclination via continuous normal thrust.

    Either i_target (rad) or delta_v (m/s) must be specified, but not both.
    """

    if (i_target is None) == (delta_v is None):
        raise ValueError("Specify exactly one of i_target or delta_v.")

    def equations(t, y):
        r, v = y[:3], y[3:]
        r_norm = np.linalg.norm(r)
        a_grav = -mu * r / r_norm**3
        h = np.cross(r, v)
        if np.linalg.norm(h) > 0:
            n_vec = h / np.linalg.norm(h)
            if delta_v is not None and delta_v < 0:
                n_vec = -n_vec
        else:
            n_vec = np.zeros(3)
        a = a_grav + a_thrust * n_vec
        return np.concatenate((v, a))

    def inclination_event(t, y):
        r, v = y[:3], y[3:]
        h = np.cross(r, v)
        h_norm = np.linalg.norm(h)
        if h_norm == 0:
            return np.inf  # Avoid divide by zero
        inclination = np.arccos(np.clip(h[2] / h_norm, -1.0, 1.0))
        if h[1] < 0:  # Use h[1] or node vector to define hemisphere
            inclination = -inclination
        return inclination - i_target
    inclination_event.terminal = True
    inclination_event.direction = 1

    def deltav_event(t, y):
        return a_thrust * (t - t0) - abs(delta_v)
    deltav_event.terminal = True
    deltav_event.direction = 1

    y0 = np.concatenate((r0, v0))
    event_fn = inclination_event if i_target is not None else deltav_event

    sol = solve_ivp(
        equations,
        (t0, t0 + max_time),
        y0,
        events=event_fn,
        method='RK45',
        rtol=1e-8,
        atol=1e-10,
        dense_output=True
    )

    if sol.status != 1 or sol.t_events[0].size == 0:
        raise ValueError("Condition not reached within max_time.")

    t_final = sol.t_events[0][0]
    y_final = sol.sol(t_final)
    r_final, v_final = y_final[:3], y_final[3:]

    if plot:
        _plot_transfer(sol, r0, v0, r_final, v_final, t0, t_final, mu, body_radius, save_path)

    return r_final, v_final, t_final


def _plot_transfer(sol, r0, v0, r_final, v_final, t0, t_final, mu, body_radius, save_path):
    def kepler_eq(t, y):
        r = y[:3]
        v = y[3:]
        r_norm = np.linalg.norm(r)
        a = -mu * r / r_norm**3
        return np.concatenate((v, a))

    r0_mag = np.linalg.norm(r0)
    v0_mag = np.linalg.norm(v0)
    a0 = 1 / (2 / r0_mag - v0_mag**2 / mu)
    T0 = 2 * np.pi * np.sqrt(abs(a0)**3 / mu)

    rf_mag = np.linalg.norm(r_final)
    vf_mag = np.linalg.norm(v_final)
    af = 1 / (2 / rf_mag - vf_mag**2 / mu)
    Tf = 2 * np.pi * np.sqrt(abs(af)**3 / mu)

    sol_initial = solve_ivp(
        kepler_eq,
        (t0, t0 + T0),
        np.concatenate((r0, v0)),
        t_eval=np.linspace(t0, t0 + T0, 200),
        rtol=1e-8,
        atol=1e-10,
    )

    sol_final = solve_ivp(
        kepler_eq,
        (t_final, t_final + Tf),
        np.concatenate((r_final, v_final)),
        t_eval=np.linspace(t_final, t_final + Tf, 200),
        rtol=1e-8,
        atol=1e-10,
    )

    t_burn = np.linspace(t0, t_final, 300)
    r_burn = sol.sol(t_burn)[:3].T

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = body_radius * np.outer(np.cos(u), np.sin(v))
    y = body_radius * np.outer(np.sin(u), np.sin(v))
    z = body_radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='blue', alpha=0.3)

    ax.plot(*sol_initial.y[:3], 'g--', label='Initial Orbit')
    ax.plot(*sol_final.y[:3], 'b--', label='Final Orbit')
    ax.plot(r_burn[:, 0], r_burn[:, 1], r_burn[:, 2], 'r-', label='Burn Trajectory')
    ax.scatter(*r0, color='green', s=50, label='Start')
    ax.scatter(*r_final, color='blue', s=50, label='End')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Inclination Transfer with Normal Thrust')
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    plt.show()

    if save_path:
        save_plot(fig, save_path)