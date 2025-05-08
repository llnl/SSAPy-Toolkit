import numpy as np
from scipy.integrate import solve_ivp
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ..plots.set_axes_equal import set_axes_equal
from ..constants import EARTH_MU, EARTH_RADIUS


def transfer_inclination_continuous(r0,
                         v0,
                         i_target,
                         a_thrust,
                         mu=EARTH_MU,
                         t0=0.0,
                         max_time=1e6,
                         body_radius=EARTH_RADIUS,
                         plot=False):
    """
    Change orbital inclination to i_target by continuous normal thrust.

    Parameters:
    - r0: Initial position vector [m]
    - v0: Initial velocity vector [m/s]
    - i_target: Target inclination [rad]
    - a_thrust: Thrust acceleration magnitude [m/s^2]
    - mu: Gravitational parameter [m^3/s^2]
    - t0: Start time [s]
    - max_time: Maximum integration time [s]
    - body_radius: Radius of the central body [m]
    - plot: Whether to plot the trajectory

    Returns:
    - r_final: Final position vector [m]
    - v_final: Final velocity vector [m/s]
    - t_final: Time at which inclination was reached [s]
    """

    def equations(t, y):
        r = y[:3]
        v = y[3:]
        r_norm = np.linalg.norm(r)
        a_grav = -mu * r / r_norm**3
        h = np.cross(r, v)
        h_norm = np.linalg.norm(h)
        n_vec = h / h_norm if h_norm > 0.0 else np.zeros(3)
        a = a_grav + a_thrust * n_vec
        return np.hstack((v, a))

    def inclination_event(t, y):
        r = y[:3]
        v = y[3:]
        h = np.cross(r, v)
        h_norm = np.linalg.norm(h)
        if h_norm == 0.0:
            return -i_target  # stay safe if undefined
        inc = np.arccos(np.clip(h[2] / h_norm, -1.0, 1.0))
        sign = np.sign(h[2])  # +z is positive inclination
        signed_inc = inc if sign >= 0 else -inc
        return signed_inc - i_target

    inclination_event.terminal = True
    inclination_event.direction = 1

    y0 = np.hstack((r0, v0))
    sol = solve_ivp(
        fun=equations,
        t_span=(t0, t0 + max_time),
        y0=y0,
        events=inclination_event,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
    )

    if sol.status != 1 or sol.t_events[0].size == 0:
        raise ValueError("Target inclination not reached within max_time")

    t_final = sol.t_events[0][0]
    y_final = sol.sol(t_final)
    r_final, v_final = y_final[:3], y_final[3:]

    energy = 0.5 * np.dot(v_final, v_final) - mu / np.linalg.norm(r_final)
    if energy > 0.0:
        warnings.warn("Final orbit is unbound (specific energy > 0)", RuntimeWarning)

    if plot:
        _plot_transfer(sol, r0, v0, r_final, v_final, t0, t_final, mu, body_radius)

    return r_final, v_final, t_final


def _plot_transfer(sol, r0, v0, r_final, v_final, t0, t_final, mu, body_radius):
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
