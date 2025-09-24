import numpy as np
from scipy.integrate import solve_ivp

from ..constants import EARTH_MU
from ..Time_Functions import Time


def transfer_coplanar_continuous(r1,
                                 v1,
                                 r2,
                                 v2=None,
                                 a_thrust=1,
                                 mu=EARTH_MU,
                                 t0=None,
                                 max_time=2 * 3600.0,
                                 tol=1.0,
                                 plot=False):
    """
    Continuous‐thrust, coplanar transfer: thrust always lies in the initial
    orbital plane (normal = r1×v1), steering to rendezvous r2 (and optionally v2).

    All units SI: r (m), v (m/s), a_thrust (m/s²), t (s), mu (m³/s²).
    """
    # Epoch
    if t0 is None:
        t0 = Time("2025-01-01", format="iso")

    # Default circular target if v2 omitted
    r2 = np.asarray(r2)
    if v2 is None:
        v2 = np.array([0.0, np.sqrt(mu/np.linalg.norm(r2)), 0.0])

    # Build orthonormal basis in initial plane
    h_vec = np.cross(r1, v1)
    h_hat = h_vec/np.linalg.norm(h_vec)
    r_hat1 = r1/np.linalg.norm(r1)
    p_hat1 = np.cross(h_hat, r_hat1)

    # y = [r(3), v(3), dv1(3), theta]
    # theta controls thrust direction: d_hat = cosθ·r_hat1 + sinθ·p_hat1
    def equations(t, y):
        r = y[0:3]
        v = y[3:6]
        dv1 = y[6:9]
        theta = y[9]

        # gravity
        r_norm = np.linalg.norm(r)
        a_grav = -mu * r / r_norm**3

        # thrust direction in plane
        d_hat = np.cos(theta)*r_hat1 + np.sin(theta)*p_hat1
        a_th = a_thrust * d_hat

        # steering law: point thrust at error vector in plane
        err = r - r2
        er = np.dot(err, r_hat1)
        ep = np.dot(err, p_hat1)
        # simple P-control on angle
        theta_dot = -(ep * np.cos(theta) - er * np.sin(theta)) / (np.linalg.norm(err)+1e-6)

        # accumulate dv1
        dv1_dot = a_th

        return np.hstack((v, a_grav + a_th, dv1_dot, theta_dot))

    # event: position error hits tol (and optionally velocity match)
    def rendezvous_event(t, y):
        r = y[0:3]
        v = y[3:6]
        err = np.linalg.norm(r - r2) - tol
        if np.linalg.norm(v - v2) > 1e-2:
            err = max(err, np.linalg.norm(v - v2) - 1e-2)
        return err

    rendezvous_event.terminal = True
    rendezvous_event.direction = -1

    # initial state: no accumulated dv, initial thrust angle = 0
    y0 = np.hstack((r1, v1, np.zeros(3), 0.0))
    sol = solve_ivp(
        fun=equations,
        t_span=(0.0, max_time),
        y0=y0,
        events=rendezvous_event,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
    )

    if sol.status != 1 or len(sol.t_events[0]) == 0:
        raise ValueError("Failed to rendezvous within max_time")

    t_final = sol.t_events[0][0]
    yf = sol.sol(t_final)
    r_final = yf[0:3]
    v_final = yf[3:6]
    dv1_vec = yf[6:9]
    total_dv1 = np.linalg.norm(dv1_vec)

    result = {
        "r_final": r_final,
        "v_final": v_final,
        "t_final": t_final,
        "delta_v1_vec": dv1_vec,
        "delta_v1": total_dv1,
    }

    if plot:
        times = np.linspace(0.0, t_final, 300)
        arc = sol.sol(times)[0:3].T
        from ..Plots import transfer_plot
        fig = transfer_plot(r1, v1, arc, None, r2, v2,
                            title=(f"t={t_final/60:.1f} min "
                                   f"|Δv₁|={total_dv1:.3f} m/s"),
                            show=False)
        result["fig"] = fig

    return result
