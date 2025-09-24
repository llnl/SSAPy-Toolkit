#!/usr/bin/env python3
"""
ellipse_fit.py  ────────────────────────────────────────────────────────────────
Generate sample points and analytic velocities along a Keplerian ellipse in ℝ³
whose first focus is at the origin.

Returns
-------
dict
    {
      # ───────── trajectory samples ─────────
      "r"            : (N,3) ndarray   Cartesian positions                       [m],
      "v"            : (N,3) ndarray   Inertial velocities                       [m s⁻¹],
      "t_rel"        : (N,)  ndarray   Relative flight-time (starts at 0)        [s],
      "t_abs"        : (N,)  ndarray   Absolute epochs (optional)                [s or datetime64],

      # ───────── classical orbital elements ─────────
      "a"            : float           Semi-major axis                           [m],
      "e"            : float           Eccentricity                              [-],
      "i"            : float           Inclination                               [rad],
      "raan"         : float           Right ascension of ascending node         [rad],
      "pa"           : float           Argument of periapsis                     [rad],
      "ta"           : float           True anomaly at r₀                        [rad],
      "L"            : float           Mean longitude                            [rad],

      # ───────── handy orbit scalars ─────────
      "rp"           : float           Periapsis radius                          [m],
      "ra"           : float           Apoapsis radius                           [m],
      "rp_alt"       : float           Periapsis altitude above Earth            [m],
      "ra_alt"       : float           Apoapsis altitude above Earth             [m],
      "b"            : float           Semi-minor axis                           [m],
      "p"            : float           Semi-latus rectum                         [m],
      "mean_motion"  : float           n = √(μ/a³)                               [rad s⁻¹],
      "eta"          : float           η = √(1 - e²) (circularity factor)        [-],
      "period"       : float           Keplerian orbital period                  [s],

      # ───────── vectors / invariants ─────────
      "h_vec"        : (3,) ndarray    Specific angular-momentum vector          [m² s⁻¹],
      "h"            : float           |h_vec|                                   [m² s⁻¹],
      "Energy"       : float           Specific mechanical energy                [J kg⁻¹],
      "e_vec"        : (3,) ndarray    Eccentricity vector                       [-],
      "n_vec"        : (3,) ndarray    Ascending-node vector                     [-],

      # ───────── convenience state values ─────────
      "r0"           : (3,) ndarray    Departure position (first sample)         [m],
      "v0"           : (3,) ndarray    Departure velocity                        [m s⁻¹],
      "F2"           : (3,) ndarray    Second focus position                     [m],

      # ───────── plane / rotation info ─────────
      "plane_basis"  : (u,v,w) tuple   Orthonormal frame: u & v in-plane, w ⟂,
      "rot_dir"      : int            +1 = CCW, -1 = CW (viewed along +w),

      # ───────── constant ─────────
      "mu"           : float           Gravitational parameter (GM)              [m³ s⁻²],
    }
"""

from ..Plots import save_plot
import numpy as np


def ellipse_fit(P1, P2, *,
                a=None, e=None, F2=None, inc: float = 0.0,
                n_pts: int = 1000, tol=1e-10,
                ccw: bool = True,
                plot=False, save_path=False,
                time_of_departure=None,
                time_of_arrival=None):

    from ..constants import EARTH_MU, EARTH_RADIUS  # [m³ s⁻²]

    # ───────────────────────── internal helpers ──────────────────────────
    def _plane_basis(p1, p2, *, incl: float = 0.0, eps: float = 1e-12):
        p1 = np.asarray(p1, float)
        p2 = np.asarray(p2, float)
        if np.linalg.norm(p1) < eps or np.linalg.norm(p2) < eps:
            raise ValueError("P1 and P2 must be non‑zero")

        u = p1 / np.linalg.norm(p1)
        v = p2 - np.dot(p2, u) * u
        w = np.cross(u, v)
        if np.linalg.norm(v) < eps:
            cand = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(u, cand)) > 1.0 - eps:
                cand = np.array([1.0, 0.0, 0.0])
            v0 = np.cross(u, cand)
            v0 /= np.linalg.norm(v0)
            w = np.cross(u, v0)
            theta = np.deg2rad(incl)
            v = np.cos(theta) * v0 + np.sin(theta) * w
        else:
            w /= np.linalg.norm(w)
            v = np.cross(w, u)
            v /= np.linalg.norm(v)
        return u, v, w

    def _in_plane(vec3, u, v):
        return np.array([np.dot(vec3, u), np.dot(vec3, v)])

    def _to_3d(xy, u, v):
        return xy[0] * u + xy[1] * v

    def _eccentricity(f2, a):
        return np.linalg.norm(f2) / (2.0 * a)

    def _a_for(f2, p):
        return 0.5 * (np.linalg.norm(p) + np.linalg.norm(p - f2))

    def _solve_focus(P1, P2, u, v, *, a=None, e=None, F2=None, tol=1e-10):
        from scipy.optimize import root, minimize, NonlinearConstraint, BFGS

        p1_2d, p2_2d = _in_plane(P1, u, v), _in_plane(P2, u, v)

        if F2 is not None:
            if (a is not None) or (e is not None):
                raise ValueError("Specify only one of a, e or F2")
            F2 = np.asarray(F2, float)
            a = _a_for(F2, P1)
            e = _eccentricity(F2, a)
            return F2, a, e

        if (a is None) and (e is None):

            def obj_min_e(xy):
                F = _to_3d(xy, u, v)
                return _eccentricity(F, _a_for(F, P1))

            def equal_sum(xy):
                F = _to_3d(xy, u, v)
                return (
                    np.linalg.norm(P1)
                    + np.linalg.norm(P1 - F)
                    - np.linalg.norm(P2)
                    - np.linalg.norm(P2 - F)
                )

            def try_minimize_with(method):
                return minimize(
                    obj_min_e,
                    0.5 * (p1_2d + p2_2d),
                    constraints=[
                        NonlinearConstraint(
                            equal_sum, 0.0, 0.0, jac="2-point", hess=BFGS()
                        )
                    ],
                    method=method,
                    tol=tol,
                )

            sol = try_minimize_with("trust-constr")
            if not sol.success:
                sol = try_minimize_with("SLSQP")
            if not sol.success:
                raise RuntimeError("Could not locate least‑eccentric solution")

            F2 = _to_3d(sol.x, u, v)
            a = _a_for(F2, P1)
            e = _eccentricity(F2, a)
            return F2, a, e

        if (a is None) == (e is None):
            raise ValueError("Provide exactly one of {a, e, F2}")

        def residual(xy):
            F = _to_3d(xy, u, v)
            s1 = np.linalg.norm(P1) + np.linalg.norm(P1 - F)
            s2 = np.linalg.norm(P2) + np.linalg.norm(P2 - F)
            r1 = s1 - s2
            r2 = s1 - (np.linalg.norm(F) / e if e is not None else 2.0 * a)
            return np.array([r1, r2])

        sol = root(residual, 0.3 * p1_2d, tol=tol)
        if not sol.success:
            raise RuntimeError("Could not satisfy the supplied (a, e)")

        F2 = _to_3d(sol.x, u, v)
        if e is None:
            e = _eccentricity(F2, a)
        else:
            a = np.linalg.norm(F2) / (2.0 * e)
        return F2, a, e

    def slice_arc_segments(full, i1, i2):
        n = len(full)

        if i2 >= i1:
            short_indices = np.arange(i1, i2 + 1)
        else:
            short_indices = np.concatenate((np.arange(i1, n), np.arange(0, i2 + 1)))

        full_indices = np.arange(n)
        mask = np.ones(n, dtype=bool)
        mask[short_indices] = False
        long_indices = full_indices[mask]
        long_indices = np.concatenate((long_indices, [i1]))

        short_arc = full[short_indices]
        long_arc = full[long_indices]

        return short_arc, long_arc

    def direction(arc3d, eps=1e-12):
        p0 = arc3d[0]
        p10 = arc3d[10]
        rotation_vector = np.cross(p0, p10)
        z = rotation_vector[2]
        if abs(z) > eps:
            return int(np.sign(z))
        x0, y0 = p0[0], p0[1]
        x1, y1 = p10[0], p10[1]
        det = x0 * y1 - y0 * x1
        if abs(det) < eps:
            return 0
        return int(np.sign(det))

    def angle_in_plane(p, u, v):
        x = np.dot(p, u)
        y = np.dot(p, v)
        return np.arctan2(y, x)

    if (time_of_departure is not None) and (time_of_arrival is not None):
        raise ValueError("Specify either time_of_departure or time_of_arrival, not both.")

    P1 = np.asarray(P1, float)
    P2 = np.asarray(P2, float)

    u, v, w = _plane_basis(P1, P2, incl=inc)

    F2, a, e = _solve_focus(P1, P2, u, v, a=a, e=e, F2=F2, tol=tol)

    f2_2d = _in_plane(F2, u, v)
    C = 0.5 * f2_2d
    c = 0.5 * np.linalg.norm(F2)
    b = np.sqrt(max(a * a - c * c, 0.0))

    phi = -np.arctan2(C[1], C[0])
    R2D = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

    t_full = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    xy_full = np.vstack((a * np.cos(t_full), b * np.sin(t_full))).T
    xy_full = (R2D.T @ xy_full.T).T + C
    full3d = np.array([_to_3d(p, u, v) for p in xy_full])

    i1 = np.argmin(np.linalg.norm(full3d - P1, axis=1))
    i2 = np.argmin(np.linalg.norm(full3d - P2, axis=1))

    arc_short, arc_long = slice_arc_segments(full3d, i1, i2)
    arc_long = arc_long[::-1]

    vel_mod = 1
    if ccw:
        if direction(arc_short) > 0:
            arc3d = arc_short
            arc3d_comp = arc_long
        else:
            arc3d = arc_long
            arc3d_comp = arc_short
            vel_mod = -1
    else:
        arc3d = arc_short
        arc3d_comp = arc_long

    rot_dir = direction(arc3d)

    xy_arc = np.array([_in_plane(p, u, v) for p in arc3d])
    f = np.arctan2(xy_arc[:, 1], xy_arc[:, 0])
    f = np.unwrap(f)
    r_norm = np.linalg.norm(arc3d, axis=1)
    h_scalar = np.sqrt(EARTH_MU * a * (1 - e ** 2))
    df = np.diff(f)
    dt_df_mid = 0.5 * (r_norm[:-1] ** 2 + r_norm[1:] ** 2) / h_scalar
    dt = dt_df_mid * df
    t_rel = np.zeros_like(f)
    t_rel[1:] = np.cumsum(dt)
    t_rel -= t_rel[0]
    if np.sum(np.diff(f)) < 0:
        t_rel = np.abs(t_rel)

    if np.linalg.norm(F2) < 1e-12:
        e_hat = arc3d[0] / np.linalg.norm(arc3d[0])
    else:
        e_hat = -F2 / np.linalg.norm(F2)

    vel3d = np.empty_like(arc3d)

    for i, r_vec in enumerate(arc3d):
        r = np.linalg.norm(r_vec)
        r_hat = r_vec / r
        cos_f = np.dot(e_hat, r_hat)
        sin_f = np.dot(w, np.cross(e_hat, r_hat))
        v_r = (EARTH_MU / h_scalar) * e * sin_f
        v_t = (EARTH_MU / h_scalar) * (1 + e * cos_f)

        xy = _in_plane(r_vec, u, v)
        norm_xy = np.linalg.norm(xy)
        if norm_xy < 1e-12:
            t_hat = np.zeros(3)
        else:
            r_hat_2d = xy / norm_xy
            t_hat_2d = np.array([-r_hat_2d[1], r_hat_2d[0]])
            t_hat = _to_3d(t_hat_2d, u, v)

        vel3d[i] = (v_r * r_hat + v_t * t_hat) * vel_mod

    r0, v0 = arc3d[0], vel3d[0]

    t_abs = None
    if time_of_departure is not None:
        t_abs = np.asarray(time_of_departure) + t_rel
    elif time_of_arrival is not None:
        t_abs = np.asarray(time_of_arrival) - (t_rel[-1] - t_rel)

    mu = EARTH_MU
    r = np.linalg.norm(r0)
    h_vec = np.cross(r0, v0)
    h = np.linalg.norm(h_vec)
    n_vec = np.cross(np.array([0.0, 0.0, 1.0]), h_vec)
    n = np.linalg.norm(n_vec)
    e_vec = (np.cross(v0, h_vec) / mu) - r0 / r
    e = np.linalg.norm(e_vec)

    i_rad = np.arccos(h_vec[2] / h)
    if n != 0:
        raan = np.arccos(n_vec[0] / n)
        if n_vec[1] < 0:
            raan = 2 * np.pi - raan
    else:
        raan = 0.0

    if n != 0:
        pa = np.arccos(np.dot(n_vec, e_vec) / (n * e)) if e != 0 else 0.0
        if e_vec[2] < 0:
            pa = 2 * np.pi - pa
    else:
        pa = 0.0

    if e != 0:
        ta = np.arccos(np.dot(e_vec, r0) / (e * r))
        if np.dot(r0, v0) < 0:
            ta = 2 * np.pi - ta
    else:
        ta = 0.0

    cosE = (e + np.cos(ta)) / (1 + e * np.cos(ta))
    sinE = (np.sin(ta) * np.sqrt(1 - e ** 2)) / (1 + e * np.cos(ta))
    E = np.arctan2(sinE, cosE)
    if E < 0:
        E += 2 * np.pi
    M = E - e * np.sin(E)
    L = (raan + pa + M) % (2 * np.pi)

    period = 2 * np.pi * np.sqrt(a ** 3 / mu)

    # ─────────────────────────── plotting  ──────────────────────────────
    if plot or save_path:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.gridspec import GridSpec
        from ssapy.simple import ssapy_orbit

        # arc‑sample timeline (minutes)
        t_minutes = t_rel / 60.0

        # run the ssapy propagator on a *1‑second grid* …
        # r_ss, v_ss, t_ss = ssapy_orbit(
        #     r=r0,
        #     v=v0,
        #     duration=(t_rel[-1], "s"),
        #     freq=(1, "s"),             # ← one sample per second
        # )
        # t_ss_seconds = np.arange(len(t_ss))        # 0,1,2,…
        # t_ss_minutes = t_ss_seconds / 60.0         # for x‑axis
        # print(ssapy_orbit(
        #     r=r0,
        #     v=v0,
        #     t=t_rel,
        # ))
        r_ss, v_ss = ssapy_orbit(
            r=r0,
            v=v0,
            t=t_rel,
        )
        # set up figure / axes
        fig = plt.figure(figsize=(18, 6))
        gs = GridSpec(1, 3, figure=fig, width_ratios=[3, 2, 2])
        ax3d   = fig.add_subplot(gs[0], projection="3d")
        ax_dist  = fig.add_subplot(gs[1])
        ax_speed = fig.add_subplot(gs[2])

        # 3‑D orbit view --------------------------------------------------
        colors = cm.get_cmap("RdYlGn_r")(np.linspace(0, 1, len(arc3d)))
        ax3d.scatter(arc3d_comp[:, 0] / 1e3, arc3d_comp[:, 1] / 1e3,
                     arc3d_comp[:, 2] / 1e3, c="black", s=10)
        ax3d.scatter(arc3d[:, 0] / 1e3, arc3d[:, 1] / 1e3,
                     arc3d[:, 2] / 1e3, c=colors, s=10)
        ax3d.plot(r_ss[:, 0] / 1e3, r_ss[:, 1] / 1e3,
                  r_ss[:, 2] / 1e3, lw=2.5, color="gray", alpha=0.6,
                  label="ssapy")
        ax3d.scatter(0, 0, 0, marker="x", s=80, color="black", label="Earth")
        ax3d.scatter(*F2 / 1e3, marker="x", s=80, color="gray", alpha=0.5,
                     label="focus")
        ax3d.scatter(*P1 / 1e3, s=40, label="P1", color="green")
        ax3d.scatter(*P2 / 1e3, s=40, label="P2", color="red")
        ax3d.set_xlabel("X [km]")
        ax3d.set_ylabel("Y [km]")
        ax3d.set_zlabel("Z [km]")
        Rmax = np.linalg.norm(arc3d, axis=1).max() / 1e3
        ax3d.set_xlim((-Rmax, Rmax)); ax3d.set_ylim((-Rmax, Rmax))
        ax3d.set_zlim((-Rmax, Rmax))
        ax3d.set_title("Full ellipse with ssapy overlay")
        ax3d.legend()

        # distance‑from‑origin plot --------------------------------------
        ax_dist.plot(t_minutes,
                     np.linalg.norm(arc3d, axis=1) / 1e3,
                     lw=5, label="ellipse")
        ax_dist.plot(t_minutes,
                     np.linalg.norm(r_ss, axis=1) / 1e3,
                     lw=5, ls="--", color="red", label="ssapy")
        ax_dist.set_xlabel("Time from P1 [min]")
        ax_dist.set_ylabel("Distance [km]")
        ax_dist.set_title("Distance from origin")
        ax_dist.grid(True); ax_dist.legend()

        # speed plot ------------------------------------------------------
        ax_speed.plot(t_minutes,
                      np.linalg.norm(vel3d, axis=1) / 1e3,
                      lw=5, label="ellipse")
        ax_speed.plot(t_minutes,
                      np.linalg.norm(v_ss, axis=1) / 1e3,
                      lw=5, ls="--", color="red", label="ssapy")
        ax_speed.set_xlabel("Time from P1 [min]")
        ax_speed.set_ylabel("Speed [km/s]")
        ax_speed.set_title("Speed over time")
        ax_speed.grid(True); ax_speed.legend()

        plt.tight_layout()
        plt.show()
        if save_path: 
            save_plot(fig, save_path)

    result = {
        "r"           : arc3d,                          # (N, 3) Cartesian positions           [m]
        "v"           : vel3d,                          # (N, 3) Inertial velocities           [m s⁻¹]
        "t_rel"       : t_rel,                          # (N,)   Relative flight‑time (0‑based) [s]
        "t_abs"       : t_abs,                          # (N,)   Absolute epochs (optional)     [s or datetime64]
        "a"           : a,                              # Semi‑major axis                      [m]
        "e"           : e,                              # Eccentricity                        [–]
        "i"           : i_rad,                          # Inclination                         [rad]
        "raan"        : raan,                           # Right ascension of ascending node   [rad]
        "pa"          : pa,                             # Argument of periapsis               [rad]
        "ta"          : ta,                             # True anomaly at r₀                  [rad]
        "L"           : L,                              # Mean longitude                      [rad]

        "rp"          : a * (1 - e),                    # Periapsis radius                    [m]
        "ra"          : a * (1 + e),                    # Apoapsis  radius                    [m]
        "rp_alt"      : a * (1 - e) - EARTH_RADIUS,     # Periapsis altitude above Earth      [m]
        "ra_alt"      : a * (1 + e) - EARTH_RADIUS,     # Apoapsis  altitude above Earth      [m]

        "b"           : b,                              # Semi‑minor axis                     [m]
        "p"           : a * (1 - e**2),                 # Semi‑latus rectum                   [m]
        "mean_motion" : np.sqrt(mu / a**3),             # Mean motion                         [rad s⁻¹]
        "eta"         : np.sqrt(1 - e**2),              # Circularity factor η ≡ √(1−e²)      [–]

        "period"      : period,                         # Keplerian period                    [s]
        "h_vec"       : h_vec,                          # Specific angular‑momentum vector    [m² s⁻¹]
        "h"           : h,                              # |h_vec|                             [m² s⁻¹]
        "Energy"      : -EARTH_MU / (2 * a),            # Specific mechanical energy          [J kg⁻¹]

        "e_vec"       : e_vec,                          # Eccentricity vector                 [–]
        "n_vec"       : n_vec,                          # Ascending‑node vector               [–]

        "r0"          : r0,                             # Departure position                  [m]
        "v0"          : v0,                             # Departure velocity                  [m s⁻¹]
        "F2"          : F2,                             # Second focus position               [m]

        "plane_basis" : (u, v, w),                      # Orthonormal frame (in‑plane u,v; w⊥)
        "rot_dir"     : rot_dir,                        # +1 = CCW, −1 = CW (viewed along +w)
        "mu"          : EARTH_MU,                       # Gravitational parameter             [m³ s⁻²]
    }

    return result
