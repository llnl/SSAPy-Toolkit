#!/usr/bin/env python3
"""
ellipse_arc.py  ────────────────────────────────────────────────────────────────
Generate sample points and analytical velocities along an ellipse in ℝ³ whose
first focus is at the origin.

Returns
-------
arc3d : (N,3) ndarray   Cartesian positions  [m]
vel3d : (N,3) ndarray   Inertial velocities  [m s⁻¹]
t_rel : (N,)  ndarray   Relative flight‑time [s]  (t_rel[0] == 0)
info  : dict
    Dictionary of orbital parameters and diagnostics including:
    - Classical orbital elements (a, e, i, RAAN, ω, ν)
    - Focus/geometry info (F2, b, c, center, rotation matrix)
    - Arc endpoints (r0, v0, r1, v1), true anomaly bounds
    - Plane basis (u, v, w)
    - Collision avoidance flag
"""

from ..plots     import save_plot
import numpy as np

# ─────────────────────────── public API ─────────────────────────────
def ellipse_arc(P1, P2, *,
                a=None, e=None, F2=None, inc: float = 0.0,
                n_pts: int = 1000, tol=1e-10,
                ccw: bool = True,
                collision=True, safe_margin=100e3,
                plot=False, save_path=False,
                debug=False):

    from ..constants import EARTH_MU, EARTH_RADIUS             # [m³ s⁻²], meters

    # ───────────────────────── internal helpers ──────────────────────────
    def _plane_basis(p1, p2, *, incl: float = 0.0, eps: float = 1e-12):
        p1 = np.asarray(p1, float);  p2 = np.asarray(p2, float)
        if np.linalg.norm(p1) < eps or np.linalg.norm(p2) < eps:
            raise ValueError("P1 and P2 must be non‑zero")

        u = p1 / np.linalg.norm(p1)
        v = p2 - np.dot(p2, u) * u
        w = np.cross(u, v)
        if np.linalg.norm(v) < eps:
            cand = np.array([0., 0., 1.])
            if abs(np.dot(u, cand)) > 1. - eps:
                cand = np.array([1., 0., 0.])
            v0 = np.cross(u, cand)
            v0 /= np.linalg.norm(v0)
            w  = np.cross(u, v0)
            θ  = np.deg2rad(incl)
            v  = np.cos(θ)*v0 + np.sin(θ)*w
        else:
            w /= np.linalg.norm(w)
            v = np.cross(w, u)
            v /= np.linalg.norm(v)
        return u, v, w

    def _in_plane(vec3, u, v):
        return np.array([np.dot(vec3, u), np.dot(vec3, v)])

    def _to_3d(xy, u, v):
        return xy[0]*u + xy[1]*v

    def _eccentricity(f2, a):
        return np.linalg.norm(f2) / (2.0 * a)

    def _a_for(f2, p):
        return 0.5 * (np.linalg.norm(p) + np.linalg.norm(p - f2))

    def _solve_focus(P1, P2, u, v, *, a=None, e=None, F2=None, tol=1e-10):
        from scipy.optimize import root, minimize, NonlinearConstraint, BFGS
        p1_2d, p2_2d = _in_plane(P1, u, v), _in_plane(P2, u, v)

        if F2 is not None:
            if (a is not None) or (e is not None):
                raise ValueError("Specify only ONE of a, e or F2")
            F2 = np.asarray(F2, float)
            a  = _a_for(F2, P1);   e = _eccentricity(F2, a)
            return F2, a, e

        if (a is None) and (e is None):
            def obj_min_e(xy):
                F = _to_3d(xy, u, v)
                return _eccentricity(F, _a_for(F, P1))

            def equal_sum(xy):
                F = _to_3d(xy, u, v)
                return (np.linalg.norm(P1)+np.linalg.norm(P1-F)
                        - np.linalg.norm(P2)-np.linalg.norm(P2-F))

            def try_minimize_with(method):
                return minimize(
                    obj_min_e, 0.5 * (p1_2d + p2_2d),  # better starting guess
                    constraints=[NonlinearConstraint(equal_sum, 0., 0., jac="2-point", hess=BFGS())],
                    method=method, tol=tol
                )

            # First attempt: trust-constr
            sol = try_minimize_with("trust-constr")

            if not sol.success:
                # Retry with SLSQP
                sol = try_minimize_with("SLSQP")

            if not sol.success:
                raise RuntimeError("Could not locate least‑eccentric solution")

            F2 = _to_3d(sol.x, u, v);  a = _a_for(F2, P1);  e = _eccentricity(F2, a)
            return F2, a, e

        if (a is None) == (e is None):
            raise ValueError("Provide exactly one of {a, e, F2}")

        def residual(xy):
            F = _to_3d(xy, u, v)
            s1 = np.linalg.norm(P1) + np.linalg.norm(P1 - F)
            s2 = np.linalg.norm(P2) + np.linalg.norm(P2 - F)
            r1 = s1 - s2
            r2 = s1 - (np.linalg.norm(F)/e if e is not None else 2.*a)
            return np.array([r1, r2])

        sol = root(residual, 0.3*p1_2d, tol=tol)
        if not sol.success:
            raise RuntimeError("Could not satisfy the supplied (a, e)")

        F2 = _to_3d(sol.x, u, v)
        if e is None:   e = _eccentricity(F2, a)
        else:           a = np.linalg.norm(F2) / (2.*e)
        return F2, a, e

    def slice_arc_segments(full, i1, i2):
        n = len(full)

        # Short arc: forward from i1 to i2 (with wraparound)
        if i2 >= i1:
            short_indices = np.arange(i1, i2 + 1)
        else:
            short_indices = np.concatenate((np.arange(i1, n), np.arange(0, i2 + 1)))

        # Long arc: the complementary portion from i2 to i1
        full_indices = np.arange(n)
        mask = np.ones(n, dtype=bool)
        mask[short_indices] = False
        long_indices = full_indices[mask]
        long_indices = np.concatenate((long_indices, [i1]))  # ensure closure

        short_arc = full[short_indices]
        long_arc = full[long_indices]

        return short_arc, long_arc

    def direction(arc3d, eps=1e-12):
        p0 = arc3d[0]
        p10 = arc3d[10]
        rotation_vector = np.cross(p0, p10)
        z = rotation_vector[2]
        if abs(z) > eps:
            return int(np.sign(z))  # +1 for CCW, -1 for CW
        else:
            x0, y0 = p0[0], p0[1]
            x1, y1 = p10[0], p10[1]
            det = x0 * y1 - y0 * x1
            if abs(det) < eps:
                return 0  # no directional sweep
            return int(np.sign(det))  # +1 for CCW, -1 for CW

    def angle_in_plane(p, u, v):
        x = np.dot(p, u)
        y = np.dot(p, v)
        return np.arctan2(y, x)

    def _rot(p):
        return R2D @ (p - C)

    P1 = np.asarray(P1, float)
    P2 = np.asarray(P2, float)

    u, v, w = _plane_basis(P1, P2, incl=inc)

    max_iter = 10
    scale_factor = 1.05

    attempt = 0
    collision_avoided = False
    while attempt < max_iter:
        F2_try, a_try, e_try = _solve_focus(P1, P2, u, v, a=a, e=e, F2=F2, tol=tol)

        f2_2d = _in_plane(F2_try, u, v)
        C     = 0.5 * f2_2d
        c     = 0.5 * np.linalg.norm(F2_try)
        b     = np.sqrt(max(a_try*a_try - c*c, 0.0))

        phi   = -np.arctan2(C[1], C[0])
        R2D   = np.array([[np.cos(phi), -np.sin(phi)],
                        [np.sin(phi),  np.cos(phi)]])

        t_full = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
        xy_full = np.vstack((a_try * np.cos(t_full), b * np.sin(t_full))).T
        xy_full = (R2D.T @ xy_full.T).T + C
        full3d_try = np.array([_to_3d(p, u, v) for p in xy_full])

        min_dist = np.min(np.linalg.norm(full3d_try, axis=1))

        if (not collision) or (min_dist > EARTH_RADIUS + safe_margin):
            # Safe orbit found
            F2, a, e = F2_try, a_try, e_try
            full3d = full3d_try
            break
        else:
            # Adjust a/e to lift periapsis
            collision_avoided = True
            if a is not None:
                a *= scale_factor
            elif e is not None:
                e *= 0.95
            else:
                a = _a_for(F2_try, P1) * scale_factor
            attempt += 1

    if attempt == max_iter:
        raise RuntimeError("Could not find a collision-free ellipse after several attempts.")


    i1 = np.argmin(np.linalg.norm(full3d - P1, axis=1))
    i2 = np.argmin(np.linalg.norm(full3d - P2, axis=1))
    if debug:
        if np.linalg.norm(full3d[i1] - P1) > 1e-3 or np.linalg.norm(full3d[i2] - P2) > 1e-3:
            print("[WARNING] P1 or P2 is far from nearest sampled point")


    arc_short, arc_long = slice_arc_segments(full3d, i1, i2)
    arc_long = arc_long[::-1]
    if debug:
        print(f"Directions: short:{direction(arc_short)}, long: {direction(arc_long)}")

    vel_mod = 1
    if ccw:
        if direction(arc_short) > 0:
            arc3d      = arc_short
            arc3d_comp = arc_long
        else:
            if debug:
                print("Changing to the long arc for CCW.")
            arc3d      = arc_long
            arc3d_comp = arc_short
            vel_mod = -1
    else:
        arc3d      = arc_short
        arc3d_comp = arc_long

    rot_dir = direction(arc3d)

    xy_arc = np.array([_in_plane(p, u, v) for p in arc3d])
    f = np.arctan2(xy_arc[:, 1], xy_arc[:, 0])
    f = np.unwrap(f)
    r_norm = np.linalg.norm(arc3d, axis=1)
    h  = np.sqrt(EARTH_MU * a * (1 - e**2))
    df = np.diff(f)
    dt_df_mid = 0.5 * (r_norm[:-1]**2 + r_norm[1:]**2) / h
    dt = dt_df_mid * df
    t_rel = np.zeros_like(f)
    t_rel[1:] = np.cumsum(dt)

    t_rel -= t_rel[0]
    if np.sum(np.diff(f)) < 0:
        t_rel = np.abs(t_rel)

    if np.linalg.norm(F2) < 1e-12:
        if debug:
            print("[DEBUG] F2 is at origin → circular orbit: e_hat set to radial unit")
        e_hat = arc3d[0] / np.linalg.norm(arc3d[0])
    else:
        e_hat = -F2 / np.linalg.norm(F2)

    vel3d = np.empty_like(arc3d)

    for i, r_vec in enumerate(arc3d):
        r = np.linalg.norm(r_vec)
        r_hat = r_vec / r

        # Radial (e_hat) direction contributions
        cos_f = np.dot(e_hat, r_hat)
        sin_f = np.dot(w, np.cross(e_hat, r_hat))
        v_r = (EARTH_MU / h) * e * sin_f
        v_t = (EARTH_MU / h) * (1 + e * cos_f)

        # Use 2D plane projection for stable tangent vector
        xy = _in_plane(r_vec, u, v)
        norm_xy = np.linalg.norm(xy)
        if norm_xy < 1e-12:
            t_hat = np.zeros(3)
        else:
            r_hat_2d = xy / norm_xy
            t_hat_2d = np.array([-r_hat_2d[1], r_hat_2d[0]])
            t_hat = _to_3d(t_hat_2d, u, v)

        if debug and not np.all(np.isfinite([v_r, v_t])):
            print(f"[DEBUG] NaNs detected at i={i}")
            print("r:", r)
            print("r_hat:", r_hat)
            print("cos_f:", cos_f)
            print("sin_f:", sin_f)
            print("v_r:", v_r)
            print("v_t:", v_t)

        vel3d[i] = (v_r * r_hat + v_t * t_hat) * vel_mod

    if debug:
        print("Rotation:", "CCW" if rot_dir > 0 else "CW" if rot_dir < 0 else "Flat/Undefined")
        print("arc3d[0] ≈", "P1" if np.linalg.norm(arc3d[0] - P1) < np.linalg.norm(arc3d[0] - P2) else "P2")

    # vel3d = -vel3d
    r0, v0 = arc3d[0], vel3d[0]
    motion_dir = -np.dot(np.cross(r0, v0), w)
    if debug:
        print("v0 direction:", "CCW" if motion_dir > 0 else "CW")

    if plot or save_path:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.gridspec import GridSpec

        t_minutes = t_rel / 60.0
        d_O = np.linalg.norm(arc3d, axis=1) / 1e3

        fig = plt.figure(figsize=(18, 6))
        gs = GridSpec(1, 3, figure=fig, width_ratios=[3, 2, 2])

        ax3d = fig.add_subplot(gs[0], projection='3d')
        ax_dist = fig.add_subplot(gs[1])
        ax_speed = fig.add_subplot(gs[2])

        colors = cm.get_cmap('RdYlGn_r')(np.linspace(0, 1, len(arc3d)))

        ax3d.scatter(arc3d_comp[:, 0] / 1e3, arc3d_comp[:, 1] / 1e3, arc3d_comp[:, 2] / 1e3, c="black", s=10)
        ax3d.scatter(arc3d[:, 0] / 1e3, arc3d[:, 1] / 1e3, arc3d[:, 2] / 1e3, c=colors, s=10)
        ax3d.scatter(0, 0, 0, marker='x', s=80, color='black', label='Earth')
        ax3d.scatter(*F2 / 1e3, marker='x', s=80, color='gray', alpha=0.5, label='focus')
        ax3d.scatter(*P1 / 1e3, s=40, label='P1', color='green')
        ax3d.scatter(*P2 / 1e3, s=40, label='P2', color='red')

        ax3d.set_xlabel('X [km]')
        ax3d.set_ylabel('Y [km]')
        ax3d.set_zlabel('Z [km]')
        ax3d.set_xlim((-np.max(d_O), np.max(d_O)))
        ax3d.set_ylim((-np.max(d_O), np.max(d_O)))
        ax3d.set_zlim((-np.max(d_O), np.max(d_O)))
        ax3d.set_title('Full ellipse with arc highlighted')
        ax3d.legend()

        ax_dist.plot(t_minutes, d_O, lw=1.5)
        vmin_d, vmax_d = d_O.min(), d_O.max()
        vrange_d = max(vmax_d - vmin_d, 1000)
        ax_dist.set_ylim(vmin_d - 0.05 * vrange_d, vmax_d + 0.05 * vrange_d)
        ax_dist.set_xlabel('Time from P1 [min]')
        ax_dist.set_ylabel('Distance [km]')
        ax_dist.set_title('Distance from origin')
        ax_dist.grid(True)

        speed = np.linalg.norm(vel3d, axis=1)
        speed_km_s = speed / 1e3
        ax_speed.plot(t_minutes, speed_km_s, lw=1.5)
        ax_speed.set_ylim((0, np.max(speed_km_s)*1.1))
        ax_speed.set_xlabel('Time from P1 [min]')
        ax_speed.set_ylabel('Speed [km/s]')
        ax_speed.set_title('Speed over time')
        ax_speed.grid(True)

        if debug:
            from ssapy import Orbit, rv, RK78Propagator, AccelKepler
            from ..integrators import leapfrog, rk4
            from ..time import get_times
            debug_times = get_times(duration=(t_rel[-1], 's'), freq=(1, 's'), t0=0).gps
            orbit = Orbit(r=r0, v=v0, t=debug_times[0])
            print(orbit, orbit.r, orbit.v, orbit.t)
            r_debug, v_debug = rv(orbit=orbit, time=t_rel, propagator=RK78Propagator(AccelKepler(), 1))
            # r_debug, v_debug = rk4(r0=r0, v0=v0, t=t_rel)
            print(f"\nEXTRA PLOT\nr0:{r0}\nv0:{v0}")
            ax3d.plot(r_debug[:, 0] / 1e3, r_debug[:, 1] / 1e3, r_debug[:, 2] / 1e3,
              color='gray', linewidth=9.0, alpha=0.6, label='Integrated')
            ax3d.set_title(f'Full ellipse with arc highlighted')
            ax_dist.plot(t_minutes, np.linalg.norm(r_debug, axis=-1) / 1e3, lw=1.5, c='red', linestyle='--')
            ax_speed.plot(t_minutes, np.linalg.norm(v_debug, axis=-1) / 1e3, lw=1.5, c='red', linestyle='--')

            print(f"shapes: {np.shape(arc3d)}, {np.shape(r_debug)}")
            print(f"Differences: {np.max(np.linalg.norm(arc3d[-1] - r_debug[-1], axis=-1) / 1e3)} km")
        plt.tight_layout()

        if plot:
            plt.show()
        if save_path:
            save_plot(fig, save_path)
    
    inclination = np.arccos(w[2])  # radians
    k_hat = np.array([0, 0, 1])
    n_vec = np.cross(k_hat, w)
    n_norm = np.linalg.norm(n_vec)

    if n_norm < 1e-12:
        RAAN = 0.0
    else:
        RAAN = np.arccos(n_vec[0] / n_norm)
        if n_vec[1] < 0:
            RAAN = 2 * np.pi - RAAN

    e_vec = (np.cross(v0, np.cross(r0, v0)) / EARTH_MU) - (r0 / np.linalg.norm(r0))
    e_vec /= np.linalg.norm(e_vec)

    if n_norm < 1e-12:
        arg_periapsis = np.arccos(np.dot(e_vec, r0) / np.linalg.norm(r0))
    else:
        arg_periapsis = np.arccos(np.dot(n_vec, e_vec) / n_norm)
        if e_vec[2] < 0:
            arg_periapsis = 2 * np.pi - arg_periapsis

    true_anomaly = np.arccos(np.dot(e_vec, r0) / (np.linalg.norm(r0)))
    if np.dot(r0, v0) < 0:
        true_anomaly = 2 * np.pi - true_anomaly

    info = {
        # ───── Orbital Elements (Classical Keplerian) ─────
        'a'              : a,                   # Semi-major axis [m]
        'e'              : e,                   # Eccentricity [-]
        'b'              : b,                   # Semi-minor axis [m]
        'c'              : c,                   # Linear eccentricity [m]
        'i'              : inclination,         # Inclination [rad]
        'raan'           : RAAN,                # Right Ascension of Ascending Node [rad]
        'ap'             : arg_periapsis,       # Argument of Periapsis [rad]
        'ta'             : true_anomaly,        # True anomaly at arc start [rad]
        'T'              : 2 * np.pi * np.sqrt(a**3 / EARTH_MU),  # Orbital period [s]
        'h'              : h,                   # Specific angular momentum [m²/s]

        # ───── Focus and Geometry ─────
        'F2'             : F2,                  # Secondary focus (origin is primary) [m]
        'center'         : C,                   # Ellipse center in orbital plane [2D]
        'R2D'            : R2D,                 # Rotation matrix from canonical ellipse to true ellipse

        # ───── Plane Vectors ─────
        'u'              : u,                   # Unit vector from origin toward P1
        'v'              : v,                   # Unit vector completing plane (in-plane)
        'w'              : w,                   # Normal to orbital plane (right-hand) [unit]

        # ───── Positions and Velocities ─────
        'r0'             : r0,                  # Position at arc start [m]
        'v0'             : v0,                  # Velocity at arc start [m/s]
        'r1'             : arc3d[-1],           # Position at arc end [m]
        'v1'             : vel3d[-1],           # Velocity at arc end [m/s]

        # ───── Arc Properties ─────
        'true_anomaly_start': f[0],             # First point in-plane angle [rad]
        'true_anomaly_end'  : f[-1],            # Last point in-plane angle [rad]
        # ───── Flags ─────
        'collision_avoided': collision_avoided, 
    }

    return arc3d, vel3d, t_rel, info
