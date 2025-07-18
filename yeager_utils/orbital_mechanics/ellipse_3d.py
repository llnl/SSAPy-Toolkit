#!/usr/bin/env python3
"""
ellipse_arc.py  ────────────────────────────────────────────────────────────────
Generate sample points and analytical velocities along an ellipse in ℝ³ whose
first focus is at the origin.

Returns
-------
arc3d : (N,3) ndarray   Cartesian positions  [km]
vel3d : (N,3) ndarray   Inertial velocities  [km s⁻¹]
t_rel : (N,)  ndarray   Relative flight‑time [s]  (t_rel[0] == 0)
info  : dict            {'a', 'e', 'F2'}
"""

from ..constants import EARTH_MU           # [m³ s⁻²]
from ..plots     import save_plot
import numpy as np

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

        sol = minimize(obj_min_e, 0.3*p1_2d,
                       constraints=[NonlinearConstraint(equal_sum, 0., 0.,
                                                        jac="2-point", hess=BFGS())],
                       method="trust-constr", tol=tol)
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

# ─────────────────────────── public API ─────────────────────────────
def ellipse_arc(P1, P2, *,
                a=None, e=None, F2=None, inc: float = 0.0,
                n_pts: int = 1000, tol=1e-10,
                ccw: bool = True,
                plot=False, save_path=False,
                debug=False):
    
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

    F2, a, e = _solve_focus(P1, P2, u, v, a=a, e=e, F2=F2, tol=tol)

    f2_2d = _in_plane(F2, u, v)
    C     = 0.5 * f2_2d
    c     = 0.5 * np.linalg.norm(F2)
    b     = np.sqrt(max(a*a - c*c, 0.0))

    phi   = -np.arctan2(C[1], C[0])
    R2D   = np.array([[np.cos(phi), -np.sin(phi)],
                      [np.sin(phi),  np.cos(phi)]])

    t_full = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    xy_full = np.vstack((a * np.cos(t_full), b * np.sin(t_full))).T
    xy_full = (R2D.T @ xy_full.T).T + C
    full3d = np.array([_to_3d(p, u, v) for p in xy_full])

    i1 = np.argmin(np.linalg.norm(full3d - P1, axis=1))
    i2 = np.argmin(np.linalg.norm(full3d - P2, axis=1))

    arc_short, arc_long = slice_arc_segments(full3d, i1, i2)
    t_short,  t_long     = slice_arc_segments(t_full,  i1, i2)
    arc_long = arc_long[::-1]
    if debug:
        print(f"Directions: short:{direction(arc_short)}, long: {direction(arc_long)}")

    if ccw:
        if direction(arc_short) > 0:
            arc3d      = arc_short
            t_arc      = t_short
            arc3d_comp = arc_long
            t_arc_comp = t_long
        else:
            if debug:
                print("Changing to the long arc for CCW.")
            arc3d      = arc_long
            t_arc      = t_long
            arc3d_comp = arc_short
            t_arc_comp = t_short            
    else:
        arc3d      = arc_short
        t_arc      = t_short
        arc3d_comp = arc_long
        t_arc_comp = t_long

    rot_dir = direction(arc3d)

    n0 = np.sqrt(EARTH_MU / a**3)
    E = 2*np.arctan(np.tan(t_arc/2) / np.sqrt((1+e)/(1-e)))
    if rot_dir > 0:
        E = np.unwrap(E)
    else:
        E = -np.unwrap(-E)
    M = E - e*np.sin(E)
    t_rel = M / n0
    t_rel -= t_rel[0]

    mu = EARTH_MU
    h  = np.sqrt(mu * a * (1 - e**2))
    e_hat = -F2 / np.linalg.norm(F2)

    vel3d = np.empty_like(arc3d)
    for i, r_vec in enumerate(arc3d):
        r      = np.linalg.norm(r_vec)
        r_hat  = r_vec / r
        cos_f  = np.dot(e_hat, r_hat)
        sin_f  = np.dot(w, np.cross(e_hat, r_hat))
        v_r    = (mu / h) * e * sin_f
        v_t    = (mu / h) * (1 + e * cos_f)
        t_hat = np.cross(w, r_hat) * rot_dir
        vel3d[i] = v_r * r_hat + v_t * t_hat

    if debug:
        print("Rotation:", "CCW" if rot_dir > 0 else "CW" if rot_dir < 0 else "Flat/Undefined")
        print("arc3d[0] ≈", "P1" if np.linalg.norm(arc3d[0] - P1) < np.linalg.norm(arc3d[0] - P2) else "P2")

    r0, v0 = arc3d[0], vel3d[0]
    motion_dir = np.dot(np.cross(r0, v0), w)
    if debug:
        print("v0 direction:", "CCW" if motion_dir > 0 else "CW")

    if plot or save_path:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(18, 6))
        gs = GridSpec(1, 3, figure=fig, width_ratios=[3, 2, 2])

        ax3d = fig.add_subplot(gs[0], projection='3d')
        ax_dist = fig.add_subplot(gs[1])
        ax_speed = fig.add_subplot(gs[2])

        colors = cm.get_cmap('RdYlGn_r')(np.linspace(0, 1, len(t_arc)))

        ax3d.scatter(arc3d_comp[:, 0], arc3d_comp[:, 1], arc3d_comp[:, 2], c="black", s=10)
        ax3d.scatter(arc3d[:, 0], arc3d[:, 1], arc3d[:, 2], c=colors, s=10)
        ax3d.scatter(0, 0, 0, marker='x', s=80, color='black', label='Earth')
        ax3d.scatter(*F2, marker='x', s=80, color='gray', alpha=0.5, label='focus')
        ax3d.scatter(*P1, s=40, label='P1', color='green')
        ax3d.scatter(*P2, s=40, label='P2', color='red')

        ax3d.set_xlabel('X [km]')
        ax3d.set_ylabel('Y [km]')
        ax3d.set_zlabel('Z [km]')
        ax3d.set_title('Full ellipse with arc highlighted')
        ax3d.legend()
        ax3d.set_box_aspect([1, 1, 1])

        t_minutes = t_rel / 60.0
        d_O = np.linalg.norm(arc3d, axis=1)
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
        vmin_s, vmax_s = speed_km_s.min(), speed_km_s.max()
        vrange_s = max(vmax_s - vmin_s, 1.0)
        ax_speed.set_ylim(vmin_s - 0.05 * vrange_s, vmax_s + 0.05 * vrange_s)
        ax_speed.set_xlabel('Time from P1 [min]')
        ax_speed.set_ylabel('Speed [km/s]')
        ax_speed.set_title('Speed over time')
        ax_speed.grid(True)

        plt.tight_layout()

        if plot:
            plt.show()
        if save_path:
            save_plot(fig, save_path)

    return arc3d, vel3d, t_rel, {'a': a, 'e': e, 'F2': F2}
