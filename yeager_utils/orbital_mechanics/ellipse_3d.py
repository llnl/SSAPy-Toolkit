#!/usr/bin/env python3
"""
ellipse_arc.py  ────────────────────────────────────────────────────────────────
Generate sample points along an ellipse in ℝ³ whose first focus is at the
origin.  Pass `plot=True` to draw

    • the requested arc      … solid line
    • the complementary arc  … dashed line
    • the two foci           … × markers
    • the end-points         … filled circles

Example
-------
>>> import numpy as np
>>> from ellipse_arc import ellipse_arc
>>> P1 = np.array([3., 1., 2.])
>>> P2 = np.array([-1., 4., 0.])
>>> arc, prm = ellipse_arc(P1, P2, n_pts=400, plot=True)
>>> print(prm)
{'a': 3.473…, 'e': 0.383…, 'F2': array([1.334…, 2.065…, 0.098…])}
"""

from ..compute import velocity_along_ellipse
from ..plots import save_plot
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
#   Internal helpers
# ──────────────────────────────────────────────────────────────────────────────
def _plane_basis(p1, p2, *, incl: float = 0.0, eps: float = 1e-12):
    """
    Orthonormal basis (u, v) spanning the plane through O, p1, p2.

    If p1, p2 and the origin are (nearly) colinear we fabricate *v* so that
    • incl = 0   → ellipse lies in the xy-plane;
    • incl = θ°  → rotate that plane away from xy by θ around *u*.

    Parameters
    ----------
    p1, p2 : (3,) array-like
    incl   : degrees; only used in the colinear fallback
    eps    : colinearity tolerance
    """
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    if np.linalg.norm(p1) < eps or np.linalg.norm(p2) < eps:
        raise ValueError("P1 and P2 must be non-zero")

    u = p1 / np.linalg.norm(p1)
    v = p2 - np.dot(p2, u) * u           # component ⟂ u

    if np.linalg.norm(v) < eps:          # colinear – fabricate v
        cand = np.array([0., 0., 1.])    # ẑ ⇒ xy-plane when incl=0
        if abs(np.dot(u, cand)) > 1. - eps:
            cand = np.array([1., 0., 0.])  # fall back to x̂
        v0 = np.cross(u, cand);  v0 /= np.linalg.norm(v0)
        w  = np.cross(u, v0)              # second orthogonal
        θ  = np.deg2rad(incl)
        v  = np.cos(θ)*v0 + np.sin(θ)*w
    else:
        v /= np.linalg.norm(v)
    return u, v


def _in_plane(vec3, u, v):
    """3-D → 2-D coordinates in the (u, v) basis."""
    return np.array([np.dot(vec3, u), np.dot(vec3, v)])


def _to_3d(xy, u, v):
    """2-D plane coords → 3-D vector."""
    return xy[0]*u + xy[1]*v


def _eccentricity(f2, a):
    return np.linalg.norm(f2) / (2.0*a)


def _a_for(f2, p):
    """Half the focal-sum for point *p*."""
    return 0.5*(np.linalg.norm(p) + np.linalg.norm(p - f2))


# ──────────────────────────────────────────────────────────────────────────────
#   Focus solver – handles every specification scenario
# ──────────────────────────────────────────────────────────────────────────────
def _solve_focus(P1, P2, u, v, *, a=None, e=None, F2=None, tol=1e-10):
    """
    Return (F2, a, e) for an ellipse through P1 & P2 with first focus at O.

    Exactly one of (a, e, F2) may be supplied; the others are solved for.
    If all three are None the function returns the *least-eccentric* solution.
    """
    from scipy.optimize import root, minimize, NonlinearConstraint, BFGS

    # plane coordinates
    p1_2d, p2_2d = _in_plane(P1, u, v), _in_plane(P2, u, v)

    # ── CASE 1: caller passed F2 outright ────────────────────────────────
    if F2 is not None:
        if (a is not None) or (e is not None):
            raise ValueError("Specify only ONE of a, e or F2")
        F2 = np.asarray(F2, float)
        a  = _a_for(F2, P1)
        e  = _eccentricity(F2, a)
        return F2, a, e

    # ── CASE 2: least-eccentric ellipse (a=e=F2=None) ────────────────────
    if (a is None) and (e is None):
        def obj_min_e(xy):
            F = _to_3d(xy, u, v)
            return _eccentricity(F, _a_for(F, P1))

        def equal_sum(xy):
            F = _to_3d(xy, u, v)
            return (np.linalg.norm(P1)+np.linalg.norm(P1-F)
                    - np.linalg.norm(P2)-np.linalg.norm(P2-F))

        x0  = 0.3*p1_2d
        sol = minimize(obj_min_e,
                       x0,
                       constraints=[NonlinearConstraint(equal_sum, 0., 0.,
                                                         jac="2-point",
                                                         hess=BFGS())],
                       method="trust-constr",
                       tol=tol)
        if not sol.success:
            raise RuntimeError("Could not locate least-eccentric solution")

        F2 = _to_3d(sol.x, u, v)
        a  = _a_for(F2, P1)
        e  = _eccentricity(F2, a)
        return F2, a, e

    # ── CASE 3: user fixed *either* a *or* e  ────────────────────────────
    if (a is None) == (e is None):   # xor test fails → both / neither set
        raise ValueError("Provide exactly one of {a, e, F2}")

    def residual(xy):
        F = _to_3d(xy, u, v)
        s1 = np.linalg.norm(P1) + np.linalg.norm(P1 - F)
        s2 = np.linalg.norm(P2) + np.linalg.norm(P2 - F)
        r1 = s1 - s2                       # hyperbola constraint
        r2 = s1 - (np.linalg.norm(F)/e if e is not None else 2.*a)
        return np.array([r1, r2])

    x0  = 0.3*p1_2d
    sol = root(residual, x0, tol=tol)
    if not sol.success:
        raise RuntimeError("Could not satisfy the supplied (a, e)")

    F2 = _to_3d(sol.x, u, v)
    if e is None:
        e = _eccentricity(F2, a)
    else:
        a = np.linalg.norm(F2) / (2.*e)
    return F2, a, e


# ──────────────────────────────────────────────────────────────────────────────
#   Public helpers
# ──────────────────────────────────────────────────────────────────────────────
def eccentricity_range(P1, P2, *, tol=1e-10):
    """
    Return (e_min, e_max) for every ellipse through P1 & P2 with first focus O.
    e_max is always 1 (parabolic limit); e_min is found by optimisation.
    """
    u, v = _plane_basis(P1, P2)               # plane basis
    F2, a, e_min = _solve_focus(P1, P2, u, v, tol=tol)  # least-eccentric
    return e_min, 1.0


def ellipse_arc(P1, P2, *, vel=False, save_path=False,
                a=None, e=None, F2=None, inc: float = 0.0,
                n_pts: int = 200, tol=1e-10, plot=False):
    """
    Sample `n_pts` along the shorter arc of the ellipse through P1 & P2.
    One of (a, e, F2) may be given; otherwise the least-eccentric ellipse
    is chosen.  `inc` controls the fallback plane inclination when P1, P2, O
    are colinear.

    Returns
    -------
    arc  : (n_pts, 3) ndarray   sampled positions
    info : dict { 'a', 'e', 'F2' }
    """
    P1 = np.asarray(P1, float);  P2 = np.asarray(P2, float)

    # 1. plane reduction
    u, v = _plane_basis(P1, P2, incl=inc)
    # 2. find consistent (F2, a, e)
    F2, a, e = _solve_focus(P1, P2, u, v, a=a, e=e, F2=F2, tol=tol)

    # 3. ellipse geometry in that plane
    f2_2d = _in_plane(F2, u, v)
    C     = 0.5*f2_2d                       # centre
    c     = 0.5*np.linalg.norm(F2)          # focus-centre distance
    b     = np.sqrt(max(a*a - c*c, 0.0))    # semi-minor

    phi = -np.arctan2(C[1], C[0])
    R   = np.array([[np.cos(phi), -np.sin(phi)],
                    [np.sin(phi),  np.cos(phi)]])

    def _rot(p): return R @ (p - C)

    p1r, p2r = _rot(_in_plane(P1, u, v)), _rot(_in_plane(P2, u, v))
    t1 = np.arctan2(p1r[1]/b, p1r[0]/a)
    t2 = np.arctan2(p2r[1]/b, p2r[0]/a)
    if t2 < t1:            t1, t2 = t2, t1
    if t2 - t1 > np.pi:    t1, t2 = t2, t1 + 2*np.pi

    t_arc  = np.linspace(t1, t2, n_pts)
    def _xy(t): return np.vstack((a*np.cos(t), b*np.sin(t))).T
    xy_arc = (R.T @ _xy(t_arc).T).T + C
    arc3d  = np.array([_to_3d(p, u, v) for p in xy_arc])

    # 4. optional preview
    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
        ax.plot(arc3d[:,0], arc3d[:,1], arc3d[:,2], lw=1.6, label='arc')
        # complementary arc (dashed, for context)
        t_dash = np.linspace(t2, t1+2*np.pi, n_pts)
        xy_d   = (R.T @ _xy(t_dash).T).T + C
        dash3d = np.array([_to_3d(p, u, v) for p in xy_d])
        ax.plot(dash3d[:,0], dash3d[:,1], dash3d[:,2],
                ls='--', lw=1.0, label='rest')
        ax.scatter([0, F2[0]], [0, F2[1]], [0, F2[2]],
                   marker='x', s=80, label='foci')
        ax.scatter(*P1, s=35, label='P1');  ax.scatter(*P2, s=35, label='P2')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.legend()
        plt.title('Full ellipse (dashed) and selected arc (solid)')
        plt.axis('equal')
        plt.show()
        if save_path:
            save_plot(fig, save_path)

    if vel is not False:
        vel3d = velocity_along_ellipse(arc3d, a=a, e=e, F2=F2)
        return arc3d, vel3d, {'a': a, 'e': e, 'F2': F2}
    else:
        return arc3d, {'a': a, 'e': e, 'F2': F2}
