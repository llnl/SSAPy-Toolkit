#!/usr/bin/env python3
"""
plot_single_feasible_transfer.py
────────────────────────────────────────────────────────────────────────
* Finds the first departure ellipse (a,e) that can, with ONE burn
  ≤ dv_max, reach rendez‑vous point r2 with velocity direction u2.
* Plots: departure ellipse, burn point, Δv arrow, transfer ellipse,
  rendez‑vous vector, and arrival direction.

All ASCII, no type‑hints, pure Python 3.8+.
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt

MU = 3.986004418e14  # [m^3 s^-2]  Earth GM

# --------------------- rendez‑vous specification -------------------------
r2 = np.array([70_000e3, 0.0, 0.0])   # rendez‑vous position [m]
u2 = np.array([0.0, 1.0, 0.0])        # desired direction (unit)
u2 /= np.linalg.norm(u2)

dv_max = 150.0                        # single impulse budget [m/s]

r2_mag = np.linalg.norm(r2)
r_hat = r2 / r2_mag
k_r = np.dot(u2, r_hat)
k_perp = np.sqrt(max(0.0, 1.0 - k_r**2))
t_hat_global = (u2 - k_r * r_hat) / (k_perp if k_perp > 0 else 1.0)

# --------------------- orbital utility functions -------------------------
def ellipse_rv(a, e, f):
    p = a * (1.0 - e * e)
    r_pf = p / (1.0 + e * np.cos(f))
    x = r_pf * np.cos(f)
    y = r_pf * np.sin(f)
    r_vec = np.array([x, y, 0.0])
    fac = np.sqrt(MU / p)
    v_vec = fac * np.array([-np.sin(f), e + np.cos(f), 0.0])
    return r_vec, v_vec

def ellipse_components_at_r2(a, e):
    """Return the true‑anomaly solutions (could be 0,1,2)."""
    if e == 0.0:
        # circle
        if abs(a - r2_mag) < 1e-6:
            return [0.0]
        return []
    cos_f = (a * (1.0 - e * e) / r2_mag - 1.0) / e
    if abs(cos_f) > 1.0:
        return []
    f1 = np.arccos(np.clip(cos_f, -1.0, 1.0))
    return [f1, -f1] if f1 != 0 else [0.0]

def vr_vt(a, e, f):
    p = a * (1.0 - e * e)
    h = np.sqrt(MU * p)
    v_r = MU / h * e * np.sin(f)
    v_t = MU / h * (1.0 + e * np.cos(f))
    return v_r, v_t, h

def feasible_departure(a, e):
    # radius limits
    rp = a * (1.0 - e)
    ra = a * (1.0 + e)
    if not (rp - 1e-3 <= r2_mag <= ra + 1e-3):
        return None
    for f in ellipse_components_at_r2(a, e):
        v_r, v_t, h = vr_vt(a, e, f)
        # choose in‑plane tangential unit aligned with u2 projection
        t_hat = t_hat_global
        v_pre = v_r * r_hat + v_t * t_hat
        s = v_pre.dot(u2)             # arrival speed maximising projection
        v_post = s * u2
        dv_vec = v_post - v_pre
        dv_mag = np.linalg.norm(dv_vec)
        if dv_mag <= dv_max + 1e-6:
            return f, v_pre, v_post, dv_vec
    return None

def post_orbit_from_state(r_vec, v_vec):
    r = np.linalg.norm(r_vec)
    v2 = np.dot(v_vec, v_vec)
    eps = v2 * 0.5 - MU / r
    if eps >= 0.0:     # unbound; skip plotting
        return None
    a = -MU / (2.0 * eps)
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    e = np.sqrt(max(0.0, 1.0 - h * h / (MU * a)))
    e_vec = (np.cross(v_vec, h_vec) / MU) - r_vec / r
    e_hat = e_vec / (np.linalg.norm(e_vec) if np.linalg.norm(e_vec) > 0 else 1.0)
    t_hat = np.cross(h_vec / h, e_hat)
    # sample positions
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    p = a * (1.0 - e * e)
    r_pf = p / (1.0 + e * np.cos(theta))
    pts = (r_pf * np.cos(theta))[:, None] * e_hat + \
          (r_pf * np.sin(theta))[:, None] * t_hat
    return pts

# --------------------- scan for first feasible ellipse --------------------
a_grid = np.linspace(r2_mag / 4.0, 5.0 * r2_mag, 300)
e_grid = np.linspace(0.0, 0.95, 300)

found = None
for a, e in itertools.product(a_grid, e_grid):
    ans = feasible_departure(a, e)
    if ans is not None:
        found = (a, e) + ans  # unpack later
        break

if found is None:
    print("No feasible ellipse in the scan domain.")
    exit()

# --------------------- unpack solution ------------------------------------
a_dep, e_dep, f_burn, v_pre, v_post, dv_vec = found
print("Departure ellipse found:")
print("  a = %.1f km   e = %.4f   true‑anomaly burn = %.2f deg"
      % (a_dep / 1e3, e_dep, np.degrees(f_burn)))
print("  |Δv| = %.2f m/s" % np.linalg.norm(dv_vec))

# sample departure ellipse
theta_dep = np.linspace(0.0, 2.0 * np.pi, 400)
r_dep = []
for tt in theta_dep:
    rr, _ = ellipse_rv(a_dep, e_dep, tt)
    r_dep.append(rr)
r_dep = np.vstack(r_dep)

r_burn, _ = ellipse_rv(a_dep, e_dep, f_burn)

# post‑burn orbit points
post_pts = post_orbit_from_state(r2, v_post)

# --------------------- plot ------------------------------------------------
plt.figure(figsize=(7, 6))
plt.plot(r_dep[:, 0] / 1e3, r_dep[:, 1] / 1e3, 'C0', lw=1.0,
         label='Departure ellipse')
plt.scatter(r_burn[0] / 1e3, r_burn[1] / 1e3, c='C0', s=40, label='Burn point')

# Δv arrow
plt.arrow(r_burn[0] / 1e3, r_burn[1] / 1e3,
          dv_vec[0] * 5.0, dv_vec[1] * 5.0,   # scale for visibility
          head_width=500, head_length=1000, color='k', linewidth=1.5,
          length_includes_head=True, label='Δv')

if post_pts is not None:
    plt.plot(post_pts[:, 0] / 1e3, post_pts[:, 1] / 1e3,
             'C2', lw=1.0, label='Transfer ellipse')

# rendez‑vous point and direction
plt.scatter(r2[0] / 1e3, r2[1] / 1e3, c='red', s=60, label='rendez‑vous')
plt.arrow(r2[0] / 1e3, r2[1] / 1e3,
          u2[0] * 30_000, u2[1] * 30_000,  # arrow scaled 30 000 km
          color='red', linewidth=2,
          length_includes_head=True, head_width=1000,
          label='Desired direction')

plt.gca().set_aspect('equal')
plt.xlabel('x  [km]')
plt.ylabel('y  [km]')
plt.title('Departure & transfer trajectory (Δv ≤ %.0f m/s)' % dv_max)
plt.grid(True, ls='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
