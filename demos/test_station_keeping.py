"""
Orbit Comparison: Keplerian Reference vs. High-Fidelity Perturbations

Compares:
- Reference: Pure Keplerian (analytical, no perturbations)
- Perturbed: Numerical with all SSAPy accelerations (J2, harmonics, Sun,
  Moon, planets, SRP, drag)
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from PIL import Image as PILImage

from ssapy.orbit import Orbit
from ssapy.propagator import KeplerianPropagator, SciPyPropagator
from ssapy.accel import AccelKepler, AccelSolRad, AccelDrag, AccelEarthRad
from ssapy.gravity import AccelHarmonic, AccelThirdBody
from ssapy.body import get_body
from ssapy.utils import find_file
import ssapy.compute as compute

from yeager_utils.constants import EARTH_MU, EARTH_RADIUS, RGEO

# ── Configuration ─────────────────────────────────────────────────────────────

# Orbit parameters
a_m = 1.0 * RGEO            # semi-major axis (m)
ecc = 0.001                 # eccentricity
inc_deg = 0.0               # inclination (degrees)
raan_deg = 0.0              # RAAN (degrees)
aop_deg = 0.0               # argument of periapsis (degrees)
nu0_deg = 0.0               # true anomaly (degrees)

# Time span
t0_str = "2026-01-01T00:00:00"
duration_hours = 24.0
dt_seconds = 60.0           # sample every 60 seconds

# Spacecraft properties (for drag and SRP)
sc_mass_kg = 100.0
sc_area_m2 = 1.0
sc_Cd = 2.2
sc_Cr = 1.3

# Output
figure_dir = os.path.expanduser("~/yu_figures/tests")
os.makedirs(figure_dir, exist_ok=True)

# ── Setup ─────────────────────────────────────────────────────────────────────

print("=" * 70)
print("ORBIT COMPARISON: Keplerian vs. High-Fidelity")
print("=" * 70)
print(f"Orbit: a={a_m/1e3:.0f} km, e={ecc}, i={inc_deg}°")
print(f"Duration: {duration_hours} hours, dt={dt_seconds} s")
print(f"S/C: m={sc_mass_kg} kg, A={sc_area_m2} m², Cd={sc_Cd}, Cr={sc_Cr}")
print()

# Time array
t0_gps = Time(t0_str, scale="utc").gps
n_points = int(duration_hours * 3600.0 / dt_seconds) + 1
times_gps = np.linspace(t0_gps, t0_gps + duration_hours * 3600.0, n_points)
times_astropy = Time(times_gps, format="gps", scale="utc")

# Create initial orbit
orbit_initial = Orbit.fromKeplerianElements(
    a_m,
    ecc,
    np.radians(inc_deg),
    np.radians(raan_deg),
    np.radians(aop_deg),
    np.radians(nu0_deg),
    t=t0_gps
)

print(f"Initial orbit created at t0 = {t0_str}")
print(f"Initial position: {orbit_initial.r / 1e3} km")
print()

# ── Reference Orbit: Pure Keplerian ───────────────────────────────────────────

print("[1/2] Propagating reference orbit (Keplerian)...")
kep_prop = KeplerianPropagator()
r_ref, v_ref = compute.rv(orbit_initial, times_gps, kep_prop)
print(f"      Shape: r_ref = {r_ref.shape}, v_ref = {v_ref.shape}")

# ── Perturbed Orbit: High-Fidelity ────────────────────────────────────────────

print("[2/2] Propagating perturbed orbit (all SSAPy forces)...")

# Get celestial bodies
earth = get_body("earth")
moon = get_body("moon")
sun = get_body("sun")

# Build high-fidelity acceleration model
accel_perturbed = (
    AccelKepler(earth.mu)
    + AccelHarmonic(earth, 20, 20)      # Earth gravity harmonics (20x20)
    + AccelThirdBody(sun)                # Sun point-mass
    + AccelThirdBody(moon)               # Moon point-mass
    + AccelHarmonic(moon, 10, 10)        # Moon harmonics (10x10)
    + AccelSolRad(CR=sc_Cr, area=sc_area_m2, mass=sc_mass_kg)
    + AccelDrag(CD=sc_Cd, area=sc_area_m2, mass=sc_mass_kg)
    + AccelEarthRad(CR=sc_Cr, area=sc_area_m2, mass=sc_mass_kg)
)

print("      Force model:")
print("        ✓ Earth point-mass + 20×20 harmonics")
print("        ✓ Sun point-mass")
print("        ✓ Moon point-mass + 10×10 harmonics")
print("        ✓ Solar radiation pressure")
print("        ✓ Atmospheric drag")
print("        ✓ Earth radiation pressure")

# Propagate with SciPy integrator
scipy_prop = SciPyPropagator(accel_perturbed)
orbit_perturbed = Orbit(
    r=orbit_initial.r,
    v=orbit_initial.v,
    t=t0_gps,
    propkw=dict(CD=sc_Cd, CR=sc_Cr, area=sc_area_m2, mass=sc_mass_kg)
)
r_pert, v_pert = compute.rv(orbit_perturbed, times_gps, scipy_prop)
print(f"      Shape: r_pert = {r_pert.shape}, v_pert = {v_pert.shape}")
print()

# ── Compute Differences ───────────────────────────────────────────────────────

print("Computing differences...")

# Position and velocity differences
dr = r_pert - r_ref
dv = v_pert - v_ref

# Separation magnitudes
sep_mag = np.linalg.norm(dr, axis=1)
vsep_mag = np.linalg.norm(dv, axis=1)

# Time arrays for plotting
times_hours = (times_gps - t0_gps) / 3600.0

print(f"  Max position error:  {np.max(sep_mag):.2f} m")
print(f"  Final position error: {sep_mag[-1]:.2f} m")
print(f"  Max velocity error:  {np.max(vsep_mag) * 1e3:.3f} mm/s")
print(f"  Final velocity error: {vsep_mag[-1] * 1e3:.3f} mm/s")
print()

# ── NTW Frame Errors ──────────────────────────────────────────────────────────


def ntw_frame(r, v):
    """Compute NTW basis vectors."""
    t = r / np.linalg.norm(r)
    w = np.cross(r, v)
    w = w / np.linalg.norm(w)
    n = np.cross(w, t)
    return n, t, w


# Compute NTW components
sep_ntw = np.zeros((len(times_gps), 3))
for i in range(len(times_gps)):
    n, t, w = ntw_frame(r_ref[i], v_ref[i])
    sep_ntw[i, 0] = np.dot(dr[i], n)  # Along-track
    sep_ntw[i, 1] = np.dot(dr[i], t)  # Radial
    sep_ntw[i, 2] = np.dot(dr[i], w)  # Cross-track

# ── Dashboard Plot ────────────────────────────────────────────────────────────

print("Creating divergence dashboard...")

fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Left: 3D orbit plot
ax_3d = fig.add_subplot(gs[:, 0], projection='3d')

# Convert to km for plotting
r_ref_km = r_ref / 1e3
r_pert_km = r_pert / 1e3

# Plot both orbits
ax_3d.plot(r_ref_km[:, 0], r_ref_km[:, 1], r_ref_km[:, 2],
           color='dodgerblue', lw=2, label='Keplerian', alpha=0.8)
ax_3d.plot(r_pert_km[:, 0], r_pert_km[:, 1], r_pert_km[:, 2],
           color='crimson', lw=2, label='High-Fidelity (All Perturbations)',
           alpha=0.8, ls='--')

# Load Earth texture and create textured globe
try:
    earth_png = PILImage.open(find_file("earth", ext=".png"))
    scale = 2  # Adjust for performance (higher = lower resolution)
    earth_png = earth_png.resize((5400 // scale, 2700 // scale))
    bm = np.array(earth_png.resize([int(d) for d in earth_png.size])) / 256.
    
    # Create mesh coordinates for textured Earth
    lons = np.linspace(-180, 180, bm.shape[1]) * np.pi / 180
    lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi / 180
    earth_radius_km = EARTH_RADIUS / 1e3
    
    mesh_x = np.outer(np.cos(lons), np.cos(lats)).T * earth_radius_km
    mesh_y = np.outer(np.sin(lons), np.cos(lats)).T * earth_radius_km
    mesh_z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T * earth_radius_km
    
    # Plot textured Earth
    ax_3d.plot_surface(mesh_x, mesh_y, mesh_z, rstride=4, cstride=4,
                       facecolors=bm, shade=False, zorder=0)
except Exception as e:
    print(f"Warning: Could not load Earth texture ({e}), using simple sphere")
    # Fallback to simple colored sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    earth_radius_km = EARTH_RADIUS / 1e3
    x_earth = earth_radius_km * np.outer(np.cos(u), np.sin(v))
    y_earth = earth_radius_km * np.outer(np.sin(u), np.sin(v))
    z_earth = earth_radius_km * np.outer(np.ones(np.size(u)), np.cos(v))
    ax_3d.plot_surface(x_earth, y_earth, z_earth, color='lightblue',
                       alpha=0.5, edgecolor='none')

ax_3d.set_xlabel("X [km]", fontsize=10, labelpad=8)
ax_3d.set_ylabel("Y [km]", fontsize=10, labelpad=8)
ax_3d.set_zlabel("Z [km]", fontsize=10, labelpad=8)
ax_3d.set_title("3D Orbit Trajectories", fontsize=12, fontweight='bold',
                pad=15)
ax_3d.legend(fontsize=10, loc='upper left')
ax_3d.grid(True, alpha=0.2)

# Equal aspect ratio - calculate proper symmetric limits
# Combine both reference and perturbed orbits for range calculation
r_combined = np.vstack([r_ref_km, r_pert_km])

# Calculate the range for each axis
max_range_x = (r_combined[:, 0].max() - r_combined[:, 0].min()) / 2.0
max_range_y = (r_combined[:, 1].max() - r_combined[:, 1].min()) / 2.0
max_range_z = (r_combined[:, 2].max() - r_combined[:, 2].min()) / 2.0
max_range = max(max_range_x, max_range_y, max_range_z)

# Calculate the center point
mid_x = (r_combined[:, 0].max() + r_combined[:, 0].min()) * 0.5
mid_y = (r_combined[:, 1].max() + r_combined[:, 1].min()) * 0.5
mid_z = (r_combined[:, 2].max() + r_combined[:, 2].min()) * 0.5

# Set symmetric limits around the center
ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

# Enforce equal aspect ratio for the 3D box
ax_3d.set_box_aspect([1, 1, 1])

# Right: NTW position errors (3 subplots stacked)
ax_n = fig.add_subplot(gs[0, 1])
ax_t = fig.add_subplot(gs[1, 1], sharex=ax_n)
ax_w = fig.add_subplot(gs[2, 1], sharex=ax_n)

axes_ntw = [ax_n, ax_t, ax_w]
labels_ntw = ["Along-track (N)", "Radial (T)", "Cross-track (W)"]
colors_ntw = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for i, (ax, label, color) in enumerate(zip(axes_ntw, labels_ntw,
                                            colors_ntw)):
    ax.plot(times_hours, sep_ntw[:, i], lw=1.5, color=color)
    ax.axhline(0, color="k", lw=0.8, alpha=0.3, ls='--')
    ax.set_ylabel(f"{label}\n[m]", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, times_hours[-1])
    # Add final value annotation
    final_val = sep_ntw[-1, i]
    ax.text(0.98, 0.95, f"Final: {final_val:.1f} m",
            transform=ax.transAxes, fontsize=9,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.8))

ax_n.set_title("NTW Position Errors", fontsize=12, fontweight='bold', pad=10)
ax_w.set_xlabel("Time [hours]", fontsize=11)
plt.setp(ax_n.get_xticklabels(), visible=False)
plt.setp(ax_t.get_xticklabels(), visible=False)

# Overall title with formatted orbital elements (2-3 significant figures)
a_km = a_m / 1e3

# Format a_km to show appropriate precision
if a_km >= 10000:
    a_str = f"{a_km:.2e}"
elif a_km >= 1000:
    a_str = f"{a_km:.0f}"
elif a_km >= 100:
    a_str = f"{a_km:.1f}"
else:
    a_str = f"{a_km:.2f}"

# Format eccentricity (3 sig figs for small values)
if ecc < 0.01:
    e_str = f"{ecc:.3f}"
else:
    e_str = f"{ecc:.2f}"

# Format inclination (2 decimal places or integer if whole number)
if inc_deg == int(inc_deg):
    i_str = f"{int(inc_deg)}"
else:
    i_str = f"{inc_deg:.2f}"

# Format duration appropriately
if duration_hours == int(duration_hours):
    dur_str = f"{int(duration_hours)}"
else:
    dur_str = f"{duration_hours:.1f}"

fig.suptitle(
    f"Orbit Comparison: Keplerian vs. High-Fidelity Perturbations\n"
    f"a={a_str} km, e={e_str}, i={i_str}°, Duration={dur_str}h",
    fontsize=14, fontweight='bold', y=0.98
)

fig.tight_layout(rect=[0, 0, 1, 0.96])
save_path = os.path.join(figure_dir, "orbit_comparison_dashboard.png")
fig.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {save_path}")
print()

# ── Summary ───────────────────────────────────────────────────────────────────

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Duration:             {duration_hours:.1f} hours")
print(f"Samples:              {len(times_gps)}")
print(f"Max position error:   {np.max(sep_mag):.2f} m "
      f"({np.max(sep_mag)/1e3:.3f} km)")
print(f"Final position error: {sep_mag[-1]:.2f} m "
      f"({sep_mag[-1]/1e3:.3f} km)")
print(f"Max velocity error:   {np.max(vsep_mag)*1e3:.3f} mm/s")
print(f"Final velocity error: {vsep_mag[-1]*1e3:.3f} mm/s")
print()
print("NTW errors at end:")
print(f"  Along-track (N): {sep_ntw[-1, 0]:10.2f} m")
print(f"  Radial (T):      {sep_ntw[-1, 1]:10.2f} m")
print(f"  Cross-track (W): {sep_ntw[-1, 2]:10.2f} m")
print("=" * 70)