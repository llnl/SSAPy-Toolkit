# Save all plots with figpath (no interactive windows).
# This script mirrors the original example but routes every figure to disk.

import numpy as np
import matplotlib.pyplot as plt

from yeager_utils import figpath, save_plot

# Pull in ssapy pieces explicitly so the call signatures are unambiguous.
from ssapy import Time
from ssapy import compute
from ssapy import constants
from ssapy import utils, plotUtils
from ssapy import Orbit, rv, get_body
from ssapy.accel import AccelKepler, AccelEarthRad, AccelSolRad
from ssapy.gravity import AccelHarmonic, AccelThirdBody
from ssapy.propagator import SciPyPropagator

# ------------------------------------------------------------
# Epoch
# ------------------------------------------------------------
t0 = Time("2024-01-01", scale="utc")  # use full ISO for clarity
print("t0:", t0)

# ------------------------------------------------------------
# Moon state (simple finite-difference velocity)
# ------------------------------------------------------------
r_moon = get_body("moon").position(t0).T              # shape (1,3)
r_moon_plus = get_body("moon").position(t0 + 1).T     # +1 day
v_moon = (r_moon - r_moon_plus) / 2.0                 # crude approx [m/s]
print("r_moon[0]:", r_moon[0], "v_moon[0]:", v_moon[0])

# Example lunar-bound-ish initial state (not used below—keeping for reference)
r0 = r_moon[0] + (1000e3 * r_moon[0] / np.linalg.norm(r_moon[0]))
v0 = v_moon[0] + 100.0
print("Example r0:", r0, "\nExample v0:", v0)

print("\nCalculating orbit...")

# ------------------------------------------------------------
# Keplerian elements → Orbit object at t0
# ------------------------------------------------------------
a    = constants.RGEO
e    = 0.0
i    = np.radians(45.0)
pa   = np.radians(0.0)
raan = np.radians(0.0)
ta   = np.radians(180.0)

kElements = [a, e, i, pa, raan, ta]
# Note: fromKeplerianElements takes t0 positionally at the end in this install
orbit = Orbit.fromKeplerianElements(*kElements, t0)

# ------------------------------------------------------------
# Spacecraft + force models
# ------------------------------------------------------------
sat_kwargs = dict(
    mass=100.0,  # [kg]
    area=1.0,    # [m^2]
    CD=2.3,      # Drag coefficient
    CR=1.3,      # Solar radiation pressure coefficient
)

moon  = get_body("moon")
sun   = get_body("sun")
Earth = get_body("earth", model="EGM2008")
# Other bodies retrieved above in the original example aren’t required by the
# current acceleration set; keeping the core set used below.

aEarth    = AccelKepler() + AccelHarmonic(Earth, 140, 140)
aSun      = AccelThirdBody(sun)
aMoon     = AccelThirdBody(moon) + AccelHarmonic(moon, 20, 20)
aSolRad   = AccelSolRad(**sat_kwargs)
aEarthRad = AccelEarthRad(**sat_kwargs)
accel     = aEarth + aMoon + aSun + aSolRad + aEarthRad

prop = SciPyPropagator(accel)

# ------------------------------------------------------------
# Time vector (Astropy Time). Pass t0 positionally for this version of ssapy.
# ------------------------------------------------------------
times = utils.get_times(duration=(2, "day"), freq=(1, "minute"), t0=t0)

# ------------------------------------------------------------
# Propagate
# ------------------------------------------------------------
r, v = rv(orbit=orbit, time=times, propagator=prop)

# ------------------------------------------------------------
# Plot: GCRF and lunar frames — save with figpath
# ------------------------------------------------------------
# GCRF
fig, ax = plotUtils.orbit_plot(r, times, frame="gcrf")
out_gcrf = figpath("tests/ssapy_orbit_gcrf")  # returns a .jpg path
save_plot(fig, save_path=out_gcrf)

# Lunar
fig, ax = plotUtils.orbit_plot(r, times, frame="lunar")
out_lunar = figpath("tests/ssapy_orbit_lunar")
save_plot(fig, save_path=out_lunar)

# ------------------------------------------------------------
# Ground track — save with figpath
# ------------------------------------------------------------
fig = plotUtils.ground_track_plot(r, times, save_path=figpath("tests/ssapy_ground_track"))

# ------------------------------------------------------------
# Lambertian reflectance (apparent magnitude) — save with figpath
# ------------------------------------------------------------
mv = compute.M_v_lambertian(r, times)

def decimal_to_datetime_label(d):
    year = int(d)
    rem = d - year
    # Leap year check in pure numpy
    is_leap = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
    days_in_year = 366 if is_leap else 365
    total_seconds = rem * days_in_year * 24 * 3600

    day = int(total_seconds // (24 * 3600))
    seconds_in_day = total_seconds % (24 * 3600)
    hour = int(seconds_in_day // 3600)
    minute = int((seconds_in_day % 3600) // 60)

    base_date = np.datetime64(f"{year}-01-01") + np.timedelta64(day, "D")
    return f"{base_date} {hour:02d}:{minute:02d}"

xticks = np.linspace(times.decimalyear[0], times.decimalyear[-1], 4)
xtick_labels = [decimal_to_datetime_label(t) for t in xticks]

plt.figure(dpi=300)
plt.plot(times.decimalyear, mv)
plt.xlabel("Date")
plt.ylabel("Lambertian Reflectance [Apparent Magnitude]")
plt.xticks(xticks, xtick_labels, rotation=0)
plt.tight_layout()

out_mv = figpath("tests/lambertian_reflectance")
plt.savefig(out_mv, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", out_mv)
