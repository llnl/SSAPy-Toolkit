import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ssapy import Orbit, Time
from ssapy.constants import RGEO
from ssapy.utils import get_times
from yeager_utils import to_gps, orbit_plot, burn_to_deltav, deltav_to_burn, figpath

print("Modules imported.")

# --- Scenario setup ---
burn_accel_ntw = np.array([50.0, 0.0, 0.0])  # NTW acceleration [m/s^2] for burn_to_deltav
t0 = Time("2025-01-01T12:00:00.000", scale="utc")
times = to_gps(get_times(duration=(12, "hour"), freq=(1, "s"), t0=t0))

t1 = 30000
t2 = t1 + 100
burn_times = times[t1:t2]

a = RGEO
e = 0.0
i = 0.0
pa = 0.0
raan = 0.0
trueAnomaly = 0.0
kElements = [a, e, i, pa, raan, trueAnomaly]
orbit = Orbit.fromKeplerianElements(*kElements, t0)

# --- Part 1: continuous burn (acceleration) vs impulsive approximation ---
res1 = burn_to_deltav(orbit, burn_times, burn_accel_ntw)
print("burn_to_deltav keys:", list(res1.keys()))

# Save orbit_plot for Part 1
plt.figure()
orbit_plot([res1['r_continuous'], res1['r_instantaneous']], burn_times, show=False)
out1 = Path(figpath("burn_to_deltav_orbit_plot"))
if out1.suffix == "":
    out1 = out1.with_suffix(".png")
out1.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out1, dpi=200, bbox_inches="tight")
print("Saved:", out1)
plt.close()

# XY trajectories
plt.figure()
plt.plot(res1["r_continuous"][:, 0] / 1e3, res1["r_continuous"][:, 1] / 1e3, label="Burn (continuous)")
plt.plot(res1["r_instantaneous"][:, 0] / 1e3, res1["r_instantaneous"][:, 1] / 1e3, label="Impulse approx")
plt.xlabel("x [km]")
plt.ylabel("y [km]")
plt.legend()
plt.title("burn_to_deltav: XY trajectories")
out2 = Path(figpath("burn_to_deltav_xy"))
if out2.suffix == "":
    out2 = out2.with_suffix(".png")
out2.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out2, dpi=200, bbox_inches="tight")
print("Saved:", out2)
plt.close()

# Separation over the burn window
plt.figure()
sep_km = np.linalg.norm(res1["r_continuous"] - res1["r_instantaneous"], axis=-1) / 1e3
plt.plot(burn_times - burn_times[0], sep_km)
plt.xlabel("Seconds since burn start [s]")
plt.ylabel("Distance between trajectories [km]")
plt.title("burn_to_deltav: separation during burn window")
out3 = Path(figpath("burn_to_deltav_separation"))
if out3.suffix == "":
    out3 = out3.with_suffix(".png")
out3.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out3, dpi=200, bbox_inches="tight")
print("Saved:", out3)
plt.close()

# --- Part 2: match an equivalent delta-v over the SAME window for deltav_to_burn ---
duration = float(burn_times[-1] - burn_times[0])
dv_ntw = burn_accel_ntw * duration  # NTW delta-v [m/s] for the same time window

res2 = deltav_to_burn(orbit, burn_times, dv_ntw)
print("deltav_to_burn keys:", list(res2.keys()))

# Save orbit_plot for Part 2
plt.figure()
orbit_plot([res2["r_continuous"], res2["r_instantaneous"]], burn_times, show=False)
out4 = Path(figpath("deltav_to_burn_orbit_plot"))
if out4.suffix == "":
    out4 = out4.with_suffix(".png")
out4.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out4, dpi=200, bbox_inches="tight")
print("Saved:", out4)
plt.close()

# XY trajectories
plt.figure()
plt.plot(res2["r_continuous"][:, 0] / 1e3, res2["r_continuous"][:, 1] / 1e3, label="Burn (continuous)")
plt.plot(res2["r_instantaneous"][:, 0] / 1e3, res2["r_instantaneous"][:, 1] / 1e3, label="Impulse approx")
plt.xlabel("x [km]")
plt.ylabel("y [km]")
plt.legend()
plt.title("deltav_to_burn: XY trajectories")
out5 = Path(figpath("deltav_to_burn_xy"))
if out5.suffix == "":
    out5 = out5.with_suffix(".png")
out5.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out5, dpi=200, bbox_inches="tight")
print("Saved:", out5)
plt.close()

# Separation over the burn window
plt.figure()
sep2_km = np.linalg.norm(res2["r_continuous"] - res2["r_instantaneous"], axis=-1) / 1e3
plt.plot(burn_times - burn_times[0], sep2_km)
plt.xlabel("Seconds since burn start [s]")
plt.ylabel("Distance between trajectories [km]")
plt.title("deltav_to_burn: separation during burn window")
out6 = Path(figpath("deltav_to_burn_separation"))
if out6.suffix == "":
    out6 = out6.with_suffix(".png")
out6.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out6, dpi=200, bbox_inches="tight")
print("Saved:", out6)
plt.close()
