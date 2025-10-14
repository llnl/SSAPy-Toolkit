import numpy as np

from yeager_utils import RGEO, Time, ellipse_fit, figpath, groundtrack_dashboard, groundtrack_plot, groundtrack_video

from ssapy import Orbit, rv, AccelKepler
from ssapy.propagator import KeplerianPropagator, RK78Propagator

t0 = Time("2025-1-15")
r1 = [0, 3 * RGEO, 0]
r2 = [0, -RGEO / 6, -RGEO]

transfer = ellipse_fit(P1=r1, P2=r2, time_of_departure=t0)

orbit = Orbit(r=transfer['r'][0], v=transfer['v'][0], t=t0)
r_ssapy, v_ssapy = rv(orbit=orbit, time=transfer['t_abs'], propagator=RK78Propagator(AccelKepler(), h=1))
r_ssapy, v_ssapy = rv(orbit=orbit, time=transfer['t_abs'], propagator=KeplerianPropagator())


# PLOTTING
import matplotlib.pyplot as plt
r_tf = np.asarray(transfer['r'])              # (N, 3)
t_abs = np.asarray(transfer['t_abs'])         # (N,) epoch seconds
r_ssa = np.asarray(r_ssapy)                   # (N, 3)

# Position differences and norm
dr = r_ssa - r_tf                             # (N, 3)
dr_norm_km = np.linalg.norm(dr, axis=1) / 1000.0

# Time since departure (hours) for the x-axis
t_hours = (t_abs - t_abs[0]) / 3600.0

# ---- Single plot (no subplots, no explicit colors) ----
plt.figure()
plt.plot(t_hours, dr_norm_km)
plt.xlabel("Time since departure [hours]")
plt.ylabel("Position difference |Δr| [km]")
plt.title("ellipse_fit transfer vs ssapy propagation: |Δr|(t)")

# Save using yeager_utils.figpath
out_path = figpath("test_transfer_vs_ssapy_diff.jpg")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

groundtrack_plot(r=[r_ssa, r_tf], t=t_abs, save_path=figpath("test_transfer_vs_ssapy_diff_groundtrack.jpg"))
# groundtrack_video(r=[r_ssa, r_tf], t=t_abs, save_path=figpath("test_transfer_vs_ssapy_diff_groundtrack.mp4"))
