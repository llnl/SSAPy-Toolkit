import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ssapy import Orbit, Time
from ssapy.constants import RGEO

from ssapy_toolkit.time_functions.convert_to_gps import to_gps
from ssapy_toolkit.plots.orbit_plot import orbit_plot
from ssapy_toolkit.orbital_mechanics.burn_to_deltav import burn_to_deltav
from ssapy_toolkit.orbital_mechanics.deltav_to_burn import deltav_to_burn
from ssapy_toolkit.plots.figpath import figpath
from ssapy_toolkit.time_functions.get_times import get_times


UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    print("Modules imported.")

    burn_accel_ntw = np.array([50.0, 0.0, 0.0])  # NTW acceleration [m/s^2]
    t0 = Time("2025-01-01T12:00:00.000", scale="utc")

    # Build a global time array mainly for the first demo section
    times = to_gps(get_times(duration=(2 if fast else 12, "hour"), freq=(1, "s"), t0=t0))

    t1 = 1000 if fast else 30000
    t2 = t1 + (20 if fast else 100)
    burn_times = times[t1:t2]

    a = RGEO
    e = 0.0
    i = 0.0
    pa = 0.0
    raan = 0.0
    trueAnomaly = 0.0
    orbit = Orbit.fromKeplerianElements(a, e, i, pa, raan, trueAnomaly, t0)

    # --- Part 1: continuous burn (acceleration) vs impulsive approximation ---
    res1 = burn_to_deltav(orbit, burn_times, burn_accel_ntw)
    print("burn_to_deltav keys:", list(res1.keys()))

    # --- Part 2: equivalent delta-v over the same window for deltav_to_burn ---
    # In fast mode, use a guaranteed uniform local grid to avoid leapfrog Δt errors.
    if fast:
        burn_times_uniform = np.arange(0.0, 20.0, 1.0)
    else:
        burn_times_uniform = burn_times

    duration = float(burn_times_uniform[-1] - burn_times_uniform[0])
    dv_ntw = burn_accel_ntw * duration
    res2 = deltav_to_burn(orbit, burn_times_uniform, dv_ntw)
    print("deltav_to_burn keys:", list(res2.keys()))

    if make_figures:
        # Plot Part 1
        out1 = Path(figpath("demo_gallery/figures/burn_to_deltav_orbit_plot"))
        if out1.suffix == "":
            out1 = out1.with_suffix(".png")
        out1.parent.mkdir(parents=True, exist_ok=True)
        orbit_plot(
            [res1["r_continuous"], res1["r_instantaneous"]],
            burn_times,
            show=False,
            save_path=str(out1),
        )
        print("Saved:", out1)

        plt.figure()
        plt.plot(res1["r_continuous"][:, 0] / 1e3, res1["r_continuous"][:, 1] / 1e3, label="Burn (continuous)")
        plt.plot(res1["r_instantaneous"][:, 0] / 1e3, res1["r_instantaneous"][:, 1] / 1e3, label="Impulse approx")
        plt.xlabel("x [km]")
        plt.ylabel("y [km]")
        plt.legend()
        plt.title("burn_to_deltav: XY trajectories")
        out2 = Path(figpath("demo_gallery/figures/burn_to_deltav_xy"))
        if out2.suffix == "":
            out2 = out2.with_suffix(".png")
        out2.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out2, dpi=200, bbox_inches="tight")
        print("Saved:", out2)
        plt.close()

        plt.figure()
        sep_km = np.linalg.norm(res1["r_continuous"] - res1["r_instantaneous"], axis=-1) / 1e3
        plt.plot(burn_times - burn_times[0], sep_km)
        plt.xlabel("Seconds since burn start [s]")
        plt.ylabel("Distance between trajectories [km]")
        plt.title("burn_to_deltav: separation during burn window")
        out3 = Path(figpath("demo_gallery/figures/burn_to_deltav_separation"))
        if out3.suffix == "":
            out3 = out3.with_suffix(".png")
        out3.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out3, dpi=200, bbox_inches="tight")
        print("Saved:", out3)
        plt.close()

        # Plot Part 2
        out4 = Path(figpath("demo_gallery/figures/deltav_to_burn_orbit_plot"))
        if out4.suffix == "":
            out4 = out4.with_suffix(".png")
        out4.parent.mkdir(parents=True, exist_ok=True)
        orbit_plot(
            [res2["r_continuous"], res2["r_instantaneous"]],
            burn_times_uniform,
            show=False,
            save_path=str(out4),
        )
        print("Saved:", out4)

        plt.figure()
        plt.plot(res2["r_continuous"][:, 0] / 1e3, res2["r_continuous"][:, 1] / 1e3, label="Burn (continuous)")
        plt.plot(res2["r_instantaneous"][:, 0] / 1e3, res2["r_instantaneous"][:, 1] / 1e3, label="Impulse approx")
        plt.xlabel("x [km]")
        plt.ylabel("y [km]")
        plt.legend()
        plt.title("deltav_to_burn: XY trajectories")
        out5 = Path(figpath("demo_gallery/figures/deltav_to_burn_xy"))
        if out5.suffix == "":
            out5 = out5.with_suffix(".png")
        out5.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out5, dpi=200, bbox_inches="tight")
        print("Saved:", out5)
        plt.close()

        plt.figure()
        sep2_km = np.linalg.norm(res2["r_continuous"] - res2["r_instantaneous"], axis=-1) / 1e3
        plt.plot(burn_times_uniform - burn_times_uniform[0], sep2_km)
        plt.xlabel("Seconds since burn start [s]")
        plt.ylabel("Distance between trajectories [km]")
        plt.title("deltav_to_burn: separation during burn window")
        out6 = Path(figpath("demo_gallery/figures/deltav_to_burn_separation"))
        if out6.suffix == "":
            out6 = out6.with_suffix(".png")
        out6.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out6, dpi=200, bbox_inches="tight")
        print("Saved:", out6)
        plt.close()

    return {
        "burn_to_deltav": res1,
        "deltav_to_burn": res2,
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)