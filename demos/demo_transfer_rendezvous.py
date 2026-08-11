import os
import sys
import time
from pathlib import Path

import numpy as np
from ssapy import Orbit
from astropy.time import Time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssapy_toolkit.orbital_mechanics.transfer_rendezvous import transfer_rendezvous
from ssapy_toolkit.constants import RGEO  # [38]
from ssapy_toolkit.plots import figsave

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
FIGDIR = "demo_gallery/figures"


def main(make_figures=None, fast=UNDER_PYTEST):
    if make_figures is None:
        make_figures = not UNDER_PYTEST

    t = Time("2025-01-01T00:00:00", scale="utc")

    orbit1 = Orbit.fromKeplerianElements(a=RGEO, e=0.001, i=0, pa=0, raan=0, trueAnomaly=0, t=t)
    orbit2 = Orbit.fromKeplerianElements(a=RGEO, e=0.001, i=0, pa=0, raan=0, trueAnomaly=np.radians(5), t=t)

    if fast:
        options = {
            "max_iter": 1,
            "time_step": 600,
            "max_duration": 6 * 3600,
            "final_duration": 6 * 3600,
            "bounds": [(-1000, 1000)] * 3,
            "popsize": 3,
            "polish": False,
            "seed": 0,
            "status": False,
        }
    else:
        options = {
            "max_iter": 20,
            "time_step": 300,
            "max_duration": 12 * 3600,
            "final_duration": 12 * 3600,
            "bounds": [(-1000, 1000)] * 3,
            "popsize": 6,
            "polish": False,
            "seed": 0,
            "status": False,
        }

    print("Running transfer_rendezvous...")
    start_time = time.time()
    result = transfer_rendezvous(orbit1, orbit2, plot=make_figures, **options)
    elapsed = time.time() - start_time

    print(f"\ntransfer_rendezvous completed in {elapsed:.2f} seconds")
    print(f"Initial Δv magnitude: {result['|delta_v1|']:.3f} m/s")
    print(f"Final Δv magnitude: {result['|delta_v2|']:.3f} m/s")
    print(f"Time of flight: {result['tof'] / 60:.2f} minutes")
    print(f"Final position error: {result['error']:.3f} m")

    if make_figures and "fig" in result:
        figsave(result["fig"], f"{FIGDIR}/demo_transfer_rendezvous.jpg")

    return result


if __name__ == "__main__":
    main(make_figures=True)
