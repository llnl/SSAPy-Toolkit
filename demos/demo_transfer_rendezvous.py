import os
import sys
import time
import numpy as np
from ssapy import Orbit
from astropy.time import Time

from ssapy_toolkit.orbital_mechanics.transfer_rendezvous import transfer_rendezvous
from ssapy_toolkit.constants import RGEO  # [38]

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, fast=UNDER_PYTEST):
    if make_figures is None:
        make_figures = not UNDER_PYTEST

    t = Time("2025-01-01T00:00:00", scale="utc")

    orbit1 = Orbit.fromKeplerianElements(a=RGEO, e=0.5, i=np.radians(0), pa=0, raan=0, trueAnomaly=0, t=t)
    orbit2 = Orbit.fromKeplerianElements(a=2 * RGEO, e=0, i=np.radians(80), pa=0, raan=0, trueAnomaly=np.radians(50), t=t)

    options = {}
    if fast:
        options = {
            "max_iter": 1,
            "time_step": 900,
            "max_duration": 3 * 3600,
            "final_duration": 3 * 3600,
            "bounds": [(-500, 500)] * 3,
            "popsize": 2,
            "polish": False,
            "seed": 0,
            "status": False,
        }
    else:
        options = {"status": True}

    print("Running transfer_rendezvous...")
    start_time = time.time()
    result = transfer_rendezvous(orbit1, orbit2, plot=make_figures, **options)
    elapsed = time.time() - start_time

    print(f"\ntransfer_rendezvous completed in {elapsed:.2f} seconds")
    print(f"Initial Δv magnitude: {result['|delta_v1|']:.3f} m/s")
    print(f"Final Δv magnitude: {result['|delta_v2|']:.3f} m/s")
    print(f"Time of flight: {result['tof'] / 60:.2f} minutes")
    print(f"Final position error: {result['error']:.3f} m")

    return result


if __name__ == "__main__":
    main(make_figures=True)
