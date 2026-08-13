import os
import sys

import numpy as np

from ssapy_toolkit.plots.orbit_plot_xy import orbit_plot_xy
from ssapy_toolkit.integrators.leap_frog import leapfrog
from ssapy_toolkit.accelerations.accel_uniform_earth import accel_uniform_earth
from ssapy_toolkit.plots.figpath import figpath
from ssapy_toolkit.constants import RGEO, VGEO  # inferred from context [21]

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    t = np.arange(0, 3600 * (6 if fast else 24))

    r1, v1 = leapfrog(r0=[RGEO, 0, 0], v0=[0, VGEO, 0], t=t)
    r2, v2 = leapfrog(r0=[RGEO, 0, 0], v0=[0, VGEO, 0], t=t, velocity=(0, 600, -1))

    if make_figures:
        orbit_plot_xy(
            r1,
            save_path=figpath("demo_gallery/figures/testing_leapfrog_RGEO.jpg"),
            pad=0.1,
            title="GEO",
            show=False,
        )
        orbit_plot_xy(
            r2,
            save_path=figpath("demo_gallery/figures/testing_leapfrog_RGEO_velocity_burn.jpg"),
            pad=0.1,
            title="GEO",
            show=False,
        )

    return {"nominal": (r1, v1), "velocity_burn": (r2, v2)}


if __name__ == "__main__":
    main(make_figures=True, fast=False)