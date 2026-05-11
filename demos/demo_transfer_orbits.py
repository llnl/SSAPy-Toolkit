#!/usr/bin/env python3

import os
import sys

from ssapy import Orbit
from astropy.time import Time

from ssapy_toolkit.constants import EARTH_RADIUS, RGEO
from ssapy_toolkit.Orbital_Mechanics.transfer_shooter import transfer_shooter
from ssapy_toolkit.Orbital_Mechanics.transfer_hohmann import transfer_hohmann
from ssapy_toolkit.Orbital_Mechanics.transfer_lambertian import transfer_lambertian
from ssapy_toolkit.Orbital_Mechanics.transfer_coplanar import transfer_coplanar
from ssapy_toolkit.Plots.plotutils import yufig  # [37]

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST

    t0 = Time("2025-01-01T00:00:00", scale="utc")
    orbit1 = Orbit.fromKeplerianElements(a=RGEO, e=0.0, i=0.0, pa=0.0, raan=0.0, trueAnomaly=0.0, t=t0)
    orbit2 = Orbit.fromKeplerianElements(a=2 * RGEO, e=0.0, i=0.0, pa=0.0, raan=0.0, trueAnomaly=0.0, t=t0)

    outputs = {}

    print("Running shooter (r1, v1, r2)")
    result = transfer_shooter(orbit1.r, orbit1.v, orbit2.r, plot=make_figures, status=True)
    outputs["shooter"] = result
    if make_figures and "fig" in result:
        yufig(result["fig"], "tests/transfers_shooter_rv")

    print("Running Lambertian (r1, v1, r2)")
    try:
        result = transfer_lambertian(orbit1.r, orbit1.v, orbit2.r, plot=make_figures)
        outputs["lambertian"] = result
        if make_figures and "fig" in result:
            yufig(result["fig"], "tests/transfers_lambertian_rv")
    except Exception as err:
        print("Lambertian (r1, v1, r2) failed:", err)
        outputs["lambertian_error"] = str(err)

    return outputs


if __name__ == "__main__":
    main(make_figures=True)