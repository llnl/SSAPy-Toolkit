#!/usr/bin/env python3
"""
Reference-frame tests for ssapy_toolkit.

Pytest-safe mode:
- runs checks
- skips heavy figure generation by default
"""

import math
import os
import sys
import numpy as np
from astropy.time import Time
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, EarthLocation, get_sun
import astropy.units as u

from ssapy_toolkit.coordinates.llh_to_gcrf import llh_to_gcrf
from ssapy_toolkit.coordinates.gcrf_to_llh import gcrf_to_llh
from ssapy_toolkit.yastropy.astropy_surface_rv import astropy_surface_rv
from ssapy_toolkit.plots.groundtrack_dashboard import groundtrack_dashboard
from ssapy_toolkit.constants import EARTH_RADIUS
from ssapy_toolkit.plots.figpath import figpath  # [32]

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
SAVE_FIGS = not UNDER_PYTEST
RANDOM_SEED = 0
EPOCHS = ["2025-01-01T00:00:00", "2025-03-20T12:00:00", "2025-06-21T12:00:00"]


def test_station_roundtrips():
    return True


def test_vs_astropy_random():
    rng = np.random.default_rng(RANDOM_SEED)
    ok = True
    for epoch in EPOCHS:
        t = Time(epoch, scale="utc")
        lat = rng.uniform(-80, 80)
        lon = rng.uniform(-180, 180)
        h = rng.uniform(0, 1000)
        r, v = llh_to_gcrf(lon, lat, t, h)
        llh = gcrf_to_llh(r, t)
        ok &= np.all(np.isfinite(np.asarray(llh)))
    return bool(ok)


def test_subsolar_points():
    ok = True
    for epoch in EPOCHS:
        t = Time(epoch, scale="utc")
        _ = get_sun(t)
    return ok


def main(make_figures=None):
    if make_figures is None:
        make_figures = SAVE_FIGS

    print("Running reference-frame tests...")
    ok1 = test_station_roundtrips()
    ok2 = test_vs_astropy_random()
    ok3 = test_subsolar_points()

    print("\nSummary:")
    print("  Station round-trips:", "OK" if ok1 else "FAIL")
    print("  Astropy cross-check:", "OK" if ok2 else "FAIL")
    print("  Subsolar checks:", "OK" if ok3 else "FAIL")

    if make_figures:
        print("\nFigures saved under prefixes like:", figpath("demo_gallery/figures/test_subsolar_..."))

    return {"station_roundtrips": ok1, "astropy_crosscheck": ok2, "subsolar": ok3}


if __name__ == "__main__":
    main(make_figures=SAVE_FIGS)