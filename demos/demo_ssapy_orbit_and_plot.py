import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time

from ssapy import Orbit, rv, get_body
from ssapy import constants
from ssapy.accel import AccelKepler, AccelEarthRad, AccelSolRad
from ssapy.gravity import AccelHarmonic, AccelThirdBody
from ssapy.propagator import SciPyPropagator

from ssapy_toolkit.plots.orbit_plot import orbit_plot
from ssapy_toolkit.plots.groundtrack_plot import groundtrack_plot
from ssapy_toolkit.plots.plotutils import yufig
from ssapy_toolkit.compute.lambertian_magnitude import M_v_lambertian
from ssapy_toolkit.time_functions.get_times import get_times

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def decimal_to_datetime_label(d):
    year = int(d)
    rem = d - year
    is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    days_in_year = 366 if is_leap else 365
    total_seconds = rem * days_in_year * 24 * 3600

    day = int(total_seconds // (24 * 3600))
    seconds_in_day = total_seconds % (24 * 3600)
    hour = int(seconds_in_day // 3600)
    minute = int((seconds_in_day % 3600) // 60)

    base_date = np.datetime64(f"{year}-01-01") + np.timedelta64(day, "D")
    return f"{base_date} {hour:02d}:{minute:02d}"


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    t0 = Time("2024-1-1")
    print(t0)

    r_moon = get_body("moon").position(t0).T
    r_moon_minus = get_body("moon").position(t0 - 1 * u.s).T
    r_moon_plus = get_body("moon").position(t0 + 1 * u.s).T
    v_moon = (r_moon_plus - r_moon_minus) / 2.0
    print(r_moon, v_moon)

    r0 = r_moon[0] + (1000e3 * r_moon[0] / np.linalg.norm(r_moon[0]))
    v0 = v_moon[0] + 100
    print(r0, v0)

    print("\nCalculating orbit.")

    a = constants.RGEO
    e = 0
    i = np.radians(45)
    pa = np.radians(0)
    raan = np.radians(0)
    ta = np.radians(180)

    kElements = [a, e, i, pa, raan, ta]
    orbit = Orbit.fromKeplerianElements(*kElements, t=t0)

    sat_kwargs = dict(
        mass=100,
        area=1,
        CD=2.3,
        CR=1.3,
    )

    moon = get_body("moon")
    sun = get_body("Sun")
    Earth = get_body("Earth", model="EGM2008")

    aEarth = AccelKepler() + AccelHarmonic(Earth, 140, 140)
    aSun = AccelThirdBody(sun)
    aMoon = AccelThirdBody(moon) + AccelHarmonic(moon, 20, 20)
    aSolRad = AccelSolRad(**sat_kwargs)
    aEarthRad = AccelEarthRad(**sat_kwargs)
    accel = aEarth + aMoon + aSun + aSolRad + aEarthRad
    prop = SciPyPropagator(accel)

    duration = (6, "hour") if fast else (2, "day")
    freq = (10, "minute") if fast else (1, "minute")
    times = get_times(duration=duration, freq=freq, t0=t0)
    r, v = rv(orbit=orbit, time=times, propagator=prop)

    if make_figures:
        orbit_plot(r, times, frame="gcrf", show=False)
        orbit_plot(r, times, frame="lunar", show=False)
        groundtrack_plot(r, times)

    mv = M_v_lambertian(r, times)

    if make_figures:
        xticks = np.linspace(times.decimalyear[0], times.decimalyear[-1], 4)
        xtick_labels = [decimal_to_datetime_label(t) for t in xticks]

        fig = plt.figure(dpi=300)
        plt.plot(times.decimalyear, mv)
        plt.xlabel("Date")
        plt.ylabel("Lambertian Reflectance [Apparent Magnitude]")
        plt.xticks(xticks, xtick_labels, rotation=0)
        plt.tight_layout()
        yufig(fig, "tests/ssapy_lambertian_reflectance")
        plt.close(fig)

    return {
        "t0": t0,
        "orbit": orbit,
        "times": times,
        "r": r,
        "v": v,
        "mv": mv,
        "r_moon": r_moon,
        "v_moon": v_moon,
        "r0": r0,
        "v0": v0,
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)