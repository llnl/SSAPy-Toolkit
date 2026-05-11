import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from ssapy import *
from ssapy import constants, compute, plotUtils, utils
from ssapy.accel import AccelKepler, AccelEarthRad, AccelSolRad
from ssapy.gravity import AccelHarmonic, AccelThirdBody
from ssapy.propagator import SciPyPropagator

from ssapy_toolkit.Plots.plotutils import yufig

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
    """
    Run the SSAPy orbit/plot demo.

    Parameters
    ----------
    make_figures : bool or None
        If None, defaults to False under pytest and True otherwise.
    fast : bool or None
        If None, defaults to True under pytest and False otherwise.

    Returns
    -------
    dict
        Computed orbit states and Lambertian reflectance.
    """
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    # Set an initial astropy time object
    t0 = Time("2024-1-1")
    print(t0)

    # Get the position of the Moon
    r_moon = get_body("moon").position(t0).T
    r_moon_minus = get_body("moon").position(t0 - 1 * u.s).T
    r_moon_plus = get_body("moon").position(t0 + 1 * u.s).T
    v_moon = (r_moon_plus - r_moon_minus) / 2.0
    print(r_moon, v_moon)

    # Get a starting position and velocity (statevector) for an orbit.
    r0 = r_moon[0] + (1000e3 * r_moon[0] / np.linalg.norm(r_moon[0]))
    v0 = v_moon[0] + 100
    print(r0, v0)

    print("\nCalculating orbit.")

    # Initialize an orbit object.
    a = constants.RGEO
    e = 0
    i = np.radians(45)
    pa = np.radians(0)
    raan = np.radians(0)
    ta = np.radians(180)

    kElements = [a, e, i, pa, raan, ta]
    orbit = Orbit.fromKeplerianElements(*kElements, t=t0)

    # Set parameters of the satellite
    sat_kwargs = dict(
        mass=100,
        area=1,
        CD=2.3,
        CR=1.3,
    )

    # Build a propagator and set custom accelerations
    moon = get_body("moon")
    sun = get_body("Sun")
    Mercury = get_body("Mercury")
    Venus = get_body("Venus")
    Earth = get_body("Earth", model="EGM2008")
    Mars = get_body("Mars")
    Jupiter = get_body("Jupiter")
    Saturn = get_body("Saturn")
    Uranus = get_body("Uranus")
    Neptune = get_body("Neptune")

    aEarth = AccelKepler() + AccelHarmonic(Earth, 140, 140)
    aSun = AccelThirdBody(sun)
    aMoon = AccelThirdBody(moon) + AccelHarmonic(moon, 20, 20)
    aSolRad = AccelSolRad(**sat_kwargs)
    aEarthRad = AccelEarthRad(**sat_kwargs)
    accel = aEarth + aMoon + aSun + aSolRad + aEarthRad
    prop = SciPyPropagator(accel)

    # Build a time array to evaluate the orbit at
    duration = (6, "hour") if fast else (2, "day")
    freq = (10, "minute") if fast else (1, "minute")
    times = utils.get_times(duration=duration, freq=freq, t0=t0)
    r, v = rv(orbit=orbit, time=times, propagator=prop)

    # Plot outputs only in demo mode
    if make_figures:
        plotUtils.orbit_plot(r, times, frame="gcrf", show=False)
        plotUtils.orbit_plot(r, times, frame="lunar", show=False)

        # Lets see a ground track of the orbit
        plotUtils.ground_track_plot(r, times)

    # Calculate the Lambertian Reflectance of the orbit
    mv = compute.M_v_lambertian(r, times)

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