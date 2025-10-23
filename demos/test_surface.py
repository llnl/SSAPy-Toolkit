#!/usr/bin/env python3
"""
Reference-frame tests for yeager_utils:
- GCRF <-> ITRF <-> LLH round-trips for real times
- Independent cross-check vs Astropy transforms
- Subsolar point sanity checks at equinoxes/solstices

Run:
  python test_frames_of_reference.py
"""

import math
import numpy as np
from astropy.time import Time
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, EarthLocation, get_sun
import astropy.units as u

from yeager_utils import (
    llh_to_gcrf,
    gcrf_to_llh,
    astropy_surface_rv,
    groundtrack_dashboard,
    EARTH_RADIUS,
    figpath,
)

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
SHOW_FIGS = False   # set True to pop up windows
SAVE_FIGS = True    # save groundtrack figures to figpath(...) prefixes
RANDOM_SEED = 42

# Tolerances
ANG_TOL_DEG = 1e-6      # ~0.0036 arcsec
H_TOL_M = 1e-3          # 1 mm height
AST_ANG_TOL_DEG = 2e-5  # looser when comparing two independent chains
AST_H_TOL_M = 0.02      # 2 cm height

# Example epochs to test
EPOCHS = [
    "1999-08-01 00:00:00",
    "2000-03-20 12:00:00",  # Mar equinox 2000-ish
    "2000-06-21 12:00:00",  # Jun solstice 2000-ish
    "2000-09-22 12:00:00",  # Sep equinox 2000-ish
    "2000-12-21 12:00:00",  # Dec solstice 2000-ish
    "2010-01-01 00:00:00",
    "2020-06-01 00:00:00",
    "2024-03-20 12:00:00",
]

# A few well-known ground sites (approx WGS84 ellipsoidal LLH)
# (lon_deg East, lat_deg North, height m)
SITES = [
    ("Greenwich Royal Observatory",       0.0000,         51.4769,       46.0),
    ("NASA DSN Goldstone (DSS-14)",     -116.7933,       35.2472,     1006.0),
    ("NASA DSN Madrid",                   -4.2481,        40.4314,      731.0),
    ("NASA DSN Canberra",                148.9819,       -35.3985,      691.0),
    ("Mauna Kea Summit",                -155.4761,        19.8206,     4205.0),
    ("Arecibo Observatory",              -66.7528,        18.3442,      498.0),
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def wrap_lon_deg(lon):
    """Wrap longitude to (-180, 180] for stable comparisons."""
    w = (lon + 180.0) % 360.0 - 180.0
    if abs(w + 180.0) < 1e-12:
        return 180.0
    return w

def ang_diff_deg(a, b):
    """Minimal angular difference in degrees, accounting for wrapping."""
    da = wrap_lon_deg(a) - wrap_lon_deg(b)
    return abs(wrap_lon_deg(da))

def print_pass(name):
    print("[PASS]", name)

def print_fail(name, msg):
    print("[FAIL]", name, "->", msg)

# -----------------------------------------------------------------------------
# Test 1: Station invariance across time (round-trip LLH via your pipeline)
# -----------------------------------------------------------------------------
def test_station_roundtrips():
    ok = True
    for label, lon, lat, h in SITES:
        for epoch in EPOCHS:
            t = Time(epoch, scale="utc")
            # Use your convenience: surface r,v at (lon,lat) and t
            r_surf, v_surf = astropy_surface_rv(lon, lat, t=t)  # GCRF r,v of the fixed site at time t
            lon2, lat2, h2 = gcrf_to_llh(r_surf, t=t)           # back to geodetic using your pipeline

            dlon = ang_diff_deg(lon2, lon)
            dlat = abs(lat2 - lat)
            dh = abs((h2 if h is None else (h2 - h))) if h is not None else abs(h2)

            if dlon < ANG_TOL_DEG and dlat < ANG_TOL_DEG:
                pass  # OK on angles; height depends on datum consistency
            else:
                ok = False
                print_fail(f"Station RT: {label} @ {epoch}",
                          "angles off: dlon=%.3e deg, dlat=%.3e deg" % (dlon, dlat))
    if ok:
        print_pass("Station round-trips across epochs")
    return ok

# -----------------------------------------------------------------------------
# Test 2: Independent cross-check vs Astropy transforms
# Start from random GCRF vectors at epochs; compare your gcrf_to_llh with:
#   GCRS(r) -> ITRS -> EarthLocation.from_geocentric -> geodetic
# -----------------------------------------------------------------------------
def test_vs_astropy_random():
    rng = np.random.default_rng(RANDOM_SEED)
    ok = True
    for epoch in EPOCHS:
        t = Time(epoch, scale="utc")

        # Random directions on sphere, radii spread near Earth surface and LEO altitudes
        for _ in range(25):
            rmag = rng.uniform(EARTH_RADIUS - 100.0, EARTH_RADIUS + 1200e3)  # from near surface to ~1200 km
            v = rng.normal(size=3); v /= np.linalg.norm(v)
            r_vec = rmag * v

            # Your path
            lon_y, lat_y, h_y = gcrf_to_llh(r_vec, t=t)

            # Astropy path (independent)
            g = GCRS(CartesianRepresentation(r_vec * u.m), obstime=t)
            itrs = g.transform_to(ITRS(obstime=t))
            x, y, z = itrs.cartesian.xyz.to_value(u.m)
            loc = EarthLocation.from_geocentric(x, y, z, unit=u.m)
            lon_a = loc.lon.to_value(u.deg)
            lat_a = loc.lat.to_value(u.deg)
            h_a = loc.height.to_value(u.m)

            # Compare
            dlon = ang_diff_deg(lon_y, lon_a)
            dlat = abs(lat_y - lat_a)
            dh = abs(h_y - h_a)
            if not (dlon < AST_ANG_TOL_DEG and dlat < AST_ANG_TOL_DEG and dh < AST_H_TOL_M):
                ok = False
                print_fail(f"Astropy xcheck @ {epoch}",
                          "dlon=%.3e deg, dlat=%.3e deg, dh=%.3e m" % (dlon, dlat, dh))
    if ok:
        print_pass("Independent cross-check vs Astropy (random GCRF vectors)")
    return ok

# -----------------------------------------------------------------------------
# Test 3: Subsolar point checks at notable dates
# Uses Astropy get_sun -> GCRS -> ITRS; the subsolar point is the
# ITRS nadir direction. Then pipe through your gcrf_to_llh for agreement.
# -----------------------------------------------------------------------------
def test_subsolar_points():
    ok = True

    # Dates ~equinox/solstice; loose latitude checks
    cases = [
        ("Mar equinox 2000-ish", "2000-03-20 12:00:00", 0.0, 0.3),
        ("Jun solstice 2000-ish", "2000-06-21 12:00:00", +23.44, 0.3),
        ("Sep equinox 2000-ish", "2000-09-22 12:00:00", 0.0, 0.3),
        ("Dec solstice 2000-ish", "2000-12-21 12:00:00", -23.44, 0.3),
    ]

    for label, epoch, lat_expect, lat_tol in cases:
        t = Time(epoch, scale="utc")

        # Sun GCRS position
        sun_gcrs = get_sun(t).transform_to(GCRS(obstime=t))
        r_sun_gcrs = sun_gcrs.cartesian.xyz.to_value(u.m)

        # Convert Sun position to ITRS, then take Earth nadir (direction to Earth center)
        sun_itrs = sun_gcrs.transform_to(ITRS(obstime=t))
        r_sun_itrs = sun_itrs.cartesian.xyz.to_value(u.m)

        # The subsolar point is the point on Earth where the Sun is at zenith:
        # direction from Earth's center toward the Sun in ITRS
        r_hat_itrs = r_sun_itrs / np.linalg.norm(r_sun_itrs)
        r_subsolar_itrs = EARTH_RADIUS * r_hat_itrs  # approximate ellipsoid as sphere for the check

        # Put that vector back into your pipeline by first mapping it to GCRS via inverse transform:
        # (Use Astropy for ITRS->GCRS of the *vector* at the same t to avoid duplicating rotation code)
        itrs_coord = ITRS(CartesianRepresentation(r_subsolar_itrs * u.m), obstime=t)
        gcrs_coord = itrs_coord.transform_to(GCRS(obstime=t))
        r_check = gcrs_coord.cartesian.xyz.to_value(u.m)

        # Your gcrf_to_llh should recover the subsolar geodetic latitude near expected
        lon_sub, lat_sub, h_sub = gcrf_to_llh(r_check, t=t)

        # Check only latitude here; longitude depends on equation-of-time nuances we are not enforcing
        if abs(lat_sub - lat_expect) <= lat_tol:
            pass
        else:
            ok = False
            print_fail(label, "lat=%.3f deg (expected %.2f ± %.2f)" % (lat_sub, lat_expect, lat_tol))

        # Optional figure
        if SAVE_FIGS:
            try:
                groundtrack_dashboard(
                    r_check,
                    t,
                    show=SHOW_FIGS,
                    save_path=figpath(f"tests/test_subsolar_{label.replace(' ', '_')}_"),
                )
            except Exception as err:
                print("  groundtrack_dashboard(subsolar) failed:", err)

    if ok:
        print_pass("Subsolar latitude checks")
    return ok

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("Running reference-frame tests...")
    ok1 = test_station_roundtrips()
    ok2 = test_vs_astropy_random()
    ok3 = test_subsolar_points()

    print("\nSummary:")
    print("  Station round-trips:", "OK" if ok1 else "FAIL")
    print("  Astropy cross-check:", "OK" if ok2 else "FAIL")
    print("  Subsolar checks:", "OK" if ok3 else "FAIL")

    if SAVE_FIGS:
        print("\nFigures saved under prefixes like:", figpath("tests/test_subsolar_..."))

if __name__ == "__main__":
    main()
