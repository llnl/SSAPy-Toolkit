"""
Regression tests for the GCRF <-> ITRF pair.

Two kinds, because they catch different things.

The ROUND TRIP catches sign errors, transposes and ordering mistakes --
the class of bug that made these two disagree by 3714 km while each looked
correct alone. It cannot catch anything the two directions share: if both
used the wrong time scale, or both omitted polar motion, the round trip
would still close to nanometres.

The ASTROPY comparison is the complement. It is an independent
implementation, so it catches errors common to both directions. This is
the same reference Table 1 already uses for star positions.

Angular sensitivity is why the lunar-distance case is here: one arcsecond
is 31 m at Earth's surface but 1.8 km at the Moon, which is the regime the
eclipse and lunar work occupies.
"""

import numpy as np
import pytest

from ssapy_toolkit.coordinates.earth_fixed import gcrf_to_itrf, itrf_to_gcrf
from ssapy_toolkit.time_functions import to_gps

R_EARTH_EQ_M = 6_378_137.0
R_EARTH_POL_M = 6_356_752.314245
R_GEO_M = 42_164_000.0
R_MOON_M = 384_400_000.0

# 2024-04-08, the total solar eclipse; J2000; and two spread epochs so a
# failure tied to a particular nutation phase cannot hide.
JD_EPOCHS = [2_460_409.0, 2_460_409.37, 2_451_545.0, 2_455_197.5]


def _gps(jd, n):
    from astropy.time import Time as ATime
    t = ATime(np.full(n, float(jd)), format="jd", scale="utc")
    gps0 = ATime("1980-01-06T00:00:00", scale="utc")
    return np.asarray((t - gps0).sec, dtype=float)


def _sample_points():
    """Axis-aligned, off-axis, and out to lunar distance."""
    return np.array([
        [R_EARTH_EQ_M, 0.0, 0.0],
        [0.0, R_EARTH_EQ_M, 0.0],
        [0.0, 0.0, R_EARTH_POL_M],
        [4.0e6, -3.0e6, 5.0e6],                 # off-axis, mixes components
        [R_GEO_M * 0.6, R_GEO_M * 0.8, 0.0],    # GEO
        [-2.0e8, 3.1e8, -9.0e7],                # roughly lunar distance
    ])


@pytest.mark.parametrize("jd", JD_EPOCHS)
def test_round_trip_closes(jd):
    """GCRF -> ITRF -> GCRF must return the input to well under a metre."""
    pts = _sample_points()
    gps = _gps(jd, len(pts))

    back = itrf_to_gcrf(gcrf_to_itrf(pts, gps), gps)
    err = np.linalg.norm(back - pts, axis=1)

    # The eclipse runtime health gate uses 1e-3 km (1 m). Hold to 1 cm:
    # still 100x inside the gate, but above float64 accumulation through
    # the einsum chain, which reaches ~1 mm on a lunar-distance vector
    # (3e-15 relative). A tighter bound tests numpy, not the transform.
    assert err.max() < 1.0e-2, (
        f"round trip left {err.max():.3f} m at JD {jd}. A residual near "
        f"twice the Earth rotation angle means the rotation is applied in "
        f"the same sense twice instead of cancelling."
    )


@pytest.mark.parametrize("jd", JD_EPOCHS)
def test_round_trip_other_direction(jd):
    """ITRF -> GCRF -> ITRF must close too; one direction passing is not enough."""
    pts = _sample_points()
    gps = _gps(jd, len(pts))

    back = gcrf_to_itrf(itrf_to_gcrf(pts, gps), gps)
    assert np.linalg.norm(back - pts, axis=1).max() < 1.0e-2


def test_forward_matches_astropy():
    """
    Absolute check against an independent implementation.

    Tolerance is angular, not absolute: this pair uses IAU 1976/1980
    precession-nutation with an approximate polar motion matrix, while
    astropy uses IAU 2006/2000A, so agreement is expected at the
    arcsecond level rather than exactly.
    """
    astropy_u = pytest.importorskip("astropy.units")
    from astropy.coordinates import GCRS, ITRS, SkyCoord
    from astropy.time import Time as ATime

    pts = _sample_points()
    jd = JD_EPOCHS[0]
    gps = _gps(jd, len(pts))
    t = ATime(np.full(len(pts), jd), format="jd", scale="utc")

    got = gcrf_to_itrf(pts, gps)

    sc = SkyCoord(x=pts[:, 0] * astropy_u.m, y=pts[:, 1] * astropy_u.m,
                  z=pts[:, 2] * astropy_u.m,
                  representation_type="cartesian", frame=GCRS(obstime=t))
    c = sc.transform_to(ITRS(obstime=t)).cartesian
    ref = np.stack([c.x.to_value(astropy_u.m), c.y.to_value(astropy_u.m),
                    c.z.to_value(astropy_u.m)], axis=-1)

    radius = np.linalg.norm(pts, axis=1)
    ang_arcsec = np.degrees(np.linalg.norm(got - ref, axis=1) / radius) * 3600.0
    assert ang_arcsec.max() < 5.0, (
        f"forward transform differs from astropy by {ang_arcsec.max():.2f} "
        f"arcsec; model differences alone should stay near 1 arcsec"
    )


def test_inverse_matches_astropy():
    """Same absolute check on the inverse direction."""
    astropy_u = pytest.importorskip("astropy.units")
    from astropy.coordinates import GCRS, ITRS, SkyCoord
    from astropy.time import Time as ATime

    pts = _sample_points()
    jd = JD_EPOCHS[0]
    gps = _gps(jd, len(pts))
    t = ATime(np.full(len(pts), jd), format="jd", scale="utc")

    got = itrf_to_gcrf(pts, gps)

    sc = SkyCoord(x=pts[:, 0] * astropy_u.m, y=pts[:, 1] * astropy_u.m,
                  z=pts[:, 2] * astropy_u.m,
                  representation_type="cartesian", frame=ITRS(obstime=t))
    c = sc.transform_to(GCRS(obstime=t)).cartesian
    ref = np.stack([c.x.to_value(astropy_u.m), c.y.to_value(astropy_u.m),
                    c.z.to_value(astropy_u.m)], axis=-1)

    radius = np.linalg.norm(pts, axis=1)
    ang_arcsec = np.degrees(np.linalg.norm(got - ref, axis=1) / radius) * 3600.0
    assert ang_arcsec.max() < 5.0


def test_transform_preserves_magnitude():
    """
    A frame rotation cannot change |r|.

    Worth asserting explicitly: the previous implementation delegated to
    ssapy.groundTrack, and a ground track is a point on the surface, so a
    silent projection to Earth's radius would be caught here rather than
    showing up as a puzzling result much later.
    """
    pts = _sample_points()
    gps = _gps(JD_EPOCHS[0], len(pts))

    for fn in (gcrf_to_itrf, itrf_to_gcrf):
        out = fn(pts, gps)
        rel = np.abs(np.linalg.norm(out, axis=1) / np.linalg.norm(pts, axis=1) - 1.0)
        # a silent projection to Earth's radius would show as ~0.85 here,
        # so 1e-10 catches it with room for rounding
        assert rel.max() < 1.0e-10, f"{fn.__name__} changed |r| by {rel.max():.2e}"