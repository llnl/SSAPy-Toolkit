import numpy as np
import pytest
from astropy.time import Time


@pytest.mark.parametrize("utc", ["2008-09-20T12:25:40", "2026-10-08T00:00:00"])
def test_earth_texture_rotation_matches_ssapy_ground_tracks(utc):
    # R2: SSAPy's groundTrack rotates GCRF to ITRF with gst94 evaluated at UT1.
    # The plot rotation angle must equal that GST (1e-9 deg), i.e. agree with
    # astropy's apparent sidereal time to within the IAU 1994 vs 2006 model
    # difference (0.5 arcsec here). Passing TT as UT1 used to put it 17 arcmin off.
    import erfa
    from ssapy.utils import iers_interp

    from ssapy_toolkit.plots.scene_primitives import earth_rotation_deg_from_time

    t = Time(utc, scale="utc")
    mjd_tt = 44244.0 + (t.gps + 51.184) / 86400.0
    d_ut1_tt, _pmx, _pmy = iers_interp(t.gps)
    expected = np.degrees(erfa.gst94(2400000.5, mjd_tt + d_ut1_tt)) % 360.0
    angle = earth_rotation_deg_from_time(t.gps)
    assert angle == pytest.approx(float(np.squeeze(expected)), abs=1e-9)
    gast = t.sidereal_time("apparent", "greenwich").deg
    assert abs(((angle - gast + 180.0) % 360.0) - 180.0) * 3600.0 < 0.5
