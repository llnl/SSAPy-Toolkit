import numpy as np
import pytest
import runpy
from pathlib import Path
from astropy.time import Time

from ssapy_toolkit.time_functions.absolute_times_to_relative import time_abs_to_rel
from ssapy_toolkit.time_functions.relative_times_to_absolute import time_rel_to_abs
from ssapy_toolkit.coordinates.llh_to_gcrf import llh_to_gcrf
from ssapy_toolkit.yastropy.astropy_gcrf_to_llh import astropy_gcrf_to_llh
from ssapy_toolkit.yastropy.astropy_llh_to_gcrf import astropy_llh_to_gcrf
from ssapy_toolkit.yastropy.astropy_surface_rv import astropy_surface_rv


def test_absolute_relative_time_roundtrips_and_validation():
    rel = np.array([10.0, 12.5, 20.0])
    start_ref = "2025-01-01T00:00:00"
    abs_start = time_rel_to_abs(rel, start_ref, anchor="start")

    np.testing.assert_allclose(time_abs_to_rel(abs_start, anchor="start"), [0.0, 2.5, 10.0])

    end_ref = Time("2025-01-01T12:00:00", scale="utc")
    abs_end = time_rel_to_abs(rel, end_ref, anchor="end")
    np.testing.assert_allclose(time_abs_to_rel(abs_end, anchor="end"), [-10.0, -7.5, 0.0])

    assert len(time_rel_to_abs([], start_ref)) == 0
    np.testing.assert_allclose(time_abs_to_rel([]), np.array([]))
    np.testing.assert_allclose(time_abs_to_rel([abs_start[0].gps, abs_start[-1].gps]), [0.0, 10.0])

    with pytest.raises(ValueError, match="anchor"):
        time_abs_to_rel(abs_start, anchor="middle")
    with pytest.raises(ValueError, match="anchor"):
        time_rel_to_abs(rel, start_ref, anchor="middle")
    with pytest.raises(TypeError, match="ref"):
        time_rel_to_abs(rel, object())


def test_astropy_llh_gcrf_roundtrip_and_surface_velocity():
    time = Time("2025-01-01T00:00:00", scale="utc")
    r_gcrf = astropy_llh_to_gcrf(lon=[0.0], lat=[0.0], alt=0.0, t=time)

    assert r_gcrf.shape == (1, 3)
    lon, lat, alt = astropy_gcrf_to_llh(r_gcrf, time)
    np.testing.assert_allclose(lon, [0.0], atol=1e-9)
    np.testing.assert_allclose(lat, [0.0], atol=1e-9)
    np.testing.assert_allclose(alt, [0.0], atol=1e-6)

    r_surface, v_surface = astropy_surface_rv(lon=0.0, lat=0.0, elevation=0.0, t=time)
    assert r_surface.shape == (3,)
    assert v_surface.shape == (3,)
    assert np.isclose(np.linalg.norm(r_surface), 6_378_137.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(v_surface, np.cross([0.0, 0.0, 7.2921150e-5], r_surface))


def test_llh_to_gcrf_returns_position_and_velocity():
    r_gcrf, v_gcrf = llh_to_gcrf(lon=0.0, lat=0.0, height=0.0, t=Time("2025-01-01T00:00:00", scale="utc"))

    assert r_gcrf.shape == (3,)
    assert v_gcrf.shape == (3,)
    assert np.isfinite(r_gcrf).all()
    assert np.isfinite(v_gcrf).all()


def test_astropy_roundtrip_script_entrypoint(capsys):
    script = Path(__import__("ssapy_toolkit.yastropy.astropy_test", fromlist=["__file__"]).__file__)
    runpy.run_path(script, run_name="__main__")
    out = capsys.readouterr().out
    assert "GCRF <-> Lat/Lon Conversion Test" in out
    assert "Round-trip conversion successful" in out
