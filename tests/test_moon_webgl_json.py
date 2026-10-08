"""Regression tests for Moon WebGL orbit JSON input."""

import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).parents[1] / "ssapy_toolkit" / "plots" / "moon_webgl.py"
_SPEC = importlib.util.spec_from_file_location("moon_webgl_test_module", _MODULE_PATH)
MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(MODULE)

_BAKER_PATH = Path(__file__).parents[1] / "ssapy_toolkit" / "io" / "moon_maps.py"
_BAKER_SPEC = importlib.util.spec_from_file_location("moon_maps_test_module", _BAKER_PATH)
BAKER = importlib.util.module_from_spec(_BAKER_SPEC)
assert _BAKER_SPEC.loader is not None
_BAKER_SPEC.loader.exec_module(BAKER)


def test_bake_normals_tilt_downslope_in_east_north_frame():
    import numpy as np

    # Same grid as the LOLA source: row 0 at the north pole, columns eastward
    # from -180 degrees.  The shader reads red as east and green as north.
    height, width = 180, 360
    latitude = np.radians(np.linspace(90.0, -90.0, height))[:, None]
    longitude = np.radians(np.linspace(-180.0, 180.0, width, endpoint=False))[None, :]
    slope = 0.01
    rises_north = slope * BAKER.R_MOON_KM * latitude * np.ones((1, width))
    rises_east = slope * BAKER.R_MOON_KM * np.cos(latitude) * longitude

    mid_latitudes = slice(30, 151)
    north = BAKER.bake_normals(rises_north)[mid_latitudes]
    east = BAKER.bake_normals(rises_east)[mid_latitudes]

    # Ground rising northward faces south; ground rising eastward faces west.
    assert np.all(north[..., 1] < 0.0)
    assert np.all(east[..., 0] < 0.0)
    np.testing.assert_allclose(north[..., 1], -slope, rtol=0.02)
    np.testing.assert_allclose(east[..., 0], -slope, rtol=0.02)
    np.testing.assert_allclose(north[..., 0], 0.0, atol=1e-12)
