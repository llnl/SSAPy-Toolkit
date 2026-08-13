from types import SimpleNamespace

import numpy as np
import pytest

from ssapy_toolkit.constants import EARTH_MU
from ssapy_toolkit.orbital_mechanics.ellipse_from_rv import ellipse_from_rv
from ssapy_toolkit.orbital_mechanics.equally_spaced_ta import equally_spaced_ta


def test_ellipse_from_rv_elliptic_output_and_absolute_time():
    radius = 7_000_000.0
    r = np.array([radius, 0.0, 0.0])
    v = np.array([0.0, np.sqrt(EARTH_MU / radius), 0.0])
    result = ellipse_from_rv(r, v, num=12, t0=100.0)

    assert result["r"].shape == (12, 3)
    assert result["v"].shape == (12, 3)
    assert result["t_rel"].shape == (12,)
    assert result["t_abs"].shape == (12,)
    assert 0.0 <= result["e"] < 1.0
    assert result["period"] is not None
    assert result["mean_motion"] == pytest.approx(2.0 * np.pi / result["period"])
    assert result["ra_alt"] is not None
    assert len(result["plane_basis"]) == 3
    assert result["rot_dir"] == 1


def test_ellipse_from_rv_hyperbolic_and_retrograde_branches():
    r = np.array([7_000_000.0, 0.0, 0.0])
    v = np.array([0.0, -11_500.0, -500.0])
    result = ellipse_from_rv(r, v, num=9, f_span=0.2)

    assert result["r"].shape == (9, 3)
    assert result["e"] > 1.0
    assert result["period"] is None
    assert result["ra"] is None
    assert result["eta"] is None
    assert result["rot_dir"] == -1
    assert np.max(np.linalg.norm(result["r"], axis=1)) <= 2.0 * np.linalg.norm(r) * (1.0 + 1e-12)


def test_ellipse_from_rv_parabolic_and_validation_branches():
    r = np.array([7_000_000.0, 0.0, 0.0])
    v_escape = np.array([0.0, np.sqrt(2.0 * EARTH_MU / np.linalg.norm(r)), 0.0])
    with pytest.warns(RuntimeWarning):
        result = ellipse_from_rv(r, v_escape, num=5)
    assert result["period"] is None
    assert result["ra"] is None
    assert result["rp"] == pytest.approx(result["p"] / 2.0)

    with pytest.raises(ValueError, match="length-3"):
        ellipse_from_rv([1.0, 2.0], [0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="Angular-momentum"):
        ellipse_from_rv([1.0, 0.0, 0.0], [2.0, 0.0, 0.0])


def test_equally_spaced_true_anomaly_modes_and_errors():
    radians = equally_spaced_ta(6, a=10.0, e=0.2, n_dense=200)
    assert radians.shape == (6,)
    assert radians[0] == 0.0
    assert np.any(np.isclose(radians, np.pi))

    degrees = equally_spaced_ta(4, rp=8.0, ra=12.0, n_dense=200, degrees=True)
    assert degrees[0] == 0.0
    assert np.any(np.isclose(degrees, 180.0))

    orbit_values = equally_spaced_ta(4, orbit=SimpleNamespace(a=np.array([10.0]), e=np.array([0.1])), n_dense=200)
    assert orbit_values.shape == (4,)

    with pytest.raises(ValueError, match="positive"):
        equally_spaced_ta(0, a=1.0, e=0.0)
    with pytest.raises(ValueError, match="even"):
        equally_spaced_ta(3, a=1.0, e=0.0)
    with pytest.raises(ValueError, match="Provide"):
        equally_spaced_ta(2)
    with pytest.raises(ValueError, match="rp and ra"):
        equally_spaced_ta(2, rp=-1.0, ra=2.0)
    with pytest.raises(ValueError, match="elliptical"):
        equally_spaced_ta(2, a=1.0, e=1.0)
    with pytest.raises(ValueError, match="Semi-major"):
        equally_spaced_ta(2, a=-1.0, e=0.0)
