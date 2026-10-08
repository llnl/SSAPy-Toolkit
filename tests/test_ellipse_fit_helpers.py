
import numpy as np
import pytest

from ssapy_toolkit.orbital_mechanics.ellipse_fit import ellipse_fit


P1 = np.array([7000e3, 0.0, 0.0])
P2 = np.array([0.0, 8000e3, 0.0])


def test_ellipse_fit_input_modes_and_outputs(tmp_path):
    for kwargs in [
        {},
        {"a_m": 10000e3},
        {"e": 0.25},
        {"F2_m": np.array([1000e3, 1000e3, 0.0])},
        {"inc_deg": 10.0, "v_pref_m_s": np.array([0.0, 7000.0, 500.0])},
    ]:
        fit = ellipse_fit(P1, P2, n_pts=16, plot=False, **kwargs)
        assert fit["r"].shape == fit["v"].shape == (16, 3)
        np.testing.assert_allclose(fit["r"][0], P1, rtol=0, atol=2e3)
        assert np.all(np.isfinite(fit["r"][-1]))
        assert fit["a"] > 0
        assert 0 <= fit["e"] < 1
        assert fit["period"] > 0
        assert fit["rot_dir"] in {-1, 1}

    fit = ellipse_fit(P1, P2, n_pts=12, time_of_departure=100.0)
    assert fit["t_abs"] is not None
    assert fit["t_abs"][0] == pytest.approx(100.0)

    arrival_fit = ellipse_fit(P1, P2, n_pts=12, time_of_arrival=200.0)
    assert arrival_fit["t_abs"][-1] == pytest.approx(200.0)
