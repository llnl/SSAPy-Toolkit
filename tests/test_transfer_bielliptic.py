import numpy as np
import pytest
from ssapy import Orbit

import ssapy_toolkit.orbital_mechanics as orbital_mechanics
from ssapy_toolkit.constants import EARTH_MU
from ssapy_toolkit.orbital_mechanics.misc import bi_elliptic_transfer_delta_v
from ssapy_toolkit.orbital_mechanics.transfer_bielliptic import (
    transfer_bi_elliptic,
    transfer_bielliptic,
)


def test_transfer_bielliptic_matches_scalar_delta_v_helper():
    radius1 = 7000e3
    radius2 = 120000e3
    intermediate_radius = 240000e3

    result = transfer_bielliptic(radius1, radius2, rb=intermediate_radius, samples_per_arc=8)
    expected = bi_elliptic_transfer_delta_v(radius1, radius2, intermediate_radius, EARTH_MU)

    assert result["schema_version"] == "ssatk.transfer.v2"
    assert result["method"] == "transfer_bielliptic"
    assert len(result["burns"]) == 3
    np.testing.assert_allclose(result["delta_v_magnitudes"], expected[:3], rtol=1e-12)
    assert result["delta_v_total"] == pytest.approx(expected[-1])
    assert result["trajectory"]["r"].shape == (15, 3)
    assert result["trajectory"]["v"].shape == (15, 3)
    assert result["tof"] == pytest.approx(result["diagnostics"]["tof1"] + result["diagnostics"]["tof2"])
    np.testing.assert_allclose(result["trajectory"]["r"][0], [radius1, 0.0, 0.0])
    np.testing.assert_allclose(result["trajectory"]["r"][-1], [radius2, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(result["diagnostics"]["intermediate_state"]["r"], [-intermediate_radius, 0.0, 0.0])
    assert "target phasing is not solved" in " ".join(result["assumptions"])
    assert orbital_mechanics.transfer_bielliptic is transfer_bielliptic
    assert orbital_mechanics.transfer_bi_elliptic is transfer_bielliptic


def test_transfer_bielliptic_accepts_orbits_alias_and_hardware():
    radius1 = 7000e3
    radius2 = 9000e3
    intermediate_radius = 20000e3
    orbit1 = Orbit(
        r=np.array([radius1, 0.0, 0.0]),
        v=np.array([0.0, np.sqrt(EARTH_MU / radius1), 0.0]),
        t=100.0,
    )
    orbit2 = Orbit(
        r=np.array([radius2, 0.0, 0.0]),
        v=np.array([0.0, np.sqrt(EARTH_MU / radius2), 0.0]),
        t=100.0,
    )

    result = transfer_bi_elliptic(
        initial=orbit1,
        target=orbit2,
        intermediate_radius=intermediate_radius,
        samples_per_arc=3,
        burn_accel=0.2,
        isp=300.0,
    )

    assert result["initial"]["t"] == 100.0
    assert result["final"]["t"] == pytest.approx(100.0 + result["tof"])
    assert result["transfer_orbits"][0].t == 100.0
    assert result["transfer_orbits"][1].t == pytest.approx(100.0 + result["diagnostics"]["tof1"])
    assert result["diagnostics"]["intermediate_radius"] == intermediate_radius
    assert all(burn["duration"] == pytest.approx(burn["delta_v_mag"] / 0.2) for burn in result["burns"])


def test_transfer_bielliptic_validation_and_plot_save(tmp_path):
    radius1 = 7000e3
    radius2 = 9000e3

    with pytest.raises(ValueError, match="larger than both"):
        transfer_bielliptic(radius1, radius2, rb=8000e3)
    with pytest.raises(ValueError, match="exactly one"):
        transfer_bielliptic(radius1, radius2, rb=20000e3, intermediate_radius=21000e3)
    with pytest.raises(ValueError, match="samples_per_arc"):
        transfer_bielliptic(radius1, radius2, rb=20000e3, samples_per_arc=1)

    r1 = np.array([radius1, 0.0, 0.0])
    v1 = np.array([100.0, np.sqrt(EARTH_MU / radius1), 0.0])
    r2 = np.array([radius2, 0.0, 0.0])
    v2 = np.array([0.0, np.sqrt(EARTH_MU / radius2), 0.0])
    with pytest.raises(ValueError, match="circular boundary"):
        transfer_bielliptic(r1, v1, r2, v2, rb=20000e3)

    output = tmp_path / "bielliptic.png"
    plotted = transfer_bielliptic(radius1, radius2, rb=20000e3, plot=True, save=output, samples_per_arc=4)
    assert output.exists()
    assert "figure" in plotted
    plotted["figure"].clf()
