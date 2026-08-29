import numpy as np

from ssapy_toolkit.constants import AU, EARTH_RADIUS
from ssapy_toolkit.plots.coverage_analysis import _pass_statistics, compute_eclipse


def test_compute_eclipse_uses_finite_sun_disk_for_partial_shadow():
    r = np.array([
        [-2.0 * EARTH_RADIUS, EARTH_RADIUS, 0.0],
        [2.0 * EARTH_RADIUS, 0.0, 0.0],
    ])
    r_sun = np.array([
        [AU, 0.0, 0.0],
        [AU, 0.0, 0.0],
    ])

    np.testing.assert_array_equal(compute_eclipse(r, r_sun), [True, False])


def test_pass_statistics_includes_window_edge_gaps():
    stats = _pass_statistics(
        np.array([False, True, False, False, True, False]),
        dt_s=60.0,
        total_s=300.0,
    )

    assert stats["n_passes"] == 2
    assert stats["max_gap_min"] == 2.0


def test_pass_statistics_single_interior_pass_has_edge_gap():
    stats = _pass_statistics(np.array([False, True, False]), dt_s=60.0, total_s=120.0)

    assert stats["n_passes"] == 1
    assert stats["max_gap_min"] == 1.0
