import matplotlib.pyplot as plt
import numpy as np
import pytest

from ssapy_toolkit.orbital_mechanics import orbital_comparison_stats as stats


def test_histogram_axis_helpers_handle_edges():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [2, 2])
    stats._pad_ylim(ax)
    assert ax.get_ylim()[0] < 2 < ax.get_ylim()[1]
    class DummyAxes:
        def get_ylim(self):
            return np.nan, np.nan

        def set_ylim(self, *_args):
            raise AssertionError("set_ylim should not be called for non-finite limits")

    stats._pad_ylim(DummyAxes())

    assert stats._round_down_sig1(123.0) == 100.0
    assert stats._round_up_sig1(123.0) == 200.0
    assert stats._round_down_sig1(-1.0) == -1.0
    assert np.isnan(stats._round_up_sig1(np.nan))

    stats._set_three_integer_xticks(ax, [1.2, 2.7, 9.1])
    assert len(ax.get_xticks()) == 3
    stats._set_three_integer_xticks(ax, [-1.2, 0.4, 2.2])
    assert len(ax.get_xticks()) == 3
    stats._set_three_integer_xticks(ax, [np.nan])

    assert stats._hist_step(ax, [1, 2, np.nan], bins=2, linestyle="-", linewidth=1, color="k") is not None
    assert stats._hist_step(ax, [np.nan], bins=2, linestyle="-", linewidth=1, color="k") is None
    before = len(ax.lines)
    stats._add_mean_max_vlines(ax, [1, 2, 3])
    assert len(ax.lines) == before + 2
    stats._add_mean_max_vlines(ax, [np.nan])
    plt.close(fig)


def test_rtn_stack_axes_and_numeric_reduction_helpers():
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    axes = stats._rtn_stack_axes(fig, gs[0, 0], time_unit="s", shared_ylabel="error", title_text="RTN")
    assert len(axes) == 4
    assert axes[2].get_xlabel() == "time (s)"
    plt.close(fig)

    Y = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
    env = stats._envelope_over_orbits(Y, percentiles=[0, 50, 100])
    assert set(env) == {"p0", "p50", "p100"}
    np.testing.assert_allclose(stats._nanmax_per_row(Y), [3.0, 5.0])
    np.testing.assert_allclose(stats._nanrms_per_row(Y), [np.sqrt(5.0), np.sqrt(20.5)])
    np.testing.assert_allclose(stats._nanfinal_per_row(Y), [3.0, 5.0])
    np.testing.assert_allclose(stats._nanfinal_per_row([[np.nan], [2.0]]), [np.nan, 2.0])


def test_interpolation_alignment_and_rtn_projection():
    t1 = np.array([0.0, 1.0, 2.0])
    t2 = np.array([1.0, 2.0, 3.0])
    r1 = np.column_stack([t1, 2 * t1, 3 * t1])
    r2 = np.column_stack([t2, 2 * t2, 3 * t2])
    v1 = np.tile([0.0, 1.0, 0.0], (3, 1))
    v2 = np.tile([0.0, 1.0, 0.0], (3, 1))

    interp = stats._interp_xyz_nan(t1, r1, np.array([-1.0, 0.5, 3.0]))
    assert np.isnan(interp[0]).all()
    np.testing.assert_allclose(interp[1], [0.5, 1.0, 1.5])
    assert np.isnan(interp[2]).all()

    grid, R, V = stats._align_all_to_grid(t_list=[t1, t2], r_list=[r1, r2], v_list=[v1, v2], reference=0, resample="intersection", n_resample=3)
    np.testing.assert_allclose(grid, [1.0, 1.5, 2.0])
    assert R.shape == V.shape == (2, 3, 3)

    grid, R, V = stats._align_all_to_grid(t_list=[t1, t1], r_list=[r1, r1], v_list=None, reference=0, resample=None, n_resample=3)
    np.testing.assert_array_equal(grid, t1)
    assert V is None

    grid, R, _ = stats._align_all_to_grid(t_list=[t1, t2], r_list=[r1, r2], v_list=[None, None], reference=1, resample="ref", n_resample=3)
    np.testing.assert_array_equal(grid, t2)
    grid, R, _ = stats._align_all_to_grid(t_list=[t1, t2], r_list=[r1, r2], v_list=None, reference=0, resample="union", n_resample=4)
    np.testing.assert_allclose(grid, [0.0, 1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="identical"):
        stats._align_all_to_grid(t_list=[t1, t2], r_list=[r1, r2], v_list=None, reference=0, resample=None, n_resample=3)
    with pytest.raises(ValueError, match="resample"):
        stats._align_all_to_grid(t_list=[t1, t2], r_list=[r1, r2], v_list=None, reference=0, resample="bad", n_resample=3)

    r_base = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    v_base = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    d_series = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    rtn = stats._to_rtn_series(r_base, v_base, d_series)
    assert rtn.shape == (1, 2, 3)
    np.testing.assert_allclose(rtn[0, 0], [1.0, 2.0, 3.0])


def test_normalize_inputs_variants_and_errors():
    r = np.zeros((2, 3, 3))
    v = np.ones((2, 3, 3))
    t = np.tile(np.arange(3.0), (2, 1))
    r_list, v_list, t_list = stats._normalize_inputs(r, v, t)
    assert len(r_list) == len(v_list) == len(t_list) == 2

    r_list, v_list, t_list = stats._normalize_inputs([np.zeros((3, 3))], None, None)
    assert v_list == [None]
    np.testing.assert_array_equal(t_list[0], [0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="r_list"):
        stats._normalize_inputs([np.zeros((3, 2))], None, None)
    with pytest.raises(ValueError, match="v_list"):
        stats._normalize_inputs([np.zeros((3, 3))], [np.zeros((3, 2))], None)
    with pytest.raises(ValueError, match="r/v length"):
        stats._normalize_inputs([np.zeros((3, 3))], [np.zeros((2, 3))], None)
    with pytest.raises(ValueError, match="t_list"):
        stats._normalize_inputs([np.zeros((3, 3))], None, [np.zeros((3, 1))])
    with pytest.raises(ValueError, match="t/r length"):
        stats._normalize_inputs([np.zeros((3, 3))], None, [np.zeros(2)])


def _dashboard_tracks(num_orbits=4, num_times=6):
    t = np.linspace(0.0, 50.0, num_times)
    base_r = np.column_stack((7000e3 + 20.0 * t, 1000.0 * t, 100.0 * np.sin(t / 10.0)))
    base_v = np.column_stack((20.0 * np.ones_like(t), 1000.0 * np.ones_like(t), 10.0 * np.cos(t / 10.0)))
    r_list = []
    v_list = []
    t_list = []
    for idx in range(num_orbits):
        offset = float(idx) * np.column_stack((10.0 + t, 2.0 * t, 5.0 * np.ones_like(t)))
        voffset = float(idx) * np.column_stack((0.1 * np.ones_like(t), 0.2 * np.ones_like(t), 0.05 * np.ones_like(t)))
        r_list.append(base_r + offset)
        v_list.append(base_v + voffset)
        t_list.append(t + idx)
    return r_list, v_list, t_list


def test_orbit_stats_dashboard_population_and_benchmark_plots():
    r_list, v_list, t_list = _dashboard_tracks()
    result = stats.orbit_stats_dashboard(
        r_list,
        v_list,
        t_list,
        baseline="mean",
        mode="population",
        resample="union",
        n_resample=8,
        make_plots=True,
        show_legend=True,
        hist_bins=4,
        envelope_on_log_threshold=1.05,
    )
    assert result["meta"]["mode"] == "population"
    assert result["population"]["rtn"]["dr_rtn"].shape[:2] == result["population"]["sep"].shape
    assert result["figure"] is not None
    assert len(result["figure"].axes) >= 7
    plt.close(result["figure"])

    no_v_result = stats.orbit_stats_dashboard(
        r_list,
        None,
        [np.arange(len(r)) for r in r_list],
        baseline="median",
        mode="population",
        resample=None,
        make_plots=True,
        show_legend=False,
        hist_bins=3,
    )
    assert no_v_result["population"]["vsep"] is None
    plt.close(no_v_result["figure"])

    benchmark = stats.orbit_stats_dashboard(
        r_list,
        v_list,
        t_list,
        mode="benchmark",
        reference=1,
        resample="ref",
        labels=["nominal", "model-a", "model-b", "model-c"],
        make_plots=True,
        show_legend=True,
        hist_bins=3,
        envelope_on_log_threshold=1.05,
    )
    assert benchmark["meta"]["baseline"] == "nominal"
    assert benchmark["population"]["population_mask"].sum() == 3
    plt.close(benchmark["figure"])

    benchmark_no_v = stats.orbit_stats_dashboard(
        r_list,
        None,
        [np.arange(len(r)) for r in r_list],
        mode="benchmark",
        make_plots=True,
        show_legend=False,
        hist_bins=3,
    )
    assert benchmark_no_v["population"]["rtn"] is None
    plt.close(benchmark_no_v["figure"])


def test_orbit_stats_dashboard_validation_errors():
    r_list, v_list, t_list = _dashboard_tracks(num_orbits=2)
    with pytest.raises(ValueError, match="at least two"):
        stats.orbit_stats_dashboard([r_list[0]])
    with pytest.raises(ValueError, match="labels"):
        stats.orbit_stats_dashboard(r_list, labels=["only-one"])
    with pytest.raises(ValueError, match="baseline"):
        stats.orbit_stats_dashboard(r_list, baseline="bad")
    with pytest.raises(ValueError, match="mode"):
        stats.orbit_stats_dashboard(r_list, mode="bad")
    with pytest.raises(ValueError, match="reference"):
        stats.orbit_stats_dashboard(r_list, reference=5)
    with pytest.raises(ValueError, match="No overlapping"):
        stats.orbit_stats_dashboard(r_list, v_list, [np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])], resample="intersection")
    with pytest.raises(ValueError, match="Invalid union"):
        stats._align_all_to_grid(t_list=[np.array([1.0]), np.array([1.0])], r_list=[r_list[0][:1], r_list[1][:1]], v_list=None, reference=0, resample="union", n_resample=3)
