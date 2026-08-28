import io
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ssapy_toolkit import utils


def test_string_array_and_dataframe_helpers():
    assert utils.pd_flatten(["1", "", "2.5", "bad", "3"], factor=2.0) == [0.5, 1.25, 1.5]
    np.testing.assert_allclose(utils.str_to_array("[1,2.5,3]"), [1.0, 2.5, 3.0])

    frame = pd.Series(["[1,2,3]", "[4,5,6]"])
    converted = utils.pdstr_to_arrays(frame)
    assert converted.shape == (2,)
    np.testing.assert_allclose(converted[0], [1.0, 2.0, 3.0])

    assert utils.b2str([b"alpha", b"beta"]) == ["alpha", "beta"]
    assert utils.byte2str([b"a", b"b"]) == ["a", "b"]
    assert utils.byte2str(b"abc") == "abc"


def test_array_filtering_shape_and_size_helpers():
    assert utils.find_indices([1, 2, 3, 4], lambda item: item % 2 == 0) == [1, 3]
    assert np.isnan(utils.nan_array(3)).all()
    np.testing.assert_allclose(utils.remove_np_nans(np.array([1.0, np.nan, 2.0])), [1.0, 2.0])
    np.testing.assert_allclose(utils.remove_zeros(np.array([[1, 0], [0, 0], [2, 3]]), axis=1), [[1, 0], [2, 3]])
    np.testing.assert_allclose(utils.remove_zeros(np.array([[1, 0, 2], [3, 0, 4]])), [[1, 2], [3, 4]])

    np.testing.assert_allclose(utils.nby3shape(np.array([1.0, 2.0, 3.0])), [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(utils.nby3shape(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])), [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    np.testing.assert_allclose(utils.nby3shape(np.array([[1.0, 2.0, 3.0]])), [[1.0, 2.0, 3.0]])

    assert utils.size([[1, 2], [3, 4]]) == 4
    assert utils.size([[1, 2], [3, 4]], axis=0) == 2


def test_formatting_sorting_type_and_math_helpers(capsys):
    assert utils.eformat(1234.0, prec=2, exp_digits=3) == "1.23e+03"
    assert utils.extractNum("frame_0042.png") == 42
    assert utils.sortbynum(["frame_10.png", "frame_2.png", "frame_1.png"]) == ["frame_1.png", "frame_2.png", "frame_10.png"]
    assert utils.sortbynum(["/tmp/frame_10.png", "/tmp/frame_2.png"]) == ["/tmp/frame_2.png", "/tmp/frame_10.png"]
    assert utils.sortbynum([]) == []
    assert utils.sortbynum(["final.png", "frame_2.png"]) == ["frame_2.png", "final.png"]

    assert utils.issorted([1, 2, 3]) is True
    assert "Yes" in capsys.readouterr().out
    assert utils.issorted([2, 1]) is False
    assert "No" in capsys.readouterr().out

    assert utils.flatten([[1, 2], [3]]) == [1, 2, 3]
    assert utils.sortbylist(["b", "a", "c"], [2, 1, 3]) == ["a", "b", "c"]
    assert utils.find_nearest(np.array([0.0, 2.0, 5.0]), value=3.0) == (np.int64(1), np.float64(-1.0))
    assert utils.isint(1) and not utils.isint(1.0)
    assert utils.isfloat(1.0) and not utils.isfloat(1)
    assert utils.isstr("x") and not utils.isstr(1)
    assert utils.divby0(4.0, 2.0) == 2.0
    assert utils.divby0(4.0, 0.0, Δ=-1.0) == -1.0


def test_stdout_and_timing_helpers(capsys, monkeypatch):
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    try:
        with utils.suppress_stdout():
            print("hidden")
    finally:
        sys.stdout = old_stdout
    assert buffer.getvalue() == ""

    times = iter([30.0, 0.05, 2.0, 120.0, 7200.0])
    monkeypatch.setattr(utils, "timer", lambda: next(times))

    utils.ETA(idx=0, total_num=200, start_loop_time=0.0)
    assert "hours" in capsys.readouterr().out
    utils.elapsed_time(0.0)
    assert "ms" in capsys.readouterr().out
    utils.elapsed_time(0.0)
    assert "seconds" in capsys.readouterr().out
    utils.elapsed_time(0.0)
    assert "minutes" in capsys.readouterr().out
    utils.elapsed_time(0.0)
    assert "hours" in capsys.readouterr().out

    times = iter([1.0])
    monkeypatch.setattr(utils, "timer", lambda: next(times))
    utils.ETA(idx=9, total_num=10, start_loop_time=0.0)
    assert "minutes" in capsys.readouterr().out


def test_geometry_helpers_and_kde():
    assert utils.close_to_any(np.array([1.0, 2.0]), np.array([3.0, 2.001]), atol=0.01)
    assert utils.distance3d(0.0, 0.0, 0.0, 1.0, 2.0, 2.0) == 3.0
    assert utils.find_local_extrema(np.array([2.0, 1.0, 3.0, 0.0, 4.0])) == ([1, 3], [2])

    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    hull = utils.contours_2d(points, plot=False)
    assert hull.shape[1] == 2
    assert len(hull) >= 3
    assert utils.graham_scan(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])).shape[1] == 2

    hull3d = utils.contours_3d(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
        plot=False,
    )
    assert hull3d.points.shape == (5, 3)

    kde = utils.kde(np.array([0.0, 0.5, 1.0, 1.5]))
    assert np.asarray(kde([0.5])).shape == (1,)


def test_body_pos_and_contour_plot_branches(monkeypatch):
    class FakeXYZ:
        def to(self, unit):
            return SimpleNamespace(value=np.array([1.0, 2.0, 3.0]))

    fake_cartesian = SimpleNamespace(get_xyz=lambda: FakeXYZ())
    fake_coord = SimpleNamespace(cartesian=fake_cartesian)
    fake_body = SimpleNamespace(
        heliocentricmeanecliptic=fake_coord,
        gcrs=fake_coord,
        icrs=fake_coord,
        barycentricmeanecliptic=fake_coord,
        barycentrictrueecliptic=fake_coord,
    )
    monkeypatch.setattr(utils, "Time", lambda date, format="jd": SimpleNamespace(date=date, format=format))
    monkeypatch.setattr(utils, "get_body", lambda body, t: fake_body)

    for coord in ["heliocentricmeanecliptic", "gcrs", "icrs", "barycentricmeanecliptic", "barycentrictrueecliptic"]:
        np.testing.assert_allclose(utils.body_pos("earth", coord=coord), [1.0, 2.0, 3.0])

    monkeypatch.setattr(plt, "show", lambda: None)
    plt.close("all")
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    assert utils.contours_2d(points, plot=True).shape[1] == 2
    plt.close("all")
    hull = utils.contours_3d(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
        plot=True,
    )
    assert hull.simplices.size > 0


def test_ssapy_kwargs_defaults_and_overrides():
    from ssapy_toolkit.ssapy_wrappers.sat_kwargs import ssapy_kwargs

    assert ssapy_kwargs() == {"mass": 250, "area": 0.022, "CD": 2.3, "CR": 1.3}
    assert ssapy_kwargs(mass=1, area=2, CD=3, CR=4) == {"mass": 1, "area": 2, "CD": 3, "CR": 4}


def test_find_local_extrema_reduces_vector_rows(capsys):
    minima, maxima = utils.find_local_extrema(np.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))

    assert minima == [1]
    assert maxima == []
    assert "reducing along last axis" in capsys.readouterr().out
