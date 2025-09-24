# flake8: noqa: E501
import os
import sys
import pytest
from unittest.mock import MagicMock

import numpy as np
from astropy.time import Time, TimeDelta

# Use a non-interactive backend for CI before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from yeager_utils import (
    RGEO,
    EARTH_RADIUS,
    ssapy_orbit,
    make_white,
    make_black,
    draw_dashed_circle,
    create_sphere,
    koe_plot,
    orbit_plot,
    tracking_plot,
    globe_plot,
    save_plot_to_pdf,
    koe_2dhist,
    display_figure,
    save_plot,
    check_numpy_array,
    groundTrackPlot,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def imname(base_dir, filename):
    """
    Create an images/ directory under base_dir and return a PNG path for filename.
    """
    save_dir = os.path.join(base_dir, "images")
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, f"{filename}.png")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

# Avoid cross-test global state; if you truly want caching, keep it local to the
# test session with scope="session".
@pytest.fixture(scope="session")
def satellite_data():
    # Simulate a simple satellite trajectory over time
    r, v, t = ssapy_orbit(a=RGEO, e=0.1, i=np.radians(30), duration=(1, "day"), freq=(1, "hour"))
    return r, t


@pytest.fixture(scope="session")
def satellite_data2():
    r, v, t = ssapy_orbit(a=RGEO * 2, e=0.1, i=np.radians(30), duration=(1, "day"), freq=(1, "hour"))
    return r, t


@pytest.fixture
def figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    return fig


# -----------------------------------------------------------------------------
# Tests for styling helpers
# -----------------------------------------------------------------------------

def test_make_white():
    fig, ax = plt.subplots()
    ax.set_title("Test Title")
    ax.set_xlabel("X-Axis")
    ax.set_ylabel("Y-Axis")

    fig, axes = make_white(fig, ax)
    # Some implementations return a tuple of axes; normalize to a list
    if not isinstance(axes, (list, tuple)):
        axes = (axes,)
    ax = axes[0]

    assert fig.patch.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1.0)


def test_make_black():
    fig, ax = plt.subplots()
    ax.set_title("Test Title")
    ax.set_xlabel("X-Axis")
    ax.set_ylabel("Y-Axis")

    fig, axes = make_black(fig, ax)
    if not isinstance(axes, (list, tuple)):
        axes = (axes,)
    ax = axes[0]

    assert fig.patch.get_facecolor() == (0.0, 0.0, 0.0, 1.0)
    assert ax.get_facecolor() == (0.0, 0.0, 0.0, 1.0)


def test_draw_dashed_circle():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    normal_vector = np.array([0, 0, 1])
    radius = 10
    dashes = 5
    draw_dashed_circle(ax, normal_vector, radius, dashes)

    lines = ax.get_lines()
    assert len(lines) > 0


def test_create_sphere():
    cx, cy, cz, r = 0, 0, 0, 10
    sphere = create_sphere(cx, cy, cz, r)
    # Do not over-specify exact shape; check the semantic structure.
    assert isinstance(sphere, np.ndarray)
    assert sphere.ndim == 3 and sphere.shape[0] == 3
    assert sphere.shape[1] > 10 and sphere.shape[2] > 10
    assert np.isfinite(sphere).all()


# -----------------------------------------------------------------------------
# KOE plots
# -----------------------------------------------------------------------------

def test_koe_plot():
    r = np.random.rand(100, 3)
    v = np.random.rand(100, 3)
    fig, ax = koe_plot(r, v)
    assert isinstance(fig, plt.Figure)
    assert hasattr(ax, "plot")


def test_koe_2dhist(tmp_path):
    stable_data = MagicMock()
    stable_data.a = np.random.rand(100)
    stable_data.e = np.random.rand(100)
    stable_data.i = np.random.rand(100)
    stable_data.ta = np.random.rand(100)

    out = imname(tmp_path, "koe_2dhist")
    fig = koe_2dhist(stable_data, save_path=out)
    assert isinstance(fig, plt.Figure)
    assert len(fig.get_axes()) > 0
    assert os.path.isfile(out)


# -----------------------------------------------------------------------------
# display_figure and save_plot
# -----------------------------------------------------------------------------

def test_display_figure_valid_file(tmp_path):
    img_path = tmp_path / "test_image.png"
    fig, ax = plt.subplots()
    ax.imshow(np.random.rand(100, 100), cmap="gray")
    fig.savefig(img_path)

    # Pass the path without extension if display_figure expects a stem
    display_figure(str(img_path.with_suffix("")), display="PIL")
    assert img_path.is_file()


def test_display_figure_no_file():
    # Expecting graceful handling (no exception)
    display_figure("non_existent_image.png")


def test_display_figure_invalid_display():
    with pytest.raises(ValueError):
        display_figure("test_image", display="InvalidDisplay")


def test_save_plot_png(figure, tmp_path):
    save_path = tmp_path / "test_plot.png"
    save_plot(figure, str(save_path))
    assert save_path.is_file()


def test_save_plot_pdf(figure, tmp_path):
    save_path = tmp_path / "test_plot.pdf"
    save_plot(figure, str(save_path))
    assert save_path.is_file()


def test_save_plot_invalid_extension(figure, tmp_path):
    target = tmp_path / "test_plot.txt"
    save_plot(figure, str(target))
    # Expect defaulting to .png next to the target stem
    fallback = target.with_suffix(".png")
    assert fallback.is_file()


def test_save_plot_error_handling(figure):
    invalid_path = "/invalid/path/test_plot.png"
    try:
        save_plot(figure, invalid_path)
    except Exception:
        # If an exception is raised, it should be handled/logged by the function
        pass


# -----------------------------------------------------------------------------
# save_plot_to_pdf
# -----------------------------------------------------------------------------

def test_save_plot_to_pdf(figure, tmp_path):
    pdf_path = tmp_path / "test_output.pdf"
    save_plot_to_pdf(figure, str(pdf_path))
    assert pdf_path.is_file()


def test_save_plot_to_pdf_merge(figure, tmp_path):
    pdf_path = tmp_path / "test_output.pdf"
    save_plot_to_pdf(figure, str(pdf_path))
    save_plot_to_pdf(figure, str(pdf_path))
    assert pdf_path.is_file()


def test_save_plot_to_pdf_invalid_path(figure):
    invalid_pdf_path = "/non_existent_directory/test_output.pdf"
    with pytest.raises(FileNotFoundError):
        save_plot_to_pdf(figure, invalid_pdf_path)


def test_save_plot_to_pdf_with_tilde(figure):
    pdf_path = os.path.expanduser("~/test_output.pdf")
    try:
        save_plot_to_pdf(figure, "~/test_output.pdf")
        assert os.path.isfile(pdf_path)
    finally:
        if os.path.isfile(pdf_path):
            os.remove(pdf_path)


def test_save_plot_to_pdf_figure_close(figure, tmp_path):
    pdf_path = tmp_path / "test_output.pdf"
    # Save and ensure the figure gets closed by the function (if that is the contract)
    save_plot_to_pdf(figure, str(pdf_path))
    assert pdf_path.is_file()
    # Prefer effect-based assertion; avoid hard-coding log text
    assert not plt.fignum_exists(figure.number), "Figure was not closed"


# -----------------------------------------------------------------------------
# tracking_plot
# -----------------------------------------------------------------------------

def test_tracking_plot_single_satellite(satellite_data, tmp_path):
    r, t = satellite_data
    fig = tracking_plot(r, t, title="Single Satellite Tracking", save_path=imname(tmp_path, "tracking_single_sat"))
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0
    assert fig.axes[0].get_title() == "Single Satellite Tracking"


def test_tracking_plot_multiple_satellites(satellite_data, satellite_data2, tmp_path):
    r1, t = satellite_data
    r2, _t2 = satellite_data2
    r_all = [r1, r2]
    fig = tracking_plot(r=r_all, t=t, title="Multiple Satellite Tracking", save_path=imname(tmp_path, "tracking_multiple_sats"))
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0
    assert fig.axes[0].get_title() == "Multiple Satellite Tracking"


def test_tracking_plot_save(satellite_data, tmp_path):
    r, t = satellite_data
    save_path = tmp_path / "test_tracking_plot.png"
    fig = tracking_plot(r, t, save_path=str(save_path), title="Save Plot Test")
    assert isinstance(fig, plt.Figure)
    assert save_path.is_file()


def test_tracking_plot_with_ground_stations(satellite_data, tmp_path):
    r, t = satellite_data
    ground_stations = [(40, -75), (51, 0)]  # New York and London
    fig = tracking_plot(r, t, ground_stations=ground_stations, title="Ground Stations Test", save_path=imname(tmp_path, "tracking_ground_stations"))
    assert isinstance(fig, plt.Figure)
    # Check for scatter collection presence
    scatter_found = any(any(hasattr(child, "get_offsets") for child in ax.get_children()) for ax in fig.axes)
    assert scatter_found, "Ground stations were not plotted"


def test_tracking_plot_custom_limits(satellite_data):
    r, t = satellite_data
    custom_limits = 2.0
    fig = tracking_plot(r=r, t=t, limits=custom_limits, title="Custom Limits Test")

    # Check first two subplots (lon-lat plots) are in degrees bounds
    for i, ax in enumerate(fig.axes[:2]):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim == (-180, 180), f"X limits not set correctly for subplot {i+1}, found: {xlim}"
        assert ylim == (-90, 90), f"Y limits not set correctly for subplot {i+1}, found: {ylim}"

    # Check last two subplots (XY and XZ/YZ planes) use the custom metric limits
    for i, ax in enumerate(fig.axes[-2:], start=1):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim == (-custom_limits, custom_limits), f"X limits not set correctly for tail subplot {i}, found: {xlim}"
        assert ylim == (-custom_limits, custom_limits), f"Y limits not set correctly for tail subplot {i}, found: {ylim}"


def test_tracking_plot_invalid_input():
    r = "invalid_input"
    t = np.linspace(0, 3600, 100)
    with pytest.raises(TypeError):
        tracking_plot(r, t)


def test_tracking_plot_empty_input():
    r = np.array([])
    t = np.array([])
    with pytest.raises(ValueError):
        tracking_plot(r, t)


# -----------------------------------------------------------------------------
# globe_plot
# -----------------------------------------------------------------------------

def test_globe_plot(tmp_path):
    # Create circular motion in the x-y plane at Earth radius
    t = np.linspace(0, 10, 100)
    r = np.zeros((100, 3))
    for i in range(100):
        angle = 2 * np.pi * t[i] / 10.0
        r[i] = [EARTH_RADIUS * np.cos(angle), EARTH_RADIUS * np.sin(angle), 0.0]

    limits = 1.5 * EARTH_RADIUS / RGEO
    title = "Satellite Tracking on Earth"
    figsize = (8, 8)
    el, az = 30, 45
    scale = 1

    fig, ax_tuple = globe_plot(
        r,
        t,
        limits=limits,
        title=title,
        figsize=figsize,
        el=el,
        az=az,
        scale=scale,
        save_path=imname(tmp_path, "globe_plot"),
    )

    # Normalize ax to a sequence, then choose the first axes
    if not isinstance(ax_tuple, (list, tuple)):
        ax_tuple = (ax_tuple,)
    ax = ax_tuple[0]

    assert isinstance(fig, plt.Figure)
    # Do not rely on Axes3D type; check for a 3D-like API
    assert hasattr(ax, "get_zlim"), f"Expected a 3D axes; got {type(ax)}"


# -----------------------------------------------------------------------------
# orbit_plot
# -----------------------------------------------------------------------------

def test_orbit_plot(satellite_data, tmp_path):
    r, t = satellite_data
    fig, axes = orbit_plot(
        r,
        t,
        title="Satellite Orbit",
        figsize=(7, 7),
        save_path=imname(tmp_path, "orbit_plot"),
        frame="gcrf",
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, (list, tuple)) and len(axes) == 4
    assert isinstance(axes[0], plt.Axes)
    assert isinstance(axes[3], plt.Axes)

    # Sanity checks on axis limits
    ax1, ax2, ax3, ax4 = axes
    assert ax1.get_xlim() != (0, 0)
    assert ax1.get_ylim() != (0, 0)
    assert ax2.get_xlim() != (0, 0)
    assert ax2.get_ylim() != (0, 0)
    assert ax3.get_xlim() != (0, 0)
    assert ax3.get_ylim() != (0, 0)
    assert ax4.get_xlim() != (0, 0)
    assert ax4.get_ylim() != (0, 0)
    assert ax4.get_zlim() != (0, 0)


# -----------------------------------------------------------------------------
# check_numpy_array
# -----------------------------------------------------------------------------

def test_check_numpy_array():
    test_cases = [
        (np.array([1, 2, 3]), "numpy array"),
        ([np.array([1, 2, 3]), np.array([4, 5, 6])], "list of numpy array"),
        ([1, 2, 3], "not numpy"),
        ("string", "not numpy"),
        (3.14, "not numpy"),
        ([], "not numpy"),
    ]
    for i, (input_data, expected_result) in enumerate(test_cases):
        result = check_numpy_array(input_data)
        assert result == expected_result, f"Test case {i+1} failed: expected {expected_result}, got {result}"


# -----------------------------------------------------------------------------
# groundTrackPlot
# -----------------------------------------------------------------------------

def test_groundTrackPlot():
    N = 100
    r = np.random.randn(3, N) * 1_000_000  # meters

    # Use seconds with TimeDelta; bare floats in Time addition are days
    start = Time("2024-01-01T00:00:00", scale="utc")
    time = start + TimeDelta(np.arange(N) * 60, format="sec")

    ground_stations = np.array([
        [45.0, -75.0],       # Ottawa
        [51.5074, -0.1278],  # London
        [-33.8688, 151.2093] # Sydney
    ])

    # Expect no exceptions
    groundTrackPlot(r, time, ground_stations)


# -----------------------------------------------------------------------------
# Pytest entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Run this file's tests directly
    pytest.main([__file__])
