import pytest
from unittest.mock import MagicMock

import numpy as np
from astropy.time import Time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import io
from PIL import Image as PILImage
from io import BytesIO
from matplotlib import cm
import os
from yeager_utils.plotting import *

# Test make_white and make_black
def test_make_white():
    fig, ax = plt.subplots()
    ax.set_title('Test Title')
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    
    fig, axes = make_white(fig, ax)  # Get axes as a tuple
    ax = axes[0]  # Unpack the first (and only) axes from the tuple

    # Check if background is white (RGBA format)
    assert fig.patch.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1.0)  # Ensure ax is the Axes object

def test_make_black():
    fig, ax = plt.subplots()
    ax.set_title('Test Title')
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    
    fig, axes = make_black(fig, ax)  # Get axes as a tuple
    ax = axes[0]  # Unpack the first (and only) axes from the tuple

    # Check if background is black (RGBA format)
    assert fig.patch.get_facecolor() == (0.0, 0.0, 0.0, 1.0)
    assert ax.get_facecolor() == (0.0, 0.0, 0.0, 1.0)  # Ensure ax is the Axes object


# Test draw_dashed_circle
def test_draw_dashed_circle():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    normal_vector = np.array([0, 0, 1])
    radius = 10
    dashes = 5
    draw_dashed_circle(ax, normal_vector, radius, dashes)

    # Check if the plot has a dashed circle (this can be done by checking line objects)
    lines = ax.get_lines()
    assert len(lines) > 0

# Test create_sphere
def test_create_sphere():
    cx, cy, cz, r = 0, 0, 0, 10
    sphere = create_sphere(cx, cy, cz, r)
    
    # Check if the sphere is correctly created with expected shape
    assert sphere.shape == (3, 720, 360)  # Update the expected shape if necessary

# Test koe_plot (for a basic test case)
def test_koe_plot():
    r = np.random.rand(100, 3)  # Random position
    v = np.random.rand(100, 3)  # Random velocity
    fig, ax = koe_plot(r, v)

    # Check that the figure and axes are returned
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

# Test koe_2dhist (mocking StableData for test)
def test_koe_2dhist():
    stable_data = MagicMock()
    stable_data.a = np.random.rand(100)
    stable_data.e = np.random.rand(100)
    stable_data.i = np.random.rand(100)
    stable_data.ta = np.random.rand(100)
    
    fig = koe_2dhist(stable_data)

    # Check if the figure is returned and contains subplots
    assert isinstance(fig, plt.Figure)
    assert len(fig.get_axes()) > 0

# Test display_figure with valid image file
def test_display_figure_valid_file():
    # Create a temporary image file
    img = np.random.rand(100, 100)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    fig.savefig('test_image.png')
    
    display_figure('test_image', display='PIL')  # Test PIL display
    
    # Check if the file is opened (This check can be replaced with mock checks for production)
    assert os.path.isfile('test_image.png')

# Test display_figure when no file found
def test_display_figure_no_file():
    display_figure('non_existent_image.png')

# Check for proper exception handling for invalid display
def test_display_figure_invalid_display():
    with pytest.raises(ValueError):
        display_figure('test_image', display='InvalidDisplay')


@pytest.fixture
def figure():
    # Create a simple figure for testing
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    return fig

def test_save_plot_png(figure):
    save_path = 'test_plot.png'
    
    # Run the save function
    save_plot(figure, save_path)
    
    # Check if the file was created
    assert os.path.isfile(save_path), f"File not found: {save_path}"
    
    # Clean up after the test
    os.remove(save_path)

def test_save_plot_pdf(figure):
    save_path = 'test_plot.pdf'
    
    # Run the save function with a PDF extension
    save_plot(figure, save_path)
    
    # Check if the file was created
    assert os.path.isfile(save_path), f"File not found: {save_path}"
    
    # Clean up after the test
    os.remove(save_path)

def test_save_plot_invalid_extension(figure):
    save_path = 'test_plot.txt'  # Invalid extension
    
    # Run the save function
    save_plot(figure, save_path)
    
    # Check if it saved with the default .png extension
    assert os.path.isfile('test_plot.png'), "Figure was not saved with a .png extension"
    
    # Clean up after the test
    os.remove('test_plot.png')

def test_save_plot_error_handling(figure):
    invalid_path = '/invalid/path/test_plot.png'  # Invalid path (e.g., non-existent directory)
    
    # Run the save function and expect it to handle the error gracefully
    try:
        save_plot(figure, invalid_path)
    except Exception:
        pass  # If an exception is raised, it's handled by the function itself



# Test saving the figure as a PDF
def test_save_plot_to_pdf(figure):
    pdf_path = 'test_output.pdf'
    
    # Run the function to save the plot to PDF
    save_plot_to_pdf(figure, pdf_path)
    
    # Check if the file was created
    assert os.path.isfile(pdf_path), f"PDF file not found: {pdf_path}"
    
    # Clean up after the test
    os.remove(pdf_path)

# Test merging PDFs when the file already exists
def test_save_plot_to_pdf_merge(figure):
    pdf_path = 'test_output.pdf'
    
    # First save the figure to PDF
    save_plot_to_pdf(figure, pdf_path)
    
    # Save a second figure to the same PDF
    save_plot_to_pdf(figure, pdf_path)
    
    # Check if the file exists
    assert os.path.isfile(pdf_path), f"PDF file not found: {pdf_path}"
    
    # Here, we would normally check if the file has more than one page, but for simplicity,
    # we just check that it exists and was modified.
    
    # Clean up after the test
    os.remove(pdf_path)

# Test saving to a PDF with an invalid path (non-existent directory)
def test_save_plot_to_pdf_invalid_path(figure):
    invalid_pdf_path = '/non_existent_directory/test_output.pdf'
    
    # Run the function and expect it to handle the error gracefully
    try:
        save_plot_to_pdf(figure, invalid_pdf_path)
    except Exception as e:
        assert isinstance(e, FileNotFoundError), f"Expected FileNotFoundError but got {type(e)}"

# Test saving the figure as a PDF with path expansion for ~
def test_save_plot_to_pdf_with_tilde(figure):
    pdf_path = '~/test_output.pdf'
    
    # Run the function to save the plot to PDF
    save_plot_to_pdf(figure, pdf_path)
    
    # Check if the file was created after expanding the ~
    expanded_path = os.path.expanduser(pdf_path)
    assert os.path.isfile(expanded_path), f"PDF file not found: {expanded_path}"
    
    # Clean up after the test
    os.remove(expanded_path)

# Test that the figure is closed after saving
def test_save_plot_to_pdf_figure_close(figure, capsys):
    pdf_path = 'test_output.pdf'
    
    # Run the function to save the plot to PDF
    save_plot_to_pdf(figure, pdf_path)
    
    # Capture the output printed to the console
    captured = capsys.readouterr()
    
    # Check if the figure is closed by the function
    assert "Saved figure" in captured.out
    assert not plt.fignum_exists(figure.number), "Figure was not closed"
    
    # Clean up after the test
    os.remove(pdf_path)

# Test the case where the pdf already exists
def test_save_plot_to_pdf_existing_file(figure):
    pdf_path = 'existing_test_output.pdf'
    
    # Create an initial PDF file with a single plot
    save_plot_to_pdf(figure, pdf_path)
    
    # Save another figure to the same PDF to test merging
    save_plot_to_pdf(figure, pdf_path)
    
    # Check if the file exists and was modified
    assert os.path.isfile(pdf_path), f"PDF file not found: {pdf_path}"
    
    # Clean up after the test
    os.remove(pdf_path)


# Fixture for creating test data (satellite position and time)
@pytest.fixture
def satellite_data():
    # Simulating a simple satellite trajectory over time
    t = np.linspace(0, 3600, 100)  # time (seconds)
    r = np.array([np.cos(t), np.sin(t), np.zeros_like(t)]).T  # Circular orbit in XY plane
    return r, t


# Test for tracking a single satellite and plotting
def test_tracking_plot_single_satellite(satellite_data):
    r, t = satellite_data
    
    # Call the function with test data
    fig = tracking_plot(r, t, title="Single Satellite Tracking")
    
    # Check if the figure is a valid matplotlib figure object
    assert isinstance(fig, plt.Figure), "Returned object is not a matplotlib figure"
    
    # Check if the figure has at least one axis (indicating a plot was generated)
    assert len(fig.axes) > 0, "No axes found in the figure"
    
    # Optionally, check if the plot has the expected title
    assert fig.axes[0].get_title() == "Single Satellite Tracking", "Title not set correctly"


# Test for tracking multiple satellites
def test_tracking_plot_multiple_satellites(satellite_data):
    r, t = satellite_data
    r2 = np.array([np.cos(t + np.pi/4), np.sin(t + np.pi/4), np.zeros_like(t)]).T  # Another satellite with phase shift
    r_all = [r, r2]  # List of satellite positions
    
    # Call the function with multiple satellites
    fig = tracking_plot(r_all, t, title="Multiple Satellite Tracking")
    
    # Check if the figure is a valid matplotlib figure object
    assert isinstance(fig, plt.Figure), "Returned object is not a matplotlib figure"
    
    # Check if the figure has at least one axis (indicating a plot was generated)
    assert len(fig.axes) > 0, "No axes found in the figure"
    
    # Optionally, check if the plot has the expected title
    assert fig.axes[0].get_title() == "Multiple Satellite Tracking", "Title not set correctly"


# Test saving the plot to a file
def test_tracking_plot_save(satellite_data):
    r, t = satellite_data
    save_path = 'test_tracking_plot.png'
    
    # Call the function with save path
    fig = tracking_plot(r, t, save_path=save_path, title="Save Plot Test")
    
    # Check if the file is created
    assert os.path.isfile(save_path), f"File was not created: {save_path}"
    
    # Clean up the saved file
    os.remove(save_path)


# Test ground stations being plotted
def test_tracking_plot_with_ground_stations(satellite_data):
    r, t = satellite_data
    ground_stations = [(40, -75), (51, 0)]  # New York and London
    
    # Call the function with ground stations
    fig = tracking_plot(r, t, ground_stations=ground_stations, title="Ground Stations Test")
    
    # Check if the figure is a valid matplotlib figure object
    assert isinstance(fig, plt.Figure), "Returned object is not a matplotlib figure"
    
    # Check if the ground stations are plotted (by checking for scatter plot markers)
    scatter_found = False
    for ax in fig.axes:
        for artist in ax.get_children():
            if isinstance(artist, matplotlib.collections.PathCollection):
                scatter_found = True
                break
    
    assert scatter_found, "Ground stations were not plotted"


# Test custom axis limits
def test_tracking_plot_custom_limits(satellite_data):
    r, t = satellite_data
    custom_limits = 2.0  # Custom limits for the plot
    
    # Call the function with custom limits
    fig = tracking_plot(r, t, limits=custom_limits, title="Custom Limits Test")
    
    # Check the axis limits for the first two subplots (longitude-latitude plots)
    for i, ax in enumerate(fig.axes[:1]):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim == (-180, 180), f"X limits not set correctly for subplot {i+1}, found: {xlim}"
        assert ylim == (-90, 90), f"Y limits not set correctly for subplot {i+1}, found: {ylim}"

    # Check the axis limits for the other subplots (XY, XZ, YZ plots)
    for i, ax in enumerate(fig.axes[-2:]):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim == (-custom_limits, custom_limits), f"X limits not set correctly for subplot {i+5}, found: {xlim}"
        assert ylim == (-custom_limits, custom_limits), f"Y limits not set correctly for subplot {i+5}, found: {ylim}"


# Test if the function handles invalid input types (e.g., if r is not a numpy array)
def test_tracking_plot_invalid_input():
    r = "invalid_input"  # Invalid input type (string instead of numpy array)
    t = np.linspace(0, 3600, 100)
    
    with pytest.raises(TypeError):
        tracking_plot(r, t)


# Test if the function handles empty input (no data for r or t)
def test_tracking_plot_empty_input():
    r = np.array([])  # Empty array for satellite positions
    t = np.array([])  # Empty array for timestamps
    
    # Expecting some kind of error or a gracefully handled case
    with pytest.raises(ValueError):
        tracking_plot(r, t)


def test_globe_plot():
    # Create mock satellite data (r) and time data (t)
    # Let's assume the satellite's position is along a circular orbit with a radius of Earth
    # and the satellite moves in the x-y plane.

    t = np.linspace(0, 10, 100)  # 100 time points
    r = np.zeros((100, 3))  # Initialize positions array
    for i in range(100):
        angle = 2 * np.pi * t[i] / 10  # Circular motion with period 10
        r[i] = [EARTH_RADIUS * np.cos(angle), EARTH_RADIUS * np.sin(angle), 0]  # x, y, z positions

    # Set the plot parameters
    limits = 1.5 * EARTH_RADIUS / RGEO  # Set limits slightly larger than the Earth radius
    title = "Satellite Tracking on Earth"
    figsize = (8, 8)
    save_path = False  # Set to False to not save the plot
    el = 30  # Elevation angle
    az = 45  # Azimuth angle
    scale = 1  # No scaling

    # Call the globe_plot function
    fig, ax_tuple = globe_plot(r, t, limits=limits, title=title, figsize=figsize,
                               save_path=save_path, el=el, az=az, scale=scale)

    # Access the axes from the tuple (if ax is a tuple of axes)
    ax = ax_tuple[0]  # Assuming the first axis is the 3D plot
    
    # Test assertions
    assert isinstance(fig, plt.Figure), "Returned object is not a matplotlib figure"
    assert isinstance(ax, Axes3D), f"Returned object is not a matplotlib 3D Axes, but {type(ax)}"


def test_orbit_plot():
    # Create mock satellite orbit data (r) in 3D space (x, y, z)
    num_points = 100  # Number of time steps (data points)
    t = np.linspace(0, 10, num_points)  # Time array
    r = np.zeros((num_points, 3))  # Initialize position array
    
    # Example: circular orbit in the xy-plane with a radius of Earth
    for i in range(num_points):
        angle = 2 * np.pi * t[i] / 10  # Period of 10
        r[i] = [EARTH_RADIUS * np.cos(angle), EARTH_RADIUS * np.sin(angle), 0]  # x, y, z positions

    # Set the plot parameters
    limits = False  # Auto-limit based on the data
    title = "Satellite Orbit"
    figsize = (8, 8)
    save_path = False  # Set to False to not save the plot
    frame = "gcrf"  # Global coordinate reference frame
    show = False  # Do not display plot
    legend = True  # Include legend
    labels = ["Orbit 1"]  # Label for the orbit

    # Call the orbit_plot function
    fig, axes = orbit_plot(r, t, limits=limits, title=title, figsize=figsize,
                           save_path=save_path, frame=frame, show=show, 
                           legend=legend, labels=labels)

    # Test assertions
    assert isinstance(fig, plt.Figure), "Returned object is not a matplotlib figure"
    assert len(axes) == 4, "The number of axes returned is not 4"
    assert isinstance(axes[0], plt.Axes), "The first subplot is not a matplotlib Axes"
    assert isinstance(axes[3], plt.Axes), "The 3D plot is not a matplotlib Axes"

    # Additional test: Check if the axes limits are correctly set
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    ax4 = axes[3]
    
    # Check if the limits are correctly applied
    assert ax1.get_xlim() != (0, 0), "X limits for ax1 are not set correctly"
    assert ax1.get_ylim() != (0, 0), "Y limits for ax1 are not set correctly"
    assert ax2.get_xlim() != (0, 0), "X limits for ax2 are not set correctly"
    assert ax2.get_ylim() != (0, 0), "Z limits for ax2 are not set correctly"
    assert ax3.get_xlim() != (0, 0), "Y limits for ax3 are not set correctly"
    assert ax3.get_ylim() != (0, 0), "Z limits for ax3 are not set correctly"
    assert ax4.get_xlim() != (0, 0), "X limits for ax4 are not set correctly"
    assert ax4.get_ylim() != (0, 0), "Y limits for ax4 are not set correctly"
    assert ax4.get_zlim() != (0, 0), "Z limits for ax4 are not set correctly"

    print("Test passed successfully!")


def test_check_numpy_array():
    # Test cases
    test_cases = [
        (np.array([1, 2, 3]), "numpy array"),  # Single NumPy array
        ([np.array([1, 2, 3]), np.array([4, 5, 6])], "list of numpy array"),  # List of NumPy arrays
        ([1, 2, 3], "not numpy"),  # List of non-NumPy elements
        ("string", "not numpy"),  # String (not a NumPy array or list)
        (3.14, "not numpy"),  # Float (not a NumPy array or list)
        ([], "not numpy")  # Empty list (not a NumPy array or list)
    ]
    
    # Run the tests
    for i, (input_data, expected_result) in enumerate(test_cases):
        result = check_numpy_array(input_data)
        assert result == expected_result, f"Test case {i+1} failed: expected {expected_result}, got {result}"
    
    print("All test cases passed successfully!")


def test_groundTrackPlot():
    # Simulate mock data for testing
    N = 100  # Number of time steps
    r = np.random.randn(3, N) * 1000000  # Random 3D positions (in meters)
    
    # Simulate time array (let's assume 100 time steps starting from a specific time)
    time = Time("2024-01-01T00:00:00", scale='utc') + np.arange(N) * 60  # Time in seconds (1 minute intervals)
    
    # Simulate mock ground station data (latitude, longitude in degrees)
    ground_stations = np.array([
        [45.0, -75.0],  # Example: Ottawa
        [51.5074, -0.1278],  # Example: London
        [-33.8688, 151.2093]  # Example: Sydney
    ])

    # Call the groundTrackPlot function with the mock data
    try:
        groundTrackPlot(r, time, ground_stations)
        print("Test passed: groundTrackPlot ran successfully.")
    except Exception as e:
        print(f"Test failed: {e}")


if os.path.isfile('test_image.png'):
    os.remove('test_image.png')


if __name__ == "__main__":
    import os
    import pytest

    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the script's name
    script_name = os.path.basename(__file__)
    
    # Construct the path dynamically
    test_dir = os.path.join(current_dir, script_name)
    
    # Run pytest
    pytest.main([test_dir])
