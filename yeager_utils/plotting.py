######################################################################
# COLLECTION OF ALL PLOTTING AND MEDIA
######################################################################
# flake8: noqa: E501
from .orbital_mechanics import calculate_orbital_elements
import numpy as np
from ssapy.body import get_body
from ssapy.compute import groundTrack
from ssapy.utils import find_file
from .compute import find_smallest_bounding_cube
from .constants import RGEO, LD, EARTH_RADIUS, MOON_RADIUS, EARTH_MU, MOON_MU
from .coordinates import gcrf_to_itrf, gcrf_to_lunar, gcrf_to_lunar_fixed
from .orbital_mechanics import lagrange_points_lunar_frame
from .utils import Time
from .vectors import norm, rotation_matrix_from_vectors
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PyPDF2 import PdfMerger
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image as PILImage
import io

from IPython.display import Image as IPythonImage, display as ipython_display
import imageio
import cv2

import ipyvolume as ipv

from typing import List, Tuple, Optional, Union

lunar_semi_major = 384399000  # m


def display_figure(figname: str, display: str = 'IPython') -> None:
    def open_image(filename: str) -> None:
        if display == 'IPython':
            img = IPythonImage(filename=filename)
            ipython_display(img)
        elif display == 'PIL':
            img = PILImage.open(filename)
            img.show()
        else:
            raise ValueError("Invalid display option. Please specify 'IPython' or 'PIL'.")

    if os.path.isfile(figname):
        open_image(figname)
        return

    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
    for ext in image_extensions:
        filename_with_ext = figname + ext
        if os.path.isfile(filename_with_ext):
            open_image(filename_with_ext)
            return

    print("No image file found.")


def make_white(fig: plt.Figure, *axes: plt.Axes) -> Tuple[plt.Figure, Tuple[plt.Axes]]:
    fig.patch.set_facecolor('white')

    for ax in axes:
        ax.set_facecolor('white')
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        if hasattr(ax, 'zaxis'):
            ax_items.append(ax.zaxis.label)
        ax_items += ax.get_xticklabels() + ax.get_yticklabels()
        if hasattr(ax, 'get_zticklabels'):
            ax_items += ax.get_zticklabels()
        ax_items += ax.get_xticklines() + ax.get_yticklines()
        if hasattr(ax, 'get_zticklines'):
            ax_items += ax.get_zticklines()
        for item in ax_items:
            item.set_color('black')

    return fig, axes


def make_black(fig: plt.Figure, *axes: plt.Axes) -> Tuple[plt.Figure, Tuple[plt.Axes]]:
    fig.patch.set_facecolor('black')

    for ax in axes:
        ax.set_facecolor('black')
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        if hasattr(ax, 'zaxis'):
            ax_items.append(ax.zaxis.label)
        ax_items += ax.get_xticklabels() + ax.get_yticklabels()
        if hasattr(ax, 'get_zticklabels'):
            ax_items += ax.get_zticklabels()
        ax_items += ax.get_xticklines() + ax.get_yticklines()
        if hasattr(ax, 'get_zticklines'):
            ax_items += ax.get_zticklines()
        for item in ax_items:
            item.set_color('white')

    return fig, axes


def draw_dashed_circle(ax: plt.Axes, normal_vector: np.ndarray, radius: float, dashes: int, dash_length: float = 0.1, label: str = 'Dashed Circle') -> None:
    # Define the circle in the xy-plane
    theta = np.linspace(0, 2 * np.pi, 1000)
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    z_circle = np.zeros_like(theta)
    
    # Stack the coordinates into a matrix
    circle_points = np.vstack((x_circle, y_circle, z_circle)).T
    
    # Create the rotation matrix to align z-axis with the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    rotation_matrix = rotation_matrix_from_vectors(np.array([0, 0, 1]), normal_vector)
    
    # Rotate the circle points
    rotated_points = circle_points @ rotation_matrix.T
    
    # Create dashed effect
    dash_points = []
    dash_gap = int(len(theta) / dashes)
    for i in range(dashes):
        start_idx = i * dash_gap
        end_idx = start_idx + int(dash_length * len(theta))
        dash_points.append(rotated_points[start_idx:end_idx])
    
    # Plot the dashed circle in 3D
    for points in dash_points:
        ax.plot(points[:, 0], points[:, 1], points[:, 2], 'k--', label=label)
        label = None  # Only one label


def create_sphere(cx: float, cy: float, cz: float, r: float, resolution: int = 360) -> np.ndarray:
    '''
    create sphere with center (cx, cy, cz) and radius r
    '''
    phi = np.linspace(0, 2 * np.pi, 2 * resolution)
    theta = np.linspace(0, np.pi, resolution)

    theta, phi = np.meshgrid(theta, phi)

    r_xy = r * np.sin(theta)
    x = cx + np.cos(phi) * r_xy
    y = cy + np.sin(phi) * r_xy
    z = cz + r * np.cos(theta)

    return np.stack([x, y, z])


def drawSphere(xCenter: float, yCenter: float, zCenter: float, r: float, res: complex = 10j, flatten: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if 'j' not in str(res):
        res = complex(0, res)
    # draw sphere
    u, v = np.mgrid[0:2 * np.pi:2 * res, 0:np.pi:res]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    # shift and scale sphere
    x = r * x + xCenter
    y = r * y + yCenter
    z = r * z + zCenter
    if flatten:
        x = np.squeeze(np.array(x).flatten())
        y = np.squeeze(np.array(y).flatten())
        z = np.squeeze(np.array(z).flatten())
    return (x, y, z)


def darken(color: str, amount: float = 0.5) -> List[str]:
    import colorsys
    try:
        c = matplotlib.colors.cnames[color]
    except Exception:
        c = color
    colors = []
    for i in amount:
        c = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(c))
        colors.append(colorsys.hls_to_rgb(c[0], 1 - i * (1 - c[1]), c[2]))
    return colors


def rgb(minimum: float, maximum: float, value: float) -> Tuple[int, int, int]:
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
    g = 255 - b - r
    return r, g, b


def generate_rainbow_colors(num_iterations: int) -> List[str]:
    cmap = plt.get_cmap('rainbow')
    colors = [matplotlib.colors.rgb2hex(cmap(i / num_iterations)) for i in range(num_iterations)]
    return colors


def write_video(video_name: str, frames: List[str], fps: int = 30) -> None:
    print(f'Writing video: {video_name}')
    """
    Writes frames to an mp4 video file
    :param video_name: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """
    img = cv2.imread(frames[0])
    h, w, layers = img.shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(video_name, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(cv2.imread(frame))

    writer.release()
    print(f'Wrote: {video_name}')
    return


def write_gif(gif_name: str, frames: List[str], fps: int = 30) -> None:
    print(f'Writing gif: {gif_name}')
    with imageio.get_writer(gif_name, mode='I', duration=1 / fps) as writer:
        for i, filename in enumerate(frames):
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f'Wrote {gif_name}')
    return


def koe_plot(r: np.ndarray, v: np.ndarray, t: Optional[Time] = None,
             elements: List[str] = ['a', 'e', 'i'], save_path: Optional[str] = False, body: str = 'Earth') -> Tuple[plt.Figure, plt.Axes]:
    """
    Generates a plot of orbital elements (eccentricity, inclination, and semi-major axis) for a given position and velocity vectors.
    
    Parameters:
    -----------
    r : np.ndarray
        Position vectors of the satellite in 3D space (shape: [n, 3]).
    
    v : np.ndarray
        Velocity vectors of the satellite in 3D space (shape: [n, 3]).
    
    t : Optional[Time], optional
        Time instance(s) associated with the orbital elements. If None, the x-axis will be indexed by the length of the y-data.
    
    elements : List[str], optional
        List of orbital elements to plot. The default is ['a', 'e', 'i'] for semi-major axis, eccentricity, and inclination.
    
    save_path : Optional[str], optional
        Path to save the plot. If False (default), the plot is not saved.
    
    body : str, optional
        Celestial body for which the orbital elements are calculated. The default is 'Earth'. Options: 'Earth' or 'Moon'.

    Returns:
    --------
    Tuple[plt.Figure, plt.Axes]
        The generated figure and axes objects.
    
    Notes:
    ------
    - The plot includes eccentricity, inclination, and semi-major axis for the satellite's orbit.
    - If `t` is None, the x-axis will display an index based on the length of the y-data.
    """
    
    # Calculate orbital elements for Earth or Moon
    if 'earth' in body.lower():
        orbital_elements = calculate_orbital_elements(r, v, mu_barycenter=EARTH_MU)
    else:
        orbital_elements = calculate_orbital_elements(r, v, mu_barycenter=MOON_MU)
    
    # Create figure and axis
    fig, ax1 = plt.subplots(dpi=100)
    ax1.plot([], [], label='semi-major axis [GEO]', c='C0', linestyle='-')
    ax2 = ax1.twinx()
    
    # Apply white background for the figure and axes
    make_white(fig, *[ax1, ax2])

    # Set time values on the x-axis or index if time is not provided
    x_values = Time(t).decimalyear if t is not None else np.arange(len(orbital_elements['e']))
    
    # Plot the orbital elements
    ax1.plot(x_values, [x for x in orbital_elements['e']], label='eccentricity', c='C1')
    ax1.plot(x_values, [x for x in orbital_elements['i']], label='inclination [rad]', c='C2')
    
    # Labels and axis limits
    ax1.set_xlabel('Year' if t is not None else 'Index')
    ax1.set_ylim((0, np.pi / 2))
    ylabel = ax1.set_ylabel('', color='black')
    
    # Adjust label positions and add custom text
    x = ylabel.get_position()[0] + 0.05
    y = ylabel.get_position()[1]
    fig.text(x - 0.001, y - 0.225, 'Eccentricity', color='C1', rotation=90)
    fig.text(x, y - 0.05, '/', color='k', rotation=90)
    fig.text(x, y - 0.025, 'Inclination [Radians]', color='C2', rotation=90)

    # Add legend
    ax1.legend(loc='upper left')

    # Plot the semi-major axis
    a = [x / RGEO for x in orbital_elements['a']]
    ax2.plot(x_values, a, label='semi-major axis [GEO]', c='C0', linestyle='-')
    ax2.set_ylabel('semi-major axis [GEO]', color='C0')
    ax2.yaxis.label.set_color('C0')
    ax2.tick_params(axis='y', colors='C0')
    ax2.spines['right'].set_color('C0')
    
    # Adjust y-axis limits if necessary
    if np.abs(np.max(a) - np.min(a)) < 2:
        ax2.set_ylim((np.min(a) - 0.5, np.max(a) + 0.5))

    # Optionally save the plot
    if save_path:
        fig.savefig(save_path)
    
    return fig, ax1


def koe_2dhist(stable_data: 'StableData', title: str = "Initial orbital elements of\n1 year stable cislunar orbits", 
               limits: List[float] = [1, 50], bins: int = 200, logscale: Union[bool, str] = False, cmap: str = 'coolwarm', 
               save_path: Optional[str] = None) -> plt.Figure:
    """
    Generates a 2D histogram plot of orbital elements for a set of stable orbital data.
    The plot includes histograms for the relationships between semi-major axis, eccentricity, 
    inclination, and true anomaly.

    Parameters:
    -----------
    stable_data : 'StableData'
        An object containing the orbital data. The `StableData` class should have attributes 
        such as `a` (semi-major axis), `e` (eccentricity), `i` (inclination), and `ta` (true anomaly).
    
    title : str, optional
        The title for the overall plot. Default is "Initial orbital elements of\n1 year stable cislunar orbits".
    
    limits : List[float], optional
        The limits for the color normalization of the histogram. Default is [1, 50].
    
    bins : int, optional
        The number of bins for the 2D histograms. Default is 200.
    
    logscale : Union[bool, str], optional
        If True or 'log', the histogram will be plotted with a logarithmic scale. 
        Default is False (linear scale).
    
    cmap : str, optional
        The colormap for the 2D histograms. Default is 'coolwarm'.
    
    save_path : Optional[str], optional
        The path where the plot will be saved. If None, the plot is not saved. Default is None.

    Returns:
    --------
    plt.Figure
        The generated matplotlib figure object containing the 2D histogram plots.

    Notes:
    ------
    - The figure consists of 9 subplots arranged in a 3x3 grid, each representing the 2D histogram 
      of different pairs of orbital elements.
    - The histograms are plotted for the relationships between:
        1. Semi-major axis vs Eccentricity
        2. Semi-major axis vs Inclination
        3. Eccentricity vs Inclination
        4. Semi-major axis vs True Anomaly
        5. Eccentricity vs True Anomaly
        6. Inclination vs True Anomaly
    - The color normalization for the histograms can be adjusted based on the `limits` and `logscale` parameters.
    - A colorbar is added to the right of the plot.
    
    Example:
    --------
    stable_data = StableData(a=np.array([...]), e=np.array([...]), i=np.array([...]), ta=np.array([...]))
    fig = koe_2dhist(stable_data, title="Orbital Elements", save_path="orbital_elements_plot.png")
    """

    # Validate angle data ranges
    if not (np.all((0 <= stable_data.i) & (stable_data.i <= 2 * np.pi))):
        raise ValueError("Inclination (`i`) must be in the range [0, 2π] radians.")
    if not (np.all((0 <= stable_data.ta) & (stable_data.ta <= 2 * np.pi))):
        raise ValueError("True Anomaly (`ta`) must be in the range [0, 2π] radians.")

    if logscale or logscale == 'log':
        norm = matplotlib.colors.LogNorm(limits[0], limits[1])
    else:
        norm = matplotlib.colors.Normalize(limits[0], limits[1])

    fig, axes = plt.subplots(dpi=100, figsize=(10, 8), nrows=3, ncols=3)
    st = fig.suptitle(title, fontsize=12)
    st.set_x(0.46)
    st.set_y(0.9)
    
    # Semi-major axis vs Eccentricity
    ax = axes.flat[0]
    ax.hist2d([x / RGEO for x in stable_data.a], [x for x in stable_data.e], bins=bins, norm=norm, cmap=cmap)
    ax.set_xlabel("")
    ax.set_ylabel("eccentricity")
    ax.set_xticks(np.arange(1, 20, 2))
    ax.set_yticks(np.arange(0, 1, 0.2))
    ax.set_xlim((1, 18))
    
    # Empty plots
    axes.flat[1].set_axis_off()
    axes.flat[2].set_axis_off()

    # Semi-major axis vs Inclination
    ax = axes.flat[3]
    ax.hist2d([x / RGEO for x in stable_data.a], [np.degrees(x) for x in stable_data.i], bins=bins, norm=norm, cmap=cmap)
    ax.set_xlabel("")
    ax.set_ylabel("inclination [deg]")
    ax.set_xticks(np.arange(1, 20, 2))
    ax.set_yticks(np.arange(0, 91, 15))
    ax.set_xlim((1, 18))
    
    # Eccentricity vs Inclination
    ax = axes.flat[4]
    ax.hist2d([x for x in stable_data.e], [np.degrees(x) for x in stable_data.i], bins=bins, norm=norm, cmap=cmap)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(0, 1, 0.2))
    ax.set_yticks(np.arange(0, 91, 15))
    
    # Empty plot
    axes.flat[5].set_axis_off()

    # Semi-major axis vs True Anomaly
    ax = axes.flat[6]
    ax.hist2d([x / RGEO for x in stable_data.a], [np.degrees(x) for x in stable_data.ta], bins=bins, norm=norm, cmap=cmap)
    ax.set_xlabel("semi-major axis [GEO]")
    ax.set_ylabel("True Anomaly [deg]")
    ax.set_xticks(np.arange(1, 20, 2))
    ax.set_yticks(np.arange(0, 361, 60))
    ax.set_xlim((1, 18))
    
    # Eccentricity vs True Anomaly
    ax = axes.flat[7]
    ax.hist2d([x for x in stable_data.e], [np.degrees(x) for x in stable_data.ta], bins=bins, norm=norm, cmap=cmap)
    ax.set_xlabel("eccentricity")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(0, 1, 0.2))
    ax.set_yticks(np.arange(0, 361, 60))
    
    # Inclination vs True Anomaly
    ax = axes.flat[8]
    ax.hist2d([np.degrees(x) for x in stable_data.i], [np.degrees(x) for x in stable_data.ta], bins=bins, norm=norm, cmap=cmap)
    ax.set_xlabel("inclination [deg]")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(0, 91, 15))
    ax.set_yticks(np.arange(0, 361, 60))

    # Adjust colorbar and make the figure white
    im = fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax, norm=norm, cmap=cmap)
    fig, ax = make_white(fig, ax)
    
    # Save plot if save_path is provided
    if save_path:
        save_plot(fig, save_path)

    return fig


def scatter2d(x: List[float], y: List[float], cs: List[float], xlabel: str = 'x', ylabel: str = 'y', title: str = '', 
              cbar_label: str = '', dotsize: int = 1, colorsMap: str = 'jet', colorscale: str = 'linear', 
              colormin: Optional[float] = None, colormax: Optional[float] = None, save_path: Optional[str] = None) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if colormax is False:
        colormax = np.max(cs)
    if colormin is False:
        colormin = np.min(cs)
    cm = plt.get_cmap(colorsMap)
    if colorscale == 'linear':
        cNorm = matplotlib.colors.Normalize(vmin=colormin, vmax=colormax)
    elif colorscale == 'log':
        cNorm = matplotlib.colors.LogNorm(vmin=colormin, vmax=colormax)
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
    ax.scatter(x, y, c=scalarMap.to_rgba(cs), s=dotsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap, shrink=.5, label=f'{cbar_label}', pad=0.04)
    plt.tight_layout()
    fig, ax = make_black(fig, ax)
    plt.show(block=False)
    if save_path:
        save_plot(fig, save_path)
    return


def scatter3d(x: List[float], y: Optional[List[float]] = None, z: Optional[List[float]] = None, cs: Optional[List[float]] = None, 
              xlabel: str = 'x', ylabel: str = 'y', zlabel: str = 'z', cbar_label: str = '', dotsize: int = 1, 
              colorsMap: str = 'jet', title: str = '', save_path: Optional[str] = None) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if x.ndim > 1:
        r = x
        x = r[:, 0]
        y = r[:, 1]
        z = r[:, 2]
    if cs is None:
        ax.scatter(x, y, z, s=dotsize)
    else:
        cm = plt.get_cmap(colorsMap)
        cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
        ax.scatter(x, y, z, c=scalarMap.to_rgba(cs), s=dotsize)
        scalarMap.set_array(cs)
        fig.colorbar(scalarMap, shrink=.5, label=f'{cbar_label}', pad=0.075)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.title(title)
    plt.tight_layout()
    fig, ax = make_black(fig, ax)
    plt.show(block=False)
    if save_path:
        save_plot(fig, save_path)
    return fig, ax


def dotcolors_scaled(num_colors: int) -> List[float]:
    return cm.rainbow(np.linspace(0, 1, num_colors))


# Make a plot of multiple cislunar orbit in GCRF frame.
def orbit_divergence_plot(rs: np.ndarray, r_moon: Optional[np.ndarray] = [], t: Optional[Union[np.ndarray, 'Time']] = False, 
                          limits: Optional[float] = False, title: str = '', save_path: Optional[str] = None) -> None:
    if limits is False:
        limits = np.nanmax(np.linalg.norm(rs, axis=1) / RGEO) * 1.2
        print(f'limits: {limits}')
    if np.size(r_moon) < 1:
        moon = get_body("moon")
        r_moon = moon.position(t)
    else:
        # print('Lunar position(s) provided.')
        if r_moon.ndim != 2:
            raise IndexError(f"input moon data shape: {np.shape(r_moon)}, input should be 2 dimensions.")
            return None
        if np.shape(r_moon)[1] == 3:
            r_moon = r_moon.T
            # print(f"Tranposed input to {np.shape(r_moon)}")
    fig = plt.figure(dpi=100, figsize=(15, 4))
    for i in range(rs.shape[-1]):
        r = rs[:, :, i]
        x = r[:, 0] / RGEO
        y = r[:, 1] / RGEO
        z = r[:, 2] / RGEO
        xm = r_moon[0] / RGEO
        ym = r_moon[1] / RGEO
        zm = r_moon[2] / RGEO
        dotcolors = cm.rainbow(np.linspace(0, 1, len(x)))

        # Creating plot
        plt.subplot(1, 3, 1)
        plt.scatter(x, y, color=dotcolors, s=1)
        plt.scatter(0, 0, color="blue", s=50)
        plt.scatter(xm, ym, color="grey", s=5)
        plt.axis('scaled')
        plt.xlabel('x [GEO]')
        plt.ylabel('y [GEO]')
        plt.xlim((-limits, limits))
        plt.ylim((-limits, limits))
        plt.text(x[0], y[0], '$\leftarrow$ start')
        plt.text(x[-1], y[-1], '$\leftarrow$ end')

        plt.subplot(1, 3, 2)
        plt.scatter(x, z, color=dotcolors, s=1)
        plt.scatter(0, 0, color="blue", s=50)
        plt.scatter(xm, zm, color="grey", s=5)
        plt.axis('scaled')
        plt.xlabel('x [GEO]')
        plt.ylabel('z [GEO]')
        plt.xlim((-limits, limits))
        plt.ylim((-limits, limits))
        plt.text(x[0], z[0], '$\leftarrow$ start')
        plt.text(x[-1], z[-1], '$\leftarrow$ end')
        plt.title(f'{title}')

        plt.subplot(1, 3, 3)
        plt.scatter(y, z, color=dotcolors, s=1)
        plt.scatter(0, 0, color="blue", s=50)
        plt.scatter(ym, zm, color="grey", s=5)
        plt.axis('scaled')
        plt.xlabel('y [GEO]')
        plt.ylabel('z [GEO]')
        plt.xlim((-limits, limits))
        plt.ylim((-limits, limits))
        plt.text(y[0], z[0], '$\leftarrow$ start')
        plt.text(y[-1], z[-1], '$\leftarrow$ end')
    plt.tight_layout()
    plt.show(block=False)
    if save_path:
        save_plot(fig, save_path)
    return

def load_earth_file() -> 'PILImage':
    earth = PILImage.open(find_file("earth", ext=".png"))
    earth = earth.resize((5400 // 5, 2700 // 5))
    return earth


def drawEarth(time: Union[np.ndarray, 'Time'], ngrid: int = 100, R: float = EARTH_RADIUS, rfactor: float = 1) -> 'Mesh':
    """
    Parameters
    ----------
    time : array_like or astropy.time.Time (n,)
        If float (array), then should correspond to GPS seconds;
        i.e., seconds since 1980-01-06 00:00:00 UTC
    ngrid: int
        Number of grid points in Earth model.
    R: float
        Earth radius in meters.  Default is WGS84 value.
    rfactor: float
        Factor by which to enlarge Earth (for visualization purposes)

    """
    earth = load_earth_file()

    from numbers import Real
    from erfa import gst94
    lat = np.linspace(-np.pi / 2, np.pi / 2, ngrid)
    lon = np.linspace(-np.pi, np.pi, ngrid)
    lat, lon = np.meshgrid(lat, lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    u = np.linspace(0, 1, ngrid)
    v, u = np.meshgrid(u, u)

    # Need earth rotation angle for t
    # Just use erfa.gst94.
    # This ignores precession/nutation, ut1-tt and polar motion, but should
    # be good enough for visualization.
    if isinstance(time, Time):
        time = time.gps
    if isinstance(time, Real):
        time = np.array([time])

    mjd_tt = 44244.0 + (time + 51.184) / 86400
    gst = gst94(2400000.5, mjd_tt)

    u = u - (gst / (2 * np.pi))[:, None, None]
    v = np.broadcast_to(v, u.shape)

    return ipv.plot_mesh(
        x * R * rfactor, y * R * rfactor, z * R * rfactor,
        u=u, v=v,
        wireframe=False,
        texture=earth
    )


def load_moon_file() -> 'PILImage':
    moon = PILImage.open(find_file("moon", ext=".png"))
    moon = moon.resize((5400 // 5, 2700 // 5))
    return moon


def drawMoon(time: Union[np.ndarray, 'Time'], ngrid: int = 100, R: float = MOON_RADIUS, rfactor: float = 1) -> 'Mesh':
    """
    Parameters
    ----------
    time : array_like or astropy.time.Time (n,)
        If float (array), then should correspond to GPS seconds;
        i.e., seconds since 1980-01-06 00:00:00 UTC
    ngrid: int
        Number of grid points in Earth model.
    R: float
        Earth radius in meters.  Default is WGS84 value.
    rfactor: float
        Factor by which to enlarge Earth (for visualization purposes)

    """
    moon = load_moon_file()

    from numbers import Real
    from erfa import gst94
    lat = np.linspace(-np.pi / 2, np.pi / 2, ngrid)
    lon = np.linspace(-np.pi, np.pi, ngrid)
    lat, lon = np.meshgrid(lat, lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    u = np.linspace(0, 1, ngrid)
    v, u = np.meshgrid(u, u)

    # Need earth rotation angle for t
    # Just use erfa.gst94.
    # This ignores precession/nutation, ut1-tt and polar motion, but should
    # be good enough for visualization.
    if isinstance(time, Time):
        time = time.gps
    if isinstance(time, Real):
        time = np.array([time])

    mjd_tt = 44244.0 + (time + 51.184) / 86400
    gst = gst94(2400000.5, mjd_tt)

    u = u - (gst / (2 * np.pi))[:, None, None]
    v = np.broadcast_to(v, u.shape)

    return ipv.plot_mesh(
        x * R * rfactor, y * R * rfactor, z * R * rfactor,
        u=u, v=v,
        wireframe=False,
        texture=moon
    )


def groundTrackPlot(r: Union[np.ndarray, List[np.ndarray]], 
                    time: Union[np.ndarray, List[np.ndarray]], 
                    ground_stations: Optional[np.ndarray] = None, 
                    save_path: Optional[str] = None) -> None:
    """
    Parameters
    ----------
    r : (3,) array_like or List of (3,) array_like - Orbit positions in meters or list of such arrays for multiple orbits.
    time : (n,) array_like or List of (n,) array_like - array of Astropy Time objects or time in gps seconds.
             If r is a list of orbits, time should be a list of time vectors (one for each orbit).
    
    optional - ground_stations: (n,2) array of ground stations (lat, lon) in degrees.
    """
    
    # Check if r is a list of orbits
    if isinstance(r, np.ndarray):
        r = [r]
    if isinstance(time, np.ndarray):
        # If multiple orbits, create a corresponding time list
        if isinstance(time, list):
            time_list = time  # If time is already a list of time arrays
        else:
            time_list = []
            for _ in r:
                time_list.append(time) 
    else:
        # If only one orbit, time should be a single time array
        time_list = [time]

    # Initialize plot
    fig = plt.figure(figsize=(15, 12))
    plt.imshow(load_earth_file(), extent=[-180, 180, -90, 90])

    # Loop through each orbit and its corresponding time
    standard_colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    
    for i, orbit in enumerate(r):
        if len(r) <= len(standard_colors):  # Use standard colors
            print(len(r), len(standard_colors))
            color = standard_colors[i]
        else:  # Use rainbow colors if there are more orbits than standard colors
            color = plt.cm.rainbow(i / len(r))
            
        lon, lat, height = groundTrack(orbit, time_list[i])  # Assuming output in radians

        # Convert longitude and latitude to degrees for plotting
        lon_deg = np.degrees(lon)
        lat_deg = np.degrees(lat)
        
        # Identify discontinuities in longitude (crossing the -180/180 boundary)
        discont_indices = np.where(np.abs(np.diff(lon_deg)) > 179)[0]
        
        # Split the data into segments for clean plotting
        segments = np.split(np.arange(len(lon_deg)), discont_indices + 1)
        
        # Plot the ground track for each orbit with different label and color
        for j, segment in enumerate(segments):
            plt.plot(lon_deg[segment], lat_deg[segment], color=color, label=f"Orbit {i+1}" if j == 0 else "")

    # Plot ground stations if provided
    if ground_stations is not None:
        for ground_station in ground_stations:
            plt.scatter(ground_station[1], ground_station[0], s=50, color='Red', label="Ground Station")

    # Set plot limits and labels
    plt.ylim(-90, 90)
    plt.xlim(-180, 180)
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")

    # Show legend
    plt.legend()

    # Show the plot
    plt.show()

    # Save plot if save_path is provided
    if save_path:
        save_plot(fig, save_path)


def groundTrackVideo(r: np.ndarray, time: Union[np.ndarray, Time]) -> None:
    """
    Visualizes the ground track of an orbiting object using 3D animation.

    This function creates an interactive 3D visualization of the object's ground track, 
    displaying the path and the position of the object over time. The Earth is shown, 
    and the object's position is marked by a sphere. A line plot represents the object's 
    trajectory.

    Parameters
    ----------
    r : np.ndarray
        A (3, N) array representing the position of the orbiting object in meters 
        at each time step, where N is the number of time steps.
    time : Union[np.ndarray, astropy.time.Time]
        A 1D array or scalar representing the time steps. If it is an array, it should 
        correspond to GPS seconds, i.e., seconds since 1980-01-06 00:00:00 UTC.

    Returns
    -------
    None
        This function does not return any value. It displays an interactive animation 
        of the object's ground track.

    Notes
    -----
    - The function uses the `ipv` package for creating interactive 3D visualizations.
    - The object's position is shown as a magenta sphere and the trajectory as a white line.
    - The animation is controlled using the `ipv.animation_control` function.
    """
    ipvfig = ipv.figure(width=2000 / 2, height=1000 / 2)
    ipv.style.set_style_dark()
    ipv.style.box_off()
    ipv.style.axes_off()
    widgets = []
    widgets.append(drawEarth(time))
    widgets.append(
        ipv.scatter(
            r[:, 0, None],
            r[:, 1, None],
            r[:, 2, None],
            marker='sphere',
            color='magenta',
            size=10  # Increase the dot size (default is 1)
        )
    )
    # Line plot showing the path
    widgets.append(
        ipv.plot(
            r[:, 0],
            r[:, 1],
            r[:, 2],
            color='white',
            linewidth=1
        )
    )
    ipv.animation_control(widgets, sequence_length=len(time), interval=0)
    ipv.xyzlim(-10_000_000, 10_000_000)
    ipvfig.camera.position = (-2, 0, 0.2)
    ipvfig.camera.up = (0, 0, 1)
    ipv.show()


def check_numpy_array(variable: Union[np.ndarray, list]) -> str:
    """
    Checks if the input variable is a NumPy array, a list of NumPy arrays, or neither.

    Parameters
    ----------
    variable : Union[np.ndarray, list]
        The variable to check. It can either be a NumPy array or a list of NumPy arrays.

    Returns
    -------
    str
        Returns a string indicating the type of the variable:
        - "numpy array" if the variable is a single NumPy array,
        - "list of numpy array" if it is a list of NumPy arrays,
        - "not numpy" if it is neither.
    """
    if isinstance(variable, np.ndarray):
        return "numpy array"
    elif isinstance(variable, list):
        if len(variable) == 0:  # Handle empty list explicitly
            return "not numpy"
        elif all(isinstance(item, np.ndarray) for item in variable):
            return "list of numpy array"
    return "not numpy"


global_lower_bound = np.array([np.inf, np.inf, np.inf])
global_upper_bound = np.array([-np.inf, -np.inf, -np.inf])


def orbit_plot(r: Union[np.ndarray, List[np.ndarray]], 
               t: Optional[np.ndarray] = None, 
               limits: Optional[bool] = False, 
               title: str = '', 
               figsize: tuple = (7, 7), 
               save_path: Optional[str] = False, 
               frame: str = "gcrf", 
               show: bool = False, 
               legend: bool = False, 
               labels: Optional[List[str]] = None) -> tuple:
    """
    Creates a 2x2 subplot showing the orbit of an object(s) in different frames of reference.

    Parameters
    ----------
    r : Union[np.ndarray, List[np.ndarray]]
        Position of orbiting object(s) in meters. If a list of arrays is provided, 
        each array represents a separate orbit.
    t : Optional[np.ndarray], optional
        Time corresponding to each position in `r`. Required for certain frames 
        (e.g., ITRF, lunar frames). Default is an empty list.
    limits : Optional[bool], optional
        Whether to automatically set axis limits based on the data. Default is False, 
        in which case limits are calculated based on the data.
    title : str, optional
        Title of the plot. Default is an empty string.
    figsize : tuple, optional
        Size of the figure in inches. Default is (7, 7).
    save_path : Optional[str], optional
        Path to save the plot. If False, the plot is not saved. Default is False.
    frame : str, optional
        The reference frame to use for the plot (e.g., "gcrf", "itrf", "lunar", etc.). 
        Default is "gcrf".
    show : bool, optional
        Whether to display the plot. Default is False.
    legend : bool, optional
        Whether to include a legend on the plot. Default is False.
    labels : Optional[List[str]], optional
        Labels for the orbits when `legend` is True. Default is None.

    Returns
    -------
    tuple
        A tuple containing the figure object and a list of the four axes (ax1, ax2, ax3, ax4).

    Notes
    -----
    The plot includes four subplots:
        - ax1: xy-plane projection.
        - ax2: xz-plane projection.
        - ax3: yz-plane projection.
        - ax4: 3D scatter plot.
    
    The function supports multiple reference frames, and automatically adjusts the 
    axis limits and labeling based on the provided data and frame type.
    """
    
    def _make_scatter(fig, ax1, ax2, ax3, ax4, r, t, limits, title='', orbit_index='', num_orbits=1, frame=False, label=None):
        global global_lower_bound, global_upper_bound
        if np.size(t) < 1:
            if frame in ["itrf", "lunar", "lunar_fixed"]:
                raise ValueError("Need to provide t for itrf, lunar or lunar fixed frames")
            r_moon = np.atleast_2d(get_body("moon").position(Time("2000-1-1")))
        else:
            r_moon = get_body("moon").position(t).T

        # Dictionary of frame transformations and titles
        def get_main_category(frame):
            variant_mapping = {
                "gcrf": "gcrf",
                "gcrs": "gcrf",
                "itrf": "itrf",
                "itrs": "itrf",
                "lunar": "lunar",
                "lunar_fixed": "lunar",
                "lunar fixed": "lunar",
                "lunar_centered": "lunar",
                "lunar centered": "lunar",
                "lunarearthfixed": "lunar axis",
                "lunarearth": "lunar axis",
                "lunar axis": "lunar axis",
                "lunar_axis": "lunar axis",
                "lunaraxis": "lunar axis",
            }
            return variant_mapping.get(frame.lower())

        frame_transformations = {
            "gcrf": ("GCRF", None),
            "itrf": ("ITRF", gcrf_to_itrf),
            "lunar": ("Lunar Frame", gcrf_to_lunar_fixed),
            "lunar axis": ("Moon on x-axis Frame", gcrf_to_lunar),
        }

        # Check if the frame is in the dictionary, and set central_dot accordingly
        frame = get_main_category(frame)
        if frame in frame_transformations:
            title2, transform_func = frame_transformations[frame]
            if transform_func:
                r = transform_func(r, t)
                r_moon = transform_func(r_moon, t)
        else:
            raise ValueError("Unknown plot type provided. Accepted: gcrf, itrf, lunar, lunar fixed")

        x = r[:, 0] / RGEO
        y = r[:, 1] / RGEO
        z = r[:, 2] / RGEO
        xm = r_moon[:, 0] / RGEO
        ym = r_moon[:, 1] / RGEO
        zm = r_moon[:, 2] / RGEO
            
        if np.size(xm) > 1:
            gradient_colors = cm.Greys(np.linspace(0, .8, len(xm)))[::-1]
            blues = cm.Blues(np.linspace(.4, .9, len(xm)))[::-1]
        else:
            gradient_colors = "grey"
            blues = 'Blue'
        
        plot_settings = {
            "gcrf": ("blue", 50, 1, xm, ym, zm, gradient_colors),
            "itrf": ("blue", 50, 1, xm, ym, zm, gradient_colors),
            "lunar": ("grey", 25, 1.3, xm, ym, zm, blues),
            "lunar axis": ("blue", 50, 1, -xm, -ym, -zm, gradient_colors)
        }

        try:
            stn = plot_settings[frame]
        except KeyError:
            raise ValueError("Unknown plot type provided. Accepted: 'gcrf', 'itrf', 'lunar', 'lunar fixed'")

        if limits is False:
            lower_bound, upper_bound = find_smallest_bounding_cube(r / RGEO)
            lower_bound = lower_bound * 1.2
            upper_bound = upper_bound * 1.2
            global_lower_bound = np.minimum(global_lower_bound, lower_bound)
            global_upper_bound = np.maximum(global_upper_bound, upper_bound)

        if orbit_index == '':
            angle = 0
            dotcolors = cm.rainbow(np.linspace(0, 1, len(x)))
        else:
            angle = orbit_index * 10
            dotcolors = cm.rainbow(np.linspace(0, 1, num_orbits))[orbit_index]

        ax1.add_patch(plt.Circle((0, 0), stn[2], color='white', linestyle='dashed', fill=False))
        scatter1 = ax1.scatter(x, y, color=dotcolors, s=1, label=label)
        ax1.scatter(0, 0, color=stn[0], s=stn[1])
        if xm is not False:
            ax1.scatter(stn[3], stn[4], color=stn[6], s=5)
        ax1.set_aspect('equal')
        ax1.set_xlabel('x [GEO]')
        ax1.set_ylabel('y [GEO]')
        ax1.set_xlim((global_lower_bound[0], global_upper_bound[0]))
        ax1.set_ylim((global_lower_bound[1], global_upper_bound[1]))
        ax1.set_title(f'Frame: {title2}', color='white')

        if 'lunar' in frame:
            colors = ['red', 'green', 'purple', 'orange', 'cyan']
            for (point, pos), color in zip(lagrange_points_lunar_frame().items(), colors):
                if 'axis' in frame:
                    pass
                else:
                    pos[0] = pos[0] - LD / RGEO
                ax1.scatter(pos[0], pos[1], color=color, label=point)
                ax1.text(pos[0], pos[1], point, color=color)

        ax2.add_patch(plt.Circle((0, 0), stn[2], color='white', linestyle='dashed', fill=False))
        scatter2 = ax2.scatter(x, z, color=dotcolors, s=1, label=label)
        ax2.scatter(0, 0, color=stn[0], s=stn[1])
        if xm is not False:
            ax2.scatter(stn[3], stn[5], color=stn[6], s=5)
        ax2.set_aspect('equal')
        ax2.set_xlabel('x [GEO]')
        ax2.set_ylabel('z [GEO]')
        ax2.set_xlim((global_lower_bound[0], global_upper_bound[0]))
        ax2.set_ylim((global_lower_bound[2], global_upper_bound[2]))
        ax2.set_title(f'Frame: {title2}', color='white')

        ax3.add_patch(plt.Circle((0, 0), stn[2], color='white', linestyle='dashed', fill=False))
        scatter3 = ax3.scatter(y, z, color=dotcolors, s=1, label=label)
        ax3.scatter(0, 0, color=stn[0], s=stn[1])
        if xm is not False:
            ax3.scatter(stn[4], stn[5], color=stn[6], s=5)
        ax3.set_aspect('equal')
        ax3.set_xlabel('y [GEO]')
        ax3.set_ylabel('z [GEO]')
        ax3.set_xlim((global_lower_bound[1], global_upper_bound[1]))
        ax3.set_ylim((global_lower_bound[2], global_upper_bound[2]))
        ax3.set_title(f'Frame: {title2}', color='white')

        ax4.scatter(x, y, z, color=dotcolors, s=1, label=label)
        ax4.set_xlabel('x [GEO]')
        ax4.set_ylabel('y [GEO]')
        ax4.set_zlabel('z [GEO]')
        ax4.set_xlim((global_lower_bound[0], global_upper_bound[0]))
        ax4.set_ylim((global_lower_bound[1], global_upper_bound[1]))
        ax4.set_zlim((global_lower_bound[2], global_upper_bound[2]))
        ax4.set_title(f'{title} - {orbit_index} orbit', color='white')

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, projection='3d')

    if isinstance(r, np.ndarray):
        r = [r]
    if labels is None:
        labels = ["Orbit " + str(i) for i in range(len(r))]

    for i, data in enumerate(r):
        _make_scatter(fig, ax1, ax2, ax3, ax4, data, t, limits, title, i, len(r), frame, labels[i])

    if legend:
        ax2.legend()

    if save_path:
        fig.savefig(save_path)

    if show:
        plt.show()

    return fig, [ax1, ax2, ax3, ax4]


def globe_plot(r: np.ndarray, t: np.ndarray, limits: Optional[float] = False, title: str = '',
               figsize: Tuple[int, int] = (7, 8), save_path: Optional[str] = False, 
               el: int = 30, az: int = 0, scale: float = 1) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate a 3D globe plot showing the position of points in Earth-centered 
    coordinates. Optionally save the plot to a file.
    """
    # Scale the coordinates by RGEO
    x = r[:, 0] / RGEO
    y = r[:, 1] / RGEO
    z = r[:, 2] / RGEO

    # Set limits if not provided
    if limits is False:
        limits = np.nanmax(np.abs([x, y, z])) * 1.2

    # Load and scale Earth image
    earth_png = PILImage.open(find_file("earth", ext=".png"))
    earth_png = earth_png.resize((5400 // scale, 2700 // scale))
    bm = np.array(earth_png.resize([int(d) for d in earth_png.size])) / 256.

    # Generate mesh for globe surface
    lons = np.linspace(-180, 180, bm.shape[1]) * np.pi / 180
    lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi / 180
    mesh_x = np.outer(np.cos(lons), np.cos(lats)).T * EARTH_RADIUS / RGEO
    mesh_y = np.outer(np.sin(lons), np.cos(lats)).T * EARTH_RADIUS / RGEO
    mesh_z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T * EARTH_RADIUS / RGEO

    # Only plot visible points
    # x, y, z = x[visible], y[visible], z[visible]

    # Set color for the scatter plot
    dotcolors = plt.cm.rainbow(np.linspace(0, 1, len(x)))

    # Create the figure and 3D axis
    fig = plt.figure(dpi=100, figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('black')
    ax.tick_params(axis='both', colors='white')
    ax.grid(True, color='grey', linestyle='--', linewidth=0.5)
    ax.set_facecolor('black')  # Set plot background color to black

    # Plot the satellite positions and the Earth surface
    ax.scatter(x, y, z, color=dotcolors, s=1)
    ax.plot_surface(mesh_x, mesh_y, mesh_z, rstride=4, cstride=4, facecolors=bm, shade=False)

    # Set the view angle and axis limits
    ax.view_init(elev=el, azim=az)
    x_ticks = np.linspace(-limits, limits, 5)
    y_ticks = np.linspace(-limits, limits, 5)
    z_ticks = np.linspace(-limits, limits, 5)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)

    # Set axis labels with white color
    ax.set_xlabel('x [GEO]', color='white')
    ax.set_ylabel('y [GEO]', color='white')
    ax.set_zlabel('z [GEO]', color='white')

    # Set tick label colors to white
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    ax.set_aspect('equal')

    # Apply black background function (assuming `make_black` function exists)
    fig, ax = make_black(fig, ax)

    # Save the plot if save_path is provided
    if save_path:
        save_plot(fig, save_path)

    return fig, ax


def tracking_plot(r: np.ndarray, t: np.ndarray, ground_stations: Optional[np.ndarray] = None, 
                  limits: bool = False, title: str = '', figsize: tuple[int, int] = (12, 8), 
                  save_path: str = False, scale: float = 1) -> None:
    """
    Create a 3D tracking plot of satellite positions over time on Earth's surface.

    Parameters
    ----------
    r : numpy.ndarray or list of numpy.ndarray
        Satellite positions in GCRF coordinates. If a single numpy array, it represents the satellite's position vector over time. If a list of numpy arrays, it represents multiple satellite position vectors.

    t : numpy.ndarray
        Timestamps corresponding to the satellite positions.

    ground_stations : list of tuples, optional
        List of ground stations represented as (latitude, longitude) pairs. Default is None.

    limits : float or bool, optional
        The plot limits for x, y, and z axes. If a float, it sets the limits for all axes. If False, the limits are automatically determined based on the data. Default is False.

    title : str, optional
        Title for the plot. Default is an empty string.

    figsize : tuple, optional
        Figure size in inches (width, height). Default is (7, 8).

    save_path : str or bool, optional
        Path to save the plot as an image or PDF. If False, the plot is not saved. Default is False.

    scale : int, optional
        Scaling factor for the Earth's image. Default is 5.

    Returns
    -------
    matplotlib.figure.Figure
        The created tracking plot figure.

    Notes
    -----
    - The function supports plotting the positions of one or multiple satellites over time.
    - Ground station locations can be optionally displayed on the plot.
    - The limits parameter can be set to specify the plot's axis limits or automatically determined if set to False.
    - The frame parameter determines the coordinate frame for the satellite positions, "gcrf" (default) or "itrf".
    """
    
    # Validate input types
    if not isinstance(r, (np.ndarray, list)):
        raise TypeError(f"Expected numpy.ndarray or list of numpy.ndarray, got {type(r)}")
    if isinstance(r, list):
        if not all(isinstance(item, np.ndarray) for item in r):
            raise TypeError("If 'r' is a list, all elements must be numpy.ndarray")


    def _make_plot(r, t, ground_stations, limits, title, figsize, save_path, scale, orbit_index=''):
        lon, lat, height = groundTrack(r, t)
        lon[np.where(np.abs(np.diff(lon)) >= np.pi)] = np.nan
        lat[np.where(np.abs(np.diff(lat)) >= np.pi)] = np.nan

        x = r[:, 0] / RGEO
        y = r[:, 1] / RGEO
        z = r[:, 2] / RGEO
        
        # Handling limits
        if isinstance(limits, (int, float)):  # Custom limit
            limits_plot = limits
        elif limits is False:  # Auto limit based on the data
            limits_plot = np.nanmax(np.abs([x, y, z])) * 1.1
        else:  # If limits is an array (per satellite data)
            limits_plot = limits

        dotcolors = cm.rainbow(np.linspace(0, 1, len(x)))

        # Creating plot
        fig = plt.figure(dpi=100, figsize=figsize)
        fig.patch.set_facecolor('black')
        earth_png = PILImage.open(find_file("earth", ext=".png"))
        earth_png = earth_png.resize((5400 // scale, 2700 // scale))
        
        # 1st subplot (longitude-latitude plot)
        ax = fig.add_subplot(2, 3, (1, 2))
        ax.imshow(earth_png, extent=[-180, 180, -90, 90])
        ax.plot(np.rad2deg(lon), np.rad2deg(lat))
        if ground_stations is not None:
            for ground_station in ground_stations:
                ax.scatter(ground_station[1], ground_station[0], s=15, color='DarkRed')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('longitude [degrees]', color='white')
        ax.set_ylabel('latitude [degrees]', color='white')
        ax.set_title(title, color='white')
        ax.tick_params(axis='both', colors='white')
        ax.set_aspect('equal')

        # 2nd subplot (longitude-latitude plot zoomed)
        ax = fig.add_subplot(2, 3, 3)
        ax.imshow(earth_png, extent=[-180, 180, -90, 90])
        ax.plot(np.rad2deg(lon), np.rad2deg(lat))
        if ground_stations is not None:
            for ground_station in ground_stations:
                ax.scatter(ground_station[1], ground_station[0], s=15, color='DarkRed')
        ax.set_xlim(-150, -60)
        ax.set_ylim(0, 90)
        ax.set_xlabel('longitude [degrees]', color='white')
        ax.set_ylabel('latitude [degrees]', color='white')
        ax.tick_params(axis='both', colors='white')
        ax.set_aspect('equal')

        # 3rd subplot (XY plot)
        ax = fig.add_subplot(2, 3, 4)
        ax.scatter(0, 0, color='blue', s=(100 * EARTH_RADIUS / RGEO)**2)
        ax.scatter(x, y, color=dotcolors, s=1)
        ax.set_xlim([-limits_plot, limits_plot])
        ax.set_ylim([-limits_plot, limits_plot])
        ax.set_aspect('equal')  # aspect ratio is 1:1:1 in data space
        ax.set_xlabel('x [GEO]', color='white')
        ax.set_ylabel('y [GEO]', color='white')
        ax.set_title('XY', color='white')
        ax.tick_params(axis='both', colors='white')
        ax.set_facecolor('black')
        ax.grid(True, color='grey', linestyle='--', linewidth=0.5)

        # 4th subplot (XZ plot)
        ax = fig.add_subplot(2, 3, 5)
        ax.scatter(0, 0, color='blue', s=(100 * EARTH_RADIUS / RGEO)**2)
        ax.scatter(x, z, color=dotcolors, s=1)
        ax.set_xlim([-limits_plot, limits_plot])
        ax.set_ylim([-limits_plot, limits_plot])
        ax.set_aspect('equal')  # aspect ratio is 1:1:1 in data space
        ax.set_xlabel('x [GEO]', color='white')
        ax.set_ylabel('z [GEO]', color='white')
        ax.set_title('XZ', color='white')
        ax.tick_params(axis='both', colors='white')
        ax.set_facecolor('black')
        ax.grid(True, color='grey', linestyle='--', linewidth=0.5)

        # 5th subplot (YZ plot)
        ax = fig.add_subplot(2, 3, 6)
        ax.scatter(0, 0, color='blue', s=(100 * EARTH_RADIUS / RGEO)**2)
        ax.scatter(y, z, color=dotcolors, s=1)
        ax.set_xlim([-limits_plot, limits_plot])
        ax.set_ylim([-limits_plot, limits_plot])
        ax.set_aspect('equal')  # aspect ratio is 1:1:1 in data space
        ax.set_xlabel('y [GEO]', color='white')
        ax.set_ylabel('z [GEO]', color='white')
        ax.set_title('YZ', color='white')
        ax.tick_params(axis='both', colors='white')
        ax.set_facecolor('black')
        ax.grid(True, color='grey', linestyle='--', linewidth=0.5)

        fig, ax = make_black(fig, ax)
        plt.tight_layout()
        
        if save_path:
            save_plot(fig, save_path)
        return fig

    input_type = check_numpy_array(r)
    fig = None
    if input_type == "numpy array":
        fig = _make_plot(
            r, t, ground_stations=ground_stations,
            limits=limits, title=title, figsize=figsize,
            save_path=save_path, scale=scale)

    if input_type == "list of numpy array":
        for i, row in enumerate(r):
            if isinstance(limits, (int, float)):  # Custom limit for each orbit
                limits_plot = limits
            else:  # Calculate the limit dynamically based on the satellite data
                limits_plot = np.nanmax([np.nanmax(norm(row) / RGEO) for row in r]) * 1.2
            fig = _make_plot(
                row, t, ground_stations=ground_stations,
                limits=limits_plot, title=title, figsize=figsize,
                save_path=save_path, scale=scale, orbit_index=i
            )
    return fig


save_plot_to_pdf_call_count = 0


def save_plot_to_pdf(figure: plt.Figure, pdf_path: str) -> None:
    """
    Save a Matplotlib figure as a PNG embedded in a PDF file.

    This function saves the figure as a temporary PNG image in memory and 
    then embeds it into a PDF file. If the specified PDF already exists, 
    the figure is appended to it. Otherwise, a new PDF file is created.

    Parameters:
        figure (matplotlib.figure.Figure): The figure object to be saved.
        pdf_path (str): The path to the PDF file where the figure will be saved.

    Returns:
        None
    """
    global save_plot_to_pdf_call_count
    save_plot_to_pdf_call_count += 1
    
    # Expand user directory if ~ is in the path
    if pdf_path.startswith('~'):
        pdf_path = os.path.expanduser(pdf_path)
    
    # Generate a temporary PDF path by replacing the original extension
    if '.' in pdf_path:
        temp_pdf_path = re.sub(r"\.[^.]+$", "_temp.pdf", pdf_path)
    else:
        temp_pdf_path = f"{pdf_path}_temp.pdf"
    
    # Save the figure as a PNG in-memory using BytesIO
    png_buffer = io.BytesIO()
    figure.savefig(png_buffer, format='png', dpi=300, bbox_inches='tight')
    
    # Rewind the buffer to the beginning
    png_buffer.seek(0)
    
    # Open the in-memory PNG using PIL
    png_image = PILImage.open(png_buffer)
    
    # Create the temporary PDF with the PNG image
    with PdfPages(temp_pdf_path) as pdf:
        # Create a new figure and axis to display the image
        img_fig, img_ax = plt.subplots()
        img_ax.imshow(png_image)
        img_ax.axis('off')  # Hide axis for clean image display
        # Save the image as a page in the PDF
        pdf.savefig(img_fig, dpi=300, bbox_inches='tight')
    
    # If the PDF already exists, merge the new page
    if os.path.exists(pdf_path):
        merger = PdfMerger()
        with open(pdf_path, "rb") as main_pdf, open(temp_pdf_path, "rb") as temp_pdf:
            merger.append(main_pdf)
            merger.append(temp_pdf)
            with open(pdf_path, "wb") as merged_pdf:
                merger.write(merged_pdf)
        os.remove(temp_pdf_path)
    else:
        # If the PDF doesn't exist, rename the temporary PDF to the desired path
        os.rename(temp_pdf_path, pdf_path)
    
    # Close all figures to release resources
    plt.close(figure)
    plt.close(img_fig)
    
    # Print the success message with the call count
    print(f"Saved figure {save_plot_to_pdf_call_count} to {pdf_path}")


def save_plot(figure: plt.Figure, save_path: str, dpi: int = 200) -> None:
    """
    Save a Python figure as a PNG image.

    Parameters:
        figure (matplotlib.figure.Figure): The figure object to be saved.
        save_path (str): The file path where the PNG image will be saved.
        dpi (int, optional): The resolution of the saved image. Default is 200.

    Returns:
        None
    """
    if save_path.lower().endswith('.pdf'):
        save_plot_to_pdf(figure, save_path)
        return
    try:
        base_name, extension = os.path.splitext(save_path)
        if extension.lower() != '.png':
            save_path = base_name + '.png'
        # Save the figure as a PNG image
        figure.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(figure)  # Close the figure to release resources
        print(f"Figure saved at: {save_path}")
    except Exception as e:
        print(f"Error occurred while saving the figure: {e}")
