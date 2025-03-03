######################################################################
# COLLECTION OF ALL PLOTTING AND MEDIA
######################################################################
# flake8: noqa: E501
import numpy as np
from ssapy.body import get_body
from ..constants import RGEO, EARTH_MU, MOON_MU
from ..orbital_mechanics import calculate_orbital_elements
from ..time import Time
from .plotutils import make_black, make_white, save_plot

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize

from typing import List, Tuple, Optional, Union

lunar_semi_major = 384399000  # m


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

    Author:
    -------
    Travis Yeager (yeager7@llnl.gov)
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


def koe_2dhist(stable_data, title: str = "Initial orbital elements of\n1 year stable cislunar orbits", 
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

    Author:
    -------
    Travis Yeager (yeager7@llnl.gov)
    """

    # Validate angle data ranges
    if not (np.all((0 <= stable_data.i) & (stable_data.i <= 2 * np.pi))):
        raise ValueError("Inclination (`i`) must be in the range [0, 2π] radians.")
    if not (np.all((0 <= stable_data.ta) & (stable_data.ta <= 2 * np.pi))):
        raise ValueError("True Anomaly (`ta`) must be in the range [0, 2π] radians.")

    if logscale or logscale == 'log':
        norm = LogNorm(limits[0], limits[1])
    else:
        norm = Normalize(limits[0], limits[1])

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
        cNorm = Normalize(vmin=colormin, vmax=colormax)
    elif colorscale == 'log':
        cNorm = LogNorm(vmin=colormin, vmax=colormax)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm)
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
        cNorm = Normalize(vmin=min(cs), vmax=max(cs))
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm)
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
        r_moon[:, 0] = r_moon[0] / RGEO
        r_moon[:, 1] = r_moon[1] / RGEO
        r_moon[:, 2] = r_moon[2] / RGEO
        dotcolors = cm.rainbow(np.linspace(0, 1, len(x)))

        # Creating plot
        plt.subplot(1, 3, 1)
        plt.scatter(x, y, color=dotcolors, s=1)
        plt.scatter(0, 0, color="blue", s=50)
        plt.scatter(r_moon[:, 0], r_moon[:, 1], color="grey", s=5)
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
        plt.scatter(r_moon[:, 0], r_moon[:, 2], color="grey", s=5)
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
        plt.scatter(r_moon[:, 1], r_moon[:, 2], color="grey", s=5)
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

