import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image as PILImage

from ssapy import groundTrack
from ssapy.utils import find_file, norm
from ..constants import RGEO, EARTH_RADIUS
from .plotutils import save_plot, make_black, check_numpy_array


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

    Author:
    -------
    Travis Yeager (yeager7@llnl.gov)
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
