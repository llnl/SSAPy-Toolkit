import matplotlib.pyplot as plt
from PIL import Image as PILImage

import numpy as np
from typing import Tuple, Optional

from ssapy.utils import find_file
from .plotutils import make_black, make_white, save_plot, valid_orbits
from ..constants import RGEO, EARTH_RADIUS


def globe_plot(r: np.ndarray, t: np.ndarray, limits: Optional[float] = False, title: str = '',
               c='black', figsize: Tuple[int, int] = (7, 8), save_path: Optional[str] = False,
               el: int = 30, az: int = 0, scale: float = 1, fontsize=18) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate a 3D globe plot showing the position of points in Earth-centered
    coordinates. Optionally save the plot to a file.

    Author:
    -------
    Travis Yeager (yeager7@llnl.gov)
    """

    r, t = valid_orbits(r, t)

    if c in ('black', 'b'):
        plotcolor = 'black'
        textcolor = 'white'
    elif c in ('white', 'w'):
        plotcolor = 'white'
        textcolor = 'black'
    else:
        plotcolor = 'white'
        textcolor = 'black'

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

    # Create the figure and 3D axis
    fig = plt.figure(dpi=100, figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor(plotcolor)
    ax.tick_params(axis='both', colors=textcolor)
    ax.grid(True, color='grey', linestyle='--', linewidth=0.5)
    ax.set_facecolor(plotcolor)  # Set plot background color to black
    ax.plot_surface(mesh_x, mesh_y, mesh_z, rstride=4, cstride=4, facecolors=bm, shade=False)

    # Plot the satellite positions and the Earth surface
    if len(r) == 1:
        # Set color for the scatter plot
        cmap = plt.cm.rainbow(np.linspace(0, 1, len(r[0])))
    else:
        cmap_vals = np.linspace(0, 1, len(r))
        cmap = [plt.cm.rainbow(val) for val in cmap_vals]

    max_extent = 0  # initialize limit tracker

    for i, ri in enumerate(r):
        color = cmap[i] if len(r) > 1 else cmap

        ri = ri / RGEO
        current_max = np.nanmax(np.abs(ri))
        max_extent = max(max_extent, current_max)  # track the largest extent
        ax.scatter(ri[:, 0], ri[:, 1], ri[:, 2], color=color, s=1)

    if limits is False:
        limits = max_extent * 1.2

    # Set the view angle and axis limits
    ax.view_init(elev=el, azim=az)
    if limits < 1:
        x_ticks = np.linspace(-limits, limits, 5)
        y_ticks = np.linspace(-limits, limits, 5)
        z_ticks = np.linspace(-limits, limits, 5)
    else:
        int_limit = int(np.ceil(limits))
        x_ticks = [-int_limit, 0, int_limit]
        y_ticks = [-int_limit, 0, int_limit]
        z_ticks = [-int_limit, 0, int_limit]

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Set axis labels with white color
    ax.set_xlabel('x [GEO]', color=textcolor, fontsize=fontsize)
    ax.set_ylabel('y [GEO]', color=textcolor, fontsize=fontsize)
    ax.set_zlabel('z [GEO]', color=textcolor, fontsize=fontsize)
    if title:
        ax.set_title(title, color=textcolor, fontsize=fontsize)

    # Set tick label colors to white
    ax.tick_params(axis='x', colors=textcolor, labelsize=fontsize)
    ax.tick_params(axis='y', colors=textcolor, labelsize=fontsize)
    ax.tick_params(axis='z', colors=textcolor, labelsize=fontsize)
    ax.set_aspect('equal')

    # Apply black background function (assuming `make_black` function exists)
    if c in ('black', 'b'):
        fig, ax = make_black(fig, ax)
    elif c in ('white', 'w'):
        fig, ax = make_white(fig, ax)

    # Save the plot if save_path is provided
    if save_path:
        save_plot(fig, save_path)

    return fig, ax
