from ..constants import RGEO, EARTH_RADIUS, MOON_RADIUS
from ..coordinates import gcrf_to_itrf, gcrf_to_lunar, gcrf_to_lunar_fixed
from .plotutils import valid_orbits, save_plot
from ..time import Time
from ..compute import find_smallest_bounding_cube
from ssapy import get_body

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def orbit_plot_xyxz(r, t=None, title='', figsize=(7, 7), save_path=False, frame="gcrf", show=False, c='black', pad=1):
    """
    Plots the trajectory of an orbiting body in different reference frames.

    The function generates four subplots:
    1. XY plane projection
    2. XZ plane projection
    3. YZ plane projection
    4. 3D representation of the orbit

    Parameters:
    -----------
    r : numpy array or list of numpy arrays
        Position vector(s) of the orbiting object(s).
        If a single numpy array, it should have shape (N, 3) representing [x, y, z] positions.
        If a list of numpy arrays, each element should have shape (Ni, 3) for multiple trajectories.

    t : numpy array, list, or None, optional
        Time array corresponding to `r`.
        If `r` is a list of arrays, `t` should be a list of corresponding time arrays.

    title : str, optional
        Title of the plot.

    figsize : tuple, optional
        Figure size in inches. Default is (7, 7).

    save_path : str or bool, optional
        Path to save the figure. If False, the plot is not saved.

    frame : str, optional
        Reference frame for the plot. Options:
        - "gcrf" (Geocentric Celestial Reference Frame)
        - "itrf" (International Terrestrial Reference Frame)
        - "lunar" (Lunar-centered frame)
        - "lunar axis" (Moon on x-axis frame)

    show : bool, optional
        If True, displays the plot.

    Returns:
    --------
    None

    Author:
    -------
    Travis Yeager (yeager7@llnl.gov)
    """
    from ..orbital_mechanics import lagrange_points_lunar_frame, lagrange_points_lunar_fixed_frame

    r, t = valid_orbits(r, t)

    if 'w' in c:
        textcolor = 'black'
        plotcolor = 'white'
    elif 'b' in c:
        textcolor = 'white'
        plotcolor = 'black'

    fig = plt.figure(dpi=100, figsize=figsize, facecolor=plotcolor)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    bounds = {"lower": np.array([np.inf, np.inf, np.inf]), "upper": np.array([-np.inf, -np.inf, -np.inf])}

    if any(np.max(np.linalg.norm(xyz, axis=-1)) >= 0.95 * RGEO for xyz in r):
        unit_conversion = RGEO
        unit_label = 'GEO'
    else:
        unit_conversion = 1e3
        unit_label = 'km'
    for orbit_index in range(len(r)):
        xyz = r[orbit_index]
        t_current = t[orbit_index]

        r_moon = get_body("moon").position(t_current).T
        r_earth = np.zeros(np.shape(r_moon))

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
                xyz = transform_func(xyz, t_current)
                r_moon = transform_func(r_moon, t_current)
                r_earth = transform_func(r_earth, t_current)
        else:
            raise ValueError("Unknown plot type provided. Accepted: gcrf, itrf, lunar, lunar fixed")

        xyz = xyz / unit_conversion
        r_moon = r_moon / unit_conversion
        r_earth = r_earth / unit_conversion

        lower_bound_temp, upper_bound_temp = find_smallest_bounding_cube(xyz, pad=pad)
        bounds["lower"] = np.minimum(bounds["lower"], lower_bound_temp)
        bounds["upper"] = np.maximum(bounds["upper"], upper_bound_temp)

        if np.size(r_moon[:, 0]) > 1:
            grey_colors = cm.Greys(np.linspace(0, .8, len(r_moon[:, 0])))[::-1]
            blues = cm.Blues(np.linspace(.4, .9, len(r_moon[:, 0])))[::-1]
        else:
            grey_colors = "grey"
            blues = 'Blue'
        plot_settings = {
            "gcrf": {
                "primary_color": "blue",
                "primary_size": (EARTH_RADIUS / unit_conversion),
                "secondary_x": r_moon[:, 0],
                "secondary_y": r_moon[:, 1],
                "secondary_z": r_moon[:, 2],
                "secondary_color": grey_colors,
                "secondary_size": (MOON_RADIUS / unit_conversion)
            },
            "itrf": {
                "primary_color": "blue",
                "primary_size": (EARTH_RADIUS / unit_conversion),
                "secondary_x": r_moon[:, 0],
                "secondary_y": r_moon[:, 1],
                "secondary_z": r_moon[:, 2],
                "secondary_color": grey_colors,
                "secondary_size": (MOON_RADIUS / unit_conversion)
            },
            "lunar": {
                "primary_color": "grey",
                "primary_size": (MOON_RADIUS / unit_conversion),
                "secondary_x": r_earth[:, 0],
                "secondary_y": r_earth[:, 1],
                "secondary_z": r_earth[:, 2],
                "secondary_color": blues,
                "secondary_size": (EARTH_RADIUS / unit_conversion)
            },
            "lunar axis": {
                "primary_color": "blue",
                "primary_size": (EARTH_RADIUS / unit_conversion),
                "secondary_x": r_moon[:, 0],
                "secondary_y": r_moon[:, 1],
                "secondary_z": r_moon[:, 2],
                "secondary_color": grey_colors,
                "secondary_size": (MOON_RADIUS / unit_conversion)
            }
        }
        try:
            stn = plot_settings[frame]
        except KeyError:
            raise ValueError("Unknown plot type provided. Accepted: 'gcrf', 'itrf', 'lunar', 'lunar fixed'")

        if len(r) == 1:
            scatter_dot_colors = cm.rainbow(np.linspace(0, 1, len(xyz[:, 0])))
        else:
            scatter_dot_colors = cm.rainbow(np.linspace(0, 1, len(r)))[orbit_index]

        ax1.scatter(xyz[:, 0], xyz[:, 1], color=scatter_dot_colors, s=1)
        ax1.add_patch(plt.Circle(xy=(0, 0), radius=1, color=textcolor, linestyle='dashed', fill=False))  # Circle marking GEO
        ax1.add_patch(plt.Circle(xy=(0, 0), radius=stn['primary_size'], color=stn['primary_color'], linestyle='dashed', fill=False))  # Circle marking EARTH or MOON
        if r_moon[:, 0] is not False:
            ax1.scatter(stn['secondary_x'], stn['secondary_y'], color=stn['secondary_color'], s=stn['secondary_size'])
        ax1.set_aspect('equal')
        ax1.set_xlabel(f'x [{unit_label}]', color=textcolor)
        ax1.set_ylabel(f'y [{unit_label}]', color=textcolor)
        ax1.set_title(f'Frame: {title2}', color=textcolor)
        if 'lunar' in frame:
            lagrange_points = lagrange_points_lunar_frame().items()
            if 'fixed' in frame:
                lagrange_points = lagrange_points_lunar_fixed_frame().items()
            for (point, pos) in lagrange_points:
                pos = pos / unit_conversion
                if bounds["lower"][0] <= pos[0] <= bounds["upper"][0] and bounds["lower"][1] <= pos[1] <= bounds["upper"][1]:
                    ax1.scatter(pos[0], pos[1], color=textcolor, label=point, s=10)
                    ax1.text(pos[0], pos[1], point, color=textcolor)

        ax2.scatter(xyz[:, 0], xyz[:, 2], color=scatter_dot_colors, s=1)
        ax2.add_patch(plt.Circle(xy=(0, 0), radius=1, color=textcolor, linestyle='dashed', fill=False))  # Circle marking GEO
        ax2.add_patch(plt.Circle(xy=(0, 0), radius=stn['primary_size'], color=stn['primary_color'], linestyle='dashed', fill=False))  # Circle marking EARTH or MOON
        if r_moon[:, 0] is not False:
            ax2.scatter(stn['secondary_x'], stn['secondary_z'], color=stn['secondary_color'], s=stn['secondary_size'])
        ax2.set_aspect('equal')
        ax2.set_xlabel(f'x [{unit_label}]', color=textcolor)
        ax2.set_ylabel(f'y [{unit_label}]', color=textcolor)
        ax2.yaxis.tick_right()  # Move y-axis ticks to the right
        ax2.yaxis.set_label_position("right")  # Move y-axis label to the right
        ax2.set_title(f'{title}', color=textcolor)
        if 'lunar' in frame:
            lagrange_points = lagrange_points_lunar_frame().items()
            if 'fixed' in frame:
                lagrange_points = lagrange_points_lunar_fixed_frame().items()
            for (point, pos) in lagrange_points:
                pos = pos / unit_conversion
                if bounds["lower"][0] <= pos[0] <= bounds["upper"][0] and bounds["lower"][2] <= pos[2] <= bounds["upper"][2]:
                    ax2.scatter(pos[0], pos[2], color=textcolor, label=point, s=10)
                    ax2.text(pos[0], pos[2], point, color=textcolor)

    ax1.set_xlim(bounds["lower"][0], bounds["upper"][0])
    ax1.set_ylim(bounds["lower"][1], bounds["upper"][1])

    ax2.set_xlim(bounds["lower"][0], bounds["upper"][0])
    ax2.set_ylim(bounds["lower"][2], bounds["upper"][2])

    for ax in [ax1, ax2]:
        ax.set_facecolor(plotcolor)
        ax.tick_params(axis='both', colors=textcolor)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(textcolor)
        for spine in ax.spines.values():
            spine.set_edgecolor(textcolor)

    if save_path:
        save_plot(fig, save_path)
    if show:
        plt.show()
    plt.close()
    return fig, [ax1, ax2]
