from ..constants import RGEO, EARTH_RADIUS, MOON_RADIUS
from ..coordinates import gcrf_to_itrf, gcrf_to_lunar, gcrf_to_lunar_fixed
from .plotutils import check_numpy_array, check_type, save_plot
from ..time import Time
from ..compute import find_smallest_bounding_cube
from ssapy import get_body

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def orbit_plot_xy(r, t=None, title='', figsize=(7, 7), save_path=False, frame="gcrf", show=False, c='black', pad=1):
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

    input_type = check_numpy_array(r)
    t_type = check_type(t)

    if input_type == "numpy array":
        num_orbits = 1
        r = [r]

    if input_type == "list of numpy array":
        num_orbits = len(r)

    if 'w' in c:
        textcolor = 'black'
        plotcolor = 'white'
    elif 'b' in c:
        textcolor = 'white'
        plotcolor = 'black'

    fig = plt.figure(dpi=100, figsize=figsize, facecolor=plotcolor)
    ax1 = fig.add_subplot(1, 1, 1)

    bounds = {"lower": np.array([np.inf, np.inf, np.inf]), "upper": np.array([-np.inf, -np.inf, -np.inf])}

    # Check if all arrays in `r` are the same shape
    for orbit_index in range(num_orbits):
        xyz = r[orbit_index]

        if t_type is None:
            if frame == "gcrf":
                r_moon = np.atleast_2d(get_body("moon").position(Time("2000-1-1")))
            else:
                raise ValueError("Need to provide t or list of t for each orbit in itrf, lunar or lunar fixed frames")
        else:
            if frame == "gcrf":
                if t_type == "Single array or np.ndarray":
                    t_current = t
                elif t_type == "List of non-arrays" or t_type == "List of arrays":
                    t_current = max(t, key=len)
            else:
                if input_type == "numpy array":
                    t_current = t
                    # Single array case
                    if np.shape(t)[0] != np.shape(r)[1]:
                        raise ValueError("For a single numpy array 'r', 't' must be a 1D array of the same length as the first dimension of 'r'.")

                elif input_type == "list of numpy array":
                    same_shape = all(np.shape(arr) == np.shape(r[0]) for arr in r)
                    if same_shape:
                        if t_type == "Single array or np.ndarray":
                            t_current = t
                        elif t_type == "List of non-arrays" or t_type == "List of arrays":
                            t_current = max(t, key=len)
                        # Single `t` array is allowed
                        if len(t_current) != len(xyz):
                            raise ValueError("When 'r' is a list of arrays with the same shape, 't' must be a single 1D array matching the length of the first dimension of the arrays in 'r'.")
                    else:
                        # `t` must be a list of 1D arrays
                        if t_type == "Single array or np.ndarray":
                            raise ValueError("When 'r' is a list of differing size numpy arrays, 't' must be a list of 1D arrays of equal length to the corresponding arrays in 'r'.")
                        elif t_type == "List of non-arrays" or t_type == "List of arrays":
                            if len(xyz) == len(t[orbit_index]):
                                t_current = t[orbit_index]
                            else:
                                print(f"length of t: {len(t_current)} and r: {len(xyz)}")
                                raise ValueError(f"'t' must be a 1D array matching the length of the first dimension of 'r[{orbit_index}]'.")

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

        if max(np.linalg.norm(xyz, axis=-1) >= .95 * RGEO):
            unit_conversion = RGEO
            unit_label = 'GEO'
        else:
            unit_conversion = 1e3
            unit_label = 'km'

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

        if input_type == "numpy array":
            scatter_dot_colors = cm.rainbow(np.linspace(0, 1, len(xyz[:, 0])))
        else:
            scatter_dot_colors = cm.rainbow(np.linspace(0, 1, num_orbits))[orbit_index]

        ax1.scatter(xyz[:, 0], xyz[:, 1], color=scatter_dot_colors, s=1)
        ax1.add_patch(plt.Circle(xy=(0, 0), radius=1, color=textcolor, linestyle='dashed', fill=False))  # Circle marking GEO
        ax1.add_patch(plt.Circle(xy=(0, 0), radius=stn['primary_size'], color=stn['primary_color'], linestyle='dashed', fill=False))  # Circle marking EARTH or MOON
        if r_moon[:, 0] is not False:
            ax1.scatter(stn['secondary_x'], stn['secondary_y'], color=stn['secondary_color'], s=stn['secondary_size'])
        ax1.set_aspect('equal')
        ax1.set_xlabel(f'x [{unit_label}]', color=textcolor)
        ax1.set_ylabel(f'y [{unit_label}]', color=textcolor)
        ax1.set_title(f'{title}\nFrame: {title2}', color=textcolor)
        if 'lunar' in frame:
            lagrange_points = lagrange_points_lunar_frame().items()
            if 'fixed' in frame:
                lagrange_points = lagrange_points_lunar_fixed_frame().items()
            for (point, pos) in lagrange_points:
                pos = pos / unit_conversion
                if bounds["lower"][0] <= pos[0] <= bounds["upper"][0] and bounds["lower"][1] <= pos[1] <= bounds["upper"][1]:
                    ax1.scatter(pos[0], pos[1], color=textcolor, label=point, s=10)
                    ax1.text(pos[0], pos[1], point, color=textcolor)

    ax1.set_xlim(bounds["lower"][0], bounds["upper"][0])
    ax1.set_ylim(bounds["lower"][1], bounds["upper"][1])

    for ax in [ax1]:
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
    return fig, [ax1]
