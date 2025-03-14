from .plotutils import check_numpy_array, check_type

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from matplotlib.legend_handler import HandlerBase
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.font_manager as font_manager

import numpy as np

from ssapy import get_body
from ..coordinates import gcrf_to_lunar_fixed
from ..constants import RGEO, EARTH_RADIUS, MOON_RADIUS
from ..compute import find_smallest_bounding_cube
from ..orbital_mechanics import lagrange_points_lunar_fixed_frame
from .plotutils import save_plot


class GradientLineHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Number of segments for the gradient
        num_segments = 10
        # X-coordinates from the left (xdescent) to the right (xdescent + width)
        x = np.linspace(xdescent, xdescent + width, num_segments + 1)
        # Y-coordinate centered vertically in the legend box
        y = ydescent + height / 2
        # Create line segments: each segment is a tuple of ((x_start, y), (x_end, y))
        segments = [((x[i], y), (x[i + 1], y)) for i in range(num_segments)]
        # Assign rainbow colors to each segment
        colors = cm.rainbow(np.linspace(0, 1, num_segments))
        # Create a LineCollection with these segments and colors
        lc = LineCollection(segments, colors=colors, linewidth=2)
        lc.set_transform(trans)  # Apply the transformation for legend positioning
        return [lc]


def cislunar_plot(r, t=None, figsize=(8, 8), fontsize=12, save_path=False, show=False, title=None, c='white'):
    input_type = check_numpy_array(r)
    t_type = check_type(t)

    if title is None:
        title = "Frame: GCRF"
    else:
        title = f"{title}\nFrame: GCRF"
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=100, subplot_kw={'projection': '3d'}, facecolor=plotcolor)

    bounds_gcrf = {"lower": np.array([np.inf, np.inf, np.inf]), "upper": np.array([-np.inf, -np.inf, -np.inf])}
    bounds_lunar = {"lower": np.array([np.inf, np.inf, np.inf]), "upper": np.array([-np.inf, -np.inf, -np.inf])}

    # Check if all arrays in `r` are the same shape

    for orbit_index in range(num_orbits):
        xyz = r[orbit_index]

        if t_type is None:
            raise ValueError("Need to provide t or list of t for each orbit in itrf, lunar or lunar fixed frames")
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

        if max(np.linalg.norm(xyz, axis=-1) >= .95 * RGEO):
            unit_conversion = RGEO
            unit_label = 'GEO'
        else:
            unit_conversion = 1e3
            unit_label = 'km'

        xyz_lunar = gcrf_to_lunar_fixed(xyz, t_current) / unit_conversion
        # r_moon_lunar = gcrf_to_lunar_fixed(r_moon, t_current) / unit_conversion
        r_earth_lunar = gcrf_to_lunar_fixed(r_earth, t_current) / unit_conversion
        xyz = xyz / unit_conversion
        r_moon = r_moon / unit_conversion
        r_earth = r_earth / unit_conversion

        lower_bound_temp, upper_bound_temp = find_smallest_bounding_cube(xyz, pad=1)
        lower_bound_lunar_temp, upper_bound_lunar_temp = find_smallest_bounding_cube(xyz_lunar, pad=1)
        bounds_gcrf["lower"] = np.minimum(bounds_gcrf["lower"], lower_bound_temp)
        bounds_gcrf["upper"] = np.maximum(bounds_gcrf["upper"], upper_bound_temp)
        bounds_lunar["lower"] = np.minimum(bounds_lunar["lower"], lower_bound_lunar_temp)
        bounds_lunar["upper"] = np.maximum(bounds_lunar["upper"], upper_bound_lunar_temp)

        mask_gcrf = (
            (r_moon[:, 0] >= bounds_gcrf["lower"][0]) & (r_moon[:, 0] <= bounds_gcrf["upper"][0])
            & (r_moon[:, 1] >= bounds_gcrf["lower"][1]) & (r_moon[:, 1] <= bounds_gcrf["upper"][1])
            & (r_moon[:, 2] >= bounds_gcrf["lower"][2]) & (r_moon[:, 2] <= bounds_gcrf["upper"][2])
        )

        mask_lunar = (
            (r_earth_lunar[:, 0] >= bounds_lunar["lower"][0]) & (r_earth_lunar[:, 0] <= bounds_lunar["upper"][0])
            & (r_earth_lunar[:, 1] >= bounds_lunar["lower"][1]) & (r_earth_lunar[:, 1] <= bounds_lunar["upper"][1])
            & (r_earth_lunar[:, 2] >= bounds_lunar["lower"][2]) & (r_earth_lunar[:, 2] <= bounds_lunar["upper"][2])
        )

        if np.size(r_moon[:, 0]) > 1:
            grey_colors = cm.Greys(np.linspace(0, .8, len(r_moon[:, 0])))[::-1][mask_gcrf]
        else:
            grey_colors = "grey"

        if np.size(r_moon[:, 0]) > 1:
            blue_colors = cm.Blues(np.linspace(0.2, .8, len(r_moon[:, 0])))[::-1][mask_lunar]
        else:
            blue_colors = "blue"

        if input_type == "numpy array":
            scatter_dot_colors = cm.rainbow(np.linspace(0, 1, len(xyz[:, 0])))
        else:
            scatter_dot_colors = cm.rainbow(np.linspace(0, 1, num_orbits))[orbit_index]

        # Create a 3d sphere of the Earth and Moon
        u = np.linspace(0, 2 * np.pi, 180)
        v = np.linspace(-np.pi / 2, np.pi / 2, 180)

        ax1.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=scatter_dot_colors, s=1)
        # ax1.add_patch(plt.Circle(xy=(xyz[:, 0], xyz[:, 1]), radius=(EARTH_RADIUS / unit_conversion), color='blue', linestyle='dashed', fill=False))
        # ax1.add_patch(plt.Circle(xy=(xyz[:, 1], xyz[:, 2]), radius=(EARTH_RADIUS / unit_conversion), color="blue", linestyle='dashed', fill=False))
        mesh_x = np.outer(np.cos(u), np.cos(v)).T * (EARTH_RADIUS / unit_conversion) + 0
        mesh_y = np.outer(np.sin(u), np.cos(v)).T * (EARTH_RADIUS / unit_conversion) + 0
        mesh_z = np.outer(np.ones(np.size(u)), np.sin(v)).T * (EARTH_RADIUS / unit_conversion) + 0
        ax1.plot_surface(mesh_x, mesh_y, mesh_z, color="blue", alpha=0.6, edgecolor='none')

        ax1.scatter3D(r_moon[mask_gcrf, 0], r_moon[mask_gcrf, 1], r_moon[mask_gcrf, 2], color=grey_colors, s=(MOON_RADIUS / unit_conversion))
        ax1.set_xlabel(f'x [{unit_label}]', color=textcolor, fontsize=fontsize)
        ax1.set_ylabel(f'y [{unit_label}]', color=textcolor, fontsize=fontsize)
        ax1.set_zlabel(f'z [{unit_label}]', color=textcolor, fontsize=fontsize)

        ax2.scatter3D(xyz_lunar[:, 0], xyz_lunar[:, 1], xyz_lunar[:, 2], color=scatter_dot_colors, s=1)
        mesh_x = np.outer(np.cos(u), np.cos(v)).T * (MOON_RADIUS / unit_conversion) + 0
        mesh_y = np.outer(np.sin(u), np.cos(v)).T * (MOON_RADIUS / unit_conversion) + 0
        mesh_z = np.outer(np.ones(np.size(u)), np.sin(v)).T * (MOON_RADIUS / unit_conversion) + 0
        ax2.plot_surface(mesh_x, mesh_y, mesh_z, color="grey", alpha=0.6, edgecolor='none')

        ax2.scatter3D(r_earth_lunar[mask_lunar, 0], r_earth_lunar[mask_lunar, 1], r_earth_lunar[mask_lunar, 2], color=blue_colors, s=55)
        ax2.set_xlabel(f'x [{unit_label}]', color=textcolor, fontsize=fontsize)
        ax2.set_ylabel(f'y [{unit_label}]', color=textcolor, fontsize=fontsize)
        ax2.set_zlabel(f'z [{unit_label}]', color=textcolor, fontsize=fontsize)
        for (point, pos) in lagrange_points_lunar_fixed_frame().items():
            pos = pos / unit_conversion
            if bounds_lunar["lower"][0] <= pos[0] <= bounds_lunar["upper"][0] and bounds_lunar["lower"][1] <= pos[1] <= bounds_lunar["upper"][1] and bounds_lunar["lower"][2] <= pos[2] <= bounds_lunar["upper"][2]:
                ax2.scatter(pos[0], pos[1], pos[2], color=textcolor, label=point, s=10)
                ax2.text(pos[0], pos[1], pos[2], point, color=textcolor)

    ax1.set_xlim(bounds_gcrf["lower"][0], bounds_gcrf["upper"][0])
    ax1.set_ylim(bounds_gcrf["lower"][1], bounds_gcrf["upper"][1])
    ax1.set_zlim(bounds_gcrf["lower"][2], bounds_gcrf["upper"][2])
    ax1.set_box_aspect([1, 1, 1])

    ax2.set_xlim(bounds_lunar["lower"][0], bounds_lunar["upper"][0])
    ax2.set_ylim(bounds_lunar["lower"][1], bounds_lunar["upper"][1])
    ax2.set_zlim(bounds_lunar["lower"][2], bounds_lunar["upper"][2])
    ax2.set_box_aspect([1, 1, 1])

    ax1.set_title(title, color=textcolor, fontsize=fontsize + 2)
    ax2.set_title("Frame: Lunar Fixed", color=textcolor, fontsize=fontsize + 2)

    rainbow_line = Line2D([0], [0], color='w', linestyle='-', linewidth=2, label='Orbit Path')

    legend_elements = [
        Patch(facecolor='lightblue', edgecolor=textcolor, label='Earth'),
        Patch(facecolor='lightgrey', edgecolor=textcolor, label='Moon'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black', markersize=6, label='Lagrange Points'),
    ]

    if num_orbits == 1:
        legend_elements.append(rainbow_line)

    font_properties = font_manager.FontProperties()

    ax2.legend(
        handles=legend_elements,
        handler_map={rainbow_line: GradientLineHandler()} if num_orbits == 1 else {},
        loc='upper left',
        fontsize=12,
        facecolor=plotcolor,
        edgecolor=textcolor,
        prop=font_properties,
        labelcolor=textcolor
    )

    plt.subplots_adjust(left=0.0, right=1.5, bottom=0.05, top=0.95, wspace=-0.0)

    for ax in [ax1, ax2]:
        ax.set_facecolor(plotcolor)
        ax.tick_params(axis='both', colors=textcolor, labelsize=fontsize)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(textcolor)
            label.set_fontsize(fontsize)
        for spine in ax.spines.values():
            spine.set_edgecolor(textcolor)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.zaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.set_zorder(2)
    ax2.set_zorder(1)

    if save_path:
        save_plot(fig, save_path)
    if show:
        plt.show()
    plt.close()
    return fig, [ax1, ax2]
