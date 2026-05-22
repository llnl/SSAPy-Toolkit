from .plotutils import valid_orbits

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


def cislunar_plot_3d(r, t=None, figsize=(8, 8), fontsize=12, save_path=False, show=False, legend=True, title='', c='white'):
    """
    Author: Travis Yeager (yeager7@llnl.gov)
    """
    from ..orbital_mechanics import lagrange_points_lunar_fixed_frame

    r, t = valid_orbits(r, t)

    if 'w' in c:
        textcolor = 'black'
        plotcolor = 'white'
    elif 'b' in c:
        textcolor = 'white'
        plotcolor = 'black'

    fig = plt.figure(figsize=figsize, dpi=100, facecolor=plotcolor)
    ax = fig.add_subplot(111, projection='3d')

    bounds_lunar = {"lower": np.array([np.inf, np.inf, np.inf]), "upper": np.array([-np.inf, -np.inf, -np.inf])}

    # Check if all arrays in `r` are the same shape

    for orbit_index in range(len(r)):
        xyz = r[orbit_index]
        t_current = t[orbit_index]

        r_moon = get_body("moon").position(t_current).T
        r_earth = np.zeros(np.shape(r_moon))

        if max(np.linalg.norm(xyz, axis=-1) >= .95 * RGEO):
            unit_conversion = RGEO
            unit_label = 'GEO'
        else:
            unit_conversion = 1e3
            unit_label = 'km'

        xyz_lunar = gcrf_to_lunar_fixed(xyz, t_current) / unit_conversion
        r_earth_lunar = gcrf_to_lunar_fixed(r_earth, t_current) / unit_conversion
        xyz = xyz / unit_conversion
        r_moon = r_moon / unit_conversion
        r_earth = r_earth / unit_conversion

        lower_bound_lunar_temp, upper_bound_lunar_temp = find_smallest_bounding_cube(xyz_lunar, pad=1)
        bounds_lunar["lower"] = np.minimum(bounds_lunar["lower"], lower_bound_lunar_temp)
        bounds_lunar["upper"] = np.maximum(bounds_lunar["upper"], upper_bound_lunar_temp)

        mask_lunar = (
            (r_earth_lunar[:, 0] >= bounds_lunar["lower"][0]) & (r_earth_lunar[:, 0] <= bounds_lunar["upper"][0])
            & (r_earth_lunar[:, 1] >= bounds_lunar["lower"][1]) & (r_earth_lunar[:, 1] <= bounds_lunar["upper"][1])
            & (r_earth_lunar[:, 2] >= bounds_lunar["lower"][2]) & (r_earth_lunar[:, 2] <= bounds_lunar["upper"][2])
        )

        if np.size(r_moon[:, 0]) > 1:
            blue_colors = cm.Blues(np.linspace(0.2, .8, len(r_moon[:, 0])))[::-1][mask_lunar]
        else:
            blue_colors = "blue"

        if len(r) == 1:
            scatter_dot_colors = cm.rainbow(np.linspace(0, 1, len(xyz[:, 0])))
        else:
            scatter_dot_colors = cm.rainbow(np.linspace(0, 1, len(r)))[orbit_index]

        # Create a 3d sphere of the Earth and Moon
        u = np.linspace(0, 2 * np.pi, 180)
        v = np.linspace(-np.pi / 2, np.pi / 2, 180)

        ax.scatter3D(xyz_lunar[:, 0], xyz_lunar[:, 1], xyz_lunar[:, 2], color=scatter_dot_colors, s=1)
        mesh_x = np.outer(np.cos(u), np.cos(v)).T * (MOON_RADIUS / unit_conversion) + 0
        mesh_y = np.outer(np.sin(u), np.cos(v)).T * (MOON_RADIUS / unit_conversion) + 0
        mesh_z = np.outer(np.ones(np.size(u)), np.sin(v)).T * (MOON_RADIUS / unit_conversion) + 0
        ax.plot_surface(mesh_x, mesh_y, mesh_z, color="grey", alpha=0.6, edgecolor='none')

        ax.scatter3D(r_earth_lunar[mask_lunar, 0], r_earth_lunar[mask_lunar, 1], r_earth_lunar[mask_lunar, 2], color=blue_colors, s=55)
        ax.set_xlabel(f'x [{unit_label}]', color=textcolor, fontsize=fontsize)
        ax.set_ylabel(f'y [{unit_label}]', color=textcolor, fontsize=fontsize)
        ax.set_zlabel(f'z [{unit_label}]', color=textcolor, fontsize=fontsize)
        for (point, pos) in lagrange_points_lunar_fixed_frame().items():
            pos = pos / unit_conversion
            if bounds_lunar["lower"][0] <= pos[0] <= bounds_lunar["upper"][0] and bounds_lunar["lower"][1] <= pos[1] <= bounds_lunar["upper"][1] and bounds_lunar["lower"][2] <= pos[2] <= bounds_lunar["upper"][2]:
                ax.scatter(pos[0], pos[1], pos[2], color=textcolor, label=point, s=10)
                ax.text(pos[0], pos[1], pos[2], point, color=textcolor)


    ax.set_xlim(bounds_lunar["lower"][0], bounds_lunar["upper"][0])
    ax.set_ylim(bounds_lunar["lower"][1], bounds_lunar["upper"][1])
    ax.set_zlim(bounds_lunar["lower"][2], bounds_lunar["upper"][2])
    ax.set_box_aspect([1, 1, 1])

    ax.set_title(f"Frame: Lunar Fixed\n{title}", color=textcolor, fontsize=fontsize + 2)

    rainbow_line = Line2D([0], [0], color='w', linestyle='-', linewidth=2, label='Orbit Path')

    legend_elements = [
        Patch(facecolor='lightblue', edgecolor=textcolor, label='Earth'),
        Patch(facecolor='lightgrey', edgecolor=textcolor, label='Moon'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black', markersize=6, label='Lagrange Points'),
    ]

    if len(r) == 1:
        legend_elements.append(rainbow_line)

    font_properties = font_manager.FontProperties()

    if legend:
        ax.legend(
            handles=legend_elements,
            handler_map={rainbow_line: GradientLineHandler()} if len(r) == 1 else {},
            loc='upper left',
            fontsize=12,
            facecolor=plotcolor,
            edgecolor=textcolor,
            prop=font_properties,
            labelcolor=textcolor
        )

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

    ax.set_zorder(1)

    if save_path:
        save_plot(fig, save_path)
    if show:
        plt.show()
    plt.close()
    return fig, ax
