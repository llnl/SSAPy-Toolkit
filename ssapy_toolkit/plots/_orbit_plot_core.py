from __future__ import annotations

from ssapy import get_body as _get_body
import matplotlib.cm as _cm
import matplotlib.pyplot as _plt
import numpy as _np

from ..compute import find_smallest_bounding_cube as _find_smallest_bounding_cube
from ..constants import EARTH_RADIUS as _EARTH_RADIUS
from ..constants import MOON_RADIUS as _MOON_RADIUS
from ..constants import RGEO as _RGEO
from ..coordinates import gcrf_to_itrf as _gcrf_to_itrf
from ..coordinates import gcrf_to_lunar as _gcrf_to_lunar
from ..coordinates import gcrf_to_lunar_fixed as _gcrf_to_lunar_fixed
from ._frames import normalize_orbit_frame as _normalize_orbit_frame
from .plotutils import save_plot as _save_plot
from .plotutils import valid_orbits as _valid_orbits


_VALID_VIEWS = ("xy", "xz", "yz", "3d")
_VIEW_AXES = {
    "xy": (0, 1, "x", "y"),
    "xz": (0, 2, "x", "z"),
    "yz": (1, 2, "y", "z"),
}


def _lagrange_points_lunar_frame():
    from ..orbital_mechanics import lagrange_points_lunar_frame

    return lagrange_points_lunar_frame()


def _orbit_plot_core(
    r,
    t=None,
    title="",
    figsize=(7, 7),
    save_path=False,
    frame="gcrf",
    show=False,
    c="black",
    pad=1,
    *,
    views=("xy", "xz", "yz", "3d"),
    xy_title_includes_title=False,
    lunar_transform="standard",
):
    views = tuple(views)
    unsupported = [view for view in views if view not in _VALID_VIEWS]
    if unsupported:
        raise ValueError(f"Unsupported orbit plot views: {unsupported}")

    r, t = _valid_orbits(r, t)

    if "w" in c:
        textcolor = "black"
        plotcolor = "white"
    else:
        textcolor = "white"
        plotcolor = "black"

    fig = _plt.figure(dpi=100, figsize=figsize, facecolor=plotcolor)
    axes = _create_orbit_axes(fig, views)

    bounds = {
        "lower": _np.array([_np.inf, _np.inf, _np.inf]),
        "upper": _np.array([-_np.inf, -_np.inf, -_np.inf]),
    }

    if any(_np.max(_np.linalg.norm(xyz, axis=-1)) >= 0.95 * _RGEO for xyz in r):
        unit_conversion = _RGEO
        unit_label = "GEO"
    else:
        unit_conversion = 1e3
        unit_label = "km"

    frame_key = _normalize_orbit_frame(frame)
    frame_transformations = _orbit_frame_transformations(lunar_transform)
    if frame_key not in frame_transformations:
        raise ValueError("Unknown plot type provided. Accepted: gcrf, itrf, lunar, lunar fixed")

    for orbit_index, xyz_raw in enumerate(r):
        xyz = xyz_raw
        t_current = t[orbit_index]
        r_moon = _get_body("moon").position(t_current).T
        r_earth = _np.zeros(_np.shape(r_moon))

        title2, transform_func = frame_transformations[frame_key]
        if transform_func:
            xyz = transform_func(xyz, t_current)
            r_moon = transform_func(r_moon, t_current)
            r_earth = transform_func(r_earth, t_current)

        xyz = xyz / unit_conversion
        r_moon = r_moon / unit_conversion
        r_earth = r_earth / unit_conversion

        lower_bound_temp, upper_bound_temp = _find_smallest_bounding_cube(xyz, pad=pad)
        bounds["lower"] = _np.minimum(bounds["lower"], lower_bound_temp)
        bounds["upper"] = _np.maximum(bounds["upper"], upper_bound_temp)

        stn = _plot_settings(frame_key, r_moon, r_earth, unit_conversion)
        if len(r) == 1:
            scatter_dot_colors = _cm.rainbow(_np.linspace(0, 1, len(xyz[:, 0])))
        else:
            scatter_dot_colors = _cm.rainbow(_np.linspace(0, 1, len(r)))[orbit_index]

        for view in views:
            if view == "3d":
                _plot_orbit_3d(
                    axes[view],
                    xyz,
                    stn,
                    bounds,
                    unit_label,
                    unit_conversion,
                    scatter_dot_colors,
                    textcolor,
                    frame_key,
                )
            else:
                _plot_orbit_view(
                    axes[view],
                    view,
                    xyz,
                    stn,
                    bounds,
                    unit_label,
                    unit_conversion,
                    scatter_dot_colors,
                    textcolor,
                    frame_key,
                    title,
                    title2,
                    xy_title_includes_title,
                )

    _set_orbit_limits(axes, bounds)
    _style_orbit_axes(axes.values(), plotcolor, textcolor)

    if save_path:
        _save_plot(fig, save_path)
    if show:
        _plt.show()
    _plt.close()
    return fig, [axes[view] for view in views]


def _create_orbit_axes(fig, views):
    if views == ("xy",):
        return {"xy": fig.add_subplot(1, 1, 1)}
    if views == ("xy", "xz"):
        return {
            "xy": fig.add_subplot(1, 2, 1),
            "xz": fig.add_subplot(1, 2, 2),
        }
    if views == ("xy", "xz", "yz", "3d"):
        return {
            "xy": fig.add_subplot(2, 2, 1),
            "xz": fig.add_subplot(2, 2, 2),
            "yz": fig.add_subplot(2, 2, 3),
            "3d": fig.add_subplot(2, 2, 4, projection="3d"),
        }

    axes = {}
    for index, view in enumerate(views, start=1):
        projection = "3d" if view == "3d" else None
        axes[view] = fig.add_subplot(1, len(views), index, projection=projection)
    return axes


def _orbit_frame_transformations(lunar_transform):
    lunar_func = _gcrf_to_lunar_fixed if lunar_transform == "fixed" else _gcrf_to_lunar
    return {
        "gcrf": ("GCRF", None),
        "itrf": ("ITRF", _gcrf_to_itrf),
        "lunar": ("Lunar Frame", lunar_func),
        "lunar axis": ("Moon on x-axis Frame", _gcrf_to_lunar),
    }


def _plot_settings(frame_key, r_moon, r_earth, unit_conversion):
    if _np.size(r_moon[:, 0]) > 1:
        grey_colors = _cm.Greys(_np.linspace(0, 0.8, len(r_moon[:, 0])))[::-1]
        blues = _cm.Blues(_np.linspace(0.4, 0.9, len(r_moon[:, 0])))[::-1]
    else:
        grey_colors = "grey"
        blues = "Blue"

    settings = {
        "gcrf": {
            "primary_color": "blue",
            "primary_size": (_EARTH_RADIUS / unit_conversion),
            "secondary_x": r_moon[:, 0],
            "secondary_y": r_moon[:, 1],
            "secondary_z": r_moon[:, 2],
            "secondary_color": grey_colors,
            "secondary_size": (_MOON_RADIUS / unit_conversion),
        },
        "itrf": {
            "primary_color": "blue",
            "primary_size": (_EARTH_RADIUS / unit_conversion),
            "secondary_x": r_moon[:, 0],
            "secondary_y": r_moon[:, 1],
            "secondary_z": r_moon[:, 2],
            "secondary_color": grey_colors,
            "secondary_size": (_MOON_RADIUS / unit_conversion),
        },
        "lunar": {
            "primary_color": "grey",
            "primary_size": (_MOON_RADIUS / unit_conversion),
            "secondary_x": r_earth[:, 0],
            "secondary_y": r_earth[:, 1],
            "secondary_z": r_earth[:, 2],
            "secondary_color": blues,
            "secondary_size": (_EARTH_RADIUS / unit_conversion),
        },
        "lunar axis": {
            "primary_color": "blue",
            "primary_size": (_EARTH_RADIUS / unit_conversion),
            "secondary_x": r_moon[:, 0],
            "secondary_y": r_moon[:, 1],
            "secondary_z": r_moon[:, 2],
            "secondary_color": grey_colors,
            "secondary_size": (_MOON_RADIUS / unit_conversion),
        },
    }
    return settings[frame_key]


def _plot_orbit_view(
    ax,
    view,
    xyz,
    stn,
    bounds,
    unit_label,
    unit_conversion,
    scatter_dot_colors,
    textcolor,
    frame_key,
    title,
    title2,
    xy_title_includes_title,
):
    x_idx, y_idx, x_name, y_name = _VIEW_AXES[view]
    secondary_y_key = f"secondary_{y_name}"

    ax.scatter(xyz[:, x_idx], xyz[:, y_idx], color=scatter_dot_colors, s=1)
    ax.add_patch(_plt.Circle(xy=(0, 0), radius=1, color=textcolor, linestyle="dashed", fill=False))
    ax.add_patch(
        _plt.Circle(
            xy=(0, 0),
            radius=stn["primary_size"],
            color=stn["primary_color"],
            linestyle="dashed",
            fill=False,
        )
    )
    ax.scatter(stn[f"secondary_{x_name}"], stn[secondary_y_key], color=stn["secondary_color"], s=stn["secondary_size"])
    ax.set_aspect("equal")
    ax.set_xlabel(f"{x_name} [{unit_label}]", color=textcolor)
    ax.set_ylabel(f"{y_name} [{unit_label}]", color=textcolor)

    if view == "xy":
        if xy_title_includes_title:
            ax.set_title(f"{title}\nFrame: {title2}", color=textcolor)
        else:
            ax.set_title(f"Frame: {title2}", color=textcolor)
    elif view == "xz":
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_title(f"{title}", color=textcolor)

    if "lunar" in frame_key:
        for point, pos in _lagrange_points_lunar_frame().items():
            pos = pos / unit_conversion
            if bounds["lower"][x_idx] <= pos[x_idx] <= bounds["upper"][x_idx] and bounds["lower"][y_idx] <= pos[y_idx] <= bounds["upper"][y_idx]:
                ax.scatter(pos[x_idx], pos[y_idx], color=textcolor, label=point, s=10)
                ax.text(pos[x_idx], pos[y_idx], point, color=textcolor)


def _plot_orbit_3d(ax, xyz, stn, bounds, unit_label, unit_conversion, scatter_dot_colors, textcolor, frame_key):
    u = _np.linspace(0, 2 * _np.pi, 180)
    v = _np.linspace(-_np.pi / 2, _np.pi / 2, 180)

    ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=scatter_dot_colors, s=1)
    mesh_x = _np.outer(_np.cos(u), _np.cos(v)).T * stn["primary_size"]
    mesh_y = _np.outer(_np.sin(u), _np.cos(v)).T * stn["primary_size"]
    mesh_z = _np.outer(_np.ones(_np.size(u)), _np.sin(v)).T * stn["primary_size"]
    ax.plot_surface(mesh_x, mesh_y, mesh_z, color=stn["primary_color"], alpha=0.3, edgecolor="none")
    ax.scatter3D(stn["secondary_x"], stn["secondary_y"], stn["secondary_z"], color=stn["secondary_color"], s=stn["secondary_size"])
    ax.set_xlabel(f"x [{unit_label}]", color=textcolor)
    ax.set_ylabel(f"y [{unit_label}]", color=textcolor)
    ax.set_zlabel(f"z [{unit_label}]", color=textcolor)

    if "lunar" in frame_key:
        for point, pos in _lagrange_points_lunar_frame().items():
            pos = pos / unit_conversion
            if all(bounds["lower"][idx] <= pos[idx] <= bounds["upper"][idx] for idx in range(3)):
                ax.scatter(pos[0], pos[1], pos[2], color=textcolor, label=point, s=10)
                ax.text(pos[0], pos[1], pos[2], point, color=textcolor)


def _set_orbit_limits(axes, bounds):
    for view, ax in axes.items():
        if view == "3d":
            ax.set_xlim(bounds["lower"][0], bounds["upper"][0])
            ax.set_ylim(bounds["lower"][1], bounds["upper"][1])
            ax.set_zlim(bounds["lower"][2], bounds["upper"][2])
            ax.set_box_aspect([1, 1, 1])
            continue

        x_idx, y_idx, _, _ = _VIEW_AXES[view]
        ax.set_xlim(bounds["lower"][x_idx], bounds["upper"][x_idx])
        ax.set_ylim(bounds["lower"][y_idx], bounds["upper"][y_idx])


def _style_orbit_axes(axes, plotcolor, textcolor):
    for ax in axes:
        ax.set_facecolor(plotcolor)
        ax.tick_params(axis="both", colors=textcolor)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(textcolor)
        for spine in ax.spines.values():
            spine.set_edgecolor(textcolor)
