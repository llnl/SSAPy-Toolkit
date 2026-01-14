import matplotlib.pyplot as plt
from PIL import Image as PILImage
import numpy as np

from ssapy.utils import find_file
from .plotutils import make_black, make_white, save_plot
from ..constants import RGEO, EARTH_RADIUS
from .plotutils import valid_orbits  # adjust import path to where you place valid_orbits


def globe_plot(
    r,
    t=None,
    limits=None,
    title="",
    c="black",
    figsize=(7, 8),
    save_path=None,
    el=5.0,
    az=5.0,
    scale=1.0,
    fontsize=18,
):
    """
    Plot a textured Earth and scatter satellite positions in 3D.

    This function relies on valid_orbits(r, t) to normalize inputs.
    """

    # ---- Normalize/validate inputs in one place ----
    r_list, t_list = valid_orbits(r, t)  # t_list is not used directly here, but kept for consistency

    # ---------- Theme ----------
    if c in ("black", "b"):
        plotcolor, textcolor = "black", "white"
    elif c in ("white", "w"):
        plotcolor, textcolor = "white", "black"
    else:
        plotcolor, textcolor = "white", "black"

    # ---------- Earth texture ----------
    scale = 1.0 if (not np.isfinite(scale) or scale <= 0) else float(scale)
    tex_w = int(round(5400 / scale))
    tex_h = int(round(2700 / scale))
    earth_png = PILImage.open(find_file("earth", ext=".png")).resize(
        (tex_w, tex_h), resample=PILImage.BILINEAR
    )
    bm = np.asarray(earth_png, dtype=float) / 255.0  # (H, W, C)

    # ---------- Globe mesh (match texture resolution) ----------
    lons = np.radians(np.linspace(-180.0, 180.0, bm.shape[1]))
    lats = np.radians(np.linspace(-90.0, 90.0, bm.shape[0])[::-1])
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    scale_fac = EARTH_RADIUS / RGEO
    mesh_x = np.cos(lat_grid) * np.cos(lon_grid) * scale_fac
    mesh_y = np.cos(lat_grid) * np.sin(lon_grid) * scale_fac
    mesh_z = np.sin(lat_grid) * scale_fac

    # ---------- Figure and axes ----------
    fig = plt.figure(dpi=100, figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor(plotcolor)
    ax.set_facecolor(plotcolor)
    ax.view_init(elev=float(el), azim=float(az))
    ax.tick_params(axis="both", colors=textcolor)
    ax.grid(True, color="grey", linestyle="--", linewidth=0.5)

    ax.plot_surface(mesh_x, mesh_y, mesh_z, rstride=4, cstride=4, facecolors=bm, shade=False)

    # ---------- Scatter satellites ----------
    if len(r_list) == 1:
        colors = plt.cm.rainbow(np.linspace(0.0, 1.0, r_list[0].shape[0]))
    else:
        colors = [plt.cm.rainbow(v) for v in np.linspace(0.0, 1.0, len(r_list))]

    max_extent = 0.0
    for i, ri in enumerate(r_list):
        color = colors if len(r_list) == 1 else colors[i]
        ri_scaled = ri / RGEO
        if ri_scaled.size:
            max_extent = max(max_extent, float(np.nanmax(np.abs(ri_scaled))))
        ax.scatter(ri_scaled[:, 0], ri_scaled[:, 1], ri_scaled[:, 2], color=color, s=1)

    # ---------- Limits, ticks, labels ----------
    if limits is None:
        limits = (max_extent if max_extent > 0 else scale_fac) * 1.2
    if limits > 1e5:
        limits = limits / RGEO

    ax.set_xlim(-limits, limits)
    ax.set_ylim(-limits, limits)
    ax.set_zlim(-limits, limits)

    L = int(limits)
    ax.set_xticks([-L, 0, L])
    ax.set_yticks([-L, 0, L])
    ax.set_zticks([-L, 0, L])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_xlabel("x [GEO]", color=textcolor, fontsize=fontsize)
    ax.set_ylabel("y [GEO]", color=textcolor, fontsize=fontsize)
    ax.set_zlabel("z [GEO]", color=textcolor, fontsize=fontsize)
    if title:
        ax.set_title(title, color=textcolor, fontsize=fontsize)

    ax.tick_params(axis="x", colors=textcolor, labelsize=fontsize)
    ax.tick_params(axis="y", colors=textcolor, labelsize=fontsize)
    ax.tick_params(axis="z", colors=textcolor, labelsize=fontsize)

    # Apply theme helpers
    if c in ("black", "b"):
        fig, ax = make_black(fig, ax)
    elif c in ("white", "w"):
        fig, ax = make_white(fig, ax)

    if save_path:
        save_plot(fig, save_path)

    return fig, ax
