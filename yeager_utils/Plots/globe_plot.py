import matplotlib.pyplot as plt
from PIL import Image as PILImage
import numpy as np

from ssapy.utils import find_file
from .plotutils import make_black, make_white, save_plot, valid_orbits
from ..constants import RGEO, EARTH_RADIUS


def globe_plot(
    r: np.ndarray | list[np.ndarray],
    t: np.ndarray | list[np.ndarray] | None = None,
    limits: float | None = None,
    title: str = "",
    c: str = "black",
    figsize: tuple[int, int] = (7, 8),
    save_path: str | None = None,
    el: float = 5.0,
    az: float = 5.0,
    scale: float = 1.0,
    fontsize: int = 18,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a textured Earth and scatter satellite positions in 3D.
    r: (N,3) array or list of (Ni,3) arrays in meters (ECI/ECEF); internally scaled by RGEO.
    t: optional time array(s); only used by valid_orbits(). If None, dummy indices are created.
    """

    # ---------- Normalize inputs and make time optional ----------
    if isinstance(r, np.ndarray):
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError("r must be (N,3) or list of (Ni,3) arrays")
        r_list = [r]
    elif isinstance(r, (list, tuple)):
        r_list = [np.asarray(ri) for ri in r]
        for ri in r_list:
            if ri.ndim != 2 or ri.shape[1] != 3:
                raise ValueError("Each track in r must be a (Ni,3) array")
    else:
        raise TypeError("r must be an ndarray or list/tuple of ndarrays")

    if t is None:
        t_list = [np.arange(ri.shape[0]) for ri in r_list]
    elif isinstance(t, np.ndarray):
        t_list = [t]
    elif isinstance(t, (list, tuple)):
        t_list = [np.asarray(ti) for ti in t]
    else:
        raise TypeError("t must be None, an ndarray, or list/tuple of ndarrays")

    r_list, t_list = valid_orbits(r_list, t_list)

    # ---------- Theme ----------
    if c in ("black", "b"):
        plotcolor, textcolor = "black", "white"
    elif c in ("white", "w"):
        plotcolor, textcolor = "white", "black"
    else:
        plotcolor, textcolor = "white", "black"

    # ---------- Earth texture ----------
    scale = 1.0 if not np.isfinite(scale) or scale <= 0 else float(scale)
    tex_w = int(round(5400 / scale))
    tex_h = int(round(2700 / scale))
    earth_png = PILImage.open(find_file("earth", ext=".png")).resize((tex_w, tex_h), resample=PILImage.BILINEAR)
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

    ax.set_xlim(-limits, limits)
    ax.set_ylim(-limits, limits)
    ax.set_zlim(-limits, limits)

    if limits < 1.0:
        xt = np.linspace(-limits, limits, 5)
        yt = np.linspace(-limits, limits, 5)
        zt = np.linspace(-limits, limits, 5)
    else:
        L = int(np.ceil(limits))
        xt = [-L, 0, L]
        yt = [-L, 0, L]
        zt = [-L, 0, L]

    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_zticks(zt)
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

    # Equal aspect for 3D
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    # Apply theme helpers
    if c in ("black", "b"):
        fig, ax = make_black(fig, ax)
    elif c in ("white", "w"):
        fig, ax = make_white(fig, ax)

    # Final view and optional save
    ax.view_init(elev=float(el), azim=float(az))
    if save_path:
        save_plot(fig, save_path)

    return fig, ax
