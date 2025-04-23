import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from ssapy import groundTrack

from ..compute import find_smallest_bounding_cube
from ..constants import EARTH_RADIUS
from ..time import to_gps
from .plotutils import save_plot


def groundtrack_dashboard(x, y, z, times, save_path=None, pad=500):
    """
    Visualizes a satellite ground track, altitude/velocity over time, and 3D trajectories.

    Generates a dashboard with six subplots: the 2D ground track on a world map,
    altitude and velocity as functions of time, and 3D orbital views in both ITRF and GCRF.
    Useful for analyzing orbital behaviors, verifying trajectory simulations,
    and presenting satellite dynamics.

    Parameters
    ----------
    x : array_like
        X positions in meters.
    y : array_like
        Y positions in meters.
    z : array_like
        Z positions in meters.
    times : array_like
        Time array in seconds or any format convertible by `to_gps`.
    save_path : str or None, optional
        If provided, saves the figure to the specified path.
    pad : float, optional
        Padding (in meters) for the 3D plot's bounding cube. Defaults to 500 m.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The complete dashboard figure object.

    Author
    ------
    Travis Yeager
    """
    times = to_gps(times)
    times = times - times[0]

    xyz = np.stack([x, y, z], axis=-1)
    x_gt, y_gt, z_gt = groundTrack(xyz, times, format='cartesian')

    lon = np.degrees(np.arctan2(y_gt, x_gt))
    lat = np.degrees(np.arcsin(z_gt / np.linalg.norm(np.stack([x_gt, y_gt, z_gt]), axis=0)))
    altitude = np.linalg.norm(xyz, axis=-1) - EARTH_RADIUS

    velocity = np.linalg.norm(np.gradient(xyz, axis=0), axis=1)

    phi_earth = np.linspace(0, np.pi, 50)
    theta_earth = np.linspace(0, 2 * np.pi, 50)
    phi_earth, theta_earth = np.meshgrid(phi_earth, theta_earth)
    earth_x = EARTH_RADIUS * np.sin(phi_earth) * np.cos(theta_earth)
    earth_y = EARTH_RADIUS * np.sin(phi_earth) * np.sin(theta_earth)
    earth_z = EARTH_RADIUS * np.cos(phi_earth)

    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0:2], projection=ccrs.PlateCarree())
    ax1.set_global()
    ax1.add_feature(cfeature.LAND, edgecolor='black')
    ax1.add_feature(cfeature.OCEAN)
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    ax1.add_feature(cfeature.LAKES, alpha=0.5)
    ax1.add_feature(cfeature.RIVERS)
    ax1.stock_img()
    gl = ax1.gridlines(draw_labels=True)
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}
    ax1.plot(lon, lat, color='red', label='Ground Track', transform=ccrs.Geodetic(), linewidth=2.5)
    ax1.plot(lon[0], lat[0], 'g*', markersize=20, label='Start', transform=ccrs.Geodetic())
    ax1.plot(lon[-1], lat[-1], 'kx', markersize=14, label='End', transform=ccrs.Geodetic())
    ax1.legend(loc='lower left', fontsize=16)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(times / 60, altitude / 1e3, color='blue', linewidth=2.5)
    ax2.set_xlabel('Time (minutes)', fontsize=18)
    ax2.set_ylabel('Altitude (km)', fontsize=18)
    ax2.set_title('Altitude vs Time', fontsize=20)
    ax2.tick_params(axis='both', labelsize=16)
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    ax3.plot_surface(earth_x / 1e3, earth_y / 1e3, earth_z / 1e3,
                     color='blue', alpha=0.5, linewidth=0)
    ax3.plot(x_gt / 1e3, y_gt / 1e3, z_gt / 1e3, color='red', linewidth=2.5)
    ax3.scatter(x_gt[0] / 1e3, y_gt[0] / 1e3, z_gt[0] / 1e3,
                color='green', marker='*', s=120, label='Start')
    ax3.scatter(x_gt[-1] / 1e3, y_gt[-1] / 1e3, z_gt[-1] / 1e3,
                color='black', marker='x', s=100, label='End')
    ax3.set_xlabel('X (km)', fontsize=16)
    ax3.set_ylabel('Y (km)', fontsize=16)
    ax3.set_zlabel('Z (km)', fontsize=16)
    ax3.tick_params(axis='both', labelsize=14)
    ax3.legend(fontsize=14)
    ax3.set_title('3D ITRF', fontsize=16)

    lower_bound, upper_bound = find_smallest_bounding_cube(xyz, pad=pad)
    max_bound = np.max(np.abs([lower_bound, upper_bound])) / 1e3
    limit = max(10.0, max_bound)
    ax3.set_xlim([-limit, limit])
    ax3.set_ylim([-limit, limit])
    ax3.set_zlim([-limit, limit])
    ax3.set_xticks([-limit, 0, limit])
    ax3.set_yticklabels([])
    ax3.set_zticklabels([])

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(times[1:-1] / 60, velocity[1:-1] / 1e3, color='purple', linewidth=2.5)
    ax4.set_xlabel('Time (minutes)', fontsize=18)
    ax4.set_ylabel('Velocity (km/s)', fontsize=18)
    ax4.set_title('Velocity vs Time', fontsize=20)
    ax4.tick_params(axis='both', labelsize=16)
    ax4.grid(True)

    ax5 = fig.add_subplot(gs[1, 2], projection='3d')
    ax5.plot_surface(earth_x / 1e3, earth_y / 1e3, earth_z / 1e3,
                     color='blue', alpha=0.5, linewidth=0)
    ax5.plot(x / 1e3, y / 1e3, z / 1e3, color='orange', linewidth=2.5)
    ax5.scatter(x[0] / 1e3, y[0] / 1e3, z[0] / 1e3,
                color='green', marker='*', s=120)
    ax5.scatter(x[-1] / 1e3, y[-1] / 1e3, z[-1] / 1e3,
                color='black', marker='x', s=100)
    ax5.set_xlabel('X (km)', fontsize=16)
    ax5.set_ylabel('Y (km)', fontsize=16)
    ax5.set_zlabel('Z (km)', fontsize=16)
    ax5.tick_params(axis='both', labelsize=14)
    ax5.set_title('3D GCRF', fontsize=16)
    ax5.set_xlim([-limit, limit])
    ax5.set_ylim([-limit, limit])
    ax5.set_zlim([-limit, limit])
    ax5.set_xticks([-limit, 0, limit])
    ax5.set_yticklabels([])
    ax5.set_zticklabels([])

    plt.tight_layout()
    plt.show()
    if save_path:
        save_plot(fig, save_path)
    return fig
