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
    Visualizes a satellite ground track, altitude over time, and 3D trajectory.

    Generates a dashboard with three subplots: the 2D ground track on a world map,
    the altitude as a function of time, and a 3D orbital view around Earth.
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
    x, y, z = groundTrack(xyz, times, format='cartesian')

    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(z / (np.sqrt(x**2 + y**2 + z**2))))

    altitude = np.sqrt(x**2 + y**2 + z**2) - EARTH_RADIUS

    phi_earth = np.linspace(0, np.pi, 50)
    theta_earth = np.linspace(0, 2 * np.pi, 50)
    phi_earth, theta_earth = np.meshgrid(phi_earth, theta_earth)
    earth_x = EARTH_RADIUS * np.sin(phi_earth) * np.cos(theta_earth)
    earth_y = EARTH_RADIUS * np.sin(phi_earth) * np.sin(theta_earth)
    earth_z = EARTH_RADIUS * np.cos(phi_earth)

    fig = plt.figure(figsize=(22, 22))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

    ax1 = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    ax1.set_global()
    ax1.add_feature(cfeature.LAND, edgecolor='black')
    ax1.add_feature(cfeature.OCEAN)
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    ax1.add_feature(cfeature.LAKES, alpha=0.5)
    ax1.add_feature(cfeature.RIVERS)
    ax1.stock_img()
    ax1.gridlines(draw_labels=True)
    ax1.coastlines()
    ax1.gridlines(draw_labels=True)

    ax1.plot(lon, lat, color='red', label='Ground Track', transform=ccrs.Geodetic())
    ax1.plot(lon[0], lat[0], 'g*', markersize=14, label='Start', transform=ccrs.Geodetic())
    ax1.plot(lon[-1], lat[-1], 'kx', markersize=12, label='End', transform=ccrs.Geodetic())
    ax1.legend(loc='lower left', fontsize=20)
    ax1.set_title("Ground Track", fontsize=28)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(times / 60, altitude / 1000, color='blue')
    ax2.set_xlabel('Time (minutes)', fontsize=22)
    ax2.set_ylabel('Altitude (km)', fontsize=22)
    ax2.set_title('Altitude vs Time', fontsize=26)
    ax2.tick_params(axis='both', labelsize=20)
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, 1], projection='3d')
    ax3.plot_surface(earth_x / 1e3, earth_y / 1e3, earth_z / 1e3,
                     color='blue', alpha=0.5, linewidth=0)
    ax3.plot(x / 1e3, y / 1e3, z / 1e3, color='red', label='Orbit')
    ax3.scatter(x[0] / 1e3, y[0] / 1e3, z[0] / 1e3,
                color='green', marker='*', s=160, label='Start')
    ax3.scatter(x[-1] / 1e3, y[-1] / 1e3, z[-1] / 1e3,
                color='black', marker='x', s=140, label='End')
    ax3.set_xlabel('X (km)', fontsize=20)
    ax3.set_ylabel('Y (km)', fontsize=20)
    ax3.set_zlabel('Z (km)', fontsize=20)
    ax3.set_title('3D Trajectory', fontsize=26)
    ax3.tick_params(axis='both', labelsize=18)
    ax3.legend(fontsize=18)

    # Calculate bounding cube
    lower_bound, upper_bound = find_smallest_bounding_cube(xyz, pad=pad)

    limit_default = 10e3  # km
    max_bound = np.max(np.abs([lower_bound, upper_bound])) / 1e3

    limit = max(limit_default, max_bound)
    ax3.set_xlim([-limit, limit])
    ax3.set_ylim([-limit, limit])
    ax3.set_zlim([-limit, limit])
    ax3.set_xticks([-limit, limit])
    ax3.set_yticks([-limit, limit])
    ax3.set_zticks([-limit, limit])

    plt.tight_layout()
    plt.show()
    if save_path:
        save_plot(fig, save_path)
    return fig
