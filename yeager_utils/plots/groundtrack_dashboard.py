import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from ssapy import groundTrack
import cartopy.crs as ccrs
from ..yastropy import astropy_gcrf_to_llh
from ..coordinates import gcrf_to_lonlat
from ..compute import find_smallest_bounding_cube
from ..constants import EARTH_RADIUS
from ..time import to_gps
from .plotutils import save_plot, valid_orbits


def clean_lonlat(lon, lat):
    wraps = np.abs(np.diff(lon)) > 180
    lon_nan = np.insert(lon, np.where(wraps)[0] + 1, np.nan)
    lat_nan = np.insert(lat, np.where(wraps)[0] + 1, np.nan)
    return lon_nan, lat_nan


def groundtrack_dashboard(r, t, save_path=None, pad=500, show=False, offline=True, show_legend=True):
    """
    Visualizes multiple satellite ground tracks, altitude/velocity over time, and 3D trajectories.

    Parameters
    ----------
    r : array_like or list of array_like
        Position vectors in meters. Either a single (n,3) array or a list of (n,3) arrays for multiple orbits.
    t : array_like or list of array_like
        Time arrays in seconds or any format convertible by `to_gps`. Single array or list matching `r`.
    save_path : str or None, optional
        If provided, saves the figure to the specified path.
    pad : float, optional
        Padding (in meters) for the 3D plot's bounding cube. Defaults to 500 m.
    show : bool, optional
        If True, displays the figure.
    offline : bool, optional
        If True (default), only offline features are used. If False, enables high-res online map data.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The complete dashboard figure object.

    Author: Travis Yeager
    """

    r, t = valid_orbits(r, t)
    
    # Ensure times are converted to GPS and normalized
    t_zero = [to_gps(t) - to_gps(t)[0] for t in t]
    
    # Process each orbit
    lons, lats, altitudes, velocities, x_gts, y_gts, z_gts = [], [], [], [], [], [], []
    for r_i, t_i in zip(r, t):
        xyz = np.array(r_i)  # Ensure r_i is (n,3)
        x_gt, y_gt, z_gt = groundTrack(xyz, t_i, format='cartesian')
        lon, lat, height = astropy_gcrf_to_llh(xyz, t_i)
        # altitude = np.linalg.norm(xyz, axis=-1) - EARTH_RADIUS
        try:
            velocity = np.linalg.norm(np.gradient(xyz, axis=0), axis=1)
        except:
            velocity = 0

        lons.append(lon)
        lats.append(lat)
        altitudes.append(height)
        velocities.append(velocity)
        x_gts.append(x_gt)
        y_gts.append(y_gt)
        z_gts.append(z_gt)

    # Earth surface for 3D plots
    phi_earth = np.linspace(0, np.pi, 50)
    theta_earth = np.linspace(0, 2 * np.pi, 50)
    phi_earth, theta_earth = np.meshgrid(phi_earth, theta_earth)
    earth_x = EARTH_RADIUS * np.sin(phi_earth) * np.cos(theta_earth)
    earth_y = EARTH_RADIUS * np.sin(phi_earth) * np.sin(theta_earth)
    earth_z = EARTH_RADIUS * np.cos(phi_earth)

    # Create figure and grid
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Ground track plot
    ax1 = fig.add_subplot(gs[0, 0:2], projection=ccrs.PlateCarree())
    ax1.set_global()

    if offline:
        from .plotutils import load_earth_file, drawEarth
        ax1.imshow(load_earth_file(), extent=[-180, 180, -90, 90])
    else:
        import cartopy.feature as cfeature
        ax1.add_feature(cfeature.LAND, edgecolor='black')
        ax1.add_feature(cfeature.OCEAN)
        ax1.add_feature(cfeature.COASTLINE)
        ax1.add_feature(cfeature.BORDERS, linestyle=':')
        ax1.add_feature(cfeature.LAKES, alpha=0.5)
        ax1.add_feature(cfeature.RIVERS)
        ax1.set_facecolor('lightblue')

    gl = ax1.gridlines(draw_labels=True)
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(r)))
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        ax1.plot(lon, lat, color=colors[i], transform=ccrs.Geodetic(), linewidth=2.5)
        ax1.plot(lon[0], lat[0], '*', color=colors[i], markersize=20)
        ax1.plot(lon[-1], lat[-1], 'x', color=colors[i], markersize=14)

    legend_elements = [
        Line2D([0], [0], color='black', linewidth=2.5, label='Orbit Track'),
        Line2D([0], [0], marker='*', color='black', linestyle='None', markersize=12, label='Orbit Start'),
        Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=10, label='Orbit End')
    ]
    ax1.legend(handles=legend_elements, loc='lower left', fontsize=16)

    # Altitude plot
    ax2 = fig.add_subplot(gs[1, 0])
    altmax = 0
    for i, (ti, alt) in enumerate(zip(t_zero, altitudes)):
        ax2.plot(ti / 60, alt / 1e3, color=colors[i], linewidth=2.5, label=f'Orbit {i+1}')
        altmax = max(altmax, np.max(alt))
    ax2.set_ylim(0, altmax  / 1e3 * 1.1)
    ax2.set_xlabel('Time (minutes)', fontsize=18)
    ax2.set_ylabel('Altitude (km)', fontsize=18)
    ax2.set_title('Altitude vs Time', fontsize=20)
    ax2.tick_params(axis='both', labelsize=16)
    ax2.grid(True)
    if show_legend:
        ax2.legend(fontsize=14)

    # ITRF 3D plot
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    ax3.plot_surface(earth_x / 1e3, earth_y / 1e3, earth_z / 1e3, color='blue', alpha=0.5, linewidth=0)
    for i, (x_gt, y_gt, z_gt) in enumerate(zip(x_gts, y_gts, z_gts)):
        ax3.plot(x_gt / 1e3, y_gt / 1e3, z_gt / 1e3, color=colors[i], linewidth=2.5)
        ax3.scatter(x_gt[0] / 1e3, y_gt[0] / 1e3, z_gt[0] / 1e3, color=colors[i], marker='*', s=120)
        ax3.scatter(x_gt[-1] / 1e3, y_gt[-1] / 1e3, z_gt[-1] / 1e3, color=colors[i], marker='x', s=100)
    ax3.set_title('ITRF', fontsize=16)
    ax3.set_xlabel('X (km)', fontsize=16)
    ax3.set_ylabel('Y (km)', fontsize=16)
    ax3.set_zlabel('Z (km)', fontsize=16)
    ax3.tick_params(axis='both', labelsize=14)
    plt.axis('equal')

    # Set 3D plot limits
    all_xyz = np.concatenate([r_i for r_i in r], axis=0)
    lower_bound, upper_bound = find_smallest_bounding_cube(all_xyz, pad=pad)
    max_bound = np.max(np.abs([lower_bound, upper_bound])) / 1e3
    limit = max(10.0, max_bound)
    ax3.set_xlim([-limit, limit])
    ax3.set_ylim([-limit, limit])
    ax3.set_zlim([-limit, limit])
    ax3.set_xticks([-limit, 0, limit])
    ax3.set_yticklabels([])
    ax3.set_zticklabels([])

    # Velocity plot
    ax4 = fig.add_subplot(gs[1, 1])
    vmax = 0
    for i, (ti, vel) in enumerate(zip(t_zero, velocities)):
        if np.ndim(vel) > 0 and len(vel) > 3:
            ti = ti[1:-1]
            vel = vel[1:-1]
        ax4.plot(ti / 60, vel / 1e3, color=colors[i], linewidth=2.5)
        vmax = max(vmax, np.max(vel))
    ax4.set_ylim(0, vmax  / 1e3 + 1)
    ax4.set_xlabel('Time (minutes)', fontsize=18)
    ax4.set_ylabel('Velocity (km/s)', fontsize=18)
    ax4.set_title('Velocity vs Time', fontsize=20)
    ax4.tick_params(axis='both', labelsize=16)
    ax4.grid(True)

    # GCRF 3D plot
    ax5 = fig.add_subplot(gs[1, 2], projection='3d')
    ax5.plot_surface(earth_x / 1e3, earth_y / 1e3, earth_z / 1e3, color='blue', alpha=0.5, linewidth=0)
    for i, r_i in enumerate(r):
        x, y, z = r_i[:, 0], r_i[:, 1], r_i[:, 2]
        ax5.plot(x / 1e3, y / 1e3, z / 1e3, color=colors[i], linewidth=2.5)
        ax5.scatter(x[0] / 1e3, y[0] / 1e3, z[0] / 1e3, color=colors[i], marker='*', s=120)
        ax5.scatter(x[-1] / 1e3, y[-1] / 1e3, z[-1] / 1e3, color=colors[i], marker='x', s=100)
    ax5.set_title('GCRF', fontsize=16)
    ax5.set_xlabel('X (km)', fontsize=16)
    ax5.set_ylabel('Y (km)', fontsize=16)
    ax5.set_zlabel('Z (km)', fontsize=16)
    ax5.tick_params(axis='both', labelsize=14)
    ax5.set_xlim([-limit, limit])
    ax5.set_ylim([-limit, limit])
    ax5.set_zlim([-limit, limit])
    ax5.set_xticks([-limit, 0, limit])
    ax5.set_yticklabels([])
    ax5.set_zticklabels([])
    plt.axis('equal')

    plt.tight_layout()
    if show:
        plt.show()
    if save_path:
        save_plot(fig, save_path)
    return fig


# if __name__ == "__main__":
#     # Example: simple circular orbit around Earth at 400 km altitude for 90 minutes
#     from yeager_utils import ssapy_orbit, RGEO

#     r, v, t = ssapy_orbit(a=RGEO, e=0.2, duration=(1, 'day'))

#     # Call the dashboard
#     fig = groundtrack_dashboard(r, t, show=True, save_path=figname("ground_dashboard_test"))
