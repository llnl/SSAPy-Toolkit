######################################################################
# COLLECTION OF ALL PLOTTING AND MEDIA
######################################################################
# flake8: noqa: E501
from .orbital_mechanics import calculate_orbital_elements
import numpy as np
from ssapy.body import get_body
from ssapy.compute import groundTrack
from ssapy.utils import find_file
from .constants import RGEO, EARTH_RADIUS
from .coordinates import gcrf_to_itrf, gcrf_to_lunar, gcrf_to_lunar_fixed
from .utils import Time
from .vectors import norm
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PyPDF2 import PdfMerger
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image as PILImage
import io

from IPython.display import Image as IPythonImage
import imageio
import cv2

import ipyvolume as ipv

plt.rcParams.update({'font.size': 7, 'figure.facecolor': 'w'})
lunar_semi_major = 384399000  # m


def make_white(fig, *axes):
    fig.patch.set_facecolor('white')

    for ax in axes:
        ax.set_facecolor('white')
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        if hasattr(ax, 'zaxis'):
            ax_items.append(ax.zaxis.label)
        ax_items += ax.get_xticklabels() + ax.get_yticklabels()
        if hasattr(ax, 'get_zticklabels'):
            ax_items += ax.get_zticklabels()
        ax_items += ax.get_xticklines() + ax.get_yticklines()
        if hasattr(ax, 'get_zticklines'):
            ax_items += ax.get_zticklines()
        for item in ax_items:
            item.set_color('black')

    return fig, *axes


def make_black(fig, *axes):
    fig.patch.set_facecolor('black')

    for ax in axes:
        ax.set_facecolor('black')
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        if hasattr(ax, 'zaxis'):
            ax_items.append(ax.zaxis.label)
        ax_items += ax.get_xticklabels() + ax.get_yticklabels()
        if hasattr(ax, 'get_zticklabels'):
            ax_items += ax.get_zticklabels()
        ax_items += ax.get_xticklines() + ax.get_yticklines()
        if hasattr(ax, 'get_zticklines'):
            ax_items += ax.get_zticklines()
        for item in ax_items:
            item.set_color('white')

    return fig, *axes


def create_sphere(cx, cy, cz, r, resolution=360):
    '''
    create sphere with center (cx, cy, cz) and radius r
    '''
    phi = np.linspace(0, 2 * np.pi, 2 * resolution)
    theta = np.linspace(0, np.pi, resolution)

    theta, phi = np.meshgrid(theta, phi)

    r_xy = r * np.sin(theta)
    x = cx + np.cos(phi) * r_xy
    y = cy + np.sin(phi) * r_xy
    z = cz + r * np.cos(theta)

    return np.stack([x, y, z])


def drawSphere(xCenter, yCenter, zCenter, r, res=10j, flatten=True):
    if 'j' not in str(res):
        res = complex(0, res)
    # draw sphere
    u, v = np.mgrid[0:2 * np.pi:2 * res, 0:np.pi:res]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    # shift and scale sphere
    x = r * x + xCenter
    y = r * y + yCenter
    z = r * z + zCenter
    if flatten:
        x = np.squeeze(np.array(x).flatten())
        y = np.squeeze(np.array(y).flatten())
        z = np.squeeze(np.array(z).flatten())
    return (x, y, z)


def darken(color, amount=0.5):
    import colorsys
    try:
        c = matplotlib.colors.cnames[color]
    except Exception:
        c = color
    colors = []
    for i in amount:
        c = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(c))
        colors.append(colorsys.hls_to_rgb(c[0], 1 - i * (1 - c[1]), c[2]))
    return colors


def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
    g = 255 - b - r
    return r, g, b


def generate_rainbow_colors(num_iterations):
    cmap = plt.get_cmap('rainbow')
    colors = [matplotlib.colors.rgb2hex(cmap(i / num_iterations)) for i in range(num_iterations)]
    return colors


def write_video(video_name, frames, fps=30):
    print(f'Writing video: {video_name}')
    """
    Writes frames to an mp4 video file
    :param video_name: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """
    img = cv2.imread(frames[0])
    h, w, layers = img.shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(video_name, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(cv2.imread(frame))

    writer.release()
    print(f'Wrote: {video_name}')
    return


def write_gif(gif_name, frames, fps=30):
    print(f'Writing gif: {gif_name}')
    with imageio.get_writer(gif_name, mode='I', duration=1 / fps) as writer:
        for i, filename in enumerate(frames):
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f'Wrote {gif_name}')
    return


def saveplot(fig, filename, bbox_inches="tight", pad_inches=0.1, transparent=True, facecolor="w", edgecolor="w", orientation="landscape"):
    fig.savefig(filename, bbox_inches=bbox_inches, pad_inches=pad_inches, transparent=True, facecolor=facecolor, edgecolor=edgecolor, orientation=orientation)
    return


def loadplot(filename):
    return IPythonImage(filename)


def koe_plot(r, v, times=Time("2025-01-01", scale='utc') + np.linspace(0, int(1 * 365.25), int(365.25 * 24)), elements=['a', 'e', 'i']):
    orbital_elements = calculate_orbital_elements(r, v)
    fig, ax1 = plt.subplots(dpi=200)
    plt.rcParams.update({'font.size': 7, 'figure.facecolor': 'w'})
    if 'a' in elements:
        ax1.plot([], [], label='semi-major axis [GEO]', c='C0')
        ax2 = ax1.twinx()
        a = [x / RGEO for x in orbital_elements['semi_major_axis']]
        ax2.plot(Time(times).decimalyear, a, label='semi-major axis [GEO]', c='C0', linestyle='--')
        ax2.yaxis.label.set_color('C0')
        ax2.tick_params(axis='y', colors='C0')
        ax2.spines['right'].set_color('C0')
        ax2.set_ylabel('semi-major axis [GEO]')
        if np.abs(np.max(a) - np.min(a)) < 2:
            ax2.set_ylim((np.min(a) - 0.5, np.max(a) + 0.5))
    if 'e' in elements:
        ax1.plot(Time(times).decimalyear, [x for x in orbital_elements['eccentricity']], label='eccentricity', c='C1')
    if 'i' in elements:
        ax1.plot(Time(times).decimalyear, [x for x in orbital_elements['inclination']], label='inclination [rad]', c='C2')

    ax1.set_xlabel('Year')
    ax1.legend(loc='upper center')
    plt.show(block=False)
    return fig, ax1


def koe_2dhist(stable_data, title="Initial orbital elements of\n20 year stable cislunar orbits", limits=[1, 100], bins=100, logscale=True):
    if logscale or logscale == 'log':
        norm = matplotlib.colors.LogNorm(limits[0], limits[1])
    else:
        norm = matplotlib.colors.Normalize(limits[0], limits[1])
    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(dpi=100, figsize=(10, 8), nrows=3, ncols=3)
    st = fig.suptitle(title, fontsize=12)
    st.set_x(0.46)
    st.set_y(0.9)
    ax = axes.flat[0]
    ax.hist2d([x / RGEO for x in stable_data.a], [x for x in stable_data.e], bins=bins, norm=norm)
    ax.set_xlabel("")
    ax.set_ylabel("eccentricity")
    ax.set_xticks(np.arange(1, 20, 2))
    ax.set_yticks(np.arange(0, 1, 0.2))
    ax.set_xlim((1, 18))
    axes.flat[1].set_axis_off()
    axes.flat[2].set_axis_off()

    ax = axes.flat[3]
    ax.hist2d([x / RGEO for x in stable_data.a], [np.degrees(x) for x in stable_data.i], bins=bins, norm=norm)
    ax.set_xlabel("")
    ax.set_ylabel("inclination [deg]")
    ax.set_xticks(np.arange(1, 20, 2))
    ax.set_yticks(np.arange(0, 91, 15))
    ax.set_xlim((1, 18))
    ax = axes.flat[4]
    ax.hist2d([x for x in stable_data.e], [np.degrees(x) for x in stable_data.i], bins=bins, norm=norm)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(0, 1, 0.2))
    ax.set_yticks(np.arange(0, 91, 15))
    axes.flat[5].set_axis_off()

    ax = axes.flat[6]
    ax.hist2d([x / RGEO for x in stable_data.a], [np.degrees(x) for x in stable_data.trueAnomaly], bins=bins, norm=norm)
    ax.set_xlabel("semi-major axis [GEO]")
    ax.set_ylabel("True Anomaly [deg]")
    ax.set_xticks(np.arange(1, 20, 2))
    ax.set_yticks(np.arange(0, 361, 60))
    ax.set_xlim((1, 18))
    ax = axes.flat[7]
    ax.hist2d([x for x in stable_data.e], [np.degrees(x) for x in stable_data.trueAnomaly], bins=bins, norm=norm)
    ax.set_xlabel("eccentricity")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(0, 1, 0.2))
    ax.set_yticks(np.arange(0, 361, 60))
    ax = axes.flat[8]
    ax.hist2d([np.degrees(x) for x in stable_data.i], [np.degrees(x) for x in stable_data.trueAnomaly], bins=bins, norm=norm)
    ax.set_xlabel("inclination [deg]")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(0, 91, 15))
    ax.set_yticks(np.arange(0, 361, 60))

    im = fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax, norm=norm)
    return


def scatter2d(x, y, cs, xlabel='x', ylabel='y', title='', cbar_label='', dotsize=1, colorsMap='jet', colorscale='linear', colormin=False, colormax=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if colormax is False:
        colormax = np.max(cs)
    if colormin is False:
        colormin = np.min(cs)
    cm = plt.get_cmap(colorsMap)
    if colorscale == 'linear':
        cNorm = matplotlib.colors.Normalize(vmin=colormin, vmax=colormax)
    elif colorscale == 'log':
        cNorm = matplotlib.colors.LogNorm(vmin=colormin, vmax=colormax)
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
    ax.scatter(x, y, c=scalarMap.to_rgba(cs), s=dotsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap, shrink=.5, label=f'{cbar_label}', pad=0.04)
    plt.tight_layout()
    fig, ax = make_black(fig, ax)
    plt.show(block=False)
    return


def scatter3d(x, y=None, z=None, cs=None, xlabel='x', ylabel='y', zlabel='z', cbar_label='', dotsize=1, colorsMap='jet', title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if x.ndim > 1:
        r = x
        x = r[:, 0]
        y = r[:, 1]
        z = r[:, 2]
    if cs is None:
        ax.scatter(x, y, z, s=dotsize)
    else:
        cm = plt.get_cmap(colorsMap)
        cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
        ax.scatter(x, y, z, c=scalarMap.to_rgba(cs), s=dotsize)
        scalarMap.set_array(cs)
        fig.colorbar(scalarMap, shrink=.5, label=f'{cbar_label}', pad=0.075)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.title(title)
    plt.tight_layout()
    fig, ax = make_black(fig, ax)
    plt.show(block=False)
    return fig, ax


def dotcolors_scaled(num_colors):
    return cm.rainbow(np.linspace(0, 1, num_colors))


# Make a plot of multiple cislunar orbit in GCRF frame.
def orbit_divergence_plot(rs, r_moon=[], times=False, limits=False, title=''):
    if limits is False:
        limits = np.nanmax(np.linalg.norm(rs, axis=1) / RGEO) * 1.2
        print(f'limits: {limits}')
    if np.size(r_moon) < 1:
        moon = get_body("moon")
        r_moon = moon.position(times)
    else:
        # print('Lunar position(s) provided.')
        if r_moon.ndim != 2:
            raise IndexError(f"input moon data shape: {np.shape(r_moon)}, input should be 2 dimensions.")
            return None
        if np.shape(r_moon)[1] == 3:
            r_moon = r_moon.T
            # print(f"Tranposed input to {np.shape(r_moon)}")
    plt.rcParams.update({'font.size': 7, 'figure.facecolor': 'w'})
    plt.figure(dpi=100, figsize=(15, 4))
    for i in range(rs.shape[-1]):
        r = rs[:, :, i]
        x = r[:, 0] / RGEO
        y = r[:, 1] / RGEO
        z = r[:, 2] / RGEO
        xm = r_moon[0] / RGEO
        ym = r_moon[1] / RGEO
        zm = r_moon[2] / RGEO
        dotcolors = cm.rainbow(np.linspace(0, 1, len(x)))

        # Creating plot
        plt.subplot(1, 3, 1)
        plt.scatter(x, y, color=dotcolors, s=1)
        plt.scatter(0, 0, color="blue", s=50)
        plt.scatter(xm, ym, color="grey", s=5)
        plt.axis('scaled')
        plt.xlabel('x [GEO]')
        plt.ylabel('y [GEO]')
        plt.xlim((-limits, limits))
        plt.ylim((-limits, limits))
        plt.text(x[0], y[0], '$\leftarrow$ start')
        plt.text(x[-1], y[-1], '$\leftarrow$ end')

        plt.subplot(1, 3, 2)
        plt.scatter(x, z, color=dotcolors, s=1)
        plt.scatter(0, 0, color="blue", s=50)
        plt.scatter(xm, zm, color="grey", s=5)
        plt.axis('scaled')
        plt.xlabel('x [GEO]')
        plt.ylabel('z [GEO]')
        plt.xlim((-limits, limits))
        plt.ylim((-limits, limits))
        plt.text(x[0], z[0], '$\leftarrow$ start')
        plt.text(x[-1], z[-1], '$\leftarrow$ end')
        plt.title(f'{title}')

        plt.subplot(1, 3, 3)
        plt.scatter(y, z, color=dotcolors, s=1)
        plt.scatter(0, 0, color="blue", s=50)
        plt.scatter(ym, zm, color="grey", s=5)
        plt.axis('scaled')
        plt.xlabel('y [GEO]')
        plt.ylabel('z [GEO]')
        plt.xlim((-limits, limits))
        plt.ylim((-limits, limits))
        plt.text(y[0], z[0], '$\leftarrow$ start')
        plt.text(y[-1], z[-1], '$\leftarrow$ end')
    plt.tight_layout()
    plt.show(block=False)


def load_earth_file():
    earth = PILImage.open(find_file("earth", ext=".png"))
    earth = earth.resize((5400 // 5, 2700 // 5))
    return earth


def drawEarth(time, ngrid=100, R=EARTH_RADIUS, rfactor=1):
    """
    Parameters
    ----------
    time : array_like or astropy.time.Time (n,)
        If float (array), then should correspond to GPS seconds;
        i.e., seconds since 1980-01-06 00:00:00 UTC
    ngrid: int
        Number of grid points in Earth model.
    R: float
        Earth radius in meters.  Default is WGS84 value.
    rfactor: float
        Factor by which to enlarge Earth (for visualization purposes)

    """
    earth = load_earth_file()

    from numbers import Real
    from erfa import gst94
    lat = np.linspace(-np.pi / 2, np.pi / 2, ngrid)
    lon = np.linspace(-np.pi, np.pi, ngrid)
    lat, lon = np.meshgrid(lat, lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    u = np.linspace(0, 1, ngrid)
    v, u = np.meshgrid(u, u)

    # Need earth rotation angle for times
    # Just use erfa.gst94.
    # This ignores precession/nutation, ut1-tt and polar motion, but should
    # be good enough for visualization.
    if isinstance(time, Time):
        time = time.gps
    if isinstance(time, Real):
        time = np.array([time])

    mjd_tt = 44244.0 + (time + 51.184) / 86400
    gst = gst94(2400000.5, mjd_tt)

    u = u - (gst / (2 * np.pi))[:, None, None]
    v = np.broadcast_to(v, u.shape)

    return ipv.plot_mesh(
        x * R * rfactor, y * R * rfactor, z * R * rfactor,
        u=u, v=v,
        wireframe=False,
        texture=earth
    )


def load_moon_file():
    moon = PILImage.open(find_file("moon", ext=".png"))
    moon = moon.resize((5400 // 5, 2700 // 5))
    return moon


def groundTrackPlot(r, time, ground_stations=None):
    """
    Parameters
    ----------
    r : (3,) array_like - Orbit positions in meters.
    times: (n,) array_like - array of Astropy Time objects or time in gps seconds.

    optional - ground_stations: (n,2) array of of ground station (lat,lon) in degrees
    """
    lon, lat, height = groundTrack(r, time)

    plt.figure(figsize=(15, 12))
    plt.imshow(load_earth_file(), extent=[-180, 180, -90, 90])
    plt.plot(np.rad2deg(lon), np.rad2deg(lat))
    if ground_stations is not None:
        for ground_station in ground_stations:
            plt.scatter(ground_station[1], ground_station[0], s=50, color='Red')
    plt.ylim(-90, 90)
    plt.xlim(-180, 180)
    plt.show()


def groundTrackVideo(r, time):
    """
    Parameters
    ----------
    r : (3,) array_like
        Position of orbiting object in meters.
    t : float or astropy.time.Time
        If float or array of float, then should correspond to GPS seconds; i.e.,
        seconds since 1980-01-06 00:00:00 UTC
    """
    ipvfig = ipv.figure(width=2000 / 2, height=1000 / 2)
    ipv.style.set_style_dark()
    ipv.style.box_off()
    ipv.style.axes_off()
    widgets = []
    widgets.append(drawEarth(time))
    widgets.append(
        ipv.scatter(
            r[:, 0, None],
            r[:, 1, None],
            r[:, 2, None],
            marker='sphere',
            color='magenta',
            size=10  # Increase the dot size (default is 1)
        )
    )
    # Line plot showing the path
    widgets.append(
        ipv.plot(
            r[:, 0],
            r[:, 1],
            r[:, 2],
            color='white',
            linewidth=1
        )
    )
    ipv.animation_control(widgets, sequence_length=len(time), interval=0)
    ipv.xyzlim(-10_000_000, 10_000_000)
    ipvfig.camera.position = (-2, 0, 0.2)
    ipvfig.camera.up = (0, 0, 1)
    ipv.show()


def check_numpy_array(variable):
    if isinstance(variable, np.ndarray):
        return "numpy array"
    elif isinstance(variable, list) and all(isinstance(item, np.ndarray) for item in variable):
        return "list of numpy array"
    else:
        return "not numpy"


def orbit_plot(r, times=[], limits=False, title='', figsize=(7, 7), save_path=False, frame="gcrf"):
    """
    Parameters
    ----------
    r : (n,3) or array of [(n,3), ..., (n,3)] array_like
        Position of orbiting object(s) in meters.
    times: optional - times when r was calculated.
    limits: optional - x and y limits of the plot
    title: optional - title of the plot
    """

    def _make_scatter(fig, ax1, ax2, ax3, ax4, r, times, limits, title='', orbit_index='', num_orbits=1, frame=False):
        if np.size(times) < 1:
            if frame in ["itrf", "lunar", "lunar_fixed"]:
                raise("Need to provide times for itrf, lunar or lunar fixed frames")
            r_moon = np.atleast_2d(get_body("moon").position(Time("2000-1-1")))
        else:
            r_moon = get_body("moon").position(times).T

        # Check if the frame is in the dictionary, and set central_dot accordingly
        if frame.lower() == "GCRF".lower():
            title2 = "GCRF"
        elif frame.lower() == "ITRF".lower():
            title2 = "ITRF"
            r = gcrf_to_itrf(r, times)
        elif frame.lower() == "Lunar".lower():
            title2 = "Lunar - Earth Centered"
            r = gcrf_to_lunar(r, times)
        elif frame.lower() == "Lunar Fixed".lower():
            title2 = "Lunar Centered"
            r = gcrf_to_lunar_fixed(r, times)
            r_moon = gcrf_to_lunar(r_moon, times)
        else:
            raise ValueError("Unknown plot type provided. Accepted: gcrf, itrf, lunar, lunar fixed")

        x = r[:, 0] / RGEO
        y = r[:, 1] / RGEO
        z = r[:, 2] / RGEO
        xm = r_moon[:, 0] / RGEO
        ym = r_moon[:, 1] / RGEO
        zm = r_moon[:, 2] / RGEO
            
        if np.size(xm) > 1:
            gradient_colors = cm.Greys(np.linspace(0, .8, len(xm)))[::-1]
            blues = cm.Blues(np.linspace(.4, .9, len(xm)))[::-1]
        else:
            gradient_colors = "grey"
            blues = 'Blue'
        plot_settings = {
            "gcrf": ("blue", 50, 1, xm, ym, zm, gradient_colors),
            "itrf": ("blue", 50, 1, xm, ym, zm, gradient_colors),
            "lunar": ("blue", 50, 1, xm, ym, zm, gradient_colors),
            "lunar fixed": ("grey", 25, 1.3, -xm, -ym, -zm, blues)
        }
        try:
            stn = plot_settings[frame.lower()]
        except KeyError:
            raise ValueError("Unknown plot type provided. Accepted: 'gcrf', 'itrf', 'lunar', 'lunar fixed'")
        if limits is False:
            limits = np.nanmax(np.abs(np.array(r))) * 1.2 / RGEO

        if orbit_index == '':
            angle = 0
            dotcolors = cm.rainbow(np.linspace(0, 1, len(x)))
        else:
            angle = orbit_index * 10
            dotcolors = cm.rainbow(np.linspace(0, 1, num_orbits))[orbit_index]
        ax1.add_patch(plt.Circle((0, 0), stn[2], color='white', linestyle='dashed', fill=False))
        ax1.scatter(x, y, color=dotcolors, s=1)
        ax1.scatter(0, 0, color=stn[0], s=stn[1])
        if xm is not False:
            ax1.scatter(stn[3], stn[4], color=stn[6], s=5)
        ax1.set_aspect('equal')
        ax1.set_xlabel('x [GEO]')
        ax1.set_ylabel('y [GEO]')
        ax1.set_xlim((-limits, limits))
        ax1.set_ylim((-limits, limits))
        ax1.text(x[0], y[0], f'← start {orbit_index}', color='white', rotation=angle)
        ax1.text(x[-1], y[-1], f'← end {orbit_index}', color='white', rotation=angle)
        ax1.set_title(f'Frame: {title2}', color='white')

        ax2.add_patch(plt.Circle((0, 0), stn[2], color='white', linestyle='dashed', fill=False))
        ax2.scatter(x, z, color=dotcolors, s=1)
        ax2.scatter(0, 0, color=stn[0], s=stn[1])
        if xm is not False:
            ax2.scatter(stn[3], stn[4], color=stn[6], s=5)
        ax2.set_aspect('equal')
        ax2.set_xlabel('x [GEO]')
        ax2.set_ylabel('z [GEO]')
        ax2.set_xlim((-limits, limits))
        ax2.set_ylim((-limits, limits))
        ax2.text(x[0], z[0], f'← start {orbit_index}', color='white', rotation=angle)
        ax2.text(x[-1], z[-1], f'← end {orbit_index}', color='white', rotation=angle)
        ax2.set_title(f'{title}', color='white')

        ax3.add_patch(plt.Circle((0, 0), stn[2], color='white', linestyle='dashed', fill=False))
        ax3.scatter(y, z, color=dotcolors, s=1)
        ax3.scatter(0, 0, color=stn[0], s=stn[1])
        if xm is not False:
            ax3.scatter(stn[3], stn[4], color=stn[6], s=5)
        ax3.set_aspect('equal')
        ax3.set_xlabel('y [GEO]')
        ax3.set_ylabel('z [GEO]')
        ax3.set_xlim((-limits, limits))
        ax3.set_ylim((-limits, limits))
        ax3.text(y[0], z[0], f'← start {orbit_index}', color='white', rotation=angle)
        ax3.text(y[-1], z[-1], f'← end {orbit_index}', color='white', rotation=angle)

        ax4.scatter3D(x, y, z, color=dotcolors, s=1)
        ax4.scatter3D(0, 0, 0, color=stn[0], s=stn[1])
        if xm is not False:
            ax4.scatter3D(stn[3], stn[4], stn[5], color=stn[6], s=5)
        ax4.set_xlim([-limits, limits])
        ax4.set_ylim([-limits, limits])
        ax4.set_zlim([-limits, limits])
        ax4.set_aspect('equal')  # aspect ratio is 1:1:1 in data space
        ax4.set_xlabel('x [GEO]')
        ax4.set_ylabel('y [GEO]')
        ax4.set_zlabel('z [GEO]')
        return fig, ax1, ax2, ax3, ax4
    input_type = check_numpy_array(r)

    plt.rcParams.update({'font.size': 9, 'figure.facecolor': 'k'})
    fig = plt.figure(dpi=100, figsize=figsize, facecolor='black')
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    
    if input_type == "numpy array":
        fig, ax1, ax2, ax3, ax4 = _make_scatter(fig, ax1, ax2, ax3, ax4, r=r, times=times, limits=limits, title=title, frame=frame)
    if input_type == "list of numpy array":
        num_orbits = np.shape(r)[0]
        for i, row in enumerate(r):
            fig, ax1, ax2, ax3, ax4 = _make_scatter(fig, ax1, ax2, ax3, ax4, r=row, times=times, limits=limits, title=title, orbit_index=i, num_orbits=num_orbits, frame=frame)

    # Set axis color to white
    for i, ax in enumerate([ax1, ax2, ax3, ax4]):
        ax.set_facecolor('black')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        if i == 3:
            ax.tick_params(axis='z', colors='white')

    # Set text color to white
    for ax in [ax1, ax2, ax3, ax4]:
        for text in ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.label, ax.yaxis.label]:
            text.set_color('white')
    
    #Save the plot
    fig.patch.set_facecolor('black')

    if save_path:
        if save_path.lower().endswith('.png'):
            save_plot_to_png(fig, save_path)
        else:
            save_plot_to_pdf(fig, save_path)
    return [fig, ax1, ax2, ax3, ax4]


def cislunar_orbit_plot(r, times=[], title='', figsize=(7, 7), save_path=False):
    """
    Parameters
    ----------
    r : (n,3) or array of [(n,3), ..., (n,3)] array_like
        Position of orbiting object(s) in meters.
    times: optional - times when r was calculated.
    limits: optional - x and y limits of the plot
    title: optional - title of the plot
    """
    def _make_scatter(fig, ax1, ax2, r_gcrf, times, title='', orbit_index='', num_orbits=1):
        if np.size(times) < 1:
            r_moon = np.atleast_2d(get_body("moon").position(Time("2000-1-1")))
        else:
            r_moon = get_body("moon").position(times).T

        # Check if the frame is in the dictionary, and set central_dot accordingly
        title2 = "GCRF"
        r_lunar = gcrf_to_lunar_fixed(r_gcrf, times)
        r_moon = gcrf_to_lunar(r_moon, times)

        x_gcrf = r_gcrf[:, 0] / RGEO
        y_gcrf = r_gcrf[:, 1] / RGEO
        z_gcrf = r_gcrf[:, 2] / RGEO
        x_lunar = r_lunar[:, 0] / RGEO
        y_lunar = r_lunar[:, 1] / RGEO
        z_lunar = r_lunar[:, 2] / RGEO
        xm = r_moon[:, 0] / RGEO
        ym = r_moon[:, 1] / RGEO
        zm = r_moon[:, 2] / RGEO
            
        if np.size(xm) > 1:
            gradient_colors = cm.Greys(np.linspace(0, .8, len(xm)))[::-1]
            blues = cm.Blues(np.linspace(.4, .9, len(xm)))[::-1]
        else:
            gradient_colors = "grey"
            blues = 'Blue'
        plot_settings = {
            "gcrf": ("blue", 50, 1, xm, ym, zm, gradient_colors),
            "itrf": ("blue", 50, 1, xm, ym, zm, gradient_colors),
            "lunar": ("blue", 50, 1, xm, ym, zm, gradient_colors),
            "lunar fixed": ("grey", 25, 1.3, -xm, -ym, -zm, blues)
        }

        if orbit_index == '':
            angle = 0
            dotcolors = cm.rainbow(np.linspace(0, 1, len(x)))
        else:
            angle = orbit_index * 10
            dotcolors = cm.rainbow(np.linspace(0, 1, num_orbits))[orbit_index]
        ax_gcrf.scatter3D(x_gcrf, y_gcrf, z_gcrf, color=dotcolors, s=1)
        ax_gcrf.scatter3D(0, 0, 0, color=stn[0], s=stn[1])
        ax_gcrf.scatter3D(stn[3], stn[4], stn[5], color=stn[6], s=5)
        ax_gcrf.set_xlim([-limits, limits])
        ax_gcrf.set_ylim([-limits, limits])
        ax_gcrf.set_zlim([-limits, limits])
        ax_gcrf.set_aspect('equal')  # aspect ratio is 1:1:1 in data space
        ax_gcrf.set_xlabel('x [GEO]')
        ax_gcrf.set_ylabel('y [GEO]')
        ax_gcrf.set_zlabel('z [GEO]')

        ax_lunar.scatter3D(x_lunar, y_lunar, z_lunar, color=dotcolors, s=1)
        ax_lunar.scatter3D(0, 0, 0, color=stn[0], s=stn[1])
        ax_lunar.scatter3D(stn[3], stn[4], stn[5], color=stn[6], s=5)
        ax_lunar.set_xlim([-limits, limits])
        ax_lunar.set_ylim([-limits, limits])
        ax_lunar.set_zlim([-limits, limits])
        ax_lunar.set_aspect('equal')  # aspect ratio is 1:1:1 in data space
        ax_lunar.set_xlabel('x [GEO]')
        ax_lunar.set_ylabel('y [GEO]')
        ax_lunar.set_zlabel('z [GEO]')
        return fig, ax1, ax2
    input_type = check_numpy_array(r)

    plt.rcParams.update({'font.size': 9, 'figure.facecolor': 'k'})
    fig = plt.figure(dpi=100, figsize=figsize, facecolor='black')
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    if input_type == "numpy array":
        fig, ax1, ax2, ax3, ax4 = _make_scatter(fig, ax1, ax2, ax3, ax4, r=r, times=times, title=title)
    if input_type == "list of numpy array":
        num_orbits = np.shape(r)[0]
        for i, row in enumerate(r):
            fig, ax1, ax2, ax3, ax4 = _make_scatter(fig, ax1, ax2, ax3, ax4, r=row, times=times, title=title, orbit_index=i, num_orbits=num_orbits)

    # Set axis color to white
    for i, ax in enumerate([ax1, ax2]):
        ax.set_facecolor('black')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        if i == 3:
            ax.tick_params(axis='z', colors='white')

    # Set text color to white
    for ax in [ax1, ax2, ax3, ax4]:
        for text in ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.label, ax.yaxis.label]:
            text.set_color('white')
    
    #Save the plot
    fig.patch.set_facecolor('black')

    if save_path:
        if save_path.lower().endswith('.png'):
            save_plot_to_png(fig, save_path)
        else:
            save_plot_to_pdf(fig, save_path)
    return [fig, ax1, ax2, ax3, ax4]


def globe_plot(r, times, limits=False, title='', figsize=(7, 8), save_path=False, el=30, az=0, scale=1):
    x = r[:, 0] / RGEO
    y = r[:, 1] / RGEO
    z = r[:, 2] / RGEO
    if limits is False:
        limits = np.nanmax(np.abs([x, y, z])) * 1.2
    
    earth_png = PILImage.open(find_file("earth", ext=".png"))
    earth_png = earth_png.resize((5400 // scale, 2700 // scale))
    bm = np.array(earth_png.resize([int(d) for d in earth_png.size])) / 256.
    lons = np.linspace(-180, 180, bm.shape[1]) * np.pi / 180
    lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi / 180
    mesh_x = np.outer(np.cos(lons), np.cos(lats)).T * 0.15126911409197252
    mesh_y = np.outer(np.sin(lons), np.cos(lats)).T * 0.15126911409197252
    mesh_z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T * 0.15126911409197252

    dotcolors = plt.cm.rainbow(np.linspace(0, 1, len(x)))
    plt.rcParams.update({'font.size': 9, 'figure.facecolor': 'black'})  # Set background color to black
    fig = plt.figure(dpi=100, figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('black')
    ax.tick_params(axis='both', colors='white')
    ax.grid(True, color='grey', linestyle='--', linewidth=0.5)
    ax.set_facecolor('black')  # Set plot background color to black
    ax.scatter(x, y, z, color=dotcolors, s=1)
    ax.plot_surface(mesh_x, mesh_y, mesh_z, rstride=4, cstride=4, facecolors=bm, shade=False)
    ax.view_init(elev=el, azim=az)
    ax.set_xlim([-limits, limits])
    ax.set_ylim([-limits, limits])
    ax.set_zlim([-limits, limits])
    ax.set_xlabel('x [GEO]', color='white')  # Set x-axis label color to white
    ax.set_ylabel('y [GEO]', color='white')  # Set y-axis label color to white
    ax.set_zlabel('z [GEO]', color='white')  # Set z-axis label color to white
    ax.tick_params(axis='x', colors='white')  # Set x-axis tick color to white
    ax.tick_params(axis='y', colors='white')  # Set y-axis tick color to white
    ax.tick_params(axis='z', colors='white')  # Set z-axis tick color to white
    ax.set_aspect('equal')
    if save_path:
        if save_path.lower().endswith('.png'):
            save_plot_to_png(fig, save_path)
        else:
            save_plot_to_pdf(fig, save_path)
    return fig, ax


def tracking_plot(r, times, ground_stations=None, limits=False, title='', figsize=(12, 8), save_path=False, scale=1):
    """
    Create a 3D tracking plot of satellite positions over time on Earth's surface.

    Parameters
    ----------
    r : numpy.ndarray or list of numpy.ndarray
        Satellite positions in GCRF coordinates. If a single numpy array, it represents the satellite's position vector over time. If a list of numpy arrays, it represents multiple satellite position vectors.

    times : numpy.ndarray
        Timestamps corresponding to the satellite positions.

    ground_stations : list of tuples, optional
        List of ground stations represented as (latitude, longitude) pairs. Default is None.

    limits : float or bool, optional
        The plot limits for x, y, and z axes. If a float, it sets the limits for all axes. If False, the limits are automatically determined based on the data. Default is False.

    title : str, optional
        Title for the plot. Default is an empty string.

    figsize : tuple, optional
        Figure size in inches (width, height). Default is (7, 8).

    save_path : str or bool, optional
        Path to save the plot as an image or PDF. If False, the plot is not saved. Default is False.

    scale : int, optional
        Scaling factor for the Earth's image. Default is 5.

    Returns
    -------
    matplotlib.figure.Figure
        The created tracking plot figure.

    Notes
    -----
    - The function supports plotting the positions of one or multiple satellites over time.
    - Ground station locations can be optionally displayed on the plot.
    - The limits parameter can be set to specify the plot's axis limits or automatically determined if set to False.
    - The frame parameter determines the coordinate frame for the satellite positions, "gcrf" (default) or "itrf".

    Example Usage
    -------------
    - Single satellite tracking plot:
      tracking_plot(r_satellite, times, ground_stations=[(40, -75)], title="Satellite Tracking")

    - Multiple satellite tracking plot:
      tracking_plot([r_satellite_1, r_satellite_2], times, title="Multiple Satellite Tracking")

    - Save the plot as a PNG image:
      tracking_plot(r_satellite, times, save_path="satellite_tracking.png")

    - Customize the plot view:
      tracking_plot(r_satellite, times, elev=45, azim=120)

    - Set custom axis limits:
      tracking_plot(r_satellite, times, limits=500)
    """
    def _make_plot(r, times, ground_stations, limits, title, figsize, save_path, scale, orbit_index=''):
        lon, lat, height = groundTrack(r, times)
        lon[np.where(np.abs(np.diff(lon)) >= np.pi)] = np.nan
        lat[np.where(np.abs(np.diff(lat)) >= np.pi)] = np.nan

        x = r[:, 0] / RGEO
        y = r[:, 1] / RGEO
        z = r[:, 2] / RGEO
        if limits is False:
            limits = np.nanmax(np.abs([x, y, z])) * 1.1
        dotcolors = cm.rainbow(np.linspace(0, 1, len(x)))

        # Creating plot
        plt.rcParams.update({'font.size': 9, 'figure.facecolor': 'w'})
        fig = plt.figure(dpi=100, figsize=figsize)
        fig.patch.set_facecolor('black')
        earth_png = PILImage.open(find_file("earth", ext=".png"))
        earth_png = earth_png.resize((5400 // scale, 2700 // scale))
        ax = fig.add_subplot(2, 3, (1, 2))
        ax.imshow(earth_png, extent=[-180, 180, -90, 90])
        ax.plot(np.rad2deg(lon), np.rad2deg(lat))
        if ground_stations is not None:
            for ground_station in ground_stations:
                ax.scatter(ground_station[1], ground_station[0], s=15, color='DarkRed')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('longitude [degrees]', color='white')
        ax.set_ylabel('latitude [degrees]', color='white')
        ax.set_title(title, color='white')
        ax.tick_params(axis='both', colors='white')
        ax.set_aspect('equal')

        ax = fig.add_subplot(2, 3, 3)
        ax.imshow(earth_png, extent=[-180, 180, -90, 90])
        ax.plot(np.rad2deg(lon), np.rad2deg(lat))
        if ground_stations is not None:
            for ground_station in ground_stations:
                ax.scatter(ground_station[1], ground_station[0], s=15, color='DarkRed')
        ax.set_xlim(-150, -60)
        ax.set_ylim(0, 90)
        ax.set_xlabel('longitude [degrees]', color='white')
        ax.set_ylabel('latitude [degrees]', color='white')
        ax.tick_params(axis='both', colors='white')
        ax.set_aspect('equal')

        ax = fig.add_subplot(2, 3, 4)
        ax.scatter(0, 0, color='blue', s=(100 * EARTH_RADIUS / RGEO)**2)
        ax.scatter(x, y, color=dotcolors, s=1)
        ax.set_xlim([-limits, limits])
        ax.set_ylim([-limits, limits])
        ax.set_aspect('equal')  # aspect ratio is 1:1:1 in data space
        ax.set_xlabel('x [GEO]', color='white')
        ax.set_ylabel('y [GEO]', color='white')
        ax.set_title('XY', color='white')
        ax.tick_params(axis='both', colors='white')
        ax.set_facecolor('black')
        ax.grid(True, color='grey', linestyle='--', linewidth=0.5)

        ax = fig.add_subplot(2, 3, 5)
        ax.scatter(0, 0, color='blue', s=(100 * EARTH_RADIUS / RGEO)**2)
        ax.scatter(x, z, color=dotcolors, s=1)
        ax.set_xlim([-limits, limits])
        ax.set_ylim([-limits, limits])
        ax.set_aspect('equal')  # aspect ratio is 1:1:1 in data space
        ax.set_xlabel('x [GEO]', color='white')
        ax.set_ylabel('z [GEO]', color='white')
        ax.set_title('XZ', color='white')
        ax.tick_params(axis='both', colors='white')
        ax.set_facecolor('black')
        ax.grid(True, color='grey', linestyle='--', linewidth=0.5)

        ax = fig.add_subplot(2, 3, 6)
        ax.scatter(0, 0, color='blue', s=(100 * EARTH_RADIUS / RGEO)**2)
        ax.scatter(y, z, color=dotcolors, s=1)
        ax.set_xlim([-limits, limits])
        ax.set_ylim([-limits, limits])
        ax.set_aspect('equal')  # aspect ratio is 1:1:1 in data space
        ax.set_xlabel('y [GEO]', color='white')
        ax.set_ylabel('z [GEO]', color='white')
        ax.set_title('YZ', color='white')
        ax.tick_params(axis='both', colors='white')
        ax.set_facecolor('black')
        ax.grid(True, color='grey', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        if save_path:
            if save_path.lower().endswith('.png'):
                save_plot_to_png(fig, save_path)
            else:
                save_plot_to_pdf(fig, save_path)
        return fig

    input_type = check_numpy_array(r)
    if input_type == "numpy array":
        fig = _make_plot(
            r, times, ground_stations=ground_stations,
            limits=limits, title=title, figsize=figsize,
            save_path=save_path, scale=scale)

    if input_type == "list of numpy array":
        limits_plot = 0
        for i, row in enumerate(r):
            if limits is False and limits_plot < np.nanmax(norm(row) / RGEO) * 1.2:
                limits_plot = np.nanmax(norm(row) / RGEO) * 1.2
            else:
                limits_plot = limits
            fig = _make_plot(
                row, times, ground_stations=ground_stations,
                limits=limits_plot, title=title, figsize=figsize,
                save_path=save_path, scale=scale, orbit_index=i
            )
    return fig


# #####################################################################
# Formatting x axis
# #####################################################################
def date_format(time_array, ax):
    n = 5  # Number of nearly evenly spaced points to select
    time_span_in_months = (time_array[-1].datetime - time_array[0].datetime).days / 30
    if time_span_in_months < 1:
        # Get the time span in hours
        time_span_in_hours = (time_array[-1].datetime - time_array[0].datetime).total_seconds() / 3600

        if time_span_in_hours < 24:
            # If the time span is less than a day, format the x-axis with hh:mm dd-mon
            selected_times = np.linspace(time_array[0], time_array[-1], n)
            selected_hour_strings = [t.strftime('%H:%M') for t in selected_times]
            selected_day_month_strings = [t.strftime('%d-%b') for t in selected_times]
            selected_tick_labels = [f'{hour} {day_month}' for hour, day_month in zip(selected_hour_strings, selected_day_month_strings)]
            selected_decimal_years = [t.decimalyear for t in selected_times]
            # Set the x-axis tick positions and labels
            ax.set_xticks(selected_decimal_years)
            ax.set_xticklabels(selected_tick_labels)
            return
    if n >= time_span_in_months:
        # Get evenly spaced points in the time_array
        selected_indices = np.round(np.linspace(0, len(time_array) - 1, n)).astype(int)
        selected_times = time_array[selected_indices]
        selected_month_year_strings = [t.strftime('%d-%b-%Y') for t in selected_times]
    else:
        # Get the first of n nearly evenly spaced months in the time
        step = int(len(time_array) / (n - 1))
        selected_times = time_array[::step]
        selected_month_year_strings = [t.strftime('%b-%Y') for t in selected_times]
    selected_decimal_years = [t.decimalyear for t in selected_times]
    # Set the x-axis tick positions and labels
    ax.set_xticks(selected_decimal_years)
    ax.set_xticklabels(selected_month_year_strings)

    # Optional: Rotate the tick labels for better visibility
    plt.xticks(rotation=0)


save_plot_to_pdf_call_count = 0


def save_plot_to_pdf(figure, pdf_path):
    global save_plot_to_pdf_call_count
    save_plot_to_pdf_call_count += 1
    if '~' == pdf_path[0]:
        pdf_path = os.path.expanduser(pdf_path)
    if '.' in pdf_path:
        temp_pdf_path = re.sub(r"\.[^.]+$", "_temp.pdf", pdf_path)
    else:
        temp_pdf_path = f"{pdf_path}_temp.pdf"
    # Save the figure as a PNG in-memory using BytesIO
    png_buffer = io.BytesIO()
    figure.savefig(png_buffer, format='png', dpi=300, bbox_inches='tight')
    # Rewind the buffer to the beginning
    png_buffer.seek(0)
    # Open the in-memory PNG using PIL
    png_image = PILImage.open(png_buffer)
    with PdfPages(temp_pdf_path) as pdf:
        # Create a new figure and axis to display the image
        img_fig, img_ax = plt.subplots()
        img_ax.imshow(png_image)
        img_ax.axis('off')
        # Save the figure with the image into the PDF
        pdf.savefig(img_fig, dpi=300, bbox_inches='tight')
    if os.path.exists(pdf_path):
        merger = PdfMerger()
        with open(pdf_path, "rb") as main_pdf, open(temp_pdf_path, "rb") as temp_pdf:
            merger.append(main_pdf)
            merger.append(temp_pdf)
            with open(pdf_path, "wb") as merged_pdf:
                merger.write(merged_pdf)
        os.remove(temp_pdf_path)
    else:
        os.rename(temp_pdf_path, pdf_path)
    plt.close(figure)
    plt.close(img_fig)  # Close the figure and new figure created
    print(f"Saved figure {save_plot_to_pdf_call_count} to {pdf_path}")
    return


def save_plot_to_png(figure, save_path, dpi=200):
    """
    Save a Python figure as a PNG image.

    Parameters:
        figure (matplotlib.figure.Figure): The figure object to be saved.
        save_path (str): The file path where the PNG image will be saved.

    Returns:
        None
    """
    try:
        # Save the figure as a PNG image
        figure.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(figure)  # Close the figure to release resources
        print(f"Figure saved at: {save_path}")
    except Exception as e:
        print(f"Error occurred while saving the figure: {e}")
