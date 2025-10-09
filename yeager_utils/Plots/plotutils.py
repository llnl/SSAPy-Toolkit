# --- Standard library ---
import io
import os
import re
from enum import Enum, auto
from numbers import Real

# --- Third-party ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import cnames, to_rgb, rgb2hex
from PIL import Image as PILImage
from PyPDF2 import PdfMerger
from IPython.display import Image as IPythonImage, display as ipython_display
from astropy.time import Time
from erfa import gst94

# --- Local modules ---
from ssapy.utils import find_file
from ..constants import EARTH_RADIUS, MOON_RADIUS
from ..vectors import rotation_matrix_from_vectors


class VarType(Enum):
    NONE = auto()
    TIME = auto()
    ARRAY = auto()
    LIST_ARRAYS = auto()
    LIST_LISTS = auto()
    MIXED_LIST = auto()
    OTHER = auto()


def is_list_of_arrays(lst):
    return all(isinstance(item, np.ndarray) for item in lst)


def is_list_of_lists(lst):
    return all(isinstance(item, list) for item in lst)


def check_type(var):
    """
    Classify 'var' into one of VarType cases.
    """
    if var is None:
        return VarType.NONE

    if isinstance(var, Time):
        return VarType.TIME

    if isinstance(var, np.ndarray):
        return VarType.ARRAY

    if isinstance(var, list):
        if len(var) == 0:
            return VarType.OTHER
        if is_list_of_arrays(var):
            if all(isinstance(item.flat[0], Time) for item in var if item.size > 0):
                return VarType.TIME
            return VarType.LIST_ARRAYS
        if is_list_of_lists(var):
            return VarType.LIST_LISTS
        if all(isinstance(item, Time) for item in var):
            return VarType.TIME
        return VarType.MIXED_LIST

    return VarType.OTHER


def valid_orbits(r, t):
    """
    Normalize r and t into parallel lists of shape-(n,3) arrays and Time objects.

    Parameters
    ----------
    r : ndarray, list of ndarrays, list, float, or int
    t : ndarray, list of ndarrays, astropy.time.Time, list of Time, float, int, or None

    Returns
    -------
    (list_of_arrays, list_of_Time)
    """

    def to_array3(x):
        """Convert input to shape-(n,3) ndarray."""
        arr = np.asarray(x, dtype=float).squeeze()
        if arr.ndim == 1 and arr.size == 3:
            return arr.reshape(1, 3)
        elif arr.ndim == 2 and arr.shape in [(3, 1), (1, 3)]:
            return arr.reshape(1, 3)
        elif arr.ndim == 2 and arr.shape[1] == 3:
            return arr
        raise ValueError(f"Cannot interpret r shape: {arr.shape}")

    # ---- classify and normalize r ----
    r_type = check_type(r)
    if r_type in {VarType.ARRAY, VarType.LIST_LISTS, VarType.MIXED_LIST, VarType.OTHER}:
        try:
            r_list = [to_array3(r)]
        except Exception:
            raise ValueError("'r' must be convertible to shape-(n,3); got {}".format(type(r)))
    elif r_type == VarType.LIST_ARRAYS:
        r_list = [to_array3(ri) for ri in r]
    else:
        raise ValueError("'r' must be an ndarray or list of ndarrays; got {}".format(r_type))

    # ---- classify t ----
    t_type = check_type(t)

    # ---- build t_list ----
    if t_type == VarType.NONE:
        t_list = [Time(np.zeros(len(rr)), format="gps") for rr in r_list]

    elif t_type == VarType.ARRAY:
        if not all(len(t) == len(rr) for rr in r_list):
            raise ValueError("Single t-array length must match all r-array lengths")
        t_list = [Time(t, format="gps") for _ in r_list]

    elif t_type == VarType.LIST_ARRAYS:
        if len(t) != len(r_list):
            raise ValueError("Number of t-arrays must equal number of r-arrays")
        t_list = []
        for rr, tt in zip(r_list, t):
            if len(tt) != len(rr):
                raise ValueError("Each t-array must match its corresponding r-array length")
            t_list.append(Time(tt, format="gps"))

    elif t_type == VarType.TIME:
        if isinstance(t, Time):
            if t.isscalar:
                t_list = [Time(np.full(len(rr), t.value), format=t.format) for rr in r_list]
            else:
                if not all(len(t) == len(rr) for rr in r_list):
                    raise ValueError("Single Time object length must match all r-array lengths")
                t_list = [t for _ in r_list]
        elif isinstance(t, list) and all(isinstance(tt, Time) for tt in t):
            if len(t) != len(r_list):
                raise ValueError("Number of Time objects must equal number of r-arrays")
            if not all(len(tt) == len(rr) for tt, rr in zip(t, r_list)):
                raise ValueError("Each Time object must match its corresponding r-array length")
            t_list = t
        else:
            raise ValueError("Unsupported Time object input format.")

    elif isinstance(t, (int, float)):
        t_list = [Time(np.full(len(rr), t), format="gps") for rr in r_list]

    else:
        raise ValueError("'t' must be None, float, ndarray, list of ndarrays, or Time object(s); got {}".format(t_type))

    try:
        print("Returning arrays shaped: {}, {}".format(np.shape(r_list), np.shape(t_list)))
    except Exception as e:
        print("Returning arrays with varying shapes: type(r_list)={}, type(t_list)={}, error={}".format(type(r_list), type(t_list), e))

    return r_list, t_list


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
    ngrid : int
        Number of grid points in Earth model.
    R : float
        Earth radius in meters. Default is WGS84 value.
    rfactor : float
        Factor by which to enlarge Earth (for visualization purposes)
    """
    import ipyvolume as ipv

    earth = load_earth_file()

    lat = np.linspace(-np.pi / 2, np.pi / 2, ngrid)
    lon = np.linspace(-np.pi, np.pi, ngrid)
    lat, lon = np.meshgrid(lat, lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    u = np.linspace(0, 1, ngrid)
    v, u = np.meshgrid(u, u)

    # Earth rotation angle for t (approximate, visualization only)
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
        u=u, v=v, wireframe=False, texture=earth
    )


def load_moon_file():
    moon = PILImage.open(find_file("moon", ext=".png"))
    moon = moon.resize((5400 // 5, 2700 // 5))
    return moon


def drawMoon(time, ngrid=100, R=MOON_RADIUS, rfactor=1):
    """
    Parameters
    ----------
    time : array_like or astropy.time.Time (n,)
        If float (array), then should correspond to GPS seconds;
        i.e., seconds since 1980-01-06 00:00:00 UTC
    ngrid : int
        Number of grid points in Moon model.
    R : float
        Moon radius in meters.
    rfactor : float
        Factor by which to enlarge Moon (for visualization purposes)
    """
    import ipyvolume as ipv

    moon = load_moon_file()

    lat = np.linspace(-np.pi / 2, np.pi / 2, ngrid)
    lon = np.linspace(-np.pi, np.pi, ngrid)
    lat, lon = np.meshgrid(lat, lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    u = np.linspace(0, 1, ngrid)
    v, u = np.meshgrid(u, u)

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
        u=u, v=v, wireframe=False, texture=moon
    )


save_plot_to_pdf_call_count = 0


def save_plot_to_pdf(figure, pdf_path):
    """
    Save a Matplotlib figure as a PNG embedded in a PDF file.

    If the specified PDF already exists, append a new page; otherwise create it.
    """
    global save_plot_to_pdf_call_count
    save_plot_to_pdf_call_count += 1

    # Expand user directory if ~ is in the path
    if pdf_path.startswith('~'):
        pdf_path = os.path.expanduser(pdf_path)

    # Temporary PDF path
    if '.' in pdf_path:
        temp_pdf_path = re.sub(r"\.[^.]+$", "_temp.pdf", pdf_path)
    else:
        temp_pdf_path = f"{pdf_path}_temp.pdf"

    # Save the figure as a PNG in-memory using BytesIO
    png_buffer = io.BytesIO()
    figure.savefig(png_buffer, format='png', dpi=300, bbox_inches='tight')
    png_buffer.seek(0)

    # Open the in-memory PNG using PIL
    png_image = PILImage.open(png_buffer)

    # Create the temporary PDF with the PNG image
    with PdfPages(temp_pdf_path) as pdf:
        img_fig, img_ax = plt.subplots()
        img_ax.imshow(png_image)
        img_ax.axis('off')
        pdf.savefig(img_fig, dpi=300, bbox_inches='tight')

    # Merge or move into place
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
    plt.close(img_fig)

    print(f"Saved figure {save_plot_to_pdf_call_count} to {pdf_path}")


def save_plot(figure, save_path, dpi=200):
    """
    Save a Matplotlib figure as JPG (or append to PDF if save_path ends with .pdf).
    """
    if save_path.lower().endswith('.pdf'):
        save_plot_to_pdf(figure, save_path)
        return
    try:
        base_name, extension = os.path.splitext(save_path)
        if extension.lower() != '.jpg':
            save_path = base_name + '.jpg'
        figure.savefig(save_path, dpi=dpi, bbox_inches=None)
        plt.close(figure)
        print(f"Figure saved at: {save_path}")
    except Exception as e:
        print(f"Error occurred while saving the figure: {e}")


def display_figure(figname, display='IPython'):
    def open_image(filename):
        if display == 'IPython':
            img = IPythonImage(filename=filename)
            ipython_display(img)
        elif display == 'PIL':
            img = PILImage.open(filename)
            img.show()
        else:
            raise ValueError("Invalid display option. Please specify 'IPython' or 'PIL'.")

    if os.path.isfile(figname):
        open_image(figname)
        return

    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
    for ext in image_extensions:
        filename_with_ext = figname + ext
        if os.path.isfile(filename_with_ext):
            open_image(filename_with_ext)
            return

    print("No image file found.")


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

    return fig, axes


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

    return fig, axes


def draw_dashed_circle(ax, normal_vector, radius, dashes, dash_length=0.1, label='Dashed Circle'):
    # Define the circle in the xy-plane
    theta = np.linspace(0, 2 * np.pi, 1000)
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    z_circle = np.zeros_like(theta)

    # Stack the coordinates into a matrix
    circle_points = np.vstack((x_circle, y_circle, z_circle)).T

    # Create the rotation matrix to align z-axis with the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    rot = rotation_matrix_from_vectors(np.array([0, 0, 1]), normal_vector)

    # Rotate the circle points
    rotated_points = circle_points @ rot.T

    # Create dashed effect
    dash_points = []
    dash_gap = int(len(theta) / dashes)
    for i in range(dashes):
        start_idx = i * dash_gap
        end_idx = start_idx + int(dash_length * len(theta))
        dash_points.append(rotated_points[start_idx:end_idx])

    # Plot the dashed circle in 3D
    for points in dash_points:
        ax.plot(points[:, 0], points[:, 1], points[:, 2], 'k--', label=label)
        label = None  # Only one label


def create_sphere(cx, cy, cz, r, resolution=360):
    """
    Create sphere coordinates with center (cx, cy, cz) and radius r.

    Returns
    -------
    np.ndarray of shape (3, 2*resolution, resolution)
    """
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
    """
    Darken a color by reducing its lightness.

    Parameters
    ----------
    color : str
        Named color or hex string.
    amount : float or iterable of floats in [0,1]
        0 -> no change, 1 -> black. Iterable returns multiple shades.

    Returns
    -------
    list of RGB tuples in 0..1
    """
    import colorsys

    # Resolve base color
    try:
        base = cnames[color]
    except Exception:
        base = color

    base_rgb = to_rgb(base)  # 0..1
    h, l, s = colorsys.rgb_to_hls(*base_rgb)

    # Normalize amount to iterable
    try:
        iterator = iter(amount)
    except TypeError:
        iterator = [amount]

    out = []
    for a in iterator:
        a = float(a)
        a = min(max(a, 0.0), 1.0)
        new_l = 1 - a * (1 - l)
        out.append(colorsys.hls_to_rgb(h, new_l, s))
    return out


def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
    g = 255 - b - r
    return r, g, b


def generate_rainbow_colors(num_iterations):
    cmap = plt.get_cmap('rainbow')
    colors = [rgb2hex(cmap(i / num_iterations)) for i in range(num_iterations)]
    return colors



