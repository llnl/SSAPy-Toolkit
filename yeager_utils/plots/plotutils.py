import numpy as np
from PIL import Image as PILImage
import io
from PyPDF2 import PdfMerger
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import Image as IPythonImage, display as ipython_display
import os
import re
import matplotlib.pyplot as plt
from matplotlib.colors import cnames, to_rgb, rgb2hex
from ssapy.utils import find_file
from ..constants import EARTH_RADIUS, MOON_RADIUS
from ..time import Time
from ..vectors import rotation_matrix_from_vectors
from typing import Any, Union
import numpy as np
from enum import Enum, auto
from typing import Any, List, Tuple, Union
from astropy.time import Time


class VarType(Enum):
    NONE = auto()
    TIME = auto()
    ARRAY = auto()
    LIST_ARRAYS = auto()
    LIST_LISTS = auto()
    MIXED_LIST = auto()
    OTHER = auto()


def is_list_of_arrays(lst: list) -> bool:
    return all(isinstance(item, np.ndarray) for item in lst)


def is_list_of_lists(lst: list) -> bool:
    return all(isinstance(item, list) for item in lst)


def check_type(var: Any) -> VarType:
    """
    Classify ‘var’ into one of VarType cases.
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


def valid_orbits(
    r: Union[np.ndarray, List[np.ndarray], list, float, int],
    t: Union[np.ndarray, List[np.ndarray], Time, List[Time], float, int, None]
) -> Tuple[List[np.ndarray], List[Time]]:
    """
    Normalize r and t into parallel lists of shape-(n,3) arrays and Time objects.
    """

    def to_array3(x) -> np.ndarray:
        """Convert input to shape-(n,3) ndarray"""
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
            raise ValueError(f"‘r’ must be convertible to shape-(n,3); got {type(r)}")
    elif r_type == VarType.LIST_ARRAYS:
        r_list = [to_array3(ri) for ri in r]
    else:
        raise ValueError(f"‘r’ must be an ndarray or list of ndarrays; got {r_type}")

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
        raise ValueError(f"‘t’ must be None, float, ndarray, list of ndarrays, or Time object(s); got {t_type}")

    try:
        print(f"Returning arrays shaped: {np.shape(r_list)}, {np.shape(t_list)}")
    except Exception as e:
        print(f"Returning arrays with varying shapes: {type(r_list)=}, {type(t_list)=}, error={e}")

    return r_list, t_list


def load_earth_file() -> 'PILImage':
    earth = PILImage.open(find_file("earth", ext=".png"))
    earth = earth.resize((5400 // 5, 2700 // 5))
    return earth


def drawEarth(time: Union[np.ndarray, 'Time'], ngrid: int = 100, R: float = EARTH_RADIUS, rfactor: float = 1):
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

    Author:
    -------
    Travis Yeager (yeager7@llnl.gov)
    """
    import ipyvolume as ipv

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

    # Need earth rotation angle for t
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


def load_moon_file() -> 'PILImage':
    moon = PILImage.open(find_file("moon", ext=".png"))
    moon = moon.resize((5400 // 5, 2700 // 5))
    return moon


def drawMoon(time: Union[np.ndarray, 'Time'], ngrid: int = 100, R: float = MOON_RADIUS, rfactor: float = 1):
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
    moon = load_moon_file()

    from numbers import Real
    from erfa import gst94
    import ipyvolume as ipv

    lat = np.linspace(-np.pi / 2, np.pi / 2, ngrid)
    lon = np.linspace(-np.pi, np.pi, ngrid)
    lat, lon = np.meshgrid(lat, lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    u = np.linspace(0, 1, ngrid)
    v, u = np.meshgrid(u, u)

    # Need earth rotation angle for t
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
        texture=moon
    )



save_plot_to_pdf_call_count = 0


def save_plot_to_pdf(figure: plt.Figure, pdf_path: str) -> None:
    """
    Save a Matplotlib figure as a PNG embedded in a PDF file.

    This function saves the figure as a temporary PNG image in memory and
    then embeds it into a PDF file. If the specified PDF already exists,
    the figure is appended to it. Otherwise, a new PDF file is created.

    Parameters:
        figure (matplotlib.figure.Figure): The figure object to be saved.
        pdf_path (str): The path to the PDF file where the figure will be saved.

    Returns:
        None

    Author:
    -------
    Travis Yeager (yeager7@llnl.gov)
    """
    global save_plot_to_pdf_call_count
    save_plot_to_pdf_call_count += 1

    # Expand user directory if ~ is in the path
    if pdf_path.startswith('~'):
        pdf_path = os.path.expanduser(pdf_path)

    # Generate a temporary PDF path by replacing the original extension
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

    # Create the temporary PDF with the PNG image
    with PdfPages(temp_pdf_path) as pdf:
        # Create a new figure and axis to display the image
        img_fig, img_ax = plt.subplots()
        img_ax.imshow(png_image)
        img_ax.axis('off')  # Hide axis for clean image display
        # Save the image as a page in the PDF
        pdf.savefig(img_fig, dpi=300, bbox_inches='tight')

    # If the PDF already exists, merge the new page
    if os.path.exists(pdf_path):
        merger = PdfMerger()
        with open(pdf_path, "rb") as main_pdf, open(temp_pdf_path, "rb") as temp_pdf:
            merger.append(main_pdf)
            merger.append(temp_pdf)
            with open(pdf_path, "wb") as merged_pdf:
                merger.write(merged_pdf)
        os.remove(temp_pdf_path)
    else:
        # If the PDF doesn't exist, rename the temporary PDF to the desired path
        os.rename(temp_pdf_path, pdf_path)

    # Close all figures to release resources
    plt.close(figure)
    plt.close(img_fig)

    # Print the success message with the call count
    print(f"Saved figure {save_plot_to_pdf_call_count} to {pdf_path}")


def save_plot(figure: plt.Figure, save_path: str, dpi: int = 200) -> None:
    """
    Save a Python figure as a JPG image.

    Parameters:
        figure (matplotlib.figure.Figure): The figure object to be saved.
        save_path (str): The file path where the PNG image will be saved.
        dpi (int, optional): The resolution of the saved image. Default is 200.

    Returns:
        None

    Author:
    -------
    Travis Yeager (yeager7@llnl.gov)
    """
    if save_path.lower().endswith('.pdf'):
        save_plot_to_pdf(figure, save_path)
        return
    try:
        base_name, extension = os.path.splitext(save_path)
        if extension.lower() != '.jpg':
            save_path = base_name + '.jpg'
        # Save the figure as a PNG image
        figure.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(figure)  # Close the figure to release resources
        print(f"Figure saved at: {save_path}")
    except Exception as e:
        print(f"Error occurred while saving the figure: {e}")


def display_figure(figname: str, display: str = 'IPython') -> None:
    def open_image(filename: str) -> None:
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


def make_white(fig: plt.Figure, *axes: plt.Axes) -> Tuple[plt.Figure, Tuple[plt.Axes]]:
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


def make_black(fig: plt.Figure, *axes: plt.Axes) -> Tuple[plt.Figure, Tuple[plt.Axes]]:
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


def draw_dashed_circle(ax: plt.Axes, normal_vector: np.ndarray, radius: float, dashes: int, dash_length: float = 0.1, label: str = 'Dashed Circle') -> None:
    # Define the circle in the xy-plane
    theta = np.linspace(0, 2 * np.pi, 1000)
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    z_circle = np.zeros_like(theta)

    # Stack the coordinates into a matrix
    circle_points = np.vstack((x_circle, y_circle, z_circle)).T

    # Create the rotation matrix to align z-axis with the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    rotation_matrix = rotation_matrix_from_vectors(np.array([0, 0, 1]), normal_vector)

    # Rotate the circle points
    rotated_points = circle_points @ rotation_matrix.T

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


def create_sphere(cx: float, cy: float, cz: float, r: float, resolution: int = 360) -> np.ndarray:
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


def drawSphere(xCenter: float, yCenter: float, zCenter: float, r: float, res: complex = 10j, flatten: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def darken(color: str, amount: float = 0.5) -> List[str]:
    import colorsys
    try:
        c = cnames[color]
    except Exception:
        c = color
    colors = []
    for i in amount:
        c = colorsys.rgb_to_hls(*to_rgb(c))
        colors.append(colorsys.hls_to_rgb(c[0], 1 - i * (1 - c[1]), c[2]))
    return colors


def rgb(minimum: float, maximum: float, value: float) -> Tuple[int, int, int]:
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
    g = 255 - b - r
    return r, g, b


def generate_rainbow_colors(num_iterations: int) -> List[str]:
    cmap = plt.get_cmap('rainbow')
    colors = [rgb2hex(cmap(i / num_iterations)) for i in range(num_iterations)]
    return colors


def write_video(video_name: str, frames: List[str], fps: int = 30) -> None:
    print(f'Writing video: {video_name}')
    """
    Writes frames to an mp4 video file
    :param video_name: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate

    Author:
    -------
    Travis Yeager (yeager7@llnl.gov)
    """
    import cv2

    img = cv2.imread(frames[0])
    h, w, layers = img.shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(video_name, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(cv2.imread(frame))

    writer.release()
    print(f'Wrote: {video_name}')
    return


def write_gif(gif_name: str, frames: List[str], fps: int = 30) -> None:
    import imageio

    print(f'Writing gif: {gif_name}')
    with imageio.get_writer(gif_name, mode='I', duration=1 / fps) as writer:
        for i, filename in enumerate(frames):
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f'Wrote {gif_name}')
    return
