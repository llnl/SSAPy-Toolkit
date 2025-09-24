# flake8: noqa: E501
from pandas import DataFrame
from ssapy.body import get_body
import scipy
from scipy import stats
import numpy as np
import random
from timeit import default_timer as timer
import os
import sys
import warnings
from contextlib import contextmanager
import re
from astropy.time import Time
from astropy import units as u
from pathlib import Path

@contextmanager
def suppress_stdout():
    """
    Context manager to suppress stdout.

    Redirects output to devnull during the context.
    Author: Travis Yeager (yeager7@llnl.gov)
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


warnings.filterwarnings("ignore")


def pd_flatten(data: list, factor: float = 1.0) -> list:
    """
    Flattens the data and converts each element to a float, dividing by a factor.

    Parameters:
        data (list): list of string elements to be converted.
        factor (float): Factor to divide each number (default is 1.0).

    Returns:
        list: list of converted floats.
    Author: Travis Yeager (yeager7@llnl.gov)
    """
    # Filter out empty strings or any non-convertible strings
    cleaned_data = [x for x in data if x.strip() != '']
    
    # Attempt to convert to float
    return [float(x) / factor for x in cleaned_data if x.strip().replace('.', '', 1).isdigit()]



def str_to_array(s: str) -> np.ndarray:
    """
    Converts a string of comma-separated values into a numpy array of floats.

    Parameters:
        s (str): Input string.

    Returns:
        np.ndarray: Array of floats converted from the input string.
    Author: Travis Yeager (yeager7@llnl.gov)
    """
    s = s.replace('[', '').replace(']', '')  # Remove square brackets
    return np.array([float(x) for x in s.split(',')])


def pdstr_to_arrays(df: DataFrame) -> np.ndarray:
    """
    Converts all string representations of arrays in a dataframe to numpy arrays.

    Parameters:
        df (pd.DataFrame): DataFrame containing string representations of arrays.

    Returns:
        np.ndarray: Numpy array of the converted arrays.
    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return df.apply(str_to_array).to_numpy()


def random_arr(low: float = 0, high: float = 1, size: tuple = (1, 10), dtype: str = 'float64') -> np.ndarray:
    """
    Generates a random array with specified bounds and size.

    Parameters:
        low (float): The lower bound for random values (default is 0).
        high (float): The upper bound for random values (default is 1).
        size (tuple): The size of the generated array (default is (1, 10)).
        dtype (str): The data type for the array elements (default is 'float64').

    Returns:
        np.ndarray: The generated random array.
    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if 'int' in dtype:
        return np.random.randint(low, high + 1, size, dtype=dtype)
    else:
        return np.random.uniform(low, high, size)


def b2str(array_: np.ndarray) -> list:
    """
    Decodes a list of byte strings into regular strings.

    Parameters:
        array_ (np.ndarray): Array of byte strings to decode.

    Returns:
        list: list of decoded strings.
    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return [i.decode("utf-8") for i in array_]


def find_indices(lst: list, condition: callable) -> list:
    """
    Finds indices in a list that satisfy a given condition.

    Parameters:
        lst (list): list to search through.
        condition (callable): A function that takes an element and returns True or False.

    Returns:
        list: list of indices where the condition is true.
    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return [i for i, elem in enumerate(lst) if condition(elem)]


def nan_array(size: int = 1) -> np.ndarray:
    """
    Creates an array of NaNs with the specified size.

    Parameters:
        size (int): The size of the array (default is 1).

    Returns:
        np.ndarray: An array filled with NaNs.
    Author: Travis Yeager (yeager7@llnl.gov)
    """
    x = np.zeros(size)
    x[:] = np.NaN
    return x


def remove_np_nans(numpy_array: np.ndarray) -> np.ndarray:
    """
    Removes NaN values from a numpy array.

    Parameters:
        numpy_array (np.ndarray): Input array with potential NaN values.

    Returns:
        np.ndarray: Array with NaN values removed.
    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return numpy_array[~np.isnan(numpy_array)]

def remove_zeros(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Removes rows or columns where all elements are zeros.

    Parameters:
        data (np.ndarray): Input array.
        axis (int): Axis along which to remove zeros (default is 1, i.e., columns).

    Returns:
        np.ndarray: Array with rows/columns removed where all elements were zero.
    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return data[~np.all(data == 0, axis=axis)]


def nby3shape(arr_: np.ndarray) -> np.ndarray:
    """
    Reshapes a 1D or 2D array to have 3 columns, preserving the structure.

    Parameters:
        arr_ (np.ndarray): Input array to reshape.

    Returns:
        np.ndarray: Reshaped array with 3 columns.
    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if arr_.ndim == 1:
        return np.reshape(arr_, (1, 3))
    if arr_.ndim == 2:
        if np.shape(arr_)[1] == 3:
            return arr_
        else:
            return arr_.T


def eformat(f: float, prec: int, exp_digits: int) -> str:
    """
    Formats a floating-point number into scientific notation with the specified precision 
    and exponent width.
    
    Parameters:
        f (float): The number to format.
        prec (int): The number of digits to show after the decimal point.
        exp_digits (int): The number of digits to show in the exponent part.
    
    Returns:
        str: The formatted string in scientific notation.
    Author: Travis Yeager (yeager7@llnl.gov)
    """
    s = "%.*e" % (prec, f)
    mantissa, exp = s.split('e')
    
    # Remove unnecessary leading zeros in the exponent part
    exp = f"{int(exp):+0{exp_digits}d}"
    
    return "%se%s" % (mantissa, exp)


def extractNum(s: str) -> int:
    """
    Extracts the first integer from a string.

    Parameters:
        s (str): The string to extract the integer from.

    Returns:
        int: The first integer found in the string.
    Author: Travis Yeager (yeager7@llnl.gov)
    """
    numre = re.compile('[0-9]+')
    return int(numre.search(s).group())


def sortbynum(files: list) -> list:
    """
    Sorts a list of file paths based on numbers embedded in the filenames.

    Parameters:
        files (list[str]): list of file paths to sort.

    Returns:
        list[str]: Sorted list of file paths.
    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if len(files[0].split('/')) > 1:
        files_shortened = []
        file_prefix = '/'.join(files[0].split('/')[:-1])
        for file in files:
            files_shortened.append(file.split('/')[-1])
        files_sorted = sorted(files_shortened, key=lambda x: float(re.findall(r"(\d+)", x)[0]))
        sorted_files = []
        for file in files_sorted:
            sorted_files.append(f'{file_prefix}/{file}')
    else:
        sorted_files = sorted(files, key=lambda x: float(re.findall(r"(\d+)", x)[0]))
    return sorted_files


def issorted(test_list: list) -> bool:
    """
    Checks if a list is sorted.

    Parameters:
        test_list (list): The list to check.

    Returns:
        bool: True if the list is sorted, False otherwise.
    Author: Travis Yeager (yeager7@llnl.gov)
    """
    flag = False
    if test_list == sorted(test_list):
        flag = True
    if flag:
        print("Yes, list is sorted.")
    else:
        print("No, list is not sorted.")
    return flag


def byte2str(byte_string: bytes) -> str:
    """
    Converts a byte string into a regular string.

    Parameters:
        byte_string (bytes): The byte string to convert.

    Returns:
        str: The decoded string.
    Author: Travis Yeager (yeager7@llnl.gov)
    """
    try:
        return [x.decode("utf-8") for x in byte_string]
    except (AttributeError, TypeError):
        return byte_string.decode("utf-8")


def flatten(t: list) -> list:
    """
    Flattens a list of lists into a single list.

    Parameters:
        t (list): The input list of lists.

    Returns:
        list: A flattened list containing all the elements.
    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return [item for sublist in t for item in sublist]


def ETA(idx: int, total_num: int, start_loop_time: float) -> None:
    """
    Estimates the remaining time to completion.

    Parameters:
        idx (int): The current iteration index.
        total_num (int): The total number of iterations.
        start_loop_time (float): The start time of the loop, used to calculate elapsed time.

    Returns:
        None

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    eta = (total_num - idx) * (timer() - start_loop_time) / 60
    if eta > 60:
        print(f'ETA: {eta/60:.1f} hours. {idx} of {total_num}')
    else:
        print(f'ETA: {eta:.1f} minutes. {idx} of {total_num}')
    return


def elapsed_time(start_time: float) -> None:
    """
    Prints the elapsed time since a given start time.

    Parameters:
        start_time (float): The time at which the process started.

    Returns:
        None

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    delta_t = (timer() - start_time)
    if delta_t < .1:
        return print(f'Elapsed time: {delta_t*1000:.2f} ms.')
    if delta_t < 60:
        return print(f'Elapsed time: {delta_t:.2f} seconds.')
    elif delta_t >= 60 and delta_t < 3600:
        return print(f'Elapsed time: {delta_t/60:.2f} minutes.')
    elif delta_t >= 3600:
        return print(f'Elapsed time: {delta_t/3600:.2f} hours.')


def size(a: np.ndarray, axis=None) -> int:
    """
    Returns the size of an array or the length of a specified axis.

    Parameters:
        a (np.ndarray): Input array.
        axis (int): Axis to check the size along (default is None).

    Returns:
        int: Size of the array or length of the specified axis.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if axis is None:
        try:
            return a.size
        except AttributeError:
            return np.asarray(a).size
    else:
        try:
            return a.shape[axis]
        except AttributeError:
            return np.asarray(a).shape[axis]


def sortbylist(X: list, Y: list) -> list:
    """
    Sorts one list based on the sorting order of another list.

    Parameters:
        X (list): The list to sort.
        Y (list): The list used to determine the sorting order.

    Returns:
        list: Sorted version of X according to the order in Y.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return [x for _, x in sorted(zip(Y, X))]


def find_nearest(array: np.ndarray, value: float = 0) -> tuple:
    """
    Finds the index and difference of the nearest element in an array to a given value.

    Parameters:
        array (np.ndarray): The input array.
        value (float): The value to find the nearest element to (default is 0).

    Returns:
        tuple: Index of the nearest element and the difference from the value.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    array = np.asarray(array)
    idx = np.nanargmin((np.abs(array - value)))
    return idx, array[idx] - value


def sample(seq: list, n: int, replacement: bool = False) -> np.ndarray:
    """
    Randomly samples n elements from a sequence with or without replacement.

    Parameters:
        seq (list): The sequence to sample from.
        n (int): Number of elements to sample.
        replacement (bool): Whether sampling is done with replacement (default is False).

    Returns:
        np.ndarray: Randomly sampled elements.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return np.random.choice(seq, n, replacement)


def rand_num(low: float = 0, high: float = 1) -> float:
    """
    Generates a random float between the specified bounds.

    Parameters:
        low (float): The lower bound (default is 0).
        high (float): The upper bound (default is 1).

    Returns:
        float: Random number between the specified bounds.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return float(np.random.uniform(low, high, 1).astype('float64'))


def isint(var_: object) -> bool:
    """
    Checks if a variable is an integer.

    Parameters:
        var_ (object): The variable to check.

    Returns:
        bool: True if the variable is an integer, False otherwise.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return isinstance(var_, int)


def isfloat(var_: object) -> bool:
    """
    Checks if a variable is a float.

    Parameters:
        var_ (object): The variable to check.

    Returns:
        bool: True if the variable is a float, False otherwise.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return isinstance(var_, float)


def isstr(var_: object) -> bool:
    """
    Checks if a variable is a string.

    Parameters:
        var_ (object): The variable to check.

    Returns:
        bool: True if the variable is a string, False otherwise.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return isinstance(var_, str)


def shuffle(x: list) -> None:
    """
    Shuffles the elements of a list in place.

    Parameters:
        x (list): The list to shuffle.

    Returns:
        None

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return random.shuffle(x)


def divby0(n: float, d: float, Δ: float = np.nan) -> float:
    """
    Performs division while handling division by zero.

    Parameters:
        n (float): The numerator.
        d (float): The denominator.
        Δ (float): The value to return in case of division by zero (default is NaN).

    Returns:
        float: The result of division, or the default value in case of zero denominator.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return n / d if d else Δ


def kde(data_: np.ndarray) -> stats.gaussian_kde:
    """
    Computes the Kernel Density Estimate (KDE) of the given data.

    Parameters:
        data_ (np.ndarray): Input data to estimate the density of.

    Returns:
        stats.gaussian_kde: KDE object.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    kde = stats.gaussian_kde(data_)
    return kde


def body_pos(body: str = 'earth', t: float = None, coord: str = 'icrs', 
             date: float = 2451545.0, format: str = 'jd') -> np.ndarray:
    """
    Computes the position of a celestial body at a given time.

    Parameters:
        body (str): Name of the celestial body (default is 'earth').
        t (float): Time to evaluate (default is None, uses `date`).
        coord (str): Coordinate system (default is 'icrs').
        date (float): Julian date if `t` is not provided (default is 2451545.0).
        format (str): Time format (default is 'jd').

    Returns:
        np.ndarray: Position of the body in the specified coordinate system.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if t is None:
        t = Time(date, format=format)
    if coord == 'heliocentricmeanecliptic':
        return get_body(body, t).heliocentricmeanecliptic.cartesian.get_xyz().to(u.m).value
    if coord == 'gcrs':
        return get_body(body, t).gcrs.cartesian.get_xyz().to(u.m).value
    if coord == 'icrs':
        return get_body(body, t).icrs.cartesian.get_xyz().to(u.m).value
    if coord == 'barycentricmeanecliptic':
        return get_body(body, t).barycentricmeanecliptic.cartesian.get_xyz().to(u.m).value
    if coord == 'barycentrictrueecliptic':
        return get_body(body, t).barycentrictrueecliptic.cartesian.get_xyz().to(u.m).value


def close_to_any(a: np.ndarray, floats: np.ndarray, **kwargs) -> bool:
    """
    Checks if any element in the array is close to any element in the list of floats.

    Parameters:
        a (np.ndarray): The array to check.
        floats (np.ndarray): The list of floats to compare to.

    Returns:
        bool: True if any element in `a` is close to any in `floats`, False otherwise.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return np.any(np.isclose(a, floats, **kwargs))


def distance3d(x: float, y: float, z: float, xe: float, ye: float, ze: float) -> float:
    """
    Calculates the 3D Euclidean distance between two points.

    Parameters:
        x, y, z (float): Coordinates of the first point.
        xe, ye, ze (float): Coordinates of the second point.

    Returns:
        float: The Euclidean distance between the two points.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    distance = ((x - xe) ** 2 + (y - ye) ** 2 + (z - ze) ** 2) ** (1 / 2)
    return distance


def find_local_extrema(arr: np.ndarray) -> list:
    """
    Finds the local minima and maxima in a 1D array.

    Parameters:
        arr (np.ndarray): The input array.

    Returns:
        tuple: list of indices of local minima and maxima.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if len(np.shape(arr)) > 1:
        print(f"array shape: {np.shape(arr)}, reducing along last axis.")
        arr = np.linalg.norm(arr, axis=-1)
        find_local_extrema(arr)
    len_data = len(arr)
    minima_indices = [i for i in range(1, len_data - 1) if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]]
    maxima_indices = [i for i in range(1, len_data - 1) if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]]
    return minima_indices, maxima_indices


def graham_scan(points: np.ndarray) -> np.ndarray:
    """
    Computes the convex hull of a set of 2D points using Graham's scan algorithm.

    Parameters:
        points (np.ndarray): Array of points, where each point is an (x, y) coordinate.

    Returns:
        np.ndarray: Array of points forming the convex hull.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    # Select the anchor point with minimum x and minimum y values
    anchor_point_index = np.lexsort((points[:, 1], points[:, 0]))[0]
    anchor_point = points[anchor_point_index]

    # Sort the points based on polar angle and distance from anchor point
    sorted_points = sorted(points, key=lambda p: (np.arctan2(p[1] - anchor_point[1], p[0] - anchor_point[0]), np.linalg.norm(p - anchor_point)))

    convex_hull = [anchor_point, sorted_points[0], sorted_points[1]]

    for i in range(2, len(sorted_points)):
        while len(convex_hull) > 1 and orientation(convex_hull[-2], convex_hull[-1], sorted_points[i]) != 2:
            convex_hull.pop()
        convex_hull.append(sorted_points[i])
    return np.array(convex_hull)


def contours_2d(points: np.ndarray, plot: bool = False) -> np.ndarray:
    """
    Computes and optionally plots the convex hull of a set of 2D points.

    Parameters:
        points (np.ndarray): Array of points, where each point is an (x, y) coordinate.
        plot (bool): Whether to plot the convex hull (default is False).

    Returns:
        np.ndarray: Array of points forming the convex hull.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    hull_vertices = graham_scan(points)
    if plot:
        import matplotlib.pyplot as plt
        plt.scatter(points[:, 0], points[:, 1], label='Points')
        plt.plot(np.append(hull_vertices[:, 0], hull_vertices[0, 0]),
                np.append(hull_vertices[:, 1], hull_vertices[0, 1]), 'r--', lw=2)

        plt.fill(hull_vertices[:, 0], hull_vertices[:, 1], alpha=0.2, color='blue', label='Convex Hull Area')
        plt.legend(loc='upper left')
        plt.xlim((0, 20))
        plt.ylim((0, 20))
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title("Bounding Contour using Graham's scan")
        plt.show()
    return hull_vertices


def contours_3d(points_3d: np.ndarray, plot: bool = False) -> scipy.spatial.ConvexHull:
    """
    Computes and optionally plots the convex hull of a set of 3D points.

    Parameters:
        points_3d (np.ndarray): Array of 3D points, where each point is an (x, y, z) coordinate.
        plot (bool): Whether to plot the convex hull (default is False).

    Returns:
        scipy.spatial.ConvexHull: The convex hull object representing the 3D hull.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    from scipy.spatial import ConvexHull
    # Compute Convex Hull using scipy's ConvexHull
    hull = ConvexHull(points_3d)
    if plot:
        import matplotlib.pyplot as plt
        # Plotting for visualization in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], label='Points')

        # Plotting the convex hull
        for simplex in hull.simplices:
            simplex = np.append(simplex, simplex[0])  # Close the loop
            ax.plot(points_3d[simplex, 0], points_3d[simplex, 1], points_3d[simplex, 2], 'r--', lw=2)

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title("Convex Hull using scipy's ConvexHull in 3D")
        plt.legend()
        plt.show()
    return hull


ENV_VAR = "YEAGER_UTILS_CACHE"
DEFAULT_FILE = "covariance_sigmas.npy"


def get_sigmas(n=25, path=None):
    """
    Returns an (n, 6) array of random sigma perturbations.
    Cached to disk in a user-configurable directory.

    Resolution order for cache *directory* (when path is None):
      1) Env var YEAGER_UTILS_CACHE (file stored as covariance_sigmas.npy)
      2) ~/.cache/yeager_utils/covariance_sigmas.npy

    If `path` is provided, it is treated as an explicit file path and used as-is.
    """
    if path is None:
        from .IO import datapath

        # Prefer env-provided directory, then fallback to ~/.cache/yeager_utils
        env_dir = os.environ.get(ENV_VAR, "").strip()
        dirs = []
        if env_dir:
            dirs.append(Path(env_dir).expanduser())
        dirs.append(Path.home() / ".cache" / "yeager_utils")

        # Use datapath to choose/create the first usable directory
        path = Path(datapath(DEFAULT_FILE, dirs=dirs))
    else:
        path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    compute_new = True
    if path.exists():
        print("Loading sigmas.")
        try:
            sigmas = np.load(path)
            compute_new = not (sigmas.ndim == 2 and sigmas.shape[0] == n and sigmas.shape[1] == 6)
        except Exception:
            # Corrupt or incompatible file -> recompute
            compute_new = True

    if compute_new:
        print("Computing sigmas.")
        sigmas = np.zeros((n, 6))
        for i in range(n):
            x_dist, v_dist = 10, 1
            while True:
                tmpx, tmpy, tmpz = np.random.uniform(-x_dist, x_dist, 3)
                tmpvx, tmpvy, tmpvz = np.random.uniform(-v_dist, v_dist, 3)
                if (
                    np.sqrt(tmpx**2 + tmpy**2 + tmpz**2) <= x_dist
                    and np.sqrt(tmpvx**2 + tmpvy**2 + tmpvz**2) <= v_dist
                ):
                    break
            sigmas[i, :] = [tmpx, tmpy, tmpz, tmpvx, tmpvy, tmpvz]
        np.save(path, sigmas)

    return sigmas
