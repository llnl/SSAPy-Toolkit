# flake8: noqa: E501

#########################################################################################################################################
#########################################################################################################################################
# ALL MY HELPFUL FRIENDS
#########################################################################################################################################
path_to_cislunar = '/p/lustre2/cislunar/cislunar_data/'
au_to_m = 149597870700

from ssapy.body import get_body
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
import psutil
from psutil._common import bytes2human


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


warnings.filterwarnings("ignore")


def mem_usage():
    mem_usage = psutil.virtual_memory()
    print(mem_usage)
    total_in_human_format = bytes2human(mem_usage[0])
    print(f'Memory used: {total_in_human_format}')
    return


def timenow():
    from datetime import datetime
    current_time = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    print(f'Current time: {current_time}')
    return current_time


def pd_flatten(data, factor=1):
    tmp = []
    for x in data:
        try:
            tmp.extend(x[1:-1].split(','))
        except TypeError:
            tmp.append(x)
    return [float(x) / factor for x in tmp]


def str_to_array(s):
    s = s.replace('[', '').replace(']', '')  # Remove square brackets
    return np.array([float(x) for x in s.split(',')])


def pdstr_to_arrays(df):
    return df.apply(str_to_array).to_numpy()


def random_arr(low=0, high=1, size=(1, 10), dtype='float64'):
    if 'int' in dtype:
        return np.random.randint(low, high + 1, size, dtype=dtype)
    else:

        return np.random.uniform(low, high, size)


def b2str(array_):
    return [i.decode("utf-8") for i in array_]


def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]


def nan_array(size=1):
    x = np.zeros(size)
    x[:] = np.NaN
    return x


def remove_np_nans(numpy_array):
    return numpy_array[~np.isnan(numpy_array)]


def remove_zeros(data, axis=1):
    return data[~np.all(data == 0, axis=axis)]


def nby3shape(arr_):
    if arr_.ndim == 1:
        return np.reshape(arr_, (1, 3))
    if arr_.ndim == 2:
        if np.shape(arr_)[1] == 3:
            return arr_
        else:
            return arr_.T


def eformat(f, prec, exp_digits):
    s = "%.*e" % (prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%+0*d" % (mantissa, exp_digits + 1, int(exp))


def extractNum(s):
    numre = re.compile('[0-9]+')
    return int(numre.search(s).group())


def sortbynum(files):
    if len(files[0].split('/')) > 1:
        files_shortened = []
        file_prefix = '/'.join(files[0].split('/')[:-1])
        for file in files:
            files_shortened.append(file.split('/')[-1])
        files_sorted = sorted(files_shortened, key=lambda x: float(re.findall("(\d+)", x)[0]))
        sorted_files = []
        for file in files_sorted:
            sorted_files.append(f'{file_prefix}/{file}')
    else:
        sorted_files = sorted(files, key=lambda x: float(re.findall("(\d+)", x)[0]))
    return sorted_files


def issorted(test_list):
    flag = False
    if test_list == sorted(test_list):
        flag = True
    if flag:
        print("Yes, List is sorted.")
    else:
        print("No, List is not sorted.")
    return flag


def byte2str(byte_string):
    try:
        return [x.decode("utf-8") for x in byte_string]
    except (AttributeError, TypeError):
        return byte_string.decode("utf-8")


def flatten(t):
    return [item for sublist in t for item in sublist]


def ETA(idx, total_num, start_loop_time):
    eta = (total_num - idx) * (timer() - start_loop_time) / 60
    if eta > 60:
        print(f'ETA: {eta/60:.1f} hours. {idx} of {total_num}')
    else:
        print(f'ETA: {eta:.1f} minutes. {idx} of {total_num}')
    return


def elapsed_time(start_time):
    delta_t = (timer() - start_time)
    if delta_t < .1:
        return print(f'Elapsed time: {delta_t*1000:.2f} ms.')
    if delta_t < 60:
        return print(f'Elapsed time: {delta_t:.2f} seconds.')
    elif delta_t >= 60 and delta_t < 3600:
        return print(f'Elapsed time: {delta_t/60:.2f} minutes.')
    elif delta_t >= 3600:
        return print(f'Elapsed time: {delta_t/3600:.2f} hours.')


def size(a, axis=None):
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


def sortbylist(X, Y):
    return [x for _, x in sorted(zip(Y, X))]


def find_nearest(array, value=0):
    array = np.asarray(array)
    idx = np.nanargmin((np.abs(array - value)))
    return idx, array[idx] - value


def sample(seq, n, replacement=False):
    return np.random.choice(seq, n, replacement)


def rand_num(low=0, high=1):
    return float(np.random.uniform(low, high, 1).astype('float64'))


def isint(var_):
    return isinstance(var_, int) or np.issubdtype(var_, np.integer)


def isfloat(var_):
    return isinstance(var_, float)


def isstr(var_):
    return isinstance(var_, str)


def shuffle(x):
    return random.shuffle(x)


def divby0(n, d, Δ=np.nan):
    return n / d if d else Δ


def kde(data_):
    kde = stats.gaussian_kde(data_)
    return kde


def body_pos(body='earth', t=None, coord='icrs', date=2451545.0, format='jd'):
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


def close_to_any(a, floats, **kwargs):
    return np.any(np.isclose(a, floats, **kwargs))


def distance3d(x, y, z, xe, ye, ze):
    distance = ((x - xe) ** 2 + (y - ye) ** 2 + (z - ze) ** 2) ** (1 / 2)
    return distance


def find_local_extrema(arr):
    if len(np.shape(arr)) > 1:
        print(f"array shape: {np.shape(arr)}, reducing along last axis.")
        arr = np.linalg.norm(arr, axis=-1)
        find_local_extrema(arr)
    len_data = len(arr)
    minima_indices = [i for i in range(1, len_data - 1) if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]]
    maxima_indices = [i for i in range(1, len_data - 1) if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]]
    return minima_indices, maxima_indices


def graham_scan(points):
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


def contours_2d(points, plot=False):
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


def contours_3d(points_3d, plot=False):
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