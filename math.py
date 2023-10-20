#########################################################################################################################################
#########################################################################################################################################
# ALL MY HELPFUL FRIENDS
#########################################################################################################################################
path_to_cislunar = '/p/lustre2/cislunar/cislunar_data/'

import ssapy
from ssapy.constants import RGEO

from scipy import stats
from scipy.signal import argrelextrema
from scipy.signal import find_peaks

import numpy as np
import math
import h5py
import fcntl
import csv
import threading
import pandas as pd
import random

from timeit import default_timer as timer

import pickle
import os
import sys
import warnings

from contextlib import contextmanager

import re

import shutil
from glob import glob

import rebound
from rebound import hash as h


# from astropy.table import QTable, Table, Column
# from astropy import constants as astc
# from astropy.coordinates import solar_system_ephemeris, get_body, get_sun, get_moon, get_body_barycentric_posvel, get_body_barycentric, SkyCoord, ICRS, GCRS, CartesianRepresentation, CartesianDifferential
from astropy.time import Time
# from astropy.modeling.models import BlackBody
from astropy import units as u


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

import psutil
from psutil._common import bytes2human


def mem_usage():
    mem_usage = psutil.virtual_memory()
    print(mem_usage)
    total_in_human_format = bytes2human(mem_usage[0])
    print(f'Memory used: {total_in_human_format}')
    return


def timenow():
    current_time = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    print(f'Current time: {current_time}')
    return current_time


W_rho = 19280  # kg/m^3 --> density of Tungsten
LD = 384399000  # lunar semi-major axis in meters
earth_rad = ssapy.constants.WGS84_EARTH_RADIUS
moon_rad = 1738.1e3
geo_alt = 35786000
earth_mu = ssapy.constants.WGS84_EARTH_MU
######################################################################
# Conversions
######################################################################
class units():
    #distances
    au_to_m = 149597870700
    pc_to_au = 206265
    pc_to_m = 3.085677581e16
    km_to_m = 1000
    #angles
    deg_to_arcsecond = 3600
    rad_to_arcsecond = 206265
    rad_to_deg = 57.3
    #Time
    day_to_second = 86400
    year_to_second = 31557600
    year_to_minute = 525960
    year_to_hour = 8766
    year_to_day = 365.25
    year_to_week = 365.25/7
    year_to_month = 365.25/12
    #mass
    kg_to_g = 1000
    #default rebound to SI
    v_rebound_to_si = 4744 * 2*np.pi #au/2pi * yr to m/s
    aupyr_to_mps = 4744
    
######################################################################
# Constants
######################################################################
class c():
    c = 299792458 # speed of light m/s
    G = 6.67408e-11 #Gravitational constant m3 kg-1 s-2
    kb = 1.38064852e-23 #boltzmann constant m2 kg s-2 K-1
    pi = np.pi
class mass():#https://en.wikipedia.org/wiki/Planetary_mass#IAU_current_best_estimates_(2012)
    sun = 1.98847e30
    mercury = 3.3010e23
    venus = 4.1380e24
    earth = 5.9722e24
    moon = 7.34767309e22
    mars = 6.4273e23
    jupiter = 1.89852e27
    saturn = 5.6846e26
    uranus = 8.6819e25
    neptune = 1.02431e26
class mu():
    sun = 1.32712440018e20
    mercury = 2.2032e13
    venus = 3.24859e14
    earth = 3.986004418e14
    moon = 4.9048695e12
    mars = 4.282837e13
    jupiter = 1.26686534e17
    saturn = 3.7931187e16
    uranus = 5.793939e15
    neptune = 6.836529e15
class planet_radius():
    sun = 696340.0e3
    mercury = 2439.7e3
    venus = 6051.8e3
    earth = 6378.1e3
    moon = 1738.1e3
    mars = 3396.2e3
    jupiter = 71492e3
    saturn = 60268e3
    uranus = 25559e3
    neptune = 24764e3
    pluto = 1195e3


class planet_a():
    mercury = 0.3871
    venus = 0.7233
    earth = 1.000
    mars = 1.5273
    jupiter = 5.2028
    saturn = 9.5388
    uranus = 19.1914
    neptune = 30.0611


class hill_radius():  # in meters
    mercury = 0.1753e9
    venus = 1.0042e9
    earth = 1.4714e9
    mars = 0.9827e9
    jupiter = 50.5736e9
    saturn = 61.6340e9
    uranus = 66.7831e9
    neptune = 115.0307e9
    ceres = 0.2048e9
    pluto = 5.9921e9
    eris = 8.1176e9
######################################################################
# Unicodes
######################################################################


def hat(symbol_):
    return f'{symbol_}\u0302'


class symbol():
    # unicode	character	description
    Delta = '\u0394'  # Δ	GREEK CAPITAL LETTER DELTA
    Omega = '\u03A9'  # Ω	GREEK CAPITAL LETTER OMEGA
    pi = '\u03C0'  # π	GREEK SMALL LETTER PI
    degree = '\u00B0'  # °	DEGREE SYMBOL
    ihat = 'i\u0302'  # î	i HAT
    jhat = 'j\u0302'  # ĵ	j HAT
    khat = 'k\u0302'  # k̂	k HAT
    uhat = 'u\u0302'  # û	u HAT
    alpha = '\u03B1'  # α	GREEK SMALL LETTER ALPHA
    beta = '\u03B2'  # β	GREEK SMALL LETTER BETA
    gamma = '\u03B3'  # γ	GREEK SMALL LETTER GAMMA
    delta = '\u03B4'  # δ	GREEK SMALL LETTER DELTA
    epsilon = '\u03B5'  # ε	GREEK SMALL LETTER EPSILON
    zeta = '\u03B6'  # ζ	GREEK SMALL LETTER ZETA
    eta = '\u03B7'  # η	GREEK SMALL LETTER ETA
    theta = '\u03B8'  # θ	GREEK SMALL LETTER THETA
    iota = '\u03B9'  # ι	GREEK SMALL LETTER IOTA
    kappa = '\u03BA'  # κ	GREEK SMALL LETTER KAPPA
    lmbda = '\u03BB'  # λ	GREEK SMALL LETTER LAMDA
    mu = '\u03BC'  # μ	GREEK SMALL LETTER MU
    nu = '\u03BD'  # ν	GREEK SMALL LETTER NU
    xi = '\u03BE'  # ξ	GREEK SMALL LETTER XI
    omicron = '\u03BF'  # GREEK SMALL LETTER OMICRON
    pi = '\u03C0'  # π	GREEK SMALL LETTER PI
    rho = '\u03C1'  # ρ	GREEK SMALL LETTER RHO
    finalsigma = '\u03C2'  # ς	GREEK SMALL LETTER FINAL SIGMA
    sigma = '\u03C3'  # σ	GREEK SMALL LETTER SIGMA
    tau = '\u03C4'  # τ	GREEK SMALL LETTER TAU
    upsilon = '\u03C5'  # υ	GREEK SMALL LETTER UPSILON
    phi = '\u03C6'  # φ	GREEK SMALL LETTER PHI
    chi = '\u03C7'  # χ	GREEK SMALL LETTER CHI
    psi = '\u03C8'  # ψ	GREEK SMALL LETTER PSI
    omega = '\u03C9'  # ω	GREEK SMALL LETTER OMEGA
    #Greek upper case letters
    Alpha = '\u0391'	#Α	GREEK CAPITAL LETTER ALPHA
    Beta = '\u0392'	#Β	GREEK CAPITAL LETTER BETA
    Gamma = '\u0393'	#Γ	GREEK CAPITAL LETTER GAMMA
    Delta = '\u0394'	#Δ	GREEK CAPITAL LETTER DELTA
    Epsilon = '\u0395'	#Ε	GREEK CAPITAL LETTER EPSILON
    Zeta = '\u0396'	#Ζ	GREEK CAPITAL LETTER ZETA
    Eta = '\u0397'	#Η	GREEK CAPITAL LETTER ETA
    Theta = '\u03F4'	#Θ	GREEK CAPITAL LETTER THETA
    Iota = '\u0399'	#Ι	GREEK CAPITAL LETTER IOTA
    Kappa = '\u039A'	#Κ	GREEK CAPITAL LETTER KAPPA
    Lambda = '\u039B'	#Λ	GREEK CAPITAL LETTER LAMDA
    Mu = '\u039C'	#Μ	GREEK CAPITAL LETTER MU
    Nu = '\u039D'	#Ν	GREEK CAPITAL LETTER NU
    Xi = '\u039E'	#Ξ	GREEK CAPITAL LETTER XI
    Omicron = '\u039F'	#Ο	GREEK CAPITAL LETTER OMICRON
    Pi = '\u03A0'	#Π	GREEK CAPITAL LETTER PI
    Rho = '\u03A1'	#Ρ	GREEK CAPITAL LETTER RHO
    Sigma = '\u03A3'	#Σ	GREEK CAPITAL LETTER SIGMA
    Tau = '\u03A4'	#Τ	GREEK CAPITAL LETTER TAU
    Upsilon = '\u03A5'	#Υ	GREEK CAPITAL LETTER UPSILON
    Phi = '\u03A6'	#Φ	GREEK CAPITAL LETTER PHI
    Chi = '\u03A7'	#Χ	GREEK CAPITAL LETTER CHI
    Pi = '\u03A8'	#Ψ	GREEK CAPITAL LETTER PSI
    Omega = '\u03A9'	#Ω	GREEK CAPITAL LETTER OMEGA

######################################################################
# Load and Save Functions
######################################################################
######################################################################
# Pickles
######################################################################
def psave(filename_, data_):
    from six.moves import cPickle as pickle #for performance
    with open(filename_, 'wb') as f:
        pickle.dump(data_, f)
    f.close()
    return
    
def pload(filename_):
    from six.moves import cPickle as pickle #for performance
    try:
        #print('Openning: ' + current_filename)
        with open(filename_,'rb') as f:
            data = pickle.load(f)
        f.close()
    except (EOFError, FileNotFoundError, OSError, pickle.UnpicklingError) as err:
        print(f'{err} - current_filename')
        return []
    return data
def merge_dicts(file_names, save_path):
    number_of_files = len(file_names); master_dict = {}
    for count, file in enumerate(file_names):
        print(f'Merging dict: {count+1} of {number_of_files}, name: {file}, num of master keys: {len(master_dict.keys())}, num of new keys: {len(master_dict.keys())}')
        master_dict.update(pload(file))
    print('Beginning final save.')
    psave(save_path, master_dict)
    return
######################################################################
# Sliceable Numpys save and load
######################################################################
def npsave(filename_, data_):
    try:
        with open(filename_, 'wb') as f:
            data = np.save(filename_, data_, allow_pickle = True)
        f.close()
    except (EOFError, FileNotFoundError, OSError, pickle.UnpicklingError) as err:
        print(f'{err} - saving')
        return
        
def npload(filename_):
    try:
        with open(filename_,'rb') as f:
            data = np.load(filename_, allow_pickle = True)
        f.close()
    except (EOFError, FileNotFoundError, OSError, pickle.UnpicklingError) as err:
        print(f'{err} - loading')
        return []
    return data

######################################################################
# HDF5 py files h5py
######################################################################
def h5append(filename_, pathname_, append_data):
    path_data_old = h5load(filename_, pathname_)
    with h5py.File(filename_, 'r+') as f:
        path_data_old = np.array(f.get(pathname_))
        del f[pathname_]
    f.close()
    
    new_data_ = np.append(path_data_old,np.array(append_data))
    h5save(filename_, pathname_, data_ = new_data_)
    return

def h5overwrite(filename_, pathname_, new_data_):
    try:
        # Write data to HDF5
        with h5py.File(filename_, "a") as f:
            f.create_dataset(pathname_, data=new_data_, maxshape=None)
        f.close()
    except (FileNotFoundError, ValueError, KeyError) as err:
        # print(f'Overwriting key: {pathname_} exists in file: {filename_}. Err: {err}')
        try:
            with h5py.File(filename_, 'r+') as f:
                del f[pathname_]
            f.close()
        except (FileNotFoundError, ValueError, KeyError) as err:
            print(f'Error: {err}')
        # Write data to HDF5
        try:
            with h5py.File(filename_, "a") as f:
                f.create_dataset(pathname_, data=new_data_, maxshape=None)
            f.close()
        except (FileNotFoundError, ValueError, KeyError) as err:
            print(f'File: {filename_}{pathname_}, Error: {err}')
    return

def h5save(filename_, pathname_, data_):
    try:
        # Write data to HDF5
        with h5py.File(filename_, "a") as f:
            f.create_dataset(pathname_, data=data_, maxshape=None)
        f.close()
    except ValueError as err:
        print(f'Did not save, key: {pathname_} exists in file: {filename_}. {err}')
#     print(f'Saved data to file: {filename_} at path: {pathname_}')
    return
def h5save_with_lock(filename_, pathname_, data_):
    try:
        # Create a file lock
        lock_file = f"{filename_}.lock"
        with open(lock_file, "wb") as lf:
            # Acquire the lock
            fcntl.flock(lf, fcntl.LOCK_EX)

            # Write data to HDF5
            with h5py.File(filename_, "a") as f:
                f.create_dataset(pathname_, data=data_, maxshape=None)
            f.close()

            # Release the lock
            fcntl.flock(lf, fcntl.LOCK_UN)
    except ValueError as err:
        print(f'Did not save, key: {pathname_} exists in file: {filename_}. {err}')
    return        
def h5load(filename_, pathname_):
    with h5py.File(filename_, 'r') as f:
        data = np.array(f.get(pathname_))
    f.close()
    return data

def h5loadall(filename_):
    return_data = {}
    with h5py.File(filename_, "r") as f:
        # List all groups
        keys = list(f.keys())
        for key in keys:
            # Get the data
            return_data[f'{key}'] = np.array(f.get(f'{key}'))
    f.close()
    return return_data, keys

def h5tree(filename_):
    import nexusformat.nexus as nx
    f = nx.nxload(filename_); print(f.tree)
    return f
    
def h5keys(filename_):    
    with h5py.File(filename_, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        group_keys = list(f.keys())
    return group_keys
def h5_key_exists(filename, key):
    try:
        with h5py.File(filename, 'r') as f:
            if str(key) in f:
                return True
            else:
                return False
    except IOError:
#         print("Error opening file:", filename)
        return False

######################################################################
# CSV
######################################################################
def makedf(df):
    if isinstance(df, pd.DataFrame):
        pass
    else:
        df = pd.DataFrame(data = df)
    return df
def header_csv(file_name, sep = '\t'):
    with open(file_name, 'r') as infile:
        reader = csv.DictReader(infile, delimiter=sep)
        fieldnames = reader.fieldnames
    return fieldnames
def read_csv(file_name, sep = '\t', col = False, to_np = False, on_bad_lines='error', drop_nan=False, skiprows=[]):
    if col is False:
        print(f'Reading: {file_name}')
        if drop_nan:
            df = pd.read_csv(file_name, sep=sep, on_bad_lines=on_bad_lines, skiprows=skiprows).dropna()
        else:
            df = pd.read_csv(file_name, sep=sep, on_bad_lines=on_bad_lines, skiprows=skiprows)
    else:
        print(f'Reading column: {col} of: {file_name}')
        if not type(col) is list:
            col = [col]
        if drop_nan:
            df = pd.read_csv(file_name, sep=sep, usecols=col, on_bad_lines=on_bad_lines, skiprows=skiprows).dropna()
        else:
            df = pd.read_csv(file_name, sep=sep, usecols=col, on_bad_lines=on_bad_lines, skiprows=skiprows)
        return np.squeeze(df.to_numpy())
    if to_np is False:
        return df
    else:
        return np.squeeze(df.to_numpy())

def save_csv(file_name, df, sep='\t'):
    df = makedf(df)
    df.to_csv(file_name, index=False, sep = sep)
    print(f'Saved {file_name} successfully.')
    return

def append_csv(file_names, save_path='', sep='\t'):
    pandas_version = pd.__version__
    print(f"Using pandas version: {pandas_version}")
    
    if int(pandas_version.split(".")[0]) < 2:
        df1 = pd.read_csv(file_names[0], sep=sep)
        number_of_files = len(file_names)
        for count, file in enumerate(file_names[1:]):
            df2 = pd.read_csv(file, sep='\t')
            print(f'Appending file: {count+2} of {number_of_files}, size of master: {np.shape(df1)}, size of appender: {np.shape(df2)}, name: {file}')
            df1 = df1.append(df2, ignore_index=True)
        print('Beginning final save.')
        save_csv(save_path, df1, sep=sep)
    else:
        dataframes = []
        for file in file_names:
            df = pd.read_csv(file, sep=sep)
            dataframes.append(df)
        combined_df = pd.concat(dataframes, ignore_index=True)
        print('Beginning final save.')
        save_csv(save_path, combined_df, sep=sep)
def save_csv_line(file_name, df, sep='\t'):
    df = makedf(df)
    if exists(file_name):
        df.to_csv(file_name, mode='a', index=False, header=False, sep = sep)
    else:
        save_csv(file_name, df, sep=sep)
    return

# Create a lock to synchronize access to the file
file_lock = threading.Lock()
def exists_in_csv(csv_file, column_name, number, sep='\t'):
    with file_lock:
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f, delimiter=sep)
                for row in reader:
                    if row[column_name] == str(number):
                        return True
        except IOError as e:
#             print(f"Error occurred while processing CSV file: {csv_file} {e}")
            return False
    return False

def pd_flatten(data, factor=1):
    tmp = []
    for x in data:
        try:
            tmp.extend(x[1:-1].split(','))
        except TypeError as err:
            tmp.append(x)
    return [float(x)/factor for x in tmp]

#TURN AN ARRAY SAVED AS A STRING BACK INTO AN ARRAY
def str_to_array(s):
    s = s.replace('[', '').replace(']', '')  # Remove square brackets
    return np.array([float(x) for x in s.split(',')])
def pdstr_to_arrays(df):
    return df.apply(str_to_array).to_numpy()
######################################################################
# Array Stuff
######################################################################
def random_arr(low=0, high=1, size=(1, 10), dtype='float64'):
    if 'int' in dtype:
        return np.random.randint(low, high+1, size, dtype=dtype)
    else:

        return np.random.uniform(low, high, size)
    
def b2str(array_):
    return [i.decode("utf-8") for i in array_]

def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]

def nan_array(size = 1):
    x = np.zeros(size); x[:] = np.NaN;
    return x
    
def remove_np_nans(numpy_array):
    return numpy_array[~np.isnan(numpy_array)]
def remove_zeros(data, axis = 1):
    return data[~np.all(data == 0, axis=axis)]
def nby3shape(arr_):
    if arr_.ndim == 1:
        return np.reshape(arr_, (1,3))
    if arr_.ndim == 2:
        if np.shape(arr_)[1] == 3:
            return arr_
        else:
            return arr_.T
def rotation_matrix_from_vectors(vec1, vec2=np.array([1, 0, 0])):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

######################################################################
# Numpy Stuff
######################################################################
def mean(arr_):
    return np.mean(arr_)
def median(arr_):
    return np.median(arr_)
def mode(arr_):
    return stats.mode(arr_)
def ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)
def floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)
    
def local_min(a):
    minima = argrelextrema(a, np.less_equal)[0]
    return minima
def local_max(a, distance = 1, height=None, threshold=None, prominence=None, width=None):
    return find_peaks(a, distance, height, threshold, prominence, width)[0]
    
def caclulate_errors(data, CI=.05):
    data_median = [];
    data = np.sort(data)
    median_ = np.nanmedian(data)
    data_median.append(median_)
    err = [median_-data[int(CI*len(data))],data[int((1-CI)*len(data))]-median_]
    return err, data_median
#Format scientific notation string
def eformat(f, prec, exp_digits):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%+0*d"%(mantissa, exp_digits+1, int(exp)) 
######################################################################
# Check if file or directory exists
######################################################################
def exists(pathname):
    if os.path.isdir(pathname):
        exists = True
    elif os.path.isfile(pathname):
        exists = True
    else:
        exists = False
    return exists
######################################################################
# move directory
######################################################################
def mvdir(source_, destination_):
    print(f'Moving {source_} to {destination_}') 
    shutil.move(source_, destination_)
    return
######################################################################
# remove directory
######################################################################
def rmdir(source_):
    if not exists(source_):
        print(f'{source_}, does not exist, no delete.')
    else:
        print(f'Deleted {source_}') 
        shutil.rmtree(source_)
    return
######################################################################
# remove file
######################################################################
def rmfile(pathname):
    if exists(pathname):
        os.remove(pathname)
        print("File: '%s' deleted." %pathname)        
    return
######################################################################
# make directory
######################################################################
def mkdir(pathname):
    if not exists(pathname):
        os.makedirs(pathname)
        print("Directory '%s' created" %pathname)        
    return

######################################################################
# list files in directory
######################################################################
def listdir(dir_path='*', files_only = False, exclude=False):
    dir_path = f'{dir_path}*'
    if files_only:
        try:
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            print(f'{len(files)} files in {dir_path}')
        except (UnboundLocalError, EOFError, FileNotFoundError, OSError, pickle.UnpicklingError) as err:
            onlyfiles = []
            print(f'Error: {err}')
    else:
        files = glob(dir_path)
        print(f'{len(files)} files in {dir_path}')
    if exclude:
        new_files = []
        for file in files:
            if exclude not in file:
                new_files.append(file)
        files = new_files
    return sorted(files)


def extractNum(s):
    numre = re.compile('[0-9]+')
    return int(numre.search(s).group())
    
def sortbynum(files):
    #check if file is a full path
    if len(files[0].split('/')) > 1:
        files_shortened = []
        file_prefix = '/'.join(files[0].split('/')[:-1])
        for file in files:
            files_shortened.append(file.split('/')[-1])
        files_sorted = sorted(files_shortened, key=lambda x:float(re.findall("(\d+)",x)[0]))
        sorted_files = []
        for file in files_sorted:
            sorted_files.append(f'{file_prefix}/{file}')
    else:
        sorted_files = sorted(files, key=lambda x:float(re.findall("(\d+)",x)[0]))
    return sorted_files

def issorted(test_list):
    flag = False
    if(test_list == sorted(test_list)):
        flag = True
    if (flag) :
        print ("Yes, List is sorted.")
    else :
        print ("No, List is not sorted.")
    return flag
def byte2str(byte_string):
    try:
        return [x.decode("utf-8") for x in byte_string]
    except (AttributeError, TypeError) as err:
        return byte_string.decode("utf-8")
######################################################################
# list all files in directory and subdirectories
######################################################################
def allfiles(dirName=os.getcwd()):
    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    return listOfFiles
        
######################################################################
# flatten list of lists
######################################################################
def flatten(t):
    return [item for sublist in t for item in sublist]

######################################################################
# Timing function
######################################################################
def ETA(idx, total_num, start_loop_time):
    eta = (total_num - idx) * (timer() - start_loop_time)/60
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
######################################################################
# Size of an object
######################################################################
def size(a, axis=None):
    if axis is None:
        try:
            return a.size
        except AttributeError:
            return asarray(a).size
    else:
        try:
            return a.shape[axis]
        except AttributeError:
            return asarray(a).shape[axis]

def sortbylist(x,y):
    [x for _, x in sorted(zip(Y, X))]
    return x

def find_nearest(array, value=0):
    array = np.asarray(array)
    idx = np.nanargmin((np.abs(array - value)))
    return idx, array[idx]-value
######################################################################
# Statistics
######################################################################
def sample(seq, n, replacement=False):
    return np.random.choice(seq, n, replacement)
    
def rand_num(low=0, high=1):
    return float(np.random.uniform(low, high, 1).astype('float64'))
######################################################################
# Random utilities
######################################################################
def isint(var_):
    return isinstance(var_, int) or np.issubdtype(var_, np.integer)

def isfloat(var_):
    return isinstance(var_, float)

def isstr(var_):
    return isinstance(var_, str)

def shuffle(x):
    return random.shuffle(x)

def FFT(data, time_between_samples = 1):
    N = len(data)
    t = np.linspace(0, len(data)*time_between_samples, time_between_samples)
    k = int(N/2)
    f = np.linspace(0.0, 1/(2*time_between_samples), N//2)
    Y = np.abs(np.fft.fft(data))[:k]
    return (f, Y)

def FFTP(data, time_between_samples = 1):
    N = len(data)
    t = np.linspace(0, len(data)*time_between_samples, time_between_samples)
    k = int(N/2)
    f = np.linspace(0.0, 1/(2*time_between_samples), N//2)
    Tp = [divby0(1,float(item), len(data)*time_between_samples) for item in f]
    Y = np.abs(np.fft.fft(data))[:k]
    return (Tp, Y)

def divby0(n, d, Δ = np.nan):
    return n / d if d else Δ
    
######################################################################
# Einsum Operations
######################################################################
def einsum_norm(a, indices='ij,ji->i'):
    return np.sqrt(np.einsum(indices, a, a))

######################################################################
# Build KDE
######################################################################
def kde(data_):
    kde = stats.gaussian_kde(data_)
    return kde
    

######################################################################
# Astropy Wrapped Function
######################################################################
def astTime(date='today', format='jd'):
    if date == 'today' or date == 'now':
        return Time.now()
    else:
        return Time(date,format=format)
def body_pos(body='earth', t=None, coord='icrs', date=2451545.0, format='jd'):
    if t is None:
        t = astTime(date, format)
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

def moon_pos(t=None, date=2451545.0, format='jd'):
    if t is None:
        t = astTime(date, format) 
    return get_moon(t).cartesian.get_xyz().to(u.m)

def body_posvel(body='earth', t=None, date=2451545.0, format='jd'):#Default is ICRS coordinates
    if t is None:
        t = astTime(date, format)        
    posvel = get_body_barycentric_posvel(body, t)
    pos = posvel[0].xyz.to("m").value
    vel = posvel[1].xyz.to("m/s").value
    return [pos, vel]


def gcrf_to_radec(gcrf_coords):
    x, y, z = gcrf_coords
    # Calculate right ascension in radians
    ra = np.arctan2(y, x)
    # Convert right ascension to degrees
    ra_deg = np.degrees(ra)
    # Normalize right ascension to the range [0, 360)
    ra_deg = ra_deg % 360
    # Calculate declination in radians
    dec_rad = np.arctan2(z, np.sqrt(x**2 + y**2))
    # Convert declination to degrees
    dec_deg = np.degrees(dec_rad)
    return (ra_deg, dec_deg)

######################################################################
# Orbital Elements
######################################################################
def mean_longitude(longitude_of_ascending_node = True, argument_of_periapsis = True,  mean_anomaly = True):
    return longitude_of_ascending_node + argument_of_periapsis + mean_anomaly
    
def true_anomaly(eccentricity = True, eccentric_anomaly = True, mean_anomaly = True, true_longitude = True, argument_of_periapsis = True, longitude_of_ascending_node = True):
    if eccentricity is not True and eccentric_anomaly is not True:
        beta = eccentricity / (1+np.sqrt(1-eccentricity**2))
        true_anomaly = eccentric_anomaly + 2 * np.arctan2(beta * np.sin(eccentric_anomaly) / (1 - beta*np.cos(eccentric_anomaly)))
    elif eccentricity is not True and mean_anomaly is not True:
        true_anomaly = mean_anomaly + (2*eccentricity - 1/4 * eccentricity**3)*np.sin(mean_anomaly) + 5/4*eccentricity**2*np.sin(2*mean_anomaly) + 13/12*eccentricity**3*np.sin(3*mean_anomaly)
    elif true_longitude is not True and longitude_of_ascending_node is not True and argument_of_periapsis is not True:
        true_anomaly = true_longitude - longitude_of_ascending_node - argument_of_periapsis
    else:
        return print('Not enough information provided to calculate true anomaly.')
    return true_anomaly

######################################################################
# Get planet ephemeris using REBOUND
######################################################################
def rebound_orbital_elements(planet='earth', year=2000, month=1, day=1, hour=12, minute=0, second=0, jd=False):
    if jd == False:
        jd = astTime({'year':year, 'month':month, 'day':day, 'hour':hour, 'minute':minute, 'second':second}, format='ymdhms').jd
    elif jd == 'today' or jd == 'now':
        jd=Time.now().jd
    try:
        if isinstance(jd, float) or isinstance(jd, int):
            jd = str(jd)
        if jd[0:2] == 'JD':
            pass
        else:
            jd = f'JD{jd}'
    except IndexError as err:
        print('Error with the date provided. Give a year/month/day or JD.')
        return
    sim = rebound.Simulation()
    with suppress_stdout():
        sim.add("sun", date=jd, hash=0)
        sim.add("mercury", date=jd, hash=1)
        sim.add("venus", date=jd, hash=2)
        sim.add("earth", date=jd, hash=3)
        sim.add("mars", date=jd, hash=4)
        sim.add("jupiter", date=jd, hash=5)
        sim.add("saturn", date=jd, hash=6)
        sim.add("uranus", date=jd, hash=7)
        sim.add("neptune", date=jd, hash=8)
        sim.move_to_com()
    if planet.lower() == "all":
        orbitals = {}
        for i, planet_str in enumerate(['mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']):
            orbitals[planet_str] = {'a':sim.particles[h(i+1)].a, 'e':sim.particles[h(i+1)].e, 'i':sim.particles[h(i+1)].inc, 'true_longitude':sim.particles[h(i+1)].theta, 'argument_of_pericenter':sim.particles[h(i+1)].omega, 'longitude_of_ascending_node':sim.particles[h(i+1)].Omega, 'true_anomaly':sim.particles[h(i+1)].f, 'longitude_of_pericenter':sim.particles[h(i+1)].pomega, 'mean_longitude':sim.particles[h(i+1)].l, 'mean_anomaly':sim.particles[h(i+1)].M, 'eccentric_anomaly':rebound.M_to_E(sim.particles[h(i+1)].e, sim.particles[h(i+1)].M)}
        return orbitals
    elif planet.lower() == "sun":
        index = 0
    elif planet.lower() == "mercury":
        index = 1
    elif planet.lower() == "venus":
        index = 2
    elif planet.lower() == "earth":
        index = 3
    elif planet.lower() == "mars":
        index = 4
    elif planet.lower() == "jupiter":
        index = 5
    elif planet.lower() == "saturn":
        index = 6
    elif planet.lower() == "uranus":
        index = 7
    elif planet.lower() == "neptune":
        index = 8
    return {'a':sim.particles[h(index)].a, 'e':sim.particles[h(index)].e, 'i':sim.particles[h(index)].inc, 'true_longitude':sim.particles[h(index)].theta, 'argument_of_pericenter':sim.particles[h(index)].omega, 'longitude_of_ascending_node':sim.particles[h(index)].Omega, 'true_anomaly':sim.particles[h(index)].f, 'longitude_of_pericenter':sim.particles[h(index)].pomega, 'mean_longitude':sim.particles[h(index)].l, 'mean_anomaly':sim.particles[h(index)].M, 'eccentric_anomaly':rebound.M_to_E(sim.particles[h(index)].e, sim.particles[h(index)].M)}

######################################################################
# Calculate JPL orbital elements https://ssd.jpl.nasa.gov/planets/approx_pos.html
######################################################################
def j2000_orbitals(planet = 'earth', Teph = 2451545.0): #date input is in jd
    #ecliptic and equinox of J2000, valid for the time-interval 1800 AD - 2050 AD table 1-https://ssd.jpl.nasa.gov/planets/approx_pos.html
    if planet.lower() == 'mercury':
        anaut = 0.38709927; enaut = 0.20563593; inaut = 7.00497902; Lnaut = 252.25032350; onaut = 77.45779628; Onaut = 48.33076593
        arate = 0.00000037; erate = 0.00001906; irate = -0.00594749; Lrate = 149472.67411175; orate = 0.16047689; Orate = -0.12534081
    elif planet.lower() == 'venus':
        anaut = 0.72333566; enaut = 0.00677672; inaut = 3.39467605; Lnaut = 181.97909950; onaut = 131.60246718; Onaut = 76.67984255;
        arate = 0.00000390; erate = -0.00004107; irate = -0.00078890; Lrate = 58517.81538729; orate = 0.00268329; Orate = -0.27769418;
    elif planet.lower() == 'earth':
        anaut = 1.00000261; enaut = 0.01671123; inaut = -0.00001531; Lnaut = 100.46457166; onaut = 102.93768193; Onaut = 0.0;
        arate = 0.00000562; erate = -0.00004392; irate = -0.01294668; Lrate = 35999.37244981; orate = 0.32327364; Orate = 0.0;
    elif planet.lower() == 'mars':
        anaut = 1.52371034; enaut = 0.09339410; inaut = 1.84969142; Lnaut = -4.55343205; onaut = -23.94362959; Onaut = 49.55953891;
        arate = 0.00001847; erate = 0.00007882; irate = -0.00813131; Lrate = 19140.30268499; orate = 0.44441088; Orate = -0.29257343;
    elif planet.lower() == 'jupiter':
        anaut = 5.20288700; enaut = 0.04838624; inaut = 1.30439695; Lnaut = 34.39644051; onaut = 14.72847983; Onaut = 100.47390909;
        arate = -0.00011607; erate = -0.00013253; irate = -0.00183714; Lrate = 3034.74612775; orate = 0.21252668; Orate = 0.20469106;
    elif planet.lower() == 'saturn':
        anaut = 9.53667594; enaut = 0.05386179; inaut = 2.48599187; Lnaut = 49.95424423; onaut = 92.59887831;   Onaut = 113.66242448;
        arate = -0.00125060; erate = -0.00050991; irate = 0.00193609; Lrate = 1222.49362201; orate = -0.41897216; Orate = -0.28867794;
    elif planet.lower() == 'uranus':
        anaut = 19.18916464; enaut = 0.04725744; inaut = 0.77263783; enaut = 313.23810451; onaut = 170.95427630; Onaut = 74.01692503;
        arate = -0.00196176; erate = -0.00004397; irate = -0.00242939; Lrate = 428.48202785; orate = 0.40805281; Orate = 0.04240589;
    elif planet.lower() == 'neptune':
        anaut = 30.06992276; enaut = 0.00859048; inaut = 1.77004347; Lnaut = -55.12002969; onaut = 44.96476227; Onaut = 131.78422574;
        arate = 0.00026291; erate = 0.00005105; irate = 0.00035372; Lrate = 218.45945325; orate = -0.32241464; Orate = -0.00508664;
    
    #number of centuries after J2000
    T = (Teph - 2451545.0)/36525
    a = anaut + arate*T
    e = enaut + erate*T
    i = inaut + irate*T
    mean_longitude = Lnaut + Lrate*T
    longitude_of_perihelion = onaut + orate*T
    longitude_of_the_ascending_node = Onaut + Orate*T
    
    return {'a':a, 'e':e, 'i':deg90to90(i), 'mean_longitude':deg0to360(mean_longitude), 'longitude_of_perihelion':deg0to360(longitude_of_perihelion), 'longitude_of_the_ascending_node':deg0to360(longitude_of_the_ascending_node)}

######################################################################
# Calculate orbital elements from a given ephemeris x y z vx vy vz
######################################################################
def kepler_to_state(a, e, i, raan, w, nu, mu=mu.earth):
    """
    Converts keplerian orbital elements to a state vector.
    
    Parameters:
    - a: semi-major axis (in meters)
    - e: eccentricity
    - i: inclination (in radians)
    - raan: right ascension of the ascending node (in radians)
    - w: argument of perigee (in radians)
    - nu: true anomaly (in radians)
    - mu: gravitational parameter of the central body (in meters^3/s^2)
    
    Returns:
    - r: position vector in the inertial frame (in meters)
    - v: velocity vector in the inertial frame (in meters/s)
    
    Can pass a 2D array of shape (N, 6) where N is the number of sets of keplerian orbital elements you want to convert. 
    The function will return two arrays of shape (N, 3) with the corresponding position and velocity vectors.
    """
    
    # Compute the position vector in the perifocal frame
    r_pf = a * (1 - e**2) / (1 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0])
    
    # Compute the velocity vector in the perifocal frame
    v_pf = np.sqrt(mu/a) * np.array([-np.sin(nu), e + np.cos(nu), 0])
    
    # Create the rotation matrix from the perifocal to the inertial frame
    R_pf_i = np.array([[np.cos(raan) * np.cos(w) - np.sin(raan) * np.cos(i) * np.sin(w),
                        -np.cos(raan) * np.sin(w) - np.sin(raan) * np.cos(i) * np.cos(w),
                        np.sin(raan) * np.sin(i)],
                       [np.sin(raan) * np.cos(w) + np.cos(raan) * np.cos(i) * np.sin(w),
                        -np.sin(raan) * np.sin(w) + np.cos(raan) * np.cos(i) * np.cos(w),
                        -np.cos(raan) * np.sin(i)],
                       [np.sin(i) * np.sin(w), np.sin(i) * np.cos(w), np.cos(i)]])
    
    # Transform the position and velocity vectors to the inertial frame
    r = np.dot(R_pf_i, r_pf)
    v = np.dot(R_pf_i, v_pf)
    
    return r, v

def state_to_kepler(r, v, mu=mu.earth):
    """
    Converts a state vector to keplerian orbital elements.
    
    Parameters:
    - r: position vector in the inertial frame (in meters)
    - v: velocity vector in the inertial frame (in meters/s)
    - mu: gravitational parameter of the central body (in meters^3/s^2)
    
    Returns:
    - a: semi-major axis (in meters)
    - e: eccentricity
    - i: inclination (in radians)
    - raan: right ascension of the ascending node (in radians)
    - w: argument of perigee (in radians)
    - nu: true anomaly (in radians)
    
    Can pass two arrays of shape (N, 3) where N is the number of state vectors you want to convert. 
    The function will return an array of shape (N, 6) with the corresponding keplerian orbital elements.
    """
    
    # Compute the angular momentum vector
    h = np.cross(r, v)
    
    # Compute the eccentricity vector
    e_vec = (np.cross(v, h) / mu) - r / np.linalg.norm(r)
    
    # Compute the semi-major axis
    a = 1 / (2/np.linalg.norm(r) - np.linalg.norm(v)**2/mu)
    
    # Compute the eccentricity
    e = np.linalg.norm(e_vec)
    
    # Compute the inclination
    i = np.arccos(h[2] / np.linalg.norm(h))
    
    # Compute the right ascension of the ascending node
    if h[0] >= 0:
        raan = np.arccos(h[0] / np.linalg.norm(h[:2]))
    else:
        raan = 2 * np.pi - np.arccos(h[0] / np.linalg.norm(h[:2]))
    
    # Compute the argument of perigee
    if e_vec[2] >= 0:
        w = np.arccos(np.dot(h, e_vec) / (np.linalg.norm(h) * e))
    else:
        w = 2 * np.pi - np.arccos(np.dot(h, e_vec) / (np.linalg.norm(h) * e))
    
    # Compute the true anomaly
    if np.dot(r, v) >= 0:
        nu = np.arccos(np.dot(e_vec, r) / (e * np.linalg.norm(r)))
    else:
        nu = 2 * np.pi - np.arccos(np.dot(e_vec, r) / (e * np.linalg.norm(r)))
    
    return a, e, i, raan, w, nu

def kepler_to_parametric(a, e, i, omega, w, theta):
    # Convert to radians
    i = np.radians(i)
    omega = np.radians(omega)
    w = np.radians(w)
    theta = np.radians(theta)

    # Compute the semi-major and semi-minor axes
    b = a * np.sqrt(1 - e**2)
    # Compute the parametric coefficients
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    z = 0
    # Rotate the ellipse about the x-axis
    x_prime = x
    y_prime = y * np.cos(i) - z * np.sin(i)
    z_prime = y * np.sin(i) + z * np.cos(i)

    # Rotate the ellipse about the z-axis
    x_prime_prime = x_prime * np.cos(omega) - y_prime * np.sin(omega)
    y_prime_prime = x_prime * np.sin(omega) + y_prime * np.cos(omega)
    z_prime_prime = z_prime

    # Translate the ellipse
    x_final = x_prime_prime + w
    y_final = y_prime_prime
    z_final = z_prime_prime
    
    return x_final, y_final, z_final

def calculate_orbital_elements(r_, v_, mu_barycenter = mu.earth):
    # mu_barycenter - all bodies interior to Earth
    # 1.0013415732186798 #All bodies of solar system
    mu_ = mu_barycenter
    rarr = nby3shape(r_)
    varr = nby3shape(v_)
    aarr = []; earr = []; incarr = []; 
    true_longitudearr = []; argument_of_periapsisarr = []; longitude_of_ascending_nodearr = []; 
    true_anomalyarr = []; hmagarr = [];
    for r, v in zip(rarr, varr):
        r = np.array(r);# print(f'r: {r}')
        v = np.array(v);# print(f'v: {v}')
        
        rmag = np.sqrt(r.dot(r))
        vmag = np.sqrt(v.dot(v))

        h = np.cross(r,v)
        hmag = np.sqrt(h.dot(h))
        n = np.cross(np.array([0,0,1]),h)

        a = 1/((2/rmag)-(vmag**2)/mu_)

        evector = np.cross(v,h)/(mu_) - r/rmag;
        e = np.sqrt(evector.dot(evector))

        inc = np.arccos(h[2]/hmag); 

        if np.dot(r,v) > 0:
            true_anomaly = np.arccos(np.dot(evector,r)/(e*rmag))
        else:
            true_anomaly = 2*np.pi - np.arccos(np.dot(evector,r)/(e*rmag))
        if evector[2] >= 0:
            argument_of_periapsis = np.arccos(np.dot(n,evector)/(e*np.sqrt(n.dot(n))))
        else:
            argument_of_periapsis = 2*np.pi - np.arccos(np.dot(n,evector)/(e*np.sqrt(n.dot(n))))
        if n[1] >= 0:
            longitude_of_ascending_node = np.arccos(n[0]/np.sqrt(n.dot(n)))
        else:
            longitude_of_ascending_node = 2*np.pi - np.arccos(n[0]/np.sqrt(n.dot(n)))

        true_longitude = true_anomaly + argument_of_periapsis + longitude_of_ascending_node
        aarr.append(a); earr.append(e); incarr.append(inc); true_longitudearr.append(true_longitude); argument_of_periapsisarr.append(argument_of_periapsis); longitude_of_ascending_nodearr.append(longitude_of_ascending_node); true_anomalyarr.append(true_anomaly); hmagarr.append(hmag)
    return {'semi_major_axis':aarr, 'eccentricity':earr, 'inclination':incarr, 'true_longitude':true_longitudearr, 'argument_of_periapsis':argument_of_periapsisarr, 'longitude_of_ascending_node':longitude_of_ascending_nodearr, 'true_anomaly':true_anomalyarr, 'Ang. Mo.':hmagarr}
    
def periapsis(eccentricity, semi_major_axis):
    return (1 - eccentricity) * semi_major_axis
def apoapsis(eccentricity, semi_major_axis):
    return (1 + eccentricity) * semi_major_axis

def keplerian_a_e_from(perigee, apogee):
    # Semi-major axis
    a = (perigee + apogee) / 2
    # Eccentricity
    e = (apogee - perigee) / (apogee + perigee)
    return {"a": a, "e": e}

    
def close_to_any(a, floats, **kwargs):
    return np.any(np.isclose(a, floats, **kwargs))

######################################################################
# Math Functions
######################################################################
def npsumdot(x, y):
    return np.sum(x*y, axis=1)

def mag(vector, axis=-1):
    return np.linalg.norm(vector, axis=axis)

    
######################################################################
# Rotate set of points about a point
######################################################################
def rotate_via_numpy(x, y, radians):
    """Use numpy to build a rotation matrix and take the dot product."""
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, [x, y])
    return float(m.T[0]), float(m.T[1])

######################################################################
# Rotate set of points about the origin
######################################################################
def rotate_origin_only(x, y, radians):
    """Only rotate a point around the origin (0, 0)."""
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)
    return xx, yy

######################################################################
# Rotation Matrices
######################################################################
def Rx(theta):
    return np.matrix([[1, 0           , 0         ],
                   [ 0, np.cos(theta),-np.sin(theta)],
                   [ 0, np.sin(theta), np.cos(theta)]])
  
def Ry(theta):
    return np.matrix([[np.cos(theta), 0, np.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-np.sin(theta), 0, np.cos(theta)] ])
  
def Rz(theta):
    return np.matrix([[np.cos(theta), -np.sin(theta), 0],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

#Rotate 2d - theta is a counterclockwise rotation
def rotate_2d(x, y, theta_to_rotate, x_origin=0, y_origin=0):
    theta = np.arctan2(y - x_origin, x - y_origin)
    distance = np.sqrt(np.power((x - x_origin),2) + np.power((y - y_origin),2))
    xrot = distance * np.cos(np.pi + (theta - theta_to_rotate))
    yrot = distance * np.sin(np.pi + (theta - theta_to_rotate))
    return xrot, yrot

#Using clockwise direction
def rotate_3d(vector, xtheta, ytheta, ztheta):
    vector = np.array(vector).flatten()
    return np.dot(vector, np.dot(Rz(ztheta), np.dot(Ry(ytheta), Rx(xtheta))))

def unit_vector(vector):
    """ Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)

#Counter-clockwise direction
def rotate_axis(vector, axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    return np.dot(rotation_matrix, vector)

def create_sphere(cx,cy,cz, r, resolution=360):
    '''
    create sphere with center (cx, cy, cz) and radius r
    '''
    phi = np.linspace(0, 2*np.pi, 2*resolution)
    theta = np.linspace(0, np.pi, resolution)

    theta, phi = np.meshgrid(theta, phi)

    r_xy = r*np.sin(theta)
    x = cx + np.cos(phi) * r_xy
    y = cy + np.sin(phi) * r_xy
    z = cz + r * np.cos(theta)

    return np.stack([x,y,z]) 

def drawSphere(xCenter, yCenter, zCenter, r, res=10j,flatten=True):
    if not 'j' in str(res):
        res = complex(0,res)
    #draw sphere
    u, v = np.mgrid[0:2*np.pi:2*res, 0:np.pi:res]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)
    # shift and scale sphere
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    if flatten:
        x = np.squeeze(np.array(x).flatten())
        y = np.squeeze(np.array(y).flatten())
        z = np.squeeze(np.array(z).flatten())
    return (x,y,z)

from contextlib import contextmanager
import os, sys
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

######################################################################
# Calculate proper motion on sky of an object.
######################################################################
def ra_dec(r=None, v=None, x=None, y=None, z=None, vx=None, vy=None, vz=None, r_earth=np.array([0, 0, 0]), v_earth=np.array([0, 0, 0]), input_unit='si'):
    if r is None or v is None:
        if x is not None and y is not None and z is not None and vx is not None and vy is not None and vz is not None:
            r = np.array([x, y, z])
            v = np.array([vx, vy, vz])
        else:
            raise ValueError("Either provide r and v arrays or individual coordinates (x, y, z) and velocities (vx, vy, vz)")

    # Subtract Earth's position and velocity from the input arrays
    r = r - r_earth
    v = v - v_earth

    d_earth_mag = einsum_norm(r, 'ij,ij->i')
    ra = rad0to2pi(np.arctan2(r[:,1], r[:,0]))# in radians
    dec = np.arcsin(r[:,2] / d_earth_mag)
    return ra, dec
  
######################################################################
# Calculate proper motion on sky of an object.
######################################################################
def proper_motion(x, y, z, vx, vy, vz, xe=0, ye=0, ze=0, vxe=0, vye=0, vze=0, input_unit='si'):
    #Units: distance (au), time (2*pi years), angles arcseconds
    #Find Position and Velocity Relative to Earth
    x_rot = x - xe
    y_rot = y - ye
    z_rot = z - ze
    vx_rot = vx - vxe
    vy_rot = vy - vye
    vz_rot = vz - vze
    
    d_earth_mag = np.linalg.norm([x_rot, y_rot, z_rot])
    if d_earth_mag == 0:
        pm = np.nan
        return pm
    v_ast_earth = np.array([vx_rot , vy_rot, vz_rot]);
    los_vector = np.array([x_rot, y_rot, z_rot])
    
    v_los = np.linalg.norm((np.dot(v_ast_earth, los_vector)/np.linalg.norm(los_vector)))
    v_transverse = np.sqrt(np.linalg.norm(v_ast_earth)**2 - v_los**2);
    if input_unit == 'si':
        pm = v_transverse / d_earth_mag * units.rad_to_arcsecond
        return pm
    elif input_unit == 'rebound':
        pm = v_transverse / d_earth_mag * units.rad_to_arcsecond / (units.year_to_second * 2 * np.pi)  # v_transverse/d_earth_mag is in (au/sim_time)/au (2*pi*year), convert to arcseconds / second
        return pm
    else:
        print('Error - units provided not available, provide either SI or rebound units.')
        return 
#Coordinates need to be in ecliptic by default, or can be in equitorial
def proper_motion_ra_dec(r=None, v=None, x=None, y=None, z=None, vx=None, vy=None, vz=None, r_earth=np.array([0, 0, 0]), v_earth=np.array([0, 0, 0]), input_unit='si'):
    if r is None or v is None:
        if x is not None and y is not None and z is not None and vx is not None and vy is not None and vz is not None:
            r = np.array([x, y, z])
            v = np.array([vx, vy, vz])
        else:
            raise ValueError("Either provide r and v arrays or individual coordinates (x, y, z) and velocities (vx, vy, vz)")

    # Subtract Earth's position and velocity from the input arrays
    r = r - r_earth
    v = v - v_earth
        
    #Distances to Earth and Sun
    d_earth_mag = einsum_norm(r, 'ij,ij->i')

    # RA / DEC calculation
    ra = rad0to2pi(np.arctan2(r[:,1], r[:,0]))# in radians
    dec = np.arcsin(r[:,2] / d_earth_mag)
    ra_unit_vector = np.array([-np.sin(ra), np.cos(ra), np.zeros(np.shape(ra))]).T
    dec_unit_vector = -np.array([np.cos(np.pi/2-dec)*np.cos(ra), np.cos(np.pi/2-dec)*np.sin(ra), -np.sin(np.pi/2-dec)]).T
    pmra = (np.einsum('ij,ij->i',v, ra_unit_vector)) / d_earth_mag * units.rad_to_arcsecond # arcseconds / second
    pmdec = (np.einsum('ij,ij->i',v, dec_unit_vector)) / d_earth_mag * units.rad_to_arcsecond # arcseconds / second
    
    if input_unit == 'si':
        return pmra, pmdec
    elif input_unit == 'rebound':
        pmra = pmra / (units.year_to_second * 2 * np.pi)
        pmdec = pmdec / (units.year_to_second * 2 * np.pi) # arcseconds * (au/sim_time)/au, convert to arcseconds / second
        return pmra, pmdec
    else:
        print('Error - units provided not available, provide either SI or rebound units.')
        return


def vcirc(r=units.au_to_m, mu_=mu.sun+mu.mercury+mu.venus):
    return np.sqrt(mu_/r)
def delv(x,y):
    return mag(np.array(x)-np.array(y))
    
######################################################################
# Calculate spherical coordinates from cartesian.
######################################################################
def cart2sph_deg(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy) * (180/np.pi)
    az = (np.arctan2(y, x)) * (180/np.pi)
    return az, el, r
def cart_to_cyl(x, y, z):
    r = mag([x,y])
    theta = np.arctan2(y, x)
    return r, theta, z-z0

######################################################################
# Calculate coordinates in Earths rotating frame
######################################################################
def inert2rot(x, y, xe, ye, xs=0, ys=0):#Places Earth at (-1,0)
    earth_theta = np.arctan2(ye - ys, xe - xs)
    theta = np.arctan2(y - ys, x - xs)
    distance = np.sqrt(np.power((x - xs),2) + np.power((y - ys),2))
    xrot = distance * np.cos(np.pi + (theta - earth_theta))
    yrot = distance * np.sin(np.pi + (theta - earth_theta))
    return xrot, yrot

######################################################################
# Calculate Earth centered coordinates
######################################################################
def solar2geo(x, y, xe, ye):
    xshft = x - xe
    yshft = y - ye
    return xshft, yshft

######################################################################
# Calculate solar centered coordinates
######################################################################
def sim2solar(x, y, xs, ys):
    xshft = x - xs
    yshft = y - ys
    return xsolar, ysolar

######################################################################
# Calculate Distance to Earth
######################################################################
def distance3d(x, y, z, xe, ye, ze):
    distance = ( (x - xe)**2+(y - ye)**2+(z - ze)**2 )**(1/2)
    return distance

######################################################################
# convert a radian angle to between 0 to 2*pi
######################################################################
def rad0to2pi(angles):
    return (2*np.pi + angles) * (angles < 0) + angles*(angles > 0)
######################################################################
# convert a degree angle to between 0 to 360
######################################################################
def deg0to360(array_):
    try:
        return [i % 360 for i in array_]
    except TypeError:
        return array_ % 360

def deg0to360array(array_):
    return [i % 360 for i in array_]

def deg90to90(val_in):
    if hasattr(val_in, "__len__"):
        val_out = []
        for i, v in enumerate(val_in):
            while v < -90:
                v += 90
            while v > 90:
                v -= 90
            val_out.append(v)
    else:
        while val_in < -90:
            val_in += 90
        while val_in > 90:
            val_in -= 90
        val_out = val_in
    return val_out

def deg90to90array(array_):
    return [i % 90 for i in array_]

######################################################################
# Calculate Latitude / Longitude - in rotating Earth coordinate system
######################################################################
def sim_lonlatrad(x,y,z,xe,ye,ze,xs,ys,zs):
    #convert all to geo coordinates
    x = x - xe
    y = y - ye
    z = z - ze
    xs = xs - xe
    ys = ys - ye
    zs = zs - ze
    #convert x y z to lon lat radius
    longitude, latitude, radius = cart2sph_deg(x, y, z)
    slongitude, slatitude, sradius = cart2sph_deg(xs, ys, zs)
    #correct so that Sun is at (0,0)
    longitude = deg0to360(slongitude - longitude)
    latitude = latitude - slatitude
    
    return longitude, latitude, radius

######################################################################
# get Right Ascension and Declination of the Sun for given date
######################################################################
def sun_ra_dec(time_):
    out = get_sun(astTime(time_,format='mjd'))
    return out.ra.to('rad').value, out.dec.to('rad').value

#Distance between two points on Earth
def lonlat_distance(lat1, lat2, lon1, lon2):
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    # Radius of earth in kilometers. Use 3956 for miles
    # calculate the result
    return(c * earth_rad)
######################################################################################
## coordinate conversion helper functions
######################################################################################
def altitude2zenithangle(altitude, deg=True):
    if deg:
        out = 90 - altitude
    else:
        out = np.pi / 2 - altitude
    return out

def zenithangle2altitude(zenith_angle, deg=True):
    if deg:
        out = 90 - zenith_angle
    else:
        out = np.pi / 2 - zenith_angle
    return out

def rightasension2hourangle(right_ascension, local_time):
    if type(right_ascension) != str:
        right_ascension = dd_to_hms(right_ascension)
    if type(local_time) != str:
        local_time = dd_to_dms(local_time)
    _ra = float(right_ascension.split(':')[0])
    _lt = float(local_time.split(':')[0])
    if _ra > _lt:
        __ltm, __lts = local_time.split(':')[1:]
        local_time = f'{24 + _lt}:{__ltm}:{__lts}'

    return dd_to_dms(hms_to_dd(local_time) - hms_to_dd(right_ascension))

def dms_to_dd(dms):#Degree minute second to Degree decimal
    dms, out = [[dms] if type(dms) == str else dms][0], []
    for i in dms:
        deg, minute, sec = [float(j) for j in i.split(':')]
        if deg < 0:
            minute, sec = float(f'-{minute}'), float(f'-{sec}')
        out.append(deg + minute / 60 + sec / 3600)
    return [out[0] if type(dms) == str or len(dms) == 1 else out][0]

def dd_to_dms(degree_decimal):
    _d, __d = np.trunc(degree_decimal), degree_decimal - np.trunc(degree_decimal)
    __d = [-__d if degree_decimal < 0 else __d][0]
    _m, __m = np.trunc(__d * 60), __d * 60 - np.trunc(__d * 60)
    _s = round(__m * 60, 4)
    _s = [int(_s) if int(_s) == _s else _s][0]
    if _s == 60:
        _m, _s = _m + 1, '00'
    elif _s > 60:
        _m, _s = _m + 1, _s - 60

    return f'{int(_d)}:{int(_m)}:{_s}'

def hms_to_dd(hms):
    _type = type(hms)
    hms, out = [[hms] if _type == str else hms][0], []
    for i in hms:
        if i[0] != '-':
            hour, minute, sec = i.split(':')
            hour, minute, sec = float(hour), float(minute), float(sec)
            out.append(hour * 15 + (minute / 4) + (sec / 240))
        else:
            print('hms cannot be negative.')

    return [out[0] if _type == str or len(hms) == 1 else out][0]

def dd_to_hms(degree_decimal):
    if type(degree_decimal) == str:
        degree_decimal = dms_to_dd(degree_decimal)
    if degree_decimal < 0:
        print('dd for HMS conversion cannot be negative, assuming positive.')
        _dd = -degree_decimal / 15
    else:
        _dd = degree_decimal / 15
    _h, __h = np.trunc(_dd), _dd - np.trunc(_dd)
    _m, __m = np.trunc(__h * 60), __h * 60 - np.trunc(__h * 60)
    _s = round(__m * 60, 4)
    _s = [int(_s) if int(_s) == _s else _s][0]
    if _s == 60:
        _m, _s = _m + 1, '00'
    elif _s > 60:
        _m, _s = _m + 1, _s - 60

    return f'{int(_h)}:{int(_m)}:{_s}'
######################################################################################
## Convert equatorial to horizontal coordinates
######################################################################################    
def equatorial_to_horizontal(observer_latitude, declination, right_ascension=None, hour_angle=None, local_time=None, hms=False):
    if right_ascension is not None:
        hour_angle = rightasension2hourangle(right_ascension, local_time)
        if hms:
            hour_angle = hms_to_dd(hour_angle)
    elif hour_angle is not None:
        if hms:
            hour_angle = hms_to_dd(hour_angle)
    elif right_ascension is not None and hour_angle is not None:
        print('Both right_ascension and hour_angle parameters are provided.\nUsing hour_angle for calculations.')
        if hms:
            hour_angle = hms_to_dd(hour_angle)
    else:
        print('Either right_ascension or hour_angle must be provided.')

    observer_latitude, hour_angle, declination = np.radians([observer_latitude, hour_angle, declination])

    zenith_angle = np.arccos(np.sin(observer_latitude) * np.sin(declination) + np.cos(observer_latitude) * np.cos(declination) * np.cos(hour_angle))

    altitude = zenithangle2altitude(zenith_angle, deg=False)

    _num = np.sin(declination) - np.sin(observer_latitude) * np.cos(zenith_angle)
    _den = np.cos(observer_latitude) * np.sin(zenith_angle)
    azimuth = np.arccos(_num / _den)

    if latitude < 0:
        azimuth = np.pi - azimuth
    altitude, azimuth = np.degrees([altitude, azimuth])

    return azimuth, altitude
######################################################################################
## Convert horizontal to equatorial coordinates
######################################################################################
def horizontal_to_equatorial(observer_latitude, azimuth, altitude):
    altitude, azimuth, latitude = np.radians([altitude, azimuth, observer_latitude])
    zenith_angle = zenithangle2altitude(altitude)

    zenith_angle = [-zenith_angle if latitude < 0 else zenith_angle][0]

    declination = np.sin(latitude) * np.cos(zenith_angle)
    declination = declination + (np.cos(latitude) * np.sin(zenith_angle) * np.cos(azimuth))
    declination = np.arcsin(declination)

    _num = np.cos(zenith_angle) - np.sin(latitude) * np.sin(declination)
    _den = np.cos(latitude) * np.cos(declination)
    hour_angle = np.arccos(_num / _den)

    if (latitude > 0 > declination) or (latitude < 0 < declination):
        hour_angle = 2 * np.pi - hour_angle

    declination, hour_angle = np.degrees([declination, hour_angle])

    return hour_angle, declination
######################################################################################
## Convert equatorial to ecliptic coordinates
######################################################################################
_ecliptic = 0.409092601 #np.radians(23.43927944)
cos_ec = 0.9174821430960974
sin_ec = 0.3977769690414367
######################################################################################
## ecliptic to equitorial cartesian coordinates
######################################################################################  
def equatorial_xyz_to_ecliptic_xyz(xq, yq, zq):
    xc = xq
    yc = cos_ec*yq + sin_ec*zq
    zc = -sin_ec*yq + cos_ec*zq
    return xc, yc, zc

######################################################################################
## equitorial to ecliptic cartesian coordinates
######################################################################################  
def ecliptic_xyz_to_equatorial_xyz(xc, yc, zc):
    xq = xc
    yq = cos_ec*yc - sin_ec*zc
    zq = sin_ec*yc + cos_ec*zc
    return xq, yq, zq

######################################################################################
## ecliptic cartesian to ecliptic angles
######################################################################################
def xyz_to_ecliptic(xc,yc,zc, xs=0,ys=0,zs=0, degrees=False):
    x_ast_to_earth = xc - xe
    y_ast_to_earth = yc - ye
    z_ast_to_earth = zc - ze
    d_earth_mag = np.sqrt(np.power(x_ast_to_earth,2) + np.power(y_ast_to_earth,2) + np.power(z_ast_to_earth,2)) 
    ec_longitude = rad0to2pi(np.arctan2(y_ast_to_earth, x_ast_to_earth))# in radians
    ec_latitude = np.arcsin(z_ast_to_earth / d_earth_mag)
    if degrees:
        return np.degrees(ec_longitude), np.degrees(ec_latitude)
    else:
        return ec_longitude, ec_latitude
    
######################################################################################
## equatorial xyz to equatorial angles
######################################################################################
def xyz_to_equatorial(xq,yq,zq, xe=0,ye=0,ze=0, degrees=False):
    # RA / DEC calculation - assumes XY plane to be celestial equator, and -x axis to be vernal equinox
    x_ast_to_earth = xq - xe
    y_ast_to_earth = yq - ye
    z_ast_to_earth = zq - ze
    d_earth_mag = np.sqrt(np.power(x_ast_to_earth,2) + np.power(y_ast_to_earth,2) + np.power(z_ast_to_earth,2)) 
    ra = rad0to2pi(np.arctan2(y_ast_to_earth, x_ast_to_earth))# in radians
    dec = np.arcsin(z_ast_to_earth / d_earth_mag)
    if degrees:
        return np.degrees(ra), np.degrees(dec)
    else:
        return ra, dec
    
######################################################################################
## equatorial xyz to equatorial angles
######################################################################################
def ecliptic_xyz_to_equatorial(xc,yc,zc, xe=0,ye=0,ze=0, degrees=False):
    #Convert ecliptic cartesian into equitorial cartesian
    x_ast_to_earth, y_ast_to_earth, z_ast_to_earth = ecliptic_xyz_to_equatorial_xyz(xc - xe, yc - ye, zc - ze)
    d_earth_mag = np.sqrt(np.power(x_ast_to_earth,2) + np.power(y_ast_to_earth,2) + np.power(z_ast_to_earth,2)) 
    ra = rad0to2pi(np.arctan2(y_ast_to_earth, x_ast_to_earth))# in radians
    dec = np.arcsin(z_ast_to_earth / d_earth_mag)
    if degrees:
        return np.degrees(ra), np.degrees(dec)
    else:
        return ra, dec
    
def equatorial_to_ecliptic(right_ascension, declination, degrees=False):
    ra, dec = np.radians(right_ascension), np.radians(declination)
    ec_latitude = np.arcsin(cos_ec * np.sin(dec) - sin_ec * np.cos(dec) * np.sin(ra))
    ec_longitude = np.arctan((cos_ec * np.cos(dec) * np.sin(ra) + sin_ec*np.sin(dec))/(np.cos(dec) * np.cos(ra)))
    if degrees:
        return deg0to360(np.degrees(ec_longitude)), np.degrees(ec_latitude)
    else:
        return rad0to2pi(ec_longitude), ec_latitude
        
def ecliptic_to_equatorial(lon, lat, degrees=False):
    lon, lat = np.radians(lon), np.radians(lat)
    ra = np.arctan( (cos_ec * np.cos(lat) * np.sin(lon) - sin_ec * np.sin(lat)) / (np.cos(lat) * np.cos(lon)))
    dec = np.arcsin(cos_ec * np.sin(lat) + sin_ec * np.cos(lat) * np.sin(lon))
    if degrees:
        return np.degrees(ra), np.degrees(dec)
    else:
        return ra, dec

######################################################################################
## Lambertian brightness functions
######################################################################################    
def getAngle(a, b, c):#a,b,c where b is the vertex
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    c = np.atleast_2d(c)
    ba = np.subtract(a,b)
    bc = np.subtract(c,b)
    cosine_angle = np.sum(ba*bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc,axis=-1))
    return np.arccos(cosine_angle)

def moon_shine(r_moon, r_sat, r_earth, r_sun, radius, albedo, albedo_moon, albedo_back, albedo_front, area_panels):#In SI units, takes single values or arrays returns a fractional flux
    #https://amostech.com/TechnicalPapers/2013/POSTER/COGNION.pdf
    moon_phase_angle = getAngle(r_sun, r_moon, r_sat)#Phase of the moon as viewed from the sat.
    sun_angle = getAngle(r_sun, r_sat, r_moon)#angle from Sun to object to Earth
    moon_to_earth_angle = getAngle(r_moon, r_sat, r_earth)
    r_moon_sat = np.linalg.norm(r_sat-r_moon, axis=-1)
    r_earth_sat = np.linalg.norm(r_sat-r_earth, axis=-1)#Earth is the observer.
    flux_moon_to_sat = 2 / 3 * albedo_moon * moon_rad**2 / (np.pi * (r_moon_sat)**2) * (np.sin(moon_phase_angle) + (np.pi - moon_phase_angle) * np.cos(moon_phase_angle))#Fraction of sunlight reflected from the Moon to satellite
    #Fraction of light from back of solar panel
    flux_back = np.zeros_like(sun_angle); 
    flux_back[sun_angle > np.pi/2] = np.abs(albedo_back * area_panels / (np.pi * r_earth_sat[sun_angle > np.pi/2]**2) * np.cos(np.pi - moon_to_earth_angle[sun_angle > np.pi/2]) * flux_moon_to_sat[sun_angle > np.pi/2]) #Fraction of Moon light reflected off back of solar panels - which are assumed to be always facing the Sun. Angle: Sun - Observer - Sat
    flux_front = np.zeros_like(sun_angle); 
    flux_front[sun_angle < np.pi / 2] = np.abs(albedo_front * area_panels / (np.pi * r_earth_sat[sun_angle < np.pi/2]**2) * np.cos(moon_to_earth_angle[sun_angle < np.pi/2]) * flux_moon_to_sat[sun_angle < np.pi/2])# Fraction of Sun light scattered off front of the solar panels - which are assumed to be always facing the Sun. Angle: Sun - Sat - Observer
    flux_panels = flux_back + flux_front
    flux_bus = 2 / 3 * albedo * radius**2 / (np.pi * r_earth_sat**2) * flux_moon_to_sat
    return {'moon_bus':flux_bus, 'moon_panels':flux_panels}

def earth_shine(r_sat, r_earth, r_sun, radius, albedo, albedo_earth, albedo_back, area_panels):#In SI units, takes single values or arrays returns a flux
    #https://amostech.com/TechnicalPapers/2013/POSTER/COGNION.pdf
    phase_angle = getAngle(r_sun, r_sat, r_earth)#angle from Sun to object to Earth
    earth_angle = np.pi - phase_angle# Sun to Earth to oject.
    r_earth_sat = np.linalg.norm(r_sat-r_earth, axis=-1)#Earth is the observer.
    flux_earth_to_sat = 2 / 3 * albedo_earth * earth_rad**2 / (np.pi * (r_earth_sat)**2) * (np.sin(earth_angle) + (np.pi - earth_angle) * np.cos(earth_angle))#Fraction of sunlight reflected from the Earth to satellite
    #Fraction of light from back of solar panel
    flux_back = np.zeros_like(phase_angle)
    flux_back[phase_angle > np.pi/2] = albedo_back * area_panels / (np.pi * r_earth_sat[phase_angle > np.pi/2]**2) * np.cos(np.pi - phase_angle[phase_angle > np.pi/2]) * flux_earth_to_sat[phase_angle > np.pi/2] #Fraction of Earth light reflected off back of solar panels - which are assumed to be always facing the Sun. Angle: Sun - Observer - Sat
    flux_bus = 2 / 3 * albedo * radius**2 / (np.pi * r_earth_sat**2) * flux_earth_to_sat
    return {'earth_bus':flux_bus, 'earth_panels':flux_back}

def sun_shine(r_sat, r_earth, r_sun, radius, albedo, albedo_front, area_panels):#In SI units, takes single values or arrays returns a fractional flux
    #https://amostech.com/TechnicalPapers/2013/POSTER/COGNION.pdf
    phase_angle = getAngle(r_sun, r_sat, r_earth)#angle from Sun to object to Earth
    r_earth_sat = np.linalg.norm(r_sat-r_earth, axis=-1)#Earth is the observer.
    flux_front = np.zeros_like(phase_angle)
    flux_front[phase_angle < np.pi / 2] = albedo_front * area_panels / (np.pi * r_earth_sat[phase_angle < np.pi/2]**2) * np.cos(phase_angle[phase_angle < np.pi/2])# Fraction of Sun light scattered off front of the solar panels - which are assumed to be always facing the Sun. Angle: Sun - Sat - Observer
    flux_bus = 2 / 3 * albedo * radius**2 / (np.pi * (r_earth_sat)**2) * (np.sin(phase_angle) + (np.pi - phase_angle) * np.cos(phase_angle)) #Fraction of light reflected off satellite from Sun
    return {'sun_bus':flux_bus, 'sun_panels':flux_front}

def M_v(r_sat, r_earth, r_sun, r_moon=False, radius=0.4, albedo=0.20, sun_Mag=4.80, albedo_earth = 0.30, albedo_moon = 0.12, albedo_back = 0.50, albedo_front = 0.05, area_panels = 100, return_components=False):
    r_sun_sat = np.linalg.norm(r_sat-r_sun, axis=-1)
    frac_flux_sun = {'sun_bus':0, 'sun_panels':0}; frac_flux_earth = {'earth_bus':0, 'earth_panels':0}; frac_flux_moon = {'moon_bus':0, 'moon_panels':0}
    frac_flux_sun = sun_shine(r_sat, r_earth, r_sun, radius, albedo, albedo_front, area_panels)
    frac_flux_earth = earth_shine(r_sat, r_earth, r_sun, radius, albedo, albedo_earth, albedo_back, area_panels)
    if r_moon is not False:
        frac_flux_moon = moon_shine(r_moon, r_sat, r_earth, r_sun, radius, albedo, albedo_moon, albedo_back, albedo_front, area_panels)
    merged_dict = {**frac_flux_sun, **frac_flux_earth, **frac_flux_moon}
    total_frac_flux = sum(merged_dict.values())
    Mag_v = (2.5 * np.log10((r_sun_sat / (10*units.pc_to_m))**2) + sun_Mag) - 2.5 * np.log10(total_frac_flux)
    if return_components:
        return Mag_v, merged_dict
    else:
        return Mag_v

######################################################################
# ast radius from albedo and H mag
######################################################################
def radius_from_H_albedo(H, albedo = .1):
    radius = 1329e3/(2*np.sqrt(albedo))*10**(-0.2*H)# http://www.physics.sfasu.edu/astro/asteroids/sizemagnitude.html
    return radius
def H_mag(radius, albedo):
    return 5 * np.log(664500 / (radius * np.sqrt(albedo)) ) / np.log(10)
#Color correction, Johnson V to LSST Filters - 
def johnsonV_to_lsst_array(M_app, filters, ast_types):
    corrections = np.zeros(np.shape(M_app))
    corrections[np.where((np.array(filters)=='u') & (np.array(ast_types)==0))] = -1.614
    corrections[np.where((np.array(filters)=='u') & (np.array(ast_types)==1))] = -1.927
    corrections[np.where((np.array(filters)=='g') & (np.array(ast_types)==0))] = -0.302
    corrections[np.where((np.array(filters)=='g') & (np.array(ast_types)==1))] = -0.395
    corrections[np.where((np.array(filters)=='r') & (np.array(ast_types)==0))] = 0.172
    corrections[np.where((np.array(filters)=='r') & (np.array(ast_types)==1))] = 0.255
    corrections[np.where((np.array(filters)=='i') & (np.array(ast_types)==0))] = 0.291
    corrections[np.where((np.array(filters)=='i') & (np.array(ast_types)==1))] = 0.455
    corrections[np.where((np.array(filters)=='z') & (np.array(ast_types)==0))] = 0.298
    corrections[np.where((np.array(filters)=='z') & (np.array(ast_types)==1))] = 0.401
    corrections[np.where((np.array(filters)=='y') & (np.array(ast_types)==0))] = 0.303
    corrections[np.where((np.array(filters)=='y') & (np.array(ast_types)==1))] = 0.406
    return M_app - corrections
def johnsonV_to_ztf_array(M_app, filters, ast_types):
    corrections = np.zeros(np.shape(M_app))
    corrections[np.where((np.array(filters)==1) & (np.array(ast_types)==0))] = -0.302
    corrections[np.where((np.array(filters)==1) & (np.array(ast_types)==1))] = -0.395
    corrections[np.where((np.array(filters)==2) & (np.array(ast_types)==0))] = 0.172
    corrections[np.where((np.array(filters)==2) & (np.array(ast_types)==1))] = 0.255
    corrections[np.where((np.array(filters)==3) & (np.array(ast_types)==0))] = 0.291
    corrections[np.where((np.array(filters)==3) & (np.array(ast_types)==1))] = 0.455
    return M_app - corrections

######################################################################
# get ETA albedo -- > P2R = fd * (pv*np.exp(-pv**2/(2*d**2))/d**2) + (1-fd)*(pv*np.exp(-pv**2/(2*b**2))/b**2)
######################################################################
def get_albedo_array(num=1):
    num = int(num);
    albedo_out = []; ast_type_out = []
    while np.size(albedo_out) != num:
        fd = 0.253; d = 0.030; b = 0.168
        albedo = np.random.uniform(0, 1, size=num)
        #Albedos from NEO population - https://iopscience.iop.org/article/10.3847/0004-6256/152/4/79
        sample_ys = np.random.uniform(0, 6, size=num);
        c_type = fd * (albedo*np.exp(-albedo**2/(2*d**2))/d**2)
        s_type = (1-fd)*(albedo*np.exp(-albedo**2/(2*b**2))/b**2)
        c_albedo = albedo[np.where(sample_ys < c_type)]; s_albedo = albedo[np.where(sample_ys < s_type)]
        c_type = np.zeros(np.size(c_albedo)); s_type = np.ones(np.size(s_albedo))
        albedo = np.concatenate((c_albedo, s_albedo))
        ast_type = np.concatenate((c_type, s_type))
        if np.size(albedo_out) == 0:
            albedo_out = albedo; ast_type_out = ast_type
            continue
        albedo_out = np.hstack((albedo_out,albedo)); ast_type_out = np.hstack((ast_type_out,ast_type))
        if np.size(albedo_out) > num:
            #Shuffle the last arrays so there's no biasing towards c_type
            temp = list(zip(albedo, ast_type))
            np.random.shuffle(temp)
            albedo, ast_type = zip(*temp)
            albedo_out = albedo_out[:num]; ast_type_out = ast_type_out[:num]
    return (albedo_out, ast_type_out)    
def granvik_low_slope(x):
    return 0.3034*x - 3.491
def granvik_high_slope(x):
    return 0.7235*x - 13.12
def get_neo_H_mag_array(num=1, upper_mag=28, min_mag=10):
    num = int(num)
    H_mag_out = []
    #Extending granvik H mags
    break_point = 23
    while np.size(H_mag_out) != num:
        xs = np.random.uniform(min_mag, upper_mag, size=num)
        ys = np.random.uniform(1, 10**granvik_high_slope(upper_mag), size=num)
        xs_low = xs[np.where(xs<23)]; ys_low = ys[np.where(xs<23)]
        xs_high = xs[np.where(xs>=23)]; ys_high = ys[np.where(xs>=23)]
        index_low = np.where(ys_low < 10**granvik_low_slope(xs_low))
        index_high = np.where(ys_high < 10**granvik_high_slope(xs_high))
        H_mag = np.hstack((xs_low[index_low],xs_high[index_high]))
        if np.size(H_mag_out) == 0:
            H_mag_out = H_mag
            continue
        H_mag_out = np.hstack((H_mag_out,H_mag))
        if np.size(H_mag_out) > num:
            H_mag_out = H_mag_out[:num]
    random.shuffle(H_mag_out)
    return H_mag_out
######################################################################
# get ETA diameter
######################################################################
def get_eta_radius_albedo_H_array(num=1, upper_mag = 28, min_mag=10):
    albedo, ast_type = get_albedo_array(num=num)
    H = get_neo_H_mag_array(num=num, upper_mag=upper_mag, min_mag=min_mag)
    radius = 1329e3/(2*np.sqrt(albedo))*10**(-0.2*H)
    return {'radius':radius, 'albedo':albedo, 'type':ast_type, 'H':H}

######################################################################
# MPI
######################################################################
def mpi_scatter(scatter_array):
    from mpi4py import MPI
    import sys
    import numpy as np
    
    comm = MPI.COMM_WORLD  # Defines the default communicator
    num_procs = comm.Get_size()  # Stores the number of processes in size.
    rank = comm.Get_rank()  # Stores the rank (pid) of the current process
    stat = MPI.Status()
    print(f'sys.argv: {sys.argv}, Number of procs: {num_procs}, rank: {rank}')
    # this is for chopping up the work load to all the processors. If you have, say, 40 tasks but 36 ranks, this will help figure out what to do with the overhang.
    # might be better to have the same number of tasks as processors, or an even multiple for even distributing.
    remainder = np.size(scatter_array) % num_procs
    # Each processor should at least do this many tasks
    base_load = np.size(scatter_array) // num_procs
    if rank == 0:
        print('All processors will process at least {0} simulations.'.format(
            base_load))
        print('{0} processors will process an additional simulations'.format(
            remainder))
    # Create a list stating how many files each processor should analyze.
    load_list = np.concatenate((np.ones(remainder) * (base_load + 1),
                                np.ones(num_procs - remainder) * base_load))
    if rank == 0:
        print('load_list={0}'.format(load_list))
    # Setup the receive arrays for the list on each processor
    if rank < remainder:
        # so this is creating an empty task list for all the processors that get the overhang.
        scatter_array_local = np.zeros(base_load + 1, dtype=np.int64)
    else:
        # and this is creating an empty task list for all the processors that do not get the overhang.
        scatter_array_local = np.zeros(base_load, dtype=np.int64)
    # Define the displacements array for the ScatterV function.
    disp = np.zeros(num_procs)
    for i in range(np.size(load_list)):
        if i == 0:
            disp[i] = 0
        else:
            disp[i] = disp[i - 1] + load_list[i - 1]
    # Scatter the list chunks to the processors. This fills those local empty tasks lists.
    # Need to use scatterv since dealing with uneven chunks
    # can use scatterv for even chunks too if that's the use case
    comm.Scatterv([scatter_array, load_list, disp, MPI.DOUBLE], scatter_array_local)
    print(f"Process {rank} received the scattered arrays: {scatter_array_local}")
    return scatter_array_local, rank
def mpi_scatter_exclude_rank_0(scatter_array):
    #Function is for rank 0 to be used as a saving processor - all other processors will complete tasks.
    from mpi4py import MPI
    import sys
    import numpy as np

    comm = MPI.COMM_WORLD
    num_procs = comm.Get_size()
    rank = comm.Get_rank()
    print(f'sys.argv: {sys.argv}, Number of procs: {num_procs}, rank: {rank}')

    num_workers = num_procs - 1
    remainder = np.size(scatter_array) % num_workers
    base_load = np.size(scatter_array) // num_workers

    if rank == 0:
        print(f'All processors will process at least {base_load} simulations.')
        print(f'{remainder} processors will process an additional simulation.')

    load_list = np.concatenate((np.zeros(1), np.ones(remainder) * (base_load + 1),
                                np.ones(num_workers - remainder) * base_load))

    if rank == 0:
        print(f'load_list={load_list}')

    scatter_array_local = np.zeros(int(load_list[rank]), dtype=np.int64)

    disp = np.zeros(num_procs)
    for i in range(1, num_procs):
        disp[i] = disp[i - 1] + load_list[i - 1]

    if rank == 0:
        dummy_recvbuf = np.zeros(1, dtype=np.int64)
        comm.Scatterv([scatter_array, load_list, disp, MPI.INT64_T], dummy_recvbuf)
    else:
        comm.Scatterv([scatter_array, load_list, disp, MPI.INT64_T], scatter_array_local)
        print(f"Process {rank} received the {len(scatter_array_local)} element scattered array: {scatter_array_local}")

    return scatter_array_local, rank

######################################################################
# Cislunar functions
######################################################################

######################################################################
# Functions to make accessing Cislunar data easier
######################################################################
def sim_idxs(sim_path, filter=None):
    if filter is None:
        return read_csv(f'{sim_path}.csv', col='ast_id')
    else:
        data = read_csv(f'{sim_path}.csv')
        data = data.query(filter)
        return data['ast_id'].reset_index(drop=True)

    
def sim_keys():
    return ['r', 'v', 'semi_major_axis', 'eccentricity', 'inclination', 'true_longitude', 'argument_of_periapsis', 'longitude_of_ascending_node', 'true_anomaly', 'stable', 'lifetime', 'ejection', 'period']

def sim_data(ast_id, key=False, sim='cislunar_orbits_150km_to_2LD_point_moon'):
    if key is False:
        data = {}
        for key in sim_keys():
            data[key] = h5load(f'{path_to_cislunar}{sim}_all_data.h5', f'{ast_id}/{key}')
        return data
    return h5load(f'{path_to_cislunar}{sim}_all_data.h5', f'{ast_id}/{key}')


class MoonRotator:
    def __init__(self):
        self.mpm = ssapy.utils.moonPos(times) # ssapy.accel.MoonPositionModel()
    def __call__(self, r, t):
        rmoon = self.mpm(t)
        vmoon = (self.mpm(t+5.0) - self.mpm(t-5.0))/10.
        xhat = ssapy.utils.normed(rmoon.T).T
        vpar = np.einsum("ab,ab->b", xhat, vmoon) * xhat
        vperp = vmoon - vpar
        yhat = ssapy.utils.normed(vperp.T).T
        zhat = np.cross(xhat, yhat, axisa=0, axisb=0).T
        R = np.empty((3, 3, len(t)))
        R[0] = xhat
        R[1] = yhat
        R[2] = zhat
        return np.einsum("abc,cb->ca", R, r)
# rotator = MoonRotator()
def cislunar_to_gcrf(r, times):
    rotator = MoonRotator(times)
    return rotator(r, times)

def rotate_points_3d(points, axis=np.array([0, 0, 1]), theta=-np.pi/2):
    """
    Rotate a set of 3D points about a 3D axis by an angle theta in radians.

    Args:
        points (np.ndarray): The set of 3D points to rotate, as an Nx3 array.
        axis (np.ndarray): The 3D axis to rotate about, as a length-3 array. Default is the z-axis.
        theta (float): The angle to rotate by, in radians. Default is pi/2.

    Returns:
        np.ndarray: The rotated set of 3D points, as an Nx3 array.
    """
    # Normalize the axis to be a unit vector
    axis = axis/np.linalg.norm(axis)

    # Compute the quaternion representing the rotation
    qw = np.cos(theta/2)
    qx, qy, qz = axis * np.sin(theta/2)

    # Construct the rotation matrix from the quaternion
    R = np.array([
        [1-2*qy**2-2*qz**2, 2*qx*qy-2*qz*qw,   2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw,   1-2*qx**2-2*qz**2, 2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw,   2*qy*qz+2*qx*qw,   1-2*qx**2-2*qy**2]
    ])

    # Apply the rotation matrix to the set of points
    rotated_points = np.dot(R, points.T).T

    return rotated_points

def perpendicular_vectors(v):
    """Returns two vectors that are perpendicular to v and each other."""
    # Check if v is the zero vector
    if np.allclose(v, np.zeros_like(v)):
        raise ValueError("Input vector cannot be the zero vector.")
    
    # Choose an arbitrary non-zero vector w that is not parallel to v
    w = np.array([1., 0., 0.])
    if np.allclose(v, w) or np.allclose(v, -w):
        w = np.array([0., 1., 0.])
    u = np.cross(v, w)
    if np.allclose(u, np.zeros_like(u)):
        w = np.array([0., 0., 1.])
        u = np.cross(v, w)
    w = np.cross(v, u)
    
    return u,w
    
def points_on_circle(r, v, rad, num_points=4):
    # Convert inputs to numpy arrays
    r = np.array(r)
    v = np.array(v)

    # Find the perpendicular vectors to the given vector v
    if np.all(v[:2] == 0):
        if np.all(v[2] == 0):
            raise ValueError("The given vector v must not be the zero vector.")
        else:
            u = np.array([1, 0, 0])
    else:
        u = np.array([-v[1], v[0], 0])
    u = u/np.linalg.norm(u)
    w = np.cross(u, v)
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-15:
        # v is parallel to z-axis
        w = np.array([0, 1, 0])
    else:
        w = w/w_norm
    # Generate a sequence of angles for equally spaced points
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    # Compute the x, y, z coordinates of each point on the circle
    x = rad * np.cos(angles) * u[0] + rad * np.sin(angles) * w[0]
    y = rad * np.cos(angles) * u[1] + rad * np.sin(angles) * w[1]
    z = rad * np.cos(angles) * u[2] + rad * np.sin(angles) * w[2]

    # Apply rotation about z-axis by 90 degrees
    rot_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    rotated_points = np.dot(rot_matrix, np.column_stack((x, y, z)).T).T

    # Translate the rotated points to the center point r
    points_rotated = rotated_points + r.reshape(1, 3)

    return points_rotated

######################################################################
# SSAPY FUNCTIONS
######################################################################
def gcrf_to_cislunar(r, times=False, r_moon=[]):
    if r.ndim != 2:
        raise IndexError(f"input data shape: {np.shape(r)}, input should be 2 dimensions.")
        return None
    if np.shape(r)[1] == 3:
        r = r.T
        # print(f"Tranposed input to {np.shape(r)}")
    if np.size(r_moon) < 1:
        r_moon = ssapy.utils.moonPos(times)
    else:    
        # print('Lunar position(s) provided.')
        if r_moon.ndim != 2:
            raise IndexError(f"input moon data shape: {np.shape(r_moon)}, input should be 2 dimensions.")
            return None
        if np.shape(r_moon)[1] == 3:
            r_moon = r_moon.T
            # print(f"Tranposed input to {np.shape(r_moon)}")

    yt = -np.arctan2(r_moon[2],r_moon[0])
    
    lunar_norm = np.cross(unit_vector(r_moon[:,0]), unit_vector(r_moon[:,1]))
    print(lunar_norm)
    for i, ri in enumerate(r_moon.T[:-1]):
        lunar_norm = np.vstack(( lunar_norm, ( np.cross(unit_vector(r_moon[:,i]), unit_vector(r_moon[:,i+1])) ) ))
    if np.size(r) > 1:
        #Rotate about Y-axis so that Moon is in X-Y plane
        rot = []; rot_moon = []
        for i, ri in enumerate(r.T):
            rot.append(np.squeeze(np.asarray( ri * Ry(yt[i]) )) )
            rot_moon.append( np.squeeze(np.asarray( r_moon[:,i] * Ry(yt[i]) )) )
            lunar_norm[i] = np.squeeze(lunar_norm[i] * Ry(yt[i]))

        #Rotate about Z-axis so that Moon is on X-axis
        rot = np.array(rot); rot_moon = np.array(rot_moon)
        zt = np.arctan2(rot_moon[:,1], rot_moon[:,0])
        for i, ri in enumerate(rot):
            rot[i] = np.squeeze(np.asarray( ri * Rz(zt[i]) ) )
            lunar_norm[i] = np.squeeze(np.asarray( lunar_norm[i] * Rz(zt[i]) ) )
        #Rotate about X-axis so that *something* is on z-axis
        # xt = np.arctan2(rot_moon[:,2], rot_moon[:,1]);
        # for i, ri in enumerate(rot):
        #     rot[i] = np.squeeze(np.asarray( ri * Rx(xt[i]) ) )
    return rot.T

#orbital period from keplerian orbital elements (koe)
def orbital_period(a, mu_barycenter=mu.earth):
    return np.sqrt(4*np.pi**2/mu_barycenter * a**3)/units.day_to_second

def get_times(duration=(30,'day'), freq=(1,'hr'), t = ssapy.utils.Time("2025-01-01", scale='utc')):
    """
    Calculate a list of times spaced equally apart over a specified duration.
    
    Parameters
    ----------
    duration: int
        The length of time to calculate times for.
    freq: int, unit: str
        frequency of time outputs in units provided
    t: ssapy.utils.Time, optional
        The starting time. Default is "2025-01-01".
    
    Returns
    -------
    times: array-like
        A list of times spaced equally apart over the specified duration.
    """
    if isinstance(t, str):
        t = ssapy.utils.Time(t, scale='utc')
    unit_dict = {'second': 1, 'sec': 1, 's': 1, 'minute': 60, 'min': 60, 'hour': 3600, 'hr': 3600, 'h': 3600, 'day': 86400, 'd': 86400, 'week': 604800, 'month':2630016, 'mo':2630016, 'year': 31557600, 'yr': 31557600}
    dur_val, dur_unit = duration
    freq_val, freq_unit = freq
    if dur_unit[-1] == 's' and len(dur_unit) > 1:
        dur_unit = dur_unit[:-1]
    if freq_unit[-1] == 's' and len(freq_unit) > 1:
        freq_unit = freq_unit[:-1]
    if dur_unit.lower() not in unit_dict:
        raise ValueError(f'Error, {dur_unit} is not a valid time unit. Valid options are: {", ".join(unit_dict.keys())}.')
    if freq_unit.lower() not in unit_dict:
        raise ValueError(f'Error, {freq_unit} is not a valid time unit. Valid options are: {", ".join(unit_dict.keys())}.')
    dur_seconds = dur_val * unit_dict[dur_unit.lower()]
    freq_seconds = freq_val * unit_dict[freq_unit.lower()]
    timesteps = int(dur_seconds/freq_seconds) + 1
    
    times = t + np.linspace(0, dur_seconds, timesteps) / unit_dict['day']
    return times

def ssapy_best_prop(integration_timestep=60):
    #Most accurate trajectory
    earth = ssapy.get_body("earth")
    moon = ssapy.get_body("moon")
    sun = ssapy.get_body("sun")

    #Accelerations - pass a body object or string of body name.
    aEarth = ssapy.AccelKepler() + ssapy.AccelHarmonic(earth)
    aMoon = ssapy.AccelThirdBody(moon) + ssapy.AccelHarmonic(moon)
    aSun = ssapy.AccelThirdBody(sun)
    # aJupiter = ssapy.accel.AccelJupiter()
    aSolRad = ssapy.AccelSolRad()
    aEarthRad = ssapy.AccelEarthRad()
    aDrag = ssapy.AccelDrag()
    accel = aEarth + aMoon + aSun + aSolRad + aEarthRad + aDrag#
    #Build propagator
    prop = ssapy.propagator.RK78Propagator(accel, h=integration_timestep)
    return prop

def ssapy_kwargs(mass=250, area=0.022, CD=2.3, CR=1.3):
    # Asteroid parameters
    kwargs = dict(
        mass = mass,  # [kg]
        area = area,  # [m^2]
        CD = CD, #Drag coefficient
        CR = CR, #Radiation pressure coefficient
    )
    return kwargs
#Uses the current best propagator and acceleration models in ssapy
def ssapy_rv_from_koe(a=RGEO, e=0, inc=0, pa=0, raan=0, trueAnomaly=0, duration=(30,'day'), freq=(1,'hr'), start_date="2025-01-01", times=None, integration_timestep=10, mass=250, area=0.022, CD=2.3, CR=1.3):
    #Everything is in SI units, except time.
    #density #kg/m^3 --> density
    #Time span of integration #Only takes integer number of days. Unless providing your own time object.
    t = ssapy.utils.Time(start_date, scale='utc')
    if times is None:
        times = get_times(duration=duration, freq=freq, t=t)
    
    kwargs = ssapy_kwargs(mass, area, CD, CR)
    kElements = [a, e, inc, pa, raan, trueAnomaly] # (a, e, i, pa, raan, trueAnomaly, t, mu)
    orbit = ssapy.Orbit.fromKeplerianElements(*kElements, t, propkw=kwargs)
    # print(f'Initial orbit: a: {a:.3e} m, e: {e:.2f}, i: {inc:.2f}, pa: {pa:.2f}, raan: {raan:.2f}, trueAnomaly: {trueAnomaly:.2f}, period: {orbit.period/units.day_to_second:.2f} days.')
    
    prop = ssapy_best_prop(integration_timestep)

    #Calculate entire satellite trajectory
    try:
        r, v = ssapy.rv(orbit, times, prop)
        return r, v
    except (RuntimeError, ValueError) as err:
        print(err)
        return np.nan, np.nan
   
def ssapy_rv_from_rv(r=RGEO, v=False, duration=(30,'day'), freq=(1,'hr'), start_date="2025-01-01", times=None, integration_timestep=10, mass=250, area=0.022, CD=2.3, CR=1.3):
    #Everything is in SI units, except time.
    #density #kg/m^3 --> density

    #Time span of integration #Only takes integer number of days. Unless providing your own time object.
    t = ssapy.utils.Time(start_date, scale='utc')
    if times is None:
        times = get_times(duration=duration, freq=freq, t=t)

    kwargs = ssapy_kwargs(mass, area, CD, CR)
    orbit = ssapy.Orbit(r, v, t, propkw=kwargs)
    
    prop = ssapy_best_prop(integration_timestep)
    #Calculate entire satellite trajectory
    try:
        r, v = ssapy.rv(orbit, times, prop)
        return r, v
    except (RuntimeError, ValueError) as err:
        print(err)
        r, v = np.nan, np.nan
    return r, v
    
def icrs_to_gcrs(r_icrs, t):
    """
    Convert ICRS coordinates to GCRS coordinates
    r_icrs: ICRS coordinates
    t: observation time as an Astropy time object
    """
    earth = get_body_barycentric(body="earth", time=t).xyz.to("m").value # in ICRS coordinates.
    return r_icrs - earth

def bcrs_to_gcrs(r_bcrs, t):
    """
    Rotate coordinates in ecliptic plane to equatorial plane then shift to an Earth centered coordinate system.
    r: cartesian coordinates measured from ecliptic plane.
    t: observation time as an Astropy time object
    """
    earth = get_body_barycentric(body="earth", time=t).xyz.to("m").value # in ICRS coordinates.
    obliquity = 0.40909280420293637
    rot_matrix = np.array([[1, 0, 0],
                           [0, np.cos(obliquity), -np.sin(obliquity)],
                           [0, np.sin(obliquity), np.cos(obliquity)]])
    return np.dot(rot_matrix, r_bcrs) - earth
def _gpsToTT(t):
    # Assume t is GPS seconds.  Convert to TT seconds by adding 51.184.
    # Divide by 86400 to get TT days.
    # Then add the TT time of the GPS epoch, expressed as an MJD, which
    # is 44244.0
    return 44244.0 + (t + 51.184)/86400
def mercuryPos(t):
    """
    Calculate the position of Mercury at a given set of times in the Geocentric Celestial Reference Frame (GCRF).
    
    Parameters
    ----------
    t: array_like
        An array of astropy.time.Time objects representing the times for which to calculate the position of mercury.
    
    Returns
    -------
    positions: ndarray
        An (n, 3) array of positions, where n is the number of times and each row contains the x, y, and z
        coordinates of mercury at a given time in the GCRF, in meters.
    """
    if not isinstance(t, Time):
        t = Time(t, format='gps')
    # Calculate the positions of Jupiter and the Earth at the given times
    mercury = get_body_barycentric(body="mercury", time=t).xyz.to("m").value
    earth = get_body_barycentric(body="earth", time=t).xyz.to("m").value
    return mercury - earth

def venusPos(t):
    """
    Calculate the position of Venus at a given set of times in the Geocentric Celestial Reference Frame (GCRF).
    
    Parameters
    ----------
    t: array_like
        An array of astropy.time.Time objects representing the times for which to calculate the position of Venus.
    
    Returns
    -------
    positions: ndarray
        An (n, 3) array of positions, where n is the number of times and each row contains the x, y, and z
        coordinates of Venus at a given time in the GCRF, in meters.
    """
    if not isinstance(t, Time):
        t = Time(t, format='gps')
    # Calculate the positions of Jupiter and the Earth at the given times
    venus = get_body_barycentric(body="venus", time=t).xyz.to("m").value
    earth = get_body_barycentric(body="earth", time=t).xyz.to("m").value
    return venus - earth

def marsPos(t):
    """
    Calculate the position of mars at a given set of times in the Geocentric Celestial Reference Frame (GCRF).
    
    Parameters
    ----------
    t: array_like
        An array of astropy.time.Time objects representing the times for which to calculate the position of mars.
    
    Returns
    -------
    positions: ndarray
        An (n, 3) array of positions, where n is the number of times and each row contains the x, y, and z
        coordinates of mars at a given time in the GCRF, in meters.
    """
    if not isinstance(t, Time):
        t = Time(t, format='gps')
    # Calculate the positions of Jupiter and the Earth at the given times
    mars = get_body_barycentric(body="mars", time=t).xyz.to("m").value
    earth = get_body_barycentric(body="earth", time=t).xyz.to("m").value
    return mars - earth
    
def jupiterPos(t, kepler=False):
    """
    Calculate the position of Jupiter at a given set of times in the Geocentric Celestial Reference Frame (GCRF).
    
    Parameters
    ----------
    t: array_like
        An array of astropy.time.Time objects representing the times for which to calculate the position of Jupiter.
    
    Returns
    -------
    positions: ndarray
        An (n, 3) array of positions, where n is the number of times and each row contains the x, y, and z
        coordinates of Jupiter at a given time in the GCRF, in meters.
    """
    if not isinstance(t, Time):
        t = Time(t, format='gps')
    if kepler:
        # Keplerian orbital elements
        a = 778000000000.
        e = 0.0489
        i = 1.303 / 180 * np.pi
        Ω = 100.464 / 180 * np.pi
        ω = 273.867 / 180 * np.pi
        M_0 = 19. / 180 * np.pi

        # Time since perihelion
        n = np.sqrt(1.3271244004193938e+20 / a**3)
        T = (t.jd - 2451545.0) * 24 * 3600
        M = M_0 + n * T

        # Eccentric anomaly
        E = np.arctan2(np.sqrt(1 - e) * np.sin(M), np.cos(M) - e)

        # True anomaly
        ν = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))

        # Distance from the Sun
        r = a * (1 - e**2) / (1 + e * np.cos(ν))

        # Heliocentric spherical coordinates
        x = r * (np.cos(Ω) * np.cos(ω + ν) - np.sin(Ω) * np.sin(ω + ν) * np.cos(i))
        y = r * (np.sin(Ω) * np.cos(ω + ν) + np.cos(Ω) * np.sin(ω + ν) * np.cos(i))
        z = r * np.sin(ω + ν) * np.sin(i)
        return bcrs_to_gcrs(np.array([x, y, z]), t)
    else:
        #This gets the position in GCRF but is very slow
        # get_body(body="jupiter", time=t).cartesian.xyz.to("m").value
#         # Set the solar system ephemeris to use - default is JPL
#         solar_system_ephemeris.set("jpl")
        # Calculate the positions of Jupiter and the Earth at the given times
        jupiter = get_body_barycentric(body="jupiter", time=t).xyz.to("m").value
        earth = get_body_barycentric(body="earth", time=t).xyz.to("m").value
        return jupiter - earth

def saturnPos(t):
    """
    Calculate the position of Saturn at a given set of times in the Geocentric Celestial Reference Frame (GCRF).
    
    Parameters
    ----------
    t: array_like
        An array of astropy.time.Time objects representing the times for which to calculate the position of Saturn.
    
    Returns
    -------
    positions: ndarray
        An (n, 3) array of positions, where n is the number of times and each row contains the x, y, and z
        coordinates of Saturn at a given time in the GCRF, in meters.
    """
    if not isinstance(t, Time):
        t = Time(t, format='gps')
    # Calculate the positions of Jupiter and the Earth at the given times
    saturn = get_body_barycentric(body="saturn", time=t).xyz.to("m").value
    earth = get_body_barycentric(body="earth", time=t).xyz.to("m").value
    return saturn - earth

def uranusPos(t):
    """
    Calculate the position of uranus at a given set of times in the Geocentric Celestial Reference Frame (GCRF).
    
    Parameters
    ----------
    t: array_like
        An array of astropy.time.Time objects representing the times for which to calculate the position of uranus.
    
    Returns
    -------
    positions: ndarray
        An (n, 3) array of positions, where n is the number of times and each row contains the x, y, and z
        coordinates of uranus at a given time in the GCRF, in meters.
    """
    if not isinstance(t, Time):
        t = Time(t, format='gps')
    # Calculate the positions of Jupiter and the Earth at the given times
    uranus = get_body_barycentric(body="uranus", time=t).xyz.to("m").value
    earth = get_body_barycentric(body="earth", time=t).xyz.to("m").value
    return uranus - earth

def neptunePos(t):
    """
    Calculate the position of Neptune at a given set of times in the Geocentric Celestial Reference Frame (GCRF).
    
    Parameters
    ----------
    t: array_like
        An array of astropy.time.Time objects representing the times for which to calculate the position of Neptune.
    
    Returns
    -------
    positions: ndarray
        An (n, 3) array of positions, where n is the number of times and each row contains the x, y, and z
        coordinates of Neptune at a given time in the GCRF, in meters.
    """
    if not isinstance(t, Time):
        t = Time(t, format='gps')
    # Calculate the positions of Jupiter and the Earth at the given times
    neptune = get_body_barycentric(body="Neptune", time=t).xyz.to("m").value
    earth = get_body_barycentric(body="earth", time=t).xyz.to("m").value
    return neptune - earth

def sunPos(t, fast=True):
    """Compute GCRF position of the sun.

    Parameters
    ----------
    t : float or astropy.time.Time
        If float or array of float, then should correspond to GPS seconds; i.e.,
        seconds since 1980-01-06 00:00:00 UTC
    fast : bool
        Use fast approximation?

    ReturnsF
    -------
    r : array_like (n)
        position in meters
    """
    if isinstance(t, Time):
        t = t.gps
    if fast:
        # MG section 3.3.2
        T = (_gpsToTT(t) - 51544.5)/36525.0
        M = 6.239998880168239 + 628.3019326367721 * T
        lam = (4.938234585592756 + M
            + 0.03341335890206922 * np.sin(M)
            + 0.00034906585039886593 * np.sin(2*M)
        )
        rs = (149.619 - 2.499*np.cos(M) - 0.021*np.cos(2*M))*1e9
        obliquity = 0.40909280420293637
        co, so = np.cos(obliquity), np.sin(obliquity)
        cl, sl = np.cos(lam), np.sin(lam)
        r = rs * np.array([cl, sl*co, sl*so])
    else:
        pvh, _ = erfa.epv00(2400000.5, _gpsToTT(t))
        r = pvh['p'] * -149597870700  # AU -> m
    return r

def moonPos(t):
    """Compute GCRF position of the moon.

    Parameters
    ----------
    t : float or astropy.time.Time
        If float or array of float, then should correspond to GPS seconds; i.e.,
        seconds since 1980-01-06 00:00:00 UTC

    Returns
    -------
    r : array_like (n)
        position in meters
    """
    if isinstance(t, Time):
        t = t.gps
    # MG section 3.3.2
    T = (_gpsToTT(t) - 51544.5)/36525.0
    # fundamental arguments (3.47)
    L0 = 3.810335976843669 + 8399.684719711557*T
    l = 2.3555473221057053 + 8328.69142518676*T
    lp = 6.23999591310851 + 628.3019403162209*T
    D = 5.198467889454092 + 7771.377143901714*T
    F = 1.6279179861529427 + 8433.46617912181*T
    # moon longitude (3.48)
    dL = (
        22640*np.sin(l) + 769*np.sin(2*l)
        - 4586*np.sin(l-2*D) + 2370*np.sin(2*D)
        - 668*np.sin(lp) - 412*np.sin(2*F)
        - 212*np.sin(2*l-2*D) - 206*np.sin(l+lp-2*D)
        + 192*np.sin(l+2*D) - 165*np.sin(lp-2*D)
        + 148*np.sin(l-lp) - 125*np.sin(D)
        - 110*np.sin(l+lp) - 55*np.sin(2*F-2*D)
    )
    L = L0 + np.deg2rad(dL/3600)
    # moon latitude (3.49)
    beta = np.deg2rad((
        18520*np.sin(F+L-L0+np.deg2rad((412*np.sin(2*F)+541*np.sin(lp))/3600))
        - 526*np.sin(F-2*D) + 44*np.sin(l+F-2*D)
        - 31*np.sin(-l+F-2*D) - 25*np.sin(-2*l+F)
        - 23*np.sin(lp+F-2*D) + 21*np.sin(-l+F)
        + 11*np.sin(-lp+F-2*D)
    )/3600)
    # moon distance (3.50)
    r = (
        385000 - 20905*np.cos(l) - 3699*np.cos(2*D-l)
        - 2956*np.cos(2*D) - 570*np.cos(2*l) + 246*np.cos(2*l-2*D)
        - 205*np.cos(lp-2*D) - 171*np.cos(l+2*D)
        - 152*np.cos(l+lp-2*D)
    )*1e3
    r_ecliptic = r*np.array([
        np.cos(L)*np.cos(beta),
        np.sin(L)*np.cos(beta),
        np.sin(beta)
    ])
    obliquity = 0.40909280420293637
    co, so = np.cos(obliquity), np.sin(obliquity)
    rot = np.array([[1,0,0],[0,co,so],[0,-so,co]])
    return rot.T @ r_ecliptic

def earthPos(t, fast=False):
    """
    Calculate the position of Earth at a given set of times in the International Celestial Reference System (ICRS).
    
    Parameters
    ----------
    t: array_like
        An array of astropy.time.Time objects representing the times for which to calculate the position of Earth. If float is provided, gps seconds are assumed.
    
    Returns
    -------
    positions: ndarray
        An (n, 3) array of positions, where n is the number of times and each row contains the x, y, and z
        coordinates of Jupiter at a given time in the GCRF, in meters.
    """
    if not isinstance(t, Time):
        t = Time(t, format='gps')
    if fast:
        pass # not implemented
    else:
        earth = get_body_barycentric(body="earth", time=t).xyz.to("m").value
    return earth

#Generate orbits near stable orbit.
def get_similar_orbits(r0,v0, rad=1e5, num_orbits=4, duration=(90,'days'), freq=(1,'hour'), start_date="2025-1-1", mass=250):
    r0 = np.reshape(r0, (1, 3))
    v0 = np.reshape(v0, (1, 3))
    print(r0, v0)
    for idx, point in enumerate(points_on_circle(r0, v0, rad=rad, num_points=num_orbits)):
        #Calculate entire satellite trajectory
        r, v = ssapy_rv_from_rv(r=point, v=v0, duration=duration, freq=freq, start_date=start_date, integration_timestep=10, mass=mass, area=mass/W_rho+.01, CD=2.3, CR=1.3)
        if idx == 0:
            trajectories = np.concatenate((r0,v0), axis=1)[:len(r)]
        rv = np.concatenate((r,v), axis=1)
        trajectories = np.dstack((trajectories, rv))
    return trajectories

def lyapunov_exponent(r, v, duration, freq, start_date, perturbation, time_between_data=1, lyapunov_type='perturbation'):
    """
    Calculate the Lyapunov exponent for a cislunar orbit.

    Parameters:
    - r: An (n,3) array of positions [x, y, z].
    - v: An (n,3) array of velocities [vx, vy, vz].
    - duration: tuple(time, unit) time with given unit to integrate the perturbed orbit.
    - freq: tuple(time, unit) time with given unit to output statevector.
    - start_date: str: sets the position of the Solar System for integration.
    - perturbation: float: Small perturbation applied to the initial position and velocity.
    - time_between_data: float: the amount of time with given unit between statevectors, default is 1 in units of duration.

    Returns:
    - The Lyapunov exponent for the cislunar orbit.
    """
    num_states = len(r)
    pr, pv = ssapy_rv_from_rv(r[0] + np.random.randn(3) * perturbation, v[0] + np.random.randn(3) * perturbation, duration=duration, freq=freq, start_date=start_date, integration_timestep=10, mass=250, area=0.022, CD=2.3, CR=1.3)
    if lyapunov_type == 'perturbation':
        delta_states = np.linalg.norm(pr - r, axis=1)
        time_series_le = np.log(delta_states / perturbation) * 1 / time_between_data
        lyapunov_exponent = np.sum(time_series_le) / num_states
    if lyapunov_type == 'derivative':
        delta_states = np.linalg.norm(pr - r, axis=1)
        time_series_le = np.log(delta_states[1:] / delta_states[:-1]) * 1 / time_between_data
        lyapunov_exponent = np.sum(time_series_le) / num_states
    return lyapunov_exponent, time_series_le

def megno(r):
    n_states = len(r)
    perturbed_states = r + 1e-8 * np.random.randn(n_states, 3)
    delta_states = perturbed_states - r
    delta_states_norm = np.linalg.norm(delta_states, axis=1)
    ln_delta_states_norm = np.log(delta_states_norm)

    megno_values = np.zeros(n_states)

    for i in range(1, n_states):
        m = np.mean(ln_delta_states_norm[:i])
        megno = (ln_delta_states_norm[i] + 2 * m) / (i)
        megno_values[i] = megno

    return np.mean(megno_values)


print("Imported cislunar_utilities.py.")