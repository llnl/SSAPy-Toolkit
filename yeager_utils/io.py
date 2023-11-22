import numpy as np
import h5py
import pandas as pd
import csv
from six.moves import cPickle as pickle  # for performance
import os
import glob
import shutil
import psutil
from mpi4py import MPI
import threading


def mpi_scatter(scatter_array):
    comm = MPI.COMM_WORLD  # Defines the default communicator
    num_procs = comm.Get_size()  # Stores the number of processes in size.
    rank = comm.Get_rank()  # Stores the rank (pid) of the current process
    # stat = MPI.Status()
    print(f'Number of procs: {num_procs}, rank: {rank}')
    remainder = np.size(scatter_array) % num_procs
    base_load = np.size(scatter_array) // num_procs
    if rank == 0:
        print('All processors will process at least {0} simulations.'.format(
            base_load))
        print('{0} processors will process an additional simulations'.format(
            remainder))
    load_list = np.concatenate((np.ones(remainder) * (base_load + 1),
                                np.ones(num_procs - remainder) * base_load))
    if rank == 0:
        print('load_list={0}'.format(load_list))
    if rank < remainder:
        scatter_array_local = np.zeros(base_load + 1, dtype=np.int64)
    else:
        scatter_array_local = np.zeros(base_load, dtype=np.int64)
    disp = np.zeros(num_procs)
    for i in range(np.size(load_list)):
        if i == 0:
            disp[i] = 0
        else:
            disp[i] = disp[i - 1] + load_list[i - 1]
    comm.Scatterv([scatter_array, load_list, disp, MPI.DOUBLE], scatter_array_local)
    print(f"Process {rank} received the scattered arrays: {scatter_array_local}")
    return scatter_array_local, rank


def mpi_scatter_exclude_rank_0(scatter_array):
    # Function is for rank 0 to be used as a saving processor - all other processors will complete tasks.
    comm = MPI.COMM_WORLD
    num_procs = comm.Get_size()
    rank = comm.Get_rank()
    print(f'Number of procs: {num_procs}, rank: {rank}')

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


def exists(pathname):
    if os.path.isdir(pathname):
        exists = True
    elif os.path.isfile(pathname):
        exists = True
    else:
        exists = False
    return exists


def mkdir(pathname):
    if not exists(pathname):
        os.makedirs(pathname)
        print("Directory '%s' created" % pathname)
    return


def mvdir(source_, destination_):
    print(f'Moving {source_} to {destination_}')
    shutil.move(source_, destination_)
    return


def rmdir(source_):
    if not exists(source_):
        print(f'{source_}, does not exist, no delete.')
    else:
        print(f'Deleted {source_}')
        shutil.rmtree(source_)
    return


def rmfile(pathname):
    if exists(pathname):
        os.remove(pathname)
        print("File: '%s' deleted." % pathname)
    return


def listdir(dir_path='*', files_only=False, exclude=None):
    expanded_paths = glob(dir_path)

    if files_only:
        files = [f for f in expanded_paths if os.path.isfile(f)]
        print(f'{len(files)} files in {dir_path}')
    else:
        files = expanded_paths
        print(f'{len(files)} files in {dir_path}')

    if exclude:
        new_files = [file for file in files if exclude not in os.path.basename(file)]
        files = new_files
    return sorted(files)


def get_memory_usage():
    print(f"Memory used: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3:.2f} GB")
######################################################################
# Load and Save Functions
######################################################################
######################################################################
# Pickles
######################################################################


def psave(filename_, data_):
    with open(filename_, 'wb') as f:
        pickle.dump(data_, f)
    f.close()
    return


def pload(filename_):
    try:
        # print('Openning: ' + current_filename)
        with open(filename_, 'rb') as f:
            data = pickle.load(f)
        f.close()
    except (EOFError, FileNotFoundError, OSError, pickle.UnpicklingError) as err:
        print(f'{err} - current_filename')
        return []
    return data


def merge_dicts(file_names, save_path):
    number_of_files = len(file_names)
    master_dict = {}
    for count, file in enumerate(file_names):
        print(f'Merging dict: {count+1} of {number_of_files}, name: {file}, num of master keys: {len(master_dict.keys())}, num of new keys: {len(master_dict.keys())}')
        master_dict.update(pload(file))
    print('Beginning final save.')
    psave(save_path, master_dict)
    return

######################################################################
# HDF5 py files h5py
######################################################################


def append_h5(filename, pathname, append_data):
    """
    Append data to key in HDF5 file.

    Args:
        filename (str): The filename of the HDF5 file.
        pathname (str): The path to the key in the HDF5 file.
        append_data (any): The data to be appended.

    Returns:
        None
    """
    try:
        with h5py.File(filename, "r+") as f:
            if pathname in f:
                path_data_old = np.array(f.get(pathname))
                new_data = np.append(path_data_old, np.array(append_data))
                f[pathname] = new_data
            else:
                f.create_dataset(pathname, data=np.array(append_data), maxshape=None)
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except (ValueError, KeyError) as err:
        print(f"Error: {err}")


def overwrite_h5(filename, pathname, new_data):
    """
    Overwrite key in HDF5 file.

    Args:
        filename (str): The filename of the HDF5 file.
        pathname (str): The path to the key in the HDF5 file.
        new_data (any): The data to be overwritten.

    Returns:
        None
    """
    try:
        with h5py.File(filename, "a") as f:
            f.create_dataset(pathname, data=new_data, maxshape=None)
        f.close()
    except (FileNotFoundError, ValueError, KeyError):
        try:
            with h5py.File(filename, 'r+') as f:
                del f[pathname]
            f.close()
        except (FileNotFoundError, ValueError, KeyError) as err:
            print(f'Error: {err}')
        try:
            with h5py.File(filename, "a") as f:
                f.create_dataset(pathname, data=new_data, maxshape=None)
            f.close()
        except (FileNotFoundError, ValueError, KeyError) as err:
            print(f'File: {filename}{pathname}, Error: {err}')


def save_h5(filename, pathname, data):
    """
    Save data to HDF5 file.

    Args:
        filename (str): The filename of the HDF5 file.
        pathname (str): The path to the data in the HDF5 file.
        data (any): The data to be saved.

    Returns:
        None
    """
    with h5py.File(filename, "a") as f:
        try:
            f.create_dataset(pathname, data=data, maxshape=None)
        except ValueError as err:
            print(f"Did not save, key: {pathname} exists in file: {filename}. {err}")


def read_h5(filename, pathname):
    """
    Load data from HDF5 file.

    Args:
        filename_ (str): The filename of the HDF5 file.
        pathname_ (str): The path to the data in the HDF5 file.

    Returns:
        The data loaded from the HDF5 file.
    """
    with h5py.File(filename, 'r') as f:
        data = np.array(f.get(pathname))
    f.close()
    return data


def read_h5_all(filename_):
    """
    Load all data from HDF5 file.

    Args:
        filename_ (str): The filename of the HDF5 file.

    Returns:
        A dictionary of data loaded from the HDF5 file.
    """
    with h5py.File(filename_, "r") as f:
        # List all groups
        keys = list(f.keys())
        return_data = {key: np.array(f.get(key)) for key in keys}

    return return_data, keys


def h5_keys(filename):
    """
    List all groups in HDF5 file.

    Args:
        filename_ (str): The filename of the HDF5 file.

    Returns:
        A list of group keys in the HDF5 file.
    """
    with h5py.File(filename, "r") as f:
        # List all groups
        group_keys = list(f.keys())
    return group_keys


def h5_key_exists(filename, key):
    """
    Checks if a key exists in an HDF5 file.

    Args:
        filename (str): The filename of the HDF5 file.
        key (str): The key to check.

    Returns:
        True if the key exists, False otherwise.
    """

    try:
        with h5py.File(filename, 'r') as f:
            return str(key) in f
    except IOError:
        return False


######################################################################
# CSV
######################################################################


def makedf(df):
    if isinstance(df, (list, dict)):
        return pd.DataFrame.from_dict(df)
    else:
        return df


def header_csv(file_name, sep=None):
    """
    Get the header of a CSV file.

    Args:
        file_name (str): The filename of the CSV file.
        sep (str) optional: The delimiter used in the CSV file.

    Returns:
        A list of the header fields.
    """
    if sep is None:
        sep = guess_csv_delimiter(file_name)  # Guess the delimiter
    with open(file_name, 'r') as infile:
        reader = csv.DictReader(infile, delimiter=sep)
        fieldnames = reader.fieldnames
    return fieldnames


def read_csv(file_name, sep=None, dtypes=None, col=False, to_np=False, drop_nan=False, skiprows=[]):
    """
    Read a CSV file with options.

    Parameters
    ----------
    file_name : str
        The path to the CSV file.
    sep : str, optional
        The delimiter used in the CSV file. If None, delimiter will be guessed.
    dtypes : dict, optional
        Dictionary specifying data types for columns.
    col : bool or list of str, optional
        Specify columns to read. If False, read all columns.
    to_np : bool, optional
        Convert the loaded data to a NumPy array.
    drop_nan : bool, optional
        Drop rows with missing values (NaNs) from the loaded DataFrame.
    skiprows : list of int, optional
        Rows to skip while reading the CSV file.

    Returns
    -------
    DataFrame or NumPy array
        The loaded data in either a DataFrame or a NumPy array format.
    """
    if sep is None:
        sep = guess_csv_delimiter(file_name)  # Guess the delimiter

    if col is False:
        try:
            df = pd.read_csv(file_name, sep=sep, on_bad_lines='skip', skiprows=skiprows, dtype=dtypes)
        except TypeError:
            df = pd.read_csv(file_name, sep=sep, skiprows=skiprows, dtype=object)
    else:
        try:
            if not isinstance(col, list):
                col = [col]
            df = pd.read_csv(file_name, sep=sep, usecols=col, on_bad_lines='skip', skiprows=skiprows, dtype=dtypes)
        except TypeError:
            df = pd.read_csv(file_name, sep=sep, usecols=col, skiprows=skiprows, dtype=object)

    if drop_nan:
        df = df.dropna()

    if to_np:
        return np.squeeze(df.to_numpy())
    else:
        return df


def guess_csv_delimiter(file_name):
    """
    Guess the delimiter used in a CSV file.

    Args:
        file_name (str): The path to the CSV file.

    Returns:
        str: Guessed delimiter (one of ',', '\t', ';')
    """
    with open(file_name, 'r', newline='') as file:
        sample = file.read(4096)  # Read a sample of the file's contents
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        return dialect.delimiter


def save_csv(file_name, df, sep='\t', dtypes=None):
    """
    Save a Pandas DataFrame to a CSV file.

    Args:
        file_name (str): The path to the CSV file.
        df (DataFrame): The Pandas DataFrame to save.
        sep (str): The delimiter used in the CSV file.
        dtypes (dict): A dictionary specifying data types for columns.

    Returns:
        None
    """
    df = makedf(df)

    if dtypes:
        df = df.astype(dtypes)

    df.to_csv(file_name, index=False, sep=sep)
    print(f'Saved {file_name} successfully.')
    return


def append_csv(file_names, save_path='combined_data.csv', sep=None, dtypes=None, progress=None):
    """
    Appends multiple CSV files into a single CSV file.

    Args:
        file_names (list): A list of CSV file names.
        save_path (str): The path to the output CSV file. If not specified, the output will be saved to the current working directory.
        sep (str): The delimiter used in the CSV files.
        dtypes (dict): A dictionary specifying data types for columns.

    Returns:
        None
    """

    error_files = []
    dataframes = []
    for i, file in enumerate(file_names):
        try:
            if sep is None:
                sep = guess_csv_delimiter(file)  # Guess the delimiter
            df = pd.read_csv(file, sep=sep)
            dataframes.append(df)
            if progress is not None:
                get_memory_usage()
                print(f"Appended {i+1} of {len(file_names)}.")
        except FileNotFoundError:
            error_files.append(file)

    combined_df = pd.concat(dataframes, ignore_index=True)

    if dtypes:
        combined_df = combined_df.astype(dtypes)

    if save_path:
        combined_df.to_csv(save_path, sep=sep, index=False)
    else:
        combined_df.to_csv('combined_data.csv', sep=sep, index=False)

    print(f'The final dataframe has {combined_df.shape[0]} rows and {combined_df.shape[1]} columns.')
    if error_files:
        print(f'The following files could not be found: {error_files}')


def append_csv_on_disk(csv_files, output_file):
    # Assumes each file has the same delimiters
    delimiter = guess_csv_delimiter(csv_files[0])
    # Open the output file for writing
    with open(output_file, 'w', newline='') as outfile:
        # Initialize the CSV writer
        writer = csv.writer(outfile, delimiter=delimiter)

        # Write the header row from the first CSV file
        with open(csv_files[0], 'r', newline='') as first_file:
            reader = csv.reader(first_file, delimiter=delimiter)
            header = next(reader)
            writer.writerow(header)

            # Write the data rows from the first CSV file
            for row in reader:
                writer.writerow(row)

        # Write the data rows from the remaining CSV files
        for file in csv_files[1:]:
            with open(file, 'r', newline='') as infile:
                reader = csv.reader(infile, delimiter=delimiter)
                next(reader)  # Skip the header row
                for row in reader:
                    writer.writerow(row)
    print(f'Completed appending of: {output_file}.')


def save_csv_line(file_name, df, sep='\t', dtypes=None):
    """
    Save a Pandas DataFrame to a CSV file, appending the DataFrame to the file if it exists.

    Args:
        file_name (str): The path to the CSV file.
        df (DataFrame): The Pandas DataFrame to save.
        sep (str): The delimiter used in the CSV file.

    Returns:
        None
    """
    df = makedf(df)
    if dtypes:
        df = df.astype(dtypes)
    if exists(file_name):
        df.to_csv(file_name, mode='a', index=False, header=False, sep=sep)
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
        except IOError:
            return False
    return False


def pd_flatten(data, factor=1):
    tmp = []
    for x in data:
        try:
            tmp.extend(x[1:-1].split(','))
        except TypeError:
            tmp.append(x)
    return [float(x) / factor for x in tmp]

#
# TURN AN ARRAY SAVED AS A STRING BACK INTO AN ARRAY


def str_to_array(s):
    s = s.replace('[', '').replace(']', '')  # Remove square brackets
    return np.array([float(x) for x in s.split(',')])


def pdstr_to_arrays(df):
    return df.apply(str_to_array).to_numpy()


def allfiles(dirName=os.getcwd()):
    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    return listOfFiles
