# flake8: noqa: E501
import numpy as np
import h5py
import pandas as pd
import csv
from six.moves import cPickle as pickle  # for performance
import os
from glob import glob
import shutil
import psutil
from .utils import sortbynum
from typing import Any, List, Dict, Optional, Union, Tuple, Callable


def file_exists(filename: str) -> bool:
    """
    Check if a file with the given name exists (including its extensions).

    Parameters
    ----------
    filename : str
        The name of the file to check.

    Returns
    -------
    bool
        True if the file exists, False otherwise.
    """
    name, _ = os.path.splitext(filename)
    return bool(glob(f"{name}.*"))


def exists(pathname: str) -> bool:
    """
    Check if a given path is either a file or a directory.

    Parameters
    ----------
    pathname : str
        The path to check.

    Returns
    -------
    bool
        True if the path exists as either a directory or file, False otherwise.
    """
    if os.path.isdir(pathname) or os.path.isfile(pathname):
        return True
    return False


def mkdir(pathname: str) -> None:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    pathname : str
        The path of the directory to create.

    Returns
    -------
    None
    """
    if not exists(pathname):
        try:
            os.makedirs(pathname)
            print(f"Directory '{pathname}' created.")
        except OSError as e:
            print(f"Error creating directory {pathname}: {e}")


def mvdir(source_: str, destination_: str) -> None:
    """
    Move a directory from the source path to the destination path.

    Parameters
    ----------
    source_ : str
        The source directory path.
    destination_ : str
        The destination directory path.

    Returns
    -------
    None
    """
    if not exists(destination_):
        print(f"Moving {source_} to {destination_}")
        shutil.move(source_, destination_)
    else:
        print(f"Destination path {destination_} already exists.")


def rmdir(source_: str) -> None:
    """
    Remove a directory and its contents.

    Parameters
    ----------
    source_ : str
        The directory path to remove.

    Returns
    -------
    None
    """
    if not exists(source_):
        print(f'{source_} does not exist, no delete.')
    else:
        try:
            shutil.rmtree(source_)
            print(f'Deleted {source_}')
        except OSError as e:
            print(f"Error deleting {source_}: {e}")


def rmfile(pathname: str) -> None:
    """
    Remove a file if it exists.

    Parameters
    ----------
    pathname : str
        The path to the file to delete.

    Returns
    -------
    None
    """
    if exists(pathname):
        os.remove(pathname)
        print(f"File: '{pathname}' deleted.")


def listdir(
    dir_path: str = '*', 
    files_only: bool = False, 
    exclude: Optional[str] = None, 
    sorted: bool = False
) -> List[str]:
    """
    List files in a directory, with options to filter, exclude, and sort.

    Parameters
    ----------
    dir_path : str, optional
        The directory path or pattern (default is '*'). If '*' is not included, it is appended.
    files_only : bool, optional
        If True, only files will be returned. Default is False.
    exclude : str, optional
        A substring to exclude from the file names.
    sorted : bool, optional
        If True, the file names will be sorted numerically.

    Returns
    -------
    List[str]
        A list of file paths in the directory that match the conditions.
    """
    if '*' not in dir_path:
        dir_path = os.path.join(dir_path, '*')
    expanded_paths = glob(dir_path)

    if files_only:
        files = [f for f in expanded_paths if os.path.isfile(f)]
        print(f'{len(files)} files in {dir_path}')
    else:
        files = expanded_paths
        print(f'{len(files)} files in {dir_path}')

    if exclude:
        files = [file for file in files if exclude not in os.path.basename(file)]
        
    if sorted:
        return sortbynum(files)
    else:
        return files


def get_memory_usage() -> float:
    """
    Get the current memory usage of the process in GB.

    Returns
    -------
    float
        The memory used by the process in gigabytes.
    """
    memory_used = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    print(f"Memory used: {memory_used:.2f} GB")
    return memory_used


def psave(filename_: str, data_: Any) -> None:
    """
    Save data to a file using pickle.

    Parameters
    ----------
    filename_ : str
        The path to the file where the data should be saved.
    data_ : Any
        The data to save, can be of any type.

    Returns
    -------
    None
    """
    with open(filename_, 'wb') as f:
        pickle.dump(data_, f)


def pload(filename_: str) -> Optional[Any]:
    """
    Load data from a pickle file.

    Parameters
    ----------
    filename_ : str
        The path to the file from which the data should be loaded.

    Returns
    -------
    Optional[Any]
        The data loaded from the file, or None if there was an error.
    """
    try:
        with open(filename_, 'rb') as f:
            data = pickle.load(f)
    except (EOFError, FileNotFoundError, OSError, pickle.UnpicklingError) as err:
        print(f'{err} - {filename_}')
        return None
    return data


def merge_dicts(file_names: List[str], save_path: str) -> None:
    """
    Merge multiple dictionaries from pickle files into a single dictionary and save it to a file.

    Parameters
    ----------
    file_names : list of str
        List of file paths containing pickled dictionaries.
    save_path : str
        The path where the merged dictionary should be saved.

    Returns
    -------
    None
    """
    number_of_files = len(file_names)
    master_dict: Dict[str, Any] = {}
    for count, file in enumerate(file_names):
        print(f'Merging dict: {count+1} of {number_of_files}, name: {file}, num of master keys: {len(master_dict.keys())}, num of new keys: {len(master_dict.keys())}')
        master_dict.update(pload(file) or {})  # Use an empty dict if loading failed
    print('Beginning final save.')
    psave(save_path, master_dict)


def append_h5(filename: str, pathname: str, append_data: Any) -> None:
    """
    Append data to an existing key in an HDF5 file.

    Args:
        filename (str): The filename of the HDF5 file.
        pathname (str): The path to the key in the HDF5 file.
        append_data (Any): The data to be appended.

    Returns:
        None
    """
    try:
        with h5py.File(filename, "a") as f:
            if pathname in f:
                path_data_old = np.array(f.get(pathname))
                append_data = np.append(path_data_old, np.array(append_data))
                del f[pathname]
            f.create_dataset(pathname, data=np.array(append_data), maxshape=None)
    except FileNotFoundError:
        print(f"File not found: {filename}\nCreating new dataset: {filename}")
        save_h5(filename, pathname, append_data)
    except (ValueError, KeyError) as err:
        print(f"Error: {err}")


def overwrite_h5(filename: str, pathname: str, new_data: Any) -> None:
    """
    Overwrite data for a key in an HDF5 file.

    Args:
        filename (str): The filename of the HDF5 file.
        pathname (str): The path to the key in the HDF5 file.
        new_data (Any): The new data to overwrite the old data.

    Returns:
        None
    """
    try:
        with h5py.File(filename, "a") as f:
            f.create_dataset(pathname, data=new_data, maxshape=None)
    except (FileNotFoundError, ValueError, KeyError) as err:
        try:
            with h5py.File(filename, 'r+') as f:
                del f[pathname]
        except (FileNotFoundError, ValueError, KeyError) as err:
            print(f'Error: {err}')
        try:
            with h5py.File(filename, "a") as f:
                f.create_dataset(pathname, data=new_data, maxshape=None)
        except (FileNotFoundError, ValueError, KeyError) as err:
            print(f'File: {filename}{pathname}, Error: {err}')


def save_h5(filename: str, pathname: str, data: Any) -> None:
    """
    Save data to an HDF5 file.

    Args:
        filename (str): The filename of the HDF5 file.
        pathname (str): The path to the data in the HDF5 file.
        data (Any): The data to be saved.

    Returns:
        None
    """
    try:
        with h5py.File(filename, "a") as f:
            f.create_dataset(pathname, data=data, maxshape=None)
            f.flush()
    except ValueError as err:
        print(f"Did not save, key: {pathname} exists in file: {filename}. {err}")
        return
    except (BlockingIOError, OSError) as err:
        print(f"\n{err}\nPath: {pathname}\nFile: {filename}\n")
        return


def read_h5(filename: str, pathname: str) -> Optional[np.ndarray]:
    """
    Load data from an HDF5 file.

    Args:
        filename (str): The filename of the HDF5 file.
        pathname (str): The path to the data in the HDF5 file.

    Returns:
        Optional[np.ndarray]: The data loaded from the HDF5 file, or None if the key does not exist.
    """
    try:
        with h5py.File(filename, 'r') as f:
            data = f.get(pathname)
            if data is None:
                return None
            else:
                return np.array(data)
    except (ValueError, KeyError, TypeError):
        return None
    except FileNotFoundError:
        print(f'File not found. {filename}')
        raise
    except (BlockingIOError, OSError) as err:
        print(f"\n{err}\nPath: {pathname}\nFile: {filename}\n")
        raise


def read_h5_all(file_path: str) -> dict:
    """
    Load all data from an HDF5 file.

    Args:
        file_path (str): The path to the HDF5 file.

    Returns:
        dict: A dictionary of all keys and their corresponding data in the HDF5 file.
    """
    data_dict = {}

    with h5py.File(file_path, 'r') as file:
        def traverse(group, path=''):
            for key, item in group.items():
                new_path = f"{path}/{key}" if path else key
                if isinstance(item, h5py.Group):
                    traverse(item, path=new_path)
                else:
                    data_dict[new_path] = item[()]

        traverse(file)
    return data_dict


def combine_h5(filename: str, files: List[str], verbose: bool = False, overwrite: bool = False) -> None:
    """
    Combine data from multiple HDF5 files into a single HDF5 file.

    Args:
        filename (str): The filename of the target HDF5 file.
        files (List[str]): A list of source HDF5 files to combine.
        verbose (bool): Whether to print detailed processing information.
        overwrite (bool): Whether to overwrite the target file if it exists.

    Returns:
        None
    """
    if overwrite and os.path.exists(filename):
        os.remove(filename)  # Ensure filename exists before trying to remove it
    for idx, file in enumerate(files):
        try:
            if not os.path.exists(file):
                print(f"Skipping inaccessible file: {file}")
                continue
            for key in h5_keys(file):
                if not h5_key_exists(filename, key):
                    save_h5(filename, key, read_h5(file, key))
        except Exception as e:
            print(f"Error processing file {file}: {e}")


def h5_keys(file_path: str) -> List[str]:
    """
    List all keys in an HDF5 file.

    Args:
        file_path (str): The file path of the HDF5 file.

    Returns:
        List[str]: A list of all keys in the HDF5 file.
    """
    keys_list = []
    with h5py.File(file_path, 'r') as file:
        def traverse(group, path=''):
            for key, item in group.items():
                new_path = f"{path}/{key}" if path else key
                if isinstance(item, h5py.Group):
                    traverse(item, path=new_path)
                else:
                    keys_list.append(new_path)
        traverse(file)
    return keys_list


def h5_root_keys(file_path: str) -> List[str]:
    """
    List all top-level keys in an HDF5 file.

    Args:
        file_path (str): The file path of the HDF5 file.

    Returns:
        List[str]: A list of top-level keys in the HDF5 file.
    """
    with h5py.File(file_path, 'r') as file:
        keys_in_root = list(file.keys())
        return keys_in_root


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


def makedf(df: Union[pd.DataFrame, List, Dict]) -> pd.DataFrame:
    """
    Converts a list or dictionary into a Pandas DataFrame.

    Args:
        df (Union[pd.DataFrame, List, Dict]): A DataFrame, list, or dictionary.

    Returns:
        pd.DataFrame: A DataFrame created from the input.
    """
    if isinstance(df, (list, dict)):
        return pd.DataFrame.from_dict(df)
    else:
        return df


def read_csv_header(file_name: str, sep: Optional[str] = None) -> List[str]:
    """
    Get the header of a CSV file.

    Args:
        file_name (str): The filename of the CSV file.
        sep (Optional[str]): The delimiter used in the CSV file.

    Returns:
        List[str]: A list of the header fields.
    """
    if sep is None:
        sep = guess_csv_delimiter(file_name)  # Guess the delimiter
    with open(file_name, 'r') as infile:
        reader = csv.DictReader(infile, delimiter=sep)
        fieldnames = reader.fieldnames
    return fieldnames


def read_csv(file_name: str, sep: Optional[str] = None, dtypes: Optional[Dict[str, Union[str, np.dtype]]] = None, 
             col: Union[bool, List[str], None] = False, to_np: bool = False, drop_nan: bool = False, 
             skiprows: List[int] = []) -> Union[pd.DataFrame, np.ndarray]:
    """
    Read a CSV file with options.

    Args:
        file_name (str): The path to the CSV file.
        sep (Optional[str]): The delimiter used in the CSV file.
        dtypes (Optional[Dict[str, Union[str, np.dtype]]]): Dictionary specifying data types for columns.
        col (Union[bool, List[str], None]): Specify columns to read.
        to_np (bool): Convert the loaded data to a NumPy array.
        drop_nan (bool): Drop rows with missing values.
        skiprows (List[int]): Rows to skip while reading the CSV file.

    Returns:
        Union[pd.DataFrame, np.ndarray]: A DataFrame or NumPy array with the loaded data.
    """
    if col and not isinstance(col, list):
        col = [col]  # Ensure col is always a list

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


def append_dict_to_csv(file_name: str, data_dict: Dict[str, List[Union[str, float, int]]], delimiter: str = '\t') -> None:
    """
    Appends data from a dictionary to a CSV file.

    Args:
        file_name (str): The path to the CSV file.
        data_dict (Dict[str, List[Union[str, float, int]]]): A dictionary where keys are column names and values are lists of column data.
        delimiter (str): The delimiter used in the CSV file.

    Returns:
        None
    """
    # Check if the input is a numpy array or DataFrame, and convert to dictionary if necessary
    if isinstance(data_dict, np.ndarray):
        # Convert ndarray to dictionary (assuming each column is a field)
        data_dict = {f'col{i}': data_dict[:, i].tolist() for i in range(data_dict.shape[1])}
    elif isinstance(data_dict, pd.DataFrame):
        # Convert DataFrame to dictionary (using columns as keys)
        data_dict = data_dict.to_dict(orient='list')

    # Extract keys and values from the dictionary
    keys = list(data_dict.keys())
    values = list(data_dict.values())

    # Determine the length of the arrays
    array_length = len(values[0])

    # Determine if the file exists
    file_exists = os.path.exists(file_name)

    # Open the CSV file in append mode
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)

        # Write header if the file doesn't exist
        if not file_exists:
            writer.writerow(keys)

        # Write each element from arrays as a new row
        for i in range(array_length):
            row = [values[j][i] for j in range(len(keys))]
            writer.writerow(row)


def guess_csv_delimiter(csv_file_path: str, sample_size: int = 32768, delimiters: List[str] = [',', ';', '\t', '|', ' ']) -> str:
    """
    Guesses the delimiter used in a CSV file.

    Args:
        csv_file_path (str): The path to the CSV file.
        sample_size (int): The number of bytes to read from the file to guess the delimiter.
        delimiters (List[str]): The list of possible delimiters to test.

    Returns:
        str: The detected delimiter or an error message if unable to detect.
    """
    with open(csv_file_path, 'r') as csvfile:
        sample = csvfile.read(sample_size)  # Read a larger sample size
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=delimiters)
            return dialect.delimiter
        except csv.Error:
            return "Could not determine delimiter"


def save_csv(file_name: str, df: pd.DataFrame, sep: str = '\t', dtypes: Optional[Dict[str, Union[str, np.dtype]]] = None) -> None:
    """
    Save a Pandas DataFrame to a CSV file.

    Args:
        file_name (str): The path to the CSV file.
        df (pd.DataFrame): The Pandas DataFrame to save.
        sep (str): The delimiter used in the CSV file.
        dtypes (Optional[Dict[str, Union[str, np.dtype]]]): A dictionary specifying data types for columns.

    Returns:
        None
    """
    df = makedf(df)

    if dtypes:
        df = df.astype(dtypes)

    df.to_csv(file_name, index=False, sep=sep)
    print(f'Saved {file_name} successfully.')
    return


def append_csv(file_names: List[str], save_path: str = 'combined_data.csv', sep: Optional[str] = None, 
               dtypes: Optional[Dict[str, Union[str, np.dtype]]] = None, progress: Optional[callable] = None) -> None:
    """
    Appends multiple CSV files into a single CSV file.

    Args:
        file_names (List[str]): A list of CSV file names.
        save_path (str): The path to the output CSV file. If not specified, the output will be saved to the current working directory.
        sep (Optional[str]): The delimiter used in the CSV files. If None, delimiter will be guessed.
        dtypes (Optional[Dict[str, Union[str, np.dtype]]]): A dictionary specifying data types for columns.
        progress (Optional[callable]): A function that can be used to track progress (e.g., printing memory usage).

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
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            error_files.append(file)
            print(f"Error processing file {file}: {e}")

    combined_df = pd.concat(dataframes, ignore_index=True)
    if dtypes:
        combined_df = combined_df.astype(dtypes)

    if save_path:
        combined_df.to_csv(save_path, sep=sep, index=False)
    else:
        combined_df.to_csv('combined_data.csv', sep=sep, index=False)

    print(f'The final dataframe has {combined_df.shape[0]} rows and {combined_df.shape[1]} columns.')
    if error_files:
        print(f'The following files ERRORED and were not included: {error_files}')
    return


def append_csv_on_disk(csv_files: List[str], output_file: str) -> None:
    """
    Appends multiple CSV files directly to a single output file on disk.

    Args:
        csv_files (List[str]): A list of CSV files to append.
        output_file (str): The path to the output CSV file.

    Returns:
        None
    """
    delimiter = guess_csv_delimiter(csv_files[0])
    # Open the output file for writing
    with open(output_file, 'w', newline='') as outfile:
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


def save_csv_header(filename: str, header: List[str], delimiter: str = '\t') -> None:
    """
    Saves a header row to a CSV file with a specified delimiter.

    Args:
        filename (str): The name of the file where the header will be saved.
        header (List[str]): A list of strings representing the column names.
        delimiter (str, optional): The delimiter to use between columns in the CSV file. Default is tab ('\t').

    Returns:
        None
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)
        writer.writerow(header)


def save_csv_array_to_line(filename: str, array: List[Union[str, float, int]], delimiter: str = '\t') -> None:
    """
    Appends a single row of data to a CSV file with a specified delimiter.

    Args:
        filename (str): The name of the file to which the row will be appended.
        array (List[Union[str, float, int]]): A list of values representing a single row of data to be appended to the CSV file.
        delimiter (str, optional): The delimiter to use between columns in the CSV file. Default is tab ('\t').

    Returns:
        None
    """
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)
        writer.writerow(array)


def save_csv_line(file_name: str, df: pd.DataFrame, sep: str = '\t', dtypes: Optional[Dict[str, Union[str, np.dtype]]] = None) -> None:
    """
    Save a Pandas DataFrame to a CSV file, appending the DataFrame to the file if it exists.

    Args:
        file_name (str): The path to the CSV file.
        df (pd.DataFrame): The Pandas DataFrame to save.
        sep (str): The delimiter used in the CSV file.
        dtypes (Optional[Dict[str, Union[str, np.dtype]]]): A dictionary specifying data types for columns.

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


_column_data = None
def exists_in_csv(csv_file: str, column: str, number: Union[int, float, str], sep: str = '\t') -> bool:
    """
    Checks if a number exists in a specific column of a CSV file.

    Args:
        csv_file (str): The path to the CSV file.
        column (str): The column name to search.
        number (Union[int, float, str]): The value to search for.
        sep (str, optional): The delimiter used in the CSV file. Default is tab ('\t').

    Returns:
        bool: True if the number exists in the column, False otherwise.
    """
    try:
        global _column_data
        if _column_data is None:
            _column_data = read_csv(csv_file, sep=sep, col=column, to_np=True)
        return np.isin(number, _column_data)
    except IOError:
        return False


def exists_in_csv_old(csv_file: str, column: str, number: Union[int, float, str], sep: str = '\t') -> bool:
    """
    Checks if a number exists in a specific column of a CSV file (older method).

    Args:
        csv_file (str): The path to the CSV file.
        column (str): The column name to search.
        number (Union[int, float, str]): The value to search for.
        sep (str, optional): The delimiter used in the CSV file. Default is tab ('\t').

    Returns:
        bool: True if the number exists in the column, False otherwise.
    """
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=sep)
            for row in reader:
                if row[column] == str(number):
                    return True
    except IOError:
        return False


def pd_flatten(data: List[Union[str, float]], factor: float = 1) -> List[float]:
    """
    Flatten a list of data by splitting string elements representing lists 
    and converting them into floats. Divides each float by a given factor.

    Parameters
    ----------
    data : list of str or float
        The list of data to flatten. Some elements may be strings representing lists.
    factor : float, optional
        A factor to divide each element by (default is 1).

    Returns
    -------
    list of float
        A flattened list of floats.
    """
    tmp = []
    for x in data:
        try:
            tmp.extend(x[1:-1].split(','))
        except TypeError:
            tmp.append(x)
    return [float(x) / factor for x in tmp]


def str_to_array(s: str) -> np.ndarray:
    """
    Convert a string representation of a list (e.g., '[1, 2, 3]') into a NumPy array of floats.

    Parameters
    ----------
    s : str
        A string representation of a list.

    Returns
    -------
    np.ndarray
        A NumPy array of floats.
    """
    s = s.replace('[', '').replace(']', '')  # Remove square brackets
    return np.array([float(x) for x in s.split(',')])

def pdstr_to_arrays(df: 'pandas.DataFrame') -> np.ndarray:
    """
    Apply `str_to_array` to each element of a DataFrame and convert it to a NumPy array of arrays.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame where each element is a string representation of an array.

    Returns
    -------
    np.ndarray
        A NumPy array where each element is a NumPy array converted from the string representation.
    """
    return df.apply(str_to_array).to_numpy()

def allfiles(dirName: str = os.getcwd()) -> List[str]:
    """
    Get a list of all files in the directory tree starting at the given path.

    Parameters
    ----------
    dirName : str, optional
        The directory path to start the search from (default is the current working directory).

    Returns
    -------
    list of str
        A list of file paths.
    """
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    return listOfFiles
