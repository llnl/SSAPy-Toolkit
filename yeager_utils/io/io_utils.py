# flake8: noqa: E501
import numpy as np
import os
from glob import glob
import shutil
from ..utils import sortbynum
from typing import List, Optional, Union


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
            os.makedirs(pathname, exist_ok=True)
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


def pdstr_to_arrays(df: 'DataFrame') -> np.ndarray:
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
