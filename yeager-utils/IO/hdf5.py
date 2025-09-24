import h5py
import numpy as np
from typing import Any, List, Optional
import os


def append_h5(filename: str, key: str, append_data: Any) -> None:
    """
    Append data to an existing key in an HDF5 file.

    Args:
        filename (str): The filename of the HDF5 file.
        key (str): The path to the key in the HDF5 file.
        append_data (Any): The data to be appended.

    Returns:
        None
    """
    try:
        with h5py.File(filename, "a") as f:
            if key in f:
                path_data_old = np.array(f.get(key))
                append_data = np.append(path_data_old, np.array(append_data))
                del f[key]
            f.create_dataset(key, data=np.array(append_data), maxshape=None)
    except FileNotFoundError:
        print(f"File not found: {filename}\nCreating new dataset: {filename}")
        save_h5(filename, key, append_data)
    except (ValueError, KeyError) as err:
        print(f"Error: {err}")


def overwrite_h5(filename: str, key: str, new_data: Any) -> None:
    """
    Overwrite data for a key in an HDF5 file.

    Args:
        filename (str): The filename of the HDF5 file.
        key (str): The path to the key in the HDF5 file.
        new_data (Any): The new data to overwrite the old data.

    Returns:
        None
    """
    try:
        with h5py.File(filename, "a") as f:
            f.create_dataset(key, data=new_data, maxshape=None)
    except (FileNotFoundError, ValueError, KeyError):
        try:
            with h5py.File(filename, 'r+') as f:
                del f[key]
        except (FileNotFoundError, ValueError, KeyError) as err:
            print(f'Error: {err}')
        try:
            with h5py.File(filename, "a") as f:
                f.create_dataset(key, data=new_data, maxshape=None)
        except (FileNotFoundError, ValueError, KeyError) as err:
            print(f'File: {filename}{key}, Error: {err}')


def save_h5(filename: str, key: str, data: Any) -> None:
    """
    Save data to an HDF5 file.

    Args:
        filename (str): The filename of the HDF5 file.
        key (str): The path to the data in the HDF5 file.
        data (Any): The data to be saved.

    Returns:
        None
    """
    try:
        with h5py.File(filename, "a") as f:
            f.create_dataset(key, data=data, maxshape=None)
            f.flush()
    except ValueError as err:
        print(f"Did not save, key: {key} exists in file: {filename}. {err}")
        return
    except (BlockingIOError, OSError) as err:
        print(f"\n{err}\nPath: {key}\nFile: {filename}\n")
        return


def read_h5(filename: str, key: str) -> Optional[np.ndarray]:
    """
    Load data from an HDF5 file.

    Args:
        filename (str): The filename of the HDF5 file.
        key (str): The path to the data in the HDF5 file.

    Returns:
        Optional[np.ndarray]: The data loaded from the HDF5 file, or None if the key does not exist.
    """
    try:
        with h5py.File(filename, 'r') as f:
            data = f.get(key)
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
        print(f"\n{err}\nPath: {key}\nFile: {filename}\n")
        raise


def read_h5_to_dict(file_path):
    def recursively_load(h5obj):
        data_dict = {}
        for key in h5obj.keys():
            item = h5obj[key]
            if isinstance(item, h5py.Group):
                data_dict[key] = recursively_load(item)
            elif isinstance(item, h5py.Dataset):
                data_dict[key] = item[()]  # Load dataset as a NumPy array or scalar
        return data_dict

    with h5py.File(file_path, 'r') as h5file:
        return recursively_load(h5file)
    

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
