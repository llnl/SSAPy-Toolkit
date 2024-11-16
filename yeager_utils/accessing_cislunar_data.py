import h5py
import numpy as np
from .constants import MOON_RADIUS
from .io import read_h5
from .time import Time, get_times
from typing import List, Dict


main_dataset = "/p/lustre2/cislunar/data_six_year_1.0GEO_to_18.2GEO/"
copy_dataset = "/p/lustre2/cislunar/data_six_year_1.0GEO_to_18.2GEO_copy/"

cislunar_h5_keys = ['ejection', 'lifetime', 'period', 'perigee', 'apogee', 
    'r', 'v', 'ra', 'dec', 'range', 'pm_ra', 'pm_dec', 'pm_r', 'M_v', 
    'semi_major_axis', 'eccentricity', 'inclination', 
    'true_longitude', 'argument_of_periapsis', 'longitude_of_ascending_node', 'true_anomaly', 
    'r_initial', 'v_initial', 'r_earth_min', 'r_earth_max', 'vmin', 'vmax', 'r_vmin', 'r_vmax', 'r_moon_min',
    'threebody_r', 'threebody_v', 
    'nearby_orbits_r', 'nearby_orbits_v', 'threebody_nearby_orbits_r', 'threebody_nearby_orbits_v',
    'covariances', 'threebody_covariances', 
    'std_divergence', 'median_divergence', 'mean_divergence', 'max_divergence',
    'threebody_std_divergence', 'threebody_median_divergence', 'threebody_mean_divergence', 'threebody_max_divergence'
]


def get_chance_radius(v: np.ndarray) -> float:
    """
    Calculate the chance radius based on the velocity vector.

    Parameters
    ----------
    v : np.ndarray
        The velocity vector of the object at the final time step.

    Returns
    -------
    float
        The calculated chance radius in arcseconds.
    """
    return np.linalg.norm(v[-1]) * 3600 / MOON_RADIUS * 4 + 2


def cislunar_orb_ids_from_index(index: int) -> np.ndarray:
    """
    Generate a range of orbital IDs based on an index.

    Parameters
    ----------
    index : int
        The index to generate orbital IDs.

    Returns
    -------
    np.ndarray
        A numpy array of orbital IDs.
    """
    lower = int(index) * 50
    return np.arange(lower, lower + 50)


def cislunar_filename_from_index(index: int, dataset: str) -> str:
    """
    Generate the filename of a dataset based on the orbital index.

    Parameters
    ----------
    index : int
        The index to determine the dataset file.
    dataset : str
        The base dataset path.

    Returns
    -------
    str
        The full filename path for the dataset.
    """
    lower = int(index) * 50
    return f"{dataset}/orb_id_{lower}_to_{lower + 49}.h5"


def cislunar_filename(orb_id: int, dataset: str = main_dataset) -> str:
    """
    Generate the filename for an orbital ID.

    Parameters
    ----------
    orb_id : int
        The orbital ID.
    dataset : str, optional
        The base dataset path (default is `main_dataset`).

    Returns
    -------
    str
        The full filename path for the orbital ID.
    """
    lower = int(orb_id // 50) * 50
    return f"{dataset}/orb_id_{lower}_to_{lower + 49}.h5"


def get_cdata(orb_id: int, key: str, dataset: str) -> np.ndarray:
    """
    Retrieve data from an HDF5 file for a specific orbital ID and key.

    Parameters
    ----------
    orb_id : int
        The orbital ID.
    key : str
        The key to retrieve from the HDF5 file.
    dataset : str
        The base dataset path.

    Returns
    -------
    np.ndarray
        The requested data from the HDF5 file.
    """
    return read_h5(filename=f"{cislunar_filename(orb_id, dataset)}", pathname=f"{int(orb_id)}/{key}")


def cislunar_keys(unique_id: int = 0, dataset: str = main_dataset) -> List[str]:
    """
    Get all the unique keys from the HDF5 dataset.

    Parameters
    ----------
    unique_id : int, optional
        The unique ID for selection (default is 0).
    dataset : str, optional
        The base dataset path (default is `main_dataset`).

    Returns
    -------
    List[str]
        A list of unique keys in the dataset.
    """
    def h5_keys(file_path: str) -> List[str]:
        keys_list = []
        with h5py.File(file_path, 'r') as file:
            # Recursive function to traverse the HDF5 file and collect keys
            def traverse(group, path=''):
                for key, item in group.items():
                    new_path = f"{path}/{key}" if path else key
                    if isinstance(item, h5py.Group):
                        traverse(item, path=new_path)
                    else:
                        keys_list.append(new_path)
            traverse(file)
        return keys_list

    keys = h5_keys(f"{cislunar_filename_from_index(unique_id, dataset)}.h5")
    varkeys = sorted(list(set([key.split('/')[-1] for key in keys])))
    return varkeys


sim_start_time = Time("1980-01-01", scale='utc')


def get_ctimes() -> np.ndarray:
    """
    Generate time array for the simulation.

    Returns
    -------
    np.ndarray
        A numpy array of times from the start time.
    """
    times = get_times(duration=(6, 'year'), freq=(1, 'hour'), t=sim_start_time)
    return times


if __name__ == "__main__":
    dataset = main_dataset
    numpy_name = f"/p/lustre2/cislunar/bad_orbits_{dataset}.npy"
    
    bad_orbits: List[int] = []
    print("Variable keys:", cislunar_keys())
    for orb_id in range(1_000_000):
        if get_cdata(orb_id, 'lifetime', dataset) is None:
            bad_orbits.append(orb_id)
    
    print(f"Bad orbits")
    np.save(numpy_name, bad_orbits)
