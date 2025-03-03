import h5py
import numpy as np
from .io.csv import save_csv_line
from .io.hdf5 import read_h5
from .io.io_utils import rmfile
from .time import Time, get_times
from typing import List
from astropy import units as u

main_dataset = "/p/lustre2/cislunar/data_six_year_1.0GEO_to_18.2GEO/"
copy_dataset = "/p/lustre2/cislunar/data_six_year_1.0GEO_to_18.2GEO_copy/"

cislunar_csv_data = "/p/lustre2/cislunar/data_six_year_1.0GEO_to_18.2GEO_descriptive_data.csv"
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


def index_from_cislunar_filename(filename: str) -> int:
    """
    Extract the index from a cislunar dataset filename.

    Parameters
    ----------
    filename : str
        The full filename path for the dataset.

    Returns
    -------
    int
        The extracted index corresponding to the orbital dataset.
    """
    import re

    # Match the orbital ID pattern in the filename
    match = re.search(r"orb_id_(\d+)_to_(\d+)\.h5", filename)
    if not match:
        raise ValueError("Filename does not match the expected format.")

    lower_bound = int(match.group(1))
    return lower_bound // 50


def cislunar_orb_ids_from_h5(h5_file: str) -> np.ndarray:
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

    lower = int(index_from_cislunar_filename(h5_file) * 50)
    return np.arange(lower, lower + 50)


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
sim_time_step_hours = 1 * u.Unit('hour').to(u.Unit('hour'))


def get_ctimes() -> np.ndarray:
    """
    Generate time array for the simulation.

    Returns
    -------
    np.ndarray
        A numpy array of times from the start time.
    """
    times = get_times(duration=(6, 'year'), freq=(1, 'hour'), t0=sim_start_time)
    return times


def get_csv_file(h5_file):
    csv_file = f"{'.'.join(h5_file.split('.')[:-1])}.csv"
    return csv_file


def build_csv_from_h5(h5_file):
    # REBUILD CSVS
    def save_to_csv(csv_file, data):
        # BUILD CSV DATAFRAME FROM DATA VAR
        df = {
            'orb_id': [data['orb_id']],
            'ejection': [data['ejection']],
            'lifetime': [data['lifetime']],
            'period': [data['period']],
            'perigee': [data['perigee']],
            'apogee': [data['apogee']],
            'r0': [data['r_initial']],
            'v0': [data['r_initial']],
            'pm_ra_min': [np.min(data['pm_ra'])],
            'pm_dec_min': [np.min(data['pm_dec'])],
            'pm_ra_max': [np.max(data['pm_ra'])],
            'pm_dec_max': [np.max(data['pm_dec'])],
            'M_v_min': [np.min(data['M_v'])],
            'M_v_max': [np.max(data['M_v'])],
            'a': [data['semi_major_axis'][0]],
            'e': [data['eccentricity'][0]],
            'i': [data['inclination'][0]],
            'tl': [data['true_longitude'][0]],
            'pa': [data['argument_of_periapsis'][0]],
            'raan': [data['longitude_of_ascending_node'][0]],
            'ta': [data['true_anomaly'][0]],
            'vmin': [data['vmin']],
            'vmax': [data['vmax']],
            'r_vmin': [data['r_vmin']],
            'r_vmax': [data['r_vmax']],
            'std_divergence': [data['std_divergence']],
            'median_divergence': [data['median_divergence']],
            'mean_divergence': [data['mean_divergence']],
            'max_divergence': [data['max_divergence']],
            'threebody_std_divergence': [data['threebody_std_divergence']],
            'threebody_median_divergence': [data['threebody_median_divergence']],
            'threebody_mean_divergence': [data['threebody_mean_divergence']],
            'threebody_max_divergence': [data['threebody_max_divergence']],
        }
        save_csv_line(csv_file, df)
        return

    csv_file = get_csv_file(h5_file)
    orb_ids = cislunar_orb_ids_from_h5(h5_file)
    rmfile(csv_file)
    for orb_id in orb_ids:
        data = {
            'orb_id': orb_id,
            'ejection': read_h5(h5_file, f"{orb_id}/ejection"),
            'lifetime': read_h5(h5_file, f"{orb_id}/lifetime"),
            'period': read_h5(h5_file, f"{orb_id}/period"),
            'perigee': read_h5(h5_file, f"{orb_id}/perigee"),
            'apogee': read_h5(h5_file, f"{orb_id}/apogee"),
            'r_initial': read_h5(h5_file, f"{orb_id}/r_initial"),
            'v_initial': read_h5(h5_file, f"{orb_id}/v_initial"),
            'pm_ra': read_h5(h5_file, f"{orb_id}/pm_ra"),
            'pm_dec': read_h5(h5_file, f"{orb_id}/pm_dec"),
            'M_v': read_h5(h5_file, f"{orb_id}/M_v"),
            'semi_major_axis': read_h5(h5_file, f"{orb_id}/semi_major_axis"),
            'eccentricity': read_h5(h5_file, f"{orb_id}/eccentricity"),
            'inclination': read_h5(h5_file, f"{orb_id}/inclination"),
            'true_longitude': read_h5(h5_file, f"{orb_id}/true_longitude"),
            'argument_of_periapsis': read_h5(h5_file, f"{orb_id}/argument_of_periapsis"),
            'longitude_of_ascending_node': read_h5(h5_file, f"{orb_id}/longitude_of_ascending_node"),
            'true_anomaly': read_h5(h5_file, f"{orb_id}/true_anomaly"),
            'vmin': read_h5(h5_file, f"{orb_id}/vmin"),
            'vmax': read_h5(h5_file, f"{orb_id}/vmax"),
            'r_vmin': read_h5(h5_file, f"{orb_id}/r_vmin"),
            'r_vmax': read_h5(h5_file, f"{orb_id}/r_vmax"),
            'std_divergence': read_h5(h5_file, f"{orb_id}/std_divergence"),
            'median_divergence': read_h5(h5_file, f"{orb_id}/median_divergence"),
            'mean_divergence': read_h5(h5_file, f"{orb_id}/mean_divergence"),
            'max_divergence': read_h5(h5_file, f"{orb_id}/max_divergence"),
            'threebody_std_divergence': read_h5(h5_file, f"{orb_id}/threebody_std_divergence"),
            'threebody_median_divergence': read_h5(h5_file, f"{orb_id}/threebody_median_divergence"),
            'threebody_mean_divergence': read_h5(h5_file, f"{orb_id}/threebody_mean_divergence"),
            'threebody_max_divergence': read_h5(h5_file, f"{orb_id}/threebody_max_divergence")
        }

        save_to_csv(csv_file, data)
    return


sim_lowest_alt = 35_786_035.36450565
sim_max_distance = 768_798_000


if __name__ == "__main__":
    dataset = main_dataset
    numpy_name = f"/p/lustre2/cislunar/bad_orbits_{dataset}.npy"

    bad_orbits: List[int] = []
    print("Variable keys:", cislunar_keys())
    for orb_id in range(1_000_000):
        if get_cdata(orb_id, 'lifetime', dataset) is None:
            bad_orbits.append(orb_id)

    print("Bad orbits")
    np.save(numpy_name, bad_orbits)
