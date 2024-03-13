# %%
# flake8: noqa: E501

import h5py
import numpy as np
from .io import read_h5
from .time import Time, get_times

chunk_boundaries_cache = None


def cislunar_get_chunk_boundaries(arr_size=1_000_000, total_splits=2800):
    arr = np.arange(arr_size)
    chunk_boundaries = []
    for unique_id in range(total_splits):
        chunk_boundaries.append(arr[int(unique_id / total_splits * arr_size):int((unique_id + 1) / total_splits * arr_size)][0])
    return chunk_boundaries


def cislunar_find_chunk(orb_id, total_splits=2800, arr_size=1_000_000):
    global chunk_boundaries_cache
    if chunk_boundaries_cache is None:
        chunk_boundaries_cache = cislunar_get_chunk_boundaries(arr_size=arr_size, total_splits=total_splits)

    chunk_index = np.searchsorted(chunk_boundaries_cache, orb_id, side='right') - 1
    return int(chunk_index)


def cislunar_keys(unique_id=0, dataset="data_1_year_1.0GEO_to_18.2GEO"):
    def h5_keys(file_path):
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
    keys = h5_keys(f"/p/lustre2/cislunar/{dataset}_h5s/rv_unique_id_{unique_id}.h5")
    varkeys = sorted(list(set([key.split('/')[-1] for key in keys])))
    return varkeys


def cislunar_filename(orb_id, dataset="data_1_year_1.0GEO_to_18.2GEO"):
    return f"/p/lustre2/cislunar/{dataset}_h5s/rv_unique_id_{cislunar_find_chunk(orb_id)}.h5"


def get_cdata(orb_id, key, dataset="data_1_year_1.0GEO_to_18.2GEO"):
    return read_h5(filename=cislunar_filename(orb_id, dataset), pathname=f"{int(orb_id)}/{key}")


def get_ctimes():
    start_time = Time("1980-01-01", scale='utc')
    times = get_times(duration=(1, 'year'), freq=(1, 'hour'), t=start_time)
    return times


if __name__ == "__main__":
    dataset = "data_1_year_1.0GEO_to_18.2GEO"
    # for orb_id in range(1_000_000):
    #     print(f"\nOrbit {orb_id}, unique_id: {find_chunk(orb_id)}")

    numpy_name = f"/p/lustre2/cislunar/bad_orbits_{dataset}.npy"
    # if not os.path.exists(numpy_name):
    bad_orbits = []
    print("Variable keys:", cislunar_keys())
    for orb_id in range(1_000_000):
        if get_cdata(orb_id, 'lifetime') is None:
            bad_orbits.append(orb_id)
    print(f"Bad orbits")
    np.save(numpy_name, bad_orbits)

# %%
