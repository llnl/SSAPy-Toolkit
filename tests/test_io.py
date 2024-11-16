import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import pytest
from yeager_utils import (
    file_exists, exists, mkdir, mvdir, rmdir, rmfile, listdir, get_memory_usage,
    psave, pload, merge_dicts, append_h5, overwrite_h5, save_h5, read_h5, read_h5_all,
    combine_h5, h5_keys, h5_root_keys, h5_key_exists, makedf, read_csv, append_dict_to_csv,
    guess_csv_delimiter, save_csv, append_csv, pd_flatten, str_to_array, pdstr_to_arrays, allfiles
)

@pytest.fixture
def temp_dir():
    """Fixture for creating and cleaning up a temporary directory."""
    directory = tempfile.mkdtemp()
    yield directory
    shutil.rmtree(directory)


@pytest.fixture
def temp_file(temp_dir):
    """Fixture for creating and cleaning up a temporary file."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix='.txt')
    temp_file.close()
    yield temp_file.name
    os.remove(temp_file.name)


def test_file_exists(temp_file, temp_dir):
    assert file_exists(temp_file)
    assert not file_exists(os.path.join(temp_dir, "nonexistent.txt"))


def test_exists(temp_file, temp_dir):
    assert exists(temp_file)
    assert exists(temp_dir)
    assert not exists(os.path.join(temp_dir, "nonexistent_path"))


def test_mkdir(temp_dir):
    new_dir = os.path.join(temp_dir, "new_folder")
    mkdir(new_dir)
    assert os.path.isdir(new_dir)


def test_mvdir(temp_dir):
    source_dir = os.path.join(temp_dir, "source_folder")
    os.mkdir(source_dir)
    destination_dir = os.path.join(temp_dir, "destination_folder")
    mvdir(source_dir, destination_dir)
    assert os.path.isdir(destination_dir)
    assert not os.path.exists(source_dir)


def test_rmdir(temp_dir):
    directory_to_remove = os.path.join(temp_dir, "to_remove")
    os.mkdir(directory_to_remove)
    rmdir(directory_to_remove)
    assert not os.path.exists(directory_to_remove)


def test_rmfile(temp_dir):
    test_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
    test_file.close()
    rmfile(test_file.name)
    assert not os.path.exists(test_file.name)


def test_listdir(temp_file, temp_dir):
    # Test files only
    files = listdir(temp_dir, files_only=True)
    assert temp_file in files

    # Test exclude option
    excluded_files = listdir(temp_dir, exclude='.txt')
    assert temp_file not in excluded_files

    # Test sorted option
    sorted_files = listdir(temp_dir, sorted=True)
    assert sorted_files == sorted(sorted_files)


def test_get_memory_usage():
    memory_used = get_memory_usage()
    assert isinstance(memory_used, float)
    assert memory_used > 0


def test_psave_and_pload(temp_dir):
    data = {'key': 'value'}
    pickle_file = os.path.join(temp_dir, 'test.pkl')
    psave(pickle_file, data)
    loaded_data = pload(pickle_file)
    assert data == loaded_data


def test_pload_nonexistent_file():
    assert pload('nonexistent.pkl') is None


def test_merge_dicts(temp_dir):
    dict1 = {'a': 1, 'b': 2}
    dict2 = {'c': 3, 'd': 4}
    dict3 = {'e': 5}
    files = []

    for i, d in enumerate([dict1, dict2, dict3], start=1):
        file_path = os.path.join(temp_dir, f'dict_{i}.pkl')
        psave(file_path, d)
        files.append(file_path)

    merged_file = os.path.join(temp_dir, 'merged.pkl')
    merge_dicts(files, merged_file)
    merged_data = pload(merged_file)

    expected_data = {**dict1, **dict2, **dict3}
    assert merged_data == expected_data


def test_save_h5_and_read_h5(temp_dir):
    data = np.array([1, 2, 3])
    h5_file = os.path.join(temp_dir, 'test.h5')
    save_h5(h5_file, 'data', data)
    loaded_data = read_h5(h5_file, 'data')
    np.testing.assert_array_equal(data, loaded_data)


def test_append_h5(temp_dir):
    h5_file = os.path.join(temp_dir, 'test.h5')
    save_h5(h5_file, 'data', [1, 2, 3])
    append_h5(h5_file, 'data', [4, 5])
    loaded_data = read_h5(h5_file, 'data')
    np.testing.assert_array_equal(loaded_data, np.array([1, 2, 3, 4, 5]))


def test_overwrite_h5(temp_dir):
    h5_file = os.path.join(temp_dir, 'test.h5')
    save_h5(h5_file, 'data', [1, 2, 3])
    overwrite_h5(h5_file, 'data', [4, 5, 6])
    loaded_data = read_h5(h5_file, 'data')
    np.testing.assert_array_equal(loaded_data, np.array([4, 5, 6]))


def test_read_h5_all(temp_dir):
    h5_file = os.path.join(temp_dir, 'test.h5')
    data1 = np.array([1, 2, 3])
    data2 = np.array([4, 5, 6])
    save_h5(h5_file, 'group1/data1', data1)
    save_h5(h5_file, 'group2/data2', data2)
    loaded_data = read_h5_all(h5_file)
    np.testing.assert_array_equal(loaded_data['group1/data1'], data1)
    np.testing.assert_array_equal(loaded_data['group2/data2'], data2)


def test_combine_h5(temp_dir):
    file1 = os.path.join(temp_dir, 'file1.h5')
    file2 = os.path.join(temp_dir, 'file2.h5')
    h5_file = os.path.join(temp_dir, 'combined.h5')
    save_h5(file1, 'data1', [1, 2, 3])
    save_h5(file2, 'data2', [4, 5, 6])
    combine_h5(h5_file, [file1, file2])
    loaded_data1 = read_h5(h5_file, 'data1')
    loaded_data2 = read_h5(h5_file, 'data2')
    np.testing.assert_array_equal(loaded_data1, np.array([1, 2, 3]))
    np.testing.assert_array_equal(loaded_data2, np.array([4, 5, 6]))


def test_makedf():
    list_data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    df = makedf(list_data)
    assert df.shape == (2, 2)  # 2 rows, 2 columns


def test_read_csv(temp_dir):
    data = 'a,b\n1,2\n3,4'
    csv_path = os.path.join(temp_dir, 'test.csv')
    with open(csv_path, 'w') as f:
        f.write(data)

    df = read_csv(csv_path)
    assert df.shape == (2, 2)


def test_save_csv(temp_dir):
    data = {'a': [1, 2], 'b': [3, 4]}
    df = pd.DataFrame(data)
    csv_path = os.path.join(temp_dir, 'test_save.csv')
    save_csv(csv_path, df)
    assert os.path.exists(csv_path)


def test_append_dict_to_csv(temp_dir):
    data = {'a': [1, 2], 'b': [3, 4]}
    df = pd.DataFrame(data)
    csv_path = os.path.join(temp_dir, 'append_test.csv')
    save_csv(csv_path, df)

    new_data = {'a': [5, 6], 'b': [7, 8]}
    new_df = pd.DataFrame(new_data)
    append_dict_to_csv(csv_path, new_df)
    new_df_read = pd.read_csv(csv_path)
    assert new_df_read.shape[0] == 4


if __name__ == "__main__":
    import os
    import pytest

    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the script's name
    script_name = os.path.basename(__file__)
    
    # Construct the path dynamically
    test_dir = os.path.join(current_dir, script_name)
    
    # Run pytest
    pytest.main([test_dir])