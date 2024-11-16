import pytest
import numpy as np
from unittest.mock import patch
from time import sleep
from timeit import default_timer as timer
from yeager_utils.utils import *


# Test for mem_usage
def test_mem_usage():
    with patch("psutil.virtual_memory") as mock_virtual_memory:
        mock_virtual_memory.return_value = (1000000000, 500000000, 500000000, 200000000, 50, 100, 50, 10)
        mem_usage()  # This should print the memory usage
        assert mock_virtual_memory.called


# Test for timenow
def test_timenow():
    from datetime import datetime
    current_time = timenow()
    now = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    assert current_time == now


# Test for pd_flatten
def test_pd_flatten():
    data = ["1","2","3", "4","5","6", "7","8","9"]
    result = pd_flatten(data)
    expected_result = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert result == expected_result


# Test for str_to_array
def test_str_to_array():
    s = "[1.2, 3.4, 5.6]"
    result = str_to_array(s)
    expected_result = np.array([1.2, 3.4, 5.6])
    assert np.array_equal(result, expected_result)


# Test for random_arr
def test_random_arr():
    arr = random_arr(low=0, high=10, size=(2, 3))
    assert arr.shape == (2, 3)
    assert np.all(arr >= 0) and np.all(arr <= 10)


# Test for find_indices
def test_find_indices():
    lst = [1, 2, 3, 4, 5]
    condition = lambda x: x > 3
    indices = find_indices(lst, condition)
    assert indices == [3, 4]


# Test for nan_array
def test_nan_array():
    result = nan_array(3)
    expected_result = np.array([np.nan, np.nan, np.nan])
    print(result, expected_result)
    assert np.all(np.isnan(result) == np.isnan(expected_result))


# Test for remove_np_nans
def test_remove_np_nans():
    arr = np.array([1, 2, np.nan, 4, np.nan])
    result = remove_np_nans(arr)
    expected_result = np.array([1, 2, 4])
    assert np.array_equal(result, expected_result)


# Test for remove_zeros
def test_remove_zeros():
    arr = np.array([[0, 0, 0], [1, 2, 3], [0, 0, 0]])
    result = remove_zeros(arr, axis=1)
    expected_result = np.array([[1, 2, 3]])
    assert np.array_equal(result, expected_result)


# Test for nby3shape
def test_nby3shape():
    arr = np.array([1, 2, 3])
    result = nby3shape(arr)
    assert np.array_equal(result, np.array([[1, 2, 3]]))

    arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    result = nby3shape(arr_2d)
    assert np.array_equal(result, arr_2d)


# Test for eformat
def test_eformat():
    f = 12345.6789
    result = eformat(f, 2, 3)
    expected_result = "1.23e+04"
    assert result == expected_result


# Test for extractNum
def test_extractNum():
    s = "abc123def"
    result = extractNum(s)
    expected_result = 123
    assert result == expected_result


# Test for sortbynum
def test_sortbynum():
    files = ["file12", "file2", "file1"]
    result = sortbynum(files)
    expected_result = ["file1", "file2", "file12"]
    assert result == expected_result


# Test for issorted
def test_issorted():
    test_list_sorted = [1, 2, 3, 4, 5]
    test_list_unsorted = [1, 3, 2, 4, 5]

    assert issorted(test_list_sorted) == True
    assert issorted(test_list_unsorted) == False


# Test for byte2str
def test_byte2str():
    byte_string = [b'hello', b'world']
    result = byte2str(byte_string)
    expected_result = ['hello', 'world']
    assert result == expected_result


# Test for flatten
def test_flatten():
    nested_list = [[1, 2], [3, 4], [5, 6]]
    result = flatten(nested_list)
    expected_result = [1, 2, 3, 4, 5, 6]
    assert result == expected_result


# Test for ETA
def test_ETA():
    start_loop_time = timer()
    sleep(1)  # Sleep for 1 second to simulate a loop
    idx = 1
    total_num = 10
    result = ETA(idx, total_num, start_loop_time)
    assert result is None  # No specific result is returned, just printing


# Test for elapsed_time
def test_elapsed_time():
    start_time = timer()
    sleep(1)
    result = elapsed_time(start_time)
    assert result is None  # No specific result is returned, just printing


# Test for size
def test_size():
    arr = np.array([1, 2, 3])
    result = size(arr)
    assert result == 3

    arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    result = size(arr_2d, axis=0)
    assert result == 2


# Test for sortbylist
def test_sortbylist():
    X = [1, 2, 3, 4]
    Y = [40, 10, 30, 20]
    result = sortbylist(X, Y)
    expected_result = [2, 4, 3, 1]
    assert result == expected_result


# Test for find_nearest
def test_find_nearest():
    arr = np.array([1, 2, 3, 4, 5])
    idx, value = find_nearest(arr, 3)
    assert idx == 2
    assert value == 0


# Test for sample
def test_sample():
    seq = [1, 2, 3, 4, 5]
    result = sample(seq, 3, replacement=False)
    assert len(result) == 3
    assert all(item in seq for item in result)


# Test for rand_num
def test_rand_num():
    result = rand_num(0, 10)
    assert 0 <= result <= 10


# Test for isint
def test_isint():
    assert isint(5) == True
    assert isint(5.5) == False


# Test for isfloat
def test_isfloat():
    assert isfloat(5.5) == True
    assert isfloat(5) == False


# Test for isstr
def test_isstr():
    assert isstr("hello") == True
    assert isstr(5) == False


# Test for shuffle
def test_shuffle():
    seq = [1, 2, 3, 4, 5]
    shuffle(seq)
    assert len(seq) == 5


# Test for divby0
def test_divby0():
    assert np.isnan(divby0(5, 0))  # Updated to check for np.nan
    assert divby0(5, 2) == 2.5


# Test for kde
def test_kde():
    data = np.random.normal(0, 1, size=1000)
    result = kde(data).evaluate(data)  # Use .evaluate() for kernel density estimation
    assert len(result) == len(data)


# Test for eval
def test_eval():
    data = {'a': 1, 'b': 2}
    result = eval('a + b', data)
    assert result == 3


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
