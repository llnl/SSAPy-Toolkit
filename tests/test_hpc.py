import numpy as np
import pytest

# Import the functions you want to test
from yeager_utils import get_unique_id, get_unique_id2, distribute_array_no_mpi

def test_get_unique_id():
    """
    Test the get_unique_id function for correctness.
    """
    # Test case 1: Test with run_number = 0, rank = 0, and cpus_per_node = 4
    result = get_unique_id(0, 0, 4)
    assert result == 0, f"Expected 0, but got {result}"

    # Test case 2: Test with run_number = 1, rank = 1, and cpus_per_node = 4
    result = get_unique_id(1, 1, 4)
    assert result == 5, f"Expected 5, but got {result}"

    # Test case 3: Test with run_number = 2, rank = 3, and cpus_per_node = 4
    result = get_unique_id(3, 2, 4)
    assert result == 11, f"Expected 11, but got {result}"

def test_get_unique_id2():
    """
    Test the get_unique_id2 function for correctness.
    """
    # Test case 1: Test with run_number = 0, rank = 0, and cpus_per_node = 4
    result = get_unique_id2(0, 0, 4)
    assert result == 0, f"Expected 0, but got {result}"

    # Test case 2: Test with run_number = 1, rank = 1, and cpus_per_node = 4
    result = get_unique_id2(1, 1, 4)
    assert result == 5, f"Expected 5, but got {result}"

    # Test case 3: Test with run_number = 2, rank = 3, and cpus_per_node = 4
    result = get_unique_id2(3, 2, 4)
    assert result == 11, f"Expected 11, but got {result}"

def test_distribute_array_no_mpi():
    """
    Test the distribute_array_no_mpi function for correctness.
    """
    array_size = 1000
    total_jobs = 10
    array = np.arange(array_size)

    # Test distribution for unique_ids
    for unique_id in range(total_jobs):
        start_idx, end_idx = distribute_array_no_mpi(unique_id, total_jobs, array_size)
        print(f"Unique ID {unique_id}: Start index: {start_idx}, End index: {end_idx}")
        # Ensure indices are within bounds
        assert start_idx >= 0 and start_idx < array_size, f"Start index {start_idx} out of bounds"
        assert end_idx > start_idx and end_idx <= array_size, f"End index {end_idx} out of bounds"
        
        # Check if the slice is correct
        array_slice = array[start_idx:end_idx]
        assert len(array_slice) == (end_idx - start_idx), f"Slice length mismatch for unique_id {unique_id}"

def test_distribute_array_no_mpi_edge_cases():
    """
    Test edge cases for distribute_array_no_mpi.
    """
    array_size = 1000
    total_jobs = 10
    array = np.arange(array_size)

    # Test case with array size smaller than total jobs
    small_array_size = 5
    small_array = np.arange(small_array_size)
    for unique_id in range(total_jobs):
        start_idx, end_idx = distribute_array_no_mpi(unique_id, total_jobs, small_array_size)
        if start_idx is None and end_idx is None:
            # No assignment for this unique_id
            print(f"Unique ID {unique_id}: No data assigned (out of bounds)")
            continue
        print(f"Unique ID {unique_id}: Start index: {start_idx}, End index: {end_idx}")
        assert start_idx >= 0 and start_idx < small_array_size, f"Start index {start_idx} out of bounds"
        assert end_idx > start_idx and end_idx <= small_array_size, f"End index {end_idx} out of bounds"


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
