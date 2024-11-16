import os
import pytest

if __name__ == "__main__":
    # Get the current directory (assumed to be the tests directory)
    tests_dir = os.path.dirname(os.path.abspath(__file__))

    # Run all tests in the current directory
    exit_code = pytest.main([tests_dir])

    # Exit with pytest's return code
    exit(exit_code)
