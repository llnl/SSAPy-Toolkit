#!/usr/bin/env python3

import os
import datetime as dt

import numpy as np

from ssapy_toolkit.IO.dict_to_from_hdf5 import save_dict_to_hdf5, load_dict_from_hdf5

try:
    from astropy.time import Time as AstroTime
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


def build_test_dict():
    data = {
        "simple_array": np.arange(10),
        "scalar_int": 42,
        "scalar_float": 3.14159,
        "string": "hello, HDF5",
        "bytes_val": b"\x00\x01\x02",

        # Datetime types
        "now": dt.datetime.now(),
        "today": dt.date.today(),
        "time_only": dt.time(12, 34, 56, 789),

        # Lists of lists (includes an empty inner list)
        "nested_lists": [
            [1, 2, 3],
            [4, 5],
            [],
            [[10, 11], [12]],
        ],

        # Mixed nested structures
        "mixed": {
            "list_of_lists": [[0, 1], [2, 3, 4]],
            "list_of_scalars": [True, False, 1.23],
        },
    }

    if HAS_ASTROPY:
        data["astro_time_scalar"] = AstroTime("2024-01-01T00:00:00", scale="utc")
        data["astro_time_array"] = AstroTime(
            ["2024-01-01T00:00:00", "2024-01-02T12:00:00"],
            scale="utc",
        )

    return data


def main():
    filename = "demo_test.h5"

    # Remove old file so we don't re-use outdated structure
    if os.path.exists(filename):
        os.remove(filename)

    print("Building test dictionary...")
    original = build_test_dict()

    print(f"Saving to {filename!r}...")
    save_dict_to_hdf5(filename, original, mode="w")

    print(f"Loading from {filename!r}...")
    loaded = load_dict_from_hdf5(filename)

    # 1. Nested lists
    print("Checking nested_lists round-trip...")
    print("  original:", original["nested_lists"])
    print("  loaded  :", loaded["nested_lists"])
    assert loaded["nested_lists"] == original["nested_lists"], \
        f"nested_lists mismatch:\noriginal={original['nested_lists']}\nloaded={loaded['nested_lists']}"

    # 2. Mixed nested structures
    print("Checking mixed.list_of_lists round-trip...")
    assert loaded["mixed"]["list_of_lists"] == original["mixed"]["list_of_lists"], \
        "mixed.list_of_lists mismatch"

    # 3. Numpy array content
    print("Checking simple_array content...")
    assert np.array_equal(loaded["simple_array"], original["simple_array"]), \
        "simple_array mismatch"

    # 4. Datetimes
    print("Checking datetime fields...")
    assert loaded["today"] == original["today"], "date mismatch"
    assert loaded["now"] == original["now"], "datetime mismatch"
    assert loaded["time_only"] == original["time_only"], "time mismatch"

    # 5. Astropy Time (if available)
    if HAS_ASTROPY:
        print("Checking astropy.time.Time scalar...")
        o_scalar = original["astro_time_scalar"]
        l_scalar = loaded["astro_time_scalar"]
        assert np.allclose(o_scalar.mjd, l_scalar.mjd), "AstroTime scalar mjd mismatch"
        assert o_scalar.scale == l_scalar.scale, "AstroTime scalar scale mismatch"

        print("Checking astropy.time.Time array...")
        o_arr = original["astro_time_array"]
        l_arr = loaded["astro_time_array"]
        assert np.allclose(o_arr.mjd, l_arr.mjd), "AstroTime array mjd mismatch"
        assert o_arr.scale == l_arr.scale, "AstroTime array scale mismatch"

    print("All checks passed.")

    # Clean up test file
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Test file {os.path.abspath(filename)} removed.")
    else:
        print("Test file was not found to remove (perhaps already deleted).")


if __name__ == "__main__":
    main()
