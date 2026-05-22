#!/usr/bin/env python3

import os
import datetime as dt
import numpy as np

from ssapy_toolkit.io.dict_to_from_hdf5 import save_dict_to_hdf5, load_dict_from_hdf5  # [34]

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
        "now": dt.datetime.now(),
        "today": dt.date.today(),
        "time_only": dt.time(12, 34, 56, 789),
        "nested_lists": [[1, 2, 3], [4, 5], [], [[10, 11], [12]]],
        "mixed": {
            "list_of_lists": [[0, 1], [2, 3, 4]],
            "list_of_scalars": [True, False, 1.23],
        },
    }

    if HAS_ASTROPY:
        data["astro_time_scalar"] = AstroTime("2024-01-01T00:00:00", scale="utc")
        data["astro_time_array"] = AstroTime(["2024-01-01T00:00:00", "2024-01-02T12:00:00"], scale="utc")

    return data


def main():
    filename = "demo_test.h5"

    if os.path.exists(filename):
        os.remove(filename)

    original = build_test_dict()
    save_dict_to_hdf5(filename, original, mode="w")
    loaded = load_dict_from_hdf5(filename)

    assert loaded["nested_lists"] == original["nested_lists"]
    assert loaded["mixed"]["list_of_lists"] == original["mixed"]["list_of_lists"]
    assert np.array_equal(loaded["simple_array"], original["simple_array"])
    assert loaded["today"] == original["today"]
    assert loaded["now"] == original["now"]
    assert loaded["time_only"] == original["time_only"]

    if HAS_ASTROPY:
        o_scalar = original["astro_time_scalar"]
        l_scalar = loaded["astro_time_scalar"]
        assert np.allclose(o_scalar.mjd, l_scalar.mjd)
        assert o_scalar.scale == l_scalar.scale

        o_arr = original["astro_time_array"]
        l_arr = loaded["astro_time_array"]
        assert np.allclose(o_arr.mjd, l_arr.mjd)
        assert o_arr.scale == l_arr.scale

    if os.path.exists(filename):
        os.remove(filename)

    return {"original": original, "loaded": loaded}


if __name__ == "__main__":
    main()