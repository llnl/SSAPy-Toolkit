#!/usr/bin/env python3

from pathlib import Path

from ssapy_toolkit.IO.yudata import yudata
from ssapy_toolkit.IO.read_3le import read_3le
from ssapy_toolkit.IO.read_3le_by_bit import read_3le_by_bit
from ssapy_toolkit.IO.tle_iter_pairs import tle_iter_pairs
from ssapy_toolkit.IO.tle_prop_to_time import tle_prop_to_time
from ssapy_toolkit.IO.pprint_utils import pprint


def main(verbose=False, fast=False):
    tle_path = yudata("full_catalog_3le.txt")

    if not Path(tle_path).exists():
        print(f"Skipping demo_parsing_3le: missing data file {tle_path}")
        return {
            "data": None,
            "skipped": True,
            "reason": "missing_data_file",
            "tle_path": tle_path,
        }

    print(f"DATA: {tle_path}")

    data = read_3le(tle_path, verbose=verbose)
    if verbose:
        pprint(data)

    pair_iter = tle_iter_pairs(tle_path)
    first_pairs = []
    for idx, triple in enumerate(pair_iter):
        first_pairs.append(triple)
        if fast and idx >= 2:
            break

    from ssapy.orbit import Orbit

    orbs = []
    for name, line1, line2 in first_pairs:
        orb = Orbit.fromTLETuple((line1, line2))
        orbs.append(orb)

    orbits_at_t, names, R, V = tle_prop_to_time(
        "2025-01-01T00:00:00",
        tle_path,
        validate_checksum=False,
        truncate=False,
        return_arrays=True,
    )

    return {
        "data": data,
        "sample_pairs": first_pairs,
        "sample_orbits": orbs,
        "propagated_orbits": orbits_at_t,
        "skipped": False,
        "tle_path": tle_path,
    }


if __name__ == "__main__":
    main(verbose=True, fast=False)