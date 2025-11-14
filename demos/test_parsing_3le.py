#!/usr/bin/env python3
from yeager_utils import yudata, read_3le, read_3le_by_bit, tle_iter_pairs, tle_prop_to_time, pprint

tle_path = yudata("full_catalog_3le.txt")
print(f"DATA: {tle_path}")

data = read_3le(tle_path, verbose=True)
pprint(data)

# data = read_3le_by_bit(data_file)
# pprint(data)

print(data.columns)
print(data.head())


for name, line1, line2 in tle_iter_pairs(tle_path):
    print(f"line1: {line1}")
    print(f"line2: {line2}")

#BUILD SSAPY ORBIT FROM TLE
from ssapy.orbit import Orbit

orbs = []
for name, line1, line2 in tle_iter_pairs(tle_path):
    orb = Orbit.fromTLETuple((line1, line2))
    orbs.append(orb)

print("\n", orbs[:10], "\n")

orbits_at_t, names, R, V = tle_prop_to_time(
    "2025-01-01T00:00:00", tle_path,
    validate_checksum=False,
    truncate=False,
    return_arrays=True
)

print(orbits_at_t[:10])