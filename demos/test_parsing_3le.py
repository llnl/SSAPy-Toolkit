#!/usr/bin/env python3
from yeager_utils import yudata, read_3le, read_3le_by_bit, pprint

data_file = yudata("full_catalog_3le.txt")
print(f"DATA: {data_file}")

data = read_3le(data_file, verbose=True)
pprint(data)

# data = read_3le_by_bit(data_file)
# pprint(data)