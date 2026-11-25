#!/usr/bin/env python3
#
# --- Checks the registered fields written to the diagnostic: they count the
# --- steps taken (1, 2 and 3 per step for the vector components), so in the
# --- last dump they are exact multiples of the step number, also after a
# --- restart.

import sys

import yt

dump = sys.argv[1]
ds = yt.load(dump)
ad = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
step = int(dump.rstrip("/")[-6:])  # e.g. diags/diag1000010

expected = {
    "step_count": step,
    "step_count_vecx": step,
    "step_count_vecy": 2 * step,
    "step_count_vecz": 3 * step,
}
for name, value in expected.items():
    data = ad["boxlib", name].v
    print(f"{name}: min = {data.min()}, max = {data.max()}, expected {value}")
    assert data.min() == data.max() == value
