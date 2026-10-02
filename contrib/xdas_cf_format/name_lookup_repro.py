"""
HDF5-only reproduction: getting the name of an object opened through an object
reference costs time proportional to the objects HDF5 visits before finding it.

h5netcdf resolves every variable's dimension names this way (DIMENSION_LIST
references followed by ``.name``), so opening a variable gets slower as the file
grows, depending on where the variable sits in the file.

    python name_lookup_repro.py
"""

import tempfile
import time
from pathlib import Path

import h5py
import numpy as np


def build(path, n_groups):
    with h5py.File(path, "w") as f:
        for i in range(n_groups):
            g = f.create_group(f"g{i:06d}")
            scale = g.create_dataset("x", (10,), dtype="f4")
            scale.make_scale("x")
            var = g.create_dataset("v", data=np.zeros(10, "f4"))
            var.dims[0].attach_scale(scale)


def lookup_ms(path, group):
    with h5py.File(path, "r") as f:
        ref = f[f"{group}/v"].attrs["DIMENSION_LIST"][0][0]
        t = time.perf_counter()
        f[ref].name  # cold: HDF5 has to find a path to the referenced object
        return (time.perf_counter() - t) * 1e3


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp())
    print("| groups | first group (ms) | last group (ms) |")
    print("|---|---|---|")
    for n in (10, 100, 1000, 10000):
        path = tmp / f"repro_{n}.h5"
        build(path, n)
        first = lookup_ms(path, "g000000")
        last = lookup_ms(path, f"g{n - 1:06d}")
        print(f"| {n} | {first:.3f} | {last:.3f} |")
    print(f"h5py {h5py.version.version}, HDF5 {h5py.version.hdf5_version}")
