"""
Write and read costs of the current SeisBench DAS format and the xdas CF prototype.

    python benchmark.py 100 500 2000

Columns are milliseconds per record: writing, a random SeisBench-style
``get_sample`` (h5py), and opening a random record with xdas (coordinates included).
Records are 500 x 200 float32 with two pick arrays.
"""

import datetime
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import h5netcdf_patch
import seisbench.data as sbd
from xdas_cf_format import XdasCFDataset, XdasCFDataWriter, XdasSaveWriter

NT, NX = 500, 200
METADATA = {
    "record_sampling_rate_hz": 100,
    "record_channel_spacing_m": 4.0,
    "record_start_time": datetime.datetime(2024, 1, 1),
}
ONE_GROUP = 10**9  # bucket size that puts every record in a single group

CONFIGS = [
    # label, writer factory, apply h5netcdf patch
    ("current SeisBench format", lambda p: sbd.DASDataWriter(p, "0"), False),
    (
        "xdas save, one group",
        lambda p: XdasSaveWriter(p, "0", bucket_size=ONE_GROUP),
        False,
    ),
    (
        "h5py writer, one group",
        lambda p: XdasCFDataWriter(p, "0", bucket_size=ONE_GROUP),
        False,
    ),
    ("h5py writer, buckets", lambda p: XdasCFDataWriter(p, "0"), False),
    (
        "h5py writer, one group + patch",
        lambda p: XdasCFDataWriter(p, "0", bucket_size=ONE_GROUP),
        True,
    ),
    ("h5py writer, buckets + patch", lambda p: XdasCFDataWriter(p, "0"), True),
]


def run(n, label, make, patched, rng):
    record = rng.standard_normal((NT, NX)).astype("float32")
    annotations = {"P": rng.random(NX) * NT, "S": rng.random(NX) * NT}
    path = Path(tempfile.mkdtemp())

    t = time.perf_counter()
    with make(path) as writer:
        for _ in range(n):
            writer.add_record(dict(METADATA), record, annotations)
    write = (time.perf_counter() - t) / n * 1e3

    current = isinstance(writer, sbd.DASDataWriter)
    dataset = sbd.DASDataset(path) if current else XdasCFDataset(path, ["0"])
    idx = rng.integers(0, n, 1000)
    t = time.perf_counter()
    for i in idx:
        dataset.get_sample(int(i), record_virtual=False)
    sample = (time.perf_counter() - t) / len(idx) * 1e3

    xdas_open = float("nan")
    if not current:
        (h5netcdf_patch.enable if patched else h5netcdf_patch.disable)()
        idx = rng.integers(0, n, 20)
        t = time.perf_counter()
        for i in idx:
            dataset.open_xdas(int(i)).values
        xdas_open = (time.perf_counter() - t) / len(idx) * 1e3
        h5netcdf_patch.disable()

    print(
        f"| {n:6d} | {label:32s} | {write:7.2f} | {sample:6.2f} | {xdas_open:8.2f} |",
        flush=True,
    )


def main(sizes):
    rng = np.random.default_rng(0)
    print("| records | setup | write | get_sample | xdas open |")
    print("|---|---|---|---|---|")
    for n in sizes:
        for label, make, patched in CONFIGS:
            run(n, label, make, patched, rng)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [100, 500, 2000])
