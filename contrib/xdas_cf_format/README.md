# Prototype: DAS records as xdas CF groups

This prototype explores storing each SeisBench DAS record as an xdas-readable CF-1.13 group, so a record can be opened with full time and distance coordinates by xdas (and read by xarray and netCDF-C), while SeisBench keeps its parquet index and fast h5py reads. It is not wired into `seisbench.data`.

## Layout

`records_$CHUNK.hdf5` is a valid netCDF4/HDF5 file:

```
/                                 attrs: file_format="seisbench-das", format_version="2.0-prototype"
/records/<bucket>/record_N/       CF-1.13 group, exactly as xdas writes a DataArray
    record     (time, distance)   the DAS data
    time_*, distance_*            xdas coordinates (CF tie-point interpolation)
    P, S, ...  (distance,)        per-channel picks in samples, NaN allowed
    attr "annotations"            names of the pick variables
```

`metadata_$CHUNK.parquet` is unchanged except that `record_name` holds the path below `records` (`<bucket>/record_N`, 256 records per bucket by default).

## Files

- `xdas_cf_format.py`: `XdasSaveWriter` (reference writer using xdas' `DataMapping.to_netcdf`), `XdasCFDataWriter` (same layout written with plain h5py), and `XdasCFDataset` (parquet index, h5py `get_sample`, `open_xdas`).
- `h5netcdf_patch.py`: candidate h5netcdf fix for slow dimension-name lookups, with `enable()`/`disable()`.
- `test_xdas_cf_format.py`: both writers produce identical records through xdas, xarray, SeisBench-style reads and at the HDF5 level (objects, dtypes, fill values, attribute types); netCDF-C reads the files; the patch resolves the same dimension names as upstream h5netcdf.
- `benchmark.py`: write and read costs against the current format.
- `name_lookup_repro.py`: HDF5-only reproduction of the name-lookup cost.

## Findings

Writing through xdas is slow, and opening records with xdas slows down as the file grows. Neither comes from the layout. Both come from h5netcdf, which xdas and xarray use for netCDF4:

1. Every append-mode `h5netcdf.File` scans the whole file (`_get_maximum_dimension_id`) to find the next netCDF dimension id. Writing through `DataMapping.to_netcdf` opens the file in append mode on each save, so writing in batches costs time proportional to the square of the record count.
2. h5netcdf reopens HDF5 objects by path on every property access, which makes creating the 13 objects per record expensive (about 20 ms).
3. To name a variable's dimensions, h5netcdf dereferences its `DIMENSION_LIST` and calls `.name`. An object opened by reference has no known path, so HDF5 searches the file for one, and the cost grows with the number of objects visited before it (`name_lookup_repro.py`: 0.09 ms for the first of 10,000 groups, 237 ms for the last).
4. h5netcdf classifies every child of a group when opening it, so a group holding all records makes every open touch all of them.

The prototype addresses these as follows:

- `XdasCFDataWriter` writes the netCDF4 layout with h5py through one handle (fixes 1 and 2). xdas and xarray still produce the content, and the output is identical to xdas' own.
- Bucketing records bounds the children per group (works around 4).
- `h5netcdf_patch.py` matches the dereferenced scales against the dimensions already in scope by object identity instead of by name (fixes 3). It is a candidate for upstream.

## Results

`python benchmark.py 100 500 2000`, milliseconds per record (500 x 200 float32 records, two pick arrays; h5py 3.16.0, HDF5 2.0.0, h5netcdf 1.8.1, xarray 2026.9.0, xdas 0.2.9):

| records | setup | write | get_sample | xdas open |
|---|---|---|---|---|
| 2000 | current SeisBench format | 0.38 | 0.51 | n/a |
| 2000 | xdas save, one group | 82.84 | 0.54 | 607.92 |
| 2000 | h5py writer, one group | 4.63 | 0.58 | 633.57 |
| 2000 | h5py writer, buckets | 4.65 | 0.55 | 699.43 |
| 2000 | h5py writer, one group + patch | 5.03 | 0.58 | 69.82 |
| 2000 | h5py writer, buckets + patch | 5.24 | 0.57 | 29.67 |

With the h5py writer, buckets and the patch, write and xdas-open costs stay flat with file size (about 30 ms per xdas open at 10,000 records in an earlier run). SeisBench's h5py `get_sample` is unaffected by the layout. The remaining write cost against the current format (about 5 ms vs 0.4 ms) is intrinsic to the CF layout: 13 HDF5 objects and about 32 attributes per record, plus xdas/xarray encoding.

## Caveats

- `XdasCFDataWriter` reproduces netCDF4 bookkeeping by hand (`_Netcdf4Dimid`, `_Netcdf4Coordinates`, dimension scale names) and uses xdas' private `Coordinate._to_dataset`. The equivalence test is what guards against drift.
- netCDF-C reads the files correctly, but its open performance on large files was not measured.
- With the patch, h5netcdf's own test suite gives the same result as without it (113 passed, 111 skipped, 20 xfailed, 1 failure that also occurs without the patch). The skipped tests were not investigated.

## Running

Requires the `das` extra plus `pytest`; `netCDF4` is optional for the netCDF-C test. From this directory:

```
python -m pytest test_xdas_cf_format.py
python benchmark.py 100 500 2000
python name_lookup_repro.py
python -m pytest -p h5netcdf_patch --pyargs h5netcdf.tests   # h5netcdf's suite with the patch
```
