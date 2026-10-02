"""
Prototype: SeisBench DAS chunks whose records are xdas-readable CF-1.13 groups.

Layout of ``records_$CHUNK.hdf5`` (a valid netCDF4/HDF5 file):

    /                                 attrs: file_format, format_version
    /records/<bucket>/record_N/       CF-1.13 group, as xdas writes a DataArray
        record     (time, distance)   the DAS data
        time_*, distance_*            xdas coordinates (CF tie-point interpolation)
        P, S, ...  (distance,)        per-channel picks in samples, NaN allowed
        attr "annotations"            names of the pick variables

``metadata_$CHUNK.parquet`` is unchanged, except that ``record_name`` holds the path
below ``records`` (``<bucket>/record_N``).

Two writers produce the same records:

* :class:`XdasSaveWriter` uses xdas' own ``DataMapping.to_netcdf``. It is the
  reference for the layout and the baseline for the benchmark.
* :class:`XdasCFDataWriter` writes the same netCDF4 layout with plain h5py through one
  handle. xdas and xarray still produce the content (coordinate serialization and CF
  encoding); only h5netcdf's per-object bookkeeping and append-mode file scans are
  avoided. Records are bucketed so that no group has an unbounded number of children.
"""

from pathlib import Path
from typing import Any

import h5netcdf
import h5py
import numpy as np
import pandas as pd
import xarray as xr
import xdas as xd

FILE_FORMAT = "seisbench-das"
FORMAT_VERSION = "2.0-prototype"
METADATA_FILE = "metadata_$CHUNK.parquet"
DATA_FILE = "records_$CHUNK.hdf5"
GROUP = "records"
DATA_VAR = "record"

_NC_DIM_NAME = "This is a netCDF dimension but not a netCDF variable.{:10d}"


def to_dataarray(metadata: dict[str, Any], data: np.ndarray) -> xd.DataArray:
    """Build the xdas DataArray for a record from SeisBench metadata."""
    nt, nx = data.shape
    t0 = np.datetime64(pd.Timestamp(metadata["record_start_time"]), "ns")
    dt = np.timedelta64(round(1e9 / metadata["record_sampling_rate_hz"]), "ns")
    dx = float(metadata["record_channel_spacing_m"])
    x0 = float(metadata.get("record_start_distance_m", 0.0))
    coords = {
        "time": xd.InterpCoordinate(
            {"tie_indices": [0, nt - 1], "tie_values": [t0, t0 + (nt - 1) * dt]},
            dim="time",
        ),
        "distance": xd.InterpCoordinate(
            {"tie_indices": [0, nx - 1], "tie_values": [x0, x0 + (nx - 1) * dx]},
            dim="distance",
        ),
    }
    return xd.DataArray(data, coords, dims=("time", "distance"), name=DATA_VAR)


class _ChunkWriter:
    """Shared bookkeeping: paths, metadata rows, parquet output, context manager."""

    def __init__(self, path, chunk: str, dtype):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.path / METADATA_FILE.replace("$CHUNK", chunk)
        self.data_path = self.path / DATA_FILE.replace("$CHUNK", chunk)
        self.dtype = dtype
        self._metadata = []
        self._count = 0

    def add_record(self, metadata, data, annotations):
        nx = np.shape(data)[1]
        for key, value in annotations.items():
            if len(value) != nx:
                raise ValueError(
                    f"Annotation {key} has length {len(value)}, "
                    f"but data has {nx} channels."
                )
        n = self._count
        self._count += 1
        name = self._record_name(n)
        metadata = dict(metadata, record_name=name)
        da = to_dataarray(metadata, np.asarray(data, dtype=self.dtype))
        self._write(name, da, annotations)
        self._metadata.append(metadata)

    def close(self):
        self._close()
        pd.DataFrame(self._metadata).to_parquet(self.metadata_path, index=False)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class XdasSaveWriter(_ChunkWriter):
    """Reference writer using xdas' ``DataMapping.to_netcdf``.

    Records are buffered and written ``flush_every`` at a time, because every
    append-mode open (xarray, then xdas, then the picks below) makes h5netcdf scan the
    whole file for the largest netCDF dimension id.
    """

    def __init__(
        self, path, chunk="", dtype="float32", bucket_size=256, flush_every=64
    ):
        super().__init__(path, chunk, dtype)
        self.bucket_size = bucket_size
        self.flush_every = flush_every
        self._buffer = {}
        with h5py.File(self.data_path, "w") as f:
            f.attrs["file_format"] = FILE_FORMAT
            f.attrs["format_version"] = FORMAT_VERSION

    def _record_name(self, n):
        return f"{n // self.bucket_size:06d}/record_{n}"

    def _write(self, name, da, annotations):
        self._buffer[name] = (da, annotations)
        if len(self._buffer) >= self.flush_every:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        # One DataMapping per bucket, as xdas names groups <mapping name>/<key>
        buckets = {}
        for name, (da, _) in self._buffer.items():
            bucket, record = name.split("/")
            buckets.setdefault(bucket, {})[record] = da
        for bucket, records in buckets.items():
            xd.DataMapping(records, name=bucket).to_netcdf(
                str(self.data_path), mode="a", group=GROUP
            )
        with h5netcdf.File(self.data_path, "a") as f:
            for name, (_, annotations) in self._buffer.items():
                g = f[f"{GROUP}/{name}"]
                for key, value in annotations.items():
                    g.create_variable(key, ("distance",), data=np.asarray(value))
                g.attrs["annotations"] = sorted(annotations)
        self._buffer = {}

    def _close(self):
        self._flush()


class XdasCFDataWriter(_ChunkWriter):
    """Writes the same layout as :class:`XdasSaveWriter` with plain h5py."""

    def __init__(self, path, chunk="", dtype="float32", bucket_size=256):
        super().__init__(path, chunk, dtype)
        self.bucket_size = bucket_size
        self._dimid = 0  # netCDF dimension ids must be unique within the file
        self._file = h5py.File(self.data_path, "w")
        self._file.attrs["file_format"] = FILE_FORMAT
        self._file.attrs["format_version"] = FORMAT_VERSION
        self._file.attrs["_NCProperties"] = np.bytes_(
            f"version=2,h5py={h5py.version.version},hdf5={h5py.version.hdf5_version}"
        )
        self._records = self._file.create_group(GROUP, track_order=True)

    def _record_name(self, n):
        return f"{n // self.bucket_size:06d}/record_{n}"

    def _create_dimension(self, g, name, size):
        ds = g.create_dataset(name, (size,), dtype=">f4", track_order=True)
        ds.attrs["_Netcdf4Dimid"] = np.int32(self._dimid)
        ds.make_scale(_NC_DIM_NAME.format(size))
        self._dimid += 1
        return ds

    @staticmethod
    def _create_variable(g, scales, name, dims, data, attrs=None, fill=None):
        attrs = dict(attrs or {})
        fill = attrs.pop("_FillValue", fill)
        ds = g.create_dataset(name, data=data, fillvalue=fill)
        for i, dim in enumerate(dims):
            ds.dims[i].attach_scale(scales[dim])
        if dims:
            dimids = [int(scales[d].attrs["_Netcdf4Dimid"]) for d in dims]
            ds.attrs["_Netcdf4Coordinates"] = np.array(dimids, dtype="int32")
            ds.attrs["_Netcdf4Dimid"] = np.int32(dimids[0])
        if fill is not None:
            ds.attrs["_FillValue"] = np.array([fill], dtype=data.dtype)
        for key, value in attrs.items():
            ds.attrs[key] = value

    def _write(self, name, da, annotations):
        bucket = name.split("/")[0]
        if bucket not in self._records:
            self._records.create_group(bucket, track_order=True)
        g = self._records.create_group(name, track_order=True)

        # Same content as xdas' save: coordinates serialize themselves, xarray encodes
        dataset = xr.Dataset(attrs={"Conventions": "CF-1.13"})
        data_attrs = {}
        for coord in da.coords.values():
            dataset, data_attrs = coord._to_dataset(dataset, data_attrs)
        variables, group_attrs = xr.conventions.cf_encoder(
            dataset.variables, dataset.attrs
        )

        for key, value in group_attrs.items():
            g.attrs[key] = value
        sizes = {**dict(dataset.sizes), **dict(da.sizes)}
        scales = {dim: self._create_dimension(g, dim, n) for dim, n in sizes.items()}
        for key, var in variables.items():
            # xarray's netCDF backend gives float variables a NaN _FillValue
            fill = np.nan if var.dtype.kind == "f" else None
            self._create_variable(g, scales, key, var.dims, var.values, var.attrs, fill)
        self._create_variable(g, scales, DATA_VAR, da.dims, da.values, data_attrs)
        for key, value in annotations.items():
            self._create_variable(g, scales, key, ("distance",), np.asarray(value))
        g.attrs["annotations"] = sorted(annotations)

    def _close(self):
        self._file.close()


class XdasCFDataset:
    """Minimal reader mirroring ``DASDataset.get_sample``: parquet index, lazy h5py."""

    def __init__(self, path, chunks: list[str]):
        self.path = Path(path)
        frames = []
        for chunk in chunks:
            with h5py.File(self.path / DATA_FILE.replace("$CHUNK", chunk), "r") as f:
                if f.attrs.get("file_format") != FILE_FORMAT:
                    raise ValueError(f"Chunk {chunk!r} is not a {FILE_FORMAT} file")
            m = pd.read_parquet(self.path / METADATA_FILE.replace("$CHUNK", chunk))
            m["chunk"] = chunk
            frames.append(m)
        self.metadata = pd.concat(frames, ignore_index=True)
        self._files = {}

    def get_sample(self, idx, record_virtual=True):
        row = self.metadata.iloc[idx].to_dict()
        chunk = row["chunk"]
        if chunk not in self._files:
            self._files[chunk] = h5py.File(
                self.path / DATA_FILE.replace("$CHUNK", chunk), "r"
            )[GROUP]
        g = self._files[chunk][row["record_name"]]
        record = g[DATA_VAR] if record_virtual else g[DATA_VAR][()]
        annotations = {k: g[k][()] for k in g.attrs.get("annotations", [])}
        return row, record, annotations

    def open_xdas(self, idx) -> xd.DataArray:
        """Open a record with xdas, including its time and distance coordinates."""
        row = self.metadata.iloc[idx]
        path = self.path / DATA_FILE.replace("$CHUNK", row["chunk"])
        return xd.DataArray.from_netcdf(
            str(path), group=f"{GROUP}/{row['record_name']}"
        )
