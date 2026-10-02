import datetime

import h5py
import numpy as np
import pytest
import xarray as xr

pytest.importorskip("xdas")
pytest.importorskip("pyarrow")

import h5netcdf_patch  # noqa: E402
from xdas_cf_format import (  # noqa: E402
    DATA_FILE,
    XdasCFDataset,
    XdasCFDataWriter,
    XdasSaveWriter,
)

METADATA = {
    "record_sampling_rate_hz": 2000,
    "record_channel_spacing_m": 1.0204,
    "record_start_time": datetime.datetime(2024, 1, 1, 12, 0, 0, 123457),
}


@pytest.fixture(scope="module")
def chunks(tmp_path_factory):
    """The same 7 records (3 buckets) written by both writers."""
    rng = np.random.default_rng(1)
    records = [rng.standard_normal((300, 40)) for _ in range(7)]
    annotations = {"P": rng.random(40) * 300, "S": np.full(40, np.nan)}
    out = {}
    for label, writer in [("h5py", XdasCFDataWriter), ("xdas", XdasSaveWriter)]:
        path = tmp_path_factory.mktemp(label)
        with writer(path, "0", bucket_size=3) as w:
            for record in records:
                w.add_record(dict(METADATA), record, annotations)
        out[label] = XdasCFDataset(path, ["0"])
    return out, records, annotations


# netCDF dimension ids differ between writers (both are valid); the references hold
# object addresses. Their dtypes and shapes are still compared.
_ID_ATTRS = {"_Netcdf4Dimid", "_Netcdf4Coordinates", "DIMENSION_LIST", "REFERENCE_LIST"}


def _structure(path, group):
    """Every object below group: type, dtype, shape, fill value, attribute types and
    attribute values (except dimension ids and references)."""
    res = {}
    with h5py.File(path) as f:

        def visit(name, obj):
            is_group = isinstance(obj, h5py.Group)
            res[name] = (
                type(obj).__name__,
                None if is_group else (obj.dtype, obj.shape, repr(obj.fillvalue)),
                {
                    k: (
                        obj.attrs.get_id(k).dtype,
                        obj.attrs.get_id(k).shape,
                        None if k in _ID_ATTRS else repr(obj.attrs[k]),
                    )
                    for k in obj.attrs
                },
            )

        f[group].visititems(visit)
        res["."] = {k: f[group].attrs.get_id(k).dtype for k in f[group].attrs}
    return res


def test_writers_produce_the_same_records(chunks):
    (out, records, annotations) = chunks
    a, b = out["h5py"], out["xdas"]
    assert list(a.metadata["record_name"]) == list(b.metadata["record_name"])
    assert a.metadata["record_name"][6] == "000002/record_6"

    for i, record in enumerate(records):
        da_a, da_b = a.open_xdas(i), b.open_xdas(i)
        assert da_a.dims == da_b.dims == ("time", "distance")
        assert da_a.attrs == da_b.attrs
        np.testing.assert_array_equal(da_a.values, da_b.values)
        np.testing.assert_array_equal(da_a.values, record.astype("float32"))
        for coord in ("time", "distance"):
            np.testing.assert_array_equal(
                da_a.coords[coord].values, da_b.coords[coord].values
            )

        group = f"records/{a.metadata['record_name'][i]}"
        path_a = a.path / DATA_FILE.replace("$CHUNK", "0")
        path_b = b.path / DATA_FILE.replace("$CHUNK", "0")
        xr.testing.assert_identical(
            xr.open_dataset(path_a, group=group, engine="h5netcdf"),
            xr.open_dataset(path_b, group=group, engine="h5netcdf"),
        )
        assert _structure(path_a, group) == _structure(path_b, group)

        _, sample, sample_annotations = a.get_sample(i, record_virtual=False)
        np.testing.assert_array_equal(sample, record.astype("float32"))
        assert sample_annotations.keys() == annotations.keys()
        for key, value in annotations.items():
            np.testing.assert_array_equal(sample_annotations[key], value)


def test_coordinates_round_trip(chunks):
    (out, _, _) = chunks
    da = out["h5py"].open_xdas(0)
    time = da.coords["time"].values
    assert time[0] == np.datetime64("2024-01-01T12:00:00.123457", "ns")
    assert time[1] - time[0] == np.timedelta64(500_000, "ns")
    distance = da.coords["distance"].values
    assert distance[1] - distance[0] == pytest.approx(1.0204)


def test_netcdf_c_reads_the_h5py_writer(chunks):
    netCDF4 = pytest.importorskip("netCDF4")
    (out, records, annotations) = chunks
    path = out["h5py"].path / DATA_FILE.replace("$CHUNK", "0")
    with netCDF4.Dataset(path) as nc:
        g = nc["records/000002/record_6"]
        assert g["record"].dimensions == ("time", "distance")
        np.testing.assert_array_equal(g["record"][:], records[6].astype("float32"))
        np.testing.assert_array_equal(g["P"][:], annotations["P"])
        assert g["time_values"].units.endswith("since 2024-01-01 12:00:00.123457")


def test_h5netcdf_patch_resolves_the_same_dimensions(chunks):
    (out, _, _) = chunks

    def all_dimensions():
        res = {}
        for label, ds in out.items():
            path = ds.path / DATA_FILE.replace("$CHUNK", "0")
            with h5netcdf_patch.core.File(path, "r") as f:

                def visit(group):
                    for name, var in group.variables.items():
                        res[(label, group.name, name)] = var.dimensions
                    for child in group.groups.values():
                        visit(child)

                visit(f)
        return res

    h5netcdf_patch.disable()
    upstream = all_dimensions()
    h5netcdf_patch.enable()
    try:
        patched = all_dimensions()
    finally:
        h5netcdf_patch.disable()
    assert patched == upstream
    assert upstream[("h5py", "/records/000000/record_0", "record")] == (
        "time",
        "distance",
    )


@pytest.mark.parametrize("writer", [XdasCFDataWriter, XdasSaveWriter])
def test_rejects_picks_not_matching_channels(tmp_path, writer):
    with writer(tmp_path, "0") as w:
        with pytest.raises(ValueError, match="Annotation P has length 2"):
            w.add_record(dict(METADATA), np.zeros((10, 4)), {"P": np.zeros(2)})


@pytest.mark.parametrize("alias", [False, True])
def test_h5netcdf_patch_shadowed_and_aliased_scales(tmp_path, alias):
    """A child variable using the root scale x while the child defines its own x; with
    alias, the root scale is also hard-linked as z."""
    path = tmp_path / "shadow.nc"
    with h5netcdf_patch.core.File(path, "w") as f:
        f.dimensions = {"x": 3}
        f.create_variable("a", ("x",), data=np.zeros(3))
        child = f.create_group("child")
        child.dimensions = {"x": 3}
        child.create_variable("b", ("x",), data=np.zeros(3))
    with h5py.File(path, "a") as f:
        v = f["child"].create_dataset("c", data=np.zeros(3))
        v.dims[0].attach_scale(f["x"])
        if alias:
            f["z"] = f["x"]

    def dimensions():
        with h5netcdf_patch.core.File(path, "r") as f:
            child = f["child"]
            return {k: child[k].dimensions for k in ("b", "c")} | {
                "a": f["a"].dimensions
            }

    h5netcdf_patch.disable()
    upstream = dimensions()
    h5netcdf_patch.enable()
    try:
        patched = dimensions()
    finally:
        h5netcdf_patch.disable()
    assert patched == upstream
