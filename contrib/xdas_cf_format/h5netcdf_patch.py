"""
Candidate h5netcdf fix: resolve DIMENSION_LIST references without HDF5 name lookups.

h5netcdf.core.BaseVariable._lookup_dimensions dereferences each DIMENSION_LIST entry
and calls ``.name`` on it. An object opened through a reference has no known path, so
HDF5 (H5Iget_name) searches the file for one, which costs time proportional to the
number of objects preceding it. netCDF only allows a variable to use dimensions of its
own group or an ancestor, which h5netcdf already holds in ``_parent._all_dimensions``,
each opened by path. Comparing the dereferenced object to those by identity (h5py
compares file number and object address) gives the same answer without the search.

Use ``enable()``/``disable()``. Loaded as a pytest plugin (``-p h5netcdf_patch``), it
enables itself, e.g. to run h5netcdf's own test suite against the patch.
"""

import h5netcdf.core as core

_original = core.BaseVariable._lookup_dimensions


def _lookup_dimensions(self):
    attrs = self._h5ds.attrs
    if (
        "_Netcdf4Coordinates" in attrs
        and attrs.get("CLASS", None) == b"DIMENSION_SCALE"
    ) or "DIMENSION_LIST" not in attrs:
        return _original(self)
    if core._unlabeled_dimension_mix(self._h5ds) != "labeled":
        return _original(self)

    candidates = [
        (name, dim._h5ds)
        for name, dim in self._parent._all_dimensions.items()
        if not dim._phony
    ]
    names = []
    for ref in list(attrs.get("DIMENSION_LIST", [])):
        # As upstream: if several scales are attached, the last one wins
        obj = self._root._h5file[ref[-1]]
        match = next((name for name, h5ds in candidates if h5ds == obj), None)
        if match is None:  # scale outside the group chain: keep upstream behavior
            return _original(self)
        names.append(match)
    return tuple(names)


def enable():
    core.BaseVariable._lookup_dimensions = _lookup_dimensions


def disable():
    core.BaseVariable._lookup_dimensions = _original


def pytest_configure(config):
    enable()
