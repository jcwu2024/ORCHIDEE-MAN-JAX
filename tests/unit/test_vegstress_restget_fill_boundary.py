"""Observe new NetCDF fill inputs at the actual reader boundary only.

IOIPSL restcom.f90:152,158,1715-1736,1800-1802,2746-2747 treats
absent-field defaults separately from present stored data. These tests are
local Python reader observations, not a native IOIPSL or physics oracle.
"""
import json

import numpy as np
import pytest
from netCDF4 import Dataset, default_fillvals

from jax_orchidee.sechiba.restart_io import read_sechiba_restart_state


@pytest.mark.parametrize("mode", [
    "default_unwritten", "custom_fill_999999", "custom_fill_1e20", "missing_value_999999",
])
def test_actual_reader_keeps_fill_backing_values_as_present_not_missing_field(tmp_path, mode):
    path = tmp_path / "vegstress-fill.nc"
    fill = {"custom_fill_999999": 999999.0, "custom_fill_1e20": 1.0e20}.get(mode)
    with Dataset(path, "w") as dataset:
        for name, size in (("time", 1), ("veget", 14), ("y", 1), ("x", 2)):
            dataset.createDimension(name, size)
        options = {} if fill is None else {"fill_value": fill}
        variable = dataset.createVariable("vegstress", "f8", ("time", "veget", "y", "x"), **options)
        if mode == "missing_value_999999":
            variable[:] = 999999.0
            variable.setncattr("missing_value", 999999.0)
    with Dataset(path) as dataset:
        variable = dataset.variables["vegstress"]
        masked = variable[:]
        assert np.ma.isMaskedArray(masked)
        assert np.all(np.ma.getmaskarray(masked))
        attributes = {name: float(variable.getncattr(name)) for name in variable.ncattrs()
                      if name in ("_FillValue", "missing_value")}
        variable.set_auto_maskandscale(False)
        raw = np.asarray(variable[:])
    expected_value = (999999.0 if mode in ("custom_fill_999999", "missing_value_999999")
                      else 1.0e20 if mode == "custom_fill_1e20" else default_fillvals["f8"])
    np.testing.assert_array_equal(raw, np.full((1, 14, 1, 2), expected_value))
    read = read_sechiba_restart_state(path)
    assert "vegstress" in read.fields
    assert not np.ma.isMaskedArray(read.fields["vegstress"])
    np.testing.assert_array_equal(read.fields["vegstress"], np.full((2, 14), expected_value))
    print("FILL_BOUNDARY " + json.dumps({
        "mode": mode, "attributes": attributes, "masked_slots": int(np.ma.getmaskarray(masked).sum()),
        "normalized_shape": list(read.fields["vegstress"].shape), "raw_value": float(expected_value),
        "field_present": True, "reader_preserves_mask": False,
        "native_IOIPSL_observation": False,
    }, sort_keys=True))
