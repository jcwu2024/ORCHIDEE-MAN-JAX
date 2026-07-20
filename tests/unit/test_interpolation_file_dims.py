from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from jax_orchidee.driver.interpolation_file_dims import (
    InterpolationFileMetadataError,
    InterpolationFileReadError,
    interpweight_get_var2dims_file,
    interpweight_get_var3dims_file,
    interpweight_get_var4dims_file,
    read_variable_dimension_metadata,
)


def _metadata():
    return {
        "dimensions": {"time_counter": 2, "soil": 3, "latitude": 5, "longitude": 7},
        "variables": {
            "surface": {"dimensions": ["latitude", "longitude"]},
            "profile": {"dimensions": ["soil", "latitude", "longitude"]},
            "forcing": {
                "dimensions": ["time_counter", "soil", "latitude", "longitude"],
                "attributes": {"coordinates": "forecast_time soil_depth"},
            },
            "forecast_time": {
                "dimensions": ["time_counter"],
                "attributes": {
                    "standard_name": "time",
                    "units": "days since 2001-01-01",
                },
            },
            "soil_depth": {
                "dimensions": ["soil"],
                "attributes": {"axis": "Z", "positive": "down"},
            },
        },
    }


def test_pure_metadata_owners_return_netcdff_fortran_dimension_order():
    metadata = _metadata()
    metadata["dimensions"]["longitude"] = np.int64(7)
    assert interpweight_get_var2dims_file(metadata, "surface") == (7, 5)
    assert interpweight_get_var3dims_file(metadata, "profile") == (7, 5, 3)
    assert interpweight_get_var4dims_file(metadata, "forcing") == (7, 5, 3, 2)


def test_metadata_keeps_file_order_separate_and_identifies_time_and_level():
    result = read_variable_dimension_metadata(_metadata(), "forcing")
    assert result.file_dimension_names == (
        "time_counter",
        "soil",
        "latitude",
        "longitude",
    )
    assert result.file_dimension_sizes == (2, 3, 5, 7)
    assert result.fortran_dimension_names == (
        "longitude",
        "latitude",
        "soil",
        "time_counter",
    )
    assert result.fortran_dimension_sizes == (7, 5, 3, 2)
    assert (result.time_dimension, result.time_fortran_axis) == ("time_counter", 4)
    assert (result.level_dimension, result.level_fortran_axis) == ("soil", 3)


def test_dimension_coordinate_names_and_cf_attributes_are_both_recognized():
    metadata = {
        "dimensions": {"time": 11, "lev": 13, "x": 17},
        "variables": {
            "field": {"dimensions": ["time", "lev", "x"]},
            "time": {"dimensions": ["time"], "attributes": {}},
            "lev": {"dimensions": ["lev"], "attributes": {}},
        },
    }
    result = read_variable_dimension_metadata(metadata, "field")
    assert (result.time_fortran_axis, result.level_fortran_axis) == (3, 2)


def test_xarray_netcdf_reader_reverses_display_order_without_reading_values():
    dataset = xr.Dataset(
        {
            "lai": (
                ("month", "pft", "lat", "lon"),
                np.zeros((2, 3, 5, 7), dtype=np.float32),
                {"coordinates": "month_index pft_level"},
            ),
            "month_index": (("month",), np.arange(2), {"axis": "T"}),
            "pft_level": (("pft",), np.arange(3), {"axis": "Z"}),
        },
        coords={"lat": np.arange(5), "lon": np.arange(7)},
    )
    # Use the system temporary directory explicitly; no project cache/output is created.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "dims.nc"
        dataset.to_netcdf(path)
        result = read_variable_dimension_metadata(path, "lai")
        assert interpweight_get_var4dims_file(path, "lai") == (7, 5, 3, 2)
    assert result.file_dimension_names == ("month", "pft", "lat", "lon")
    assert result.fortran_dimension_names == ("lon", "lat", "pft", "month")
    assert (result.time_fortran_axis, result.level_fortran_axis) == (4, 3)


@pytest.mark.parametrize(
    ("owner", "name", "expected"),
    [
        (interpweight_get_var2dims_file, "profile", 2),
        (interpweight_get_var3dims_file, "surface", 3),
        (interpweight_get_var4dims_file, "profile", 4),
    ],
)
def test_rank_guards_are_exact(owner, name, expected):
    with pytest.raises(InterpolationFileMetadataError, match=rf"expected {expected}"):
        owner(_metadata(), name)


def test_missing_and_malformed_metadata_are_errors_not_defaults():
    with pytest.raises(InterpolationFileMetadataError, match="non-empty"):
        interpweight_get_var2dims_file(_metadata(), " ")
    with pytest.raises(InterpolationFileMetadataError, match="was not found"):
        interpweight_get_var2dims_file(_metadata(), "missing")
    bad_size = _metadata()
    bad_size["dimensions"]["longitude"] = -1
    with pytest.raises(InterpolationFileMetadataError, match="non-negative integer"):
        interpweight_get_var2dims_file(bad_size, "surface")
    missing_size = _metadata()
    del missing_size["dimensions"]["longitude"]
    with pytest.raises(InterpolationFileMetadataError, match="no size"):
        interpweight_get_var2dims_file(missing_size, "surface")


def test_conflicting_coordinate_classification_is_rejected():
    metadata = _metadata()
    metadata["variables"]["second_time"] = {
        "dimensions": ["soil"],
        "attributes": {"axis": "T"},
    }
    metadata["variables"]["forcing"]["attributes"]["coordinates"] += " second_time"
    with pytest.raises(
        InterpolationFileMetadataError, match="multiple time dimensions"
    ):
        read_variable_dimension_metadata(metadata, "forcing")


def test_netcdf_open_failure_is_not_converted_to_dimension_defaults():
    with tempfile.TemporaryDirectory() as directory:
        missing = Path(directory) / "missing.nc"
        with pytest.raises(
            InterpolationFileReadError, match="could not read NetCDF metadata"
        ):
            interpweight_get_var2dims_file(missing, "field")
