from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from jax_orchidee.driver.forcing_domain_io import (
    FlingetBuffer,
    ForcingDomainIOError,
    IndexBounds,
    MPIInfrastructureBoundary,
    domain_size,
    flinget_buffer,
)


def test_domain_size_global_and_regional_zoom_have_explicit_index_bounds():
    lon = np.asarray([-180.0, -90.0, 0.0, 90.0, 180.0])
    lat = np.asarray([90.0, 30.0, -30.0, -90.0])
    global_domain = domain_size(-180, 180, 90, -90, lon, lat)
    np.testing.assert_array_equal(global_domain.iind, [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(global_domain.jind_zero_based, [0, 1, 2, 3])
    assert (global_domain.iim_g_begin, global_domain.iim_g_end) == (1, 4)

    regional = domain_size(-100, 10, 40, -40, lon, lat)
    np.testing.assert_array_equal(regional.iind_zero_based, [1, 2])
    np.testing.assert_array_equal(regional.jind, [2, 3])
    assert regional.i_segments == (IndexBounds(2, 3),)
    assert regional.j_segments == (IndexBounds(2, 3),)


def test_domain_size_normalizes_cyclic_longitudes_and_keeps_source_dateline_arm():
    lon = np.asarray([0.0, 90.0, 180.0, 270.0, 360.0])
    lat = np.asarray([-20.0, 20.0])
    regional = domain_size(-100, 10, 30, -30, lon, lat)
    np.testing.assert_array_equal(regional.iind, [1, 4, 5])

    dateline = domain_size(100, -100, 30, -30, lon, lat)
    assert dateline.over_dateline
    assert (dateline.iim_g_begin, dateline.iim_g_end) == (3, 5)
    # readdim2.f90:2316 is <= west OR >= east, not the usual inverse pair.
    np.testing.assert_array_equal(dateline.iind, [1, 2, 3, 4, 5])


def test_domain_size_accepts_fortran_2d_regular_grid_and_rejects_bad_inputs():
    lon_axis = np.asarray([-20.0, 0.0, 20.0])
    lat_axis = np.asarray([10.0, -10.0])
    lon, lat = np.meshgrid(lon_axis, lat_axis, indexing="ij")
    selected = domain_size(-20, 20, 20, -20, lon, lat)
    assert (selected.iim, selected.jjm) == (3, 2)

    with pytest.raises(ForcingDomainIOError, match="regular"):
        domain_size(-20, 20, 20, -20, lon + np.asarray([[0, 1]]), lat)
    with pytest.raises(ForcingDomainIOError, match="longitude"):
        domain_size(-181, 20, 20, -20, lon, lat)
    with pytest.raises(ForcingDomainIOError, match="south < north"):
        domain_size(-20, 20, -20, 20, lon, lat)


def test_array_buffer_slices_time_reuses_cache_and_guards_eof():
    values = np.arange(3 * 2 * 5, dtype=np.float64).reshape(3, 2, 5)
    buffer = FlingetBuffer({"Tair": values}, nbuff=3)
    first = flinget_buffer(buffer, "Tair", 3, 2, 1, 5, 2, 2)
    second = buffer.read("Tair", 3, 2, 1, 5, 4, 4)
    last = buffer.read("Tair", 3, 2, 1, 5, 5, 5)
    np.testing.assert_array_equal(first, values[:, :, 1])
    np.testing.assert_array_equal(second, values[:, :, 3])
    np.testing.assert_array_equal(last, values[:, :, 4])
    assert buffer.read_count == {"Tair": 2}

    with pytest.raises(EOFError, match="outside 1..5"):
        buffer.read("Tair", 3, 2, 1, 5, 6, 6)
    with pytest.raises(ForcingDomainIOError, match="ite must equal itb"):
        buffer.read("Tair", 3, 2, 1, 5, 1, 2)


def test_explicit_array_axis_order_and_variables_have_independent_buffers():
    canonical = np.arange(4 * 3 * 2, dtype=np.float64).reshape(4, 3, 2)
    ytx = np.transpose(canonical, (1, 2, 0))
    source = {"a": (ytx, ("latitude", "time", "longitude")), "b": canonical + 100}
    buffer = FlingetBuffer(source, nbuff=2)
    np.testing.assert_array_equal(
        buffer.read("a", 4, 3, 1, 2, 2, 2), canonical[:, :, 1]
    )
    np.testing.assert_array_equal(
        buffer.read("b", 4, 3, 1, 2, 1, 1), canonical[:, :, 0] + 100
    )
    assert buffer.read_count == {"a": 1, "b": 1}


def test_temporary_netcdf_file_order_is_converted_to_fortran_ij_order():
    file_values = np.arange(3 * 2 * 4, dtype=np.float64).reshape(3, 2, 4)
    dataset = xr.Dataset({"Qair": (("time", "latitude", "longitude"), file_values)})
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "forcing.nc"
        dataset.to_netcdf(path)
        buffer = FlingetBuffer(path, nbuff=0)
        slab = buffer.read("Qair", 4, 2, 1, 3, 3, 3)
    np.testing.assert_array_equal(slab, file_values[2].T)
    assert buffer.nbuff == 3


def test_static_2d_array_and_coherence_guards_are_explicit():
    static = np.arange(6, dtype=np.float64).reshape(3, 2)
    buffer = FlingetBuffer({"contfrac": static}, nbuff=9, max_variables=1)
    np.testing.assert_array_equal(buffer.read("contfrac", 3, 2, 1, 1, 1, 1), static)
    assert buffer.nbuff == 1
    with pytest.raises(ForcingDomainIOError, match="ttm 2 differs"):
        buffer.read("contfrac", 3, 2, 1, 2, 1, 1)
    with pytest.raises(ForcingDomainIOError, match="too many"):
        buffer.read("other", 3, 2, 1, 1, 1, 1)
    with pytest.raises(ForcingDomainIOError, match="NBUFF"):
        FlingetBuffer({"x": static}, nbuff=-1)


def test_mpi_calls_stop_at_an_explicit_infrastructure_boundary():
    with pytest.raises(MPIInfrastructureBoundary, match="MPI"):
        domain_size(-180, 180, 90, -90, [0], [0], mpi_size=2)
    buffer = FlingetBuffer({"x": np.ones((1, 1, 1))})
    with pytest.raises(MPIInfrastructureBoundary, match="MPI"):
        buffer.read("x", 1, 1, 1, 1, 1, 1, mpi_size=2)
