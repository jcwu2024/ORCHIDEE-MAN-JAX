from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from jax_orchidee.driver.domain import read_domain_grid
from jax_orchidee.driver.geometry_grid import (
    AllocationContractError,
    MPIInfrastructureBoundary,
    ProjectionInfrastructureBoundary,
    grid_allocate_glo,
    grid_init,
    grid_set_glo,
    grid_stuff,
    grid_toij_1d,
    grid_toij_2d,
    grid_toij_scal,
    grid_topolylist,
)
from jax_orchidee.driver.geometry_llxy import PROJ_LATLON, ProjectionInfo, ij_to_latlon
from jax_orchidee.driver.static import (
    model_bbox_from_lalo_resolution,
    regular_lonlat_grid_geometry,
)


def _regular_grid() -> tuple[np.ndarray, np.ndarray]:
    lon_axis = np.asarray([107.0, 109.0, 111.0])
    lat_axis = np.asarray([23.0, 21.0, 19.0])
    return np.repeat(lon_axis[:, None], 3, axis=1), np.repeat(
        lat_axis[None, :], 3, axis=0
    )


def _initialized(npts: int, iim: int, jjm: int, grid_type: str = "RegLonLat"):
    global_state = grid_allocate_glo(grid_set_glo(iim, jjm, npts), 4)
    return grid_init(npts, 4, grid_type, "ForcingGrid", global_state=global_state)


def test_grid_init_global_state_defaults_and_allocation_guards():
    global_state = grid_set_glo(3, 2, 4)
    allocated = grid_allocate_glo(global_state, 4)
    state = grid_init(4, 4, "prefix-RegLonLat-grid", "forcing", global_state=allocated)

    assert (state.grid_type, state.global_grid, state.nb_neighbours) == (
        "RegLonLat",
        True,
        8,
    )
    assert state.corners.shape == (4, 4, 2)
    assert allocated.lon.shape == (3, 2)
    assert (
        grid_init(4, 4, "RegLonLat", "forcing", global_state=allocated, existing=state)
        is not state
    )
    with pytest.raises(AllocationContractError, match="already allocated"):
        grid_allocate_glo(allocated, 4)
    with pytest.raises(AllocationContractError, match="requested contract"):
        grid_init(3, 4, "RegLonLat", "forcing", global_state=allocated, existing=state)
    with pytest.raises(AllocationContractError, match="change global dimensions"):
        grid_set_glo(4, 2, 4, state=allocated)


def test_grid_init_grid_type_defaults_and_validation_are_source_exact():
    allocated = grid_allocate_glo(grid_set_glo(1, 1, 1), 4)
    assert not grid_init(1, 4, "RegXY", "wrf", global_state=allocated).global_grid
    assert grid_init(1, 5, "UnStruct", "mesh", global_state=allocated).global_grid
    with pytest.raises(ValueError, match="exactly four"):
        grid_init(1, 3, "RegLonLat", "bad", global_state=allocated)
    with pytest.raises(ValueError, match="gtype"):
        grid_init(1, 4, "Gaussian", "bad", global_state=allocated)


def test_grid_topolylist_cross_checks_existing_static_regular_geometry():
    lon, lat = _regular_grid()
    kindex = np.arange(1, 10, dtype=np.int32)
    topology = grid_topolylist("RegLonLat", 4, 9, 3, 3, lon, lat, kindex)
    resolution, neighbours, area, seglength, corners = regular_lonlat_grid_geometry(
        lon, lat, kindex, return_corners=True
    )

    np.testing.assert_array_equal(topology.neighbours, neighbours)
    np.testing.assert_allclose(topology.seglength, seglength)
    np.testing.assert_allclose(topology.area, area)
    np.testing.assert_allclose(topology.corners, corners)
    np.testing.assert_allclose(
        np.column_stack(
            (
                (topology.seglength[:, 0] + topology.seglength[:, 2]) / 2,
                (topology.seglength[:, 1] + topology.seglength[:, 3]) / 2,
            )
        ),
        resolution,
    )
    np.testing.assert_array_equal(topology.ilandindex, np.tile([1, 2, 3], 3))
    np.testing.assert_array_equal(topology.jlandindex, np.repeat([1, 2, 3], 3))
    assert topology.corners.shape == (9, 4, 2)


def test_grid_stuff_serial_workflow_preserves_resolution_area_mask_and_bbox():
    lon, lat = _regular_grid()
    kindex = np.asarray([1, 5, 9], dtype=np.int32)
    fractions = np.asarray([1.0, 0.4, 0.0])
    completed = grid_stuff(_initialized(3, 3, 3), 3, 3, 3, lon, lat, kindex, fractions)
    static_resolution, static_neighbours, static_area, _, static_corners = (
        regular_lonlat_grid_geometry(lon, lat, kindex, return_corners=True)
    )

    np.testing.assert_allclose(completed.resolution, static_resolution)
    np.testing.assert_allclose(completed.area, static_area)
    np.testing.assert_array_equal(completed.neighbours, static_neighbours)
    np.testing.assert_allclose(completed.corners, static_corners)
    np.testing.assert_array_equal(completed.contfrac, fractions)
    np.testing.assert_array_equal(completed.ilandindex, [1, 2, 3])
    np.testing.assert_array_equal(completed.jlandindex, [1, 2, 3])
    bbox = model_bbox_from_lalo_resolution(completed.lalo, completed.resolution)
    assert np.all(bbox[:, 0] < completed.lalo[:, 1])
    assert np.all(bbox[:, 1] > completed.lalo[:, 1])


def test_grid_stuff_topology_overwrites_preexisting_land_indices():
    lon, lat = _regular_grid()
    state = _initialized(3, 3, 3)
    state = replace(
        state,
        ilandindex=np.asarray([9, 8, 7], dtype=np.int32),
        jlandindex=np.asarray([6, 5, 4], dtype=np.int32),
    )
    completed = grid_stuff(
        state,
        3,
        3,
        3,
        lon,
        lat,
        np.asarray([1, 5, 9], dtype=np.int32),
        np.asarray([1.0, 0.5, 0.25]),
    )

    np.testing.assert_array_equal(completed.ilandindex, [1, 2, 3])
    np.testing.assert_array_equal(completed.jlandindex, [1, 2, 3])


def test_grid_stuff_single_landpoint_and_mpi_boundary_are_explicit():
    lon = np.asarray([[109.0]])
    lat = np.asarray([[21.0]])
    completed = grid_stuff(
        _initialized(1, 1, 1), 1, 1, 1, lon, lat, np.asarray([1]), np.asarray([0.75])
    )
    assert completed.area[0] > 0.0
    assert completed.resolution.shape == (1, 2)
    np.testing.assert_array_equal(completed.neighbours, np.full((1, 8), -1))
    with pytest.raises(MPIInfrastructureBoundary, match="MPI"):
        grid_stuff(
            _initialized(1, 1, 1),
            1,
            1,
            1,
            lon,
            lat,
            np.asarray([1]),
            np.asarray([1.0]),
            mpi_size=2,
        )
    with pytest.raises(ValueError, match="RegLonLat or RegXY"):
        grid_topolylist("UnStruct", 4, 1, 1, 1, lon, lat, np.asarray([1]))


def test_grid_topolylist_and_stuff_implement_regxy_projection_segments_and_xyarea():
    projection = ProjectionInfo(
        PROJ_LATLON,
        init=True,
        lat1=40.0,
        lon1=100.0,
        latinc=1.0,
        loninc=2.0,
        knowni=1.0,
        knownj=1.0,
    )
    axis_i, axis_j = np.meshgrid(
        np.arange(1.0, 4.0), np.arange(1.0, 4.0), indexing="ij"
    )
    lat, lon = ij_to_latlon(projection, axis_i, axis_j)

    def transform(i: float, j: float) -> tuple[float, float]:
        return ij_to_latlon(projection, i, j)

    dxwrf = np.asarray([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]])
    dywrf = np.asarray([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
    kindex = np.asarray([1, 5, 9], dtype=np.int32)

    topology = grid_topolylist(
        "RegXY",
        4,
        3,
        3,
        3,
        lon,
        lat,
        kindex,
        projection=projection,
        ij_to_latlon=transform,
        dxwrf=dxwrf,
        dywrf=dywrf,
    )
    completed = grid_stuff(
        _initialized(3, 3, 3, "RegXY"),
        3,
        3,
        3,
        lon,
        lat,
        kindex,
        np.asarray([1.0, 0.5, 0.25]),
        projection=projection,
        ij_to_latlon=transform,
        dxwrf=dxwrf,
        dywrf=dywrf,
    )

    assert not topology.global_grid
    np.testing.assert_allclose(topology.corners[1, 0], [101.0, 40.5])
    np.testing.assert_allclose(topology.area, [20.0, 126.0, 320.0])
    np.testing.assert_allclose(completed.area, topology.area)
    np.testing.assert_allclose(completed.seglength, topology.seglength)
    assert np.all(completed.resolution > 0.0)
    with pytest.raises(ProjectionInfrastructureBoundary, match="projection and"):
        grid_topolylist("RegXY", 4, 3, 3, 3, lon, lat, kindex, dxwrf=dxwrf, dywrf=dywrf)
    with pytest.raises(ProjectionInfrastructureBoundary, match="dxwrf"):
        grid_topolylist(
            "RegXY",
            4,
            3,
            3,
            3,
            lon,
            lat,
            kindex,
            projection=projection,
            ij_to_latlon=transform,
        )


def _write_domain_case(path: Path) -> Path:
    forcing = path / "forcing_1961.nc"
    lon, lat = _regular_grid()
    contfrac = np.asarray([[1.0, 0.0, 0.5], [1.0, 1.0, 1.0], [0.25, 0.0, 1.0]]).T
    ds = xr.Dataset(
        {
            "nav_lon": (("y", "x"), lon.T),
            "nav_lat": (("y", "x"), lat.T),
            "contfrac": (("y", "x"), contfrac),
            "Areas": (("y", "x"), np.ones((3, 3))),
        }
    )
    ds.to_netcdf(forcing)
    config = path / "case.yaml"
    config.write_text(
        "drivers:\n  atmospheric_forcing:\n    file_pattern: '"
        + str(path / "forcing_{year}.nc").replace("\\", "/")
        + "'\ndomain:\n  west: 106\n  east: 112\n  south: 18\n  north: 24\n",
        encoding="utf-8",
    )
    return config


def test_grid_workflow_cross_checks_domain_regular_geometry(tmp_path):
    domain = read_domain_grid(_write_domain_case(tmp_path))
    completed = grid_stuff(
        _initialized(domain.nbindex, domain.iim, domain.jjm),
        domain.nbindex,
        domain.iim,
        domain.jjm,
        domain.lon,
        domain.lat,
        domain.kindex,
        domain.contfrac_land,
    )

    np.testing.assert_allclose(completed.lalo, domain.lalo)
    np.testing.assert_allclose(completed.resolution, domain.resolution)
    np.testing.assert_allclose(completed.area, domain.area)
    np.testing.assert_array_equal(completed.neighbours, domain.neighbours)
    np.testing.assert_allclose(completed.corners, domain.corners)
    np.testing.assert_array_equal(completed.contfrac, domain.contfrac_land)


@dataclass
class _Projection:
    def latlon_to_ij(self, lat: float, lon: float) -> tuple[float, float]:
        return lon / 2.0 + 1.0, 11.0 - lat / 5.0


def test_grid_toij_scalar_contract_makes_fortran_python_base_conversion_explicit():
    projection = _Projection()
    assert grid_toij_scal(8.0, 5.0, projection=projection) == (5.0, 10.0)
    assert grid_toij_scal(8.0, 5.0, projection=projection, index_base="python") == (
        4.0,
        9.0,
    )
    with pytest.raises(ProjectionInfrastructureBoundary, match="projection adapter"):
        grid_toij_scal(8.0, 5.0, projection=None)


def test_grid_toij_rank_wrappers_preserve_shape_values_and_base():
    projection = _Projection()
    lon = np.asarray([0.0, 2.0, 4.0])
    lat = np.asarray([0.0, 5.0, 10.0])
    ri, rj = grid_toij_1d(lon, lat, projection=projection, index_base="python")
    np.testing.assert_allclose(ri, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(rj, [10.0, 9.0, 8.0])

    ri2, rj2 = grid_toij_2d(lon.reshape(1, 3), lat.reshape(1, 3), projection=projection)
    np.testing.assert_allclose(ri2, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(rj2, [[11.0, 10.0, 9.0]])
    with pytest.raises(ValueError, match="rank-1"):
        grid_toij_1d(lon.reshape(1, 3), lat.reshape(1, 3), projection=projection)
