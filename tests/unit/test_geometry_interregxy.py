from __future__ import annotations

import numpy as np
import pytest

from jax_orchidee.driver.geometry_interregxy import (
    R_EARTH,
    UNDEFINED_INDEX,
    interregxy_aggr2d,
    interregxy_aggrve,
    interregxy_findpoints,
)
from jax_orchidee.driver.geometry_polygons import FortranUndefinedOutputError


def _target(shape=(1, 1)):
    return np.full(shape, 10.0), np.full(shape, 20.0)


def test_findpoints_full_cell_overlap_and_undefined_tail_contract():
    ri = np.array([1.0, 0.5, 1.5, 1.5, 0.5])
    rj = np.array([1.0, 0.5, 0.5, 1.5, 1.5])
    dx, dy = _target()
    result = interregxy_findpoints(
        1, 1, 99.0, -99.0, 7, 9, ri, rj, 3, False, dx_wrf=dx, dy_wrf=dy
    )
    np.testing.assert_array_equal(result.nbpt, [[1]])
    np.testing.assert_array_equal(result.sourcei[0, 0], [7, UNDEFINED_INDEX, UNDEFINED_INDEX])
    np.testing.assert_array_equal(result.sourcej[0, 0], [9, UNDEFINED_INDEX, UNDEFINED_INDEX])
    assert result.overlap_area[0, 0, 0] == pytest.approx(200.0)
    assert np.all(np.isnan(result.overlap_area[0, 0, 1:]))


def test_findpoints_half_overlap_and_accumulates_in_call_order():
    ri = np.array([0.75, 0.5, 1.0, 1.0, 0.5])
    rj = np.array([1.0, 0.5, 0.5, 1.5, 1.5])
    dx, dy = _target()
    first = interregxy_findpoints(
        1, 1, 0.0, 0.0, 3, 4, ri, rj, 2, False, dx_wrf=dx, dy_wrf=dy
    )
    second = interregxy_findpoints(
        1, 1, 0.0, 0.0, 8, 6, ri, rj, 2, False, *first, dx_wrf=dx, dy_wrf=dy
    )
    np.testing.assert_array_equal(second.nbpt, [[2]])
    np.testing.assert_array_equal(second.sourcei[0, 0], [3, 8])
    np.testing.assert_array_equal(second.sourcej[0, 0], [4, 6])
    np.testing.assert_allclose(second.overlap_area[0, 0], [100.0, 100.0])


def test_findpoints_capacity_and_collinear_degenerate_paths_are_not_approximated():
    ri = np.array([1.0, 0.5, 1.5, 1.5, 0.5])
    rj = np.array([1.0, 0.5, 0.5, 1.5, 1.5])
    dx, dy = _target()
    first = interregxy_findpoints(
        1, 1, 0.0, 0.0, 1, 1, ri, rj, 1, False, dx_wrf=dx, dy_wrf=dy
    )
    with pytest.raises(ValueError, match="nbpt_max"):
        interregxy_findpoints(
            1, 1, 0.0, 0.0, 2, 1, ri, rj, 1, False, *first, dx_wrf=dx, dy_wrf=dy
        )
    dx2, dy2 = _target((2, 1))
    with pytest.raises(FortranUndefinedOutputError, match="before assigning nvert_out"):
        interregxy_findpoints(
            2, 1, 0.0, 0.0, 1, 1, ri, rj, 3, False, dx_wrf=dx2, dy_wrf=dy2
        )


def test_findpoints_large_boundary_workspace_branch_can_return_no_candidates():
    # 60 boundary points select dimpoly=4*nbr at lines 371-373. Coordinates
    # remain outside the clipped target scan, so no undefined geometry is read.
    angles = np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False)
    ri = np.concatenate(([-5.0], -5.0 + 0.1 * np.cos(angles)))
    rj = np.concatenate(([-5.0], -5.0 + 0.1 * np.sin(angles)))
    dx, dy = _target()
    result = interregxy_findpoints(
        1, 1, 0.0, 0.0, 1, 1, ri, rj, 1, False, dx_wrf=dx, dy_wrf=dy
    )
    np.testing.assert_array_equal(result.nbpt, [[0]])


def test_aggr2d_keeps_source_scan_order_mask_nonuse_and_land_gather():
    lalo = np.array([[0.0, 0.0], [3.0, 3.0]])
    neighbours = np.zeros((2, 4), dtype=np.int32)
    resolution = np.ones((2, 2))
    contfrac = np.ones(2)
    lon = np.array([[1.0, 1.0], [2.0, 2.0]])
    lat = np.array([[1.0, 2.0], [1.0, 2.0]])
    dx, dy = _target((3, 3))
    kwargs = dict(
        grid_toij=lambda lons, lats: (lons + 0.1, lats + 0.1),
        land_indices=np.array([[1, 1], [2, 2]]),
        dx_wrf=dx,
        dy_wrf=dy,
    )
    result = interregxy_aggr2d(
        2, lalo, neighbours, resolution, contfrac, 2, 2, lon, lat,
        np.zeros((2, 2), dtype=np.int32), "oracle", 8, **kwargs
    )
    masked = interregxy_aggr2d(
        2, lalo, neighbours, resolution, contfrac, 2, 2, lon, lat,
        np.ones((2, 2), dtype=np.int32), "oracle", 8, **kwargs
    )
    np.testing.assert_array_equal(result.indinc, masked.indinc)
    np.testing.assert_allclose(result.areaoverlap, masked.areaoverlap)
    np.testing.assert_array_equal(result.indinc[0, 0], [1, 1])
    np.testing.assert_array_equal(
        result.indinc[1, :4], [[1, 1], [1, 2], [2, 1], [2, 2]]
    )
    np.testing.assert_allclose(result.areaoverlap[0, :1], [162.0])
    np.testing.assert_allclose(result.areaoverlap[1, :4], [2.0, 18.0, 18.0, 162.0])
    assert np.all(result.indinc[:, 4:] == UNDEFINED_INDEX)
    assert result.ok


def test_aggr2d_descending_latitude_branch_and_strict_bbox_exclusion():
    lalo = np.array([[0.0, 0.0], [3.0, 3.0]])
    common = (2, lalo, np.zeros((2, 4), dtype=int), np.ones((2, 2)), np.ones(2))
    lon = np.array([[1.0, 1.0], [2.0, 2.0]])
    descending_lat = np.array([[2.0, 1.0], [2.0, 1.0]])
    dx, dy = _target((3, 3))
    result = interregxy_aggr2d(
        *common, 2, 2, lon, descending_lat, np.zeros((2, 2), dtype=int), "descending", 8,
        grid_toij=lambda lons, lats: (lons + 10.0, lats + 10.0),
        land_indices=np.array([[1, 1], [2, 2]]), dx_wrf=dx, dy_wrf=dy,
    )
    assert np.all(result.indinc == UNDEFINED_INDEX)
    assert np.all(result.areaoverlap == 0.0)


def test_aggrve_uses_source_earth_conversion_and_gathers_only_source_i():
    lalo = np.array([[0.0, 0.0], [3.0, 3.0]])
    half_width_point_eight_degrees = 0.8 * R_EARTH / (180.0 / (2.0 * np.pi))
    dx, dy = _target((3, 3))
    result = interregxy_aggrve(
        2, lalo, np.zeros((2, 4), dtype=int), np.ones((2, 2)), np.ones(2),
        2, np.array([1.0, 2.0]), np.array([1.0, 2.0]),
        half_width_point_eight_degrees, half_width_point_eight_degrees, "vector", 5,
        grid_toij=lambda lons, lats: (lons + 0.1, lats + 0.1),
        land_indices=np.array([[1, 1], [2, 2]]), dx_wrf=dx, dy_wrf=dy,
    )
    np.testing.assert_array_equal(result.indinc[:, 0], [1, 2])
    assert np.all(result.indinc[:, 1:] == UNDEFINED_INDEX)
    np.testing.assert_allclose(result.areaoverlap[:, 0], [128.01130986, 128.04057527])
    assert result.ok

