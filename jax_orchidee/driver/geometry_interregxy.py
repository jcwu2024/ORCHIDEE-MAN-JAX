"""Explicit-state port of ORCHIDEE ``interregxy.f90`` polygon aggregation."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from jax_orchidee.driver.geometry_polygons import (
    polygones_area,
    polygones_cleanup,
    polygones_convexhull,
    polygones_crossing,
    polygones_extend,
    polygones_intersection,
)


INTERREGXY_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_global/interregxy.f90::interregxy_aggr2d lines 18-174",
    "fortran_source/ORCHIDEE/src_global/interregxy.f90::interregxy_aggrve lines 178-327",
    "fortran_source/ORCHIDEE/src_global/interregxy.f90::interregxy_findpoints lines 331-488",
)

UNDEFINED_INDEX = np.iinfo(np.int32).min
R_EARTH = 6_378_000.0
SBD = 5


class FindPointsResult(NamedTuple):
    nbpt: np.ndarray
    sourcei: np.ndarray
    sourcej: np.ndarray
    overlap_area: np.ndarray


class Aggregation2DResult(NamedTuple):
    indinc: np.ndarray
    areaoverlap: np.ndarray
    ok: bool


class AggregationVectorResult(NamedTuple):
    indinc: np.ndarray
    areaoverlap: np.ndarray
    ok: bool


GridToIJ = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def _target_geometry(dx_wrf, dy_wrf) -> tuple[np.ndarray, np.ndarray]:
    dx = np.asarray(dx_wrf, dtype=np.float64)
    dy = np.asarray(dy_wrf, dtype=np.float64)
    if dx.ndim != 2 or dy.shape != dx.shape:
        raise ValueError("dx_wrf and dy_wrf must have the same rank-2 shape")
    return dx, dy


def _findpoint_state(
    nb_we: int,
    nb_sn: int,
    nbpt_max: int,
    nbpt,
    sourcei,
    sourcej,
    overlap_area,
) -> FindPointsResult:
    counts = np.zeros((nb_we, nb_sn), dtype=np.int32) if nbpt is None else np.asarray(nbpt).copy()
    indices_i = (
        np.full((nb_we, nb_sn, nbpt_max), UNDEFINED_INDEX, dtype=np.int32)
        if sourcei is None
        else np.asarray(sourcei).copy()
    )
    indices_j = (
        np.full((nb_we, nb_sn, nbpt_max), UNDEFINED_INDEX, dtype=np.int32)
        if sourcej is None
        else np.asarray(sourcej).copy()
    )
    areas = (
        np.full((nb_we, nb_sn, nbpt_max), np.nan, dtype=np.float64)
        if overlap_area is None
        else np.asarray(overlap_area, dtype=np.float64).copy()
    )
    if counts.shape != (nb_we, nb_sn):
        raise ValueError(f"nbpt must have shape {(nb_we, nb_sn)}")
    expected = (nb_we, nb_sn, nbpt_max)
    if indices_i.shape != expected or indices_j.shape != expected or areas.shape != expected:
        raise ValueError(f"sourcei, sourcej, and overlap_area must have shape {expected}")
    return FindPointsResult(counts, indices_i, indices_j, areas)


def interregxy_findpoints(
    nb_we: int,
    nb_sn: int,
    lon_cen: float,
    lat_cen: float,
    is_: int,
    js: int,
    ri,
    rj,
    nbpt_max: int,
    checkprint: bool,
    nbpt=None,
    sourcei=None,
    sourcej=None,
    overlap_area=None,
    *,
    dx_wrf,
    dy_wrf,
) -> FindPointsResult:
    """Accumulate source-cell overlap in target cells in Fortran scan order."""

    del lon_cen, lat_cen  # Declared and passed by Fortran, but never read at lines 331-488.
    ri_array = np.asarray(ri, dtype=np.float64)
    rj_array = np.asarray(rj, dtype=np.float64)
    if ri_array.ndim != 1 or rj_array.shape != ri_array.shape:
        raise ValueError("ri and rj must be rank-1 arrays with the same shape")
    nbr = ri_array.size
    dx, dy = _target_geometry(dx_wrf, dy_wrf)
    if dx.shape != (nb_we, nb_sn):
        raise ValueError(f"target geometry must have shape {(nb_we, nb_sn)}")
    state = _findpoint_state(nb_we, nb_sn, nbpt_max, nbpt, sourcei, sourcej, overlap_area)
    counts, indices_i, indices_j, areas = state

    dimpoly = 200
    if dimpoly < 4 * (nbr - 1):
        dimpoly = 4 * nbr
    poly_in = np.column_stack((ri_array[1:], rj_array[1:]))
    poly_wrf = np.array(
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
        dtype=np.result_type(ri_array, rj_array, np.float64),
    )
    ilow = max(int(np.floor(np.min(ri_array))), 1)
    iup = min(int(np.ceil(np.max(ri_array))), nb_we)
    jlow = max(int(np.floor(np.min(rj_array))), 1)
    jup = min(int(np.ceil(np.max(rj_array))), nb_sn)

    for ix in range(ilow, iup + 1):
        for jx in range(jlow, jup + 1):
            shifted = poly_in - np.array([ix, jx])
            if checkprint and ix == 50 and jx == 46:
                print("X = ", shifted[:, 0])
                print("Y = ", shifted[:, 1])
            extended_in = polygones_extend(nbr - 1, shifted, 5, output_capacity=dimpoly)
            extended_wrf = polygones_extend(4, poly_wrf, 5, output_capacity=dimpoly)
            if checkprint and ix == 50 and jx == 46:
                print("X_extend = ", extended_in.vertices[:, 0])
                print("Y_extend = ", extended_in.vertices[:, 1])
            inside = polygones_intersection(
                extended_in.nvert,
                extended_in.vertices,
                extended_wrf.nvert,
                extended_wrf.vertices,
                output_capacity=dimpoly,
                poly_b_capacity=dimpoly,
            )
            crossings = polygones_crossing(
                nbr - 1, shifted, 4, poly_wrf, output_capacity=dimpoly
            )
            total = inside.nvert + crossings.nvert
            if checkprint and ix == 50 and jx == 46:
                print("nbint = nbcross = ", inside.nvert, crossings.nvert)
            if total > dimpoly:
                raise ValueError("interregxy_findpoints: intersection exceeds dimpoly")
            if total > 2:
                combined = np.concatenate((inside.vertices, crossings.vertices), axis=0)
                if checkprint and ix == 50 and jx == 46:
                    print("X_overlap = ", combined[:, 0])
                    print("Y_overlap = ", combined[:, 1])
                cleaned = polygones_cleanup(total, combined, output_capacity=dimpoly)
                # This source diagnostic is unreachable because total is an integer > 2.
                if total < 3:
                    print("Cleaning goes too far: ", total, " to ", cleaned.nvert)
                if checkprint and ix == 50 and jx == 46:
                    print("X_final = ", cleaned.vertices[:, 0])
                    print("Y_final = ", cleaned.vertices[:, 1])
                if cleaned.nvert > 2:
                    hull = polygones_convexhull(
                        cleaned.nvert, cleaned.vertices, output_capacity=dimpoly
                    )
                    if checkprint and ix == 50 and jx == 46:
                        print("X_sorted = ", hull.vertices[:, 0])
                        print("Y_sorted = ", hull.vertices[:, 1])
                    overlap = polygones_area(
                        hull.nvert, hull.vertices, dx[ix - 1, jx - 1], dy[ix - 1, jx - 1]
                    )
                else:
                    overlap = 0.0
                if checkprint and ix == 50 and jx == 46:
                    print(ix, jx, "overlaparea = ", overlap)
                if overlap > 0.0:
                    slot = int(counts[ix - 1, jx - 1])
                    counts[ix - 1, jx - 1] = slot + 1
                    if slot + 1 > nbpt_max:
                        raise ValueError("interregxy_findpoints: nbpt_max is too small")
                    indices_i[ix - 1, jx - 1, slot] = is_
                    indices_j[ix - 1, jx - 1, slot] = js
                    areas[ix - 1, jx - 1, slot] = overlap
    return FindPointsResult(counts, indices_i, indices_j, areas)


def _inputs_common(nbpt, lalo, neighbours, resolution, contfrac, incmax, land_indices):
    coordinates = np.asarray(lalo, dtype=np.float64)
    neigh = np.asarray(neighbours)
    resol = np.asarray(resolution, dtype=np.float64)
    fraction = np.asarray(contfrac, dtype=np.float64)
    land = np.asarray(land_indices)
    if coordinates.shape != (nbpt, 2):
        raise ValueError(f"lalo must have shape {(nbpt, 2)}")
    if neigh.ndim != 2 or neigh.shape[0] != nbpt:
        raise ValueError("neighbours must be rank 2 with nbpt rows")
    if resol.shape != (nbpt, 2) or fraction.shape != (nbpt,):
        raise ValueError("resolution/contfrac shapes do not match nbpt")
    if land.shape != (nbpt, 2):
        raise ValueError(f"land_indices must have shape {(nbpt, 2)}")
    if incmax < 0:
        raise ValueError("incmax must be non-negative")
    # neighbours, resolution and contfrac are source arguments but lines 18-327 never read them.
    return coordinates, land


def _inside_bbox(lons: np.ndarray, lats: np.ndarray, bbox_lon, bbox_lat) -> bool:
    return (
        ((np.min(lons) > bbox_lon[0] and np.min(lons) < bbox_lon[1])
         or (np.max(lons) > bbox_lon[0] and np.max(lons) < bbox_lon[1]))
        and ((np.min(lats) > bbox_lat[0] and np.min(lats) < bbox_lat[1])
             or (np.max(lats) > bbox_lat[0] and np.max(lats) < bbox_lat[1]))
    )


def _gather_2d(nbpt, incmax, land, state: FindPointsResult) -> Aggregation2DResult:
    indinc = np.full((nbpt, incmax, 2), UNDEFINED_INDEX, dtype=np.int32)
    areaoverlap = np.zeros((nbpt, incmax), dtype=state.overlap_area.dtype)
    for point in range(nbpt):
        i, j = (int(land[point, 0]) - 1, int(land[point, 1]) - 1)
        count = int(state.nbpt[i, j])
        indinc[point, :count, 0] = state.sourcei[i, j, :count]
        indinc[point, :count, 1] = state.sourcej[i, j, :count]
        areaoverlap[point, :count] = state.overlap_area[i, j, :count]
    return Aggregation2DResult(indinc, areaoverlap, True)


def interregxy_aggr2d(
    nbpt: int,
    lalo,
    neighbours,
    resolution,
    contfrac,
    iml: int,
    jml: int,
    lon_rel,
    lat_rel,
    mask,
    callsign: str,
    incmax: int,
    *,
    grid_toij: GridToIJ,
    land_indices,
    dx_wrf,
    dy_wrf,
) -> Aggregation2DResult:
    """Port of ``interregxy_aggr2d`` with module globals made explicit."""

    del callsign
    coordinates, land = _inputs_common(
        nbpt, lalo, neighbours, resolution, contfrac, incmax, land_indices
    )
    lon = np.asarray(lon_rel, dtype=np.float64)
    lat = np.asarray(lat_rel, dtype=np.float64)
    source_mask = np.asarray(mask)
    if lon.shape != (iml, jml) or lat.shape != lon.shape or source_mask.shape != lon.shape:
        raise ValueError(f"lon_rel, lat_rel, and mask must have shape {(iml, jml)}")
    dx, dy = _target_geometry(dx_wrf, dy_wrf)
    nb_we, nb_sn = dx.shape
    state = _findpoint_state(nb_we, nb_sn, incmax, None, None, None, None)
    bbox_lon = (np.min(coordinates[:, 1]), np.max(coordinates[:, 1]))
    bbox_lat = (np.min(coordinates[:, 0]), np.max(coordinates[:, 0]))

    for is0 in range(iml):
        for js0 in range(jml):
            center_lon, center_lat = lon[is0, js0], lat[is0, js0]
            dle = (lon[min(is0 + 1, iml - 1), js0] - center_lon) / 2.0
            dlw = (center_lon - lon[max(is0 - 1, 0), js0]) / 2.0
            if lat[0, 0] < lat[-1, -1]:
                dls = (center_lat - lat[is0, max(js0 - 1, 0)]) / 2.0
                dln = (lat[is0, min(js0 + 1, jml - 1)] - center_lat) / 2.0
            else:
                dls = (center_lat - lat[is0, min(js0 + 1, jml - 1)]) / 2.0
                dln = (lat[is0, max(js0 - 1, 0)] - center_lat) / 2.0
            if dlw < dle / 2.0:
                dlw = dle
            if dle < dlw / 2.0:
                dle = dlw
            if dls < dln / 2.0:
                dls = dln
            if dln < dls / 2.0:
                dln = dls
            lons, lats = _sample_2d_box(center_lon, center_lat, dlw, dle, dls, dln)
            if _inside_bbox(lons, lats, bbox_lon, bbox_lat):
                ri, rj = grid_toij(lons.copy(), lats.copy())
                ri = np.asarray(ri, dtype=np.float64)
                rj = np.asarray(rj, dtype=np.float64)
                ilow = max(int(np.floor(np.min(ri))), 1)
                iup = min(int(np.ceil(np.max(ri))), nb_we)
                jlow = max(int(np.floor(np.min(rj))), 1)
                jup = min(int(np.ceil(np.max(rj))), nb_sn)
                if ilow < iup or jlow < jup:
                    state = interregxy_findpoints(
                        nb_we, nb_sn, center_lon, center_lon, is0 + 1, js0 + 1,
                        ri, rj, incmax, False, *state, dx_wrf=dx, dy_wrf=dy
                    )
    return _gather_2d(nbpt, incmax, land, state)


def _sample_2d_box(center_lon, center_lat, dlw, dle, dls, dln):
    lons = np.empty(4 * (SBD + 1) + 1, dtype=np.result_type(center_lon, dlw, dle))
    lats = np.empty_like(lons)
    lons[0], lats[0] = center_lon, center_lat
    for i in range(SBD + 1):
        lons[1 + i] = center_lon - dlw + i * ((dlw + dle) / (SBD + 1))
        lats[1 + i] = center_lat - dls
        lons[1 + (SBD + 1) + i] = center_lon + dle
        lats[1 + (SBD + 1) + i] = center_lat - dls + i * ((dls + dln) / (SBD + 1))
        lons[1 + 2 * (SBD + 1) + i] = center_lon + dle - i * ((dlw + dle) / (SBD + 1))
        lats[1 + 2 * (SBD + 1) + i] = center_lat + dln
        lons[1 + 3 * (SBD + 1) + i] = center_lon - dlw
        lats[1 + 3 * (SBD + 1) + i] = center_lat + dln - i * ((dls + dln) / (SBD + 1))
    return lons, lats


def interregxy_aggrve(
    nbpt: int,
    lalo,
    neighbours,
    resolution,
    contfrac,
    iml: int,
    lon_rel,
    lat_rel,
    resol_lon: float,
    resol_lat: float,
    callsign: str,
    incmax: int,
    *,
    grid_toij: GridToIJ,
    land_indices,
    dx_wrf,
    dy_wrf,
) -> AggregationVectorResult:
    """Port of ``interregxy_aggrve`` with module globals made explicit."""

    del callsign
    coordinates, land = _inputs_common(
        nbpt, lalo, neighbours, resolution, contfrac, incmax, land_indices
    )
    lon = np.asarray(lon_rel, dtype=np.float64)
    lat = np.asarray(lat_rel, dtype=np.float64)
    if lon.shape != (iml,) or lat.shape != (iml,):
        raise ValueError(f"lon_rel and lat_rel must have shape {(iml,)}")
    dx, dy = _target_geometry(dx_wrf, dy_wrf)
    nb_we, nb_sn = dx.shape
    state = _findpoint_state(nb_we, nb_sn, incmax, None, None, None, None)
    bbox_lon = (np.min(coordinates[:, 1]), np.max(coordinates[:, 1]))
    bbox_lat = (np.min(coordinates[:, 0]), np.max(coordinates[:, 0]))
    trtmp = 180.0 / (2.0 * np.pi)
    half_lon, half_lat = resol_lon / 2.0, resol_lat / 2.0
    dln = dls = half_lat * trtmp / R_EARTH

    for is0 in range(iml):
        lons = np.empty(4 * (SBD + 1) + 1, dtype=np.result_type(lon, np.float64))
        lats = np.empty_like(lons)
        lons[0], lats[0] = lon[is0], lat[is0]
        for i in range(SBD + 1):
            lats[1 + i] = lat[is0] - dls
            dlw = half_lon * trtmp / R_EARTH / np.cos(lats[1 + i] * np.pi / 180.0)
            lons[1 + i] = lon[is0] - dlw + i * ((2.0 * dlw) / (SBD + 1))
            pos = 1 + (SBD + 1) + i
            lats[pos] = lat[is0] - dls + i * ((dls + dln) / (SBD + 1))
            dle = half_lon * trtmp / R_EARTH / np.cos(lats[pos] * np.pi / 180.0)
            lons[pos] = lon[is0] + dle
            pos = 1 + 2 * (SBD + 1) + i
            lats[pos] = lat[is0] + dln
            dle = half_lon * trtmp / R_EARTH / np.cos(lats[pos] * np.pi / 180.0)
            lons[pos] = lon[is0] + dle - i * ((2.0 * dle) / (SBD + 1))
            pos = 1 + 3 * (SBD + 1) + i
            lats[pos] = lat[is0] + dln - i * ((dls + dln) / (SBD + 1))
            dlw = half_lon * trtmp / R_EARTH / np.cos(lats[pos] * np.pi / 180.0)
            lons[pos] = lon[is0] - dlw
        if _inside_bbox(lons, lats, bbox_lon, bbox_lat):
            ri, rj = grid_toij(lons.copy(), lats.copy())
            ri = np.asarray(ri, dtype=np.float64)
            rj = np.asarray(rj, dtype=np.float64)
            ilow = max(int(np.floor(np.min(ri))), 1)
            iup = min(int(np.ceil(np.max(ri))), nb_we)
            jlow = max(int(np.floor(np.min(rj))), 1)
            jup = min(int(np.ceil(np.max(rj))), nb_sn)
            if ilow < iup or jlow < jup:
                state = interregxy_findpoints(
                    nb_we, nb_sn, lon[is0], lon[is0], is0 + 1, 1,
                    ri, rj, incmax, False, *state, dx_wrf=dx, dy_wrf=dy
                )
    gathered = _gather_2d(nbpt, incmax, land, state)
    return AggregationVectorResult(gathered.indinc[:, :, 0], gathered.areaoverlap, True)
