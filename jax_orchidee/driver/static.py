"""Explicit-geometry static input helpers for driver Phase 1E.

These functions are deliberately generic kernels that require their geometry
inputs from the caller. They do not derive paper-case `resolution`,
`neighbours`, source-cell overlaps, or model bboxes from guessed grid metadata.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from jax_orchidee.driver.domain import load_case_config


R_EARTH_ORCHIDEE = 6_378_000.0
MIN_COSLAT_ORCHIDEE = 0.001


def model_bbox_from_lalo_resolution(
    lalo,
    resolution_m,
    *,
    r_earth: float = R_EARTH_ORCHIDEE,
    min_coslat: float = MIN_COSLAT_ORCHIDEE,
) -> np.ndarray:
    """Compute model-cell lon/lat bbox from explicit ORCHIDEE geometry.

    Fortran provenance: `fortran_source/ORCHIDEE/src_sechiba/slowproc.f90`,
    subroutine `slowproc_read_data`, lines 6126-6140, and subroutine
    `slowproc_read_annual`, lines 6404-6418. `lalo(:,1)` is latitude,
    `lalo(:,2)` is longitude, and `resolution(:,1:2)` is in meters in this
    path, matching `grid.f90` lines 697-707 where resolution is computed from
    segment lengths.
    """

    lalo = np.asarray(lalo, dtype=np.float64)
    resolution_m = np.asarray(resolution_m, dtype=np.float64)
    if lalo.ndim != 2 or lalo.shape[1] != 2:
        raise ValueError("lalo must have shape [npts,2] with columns [lat,lon]")
    if resolution_m.shape != lalo.shape:
        raise ValueError("resolution_m must have shape [npts,2] with columns [x,y]")
    if np.any(resolution_m <= 0.0):
        raise ValueError("resolution_m values must be positive explicit geometry")

    lat = lalo[:, 0]
    lon = lalo[:, 1]
    pi = np.pi
    coslat = np.maximum(np.cos(np.deg2rad(lat)), min_coslat) * pi / 180.0 * r_earth
    meridional = pi / 180.0 * r_earth
    half_lon = resolution_m[:, 0] / (2.0 * coslat)
    half_lat = resolution_m[:, 1] / (2.0 * meridional)
    return np.column_stack((lon - half_lon, lon + half_lon, lat - half_lat, lat + half_lat))


def slowproc_nearest_indices(source_lon, source_lat, lalo) -> np.ndarray:
    """Return Fortran ``slowproc_nearest`` source indices for model points.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/slowproc.f90``,
    subroutine ``slowproc_nearest``, lines 4209-4263. The nearest source cell
    maximizes the spherical ``cosang`` expression used by Fortran ``MAXLOC``.
    ``lalo`` columns are ``[lat, lon]``.
    """

    source_lon = np.asarray(source_lon, dtype=np.float64).reshape(-1)
    source_lat = np.asarray(source_lat, dtype=np.float64).reshape(-1)
    lalo = np.asarray(lalo, dtype=np.float64)
    if source_lon.shape != source_lat.shape:
        raise ValueError("source_lon and source_lat must share shape")
    if source_lon.size == 0:
        raise ValueError("slowproc_nearest requires at least one source point")
    if lalo.ndim != 2 or lalo.shape[1] != 2:
        raise ValueError("lalo must have shape [npts,2] with columns [lat,lon]")

    pa = np.pi / 2.0 - np.deg2rad(lalo[:, 0])
    cospa = np.cos(pa)[:, None]
    sinpa = np.sin(pa)[:, None]
    sincolat = np.sin(np.pi / 2.0 - np.deg2rad(source_lat))[None, :]
    coscolat = np.cos(np.pi / 2.0 - np.deg2rad(source_lat))[None, :]
    p = np.deg2rad(lalo[:, 1])[:, None] - np.deg2rad(source_lon)[None, :]
    cosang = cospa * coscolat + sinpa * sincolat * np.cos(p)
    return np.argmax(cosang, axis=1).astype(np.int32)


def haversine_radial_distance(
    lon_start,
    lat_start,
    heading_degrees,
    distance_m,
    *,
    r_earth: float = R_EARTH_ORCHIDEE,
) -> np.ndarray:
    """Project one point along a great-circle heading and distance.

    Fortran provenance: ``src_global/haversine.f90::haversine_radialdis``
    lines 971-992.
    """

    lon1 = np.deg2rad(float(lon_start))
    lat1 = np.deg2rad(float(lat_start))
    tc = np.deg2rad(float(heading_degrees))
    dis = float(distance_m)
    lat = np.arcsin(np.sin(lat1) * np.cos(dis / r_earth) + np.cos(lat1) * np.sin(dis / r_earth) * np.cos(tc))
    dlon = np.arctan2(
        np.sin(tc) * np.sin(dis / r_earth) * np.cos(lat1),
        np.cos(dis / r_earth) - np.sin(lat1) * np.sin(lat),
    )
    lon = np.mod(lon1 + dlon + np.pi, 2.0 * np.pi) - np.pi
    return np.asarray([np.rad2deg(lon), np.rad2deg(lat)], dtype=np.float64)


def paper_single_point_grid_geometry(lalo) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the active paper-case single-point ``grid_stuff`` geometry.

    Fortran provenance: ``src_global/haversine.f90::haversine_singlepointploy``
    lines 394-488 constructs a 111111 m by 111111 m synthetic polygon for the
    ``nland == 1`` path. ``haversine_laloseglen`` lines 667-736 and
    ``haversine_laloarea`` lines 863-888 produce segment lengths, area, and
    the resolution used by ``grid.f90::grid_scatter`` lines 698-707.
    """

    lalo_arr = np.asarray(lalo, dtype=np.float64)
    if lalo_arr.shape != (1, 2):
        raise NotImplementedError("paper single-point geometry currently supports exactly one land point")
    lat = float(lalo_arr[0, 0])
    lon = float(lalo_arr[0, 1])
    half_side = 111111.0 / 2.0
    north = haversine_radial_distance(lon, lat, 0.0, half_side)
    east = haversine_radial_distance(lon, lat, 90.0, half_side)
    south = haversine_radial_distance(lon, lat, 180.0, half_side)
    west = haversine_radial_distance(lon, lat, 270.0, half_side)
    corners = np.asarray(
        [
            [west[0], north[1]],
            [east[0], north[1]],
            [east[0], south[1]],
            [west[0], south[1]],
        ],
        dtype=np.float64,
    )
    seglength = np.asarray(
        [
            abs(corners[0, 0] - corners[1, 0]) * np.pi / 180.0 * R_EARTH_ORCHIDEE * max(np.cos(np.deg2rad(corners[0, 1])), MIN_COSLAT_ORCHIDEE),
            abs(corners[1, 1] - corners[2, 1]) * np.pi / 180.0 * R_EARTH_ORCHIDEE,
            abs(corners[2, 0] - corners[3, 0]) * np.pi / 180.0 * R_EARTH_ORCHIDEE * max(np.cos(np.deg2rad(corners[2, 1])), MIN_COSLAT_ORCHIDEE),
            abs(corners[3, 1] - corners[0, 1]) * np.pi / 180.0 * R_EARTH_ORCHIDEE,
        ],
        dtype=np.float64,
    )
    resolution = np.asarray([[(seglength[0] + seglength[2]) / 2.0, (seglength[1] + seglength[3]) / 2.0]], dtype=np.float64)
    area = np.asarray([resolution[0, 0] * resolution[0, 1]], dtype=np.float64)
    return resolution, np.full((1, 8), -1, dtype=np.int32), area


def _haversine_heading(lon_start, lat_start, lon_end, lat_end) -> float:
    dlon = np.deg2rad(float(lon_end) - float(lon_start))
    lat1 = np.deg2rad(float(lat_start))
    lat2 = np.deg2rad(float(lat_end))
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return float(np.mod(np.rad2deg(np.arctan2(y, x)) + 360.0, 360.0))


def _haversine_clockwise_indices(headings, *, nbseg: int = 4, start: float = 0.0) -> np.ndarray:
    """Return Fortran `haversine_clockwise` zero-based ordering."""

    heading = np.asarray(headings, dtype=np.float64)
    if heading.shape != (nbseg * 2,):
        raise ValueError("heading must have length nbseg*2")
    delta = 360.0 / float(nbseg * 2)
    work = heading.copy()
    out = np.zeros(nbseg * 2, dtype=np.int64)
    undef = 9999999999.99999
    for iseg in range(nbseg * 2):
        for js in range(nbseg * 2):
            if work[js] < undef:
                work[js] = np.mod(heading[js] - (float(start) + iseg * delta) + 360.0, 360.0)
                if work[js] > 180.0:
                    work[js] = work[js] - 360.0
        imin = int(np.argmin(np.abs(work)))
        out[iseg] = imin
        work[imin] = undef
    return out


def _haversine_laloseglen_from_sorted_polygon(lonpoly, latpoly, *, nbseg: int = 4) -> np.ndarray:
    lonpoly = np.asarray(lonpoly, dtype=np.float64)
    latpoly = np.asarray(latpoly, dtype=np.float64)
    slpm1 = _haversine_heading(lonpoly[0], latpoly[0], lonpoly[nbseg * 2 - 1], latpoly[nbseg * 2 - 1])
    slpp1 = _haversine_heading(lonpoly[0], latpoly[0], lonpoly[1], latpoly[1])
    ioff = -1 if abs(np.mod(slpp1 - slpm1, 360.0)) > 135.0 else 0
    out = np.zeros(nbseg, dtype=np.float64)
    for iseg, iv in enumerate(range(0, nbseg * 2, 2)):
        istart = (iv + ioff) % (nbseg * 2)
        iend = (iv + ioff + 2) % (nbseg * 2)
        if abs(lonpoly[istart] - lonpoly[iend]) < np.finfo(np.float64).eps:
            out[iseg] = abs(latpoly[istart] - latpoly[iend]) * np.pi / 180.0 * R_EARTH_ORCHIDEE
        elif abs(latpoly[istart] - latpoly[iend]) < np.finfo(np.float64).eps:
            coslat = max(np.cos(np.deg2rad(latpoly[istart])), MIN_COSLAT_ORCHIDEE)
            out[iseg] = abs(lonpoly[istart] - lonpoly[iend]) * np.pi / 180.0 * R_EARTH_ORCHIDEE * coslat
        else:
            raise ValueError("regular lon/lat polygon segment is not zonal or meridional")
    return out


def regular_lonlat_grid_geometry(
    lon,
    lat,
    kindex,
    *,
    global_grid: bool | None = None,
    return_corners: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build `grid_stuff` geometry for the regular lon/lat multi-point path.

    Fortran provenance: `src_global/grid.f90::grid_topolylist` lines 583-620
    calls `haversine_reglatlontoploy`, `haversine_polyheadings`,
    `haversine_polysort`, `haversine_laloseglen`, and
    `haversine_laloarea` for `nland > 1` regular lon/lat grids.
    `grid_scatter` lines 688-707 then exposes sorted neighbours, area, and
    `resolution(:,1:2) = ((seg1+seg3)/2, (seg2+seg4)/2)`.
    """

    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    kindex = np.asarray(kindex, dtype=np.int64)
    if lon.shape != lat.shape or lon.ndim != 2:
        raise ValueError("lon and lat must be matching 2-D regular-grid arrays")
    iim, jjm = lon.shape
    if kindex.ndim != 1 or kindex.size < 1:
        raise ValueError("kindex must be a non-empty one-dimensional Fortran index vector")
    if np.any(kindex < 1) or np.any(kindex > iim * jjm):
        raise ValueError("kindex values must be inside the local lon/lat grid")
    if kindex.size == 1:
        lalo = np.asarray([[lat[(kindex[0] - 1) % iim, (kindex[0] - 1) // iim], lon[(kindex[0] - 1) % iim, (kindex[0] - 1) // iim]]])
        resolution, neighbours, area = paper_single_point_grid_geometry(lalo)
        seglength = np.asarray([[resolution[0, 0], resolution[0, 1], resolution[0, 0], resolution[0, 1]]], dtype=np.float64)
        if return_corners:
            lat0 = float(lalo[0, 0])
            lon0 = float(lalo[0, 1])
            half_side = 111111.0 / 2.0
            north = haversine_radial_distance(lon0, lat0, 0.0, half_side)
            east = haversine_radial_distance(lon0, lat0, 90.0, half_side)
            south = haversine_radial_distance(lon0, lat0, 180.0, half_side)
            west = haversine_radial_distance(lon0, lat0, 270.0, half_side)
            corners = np.asarray(
                [[[west[0], north[1]], [east[0], north[1]], [east[0], south[1]], [west[0], south[1]]]],
                dtype=np.float64,
            )
            return resolution, neighbours, area, seglength, corners
        return resolution, neighbours, area, seglength
    if iim < 2 or jjm < 2:
        raise ValueError("regular lon/lat multi-point geometry needs at least two longitudes and two latitudes")

    if global_grid is None:
        lon_axis = lon[:, 0]
        max_dellon = float(np.max(np.abs(lon_axis[:-1] - lon_axis[1:])))
        maxlon = float(np.max(lon_axis))
        minlon = float(np.min(lon_axis))
        if minlon > 0.0 and maxlon > 180.0:
            global_grid = (minlon - max_dellon / 2.0) <= 0.0 and (maxlon + max_dellon / 2.0) >= 360.0
        elif minlon < 0.0 and maxlon > 0.0:
            global_grid = (minlon - max_dellon / 2.0) <= -180.0 and (maxlon + max_dellon / 2.0) >= 180.0
        else:
            global_grid = False

    nbpt = int(kindex.size)
    correspondance = np.full((iim, jjm), -1, dtype=np.int32)
    iorig = np.zeros(nbpt, dtype=np.int64)
    jorig = np.zeros(nbpt, dtype=np.int64)
    for land, index in enumerate(kindex):
        jp = int((int(index) - 1) // iim)
        ip = int(int(index) - 1 - jp * iim)
        correspondance[ip, jp] = land + 1
        iorig[land] = ip
        jorig[land] = jp

    resolution = np.zeros((nbpt, 2), dtype=np.float64)
    area = np.zeros(nbpt, dtype=np.float64)
    seglength = np.zeros((nbpt, 4), dtype=np.float64)
    corners = np.zeros((nbpt, 4, 2), dtype=np.float64)
    neighbours = np.full((nbpt, 8), -1, dtype=np.int32)
    undef = 999999.0
    for land in range(nbpt):
        ip = int(iorig[land])
        jp = int(jorig[land])
        ipm1 = ip - 1
        ipp1 = ip + 1
        jpm1 = jp - 1
        jpp1 = jp + 1

        if ipp1 < iim:
            dlonp1 = (lon[ipp1, jp] - lon[ip, jp]) / 2.0
        elif ipm1 >= 0:
            dlonp1 = (lon[ip, jp] - lon[ipm1, jp]) / 2.0
            if global_grid:
                ipp1 = 0
        else:
            dlonp1 = undef
        if ipm1 >= 0:
            dlonm1 = (lon[ip, jp] - lon[ipm1, jp]) / 2.0
        elif ipp1 < iim:
            dlonm1 = (lon[ipp1, jp] - lon[ip, jp]) / 2.0
            if global_grid:
                ipm1 = iim - 1
        else:
            dlonm1 = undef
        if dlonp1 >= undef - 1.0:
            dlonp1 = dlonm1
        if dlonm1 >= undef - 1.0:
            dlonm1 = dlonp1
        if dlonp1 >= undef - 1.0 and dlonm1 >= undef - 1.0:
            raise ValueError("not enough longitude points to estimate grid-box bounds")

        if jpp1 < jjm:
            dlatp1 = (lat[ip, jpp1] - lat[ip, jp]) / 2.0
        elif jpm1 >= 0:
            dlatp1 = (lat[ip, jp] - lat[ip, jpm1]) / 2.0
        else:
            dlatp1 = undef
        if jpm1 >= 0:
            dlatm1 = (lat[ip, jp] - lat[ip, jpm1]) / 2.0
        elif jpp1 < jjm:
            dlatm1 = (lat[ip, jpp1] - lat[ip, jp]) / 2.0
        else:
            dlatm1 = undef
        if dlatp1 >= undef - 1.0:
            dlatp1 = dlatm1
        if dlatm1 >= undef - 1.0:
            dlatm1 = dlatp1
        if dlatp1 >= undef - 1.0 and dlatm1 >= undef - 1.0:
            raise ValueError("not enough latitude points to estimate grid-box bounds")

        lonpoly = np.asarray(
            [
                lon[ip, jp] - dlonm1,
                lon[ip, jp],
                lon[ip, jp] + dlonp1,
                lon[ip, jp] + dlonp1,
                lon[ip, jp] + dlonp1,
                lon[ip, jp],
                lon[ip, jp] - dlonm1,
                lon[ip, jp] - dlonm1,
            ],
            dtype=np.float64,
        )
        latpoly = np.asarray(
            [
                lat[ip, jp] - dlatp1,
                lat[ip, jp] - dlatp1,
                lat[ip, jp] - dlatp1,
                lat[ip, jp],
                lat[ip, jp] + dlatm1,
                lat[ip, jp] + dlatm1,
                lat[ip, jp] + dlatm1,
                lat[ip, jp],
            ],
            dtype=np.float64,
        )
        corners[land] = np.asarray(
            [
                [lonpoly[0], latpoly[0]],
                [lonpoly[2], latpoly[2]],
                [lonpoly[4], latpoly[4]],
                [lonpoly[6], latpoly[6]],
            ],
            dtype=np.float64,
        )
        neigh = np.full(8, -1, dtype=np.int32)
        if ipm1 >= 0 and jpm1 >= 0:
            neigh[0] = correspondance[ipm1, jpm1]
        if jpm1 >= 0:
            neigh[1] = correspondance[ip, jpm1]
        if ipp1 < iim and jpm1 >= 0:
            neigh[2] = correspondance[ipp1, jpm1]
        if ipp1 < iim:
            neigh[3] = correspondance[ipp1, jp]
        if ipp1 < iim and jpp1 < jjm:
            neigh[4] = correspondance[ipp1, jpp1]
        if jpp1 < jjm:
            neigh[5] = correspondance[ip, jpp1]
        if ipm1 >= 0 and jpp1 < jjm:
            neigh[6] = correspondance[ipm1, jpp1]
        if ipm1 >= 0:
            neigh[7] = correspondance[ipm1, jp]

        headings = np.asarray(
            [
                np.mod(_haversine_heading(lonpoly[idx], latpoly[idx], lon[ip, jp], lat[ip, jp]) + 180.0, 360.0)
                for idx in range(8)
            ],
            dtype=np.float64,
        )
        order = _haversine_clockwise_indices(headings)
        lon_sorted = lonpoly[order]
        lat_sorted = latpoly[order]
        neigh_sorted = neigh[order]
        seg = _haversine_laloseglen_from_sorted_polygon(lon_sorted, lat_sorted)
        seglength[land] = seg
        neighbours[land] = neigh_sorted
        resolution[land, 0] = (seg[0] + seg[2]) / 2.0
        resolution[land, 1] = (seg[1] + seg[3]) / 2.0
        area[land] = resolution[land, 0] * resolution[land, 1]
    if return_corners:
        return resolution, neighbours, area, seglength, corners
    return resolution, neighbours, area, seglength


def bbox_center_mean(
    lalo,
    resolution_m,
    source_lon,
    source_lat,
    values,
    *,
    valid_mask=None,
    allow_nearest_fallback: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Average source center points that fall inside each explicit model bbox.

    Fortran provenance: `slowproc.f90`, subroutine `slowproc_read_data`, lines
    6109-6117 flatten source centers, 6126-6140 build bboxes, 6182-6239 count
    valid source centers inside `lon_low <= lon < lon_up` and
    `lat_low <= lat < lat_up`, 6261-6263 divide by the count, and 6270-6274
    call ``slowproc_nearest`` if no source center is found.
    `slowproc_read_annual`, lines 6382-6390, 6404-6418, 6461-6512, and
    6533-6546 use the same center-point mean/fallback pattern for annual
    scalar fields.

    By default this helper raises if a target bbox contains no valid source
    centers. Set ``allow_nearest_fallback=True`` only for callers following the
    Fortran fallback branch explicitly; those targets keep ``counts == 0``.
    """

    bbox = model_bbox_from_lalo_resolution(lalo, resolution_m)
    lon_grid, lat_grid = np.meshgrid(np.asarray(source_lon, dtype=np.float64), np.asarray(source_lat, dtype=np.float64), indexing="ij")
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.shape[:2] != lon_grid.shape:
        raise ValueError("values must have shape [nlon,nlat,...]")

    flat_lon = lon_grid.reshape(-1).copy()
    nearest_lon = flat_lon.copy()
    flat_lat = lat_grid.reshape(-1)
    flat_values = values_arr.reshape((-1, *values_arr.shape[2:]))
    if valid_mask is None:
        valid = np.all(flat_values > -9000.0, axis=tuple(range(1, flat_values.ndim))) if flat_values.ndim > 1 else flat_values > -9000.0
    else:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid.shape != flat_lon.shape:
            raise ValueError("valid_mask must have shape [nlon,nlat]")

    lon_min = float(np.min(bbox[:, 0]))
    lon_max = float(np.max(bbox[:, 1]))
    flat_lon[flat_lon < lon_min] += 360.0
    flat_lon[flat_lon > lon_max] -= 360.0

    out_shape = (bbox.shape[0], *flat_values.shape[1:])
    means = np.zeros(out_shape, dtype=np.float64)
    counts = np.zeros(bbox.shape[0], dtype=np.int32)
    nearest_indices = None
    for ib, (lon_low, lon_up, lat_low, lat_up) in enumerate(bbox):
        inside = (
            valid
            & (flat_lon >= lon_low)
            & (flat_lon < lon_up)
            & (flat_lat >= lat_low)
            & (flat_lat < lat_up)
        )
        count = int(np.count_nonzero(inside))
        if count == 0:
            if not allow_nearest_fallback:
                raise ValueError(f"no valid source centers found inside bbox for target {ib}")
            if nearest_indices is None:
                nearest_indices = slowproc_nearest_indices(nearest_lon, flat_lat, lalo)
            means[ib] = flat_values[int(nearest_indices[ib])]
            continue
        counts[ib] = count
        means[ib] = np.mean(flat_values[inside], axis=0)
    return means, counts


def _fortran_lon_lat_axes_from_1d(source_lon, source_lat) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(source_lon, dtype=np.float64)
    lat = np.asarray(source_lat, dtype=np.float64)
    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("source lon/lat coordinates must be one-dimensional")
    return lon, lat


def read_slowproc_annual_bbox_field(
    path: str | Path,
    *,
    variable: str,
    lalo,
    resolution_m,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a scalar annual field through ``slowproc_read_annual`` bbox means.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/slowproc.f90``,
    subroutine ``slowproc_read_annual``, lines 6300-6570. This covers the
    source-center counting, mean, and nearest fallback branch used by salinity
    and other annual scalar inputs.
    """

    with xr.open_dataset(path, decode_times=False) as ds:
        source_lon, source_lat = _fortran_lon_lat_axes_from_1d(ds["lon"].values, ds["lat"].values)
        values = np.asarray(ds[variable].isel(time_counter=0).values, dtype=np.float64).T
    means, counts = bbox_center_mean(
        lalo,
        resolution_m,
        source_lon,
        source_lat,
        values,
        allow_nearest_fallback=True,
    )
    return means.reshape(-1), counts


def read_slowproc_time_bbox_field(
    path: str | Path,
    *,
    variable: str,
    lalo,
    resolution_m,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a time-varying field through ``slowproc_read_data`` bbox means.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/slowproc.f90``,
    subroutine ``slowproc_read_data``, lines 6029-6297. This covers the mean
    branch over all ``tml`` time slots used by the paper tide input.
    """

    with xr.open_dataset(path, decode_times=False) as ds:
        source_lon, source_lat = _fortran_lon_lat_axes_from_1d(ds["lon"].values, ds["lat"].values)
        values = np.asarray(ds[variable].values, dtype=np.float64).transpose(2, 1, 0)
    return bbox_center_mean(
        lalo,
        resolution_m,
        source_lon,
        source_lat,
        values,
        allow_nearest_fallback=True,
    )


def read_paper_salinity_tide_static_fields(
    config_path: str | Path,
    *,
    lalo,
    resolution_m,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Read paper-case salinity and tide static fields from local input files."""

    config = load_case_config(config_path)
    salinity, salinity_count = read_slowproc_annual_bbox_field(
        config["drivers"]["salinity"]["file"],
        variable=config["drivers"]["salinity"]["variable"],
        lalo=lalo,
        resolution_m=resolution_m,
    )
    tide_height, tide_count = read_slowproc_time_bbox_field(
        config["drivers"]["tide"]["file"],
        variable=config["drivers"]["tide"]["variable"],
        lalo=lalo,
        resolution_m=resolution_m,
    )
    return salinity, tide_height, {
        "salinity_source_count": salinity_count,
        "tide_source_count": tide_count,
    }


USDA_TEXTFRAC_TABLE = np.asarray(
    [
        [0.04, 0.93, 0.03],
        [0.13, 0.81, 0.06],
        [0.26, 0.63, 0.11],
        [0.64, 0.17, 0.19],
        [0.84, 0.06, 0.10],
        [0.40, 0.40, 0.20],
        [0.19, 0.54, 0.27],
        [0.59, 0.08, 0.33],
        [0.37, 0.30, 0.33],
        [0.11, 0.48, 0.41],
        [0.48, 0.06, 0.46],
        [0.30, 0.15, 0.55],
    ],
    dtype=np.float64,
)
ZOBLER_TEXTFRAC_TABLE = np.asarray(
    [
        [0.12, 0.82, 0.06],
        [0.32, 0.58, 0.10],
        [0.39, 0.43, 0.18],
        [0.15, 0.58, 0.27],
        [0.34, 0.32, 0.34],
        [0.00, 1.00, 0.00],
        [0.39, 0.43, 0.18],
    ],
    dtype=np.float64,
)
SOILCLASS_DEFAULT_FAO = np.asarray([0.28, 0.52, 0.20], dtype=np.float64)
SOILCLASS_DEFAULT_USDA = np.asarray([0.28, 0.52, 0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def aggregate_2d_overlap(
    lalo,
    resolution_m,
    source_lon,
    source_lat,
    mask,
    *,
    incmax: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate active ``interpol_help.f90::aggregate_2d`` area overlaps.

    Fortran provenance: ``src_global/interpol_help.f90::aggregate_2d`` lines
    46-466. The active paper soil path uses the regular 2D map branch:
    source-cell bounds are built from neighboring lon/lat centers (lines
    149-188), the regional source search uses model ``lalo``/``resolution``
    (lines 214-260), and overlap areas are computed at lines 296-389. This
    helper implements that branch only; vector and RegXY initialized branches
    remain unsupported.
    """

    lalo = np.asarray(lalo, dtype=np.float64)
    resolution_m = np.asarray(resolution_m, dtype=np.float64)
    lon_rel = np.asarray(source_lon, dtype=np.float64)
    lat_rel = np.asarray(source_lat, dtype=np.float64)
    mask_arr = np.asarray(mask, dtype=bool)
    if lalo.ndim != 2 or lalo.shape[1] != 2:
        raise ValueError("lalo must have shape [npts,2]")
    if resolution_m.shape != lalo.shape:
        raise ValueError("resolution_m must have shape [npts,2]")
    if lon_rel.shape != lat_rel.shape or lon_rel.ndim != 2:
        raise ValueError("source_lon/source_lat must be 2D arrays with matching shape")
    if mask_arr.shape != lon_rel.shape:
        raise ValueError("mask must match source grid shape")

    iml, jml = lon_rel.shape
    lon_ful = np.zeros((iml + 2, jml + 2), dtype=np.float64)
    lat_ful = np.zeros((iml + 2, jml + 2), dtype=np.float64)
    lon_ful[1 : iml + 1, 1 : jml + 1] = lon_rel
    lat_ful[1 : iml + 1, 1 : jml + 1] = lat_rel

    if lon_rel[-1, 0] < lon_ful[1, 1]:
        lon_ful[0, 1 : jml + 1] = lon_rel[-1, :]
    else:
        lon_ful[0, 1 : jml + 1] = lon_rel[-1, :] - 360.0
    lat_ful[0, 1 : jml + 1] = lat_rel[-1, :]

    if lon_rel[0, 0] > lon_ful[iml, 1]:
        lon_ful[iml + 1, 1 : jml + 1] = lon_rel[0, :]
    else:
        lon_ful[iml + 1, 1 : jml + 1] = lon_rel[0, :] + 360.0
    lat_ful[iml + 1, 1 : jml + 1] = lat_rel[0, :]

    lat_resol = lat_rel[0, 1] - lat_rel[0, 0]
    lat_ful[1 : iml + 1, 0] = lat_rel[:, 0] - lat_resol
    lat_ful[1 : iml + 1, jml + 1] = lat_rel[:, -1] + lat_resol
    lat_ful[0, 0] = lat_ful[iml, 0]
    lat_ful[iml + 1, 0] = lat_ful[1, 0]
    lat_ful[0, jml + 1] = lat_ful[iml, jml + 1]
    lat_ful[iml + 1, jml + 1] = lat_ful[1, jml + 1]

    lon_resol = lon_ful[2, 1] - lon_ful[1, 1]
    lon_ful[:, 0] = lon_ful[:, 1]
    lon_ful[:, jml + 1] = lon_ful[:, jml]
    lon_ful[0, :] = lon_ful[1, 0] - lon_resol
    lon_ful[iml + 1, :] = lon_ful[iml, 0] + lon_resol

    loup_rel = np.zeros_like(lon_rel)
    lolow_rel = np.zeros_like(lon_rel)
    laup_rel = np.zeros_like(lat_rel)
    lalow_rel = np.zeros_like(lat_rel)
    for ip in range(iml):
        for jp in range(jml):
            west_mid = 0.5 * (lon_ful[ip, jp + 1] + lon_ful[ip + 1, jp + 1])
            east_mid = 0.5 * (lon_ful[ip + 1, jp + 1] + lon_ful[ip + 2, jp + 1])
            loup_rel[ip, jp] = max(west_mid, east_mid)
            lolow_rel[ip, jp] = min(west_mid, east_mid)
            south_mid = 0.5 * (lat_ful[ip + 1, jp] + lat_ful[ip + 1, jp + 1])
            north_mid = 0.5 * (lat_ful[ip + 1, jp + 1] + lat_ful[ip + 1, jp + 2])
            laup_rel[ip, jp] = max(south_mid, north_mid)
            lalow_rel[ip, jp] = min(south_mid, north_mid)

    npts = lalo.shape[0]
    max_lon_idx = int(np.argmax(lalo[:, 1]))
    min_lon_idx = int(np.argmin(lalo[:, 1]))
    lon_scale_min = max(np.cos(np.deg2rad(lalo[min_lon_idx, 0])), MIN_COSLAT_ORCHIDEE) * np.pi / 180.0 * R_EARTH_ORCHIDEE
    lon_scale_max = max(np.cos(np.deg2rad(lalo[max_lon_idx, 0])), MIN_COSLAT_ORCHIDEE) * np.pi / 180.0 * R_EARTH_ORCHIDEE
    domain_minlon = lalo[min_lon_idx, 1] - resolution_m[min_lon_idx, 0] / (2.0 * lon_scale_min)
    domain_maxlon = lalo[max_lon_idx, 1] + resolution_m[max_lon_idx, 0] / (2.0 * lon_scale_max)
    lat_scale = np.pi / 180.0 * R_EARTH_ORCHIDEE
    domain_minlat = float(np.min(lalo[:, 0])) - resolution_m[max_lon_idx, 1] / (2.0 * lat_scale)
    domain_maxlat = float(np.max(lalo[:, 0])) + resolution_m[max_lon_idx, 1] / (2.0 * lat_scale)

    search = [
        (ip, jp)
        for jp in range(jml)
        for ip in range(iml)
        if (
            loup_rel[ip, jp] >= domain_minlon
            and lolow_rel[ip, jp] <= domain_maxlon
            and laup_rel[ip, jp] >= domain_minlat
            and lalow_rel[ip, jp] <= domain_maxlat
            and mask_arr[ip, jp]
        )
    ]

    found_indices: list[list[tuple[int, int]]] = [[] for _ in range(npts)]
    found_areas: list[list[float]] = [[] for _ in range(npts)]
    for ib in range(npts):
        lon_scale = max(np.cos(np.deg2rad(lalo[ib, 0])), MIN_COSLAT_ORCHIDEE) * np.pi / 180.0 * R_EARTH_ORCHIDEE
        lon_up = lalo[ib, 1] + resolution_m[ib, 0] / (2.0 * lon_scale)
        lon_low = lalo[ib, 1] - resolution_m[ib, 0] / (2.0 * lon_scale)
        lat_up = lalo[ib, 0] + resolution_m[ib, 1] / (2.0 * lat_scale)
        lat_low = lalo[ib, 0] - resolution_m[ib, 1] / (2.0 * lat_scale)
        for ip, jp in search:
            if lon_low >= -180.0 and lon_up <= 180.0:
                lonrel = lon_rel[ip, jp]
                lolowrel = lolow_rel[ip, jp]
                louprel = loup_rel[ip, jp]
            elif lon_low < -180.0:
                lonrel = np.mod(lon_rel[ip, jp] - 360.0, 360.0)
                lolowrel = np.mod(lolow_rel[ip, jp] - 360.0, 360.0)
                louprel = np.mod(loup_rel[ip, jp] - 360.0, 360.0)
            elif lon_up > 180.0:
                lonrel = np.mod(lon_rel[ip, jp] + 360.0, 360.0)
                lolowrel = np.mod(lolow_rel[ip, jp] + 360.0, 360.0)
                louprel = np.mod(loup_rel[ip, jp] + 360.0, 360.0)
            else:
                raise AssertionError("unreachable longitude branch")
            lon_overlap = (
                (lonrel > lon_low and lonrel < lon_up)
                or (lolowrel < lon_low and louprel > lon_low)
                or (lolowrel < lon_up and louprel > lon_up)
            )
            lat_overlap = (
                (lat_rel[ip, jp] > lat_low and lat_rel[ip, jp] < lat_up)
                or (lalow_rel[ip, jp] < lat_low and laup_rel[ip, jp] > lat_low)
                or (lalow_rel[ip, jp] < lat_up and laup_rel[ip, jp] > lat_up)
            )
            if lon_overlap and lat_overlap:
                coslat = max(np.cos(np.deg2rad(lat_rel[ip, jp])), MIN_COSLAT_ORCHIDEE)
                ax = (min(lon_up, louprel) - max(lon_low, lolowrel)) * np.pi / 180.0 * R_EARTH_ORCHIDEE * coslat
                ay = (min(lat_up, laup_rel[ip, jp]) - max(lat_low, lalow_rel[ip, jp])) * np.pi / 180.0 * R_EARTH_ORCHIDEE
                found_indices[ib].append((ip + 1, jp + 1))
                found_areas[ib].append(ax * ay)
    width = incmax if incmax is not None else max((len(item) for item in found_indices), default=0)
    sub_index = np.zeros((npts, width, 2), dtype=np.int32)
    sub_area = np.zeros((npts, width), dtype=np.float64)
    for ib in range(npts):
        if len(found_indices[ib]) > width:
            raise ValueError("incmax too small for overlap result")
        for idx, ((ip, jp), area) in enumerate(zip(found_indices[ib], found_areas[ib], strict=True)):
            sub_index[ib, idx] = (ip, jp)
            sub_area[ib, idx] = area
    return sub_index, sub_area


def weighted_mean_from_explicit_overlap(source_values, sub_index, sub_area, *, index_base: int = 1) -> np.ndarray:
    """Compute weighted means from explicit `aggregate_p` overlap outputs.

    Fortran provenance: `interpol_help.f90`, subroutine `aggregate_2d_p`,
    lines 830-889, returns `sub_index(nbpt,nbvmax,2)` and
    `sub_area(nbpt,nbvmax)` from `aggregate_2d`; `slowproc.f90`,
    `slowproc_soilt`, lines 4612-4663 and 4686-4692, consumes those explicit
    overlaps as area weights before normalizing by their summed area.
    """

    source_values = np.asarray(source_values, dtype=np.float64)
    sub_index = np.asarray(sub_index, dtype=np.int64)
    sub_area = np.asarray(sub_area, dtype=np.float64)
    if sub_index.ndim != 3 or sub_index.shape[2] != 2:
        raise ValueError("sub_index must have shape [npts,nbvmax,2]")
    if sub_area.shape != sub_index.shape[:2]:
        raise ValueError("sub_area must have shape [npts,nbvmax]")
    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 or 1")

    trailing_shape = source_values.shape[2:]
    out = np.zeros((sub_index.shape[0], *trailing_shape), dtype=np.float64)
    weights = np.zeros(sub_index.shape[0], dtype=np.float64)
    for ib in range(sub_index.shape[0]):
        for ilf in range(sub_index.shape[1]):
            area = float(sub_area[ib, ilf])
            if area <= 0.0:
                continue
            ip = int(sub_index[ib, ilf, 0]) - index_base
            jp = int(sub_index[ib, ilf, 1]) - index_base
            if ip < 0 or jp < 0:
                continue
            out[ib] += source_values[ip, jp] * area
            weights[ib] += area
    if np.any(weights <= 0.0):
        raise ValueError("each target point needs positive explicit overlap area")
    return out / weights.reshape((-1, *([1] * len(trailing_shape))))


def thermosoil_refsoc_from_explicit_overlap(source_values, sub_index, sub_area, *, index_base: int = 1) -> np.ndarray:
    """Average 3D ``refSOC`` source cells with the ``read_refSOCfile`` fallback.

    Fortran provenance: ``thermosoil.f90::read_refSOCfile`` lines 3374-3392.
    Positive ``sub_area`` rows are area-weighted and normalized by ``totarea``;
    if a target point has no positive overlap, Fortran sets ``refSOC(ib,:)`` to
    zero instead of using a climatological mean.
    """

    source_values = np.asarray(source_values, dtype=np.float64)
    sub_index = np.asarray(sub_index, dtype=np.int64)
    sub_area = np.asarray(sub_area, dtype=np.float64)
    if sub_index.ndim != 3 or sub_index.shape[2] != 2:
        raise ValueError("sub_index must have shape [npts,nbvmax,2]")
    if sub_area.shape != sub_index.shape[:2]:
        raise ValueError("sub_area must have shape [npts,nbvmax]")
    if source_values.ndim != 3:
        raise ValueError("source_values must have shape [nlon,nlat,ngrnd]")
    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 or 1")

    out = np.zeros((sub_index.shape[0], source_values.shape[2]), dtype=np.float64)
    weights = np.zeros(sub_index.shape[0], dtype=np.float64)
    for ib in range(sub_index.shape[0]):
        for ilf in range(sub_index.shape[1]):
            area = float(sub_area[ib, ilf])
            if area <= 0.0:
                continue
            ip = int(sub_index[ib, ilf, 0]) - index_base
            jp = int(sub_index[ib, ilf, 1]) - index_base
            if ip < 0 or jp < 0:
                continue
            out[ib] += source_values[ip, jp, :] * area
            weights[ib] += area
        if weights[ib] > 0.0:
            out[ib] /= weights[ib]
    return out


def hydrol_refsoc_1d_from_explicit_overlap(source_values, sub_index, sub_area, *, index_base: int = 1) -> np.ndarray:
    """Average 2D ``refSOC_1d`` source cells with HYDROL's zero fallback.

    Fortran provenance: ``hydrol.f90::read_refSOC_1dfile`` lines 9721-9847
    reads ``SOIL_REFSOC_1d_FILE`` variables ``longitude``, ``latitude``,
    ``mask``, and ``soil_organic_carbon_1d``; lines 9806-9828 call
    ``aggregate_p``; lines 9830-9844 compute the area-weighted mean or set
    ``refSOC_1d(ib)=0`` when no source cell overlaps the target point.
    """

    source_values = np.asarray(source_values, dtype=np.float64)
    sub_index = np.asarray(sub_index, dtype=np.int64)
    sub_area = np.asarray(sub_area, dtype=np.float64)
    if source_values.ndim != 2:
        raise ValueError("source_values must have shape [nlon,nlat]")
    if sub_index.ndim != 3 or sub_index.shape[2] != 2:
        raise ValueError("sub_index must have shape [npts,nbvmax,2]")
    if sub_area.shape != sub_index.shape[:2]:
        raise ValueError("sub_area must have shape [npts,nbvmax]")
    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 or 1")

    out = np.zeros(sub_index.shape[0], dtype=np.float64)
    weights = np.zeros(sub_index.shape[0], dtype=np.float64)
    for ib in range(sub_index.shape[0]):
        for ilf in range(sub_index.shape[1]):
            area = float(sub_area[ib, ilf])
            if area <= 0.0:
                continue
            ip = int(sub_index[ib, ilf, 0]) - index_base
            jp = int(sub_index[ib, ilf, 1]) - index_base
            if ip < 0 or jp < 0:
                continue
            out[ib] += source_values[ip, jp] * area
            weights[ib] += area
        if weights[ib] > 0.0:
            out[ib] /= weights[ib]
    return out


def read_paper_thermosoil_refsoc_static_field(
    config_path: str | Path,
    *,
    lalo,
    resolution_m,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read paper-case ``refSOC`` through THERMOSOIL ``read_refSOCfile``.

    Fortran provenance: ``thermosoil.f90::thermosoil_initialize`` lines
    563-565 calls ``read_refSOCfile`` when restart ``refSOC`` is absent and
    ``use_refSOC`` is true. ``read_refSOCfile`` lines 3278-3318 read
    ``SOIL_REFSOC_FILE`` variables ``longitude``, ``latitude``, ``mask``, and
    ``soil_organic_carbon``; lines 3330-3369 call ``aggregate_p``; lines
    3374-3392 compute the area-weighted profile or zero fallback.
    """

    config = load_case_config(config_path)
    path = Path(config["static_inputs"]["refsoc_3d_file"])
    with xr.open_dataset(path, decode_times=False) as ds:
        lon_1d = np.asarray(ds["longitude"].values, dtype=np.float64)
        lat_1d = np.asarray(ds["latitude"].values, dtype=np.float64)
        lon, lat = np.meshgrid(lon_1d, lat_1d, indexing="xy")
        lon = lon.T
        lat = lat.T
        mask = np.asarray(ds["mask"].values, dtype=np.float64).T > 0.0
        refsoc_source = np.asarray(ds["soil_organic_carbon"].isel(time_counter=0).values, dtype=np.float64).transpose(2, 1, 0)
    sub_index, sub_area = aggregate_2d_overlap(lalo, resolution_m, lon, lat, mask)
    refsoc = thermosoil_refsoc_from_explicit_overlap(refsoc_source, sub_index, sub_area)
    return refsoc, {"refsoc_sub_index": sub_index, "refsoc_sub_area": sub_area}


def read_paper_hydrol_refsoc_1d_static_field(
    config_path: str | Path,
    *,
    lalo,
    resolution_m,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read paper-case HYDROL ``refSOC_1d`` through ``read_refSOC_1dfile``.

    Fortran provenance: ``hydrol.f90::hydrol_initialize`` lines 625-636 calls
    ``read_refSOC_1dfile`` when restart ``refSOC_1d`` is absent and
    ``use_refSOC_hydrol`` is true. ``read_refSOC_1dfile`` lines 9721-9847
    perform the static file read and area-weighted overlap.
    """

    config = load_case_config(config_path)
    path = Path(config["static_inputs"]["refsoc_1d_file"])
    with xr.open_dataset(path, decode_times=False) as ds:
        lon_1d = np.asarray(ds["longitude"].values, dtype=np.float64)
        lat_1d = np.asarray(ds["latitude"].values, dtype=np.float64)
        lon, lat = np.meshgrid(lon_1d, lat_1d, indexing="xy")
        lon = lon.T
        lat = lat.T
        mask = np.asarray(ds["mask"].values, dtype=np.float64).T > 0.0
        refsoc_source = np.squeeze(np.asarray(ds["soil_organic_carbon_1d"].values, dtype=np.float64))
        if refsoc_source.ndim != 2:
            raise ValueError("soil_organic_carbon_1d must reduce to a two-dimensional field")
        refsoc_source = refsoc_source.T
    sub_index, sub_area = aggregate_2d_overlap(lalo, resolution_m, lon, lat, mask)
    refsoc_1d = hydrol_refsoc_1d_from_explicit_overlap(refsoc_source, sub_index, sub_area)
    return refsoc_1d, {"refsoc_1d_sub_index": sub_index, "refsoc_1d_sub_area": sub_area}


def read_paper_thermosoil_reftemp_static_field(
    config_path: str | Path,
    *,
    lalo,
    resolution_m,
    ngrnd: int,
    nvm: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read no-restart THERMOSOIL ``ptn`` from ``REFTEMP_FILE``.

    Fortran provenance: ``thermosoil.f90::thermosoil_initialize`` lines
    537-545 calls ``thermosoil_read_reftempfile`` when restart ``ptn`` is
    absent and ``READ_REFTEMP`` is true. ``thermosoil_read_reftempfile`` lines
    3167-3214 reads variable ``temperature`` from ``REFTEMP_FILE`` with
    ``interpweight_2Dcont`` over ``nav_lon``/``nav_lat`` and copies the
    interpolated Celsius value to all ``ngrnd`` layers after adding
    ``ZeroCelsius``.
    """

    config = load_case_config(config_path)
    path = Path(config["static_inputs"]["reftemp_file"])
    ngrnd = int(ngrnd)
    nvm = int(nvm)
    if ngrnd < 1 or nvm < 1:
        raise ValueError("ngrnd and nvm must be positive")
    with xr.open_dataset(path, decode_times=False) as ds:
        lon = np.asarray(ds["nav_lon"].values, dtype=np.float64).T
        lat = np.asarray(ds["nav_lat"].values, dtype=np.float64).T
        temp_c = np.asarray(ds["temperature"].values, dtype=np.float64).T
    mask = np.ones_like(temp_c, dtype=bool)
    sub_index, sub_area = aggregate_2d_overlap(lalo, resolution_m, lon, lat, mask)
    reftemp_c = weighted_mean_from_explicit_overlap(temp_c[:, :, None], sub_index, sub_area)[:, 0]
    reftemp_k = reftemp_c + 273.15
    ptn = np.repeat(reftemp_k[:, None, None], ngrnd, axis=1)
    ptn = np.repeat(ptn, nvm, axis=2)
    return ptn, {
        "reftemp_celsius": reftemp_c,
        "reftemp_sub_index": sub_index,
        "reftemp_sub_area": sub_area,
    }


def read_paper_condveg_background_soilalbedo(
    config_path: str | Path,
    *,
    lalo,
    resolution_m,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read paper-case MODIS background bare-soil albedo.

    Fortran provenance: ``condveg.f90::condveg_initialize`` lines 180-192
    calls ``condveg_background_soilalb`` when restart ``soilalbedo_bg`` is
    absent and ``ALB_BG_MODIS=y``. ``condveg_background_soilalb`` lines
    1214-1274 reads ``ALB_BG_FILE`` variables ``bg_alb_vis`` and
    ``bg_alb_nir`` with mask variable ``mask`` via ``interpweight_2Dcont``.

    The continuous-field interpolation follows ``interpweight_2Dcont``:
    ``interpweight.f90`` lines 2132-2192 builds explicit area overlaps with
    ``aggregate_p`` and calls ``interpweight_provide_interpolation2D``; lines
    3836-3879 compute an area-weighted mean, using the supplied default only
    when no positive overlap is available. The source passes
    ``albbg_default(inir)`` for both bands at ``condveg.f90`` lines 1259-1274.
    """

    config = load_case_config(config_path)
    path = Path(config["static_inputs"]["alb_bg_file"])
    with xr.open_dataset(path, decode_times=False) as ds:
        source_lon_1d, source_lat_1d = _fortran_lon_lat_axes_from_1d(ds["longitude"].values, ds["latitude"].values)
        source_lon, source_lat = np.meshgrid(source_lon_1d, source_lat_1d, indexing="xy")
        source_lon = source_lon.T
        source_lat = source_lat.T
        mask = np.asarray(ds["mask"].values, dtype=np.float64).T > 0.0
        vis = np.asarray(ds["bg_alb_vis"].values, dtype=np.float64).T
        nir = np.asarray(ds["bg_alb_nir"].values, dtype=np.float64).T

    sub_index, sub_area = aggregate_2d_overlap(lalo, resolution_m, source_lon, source_lat, mask)
    overlap_count = np.count_nonzero(sub_area > 0.0, axis=1).astype(np.int32)
    npts = np.asarray(lalo, dtype=np.float64).shape[0]
    source_values = np.stack((vis, nir), axis=2)
    bands = np.full((npts, 2), 0.247, dtype=np.float64)
    valid_targets = overlap_count > 0
    if np.any(valid_targets):
        bands[valid_targets] = weighted_mean_from_explicit_overlap(
            source_values,
            sub_index[valid_targets],
            sub_area[valid_targets],
        )
    soilalbedo_bg = bands
    return soilalbedo_bg, {
        "soilalbedo_bg_source_count_vis": overlap_count,
        "soilalbedo_bg_source_count_nir": overlap_count.copy(),
        "soilalbedo_bg_sub_index": sub_index,
        "soilalbedo_bg_sub_area": sub_area,
    }


def zobler_soilclass_from_explicit_overlap(soiltext, sub_index, sub_area, *, index_base: int = 1) -> np.ndarray:
    """Map explicit overlapped Zobler soiltext cells to 3 ORCHIDEE classes.

    Fortran provenance: `slowproc.f90`, subroutine `slowproc_soilt`, lines
    4575-4692. In the Zobler branch, source `soiltext` classes map as
    `1 -> coarse`, `2,3,4,6 -> medium`, and `5 -> fine`; accumulated
    `sub_area` values are normalized by `sgn = SUM(sub_area)` at lines
    4686-4692.
    """

    soiltext = np.asarray(soiltext)
    sub_index = np.asarray(sub_index, dtype=np.int64)
    sub_area = np.asarray(sub_area, dtype=np.float64)
    if sub_index.ndim != 3 or sub_index.shape[2] != 2:
        raise ValueError("sub_index must have shape [npts,nbvmax,2]")
    if sub_area.shape != sub_index.shape[:2]:
        raise ValueError("sub_area must have shape [npts,nbvmax]")

    out = np.zeros((sub_index.shape[0], 3), dtype=np.float64)
    weights = np.zeros(sub_index.shape[0], dtype=np.float64)
    for ib in range(sub_index.shape[0]):
        for ilf in range(sub_index.shape[1]):
            area = float(sub_area[ib, ilf])
            if area <= 0.0:
                continue
            ip = int(sub_index[ib, ilf, 0]) - index_base
            jp = int(sub_index[ib, ilf, 1]) - index_base
            if ip < 0 or jp < 0:
                continue
            klass = int(round(float(soiltext[ip, jp])))
            if klass == 1:
                out[ib, 0] += area
            elif klass in (2, 3, 4, 6):
                out[ib, 1] += area
            elif klass == 5:
                out[ib, 2] += area
            else:
                raise ValueError(f"bad Zobler soiltext class {klass}")
            weights[ib] += area
    if np.any(weights <= 0.0):
        raise ValueError("each target point needs positive explicit overlap area")
    return out / weights[:, None]


def slowproc_soilt_from_explicit_overlap(
    soiltext,
    soilbd,
    soil_ph_source,
    poor_soils_source,
    sub_index,
    sub_area,
    *,
    soil_classif: str,
    index_base: int = 1,
    do_poor_soils: bool = True,
    impsoilt: bool = False,
    clayfraction_default: float = 0.2,
    sandfraction_default: float = 0.4,
    siltfraction_default: float = 0.4,
    bulk_density_default: float = 1.65,
    soil_ph_default: float = 7.0,
) -> dict[str, np.ndarray]:
    """Apply the pure scientific core of Fortran ``slowproc_soilt``.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/slowproc.f90``,
    subroutine ``slowproc_soilt``, lines 4380-4903. The caller must provide
    the exact ``aggregate_p`` overlap packet from lines 4455-4555. NetCDF,
    MPI broadcast, and overlap construction remain strict external input
    boundaries; this function never invents source cells or nearest values.

    ``impsoilt`` is rejected because ``slowproc_init`` does not call this
    routine on that path (lines 1928-2012); prescribed fields belong to the
    restart/run-definition boundary, not to this map interpolation process.
    """

    classification = str(soil_classif).lower()
    if impsoilt:
        raise ValueError("slowproc_soilt is not entered when impsoilt=true; provide prescribed soil state at slowproc_init")
    if classification not in {"none", "zobler", "fao", "usda"}:
        raise ValueError(f"unsupported soil classification {soil_classif!r}")

    sub_index = np.asarray(sub_index, dtype=np.int64)
    sub_area = np.asarray(sub_area, dtype=np.float64)
    if sub_index.ndim != 3 or sub_index.shape[2] != 2:
        raise ValueError("sub_index must have shape [npts,nbvmax,2]")
    if sub_area.shape != sub_index.shape[:2]:
        raise ValueError("sub_area must have shape [npts,nbvmax]")
    if np.any(sub_area < 0.0):
        raise ValueError("sub_area cannot be negative")

    npts = sub_index.shape[0]
    if classification == "none":
        raise ValueError(
            "Fortran none branch leaves INTENT(out) poor_soils undefined; "
            "a complete deterministic state packet cannot be produced source-faithfully"
        )
    if classification == "fao":
        raise ValueError(
            "Fortran fao branch calls get_soilcorr_zobler with nscm_fao=3, "
            "but that routine requires 7 classes; source reaches a fatal consistency check"
        )

    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 or 1")
    soiltext_array = np.asarray(soiltext, dtype=np.float64)
    arrays = {
        "soiltext": soiltext_array,
        "soilbd": np.asarray(soilbd, dtype=np.float64),
        "soil_ph_source": np.asarray(soil_ph_source, dtype=np.float64),
        "poor_soils_source": (
            np.asarray(poor_soils_source, dtype=np.float64)
            if do_poor_soils
            else np.zeros_like(soiltext_array)
        ),
    }
    source_shape = arrays["soiltext"].shape
    if len(source_shape) != 2 or any(value.shape != source_shape for value in arrays.values()):
        raise ValueError("all soil source fields must share the same two-dimensional shape")

    nclasses = 3 if classification == "zobler" else 12
    default_class = SOILCLASS_DEFAULT_FAO if classification == "zobler" else SOILCLASS_DEFAULT_USDA
    table = ZOBLER_TEXTFRAC_TABLE if classification == "zobler" else USDA_TEXTFRAC_TABLE
    soilclass = np.zeros((npts, nclasses), dtype=np.float64)
    clay = np.zeros(npts, dtype=np.float64)
    sand = np.zeros(npts, dtype=np.float64)
    silt = np.zeros(npts, dtype=np.float64)
    bulk = np.zeros(npts, dtype=np.float64)
    ph = np.zeros(npts, dtype=np.float64)
    poor = np.zeros(npts, dtype=np.float64)
    used_default = np.zeros(npts, dtype=bool)
    zobler_target = {1: 0, 2: 1, 3: 1, 4: 1, 5: 2, 7: 1}

    for ib in range(npts):
        positive = sub_area[ib] > 0.0
        fopt = int(np.count_nonzero(positive))
        if fopt and not np.all(positive[:fopt]):
            raise ValueError("positive sub_area entries must be packed first, as returned by aggregate_p")
        sgn = 0.0
        for ilf in range(fopt):
            ip = int(sub_index[ib, ilf, 0]) - index_base
            jp = int(sub_index[ib, ilf, 1]) - index_base
            if ip < 0 or jp < 0 or ip >= source_shape[0] or jp >= source_shape[1]:
                raise ValueError("positive overlap references an out-of-range source index")
            source_class = int(arrays["soiltext"][ip, jp])
            max_class = 7 if classification == "zobler" else 12
            if source_class > max_class:
                raise ValueError(f"bad {classification} soiltext class {source_class}")
            if source_class <= 0 or (classification == "zobler" and source_class == 6):
                continue

            area = float(sub_area[ib, ilf])
            source_bd = float(arrays["soilbd"][ip, jp])
            source_ph = float(arrays["soil_ph_source"][ip, jp])
            if classification == "zobler":
                source_bd = bulk_density_default if source_bd < 0.0 else source_bd
                source_ph = soil_ph_default if source_ph < 0.0 else source_ph
                target_class = zobler_target[source_class]
            else:
                target_class = source_class - 1
            fractions = table[source_class - 1]
            soilclass[ib, target_class] += area
            silt[ib] += fractions[0] * area
            sand[ib] += fractions[1] * area
            clay[ib] += fractions[2] * area
            bulk[ib] += source_bd * area
            ph[ib] += source_ph * area
            if do_poor_soils:
                poor[ib] += float(arrays["poor_soils_source"][ip, jp]) * area
            sgn += area

        if sgn < 1.0e-8:
            used_default[ib] = True
            soilclass[ib] = default_class
            clay[ib] = clayfraction_default
            sand[ib] = sandfraction_default
            silt[ib] = siltfraction_default
            bulk[ib] = bulk_density_default
            ph[ib] = soil_ph_default
            poor[ib] = 0.0
        else:
            soilclass[ib] /= sgn
            clay[ib] /= sgn
            sand[ib] /= sgn
            silt[ib] /= sgn
            bulk[ib] /= sgn
            ph[ib] /= sgn
            poor[ib] /= sgn

    return {
        "soilclass": soilclass,
        "njsc": njsc_from_soilclass(soilclass),
        "clay_frac": clay,
        "sand_frac": sand,
        "silt_frac": silt,
        "bulk_dens": bulk,
        "soil_ph": ph,
        "poor_soils": poor,
        "used_default": used_default,
    }


def njsc_from_soilclass(soilclass) -> np.ndarray:
    """Return Fortran `MAXLOC(soilclass(ji,:),1)` as one-based class index.

    Fortran provenance: `slowproc.f90`, subroutine `slowproc_init`, lines
    2007-2010 and 2190-2192, assigns `njsc(ji) = MAXLOC(soilclass(ji,:),1)`.
    Fortran `MAXLOC` returns a one-based index and picks the first maximum.
    """

    soilclass = np.asarray(soilclass, dtype=np.float64)
    if soilclass.ndim != 2:
        raise ValueError("soilclass must have shape [npts,nscm]")
    return np.argmax(soilclass, axis=1).astype(np.int32) + 1


def usda_soil_fields_from_explicit_overlap(
    soiltext,
    soilbd,
    soil_ph_source,
    poor_soils_source,
    sub_index,
    sub_area,
    *,
    index_base: int = 1,
    soil_ph_default: float = 7.0,
) -> dict[str, np.ndarray]:
    """Build USDA ``slowproc_soilt`` fields from explicit overlap arrays.

    Fortran provenance: ``slowproc.f90::slowproc_soilt`` USDA branch lines
    4797-4889; ``get_soilcorr_usda`` lines 5563-5621; defaults from
    ``constantes_var.f90`` lines 830-836 and 1396-1398 and
    ``constantes_soil_var.f90`` lines 245-248. This mirrors the active
    accumulation/normalization branch and does not use restart defaults unless
    no valid overlapped source cell contributes.
    """

    result = slowproc_soilt_from_explicit_overlap(
        soiltext,
        soilbd,
        soil_ph_source,
        poor_soils_source,
        sub_index,
        sub_area,
        soil_classif="usda",
        index_base=index_base,
        soil_ph_default=soil_ph_default,
    )
    result.pop("used_default")
    return result


def read_paper_usda_soil_static_fields(
    config_path: str | Path,
    *,
    lalo,
    resolution_m,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Read paper-case USDA soil static fields from local ``SOILCLASS_FILE``.

    Fortran provenance: ``slowproc.f90::slowproc_soilt`` lines 4284-4925
    reads ``SOILCLASS_FILE`` variables, builds a ``soiltext > min_sechiba``
    mask, calls ``aggregate_p``/``aggregate_2d`` at lines 4536-4546, and uses
    the USDA branch lines 4797-4889 to compute static soil fields.
    """

    config = load_case_config(config_path)
    path = Path(config["static_inputs"]["soilclass_file"])
    with xr.open_dataset(path, decode_times=False) as ds:
        lon = np.asarray(ds["nav_lon"].values, dtype=np.float64).T
        lat = np.asarray(ds["nav_lat"].values, dtype=np.float64).T
        soiltext = np.asarray(ds["soiltext"].values, dtype=np.float64).T
        soilbd = np.asarray(ds["soilbd"].values, dtype=np.float64).T
        soilph = np.asarray(ds["soil_ph"].values, dtype=np.float64).T
        poorsol = np.asarray(ds["poor_soils"].values, dtype=np.float64).T
    mask = soiltext > 1.0e-8
    sub_index, sub_area = aggregate_2d_overlap(lalo, resolution_m, lon, lat, mask)
    fields = usda_soil_fields_from_explicit_overlap(soiltext, soilbd, soilph, poorsol, sub_index, sub_area)
    return fields, {"soilclass_sub_index": sub_index, "soilclass_sub_area": sub_area}
