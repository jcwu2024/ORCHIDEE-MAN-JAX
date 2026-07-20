"""Source-faithful spherical and regular-grid geometry from ``haversine.f90``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


R_EARTH = 6_378_000.0
MIN_COS = 0.0001
UNDEF_SECHIBA = 1.0e20


class HaversineGeometryError(ValueError):
    """Fatal ``ipslerr(3, ...)`` condition raised by the Fortran geometry."""


@dataclass(frozen=True)
class PolygonGeometry:
    lonpoly: np.ndarray
    latpoly: np.ndarray
    center: np.ndarray
    neighb_loc: np.ndarray
    iorig: np.ndarray
    jorig: np.ndarray


def _real_array(value, *, name: str, ndim: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    return result


def _fortran_mod(value, divisor):
    """Fortran ``MOD``: quotient truncates toward zero, unlike Python ``%``."""

    value = np.asarray(value)
    return value - np.trunc(value / divisor) * divisor


def haversine_dtor(degree):
    """``haversine.f90::haversine_dtor``, lines 998-1002."""

    return np.asarray(degree) * np.pi / 180.0


def haversine_rtod(radian):
    """``haversine.f90::haversine_rtod``, lines 1008-1013."""

    return np.asarray(radian) * 180.0 / np.pi


def haversine_heading(lon_start, lat_start, lon_end, lat_end):
    """Initial great-circle heading in degrees, source lines 933-949."""

    dlon = haversine_dtor(np.asarray(lon_end) - np.asarray(lon_start))
    lat1 = haversine_dtor(lat_start)
    lat2 = haversine_dtor(lat_end)
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    result = _fortran_mod(haversine_rtod(np.arctan2(y, x)) + 360.0, 360.0)
    return float(result) if np.ndim(result) == 0 else result


def haversine_distance(lon_start, lat_start, lon_end, lat_end, *, r_earth=R_EARTH):
    """Great-circle distance in metres, source lines 953-967."""

    dlat = haversine_dtor(np.asarray(lat_end) - np.asarray(lat_start))
    dlon = haversine_dtor(np.asarray(lon_end) - np.asarray(lon_start))
    lat1 = haversine_dtor(lat_start)
    lat2 = haversine_dtor(lat_end)
    a = np.sin(dlat / 2.0) ** 2 + np.sin(dlon / 2.0) ** 2 * np.cos(lat1) * np.cos(lat2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    result = c * r_earth
    return float(result) if np.ndim(result) == 0 else result


def haversine_radialdis(lon_start, lat_start, head, dis, *, r_earth=R_EARTH) -> np.ndarray:
    """Destination ``[lon, lat]`` along a great circle, source lines 971-992."""

    lon1 = haversine_dtor(lon_start)
    lat1 = haversine_dtor(lat_start)
    tc = haversine_dtor(head)
    angular = np.asarray(dis) / r_earth
    lat = np.arcsin(np.sin(lat1) * np.cos(angular) + np.cos(lat1) * np.sin(angular) * np.cos(tc))
    dlon = np.arctan2(
        np.sin(tc) * np.sin(angular) * np.cos(lat1),
        np.cos(angular) - np.sin(lat1) * np.sin(lat),
    )
    lon = _fortran_mod(lon1 + dlon + np.pi, 2.0 * np.pi) - np.pi
    return np.asarray([haversine_rtod(lon), haversine_rtod(lat)], dtype=np.float64)


def _grid_inputs(lon, lat, index_loc, nbseg: int):
    lon_arr = _real_array(lon, name="lon", ndim=2)
    lat_arr = _real_array(lat, name="lat", ndim=2)
    if lon_arr.shape != lat_arr.shape:
        raise ValueError("lon and lat must have the same shape")
    if nbseg != 4:
        raise HaversineGeometryError("haversine grid polygons require nbseg=4")
    indices = np.asarray(index_loc)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise TypeError("index_loc must be a one-dimensional integer array")
    iim, jjm = lon_arr.shape
    if np.any(indices < 1) or np.any(indices > iim * jjm):
        raise IndexError("index_loc contains an invalid Fortran 1-based grid index")
    return lon_arr, lat_arr, indices.astype(np.int64, copy=False), iim, jjm


def _origins_and_correspondence(indices, iim: int, jjm: int):
    nbpt = indices.size
    iorig = (indices - 1) % iim
    jorig = (indices - 1) // iim
    correspondence = np.full((iim, jjm), -1, dtype=np.int64)
    for point in range(nbpt):
        correspondence[iorig[point], jorig[point]] = point + 1
    return iorig, jorig, correspondence


def _set_neighbours(
    neighbours: np.ndarray,
    point: int,
    ip: int,
    jp: int,
    ipm1: int,
    ipp1: int,
    jpm1: int,
    jpp1: int,
    correspondence: np.ndarray,
) -> None:
    iim, jjm = correspondence.shape
    if ipm1 >= 0 and jpm1 >= 0:
        neighbours[point, 0] = correspondence[ipm1, jpm1]
    if jpm1 >= 0:
        neighbours[point, 1] = correspondence[ip, jpm1]
    if ipp1 < iim and jpm1 >= 0:
        neighbours[point, 2] = correspondence[ipp1, jpm1]
    if ipp1 < iim:
        neighbours[point, 3] = correspondence[ipp1, jp]
    if ipp1 < iim and jpp1 < jjm:
        neighbours[point, 4] = correspondence[ipp1, jpp1]
    if jpp1 < jjm:
        neighbours[point, 5] = correspondence[ip, jpp1]
    if ipm1 >= 0 and jpp1 < jjm:
        neighbours[point, 6] = correspondence[ipm1, jpp1]
    if ipm1 >= 0:
        neighbours[point, 7] = correspondence[ipm1, jp]


def haversine_reglatlontoploy(
    lon,
    lat,
    index_loc,
    *,
    global_grid: bool = False,
    nbseg: int = 4,
    undef_sechiba: float = UNDEF_SECHIBA,
) -> PolygonGeometry:
    """Construct regular lon/lat polygons, source lines 72-250.

    Inputs retain the Fortran ``(iim,jjm)`` layout and ``index_loc`` is 1-based.
    """

    lon, lat, indices, iim, jjm = _grid_inputs(lon, lat, index_loc, nbseg)
    iorig, jorig, correspondence = _origins_and_correspondence(indices, iim, jjm)
    nbpt = indices.size
    lonpoly = np.empty((nbpt, 2 * nbseg), dtype=np.float64)
    latpoly = np.empty_like(lonpoly)
    center = np.empty((nbpt, 2), dtype=np.float64)
    neighbours = np.full((nbpt, 2 * nbseg), -1, dtype=np.int64)

    for point in range(nbpt):
        ip, jp = int(iorig[point]), int(jorig[point])
        ipm1, ipp1, jpm1, jpp1 = ip - 1, ip + 1, jp - 1, jp + 1

        if ipp1 < iim:
            dlonp1 = (lon[ipp1, jp] - lon[ip, jp]) / 2.0
        elif ipm1 >= 0:
            dlonp1 = (lon[ip, jp] - lon[ipm1, jp]) / 2.0
            if global_grid:
                ipp1 = 0
        else:
            dlonp1 = undef_sechiba
        if ipm1 >= 0:
            dlonm1 = (lon[ip, jp] - lon[ipm1, jp]) / 2.0
        elif ipp1 < iim:
            dlonm1 = (lon[ipp1, jp] - lon[ip, jp]) / 2.0
            if global_grid:
                ipm1 = iim - 1
        else:
            dlonm1 = undef_sechiba
        if dlonp1 >= undef_sechiba - 1.0:
            dlonp1 = dlonm1
        if dlonm1 >= undef_sechiba - 1.0:
            dlonm1 = dlonp1
        if dlonp1 >= undef_sechiba - 1.0 and dlonm1 >= undef_sechiba - 1.0:
            raise HaversineGeometryError("not enough points in longitude to estimate polygon bounds")

        if jpp1 < jjm:
            dlatp1 = (lat[ip, jpp1] - lat[ip, jp]) / 2.0
        elif jpm1 >= 0:
            dlatp1 = (lat[ip, jp] - lat[ip, jpm1]) / 2.0
        else:
            dlatp1 = undef_sechiba
        if jpm1 >= 0:
            dlatm1 = (lat[ip, jp] - lat[ip, jpm1]) / 2.0
        elif jpp1 < jjm:
            dlatm1 = (lat[ip, jpp1] - lat[ip, jp]) / 2.0
        else:
            dlatm1 = undef_sechiba
        if dlatp1 >= undef_sechiba - 1.0:
            dlatp1 = dlatm1
        if dlatm1 >= undef_sechiba - 1.0:
            dlatm1 = dlatp1
        if dlatp1 >= undef_sechiba - 1.0 and dlatm1 >= undef_sechiba - 1.0:
            raise HaversineGeometryError("not enough points in latitude to estimate polygon bounds")

        lonpoly[point] = (
            lon[ip, jp] - dlonm1,
            lon[ip, jp],
            lon[ip, jp] + dlonp1,
            lon[ip, jp] + dlonp1,
            lon[ip, jp] + dlonp1,
            lon[ip, jp],
            lon[ip, jp] - dlonm1,
            lon[ip, jp] - dlonm1,
        )
        latpoly[point] = (
            lat[ip, jp] - dlatp1,
            lat[ip, jp] - dlatp1,
            lat[ip, jp] - dlatp1,
            lat[ip, jp],
            lat[ip, jp] + dlatm1,
            lat[ip, jp] + dlatm1,
            lat[ip, jp] + dlatm1,
            lat[ip, jp],
        )
        center[point] = (lon[ip, jp], lat[ip, jp])
        _set_neighbours(neighbours, point, ip, jp, ipm1, ipp1, jpm1, jpp1, correspondence)

    return PolygonGeometry(lonpoly, latpoly, center, neighbours, iorig + 1, jorig + 1)


def haversine_regxytoploy(
    lon,
    lat,
    index_loc,
    *,
    projection_code: int,
    ij_to_latlon: Callable[[float, float], tuple[float, float]],
    nbseg: int = 4,
) -> PolygonGeometry:
    """Construct projected regular-X/Y polygons, source lines 263-382.

    ``ij_to_latlon(i,j)`` must return ``(lat,lon)`` for Fortran 1-based real
    projection coordinates, matching ``module_llxy::ij_to_latlon``.
    """

    lon, lat, indices, iim, jjm = _grid_inputs(lon, lat, index_loc, nbseg)
    if not (projection_code > 0 and projection_code < 203):
        raise HaversineGeometryError("unknown projection code")
    iorig, jorig, correspondence = _origins_and_correspondence(indices, iim, jjm)
    nbpt = indices.size
    lonpoly = np.empty((nbpt, 2 * nbseg), dtype=np.float64)
    latpoly = np.empty_like(lonpoly)
    center = np.empty((nbpt, 2), dtype=np.float64)
    neighbours = np.full((nbpt, 2 * nbseg), -1, dtype=np.int64)
    offsets = ((-0.5, -0.5), (0.0, -0.5), (0.5, -0.5), (0.5, 0.0),
               (0.5, 0.5), (0.0, 0.5), (-0.5, 0.5), (-0.5, 0.0))
    for point in range(nbpt):
        ip, jp = int(iorig[point]), int(jorig[point])
        for vertex, (di, dj) in enumerate(offsets):
            plat, plon = ij_to_latlon(ip + 1.0 + di, jp + 1.0 + dj)
            latpoly[point, vertex] = plat
            lonpoly[point, vertex] = plon
        center[point] = (lon[ip, jp], lat[ip, jp])
        _set_neighbours(
            neighbours, point, ip, jp, ip - 1, ip + 1, jp - 1, jp + 1, correspondence
        )
    return PolygonGeometry(lonpoly, latpoly, center, neighbours, iorig + 1, jorig + 1)


def haversine_singlepointploy(lon, lat, *, nbseg: int = 4) -> PolygonGeometry:
    """Construct the fixed 111111 m single-point polygon, source lines 394-488."""

    lon_arr = _real_array(lon, name="lon", ndim=2)
    lat_arr = _real_array(lat, name="lat", ndim=2)
    if lon_arr.shape != lat_arr.shape or 0 in lon_arr.shape:
        raise ValueError("lon and lat must be matching non-empty 2-D arrays")
    iim, jjm = lon_arr.shape
    if iim != 1 and jjm != 1:  # Preserve the source's AND, despite its stricter diagnostic text.
        raise HaversineGeometryError("haversine_singlepointploy can only be used if iim=jjm=1")
    if nbseg != 4:
        raise HaversineGeometryError("haversine_singlepointploy requires nbseg=4")
    half_side = np.sqrt(111111.0 * 111111.0) / 2.0
    north = haversine_radialdis(lon_arr[0, 0], lat_arr[0, 0], 0.0, half_side)
    east = haversine_radialdis(lon_arr[0, 0], lat_arr[0, 0], 90.0, half_side)
    south = haversine_radialdis(lon_arr[0, 0], lat_arr[0, 0], 180.0, half_side)
    west = haversine_radialdis(lon_arr[0, 0], lat_arr[0, 0], 270.0, half_side)
    lonpoly = np.asarray([[west[0], north[0], east[0], east[0], east[0], south[0], west[0], west[0]]])
    latpoly = np.asarray([[north[1], north[1], north[1], east[1], south[1], south[1], south[1], west[1]]])
    return PolygonGeometry(
        lonpoly,
        latpoly,
        np.asarray([[lon_arr[0, 0], lat_arr[0, 0]]]),
        np.full((1, 8), -1, dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
    )


def haversine_polyheadings(lonpoly, latpoly, center) -> np.ndarray:
    """Outward headings for polygon points, source lines 503-531."""

    lonpoly = _real_array(lonpoly, name="lonpoly", ndim=2)
    latpoly = _real_array(latpoly, name="latpoly", ndim=2)
    center = _real_array(center, name="center", ndim=2)
    if lonpoly.shape != latpoly.shape or center.shape != (lonpoly.shape[0], 2):
        raise ValueError("polygon and center shapes are inconsistent")
    return _fortran_mod(
        haversine_heading(lonpoly, latpoly, center[:, 0, None], center[:, 1, None]) + 180.0,
        360.0,
    )


def haversine_clockwise(nbseg: int, heading, start: float) -> np.ndarray:
    """Fortran 1-based clockwise sort indices, source lines 749-798."""

    headings = _real_array(heading, name="heading", ndim=1)
    if headings.shape != (nbseg * 2,):
        raise ValueError("heading must have length nbseg*2")
    delang = 360.0 / (nbseg * 2)
    undef = 9999999999.99999
    workhead = headings.copy()
    sortindex = np.empty(nbseg * 2, dtype=np.int64)
    for is_ in range(nbseg * 2):
        for js in range(nbseg * 2):
            if workhead[js] < undef:
                workhead[js] = _fortran_mod(
                    headings[js] - (start + is_ * delang) + 360.0, 360.0
                )
                if workhead[js] > 180.0:
                    workhead[js] -= 360.0
        imin = int(np.argmin(np.abs(workhead)))
        sortindex[is_] = imin + 1
        workhead[imin] = undef
    return sortindex


def haversine_polysort(lonpoly, latpoly, headings, neighb):
    """Sort polygons clockwise, source lines 543-593."""

    lon = _real_array(lonpoly, name="lonpoly", ndim=2).copy()
    lat = _real_array(latpoly, name="latpoly", ndim=2).copy()
    head = _real_array(headings, name="headings", ndim=2).copy()
    neighbours = np.asarray(neighb).copy()
    if not (lon.shape == lat.shape == head.shape == neighbours.shape) or lon.shape[1] % 2:
        raise ValueError("polygon arrays must share shape (nbpt,nbseg*2)")
    nbseg = lon.shape[1] // 2
    for point in range(lon.shape[0]):
        order = haversine_clockwise(nbseg, head[point], 0.0) - 1
        lon[point], lat[point] = lon[point, order], lat[point, order]
        head[point], neighbours[point] = head[point, order], neighbours[point, order]
    return lon, lat, head, neighbours


def _segment_inputs(lonpoly, latpoly, nbseg: int | None):
    lon = _real_array(lonpoly, name="lonpoly", ndim=2)
    lat = _real_array(latpoly, name="latpoly", ndim=2)
    if lon.shape != lat.shape or lon.shape[1] % 2:
        raise ValueError("lonpoly and latpoly must share shape (nbpt,nbseg*2)")
    inferred = lon.shape[1] // 2
    if nbseg is not None and nbseg != inferred:
        raise ValueError("nbseg does not match polygon width")
    return lon, lat, inferred


def _segment_offset(lon, lat, point: int, nbseg: int) -> int:
    slpm1 = haversine_heading(lon[point, 0], lat[point, 0], lon[point, -1], lat[point, -1])
    slpp1 = haversine_heading(lon[point, 0], lat[point, 0], lon[point, 1], lat[point, 1])
    return -1 if abs(_fortran_mod(slpp1 - slpm1, 360.0)) > 135.0 else 0


def haversine_polyseglen(lonpoly, latpoly, *, nbseg: int | None = None) -> np.ndarray:
    """Great-circle polygon segment lengths, source lines 605-655."""

    lon, lat, nbseg = _segment_inputs(lonpoly, latpoly, nbseg)
    result = np.empty((lon.shape[0], nbseg), dtype=np.float64)
    for point in range(lon.shape[0]):
        offset = _segment_offset(lon, lat, point, nbseg)
        for segment, vertex in enumerate(range(0, 2 * nbseg, 2)):
            start = (vertex + offset) % (2 * nbseg)
            end = (vertex + offset + 2) % (2 * nbseg)
            result[point, segment] = haversine_distance(
                lon[point, start], lat[point, start], lon[point, end], lat[point, end]
            )
    return result


def haversine_laloseglen(
    lonpoly, latpoly, *, nbseg: int | None = None, mincos: float = MIN_COS
) -> np.ndarray:
    """Regular lon/lat segment lengths, source lines 667-736."""

    lon, lat, nbseg = _segment_inputs(lonpoly, latpoly, nbseg)
    result = np.empty((lon.shape[0], nbseg), dtype=np.float64)
    epsilon = np.finfo(np.float64).eps
    for point in range(lon.shape[0]):
        offset = _segment_offset(lon, lat, point, nbseg)
        for segment, vertex in enumerate(range(0, 2 * nbseg, 2)):
            start = (vertex + offset) % (2 * nbseg)
            end = (vertex + offset + 2) % (2 * nbseg)
            if abs(lon[point, start] - lon[point, end]) < epsilon:
                result[point, segment] = abs(lat[point, start] - lat[point, end]) * np.pi / 180.0 * R_EARTH
            elif abs(lat[point, start] - lat[point, end]) < epsilon:
                coslat = max(np.cos(lat[point, start] * np.pi / 180.0), mincos)
                result[point, segment] = abs(lon[point, start] - lon[point, end]) * np.pi / 180.0 * R_EARTH * coslat
            else:
                raise HaversineGeometryError("polygon does not originate from a regular latitude/longitude grid")
    return result


def haversine_polyarea(lonpoly, latpoly, *, nbseg: int | None = None) -> np.ndarray:
    """Girard great-circle polygon area, source lines 813-850."""

    lon, lat, nbseg = _segment_inputs(lonpoly, latpoly, nbseg)
    area = np.empty(lon.shape[0], dtype=np.float64)
    for point in range(lon.shape[0]):
        angles = np.empty(nbseg, dtype=np.float64)
        for segment, vertex in enumerate(range(0, 2 * nbseg, 2)):
            next_vertex = (vertex + 2) % (2 * nbseg)
            previous_vertex = (vertex - 2) % (2 * nbseg)
            beta1 = haversine_dtor(haversine_heading(
                lon[point, vertex], lat[point, vertex], lon[point, next_vertex], lat[point, next_vertex]
            ))
            beta2 = haversine_dtor(haversine_heading(
                lon[point, vertex], lat[point, vertex], lon[point, previous_vertex], lat[point, previous_vertex]
            ))
            angles[segment] = np.arccos(np.cos(-beta1) * np.cos(-beta2) + np.sin(-beta1) * np.sin(-beta2))
        area[point] = (np.sum(angles) - (nbseg - 2) * np.pi) * R_EARTH**2
    return area


def haversine_laloarea(seglen, *, nbseg: int | None = None) -> np.ndarray:
    """Regular four-sided lon/lat area, source lines 863-888."""

    lengths = _real_array(seglen, name="seglen", ndim=2)
    inferred = lengths.shape[1]
    if nbseg is not None and nbseg != inferred:
        raise ValueError("nbseg does not match seglen width")
    if inferred != 4:
        raise HaversineGeometryError("haversine_laloarea requires four polygon segments")
    return (lengths[:, 0] + lengths[:, 2]) / 2.0 * (lengths[:, 1] + lengths[:, 3]) / 2.0


def haversine_xyarea(ilandtoij, jlandtoij, dx, dy) -> np.ndarray:
    """Projected-grid cell area with Fortran 1-based indices, source lines 901-922."""

    ii = np.asarray(ilandtoij)
    jj = np.asarray(jlandtoij)
    dx_arr = _real_array(dx, name="dx", ndim=2)
    dy_arr = _real_array(dy, name="dy", ndim=2)
    if ii.shape != jj.shape or ii.ndim != 1 or dx_arr.shape != dy_arr.shape:
        raise ValueError("index and projected-grid array shapes are inconsistent")
    return dx_arr[ii - 1, jj - 1] * dy_arr[ii - 1, jj - 1]
