"""Regular-grid polygon aggregation from ORCHIDEE ``interpol_help.f90``.

The public indices remain Fortran one-based.  Positive overlap slots form a
contiguous prefix; unused slots are index zero and area ``-1`` exactly as in
the source routines.  MPI transport and RegXY projection are explicit outer
boundaries rather than numerical approximations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Protocol

import numpy as np

from jax_orchidee.driver.geometry_polygons import polygones_area


INTERPOLATION_AGGREGATE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_global/interpol_help.f90::aggregate_2d lines 46-466",
    "fortran_source/ORCHIDEE/src_global/interpol_help.f90::aggregate_vec lines 473-779",
    "fortran_source/ORCHIDEE/src_global/interpol_help.f90::aggregate_vec_p lines 782-828",
    "fortran_source/ORCHIDEE/src_global/interpol_help.f90::aggregate_2d_p lines 830-889",
)

R_EARTH = 6_378_000.0
MIN_COS = 0.001
UNDEFINED_AREA = -1.0


def normalized_overlap_weights(areaoverlap) -> np.ndarray:
    """Normalize each positive overlap prefix; no-overlap rows remain zero."""

    area = np.asarray(areaoverlap, dtype=np.float64)
    if area.ndim != 2:
        raise ValueError("areaoverlap must be rank 2")
    positive = np.where(area > 0.0, area, 0.0)
    totals = np.sum(positive, axis=1, keepdims=True)
    return np.divide(positive, totals, out=np.zeros_like(positive), where=totals > 0.0)


class Aggregation2DResult(NamedTuple):
    indinc: np.ndarray
    areaoverlap: np.ndarray
    ok: bool

    @property
    def weights(self) -> np.ndarray:
        return normalized_overlap_weights(self.areaoverlap)

    @property
    def counts(self) -> np.ndarray:
        return np.count_nonzero(self.areaoverlap > 0.0, axis=1)


class AggregationVectorResult(NamedTuple):
    indinc: np.ndarray
    areaoverlap: np.ndarray
    ok: bool

    @property
    def weights(self) -> np.ndarray:
        return normalized_overlap_weights(self.areaoverlap)

    @property
    def counts(self) -> np.ndarray:
        return np.count_nonzero(self.areaoverlap > 0.0, axis=1)


class MPIBackendRequiredError(RuntimeError):
    """Raised when a multi-rank ``*_p`` call has no transport backend."""


class ParallelAggregateBackend(Protocol):
    """Transport owner for explicit multi-rank aggregation."""

    def aggregate_vec_p(self, core: Callable[..., AggregationVectorResult], **kwargs) -> AggregationVectorResult: ...

    def aggregate_2d_p(self, core: Callable[..., Aggregation2DResult], **kwargs) -> Aggregation2DResult: ...


def _common_inputs(nbpt, lalo, neighbours, resolution, contfrac, incmax):
    if int(nbpt) <= 0:
        raise ValueError("nbpt must be positive")
    if int(incmax) < 0:
        raise ValueError("incmax must be non-negative")
    coordinates = np.asarray(lalo, dtype=np.float64)
    neigh = np.asarray(neighbours)
    resol = np.asarray(resolution, dtype=np.float64)
    fraction = np.asarray(contfrac, dtype=np.float64)
    if coordinates.shape != (nbpt, 2):
        raise ValueError(f"lalo must have shape {(nbpt, 2)}")
    if neigh.ndim != 2 or neigh.shape[0] != nbpt:
        raise ValueError("neighbours must be rank 2 with nbpt rows")
    if resol.shape != (nbpt, 2):
        raise ValueError(f"resolution must have shape {(nbpt, 2)}")
    if fraction.shape != (nbpt,):
        raise ValueError(f"contfrac must have shape {(nbpt,)}")
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(resol)):
        raise ValueError("lalo and resolution must be finite")
    if np.any(resol <= 0.0):
        raise ValueError("resolution must be strictly positive")
    # neighbours and contfrac are source ABI inputs but are not read at lines
    # 46-466 or 473-779. Only their explicit Fortran shape is validated here.
    return coordinates, resol


def _nint(value: float) -> int:
    """Fortran NINT for finite scalar values (ties away from zero)."""

    return int(np.floor(value + 0.5)) if value >= 0.0 else int(np.ceil(value - 0.5))


def _fmod_360(value: float) -> float:
    """Fortran MOD(value, 360), whose result has the dividend's sign."""

    return float(np.fmod(value, 360.0))


def _target_bounds(coordinates: np.ndarray, resolution: np.ndarray, point: int):
    latitude, longitude = coordinates[point]
    xscale = max(np.cos(np.deg2rad(latitude)), MIN_COS) * np.pi / 180.0 * R_EARTH
    yscale = np.pi / 180.0 * R_EARTH
    return (
        longitude - resolution[point, 0] / (2.0 * xscale),
        longitude + resolution[point, 0] / (2.0 * xscale),
        latitude - resolution[point, 1] / (2.0 * yscale),
        latitude + resolution[point, 1] / (2.0 * yscale),
    )


def _overlaps(center, low, high, target_low, target_high) -> bool:
    return (
        target_low < center < target_high
        or low < target_low < high
        or low < target_high < high
    )


def _rectangle_overlap_area(lon_low, lon_up, lat_low, lat_up, source_lon_low, source_lon_up,
                            source_lat_low, source_lat_up, source_lat) -> float:
    """The lines 386-390/734-738 rectangle polygon and metric scaling."""

    west = max(lon_low, source_lon_low)
    east = min(lon_up, source_lon_up)
    south = max(lat_low, source_lat_low)
    north = min(lat_up, source_lat_up)
    if east <= west or north <= south:
        return 0.0
    polygon = np.asarray(((west, south), (east, south), (east, north), (west, north)))
    dx = np.pi / 180.0 * R_EARTH * max(np.cos(np.deg2rad(source_lat)), MIN_COS)
    dy = np.pi / 180.0 * R_EARTH
    return polygones_area(4, polygon, dx, dy)


def _source_2d_bounds(lon: np.ndarray, lat: np.ndarray):
    iml, jml = lon.shape
    lon_full = np.empty((iml + 2, jml + 2), dtype=np.float64)
    lat_full = np.empty_like(lon_full)
    lon_full[1:-1, 1:-1] = lon
    lat_full[1:-1, 1:-1] = lat
    if lon[-1, 0] < lon_full[1, 1]:
        lon_full[0, 1:-1] = lon[-1, :]
    else:
        lon_full[0, 1:-1] = lon[-1, :] - 360.0
    lat_full[0, 1:-1] = lat[-1, :]
    if lon[0, 0] > lon_full[-2, 1]:
        lon_full[-1, 1:-1] = lon[0, :]
    else:
        lon_full[-1, 1:-1] = lon[0, :] + 360.0
    lat_full[-1, 1:-1] = lat[0, :]

    lat_resol = lat[0, 1] - lat[0, 0]
    lat_full[1:-1, 0] = lat[0, 0] - lat_resol
    lat_full[1:-1, -1] = lat[0, -1] + lat_resol
    lat_full[0, 0], lat_full[-1, 0] = lat_full[-2, 0], lat_full[1, 0]
    lat_full[0, -1], lat_full[-1, -1] = lat_full[-2, -1], lat_full[1, -1]
    lon_resol = lon_full[2, 1] - lon_full[1, 1]
    lon_full[:, 0] = lon_full[:, 1]
    lon_full[:, -1] = lon_full[:, -2]
    lon_full[0, :] = lon_full[1, 0] - lon_resol
    lon_full[-1, :] = lon_full[-2, 0] + lon_resol

    lon_low = np.empty_like(lon)
    lon_up = np.empty_like(lon)
    lat_low = np.empty_like(lat)
    lat_up = np.empty_like(lat)
    for i in range(iml):
        for j in range(jml):
            west = 0.5 * (lon_full[i, j + 1] + lon_full[i + 1, j + 1])
            east = 0.5 * (lon_full[i + 1, j + 1] + lon_full[i + 2, j + 1])
            south = 0.5 * (lat_full[i + 1, j] + lat_full[i + 1, j + 1])
            north = 0.5 * (lat_full[i + 1, j + 1] + lat_full[i + 1, j + 2])
            lon_low[i, j], lon_up[i, j] = min(west, east), max(west, east)
            lat_low[i, j], lat_up[i, j] = min(south, north), max(south, north)
    return lon_low, lon_up, lat_low, lat_up


def aggregate_2d(
    nbpt: int, lalo, neighbours, resolution, contfrac, iml: int, jml: int,
    lon_rel, lat_rel, mask, callsign: str, incmax: int, *, global_grid: bool = False,
    opt_nbpt_start: int | None = None, opt_nbpt_end: int | None = None,
) -> Aggregation2DResult:
    """Port of ``interpol_help.f90::aggregate_2d`` lines 46-466."""

    del callsign
    coordinates, resol = _common_inputs(nbpt, lalo, neighbours, resolution, contfrac, incmax)
    lon = np.asarray(lon_rel, dtype=np.float64)
    lat = np.asarray(lat_rel, dtype=np.float64)
    source_mask = np.asarray(mask)
    if iml < 2 or jml < 2:
        raise ValueError("aggregate_2d source lines 156 and 166 require iml>=2 and jml>=2")
    if lon.shape != (iml, jml) or lat.shape != lon.shape or source_mask.shape != lon.shape:
        raise ValueError(f"lon_rel, lat_rel, and mask must have shape {(iml, jml)}")
    if not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
        raise ValueError("source coordinates must be finite")
    start = 1 if opt_nbpt_start is None else int(opt_nbpt_start)
    end = nbpt if opt_nbpt_end is None else int(opt_nbpt_end)
    if (opt_nbpt_start is None) != (opt_nbpt_end is None):
        # Fortran ignores a lone optional bound; make that surprising state explicit.
        start, end = 1, nbpt
    if start < 1 or end < start or end > nbpt:
        raise ValueError("opt_nbpt_start/end must define a non-empty one-based range")

    lon_low_src, lon_up_src, lat_low_src, lat_up_src = _source_2d_bounds(lon, lat)
    target_bounds = [_target_bounds(coordinates, resol, point) for point in range(nbpt)]
    min_lon_point = int(np.argmin(coordinates[:, 1]))
    max_lon_point = int(np.argmax(coordinates[:, 1]))
    domain_minlon = target_bounds[min_lon_point][0]
    domain_maxlon = target_bounds[max_lon_point][1]
    # Lines 222-223 use the max-longitude point's Y resolution for both limits.
    yscale = np.pi / 180.0 * R_EARTH
    domain_minlat = np.min(coordinates[:, 0]) - resol[max_lon_point, 1] / (2.0 * yscale)
    domain_maxlat = np.max(coordinates[:, 0]) + resol[max_lon_point, 1] / (2.0 * yscale)

    search: list[tuple[int, int]] = []
    for j in range(jml):
        for i in range(iml):
            in_domain = global_grid or (
                lon_up_src[i, j] >= domain_minlon and lon_low_src[i, j] <= domain_maxlon
                and lat_up_src[i, j] >= domain_minlat and lat_low_src[i, j] <= domain_maxlat
            )
            if in_domain and source_mask[i, j] == 1:
                search.append((i, j))

    nlocal = end - start + 1
    indices = np.zeros((nlocal, incmax, 2), dtype=np.int32)
    areas = np.full((nlocal, incmax), UNDEFINED_AREA, dtype=np.float64)
    for local, point in enumerate(range(start - 1, end)):
        lon_low, lon_up, lat_low, lat_up = target_bounds[point]
        kept: list[tuple[int, int]] = []
        count = 0
        for i, j in search:
            if lon_low >= -180.0 and lon_up <= 180.0:
                lon_center = lon[i, j]
                source_lon_low, source_lon_up = lon_low_src[i, j], lon_up_src[i, j]
            elif lon_low < -180.0:
                lon_center = _fmod_360(lon[i, j] - 360.0)
                source_lon_low = _fmod_360(lon_low_src[i, j] - 360.0)
                source_lon_up = _fmod_360(lon_up_src[i, j] - 360.0)
            else:
                lon_center = _fmod_360(lon[i, j] + 360.0)
                source_lon_low = _fmod_360(lon_low_src[i, j] + 360.0)
                source_lon_up = _fmod_360(lon_up_src[i, j] + 360.0)
            full_inside = False
            if _overlaps(lon_center, source_lon_low, source_lon_up, lon_low, lon_up) and _overlaps(
                lat[i, j], lat_low_src[i, j], lat_up_src[i, j], lat_low, lat_up
            ):
                count += 1
                if count > incmax:
                    return Aggregation2DResult(indices, areas, False)
                slot = count - 1
                areas[local, slot] = _rectangle_overlap_area(
                    lon_low, lon_up, lat_low, lat_up, source_lon_low, source_lon_up,
                    lat_low_src[i, j], lat_up_src[i, j], lat[i, j],
                )
                indices[local, slot] = (i + 1, j + 1)
                full_inside = (
                    source_lon_up < lon_up and source_lon_low > lon_low
                    and lat_up_src[i, j] < lat_up and lat_low_src[i, j] > lat_low
                )
            if not full_inside:
                kept.append((i, j))
        search = kept
    return Aggregation2DResult(indices, areas, True)


def _coarse_lookup(point_lon: float, point_lat: float, bounds, incp: int) -> int:
    """Equivalent value of overwritten ``fine_ind(NINT(lon*incp),...)``."""

    ilon, ilat = _nint(point_lon * incp), _nint(point_lat * incp)
    result = 0
    for point, (lon_low, lon_up, lat_low, lat_up) in enumerate(bounds, start=1):
        if (_nint(lon_low * incp) <= ilon <= _nint(lon_up * incp)
                and _nint(lat_low * incp) <= ilat <= _nint(lat_up * incp)):
            result = point
    return result


def aggregate_vec(
    nbpt: int, lalo, neighbours, resolution, contfrac, iml: int, lon_rel, lat_rel,
    resol_lon: float, resol_lat: float, callsign: str, incmax: int,
) -> AggregationVectorResult:
    """Port of ``interpol_help.f90::aggregate_vec`` lines 473-779."""

    del callsign
    coordinates, resol = _common_inputs(nbpt, lalo, neighbours, resolution, contfrac, incmax)
    lon = np.asarray(lon_rel, dtype=np.float64)
    lat = np.asarray(lat_rel, dtype=np.float64)
    if iml < 0 or lon.shape != (iml,) or lat.shape != (iml,):
        raise ValueError(f"lon_rel and lat_rel must have shape {(iml,)}")
    if not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
        raise ValueError("source coordinates must be finite")
    if not np.isfinite(resol_lon) or not np.isfinite(resol_lat) or resol_lon <= 0.0 or resol_lat <= 0.0:
        raise ValueError("resol_lon and resol_lat must be finite and positive")

    bounds = [_target_bounds(coordinates, resol, point) for point in range(nbpt)]
    min_lon_point = int(np.argmin(coordinates[:, 1]))
    max_lon_point = int(np.argmax(coordinates[:, 1]))
    domain_minlon = bounds[min_lon_point][0]
    domain_maxlon = bounds[max_lon_point][1]
    yscale = np.pi / 180.0 * R_EARTH
    domain_minlat = np.min(coordinates[:, 0]) - resol[max_lon_point, 1] / (2.0 * yscale)
    domain_maxlat = np.max(coordinates[:, 0]) + resol[max_lon_point, 1] / (2.0 * yscale)
    min_x = int(np.argmin(resol[:, 0]))
    minlon = resol[min_x, 0] / (
        2.0 * max(np.cos(np.deg2rad(coordinates[min_x, 0])), MIN_COS) * np.pi / 180.0 * R_EARTH
    )
    minlat = np.min(resol[:, 1]) / (2.0 * yscale)
    mini = min(minlon, minlat)
    incp = 100 if mini < 0.1 else 10
    # The source's ELSE IF mini<0.01 is unreachable after IF mini<0.1.

    whole_globe = (
        domain_minlon <= -179.5 and domain_maxlon >= 179.5
        and domain_minlat <= -89.5 and domain_maxlat >= 89.5
    )
    search: list[int] = []
    for source in range(iml):
        xscale = max(np.cos(np.deg2rad(lat[source])), MIN_COS) * np.pi / 180.0 * R_EARTH
        source_bounds = (
            max(lon[source] - resol_lon / (2.0 * xscale), -180.0),
            min(lon[source] + resol_lon / (2.0 * xscale), 180.0),
            max(lat[source] - resol_lat / (2.0 * yscale), -90.0),
            min(lat[source] + resol_lat / (2.0 * yscale), 90.0),
        )
        if whole_globe or (
            source_bounds[1] >= domain_minlon and source_bounds[0] <= domain_maxlon
            and source_bounds[3] >= domain_minlat and source_bounds[2] <= domain_maxlat
        ):
            search.append(source)

    indices = np.zeros((nbpt, incmax), dtype=np.int32)
    areas = np.full((nbpt, incmax), UNDEFINED_AREA, dtype=np.float64)
    counts = np.zeros(nbpt, dtype=np.int32)
    for source in search:
        xscale = max(np.cos(np.deg2rad(lat[source])), MIN_COS) * np.pi / 180.0 * R_EARTH
        source_lon_low = max(lon[source] - resol_lon / (2.0 * xscale), domain_minlon)
        source_lon_up = min(lon[source] + resol_lon / (2.0 * xscale), domain_maxlon)
        source_lat_low = max(lat[source] - resol_lat / (2.0 * yscale), domain_minlat)
        source_lat_up = min(lat[source] + resol_lat / (2.0 * yscale), domain_maxlat)
        query_points = (
            (lon[source], lat[source]),
            (source_lon_up, source_lat_up),
            (source_lon_up, source_lat_low),
            (source_lon_low, source_lat_up),
            (source_lon_low, source_lat_low),
        )
        candidates: list[int] = []
        for qlon, qlat in query_points:
            target = _coarse_lookup(qlon, qlat, bounds, incp)
            if target != 0 and target not in candidates:
                candidates.append(target)
        for target_one_based in candidates:
            target = target_one_based - 1
            lon_low, lon_up, lat_low, lat_up = bounds[target]
            if _overlaps(lon[source], source_lon_low, source_lon_up, lon_low, lon_up) and _overlaps(
                lat[source], source_lat_low, source_lat_up, lat_low, lat_up
            ):
                counts[target] += 1
                if counts[target] > incmax:
                    return AggregationVectorResult(indices, areas, False)
                slot = counts[target] - 1
                indices[target, slot] = source + 1
                areas[target, slot] = _rectangle_overlap_area(
                    lon_low, lon_up, lat_low, lat_up, source_lon_low, source_lon_up,
                    source_lat_low, source_lat_up, lat[source],
                )
    return AggregationVectorResult(indices, areas, True)


def _parallel_boundary(name: str, size: int, backend):
    if int(size) < 1:
        raise ValueError("size must be positive")
    if int(size) > 1 and backend is None:
        raise MPIBackendRequiredError(f"{name}: multi-rank gather/broadcast/scatter requires an MPI backend")


def aggregate_vec_p(*args, grid_type: str = "RegLonLat", size: int = 1,
                    backend: ParallelAggregateBackend | None = None, regxy_kwargs=None, **kwargs):
    """Serial identity or explicit MPI boundary for source lines 782-828."""

    _parallel_boundary("aggregate_vec_p", size, backend)
    if size > 1:
        return backend.aggregate_vec_p(aggregate_vec, args=args, kwargs=kwargs, grid_type=grid_type,
                                       regxy_kwargs=regxy_kwargs)
    if "RegLonLat" in grid_type:
        return aggregate_vec(*args, **kwargs)
    if "RegXY" in grid_type:
        if regxy_kwargs is None:
            raise ValueError("aggregate_vec_p: RegXY requires explicit projection/grid geometry inputs")
        from jax_orchidee.driver.geometry_interregxy import interregxy_aggrve
        return interregxy_aggrve(*args, **kwargs, **regxy_kwargs)
    raise ValueError("aggregate_vec_p supports only RegLonLat or RegXY grids")


def aggregate_2d_p(*args, grid_type: str = "RegLonLat", size: int = 1,
                   backend: ParallelAggregateBackend | None = None, regxy_kwargs=None, **kwargs):
    """Serial identity or explicit MPI boundary for source lines 830-889."""

    _parallel_boundary("aggregate_2d_p", size, backend)
    if size > 1:
        return backend.aggregate_2d_p(aggregate_2d, args=args, kwargs=kwargs, grid_type=grid_type,
                                      regxy_kwargs=regxy_kwargs)
    if "RegLonLat" in grid_type:
        return aggregate_2d(*args, **kwargs)
    if "RegXY" in grid_type:
        if regxy_kwargs is None:
            raise ValueError("aggregate_2d_p: RegXY requires explicit projection/grid geometry inputs")
        from jax_orchidee.driver.geometry_interregxy import interregxy_aggr2d
        return interregxy_aggr2d(*args, **kwargs, **regxy_kwargs)
    raise ValueError("aggregate_2d_p supports only RegLonLat or RegXY grids")
