"""Domain zoom and forcing-buffer owners from ``readdim2.f90``.

The public contracts use Fortran ``(i, j)`` spatial order and one-based,
inclusive forcing time indices. NetCDF/file order is converted explicitly at
the IO boundary; cached values are always stored as ``(i, j, time)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DOMAIN_SIZE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_driver/readdim2.f90::domain_size lines 2263-2341"
)
FLINGET_BUFFER_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_driver/readdim2.f90::flinget_buffer lines 2345-2475"
)


class ForcingDomainIOError(ValueError):
    """Raised when forcing geometry or data violates the Fortran contract."""


class MPIInfrastructureBoundary(RuntimeError):
    """Raised when a distributed call has no explicitly supplied MPI backend."""


@dataclass(frozen=True)
class IndexBounds:
    """One-based inclusive bounds for one contiguous selected index segment."""

    begin: int
    end: int


@dataclass(frozen=True)
class DomainSize:
    """Selected regular-grid domain and its Fortran-compatible index metadata."""

    iim: int
    jjm: int
    iind: np.ndarray
    jind: np.ndarray
    iind_zero_based: np.ndarray
    jind_zero_based: np.ndarray
    iim_g_begin: int
    iim_g_end: int
    jjm_g_begin: int
    jjm_g_end: int
    i_segments: tuple[IndexBounds, ...]
    j_segments: tuple[IndexBounds, ...]
    over_dateline: bool

    @property
    def i_indices_one_based(self) -> np.ndarray:
        """Compatibility name retained from the concurrent parent draft."""

        return self.iind

    @property
    def j_indices_one_based(self) -> np.ndarray:
        """Compatibility name retained from the concurrent parent draft."""

        return self.jind


# Keep the parent's result type name as a non-divergent alias.
DomainSelection = DomainSize


def _serial_only(mpi_size: int, operation: str) -> None:
    if isinstance(mpi_size, bool) or int(mpi_size) < 1:
        raise ForcingDomainIOError("mpi_size must be a positive integer")
    if int(mpi_size) != 1:
        raise MPIInfrastructureBoundary(
            f"{operation} requires an MPI partition/broadcast backend for mpi_size != 1"
        )


def _regular_axes(lon: Any, lat: Any) -> tuple[np.ndarray, np.ndarray]:
    longitude = np.asarray(lon, dtype=np.float64)
    latitude = np.asarray(lat, dtype=np.float64)
    if longitude.ndim == latitude.ndim == 1:
        lon_axis, lat_axis = longitude, latitude
    elif longitude.ndim == latitude.ndim == 2 and longitude.shape == latitude.shape:
        if longitude.size == 0:
            raise ForcingDomainIOError("forcing grid must be non-empty")
        lon_axis = longitude[:, 0]
        lat_axis = latitude[0, :]
        if not np.allclose(longitude, lon_axis[:, None], rtol=0.0, atol=1e-10):
            raise ForcingDomainIOError(
                "longitude must define a regular Fortran (i,j) grid"
            )
        if not np.allclose(latitude, lat_axis[None, :], rtol=0.0, atol=1e-10):
            raise ForcingDomainIOError(
                "latitude must define a regular Fortran (i,j) grid"
            )
    else:
        raise ForcingDomainIOError(
            "lon and lat must be non-empty 1-D axes or equal-shape 2-D arrays"
        )
    if lon_axis.size == 0 or lat_axis.size == 0:
        raise ForcingDomainIOError("forcing grid must be non-empty")
    if not np.all(np.isfinite(lon_axis)) or not np.all(np.isfinite(lat_axis)):
        raise ForcingDomainIOError("forcing longitude and latitude must be finite")
    return lon_axis, lat_axis


def _normalize_longitude(values: np.ndarray) -> np.ndarray:
    normalized = np.mod(values + 180.0, 360.0) - 180.0
    # The source keeps +180 rather than converting it to -180.
    return np.where((normalized == -180.0) & (values > 0.0), 180.0, normalized)


def _segments(indices: np.ndarray) -> tuple[IndexBounds, ...]:
    if indices.size == 0:
        return ()
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    groups = np.split(indices, breaks)
    return tuple(IndexBounds(int(group[0]) + 1, int(group[-1]) + 1) for group in groups)


def domain_size(
    limit_west: float,
    limit_east: float,
    limit_north: float,
    limit_south: float,
    lon: Any,
    lat: Any,
    *,
    mpi_size: int = 1,
) -> DomainSize:
    """Select a regular forcing-grid zoom.

    Fortran provenance: ``readdim2.f90::domain_size``, lines 2263-2341.
    The source's inclusive dateline arm is retained exactly: when west is
    greater than east it selects ``lon <= west OR lon >= east``.
    """

    _serial_only(mpi_size, "domain_size")
    west, east = float(limit_west), float(limit_east)
    north, south = float(limit_north), float(limit_south)
    if not np.all(np.isfinite([west, east, north, south])):
        raise ForcingDomainIOError("domain limits must be finite")
    if abs(west) > 180.0 or abs(east) > 180.0:
        raise ForcingDomainIOError(
            "longitude limits limit_west and limit_east must lie in [-180, 180]"
        )
    if south < -90.0 or north > 90.0 or south >= north:
        raise ForcingDomainIOError("latitude limits require -90 <= south < north <= 90")

    lon_axis, lat_axis = _regular_axes(lon, lat)
    normalized_lon = _normalize_longitude(lon_axis)
    over_dateline = west > east
    if over_dateline:
        i_mask = (normalized_lon <= west) | (normalized_lon >= east)
    else:
        i_mask = (normalized_lon >= west) & (normalized_lon <= east)
    j_mask = (lat_axis >= south) & (lat_axis <= north)
    i0 = np.flatnonzero(i_mask).astype(np.int64)
    j0 = np.flatnonzero(j_mask).astype(np.int64)

    i_segments = _segments(i0)
    j_segments = _segments(j0)
    # Lines 2312-2313 and 2331-2332 update the module bounds from the raw
    # source axes, independently of the normalized/dateline selection mask.
    # At full-domain edges Fortran leaves a bound unset; retain a deterministic
    # selected-segment fallback only for that undefined source case.
    i_begin_updates = np.flatnonzero(lon_axis < west) + 2
    i_end_updates = np.flatnonzero(lon_axis < east) + 1
    j_begin_updates = np.flatnonzero(lat_axis > north) + 2
    j_end_updates = np.flatnonzero(lat_axis > south) + 1
    i_begin = (
        int(i_begin_updates[-1])
        if i_begin_updates.size
        else (i_segments[0].begin if i_segments else 0)
    )
    i_end = (
        int(i_end_updates[-1])
        if i_end_updates.size
        else (i_segments[-1].end if i_segments else 0)
    )
    j_begin = (
        int(j_begin_updates[-1])
        if j_begin_updates.size
        else (j_segments[0].begin if j_segments else 0)
    )
    j_end = (
        int(j_end_updates[-1])
        if j_end_updates.size
        else (j_segments[-1].end if j_segments else 0)
    )
    return DomainSize(
        iim=int(i0.size),
        jjm=int(j0.size),
        iind=i0 + 1,
        jind=j0 + 1,
        iind_zero_based=i0,
        jind_zero_based=j0,
        iim_g_begin=i_begin,
        iim_g_end=i_end,
        jjm_g_begin=j_begin,
        jjm_g_end=j_end,
        i_segments=i_segments,
        j_segments=j_segments,
        over_dateline=over_dateline,
    )


_AXIS_ALIASES = {
    "x": "x",
    "i": "x",
    "lon": "x",
    "longitude": "x",
    "x_grid": "x",
    "y": "y",
    "j": "y",
    "lat": "y",
    "latitude": "y",
    "y_grid": "y",
    "t": "time",
    "time": "time",
    "tstep": "time",
    "time_counter": "time",
    "z": "level",
    "lev": "level",
    "level": "level",
    "levels": "level",
    "height": "level",
    "depth": "level",
}


@dataclass
class _BufferedVariable:
    start: int
    end: int
    data: np.ndarray


def _canonical_axes(dimensions: Sequence[str], *, context: str) -> tuple[str, ...]:
    result: list[str] = []
    for name in dimensions:
        key = str(name).strip().lower()
        try:
            result.append(_AXIS_ALIASES[key])
        except KeyError as exc:
            raise ForcingDomainIOError(
                f"{context} has unsupported axis {name!r}; provide explicit x/y/time axis_order"
            ) from exc
    if len(set(result)) != len(result):
        raise ForcingDomainIOError(f"{context} repeats a canonical axis: {result}")
    return tuple(result)


class FlingetBuffer:
    """Stateful, per-variable forcing buffer with Fortran time semantics."""

    def __init__(
        self,
        source: str | PathLike[str] | Mapping[str, Any] | Any,
        *,
        nbuff: int = 1,
        axis_order: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
        max_variables: int = 20,
        mpi_size: int = 1,
    ) -> None:
        _serial_only(mpi_size, "flinget_buffer")
        if isinstance(nbuff, bool) or int(nbuff) != nbuff or int(nbuff) < 0:
            raise ForcingDomainIOError("NBUFF must be a non-negative integer")
        if isinstance(max_variables, bool) or int(max_variables) < 1:
            raise ForcingDomainIOError("max_variables must be a positive integer")
        self.source = source
        self.requested_nbuff = int(nbuff)
        self.axis_order = axis_order
        self.max_variables = int(max_variables)
        self._ttm: int | None = None
        self._nbuff: int | None = None
        self._variables: dict[str, _BufferedVariable] = {}
        self.read_count: dict[str, int] = {}

    @property
    def nbuff(self) -> int | None:
        return self._nbuff

    def _dimensions_for(self, name: str, default: Sequence[str]) -> tuple[str, ...]:
        order = self.axis_order
        if isinstance(order, Mapping):
            value = order.get(name, default)
        elif order is None:
            value = default
        else:
            value = order
        return tuple(str(item) for item in value)

    def _array_and_dimensions(self, name: str) -> tuple[np.ndarray, tuple[str, ...]]:
        source = self.source
        if isinstance(source, (str, PathLike)):
            import xarray as xr

            path = Path(source)
            try:
                with xr.open_dataset(path, decode_cf=False, cache=False) as dataset:
                    if name not in dataset:
                        raise ForcingDomainIOError(f"variable {name!r} was not found")
                    variable = dataset[name]
                    return np.asarray(variable.values), self._dimensions_for(
                        name, variable.dims
                    )
            except ForcingDomainIOError:
                raise
            except Exception as exc:
                raise ForcingDomainIOError(
                    f"could not read variable {name!r} from {str(path)!r}: {exc}"
                ) from exc

        if hasattr(source, "variables") and hasattr(source, "__getitem__"):
            if name not in source:
                raise ForcingDomainIOError(f"variable {name!r} was not found")
            variable = source[name]
            dims = getattr(variable, "dims", getattr(variable, "dimensions", ()))
            return np.asarray(variable.values), self._dimensions_for(name, dims)

        if not isinstance(source, Mapping) or name not in source:
            raise ForcingDomainIOError(f"variable {name!r} was not found")
        value = source[name]
        explicit_dims: Sequence[str] | None = None
        if isinstance(value, tuple) and len(value) == 2:
            value, explicit_dims = value
        array = np.asarray(value)
        default = ("x", "y") if array.ndim == 2 else ("x", "y", "time")
        return array, self._dimensions_for(name, explicit_dims or default)

    def _load_chunk(
        self,
        name: str,
        *,
        iim: int,
        jjm: int,
        llm: int,
        ttm: int,
        start: int,
        end: int,
    ) -> np.ndarray:
        array, dimensions = self._array_and_dimensions(name)
        if array.ndim != len(dimensions):
            raise ForcingDomainIOError(
                f"variable {name!r} rank {array.ndim} does not match axis_order {dimensions}"
            )
        axes = _canonical_axes(dimensions, context=f"variable {name!r}")
        if "x" not in axes or "y" not in axes:
            raise ForcingDomainIOError(f"variable {name!r} requires x and y axes")
        if "level" in axes:
            level_axis = axes.index("level")
            if array.shape[level_axis] != llm or llm != 1:
                raise ForcingDomainIOError(
                    "flinget_buffer's 2-D output supports only a singleton level axis"
                )
            array = np.take(array, 0, axis=level_axis)
            axes = tuple(axis for axis in axes if axis != "level")
        elif llm != 1:
            raise ForcingDomainIOError(
                "llm_full must be 1 for a variable without a level axis"
            )

        has_time = "time" in axes
        if has_time:
            time_axis = axes.index("time")
            if array.shape[time_axis] != ttm:
                raise ForcingDomainIOError(
                    f"variable {name!r} time length {array.shape[time_axis]} != ttm {ttm}"
                )
            slicer = [slice(None)] * array.ndim
            slicer[time_axis] = slice(start - 1, end)
            array = array[tuple(slicer)]
        elif ttm != 1:
            raise ForcingDomainIOError(
                f"variable {name!r} has no time axis but ttm is {ttm}, not 1"
            )
        else:
            array = np.expand_dims(array, axis=-1)
            axes = axes + ("time",)

        permutation = tuple(axes.index(axis) for axis in ("x", "y", "time"))
        result = np.asarray(np.transpose(array, permutation))
        expected = (iim, jjm, end - start + 1)
        if result.shape != expected:
            raise ForcingDomainIOError(
                f"variable {name!r} normalized shape {result.shape} != expected {expected}"
            )
        return result.copy()

    def read(
        self,
        varname: str,
        iim_full: int,
        jjm_full: int,
        llm_full: int,
        ttm: int,
        itb: int,
        ite: int,
        *,
        mpi_size: int = 1,
    ) -> np.ndarray:
        """Return one ``(i,j)`` slab for one-based timestep ``itb``."""

        _serial_only(mpi_size, "flinget_buffer")
        name = str(varname).strip()
        if not name:
            raise ForcingDomainIOError("varname must be non-empty")
        dimensions = (iim_full, jjm_full, llm_full, ttm, itb, ite)
        if any(isinstance(value, bool) or int(value) != value for value in dimensions):
            raise ForcingDomainIOError(
                "forcing dimensions and time indices must be integers"
            )
        iim, jjm, llm, nt, begin, finish = map(int, dimensions)
        if min(iim, jjm, llm, nt) < 1:
            raise ForcingDomainIOError("forcing dimensions and ttm must be positive")
        if begin != finish:
            raise ForcingDomainIOError("ite must equal itb")
        if begin < 1 or begin > nt:
            raise EOFError(f"forcing timestep {begin} is outside 1..{nt}")

        if self._ttm is None:
            self._ttm = nt
            self._nbuff = (
                nt
                if self.requested_nbuff == 0 or self.requested_nbuff > nt
                else self.requested_nbuff
            )
        elif nt != self._ttm:
            raise ForcingDomainIOError(
                f"ttm {nt} differs from initialized ttm {self._ttm}"
            )
        assert self._nbuff is not None

        buffered = self._variables.get(name)
        if buffered is None and len(self._variables) >= self.max_variables:
            raise ForcingDomainIOError(
                f"too many buffered variables; max_variables={self.max_variables}"
            )
        if buffered is None or begin < buffered.start or begin > buffered.end:
            end = min(nt, begin + self._nbuff - 1)
            data = self._load_chunk(
                name,
                iim=iim,
                jjm=jjm,
                llm=llm,
                ttm=nt,
                start=begin,
                end=end,
            )
            buffered = _BufferedVariable(begin, end, data)
            self._variables[name] = buffered
            self.read_count[name] = self.read_count.get(name, 0) + 1
        index = begin - buffered.start
        return buffered.data[:, :, index].copy()


def flinget_buffer(
    buffer: FlingetBuffer,
    varname: str,
    iim_full: int,
    jjm_full: int,
    llm_full: int,
    ttm: int,
    itb: int,
    ite: int,
    *,
    mpi_size: int = 1,
) -> np.ndarray:
    """Procedural adapter matching ``readdim2.f90::flinget_buffer`` arguments."""

    if not isinstance(buffer, FlingetBuffer):
        raise TypeError("buffer must be a FlingetBuffer")
    return buffer.read(
        varname,
        iim_full,
        jjm_full,
        llm_full,
        ttm,
        itb,
        ite,
        mpi_size=mpi_size,
    )
