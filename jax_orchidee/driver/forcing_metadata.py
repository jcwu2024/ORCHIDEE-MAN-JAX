"""Strict forcing metadata owners for ``readdim2``.

Fortran provenance: ``src_driver/readdim2.f90::forcing_info`` lines 49-495
and ``forcing_vertical_ioipsl`` lines 2057-2257.  IOIPSL fallback branches
are represented by explicit configuration or explicit errors; this boundary
never invents missing file data.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import xarray as xr

from jax_orchidee.driver.domain import _closed_domain_indices
from jax_orchidee.driver.interpolation_file_dims import (
    InterpolationFileMetadataError,
    read_variable_dimension_metadata,
)


FORCING_INFO_PROVENANCE = "fortran_source/ORCHIDEE/src_driver/readdim2.f90::forcing_info lines 49-495"
VERTICAL_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_driver/readdim2.f90::forcing_vertical_ioipsl lines 2057-2257"
)
FORCING_METADATA_PROVENANCE = (FORCING_INFO_PROVENANCE, VERTICAL_PROVENANCE)


class ForcingMetadataError(ValueError):
    """A forcing file cannot satisfy the source-backed metadata contract."""


class ForcingMetadataReadError(OSError):
    """A forcing source could not be opened."""


@dataclass(frozen=True)
class NetCDFVariableMetadata:
    name: str
    file_dimensions: tuple[str, ...]
    file_shape: tuple[int, ...]
    fortran_dimensions: tuple[str, ...]
    fortran_shape: tuple[int, ...]
    attributes: Mapping[str, Any]
    time_dimension: str | None = None
    level_dimension: str | None = None


@dataclass(frozen=True)
class VerticalCoordinate:
    zfixed: bool = False
    zsigma: bool = False
    zhybrid: bool = False
    zlevels: bool = False
    zheight: bool = False
    zsamelev_uv: bool = True
    zlev_fixed: float | None = None
    zlevuv_fixed: float | None = None
    zhybrid_a: float | None = None
    zhybrid_b: float | None = None
    zhybriduv_a: float | None = None
    zhybriduv_b: float | None = None
    source_variables: tuple[str, ...] = ()
    used_run_def: bool = False
    warning: str | None = None
    kind: str = ""
    is_watchout: bool = False
    thermodynamic_level_count: int = 1
    wind_level_count: int = 1
    thermodynamic_values: np.ndarray | None = None
    wind_values: np.ndarray | None = None
    hybrid_a_values: np.ndarray | None = None
    hybrid_b_values: np.ndarray | None = None
    hybrid_uv_a_values: np.ndarray | None = None
    hybrid_uv_b_values: np.ndarray | None = None


@dataclass(frozen=True)
class ForcingInfo:
    filename: str
    iim_full: int
    jjm_full: int
    llm_full: int
    ttm_full: int
    iim: int
    jjm: int
    llm: int
    tm: int
    date0: float
    dt_force: float
    calendar: str
    one_year: float
    one_day: float
    have_zaxis: bool
    interpol: bool
    daily_interpol: bool
    weathergen: bool
    is_watchout: bool
    i_index_fortran: np.ndarray
    j_index_fortran: np.ndarray
    land_index_fortran: np.ndarray
    nbpoint: int
    vertical: VerticalCoordinate
    variables: Mapping[str, NetCDFVariableMetadata]


ForcingFileMetadata = ForcingInfo
ForcingVerticalMetadata = VerticalCoordinate


@contextmanager
def _open(source: str | PathLike[str] | xr.Dataset) -> Iterator[tuple[xr.Dataset, str]]:
    if isinstance(source, xr.Dataset):
        yield source, "<xarray.Dataset>"
        return
    path = Path(source)
    try:
        ds = xr.open_dataset(path, decode_times=False, mask_and_scale=False, cache=False)
    except Exception as exc:
        raise ForcingMetadataReadError(f"could not open forcing file {path}") from exc
    try:
        yield ds, str(path)
    finally:
        ds.close()


def _require(ds: xr.Dataset, names: Sequence[str], context: str) -> None:
    missing = [name for name in names if name not in ds.variables]
    if missing:
        raise ForcingMetadataError(f"{context} missing required variables: {missing}")


def _metadata_mapping(ds: xr.Dataset) -> dict[str, object]:
    return {
        "dimensions": {name: int(size) for name, size in ds.sizes.items()},
        "variables": {
            name: {"dimensions": tuple(var.dims), "attributes": dict(var.attrs)}
            for name, var in ds.variables.items()
        },
    }


def _variable_metadata(ds: xr.Dataset) -> dict[str, NetCDFVariableMetadata]:
    source = _metadata_mapping(ds)
    result: dict[str, NetCDFVariableMetadata] = {}
    for name, var in ds.variables.items():
        try:
            inspected = read_variable_dimension_metadata(source, name)
        except InterpolationFileMetadataError as exc:
            raise ForcingMetadataError(str(exc)) from exc
        result[name] = NetCDFVariableMetadata(
            name=name,
            file_dimensions=inspected.file_dimension_names,
            file_shape=inspected.file_dimension_sizes,
            fortran_dimensions=inspected.fortran_dimension_names,
            fortran_shape=inspected.fortran_dimension_sizes,
            attributes=dict(var.attrs),
            time_dimension=inspected.time_dimension,
            level_dimension=inspected.level_dimension,
        )
    return result


def _coordinate(ds: xr.Dataset, candidates: Sequence[str], *, axis: str) -> str:
    for name in candidates:
        if name in ds.variables:
            return name
    found = []
    for name, var in ds.variables.items():
        attrs = {str(k).lower(): str(v).strip().lower() for k, v in var.attrs.items()}
        if attrs.get("axis") == axis.lower() or attrs.get("standard_name") == (
            "longitude" if axis == "X" else "latitude"
        ):
            found.append(name)
    if len(found) != 1:
        raise ForcingMetadataError(f"forcing file must identify one {axis} coordinate; found {found}")
    return found[0]


def _grid(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, str, str]:
    lon_name = _coordinate(ds, ("nav_lon", "lon", "longitude"), axis="X")
    lat_name = _coordinate(ds, ("nav_lat", "lat", "latitude"), axis="Y")
    lon_var, lat_var = ds[lon_name], ds[lat_name]
    if lon_var.ndim == lat_var.ndim == 1:
        x_dim, y_dim = lon_var.dims[0], lat_var.dims[0]
        lon, lat = np.meshgrid(
            np.asarray(lon_var.values, dtype=np.float64),
            np.asarray(lat_var.values, dtype=np.float64),
            indexing="ij",
        )
    elif lon_var.ndim == lat_var.ndim == 2 and lon_var.dims == lat_var.dims:
        y_dim, x_dim = lon_var.dims
        lon = np.asarray(lon_var.values, dtype=np.float64).T.copy()
        lat = np.asarray(lat_var.values, dtype=np.float64).T.copy()
    else:
        raise ForcingMetadataError("longitude/latitude must be matching 1-D axes or 2-D fields")
    if not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
        raise ForcingMetadataError("longitude/latitude contain missing values")
    return lon, lat, x_dim, y_dim


_TIME_RE = re.compile(
    r"^\s*(seconds?|minutes?|hours?|days?)\s+since\s+(.+?)\s*$", re.IGNORECASE
)


def _time(ds: xr.Dataset) -> tuple[str, np.ndarray, float, str]:
    candidates = []
    for name, var in ds.variables.items():
        attrs = {str(k).lower(): str(v).strip().lower() for k, v in var.attrs.items()}
        if var.ndim == 1 and (
            name.lower() in {"time", "times", "tstep", "time_counter"}
            or attrs.get("axis") == "t"
            or attrs.get("standard_name") == "time"
        ):
            candidates.append(name)
    if len(candidates) != 1:
        raise ForcingMetadataError(f"forcing file must identify one time coordinate; found {candidates}")
    name = candidates[0]
    values = np.asarray(ds[name].values, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values[:2])):
        raise ForcingMetadataError("forcing file must contain at least two finite time steps")
    units = str(ds[name].attrs.get("units", ""))
    match = _TIME_RE.match(units)
    if match is None:
        raise ForcingMetadataError("forcing time units must be '<unit> since <origin>'")
    scales = {"second": 1.0, "minute": 60.0, "hour": 3600.0, "day": 86400.0}
    scale = scales[match.group(1).lower().rstrip("s")]
    dt = float(values[1] - values[0]) * scale
    if dt <= 0.0 or not np.isfinite(dt):
        raise ForcingMetadataError("forcing time coordinate must increase")
    calendar = str(ds[name].attrs.get("calendar", "")).split("\x00", 1)[0].strip()
    return name, values, dt, calendar


def _calendar(calendar: str) -> tuple[float, float]:
    key = calendar.lower()
    if key in {"noleap", "365_day"}:
        return 365.0, 1.0
    if key in {"360d", "360_day"}:
        return 360.0, 1.0
    if key == "julian":
        return 365.25, 1.0
    if key in {"gregorian", "standard", "proleptic_gregorian"}:
        return 365.2425, 1.0
    raise ForcingMetadataError(f"unsupported forcing calendar {calendar!r}")


def _values(ds: xr.Dataset, name: str) -> np.ndarray:
    values = np.asarray(ds[name].values, dtype=np.float64)
    if values.size < 1 or not np.all(np.isfinite(values)):
        raise ForcingMetadataError(f"vertical variable {name!r} is empty or contains missing values")
    return values.copy()


def _level_count(var: xr.DataArray, horizontal: set[str], time_dim: str | None) -> int:
    excluded = set(horizontal)
    if time_dim:
        excluded.add(time_dim)
    residual = [name for name in var.dims if name not in excluded]
    if len(residual) > 1:
        raise ForcingMetadataError(f"vertical variable {var.name!r} has multiple level dimensions {residual}")
    return int(var.sizes[residual[0]]) if residual else 1


def _counts(
    ds: xr.Dataset, main: str, uv: str | None, horizontal: set[str], time_dim: str | None
) -> tuple[int, int]:
    nmain = _level_count(ds[main], horizontal, time_dim)
    nuv = nmain if uv is None else _level_count(ds[uv], horizontal, time_dim)
    if nmain != nuv:
        raise ForcingMetadataError(
            f"thermodynamic and wind vertical level counts differ: {nmain} != {nuv}"
        )
    return nmain, nuv


def forcing_vertical_ioipsl(
    source: str | PathLike[str] | xr.Dataset | Mapping[str, Any] | None,
    *,
    force_id: int = 1,
    height_lev1: float | None = None,
    height_levw: float | None = None,
    horizontal_dimensions: Sequence[str] = ("x", "y", "lon", "lat"),
    time_dimension: str | None = None,
) -> VerticalCoordinate:
    """Resolve vertical convention in Fortran priority order."""

    if force_id <= 0:
        if height_lev1 is None:
            height_lev1 = 2.0
        if height_levw is None:
            height_levw = 10.0
        same = abs(height_levw - height_lev1) <= np.spacing(float(height_lev1))
        return VerticalCoordinate(
            zheight=True, zsamelev_uv=bool(same), zlev_fixed=float(height_lev1),
            zlevuv_fixed=float(height_levw), used_run_def=True, kind="configured_height",
            thermodynamic_values=np.asarray([height_lev1]), wind_values=np.asarray([height_levw]),
        )
    if source is None:
        raise ForcingMetadataError("positive force_id requires a forcing source")
    if isinstance(source, Mapping) and not isinstance(source, xr.Dataset):
        data_vars = {name: xr.DataArray(np.asarray(value.values if hasattr(value, "values") else value)) for name, value in source.items()}
        ds_context = _open(xr.Dataset(data_vars))
    else:
        ds_context = _open(source)
    with ds_context as (ds, _):
        names = set(ds.variables)
        horizontal = set(horizontal_dimensions)
        if "Sigma" in names:
            uv = "Sigma_uv" if "Sigma_uv" in names else None
            main, wind = _values(ds, "Sigma"), _values(ds, uv) if uv else _values(ds, "Sigma")
            nmain, nuv = _counts(ds, "Sigma", uv, horizontal, time_dimension)
            return VerticalCoordinate(
                zsigma=True, zsamelev_uv=uv is None, zhybrid_a=0.0,
                zhybrid_b=float(main.reshape(-1)[0]), zhybriduv_a=0.0 if uv else None,
                zhybriduv_b=float(wind.reshape(-1)[0]) if uv else None,
                source_variables=("Sigma",) + ((uv,) if uv else ()), kind="sigma",
                thermodynamic_level_count=nmain, wind_level_count=nuv,
                thermodynamic_values=main, wind_values=wind,
                hybrid_a_values=np.zeros_like(main), hybrid_b_values=main,
                hybrid_uv_a_values=np.zeros_like(wind), hybrid_uv_b_values=wind,
            )
        has_a, has_b = "HybSigA" in names, "HybSigB" in names
        has_ua, has_ub = "HybSigA_uv" in names, "HybSigB_uv" in names
        if has_a and not has_b:
            raise ForcingMetadataError("Missing the B coefficient for Hybrid vertical levels for T and Q")
        if has_ua and not has_ub:
            raise ForcingMetadataError("Missing the B coefficient for Hybrid vertical levels for U and V")
        if has_a:
            a, b = _values(ds, "HybSigA"), _values(ds, "HybSigB")
            if a.shape != b.shape:
                raise ForcingMetadataError("thermodynamic hybrid A/B coefficient shapes differ")
            ua, ub = (_values(ds, "HybSigA_uv"), _values(ds, "HybSigB_uv")) if has_ua else (a.copy(), b.copy())
            if ua.shape != ub.shape:
                raise ForcingMetadataError("wind hybrid A/B coefficient shapes differ")
            nmain, nuv = _counts(ds, "HybSigA", "HybSigA_uv" if has_ua else None, horizontal, time_dimension)
            return VerticalCoordinate(
                zhybrid=True, zsamelev_uv=not has_ua, zhybrid_a=float(a.reshape(-1)[0]),
                zhybrid_b=float(b.reshape(-1)[0]), zhybriduv_a=float(ua.reshape(-1)[0]) if has_ua else None,
                zhybriduv_b=float(ub.reshape(-1)[0]) if has_ua else None,
                source_variables=("HybSigA", "HybSigB") + (("HybSigA_uv", "HybSigB_uv") if has_ua else ()),
                kind="hybrid", thermodynamic_level_count=nmain, wind_level_count=nuv,
                thermodynamic_values=b, wind_values=ub, hybrid_a_values=a, hybrid_b_values=b,
                hybrid_uv_a_values=ua, hybrid_uv_b_values=ub,
            )
        if "Levels" in names:
            uv = "Levels_uv" if "Levels_uv" in names else None
            main, wind = _values(ds, "Levels"), _values(ds, uv) if uv else _values(ds, "Levels")
            nmain, nuv = _counts(ds, "Levels", uv, horizontal, time_dimension)
            return VerticalCoordinate(
                zlevels=True, zsamelev_uv=uv is None, source_variables=("Levels",) + ((uv,) if uv else ()),
                kind="levels", thermodynamic_level_count=nmain, wind_level_count=nuv,
                thermodynamic_values=main, wind_values=wind,
            )
        if "Height_Lev1" in names:
            uv = "Height_Levuv" if "Height_Levuv" in names else None
            main, wind = _values(ds, "Height_Lev1"), _values(ds, uv) if uv else _values(ds, "Height_Lev1")
            return VerticalCoordinate(
                zheight=True, zsamelev_uv=uv is None, zlev_fixed=float(main.reshape(-1)[0]),
                zlevuv_fixed=float(wind.reshape(-1)[0]) if uv else None,
                source_variables=("Height_Lev1",) + ((uv,) if uv else ()), kind="height",
                thermodynamic_values=main, wind_values=wind,
            )
        if "lev" in names:
            value = _values(ds, "lev")
            return VerticalCoordinate(
                zheight=True, zlev_fixed=float(value.reshape(-1)[0]), source_variables=("lev",),
                kind="legacy_height", thermodynamic_values=value, wind_values=value.copy(),
            )
        if height_lev1 is None:
            height_lev1 = 2.0
        if height_levw is None:
            height_levw = 10.0
        same = abs(height_levw - height_lev1) <= np.spacing(float(height_lev1))
        return VerticalCoordinate(
            zheight=True, zsamelev_uv=bool(same), zlev_fixed=float(height_lev1),
            zlevuv_fixed=float(height_levw), used_run_def=True, kind="configured_height",
            warning="vertical metadata absent; HEIGHT_LEV1/HEIGHT_LEVW were used",
            thermodynamic_values=np.asarray([height_lev1]), wind_values=np.asarray([height_levw]),
        )


def _first_horizontal(ds: xr.Dataset, name: str, time_dim: str, y_dim: str, x_dim: str) -> np.ndarray:
    _require(ds, (name,), "forcing metadata")
    var = ds[name]
    if time_dim in var.dims:
        var = var.isel({time_dim: 0})
    residual = [dim for dim in var.dims if dim not in {y_dim, x_dim}]
    if len(residual) > 1:
        raise ForcingMetadataError(f"{name} has multiple level dimensions {residual}")
    if residual:
        # flinget_buffer requests the first level while forcing_info discovers
        # land points; the full level count was validated separately above.
        var = var.isel({residual[0]: 0})
    if tuple(var.dims) != (y_dim, x_dim):
        raise ForcingMetadataError(f"{name} dimensions {var.dims} must reduce to {(y_dim, x_dim)}")
    values = np.asarray(var.values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ForcingMetadataError(f"{name} contains missing values")
    return values.T.copy()


def forcing_info(
    source: str | PathLike[str] | xr.Dataset,
    *,
    date0: float,
    allow_weathergen: bool = False,
    dt_weathgen: float = 1800.0,
    limit_west: float = -180.0,
    limit_east: float = 180.0,
    limit_north: float = 90.0,
    limit_south: float = -90.0,
    zonal_res: float = 2.0,
    merid_res: float = 2.0,
    mpi_size: int = 1,
    height_lev1: float = 2.0,
    height_levw: float = 10.0,
) -> ForcingInfo:
    """Inspect grid, time, variable, land-mask, and vertical metadata."""

    with _open(source) as (ds, filename):
        if not (-180.0 <= limit_west <= 180.0 and -180.0 <= limit_east <= 180.0):
            raise ForcingMetadataError("invalid west/east forcing limits")
        if not (-90.0 <= limit_south < limit_north <= 90.0):
            raise ForcingMetadataError("invalid south/north forcing limits")
        lon, lat, x_dim, y_dim = _grid(ds)
        time_name, times, dt_force, calendar = _time(ds)
        time_dim = ds[time_name].dims[0]
        if not calendar or calendar == "XXXX":
            calendar = "noleap" if allow_weathergen else "gregorian"
        one_year, one_day = _calendar(calendar)
        delta_days = dt_force / 86400.0
        monthly = abs(delta_days - 30.0) <= 2.0
        if monthly:
            if not allow_weathergen:
                raise ForcingMetadataError("monthly forcing requires ALLOW_WEATHERGEN")
            if dt_weathgen <= 0.0 or zonal_res <= 0.0 or merid_res <= 0.0:
                raise ForcingMetadataError("weather-generator timestep and resolution must be positive")
            weathergen, interpol, daily = True, False, False
            dt_force = float(dt_weathgen)
        elif delta_days <= 0.25 or delta_days == 1.0:
            weathergen, interpol, daily = False, True, delta_days == 1.0
        else:
            raise ForcingMetadataError(f"forcing time step {delta_days:g} days is not suitable")

        vertical = forcing_vertical_ioipsl(
            ds, height_lev1=height_lev1, height_levw=height_levw,
            horizontal_dimensions=(x_dim, y_dim), time_dimension=time_dim
        )

        iim_full, jjm_full = lon.shape
        if interpol:
            i0, j0 = _closed_domain_indices(
                lon, lat,
                {"west": limit_west, "east": limit_east,
                 "north": limit_north, "south": limit_south},
            )
            iim, jjm = int(i0.size), int(j0.size)
            if iim == 0 or jjm == 0:
                raise ForcingMetadataError("forcing limits select an empty domain")
            qair = _first_horizontal(ds, "Qair", time_dim, y_dim, x_dim)[np.ix_(i0, j0)]
            if "contfrac" in ds.variables:
                contfrac = _first_horizontal(ds, "contfrac", time_dim, y_dim, x_dim)[np.ix_(i0, j0)]
            else:
                contfrac = np.ones_like(qair)
            valid = (contfrac > np.finfo(np.float32).eps) & (qair < 999999.0)
            local = np.argwhere(valid.T)[:, [1, 0]] if np.any(valid) else np.empty((0, 2), dtype=int)
            if local.size == 0 or np.any(local[:, 0] < 0) or np.any(local[:, 0] >= iim) or np.any(local[:, 1] < 0) or np.any(local[:, 1] >= jjm):
                raise ForcingMetadataError("forcing zoom has no valid landpoint or an index is outside it")
            if np.unique(local, axis=0).shape[0] != local.shape[0]:
                raise ForcingMetadataError("landpoint indices contain duplicates")
            land = local[:, 0] + local[:, 1] * iim + 1
            land = np.sort(land).astype(np.int32)
            i_fortran, j_fortran = i0.astype(np.int32) + 1, j0.astype(np.int32) + 1
            tm = int(times.size)
        else:
            east = limit_east + 360.0 if limit_west > limit_east else limit_east
            lon_width = east - limit_west
            if limit_west >= east or east > 360.0 or limit_west < -180.0 or lon_width > 360.0:
                raise ForcingMetadataError("invalid weather-generator longitude limits")
            if limit_south < -90.0 or limit_north > 90.0 or limit_south >= limit_north:
                raise ForcingMetadataError("invalid weather-generator latitude limits")
            if zonal_res <= 0.0 or zonal_res > lon_width:
                raise ForcingMetadataError("invalid weather-generator zonal resolution")
            if merid_res <= 0.0 or merid_res > limit_north - limit_south:
                raise ForcingMetadataError("invalid weather-generator meridional resolution")
            # weather.f90::weathgen_domain_size lines 3325-3326 use NINT,
            # producing cell counts rather than endpoint-inclusive axes.
            iim = int(np.floor(max(lon_width / zonal_res, 1.0) + 0.5))
            jjm = int(
                np.floor(max((limit_north - limit_south) / merid_res, 1.0) + 0.5)
            )
            i_fortran = np.arange(1, iim + 1, dtype=np.int32)
            j_fortran = np.arange(1, jjm + 1, dtype=np.int32)
            land = np.arange(1, iim * jjm + 1, dtype=np.int32)
            tm = int(round(one_year * 86400.0 / dt_force))
        if land.size < mpi_size:
            raise ForcingMetadataError(f"number of landpoints {land.size} is less than mpi_size {mpi_size}")
        level_dims = [dim for dim in ds.sizes if dim not in {x_dim, y_dim, time_dim}]
        llm_full = vertical.thermodynamic_level_count if level_dims else 0
        return ForcingInfo(
            filename, iim_full, jjm_full, llm_full, int(times.size), iim, jjm, 1, tm,
            float(date0), dt_force, calendar, one_year, one_day, llm_full >= 1,
            interpol, daily, weathergen, vertical.is_watchout, i_fortran, j_fortran,
            land, int(land.size), vertical, _variable_metadata(ds),
        )
