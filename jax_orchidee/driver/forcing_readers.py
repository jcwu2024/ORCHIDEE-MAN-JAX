"""Source-faithful forcing readers from ``src_driver/readdim2.f90``.

Horizontal arrays use Fortran ``(i,j)`` order. File dimensions are inspected
before the usual NetCDF ``(y,x)`` data are transposed; configuration absent
from both the file and explicit arguments is never defaulted here.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, replace
from pathlib import Path
from typing import Callable, Mapping, MutableMapping

import numpy as np
import xarray as xr

CP_AIR = 1004.675
GRAVITY = 9.80665
MOLAR_GAS_CONSTANT = 287.05
VAL_EXP = 1.0e20


class ForcingReaderError(ValueError):
    """Fatal forcing contract error corresponding to source ``ipslerr``."""


@dataclass(frozen=True)
class VerticalCoordinates:
    mode: str
    same_uv: bool
    lev1: float | None = None
    levuv: float | None = None
    a: float | None = None
    b: float | None = None
    a_uv: float | None = None
    b_uv: float | None = None


@dataclass(frozen=True)
class VerticalCoordinate:
    """Compatibility spelling for explicit array-boundary callers."""

    kind: str
    same_wind_level: bool
    level_tq: float | None = None
    level_uv: float | None = None
    hybrid_a: float | None = None
    hybrid_b: float | None = None
    hybrid_uv_a: float | None = None
    hybrid_uv_b: float | None = None

    def normalized(self) -> VerticalCoordinates:
        return VerticalCoordinates(
            self.kind, self.same_wind_level, self.level_tq, self.level_uv,
            self.hybrid_a, self.hybrid_b, self.hybrid_uv_a, self.hybrid_uv_b,
        )


@dataclass(frozen=True)
class VerticalForcingOptions:
    zheight: bool = False
    zsigma: bool = False
    zhybrid: bool = False
    zlevels: bool = False
    zsamelev_uv: bool = False
    zlev_fixed: float | None = None
    zlevuv_fixed: float | None = None
    zhybrid_a: float | None = None
    zhybrid_b: float | None = None
    zhybriduv_a: float | None = None
    zhybriduv_b: float | None = None
    cte_molr: float = 287.0
    cte_grav: float = GRAVITY
    val_exp: float = VAL_EXP

    def normalized(self) -> VerticalCoordinates:
        enabled = [self.zheight, self.zsigma, self.zhybrid, self.zlevels]
        if sum(enabled) != 1:
            raise ForcingReaderError("exactly one vertical-coordinate mode must be selected")
        mode = ("height", "sigma", "hybrid", "levels")[enabled.index(True)]
        return VerticalCoordinates(
            mode, self.zsamelev_uv, self.zlev_fixed, self.zlevuv_fixed,
            self.zhybrid_a, self.zhybrid_b, self.zhybriduv_a, self.zhybriduv_b,
        )


@dataclass(frozen=True)
class ForcingDispatchContext:
    itauin: int
    istp: int
    itau_split: int
    lrstread: bool
    fcontfrac_seed: np.ndarray | None = None


@dataclass(frozen=True)
class ForcingGridState:
    """Complete grid writeback owned by ``forcing_read`` and its callees.

    Fortran provenance: ``readdim2.f90::forcing_read`` lines 499-655.  The
    arrays retain the caller's full ``(i,j[,component])`` layout; ``kindex``
    contains the defined one-based entries through ``nbindex``.
    """

    fcontfrac: np.ndarray
    fneighbours: np.ndarray
    fresolution: np.ndarray
    kindex: np.ndarray
    nbindex: int
    initialized: bool


@dataclass(frozen=True)
class ForcingOwnerTransition:
    fields: "ForcingFields"
    grid: ForcingGridState


@dataclass(frozen=True)
class LandIndexResult:
    nbindex: int
    kindex: np.ndarray
    i_test: int
    j_test: int
    ij_zero_based: np.ndarray

    def __iter__(self):
        """Retain the earlier ``kindex, i_test, j_test`` unpacking API."""

        yield self.kindex
        yield self.i_test
        yield self.j_test


@dataclass(frozen=True)
class ForcingFields:
    itb: int
    ite: int
    zlev: np.ndarray
    zlev_uv: np.ndarray
    swdown: np.ndarray
    rainf: np.ndarray
    snowf: np.ndarray | None
    tair: np.ndarray
    u: np.ndarray
    v: np.ndarray
    qair: np.ndarray
    pb: np.ndarray
    lwdown: np.ndarray
    SWnet: np.ndarray | None = None
    Eair: np.ndarray | None = None
    petAcoef: np.ndarray | None = None
    peqAcoef: np.ndarray | None = None
    petBcoef: np.ndarray | None = None
    peqBcoef: np.ndarray | None = None
    cdrag: np.ndarray | None = None
    ccanopy: np.ndarray | None = None

    @property
    def watchout(self) -> Mapping[str, np.ndarray]:
        names = ("SWnet", "Eair", "petAcoef", "peqAcoef", "petBcoef", "peqBcoef", "cdrag", "ccanopy")
        return {name: getattr(self, name) for name in names if getattr(self, name) is not None}


@dataclass(frozen=True)
class ForcingReadResult:
    itauin: int
    itau_read: int
    fields: ForcingFields
    contfrac: np.ndarray
    areas: np.ndarray
    land: LandIndexResult
    land_fields: Mapping[str, np.ndarray]
    contfrac_land: np.ndarray
    areas_land: np.ndarray
    grid: ForcingGridState


def _open(source):
    if isinstance(source, xr.Dataset):
        return source, False
    return xr.open_dataset(Path(source), decode_cf=False, mask_and_scale=False), True


def _dims(ds: xr.Dataset) -> tuple[str, str, str]:
    coordinate = ds.get("nav_lon", ds.get("nav_lat"))
    if coordinate is None or coordinate.ndim != 2:
        raise ForcingReaderError("a two-dimensional nav_lon or nav_lat is required")
    y_dim, x_dim = coordinate.dims
    candidates: set[str] = set()
    for name in ("Tair", "Tmin", "SWdown", "PSurf"):
        if name in ds:
            candidates.update(dim for dim in ds[name].dims if dim not in (y_dim, x_dim))
    if len(candidates) != 1:
        raise ForcingReaderError(f"exactly one forcing time dimension is required, found {sorted(candidates)}")
    return y_dim, x_dim, candidates.pop()


def _scalar(ds: xr.Dataset, name: str) -> float:
    if name not in ds:
        raise ForcingReaderError(f"required scalar {name!r} is missing")
    value = np.asarray(ds[name].values)
    if value.size != 1:
        raise ForcingReaderError(f"{name} must contain one value")
    return float(value.reshape(-1)[0])


def _read_field(ds, name, *, y_dim, x_dim, time_dim, t0, ii, jj):
    if name not in ds:
        raise ForcingReaderError(f"required forcing variable {name!r} is missing")
    variable = ds[name]
    if time_dim in variable.dims:
        variable = variable.isel({time_dim: t0})
    if set(variable.dims) != {y_dim, x_dim}:
        raise ForcingReaderError(f"{name} must reduce exactly to dimensions {(y_dim, x_dim)}")
    result = np.asarray(variable.transpose(x_dim, y_dim).values, dtype=np.float64)
    if ii is None and jj is None:
        return result
    if ii is None or jj is None:
        raise ValueError("both i_indices and j_indices are required")
    i_arr, j_arr = np.asarray(ii, dtype=np.int64), np.asarray(jj, dtype=np.int64)
    if i_arr.ndim != 1 or j_arr.ndim != 1:
        raise ValueError("zoom indices must be one-dimensional")
    if np.any(i_arr < 0) or np.any(i_arr >= result.shape[0]) or np.any(j_arr < 0) or np.any(j_arr >= result.shape[1]):
        raise IndexError("zoom index outside forcing grid")
    return result[np.ix_(i_arr, j_arr)]


def _vertical_from_file(ds: xr.Dataset) -> VerticalCoordinates | None:
    # Source precedence: Sigma, hybrid, Levels, fixed Height, legacy lev.
    if "Sigma" in ds:
        same = "Sigma_uv" not in ds
        return VerticalCoordinates("sigma", same, a=0.0, b=_scalar(ds, "Sigma"), a_uv=None if same else 0.0, b_uv=None if same else _scalar(ds, "Sigma_uv"))
    if "HybSigA" in ds:
        if "HybSigB" not in ds:
            raise ForcingReaderError("HybSigA exists but HybSigB is missing")
        if ("HybSigA_uv" in ds) != ("HybSigB_uv" in ds):
            raise ForcingReaderError("hybrid UV coordinates require both A and B")
        same = "HybSigA_uv" not in ds
        return VerticalCoordinates("hybrid", same, a=_scalar(ds, "HybSigA"), b=_scalar(ds, "HybSigB"), a_uv=None if same else _scalar(ds, "HybSigA_uv"), b_uv=None if same else _scalar(ds, "HybSigB_uv"))
    if "Levels" in ds:
        return VerticalCoordinates("levels", "Levels_uv" not in ds)
    if "Height_Lev1" in ds:
        same = "Height_Levuv" not in ds
        return VerticalCoordinates("height", same, lev1=_scalar(ds, "Height_Lev1"), levuv=None if same else _scalar(ds, "Height_Levuv"))
    if "lev" in ds:
        return VerticalCoordinates("height", True, lev1=_scalar(ds, "lev"))
    return None


def _validate_vertical(spec: VerticalCoordinates) -> None:
    if spec.mode not in {"height", "sigma", "hybrid", "levels"}:
        raise ForcingReaderError(f"no case for vertical mode {spec.mode!r}")
    if spec.mode == "height" and spec.lev1 is None:
        raise ForcingReaderError("height mode requires lev1")
    if spec.mode in {"sigma", "hybrid"} and (spec.a is None or spec.b is None):
        raise ForcingReaderError(f"{spec.mode} mode requires a and b")
    if not spec.same_uv:
        if spec.mode == "height" and spec.levuv is None:
            raise ForcingReaderError("distinct wind height requires levuv")
        if spec.mode in {"sigma", "hybrid"} and (spec.a_uv is None or spec.b_uv is None):
            raise ForcingReaderError("distinct wind pressure level requires a_uv and b_uv")


def _pressure_height(tair, pb, a, b, val_exp):
    valid = tair < val_exp
    density = np.divide(pb, MOLAR_GAS_CONSTANT * tair, out=np.zeros_like(pb), where=valid)
    return np.divide(pb - (a + b * pb), density * GRAVITY, out=np.zeros_like(pb), where=valid & (density != 0.0))


def forcing_landind(tair, *, check: bool = False) -> LandIndexResult:
    """``forcing_landind`` lines 1899-1968; 12 audited arms."""

    value = np.asarray(tair, dtype=np.float64)
    if value.ndim != 2 or value.size == 0:
        raise ValueError("tair must be a non-empty rank-2 Fortran array")
    iim, jjm = value.shape
    cutoff = 100.0 if np.min(value) < 100.0 else 500.0
    indices, pairs = [], []
    i_test = j_test = 0
    for j in range(jjm):
        for i in range(iim):
            if value[i, j] < cutoff:
                indices.append(j * iim + i + 1)
                pairs.append((i, j))
                if len(indices) > (iim * jjm) // 2 and i_test < 1:
                    i_test, j_test = i + 1, j + 1
                    _ = check  # source logging branch has no numerical state
    return LandIndexResult(len(indices), np.asarray(indices, dtype=np.int32), i_test, j_test, np.asarray(pairs, dtype=np.int64).reshape(-1, 2))


def forcing_grid(*, mode=None, weathergen=None, interpol=None, iim, jjm, init_f=True, limit_west=None, limit_east=None, limit_north=None, limit_south=None, merid_res=None, zonal_res=None, lon_full=None, lat_full=None, i_indices=None, j_indices=None, i_index=None, j_index=None, i_indices_one_based=None, j_indices_one_based=None, global_lon=None, global_lat=None, lon_global=None, lat_global=None, j_begin=None, j_end=None, local_j_begin=None, local_j_end=None):
    """``forcing_grid`` lines 1972-2030; interpolation/weather arms."""

    iim, jjm = int(iim), int(jjm)
    if mode is None:
        if weathergen:
            mode = "weathergen"
        elif interpol:
            mode = "interpol"
        else:
            raise ForcingReaderError("neither interpolation nor weather generator is specified")
    i_indices = i_index if i_index is not None else i_indices
    j_indices = j_index if j_index is not None else j_indices
    global_lon = lon_global if lon_global is not None else global_lon
    global_lat = lat_global if lat_global is not None else global_lat
    j_begin = local_j_begin if local_j_begin is not None else j_begin
    j_end = local_j_end if local_j_end is not None else j_end
    if mode == "interpol":
        if i_indices_one_based is not None:
            i_indices = np.asarray(i_indices_one_based) - 1
        if j_indices_one_based is not None:
            j_indices = np.asarray(j_indices_one_based) - 1
        if lon_full is None or lat_full is None or i_indices is None or j_indices is None:
            raise ForcingReaderError("interpol grid requires coordinates and zoom indices")
        ii, jj = np.asarray(i_indices, dtype=np.int64), np.asarray(j_indices, dtype=np.int64)
        lon, lat = np.asarray(lon_full)[np.ix_(ii, jj)], np.asarray(lat_full)[np.ix_(ii, jj)]
    elif mode == "weathergen":
        if init_f:
            values = (limit_west, limit_east, limit_north, limit_south, merid_res, zonal_res)
            if any(item is None for item in values):
                raise ForcingReaderError("weathergen grid requires all limits and resolutions")
            i, j = np.arange(iim)[:, None], np.arange(jjm)[None, :]
            lon = np.broadcast_to(limit_west + merid_res / 2.0 + i * (limit_east - limit_west) / iim, (iim, jjm)).copy()
            lat = np.broadcast_to(limit_north - zonal_res / 2.0 - j * (limit_north - limit_south) / jjm, (iim, jjm)).copy()
        else:
            if global_lon is None or global_lat is None or j_begin is None or j_end is None:
                raise ForcingReaderError("non-initial weather grid requires global arrays and one-based j bounds")
            lon = np.asarray(global_lon)[:, int(j_begin) - 1:int(j_end)]
            lat = np.asarray(global_lat)[:, int(j_begin) - 1:int(j_end)]
    else:
        raise ForcingReaderError("neither interpolation nor weather generator is specified")
    if lon.shape != (iim, jjm) or lat.shape != (iim, jjm):
        raise ValueError(f"forcing grid must have shape {(iim, jjm)}")
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def _forcing_just_read_dataset(source, *, itb, ite, daily_interpol, wind_n_exists, is_watchout, vertical, i_indices, j_indices, val_exp, check):
    if int(itb) != int(ite):
        raise ForcingReaderError("ite not equal itb")
    ds, close = _open(source)
    try:
        y_dim, x_dim, time_dim = _dims(ds)
        ttm = int(ds.sizes[time_dim])
        if not 1 <= int(itb) <= ttm:
            raise IndexError(f"Fortran forcing index {itb} outside 1..{ttm}")
        def read(name):
            return _read_field(ds, name, y_dim=y_dim, x_dim=x_dim, time_dim=time_dim, t0=int(itb)-1, ii=i_indices, jj=j_indices)
        if daily_interpol:
            tair, rainf, snowf = read("Tmin"), read("precip"), None
        else:
            tair, rainf, snowf = read("Tair"), read("Rainf"), read("Snowf")
        swdown, lwdown, pb, qair = (read(name) for name in ("SWdown", "LWdown", "PSurf", "Qair"))
        if wind_n_exists:
            if "Wind_E" not in ds:
                raise ForcingReaderError("Wind_N exists but Wind_E is missing")
            u, v = read("Wind_N"), read("Wind_E")
        else:
            u, v = read("Wind"), np.zeros_like(tair)
        spec = vertical.normalized() if isinstance(vertical, VerticalCoordinate) else vertical
        spec = _vertical_from_file(ds) if spec is None else spec
        if spec is None:
            raise ForcingReaderError("vertical coordinates are absent; explicit run-definition values are required")
        _validate_vertical(spec)
        if spec.mode == "height":
            zlev = np.full_like(tair, spec.lev1)
        elif spec.mode in {"sigma", "hybrid"}:
            zlev = _pressure_height(tair, pb, spec.a, spec.b, val_exp)
        else:
            zlev = read("Levels")
        if spec.same_uv:
            zlev_uv = zlev.copy()
        elif spec.mode == "height":
            zlev_uv = np.full_like(tair, spec.levuv)
        elif spec.mode in {"sigma", "hybrid"}:
            zlev_uv = _pressure_height(tair, pb, spec.a_uv, spec.b_uv, val_exp)
        else:
            zlev_uv = read("Levels_uv")
        watch = {name: None for name in ("SWnet", "Eair", "petAcoef", "peqAcoef", "petBcoef", "peqBcoef", "cdrag", "ccanopy")}
        if is_watchout:
            zlev = read("levels")
            zlev_uv = zlev.copy()
            watch = {name: read(name) for name in watch}
        _ = check
        return ForcingFields(int(itb), int(ite), zlev, zlev_uv, swdown, rainf, snowf, tair, u, v, qair, pb, lwdown, **watch)
    finally:
        if close:
            ds.close()


def _forcing_just_read_mapping(source, vertical, *, ttm, itb, ite, daily_interpol, wind_n_exists, is_watchout, i_index, j_index):
    if int(itb) != int(ite):
        raise ValueError("forcing_just_read requires itb == ite")
    if not 1 <= int(itb) <= int(ttm):
        raise IndexError(f"Fortran forcing index {itb} outside 1..{ttm}")
    spec = vertical.normalized() if hasattr(vertical, "normalized") else vertical
    if not isinstance(spec, VerticalCoordinates):
        raise TypeError("vertical must define explicit vertical-coordinate options")
    _validate_vertical(spec)
    ii = np.arange(np.asarray(next(iter(source.values()))).shape[0]) if i_index is None else np.asarray(i_index)
    jj = np.arange(np.asarray(next(iter(source.values()))).shape[1]) if j_index is None else np.asarray(j_index)
    def read(name):
        if name not in source:
            raise KeyError(f"missing forcing variable {name}")
        value = np.asarray(source[name], dtype=np.float64)
        if value.ndim != 3 or value.shape[2] != int(ttm):
            raise ValueError(f"{name} must have Fortran shape (iim,jjm,ttm)")
        return value[np.ix_(ii, jj, [int(itb)-1])][:, :, 0]
    if daily_interpol:
        tair, rainf, snowf = read("Tmin"), read("precip"), None
    else:
        tair, rainf, snowf = read("Tair"), read("Rainf"), read("Snowf")
    swdown, lwdown, pb, qair = (read(name) for name in ("SWdown", "LWdown", "PSurf", "Qair"))
    u, v = (read("Wind_N"), read("Wind_E")) if wind_n_exists else (read("Wind"), np.zeros_like(tair))
    constants = vertical if isinstance(vertical, VerticalForcingOptions) else None
    molr = constants.cte_molr if constants is not None else MOLAR_GAS_CONSTANT
    grav = constants.cte_grav if constants is not None else GRAVITY
    missing = constants.val_exp if constants is not None else VAL_EXP
    def pressure_height(a, b):
        valid = tair < missing
        density = np.divide(pb, molr * tair, out=np.zeros_like(pb), where=valid)
        return np.divide(pb-(a+b*pb), density*grav, out=np.zeros_like(pb), where=valid & (density != 0))
    if spec.mode == "height":
        zlev = np.full_like(tair, spec.lev1)
    elif spec.mode in {"sigma", "hybrid"}:
        zlev = pressure_height(spec.a, spec.b)
    else:
        zlev = read("Levels")
    if spec.same_uv:
        zlev_uv = zlev.copy()
    elif spec.mode == "height":
        zlev_uv = np.full_like(tair, spec.levuv)
    elif spec.mode in {"sigma", "hybrid"}:
        zlev_uv = pressure_height(spec.a_uv, spec.b_uv)
    else:
        zlev_uv = read("Levels_uv")
    watch = {name: None for name in ("SWnet", "Eair", "petAcoef", "peqAcoef", "petBcoef", "peqBcoef", "cdrag", "ccanopy")}
    if is_watchout:
        zlev = read("levels")
        zlev_uv = zlev.copy()
        watch = {name: read(name) for name in watch}
    return ForcingFields(int(itb), int(ite), zlev, zlev_uv, swdown, rainf, snowf, tair, u, v, qair, pb, lwdown, **watch)


def forcing_just_read(source, vertical=None, *, ttm=None, itb=1, ite=1, daily_interpol=False, wind_n_exists=True, is_watchout=False, i_indices=None, j_indices=None, i_index=None, j_index=None, val_exp=VAL_EXP, check=False, **_compat):
    """``forcing_just_read`` lines 1691-1869; 23 audited arms."""

    if isinstance(source, Mapping) and not isinstance(source, xr.Dataset):
        if ttm is None:
            raise ValueError("array forcing mapping requires explicit ttm")
        return _forcing_just_read_mapping(source, vertical, ttm=ttm, itb=itb, ite=ite, daily_interpol=daily_interpol, wind_n_exists=wind_n_exists, is_watchout=is_watchout, i_index=i_index if i_index is not None else i_indices, j_index=j_index if j_index is not None else j_indices)
    return _forcing_just_read_dataset(source, itb=itb, ite=ite, daily_interpol=daily_interpol, wind_n_exists=wind_n_exists, is_watchout=is_watchout, vertical=vertical, i_indices=i_indices, j_indices=j_indices, val_exp=val_exp, check=check)


def _fortran_mod(value, divisor):
    return int(value) - int(np.trunc(int(value) / int(divisor))) * int(divisor)


def cyclic_forcing_record(itauin: int, ttm: int) -> int:
    """Line 780: ``MOD((itauin-1),ttm)+1`` with Fortran semantics."""

    if int(ttm) < 1:
        raise ValueError("ttm must be positive")
    return _fortran_mod(int(itauin)-1, int(ttm)) + 1


def route_forcing_landpoints(field, kindex):
    """Extract a Fortran ``(i,j)`` field using one-based linear indices."""

    value = np.asarray(field)
    index = np.asarray(kindex, dtype=np.int64)
    if value.ndim != 2 or index.ndim != 1:
        raise ValueError("field must be rank 2 and kindex rank 1")
    if np.any(index < 1) or np.any(index > value.size):
        raise IndexError("kindex outside field")
    return value.ravel(order="F")[index-1]


def _land_vectors(packet, land):
    if land.nbindex == 0:
        return {}
    ii, jj = land.ij_zero_based[:, 0], land.ij_zero_based[:, 1]
    return {item.name: np.asarray(value[ii, jj]) for item in dataclass_fields(packet) if isinstance((value := getattr(packet, item.name)), np.ndarray)}


def _writeback(target, packet):
    for item in dataclass_fields(packet):
        value = getattr(packet, item.name)
        if not isinstance(value, np.ndarray):
            continue
        if isinstance(target, MutableMapping):
            if item.name in target:
                np.asarray(target[item.name])[...] = value
            else:
                target[item.name] = value.copy()
        elif hasattr(target, item.name):
            np.asarray(getattr(target, item.name))[...] = value
        else:
            raise ForcingReaderError(f"writeback target has no {item.name!r} array")


def _grid_state_from_dataset(
    ds: xr.Dataset,
    *,
    y_dim: str,
    x_dim: str,
    time_dim: str,
    i_indices,
    j_indices,
    contfrac: np.ndarray,
    land: LandIndexResult,
    watchout: bool,
) -> ForcingGridState:
    """Materialize the full ``forcing_read`` grid writeback contract."""

    iim, jjm = contfrac.shape
    neighbours = np.full((iim, jjm, 8), -1, dtype=np.int32)
    resolution = np.zeros((iim, jjm, 2), dtype=np.float64)
    neighbour_names = (
        "neighboursNN", "neighboursNE", "neighboursEE", "neighboursSE",
        "neighboursSS", "neighboursSW", "neighboursWW", "neighboursNW",
    )
    if watchout:
        missing = [name for name in (*neighbour_names, "resolutionX", "resolutionY") if name not in ds]
        if missing:
            raise ForcingReaderError(
                f"Watchout grid writeback is missing required variables: {missing}"
            )
        for component, name in enumerate(neighbour_names):
            values = _read_field(
                ds, name, y_dim=y_dim, x_dim=x_dim, time_dim=time_dim,
                t0=0, ii=i_indices, jj=j_indices,
            )
            valid = values < 999999999
            neighbours[..., component][valid] = np.rint(values[valid]).astype(np.int32)
        resolution[..., 0] = _read_field(
            ds, "resolutionX", y_dim=y_dim, x_dim=x_dim, time_dim=time_dim,
            t0=0, ii=i_indices, jj=j_indices,
        )
        resolution[..., 1] = _read_field(
            ds, "resolutionY", y_dim=y_dim, x_dim=x_dim, time_dim=time_dim,
            t0=0, ii=i_indices, jj=j_indices,
        )
    elif land.nbindex:
        lon_name = "nav_lon" if "nav_lon" in ds else "lon"
        lat_name = "nav_lat" if "nav_lat" in ds else "lat"
        lon = _read_field(
            ds, lon_name, y_dim=y_dim, x_dim=x_dim, time_dim=time_dim,
            t0=0, ii=i_indices, jj=j_indices,
        )
        lat = _read_field(
            ds, lat_name, y_dim=y_dim, x_dim=x_dim, time_dim=time_dim,
            t0=0, ii=i_indices, jj=j_indices,
        )
        from jax_orchidee.driver.static import regular_lonlat_grid_geometry

        land_resolution, land_neighbours, _, _ = regular_lonlat_grid_geometry(
            lon, lat, land.kindex
        )
        ii, jj = land.ij_zero_based[:, 0], land.ij_zero_based[:, 1]
        neighbours[ii, jj, :] = land_neighbours
        resolution[ii, jj, :] = land_resolution
    return ForcingGridState(
        np.asarray(contfrac, dtype=np.float64).copy(), neighbours, resolution,
        land.kindex.copy(), int(land.nbindex), True,
    )


def _forcing_read_netcdf(source, *, itauin, itau_split, interpol, weathergen, daily_interpol, vertical, i_indices=None, j_indices=None, land_indices=None, lrstread, is_watchout=None, wind_n_exists=None, weather_reader: Callable | None=None, val_exp=VAL_EXP, writeback=None):
    """``forcing_read`` lines 499-655; cyclic routing and writeback (6 arms)."""

    ds, close = _open(source)
    try:
        y_dim, x_dim, time_dim = _dims(ds)
        ttm = int(ds.sizes[time_dim])
        itau_read = _fortran_mod(int(itauin)-1, ttm) + 1
        if not 0 <= itau_read <= ttm:
            raise IndexError(f"source cyclic index {itau_read} outside 0..{ttm}")
        watchout = "levels" in ds if is_watchout is None else bool(is_watchout)
        wind_n = "Wind_N" in ds if wind_n_exists is None else bool(wind_n_exists)
        if interpol:
            packet = forcing_just_read(ds, vertical, itb=1 if itau_read == 0 else itau_read, ite=1 if itau_read == 0 else itau_read, daily_interpol=daily_interpol, wind_n_exists=wind_n, is_watchout=watchout, i_indices=i_indices, j_indices=j_indices)
        elif weathergen:
            if weather_reader is None:
                raise ForcingReaderError("weathergen requires a source-backed weather_reader")
            packet = weather_reader(itauin=int(itauin), itau_split=int(itau_split))
            if not isinstance(packet, ForcingFields):
                raise TypeError("weather_reader must return ForcingFields")
        else:
            raise ForcingReaderError("neither interpolation nor weather generator is specified")
        def static(name):
            return _read_field(ds, name, y_dim=y_dim, x_dim=x_dim, time_dim=time_dim, t0=0, ii=i_indices, jj=j_indices)
        if "contfrac" in ds:
            contfrac = static("contfrac")
        elif weathergen and lrstread:
            contfrac = np.ones_like(packet.tair)
        else:
            raise ForcingReaderError("contfrac is missing and no source branch defines it")
        if "Areas" in ds:
            areas = static("Areas")
        elif watchout and "resolutionX" in ds and "resolutionY" in ds:
            areas = static("resolutionX") * static("resolutionY")
        else:
            raise ForcingReaderError("Areas is required unless Watchout resolutionX/resolutionY define area")
        masked = packet.tair.copy()
        masked[contfrac <= np.finfo(np.float32).eps] = 999999.0
        land = forcing_landind(masked) if land_indices is None else land_indices
        if not watchout:
            valid = packet.tair < val_exp
            if not np.all(valid) and packet.Eair is None:
                raise ForcingReaderError("prior Eair state is required at invalid tair cells")
            eair = np.empty_like(packet.tair) if packet.Eair is None else packet.Eair.copy()
            eair[valid] = CP_AIR * packet.tair[valid] + GRAVITY * packet.zlev[valid]
            packet = ForcingFields(**{**packet.__dict__, "Eair": eair})
        grid = _grid_state_from_dataset(
            ds, y_dim=y_dim, x_dim=x_dim, time_dim=time_dim,
            i_indices=i_indices, j_indices=j_indices, contfrac=contfrac,
            land=land, watchout=watchout,
        )
        if writeback is not None:
            _writeback(writeback, packet)
        ii, jj = land.ij_zero_based[:, 0], land.ij_zero_based[:, 1]
        return ForcingReadResult(
            int(itauin), itau_read, packet, contfrac, areas, land,
            _land_vectors(packet, land), contfrac[ii, jj], areas[ii, jj], grid,
        )
    finally:
        if close:
            ds.close()


def forcing_read(source=None, *, context=None, interpol, weathergen, interpol_owner=None, weathergen_owner=None, is_watchout=None, cp_air=CP_AIR, cte_grav=GRAVITY, val_exp=VAL_EXP, return_complete=False, **kwargs):
    """Dispatch ``forcing_read`` for either process-owner or NetCDF boundaries."""

    if context is None:
        return _forcing_read_netcdf(
            source, interpol=interpol, weathergen=weathergen,
            is_watchout=is_watchout, val_exp=val_exp, **kwargs,
        )
    if not isinstance(context, ForcingDispatchContext):
        raise TypeError("context must be ForcingDispatchContext")
    if interpol:
        if interpol_owner is None:
            raise ForcingReaderError("forcing_read_interpol owner is required")
        owner_value = interpol_owner(context)
    elif weathergen:
        if weathergen_owner is None:
            raise ForcingReaderError("weathergen owner is required")
        call_context = context
        if context.itauin == 0 and context.itau_split == 0:
            call_context = ForcingDispatchContext(context.istp, context.istp, context.itau_split, context.lrstread, context.fcontfrac_seed)
        if context.lrstread:
            if context.fcontfrac_seed is None:
                raise ForcingReaderError("lrstread weather branch requires fcontfrac_seed shape")
            call_context = ForcingDispatchContext(call_context.itauin, call_context.istp, call_context.itau_split, call_context.lrstread, np.ones_like(context.fcontfrac_seed))
        owner_value = weathergen_owner(call_context)
    else:
        raise ForcingReaderError("neither interpolation nor weather generator is specified")
    if isinstance(owner_value, ForcingOwnerTransition):
        packet, grid = owner_value.fields, owner_value.grid
        if weathergen and context.lrstread:
            grid = replace(grid, fcontfrac=np.ones_like(grid.fcontfrac))
    else:
        packet, grid = owner_value, None
    if not isinstance(packet, ForcingFields):
        raise TypeError("forcing process owner must return ForcingFields")
    if not is_watchout:
        valid = packet.tair < val_exp
        if not np.all(valid) and packet.Eair is None:
            raise ForcingReaderError("prior Eair state is required at invalid tair cells")
        eair = np.empty_like(packet.tair) if packet.Eair is None else packet.Eair.copy()
        eair[valid] = cp_air * packet.tair[valid] + cte_grav * packet.zlev[valid]
        packet = ForcingFields(**{**packet.__dict__, "Eair": eair})
    if return_complete:
        if grid is None:
            raise ForcingReaderError(
                "complete forcing_read requires owner grid-state writeback"
            )
        return ForcingOwnerTransition(packet, grid)
    return packet


def forcing_read_energy(tair, zlev, *, is_watchout, val_exp=VAL_EXP, cp_air=CP_AIR, cte_grav=GRAVITY):
    """Compatibility helper for the final non-Watchout ``WHERE`` branch."""

    if is_watchout:
        return None
    tair, zlev = np.asarray(tair), np.asarray(zlev)
    return np.where(tair < val_exp, cp_air * tair + cte_grav * zlev, np.nan)
