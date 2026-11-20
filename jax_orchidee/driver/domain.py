"""Phase 1B driver/domain/init readers for the ORCHIDEE-MAN paper case.

The routines here intentionally implement only the active paper-case path:
regular lon/lat forcing, closed domain zoom, first forcing read, annual CO2,
and the water-table text sequences. They do not implement inactive branches
such as weather-generator forcing, alternate grids, or TOPMODEL_NEW.
"""

from __future__ import annotations

import math
import os
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml


@dataclass(frozen=True)
class DomainGrid:
    """Minimal driver grid produced by the dim2_driver/readdim2 active path."""

    iim: int
    jjm: int
    nbindex: int
    kindex: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    lalo: np.ndarray
    contfrac: np.ndarray
    contfrac_land: np.ndarray
    area: np.ndarray
    forcing_indices_zero_based: np.ndarray
    resolution: np.ndarray | None = None
    neighbours: np.ndarray | None = None
    corners: np.ndarray | None = None
    seglength: np.ndarray | None = None


@dataclass(frozen=True)
class ForcingStep:
    """One forcing slab with both file names and Fortran-facing names."""

    tstep: int
    Tair: np.ndarray
    PSurf: np.ndarray
    Qair: np.ndarray
    Wind_E: np.ndarray
    Wind_N: np.ndarray
    Rainf: np.ndarray
    Snowf: np.ndarray | None
    SWdown: np.ndarray
    LWdown: np.ndarray
    Areas: np.ndarray
    contfrac: np.ndarray
    Height_Lev1: float | None
    Height_Levuv: float | None
    temp_air: np.ndarray
    pb: np.ndarray
    qair: np.ndarray
    u: np.ndarray
    v: np.ndarray
    precip_rain: np.ndarray
    precip_snow: np.ndarray | None
    swdown: np.ndarray
    lwdown: np.ndarray
    zlev: np.ndarray
    zlevuv: np.ndarray
    is_watchout: bool = False
    swnet: np.ndarray | None = None
    eair: np.ndarray | None = None
    petAcoef: np.ndarray | None = None
    peqAcoef: np.ndarray | None = None
    petBcoef: np.ndarray | None = None
    peqBcoef: np.ndarray | None = None
    cdrag: np.ndarray | None = None
    ccanopy: np.ndarray | None = None
    dt_force: float | None = None


@dataclass(frozen=True)
class PaperPointForcingCache:
    """Selected land-point forcing series for the audited driver path."""

    Tair: np.ndarray
    PSurf: np.ndarray
    Qair: np.ndarray
    Wind_E: np.ndarray
    Wind_N: np.ndarray
    Rainf: np.ndarray
    Snowf: np.ndarray | None
    SWdown: np.ndarray
    LWdown: np.ndarray
    Areas: np.ndarray
    contfrac: np.ndarray
    Height_Lev1: float | None
    Height_Levuv: float | None
    is_watchout: bool = False
    swnet: np.ndarray | None = None
    eair: np.ndarray | None = None
    petAcoef: np.ndarray | None = None
    peqAcoef: np.ndarray | None = None
    petBcoef: np.ndarray | None = None
    peqBcoef: np.ndarray | None = None
    cdrag: np.ndarray | None = None
    ccanopy: np.ndarray | None = None
    zlev: np.ndarray | None = None
    zlevuv: np.ndarray | None = None
    daily_interpol: bool = False
    dt_force: float | None = None
    Tmax: np.ndarray | None = None


_WATCHOUT_FILE_FIELDS = {
    "swnet": "SWnet", "eair": "Eair", "petAcoef": "petAcoef",
    "peqAcoef": "peqAcoef", "petBcoef": "petBcoef", "peqBcoef": "peqBcoef",
    "cdrag": "cdrag", "ccanopy": "ccanopy",
}


@dataclass(frozen=True)
class WaterTableSequences:
    """Water-table forcing text sequences read by HYDROL."""

    positive: np.ndarray
    differential: np.ndarray


@lru_cache(maxsize=16)
def _load_case_config_cached(
    config_path: str,
    repo_root_env: str | None,
    data_root_env: str | None,
    reference_root_env: str | None,
    output_root_env: str | None,
) -> dict[str, Any]:
    resolved_config = Path(config_path).resolve()
    with resolved_config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    repo_root = resolved_config.parents[1]
    replacements = {
        "${ORCHIDEE_REPO_ROOT}": repo_root_env or str(repo_root),
        "${ORCHIDEE_DATA_ROOT}": data_root_env or str(repo_root / "data"),
        "${ORCHIDEE_REFERENCE_ROOT}": reference_root_env or str(repo_root / "reference"),
        "${ORCHIDEE_OUTPUT_ROOT}": output_root_env or str(repo_root / "outputs"),
    }

    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            contains_root_token = any(token in value for token in replacements)
            expanded = value
            for token, replacement in replacements.items():
                expanded = expanded.replace(token, replacement)
            expanded = os.path.expandvars(os.path.expanduser(expanded))
            return os.path.normpath(expanded) if contains_root_token else expanded
        return value

    return expand(config)


def load_case_config(config_path: str | Path) -> dict[str, Any]:
    """Load the local paper-case YAML config.

    Fortran provenance: values in this YAML are distilled from
    `fortran_run_scripts/paper_250919/Job0_bio` and `run.def.vn`; the driver
    consumes them through `getin_p` in `dim2_driver.f90`, lines 147-175 and
    `readdim2.f90`, lines 222-272.
    """

    return deepcopy(
        _load_case_config_cached(
            str(Path(config_path).resolve()),
            os.environ.get("ORCHIDEE_REPO_ROOT"),
            os.environ.get("ORCHIDEE_DATA_ROOT"),
            os.environ.get("ORCHIDEE_REFERENCE_ROOT"),
            os.environ.get("ORCHIDEE_OUTPUT_ROOT"),
        )
    )


def forcing_file_for_year(config: dict[str, Any], year: int) -> Path:
    """Resolve the configured CRUNCEP forcing file for a year.

    Fortran/run provenance: `fortran_run_scripts/paper_250919/Job0_bio`,
    lines 325-327, writes `FORCING_FILE` for the current year; `dim2_driver.f90`,
    lines 164-175, reads `FORCING_FILE` before calling `forcing_info`.
    """

    pattern = config["drivers"]["atmospheric_forcing"]["file_pattern"]
    return Path(pattern.format(year=year))


def _as_fortran_lon_lat(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return nav_lon/nav_lat in Fortran `(longitude, latitude)` order.

    Fortran provenance: `readdim2.f90`, subroutine `forcing_info`, lines
    100-118, allocates `lon_full(iim_full,jjm_full)` and `lat_full`, and
    `flinopen` fills those arrays in Fortran `(i,j)` order.
    """

    return np.asarray(ds["nav_lon"].values).T, np.asarray(ds["nav_lat"].values).T


def _as_fortran_2d(ds: xr.Dataset, name: str) -> np.ndarray:
    """Return a 2D forcing variable in Fortran `(longitude, latitude)` order.

    Fortran provenance: `readdim2.f90`, subroutine `forcing_zoom`, lines
    2034-2051, indexes 2D fields as `x_f(i_index(i),j_index(j))`.
    """

    return np.asarray(ds[name].values).T


def _as_fortran_timestep(ds: xr.Dataset, name: str, tstep: int) -> np.ndarray:
    """Return a timestep variable in Fortran `(longitude, latitude)` order.

    Fortran provenance: `readdim2.f90`, subroutine `forcing_just_read`, lines
    1751-1773, reads one forcing timestep into `data_full(iim_full,jjm_full)`
    before `forcing_zoom`.
    """

    return np.asarray(ds[name].isel(tstep=tstep).values).T


def _watchout_forcing_cadence(ds: xr.Dataset, config: dict[str, Any], time_dim: str) -> float:
    """Resolve the file interval used by ``forcing_info`` lines 170-195.

    The explicit case cadence remains a fallback for legacy files whose
    numeric record coordinate has no units. WATCHOUT recognition itself is
    exclusively the lowercase ``levels`` query at source lines 805-807.
    """

    from .forcing_readers import ForcingReaderError

    factors = {"second": 1.0, "seconds": 1.0, "s": 1.0,
               "minute": 60.0, "minutes": 60.0,
               "hour": 3600.0, "hours": 3600.0,
               "day": 86400.0, "days": 86400.0}
    for name in ("time_counter", "time", time_dim):
        variable = ds.get(name)
        if variable is None or variable.dims != (time_dim,) or variable.size < 2:
            continue
        unit = str(variable.attrs.get("units", "")).strip().lower().split(" since ")[0]
        if unit in factors:
            times = np.asarray(variable.isel({time_dim: slice(0, 2)}).values, dtype=np.float64)
            cadence = float(abs(times[1] - times[0])) * factors[unit]
            if not np.isfinite(cadence) or cadence <= 0.0:
                raise ForcingReaderError("forcing records must have a positive finite time interval")
            return cadence
    hours = config.get("time", {}).get("forcing_step_hours")
    if hours is None or not np.isfinite(float(hours)) or float(hours) <= 0.0:
        raise ForcingReaderError("WATCHOUT forcing requires time-coordinate units or explicit forcing_step_hours")
    return float(hours) * 3600.0


def _read_watchout_land_cache(
    ds: xr.Dataset,
    config: dict[str, Any],
    land_i: np.ndarray,
    land_j: np.ndarray,
    *,
    tstep: int | None = None,
) -> PaperPointForcingCache:
    """Read WATCHOUT into the selected-land cache without changing units.

    Provenance: ``readdim2.f90::forcing_just_read`` lines 1745-1863 reads
    meteorology, then lowercase levels and the eight payloads. Lines
    1833-1839 override both heights. The source reader is also called once
    to retain vertical-coordinate and required-field validation.
    """

    from .forcing_readers import ForcingReaderError, _dims, _read_field, forcing_just_read

    y_dim, x_dim, time_dim = _dims(ds)
    nt = int(ds.sizes[time_dim])
    if tstep is not None and not 0 <= tstep < nt:
        raise IndexError(f"tstep {tstep} outside forcing range 0..{nt - 1}")
    dt_force = _watchout_forcing_cadence(ds, config, time_dim)
    daily = dt_force == 86400.0
    forcing_just_read(
        ds, itb=1 if tstep is None else tstep + 1, ite=1 if tstep is None else tstep + 1,
        daily_interpol=daily, wind_n_exists="Wind_N" in ds, is_watchout=True,
    )

    def series_at(name: str) -> np.ndarray:
        if name not in ds:
            raise ForcingReaderError(f"required forcing variable {name!r} is missing")
        variable = ds[name]
        expected = {y_dim, x_dim}
        if set(variable.dims) not in (expected, expected | {time_dim}):
            raise ForcingReaderError(f"{name} must have horizontal dimensions and at most the forcing time dimension")
        variable = variable.isel({y_dim: xr.DataArray(land_j, dims="land"), x_dim: xr.DataArray(land_i, dims="land")})
        if time_dim in variable.dims:
            if tstep is not None:
                variable = variable.isel({time_dim: slice(tstep, tstep + 1)})
            values = np.asarray(variable.transpose(time_dim, "land").values, dtype=np.float64)
        else:
            values = np.broadcast_to(np.asarray(variable.values, dtype=np.float64), (nt if tstep is None else 1, land_i.size))
        return values.reshape((-1, land_i.size, 1))

    levels = series_at("levels")
    wind_n = "Wind_N" in ds
    wind_u = series_at("Wind_N" if wind_n else "Wind")
    # Source lines 1063-1064 derive Watchout area from file resolution;
    # the standard forcing's Areas variable is not required in this branch.
    def resolution_land(name):
        grid = _read_field(ds, name, y_dim=y_dim, x_dim=x_dim, time_dim=time_dim, t0=0, ii=None, jj=None)
        return _land_column_values(grid, land_i, land_j)

    return PaperPointForcingCache(
        Tair=series_at("Tmin" if daily else "Tair"), PSurf=series_at("PSurf"), Qair=series_at("Qair"),
        Wind_E=series_at("Wind_E") if wind_n else np.zeros_like(wind_u), Wind_N=wind_u,
        Rainf=series_at("precip" if daily else "Rainf"), Snowf=None if daily else series_at("Snowf"),
        SWdown=series_at("SWdown"), LWdown=series_at("LWdown"),
        Areas=resolution_land("resolutionX") * resolution_land("resolutionY"),
        contfrac=_land_column_values(_as_fortran_2d(ds, "contfrac"), land_i, land_j),
        Height_Lev1=float(np.asarray(ds["Height_Lev1"].values)) if "Height_Lev1" in ds else None,
        Height_Levuv=float(np.asarray(ds["Height_Levuv"].values)) if "Height_Levuv" in ds else None,
        is_watchout=True, zlev=levels, zlevuv=levels, daily_interpol=daily, dt_force=dt_force,
        Tmax=series_at("Tmax") if daily and tstep is None and "Tmax" in ds else None,
        **{public: series_at(file_name) for public, file_name in _WATCHOUT_FILE_FIELDS.items()},
    )


@lru_cache(maxsize=4)
def _read_paper_point_forcing_cache(config_path: str, year: int, i0: int, j0: int) -> PaperPointForcingCache:
    """Read the active point's yearly forcing once, preserving Fortran order."""

    return _read_forcing_land_cache(config_path, year, ((int(i0), int(j0)),))


@lru_cache(maxsize=4)
def _read_forcing_land_cache(config_path: str, year: int, land_indices_key: tuple[tuple[int, int], ...]) -> PaperPointForcingCache:
    """Read selected land-point yearly forcing once, preserving land order.

    Fortran provenance: `readdim2.f90::forcing_landind`, lines 1899-1968,
    determines the selected `(i,j)` land order; `forcing_just_read`, lines
    1751-1773, reads each 2-D forcing slab before the land-vector handoff.
    """

    if not land_indices_key:
        raise ValueError("land_indices_key must contain at least one land point")
    land_indices = np.asarray(land_indices_key, dtype=np.int64)
    if land_indices.ndim != 2 or land_indices.shape[1] != 2:
        raise ValueError("land_indices_key must be a tuple of (i,j) pairs")
    land_i = land_indices[:, 0]
    land_j = land_indices[:, 1]
    config = load_case_config(config_path)
    forcing_path = forcing_file_for_year(config, year)
    with xr.open_dataset(forcing_path, decode_times=False) as ds:
        if "levels" in ds:
            return _read_watchout_land_cache(ds, config, land_i, land_j)
        def series_at(name: str) -> np.ndarray:
            values = np.asarray(ds[name].values, dtype=np.float64)
            return values[:, land_j, land_i].reshape((values.shape[0], land_i.size, 1))

        Areas = _land_column_values(_as_fortran_2d(ds, "Areas"), land_i, land_j)
        contfrac = _land_column_values(_as_fortran_2d(ds, "contfrac"), land_i, land_j)
        height_lev1 = float(np.asarray(ds["Height_Lev1"].values))
        height_levuv = float(np.asarray(ds["Height_Levuv"].values))
        return PaperPointForcingCache(
            Tair=series_at("Tair"),
            PSurf=series_at("PSurf"),
            Qair=series_at("Qair"),
            Wind_E=series_at("Wind_E"),
            Wind_N=series_at("Wind_N"),
            Rainf=series_at("Rainf"),
            Snowf=series_at("Snowf"),
            SWdown=series_at("SWdown"),
            LWdown=series_at("LWdown"),
            Areas=Areas,
            contfrac=contfrac,
            Height_Lev1=height_lev1,
            Height_Levuv=height_levuv,
        )


def _forcing_step_from_cached_point(cache: PaperPointForcingCache, *, tstep: int) -> ForcingStep:
    nt = int(cache.Tair.shape[0])
    if tstep < 0 or tstep >= nt:
        raise IndexError(f"tstep {tstep} outside forcing range 0..{nt - 1}")
    return _forcing_step_with_values(
        cache, tstep=tstep,
        **{name: None if (value := getattr(cache, name)) is None else value[tstep]
           for name in ("Tair", "PSurf", "Qair", "Wind_E", "Wind_N", "Rainf", "Snowf", "SWdown", "LWdown")},
        zlev=None if cache.zlev is None else cache.zlev[tstep],
        zlevuv=None if cache.zlevuv is None else cache.zlevuv[tstep],
        watchout_values={name: getattr(cache, name)[tstep] for name in _WATCHOUT_FILE_FIELDS} if cache.is_watchout else None,
    )


def _forcing_step_with_values(
    cache: PaperPointForcingCache,
    *,
    tstep: int,
    Tair: np.ndarray,
    PSurf: np.ndarray,
    Qair: np.ndarray,
    Wind_E: np.ndarray,
    Wind_N: np.ndarray,
    Rainf: np.ndarray,
    Snowf: np.ndarray | None,
    SWdown: np.ndarray,
    LWdown: np.ndarray,
    zlev: np.ndarray | None = None,
    zlevuv: np.ndarray | None = None,
    watchout_values: dict[str, np.ndarray] | None = None,
) -> ForcingStep:
    if zlev is None:
        if cache.Height_Lev1 is None:
            raise ValueError("forcing requires explicit zlev or Height_Lev1")
        zlev = np.full_like(Tair, cache.Height_Lev1)
    # Active paper files are non-Watchout CRUNCEP forcing. In
    # readdim2.f90::forcing_read_interpol, lines 1554-1561 set zlev/zlevuv
    # only inside the is_watchout branch; for non-Watchout files zlevuv keeps
    # the forcing_just_read Height_Levuv value from lines 1808-1811.
    if zlevuv is None:
        if cache.Height_Levuv is None:
            raise ValueError("forcing requires explicit zlevuv or Height_Levuv")
        zlevuv = np.full_like(Tair, cache.Height_Levuv)
    return ForcingStep(
        tstep=int(tstep),
        Tair=Tair,
        PSurf=PSurf,
        Qair=Qair,
        Wind_E=Wind_E,
        Wind_N=Wind_N,
        Rainf=Rainf,
        Snowf=Snowf,
        SWdown=SWdown,
        LWdown=LWdown,
        Areas=cache.Areas,
        contfrac=cache.contfrac,
        Height_Lev1=cache.Height_Lev1,
        Height_Levuv=cache.Height_Levuv,
        temp_air=Tair,
        pb=PSurf,
        qair=Qair,
        u=Wind_N,
        v=Wind_E,
        precip_rain=Rainf,
        precip_snow=Snowf,
        swdown=SWdown,
        lwdown=LWdown,
        zlev=zlev,
        zlevuv=zlevuv,
        is_watchout=cache.is_watchout,
        dt_force=cache.dt_force,
        **(watchout_values or {}),
    )


def _paper_forcing_raw_index_for_model_read(itau_read: int, nt: int) -> int:
    """Convert Fortran 1-based cyclic forcing read index to a Python index."""

    if nt < 1:
        raise ValueError("forcing series is empty")
    return (int(itau_read) - 1) % int(nt)


def _fortran_time_zone(lon: np.ndarray, *, gmt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return `solar.f90::time_zone` zone/lhour for a lon grid.

    Fortran provenance: `fortran_source/ORCHIDEE/src_global/solar.f90`,
    subroutine `time_zone`, lines 201-268. Note that Fortran assigns
    `deg = lon(ilon,1)` to an INTEGER, so this intentionally truncates toward
    zero before the 15-degree-zone search.
    """

    lon = np.asarray(lon, dtype=np.float64)
    lon_first_lat = lon[:, :1]
    deg = lon_first_lat.astype(np.int32)
    zone = np.zeros_like(deg, dtype=np.int32)
    assigned = np.zeros_like(deg, dtype=bool)
    for index in range(1, 26):
        matches = (~assigned) & (deg < (-187.5 + (15.0 * index)))
        zone = np.where(matches, index, zone)
        assigned |= matches
    if not np.all(assigned):
        raise ValueError("solar.time_zone received longitude outside the source search range")
    zone = np.where(zone == 25, 1, zone)
    lhour = float(gmt) + (zone.astype(np.float64) - 13.0)
    lhour = np.where(lhour < 0.0, lhour + 24.0, lhour)
    lhour = np.where(lhour >= 24.0, lhour - 24.0, lhour)
    return zone[:, 0], lhour[:, 0]


def _solarang_gswp(
    *,
    julian_since_year_start: float,
    lon: np.ndarray,
    lat: np.ndarray,
    one_year: float = 365.0,
) -> np.ndarray:
    """Replicate `solar.f90::solarang` for the active forcing year.

    Fortran provenance: `fortran_source/ORCHIDEE/src_global/solar.f90`,
    subroutine `solarang`, lines 55-181. The paper forcing is read after
    `readdim2.f90::forcing_info`, lines 150-168, forces the non-weathergen
    calendar to Gregorian/noleap-compatible day constants for this no-leap
    1961 forcing, so the active first-year denominator is 365 days.
    """

    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    gamma = 2.0 * math.pi * float(julian_since_year_start) / float(one_year)
    dec = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.00148 * math.sin(3.0 * gamma)
    )
    et = (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.04089 * math.sin(2.0 * gamma)
    ) * 229.18
    gmt = 24.0 * (float(julian_since_year_start) - math.floor(float(julian_since_year_start)))
    zone, lhour = _fortran_time_zone(lon, gmt=gmt)
    ls = ((zone.astype(np.float64) - 1.0) * 15.0) - 180.0
    le = lon[:, 0]
    lcorr = 4.0 * (ls - le) * -1.0
    latime = lhour + (lcorr / 60.0) + (et / 60.0)
    latime = np.where(latime < 0.0, latime + 24.0, latime)
    latime = np.where(latime > 24.0, latime - 24.0, latime)
    omega = (latime - 12.0) * (-15.0) * math.pi / 180.0
    llat = lat[0, :] * math.pi / 180.0
    return np.maximum(
        0.0,
        (math.sin(dec) * np.sin(llat))[None, :]
        + (math.cos(dec) * np.cos(llat))[None, :] * np.cos(omega)[:, None],
    )


@lru_cache(maxsize=8192)
def _paper_model_interval_mean_coszang_cached(
    *,
    forcing_interval: int,
    split: int,
    dt_force: float,
    lon_key: tuple[tuple[float, ...], ...],
    lat_key: tuple[tuple[float, ...], ...],
) -> np.ndarray:
    interval_start_day = forcing_interval * (float(dt_force) / 86400.0)
    lon = np.asarray(lon_key, dtype=np.float64)
    lat = np.asarray(lat_key, dtype=np.float64)
    mean_coszang = np.zeros_like(lon, dtype=np.float64)
    for split_index in range(1, int(split) + 1):
        julian = interval_start_day + ((float(split_index) - 0.5) / float(split)) * float(dt_force) / 86400.0
        mean_coszang = mean_coszang + _solarang_gswp(julian_since_year_start=julian, lon=lon, lat=lat)
    return mean_coszang / float(split)


@lru_cache(maxsize=65536)
def _paper_model_step_solarang_terms_cached(
    *,
    model_tstep: int,
    split: int,
    dt_force: float,
    lon_key: tuple[tuple[float, ...], ...],
    lat_key: tuple[tuple[float, ...], ...],
) -> tuple[np.ndarray, np.ndarray]:
    forcing_interval = int(model_tstep) // int(split)
    itau_split = int(model_tstep) % int(split) + 1
    interval_start_day = forcing_interval * (float(dt_force) / 86400.0)
    lon = np.asarray(lon_key, dtype=np.float64)
    lat = np.asarray(lat_key, dtype=np.float64)
    mean_coszang = _paper_model_interval_mean_coszang_cached(
        forcing_interval=forcing_interval,
        split=int(split),
        dt_force=float(dt_force),
        lon_key=lon_key,
        lat_key=lat_key,
    )
    julian = interval_start_day + ((float(itau_split) - 0.5) / float(split)) * float(dt_force) / 86400.0
    coszang = _solarang_gswp(julian_since_year_start=julian, lon=lon, lat=lat)
    return coszang, mean_coszang


def _array_tuple_key(value: np.ndarray) -> tuple[tuple[float, ...], ...]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("cached paper solarang geometry expects a 2-D array")
    return tuple(tuple(float(item) for item in row) for row in arr)


def _paper_model_step_swdown_from_solarang(
    swdown_n: np.ndarray,
    *,
    model_tstep: int,
    split: int,
    dt_force: float,
    lon: np.ndarray | None,
    lat: np.ndarray | None,
) -> np.ndarray:
    """Redistribute forcing SWdown over split steps using solar angle.

    Fortran provenance: `readdim2.f90::forcing_read_interpol`, lines
    1526-1550 compute `mean_coszang` over the forcing interval and lines
    1620-1639 set `swdown = swdown_n * coszang / mean_coszang` for forcing
    timesteps longer than one hour. This helper implements that active
    non-Watchout paper-case path.
    """

    if lon is None or lat is None or float(dt_force) <= 3600.0:
        return np.asarray(swdown_n, dtype=np.float64)
    coszang, mean_coszang = _paper_model_step_solarang_terms_cached(
        model_tstep=int(model_tstep),
        split=int(split),
        dt_force=float(dt_force),
        lon_key=_array_tuple_key(lon),
        lat_key=_array_tuple_key(lat),
    )
    swdown = np.divide(
        np.asarray(swdown_n, dtype=np.float64) * coszang,
        mean_coszang,
        out=np.zeros_like(mean_coszang, dtype=np.float64),
        where=mean_coszang > 0.0,
    )
    return np.where(swdown > 2000.0, 2000.0, swdown)


def _paper_model_step_swdown_land_from_solarang(
    swdown_n: np.ndarray,
    *,
    model_tstep: int,
    split: int,
    dt_force: float,
    lon: np.ndarray | None,
    lat: np.ndarray | None,
    land_indices_key: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Redistribute SWdown on the full zoom grid, then select land points.

    Fortran provenance: `readdim2.f90::forcing_read_interpol`, lines
    1526-1550 compute `coszang`/`mean_coszang` on the forcing grid and lines
    1620-1639 redistribute `swdown_n`; `forcing_landind`, lines 1899-1968,
    defines the later selected land-vector order.
    """

    if lon is None or lat is None or float(dt_force) <= 3600.0:
        return np.asarray(swdown_n, dtype=np.float64)
    land_indices = np.asarray(land_indices_key, dtype=np.int64)
    if land_indices.ndim != 2 or land_indices.shape[1] != 2:
        raise ValueError("land_indices_key must be a tuple of local (i,j) pairs")
    coszang, mean_coszang = _paper_model_step_solarang_terms_cached(
        model_tstep=int(model_tstep),
        split=int(split),
        dt_force=float(dt_force),
        lon_key=_array_tuple_key(lon),
        lat_key=_array_tuple_key(lat),
    )
    coszang_land = _land_column_values(coszang, land_indices[:, 0], land_indices[:, 1])
    mean_coszang_land = _land_column_values(mean_coszang, land_indices[:, 0], land_indices[:, 1])
    swdown = np.divide(
        np.asarray(swdown_n, dtype=np.float64) * coszang_land,
        mean_coszang_land,
        out=np.zeros_like(mean_coszang_land, dtype=np.float64),
        where=mean_coszang_land > 0.0,
    )
    return np.where(swdown > 2000.0, 2000.0, swdown)


class _WatchoutModelCursor:
    """Bounded IO cursor for source SAVE variables, never checkpoint state.

    Sequential calls retain only the final interpolation state/result. A
    backwards request replays from initialization, preserving the source's
    daily half-day and solar-history transitions rather than inventing a
    stateless daily approximation. The records are selected-land views.

    Fortran provenance: ``readdim2.f90::forcing_read_interpol`` lines
    844-848 initializes the raw record, 1079-1407 owns daily progression,
    and 1425-1661 owns ordinary progression, through the shared owner.
    """

    def __init__(self, cache, *, split, nb_spread, dt_force, lon, lat, land_indices_key, netrad_cons=False):
        from .forcing_interpolation import ForcingInterpolationRecord

        self.cache = cache
        self.split = int(split)
        self.nb_spread = min(int(nb_spread), self.split)
        self.dt_force = float(dt_force)
        self.lon = np.asarray(lon, dtype=np.float64)
        self.lat = np.asarray(lat, dtype=np.float64)
        self.land_indices = np.asarray(land_indices_key, dtype=np.int64)
        self.netrad_cons = bool(netrad_cons)
        self.records = tuple(
            ForcingInterpolationRecord.from_mapping({
                "tair": cache.Tair[index], "pb": cache.PSurf[index], "qair": cache.Qair[index],
                "u": cache.Wind_N[index], "v": cache.Wind_E[index],
                "rainf": cache.Rainf[index], "snowf": None if cache.Snowf is None else cache.Snowf[index],
                "swdown": cache.SWdown[index], "lwdown": cache.LWdown[index],
                "zlev": cache.zlev[index], "zlevuv": cache.zlevuv[index],
                "tmax": None if cache.Tmax is None else cache.Tmax[index],
                **{name.lower(): getattr(cache, name)[index] for name in _WATCHOUT_FILE_FIELDS},
            })
            for index in range(cache.Tair.shape[0])
        )
        self._reset()

    def _solar_selected(self, julian, julian0, lon, lat):
        # solarang uses the regular grid's first lon column/lat row. Compute
        # before gathering, as readdim2 does; a land column is not that grid.
        from .driver_forcing_completion import solarang

        grid = solarang(julian, julian0, lon, lat)
        return _land_column_values(grid, self.land_indices[:, 0], self.land_indices[:, 1])

    def _transition(self, state, *, itauin, itau_split):
        from .forcing_interpolation import forcing_read_interpol_transition

        return forcing_read_interpol_transition(
            self.records, state, itauin=itauin, itau_split=itau_split,
            split=self.split, nb_spread=self.nb_spread, dt_force=self.dt_force,
            lon=self.lon, lat=self.lat, daily_interpol=self.cache.daily_interpol,
            is_watchout=True, netrad_cons=self.netrad_cons,
            solarang_function=self._solar_selected,
        )

    def _reset(self):
        from .forcing_interpolation import ForcingInterpolationState

        self.state = self._transition(ForcingInterpolationState(), itauin=0, itau_split=0).state
        self.last_step = -1
        self.last_values = None

    def read(self, model_tstep):
        model_tstep = int(model_tstep)
        if model_tstep < 0:
            raise IndexError("model_tstep must be non-negative")
        if model_tstep < self.last_step:
            self._reset()
        while self.last_step < model_tstep:
            next_step = self.last_step + 1
            result = self._transition(
                self.state, itauin=next_step // self.split + 1,
                itau_split=next_step % self.split + 1,
            )
            self.state, self.last_values = result.state, result.values
            self.last_step = next_step
        values = self.last_values
        return _forcing_step_with_values(
            self.cache, tstep=model_tstep,
            Tair=values.tair, PSurf=values.pb, Qair=values.qair,
            Wind_E=values.v, Wind_N=values.u, Rainf=values.rainf, Snowf=values.snowf,
            SWdown=values.swdown, LWdown=values.lwdown, zlev=values.zlev, zlevuv=values.zlevuv,
            watchout_values={name: getattr(values, name.lower()) for name in _WATCHOUT_FILE_FIELDS},
        )


@lru_cache(maxsize=4)
def _watchout_model_cursor(
    config_path, year, land_indices_key, split, nb_spread, dt_force,
    lon_key, lat_key, local_land_indices_key, netrad_cons, cache_identity,
):
    """Cache host IO progression only; no coefficients enter model restart."""

    cache = _read_forcing_land_cache(config_path, year, land_indices_key)
    return _WatchoutModelCursor(
        cache, split=split, nb_spread=nb_spread, dt_force=dt_force,
        lon=np.asarray(lon_key), lat=np.asarray(lat_key),
        land_indices_key=local_land_indices_key, netrad_cons=netrad_cons,
    )


def _forcing_model_step_from_cached_point(
    cache: PaperPointForcingCache,
    *,
    model_tstep: int,
    split: int,
    nb_spread: int,
    lon: np.ndarray | None = None,
    lat: np.ndarray | None = None,
    land_indices_key: tuple[tuple[int, int], ...] | None = None,
    dt_force: float = 21600.0,
    netrad_cons: bool = False,
) -> ForcingStep:
    """Return Fortran's active paper-case model-step forcing.

    Fortran provenance: `fortran_source/ORCHIDEE/src_driver/dim2_driver.f90`,
    lines 293-310 define `split`, lines 817-848 call `forcing_READ` with
    `(it_force, istp, is)`, and
    `fortran_source/ORCHIDEE/src_driver/readdim2.f90::forcing_read_interpol`,
    lines 1450-1661 implement the non-`daily_interpol` interpolation/spreading
    path used by the paper CRUNCEP forcing (`NO_INTER=TRUE`,
    `INTER_LIN=FALSE`, 6-hour forcing, 1800 s SECHIBA).
    """

    model_tstep = int(model_tstep)
    split = int(split)
    nb_spread = int(nb_spread)
    if model_tstep < 0:
        raise IndexError("model_tstep must be non-negative")
    if split < 1:
        raise ValueError("split must be positive")
    if nb_spread < 1:
        raise ValueError("nb_spread must be positive")
    nb_spread = min(nb_spread, split)

    if cache.is_watchout:
        if lon is None or lat is None or land_indices_key is None:
            raise ValueError("WATCHOUT model interpolation requires full domain geometry and land indices")
        return _WatchoutModelCursor(
            cache, split=split, nb_spread=nb_spread,
            dt_force=cache.dt_force if cache.dt_force is not None else dt_force,
            lon=lon, lat=lat, land_indices_key=land_indices_key, netrad_cons=netrad_cons,
        ).read(model_tstep)

    nt = int(cache.Tair.shape[0])
    itau_read = model_tstep // split + 1
    itau_split = model_tstep % split + 1
    itau_read_nm1 = (
        itau_read - 1 if itau_read > 1 else int(round(86400.0 / dt_force))
    )
    prev_index = _paper_forcing_raw_index_for_model_read(itau_read_nm1, nt)
    curr_index = _paper_forcing_raw_index_for_model_read(itau_read, nt)
    rw = float(itau_split) / float(split)

    def lin(name: str) -> np.ndarray:
        values = getattr(cache, name)
        return (values[curr_index] - values[prev_index]) * rw + values[prev_index]

    def spread(name: str) -> np.ndarray:
        values = getattr(cache, name)
        if itau_split <= nb_spread:
            return values[curr_index] * (float(split) / float(nb_spread))
        return np.zeros_like(values[curr_index], dtype=np.float64)

    if float(dt_force) <= 3600.0:
        # readdim2.f90:1613-1617 uses ordinary linear interpolation at
        # hourly or shorter forcing cadence, unless NETRAD_CONS is enabled.
        swdown = cache.SWdown[curr_index] if netrad_cons else lin("SWdown")
    elif land_indices_key is None:
        swdown = _paper_model_step_swdown_from_solarang(
            cache.SWdown[curr_index],
            model_tstep=model_tstep,
            split=split,
            dt_force=dt_force,
            lon=lon,
            lat=lat,
        )
    else:
        swdown = _paper_model_step_swdown_land_from_solarang(
            cache.SWdown[curr_index],
            model_tstep=model_tstep,
            split=split,
            dt_force=dt_force,
            lon=lon,
            lat=lat,
            land_indices_key=land_indices_key,
        )

    # The paper forcing has 6-hourly CRUNCEP records, so the active branch
    # linearly interpolates state variables, redistributes shortwave radiation
    # by solar angle, and spreads precipitation over the configured first half
    # of each forcing interval.
    return _forcing_step_with_values(
        cache,
        tstep=model_tstep,
        Tair=lin("Tair"),
        PSurf=lin("PSurf"),
        Qair=lin("Qair"),
        Wind_E=lin("Wind_E"),
        Wind_N=lin("Wind_N"),
        Rainf=spread("Rainf"),
        Snowf=spread("Snowf"),
        SWdown=swdown,
        LWdown=cache.LWdown[curr_index] if netrad_cons else lin("LWdown"),
    )


def _closed_domain_indices(lon_full: np.ndarray, lat_full: np.ndarray, domain: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Replicate `readdim2.domain_size` closed-interval zoom selection.

    Fortran provenance: `fortran_source/ORCHIDEE/src_driver/readdim2.f90`,
    subroutine `domain_size`, lines 2263-2341. The active case does not cross
    the dateline, so the non-`over_dateline` branch is used.
    """

    west = float(domain["west"])
    east = float(domain["east"])
    south = float(domain["south"])
    north = float(domain["north"])
    lon_axis = lon_full[:, 0]
    lat_axis = lat_full[0, :]

    lon_norm = np.where(lon_axis > 180.0, lon_axis - 360.0, lon_axis)
    lon_norm = np.where(lon_norm < -180.0, lon_norm + 360.0, lon_norm)
    i_index = np.nonzero((lon_norm >= west) & (lon_norm <= east))[0]
    j_index = np.nonzero((lat_axis >= south) & (lat_axis <= north))[0]

    if i_index.size == 0 or j_index.size == 0:
        raise ValueError("domain limits selected no forcing grid cells")
    return i_index, j_index


def _zoom_2d(field: np.ndarray, i_index: np.ndarray, j_index: np.ndarray) -> np.ndarray:
    """Replicate `readdim2.forcing_zoom` for 2D arrays.

    Fortran provenance: `readdim2.f90`, subroutine `forcing_zoom`, lines
    2034-2051.
    """

    return field[np.ix_(i_index, j_index)]


def _land_column_values(field: np.ndarray, land_i: np.ndarray, land_j: np.ndarray) -> np.ndarray:
    """Select land-point values in `forcing_landind` order as a column.

    Fortran provenance: `readdim2.f90::forcing_landind`, lines 1899-1968,
    establishes the land-point order from `(i,j)` indices; `forcing_just_read`,
    lines 1751-1773, reads a full 2-D forcing slab before the selected land
    points are passed onward as `kjpindex` vectors.
    """

    values = np.asarray(field, dtype=np.float64)
    land_i = np.asarray(land_i, dtype=np.int64)
    land_j = np.asarray(land_j, dtype=np.int64)
    if land_i.shape != land_j.shape:
        raise ValueError("land_i and land_j must have matching shapes")
    return values[land_i, land_j].reshape((-1, 1))


def _land_kindex(contfrac: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replicate the active `forcing_landind` land-point order.

    Fortran provenance: `fortran_source/ORCHIDEE/src_driver/readdim2.f90`,
    subroutine `forcing_landind`, lines 1899-1968. In this active path
    `contfrac <= EPSILON(1.)` was converted to missing Tair before the land
    index pass, so positive contfrac determines land points.
    """

    iim, jjm = contfrac.shape
    kindex: list[int] = []
    ij_pairs: list[tuple[int, int]] = []
    for j in range(jjm):
        for i in range(iim):
            if contfrac[i, j] > np.finfo(float).eps:
                kindex.append(j * iim + i + 1)
                ij_pairs.append((i, j))
    if not kindex:
        raise ValueError("zoomed domain contains no positive-contfrac land points")
    return np.asarray(kindex, dtype=np.int32), np.asarray(ij_pairs, dtype=np.int32)


def _regular_lonlat_geometry_if_available(
    lon: np.ndarray,
    lat: np.ndarray,
    kindex: np.ndarray,
    lalo: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return exact regular lon/lat `grid_stuff` geometry when applicable.

    Fortran provenance: `grid.f90::grid_topolylist` lines 575-620 chooses the
    single-point polygon path for `nland == 1` and `haversine_reglatlontoploy`
    for larger regular lon/lat domains; `grid_scatter` lines 688-707 exposes
    sorted neighbours, final area, and segment-derived resolution.
    """

    lon_arr = np.asarray(lon)
    lat_arr = np.asarray(lat)
    kindex_arr = np.asarray(kindex)
    if lon_arr.ndim != 2 or lat_arr.shape != lon_arr.shape or kindex_arr.ndim != 1:
        return None, None, None, None, None
    if kindex_arr.size == 1 and np.asarray(lalo).shape == (1, 2):
        from jax_orchidee.driver.static import regular_lonlat_grid_geometry

        resolution, neighbours, area, seglength, corners = regular_lonlat_grid_geometry(lon_arr, lat_arr, kindex_arr, return_corners=True)
        return resolution, neighbours, area, corners, seglength
    if lon_arr.shape[0] < 2 or lon_arr.shape[1] < 2:
        return None, None, None, None, None
    from jax_orchidee.driver.static import regular_lonlat_grid_geometry

    resolution, neighbours, area, seglength, corners = regular_lonlat_grid_geometry(lon_arr, lat_arr, kindex_arr, return_corners=True)
    return resolution, neighbours, area, corners, seglength


def read_domain_grid(
    config_path: str | Path,
    year: int = 1961,
    *,
    domain_override: dict[str, float] | None = None,
) -> DomainGrid:
    """Read the active paper-case driver domain from config and forcing.

    Fortran provenance: `dim2_driver.f90`, lines 147-175 and 415-435;
    `readdim2.f90`, subroutines `forcing_info` (49-495), `domain_size`
    (2263-2341), `forcing_grid`/`forcing_zoom` (1972-2051), and
    `forcing_landind` (1899-1968). For the paper case's single land point,
    `resolution`, `neighbours`, and final `area` are computed by the exact
    single-point grid path documented in `_paper_single_point_geometry_if_active`.
    """

    config = load_case_config(config_path)
    forcing_path = forcing_file_for_year(config, year)
    domain = dict(config["domain"])
    if domain_override is not None:
        domain.update({key: float(value) for key, value in domain_override.items()})
    with xr.open_dataset(forcing_path, decode_times=False) as ds:
        lon_full, lat_full = _as_fortran_lon_lat(ds)
        i_index, j_index = _closed_domain_indices(lon_full, lat_full, domain)
        lon = _zoom_2d(lon_full, i_index, j_index)
        lat = _zoom_2d(lat_full, i_index, j_index)
        contfrac = _zoom_2d(_as_fortran_2d(ds, "contfrac"), i_index, j_index)
        is_watchout = "levels" in ds
        area2d = None if is_watchout else _zoom_2d(_as_fortran_2d(ds, "Areas"), i_index, j_index)
        if is_watchout:
            from .forcing_readers import _dims, _read_field

            y_dim, x_dim, time_dim = _dims(ds)

            def watchout_grid_field(name):
                return _read_field(ds, name, y_dim=y_dim, x_dim=x_dim, time_dim=time_dim, t0=0, ii=i_index, jj=j_index)

            # readdim2.f90::forcing_read_interpol lines 976-1064 consumes
            # the file geometry, without regular-grid reconstruction.
            watchout_neighbours = np.full((*lon.shape, 8), 999999999, dtype=np.int32)
            for component, suffix in enumerate(("NN", "NE", "EE", "SE", "SS", "SW", "WW", "NW")):
                values = watchout_grid_field("neighbours" + suffix)
                valid = values < 999999999
                rounded = np.where(values[valid] >= 0.0, np.floor(values[valid] + 0.5), np.ceil(values[valid] - 0.5))
                watchout_neighbours[..., component][valid] = rounded.astype(np.int32)
            watchout_resolution = np.stack((watchout_grid_field("resolutionX"), watchout_grid_field("resolutionY")), axis=-1)

    kindex, ij_pairs = _land_kindex(contfrac)
    land_i = ij_pairs[:, 0]
    land_j = ij_pairs[:, 1]
    lalo = np.column_stack((lat[land_i, land_j], lon[land_i, land_j]))
    contfrac_land = contfrac[land_i, land_j]
    area = None if area2d is None else area2d[land_i, land_j]
    forcing_indices = np.column_stack((i_index[land_i], j_index[land_j]))
    if is_watchout:
        resolution = watchout_resolution[land_i, land_j]
        neighbours = watchout_neighbours[land_i, land_j]
        area = resolution[:, 0] * resolution[:, 1]
        corners = seglength = None
    else:
        resolution, neighbours, grid_area, corners, seglength = _regular_lonlat_geometry_if_available(lon, lat, kindex, lalo)
        if grid_area is not None:
            area = grid_area

    return DomainGrid(
        iim=int(lon.shape[0]),
        jjm=int(lon.shape[1]),
        nbindex=int(kindex.size),
        kindex=kindex,
        lon=lon,
        lat=lat,
        lalo=lalo,
        contfrac=contfrac,
        contfrac_land=contfrac_land,
        area=area,
        forcing_indices_zero_based=forcing_indices.astype(np.int32),
        resolution=resolution,
        neighbours=neighbours,
        corners=corners,
        seglength=seglength,
    )


def read_forcing_step(
    config_path: str | Path,
    *,
    domain: DomainGrid | None = None,
    year: int = 1961,
    tstep: int = 0,
) -> ForcingStep:
    """Read one raw forcing slab for the active paper-case path.

    Fortran provenance: `fortran_source/ORCHIDEE/src_driver/readdim2.f90`,
    subroutine `forcing_read_interpol`, lines 660-1687, and subroutine
    `forcing_just_read`, lines 1691-1869. This helper covers the audited
    no-interpolation-needed forcing slab read from NetCDF variables
    `Tair`, `PSurf`, `Qair`, `Wind_N`, `Wind_E`, `Rainf`, `Snowf`,
    `SWdown`, and `LWdown`, for all selected land points in the Fortran
    `forcing_landind` order. It does not implement inactive weather-generator
    or temporal interpolation branches.
    """

    config = load_case_config(config_path)
    forcing_path = forcing_file_for_year(config, year)
    domain = read_domain_grid(config_path, year) if domain is None else domain
    i_index = domain.forcing_indices_zero_based[:, 0]
    j_index = domain.forcing_indices_zero_based[:, 1]
    if int(domain.nbindex) != int(i_index.size) or int(domain.nbindex) != int(j_index.size):
        raise ValueError("domain nbindex must match forcing_indices_zero_based length")
    with xr.open_dataset(forcing_path, decode_times=False) as ds:
        if "levels" in ds:
            cache = _read_watchout_land_cache(ds, config, i_index, j_index, tstep=int(tstep))
            return replace(_forcing_step_from_cached_point(cache, tstep=0), tstep=int(tstep))
        nt = int(ds.sizes.get("tstep", 0))
        if tstep < 0 or tstep >= nt:
            raise IndexError(f"tstep {tstep} outside forcing range 0..{nt - 1}")

        def land_values_at(name: str) -> np.ndarray:
            return _land_column_values(_as_fortran_timestep(ds, name, tstep), i_index, j_index)

        Tair = land_values_at("Tair")
        PSurf = land_values_at("PSurf")
        Qair = land_values_at("Qair")
        Wind_E = land_values_at("Wind_E")
        Wind_N = land_values_at("Wind_N")
        Rainf = land_values_at("Rainf")
        Snowf = land_values_at("Snowf")
        SWdown = land_values_at("SWdown")
        LWdown = land_values_at("LWdown")
        Areas = _land_column_values(_as_fortran_2d(ds, "Areas"), i_index, j_index)
        contfrac = _land_column_values(_as_fortran_2d(ds, "contfrac"), i_index, j_index)
        height_lev1 = float(np.asarray(ds["Height_Lev1"].values))
        height_levuv = float(np.asarray(ds["Height_Levuv"].values))

    zlev = np.full_like(Tair, height_lev1)
    zlevuv = np.full_like(Tair, height_levuv)
    return ForcingStep(
        tstep=int(tstep),
        Tair=Tair,
        PSurf=PSurf,
        Qair=Qair,
        Wind_E=Wind_E,
        Wind_N=Wind_N,
        Rainf=Rainf,
        Snowf=Snowf,
        SWdown=SWdown,
        LWdown=LWdown,
        Areas=Areas,
        contfrac=contfrac,
        Height_Lev1=height_lev1,
        Height_Levuv=height_levuv,
        temp_air=Tair,
        pb=PSurf,
        qair=Qair,
        u=Wind_N,
        v=Wind_E,
        precip_rain=Rainf,
        precip_snow=Snowf,
        swdown=SWdown,
        lwdown=LWdown,
        zlev=zlev,
        zlevuv=zlevuv,
    )


def read_forcing_step_cached(
    config_path: str | Path,
    *,
    domain: DomainGrid,
    year: int = 1961,
    tstep: int = 0,
) -> ForcingStep:
    """Read one forcing slab through a cached selected-land yearly series.

    Fortran provenance and values match ``read_forcing_step``. The only
    difference is IO strategy: selected land points are loaded once per
    config/year/domain index set so driver day loops do not reopen the NetCDF
    file at every half-hour timestep.
    """

    config_resolved = str(Path(config_path).resolve())
    land_indices_key = tuple(
        (int(i), int(j)) for i, j in np.asarray(domain.forcing_indices_zero_based, dtype=np.int64)
    )
    if int(domain.nbindex) != len(land_indices_key):
        raise ValueError("domain nbindex must match forcing_indices_zero_based length")
    cache = _read_forcing_land_cache(config_resolved, int(year), land_indices_key)
    return _forcing_step_from_cached_point(cache, tstep=int(tstep))


def read_forcing_model_step_cached(
    config_path: str | Path,
    *,
    domain: DomainGrid,
    year: int = 1961,
    model_tstep: int = 0,
    split: int = 12,
    nb_spread: int = 6,
    netrad_cons: bool = False,
) -> ForcingStep:
    """Read Fortran-expanded forcing for one SECHIBA model timestep.

    This differs from ``read_forcing_step_cached``, which returns a raw NetCDF
    forcing slab. The paper driver loop calls ``forcing_READ`` for every
    half-hour SECHIBA step with a 6-hour forcing read index and split index;
    see ``_forcing_model_step_from_cached_point`` for line-level provenance.
    """

    config_resolved = str(Path(config_path).resolve())
    land_indices_key = tuple(
        (int(i), int(j)) for i, j in np.asarray(domain.forcing_indices_zero_based, dtype=np.int64)
    )
    if int(domain.nbindex) != len(land_indices_key):
        raise ValueError("domain nbindex must match forcing_indices_zero_based length")
    local_i = (np.asarray(domain.kindex, dtype=np.int64) - 1) % int(domain.iim)
    local_j = (np.asarray(domain.kindex, dtype=np.int64) - 1) // int(domain.iim)
    local_land_indices_key = tuple((int(i), int(j)) for i, j in zip(local_i, local_j))
    cache = _read_forcing_land_cache(config_resolved, int(year), land_indices_key)
    if cache.is_watchout:
        if split < 1 or nb_spread < 1:
            raise ValueError("split and nb_spread must be positive")
        cursor = _watchout_model_cursor(
            config_resolved, int(year), land_indices_key, int(split), int(nb_spread), float(cache.dt_force),
            _array_tuple_key(domain.lon), _array_tuple_key(domain.lat), local_land_indices_key, bool(netrad_cons), id(cache),
        )
        return cursor.read(int(model_tstep))
    return _forcing_model_step_from_cached_point(
        cache,
        model_tstep=int(model_tstep),
        split=int(split),
        nb_spread=int(nb_spread),
        lon=domain.lon,
        lat=domain.lat,
        land_indices_key=local_land_indices_key,
        dt_force=float(split) * 1800.0,
        netrad_cons=bool(netrad_cons),
    )


def read_forcing_first_step(config_path: str | Path, domain: DomainGrid | None = None, year: int = 1961) -> ForcingStep:
    """Read Fortran's first forcing slab for the active paper-case path.

    Fortran provenance: this is the initial ``forcing_just_read(..., itb=1,
    ite=1)`` slab before later temporal splitting; see ``read_forcing_step``.
    """

    return read_forcing_step(config_path, domain=domain, year=year, tstep=0)


@lru_cache(maxsize=128)
def _read_annual_co2_cached(config_path: str, year: int) -> float:
    config = load_case_config(config_path)
    co2_path = Path(config["drivers"]["co2"]["file"])
    with co2_path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if parts and int(parts[0]) == year:
                return float(parts[1])
    raise ValueError(f"no CO2 value found for year {year}")


def read_annual_co2(config_path: str | Path, year: int) -> float:
    """Read annual atmospheric CO2 for `ATM_CO2`.

    Fortran/run provenance: `fortran_run_scripts/paper_250919/Job0_bio`, lines
    334-335, writes the selected annual value into `ATM_CO2`; `dim2_driver.f90`,
    lines 708-719 and 1009, reads `ATM_CO2` and fills `for_ccanopy`.
    """

    return _read_annual_co2_cached(str(Path(config_path).resolve()), int(year))


def read_water_table_sequences(config_path: str | Path) -> WaterTableSequences:
    """Read the two single-column water-table text sequences.

    Fortran provenance: `dim2_driver.f90`, lines 147-151, opens the two files
    on units 103/104; `hydrol.f90`, subroutine `hydrol_soil`, lines 5645-5655,
    reads one row from each unit when `tides` is true.
    """

    config = load_case_config(config_path)
    wt = config["drivers"]["water_table_text"]
    positive = np.loadtxt(Path(wt["positive_file"]), dtype=np.float64)
    differential = np.loadtxt(Path(wt["differential_file"]), dtype=np.float64)
    if positive.ndim != 1 or differential.ndim != 1:
        raise ValueError("water-table forcing files must be single-column sequences")
    if positive.shape != differential.shape:
        raise ValueError("water-table forcing files must have the same length")
    return WaterTableSequences(positive=positive, differential=differential)
