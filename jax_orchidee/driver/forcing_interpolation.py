"""Stateful forcing interpolation owned by ``forcing_read_interpol``.

Fortran source truth: ``src_driver/readdim2.f90::forcing_read_interpol``,
lines 660-1687.  The state object below makes the procedure's local ``SAVE``
variables explicit so one compiled transition can serve different landpoints,
forcing cadences, and standard or Watchout records.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Mapping

import numpy as np

from .driver_forcing_completion import solarang, weathgen_qsat_2d


_BASE_FIELDS = (
    "zlev", "zlevuv", "swdown", "rainf", "snowf", "tair", "u", "v",
    "qair", "pb", "lwdown",
)
_WATCHOUT_FIELDS = (
    "swnet", "eair", "petacoef", "peqacoef", "petbcoef", "peqbcoef",
    "cdrag", "ccanopy",
)


@dataclass(frozen=True)
class ForcingInterpolationRecord:
    zlev: np.ndarray
    zlevuv: np.ndarray
    swdown: np.ndarray
    rainf: np.ndarray
    snowf: np.ndarray
    tair: np.ndarray
    u: np.ndarray
    v: np.ndarray
    qair: np.ndarray
    pb: np.ndarray
    lwdown: np.ndarray
    swnet: np.ndarray
    eair: np.ndarray
    petacoef: np.ndarray
    peqacoef: np.ndarray
    petbcoef: np.ndarray
    peqbcoef: np.ndarray
    cdrag: np.ndarray
    ccanopy: np.ndarray
    tmax: np.ndarray | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, np.ndarray]) -> "ForcingInterpolationRecord":
        base = np.asarray(values["tair"], dtype=np.float64)
        zero = np.zeros_like(base)
        kwargs = {}
        for item in fields(cls):
            value = values.get(item.name)
            if item.name == "tmax" and value is None:
                kwargs[item.name] = None
            elif value is None:
                kwargs[item.name] = zero.copy()
            else:
                array = np.asarray(value, dtype=np.float64)
                if array.shape != base.shape:
                    raise ValueError(f"{item.name} shape {array.shape} != tair shape {base.shape}")
                kwargs[item.name] = array
        return cls(**kwargs)


@dataclass(frozen=True)
class ForcingInterpolationState:
    initialized: bool = False
    last_read: int = 0
    itau_read_nm1: int = 0
    itau_read_n: int = 0
    previous: ForcingInterpolationRecord | None = None
    current: ForcingInterpolationRecord | None = None
    tmin_nm2: np.ndarray | None = None
    daylength_nm1: np.ndarray | None = None
    daylength_n: np.ndarray | None = None
    mean_coszang: np.ndarray | None = None
    julian_for: float = 0.0
    julian0: float = 0.0
    wind_n_exists: bool = False
    contfrac_exists: bool = False
    neighbours_exists: bool = False
    is_watchout: bool = False


@dataclass(frozen=True)
class ForcingInterpolationResult:
    values: ForcingInterpolationRecord
    coszang: np.ndarray
    state: ForcingInterpolationState


@dataclass(frozen=True)
class ForcingGridMetadata:
    contfrac: np.ndarray
    kindex: np.ndarray
    nbindex: int
    neighbours: np.ndarray
    resolution: np.ndarray


def forcing_grid_metadata(
    tair: np.ndarray,
    *,
    contfrac: np.ndarray | None,
    watchout: bool,
    neighbours: np.ndarray | None = None,
    resolution: np.ndarray | None = None,
    ii_begin: int = 1,
    ii_end: int | None = None,
) -> ForcingGridMetadata:
    """Grid initialization at lines 851-1070, including overlap masks."""

    tair = np.asarray(tair, dtype=np.float64)
    if tair.ndim != 2:
        raise ValueError("tair must be two-dimensional")
    iim, jjm = tair.shape
    ii_end = iim if ii_end is None else int(ii_end)
    if watchout and contfrac is None:
        raise ValueError("Watchout forcing requires contfrac")
    fraction = np.ones_like(tair) if contfrac is None else np.asarray(contfrac, dtype=np.float64).copy()
    if fraction.shape != tair.shape:
        raise ValueError("contfrac shape must match tair")
    if watchout or contfrac is not None:
        fraction[: max(int(ii_begin) - 1, 0), 0] = 0.0
        fraction[min(ii_end, iim) :, -1] = 0.0
    valid = fraction > np.finfo(np.float64).eps
    kindex = np.asarray(
        [j * iim + i + 1 for j in range(jjm) for i in range(iim) if valid[i, j]],
        dtype=np.int32,
    )
    if not kindex.size:
        raise ValueError("forcing grid contains no land points")
    if watchout:
        if neighbours is None or resolution is None:
            raise ValueError("Watchout forcing requires neighbours and resolution")
        neighbours_out = np.asarray(neighbours, dtype=np.int32)
        resolution_out = np.asarray(resolution, dtype=np.float64)
        if neighbours_out.shape != (iim, jjm, 8):
            raise ValueError("neighbours must have shape (iim, jjm, 8)")
        if resolution_out.shape != (iim, jjm, 2):
            raise ValueError("resolution must have shape (iim, jjm, 2)")
    else:
        neighbours_out = np.full((iim, jjm, 8), -1, dtype=np.int32) if neighbours is None else np.asarray(neighbours, dtype=np.int32)
        resolution_out = np.zeros((iim, jjm, 2), dtype=np.float64) if resolution is None else np.asarray(resolution, dtype=np.float64)
    return ForcingGridMetadata(fraction, kindex, int(kindex.size), neighbours_out, resolution_out)


def _record(values: dict[str, np.ndarray], template: ForcingInterpolationRecord) -> ForcingInterpolationRecord:
    return ForcingInterpolationRecord(**{item.name: values.get(item.name, getattr(template, item.name)) for item in fields(template)})


def _linear(previous: np.ndarray, current: np.ndarray, weight: float) -> np.ndarray:
    return (current - previous) * weight + previous


def _daily_wave(
    hour: float,
    start_n: np.ndarray,
    start_nm1: np.ndarray,
    minimum_n: np.ndarray,
    minimum_nm1: np.ndarray,
    maximum_n: np.ndarray,
    maximum_nm1: np.ndarray,
) -> np.ndarray:
    morning = (maximum_n - minimum_n) / 2.0 * np.sin(
        math.pi / (14.0 - start_n) * (hour - 0.5 * (14.0 - start_n) - start_n)
    ) + (maximum_n + minimum_n) / 2.0
    early_afternoon = (maximum_nm1 - minimum_nm1) / 2.0 * np.sin(
        math.pi / (14.0 - start_n) * (hour - 0.5 * (14.0 - start_n) - start_n)
    ) + (maximum_nm1 + minimum_nm1) / 2.0
    before_dawn = (maximum_nm1 - minimum_n) / 2.0 * np.sin(
        math.pi / (24.0 - 14.0 + start_nm1)
        * (hour + 24.0 + 0.5 * (24.0 - 14.0 + start_nm1) - 14.0)
    ) + (maximum_nm1 + minimum_n) / 2.0
    evening = (maximum_nm1 - minimum_n) / 2.0 * np.sin(
        math.pi / (24.0 - 14.0 + start_n)
        * (hour + 0.5 * (24.0 - 14.0 + start_n) - 14.0)
    ) + (maximum_nm1 + minimum_n) / 2.0
    return np.where(
        (hour >= start_n) & (hour > 12.0) & (hour <= 14.0), early_afternoon,
        np.where((hour >= start_n) & (hour <= 12.0), morning, np.where(hour < start_n, before_dawn, evening)),
    )


def forcing_read_interpol_transition(
    records: tuple[ForcingInterpolationRecord, ...],
    state: ForcingInterpolationState,
    *,
    itauin: int,
    itau_split: int,
    split: int,
    nb_spread: int,
    dt_force: float,
    lon: np.ndarray,
    lat: np.ndarray,
    daily_interpol: bool,
    is_watchout: bool,
    netrad_cons: bool = False,
    julian_for: float = 0.0,
    julian0: float = 0.0,
) -> ForcingInterpolationResult:
    """Execute defined scientific transitions from lines 780-1687.

    Records use Fortran one-based time semantics through ``itauin`` and wrap
    at the end of the forcing file. Initialization is represented explicitly
    by ``itauin=0, itau_split=0``.
    """

    if not records or split < 1 or nb_spread < 1:
        raise ValueError("records, split, and nb_spread must be non-empty/positive")
    ttm = len(records)
    if itauin == 0 and itau_split == 0:
        first = records[0]
        zero = np.zeros_like(first.tair)
        initialized = ForcingInterpolationState(
            initialized=True, previous=None, current=None,
            tmin_nm2=zero.copy() if daily_interpol else None,
            daylength_nm1=zero.copy() if daily_interpol else None,
            daylength_n=zero.copy() if daily_interpol else None,
            mean_coszang=zero.copy(), julian_for=julian_for, julian0=julian0,
            wind_n_exists=state.wind_n_exists, contfrac_exists=state.contfrac_exists,
            neighbours_exists=state.neighbours_exists, is_watchout=is_watchout,
        )
        return ForcingInterpolationResult(first, zero, initialized)
    if not state.initialized:
        raise RuntimeError("forcing_read_interpol must be initialized first")
    itau_read = (int(itauin) - 1) % ttm + 1
    previous = state.previous
    current = state.current
    last_read = state.last_read
    read_nm1 = state.itau_read_nm1
    read_n = state.itau_read_n
    tmin_nm2 = state.tmin_nm2
    day_nm1 = state.daylength_nm1
    day_n = state.daylength_n
    mean_cos = state.mean_coszang
    interval_julian = state.julian_for
    weight = (
        1.0
        if split == 1
        else (float(itau_split + split / 2.0) / split)
        if daily_interpol and itau_split <= split / 2.0
        else (float(itau_split - split / 2.0) / split)
        if daily_interpol
        else float(itau_split) / float(split)
        if split > 1
        else 1.0
    )
    should_read = (
        last_read == 0 or math.isclose(weight, 1.0 / split, rel_tol=0.0, abs_tol=0.0)
        if daily_interpol
        else itau_read != last_read
    )
    first_daily_read = daily_interpol and last_read == 0
    if should_read:
        if read_n == 0:
            read_nm1 = itau_read - 1 if itau_read > 1 else int(round(86400.0 / dt_force))
            if not 1 <= read_nm1 <= ttm:
                raise ValueError("forcing file contains less than 24 hours")
            previous = records[read_nm1 - 1]
            if daily_interpol:
                tmin_nm2 = previous.tair.copy()
        else:
            previous = current
            read_nm1 = read_n
        read_n = itau_read if (daily_interpol and last_read == 0) else itau_read + int(daily_interpol)
        if read_n > ttm:
            read_n = 1
        current = records[read_n - 1]
        last_read = read_n
        interval_julian = julian_for + (itau_read - 1) * dt_force / 86400.0
        if dt_force > 3600.0 and not daily_interpol:
            mean_cos = np.zeros_like(current.tair)
            for index in range(1, split + 1):
                mean_cos += solarang(
                    interval_julian + (index - 0.5) / split * dt_force / 86400.0,
                    julian0,
                    lon,
                    lat,
                )
            mean_cos /= split
        if first_daily_read and dt_force > 3600.0:
            mean_cos = np.zeros_like(current.tair)
            day_n = np.zeros_like(current.tair)
            for index in range(1, split + 1):
                angle = solarang(
                    interval_julian + (index - 0.5) / split * dt_force / 86400.0,
                    julian0,
                    lon * 0.0,
                    lat,
                )
                mean_cos += angle
                day_n += np.where(angle > 0.0, 24.0 / split, 0.0)
            mean_cos /= split
            day_nm1 = day_n.copy()
    if previous is None or current is None:
        raise RuntimeError("forcing interpolation records were not initialized")
    values = {name: _linear(getattr(previous, name), getattr(current, name), weight) for name in ("pb", "u", "v")}
    cosz = np.zeros_like(current.tair)
    if daily_interpol:
        if current.tmax is None or previous.tmax is None:
            raise ValueError("daily interpolation requires tmax in every record")
        if day_n is None or day_nm1 is None or mean_cos is None:
            raise ValueError("daily interpolation state is incomplete")
        if itau_split == 1 and dt_force > 3600.0:
            day_nm1 = day_n
            day_n = np.zeros_like(current.tair)
            mean_cos = np.zeros_like(current.tair)
            for index in range(1, split + 1):
                angle = solarang(interval_julian + (index - 0.5) / split * dt_force / 86400.0, julian0, lon * 0.0, lat)
                mean_cos += angle
                day_n += np.where(angle > 0.0, 24.0 / split, 0.0)
            mean_cos /= split
        hour = float(itau_split) / split * 24.0
        start_n = 12.0 - day_n / 2.0
        start_nm1 = 12.0 - day_nm1 / 2.0
        tair = _daily_wave(hour, start_n, start_nm1, current.tair, previous.tair, current.tmax, previous.tmax)
        qs_n = weathgen_qsat_2d(current.tair, values["pb"])
        qs_nm1 = weathgen_qsat_2d(previous.tair, values["pb"])
        qs_tair = weathgen_qsat_2d(tair, values["pb"])
        qmin_n = np.minimum(current.qair, 0.99 * qs_n)
        qmin_nm1 = np.minimum(previous.qair, 0.99 * qs_nm1)
        qmax_n = 2.0 * current.qair - qmin_n
        qmax_nm1 = 2.0 * previous.qair - qmin_nm1
        qair = np.minimum(0.99 * qs_tair, _daily_wave(hour, start_n, start_nm1, qmin_n, qmin_nm1, qmax_n, qmax_nm1))
        values.update(tair=tair, qair=qair, zlev=_linear(previous.zlev, current.zlev, weight), zlevuv=_linear(previous.zlevuv, current.zlevuv, weight))
        source = current.rainf if hour <= 12.0 else previous.rainf
        active = itau_split <= nb_spread
        amount = source * (float(split) / nb_spread) if active else np.zeros_like(source)
        values["rainf"] = np.where(tair >= 273.15, amount, 0.0)
        values["snowf"] = np.where(tair < 273.15, amount, 0.0)
    else:
        values.update({name: _linear(getattr(previous, name), getattr(current, name), weight) for name in ("tair", "qair")})
        active = itau_split <= nb_spread
        values["rainf"] = current.rainf * (float(split) / nb_spread) if active else np.zeros_like(current.rainf)
        values["snowf"] = current.snowf * (float(split) / nb_spread) if active else np.zeros_like(current.snowf)
        if is_watchout:
            values["zlev"] = _linear(previous.zlev, current.zlev, weight)
            values["zlevuv"] = values["zlev"]
    for name in _WATCHOUT_FIELDS:
        if is_watchout:
            values[name] = _linear(getattr(previous, name), getattr(current, name), weight)
    values["lwdown"] = current.lwdown if netrad_cons else _linear(previous.lwdown, current.lwdown, weight)
    if dt_force > 3600.0:
        cosz = solarang(interval_julian + (itau_split - 0.5) / split * dt_force / 86400.0, julian0, lon * (0.0 if daily_interpol else 1.0), lat)
        denominator = np.asarray(mean_cos)
        solar_source = current.swdown
        if daily_interpol:
            solar_source = np.where(float(itau_split) / split * 24.0 <= 12.0, current.swdown, previous.swdown)
        values["swdown"] = np.divide(
            solar_source * cosz,
            denominator,
            out=np.zeros_like(solar_source),
            where=denominator > 0.0,
        )
        values["swdown"] = np.minimum(values["swdown"], 2000.0)
    else:
        values["swdown"] = current.swdown if netrad_cons else _linear(previous.swdown, current.swdown, weight)
    next_state = ForcingInterpolationState(
        initialized=True, last_read=last_read, itau_read_nm1=read_nm1,
        itau_read_n=read_n, previous=previous, current=current,
        tmin_nm2=tmin_nm2, daylength_nm1=day_nm1, daylength_n=day_n,
        mean_coszang=mean_cos, julian_for=interval_julian, julian0=julian0,
        wind_n_exists=state.wind_n_exists, contfrac_exists=state.contfrac_exists,
        neighbours_exists=state.neighbours_exists, is_watchout=is_watchout,
    )
    return ForcingInterpolationResult(_record(values, current), cosz, next_state)
