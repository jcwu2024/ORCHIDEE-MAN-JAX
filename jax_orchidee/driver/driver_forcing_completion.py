"""Remaining source-backed driver, solar and weather primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from jax_orchidee.driver.dim2 import Dim2TimeControl, Dim2TimeIndex, iter_dim2_time_indices
from jax_orchidee.driver.domain import _fortran_time_zone


@dataclass(frozen=True)
class DriverWeatherTimestep:
    dt: float
    split: int
    weathergen: bool


@dataclass(frozen=True)
class DriverStepDispatchStatus:
    index: Dim2TimeIndex
    initialize_intersurf: bool
    final_forcing_step: bool
    prepare_load_balance: bool


@dataclass(frozen=True)
class DriverOutputStatus:
    write_standard_restart_fields: bool
    write_pbl_restart_fields: bool
    close_forcing_history: bool
    close_restart_and_dump_config: bool


@dataclass(frozen=True)
class DownwardSolarFluxState:
    initialized: bool = False
    step: float | None = None
    eccentricity: float = 0.016724
    perihelie: float = 102.04
    obliquity: float = 23.446


@dataclass(frozen=True)
class DownwardSolarFluxResult:
    solad: np.ndarray
    solai: np.ndarray
    state: DownwardSolarFluxState

    def __iter__(self) -> Iterator[np.ndarray]:
        """Preserve the earlier ``solad, solai = result`` interface."""

        yield self.solad
        yield self.solai


class NetCDFStatusError(RuntimeError):
    def __init__(self, status: int, message: str):
        self.status = int(status)
        self.netcdf_message = str(message)
        super().__init__(f"nccheck: NetCDF error {status}: {message}")


def nccheck(
    status: int,
    *,
    no_error: int = 0,
    strerror: Callable[[int], str] | None = None,
) -> None:
    """Raise for the error arm of ``src_global/utils.f90::nccheck``."""

    if int(status) != int(no_error):
        message = strerror(int(status)) if strerror is not None else f"status {status}"
        raise NetCDFStatusError(int(status), message)


def resolve_driver_weather_timestep(
    *, dt_force: float, dt_sechiba: float, weathergen: bool = False
) -> DriverWeatherTimestep:
    """Implement ``dim2_driver.f90::driver`` lines 285-311."""

    dt_force, dt_sechiba = float(dt_force), float(dt_sechiba)
    if dt_force <= 0.0 or dt_sechiba <= 0.0:
        raise ValueError("dt_force and dt_sechiba must be positive")
    if weathergen:
        return DriverWeatherTimestep(dt_force, 1, True)
    split = int(dt_force / dt_sechiba)
    if split < 1:
        raise ValueError("DT_SECHIBA exceeds the forcing timestep")
    return DriverWeatherTimestep(dt_sechiba, split, False)


def require_fixed_paper_weather_gate(
    *, dt_force: float, dt_sechiba: float, weathergen: bool = False
) -> DriverWeatherTimestep:
    """Enforce the paper run's fixed ``ALLOW_WEATHERGEN=false`` protocol."""

    result = resolve_driver_weather_timestep(
        dt_force=dt_force, dt_sechiba=dt_sechiba, weathergen=weathergen
    )
    if result.weathergen:
        raise NotImplementedError("the fixed paper driver requires ALLOW_WEATHERGEN=false")
    return result


def driver_step_dispatch_status(
    time: Dim2TimeControl, index: Dim2TimeIndex
) -> DriverStepDispatchStatus:
    """Resolve lines 835-837, 1051-1058, and 1403-1407 for one step."""

    if not (time.itau_dep < index.it <= time.itau_fin and 1 <= index.isplit <= time.split):
        raise ValueError("index is outside the supplied DIM2 time control")
    return DriverStepDispatchStatus(
        index,
        bool(index.lstep_init),
        bool(index.lstep_last),
        index.it == time.itau_fin - 1,
    )


def iter_driver_step_dispatch_statuses(
    time: Dim2TimeControl,
) -> tuple[DriverStepDispatchStatus, ...]:
    return tuple(driver_step_dispatch_status(time, item) for item in iter_dim2_time_indices(time))


def driver_output_status(*, is_watchout: bool, is_root_prc: bool) -> DriverOutputStatus:
    """Resolve common/PBL restart writes and root-only closes, lines 1415-1446."""

    return DriverOutputStatus(True, not bool(is_watchout), True, bool(is_root_prc))


def weathgen_qsat_2d(t: np.ndarray, p: np.ndarray, *, zero_t: float = 273.15) -> np.ndarray:
    """Exact polynomial in ``weather.f90::weathgen_qsat_2d`` 3623-3668."""

    t, p = np.broadcast_arrays(np.asarray(t, dtype=np.float64), np.asarray(p, dtype=np.float64))
    a = (6.1078, 4.4365185e-1, 1.4289458e-2, 2.6506485e-4, 3.0312404e-6, 2.0340809e-8, 6.1368209e-11)
    b = (6.109178, 5.034699e-1, 1.8860134e-2, 4.1762237e-4, 5.8247203e-6, 4.8388032e-8, 1.8388269e-10)
    tl = np.minimum(100.0, np.maximum(t - zero_t, 0.0))
    ti = np.maximum(-60.0, np.minimum(t - zero_t, 0.0))
    e = 100.0 * (np.where(t > zero_t, a[0], b[0]) + tl * (a[1] + tl * (a[2] + tl * (a[3] + tl * (a[4] + tl * (a[5] + tl * a[6]))))) + ti * (b[1] + ti * (b[2] + ti * (b[3] + ti * (b[4] + ti * (b[5] + ti * b[6]))))))
    return 0.622 * e / np.maximum(p - (1.0 - 0.622) * e, 0.622 * e)


def solarang(julian: float, julian0: float, lon: np.ndarray, lat: np.ndarray, *, one_year: float = 365.0) -> np.ndarray:
    """Cosine of zenith angle from ``solar.f90::solarang`` lines 55-181."""

    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    if lon.ndim != 2 or lat.shape != lon.shape or not all(lon.shape):
        raise ValueError("lon and lat must have the same non-empty two-dimensional shape")
    if float(one_year) <= 0.0:
        raise ValueError("one_year must be positive")
    gamma = 2.0 * math.pi * (float(julian) - float(julian0)) / float(one_year)
    dec = (
        0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma) + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma) + 0.00148 * math.sin(3.0 * gamma)
    )
    et = (
        0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma) - 0.04089 * math.sin(2.0 * gamma)
    ) * 229.18
    # Gamma uses julian-julian0, but source GMT uses absolute julian.
    gmt = 24.0 * (float(julian) - math.trunc(float(julian)))
    zone, lhour = _fortran_time_zone(lon, gmt=gmt)
    ls = ((zone.astype(np.float64) - 1.0) * 15.0) - 180.0
    latime = lhour + (4.0 * (ls - lon[:, 0]) * -1.0) / 60.0 + et / 60.0
    latime = np.where(latime < 0.0, latime + 24.0, latime)
    latime = np.where(latime > 24.0, latime - 24.0, latime)
    omega = (latime - 12.0) * -15.0 * math.pi / 180.0
    llat = lat[0, :] * math.pi / 180.0
    return np.maximum(
        0.0,
        math.sin(dec) * np.sin(llat)[None, :]
        + math.cos(dec) * np.cos(llat)[None, :] * np.cos(omega)[:, None],
    )


def downward_solar_flux(
    latitude: np.ndarray,
    *,
    calendar: str | None = None,
    calendar_str: str | None = None,
    jday: float,
    rtime: float,
    cloud: np.ndarray,
    nband: int,
    one_year: float = 365.0,
    eccentricity: float = 0.016724,
    perihelie: float = 102.04,
    obliquity: float = 23.446,
    state: DownwardSolarFluxState | None = None,
) -> DownwardSolarFluxResult:
    """Direct/diffuse radiation from ``downward_solar_flux`` 290-597."""

    latitude, cloud = np.broadcast_arrays(np.asarray(latitude, dtype=np.float64), np.asarray(cloud, dtype=np.float64))
    if latitude.ndim != 1:
        raise ValueError("latitude and cloud must be one-dimensional")
    if int(nband) != nband or int(nband) < 1:
        raise ValueError("nband must be a positive integer")
    selected_calendar = calendar_str if calendar_str is not None else calendar
    if selected_calendar is None:
        raise ValueError("calendar or calendar_str is required")
    current = state or DownwardSolarFluxState(
        eccentricity=eccentricity, perihelie=perihelie, obliquity=obliquity
    )
    if not current.initialized:
        step = 1.0 if selected_calendar.strip() == "gregorian" else float(one_year) / 365.2425
        current = DownwardSolarFluxState(
            True, step, current.eccentricity, current.perihelie, current.obliquity
        )
    if current.step is None:
        raise ValueError("initialized downward-solar state requires step")
    step = current.step
    eccentricity, perihelie, obliquity = (
        current.eccentricity,
        current.perihelie,
        current.obliquity,
    )
    pir = math.pi / 180.0
    xl = eccentricity * 0.0 + perihelie + 180.0
    so = math.sin(obliquity * pir)
    xllp = xl * pir
    xee = eccentricity * eccentricity
    xse = math.sqrt(1.0 - xee)
    xlam = (eccentricity / 2.0 + eccentricity * xee / 8.0) * (1.0 + xse) * math.sin(xllp) - xee / 4.0 * (0.5 + xse) * math.sin(2.0 * xllp) + eccentricity * xee / 8.0 * (1.0 / 3.0 + xse) * math.sin(3.0 * xllp)
    xlam = 2.0 * xlam / pir
    dlamm = xlam + (int(jday) - 79) * step
    ranm = (dlamm - xl) * pir
    xee3 = xee * eccentricity
    ranv = ranm + (2.0 * eccentricity - xee3 / 4.0) * math.sin(ranm) + 5.0 / 4.0 * eccentricity**2 * math.sin(2.0 * ranm) + 13.0 / 12.0 * xee3 * math.sin(3.0 * ranm)
    rlam = (ranv / pir + xl) * pir
    sd = so * math.sin(rlam)
    cd = math.sqrt(1.0 - sd * sd)
    xdecl = math.atan(sd / cd)
    distance = (1.0 - eccentricity**2) / (1.0 + eccentricity * math.cos(ranv))
    sw = (1.0 / distance) ** 2 * 1370.0
    angle = 2.0 * math.pi * (float(rtime) - 12.0) / 24.0
    xlat = latitude * math.pi / 180.0
    coszen = np.maximum(0.0, np.sin(xlat) * math.sin(xdecl) + np.cos(xlat) * math.cos(xdecl) * math.cos(angle))
    trans = 0.251 + 0.509 * (1.0 - cloud)
    fdiffuse = 1.0045 + trans * (0.0435 + trans * (-3.5227 + trans * 2.6313))
    fdiffuse = np.where(trans > 0.75, 0.166, fdiffuse)
    fractions = np.asarray([0.46 + 0.08 * ib for ib in range(nband)] if nband == 2 else [1.0 / nband] * nband)
    total = sw * coszen[..., None] * fractions
    return DownwardSolarFluxResult(
        total * trans[..., None] * (1.0 - fdiffuse[..., None]),
        total * trans[..., None] * fdiffuse[..., None],
        current,
    )


def validate_driver_dimensions(*, forcing_shape: tuple[int, int, int], model_shape: tuple[int, int, int], forcing_steps: int, requested_steps: int) -> None:
    """Own the scientific dimension/time guards in ``dim2_driver::driver``.

    Lines 443-452 and 569 select multi-cell testing, reject inconsistent
    dimensions and guard requests longer than the forcing record count.
    """

    if tuple(forcing_shape) != tuple(model_shape):
        raise ValueError("forcing and model dimensions differ")
    if int(forcing_steps) < int(requested_steps):
        raise ValueError("forcing file is shorter than requested simulation")
