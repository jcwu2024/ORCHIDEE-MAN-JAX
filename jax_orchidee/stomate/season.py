"""STOMATE season-memory updates from the audited Fortran ``season`` path.

This module intentionally covers only local memory algebra whose inputs are
already explicit. It does not synthesize phenology, yearly climatology, or
herbivore state absent from the caller.
"""

from __future__ import annotations

from typing import NamedTuple

from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp


ZERO_CELSIUS = 273.15
UNDEF = -9999.0
MIN_STOMATE = 1.0e-8
ONE_YEAR_DAYS = 365.0
SPRING_DAYS_MAX = 40
T_LONG_REF_MIN = 253.1
T_LONG_REF_MAX = 303.1
FORTRAN_EPSILON_R4 = jnp.finfo(jnp.float32).eps

SEASON_MEMORY_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 4221-4251 calls season with dt_days",
    "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 386-473 initializes uninitialized season memory",
    "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 577-772 updates moisture, air-temperature, spring Tmin, soil-temperature, and soil-humidity memories",
    "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 832-846 updates gdd_from_growthinit for non-natural PFTs and sets natural PFTs to undef",
    "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 1419-1443 derives and resets Tseason at EndOfYear",
    "fortran_source/ORCHIDEE/src_parameters/constantes.f90 lines 1351-1397, 1423-1429, and 2077-2101 define the run.def keys used here",
    "fortran_source/ORCHIDEE/src_stomate/stomate_data.f90 lines 697-700 sets tau_longterm_max = coeff_tau_longterm * one_year",
)


class SeasonTimeScales(NamedTuple):
    """Time constants used by ``stomate_season::season`` in days."""

    tau_hum_month: float = 20.0
    tau_hum_week: float = 7.0
    tau_t2m_month: float = 20.0
    tau_t2m_week: float = 7.0
    tau_tsoil_month: float = 20.0
    tau_soilhum_month: float = 20.0
    tau_gpp_week: float = 7.0
    tau_gdd: float = 40.0
    tau_ngd: float = 50.0
    tau_climatology: float = 20.0
    coeff_tau_longterm: float = 3.0
    tlong_ref_min: float = T_LONG_REF_MIN
    tlong_ref_max: float = T_LONG_REF_MAX
    one_year: float = ONE_YEAR_DAYS

    @property
    def tau_longterm_max(self) -> float:
        return float(self.coeff_tau_longterm) * float(self.one_year)


class SeasonMemoryState(NamedTuple):
    """Restart-backed memory fields updated before STOMATE carbon processes."""

    moiavail_month: jnp.ndarray
    moiavail_week: jnp.ndarray
    t2m_longterm: jnp.ndarray
    tau_longterm: float
    t2m_month: jnp.ndarray
    t2m_week: jnp.ndarray
    tseason: jnp.ndarray
    tseason_length: jnp.ndarray
    tseason_tmp: jnp.ndarray
    tmin_spring_time: jnp.ndarray
    onset_date: jnp.ndarray
    tsoil_month: jnp.ndarray
    soilhum_month: jnp.ndarray
    gdd_from_growthinit: jnp.ndarray
    tsurf_year: jnp.ndarray | None = None
    t2m_14: jnp.ndarray | None = None


class SeasonSurfaceTemperatureMemoryResult(NamedTuple):
    """The two scalar-per-land season memories omitted by legacy callers."""

    tsurf_year: jnp.ndarray
    t2m_14: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 439-452",
        "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 736-744",
    )


def season_surface_temperature_memory_step(
    *,
    tsurf_year,
    t2m_14,
    tsurf_daily,
    t2m_daily,
    dt_days,
    firstcall: bool,
    tau_t2m_14: float = 14.0,
    min_stomate: float = MIN_STOMATE,
) -> SeasonSurfaceTemperatureMemoryResult:
    """Advance ``tsurf_year`` and ``t2m_14`` exactly as Fortran ``season``.

    ``tsurf_year`` has only the first-call initialization in this source;
    ``t2m_14`` is initialized first and then relaxed on every daily call.
    """

    tsurf = jnp.asarray(tsurf_year)
    t14 = jnp.asarray(t2m_14)
    tsurf_day = jnp.asarray(tsurf_daily, dtype=tsurf.dtype)
    t2m_day = jnp.asarray(t2m_daily, dtype=t14.dtype)
    dt = jnp.asarray(dt_days, dtype=t14.dtype)
    tau = jnp.asarray(tau_t2m_14, dtype=t14.dtype)
    if (
        tsurf.shape != tsurf_day.shape
        or t14.shape != t2m_day.shape
        or tsurf.shape != t14.shape
    ):
        raise ValueError(
            "season surface-temperature memories and daily inputs must share shape (npts,)"
        )
    if bool(firstcall):
        tsurf = jnp.where(jnp.abs(jnp.sum(tsurf)) < min_stomate, tsurf_day, tsurf)
        t14 = jnp.where(jnp.abs(jnp.sum(t14)) < min_stomate, t2m_day, t14)
    t14 = (t14 * (tau - dt) + t2m_day * dt) / tau
    t14 = jnp.where(jnp.abs(t14) < jnp.finfo(t14.dtype).eps, t2m_day, t14)
    return SeasonSurfaceTemperatureMemoryResult(tsurf_year=tsurf, t2m_14=t14)


class SeasonMemoryStepResult(NamedTuple):
    """Updated season-memory state and provenance for the local step."""

    state: SeasonMemoryState
    provenance: tuple[str, ...] = SEASON_MEMORY_PROVENANCE
    notes: tuple[str, ...] = (
        "This is the local season-memory subset before phenology/allocation/turnover.",
        "It leaves gdd_m5_dormance, gdd_midwinter, ncd_dormance, ngd_minus5, GPP/NPP climatology, and herbivores to their source-backed process implementations.",
    )


class SeasonAnnualState(NamedTuple):
    """Long-term carbon fluxes and annual statistics updated in ``season``."""

    npp_longterm: jnp.ndarray
    turnover_longterm: jnp.ndarray
    gpp_week: jnp.ndarray
    maxmoiavail_lastyear: jnp.ndarray
    maxmoiavail_thisyear: jnp.ndarray
    minmoiavail_lastyear: jnp.ndarray
    minmoiavail_thisyear: jnp.ndarray
    maxgppweek_lastyear: jnp.ndarray
    maxgppweek_thisyear: jnp.ndarray
    gdd0_lastyear: jnp.ndarray
    gdd0_thisyear: jnp.ndarray
    precip_lastyear: jnp.ndarray
    precip_thisyear: jnp.ndarray
    lm_lastyearmax: jnp.ndarray
    lm_thisyearmax: jnp.ndarray
    maxfpc_lastyear: jnp.ndarray
    maxfpc_thisyear: jnp.ndarray


class SeasonAnnualStepResult(NamedTuple):
    """Updated annual/long-term state for ``stomate_season`` sections 12-21."""

    state: SeasonAnnualState
    herbivores: jnp.ndarray
    provenance: tuple[str, ...] = (
        *SEASON_MEMORY_PROVENANCE,
        "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 1147-1233 updates npp_longterm, turnover_longterm, gpp_week, and moisture extrema",
        "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 1236-1380 updates annual GPP, GDD0, precipitation, leaf-mass, and maxfpc statistics",
        "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 1389-1459 rolls annual statistics at EndOfYear",
        "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 1488-1580 diagnoses herbivores",
        "fortran_source/ORCHIDEE/src_parameters/constantes.f90 lines 1399-1405 and 2039-2045 define TAU_GPP_WEEK and TAU_CLIMATOLOGY",
        "fortran_source/ORCHIDEE/src_parameters/constantes.f90 lines 2053-2117 define HVC and green-age constants",
    )
    notes: tuple[str, ...] = (
        "The ok_dgvm=false paper path and ok_dgvm=true annual maxfpc/lm relaxation branch are source-covered locally.",
    )


class SeasonBiometeorologyState(NamedTuple):
    """Dormancy/GDD biometeorological memory updated by ``season`` sections 7-11."""

    gdd_m5_dormance: jnp.ndarray
    gdd_midwinter: jnp.ndarray
    ncd_dormance: jnp.ndarray
    ngd_minus5: jnp.ndarray
    time_hum_min: jnp.ndarray
    hum_min_dormance: jnp.ndarray


class SeasonBiometeorologyStepResult(NamedTuple):
    """Updated section 7-11 season biometeorology memory."""

    state: SeasonBiometeorologyState
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 848-911 updates gdd_m5_dormance",
        "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 914-976 updates gdd_midwinter",
        "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 979-1029 updates ncd_dormance",
        "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 1032-1071 updates ngd_minus5",
        "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 1073-1131 updates time_hum_min and hum_min_dormance",
    )


def season_midwinter_solad(
    latitude,
    julian_diff,
    *,
    calendar_str="gregorian",
    rtime=12.0,
    cloud=None,
    eccentricity=0.016724,
    perihelie=102.04,
    obliquity=23.446,
) -> jnp.ndarray:
    """Return direct noon solar flux used by ``season`` for GDD init date.

    Fortran provenance: ``src_global/solar.f90::downward_solar_flux`` lines
    290-597 with ``nband=1``, ``rtime=12``, and zero cloud as called by
    ``src_stomate/stomate_season.f90::season`` lines 537-543.
    """

    latitude = jnp.asarray(latitude)
    cloud_arr = jnp.zeros_like(latitude) if cloud is None else jnp.asarray(cloud)
    pir = jnp.pi / 180.0
    step = 1.0 if str(calendar_str).strip() == "gregorian" else ONE_YEAR_DAYS / 365.2425
    ecc = jnp.asarray(eccentricity, dtype=latitude.dtype)
    perh = jnp.asarray(perihelie, dtype=latitude.dtype)
    xob = jnp.asarray(obliquity, dtype=latitude.dtype)
    jday = jnp.asarray(julian_diff, dtype=latitude.dtype)

    xl = perh + 180.0
    so = jnp.sin(xob * pir)
    xllp = xl * pir
    xee = ecc * ecc
    xse = jnp.sqrt(1.0 - xee)
    xlam = (ecc / 2.0 + ecc * xee / 8.0) * (1.0 + xse) * jnp.sin(xllp)
    xlam = xlam - xee / 4.0 * (0.5 + xse) * jnp.sin(2.0 * xllp)
    xlam = xlam + ecc * xee / 8.0 * (1.0 / 3.0 + xse) * jnp.sin(3.0 * xllp)
    xlam = 2.0 * xlam / pir
    dlamm = xlam + (jnp.floor(jday).astype(jnp.int32) - 79) * step
    anm = dlamm - xl
    ranm = anm * pir
    xee3 = xee * ecc
    ranv = ranm + (2.0 * ecc - xee3 / 4.0) * jnp.sin(ranm)
    ranv = ranv + 5.0 / 4.0 * ecc * ecc * jnp.sin(2.0 * ranm)
    ranv = ranv + 13.0 / 12.0 * xee3 * jnp.sin(3.0 * ranm)
    tls = ranv / pir + xl
    rlam = tls * pir
    sd = so * jnp.sin(rlam)
    cd = jnp.sqrt(1.0 - sd * sd)
    deltar = jnp.arctan(sd / cd)
    dis_st = (1.0 - ecc * ecc) / (1.0 + ecc * jnp.cos(ranv))
    ddt = 1.0 / dis_st
    sw = ddt * ddt * 1370.0
    angle = 2.0 * jnp.pi * (jnp.asarray(rtime, dtype=latitude.dtype) - 12.0) / 24.0
    xlat = latitude * jnp.pi / 180.0
    coszen = jnp.maximum(
        0.0,
        jnp.sin(xlat) * jnp.sin(deltar)
        + jnp.cos(xlat) * jnp.cos(deltar) * jnp.cos(angle),
    )
    trans = 0.251 + 0.509 * (1.0 - cloud_arr)
    fdiffuse = 1.0045 + trans * (0.0435 + trans * (-3.5227 + trans * 2.6313))
    fdiffuse = jnp.where(trans > 0.75, 0.166, fdiffuse)
    return sw * coszen * trans * (1.0 - fdiffuse)


def season_update_gdd_init_date(
    gdd_init_date, *, latitude, julian_diff, calendar_str="gregorian"
) -> jnp.ndarray:
    """Update ``gdd_init_date`` from the season midwinter solar diagnostic."""

    current = jnp.asarray(gdd_init_date)
    solad = season_midwinter_solad(latitude, julian_diff, calendar_str=calendar_str)
    return jnp.where(
        (solad < current[:, 1])[:, None],
        jnp.stack(
            [jnp.full_like(solad, jnp.asarray(julian_diff, dtype=solad.dtype)), solad],
            axis=1,
        ),
        current,
    )


def _relax(previous, daily, tau, dt):
    previous = jnp.asarray(previous)
    daily = jnp.asarray(daily)
    if previous.shape != daily.shape:
        raise ValueError(
            f"season relaxation shape mismatch: previous {previous.shape} versus daily {daily.shape}"
        )
    return (previous * (jnp.asarray(tau, dtype=previous.dtype) - dt) + daily * dt) / tau


def _zero_small_except_bare(values):
    arr = jnp.asarray(values)
    pft_index = jnp.arange(arr.shape[1])
    return jnp.where(
        (pft_index[None, :] > 0) & (jnp.abs(arr) < FORTRAN_EPSILON_R4), 0.0, arr
    )


def _zero_small(values):
    arr = jnp.asarray(values)
    return jnp.where(jnp.abs(arr) < FORTRAN_EPSILON_R4, 0.0, arr)


def _initialise_firstcall(
    state: SeasonMemoryState,
    *,
    moiavail_daily,
    t2m_daily,
    tsoil_daily,
    soilhum_daily,
    min_stomate=MIN_STOMATE,
) -> SeasonMemoryState:
    """Apply first-call meteorological initialization from ``season``.

    Fortran provenance: ``stomate_season.f90`` lines 386-473. The checks ignore
    bare soil for PFT fields by summing columns 2:nvm, matching Fortran's
    one-based ``(:,2:nvm)`` slices.
    """

    moiavail_daily = jnp.asarray(moiavail_daily)
    t2m_daily = jnp.asarray(t2m_daily)
    tsoil_daily = jnp.asarray(tsoil_daily)
    soilhum_daily = jnp.asarray(soilhum_daily)

    moiavail_month = jnp.where(
        jnp.abs(jnp.sum(state.moiavail_month[:, 1:])) < min_stomate,
        moiavail_daily,
        state.moiavail_month,
    )
    moiavail_week = jnp.where(
        jnp.abs(jnp.sum(state.moiavail_week[:, 1:])) < min_stomate,
        moiavail_daily,
        state.moiavail_week,
    )
    t2m_month = jnp.where(
        jnp.abs(jnp.sum(state.t2m_month)) < min_stomate, t2m_daily, state.t2m_month
    )
    tseason = jnp.where(
        jnp.abs(jnp.sum(state.tseason)) < min_stomate, t2m_daily, state.tseason
    )
    t2m_week = jnp.where(
        jnp.abs(jnp.sum(state.t2m_week)) < min_stomate, t2m_daily, state.t2m_week
    )
    tsoil_month = jnp.where(
        jnp.abs(jnp.sum(state.tsoil_month)) < min_stomate,
        tsoil_daily,
        state.tsoil_month,
    )
    soilhum_month = jnp.where(
        jnp.abs(jnp.sum(state.soilhum_month)) < min_stomate,
        soilhum_daily,
        state.soilhum_month,
    )

    return state._replace(
        moiavail_month=moiavail_month,
        moiavail_week=moiavail_week,
        t2m_month=t2m_month,
        tseason=tseason,
        t2m_week=t2m_week,
        tsoil_month=tsoil_month,
        soilhum_month=soilhum_month,
    )


def season_memory_step(
    state: SeasonMemoryState,
    *,
    dt_days,
    end_of_year,
    moiavail_daily,
    t2m_daily,
    tsurf_daily=None,
    tsoil_daily,
    soilhum_daily,
    begin_leaves,
    julian_diff,
    when_growthinit,
    natural,
    leaf_tab,
    pheno_type,
    pft_to_mtc,
    time_scales: SeasonTimeScales | None = None,
    firstcall=False,
    zero_celsius=ZERO_CELSIUS,
    undef=UNDEF,
    min_stomate=MIN_STOMATE,
) -> SeasonMemoryStepResult:
    """Advance the closed local part of ``stomate_season::season`` by one step.

    Fortran provenance is listed in ``SEASON_MEMORY_PROVENANCE``. ``dt_days`` is
    the STOMATE time step in days because ``stomate_main`` passes ``dt_days`` to
    ``season`` at lines 4225-4226.
    """

    scales = time_scales or SeasonTimeScales()
    dt = jnp.asarray(dt_days, dtype=jnp.asarray(state.t2m_month).dtype)
    current = state
    if firstcall:
        current = _initialise_firstcall(
            current,
            moiavail_daily=moiavail_daily,
            t2m_daily=t2m_daily,
            tsoil_daily=tsoil_daily,
            soilhum_daily=soilhum_daily,
            min_stomate=min_stomate,
        )

    tsurf_year = current.tsurf_year
    t2m_14 = current.t2m_14
    if (tsurf_year is None) != (t2m_14 is None):
        raise ValueError(
            "tsurf_year and t2m_14 must both be present or both be omitted"
        )
    if tsurf_year is not None:
        if tsurf_daily is None:
            raise ValueError(
                "tsurf_daily is required when season surface-temperature state is present"
            )
        surface_temperature = season_surface_temperature_memory_step(
            tsurf_year=tsurf_year,
            t2m_14=t2m_14,
            tsurf_daily=tsurf_daily,
            t2m_daily=t2m_daily,
            dt_days=dt_days,
            firstcall=firstcall,
        )
        tsurf_year = surface_temperature.tsurf_year
        t2m_14 = surface_temperature.t2m_14

    moiavail_month = _zero_small_except_bare(
        _relax(current.moiavail_month, moiavail_daily, scales.tau_hum_month, dt)
    )
    moiavail_week = _zero_small_except_bare(
        _relax(current.moiavail_week, moiavail_daily, scales.tau_hum_week, dt)
    )

    tau_longterm = jnp.minimum(
        jnp.asarray(current.tau_longterm) + jnp.asarray(dt_days),
        scales.tau_longterm_max,
    )
    t2m_longterm = _relax(current.t2m_longterm, t2m_daily, tau_longterm, dt)
    t2m_longterm = jnp.maximum(
        scales.tlong_ref_min, jnp.minimum(scales.tlong_ref_max, t2m_longterm)
    )

    t2m_month = _zero_small(
        _relax(current.t2m_month, t2m_daily, scales.tau_t2m_month, dt)
    )

    begin = jnp.asarray(begin_leaves, dtype=bool)
    onset_date = jnp.where(begin, julian_diff, 0.0)
    old_t2m_week = jnp.asarray(current.t2m_week)
    warm_week = old_t2m_week > zero_celsius
    tseason_tmp = jnp.where(
        warm_week, current.tseason_tmp + old_t2m_week, current.tseason_tmp
    )
    tseason_length = jnp.where(
        warm_week, current.tseason_length + dt, current.tseason_length
    )
    tseason_tmp = jnp.where(warm_week, tseason_tmp + old_t2m_week, tseason_tmp)
    tseason_length = jnp.where(warm_week, tseason_length + dt, tseason_length)
    tseason_tmp = _zero_small(tseason_tmp)
    tseason_length = _zero_small(tseason_length)

    natural_arr = jnp.asarray(natural, dtype=bool)
    pft_to_mtc_arr = jnp.asarray(pft_to_mtc)
    leaf_tab_arr = jnp.asarray(leaf_tab)
    pheno_type_arr = jnp.asarray(pheno_type)
    tmin_running_first = jnp.where(
        (current.tmin_spring_time > 0.0) & (current.tmin_spring_time < SPRING_DAYS_MAX),
        current.tmin_spring_time + 1.0,
        0.0,
    )
    broad_summer = (leaf_tab_arr == 1) & (pheno_type_arr == 2)
    tmin_spring_time = jnp.where(
        broad_summer[None, :],
        jnp.where(begin, 1.0, tmin_running_first),
        current.tmin_spring_time,
    )
    tmin_running_second = jnp.where(
        (tmin_spring_time > 0.0) & (tmin_spring_time < SPRING_DAYS_MAX),
        tmin_spring_time + 1.0,
        0.0,
    )
    mtc_legacy_spring = ((pft_to_mtc_arr == 8) | (pft_to_mtc_arr == 6)) & (
        jnp.arange(natural_arr.shape[0]) > 0
    )
    tmin_spring_time = jnp.where(
        mtc_legacy_spring[None, :],
        jnp.where(begin, 1.0, tmin_running_second),
        tmin_spring_time,
    )

    t2m_week = _zero_small(_relax(current.t2m_week, t2m_daily, scales.tau_t2m_week, dt))
    tsoil_month = _zero_small(
        _relax(current.tsoil_month, tsoil_daily, scales.tau_tsoil_month, dt)
    )
    soilhum_month = _zero_small(
        _relax(current.soilhum_month, soilhum_daily, scales.tau_soilhum_month, dt)
    )

    gdd_from_growthinit = jnp.asarray(current.gdd_from_growthinit)
    nonnatural = ~natural_arr
    reset_growth = jnp.asarray(when_growthinit) == 0.0
    warm_daily = jnp.asarray(t2m_daily) > zero_celsius
    gdd_reset = jnp.where(reset_growth, 0.0, gdd_from_growthinit)
    gdd_added = jnp.where(
        warm_daily[:, None],
        gdd_reset + dt * (jnp.asarray(t2m_daily)[:, None] - zero_celsius),
        gdd_reset,
    )
    pft_loop_mask = jnp.arange(natural_arr.shape[0]) > 0
    gdd_from_growthinit = jnp.where(
        pft_loop_mask[None, :] & nonnatural[None, :], gdd_added, gdd_from_growthinit
    )
    gdd_from_growthinit = jnp.where(
        pft_loop_mask[None, :] & natural_arr[None, :], undef, gdd_from_growthinit
    )

    end_of_year_arr = jnp.asarray(end_of_year, dtype=bool)
    tseason = jnp.where(
        end_of_year_arr,
        jnp.where(tseason_length > 0.0, tseason_tmp / tseason_length, 0.0),
        jnp.asarray(current.tseason),
    )
    tseason_tmp = jnp.where(
        end_of_year_arr, jnp.zeros_like(tseason_tmp), tseason_tmp
    )
    tseason_length = jnp.where(
        end_of_year_arr, jnp.zeros_like(tseason_length), tseason_length
    )
    tmin_spring_time = jnp.where(
        end_of_year_arr, jnp.zeros_like(tmin_spring_time), tmin_spring_time
    )
    onset_date = jnp.where(
        end_of_year_arr, jnp.zeros_like(onset_date), onset_date
    )

    return SeasonMemoryStepResult(
        state=SeasonMemoryState(
            moiavail_month=moiavail_month,
            moiavail_week=moiavail_week,
            t2m_longterm=t2m_longterm,
            tau_longterm=tau_longterm,
            t2m_month=t2m_month,
            t2m_week=t2m_week,
            tseason=tseason,
            tseason_length=tseason_length,
            tseason_tmp=tseason_tmp,
            tmin_spring_time=tmin_spring_time,
            onset_date=onset_date,
            tsoil_month=tsoil_month,
            soilhum_month=soilhum_month,
            gdd_from_growthinit=gdd_from_growthinit,
            tsurf_year=tsurf_year,
            t2m_14=t2m_14,
        )
    )


def _zero_small_except_bare_nd(values):
    arr = jnp.asarray(values)
    pft_index = jnp.arange(arr.shape[1])
    mask_shape = (1, arr.shape[1]) + (1,) * (arr.ndim - 2)
    return jnp.where(
        (pft_index.reshape(mask_shape) > 0) & (jnp.abs(arr) < FORTRAN_EPSILON_R4),
        0.0,
        arr,
    )


def season_annual_step(
    state: SeasonAnnualState,
    *,
    dt_days,
    tau_longterm,
    end_of_year,
    firstcall=False,
    veget,
    veget_max,
    moiavail_daily,
    t2m_daily,
    precip_daily,
    biomass,
    npp_daily,
    turnover_daily,
    gpp_daily,
    natural,
    pasture,
    leaflife_tab,
    pheno_model=None,
    hvc1=0.019,
    hvc2=1.38,
    leaf_frac_hvc=0.33,
    green_age_ever=2.0,
    green_age_dec=0.5,
    ok_stomate=True,
    ok_dgvm=False,
    time_scales: SeasonTimeScales | None = None,
    zero_celsius=ZERO_CELSIUS,
    one_year=ONE_YEAR_DAYS,
    large_value=1.0e33,
    min_stomate=MIN_STOMATE,
) -> SeasonAnnualStepResult:
    """Advance local annual and long-term season statistics.

    This implements the source-closed annual statistics branch used by the
    paper case plus the local ``ok_dgvm=.TRUE.`` annual maxfpc/lm relaxation.
    Fortran provenance is listed on ``SeasonAnnualStepResult``.
    """

    scales = time_scales or SeasonTimeScales()
    dt = jnp.asarray(dt_days, dtype=jnp.asarray(state.npp_longterm).dtype)
    tau_longterm_arr = jnp.asarray(
        tau_longterm, dtype=jnp.asarray(state.npp_longterm).dtype
    )

    npp_longterm = _zero_small_except_bare(
        (
            jnp.asarray(state.npp_longterm) * (tau_longterm_arr - dt)
            + jnp.asarray(npp_daily) * one_year * dt
        )
        / tau_longterm_arr
    )
    turnover_longterm = _zero_small_except_bare_nd(
        (
            jnp.asarray(state.turnover_longterm) * (tau_longterm_arr - dt)
            + jnp.asarray(turnover_daily) * one_year * dt
        )
        / tau_longterm_arr
    )

    veget_max_arr = jnp.asarray(veget_max)
    gpp_week_candidate = (
        jnp.asarray(state.gpp_week) * (scales.tau_gpp_week - dt)
        + jnp.asarray(gpp_daily) * dt
    ) / scales.tau_gpp_week
    gpp_week = jnp.where(veget_max_arr > 0.0, gpp_week_candidate, 0.0)
    gpp_week = _zero_small_except_bare(gpp_week)

    minmoiavail_thisyear_initial = jnp.asarray(state.minmoiavail_thisyear)
    if bool(firstcall):
        minmoiavail_thisyear_initial = jnp.where(
            jnp.abs(jnp.sum(minmoiavail_thisyear_initial[:, 1:])) < min_stomate,
            jnp.full_like(minmoiavail_thisyear_initial, large_value),
            minmoiavail_thisyear_initial,
        )
    maxmoiavail_thisyear = jnp.where(
        jnp.asarray(moiavail_daily) > state.maxmoiavail_thisyear,
        moiavail_daily,
        state.maxmoiavail_thisyear,
    )
    minmoiavail_thisyear = jnp.where(
        jnp.asarray(moiavail_daily) < minmoiavail_thisyear_initial,
        moiavail_daily,
        minmoiavail_thisyear_initial,
    )
    maxgppweek_thisyear = jnp.where(
        gpp_week > state.maxgppweek_thisyear, gpp_week, state.maxgppweek_thisyear
    )
    gdd0_thisyear = jnp.where(
        jnp.asarray(t2m_daily) > zero_celsius,
        state.gdd0_thisyear + dt * (jnp.asarray(t2m_daily) - zero_celsius),
        state.gdd0_thisyear,
    )
    precip_thisyear = state.precip_thisyear + dt * jnp.asarray(precip_daily)

    natural_arr = jnp.asarray(natural, dtype=bool)
    pasture_arr = jnp.asarray(pasture, dtype=bool)
    lm_thisyearmax = jnp.asarray(state.lm_thisyearmax)
    maxfpc_lastyear = jnp.asarray(state.maxfpc_lastyear)
    maxfpc_thisyear = jnp.asarray(state.maxfpc_thisyear)
    pft_loop_mask = jnp.arange(lm_thisyearmax.shape[1]) > 0
    leaf_biomass = jnp.asarray(biomass)[:, :, 0, 0]
    if bool(ok_stomate):
        if bool(ok_dgvm):
            nonnatural_or_pasture = pft_loop_mask & ((~natural_arr) | pasture_arr)
            fracnat = 1.0 - jnp.sum(
                jnp.where(nonnatural_or_pasture[None, :], veget_max_arr, 0.0), axis=1
            )
            leaflife = jnp.asarray(leaflife_tab, dtype=lm_thisyearmax.dtype)
            tau_leaf = one_year / leaflife
            natural_nonpasture = pft_loop_mask & natural_arr & (~pasture_arr)
            safe_fracnat = jnp.where(fracnat > min_stomate, fracnat, 1.0)
            maxfpc_relaxed = (
                maxfpc_lastyear * (tau_leaf[None, :] - dt)
                + jnp.asarray(veget) / safe_fracnat[:, None] * dt
            ) / tau_leaf[None, :]
            maxfpc_update_mask = (
                natural_nonpasture[None, :]
                & (fracnat[:, None] > min_stomate)
                & (leaf_biomass > jnp.asarray(state.lm_lastyearmax) * 0.75)
            )
            maxfpc_lastyear = jnp.where(
                maxfpc_update_mask, maxfpc_relaxed, maxfpc_lastyear
            )
            maxfpc_thisyear = jnp.where(
                natural_nonpasture[None, :], maxfpc_lastyear, maxfpc_thisyear
            )
            lm_relaxed = (
                lm_thisyearmax * (tau_leaf[None, :] - dt) + leaf_biomass * dt
            ) / tau_leaf[None, :]
            lm_update_mask = (
                pft_loop_mask[None, :]
                & (lm_thisyearmax > min_stomate)
                & (leaf_biomass > lm_thisyearmax * 0.75)
            )
            lm_reset_mask = pft_loop_mask[None, :] & (lm_thisyearmax <= min_stomate)
            lm_thisyearmax = jnp.where(lm_update_mask, lm_relaxed, lm_thisyearmax)
            lm_thisyearmax = jnp.where(lm_reset_mask, leaf_biomass, lm_thisyearmax)
        else:
            lm_thisyearmax = jnp.where(
                pft_loop_mask[None, :] & (leaf_biomass > lm_thisyearmax),
                leaf_biomass,
                lm_thisyearmax,
            )
    else:
        raise NotImplementedError(
            "season_annual_step requires explicit sla_calc/lai_max before closing ok_stomate=false"
        )

    maxfpc_thisyear = jnp.where(
        jnp.asarray(veget) > maxfpc_thisyear, veget, maxfpc_thisyear
    )

    maxmoiavail_lastyear = jnp.asarray(state.maxmoiavail_lastyear)
    minmoiavail_lastyear = jnp.asarray(state.minmoiavail_lastyear)
    maxgppweek_lastyear = jnp.asarray(state.maxgppweek_lastyear)
    gdd0_lastyear = jnp.asarray(state.gdd0_lastyear)
    precip_lastyear = jnp.asarray(state.precip_lastyear)
    lm_lastyearmax = jnp.asarray(state.lm_lastyearmax)

    natural_arr = jnp.asarray(natural, dtype=bool)
    end_of_year_arr = jnp.asarray(end_of_year, dtype=bool)
    tau_clim = jnp.asarray(scales.tau_climatology, dtype=maxmoiavail_lastyear.dtype)
    maxmoiavail_lastyear = jnp.where(
        end_of_year_arr,
        (maxmoiavail_lastyear * (tau_clim - 1.0) + maxmoiavail_thisyear) / tau_clim,
        maxmoiavail_lastyear,
    )
    minmoiavail_lastyear = jnp.where(
        end_of_year_arr,
        (minmoiavail_lastyear * (tau_clim - 1.0) + minmoiavail_thisyear) / tau_clim,
        minmoiavail_lastyear,
    )
    maxgppweek_lastyear = jnp.where(
        end_of_year_arr,
        (maxgppweek_lastyear * (tau_clim - 1.0) + maxgppweek_thisyear) / tau_clim,
        maxgppweek_lastyear,
    )
    gdd0_lastyear = jnp.where(end_of_year_arr, gdd0_thisyear, gdd0_lastyear)
    precip_lastyear = jnp.where(
        end_of_year_arr, precip_thisyear, precip_lastyear
    )
    lm_lastyearmax = jnp.where(
        end_of_year_arr, lm_thisyearmax, lm_lastyearmax
    )
    maxfpc_lastyear = jnp.where(
        end_of_year_arr, maxfpc_thisyear, maxfpc_lastyear
    )
    nonnatural_or_pasture = (jnp.arange(natural_arr.shape[0]) > 0) & (
        (~natural_arr) | pasture_arr
    )
    maxfpc_lastyear = jnp.where(
        end_of_year_arr & nonnatural_or_pasture[None, :], 0.0, maxfpc_lastyear
    )
    maxmoiavail_thisyear = jnp.where(
        end_of_year_arr, jnp.zeros_like(maxmoiavail_thisyear), maxmoiavail_thisyear
    )
    minmoiavail_thisyear = jnp.where(
        end_of_year_arr,
        jnp.ones_like(minmoiavail_thisyear) * large_value,
        minmoiavail_thisyear,
    )
    maxgppweek_thisyear = jnp.where(
        end_of_year_arr, jnp.zeros_like(maxgppweek_thisyear), maxgppweek_thisyear
    )
    gdd0_thisyear = jnp.where(end_of_year_arr, jnp.zeros_like(gdd0_thisyear), gdd0_thisyear)
    precip_thisyear = jnp.where(
        end_of_year_arr, jnp.zeros_like(precip_thisyear), precip_thisyear
    )
    lm_thisyearmax = jnp.where(
        end_of_year_arr, jnp.zeros_like(lm_thisyearmax), lm_thisyearmax
    )
    maxfpc_thisyear = jnp.where(
        end_of_year_arr, jnp.zeros_like(maxfpc_thisyear), maxfpc_thisyear
    )

    if pheno_model is None:
        pheno_none = jnp.zeros_like(natural_arr, dtype=bool)
    else:
        pheno_none = jnp.asarray(
            [str(model).strip().lower() == "none" for model in pheno_model], dtype=bool
        )
    nlflong_nat = jnp.where(natural_arr[None, :], npp_longterm * leaf_frac_hvc, 0.0)
    green_age_base = jnp.where(pheno_none, green_age_ever, green_age_dec)
    green_age = jnp.where(
        natural_arr[None, :], green_age_base[None, :] * lm_lastyearmax, 0.0
    )
    green_age = jnp.where(lm_lastyearmax > 0.0, green_age / lm_lastyearmax, 1.0)
    consumption = hvc1 * (nlflong_nat**hvc2)
    herbivores = jnp.where(
        natural_arr[None, :] & (nlflong_nat > 0.0),
        one_year * green_age * nlflong_nat / consumption,
        100000.0,
    )
    herbivores = herbivores.at[:, 0].set(0.0)

    return SeasonAnnualStepResult(
        state=SeasonAnnualState(
            npp_longterm=npp_longterm,
            turnover_longterm=turnover_longterm,
            gpp_week=gpp_week,
            maxmoiavail_lastyear=maxmoiavail_lastyear,
            maxmoiavail_thisyear=maxmoiavail_thisyear,
            minmoiavail_lastyear=minmoiavail_lastyear,
            minmoiavail_thisyear=minmoiavail_thisyear,
            maxgppweek_lastyear=maxgppweek_lastyear,
            maxgppweek_thisyear=maxgppweek_thisyear,
            gdd0_lastyear=gdd0_lastyear,
            gdd0_thisyear=gdd0_thisyear,
            precip_lastyear=precip_lastyear,
            precip_thisyear=precip_thisyear,
            lm_lastyearmax=lm_lastyearmax,
            lm_thisyearmax=lm_thisyearmax,
            maxfpc_lastyear=maxfpc_lastyear,
            maxfpc_thisyear=maxfpc_thisyear,
        ),
        herbivores=herbivores,
    )


def season_biometeorology_step(
    state: SeasonBiometeorologyState,
    *,
    dt_days,
    julian_diff,
    t2m_daily,
    t2m_month,
    t2m_week,
    t2m_longterm,
    moiavail_month,
    when_growthinit,
    gdd_init_date,
    pheno_gdd_crit,
    ncdgdd_temp,
    hum_min_time,
    firstcall=False,
    time_scales: SeasonTimeScales | None = None,
    zero_celsius=ZERO_CELSIUS,
    undef=UNDEF,
    gdd_threshold=5.0,
    ncd_max=ONE_YEAR_DAYS * 3.0,
    min_stomate=MIN_STOMATE,
) -> SeasonBiometeorologyStepResult:
    """Advance ``stomate_season`` sections 7-11.

    This uses only explicit state and PFT parameter arrays. It does not compute
    phenology onset; it updates the biometeorological memories phenology later
    consumes.
    """

    scales = time_scales or SeasonTimeScales()
    dt = jnp.asarray(dt_days, dtype=jnp.asarray(state.gdd_m5_dormance).dtype)
    pft_loop_mask = jnp.arange(jnp.asarray(state.gdd_m5_dormance).shape[1]) > 0
    t2m_daily_arr = jnp.asarray(t2m_daily)
    gdd_init_today = jnp.asarray(gdd_init_date)[:, 0] == julian_diff

    gdd_m5 = jnp.asarray(state.gdd_m5_dormance)
    gdd_mid_initial = jnp.asarray(state.gdd_midwinter)
    ncd_initial = jnp.asarray(state.ncd_dormance)
    if bool(firstcall):
        gdd_m5 = jnp.where(
            jnp.abs(jnp.sum(gdd_m5[:, 1:])) < min_stomate,
            jnp.full_like(gdd_m5, undef),
            gdd_m5,
        )
        gdd_mid_initial = jnp.where(
            jnp.abs(jnp.sum(gdd_mid_initial[:, 1:])) < min_stomate,
            jnp.full_like(gdd_mid_initial, undef),
            gdd_mid_initial,
        )
        ncd_initial = jnp.where(
            jnp.abs(jnp.sum(ncd_initial[:, 1:])) < min_stomate,
            jnp.full_like(ncd_initial, undef),
            ncd_initial,
        )
    pheno_gdd_defined = jnp.all(jnp.asarray(pheno_gdd_crit) != undef, axis=1)
    gdd_m5_update = jnp.where(gdd_init_today[:, None], 0.0, gdd_m5)
    gdd_m5_update = jnp.where(jnp.asarray(when_growthinit) == 0.0, undef, gdd_m5_update)
    add_gdd_m5 = (t2m_daily_arr > (zero_celsius - gdd_threshold))[:, None] & (
        gdd_m5_update != undef
    )
    gdd_m5_update = jnp.where(
        add_gdd_m5,
        gdd_m5_update + dt * (t2m_daily_arr[:, None] - (zero_celsius - gdd_threshold)),
        gdd_m5_update,
    )
    gdd_m5_update = jnp.where(
        gdd_m5_update != undef,
        gdd_m5_update * (scales.tau_gdd - dt) / scales.tau_gdd,
        gdd_m5_update,
    )
    gdd_m5 = jnp.where(
        pft_loop_mask[None, :] & pheno_gdd_defined[None, :], gdd_m5_update, gdd_m5
    )
    gdd_m5 = _zero_small_except_bare(gdd_m5)

    ncd_temp = jnp.asarray(ncdgdd_temp)
    ncd_defined = ncd_temp != undef
    gdd_mid = gdd_mid_initial
    gdd_mid_update = jnp.where(gdd_init_today[:, None], 0.0, gdd_mid)
    midsummer = (jnp.asarray(t2m_month) > jnp.asarray(t2m_week)) & (
        jnp.asarray(t2m_month) > jnp.asarray(t2m_longterm)
    )
    gdd_mid_update = jnp.where(midsummer[:, None], undef, gdd_mid_update)
    add_mid = (gdd_mid_update != undef) & (
        t2m_daily_arr[:, None] > (ncd_temp[None, :] + zero_celsius)
    )
    gdd_mid_update = jnp.where(
        add_mid,
        gdd_mid_update
        + dt * (t2m_daily_arr[:, None] - (ncd_temp[None, :] + zero_celsius)),
        gdd_mid_update,
    )
    gdd_mid = jnp.where(
        pft_loop_mask[None, :] & ncd_defined[None, :], gdd_mid_update, gdd_mid
    )

    ncd = ncd_initial
    ncd_update = jnp.where(gdd_init_today[:, None], 0.0, ncd)
    ncd_update = jnp.where(jnp.asarray(when_growthinit) == 0.0, undef, ncd_update)
    add_ncd = (ncd_update != undef) & (
        t2m_daily_arr[:, None] <= (ncd_temp[None, :] + zero_celsius)
    )
    ncd_update = jnp.where(add_ncd, jnp.minimum(ncd_update + dt, ncd_max), ncd_update)
    ncd = jnp.where(pft_loop_mask[None, :] & ncd_defined[None, :], ncd_update, ncd)

    ngd = jnp.asarray(state.ngd_minus5)
    ngd_update = jnp.where(gdd_init_today[:, None], 0.0, ngd)
    add_ngd = t2m_daily_arr > (zero_celsius - gdd_threshold)
    ngd_update = jnp.where(add_ngd[:, None], ngd_update + dt, ngd_update)
    ngd_update = ngd_update * (scales.tau_ngd - dt) / scales.tau_ngd
    ngd = jnp.where(pft_loop_mask[None, :], ngd_update, ngd)
    ngd = _zero_small_except_bare(ngd)

    hum_defined = jnp.asarray(hum_min_time) != undef
    time_hum = jnp.asarray(state.time_hum_min)
    hum_min = jnp.asarray(state.hum_min_dormance)
    reset_hum = jnp.asarray(when_growthinit) == 0.0
    time_update = jnp.where(reset_hum, 0.0, time_hum)
    hum_update = jnp.where(reset_hum, moiavail_month, hum_min)
    time_update = jnp.where(hum_update != undef, time_update + dt, time_update)
    new_min = (hum_update != undef) & (jnp.asarray(moiavail_month) <= hum_update)
    hum_update = jnp.where(new_min, moiavail_month, hum_update)
    time_update = jnp.where(new_min, 0.0, time_update)
    time_hum = jnp.where(
        pft_loop_mask[None, :] & hum_defined[None, :], time_update, time_hum
    )
    hum_min = jnp.where(
        pft_loop_mask[None, :] & hum_defined[None, :], hum_update, hum_min
    )

    return SeasonBiometeorologyStepResult(
        state=SeasonBiometeorologyState(
            gdd_m5_dormance=gdd_m5,
            gdd_midwinter=gdd_mid,
            ncd_dormance=ncd,
            ngd_minus5=ngd,
            time_hum_min=time_hum,
            hum_min_dormance=hum_min,
        )
    )
