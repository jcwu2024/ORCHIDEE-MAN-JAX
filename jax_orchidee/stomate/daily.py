"""Daily accumulation and process-boundary helpers for STOMATE Phase 1D.

This module contains closed scheduling algebra only. It does not run
phenology, allocation, NPP, turnover, or the full STOMATE loop.
"""

from __future__ import annotations

from typing import NamedTuple

from jax import config, jit

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jax_orchidee.stomate.carbon_kernels import (
    DEFAULT_MAINT_RESP_COEFF,
    DEFAULT_MAINT_RESP_MIN_VMAX,
    MaintenanceRespirationResult,
    _maintenance_respiration_core,
    maintenance_respiration,
    maintenance_respiration_core_jit,
)
from jax_orchidee.stomate.daily_inputs import MIN_STOMATE, stomate_gpp_daily_increment_from_veget_cov_max


ZERO_CELSIUS = 273.15

REQUIRED_CARBON_TRACE_FIELDS = (
    "f_alloc",
    "bm_alloc",
    "biomass_before_alloc",
    "biomass_after_alloc",
    "biomass_before_npp",
    "biomass_after_npp",
    "resp_maint_part",
    "resp_growth",
    "npp_daily",
    "agr_pools_before",
    "agr_pools_after",
)


class DailyCarbonBoundary(NamedTuple):
    """Explicit inputs available at the maint/alloc/npp scheduling boundary."""

    gpp_daily: jnp.ndarray
    biomass: jnp.ndarray
    resp_maint_part: jnp.ndarray
    resp_maint: jnp.ndarray | None
    pft_present: jnp.ndarray | None
    f_alloc: jnp.ndarray | None
    requires_trace: tuple[str, ...]


class StomateInstantDailyPrep(NamedTuple):
    """Instantaneous values feeding STOMATE daily accumulators."""

    end_of_month: bool
    st2m: jnp.ndarray
    ugdh: jnp.ndarray | None
    uphoi: jnp.ndarray | None


class StomateFirstStepDailyAccumulation(NamedTuple):
    """Closed first-step ``stomate_main`` section 4.1/4.2 daily updates."""

    local_prep: object | None
    fields: dict[str, jnp.ndarray]
    missing_inputs: tuple[str, ...]
    provenance: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_inputs and self.local_prep is not None


class StomateDailyAccumulationFold(NamedTuple):
    """Sequential daily accumulator state across explicit STOMATE entries."""

    fields: dict[str, jnp.ndarray]
    step_results: tuple[StomateFirstStepDailyAccumulation, ...]
    missing_inputs: tuple[str, ...]
    failed_step: int | None
    provenance: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_inputs and self.failed_step is None


class StomateMaintenanceFold(NamedTuple):
    """Sequential maintenance-respiration state across explicit entries."""

    resp_maint_part: jnp.ndarray
    step_results: tuple[MaintenanceRespirationResult, ...]
    resp_maint_radia: jnp.ndarray | None
    flood_root_radia: jnp.ndarray | None
    missing_inputs: tuple[str, ...]
    failed_step: int | None
    provenance: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_inputs and self.failed_step is None


class StomateMaintenanceStaticCache(NamedTuple):
    """Maintenance respiration terms fixed through one daily fold."""

    biomass: jnp.ndarray
    biomass_carbon: jnp.ndarray
    z_soil: jnp.ndarray
    rprof: jnp.ndarray
    root_weights: jnp.ndarray
    lai: jnp.ndarray
    leaf_factor: jnp.ndarray
    coeff_maint_zero: jnp.ndarray
    slope: jnp.ndarray
    ext_coeff: jnp.ndarray
    is_tree: jnp.ndarray
    dt_days: jnp.ndarray


class StomateDailyProcessFold(NamedTuple):
    """Combined daily accumulator and maintenance boundary for do_slow."""

    accumulator: StomateDailyAccumulationFold
    maintenance: StomateMaintenanceFold
    daily_fields: dict[str, jnp.ndarray]
    missing_inputs: tuple[str, ...]
    provenance: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.accumulator.ok and self.maintenance.ok and not self.missing_inputs


class StomateCforcingSlotUpdate(NamedTuple):
    """Result of one source-order Cforcing slot accumulation."""

    fields: dict[str, jnp.ndarray]
    nforce: jnp.ndarray
    iatt: int
    reset_window: bool
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 4585-4726",
    )


class _AccumulatorStateView:
    """Expose updated fold fields through the restart-state attribute names."""

    def __init__(self, base, fields: dict[str, jnp.ndarray]):
        self._base = base
        self._fields = fields

    def __getattr__(self, name: str):
        if name in self._fields:
            return self._fields[name]
        return getattr(self._base, name)


def stomate_accumulate_daily(current, increment, do_slow, *, dt_sechiba, dt_stomate):
    """Accumulate a STOMATE daily field with Fortran `stomate_accu` semantics.

    Fortran provenance: `src_stomate/stomate.f90`, generic calls in
    `stomate_main` lines 3198-3208 and `stomate_accu_r1d/r2d/r3d` lines
    9341-9411. Formula: `field_out += field_in * dt_sechiba`; if `ldmean`
    (`do_slow`) is true, divide accumulated `field_out` by `dt_stomate`.

    `dt_sechiba` and `dt_stomate` must use the same time unit, matching the
    Fortran seconds-based call sites.
    """

    current = jnp.asarray(current)
    increment = jnp.asarray(increment)
    accumulated = current + increment * dt_sechiba
    if isinstance(do_slow, (bool, np.bool_)):
        return accumulated / dt_stomate if bool(do_slow) else accumulated
    return jnp.where(do_slow, accumulated / dt_stomate, accumulated)


def stomate_end_of_month(day, sec, *, dt_sechiba):
    """Return Fortran's first-step-of-month ``EndOfMonth`` flag.

    Fortran provenance: ``src_stomate/stomate.f90``, subroutine
    ``stomate_main``, lines 3128-3134:
    ``EndOfMonth = day == 1 .AND. sec < dt_sechiba``.
    """

    return bool(int(day) == 1 and float(sec) < float(dt_sechiba))


def stomate_cforcing_slot_index(date, *, one_year=365.0, nbyear=1, dt_forcesoil=1.0, nslots=None) -> int:
    """Return the zero-based Cforcing slot used by Fortran ``iatt``.

    Fortran provenance: ``src_stomate/stomate.f90::stomate_main`` lines
    4592-4594 and 4682-4683 compute
    ``iatt = FLOOR(MODULO(date-1, one_year*nbyear)/dt_forcesoil) + 1``.
    This helper returns ``iatt - 1`` so callers can index Python/JAX arrays.
    """

    slot = int(float(date) - 1.0)
    period = float(one_year) * float(nbyear)
    if period <= 0.0 or float(dt_forcesoil) <= 0.0:
        raise ValueError("one_year*nbyear and dt_forcesoil must be positive")
    iatt0 = int((slot % period) // float(dt_forcesoil))
    if nslots is not None and (iatt0 < 0 or iatt0 >= int(nslots)):
        raise ValueError("computed Cforcing slot is outside the allocated forcing window")
    return iatt0


def stomate_cforcing_accumulate_slot(
    fields,
    increments,
    nforce,
    *,
    date,
    iatt_old=None,
    cumul_cforcing: bool = False,
    one_year=365.0,
    nbyear=1,
    dt_forcesoil=1.0,
) -> StomateCforcingSlotUpdate:
    """Accumulate daily Cforcing fields into the source-selected slot.

    Fortran provenance: ``stomate.f90::stomate_main`` lines 4585-4726.
    When the slot wraps and ``cumul_Cforcing`` is false, Fortran zeros the
    cached forcing arrays and ``nforce`` before accumulating the new day.
    """

    nforce_arr = jnp.asarray(nforce)
    iatt = stomate_cforcing_slot_index(
        date,
        one_year=one_year,
        nbyear=nbyear,
        dt_forcesoil=dt_forcesoil,
        nslots=nforce_arr.shape[0],
    )
    reset_window = iatt_old is not None and iatt < int(iatt_old) and not bool(cumul_cforcing)
    current = {name: jnp.asarray(value) for name, value in dict(fields).items()}
    if reset_window:
        current = {name: jnp.zeros_like(value) for name, value in current.items()}
        nforce_arr = jnp.zeros_like(nforce_arr)
    nforce_arr = nforce_arr.at[iatt].add(1)
    for name, increment in dict(increments).items():
        if name not in current:
            raise KeyError(f"Missing Cforcing field {name!r}")
        current[name] = current[name].at[..., iatt].add(jnp.asarray(increment))
    return StomateCforcingSlotUpdate(
        fields=current,
        nforce=nforce_arr,
        iatt=iatt,
        reset_window=bool(reset_window),
    )


def stomate_cforcing_finalize_slots(fields, nforce, *, reciprocal_fields=()):
    """Average accumulated Cforcing slots and zero empty slots.

    Fortran provenance: ``stomate.f90::stomate`` write sections lines
    5587-5629 and 6138-6168. Most fields are divided by ``nforce``; selected
    permafrost forcing fields use the source reciprocal mean form
    ``1/(sum/nforce)``.
    """

    nforce_arr = jnp.asarray(nforce)
    positive = nforce_arr > 0
    reciprocal = set(reciprocal_fields)
    finalized: dict[str, jnp.ndarray] = {}
    for name, value in dict(fields).items():
        arr = jnp.asarray(value)
        denom_shape = (1,) * (arr.ndim - 1) + (nforce_arr.shape[0],)
        denom = nforce_arr.reshape(denom_shape)
        positive_mask = positive.reshape(denom_shape)
        mean = jnp.where(positive_mask, arr / denom, 0.0)
        if name in reciprocal:
            mean = jnp.where(positive_mask, 1.0 / mean, 0.0)
        finalized[name] = mean
    return finalized


def stomate_crop_growth_degree_hour(t2m, ok_LAIdev, SP_tdmin, SP_tdmax, *, zero_celsius=ZERO_CELSIUS):
    """Compute instantaneous crop ``ugdh`` exactly as ``stomate_main`` does.

    Fortran provenance: ``src_stomate/stomate.f90``, subroutine
    ``stomate_main``, lines 3143-3156. Air temperature is converted to
    Celsius, PFT1/bare soil is untouched at zero, and only PFTs whose
    ``ok_LAIdev`` flag is true receive clipped ``st2m - SP_tdmin`` values.
    """

    t2m = jnp.asarray(t2m)
    ok = jnp.asarray(ok_LAIdev, dtype=bool)
    tdmin = jnp.asarray(SP_tdmin)
    tdmax = jnp.asarray(SP_tdmax)
    st2m = t2m - jnp.asarray(zero_celsius, dtype=t2m.dtype)
    raw = st2m[:, None] - tdmin[None, :]
    capped = jnp.where(st2m[:, None] < tdmin[None, :], 0.0, raw)
    capped = jnp.where(st2m[:, None] > tdmax[None, :], tdmax[None, :] - tdmin[None, :], capped)
    active = ok.at[0].set(False) if ok.shape[0] else ok
    return jnp.where(active[None, :], capped, 0.0)


def stomate_crop_photoperiod_unit(swdown):
    """Compute instantaneous crop photoperiod ``uphoi`` from shortwave input.

    Fortran provenance: ``src_stomate/stomate.f90``, subroutine
    ``stomate_main``, lines 3159-3168: pixels with ``swdown <= 0`` get zero,
    otherwise one hour-equivalent unit.
    """

    swdown = jnp.asarray(swdown)
    return jnp.where(swdown <= 0.0, 0.0, 1.0)


def stomate_instant_daily_prep(
    *,
    day,
    sec,
    dt_sechiba,
    t2m,
    ok_LAIdev,
    SP_tdmin=None,
    SP_tdmax=None,
    swdown=None,
) -> StomateInstantDailyPrep:
    """Prepare the local instantaneous values before STOMATE accumulation.

    Fortran provenance: ``src_stomate/stomate.f90``, subroutine
    ``stomate_main``, lines 3128-3168. Crop-specific ``ugdh`` and ``uphoi``
    are returned only when ``ANY(ok_LAIdev)`` is true, matching the later
    accumulation guard at lines 3187-3196.
    """

    ok = jnp.asarray(ok_LAIdev, dtype=bool)
    any_crop = bool(jnp.any(ok))
    if any_crop:
        missing = [
            name
            for name, value in (
                ("SP_tdmin", SP_tdmin),
                ("SP_tdmax", SP_tdmax),
                ("swdown", swdown),
            )
            if value is None
        ]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Crop daily prep requires explicit source-backed inputs: {joined}")
        ugdh = stomate_crop_growth_degree_hour(t2m, ok, SP_tdmin, SP_tdmax)
        uphoi = stomate_crop_photoperiod_unit(swdown)
    else:
        ugdh = None
        uphoi = None
    return StomateInstantDailyPrep(
        end_of_month=stomate_end_of_month(day, sec, dt_sechiba=dt_sechiba),
        st2m=jnp.asarray(t2m) - ZERO_CELSIUS,
        ugdh=ugdh,
        uphoi=uphoi,
    )


def stomate_update_temperature_extrema_daily(*, t2m_min, t2m_max, t2m_min_daily, t2m_max_daily):
    """Update daily 2m temperature extrema.

    Fortran provenance: ``src_stomate/stomate.f90``, subroutine
    ``stomate_main``, lines 3235-3238:
    ``t2m_min_daily = MIN(t2m_min, t2m_min_daily)`` and
    ``t2m_max_daily = MAX(t2m_max, t2m_max_daily)``.
    """

    return (
        jnp.minimum(jnp.asarray(t2m_min), jnp.asarray(t2m_min_daily)),
        jnp.maximum(jnp.asarray(t2m_max), jnp.asarray(t2m_max_daily)),
    )


def stomate_accumulate_named_daily(
    current: dict[str, object],
    increments: dict[str, object],
    do_slow,
    *,
    dt_sechiba,
    dt_stomate,
) -> dict[str, jnp.ndarray]:
    """Apply ``stomate_accu`` semantics to an explicit set of named fields.

    Fortran provenance: ``src_stomate/stomate.f90``, subroutine
    ``stomate_main``, lines 3187-3240, and generic ``stomate_accu`` routines
    lines 9341-9411. The caller supplies each exact accumulator and increment;
    this helper does not synthesize missing fields.
    """

    result = {name: jnp.asarray(value) for name, value in current.items()}
    for name, increment in increments.items():
        if name not in result:
            raise KeyError(f"Cannot accumulate missing daily field: {name}")
        result[name] = stomate_accumulate_daily(
            result[name],
            increment,
            do_slow,
            dt_sechiba=dt_sechiba,
            dt_stomate=dt_stomate,
        )
    return result


@jit
def _stomate_daily_accumulation_fold_runtime_peat_static_jit(
    initial_fields,
    payload_stacks,
    veget_cov_max,
    dt_sechiba,
    dt_stomate,
):
    """Compiled daily accumulator for the runtime full-day PFT14 peat path."""

    (
        humrel_daily,
        litterhum_daily,
        t2m_daily,
        tsurf_daily,
        tsoil_daily,
        soilhum_daily,
        precip_daily,
        gpp_daily,
        wspeed_daily,
        snowfall_daily,
        snowmass_daily,
        tmc_topgrass_daily,
        t2m_min_daily,
        t2m_max_daily,
        fwet_daily,
        liqwt_daily,
    ) = initial_fields
    (
        precip_rain,
        precip_snow,
        gpp,
        humrel,
        litterhumdiag,
        t2m,
        temp_sol,
        stempdiag,
        shumdiag,
        t2m_min,
        t2m_max,
        wspeed,
        snow,
        tmc_topgrass,
        fwet_new,
        liqwt_ratio,
    ) = payload_stacks
    nsteps = int(t2m.shape[0])
    dt_sechiba = jnp.asarray(dt_sechiba)
    dt_stomate = jnp.asarray(dt_stomate)
    one_day_scale = jnp.asarray(86400.0) / dt_sechiba

    for index in range(nsteps):
        divide = index == nsteps - 1
        precip = (precip_rain[index] + precip_snow[index]) * one_day_scale
        gpp_d = stomate_gpp_daily_increment_from_veget_cov_max(
            gpp[index],
            veget_cov_max,
            dt_sechiba=dt_sechiba,
        )
        humrel_daily = humrel_daily + humrel[index] * dt_sechiba
        litterhum_daily = litterhum_daily + litterhumdiag[index] * dt_sechiba
        t2m_daily = t2m_daily + t2m[index] * dt_sechiba
        tsurf_daily = tsurf_daily + temp_sol[index] * dt_sechiba
        tsoil_daily = tsoil_daily + stempdiag[index] * dt_sechiba
        soilhum_daily = soilhum_daily + shumdiag[index] * dt_sechiba
        precip_daily = precip_daily + precip * dt_sechiba
        gpp_daily = gpp_daily + gpp_d * dt_sechiba
        wspeed_daily = wspeed_daily + wspeed[index] * dt_sechiba
        snowfall_daily = snowfall_daily + precip_snow[index] * dt_sechiba
        snowmass_daily = snowmass_daily + snow[index] * dt_sechiba
        tmc_topgrass_daily = tmc_topgrass_daily + tmc_topgrass[index] * dt_sechiba
        fwet_daily = fwet_daily + fwet_new[index] * dt_sechiba
        liqwt_daily = liqwt_daily + liqwt_ratio[index] * dt_sechiba
        if divide:
            humrel_daily = humrel_daily / dt_stomate
            litterhum_daily = litterhum_daily / dt_stomate
            t2m_daily = t2m_daily / dt_stomate
            tsurf_daily = tsurf_daily / dt_stomate
            tsoil_daily = tsoil_daily / dt_stomate
            soilhum_daily = soilhum_daily / dt_stomate
            precip_daily = precip_daily / dt_stomate
            gpp_daily = gpp_daily / dt_stomate
            wspeed_daily = wspeed_daily / dt_stomate
            snowfall_daily = snowfall_daily / dt_stomate
            snowmass_daily = snowmass_daily / dt_stomate
            tmc_topgrass_daily = tmc_topgrass_daily / dt_stomate
            fwet_daily = fwet_daily / dt_stomate
            liqwt_daily = liqwt_daily / dt_stomate
        t2m_min_daily = jnp.minimum(t2m_min[index], t2m_min_daily)
        t2m_max_daily = jnp.maximum(t2m_max[index], t2m_max_daily)

    return (
        humrel_daily,
        litterhum_daily,
        t2m_daily,
        tsurf_daily,
        tsoil_daily,
        soilhum_daily,
        precip_daily,
        gpp_daily,
        wspeed_daily,
        snowfall_daily,
        snowmass_daily,
        tmc_topgrass_daily,
        t2m_min_daily,
        t2m_max_daily,
        fwet_daily,
        liqwt_daily,
    )


def _stomate_daily_accumulation_fold_runtime_numpy_from_entries(
    *,
    payloads: tuple[dict[str, object], ...],
    fields: dict[str, object],
    shared_veget_cov_max,
    dt_sechiba,
    dt_stomate,
    peat_occur: bool,
) -> dict[str, np.ndarray]:
    """Small-array runtime accumulator using the same source-order recurrence."""

    result = {name: np.asarray(value) for name, value in fields.items()}
    veget_cov_max = np.asarray(shared_veget_cov_max)
    dt_sechiba = float(dt_sechiba)
    dt_stomate = float(dt_stomate)
    scale = 86400.0 / dt_sechiba
    for index, payload in enumerate(payloads):
        do_slow = index == len(payloads) - 1
        precip = (np.asarray(payload["precip_rain"]) + np.asarray(payload["precip_snow"])) * scale
        gpp = np.asarray(payload["gpp"])
        safe_veget = np.where(veget_cov_max > MIN_STOMATE, veget_cov_max, 1.0)
        gpp_d = np.where(veget_cov_max > MIN_STOMATE, gpp / safe_veget * scale, 0.0)
        gpp_d = np.array(gpp_d, copy=True)
        gpp_d[:, 0] = 0.0
        increments = {
            "humrel_daily": payload["humrel"],
            "litterhum_daily": payload["litterhumdiag"],
            "t2m_daily": payload["t2m"],
            "tsurf_daily": payload["temp_sol"],
            "tsoil_daily": payload["stempdiag"],
            "soilhum_daily": payload["shumdiag"],
            "precip_daily": precip,
            "gpp_daily": gpp_d,
            "wspeed_daily": payload["wspeed"],
            "snowfall_daily": payload["precip_snow"],
            "snowmass_daily": payload["snow"],
            "tmc_topgrass_daily": payload["tmc_topgrass"],
        }
        if bool(peat_occur):
            increments["fwet_daily"] = payload["fwet_new"]
            increments["liqwt_daily"] = payload["liqwt_ratio"]
        for name, increment in increments.items():
            result[name] = result[name] + np.asarray(increment) * dt_sechiba
            if do_slow:
                result[name] = result[name] / dt_stomate
        result["t2m_min_daily"] = np.minimum(np.asarray(payload["t2m_min"]), result["t2m_min_daily"])
        result["t2m_max_daily"] = np.maximum(np.asarray(payload["t2m_max"]), result["t2m_max_daily"])
    return result


def accumulate_resp_maint_part(current, resp_maint_part_radia):
    """Accumulate separated maintenance respiration by plant part.

    Fortran provenance: `src_stomate/stomate.f90`, subroutine `stomate_main`,
    lines 3265-3267: `resp_maint_part = resp_maint_part +
    resp_maint_part_radia`.
    """

    return jnp.asarray(current) + jnp.asarray(resp_maint_part_radia)


def sum_resp_maint_radia(resp_maint_part_radia, *, flood_frac=None, root_index=5):
    """Sum per-part maintenance respiration after `maint_respiration`.

    Fortran provenance: `src_stomate/stomate.f90`, subroutine `stomate_main`,
    lines 3254-3263. The total is the sum over plant parts for PFTs 2..nvm;
    bare soil/PFT1 remains zero. ``flood_root_radia`` is always an allocated
    zero array, matching line 3256; when supplied, ``flood_frac`` applies the
    line-3258 root-flood update.
    """

    parts = jnp.asarray(resp_maint_part_radia)
    total = jnp.sum(parts, axis=2)
    total = total.at[:, 0].set(0.0)
    if flood_frac is None:
        flood_root = jnp.zeros_like(total)
    else:
        flood_root = flood_frac[:, None] * parts[:, :, root_index]
        flood_root = flood_root.at[:, 0].set(0.0)
    return total, flood_root


def _maintenance_static_cache(
    *,
    biomass,
    t2m_longterm,
    z_soil,
    rprof,
    sla_calc,
    coeff_maint_zero,
    maint_resp_slope,
    ext_coeff,
    is_tree,
    dt_days,
    min_stomate=MIN_STOMATE,
    maint_resp_min_vmax=DEFAULT_MAINT_RESP_MIN_VMAX,
    maint_resp_coeff=DEFAULT_MAINT_RESP_COEFF,
) -> StomateMaintenanceStaticCache:
    biomass_arr = jnp.asarray(biomass)
    z_soil_arr = jnp.asarray(z_soil)
    rprof_arr = jnp.asarray(rprof)
    if biomass_arr.ndim != 4:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    if biomass_arr.shape[2] != 12:
        raise ValueError("biomass pool axis must have length 12")
    if z_soil_arr.ndim != 1:
        raise ValueError("z_soil must be one-dimensional")
    rpc = 1.0 / (1.0 - jnp.exp(-z_soil_arr[-1] / rprof_arr))
    root_weights = rpc[..., None] * (
        jnp.exp(-z_soil_arr[:-1] / rprof_arr[..., None])
        - jnp.exp(-z_soil_arr[1:] / rprof_arr[..., None])
    )
    sla_calc_arr = jnp.asarray(sla_calc)
    lai = biomass_arr[:, :, 0, 0] * sla_calc_arr
    lai = lai.at[:, 0].set(0.0)
    ext_coeff_arr = jnp.asarray(ext_coeff)
    leaf_factor = (
        maint_resp_min_vmax * lai
        + maint_resp_coeff * (1.0 - jnp.exp(-ext_coeff_arr[None, :] * lai))
    ) / jnp.where(lai > min_stomate, lai, 1.0)
    t2m_longterm_arr = jnp.asarray(t2m_longterm)
    maint_resp_slope_arr = jnp.asarray(maint_resp_slope)
    tl = t2m_longterm_arr - ZERO_CELSIUS
    slope = (
        maint_resp_slope_arr[None, :, 0]
        + tl[:, None] * maint_resp_slope_arr[None, :, 1]
        + tl[:, None] * tl[:, None] * maint_resp_slope_arr[None, :, 2]
    )
    return StomateMaintenanceStaticCache(
        biomass=biomass_arr,
        biomass_carbon=biomass_arr[:, :, :, 0],
        z_soil=z_soil_arr,
        rprof=rprof_arr,
        root_weights=root_weights,
        lai=lai,
        leaf_factor=leaf_factor,
        coeff_maint_zero=jnp.asarray(coeff_maint_zero),
        slope=slope,
        ext_coeff=ext_coeff_arr,
        is_tree=jnp.asarray(is_tree, dtype=bool),
        dt_days=jnp.asarray(dt_days),
    )


def _maintenance_respiration_from_static_cache(
    cache: StomateMaintenanceStaticCache,
    *,
    t2m,
    stempdiag,
    min_stomate=MIN_STOMATE,
) -> MaintenanceRespirationResult:
    t2m_arr = jnp.asarray(t2m)
    stempdiag_arr = jnp.asarray(stempdiag)
    if stempdiag_arr.shape[-1] != cache.z_soil.shape[0] - 1:
        raise ValueError("stempdiag last axis must match len(z_soil) - 1")
    t_root = jnp.sum(stempdiag_arr[:, None, :] * cache.root_weights, axis=-1)
    npts, nvm, _, _ = cache.biomass.shape
    t_maint = jnp.zeros((npts, nvm, 12), dtype=cache.biomass.dtype)
    above_parts = jnp.asarray([0, 1, 3, 6, 8, 9, 10, 11])
    below_parts = jnp.asarray([2, 4, 5])
    t_maint = t_maint.at[:, :, above_parts].set(t2m_arr[:, None, None])
    t_maint = t_maint.at[:, :, below_parts].set(t_root[:, :, None])
    reserve_temp = jnp.where(cache.is_tree[None, :], t2m_arr[:, None], t_root)
    t_maint = t_maint.at[:, :, 7].set(reserve_temp)

    coeff_maint = jnp.maximum(
        (cache.coeff_maint_zero[None, :, :] * cache.dt_days)
        * (1.0 + cache.slope[:, :, None] * (t_maint - ZERO_CELSIUS)),
        0.0,
    )
    plain_resp = coeff_maint * cache.biomass_carbon
    leaf_resp = coeff_maint[:, :, 0] * cache.biomass_carbon[:, :, 0] * cache.leaf_factor
    leaf_resp = jnp.where(
        (cache.biomass_carbon[:, :, 0] > min_stomate) & (cache.lai > min_stomate),
        leaf_resp,
        0.0,
    )
    resp_maint_part = plain_resp.at[:, :, 0].set(leaf_resp)
    resp_maint_part = resp_maint_part.at[:, 0, :].set(0.0)
    return MaintenanceRespirationResult(
        lai=cache.lai,
        t_root=t_root,
        coeff_maint=coeff_maint,
        resp_maint_part=resp_maint_part,
    )


def _maintenance_respiration_parts_from_static_cache_stack(
    cache: StomateMaintenanceStaticCache,
    *,
    t2m_stack,
    stempdiag_stack,
    min_stomate=MIN_STOMATE,
):
    """Compute per-entry maintenance parts for a runtime daily fold."""

    t2m_arr = jnp.asarray(t2m_stack)
    stempdiag_arr = jnp.asarray(stempdiag_stack)
    if stempdiag_arr.shape[-1] != cache.z_soil.shape[0] - 1:
        raise ValueError("stempdiag last axis must match len(z_soil) - 1")
    t_root = jnp.sum(stempdiag_arr[:, :, None, :] * cache.root_weights[None, :, :, :], axis=-1)
    nsteps, npts = t2m_arr.shape
    nvm = cache.biomass.shape[1]
    t_maint = jnp.zeros((nsteps, npts, nvm, 12), dtype=cache.biomass.dtype)
    above_parts = jnp.asarray([0, 1, 3, 6, 8, 9, 10, 11])
    below_parts = jnp.asarray([2, 4, 5])
    t_maint = t_maint.at[:, :, :, above_parts].set(t2m_arr[:, :, None, None])
    t_maint = t_maint.at[:, :, :, below_parts].set(t_root[:, :, :, None])
    reserve_temp = jnp.where(cache.is_tree[None, None, :], t2m_arr[:, :, None], t_root)
    t_maint = t_maint.at[:, :, :, 7].set(reserve_temp)

    coeff_maint = jnp.maximum(
        (cache.coeff_maint_zero[None, None, :, :] * cache.dt_days)
        * (1.0 + cache.slope[None, :, :, None] * (t_maint - ZERO_CELSIUS)),
        0.0,
    )
    plain_resp = coeff_maint * cache.biomass_carbon[None, :, :, :]
    leaf_resp = (
        coeff_maint[:, :, :, 0]
        * cache.biomass_carbon[None, :, :, 0]
        * cache.leaf_factor[None, :, :]
    )
    leaf_resp = jnp.where(
        (cache.biomass_carbon[None, :, :, 0] > min_stomate) & (cache.lai[None, :, :] > min_stomate),
        leaf_resp,
        0.0,
    )
    resp_maint_part = plain_resp.at[:, :, :, 0].set(leaf_resp)
    return resp_maint_part.at[:, :, 0, :].set(0.0)


_maintenance_respiration_parts_from_static_cache_stack_jit = jit(
    _maintenance_respiration_parts_from_static_cache_stack
)


def stomate_maintenance_respiration_parts_from_entries(
    *,
    entry_payloads,
    biomass,
    t2m_longterm,
    z_soil,
    rprof,
    sla_calc,
    coeff_maint_zero,
    maint_resp_slope,
    ext_coeff,
    is_tree,
    dt_sechiba,
    one_day=86400.0,
):
    """Return all source-ordered half-hour maintenance parts in one dispatch."""

    payloads = tuple(entry_payloads)
    if not payloads:
        raise ValueError("At least one explicit STOMATE entry payload is required")
    return stomate_maintenance_respiration_parts_from_stacks(
        t2m_stack=jnp.stack([jnp.asarray(payload["t2m"]) for payload in payloads]),
        stempdiag_stack=jnp.stack(
            [jnp.asarray(payload["stempdiag"]) for payload in payloads]
        ),
        biomass=biomass,
        t2m_longterm=t2m_longterm,
        z_soil=z_soil,
        rprof=rprof,
        sla_calc=sla_calc,
        coeff_maint_zero=coeff_maint_zero,
        maint_resp_slope=maint_resp_slope,
        ext_coeff=ext_coeff,
        is_tree=is_tree,
        dt_sechiba=dt_sechiba,
        one_day=one_day,
    )


def stomate_maintenance_respiration_parts_from_stacks(
    *,
    t2m_stack,
    stempdiag_stack,
    biomass,
    t2m_longterm,
    z_soil,
    rprof,
    sla_calc,
    coeff_maint_zero,
    maint_resp_slope,
    ext_coeff,
    is_tree,
    dt_sechiba,
    one_day=86400.0,
):
    """Return maintenance parts from an already stacked device-resident day."""

    cache = _maintenance_static_cache(
        biomass=biomass,
        t2m_longterm=t2m_longterm,
        z_soil=z_soil,
        rprof=rprof,
        sla_calc=sla_calc,
        coeff_maint_zero=coeff_maint_zero,
        maint_resp_slope=maint_resp_slope,
        ext_coeff=ext_coeff,
        is_tree=is_tree,
        dt_days=jnp.asarray(dt_sechiba) / jnp.asarray(one_day),
    )
    return _maintenance_respiration_parts_from_static_cache_stack_jit(
        cache,
        t2m_stack=t2m_stack,
        stempdiag_stack=stempdiag_stack,
    )


def stomate_maintenance_fold_from_entries(
    *,
    entry_payloads,
    resp_maint_part_current,
    biomass,
    t2m_longterm,
    z_soil,
    rprof,
    sla_calc,
    coeff_maint_zero,
    maint_resp_slope,
    ext_coeff,
    is_tree,
    dt_sechiba,
    flood_frac=None,
    one_day=86400.0,
    retain_step_results: bool = True,
    use_jit: bool = False,
) -> StomateMaintenanceFold:
    """Fold maintenance respiration over ordered explicit STOMATE entries.

    Fortran provenance: ``src_stomate/stomate.f90::stomate_main`` lines
    3244-3267 call ``maint_respiration`` at every STOMATE invocation, sum
    ``resp_maint_radia``/``flood_root_radia``, and accumulate
    ``resp_maint_part = resp_maint_part + resp_maint_part_radia``. This helper
    preserves that recurrence only; it does not run allocation, NPP, turnover,
    litter, or the daily reset block.
    """

    payloads = tuple(entry_payloads)
    if not payloads:
        raise ValueError("At least one explicit STOMATE entry payload is required")

    provenance = (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3244-3267",
        "fortran_source/ORCHIDEE/src_stomate/stomate_resp.f90::maint_respiration lines 122-376",
    )
    required = ("t2m", "stempdiag")
    current = jnp.asarray(resp_maint_part_current)
    dt_days = jnp.asarray(dt_sechiba) / jnp.asarray(one_day)
    static_cache = _maintenance_static_cache(
        biomass=biomass,
        t2m_longterm=t2m_longterm,
        z_soil=z_soil,
        rprof=rprof,
        sla_calc=sla_calc,
        coeff_maint_zero=coeff_maint_zero,
        maint_resp_slope=maint_resp_slope,
        ext_coeff=ext_coeff,
        is_tree=is_tree,
        dt_days=dt_days,
    )
    step_results: list[MaintenanceRespirationResult] = []
    resp_maint_radia = None
    flood_root_radia = None

    if not retain_step_results and not use_jit:
        t2m_values = []
        stempdiag_values = []
        for index, payload in enumerate(payloads):
            missing = tuple(name for name in required if name not in payload)
            if missing:
                return StomateMaintenanceFold(
                    resp_maint_part=current,
                    step_results=(),
                    resp_maint_radia=resp_maint_radia,
                    flood_root_radia=flood_root_radia,
                    missing_inputs=missing,
                    failed_step=index,
                    provenance=provenance,
                    notes=("Maintenance runtime fold stopped without fabricating missing entry inputs.",),
                )
            stempdiag_arr = jnp.asarray(payload["stempdiag"])
            if stempdiag_arr.shape[-1] != static_cache.z_soil.shape[0] - 1:
                raise ValueError("stempdiag last axis must match len(z_soil) - 1")
            t2m_values.append(jnp.asarray(payload["t2m"]))
            stempdiag_values.append(stempdiag_arr)
        resp_parts = _maintenance_respiration_parts_from_static_cache_stack(
            static_cache,
            t2m_stack=jnp.stack(t2m_values, axis=0),
            stempdiag_stack=jnp.stack(stempdiag_values, axis=0),
        )
        for index in range(len(payloads)):
            current = accumulate_resp_maint_part(current, resp_parts[index])
        resp_maint_radia, flood_root_radia = sum_resp_maint_radia(
            resp_parts[-1],
            flood_frac=flood_frac,
        )
        return StomateMaintenanceFold(
            resp_maint_part=current,
            step_results=(),
            resp_maint_radia=resp_maint_radia,
            flood_root_radia=flood_root_radia,
            missing_inputs=(),
            failed_step=None,
            provenance=provenance,
            notes=(
                "Runtime maintenance respiration is computed for all entries and accumulated in Fortran order.",
                "Per-entry debug maintenance objects are omitted because retain_step_results=False.",
            ),
        )

    for index, payload in enumerate(payloads):
        missing = tuple(name for name in required if name not in payload)
        if missing:
            return StomateMaintenanceFold(
                resp_maint_part=current,
                step_results=tuple(step_results),
                resp_maint_radia=resp_maint_radia,
                flood_root_radia=flood_root_radia,
                missing_inputs=missing,
                failed_step=index,
                provenance=provenance,
                notes=("Maintenance fold stopped without fabricating missing entry inputs.",),
            )
        stempdiag_arr = jnp.asarray(payload["stempdiag"])
        if stempdiag_arr.shape[-1] != static_cache.z_soil.shape[0] - 1:
            raise ValueError("stempdiag last axis must match len(z_soil) - 1")
        if use_jit:
            maintenance = maintenance_respiration_core_jit(
                static_cache.biomass,
                jnp.asarray(payload["t2m"]),
                jnp.asarray(t2m_longterm),
                stempdiag_arr,
                static_cache.z_soil,
                static_cache.rprof,
                jnp.asarray(sla_calc),
                static_cache.coeff_maint_zero,
                jnp.asarray(maint_resp_slope),
                static_cache.ext_coeff,
                static_cache.is_tree,
                dt_days,
                MIN_STOMATE,
                DEFAULT_MAINT_RESP_MIN_VMAX,
                DEFAULT_MAINT_RESP_COEFF,
            )
        else:
            maintenance = _maintenance_respiration_from_static_cache(
                static_cache,
                t2m=payload["t2m"],
                stempdiag=stempdiag_arr,
            )
        if retain_step_results:
            step_results.append(maintenance)
        resp_maint_radia, flood_root_radia = sum_resp_maint_radia(
            maintenance.resp_maint_part,
            flood_frac=flood_frac,
        )
        current = accumulate_resp_maint_part(current, maintenance.resp_maint_part)

    return StomateMaintenanceFold(
        resp_maint_part=current,
        step_results=tuple(step_results),
        resp_maint_radia=resp_maint_radia,
        flood_root_radia=flood_root_radia,
        missing_inputs=(),
        failed_step=None,
        provenance=provenance,
        notes=(
            "Sequential maintenance respiration is accumulated through the supplied explicit entry payloads.",
            "Downstream carbon processes and end-of-day reset remain separate STOMATE stages.",
        ),
    )


def reset_daily_on_slow(fields: dict[str, object], do_slow: bool, names: tuple[str, ...]) -> dict[str, jnp.ndarray]:
    """Reset selected daily fields to zero at the end of a slow STOMATE step.

    Fortran provenance: `src_stomate/stomate.f90`, subroutine `stomate_main`,
    daily reset block lines 4950-5000, including `gpp_daily(:,:)=zero`,
    `resp_maint_part(:,:,:)=zero`, and multiple accumulated daily forcing
    fields. This helper only resets fields explicitly named by the caller.
    """

    result = {name: jnp.asarray(value) for name, value in fields.items()}
    if do_slow:
        for name in names:
            if name not in result:
                raise KeyError(f"Cannot reset missing daily field: {name}")
            result[name] = jnp.zeros_like(result[name])
    return result


def prepare_daily_carbon_inputs(
    *,
    gpp_daily,
    biomass,
    resp_maint_part,
    resp_maint=None,
    pft_present=None,
    f_alloc=None,
) -> DailyCarbonBoundary:
    """Package explicit inputs for a future maint/alloc/npp adapter.

    Fortran provenance: `src_stomate/stomate.f90`, `StomateLpj` call boundary
    lines 4297-4335; `src_stomate/stomate_lpj.f90`, active path order lines
    1070-1131 (`phenology -> alloc -> setlai -> npp_calc`). This function does
    not compute missing process state. If `f_alloc` is absent, it is reported in
    `requires_trace` rather than fabricated.
    """

    missing = list(REQUIRED_CARBON_TRACE_FIELDS)
    if f_alloc is not None:
        missing.remove("f_alloc")
    if resp_maint_part is not None and "resp_maint_part" in missing:
        missing.remove("resp_maint_part")

    return DailyCarbonBoundary(
        gpp_daily=jnp.asarray(gpp_daily),
        biomass=jnp.asarray(biomass),
        resp_maint_part=jnp.asarray(resp_maint_part),
        resp_maint=None if resp_maint is None else jnp.asarray(resp_maint),
        pft_present=None if pft_present is None else jnp.asarray(pft_present),
        f_alloc=None if f_alloc is None else jnp.asarray(f_alloc),
        requires_trace=tuple(missing),
    )


def require_explicit_f_alloc(boundary: DailyCarbonBoundary):
    """Return `f_alloc` or fail before any downstream NPP integration.

    Fortran provenance: `src_stomate/stomate_lpj.f90`, lines 1093-1102 create
    `f_alloc` via `alloc`, and lines 1118-1123 pass it into `npp_calc`.
    Because current restart/history files do not contain full `f_alloc`, this
    guard prevents accidental placeholder allocation.
    """

    if boundary.f_alloc is None:
        raise ValueError("f_alloc is required from alloc trace/output; it must not be fabricated")
    return boundary.f_alloc


def stomate_first_step_daily_accumulation_from_entry(
    *,
    entry_payload: dict[str, object],
    accumulator_state,
    dt_sechiba,
    dt_stomate,
    do_slow,
    peat_occur=False,
    date=None,
):
    """Run the source-backed daily accumulation subset at STOMATE entry.

    Fortran provenance: ``src_stomate/stomate.f90::stomate_main`` lines
    2917-3032 build local ``precip``, vegetation fractions, and ``gpp_d``;
    lines 3187-3240 call ``stomate_accu`` for the first daily accumulator
    group and update daily temperature extrema. Crop and permafrost daily
    accumulators are deliberately outside this helper until their branch
    controls and full accumulator state are supplied.
    """

    from jax_orchidee.stomate.daily_inputs import stomate_entry_local_prep_explicit

    required = (
        "precip_rain",
        "precip_snow",
        "veget",
        "veget_max",
        "totfrac_nobio",
        "gpp",
        "humrel",
        "litterhumdiag",
        "t2m",
        "temp_sol",
        "stempdiag",
        "shumdiag",
        "t2m_min",
        "t2m_max",
        "wspeed",
        "snow",
        "tmc_topgrass",
    )
    missing = [name for name in required if name not in entry_payload]
    peat_required = ("fwet_new", "liqwt_ratio") if bool(peat_occur) else ()
    missing.extend(name for name in peat_required if name not in entry_payload)
    if missing:
        return StomateFirstStepDailyAccumulation(
            local_prep=None,
            fields={},
            missing_inputs=tuple(dict.fromkeys(missing)),
            provenance=(
                "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 2917-3032",
                "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3187-3240",
            ),
            notes=("Daily accumulation subset skipped because exact entry inputs are missing.",),
        )

    local = stomate_entry_local_prep_explicit(
        precip_rain=entry_payload["precip_rain"],
        precip_snow=entry_payload["precip_snow"],
        dt_sechiba=dt_sechiba,
        veget=entry_payload["veget"],
        veget_max=entry_payload["veget_max"],
        totfrac_nobio=entry_payload["totfrac_nobio"],
        gpp=entry_payload["gpp"],
        glccNetLCC=entry_payload.get("glccNetLCC"),
        glccSecondShift=entry_payload.get("glccSecondShift"),
        glccPrimaryShift=entry_payload.get("glccPrimaryShift"),
        harvest_matrix=entry_payload.get("harvest_matrix"),
        date=date if "vegetnew_firstday" in entry_payload else None,
        vegetnew_firstday=entry_payload.get("vegetnew_firstday"),
        totfrac_nobio_new=entry_payload.get("totfrac_nobio_new"),
        veget_max_new=entry_payload.get("veget_max_new"),
        do_now_stomate_lcchange=entry_payload.get("do_now_stomate_lcchange", False),
        dyn_peat=entry_payload.get("dyn_peat", False),
        update_peatfrac=entry_payload.get("update_peatfrac", False),
    )
    current = {
        "humrel_daily": accumulator_state.humrel_daily,
        "litterhum_daily": accumulator_state.litterhum_daily,
        "t2m_daily": accumulator_state.t2m_daily,
        "tsurf_daily": accumulator_state.tsurf_daily,
        "tsoil_daily": accumulator_state.tsoil_daily,
        "soilhum_daily": accumulator_state.soilhum_daily,
        "precip_daily": accumulator_state.precip_daily,
        "gpp_daily": accumulator_state.gpp_daily,
        "wspeed_daily": accumulator_state.wspeed_daily,
        "snowfall_daily": accumulator_state.snowfall_daily,
        "snowmass_daily": accumulator_state.snowmass_daily,
        "tmc_topgrass_daily": accumulator_state.tmc_topgrass_daily,
    }
    increments = {
        "humrel_daily": entry_payload["humrel"],
        "litterhum_daily": entry_payload["litterhumdiag"],
        "t2m_daily": entry_payload["t2m"],
        "tsurf_daily": entry_payload["temp_sol"],
        "tsoil_daily": entry_payload["stempdiag"],
        "soilhum_daily": entry_payload["shumdiag"],
        "precip_daily": local.precip,
        "gpp_daily": local.gpp_d,
        "wspeed_daily": entry_payload["wspeed"],
        "snowfall_daily": entry_payload["precip_snow"],
        "snowmass_daily": entry_payload["snow"],
        "tmc_topgrass_daily": entry_payload["tmc_topgrass"],
    }
    if bool(peat_occur):
        current["fwet_daily"] = accumulator_state.fwet_daily
        current["liqwt_daily"] = accumulator_state.liqwt_daily
        increments["fwet_daily"] = entry_payload["fwet_new"]
        increments["liqwt_daily"] = entry_payload["liqwt_ratio"]

    fields = stomate_accumulate_named_daily(
        current,
        increments,
        do_slow,
        dt_sechiba=dt_sechiba,
        dt_stomate=dt_stomate,
    )
    t2m_min_daily, t2m_max_daily = stomate_update_temperature_extrema_daily(
        t2m_min=entry_payload["t2m_min"],
        t2m_max=entry_payload["t2m_max"],
        t2m_min_daily=accumulator_state.t2m_min_daily,
        t2m_max_daily=accumulator_state.t2m_max_daily,
    )
    fields["t2m_min_daily"] = t2m_min_daily
    fields["t2m_max_daily"] = t2m_max_daily
    return StomateFirstStepDailyAccumulation(
        local_prep=local,
        fields=fields,
        missing_inputs=(),
        provenance=(
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 2917-3032",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3187-3240",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_accu_r1d/r2d/r3d lines 9341-9411",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3211-3213",
        ),
        notes=(
            "Covers the non-crop, non-permafrost first daily accumulator group and temperature extrema.",
            "Maintenance respiration and downstream carbon process calls remain separate STOMATE stages.",
        ),
    )


def _stomate_daily_accumulation_fold_runtime_from_entries(
    *,
    entry_payloads,
    accumulator_state,
    do_slow_flags,
    dt_sechiba,
    dt_stomate,
    peat_occur=False,
    use_compiled_runtime: bool = False,
    use_numpy_runtime: bool = False,
) -> StomateDailyAccumulationFold:
    """Runtime-only daily accumulation fold without per-entry debug objects."""

    from jax_orchidee.stomate.daily_inputs import (
        stomate_gpp_daily_increment_from_veget_cov_max,
        stomate_gpp_daily_increment,
        stomate_precip_increment,
        stomate_veget_cov_max,
    )

    payloads = tuple(entry_payloads)
    flags = tuple(bool(flag) for flag in do_slow_flags)
    if len(payloads) != len(flags):
        raise ValueError("entry_payloads and do_slow_flags must have the same length")
    if not payloads:
        raise ValueError("At least one explicit STOMATE entry payload is required")
    if any(flags[:-1]):
        raise ValueError("A do_slow boundary must be the final entry in this daily fold")

    required = (
        "precip_rain",
        "precip_snow",
        "veget",
        "veget_max",
        "totfrac_nobio",
        "gpp",
        "humrel",
        "litterhumdiag",
        "t2m",
        "temp_sol",
        "stempdiag",
        "shumdiag",
        "t2m_min",
        "t2m_max",
        "wspeed",
        "snow",
        "tmc_topgrass",
    )
    peat_required = ("fwet_new", "liqwt_ratio") if bool(peat_occur) else ()
    provenance = (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3187-3240",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_accu_r1d/r2d/r3d lines 9341-9411",
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 656-662",
    )

    fields: dict[str, jnp.ndarray] = {
        "humrel_daily": jnp.asarray(accumulator_state.humrel_daily),
        "litterhum_daily": jnp.asarray(accumulator_state.litterhum_daily),
        "t2m_daily": jnp.asarray(accumulator_state.t2m_daily),
        "tsurf_daily": jnp.asarray(accumulator_state.tsurf_daily),
        "tsoil_daily": jnp.asarray(accumulator_state.tsoil_daily),
        "soilhum_daily": jnp.asarray(accumulator_state.soilhum_daily),
        "precip_daily": jnp.asarray(accumulator_state.precip_daily),
        "gpp_daily": jnp.asarray(accumulator_state.gpp_daily),
        "wspeed_daily": jnp.asarray(accumulator_state.wspeed_daily),
        "snowfall_daily": jnp.asarray(accumulator_state.snowfall_daily),
        "snowmass_daily": jnp.asarray(accumulator_state.snowmass_daily),
        "tmc_topgrass_daily": jnp.asarray(accumulator_state.tmc_topgrass_daily),
        "t2m_min_daily": jnp.asarray(accumulator_state.t2m_min_daily),
        "t2m_max_daily": jnp.asarray(accumulator_state.t2m_max_daily),
    }
    if bool(peat_occur):
        fields["fwet_daily"] = jnp.asarray(accumulator_state.fwet_daily)
        fields["liqwt_daily"] = jnp.asarray(accumulator_state.liqwt_daily)

    shared_veget_cov_max = None
    try:
        if bool(use_compiled_runtime):
            shared_veget_cov_max = stomate_veget_cov_max(
                payloads[0]["veget_max"],
                payloads[0]["totfrac_nobio"],
            )
        else:
            first_veget_max = np.asarray(payloads[0]["veget_max"])
            first_totfrac_nobio = np.asarray(payloads[0]["totfrac_nobio"])
            if all(
                np.array_equal(np.asarray(payload["veget_max"]), first_veget_max)
                and np.array_equal(np.asarray(payload["totfrac_nobio"]), first_totfrac_nobio)
                for payload in payloads[1:]
            ):
                shared_veget_cov_max = stomate_veget_cov_max(
                    first_veget_max,
                    first_totfrac_nobio,
                )
    except KeyError:
        shared_veget_cov_max = None

    if bool(use_compiled_runtime) and bool(peat_occur) and shared_veget_cov_max is not None and bool(flags[-1]) and not any(flags[:-1]):
        for index, payload in enumerate(payloads):
            step_missing = [name for name in required if name not in payload]
            step_missing.extend(name for name in peat_required if name not in payload)
            if step_missing:
                return StomateDailyAccumulationFold(
                    fields=fields,
                    step_results=(),
                    missing_inputs=tuple(dict.fromkeys(step_missing)),
                    failed_step=index,
                    provenance=provenance,
                    notes=(
                        "Runtime daily fold stopped without fabricating missing STOMATE entry inputs.",
                        "Per-entry debug local_prep objects are omitted because retain_step_results=False.",
                    ),
                )
        initial_fields = (
            fields["humrel_daily"],
            fields["litterhum_daily"],
            fields["t2m_daily"],
            fields["tsurf_daily"],
            fields["tsoil_daily"],
            fields["soilhum_daily"],
            fields["precip_daily"],
            fields["gpp_daily"],
            fields["wspeed_daily"],
            fields["snowfall_daily"],
            fields["snowmass_daily"],
            fields["tmc_topgrass_daily"],
            fields["t2m_min_daily"],
            fields["t2m_max_daily"],
            fields["fwet_daily"],
            fields["liqwt_daily"],
        )
        payload_stacks = (
            jnp.stack([jnp.asarray(payload["precip_rain"]) for payload in payloads], axis=0),
            jnp.stack([jnp.asarray(payload["precip_snow"]) for payload in payloads], axis=0),
            jnp.stack([jnp.asarray(payload["gpp"]) for payload in payloads], axis=0),
            jnp.stack([jnp.asarray(payload["humrel"]) for payload in payloads], axis=0),
            jnp.stack([jnp.asarray(payload["litterhumdiag"]) for payload in payloads], axis=0),
            jnp.stack([jnp.asarray(payload["t2m"]) for payload in payloads], axis=0),
            jnp.stack([jnp.asarray(payload["temp_sol"]) for payload in payloads], axis=0),
            jnp.stack([jnp.asarray(payload["stempdiag"]) for payload in payloads], axis=0),
            jnp.stack([jnp.asarray(payload["shumdiag"]) for payload in payloads], axis=0),
            jnp.stack([jnp.asarray(payload["t2m_min"]) for payload in payloads], axis=0),
            jnp.stack([jnp.asarray(payload["t2m_max"]) for payload in payloads], axis=0),
            jnp.stack([jnp.asarray(payload["wspeed"]) for payload in payloads], axis=0),
            jnp.stack([jnp.asarray(payload["snow"]) for payload in payloads], axis=0),
            jnp.stack([jnp.asarray(payload["tmc_topgrass"]) for payload in payloads], axis=0),
            jnp.stack([jnp.asarray(payload["fwet_new"]) for payload in payloads], axis=0),
            jnp.stack([jnp.asarray(payload["liqwt_ratio"]) for payload in payloads], axis=0),
        )
        compiled_fields = _stomate_daily_accumulation_fold_runtime_peat_static_jit(
            initial_fields,
            payload_stacks,
            shared_veget_cov_max,
            dt_sechiba,
            dt_stomate,
        )
        fields = {
            "humrel_daily": compiled_fields[0],
            "litterhum_daily": compiled_fields[1],
            "t2m_daily": compiled_fields[2],
            "tsurf_daily": compiled_fields[3],
            "tsoil_daily": compiled_fields[4],
            "soilhum_daily": compiled_fields[5],
            "precip_daily": compiled_fields[6],
            "gpp_daily": compiled_fields[7],
            "wspeed_daily": compiled_fields[8],
            "snowfall_daily": compiled_fields[9],
            "snowmass_daily": compiled_fields[10],
            "tmc_topgrass_daily": compiled_fields[11],
            "t2m_min_daily": compiled_fields[12],
            "t2m_max_daily": compiled_fields[13],
            "fwet_daily": compiled_fields[14],
            "liqwt_daily": compiled_fields[15],
        }
        return StomateDailyAccumulationFold(
            fields=fields,
            step_results=(),
            missing_inputs=(),
            failed_step=None,
            provenance=provenance,
            notes=(
                "Runtime daily accumulators are advanced in Fortran entry order.",
                "Per-entry debug local_prep objects are omitted because retain_step_results=False.",
            ),
        )

    if bool(use_numpy_runtime) and shared_veget_cov_max is not None and bool(flags[-1]) and not any(flags[:-1]):
        for index, payload in enumerate(payloads):
            step_missing = [name for name in required if name not in payload]
            step_missing.extend(name for name in peat_required if name not in payload)
            if step_missing:
                return StomateDailyAccumulationFold(
                    fields=fields,
                    step_results=(),
                    missing_inputs=tuple(dict.fromkeys(step_missing)),
                    failed_step=index,
                    provenance=provenance,
                    notes=(
                        "Runtime daily fold stopped without fabricating missing STOMATE entry inputs.",
                        "Per-entry debug local_prep objects are omitted because retain_step_results=False.",
                    ),
                )
        fields = _stomate_daily_accumulation_fold_runtime_numpy_from_entries(
            payloads=payloads,
            fields=fields,
            shared_veget_cov_max=shared_veget_cov_max,
            dt_sechiba=dt_sechiba,
            dt_stomate=dt_stomate,
            peat_occur=bool(peat_occur),
        )
        return StomateDailyAccumulationFold(
            fields=fields,
            step_results=(),
            missing_inputs=(),
            failed_step=None,
            provenance=provenance,
            notes=(
                "Runtime daily accumulators are advanced in Fortran entry order.",
            "Per-entry debug local_prep objects are omitted because retain_step_results=False.",
        ),
    )

    def _accumulate_runtime(current, increment, do_slow):
        accumulated = current + increment * dt_sechiba
        return accumulated / dt_stomate if do_slow else accumulated

    for index, (payload, do_slow) in enumerate(zip(payloads, flags, strict=True)):
        missing = [name for name in required if name not in payload]
        missing.extend(name for name in peat_required if name not in payload)
        if missing:
            return StomateDailyAccumulationFold(
                fields=fields,
                step_results=(),
                missing_inputs=tuple(dict.fromkeys(missing)),
                failed_step=index,
                provenance=provenance,
                notes=(
                    "Runtime daily fold stopped without fabricating missing STOMATE entry inputs.",
                    "Per-entry debug local_prep objects are omitted because retain_step_results=False.",
                ),
            )

        precip = stomate_precip_increment(
            payload["precip_rain"],
            payload["precip_snow"],
            dt_sechiba=dt_sechiba,
        )
        if shared_veget_cov_max is None:
            gpp_d = stomate_gpp_daily_increment(
                payload["gpp"],
                payload["veget_max"],
                payload["totfrac_nobio"],
                dt_sechiba=dt_sechiba,
            )
        else:
            gpp_d = stomate_gpp_daily_increment_from_veget_cov_max(
                payload["gpp"],
                shared_veget_cov_max,
                dt_sechiba=dt_sechiba,
            )
        increments = {
            "humrel_daily": payload["humrel"],
            "litterhum_daily": payload["litterhumdiag"],
            "t2m_daily": payload["t2m"],
            "tsurf_daily": payload["temp_sol"],
            "tsoil_daily": payload["stempdiag"],
            "soilhum_daily": payload["shumdiag"],
            "precip_daily": precip,
            "gpp_daily": gpp_d,
            "wspeed_daily": payload["wspeed"],
            "snowfall_daily": payload["precip_snow"],
            "snowmass_daily": payload["snow"],
            "tmc_topgrass_daily": payload["tmc_topgrass"],
        }
        if bool(peat_occur):
            increments["fwet_daily"] = payload["fwet_new"]
            increments["liqwt_daily"] = payload["liqwt_ratio"]
        for name, increment in increments.items():
            fields[name] = _accumulate_runtime(fields[name], increment, do_slow)
        fields["t2m_min_daily"], fields["t2m_max_daily"] = stomate_update_temperature_extrema_daily(
            t2m_min=payload["t2m_min"],
            t2m_max=payload["t2m_max"],
            t2m_min_daily=fields["t2m_min_daily"],
            t2m_max_daily=fields["t2m_max_daily"],
        )

    return StomateDailyAccumulationFold(
        fields=fields,
        step_results=(),
        missing_inputs=(),
        failed_step=None,
        provenance=provenance,
        notes=(
            "Runtime daily accumulators are advanced in Fortran entry order.",
            "Per-entry debug local_prep objects are omitted because retain_step_results=False.",
        ),
    )


def stomate_daily_accumulation_fold_from_entries(
    *,
    entry_payloads,
    accumulator_state,
    do_slow_flags,
    dt_sechiba,
    dt_stomate,
    peat_occur=False,
    dates=None,
    date=None,
    retain_step_results: bool = True,
    use_compiled_runtime: bool = False,
    use_numpy_runtime: bool = False,
) -> StomateDailyAccumulationFold:
    """Fold STOMATE daily accumulators over ordered explicit entry payloads.

    Fortran provenance: ``src_stomate/stomate.f90::stomate_main`` lines
    3187-3240 call ``stomate_accu`` every STOMATE invocation, while
    ``src_sechiba/slowproc.f90::slowproc_main`` lines 656-662 raises
    ``do_slow`` only on the last SECHIBA step of a STOMATE day. This helper
    only reproduces that accumulator recurrence. It does not run ``season``,
    ``StomateLpj``, maintenance respiration, or the end-of-day reset block.
    """

    payloads = tuple(entry_payloads)
    flags = tuple(bool(flag) for flag in do_slow_flags)
    if len(payloads) != len(flags):
        raise ValueError("entry_payloads and do_slow_flags must have the same length")
    if not payloads:
        raise ValueError("At least one explicit STOMATE entry payload is required")
    if any(flags[:-1]):
        raise ValueError("A do_slow boundary must be the final entry in this daily fold")

    if not retain_step_results:
        return _stomate_daily_accumulation_fold_runtime_from_entries(
            entry_payloads=payloads,
            accumulator_state=accumulator_state,
            do_slow_flags=flags,
            dt_sechiba=dt_sechiba,
            dt_stomate=dt_stomate,
            peat_occur=peat_occur,
            use_compiled_runtime=use_compiled_runtime,
            use_numpy_runtime=use_numpy_runtime,
        )

    if dates is None:
        step_dates = (date,) * len(payloads)
    else:
        step_dates = tuple(dates)
        if len(step_dates) != len(payloads):
            raise ValueError("dates must have the same length as entry_payloads")

    current_fields: dict[str, jnp.ndarray] = {}
    step_results: list[StomateFirstStepDailyAccumulation] = []
    current_state = accumulator_state
    provenance = (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3187-3240",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_accu_r1d/r2d/r3d lines 9341-9411",
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 656-662",
    )

    for index, (payload, do_slow, step_date) in enumerate(zip(payloads, flags, step_dates, strict=True)):
        result = stomate_first_step_daily_accumulation_from_entry(
            entry_payload=payload,
            accumulator_state=current_state,
            dt_sechiba=dt_sechiba,
            dt_stomate=dt_stomate,
            do_slow=do_slow,
            peat_occur=peat_occur,
            date=step_date,
        )
        if retain_step_results:
            step_results.append(result)
        if result.missing_inputs:
            return StomateDailyAccumulationFold(
                fields=current_fields,
                step_results=tuple(step_results),
                missing_inputs=result.missing_inputs,
                failed_step=index,
                provenance=provenance,
                notes=(
                    "Daily fold stopped without fabricating missing STOMATE entry inputs.",
                    "Returned fields are the last fully accumulated state before the failed step.",
                ),
            )
        current_fields = result.fields
        current_state = _AccumulatorStateView(accumulator_state, current_fields)

    return StomateDailyAccumulationFold(
        fields=current_fields,
        step_results=tuple(step_results),
        missing_inputs=(),
        failed_step=None,
        provenance=provenance,
        notes=(
            "Sequential daily accumulators are advanced through the supplied explicit entry payloads.",
            "Downstream carbon processes and end-of-day reset remain separate STOMATE stages.",
        ),
    )


def stomate_daily_process_fold_from_entries(
    *,
    entry_payloads,
    accumulator_state,
    resp_maint_part_current,
    do_slow_flags,
    dt_sechiba,
    dt_stomate,
    biomass,
    t2m_longterm,
    z_soil,
    rprof,
    sla_calc,
    coeff_maint_zero,
    maint_resp_slope,
    ext_coeff,
    is_tree,
    flood_frac=None,
    peat_occur=False,
    dates=None,
    date=None,
    one_day=86400.0,
    retain_step_results: bool = True,
    use_maintenance_jit: bool = False,
    use_compiled_accumulator: bool = False,
    use_numpy_accumulator: bool = False,
    single_pass: bool = False,
) -> StomateDailyProcessFold:
    """Build the exact daily boundary consumed at a ``do_slow`` STOMATE day.

    Fortran provenance combines the daily accumulator recurrence
    (``stomate.f90`` lines 3187-3240), the maintenance recurrence
    (lines 3244-3267), and the daily-process gate before ``season``/
    ``StomateLpj`` (lines 3589 and 4225-4364). This function does not run
    ``season``, ``StomateLpj``, OK_LEAK, or reset the daily accumulators.
    """

    if single_pass:
        return stomate_daily_process_fold_single_pass_from_entries(
            entry_payloads=entry_payloads,
            accumulator_state=accumulator_state,
            resp_maint_part_current=resp_maint_part_current,
            do_slow_flags=do_slow_flags,
            dt_sechiba=dt_sechiba,
            dt_stomate=dt_stomate,
            biomass=biomass,
            t2m_longterm=t2m_longterm,
            z_soil=z_soil,
            rprof=rprof,
            sla_calc=sla_calc,
            coeff_maint_zero=coeff_maint_zero,
            maint_resp_slope=maint_resp_slope,
            ext_coeff=ext_coeff,
            is_tree=is_tree,
            flood_frac=flood_frac,
            peat_occur=peat_occur,
            dates=dates,
            date=date,
            one_day=one_day,
            retain_step_results=retain_step_results,
            use_maintenance_jit=use_maintenance_jit,
        )

    accumulator = stomate_daily_accumulation_fold_from_entries(
        entry_payloads=entry_payloads,
        accumulator_state=accumulator_state,
        do_slow_flags=do_slow_flags,
        dt_sechiba=dt_sechiba,
        dt_stomate=dt_stomate,
        peat_occur=peat_occur,
        dates=dates,
        date=date,
        retain_step_results=retain_step_results,
        use_compiled_runtime=use_compiled_accumulator,
        use_numpy_runtime=use_numpy_accumulator,
    )
    maintenance = stomate_maintenance_fold_from_entries(
        entry_payloads=entry_payloads,
        resp_maint_part_current=resp_maint_part_current,
        biomass=biomass,
        t2m_longterm=t2m_longterm,
        z_soil=z_soil,
        rprof=rprof,
        sla_calc=sla_calc,
        coeff_maint_zero=coeff_maint_zero,
        maint_resp_slope=maint_resp_slope,
        ext_coeff=ext_coeff,
        is_tree=is_tree,
        dt_sechiba=dt_sechiba,
        flood_frac=flood_frac,
        one_day=one_day,
        retain_step_results=retain_step_results,
        use_jit=use_maintenance_jit,
    )
    missing = tuple(dict.fromkeys((*accumulator.missing_inputs, *maintenance.missing_inputs)))
    daily_fields = dict(accumulator.fields)
    if maintenance.ok:
        daily_fields["resp_maint_part"] = maintenance.resp_maint_part
        if maintenance.resp_maint_radia is not None:
            daily_fields["resp_maint_radia"] = maintenance.resp_maint_radia
        if maintenance.flood_root_radia is not None:
            daily_fields["flood_root_radia"] = maintenance.flood_root_radia
    return StomateDailyProcessFold(
        accumulator=accumulator,
        maintenance=maintenance,
        daily_fields=daily_fields,
        missing_inputs=missing,
        provenance=(
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3187-3267",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3589 and 4225-4364",
        ),
        notes=(
            "Combined fold prepares the daily fields required before do_slow STOMATE carbon processes.",
            "It intentionally stops before season, StomateLpj, OK_LEAK, modelout, and daily reset.",
        ),
    )


def stomate_daily_process_fold_single_pass_from_entries(
    *,
    entry_payloads,
    accumulator_state,
    resp_maint_part_current,
    do_slow_flags,
    dt_sechiba,
    dt_stomate,
    biomass,
    t2m_longterm,
    z_soil,
    rprof,
    sla_calc,
    coeff_maint_zero,
    maint_resp_slope,
    ext_coeff,
    is_tree,
    flood_frac=None,
    peat_occur=False,
    dates=None,
    date=None,
    one_day=86400.0,
    retain_step_results: bool = True,
    use_maintenance_jit: bool = False,
) -> StomateDailyProcessFold:
    """Single-pass runtime fold for the daily accumulator/maintenance boundary."""

    payloads = tuple(entry_payloads)
    flags = tuple(bool(flag) for flag in do_slow_flags)
    if len(payloads) != len(flags):
        raise ValueError("entry_payloads and do_slow_flags must have the same length")
    if not payloads:
        raise ValueError("At least one explicit STOMATE entry payload is required")
    if any(flags[:-1]):
        raise ValueError("A do_slow boundary must be the final entry in this daily fold")
    if dates is None:
        step_dates = (date,) * len(payloads)
    else:
        step_dates = tuple(dates)
        if len(step_dates) != len(payloads):
            raise ValueError("dates must have the same length as entry_payloads")

    accumulator_provenance = (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3187-3240",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_accu_r1d/r2d/r3d lines 9341-9411",
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 656-662",
    )
    maintenance_provenance = (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3244-3267",
        "fortran_source/ORCHIDEE/src_stomate/stomate_resp.f90::maint_respiration lines 122-376",
    )
    acc_step_results: list[StomateFirstStepDailyAccumulation] = []
    maint_step_results: list[MaintenanceRespirationResult] = []
    current_acc_state = accumulator_state
    current_acc_fields: dict[str, jnp.ndarray] = {}
    current_resp = jnp.asarray(resp_maint_part_current)
    resp_maint_radia = None
    flood_root_radia = None

    biomass_arr = jnp.asarray(biomass)
    t2m_longterm_arr = jnp.asarray(t2m_longterm)
    z_soil_arr = jnp.asarray(z_soil)
    rprof_arr = jnp.asarray(rprof)
    sla_calc_arr = jnp.asarray(sla_calc)
    coeff_maint_zero_arr = jnp.asarray(coeff_maint_zero)
    maint_resp_slope_arr = jnp.asarray(maint_resp_slope)
    ext_coeff_arr = jnp.asarray(ext_coeff)
    is_tree_arr = jnp.asarray(is_tree, dtype=bool)
    dt_days = jnp.asarray(dt_sechiba) / jnp.asarray(one_day)
    batch_maintenance = not retain_step_results and not use_maintenance_jit
    static_cache = None
    t2m_values = []
    stempdiag_values = []
    if biomass_arr.ndim != 4:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    if biomass_arr.shape[2] != 12:
        raise ValueError("biomass pool axis must have length 12")
    if z_soil_arr.ndim != 1:
        raise ValueError("z_soil must be one-dimensional")
    if batch_maintenance:
        static_cache = _maintenance_static_cache(
            biomass=biomass_arr,
            t2m_longterm=t2m_longterm_arr,
            z_soil=z_soil_arr,
            rprof=rprof_arr,
            sla_calc=sla_calc_arr,
            coeff_maint_zero=coeff_maint_zero_arr,
            maint_resp_slope=maint_resp_slope_arr,
            ext_coeff=ext_coeff_arr,
            is_tree=is_tree_arr,
            dt_days=dt_days,
        )

    for index, (payload, do_slow, step_date) in enumerate(zip(payloads, flags, step_dates, strict=True)):
        acc_result = stomate_first_step_daily_accumulation_from_entry(
            entry_payload=payload,
            accumulator_state=current_acc_state,
            dt_sechiba=dt_sechiba,
            dt_stomate=dt_stomate,
            do_slow=do_slow,
            peat_occur=peat_occur,
            date=step_date,
        )
        if retain_step_results:
            acc_step_results.append(acc_result)
        if acc_result.missing_inputs:
            accumulator = StomateDailyAccumulationFold(
                fields=current_acc_fields,
                step_results=tuple(acc_step_results),
                missing_inputs=acc_result.missing_inputs,
                failed_step=index,
                provenance=accumulator_provenance,
                notes=("Daily fold stopped without fabricating missing STOMATE entry inputs.",),
            )
            maintenance = StomateMaintenanceFold(
                resp_maint_part=current_resp,
                step_results=tuple(maint_step_results),
                resp_maint_radia=resp_maint_radia,
                flood_root_radia=flood_root_radia,
                missing_inputs=(),
                failed_step=None,
                provenance=maintenance_provenance,
                notes=("Maintenance fold contains steps completed before the accumulator gap.",),
            )
            return StomateDailyProcessFold(
                accumulator=accumulator,
                maintenance=maintenance,
                daily_fields=dict(current_acc_fields),
                missing_inputs=acc_result.missing_inputs,
                provenance=(
                    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3187-3267",
                    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3589 and 4225-4364",
                ),
                notes=("Single-pass fold stopped before fabricating missing inputs.",),
            )
        current_acc_fields = acc_result.fields
        current_acc_state = _AccumulatorStateView(accumulator_state, current_acc_fields)

        missing_maintenance = tuple(name for name in ("t2m", "stempdiag") if name not in payload)
        if missing_maintenance:
            accumulator = StomateDailyAccumulationFold(
                fields=current_acc_fields,
                step_results=tuple(acc_step_results),
                missing_inputs=(),
                failed_step=None,
                provenance=accumulator_provenance,
                notes=("Daily accumulator completed through the maintenance gap step.",),
            )
            maintenance = StomateMaintenanceFold(
                resp_maint_part=current_resp,
                step_results=tuple(maint_step_results),
                resp_maint_radia=resp_maint_radia,
                flood_root_radia=flood_root_radia,
                missing_inputs=missing_maintenance,
                failed_step=index,
                provenance=maintenance_provenance,
                notes=("Maintenance fold stopped without fabricating missing entry inputs.",),
            )
            daily_fields = dict(current_acc_fields)
            return StomateDailyProcessFold(
                accumulator=accumulator,
                maintenance=maintenance,
                daily_fields=daily_fields,
                missing_inputs=missing_maintenance,
                provenance=(
                    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3187-3267",
                    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3589 and 4225-4364",
                ),
                notes=("Single-pass fold stopped before fabricating missing inputs.",),
            )
        stempdiag_arr = jnp.asarray(payload["stempdiag"])
        if stempdiag_arr.shape[-1] != z_soil_arr.shape[0] - 1:
            raise ValueError("stempdiag last axis must match len(z_soil) - 1")
        if batch_maintenance:
            t2m_values.append(jnp.asarray(payload["t2m"]))
            stempdiag_values.append(stempdiag_arr)
            continue
        if use_maintenance_jit:
            maintenance_step = maintenance_respiration_core_jit(
                biomass_arr,
                jnp.asarray(payload["t2m"]),
                t2m_longterm_arr,
                stempdiag_arr,
                z_soil_arr,
                rprof_arr,
                sla_calc_arr,
                coeff_maint_zero_arr,
                maint_resp_slope_arr,
                ext_coeff_arr,
                is_tree_arr,
                dt_days,
                MIN_STOMATE,
                DEFAULT_MAINT_RESP_MIN_VMAX,
                DEFAULT_MAINT_RESP_COEFF,
            )
        else:
            maintenance_step = _maintenance_respiration_core(
                biomass_arr,
                jnp.asarray(payload["t2m"]),
                t2m_longterm_arr,
                stempdiag_arr,
                z_soil_arr,
                rprof_arr,
                sla_calc_arr,
                coeff_maint_zero_arr,
                maint_resp_slope_arr,
                ext_coeff_arr,
                is_tree_arr,
                dt_days,
                MIN_STOMATE,
                DEFAULT_MAINT_RESP_MIN_VMAX,
                DEFAULT_MAINT_RESP_COEFF,
            )
        if retain_step_results:
            maint_step_results.append(maintenance_step)
        resp_maint_radia, flood_root_radia = sum_resp_maint_radia(
            maintenance_step.resp_maint_part,
            flood_frac=flood_frac,
        )
        current_resp = accumulate_resp_maint_part(current_resp, maintenance_step.resp_maint_part)

    if batch_maintenance:
        resp_parts = _maintenance_respiration_parts_from_static_cache_stack(
            static_cache,
            t2m_stack=jnp.stack(t2m_values, axis=0),
            stempdiag_stack=jnp.stack(stempdiag_values, axis=0),
        )
        for index in range(len(payloads)):
            current_resp = accumulate_resp_maint_part(current_resp, resp_parts[index])
        resp_maint_radia, flood_root_radia = sum_resp_maint_radia(
            resp_parts[-1],
            flood_frac=flood_frac,
        )

    accumulator = StomateDailyAccumulationFold(
        fields=current_acc_fields,
        step_results=tuple(acc_step_results),
        missing_inputs=(),
        failed_step=None,
        provenance=accumulator_provenance,
        notes=("Sequential daily accumulators are advanced through the supplied explicit entry payloads.",),
    )
    maintenance = StomateMaintenanceFold(
        resp_maint_part=current_resp,
        step_results=tuple(maint_step_results),
        resp_maint_radia=resp_maint_radia,
        flood_root_radia=flood_root_radia,
        missing_inputs=(),
        failed_step=None,
        provenance=maintenance_provenance,
        notes=("Sequential maintenance respiration is accumulated through the supplied explicit entry payloads.",),
    )
    daily_fields = dict(current_acc_fields)
    daily_fields["resp_maint_part"] = current_resp
    if resp_maint_radia is not None:
        daily_fields["resp_maint_radia"] = resp_maint_radia
    if flood_root_radia is not None:
        daily_fields["flood_root_radia"] = flood_root_radia
    return StomateDailyProcessFold(
        accumulator=accumulator,
        maintenance=maintenance,
        daily_fields=daily_fields,
        missing_inputs=(),
        provenance=(
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3187-3267",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3589 and 4225-4364",
        ),
        notes=(
            "Single-pass runtime fold preserves entry order while avoiding separate accumulator and maintenance scans.",
            "It intentionally stops before season, StomateLpj, OK_LEAK, modelout, and daily reset.",
        ),
    )
