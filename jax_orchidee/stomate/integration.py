"""Explicit-input STOMATE carbon integration adapters.

The adapters in this module compose audited Phase 1B/1D helpers only when
their process-boundary inputs are supplied explicitly. They do not compute
phenology, allocation ratios, dynamic vegetation, or full STOMATE control
flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, NamedTuple

from jax import config, jit

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jax_orchidee.stomate.carbon_kernels import (
    AllocationStepResult,
    ConstraintsResult,
    CoverResult,
    CrownResult,
    EstablishmentBiomassResult,
    EstablishmentRatesResult,
    GapMortalityResult,
    HarvestAgriResult,
    IABOVE,
    IAGRSAPPN,
    IAGRSAPST,
    IAGRHRTPN,
    IAGRHRTST,
    ICARBON,
    ILEAF,
    IROOT,
    ISAPABOVE,
    ISAPBELOW,
    IWPLCC,
    KillResult,
    LccGrossGlcchangeStepResult,
    LcchangeDeffireResult,
    LcchangeMainAgripeatResult,
    LcchangeMainLeakResult,
    LightCompetitionResult,
    LittercalcLeakCoreResult,
    LpjCoverPeatResult,
    MaintenanceRespirationResult,
    NLITT,
    NPPAgeSLAResult,
    NPPUpdateResult,
    PhenologyResult,
    PrescribeResult,
    StomateLpjOutputDiagnostics,
    TurnoverResult,
    VmaxResult,
    allocation_step,
    constraints_step,
    cover_step,
    crown_step,
    crown_woodmass_ind,
    establishment_biomass_step,
    establishment_rates_step,
    gap_mortality_step,
    harvest_agri_step,
    lcchange_deffire_step,
    lcchange_main_agripeat_step,
    lcchange_main_leak_step,
    kill_pfts_step,
    lcc_gross_glcchange_step,
    light_competition_step,
    lpj_cover_peat_step,
    littercalc_leak_core_with_controls,
    littercalc_leak_core_with_controls_jit,
    maintenance_respiration,
    npp_leaf_age_sla_age_update,
    phenology_step,
    prescribe_step,
    set_lai_from_biomass,
    stomate_lpj_cover_dispatch_step,
    stomate_lpj_litter_availability_from_fraction,
    stomate_lpj_output_diagnostics,
    stomate_littercalc_entry_prep,
    turnover_step,
    vmax_step,
)
from jax_orchidee.stomate.carbon_kernels import npp_closed_update
from jax_orchidee.stomate.daily import (
    DailyCarbonBoundary,
    accumulate_resp_maint_part,
    prepare_daily_carbon_inputs,
    require_explicit_f_alloc,
    stomate_accumulate_daily,
    sum_resp_maint_radia,
)
from jax_orchidee.stomate.daily_inputs import stomate_gpp_daily_increment
from jax_orchidee.stomate.modelout import ModeloutResult, compute_modelout_from_fields, stomate_lpj_history_fields_from_state
from jax_orchidee.stomate.soilcarbon_kernels import (
    DeepCarbonGasDiffusionCoefficients,
    DeepCarbonNonMethaneCoreResult,
    DeepCarbonSnowInterpolResult,
    SoilcarbonDocExportAggregate,
    SoilcarbonCryoturbationCoefficients,
    DeepCarbonCryoturbationCoefficients,
    SoilcarbonLeakCoreResult,
    deep_carbon_altcalc_step,
    deep_carbon_gasdiff_properties_step,
    deep_carbon_nonmethane_core_step,
    deep_carbon_root_depth_step,
    deep_carbon_snow_interpol_step,
    deep_carbon_snowlevels_step,
    deep_carbon_soil_gasdiff_coefficients,
    deep_carbon_soil_gasdiff_diffuse,
    deep_carbon_yedoma_reset_step,
    soilcarbon_leak_core_step,
    soilcarbon_leak_core_step_jit,
    soilcarbon_leak_doc_export_aggregate,
    soilcarbon_leak_tf_doc_ground_fluxes,
)


class ExplicitDailyCarbonResult(NamedTuple):
    """Outputs from explicit maint/alloc/npp boundary composition."""

    boundary: DailyCarbonBoundary
    npp_update: NPPUpdateResult
    age_sla: NPPAgeSLAResult | None
    requires_trace: tuple[str, ...]


class ExplicitDailyCarbonWithAllocResult(NamedTuple):
    """Outputs from source-backed allocation followed by NPP bookkeeping."""

    allocation: AllocationStepResult
    daily_carbon: ExplicitDailyCarbonResult
    lai_after_alloc: jnp.ndarray
    requires_trace: tuple[str, ...]


class ExplicitDailyCarbonTurnoverResult(NamedTuple):
    """Outputs from explicit NPP bookkeeping followed by turnover."""

    daily_carbon: ExplicitDailyCarbonResult
    turnover: TurnoverResult
    requires_trace: tuple[str, ...]


class ExplicitDailyCarbonGapTurnoverResult(NamedTuple):
    """Outputs from explicit NPP bookkeeping followed by gap and turnover."""

    daily_carbon: ExplicitDailyCarbonResult
    gap: GapMortalityResult
    turnover: TurnoverResult
    bm_to_litter: jnp.ndarray
    requires_trace: tuple[str, ...]


class ExplicitDailyCarbonKillGapTurnoverResult(NamedTuple):
    """Outputs from explicit NPP bookkeeping through turn, light, kill."""

    daily_carbon: ExplicitDailyCarbonResult
    kill_after_npp: KillResult
    crown_after_npp: CrownResult
    gap: GapMortalityResult
    kill_after_gap: KillResult
    turnover: TurnoverResult
    light: LightCompetitionResult | None
    kill_after_light: KillResult | None
    establishment_rates: EstablishmentRatesResult | None
    establishment_biomass: EstablishmentBiomassResult | None
    crown_after_establish: CrownResult | None
    cover: CoverResult | LpjCoverPeatResult | None
    harvest_agri: HarvestAgriResult | None
    lai_after_setlai: jnp.ndarray | None
    vmax: VmaxResult | None
    output_diagnostics: StomateLpjOutputDiagnostics | None
    modelout_fields: dict[str, jnp.ndarray] | None
    modelout: ModeloutResult | None
    bm_to_litter: jnp.ndarray
    requires_trace: tuple[str, ...]


class ExplicitDailyCarbonKillGapTurnoverWithAllocResult(NamedTuple):
    """Outputs from allocation through the explicit post-NPP chain."""

    allocation: AllocationStepResult
    post_npp: ExplicitDailyCarbonKillGapTurnoverResult
    lai_after_alloc: jnp.ndarray
    requires_trace: tuple[str, ...]


class ExplicitDailyCarbonPrescribeAllocKillGapTurnoverResult(NamedTuple):
    """Outputs from prescribe through allocation and the post-NPP chain."""

    prescribe: PrescribeResult
    allocation: AllocationStepResult
    post_npp: ExplicitDailyCarbonKillGapTurnoverResult
    lai_after_prescribe: jnp.ndarray
    lai_after_alloc: jnp.ndarray
    requires_trace: tuple[str, ...]


class ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult(NamedTuple):
    """Outputs from prescribe, constraints, allocation, and post-NPP chain."""

    prescribe: PrescribeResult
    constraints: ConstraintsResult
    phenology: PhenologyResult | None
    allocation: AllocationStepResult
    post_npp: ExplicitDailyCarbonKillGapTurnoverResult
    lai_after_prescribe: jnp.ndarray
    lai_after_alloc: jnp.ndarray
    requires_trace: tuple[str, ...]


class ExplicitDailyCarbonScheduledGppPrescribeConstraintsAllocKillGapTurnoverResult(NamedTuple):
    """Outputs from scheduled GPP accumulation through post-NPP chain."""

    gpp_d: jnp.ndarray
    gpp_daily: jnp.ndarray
    chain: ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult
    requires_trace: tuple[str, ...]


class ExplicitDailyCarbonScheduledGppMaintenancePrescribeConstraintsAllocKillGapTurnoverResult(NamedTuple):
    """Outputs from local maintenance/GPP scheduling through post-NPP chain."""

    maintenance: MaintenanceRespirationResult
    resp_maint_radia: jnp.ndarray
    flood_root_radia: jnp.ndarray | None
    resp_maint_part: jnp.ndarray
    scheduled_gpp_chain: ExplicitDailyCarbonScheduledGppPrescribeConstraintsAllocKillGapTurnoverResult
    requires_trace: tuple[str, ...]


class ExplicitOkLeakResult(NamedTuple):
    """Outputs from explicit littercalc -> soilcarbon_leak composition."""

    littercalc: LittercalcLeakCoreResult
    soilcarbon: SoilcarbonLeakCoreResult
    doc_export: SoilcarbonDocExportAggregate
    wet_dep_ground: jnp.ndarray
    wet_dep_flood: jnp.ndarray
    interception_storage: jnp.ndarray
    transfers: OkLeakTransferStepV1 | None
    requires_trace: tuple[str, ...]


class OkLeakTransferStepV1(NamedTuple):
    """Source-resolved amount tensors from one OK_LEAK call."""

    litter_respiration: jnp.ndarray
    litter_flood_respiration: jnp.ndarray
    litter_to_doc: jnp.ndarray
    floodcarbon_input: jnp.ndarray
    poc_gross_decomposition: jnp.ndarray
    poc_flood_gross_decomposition: jnp.ndarray
    doc_gross_decomposition: jnp.ndarray
    doc_flood_gross_decomposition: jnp.ndarray
    doc_to_topsoil: jnp.ndarray
    doc_to_subsoil: jnp.ndarray
    doc_precip2ground: jnp.ndarray
    doc_precip2canopy: jnp.ndarray
    dry_dep_canopy: jnp.ndarray
    wet_dep_ground: jnp.ndarray
    wet_dep_flood: jnp.ndarray
    doc_run: jnp.ndarray
    doc_drain: jnp.ndarray
    doc_flood: jnp.ndarray
    doc_run_2_peat: jnp.ndarray
    doc_free_to_adsorbed: jnp.ndarray
    doc_advective_interface: jnp.ndarray
    doc_diffusive_interface: jnp.ndarray
    cryoturbation_carbon_transfer: jnp.ndarray
    cryoturbation_doc_transfer: jnp.ndarray
    cryoturbation_litter_transfer: jnp.ndarray
    perma_peat_carbon_transfer: jnp.ndarray


class ExplicitOkLeakWithMaintenanceResult(NamedTuple):
    """Outputs from stomate_main local maint/prep -> explicit OK_LEAK."""

    maintenance: MaintenanceRespirationResult
    resp_maint_radia: jnp.ndarray
    flood_root_radia: jnp.ndarray
    resp_maint_part: jnp.ndarray
    ok_leak: ExplicitOkLeakResult
    requires_trace: tuple[str, ...]


class ExplicitOkLeakFromPostNppResult(NamedTuple):
    """Outputs from post-NPP STOMATE state wired into OK_LEAK."""

    prep: object
    ok_leak: ExplicitOkLeakResult
    requires_trace: tuple[str, ...]


class ExplicitOkPcDeepCarbcycleResult(NamedTuple):
    """Outputs from explicit OK_PC ``deep_carbcycle`` daily composition."""

    core: DeepCarbonNonMethaneCoreResult
    altcalc: object
    zi_snow: jnp.ndarray
    zf_snow: jnp.ndarray
    O2_snow: jnp.ndarray
    CH4_snow: jnp.ndarray
    heat_Zimov: jnp.ndarray
    O2_soil: jnp.ndarray
    CH4_soil: jnp.ndarray | None
    airvol_soil: jnp.ndarray
    totporO2_soil: jnp.ndarray
    totporCH4_soil: jnp.ndarray
    diffO2_soil: jnp.ndarray
    diffCH4_soil: jnp.ndarray
    airvol_snow: jnp.ndarray
    totporO2_snow: jnp.ndarray
    totporCH4_snow: jnp.ndarray
    diffO2_snow: jnp.ndarray
    diffCH4_snow: jnp.ndarray
    z_root: jnp.ndarray
    rootlev: jnp.ndarray
    hslong: jnp.ndarray
    heights_snow: jnp.ndarray
    gasdiff_coefficients: DeepCarbonGasDiffusionCoefficients | None
    cryoturbation_coefficients: DeepCarbonCryoturbationCoefficients | None
    yedoma: object
    requires_trace: tuple[str, ...]


class ExplicitOkPcDeepCarbonSidecarState(NamedTuple):
    """Cross-call OK_PC state not present in the main STOMATE restart object."""

    O2_soil: jnp.ndarray
    CH4_soil: jnp.ndarray
    O2_snow: jnp.ndarray
    CH4_snow: jnp.ndarray
    zi_snow: jnp.ndarray
    zf_snow: jnp.ndarray
    airvol_soil: jnp.ndarray
    totporO2_soil: jnp.ndarray
    totporCH4_soil: jnp.ndarray
    diffO2_soil: jnp.ndarray
    diffCH4_soil: jnp.ndarray
    airvol_snow: jnp.ndarray
    totporO2_snow: jnp.ndarray
    totporCH4_snow: jnp.ndarray
    diffO2_snow: jnp.ndarray
    diffCH4_snow: jnp.ndarray
    alt: jnp.ndarray
    alt_ind: jnp.ndarray
    altmax_ind: jnp.ndarray
    altmax_lastyear: jnp.ndarray
    altmax_ind_lastyear: jnp.ndarray
    z_root: jnp.ndarray
    rootlev: jnp.ndarray
    heights_snow: jnp.ndarray
    gasdiff_coefficients: DeepCarbonGasDiffusionCoefficients | None
    cryoturbation_coefficients: DeepCarbonCryoturbationCoefficients | None
    Tref: jnp.ndarray | None = None
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90 module SAVE variables lines 47-155",
        "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90::deep_carbcycle lines 914-1077 consumes previous gas/snow/active-layer SAVE state and writes next-step state",
    )


class ExplicitOkPcFromRestartStateResult(NamedTuple):
    """OK_PC ``deep_carbcycle`` result with main restart and sidecar writeback."""

    ok_pc: ExplicitOkPcDeepCarbcycleResult
    state_after: object
    sidecar_after: ExplicitOkPcDeepCarbonSidecarState
    carbon_surf: jnp.ndarray
    resp_hetero_soil: jnp.ndarray
    heat_Zimov: jnp.ndarray
    sfluxCH4_deep: jnp.ndarray
    sfluxCO2_deep: jnp.ndarray
    requires_trace: tuple[str, ...]


class ExplicitStomateLpjOutputsResult(NamedTuple):
    """End-of-STOMATE output diagnostics and paper modelout fields."""

    output_diagnostics: StomateLpjOutputDiagnostics
    modelout_fields: dict[str, jnp.ndarray]
    modelout: ModeloutResult
    requires_trace: tuple[str, ...]


class ExplicitLccGrossGlcchangeResult(NamedTuple):
    """Gross-LCC result wired from explicit STOMATE restart/season state."""

    lcc: LccGrossGlcchangeStepResult
    state_after: object
    season_after: object
    prod10_total: jnp.ndarray
    prod100_total: jnp.ndarray
    requires_trace: tuple[str, ...]


class ExplicitLcchangeMainLeakResult(NamedTuple):
    """Ordinary non-age-class ``lcchange_main`` result wired to restart state."""

    lcchange: LcchangeMainLeakResult
    state_after: object
    prod10_total: jnp.ndarray
    prod100_total: jnp.ndarray
    requires_trace: tuple[str, ...]


class ExplicitLcchangeDeffireResult(NamedTuple):
    """Non-age-class ``lcchange_deffire`` result wired to restart state."""

    lcchange: LcchangeDeffireResult
    state_after: object
    prod10_total: jnp.ndarray
    prod100_total: jnp.ndarray
    requires_trace: tuple[str, ...]


class ExplicitLcchangeMainAgripeatResult(NamedTuple):
    """Non-age-class ``lcchange_main_agripeat`` result wired to restart state."""

    lcchange: LcchangeMainAgripeatResult
    state_after: object
    prod10_total: jnp.ndarray
    prod100_total: jnp.ndarray
    requires_trace: tuple[str, ...]


class ExplicitDailyLccGrossGlcchangeResult(NamedTuple):
    """Daily ``StomateLpj`` LCC scheduling result for the gross-LCC branch."""

    active: bool
    lcc: ExplicitLccGrossGlcchangeResult | ExplicitLcchangeMainLeakResult | ExplicitLcchangeDeffireResult | ExplicitLcchangeMainAgripeatResult | None
    state_after: object
    season_after: object
    veget_max_after: jnp.ndarray
    do_now_stomate_lcchange_after: bool
    done_stomate_lcchange: bool
    prod10_total: jnp.ndarray
    prod100_total: jnp.ndarray
    requires_trace: tuple[str, ...]


class ExplicitLpjCoverPeatRestartResult(NamedTuple):
    """Dynamic-peat cover update wired back to restart-entry state."""

    cover: LpjCoverPeatResult
    state_after: object
    requires_trace: tuple[str, ...]


def _stomate_lpj_normalize_lcc_veget_max(veget_max, *, min_stomate=1.0e-8):
    """Normalize ``veget_max`` after active LCC as in ``StomateLpj``."""

    veget = jnp.asarray(veget_max)
    total = jnp.sum(veget, axis=1, keepdims=True)
    return jnp.where(total > min_stomate, veget / total, jnp.zeros_like(veget))


def _require_explicit_daily_lcc_inputs(inputs: dict[str, object]):
    missing = tuple(name for name, value in inputs.items() if value is None)
    if missing:
        raise ValueError("active gross LCC requires explicit inputs: " + ", ".join(missing))


def apply_lpj_cover_peat_to_restart_state(*, state, cover) -> object:
    """Write ``lpj_cover_peat`` inout fields back to restart-entry state.

    Fortran provenance: ``src_stomate/stomate_lpj.f90::StomateLpj`` lines
    1358-1380 dispatch ``lpj_cover_peat`` when ``update_peatfrac`` is true.
    ``src_stomate/stomate_lpj.f90::lpj_cover_peat`` lines 2583-3337 updates
    cover, live biomass/state flags, litter/fuel/carbon pools, deep-carbon
    pools, daily flux memories, and peat save buffers. This helper only writes
    fields that the caller's restart-state object actually owns; sidecar save
    buffers such as ``carbon_save`` remain on the returned cover result unless
    the state contract explicitly carries them.
    """

    updates = {
        "veget_max": cover.veget_max,
        "biomass": cover.biomass,
        "ind": cover.ind,
        "age": cover.age,
        "pft_present": cover.pft_present,
        "senescence": cover.senescence,
        "when_growthinit": cover.when_growthinit,
        "everywhere": cover.everywhere,
        "leaf_frac": cover.leaf_frac,
        "lm_lastyearmax": cover.lm_lastyearmax,
        "npp_longterm": cover.npp_longterm,
        "litter": cover.litter,
        "litter_avail": cover.litter_avail,
        "litter_not_avail": cover.litter_not_avail,
        "carbon": cover.carbon,
        "fuel_1hr": cover.fuel_1hr,
        "fuel_10hr": cover.fuel_10hr,
        "fuel_100hr": cover.fuel_100hr,
        "fuel_1000hr": cover.fuel_1000hr,
        "turnover_daily": cover.turnover_daily,
        "bm_to_litter": cover.bm_to_litter,
        "co2_to_bm": cover.co2_to_bm,
        "co2_fire": cover.co2_fire,
        "resp_hetero": cover.resp_hetero,
        "resp_maint": cover.resp_maint,
        "resp_growth": cover.resp_growth,
        "gpp_daily": cover.gpp_daily,
        "deepC_a": cover.deepC_a,
        "deepC_s": cover.deepC_s,
        "deepC_p": cover.deepC_p,
        "soilc_total": cover.deepC_a + cover.deepC_s + cover.deepC_p,
        "carbon_save": cover.carbon_save,
        "deepC_a_save": cover.deepC_a_save,
        "deepC_s_save": cover.deepC_s_save,
        "deepC_p_save": cover.deepC_p_save,
        "delta_fsave": cover.delta_fsave,
    }
    if not hasattr(state, "_fields"):
        raise TypeError("state must be a NamedTuple-like restart state with _replace")
    updates = {name: value for name, value in updates.items() if name in state._fields}
    return state._replace(**updates)


def stomate_lpj_cover_peat_from_restart_state(
    *,
    state,
    veget_max_new,
    carbon_save,
    deepC_a_save,
    deepC_s_save,
    deepC_p_save,
    delta_fsave,
    liqwt_max_lastyear,
    natural,
    pasture,
    is_peat,
    is_tree,
    bm_sapl,
    cn_sapl,
    cn_ind,
    dt_days,
    veget_max_current=None,
    veget_max_old=None,
    litter_avail=None,
    litter_not_avail=None,
    ok_dgvm=False,
    ok_pc=False,
    min_stomate=1.0e-8,
    npp_longterm_init=10.0,
) -> ExplicitLpjCoverPeatRestartResult:
    """Wire restart state through the dynamic-peat cover branch.

    Fortran provenance: ``src_stomate/stomate_lpj.f90::StomateLpj`` lines
    1358-1380 and ``lpj_cover_peat`` lines 2583-3337. The caller must provide
    the peat SAVE/sidecar buffers explicitly because they are not part of the
    base paper restart-entry contract.
    """

    biomass = jnp.asarray(state.biomass)
    npts, nvm = biomass.shape[:2]
    if veget_max_current is None:
        if not hasattr(state, "veget_max"):
            raise ValueError("dynamic-peat cover requires explicit veget_max_current when state has no veget_max field")
        veget_max_current = state.veget_max
    if veget_max_old is None:
        veget_max_old = veget_max_current
    if litter_avail is None:
        litter_avail = getattr(state, "litter_avail", jnp.zeros((npts, NLITT, nvm), dtype=biomass.dtype))
    if litter_not_avail is None:
        litter_not_avail = getattr(state, "litter_not_avail", jnp.zeros((npts, NLITT, nvm), dtype=biomass.dtype))

    cover = lpj_cover_peat_step(
        cn_ind=cn_ind,
        ind=state.ind,
        biomass=state.biomass,
        veget_max_new=veget_max_new,
        veget_max=veget_max_current,
        veget_max_old=veget_max_old,
        litter=state.litter,
        litter_avail=litter_avail,
        litter_not_avail=litter_not_avail,
        carbon=state.carbon,
        fuel_1hr=state.fuel_1hr,
        fuel_10hr=state.fuel_10hr,
        fuel_100hr=state.fuel_100hr,
        fuel_1000hr=state.fuel_1000hr,
        turnover_daily=state.turnover_daily,
        bm_to_litter=state.bm_to_litter,
        co2_to_bm=state.co2_to_bm,
        co2_fire=getattr(state, "co2_fire", jnp.zeros((npts, nvm), dtype=biomass.dtype)),
        resp_hetero=getattr(state, "resp_hetero", jnp.zeros((npts, nvm), dtype=biomass.dtype)),
        resp_maint=state.resp_maint,
        resp_growth=state.resp_growth,
        gpp_daily=state.gpp_daily,
        deepC_a=state.deepC_a,
        deepC_s=state.deepC_s,
        deepC_p=state.deepC_p,
        dt_days=dt_days,
        age=state.age,
        pft_present=state.pft_present,
        senescence=state.senescence,
        when_growthinit=state.when_growthinit,
        everywhere=state.everywhere,
        leaf_frac=state.leaf_frac,
        lm_lastyearmax=state.lm_lastyearmax,
        npp_longterm=state.npp_longterm,
        carbon_save=carbon_save,
        deepC_a_save=deepC_a_save,
        deepC_s_save=deepC_s_save,
        deepC_p_save=deepC_p_save,
        delta_fsave=delta_fsave,
        liqwt_max_lastyear=liqwt_max_lastyear,
        natural=natural,
        pasture=pasture,
        is_peat=is_peat,
        is_tree=is_tree,
        bm_sapl=bm_sapl,
        cn_sapl=cn_sapl,
        ok_dgvm=ok_dgvm,
        ok_pc=ok_pc,
        min_stomate=min_stomate,
        npp_longterm_init=npp_longterm_init,
    )
    return ExplicitLpjCoverPeatRestartResult(
        cover=cover,
        state_after=apply_lpj_cover_peat_to_restart_state(state=state, cover=cover),
        requires_trace=(),
    )


def apply_lcc_gross_glcchange_to_restart_state(*, state, season, lcc):
    """Apply source-backed gross-LCC inout fields to restart/season state.

    Fortran provenance: ``src_stomate/stomate_lpj.f90::StomateLpj`` lines
    1413-1548 calls ``gross_glcchange_fh`` and then derives product totals at
    lines 1546-1548. ``src_stomate/stomate_glcchange_fh.f90::
    gross_glcchange_fh`` lines 1587-1658 updates biomass, litter, legacy
    carbon/deepC, product pools, and PFT/season memories through the receiver
    loop; lines 1669-1697 age the product pools and product fluxes.
    """

    applied = lcc.applied
    updated = applied.state
    state_after = state._replace(
        biomass=updated.biomass,
        bm_to_litter=updated.bm_to_litter,
        carbon=updated.carbon,
        litter=updated.litter,
        lignin_struc=updated.lignin_struc,
        deepC_a=updated.deepC_a,
        deepC_s=updated.deepC_s,
        deepC_p=updated.deepC_p,
        soilc_total=updated.deepC_a + updated.deepC_s + updated.deepC_p,
        fuel_1hr=updated.fuel_1hr,
        fuel_10hr=updated.fuel_10hr,
        fuel_100hr=updated.fuel_100hr,
        fuel_1000hr=updated.fuel_1000hr,
        co2_to_bm=updated.co2_to_bm,
        gpp_daily=updated.gpp_daily,
        npp_daily=updated.npp_daily,
        resp_maint=updated.resp_maint,
        resp_growth=updated.resp_growth,
        prod10=applied.prod10,
        prod100=applied.prod100,
        flux10=applied.flux10,
        flux100=applied.flux100,
        prod10_total=jnp.sum(applied.prod10, axis=(1, 2)),
        prod100_total=jnp.sum(applied.prod100, axis=(1, 2)),
        npp_longterm=updated.npp_longterm,
        ind=updated.ind,
        lm_lastyearmax=updated.lm_lastyearmax,
        age=updated.age,
        everywhere=updated.everywhere,
        leaf_frac=updated.leaf_frac,
        leaf_age=updated.leaf_age,
        pft_present=updated.pft_present,
        senescence=updated.senescence,
        when_growthinit=applied.when_growthinit,
    )
    season_after = season._replace(
        gpp_week=applied.gpp_week,
        gdd_from_growthinit=applied.gdd_from_growthinit,
        gdd_midwinter=applied.gdd_midwinter,
        time_hum_min=applied.time_hum_min,
        gdd_m5_dormance=applied.gdd_m5_dormance,
        ncd_dormance=applied.ncd_dormance,
        moiavail_month=applied.moiavail_month,
        moiavail_week=applied.moiavail_week,
        ngd_minus5=applied.ngd_minus5,
    )
    return state_after, season_after


def stomate_lcc_gross_glcchange_from_restart_state(
    *,
    state,
    season,
    veget_max,
    harvest_matrix,
    glcc_second_shift,
    glcc_primary_shift,
    glcc_net_lcc,
    start_index,
    nagec_pft,
    is_tree,
    natural,
    is_grassland_manag,
    pft_to_mtc,
    agec_group,
    coeff_lcchange_1,
    coeff_lcchange_10,
    coeff_lcchange_100,
    bm_sapl,
    cn_sapl,
    npp_longterm_init,
    resp_hetero,
    co2_fire,
    dt_days,
    one_year=365.0,
    nagec_tree=1,
    nagec_herb=1,
    single_age_class=False,
    allow_deforest_fire=False,
    min_stomate=1.0e-8,
) -> ExplicitLccGrossGlcchangeResult:
    """Wire restart/season state into the source-backed gross-LCC kernel.

    Fortran provenance: ``src_stomate/stomate_lpj.f90::StomateLpj`` lines
    1413-1548 calls ``gross_glcchange_fh`` when land-cover change is active;
    ``src_stomate/stomate_glcchange_fh.f90::gross_glcchange_fh`` lines
    1565-1697 performs first-day allocation, receiver application, and
    product-pool aging. When ``single_age_class`` is true, the allocation
    boundary follows ``stomate_glcchange_SinAgeC_fh.f90`` lines 1358-1950 and
    the downstream receiver/product-pool sequence follows lines 533-905. This
    adapter only maps explicit state fields into that kernel. It deliberately
    requires ``resp_hetero`` and ``co2_fire`` as explicit caller inputs because
    they are process outputs, not restart-entry state fields.
    """

    if resp_hetero is None:
        raise ValueError("resp_hetero must be supplied explicitly for LCC legacy collection")
    if co2_fire is None:
        raise ValueError("co2_fire must be supplied explicitly for LCC legacy collection")

    lcc = lcc_gross_glcchange_step(
        veget_max_org=veget_max,
        harvest_matrix=harvest_matrix,
        glcc_second_shift=glcc_second_shift,
        glcc_primary_shift=glcc_primary_shift,
        glcc_net_lcc=glcc_net_lcc,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        pft_to_mtc=pft_to_mtc,
        agec_group=agec_group,
        biomass=state.biomass,
        bm_to_litter=state.bm_to_litter,
        carbon=state.carbon,
        litter=state.litter,
        deepC_a=state.deepC_a,
        deepC_s=state.deepC_s,
        deepC_p=state.deepC_p,
        fuel_1hr=state.fuel_1hr,
        fuel_10hr=state.fuel_10hr,
        fuel_100hr=state.fuel_100hr,
        fuel_1000hr=state.fuel_1000hr,
        lignin_struc=state.lignin_struc,
        co2_to_bm=state.co2_to_bm,
        gpp_daily=state.gpp_daily,
        npp_daily=state.npp_daily,
        resp_maint=state.resp_maint,
        resp_growth=state.resp_growth,
        resp_hetero=resp_hetero,
        co2_fire=co2_fire,
        convflux=jnp.zeros((jnp.asarray(state.biomass).shape[0], 2), dtype=jnp.asarray(state.biomass).dtype),
        prod10=state.prod10,
        prod100=state.prod100,
        flux10=state.flux10,
        flux100=state.flux100,
        cflux_prod10=jnp.zeros((jnp.asarray(state.biomass).shape[0], 2), dtype=jnp.asarray(state.biomass).dtype),
        cflux_prod100=jnp.zeros((jnp.asarray(state.biomass).shape[0], 2), dtype=jnp.asarray(state.biomass).dtype),
        coeff_lcchange_1=coeff_lcchange_1,
        coeff_lcchange_10=coeff_lcchange_10,
        coeff_lcchange_100=coeff_lcchange_100,
        bm_sapl=bm_sapl,
        cn_sapl=cn_sapl,
        npp_longterm_init=npp_longterm_init,
        npp_longterm=state.npp_longterm,
        ind=state.ind,
        lm_lastyearmax=state.lm_lastyearmax,
        age=state.age,
        everywhere=state.everywhere,
        leaf_frac=state.leaf_frac,
        leaf_age=state.leaf_age,
        pft_present=state.pft_present,
        senescence=state.senescence,
        gpp_week=season.gpp_week,
        when_growthinit=state.when_growthinit,
        gdd_from_growthinit=season.gdd_from_growthinit,
        gdd_midwinter=season.gdd_midwinter,
        time_hum_min=season.time_hum_min,
        gdd_m5_dormance=season.gdd_m5_dormance,
        ncd_dormance=season.ncd_dormance,
        moiavail_month=season.moiavail_month,
        moiavail_week=season.moiavail_week,
        ngd_minus5=season.ngd_minus5,
        dt_days=dt_days,
        one_year=one_year,
        nagec_tree=nagec_tree,
        nagec_herb=nagec_herb,
        single_age_class=single_age_class,
        allow_deforest_fire=allow_deforest_fire,
        min_stomate=min_stomate,
    )
    state_after, season_after = apply_lcc_gross_glcchange_to_restart_state(state=state, season=season, lcc=lcc)

    return ExplicitLccGrossGlcchangeResult(
        lcc=lcc,
        state_after=state_after,
        season_after=season_after,
        prod10_total=jnp.sum(lcc.applied.prod10, axis=(1, 2)),
        prod100_total=jnp.sum(lcc.applied.prod100, axis=(1, 2)),
        requires_trace=(
            "resp_hetero",
            "co2_fire",
            "LCC parameter/mapping arrays for non-paper active branches",
        ),
    )


def stomate_lcchange_main_leak_from_restart_state(
    *,
    state,
    veget_max_new,
    veget_max_old=None,
    is_tree,
    is_grassland_manag,
    is_grassland_grazed=None,
    coeff_lcchange_1,
    coeff_lcchange_10,
    coeff_lcchange_100,
    bm_sapl,
    cn_sapl,
    cn_ind,
    npp_longterm_init,
    dt_days,
    one_year=365.0,
    ok_pc=False,
    min_stomate=1.0e-8,
) -> ExplicitLcchangeMainLeakResult:
    """Wire restart state into ordinary non-age-class ``lcchange_main``.

    Fortran provenance: ``src_stomate/stomate_lpj.f90::StomateLpj`` lines
    1485-1511 call ``src_stomate/stomate_lcchange.f90::lcchange_main`` with
    ``veget_max_new`` as new cover, current ``veget_max`` as old cover, and
    only the ``iwplcc`` product-pool slice. ``lcchange_main`` lines 79-498
    update live biomass, litter/soil/DOC leak pools, product pools, fuel pools,
    PFT state flags, and product flux diagnostics.
    """

    if veget_max_old is None:
        veget_max_old = state.veget_max

    lcc_pool = IWPLCC
    lcchange = lcchange_main_leak_step(
        dt_days=dt_days,
        veget_max=veget_max_new,
        veget_max_old=veget_max_old,
        biomass=state.biomass,
        ind=state.ind,
        age=state.age,
        pft_present=state.pft_present,
        senescence=state.senescence,
        when_growthinit=state.when_growthinit,
        everywhere=state.everywhere,
        co2_to_bm=state.co2_to_bm,
        bm_to_litter=state.bm_to_litter,
        turnover_daily=state.turnover_daily,
        bm_sapl=bm_sapl,
        cn_ind=cn_ind,
        flux10=state.flux10[:, :, lcc_pool],
        flux100=state.flux100[:, :, lcc_pool],
        prod10=state.prod10[:, :, lcc_pool],
        prod100=state.prod100[:, :, lcc_pool],
        leaf_frac=state.leaf_frac,
        npp_longterm=state.npp_longterm,
        lm_lastyearmax=state.lm_lastyearmax,
        litter_avail=getattr(state, "litter_avail", jnp.zeros((jnp.asarray(state.biomass).shape[0], NLITT, jnp.asarray(state.biomass).shape[1]))),
        litter_not_avail=getattr(state, "litter_not_avail", jnp.zeros((jnp.asarray(state.biomass).shape[0], NLITT, jnp.asarray(state.biomass).shape[1]))),
        carbon=state.carbon,
        deepC_a=state.deepC_a,
        deepC_s=state.deepC_s,
        deepC_p=state.deepC_p,
        fuel_1hr=state.fuel_1hr,
        fuel_10hr=state.fuel_10hr,
        fuel_100hr=state.fuel_100hr,
        fuel_1000hr=state.fuel_1000hr,
        litter_above=state.litter_above,
        litter_below=state.litter_below,
        carbon_32l=state.carbon_32l,
        doc=state.DOC,
        cn_sapl=cn_sapl,
        is_tree=is_tree,
        coeff_lcchange_1=coeff_lcchange_1,
        coeff_lcchange_10=coeff_lcchange_10,
        coeff_lcchange_100=coeff_lcchange_100,
        is_grassland_manag=is_grassland_manag,
        is_grassland_grazed=is_grassland_grazed,
        ok_pc=ok_pc,
        min_stomate=min_stomate,
        one_year=one_year,
        npp_longterm_init=npp_longterm_init,
    )

    prod10 = jnp.asarray(state.prod10).at[:, :, lcc_pool].set(lcchange.prod10)
    prod100 = jnp.asarray(state.prod100).at[:, :, lcc_pool].set(lcchange.prod100)
    flux10 = jnp.asarray(state.flux10).at[:, :, lcc_pool].set(lcchange.flux10)
    flux100 = jnp.asarray(state.flux100).at[:, :, lcc_pool].set(lcchange.flux100)
    state_after = state._replace(
        biomass=lcchange.biomass,
        ind=lcchange.ind,
        age=lcchange.age,
        pft_present=lcchange.pft_present,
        senescence=lcchange.senescence,
        when_growthinit=lcchange.when_growthinit,
        everywhere=lcchange.everywhere,
        co2_to_bm=lcchange.co2_to_bm,
        bm_to_litter=lcchange.bm_to_litter,
        turnover_daily=lcchange.turnover_daily,
        prod10=prod10,
        prod100=prod100,
        flux10=flux10,
        flux100=flux100,
        prod10_total=jnp.sum(prod10, axis=(1, 2)),
        prod100_total=jnp.sum(prod100, axis=(1, 2)),
        leaf_frac=lcchange.leaf_frac,
        npp_longterm=lcchange.npp_longterm,
        lm_lastyearmax=lcchange.lm_lastyearmax,
        carbon=lcchange.carbon,
        deepC_a=lcchange.deepC_a,
        deepC_s=lcchange.deepC_s,
        deepC_p=lcchange.deepC_p,
        soilc_total=lcchange.deepC_a + lcchange.deepC_s + lcchange.deepC_p,
        fuel_1hr=lcchange.fuel_1hr,
        fuel_10hr=lcchange.fuel_10hr,
        fuel_100hr=lcchange.fuel_100hr,
        fuel_1000hr=lcchange.fuel_1000hr,
        litter_above=lcchange.litter_above,
        litter_below=lcchange.litter_below,
        carbon_32l=lcchange.carbon_32l,
        DOC=lcchange.doc,
    )

    return ExplicitLcchangeMainLeakResult(
        lcchange=lcchange,
        state_after=state_after,
        prod10_total=state_after.prod10_total,
        prod100_total=state_after.prod100_total,
        requires_trace=(),
    )


def stomate_lcchange_deffire_from_restart_state(
    *,
    state,
    veget_max_new,
    veget_max_old,
    is_tree,
    is_grassland_manag,
    is_grassland_grazed=None,
    coeff_lcchange_1,
    coeff_lcchange_10,
    coeff_lcchange_100,
    bm_sapl,
    cn_sapl,
    cn_ind,
    npp_longterm_init,
    lcc,
    bafrac_deforest_accu,
    emideforest_litter_accu,
    emideforest_biomass_accu,
    deflitsup_total,
    defbiosup_total,
    dt_days,
    one_year=365.0,
    ok_pc=False,
    min_stomate=1.0e-8,
) -> ExplicitLcchangeDeffireResult:
    """Wire restart state into non-age-class ``lcchange_deffire``.

    Fortran provenance: ``src_stomate/stomate_lpj.f90::StomateLpj`` lines
    1469-1483 call ``src_stomate/stomate_lcchange.f90::lcchange_deffire``
    when ``use_age_class=.FALSE.`` and deforestation fire is enabled. The
    fire proxy arrays are created earlier at ``stomate_lpj.f90`` lines
    835-859 and must be supplied explicitly here.
    """

    lcc_pool = IWPLCC
    lcchange = lcchange_deffire_step(
        dt_days=dt_days,
        veget_max=veget_max_old,
        veget_max_new=veget_max_new,
        biomass=state.biomass,
        ind=state.ind,
        age=state.age,
        pft_present=state.pft_present,
        senescence=state.senescence,
        when_growthinit=state.when_growthinit,
        everywhere=state.everywhere,
        co2_to_bm=state.co2_to_bm,
        bm_to_litter=state.bm_to_litter,
        turnover_daily=state.turnover_daily,
        bm_sapl=bm_sapl,
        cn_ind=cn_ind,
        flux10=state.flux10[:, :, lcc_pool],
        flux100=state.flux100[:, :, lcc_pool],
        prod10=state.prod10[:, :, lcc_pool],
        prod100=state.prod100[:, :, lcc_pool],
        leaf_frac=state.leaf_frac,
        npp_longterm=state.npp_longterm,
        lm_lastyearmax=state.lm_lastyearmax,
        litter=state.litter,
        litter_avail=getattr(state, "litter_avail", jnp.zeros((jnp.asarray(state.biomass).shape[0], NLITT, jnp.asarray(state.biomass).shape[1]))),
        litter_not_avail=getattr(state, "litter_not_avail", jnp.zeros((jnp.asarray(state.biomass).shape[0], NLITT, jnp.asarray(state.biomass).shape[1]))),
        carbon=state.carbon,
        deepC_a=state.deepC_a,
        deepC_s=state.deepC_s,
        deepC_p=state.deepC_p,
        fuel_1hr=state.fuel_1hr,
        fuel_10hr=state.fuel_10hr,
        fuel_100hr=state.fuel_100hr,
        fuel_1000hr=state.fuel_1000hr,
        lcc=lcc,
        bafrac_deforest_accu=bafrac_deforest_accu,
        emideforest_litter_accu=emideforest_litter_accu,
        emideforest_biomass_accu=emideforest_biomass_accu,
        deflitsup_total=deflitsup_total,
        defbiosup_total=defbiosup_total,
        cn_sapl=cn_sapl,
        is_tree=is_tree,
        is_grassland_manag=is_grassland_manag,
        is_grassland_grazed=is_grassland_grazed,
        ok_pc=ok_pc,
        min_stomate=min_stomate,
        one_year=one_year,
        npp_longterm_init=npp_longterm_init,
    )
    prod10 = jnp.asarray(state.prod10).at[:, :, lcc_pool].set(lcchange.prod10)
    prod100 = jnp.asarray(state.prod100).at[:, :, lcc_pool].set(lcchange.prod100)
    flux10 = jnp.asarray(state.flux10).at[:, :, lcc_pool].set(lcchange.flux10)
    flux100 = jnp.asarray(state.flux100).at[:, :, lcc_pool].set(lcchange.flux100)
    state_after = state._replace(
        biomass=lcchange.biomass,
        ind=lcchange.ind,
        age=lcchange.age,
        pft_present=lcchange.pft_present,
        senescence=lcchange.senescence,
        when_growthinit=lcchange.when_growthinit,
        everywhere=lcchange.everywhere,
        co2_to_bm=lcchange.co2_to_bm,
        bm_to_litter=lcchange.bm_to_litter,
        turnover_daily=lcchange.turnover_daily,
        prod10=prod10,
        prod100=prod100,
        flux10=flux10,
        flux100=flux100,
        prod10_total=jnp.sum(prod10, axis=(1, 2)),
        prod100_total=jnp.sum(prod100, axis=(1, 2)),
        leaf_frac=lcchange.leaf_frac,
        npp_longterm=lcchange.npp_longterm,
        lm_lastyearmax=lcchange.lm_lastyearmax,
        litter=lcchange.litter,
        carbon=lcchange.carbon,
        deepC_a=lcchange.deepC_a,
        deepC_s=lcchange.deepC_s,
        deepC_p=lcchange.deepC_p,
        soilc_total=lcchange.deepC_a + lcchange.deepC_s + lcchange.deepC_p,
        fuel_1hr=lcchange.fuel_1hr,
        fuel_10hr=lcchange.fuel_10hr,
        fuel_100hr=lcchange.fuel_100hr,
        fuel_1000hr=lcchange.fuel_1000hr,
    )
    if hasattr(state_after, "veget_max"):
        state_after = state_after._replace(veget_max=lcchange.veget_max)

    return ExplicitLcchangeDeffireResult(
        lcchange=lcchange,
        state_after=state_after,
        prod10_total=state_after.prod10_total,
        prod100_total=state_after.prod100_total,
        requires_trace=(),
    )


def stomate_lcchange_main_agripeat_from_restart_state(
    *,
    state,
    veget_max_new,
    veget_max_old=None,
    natural,
    pasture,
    is_peat,
    is_tree,
    pft_to_mtc,
    is_grassland_manag,
    is_grassland_grazed=None,
    coeff_lcchange_1,
    coeff_lcchange_10,
    coeff_lcchange_100,
    bm_sapl,
    cn_sapl,
    cn_ind,
    npp_longterm_init,
    dt_days,
    one_year=365.0,
    agri_peat_prop=False,
    agri_peat_mincrop=False,
    agri_peat_maxcrop=False,
    ok_pc=False,
    min_stomate=1.0e-8,
) -> ExplicitLcchangeMainAgripeatResult:
    """Wire restart state into non-age-class ``lcchange_main_agripeat``.

    Fortran provenance: ``src_stomate/stomate_lpj.f90::StomateLpj`` lines
    1485-1511 dispatch the non-age-class non-fire LCC path and lines
    1494-1505 call ``src_stomate/stomate_lcchange.f90::
    lcchange_main_agripeat`` when ``agri_peat`` is true. The callee's
    hard-coded PFT12-16 agricultural-peat sequence is implemented in
    ``lcchange_main_agripeat`` lines 1253-1613; this adapter only maps
    restart-entry fields to that source-backed kernel and writes its state
    back to the restart object.
    """

    if veget_max_old is None:
        veget_max_old = state.veget_max
    if sum(bool(flag) for flag in (agri_peat_prop, agri_peat_mincrop, agri_peat_maxcrop)) != 1:
        raise ValueError("agri_peat requires exactly one source strategy: agri_peat_prop, agri_peat_mincrop, or agri_peat_maxcrop")

    biomass = jnp.asarray(state.biomass)
    npts, nvm = biomass.shape[:2]
    lcc_pool = IWPLCC
    zero_litter_avail = jnp.zeros((npts, NLITT, nvm), dtype=biomass.dtype)
    lcchange = lcchange_main_agripeat_step(
        dt_days=dt_days,
        veget_max=veget_max_new,
        veget_max_old=veget_max_old,
        biomass=state.biomass,
        ind=state.ind,
        age=state.age,
        pft_present=state.pft_present,
        senescence=state.senescence,
        when_growthinit=state.when_growthinit,
        everywhere=state.everywhere,
        co2_to_bm=state.co2_to_bm,
        bm_to_litter=state.bm_to_litter,
        turnover_daily=state.turnover_daily,
        bm_sapl=bm_sapl,
        cn_ind=cn_ind,
        flux10=state.flux10[:, :, lcc_pool],
        flux100=state.flux100[:, :, lcc_pool],
        prod10=state.prod10[:, :, lcc_pool],
        prod100=state.prod100[:, :, lcc_pool],
        leaf_frac=state.leaf_frac,
        npp_longterm=state.npp_longterm,
        lm_lastyearmax=state.lm_lastyearmax,
        litter=state.litter,
        litter_avail=getattr(state, "litter_avail", zero_litter_avail),
        litter_not_avail=getattr(state, "litter_not_avail", zero_litter_avail),
        carbon=state.carbon,
        deepC_a=state.deepC_a,
        deepC_s=state.deepC_s,
        deepC_p=state.deepC_p,
        fuel_1hr=state.fuel_1hr,
        fuel_10hr=state.fuel_10hr,
        fuel_100hr=state.fuel_100hr,
        fuel_1000hr=state.fuel_1000hr,
        natural=natural,
        pasture=pasture,
        is_peat=is_peat,
        is_tree=is_tree,
        pft_to_mtc=pft_to_mtc,
        coeff_lcchange_1=coeff_lcchange_1,
        coeff_lcchange_10=coeff_lcchange_10,
        coeff_lcchange_100=coeff_lcchange_100,
        cn_sapl=cn_sapl,
        is_grassland_manag=is_grassland_manag,
        is_grassland_grazed=is_grassland_grazed,
        agri_peat_prop=agri_peat_prop,
        agri_peat_mincrop=agri_peat_mincrop,
        agri_peat_maxcrop=agri_peat_maxcrop,
        ok_pc=ok_pc,
        min_stomate=min_stomate,
        one_year=one_year,
        npp_longterm_init=npp_longterm_init,
    )

    prod10 = jnp.asarray(state.prod10).at[:, :, lcc_pool].set(lcchange.prod10)
    prod100 = jnp.asarray(state.prod100).at[:, :, lcc_pool].set(lcchange.prod100)
    flux10 = jnp.asarray(state.flux10).at[:, :, lcc_pool].set(lcchange.flux10)
    flux100 = jnp.asarray(state.flux100).at[:, :, lcc_pool].set(lcchange.flux100)
    updates = {
        "biomass": lcchange.biomass,
        "ind": lcchange.ind,
        "age": lcchange.age,
        "pft_present": lcchange.pft_present,
        "senescence": lcchange.senescence,
        "when_growthinit": lcchange.when_growthinit,
        "everywhere": lcchange.everywhere,
        "co2_to_bm": lcchange.co2_to_bm,
        "bm_to_litter": lcchange.bm_to_litter,
        "turnover_daily": lcchange.turnover_daily,
        "prod10": prod10,
        "prod100": prod100,
        "flux10": flux10,
        "flux100": flux100,
        "prod10_total": jnp.sum(prod10, axis=(1, 2)),
        "prod100_total": jnp.sum(prod100, axis=(1, 2)),
        "leaf_frac": lcchange.leaf_frac,
        "npp_longterm": lcchange.npp_longterm,
        "lm_lastyearmax": lcchange.lm_lastyearmax,
        "litter": lcchange.litter,
        "litter_avail": lcchange.litter_avail,
        "litter_not_avail": lcchange.litter_not_avail,
        "carbon": lcchange.carbon,
        "deepC_a": lcchange.deepC_a,
        "deepC_s": lcchange.deepC_s,
        "deepC_p": lcchange.deepC_p,
        "soilc_total": lcchange.deepC_a + lcchange.deepC_s + lcchange.deepC_p,
        "fuel_1hr": lcchange.fuel_1hr,
        "fuel_10hr": lcchange.fuel_10hr,
        "fuel_100hr": lcchange.fuel_100hr,
        "fuel_1000hr": lcchange.fuel_1000hr,
        "veget_max": lcchange.veget_max_adjusted,
    }
    if hasattr(state, "_fields"):
        updates = {name: value for name, value in updates.items() if name in state._fields}
    state_after = state._replace(**updates)

    return ExplicitLcchangeMainAgripeatResult(
        lcchange=lcchange,
        state_after=state_after,
        prod10_total=state_after.prod10_total,
        prod100_total=state_after.prod100_total,
        requires_trace=(),
    )


def stomate_daily_lcc_gross_glcchange_explicit(
    *,
    state,
    season,
    do_now_stomate_lcchange,
    veget_max=None,
    veget_max_old=None,
    local_prep=None,
    harvest_matrix=None,
    glcc_second_shift=None,
    glcc_primary_shift=None,
    glcc_net_lcc=None,
    start_index=None,
    nagec_pft=None,
    is_tree=None,
    natural=None,
    pasture=None,
    is_peat=None,
    is_grassland_manag=None,
    is_grassland_grazed=None,
    pft_to_mtc=None,
    agec_group=None,
    coeff_lcchange_1=None,
    coeff_lcchange_10=None,
    coeff_lcchange_100=None,
    bm_sapl=None,
    cn_sapl=None,
    cn_ind=None,
    npp_longterm_init=None,
    lcc=None,
    bafrac_deforest_accu=None,
    emideforest_litter_accu=None,
    emideforest_biomass_accu=None,
    deflitsup_total=None,
    defbiosup_total=None,
    resp_hetero=None,
    co2_fire=None,
    dt_days=1.0,
    one_year=365.0,
    nagec_tree=1,
    nagec_herb=1,
    use_age_class=True,
    single_age_class=False,
    allow_deforest_fire=False,
    dyn_peat=False,
    update_peatfrac=False,
    agri_peat=False,
    agri_peat_prop=False,
    agri_peat_mincrop=False,
    agri_peat_maxcrop=False,
    ok_pc=False,
    min_stomate=1.0e-8,
) -> ExplicitDailyLccGrossGlcchangeResult:
    """Schedule the active gross-LCC branch from explicit daily inputs.

    Fortran provenance: ``src_stomate/stomate_lpj.f90::StomateLpj`` lines
    1413-1548 guard LCC with ``do_now_stomate_lcchange``, initialize product
    flux temporaries, dispatch the multi-age ``gross_glcchange_fh`` branch,
    clear ``do_now_stomate_lcchange``, set ``done_stomate_lcchange``, normalize
    ``veget_max``, and derive product totals. The supported active branch here
    is the multi-age gross-LCC call at lines 1446-1464. Its deforestation-fire
    proxy follows lines 835-859, where tree-to-non-tree LCC creates
    ``deforest_biomass_remain`` before the gross-LCC call. The
    ``SingleAgeClass`` branch at lines 1423-1428 uses the
    ``gross_glcchange_SinAgeC_fh`` allocation boundary when requested. The ordinary
    non-age-class ``lcchange_main`` and ``lcchange_deffire`` branches are
    accepted only through explicit source-backed adapters. The hard-coded
    agricultural-peat non-age-class branch follows the explicit
    ``lcchange_main_agripeat`` adapter. The ``dyn_peat`` ordinary non-age-class
    branch follows the source ``IF (.NOT. dyn_peat)`` skip: it consumes the
    daily LCC schedule, normalizes the current cover, and makes no
    ``lcchange_main`` state update. ``update_peatfrac`` belongs to section 13
    before this LCC boundary and must be applied by
    ``stomate_lpj_cover_peat_from_restart_state`` first.
    ``src_stomate/stomate.f90::stomate_main`` lines 2935-2965 normalize GLCC
    matrices before this boundary; callers may pass that result as
    ``local_prep``.
    """

    if local_prep is not None:
        if veget_max is None:
            veget_max = getattr(local_prep, "veget_cov_max")
        if harvest_matrix is None:
            harvest_matrix = getattr(local_prep, "harvest_matrix")
        if glcc_second_shift is None:
            glcc_second_shift = getattr(local_prep, "glccSecondShift")
        if glcc_primary_shift is None:
            glcc_primary_shift = getattr(local_prep, "glccPrimaryShift")
        if glcc_net_lcc is None:
            glcc_net_lcc = getattr(local_prep, "glccNetLCC")

    if veget_max is None:
        veget_max = state.veget_max

    prod10_total = getattr(state, "prod10_total", jnp.sum(state.prod10, axis=(1, 2)))
    prod100_total = getattr(state, "prod100_total", jnp.sum(state.prod100, axis=(1, 2)))

    if not bool(do_now_stomate_lcchange):
        return ExplicitDailyLccGrossGlcchangeResult(
            active=False,
            lcc=None,
            state_after=state,
            season_after=season,
            veget_max_after=jnp.asarray(veget_max),
            do_now_stomate_lcchange_after=False,
            done_stomate_lcchange=False,
            prod10_total=prod10_total,
            prod100_total=prod100_total,
            requires_trace=(),
        )

    if not bool(use_age_class):
        if bool(update_peatfrac):
            raise ValueError(
                "active non-age-class LCC with update_peatfrac requires the source-order "
                "peat cover update to be applied before this LCC boundary"
            )
        if bool(dyn_peat) and not bool(agri_peat) and not bool(allow_deforest_fire):
            current_veget = veget_max_old
            if current_veget is None:
                current_veget = getattr(state, "veget_max", None)
            if current_veget is None:
                current_veget = veget_max
            veget_max_after = _stomate_lpj_normalize_lcc_veget_max(
                current_veget,
                min_stomate=min_stomate,
            )
            prod10 = jnp.asarray(state.prod10).at[:, 0, :].set(0.0)
            prod100 = jnp.asarray(state.prod100).at[:, 0, :].set(0.0)
            updates = {
                "prod10": prod10,
                "prod100": prod100,
                "prod10_total": jnp.sum(prod10, axis=(1, 2)),
                "prod100_total": jnp.sum(prod100, axis=(1, 2)),
                "veget_max": veget_max_after,
            }
            if hasattr(state, "_fields"):
                updates = {name: value for name, value in updates.items() if name in state._fields}
            state_after = state._replace(**updates)
            return ExplicitDailyLccGrossGlcchangeResult(
                active=True,
                lcc=None,
                state_after=state_after,
                season_after=season,
                veget_max_after=veget_max_after,
                do_now_stomate_lcchange_after=False,
                done_stomate_lcchange=True,
                prod10_total=jnp.sum(prod10, axis=(1, 2)),
                prod100_total=jnp.sum(prod100, axis=(1, 2)),
                requires_trace=(),
            )
        _require_explicit_daily_lcc_inputs(
            {
                "veget_max": veget_max,
                "is_tree": is_tree,
                "is_grassland_manag": is_grassland_manag,
                "coeff_lcchange_1": coeff_lcchange_1,
                "coeff_lcchange_10": coeff_lcchange_10,
                "coeff_lcchange_100": coeff_lcchange_100,
                "bm_sapl": bm_sapl,
                "cn_sapl": cn_sapl,
                "cn_ind": cn_ind,
                "npp_longterm_init": npp_longterm_init,
            }
        )
        if veget_max_old is None:
            veget_max_old = getattr(state, "veget_max", None)
        if veget_max_old is None:
            raise ValueError("active non-age-class LCC requires explicit veget_max_old or state.veget_max")
        if bool(agri_peat):
            if bool(allow_deforest_fire):
                raise ValueError("active agri_peat dispatch does not cover allow_deforest_fire")
            _require_explicit_daily_lcc_inputs(
                {
                    "natural": natural,
                    "pasture": pasture,
                    "is_peat": is_peat,
                    "pft_to_mtc": pft_to_mtc,
                }
            )
            agri = stomate_lcchange_main_agripeat_from_restart_state(
                state=state,
                veget_max_new=veget_max,
                veget_max_old=veget_max_old,
                natural=natural,
                pasture=pasture,
                is_peat=is_peat,
                is_tree=is_tree,
                pft_to_mtc=pft_to_mtc,
                is_grassland_manag=is_grassland_manag,
                is_grassland_grazed=is_grassland_grazed,
                coeff_lcchange_1=coeff_lcchange_1,
                coeff_lcchange_10=coeff_lcchange_10,
                coeff_lcchange_100=coeff_lcchange_100,
                bm_sapl=bm_sapl,
                cn_sapl=cn_sapl,
                cn_ind=cn_ind,
                npp_longterm_init=npp_longterm_init,
                dt_days=dt_days,
                one_year=one_year,
                agri_peat_prop=agri_peat_prop,
                agri_peat_mincrop=agri_peat_mincrop,
                agri_peat_maxcrop=agri_peat_maxcrop,
                ok_pc=ok_pc,
                min_stomate=min_stomate,
            )
            veget_max_after = _stomate_lpj_normalize_lcc_veget_max(
                agri.lcchange.veget_max_adjusted,
                min_stomate=min_stomate,
            )
            state_after = agri.state_after
            if hasattr(state_after, "veget_max"):
                state_after = state_after._replace(veget_max=veget_max_after)
                agri = agri._replace(state_after=state_after)
            return ExplicitDailyLccGrossGlcchangeResult(
                active=True,
                lcc=agri,
                state_after=state_after,
                season_after=season,
                veget_max_after=veget_max_after,
                do_now_stomate_lcchange_after=False,
                done_stomate_lcchange=True,
                prod10_total=agri.prod10_total,
                prod100_total=agri.prod100_total,
                requires_trace=agri.requires_trace,
            )
        if bool(allow_deforest_fire):
            _require_explicit_daily_lcc_inputs(
                {
                    "lcc": lcc,
                    "bafrac_deforest_accu": bafrac_deforest_accu,
                    "emideforest_litter_accu": emideforest_litter_accu,
                    "emideforest_biomass_accu": emideforest_biomass_accu,
                    "deflitsup_total": deflitsup_total,
                    "defbiosup_total": defbiosup_total,
                }
            )
            deffire = stomate_lcchange_deffire_from_restart_state(
                state=state,
                veget_max_new=veget_max,
                veget_max_old=veget_max_old,
                is_tree=is_tree,
                is_grassland_manag=is_grassland_manag,
                is_grassland_grazed=is_grassland_grazed,
                coeff_lcchange_1=coeff_lcchange_1,
                coeff_lcchange_10=coeff_lcchange_10,
                coeff_lcchange_100=coeff_lcchange_100,
                bm_sapl=bm_sapl,
                cn_sapl=cn_sapl,
                cn_ind=cn_ind,
                npp_longterm_init=npp_longterm_init,
                lcc=lcc,
                bafrac_deforest_accu=bafrac_deforest_accu,
                emideforest_litter_accu=emideforest_litter_accu,
                emideforest_biomass_accu=emideforest_biomass_accu,
                deflitsup_total=deflitsup_total,
                defbiosup_total=defbiosup_total,
                dt_days=dt_days,
                one_year=one_year,
                ok_pc=ok_pc,
                min_stomate=min_stomate,
            )
            veget_max_after = _stomate_lpj_normalize_lcc_veget_max(
                deffire.lcchange.veget_max,
                min_stomate=min_stomate,
            )
            state_after = deffire.state_after
            if hasattr(state_after, "veget_max"):
                state_after = state_after._replace(veget_max=veget_max_after)
                deffire = deffire._replace(state_after=state_after)
            return ExplicitDailyLccGrossGlcchangeResult(
                active=True,
                lcc=deffire,
                state_after=state_after,
                season_after=season,
                veget_max_after=veget_max_after,
                do_now_stomate_lcchange_after=False,
                done_stomate_lcchange=True,
                prod10_total=deffire.prod10_total,
                prod100_total=deffire.prod100_total,
                requires_trace=deffire.requires_trace,
            )
        ordinary = stomate_lcchange_main_leak_from_restart_state(
            state=state,
            veget_max_new=veget_max,
            veget_max_old=veget_max_old,
            is_tree=is_tree,
            is_grassland_manag=is_grassland_manag,
            is_grassland_grazed=is_grassland_grazed,
            coeff_lcchange_1=coeff_lcchange_1,
            coeff_lcchange_10=coeff_lcchange_10,
            coeff_lcchange_100=coeff_lcchange_100,
            bm_sapl=bm_sapl,
            cn_sapl=cn_sapl,
            cn_ind=cn_ind,
            npp_longterm_init=npp_longterm_init,
            dt_days=dt_days,
            one_year=one_year,
            ok_pc=ok_pc,
            min_stomate=min_stomate,
        )
        veget_max_after = _stomate_lpj_normalize_lcc_veget_max(
            ordinary.lcchange.veget_max,
            min_stomate=min_stomate,
        )
        state_after = ordinary.state_after
        if hasattr(state_after, "veget_max"):
            state_after = state_after._replace(veget_max=veget_max_after)
            ordinary = ordinary._replace(state_after=state_after)
        return ExplicitDailyLccGrossGlcchangeResult(
            active=True,
            lcc=ordinary,
            state_after=state_after,
            season_after=season,
            veget_max_after=veget_max_after,
            do_now_stomate_lcchange_after=False,
            done_stomate_lcchange=True,
            prod10_total=ordinary.prod10_total,
            prod100_total=ordinary.prod100_total,
            requires_trace=ordinary.requires_trace,
        )
    if bool(single_age_class):
        if nagec_pft is None:
            raise ValueError("active gross LCC SingleAgeClass branch requires explicit nagec_pft")
    if bool(update_peatfrac):
        raise ValueError(
            "active gross LCC with update_peatfrac requires the source-order peat cover update "
            "to be applied before this LCC boundary"
        )

    _require_explicit_daily_lcc_inputs(
        {
            "veget_max": veget_max,
            "harvest_matrix": harvest_matrix,
            "glcc_second_shift": glcc_second_shift,
            "glcc_primary_shift": glcc_primary_shift,
            "glcc_net_lcc": glcc_net_lcc,
            "start_index": start_index,
            "nagec_pft": nagec_pft,
            "is_tree": is_tree,
            "natural": natural,
            "is_grassland_manag": is_grassland_manag,
            "pft_to_mtc": pft_to_mtc,
            "agec_group": agec_group,
            "coeff_lcchange_1": coeff_lcchange_1,
            "coeff_lcchange_10": coeff_lcchange_10,
            "coeff_lcchange_100": coeff_lcchange_100,
            "bm_sapl": bm_sapl,
            "cn_sapl": cn_sapl,
            "npp_longterm_init": npp_longterm_init,
            "resp_hetero": resp_hetero,
            "co2_fire": co2_fire,
        }
    )

    lcc = stomate_lcc_gross_glcchange_from_restart_state(
        state=state,
        season=season,
        veget_max=veget_max,
        harvest_matrix=harvest_matrix,
        glcc_second_shift=glcc_second_shift,
        glcc_primary_shift=glcc_primary_shift,
        glcc_net_lcc=glcc_net_lcc,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        pft_to_mtc=pft_to_mtc,
        agec_group=agec_group,
        coeff_lcchange_1=coeff_lcchange_1,
        coeff_lcchange_10=coeff_lcchange_10,
        coeff_lcchange_100=coeff_lcchange_100,
        bm_sapl=bm_sapl,
        cn_sapl=cn_sapl,
        npp_longterm_init=npp_longterm_init,
        resp_hetero=resp_hetero,
        co2_fire=co2_fire,
        dt_days=dt_days,
        one_year=one_year,
        nagec_tree=nagec_tree,
        nagec_herb=nagec_herb,
        single_age_class=single_age_class,
        allow_deforest_fire=allow_deforest_fire,
        min_stomate=min_stomate,
    )
    veget_max_after = _stomate_lpj_normalize_lcc_veget_max(
        lcc.lcc.applied.state.veget_max,
        min_stomate=min_stomate,
    )
    state_after = lcc.state_after
    if hasattr(state_after, "veget_max"):
        state_after = state_after._replace(veget_max=veget_max_after)

    return ExplicitDailyLccGrossGlcchangeResult(
        active=True,
        lcc=lcc,
        state_after=state_after,
        season_after=lcc.season_after,
        veget_max_after=veget_max_after,
        do_now_stomate_lcchange_after=False,
        done_stomate_lcchange=True,
        prod10_total=lcc.prod10_total,
        prod100_total=lcc.prod100_total,
        requires_trace=lcc.requires_trace,
    )


def stomate_daily_carbon_explicit(
    *,
    gpp_daily,
    biomass_before,
    resp_maint_part,
    f_alloc,
    pft_present,
    frac_growthresp,
    dt_days=1.0,
    tax_max=0.8,
    min_stomate=0.0,
    leaf_age=None,
    leaf_frac=None,
    age=None,
    is_tree=None,
    sla_age1=None,
    sla_calc=None,
    sla_max=None,
    sla_min=None,
) -> ExplicitDailyCarbonResult:
    """Run the closed NPP update with explicit process-boundary inputs.

    Fortran provenance: `src_stomate/stomate_lpj.f90`, active order lines
    1093-1131 (`alloc -> setlai -> npp_calc`), and
    `src_stomate/stomate_npp.f90`, `npp_calc` lines 116-180, 230-247,
    280-295, 299-386, 447-531. `f_alloc` must be supplied from an audited
    allocation trace or caller-owned exact allocation output; this adapter never
    fabricates it from restart/history.
    """

    if f_alloc is None:
        raise ValueError("f_alloc must be supplied explicitly from alloc output/trace")

    boundary = prepare_daily_carbon_inputs(
        gpp_daily=gpp_daily,
        biomass=biomass_before,
        resp_maint_part=resp_maint_part,
        pft_present=pft_present,
        f_alloc=f_alloc,
    )
    explicit_f_alloc = require_explicit_f_alloc(boundary)
    biomass_before_arr = jnp.asarray(biomass_before)
    npp_update = npp_closed_update(
        biomass=biomass_before_arr,
        gpp=jnp.asarray(gpp_daily),
        f_alloc=explicit_f_alloc,
        resp_maint_part=jnp.asarray(resp_maint_part),
        pft_present=jnp.asarray(pft_present),
        frac_growthresp=jnp.asarray(frac_growthresp),
        dt_days=dt_days,
        tax_max=tax_max,
        min_stomate=min_stomate,
    )
    age_sla_inputs = {
        "leaf_age": leaf_age,
        "leaf_frac": leaf_frac,
        "age": age,
        "is_tree": is_tree,
        "sla_age1": sla_age1,
        "sla_calc": sla_calc,
        "sla_max": sla_max,
        "sla_min": sla_min,
    }
    if all(value is None for value in age_sla_inputs.values()):
        age_sla = None
    else:
        missing = tuple(name for name, value in age_sla_inputs.items() if value is None)
        if missing:
            raise ValueError(f"NPP age/SLA bookkeeping requires explicit {', '.join(missing)}")
        age_sla = npp_leaf_age_sla_age_update(
            npp_update.biomass,
            npp_update.biomass_before_alloc,
            npp_update.bm_alloc,
            leaf_age,
            leaf_frac,
            age,
            pft_present,
            is_tree,
            sla_age1,
            sla_calc,
            sla_max,
            sla_min,
            dt_days=dt_days,
            min_stomate=min_stomate,
        )
    return ExplicitDailyCarbonResult(
        boundary=boundary,
        npp_update=npp_update,
        age_sla=age_sla,
        requires_trace=(),
    )


def stomate_daily_carbon_with_alloc_explicit(
    *,
    gpp_daily,
    biomass_before,
    resp_maint_part,
    pft_present,
    frac_growthresp,
    lai,
    veget_max,
    senescence,
    when_growthinit,
    moiavail_week,
    tsoil_month,
    soilhum_month,
    age,
    leaf_age,
    leaf_frac,
    z_soil,
    sla_calc,
    natural,
    pasture,
    is_tree,
    ok_LAIdev,
    r0,
    s0,
    ext_coeff,
    lai_max,
    lai_max_to_happy,
    tau_leafinit,
    alloc_min,
    alloc_max,
    demi_alloc,
    alloc_agr_st,
    alloc_agr_pn,
    sla_age1,
    sla_max,
    sla_min,
    when_growthinit_cut=None,
    is_grassland_manag=None,
    dt_days=1.0,
    tax_max=0.8,
    min_stomate=0.0,
    **allocation_kwargs,
) -> ExplicitDailyCarbonWithAllocResult:
    """Compose ``stomate_alloc::alloc`` with ``stomate_npp::npp_calc``.

    Fortran provenance: ``src_stomate/stomate_lpj.f90`` calls allocation at
    lines 1093-1102, updates LAI via ``setlai`` at lines 1118-1120, and then
    calls ``npp_calc`` at lines 1122-1131. Process details are delegated to
    ``stomate_alloc.f90::alloc`` lines 280-817 and
    ``stomate_npp.f90::npp_calc`` lines 116-725. This adapter still requires
    explicit phenology/season/maintenance/GPP inputs and does not invent them.
    """

    allocation = allocation_step(
        lai=lai,
        veget_max=veget_max,
        senescence=senescence,
        when_growthinit=when_growthinit,
        moiavail_week=moiavail_week,
        tsoil_month=tsoil_month,
        soilhum_month=soilhum_month,
        biomass=biomass_before,
        age=age,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        z_soil=z_soil,
        sla_calc=sla_calc,
        natural=natural,
        pasture=pasture,
        is_tree=is_tree,
        ok_LAIdev=ok_LAIdev,
        r0=r0,
        s0=s0,
        ext_coeff=ext_coeff,
        lai_max=lai_max,
        lai_max_to_happy=lai_max_to_happy,
        tau_leafinit=tau_leafinit,
        alloc_min=alloc_min,
        alloc_max=alloc_max,
        demi_alloc=demi_alloc,
        alloc_agr_st=alloc_agr_st,
        alloc_agr_pn=alloc_agr_pn,
        when_growthinit_cut=when_growthinit_cut,
        is_grassland_manag=is_grassland_manag,
        dt_days=dt_days,
        min_stomate=min_stomate,
        **allocation_kwargs,
    )
    lai_after_alloc = set_lai_from_biomass(allocation.biomass, sla_calc)
    daily_carbon = stomate_daily_carbon_explicit(
        gpp_daily=gpp_daily,
        biomass_before=allocation.biomass,
        resp_maint_part=resp_maint_part,
        f_alloc=allocation.f_alloc,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
        dt_days=dt_days,
        tax_max=tax_max,
        min_stomate=min_stomate,
        leaf_age=allocation.leaf_age,
        leaf_frac=allocation.leaf_frac,
        age=age,
        is_tree=is_tree,
        sla_age1=sla_age1,
        sla_calc=sla_calc,
        sla_max=sla_max,
        sla_min=sla_min,
    )
    return ExplicitDailyCarbonWithAllocResult(
        allocation=allocation,
        daily_carbon=daily_carbon,
        lai_after_alloc=lai_after_alloc,
        requires_trace=(),
    )


def stomate_daily_carbon_turnover_explicit(
    *,
    gpp_daily,
    biomass_before,
    resp_maint_part,
    f_alloc,
    pft_present,
    frac_growthresp,
    herbivores,
    maxmoiavail_lastyear,
    minmoiavail_lastyear,
    moiavail_week,
    t2m_longterm,
    t2m_month,
    t2m_week,
    veget_max,
    gdd_from_growthinit,
    leaf_age,
    leaf_frac,
    age,
    lai,
    turnover_time,
    nrec,
    sla_calc,
    senescence_type,
    is_tree,
    natural,
    is_grassland_manag,
    ok_laidev,
    min_leaf_age_for_senescence,
    gdd_senescence,
    senescence_temp,
    hum_frac,
    senescence_hum,
    nosenescence_hum,
    max_turnover_time,
    min_turnover_time,
    leaffall,
    lai_max,
    leafagecrit,
    lai_initmin,
    tau_fruit,
    tau_sap,
    dt_days=1.0,
    tax_max=0.8,
    min_stomate=0.0,
    sla_age1=None,
    sla_max=None,
    sla_min=None,
    prc_residual=0.5,
    ok_herbivores=False,
    ok_dgvm=False,
) -> ExplicitDailyCarbonTurnoverResult:
    """Compose explicit ``npp_calc`` bookkeeping and ``turn`` turnover.

    Fortran provenance: ``src_stomate/stomate_lpj.f90`` lines 1118-1131
    (``npp_calc``) and 1260-1268 (``turn``), with process internals from
    ``src_stomate/stomate_npp.f90`` lines 116-672 and
    ``src_stomate/stomate_turnover.f90`` lines 169-947. All allocation,
    maintenance, climate, season, and PFT parameter inputs remain explicit.
    """

    missing_age_sla = tuple(
        name
        for name, value in {
            "sla_age1": sla_age1,
            "sla_max": sla_max,
            "sla_min": sla_min,
        }.items()
        if value is None
    )
    if missing_age_sla:
        raise ValueError(f"turnover composition requires NPP age/SLA bookkeeping inputs: {', '.join(missing_age_sla)}")

    daily_carbon = stomate_daily_carbon_explicit(
        gpp_daily=gpp_daily,
        biomass_before=biomass_before,
        resp_maint_part=resp_maint_part,
        f_alloc=f_alloc,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
        dt_days=dt_days,
        tax_max=tax_max,
        min_stomate=min_stomate,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        age=age,
        is_tree=is_tree,
        sla_age1=sla_age1,
        sla_calc=sla_calc,
        sla_max=sla_max,
        sla_min=sla_min,
    )
    if daily_carbon.age_sla is None:
        turnover_leaf_age = leaf_age
        turnover_leaf_frac = leaf_frac
        turnover_age = age
        turnover_sla_calc = sla_calc
    else:
        turnover_leaf_age = daily_carbon.age_sla.leaf_age
        turnover_leaf_frac = daily_carbon.age_sla.leaf_frac
        turnover_age = daily_carbon.age_sla.age
        turnover_sla_calc = daily_carbon.age_sla.sla_calc

    turnover = turnover_step(
        pft_present=pft_present,
        herbivores=herbivores,
        maxmoiavail_lastyear=maxmoiavail_lastyear,
        minmoiavail_lastyear=minmoiavail_lastyear,
        moiavail_week=moiavail_week,
        t2m_longterm=t2m_longterm,
        t2m_month=t2m_month,
        t2m_week=t2m_week,
        veget_max=veget_max,
        gdd_from_growthinit=gdd_from_growthinit,
        leaf_age=turnover_leaf_age,
        leaf_frac=turnover_leaf_frac,
        age=turnover_age,
        lai=lai,
        biomass=daily_carbon.npp_update.biomass,
        turnover_time=turnover_time,
        nrec=nrec,
        sla_calc=turnover_sla_calc,
        senescence_type=senescence_type,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        ok_laidev=ok_laidev,
        min_leaf_age_for_senescence=min_leaf_age_for_senescence,
        gdd_senescence=gdd_senescence,
        senescence_temp=senescence_temp,
        hum_frac=hum_frac,
        senescence_hum=senescence_hum,
        nosenescence_hum=nosenescence_hum,
        max_turnover_time=max_turnover_time,
        min_turnover_time=min_turnover_time,
        leaffall=leaffall,
        lai_max=lai_max,
        leafagecrit=leafagecrit,
        lai_initmin=lai_initmin,
        tau_fruit=tau_fruit,
        tau_sap=tau_sap,
        prc_residual=prc_residual,
        ok_herbivores=ok_herbivores,
        ok_dgvm=ok_dgvm,
        dt_days=dt_days,
    )
    return ExplicitDailyCarbonTurnoverResult(
        daily_carbon=daily_carbon,
        turnover=turnover,
        requires_trace=(),
    )


def stomate_daily_carbon_gap_turnover_explicit(
    *,
    gpp_daily,
    biomass_before,
    resp_maint_part,
    f_alloc,
    pft_present,
    frac_growthresp,
    npp_longterm,
    turnover_longterm,
    lm_lastyearmax,
    ind,
    bm_to_litter,
    t2m_min_daily,
    tmin_spring_time,
    herbivores,
    maxmoiavail_lastyear,
    minmoiavail_lastyear,
    moiavail_week,
    t2m_longterm,
    t2m_month,
    t2m_week,
    veget_max,
    gdd_from_growthinit,
    leaf_age,
    leaf_frac,
    age,
    lai,
    turnover_time,
    nrec,
    sla_calc,
    senescence_type,
    is_tree,
    natural,
    pasture,
    is_grassland_manag,
    ok_laidev,
    availability_fact,
    residence_time,
    tmin_crit,
    leaf_tab,
    pheno_type,
    min_leaf_age_for_senescence,
    gdd_senescence,
    senescence_temp,
    hum_frac,
    senescence_hum,
    nosenescence_hum,
    max_turnover_time,
    min_turnover_time,
    leaffall,
    lai_max,
    leafagecrit,
    lai_initmin,
    tau_fruit,
    tau_sap,
    dt_days=1.0,
    tax_max=0.8,
    min_stomate=0.0,
    sla_age1=None,
    sla_max=None,
    sla_min=None,
    lpj_gap_const_mort=False,
    ok_dgvm=False,
    min_avail=0.01,
    ref_greff=0.035,
    npp_longterm_init=10.0,
    coldness_mort=0.01,
    frost_damage_limit=273.15,
    spring_days_max=40,
    prc_residual=0.5,
    ok_herbivores=False,
) -> ExplicitDailyCarbonGapTurnoverResult:
    """Compose explicit ``npp_calc`` bookkeeping, ``gap``, then ``turn``.

    Fortran provenance: ``src_stomate/stomate_lpj.f90`` lines 1118-1131,
    1237-1241, and 1260-1268. Process internals come from
    ``src_stomate/stomate_npp.f90`` lines 116-672,
    ``src_stomate/lpj_gap.f90`` lines 117-364, and
    ``src_stomate/stomate_turnover.f90`` lines 169-947.
    """

    missing_age_sla = tuple(
        name
        for name, value in {
            "sla_age1": sla_age1,
            "sla_max": sla_max,
            "sla_min": sla_min,
        }.items()
        if value is None
    )
    if missing_age_sla:
        raise ValueError(f"gap/turnover composition requires NPP age/SLA bookkeeping inputs: {', '.join(missing_age_sla)}")

    daily_carbon = stomate_daily_carbon_explicit(
        gpp_daily=gpp_daily,
        biomass_before=biomass_before,
        resp_maint_part=resp_maint_part,
        f_alloc=f_alloc,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        age=age,
        sla_calc=sla_calc,
        is_tree=is_tree,
        dt_days=dt_days,
        tax_max=tax_max,
        min_stomate=min_stomate,
        sla_age1=sla_age1,
        sla_max=sla_max,
        sla_min=sla_min,
    )
    gap = gap_mortality_step(
        npp_longterm=npp_longterm,
        turnover_longterm=turnover_longterm,
        lm_lastyearmax=lm_lastyearmax,
        pft_present=pft_present,
        biomass=daily_carbon.npp_update.biomass,
        ind=ind,
        bm_to_litter=bm_to_litter,
        t2m_min_daily=t2m_min_daily,
        tmin_spring_time=tmin_spring_time,
        sla_calc=daily_carbon.age_sla.sla_calc,
        natural=natural,
        is_tree=is_tree,
        pasture=pasture,
        availability_fact=availability_fact,
        residence_time=residence_time,
        tmin_crit=tmin_crit,
        leaf_tab=leaf_tab,
        pheno_type=pheno_type,
        dt_days=dt_days,
        lpj_gap_const_mort=lpj_gap_const_mort,
        ok_dgvm=ok_dgvm,
        min_stomate=min_stomate,
        min_avail=min_avail,
        ref_greff=ref_greff,
        npp_longterm_init=npp_longterm_init,
        coldness_mort=coldness_mort,
        frost_damage_limit=frost_damage_limit,
        spring_days_max=spring_days_max,
    )
    turnover = turnover_step(
        pft_present=pft_present,
        herbivores=herbivores,
        maxmoiavail_lastyear=maxmoiavail_lastyear,
        minmoiavail_lastyear=minmoiavail_lastyear,
        moiavail_week=moiavail_week,
        t2m_longterm=t2m_longterm,
        t2m_month=t2m_month,
        t2m_week=t2m_week,
        veget_max=veget_max,
        gdd_from_growthinit=gdd_from_growthinit,
        leaf_age=daily_carbon.age_sla.leaf_age,
        leaf_frac=daily_carbon.age_sla.leaf_frac,
        age=daily_carbon.age_sla.age,
        lai=lai,
        biomass=gap.biomass,
        turnover_time=turnover_time,
        nrec=nrec,
        sla_calc=daily_carbon.age_sla.sla_calc,
        senescence_type=senescence_type,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        ok_laidev=ok_laidev,
        min_leaf_age_for_senescence=min_leaf_age_for_senescence,
        gdd_senescence=gdd_senescence,
        senescence_temp=senescence_temp,
        hum_frac=hum_frac,
        senescence_hum=senescence_hum,
        nosenescence_hum=nosenescence_hum,
        max_turnover_time=max_turnover_time,
        min_turnover_time=min_turnover_time,
        leaffall=leaffall,
        lai_max=lai_max,
        leafagecrit=leafagecrit,
        lai_initmin=lai_initmin,
        tau_fruit=tau_fruit,
        tau_sap=tau_sap,
        prc_residual=prc_residual,
        ok_herbivores=ok_herbivores,
        ok_dgvm=ok_dgvm,
        dt_days=dt_days,
    )
    return ExplicitDailyCarbonGapTurnoverResult(
        daily_carbon=daily_carbon,
        gap=gap,
        turnover=turnover,
        bm_to_litter=gap.bm_to_litter,
        requires_trace=(),
    )


def stomate_daily_carbon_kill_gap_turnover_explicit(
    *,
    gpp_daily,
    biomass_before,
    resp_maint_part,
    f_alloc,
    pft_present,
    frac_growthresp,
    npp_longterm,
    turnover_longterm,
    lm_lastyearmax,
    ind,
    cn_ind,
    bm_to_litter,
    senescence,
    rip_time,
    when_growthinit,
    everywhere,
    height,
    t2m_min_daily,
    tmin_spring_time,
    herbivores,
    maxmoiavail_lastyear,
    minmoiavail_lastyear,
    moiavail_week,
    t2m_longterm,
    t2m_month,
    t2m_week,
    veget_max,
    gdd_from_growthinit,
    leaf_age,
    leaf_frac,
    age,
    lai,
    turnover_time,
    nrec,
    sla_calc,
    senescence_type,
    is_tree,
    natural,
    pasture,
    is_grassland_manag,
    ok_laidev,
    availability_fact,
    residence_time,
    tmin_crit,
    leaf_tab,
    pheno_type,
    maxdia,
    min_leaf_age_for_senescence,
    gdd_senescence,
    senescence_temp,
    hum_frac,
    senescence_hum,
    nosenescence_hum,
    max_turnover_time,
    min_turnover_time,
    leaffall,
    lai_max,
    leafagecrit,
    lai_initmin,
    tau_fruit,
    tau_sap,
    fpc_max=None,
    maxfpc_lastyear=None,
    veget_lastlight=None,
    ext_coeff=None,
    regenerate=None,
    neighbours=None,
    resolution=None,
    need_adjacent=None,
    precip_lastyear=None,
    gdd0_lastyear=None,
    avail_tree=None,
    avail_grass=None,
    co2_to_bm=None,
    bm_sapl=None,
    cn_sapl=None,
    cover_litter=None,
    cover_litter_avail=None,
    cover_litter_not_avail=None,
    cover_carbon=None,
    cover_fuel_1hr=None,
    cover_fuel_10hr=None,
    cover_fuel_100hr=None,
    cover_fuel_1000hr=None,
    cover_co2_fire=None,
    cover_resp_hetero=None,
    cover_resp_maint=None,
    cover_resp_growth=None,
    cover_gpp_daily=None,
    cover_deepC_a=None,
    cover_deepC_s=None,
    cover_deepC_p=None,
    cover_veget_max_new=None,
    cover_veget_max_old=None,
    cover_carbon_save=None,
    cover_deepC_a_save=None,
    cover_deepC_s_save=None,
    cover_deepC_p_save=None,
    cover_delta_fsave=None,
    cover_liqwt_max_lastyear=None,
    dt_days=1.0,
    tax_max=0.8,
    min_stomate=0.0,
    sla_age1=None,
    sla_max=None,
    sla_min=None,
    lpj_gap_const_mort=False,
    ok_dgvm=False,
    min_avail=0.01,
    ref_greff=0.035,
    npp_longterm_init=10.0,
    coldness_mort=0.01,
    frost_damage_limit=273.15,
    spring_days_max=40,
    prc_residual=0.5,
    ok_herbivores=False,
    large_value=1.0e33,
    fpc_crit=0.95,
    min_cover=0.05,
    annual_increase=True,
    val_exp=999999.0,
    ok_herbivores_establish=False,
    treat_expansion=False,
    migrate=None,
    estab_max_tree=0.12,
    estab_max_grass=0.12,
    establish_scal_fact=5.0,
    max_tree_coverage=0.98,
    ind_0_estab=0.2,
    precip_crit=100.0,
    gdd_crit_estab=150.0,
    regenerate_crit=None,
    pipe_density=2.0e5,
    pipe_tune1=100.0,
    pipe_tune2=40.0,
    pipe_tune3=0.5,
    pipe_tune_exp_coeff=1.6,
    wire_cover=False,
    update_peatfrac=False,
    ok_pc_cover=False,
    is_peat=None,
    is_grassland_grazed=None,
    wire_harvest_agri=False,
    frac_turnover_daily=0.55,
    vcmax25=None,
    n_limfert=None,
    leaf_timecst=None,
    vmax_leafagecrit=None,
    vmax_pheno_type=None,
    vmax_leaf_tab=None,
    vmax_ok_laidev=None,
    wire_vmax=False,
    ok_nlim_vmax=False,
    vmax_offset=0.3,
    leafage_firstmax=0.03,
    leafage_lastmax=0.5,
    leafage_old=1.0,
    output_litter_above=None,
    output_litter_below=None,
    output_carbon_32l=None,
    output_doc=None,
    output_z_soil=None,
    output_zf_soil=None,
    output_carb_mass_total_old=None,
    output_prod10_total=0.0,
    output_prod100_total=0.0,
    output_contfrac=1.0,
    output_one_day=86400.0,
    wire_output_diagnostics=False,
    wire_modelout=False,
) -> ExplicitDailyCarbonKillGapTurnoverResult:
    """Compose explicit post-NPP mortality, turnover, light, and establishment.

    Fortran provenance: ``src_stomate/stomate_lpj.f90`` lines 1118-1148,
    1150-1176, 1237-1250, 1260-1268, 1292-1307, 1300-1318,
    1372-1392, and 1552-1557. Process internals come from
    ``src_stomate/stomate_npp.f90`` lines 116-672,
    ``src_stomate/lpj_kill.f90`` lines 63-272,
    ``src_stomate/lpj_crown.f90`` lines 78-201,
    ``src_stomate/lpj_gap.f90`` lines 117-364, and
    ``src_stomate/stomate_turnover.f90`` lines 169-947, and
    ``src_stomate/lpj_light.f90`` lines 103-648, and
    ``src_stomate/lpj_establish.f90`` lines 100-855,
    ``src_stomate/lpj_cover.f90`` lines 73-387, and
    ``src_stomate/stomate_lpj.f90`` harvest lines 2502-2566, and
    ``src_stomate/stomate_vmax.f90`` lines 105-363. End-of-process
    diagnostic output pool preparation follows ``src_stomate/stomate_lpj.f90``
    lines 1578-1663 when ``wire_output_diagnostics`` is true. Paper modelout
    field mapping follows ``src_stomate/stomate_lpj.f90`` lines 1677-1699 and
    2200-2246 when ``wire_modelout`` is true.
    """

    missing_age_sla = tuple(
        name
        for name, value in {
            "sla_age1": sla_age1,
            "sla_max": sla_max,
            "sla_min": sla_min,
        }.items()
        if value is None
    )
    if missing_age_sla:
        raise ValueError(f"kill/gap/turnover composition requires NPP age/SLA bookkeeping inputs: {', '.join(missing_age_sla)}")

    daily_carbon = stomate_daily_carbon_explicit(
        gpp_daily=gpp_daily,
        biomass_before=biomass_before,
        resp_maint_part=resp_maint_part,
        f_alloc=f_alloc,
        pft_present=pft_present,
        frac_growthresp=frac_growthresp,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        age=age,
        sla_calc=sla_calc,
        is_tree=is_tree,
        dt_days=dt_days,
        tax_max=tax_max,
        min_stomate=min_stomate,
        sla_age1=sla_age1,
        sla_max=sla_max,
        sla_min=sla_min,
    )
    run_pre_gap_kill = bool(ok_dgvm) or not bool(lpj_gap_const_mort)
    if run_pre_gap_kill:
        kill_after_npp = kill_pfts_step(
            lm_lastyearmax=lm_lastyearmax,
            ind=ind,
            pft_present=pft_present,
            cn_ind=cn_ind,
            biomass=daily_carbon.npp_update.biomass,
            senescence=senescence,
            rip_time=rip_time,
            age=daily_carbon.age_sla.age,
            leaf_age=daily_carbon.age_sla.leaf_age,
            leaf_frac=daily_carbon.age_sla.leaf_frac,
            npp_longterm=npp_longterm,
            when_growthinit=when_growthinit,
            everywhere=everywhere,
            veget_max=veget_max,
            bm_to_litter=bm_to_litter,
            natural=natural,
            pasture=pasture,
            is_tree=is_tree,
            ok_dgvm=ok_dgvm,
            lpj_gap_const_mort=lpj_gap_const_mort,
            min_stomate=min_stomate,
            large_value=large_value,
        )
        woodmass_ind = crown_woodmass_ind(
            kill_after_npp.biomass,
            kill_after_npp.ind,
            kill_after_npp.veget_max,
            ok_dgvm=ok_dgvm,
            min_stomate=min_stomate,
        )
        crown_after_npp = crown_step(
            pft_present=kill_after_npp.pft_present,
            ind=kill_after_npp.ind,
            biomass=kill_after_npp.biomass,
            woodmass_ind=woodmass_ind,
            veget_max=kill_after_npp.veget_max,
            height=height,
            is_tree=is_tree,
            natural=natural,
            maxdia=maxdia,
            min_stomate=min_stomate,
        )
    else:
        ind_array = jnp.asarray(ind)
        pft_present_array = jnp.asarray(pft_present, dtype=bool)
        kill_after_npp = KillResult(
            pft_present=pft_present_array,
            cn_ind=jnp.asarray(cn_ind),
            ind=ind_array,
            biomass=daily_carbon.npp_update.biomass,
            senescence=jnp.asarray(senescence, dtype=bool),
            rip_time=jnp.asarray(rip_time),
            age=daily_carbon.age_sla.age,
            leaf_age=daily_carbon.age_sla.leaf_age,
            leaf_frac=daily_carbon.age_sla.leaf_frac,
            npp_longterm=jnp.asarray(npp_longterm),
            when_growthinit=jnp.asarray(when_growthinit),
            everywhere=jnp.asarray(everywhere),
            veget_max=jnp.asarray(veget_max),
            bm_to_litter=jnp.asarray(bm_to_litter),
            was_killed=jnp.zeros_like(pft_present_array, dtype=bool),
        )
        crown_after_npp = CrownResult(
            woodmass_ind=jnp.zeros_like(ind_array),
            cn_ind=kill_after_npp.cn_ind,
            height=jnp.asarray(height),
            veget_max=kill_after_npp.veget_max,
        )
    gap = gap_mortality_step(
        npp_longterm=kill_after_npp.npp_longterm,
        turnover_longterm=turnover_longterm,
        lm_lastyearmax=lm_lastyearmax,
        pft_present=kill_after_npp.pft_present,
        biomass=kill_after_npp.biomass,
        ind=kill_after_npp.ind,
        bm_to_litter=kill_after_npp.bm_to_litter,
        t2m_min_daily=t2m_min_daily,
        tmin_spring_time=tmin_spring_time,
        sla_calc=daily_carbon.age_sla.sla_calc,
        natural=natural,
        is_tree=is_tree,
        pasture=pasture,
        availability_fact=availability_fact,
        residence_time=residence_time,
        tmin_crit=tmin_crit,
        leaf_tab=leaf_tab,
        pheno_type=pheno_type,
        dt_days=dt_days,
        lpj_gap_const_mort=lpj_gap_const_mort,
        ok_dgvm=ok_dgvm,
        min_stomate=min_stomate,
        min_avail=min_avail,
        ref_greff=ref_greff,
        npp_longterm_init=npp_longterm_init,
        coldness_mort=coldness_mort,
        frost_damage_limit=frost_damage_limit,
        spring_days_max=spring_days_max,
    )
    if bool(ok_dgvm):
        kill_after_gap = kill_pfts_step(
            lm_lastyearmax=lm_lastyearmax,
            ind=gap.ind,
            pft_present=kill_after_npp.pft_present,
            cn_ind=kill_after_npp.cn_ind,
            biomass=gap.biomass,
            senescence=kill_after_npp.senescence,
            rip_time=kill_after_npp.rip_time,
            age=kill_after_npp.age,
            leaf_age=kill_after_npp.leaf_age,
            leaf_frac=kill_after_npp.leaf_frac,
            npp_longterm=kill_after_npp.npp_longterm,
            when_growthinit=kill_after_npp.when_growthinit,
            everywhere=kill_after_npp.everywhere,
            veget_max=crown_after_npp.veget_max,
            bm_to_litter=gap.bm_to_litter,
            natural=natural,
            pasture=pasture,
            is_tree=is_tree,
            ok_dgvm=ok_dgvm,
            lpj_gap_const_mort=lpj_gap_const_mort,
            min_stomate=min_stomate,
            large_value=large_value,
        )
    else:
        kill_after_gap = KillResult(
            pft_present=kill_after_npp.pft_present,
            cn_ind=kill_after_npp.cn_ind,
            ind=gap.ind,
            biomass=gap.biomass,
            senescence=kill_after_npp.senescence,
            rip_time=kill_after_npp.rip_time,
            age=kill_after_npp.age,
            leaf_age=kill_after_npp.leaf_age,
            leaf_frac=kill_after_npp.leaf_frac,
            npp_longterm=kill_after_npp.npp_longterm,
            when_growthinit=kill_after_npp.when_growthinit,
            everywhere=kill_after_npp.everywhere,
            veget_max=crown_after_npp.veget_max,
            bm_to_litter=gap.bm_to_litter,
            was_killed=jnp.zeros_like(kill_after_npp.pft_present, dtype=bool),
        )
    turnover = turnover_step(
        pft_present=kill_after_gap.pft_present,
        herbivores=herbivores,
        maxmoiavail_lastyear=maxmoiavail_lastyear,
        minmoiavail_lastyear=minmoiavail_lastyear,
        moiavail_week=moiavail_week,
        t2m_longterm=t2m_longterm,
        t2m_month=t2m_month,
        t2m_week=t2m_week,
        veget_max=kill_after_gap.veget_max,
        gdd_from_growthinit=gdd_from_growthinit,
        leaf_age=kill_after_gap.leaf_age,
        leaf_frac=kill_after_gap.leaf_frac,
        age=kill_after_gap.age,
        lai=lai,
        biomass=kill_after_gap.biomass,
        turnover_time=turnover_time,
        nrec=nrec,
        sla_calc=daily_carbon.age_sla.sla_calc,
        senescence_type=senescence_type,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        ok_laidev=ok_laidev,
        min_leaf_age_for_senescence=min_leaf_age_for_senescence,
        gdd_senescence=gdd_senescence,
        senescence_temp=senescence_temp,
        hum_frac=hum_frac,
        senescence_hum=senescence_hum,
        nosenescence_hum=nosenescence_hum,
        max_turnover_time=max_turnover_time,
        min_turnover_time=min_turnover_time,
        leaffall=leaffall,
        lai_max=lai_max,
        leafagecrit=leafagecrit,
        lai_initmin=lai_initmin,
        tau_fruit=tau_fruit,
        tau_sap=tau_sap,
        prc_residual=prc_residual,
        ok_herbivores=ok_herbivores,
        ok_dgvm=ok_dgvm,
        dt_days=dt_days,
    )
    light = None
    kill_after_light = None
    final_bm_to_litter = kill_after_gap.bm_to_litter
    establish_source_ind = kill_after_gap.ind
    establish_source_pft_present = kill_after_gap.pft_present
    establish_source_cn_ind = kill_after_gap.cn_ind
    establish_source_biomass = turnover.biomass
    establish_source_senescence = turnover.senescence
    establish_source_age = turnover.age
    establish_source_leaf_age = turnover.leaf_age
    establish_source_leaf_frac = turnover.leaf_frac
    establish_source_everywhere = kill_after_gap.everywhere
    establish_source_veget_max = kill_after_gap.veget_max
    if ok_dgvm:
        missing_light = tuple(
            name
            for name, value in {
                "fpc_max": fpc_max,
                "maxfpc_lastyear": maxfpc_lastyear,
                "veget_lastlight": veget_lastlight,
                "ext_coeff": ext_coeff,
            }.items()
            if value is None
        )
        if missing_light:
            raise ValueError(f"DGVM light competition requires explicit {', '.join(missing_light)}")
        light = light_competition_step(
            veget_max=kill_after_gap.veget_max,
            fpc_max=fpc_max,
            pft_present=kill_after_gap.pft_present,
            cn_ind=kill_after_gap.cn_ind,
            lai=turnover.lai,
            maxfpc_lastyear=maxfpc_lastyear,
            lm_lastyearmax=lm_lastyearmax,
            ind=kill_after_gap.ind,
            biomass=turnover.biomass,
            veget_lastlight=veget_lastlight,
            bm_to_litter=kill_after_gap.bm_to_litter,
            mortality=gap.mortality,
            sla_calc=daily_carbon.age_sla.sla_calc,
            natural=natural,
            pasture=pasture,
            is_tree=is_tree,
            ext_coeff=ext_coeff,
            dt_days=dt_days,
            ok_dgvm=True,
            fpc_crit=fpc_crit,
            min_cover=min_cover,
            min_stomate=min_stomate,
            annual_increase=annual_increase,
            val_exp=val_exp,
        )
        kill_after_light = kill_pfts_step(
            lm_lastyearmax=lm_lastyearmax,
            ind=light.ind,
            pft_present=kill_after_gap.pft_present,
            cn_ind=kill_after_gap.cn_ind,
            biomass=light.biomass,
            senescence=turnover.senescence,
            rip_time=kill_after_gap.rip_time,
            age=turnover.age,
            leaf_age=turnover.leaf_age,
            leaf_frac=turnover.leaf_frac,
            npp_longterm=kill_after_gap.npp_longterm,
            when_growthinit=kill_after_gap.when_growthinit,
            everywhere=kill_after_gap.everywhere,
            veget_max=kill_after_gap.veget_max,
            bm_to_litter=light.bm_to_litter,
            natural=natural,
            pasture=pasture,
            is_tree=is_tree,
            ok_dgvm=ok_dgvm,
            lpj_gap_const_mort=lpj_gap_const_mort,
            min_stomate=min_stomate,
            large_value=large_value,
        )
        final_bm_to_litter = kill_after_light.bm_to_litter
        establish_source_ind = kill_after_light.ind
        establish_source_pft_present = kill_after_light.pft_present
        establish_source_cn_ind = kill_after_light.cn_ind
        establish_source_biomass = kill_after_light.biomass
        establish_source_senescence = kill_after_light.senescence
        establish_source_age = kill_after_light.age
        establish_source_leaf_age = kill_after_light.leaf_age
        establish_source_leaf_frac = kill_after_light.leaf_frac
        establish_source_everywhere = kill_after_light.everywhere
        establish_source_veget_max = kill_after_light.veget_max

    establishment_rates = None
    establishment_biomass = None
    crown_after_establish = None
    if ok_dgvm or not lpj_gap_const_mort:
        missing_establish = tuple(
            name
            for name, value in {
                "regenerate": regenerate,
                "neighbours": neighbours,
                "resolution": resolution,
                "need_adjacent": need_adjacent,
                "herbivores": herbivores,
                "precip_lastyear": precip_lastyear,
                "gdd0_lastyear": gdd0_lastyear,
                "avail_tree": avail_tree,
                "avail_grass": avail_grass,
                "co2_to_bm": co2_to_bm,
                "bm_sapl": bm_sapl,
                "ext_coeff": ext_coeff,
            }.items()
            if value is None
        )
        if missing_establish:
            raise ValueError(f"establishment composition requires explicit {', '.join(missing_establish)}")
        if regenerate_crit is None:
            regenerate_crit = 1.0 / jnp.e
        establishment_rates = establishment_rates_step(
            veget_max=establish_source_veget_max,
            pft_present=establish_source_pft_present,
            regenerate=regenerate,
            neighbours=neighbours,
            resolution=resolution,
            need_adjacent=need_adjacent,
            herbivores=herbivores,
            precip_annual=precip_lastyear,
            gdd0=gdd0_lastyear,
            lm_lastyearmax=lm_lastyearmax,
            cn_ind=establish_source_cn_ind,
            lai=turnover.lai,
            avail_tree=avail_tree,
            avail_grass=avail_grass,
            npp_longterm=kill_after_gap.npp_longterm,
            ind=establish_source_ind,
            everywhere=establish_source_everywhere,
            sla_calc=daily_carbon.age_sla.sla_calc,
            natural=natural,
            pasture=pasture,
            is_tree=is_tree,
            ext_coeff=ext_coeff,
            dt_days=dt_days,
            ok_dgvm=ok_dgvm,
            ok_herbivores=ok_herbivores_establish,
            treat_expansion=treat_expansion,
            migrate=migrate,
            min_stomate=min_stomate,
            val_exp=val_exp,
            estab_max_tree=estab_max_tree,
            estab_max_grass=estab_max_grass,
            establish_scal_fact=establish_scal_fact,
            max_tree_coverage=max_tree_coverage,
            ind_0_estab=ind_0_estab,
            precip_crit=precip_crit,
            gdd_crit_estab=gdd_crit_estab,
            regenerate_crit=regenerate_crit,
            min_cover=min_cover,
        )
        establishment_biomass = establishment_biomass_step(
            d_ind=establishment_rates.d_ind,
            ind=establish_source_ind,
            biomass=establish_source_biomass,
            leaf_age=establish_source_leaf_age,
            leaf_frac=establish_source_leaf_frac,
            age=establish_source_age,
            co2_to_bm=co2_to_bm,
            veget_max=establish_source_veget_max,
            woodmass_ind=crown_woodmass_ind(
                establish_source_biomass,
                establish_source_ind,
                establish_source_veget_max,
                ok_dgvm=ok_dgvm,
                min_stomate=min_stomate,
            ),
            mortality=gap.mortality,
            bm_to_litter=final_bm_to_litter,
            bm_sapl=bm_sapl,
            npp_longterm=kill_after_gap.npp_longterm,
            natural=natural,
            pasture=pasture,
            is_tree=is_tree,
            maxdia=maxdia,
            dt_days=dt_days,
            ok_dgvm=ok_dgvm,
            min_stomate=min_stomate,
            pipe_density=pipe_density,
            pipe_tune1=pipe_tune1,
            pipe_tune2=pipe_tune2,
            pipe_tune3=pipe_tune3,
            pipe_tune_exp_coeff=pipe_tune_exp_coeff,
        )
        crown_after_establish = crown_step(
            pft_present=establish_source_pft_present,
            ind=establishment_biomass.ind,
            biomass=establishment_biomass.biomass,
            woodmass_ind=establishment_biomass.woodmass_ind,
            veget_max=establish_source_veget_max,
            height=height,
            is_tree=is_tree,
            natural=natural,
            maxdia=maxdia,
            min_stomate=min_stomate,
            pipe_density=pipe_density,
            pipe_tune1=pipe_tune1,
            pipe_tune2=pipe_tune2,
            pipe_tune3=pipe_tune3,
            pipe_tune_exp_coeff=pipe_tune_exp_coeff,
        )
        final_bm_to_litter = establishment_biomass.bm_to_litter
    cover = None
    if wire_cover:
        missing_cover = tuple(
            name
            for name, value in {
                "cover_litter": cover_litter,
                "cover_litter_avail": cover_litter_avail,
                "cover_litter_not_avail": cover_litter_not_avail,
                "cover_carbon": cover_carbon,
                "cover_fuel_1hr": cover_fuel_1hr,
                "cover_fuel_10hr": cover_fuel_10hr,
                "cover_fuel_100hr": cover_fuel_100hr,
                "cover_fuel_1000hr": cover_fuel_1000hr,
                "cover_co2_fire": cover_co2_fire,
                "cover_resp_hetero": cover_resp_hetero,
                "cover_deepC_a": cover_deepC_a,
                "cover_deepC_s": cover_deepC_s,
                "cover_deepC_p": cover_deepC_p,
                "is_peat": is_peat,
            }.items()
            if value is None
        )
        missing_peat_cover = ()
        if update_peatfrac:
            missing_peat_cover = tuple(
                name
                for name, value in {
                    "cover_veget_max_new": cover_veget_max_new,
                    "cover_carbon_save": cover_carbon_save,
                    "cover_deepC_a_save": cover_deepC_a_save,
                    "cover_deepC_s_save": cover_deepC_s_save,
                    "cover_deepC_p_save": cover_deepC_p_save,
                    "cover_delta_fsave": cover_delta_fsave,
                    "cover_liqwt_max_lastyear": cover_liqwt_max_lastyear,
                    "co2_to_bm": co2_to_bm,
                    "bm_sapl": bm_sapl,
                    "cn_sapl": cn_sapl,
                }.items()
                if value is None
            )
        if missing_cover:
            raise ValueError(f"cover composition requires explicit {', '.join(missing_cover)}")
        if missing_peat_cover:
            raise ValueError(f"peat cover composition requires explicit {', '.join(missing_peat_cover)}")
        cover_source_ind = establishment_biomass.ind if establishment_biomass is not None else establish_source_ind
        cover_source_biomass = establishment_biomass.biomass if establishment_biomass is not None else establish_source_biomass
        cover_source_cn_ind = crown_after_establish.cn_ind if crown_after_establish is not None else establish_source_cn_ind
        cover_source_veget_max = crown_after_establish.veget_max if crown_after_establish is not None else establish_source_veget_max
        cover_source_lai = turnover.lai
        cover_source_turnover = turnover.turnover
        cover_source_bm_to_litter = final_bm_to_litter
        cover_source_co2_to_bm = establishment_biomass.co2_to_bm if establishment_biomass is not None else co2_to_bm
        if cover_veget_max_old is None:
            cover_veget_max_old = veget_max
        cover_inputs = dict(
            cn_ind=cover_source_cn_ind,
            ind=cover_source_ind,
            biomass=cover_source_biomass,
            veget_max=cover_source_veget_max,
            veget_max_old=cover_veget_max_old,
            lai=cover_source_lai,
            litter=cover_litter,
            litter_avail=cover_litter_avail,
            litter_not_avail=cover_litter_not_avail,
            carbon=cover_carbon,
            fuel_1hr=cover_fuel_1hr,
            fuel_10hr=cover_fuel_10hr,
            fuel_100hr=cover_fuel_100hr,
            fuel_1000hr=cover_fuel_1000hr,
            turnover_daily=cover_source_turnover,
            bm_to_litter=cover_source_bm_to_litter,
            co2_to_bm=cover_source_co2_to_bm,
            co2_fire=cover_co2_fire,
            resp_hetero=cover_resp_hetero,
            resp_maint=daily_carbon.npp_update.resp_maint,
            resp_growth=daily_carbon.npp_update.resp_growth,
            gpp_daily=gpp_daily,
            deepC_a=cover_deepC_a,
            deepC_s=cover_deepC_s,
            deepC_p=cover_deepC_p,
            natural=natural,
            pasture=pasture,
            is_peat=is_peat,
            is_grassland_manag=is_grassland_manag,
            is_grassland_grazed=is_grassland_grazed,
            ok_dgvm=ok_dgvm,
            ok_pc=ok_pc_cover,
            min_stomate=min_stomate,
        )
        peat_cover_inputs = None
        if update_peatfrac:
            peat_cover_inputs = dict(
                cn_ind=cover_source_cn_ind,
                ind=cover_source_ind,
                biomass=cover_source_biomass,
                veget_max_new=cover_veget_max_new,
                veget_max=cover_source_veget_max,
                veget_max_old=cover_veget_max_old,
                litter=cover_litter,
                litter_avail=cover_litter_avail,
                litter_not_avail=cover_litter_not_avail,
                carbon=cover_carbon,
                fuel_1hr=cover_fuel_1hr,
                fuel_10hr=cover_fuel_10hr,
                fuel_100hr=cover_fuel_100hr,
                fuel_1000hr=cover_fuel_1000hr,
                turnover_daily=cover_source_turnover,
                bm_to_litter=cover_source_bm_to_litter,
                co2_to_bm=cover_source_co2_to_bm,
                co2_fire=cover_co2_fire,
                resp_hetero=cover_resp_hetero,
                resp_maint=cover_resp_maint,
                resp_growth=cover_resp_growth,
                gpp_daily=cover_gpp_daily,
                deepC_a=cover_deepC_a,
                deepC_s=cover_deepC_s,
                deepC_p=cover_deepC_p,
                dt_days=dt_days,
                age=kill_after_gap.age,
                pft_present=kill_after_gap.pft_present,
                senescence=turnover.senescence,
                when_growthinit=kill_after_gap.when_growthinit,
                everywhere=kill_after_gap.everywhere,
                leaf_frac=turnover.leaf_frac,
                lm_lastyearmax=lm_lastyearmax,
                npp_longterm=kill_after_gap.npp_longterm,
                carbon_save=cover_carbon_save,
                deepC_a_save=cover_deepC_a_save,
                deepC_s_save=cover_deepC_s_save,
                deepC_p_save=cover_deepC_p_save,
                delta_fsave=cover_delta_fsave,
                liqwt_max_lastyear=cover_liqwt_max_lastyear,
                natural=natural,
                pasture=pasture,
                is_peat=is_peat,
                is_tree=is_tree,
                bm_sapl=bm_sapl,
                cn_sapl=cn_sapl,
                is_grassland_manag=is_grassland_manag,
                is_grassland_grazed=is_grassland_grazed,
                ok_dgvm=ok_dgvm,
                ok_pc=ok_pc_cover,
                min_stomate=min_stomate,
                npp_longterm_init=npp_longterm_init,
            )
        cover_dispatch = stomate_lpj_cover_dispatch_step(
            update_peatfrac=update_peatfrac,
            done_update_peatfrac=False,
            peat_cover_inputs=peat_cover_inputs,
            cover_inputs=cover_inputs,
        )
        cover = cover_dispatch.peat_cover if cover_dispatch.used_peat_cover else cover_dispatch.cover
        final_bm_to_litter = cover.bm_to_litter
    harvest_agri = None
    if wire_harvest_agri:
        if is_peat is None:
            raise ValueError("harvest_agri composition requires explicit is_peat")
        harvest_source_veget_max = cover.veget_max if cover is not None else (
            crown_after_establish.veget_max if crown_after_establish is not None else establish_source_veget_max
        )
        harvest_source_turnover = cover.turnover_daily if cover is not None else turnover.turnover
        harvest_source_bm_to_litter = cover.bm_to_litter if cover is not None else final_bm_to_litter
        harvest_agri = harvest_agri_step(
            veget_max=harvest_source_veget_max,
            bm_to_litter=harvest_source_bm_to_litter,
            turnover_daily=harvest_source_turnover,
            natural=natural,
            is_peat=is_peat,
            frac_turnover_daily=frac_turnover_daily,
        )
        final_bm_to_litter = harvest_agri.bm_to_litter
    vmax = None
    setlai_source_biomass = cover.biomass if cover is not None else (
        establishment_biomass.biomass if establishment_biomass is not None else establish_source_biomass
    )
    lai_after_setlai = set_lai_from_biomass(setlai_source_biomass, daily_carbon.age_sla.sla_calc)
    if wire_vmax:
        missing_vmax = tuple(
            name
            for name, value in {
                "vcmax25": vcmax25,
                "n_limfert": n_limfert,
                "leaf_timecst": leaf_timecst,
                "vmax_leafagecrit": vmax_leafagecrit,
                "vmax_pheno_type": vmax_pheno_type,
                "vmax_leaf_tab": vmax_leaf_tab,
                "vmax_ok_laidev": vmax_ok_laidev,
            }.items()
            if value is None
        )
        if missing_vmax:
            raise ValueError(f"vmax composition requires explicit {', '.join(missing_vmax)}")
        vmax_source_leaf_age = (
            establishment_biomass.leaf_age if establishment_biomass is not None else establish_source_leaf_age
        )
        vmax_source_leaf_frac = (
            establishment_biomass.leaf_frac if establishment_biomass is not None else establish_source_leaf_frac
        )
        vmax = vmax_step(
            leaf_age=vmax_source_leaf_age,
            leaf_frac=vmax_source_leaf_frac,
            vcmax25=vcmax25,
            n_limfert=n_limfert,
            leaf_timecst=leaf_timecst,
            leafagecrit=vmax_leafagecrit,
            pheno_type=vmax_pheno_type,
            leaf_tab=vmax_leaf_tab,
            ok_laidev=vmax_ok_laidev,
            dt_days=dt_days,
            ok_dgvm=ok_dgvm,
            ok_nlim=ok_nlim_vmax,
            vmax_offset=vmax_offset,
            leafage_firstmax=leafage_firstmax,
            leafage_lastmax=leafage_lastmax,
            leafage_old=leafage_old,
            min_stomate=min_stomate,
        )
    output_diagnostics = None
    if wire_output_diagnostics:
        missing_output = tuple(
            name
            for name, value in {
                "output_litter_above": output_litter_above,
                "output_litter_below": output_litter_below,
                "output_carbon_32l": output_carbon_32l,
                "output_doc": output_doc,
                "output_z_soil": output_z_soil,
                "output_zf_soil": output_zf_soil,
                "output_carb_mass_total_old": output_carb_mass_total_old,
            }.items()
            if value is None
        )
        if missing_output:
            raise ValueError(f"output diagnostics require explicit {', '.join(missing_output)}")
        output_source_biomass = cover.biomass if cover is not None else (
            establishment_biomass.biomass if establishment_biomass is not None else establish_source_biomass
        )
        output_source_veget_max = cover.veget_max if cover is not None else (
            crown_after_establish.veget_max if crown_after_establish is not None else establish_source_veget_max
        )
        output_source_turnover = harvest_agri.turnover_daily if harvest_agri is not None else (
            cover.turnover_daily if cover is not None else turnover.turnover
        )
        output_source_bm_to_litter = harvest_agri.bm_to_litter if harvest_agri is not None else (
            cover.bm_to_litter if cover is not None else final_bm_to_litter
        )
        output_diagnostics = stomate_lpj_output_diagnostics(
            biomass=output_source_biomass,
            turnover_daily=output_source_turnover,
            bm_to_litter=output_source_bm_to_litter,
            litter_above=output_litter_above,
            litter_below=output_litter_below,
            carbon_32l=output_carbon_32l,
            doc=output_doc,
            veget_max=output_source_veget_max,
            z_soil=output_z_soil,
            zf_soil=output_zf_soil,
            carb_mass_total_old=output_carb_mass_total_old,
            prod10_total=output_prod10_total,
            prod100_total=output_prod100_total,
            contfrac=output_contfrac,
            one_day=output_one_day,
        )
    modelout_fields = None
    modelout = None
    if wire_modelout:
        modelout_source_biomass = cover.biomass if cover is not None else (
            establishment_biomass.biomass if establishment_biomass is not None else establish_source_biomass
        )
        modelout_fields = stomate_lpj_history_fields_from_state(
            biomass=modelout_source_biomass,
            gpp_daily=gpp_daily,
            npp_daily=daily_carbon.npp_update.npp,
        )
        modelout = compute_modelout_from_fields(modelout_fields)
    return ExplicitDailyCarbonKillGapTurnoverResult(
        daily_carbon=daily_carbon,
        kill_after_npp=kill_after_npp,
        crown_after_npp=crown_after_npp,
        gap=gap,
        kill_after_gap=kill_after_gap,
        turnover=turnover,
        light=light,
        kill_after_light=kill_after_light,
        establishment_rates=establishment_rates,
        establishment_biomass=establishment_biomass,
        crown_after_establish=crown_after_establish,
        cover=cover,
        harvest_agri=harvest_agri,
        lai_after_setlai=lai_after_setlai,
        vmax=vmax,
        output_diagnostics=output_diagnostics,
        modelout_fields=modelout_fields,
        modelout=modelout,
        bm_to_litter=final_bm_to_litter,
        requires_trace=(),
    )


def stomate_daily_alloc_kill_gap_turnover_explicit(
    *,
    alloc_inputs,
    post_npp_inputs,
    allocation_kwargs=None,
) -> ExplicitDailyCarbonKillGapTurnoverWithAllocResult:
    """Run source-backed allocation before the existing post-NPP chain.

    Fortran provenance: ``src_stomate/stomate_lpj.f90`` lines 1093-1131 for
    ``alloc -> setlai -> npp_calc`` followed by lines 1137-1557 for the
    post-NPP chain already implemented in
    ``stomate_daily_carbon_kill_gap_turnover_explicit``. This wrapper forbids
    caller-supplied ``f_alloc`` at the post-NPP boundary so allocation fractions
    come only from ``stomate_alloc.f90::alloc``.
    """

    alloc_inputs = dict(alloc_inputs)
    post_npp_inputs = dict(post_npp_inputs)
    allocation_kwargs = {} if allocation_kwargs is None else dict(allocation_kwargs)

    forbidden = ("f_alloc", "biomass_before", "leaf_age", "leaf_frac", "lai")
    overlap = tuple(name for name in forbidden if name in post_npp_inputs)
    if overlap:
        raise ValueError(
            "post_npp_inputs must not supply allocation-owned fields: "
            + ", ".join(overlap)
        )

    allocation = allocation_step(**alloc_inputs, **allocation_kwargs)
    lai_after_alloc = set_lai_from_biomass(allocation.biomass, alloc_inputs["sla_calc"])
    post_npp = stomate_daily_carbon_kill_gap_turnover_explicit(
        **post_npp_inputs,
        biomass_before=allocation.biomass,
        f_alloc=allocation.f_alloc,
        leaf_age=allocation.leaf_age,
        leaf_frac=allocation.leaf_frac,
        lai=lai_after_alloc,
    )
    return ExplicitDailyCarbonKillGapTurnoverWithAllocResult(
        allocation=allocation,
        post_npp=post_npp,
        lai_after_alloc=lai_after_alloc,
        requires_trace=(),
    )


def stomate_daily_prescribe_alloc_kill_gap_turnover_explicit(
    *,
    prescribe_inputs,
    alloc_inputs,
    post_npp_inputs,
    allocation_kwargs=None,
) -> ExplicitDailyCarbonPrescribeAllocKillGapTurnoverResult:
    """Run ``prescribe -> alloc`` before the explicit post-NPP chain.

    Fortran provenance: ``src_stomate/stomate_lpj.f90`` lines 944-951 call
    ``stomate_prescribe::prescribe`` before phenology/allocation. The local
    prescribed/static vegetation behavior is from
    ``src_stomate/stomate_prescribe.f90`` lines 126-346; allocation and
    downstream post-NPP provenance follows
    ``stomate_daily_alloc_kill_gap_turnover_explicit``.

    This is still an explicit-input adapter: phenology outputs and season
    variables must be supplied in ``alloc_inputs``/``post_npp_inputs``. It only
    prevents caller-owned state from bypassing the prescribed-vegetation update.
    """

    prescribe_inputs = dict(prescribe_inputs)
    alloc_inputs = dict(alloc_inputs)
    post_npp_inputs = dict(post_npp_inputs)
    allocation_kwargs = {} if allocation_kwargs is None else dict(allocation_kwargs)

    forbidden_alloc = ("biomass", "leaf_frac", "veget_max")
    forbidden_post = ("biomass_before", "f_alloc", "pft_present", "leaf_frac", "ind", "cn_ind", "when_growthinit", "everywhere", "lai")
    overlap_alloc = tuple(name for name in forbidden_alloc if name in alloc_inputs)
    overlap_post = tuple(name for name in forbidden_post if name in post_npp_inputs)
    if overlap_alloc:
        raise ValueError("alloc_inputs must not supply prescribe-owned fields: " + ", ".join(overlap_alloc))
    if overlap_post:
        raise ValueError("post_npp_inputs must not supply prescribe/allocation-owned fields: " + ", ".join(overlap_post))

    prescribe = prescribe_step(**prescribe_inputs)
    lai_after_prescribe = set_lai_from_biomass(prescribe.biomass, alloc_inputs["sla_calc"])
    allocation = allocation_step(
        **alloc_inputs,
        veget_max=prescribe_inputs["veget_max"],
        biomass=prescribe.biomass,
        leaf_frac=prescribe.leaf_frac,
        **allocation_kwargs,
    )
    lai_after_alloc = set_lai_from_biomass(allocation.biomass, alloc_inputs["sla_calc"])
    post_npp = stomate_daily_carbon_kill_gap_turnover_explicit(
        **post_npp_inputs,
        biomass_before=allocation.biomass,
        f_alloc=allocation.f_alloc,
        pft_present=prescribe.pft_present,
        leaf_age=allocation.leaf_age,
        leaf_frac=allocation.leaf_frac,
        ind=prescribe.ind,
        cn_ind=prescribe.cn_ind,
        when_growthinit=prescribe.when_growthinit,
        everywhere=prescribe.everywhere,
        lai=lai_after_alloc,
    )
    return ExplicitDailyCarbonPrescribeAllocKillGapTurnoverResult(
        prescribe=prescribe,
        allocation=allocation,
        post_npp=post_npp,
        lai_after_prescribe=lai_after_prescribe,
        lai_after_alloc=lai_after_alloc,
        requires_trace=(),
    )


def stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit(
    *,
    prescribe_inputs,
    constraints_inputs,
    phenology_inputs=None,
    alloc_inputs,
    post_npp_inputs,
    constraints_kwargs=None,
    phenology_kwargs=None,
    allocation_kwargs=None,
) -> ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult:
    """Run ``prescribe -> constraints -> phenology -> alloc`` before post-NPP.

    Fortran provenance: ``src_stomate/stomate_lpj.f90`` lines 944-960 calls
    ``prescribe`` and then ``constraints`` before phenology; lines 1068-1102
    then call ``phenology`` followed by ``alloc``.
    ``lpj_constraints.f90::constraints`` lines 99-242 updates ``adapted`` and
    ``regenerate``. ``stomate_phenology.f90::phenology`` lines 319-563 is
    wired for the audited ``pheno_model='none'`` active-PFT path. Allocation and
    downstream provenance follows
    ``stomate_daily_prescribe_alloc_kill_gap_turnover_explicit``.

    This remains an explicit-input adapter for season accumulators, maintenance
    respiration, and GPP scheduling. It prevents callers from bypassing the
    source-backed ``regenerate`` state when downstream establishment is active.
    """

    prescribe_inputs = dict(prescribe_inputs)
    constraints_inputs = dict(constraints_inputs)
    phenology_inputs = None if phenology_inputs is None else dict(phenology_inputs)
    alloc_inputs = dict(alloc_inputs)
    post_npp_inputs = dict(post_npp_inputs)
    constraints_kwargs = {} if constraints_kwargs is None else dict(constraints_kwargs)
    phenology_kwargs = {} if phenology_kwargs is None else dict(phenology_kwargs)
    allocation_kwargs = {} if allocation_kwargs is None else dict(allocation_kwargs)

    forbidden_constraints = ("when_growthinit",)
    forbidden_phenology = ("pft_present", "when_growthinit", "biomass", "leaf_frac", "co2_to_bm")
    forbidden_alloc = ("biomass", "leaf_frac", "veget_max", "when_growthinit")
    forbidden_post = (
        "biomass_before",
        "f_alloc",
        "pft_present",
        "leaf_frac",
        "ind",
        "cn_ind",
        "when_growthinit",
        "everywhere",
        "lai",
        "regenerate",
    )
    overlap_constraints = tuple(name for name in forbidden_constraints if name in constraints_inputs)
    overlap_phenology = () if phenology_inputs is None else tuple(name for name in forbidden_phenology if name in phenology_inputs)
    overlap_alloc = tuple(name for name in forbidden_alloc if name in alloc_inputs)
    overlap_post = tuple(name for name in forbidden_post if name in post_npp_inputs)
    if overlap_constraints:
        raise ValueError("constraints_inputs must not supply prescribe-owned fields: " + ", ".join(overlap_constraints))
    if overlap_phenology:
        raise ValueError("phenology_inputs must not supply prescribe-owned fields: " + ", ".join(overlap_phenology))
    if overlap_alloc:
        raise ValueError("alloc_inputs must not supply prescribe/constraints/phenology-owned fields: " + ", ".join(overlap_alloc))
    if overlap_post:
        raise ValueError(
            "post_npp_inputs must not supply prescribe/constraints/allocation-owned fields: "
            + ", ".join(overlap_post)
        )

    prescribe = prescribe_step(**prescribe_inputs)
    constraints = constraints_step(
        **constraints_inputs,
        when_growthinit=prescribe.when_growthinit,
        **constraints_kwargs,
    )
    phenology = None
    phenology_biomass = prescribe.biomass
    phenology_leaf_frac = prescribe.leaf_frac
    phenology_when_growthinit = prescribe.when_growthinit
    if phenology_inputs is not None:
        phenology = phenology_step(
            **phenology_inputs,
            pft_present=prescribe.pft_present,
            when_growthinit=prescribe.when_growthinit,
            biomass=prescribe.biomass,
            leaf_frac=prescribe.leaf_frac,
            co2_to_bm=prescribe.co2_to_bm,
            **phenology_kwargs,
        )
        phenology_biomass = phenology.biomass
        phenology_leaf_frac = phenology.leaf_frac
        phenology_when_growthinit = phenology.when_growthinit
    lai_after_prescribe = set_lai_from_biomass(prescribe.biomass, alloc_inputs["sla_calc"])
    allocation = allocation_step(
        **alloc_inputs,
        veget_max=prescribe_inputs["veget_max"],
        when_growthinit=phenology_when_growthinit,
        biomass=phenology_biomass,
        leaf_frac=phenology_leaf_frac,
        **allocation_kwargs,
    )
    lai_after_alloc = set_lai_from_biomass(allocation.biomass, alloc_inputs["sla_calc"])
    post_npp = stomate_daily_carbon_kill_gap_turnover_explicit(
        **post_npp_inputs,
        biomass_before=allocation.biomass,
        f_alloc=allocation.f_alloc,
        pft_present=prescribe.pft_present,
        leaf_age=allocation.leaf_age,
        leaf_frac=allocation.leaf_frac,
        ind=prescribe.ind,
        cn_ind=prescribe.cn_ind,
        when_growthinit=phenology_when_growthinit,
        everywhere=prescribe.everywhere,
        lai=lai_after_alloc,
        regenerate=constraints.regenerate,
    )
    return ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult(
        prescribe=prescribe,
        constraints=constraints,
        phenology=phenology,
        allocation=allocation,
        post_npp=post_npp,
        lai_after_prescribe=lai_after_prescribe,
        lai_after_alloc=lai_after_alloc,
        requires_trace=(),
    )


_DAILY_CARBON_BOUNDARY_REQUIRES_TRACE = (
    "bm_alloc",
    "biomass_before_alloc",
    "biomass_after_alloc",
    "biomass_before_npp",
    "biomass_after_npp",
    "resp_growth",
    "npp_daily",
    "agr_pools_before",
    "agr_pools_after",
)


def _bool_tuple(values) -> tuple[bool, ...]:
    return tuple(bool(value) for value in np.asarray(values, dtype=bool))


def _float_tuple(values) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values, dtype=float))


def _daily_carbon_without_boundary_trace(result):
    boundary = result.post_npp.daily_carbon.boundary._replace(requires_trace=())
    daily_carbon = result.post_npp.daily_carbon._replace(boundary=boundary)
    post_npp = result.post_npp._replace(daily_carbon=daily_carbon)
    return result._replace(post_npp=post_npp)


def _daily_carbon_with_boundary_trace(result):
    boundary = result.post_npp.daily_carbon.boundary._replace(
        requires_trace=_DAILY_CARBON_BOUNDARY_REQUIRES_TRACE
    )
    daily_carbon = result.post_npp.daily_carbon._replace(boundary=boundary)
    post_npp = result.post_npp._replace(daily_carbon=daily_carbon)
    return result._replace(post_npp=post_npp)


def _daily_carbon_static_dispatch(
    prescribe_inputs,
    constraints_inputs,
    phenology_inputs,
    alloc_inputs,
    post_npp_inputs,
    allocation_kwargs,
    *,
    pres_ok_dgvm,
    pres_lpj_gap_const_mort,
    pres_natural,
    pres_pasture,
    pres_is_tree,
    pres_pheno_is_none,
    cons_natural,
    cons_is_tree,
    cons_is_peat,
    cons_pheno_is_none,
    cons_tmin_crit,
    cons_tcm_crit,
    phenology_active_pft_mask,
    pheno_model,
    post_ok_dgvm,
    post_lpj_gap_const_mort,
    post_wire_vmax,
    post_ok_nlim_vmax,
):
    prescribe = dict(prescribe_inputs)
    prescribe.update(
        {
            "ok_dgvm": pres_ok_dgvm,
            "lpj_gap_const_mort": pres_lpj_gap_const_mort,
            "natural": pres_natural,
            "pasture": pres_pasture,
            "is_tree": pres_is_tree,
            "pheno_is_none": pres_pheno_is_none,
        }
    )
    constraints = dict(constraints_inputs)
    constraints.update(
        {
            "natural": cons_natural,
            "is_tree": cons_is_tree,
            "is_peat": cons_is_peat,
            "pheno_is_none": cons_pheno_is_none,
            "tmin_crit": cons_tmin_crit,
            "tcm_crit": cons_tcm_crit,
        }
    )
    phenology = dict(phenology_inputs)
    phenology["active_pft_mask"] = phenology_active_pft_mask
    phenology["pheno_model"] = pheno_model
    post_npp = dict(post_npp_inputs)
    post_npp.update(
        {
            "ok_dgvm": post_ok_dgvm,
            "lpj_gap_const_mort": post_lpj_gap_const_mort,
            "wire_vmax": post_wire_vmax,
            "ok_nlim_vmax": post_ok_nlim_vmax,
        }
    )
    return _daily_carbon_without_boundary_trace(
        stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit(
            prescribe_inputs=prescribe,
            constraints_inputs=constraints,
            phenology_inputs=phenology,
            alloc_inputs=alloc_inputs,
            post_npp_inputs=post_npp,
            allocation_kwargs=allocation_kwargs,
        )
    )


_daily_carbon_static_dispatch_jit = jit(
    _daily_carbon_static_dispatch,
    static_argnames=(
        "pres_ok_dgvm",
        "pres_lpj_gap_const_mort",
        "pres_natural",
        "pres_pasture",
        "pres_is_tree",
        "pres_pheno_is_none",
        "cons_natural",
        "cons_is_tree",
        "cons_is_peat",
        "cons_pheno_is_none",
        "cons_tmin_crit",
        "cons_tcm_crit",
        "phenology_active_pft_mask",
        "pheno_model",
        "post_ok_dgvm",
        "post_lpj_gap_const_mort",
        "post_wire_vmax",
        "post_ok_nlim_vmax",
    ),
)


def stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_outer_compiled(
    *,
    prescribe_inputs,
    constraints_inputs,
    phenology_inputs,
    alloc_inputs,
    post_npp_inputs,
    static_dispatch,
    allocation_kwargs=None,
) -> ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult:
    """Run the static dispatch inside an already compiled outer transition."""

    prescribe = dict(prescribe_inputs)
    constraints = dict(constraints_inputs)
    phenology = {} if phenology_inputs is None else dict(phenology_inputs)
    post_npp = dict(post_npp_inputs)
    allocation_kwargs = {} if allocation_kwargs is None else dict(allocation_kwargs)
    for name in (
        "ok_dgvm",
        "lpj_gap_const_mort",
        "natural",
        "pasture",
        "is_tree",
        "pheno_is_none",
    ):
        prescribe.pop(name)
    for name in (
        "natural",
        "is_tree",
        "is_peat",
        "pheno_is_none",
        "tmin_crit",
        "tcm_crit",
    ):
        constraints.pop(name)
    phenology.pop("pheno_model")
    for name in ("ok_dgvm", "lpj_gap_const_mort", "wire_vmax", "ok_nlim_vmax"):
        post_npp.pop(name)
    result = _daily_carbon_static_dispatch(
        prescribe,
        constraints,
        phenology,
        alloc_inputs,
        post_npp,
        allocation_kwargs,
        **dict(static_dispatch),
    )
    return _daily_carbon_with_boundary_trace(result)


def stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_static_jit(
    *,
    prescribe_inputs,
    constraints_inputs,
    phenology_inputs,
    alloc_inputs,
    post_npp_inputs,
    allocation_kwargs=None,
) -> ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult:
    """Compiled paper-case daily-carbon adapter with static PFT dispatch."""

    prescribe = dict(prescribe_inputs)
    constraints = dict(constraints_inputs)
    phenology = {} if phenology_inputs is None else dict(phenology_inputs)
    post_npp = dict(post_npp_inputs)
    allocation_kwargs = {} if allocation_kwargs is None else dict(allocation_kwargs)
    veget_max = np.asarray(prescribe["veget_max"])
    if veget_max.ndim != 2:
        raise ValueError("prescribe_inputs['veget_max'] must have shape (npts, nvm)")
    active_pft_mask = tuple(bool(value) for value in np.any(veget_max > 0.0, axis=0))
    pres_ok_dgvm = bool(prescribe.pop("ok_dgvm"))
    pres_lpj_gap_const_mort = bool(prescribe.pop("lpj_gap_const_mort"))
    pres_natural = _bool_tuple(prescribe.pop("natural"))
    pres_pasture = _bool_tuple(prescribe.pop("pasture"))
    pres_is_tree = _bool_tuple(prescribe.pop("is_tree"))
    pres_pheno_is_none = _bool_tuple(prescribe.pop("pheno_is_none"))
    cons_natural = _bool_tuple(constraints.pop("natural"))
    cons_is_tree = _bool_tuple(constraints.pop("is_tree"))
    cons_is_peat = _bool_tuple(constraints.pop("is_peat"))
    cons_pheno_is_none = _bool_tuple(constraints.pop("pheno_is_none"))
    cons_tmin_crit = _float_tuple(constraints.pop("tmin_crit"))
    cons_tcm_crit = _float_tuple(constraints.pop("tcm_crit"))
    pheno_model = tuple(phenology.pop("pheno_model"))
    post_ok_dgvm = bool(post_npp.pop("ok_dgvm"))
    post_lpj_gap_const_mort = bool(post_npp.pop("lpj_gap_const_mort"))
    post_wire_vmax = bool(post_npp.pop("wire_vmax"))
    post_ok_nlim_vmax = bool(post_npp.pop("ok_nlim_vmax"))
    result = _daily_carbon_static_dispatch_jit(
        prescribe,
        constraints,
        phenology,
        alloc_inputs,
        post_npp,
        allocation_kwargs,
        pres_ok_dgvm=pres_ok_dgvm,
        pres_lpj_gap_const_mort=pres_lpj_gap_const_mort,
        pres_natural=pres_natural,
        pres_pasture=pres_pasture,
        pres_is_tree=pres_is_tree,
        pres_pheno_is_none=pres_pheno_is_none,
        cons_natural=cons_natural,
        cons_is_tree=cons_is_tree,
        cons_is_peat=cons_is_peat,
        cons_pheno_is_none=cons_pheno_is_none,
        cons_tmin_crit=cons_tmin_crit,
        cons_tcm_crit=cons_tcm_crit,
        phenology_active_pft_mask=active_pft_mask,
        pheno_model=pheno_model,
        post_ok_dgvm=post_ok_dgvm,
        post_lpj_gap_const_mort=post_lpj_gap_const_mort,
        post_wire_vmax=post_wire_vmax,
        post_ok_nlim_vmax=post_ok_nlim_vmax,
    )
    return _daily_carbon_with_boundary_trace(result)


def stomate_daily_scheduled_gpp_prescribe_constraints_alloc_kill_gap_turnover_explicit(
    *,
    gpp,
    gpp_daily_current,
    veget_max,
    totfrac_nobio,
    dt_sechiba,
    dt_stomate,
    do_slow,
    prescribe_inputs,
    constraints_inputs,
    phenology_inputs=None,
    alloc_inputs,
    post_npp_inputs,
    constraints_kwargs=None,
    phenology_kwargs=None,
    allocation_kwargs=None,
    one_day=86400.0,
) -> ExplicitDailyCarbonScheduledGppPrescribeConstraintsAllocKillGapTurnoverResult:
    """Derive ``gpp_daily`` from the local STOMATE schedule before the chain.

    Fortran provenance: ``src_stomate/stomate.f90`` lines 3020-3032 construct
    ``gpp_d`` from instantaneous ``gpp`` and ``veget_cov_max``; line 3208
    passes ``gpp_d`` into ``stomate_accu``; ``stomate_accu_r2d`` lines
    9365-9387 accumulates ``field_out += field_in * dt_sechiba`` and divides
    by ``dt_stomate`` when ``do_slow`` is true. The downstream chain provenance
    follows ``stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit``.
    """

    post_npp_inputs = dict(post_npp_inputs)
    if "gpp_daily" in post_npp_inputs:
        raise ValueError("post_npp_inputs must not supply scheduling-owned field: gpp_daily")

    gpp_d = stomate_gpp_daily_increment(
        gpp,
        veget_max,
        totfrac_nobio,
        dt_sechiba=dt_sechiba,
        one_day=one_day,
    )
    gpp_daily = stomate_accumulate_daily(
        gpp_daily_current,
        gpp_d,
        do_slow,
        dt_sechiba=dt_sechiba,
        dt_stomate=dt_stomate,
    )
    chain = stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit(
        prescribe_inputs=prescribe_inputs,
        constraints_inputs=constraints_inputs,
        phenology_inputs=phenology_inputs,
        alloc_inputs=alloc_inputs,
        post_npp_inputs={**post_npp_inputs, "gpp_daily": gpp_daily},
        constraints_kwargs=constraints_kwargs,
        phenology_kwargs=phenology_kwargs,
        allocation_kwargs=allocation_kwargs,
    )
    return ExplicitDailyCarbonScheduledGppPrescribeConstraintsAllocKillGapTurnoverResult(
        gpp_d=gpp_d,
        gpp_daily=gpp_daily,
        chain=chain,
        requires_trace=(),
    )


def stomate_daily_scheduled_gpp_maintenance_prescribe_constraints_alloc_kill_gap_turnover_explicit(
    *,
    gpp,
    gpp_daily_current,
    veget_max,
    totfrac_nobio,
    dt_sechiba,
    dt_stomate,
    do_slow,
    t2m,
    t2m_longterm,
    stempdiag,
    z_soil,
    rprof,
    sla_calc,
    coeff_maint_zero,
    maint_resp_slope,
    ext_coeff,
    is_tree,
    resp_maint_part_current,
    prescribe_inputs,
    constraints_inputs,
    phenology_inputs=None,
    alloc_inputs,
    post_npp_inputs,
    flood_frac=None,
    constraints_kwargs=None,
    phenology_kwargs=None,
    allocation_kwargs=None,
    one_day=86400.0,
) -> ExplicitDailyCarbonScheduledGppMaintenancePrescribeConstraintsAllocKillGapTurnoverResult:
    """Derive maintenance respiration and ``gpp_daily`` before STOMATE chain.

    Fortran provenance: ``src_stomate/stomate.f90`` lines 3020-3032 and 3208
    schedule ``gpp_daily``; lines 3244-3267 call ``maint_respiration``, compute
    ``resp_maint_radia``/``flood_root_radia``, and accumulate
    ``resp_maint_part``. The downstream process order follows
    ``stomate_daily_scheduled_gpp_prescribe_constraints_alloc_kill_gap_turnover_explicit``.
    """

    post_npp_inputs = dict(post_npp_inputs)
    if "resp_maint_part" in post_npp_inputs:
        raise ValueError("post_npp_inputs must not supply maintenance-owned field: resp_maint_part")

    dt_days = jnp.asarray(dt_sechiba) / jnp.asarray(one_day)
    maintenance = maintenance_respiration(
        prescribe_inputs["biomass"],
        t2m,
        t2m_longterm,
        stempdiag,
        z_soil,
        rprof,
        sla_calc,
        coeff_maint_zero,
        maint_resp_slope,
        ext_coeff,
        is_tree,
        dt_sechiba_days=dt_days,
    )
    resp_maint_radia, flood_root_radia = sum_resp_maint_radia(
        maintenance.resp_maint_part,
        flood_frac=flood_frac,
    )
    resp_maint_part = accumulate_resp_maint_part(resp_maint_part_current, maintenance.resp_maint_part)
    scheduled_gpp_chain = stomate_daily_scheduled_gpp_prescribe_constraints_alloc_kill_gap_turnover_explicit(
        gpp=gpp,
        gpp_daily_current=gpp_daily_current,
        veget_max=veget_max,
        totfrac_nobio=totfrac_nobio,
        dt_sechiba=dt_sechiba,
        dt_stomate=dt_stomate,
        do_slow=do_slow,
        prescribe_inputs=prescribe_inputs,
        constraints_inputs=constraints_inputs,
        phenology_inputs=phenology_inputs,
        alloc_inputs=alloc_inputs,
        post_npp_inputs={**post_npp_inputs, "resp_maint_part": resp_maint_part},
        constraints_kwargs=constraints_kwargs,
        phenology_kwargs=phenology_kwargs,
        allocation_kwargs=allocation_kwargs,
        one_day=one_day,
    )
    return ExplicitDailyCarbonScheduledGppMaintenancePrescribeConstraintsAllocKillGapTurnoverResult(
        maintenance=maintenance,
        resp_maint_radia=resp_maint_radia,
        flood_root_radia=flood_root_radia,
        resp_maint_part=resp_maint_part,
        scheduled_gpp_chain=scheduled_gpp_chain,
        requires_trace=(),
    )


def stomate_ok_leak_explicit(
    *,
    litter_above,
    litter_below,
    lignin_struc_above,
    lignin_struc_below,
    litterpart,
    dead_leaves,
    fuel_1hr,
    fuel_10hr,
    fuel_100hr,
    fuel_1000hr,
    bm_to_litter,
    turnover,
    rprof,
    veget_max,
    sla_calc,
    fbact_litter,
    poor_soils,
    flood_frac,
    carbon_32l,
    doc,
    doc_precip2ground,
    doc_precip2canopy,
    dry_dep_canopy,
    interception_storage,
    canopy2ground,
    doc_to_topsoil,
    doc_to_subsoil,
    fbact_doc,
    fbact_soilcarbon,
    soil_mc,
    soil_mc_32l,
    wat_flux,
    soilwater_31mm,
    runoff_per_soil,
    drainage_per_soil,
    runoff2peat,
    fastr,
    pref_soil_veg,
    tprof,
    clay,
    bulk_dens,
    z_soil,
    zf_soil_b,
    flux_red,
    natural,
    is_peat,
    is_c4,
    resp_maint_part_radia,
    flood_root_radia,
    dt_days,
    control_temp_above,
    control_moist_above,
    soil_mc_top_by_pft,
    dif_doc=None,
    cue=0.3,
    nslm: int | None = None,
    ndeep: int | None = None,
    sro_bottom=5,
    conc_doc_max=100.0,
    min_sechiba=0.0,
    ok_cryoturb: bool = False,
    cryoturbation_coefficients: SoilcarbonCryoturbationCoefficients | None = None,
    altmax_ind=None,
    altmax_lastyear=None,
    fixed_cryoturbation_depth=None,
    veget_mask=None,
    zi_soil=None,
    cryoturbation_diff_k_in=0.001,
    bioturbation_diff_k_in=0.0001,
    use_new_cryoturbation: bool = False,
    cryoturbation_method: int = 4,
    max_cryoturb_alt=3.0,
    min_cryoturb_alt=0.01,
    use_fixed_cryoturbation_depth: bool = False,
    bioturbation_depth=2.0,
    perma_peat: bool = False,
    cmax_peat=None,
    perma_peat_veget_mask=None,
    frac1=0.95,
    frac2=0.05,
    capture_transfers: bool = False,
) -> ExplicitOkLeakResult:
    """Compose the audited OK_LEAK litter and soilcarbon path from explicit inputs.

    Fortran provenance: ``stomate_litter.f90`` ``littercalc_leak`` lines
    1950-2799, ``stomate_soilcarbon.f90`` ``soilcarbon_leak`` lines
    1176-2303, and ``stomate.f90`` DOC aggregation lines 3415-3458. This
    adapter does not derive phenology, turnover, allocation, maintenance
    respiration, hydrology, TF-DOC canopy inputs, or cryoturbation state;
    callers must supply those process-boundary values explicitly.
    """

    required = {
        "control_temp_above": control_temp_above,
        "control_moist_above": control_moist_above,
        "soil_mc_top_by_pft": soil_mc_top_by_pft,
        "resp_maint_part_radia": resp_maint_part_radia,
        "flood_root_radia": flood_root_radia,
    }
    missing = tuple(name for name, value in required.items() if value is None)
    if missing:
        raise ValueError(f"explicit OK_LEAK adapter requires {', '.join(missing)}")

    soil_mc = jnp.asarray(soil_mc)
    if nslm is None:
        nslm = soil_mc.shape[1]
    if ndeep is None:
        ndeep = jnp.asarray(carbon_32l).shape[3]

    littercalc_core = littercalc_leak_core_with_controls if ok_cryoturb else littercalc_leak_core_with_controls_jit
    littercalc = littercalc_core(
        litter_above,
        litter_below,
        lignin_struc_above,
        lignin_struc_below,
        litterpart,
        dead_leaves,
        fuel_1hr,
        fuel_10hr,
        fuel_100hr,
        fuel_1000hr,
        bm_to_litter,
        turnover,
        rprof,
        z_soil,
        veget_max,
        sla_calc,
        fbact_litter,
        poor_soils,
        flood_frac,
        dt_days=dt_days,
        control_temp_above=control_temp_above,
        control_moist_above=control_moist_above,
        soil_mc_top_by_pft=soil_mc_top_by_pft,
        nslm=int(nslm),
        ndeep=int(ndeep),
        sro_bottom=sro_bottom,
    )
    tf_doc = soilcarbon_leak_tf_doc_ground_fluxes(
        doc_precip2ground,
        doc_precip2canopy,
        dry_dep_canopy,
        interception_storage,
        canopy2ground,
        veget_max,
        flood_frac,
        conc_doc_max=conc_doc_max,
    )
    soilcarbon_core = soilcarbon_leak_core_step if ok_cryoturb else soilcarbon_leak_core_step_jit
    soilcarbon = soilcarbon_core(
        carbon_32l,
        doc,
        littercalc.soilcarbon_input_doc,
        doc_to_topsoil,
        doc_to_subsoil,
        tf_doc.wet_dep_ground,
        tf_doc.wet_dep_flood,
        littercalc.litter_above,
        littercalc.litter_below,
        littercalc.lignin_struc_above,
        littercalc.lignin_struc_below,
        fbact_doc,
        fbact_soilcarbon,
        soil_mc,
        soil_mc_32l,
        wat_flux,
        soilwater_31mm,
        runoff_per_soil,
        drainage_per_soil,
        runoff2peat,
        fastr,
        pref_soil_veg,
        veget_max,
        flood_frac,
        littercalc.floodcarbon_input,
        tprof,
        clay,
        bulk_dens,
        z_soil,
        zf_soil_b,
        flux_red,
        natural,
        is_peat,
        is_c4,
        dt_days=dt_days,
        dif_doc=dif_doc,
        cue=cue,
        nslm=int(nslm),
        sro_bottom=sro_bottom,
        ok_cryoturb=ok_cryoturb,
        cryoturbation_coefficients=cryoturbation_coefficients,
        altmax_ind=altmax_ind,
        altmax_lastyear=altmax_lastyear,
        fixed_cryoturbation_depth=fixed_cryoturbation_depth,
        veget_mask=veget_mask,
        zi_soil=zi_soil,
        cryoturbation_diff_k_in=cryoturbation_diff_k_in,
        bioturbation_diff_k_in=bioturbation_diff_k_in,
        use_new_cryoturbation=use_new_cryoturbation,
        cryoturbation_method=cryoturbation_method,
        max_cryoturb_alt=max_cryoturb_alt,
        min_cryoturb_alt=min_cryoturb_alt,
        use_fixed_cryoturbation_depth=use_fixed_cryoturbation_depth,
        bioturbation_depth=bioturbation_depth,
        perma_peat=perma_peat,
        cmax_peat=cmax_peat,
        perma_peat_veget_mask=perma_peat_veget_mask,
        frac1=frac1,
        frac2=frac2,
        capture_transfers=capture_transfers,
    )
    doc_export = soilcarbon_leak_doc_export_aggregate(
        soilcarbon.doc_exp,
        veget_max,
        littercalc.resp_hetero_litter,
        soilcarbon.resp_hetero_soil,
        littercalc.resp_hetero_flood,
        soilcarbon.resp_flood_soil,
        resp_maint_part_radia,
        flood_root_radia,
        runoff_per_soil,
        drainage_per_soil,
        pref_soil_veg,
        flood_frac,
        dt_days=dt_days,
        min_sechiba=min_sechiba,
    )
    transfers = None
    if capture_transfers:
        if soilcarbon.transfers is None:
            raise RuntimeError("soilcarbon transfer capture was requested but not returned")
        core_transfers = soilcarbon.transfers
        transfers = OkLeakTransferStepV1(
            litter_respiration=littercalc.resp_hetero_litter * dt_days,
            litter_flood_respiration=littercalc.resp_hetero_flood * dt_days,
            litter_to_doc=littercalc.soilcarbon_input_doc * dt_days,
            floodcarbon_input=littercalc.floodcarbon_input * dt_days,
            poc_gross_decomposition=soilcarbon.fluxtot,
            poc_flood_gross_decomposition=soilcarbon.fluxtot_flood,
            doc_gross_decomposition=soilcarbon.fluxtot_doc,
            doc_flood_gross_decomposition=soilcarbon.fluxtot_doc_flood,
            doc_to_topsoil=jnp.asarray(doc_to_topsoil) * dt_days,
            doc_to_subsoil=jnp.asarray(doc_to_subsoil) * dt_days,
            doc_precip2ground=jnp.asarray(doc_precip2ground),
            doc_precip2canopy=jnp.asarray(doc_precip2canopy),
            dry_dep_canopy=jnp.asarray(dry_dep_canopy),
            wet_dep_ground=tf_doc.wet_dep_ground,
            wet_dep_flood=tf_doc.wet_dep_flood,
            doc_run=soilcarbon.doc_run,
            doc_drain=soilcarbon.doc_drain,
            doc_flood=soilcarbon.doc_flood,
            doc_run_2_peat=core_transfers.doc_run_2_peat,
            doc_free_to_adsorbed=core_transfers.doc_free_to_adsorbed,
            doc_advective_interface=core_transfers.doc_advective_interface,
            doc_diffusive_interface=core_transfers.doc_diffusive_interface,
            cryoturbation_carbon_transfer=core_transfers.cryoturbation_carbon_transfer,
            cryoturbation_doc_transfer=core_transfers.cryoturbation_doc_transfer,
            cryoturbation_litter_transfer=core_transfers.cryoturbation_litter_transfer,
            perma_peat_carbon_transfer=core_transfers.perma_peat_carbon_transfer,
        )
    return ExplicitOkLeakResult(
        littercalc=littercalc,
        soilcarbon=soilcarbon,
        doc_export=doc_export,
        wet_dep_ground=tf_doc.wet_dep_ground,
        wet_dep_flood=tf_doc.wet_dep_flood,
        interception_storage=tf_doc.interception_storage,
        transfers=transfers,
        requires_trace=(),
    )


def stomate_ok_pc_deep_carbcycle_explicit(
    *,
    deepC_a,
    deepC_s,
    deepC_p,
    soilc_in,
    fbact_out,
    O2_soil,
    CH4_soil,
    O2_snow,
    CH4_snow,
    hslong_in,
    snowdz,
    snowrho,
    tprof,
    tsurf,
    pb,
    clay,
    veget_max,
    veget_mask,
    zi_soil,
    zf_soil,
    rprof,
    alt,
    alt_ind,
    altmax,
    altmax_ind,
    altmax_lastyear,
    altmax_ind_lastyear,
    fixed_cryoturbation_depth,
    time_step_seconds,
    dayno,
    gasdiff_coefficients: DeepCarbonGasDiffusionCoefficients | None = None,
    cryoturbation_coefficients: DeepCarbonCryoturbationCoefficients | None = None,
    zi_snow=None,
    zf_snow=None,
    airvol_soil=None,
    totporO2_soil=None,
    totporCH4_soil=None,
    diffO2_soil=None,
    diffCH4_soil=None,
    airvol_snow=None,
    totporO2_snow=None,
    totporCH4_snow=None,
    diffO2_snow=None,
    diffCH4_snow=None,
    firstcall: bool = False,
    firstcall_altcalc: bool = False,
    soilc_isspinup: bool = False,
    newaltcalc: bool = False,
    no_pfrost_decomp: bool = False,
    max_shum_value=1.0,
    z_root_max=2.0,
    ok_cryoturb: bool = False,
    cryoturbation_diff_k_in=0.001,
    bioturbation_diff_k_in=0.0001,
    use_new_cryoturbation: bool = False,
    cryoturbation_method: int = 4,
    max_cryoturb_alt=3.0,
    min_cryoturb_alt=0.01,
    use_fixed_cryoturbation_depth: bool = False,
    bioturbation_depth=2.0,
    ok_methane: bool = False,
    oxlim: bool = False,
    tau_CH4troph=432000.0,
    fbactratio=9.0,
    O2m=3.0,
    MG_useallCpools: bool = True,
    enable_plant_transport: bool = True,
    enable_ebullition: bool = True,
    Tref=None,
    Tgr=0.0,
    refdep=0.75,
    heat_co2_act=40.0e6,
    heat_co2_slo=30.0e6,
    heat_co2_pas=10.0e6,
    heat_ch4_gen=0.0,
    heat_ch4_troph=0.0,
    perma_peat: bool = False,
    cmax_peat=None,
    is_peat=None,
    frac1=0.95,
    frac2=0.05,
    reset_yedoma: bool = False,
    zz_deep=None,
    yedoma_depth=None,
    yedoma_cinit_act=None,
    yedoma_cinit_slo=None,
    yedoma_cinit_pas=None,
    yedoma_map_filename: str = "NONE",
    yedoma_values=None,
    one_day=86400.0,
    one_year=365.0,
) -> ExplicitOkPcDeepCarbcycleResult:
    """Compose the explicit OK_PC ``deep_carbcycle`` daily path.

    Fortran provenance: ``src_stomate/stomate_permafrost_soilcarbon.f90``
    ``deep_carbcycle`` lines 892-1077. This adapter wires the local source
    kernels in the executable order: optional saved gas diffusion, snow
    interpolation, active-layer/root-depth update, optional cryoturbation,
    carbon input/decomposition/methane aggregation, new gas properties,
    saved gas-diffusion coefficients, and vertical carbon integration.
    """

    deep_a = jnp.asarray(deepC_a)
    deep_s = jnp.asarray(deepC_s)
    deep_p = jnp.asarray(deepC_p)
    mask = jnp.asarray(veget_mask, dtype=bool)
    veget = jnp.asarray(veget_max)
    hslong = jnp.maximum(jnp.minimum(jnp.asarray(hslong_in), max_shum_value), 0.0)
    snowdz_arr = jnp.asarray(snowdz)
    heights_snow = jnp.repeat(jnp.sum(snowdz_arr, axis=1)[:, None], mask.shape[1], axis=1)

    if bool(reset_yedoma):
        required_yedoma = {
            "zz_deep": zz_deep,
            "yedoma_depth": yedoma_depth,
            "yedoma_cinit_act": yedoma_cinit_act,
            "yedoma_cinit_slo": yedoma_cinit_slo,
            "yedoma_cinit_pas": yedoma_cinit_pas,
        }
        missing = tuple(name for name, value in required_yedoma.items() if value is None)
        if missing:
            raise ValueError(f"reset_yedoma requires explicit {', '.join(missing)}")
        yedoma = deep_carbon_yedoma_reset_step(
            deep_a,
            deep_s,
            deep_p,
            zz_deep,
            altmax_ind,
            mask,
            yedoma_depth=yedoma_depth,
            yedoma_cinit_act=yedoma_cinit_act,
            yedoma_cinit_slo=yedoma_cinit_slo,
            yedoma_cinit_pas=yedoma_cinit_pas,
            yedoma_map_filename=yedoma_map_filename,
            yedoma_values=yedoma_values,
        )
        deep_a, deep_s, deep_p = yedoma.deepC_a, yedoma.deepC_s, yedoma.deepC_p
    else:
        yedoma = None

    if zi_snow is None or zf_snow is None:
        snow_geometry = deep_carbon_snow_interpol_step(
            O2_snow,
            CH4_snow,
            jnp.zeros_like(jnp.asarray(O2_snow)),
            jnp.zeros((jnp.asarray(O2_snow).shape[0], jnp.asarray(O2_snow).shape[1] + 1, jnp.asarray(O2_snow).shape[2])),
            veget,
            snowdz_arr,
            jnp.zeros_like(mask),
        )
        zi_snow_work = snow_geometry.zi_snow
        zf_snow_work = snow_geometry.zf_snow
    else:
        zi_snow_work = jnp.asarray(zi_snow)
        zf_snow_work = jnp.asarray(zf_snow)

    o2_snow = jnp.asarray(O2_snow)
    ch4_snow = jnp.asarray(CH4_snow)
    o2_soil = jnp.asarray(O2_soil)
    ch4_soil = jnp.asarray(CH4_soil)

    if gasdiff_coefficients is not None:
        gas_state = deep_carbon_soil_gasdiff_diffuse(
            o2_snow,
            ch4_snow,
            o2_soil,
            ch4_soil,
            gasdiff_coefficients,
            pb,
            tsurf,
            mask,
        )
        o2_snow, ch4_snow = gas_state.O2_snow, gas_state.CH4_snow
        o2_soil, ch4_soil = gas_state.O2_soil, gas_state.CH4_soil

    snow_interp: DeepCarbonSnowInterpolResult = deep_carbon_snow_interpol_step(
        o2_snow,
        ch4_snow,
        zi_snow_work,
        zf_snow_work,
        veget,
        snowdz_arr,
        mask,
    )
    o2_snow, ch4_snow = snow_interp.snowO2, snow_interp.snowCH4
    zi_snow_work, zf_snow_work = snow_interp.zi_snow, snow_interp.zf_snow

    altcalc = deep_carbon_altcalc_step(
        tprof,
        zi_soil,
        altmax,
        altmax_ind,
        altmax_lastyear,
        altmax_ind_lastyear,
        mask,
        firstcall=firstcall_altcalc,
        newaltcalc=newaltcalc,
        dayno=int(dayno),
        soilc_isspinup=soilc_isspinup,
    )
    root = deep_carbon_root_depth_step(
        altcalc.altmax_lastyear,
        altcalc.altmax_ind_lastyear,
        mask,
        z_root_max=z_root_max,
    )

    if airvol_soil is None or totporO2_soil is None or totporCH4_soil is None or diffO2_soil is None or diffCH4_soil is None:
        current_props = deep_carbon_gasdiff_properties_step(hslong, snowrho, mask)
        airvol_soil = current_props.airvol_soil
        totporO2_soil = current_props.totporO2_soil
        totporCH4_soil = current_props.totporCH4_soil
        diffO2_soil = current_props.diffO2_soil
        diffCH4_soil = current_props.diffCH4_soil
        if airvol_snow is None:
            airvol_snow = current_props.airvol_snow
        if totporO2_snow is None:
            totporO2_snow = current_props.totporO2_snow
        if totporCH4_snow is None:
            totporCH4_snow = current_props.totporCH4_snow
        if diffO2_snow is None:
            diffO2_snow = current_props.diffO2_snow
        if diffCH4_snow is None:
            diffCH4_snow = current_props.diffCH4_snow

    core = deep_carbon_nonmethane_core_step(
        deep_a,
        deep_s,
        deep_p,
        soilc_in,
        fbact_out,
        o2_soil,
        totporO2_soil,
        clay,
        mask,
        zi_soil,
        zf_soil,
        root.z_root,
        altcalc.altmax_lastyear,
        rprof,
        time_step_seconds=time_step_seconds,
        airvol_soil=airvol_soil,
        oxlim=oxlim,
        no_pfrost_decomp=no_pfrost_decomp,
        perma_peat=perma_peat,
        cmax_peat=cmax_peat,
        is_peat=is_peat,
        frac1=frac1,
        frac2=frac2,
        one_day=one_day,
        ok_methane=ok_methane,
        CH4_soil=ch4_soil,
        totporCH4_soil=totporCH4_soil,
        hslong=hslong,
        tprof=tprof,
        tau_CH4troph=tau_CH4troph,
        fbactratio=fbactratio,
        O2m=O2m,
        MG_useallCpools=MG_useallCpools,
        firstcall=firstcall,
        enable_plant_transport=enable_plant_transport,
        enable_ebullition=enable_ebullition,
        Tref=Tref,
        rootlev=root.rootlev,
        Tgr=Tgr,
        refdep=refdep,
        ok_cryoturb=ok_cryoturb,
        cryoturbation_coefficients=cryoturbation_coefficients,
        altmax_ind=altcalc.altmax_ind_lastyear,
        fixed_cryoturbation_depth=fixed_cryoturbation_depth,
        cryoturbation_diff_k_const=cryoturbation_diff_k_in / (one_day * one_year),
        cryoturbation_bio_diff_k_const=bioturbation_diff_k_in / (one_day * one_year),
        use_new_cryoturbation=use_new_cryoturbation,
        cryoturbation_method=cryoturbation_method,
        max_cryoturb_alt=max_cryoturb_alt,
        min_cryoturb_alt=min_cryoturb_alt,
        use_fixed_cryoturbation_depth=use_fixed_cryoturbation_depth,
        bioturbation_depth=bioturbation_depth,
        heat_co2_act=heat_co2_act,
        heat_co2_slo=heat_co2_slo,
        heat_co2_pas=heat_co2_pas,
        heat_ch4_gen=heat_ch4_gen,
        heat_ch4_troph=heat_ch4_troph,
    )

    ch4_soil_after = core.CH4_soil if core.CH4_soil is not None else ch4_soil
    next_props = deep_carbon_gasdiff_properties_step(hslong, snowrho, mask)
    next_gasdiff_coefficients = deep_carbon_soil_gasdiff_coefficients(
        o2_snow,
        ch4_snow,
        next_props.diffO2_snow,
        next_props.diffCH4_snow,
        next_props.totporO2_snow,
        next_props.totporCH4_snow,
        core.O2_soil,
        ch4_soil_after,
        next_props.diffO2_soil,
        next_props.diffCH4_soil,
        next_props.totporO2_soil,
        next_props.totporCH4_soil,
        zi_snow_work,
        zf_snow_work,
        zi_soil,
        zf_soil,
        mask,
        heights_snow,
        dt_seconds=time_step_seconds,
    )

    return ExplicitOkPcDeepCarbcycleResult(
        core=core,
        altcalc=altcalc,
        zi_snow=zi_snow_work,
        zf_snow=zf_snow_work,
        O2_snow=o2_snow,
        CH4_snow=ch4_snow,
        heat_Zimov=core.heat_Zimov,
        O2_soil=core.O2_soil,
        CH4_soil=ch4_soil_after,
        airvol_soil=next_props.airvol_soil,
        totporO2_soil=next_props.totporO2_soil,
        totporCH4_soil=next_props.totporCH4_soil,
        diffO2_soil=next_props.diffO2_soil,
        diffCH4_soil=next_props.diffCH4_soil,
        airvol_snow=next_props.airvol_snow,
        totporO2_snow=next_props.totporO2_snow,
        totporCH4_snow=next_props.totporCH4_snow,
        diffO2_snow=next_props.diffO2_snow,
        diffCH4_snow=next_props.diffCH4_snow,
        z_root=root.z_root,
        rootlev=root.rootlev,
        hslong=hslong,
        heights_snow=heights_snow,
        gasdiff_coefficients=next_gasdiff_coefficients,
        cryoturbation_coefficients=core.cryoturbation_coefficients,
        yedoma=yedoma,
        requires_trace=(),
    )


def stomate_ok_pc_deep_carbcycle_from_restart_state(
    *,
    state,
    sidecar: ExplicitOkPcDeepCarbonSidecarState,
    soilc_in,
    fbact_out,
    hslong_in,
    snowdz,
    snowrho,
    tprof,
    tsurf,
    pb,
    clay,
    veget_max,
    veget_mask,
    zi_soil,
    zf_soil,
    rprof,
    time_step_seconds,
    dayno,
    firstcall: bool = False,
    firstcall_altcalc: bool = False,
    soilc_isspinup: bool = False,
    newaltcalc: bool = False,
    no_pfrost_decomp: bool = False,
    max_shum_value=1.0,
    z_root_max=2.0,
    ok_cryoturb: bool = False,
    cryoturbation_diff_k_in=0.001,
    bioturbation_diff_k_in=0.0001,
    use_new_cryoturbation: bool = False,
    cryoturbation_method: int = 4,
    max_cryoturb_alt=3.0,
    min_cryoturb_alt=0.01,
    use_fixed_cryoturbation_depth: bool = False,
    bioturbation_depth=2.0,
    ok_methane: bool = False,
    oxlim: bool = False,
    tau_CH4troph=432000.0,
    fbactratio=9.0,
    O2m=3.0,
    MG_useallCpools: bool = True,
    enable_plant_transport: bool = True,
    enable_ebullition: bool = True,
    Tgr=0.0,
    refdep=0.75,
    heat_co2_act=40.0e6,
    heat_co2_slo=30.0e6,
    heat_co2_pas=10.0e6,
    heat_ch4_gen=0.0,
    heat_ch4_troph=0.0,
    perma_peat: bool = False,
    cmax_peat=None,
    is_peat=None,
    frac1=0.95,
    frac2=0.05,
    reset_yedoma: bool = False,
    zz_deep=None,
    yedoma_depth=None,
    yedoma_cinit_act=None,
    yedoma_cinit_slo=None,
    yedoma_cinit_pas=None,
    yedoma_map_filename: str = "NONE",
    yedoma_values=None,
    one_day=86400.0,
    one_year=365.0,
) -> ExplicitOkPcFromRestartStateResult:
    """Run OK_PC ``deep_carbcycle`` and write its state back explicitly.

    Fortran provenance: ``src_stomate/stomate.f90::stomate_main`` lines
    3336-3383 and 3628-3655 pass restart/state arrays into
    ``deep_carbcycle`` and then update ``soilc_total`` from ``deepC_*``.
    ``src_stomate/stomate_permafrost_soilcarbon.f90`` lines 47-155 define
    the gas, snow-grid, gas-diffusion, cryoturbation, and active-layer SAVE
    variables that must be carried outside the main restart object here.
    """

    ok_pc = stomate_ok_pc_deep_carbcycle_explicit(
        deepC_a=state.deepC_a,
        deepC_s=state.deepC_s,
        deepC_p=state.deepC_p,
        soilc_in=soilc_in,
        fbact_out=fbact_out,
        O2_soil=sidecar.O2_soil,
        CH4_soil=sidecar.CH4_soil,
        O2_snow=sidecar.O2_snow,
        CH4_snow=sidecar.CH4_snow,
        hslong_in=hslong_in,
        snowdz=snowdz,
        snowrho=snowrho,
        tprof=tprof,
        tsurf=tsurf,
        pb=pb,
        clay=clay,
        veget_max=veget_max,
        veget_mask=veget_mask,
        zi_soil=zi_soil,
        zf_soil=zf_soil,
        rprof=rprof,
        alt=sidecar.alt,
        alt_ind=sidecar.alt_ind,
        altmax=state.altmax,
        altmax_ind=sidecar.altmax_ind,
        altmax_lastyear=sidecar.altmax_lastyear,
        altmax_ind_lastyear=sidecar.altmax_ind_lastyear,
        fixed_cryoturbation_depth=state.fixed_cryoturbation_depth,
        time_step_seconds=time_step_seconds,
        dayno=dayno,
        gasdiff_coefficients=sidecar.gasdiff_coefficients,
        cryoturbation_coefficients=sidecar.cryoturbation_coefficients,
        zi_snow=sidecar.zi_snow,
        zf_snow=sidecar.zf_snow,
        airvol_soil=sidecar.airvol_soil,
        totporO2_soil=sidecar.totporO2_soil,
        totporCH4_soil=sidecar.totporCH4_soil,
        diffO2_soil=sidecar.diffO2_soil,
        diffCH4_soil=sidecar.diffCH4_soil,
        airvol_snow=sidecar.airvol_snow,
        totporO2_snow=sidecar.totporO2_snow,
        totporCH4_snow=sidecar.totporCH4_snow,
        diffO2_snow=sidecar.diffO2_snow,
        diffCH4_snow=sidecar.diffCH4_snow,
        firstcall=firstcall,
        firstcall_altcalc=firstcall_altcalc,
        soilc_isspinup=soilc_isspinup,
        newaltcalc=newaltcalc,
        no_pfrost_decomp=no_pfrost_decomp,
        max_shum_value=max_shum_value,
        z_root_max=z_root_max,
        ok_cryoturb=ok_cryoturb,
        cryoturbation_diff_k_in=cryoturbation_diff_k_in,
        bioturbation_diff_k_in=bioturbation_diff_k_in,
        use_new_cryoturbation=use_new_cryoturbation,
        cryoturbation_method=cryoturbation_method,
        max_cryoturb_alt=max_cryoturb_alt,
        min_cryoturb_alt=min_cryoturb_alt,
        use_fixed_cryoturbation_depth=use_fixed_cryoturbation_depth,
        bioturbation_depth=bioturbation_depth,
        ok_methane=ok_methane,
        oxlim=oxlim,
        tau_CH4troph=tau_CH4troph,
        fbactratio=fbactratio,
        O2m=O2m,
        MG_useallCpools=MG_useallCpools,
        enable_plant_transport=enable_plant_transport,
        enable_ebullition=enable_ebullition,
        Tref=sidecar.Tref,
        Tgr=Tgr,
        refdep=refdep,
        heat_co2_act=heat_co2_act,
        heat_co2_slo=heat_co2_slo,
        heat_co2_pas=heat_co2_pas,
        heat_ch4_gen=heat_ch4_gen,
        heat_ch4_troph=heat_ch4_troph,
        perma_peat=perma_peat,
        cmax_peat=cmax_peat,
        is_peat=is_peat,
        frac1=frac1,
        frac2=frac2,
        reset_yedoma=reset_yedoma,
        zz_deep=zz_deep,
        yedoma_depth=yedoma_depth,
        yedoma_cinit_act=yedoma_cinit_act,
        yedoma_cinit_slo=yedoma_cinit_slo,
        yedoma_cinit_pas=yedoma_cinit_pas,
        yedoma_map_filename=yedoma_map_filename,
        yedoma_values=yedoma_values,
        one_day=one_day,
        one_year=one_year,
    )

    soilc_total = ok_pc.core.deepC_a + ok_pc.core.deepC_s + ok_pc.core.deepC_p
    state_after = state._replace(
        deepC_a=ok_pc.core.deepC_a,
        deepC_s=ok_pc.core.deepC_s,
        deepC_p=ok_pc.core.deepC_p,
        carbon=ok_pc.core.carbon,
        soilc_total=soilc_total,
        altmax=ok_pc.altcalc.altmax,
        fixed_cryoturbation_depth=state.fixed_cryoturbation_depth,
    )
    sidecar_after = ExplicitOkPcDeepCarbonSidecarState(
        O2_soil=ok_pc.O2_soil,
        CH4_soil=ok_pc.CH4_soil,
        O2_snow=ok_pc.O2_snow,
        CH4_snow=ok_pc.CH4_snow,
        zi_snow=ok_pc.zi_snow,
        zf_snow=ok_pc.zf_snow,
        airvol_soil=ok_pc.airvol_soil,
        totporO2_soil=ok_pc.totporO2_soil,
        totporCH4_soil=ok_pc.totporCH4_soil,
        diffO2_soil=ok_pc.diffO2_soil,
        diffCH4_soil=ok_pc.diffCH4_soil,
        airvol_snow=ok_pc.airvol_snow,
        totporO2_snow=ok_pc.totporO2_snow,
        totporCH4_snow=ok_pc.totporCH4_snow,
        diffO2_snow=ok_pc.diffO2_snow,
        diffCH4_snow=ok_pc.diffCH4_snow,
        alt=ok_pc.altcalc.alt,
        alt_ind=ok_pc.altcalc.alt_ind,
        altmax_ind=ok_pc.altcalc.altmax_ind,
        altmax_lastyear=ok_pc.altcalc.altmax_lastyear,
        altmax_ind_lastyear=ok_pc.altcalc.altmax_ind_lastyear,
        z_root=ok_pc.z_root,
        rootlev=ok_pc.rootlev,
        heights_snow=ok_pc.heights_snow,
        gasdiff_coefficients=ok_pc.gasdiff_coefficients,
        cryoturbation_coefficients=ok_pc.cryoturbation_coefficients,
        Tref=ok_pc.core.Tref,
    )
    veget = jnp.asarray(veget_max)
    resp = jnp.asarray(ok_pc.core.resp_hetero_soil)
    sflux_co2_deep = jnp.sum(veget * resp, axis=1) / jnp.asarray(one_day, dtype=resp.dtype)
    if ok_pc.core.sfluxCH4 is None:
        sflux_ch4_deep = jnp.zeros(veget.shape[0], dtype=resp.dtype)
    else:
        sflux_ch4 = jnp.asarray(ok_pc.core.sfluxCH4)
        sflux_ch4_deep = jnp.sum(veget * sflux_ch4, axis=1) / jnp.asarray(one_day, dtype=sflux_ch4.dtype)

    return ExplicitOkPcFromRestartStateResult(
        ok_pc=ok_pc,
        state_after=state_after,
        sidecar_after=sidecar_after,
        carbon_surf=ok_pc.core.carbon_surf,
        resp_hetero_soil=ok_pc.core.resp_hetero_soil,
        heat_Zimov=ok_pc.core.heat_Zimov,
        sfluxCH4_deep=sflux_ch4_deep,
        sfluxCO2_deep=sflux_co2_deep,
        requires_trace=(),
    )


def stomate_ok_pc_firstcall_sidecar_from_restart_gas(
    *,
    gas_state,
    hslong_in,
    snowdz,
    snowrho,
    tprof,
    tsurf,
    pb,
    veget_max,
    veget_mask,
    zi_soil,
    zf_soil,
    altmax,
    time_step_seconds,
    dayno,
    soilc_isspinup: bool = False,
    newaltcalc: bool = False,
    max_shum_value=1.0,
    z_root_max=2.0,
) -> ExplicitOkPcDeepCarbonSidecarState:
    """Initialize OK_PC runtime SAVE sidecar from restart gas and boundaries.

    Fortran provenance: ``src_stomate/stomate_permafrost_soilcarbon.f90``
    ``deep_carbcycle`` firstcall block lines 703-797 allocates/zeros SAVE
    state, builds snow geometry, computes gas porosity/diffusivity, computes
    initial gas-diffusion coefficients, runs ``altcalc``, and leaves
    cryoturbation coefficients to the ordinary end-of-call coefficients step.
    """

    mask = jnp.asarray(veget_mask, dtype=bool)
    veget = jnp.asarray(veget_max)
    hslong = jnp.maximum(jnp.minimum(jnp.asarray(hslong_in), max_shum_value), 0.0)
    snowdz_arr = jnp.asarray(snowdz)
    altmax_arr = jnp.asarray(altmax)
    zero_ind = jnp.zeros_like(altmax_arr, dtype=jnp.int32)
    zero_alt = jnp.zeros_like(altmax_arr)
    snow_geometry = deep_carbon_snowlevels_step(snowdz_arr, veget)
    heights_snow = jnp.repeat(jnp.sum(snowdz_arr, axis=1)[:, None], mask.shape[1], axis=1)
    props = deep_carbon_gasdiff_properties_step(hslong, snowrho, mask)
    gasdiff = deep_carbon_soil_gasdiff_coefficients(
        gas_state.O2_snow,
        gas_state.CH4_snow,
        props.diffO2_snow,
        props.diffCH4_snow,
        props.totporO2_snow,
        props.totporCH4_snow,
        gas_state.O2_soil,
        gas_state.CH4_soil,
        props.diffO2_soil,
        props.diffCH4_soil,
        props.totporO2_soil,
        props.totporCH4_soil,
        snow_geometry.zi_snow,
        snow_geometry.zf_snow,
        zi_soil,
        zf_soil,
        mask,
        heights_snow,
        dt_seconds=time_step_seconds,
    )
    altcalc = deep_carbon_altcalc_step(
        tprof,
        zi_soil,
        altmax_arr,
        zero_ind,
        zero_alt,
        zero_ind,
        mask,
        firstcall=True,
        newaltcalc=newaltcalc,
        dayno=int(dayno),
        soilc_isspinup=soilc_isspinup,
    )
    root = deep_carbon_root_depth_step(
        altcalc.altmax_lastyear,
        altcalc.altmax_ind_lastyear,
        mask,
        z_root_max=z_root_max,
    )
    return ExplicitOkPcDeepCarbonSidecarState(
        O2_soil=jnp.asarray(gas_state.O2_soil),
        CH4_soil=jnp.asarray(gas_state.CH4_soil),
        O2_snow=jnp.asarray(gas_state.O2_snow),
        CH4_snow=jnp.asarray(gas_state.CH4_snow),
        zi_snow=snow_geometry.zi_snow,
        zf_snow=snow_geometry.zf_snow,
        airvol_soil=props.airvol_soil,
        totporO2_soil=props.totporO2_soil,
        totporCH4_soil=props.totporCH4_soil,
        diffO2_soil=props.diffO2_soil,
        diffCH4_soil=props.diffCH4_soil,
        airvol_snow=props.airvol_snow,
        totporO2_snow=props.totporO2_snow,
        totporCH4_snow=props.totporCH4_snow,
        diffO2_snow=props.diffO2_snow,
        diffCH4_snow=props.diffCH4_snow,
        alt=altcalc.alt,
        alt_ind=altcalc.alt_ind,
        altmax_ind=altcalc.altmax_ind,
        altmax_lastyear=altcalc.altmax_lastyear,
        altmax_ind_lastyear=altcalc.altmax_ind_lastyear,
        z_root=root.z_root,
        rootlev=root.rootlev,
        heights_snow=heights_snow,
        gasdiff_coefficients=gasdiff,
        cryoturbation_coefficients=None,
        Tref=jnp.zeros_like(altmax_arr),
    )


def stomate_ok_pc_firstcall_from_restart_gas(
    *,
    state,
    gas_state,
    soilc_in,
    fbact_out,
    hslong_in,
    snowdz,
    snowrho,
    tprof,
    tsurf,
    pb,
    clay,
    veget_max,
    veget_mask,
    zi_soil,
    zf_soil,
    rprof,
    time_step_seconds,
    dayno,
    soilc_isspinup: bool = False,
    newaltcalc: bool = False,
    no_pfrost_decomp: bool = False,
    max_shum_value=1.0,
    z_root_max=2.0,
    ok_cryoturb: bool = False,
    cryoturbation_diff_k_in=0.001,
    bioturbation_diff_k_in=0.0001,
    use_new_cryoturbation: bool = False,
    cryoturbation_method: int = 4,
    max_cryoturb_alt=3.0,
    min_cryoturb_alt=0.01,
    use_fixed_cryoturbation_depth: bool = False,
    bioturbation_depth=2.0,
    ok_methane: bool = False,
    oxlim: bool = False,
    tau_CH4troph=432000.0,
    fbactratio=9.0,
    O2m=3.0,
    MG_useallCpools: bool = True,
    enable_plant_transport: bool = True,
    enable_ebullition: bool = True,
    Tgr=0.0,
    refdep=0.75,
    heat_co2_act=40.0e6,
    heat_co2_slo=30.0e6,
    heat_co2_pas=10.0e6,
    heat_ch4_gen=0.0,
    heat_ch4_troph=0.0,
    perma_peat: bool = False,
    cmax_peat=None,
    is_peat=None,
    frac1=0.95,
    frac2=0.05,
    reset_yedoma: bool = False,
    zz_deep=None,
    yedoma_depth=None,
    yedoma_cinit_act=None,
    yedoma_cinit_slo=None,
    yedoma_cinit_pas=None,
    yedoma_map_filename: str = "NONE",
    yedoma_values=None,
    one_day=86400.0,
    one_year=365.0,
) -> ExplicitOkPcFromRestartStateResult:
    """Run the first OK_PC daily call from restart-backed gas fields.

    Fortran provenance: ``stomate_io.f90::readstart`` lines 1177-1199 read
    soil/snow gas state; ``stomate_permafrost_soilcarbon.f90::deep_carbcycle``
    firstcall lines 703-797 initializes runtime SAVE state; the same
    ``deep_carbcycle`` call then continues through lines 914-1077. This
    wrapper composes those two boundaries without inventing missing state.
    """

    sidecar = stomate_ok_pc_firstcall_sidecar_from_restart_gas(
        gas_state=gas_state,
        hslong_in=hslong_in,
        snowdz=snowdz,
        snowrho=snowrho,
        tprof=tprof,
        tsurf=tsurf,
        pb=pb,
        veget_max=veget_max,
        veget_mask=veget_mask,
        zi_soil=zi_soil,
        zf_soil=zf_soil,
        altmax=state.altmax,
        time_step_seconds=time_step_seconds,
        dayno=dayno,
        soilc_isspinup=soilc_isspinup,
        newaltcalc=newaltcalc,
        max_shum_value=max_shum_value,
        z_root_max=z_root_max,
    )
    return stomate_ok_pc_deep_carbcycle_from_restart_state(
        state=state,
        sidecar=sidecar,
        soilc_in=soilc_in,
        fbact_out=fbact_out,
        hslong_in=hslong_in,
        snowdz=snowdz,
        snowrho=snowrho,
        tprof=tprof,
        tsurf=tsurf,
        pb=pb,
        clay=clay,
        veget_max=veget_max,
        veget_mask=veget_mask,
        zi_soil=zi_soil,
        zf_soil=zf_soil,
        rprof=rprof,
        time_step_seconds=time_step_seconds,
        dayno=dayno,
        firstcall=True,
        firstcall_altcalc=False,
        soilc_isspinup=soilc_isspinup,
        newaltcalc=newaltcalc,
        no_pfrost_decomp=no_pfrost_decomp,
        max_shum_value=max_shum_value,
        z_root_max=z_root_max,
        ok_cryoturb=ok_cryoturb,
        cryoturbation_diff_k_in=cryoturbation_diff_k_in,
        bioturbation_diff_k_in=bioturbation_diff_k_in,
        use_new_cryoturbation=use_new_cryoturbation,
        cryoturbation_method=cryoturbation_method,
        max_cryoturb_alt=max_cryoturb_alt,
        min_cryoturb_alt=min_cryoturb_alt,
        use_fixed_cryoturbation_depth=use_fixed_cryoturbation_depth,
        bioturbation_depth=bioturbation_depth,
        ok_methane=ok_methane,
        oxlim=oxlim,
        tau_CH4troph=tau_CH4troph,
        fbactratio=fbactratio,
        O2m=O2m,
        MG_useallCpools=MG_useallCpools,
        enable_plant_transport=enable_plant_transport,
        enable_ebullition=enable_ebullition,
        Tgr=Tgr,
        refdep=refdep,
        heat_co2_act=heat_co2_act,
        heat_co2_slo=heat_co2_slo,
        heat_co2_pas=heat_co2_pas,
        heat_ch4_gen=heat_ch4_gen,
        heat_ch4_troph=heat_ch4_troph,
        perma_peat=perma_peat,
        cmax_peat=cmax_peat,
        is_peat=is_peat,
        frac1=frac1,
        frac2=frac2,
        reset_yedoma=reset_yedoma,
        zz_deep=zz_deep,
        yedoma_depth=yedoma_depth,
        yedoma_cinit_act=yedoma_cinit_act,
        yedoma_cinit_slo=yedoma_cinit_slo,
        yedoma_cinit_pas=yedoma_cinit_pas,
        yedoma_map_filename=yedoma_map_filename,
        yedoma_values=yedoma_values,
        one_day=one_day,
        one_year=one_year,
    )


def stomate_ok_leak_with_maintenance_explicit(
    *,
    biomass,
    t2m,
    t2m_longterm,
    stempdiag,
    z_soil,
    rprof,
    sla_calc,
    coeff_maint_zero,
    maint_resp_slope,
    ext_coeff,
    is_tree,
    resp_maint_part,
    soil_mc,
    turnover_daily,
    bm_to_litter_daily,
    dt_sechiba,
    one_day=86400.0,
    nslm: int | None = None,
    ndeep: int | None = None,
    **ok_leak_inputs,
) -> ExplicitOkLeakWithMaintenanceResult:
    """Compose the local ``stomate_main`` maintenance/prep block before OK_LEAK.

    Fortran provenance: ``src_stomate/stomate.f90`` lines 3244-3293 compute
    maintenance respiration, aggregate ``resp_maint_radia``/``flood_root_radia``,
    accumulate ``resp_maint_part``, build ``soil_mc_32l``, and scale
    ``turnover_daily``/``bm_to_litter`` before ``littercalc_leak``. The actual
    litter/soilcarbon/DOC path is delegated to ``stomate_ok_leak_explicit``.

    This adapter still requires explicit process state for litter, soilcarbon,
    hydrology, TF-DOC, cryoturbation, and PERMA_PEAT; it does not run
    phenology, allocation, turnover, or full ``stomate_main`` control flow.
    """

    soil_mc_arr = jnp.asarray(soil_mc)
    if nslm is None:
        nslm = soil_mc_arr.shape[1]
    if ndeep is None:
        ndeep = jnp.asarray(ok_leak_inputs["carbon_32l"]).shape[3]

    dt_days = jnp.asarray(dt_sechiba) / jnp.asarray(one_day)
    maintenance = maintenance_respiration(
        biomass,
        t2m,
        t2m_longterm,
        stempdiag,
        z_soil,
        rprof,
        sla_calc,
        coeff_maint_zero,
        maint_resp_slope,
        ext_coeff,
        is_tree,
        dt_sechiba_days=dt_days,
    )
    flood_frac = ok_leak_inputs["flood_frac"]
    resp_maint_radia, flood_root_radia = sum_resp_maint_radia(
        maintenance.resp_maint_part,
        flood_frac=flood_frac,
    )
    resp_maint_part_updated = accumulate_resp_maint_part(resp_maint_part, maintenance.resp_maint_part)
    prep = stomate_littercalc_entry_prep(
        soil_mc_arr,
        turnover_daily,
        bm_to_litter_daily,
        dt_sechiba=dt_sechiba,
        ndeep=int(ndeep),
        nslm=int(nslm),
    )

    ok_args = dict(ok_leak_inputs)
    for local_name in (
        "bm_to_litter",
        "turnover",
        "soil_mc",
        "soil_mc_32l",
        "resp_maint_part_radia",
        "flood_root_radia",
        "dt_days",
        "nslm",
        "ndeep",
        "rprof",
        "z_soil",
        "sla_calc",
    ):
        ok_args.pop(local_name, None)
    ok_args.update(
        {
            "bm_to_litter": prep.bm_to_littercalc,
            "turnover": prep.turnover_littercalc,
            "soil_mc": soil_mc_arr,
            "soil_mc_32l": prep.soil_mc_32l,
            "resp_maint_part_radia": maintenance.resp_maint_part,
            "flood_root_radia": flood_root_radia,
            "dt_days": dt_days,
            "nslm": int(nslm),
            "ndeep": int(ndeep),
            "rprof": rprof,
            "z_soil": z_soil,
            "sla_calc": sla_calc,
        }
    )
    ok_leak = stomate_ok_leak_explicit(**ok_args)
    return ExplicitOkLeakWithMaintenanceResult(
        maintenance=maintenance,
        resp_maint_radia=resp_maint_radia,
        flood_root_radia=flood_root_radia,
        resp_maint_part=resp_maint_part_updated,
        ok_leak=ok_leak,
        requires_trace=(),
    )


def stomate_ok_leak_from_post_npp_explicit(
    *,
    post_npp: ExplicitDailyCarbonKillGapTurnoverResult,
    resp_maint_part_radia,
    flood_root_radia,
    soil_mc,
    dt_sechiba,
    one_day=86400.0,
    nslm: int | None = None,
    ndeep: int | None = None,
    **ok_leak_inputs,
) -> ExplicitOkLeakFromPostNppResult:
    """Wire post-NPP turnover/litter state into the audited OK_LEAK adapter.

    Fortran provenance: ``src_stomate/stomate.f90`` lines 3288-3293 scales
    ``turnover_daily`` and ``bm_to_litter`` by ``dt_sechiba/one_day`` before
    ``littercalc_leak``; lines 3301-3489 then run the OK_LEAK litter,
    soil-carbon, and DOC path. The post-NPP producer provenance is
    ``src_stomate/stomate_lpj.f90`` lines 1118-1557. This adapter intentionally
    consumes already computed maintenance respiration and does not recompute it.
    """

    soil_mc_arr = jnp.asarray(soil_mc)
    if nslm is None:
        nslm = soil_mc_arr.shape[1]
    if ndeep is None:
        ndeep = jnp.asarray(ok_leak_inputs["carbon_32l"]).shape[3]

    prep = stomate_littercalc_entry_prep(
        soil_mc_arr,
        post_npp.turnover.turnover,
        post_npp.bm_to_litter,
        dt_sechiba=dt_sechiba,
        ndeep=int(ndeep),
        nslm=int(nslm),
    )

    ok_args = dict(ok_leak_inputs)
    for local_name in (
        "bm_to_litter",
        "turnover",
        "soil_mc",
        "soil_mc_32l",
        "resp_maint_part_radia",
        "flood_root_radia",
        "dt_days",
        "nslm",
        "ndeep",
    ):
        ok_args.pop(local_name, None)
    ok_args.update(
        {
            "bm_to_litter": prep.bm_to_littercalc,
            "turnover": prep.turnover_littercalc,
            "soil_mc": soil_mc_arr,
            "soil_mc_32l": prep.soil_mc_32l,
            "resp_maint_part_radia": resp_maint_part_radia,
            "flood_root_radia": flood_root_radia,
            "dt_days": jnp.asarray(dt_sechiba) / jnp.asarray(one_day),
            "nslm": int(nslm),
            "ndeep": int(ndeep),
        }
    )
    ok_leak = stomate_ok_leak_explicit(**ok_args)
    return ExplicitOkLeakFromPostNppResult(
        prep=prep,
        ok_leak=ok_leak,
        requires_trace=(),
    )


def stomate_lpj_outputs_from_post_npp_ok_leak_explicit(
    *,
    post_npp: ExplicitDailyCarbonKillGapTurnoverResult,
    ok_leak: ExplicitOkLeakFromPostNppResult,
    veget_max,
    z_soil,
    zf_soil,
    carb_mass_total_old,
    gpp_daily,
    prod10_total=0.0,
    prod100_total=0.0,
    contfrac=1.0,
    one_day=86400.0,
) -> ExplicitStomateLpjOutputsResult:
    """Build ``StomateLpj`` output diagnostics after post-NPP and OK_LEAK.

    Fortran provenance: post-NPP state follows ``src_stomate/stomate_lpj.f90``
    lines 1118-1557; litter/soil/DOC state follows ``stomate.f90`` lines
    3288-3489. Output diagnostic preparation follows ``stomate_lpj.f90`` lines
    1578-1663 and paper history/modelout fields lines 1677-1699 and
    2200-2246.
    """

    diagnostics = stomate_lpj_output_diagnostics(
        biomass=post_npp.turnover.biomass,
        turnover_daily=post_npp.turnover.turnover,
        bm_to_litter=post_npp.bm_to_litter,
        litter_above=ok_leak.ok_leak.littercalc.litter_above,
        litter_below=ok_leak.ok_leak.littercalc.litter_below,
        carbon_32l=ok_leak.ok_leak.soilcarbon.carbon_32l,
        doc=ok_leak.ok_leak.soilcarbon.doc,
        veget_max=veget_max,
        z_soil=z_soil,
        zf_soil=zf_soil,
        carb_mass_total_old=carb_mass_total_old,
        prod10_total=prod10_total,
        prod100_total=prod100_total,
        contfrac=contfrac,
        one_day=one_day,
    )
    modelout_fields = stomate_lpj_history_fields_from_state(
        biomass=post_npp.turnover.biomass,
        gpp_daily=gpp_daily,
        npp_daily=post_npp.daily_carbon.npp_update.npp,
    )
    # StomateLpj writes these beside the modelout inputs
    # (stomate_lpj.f90:2193, 2284-2286, 2302). Retaining them here permits
    # annual first-divergence diagnosis without retaining full day scaffolds.
    modelout_fields.update(
        {
            "MAINT_RESP": post_npp.daily_carbon.npp_update.resp_maint,
            "GROWTH_RESP": post_npp.daily_carbon.npp_update.resp_growth,
        }
    )
    if post_npp.lai_after_setlai is not None:
        modelout_fields["LAI"] = post_npp.lai_after_setlai
    if post_npp.vmax is not None:
        modelout_fields["VCMAX"] = post_npp.vmax.vcmax
    resp_parts = post_npp.daily_carbon.boundary.resp_maint_part
    bm_alloc = post_npp.daily_carbon.npp_update.bm_alloc
    modelout_fields.update(
        {
            "MAINT_RESP_AGRSAPST": resp_parts[:, :, IAGRSAPST],
            "MAINT_RESP_AGRSAPPN": resp_parts[:, :, IAGRSAPPN],
            "MAINT_RESP_AGRHRTST": resp_parts[:, :, IAGRHRTST],
            "MAINT_RESP_AGRHRTPN": resp_parts[:, :, IAGRHRTPN],
            "BM_ALLOC_LEAF": bm_alloc[:, :, ILEAF, ICARBON],
            "BM_ALLOC_SAP_AB": bm_alloc[:, :, ISAPABOVE, ICARBON],
            "BM_ALLOC_SAP_BE": bm_alloc[:, :, ISAPBELOW, ICARBON],
            "BM_ALLOC_ROOT": bm_alloc[:, :, IROOT, ICARBON],
        }
    )
    modelout = compute_modelout_from_fields(modelout_fields)
    return ExplicitStomateLpjOutputsResult(
        output_diagnostics=diagnostics,
        modelout_fields=modelout_fields,
        modelout=modelout,
        requires_trace=(),
    )


STOMATE_LPJ_PFT14_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj lines 800-942",
    "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj lines 944-1318",
    "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj lines 1321-1559",
    "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj lines 1578-2437",
)


@dataclass(frozen=True)
class StomateLpjPFT14FixedSwitches:
    """Structural switches fixed by the paper PFT14 ``StomateLpj`` path."""

    nvm: int = 14
    ok_dgvm: bool = False
    lpj_gap_const_mort: bool = True
    disable_fire: bool = True
    enable_grazing: bool = False
    update_peatfrac: bool = False
    dyn_peat: bool = False
    do_now_stomate_lcchange: bool = False
    use_age_class: bool = False
    allow_deforest_fire: bool = False
    agri_peat: bool = False
    harvest_agri: bool = True
    ok_pc: bool = False
    ok_nlim: bool = False
    agriculture: bool = True
    stomate_restart_none: bool = False
    pheno_model: tuple[str, ...] = ("none",) * 14


class StomateLpjPFT14Result(NamedTuple):
    """Exact subowner result and state written by one fixed PFT14 LPJ call."""

    chain: ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult
    state_after: object
    process_order: tuple[str, ...]
    provenance: tuple[str, ...] = STOMATE_LPJ_PFT14_PROVENANCE


def validate_stomate_lpj_pft14_switches(switches: StomateLpjPFT14FixedSwitches) -> None:
    """Reject structural branches outside the audited paper process graph."""

    expected = StomateLpjPFT14FixedSwitches()
    mismatches = tuple(
        f"{name}={getattr(switches, name)!r} (required {getattr(expected, name)!r})"
        for name in expected.__dataclass_fields__
        if getattr(switches, name) != getattr(expected, name)
    )
    if mismatches:
        raise NotImplementedError(
            "unsupported StomateLpj PFT14 fixed-switch branch: " + ", ".join(mismatches)
        )


def _lpj_state_mapping(state: object) -> dict[str, object]:
    if isinstance(state, Mapping):
        return dict(state)
    if hasattr(state, "_asdict"):
        return dict(state._asdict())
    raise TypeError("state must be a mapping or NamedTuple-like state object")


def _lpj_write_state(state: object, updates: Mapping[str, object]) -> object:
    if isinstance(state, Mapping):
        result = dict(state)
        result.update(updates)
        return result
    fields = set(state._fields)
    return state._replace(**{name: value for name, value in updates.items() if name in fields})


def _lpj_seed_owned(
    group: dict[str, object],
    values: Mapping[str, object],
    names: tuple[str, ...],
    *,
    boundary: str,
) -> None:
    overlap = tuple(name for name in names if name in group)
    if overlap:
        raise ValueError(f"{boundary} must not override StomateLpj-owned fields: " + ", ".join(overlap))
    missing = tuple(name for name in names if name not in values)
    if missing:
        raise ValueError(f"state is missing {boundary} fields: " + ", ".join(missing))
    group.update({name: values[name] for name in names})


def stomate_lpj_main_pft14_fixed(
    *,
    state: object,
    prescribe_inputs: Mapping[str, object],
    constraints_inputs: Mapping[str, object],
    phenology_inputs: Mapping[str, object],
    alloc_inputs: Mapping[str, object],
    post_npp_inputs: Mapping[str, object],
    firstcall: bool,
    switches: StomateLpjPFT14FixedSwitches = StomateLpjPFT14FixedSwitches(),
) -> StomateLpjPFT14Result:
    """Run reachable PFT14 ``StomateLpj`` branches in Fortran source order.

    Process algebra is delegated to the exact owners composed by
    ``stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit``.
    Land-point arrays and continuous parameters remain caller supplied. The
    fixed paper switches are installed here and cannot be overridden through a
    process-input bundle.

    Fortran provenance: ``src_stomate/stomate_lpj.f90::StomateLpj`` lines
    800-910 initialize call-local fields; lines 919-942 skip initial crown for
    the fixed mortality switches; lines 944-1318 execute prescribe through
    turnover while skipping DGVM/fire/establishment; lines 1321-1559 execute
    ordinary cover, agricultural harvest, final LAI, and vmax while skipping
    grazing, dynamic peat, age classes, and LCC; lines 1578-2437 prepare output
    state. ``cover`` receives current-call respiration/GPP outputs, not restart
    values.
    """

    validate_stomate_lpj_pft14_switches(switches)
    values = _lpj_state_mapping(state)
    for required in ("biomass", "veget_max", "gpp_daily"):
        if required not in values:
            raise ValueError(f"state is missing StomateLpj field: {required}")
    biomass = jnp.asarray(values["biomass"])
    veget_max = jnp.asarray(values["veget_max"])
    gpp_daily = jnp.asarray(values["gpp_daily"])
    if biomass.ndim < 3 or veget_max.ndim != 2 or gpp_daily.ndim != 2:
        raise ValueError("biomass, veget_max, and gpp_daily must carry land and PFT axes")
    if biomass.shape[:2] != veget_max.shape or gpp_daily.shape != veget_max.shape:
        raise ValueError("StomateLpj land/PFT axes must agree")
    if veget_max.shape[0] < 1 or veget_max.shape[1] != switches.nvm:
        raise ValueError("the PFT14 owner requires shape (nland, 14) with nland >= 1")

    prescribe = dict(prescribe_inputs)
    constraints = dict(constraints_inputs)
    phenology = dict(phenology_inputs)
    alloc = dict(alloc_inputs)
    post = dict(post_npp_inputs)

    fixed_group_fields = {
        "prescribe": (prescribe, ("co2_to_bm", "ok_dgvm", "lpj_gap_const_mort", "firstcall", "stomate_restart_none")),
        "constraints": (constraints, ("agriculture",)),
        "phenology": (phenology, ("pheno_model", "active_pft_mask")),
    }
    for boundary, (group, names) in fixed_group_fields.items():
        overlap = tuple(name for name in names if name in group)
        if overlap:
            raise ValueError(f"{boundary} must not override fixed/owned fields: " + ", ".join(overlap))

    _lpj_seed_owned(
        prescribe,
        values,
        ("veget_max", "pft_present", "everywhere", "when_growthinit", "biomass", "leaf_frac", "ind", "cn_ind"),
        boundary="prescribe",
    )
    prescribe["co2_to_bm"] = jnp.zeros_like(gpp_daily)
    prescribe["ok_dgvm"] = switches.ok_dgvm
    prescribe["lpj_gap_const_mort"] = switches.lpj_gap_const_mort
    prescribe["firstcall"] = bool(firstcall)
    prescribe["stomate_restart_none"] = switches.stomate_restart_none

    _lpj_seed_owned(constraints, values, ("adapted", "regenerate"), boundary="constraints")
    constraints["agriculture"] = switches.agriculture

    _lpj_seed_owned(phenology, values, ("leaf_age",), boundary="phenology")
    phenology["pheno_model"] = switches.pheno_model
    phenology["active_pft_mask"] = tuple(bool(value) for value in np.any(np.asarray(values["pft_present"]), axis=0))

    _lpj_seed_owned(alloc, values, ("age", "leaf_age"), boundary="allocation")

    post_owned_from_state = (
        "npp_longterm",
        "turnover_longterm",
        "lm_lastyearmax",
        "rip_time",
        "height",
        "tmin_spring_time",
        "turnover_time",
    )
    _lpj_seed_owned(post, values, post_owned_from_state, boundary="post-NPP")
    zero_pft = jnp.zeros_like(gpp_daily)
    zero_biomass = jnp.zeros_like(biomass)
    for name, value in {
        "gpp_daily": gpp_daily,
        "resp_maint_part": values["resp_maint_part"],
        "bm_to_litter": zero_biomass,
        "co2_to_bm": zero_pft,
    }.items():
        if name in post:
            raise ValueError(f"post-NPP must not override StomateLpj-owned field: {name}")
        post[name] = value

    litter_avail_frac = post.pop("litter_avail_frac", None)
    if litter_avail_frac is None:
        raise ValueError("post-NPP requires dynamic litter_avail_frac for the disabled-fire availability update")
    cover_state_names = (
        "litter",
        "carbon",
        "fuel_1hr",
        "fuel_10hr",
        "fuel_100hr",
        "fuel_1000hr",
        "resp_hetero",
        "deepC_a",
        "deepC_s",
        "deepC_p",
    )
    missing = tuple(name for name in cover_state_names if name not in values)
    if missing:
        raise ValueError("state is missing cover fields: " + ", ".join(missing))
    availability = stomate_lpj_litter_availability_from_fraction(
        litter_above_carbon=jnp.asarray(values["litter"])[:, :, :, IABOVE, ICARBON],
        litter_avail_frac=litter_avail_frac,
    )
    fixed_post = {
        "ok_dgvm": switches.ok_dgvm,
        "lpj_gap_const_mort": switches.lpj_gap_const_mort,
        "wire_cover": True,
        "update_peatfrac": switches.update_peatfrac,
        "ok_pc_cover": switches.ok_pc,
        "wire_harvest_agri": switches.harvest_agri,
        "wire_vmax": True,
        "ok_nlim_vmax": switches.ok_nlim,
        "wire_output_diagnostics": True,
        "wire_modelout": True,
    }
    overlap = tuple(name for name in fixed_post if name in post)
    if overlap:
        raise ValueError("post-NPP must not override fixed paper switches: " + ", ".join(overlap))
    post.update(fixed_post)
    cover_owned = tuple(
        f"cover_{name}"
        for name in (
            "litter",
            "litter_avail",
            "litter_not_avail",
            "carbon",
            "fuel_1hr",
            "fuel_10hr",
            "fuel_100hr",
            "fuel_1000hr",
            "co2_fire",
            "resp_hetero",
            "deepC_a",
            "deepC_s",
            "deepC_p",
            "veget_max_old",
        )
    )
    overlap = tuple(name for name in cover_owned if name in post)
    if overlap:
        raise ValueError("post-NPP must not override source-ordered cover state: " + ", ".join(overlap))
    post.update(
        cover_litter=values["litter"],
        cover_litter_avail=availability.litter_avail,
        cover_litter_not_avail=availability.litter_not_avail,
        cover_carbon=values["carbon"],
        cover_fuel_1hr=values["fuel_1hr"],
        cover_fuel_10hr=values["fuel_10hr"],
        cover_fuel_100hr=values["fuel_100hr"],
        cover_fuel_1000hr=values["fuel_1000hr"],
        cover_co2_fire=zero_pft,
        cover_resp_hetero=values["resp_hetero"],
        cover_deepC_a=values["deepC_a"],
        cover_deepC_s=values["deepC_s"],
        cover_deepC_p=values["deepC_p"],
        cover_veget_max_old=veget_max,
    )

    chain = stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit(
        prescribe_inputs=prescribe,
        constraints_inputs=constraints,
        phenology_inputs=phenology,
        alloc_inputs=alloc,
        post_npp_inputs=post,
    )
    post_result = chain.post_npp
    cover = post_result.cover
    if (
        cover is None
        or post_result.harvest_agri is None
        or post_result.vmax is None
        or post_result.output_diagnostics is None
    ):
        raise RuntimeError("fixed StomateLpj subowner omitted cover, harvest, vmax, or diagnostics output")

    final_kill = post_result.kill_after_gap
    turnover = post_result.turnover
    npp = post_result.daily_carbon.npp_update
    updates = {
        "adapted": chain.constraints.adapted,
        "regenerate": chain.constraints.regenerate,
        "pft_present": final_kill.pft_present,
        "everywhere": final_kill.everywhere,
        "when_growthinit": final_kill.when_growthinit,
        "ind": final_kill.ind,
        "cn_ind": final_kill.cn_ind,
        "npp_longterm": final_kill.npp_longterm,
        "rip_time": final_kill.rip_time,
        "senescence": turnover.senescence,
        "age": turnover.age,
        "leaf_age": post_result.vmax.leaf_age,
        "leaf_frac": post_result.vmax.leaf_frac,
        "sla_calc": post_result.daily_carbon.age_sla.sla_calc,
        "turnover_time": turnover.turnover_time,
        "biomass": cover.biomass,
        "veget_max": cover.veget_max,
        "litter": cover.litter,
        "litter_avail": cover.litter_avail,
        "litter_not_avail": cover.litter_not_avail,
        "carbon": cover.carbon,
        "fuel_1hr": cover.fuel_1hr,
        "fuel_10hr": cover.fuel_10hr,
        "fuel_100hr": cover.fuel_100hr,
        "fuel_1000hr": cover.fuel_1000hr,
        "deepC_a": cover.deepC_a,
        "deepC_s": cover.deepC_s,
        "deepC_p": cover.deepC_p,
        "soilc_total": cover.deepC_a + cover.deepC_s + cover.deepC_p,
        "co2_to_bm": cover.co2_to_bm,
        "co2_fire": cover.co2_fire,
        "resp_hetero": cover.resp_hetero,
        "resp_maint": npp.resp_maint,
        "resp_growth": npp.resp_growth,
        "gpp_daily": cover.gpp_daily,
        "npp_daily": npp.npp,
        "turnover_daily": post_result.harvest_agri.turnover_daily,
        "bm_to_litter": post_result.harvest_agri.bm_to_litter,
        "harvest_above": post_result.harvest_agri.harvest_above,
        "crop_export": turnover.c_export,
        "lai": post_result.lai_after_setlai,
        "vcmax": post_result.vmax.vcmax,
        "woodmass_ind": jnp.zeros_like(gpp_daily),
        "carb_mass_total": post_result.output_diagnostics.carb_mass_total,
    }
    process_order = (
        "initialize_call_locals",
        "prescribe",
        "constraints",
        "phenology",
        "alloc",
        "setlai_after_alloc",
        "npp_calc",
        "skip_spitfire_disable_fire",
        "litter_availability",
        "gap",
        "turn",
        "skip_light_establish_static_mortality",
        "skip_grassland_management",
        "cover",
        "harvest",
        "skip_age_class_and_lcchange",
        "setlai",
        "vmax",
        "output_diagnostics",
    )
    return StomateLpjPFT14Result(
        chain=chain,
        state_after=_lpj_write_state(state, updates),
        process_order=process_order,
    )
