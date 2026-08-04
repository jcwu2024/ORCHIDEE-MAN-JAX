"""HYDROL Phase 1A/1B kernels.

This module contains the audited Phase 1A selected-coefficient algebra and the
Phase 1B source-driven mineral CWRR table builder. It still intentionally stops
short of a full `hydrol_soil` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Mapping, NamedTuple

from jax import config, jit

config.update("jax_enable_x64", True)

import numpy as np
import jax
import jax.numpy as jnp

from jax_orchidee.sechiba.enerbil_bridge import (
    EnerbilToHydrolBoundaryValidation,
    validate_enerbil_to_hydrol_boundary_payload,
)
from jax_orchidee.sechiba.hydrol_thermosoil_completion import hydro_subgrid_main  # noqa: E402
from jax_orchidee.sechiba.routing import routing_zero_outputs  # noqa: E402
from jax_orchidee.sechiba.thermosoil import thermosoil_diaglev_from_ptn


USDA_NVAN = (
    2.68,
    2.28,
    1.89,
    1.41,
    1.37,
    1.56,
    1.48,
    1.23,
    1.31,
    1.23,
    1.09,
    1.09,
)

USDA_AVAN = (
    0.0145,
    0.0124,
    0.0075,
    0.0020,
    0.0016,
    0.0036,
    0.0059,
    0.0010,
    0.0019,
    0.0027,
    0.0005,
    0.0008,
)
USDA_MCR = (
    0.045,
    0.057,
    0.065,
    0.067,
    0.034,
    0.078,
    0.100,
    0.089,
    0.095,
    0.100,
    0.070,
    0.068,
)
USDA_KS = (
    7128.0,
    3501.6,
    1060.8,
    108.0,
    60.0,
    249.6,
    314.4,
    16.8,
    62.4,
    28.8,
    4.8,
    48.0,
)
PEAT_NVAN = 1.38
PEAT_AVAN = 0.00507
PEAT_MCR = 0.15
PEAT_MCS = 0.90
PEAT_KS = 2120.0
PEAT_MCW = 0.210
PEAT_MCF = 0.406
PEAT_MC_AWET = 0.25
PEAT_MC_ADRY = 0.1
PEAT_PCENT = 0.8
USDA_MCS = (
    0.43,
    0.41,
    0.41,
    0.45,
    0.46,
    0.43,
    0.39,
    0.43,
    0.41,
    0.38,
    0.36,
    0.38,
)
USDA_MCF = (
    0.0493,
    0.0710,
    0.1218,
    0.2402,
    0.2582,
    0.1654,
    0.1695,
    0.3383,
    0.2697,
    0.2672,
    0.3370,
    0.3469,
)
USDA_MCW = (
    0.0450,
    0.0570,
    0.0657,
    0.1039,
    0.0901,
    0.0884,
    0.1112,
    0.1967,
    0.1496,
    0.1704,
    0.2665,
    0.2707,
)
USDA_VG_M = (
    0.6269,
    0.5614,
    0.4709,
    0.2908,
    0.2701,
    0.3590,
    0.3243,
    0.1870,
    0.2366,
    0.1870,
    0.0826,
    0.0826,
)
USDA_VG_PSI_FC = (
    1000.0,
    1000.0,
    1000.0,
    3300.0,
    3300.0,
    3300.0,
    3300.0,
    3300.0,
    3300.0,
    3300.0,
    3300.0,
    3300.0,
)
USDA_VG_PSI_WP = (150000.0,) * 12
SOILC_MAX = 130000.0
POROS_ORG = 0.92
USDA_PCENT = (0.8,) * 12
USDA_MC_AWET = (0.25,) * 12
USDA_MC_ADRY = (0.1,) * 12


class HydrolSoilCoefResult(NamedTuple):
    """Selected mineral HYDROL coefficients and diagnostics."""

    a: jnp.ndarray
    b: jnp.ndarray
    d: jnp.ndarray
    k: jnp.ndarray
    mc_used: jnp.ndarray
    k_eval: jnp.ndarray


class HydrolRefSocThresholds(NamedTuple):
    """HYDROL water thresholds after optional 1D SOC mixing."""

    mcs: jnp.ndarray
    mcw: jnp.ndarray
    mcf: jnp.ndarray
    zx1: jnp.ndarray
    provenance: tuple[str, ...]


class MineralCWRRBinSelection(NamedTuple):
    """Fortran-style mineral CWRR bin selection."""

    mc_used: jnp.ndarray
    bin_i: jnp.ndarray
    bin_offset: jnp.ndarray


class HydrolSoilCoefTableResult(NamedTuple):
    """Mineral `hydrol_soil_coef` outputs from source-built tables."""

    a: jnp.ndarray
    b: jnp.ndarray
    d: jnp.ndarray
    k: jnp.ndarray
    mc_used: jnp.ndarray
    bin_i: jnp.ndarray
    bin_offset: jnp.ndarray
    k_eval: jnp.ndarray
    a_raw: jnp.ndarray
    b_raw: jnp.ndarray
    d_raw: jnp.ndarray
    k_floor_raw: jnp.ndarray


class SelectedMineralHydrolCoefficients(NamedTuple):
    """Already-selected mineral CWRR coefficients for one HYDROL bin.

    These values correspond to the selected raw `a_lin`, `b_lin`, `d_lin`, and
    lower conductivity floor used by `hydrol_soil_coef`. Phase 1A receives them
    from audited traces; Phase 1B should source them from the Fortran table
    builder after its parameters are audited.
    """

    a_raw: jnp.ndarray
    b_raw: jnp.ndarray
    d_raw: jnp.ndarray
    k_floor_raw: jnp.ndarray


class MineralCWRRTables(NamedTuple):
    """Source-built mineral CWRR lookup tables.

    Arrays use Python offsets: Fortran bin or bound `i` is stored at
    `i - imin`. The incoming `njsc` texture code remains Fortran 1-based.
    """

    mc_lin: jnp.ndarray
    k_lin: jnp.ndarray
    a_lin: jnp.ndarray
    b_lin: jnp.ndarray
    d_lin: jnp.ndarray
    kfact: jnp.ndarray
    nfact: jnp.ndarray
    afact: jnp.ndarray
    nvan_mod: jnp.ndarray
    avan_mod: jnp.ndarray
    imin: int
    imax: int


class PeatCWRRTables(NamedTuple):
    """Source-built peat CWRR lookup tables."""

    mc_lin: jnp.ndarray
    k_lin: jnp.ndarray
    a_lin: jnp.ndarray
    b_lin: jnp.ndarray
    d_lin: jnp.ndarray
    kfact: jnp.ndarray
    nfact: jnp.ndarray
    afact: jnp.ndarray
    nvan_mod: jnp.ndarray
    avan_mod: jnp.ndarray
    mcr: float
    mcs: float
    ks: float
    imin: int
    imax: int


class HydrolVarInitResult(NamedTuple):
    """Source-ordered state produced by ``hydrol_var_init`` through line 4624."""

    zz: jnp.ndarray
    dz: jnp.ndarray
    dh: jnp.ndarray
    mcs: jnp.ndarray
    mcw: jnp.ndarray
    mcf: jnp.ndarray
    nroot: jnp.ndarray
    humcste_use: jnp.ndarray
    mineral_tables: MineralCWRRTables
    peat_tables: PeatCWRRTables | None
    mx_eau_var: jnp.ndarray
    tmcs: jnp.ndarray
    tmcr: jnp.ndarray
    tmc: jnp.ndarray
    tmc_litter: jnp.ndarray
    tmc_litter_wilt: jnp.ndarray
    tmc_litter_res: jnp.ndarray
    tmc_litter_field: jnp.ndarray
    tmc_litter_sat: jnp.ndarray
    tmc_litter_awet: jnp.ndarray
    tmc_litter_adry: jnp.ndarray
    tmc_trampling: jnp.ndarray
    tmc_topgrass: jnp.ndarray
    humrelv: jnp.ndarray
    soilmoist: jnp.ndarray
    shumdiag_perma: jnp.ndarray
    shumdiag_peat: jnp.ndarray
    shumdiag_croppeat: jnp.ndarray
    shumdiag_man: jnp.ndarray
    drysoil_frac: jnp.ndarray
    profil_froz_hydro_ns: jnp.ndarray
    mc: jnp.ndarray
    mcl: jnp.ndarray
    water2infilt: jnp.ndarray
    resdist: jnp.ndarray
    vegtot_old: jnp.ndarray
    qsintveg: jnp.ndarray
    parameters: Mapping[str, float]
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class HydrolInitPFT14Switches:
    """Fixed paper-run branches used by the PFT14 ``hydrol_init`` owner."""

    soiltype_classif: int = 12
    ok_freeze_cwrr: bool = True
    ok_explicitsnow: bool = True
    check_waterbal: bool = False
    peat_hydro: bool = True
    peat_nodr: bool = True
    tides: bool = True
    agri_peat: bool = False
    agri_drain: bool = False
    ok_pc: bool = False
    ok_leak: bool = True


class HydrolInitPFT14Result(NamedTuple):
    """Composed, source-ordered PFT14 initialization and audit state."""

    state: Mapping[str, object]
    span_state: Mapping[str, object]
    cold_start: HydrolColdStartState
    restart_static: HydrolVarInitResult
    vegetation_static: HydrolVegupdStaticState
    soil_parameters: Mapping[str, jnp.ndarray]
    allocation_only: Mapping[str, tuple[int, ...]]
    restart_inputs: tuple[str, ...]
    process_order: tuple[str, ...]
    switches: HydrolInitPFT14Switches
    provenance: tuple[str, ...]
    notes: tuple[str, ...]


class DrainageCorrectionResult(NamedTuple):
    """Drainage after HYDROL conservation correction."""

    dr_corrnum: jnp.ndarray
    dr_after: jnp.ndarray


class HydrolOverSaturationCorrectionResult(NamedTuple):
    """HYDROL over-saturation clipping result."""

    mc: jnp.ndarray
    excess: jnp.ndarray
    rudr_corr: jnp.ndarray
    is_over_mcs: jnp.ndarray


class HydrolOverSaturationRoutingResult(NamedTuple):
    """HYDROL over-saturation correction routed to runoff or drainage."""

    mc: jnp.ndarray
    excess: jnp.ndarray
    ru_corr_ns: jnp.ndarray
    dr_corr_ns: jnp.ndarray
    ru_ns: jnp.ndarray
    dr_ns: jnp.ndarray
    check_over_ns: jnp.ndarray | None
    is_over_mcs: jnp.ndarray


class HydrolNegativeRunoffCorrectionResult(NamedTuple):
    """HYDROL negative runoff correction into drainage."""

    ru_ns: jnp.ndarray
    dr_ns: jnp.ndarray
    ru_corr2_ns: jnp.ndarray


class HydrolForcedWaterTableResult(NamedTuple):
    """HYDROL forced water-table saturation result."""

    mc: jnp.ndarray
    dmc: jnp.ndarray
    dr_force_ns: jnp.ndarray
    dr_ns: jnp.ndarray


class HydrolWaterTableDepthResult(NamedTuple):
    """HYDROL effective water-table depth diagnostic."""

    wtd_ns: jnp.ndarray


class HydrolRunoffPeatTideRoutingResult(NamedTuple):
    """HYDROL runoff-to-peat/tide routing inside the soil-tile loop."""

    ru_ns: jnp.ndarray
    runoff2peat: jnp.ndarray
    water2infilt: jnp.ndarray
    run2peat: jnp.ndarray
    run2man: jnp.ndarray
    wt_ab: jnp.ndarray
    wt_ab_tide: jnp.ndarray


class HydrolRunoffPeatPostLoopResult(NamedTuple):
    """HYDROL runoff-to-peat post-loop water reinjection."""

    ru_ns: jnp.ndarray
    water2infilt: jnp.ndarray
    tmc: jnp.ndarray
    tmc_soil: jnp.ndarray
    run2peat: jnp.ndarray


class HydrolUnderResidualCorrectionResult(NamedTuple):
    """HYDROL under-residual smoothing result."""

    mc: jnp.ndarray
    excess_top: jnp.ndarray
    is_under_mcr: jnp.ndarray
    check_under_ns: jnp.ndarray | None


class HydrolSoilSetupResult(NamedTuple):
    """Tridiagonal setup coefficients for HYDROL soil solve."""

    e: jnp.ndarray
    f: jnp.ndarray
    g1: jnp.ndarray
    ep: jnp.ndarray
    fp: jnp.ndarray
    gp: jnp.ndarray


class HydrolSoilTridiagResult(NamedTuple):
    """HYDROL tridiagonal solve result and forward-sweep diagnostics."""

    mcl: jnp.ndarray
    bet: jnp.ndarray
    gam: jnp.ndarray


class HydrolSoilRHSResult(NamedTuple):
    """HYDROL RHS and left-hand tridiagonal coefficients."""

    rhs: jnp.ndarray
    e: jnp.ndarray
    f: jnp.ndarray
    g1: jnp.ndarray


class HydrolSoilSolveDiagnostics(NamedTuple):
    """Composed HYDROL solve diagnostics, not a full `hydrol_soil` step."""

    rhs: jnp.ndarray
    mcl_after: jnp.ndarray
    bet: jnp.ndarray
    gam: jnp.ndarray
    dr_before_corr: jnp.ndarray | None
    dr_corrnum: jnp.ndarray | None
    dr_after_corr: jnp.ndarray | None


class HydrolSoilExplicitStepResult(NamedTuple):
    """Explicit-input HYDROL solve adapter result."""

    rhs: jnp.ndarray
    mcl_after_tridiag: jnp.ndarray
    bet: jnp.ndarray
    gam: jnp.ndarray
    tmci: jnp.ndarray
    tmcf: jnp.ndarray
    dr_before_corr: jnp.ndarray
    check_tr_ns: jnp.ndarray
    dr_corrnum: jnp.ndarray
    dr_after_corr: jnp.ndarray
    mc_after_mcl_update: jnp.ndarray
    over_mcs: HydrolOverSaturationCorrectionResult | None
    over_mcs_routing: HydrolOverSaturationRoutingResult | None
    negative_runoff: HydrolNegativeRunoffCorrectionResult | None
    forced_water_table: HydrolForcedWaterTableResult | None
    water_table_depth: HydrolWaterTableDepthResult | None
    under_mcr: HydrolUnderResidualCorrectionResult | None
    mc_final: jnp.ndarray
    ru_ns_final: jnp.ndarray | None
    dr_ns_final: jnp.ndarray


class HydrolSoilFluxDiagnostics(NamedTuple):
    """Vertical liquid water fluxes diagnosed after the HYDROL soil solve."""

    qflux: jnp.ndarray
    surface_balance: jnp.ndarray


class HydrolCanopResult(NamedTuple):
    """Canopy interception update from ``hydrol_canop``."""

    qsintveg: jnp.ndarray
    precisol: jnp.ndarray
    precip2canopy: jnp.ndarray
    precip2ground: jnp.ndarray
    canopy2ground: jnp.ndarray


class HydrolFloodResult(NamedTuple):
    """Floodplain reservoir update from ``hydrol_flood``."""

    vevapflo: jnp.ndarray
    flood_res: jnp.ndarray
    floodout: jnp.ndarray
    precisol: jnp.ndarray
    subsinksoil: jnp.ndarray


class HydrolSplitSoilResult(NamedTuple):
    """PFT-to-soil-tile flux split from ``hydrol_split_soil``."""

    precisol_ns: jnp.ndarray
    vevapnu_ns: jnp.ndarray
    ae_ns: jnp.ndarray
    vevapnu_old: jnp.ndarray
    tr_ns: jnp.ndarray
    rootsink: jnp.ndarray


class HydrolVegupdStaticState(NamedTuple):
    """Static PFT/soil-tile masks from the post-``hydrol_tmc_update`` block."""

    mask_veget: jnp.ndarray
    mask_soiltile: jnp.ndarray
    vegetmax_soil: jnp.ndarray
    frac_bare: jnp.ndarray
    frac_bare_ns: jnp.ndarray


class HydrolTmcUpdateResult(NamedTuple):
    """HYDROL state after vegetation and soil-tile fraction migration."""

    mc: jnp.ndarray
    water2infilt: jnp.ndarray
    qsintveg: jnp.ndarray
    drain_upd: jnp.ndarray
    runoff_upd: jnp.ndarray
    tmc: jnp.ndarray
    humtot: jnp.ndarray
    resdist: jnp.ndarray
    water_balance_error: jnp.ndarray | None
    water_balance_failed: jnp.ndarray | None
    vegtot_updated: jnp.ndarray
    soil_updated: jnp.ndarray


class HydrolSoilSurfaceWaterSetupResult(NamedTuple):
    """Surface flux setup before ``hydrol_soil_infilt``."""

    temp: jnp.ndarray
    water2infilt: jnp.ndarray
    water2extract: jnp.ndarray
    ae_ns: jnp.ndarray
    flux_top: jnp.ndarray
    flux_infilt: jnp.ndarray


class HydrolIrrigationDemandResult(NamedTuple):
    """Normalized HYDROL irrigation demand ratio by PFT."""

    irrig_demand_ratio: jnp.ndarray


class HydrolRoutingSoilWaterSplitResult(NamedTuple):
    """Routing/reinfiltration/irrigation split before the soil-tile loop."""

    returnflow_soil: jnp.ndarray
    reinfiltration_soil: jnp.ndarray
    irrigation_soil: jnp.ndarray
    irrig_fin: jnp.ndarray


class HydrolSoilInfiltResult(NamedTuple):
    """Explicit ``hydrol_soil_infilt`` infiltration update for one soil tile."""

    mc: jnp.ndarray
    qinfilt: jnp.ndarray
    ru_infilt: jnp.ndarray
    infilt_tot: jnp.ndarray
    wat_inf_first: jnp.ndarray
    dt_remaining: jnp.ndarray
    check: jnp.ndarray | None


class HydrolSoilTileExplicitStepResult(NamedTuple):
    """Long explicit-input HYDROL soil-tile step through main solve."""

    surface: HydrolSoilSurfaceWaterSetupResult
    infilt: HydrolSoilInfiltResult
    coef_before_infilt: HydrolSoilCoefTableResult | None
    coef_after_infilt: HydrolSoilCoefTableResult | None
    setup: HydrolSoilSetupResult
    solve: HydrolSoilExplicitStepResult
    mclint_for_flux: jnp.ndarray
    k_bottom: jnp.ndarray
    ru_ns_after_reinf: jnp.ndarray
    water2infilt_after_reinf: jnp.ndarray


class HydrolSoilAllTilesExplicitResult(NamedTuple):
    """Explicit all-soil-tile HYDROL soil step result."""

    tile_results: tuple[HydrolSoilTileExplicitStepResult, ...]
    mc: jnp.ndarray
    mcl: jnp.ndarray
    profil_froz_hydro_ns: jnp.ndarray
    qflux: jnp.ndarray
    ru_ns: jnp.ndarray
    dr_ns: jnp.ndarray
    water2infilt: jnp.ndarray
    tmc: jnp.ndarray
    tmc_soil: jnp.ndarray
    wtd_ns: jnp.ndarray
    is_under_mcr: jnp.ndarray
    runoff2peat: jnp.ndarray
    run2peat: jnp.ndarray
    run2man: jnp.ndarray
    wt_ab: jnp.ndarray
    wt_ab_tide: jnp.ndarray
    evap_bare_limit_alt: HydrolAltResidualSolveResult | None
    tmcint_for_evap_bare_limit: jnp.ndarray | None


class HydrolModuleExplicitResult(NamedTuple):
    """Explicit HYDROL module-step assembly result."""

    vegupd: HydrolVegupdStaticState
    canop: HydrolCanopResult
    flood: HydrolFloodResult
    split: HydrolSplitSoilResult
    soil: HydrolSoilAllTilesExplicitResult


class HydrolModuleOutputs(NamedTuple):
    """HYDROL module-level outputs used by downstream SECHIBA/STOMATE."""

    runoff_per_soil: jnp.ndarray
    drainage_per_soil: jnp.ndarray
    runoff2peat: jnp.ndarray
    tmc: jnp.ndarray
    tmc_soil: jnp.ndarray
    wat_flux: jnp.ndarray
    wtd: jnp.ndarray
    wtd_ns: jnp.ndarray
    precip2canopy: jnp.ndarray
    precip2ground: jnp.ndarray
    canopy2ground: jnp.ndarray
    floodout: jnp.ndarray


class HydrolModuleDiagnostics(NamedTuple):
    """HYDROL downstream diagnostic boundary arrays."""

    layer_by_tile: tuple[HydrolSoilLayerDiagnostics, ...]
    stress_by_tile: tuple[HydrolWaterStressDiagnostics, ...]
    nroot: jnp.ndarray
    soil_wet_ns: jnp.ndarray
    us: jnp.ndarray
    humrelv: jnp.ndarray
    vegstressv: jnp.ndarray
    humrel: jnp.ndarray
    vegstress: jnp.ndarray
    profil_froz_hydro: jnp.ndarray | None
    soilmoist: jnp.ndarray
    soilmoist_liquid: jnp.ndarray
    mc_layh: jnp.ndarray
    mcl_layh: jnp.ndarray
    mc_layh_s: jnp.ndarray
    mcl_layh_s: jnp.ndarray
    shumdiag: jnp.ndarray
    shumdiag_perma: jnp.ndarray
    shumdiag_peat: jnp.ndarray
    shumdiag_croppeat: jnp.ndarray
    shumdiag_man: jnp.ndarray
    undermcr: jnp.ndarray
    tmc_litter: jnp.ndarray
    soil_wet_litter: jnp.ndarray
    tmc_trampling: jnp.ndarray
    tmc_topgrass: jnp.ndarray
    liqwt_ratio: jnp.ndarray
    wtp: jnp.ndarray
    fwet_new: jnp.ndarray | None
    mc_peat_above: jnp.ndarray
    mc_croppeat_above: jnp.ndarray
    mc_man_above: jnp.ndarray
    k_litt: jnp.ndarray
    litterhumdiag: jnp.ndarray
    drysoil_frac: jnp.ndarray
    runoff: jnp.ndarray
    drainage: jnp.ndarray
    humtot: jnp.ndarray
    vevapnu: jnp.ndarray
    evap_bare_lim: jnp.ndarray | None
    ae_ns: jnp.ndarray
    evap_bare_lim_ns: jnp.ndarray | None


class HydrolMainResult(NamedTuple):
    """Source-ordered ``hydrol_main`` result for lines 1159-1650."""

    module: HydrolModuleExplicitResult
    diagnostics: HydrolModuleDiagnostics
    outputs: HydrolModuleOutputs
    snow_step: HydrolExplicitSnowStepResult | None
    bucket_snow_step: HydrolBucketSnowResult | None
    irrigation_demand: HydrolIrrigationDemandResult
    routing: HydrolRoutingSoilWaterSplitResult
    kfact_root: jnp.ndarray
    fsat: jnp.ndarray
    fwet: jnp.ndarray
    fwet_out: jnp.ndarray
    fwt1: jnp.ndarray
    fwt2: jnp.ndarray
    fwt3: jnp.ndarray
    fwt4: jnp.ndarray
    runoff: jnp.ndarray
    drainage: jnp.ndarray
    soil_deficit: jnp.ndarray
    soilwet: jnp.ndarray
    delintercept: jnp.ndarray
    delsoilmoist: jnp.ndarray
    delswe: jnp.ndarray
    humtot_top: jnp.ndarray
    twbr: jnp.ndarray
    land_nroot: jnp.ndarray
    land_dlh: jnp.ndarray
    land_mcs: jnp.ndarray
    water_balance: HydrolWaterBalanceResult | None
    state_writeback: Mapping[str, jnp.ndarray]
    provenance: tuple[str, ...]


class HydrolInitializeCompletionResult(NamedTuple):
    """Fixed PFT14 dispatch state from ``hydrol_initialize`` lines 625-904."""

    peat_nodr: bool
    ok_ru2peat: bool
    ok_wt_ab: bool
    max_wt_ab: float
    topmodel_state: Mapping[str, jnp.ndarray]
    alma: HydrolAlmaResult | None
    water_balance: HydrolWaterBalanceResult | None
    tot_melt: jnp.ndarray
    itopmax: int
    soilmoist_out: jnp.ndarray
    provenance: tuple[str, ...]


class HydrolSoilTailDiagnostics(NamedTuple):
    """Grid aggregates and exports after the ``hydrol_soil`` tile loop."""

    wtp_tide: float | jnp.ndarray
    dwtp_tide: float | jnp.ndarray
    wtd: jnp.ndarray
    ru_corr: jnp.ndarray
    ru_corr2: jnp.ndarray
    dr_corr: jnp.ndarray
    dr_corrnum: jnp.ndarray
    dr_force: jnp.ndarray
    ru_infilt: jnp.ndarray
    qinfilt: jnp.ndarray
    check_infilt: jnp.ndarray
    check_tr: jnp.ndarray
    check_over: jnp.ndarray
    check_under: jnp.ndarray
    evap_bare_lim_ns: jnp.ndarray
    evap_bare_lim: jnp.ndarray
    r_soil: jnp.ndarray
    soil_mc: jnp.ndarray
    drainage_per_soil: jnp.ndarray
    runoff_per_soil: jnp.ndarray
    wat_flux: jnp.ndarray
    provenance: tuple[str, ...]


class HydrolScientificField(NamedTuple):
    """One pure scientific field before an external XIOS/IOIPSL serializer."""

    name: str
    value: jnp.ndarray


class HydrolMainOutputPacket(NamedTuple):
    """Pure scientific output packet for ``hydrol_main`` lines 1396-1655."""

    xios_fields: tuple[HydrolScientificField, ...]
    primary_history_enabled: bool
    secondary_history_enabled: bool
    history_serializer_boundary: str
    provenance: tuple[str, ...]


class HydrolThermosoilMoistureInputs(NamedTuple):
    """HYDROL moisture payload passed to ``thermosoil_main``."""

    shumdiag_perma: jnp.ndarray
    mc_layh: jnp.ndarray
    mcl_layh: jnp.ndarray
    tmc_layh: jnp.ndarray
    mc_layh_pft: jnp.ndarray
    mcl_layh_pft: jnp.ndarray
    tmc_layh_pft: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1093-1118",
        "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_soil lines 6729-6741",
        "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_humlev lines 2258-2399",
    )


class HydrolSnowState(NamedTuple):
    """HYDROL explicit-snow state after the snow branch."""

    snow: jnp.ndarray
    snowdz: jnp.ndarray
    snowrho: jnp.ndarray
    snowtemp: jnp.ndarray
    snowheat: jnp.ndarray
    snowgrain: jnp.ndarray
    snow_age: jnp.ndarray
    snow_nobio: jnp.ndarray
    snow_nobio_age: jnp.ndarray
    tot_melt: jnp.ndarray
    provenance: tuple[str, ...]
    notes: tuple[str, ...]


class HydrolExplicitSnowStepResult(NamedTuple):
    """HYDROL-facing result after the explicit snow branch."""

    snow_state: HydrolSnowState
    snowliq: jnp.ndarray
    vevapsno: jnp.ndarray
    subsnownobio: jnp.ndarray
    subsinksoil: jnp.ndarray
    grndflux: jnp.ndarray
    snowmelt: jnp.ndarray
    temp_sol_add: jnp.ndarray
    snowmelt_from_maxmass: jnp.ndarray
    melt_mass_by_layer: jnp.ndarray
    refreeze_mass_by_layer: jnp.ndarray
    liquid_percolation_by_interface: jnp.ndarray
    melt_refreeze_energy: jnp.ndarray
    liquid_excess_energy: jnp.ndarray
    provenance: tuple[str, ...]
    notes: tuple[str, ...]


class HydrolBucketSnowResult(NamedTuple):
    """Scalar bucket-snow result from ``hydrol_snow``."""

    snow: jnp.ndarray
    snow_age: jnp.ndarray
    snow_nobio: jnp.ndarray
    snow_nobio_age: jnp.ndarray
    vevapsno: jnp.ndarray
    tot_melt: jnp.ndarray
    snowmelt: jnp.ndarray
    snowdepth: jnp.ndarray
    subsnownobio: jnp.ndarray
    subsnowveg: jnp.ndarray
    subsinksoil: jnp.ndarray
    icemelt: jnp.ndarray
    provenance: tuple[str, ...]
    notes: tuple[str, ...]


class ExplicitSnowFallResult(NamedTuple):
    """State after ``explicitsnow_fall`` snowfall incorporation."""

    snowrho: jnp.ndarray
    snowdz: jnp.ndarray
    snowheat: jnp.ndarray
    snowgrain: jnp.ndarray
    psnowhmass: jnp.ndarray
    dsnowfall: jnp.ndarray
    rhosnew: jnp.ndarray
    newgrain: jnp.ndarray
    provenance: tuple[str, ...]


class ExplicitSnowLevelsResult(NamedTuple):
    """Three-layer explicit snow geometry from ``explicitsnow_levels``."""

    snowdz: jnp.ndarray
    provenance: tuple[str, ...]


class ExplicitSnowGrainResult(NamedTuple):
    """Snow grain size after ``explicitsnow_grain`` metamorphism."""

    snowgrain: jnp.ndarray
    ztheta: jnp.ndarray
    provenance: tuple[str, ...]


class ExplicitSnowCompactionResult(NamedTuple):
    """Snow density/thickness after ``explicitsnow_compactn`` settling."""

    snowrho: jnp.ndarray
    snowdz: jnp.ndarray
    zsmass: jnp.ndarray
    zsettle: jnp.ndarray
    zviscocity: jnp.ndarray
    provenance: tuple[str, ...]


class ExplicitSnowGoneResult(NamedTuple):
    """State after ``explicitsnow_gone`` complete snowpack removal check."""

    snowtemp: jnp.ndarray
    snowdz: jnp.ndarray
    snowrho: jnp.ndarray
    snowliq: jnp.ndarray
    grndflux: jnp.ndarray
    snowmelt: jnp.ndarray
    totsnowheat: jnp.ndarray
    provenance: tuple[str, ...]


class ExplicitSnowProfileResult(NamedTuple):
    """Snow temperature profile after ``explicitsnow_profile``."""

    snowtemp: jnp.ndarray
    temp_sol_add: jnp.ndarray
    provenance: tuple[str, ...]


class ExplicitSnowTransformResult(NamedTuple):
    """Snow state after ``explicitsnow_transf`` layer redistribution."""

    snowrho: jnp.ndarray
    snowdz: jnp.ndarray
    snowheat: jnp.ndarray
    snowgrain: jnp.ndarray
    provenance: tuple[str, ...]


class ExplicitSnowMeltRefreezeResult(NamedTuple):
    """Snow state after ``explicitsnow_melt_refrz``."""

    snowtemp: jnp.ndarray
    snowdz: jnp.ndarray
    snowrho: jnp.ndarray
    snowliq: jnp.ndarray
    snowmelt: jnp.ndarray
    grndflux: jnp.ndarray
    meltxs: jnp.ndarray
    melt_mass_by_layer: jnp.ndarray
    refreeze_mass_by_layer: jnp.ndarray
    liquid_percolation_by_interface: jnp.ndarray
    provenance: tuple[str, ...]


class ExplicitSnowSublimationResult(NamedTuple):
    """Snow state after vegetated/ice sublimation removal in main."""

    snow: jnp.ndarray
    snowdz: jnp.ndarray
    snowliq: jnp.ndarray
    snowtemp: jnp.ndarray
    vevapsno: jnp.ndarray
    subsnownobio: jnp.ndarray
    subsnowveg: jnp.ndarray
    subsinksoil: jnp.ndarray
    provenance: tuple[str, ...]


class ExplicitSnowMaxMassResult(NamedTuple):
    """Snow depth after ``snowmelt_from_maxmass`` removal in main."""

    snow: jnp.ndarray
    snowdz: jnp.ndarray
    snowmelt_from_maxmass: jnp.ndarray
    provenance: tuple[str, ...]


class ExplicitSnowAgeIceResult(NamedTuple):
    """Land snow age and land-ice snow state after explicit-snow main."""

    snow_age: jnp.ndarray
    snow_nobio: jnp.ndarray
    snow_nobio_age: jnp.ndarray
    snowmelt_ice: jnp.ndarray
    icemelt: jnp.ndarray
    provenance: tuple[str, ...]


class ExplicitSnowLiquidHeatExcessResult(NamedTuple):
    """Snow liquid-water cap and ground-flux feedback after grain update."""

    snowliq: jnp.ndarray
    grndflux: jnp.ndarray
    zliqheatxs: jnp.ndarray
    provenance: tuple[str, ...]


class ExplicitSnowFinalCleanupResult(NamedTuple):
    """Final snow cleanup and total melt synthesis from explicit-snow main."""

    snowrho: jnp.ndarray
    snowgrain: jnp.ndarray
    snowdz: jnp.ndarray
    snowliq: jnp.ndarray
    tot_melt: jnp.ndarray
    provenance: tuple[str, ...]


class ExplicitSnowMainResult(NamedTuple):
    """Full explicit-snow main-step state assembled in Fortran call order."""

    snow: jnp.ndarray
    snowdz: jnp.ndarray
    snowrho: jnp.ndarray
    snowtemp: jnp.ndarray
    snowheat: jnp.ndarray
    snowliq: jnp.ndarray
    snowgrain: jnp.ndarray
    snow_age: jnp.ndarray
    snow_nobio: jnp.ndarray
    snow_nobio_age: jnp.ndarray
    vevapsno: jnp.ndarray
    subsnownobio: jnp.ndarray
    subsinksoil: jnp.ndarray
    grndflux: jnp.ndarray
    snowmelt: jnp.ndarray
    tot_melt: jnp.ndarray
    temp_sol_add: jnp.ndarray
    snowmelt_from_maxmass: jnp.ndarray
    melt_mass_by_layer: jnp.ndarray
    refreeze_mass_by_layer: jnp.ndarray
    liquid_percolation_by_interface: jnp.ndarray
    melt_refreeze_energy: jnp.ndarray
    liquid_excess_energy: jnp.ndarray
    provenance: tuple[str, ...]
    notes: tuple[str, ...]


class HydrolColdStartState(NamedTuple):
    """HYDROL no-restart initialization state from ``hydrol_init``."""

    mc: jnp.ndarray
    mcl: jnp.ndarray
    us: jnp.ndarray
    humrelv: jnp.ndarray
    vegstressv: jnp.ndarray
    humrel: jnp.ndarray
    water2infilt: jnp.ndarray
    ae_ns: jnp.ndarray
    evap_bare_lim_ns: jnp.ndarray
    evap_bare_lim: jnp.ndarray
    zwt_force: jnp.ndarray
    free_drain_coef: jnp.ndarray
    zforce: bool
    snow: jnp.ndarray
    snow_age: jnp.ndarray
    snow_nobio: jnp.ndarray
    snow_nobio_age: jnp.ndarray
    qsintveg: jnp.ndarray
    profil_froz_hydro: jnp.ndarray | None
    profil_froz_hydro_ns: jnp.ndarray | None
    kk: jnp.ndarray | None
    kk_moy: jnp.ndarray | None
    temp_hydro: jnp.ndarray | None
    fwet_out: jnp.ndarray
    run2peat: jnp.ndarray
    wt_ab: jnp.ndarray
    wtp: jnp.ndarray
    fwet_new: jnp.ndarray
    liqwt_ratio: jnp.ndarray
    wt_ab_tide: jnp.ndarray
    run2man: jnp.ndarray
    resdist: jnp.ndarray
    vegtot_old: jnp.ndarray
    vegtot: jnp.ndarray
    mask_veget: jnp.ndarray
    mask_soiltile: jnp.ndarray
    explicit_snow: HydrolSnowState | None
    provenance: tuple[str, ...]
    notes: tuple[str, ...]


class HydrolWaterbalBeginState(NamedTuple):
    """HYDROL ALMA water-balance begin fields initialized before first step."""

    tot_watveg_beg: jnp.ndarray
    tot_watsoil_beg: jnp.ndarray
    snow_beg: jnp.ndarray
    tot_watveg_end: jnp.ndarray
    tot_watsoil_end: jnp.ndarray
    snow_end: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_init lines 3092-3104",
        "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_alma lines 9518-9541",
    )


class HydrolAlmaResult(NamedTuple):
    """State transition from ``hydrol_alma`` lines 9533-9575."""

    tot_watveg_beg: jnp.ndarray
    tot_watsoil_beg: jnp.ndarray
    snow_beg: jnp.ndarray
    tot_watveg_end: jnp.ndarray
    tot_watsoil_end: jnp.ndarray
    snow_end: jnp.ndarray
    delintercept: jnp.ndarray | None
    delsoilmoist: jnp.ndarray | None
    delswe: jnp.ndarray | None
    soilwet: jnp.ndarray | None


class HydrolWaterBalanceResult(NamedTuple):
    """Pure state/diagnostic result of ``hydrol_waterbal`` lines 9407-9475."""

    tot_water_beg: jnp.ndarray
    tot_water_end: jnp.ndarray
    tot_flux: jnp.ndarray
    delta_water: jnp.ndarray
    residual: jnp.ndarray
    fraction_error: jnp.ndarray
    conservation_warning: jnp.ndarray


class HydrolEvapBareLimitDiagnostic(NamedTuple):
    """Bare-soil evaporation beta diagnostic from the alternate residual solve."""

    water_budget_evap: jnp.ndarray
    bare_weighted_evap: jnp.ndarray
    evap_bare_lim: jnp.ndarray
    evap_bare_lim_ns: jnp.ndarray


class HydrolTraceBackedFullStepResult(NamedTuple):
    """Trace-backed HYDROL full-step skeleton result."""

    explicit: HydrolSoilExplicitStepResult
    max_abs_errors: dict[str, float]
    requires_trace: tuple[str, ...]


class HydrolFirstStepPrecallAssembly(NamedTuple):
    """First-step HYDROL pre-call payload assembled from exact local sources."""

    payload: Mapping[str, object]
    missing_inputs: tuple[str, ...]
    enerbil_boundary: EnerbilToHydrolBoundaryValidation
    enerbil_inputs: tuple[str, ...]
    restart_inputs: tuple[str, ...]
    slowproc_inputs: tuple[str, ...]
    driver_inputs: tuple[str, ...]
    source_kernel_inputs: tuple[str, ...]
    provenance: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True only when every audited HYDROL pre-call input is present."""

        return not self.missing_inputs


class HydrolFirstStepModuleClosure(NamedTuple):
    """First-step HYDROL module execution and downstream boundary payloads."""

    module: HydrolModuleExplicitResult | None
    outputs: HydrolModuleOutputs | None
    diagnostics: HydrolModuleDiagnostics | None
    thermosoil_moisture: HydrolThermosoilMoistureInputs | None
    snow_state: HydrolSnowState | None
    snow_step: HydrolExplicitSnowStepResult | None
    module_inputs: Mapping[str, object]
    diagnostic_inputs: Mapping[str, object]
    missing_inputs: tuple[str, ...]
    provenance: tuple[str, ...]
    notes: tuple[str, ...]
    bucket_snow_step: HydrolBucketSnowResult | None = None
    routing_zero: Mapping[str, np.ndarray] | None = None

    @property
    def ok(self) -> bool:
        """True only when HYDROL main, outputs, diagnostics, and moisture exports ran."""

        return not self.missing_inputs and self.module is not None and self.outputs is not None and self.diagnostics is not None


class HydrolAltResidualCheckResult(NamedTuple):
    """Trace-backed alternate residual-boundary check result."""

    resolv_expected: jnp.ndarray
    residual_top_rhs_expected: jnp.ndarray
    residual_top_f_expected: jnp.ndarray
    residual_top_g1_expected: jnp.ndarray
    max_abs_errors: dict[str, float]
    requires_trace: tuple[str, ...]


class HydrolAltResidualSolveResult(NamedTuple):
    """Explicit alternate residual-boundary solve result."""

    saved: HydrolSoilRHSResult
    first: HydrolSoilTridiagResult
    residual_equations: HydrolSoilRHSResult
    second: HydrolSoilTridiagResult
    resolv_alt: jnp.ndarray
    mc_after: jnp.ndarray
    flux_bottom: jnp.ndarray
    tmc: jnp.ndarray


HYDROL_FIRST_STEP_PRECALL_REQUIRED_FIELDS = (
    "precip_rain",
    "precip_snow",
    "temp_sol_new",
    "pgflux",
    "temp_sol_add",
    "vevapsno",
    "vevapwet",
    "vevapflo",
    "transpir",
    "transpot",
    "vevapnu",
    "vevapnu_pft",
    "evapot",
    "evapot_corr",
    "veget",
    "veget_max",
    "qsintmax",
    "qsintveg",
    "tot_melt",
    "snow",
    "snowdz",
    "snowrho",
    "snowtemp",
    "snow_age",
    "snow_nobio",
    "snow_nobio_age",
    "frac_nobio",
    "vegtot",
    "throughfall_by_pft",
    "pref_soil_veg",
    "soiltile",
    "flood_frac",
    "flood_res",
    "subsinksoil",
    "humrel",
    "humrelv",
    "us",
    "ae_ns",
    "evap_bare_lim",
    "evap_bare_lim_ns",
    "evapot",
    "water2infilt",
    "mc",
    "mcl",
    "profil_froz",
    "kfact_root",
    "free_drain_coef",
    "zwt_force",
)

HYDROL_EXPLICIT_SNOW_STEP_REQUIRED_FIELDS = (
    "precip_rain",
    "precip_snow",
    "temp_air",
    "pb",
    "u",
    "v",
    "temp_sol_new",
    "soilcap",
    "pgflux",
    "frac_nobio",
    "totfrac_nobio",
    "gtemp",
    "lambda_snow",
    "cgrnd_snow",
    "dgrnd_snow",
    "vevapsno",
    "snow_age",
    "snow_nobio_age",
    "snow_nobio",
    "snowrho",
    "snowgrain",
    "snowdz",
    "snowtemp",
    "snowheat",
    "snow",
    "temp_sol_add",
    "snowliq",
    "subsnownobio",
    "grndflux",
    "snowmelt",
    "soilflxresid",
)

HYDROL_FREEZE_PROFILE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_main lines 5677-5689",
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_calculate_temp_hydro lines 9597-9662",
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_soil_froz lines 8332-8460",
)
HYDROL_FIRST_STEP_PRECALL_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1013-1072",
    "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_main lines 485-545",
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_main lines 984-1090 and 1182-1284",
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_finalize lines 1727-1747",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_veget lines 2858-2921",
    "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_main lines 470-710",
)
HYDROL_FIRST_STEP_MODULE_CLOSURE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1049-1118",
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_main lines 1206-1296",
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_soil lines 5405-6815",
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_diag_soil lines 9089-9284",
    "fortran_source/ORCHIDEE/src_sechiba/vertical_soil.f90::vertical_soil_init lines 268-363",
)
HYDROL_EXPLICIT_SNOW_ZERO_STATE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_main lines 1178-1190",
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_main lines 207-254",
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_main lines 268-328",
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_main lines 410-484",
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_main lines 494-503",
)
HYDROL_COLD_START_STATE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_init lines 2813-2843",
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_init lines 2865-2994",
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_init lines 2998-3073",
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_init lines 3081-3115",
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_initialize lines 56-88",
)
EXPLICITSNOW_INITIALIZE_ZERO_STATE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_init lines 3111-3115",
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_initialize lines 56-88",
)
EXPLICITSNOW_FALL_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_fall lines 1139-1274",
    "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90::snow3lgrain_0d lines 1055-1091",
    "fortran_source/ORCHIDEE/src_parameters/constantes_var.f90 lines 376, 460, 585-600",
)
EXPLICITSNOW_LEVELS_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_levels lines 1568-1630",
)
SNOW3L_THERMODYNAMIC_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90::snow3lhold_2d lines 593-637",
    "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90::snow3lhold_1d lines 642-686",
    "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90::snow3lhold_0d lines 690-737",
    "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90::snow3lheat_2d lines 741-781",
    "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90::snow3lheat_1d lines 785-826",
    "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90::snow3ltemp_2d lines 891-929",
    "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90::snow3ltemp_1d lines 933-972",
    "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90::snow3lliq_2d lines 1094-1152",
    "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90::snow3lliq_1d lines 1155-1204",
)
EXPLICITSNOW_GRAIN_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_grain lines 609-752",
    "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90::snow3lgrain_1d lines 1012-1052",
)
EXPLICITSNOW_COMPACTION_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_compactn lines 760-869",
    "fortran_source/ORCHIDEE/src_parameters/constantes_var.f90 lines 393, 581, 636-642",
)
EXPLICITSNOW_GONE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_gone lines 1300-1366",
    "fortran_source/ORCHIDEE/src_parameters/constantes_var.f90 lines 376, 392",
)
EXPLICITSNOW_PROFILE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_profile lines 1655-1694",
)
EXPLICITSNOW_TRANSFORM_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_transf lines 891-1132",
)
EXPLICITSNOW_MELT_REFREEZE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_melt_refrz lines 1385-1563",
    "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90::snow3lhold_1d lines 642-686",
)
EXPLICITSNOW_SUBLIMATION_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_main lines 255-323",
)
EXPLICITSNOW_MAXMASS_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_main lines 325-376",
)
EXPLICITSNOW_AGE_ICE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_main lines 408-499",
)
EXPLICITSNOW_LIQUID_HEAT_EXCESS_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_main lines 386-402",
)
EXPLICITSNOW_FINAL_CLEANUP_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_main lines 493-515",
)
EXPLICITSNOW_MAIN_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_main lines 108-553",
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_fall lines 1139-1274",
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_levels lines 1568-1630",
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_transf lines 891-1132",
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_compactn lines 760-869",
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_profile lines 1655-1694",
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_gone lines 1300-1366",
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_melt_refrz lines 1385-1563",
    "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_grain lines 609-752",
)


def _np_array(value):
    import numpy as np

    return np.asarray(value, dtype=np.float64)


def _add_payload_field(
    payload: dict[str, object],
    source_names: list[str],
    name: str,
    value,
    *,
    dtype=None,
) -> None:
    arr = (
        jnp.asarray(value, dtype=dtype)
        if isinstance(value, jax.core.Tracer)
        else np.asarray(value, dtype=dtype)
    )
    payload[name] = arr
    source_names.append(name)


def _slowproc_restart_vegetation_for_hydrol(
    slowproc_restart,
    *,
    pref_soil_veg,
    nstm: int,
    ext_coeff_vegetfrac,
    ok_dgvm=False,
):
    """Return the SLOWPROC vegetation state to pass into HYDROL.

    Fortran provenance: ``slowproc.f90::slowproc_init`` lines 2134-2170 build
    ``veget``/``soiltile`` through ``slowproc_veget`` before missing no-restart
    LAI is reset at lines 2249-2277. ``hydrol.f90::hydrol_main`` lines
    1206-1211 then consumes that established SLOWPROC state in
    ``hydrol_vegupd``. Therefore a caller-supplied ``veget``/``soiltile``
    boundary must not be recomputed later from the reset ``lai`` field.
    """

    from types import SimpleNamespace

    from jax_orchidee.sechiba.slowproc import slowproc_veget_explicit

    has_explicit = (
        getattr(slowproc_restart, "veget", None) is not None
        and getattr(slowproc_restart, "veget_max", None) is not None
        and getattr(slowproc_restart, "soiltile", None) is not None
    )
    if has_explicit:
        veget_max = jnp.asarray(slowproc_restart.veget_max, dtype=jnp.float64)
        if getattr(slowproc_restart, "totfrac_nobio", None) is not None:
            totfrac_nobio = jnp.asarray(slowproc_restart.totfrac_nobio, dtype=jnp.float64)
        elif getattr(slowproc_restart, "vegtot", None) is not None:
            totfrac_nobio = 1.0 - jnp.asarray(slowproc_restart.vegtot, dtype=jnp.float64)
        else:
            totfrac_nobio = jnp.sum(
                jnp.asarray(slowproc_restart.frac_nobio, dtype=jnp.float64),
                axis=1,
            )
        return SimpleNamespace(
            frac_nobio=jnp.asarray(slowproc_restart.frac_nobio, dtype=jnp.float64),
            veget_max=veget_max,
            veget=jnp.asarray(slowproc_restart.veget, dtype=jnp.float64),
            totfrac_nobio=totfrac_nobio,
            soiltile=jnp.asarray(slowproc_restart.soiltile, dtype=jnp.float64),
            used_explicit_restart_boundary=True,
        )

    veget = slowproc_veget_explicit(
        lai=slowproc_restart.lai,
        frac_nobio=slowproc_restart.frac_nobio,
        veget_max=slowproc_restart.veget_max,
        pref_soil_veg=pref_soil_veg,
        ext_coeff_vegetfrac=ext_coeff_vegetfrac,
        nstm=nstm,
        ok_dgvm=ok_dgvm,
    )
    return SimpleNamespace(
        frac_nobio=veget.frac_nobio,
        veget_max=veget.veget_max,
        veget=veget.veget,
        totfrac_nobio=veget.totfrac_nobio,
        soiltile=veget.soiltile,
        used_explicit_restart_boundary=False,
    )


def hydrol_static_precall_template_from_slowproc(
    *,
    slowproc_restart,
    pref_soil_veg,
    nstm: int,
    ext_coeff_vegetfrac,
    hydrol_njsc,
    refSOC_1d=None,
    use_refSOC_hydrol=False,
    throughfall_by_pft=None,
    humcste=None,
    zz_mm=None,
    peat_hydro=False,
    ok_dgvm=False,
    dt_days=None,
) -> dict[str, object]:
    """Build the slow/static HYDROL pre-call fields for a driver day loop.

    Fortran provenance is the same as ``assemble_hydrol_first_step_precall_payload``:
    ``slowproc_veget`` prepares ``veget``/``soiltile`` and ``hydrol_vegupd``
    prepares per-soil-tile rooting controls before ``hydrol_main`` consumes
    them. This helper is only an orchestration cache; dynamic HYDROL state and
    same-step ENERBIL/DIFFUCO fields are still supplied by each timestep.
    """

    payload: dict[str, object] = {}
    source_kernel_inputs: list[str] = []
    slowproc_inputs: list[str] = []
    restart_inputs: list[str] = []
    veget = _slowproc_restart_vegetation_for_hydrol(
        slowproc_restart,
        pref_soil_veg=pref_soil_veg,
        nstm=nstm,
        ext_coeff_vegetfrac=ext_coeff_vegetfrac,
        ok_dgvm=ok_dgvm,
    )
    _add_payload_field(payload, source_kernel_inputs, "veget", veget.veget)
    _add_payload_field(payload, source_kernel_inputs, "veget_max", veget.veget_max)
    _add_payload_field(payload, slowproc_inputs, "frac_nobio", slowproc_restart.frac_nobio)
    _add_payload_field(payload, source_kernel_inputs, "vegtot", 1.0 - veget.totfrac_nobio)
    _add_payload_field(payload, source_kernel_inputs, "soiltile", veget.soiltile)
    _add_payload_field(payload, slowproc_inputs, "pref_soil_veg", pref_soil_veg, dtype="int32")
    if humcste is not None and zz_mm is not None and hydrol_njsc is not None:
        pref_soil_veg_np = np.asarray(pref_soil_veg, dtype=np.int32)
        pref_indices = tuple(int(value) - 1 for value in pref_soil_veg_np.tolist())
        _, _, _, _, _, kfact_root = _hydrol_vegupd_kfact_root_jit(
            veget=veget.veget,
            veget_max=veget.veget_max,
            soiltile=veget.soiltile,
            vegtot=1.0 - veget.totfrac_nobio,
            humcste=humcste,
            zz_mm=zz_mm,
            njsc=hydrol_njsc,
            pref_indices=pref_indices,
            peat_hydro=bool(peat_hydro),
        )
        _add_payload_field(
            payload,
            source_kernel_inputs,
            "kfact_root",
            kfact_root,
        )
    if throughfall_by_pft is not None:
        _add_payload_field(payload, source_kernel_inputs, "throughfall_by_pft", _np_array(throughfall_by_pft) / 100.0)
    if zz_mm is not None:
        _add_payload_field(payload, source_kernel_inputs, "dz_mm", hydrol_layer_thickness_from_zz_mm(zz_mm))
        _add_payload_field(payload, source_kernel_inputs, "zz_mm", zz_mm)
    if dt_days is not None:
        _add_payload_field(payload, source_kernel_inputs, "dt_days", dt_days)
    if hydrol_njsc is not None:
        _add_payload_field(payload, restart_inputs, "njsc", hydrol_njsc, dtype="int32")
        tex = jnp.asarray(hydrol_njsc, dtype=jnp.int32) - 1
        _add_payload_field(
            payload,
            source_kernel_inputs,
            "mcr",
            jnp.asarray(USDA_MCR, dtype=jnp.float64)[tex],
        )
        if bool(use_refSOC_hydrol):
            thresholds = hydrol_refsoc_1d_thresholds(
                njsc=hydrol_njsc,
                refSOC_1d=refSOC_1d,
                use_refSOC_hydrol=True,
            )
            _add_payload_field(payload, source_kernel_inputs, "mcs", thresholds.mcs)
            _add_payload_field(payload, source_kernel_inputs, "mcw", thresholds.mcw)
            _add_payload_field(payload, source_kernel_inputs, "mcf", thresholds.mcf)
            _add_payload_field(payload, source_kernel_inputs, "refSOC_1d", refSOC_1d)
        else:
            _add_payload_field(payload, source_kernel_inputs, "mcs", jnp.asarray(USDA_MCS, dtype=jnp.float64)[tex])
            _add_payload_field(payload, source_kernel_inputs, "mcw", jnp.asarray(USDA_MCW, dtype=jnp.float64)[tex])
            _add_payload_field(payload, source_kernel_inputs, "mcf", jnp.asarray(USDA_MCF, dtype=jnp.float64)[tex])
    payload["_source_kernel_inputs"] = tuple(dict.fromkeys(source_kernel_inputs))
    payload["_slowproc_inputs"] = tuple(dict.fromkeys(slowproc_inputs))
    payload["_restart_inputs"] = tuple(dict.fromkeys(restart_inputs))
    return payload


def _transpose_restart_soil_layers(name: str, value):
    arr = (
        jnp.asarray(value, dtype=jnp.float64)
        if isinstance(value, jax.core.Tracer)
        else np.asarray(value, dtype=np.float64)
    )
    if arr.ndim != 3:
        raise ValueError(f"{name} restart field must have shape (npts,nstm,nslm)")
    return arr.transpose((0, 2, 1))


def hydrol_refsoc_1d_thresholds(
    *,
    njsc,
    refSOC_1d=None,
    use_refSOC_hydrol=True,
    nscm: int = 12,
    soilc_max: float = SOILC_MAX,
    poros_org: float = POROS_ORG,
) -> HydrolRefSocThresholds:
    """Apply HYDROL's ``use_refSOC_hydrol`` threshold update.

    Fortran provenance:
    ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_initialize``
    lines 625-636 reads restart/static ``refSOC_1d`` when
    ``use_refSOC_hydrol`` is true, and ``hydrol_var_init`` lines 4071-4088
    replaces ``mcs``, ``mcw``, and ``mcf`` from the USDA mineral values using
    ``refSOC_1d / soilc_max`` and the Van Genuchten equations. The source
    aborts unless ``nscm == 12``.
    """

    njsc_arr = _as_1d("njsc", njsc).astype(jnp.int32)
    npts = int(njsc_arr.shape[0])
    if bool(jnp.any(njsc_arr < 1)) or bool(jnp.any(njsc_arr > 12)):
        raise ValueError("HYDROL refSOC thresholds require USDA njsc in 1..12")
    tex = njsc_arr - 1
    mineral_mcs = jnp.asarray(USDA_MCS, dtype=jnp.float64)[tex]
    mineral_mcw = jnp.asarray(USDA_MCW, dtype=jnp.float64)[tex]
    mineral_mcf = jnp.asarray(USDA_MCF, dtype=jnp.float64)[tex]
    if not bool(use_refSOC_hydrol):
        return HydrolRefSocThresholds(
            mcs=mineral_mcs,
            mcw=mineral_mcw,
            mcf=mineral_mcf,
            zx1=jnp.zeros((npts,), dtype=jnp.float64),
            provenance=(
                "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_var_init lines 4071-4073",
            ),
        )
    if int(nscm) != 12:
        raise ValueError("use_refSOC_hydrol=true only applies to USDA nscm=12")
    if refSOC_1d is None:
        raise ValueError("use_refSOC_hydrol=True requires explicit refSOC_1d")
    refsoc = _as_1d("refSOC_1d", refSOC_1d)
    if refsoc.shape[0] != npts:
        raise ValueError("refSOC_1d must have shape (npts,)")

    mcr = jnp.asarray(USDA_MCR, dtype=jnp.float64)[tex]
    alpha = jnp.asarray(USDA_AVAN, dtype=jnp.float64)[tex]
    vg_n = jnp.asarray(USDA_NVAN, dtype=jnp.float64)[tex]
    vg_m = jnp.asarray(USDA_VG_M, dtype=jnp.float64)[tex]
    psi_wp = jnp.asarray(USDA_VG_PSI_WP, dtype=jnp.float64)[tex]
    psi_fc = jnp.asarray(USDA_VG_PSI_FC, dtype=jnp.float64)[tex]
    zx1 = jnp.minimum(refsoc / jnp.asarray(soilc_max, dtype=jnp.float64), 1.0)
    mcs = zx1 * jnp.asarray(poros_org, dtype=jnp.float64) + (1.0 - zx1) * mineral_mcs
    mcw = mcr + (mcs - mcr) / (1.0 + (alpha * psi_wp) ** vg_n) ** vg_m
    mcf = mcr + (mcs - mcr) / (1.0 + (alpha * psi_fc) ** vg_n) ** vg_m
    return HydrolRefSocThresholds(
        mcs=mcs,
        mcw=mcw,
        mcf=mcf,
        zx1=zx1,
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_initialize lines 625-636",
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_var_init lines 4071-4088",
        ),
    )


def hydrol_layer_thickness_from_zz_mm(zz_mm):
    """Build HYDROL ``dh`` layer thicknesses from node depths.

    Fortran provenance: ``hydrol.f90::hydrol_soil_setup`` line 4069 assigns
    ``dh(jsl)=dlh(jsl)*mille`` after the vertical grid has built the CWRR soil
    layer thicknesses. For the restart-aligned CWRR grid used here, adjacent
    ``zz`` node spacings are the exact ``dh`` weights consumed by
    ``hydrol_soil_froz`` lines 8440-8449.
    """

    zz_mm = jnp.asarray(zz_mm, dtype=jnp.float64)
    if zz_mm.ndim != 1 or zz_mm.shape[0] < 2:
        raise ValueError("zz_mm must be a one-dimensional HYDROL depth vector with at least two layers")
    first = zz_mm[0]
    return jnp.concatenate((first[None], zz_mm[1:] - zz_mm[:-1]))


def hydrol_calculate_temp_hydro_profile(
    *,
    stempdiag,
    zz_mm,
    diaglev_m,
    snow=None,
    snowdz=None,
    ok_explicitsnow=True,
    snow_density=330.0,
    min_sechiba=1.0e-8,
) -> jnp.ndarray:
    """Interpolate THERMOSOIL diagnostic temperatures onto HYDROL levels.

    Fortran provenance: ``hydrol.f90::hydrol_calculate_temp_hydro`` lines
    9597-9662. ``snowdz`` is accepted to preserve the source signature; the
    active source branch only uses total snow depth when explicit snow is
    disabled.
    """

    del snowdz
    stempdiag = jnp.asarray(stempdiag, dtype=jnp.float64)
    if stempdiag.ndim != 2:
        raise ValueError("stempdiag must have shape (npts,nslm)")
    zz_mm = jnp.asarray(zz_mm, dtype=jnp.float64)
    diaglev_m = jnp.asarray(diaglev_m, dtype=jnp.float64)
    if zz_mm.ndim != 1 or diaglev_m.ndim != 1:
        raise ValueError("zz_mm and diaglev_m must be one-dimensional")
    if zz_mm.shape[0] != stempdiag.shape[1] or diaglev_m.shape[0] != stempdiag.shape[1]:
        raise ValueError("zz_mm, diaglev_m, and stempdiag layer axes must match")
    npts, nslm = stempdiag.shape
    if snow is None:
        snow = jnp.zeros((npts,), dtype=stempdiag.dtype)
    else:
        snow = _as_1d("snow", snow)
        _require_same_npts(stempdiag, snow)

    zz_m = zz_mm / 1000.0
    lev_diag_base = jnp.concatenate(
        (
            zz_m[1:2] / 2.0,
            zz_m[1:-1] + (zz_m[2:] - zz_m[1:-1]) / 2.0,
            zz_m[-1:],
        )
    )
    snow_h = jnp.zeros((npts,), dtype=stempdiag.dtype)
    if not bool(ok_explicitsnow):
        snow_h = snow / jnp.asarray(snow_density, dtype=stempdiag.dtype)
    lev_diag_raw = lev_diag_base[None, :] + snow_h[:, None]
    prev_prog = jnp.concatenate((jnp.zeros((1,), dtype=diaglev_m.dtype), diaglev_m[:-1]))
    last_prog = diaglev_m[-1]

    def _scan_layer(prev_diag, raw_lev_diag):
        crosses_bottom = (raw_lev_diag > last_prog) & (prev_diag < last_prog - min_sechiba)
        lev_diag = jnp.where(crosses_bottom, last_prog, raw_lev_diag)
        denom = lev_diag - prev_diag
        overlap = jnp.maximum(
            jnp.minimum(lev_diag[:, None], diaglev_m[None, :])
            - jnp.maximum(prev_diag[:, None], prev_prog[None, :]),
            0.0,
        )
        weights = overlap / denom[:, None]
        bottom_tail = (lev_diag > last_prog) & (prev_diag >= last_prog - min_sechiba)
        bottom_weights = jnp.zeros_like(weights).at[:, -1].set(1.0)
        weights = jnp.where(bottom_tail[:, None], bottom_weights, weights)
        return lev_diag, weights

    _, weights_by_layer = jax.lax.scan(_scan_layer, snow_h, lev_diag_raw.T)
    weights = jnp.transpose(weights_by_layer, (1, 0, 2))
    return jnp.einsum("nhl,nl->nh", weights, stempdiag)


def hydrol_soil_froz_profile(
    *,
    temp_hydro,
    mc,
    njsc,
    dh_mm,
    peat_hydro=False,
    peat_tiles=(4, 5, 6),
    ok_thermodynamical_freezing=True,
    fr_dt=2.0,
    froz_frac_corr=1.0,
    smtot_corr=2.0,
    max_froz_hydro=1.0,
    zero_celsius=273.15,
    lhf=0.3336e6,
    min_sechiba=1.0e-8,
    nvan=USDA_NVAN,
    avan=USDA_AVAN,
    mcr=USDA_MCR,
    mcs=USDA_MCS,
    nvan_peat=PEAT_NVAN,
    avan_peat=PEAT_AVAN,
    mcr_peat=PEAT_MCR,
    mcs_peat=PEAT_MCS,
) -> jnp.ndarray:
    """Compute ``profil_froz_hydro_ns`` for all HYDROL soil tiles.

    Fortran provenance: ``hydrol.f90::hydrol_soil_froz`` lines 8332-8460,
    called from ``hydrol_main`` lines 5677-5689 after
    ``hydrol_calculate_temp_hydro`` has populated ``temp_hydro``.
    """

    temp_hydro = jnp.asarray(temp_hydro, dtype=jnp.float64)
    mc = jnp.asarray(mc, dtype=jnp.float64)
    if temp_hydro.ndim != 2:
        raise ValueError("temp_hydro must have shape (npts,nslm)")
    if mc.ndim != 3:
        raise ValueError("mc must have shape (npts,nslm,nstm)")
    npts, nslm, nstm = mc.shape
    if temp_hydro.shape != (npts, nslm):
        raise ValueError("temp_hydro must match mc point/layer axes")
    njsc = _as_1d("njsc", njsc).astype(jnp.int32)
    _require_same_npts(mc, njsc)
    dh_mm = _as_1d("dh_mm", dh_mm)
    if dh_mm.shape[0] != nslm:
        raise ValueError("dh_mm length must match nslm")

    tex = njsc - 1
    nvan_point = jnp.asarray(nvan, dtype=jnp.float64)[tex]
    avan_point = jnp.asarray(avan, dtype=jnp.float64)[tex]
    mcr_point = jnp.asarray(mcr, dtype=jnp.float64)[tex]
    mcs_point = jnp.asarray(mcs, dtype=jnp.float64)[tex]
    lower = zero_celsius - fr_dt / 2.0
    upper = zero_celsius + fr_dt / 2.0

    profil = jnp.zeros_like(mc)
    for tile_offset in range(nstm):
        peat_active = bool(peat_hydro) and (tile_offset + 1) in tuple(int(tile) for tile in peat_tiles)
        if peat_active:
            nvan_use = jnp.full((npts,), nvan_peat, dtype=mc.dtype)
            avan_use = jnp.full((npts,), avan_peat, dtype=mc.dtype)
            mcr_use = jnp.full((npts,), mcr_peat, dtype=mc.dtype)
            mcs_use = jnp.full((npts,), mcs_peat, dtype=mc.dtype)
        else:
            nvan_use = nvan_point
            avan_use = avan_point
            mcr_use = mcr_point
            mcs_use = mcs_point
        m_vg = 1.0 - 1.0 / nvan_use
        mc_tile = mc[:, :, tile_offset]
        linear_x = jnp.clip((temp_hydro - lower) / fr_dt, 0.0, 1.0)
        thermo_base = (
            2.2
            * 1000.0
            * avan_use[:, None]
            * (upper - temp_hydro)
            * lhf
            / zero_celsius
            / 10.0
        )
        use_linear = (not bool(ok_thermodynamical_freezing)) | (
            mc_tile < (mcr_use[:, None] + min_sechiba)
        )
        thermo_active = (
            (~use_linear)
            & (temp_hydro >= lower)
            & (temp_hydro < upper)
        )
        safe_thermo_base = jnp.where(thermo_active, thermo_base, 1.0)
        safe_moisture = jnp.where(
            thermo_active,
            mc_tile - mcr_use[:, None],
            1.0,
        )
        thermo_liq = (
            (mcs_use - mcr_use)[:, None]
            * (safe_thermo_base ** nvan_use[:, None] + 1.0) ** (-m_vg[:, None])
        ) / safe_moisture
        thermo_x = jnp.where(temp_hydro >= upper, 1.0, jnp.where(temp_hydro >= lower, jnp.minimum(thermo_liq, 1.0), 0.0))
        x = jnp.where(use_linear, linear_x, thermo_x)
        profil_tile = 1.0 - x

        mc_ns = mc_tile / mcs_use[:, None]
        froz_frac_moy = jnp.sum(dh_mm[None, :] * profil_tile, axis=1) / jnp.sum(dh_mm)
        smtot_moy = jnp.sum(dh_mm[None, :-1] * mc_ns[:, :-1], axis=1) / jnp.sum(dh_mm[:-1])
        corrected = jnp.minimum(
            profil_tile * (froz_frac_moy[:, None] ** froz_frac_corr) * (smtot_moy[:, None] ** smtot_corr),
            max_froz_hydro,
        )
        profil = profil.at[:, :, tile_offset].set(corrected)
    return profil


def hydrol_first_step_profil_froz_from_arrays(
    *,
    ptn,
    veget_max,
    mc,
    njsc,
    zz_mm,
    diaglev_m=None,
    snow=None,
    snowdz=None,
    ok_explicitsnow=True,
    peat_hydro=False,
    ok_thermodynamical_freezing=True,
    fr_dt=2.0,
    froz_frac_corr=1.0,
    smtot_corr=2.0,
    max_froz_hydro=1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Derive first-step HYDROL freezing state from THERMOSOIL arrays."""

    zz_mm = jnp.asarray(zz_mm, dtype=jnp.float64)
    diaglev_m = zz_mm / 1000.0 if diaglev_m is None else jnp.asarray(diaglev_m, dtype=jnp.float64)
    stempdiag = thermosoil_diaglev_from_ptn(ptn=ptn, veget_max=veget_max, nslm=zz_mm.shape[0])
    temp_hydro = hydrol_calculate_temp_hydro_profile(
        stempdiag=stempdiag,
        zz_mm=zz_mm,
        diaglev_m=diaglev_m,
        snow=snow,
        snowdz=snowdz,
        ok_explicitsnow=ok_explicitsnow,
    )
    profil = hydrol_soil_froz_profile(
        temp_hydro=temp_hydro,
        mc=mc,
        njsc=njsc,
        dh_mm=hydrol_layer_thickness_from_zz_mm(zz_mm),
        peat_hydro=peat_hydro,
        ok_thermodynamical_freezing=ok_thermodynamical_freezing,
        fr_dt=fr_dt,
        froz_frac_corr=froz_frac_corr,
        smtot_corr=smtot_corr,
        max_froz_hydro=max_froz_hydro,
    )
    return profil, stempdiag, temp_hydro


HYDROL_FIRST_STEP_PROFIL_FROZ_JIT_STATIC_ARGNAMES = (
    "ok_explicitsnow",
    "peat_hydro",
    "ok_thermodynamical_freezing",
)


_hydrol_first_step_profil_froz_from_arrays_jit = jit(
    hydrol_first_step_profil_froz_from_arrays,
    static_argnames=HYDROL_FIRST_STEP_PROFIL_FROZ_JIT_STATIC_ARGNAMES,
)


def hydrol_first_step_profil_froz_from_restart_thermosoil(
    *,
    thermosoil_restart,
    veget_max,
    mc,
    njsc,
    zz_mm,
    diaglev_m=None,
    snow=None,
    snowdz=None,
    ok_explicitsnow=True,
    peat_hydro=False,
    ok_thermodynamical_freezing=True,
    fr_dt=2.0,
    froz_frac_corr=1.0,
    smtot_corr=2.0,
    max_froz_hydro=1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Derive first-step HYDROL freezing state from THERMOSOIL restart state."""

    return hydrol_first_step_profil_froz_from_arrays(
        ptn=thermosoil_restart.ptn,
        veget_max=veget_max,
        mc=mc,
        njsc=njsc,
        zz_mm=zz_mm,
        diaglev_m=diaglev_m,
        snow=snow,
        snowdz=snowdz,
        ok_explicitsnow=ok_explicitsnow,
        peat_hydro=peat_hydro,
        ok_thermodynamical_freezing=ok_thermodynamical_freezing,
        fr_dt=fr_dt,
        froz_frac_corr=froz_frac_corr,
        smtot_corr=smtot_corr,
        max_froz_hydro=max_froz_hydro,
    )


def hydrol_nroot_from_humcste(
    *,
    humcste,
    dz_mm,
    zz_mm,
    altmax=None,
    ok_pc=False,
    ok_leak=False,
    min_sechiba=1.0e-8,
) -> jnp.ndarray:
    """Compute HYDROL normalized root fractions from ``humcste``.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    ``hydrol_var_init`` lines 4097-4143. The active ALT/permafrost crop uses
    ``altmax`` only when ``ok_pc`` or ``ok_leak`` is true; missing ``altmax`` is
    therefore a strict input error on those branches.
    """

    humcste = _as_1d("humcste", humcste)
    dz_mm = _as_1d("dz_mm", dz_mm)
    zz_mm = _as_1d("zz_mm", zz_mm)
    if dz_mm.shape != zz_mm.shape:
        raise ValueError("dz_mm and zz_mm must have the same HYDROL layer axis")
    nvm = int(humcste.shape[0])
    nslm = int(zz_mm.shape[0])
    if nslm < 2:
        raise ValueError("HYDROL nroot requires at least two layers")
    if bool(ok_pc) or bool(ok_leak):
        if altmax is None:
            raise ValueError("altmax is required for HYDROL nroot when ok_pc or ok_leak is true")
        altmax = _as_2d("altmax", altmax)
        npts = int(altmax.shape[0])
        if altmax.shape[1] != nvm:
            raise ValueError("altmax must have shape (npts,nvm)")
    else:
        npts = 1
        altmax = jnp.zeros((npts, nvm), dtype=jnp.float64)

    nroot = jnp.zeros((npts, nvm, nslm), dtype=jnp.float64)
    denom = jnp.exp(-humcste * dz_mm[1] / 1000.0 / 2.0) - jnp.exp(-humcste * zz_mm[-1] / 1000.0)
    for jv in range(nvm):
        for jsl in range(1, nslm - 1):
            value = (
                jnp.exp(-humcste[jv] * zz_mm[jsl] / 1000.0)
                * (
                    jnp.exp(humcste[jv] * dz_mm[jsl] / 1000.0 / 2.0)
                    - jnp.exp(-humcste[jv] * dz_mm[jsl + 1] / 1000.0 / 2.0)
                )
                / denom[jv]
            )
            nroot = nroot.at[:, jv, jsl].set(value)
        last = nslm - 1
        last_value = (
            (jnp.exp(humcste[jv] * dz_mm[last] / 1000.0 / 2.0) - 1.0)
            * jnp.exp(-humcste[jv] * zz_mm[last] / 1000.0)
            / denom[jv]
        )
        nroot = nroot.at[:, jv, last].set(last_value)

    if bool(ok_pc) or bool(ok_leak):
        znh_m = zz_mm / 1000.0
        active = (altmax[:, :, None] > 0.0) & (znh_m[None, None, :] >= altmax[:, :, None])
        nroot = jnp.where(active, 0.0, nroot)
        normalizer = jnp.sum(nroot, axis=2)
        positive_normalizer = normalizer > 0.0
        safe_normalizer = jnp.where(positive_normalizer, normalizer, 1.0)
        nroot = jnp.where(
            positive_normalizer[:, :, None],
            nroot / safe_normalizer[:, :, None],
            nroot,
        )
    return nroot


@lru_cache(maxsize=32)
def _cached_hydrol_nroot_from_humcste(
    humcste_key,
    dz_mm_key,
    zz_mm_key,
    altmax_key: tuple[tuple[float, ...], ...] | None,
    ok_pc: bool,
    ok_leak: bool,
) -> jnp.ndarray:
    _, humcste_shape, humcste_values = humcste_key
    _, dz_mm_shape, dz_mm_values = dz_mm_key
    _, zz_mm_shape, zz_mm_values = zz_mm_key
    altmax = None if altmax_key is None else jnp.asarray(altmax_key, dtype=jnp.float64)
    return hydrol_nroot_from_humcste(
        humcste=jnp.asarray(humcste_values, dtype=jnp.float64).reshape(humcste_shape),
        dz_mm=jnp.asarray(dz_mm_values, dtype=jnp.float64).reshape(dz_mm_shape),
        zz_mm=jnp.asarray(zz_mm_values, dtype=jnp.float64).reshape(zz_mm_shape),
        altmax=altmax,
        ok_pc=ok_pc,
        ok_leak=ok_leak,
    )


def _hydrol_initial_tmc_from_restart_mc_kernel(mc, water2infilt, dz_mm) -> jnp.ndarray:
    first = dz_mm[1] * (3.0 * mc[:, 0, :] + mc[:, 1, :]) / 8.0
    middle = (
        dz_mm[1:-1][None, :, None] * (3.0 * mc[:, 1:-1, :] + mc[:, :-2, :]) / 8.0
        + dz_mm[2:][None, :, None] * (3.0 * mc[:, 1:-1, :] + mc[:, 2:, :]) / 8.0
    )
    last = dz_mm[-1] * (3.0 * mc[:, -1, :] + mc[:, -2, :]) / 8.0
    return first + jnp.sum(middle, axis=1) + last + water2infilt


_hydrol_initial_tmc_from_restart_mc_jit = jit(_hydrol_initial_tmc_from_restart_mc_kernel)


def hydrol_initial_tmc_from_restart_mc(*, mc, water2infilt, dz_mm) -> jnp.ndarray:
    """Compute initialized ``tmc`` from restart ``mc`` and ``water2infilt``.

    Fortran provenance: ``hydrol.f90::hydrol_var_init`` lines 4363-4405:
    ``tmc`` is rebuilt from node moisture using ``dz`` and then incremented by
    restart ``water2infilt`` before the first ``hydrol_main`` call.
    """

    mc = _as_3d("mc", mc)
    water2infilt = _as_2d("water2infilt", water2infilt)
    dz_mm = _as_1d("dz_mm", dz_mm)
    if water2infilt.shape != (mc.shape[0], mc.shape[2]) or dz_mm.shape[0] != mc.shape[1]:
        raise ValueError("mc, water2infilt, and dz_mm dimensions must match")
    if mc.shape[1] < 2:
        raise ValueError("HYDROL tmc integration requires at least two layers")

    return _hydrol_initial_tmc_from_restart_mc_jit(mc, water2infilt, dz_mm)


def hydrol_cold_start_waterbal_begin_state(
    state: HydrolColdStartState,
    *,
    veget_max,
    soiltile,
    dz_mm,
) -> HydrolWaterbalBeginState:
    """Initialize HYDROL ALMA begin fields for no-restart runs.

    ``hydrol_init`` reads these fields from restart; when absent,
    ``hydrol_alma(lstep_init=.TRUE.)`` sets intercepted water to
    ``SUM(qsintveg)``, soil water to the current ``humtot``, and snow water to
    ``snow + SUM(snow_nobio)``. This helper builds the same cold-start
    pre-first-step state from already-initialized CWRR tile moisture.
    """

    veget_max = jnp.asarray(veget_max, dtype=jnp.float64)
    soiltile = jnp.asarray(soiltile, dtype=jnp.float64)
    dz_mm = jnp.asarray(dz_mm, dtype=jnp.float64)
    if veget_max.shape != state.qsintveg.shape:
        raise ValueError("veget_max must match state.qsintveg shape")
    if soiltile.shape != state.resdist.shape:
        raise ValueError("soiltile must match HYDROL tile shape")
    if dz_mm.ndim != 1 or dz_mm.shape[0] != state.mc.shape[1]:
        raise ValueError("dz_mm must have one value per HYDROL soil layer")

    watveg = jnp.sum(state.qsintveg, axis=1)
    tile_water = hydrol_initial_tmc_from_restart_mc(
        mc=state.mc,
        water2infilt=state.water2infilt,
        dz_mm=dz_mm,
    )
    humtot = state.vegtot[:, None] * soiltile * tile_water
    humtot = jnp.sum(humtot, axis=1)
    snow_beg = state.snow + jnp.sum(state.snow_nobio, axis=1)
    return HydrolWaterbalBeginState(
        tot_watveg_beg=watveg,
        tot_watsoil_beg=humtot,
        snow_beg=snow_beg,
        tot_watveg_end=watveg,
        tot_watsoil_end=humtot,
        snow_end=snow_beg,
    )


def hydrol_waterbal_step(
    *,
    tot_water_beg,
    vegtot,
    totfrac_nobio,
    qsintveg,
    humtot,
    snow,
    snow_nobio,
    precip_rain,
    precip_snow,
    returnflow,
    reinfiltration,
    irrigation,
    vevapwet,
    transpir,
    vevapnu,
    vevapsno,
    vevapflo,
    floodout,
    runoff,
    drainage,
    allowed_err=2.0e-8,
    raise_fraction_error: bool = True,
) -> HydrolWaterBalanceResult:
    """Evaluate ``hydrol_waterbal`` lines 9407-9475 in source order.

    The source stops only for an invalid vegetation/non-biological fraction.
    A water-conservation residual above ``2*allowed_err`` is diagnostic because
    the assignment that would set ``error`` is commented out at lines 9467-9468.
    """

    tot_water_beg = _as_1d("tot_water_beg", tot_water_beg)
    vegtot = _as_1d("vegtot", vegtot)
    totfrac_nobio = _as_1d("totfrac_nobio", totfrac_nobio)
    qsintveg = _as_2d("qsintveg", qsintveg)
    humtot = _as_1d("humtot", humtot)
    snow = _as_1d("snow", snow)
    snow_nobio = _as_2d("snow_nobio", snow_nobio)
    vevapwet = _as_2d("vevapwet", vevapwet)
    transpir = _as_2d("transpir", transpir)
    vectors = {
        name: _as_1d(name, value)
        for name, value in {
            "precip_rain": precip_rain,
            "precip_snow": precip_snow,
            "returnflow": returnflow,
            "reinfiltration": reinfiltration,
            "irrigation": irrigation,
            "vevapnu": vevapnu,
            "vevapsno": vevapsno,
            "vevapflo": vevapflo,
            "floodout": floodout,
            "runoff": runoff,
            "drainage": drainage,
        }.items()
    }
    _require_same_npts(
        tot_water_beg,
        vegtot,
        totfrac_nobio,
        qsintveg,
        humtot,
        snow,
        snow_nobio,
        vevapwet,
        transpir,
        *vectors.values(),
    )
    if qsintveg.shape != vevapwet.shape or qsintveg.shape != transpir.shape:
        raise ValueError("qsintveg, vevapwet, and transpir must share (npts,nvm) shape")

    fraction_error = jnp.abs(1.0 - (totfrac_nobio + vegtot)) > allowed_err
    if bool(raise_fraction_error) and bool(np.any(np.asarray(fraction_error))):
        points = tuple(int(index) for index in np.flatnonzero(np.asarray(fraction_error)))
        raise ValueError(f"hydrol_waterbal vegetation/non-biological fractions fail at zero-based points {points}")

    watveg = jnp.sum(qsintveg, axis=1)
    tot_water_end = humtot + watveg + snow + jnp.sum(snow_nobio, axis=1)
    tot_flux = (
        vectors["precip_rain"]
        + vectors["precip_snow"]
        + vectors["irrigation"]
        - jnp.sum(vevapwet, axis=1)
        - jnp.sum(transpir, axis=1)
        - vectors["vevapnu"]
        - vectors["vevapsno"]
        - vectors["vevapflo"]
        + vectors["floodout"]
        - vectors["runoff"]
        - vectors["drainage"]
        + vectors["returnflow"]
        + vectors["reinfiltration"]
    )
    delta_water = tot_water_end - tot_water_beg
    residual = delta_water - tot_flux
    conservation_warning = jnp.abs(residual) > 2.0 * allowed_err
    return HydrolWaterBalanceResult(
        tot_water_end,
        tot_water_end,
        tot_flux,
        delta_water,
        residual,
        fraction_error,
        conservation_warning,
    )


def hydrol_alma_step(
    *,
    qsintveg,
    humtot,
    snow,
    snow_nobio,
    tot_watveg_beg,
    tot_watsoil_beg,
    snow_beg,
    mx_eau_var,
    lstep_init: bool,
) -> HydrolAlmaResult:
    """Apply the exact ALMA inventory transition from lines 9533-9575.

    On ``lstep_init`` Fortran returns before assigning deltas or ``soilwet``;
    those outputs remain ``None`` rather than receiving fabricated zeros.
    """

    qsintveg = _as_2d("qsintveg", qsintveg)
    humtot = _as_1d("humtot", humtot)
    snow = _as_1d("snow", snow)
    snow_nobio = _as_2d("snow_nobio", snow_nobio)
    tot_watveg_beg = _as_1d("tot_watveg_beg", tot_watveg_beg)
    tot_watsoil_beg = _as_1d("tot_watsoil_beg", tot_watsoil_beg)
    snow_beg = _as_1d("snow_beg", snow_beg)
    mx_eau_var = _as_1d("mx_eau_var", mx_eau_var)
    _require_same_npts(
        qsintveg,
        humtot,
        snow,
        snow_nobio,
        tot_watveg_beg,
        tot_watsoil_beg,
        snow_beg,
        mx_eau_var,
    )

    tot_watveg_end = jnp.sum(qsintveg, axis=1)
    tot_watsoil_end = humtot
    snow_end = snow + jnp.sum(snow_nobio, axis=1)
    if bool(lstep_init):
        return HydrolAlmaResult(
            tot_watveg_end,
            tot_watsoil_end,
            snow_end,
            tot_watveg_end,
            tot_watsoil_end,
            snow_end,
            None,
            None,
            None,
            None,
        )

    delintercept = tot_watveg_end - tot_watveg_beg
    delsoilmoist = tot_watsoil_end - tot_watsoil_beg
    delswe = snow_end - snow_beg
    soilwet = jnp.where(mx_eau_var > 0.0, tot_watsoil_end / mx_eau_var, 0.0)
    return HydrolAlmaResult(
        tot_watveg_end,
        tot_watsoil_end,
        snow_end,
        tot_watveg_end,
        tot_watsoil_end,
        snow_end,
        delintercept,
        delsoilmoist,
        delswe,
        soilwet,
    )


def explicitsnow_initialize_zero_state(
    *,
    kjpindex,
    nsnow=3,
    zero_celsius=273.15,
    snow_density=50.0,
    dtype=jnp.float64,
) -> HydrolSnowState:
    """Build explicit-snow no-restart defaults from ``explicitsnow_initialize``.

    Fortran provenance: ``explicitsnow.f90::explicitsnow_initialize`` lines
    73-86 set missing restart ``snowrho=xrhosmin``, ``snowtemp=tp_00``,
    ``snowdz=0``, ``snowheat=0``, and ``snowgrain=0``. HYDROL scalar snow mass
    and age fields are initialized separately in ``hydrol_init`` and are
    returned as zero placeholders here only to keep a complete snow state
    object for cold-start coverage.
    """

    kjpindex = int(kjpindex)
    nsnow = int(nsnow)
    if kjpindex < 1 or nsnow < 1:
        raise ValueError("kjpindex and nsnow must be positive")
    layer_shape = (kjpindex, nsnow)
    zero_1d = jnp.zeros((kjpindex,), dtype=dtype)
    zero_layers = jnp.zeros(layer_shape, dtype=dtype)
    return HydrolSnowState(
        snow=zero_1d,
        snowdz=zero_layers,
        snowrho=jnp.full(layer_shape, float(snow_density), dtype=dtype),
        snowtemp=jnp.full(layer_shape, float(zero_celsius), dtype=dtype),
        snowheat=zero_layers,
        snowgrain=zero_layers,
        snow_age=zero_1d,
        snow_nobio=jnp.zeros((kjpindex, 1), dtype=dtype),
        snow_nobio_age=jnp.zeros((kjpindex, 1), dtype=dtype),
        tot_melt=zero_1d,
        provenance=EXPLICITSNOW_INITIALIZE_ZERO_STATE_PROVENANCE,
        notes=("Covers only the no-restart defaults created before the first explicitsnow_main call.",),
    )


def snow3lgrain_0d_explicit(
    snowrho,
    *,
    zsnowrad_agrain=1.6e-4,
    zsnowrad_bgrain=1.1e-13,
    zdgrain_max=2.796e-3,
):
    """Compute the Fortran ``snow3lgrain_0d`` new-snow grain size.

    Fortran provenance: ``qsat_moisture.f90::snow3lgrain_0d`` lines
    1055-1091.
    """

    snowrho = jnp.asarray(snowrho, dtype=jnp.float64)
    grain = jnp.asarray(zsnowrad_agrain, dtype=snowrho.dtype) + jnp.asarray(zsnowrad_bgrain, dtype=snowrho.dtype) * (
        snowrho**4
    )
    return jnp.minimum(jnp.asarray(zdgrain_max, dtype=snowrho.dtype), grain)


def snow3lscap_explicit(snowrho, *, xci=2.106e3):
    """Snow heat capacity from Fortran ``snow3lscap_1d/2d``."""

    snowrho = jnp.asarray(snowrho, dtype=jnp.float64)
    return snowrho * jnp.asarray(xci, dtype=snowrho.dtype)


def snow3lhold_explicit(
    snowrho,
    snowdz,
    *,
    xrhosmax=750.0,
    xwsnowholdmax1=0.03,
    xwsnowholdmax2=0.10,
    xsnowrhohold=200.0,
    ph2o=1000.0,
):
    """Maximum liquid-water holding capacity of snow layers.

    Fortran provenance: ``qsat_moisture.f90::snow3lhold_0d/1d/2d`` lines
    593-737.
    """

    snowrho = jnp.asarray(snowrho, dtype=jnp.float64)
    snowdz = jnp.asarray(snowdz, dtype=snowrho.dtype)
    if snowrho.shape != snowdz.shape:
        raise ValueError("snowrho and snowdz must share shape")
    zsnowrho = jnp.minimum(jnp.asarray(xrhosmax, dtype=snowrho.dtype), snowrho)
    hold_ratio = jnp.asarray(xwsnowholdmax1, dtype=snowrho.dtype) + (
        jnp.asarray(xwsnowholdmax2, dtype=snowrho.dtype) - jnp.asarray(xwsnowholdmax1, dtype=snowrho.dtype)
    ) * jnp.maximum(0.0, jnp.asarray(xsnowrhohold, dtype=snowrho.dtype) - zsnowrho) / jnp.asarray(
        xsnowrhohold, dtype=snowrho.dtype
    )
    hold = hold_ratio * snowdz * zsnowrho / jnp.asarray(ph2o, dtype=snowrho.dtype)
    return jnp.where(zsnowrho >= jnp.asarray(xrhosmax, dtype=snowrho.dtype), 0.0, hold)


def snow3lheat_explicit(
    snowliq,
    snowrho,
    snowdz,
    snowtemp,
    *,
    zero_celsius=273.15,
    xci=2.106e3,
    chalfu0=0.3336e6,
    ph2o=1000.0,
):
    """Compute snow enthalpy from liquid water, density, depth, and temperature.

    Fortran provenance: ``qsat_moisture.f90::snow3lheat_1d`` lines 803-826
    and ``snow3lheat_2d`` lines 759-781.
    """

    snowliq = jnp.asarray(snowliq, dtype=jnp.float64)
    snowrho = jnp.asarray(snowrho, dtype=snowliq.dtype)
    snowdz = jnp.asarray(snowdz, dtype=snowliq.dtype)
    snowtemp = jnp.asarray(snowtemp, dtype=snowliq.dtype)
    if snowliq.shape != snowrho.shape or snowliq.shape != snowdz.shape or snowliq.shape != snowtemp.shape:
        raise ValueError("snowliq, snowrho, snowdz, and snowtemp must share shape")
    scap = snow3lscap_explicit(snowrho, xci=xci)
    return snowdz * (
        scap * (snowtemp - jnp.asarray(zero_celsius, dtype=snowliq.dtype))
        - jnp.asarray(chalfu0, dtype=snowliq.dtype) * snowrho
    ) + jnp.asarray(chalfu0, dtype=snowliq.dtype) * jnp.asarray(ph2o, dtype=snowliq.dtype) * snowliq


def snow3ltemp_explicit(
    snowheat,
    snowrho,
    snowdz,
    *,
    zero_celsius=273.15,
    xci=2.106e3,
    chalfu0=0.3336e6,
    one_dimensional_guard=True,
):
    """Diagnose snow temperature from heat content.

    Fortran provenance: ``qsat_moisture.f90::snow3ltemp_1d`` lines 949-972
    and ``snow3ltemp_2d`` lines 907-929. The 1D source helper resets
    temperatures ``<= 100 K`` to ``tp_00``; keep that behavior enabled by
    default because ``explicitsnow_main`` calls the 1D helper per land point.
    """

    snowheat = jnp.asarray(snowheat, dtype=jnp.float64)
    snowrho = jnp.asarray(snowrho, dtype=snowheat.dtype)
    snowdz = jnp.asarray(snowdz, dtype=snowheat.dtype)
    if snowheat.shape != snowrho.shape or snowheat.shape != snowdz.shape:
        raise ValueError("snowheat, snowrho, and snowdz must share shape")
    scap = snow3lscap_explicit(snowrho, xci=xci)
    safe_dz = jnp.where(snowdz != 0.0, snowdz, 1.0)
    raw = jnp.asarray(zero_celsius, dtype=snowheat.dtype) + (
        (snowheat / safe_dz + jnp.asarray(chalfu0, dtype=snowheat.dtype) * snowrho) / scap
    )
    temp = jnp.minimum(jnp.asarray(zero_celsius, dtype=snowheat.dtype), raw)
    if bool(one_dimensional_guard):
        temp = jnp.where(temp <= 100.0, jnp.asarray(zero_celsius, dtype=snowheat.dtype), temp)
    return temp


def snow3lliq_explicit(
    snowheat,
    snowrho,
    snowdz,
    snowtemp,
    *,
    zero_celsius=273.15,
    xci=2.106e3,
    chalfu0=0.3336e6,
    ph2o=1000.0,
):
    """Diagnose snow liquid-water content from heat and temperature.

    Fortran provenance: ``qsat_moisture.f90::snow3lliq_1d`` lines 1175-1204
    and ``snow3lliq_2d`` lines 1114-1152.
    """

    snowheat = jnp.asarray(snowheat, dtype=jnp.float64)
    snowrho = jnp.asarray(snowrho, dtype=snowheat.dtype)
    snowdz = jnp.asarray(snowdz, dtype=snowheat.dtype)
    snowtemp = jnp.asarray(snowtemp, dtype=snowheat.dtype)
    if snowheat.shape != snowrho.shape or snowheat.shape != snowdz.shape or snowheat.shape != snowtemp.shape:
        raise ValueError("snowheat, snowrho, snowdz, and snowtemp must share shape")
    scap = snow3lscap_explicit(snowrho, xci=xci)
    liq = (
        ((jnp.asarray(zero_celsius, dtype=snowheat.dtype) - snowtemp) * scap + jnp.asarray(chalfu0, dtype=snowheat.dtype) * snowrho)
        * snowdz
        + snowheat
    ) / (jnp.asarray(chalfu0, dtype=snowheat.dtype) * jnp.asarray(ph2o, dtype=snowheat.dtype))
    return jnp.maximum(0.0, liq)


def explicitsnow_fall_step(
    *,
    precip_snow,
    temp_air,
    u,
    v,
    totfrac_nobio,
    snowrho,
    snowdz,
    snowheat,
    snowgrain,
    snowtemp,
    zero_celsius=273.15,
    min_wind=0.1,
    xrhosmin=50.0,
    snowfall_a_sn=109.0,
    snowfall_b_sn=6.0,
    snowfall_c_sn=26.0,
    xci=2.106e3,
    chalfu0=0.3336e6,
    dgrain_new_max=2.0e-4,
    zero=0.0,
) -> ExplicitSnowFallResult:
    """Incorporate snowfall into the explicit three-layer snowpack.

    Fortran provenance: ``explicitsnow.f90::explicitsnow_fall`` lines
    1139-1274. This is the snowfall subroutine only: it updates density,
    thickness, heat, and grain size for new snow, including the snow-free
    ground branch that initializes all snow layers evenly. Later
    ``explicitsnow_main`` calls such as layer transformation, melt/refreeze,
    grain metamorphism, and max-mass removal remain separate kernels.
    """

    precip_snow = _as_1d("precip_snow", precip_snow)
    temp_air = _as_1d("temp_air", temp_air)
    u = _as_1d("u", u)
    v = _as_1d("v", v)
    totfrac_nobio = _as_1d("totfrac_nobio", totfrac_nobio)
    snowrho = _as_2d("snowrho", snowrho)
    snowdz = _as_2d("snowdz", snowdz)
    snowheat = _as_2d("snowheat", snowheat)
    snowgrain = _as_2d("snowgrain", snowgrain)
    snowtemp = _as_2d("snowtemp", snowtemp)
    _require_same_npts(precip_snow, temp_air, u, v, totfrac_nobio, snowrho, snowdz, snowheat, snowgrain, snowtemp)
    if snowrho.shape != snowdz.shape or snowheat.shape != snowdz.shape or snowgrain.shape != snowdz.shape or snowtemp.shape != snowdz.shape:
        raise ValueError("snowrho, snowdz, snowheat, snowgrain, and snowtemp must share shape (npts,nsnow)")

    nsnow = snowdz.shape[1]
    dtype = snowdz.dtype
    zero_celsius = jnp.asarray(zero_celsius, dtype=dtype)
    speed = jnp.maximum(jnp.asarray(min_wind, dtype=dtype), jnp.sqrt(u * u + v * v))
    active = precip_snow > jnp.asarray(zero, dtype=dtype)
    land_frac = 1.0 - totfrac_nobio
    psnowhmass = jnp.where(
        active,
        precip_snow * land_frac * (jnp.asarray(xci, dtype=dtype) * (snowtemp[:, 0] - zero_celsius) - jnp.asarray(chalfu0, dtype=dtype)),
        0.0,
    )
    rhosnew = jnp.maximum(
        jnp.asarray(xrhosmin, dtype=dtype),
        jnp.asarray(snowfall_a_sn, dtype=dtype)
        + jnp.asarray(snowfall_b_sn, dtype=dtype) * (temp_air - zero_celsius)
        + jnp.asarray(snowfall_c_sn, dtype=dtype) * jnp.sqrt(speed),
    )
    dsnowfall = jnp.where(active, precip_snow * land_frac / rhosnew, 0.0)
    snow_depth_old = jnp.sum(snowdz, axis=1)
    snowdz_old = snowdz
    newgrain = jnp.minimum(jnp.asarray(dgrain_new_max, dtype=dtype), snow3lgrain_0d_explicit(rhosnew))

    has_new_snow = dsnowfall > jnp.asarray(zero, dtype=dtype)
    top_denom = snowdz[:, 0] + dsnowfall
    safe_top_denom = jnp.where(has_new_snow, top_denom, 1.0)
    top_rho = (snowdz[:, 0] * snowrho[:, 0] + dsnowfall * rhosnew) / safe_top_denom
    top_grain = (snowdz_old[:, 0] * snowgrain[:, 0] + dsnowfall * newgrain) / safe_top_denom
    snowrho = snowrho.at[:, 0].set(jnp.where(has_new_snow, top_rho, snowrho[:, 0]))
    snowdz = snowdz.at[:, 0].set(jnp.where(has_new_snow, top_denom, snowdz[:, 0]))
    snowheat = snowheat.at[:, 0].set(jnp.where(has_new_snow, snowheat[:, 0] + psnowhmass, snowheat[:, 0]))
    snowgrain = snowgrain.at[:, 0].set(jnp.where(has_new_snow, top_grain, snowgrain[:, 0]))

    snow_free_new = has_new_snow & (snow_depth_old == 0.0)
    snowdz_even = dsnowfall[:, None] / jnp.asarray(nsnow, dtype=dtype)
    snowheat_even = psnowhmass[:, None] / jnp.asarray(nsnow, dtype=dtype)
    snowrho_even = jnp.broadcast_to(rhosnew[:, None], snowdz.shape)
    snowgrain_even = jnp.broadcast_to(newgrain[:, None], snowdz.shape)
    snowdz = jnp.where(snow_free_new[:, None], snowdz_even, snowdz)
    snowheat = jnp.where(snow_free_new[:, None], snowheat_even, snowheat)
    snowrho = jnp.where(snow_free_new[:, None], snowrho_even, snowrho)
    snowgrain = jnp.where(snow_free_new[:, None], snowgrain_even, snowgrain)

    return ExplicitSnowFallResult(
        snowrho=snowrho,
        snowdz=snowdz,
        snowheat=snowheat,
        snowgrain=snowgrain,
        psnowhmass=psnowhmass,
        dsnowfall=dsnowfall,
        rhosnew=rhosnew,
        newgrain=newgrain,
        provenance=EXPLICITSNOW_FALL_PROVENANCE,
    )


def explicitsnow_levels_step(
    snow_thick,
    *,
    nsnow: int = 3,
    zsgcoef1=(0.25, 0.50, 0.25),
    zsgcoef2=(0.05, 0.34),
    zsnowtrans=0.20,
    xsnowcritd=0.03,
    dtype=jnp.float64,
) -> ExplicitSnowLevelsResult:
    """Compute three-layer explicit-snow thickness from total snow depth.

    Fortran provenance: ``explicitsnow.f90::explicitsnow_levels`` lines
    1568-1630. The source module is compiled for ``nsnow=3``; this helper
    keeps that contract explicit instead of inventing a generalized layer
    scheme.
    """

    if int(nsnow) != 3:
        raise ValueError("explicitsnow_levels_step follows the Fortran nsnow=3 coefficients")
    snow_thick = _as_1d("snow_thick", snow_thick).astype(dtype)
    coef1 = jnp.asarray(zsgcoef1, dtype=dtype)
    coef2 = jnp.asarray(zsgcoef2, dtype=dtype)
    snowdz = jnp.zeros((snow_thick.shape[0], 3), dtype=dtype)

    shallow = snow_thick <= (jnp.asarray(xsnowcritd, dtype=dtype) + 0.01)
    shallow_edge = jnp.minimum(0.01, snow_thick / 3.0)
    shallow_mid = snow_thick - shallow_edge - shallow_edge
    shallow_dz = jnp.stack((shallow_edge, shallow_mid, shallow_edge), axis=1)

    mid = (snow_thick <= jnp.asarray(zsnowtrans, dtype=dtype)) & (
        snow_thick > (jnp.asarray(xsnowcritd, dtype=dtype) + 0.01)
    )
    mid_dz = snow_thick[:, None] * coef1[None, :]

    deep = snow_thick > jnp.asarray(zsnowtrans, dtype=dtype)
    top = jnp.full_like(snow_thick, coef2[0])
    middle = (snow_thick - coef2[0]) * coef2[1] + coef2[0]
    middle = jnp.minimum(10.0 * coef2[0], middle)
    bottom = snow_thick - middle - top
    deep_dz = jnp.stack((top, middle, bottom), axis=1)

    snowdz = jnp.where(shallow[:, None], shallow_dz, snowdz)
    snowdz = jnp.where(mid[:, None], mid_dz, snowdz)
    snowdz = jnp.where(deep[:, None], deep_dz, snowdz)
    return ExplicitSnowLevelsResult(
        snowdz=snowdz,
        provenance=EXPLICITSNOW_LEVELS_PROVENANCE,
    )


def explicitsnow_grain_step(
    *,
    snowliq,
    snowdz,
    gtemp,
    snowtemp,
    pb,
    snowgrain,
    dt_sechiba=1800.0,
    zero_celsius=273.15,
    xsnowdmin=1.0e-6,
    xrhosmin=50.0,
    chalsu0=2.8345e6,
    chalev0=2.5008e6,
    xrv=6.0221367e23 * 1.380658e-23 / 18.0153e-3,
    ztheta_crit=0.02,
    zc1_ice=8.047e9,
    zc1_liq=5.726e8,
    zdeos=0.92e-4,
    zg1=5.0e-7,
    zg2=4.0e-12,
    ztheta_w=0.05,
    ztheta_crit_w=0.14,
    zdzmin=0.01,
    xp00=1.0e5,
) -> ExplicitSnowGrainResult:
    """Apply explicit-snow dry/wet grain-size metamorphism.

    Fortran provenance: ``explicitsnow.f90::explicitsnow_grain`` lines
    609-752. This kernel updates only ``snowgrain``; it does not perform
    snowfall insertion, layer transformation, compaction, heat solve, or melt.
    """

    snowliq = _as_2d("snowliq", snowliq)
    snowdz = _as_2d("snowdz", snowdz)
    snowtemp = _as_2d("snowtemp", snowtemp)
    snowgrain = _as_2d("snowgrain", snowgrain)
    gtemp = _as_1d("gtemp", gtemp)
    pb = _as_1d("pb", pb)
    _require_same_npts(snowliq, snowdz, snowtemp, snowgrain, gtemp, pb)
    if snowliq.shape != snowdz.shape or snowliq.shape != snowtemp.shape or snowliq.shape != snowgrain.shape:
        raise ValueError("snowliq, snowdz, snowtemp, and snowgrain must share shape (npts,nsnow)")

    npts, nsnow = snowdz.shape
    dtype = snowdz.dtype
    zsnowdz = jnp.maximum(jnp.asarray(xsnowdmin, dtype=dtype) / jnp.asarray(nsnow, dtype=dtype), snowdz)
    zdz_mid = zsnowdz[:, :-1] + zsnowdz[:, 1:]
    zdz = jnp.concatenate((zdz_mid, zsnowdz[:, -1:]), axis=1)
    ztheta = snowliq / jnp.maximum(jnp.asarray(xsnowdmin, dtype=dtype), zsnowdz)

    zthetaa_mid = (zsnowdz[:, :-1] * ztheta[:, :-1] + zsnowdz[:, 1:] * ztheta[:, 1:]) / zdz_mid
    zthetaa = jnp.concatenate((ztheta[:, :1], zthetaa_mid, ztheta[:, -1:]), axis=1)
    ztemp_mid = (zsnowdz[:, :-1] * snowtemp[:, :-1] + zsnowdz[:, 1:] * snowtemp[:, 1:]) / zdz_mid
    ztemp = jnp.concatenate((snowtemp[:, :1], ztemp_mid, snowtemp[:, -1:]), axis=1)

    chalsu0 = jnp.asarray(chalsu0, dtype=dtype)
    chalev0 = jnp.asarray(chalev0, dtype=dtype)
    xrv = jnp.asarray(xrv, dtype=dtype)
    zexpo_ice = chalsu0 / (xrv * ztemp)
    zckt_ice = (jnp.asarray(zc1_ice, dtype=dtype) / ztemp**2) * (zexpo_ice - 1.0) * jnp.exp(-zexpo_ice)
    zexpo_liq = chalev0 / (xrv * ztemp)
    zckt_liq = (jnp.asarray(zc1_liq, dtype=dtype) / ztemp**2) * (zexpo_liq - 1.0) * jnp.exp(-zexpo_liq)
    zfrac = jnp.minimum(1.0, zthetaa / jnp.asarray(ztheta_crit, dtype=dtype))
    zckt = zfrac * zckt_liq + (1.0 - zfrac) * zckt_ice
    zdiff = (
        jnp.asarray(zdeos, dtype=dtype)
        * (jnp.asarray(xp00, dtype=dtype) / (pb[:, None] * 100.0))
        * ((ztemp / jnp.asarray(zero_celsius, dtype=dtype)) ** 6)
        * zckt
    )

    mid_grad = 2.0 * (snowtemp[:, :-1] - snowtemp[:, 1:]) / jnp.maximum(jnp.asarray(zdzmin, dtype=dtype), zdz_mid)
    base_grad = (
        2.0
        * (snowtemp[:, -1:] - jnp.minimum(jnp.asarray(zero_celsius, dtype=dtype), gtemp[:, None]))
        / jnp.maximum(jnp.asarray(zdzmin, dtype=dtype), zdz[:, -1:])
    )
    ztgrad = jnp.concatenate((jnp.zeros((npts, 1), dtype=dtype), mid_grad, base_grad), axis=1)

    zgrainmin = snow3lgrain_0d_explicit(jnp.asarray(xrhosmin, dtype=dtype))
    denom = jnp.maximum(zgrainmin, snowgrain)
    dry_increment = (
        jnp.asarray(dt_sechiba, dtype=dtype)
        * jnp.asarray(zg1, dtype=dtype)
        / denom
        * (
            zdiff[:, :-1] * jnp.maximum(0.0, ztgrad[:, :-1])
            - zdiff[:, 1:] * jnp.minimum(0.0, ztgrad[:, 1:])
        )
    )
    wet_increment = (
        jnp.asarray(dt_sechiba, dtype=dtype)
        * jnp.asarray(zg2, dtype=dtype)
        / denom
        * jnp.minimum(jnp.asarray(ztheta_crit_w, dtype=dtype), ztheta + jnp.asarray(ztheta_w, dtype=dtype))
    )
    updated = snowgrain + jnp.where(ztheta == 0.0, dry_increment, wet_increment)
    return ExplicitSnowGrainResult(
        snowgrain=updated,
        ztheta=ztheta,
        provenance=EXPLICITSNOW_GRAIN_PROVENANCE,
    )


def explicitsnow_compactn_step(
    *,
    snowtemp,
    snowrho,
    snowdz,
    dt_sechiba=1800.0,
    zero_celsius=273.15,
    xrhosmax=750.0,
    zsnowcmpct_rhod=150.0,
    zsnowcmpct_acm=2.8e-6,
    zsnowcmpct_bcm=0.04,
    zsnowcmpct_ccm=460.0,
    zsnowcmpct_v0=3.7e7,
    zsnowcmpct_vt=0.081,
    zsnowcmpct_vr=0.018,
    cte_grav=9.80665,
) -> ExplicitSnowCompactionResult:
    """Apply explicit-snow overburden and fresh-snow settling.

    Fortran provenance: ``explicitsnow.f90::explicitsnow_compactn`` lines
    760-869. The source updates a layer only when the grid point has nonzero
    total snow depth and the layer density is below ``xrhosmax``; it does not
    clamp the post-compaction density back to ``xrhosmax``.
    """

    snowtemp = _as_2d("snowtemp", snowtemp)
    snowrho = _as_2d("snowrho", snowrho)
    snowdz = _as_2d("snowdz", snowdz)
    _require_same_npts(snowtemp, snowrho, snowdz)
    if snowtemp.shape != snowrho.shape or snowrho.shape != snowdz.shape:
        raise ValueError("snowtemp, snowrho, and snowdz must share shape (npts,nsnow)")

    dtype = snowdz.dtype
    zwsnowdz = snowdz * snowrho
    zsmass = jnp.cumsum(zwsnowdz, axis=1)
    point_has_snow = jnp.sum(snowdz, axis=1) > 0.0
    below_density_limit = snowrho < jnp.asarray(xrhosmax, dtype=dtype)
    active = point_has_snow[:, None] & below_density_limit

    settle = jnp.asarray(zsnowcmpct_acm, dtype=dtype) * jnp.exp(
        -jnp.asarray(zsnowcmpct_bcm, dtype=dtype)
        * (jnp.asarray(zero_celsius, dtype=dtype) - jnp.minimum(jnp.asarray(zero_celsius, dtype=dtype), snowtemp))
        - jnp.asarray(zsnowcmpct_ccm, dtype=dtype) * jnp.maximum(0.0, snowrho - jnp.asarray(zsnowcmpct_rhod, dtype=dtype))
    )
    viscocity = jnp.asarray(zsnowcmpct_v0, dtype=dtype) * jnp.exp(
        jnp.asarray(zsnowcmpct_vt, dtype=dtype)
        * (jnp.asarray(zero_celsius, dtype=dtype) - jnp.minimum(jnp.asarray(zero_celsius, dtype=dtype), snowtemp))
        + jnp.asarray(zsnowcmpct_vr, dtype=dtype) * snowrho
    )
    new_rho = snowrho + snowrho * jnp.asarray(dt_sechiba, dtype=dtype) * (
        (jnp.asarray(cte_grav, dtype=dtype) * zsmass / viscocity) + settle
    )
    safe_new_rho = jnp.where(active, new_rho, snowrho)
    new_dz = snowdz * (snowrho / safe_new_rho)
    new_dz = jnp.where(active, new_dz, snowdz)

    zsettle = jnp.where(active, settle, jnp.asarray(zsnowcmpct_acm, dtype=dtype))
    zviscocity = jnp.where(active, viscocity, jnp.asarray(zsnowcmpct_v0, dtype=dtype))
    return ExplicitSnowCompactionResult(
        snowrho=safe_new_rho,
        snowdz=new_dz,
        zsmass=zsmass,
        zsettle=zsettle,
        zviscocity=zviscocity,
        provenance=EXPLICITSNOW_COMPACTION_PROVENANCE,
    )


def explicitsnow_gone_step(
    *,
    pgflux,
    snowheat,
    snowtemp,
    snowdz,
    snowrho,
    snowliq,
    grndflux,
    snowmelt,
    soilflxresid,
    dt_sechiba=1800.0,
    zero_celsius=273.15,
    min_sechiba=1.0e-8,
) -> ExplicitSnowGoneResult:
    """Remove snowpack when available energy can melt all snow heat.

    Fortran provenance: ``explicitsnow.f90::explicitsnow_gone`` lines
    1300-1366. The source resets ``snowmelt(:)=0`` on entry, preserves
    density, and only resets snow temperature to ``tp_00`` for the energy-
    removed snowpack branch.
    """

    pgflux = _as_1d("pgflux", pgflux)
    soilflxresid = _as_1d("soilflxresid", soilflxresid)
    grndflux = _as_1d("grndflux", grndflux)
    _as_1d("snowmelt", snowmelt)
    snowheat = _as_2d("snowheat", snowheat)
    snowtemp = _as_2d("snowtemp", snowtemp)
    snowdz = _as_2d("snowdz", snowdz)
    snowrho = _as_2d("snowrho", snowrho)
    snowliq = _as_2d("snowliq", snowliq)
    _require_same_npts(pgflux, soilflxresid, grndflux, snowheat, snowtemp, snowdz, snowrho, snowliq)
    if (
        snowheat.shape != snowtemp.shape
        or snowheat.shape != snowdz.shape
        or snowheat.shape != snowrho.shape
        or snowheat.shape != snowliq.shape
    ):
        raise ValueError("snowheat, snowtemp, snowdz, snowrho, and snowliq must share shape (npts,nsnow)")

    dtype = snowdz.dtype
    snowmelt_reset = jnp.zeros_like(pgflux)
    totsnowheat = jnp.sum(snowheat, axis=1)
    has_snow = jnp.sum(snowdz, axis=1) > jnp.asarray(min_sechiba, dtype=dtype)
    enough_energy = pgflux + soilflxresid >= -totsnowheat / jnp.asarray(dt_sechiba, dtype=dtype)
    remove = has_snow & enough_energy
    snow_mass = jnp.sum(snowrho * snowdz, axis=1)

    snowgone_delta = jnp.where(remove, 0.0, 1.0)
    new_grndflux = jnp.where(
        remove,
        pgflux + totsnowheat / jnp.asarray(dt_sechiba, dtype=dtype) + soilflxresid,
        grndflux,
    )
    new_snowmelt = jnp.where(remove, snowmelt_reset + snow_mass, snowmelt_reset)
    new_snowdz = jnp.where(has_snow[:, None], snowdz * snowgone_delta[:, None], 0.0)
    new_snowliq = jnp.where(has_snow[:, None], snowliq * snowgone_delta[:, None], 0.0)
    new_snowtemp = jnp.where(
        has_snow[:, None],
        (1.0 - snowgone_delta[:, None]) * jnp.asarray(zero_celsius, dtype=dtype)
        + snowtemp * snowgone_delta[:, None],
        snowtemp,
    )

    return ExplicitSnowGoneResult(
        snowtemp=new_snowtemp,
        snowdz=new_snowdz,
        snowrho=snowrho,
        snowliq=new_snowliq,
        grndflux=new_grndflux,
        snowmelt=new_snowmelt,
        totsnowheat=totsnowheat,
        provenance=EXPLICITSNOW_GONE_PROVENANCE,
    )


def explicitsnow_profile_step(
    *,
    cgrnd_snow,
    dgrnd_snow,
    lambda_snow,
    temp_sol_new,
    snowtemp,
    snowdz,
    temp_sol_add,
) -> ExplicitSnowProfileResult:
    """Update the explicit snow temperature profile from diffusion coefficients.

    Fortran provenance: ``explicitsnow.f90::explicitsnow_profile`` lines
    1655-1694. For ``nsnow=3`` the second and third snow temperatures are
    recursively diagnosed from coefficient columns 1 and 2 in Fortran order.
    """

    cgrnd_snow = _as_2d("cgrnd_snow", cgrnd_snow)
    dgrnd_snow = _as_2d("dgrnd_snow", dgrnd_snow)
    snowtemp = _as_2d("snowtemp", snowtemp)
    snowdz = _as_2d("snowdz", snowdz)
    lambda_snow = _as_1d("lambda_snow", lambda_snow)
    temp_sol_new = _as_1d("temp_sol_new", temp_sol_new)
    temp_sol_add = _as_1d("temp_sol_add", temp_sol_add)
    _require_same_npts(cgrnd_snow, dgrnd_snow, snowtemp, snowdz, lambda_snow, temp_sol_new, temp_sol_add)
    if cgrnd_snow.shape != dgrnd_snow.shape or cgrnd_snow.shape != snowtemp.shape or cgrnd_snow.shape != snowdz.shape:
        raise ValueError("cgrnd_snow, dgrnd_snow, snowtemp, and snowdz must share shape (npts,nsnow)")

    dtype = snowtemp.dtype
    has_snow = jnp.sum(snowdz, axis=1) > 0.0
    first = (lambda_snow * cgrnd_snow[:, 0] + (temp_sol_new + temp_sol_add)) / (
        lambda_snow * (1.0 - dgrnd_snow[:, 0]) + 1.0
    )
    updated = snowtemp.at[:, 0].set(jnp.where(has_snow, first, snowtemp[:, 0]))
    for jg in range(0, int(snowtemp.shape[1]) - 1):
        next_temp = cgrnd_snow[:, jg] + dgrnd_snow[:, jg] * updated[:, jg]
        updated = updated.at[:, jg + 1].set(jnp.where(has_snow, next_temp, updated[:, jg + 1]))
    new_temp_sol_add = jnp.where(has_snow, jnp.asarray(0.0, dtype=dtype), temp_sol_add)
    return ExplicitSnowProfileResult(
        snowtemp=updated,
        temp_sol_add=new_temp_sol_add,
        provenance=EXPLICITSNOW_PROFILE_PROVENANCE,
    )


def explicitsnow_transf_step(
    *,
    snowdz_old,
    snowdz,
    snowrho,
    snowheat,
    snowgrain,
    xsnowcritd=0.03,
) -> ExplicitSnowTransformResult:
    """Redistribute snow mass, heat, and grain after layer-depth resetting.

    Fortran provenance: ``explicitsnow.f90::explicitsnow_transf`` lines
    891-1132. This follows the compiled ``nsnow=3`` source path and preserves
    the source formulas, including the thin/new-snow mixing branch that weights
    SWE and grain by ``snowdz_old``.
    """

    snowdz_old = _as_2d("snowdz_old", snowdz_old)
    snowdz = _as_2d("snowdz", snowdz)
    snowrho = _as_2d("snowrho", snowrho)
    snowheat = _as_2d("snowheat", snowheat)
    snowgrain = _as_2d("snowgrain", snowgrain)
    _require_same_npts(snowdz_old, snowdz, snowrho, snowheat, snowgrain)
    if (
        snowdz_old.shape != snowdz.shape
        or snowdz.shape != snowrho.shape
        or snowdz.shape != snowheat.shape
        or snowdz.shape != snowgrain.shape
    ):
        raise ValueError("snowdz_old, snowdz, snowrho, snowheat, and snowgrain must share shape (npts,nsnow)")
    if int(snowdz.shape[1]) != 3:
        raise ValueError("explicitsnow_transf_step follows the Fortran nsnow=3 source path")

    dtype = snowdz.dtype
    xsnowcritd = jnp.asarray(xsnowcritd, dtype=dtype)
    psnow = jnp.sum(snowdz, axis=1)
    old_nonzero = jnp.all(snowdz_old != 0.0, axis=1)
    stable = (psnow >= xsnowcritd) & old_nonzero
    thin_mix = (psnow > 0.0) & ((psnow < xsnowcritd) | (~old_nonzero))

    zsnowzo = jnp.cumsum(snowdz_old, axis=1)
    zsnowzn = jnp.cumsum(snowdz, axis=1)
    zsnowddz = zsnowzn - zsnowzo
    zdelta = jnp.where(zsnowddz > 0.0, 1.0, 0.0)
    safe_new_dz = jnp.where(snowdz != 0.0, snowdz, 1.0)
    safe_old_dz = jnp.where(snowdz_old != 0.0, snowdz_old, 1.0)

    stable_rho = jnp.zeros_like(snowrho)
    stable_heat = jnp.zeros_like(snowheat)
    stable_grain = jnp.zeros_like(snowgrain)

    top_rho = (
        snowdz_old[:, 0] * snowrho[:, 0]
        + zsnowddz[:, 0] * (zdelta[:, 0] * snowrho[:, 1] + (1.0 - zdelta[:, 0]) * snowrho[:, 0])
    ) / safe_new_dz[:, 0]
    top_heat = snowheat[:, 0] + zsnowddz[:, 0] * (
        zdelta[:, 0] * snowheat[:, 1] / safe_old_dz[:, 1]
        + (1.0 - zdelta[:, 0]) * snowheat[:, 0] / safe_old_dz[:, 0]
    )
    top_grain = (
        snowdz_old[:, 0] * snowgrain[:, 0]
        + zsnowddz[:, 0] * (zdelta[:, 0] * snowgrain[:, 1] + (1.0 - zdelta[:, 0]) * snowgrain[:, 0])
    ) / safe_new_dz[:, 0]

    bottom_rho = (
        snowdz_old[:, 2] * snowrho[:, 2]
        - zsnowddz[:, 1] * (zdelta[:, 1] * snowrho[:, 2] + (1.0 - zdelta[:, 1]) * snowrho[:, 1])
    ) / safe_new_dz[:, 2]
    bottom_heat = snowheat[:, 2] - zsnowddz[:, 1] * (
        zdelta[:, 1] * snowheat[:, 2] / safe_old_dz[:, 2]
        + (1.0 - zdelta[:, 1]) * snowheat[:, 1] / safe_old_dz[:, 1]
    )
    bottom_grain = (
        snowdz_old[:, 2] * snowgrain[:, 2]
        - zsnowddz[:, 1] * (zdelta[:, 1] * snowgrain[:, 2] + (1.0 - zdelta[:, 1]) * snowgrain[:, 1])
    ) / safe_new_dz[:, 2]

    upper_bound = zsnowzn[:, 0]
    lower_bound = zsnowzn[:, 1]
    loc_upper = jnp.sum(upper_bound[:, None] > zsnowzo, axis=1)
    loc_lower = jnp.sum(lower_bound[:, None] > zsnowzo, axis=1)
    loc_upper = jnp.minimum(loc_upper, 2)
    loc_lower = jnp.minimum(loc_lower, 2)
    same_loc = loc_upper == loc_lower
    npts = snowdz.shape[0]
    rows = jnp.arange(npts)
    upper_zo = zsnowzo[rows, loc_upper]
    lower_prev_index = jnp.maximum(loc_lower - 1, 0)
    lower_prev_zo = zsnowzo[rows, lower_prev_index]
    upper_rho = snowrho[rows, loc_upper]
    lower_rho = snowrho[rows, loc_lower]
    upper_heat = snowheat[rows, loc_upper]
    lower_heat = snowheat[rows, loc_lower]
    upper_grain = snowgrain[rows, loc_upper]
    lower_grain = snowgrain[rows, loc_lower]
    upper_old_dz = safe_old_dz[rows, loc_upper]
    lower_old_dz = safe_old_dz[rows, loc_lower]

    middle_rho_diff = upper_rho * (upper_zo - zsnowzn[:, 0]) + lower_rho * (zsnowzn[:, 1] - lower_prev_zo)
    middle_heat_diff = (
        upper_heat * (upper_zo - zsnowzn[:, 0]) / upper_old_dz
        + lower_heat * (zsnowzn[:, 1] - lower_prev_zo) / lower_old_dz
    )
    middle_grain_diff = upper_grain * (upper_zo - zsnowzn[:, 0]) + lower_grain * (zsnowzn[:, 1] - lower_prev_zo)
    for kk in range(3):
        in_source_loop = (kk >= loc_upper) & (kk < loc_lower)
        factor = (jnp.asarray(kk, dtype=dtype) - loc_upper.astype(dtype))
        middle_rho_diff = middle_rho_diff + jnp.where(in_source_loop, factor * snowrho[:, kk] * snowdz_old[:, kk], 0.0)
        middle_heat_diff = middle_heat_diff + jnp.where(in_source_loop, factor * snowheat[:, kk], 0.0)
        middle_grain_diff = middle_grain_diff + jnp.where(
            in_source_loop,
            factor * snowgrain[:, kk] * snowdz_old[:, kk],
            0.0,
        )

    middle_rho_same = snowrho[rows, loc_upper]
    middle_heat_same = snowheat[rows, loc_upper] * snowdz[:, 1] / safe_old_dz[rows, loc_upper]
    middle_grain_same = snowgrain[rows, loc_upper]
    middle_rho = jnp.where(same_loc, middle_rho_same, middle_rho_diff / safe_new_dz[:, 1])
    middle_heat = jnp.where(same_loc, middle_heat_same, middle_heat_diff)
    middle_grain = jnp.where(same_loc, middle_grain_same, middle_grain_diff / safe_new_dz[:, 1])

    stable_rho = stable_rho.at[:, 0].set(top_rho)
    stable_rho = stable_rho.at[:, 1].set(middle_rho)
    stable_rho = stable_rho.at[:, 2].set(bottom_rho)
    stable_heat = stable_heat.at[:, 0].set(top_heat)
    stable_heat = stable_heat.at[:, 1].set(middle_heat)
    stable_heat = stable_heat.at[:, 2].set(bottom_heat)
    stable_grain = stable_grain.at[:, 0].set(top_grain)
    stable_grain = stable_grain.at[:, 1].set(middle_grain)
    stable_grain = stable_grain.at[:, 2].set(bottom_grain)

    zsumheat = jnp.sum(snowheat, axis=1)
    zsumswe = jnp.sum(snowrho * snowdz_old, axis=1)
    zsumgrain = jnp.sum(snowgrain * snowdz_old, axis=1)
    safe_psnow = jnp.where(psnow != 0.0, psnow, 1.0)
    mixed_dz = jnp.broadcast_to((psnow / 3.0)[:, None], snowdz.shape)
    mixed_rho = jnp.broadcast_to((zsumswe / safe_psnow)[:, None], snowrho.shape)
    mixed_heat = jnp.broadcast_to((zsumheat / 3.0)[:, None], snowheat.shape)
    mixed_grain = jnp.broadcast_to((zsumgrain / safe_psnow)[:, None], snowgrain.shape)

    out_rho = jnp.where(stable[:, None], stable_rho, snowrho)
    out_heat = jnp.where(stable[:, None], stable_heat, snowheat)
    out_grain = jnp.where(stable[:, None], stable_grain, snowgrain)
    out_dz = snowdz
    out_rho = jnp.where(thin_mix[:, None], mixed_rho, out_rho)
    out_heat = jnp.where(thin_mix[:, None], mixed_heat, out_heat)
    out_grain = jnp.where(thin_mix[:, None], mixed_grain, out_grain)
    out_dz = jnp.where(thin_mix[:, None], mixed_dz, out_dz)

    return ExplicitSnowTransformResult(
        snowrho=out_rho,
        snowdz=out_dz,
        snowheat=out_heat,
        snowgrain=out_grain,
        provenance=EXPLICITSNOW_TRANSFORM_PROVENANCE,
    )


def explicitsnow_melt_refrz_step(
    *,
    precip_rain,
    pgflux,
    soilcap,
    snowtemp,
    snowdz,
    snowrho,
    snowliq,
    snowmelt,
    grndflux,
    temp_air,
    soilflxresid,
    dt_sechiba=1800.0,
    min_sechiba=1.0e-8,
    zero_celsius=273.15,
    xci=2.106e3,
    chalfu0=0.3336e6,
    ph2o=1000.0,
    xsnowdmin=1.0e-6,
) -> ExplicitSnowMeltRefreezeResult:
    """Apply snow melt, refreezing, liquid drainage, and melt heat feedback.

    Fortran provenance: ``explicitsnow.f90::explicitsnow_melt_refrz`` lines
    1385-1563. Rain input is intentionally diagnosed as zero inside the source
    body, so ``precip_rain`` is accepted for boundary parity but does not alter
    the local liquid-flow calculation.
    """

    precip_rain = _as_1d("precip_rain", precip_rain)
    pgflux = _as_1d("pgflux", pgflux)
    soilcap = _as_1d("soilcap", soilcap)
    snowmelt = _as_1d("snowmelt", snowmelt)
    grndflux = _as_1d("grndflux", grndflux)
    temp_air = _as_1d("temp_air", temp_air)
    soilflxresid = _as_1d("soilflxresid", soilflxresid)
    snowtemp = _as_2d("snowtemp", snowtemp)
    snowdz = _as_2d("snowdz", snowdz)
    snowrho = _as_2d("snowrho", snowrho)
    snowliq = _as_2d("snowliq", snowliq)
    _require_same_npts(
        precip_rain,
        pgflux,
        soilcap,
        snowmelt,
        grndflux,
        temp_air,
        soilflxresid,
        snowtemp,
        snowdz,
        snowrho,
        snowliq,
    )
    if snowtemp.shape != snowdz.shape or snowtemp.shape != snowrho.shape or snowtemp.shape != snowliq.shape:
        raise ValueError("snowtemp, snowdz, snowrho, and snowliq must share shape (npts,nsnow)")

    dtype = snowdz.dtype
    nsnow = int(snowdz.shape[1])
    snowmass = jnp.sum(snowrho * snowdz, axis=1)
    active = snowmass > jnp.asarray(min_sechiba, dtype=dtype)
    pcapa_snow = snowrho * jnp.asarray(xci, dtype=dtype)
    zsnowlwe = snowrho * snowdz / jnp.asarray(ph2o, dtype=dtype)
    chalfu0 = jnp.asarray(chalfu0, dtype=dtype)
    ph2o = jnp.asarray(ph2o, dtype=dtype)
    safe_heat_denom = jnp.where(pcapa_snow * snowdz != 0.0, pcapa_snow * snowdz, 1.0)

    zphase = jnp.minimum(
        pcapa_snow * jnp.maximum(0.0, snowtemp - jnp.asarray(zero_celsius, dtype=dtype)) * snowdz,
        jnp.maximum(0.0, zsnowlwe - snowliq) * chalfu0 * ph2o,
    )
    zsnowmelt = zphase / (chalfu0 * ph2o)
    zsnowtemp = snowtemp - zphase / safe_heat_denom
    melted_temp = jnp.minimum(jnp.asarray(zero_celsius, dtype=dtype), zsnowtemp)
    zmeltxs = (zsnowtemp - melted_temp) * pcapa_snow * snowdz
    zwholdmax = snow3lhold_explicit(snowrho, snowdz)
    denom = zsnowlwe - jnp.minimum(snowliq, zwholdmax)
    numer = zsnowlwe - jnp.minimum(snowliq + zsnowmelt, zwholdmax)
    nonzero_denom = denom != 0.0
    safe_denom = jnp.where(nonzero_denom, denom, 1.0)
    zcmprsfact = jnp.where(nonzero_denom, numer / safe_denom, 1.0)
    melted_dz = snowdz * zcmprsfact
    nonzero_melted_dz = melted_dz != 0.0
    safe_melted_dz = jnp.where(nonzero_melted_dz, melted_dz, 1.0)
    melted_rho = jnp.where(
        nonzero_melted_dz,
        zsnowlwe * ph2o / safe_melted_dz,
        snowrho,
    )
    melted_liq = snowliq + zsnowmelt

    zscap = melted_rho * jnp.asarray(xci, dtype=dtype)
    zphase2 = jnp.minimum(
        zscap * jnp.maximum(0.0, jnp.asarray(zero_celsius, dtype=dtype) - melted_temp) * melted_dz,
        melted_liq * chalfu0 * ph2o,
    )
    zsnowdz = jnp.maximum(jnp.asarray(xsnowdmin, dtype=dtype) / jnp.asarray(nsnow, dtype=dtype), melted_dz)
    safe_freeze_denom = jnp.where(zscap * zsnowdz != 0.0, zscap * zsnowdz, 1.0)
    refrozen_temp = melted_temp + zphase2 / safe_freeze_denom
    refrozen_liq = melted_liq - ((refrozen_temp - melted_temp) * zscap * zsnowdz / (chalfu0 * ph2o))
    refrozen_liq = jnp.maximum(refrozen_liq, 0.0)

    zwholdmax2 = snow3lhold_explicit(melted_rho, melted_dz)
    flowliq = jnp.maximum(0.0, refrozen_liq - zwholdmax2)
    drained_liq = refrozen_liq - flowliq
    safe_melted_rho = jnp.where(melted_rho != 0.0, melted_rho, 1.0)
    drained_dz = jnp.maximum(
        0.0,
        melted_dz - flowliq * ph2o / safe_melted_rho,
    )

    zflowliqt = jnp.zeros((snowdz.shape[0], nsnow + 1), dtype=dtype)
    for jj in range(nsnow):
        zflowliqt = zflowliqt.at[:, jj + 1].set(flowliq[:, jj])

    flow_after = jnp.zeros_like(flowliq)
    percolated_liq = drained_liq
    percolated_rho = melted_rho
    zsnowliq = drained_liq
    min_layer_dz = jnp.asarray(xsnowdmin, dtype=dtype) / jnp.asarray(nsnow, dtype=dtype)
    for jj in range(nsnow):
        layer_liq = percolated_liq[:, jj] + zflowliqt[:, jj]
        layer_flow = jnp.maximum(0.0, layer_liq - zwholdmax2[:, jj])
        layer_liq = layer_liq - layer_flow
        density_increment = (layer_liq - zsnowliq[:, jj]) * ph2o / jnp.maximum(min_layer_dz, drained_dz[:, jj])
        percolated_liq = percolated_liq.at[:, jj].set(layer_liq)
        flow_after = flow_after.at[:, jj].set(layer_flow)
        percolated_rho = percolated_rho.at[:, jj].set(percolated_rho[:, jj] + density_increment)
        zflowliqt = zflowliqt.at[:, jj + 1].set(zflowliqt[:, jj + 1] + layer_flow)

    meltxs = jnp.sum(zmeltxs, axis=1) / jnp.asarray(dt_sechiba, dtype=dtype)
    active_snowmelt = snowmelt + zflowliqt[:, nsnow] * ph2o
    active_grndflux = grndflux + meltxs

    out_snowtemp = jnp.where(active[:, None], refrozen_temp, snowtemp)
    out_snowdz = jnp.where(active[:, None], drained_dz, 0.0)
    out_snowrho = jnp.where(active[:, None], percolated_rho, snowrho)
    out_snowliq = jnp.where(active[:, None], percolated_liq, 0.0)
    out_snowmelt = jnp.where(active, active_snowmelt, snowmelt)
    out_grndflux = jnp.where(active, active_grndflux, grndflux)
    out_meltxs = jnp.where(active, meltxs, 0.0)

    return ExplicitSnowMeltRefreezeResult(
        snowtemp=out_snowtemp,
        snowdz=out_snowdz,
        snowrho=out_snowrho,
        snowliq=out_snowliq,
        snowmelt=out_snowmelt,
        grndflux=out_grndflux,
        meltxs=out_meltxs,
        melt_mass_by_layer=jnp.where(active[:, None], zsnowmelt * ph2o, 0.0),
        refreeze_mass_by_layer=jnp.where(active[:, None], zphase2 / chalfu0, 0.0),
        liquid_percolation_by_interface=jnp.where(
            active[:, None],
            zflowliqt[:, 1:] * ph2o,
            0.0,
        ),
        provenance=EXPLICITSNOW_MELT_REFREEZE_PROVENANCE,
    )


def explicitsnow_sublimation_step(
    *,
    vevapsno,
    frac_nobio,
    totfrac_nobio,
    snowrho,
    snowdz,
    snowliq,
    snowtemp,
    snowcri=1.5,
    min_sechiba=1.0e-8,
    zero_celsius=273.15,
    iice: int = 0,
) -> ExplicitSnowSublimationResult:
    """Remove vegetated-fraction snow sublimation from explicit snow layers.

    Fortran provenance: ``explicitsnow.f90::explicitsnow_main`` lines
    255-323. ``iice`` is zero-based here and corresponds to Fortran ``iice``
    in the ``frac_nobio``/``subsnownobio`` axis.
    """

    vevapsno = _as_1d("vevapsno", vevapsno)
    frac_nobio = _as_2d("frac_nobio", frac_nobio)
    totfrac_nobio = _as_1d("totfrac_nobio", totfrac_nobio)
    snowrho = _as_2d("snowrho", snowrho)
    snowdz = _as_2d("snowdz", snowdz)
    snowliq = _as_2d("snowliq", snowliq)
    snowtemp = _as_2d("snowtemp", snowtemp)
    _require_same_npts(vevapsno, frac_nobio, totfrac_nobio, snowrho, snowdz, snowliq, snowtemp)
    if snowrho.shape != snowdz.shape or snowrho.shape != snowliq.shape or snowrho.shape != snowtemp.shape:
        raise ValueError("snowrho, snowdz, snowliq, and snowtemp must share shape (npts,nsnow)")
    iice = int(iice)
    if iice < 0 or iice >= frac_nobio.shape[1]:
        raise ValueError("iice must select an existing frac_nobio column")

    dtype = snowdz.dtype
    snow = jnp.sum(snowrho * snowdz, axis=1)
    frac_ice = frac_nobio[:, iice]
    enough_snow = snow > jnp.asarray(snowcri, dtype=dtype)
    has_ice_frac = frac_ice > jnp.asarray(min_sechiba, dtype=dtype)
    subsnownobio_iice = jnp.where(
        enough_snow,
        frac_ice * vevapsno,
        jnp.where(has_ice_frac, vevapsno, 0.0),
    )
    subsnowveg = jnp.where(enough_snow, vevapsno - subsnownobio_iice, jnp.where(has_ice_frac, 0.0, vevapsno))

    deplete = subsnowveg > snow
    veg_fraction = 1.0 - totfrac_nobio
    subsinksoil = jnp.where(
        deplete & (veg_fraction > jnp.asarray(min_sechiba, dtype=dtype)),
        (subsnowveg - snow) / veg_fraction,
        0.0,
    )
    limited_subsnowveg = jnp.where(deplete, snow, subsnowveg)
    limited_vevapsno = jnp.where(deplete, limited_subsnowveg + subsnownobio_iice, vevapsno)

    mass_cum = jnp.cumsum(snowrho * snowdz, axis=1)
    loc = jnp.zeros_like(snow, dtype=jnp.int32)
    loc = jnp.where((mass_cum[:, 0] <= limited_subsnowveg) & (mass_cum[:, 1] >= limited_subsnowveg), 1, loc)
    loc = jnp.where((mass_cum[:, 1] <= limited_subsnowveg) & (mass_cum[:, 2] >= limited_subsnowveg), 2, loc)
    partial = ~deplete
    remove_top = limited_subsnowveg / snowrho[:, 0]
    dz_loc0 = snowdz.at[:, 0].set(jnp.maximum(0.0, snowdz[:, 0] - remove_top))

    snowacc1 = snowdz[:, 0] * snowrho[:, 0]
    remove_layer1 = (limited_subsnowveg - snowacc1) / snowrho[:, 1]
    dz_loc1 = snowdz.at[:, 0].set(0.0)
    dz_loc1 = dz_loc1.at[:, 1].set(jnp.maximum(0.0, snowdz[:, 1] - remove_layer1))

    snowacc2 = snowdz[:, 0] * snowrho[:, 0] + snowdz[:, 1] * snowrho[:, 1]
    remove_layer2 = (limited_subsnowveg - snowacc2) / snowrho[:, 2]
    dz_loc2 = snowdz.at[:, 0].set(0.0)
    dz_loc2 = dz_loc2.at[:, 1].set(0.0)
    dz_loc2 = dz_loc2.at[:, 2].set(jnp.maximum(0.0, snowdz[:, 2] - remove_layer2))
    partial_dz = jnp.where((loc == 0)[:, None], dz_loc0, jnp.where((loc == 1)[:, None], dz_loc1, dz_loc2))

    new_snowdz = jnp.where(deplete[:, None], 0.0, jnp.where(partial[:, None], partial_dz, snowdz))
    new_snowliq = jnp.where(deplete[:, None], 0.0, snowliq)
    new_snowtemp = jnp.where(deplete[:, None], jnp.asarray(zero_celsius, dtype=dtype), snowtemp)
    new_snow = jnp.where(deplete, 0.0, jnp.sum(snowrho * new_snowdz, axis=1))
    subsnownobio = jnp.zeros((snowdz.shape[0], frac_nobio.shape[1]), dtype=dtype)
    subsnownobio = subsnownobio.at[:, iice].set(subsnownobio_iice)

    return ExplicitSnowSublimationResult(
        snow=new_snow,
        snowdz=new_snowdz,
        snowliq=new_snowliq,
        snowtemp=new_snowtemp,
        vevapsno=limited_vevapsno,
        subsnownobio=subsnownobio,
        subsnowveg=limited_subsnowveg,
        subsinksoil=subsinksoil,
        provenance=EXPLICITSNOW_SUBLIMATION_PROVENANCE,
    )


def explicitsnow_maxmass_step(
    *,
    soilcap,
    snowrho,
    snowdz,
    maxmass_snow=3000.0,
    chalfu0=0.3336e6,
) -> ExplicitSnowMaxMassResult:
    """Remove excessive snow mass from the bottom layers.

    Fortran provenance: ``explicitsnow.f90::explicitsnow_main`` lines
    325-376. The source computes a per-step removal cap from ``soilcap`` and
    removes mass from the lowest snow layer upward.
    """

    soilcap = _as_1d("soilcap", soilcap)
    snowrho = _as_2d("snowrho", snowrho)
    snowdz = _as_2d("snowdz", snowdz)
    _require_same_npts(soilcap, snowrho, snowdz)
    if snowrho.shape != snowdz.shape:
        raise ValueError("snowrho and snowdz must share shape (npts,nsnow)")
    if int(snowdz.shape[1]) != 3:
        raise ValueError("explicitsnow_maxmass_step follows the Fortran nsnow=3 source path")

    dtype = snowdz.dtype
    snow = jnp.sum(snowrho * snowdz, axis=1)
    maxmass_snow = jnp.asarray(maxmass_snow, dtype=dtype)
    snow_d1k = jnp.where(
        snow > 1.2 * maxmass_snow,
        3.0 * soilcap / jnp.asarray(chalfu0, dtype=dtype),
        soilcap / jnp.asarray(chalfu0, dtype=dtype),
    )
    snowmelt_from_maxmass = jnp.where(
        snow > maxmass_snow,
        jnp.minimum(snow - maxmass_snow, snow_d1k),
        0.0,
    )

    layer_mass = snowrho * snowdz
    smass_bottom = jnp.stack(
        (
            layer_mass[:, 2] + layer_mass[:, 1] + layer_mass[:, 0],
            layer_mass[:, 2] + layer_mass[:, 1],
            layer_mass[:, 2],
        ),
        axis=1,
    )
    loc = jnp.full_like(snowmelt_from_maxmass, 2, dtype=jnp.int32)
    loc = jnp.where(
        (smass_bottom[:, 2] <= snowmelt_from_maxmass) & (smass_bottom[:, 1] >= snowmelt_from_maxmass),
        1,
        loc,
    )
    loc = jnp.where(
        (smass_bottom[:, 1] <= snowmelt_from_maxmass) & (smass_bottom[:, 0] >= snowmelt_from_maxmass),
        0,
        loc,
    )

    remove_bottom = snowmelt_from_maxmass / snowrho[:, 2]
    dz_loc2 = snowdz.at[:, 2].set(jnp.maximum(0.0, snowdz[:, 2] - remove_bottom))

    snow_remove_loc1 = layer_mass[:, 2]
    remove_layer1 = (snowmelt_from_maxmass - snow_remove_loc1) / snowrho[:, 1]
    dz_loc1 = snowdz.at[:, 2].set(0.0)
    dz_loc1 = dz_loc1.at[:, 1].set(jnp.maximum(0.0, snowdz[:, 1] - remove_layer1))

    snow_remove_loc0 = layer_mass[:, 2] + layer_mass[:, 1]
    remove_layer0 = (snowmelt_from_maxmass - snow_remove_loc0) / snowrho[:, 0]
    dz_loc0 = snowdz.at[:, 2].set(0.0)
    dz_loc0 = dz_loc0.at[:, 1].set(0.0)
    dz_loc0 = dz_loc0.at[:, 0].set(jnp.maximum(0.0, snowdz[:, 0] - remove_layer0))

    candidate = jnp.where((loc == 2)[:, None], dz_loc2, jnp.where((loc == 1)[:, None], dz_loc1, dz_loc0))
    active = snow > maxmass_snow
    new_snowdz = jnp.where(active[:, None], candidate, snowdz)
    new_snow = jnp.sum(snowrho * new_snowdz, axis=1)
    return ExplicitSnowMaxMassResult(
        snow=new_snow,
        snowdz=new_snowdz,
        snowmelt_from_maxmass=snowmelt_from_maxmass,
        provenance=EXPLICITSNOW_MAXMASS_PROVENANCE,
    )


def explicitsnow_age_ice_step(
    *,
    precip_snow,
    precip_rain,
    temp_sol_new_old,
    temp_sol_new,
    soilcap,
    frac_nobio,
    subsnownobio,
    snow,
    snow_age,
    snow_nobio,
    snow_nobio_age,
    dt_sechiba=1800.0,
    one_day=86400.0,
    zero_celsius=273.15,
    chalfu0=0.3336e6,
    max_snow_age=50.0,
    snow_trans=0.2,
    maxmass_snow=3000.0,
    iice: int = 0,
) -> ExplicitSnowAgeIceResult:
    """Update land snow age and the land-ice snow reservoir.

    Fortran provenance: ``explicitsnow.f90::explicitsnow_main`` lines
    408-499. The source stops when more than one non-bio surface type is
    present; this helper keeps the same supported boundary by requiring a
    single ``frac_nobio`` column unless a future audited branch handles others.
    """

    precip_snow = _as_1d("precip_snow", precip_snow)
    precip_rain = _as_1d("precip_rain", precip_rain)
    temp_sol_new_old = _as_1d("temp_sol_new_old", temp_sol_new_old)
    temp_sol_new = _as_1d("temp_sol_new", temp_sol_new)
    soilcap = _as_1d("soilcap", soilcap)
    snow = _as_1d("snow", snow)
    snow_age = _as_1d("snow_age", snow_age)
    frac_nobio = _as_2d("frac_nobio", frac_nobio)
    subsnownobio = _as_2d("subsnownobio", subsnownobio)
    snow_nobio = _as_2d("snow_nobio", snow_nobio)
    snow_nobio_age = _as_2d("snow_nobio_age", snow_nobio_age)
    _require_same_npts(
        precip_snow,
        precip_rain,
        temp_sol_new_old,
        temp_sol_new,
        soilcap,
        snow,
        snow_age,
        frac_nobio,
        subsnownobio,
        snow_nobio,
        snow_nobio_age,
    )
    if frac_nobio.shape != subsnownobio.shape or frac_nobio.shape != snow_nobio.shape or frac_nobio.shape != snow_nobio_age.shape:
        raise ValueError("frac_nobio, subsnownobio, snow_nobio, and snow_nobio_age must share shape")
    if frac_nobio.shape[1] > 1:
        raise ValueError("explicitsnow_age_ice_step follows the source branch with nnobio == 1")
    iice = int(iice)
    if iice < 0 or iice >= frac_nobio.shape[1]:
        raise ValueError("iice must select an existing frac_nobio column")

    dtype = snow.dtype
    frac_ice = frac_nobio[:, iice]
    snow_nobio_ice = snow_nobio[:, iice] + frac_ice * (precip_snow + precip_rain)
    snow_nobio_ice = snow_nobio_ice - subsnownobio[:, iice]
    snowmelt_tmp = jnp.where(
        temp_sol_new_old > jnp.asarray(zero_celsius, dtype=dtype),
        frac_ice * (temp_sol_new_old - jnp.asarray(zero_celsius, dtype=dtype)) * soilcap / jnp.asarray(chalfu0, dtype=dtype),
        0.0,
    )
    snowmelt_tmp = jnp.where(snowmelt_tmp > snow_nobio_ice, jnp.maximum(0.0, snow_nobio_ice), snowmelt_tmp)
    snow_nobio_ice = snow_nobio_ice - snowmelt_tmp
    icemelt = jnp.where(snow_nobio_ice >= jnp.asarray(maxmass_snow, dtype=dtype), snow_nobio_ice - maxmass_snow, 0.0)
    snow_nobio_ice = jnp.where(snow_nobio_ice >= jnp.asarray(maxmass_snow, dtype=dtype), maxmass_snow, snow_nobio_ice)
    new_snow_nobio = snow_nobio.at[:, iice].set(snow_nobio_ice)

    age_increment = (
        snow_age
        + (1.0 - snow_age / jnp.asarray(max_snow_age, dtype=dtype)) * jnp.asarray(dt_sechiba / one_day, dtype=dtype)
    ) * jnp.exp(-precip_snow / jnp.asarray(snow_trans, dtype=dtype))
    new_snow_age = jnp.where(snow <= 0.0, 0.0, age_increment)

    old_ice_age = snow_nobio_age[:, iice]
    d_age = (
        old_ice_age
        + (1.0 - old_ice_age / jnp.asarray(max_snow_age, dtype=dtype)) * jnp.asarray(dt_sechiba / one_day, dtype=dtype)
    ) * jnp.exp(-precip_snow / jnp.asarray(snow_trans, dtype=dtype)) - old_ice_age
    xx = jnp.maximum(jnp.asarray(zero_celsius, dtype=dtype) - temp_sol_new, 0.0)
    xx = (xx / 7.0) ** 4.0
    d_age = jnp.where(d_age > 0.0, d_age / (1.0 + xx), d_age)
    new_ice_age = jnp.where(snow_nobio_ice <= 0.0, 0.0, jnp.maximum(old_ice_age + d_age, 0.0))
    new_snow_nobio_age = snow_nobio_age.at[:, iice].set(new_ice_age)

    return ExplicitSnowAgeIceResult(
        snow_age=new_snow_age,
        snow_nobio=new_snow_nobio,
        snow_nobio_age=new_snow_nobio_age,
        snowmelt_ice=snowmelt_tmp,
        icemelt=icemelt,
        provenance=EXPLICITSNOW_AGE_ICE_PROVENANCE,
    )


def explicitsnow_liquid_heat_excess_step(
    *,
    snowliq,
    snowdz,
    snowrho,
    grndflux,
    dt_sechiba=1800.0,
    chalfu0=0.3336e6,
    ph2o=1000.0,
) -> ExplicitSnowLiquidHeatExcessResult:
    """Limit excessive snow liquid heat and add the excess to ground flux.

    Fortran provenance: ``explicitsnow.f90::explicitsnow_main`` lines
    386-402.
    """

    snowliq = _as_2d("snowliq", snowliq)
    snowdz = _as_2d("snowdz", snowdz)
    snowrho = _as_2d("snowrho", snowrho)
    grndflux = _as_1d("grndflux", grndflux)
    _require_same_npts(snowliq, snowdz, snowrho, grndflux)
    if snowliq.shape != snowdz.shape or snowliq.shape != snowrho.shape:
        raise ValueError("snowliq, snowdz, and snowrho must share shape (npts,nsnow)")

    dtype = snowliq.dtype
    zliqheatxs = (
        jnp.maximum(0.0, snowliq * jnp.asarray(ph2o, dtype=dtype) - 0.10 * snowdz * snowrho)
        * jnp.asarray(chalfu0, dtype=dtype)
        / jnp.asarray(dt_sechiba, dtype=dtype)
    )
    new_snowliq = snowliq - zliqheatxs * jnp.asarray(dt_sechiba, dtype=dtype) / (
        jnp.asarray(ph2o, dtype=dtype) * jnp.asarray(chalfu0, dtype=dtype)
    )
    new_snowliq = jnp.maximum(0.0, new_snowliq)
    new_grndflux = grndflux + jnp.sum(zliqheatxs, axis=1)
    return ExplicitSnowLiquidHeatExcessResult(
        snowliq=new_snowliq,
        grndflux=new_grndflux,
        zliqheatxs=zliqheatxs,
        provenance=EXPLICITSNOW_LIQUID_HEAT_EXCESS_PROVENANCE,
    )


def explicitsnow_final_cleanup_step(
    *,
    snow,
    snowrho,
    snowgrain,
    snowdz,
    snowliq,
    icemelt,
    snowmelt,
    snowmelt_ice,
    snowmelt_from_maxmass,
    xrhosmin=50.0,
) -> ExplicitSnowFinalCleanupResult:
    """Apply final no-snow cleanup and synthesize ``tot_melt``.

    Fortran provenance: ``explicitsnow.f90::explicitsnow_main`` lines
    493-515.
    """

    snow = _as_1d("snow", snow)
    icemelt = _as_1d("icemelt", icemelt)
    snowmelt = _as_1d("snowmelt", snowmelt)
    snowmelt_ice = _as_1d("snowmelt_ice", snowmelt_ice)
    snowmelt_from_maxmass = _as_1d("snowmelt_from_maxmass", snowmelt_from_maxmass)
    snowrho = _as_2d("snowrho", snowrho)
    snowgrain = _as_2d("snowgrain", snowgrain)
    snowdz = _as_2d("snowdz", snowdz)
    snowliq = _as_2d("snowliq", snowliq)
    _require_same_npts(snow, icemelt, snowmelt, snowmelt_ice, snowmelt_from_maxmass, snowrho, snowgrain, snowdz, snowliq)
    if snowrho.shape != snowgrain.shape or snowrho.shape != snowdz.shape or snowrho.shape != snowliq.shape:
        raise ValueError("snowrho, snowgrain, snowdz, and snowliq must share shape (npts,nsnow)")

    no_snow = snow == 0.0
    dtype = snowrho.dtype
    new_snowrho = jnp.where(no_snow[:, None], jnp.asarray(xrhosmin, dtype=dtype), snowrho)
    new_snowgrain = jnp.where(no_snow[:, None], 0.0, snowgrain)
    new_snowdz = jnp.where(no_snow[:, None], 0.0, snowdz)
    new_snowliq = jnp.where(no_snow[:, None], 0.0, snowliq)
    tot_melt = icemelt + snowmelt + snowmelt_ice + snowmelt_from_maxmass
    return ExplicitSnowFinalCleanupResult(
        snowrho=new_snowrho,
        snowgrain=new_snowgrain,
        snowdz=new_snowdz,
        snowliq=new_snowliq,
        tot_melt=tot_melt,
        provenance=EXPLICITSNOW_FINAL_CLEANUP_PROVENANCE,
    )


def hydrol_bucket_snow_step(
    *,
    precip_rain,
    precip_snow,
    temp_sol_new,
    soilcap,
    frac_nobio,
    totfrac_nobio,
    vevapsno,
    snow,
    snow_age,
    snow_nobio,
    snow_nobio_age,
    dt_sechiba=1800.0,
    one_day=86400.0,
    zero_celsius=273.15,
    chalfu0=0.3336e6,
    snowcri=1.5,
    sneige=1.5e-3,
    maxmass_snow=3000.0,
    max_snow_age=50.0,
    snow_trans=0.2,
    sn_dens=330.0,
    min_sechiba=1.0e-8,
    iice: int = 0,
) -> HydrolBucketSnowResult:
    """Update the legacy scalar snow bucket from ``hydrol_snow``.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_snow``, lines 4693-4966. ``iice`` is zero-based here
    and corresponds to the Fortran land-ice non-bio column. The source raises a
    fatal error when ``nnobio > 1``; this helper keeps the same supported
    boundary and refuses additional non-bio columns until audited separately.
    """

    precip_rain = _as_1d("precip_rain", precip_rain)
    precip_snow = _as_1d("precip_snow", precip_snow)
    temp_sol_new = _as_1d("temp_sol_new", temp_sol_new)
    soilcap = _as_1d("soilcap", soilcap)
    frac_nobio = _as_2d("frac_nobio", frac_nobio)
    totfrac_nobio = _as_1d("totfrac_nobio", totfrac_nobio)
    vevapsno = _as_1d("vevapsno", vevapsno)
    snow = _as_1d("snow", snow)
    snow_age = _as_1d("snow_age", snow_age)
    snow_nobio = _as_2d("snow_nobio", snow_nobio)
    snow_nobio_age = _as_2d("snow_nobio_age", snow_nobio_age)
    _require_same_npts(
        precip_rain,
        precip_snow,
        temp_sol_new,
        soilcap,
        frac_nobio,
        totfrac_nobio,
        vevapsno,
        snow,
        snow_age,
        snow_nobio,
        snow_nobio_age,
    )
    if frac_nobio.shape != snow_nobio.shape or frac_nobio.shape != snow_nobio_age.shape:
        raise ValueError("frac_nobio, snow_nobio, and snow_nobio_age must share shape (npts,nnobio)")
    if int(frac_nobio.shape[1]) > 1:
        raise ValueError("hydrol_snow source path raises fatal error for nnobio > 1")
    iice = int(iice)
    if iice < 0 or iice >= int(frac_nobio.shape[1]):
        raise ValueError("iice must select an existing non-bio column")

    dtype = snow.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    tp_00 = jnp.asarray(zero_celsius, dtype=dtype)
    chalfu0 = jnp.asarray(chalfu0, dtype=dtype)
    min_sechiba = jnp.asarray(min_sechiba, dtype=dtype)
    frac_ice = frac_nobio[:, iice]

    subsnownobio_iice = jnp.zeros_like(snow)
    subsnowveg = jnp.zeros_like(snow)
    subsinksoil = jnp.zeros_like(snow)
    icemelt = jnp.zeros_like(snow)

    snow = snow + (one - totfrac_nobio) * precip_snow
    enough_snow = snow > jnp.asarray(snowcri, dtype=dtype)
    has_ice_frac = frac_ice > min_sechiba
    subsnownobio_iice = jnp.where(
        enough_snow,
        frac_ice * vevapsno,
        jnp.where(has_ice_frac, vevapsno, zero),
    )
    subsnowveg = jnp.where(enough_snow, vevapsno - subsnownobio_iice, jnp.where(has_ice_frac, zero, vevapsno))

    deplete_veg_snow = subsnowveg > snow
    veg_fraction = one - totfrac_nobio
    subsinksoil = jnp.where(
        deplete_veg_snow & (veg_fraction > min_sechiba),
        (subsnowveg - snow) / veg_fraction,
        zero,
    )
    subsnowveg = jnp.where(deplete_veg_snow, snow, subsnowveg)
    snow_after_sub = jnp.where(deplete_veg_snow, zero, snow - subsnowveg)
    vevapsno = jnp.where(deplete_veg_snow, subsnowveg + subsnownobio_iice, vevapsno)

    positive_temp = temp_sol_new > tp_00
    raw_snowmelt = (one - frac_ice) * (temp_sol_new - tp_00) * soilcap / chalfu0
    has_snow = snow_after_sub > jnp.asarray(sneige, dtype=dtype)
    enough_to_melt = raw_snowmelt < snow_after_sub
    snowmelt_warm = jnp.where(
        has_snow,
        jnp.where(enough_to_melt, raw_snowmelt, snow_after_sub),
        jnp.where(snow_after_sub >= zero, snow_after_sub, zero),
    )
    snow_warm = jnp.where(has_snow, jnp.where(enough_to_melt, snow_after_sub - raw_snowmelt, zero), zero)
    snowmelt = jnp.where(positive_temp, snowmelt_warm, zero)
    snow = jnp.where(positive_temp, snow_warm, snow_after_sub)

    snow_d1k = soilcap / chalfu0
    over_maxmass = snow > jnp.asarray(maxmass_snow, dtype=dtype)
    maxmass_melt_increment = jnp.minimum(snow - jnp.asarray(maxmass_snow, dtype=dtype), snow_d1k)
    snowmelt_after_maxmass = snowmelt + jnp.where(over_maxmass, maxmass_melt_increment, zero)
    snow = jnp.where(over_maxmass, snow - snowmelt_after_maxmass, snow)
    snowmelt = snowmelt_after_maxmass

    snow_nobio_iice = snow_nobio[:, iice]
    snow_nobio_iice = snow_nobio_iice + frac_ice * precip_snow + frac_ice * precip_rain
    snow_nobio_iice = snow_nobio_iice - subsnownobio_iice

    snowmelt_tmp_raw = frac_ice * (temp_sol_new - tp_00) * soilcap / chalfu0
    snowmelt_tmp_limited = jnp.where(
        snowmelt_tmp_raw > snow_nobio_iice,
        jnp.maximum(zero, snow_nobio_iice),
        snowmelt_tmp_raw,
    )
    snowmelt_tmp = jnp.where(positive_temp, snowmelt_tmp_limited, zero)
    snowmelt = snowmelt + snowmelt_tmp
    snow_nobio_iice = snow_nobio_iice - snowmelt_tmp

    ice_over_maxmass = snow_nobio_iice > jnp.asarray(maxmass_snow, dtype=dtype)
    icemelt = jnp.where(
        ice_over_maxmass,
        jnp.minimum(snow_nobio_iice - jnp.asarray(maxmass_snow, dtype=dtype), snow_d1k),
        zero,
    )
    snow_nobio_iice = jnp.where(ice_over_maxmass, snow_nobio_iice - icemelt, snow_nobio_iice)
    snow_nobio = snow_nobio.at[:, iice].set(snow_nobio_iice)

    tot_melt = icemelt + snowmelt
    dt_days = jnp.asarray(dt_sechiba, dtype=dtype) / jnp.asarray(one_day, dtype=dtype)
    snow_age = jnp.where(
        snow <= zero,
        zero,
        (snow_age + (one - snow_age / jnp.asarray(max_snow_age, dtype=dtype)) * dt_days)
        * jnp.exp(-precip_snow / jnp.asarray(snow_trans, dtype=dtype)),
    )

    snow_nobio_age_iice = snow_nobio_age[:, iice]
    d_age = (
        snow_nobio_age_iice
        + (one - snow_nobio_age_iice / jnp.asarray(max_snow_age, dtype=dtype)) * dt_days
    ) * jnp.exp(-precip_snow / jnp.asarray(snow_trans, dtype=dtype)) - snow_nobio_age_iice
    cold_slowdown = (jnp.maximum(tp_00 - temp_sol_new, zero) / jnp.asarray(7.0, dtype=dtype)) ** 4.0
    d_age = jnp.where(d_age > min_sechiba, d_age / (one + cold_slowdown), d_age)
    snow_nobio_age_iice = jnp.where(
        snow_nobio_iice <= zero,
        zero,
        jnp.maximum(snow_nobio_age_iice + d_age, zero),
    )
    snow_nobio_age = snow_nobio_age.at[:, iice].set(snow_nobio_age_iice)
    snowdepth = snow / jnp.asarray(sn_dens, dtype=dtype)
    subsnownobio = jnp.zeros_like(snow_nobio).at[:, iice].set(subsnownobio_iice)

    return HydrolBucketSnowResult(
        snow=snow,
        snow_age=snow_age,
        snow_nobio=snow_nobio,
        snow_nobio_age=snow_nobio_age,
        vevapsno=vevapsno,
        tot_melt=tot_melt,
        snowmelt=snowmelt,
        snowdepth=snowdepth,
        subsnownobio=subsnownobio,
        subsnowveg=subsnowveg,
        subsinksoil=subsinksoil,
        icemelt=icemelt,
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_snow lines 4693-4966",
            "fortran_source/ORCHIDEE/src_parameters/constantes.f90 lines 430-469",
            "fortran_source/ORCHIDEE/src_parameters/constantes_soil_var.f90 lines 97-99",
        ),
        notes=("Scalar bucket-snow path for OK_EXPLICITSNOW=n; full driver scheduling remains separate.",),
    )


def explicitsnow_main_step(
    *,
    precip_rain,
    precip_snow,
    temp_air,
    pb,
    u,
    v,
    temp_sol_new,
    soilcap,
    pgflux,
    frac_nobio,
    totfrac_nobio,
    gtemp,
    lambda_snow,
    cgrnd_snow,
    dgrnd_snow,
    vevapsno,
    snow_age,
    snow_nobio_age,
    snow_nobio,
    snowrho,
    snowgrain,
    snowdz,
    snowtemp,
    snowheat,
    snow,
    temp_sol_add,
    snowliq,
    subsnownobio,
    grndflux,
    snowmelt,
    soilflxresid,
    dt_sechiba=1800.0,
    one_day=86400.0,
    iice: int = 0,
) -> ExplicitSnowMainResult:
    """Run the source-backed explicit-snow main sequence.

    Fortran provenance: ``explicitsnow.f90::explicitsnow_main`` lines
    108-553. This adapter only assembles already audited local kernels in the
    Fortran order; it keeps ``nnobio == 1`` explicit because the source stops
    for additional non-bio surface types.
    """

    precip_rain = _as_1d("precip_rain", precip_rain)
    precip_snow = _as_1d("precip_snow", precip_snow)
    temp_air = _as_1d("temp_air", temp_air)
    pb = _as_1d("pb", pb)
    u = _as_1d("u", u)
    v = _as_1d("v", v)
    temp_sol_new = _as_1d("temp_sol_new", temp_sol_new)
    soilcap = _as_1d("soilcap", soilcap)
    pgflux = _as_1d("pgflux", pgflux)
    frac_nobio = _as_2d("frac_nobio", frac_nobio)
    totfrac_nobio = _as_1d("totfrac_nobio", totfrac_nobio)
    gtemp = _as_1d("gtemp", gtemp)
    lambda_snow = _as_1d("lambda_snow", lambda_snow)
    cgrnd_snow = _as_2d("cgrnd_snow", cgrnd_snow)
    dgrnd_snow = _as_2d("dgrnd_snow", dgrnd_snow)
    vevapsno = _as_1d("vevapsno", vevapsno)
    snow_age = _as_1d("snow_age", snow_age)
    snow_nobio_age = _as_2d("snow_nobio_age", snow_nobio_age)
    snow_nobio = _as_2d("snow_nobio", snow_nobio)
    snowrho = _as_2d("snowrho", snowrho)
    snowgrain = _as_2d("snowgrain", snowgrain)
    snowdz = _as_2d("snowdz", snowdz)
    snowtemp = _as_2d("snowtemp", snowtemp)
    snowheat = _as_2d("snowheat", snowheat)
    snow = _as_1d("snow", snow)
    temp_sol_add = _as_1d("temp_sol_add", temp_sol_add)
    snowliq = _as_2d("snowliq", snowliq)
    subsnownobio = _as_2d("subsnownobio", subsnownobio)
    grndflux = _as_1d("grndflux", grndflux)
    snowmelt = _as_1d("snowmelt", snowmelt)
    soilflxresid = _as_1d("soilflxresid", soilflxresid)
    _require_same_npts(
        precip_rain,
        precip_snow,
        temp_air,
        pb,
        u,
        v,
        temp_sol_new,
        soilcap,
        pgflux,
        frac_nobio,
        totfrac_nobio,
        gtemp,
        lambda_snow,
        cgrnd_snow,
        dgrnd_snow,
        vevapsno,
        snow_age,
        snow_nobio_age,
        snow_nobio,
        snowrho,
        snowgrain,
        snowdz,
        snowtemp,
        snowheat,
        snow,
        temp_sol_add,
        snowliq,
        subsnownobio,
        grndflux,
        snowmelt,
        soilflxresid,
    )
    if not (
        cgrnd_snow.shape
        == dgrnd_snow.shape
        == snowrho.shape
        == snowgrain.shape
        == snowdz.shape
        == snowtemp.shape
        == snowheat.shape
        == snowliq.shape
    ):
        raise ValueError("explicit-snow layer fields must share shape (npts,nsnow)")
    if frac_nobio.shape != snow_nobio.shape or frac_nobio.shape != snow_nobio_age.shape or frac_nobio.shape != subsnownobio.shape:
        raise ValueError("frac_nobio, snow_nobio, snow_nobio_age, and subsnownobio must share shape")
    if int(snowdz.shape[1]) != 3:
        raise ValueError("explicitsnow_main_step follows the Fortran nsnow=3 source path")
    if frac_nobio.shape[1] > 1:
        raise ValueError("explicitsnow_main_step follows the source branch with nnobio == 1")

    temp_sol_new_old = temp_sol_new
    snowmelt_ice = jnp.zeros_like(snow)
    icemelt = jnp.zeros_like(snow)
    snowmelt = jnp.zeros_like(snowmelt)

    fall = explicitsnow_fall_step(
        precip_snow=precip_snow,
        temp_air=temp_air,
        u=u,
        v=v,
        totfrac_nobio=totfrac_nobio,
        snowrho=snowrho,
        snowdz=snowdz,
        snowheat=snowheat,
        snowgrain=snowgrain,
        snowtemp=snowtemp,
    )
    snowrho = fall.snowrho
    snowdz = fall.snowdz
    snowheat = fall.snowheat
    snowgrain = fall.snowgrain

    snow_depth_tmp = jnp.sum(snowdz, axis=1)
    snowdz_old = snowdz
    levels = explicitsnow_levels_step(snow_depth_tmp)
    snowdz = levels.snowdz

    transf = explicitsnow_transf_step(
        snowdz_old=snowdz_old,
        snowdz=snowdz,
        snowrho=snowrho,
        snowheat=snowheat,
        snowgrain=snowgrain,
    )
    snowrho = transf.snowrho
    snowdz = transf.snowdz
    snowheat = transf.snowheat
    snowgrain = transf.snowgrain

    has_snow = jnp.sum(snowdz, axis=1) > 0.0
    diagnosed_temp = snow3ltemp_explicit(snowheat, snowrho, snowdz)
    snowtemp = jnp.where(has_snow[:, None], diagnosed_temp, 273.15)
    diagnosed_liq = snow3lliq_explicit(snowheat, snowrho, snowdz, snowtemp)
    snowliq = jnp.where(has_snow[:, None], diagnosed_liq, 0.0)

    compact = explicitsnow_compactn_step(
        snowtemp=snowtemp,
        snowrho=snowrho,
        snowdz=snowdz,
        dt_sechiba=dt_sechiba,
    )
    snowrho = compact.snowrho
    snowdz = compact.snowdz
    snowheat = snow3lheat_explicit(snowliq, snowrho, snowdz, snowtemp)

    profile = explicitsnow_profile_step(
        cgrnd_snow=cgrnd_snow,
        dgrnd_snow=dgrnd_snow,
        lambda_snow=lambda_snow,
        temp_sol_new=temp_sol_new,
        snowtemp=snowtemp,
        snowdz=snowdz,
        temp_sol_add=temp_sol_add,
    )
    snowtemp = profile.snowtemp
    temp_sol_add = profile.temp_sol_add

    grndflux = jnp.zeros_like(grndflux)
    gone = explicitsnow_gone_step(
        pgflux=pgflux,
        snowheat=snowheat,
        snowtemp=snowtemp,
        snowdz=snowdz,
        snowrho=snowrho,
        snowliq=snowliq,
        grndflux=grndflux,
        snowmelt=snowmelt,
        soilflxresid=soilflxresid,
        dt_sechiba=dt_sechiba,
    )
    snowtemp = gone.snowtemp
    snowdz = gone.snowdz
    snowrho = gone.snowrho
    snowliq = gone.snowliq
    grndflux = gone.grndflux
    snowmelt = gone.snowmelt

    melt = explicitsnow_melt_refrz_step(
        precip_rain=precip_rain,
        pgflux=pgflux,
        soilcap=soilcap,
        snowtemp=snowtemp,
        snowdz=snowdz,
        snowrho=snowrho,
        snowliq=snowliq,
        snowmelt=snowmelt,
        grndflux=grndflux,
        temp_air=temp_air,
        soilflxresid=soilflxresid,
        dt_sechiba=dt_sechiba,
    )
    snowtemp = melt.snowtemp
    snowdz = melt.snowdz
    snowrho = melt.snowrho
    snowliq = melt.snowliq
    snowmelt = melt.snowmelt
    grndflux = melt.grndflux

    sublim = explicitsnow_sublimation_step(
        vevapsno=vevapsno,
        frac_nobio=frac_nobio,
        totfrac_nobio=totfrac_nobio,
        snowrho=snowrho,
        snowdz=snowdz,
        snowliq=snowliq,
        snowtemp=snowtemp,
        iice=iice,
    )
    snow = sublim.snow
    snowdz = sublim.snowdz
    snowliq = sublim.snowliq
    snowtemp = sublim.snowtemp
    vevapsno = sublim.vevapsno
    subsnownobio = sublim.subsnownobio
    subsinksoil = sublim.subsinksoil

    maxmass = explicitsnow_maxmass_step(
        soilcap=soilcap,
        snowrho=snowrho,
        snowdz=snowdz,
    )
    snow = maxmass.snow
    snowdz = maxmass.snowdz
    snowmelt_from_maxmass = maxmass.snowmelt_from_maxmass

    grain = explicitsnow_grain_step(
        snowliq=snowliq,
        snowdz=snowdz,
        gtemp=gtemp,
        snowtemp=snowtemp,
        pb=pb,
        snowgrain=snowgrain,
        dt_sechiba=dt_sechiba,
    )
    snowgrain = grain.snowgrain

    liquid_excess = explicitsnow_liquid_heat_excess_step(
        snowliq=snowliq,
        snowdz=snowdz,
        snowrho=snowrho,
        grndflux=grndflux,
        dt_sechiba=dt_sechiba,
    )
    snowliq = liquid_excess.snowliq
    grndflux = liquid_excess.grndflux

    snow = jnp.sum(snowrho * snowdz, axis=1)
    snowheat = snow3lheat_explicit(snowliq, snowrho, snowdz, snowtemp)

    age_ice = explicitsnow_age_ice_step(
        precip_snow=precip_snow,
        precip_rain=precip_rain,
        temp_sol_new_old=temp_sol_new_old,
        temp_sol_new=temp_sol_new,
        soilcap=soilcap,
        frac_nobio=frac_nobio,
        subsnownobio=subsnownobio,
        snow=snow,
        snow_age=snow_age,
        snow_nobio=snow_nobio,
        snow_nobio_age=snow_nobio_age,
        dt_sechiba=dt_sechiba,
        one_day=one_day,
        iice=iice,
    )
    snow_age = age_ice.snow_age
    snow_nobio = age_ice.snow_nobio
    snow_nobio_age = age_ice.snow_nobio_age
    snowmelt_ice = snowmelt_ice + age_ice.snowmelt_ice
    icemelt = icemelt + age_ice.icemelt

    cleanup = explicitsnow_final_cleanup_step(
        snow=snow,
        snowrho=snowrho,
        snowgrain=snowgrain,
        snowdz=snowdz,
        snowliq=snowliq,
        icemelt=icemelt,
        snowmelt=snowmelt,
        snowmelt_ice=snowmelt_ice,
        snowmelt_from_maxmass=snowmelt_from_maxmass,
    )
    snowrho = cleanup.snowrho
    snowgrain = cleanup.snowgrain
    snowdz = cleanup.snowdz
    snowliq = cleanup.snowliq
    tot_melt = cleanup.tot_melt

    return ExplicitSnowMainResult(
        snow=snow,
        snowdz=snowdz,
        snowrho=snowrho,
        snowtemp=snowtemp,
        snowheat=snowheat,
        snowliq=snowliq,
        snowgrain=snowgrain,
        snow_age=snow_age,
        snow_nobio=snow_nobio,
        snow_nobio_age=snow_nobio_age,
        vevapsno=vevapsno,
        subsnownobio=subsnownobio,
        subsinksoil=subsinksoil,
        grndflux=grndflux,
        snowmelt=snowmelt,
        tot_melt=tot_melt,
        temp_sol_add=temp_sol_add,
        snowmelt_from_maxmass=snowmelt_from_maxmass,
        melt_mass_by_layer=melt.melt_mass_by_layer,
        refreeze_mass_by_layer=melt.refreeze_mass_by_layer,
        liquid_percolation_by_interface=melt.liquid_percolation_by_interface,
        melt_refreeze_energy=melt.meltxs,
        liquid_excess_energy=liquid_excess.zliqheatxs,
        provenance=EXPLICITSNOW_MAIN_PROVENANCE,
        notes=(
            "Full adapter is source-ordered for nsnow=3 and nnobio=1.",
            "Output diagnostics sent only to XIOS in Fortran are not materialized here.",
        ),
    )


def hydrol_cold_start_state(
    *,
    veget_max,
    soiltile,
    pref_soil_veg,
    nslm: int,
    nsnow: int = 3,
    frac_nobio=None,
    hydrol_moisture_content=0.3,
    ok_freeze_cwrr=True,
    ok_explicitsnow=True,
    zwt_force_default=1.0e20,
    free_drain_coef_default=None,
    peat_hydro=False,
    peat_nodr=False,
    tides=False,
    agri_peat=False,
    agri_drain=False,
    drain_factor=1.0,
    zmaxh=2.0,
    snow_density=50.0,
    zero_celsius=273.15,
    min_sechiba=1.0e-8,
    dtype=jnp.float64,
) -> HydrolColdStartState:
    """Build source-backed HYDROL initialization for ``RESTART_FILEIN=NONE``.

    Fortran provenance follows ``hydrol.f90::hydrol_init`` lines 2813-3115:
    missing restart fields are set with ``setvar_p`` defaults, vegetation
    masks are derived from ``veget_max``/``soiltile``, and explicit-snow layer
    fields come from ``explicitsnow_initialize``. The helper does not run
    ``hydrol_main`` diagnostics and does not infer later same-step states.
    """

    veget_max = jnp.asarray(veget_max, dtype=dtype)
    soiltile = jnp.asarray(soiltile, dtype=dtype)
    pref_soil_veg = jnp.asarray(pref_soil_veg, dtype=jnp.int32)
    if veget_max.ndim != 2:
        raise ValueError("veget_max must have shape (npts,nvm)")
    npts, nvm = veget_max.shape
    nslm = int(nslm)
    nsnow = int(nsnow)
    if npts < 1 or nvm < 1 or nslm < 1 or nsnow < 1:
        raise ValueError("npts, nvm, nslm, and nsnow must be positive")
    if soiltile.ndim != 2 or soiltile.shape[0] != npts:
        raise ValueError("soiltile must have shape (npts,nstm)")
    nstm = soiltile.shape[1]
    if pref_soil_veg.shape != (nvm,):
        raise ValueError("pref_soil_veg must have shape (nvm,)")
    if bool(jnp.any(pref_soil_veg < 1)) or bool(jnp.any(pref_soil_veg > nstm)):
        raise ValueError("pref_soil_veg must contain Fortran 1-based soil tile indices in 1..nstm")
    if frac_nobio is None:
        frac_nobio = jnp.zeros((npts, 1), dtype=dtype)
    else:
        frac_nobio = jnp.asarray(frac_nobio, dtype=dtype)
        if frac_nobio.ndim != 2 or frac_nobio.shape[0] != npts:
            raise ValueError("frac_nobio must have shape (npts,nnobio)")
    nnobio = frac_nobio.shape[1]

    mc = jnp.full((npts, nslm, nstm), float(hydrol_moisture_content), dtype=dtype)
    mcl = mc.copy()
    us = jnp.zeros((npts, nvm, nstm, nslm), dtype=dtype)
    humrelv = jnp.sum(us, axis=3)
    vegstressv = humrelv
    humrel = jnp.zeros((npts, nvm), dtype=dtype)
    water2infilt = jnp.zeros((npts, nstm), dtype=dtype)
    ae_ns = jnp.zeros((npts, nstm), dtype=dtype)
    evap_bare_lim_ns = jnp.zeros((npts, nstm), dtype=dtype)
    vegtot_old = jnp.sum(veget_max, axis=1)
    vegtot = vegtot_old
    evap_bare_lim = jnp.sum(evap_bare_lim_ns * vegtot[:, None] * soiltile, axis=1)

    if free_drain_coef_default is None:
        free_drain = jnp.ones((nstm,), dtype=dtype)
        if bool(peat_hydro) and bool(peat_nodr) and nstm >= 4:
            free_drain = free_drain.at[3].set(0.0)
            if bool(tides) and nstm >= 6:
                free_drain = free_drain.at[5].set(0.0)
        if bool(agri_peat) and bool(agri_drain) and nstm >= 5:
            free_drain = free_drain.at[4].set(float(drain_factor))
    else:
        free_drain = jnp.asarray(free_drain_coef_default, dtype=dtype)
        if free_drain.shape != (nstm,):
            raise ValueError("free_drain_coef_default must have shape (nstm,)")
    free_drain_coef = jnp.broadcast_to(free_drain[None, :], (npts, nstm))
    zwt_force = jnp.full((npts, nstm), float(zwt_force_default), dtype=dtype)
    zforce = bool(jnp.any(zwt_force[0, :] <= jnp.asarray(zmaxh, dtype=dtype)))

    snow = jnp.zeros((npts,), dtype=dtype)
    snow_age = jnp.zeros((npts,), dtype=dtype)
    snow_nobio = jnp.zeros((npts, nnobio), dtype=dtype)
    snow_nobio_age = jnp.zeros((npts, nnobio), dtype=dtype)
    qsintveg = jnp.zeros((npts, nvm), dtype=dtype)

    profil_froz_hydro = None
    profil_froz_hydro_ns = None
    kk = None
    kk_moy = None
    temp_hydro = None
    if bool(ok_freeze_cwrr):
        profil_froz_hydro = jnp.zeros((npts, nslm), dtype=dtype)
        profil_froz_hydro_ns = jnp.zeros((npts, nslm, nstm), dtype=dtype)
        kk = jnp.full((npts, nslm, nstm), 276.48, dtype=dtype)
        kk_moy = jnp.full((npts, nslm), 276.48, dtype=dtype)
        temp_hydro = jnp.full((npts, nslm), 280.0, dtype=dtype)

    zero_1d = jnp.zeros((npts,), dtype=dtype)
    resdist = soiltile
    mask_veget = jnp.where(veget_max > min_sechiba, 1, 0).astype(jnp.int32)
    mask_soiltile = jnp.where(soiltile > min_sechiba, 1, 0).astype(jnp.int32)
    explicit_snow = None
    if bool(ok_explicitsnow):
        explicit_snow = explicitsnow_initialize_zero_state(
            kjpindex=npts,
            nsnow=nsnow,
            zero_celsius=zero_celsius,
            snow_density=snow_density,
            dtype=dtype,
        )
        explicit_snow = explicit_snow._replace(
            snow=snow,
            snow_age=snow_age,
            snow_nobio=snow_nobio,
            snow_nobio_age=snow_nobio_age,
        )
    return HydrolColdStartState(
        mc=mc,
        mcl=mcl,
        us=us,
        humrelv=humrelv,
        vegstressv=vegstressv,
        humrel=humrel,
        water2infilt=water2infilt,
        ae_ns=ae_ns,
        evap_bare_lim_ns=evap_bare_lim_ns,
        evap_bare_lim=evap_bare_lim,
        zwt_force=zwt_force,
        free_drain_coef=free_drain_coef,
        zforce=zforce,
        snow=snow,
        snow_age=snow_age,
        snow_nobio=snow_nobio,
        snow_nobio_age=snow_nobio_age,
        qsintveg=qsintveg,
        profil_froz_hydro=profil_froz_hydro,
        profil_froz_hydro_ns=profil_froz_hydro_ns,
        kk=kk,
        kk_moy=kk_moy,
        temp_hydro=temp_hydro,
        fwet_out=zero_1d,
        run2peat=zero_1d,
        wt_ab=zero_1d,
        wtp=zero_1d,
        fwet_new=zero_1d,
        liqwt_ratio=zero_1d,
        wt_ab_tide=zero_1d,
        run2man=zero_1d,
        resdist=resdist,
        vegtot_old=vegtot_old,
        vegtot=vegtot,
        mask_veget=mask_veget,
        mask_soiltile=mask_soiltile,
        explicit_snow=explicit_snow,
        provenance=HYDROL_COLD_START_STATE_PROVENANCE,
        notes=(
            "Covers hydrol_init missing-restart defaults before hydrol_main.",
            "vegstress restart fallback is not exposed as a closed grid-PFT field here because the source accumulation target is not initialized in this local span.",
            "Later HYDROL soil solve, diagnostics, and same-step moisture exports remain separate process kernels.",
        ),
    )


def _hydrol_init_usda_parameters(config_values: Mapping[str, object] | None = None) -> dict[str, jnp.ndarray]:
    """Select and validate ``hydrol_init`` USDA parameters in source order."""

    defaults = {
        "nvan": USDA_NVAN,
        "avan": USDA_AVAN,
        "mcr": USDA_MCR,
        "mcs_mineral": USDA_MCS,
        "ks": USDA_KS,
        "pcent": USDA_PCENT,
        "mcf_mineral": USDA_MCF,
        "mcw_mineral": USDA_MCW,
        "mc_awet": USDA_MC_AWET,
        "mc_adry": USDA_MC_ADRY,
    }
    config_names = {
        "CWRR_N_VANGENUCHTEN": "nvan",
        "CWRR_A_VANGENUCHTEN": "avan",
        "VWC_RESIDUAL": "mcr",
        "VWC_SAT": "mcs_mineral",
        "CWRR_KS": "ks",
        "WETNESS_TRANSPIR_MAX": "pcent",
        "VWC_FC": "mcf_mineral",
        "VWC_WP": "mcw_mineral",
        "VWC_MIN_FOR_WET_ALB": "mc_awet",
        "VWC_MAX_FOR_DRY_ALB": "mc_adry",
    }
    supplied = {} if config_values is None else dict(config_values)
    unknown = set(supplied) - set(config_names)
    if unknown:
        raise ValueError(f"unsupported hydrol_init config key(s): {sorted(unknown)}")
    parameters = {name: jnp.asarray(values, dtype=jnp.float64) for name, values in defaults.items()}
    for config_name, values in supplied.items():
        parameters[config_names[config_name]] = jnp.asarray(values, dtype=jnp.float64)
    for name, values in parameters.items():
        if values.shape != (12,):
            raise ValueError(f"{name} must have shape (12,) for SOILTYPE_CLASSIF=usda")

    nvan = parameters["nvan"]
    avan = parameters["avan"]
    mcr = parameters["mcr"]
    mcs = parameters["mcs_mineral"]
    ks = parameters["ks"]
    pcent = parameters["pcent"]
    mcf = parameters["mcf_mineral"]
    mcw = parameters["mcw_mineral"]
    awet = parameters["mc_awet"]
    adry = parameters["mc_adry"]
    checks = (
        (jnp.any(nvan <= 0.0), "CWRR_N_VANGENUCHTEN must be positive"),
        (jnp.any(avan <= 0.0), "CWRR_A_VANGENUCHTEN must be positive"),
        (jnp.any((mcr < 0.0) | (mcr > 1.0)), "VWC_RESIDUAL must be in [0,1]"),
        (jnp.any((mcs < 0.0) | (mcs > 1.0) | (mcs <= mcr)), "VWC_SAT must be in [0,1] and above VWC_RESIDUAL"),
        (jnp.any(ks <= 0.0), "CWRR_KS must be positive"),
        (jnp.any((pcent <= 0.0) | (pcent > 1.0)), "WETNESS_TRANSPIR_MAX must be in (0,1]"),
        (jnp.any(mcf > mcs), "VWC_FC must not exceed VWC_SAT"),
        (jnp.any((mcw > mcf) | (mcw < mcr)), "VWC_WP must lie between VWC_RESIDUAL and VWC_FC"),
        (jnp.any(awet < 0.0), "VWC_MIN_FOR_WET_ALB must be non-negative"),
        (jnp.any((adry < 0.0) | (adry > awet)), "VWC_MAX_FOR_DRY_ALB must be non-negative and not exceed VWC_MIN_FOR_WET_ALB"),
    )
    for failed, message in checks:
        if bool(failed):
            raise ValueError(message)
    return parameters


def _hydrol_init_allocation_shapes(*, npts: int, nvm: int, nstm: int, nslm: int) -> dict[str, tuple[int, ...]]:
    """Represent Fortran allocations whose values are assigned by later owners."""

    shapes: dict[str, tuple[int, ...]] = {}
    for name in ("precisol", "precisol_nc", "free_drain_coef", "zwt_force", "water2infilt", "ae_ns", "evap_bare_lim_ns", "resdist"):
        shapes[name] = (npts, nstm)
    for name in ("mask_veget", "humrelv", "vegstressv"):
        shapes[name] = (npts, nvm, nstm) if name != "mask_veget" else (npts, nvm)
    shapes.update(
        us=(npts, nvm, nstm, nslm),
        mc=(npts, nslm, nstm),
        mcl=(npts, nslm, nstm),
        profil_froz_hydro_ns=(npts, nslm, nstm),
        nroot=(npts, nvm, nslm),
        kfact_root=(npts, nslm, nstm),
        vegetmax_soil=(npts, nvm, nstm),
        snow=(npts,),
        snow_age=(npts,),
        qsintveg=(npts, nvm),
        humrel=(npts, nvm),
        vegstress=(npts, nvm),
        soilmoist=(npts, nslm),
        tmc=(npts, nstm),
        tmcs=(npts, nstm),
        tmcr=(npts, nstm),
        ru_ns=(npts, nstm),
        dr_ns=(npts, nstm),
        tr_ns=(npts, nstm),
        tot_watveg_beg=(npts,),
        tot_watsoil_beg=(npts,),
        snow_beg=(npts,),
    )
    return shapes


def hydrol_init_pft14_owner(
    *,
    veget,
    veget_max,
    soiltile,
    pref_soil_veg,
    njsc,
    znh_m,
    dnh_m,
    dlh_m,
    humcste,
    altmax,
    frac_nobio=None,
    restart_state: Mapping[str, object] | None = None,
    hydrol_init_config: Mapping[str, object] | None = None,
    hydrol_var_config: Mapping[str, float] | None = None,
    refSOC_1d=None,
    use_refSOC_hydrol=False,
    switches: HydrolInitPFT14Switches | None = None,
    nsnow: int = 3,
    min_sechiba=1.0e-8,
) -> HydrolInitPFT14Result:
    """Own the source-ordered PFT14 ``hydrol_init`` scientific state.

    The restart mapping is the explicit replacement for ``restget_p``. A
    missing field follows the corresponding ``setvar_p`` or all-missing
    fallback in lines 2813-3090. Allocation-only arrays are represented by
    shape contracts and are not assigned invented numerical values.

    Fortran provenance: ``hydrol.f90::hydrol_init`` lines 2028-3120 and
    ``hydrol.f90::hydrol_var_init`` lines 3946-4624, called immediately after
    ``hydrol_init`` by ``hydrol_initialize`` lines 659-676.
    """

    switches = HydrolInitPFT14Switches() if switches is None else switches
    if int(switches.soiltype_classif) != 12:
        raise ValueError("PFT14 hydrol_init owner supports the paper USDA SOILTYPE_CLASSIF=12 branch only")
    veget = _as_2d("veget", veget)
    veget_max = _as_2d("veget_max", veget_max)
    soiltile = _as_2d("soiltile", soiltile)
    if veget.shape != veget_max.shape or veget.shape[0] != soiltile.shape[0]:
        raise ValueError("veget, veget_max, and soiltile dimensions must agree")
    npts, nvm = veget.shape
    nstm = soiltile.shape[1]
    nslm = int(jnp.asarray(znh_m).shape[0])
    soil_parameters = _hydrol_init_usda_parameters(hydrol_init_config)
    cold_start = hydrol_cold_start_state(
        veget_max=veget_max,
        soiltile=soiltile,
        pref_soil_veg=pref_soil_veg,
        nslm=nslm,
        nsnow=nsnow,
        frac_nobio=frac_nobio,
        ok_freeze_cwrr=switches.ok_freeze_cwrr,
        ok_explicitsnow=switches.ok_explicitsnow,
        peat_hydro=switches.peat_hydro,
        peat_nodr=switches.peat_nodr,
        tides=switches.tides,
        agri_peat=switches.agri_peat,
        agri_drain=switches.agri_drain,
        min_sechiba=min_sechiba,
    )
    restart = {} if restart_state is None else dict(restart_state)
    allowed_restart = set(cold_start._fields) | {
        "vegstress",
        "humrel",
        "evap_bare_lim",
        "drysoil_frac",
        "snowrho",
        "snowtemp",
        "snowdz",
        "snowheat",
        "snowgrain",
        "tot_watveg_beg",
        "tot_watsoil_beg",
        "snow_beg",
    }
    unknown_restart = set(restart) - allowed_restart
    if unknown_restart:
        raise ValueError(f"unsupported hydrol_init restart field(s): {sorted(unknown_restart)}")
    span_state = {name: getattr(cold_start, name) for name in cold_start._fields if name not in {"provenance", "notes"}}
    span_state.update(restart)
    if "mc" in restart and "mcl" not in restart:
        span_state["mcl"] = span_state["mc"]
    if "zwt_force" in restart:
        zwt_force = _as_2d("zwt_force", span_state["zwt_force"])
        span_state["zforce"] = bool(jnp.any(zwt_force[0, :] <= 2.0))
    restart_inputs = tuple(sorted(restart))

    us = _as_4d("us", span_state["us"])
    if us.shape != (npts, nvm, nstm, nslm):
        raise ValueError("us must have shape (npts,nvm,nstm,nslm)")
    humrelv = jnp.sum(us, axis=3)
    span_state["humrelv"] = humrelv
    span_state["vegstressv"] = humrelv
    pref = np.asarray(pref_soil_veg, dtype=np.int32)
    if pref.shape != (nvm,) or np.any(pref < 1) or np.any(pref > nstm):
        raise ValueError("pref_soil_veg must contain nvm Fortran 1-based soil tile indices")
    selected_stress = jnp.stack([humrelv[:, jv, int(pref[jv]) - 1] for jv in range(nvm)], axis=1)
    if "vegstress" not in restart:
        span_state["vegstress"] = selected_stress
    if "humrel" not in restart:
        span_state["humrel"] = selected_stress
    if "evap_bare_lim" not in restart:
        span_state["evap_bare_lim"] = jnp.sum(
            span_state["evap_bare_lim_ns"] * span_state["vegtot"][:, None] * soiltile,
            axis=1,
        )

    restart_static = hydrol_var_init(
        veget=veget,
        veget_max=veget_max,
        soiltile=soiltile,
        njsc=njsc,
        znh_m=znh_m,
        dnh_m=dnh_m,
        dlh_m=dlh_m,
        humcste=humcste,
        altmax=altmax,
        mc=span_state["mc"],
        mcl=span_state["mcl"],
        water2infilt=span_state["water2infilt"],
        resdist=span_state["resdist"],
        vegtot_old=span_state["vegtot_old"],
        qsintveg=span_state["qsintveg"],
        humrelv=humrelv,
        drysoil_frac=span_state.get("drysoil_frac"),
        refSOC_1d=refSOC_1d,
        use_refSOC_hydrol=use_refSOC_hydrol,
        ok_pc=switches.ok_pc,
        ok_leak=switches.ok_leak,
        peat_hydro=switches.peat_hydro,
        tides=switches.tides,
        agri_peat=switches.agri_peat,
        ok_freeze_cwrr=switches.ok_freeze_cwrr,
        config_values=hydrol_var_config,
        soil_parameters=soil_parameters,
    )
    vegetation_static = hydrol_vegupd_static_state(
        veget=veget,
        veget_max=veget_max,
        soiltile=soiltile,
        vegtot=jnp.sum(veget, axis=1),
        pref_soil_veg=pref_soil_veg,
        min_sechiba=min_sechiba,
    )
    state = dict(span_state)
    state.update(restart_static._asdict())
    state.update(vegetation_static._asdict())
    state["vegstress"] = span_state["vegstress"]
    state["humrel"] = span_state["humrel"]
    state["free_drain_coef"] = span_state["free_drain_coef"]
    state["zwt_force"] = span_state["zwt_force"]
    state["zforce"] = span_state["zforce"]
    state["explicit_snow"] = span_state["explicit_snow"]
    provenance = (
        "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_init lines 2028-2252",
        "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_init lines 2255-2665",
        "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_init lines 2667-3120",
        "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_var_init lines 3946-4624",
        "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_initialize lines 56-88",
    )
    return HydrolInitPFT14Result(
        state=state,
        span_state=span_state,
        cold_start=cold_start,
        restart_static=restart_static,
        vegetation_static=vegetation_static,
        soil_parameters=soil_parameters,
        allocation_only=_hydrol_init_allocation_shapes(npts=npts, nvm=nvm, nstm=nstm, nslm=nslm),
        restart_inputs=restart_inputs,
        process_order=("soil_parameter_selection", "allocation", "restart_read", "setvar_defaults", "vegetation_masks", "stress_fallback", "explicit_snow_initialize", "hydrol_var_init"),
        switches=switches,
        provenance=provenance,
        notes=(
            "restget_p and getin_p are explicit mapping boundaries; no file IO occurs in this owner.",
            "Allocation-only arrays are shape contracts until their source owner assigns a value.",
            "TOPMODEL initialization belongs to hydrol_initialize after hydrol_var_init and is not synthesized here.",
        ),
    )


def hydrol_initialize_pft14_completion(
    *,
    qsintveg,
    humtot,
    snow,
    snow_nobio,
    tot_watveg_beg,
    tot_watsoil_beg,
    snow_beg,
    mx_eau_var,
    znh_m,
    soilmoist,
    val_exp=1.0e20,
    use_refSOC_hydrol=False,
    peat_hydro=True,
    tides=True,
    topm_calcul=False,
    topmodel_new=False,
    check_waterbal=False,
    waterbal_inputs: Mapping[str, object] | None = None,
) -> HydrolInitializeCompletionResult:
    """Complete the fixed PFT14 ``hydrol_initialize`` scientific dispatch.

    Fortran provenance: ``hydrol.f90::hydrol_initialize`` lines 625-688 and
    833-904. The paper path disables refSOC file loading, legacy TOPMODEL and
    new-TOPMODEL file loading. Peat/tide switches are assigned in source order,
    missing ALMA restart inventories are initialized by ``hydrol_alma``, and
    ``itopmax`` follows the one-based loop at lines 893-900. File readers remain
    explicit unsupported boundaries instead of being replaced by constants.
    """

    if bool(use_refSOC_hydrol):
        raise NotImplementedError("hydrol_initialize lines 629-637 require external refSOC restart/file IO")
    if bool(topm_calcul):
        raise NotImplementedError("hydrol_initialize lines 688-866 require external TOPMODEL parameter IO")
    if bool(topmodel_new):
        raise NotImplementedError("hydrol_initialize lines 871-874 require external new-TOPMODEL parameter IO")

    qsintveg = _as_2d("qsintveg", qsintveg)
    humtot = _as_1d("humtot", humtot)
    snow = _as_1d("snow", snow)
    snow_nobio = _as_2d("snow_nobio", snow_nobio)
    tot_watveg_beg = _as_1d("tot_watveg_beg", tot_watveg_beg)
    tot_watsoil_beg = _as_1d("tot_watsoil_beg", tot_watsoil_beg)
    snow_beg = _as_1d("snow_beg", snow_beg)
    mx_eau_var = _as_1d("mx_eau_var", mx_eau_var)
    znh_m = _as_1d("znh_m", znh_m)
    soilmoist = _as_2d("soilmoist", soilmoist)
    _require_same_npts(
        qsintveg,
        humtot,
        snow,
        snow_nobio,
        tot_watveg_beg,
        tot_watsoil_beg,
        snow_beg,
        mx_eau_var,
        soilmoist,
    )

    peat_or_tides = bool(peat_hydro) or bool(tides)
    npts = humtot.shape[0]
    topmodel_state = {
        name: jnp.zeros((npts,), dtype=humtot.dtype)
        for name in ("ZMIN", "ZMAX", "ZMEAN", "ZSTDT", "ZSKEW", "ZZPAS")
    }
    topmodel_state.update(
        {
            name: jnp.zeros((npts, 1000), dtype=humtot.dtype)
            for name in ("ZTAB_FSAT", "ZTAB_WTOP", "ZTAB_FWET", "ZTAB_WTOP_WET")
        }
    )

    needs_alma_init = (
        bool(jnp.all(tot_watveg_beg == val_exp))
        or bool(jnp.all(tot_watsoil_beg == val_exp))
        or bool(jnp.all(snow_beg == val_exp))
    )
    alma = None
    if needs_alma_init:
        alma = hydrol_alma_step(
            qsintveg=qsintveg,
            humtot=humtot,
            snow=snow,
            snow_nobio=snow_nobio,
            tot_watveg_beg=tot_watveg_beg,
            tot_watsoil_beg=tot_watsoil_beg,
            snow_beg=snow_beg,
            mx_eau_var=mx_eau_var,
            lstep_init=True,
        )

    tot_melt = jnp.zeros((npts,), dtype=humtot.dtype)
    water_balance = None
    if bool(check_waterbal):
        if waterbal_inputs is None:
            raise ValueError("check_waterbal requires explicit hydrol_waterbal inputs")
        waterbal_args = dict(waterbal_inputs)
        water_balance = hydrol_waterbal_step(**waterbal_args)

    itopmax = 1
    for jsl, depth in enumerate(np.asarray(znh_m), start=1):
        if depth <= 0.1:
            itopmax = jsl

    return HydrolInitializeCompletionResult(
        peat_nodr=peat_or_tides,
        ok_ru2peat=peat_or_tides,
        ok_wt_ab=peat_or_tides,
        max_wt_ab=100.0,
        topmodel_state=topmodel_state,
        alma=alma,
        water_balance=water_balance,
        tot_melt=tot_melt,
        itopmax=itopmax,
        soilmoist_out=jnp.array(soilmoist, copy=True),
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_initialize lines 625-688",
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_initialize lines 833-904",
        ),
    )


def _hydrol_explicit_snow_zero_checked_arrays(
    precip_snow,
    vevapsno,
    snow,
    snowdz,
    snowrho,
    snowtemp,
    snow_age,
    snow_nobio,
    snow_nobio_age,
    frac_nobio,
    totfrac_nobio,
    zero_celsius,
    snow_density,
    min_sechiba,
):
    active = (
        jnp.any(jnp.abs(precip_snow) > min_sechiba)
        | jnp.any(jnp.abs(vevapsno) > min_sechiba)
        | jnp.any(jnp.abs(snow) > min_sechiba)
        | jnp.any(jnp.abs(snowdz) > min_sechiba)
        | jnp.any(jnp.abs(snow_nobio) > min_sechiba)
        | jnp.any(jnp.abs(frac_nobio) > min_sechiba)
        | jnp.any(jnp.abs(totfrac_nobio) > min_sechiba)
    )
    return (
        active,
        jnp.zeros_like(snow),
        jnp.zeros_like(snowdz),
        jnp.full_like(snowrho, snow_density),
        jnp.full_like(snowtemp, zero_celsius),
        jnp.zeros_like(snowdz),
        jnp.zeros_like(snowdz),
        jnp.zeros_like(snow_age),
        jnp.zeros_like(snow_nobio),
        jnp.zeros_like(snow_nobio_age),
        jnp.zeros_like(snow),
    )


_hydrol_explicit_snow_zero_checked_arrays_jit = jit(_hydrol_explicit_snow_zero_checked_arrays)


def hydrol_explicit_snow_zero_state(
    *,
    precip_snow,
    precip_rain,
    vevapsno,
    snow,
    snowdz,
    snowrho,
    snowtemp,
    snow_age,
    snow_nobio,
    snow_nobio_age,
    frac_nobio,
    totfrac_nobio,
    zero_celsius=273.15,
    snow_density=50.0,
    min_sechiba=1.0e-8,
    use_jit=False,
) -> HydrolSnowState:
    """Return the exact explicit-snow state for the no-snow/no-snowfall path.

    Fortran provenance follows ``hydrol_main`` calling ``explicitsnow_main``
    and the zero-snow branches listed in
    ``HYDROL_EXPLICIT_SNOW_ZERO_STATE_PROVENANCE``. This helper is deliberately
    narrow: it only closes the source-proven no-snow case and raises if any
    snow mass, snowfall, snow sublimation, or nobio snow path is active. Rain
    is allowed only when ``frac_nobio`` is zero, because the Fortran land-snow
    branch does not create snow from rain and the nobio rain branch is inactive.
    """

    precip_snow = _as_1d("precip_snow", precip_snow)
    precip_rain = _as_1d("precip_rain", precip_rain)
    vevapsno = _as_1d("vevapsno", vevapsno)
    snow = _as_1d("snow", snow)
    snowdz = _as_2d("snowdz", snowdz)
    snowrho = _as_2d("snowrho", snowrho)
    snowtemp = _as_2d("snowtemp", snowtemp)
    snow_age = _as_1d("snow_age", snow_age)
    snow_nobio = _as_2d("snow_nobio", snow_nobio)
    snow_nobio_age = _as_2d("snow_nobio_age", snow_nobio_age)
    frac_nobio = _as_2d("frac_nobio", frac_nobio)
    totfrac_nobio = _as_1d("totfrac_nobio", totfrac_nobio)
    _require_same_npts(
        precip_snow,
        precip_rain,
        vevapsno,
        snow,
        snowdz,
        snowrho,
        snowtemp,
        snow_age,
        snow_nobio,
        snow_nobio_age,
        frac_nobio,
        totfrac_nobio,
    )
    if snowrho.shape != snowdz.shape or snowtemp.shape != snowdz.shape:
        raise ValueError("snowdz, snowrho, and snowtemp must share shape (npts, nsnow)")
    if snow_nobio.shape != frac_nobio.shape or snow_nobio_age.shape != snow_nobio.shape:
        raise ValueError("snow_nobio, snow_nobio_age, and frac_nobio must share shape (npts, nnobio)")

    if bool(use_jit):
        (
            active,
            snow_out,
            snowdz_out,
            snowrho_out,
            snowtemp_out,
            snowheat_out,
            snowgrain_out,
            snow_age_out,
            snow_nobio_out,
            snow_nobio_age_out,
            tot_melt_out,
        ) = _hydrol_explicit_snow_zero_checked_arrays_jit(
            precip_snow,
            vevapsno,
            snow,
            snowdz,
            snowrho,
            snowtemp,
            snow_age,
            snow_nobio,
            snow_nobio_age,
            frac_nobio,
            totfrac_nobio,
            zero_celsius,
            snow_density,
            min_sechiba,
        )
        if bool(active):
            return hydrol_explicit_snow_zero_state(
                precip_snow=precip_snow,
                precip_rain=precip_rain,
                vevapsno=vevapsno,
                snow=snow,
                snowdz=snowdz,
                snowrho=snowrho,
                snowtemp=snowtemp,
                snow_age=snow_age,
                snow_nobio=snow_nobio,
                snow_nobio_age=snow_nobio_age,
                frac_nobio=frac_nobio,
                totfrac_nobio=totfrac_nobio,
                zero_celsius=zero_celsius,
                snow_density=snow_density,
                min_sechiba=min_sechiba,
                use_jit=False,
            )
        return HydrolSnowState(
            snow=snow_out,
            snowdz=snowdz_out,
            snowrho=snowrho_out,
            snowtemp=snowtemp_out,
            snowheat=snowheat_out,
            snowgrain=snowgrain_out,
            snow_age=snow_age_out,
            snow_nobio=snow_nobio_out,
            snow_nobio_age=snow_nobio_age_out,
            tot_melt=tot_melt_out,
            provenance=HYDROL_EXPLICIT_SNOW_ZERO_STATE_PROVENANCE,
            notes=(
                "Closed only for the audited no-snow/no-snowfall/no-nobio-snow path.",
                "Rain is permitted on this path only because frac_nobio and totfrac_nobio are zero.",
                "Nonzero explicit-snow physics remains outside this narrow state adapter.",
            ),
        )

    checks = (
        ("precip_snow", precip_snow),
        ("vevapsno", vevapsno),
        ("snow", snow),
        ("snowdz", snowdz),
        ("snow_nobio", snow_nobio),
        ("frac_nobio", frac_nobio),
        ("totfrac_nobio", totfrac_nobio),
    )
    active = [name for name, value in checks if bool(jnp.any(jnp.abs(value) > min_sechiba))]
    if active:
        raise ValueError(
            "hydrol_explicit_snow_zero_state only covers the exact no-snow path; "
            f"active fields: {tuple(active)}"
        )

    zero_1d = jnp.zeros_like(snow)
    zero_nobio = jnp.zeros_like(snow_nobio)
    snowdz_out = jnp.zeros_like(snowdz)
    return HydrolSnowState(
        snow=zero_1d,
        snowdz=snowdz_out,
        snowrho=jnp.full_like(snowrho, snow_density),
        snowtemp=jnp.full_like(snowtemp, zero_celsius),
        snowheat=jnp.zeros_like(snowdz),
        snowgrain=jnp.zeros_like(snowdz),
        snow_age=jnp.zeros_like(snow_age),
        snow_nobio=zero_nobio,
        snow_nobio_age=jnp.zeros_like(snow_nobio_age),
        tot_melt=zero_1d,
        provenance=HYDROL_EXPLICIT_SNOW_ZERO_STATE_PROVENANCE,
        notes=(
            "Closed only for the audited no-snow/no-snowfall/no-nobio-snow path.",
            "Rain is permitted on this path only because frac_nobio and totfrac_nobio are zero.",
            "Nonzero explicit-snow physics remains outside this narrow state adapter.",
        ),
    )


def hydrol_explicit_snow_step(
    *,
    precip_rain,
    precip_snow,
    temp_air,
    pb,
    u,
    v,
    temp_sol_new,
    soilcap,
    pgflux,
    frac_nobio,
    totfrac_nobio,
    gtemp,
    lambda_snow,
    cgrnd_snow,
    dgrnd_snow,
    vevapsno,
    snow_age,
    snow_nobio_age,
    snow_nobio,
    snowrho,
    snowgrain,
    snowdz,
    snowtemp,
    snowheat,
    snow,
    temp_sol_add,
    snowliq,
    subsnownobio,
    grndflux,
    snowmelt,
    soilflxresid,
    dt_sechiba=1800.0,
    one_day=86400.0,
) -> HydrolExplicitSnowStepResult:
    """Run HYDROL's explicit-snow branch with full source-backed inputs.

    Fortran provenance: ``hydrol.f90::hydrol_main`` lines 1177-1197 call
    ``explicitsnow_main``; the snow physics is assembled by
    ``explicitsnow_main_step`` from ``explicitsnow.f90`` lines 108-553.
    """

    main = explicitsnow_main_step(
        precip_rain=precip_rain,
        precip_snow=precip_snow,
        temp_air=temp_air,
        pb=pb,
        u=u,
        v=v,
        temp_sol_new=temp_sol_new,
        soilcap=soilcap,
        pgflux=pgflux,
        frac_nobio=frac_nobio,
        totfrac_nobio=totfrac_nobio,
        gtemp=gtemp,
        lambda_snow=lambda_snow,
        cgrnd_snow=cgrnd_snow,
        dgrnd_snow=dgrnd_snow,
        vevapsno=vevapsno,
        snow_age=snow_age,
        snow_nobio_age=snow_nobio_age,
        snow_nobio=snow_nobio,
        snowrho=snowrho,
        snowgrain=snowgrain,
        snowdz=snowdz,
        snowtemp=snowtemp,
        snowheat=snowheat,
        snow=snow,
        temp_sol_add=temp_sol_add,
        snowliq=snowliq,
        subsnownobio=subsnownobio,
        grndflux=grndflux,
        snowmelt=snowmelt,
        soilflxresid=soilflxresid,
        dt_sechiba=dt_sechiba,
        one_day=one_day,
    )
    snow_state = HydrolSnowState(
        snow=main.snow,
        snowdz=main.snowdz,
        snowrho=main.snowrho,
        snowtemp=main.snowtemp,
        snowheat=main.snowheat,
        snowgrain=main.snowgrain,
        snow_age=main.snow_age,
        snow_nobio=main.snow_nobio,
        snow_nobio_age=main.snow_nobio_age,
        tot_melt=main.tot_melt,
        provenance=EXPLICITSNOW_MAIN_PROVENANCE,
        notes=(
            "HYDROL explicit-snow branch is source-ordered through explicitsnow_main_step.",
            "Additional inout fields are returned on HydrolExplicitSnowStepResult.",
        ),
    )
    return HydrolExplicitSnowStepResult(
        snow_state=snow_state,
        snowliq=main.snowliq,
        vevapsno=main.vevapsno,
        subsnownobio=main.subsnownobio,
        subsinksoil=main.subsinksoil,
        grndflux=main.grndflux,
        snowmelt=main.snowmelt,
        temp_sol_add=main.temp_sol_add,
        snowmelt_from_maxmass=main.snowmelt_from_maxmass,
        melt_mass_by_layer=main.melt_mass_by_layer,
        refreeze_mass_by_layer=main.refreeze_mass_by_layer,
        liquid_percolation_by_interface=main.liquid_percolation_by_interface,
        melt_refreeze_energy=main.melt_refreeze_energy,
        liquid_excess_energy=main.liquid_excess_energy,
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_main lines 1177-1197",
            *EXPLICITSNOW_MAIN_PROVENANCE,
        ),
        notes=(
            "No trace-backed or heuristic snow state is synthesized; all Fortran boundary inputs are explicit.",
        ),
    )


def assemble_hydrol_first_step_precall_payload(
    *,
    enerbil_payload: Mapping[str, object] | None,
    hydrol_restart,
    slowproc_restart,
    pref_soil_veg,
    nstm: int,
    ext_coeff_vegetfrac,
    refSOC_1d=None,
    use_refSOC_hydrol=False,
    precip_rain=None,
    precip_snow=None,
    diffuco_payload: Mapping[str, object] | None = None,
    thermosoil_restart=None,
    throughfall_by_pft=None,
    humcste=None,
    zz_mm=None,
    diaglev_m=None,
    peat_hydro=False,
    ok_dgvm=False,
    ok_explicitsnow=True,
    ok_freeze_cwrr=True,
    ok_thermodynamical_freezing=True,
    fr_dt=2.0,
    froz_frac_corr=1.0,
    smtot_corr=2.0,
    max_froz_hydro=1.0,
    dt_days=None,
    static_payload_template: Mapping[str, object] | None = None,
    use_jit=False,
) -> HydrolFirstStepPrecallAssembly:
    """Assemble exact first-step fields immediately before ``hydrol_main``.

    This helper validates only the first-step boundary. It does not run
    ``hydrol_main`` and it keeps HYDROL-internal process inputs missing unless
    an exact upstream/restart/source-backed value is supplied.
    """

    payload: dict[str, object] = {}
    enerbil_inputs: list[str] = []
    restart_inputs: list[str] = []
    slowproc_inputs: list[str] = []
    driver_inputs: list[str] = []
    source_kernel_inputs: list[str] = []
    static_template = dict(static_payload_template or {})
    static_source_names = tuple(static_template.pop("_source_kernel_inputs", ()))
    static_slowproc_names = tuple(static_template.pop("_slowproc_inputs", ()))
    static_restart_names = tuple(static_template.pop("_restart_inputs", ()))

    def as_runtime_array(value, *, dtype=None):
        return jnp.asarray(value, dtype=dtype) if isinstance(value, jax.core.Tracer) else np.asarray(value, dtype=dtype)

    enerbil = dict(enerbil_payload or {})
    diffuco = dict(diffuco_payload or {})
    boundary = validate_enerbil_to_hydrol_boundary_payload(enerbil)
    for field in boundary.covered_fields:
        _add_payload_field(payload, enerbil_inputs, field, enerbil[field])
    for field in ("soilcap",):
        if field in enerbil:
            _add_payload_field(payload, enerbil_inputs, field, enerbil[field])

    if precip_rain is not None:
        _add_payload_field(payload, driver_inputs, "precip_rain", precip_rain)
    elif "precip_rain" in enerbil:
        _add_payload_field(payload, driver_inputs, "precip_rain", enerbil["precip_rain"])
    if precip_snow is not None:
        _add_payload_field(payload, driver_inputs, "precip_snow", precip_snow)

    for field in ("gtemp", "lambda_snow", "cgrnd_snow", "dgrnd_snow"):
        value = getattr(thermosoil_restart, field, None)
        if value is not None:
            _add_payload_field(payload, restart_inputs, field, value)

    for field in (
        "qsintmax",
        "qsintveg",
        "evap_bare_lim",
        "flood_frac",
        "flood_res",
        "snow",
        "snowdz",
        "snowrho",
        "snowtemp",
        "snow_age",
        "snow_nobio",
        "snow_nobio_age",
        "snowheat",
        "snowgrain",
        "snowliq",
    ):
        if field in diffuco:
            _add_payload_field(payload, source_kernel_inputs, field, diffuco[field])

    if static_template:
        for name, value in static_template.items():
            if name.startswith("_"):
                continue
            payload[name] = value
        source_kernel_inputs.extend(static_source_names)
        slowproc_inputs.extend(static_slowproc_names)
        restart_inputs.extend(static_restart_names)
    else:
        veget = _slowproc_restart_vegetation_for_hydrol(
            slowproc_restart,
            pref_soil_veg=pref_soil_veg,
            nstm=nstm,
            ext_coeff_vegetfrac=ext_coeff_vegetfrac,
            ok_dgvm=ok_dgvm,
        )
        _add_payload_field(payload, source_kernel_inputs, "veget", veget.veget)
        _add_payload_field(payload, source_kernel_inputs, "veget_max", veget.veget_max)
        _add_payload_field(payload, slowproc_inputs, "frac_nobio", slowproc_restart.frac_nobio)
        _add_payload_field(payload, source_kernel_inputs, "vegtot", 1.0 - veget.totfrac_nobio)
        _add_payload_field(payload, source_kernel_inputs, "soiltile", veget.soiltile)
        _add_payload_field(payload, slowproc_inputs, "pref_soil_veg", pref_soil_veg, dtype="int32")
        vegupd = hydrol_vegupd_static_state(
            veget=veget.veget,
            veget_max=veget.veget_max,
            soiltile=veget.soiltile,
            vegtot=1.0 - veget.totfrac_nobio,
            pref_soil_veg=pref_soil_veg,
        )
        if humcste is not None and zz_mm is not None and getattr(hydrol_restart, "njsc", None) is not None:
            _add_payload_field(
                payload,
                source_kernel_inputs,
                "kfact_root",
                hydrol_kfact_root_from_vegupd(
                    vegetmax_soil=vegupd.vegetmax_soil,
                    soiltile=veget.soiltile,
                    pref_soil_veg=pref_soil_veg,
                    humcste=humcste,
                    zz_mm=zz_mm,
                    njsc=hydrol_restart.njsc,
                    peat_hydro=peat_hydro,
                ),
            )

        if throughfall_by_pft is not None:
            _add_payload_field(payload, source_kernel_inputs, "throughfall_by_pft", _np_array(throughfall_by_pft) / 100.0)

    if "snowdz" in payload:
        snowdz = as_runtime_array(payload["snowdz"], dtype=np.float64)
        if snowdz.ndim == 2:
            zeros = jnp.zeros if isinstance(snowdz, jax.core.Tracer) else np.zeros
            _add_payload_field(payload, source_kernel_inputs, "tot_melt", zeros(snowdz.shape[0], dtype=np.float64))

    for field in ("ae_ns", "water2infilt", "free_drain_coef", "zwt_force", "wt_ab", "wt_ab_tide", "evap_bare_lim_ns", "us"):
        value = getattr(hydrol_restart, field, None)
        if value is not None:
            _add_payload_field(payload, restart_inputs, field, value)
    for field in ("snowheat", "snowgrain", "snowliq", "subsnownobio", "grndflux", "snowmelt", "soilflxresid"):
        value = getattr(hydrol_restart, field, None)
        if value is not None:
            _add_payload_field(payload, restart_inputs, field, value)
    if "us" in payload and "humrelv" not in payload:
        us = as_runtime_array(payload["us"], dtype=np.float64)
        _add_payload_field(payload, source_kernel_inputs, "humrelv", us.sum(axis=3))
    if getattr(hydrol_restart, "moistc", None) is not None:
        _add_payload_field(payload, restart_inputs, "mc", _transpose_restart_soil_layers("moistc", hydrol_restart.moistc))
    if getattr(hydrol_restart, "moistcl", None) is not None:
        _add_payload_field(payload, restart_inputs, "mcl", _transpose_restart_soil_layers("moistcl", hydrol_restart.moistcl))
    if getattr(hydrol_restart, "humrel", None) is not None:
        _add_payload_field(payload, restart_inputs, "humrel", hydrol_restart.humrel)
    if (
        bool(ok_freeze_cwrr)
        and thermosoil_restart is not None
        and "mc" in payload
        and "veget_max" in payload
        and getattr(hydrol_restart, "njsc", None) is not None
        and zz_mm is not None
    ):
        snow = payload.get("snow")
        snowdz = payload.get("snowdz")
        profil_froz_kernel = (
            _hydrol_first_step_profil_froz_from_arrays_jit
            if bool(use_jit)
            else hydrol_first_step_profil_froz_from_arrays
        )
        profil_froz, stempdiag, temp_hydro = profil_froz_kernel(
            ptn=thermosoil_restart.ptn,
            veget_max=payload["veget_max"],
            mc=payload["mc"],
            njsc=hydrol_restart.njsc,
            zz_mm=zz_mm,
            diaglev_m=diaglev_m,
            snow=snow,
            snowdz=snowdz,
            ok_explicitsnow=ok_explicitsnow,
            peat_hydro=peat_hydro,
            ok_thermodynamical_freezing=ok_thermodynamical_freezing,
            fr_dt=fr_dt,
            froz_frac_corr=froz_frac_corr,
            smtot_corr=smtot_corr,
            max_froz_hydro=max_froz_hydro,
        )
        _add_payload_field(payload, source_kernel_inputs, "profil_froz", profil_froz)
        _add_payload_field(payload, source_kernel_inputs, "stempdiag", stempdiag)
        _add_payload_field(payload, source_kernel_inputs, "temp_hydro", temp_hydro)
    elif bool(ok_freeze_cwrr) and getattr(hydrol_restart, "profil_froz_hydro_ns", None) is not None:
        _add_payload_field(payload, restart_inputs, "profil_froz", hydrol_restart.profil_froz_hydro_ns)
        if getattr(hydrol_restart, "temp_hydro", None) is not None:
            _add_payload_field(payload, restart_inputs, "temp_hydro", hydrol_restart.temp_hydro)
        if getattr(hydrol_restart, "stempdiag", None) is not None:
            _add_payload_field(payload, restart_inputs, "stempdiag", hydrol_restart.stempdiag)
    elif not bool(ok_freeze_cwrr) and "mc" in payload:
        mc = as_runtime_array(payload["mc"], dtype=np.float64)
        zeros_like = jnp.zeros_like if isinstance(mc, jax.core.Tracer) else np.zeros_like
        _add_payload_field(payload, source_kernel_inputs, "profil_froz", zeros_like(mc))

    if "flood_res" in payload and "subsinksoil" not in payload:
        flood_res = as_runtime_array(payload["flood_res"], dtype=np.float64)
        zeros_like = jnp.zeros_like if isinstance(flood_res, jax.core.Tracer) else np.zeros_like
        _add_payload_field(payload, source_kernel_inputs, "subsinksoil", zeros_like(flood_res))
    if zz_mm is not None and "dz_mm" not in payload:
        _add_payload_field(payload, source_kernel_inputs, "dz_mm", hydrol_layer_thickness_from_zz_mm(zz_mm))
    if zz_mm is not None and "zz_mm" not in payload:
        _add_payload_field(payload, source_kernel_inputs, "zz_mm", zz_mm)
    if dt_days is not None and "dt_days" not in payload:
        _add_payload_field(payload, source_kernel_inputs, "dt_days", dt_days)
    if getattr(hydrol_restart, "njsc", None) is not None:
        if "njsc" not in payload:
            _add_payload_field(payload, restart_inputs, "njsc", hydrol_restart.njsc, dtype="int32")
        tex = as_runtime_array(hydrol_restart.njsc, dtype=np.int32) - 1
        table_array = jnp.asarray if isinstance(tex, jax.core.Tracer) else np.asarray
        if "mcr" not in payload:
            _add_payload_field(payload, source_kernel_inputs, "mcr", table_array(USDA_MCR, dtype=np.float64)[tex])
        if "mcs" not in payload:
            if bool(use_refSOC_hydrol):
                thresholds = hydrol_refsoc_1d_thresholds(
                    njsc=hydrol_restart.njsc,
                    refSOC_1d=refSOC_1d,
                    use_refSOC_hydrol=True,
                )
                _add_payload_field(payload, source_kernel_inputs, "mcs", thresholds.mcs)
                _add_payload_field(payload, source_kernel_inputs, "mcw", thresholds.mcw)
                _add_payload_field(payload, source_kernel_inputs, "mcf", thresholds.mcf)
                _add_payload_field(payload, source_kernel_inputs, "refSOC_1d", refSOC_1d)
            else:
                _add_payload_field(payload, source_kernel_inputs, "mcs", table_array(USDA_MCS, dtype=np.float64)[tex])
        if "mcw" not in payload:
            _add_payload_field(payload, source_kernel_inputs, "mcw", table_array(USDA_MCW, dtype=np.float64)[tex])
        if "mcf" not in payload:
            _add_payload_field(payload, source_kernel_inputs, "mcf", table_array(USDA_MCF, dtype=np.float64)[tex])

    missing = tuple(field for field in HYDROL_FIRST_STEP_PRECALL_REQUIRED_FIELDS if field not in payload)
    return HydrolFirstStepPrecallAssembly(
        payload=payload,
        missing_inputs=missing,
        enerbil_boundary=boundary,
        enerbil_inputs=tuple(dict.fromkeys(enerbil_inputs)),
        restart_inputs=tuple(dict.fromkeys(restart_inputs)),
        slowproc_inputs=tuple(dict.fromkeys(slowproc_inputs)),
        driver_inputs=tuple(dict.fromkeys(driver_inputs)),
        source_kernel_inputs=tuple(dict.fromkeys(source_kernel_inputs)),
        provenance=HYDROL_FIRST_STEP_PRECALL_PROVENANCE,
        notes=(
            "This is a boundary assembly only; it does not execute hydrol_main.",
            "moistc/moistcl restart arrays are transposed from (npts,nstm,nslm) to HYDROL kernel order (npts,nslm,nstm).",
            "throughfall_by_pft is converted from the run.def percent parameter to the HYDROL fraction used in hydrol_canop.",
            "HYDROL-internal fields such as humrelv, us, evap_bare_lim_ns, and profil_froz remain explicit gaps until source-backed kernels provide them.",
        ),
    )


class HydrolSoilLayerDiagnostics(NamedTuple):
    """HYDROL per-tile layer moisture and stress-threshold diagnostics."""

    sm: jnp.ndarray
    smt: jnp.ndarray
    smw: jnp.ndarray
    smf: jnp.ndarray
    sms: jnp.ndarray
    smw_tmp: jnp.ndarray
    smf_tmp: jnp.ndarray
    sms_tmp: jnp.ndarray
    sm_nostress: jnp.ndarray
    soil_wet_ns: jnp.ndarray


class HydrolWaterStressDiagnostics(NamedTuple):
    """HYDROL per-tile root-weighted water-stress diagnostics."""

    us: jnp.ndarray
    humrelv: jnp.ndarray
    vegstressv: jnp.ndarray
    nroot: jnp.ndarray
    soil_wet_ns: jnp.ndarray
    undermcr_increment: jnp.ndarray


class HydrolSoilMoistureAggregates(NamedTuple):
    """HYDROL tile-aggregated moisture exports."""

    soilmoist: jnp.ndarray
    soilmoist_liquid: jnp.ndarray
    mc_layh: jnp.ndarray
    mcl_layh: jnp.ndarray
    mc_layh_s: jnp.ndarray
    mcl_layh_s: jnp.ndarray


class HydrolShumdiagDiagnostics(NamedTuple):
    """HYDROL downstream soil-humidity diagnostics."""

    shumdiag: jnp.ndarray
    shumdiag_perma: jnp.ndarray
    shumdiag_peat: jnp.ndarray
    shumdiag_croppeat: jnp.ndarray
    shumdiag_man: jnp.ndarray


class HydrolStressAggregates(NamedTuple):
    """HYDROL PFT-level stress outputs aggregated across soil tiles."""

    humrel: jnp.ndarray
    vegstress: jnp.ndarray


class HydrolLitterTopDiagnostics(NamedTuple):
    """HYDROL litter and top-grass moisture diagnostics."""

    tmc_litter: jnp.ndarray
    tmc_litter_wilt: jnp.ndarray
    tmc_litter_res: jnp.ndarray
    tmc_litter_field: jnp.ndarray
    tmc_litter_sat: jnp.ndarray
    tmc_litter_awet: jnp.ndarray
    tmc_litter_adry: jnp.ndarray
    soil_wet_litter: jnp.ndarray
    tmc_trampling: jnp.ndarray
    tmc_topgrass: jnp.ndarray
    mc_peat_above: jnp.ndarray
    mc_croppeat_above: jnp.ndarray
    mc_man_above: jnp.ndarray


class HydrolPeatWaterTableDiagnostics(NamedTuple):
    """HYDROL peat/mangrove water-table exports for STOMATE."""

    liqwt_ratio: jnp.ndarray
    wtp: jnp.ndarray
    meanwt: jnp.ndarray | None
    fwet_new: jnp.ndarray | None


class HydrolGridFluxAggregates(NamedTuple):
    """HYDROL grid-cell runoff/drainage/water aggregates."""

    ae_ns: jnp.ndarray
    runoff: jnp.ndarray
    drainage: jnp.ndarray
    humtot: jnp.ndarray
    vevapnu: jnp.ndarray


class HydrolLitterGridDiagnostics(NamedTuple):
    """HYDROL litter conductivity and albedo humidity diagnostics."""

    k_litt: jnp.ndarray
    litterhumdiag: jnp.ndarray
    drysoil_frac: jnp.ndarray


def hydrol_var_init(
    *,
    veget,
    veget_max,
    soiltile,
    njsc,
    znh_m,
    dnh_m,
    dlh_m,
    humcste,
    altmax,
    mc=None,
    mcl=None,
    water2infilt=None,
    resdist=None,
    vegtot_old=None,
    qsintveg=None,
    humrelv=None,
    drysoil_frac=None,
    refSOC_1d=None,
    use_refSOC_hydrol=False,
    ok_pc=False,
    ok_leak=False,
    peat_hydro=False,
    tides=False,
    agri_peat=False,
    ok_freeze_cwrr=True,
    config_values: Mapping[str, float] | None = None,
    soil_parameters: Mapping[str, object] | None = None,
    zmaxh=2.0,
    mx_eau_nobio=150.0,
    val_exp=999999.0,
) -> HydrolVarInitResult:
    """Initialize the PFT/soil state in Fortran statement order.

    ``config_values`` is the explicit IO boundary corresponding to ``getin_p``;
    this numerical routine never reads a file. Missing restart arrays use the
    defaults established by ``hydrol_init`` before this subroutine is called.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::``
    ``hydrol_var_init`` lines 3946-4624. Restart defaults originate in
    ``hydrol_init`` lines 2813-3007. USDA and peat constants originate in
    ``src_parameters/constantes_soil_var.f90`` lines 135-167 and 250-273.
    """

    veget = _as_2d("veget", veget)
    veget_max = _as_2d("veget_max", veget_max)
    soiltile = _as_2d("soiltile", soiltile)
    if veget.shape != veget_max.shape:
        raise ValueError("veget and veget_max must have the same (npts,nvm) shape")
    npts, nvm = veget.shape
    if soiltile.shape[0] != npts:
        raise ValueError("soiltile must have shape (npts,nstm)")
    nstm = int(soiltile.shape[1])
    njsc = _as_1d("njsc", njsc).astype(jnp.int32)
    humcste = _as_1d("humcste", humcste)
    altmax = _as_2d("altmax", altmax)
    znh_m = _as_1d("znh_m", znh_m)
    dnh_m = _as_1d("dnh_m", dnh_m)
    dlh_m = _as_1d("dlh_m", dlh_m)
    if njsc.shape != (npts,) or humcste.shape != (nvm,) or altmax.shape != (npts, nvm):
        raise ValueError("njsc, humcste, and altmax dimensions must match vegetation")
    if not (znh_m.shape == dnh_m.shape == dlh_m.shape):
        raise ValueError("znh_m, dnh_m, and dlh_m must have the same layer axis")
    nslm = int(znh_m.shape[0])
    if nslm < 7:
        raise ValueError("hydrol_var_init lines 4442-4508 require at least seven soil layers")
    if bool(jnp.any(njsc < 1)) or bool(jnp.any(njsc > 12)):
        raise ValueError("PFT14 HYDROL initialization requires USDA njsc in 1..12")
    if (bool(peat_hydro) or bool(tides) or bool(agri_peat)) and nstm < 6:
        raise ValueError("peat/tide initialization references Fortran soil tiles 4, 5, and 6")

    cfg = {} if config_values is None else dict(config_values)
    allowed = {
        "altmax_for_nroot_thresh",
        "CWRR_NKS_N0",
        "CWRR_NKS_POWER",
        "CWRR_AKS_A0",
        "CWRR_AKS_POWER",
        "KFACT_DECAY_RATE",
        "KFACT_STARTING_DEPTH",
        "KFACT_MAX",
    }
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"unsupported hydrol_var_init config key(s): {sorted(unknown)}")
    parameters = {
        "altmax_for_nroot_thresh": float(cfg.get("altmax_for_nroot_thresh", 3.0)),
        "CWRR_NKS_N0": float(cfg.get("CWRR_NKS_N0", 0.95)),
        "CWRR_NKS_POWER": float(cfg.get("CWRR_NKS_POWER", 0.34)),
        "CWRR_AKS_A0": float(cfg.get("CWRR_AKS_A0", 0.00012)),
        "CWRR_AKS_POWER": float(cfg.get("CWRR_AKS_POWER", 0.53)),
        "KFACT_DECAY_RATE": float(cfg.get("KFACT_DECAY_RATE", 2.0)),
        "KFACT_STARTING_DEPTH": float(cfg.get("KFACT_STARTING_DEPTH", 0.3)),
        "KFACT_MAX": float(cfg.get("KFACT_MAX", 10.0)),
    }
    n0 = parameters["CWRR_NKS_N0"]
    nk_rel = parameters["CWRR_NKS_POWER"]
    a0 = parameters["CWRR_AKS_A0"]
    ak_rel = parameters["CWRR_AKS_POWER"]
    f_ks = parameters["KFACT_DECAY_RATE"]
    dp_comp = parameters["KFACT_STARTING_DEPTH"]
    kfact_max = parameters["KFACT_MAX"]
    # Keep the source's line-4000 check: AKS_POWER accidentally tests nk_rel.
    checks = (
        (n0 < 0.0, "CWRR_NKS_N0 must be non-negative"),
        (nk_rel < 0.0, "CWRR_NKS_POWER must be non-negative"),
        (a0 < 0.0, "CWRR_AKS_A0 must be non-negative"),
        (nk_rel < 0.0, "CWRR_AKS_POWER failed the source nk_rel check"),
        (f_ks < 0.0, "KFACT_DECAY_RATE must be non-negative"),
        (dp_comp <= 0.0, "KFACT_STARTING_DEPTH must be positive"),
        (kfact_max < 10.0, "KFACT_MAX must be at least 10"),
    )
    for failed, message in checks:
        if failed:
            raise ValueError(message)

    soil_parameter_defaults = {
        "nvan": USDA_NVAN,
        "avan": USDA_AVAN,
        "mcr": USDA_MCR,
        "mcs_mineral": USDA_MCS,
        "ks": USDA_KS,
        "pcent": USDA_PCENT,
        "mcf_mineral": USDA_MCF,
        "mcw_mineral": USDA_MCW,
        "mc_awet": USDA_MC_AWET,
        "mc_adry": USDA_MC_ADRY,
    }
    supplied_soil_parameters = {} if soil_parameters is None else dict(soil_parameters)
    unknown_soil_parameters = set(supplied_soil_parameters) - set(soil_parameter_defaults)
    if unknown_soil_parameters:
        raise ValueError(f"unsupported hydrol_init soil parameter(s): {sorted(unknown_soil_parameters)}")
    soil = {
        name: jnp.asarray(supplied_soil_parameters.get(name, default), dtype=jnp.float64)
        for name, default in soil_parameter_defaults.items()
    }
    for name, values in soil.items():
        if values.shape != (12,):
            raise ValueError(f"{name} must have shape (12,) for USDA soil classification")

    zz = znh_m * 1000.0
    dz = dnh_m * 1000.0
    dh = dlh_m * 1000.0
    texture = njsc - 1
    mineral_mcs = soil["mcs_mineral"][texture]
    if bool(use_refSOC_hydrol):
        if refSOC_1d is None:
            raise ValueError("use_refSOC_hydrol=True requires explicit refSOC_1d")
        refsoc = _as_1d("refSOC_1d", refSOC_1d)
        if refsoc.shape != (npts,):
            raise ValueError("refSOC_1d must have shape (npts,)")
        zx1 = jnp.minimum(refsoc / SOILC_MAX, 1.0)
        mcs = zx1 * POROS_ORG + (1.0 - zx1) * mineral_mcs
        mcr_point = soil["mcr"][texture]
        alpha = jnp.asarray(USDA_AVAN, dtype=jnp.float64)[texture]
        vg_n = jnp.asarray(USDA_NVAN, dtype=jnp.float64)[texture]
        vg_m = jnp.asarray(USDA_VG_M, dtype=jnp.float64)[texture]
        psi_wp = jnp.asarray(USDA_VG_PSI_WP, dtype=jnp.float64)[texture]
        psi_fc = jnp.asarray(USDA_VG_PSI_FC, dtype=jnp.float64)[texture]
        mcw = mcr_point + (mcs - mcr_point) / (1.0 + (alpha * psi_wp) ** vg_n) ** vg_m
        mcf = mcr_point + (mcs - mcr_point) / (1.0 + (alpha * psi_fc) ** vg_n) ** vg_m
    else:
        mcs = mineral_mcs
        mcw = soil["mcw_mineral"][texture]
        mcf = soil["mcf_mineral"][texture]
    humcste_use = jnp.broadcast_to(humcste[None, :], (npts, nvm))
    nroot = hydrol_nroot_from_humcste(
        humcste=humcste,
        dz_mm=dz,
        zz_mm=zz,
        altmax=altmax,
        ok_pc=ok_pc,
        ok_leak=ok_leak,
    )
    if nroot.shape[0] == 1 and npts != 1:
        nroot = jnp.broadcast_to(nroot, (npts, nvm, nslm))
    mineral_tables = build_mineral_cwrr_tables(
        njsc,
        mcs,
        znh_m,
        mcr=soil["mcr"],
        nvan=soil["nvan"],
        avan=soil["avan"],
        ks=soil["ks"],
        n0=n0,
        nk_rel=nk_rel,
        a0=a0,
        ak_rel=ak_rel,
        f_ks=f_ks,
        dp_comp=dp_comp,
        kfact_max=kfact_max,
    )
    peat_tables = None
    if bool(peat_hydro):
        peat_tables = build_peat_cwrr_tables(
            znh_m,
            n0=n0,
            nk_rel=nk_rel,
            a0=a0,
            ak_rel=ak_rel,
            f_ks=f_ks,
            dp_comp=dp_comp,
            kfact_max=kfact_max,
        )

    shape3 = (npts, nslm, nstm)
    if mc is None:
        mc = jnp.full(shape3, 0.3, dtype=jnp.float64)
    else:
        mc = _as_3d("mc", mc)
        if mc.shape != shape3:
            raise ValueError("mc must have shape (npts,nslm,nstm)")
    if mcl is None:
        mcl = mc.copy()
    else:
        mcl = _as_3d("mcl", mcl)
        if mcl.shape != shape3:
            raise ValueError("mcl must have shape (npts,nslm,nstm)")
    if water2infilt is None:
        water2infilt = jnp.zeros((npts, nstm), dtype=mc.dtype)
    else:
        water2infilt = _as_2d("water2infilt", water2infilt)
    if resdist is None:
        resdist = soiltile
    else:
        resdist = _as_2d("resdist", resdist)
    if water2infilt.shape != (npts, nstm) or resdist.shape != (npts, nstm):
        raise ValueError("water2infilt and resdist must match soiltile")
    if vegtot_old is None:
        vegtot_old = jnp.sum(veget_max, axis=1)
    else:
        vegtot_old = _as_1d("vegtot_old", vegtot_old)
    if vegtot_old.shape != (npts,):
        raise ValueError("vegtot_old must have shape (npts,)")
    if qsintveg is None:
        qsintveg = jnp.zeros((npts, nvm), dtype=mc.dtype)
    else:
        qsintveg = _as_2d("qsintveg", qsintveg)
        if qsintveg.shape != (npts, nvm):
            raise ValueError("qsintveg must match vegetation")

    peat_mask = jnp.asarray([(jst + 1) in (4, 5, 6) for jst in range(nstm)])[None, :]
    point_mcr = soil["mcr"].astype(mc.dtype)[njsc - 1]
    tmcs = jnp.broadcast_to(float(zmaxh) * 1000.0 * mcs[:, None], (npts, nstm))
    tmcr = jnp.broadcast_to(float(zmaxh) * 1000.0 * point_mcr[:, None], (npts, nstm))
    if bool(peat_hydro):
        tmcs = jnp.where(peat_mask, float(zmaxh) * 1000.0 * PEAT_MCS, tmcs)
        tmcr = jnp.where(peat_mask, float(zmaxh) * 1000.0 * PEAT_MCR, tmcr)

    vegtot = jnp.sum(veget, axis=1)
    if bool(peat_hydro):
        peat_tiles = [3]
        if bool(agri_peat):
            peat_tiles.append(4)
        if bool(tides):
            peat_tiles.append(5)
        peat_fraction = jnp.sum(soiltile[:, jnp.asarray(peat_tiles)], axis=1)
        mx_eau_var = float(zmaxh) * 1000.0 * (mcs * (1.0 - peat_fraction) + PEAT_MCS * peat_fraction)
    else:
        mx_eau_var = float(zmaxh) * 1000.0 * mcs
    mx_eau_var = jnp.where(vegtot <= 0.0, float(mx_eau_nobio) * float(zmaxh), mx_eau_var)

    tmc = hydrol_initial_tmc_from_restart_mc(mc=mc, water2infilt=water2infilt, dz_mm=dz)
    def integrate_layers(values):
        out = jnp.zeros_like(values)
        out = out.at[:, 0, :].set(dz[1] * (3.0 * values[:, 0, :] + values[:, 1, :]) / 8.0)
        for jsl in range(1, nslm - 1):
            out = out.at[:, jsl, :].set(
                dz[jsl] * (3.0 * values[:, jsl, :] + values[:, jsl - 1, :]) / 8.0
                + dz[jsl + 1] * (3.0 * values[:, jsl, :] + values[:, jsl + 1, :]) / 8.0
            )
        return out.at[:, -1, :].set(dz[-1] * (3.0 * values[:, -1, :] + values[:, -2, :]) / 8.0)

    integrated_mc = integrate_layers(mc)
    tmc_litter = jnp.sum(integrated_mc[:, :4, :], axis=1)
    tmc_trampling = jnp.sum(integrated_mc[:, :6, :], axis=1)
    threshold_sets = (
        (mcw, PEAT_MCW),
        (soil["mcr"][texture], PEAT_MCR),
        (mcf, PEAT_MCF),
        (mcs, PEAT_MCS),
        (soil["mc_awet"][texture], PEAT_MC_AWET),
        (soil["mc_adry"][texture], PEAT_MC_ADRY),
    )
    litter_depth = dz[1] / 2.0 + jnp.sum((dz[1:4] + dz[2:5]) / 2.0)
    litter_thresholds = []
    for mineral, peat_value in threshold_sets:
        values = jnp.broadcast_to(mineral[:, None] * litter_depth, (npts, nstm))
        if bool(peat_hydro):
            values = jnp.where(peat_mask, float(peat_value) * litter_depth, values)
        litter_thresholds.append(values)
    tmc_topgrass = tmc_trampling[:, 2] / (jnp.sum(dz[:6]) + dz[6] / 2.0)
    if humrelv is None:
        humrelv = jnp.zeros((npts, nvm, nstm), dtype=mc.dtype)
    else:
        humrelv = _as_3d("humrelv", humrelv)
        if humrelv.shape != (npts, nvm, nstm):
            raise ValueError("humrelv must have shape (npts,nvm,nstm)")
        humrelv = humrelv.at[:, 0, :].set(0.0)

    soilmoist = jnp.sum(integrated_mc * resdist[:, None, :], axis=2) * vegtot_old[:, None]
    shumdiag_peat = jnp.zeros((npts, nslm), dtype=mc.dtype)
    shumdiag_croppeat = jnp.zeros_like(shumdiag_peat)
    shumdiag_man = jnp.zeros_like(shumdiag_peat)
    if bool(peat_hydro):
        shumdiag_peat = jnp.clip(integrated_mc[:, :, 3] / dh[None, :], 0.0, 1.0)
    if bool(agri_peat):
        shumdiag_croppeat = jnp.clip(integrated_mc[:, :, 4] / dh[None, :], 0.0, 1.0)
    if bool(peat_hydro) and bool(tides):
        shumdiag_man = jnp.clip(integrated_mc[:, :, 5] / dh[None, :], 0.0, 1.0)
    if bool(peat_hydro):
        shumdiag_perma = soilmoist / dh[None, :]
    else:
        shumdiag_perma = soilmoist / dh[None, :] / mcs[:, None] * mineral_mcs[:, None]
    shumdiag_perma = jnp.clip(shumdiag_perma, 0.0, 1.0)

    if drysoil_frac is None:
        drysoil_frac = jnp.full((npts,), float(val_exp), dtype=mc.dtype)
    else:
        drysoil_frac = _as_1d("drysoil_frac", drysoil_frac)
        if drysoil_frac.shape != (npts,):
            raise ValueError("drysoil_frac must have shape (npts,)")
    if bool(jnp.all(drysoil_frac == float(val_exp))):
        drysoil_frac = jnp.full((npts,), 0.5, dtype=mc.dtype)
    profil_froz_hydro_ns = jnp.zeros_like(mc)

    return HydrolVarInitResult(
        zz=zz,
        dz=dz,
        dh=dh,
        mcs=mcs,
        mcw=mcw,
        mcf=mcf,
        nroot=nroot,
        humcste_use=humcste_use,
        mineral_tables=mineral_tables,
        peat_tables=peat_tables,
        mx_eau_var=mx_eau_var,
        tmcs=tmcs,
        tmcr=tmcr,
        tmc=tmc,
        tmc_litter=tmc_litter,
        tmc_litter_wilt=litter_thresholds[0],
        tmc_litter_res=litter_thresholds[1],
        tmc_litter_field=litter_thresholds[2],
        tmc_litter_sat=litter_thresholds[3],
        tmc_litter_awet=litter_thresholds[4],
        tmc_litter_adry=litter_thresholds[5],
        tmc_trampling=tmc_trampling,
        tmc_topgrass=tmc_topgrass,
        humrelv=humrelv,
        soilmoist=soilmoist,
        shumdiag_perma=shumdiag_perma,
        shumdiag_peat=shumdiag_peat,
        shumdiag_croppeat=shumdiag_croppeat,
        shumdiag_man=shumdiag_man,
        drysoil_frac=drysoil_frac,
        profil_froz_hydro_ns=profil_froz_hydro_ns,
        mc=mc,
        mcl=mcl,
        water2infilt=water2infilt,
        resdist=resdist,
        vegtot_old=vegtot_old,
        qsintveg=qsintveg,
        parameters=parameters,
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_var_init lines 3946-4624",
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_init lines 2813-3007",
            "fortran_source/ORCHIDEE/src_parameters/constantes_soil_var.f90 lines 135-167,250-273",
        ),
    )


def build_mineral_cwrr_tables(
    njsc,
    mcs,
    z_m,
    *,
    mcr=USDA_MCR,
    nvan=USDA_NVAN,
    avan=USDA_AVAN,
    ks=USDA_KS,
    imin=1,
    imax=51,
    n0=0.95,
    nk_rel=0.34,
    a0=0.00012,
    ak_rel=0.53,
    f_ks=2.0,
    dp_comp=0.3,
    kfact_max=10.0,
):
    """Build mineral CWRR linearized hydraulic tables.

    Fortran provenance: `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`,
    subroutine `hydrol_init`, lines 4172-4244; depth-factor setup lines
    4150-4159. Depth grid provenance:
    `fortran_source/ORCHIDEE/src_parameters/vertical_soil.f90`, lines 185-263
    and 331-363. USDA parameter provenance:
    `fortran_source/ORCHIDEE/src_parameters/constantes_soil_var.f90`, lines
    250-273.

    `njsc` uses Fortran 1-based USDA texture codes. Returned table arrays use
    Python offsets, so Fortran index `i` is selected with `i - imin`.
    """

    njsc = jnp.atleast_1d(jnp.asarray(njsc, dtype=jnp.int32))
    mcs = jnp.atleast_1d(jnp.asarray(mcs))
    z_m = jnp.atleast_1d(jnp.asarray(z_m))
    mcr = jnp.asarray(mcr)
    nvan = jnp.asarray(nvan)
    avan = jnp.asarray(avan)
    ks = jnp.asarray(ks)

    if njsc.ndim != 1:
        raise ValueError("njsc must be a scalar or one-dimensional array")
    if mcs.ndim != 1:
        raise ValueError("mcs must be a scalar or one-dimensional array")
    if z_m.ndim != 1:
        raise ValueError("z_m must be a one-dimensional hydrology-layer depth array in meters")
    if njsc.shape[0] != mcs.shape[0]:
        if njsc.shape[0] == 1:
            njsc = jnp.repeat(njsc, mcs.shape[0])
        elif mcs.shape[0] == 1:
            mcs = jnp.repeat(mcs, njsc.shape[0])
        else:
            raise ValueError("njsc and mcs must have matching lengths or one scalar input")
    if imin >= imax:
        raise ValueError("imin must be smaller than imax")

    npts = int(njsc.shape[0])
    nslm = int(z_m.shape[0])
    n_bounds = imax - imin + 1
    n_bins = imax - imin
    texture_index = njsc - 1

    if bool(jnp.any(texture_index < 0)) or bool(jnp.any(texture_index >= mcr.shape[0])):
        raise ValueError("njsc contains a texture code outside the provided parameter arrays")

    kfact = jnp.minimum(
        jnp.maximum(jnp.exp(-f_ks * (z_m[:, None] - dp_comp)), 1.0 / kfact_max),
        1.0,
    )
    kfact = jnp.repeat(kfact, npts, axis=1)
    nfact = kfact**nk_rel
    afact = kfact**ak_rel

    tex_mcr = mcr[texture_index]
    tex_nvan = nvan[texture_index]
    tex_avan = avan[texture_index]
    tex_ks = ks[texture_index]

    nvan_mod = n0 + (tex_nvan[None, :] - n0) * nfact
    avan_mod = a0 + (tex_avan[None, :] - a0) * afact
    van_m = 1.0 - 1.0 / nvan_mod

    bound_steps = jnp.arange(n_bounds)
    mc_lin = tex_mcr[None, :] + bound_steps[:, None] * (mcs[None, :] - tex_mcr[None, :]) / n_bins
    mc_lin = mc_lin.at[0, :].set(tex_mcr)
    mc_lin = mc_lin.at[-1, :].set(mcs)

    k_lin = jnp.zeros((n_bounds, nslm, npts), dtype=mc_lin.dtype)
    a_lin = jnp.zeros((n_bins, nslm, npts), dtype=mc_lin.dtype)
    b_lin = jnp.zeros((n_bins, nslm, npts), dtype=mc_lin.dtype)
    d_lin = jnp.zeros((n_bins, nslm, npts), dtype=mc_lin.dtype)

    for jp in range(npts):
        denom = mcs[jp] - tex_mcr[jp]
        for jsl in range(nslm):
            for bound_offset in range(n_bounds - 1, -1, -1):
                frac = jnp.minimum(1.0, (mc_lin[bound_offset, jp] - tex_mcr[jp]) / denom)
                conductivity = (
                    tex_ks[jp]
                    * kfact[jsl, jp]
                    * frac**0.5
                    * (1.0 - (1.0 - frac ** (1.0 / van_m[jsl, jp])) ** van_m[jsl, jp]) ** 2
                )
                k_lin = k_lin.at[bound_offset, jsl, jp].set(conductivity)

            ji = n_bins - 1
            jiref = ji
            while ji >= 0 and float(k_lin[ji, jsl, jp]) > 1.0e-32:
                jiref = ji
                ji -= 1
            for bin_offset in range(jiref - 1, -1, -1):
                k_lin = k_lin.at[bin_offset, jsl, jp].set(k_lin[bin_offset + 1, jsl, jp] / 10.0)

            frac = jnp.nan
            for bin_offset in range(n_bins):
                slope = (k_lin[bin_offset + 1, jsl, jp] - k_lin[bin_offset, jsl, jp]) / (
                    mc_lin[bin_offset + 1, jp] - mc_lin[bin_offset, jp]
                )
                intercept = k_lin[bin_offset, jsl, jp] - slope * mc_lin[bin_offset, jp]
                a_lin = a_lin.at[bin_offset, jsl, jp].set(slope)
                b_lin = b_lin.at[bin_offset, jsl, jp].set(intercept)

                if bin_offset != 0 and bin_offset != n_bins - 1:
                    frac = jnp.minimum(1.0, (mc_lin[bin_offset, jp] - tex_mcr[jp]) / denom)
                    d_lower = (
                        k_lin[bin_offset, jsl, jp]
                        / (avan_mod[jsl, jp] * van_m[jsl, jp] * nvan_mod[jsl, jp])
                        * (frac ** (-1.0 / van_m[jsl, jp]) / (mc_lin[bin_offset, jp] - tex_mcr[jp]))
                        * (frac ** (-1.0 / van_m[jsl, jp]) - 1.0) ** (-van_m[jsl, jp])
                    )
                    frac = jnp.minimum(1.0, (mc_lin[bin_offset + 1, jp] - tex_mcr[jp]) / denom)
                    d_upper = (
                        k_lin[bin_offset + 1, jsl, jp]
                        / (avan_mod[jsl, jp] * van_m[jsl, jp] * nvan_mod[jsl, jp])
                        * (frac ** (-1.0 / van_m[jsl, jp]) / (mc_lin[bin_offset + 1, jp] - tex_mcr[jp]))
                        * (frac ** (-1.0 / van_m[jsl, jp]) - 1.0) ** (-van_m[jsl, jp])
                    )
                    d_lin = d_lin.at[bin_offset, jsl, jp].set(0.5 * (d_lower + d_upper))
                    if bin_offset + 1 < n_bins:
                        d_lin = d_lin.at[bin_offset + 1, jsl, jp].set(d_upper)
                elif bin_offset == n_bins - 1:
                    d_value = (
                        k_lin[bin_offset, jsl, jp]
                        / (avan_mod[jsl, jp] * van_m[jsl, jp] * nvan_mod[jsl, jp])
                        * (frac ** (-1.0 / van_m[jsl, jp]) / (mc_lin[bin_offset, jp] - tex_mcr[jp]))
                        * (frac ** (-1.0 / van_m[jsl, jp]) - 1.0) ** (-van_m[jsl, jp])
                    )
                    d_lin = d_lin.at[bin_offset, jsl, jp].set(d_value)

            d_lin = d_lin.at[0, jsl, jp].set(d_lin[1, jsl, jp] / 1000.0)
            for bin_offset in range(jiref - 1, -1, -1):
                d_lin = d_lin.at[bin_offset, jsl, jp].set(d_lin[bin_offset + 1, jsl, jp] / 10.0)

    return MineralCWRRTables(
        mc_lin=mc_lin,
        k_lin=k_lin,
        a_lin=a_lin,
        b_lin=b_lin,
        d_lin=d_lin,
        kfact=kfact,
        nfact=nfact,
        afact=afact,
        nvan_mod=nvan_mod,
        avan_mod=avan_mod,
        imin=imin,
        imax=imax,
    )


def build_peat_cwrr_tables(
    z_m,
    *,
    mcr=PEAT_MCR,
    mcs=PEAT_MCS,
    nvan=PEAT_NVAN,
    avan=PEAT_AVAN,
    ks=PEAT_KS,
    imin=1,
    imax=51,
    n0=0.95,
    nk_rel=0.34,
    a0=0.00012,
    ak_rel=0.53,
    f_ks=2.0,
    dp_comp=0.3,
    kfact_max=10.0,
):
    """Build peat CWRR linearized hydraulic tables.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_var_init``, peat depth factors at lines 4163-4168 and
    peat table build at lines 4249-4306. Peat constants come from
    ``src_parameters/constantes_soil_var.f90``, lines 163-167.
    """

    z_m = jnp.atleast_1d(jnp.asarray(z_m))
    if z_m.ndim != 1:
        raise ValueError("z_m must be a one-dimensional hydrology-layer depth array in meters")
    if imin >= imax:
        raise ValueError("imin must be smaller than imax")

    nslm = int(z_m.shape[0])
    n_bounds = imax - imin + 1
    n_bins = imax - imin
    dtype = z_m.dtype
    mcr = jnp.asarray(mcr, dtype=dtype)
    mcs = jnp.asarray(mcs, dtype=dtype)
    nvan = jnp.asarray(nvan, dtype=dtype)
    avan = jnp.asarray(avan, dtype=dtype)
    ks = jnp.asarray(ks, dtype=dtype)

    kfact = jnp.minimum(
        jnp.maximum(jnp.exp(-f_ks * (z_m - dp_comp)), 1.0 / kfact_max),
        1.0,
    )
    nfact = kfact**nk_rel
    afact = kfact**ak_rel

    nvan_mod = n0 + (nvan - n0) * nfact
    avan_mod = a0 + (avan - a0) * afact
    van_m = 1.0 - 1.0 / nvan_mod

    bound_steps = jnp.arange(n_bounds)
    mc_lin = mcr + bound_steps * (mcs - mcr) / n_bins
    mc_lin = mc_lin.at[0].set(mcr)
    mc_lin = mc_lin.at[-1].set(mcs)

    k_lin = jnp.zeros((n_bounds, nslm), dtype=dtype)
    a_lin = jnp.zeros((n_bins, nslm), dtype=dtype)
    b_lin = jnp.zeros((n_bins, nslm), dtype=dtype)
    d_lin = jnp.zeros((n_bins, nslm), dtype=dtype)

    denom = mcs - mcr
    for jsl in range(nslm):
        for bound_offset in range(n_bounds - 1, -1, -1):
            frac = jnp.minimum(1.0, (mc_lin[bound_offset] - mcr) / denom)
            conductivity = (
                ks
                * kfact[jsl]
                * frac**0.5
                * (1.0 - (1.0 - frac ** (1.0 / van_m[jsl])) ** van_m[jsl]) ** 2
            )
            k_lin = k_lin.at[bound_offset, jsl].set(conductivity)

        ji = n_bins - 1
        jiref = ji
        while ji >= 0 and float(k_lin[ji, jsl]) > 1.0e-32:
            jiref = ji
            ji -= 1
        for bin_offset in range(jiref - 1, -1, -1):
            k_lin = k_lin.at[bin_offset, jsl].set(k_lin[bin_offset + 1, jsl] / 10.0)

        frac = jnp.nan
        for bin_offset in range(n_bins):
            slope = (k_lin[bin_offset + 1, jsl] - k_lin[bin_offset, jsl]) / (
                mc_lin[bin_offset + 1] - mc_lin[bin_offset]
            )
            intercept = k_lin[bin_offset, jsl] - slope * mc_lin[bin_offset]
            a_lin = a_lin.at[bin_offset, jsl].set(slope)
            b_lin = b_lin.at[bin_offset, jsl].set(intercept)

            if bin_offset != 0 and bin_offset != n_bins - 1:
                frac = jnp.minimum(1.0, (mc_lin[bin_offset] - mcr) / denom)
                d_lower = (
                    k_lin[bin_offset, jsl]
                    / (avan_mod[jsl] * van_m[jsl] * nvan_mod[jsl])
                    * (frac ** (-1.0 / van_m[jsl]) / (mc_lin[bin_offset] - mcr))
                    * (frac ** (-1.0 / van_m[jsl]) - 1.0) ** (-van_m[jsl])
                )
                frac = jnp.minimum(1.0, (mc_lin[bin_offset + 1] - mcr) / denom)
                d_upper = (
                    k_lin[bin_offset + 1, jsl]
                    / (avan_mod[jsl] * van_m[jsl] * nvan_mod[jsl])
                    * (frac ** (-1.0 / van_m[jsl]) / (mc_lin[bin_offset + 1] - mcr))
                    * (frac ** (-1.0 / van_m[jsl]) - 1.0) ** (-van_m[jsl])
                )
                d_lin = d_lin.at[bin_offset, jsl].set(0.5 * (d_lower + d_upper))
                if bin_offset + 1 < n_bins:
                    d_lin = d_lin.at[bin_offset + 1, jsl].set(d_upper)
            elif bin_offset == n_bins - 1:
                d_value = (
                    k_lin[bin_offset, jsl]
                    / (avan_mod[jsl] * van_m[jsl] * nvan_mod[jsl])
                    * (frac ** (-1.0 / van_m[jsl]) / (mc_lin[bin_offset] - mcr))
                    * (frac ** (-1.0 / van_m[jsl]) - 1.0) ** (-van_m[jsl])
                )
                d_lin = d_lin.at[bin_offset, jsl].set(d_value)

        d_lin = d_lin.at[0, jsl].set(d_lin[1, jsl] / 1000.0)
        for bin_offset in range(jiref - 1, -1, -1):
            d_lin = d_lin.at[bin_offset, jsl].set(d_lin[bin_offset + 1, jsl] / 10.0)

    return PeatCWRRTables(
        mc_lin=mc_lin,
        k_lin=k_lin,
        a_lin=a_lin,
        b_lin=b_lin,
        d_lin=d_lin,
        kfact=kfact,
        nfact=nfact,
        afact=afact,
        nvan_mod=nvan_mod,
        avan_mod=avan_mod,
        mcr=float(mcr),
        mcs=float(mcs),
        ks=float(ks),
        imin=imin,
        imax=imax,
    )


def _array_cache_key(value) -> tuple[str, tuple[int, ...], tuple[float | int, ...]]:
    array = np.asarray(value)
    return (str(array.dtype), tuple(array.shape), tuple(array.ravel().tolist()))


@lru_cache(maxsize=32)
def _cached_mineral_cwrr_tables(njsc_key, mcs_key, z_m_key, ks_key):
    _, njsc_shape, njsc_values = njsc_key
    _, mcs_shape, mcs_values = mcs_key
    _, z_m_shape, z_m_values = z_m_key
    _, ks_shape, ks_values = ks_key
    njsc = np.asarray(njsc_values, dtype=np.int32).reshape(njsc_shape)
    mcs = np.asarray(mcs_values, dtype=np.float64).reshape(mcs_shape)
    z_m = np.asarray(z_m_values, dtype=np.float64).reshape(z_m_shape)
    ks = np.asarray(ks_values, dtype=np.float64).reshape(ks_shape)
    return build_mineral_cwrr_tables(njsc=njsc, mcs=mcs, z_m=z_m, ks=ks)


@lru_cache(maxsize=8)
def _cached_peat_cwrr_tables(z_m_key):
    _, z_m_shape, z_m_values = z_m_key
    z_m = np.asarray(z_m_values, dtype=np.float64).reshape(z_m_shape)
    return build_peat_cwrr_tables(z_m)


class HydrolRuntimeStaticTables(NamedTuple):
    """CWRR coefficient tables bound outside the half-hour transition."""

    mineral: MineralCWRRTables
    peat: PeatCWRRTables | None


def hydrol_runtime_static_tables_from_payload(
    payload: Mapping[str, object],
    *,
    ks=USDA_KS,
    peat_hydro=False,
) -> HydrolRuntimeStaticTables:
    """Build immutable HYDROL tables from source-backed day-static inputs."""

    njsc = np.asarray(payload["njsc"], dtype=np.int32)
    mcs = np.asarray(payload["mcs"], dtype=np.float64)
    zz_mm = np.asarray(payload["zz_mm"], dtype=np.float64)
    ks_array = np.asarray(ks, dtype=np.float64)
    mineral = _cached_mineral_cwrr_tables(
        _array_cache_key(njsc),
        _array_cache_key(mcs),
        _array_cache_key(zz_mm / 1000.0),
        _array_cache_key(ks_array),
    )
    peat = _cached_peat_cwrr_tables(_array_cache_key(zz_mm / 1000.0)) if bool(peat_hydro) else None
    return HydrolRuntimeStaticTables(mineral=mineral, peat=peat)


def select_mineral_cwrr_bin(
    mc,
    profil_froz,
    mcr,
    mcs,
    *,
    imin=1,
    imax=51,
):
    """Select the Fortran mineral CWRR bin for liquid water content.

    Fortran provenance: `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`,
    subroutine `hydrol_soil_coef`, lines 8248-8280. This implements the
    mineral branch `mc_used` and `i = MAX(imin, MIN(imax-1, INT(...)))`
    calculation. `bin_i` is the Fortran 1-based table/bin index; `bin_offset`
    is the Python offset into arrays returned by `build_mineral_cwrr_tables`.
    """

    mc = jnp.asarray(mc)
    profil_froz = jnp.asarray(profil_froz)
    mcr = jnp.asarray(mcr)
    mcs = jnp.asarray(mcs)

    x = 1.0 - profil_froz
    mc_used = mcr + x * jnp.maximum(mc - mcr, 0.0)
    bin_real = imin + (imax - imin) * (mc_used - mcr) / (mcs - mcr)
    bin_i = jnp.maximum(imin, jnp.minimum(imax - 1, bin_real.astype(jnp.int32)))
    return MineralCWRRBinSelection(
        mc_used=mc_used,
        bin_i=bin_i,
        bin_offset=bin_i - imin,
    )


def hydrol_soil_coef_mineral_from_selected_coefficients(
    mc,
    profil_froz,
    kfact_root,
    coefficients: SelectedMineralHydrolCoefficients,
    *,
    mcr=0.0,
):
    """Apply the mineral `hydrol_soil_coef` active-branch algebra.

    Fortran provenance: `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`,
    subroutine `hydrol_soil_coef`, lines 8248-8280. This trace-driven kernel
    assumes the mineral CWRR table bin and coefficients have already been
    selected. It is not a CWRR table builder.
    """

    mc = jnp.asarray(mc)
    profil_froz = jnp.asarray(profil_froz)
    kfact_root = jnp.asarray(kfact_root)
    a_raw = jnp.asarray(coefficients.a_raw)
    b_raw = jnp.asarray(coefficients.b_raw)
    d_raw = jnp.asarray(coefficients.d_raw)
    k_floor_raw = jnp.asarray(coefficients.k_floor_raw)
    mcr = jnp.asarray(mcr)

    x = 1.0 - profil_froz
    mc_used = mcr + x * jnp.maximum(mc - mcr, 0.0)
    k_eval = a_raw * mc_used + b_raw

    return HydrolSoilCoefResult(
        a=a_raw * kfact_root,
        b=b_raw * kfact_root,
        d=d_raw * kfact_root,
        k=jnp.maximum(k_floor_raw, k_eval),
        mc_used=mc_used,
        k_eval=k_eval,
    )


hydrol_soil_coef_mineral_trace = hydrol_soil_coef_mineral_from_selected_coefficients


def hydrol_soil_coef_mineral_from_tables(
    mc,
    profil_froz,
    kfact_root,
    tables: MineralCWRRTables,
    mcr,
    mcs,
    *,
    layer_index,
    point_index=0,
):
    """Apply mineral `hydrol_soil_coef` using source-built CWRR tables.

    Fortran provenance: `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`,
    subroutine `hydrol_soil_coef`, lines 8248-8280. The caller supplies the
    hydrology layer and grid-point table selectors explicitly; case-specific
    choices such as bottom-layer diagnostics belong outside this generic
    kernel.
    """

    selection = select_mineral_cwrr_bin(
        mc=mc,
        profil_froz=profil_froz,
        mcr=mcr,
        mcs=mcs,
        imin=tables.imin,
        imax=tables.imax,
    )
    bin_offset = selection.bin_offset

    layer_index = jnp.asarray(layer_index)
    point_index = jnp.asarray(point_index)
    a_raw = tables.a_lin[bin_offset, layer_index, point_index]
    b_raw = tables.b_lin[bin_offset, layer_index, point_index]
    d_raw = tables.d_lin[bin_offset, layer_index, point_index]
    k_floor_raw = tables.k_lin[tables.imin + 1 - tables.imin, layer_index, point_index]

    selected = hydrol_soil_coef_mineral_from_selected_coefficients(
        mc=mc,
        profil_froz=profil_froz,
        kfact_root=kfact_root,
        coefficients=SelectedMineralHydrolCoefficients(
            a_raw=a_raw,
            b_raw=b_raw,
            d_raw=d_raw,
            k_floor_raw=k_floor_raw,
        ),
        mcr=mcr,
    )

    return HydrolSoilCoefTableResult(
        a=selected.a,
        b=selected.b,
        d=selected.d,
        k=selected.k,
        mc_used=selection.mc_used,
        bin_i=selection.bin_i,
        bin_offset=bin_offset,
        k_eval=selected.k_eval,
        a_raw=a_raw,
        b_raw=b_raw,
        d_raw=d_raw,
        k_floor_raw=jnp.broadcast_to(k_floor_raw, selected.k.shape),
    )


def hydrol_soil_coef_mineral_profile_from_tables(
    mc,
    profil_froz,
    kfact_root,
    tables: MineralCWRRTables,
    mcr,
    mcs,
    *,
    point_index=None,
    ok_freeze_cwrr=True,
):
    """Apply mineral ``hydrol_soil_coef`` to a full grid/layer profile.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil_coef``, lines 8248-8280 for the freezing mineral
    branch and lines 8290-8307 for the non-freezing mineral branch. Peat table
    routing is intentionally excluded; callers must use a peat-specific source
    implementation for PFT14 peat tiles.
    """

    mc = _as_2d("mc", mc)
    profil_froz = _as_2d("profil_froz", profil_froz)
    kfact_root = _as_2d("kfact_root", kfact_root)
    if not (mc.shape == profil_froz.shape == kfact_root.shape):
        raise ValueError("mc, profil_froz, and kfact_root must have matching shape (npts, nslm)")

    npts, nslm = mc.shape
    if int(tables.a_lin.shape[1]) != nslm:
        raise ValueError("tables layer axis must match mc.shape[1]")

    mcr = _as_1d("mcr", mcr)
    mcs = _as_1d("mcs", mcs)
    if mcr.shape[0] == 1:
        mcr = jnp.repeat(mcr, npts)
    if mcs.shape[0] == 1:
        mcs = jnp.repeat(mcs, npts)
    if mcr.shape[0] != npts or mcs.shape[0] != npts:
        raise ValueError("mcr and mcs must be scalar or have shape (npts,)")

    if point_index is None:
        point_index = jnp.arange(npts, dtype=jnp.int32)
    else:
        point_index_np = np.asarray(point_index, dtype=np.int32).reshape(-1)
        if point_index_np.shape[0] == 1:
            point_index_np = np.repeat(point_index_np, npts)
        if point_index_np.shape[0] != npts:
            raise ValueError("point_index must be scalar or have shape (npts,)")
        if bool(np.any(point_index_np < 0)) or bool(np.any(point_index_np >= tables.a_lin.shape[2])):
            raise ValueError("point_index selects outside the mineral CWRR table point axis")
        point_index = jnp.asarray(point_index_np, dtype=jnp.int32)

    mcr_col = mcr[:, None]
    mcs_col = mcs[:, None]
    if ok_freeze_cwrr:
        x = 1.0 - profil_froz
        mc_used = mcr_col + x * jnp.maximum(mc - mcr_col, 0.0)
        mc_for_k = mc_used
        bin_real = tables.imin + (tables.imax - tables.imin) * (mc_used - mcr_col) / (mcs_col - mcr_col)
    else:
        mc_ratio = jnp.maximum(mc - mcr_col, 0.0) / (mcs_col - mcr_col)
        mc_used = mc
        mc_for_k = mc
        bin_real = tables.imin + (tables.imax - tables.imin) * mc_ratio

    bin_i = jnp.maximum(tables.imin, jnp.minimum(tables.imax - 1, bin_real.astype(jnp.int32)))
    bin_offset = bin_i - tables.imin
    layer_index = jnp.broadcast_to(jnp.arange(nslm, dtype=jnp.int32)[None, :], (npts, nslm))
    point_index_2d = jnp.broadcast_to(point_index[:, None], (npts, nslm))

    a_raw = tables.a_lin[bin_offset, layer_index, point_index_2d]
    b_raw = tables.b_lin[bin_offset, layer_index, point_index_2d]
    d_raw = tables.d_lin[bin_offset, layer_index, point_index_2d]
    k_floor_raw = tables.k_lin[tables.imin + 1 - tables.imin, layer_index, point_index_2d]
    k_eval = a_raw * mc_for_k + b_raw

    return HydrolSoilCoefTableResult(
        a=a_raw * kfact_root,
        b=b_raw * kfact_root,
        d=d_raw * kfact_root,
        k=jnp.maximum(k_floor_raw, k_eval),
        mc_used=mc_used,
        bin_i=bin_i,
        bin_offset=bin_offset,
        k_eval=k_eval,
        a_raw=a_raw,
        b_raw=b_raw,
        d_raw=d_raw,
        k_floor_raw=k_floor_raw,
    )


def hydrol_soil_coef_peat_profile_from_tables(
    mc,
    profil_froz,
    kfact_root,
    tables: PeatCWRRTables,
    *,
    ok_freeze_cwrr=True,
):
    """Apply peat ``hydrol_soil_coef`` to a full grid/layer profile.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil_coef``, peat freezing branch lines 8259-8266 and
    peat non-freezing branch lines 8290-8297. The caller is responsible for
    only invoking this for active peat tiles ``ins`` 4, 5, or 6 when
    ``peat_hydro`` is true.
    """

    mc = _as_2d("mc", mc)
    profil_froz = _as_2d("profil_froz", profil_froz)
    kfact_root = _as_2d("kfact_root", kfact_root)
    if not (mc.shape == profil_froz.shape == kfact_root.shape):
        raise ValueError("mc, profil_froz, and kfact_root must have matching shape (npts, nslm)")

    npts, nslm = mc.shape
    if int(tables.a_lin.shape[1]) != nslm:
        raise ValueError("tables layer axis must match mc.shape[1]")

    mcr = jnp.asarray(tables.mcr, dtype=mc.dtype)
    mcs = jnp.asarray(tables.mcs, dtype=mc.dtype)
    if ok_freeze_cwrr:
        x = 1.0 - profil_froz
        mc_used = mcr + x * jnp.maximum(mc - mcr, 0.0)
        mc_for_k = mc_used
        bin_real = tables.imin + (tables.imax - tables.imin) * (mc_used - mcr) / (mcs - mcr)
    else:
        mc_ratio = jnp.maximum(mc - mcr, 0.0) / (mcs - mcr)
        mc_used = mc
        mc_for_k = mc
        bin_real = tables.imin + (tables.imax - tables.imin) * mc_ratio

    bin_i = jnp.maximum(tables.imin, jnp.minimum(tables.imax - 1, bin_real.astype(jnp.int32)))
    bin_offset = bin_i - tables.imin
    layer_index = jnp.broadcast_to(jnp.arange(nslm, dtype=jnp.int32)[None, :], (npts, nslm))

    a_raw = tables.a_lin[bin_offset, layer_index]
    b_raw = tables.b_lin[bin_offset, layer_index]
    d_raw = tables.d_lin[bin_offset, layer_index]
    k_floor_raw = tables.k_lin[tables.imin + 1 - tables.imin, layer_index]
    k_eval = a_raw * mc_for_k + b_raw

    return HydrolSoilCoefTableResult(
        a=a_raw * kfact_root,
        b=b_raw * kfact_root,
        d=d_raw * kfact_root,
        k=jnp.maximum(k_floor_raw, k_eval),
        mc_used=mc_used,
        bin_i=bin_i,
        bin_offset=bin_offset,
        k_eval=k_eval,
        a_raw=a_raw,
        b_raw=b_raw,
        d_raw=d_raw,
        k_floor_raw=k_floor_raw,
    )


def hydrol_soil_setup_coefficients(
    a,
    d,
    dz_mm,
    dt_days,
    free_drain_coef,
    *,
    w_time=1.0,
):
    """Compute HYDROL tridiagonal setup coefficients.

    Fortran provenance: `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`,
    subroutine `hydrol_soil_setup`, lines 8480-8568. Layer geometry source:
    `hydrol_init`, lines 4067-4069 (`dz=dnh*1000`), with `dnh` from
    `fortran_source/ORCHIDEE/src_parameters/vertical_soil.f90`, lines 331-363.
    `w_time` provenance: `constantes_soil_var.f90`, line 350.
    """

    a = jnp.asarray(a)
    d = jnp.asarray(d)
    dz_mm = jnp.asarray(dz_mm)
    dt_days = jnp.asarray(dt_days)
    free_drain_coef = jnp.asarray(free_drain_coef)

    if a.ndim == 1:
        a = a[None, :]
    if d.ndim == 1:
        d = d[None, :]
    if a.ndim != 2 or d.ndim != 2:
        raise ValueError("a and d must have shape (npts, nslm) or (nslm,)")
    if a.shape != d.shape:
        raise ValueError("a and d must have matching shapes")
    if dz_mm.ndim != 1:
        raise ValueError("dz_mm must be a one-dimensional hydrology internode-distance array in mm")
    if dz_mm.shape[0] != a.shape[1]:
        raise ValueError("dz_mm length must match the hydrology layer axis")

    npts, nslm = a.shape
    free_drain_coef = jnp.broadcast_to(free_drain_coef, (npts,))
    temp3 = w_time * dt_days / 2.0
    temp4 = (1.0 - w_time) * dt_days / 2.0

    e = jnp.zeros_like(a)
    f = jnp.zeros_like(a)
    g1 = jnp.zeros_like(a)
    ep = jnp.zeros_like(a)
    fp = jnp.zeros_like(a)
    gp = jnp.zeros_like(a)

    f = f.at[:, 0].set(3.0 * dz_mm[1] / 8.0 + temp3 * ((d[:, 0] + d[:, 1]) / dz_mm[1] + a[:, 0]))
    g1 = g1.at[:, 0].set(dz_mm[1] / 8.0 - temp3 * ((d[:, 0] + d[:, 1]) / dz_mm[1] - a[:, 1]))
    fp = fp.at[:, 0].set(3.0 * dz_mm[1] / 8.0 - temp4 * ((d[:, 0] + d[:, 1]) / dz_mm[1] + a[:, 0]))
    gp = gp.at[:, 0].set(dz_mm[1] / 8.0 + temp4 * ((d[:, 0] + d[:, 1]) / dz_mm[1] - a[:, 1]))

    for jsl in range(1, nslm - 1):
        e = e.at[:, jsl].set(
            dz_mm[jsl] / 8.0 - temp3 * ((d[:, jsl] + d[:, jsl - 1]) / dz_mm[jsl] + a[:, jsl - 1])
        )
        f = f.at[:, jsl].set(
            3.0 * (dz_mm[jsl] + dz_mm[jsl + 1]) / 8.0
            + temp3
            * (
                (d[:, jsl] + d[:, jsl - 1]) / dz_mm[jsl]
                + (d[:, jsl] + d[:, jsl + 1]) / dz_mm[jsl + 1]
            )
        )
        g1 = g1.at[:, jsl].set(
            dz_mm[jsl + 1] / 8.0
            - temp3 * ((d[:, jsl] + d[:, jsl + 1]) / dz_mm[jsl + 1] - a[:, jsl + 1])
        )
        ep = ep.at[:, jsl].set(
            dz_mm[jsl] / 8.0 + temp4 * ((d[:, jsl] + d[:, jsl - 1]) / dz_mm[jsl] + a[:, jsl - 1])
        )
        fp = fp.at[:, jsl].set(
            3.0 * (dz_mm[jsl] + dz_mm[jsl + 1]) / 8.0
            - temp4
            * (
                (d[:, jsl] + d[:, jsl - 1]) / dz_mm[jsl]
                + (d[:, jsl] + d[:, jsl + 1]) / dz_mm[jsl + 1]
            )
        )
        gp = gp.at[:, jsl].set(
            dz_mm[jsl + 1] / 8.0
            + temp4 * ((d[:, jsl] + d[:, jsl + 1]) / dz_mm[jsl + 1] - a[:, jsl + 1])
        )

    last = nslm - 1
    e = e.at[:, last].set(
        dz_mm[last] / 8.0 - temp3 * ((d[:, last] + d[:, last - 1]) / dz_mm[last] + a[:, last - 1])
    )
    f = f.at[:, last].set(
        3.0 * dz_mm[last] / 8.0
        + temp3
        * (
            (d[:, last] + d[:, last - 1]) / dz_mm[last]
            - a[:, last] * (1.0 - 2.0 * free_drain_coef)
        )
    )
    ep = ep.at[:, last].set(
        dz_mm[last] / 8.0 + temp4 * ((d[:, last] + d[:, last - 1]) / dz_mm[last] + a[:, last - 1])
    )
    fp = fp.at[:, last].set(
        3.0 * dz_mm[last] / 8.0
        - temp4
        * (
            (d[:, last] + d[:, last - 1]) / dz_mm[last]
            - a[:, last] * (1.0 - 2.0 * free_drain_coef)
        )
    )

    return HydrolSoilSetupResult(e=e, f=f, g1=g1, ep=ep, fp=fp, gp=gp)


def hydrol_soil_rhs_main(
    setup: HydrolSoilSetupResult,
    mcl,
    b,
    dt_days,
    free_drain_coef,
    *,
    flux_top=0.0,
    rootsink=0.0,
):
    """Build the main HYDROL diffusion RHS and left-hand coefficients.

    Fortran provenance: `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`,
    subroutine `hydrol_soil`, lines 5953-5991. It maps `e/f/g1` into
    `tmat(:,:,1:3)` and constructs `rhs` from `ep/fp/gp`, `mcl`, `b`,
    `flux_top`, `rootsink`, `free_drain_coef`, and `dt_sechiba/one_day`.
    It does not solve the system or update `mcl`.
    """

    mcl = jnp.asarray(mcl)
    b = jnp.asarray(b)
    if mcl.ndim == 1:
        mcl = mcl[None, :]
    if b.ndim == 1:
        b = b[None, :]
    if mcl.shape != b.shape:
        raise ValueError("mcl and b must have matching shape (npts, nslm) or (nslm,)")

    npts, nslm = mcl.shape
    flux_top = jnp.broadcast_to(jnp.asarray(flux_top), (npts,))
    rootsink = jnp.asarray(rootsink)
    if rootsink.ndim == 0:
        rootsink = jnp.broadcast_to(rootsink, (npts, nslm))
    elif rootsink.ndim == 1:
        rootsink = rootsink[None, :]
    if rootsink.shape != mcl.shape:
        raise ValueError("rootsink must be scalar or match mcl shape")
    free_drain_coef = jnp.broadcast_to(jnp.asarray(free_drain_coef), (npts,))
    dt_days = jnp.asarray(dt_days)

    rhs = jnp.zeros_like(mcl)
    rhs = rhs.at[:, 0].set(
        setup.fp[:, 0] * mcl[:, 0]
        + setup.gp[:, 0] * mcl[:, 1]
        - flux_top
        - (b[:, 0] + b[:, 1]) * dt_days / 2.0
        - rootsink[:, 0]
    )
    for jsl in range(1, nslm - 1):
        rhs = rhs.at[:, jsl].set(
            setup.ep[:, jsl] * mcl[:, jsl - 1]
            + setup.fp[:, jsl] * mcl[:, jsl]
            + setup.gp[:, jsl] * mcl[:, jsl + 1]
            + (b[:, jsl - 1] - b[:, jsl + 1]) * dt_days / 2.0
            - rootsink[:, jsl]
        )

    last = nslm - 1
    rhs = rhs.at[:, last].set(
        setup.ep[:, last] * mcl[:, last - 1]
        + setup.fp[:, last] * mcl[:, last]
        + (b[:, last - 1] + b[:, last] * (1.0 - 2.0 * free_drain_coef)) * dt_days / 2.0
        - rootsink[:, last]
    )

    return HydrolSoilRHSResult(
        rhs=rhs,
        e=setup.e,
        f=setup.f,
        g1=setup.g1,
    )


def hydrol_soil_residual_boundary_rhs(
    saved: HydrolSoilRHSResult,
    mcr,
):
    """Apply the alternate top residual-moisture boundary equation.

    Fortran provenance: `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`,
    subroutine `hydrol_soil`, lines 7018-7044. The caller is responsible for
    deciding `resolv` from lines 7010-7017; this function only restores the
    saved equations and imposes `tmat(:,1,2)=1`, `tmat(:,1,3)=0`, `rhs(:,1)=mcr`.
    """

    rhs = jnp.asarray(saved.rhs)
    e = jnp.asarray(saved.e)
    f = jnp.asarray(saved.f)
    g1 = jnp.asarray(saved.g1)
    mcr = jnp.broadcast_to(jnp.asarray(mcr), (rhs.shape[0],))

    f = f.at[:, 0].set(1.0)
    g1 = g1.at[:, 0].set(0.0)
    rhs = rhs.at[:, 0].set(mcr)
    return HydrolSoilRHSResult(rhs=rhs, e=e, f=f, g1=g1)


def hydrol_soil_tridiag_solve(
    e,
    f,
    g1,
    rhs,
    *,
    resolv=True,
    initial_mcl=None,
):
    """Solve HYDROL's tridiagonal system with the Fortran sweep order.

    Fortran provenance: `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`,
    subroutine `hydrol_soil_tridiag`, lines 8147-8198. It consumes
    `tmat(:,:,1)=e`, `tmat(:,:,2)=f`, `tmat(:,:,3)=g1`, `rhs`, and `resolv`,
    and updates `mcl` for grid points where `resolv` is true. RHS construction
    is upstream in `hydrol_soil`, lines 5963-5991 and 6971-6991; this solver
    deliberately does not build RHS.
    """

    e = jnp.asarray(e)
    f = jnp.asarray(f)
    g1 = jnp.asarray(g1)
    rhs = jnp.asarray(rhs)

    if e.ndim == 1:
        e = e[None, :]
    if f.ndim == 1:
        f = f[None, :]
    if g1.ndim == 1:
        g1 = g1[None, :]
    if rhs.ndim == 1:
        rhs = rhs[None, :]
    if not (e.shape == f.shape == g1.shape == rhs.shape):
        raise ValueError("e, f, g1, and rhs must have matching shape (npts, nslm) or (nslm,)")

    npts, nslm = rhs.shape
    resolv = jnp.broadcast_to(jnp.asarray(resolv, dtype=bool), (npts,))
    if initial_mcl is None:
        initial_mcl = jnp.zeros_like(rhs)
    else:
        initial_mcl = jnp.asarray(initial_mcl)
        if initial_mcl.ndim == 1:
            initial_mcl = initial_mcl[None, :]
        if initial_mcl.shape != rhs.shape:
            raise ValueError("initial_mcl must match rhs shape")

    bet = jnp.zeros_like(rhs)
    gam = jnp.zeros_like(rhs)
    mcl = initial_mcl

    bet0 = f[:, 0]
    safe_bet0 = jnp.where(resolv, bet0, 1.0)
    mcl0 = rhs[:, 0] / safe_bet0
    bet = bet.at[:, 0].set(jnp.where(resolv, bet0, bet[:, 0]))
    mcl = mcl.at[:, 0].set(jnp.where(resolv, mcl0, mcl[:, 0]))

    for jsl in range(1, nslm):
        safe_previous_bet = jnp.where(resolv, bet[:, jsl - 1], 1.0)
        gam_j = g1[:, jsl - 1] / safe_previous_bet
        bet_j = f[:, jsl] - e[:, jsl] * gam_j
        safe_bet_j = jnp.where(resolv, bet_j, 1.0)
        mcl_j = (rhs[:, jsl] - e[:, jsl] * mcl[:, jsl - 1]) / safe_bet_j
        gam = gam.at[:, jsl].set(jnp.where(resolv, gam_j, gam[:, jsl]))
        bet = bet.at[:, jsl].set(jnp.where(resolv, bet_j, bet[:, jsl]))
        mcl = mcl.at[:, jsl].set(jnp.where(resolv, mcl_j, mcl[:, jsl]))

    for jsl in range(nslm - 2, -1, -1):
        updated = mcl[:, jsl] - gam[:, jsl + 1] * mcl[:, jsl + 1]
        mcl = mcl.at[:, jsl].set(jnp.where(resolv, updated, mcl[:, jsl]))

    return HydrolSoilTridiagResult(mcl=mcl, bet=bet, gam=gam)


def bottom_drainage(
    k_bottom,
    free_drain_coef,
    dt_days,
    *,
    mask_soiltile=1.0,
    resolv=True,
):
    """Compute bottom-boundary drainage in mm per SECHIBA time step.

    Fortran provenance: `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`,
    subroutine `hydrol_soil`, lines 6007-6016.
    """

    k_bottom = jnp.asarray(k_bottom)
    free_drain_coef = jnp.asarray(free_drain_coef)
    dt_days = jnp.asarray(dt_days)
    mask_soiltile = jnp.asarray(mask_soiltile)
    resolv = jnp.asarray(resolv)

    dr = mask_soiltile * k_bottom * free_drain_coef * dt_days
    return jnp.where(resolv, dr, 0.0)


def drainage_correction(dr_before, check_tr_ns):
    """Apply HYDROL drainage conservation correction.

    Fortran provenance: `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`,
    subroutine `hydrol_soil`, lines 6042-6047.
    """

    dr_before = jnp.asarray(dr_before)
    check_tr_ns = jnp.asarray(check_tr_ns)

    dr_corrnum = jnp.where(
        check_tr_ns < 0.0,
        -check_tr_ns,
        -jnp.minimum(dr_before, check_tr_ns),
    )
    return DrainageCorrectionResult(
        dr_corrnum=dr_corrnum,
        dr_after=dr_before + dr_corrnum,
    )


def hydrol_layer_moisture_content(moisture, dz_mm):
    """Integrate node moisture into HYDROL layer water contents.

    Fortran provenance: `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`,
    subroutine `hydrol_soil`, lines 6473-6521. The same trapezoid-like layer
    weights are also used for `tmci`, `tmcf`, and `tmc` in lines 5923-5929,
    6018-6025, and 6300-6313.
    """

    moisture = jnp.asarray(moisture)
    dz_mm = jnp.asarray(dz_mm)
    if moisture.ndim == 1:
        moisture = moisture[None, :]
    if moisture.ndim != 2:
        raise ValueError("moisture must have shape (npts, nslm) or (nslm,)")
    if dz_mm.ndim != 1:
        raise ValueError("dz_mm must be a one-dimensional hydrology layer-distance array in mm")
    if dz_mm.shape[0] != moisture.shape[1]:
        raise ValueError("dz_mm length must match the hydrology layer axis")

    npts, nslm = moisture.shape
    if nslm < 2:
        raise ValueError("HYDROL layer integration requires at least two layers")

    first = dz_mm[1] * (3.0 * moisture[:, :1] + moisture[:, 1:2]) / 8.0
    middle = (
        dz_mm[1:-1][None, :] * (3.0 * moisture[:, 1:-1] + moisture[:, :-2]) / 8.0
        + dz_mm[2:][None, :] * (3.0 * moisture[:, 1:-1] + moisture[:, 2:]) / 8.0
    )
    last = dz_mm[-1] * (3.0 * moisture[:, -1:] + moisture[:, -2:-1]) / 8.0
    return jnp.concatenate((first, middle, last), axis=1)


def _hydrol_layer_moisture_content_by_tile(moisture, dz_mm):
    """Vectorized layer integration for ``(npts, nslm, nstm)`` tile arrays."""

    moisture = _as_3d("moisture", moisture)
    dz_mm = _as_1d("dz_mm", dz_mm)
    if dz_mm.shape[0] != moisture.shape[1]:
        raise ValueError("dz_mm length must match the hydrology layer axis")
    if moisture.shape[1] < 2:
        raise ValueError("HYDROL layer integration requires at least two layers")

    first = dz_mm[1] * (3.0 * moisture[:, :1, :] + moisture[:, 1:2, :]) / 8.0
    middle = (
        dz_mm[1:-1][None, :, None] * (3.0 * moisture[:, 1:-1, :] + moisture[:, :-2, :]) / 8.0
        + dz_mm[2:][None, :, None] * (3.0 * moisture[:, 1:-1, :] + moisture[:, 2:, :]) / 8.0
    )
    last = dz_mm[-1] * (3.0 * moisture[:, -1:, :] + moisture[:, -2:-1, :]) / 8.0
    return jnp.concatenate((first, middle, last), axis=1)


def hydrol_total_moisture_content(moisture, dz_mm):
    """Integrate HYDROL node moisture into total column water content.

    Fortran provenance: `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`,
    subroutine `hydrol_soil`, lines 5923-5929 (`tmci`), 6018-6025 (`tmcf`),
    and 6300-6313 (`tmc`).
    """

    return jnp.sum(hydrol_layer_moisture_content(moisture, dz_mm), axis=1)


def hydrol_soil_layer_diagnostics(
    *,
    mc,
    mcl,
    dz_mm,
    njsc,
    mcs=None,
    mcf=None,
    mcw=None,
    mcs_mineral=USDA_MCS,
    mcf_mineral=USDA_MCF,
    mcw_mineral=USDA_MCW,
    pcent=None,
    peat_hydro=False,
    soil_tile_index=1,
    peat_tiles=(4, 5, 6),
    mcs_peat=PEAT_MCS,
    mcf_peat=PEAT_MCF,
    mcw_peat=PEAT_MCW,
    pcent_peat=PEAT_PCENT,
):
    """Build HYDROL's per-tile soil-moisture diagnostic layers.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil``, lines 6474-6562. The helper implements the
    source layer integrations for ``sm/smt/smw/smf/sms``, mineral temporary
    thresholds ``*_tmp``, ``sm_nostress``, and ``soil_wet_ns`` for one soil
    tile after the prognostic ``mc/mcl`` updates are complete.
    """

    mc = jnp.asarray(mc)
    mcl = jnp.asarray(mcl)
    if mc.ndim == 1:
        mc = mc[None, :]
    if mcl.ndim == 1:
        mcl = mcl[None, :]
    if mc.ndim != 2 or mcl.ndim != 2 or mc.shape != mcl.shape:
        raise ValueError("mc and mcl must have matching shape (npts, nslm)")
    dz_mm = _as_1d("dz_mm", dz_mm)
    if dz_mm.shape[0] != mc.shape[1]:
        raise ValueError("dz_mm length must match nslm")

    npts, nslm = mc.shape
    mineral_mcs = _per_point_threshold("mcs_mineral", mcs_mineral, njsc, npts, USDA_MCS)
    mineral_mcf = _per_point_threshold("mcf_mineral", mcf_mineral, njsc, npts, USDA_MCF)
    mineral_mcw = _per_point_threshold("mcw_mineral", mcw_mineral, njsc, npts, USDA_MCW)
    point_mcs = _per_point_threshold("mcs", mcs, njsc, npts, USDA_MCS)
    point_mcf = _per_point_threshold("mcf", mcf, njsc, npts, USDA_MCF)
    point_mcw = _per_point_threshold("mcw", mcw, njsc, npts, USDA_MCW)
    point_pcent = _per_point_threshold("pcent", pcent, njsc, npts, USDA_PCENT)

    peat_active = bool(peat_hydro) and int(soil_tile_index) in tuple(int(tile) for tile in peat_tiles)
    if peat_active:
        point_mcs = jnp.full((npts,), mcs_peat, dtype=mc.dtype)
        point_mcf = jnp.full((npts,), mcf_peat, dtype=mc.dtype)
        point_mcw = jnp.full((npts,), mcw_peat, dtype=mc.dtype)
        mineral_mcs = point_mcs
        mineral_mcf = point_mcf
        mineral_mcw = point_mcw
        point_pcent = jnp.full((npts,), pcent_peat, dtype=mc.dtype)

    def threshold_profile(values):
        return hydrol_layer_moisture_content(jnp.repeat(values[:, None], nslm, axis=1), dz_mm)
    sm = hydrol_layer_moisture_content(mcl, dz_mm)
    smt = hydrol_layer_moisture_content(mc, dz_mm)
    smw = threshold_profile(point_mcw)
    smf = threshold_profile(point_mcf)
    sms = threshold_profile(point_mcs)
    smw_tmp = threshold_profile(mineral_mcw)
    smf_tmp = threshold_profile(mineral_mcf)
    sms_tmp = threshold_profile(mineral_mcs)
    sm_nostress = smw + point_pcent[:, None] * (smf - smw)
    soil_wet_ns = jnp.clip(
        (sm - smw_tmp) * (sms - smw) / (sms_tmp - smw_tmp) ** 2,
        0.0,
        1.0,
    )

    return HydrolSoilLayerDiagnostics(
        sm=sm,
        smt=smt,
        smw=smw,
        smf=smf,
        sms=sms,
        smw_tmp=smw_tmp,
        smf_tmp=smf_tmp,
        sms_tmp=sms_tmp,
        sm_nostress=sm_nostress,
        soil_wet_ns=soil_wet_ns,
    )


def hydrol_water_stress_diagnostics(
    *,
    layer: HydrolSoilLayerDiagnostics,
    nroot,
    vegetmax_soil_tile,
    is_under_mcr,
    njsc,
    pcent=None,
    pcent_mineral=USDA_PCENT,
    peat_hydro=False,
    soil_tile_index=1,
    peat_tiles=(4, 5, 6),
    pcent_peat=PEAT_PCENT,
    dyn_nroot_larix=False,
    is_tree=None,
    znt=None,
    new_watstress=False,
    alpha_watstress=1.0,
    min_sechiba=1.0e-8,
):
    """Compute HYDROL ``us``, ``humrelv``, and ``vegstressv`` for one tile.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil``, lines 6565-6725. The optional dynamic-root
    branch follows lines 6584-6629 and therefore requires the explicit source
    inputs ``is_tree`` and ``znt`` when enabled.
    """

    nroot = _as_3d("nroot", nroot)
    vegetmax_soil_tile = _as_2d("vegetmax_soil_tile", vegetmax_soil_tile)
    is_under_mcr = _as_1d("is_under_mcr", is_under_mcr).astype(bool)
    _require_same_npts(nroot, vegetmax_soil_tile, is_under_mcr)
    npts, nvm, nslm = nroot.shape
    if vegetmax_soil_tile.shape != (npts, nvm):
        raise ValueError("vegetmax_soil_tile must have shape (npts, nvm)")
    for name in ("sm", "smw", "smf", "smw_tmp", "smf_tmp", "sm_nostress", "soil_wet_ns"):
        if getattr(layer, name).shape != (npts, nslm):
            raise ValueError(f"layer.{name} must have shape (npts, nslm)")

    mineral_pcent_values = _per_point_threshold("pcent", pcent, njsc, npts, pcent_mineral)
    pcent_values = mineral_pcent_values
    peat_active = bool(peat_hydro) and int(soil_tile_index) in tuple(int(tile) for tile in peat_tiles)
    if peat_active:
        pcent_values = jnp.full((npts,), pcent_peat, dtype=layer.sm.dtype)

    # hydrol.f90:6574-6670 assigns the top layer to zero and evaluates only
    # layers 2:nslm. Avoid constructing the source-absent top-layer 0/0 in the
    # AD graph, and mirror NEW_WATSTRESS as static Fortran control flow.
    active_layer = jnp.arange(nslm) > 0
    threshold_span = layer.smf_tmp - layer.smw_tmp
    safe_threshold_span = jnp.where(active_layer[None, :], threshold_span, 1.0)
    old_factor = jnp.clip(
        (layer.sm - layer.smw_tmp)
        / (pcent_values[:, None] * safe_threshold_span)
        * (layer.smf - layer.smw)
        / safe_threshold_span,
        0.0,
        1.0,
    )
    old_factor = jnp.where(active_layer[None, :], old_factor, 0.0)
    if bool(new_watstress):
        moisture_above_wilt = layer.sm - layer.smw
        positive = (moisture_above_wilt > min_sechiba) & active_layer[None, :]
        safe_moisture_above_wilt = jnp.where(positive, moisture_above_wilt, 1.0)
        nostress_span = layer.sm_nostress - layer.smw
        safe_nostress_span = jnp.where(positive, nostress_span, 1.0)
        stress_factor = jnp.where(
            positive,
            jnp.clip(
                jnp.exp(
                    -jnp.asarray(alpha_watstress, dtype=layer.sm.dtype)
                    * ((layer.smf - layer.smw) / safe_nostress_span)
                    * ((layer.sm_nostress - layer.sm) / safe_moisture_above_wilt)
                ),
                0.0,
                1.0,
            ),
            0.0,
        )
    else:
        stress_factor = old_factor

    nroot_work = nroot
    if bool(dyn_nroot_larix):
        if is_tree is None or znt is None:
            raise ValueError("dyn_nroot_larix requires explicit is_tree and znt inputs")
        is_tree = _as_1d("is_tree", is_tree).astype(bool)
        znt = _as_1d("znt", znt)
        if is_tree.shape[0] != nvm or znt.shape[0] != nslm:
            raise ValueError("is_tree must have length nvm and znt must have length nslm")
        # hydrol.f90:6584-6635 updates nroot with the legacy threshold
        # expression even when NEW_WATSTRESS selects the exponential formula
        # later for ``us``.  Its peat-tile mask also differs for trees: tile 6
        # uses mineral pcent for the root update, while lines 6660-6662 use
        # pcent_peat for the subsequent water-stress calculation.
        mineral_dyn_factor = jnp.clip(
            (layer.sm - layer.smw_tmp)
            / (mineral_pcent_values[:, None] * (layer.smf_tmp - layer.smw_tmp))
            * (layer.smf - layer.smw)
            / (layer.smf_tmp - layer.smw_tmp),
            0.0,
            1.0,
        )
        peat_dyn_factor = jnp.clip(
            (layer.sm - layer.smw_tmp)
            / (float(pcent_peat) * (layer.smf_tmp - layer.smw_tmp))
            * (layer.smf - layer.smw)
            / (layer.smf_tmp - layer.smw_tmp),
            0.0,
            1.0,
        )
        for jv in range(nvm):
            if bool(peat_hydro) and int(soil_tile_index) in (4, 5):
                dyn_base = peat_dyn_factor
            elif bool(peat_hydro) and int(soil_tile_index) == 6:
                dyn_base = jnp.where(is_tree[jv], mineral_dyn_factor, peat_dyn_factor)
            else:
                dyn_base = mineral_dyn_factor
            dyn_base = dyn_base.at[:, 0].set(0.0)
            allowed = nroot_work[:, jv, :] != 0.0
            depth_allowed = jnp.where(is_tree[jv], jnp.ones_like(znt, dtype=bool), znt < 1.0)
            allowed = allowed & depth_allowed[None, :]
            candidate = jnp.where(allowed, dyn_base, 0.0)
            denom = jnp.sum(candidate, axis=1)
            positive_denom = denom > 0.0
            safe_denom = jnp.where(positive_denom, denom, 1.0)
            updated = jnp.where(
                positive_denom[:, None],
                candidate / safe_denom[:, None],
                nroot_work[:, jv, :],
            )
            nroot_work = nroot_work.at[:, jv, :].set(updated)

    us = stress_factor[:, None, :] * nroot_work
    us = us.at[:, :, 0].set(0.0)
    us = us.at[:, 0, :].set(0.0)
    humrelv = jnp.sum(us, axis=2)

    vegstress_factor = jnp.clip(
        (layer.sm - layer.smw_tmp) * (layer.smf - layer.smw) / (layer.smf_tmp - layer.smw_tmp) ** 2,
        0.0,
        1.0,
    )
    vegstressv = jnp.sum(vegstress_factor[:, None, :] * nroot_work, axis=2)
    vegstressv = vegstressv.at[:, 0].set(0.0)

    absent = vegetmax_soil_tile < min_sechiba
    humrelv = jnp.where(absent, 0.0, humrelv)
    vegstressv = jnp.where(absent, 0.0, vegstressv)
    us = jnp.where(absent[:, :, None], 0.0, us)
    us = jnp.where(is_under_mcr[:, None, None], 0.0, us)
    humrelv = jnp.where(is_under_mcr[:, None], 0.0, humrelv)
    soil_wet_ns = jnp.where(is_under_mcr[:, None], 0.0, layer.soil_wet_ns)

    return HydrolWaterStressDiagnostics(
        us=us,
        humrelv=humrelv,
        vegstressv=vegstressv,
        nroot=nroot_work,
        soil_wet_ns=soil_wet_ns,
        undermcr_increment=is_under_mcr.astype(layer.sm.dtype),
    )


def hydrol_stress_aggregates(*, humrelv, vegstressv, veget_max, min_sechiba=1.0e-8):
    """Aggregate HYDROL water-stress diagnostics across soil tiles.

    Fortran provenance: ``hydrol_diag_soil`` lines 9089-9107. ``vegstress``
    is accumulated only for present PFTs, while ``humrel`` is the non-negative
    sum of ``humrelv`` across soil tiles.
    """

    humrelv = _as_3d("humrelv", humrelv)
    vegstressv = _as_3d("vegstressv", vegstressv)
    veget_max = _as_2d("veget_max", veget_max)
    _require_same_npts(humrelv, vegstressv, veget_max)
    if humrelv.shape != vegstressv.shape or humrelv.shape[:2] != veget_max.shape:
        raise ValueError("humrelv/vegstressv must be (npts,nvm,nstm) and match veget_max")
    humrel = jnp.maximum(jnp.sum(humrelv, axis=2), 0.0)
    vegstress = jnp.maximum(jnp.sum(vegstressv, axis=2), 0.0)
    vegstress = jnp.where(veget_max > min_sechiba, vegstress, 0.0)
    return HydrolStressAggregates(humrel=humrel, vegstress=vegstress)


def hydrol_soilmoist_aggregates(*, mc, mcl, soiltile, vegtot, vegtot_old=None, dz_mm=None):
    """Aggregate HYDROL tile moisture into thermosoil and output fields.

    Fortran provenance: ``hydrol_soil`` lines 6729-6741 and
    ``hydrol_diag_soil`` lines 9159-9195. ``mc_layh/mcl_layh`` are
    tile-weighted volumetric contents multiplied by ``vegtot``; ``soilmoist``
    and ``soilmoist_liquid`` are layer-integrated water contents weighted over
    soil tiles and then converted to grid-cell averages. The source assigns
    both tile-layer export arrays from ``mc`` at lines 6734-6735.
    """

    mc = _as_3d("mc", mc)
    mcl = _as_3d("mcl", mcl)
    soiltile = _as_2d("soiltile", soiltile)
    vegtot = _as_1d("vegtot", vegtot)
    if vegtot_old is None:
        vegtot_old = vegtot
    vegtot_old = _as_1d("vegtot_old", vegtot_old)
    _require_same_npts(mc, mcl, soiltile, vegtot, vegtot_old)
    if mc.shape != mcl.shape or mc.shape[2] != soiltile.shape[1]:
        raise ValueError("mc/mcl must be (npts,nslm,nstm) and match soiltile")
    if dz_mm is None:
        raise ValueError("dz_mm is required to compute soilmoist and soilmoist_liquid")
    dz_mm = _as_1d("dz_mm", dz_mm)
    if dz_mm.shape[0] != mc.shape[1]:
        raise ValueError("dz_mm length must match nslm")

    mc_layers = _hydrol_layer_moisture_content_by_tile(mc, dz_mm)
    mcl_layers = _hydrol_layer_moisture_content_by_tile(mcl, dz_mm)
    soilmoist = jnp.sum(soiltile[:, None, :] * mc_layers, axis=2)
    soilmoist_liquid = jnp.sum(soiltile[:, None, :] * mcl_layers, axis=2)
    soilmoist = soilmoist * vegtot[:, None]
    soilmoist_liquid = soilmoist_liquid * vegtot_old[:, None]
    mc_layh = jnp.sum(mc * soiltile[:, None, :] * vegtot[:, None, None], axis=2)
    mcl_layh = jnp.sum(mcl * soiltile[:, None, :] * vegtot[:, None, None], axis=2)
    return HydrolSoilMoistureAggregates(
        soilmoist=soilmoist,
        soilmoist_liquid=soilmoist_liquid,
        mc_layh=mc_layh,
        mcl_layh=mcl_layh,
        mc_layh_s=mc,
        mcl_layh_s=mc,
    )


def hydrol_shumdiag(
    *,
    mc,
    soil_wet_ns,
    soilmoist,
    soiltile,
    dh_mm,
    dz_mm,
    njsc,
    mcs=None,
    mcf=None,
    mcw=None,
    mcs_mineral=USDA_MCS,
    peat_hydro=False,
    agri_peat=False,
    tides=False,
    mcs_peat=PEAT_MCS,
    mcf_peat=PEAT_MCF,
    mcw_peat=PEAT_MCW,
):
    """Compute HYDROL ``shumdiag`` and related peat/permafrost diagnostics.

    Fortran provenance: ``hydrol_diag_soil`` lines 9199-9284. The helper
    consumes explicit post-solve ``soil_wet_ns`` and ``soilmoist`` state and
    applies the source tile weighting and clamping rules for mineral,
    peatland, crop-peat, mangrove/tide, and permafrost diagnostics.
    """

    mc = _as_3d("mc", mc)
    soil_wet_ns = _as_3d("soil_wet_ns", soil_wet_ns)
    soilmoist = _as_2d("soilmoist", soilmoist)
    soiltile = _as_2d("soiltile", soiltile)
    dh_mm = _as_1d("dh_mm", dh_mm)
    dz_mm = _as_1d("dz_mm", dz_mm)
    _require_same_npts(mc, soil_wet_ns, soilmoist, soiltile)
    if mc.shape != soil_wet_ns.shape or mc.shape[2] != soiltile.shape[1]:
        raise ValueError("mc and soil_wet_ns must be (npts,nslm,nstm) and match soiltile")
    if soilmoist.shape != mc.shape[:2] or dh_mm.shape[0] != mc.shape[1] or dz_mm.shape[0] != mc.shape[1]:
        raise ValueError("soilmoist, dh_mm, and dz_mm must match nslm")

    npts, nslm, nstm = mc.shape
    point_mcs = _per_point_threshold("mcs", mcs, njsc, npts, USDA_MCS)
    point_mcf = _per_point_threshold("mcf", mcf, njsc, npts, USDA_MCF)
    point_mcw = _per_point_threshold("mcw", mcw, njsc, npts, USDA_MCW)
    mineral_mcs = _per_point_threshold("mcs_mineral", mcs_mineral, njsc, npts, USDA_MCS)

    shumdiag = jnp.zeros((npts, nslm), dtype=mc.dtype)
    for jst in range(nstm):
        peat_tile = bool(peat_hydro) and (jst + 1) in (4, 5, 6)
        if peat_tile:
            scale = (mcs_peat - mcw_peat) / (mcf_peat - mcw_peat)
            scale = jnp.full((npts,), scale, dtype=mc.dtype)
        else:
            scale = (point_mcs - point_mcw) / (point_mcf - point_mcw)
        shumdiag = jnp.clip(shumdiag + soil_wet_ns[:, :, jst] * soiltile[:, jst, None] * scale[:, None], 0.0, 1.0)

    zeros = jnp.zeros((npts, nslm), dtype=mc.dtype)
    shumdiag_peat = zeros
    shumdiag_croppeat = zeros
    shumdiag_man = zeros
    layer_moisture = None
    if bool(peat_hydro) and nstm >= 4:
        layer_moisture = _hydrol_layer_moisture_content_by_tile(mc, dz_mm)
        shumdiag_peat = jnp.clip(layer_moisture[:, :, 3] / dh_mm[None, :], 0.0, 1.0)
        if bool(tides) and nstm >= 6:
            shumdiag_man = jnp.clip(layer_moisture[:, :, 5] / dh_mm[None, :], 0.0, 1.0)
    if bool(agri_peat) and nstm >= 5:
        if layer_moisture is None:
            layer_moisture = _hydrol_layer_moisture_content_by_tile(mc, dz_mm)
        shumdiag_croppeat = jnp.clip(layer_moisture[:, :, 4] / dh_mm[None, :], 0.0, 1.0)

    if bool(peat_hydro):
        shumdiag_perma = soilmoist / dh_mm[None, :]
    else:
        shumdiag_perma = soilmoist / dh_mm[None, :] / point_mcs[:, None] * mineral_mcs[:, None]
    shumdiag_perma = jnp.clip(shumdiag_perma, 0.0, 1.0)

    return HydrolShumdiagDiagnostics(
        shumdiag=shumdiag,
        shumdiag_perma=shumdiag_perma,
        shumdiag_peat=shumdiag_peat,
        shumdiag_croppeat=shumdiag_croppeat,
        shumdiag_man=shumdiag_man,
    )


def _hydrol_top_partial_moisture(moisture, dz_mm, max_layer):
    moisture = jnp.asarray(moisture)
    if moisture.ndim == 1:
        moisture = moisture[None, :]
    dz_mm = _as_1d("dz_mm", dz_mm)
    if moisture.ndim != 2 or dz_mm.shape[0] != moisture.shape[1]:
        raise ValueError("moisture must be (npts,nslm) and dz_mm length must match nslm")
    nslm = moisture.shape[1]
    if nslm < max_layer + 1:
        raise ValueError("HYDROL top partial moisture diagnostic requires enough layers for source stencil")
    total = dz_mm[1] * (3.0 * moisture[:, 0] + moisture[:, 1]) / 8.0
    for jsl in range(1, max_layer):
        total = total + dz_mm[jsl] * (3.0 * moisture[:, jsl] + moisture[:, jsl - 1]) / 8.0
        total = total + dz_mm[jsl + 1] * (3.0 * moisture[:, jsl] + moisture[:, jsl + 1]) / 8.0
    return total


def hydrol_litter_top_diagnostics(
    *,
    mc,
    dz_mm,
    dh_mm,
    njsc,
    mcr=USDA_MCR,
    mcs=None,
    mcf=None,
    mcw=None,
    mc_awet=USDA_MC_AWET,
    mc_adry=USDA_MC_ADRY,
    peat_hydro=False,
    perma_peat=None,
    agri_peat=False,
    tides=False,
    peat_tiles=(4, 5, 6),
    mcr_peat=PEAT_MCR,
    mcs_peat=PEAT_MCS,
    mcf_peat=PEAT_MCF,
    mcw_peat=PEAT_MCW,
    mc_awet_peat=PEAT_MC_AWET,
    mc_adry_peat=PEAT_MC_ADRY,
):
    """Compute HYDROL litter and top-grass moisture diagnostics.

    Fortran provenance: ``hydrol_soil`` lines 6414-6470 and 6793-6795, plus
    the matching initialization formulas at lines 4418-4508. The source uses
    the upper four HYDROL layers for litter water and the upper five layers
    for grazing/top-grass moisture, then derives peat/croppeat/man above-layer
    moisture when permafrost peat logic is active.
    """

    mc = _as_3d("mc", mc)
    dz_mm = _as_1d("dz_mm", dz_mm)
    dh_mm = _as_1d("dh_mm", dh_mm)
    if dz_mm.shape[0] != mc.shape[1] or dh_mm.shape[0] != mc.shape[1]:
        raise ValueError("dz_mm and dh_mm must match nslm")
    if mc.shape[1] < 7:
        raise ValueError("litter/topgrass diagnostics require at least seven HYDROL layers")
    npts, _nslm, nstm = mc.shape
    mcr_values = _per_point_threshold("mcr", mcr, njsc, npts, USDA_MCR)
    point_mcs = _per_point_threshold("mcs", mcs, njsc, npts, USDA_MCS)
    point_mcf = _per_point_threshold("mcf", mcf, njsc, npts, USDA_MCF)
    point_mcw = _per_point_threshold("mcw", mcw, njsc, npts, USDA_MCW)
    point_awet = _per_point_threshold("mc_awet", mc_awet, njsc, npts, USDA_MC_AWET)
    point_adry = _per_point_threshold("mc_adry", mc_adry, njsc, npts, USDA_MC_ADRY)

    tmc_litter = jnp.zeros((npts, nstm), dtype=mc.dtype)
    tmc_trampling = jnp.zeros_like(tmc_litter)
    tmc_wilt = jnp.zeros_like(tmc_litter)
    tmc_res = jnp.zeros_like(tmc_litter)
    tmc_field = jnp.zeros_like(tmc_litter)
    tmc_sat = jnp.zeros_like(tmc_litter)
    tmc_awet = jnp.zeros_like(tmc_litter)
    tmc_adry = jnp.zeros_like(tmc_litter)

    for jst in range(nstm):
        peat_tile = bool(peat_hydro) and (jst + 1) in tuple(int(tile) for tile in peat_tiles)
        tmc_litter = tmc_litter.at[:, jst].set(_hydrol_top_partial_moisture(mc[:, :, jst], dz_mm, 4))
        tmc_trampling = tmc_trampling.at[:, jst].set(_hydrol_top_partial_moisture(mc[:, :, jst], dz_mm, 6))
        if peat_tile:
            wilt = jnp.full((npts,), mcw_peat, dtype=mc.dtype)
            res = jnp.full((npts,), mcr_peat, dtype=mc.dtype)
            field = jnp.full((npts,), mcf_peat, dtype=mc.dtype)
            sat = jnp.full((npts,), mcs_peat, dtype=mc.dtype)
            awet = jnp.full((npts,), mc_awet_peat, dtype=mc.dtype)
            adry = jnp.full((npts,), mc_adry_peat, dtype=mc.dtype)
        else:
            wilt, res, field, sat, awet, adry = point_mcw, mcr_values, point_mcf, point_mcs, point_awet, point_adry
        for arr_name, values in (
            ("wilt", wilt),
            ("res", res),
            ("field", field),
            ("sat", sat),
            ("awet", awet),
            ("adry", adry),
        ):
            threshold = _hydrol_top_partial_moisture(jnp.repeat(values[:, None], mc.shape[1], axis=1), dz_mm, 4)
            if arr_name == "wilt":
                tmc_wilt = tmc_wilt.at[:, jst].set(threshold)
            elif arr_name == "res":
                tmc_res = tmc_res.at[:, jst].set(threshold)
            elif arr_name == "field":
                tmc_field = tmc_field.at[:, jst].set(threshold)
            elif arr_name == "sat":
                tmc_sat = tmc_sat.at[:, jst].set(threshold)
            elif arr_name == "awet":
                tmc_awet = tmc_awet.at[:, jst].set(threshold)
            else:
                tmc_adry = tmc_adry.at[:, jst].set(threshold)

    soil_wet_litter = jnp.clip((tmc_litter - tmc_wilt) / (tmc_field - tmc_wilt), 0.0, 1.0)
    tmc_topgrass = tmc_trampling[:, 2] / (jnp.sum(dz_mm[:6]) + dz_mm[6] / 2.0)
    peat_active = bool(peat_hydro) if perma_peat is None else bool(perma_peat)
    top4_depth = jnp.sum(dh_mm[:4])
    mc_peat_above = jnp.zeros((npts,), dtype=mc.dtype)
    mc_croppeat_above = jnp.zeros_like(mc_peat_above)
    mc_man_above = jnp.zeros_like(mc_peat_above)
    if peat_active and nstm >= 4:
        mc_peat_above = tmc_litter[:, 3] / top4_depth
        if bool(agri_peat) and nstm >= 5:
            mc_croppeat_above = tmc_litter[:, 4] / top4_depth
        if bool(tides) and nstm >= 6:
            mc_man_above = tmc_litter[:, 5] / top4_depth

    return HydrolLitterTopDiagnostics(
        tmc_litter=tmc_litter,
        tmc_litter_wilt=tmc_wilt,
        tmc_litter_res=tmc_res,
        tmc_litter_field=tmc_field,
        tmc_litter_sat=tmc_sat,
        tmc_litter_awet=tmc_awet,
        tmc_litter_adry=tmc_adry,
        soil_wet_litter=soil_wet_litter,
        tmc_trampling=tmc_trampling,
        tmc_topgrass=tmc_topgrass,
        mc_peat_above=mc_peat_above,
        mc_croppeat_above=mc_croppeat_above,
        mc_man_above=mc_man_above,
    )


def hydrol_peat_water_table_diagnostics(
    *,
    mc,
    mcl,
    soiltile,
    dz_mm,
    dh_mm,
    wt_ab,
    wt_ab_tide=None,
    peat_hydro=False,
    tides=False,
    liqlayers=10,
    numlayers=11,
    topmodel_new=False,
    temp_hydro=None,
    mcs=None,
    param_vp=None,
    param_kp=None,
    param_qp=None,
    param_fmax=None,
    mcr_peat=PEAT_MCR,
    mcs_peat=PEAT_MCS,
    undef_sechiba=1.0e20,
    zero_celsius=273.15,
    min_sechiba=1.0e-8,
):
    """Diagnose peat liquid water, water-table position, and optional fwet.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil`` lines 7180-7342. ``fwet_new`` is returned only
    for the explicit ``topmodel_new`` branch because it requires topmodel
    parameters and the freeze-depth state.
    """

    mc = jnp.asarray(mc)
    mcl = jnp.asarray(mcl)
    if mc.ndim != 3 or mcl.shape != mc.shape:
        raise ValueError("mc and mcl must have matching shape (npts, nslm, nstm)")
    soiltile = _as_2d("soiltile", soiltile)
    dz_mm = _as_1d("dz_mm", dz_mm)
    dh_mm = _as_1d("dh_mm", dh_mm)
    wt_ab = _as_1d("wt_ab", wt_ab)
    _require_same_npts(mc, soiltile, wt_ab)
    if soiltile.shape[1] != mc.shape[2] or dz_mm.shape[0] != mc.shape[1] or dh_mm.shape[0] != mc.shape[1]:
        raise ValueError("soiltile, dz_mm, and dh_mm must match mc dimensions")

    npts, nslm, nstm = mc.shape
    if wt_ab_tide is None:
        wt_ab_tide = jnp.zeros_like(wt_ab)
    else:
        wt_ab_tide = _as_1d("wt_ab_tide", wt_ab_tide)
        _require_same_npts(mc, wt_ab_tide)

    liqwt_ratio = jnp.zeros((npts,), dtype=mc.dtype)
    wtp = jnp.zeros((npts,), dtype=mc.dtype)
    if bool(peat_hydro) and nstm >= 4:
        liqlayers = int(liqlayers)
        if liqlayers > int(numlayers):
            raise ValueError("liqlayers must not exceed numlayers")
        if liqlayers < 1 or liqlayers + 1 > nslm:
            raise ValueError("liqlayers requires layers 1..liqlayers+1 in HYDROL indexing")
        tmcs_peat = dz_mm[1] * (3.0 * mcs_peat + mcs_peat) / 8.0
        tmcl_peat = dz_mm[1] * (3.0 * mcl[:, 0, 3] + mcl[:, 1, 3]) / 8.0
        for jsl in range(1, liqlayers):
            tmcs_peat = tmcs_peat + dz_mm[jsl] * (3.0 * mcs_peat + mcs_peat) / 8.0 + dz_mm[jsl + 1] * (3.0 * mcs_peat + mcs_peat) / 8.0
            tmcl_peat = (
                tmcl_peat
                + dz_mm[jsl] * (3.0 * mcl[:, jsl, 3] + mcl[:, jsl - 1, 3]) / 8.0
                + dz_mm[jsl + 1] * (3.0 * mcl[:, jsl, 3] + mcl[:, jsl + 1, 3]) / 8.0
            )
        liqwt_ratio = tmcl_peat / tmcs_peat

        h_eau = jnp.clip((mc[:, :, 3] - mcr_peat) / (mcs_peat - mcr_peat), 0.0, 1.0)
        peat_wtp = 2000.0 - (jnp.sum(h_eau * dz_mm[None, :], axis=1) + jnp.where(wt_ab > 0.0, wt_ab, 0.0))
        wtp = jnp.where(soiltile[:, 3] > min_sechiba, peat_wtp, wtp)

    if bool(tides) and nstm >= 6:
        h_eau = jnp.clip((mc[:, :, 5] - mcr_peat) / (mcs_peat - mcr_peat), 0.0, 1.0)
        tide_wtp = 2000.0 - (jnp.sum(h_eau * dz_mm[None, :], axis=1) + jnp.where(wt_ab > 0.0, wt_ab, 0.0))
        wtp = jnp.where(soiltile[:, 5] > min_sechiba, tide_wtp, wtp)

    meanwt = None
    fwet_new = None
    if bool(topmodel_new):
        if temp_hydro is None or mcs is None:
            raise ValueError("topmodel_new requires temp_hydro and mcs")
        missing_params = [name for name, value in (("param_vp", param_vp), ("param_kp", param_kp), ("param_qp", param_qp), ("param_fmax", param_fmax)) if value is None]
        if missing_params:
            raise ValueError("topmodel_new requires " + ", ".join(missing_params))
        temp_hydro = jnp.asarray(temp_hydro)
        if temp_hydro.ndim == 3:
            temp_profile = temp_hydro[:, :, 0]
        elif temp_hydro.ndim == 2:
            temp_profile = temp_hydro
        else:
            raise ValueError("temp_hydro must have shape (npts,nslm) or (npts,nslm,nstm)")
        if temp_profile.shape[0] != npts or temp_profile.shape[1] < nslm:
            raise ValueError("temp_hydro must match HYDROL soil layers")
        mcs = _as_1d("mcs", mcs)
        if mcs.shape[0] == 1:
            mcs = jnp.repeat(mcs, npts)
        _require_same_npts(mc, mcs)
        param_vp = _as_1d("param_vp", param_vp)
        param_kp = _as_1d("param_kp", param_kp)
        param_qp = _as_1d("param_qp", param_qp)
        param_fmax = _as_1d("param_fmax", param_fmax)
        _require_same_npts(mc, param_vp, param_kp, param_qp, param_fmax)

        max_layers = min(int(numlayers), nslm)
        unfrozen_count = jnp.sum(temp_profile[:, :max_layers] >= zero_celsius, axis=1)
        active = unfrozen_count > 6
        unfrozen_depth = jnp.sum(jnp.where(jnp.arange(nslm)[None, :] < unfrozen_count[:, None], dh_mm[None, :], 0.0), axis=1)
        wtp_temp = jnp.zeros((npts, nstm), dtype=mc.dtype)
        for jst in range(nstm):
            depth_sum = jnp.sum(jnp.where(jnp.arange(nslm)[None, :] < unfrozen_count[:, None], mc[:, :, jst] / (mcs_peat if bool(peat_hydro) else mcs[:, None]) * dh_mm[None, :], 0.0), axis=1)
            tile_wtp = depth_sum / 1000.0 - unfrozen_depth / 1000.0
            if bool(peat_hydro):
                tile_wtp = jnp.where(tile_wtp > 0.0, tile_wtp * mcs_peat, tile_wtp)
                add_ab = wt_ab_tide if bool(tides) and nstm >= 6 and jst == 5 else wt_ab
                tile_wtp = tile_wtp + add_ab / 1000.0
            else:
                tile_wtp = jnp.where(tile_wtp > 0.0, tile_wtp * mcs, tile_wtp)
            tile_wtp = jnp.where(active, tile_wtp, undef_sechiba)
            wtp_temp = wtp_temp.at[:, jst].set(tile_wtp)
        meanwt = jnp.where(active, jnp.sum(soiltile * wtp_temp, axis=1), undef_sechiba)

        bad_params = (param_vp == undef_sechiba) | (param_kp == undef_sechiba) | (param_qp == undef_sechiba) | (param_fmax == undef_sechiba) | (meanwt == undef_sechiba)
        param_tmp = -param_kp * (meanwt - param_qp)
        huge = jnp.asarray(jnp.finfo(mc.dtype).max, dtype=mc.dtype)
        overflow = param_tmp > 700.0
        overflow_base = jnp.where(param_vp < -1.0, 1.0 - huge, jnp.where(param_vp > 1.0, 1.0 + huge, 1.0 + param_vp * huge))
        regular_base = 1.0 + param_vp * jnp.exp(jnp.minimum(param_tmp, 700.0))
        fwet_cal = jnp.where(overflow, overflow_base ** (-1.0 / param_vp), regular_base ** (-1.0 / param_vp))
        fwet_new = jnp.where(bad_params, 0.0, jnp.maximum(0.0, jnp.minimum(param_fmax, fwet_cal)))

    return HydrolPeatWaterTableDiagnostics(
        liqwt_ratio=liqwt_ratio,
        wtp=wtp,
        meanwt=meanwt,
        fwet_new=fwet_new,
    )


def hydrol_grid_flux_aggregates(
    *,
    ae_ns,
    dr_ns,
    ru_ns,
    tmc,
    soiltile,
    vegtot,
    frac_bare_ns,
    vevapnu,
    subsinksoil,
    tot_melt,
    irrigation,
    returnflow,
    reinfiltration,
    mask_soiltile=None,
    min_sechiba=1.0e-8,
):
    """Aggregate HYDROL tile fluxes to grid-cell runoff/drainage/water state.

    Fortran provenance: ``hydrol_diag_soil`` lines 9006-9082. The source masks
    inactive soil tiles, weights bare-soil evaporation by ``frac_bare_ns``,
    averages runoff/drainage/humtot by ``vegtot*soiltile``, applies the no-veg
    runoff fallback, and adds ``subsinksoil*vegtot`` to ``vevapnu``.
    """

    ae_ns = _as_2d("ae_ns", ae_ns)
    dr_ns = _as_2d("dr_ns", dr_ns)
    ru_ns = _as_2d("ru_ns", ru_ns)
    tmc = _as_2d("tmc", tmc)
    soiltile = _as_2d("soiltile", soiltile)
    frac_bare_ns = _as_2d("frac_bare_ns", frac_bare_ns)
    vegtot = _as_1d("vegtot", vegtot)
    vevapnu = _as_1d("vevapnu", vevapnu)
    subsinksoil = _as_1d("subsinksoil", subsinksoil)
    tot_melt = _as_1d("tot_melt", tot_melt)
    irrigation = _as_1d("irrigation", irrigation)
    returnflow = _as_1d("returnflow", returnflow)
    reinfiltration = _as_1d("reinfiltration", reinfiltration)
    _require_same_npts(ae_ns, dr_ns, ru_ns, tmc, soiltile, frac_bare_ns, vegtot, vevapnu)
    if not (ae_ns.shape == dr_ns.shape == ru_ns.shape == tmc.shape == soiltile.shape == frac_bare_ns.shape):
        raise ValueError("tile arrays must share shape (npts,nstm)")
    if mask_soiltile is None:
        mask_soiltile = jnp.ones_like(soiltile)
    else:
        mask_soiltile = _as_2d("mask_soiltile", mask_soiltile)
    if mask_soiltile.shape != soiltile.shape:
        raise ValueError("mask_soiltile must match soiltile shape")

    ae_masked = ae_ns * mask_soiltile
    dr_masked = dr_ns * mask_soiltile
    ru_masked = ru_ns * mask_soiltile
    tmc_masked = tmc * mask_soiltile
    mask_vegtot = (vegtot > min_sechiba).astype(ae_ns.dtype)
    ae_out = mask_vegtot[:, None] * ae_masked * frac_bare_ns
    weighted = vegtot[:, None] * soiltile
    drainage = mask_vegtot * jnp.sum(weighted * dr_masked, axis=1)
    runoff_veg = mask_vegtot * jnp.sum(weighted * ru_masked, axis=1)
    runoff_noveg = (1.0 - mask_vegtot) * (tot_melt + irrigation + returnflow + reinfiltration)
    humtot = mask_vegtot * jnp.sum(weighted * tmc_masked, axis=1)
    vevapnu_out = vevapnu + subsinksoil * vegtot
    return HydrolGridFluxAggregates(
        ae_ns=ae_out,
        runoff=runoff_veg + runoff_noveg,
        drainage=drainage,
        humtot=humtot,
        vevapnu=vevapnu_out,
    )


def hydrol_litter_grid_diagnostics(
    *,
    litter: HydrolLitterTopDiagnostics,
    soiltile,
    vegtot,
    njsc,
    mineral_tables: MineralCWRRTables | None = None,
    peat_tables: PeatCWRRTables | None = None,
    ks=USDA_KS,
    peat_hydro=False,
    peat_tiles=(4, 5, 6),
    k_lin_layer1=None,
    k_lin_peat_layer1=None,
    imin=1,
    imax=51,
):
    """Aggregate HYDROL litter conductivity, humidity, and dry-soil fraction.

    Fortran provenance: ``hydrol_diag_soil`` lines 9112-9155. The conductivity
    lookup is table-driven from source CWRR `k_lin`/`k_lin_peat` values at
    layer 1; callers must pass built tables or explicit layer-1 tables.
    """

    soiltile = _as_2d("soiltile", soiltile)
    vegtot = _as_1d("vegtot", vegtot)
    _require_same_npts(soiltile, vegtot, litter.tmc_litter)
    if litter.tmc_litter.shape != soiltile.shape:
        raise ValueError("litter tile arrays must match soiltile")
    npts, nstm = soiltile.shape
    ks_point = _per_point_threshold("ks", ks, njsc, npts, USDA_KS)

    if mineral_tables is not None:
        k_lin_layer1 = mineral_tables.k_lin[:, 0, :]
        imin = mineral_tables.imin
        imax = mineral_tables.imax
    if peat_tables is not None:
        k_lin_peat_layer1 = peat_tables.k_lin[:, 0]
    if k_lin_layer1 is None:
        raise ValueError("mineral k_litt requires mineral_tables or k_lin_layer1")
    k_lin_layer1 = jnp.asarray(k_lin_layer1)
    if k_lin_layer1.ndim == 1:
        k_lin_layer1 = jnp.repeat(k_lin_layer1[:, None], npts, axis=1)
    if k_lin_layer1.shape[1] != npts:
        raise ValueError("k_lin_layer1 must have shape (nbounds,npts) or (nbounds,)")
    if bool(peat_hydro):
        if k_lin_peat_layer1 is None:
            raise ValueError("peat k_litt requires peat_tables or k_lin_peat_layer1")
        k_lin_peat_layer1 = _as_1d("k_lin_peat_layer1", k_lin_peat_layer1)

    ratio = (litter.tmc_litter - litter.tmc_litter_res) / (litter.tmc_litter_sat - litter.tmc_litter_res)
    bin_i = jnp.where(
        litter.tmc_litter < litter.tmc_litter_res,
        imin,
        jnp.maximum(jnp.minimum(((imax - imin) * ratio).astype(jnp.int32) + imin, imax - 1), imin),
    )
    offsets = bin_i - imin
    point_index = jnp.arange(npts)[:, None]
    mineral_k = k_lin_layer1[offsets, point_index] * ks_point[:, None]
    k_tmp = mineral_k
    if bool(peat_hydro):
        peat_tile_mask = jnp.asarray([(jst + 1) in tuple(int(tile) for tile in peat_tiles) for jst in range(nstm)])
        peat_k = k_lin_peat_layer1[offsets] * PEAT_KS
        k_tmp = jnp.where(peat_tile_mask[None, :], peat_k, mineral_k)
    k_litt = jnp.sum(vegtot[:, None] * soiltile * jnp.sqrt(jnp.maximum(k_tmp, 0.0)), axis=1)

    litterhumdiag = jnp.sum(litter.soil_wet_litter * soiltile, axis=1)
    wet_mean = jnp.sum(litter.tmc_litter_awet * soiltile, axis=1)
    dry_mean = jnp.sum(litter.tmc_litter_adry * soiltile, axis=1)
    litter_mean = jnp.sum(litter.tmc_litter * soiltile, axis=1)
    drysoil_frac = jnp.where(
        wet_mean - dry_mean > 0.0,
        1.0 + jnp.maximum(jnp.minimum((dry_mean - litter_mean) / (wet_mean - dry_mean), 0.0), -1.0),
        0.0,
    )
    return HydrolLitterGridDiagnostics(
        k_litt=k_litt,
        litterhumdiag=litterhumdiag,
        drysoil_frac=drysoil_frac,
    )


def hydrol_module_diagnostics(
    result: HydrolModuleExplicitResult,
    *,
    nroot,
    dz_mm,
    dh_mm,
    njsc,
    veget_max=None,
    soiltile=None,
    vegtot=None,
    vegtot_old=None,
    mcs=None,
    mcf=None,
    mcw=None,
    mcs_mineral=USDA_MCS,
    mcf_mineral=USDA_MCF,
    mcw_mineral=USDA_MCW,
    pcent=None,
    mineral_tables: MineralCWRRTables | None = None,
    peat_tables: PeatCWRRTables | None = None,
    ks=USDA_KS,
    vevapnu=None,
    tot_melt=None,
    irrigation=None,
    returnflow=None,
    reinfiltration=None,
    ok_freeze_cwrr=False,
    peat_hydro=False,
    peat_hydro_water_stress=None,
    perma_peat=None,
    peat_tiles=(4, 5, 6),
    agri_peat=False,
    tides=False,
    dyn_nroot_larix=False,
    is_tree=None,
    znt=None,
    new_watstress=False,
    alpha_watstress=1.0,
    evap_bare_limit_alt: HydrolAltResidualSolveResult | None = None,
    tmcint=None,
    evapot=None,
    do_rsoil=False,
    liqlayers=10,
    numlayers=11,
    topmodel_new=False,
    temp_hydro=None,
    param_vp=None,
    param_kp=None,
    param_qp=None,
    param_fmax=None,
    undef_sechiba=1.0e20,
    zero_celsius=273.15,
    min_sechiba=1.0e-8,
):
    """Build HYDROL module diagnostics from a completed explicit soil step.

    Fortran provenance: ``hydrol_soil`` lines 6395-6741 computes the per-tile
    soil moisture, water-stress, and thermosoil exports after the prognostic
    solve; ``hydrol_diag_soil`` lines 9089-9284 aggregates stress,
    ``soilmoist``, ``shumdiag``, and peat/man/croppeat diagnostics for
    downstream SECHIBA/STOMATE. Inputs that are Fortran module state
    (``nroot``, ``dh``, thresholds, litter conductivity tables, and optional
    dynamic-root controls) remain explicit here.
    """

    mc = result.soil.mc
    mcl = result.soil.mcl
    if veget_max is None:
        veget_max = result.vegupd.vegetmax_soil.sum(axis=2)
    if soiltile is None:
        raise ValueError("soiltile is required for HYDROL diagnostic aggregation")
    if vegtot is None:
        raise ValueError("vegtot is required for HYDROL diagnostic aggregation")
    veget_max = _as_2d("veget_max", veget_max)
    soiltile = _as_2d("soiltile", soiltile)
    vegtot = _as_1d("vegtot", vegtot)
    nroot = _as_3d("nroot", nroot)
    _require_same_npts(mc, mcl, veget_max, soiltile, vegtot, nroot)
    if mc.shape[2] != soiltile.shape[1] or nroot.shape[:2] != veget_max.shape or nroot.shape[2] != mc.shape[1]:
        raise ValueError("diagnostic inputs must match mc/mcl, soiltile, veget_max, and nroot dimensions")

    npts, nslm, nstm = mc.shape
    layer_by_tile = []
    stress_by_tile = []
    nroot_work = nroot
    if peat_hydro_water_stress is None:
        peat_hydro_water_stress = peat_hydro

    for jst in range(nstm):
        layer = hydrol_soil_layer_diagnostics(
            mc=mc[:, :, jst],
            mcl=mcl[:, :, jst],
            dz_mm=dz_mm,
            njsc=njsc,
            mcs=mcs,
            mcf=mcf,
            mcw=mcw,
            mcs_mineral=mcs_mineral,
            mcf_mineral=mcf_mineral,
            mcw_mineral=mcw_mineral,
            pcent=pcent,
            peat_hydro=peat_hydro,
            soil_tile_index=jst + 1,
            peat_tiles=peat_tiles,
        )
        stress_layer = layer
        if bool(peat_hydro) and not bool(peat_hydro_water_stress) and (jst + 1) in tuple(int(tile) for tile in peat_tiles):
            stress_layer = hydrol_soil_layer_diagnostics(
                mc=mc[:, :, jst],
                mcl=mcl[:, :, jst],
                dz_mm=dz_mm,
                njsc=njsc,
                mcs=mcs,
                mcf=mcf,
                mcw=mcw,
                mcs_mineral=mcs_mineral,
                mcf_mineral=mcf_mineral,
                mcw_mineral=mcw_mineral,
                pcent=pcent,
                peat_hydro=False,
                soil_tile_index=jst + 1,
                peat_tiles=peat_tiles,
            )
        is_under = result.soil.is_under_mcr[:, jst] if result.soil.is_under_mcr.shape[1] > jst else jnp.zeros((npts,), dtype=bool)
        stress = hydrol_water_stress_diagnostics(
            layer=stress_layer,
            nroot=nroot_work,
            vegetmax_soil_tile=result.vegupd.vegetmax_soil[:, :, jst],
            is_under_mcr=is_under,
            njsc=njsc,
            pcent=pcent,
            pcent_mineral=USDA_PCENT,
            peat_hydro=peat_hydro_water_stress,
            soil_tile_index=jst + 1,
            peat_tiles=peat_tiles,
            dyn_nroot_larix=dyn_nroot_larix,
            is_tree=is_tree,
            znt=znt,
            new_watstress=new_watstress,
            alpha_watstress=alpha_watstress,
            min_sechiba=min_sechiba,
        )
        if stress_layer is not layer:
            stress = stress._replace(soil_wet_ns=layer.soil_wet_ns)
        nroot_work = stress.nroot
        layer_by_tile.append(layer)
        stress_by_tile.append(stress)

    us = jnp.stack(tuple(stress.us for stress in stress_by_tile), axis=2)
    humrelv = jnp.stack(tuple(stress.humrelv for stress in stress_by_tile), axis=2)
    vegstressv = jnp.stack(tuple(stress.vegstressv for stress in stress_by_tile), axis=2)
    soil_wet_ns = jnp.stack(tuple(stress.soil_wet_ns for stress in stress_by_tile), axis=2)
    undermcr = jnp.sum(jnp.stack(tuple(stress.undermcr_increment for stress in stress_by_tile), axis=1), axis=1)

    stress_agg = hydrol_stress_aggregates(
        humrelv=humrelv,
        vegstressv=vegstressv,
        veget_max=veget_max,
        min_sechiba=min_sechiba,
    )
    profil_froz_hydro = None
    if bool(ok_freeze_cwrr):
        profil_ns = _as_3d("profil_froz_hydro_ns", result.soil.profil_froz_hydro_ns)
        if profil_ns.shape != mc.shape:
            raise ValueError("profil_froz_hydro_ns must match HYDROL soil state")
        mask_vegtot = (vegtot > min_sechiba).astype(profil_ns.dtype)
        weighted_tiles = mask_vegtot[:, None, None] * vegtot[:, None, None] * soiltile[:, None, :]
        profil_froz_hydro = jnp.sum(weighted_tiles * profil_ns, axis=2)
    moisture = hydrol_soilmoist_aggregates(
        mc=mc,
        mcl=mcl,
        soiltile=soiltile,
        vegtot=vegtot,
        vegtot_old=vegtot_old,
        dz_mm=dz_mm,
    )
    shum = hydrol_shumdiag(
        mc=mc,
        soil_wet_ns=soil_wet_ns,
        soilmoist=moisture.soilmoist,
        soiltile=soiltile,
        dh_mm=dh_mm,
        dz_mm=dz_mm,
        njsc=njsc,
        mcs=mcs,
        mcf=mcf,
        mcw=mcw,
        mcs_mineral=mcs_mineral,
        peat_hydro=peat_hydro,
        agri_peat=agri_peat,
        tides=tides,
    )
    litter = hydrol_litter_top_diagnostics(
        mc=mc,
        dz_mm=dz_mm,
        dh_mm=dh_mm,
        njsc=njsc,
        mcs=mcs,
        mcf=mcf,
        mcw=mcw,
        peat_hydro=peat_hydro,
        perma_peat=perma_peat,
        agri_peat=agri_peat,
        tides=tides,
        peat_tiles=peat_tiles,
    )
    peat_water_table = hydrol_peat_water_table_diagnostics(
        mc=result.soil.mc,
        mcl=result.soil.mcl,
        soiltile=soiltile,
        dz_mm=dz_mm,
        dh_mm=dh_mm,
        wt_ab=result.soil.wt_ab,
        wt_ab_tide=result.soil.wt_ab_tide,
        peat_hydro=peat_hydro,
        tides=tides,
        liqlayers=liqlayers,
        numlayers=numlayers,
        topmodel_new=topmodel_new,
        temp_hydro=temp_hydro,
        mcs=mcs,
        param_vp=param_vp,
        param_kp=param_kp,
        param_qp=param_qp,
        param_fmax=param_fmax,
        undef_sechiba=undef_sechiba,
        zero_celsius=zero_celsius,
        min_sechiba=min_sechiba,
    )
    litter_grid = hydrol_litter_grid_diagnostics(
        litter=litter,
        soiltile=soiltile,
        vegtot=vegtot,
        njsc=njsc,
        mineral_tables=mineral_tables,
        peat_tables=peat_tables,
        ks=ks,
        peat_hydro=peat_hydro,
        peat_tiles=peat_tiles,
    )
    if vevapnu is None:
        vevapnu = jnp.zeros_like(vegtot)
    if tot_melt is None:
        tot_melt = jnp.zeros_like(vegtot)
    if irrigation is None:
        irrigation = jnp.zeros_like(vegtot)
    if returnflow is None:
        returnflow = jnp.zeros_like(vegtot)
    if reinfiltration is None:
        reinfiltration = jnp.zeros_like(vegtot)
    grid_flux = hydrol_grid_flux_aggregates(
        ae_ns=result.split.ae_ns,
        dr_ns=result.soil.dr_ns,
        ru_ns=result.soil.ru_ns,
        tmc=result.soil.tmc,
        soiltile=soiltile,
        vegtot=vegtot,
        frac_bare_ns=result.vegupd.frac_bare_ns,
        vevapnu=vevapnu,
        subsinksoil=result.flood.subsinksoil,
        tot_melt=tot_melt,
        irrigation=irrigation,
        returnflow=returnflow,
        reinfiltration=reinfiltration,
        mask_soiltile=result.vegupd.mask_soiltile,
        min_sechiba=min_sechiba,
    )

    evap_bare_limit = None
    evap_bare_limit_grid = None
    if evap_bare_limit_alt is not None:
        missing = [
            name
            for name, value in (
                ("tmcint", tmcint),
                ("evapot", evapot),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "evap_bare_limit_alt requires explicit Fortran state: " + ", ".join(missing)
            )
        evap_bare_limit_result = hydrol_evap_bare_limit_diagnostic(
            tmcint=tmcint,
            tmc_dummy=evap_bare_limit_alt.tmc,
            flux_bottom=evap_bare_limit_alt.flux_bottom,
            mask_soiltile=result.vegupd.mask_soiltile,
            soiltile=soiltile,
            frac_bare_ns=result.vegupd.frac_bare_ns,
            vegtot=vegtot,
            evapot=evapot,
            tmc_litter=litter.tmc_litter,
            tmc_litter_wilt=litter.tmc_litter_wilt,
            tmc_litter_res=litter.tmc_litter_res,
            is_under_mcr=result.soil.is_under_mcr,
            do_rsoil=do_rsoil,
            min_sechiba=min_sechiba,
        )
        evap_bare_limit = evap_bare_limit_result.evap_bare_lim_ns
        evap_bare_limit_grid = evap_bare_limit_result.evap_bare_lim

    return HydrolModuleDiagnostics(
        layer_by_tile=tuple(layer_by_tile),
        stress_by_tile=tuple(stress_by_tile),
        nroot=nroot_work,
        soil_wet_ns=soil_wet_ns,
        us=us,
        humrelv=humrelv,
        vegstressv=vegstressv,
        humrel=stress_agg.humrel,
        vegstress=stress_agg.vegstress,
        profil_froz_hydro=profil_froz_hydro,
        soilmoist=moisture.soilmoist,
        soilmoist_liquid=moisture.soilmoist_liquid,
        mc_layh=moisture.mc_layh,
        mcl_layh=moisture.mcl_layh,
        mc_layh_s=moisture.mc_layh_s,
        mcl_layh_s=moisture.mcl_layh_s,
        shumdiag=shum.shumdiag,
        shumdiag_perma=shum.shumdiag_perma,
        shumdiag_peat=shum.shumdiag_peat,
        shumdiag_croppeat=shum.shumdiag_croppeat,
        shumdiag_man=shum.shumdiag_man,
        undermcr=undermcr,
        tmc_litter=litter.tmc_litter,
        soil_wet_litter=litter.soil_wet_litter,
        tmc_trampling=litter.tmc_trampling,
        tmc_topgrass=litter.tmc_topgrass,
        liqwt_ratio=peat_water_table.liqwt_ratio,
        wtp=peat_water_table.wtp,
        fwet_new=peat_water_table.fwet_new,
        mc_peat_above=litter.mc_peat_above,
        mc_croppeat_above=litter.mc_croppeat_above,
        mc_man_above=litter.mc_man_above,
        k_litt=litter_grid.k_litt,
        litterhumdiag=litter_grid.litterhumdiag,
        drysoil_frac=litter_grid.drysoil_frac,
        runoff=grid_flux.runoff,
        drainage=grid_flux.drainage,
        humtot=grid_flux.humtot,
        vevapnu=grid_flux.vevapnu,
        evap_bare_lim=evap_bare_limit_grid,
        ae_ns=grid_flux.ae_ns,
        evap_bare_lim_ns=evap_bare_limit,
    )


def hydrol_diag_soil_source_owner(
    result: HydrolModuleExplicitResult,
    **diagnostic_inputs,
) -> HydrolModuleDiagnostics:
    """Compose the exact kernels covering ``hydrol_diag_soil`` lines 9006-9284.

    This owner deliberately contains no duplicate equations. It fixes the
    source-order composition boundary around grid flux aggregation, stress,
    litter conductivity, layer moisture and peat/tide humidity diagnostics.
    """

    return hydrol_module_diagnostics(result, **diagnostic_inputs)


def hydrol_soil_tail_diagnostics(
    *,
    soiltile,
    vegtot,
    wtd_ns,
    ru_corr_ns,
    ru_corr2_ns,
    dr_corr_ns,
    dr_corrnum_ns,
    dr_force_ns,
    ru_infilt_ns,
    qinfilt_ns,
    evap_bare_lim_ns,
    r_soil_ns,
    mc,
    dr_ns,
    ru_ns,
    qflux,
    check_infilt_ns=None,
    check_tr_ns=None,
    check_over_ns=None,
    check_under_ns=None,
    check_cwrr2=False,
    tides=False,
    wtp_tide=None,
    dwtp_tide=None,
    error=False,
    min_sechiba=1.0e-8,
) -> HydrolSoilTailDiagnostics:
    """Close ``hydrol_soil`` pre-loop forcing and post-loop diagnostic spans.

    Fortran provenance: ``hydrol.f90::hydrol_soil`` lines 5646-5655,
    6801-6857 and 7347-7398. Tidal values are explicit forcing-boundary
    inputs replacing Fortran units 103/104; this function never performs file
    IO. The fatal aggregate ``error`` guard is preserved.
    """

    if bool(tides) and (wtp_tide is None or dwtp_tide is None):
        raise NotImplementedError("hydrol_soil lines 5646-5655 require explicit tidal forcing values")
    if wtp_tide is None:
        wtp_tide = 0.0
    if dwtp_tide is None:
        dwtp_tide = 0.0

    soiltile = _as_2d("soiltile", soiltile)
    vegtot = _as_1d("vegtot", vegtot)
    wtd_ns = _as_2d("wtd_ns", wtd_ns)
    npts, nstm = soiltile.shape
    tile_arrays = {
        name: _as_2d(name, value)
        for name, value in {
            "ru_corr_ns": ru_corr_ns,
            "ru_corr2_ns": ru_corr2_ns,
            "dr_corr_ns": dr_corr_ns,
            "dr_corrnum_ns": dr_corrnum_ns,
            "dr_force_ns": dr_force_ns,
            "ru_infilt_ns": ru_infilt_ns,
            "qinfilt_ns": qinfilt_ns,
            "evap_bare_lim_ns": evap_bare_lim_ns,
            "r_soil_ns": r_soil_ns,
            "dr_ns": dr_ns,
            "ru_ns": ru_ns,
        }.items()
    }
    if wtd_ns.shape != (npts, nstm) or any(value.shape != (npts, nstm) for value in tile_arrays.values()):
        raise ValueError("hydrol_soil tail tile arrays must match soiltile shape")
    mc = _as_3d("mc", mc)
    qflux = _as_3d("qflux", qflux)
    if mc.shape[0] != npts or mc.shape[2] != nstm or qflux.shape != mc.shape:
        raise ValueError("mc and qflux must share (npts,nslm,nstm)")

    active = (vegtot > min_sechiba).astype(soiltile.dtype)
    weighted = active[:, None] * vegtot[:, None] * soiltile
    wtd = jnp.sum(soiltile * wtd_ns, axis=1)
    ru_corr = jnp.sum(weighted * tile_arrays["ru_corr_ns"], axis=1)
    ru_corr2 = jnp.sum(weighted * tile_arrays["ru_corr2_ns"], axis=1)
    dr_corr = jnp.sum(weighted * tile_arrays["dr_corr_ns"], axis=1)
    dr_corrnum = jnp.sum(weighted * tile_arrays["dr_corrnum_ns"], axis=1)
    dr_force = -jnp.sum(weighted * tile_arrays["dr_force_ns"], axis=1)
    ru_infilt = jnp.sum(weighted * tile_arrays["ru_infilt_ns"], axis=1)
    qinfilt = jnp.sum(weighted * tile_arrays["qinfilt_ns"], axis=1)

    def _check_aggregate(name, value):
        if not bool(check_cwrr2):
            return jnp.zeros((npts,), dtype=soiltile.dtype)
        if value is None:
            raise ValueError(f"check_cwrr2 requires {name}")
        value = _as_2d(name, value)
        if value.shape != (npts, nstm):
            raise ValueError(f"{name} must match soiltile shape")
        return jnp.sum(weighted * value, axis=1)

    check_infilt = _check_aggregate("check_infilt_ns", check_infilt_ns)
    check_tr = _check_aggregate("check_tr_ns", check_tr_ns)
    check_over = _check_aggregate("check_over_ns", check_over_ns)
    check_under = _check_aggregate("check_under_ns", check_under_ns)

    evap_ns = tile_arrays["evap_bare_lim_ns"]
    evap_grid = jnp.sum(evap_ns * vegtot[:, None] * soiltile, axis=1)
    evap_ns = jnp.where((evap_grid < 1.0e-3)[:, None], 0.0, evap_ns)
    evap_grid = jnp.sum(evap_ns * vegtot[:, None] * soiltile, axis=1)
    r_soil = jnp.sum(tile_arrays["r_soil_ns"] * vegtot[:, None] * soiltile, axis=1)

    if bool(jnp.any(jnp.asarray(error))):
        raise RuntimeError("hydrol_soil lines 7392-7398: one or more fatal errors were detected")

    return HydrolSoilTailDiagnostics(
        wtp_tide=wtp_tide,
        dwtp_tide=dwtp_tide,
        wtd=wtd,
        ru_corr=ru_corr,
        ru_corr2=ru_corr2,
        dr_corr=dr_corr,
        dr_corrnum=dr_corrnum,
        dr_force=dr_force,
        ru_infilt=ru_infilt,
        qinfilt=qinfilt,
        check_infilt=check_infilt,
        check_tr=check_tr,
        check_over=check_over,
        check_under=check_under,
        evap_bare_lim_ns=evap_ns,
        evap_bare_lim=evap_grid,
        r_soil=r_soil,
        soil_mc=mc,
        drainage_per_soil=tile_arrays["dr_ns"],
        runoff_per_soil=tile_arrays["ru_ns"],
        wat_flux=qflux,
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_soil lines 5646-5655",
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_soil lines 6801-6857",
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_soil lines 7347-7398",
        ),
    )


def _texture_lookup(name, values, njsc, npts):
    values = jnp.asarray(values)
    njsc = jnp.asarray(njsc, dtype=jnp.int32).reshape(-1)
    if njsc.shape[0] == 1:
        njsc = jnp.repeat(njsc, npts)
    if njsc.shape[0] != npts:
        raise ValueError(f"{name} lookup requires njsc to be scalar or have shape (npts,)")
    return values[njsc - 1]


def _per_point_threshold(name, value, njsc, npts, defaults):
    if value is None:
        return _texture_lookup(name, defaults, njsc, npts)
    array = _as_1d(name, value)
    if array.shape[0] == 1:
        array = jnp.repeat(array, npts)
    if array.shape[0] == len(defaults) and npts != len(defaults):
        array = _texture_lookup(name, array, njsc, npts)
    if array.shape[0] != npts:
        raise ValueError(f"{name} must be scalar, per-point, or a texture table")
    return array


def hydrol_canop_interception(
    *,
    precip_rain,
    vevapwet,
    veget_max,
    veget,
    qsintmax,
    qsintveg,
    tot_melt,
    vegtot,
    throughfall_by_pft,
    min_sechiba=1.0e-8,
):
    """Update canopy interception and throughfall as ``hydrol_canop``.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_canop``, lines 4992-5117. The helper receives
    ``vegtot`` and ``throughfall_by_pft`` explicitly because they are Fortran
    module state/parameter arrays, not local arguments of ``hydrol_canop``.
    """

    precip_rain = _as_1d("precip_rain", precip_rain)
    vevapwet = _as_2d("vevapwet", vevapwet)
    veget_max = _as_2d("veget_max", veget_max)
    veget = _as_2d("veget", veget)
    qsintmax = _as_2d("qsintmax", qsintmax)
    qsintveg = _as_2d("qsintveg", qsintveg)
    tot_melt = _as_1d("tot_melt", tot_melt)
    vegtot = _as_1d("vegtot", vegtot)
    throughfall_by_pft = _as_1d("throughfall_by_pft", throughfall_by_pft)
    _require_same_npts(precip_rain, vevapwet, veget_max, veget, qsintmax, qsintveg, tot_melt, vegtot)
    if not (vevapwet.shape == veget_max.shape == veget.shape == qsintmax.shape == qsintveg.shape):
        raise ValueError("PFT arrays must share shape (npts, nvm)")
    if throughfall_by_pft.shape[0] != qsintveg.shape[1]:
        raise ValueError("throughfall_by_pft length must match nvm")

    nvm = qsintveg.shape[1]
    zero = jnp.asarray(0.0, dtype=qsintveg.dtype)
    rain = precip_rain[:, None]
    throughfall = throughfall_by_pft[None, :]

    qs_work = qsintveg.at[:, 1:nvm].add(-vevapwet[:, 1:nvm])
    intercepted = veget * (1.0 - throughfall) * rain
    qs_work = qs_work.at[:, 0].set(zero)
    qs_work = qs_work.at[:, 1:nvm].add(intercepted[:, 1:nvm])

    precip2canopy = intercepted.at[:, 0].set(zero)
    precisol = jnp.zeros_like(qsintveg)
    precip2ground = jnp.zeros_like(qsintveg)
    canopy2ground = jnp.zeros_like(qsintveg)

    precisol = precisol.at[:, 0].set(veget_max[:, 0] * precip_rain)
    precip2ground = precip2ground.at[:, 0].set(precisol[:, 0])

    zqsintvegnew = jnp.minimum(qs_work, qsintmax)
    bare_gap = (veget_max - veget) * rain
    pft_precisol = veget * throughfall * rain + qs_work - zqsintvegnew + bare_gap
    pft_precip2ground = veget * throughfall * rain + bare_gap
    precisol = precisol.at[:, 1:nvm].set(pft_precisol[:, 1:nvm])
    precip2ground = precip2ground.at[:, 1:nvm].set(pft_precip2ground[:, 1:nvm])
    canopy2ground = canopy2ground.at[:, 1:nvm].set(
        jnp.maximum(precisol[:, 1:nvm] - precip2ground[:, 1:nvm], zero)
    )

    melt_share = jnp.where(
        vegtot[:, None] > jnp.asarray(min_sechiba, dtype=qsintveg.dtype),
        tot_melt[:, None] * veget_max / vegtot[:, None],
        zero,
    )
    precisol = precisol + melt_share
    precip2ground = precip2ground + melt_share
    qs_out = qs_work.at[:, 1:nvm].set(zqsintvegnew[:, 1:nvm])

    return HydrolCanopResult(
        qsintveg=qs_out,
        precisol=precisol,
        precip2canopy=precip2canopy,
        precip2ground=precip2ground,
        canopy2ground=canopy2ground,
    )


def hydrol_flood_reservoir(*, vevapflo, flood_frac, flood_res, precisol, subsinksoil):
    """Update floodplain evaporation reservoir and throughfall scaling.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_flood``, lines 5276-5334. ``precisol`` and
    ``subsinksoil`` are Fortran module variables; this helper takes and returns
    them explicitly to avoid hidden state.
    """

    vevapflo = _as_1d("vevapflo", vevapflo)
    flood_frac = _as_1d("flood_frac", flood_frac)
    flood_res = _as_1d("flood_res", flood_res)
    precisol = _as_2d("precisol", precisol)
    subsinksoil = _as_1d("subsinksoil", subsinksoil)
    _require_same_npts(vevapflo, flood_frac, flood_res, precisol, subsinksoil)

    temp = jnp.minimum(flood_res, vevapflo)
    flood_res_out = flood_res - temp
    subsinksoil_out = subsinksoil + vevapflo - temp
    vevapflo_out = temp
    floodout = vevapflo_out - flood_frac * jnp.sum(precisol, axis=1)
    precisol_out = precisol * (1.0 - flood_frac[:, None])

    return HydrolFloodResult(
        vevapflo=vevapflo_out,
        flood_res=flood_res_out,
        floodout=floodout,
        precisol=precisol_out,
        subsinksoil=subsinksoil_out,
    )


def _hydrol_tmc_source_order(mc, dz_mm):
    """Integrate ``mc`` with the statement order used by ``hydrol_tmc_update``."""

    nslm = mc.shape[1]
    value = dz_mm[1] * (3.0 * mc[:, 0, :] + mc[:, 1, :]) / 8.0
    for jsl in range(1, nslm - 1):
        value = value + dz_mm[jsl] * (3.0 * mc[:, jsl, :] + mc[:, jsl - 1, :]) / 8.0
        value = value + dz_mm[jsl + 1] * (3.0 * mc[:, jsl, :] + mc[:, jsl + 1, :]) / 8.0
    return value + dz_mm[-1] * (3.0 * mc[:, -1, :] + mc[:, -2, :]) / 8.0


def hydrol_tmc_update(
    *,
    veget_max,
    soiltile,
    qsintveg,
    mc,
    water2infilt,
    tmc,
    resdist,
    vegtot_old,
    vegtot,
    pref_soil_veg,
    dz_mm,
    min_sechiba=1.0e-8,
    check_cwrr=False,
    allowed_err=0.0,
    raise_on_water_balance_error=True,
):
    """Migrate HYDROL water state after vegetation fractions change.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_tmc_update``, lines 3314-3588. The implementation
    follows source order through canopy-water recovery (3355-3369), changes
    in vegetated grid-cell fraction (3371-3425), soil-tile redistribution
    (3427-3510), state writeback (3512-3532), the optional water-balance check
    (3534-3572), and ``resdist`` freshness update (3574-3575).

    ``pref_soil_veg`` retains the Fortran one-based tile convention. The two
    update flags are zero-dimensional JAX booleans because the Fortran source
    decides whether to enter each migration block across the complete local
    domain, rather than independently for each land point.
    """

    veget_max = _as_2d("veget_max", veget_max)
    soiltile = _as_2d("soiltile", soiltile)
    qsintveg_state = _as_2d("qsintveg", qsintveg)
    mc_state = _as_3d("mc", mc)
    water2infilt_state = _as_2d("water2infilt", water2infilt)
    tmc_before = _as_2d("tmc", tmc)
    resdist_before = _as_2d("resdist", resdist)
    vegtot_old = _as_1d("vegtot_old", vegtot_old)
    vegtot = _as_1d("vegtot", vegtot)
    dz_mm = _as_1d("dz_mm", dz_mm)
    _require_same_npts(
        veget_max,
        soiltile,
        qsintveg_state,
        mc_state,
        water2infilt_state,
        tmc_before,
        resdist_before,
        vegtot_old,
        vegtot,
    )
    npts, nvm = veget_max.shape
    nstm = soiltile.shape[1]
    nslm = mc_state.shape[1]
    if qsintveg_state.shape != (npts, nvm):
        raise ValueError("qsintveg must match veget_max shape (npts,nvm)")
    if mc_state.shape != (npts, nslm, nstm):
        raise ValueError("mc must have shape (npts,nslm,nstm) matching soiltile")
    for name, value in (
        ("water2infilt", water2infilt_state),
        ("tmc", tmc_before),
        ("resdist", resdist_before),
    ):
        if value.shape != (npts, nstm):
            raise ValueError(f"{name} must match soiltile shape (npts,nstm)")
    if dz_mm.shape[0] != nslm or nslm < 2:
        raise ValueError("dz_mm must match the mc layer axis, with at least two layers")
    pref = np.asarray(pref_soil_veg, dtype=np.int32)
    if pref.ndim != 1 or pref.shape[0] != nvm:
        raise ValueError("pref_soil_veg must be one-dimensional with length nvm")
    if np.any(pref < 1) or np.any(pref > nstm):
        raise ValueError("pref_soil_veg must contain Fortran one-based soil-tile indices")

    dtype = jnp.result_type(
        veget_max,
        soiltile,
        qsintveg_state,
        mc_state,
        water2infilt_state,
        tmc_before,
        resdist_before,
        vegtot_old,
        vegtot,
        dz_mm,
    )
    minv = jnp.asarray(min_sechiba, dtype=dtype)
    zero = jnp.asarray(0.0, dtype=dtype)
    drain_upd = jnp.zeros((npts,), dtype=dtype)
    runoff_upd = jnp.zeros((npts,), dtype=dtype)

    qsintveg_old = qsintveg_state if check_cwrr else None
    for jv, jst_fortran in enumerate(pref.tolist()):
        jst = jst_fortran - 1
        disappeared = (
            (vegtot_old > minv)
            & (veget_max[:, jv] < minv)
            & (qsintveg_state[:, jv] > zero)
        )
        recovered = qsintveg_state[:, jv] / (resdist_before[:, jst] * vegtot_old)
        water2infilt_state = water2infilt_state.at[:, jst].add(
            jnp.where(disappeared, recovered, zero)
        )
        qsintveg_state = qsintveg_state.at[:, jv].set(
            jnp.where(disappeared, zero, qsintveg_state[:, jv])
        )

    delvegtot = vegtot - vegtot_old
    vegtot_updated = jnp.sum(jnp.abs(delvegtot)) > zero
    for jst in range(nstm):
        increases = delvegtot > minv
        increase_scale = vegtot_old / jnp.where(increases, vegtot, 1.0)
        mc_state = mc_state.at[:, :, jst].set(
            jnp.where(increases[:, None], mc_state[:, :, jst] * increase_scale[:, None], mc_state[:, :, jst])
        )
        water2infilt_state = water2infilt_state.at[:, jst].set(
            jnp.where(increases, water2infilt_state[:, jst] * increase_scale, water2infilt_state[:, jst])
        )

        positive_vegtot = vegtot > minv
        decrease_scale = (vegtot_old - vegtot) / jnp.where(positive_vegtot, vegtot, 1.0)
        mcaux = jnp.where(
            positive_vegtot[:, None],
            mc_state[:, :, jst] * decrease_scale[:, None],
            mc_state[:, :, jst],
        )
        drain_candidate = _hydrol_tmc_source_order(mcaux[:, :, None], dz_mm)[:, 0]
        runoff_candidate = jnp.where(
            positive_vegtot,
            water2infilt_state[:, jst] * decrease_scale,
            water2infilt_state[:, jst],
        )
        apply_decrease_arm = vegtot_updated & ~increases
        drain_upd = drain_upd + jnp.where(apply_decrease_arm, drain_candidate, zero)
        runoff_upd = runoff_upd + jnp.where(apply_decrease_arm, runoff_candidate, zero)

    vmr = soiltile - resdist_before
    soil_updated = jnp.sum(jnp.abs(vmr)) > zero
    vmr_sum = jnp.sum(jnp.where(vmr < zero, vmr, zero), axis=1)
    mc_dilu = jnp.zeros((npts, nslm), dtype=dtype)
    infil_dilu = jnp.zeros((npts,), dtype=dtype)
    for jst in range(nstm):
        shrinking = vmr[:, jst] < -minv
        share = vmr[:, jst] / jnp.where(shrinking, vmr_sum, 1.0)
        mc_dilu = mc_dilu + jnp.where(shrinking[:, None], mc_state[:, :, jst] * share[:, None], zero)
        infil_dilu = infil_dilu + jnp.where(shrinking, water2infilt_state[:, jst] * share, zero)

    for jst in range(nstm):
        growing = vmr[:, jst] > minv
        safe_soiltile = jnp.where(growing, soiltile[:, jst], 1.0)
        redistributed_mc = (
            mc_state[:, :, jst] * resdist_before[:, jst, None]
            + mc_dilu * vmr[:, jst, None]
        ) / safe_soiltile[:, None]
        redistributed_infil = (
            water2infilt_state[:, jst] * resdist_before[:, jst]
            + infil_dilu * vmr[:, jst]
        ) / safe_soiltile
        mc_state = mc_state.at[:, :, jst].set(
            jnp.where((soil_updated & growing)[:, None], redistributed_mc, mc_state[:, :, jst])
        )
        water2infilt_state = water2infilt_state.at[:, jst].set(
            jnp.where(soil_updated & growing, redistributed_infil, water2infilt_state[:, jst])
        )
        deleted = soil_updated & (soiltile[:, jst] < minv)
        mc_state = mc_state.at[:, :, jst].set(jnp.where(deleted[:, None], zero, mc_state[:, :, jst]))
        water2infilt_state = water2infilt_state.at[:, jst].set(
            jnp.where(deleted, zero, water2infilt_state[:, jst])
        )

    tmc_state = _hydrol_tmc_source_order(mc_state, dz_mm) + water2infilt_state
    humtot = jnp.zeros((npts,), dtype=dtype)
    for jst in range(nstm):
        humtot = humtot + vegtot * soiltile[:, jst] * tmc_state[:, jst]

    water_balance_error = None
    water_balance_failed = None
    if check_cwrr:
        old_total = jnp.sum(tmc_before * resdist_before * vegtot_old[:, None], axis=1)
        old_canopy = jnp.sum(qsintveg_old, axis=1)
        new_total = jnp.sum(tmc_state * soiltile * vegtot[:, None], axis=1)
        new_canopy = jnp.sum(qsintveg_state, axis=1)
        water_balance_error = new_total - old_total + new_canopy - old_canopy + drain_upd + runoff_upd
        water_balance_failed = jnp.abs(water_balance_error) > 10.0 * jnp.asarray(allowed_err, dtype=dtype)
        if raise_on_water_balance_error and bool(np.any(np.asarray(water_balance_failed))):
            raise RuntimeError("hydrol_tmc_update: Fortran water-balance tolerance exceeded")

    return HydrolTmcUpdateResult(
        mc=mc_state,
        water2infilt=water2infilt_state,
        qsintveg=qsintveg_state,
        drain_upd=drain_upd,
        runoff_upd=runoff_upd,
        tmc=tmc_state,
        humtot=humtot,
        resdist=soiltile,
        water_balance_error=water_balance_error,
        water_balance_failed=water_balance_failed,
        vegtot_updated=vegtot_updated,
        soil_updated=soil_updated,
    )


def hydrol_vegupd_static_state(
    *,
    veget,
    veget_max,
    soiltile,
    vegtot,
    pref_soil_veg,
    min_sechiba=1.0e-8,
):
    """Compute the ``hydrol_vegupd`` masks used by later HYDROL kernels.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_vegupd``, lines 5191-5248. This starts after the
    stateful ``hydrol_tmc_update`` call; reservoir redistribution and
    ``drain_upd``/``runoff_upd`` are intentionally outside this helper.
    """

    veget = _as_2d("veget", veget)
    veget_max = _as_2d("veget_max", veget_max)
    soiltile = _as_2d("soiltile", soiltile)
    vegtot = _as_1d("vegtot", vegtot)
    pref_soil_veg_np = np.asarray(pref_soil_veg, dtype=np.int32)
    _require_same_npts(veget, veget_max, soiltile, vegtot)
    if veget.shape != veget_max.shape:
        raise ValueError("veget and veget_max must share shape (npts, nvm)")
    if pref_soil_veg_np.ndim != 1 or pref_soil_veg_np.shape[0] != veget.shape[1]:
        raise ValueError("pref_soil_veg must be one-dimensional with length nvm")
    if bool(np.any(pref_soil_veg_np < 1)) or bool(np.any(pref_soil_veg_np > soiltile.shape[1])):
        raise ValueError("pref_soil_veg must contain Fortran 1-based soil tile indices")
    pref_indices = tuple(int(value) - 1 for value in pref_soil_veg_np.tolist())

    npts, nvm = veget.shape
    nstm = soiltile.shape[1]
    minv = jnp.asarray(min_sechiba, dtype=veget.dtype)
    zero = jnp.asarray(0.0, dtype=veget.dtype)
    one = jnp.asarray(1.0, dtype=veget.dtype)
    mask_soiltile = jnp.where(soiltile > minv, one, zero)
    mask_veget = jnp.where(veget_max > minv, one, zero)

    vegetmax_soil = jnp.zeros((npts, nvm, nstm), dtype=veget.dtype)
    for jv in range(nvm):
        jst = pref_indices[jv]
        active = (mask_soiltile[:, jst] > zero) & (vegtot > minv)
        vegetmax_soil = vegetmax_soil.at[:, jv, jst].set(
            jnp.where(active, veget_max[:, jv] / jnp.where(soiltile[:, jst] > minv, soiltile[:, jst], 1.0), zero)
        )

    frac_bare = jnp.zeros_like(veget)
    frac_bare = frac_bare.at[:, 0].set(jnp.where(veget_max[:, 0] > minv, one, zero))
    for jv in range(1, nvm):
        frac_bare = frac_bare.at[:, jv].set(
            jnp.where(
                veget_max[:, jv] > minv,
                one - veget[:, jv] / jnp.where(veget_max[:, jv] > minv, veget_max[:, jv], 1.0),
                zero,
            )
        )

    frac_bare_ns = jnp.zeros((npts, nstm), dtype=veget.dtype)
    for jst in range(nstm):
        tile_sum = jnp.sum(vegetmax_soil[:, :, jst] * frac_bare, axis=1)
        frac_bare_ns = frac_bare_ns.at[:, jst].set(
            jnp.where(vegtot > minv, tile_sum / jnp.where(vegtot > minv, vegtot, 1.0), zero)
        )

    return HydrolVegupdStaticState(
        mask_veget=mask_veget,
        mask_soiltile=mask_soiltile,
        vegetmax_soil=vegetmax_soil,
        frac_bare=frac_bare,
        frac_bare_ns=frac_bare_ns,
    )


def hydrol_kfact_root_from_vegupd(
    *,
    vegetmax_soil,
    soiltile,
    pref_soil_veg,
    humcste,
    zz_mm,
    njsc,
    ks=USDA_KS,
    ks_peat=PEAT_KS,
    peat_hydro=False,
    peat_tiles=(4, 5),
    min_sechiba=1.0e-8,
):
    """Compute ``kfact_root`` after ``hydrol_vegupd``.

    Fortran provenance: ``hydrol.f90::hydrol_main`` lines 1210-1229
    initialize ``kfact_root`` to one and multiply by a root-depth conductivity
    factor for each PFT/soil-tile pair. The peat branch applies when
    ``peat_hydro`` is true and the Fortran soil tile is 4 or 5 in this source
    span.
    """

    vegetmax_soil = _as_3d("vegetmax_soil", vegetmax_soil)
    soiltile = _as_2d("soiltile", soiltile)
    pref_soil_veg = jnp.asarray(pref_soil_veg, dtype=jnp.int32)
    humcste = _as_1d("humcste", humcste)
    zz_mm = _as_1d("zz_mm", zz_mm)
    njsc = jnp.asarray(njsc, dtype=jnp.int32)
    if njsc.ndim == 0:
        njsc = njsc[None]
    npts, nvm, nstm = vegetmax_soil.shape
    nslm = int(zz_mm.shape[0])
    if soiltile.shape != (npts, nstm):
        raise ValueError("soiltile must have shape (npts,nstm)")
    if pref_soil_veg.shape != (nvm,) or humcste.shape != (nvm,):
        raise ValueError("pref_soil_veg and humcste must have shape (nvm,)")
    if njsc.shape != (npts,):
        raise ValueError("njsc must have shape (npts,)")
    if bool(jnp.any(pref_soil_veg < 1)) or bool(jnp.any(pref_soil_veg > nstm)):
        raise ValueError("pref_soil_veg must contain Fortran 1-based soil tile indices")

    ks_values = jnp.asarray(ks, dtype=jnp.float64)
    max_ks = jnp.max(ks_values)
    mineral_ks = ks_values[njsc - 1]
    peat_tile_set = {int(tile) for tile in peat_tiles}
    result = jnp.ones((npts, nslm, nstm), dtype=jnp.float64)
    zz_m = zz_mm / 1000.0
    for jv in range(1, nvm):
        jst_fortran = int(pref_soil_veg[jv])
        jst = jst_fortran - 1
        denom_ks = (
            jnp.full((npts,), float(ks_peat), dtype=jnp.float64)
            if bool(peat_hydro) and jst_fortran in peat_tile_set
            else mineral_ks
        )
        base = max_ks / denom_ks
        exponent = (
            -vegetmax_soil[:, jv, jst][:, None]
            / 2.0
            * (humcste[jv] * zz_m[None, :] - 1.0)
            / 2.0
        )
        factor = jnp.maximum(base[:, None] ** exponent, 1.0)
        active = soiltile[:, jst] > min_sechiba
        current = result[:, :, jst]
        result = result.at[:, :, jst].set(jnp.where(active[:, None], current * factor, current))
    return result


@partial(jit, static_argnames=("pref_indices", "peat_hydro", "peat_tiles"))
def _hydrol_vegupd_kfact_root_jit(
    *,
    veget,
    veget_max,
    soiltile,
    vegtot,
    humcste,
    zz_mm,
    njsc,
    pref_indices: tuple[int, ...],
    peat_hydro: bool,
    peat_tiles=(4, 5),
    min_sechiba=1.0e-8,
):
    veget = jnp.asarray(veget, dtype=jnp.float64)
    veget_max = jnp.asarray(veget_max, dtype=jnp.float64)
    soiltile = jnp.asarray(soiltile, dtype=jnp.float64)
    vegtot = jnp.asarray(vegtot, dtype=jnp.float64)
    humcste = jnp.asarray(humcste, dtype=jnp.float64)
    zz_mm = jnp.asarray(zz_mm, dtype=jnp.float64)
    njsc = jnp.asarray(njsc, dtype=jnp.int32)
    if njsc.ndim == 0:
        njsc = njsc[None]

    npts, nvm = veget.shape
    nstm = soiltile.shape[1]
    minv = jnp.asarray(min_sechiba, dtype=veget.dtype)
    zero = jnp.asarray(0.0, dtype=veget.dtype)
    one = jnp.asarray(1.0, dtype=veget.dtype)
    mask_soiltile = jnp.where(soiltile > minv, one, zero)
    mask_veget = jnp.where(veget_max > minv, one, zero)

    vegetmax_soil = jnp.zeros((npts, nvm, nstm), dtype=veget.dtype)
    for jv, jst in enumerate(pref_indices):
        active = (mask_soiltile[:, jst] > zero) & (vegtot > minv)
        vegetmax_soil = vegetmax_soil.at[:, jv, jst].set(
            jnp.where(active, veget_max[:, jv] / jnp.where(soiltile[:, jst] > minv, soiltile[:, jst], 1.0), zero)
        )

    frac_bare = jnp.zeros_like(veget)
    frac_bare = frac_bare.at[:, 0].set(jnp.where(veget_max[:, 0] > minv, one, zero))
    for jv in range(1, nvm):
        frac_bare = frac_bare.at[:, jv].set(
            jnp.where(
                veget_max[:, jv] > minv,
                one - veget[:, jv] / jnp.where(veget_max[:, jv] > minv, veget_max[:, jv], 1.0),
                zero,
            )
        )

    frac_bare_ns = jnp.zeros((npts, nstm), dtype=veget.dtype)
    for jst in range(nstm):
        tile_sum = jnp.sum(vegetmax_soil[:, :, jst] * frac_bare, axis=1)
        frac_bare_ns = frac_bare_ns.at[:, jst].set(
            jnp.where(vegtot > minv, tile_sum / jnp.where(vegtot > minv, vegtot, 1.0), zero)
        )

    ks_values = jnp.asarray(USDA_KS, dtype=jnp.float64)
    max_ks = jnp.max(ks_values)
    mineral_ks = ks_values[njsc - 1]
    peat_tile_set = {int(tile) for tile in peat_tiles}
    kfact_root = jnp.ones((npts, int(zz_mm.shape[0]), nstm), dtype=jnp.float64)
    zz_m = zz_mm / 1000.0
    for jv in range(1, nvm):
        jst = int(pref_indices[jv])
        jst_fortran = jst + 1
        denom_ks = (
            jnp.full((npts,), float(PEAT_KS), dtype=jnp.float64)
            if bool(peat_hydro) and jst_fortran in peat_tile_set
            else mineral_ks
        )
        base = max_ks / denom_ks
        exponent = (
            -vegetmax_soil[:, jv, jst][:, None]
            / 2.0
            * (humcste[jv] * zz_m[None, :] - 1.0)
            / 2.0
        )
        factor = jnp.maximum(base[:, None] ** exponent, 1.0)
        active = soiltile[:, jst] > minv
        current = kfact_root[:, :, jst]
        kfact_root = kfact_root.at[:, :, jst].set(jnp.where(active[:, None], current * factor, current))

    return mask_veget, mask_soiltile, vegetmax_soil, frac_bare, frac_bare_ns, kfact_root


def hydrol_to_thermosoil_moisture_inputs(
    diagnostics: HydrolModuleDiagnostics,
    *,
    pref_soil_veg,
):
    """Map completed HYDROL diagnostics onto THERMOSOIL moisture inputs.

    Fortran provenance: ``sechiba.f90::sechiba_main`` lines 1093-1118. The
    source maps ``mc_layh_s(:,:,pref_soil_veg(jv))`` and
    ``mcl_layh_s(:,:,pref_soil_veg(jv))`` onto the PFT-resolved THERMOSOIL
    arrays, while ``soilmoist_pft(:,:,jv)=soilmoist(:,:)`` on the active
    paper-case branch. No missing moisture state is inferred here.
    """

    pref_soil_veg = jnp.asarray(pref_soil_veg, dtype=jnp.int32)
    mc_layh_s = _as_3d("diagnostics.mc_layh_s", diagnostics.mc_layh_s)
    mcl_layh_s = _as_3d("diagnostics.mcl_layh_s", diagnostics.mcl_layh_s)
    soilmoist = _as_2d("diagnostics.soilmoist", diagnostics.soilmoist)
    if mcl_layh_s.shape != mc_layh_s.shape:
        raise ValueError("diagnostics.mcl_layh_s must match diagnostics.mc_layh_s")
    npts, nslm, nstm = mc_layh_s.shape
    if soilmoist.shape != (npts, nslm):
        raise ValueError("diagnostics.soilmoist must match (npts,nslm)")
    if pref_soil_veg.ndim != 1:
        raise ValueError("pref_soil_veg must be a one-dimensional Fortran 1-based PFT-to-soil-tile map")
    if bool(jnp.any(pref_soil_veg < 1)) or bool(jnp.any(pref_soil_veg > nstm)):
        raise ValueError("pref_soil_veg contains a soil tile outside diagnostics.mc_layh_s")

    nvm = pref_soil_veg.shape[0]
    pref_indices = pref_soil_veg - 1
    mc_layh_pft = jnp.take(mc_layh_s, pref_indices, axis=2)
    mcl_layh_pft = jnp.take(mcl_layh_s, pref_indices, axis=2)
    tmc_layh_pft = jnp.broadcast_to(soilmoist[:, :, None], (npts, nslm, nvm))

    return HydrolThermosoilMoistureInputs(
        shumdiag_perma=diagnostics.shumdiag_perma,
        mc_layh=diagnostics.mc_layh,
        mcl_layh=diagnostics.mcl_layh,
        tmc_layh=diagnostics.soilmoist,
        mc_layh_pft=mc_layh_pft,
        mcl_layh_pft=mcl_layh_pft,
        tmc_layh_pft=tmc_layh_pft,
    )


def _hydrol_module_outputs_and_moisture_arrays(
    result: HydrolModuleExplicitResult,
    diagnostics: HydrolModuleDiagnostics,
    soiltile,
    pref_soil_veg,
):
    soil = result.soil
    soiltile = jnp.asarray(soiltile, dtype=jnp.float64)
    pref_indices = jnp.asarray(pref_soil_veg, dtype=jnp.int32) - 1
    soilmoist = jnp.asarray(diagnostics.soilmoist, dtype=jnp.float64)
    mc_layh_s = jnp.asarray(diagnostics.mc_layh_s, dtype=jnp.float64)
    mcl_layh_s = jnp.asarray(diagnostics.mcl_layh_s, dtype=jnp.float64)
    npts, nslm = soilmoist.shape
    nvm = pref_indices.shape[0]
    outputs = (
        soil.ru_ns,
        soil.dr_ns,
        soil.runoff2peat,
        soil.tmc,
        soil.tmc_soil,
        soil.qflux,
        jnp.sum(soiltile * soil.wtd_ns, axis=1),
        soil.wtd_ns,
        result.canop.precip2canopy,
        result.canop.precip2ground,
        result.canop.canopy2ground,
        result.flood.floodout,
    )
    moisture = (
        diagnostics.shumdiag_perma,
        diagnostics.mc_layh,
        diagnostics.mcl_layh,
        soilmoist,
        jnp.take(mc_layh_s, pref_indices, axis=2),
        jnp.take(mcl_layh_s, pref_indices, axis=2),
        jnp.broadcast_to(soilmoist[:, :, None], (npts, nslm, nvm)),
    )
    return outputs, moisture


_hydrol_module_outputs_and_moisture_arrays_jit = jit(_hydrol_module_outputs_and_moisture_arrays)


def _validate_hydrol_outputs_and_moisture_inputs(
    result: HydrolModuleExplicitResult,
    diagnostics: HydrolModuleDiagnostics,
    *,
    soiltile,
    pref_soil_veg,
) -> None:
    soiltile_arr = jnp.asarray(soiltile)
    if soiltile_arr.shape != result.soil.ru_ns.shape:
        raise ValueError("soiltile shape must match runoff/drainage soil-tile arrays")
    pref = np.asarray(pref_soil_veg, dtype=np.int32)
    if pref.ndim != 1:
        raise ValueError("pref_soil_veg must be a one-dimensional Fortran 1-based PFT-to-soil-tile map")
    mc_layh_s = jnp.asarray(diagnostics.mc_layh_s)
    mcl_layh_s = jnp.asarray(diagnostics.mcl_layh_s)
    soilmoist = jnp.asarray(diagnostics.soilmoist)
    if mcl_layh_s.shape != mc_layh_s.shape:
        raise ValueError("diagnostics.mcl_layh_s must match diagnostics.mc_layh_s")
    if soilmoist.shape != mc_layh_s.shape[:2]:
        raise ValueError("diagnostics.soilmoist must match (npts,nslm)")
    nstm = mc_layh_s.shape[2]
    if bool(np.any(pref < 1)) or bool(np.any(pref > nstm)):
        raise ValueError("pref_soil_veg contains a soil tile outside diagnostics.mc_layh_s")


def hydrol_module_outputs_and_moisture_inputs_jit(
    result: HydrolModuleExplicitResult,
    diagnostics: HydrolModuleDiagnostics,
    *,
    soiltile,
    pref_soil_veg,
) -> tuple[HydrolModuleOutputs, HydrolThermosoilMoistureInputs]:
    """Build HYDROL downstream outputs and THERMOSOIL moisture in one JIT call."""

    _validate_hydrol_outputs_and_moisture_inputs(
        result,
        diagnostics,
        soiltile=soiltile,
        pref_soil_veg=pref_soil_veg,
    )
    output_arrays, moisture_arrays = _hydrol_module_outputs_and_moisture_arrays_jit(
        result,
        diagnostics,
        soiltile,
        pref_soil_veg,
    )
    outputs = HydrolModuleOutputs(*output_arrays)
    moisture = HydrolThermosoilMoistureInputs(*moisture_arrays)
    return outputs, moisture


def hydrol_cold_start_thermosoil_moisture_inputs(
    state: HydrolColdStartState,
    *,
    pref_soil_veg,
    dz_mm,
    dh_mm,
    njsc,
    mcs=None,
    mcs_mineral=USDA_MCS,
    peat_hydro=False,
) -> HydrolThermosoilMoistureInputs:
    """Map ``hydrol_init`` cold-start moisture to THERMOSOIL init inputs.

    Fortran provenance: ``hydrol_var_init`` lines 4361-4405 computes ``tmc``
    from no-restart ``mc``/``water2infilt``; lines 4512-4611 compute
    ``shumdiag_perma`` from ``resdist``-weighted ``soilmoist``; lines
    4629-4644 populate ``mc_layh``, ``mcl_layh``, ``mc_layh_s``, and
    ``mcl_layh_s`` for the following ``thermosoil_initialize`` call. The
    ``sechiba_initialize`` PFT mapping follows lines 703-707.
    """

    pref_soil_veg = jnp.asarray(pref_soil_veg, dtype=jnp.int32)
    dz_mm = _as_1d("dz_mm", dz_mm)
    dh_mm = _as_1d("dh_mm", dh_mm)
    moisture = hydrol_soilmoist_aggregates(
        mc=state.mc,
        mcl=state.mcl,
        soiltile=state.resdist,
        vegtot=state.vegtot_old,
        vegtot_old=state.vegtot_old,
        dz_mm=dz_mm,
    )
    if dh_mm.shape[0] != moisture.soilmoist.shape[1]:
        raise ValueError("dh_mm length must match nslm")
    npts, nslm, nstm = moisture.mc_layh_s.shape
    if pref_soil_veg.ndim != 1:
        raise ValueError("pref_soil_veg must be one-dimensional")
    if bool(jnp.any(pref_soil_veg < 1)) or bool(jnp.any(pref_soil_veg > nstm)):
        raise ValueError("pref_soil_veg contains a soil tile outside cold-start mc_layh_s")

    njsc = jnp.asarray(njsc, dtype=jnp.int32)
    if njsc.ndim == 0:
        njsc = njsc[None]
    if njsc.shape != (npts,):
        raise ValueError("njsc must have shape (npts,)")
    point_mcs = _per_point_threshold("mcs", mcs, njsc, npts, USDA_MCS)
    mineral_mcs = _per_point_threshold("mcs_mineral", mcs_mineral, njsc, npts, USDA_MCS)
    if bool(peat_hydro):
        shumdiag_perma = moisture.soilmoist / dh_mm[None, :]
    else:
        shumdiag_perma = moisture.soilmoist / dh_mm[None, :] / point_mcs[:, None] * mineral_mcs[:, None]
    shumdiag_perma = jnp.clip(shumdiag_perma, 0.0, 1.0)

    nvm = pref_soil_veg.shape[0]
    pref_indices = pref_soil_veg - 1
    mc_layh_pft = jnp.take(moisture.mc_layh_s, pref_indices, axis=2)
    mcl_layh_pft = jnp.take(moisture.mcl_layh_s, pref_indices, axis=2)
    tmc_layh_pft = jnp.broadcast_to(moisture.soilmoist[:, :, None], (npts, nslm, nvm))

    return HydrolThermosoilMoistureInputs(
        shumdiag_perma=shumdiag_perma,
        mc_layh=moisture.mc_layh,
        mcl_layh=moisture.mcl_layh,
        tmc_layh=moisture.soilmoist,
        mc_layh_pft=mc_layh_pft,
        mcl_layh_pft=mcl_layh_pft,
        tmc_layh_pft=tmc_layh_pft,
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_var_init lines 4361-4405",
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_var_init lines 4512-4644",
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_initialize lines 703-707",
            "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_humlev lines 2258-2399",
        ),
    )


HYDROL_MODULE_EXPLICIT_STEP_FIELDS = (
    "precip_rain",
    "vevapwet",
    "veget",
    "veget_max",
    "qsintmax",
    "qsintveg",
    "tot_melt",
    "vegtot",
    "throughfall_by_pft",
    "pref_soil_veg",
    "soiltile",
    "vevapflo",
    "flood_frac",
    "flood_res",
    "subsinksoil",
    "vevapnu",
    "vevapnu_pft",
    "transpir",
    "humrel",
    "humrelv",
    "us",
    "ae_ns",
    "evap_bare_lim",
    "evap_bare_lim_ns",
    "evapot",
    "evapot_corr",
    "water2infilt",
    "reinfiltration_soil",
    "irrigation_soil",
    "mc",
    "mcl",
    "profil_froz",
    "mcr",
    "mcs",
    "dz_mm",
    "dt_days",
    "free_drain_coef",
    "resolv",
    "kfact_root",
    "ks",
    "kfact",
    "tmc",
    "njsc",
    "mineral_tables",
    "peat_tables",
    "reinf_slope",
    "doponds",
    "peat_hydro",
    "ok_freeze_cwrr",
    "temp_hydro",
    "zwt_force",
    "zz_mm",
    "zmaxh_m",
    "run2peat",
    "run2man",
    "wt_ab",
    "wt_ab_tide",
    "runoff2peat",
    "tides",
    "ok_ru2peat",
    "ok_wt_ab",
    "agri_peat",
    "max_wt_ab",
)

HYDROL_MODULE_JIT_STATIC_ARGNAMES = (
    "pref_soil_veg",
    "is_crop_soil",
    "doponds",
    "peat_hydro",
    "peat_tiles",
    "ok_freeze_cwrr",
    "tides",
    "ok_ru2peat",
    "ok_wt_ab",
    "agri_peat",
)

HYDROL_DIAGNOSTICS_JIT_STATIC_ARGNAMES = (
    "ok_freeze_cwrr",
    "peat_hydro",
    "peat_hydro_water_stress",
    "peat_tiles",
    "agri_peat",
    "tides",
    "dyn_nroot_larix",
    "is_tree",
    "new_watstress",
    "topmodel_new",
    "do_rsoil",
    "liqlayers",
    "numlayers",
)

def run_hydrol_first_step_module_from_precall(
    precall: HydrolFirstStepPrecallAssembly,
    *,
    temp_air=None,
    pb=None,
    u=None,
    v=None,
    humcste,
    nroot_state=None,
    dz_mm,
    dh_mm,
    zz_mm,
    altmax=None,
    ks=USDA_KS,
    reinf_slope=None,
    zmaxh_m=2.0,
    doponds=False,
    peat_hydro=False,
    peat_hydro_water_stress=None,
    ok_freeze_cwrr=True,
    ok_explicitsnow=True,
    ok_pc=False,
    ok_leak=False,
    ok_ru2peat=False,
    ok_wt_ab=False,
    tides=False,
    agri_peat=False,
    max_wt_ab=100.0,
    dyn_nroot_larix=False,
    is_tree=None,
    znt=None,
    run2peat=None,
    run2man=None,
    runoff2peat=None,
    dt_sechiba=1800.0,
    one_day=86400.0,
    use_jit=False,
    static_table_payload: Mapping[str, object] | None = None,
    runtime_static_tables: HydrolRuntimeStaticTables | None = None,
) -> HydrolFirstStepModuleClosure:
    """Run the first-step HYDROL module and expose downstream boundaries.

    Fortran provenance follows ``sechiba_main`` HYDROL ordering at lines
    1049-1118 and ``hydrol_main``/``hydrol_soil``/``hydrol_diag_soil`` through
    the spans listed in ``HYDROL_FIRST_STEP_MODULE_CLOSURE_PROVENANCE``. The
    helper consumes an already audited pre-call payload and refuses to execute
    if any required source-backed field is absent.
    """

    payload = dict(precall.payload)
    missing: list[str] = list(precall.missing_inputs)
    if not precall.ok:
        return HydrolFirstStepModuleClosure(
            module=None,
            outputs=None,
            diagnostics=None,
            thermosoil_moisture=None,
            snow_state=None,
            snow_step=None,
            module_inputs={},
            diagnostic_inputs={},
            missing_inputs=tuple(dict.fromkeys(missing)),
            provenance=HYDROL_FIRST_STEP_MODULE_CLOSURE_PROVENANCE,
            notes=("HYDROL pre-call payload is incomplete; module execution was skipped.",),
        )

    npts = _as_2d("soiltile", payload["soiltile"]).shape[0]
    routing_zero = routing_zero_outputs(npts, 1)
    dz_mm = _as_1d("dz_mm", dz_mm)
    dh_mm = _as_1d("dh_mm", dh_mm)
    zz_mm = _as_1d("zz_mm", zz_mm)
    njsc = _as_1d("njsc", payload["njsc"]).astype(jnp.int32)
    mcs = _as_1d("mcs", payload["mcs"])
    if dz_mm.shape != dh_mm.shape or dz_mm.shape != zz_mm.shape:
        raise ValueError("dz_mm, dh_mm, and zz_mm must share the HYDROL layer axis")

    if runtime_static_tables is None:
        table_payload = static_table_payload or {"njsc": njsc, "mcs": mcs, "zz_mm": zz_mm}
        runtime_static_tables = hydrol_runtime_static_tables_from_payload(
            table_payload,
            ks=ks,
            peat_hydro=peat_hydro,
        )
    mineral_tables = runtime_static_tables.mineral
    peat_tables = runtime_static_tables.peat
    soiltile = _as_2d("soiltile", payload["soiltile"])
    module_inputs = dict(payload)
    module_inputs.update(
        {
            "dz_mm": dz_mm,
            "zz_mm": zz_mm,
            "resolv": soiltile > 0.0,
            "ks": ks,
            "kfact": mineral_tables.kfact,
            "mineral_tables": mineral_tables,
            "peat_tables": peat_tables,
            "tmc": hydrol_initial_tmc_from_restart_mc(
                mc=payload["mc"],
                water2infilt=payload["water2infilt"],
                dz_mm=dz_mm,
            ),
            # sechiba.f90:1235-1248 sets the independent single-landpoint
            # no-routing return, reinfiltration, and irrigation fields to zero.
            "reinfiltration_soil": jnp.broadcast_to(
                jnp.asarray(routing_zero["reinfiltration"])[:, :1], soiltile.shape
            ),
            "irrigation_soil": jnp.broadcast_to(
                jnp.asarray(routing_zero["irrigation"])[:, :1], soiltile.shape
            ),
            "reinf_slope": reinf_slope,
            "zmaxh_m": zmaxh_m,
            "doponds": doponds,
            "peat_hydro": peat_hydro,
            "ok_freeze_cwrr": ok_freeze_cwrr,
            "tides": tides,
            "ok_ru2peat": ok_ru2peat,
            "ok_wt_ab": ok_wt_ab,
            "agri_peat": agri_peat,
            "max_wt_ab": max_wt_ab,
        }
    )
    for name, value in (
        ("run2peat", run2peat),
        ("run2man", run2man),
        ("runoff2peat", runoff2peat),
    ):
        if value is not None:
            module_inputs[name] = value

    # Fortran hydrol_main lines 1177-1202 update snow before canopy/soil.
    # In particular, same-step melt and the explicit-snow soil sink must feed
    # hydrol_canop/hydrol_soil rather than being written only after the solve.
    snow_step = None
    bucket_snow_step = None
    snow_state = None
    totfrac_nobio = module_inputs.get("totfrac_nobio")
    if totfrac_nobio is None:
        totfrac_nobio = 1.0 - jnp.asarray(module_inputs["vegtot"], dtype=jnp.float64)
    if bool(ok_explicitsnow):
        snow_inputs = {
            "precip_rain": module_inputs.get("precip_rain"),
            "precip_snow": module_inputs.get("precip_snow"),
            "temp_air": temp_air,
            "pb": pb,
            "u": u,
            "v": v,
            "temp_sol_new": module_inputs.get("temp_sol_new"),
            "soilcap": module_inputs.get("soilcap"),
            "pgflux": module_inputs.get("pgflux"),
            "frac_nobio": module_inputs.get("frac_nobio"),
            "totfrac_nobio": totfrac_nobio,
            "gtemp": module_inputs.get("gtemp"),
            "lambda_snow": module_inputs.get("lambda_snow"),
            "cgrnd_snow": module_inputs.get("cgrnd_snow"),
            "dgrnd_snow": module_inputs.get("dgrnd_snow"),
            "vevapsno": module_inputs.get("vevapsno"),
            "snow_age": module_inputs.get("snow_age"),
            "snow_nobio_age": module_inputs.get("snow_nobio_age"),
            "snow_nobio": module_inputs.get("snow_nobio"),
            "snowrho": module_inputs.get("snowrho"),
            "snowgrain": module_inputs.get("snowgrain"),
            "snowdz": module_inputs.get("snowdz"),
            "snowtemp": module_inputs.get("snowtemp"),
            "snowheat": module_inputs.get("snowheat"),
            "snow": module_inputs.get("snow"),
            "temp_sol_add": module_inputs.get("temp_sol_add"),
            "snowliq": module_inputs.get("snowliq"),
            "subsnownobio": module_inputs.get("subsnownobio"),
            "grndflux": module_inputs.get("grndflux"),
            "snowmelt": module_inputs.get("snowmelt"),
            "soilflxresid": module_inputs.get("soilflxresid"),
        }
        missing_snow = tuple(
            f"explicit_snow:{name}"
            for name in HYDROL_EXPLICIT_SNOW_STEP_REQUIRED_FIELDS
            if snow_inputs.get(name) is None
        )
        tracing_compiled_transition = any(
            isinstance(value, jax.core.Tracer)
            for value in snow_inputs.values()
            if value is not None
        )
        if tracing_compiled_transition and not missing_snow:
            snow_step = hydrol_explicit_snow_step(
                **snow_inputs,
                dt_sechiba=dt_sechiba,
                one_day=one_day,
            )
            snow_state = snow_step.snow_state
        else:
            try:
                snow_state = hydrol_explicit_snow_zero_state(
                    precip_snow=module_inputs["precip_snow"],
                    precip_rain=module_inputs["precip_rain"],
                    vevapsno=module_inputs["vevapsno"],
                    snow=module_inputs["snow"],
                    snowdz=module_inputs["snowdz"],
                    snowrho=module_inputs["snowrho"],
                    snowtemp=module_inputs["snowtemp"],
                    snow_age=module_inputs["snow_age"],
                    snow_nobio=module_inputs["snow_nobio"],
                    snow_nobio_age=module_inputs["snow_nobio_age"],
                    frac_nobio=module_inputs["frac_nobio"],
                    totfrac_nobio=totfrac_nobio,
                    use_jit=bool(use_jit),
                )
            except ValueError as exc:
                if missing_snow:
                    return HydrolFirstStepModuleClosure(
                        module=None,
                        outputs=None,
                        diagnostics=None,
                        thermosoil_moisture=None,
                        snow_state=None,
                        snow_step=None,
                        module_inputs={},
                        diagnostic_inputs={},
                        missing_inputs=missing_snow,
                        provenance=HYDROL_FIRST_STEP_MODULE_CLOSURE_PROVENANCE,
                        notes=(
                            "HYDROL execution stopped at the source-ordered explicit-snow boundary.",
                            str(exc),
                        ),
                        routing_zero=routing_zero,
                    )
                snow_step = hydrol_explicit_snow_step(
                    **snow_inputs,
                    dt_sechiba=dt_sechiba,
                    one_day=one_day,
                )
                snow_state = snow_step.snow_state
        module_inputs["tot_melt"] = snow_state.tot_melt
        if snow_step is not None:
            module_inputs["subsinksoil"] = snow_step.subsinksoil
    else:
        bucket_inputs = {
            "precip_rain": module_inputs.get("precip_rain"),
            "precip_snow": module_inputs.get("precip_snow"),
            "temp_sol_new": module_inputs.get("temp_sol_new"),
            "soilcap": module_inputs.get("soilcap"),
            "frac_nobio": module_inputs.get("frac_nobio"),
            "totfrac_nobio": totfrac_nobio,
            "vevapsno": module_inputs.get("vevapsno"),
            "snow": module_inputs.get("snow"),
            "snow_age": module_inputs.get("snow_age"),
            "snow_nobio": module_inputs.get("snow_nobio"),
            "snow_nobio_age": module_inputs.get("snow_nobio_age"),
        }
        missing_bucket = tuple(f"bucket_snow:{name}" for name, value in bucket_inputs.items() if value is None)
        if missing_bucket:
            return HydrolFirstStepModuleClosure(
                module=None,
                outputs=None,
                diagnostics=None,
                thermosoil_moisture=None,
                snow_state=None,
                snow_step=None,
                module_inputs={},
                diagnostic_inputs={},
                missing_inputs=missing_bucket,
                provenance=HYDROL_FIRST_STEP_MODULE_CLOSURE_PROVENANCE,
                notes=("HYDROL execution stopped at the source-ordered bucket-snow boundary.",),
                routing_zero=routing_zero,
            )
        bucket_snow_step = hydrol_bucket_snow_step(
            **bucket_inputs,
            dt_sechiba=dt_sechiba,
            one_day=one_day,
        )
        module_inputs["tot_melt"] = bucket_snow_step.tot_melt
        module_inputs["subsinksoil"] = bucket_snow_step.subsinksoil

    if not bool(doponds) and reinf_slope is None:
        missing.append("reinf_slope")
    for name in HYDROL_MODULE_EXPLICIT_STEP_FIELDS:
        if name not in module_inputs and name not in {"peat_tables", "run2peat", "run2man", "runoff2peat"}:
            missing.append(name)
    if bool(peat_hydro) and peat_tables is None:
        missing.append("peat_tables")
    if (bool(ok_pc) or bool(ok_leak)) and altmax is None:
        missing.append("altmax")
    if missing:
        return HydrolFirstStepModuleClosure(
            module=None,
            outputs=None,
            diagnostics=None,
            thermosoil_moisture=None,
            snow_state=None,
            snow_step=None,
            module_inputs={name: module_inputs[name] for name in HYDROL_MODULE_EXPLICIT_STEP_FIELDS if name in module_inputs},
            diagnostic_inputs={},
            missing_inputs=tuple(dict.fromkeys(missing)),
            provenance=HYDROL_FIRST_STEP_MODULE_CLOSURE_PROVENANCE,
            notes=("HYDROL first-step module execution was skipped because strict inputs are missing.",),
        )

    selected_module_inputs = {name: module_inputs[name] for name in HYDROL_MODULE_EXPLICIT_STEP_FIELDS if name in module_inputs}
    if nroot_state is None:
        nroot_inputs = (humcste, dz_mm, zz_mm, altmax)
        if any(isinstance(value, jax.core.Tracer) for value in nroot_inputs if value is not None):
            nroot = hydrol_nroot_from_humcste(
                humcste=humcste,
                dz_mm=dz_mm,
                zz_mm=zz_mm,
                altmax=altmax,
                ok_pc=ok_pc,
                ok_leak=ok_leak,
            )
        else:
            altmax_key = None
            if bool(ok_pc) or bool(ok_leak):
                altmax_key = tuple(
                    tuple(float(value) for value in row)
                    for row in np.asarray(altmax, dtype=np.float64)
                )
            nroot = _cached_hydrol_nroot_from_humcste(
                _array_cache_key(humcste),
                _array_cache_key(dz_mm),
                _array_cache_key(zz_mm),
                altmax_key,
                bool(ok_pc),
                bool(ok_leak),
            )
    else:
        nroot = _as_3d("nroot_state", nroot_state)
    if bool(use_jit):
        selected_module_inputs = dict(selected_module_inputs)
        selected_module_inputs["pref_soil_veg"] = tuple(int(value) for value in np.asarray(selected_module_inputs["pref_soil_veg"]).tolist())
        if "peat_tiles" in selected_module_inputs:
            selected_module_inputs["peat_tiles"] = tuple(int(value) for value in selected_module_inputs["peat_tiles"])
        module, diagnostics, output_arrays, moisture_arrays = _hydrol_module_closure_core_jit(
            **selected_module_inputs,
            nroot=nroot,
            dh_mm=dh_mm,
            peat_hydro_water_stress=peat_hydro if peat_hydro_water_stress is None else peat_hydro_water_stress,
            dyn_nroot_larix=dyn_nroot_larix,
            is_tree=tuple(bool(value) for value in np.asarray(is_tree).tolist()) if is_tree is not None else None,
            znt=znt,
        )
        outputs = HydrolModuleOutputs(*output_arrays)
        moisture = HydrolThermosoilMoistureInputs(*moisture_arrays)
    else:
        module = hydrol_module_explicit_step(**selected_module_inputs)
        diagnostic_inputs = {
            "nroot": nroot,
            "dz_mm": dz_mm,
            "dh_mm": dh_mm,
            "njsc": njsc,
            "veget_max": module_inputs["veget_max"],
            "soiltile": module_inputs["soiltile"],
            "vegtot": module_inputs["vegtot"],
            "mcs": module_inputs["mcs"],
            "mcf": module_inputs.get("mcf"),
            "mcw": module_inputs.get("mcw"),
            "mineral_tables": mineral_tables,
            "peat_tables": peat_tables,
            "ks": ks,
            "vevapnu": module_inputs["vevapnu"],
            "tot_melt": module_inputs["tot_melt"],
            "ok_freeze_cwrr": bool(selected_module_inputs.get("ok_freeze_cwrr", False)),
            "peat_hydro": peat_hydro,
            "peat_hydro_water_stress": peat_hydro if peat_hydro_water_stress is None else peat_hydro_water_stress,
            "agri_peat": agri_peat,
            "tides": tides,
            "dyn_nroot_larix": dyn_nroot_larix,
            "is_tree": is_tree,
            "znt": znt,
            "temp_hydro": module_inputs.get("temp_hydro"),
            "evapot": module_inputs["evapot"],
        }
        diagnostic_inputs["evap_bare_limit_alt"] = module.soil.evap_bare_limit_alt
        diagnostic_inputs["tmcint"] = module.soil.tmcint_for_evap_bare_limit
        diagnostics = hydrol_module_diagnostics(module, **diagnostic_inputs)
        outputs = hydrol_module_outputs(module, soiltile=module_inputs["soiltile"])
        moisture = hydrol_to_thermosoil_moisture_inputs(diagnostics, pref_soil_veg=module_inputs["pref_soil_veg"])
    diagnostic_inputs = {
        "nroot": nroot,
        "dz_mm": dz_mm,
        "dh_mm": dh_mm,
        "njsc": njsc,
        "veget_max": module_inputs["veget_max"],
        "soiltile": module_inputs["soiltile"],
        "vegtot": module_inputs["vegtot"],
        "mcs": module_inputs["mcs"],
        "mcf": module_inputs.get("mcf"),
        "mcw": module_inputs.get("mcw"),
        "mineral_tables": mineral_tables,
        "peat_tables": peat_tables,
        "ks": ks,
        "vevapnu": module_inputs["vevapnu"],
        "tot_melt": module_inputs["tot_melt"],
        "ok_freeze_cwrr": bool(selected_module_inputs.get("ok_freeze_cwrr", False)),
        "peat_hydro": peat_hydro,
        "peat_hydro_water_stress": peat_hydro if peat_hydro_water_stress is None else peat_hydro_water_stress,
        "agri_peat": agri_peat,
        "tides": tides,
        "dyn_nroot_larix": dyn_nroot_larix,
        "is_tree": is_tree,
        "znt": znt,
        "temp_hydro": module_inputs.get("temp_hydro"),
        "evapot": module_inputs["evapot"],
    }
    diagnostic_inputs["evap_bare_limit_alt"] = module.soil.evap_bare_limit_alt
    diagnostic_inputs["tmcint"] = module.soil.tmcint_for_evap_bare_limit
    return HydrolFirstStepModuleClosure(
        module=module,
        outputs=outputs,
        diagnostics=diagnostics,
        thermosoil_moisture=moisture,
        snow_state=snow_state,
        snow_step=snow_step,
        module_inputs=selected_module_inputs,
        diagnostic_inputs=diagnostic_inputs,
        missing_inputs=(),
        provenance=HYDROL_FIRST_STEP_MODULE_CLOSURE_PROVENANCE,
        notes=(
            "Runs HYDROL first-step main outputs, diagnostics, and HYDROL-to-THERMOSOIL moisture from explicit source-backed inputs.",
            "Same-step CONDVEG and THERMOSOIL execution are outside this HYDROL module closure.",
        ),
        bucket_snow_step=bucket_snow_step,
        routing_zero=routing_zero,
    )


def hydrol_split_soil_fluxes(
    *,
    veget_max,
    soiltile,
    precisol,
    vevapnu,
    vevapnu_pft,
    transpir,
    humrel,
    humrelv,
    us,
    ae_ns,
    evap_bare_lim,
    evap_bare_lim_ns,
    frac_bare_ns,
    tot_bare_soil,
    vegetmax_soil,
    vegtot,
    pref_soil_veg,
    min_sechiba=1.0e-8,
):
    """Split PFT water fluxes to HYDROL soil tiles.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_split_soil``, lines 8603-8925. This helper implements
    the source-order split for ``precisol_ns``, ``vevapnu_ns``, ``ae_ns``,
    ``tr_ns``, and ``rootsink``. Module state arrays are explicit inputs and
    outputs; no soil-solve forcing is defaulted.
    """

    veget_max = _as_2d("veget_max", veget_max)
    soiltile = _as_2d("soiltile", soiltile)
    precisol = _as_2d("precisol", precisol)
    vevapnu = _as_1d("vevapnu", vevapnu)
    vevapnu_pft = _as_2d("vevapnu_pft", vevapnu_pft)
    transpir = _as_2d("transpir", transpir)
    humrel = _as_2d("humrel", humrel)
    humrelv = _as_3d("humrelv", humrelv)
    us = _as_4d("us", us)
    ae_ns = _as_2d("ae_ns", ae_ns)
    evap_bare_lim = _as_1d("evap_bare_lim", evap_bare_lim)
    evap_bare_lim_ns = _as_2d("evap_bare_lim_ns", evap_bare_lim_ns)
    frac_bare_ns = _as_2d("frac_bare_ns", frac_bare_ns)
    tot_bare_soil = _as_1d("tot_bare_soil", tot_bare_soil)
    vegetmax_soil = _as_3d("vegetmax_soil", vegetmax_soil)
    vegtot = _as_1d("vegtot", vegtot)
    pref_soil_veg_np = np.asarray(pref_soil_veg, dtype=np.int32)
    _require_same_npts(
        veget_max,
        soiltile,
        precisol,
        vevapnu,
        vevapnu_pft,
        transpir,
        humrel,
        humrelv,
        us,
        ae_ns,
        evap_bare_lim,
        evap_bare_lim_ns,
        frac_bare_ns,
        tot_bare_soil,
        vegetmax_soil,
        vegtot,
    )
    if not (veget_max.shape == precisol.shape == vevapnu_pft.shape == transpir.shape == humrel.shape):
        raise ValueError("PFT arrays must share shape (npts, nvm)")
    if not (soiltile.shape == ae_ns.shape == evap_bare_lim_ns.shape == frac_bare_ns.shape):
        raise ValueError("soil-tile arrays must share shape (npts, nstm)")
    if vegetmax_soil.shape != (veget_max.shape[0], veget_max.shape[1], soiltile.shape[1]):
        raise ValueError("vegetmax_soil must have shape (npts, nvm, nstm)")
    if humrelv.shape != vegetmax_soil.shape:
        raise ValueError("humrelv must have shape (npts, nvm, nstm)")
    if us.shape[:3] != vegetmax_soil.shape:
        raise ValueError("us must have shape (npts, nvm, nstm, nslm)")
    if pref_soil_veg_np.ndim != 1 or pref_soil_veg_np.shape[0] != veget_max.shape[1]:
        raise ValueError("pref_soil_veg must be one-dimensional with length nvm")
    if bool(np.any(pref_soil_veg_np < 1)) or bool(np.any(pref_soil_veg_np > soiltile.shape[1])):
        raise ValueError("pref_soil_veg must contain Fortran 1-based soil tile indices")
    pref_indices = tuple(int(value) - 1 for value in pref_soil_veg_np.tolist())

    npts, nvm = veget_max.shape
    nstm = soiltile.shape[1]
    nslm = us.shape[3]
    zero = jnp.asarray(0.0, dtype=veget_max.dtype)
    minv = jnp.asarray(min_sechiba, dtype=veget_max.dtype)

    precisol_ns = jnp.zeros((npts, nstm), dtype=veget_max.dtype)
    for jv in range(nvm):
        jst = pref_indices[jv]
        denom = soiltile[:, jst] * vegtot
        contribution = jnp.where(
            (veget_max[:, jv] > minv) & (denom > minv),
            precisol[:, jv] / jnp.where(denom > minv, denom, 1.0),
            zero,
        )
        precisol_ns = precisol_ns.at[:, jst].add(contribution)

    vevapnu_ns = jnp.zeros((npts, nstm), dtype=veget_max.dtype)
    for jv in range(nvm):
        safe_vegtot = jnp.where(vegtot > minv, vegtot, 1.0)
        safe_veget_max = jnp.where(veget_max[:, jv] > minv, veget_max[:, jv], 1.0)
        contribution = jnp.where(
            veget_max[:, jv, None] > minv,
            vevapnu_pft[:, jv, None]
            * vegetmax_soil[:, jv, :]
            / safe_vegtot[:, None]
            / safe_veget_max[:, None],
            zero,
        )
        vevapnu_ns = vevapnu_ns + contribution

    vevapnu_old = jnp.sum(
        jnp.where(vegtot[:, None] > minv, ae_ns * soiltile * vegtot[:, None], zero),
        axis=1,
    )
    ae_new = ae_ns
    aens_old_tmp = jnp.asarray(1000.0, dtype=veget_max.dtype) * vevapnu_old
    for jst in range(nstm):
        old_active = vevapnu_old > minv
        evap_lim_active = evap_bare_lim > minv
        old_scale = jnp.where(
            ae_new[:, jst] > aens_old_tmp,
            zero,
            ae_new[:, jst] * vevapnu / jnp.where(vevapnu_old > minv, vevapnu_old, 1.0),
        )
        old_case = jnp.where(
            evap_lim_active,
            vevapnu * evap_bare_lim_ns[:, jst] / jnp.where(evap_bare_lim > minv, evap_bare_lim, 1.0),
            old_scale,
        )
        no_old_case = jnp.where(
            frac_bare_ns[:, jst] > minv,
            jnp.where(
                evap_lim_active,
                vevapnu
                * evap_bare_lim_ns[:, jst]
                / jnp.where(evap_bare_lim > minv, evap_bare_lim, 1.0),
                jnp.where(
                    tot_bare_soil > minv,
                    vevapnu * frac_bare_ns[:, jst] / jnp.where(tot_bare_soil > minv, tot_bare_soil, 1.0),
                    zero,
                ),
            ),
            ae_new[:, jst],
        )
        ae_new = ae_new.at[:, jst].set(jnp.where(old_active, old_case, no_old_case))

    tr_ns = jnp.zeros((npts, nstm), dtype=veget_max.dtype)
    for jv in range(nvm):
        jst = pref_indices[jv]
        denom = soiltile[:, jst] * vegtot
        contribution = jnp.where(
            (humrel[:, jv] > minv) & (denom > minv),
            transpir[:, jv]
            * (humrelv[:, jv, jst] / jnp.where(humrel[:, jv] > minv, humrel[:, jv], 1.0))
            / jnp.where(denom > minv, denom, 1.0),
            zero,
        )
        tr_ns = tr_ns.at[:, jst].add(contribution)

    rootsink = jnp.zeros((npts, nslm, nstm), dtype=veget_max.dtype)
    for jv in range(nvm):
        jst = pref_indices[jv]
        denom = soiltile[:, jst] * vegtot
        contribution = jnp.where(
            ((humrel[:, jv] > minv) & (denom > minv))[:, None],
            transpir[:, jv, None]
            * (us[:, jv, jst, :] / jnp.where(humrel[:, jv, None] > minv, humrel[:, jv, None], 1.0))
            / jnp.where(denom[:, None] > minv, denom[:, None], 1.0),
            zero,
        )
        rootsink = rootsink.at[:, :, jst].add(contribution)

    return HydrolSplitSoilResult(
        precisol_ns=precisol_ns,
        vevapnu_ns=vevapnu_ns,
        ae_ns=ae_new,
        vevapnu_old=vevapnu_old,
        tr_ns=tr_ns,
        rootsink=rootsink,
    )


def hydrol_irrigation_demand_ratio(
    *,
    veget,
    veget_max,
    transpot,
    evapot,
    precip_rain,
    vegstress_old,
    soil_deficit,
    ok_laidev,
    irrig_threshold,
    irrig_fulfill,
    irrig_drip: bool,
    irrig_dosmax,
):
    """Compute normalized HYDROL irrigation demand ratios by PFT.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_main``, lines 1241-1274. The source initializes
    ``irrig_demand_ratio`` to zero, loops over PFTs ``jv=2,nvm``, applies
    either the drip or flooding demand formula for ``ok_LAIdev`` PFTs below
    their irrigation threshold, then normalizes each grid-cell row by its sum.
    """

    veget = _as_2d("veget", veget)
    veget_max = _as_2d("veget_max", veget_max)
    transpot = _as_2d("transpot", transpot)
    vegstress_old = _as_2d("vegstress_old", vegstress_old)
    soil_deficit = _as_2d("soil_deficit", soil_deficit)
    if not (veget.shape == veget_max.shape == transpot.shape == vegstress_old.shape == soil_deficit.shape):
        raise ValueError("veget, veget_max, transpot, vegstress_old, and soil_deficit must share shape (npts,nvm)")
    npts, nvm = veget.shape
    evapot = _as_1d("evapot", evapot)
    precip_rain = _as_1d("precip_rain", precip_rain)
    _require_same_npts(veget, evapot, precip_rain)
    ok_laidev = jnp.asarray(ok_laidev, dtype=bool)
    irrig_threshold = _as_1d("irrig_threshold", irrig_threshold)
    irrig_fulfill = _as_1d("irrig_fulfill", irrig_fulfill)
    if ok_laidev.shape != (nvm,) or irrig_threshold.shape != (nvm,) or irrig_fulfill.shape != (nvm,):
        raise ValueError("ok_laidev, irrig_threshold, and irrig_fulfill must have shape (nvm,)")

    zero = jnp.asarray(0.0, dtype=veget.dtype)
    ratio = jnp.zeros_like(veget)
    irrig_dosmax = jnp.asarray(irrig_dosmax, dtype=veget.dtype)
    for jv in range(1, nvm):
        active_base = (veget_max[:, jv] > zero) & ok_laidev[jv] & (vegstress_old[:, jv] < irrig_threshold[jv])
        if bool(irrig_drip):
            tempfrac = veget[:, jv] / jnp.where(veget_max[:, jv] > zero, veget_max[:, jv], 1.0)
            demand = transpot[:, jv] * tempfrac + evapot * (1.0 - tempfrac) - precip_rain
            active = active_base & (demand > zero)
            raw = jnp.minimum(irrig_dosmax, irrig_fulfill[jv] * demand) * veget_max[:, jv]
        else:
            active = active_base
            raw = jnp.minimum(irrig_dosmax, jnp.maximum(zero, soil_deficit[:, jv])) * veget_max[:, jv]
        ratio = ratio.at[:, jv].set(jnp.where(active, raw, zero))

    row_sum = jnp.sum(ratio, axis=1)
    ratio = jnp.where(row_sum[:, None] > zero, ratio / jnp.where(row_sum[:, None] > zero, row_sum[:, None], 1.0), ratio)
    return HydrolIrrigationDemandResult(irrig_demand_ratio=ratio)


def hydrol_routing_soil_water_split(
    *,
    vegtot,
    returnflow,
    reinfiltration,
    irrigation,
    irrig_demand_ratio,
    veget_max,
    pref_soil_veg,
    soiltile,
    is_crop_soil,
    ok_laidev,
    min_sechiba=1.0e-8,
):
    """Split routing return/reinfiltration and irrigation before soil solve.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil``, lines 5661-5751. The CWRR path sends
    ``returnflow_soil`` to zero, distributes ``returnflow + reinfiltration`` by
    ``vegtot``, and routes irrigation only to crop soil tiles selected by
    one-based ``pref_soil_veg``. The source stops when an active irrigated PFT
    is not mapped to a crop soil tile; this helper raises ``ValueError`` for
    the same unsupported branch.
    """

    vegtot = _as_1d("vegtot", vegtot)
    returnflow = _as_1d("returnflow", returnflow)
    reinfiltration = _as_1d("reinfiltration", reinfiltration)
    irrigation = _as_1d("irrigation", irrigation)
    irrig_demand_ratio = _as_2d("irrig_demand_ratio", irrig_demand_ratio)
    veget_max = _as_2d("veget_max", veget_max)
    soiltile = _as_2d("soiltile", soiltile)
    _require_same_npts(vegtot, returnflow, reinfiltration, irrigation, irrig_demand_ratio, veget_max, soiltile)
    if irrig_demand_ratio.shape != veget_max.shape:
        raise ValueError("irrig_demand_ratio and veget_max must share shape (npts,nvm)")
    if soiltile.ndim != 2:
        raise ValueError("soiltile must have shape (npts,nstm)")

    npts, nvm = veget_max.shape
    nstm = soiltile.shape[1]
    pref = np.asarray(pref_soil_veg, dtype=np.int64)
    crop = np.asarray(is_crop_soil, dtype=bool)
    ok = np.asarray(ok_laidev, dtype=bool)
    if pref.shape != (nvm,):
        raise ValueError("pref_soil_veg must have shape (nvm,) and use Fortran one-based soil tile indices")
    if np.any((pref < 1) | (pref > nstm)):
        raise ValueError("pref_soil_veg must use Fortran one-based soil tile indices within the soil-tile axis")
    if crop.shape != (nstm,):
        raise ValueError("is_crop_soil must have shape (nstm,)")
    if ok.shape != (nvm,):
        raise ValueError("ok_laidev must have shape (nvm,)")

    active_ratio = np.asarray(irrig_demand_ratio) > float(min_sechiba)
    for jv in range(1, nvm):
        tile = int(pref[jv]) - 1
        if ok[jv] and np.any(active_ratio[:, jv]) and not crop[tile]:
            raise ValueError("active HYDROL irrigation demand must map to a crop soil tile")
        if ok[jv] and np.any(active_ratio[:, jv]) and np.any(np.asarray(soiltile)[:, tile] <= float(min_sechiba)):
            raise ValueError("active HYDROL irrigation demand requires positive target soiltile fraction")

    zero = jnp.asarray(0.0, dtype=vegtot.dtype)
    active_cell = vegtot > jnp.asarray(min_sechiba, dtype=vegtot.dtype)
    returnflow_soil = jnp.zeros_like(vegtot)
    reinfiltration_soil = jnp.where(
        active_cell,
        (returnflow + reinfiltration) / jnp.where(active_cell, vegtot, 1.0),
        zero,
    )
    irrigation_soil = jnp.zeros((npts, nstm), dtype=vegtot.dtype)
    irrig_fin = jnp.zeros((npts, nvm), dtype=vegtot.dtype)

    for jv in range(1, nvm):
        tile = int(pref[jv]) - 1
        active = active_cell & (irrig_demand_ratio[:, jv] > zero) & (pref[jv] >= 4)
        irrig_fin = irrig_fin.at[:, jv].set(
            jnp.where(
                active,
                irrigation * irrig_demand_ratio[:, jv] / jnp.where(veget_max[:, jv] > zero, veget_max[:, jv], 1.0),
                zero,
            )
        )
        irrigation_soil = irrigation_soil.at[:, tile].add(
            jnp.where(
                active,
                irrigation * irrig_demand_ratio[:, jv] / jnp.where(soiltile[:, tile] > zero, soiltile[:, tile], 1.0),
                zero,
            )
        )

    return HydrolRoutingSoilWaterSplitResult(
        returnflow_soil=returnflow_soil,
        reinfiltration_soil=reinfiltration_soil,
        irrigation_soil=irrigation_soil,
        irrig_fin=irrig_fin,
    )


def hydrol_soil_surface_water_setup(
    *,
    water2infilt,
    ae_ns_tile,
    subsinksoil,
    precisol_ns_tile,
    reinfiltration_soil,
    is_crop_soil,
    irrigation_soil_tile=None,
):
    """Reduce surface infiltration and extraction before infiltration.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil``, lines 5800-5847. This helper covers the
    source-order setup immediately before ``hydrol_soil_infilt``: cancellation
    between water to infiltrate and water to evaporate/extract, update of
    ``water2infilt``, calculation of ``water2extract``/``flux_top``, and adding
    ``subsinksoil`` into ``ae_ns`` for the active soil tile.
    """

    water2infilt = _as_1d("water2infilt", water2infilt)
    ae_ns_tile = _as_1d("ae_ns_tile", ae_ns_tile)
    subsinksoil = _as_1d("subsinksoil", subsinksoil)
    precisol_ns_tile = _as_1d("precisol_ns_tile", precisol_ns_tile)
    reinfiltration_soil = _as_1d("reinfiltration_soil", reinfiltration_soil)
    _require_same_npts(water2infilt, ae_ns_tile, subsinksoil, precisol_ns_tile, reinfiltration_soil)
    if irrigation_soil_tile is None:
        irrigation_soil_tile = jnp.zeros_like(water2infilt)
    else:
        irrigation_soil_tile = _as_1d("irrigation_soil_tile", irrigation_soil_tile)
        _require_same_npts(water2infilt, irrigation_soil_tile)

    zero = jnp.asarray(0.0, dtype=water2infilt.dtype)
    crop_input = jnp.where(bool(is_crop_soil), irrigation_soil_tile, zero)
    water_to_infiltrate = (
        water2infilt
        + crop_input
        + reinfiltration_soil
        - jnp.minimum(ae_ns_tile, zero)
        - jnp.minimum(subsinksoil, zero)
        + precisol_ns_tile
    )
    water_to_extract = jnp.maximum(ae_ns_tile, zero) + jnp.maximum(subsinksoil, zero)
    temp = jnp.minimum(water_to_infiltrate, water_to_extract)
    water2infilt_out = water_to_infiltrate - temp
    water2extract = water_to_extract - temp
    ae_ns_out = ae_ns_tile + subsinksoil

    return HydrolSoilSurfaceWaterSetupResult(
        temp=temp,
        water2infilt=water2infilt_out,
        water2extract=water2extract,
        ae_ns=ae_ns_out,
        flux_top=water2extract,
        flux_infilt=water2infilt_out,
    )


def hydrol_soil_infilt_explicit(
    *,
    mc,
    flux_infilt,
    k,
    dz_mm,
    dt_days,
    mcs,
    ks,
    kfact,
    kfact_root,
    njsc=None,
    soil_tile_index=1,
    peat_hydro=False,
    peat_tiles=(4, 5, 6),
    mcs_peat=None,
    ks_peat=None,
    kfact_peat=None,
    ok_freeze_cwrr=False,
    temp_hydro=None,
    zero_celsius=273.15,
    min_sechiba=1.0e-8,
    check_cwrr2=False,
):
    """Run ``hydrol_soil_infilt`` for one explicit soil tile.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil_infilt``, lines 7432-7590. This helper updates
    one tile's ``mc`` profile from a supplied ``flux_infilt`` using exact
    hydraulic state inputs. It does not compute ``k``/``kfact_root`` or choose
    the active soil tile.
    """

    mc = jnp.asarray(mc)
    if mc.ndim == 1:
        mc = mc[None, :]
    if mc.ndim != 2:
        raise ValueError("mc must have shape (npts, nslm) or (nslm,)")
    flux_infilt = _as_1d("flux_infilt", flux_infilt)
    k = jnp.asarray(k)
    if k.ndim == 1:
        k = k[None, :]
    if k.shape != mc.shape:
        raise ValueError("k must have the same shape as mc")
    dz_mm = _as_1d("dz_mm", dz_mm)
    if dz_mm.shape[0] != mc.shape[1]:
        raise ValueError("dz_mm length must match nslm")
    mcs = _as_1d("mcs", mcs)
    kfact_root = jnp.asarray(kfact_root)
    if kfact_root.ndim == 1:
        kfact_root = kfact_root[None, :]
    if kfact_root.shape != mc.shape:
        raise ValueError("kfact_root must have the same shape as mc")
    _require_same_npts(mc, flux_infilt, k, mcs, kfact_root)

    npts, nslm = mc.shape
    if nslm < 2:
        raise ValueError("hydrol_soil_infilt requires at least two soil layers")
    dtype = mc.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    two = jnp.asarray(2.0, dtype=dtype)
    minv = jnp.asarray(min_sechiba, dtype=dtype)
    dt_days = jnp.asarray(dt_days, dtype=dtype)
    soil_tile_index = int(soil_tile_index)
    peat_active = bool(peat_hydro) and soil_tile_index in tuple(int(tile) for tile in peat_tiles)

    if peat_active:
        if mcs_peat is None or ks_peat is None or kfact_peat is None:
            raise ValueError("peat infiltration requires mcs_peat, ks_peat, and kfact_peat")
        sat = jnp.broadcast_to(jnp.asarray(mcs_peat, dtype=dtype), (npts,))
        ks_profile = jnp.broadcast_to(jnp.asarray(ks_peat, dtype=dtype) * _as_1d("kfact_peat", kfact_peat), (npts, nslm))
    else:
        sat = mcs
        if njsc is None:
            texture_index = jnp.zeros((npts,), dtype=jnp.int32)
        else:
            texture_index = _as_1d("njsc", njsc).astype(jnp.int32) - 1
        ks = jnp.asarray(ks, dtype=dtype)
        kfact = jnp.asarray(kfact, dtype=dtype)
        if ks.ndim == 0:
            ks_profile = jnp.broadcast_to(ks, (npts, nslm))
        elif ks.ndim == 1 and ks.shape[0] == npts:
            ks_profile = ks[:, None]
        elif ks.ndim == 1:
            ks_profile = ks[texture_index][:, None]
        else:
            raise ValueError("ks must be scalar, per-point, or texture table")
        if kfact.ndim == 1:
            kfact_profile = jnp.broadcast_to(kfact, (npts, nslm))
        elif kfact.ndim == 2 and kfact.shape[0] == nslm:
            kfact_profile = kfact[:, texture_index].T
        elif kfact.ndim == 2 and kfact.shape == (npts, nslm):
            kfact_profile = kfact
        else:
            raise ValueError("kfact must have shape (nslm,), (nslm,ntexture), or (npts,nslm)")
        ks_profile = ks_profile * kfact_profile

    if ok_freeze_cwrr:
        if temp_hydro is None:
            raise ValueError("temp_hydro is required when ok_freeze_cwrr=True")
        temp_hydro = jnp.asarray(temp_hydro)
        if temp_hydro.ndim == 1:
            temp_hydro = temp_hydro[None, :]
        if temp_hydro.shape != mc.shape:
            raise ValueError("temp_hydro must have the same shape as mc")
    else:
        temp_hydro = jnp.full_like(mc, jnp.asarray(zero_celsius, dtype=dtype))

    tmci = hydrol_total_moisture_content(mc, dz_mm) if check_cwrr2 else None
    mc_out = mc
    wat_inf_pot = jnp.maximum((sat - mc_out[:, 0]) * dz_mm[1] / two, zero)
    wat_inf_first = jnp.minimum(wat_inf_pot, flux_infilt)
    mc_out = mc_out.at[:, 0].add(wat_inf_first * two / dz_mm[1])

    dt_tmp = jnp.full((npts,), dt_days, dtype=dtype)
    infilt_tot = wat_inf_first
    flux_tmp = (flux_infilt - wat_inf_first) / dt_tmp

    for jsl in range(1, nslm - 1):
        k_m = (k[:, jsl] + ks_profile[:, jsl - 1] * kfact_root[:, jsl]) / two
        k_m = jnp.where(
            bool(ok_freeze_cwrr) & (temp_hydro[:, jsl] < jnp.asarray(zero_celsius, dtype=dtype)),
            k[:, jsl],
            k_m,
        )
        infilt_tmp = k_m * (1.0 - jnp.exp(-flux_tmp / k_m))
        wat_inf_pot = jnp.maximum((sat - mc_out[:, jsl]) * (dz_mm[jsl] + dz_mm[jsl + 1]) / two, zero)
        safe_infilt_tmp = jnp.where(infilt_tmp > minv, infilt_tmp, 1.0)
        dt_inf = jnp.where(infilt_tmp > minv, jnp.minimum(wat_inf_pot / safe_infilt_tmp, dt_tmp), dt_tmp)
        overshoots_available = dt_inf * infilt_tmp > flux_infilt - infilt_tot
        dt_inf = jnp.where(
            (infilt_tmp > minv) & overshoots_available,
            jnp.maximum(flux_infilt - infilt_tot, zero) / safe_infilt_tmp,
            dt_inf,
        )
        wat_inf = dt_inf * infilt_tmp
        mc_out = mc_out.at[:, jsl].add(wat_inf * two / (dz_mm[jsl] + dz_mm[jsl + 1]))
        dt_tmp = dt_tmp - dt_inf
        infilt_tot = infilt_tot + infilt_tmp * dt_inf

    ru_infilt = flux_infilt - infilt_tot
    qinfilt = infilt_tot
    check = None
    if check_cwrr2:
        check = hydrol_total_moisture_content(mc_out, dz_mm) - (tmci + infilt_tot)

    return HydrolSoilInfiltResult(
        mc=mc_out,
        qinfilt=qinfilt,
        ru_infilt=ru_infilt,
        infilt_tot=infilt_tot,
        wat_inf_first=wat_inf_first,
        dt_remaining=dt_tmp,
        check=check,
    )


def hydrol_soil_tile_explicit_step(
    *,
    water2infilt,
    ae_ns_tile,
    subsinksoil,
    precisol_ns_tile,
    reinfiltration_soil,
    is_crop_soil,
    mc,
    mcl,
    profil_froz,
    mcr,
    mcs,
    a=None,
    b=None,
    d=None,
    k=None,
    k_after_infilt=None,
    dz_mm,
    dt_days,
    free_drain_coef,
    rootsink,
    resolv,
    mask_soiltile,
    kfact_root,
    ks,
    kfact,
    njsc=None,
    mineral_tables: MineralCWRRTables | None = None,
    peat_tables: PeatCWRRTables | None = None,
    coef_point_index=None,
    irrigation_soil_tile=None,
    reinf_slope=None,
    doponds=False,
    soil_tile_index=1,
    peat_hydro=False,
    peat_tiles=(4, 5, 6),
    mcs_peat=None,
    ks_peat=None,
    kfact_peat=None,
    ok_freeze_cwrr=False,
    temp_hydro=None,
    check_infilt=False,
    check_tr_ns=None,
    clip_over_mcs=False,
    route_over_mcs=False,
    ok_freeze_cwrr_solve=None,
    check_over_mcs=False,
    correct_negative_runoff=False,
    force_water_table=False,
    zwt_force=None,
    zz_mm=None,
    zmaxh_m=None,
    diagnose_wtd=False,
    undef_sechiba=1.0e20,
    smooth_under_mcr=False,
    check_under_mcr=False,
):
    """Compose a longer explicit-input ``hydrol_soil`` tile step.

    Fortran provenance: ``hydrol_soil`` surface setup lines 5800-5847,
    infiltration call line 5864 with ``hydrol_soil_infilt`` lines 7432-7590,
    runoff reinfiltration update lines 5871-5880, setup line 5896 via
    ``hydrol_soil_coef`` line 5858 and 5891 via lines 8248-8312,
    ``hydrol_soil_setup`` lines 8480-8568, RHS/solve/post-solve lines
    5953-6075. This adapter requires exact inputs and deliberately stops
    before full over/under-residual smoothing, diagnostics, and later
    irrigation/water-table branches.
    """

    table_mode = mineral_tables is not None or peat_tables is not None
    if mineral_tables is not None and peat_tables is not None:
        raise ValueError("pass either mineral_tables or peat_tables, not both")
    if table_mode and any(value is not None for value in (a, b, d, k, k_after_infilt)):
        raise ValueError("pass either explicit a/b/d/k inputs or coefficient tables, not both")
    if not table_mode and any(value is None for value in (a, b, d, k)):
        raise ValueError("explicit a, b, d, and k are required when coefficient tables are not provided")

    coef_before_infilt = None
    coef_after_infilt = None
    peat_active = bool(peat_hydro) and int(soil_tile_index) in tuple(int(tile) for tile in peat_tiles)
    if mineral_tables is not None:
        if peat_active:
            raise NotImplementedError("mineral_tables do not cover the Fortran peat hydrol_soil_coef branch")
        coef_before_infilt = hydrol_soil_coef_mineral_profile_from_tables(
            mc=mc,
            profil_froz=profil_froz,
            kfact_root=kfact_root,
            tables=mineral_tables,
            mcr=mcr,
            mcs=mcs,
            point_index=coef_point_index,
            ok_freeze_cwrr=ok_freeze_cwrr,
        )
        k_for_infilt = coef_before_infilt.k
    elif peat_tables is not None:
        if not peat_active:
            raise ValueError("peat_tables require peat_hydro=True and an active peat soil_tile_index")
        coef_before_infilt = hydrol_soil_coef_peat_profile_from_tables(
            mc=mc,
            profil_froz=profil_froz,
            kfact_root=kfact_root,
            tables=peat_tables,
            ok_freeze_cwrr=ok_freeze_cwrr,
        )
        k_for_infilt = coef_before_infilt.k
        mcs_peat = peat_tables.mcs
        ks_peat = peat_tables.ks
        kfact_peat = peat_tables.kfact
    else:
        k_for_infilt = k

    surface = hydrol_soil_surface_water_setup(
        water2infilt=water2infilt,
        ae_ns_tile=ae_ns_tile,
        subsinksoil=subsinksoil,
        precisol_ns_tile=precisol_ns_tile,
        reinfiltration_soil=reinfiltration_soil,
        irrigation_soil_tile=irrigation_soil_tile,
        is_crop_soil=is_crop_soil,
    )
    infilt = hydrol_soil_infilt_explicit(
        mc=mc,
        flux_infilt=surface.flux_infilt,
        k=k_for_infilt,
        dz_mm=dz_mm,
        dt_days=dt_days,
        mcs=mcs,
        ks=ks,
        kfact=kfact,
        kfact_root=kfact_root,
        njsc=njsc,
        soil_tile_index=soil_tile_index,
        peat_hydro=peat_hydro,
        peat_tiles=peat_tiles,
        mcs_peat=mcs_peat,
        ks_peat=ks_peat,
        kfact_peat=kfact_peat,
        ok_freeze_cwrr=ok_freeze_cwrr,
        temp_hydro=temp_hydro,
        check_cwrr2=check_infilt,
    )
    if mineral_tables is not None:
        coef_after_infilt = hydrol_soil_coef_mineral_profile_from_tables(
            mc=infilt.mc,
            profil_froz=profil_froz,
            kfact_root=kfact_root,
            tables=mineral_tables,
            mcr=mcr,
            mcs=mcs,
            point_index=coef_point_index,
            ok_freeze_cwrr=ok_freeze_cwrr,
        )
        a_for_setup = coef_after_infilt.a
        b_for_solve = coef_after_infilt.b
        d_for_setup = coef_after_infilt.d
        k_for_solve = coef_after_infilt.k
    elif peat_tables is not None:
        coef_after_infilt = hydrol_soil_coef_peat_profile_from_tables(
            mc=infilt.mc,
            profil_froz=profil_froz,
            kfact_root=kfact_root,
            tables=peat_tables,
            ok_freeze_cwrr=ok_freeze_cwrr,
        )
        a_for_setup = coef_after_infilt.a
        b_for_solve = coef_after_infilt.b
        d_for_setup = coef_after_infilt.d
        k_for_solve = coef_after_infilt.k
    else:
        a_for_setup = a
        b_for_solve = b
        d_for_setup = d
        k_for_solve = k if k_after_infilt is None else k_after_infilt

    if doponds:
        water2infilt_after_reinf = jnp.zeros_like(infilt.ru_infilt)
    else:
        if reinf_slope is None:
            raise ValueError("reinf_slope is required when doponds=False")
        water2infilt_after_reinf = _as_1d("reinf_slope", reinf_slope) * infilt.ru_infilt
    setup = hydrol_soil_setup_coefficients(
        a=a_for_setup,
        d=d_for_setup,
        dz_mm=dz_mm,
        dt_days=dt_days,
        free_drain_coef=free_drain_coef,
    )
    mcl_before = hydrol_mc_to_mcl(infilt.mc, profil_froz, mcr)
    k_array = jnp.asarray(k_for_solve)
    if k_array.ndim == 1:
        k_bottom = k_array[None, -1]
    else:
        k_bottom = k_array[:, -1]
    solve = hydrol_soil_explicit_solve_step(
        setup=setup,
        mcl_before=mcl_before,
        mc_before=infilt.mc,
        profil_froz=profil_froz,
        mcr=mcr,
        b=b_for_solve,
        dz_mm=dz_mm,
        dt_days=dt_days,
        free_drain_coef=free_drain_coef,
        flux_top=surface.flux_top,
        rootsink=rootsink,
        resolv=resolv,
        mask_soiltile=mask_soiltile,
        k_bottom=k_bottom,
        check_tr_ns=check_tr_ns,
        mcs=mcs,
        clip_over_mcs=clip_over_mcs,
        ru_ns=infilt.ru_infilt - water2infilt_after_reinf,
        route_over_mcs=route_over_mcs,
        ok_freeze_cwrr=ok_freeze_cwrr if ok_freeze_cwrr_solve is None else ok_freeze_cwrr_solve,
        check_over_mcs=check_over_mcs,
        correct_negative_runoff=correct_negative_runoff,
        force_water_table=force_water_table,
        zwt_force=zwt_force,
        zz_mm=zz_mm,
        mcs_peat=mcs_peat if mcs_peat is not None else PEAT_MCS,
        diagnose_wtd=diagnose_wtd,
        undef_sechiba=undef_sechiba,
        smooth_under_mcr=smooth_under_mcr,
        zmaxh_m=zmaxh_m,
        peat_hydro=peat_hydro,
        soil_tile_index=soil_tile_index,
        peat_tiles=peat_tiles,
        mcr_peat=PEAT_MCR,
        check_under_mcr=check_under_mcr,
    )

    return HydrolSoilTileExplicitStepResult(
        surface=surface,
        infilt=infilt,
        coef_before_infilt=coef_before_infilt,
        coef_after_infilt=coef_after_infilt,
        setup=setup,
        solve=solve,
        mclint_for_flux=mcl_before,
        k_bottom=k_bottom,
        ru_ns_after_reinf=infilt.ru_infilt - water2infilt_after_reinf,
        water2infilt_after_reinf=water2infilt_after_reinf,
    )


def hydrol_soil_all_tiles_explicit_step(
    *,
    water2infilt,
    ae_ns,
    subsinksoil,
    precisol_ns,
    reinfiltration_soil,
    is_crop_soil,
    mc,
    mcl,
    profil_froz,
    mcr,
    mcs,
    dz_mm,
    dt_days,
    free_drain_coef,
    rootsink,
    resolv,
    mask_soiltile,
    kfact_root,
    ks,
    kfact,
    soiltile,
    tmc,
    evapot=None,
    evapot_corr=None,
    frac_bare_ns=None,
    njsc=None,
    a=None,
    b=None,
    d=None,
    k=None,
    k_after_infilt=None,
    mineral_tables: MineralCWRRTables | None = None,
    peat_tables: PeatCWRRTables | None = None,
    irrigation_soil=None,
    reinf_slope=None,
    doponds=False,
    peat_hydro=False,
    peat_tiles=(4, 5, 6),
    ok_freeze_cwrr=False,
    temp_hydro=None,
    zwt_force=None,
    zz_mm=None,
    zmaxh_m=None,
    run2peat=None,
    run2man=None,
    wt_ab=None,
    wt_ab_tide=None,
    runoff2peat=None,
    tides=False,
    ok_ru2peat=False,
    ok_wt_ab=False,
    agri_peat=False,
    wtp_tide=0.0,
    dwtp_tide=0.0,
    max_wt_ab=100.0,
    min_sechiba=1.0e-8,
    undef_sechiba=1.0e20,
):
    """Run the audited HYDROL soil chain for every soil tile explicitly.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil``, source-order tile loop from the surface setup
    at lines 5800-5847 through post-solve routing lines 6183-6296, followed by
    the post-loop runoff reinjection and ``tmc`` update at lines 6757-6787.
    Inputs remain explicit; this function does not compute upstream forcing,
    PFT-to-soil splits, or diagnostic moisture stress.
    """

    water2infilt_state = _as_2d("water2infilt", water2infilt)
    ae_ns = _as_2d("ae_ns", ae_ns)
    precisol_ns = _as_2d("precisol_ns", precisol_ns)
    reinfiltration_soil = _as_2d("reinfiltration_soil", reinfiltration_soil)
    mc_state = jnp.asarray(mc)
    mcl_state = jnp.asarray(mcl)
    profil_froz = jnp.asarray(profil_froz)
    free_drain_coef = _as_2d("free_drain_coef", free_drain_coef)
    rootsink = jnp.asarray(rootsink)
    resolv = _as_2d("resolv", resolv)
    mask_soiltile = _as_2d("mask_soiltile", mask_soiltile)
    kfact_root = jnp.asarray(kfact_root)
    soiltile = _as_2d("soiltile", soiltile)
    tmc_state = _as_2d("tmc", tmc)
    if evapot is None:
        evapot = jnp.zeros((water2infilt_state.shape[0],), dtype=water2infilt_state.dtype)
    else:
        evapot = _as_1d("evapot", evapot)
    if evapot_corr is None:
        evapot_corr = evapot
    else:
        evapot_corr = jnp.asarray(evapot_corr)
        if evapot_corr.ndim not in (1, 2):
            raise ValueError("evapot_corr must have shape (npts,) or (npts,nstm)")
    if frac_bare_ns is None:
        frac_bare_ns = jnp.where(soiltile > 0.0, 1.0, 0.0)
    else:
        frac_bare_ns = _as_2d("frac_bare_ns", frac_bare_ns)
    if mc_state.ndim != 3 or mcl_state.ndim != 3 or profil_froz.ndim != 3 or kfact_root.ndim != 3:
        raise ValueError("mc, mcl, profil_froz, and kfact_root must have shape (npts,nslm,nstm)")
    if rootsink.ndim != 3:
        raise ValueError("rootsink must have shape (npts,nslm,nstm)")

    npts, nslm, nstm = mc_state.shape
    crop_tiles = np.asarray(is_crop_soil, dtype=bool)
    if crop_tiles.ndim == 0:
        crop_tiles = np.full((nstm,), bool(crop_tiles), dtype=bool)
    if crop_tiles.shape != (nstm,):
        raise ValueError("is_crop_soil must be scalar or have shape (nstm,)")
    for name, arr in (
        ("water2infilt", water2infilt_state),
        ("ae_ns", ae_ns),
        ("precisol_ns", precisol_ns),
        ("reinfiltration_soil", reinfiltration_soil),
        ("free_drain_coef", free_drain_coef),
        ("resolv", resolv),
        ("mask_soiltile", mask_soiltile),
        ("soiltile", soiltile),
        ("frac_bare_ns", frac_bare_ns),
        ("tmc", tmc_state),
    ):
        if arr.shape != (npts, nstm):
            raise ValueError(f"{name} must have shape (npts,nstm)")
    if evapot.shape != (npts,):
        raise ValueError("evapot must have shape (npts,)")
    if evapot_corr.shape not in ((npts,), (npts, nstm)):
        raise ValueError("evapot_corr must have shape (npts,) or (npts,nstm)")
    if mcl_state.shape != mc_state.shape or profil_froz.shape != mc_state.shape or kfact_root.shape != mc_state.shape:
        raise ValueError("mc, mcl, profil_froz, and kfact_root must have matching shapes")
    if rootsink.shape != mc_state.shape:
        raise ValueError("rootsink must match mc shape")

    subsinksoil = _as_1d("subsinksoil", subsinksoil)
    mcr = _as_1d("mcr", mcr)
    mcs = _as_1d("mcs", mcs)
    _require_same_npts(water2infilt_state, subsinksoil, mcr, mcs)
    if reinf_slope is None and not doponds:
        raise ValueError("reinf_slope is required when doponds=False")
    if reinf_slope is not None:
        reinf_slope = _as_1d("reinf_slope", reinf_slope)
        _require_same_npts(water2infilt_state, reinf_slope)
    if irrigation_soil is None:
        irrigation_soil = jnp.zeros_like(water2infilt_state)
    else:
        irrigation_soil = _as_2d("irrigation_soil", irrigation_soil)
    if run2peat is None:
        run2peat = jnp.zeros((npts,), dtype=water2infilt_state.dtype)
    else:
        run2peat = _as_1d("run2peat", run2peat)
    if run2man is None:
        run2man = jnp.zeros((npts,), dtype=water2infilt_state.dtype)
    else:
        run2man = _as_1d("run2man", run2man)
    if wt_ab is None:
        wt_ab = jnp.zeros((npts,), dtype=water2infilt_state.dtype)
    else:
        wt_ab = _as_1d("wt_ab", wt_ab)
    if wt_ab_tide is None:
        wt_ab_tide = jnp.zeros((npts,), dtype=water2infilt_state.dtype)
    else:
        wt_ab_tide = _as_1d("wt_ab_tide", wt_ab_tide)
    if runoff2peat is None:
        runoff2peat = jnp.zeros_like(water2infilt_state)
    else:
        runoff2peat = _as_2d("runoff2peat", runoff2peat)

    if zwt_force is None:
        zwt_force = jnp.full((npts, nstm), jnp.inf, dtype=water2infilt_state.dtype)
    else:
        zwt_force = _as_2d("zwt_force", zwt_force)
    if zz_mm is None:
        raise ValueError("zz_mm is required for all-tile HYDROL explicit step")
    if zmaxh_m is None:
        raise ValueError("zmaxh_m is required for all-tile HYDROL explicit step")

    tile_results = []
    ru_ns_state = jnp.zeros_like(water2infilt_state)
    dr_ns_state = jnp.zeros_like(water2infilt_state)
    qflux_state = jnp.zeros_like(mc_state)
    wtd_ns_state = jnp.full_like(water2infilt_state, undef_sechiba)
    is_under_mcr_state = jnp.zeros((npts, nstm), dtype=bool)

    for tile_offset in range(nstm):
        soil_tile_index = tile_offset + 1
        peat_active = bool(peat_hydro) and soil_tile_index in tuple(int(tile) for tile in peat_tiles)
        tile_kwargs = {
            "water2infilt": water2infilt_state[:, tile_offset],
            "ae_ns_tile": ae_ns[:, tile_offset],
            "subsinksoil": subsinksoil,
            "precisol_ns_tile": precisol_ns[:, tile_offset],
            "reinfiltration_soil": reinfiltration_soil[:, tile_offset],
            "is_crop_soil": bool(crop_tiles[tile_offset]),
            "mc": mc_state[:, :, tile_offset],
            "mcl": mcl_state[:, :, tile_offset],
            "profil_froz": profil_froz[:, :, tile_offset],
            "mcr": mcr if not peat_active else jnp.full_like(mcr, PEAT_MCR),
            "mcs": mcs if not peat_active else jnp.full_like(mcs, PEAT_MCS),
            "dz_mm": dz_mm,
            "dt_days": dt_days,
            "free_drain_coef": free_drain_coef[:, tile_offset],
            "rootsink": rootsink[:, :, tile_offset],
            "resolv": resolv[:, tile_offset],
            "mask_soiltile": mask_soiltile[:, tile_offset],
            "kfact_root": kfact_root[:, :, tile_offset],
            "ks": ks,
            "kfact": kfact,
            "njsc": njsc,
            "irrigation_soil_tile": irrigation_soil[:, tile_offset],
            "reinf_slope": reinf_slope,
            "doponds": doponds,
            "soil_tile_index": soil_tile_index,
            "peat_hydro": peat_hydro,
            "peat_tiles": peat_tiles,
            "ok_freeze_cwrr": ok_freeze_cwrr,
            "temp_hydro": temp_hydro,
            "route_over_mcs": True,
            "correct_negative_runoff": True,
            "force_water_table": True,
            "zwt_force": zwt_force[:, tile_offset],
            "zz_mm": zz_mm,
            "zmaxh_m": zmaxh_m,
            "diagnose_wtd": True,
            "undef_sechiba": undef_sechiba,
            "smooth_under_mcr": True,
        }
        if peat_active:
            if peat_tables is None:
                raise ValueError("peat_tables are required for active peat tiles")
            tile_kwargs["peat_tables"] = peat_tables
        elif mineral_tables is not None:
            tile_kwargs["mineral_tables"] = mineral_tables
        else:
            if any(value is None for value in (a, b, d, k)):
                raise ValueError("explicit a/b/d/k are required for non-table all-tile mode")
            tile_kwargs.update(
                a=jnp.asarray(a)[:, :, tile_offset],
                b=jnp.asarray(b)[:, :, tile_offset],
                d=jnp.asarray(d)[:, :, tile_offset],
                k=jnp.asarray(k)[:, :, tile_offset],
            )
            if k_after_infilt is not None:
                tile_kwargs["k_after_infilt"] = jnp.asarray(k_after_infilt)[:, :, tile_offset]

        result = hydrol_soil_tile_explicit_step(**tile_kwargs)
        tile_results.append(result)
        # Fortran hydrol_soil section 4.2 recomputes module mcl from the final
        # post-correction mc profile before diagnostics/export.
        final_mcl = hydrol_mc_to_mcl(
            result.solve.mc_final,
            profil_froz[:, :, tile_offset],
            tile_kwargs["mcr"],
        )
        mc_state = mc_state.at[:, :, tile_offset].set(result.solve.mc_final)
        mcl_state = mcl_state.at[:, :, tile_offset].set(final_mcl)
        ru_ns_tile = result.solve.ru_ns_final
        if ru_ns_tile is None:
            ru_ns_tile = result.ru_ns_after_reinf
        ru_ns_state = ru_ns_state.at[:, tile_offset].set(ru_ns_tile)
        dr_ns_tile = jnp.where(
            mask_soiltile[:, tile_offset] > 0,
            result.solve.dr_ns_final,
            0.0,
        )
        dr_ns_state = dr_ns_state.at[:, tile_offset].set(dr_ns_tile)
        water2infilt_state = water2infilt_state.at[:, tile_offset].set(result.water2infilt_after_reinf)
        if result.solve.water_table_depth is not None:
            wtd_ns_state = wtd_ns_state.at[:, tile_offset].set(result.solve.water_table_depth.wtd_ns)
        if result.solve.under_mcr is not None:
            is_under_mcr_state = is_under_mcr_state.at[:, tile_offset].set(result.solve.under_mcr.is_under_mcr)
        flux_diag = hydrol_soil_flux_diagnostics(
            mcl_after=final_mcl,
            mcl_before=result.mclint_for_flux,
            dr_ns=dr_ns_tile,
            rootsink=rootsink[:, :, tile_offset],
            dz_mm=dz_mm,
            flux_top=result.surface.flux_top,
        )
        qflux_state = qflux_state.at[:, :, tile_offset].set(flux_diag.qflux)

        routing = hydrol_after_under_mcr_runoff_peat_tide_routing(
            soil_tile_index=soil_tile_index,
            ru_ns=ru_ns_state,
            runoff2peat=runoff2peat,
            soiltile=soiltile,
            water2infilt=water2infilt_state,
            run2peat=run2peat,
            run2man=run2man,
            wt_ab=wt_ab,
            wt_ab_tide=wt_ab_tide,
            peat_hydro=peat_hydro,
            ok_ru2peat=ok_ru2peat,
            ok_wt_ab=ok_wt_ab,
            tides=tides,
            wtp_tide=wtp_tide,
            dwtp_tide=dwtp_tide,
            min_sechiba=min_sechiba,
            max_wt_ab=max_wt_ab,
        )
        ru_ns_state = routing.ru_ns
        runoff2peat = routing.runoff2peat
        water2infilt_state = routing.water2infilt
        run2peat = routing.run2peat
        run2man = routing.run2man
        wt_ab = routing.wt_ab
        wt_ab_tide = routing.wt_ab_tide

        tmc_state = tmc_state.at[:, tile_offset].set(hydrol_total_moisture_content(mc_state[:, :, tile_offset], dz_mm))

    post = hydrol_runoff_peat_post_loop_reinjection(
        ru_ns=ru_ns_state,
        water2infilt=water2infilt_state,
        tmc=tmc_state,
        soiltile=soiltile,
        run2peat=run2peat,
        run2man=run2man,
        wt_ab=wt_ab,
        peat_hydro=peat_hydro,
        agri_peat=agri_peat,
        ok_ru2peat=ok_ru2peat,
        tides=tides,
        min_sechiba=min_sechiba,
    )
    evap_bare_limit_alt = hydrol_alt_residual_all_tiles_from_results(
        tuple(tile_results),
        profil_froz=profil_froz,
        kfact_root=kfact_root,
        mineral_tables=mineral_tables,
        peat_tables=peat_tables,
        mcr=mcr,
        mcs=mcs,
        dz_mm=dz_mm,
        dt_days=dt_days,
        free_drain_coef=free_drain_coef,
        resolv=resolv,
        mask_soiltile=mask_soiltile,
        evapot=evapot,
        evapot_penm=evapot_corr,
        frac_bare_ns=frac_bare_ns,
        peat_hydro=peat_hydro,
        peat_tiles=peat_tiles,
        ok_freeze_cwrr=ok_freeze_cwrr,
        min_sechiba=min_sechiba,
    )
    tmcint_for_evap_bare_limit = post.tmc - post.water2infilt

    return HydrolSoilAllTilesExplicitResult(
        tile_results=tuple(tile_results),
        mc=mc_state,
        mcl=mcl_state,
        profil_froz_hydro_ns=profil_froz,
        qflux=qflux_state,
        ru_ns=post.ru_ns,
        dr_ns=dr_ns_state,
        water2infilt=post.water2infilt,
        tmc=post.tmc,
        tmc_soil=post.tmc_soil,
        wtd_ns=wtd_ns_state,
        is_under_mcr=is_under_mcr_state,
        runoff2peat=runoff2peat,
        run2peat=post.run2peat,
        run2man=run2man,
        wt_ab=wt_ab,
        wt_ab_tide=wt_ab_tide,
        evap_bare_limit_alt=evap_bare_limit_alt,
        tmcint_for_evap_bare_limit=tmcint_for_evap_bare_limit,
    )


def hydrol_module_explicit_step(
    *,
    precip_rain,
    vevapwet,
    veget,
    veget_max,
    qsintmax,
    qsintveg,
    tot_melt,
    vegtot,
    throughfall_by_pft,
    pref_soil_veg,
    soiltile,
    vevapflo,
    flood_frac,
    flood_res,
    subsinksoil,
    vevapnu,
    vevapnu_pft,
    transpir,
    humrel,
    humrelv,
    us,
    ae_ns,
    evap_bare_lim,
    evap_bare_lim_ns,
    evapot=None,
    evapot_corr=None,
    tot_bare_soil=None,
    water2infilt=None,
    reinfiltration_soil=None,
    is_crop_soil=False,
    mc=None,
    mcl=None,
    profil_froz=None,
    mcr=None,
    mcs=None,
    dz_mm=None,
    dt_days=None,
    free_drain_coef=None,
    resolv=None,
    kfact_root=None,
    ks=None,
    kfact=None,
    tmc=None,
    njsc=None,
    mineral_tables: MineralCWRRTables | None = None,
    peat_tables: PeatCWRRTables | None = None,
    irrigation_soil=None,
    reinf_slope=None,
    doponds=False,
    peat_hydro=False,
    peat_tiles=(4, 5, 6),
    ok_freeze_cwrr=False,
    temp_hydro=None,
    zwt_force=None,
    zz_mm=None,
    zmaxh_m=None,
    run2peat=None,
    run2man=None,
    wt_ab=None,
    wt_ab_tide=None,
    runoff2peat=None,
    tides=False,
    ok_ru2peat=False,
    ok_wt_ab=False,
    agri_peat=False,
    wtp_tide=0.0,
    dwtp_tide=0.0,
    max_wt_ab=100.0,
    min_sechiba=1.0e-8,
    undef_sechiba=1.0e20,
):
    """Assemble the audited HYDROL module path through the soil loop.

    Fortran provenance: ``hydrol_main`` calls the relevant blocks in source
    order: ``hydrol_vegupd`` after state migration at lines 1206 and 5191-5248,
    ``hydrol_canop`` lines 1232-1234 and 4992-5117, ``hydrol_flood`` lines
    1237-1238 and 5276-5334, then ``hydrol_soil``/``hydrol_split_soil`` for
    the explicit soil-tile chain. Stateful reservoir migration in
    ``hydrol_tmc_update`` and later moisture-stress diagnostics remain outside
    this assembly.
    """

    veget = _as_2d("veget", veget)
    veget_max = _as_2d("veget_max", veget_max)
    soiltile = _as_2d("soiltile", soiltile)
    vegtot = _as_1d("vegtot", vegtot)
    _require_same_npts(veget, veget_max, soiltile, vegtot)
    npts = veget.shape[0]
    nstm = soiltile.shape[1]
    if water2infilt is None:
        water2infilt = jnp.zeros((npts, nstm), dtype=veget.dtype)
    if reinfiltration_soil is None:
        reinfiltration_soil = jnp.zeros((npts, nstm), dtype=veget.dtype)

    vegupd = hydrol_vegupd_static_state(
        veget=veget,
        veget_max=veget_max,
        soiltile=soiltile,
        vegtot=vegtot,
        pref_soil_veg=pref_soil_veg,
        min_sechiba=min_sechiba,
    )
    canop = hydrol_canop_interception(
        precip_rain=precip_rain,
        vevapwet=vevapwet,
        veget_max=veget_max,
        veget=veget,
        qsintmax=qsintmax,
        qsintveg=qsintveg,
        tot_melt=tot_melt,
        vegtot=vegtot,
        throughfall_by_pft=throughfall_by_pft,
        min_sechiba=min_sechiba,
    )
    flood = hydrol_flood_reservoir(
        vevapflo=vevapflo,
        flood_frac=flood_frac,
        flood_res=flood_res,
        precisol=canop.precisol,
        subsinksoil=subsinksoil,
    )
    if tot_bare_soil is None:
        tot_bare_soil = jnp.sum(vegupd.frac_bare * veget_max, axis=1)
    split = hydrol_split_soil_fluxes(
        veget_max=veget_max,
        soiltile=soiltile,
        precisol=flood.precisol,
        vevapnu=vevapnu,
        vevapnu_pft=vevapnu_pft,
        transpir=transpir,
        humrel=humrel,
        humrelv=humrelv,
        us=us,
        ae_ns=ae_ns,
        evap_bare_lim=evap_bare_lim,
        evap_bare_lim_ns=evap_bare_lim_ns,
        frac_bare_ns=vegupd.frac_bare_ns,
        tot_bare_soil=tot_bare_soil,
        vegetmax_soil=vegupd.vegetmax_soil,
        vegtot=vegtot,
        pref_soil_veg=pref_soil_veg,
        min_sechiba=min_sechiba,
    )
    soil = hydrol_soil_all_tiles_explicit_step(
        water2infilt=water2infilt,
        ae_ns=split.ae_ns,
        subsinksoil=flood.subsinksoil,
        precisol_ns=split.precisol_ns,
        reinfiltration_soil=reinfiltration_soil,
        is_crop_soil=is_crop_soil,
        mc=mc,
        mcl=mcl,
        profil_froz=profil_froz,
        mcr=mcr,
        mcs=mcs,
        dz_mm=dz_mm,
        dt_days=dt_days,
        free_drain_coef=free_drain_coef,
        rootsink=split.rootsink,
        resolv=resolv,
        mask_soiltile=vegupd.mask_soiltile,
        kfact_root=kfact_root,
        ks=ks,
        kfact=kfact,
        soiltile=soiltile,
        tmc=tmc,
        evapot=evapot,
        evapot_corr=evapot_corr,
        frac_bare_ns=vegupd.frac_bare_ns,
        njsc=njsc,
        mineral_tables=mineral_tables,
        peat_tables=peat_tables,
        irrigation_soil=irrigation_soil,
        reinf_slope=reinf_slope,
        doponds=doponds,
        peat_hydro=peat_hydro,
        peat_tiles=peat_tiles,
        ok_freeze_cwrr=ok_freeze_cwrr,
        temp_hydro=temp_hydro,
        zwt_force=zwt_force,
        zz_mm=zz_mm,
        zmaxh_m=zmaxh_m,
        run2peat=run2peat,
        run2man=run2man,
        wt_ab=wt_ab,
        wt_ab_tide=wt_ab_tide,
        runoff2peat=runoff2peat,
        tides=tides,
        ok_ru2peat=ok_ru2peat,
        ok_wt_ab=ok_wt_ab,
        agri_peat=agri_peat,
        wtp_tide=wtp_tide,
        dwtp_tide=dwtp_tide,
        max_wt_ab=max_wt_ab,
        min_sechiba=min_sechiba,
        undef_sechiba=undef_sechiba,
    )

    return HydrolModuleExplicitResult(
        vegupd=vegupd,
        canop=canop,
        flood=flood,
        split=split,
        soil=soil,
    )


def hydrol_module_outputs(result: HydrolModuleExplicitResult, *, soiltile):
    """Build HYDROL module-level output arrays from an explicit step.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil`` assigns ``drainage_per_soil = dr_ns`` and
    ``runoff_per_soil = ru_ns`` at lines 7361-7362, exposes ``runoff2peat`` as
    an output argument, and aggregates ``wtd`` across soil tiles at lines
    6806-6815.
    """

    soiltile = _as_2d("soiltile", soiltile)
    soil = result.soil
    if soiltile.shape != soil.ru_ns.shape:
        raise ValueError("soiltile shape must match runoff/drainage soil-tile arrays")
    wtd = jnp.sum(soiltile * soil.wtd_ns, axis=1)
    return HydrolModuleOutputs(
        runoff_per_soil=soil.ru_ns,
        drainage_per_soil=soil.dr_ns,
        runoff2peat=soil.runoff2peat,
        tmc=soil.tmc,
        tmc_soil=soil.tmc_soil,
        wat_flux=soil.qflux,
        wtd=wtd,
        wtd_ns=soil.wtd_ns,
        precip2canopy=result.canop.precip2canopy,
        precip2ground=result.canop.precip2ground,
        canopy2ground=result.canop.canopy2ground,
        floodout=result.flood.floodout,
    )


def _hydrol_main_top_diagnostics(
    *, mc, soiltile, vegtot, veget_max, nroot, dlh, tmcs, dz_mm, itopmax, zmaxh_m, min_sechiba
):
    """Evaluate Fortran ``hydrol.f90::hydrol_main`` lines 1330-1395 in statement order."""

    mc = _as_3d("mc", mc)
    soiltile = _as_2d("soiltile", soiltile)
    vegtot = _as_1d("vegtot", vegtot)
    veget_max = _as_2d("veget_max", veget_max)
    nroot = _as_3d("nroot", nroot)
    dlh = _as_1d("dlh", dlh)
    tmcs = _as_2d("tmcs", tmcs)
    dz_mm = _as_1d("dz_mm", dz_mm)
    npts, nslm, nstm = mc.shape
    if soiltile.shape != (npts, nstm) or tmcs.shape != (npts, nstm):
        raise ValueError("soiltile and tmcs must match mc landpoint/soil-tile axes")
    if nroot.shape[0] != npts or nroot.shape[2] != nslm or dlh.shape != (nslm,) or dz_mm.shape != (nslm,):
        raise ValueError("nroot, dlh, and dz_mm must match mc landpoint/layer axes")
    if not 1 <= int(itopmax) < nslm:
        raise ValueError("itopmax must be a Fortran one-based layer index below nslm")

    tmc_top = dz_mm[1] * (3.0 * mc[:, 0, :] + mc[:, 1, :]) / 8.0
    for jsl in range(1, int(itopmax)):
        tmc_top = tmc_top + dz_mm[jsl] * (3.0 * mc[:, jsl, :] + mc[:, jsl - 1, :]) / 8.0
        tmc_top = tmc_top + dz_mm[jsl + 1] * (3.0 * mc[:, jsl, :] + mc[:, jsl + 1, :]) / 8.0
    humtot_top = jnp.sum(soiltile * tmc_top, axis=1) * vegtot

    if veget_max.shape != nroot.shape[:2]:
        raise ValueError("veget_max must match the landpoint/PFT axes of nroot")
    rooted = jnp.sum(nroot[:, 1:, :] * veget_max[:, 1:, None], axis=1)
    land_nroot = jnp.where(vegtot[:, None] > min_sechiba, rooted / vegtot[:, None], 0.0)
    land_dlh = jnp.broadcast_to(dlh[None, :], (npts, nslm))
    land_mcs_1d = jnp.sum(soiltile * tmcs, axis=1) * vegtot
    land_mcs = jnp.broadcast_to(land_mcs_1d[:, None], (npts, nslm)) / (float(zmaxh_m) * 1000.0)
    return humtot_top, land_nroot, land_dlh, land_mcs


def hydrol_main_step(
    *,
    module_inputs: Mapping[str, object],
    diagnostic_inputs: Mapping[str, object],
    snow_inputs: Mapping[str, object],
    irrigation_inputs: Mapping[str, object],
    routing_inputs: Mapping[str, object],
    humcste,
    dlh,
    tmcs,
    itopmax: int,
    zmaxh_m: float,
    ok_explicitsnow: bool = True,
    topm_calcul: bool = False,
    topmodel_inputs: Mapping[str, object] | None = None,
    drain_upd=None,
    runoff_upd=None,
    natural=None,
    alma_inputs: Mapping[str, object] | None = None,
    check_waterbal: bool = False,
    dt_sechiba=1800.0,
    one_day=86400.0,
    min_sechiba=1.0e-8,
) -> HydrolMainResult:
    """Run PFT14-reachable ``hydrol.f90::hydrol_main`` lines 1159-1650.

    The process owners are called in source order: snow (1177-1197),
    vegetation/root factors (1200-1229), canopy/flood/irrigation/soil
    (1232-1315), then diagnostics and state exports (1317-1650). ``TOPM_calcul``
    is routed to the separately sourced ``hydro_subgrid_main`` owner; its
    false arm remains the source's six zero assignments.
    History/XIOS calls are represented by returned arrays, not IO side effects.
    """

    if alma_inputs is None:
        raise ValueError("alma_inputs are required by the unconditional hydrol_alma call at line 1327")
    if bool(topm_calcul) and topmodel_inputs is None:
        raise ValueError("TOPM_calcul requires raw hydro_subgrid_main inputs")
    module_args = dict(module_inputs)
    diagnostic_args = dict(diagnostic_inputs)
    diagnostic_args.setdefault("ok_freeze_cwrr", bool(module_args.get("ok_freeze_cwrr", False)))
    snow_args = dict(snow_inputs)
    irrigation_args = dict(irrigation_inputs)
    routing_args = dict(routing_inputs)
    veget_max = _as_2d("veget_max", module_args["veget_max"])
    npts, nvm = veget_max.shape
    zero = jnp.zeros((npts,), dtype=veget_max.dtype)
    fsat = zero
    fwet = zero
    fwt1 = zero
    fwt2 = zero
    fwt3 = zero
    fwt4 = zero
    if bool(topm_calcul):
        topmodel = hydro_subgrid_main(**dict(topmodel_inputs))
        if topmodel.fsat.shape != (npts,):
            raise ValueError("hydro_subgrid_main landpoint count must match hydrol_main")
        fsat = jnp.asarray(topmodel.fsat)
        fwet = jnp.asarray(topmodel.fwet)
        fwt1 = jnp.asarray(topmodel.fwt1)
        fwt2 = jnp.asarray(topmodel.fwt2)
        fwt3 = jnp.asarray(topmodel.fwt3)
        fwt4 = jnp.asarray(topmodel.fwt4)

    if bool(ok_explicitsnow):
        snow_step = hydrol_explicit_snow_step(
            **snow_args,
            dt_sechiba=dt_sechiba,
            one_day=one_day,
        )
        bucket_snow_step = None
        tot_melt = snow_step.snow_state.tot_melt
        subsinksoil = snow_step.subsinksoil
        vevapsno = snow_step.vevapsno
        snow = snow_step.snow_state.snow
        snow_nobio = snow_step.snow_state.snow_nobio
        snowmelt = snow_step.snowmelt
    else:
        bucket_snow_step = hydrol_bucket_snow_step(
            **snow_args,
            dt_sechiba=dt_sechiba,
            one_day=one_day,
        )
        snow_step = None
        tot_melt = bucket_snow_step.tot_melt
        subsinksoil = bucket_snow_step.subsinksoil
        vevapsno = bucket_snow_step.vevapsno
        snow = bucket_snow_step.snow
        snow_nobio = bucket_snow_step.snow_nobio
        snowmelt = bucket_snow_step.snowmelt

    vegupd = hydrol_vegupd_static_state(
        veget=module_args["veget"],
        veget_max=veget_max,
        soiltile=module_args["soiltile"],
        vegtot=module_args["vegtot"],
        pref_soil_veg=module_args["pref_soil_veg"],
        min_sechiba=min_sechiba,
    )
    kfact_root = hydrol_kfact_root_from_vegupd(
        vegetmax_soil=vegupd.vegetmax_soil,
        soiltile=module_args["soiltile"],
        pref_soil_veg=module_args["pref_soil_veg"],
        humcste=humcste,
        zz_mm=module_args["zz_mm"],
        njsc=module_args["njsc"],
        ks=module_args.get("ks", USDA_KS),
        peat_hydro=bool(module_args.get("peat_hydro", False)),
        min_sechiba=min_sechiba,
    )
    irrigation_demand = hydrol_irrigation_demand_ratio(**irrigation_args)
    routing_args["irrig_demand_ratio"] = irrigation_demand.irrig_demand_ratio
    routing = hydrol_routing_soil_water_split(**routing_args)

    module_args.update(
        tot_melt=tot_melt,
        subsinksoil=subsinksoil,
        kfact_root=kfact_root,
        reinfiltration_soil=jnp.broadcast_to(
            routing.reinfiltration_soil[:, None], _as_2d("soiltile", module_args["soiltile"]).shape
        ),
        irrigation_soil=routing.irrigation_soil,
    )
    module = hydrol_module_explicit_step(**module_args)
    outputs = hydrol_module_outputs(module, soiltile=module_args["soiltile"])
    diagnostic_args.update(
        veget_max=veget_max,
        soiltile=module_args["soiltile"],
        vegtot=module_args["vegtot"],
        tot_melt=tot_melt,
        irrigation=routing_args["irrigation"],
        returnflow=routing_args["returnflow"],
        reinfiltration=routing_args["reinfiltration"],
        evap_bare_limit_alt=module.soil.evap_bare_limit_alt,
        tmcint=module.soil.tmcint_for_evap_bare_limit,
    )
    if module.soil.evap_bare_limit_alt is not None and diagnostic_args.get("evapot") is None:
        diagnostic_args["evapot"] = jnp.zeros((npts,), dtype=veget_max.dtype)
    diagnostics = hydrol_module_diagnostics(module, **diagnostic_args)

    alma = {name: _as_1d(name, value) for name, value in alma_inputs.items()}
    required_alma = ("tot_watveg_beg", "tot_watsoil_beg", "snow_beg", "mx_eau_var")
    missing_alma = tuple(name for name in required_alma if name not in alma)
    if missing_alma:
        raise ValueError("alma_inputs missing hydrol_alma state: " + ", ".join(missing_alma))
    alma_step = hydrol_alma_step(
        qsintveg=module.canop.qsintveg,
        humtot=diagnostics.humtot,
        snow=snow,
        snow_nobio=snow_nobio,
        tot_watveg_beg=alma["tot_watveg_beg"],
        tot_watsoil_beg=alma["tot_watsoil_beg"],
        snow_beg=alma["snow_beg"],
        mx_eau_var=alma["mx_eau_var"],
        lstep_init=False,
    )
    delintercept = alma_step.delintercept
    delsoilmoist = alma_step.delsoilmoist
    delswe = alma_step.delswe
    soilwet = alma_step.soilwet
    assert delintercept is not None and delsoilmoist is not None and delswe is not None and soilwet is not None

    drain_upd = zero if drain_upd is None else _as_1d("drain_upd", drain_upd)
    runoff_upd = zero if runoff_upd is None else _as_1d("runoff_upd", runoff_upd)
    drainage = diagnostics.drainage + drain_upd
    runoff = diagnostics.runoff + runoff_upd
    water_balance = None
    if bool(check_waterbal):
        if "tot_water_beg" not in alma:
            raise ValueError("check_waterbal requires alma_inputs['tot_water_beg']")
        if "totfrac_nobio" not in module_args:
            raise ValueError("check_waterbal requires module_inputs['totfrac_nobio']")
        water_balance = hydrol_waterbal_step(
            tot_water_beg=alma["tot_water_beg"],
            vegtot=module_args["vegtot"],
            totfrac_nobio=module_args["totfrac_nobio"],
            qsintveg=module.canop.qsintveg,
            humtot=diagnostics.humtot,
            snow=snow,
            snow_nobio=snow_nobio,
            precip_rain=module_args["precip_rain"],
            precip_snow=snow_args["precip_snow"],
            returnflow=routing_args["returnflow"],
            reinfiltration=routing_args["reinfiltration"],
            irrigation=routing_args["irrigation"],
            vevapwet=module_args["vevapwet"],
            transpir=module_args["transpir"],
            vevapnu=diagnostics.vevapnu,
            vevapsno=vevapsno,
            vevapflo=module.flood.vevapflo,
            floodout=module.flood.floodout,
            runoff=runoff,
            drainage=drainage,
        )
    natural = np.ones((nvm,), dtype=bool) if natural is None else np.asarray(natural, dtype=bool)
    ok_laidev = np.asarray(irrigation_args["ok_laidev"], dtype=bool)
    if natural.shape != (nvm,):
        raise ValueError("natural must have shape (nvm,)")
    soil_deficit = _as_2d("soil_deficit", irrigation_args["soil_deficit"])
    pref = np.asarray(module_args["pref_soil_veg"], dtype=np.int32)
    is_crop = np.asarray(routing_args["is_crop_soil"], dtype=bool)
    irrig_fulfill = _as_1d("irrig_fulfill", irrigation_args["irrig_fulfill"])
    for jv in range(1, nvm):
        if not natural[jv] and ok_laidev[jv]:
            jst = int(pref[jv]) - 1
            if not is_crop[jst]:
                raise ValueError("hydrol_main lines 1298-1308 require irrigated PFTs on crop soil")
            value = jnp.maximum(0.0, irrig_fulfill[jv] * _as_2d("tmcs", tmcs)[:, jst] - module.soil.tmc[:, jst])
            soil_deficit = soil_deficit.at[:, jv].set(value)

    humtot_top, land_nroot, land_dlh, land_mcs = _hydrol_main_top_diagnostics(
        mc=module.soil.mc,
        soiltile=module_args["soiltile"],
        vegtot=module_args["vegtot"],
        veget_max=veget_max,
        nroot=diagnostics.nroot,
        dlh=dlh,
        tmcs=tmcs,
        dz_mm=module_args["dz_mm"],
        itopmax=itopmax,
        zmaxh_m=zmaxh_m,
        min_sechiba=min_sechiba,
    )

    twbr = (
        delsoilmoist + delintercept + delswe
        - (
            module_args["precip_rain"] + snow_args["precip_snow"] + routing_args["irrigation"]
            + module.flood.floodout + routing_args["returnflow"] + routing_args["reinfiltration"]
        )
        + (
            runoff + drainage + jnp.sum(module_args["vevapwet"], axis=1)
            + jnp.sum(module_args["transpir"], axis=1) + diagnostics.vevapnu + vevapsno
            + module.flood.vevapflo
        )
    )

    state_writeback = {
        "mc": module.soil.mc,
        "mcl": module.soil.mcl,
        "tmc": module.soil.tmc,
        "qsintveg": module.canop.qsintveg,
        "flood_res": module.flood.flood_res,
        "water2infilt": module.soil.water2infilt,
        "run2peat": module.soil.run2peat,
        "run2man": module.soil.run2man,
        "wt_ab": module.soil.wt_ab,
        "wt_ab_tide": module.soil.wt_ab_tide,
        "soil_deficit": soil_deficit,
        "tot_watveg_beg": alma_step.tot_watveg_beg,
        "tot_watsoil_beg": alma_step.tot_watsoil_beg,
        "snow_beg": alma_step.snow_beg,
        "snow": snow,
        "snow_nobio": snow_nobio,
        "snowmelt": snowmelt,
        "tot_melt": tot_melt,
    }
    if water_balance is not None:
        state_writeback["tot_water_beg"] = water_balance.tot_water_beg
    if snow_step is not None:
        state_writeback.update(
            {
                name: getattr(snow_step.snow_state, name)
                for name in (
                    "snow",
                    "snowdz",
                    "snowrho",
                    "snowtemp",
                    "snowheat",
                    "snowgrain",
                    "snow_age",
                    "snow_nobio",
                    "snow_nobio_age",
                    "tot_melt",
                )
            }
        )
        state_writeback["snowliq"] = snow_step.snowliq
        state_writeback["snowheat"] = snow_step.snow_state.snowheat
        state_writeback["snowgrain"] = snow_step.snow_state.snowgrain
    else:
        state_writeback["snowdepth"] = bucket_snow_step.snowdepth

    return HydrolMainResult(
        module=module,
        diagnostics=diagnostics,
        outputs=outputs,
        snow_step=snow_step,
        bucket_snow_step=bucket_snow_step,
        irrigation_demand=irrigation_demand,
        routing=routing,
        kfact_root=kfact_root,
        fsat=fsat,
        fwet=fwet,
        fwet_out=fwet,
        fwt1=fwt1,
        fwt2=fwt2,
        fwt3=fwt3,
        fwt4=fwt4,
        runoff=runoff,
        drainage=drainage,
        soil_deficit=soil_deficit,
        soilwet=soilwet,
        delintercept=delintercept,
        delsoilmoist=delsoilmoist,
        delswe=delswe,
        humtot_top=humtot_top,
        twbr=twbr,
        land_nroot=land_nroot,
        land_dlh=land_dlh,
        land_mcs=land_mcs,
        water_balance=water_balance,
        state_writeback=state_writeback,
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_main lines 1159-1650",
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_soil lines 5405-6815",
        ),
    )


def hydrol_main_scientific_output_packet(
    result: HydrolMainResult,
    *,
    precip_rain,
    precip_snow,
    qsintmax,
    dt_sechiba=1800.0,
    one_day=86400.0,
    ok_explicitsnow=True,
    ok_freeze_cwrr=True,
    peat_hydro=True,
    tides=True,
    topmodel_new=False,
    do_floodplains=False,
    check_waterbal=False,
    almaoutput=False,
    hist2_id=-1,
    extra_fields: Mapping[str, object] | None = None,
) -> HydrolMainOutputPacket:
    """Build the scientific arrays selected by ``hydrol_main`` output arms.

    Fortran provenance: ``hydrol.f90::hydrol_main`` lines 1396-1482 and
    1485-1655. XIOS/history serializers are external boundaries; the returned
    packet owns scientific values, scaling, and fixed branch selection only.
    """

    extra = {} if extra_fields is None else dict(extra_fields)
    precip_rain = _as_1d("precip_rain", precip_rain)
    precip_snow = _as_1d("precip_snow", precip_snow)
    qsintmax = _as_2d("qsintmax", qsintmax)
    dt = float(dt_sechiba)
    if dt <= 0.0:
        raise ValueError("dt_sechiba must be positive")

    fields: list[HydrolScientificField] = []

    def add(name, value):
        fields.append(HydrolScientificField(name, jnp.asarray(value)))

    add("water2infilt", result.module.soil.water2infilt)
    add("mc", result.module.soil.mc)
    add("kfact_root", result.kfact_root)
    add("vegetmax_soil", result.module.vegupd.vegetmax_soil)
    add("mcl", result.module.soil.mcl)
    add("evapnu_soil", result.diagnostics.ae_ns / dt)
    add("drainage_soil", result.module.soil.dr_ns / dt)
    add("transpir_soil", result.module.split.tr_ns / dt)
    add("runoff_soil", result.module.soil.ru_ns / dt)
    add("humrel", result.diagnostics.humrel)
    add("irrig_fin", result.routing.irrig_fin * float(one_day) / dt)
    add("drainage", result.drainage / dt)
    add("runoff", result.runoff / dt)
    add("precisol", result.module.canop.precisol / dt)
    add("precip_rain", precip_rain / dt)
    add("precip_snow", precip_snow / dt)
    add("qsintmax", qsintmax)
    add("qsintveg", result.module.canop.qsintveg)
    add("qsintveg_tot", jnp.sum(result.module.canop.qsintveg, axis=1))
    add("prveg", (precip_rain - jnp.sum(result.module.canop.precisol, axis=1)) / dt)
    if bool(do_floodplains):
        add("floodout", result.module.flood.floodout / dt)
    if bool(check_waterbal):
        if result.water_balance is None:
            raise ValueError("check_waterbal output requires a HydrolWaterBalanceResult")
        add("tot_flux", result.water_balance.tot_flux / dt)
    add("snowmelt", result.state_writeback["snowmelt"] / dt)
    add("tot_melt", result.state_writeback["tot_melt"] / dt)
    add("soilmoist", result.diagnostics.soilmoist)
    add("tmc", result.module.soil.tmc)
    add("humtot", result.diagnostics.humtot)
    add("humtot_top", result.humtot_top)
    snow_name = "snowdz" if bool(ok_explicitsnow) else "snowdepth"
    if snow_name not in result.state_writeback:
        raise ValueError(f"{snow_name} is required by the selected snow output arm")
    add("snowdz", result.state_writeback[snow_name])
    add("frac_bare", result.module.vegupd.frac_bare)
    add("soilwet", result.soilwet)
    add("delsoilmoist", result.delsoilmoist)
    add("delswe", result.delswe)
    add("delintercept", result.delintercept)
    if bool(ok_freeze_cwrr):
        if result.diagnostics.profil_froz_hydro is None:
            raise ValueError("ok_freeze_cwrr requires profil_froz_hydro")
        for name, value in (
            ("profil_froz_hydro", result.diagnostics.profil_froz_hydro),
            ("temp_hydro", extra.get("temp_hydro")),
            ("kk_moy", extra.get("kk_moy")),
            ("profil_froz_hydro_ns", result.module.soil.profil_froz_hydro_ns),
        ):
            if value is None:
                raise ValueError(f"ok_freeze_cwrr output requires {name}")
            add(name, value)
    if bool(peat_hydro):
        add("run2peat", result.module.soil.run2peat / dt)
        add("wt_ab", result.module.soil.wt_ab)
        add("wtp", result.diagnostics.wtp)
    if bool(topmodel_new):
        if result.diagnostics.fwet_new is None:
            raise ValueError("topmodel_new output requires fwet_new")
        add("fwet_new", result.diagnostics.fwet_new)
    if bool(tides):
        add("wt_ab_tide", result.module.soil.wt_ab_tide)
        add("run2man", result.module.soil.run2man / dt)
    for name, value in (
        ("fwet", result.fwet),
        ("fsat", result.fsat),
        ("fwt1", result.fwt1),
        ("fwt2", result.fwt2),
        ("fwt3", result.fwt3),
        ("fwt4", result.fwt4),
    ):
        add(name, value)

    return HydrolMainOutputPacket(
        xios_fields=tuple(fields),
        primary_history_enabled=not bool(almaoutput),
        secondary_history_enabled=int(hist2_id) > 0,
        history_serializer_boundary=(
            "The IOIPSL histwrite_p/XIOS serializer is external; packet values and branch selection are source-owned."
        ),
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_main lines 1396-1482",
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_main lines 1485-1655",
        ),
    )


_hydrol_module_explicit_step_jit = jit(
    hydrol_module_explicit_step,
    static_argnames=HYDROL_MODULE_JIT_STATIC_ARGNAMES,
)

_hydrol_module_diagnostics_jit = jit(
    hydrol_module_diagnostics,
    static_argnames=HYDROL_DIAGNOSTICS_JIT_STATIC_ARGNAMES,
)


def _hydrol_module_closure_core(
    *,
    nroot,
    dh_mm,
    peat_hydro_water_stress=None,
    dyn_nroot_larix=False,
    is_tree=None,
    znt=None,
    **module_inputs,
) -> tuple[
    HydrolModuleExplicitResult,
    HydrolModuleDiagnostics,
    tuple[jnp.ndarray, ...],
    tuple[jnp.ndarray, ...],
]:
    module = hydrol_module_explicit_step(**module_inputs)
    diagnostics = hydrol_module_diagnostics(
        module,
        nroot=nroot,
        dz_mm=module_inputs["dz_mm"],
        dh_mm=dh_mm,
        njsc=module_inputs["njsc"],
        veget_max=module_inputs["veget_max"],
        soiltile=module_inputs["soiltile"],
        vegtot=module_inputs["vegtot"],
        mcs=module_inputs["mcs"],
        mcf=module_inputs.get("mcf"),
        mcw=module_inputs.get("mcw"),
        mineral_tables=module_inputs["mineral_tables"],
        peat_tables=module_inputs.get("peat_tables"),
        ks=module_inputs["ks"],
        vevapnu=module_inputs["vevapnu"],
        tot_melt=module_inputs["tot_melt"],
        ok_freeze_cwrr=module_inputs["ok_freeze_cwrr"],
        peat_hydro=module_inputs["peat_hydro"],
        peat_hydro_water_stress=peat_hydro_water_stress,
        agri_peat=module_inputs["agri_peat"],
        tides=module_inputs["tides"],
        dyn_nroot_larix=dyn_nroot_larix,
        is_tree=is_tree,
        znt=znt,
        temp_hydro=module_inputs.get("temp_hydro"),
        evapot=module_inputs["evapot"],
        evap_bare_limit_alt=module.soil.evap_bare_limit_alt,
        tmcint=module.soil.tmcint_for_evap_bare_limit,
    )
    output_arrays, moisture_arrays = _hydrol_module_outputs_and_moisture_arrays(
        module,
        diagnostics,
        module_inputs["soiltile"],
        module_inputs["pref_soil_veg"],
    )
    return module, diagnostics, output_arrays, moisture_arrays


HYDROL_MODULE_CLOSURE_JIT_STATIC_ARGNAMES = tuple(
    dict.fromkeys(
        (
            *HYDROL_MODULE_JIT_STATIC_ARGNAMES,
            "peat_hydro_water_stress",
            "dyn_nroot_larix",
            "is_tree",
            "new_watstress",
            "topmodel_new",
            "do_rsoil",
            "liqlayers",
            "numlayers",
        )
    )
)


_hydrol_module_closure_core_jit = jit(
    _hydrol_module_closure_core,
    static_argnames=HYDROL_MODULE_CLOSURE_JIT_STATIC_ARGNAMES,
)


def hydrol_mc_to_mcl(mc, profil_froz, mcr):
    """Reconstruct liquid moisture `mcl` from total moisture `mc`.

    Fortran provenance: `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`,
    subroutine `hydrol_soil`, mineral non-peat formulas at lines 5899-5908
    and 6316-6328. Peat tile constants are intentionally not embedded in this
    generic kernel; callers must pass the applicable residual moisture `mcr`.
    """

    mc = jnp.asarray(mc)
    profil_froz = jnp.asarray(profil_froz)
    if mc.ndim == 1:
        mc = mc[None, :]
    if profil_froz.ndim == 1:
        profil_froz = profil_froz[None, :]
    if mc.shape != profil_froz.shape:
        raise ValueError("mc and profil_froz must have matching shape (npts, nslm) or (nslm,)")

    mcr = jnp.asarray(mcr)
    if mcr.ndim == 0:
        mcr = jnp.broadcast_to(mcr, (mc.shape[0],))
    mcr = mcr[:, None]
    return jnp.minimum(mc, mcr + (1.0 - profil_froz) * (mc - mcr))


def hydrol_mcl_to_mc_after_solve(mcl_after, mc_before, profil_froz, mcr):
    """Update total moisture `mc` after the liquid tridiagonal solve.

    Fortran provenance: `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`,
    subroutine `hydrol_soil`, mineral non-peat formulas at lines 6063-6075,
    7044-7058, and 7078-7092. The pre-solve `mc_before` is an explicit input
    because Fortran preserves the frozen fraction through that state.
    """

    mcl_after = jnp.asarray(mcl_after)
    mc_before = jnp.asarray(mc_before)
    profil_froz = jnp.asarray(profil_froz)
    if mcl_after.ndim == 1:
        mcl_after = mcl_after[None, :]
    if mc_before.ndim == 1:
        mc_before = mc_before[None, :]
    if profil_froz.ndim == 1:
        profil_froz = profil_froz[None, :]
    if not (mcl_after.shape == mc_before.shape == profil_froz.shape):
        raise ValueError("mcl_after, mc_before, and profil_froz must have matching shape")

    mcr = jnp.asarray(mcr)
    if mcr.ndim == 0:
        mcr = jnp.broadcast_to(mcr, (mcl_after.shape[0],))
    mcr = mcr[:, None]
    return jnp.maximum(mcl_after, mcl_after + profil_froz * (mc_before - mcr))


def hydrol_liquid_redistribution_check(tmci, tmcf, flux_top, dr_ns, rootsink):
    """Compute HYDROL's redistribution conservation residual `check_tr_ns`.

    Fortran provenance: `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`,
    subroutine `hydrol_soil`, lines 6027-6033 and optional recheck lines
    6050-6052.
    """

    tmci = jnp.asarray(tmci)
    tmcf = jnp.asarray(tmcf)
    flux_top = jnp.asarray(flux_top)
    dr_ns = jnp.asarray(dr_ns)
    rootsink = jnp.asarray(rootsink)
    if rootsink.ndim == 2:
        rootsink = jnp.sum(rootsink, axis=1)
    return tmcf - (tmci - flux_top - dr_ns - rootsink)


def hydrol_soil_flux_diagnostics(*, mcl_after, mcl_before, dr_ns, rootsink, dz_mm, flux_top):
    """Diagnose HYDROL's downward ``qflux`` between soil layers.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil_flux`` lines 8030-8126. The source initializes
    ``qflux(:,nslm,ins)`` from ``dr_ns(:,ins)`` and walks upward from the
    layer water budget using the saved pre-solve ``mclint`` profile.
    """

    mcl_after = jnp.asarray(mcl_after)
    mcl_before = jnp.asarray(mcl_before)
    if mcl_after.ndim == 1:
        mcl_after = mcl_after[None, :]
    if mcl_before.ndim == 1:
        mcl_before = mcl_before[None, :]
    if mcl_after.ndim != 2 or mcl_before.shape != mcl_after.shape:
        raise ValueError("mcl_after and mcl_before must have matching shape (npts, nslm)")

    npts, nslm = mcl_after.shape
    if nslm < 2:
        raise ValueError("hydrol_soil_flux_diagnostics requires at least two soil layers")
    rootsink = jnp.asarray(rootsink)
    if rootsink.ndim == 1:
        rootsink = rootsink[None, :]
    if rootsink.shape != mcl_after.shape:
        raise ValueError("rootsink must match mcl_after shape")
    dr_ns = _as_1d("dr_ns", dr_ns)
    flux_top = _as_1d("flux_top", flux_top)
    _require_same_npts(mcl_after, rootsink, dr_ns, flux_top)
    dz_mm = _as_1d("dz_mm", dz_mm)
    if dz_mm.shape[0] != nslm:
        raise ValueError("dz_mm length must match nslm")

    delta = mcl_after - mcl_before
    qflux = jnp.zeros((npts, nslm), dtype=mcl_after.dtype)
    last = nslm - 1
    qflux = qflux.at[:, last].set(dr_ns)
    jsl = nslm - 2
    qflux = qflux.at[:, jsl].set(
        qflux[:, jsl + 1]
        + (delta[:, jsl] + 3.0 * delta[:, jsl + 1]) * dz_mm[jsl + 1] / 8.0
        + rootsink[:, jsl + 1]
    )
    for jsl in range(nslm - 3, -1, -1):
        qflux = qflux.at[:, jsl].set(
            qflux[:, jsl + 1]
            + (delta[:, jsl] + 3.0 * delta[:, jsl + 1]) * dz_mm[jsl + 1] / 8.0
            + rootsink[:, jsl + 1]
            + (3.0 * delta[:, jsl + 1] + delta[:, jsl + 2]) * dz_mm[jsl + 2] / 8.0
        )

    surface_balance = (
        qflux[:, 0]
        + (3.0 * delta[:, 0] + delta[:, 1]) * dz_mm[1] / 8.0
        + rootsink[:, 0]
        + flux_top
    )
    return HydrolSoilFluxDiagnostics(qflux=qflux, surface_balance=surface_balance)


def hydrol_alt_residual_trigger(mcl_top_after_first_solve, flux_top, mcr, min_sechiba):
    """Evaluate the alternate residual-boundary trigger.

    Fortran provenance: `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`,
    subroutine `hydrol_soil`, non-peat trigger at lines 7011-7018. Peat tile
    branching is intentionally not embedded; callers pass the residual
    moisture threshold used by the active trace record.
    """

    return (jnp.asarray(mcl_top_after_first_solve) < jnp.asarray(mcr)) & (
        jnp.asarray(flux_top) > jnp.asarray(min_sechiba)
    )


def hydrol_alt_residual_solve_step(
    setup: HydrolSoilSetupResult,
    *,
    mcl_before,
    mc_before,
    profil_froz,
    mcr,
    b,
    dz_mm,
    dt_days,
    free_drain_coef,
    flux_top,
    resolv,
    mask_soiltile,
    k_bottom,
    min_sechiba=1.0e-8,
):
    """Run HYDROL's alternate residual top-boundary solve slice.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil``, lines 6971-7044 for the no-rootsink solve and
    residual-boundary retry, lines 7044-7108 for ``mc`` reconstruction,
    bottom flux, and ``tmc``. Coefficient recomputation for
    ``ok_freeze_cwrr`` remains outside this minimal adapter; callers pass the
    exact bottom conductivity to use.
    """

    saved = hydrol_soil_rhs_main(
        setup=setup,
        mcl=mcl_before,
        b=b,
        dt_days=dt_days,
        free_drain_coef=free_drain_coef,
        flux_top=flux_top,
        rootsink=0.0,
    )
    first = hydrol_soil_tridiag_solve(
        e=saved.e,
        f=saved.f,
        g1=saved.g1,
        rhs=saved.rhs,
        resolv=resolv,
        initial_mcl=mcl_before,
    )
    resolv_alt = hydrol_alt_residual_trigger(
        mcl_top_after_first_solve=first.mcl[:, 0],
        flux_top=flux_top,
        mcr=mcr,
        min_sechiba=min_sechiba,
    )
    residual_equations = hydrol_soil_residual_boundary_rhs(saved, mcr=mcr)
    second = hydrol_soil_tridiag_solve(
        e=residual_equations.e,
        f=residual_equations.f,
        g1=residual_equations.g1,
        rhs=residual_equations.rhs,
        resolv=resolv_alt,
        initial_mcl=first.mcl,
    )
    mc_after = hydrol_mcl_to_mc_after_solve(
        mcl_after=second.mcl,
        mc_before=mc_before,
        profil_froz=profil_froz,
        mcr=mcr,
    )
    flux_bottom = bottom_drainage(
        k_bottom=k_bottom,
        free_drain_coef=free_drain_coef,
        dt_days=dt_days,
        mask_soiltile=mask_soiltile,
        resolv=jnp.ones_like(resolv_alt, dtype=bool),
    )
    tmc = hydrol_total_moisture_content(mc_after, dz_mm)
    return HydrolAltResidualSolveResult(
        saved=saved,
        first=first,
        residual_equations=residual_equations,
        second=second,
        resolv_alt=resolv_alt,
        mc_after=mc_after,
        flux_bottom=flux_bottom,
        tmc=tmc,
    )


def hydrol_soil_resistance_evaporation(
    *, evapot_penm, u, v, tq_cdrag, tmc_litter, tmcs_litter,
    min_wind=0.1, min_sechiba=1.0e-8,
):
    """Apply the CWRR bare-soil resistance to potential evaporation.

    Fortran provenance: ``hydrol.f90::hydrol_soil`` lines 6918-6935.
    """
    evapot_penm = _as_1d("evapot_penm", evapot_penm)
    u = _as_1d("u", u)
    v = _as_1d("v", v)
    tq_cdrag = _as_1d("tq_cdrag", tq_cdrag)
    tmc_litter = _as_2d("tmc_litter", tmc_litter)
    tmcs_litter = _as_2d("tmcs_litter", tmcs_litter)
    _require_same_npts(evapot_penm, u, v, tq_cdrag, tmc_litter, tmcs_litter)
    if tmc_litter.shape != tmcs_litter.shape:
        raise ValueError("tmc_litter and tmcs_litter must share shape (npts,nstm)")
    mc_rel = tmc_litter / tmcs_litter
    r_soil_ns = jnp.exp(8.206 - 4.255 * mc_rel)
    speed = jnp.maximum(min_wind, jnp.sqrt(u * u + v * v))
    conductance = speed * tq_cdrag
    ra = 1.0 / jnp.where(conductance > min_sechiba, conductance, 1.0)
    return jnp.where(
        conductance[:, None] > min_sechiba,
        evapot_penm[:, None] / (1.0 + r_soil_ns / ra[:, None]),
        evapot_penm[:, None],
    )


def hydrol_alt_residual_all_tiles_from_results(
    tile_results: tuple[HydrolSoilTileExplicitStepResult, ...],
    *,
    profil_froz,
    kfact_root=None,
    mineral_tables: MineralCWRRTables | None = None,
    peat_tables: PeatCWRRTables | None = None,
    mcr,
    mcs=None,
    dz_mm,
    dt_days,
    free_drain_coef,
    resolv,
    mask_soiltile,
    evapot,
    evapot_penm,
    frac_bare_ns,
    peat_hydro=False,
    peat_tiles=(4, 5, 6),
    mcr_peat=PEAT_MCR,
    ok_freeze_cwrr=False,
    min_sechiba=1.0e-8,
) -> HydrolAltResidualSolveResult:
    """Run the HYDROL bare-evaporation dummy solve for every soil tile.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil``, lines 6863-7174. The normal tile solve has
    already produced post-integration moisture. This helper replays the
    source-order dummy branch: lines 6889-6900 reconstruct ``mcl`` from final
    ``mc`` and call ``hydrol_soil_coef`` again before lines 6902-6936 build
    setup coefficients and the potential-evaporation top boundary. It then
    applies the residual top-boundary retry for the evaporation-limit
    diagnostic only. It does not mutate the prognostic ``mc/mcl/tmc`` state
    restored at source lines 7165-7174.
    """

    if not tile_results:
        raise ValueError("tile_results must not be empty")
    profil_froz = _as_3d("profil_froz", profil_froz)
    kfact_root = None if kfact_root is None else jnp.asarray(kfact_root)
    free_drain_coef = _as_2d("free_drain_coef", free_drain_coef)
    resolv = _as_2d("resolv", resolv)
    mask_soiltile = _as_2d("mask_soiltile", mask_soiltile)
    evapot = _as_1d("evapot", evapot)
    evapot_penm = jnp.asarray(evapot_penm)
    if evapot_penm.ndim == 1:
        evapot_penm = jnp.broadcast_to(evapot_penm[:, None], mask_soiltile.shape)
    elif evapot_penm.ndim != 2:
        raise ValueError("evapot_penm must have shape (npts,) or (npts,nstm)")
    frac_bare_ns = _as_2d("frac_bare_ns", frac_bare_ns)
    npts, nslm, nstm = profil_froz.shape
    if len(tile_results) != nstm:
        raise ValueError("tile_results length must match profil_froz tile axis")
    if kfact_root is not None and kfact_root.shape != profil_froz.shape:
        raise ValueError("kfact_root must match profil_froz shape (npts,nslm,nstm)")
    for name, value in (
        ("free_drain_coef", free_drain_coef),
        ("resolv", resolv),
        ("mask_soiltile", mask_soiltile),
        ("frac_bare_ns", frac_bare_ns),
    ):
        if value.shape != (npts, nstm):
            raise ValueError(f"{name} must have shape (npts,nstm)")
    if evapot.shape != (npts,):
        raise ValueError("evapot must have shape (npts,)")
    if evapot_penm.shape != (npts, nstm):
        raise ValueError("evapot_penm must have shape (npts,) or (npts,nstm)")
    mcr = _as_1d("mcr", mcr)
    if mcr.shape[0] == 1 and npts != 1:
        mcr = jnp.repeat(mcr, npts)
    if mcr.shape != (npts,):
        raise ValueError("mcr must be scalar or have shape (npts,)")
    if mineral_tables is not None and mcs is None:
        raise ValueError("mcs is required when recomputing mineral dummy coefficients")
    if mineral_tables is not None and peat_tables is not None:
        table_mode = "mixed"
    elif mineral_tables is not None:
        table_mode = "mineral"
    elif peat_tables is not None:
        table_mode = "peat"
    else:
        table_mode = "reuse"

    saved_rhs = []
    saved_e = []
    saved_f = []
    saved_g1 = []
    first_mcl = []
    first_bet = []
    first_gam = []
    residual_rhs = []
    residual_e = []
    residual_f = []
    residual_g1 = []
    second_mcl = []
    second_bet = []
    second_gam = []
    resolv_alt = []
    mc_after = []
    flux_bottom = []
    tmc = []
    for tile_offset, tile in enumerate(tile_results):
        tile_mcr = (
            jnp.full((npts,), mcr_peat, dtype=mcr.dtype)
            if bool(peat_hydro) and (tile_offset + 1) in tuple(int(tile_index) for tile_index in peat_tiles)
            else mcr
        )
        tile_profil = profil_froz[:, :, tile_offset]
        tile_mc = tile.solve.mc_final
        tile_mcl = hydrol_mc_to_mcl(tile_mc, tile_profil, tile_mcr)
        tile_setup = tile.setup
        tile_b = jnp.zeros((npts, nslm), dtype=profil_froz.dtype)
        tile_k_bottom = tile.k_bottom
        if table_mode != "reuse":
            if kfact_root is None:
                raise ValueError("kfact_root is required when recomputing dummy coefficients")
            peat_active = bool(peat_hydro) and (tile_offset + 1) in tuple(int(tile_index) for tile_index in peat_tiles)
            if peat_active:
                if peat_tables is None:
                    raise ValueError("peat_tables are required for active peat dummy coefficients")
                dummy_coef = hydrol_soil_coef_peat_profile_from_tables(
                    mc=tile_mc,
                    profil_froz=tile_profil,
                    kfact_root=kfact_root[:, :, tile_offset],
                    tables=peat_tables,
                    ok_freeze_cwrr=ok_freeze_cwrr,
                )
                tile_b = dummy_coef.b
            else:
                if mineral_tables is None:
                    raise ValueError("mineral_tables are required for non-peat dummy coefficients")
                dummy_coef = hydrol_soil_coef_mineral_profile_from_tables(
                    mc=tile_mc,
                    profil_froz=tile_profil,
                    kfact_root=kfact_root[:, :, tile_offset],
                    tables=mineral_tables,
                    mcr=mcr,
                    mcs=mcs,
                    ok_freeze_cwrr=ok_freeze_cwrr,
                )
                tile_b = dummy_coef.b
            tile_setup = hydrol_soil_setup_coefficients(
                a=dummy_coef.a,
                d=dummy_coef.d,
                dz_mm=dz_mm,
                dt_days=dt_days,
                free_drain_coef=free_drain_coef[:, tile_offset],
            )
            tile_b = dummy_coef.b
            tile_k_bottom = dummy_coef.k[:, -1]
        bare_evap_mask = jnp.floor(
            frac_bare_ns[:, tile_offset] + jnp.asarray(1.0 - min_sechiba, dtype=frac_bare_ns.dtype)
        )
        evap_flux_top = evapot_penm[:, tile_offset] * bare_evap_mask
        alt = hydrol_alt_residual_solve_step(
            setup=tile_setup,
            mcl_before=tile_mcl,
            mc_before=tile_mc,
            profil_froz=tile_profil,
            mcr=tile_mcr,
            b=tile_b,
            dz_mm=dz_mm,
            dt_days=dt_days,
            free_drain_coef=free_drain_coef[:, tile_offset],
            flux_top=evap_flux_top,
            resolv=resolv[:, tile_offset],
            mask_soiltile=mask_soiltile[:, tile_offset],
            k_bottom=tile_k_bottom,
            min_sechiba=min_sechiba,
        )
        saved_rhs.append(alt.saved.rhs)
        saved_e.append(alt.saved.e)
        saved_f.append(alt.saved.f)
        saved_g1.append(alt.saved.g1)
        first_mcl.append(alt.first.mcl)
        first_bet.append(alt.first.bet)
        first_gam.append(alt.first.gam)
        residual_rhs.append(alt.residual_equations.rhs)
        residual_e.append(alt.residual_equations.e)
        residual_f.append(alt.residual_equations.f)
        residual_g1.append(alt.residual_equations.g1)
        second_mcl.append(alt.second.mcl)
        second_bet.append(alt.second.bet)
        second_gam.append(alt.second.gam)
        resolv_alt.append(alt.resolv_alt)
        mc_after.append(alt.mc_after)
        flux_bottom.append(alt.flux_bottom)
        tmc.append(alt.tmc)

    stack_3d = lambda values: jnp.stack(values, axis=2)
    stack_2d = lambda values: jnp.stack(values, axis=1)
    return HydrolAltResidualSolveResult(
        saved=HydrolSoilRHSResult(
            rhs=stack_3d(saved_rhs),
            e=stack_3d(saved_e),
            f=stack_3d(saved_f),
            g1=stack_3d(saved_g1),
        ),
        first=HydrolSoilTridiagResult(
            mcl=stack_3d(first_mcl),
            bet=stack_3d(first_bet),
            gam=stack_3d(first_gam),
        ),
        residual_equations=HydrolSoilRHSResult(
            rhs=stack_3d(residual_rhs),
            e=stack_3d(residual_e),
            f=stack_3d(residual_f),
            g1=stack_3d(residual_g1),
        ),
        second=HydrolSoilTridiagResult(
            mcl=stack_3d(second_mcl),
            bet=stack_3d(second_bet),
            gam=stack_3d(second_gam),
        ),
        resolv_alt=stack_2d(resolv_alt),
        mc_after=stack_3d(mc_after),
        flux_bottom=stack_2d(flux_bottom),
        tmc=stack_2d(tmc),
    )


def hydrol_evap_bare_limit_diagnostic(
    *,
    tmcint,
    tmc_dummy,
    flux_bottom,
    mask_soiltile,
    soiltile=None,
    frac_bare_ns,
    vegtot,
    evapot,
    tmc_litter,
    tmc_litter_wilt,
    tmc_litter_res,
    is_under_mcr,
    do_rsoil=False,
    min_sechiba=1.0e-8,
):
    """Compute ``evap_bare_lim_ns`` from the dummy alternate residual solve.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil``, lines 7110-7163. The caller supplies the
    saved pre-dummy column water ``tmcint``, dummy-solve ``tmc`` and
    ``flux_bottom``, litter thresholds, vegetation masks, and final
    ``is_under_mcr`` state. Lines 7165-7174 restore prognostic ``mc/mcl/tmc``;
    this helper returns diagnostics only and never mutates prognostic state.
    """

    tmc_dummy = _as_2d("tmc_dummy", tmc_dummy)
    tmcint = jnp.asarray(tmcint)
    if tmcint.ndim == 0:
        tmcint = jnp.broadcast_to(tmcint, (tmc_dummy.shape[0],))
    if tmcint.ndim == 1:
        tmcint = jnp.broadcast_to(tmcint[:, None], tmc_dummy.shape)
    elif tmcint.ndim != 2:
        raise ValueError("tmcint must be scalar, one-dimensional, or have shape (npts,nstm)")
    if tmcint.shape != tmc_dummy.shape:
        raise ValueError("tmcint must have shape (npts,) or match tmc_dummy shape (npts,nstm)")
    flux_bottom = _as_2d("flux_bottom", flux_bottom)
    mask_soiltile = _as_2d("mask_soiltile", mask_soiltile)
    if soiltile is None:
        soiltile = mask_soiltile
    soiltile = _as_2d("soiltile", soiltile)
    frac_bare_ns = _as_2d("frac_bare_ns", frac_bare_ns)
    vegtot = _as_1d("vegtot", vegtot)
    evapot = _as_1d("evapot", evapot)
    tmc_litter = _as_2d("tmc_litter", tmc_litter)
    tmc_litter_wilt = _as_2d("tmc_litter_wilt", tmc_litter_wilt)
    tmc_litter_res = _as_2d("tmc_litter_res", tmc_litter_res)
    is_under_mcr = jnp.asarray(is_under_mcr, dtype=bool)
    _require_same_npts(
        tmc_dummy,
        tmcint,
        flux_bottom,
        mask_soiltile,
        soiltile,
        frac_bare_ns,
        vegtot,
        evapot,
        tmc_litter,
        tmc_litter_wilt,
        tmc_litter_res,
        is_under_mcr,
    )
    tile_shape = tmc_dummy.shape
    if not (
        flux_bottom.shape
        == mask_soiltile.shape
        == soiltile.shape
        == frac_bare_ns.shape
        == tmc_litter.shape
        == tmc_litter_wilt.shape
        == tmc_litter_res.shape
        == is_under_mcr.shape
        == tile_shape
    ):
        raise ValueError("tile diagnostics must share shape (npts, nstm)")

    minv = jnp.asarray(min_sechiba, dtype=tmc_dummy.dtype)
    water_budget = mask_soiltile * (tmcint - tmc_dummy - flux_bottom)
    bare_weighted = jnp.where(vegtot[:, None] > minv, water_budget * frac_bare_ns, 0.0)

    if bool(do_rsoil):
        beta = jnp.where(evapot[:, None] > minv, bare_weighted / jnp.where(evapot[:, None] > minv, evapot[:, None], 1.0), 0.0)
    else:
        evap_active = evapot[:, None] > minv
        full_litter = tmc_litter > tmc_litter_wilt
        residual_litter = tmc_litter > tmc_litter_res
        beta = jnp.where(
            evap_active & full_litter,
            bare_weighted / jnp.where(evap_active, evapot[:, None], 1.0),
            jnp.where(
                evap_active & residual_litter,
                0.5 * bare_weighted / jnp.where(evap_active, evapot[:, None], 1.0),
                0.0,
            ),
        )
    beta = jnp.maximum(jnp.minimum(beta, 1.0), 0.0)
    beta = jnp.where(is_under_mcr, 0.0, beta)
    evap_bare_lim = hydrol_evap_bare_limit_grid_beta(
        evap_bare_lim_ns=beta,
        vegtot=vegtot,
        soiltile=soiltile,
    )
    beta = jnp.where(evap_bare_lim[:, None] < 1.0e-3, 0.0, beta)
    evap_bare_lim = hydrol_evap_bare_limit_grid_beta(
        evap_bare_lim_ns=beta,
        vegtot=vegtot,
        soiltile=soiltile,
    )
    return HydrolEvapBareLimitDiagnostic(
        water_budget_evap=water_budget,
        bare_weighted_evap=bare_weighted,
        evap_bare_lim=evap_bare_lim,
        evap_bare_lim_ns=beta,
    )


def hydrol_evap_bare_limit_grid_beta(*, evap_bare_lim_ns, vegtot, soiltile):
    """Aggregate soil-tile bare-evaporation beta to grid scale.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil``, lines 7349-7355. The source computes
    ``SUM(evap_bare_lim_ns * vegtot * soiltile)``. Callers that own the tile
    array apply the following source clear-and-recompute sequence when the
    first grid-cell beta is below ``1e-3``.
    """

    evap_bare_lim_ns = _as_2d("evap_bare_lim_ns", evap_bare_lim_ns)
    vegtot = _as_1d("vegtot", vegtot)
    soiltile = _as_2d("soiltile", soiltile)
    _require_same_npts(evap_bare_lim_ns, vegtot, soiltile)
    if evap_bare_lim_ns.shape != soiltile.shape:
        raise ValueError("evap_bare_lim_ns and soiltile must share shape (npts, nstm)")
    return jnp.sum(evap_bare_lim_ns * vegtot[:, None] * soiltile, axis=1)


def hydrol_soil_clip_over_mcs2(mc, mcs, dz_mm):
    """Apply the `hydrol_soil_smooth_over_mcs2` direct clipping rule.

    Fortran provenance: `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`,
    subroutine `hydrol_soil_smooth_over_mcs2`, lines 7935-8026. This generic
    mineral/explicit-threshold kernel implements the closed active algebra:
    excess above `mcs` is removed from `mc` and integrated into `rudr_corr`.
    Routing that correction to runoff or drainage remains outside this helper.
    """

    mc = jnp.asarray(mc)
    if mc.ndim == 1:
        mc = mc[None, :]
    mcs = jnp.asarray(mcs)
    if mcs.ndim == 0:
        mcs = jnp.broadcast_to(mcs, (mc.shape[0],))
    if mcs.ndim != 1 or mcs.shape[0] != mc.shape[0]:
        raise ValueError("mcs must be scalar or have shape (npts,)")

    excess = jnp.maximum(mc - mcs[:, None], 0.0)
    clipped = mc - excess
    return HydrolOverSaturationCorrectionResult(
        mc=clipped,
        excess=excess,
        rudr_corr=hydrol_total_moisture_content(excess, dz_mm),
        is_over_mcs=jnp.zeros((mc.shape[0],), dtype=bool),
    )


def hydrol_route_over_mcs2(
    *,
    mc,
    mcs,
    dz_mm,
    ru_ns,
    dr_ns,
    free_drain_coef,
    ok_freeze_cwrr,
    check_cwrr2=False,
):
    """Clip over-saturation and route its correction as HYDROL does.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil`` lines 6076-6092 plus
    ``hydrol_soil_smooth_over_mcs2`` lines 7935-8026. The source first
    computes the excess as ``ru_corr_ns`` and then, only when
    ``free_drain_coef >= 0.5`` and ``ok_freeze_cwrr`` is false, reroutes that
    correction to ``dr_corr_ns``.
    """

    clipped = hydrol_soil_clip_over_mcs2(mc=mc, mcs=mcs, dz_mm=dz_mm)
    ru_ns = _as_1d("ru_ns", ru_ns)
    dr_ns = _as_1d("dr_ns", dr_ns)
    free_drain_coef = _as_1d("free_drain_coef", free_drain_coef)
    _require_same_npts(clipped.rudr_corr, ru_ns, dr_ns, free_drain_coef)

    route_to_drainage = (free_drain_coef >= 0.5) & (not bool(ok_freeze_cwrr))
    ru_corr_ns = jnp.where(route_to_drainage, 0.0, clipped.rudr_corr)
    dr_corr_ns = jnp.where(route_to_drainage, clipped.rudr_corr, 0.0)

    check_over_ns = None
    if check_cwrr2:
        tmci = hydrol_total_moisture_content(mc, dz_mm)
        tmcf = hydrol_total_moisture_content(clipped.mc, dz_mm)
        check_over_ns = tmcf - (tmci - clipped.rudr_corr)

    return HydrolOverSaturationRoutingResult(
        mc=clipped.mc,
        excess=clipped.excess,
        ru_corr_ns=ru_corr_ns,
        dr_corr_ns=dr_corr_ns,
        ru_ns=ru_ns + ru_corr_ns,
        dr_ns=dr_ns + dr_corr_ns,
        check_over_ns=check_over_ns,
        is_over_mcs=clipped.is_over_mcs,
    )


def hydrol_correct_negative_runoff(ru_ns, dr_ns):
    """Move negative ``ru_ns`` into drainage as in HYDROL.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil``, lines 6102-6114.
    """

    ru_ns = _as_1d("ru_ns", ru_ns)
    dr_ns = _as_1d("dr_ns", dr_ns)
    _require_same_npts(ru_ns, dr_ns)

    negative = ru_ns < 0.0
    ru_corr2_ns = jnp.where(negative, -ru_ns, 0.0)
    return HydrolNegativeRunoffCorrectionResult(
        ru_ns=jnp.where(negative, 0.0, ru_ns),
        dr_ns=jnp.where(negative, dr_ns + ru_ns, dr_ns),
        ru_corr2_ns=ru_corr2_ns,
    )


def hydrol_force_water_table_saturation(
    *,
    mc,
    dr_ns,
    zwt_force,
    zz_mm,
    dz_mm,
    zmaxh_m,
    mcs,
    peat_hydro=False,
    soil_tile_index=1,
    peat_tiles=(4, 5, 6),
    mcs_peat=PEAT_MCS,
):
    """Force saturation below prescribed water-table depth.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil``, lines 6116-6158. The activation test follows
    the source exactly: only ``zwt_force(1,jst) <= zmaxh`` decides whether this
    block runs for the tile.
    """

    mc = jnp.asarray(mc)
    if mc.ndim == 1:
        mc = mc[None, :]
    if mc.ndim != 2:
        raise ValueError("mc must have shape (npts, nslm) or (nslm,)")
    dr_ns = _as_1d("dr_ns", dr_ns)
    zwt_force = _as_1d("zwt_force", zwt_force)
    zz_mm = _as_1d("zz_mm", zz_mm)
    dz_mm = _as_1d("dz_mm", dz_mm)
    _require_same_npts(mc, dr_ns, zwt_force)
    if zz_mm.shape[0] != mc.shape[1] or dz_mm.shape[0] != mc.shape[1]:
        raise ValueError("zz_mm and dz_mm length must match nslm")

    npts = mc.shape[0]
    mcs = _as_1d("mcs", mcs)
    if mcs.shape[0] == 1:
        mcs = jnp.repeat(mcs, npts)
    if mcs.shape[0] != npts:
        raise ValueError("mcs must be scalar or have shape (npts,)")

    peat_active = bool(peat_hydro) and int(soil_tile_index) in tuple(int(tile) for tile in peat_tiles)
    saturation = jnp.full((npts,), mcs_peat, dtype=mc.dtype) if peat_active else mcs
    active = zwt_force[0] <= jnp.asarray(zmaxh_m, dtype=mc.dtype)
    force_mask = (zz_mm[None, :] >= zwt_force[:, None] * 1000.0) & active
    dmc = jnp.where(force_mask, saturation[:, None] - mc, 0.0)
    mc_forced = jnp.where(force_mask, saturation[:, None], mc)
    dr_force_ns = hydrol_total_moisture_content(dmc, dz_mm)

    return HydrolForcedWaterTableResult(
        mc=mc_forced,
        dmc=dmc,
        dr_force_ns=dr_force_ns,
        dr_ns=dr_ns - dr_force_ns,
    )


def hydrol_effective_water_table_depth(
    *,
    mc,
    zz_mm,
    mcs,
    peat_hydro=False,
    soil_tile_index=1,
    peat_tiles=(4, 5, 6),
    mcs_peat=PEAT_MCS,
    undef_sechiba=1.0e20,
):
    """Diagnose HYDROL effective water-table depth.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil``, lines 6162-6180. The diagnostic walks upward
    from the bottom while nodes are exactly at saturation.
    """

    mc = jnp.asarray(mc)
    if mc.ndim == 1:
        mc = mc[None, :]
    if mc.ndim != 2:
        raise ValueError("mc must have shape (npts, nslm) or (nslm,)")
    zz_mm = _as_1d("zz_mm", zz_mm)
    if zz_mm.shape[0] != mc.shape[1]:
        raise ValueError("zz_mm length must match nslm")
    npts, nslm = mc.shape
    mcs = _as_1d("mcs", mcs)
    if mcs.shape[0] == 1:
        mcs = jnp.repeat(mcs, npts)
    if mcs.shape[0] != npts:
        raise ValueError("mcs must be scalar or have shape (npts,)")

    peat_active = bool(peat_hydro) and int(soil_tile_index) in tuple(int(tile) for tile in peat_tiles)
    saturation = jnp.full((npts,), mcs_peat, dtype=mc.dtype) if peat_active else mcs
    wtd = jnp.full((npts,), undef_sechiba, dtype=mc.dtype)
    still_saturated = jnp.ones((npts,), dtype=bool)
    for jsl in range(nslm - 1, 0, -1):
        saturated = still_saturated & (mc[:, jsl] == saturation)
        wtd = jnp.where(saturated, zz_mm[jsl] / 1000.0, wtd)
        still_saturated = saturated

    return HydrolWaterTableDepthResult(wtd_ns=wtd)


def hydrol_after_under_mcr_runoff_peat_tide_routing(
    *,
    soil_tile_index,
    ru_ns,
    runoff2peat,
    soiltile,
    water2infilt,
    run2peat,
    run2man,
    wt_ab,
    wt_ab_tide,
    peat_hydro,
    ok_ru2peat,
    ok_wt_ab,
    tides,
    wtp_tide,
    dwtp_tide,
    min_sechiba=1.0e-8,
    max_wt_ab=100.0,
):
    """Route mineral runoff to peat/mangrove reservoirs after under-MCR.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil``, lines 6183-6296. This function represents one
    pass through the source ``jst`` soil-tile loop and therefore preserves the
    source reset of ``runoff2peat(:,:)`` at line 6192.
    """

    ru_ns = _as_2d("ru_ns", ru_ns)
    runoff2peat = jnp.zeros_like(_as_2d("runoff2peat", runoff2peat))
    soiltile = _as_2d("soiltile", soiltile)
    water2infilt = _as_2d("water2infilt", water2infilt)
    run2peat = _as_1d("run2peat", run2peat)
    run2man = _as_1d("run2man", run2man)
    wt_ab = _as_1d("wt_ab", wt_ab)
    wt_ab_tide = _as_1d("wt_ab_tide", wt_ab_tide)
    _require_same_npts(ru_ns, runoff2peat, soiltile, water2infilt, run2peat, run2man, wt_ab, wt_ab_tide)
    if ru_ns.shape[1] < 6 or soiltile.shape[1] < 6 or water2infilt.shape[1] < 6:
        raise ValueError("runoff/tide routing requires at least six soil tiles")

    jst = int(soil_tile_index)
    j = jst - 1
    mineral_sum = ru_ns[:, 0] * soiltile[:, 0] + ru_ns[:, 1] * soiltile[:, 1] + ru_ns[:, 2] * soiltile[:, 2]

    if bool(tides):
        if bool(peat_hydro) and bool(ok_ru2peat):
            mask6 = soiltile[:, 5] > min_sechiba
            mask46 = mask6 & (soiltile[:, 3] > min_sechiba)
            runoff2peat = runoff2peat.at[:, 0].set(jnp.where(mask6, ru_ns[:, 0], runoff2peat[:, 0]))
            runoff2peat = runoff2peat.at[:, 1].set(jnp.where(mask6, ru_ns[:, 1], runoff2peat[:, 1]))
            runoff2peat = runoff2peat.at[:, 2].set(jnp.where(mask6, ru_ns[:, 2], runoff2peat[:, 2]))
            split_den = soiltile[:, 3] + soiltile[:, 5]
            run2man_candidate = jnp.where(
                mask46,
                soiltile[:, 5] / split_den * mineral_sum,
                soiltile[:, 5] * mineral_sum,
            )
            run2peat_candidate = jnp.where(mask46, soiltile[:, 3] / split_den * mineral_sum, 0.0)
            run2man = jnp.where(mask6, run2man_candidate, run2man)
            run2peat = jnp.where(mask6, run2peat_candidate, run2peat)
            ru_ns = ru_ns.at[:, 0].set(jnp.where(mask6, 0.0, ru_ns[:, 0]))
            ru_ns = ru_ns.at[:, 1].set(jnp.where(mask6, 0.0, ru_ns[:, 1]))
            ru_ns = ru_ns.at[:, 2].set(jnp.where(mask6, 0.0, ru_ns[:, 2]))
    else:
        if bool(peat_hydro) and bool(ok_ru2peat) and jst == 4:
            mask4 = soiltile[:, 3] > min_sechiba
            run2peat = jnp.where(mask4, mineral_sum, run2peat)
            ru_ns = ru_ns.at[:, 0].set(jnp.where(mask4, 0.0, ru_ns[:, 0]))
            ru_ns = ru_ns.at[:, 1].set(jnp.where(mask4, 0.0, ru_ns[:, 1]))
            ru_ns = ru_ns.at[:, 2].set(jnp.where(mask4, 0.0, ru_ns[:, 2]))

    if bool(peat_hydro) and bool(ok_wt_ab) and jst == 4:
        wt_ab = ru_ns[:, j]
        ru_ns = ru_ns.at[:, j].set(jnp.maximum(wt_ab - max_wt_ab, 0.0))
        wt_ab = jnp.minimum(max_wt_ab, wt_ab)

    if bool(tides) and jst == 4:
        wtp_tide = jnp.asarray(wtp_tide, dtype=ru_ns.dtype)
        dwtp_tide = jnp.asarray(dwtp_tide, dtype=ru_ns.dtype)
        positive_increasing = (wtp_tide > 0.0) & (dwtp_tide > 0.0)
        nonpositive = (wtp_tide <= 0.0) & (dwtp_tide <= 0.0)
        positive_decreasing = (wtp_tide > 0.0) & (dwtp_tide < 0.0)
        previous_ru = ru_ns[:, j]
        previous_wt = wt_ab_tide
        increasing_wt = wtp_tide + previous_ru

        # Preserve the source ELSE IF priority. In particular, the subsequent
        # wtp<=0/dwtp<0 arm is shadowed by ``nonpositive`` in hydrol.f90.
        next_ru = jnp.where(
            positive_increasing,
            0.0,
            jnp.where(
                nonpositive,
                previous_ru,
                jnp.where(positive_decreasing, previous_ru - dwtp_tide, previous_ru),
            ),
        )
        next_wt = jnp.where(
            positive_increasing,
            increasing_wt,
            jnp.where(nonpositive, 0.0, jnp.where(positive_decreasing, wtp_tide, previous_wt)),
        )
        infiltrating = positive_increasing | positive_decreasing
        ru_ns = ru_ns.at[:, j].set(next_ru)
        wt_ab_tide = next_wt
        water2infilt = water2infilt.at[:, j].add(jnp.where(infiltrating, next_wt, 0.0))

    return HydrolRunoffPeatTideRoutingResult(
        ru_ns=ru_ns,
        runoff2peat=runoff2peat,
        water2infilt=water2infilt,
        run2peat=run2peat,
        run2man=run2man,
        wt_ab=wt_ab,
        wt_ab_tide=wt_ab_tide,
    )


def hydrol_runoff_peat_post_loop_reinjection(
    *,
    ru_ns,
    water2infilt,
    tmc,
    soiltile,
    run2peat,
    run2man,
    wt_ab,
    peat_hydro,
    agri_peat,
    ok_ru2peat,
    tides,
    min_sechiba=1.0e-8,
):
    """Apply post-soil-tile runoff-to-peat reinjection and update ``tmc``.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil``, lines 6757-6787.
    """

    ru_ns = _as_2d("ru_ns", ru_ns)
    water2infilt = _as_2d("water2infilt", water2infilt)
    tmc = _as_2d("tmc", tmc)
    soiltile = _as_2d("soiltile", soiltile)
    run2peat = _as_1d("run2peat", run2peat)
    run2man = _as_1d("run2man", run2man)
    wt_ab = _as_1d("wt_ab", wt_ab)
    _require_same_npts(ru_ns, water2infilt, tmc, soiltile, run2peat, run2man, wt_ab)
    if ru_ns.shape[1] < 6 or water2infilt.shape[1] < 6 or tmc.shape[1] < 6 or soiltile.shape[1] < 6:
        raise ValueError("post-loop runoff reinjection requires at least six soil tiles")

    if bool(peat_hydro):
        if bool(agri_peat) and bool(ok_ru2peat):
            run2peat = run2peat + ru_ns[:, 4] * soiltile[:, 4]
            ru_ns = ru_ns.at[:, 4].set(0.0)
        mask4 = bool(ok_ru2peat) & (soiltile[:, 3] > min_sechiba)
        water2infilt = water2infilt.at[:, 3].add(jnp.where(mask4, run2peat / soiltile[:, 3], 0.0))
        water2infilt = water2infilt.at[:, 3].add(wt_ab)
        if bool(tides):
            mask6 = bool(ok_ru2peat) & (soiltile[:, 5] > min_sechiba)
            water2infilt = water2infilt.at[:, 5].add(jnp.where(mask6, run2man / soiltile[:, 5], 0.0))
            water2infilt = water2infilt.at[:, 5].add(wt_ab)

    tmc_soil = tmc
    tmc = tmc + water2infilt
    return HydrolRunoffPeatPostLoopResult(
        ru_ns=ru_ns,
        water2infilt=water2infilt,
        tmc=tmc,
        tmc_soil=tmc_soil,
        run2peat=run2peat,
    )


def hydrol_soil_smooth_under_mcr(
    *,
    mc,
    mcr,
    dz_mm,
    zmaxh_m,
    mask_soiltile,
    peat_hydro=False,
    soil_tile_index=1,
    peat_tiles=(4, 5, 6),
    mcr_peat=PEAT_MCR,
    min_sechiba=1.0e-8,
    check_cwrr2=False,
):
    """Redistribute under-residual deficits and diagnose ``is_under_mcr``.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/hydrol.f90``,
    subroutine ``hydrol_soil_smooth_under_mcr``, lines 7623-7770. The branch
    conditions intentionally preserve the source asymmetry where bottom/top
    diagnostics treat peat tiles 4/5 differently from earlier loops.
    """

    mc = jnp.asarray(mc)
    if mc.ndim == 1:
        mc = mc[None, :]
    if mc.ndim != 2:
        raise ValueError("mc must have shape (npts, nslm) or (nslm,)")
    dz_mm = _as_1d("dz_mm", dz_mm)
    if dz_mm.shape[0] != mc.shape[1]:
        raise ValueError("dz_mm length must match nslm")
    mask_soiltile = _as_1d("mask_soiltile", mask_soiltile)
    _require_same_npts(mc, mask_soiltile)

    npts, nslm = mc.shape
    if nslm < 2:
        raise ValueError("hydrol_soil_smooth_under_mcr requires at least two layers")
    mcr = _as_1d("mcr", mcr)
    if mcr.shape[0] == 1:
        mcr = jnp.repeat(mcr, npts)
    if mcr.shape[0] != npts:
        raise ValueError("mcr must be scalar or have shape (npts,)")

    soil_tile_index = int(soil_tile_index)
    peat_full = bool(peat_hydro) and soil_tile_index in tuple(int(tile) for tile in peat_tiles)
    peat_45 = bool(peat_hydro) and soil_tile_index in (4, 5)
    threshold_full = jnp.full((npts,), mcr_peat, dtype=mc.dtype) if peat_full else mcr
    threshold_45 = jnp.full((npts,), mcr_peat, dtype=mc.dtype) if peat_45 else mcr

    tmci = hydrol_total_moisture_content(mc, dz_mm) if check_cwrr2 else None
    mc_out = mc

    for jsl in range(0, nslm - 2):
        excess = jnp.maximum(threshold_full - mc_out[:, jsl], 0.0)
        mc_out = mc_out.at[:, jsl].add(excess)
        ratio = (dz_mm[jsl] + dz_mm[jsl + 1]) / (dz_mm[jsl + 1] + dz_mm[jsl + 2])
        mc_out = mc_out.at[:, jsl + 1].add(-excess * ratio)

    jsl = nslm - 2
    excess = jnp.maximum(threshold_full - mc_out[:, jsl], 0.0)
    mc_out = mc_out.at[:, jsl].add(excess)
    mc_out = mc_out.at[:, jsl + 1].add(-excess * (dz_mm[jsl] + dz_mm[jsl + 1]) / dz_mm[jsl + 1])

    jsl = nslm - 1
    excess = jnp.maximum(threshold_45 - mc_out[:, jsl], 0.0)
    mc_out = mc_out.at[:, jsl].add(excess)
    mc_out = mc_out.at[:, jsl - 1].add(-excess * dz_mm[jsl] / (dz_mm[jsl - 1] + dz_mm[jsl]))

    for jsl in range(nslm - 2, 0, -1):
        excess = jnp.maximum(threshold_full - mc_out[:, jsl], 0.0)
        mc_out = mc_out.at[:, jsl].add(excess)
        ratio = (dz_mm[jsl] + dz_mm[jsl + 1]) / (dz_mm[jsl - 1] + dz_mm[jsl])
        mc_out = mc_out.at[:, jsl - 1].add(-excess * ratio)

    excess_top = mask_soiltile * jnp.maximum(threshold_45 - mc_out[:, 0], 0.0)
    mc_out = mc_out.at[:, 0].add(excess_top)
    is_under_mcr = excess_top > min_sechiba
    mc_out = mc_out - excess_top[:, None] * dz_mm[1] / (2.0 * jnp.asarray(zmaxh_m, dtype=mc.dtype) * 1000.0)
    mc_out = mask_soiltile[:, None] * mc_out

    check_under_ns = None
    if check_cwrr2:
        tmcf = hydrol_total_moisture_content(mc_out, dz_mm)
        check_under_ns = tmcf - tmci

    return HydrolUnderResidualCorrectionResult(
        mc=mc_out,
        excess_top=excess_top,
        is_under_mcr=is_under_mcr,
        check_under_ns=check_under_ns,
    )


def hydrol_soil_solve_diagnostics(
    setup: HydrolSoilSetupResult,
    mcl_before,
    b,
    dt_days,
    free_drain_coef,
    *,
    flux_top,
    rootsink,
    resolv,
    k_bottom=None,
    mask_soiltile=None,
    check_tr_ns=None,
):
    """Compose audited HYDROL solve diagnostics without full state update.

    Fortran provenance: main RHS construction in `hydrol_soil`, lines
    5953-5991; tridiagonal solve call at line 6004 and solver lines
    8147-8198; optional bottom drainage lines 6007-6016; optional drainage
    correction lines 6042-6047. This function does not reconstruct `mcl`,
    update total `mc`, compute `tmci/tmcf`, or evaluate alternate-branch
    triggers.
    """

    rhs_result = hydrol_soil_rhs_main(
        setup=setup,
        mcl=mcl_before,
        b=b,
        dt_days=dt_days,
        free_drain_coef=free_drain_coef,
        flux_top=flux_top,
        rootsink=rootsink,
    )
    solved = hydrol_soil_tridiag_solve(
        e=rhs_result.e,
        f=rhs_result.f,
        g1=rhs_result.g1,
        rhs=rhs_result.rhs,
        resolv=resolv,
        initial_mcl=mcl_before,
    )

    dr_before_corr = None
    dr_corrnum = None
    dr_after_corr = None
    if k_bottom is not None:
        if mask_soiltile is None:
            raise ValueError("mask_soiltile is required when k_bottom is provided")
        dr_before_corr = bottom_drainage(
            k_bottom=k_bottom,
            free_drain_coef=free_drain_coef,
            dt_days=dt_days,
            mask_soiltile=mask_soiltile,
            resolv=resolv,
        )
        if check_tr_ns is not None:
            corrected = drainage_correction(dr_before_corr, check_tr_ns)
            dr_corrnum = corrected.dr_corrnum
            dr_after_corr = corrected.dr_after

    return HydrolSoilSolveDiagnostics(
        rhs=rhs_result.rhs,
        mcl_after=solved.mcl,
        bet=solved.bet,
        gam=solved.gam,
        dr_before_corr=dr_before_corr,
        dr_corrnum=dr_corrnum,
        dr_after_corr=dr_after_corr,
    )


def hydrol_soil_explicit_solve_step(
    setup: HydrolSoilSetupResult,
    *,
    mcl_before,
    mc_before,
    profil_froz,
    mcr,
    b,
    dz_mm,
    dt_days,
    free_drain_coef,
    flux_top,
    rootsink,
    resolv,
    mask_soiltile,
    k_bottom,
    check_tr_ns=None,
    mcs=None,
    clip_over_mcs=False,
    ru_ns=None,
    route_over_mcs=False,
    ok_freeze_cwrr=False,
    check_over_mcs=False,
    correct_negative_runoff=False,
    force_water_table=False,
    zwt_force=None,
    zz_mm=None,
    mcs_peat=PEAT_MCS,
    diagnose_wtd=False,
    undef_sechiba=1.0e20,
    smooth_under_mcr=False,
    zmaxh_m=None,
    peat_hydro=False,
    soil_tile_index=1,
    peat_tiles=(4, 5, 6),
    mcr_peat=PEAT_MCR,
    check_under_mcr=False,
):
    """Compose one explicit-input HYDROL solve diagnostic step.

    Fortran provenance: RHS construction in `hydrol_soil`, lines 5953-5991;
    tridiagonal call at line 6004 and solver lines 8147-8198; drainage and
    conservation lines 6007-6052; total moisture update lines 6063-6075;
    optional over-saturation clipping/routing in lines 6076-6092 and
    `hydrol_soil_smooth_over_mcs2` lines 7935-8026, optional negative-runoff
    correction lines 6102-6114, optional forced water-table saturation lines
    6116-6158, water-table depth diagnosis lines 6162-6180, and optional
    under-residual smoothing through `hydrol_soil_smooth_under_mcr` lines
    7623-7770. This adapter does not compute hidden upstream state such as
    `flux_top`, `rootsink`, `resolv`, freeze profile, or alternate-branch
    triggers.
    """

    rhs_result = hydrol_soil_rhs_main(
        setup=setup,
        mcl=mcl_before,
        b=b,
        dt_days=dt_days,
        free_drain_coef=free_drain_coef,
        flux_top=flux_top,
        rootsink=rootsink,
    )
    solved = hydrol_soil_tridiag_solve(
        e=rhs_result.e,
        f=rhs_result.f,
        g1=rhs_result.g1,
        rhs=rhs_result.rhs,
        resolv=resolv,
        initial_mcl=mcl_before,
    )
    tmci = hydrol_total_moisture_content(mcl_before, dz_mm)
    tmcf = hydrol_total_moisture_content(solved.mcl, dz_mm)
    dr_before_corr = bottom_drainage(
        k_bottom=k_bottom,
        free_drain_coef=free_drain_coef,
        dt_days=dt_days,
        mask_soiltile=mask_soiltile,
        resolv=resolv,
    )
    if check_tr_ns is None:
        check_tr_ns = hydrol_liquid_redistribution_check(
            tmci=tmci,
            tmcf=tmcf,
            flux_top=flux_top,
            dr_ns=dr_before_corr,
            rootsink=rootsink,
        )
    corrected = drainage_correction(dr_before_corr, check_tr_ns)
    mc_after = hydrol_mcl_to_mc_after_solve(
        mcl_after=solved.mcl,
        mc_before=mc_before,
        profil_froz=profil_froz,
        mcr=mcr,
    )

    over_mcs = None
    over_mcs_routing = None
    negative_runoff = None
    forced_water_table = None
    water_table_depth = None
    under_mcr = None
    mc_final = mc_after
    ru_ns_final = None
    dr_ns_final = corrected.dr_after
    if clip_over_mcs:
        if mcs is None:
            raise ValueError("mcs is required when clip_over_mcs=True")
        over_mcs = hydrol_soil_clip_over_mcs2(mc=mc_after, mcs=mcs, dz_mm=dz_mm)
        mc_final = over_mcs.mc

    if route_over_mcs:
        if mcs is None:
            raise ValueError("mcs is required when route_over_mcs=True")
        if ru_ns is None:
            raise ValueError("ru_ns is required when route_over_mcs=True")
        over_mcs_routing = hydrol_route_over_mcs2(
            mc=mc_after,
            mcs=mcs,
            dz_mm=dz_mm,
            ru_ns=ru_ns,
            dr_ns=corrected.dr_after,
            free_drain_coef=free_drain_coef,
            ok_freeze_cwrr=ok_freeze_cwrr,
            check_cwrr2=check_over_mcs,
        )
        mc_final = over_mcs_routing.mc
        ru_ns_final = over_mcs_routing.ru_ns
        dr_ns_final = over_mcs_routing.dr_ns

    if correct_negative_runoff:
        if ru_ns_final is None:
            if ru_ns is None:
                raise ValueError("ru_ns is required when correct_negative_runoff=True")
            ru_ns_final = _as_1d("ru_ns", ru_ns)
        negative_runoff = hydrol_correct_negative_runoff(ru_ns=ru_ns_final, dr_ns=dr_ns_final)
        ru_ns_final = negative_runoff.ru_ns
        dr_ns_final = negative_runoff.dr_ns

    if force_water_table:
        if zmaxh_m is None:
            raise ValueError("zmaxh_m is required when force_water_table=True")
        if zwt_force is None:
            raise ValueError("zwt_force is required when force_water_table=True")
        if zz_mm is None:
            raise ValueError("zz_mm is required when force_water_table=True")
        if mcs is None:
            raise ValueError("mcs is required when force_water_table=True")
        forced_water_table = hydrol_force_water_table_saturation(
            mc=mc_final,
            dr_ns=dr_ns_final,
            zwt_force=zwt_force,
            zz_mm=zz_mm,
            dz_mm=dz_mm,
            zmaxh_m=zmaxh_m,
            mcs=mcs,
            peat_hydro=peat_hydro,
            soil_tile_index=soil_tile_index,
            peat_tiles=peat_tiles,
            mcs_peat=mcs_peat,
        )
        mc_final = forced_water_table.mc
        dr_ns_final = forced_water_table.dr_ns

    if diagnose_wtd:
        if zz_mm is None:
            raise ValueError("zz_mm is required when diagnose_wtd=True")
        if mcs is None:
            raise ValueError("mcs is required when diagnose_wtd=True")
        water_table_depth = hydrol_effective_water_table_depth(
            mc=mc_final,
            zz_mm=zz_mm,
            mcs=mcs,
            peat_hydro=peat_hydro,
            soil_tile_index=soil_tile_index,
            peat_tiles=peat_tiles,
            mcs_peat=mcs_peat,
            undef_sechiba=undef_sechiba,
        )

    if smooth_under_mcr:
        if zmaxh_m is None:
            raise ValueError("zmaxh_m is required when smooth_under_mcr=True")
        under_mcr = hydrol_soil_smooth_under_mcr(
            mc=mc_final,
            mcr=mcr,
            dz_mm=dz_mm,
            zmaxh_m=zmaxh_m,
            mask_soiltile=mask_soiltile,
            peat_hydro=peat_hydro,
            soil_tile_index=soil_tile_index,
            peat_tiles=peat_tiles,
            mcr_peat=mcr_peat,
            check_cwrr2=check_under_mcr,
        )
        mc_final = under_mcr.mc

    return HydrolSoilExplicitStepResult(
        rhs=rhs_result.rhs,
        mcl_after_tridiag=solved.mcl,
        bet=solved.bet,
        gam=solved.gam,
        tmci=tmci,
        tmcf=tmcf,
        dr_before_corr=dr_before_corr,
        check_tr_ns=jnp.asarray(check_tr_ns),
        dr_corrnum=corrected.dr_corrnum,
        dr_after_corr=corrected.dr_after,
        mc_after_mcl_update=mc_after,
        over_mcs=over_mcs,
        over_mcs_routing=over_mcs_routing,
        negative_runoff=negative_runoff,
        forced_water_table=forced_water_table,
        water_table_depth=water_table_depth,
        under_mcr=under_mcr,
        mc_final=mc_final,
        ru_ns_final=ru_ns_final,
        dr_ns_final=dr_ns_final,
    )


def hydrol_trace_backed_full_step(slice_record):
    """Replay one HYDROL full-step trace slice through audited kernels.

    Fortran provenance: this adapter composes existing kernels for
    `hydrol_soil` lines 5953-5991 (RHS), line 6004 plus
    `hydrol_soil_tridiag` lines 8147-8198 (solve), lines 6007-6052
    (drainage/conservation), and lines 6063-6075 (total moisture update). It
    consumes only fields explicitly emitted by the list-directed trace reader
    and reports missing full-step fields through `requires_trace`.
    """

    pre = slice_record.pre_columns()
    post = slice_record.post_tile.values
    update = slice_record.update_columns()

    setup = HydrolSoilSetupResult(
        e=jnp.asarray([pre["e"]]),
        f=jnp.asarray([pre["f"]]),
        g1=jnp.asarray([pre["g1"]]),
        ep=jnp.asarray([pre["ep"]]),
        fp=jnp.asarray([pre["fp"]]),
        gp=jnp.asarray([pre["gp"]]),
    )
    check_before_correction = post["check_tr_ns"]
    dr_before = post["dr_ns"] - post["dr_corrnum_ns"]

    explicit = hydrol_soil_explicit_solve_step(
        setup=setup,
        mcl_before=jnp.asarray([pre["mcl"]]),
        mc_before=jnp.asarray([pre["mc"]]),
        profil_froz=jnp.asarray([pre["profil_froz_hydro_ns"]]),
        mcr=jnp.asarray([pre["mcr"][0]]),
        b=jnp.asarray([pre["b"]]),
        dz_mm=_trace_dz_mm_from_right_geometry(jnp.asarray(pre["ep"]), jnp.asarray(pre["gp"])),
        dt_days=pre["dt_days"][0],
        free_drain_coef=jnp.asarray([post["free_drain_coef"]]),
        flux_top=jnp.asarray([post["flux_top"]]),
        rootsink=jnp.asarray([pre["rootsink"]]),
        resolv=jnp.asarray([post["resolv"]]),
        mask_soiltile=jnp.asarray([pre["mask_soiltile"][0]]),
        k_bottom=jnp.asarray([post["k_bottom"]]),
        check_tr_ns=jnp.asarray([check_before_correction]),
    )

    trace_rhs = jnp.asarray([pre["rhs"]])
    trace_mcl_after = jnp.asarray([update["mcl"]])
    trace_mc_after = jnp.asarray([update["mc"]])
    trace_dr_before = jnp.asarray([dr_before])
    trace_dr_corrnum = jnp.asarray([post["dr_corrnum_ns"]])
    trace_dr_after = jnp.asarray([post["dr_ns"]])

    errors = {
        "rhs": float(jnp.max(jnp.abs(explicit.rhs - trace_rhs))),
        "mcl_after_tridiag": float(jnp.max(jnp.abs(explicit.mcl_after_tridiag - trace_mcl_after))),
        "mc_after_update": float(jnp.max(jnp.abs(explicit.mc_after_mcl_update - trace_mc_after))),
        "tmci": float(jnp.max(jnp.abs(explicit.tmci - jnp.asarray([post["tmci"]])))),
        "tmcf": float(jnp.max(jnp.abs(explicit.tmcf - jnp.asarray([post["tmcf"]])))),
        "check_tr_ns": float(jnp.max(jnp.abs(explicit.check_tr_ns - jnp.asarray([post["check_tr_ns"]])))),
        "dr_before_corr": float(jnp.max(jnp.abs(explicit.dr_before_corr - trace_dr_before))),
        "dr_corrnum": float(jnp.max(jnp.abs(explicit.dr_corrnum - trace_dr_corrnum))),
        "dr_after_corr": float(jnp.max(jnp.abs(explicit.dr_after_corr - trace_dr_after))),
    }

    return HydrolTraceBackedFullStepResult(
        explicit=explicit,
        max_abs_errors=errors,
        requires_trace=tuple(slice_record.requires_trace),
    )


def hydrol_trace_backed_alt_residual(slice_record):
    """Check one alternate residual-boundary trace slice.

    Fortran provenance: trigger logic comes from `hydrol_soil` lines
    7011-7018; residual boundary reset comes from lines 7031-7039; inactive
    preservation follows the `hydrol_soil_tridiag` `resolv` guard at lines
    8171-8185 and 8191-8195. This function checks only fields present in the
    2026-06-23 list-directed trace.
    """

    first = slice_record.alt_first_solve.values
    residual = slice_record.residual_columns()

    resolv_expected = hydrol_alt_residual_trigger(
        mcl_top_after_first_solve=first["mcl_top_after_first_solve"],
        flux_top=first["flux_top"],
        mcr=first["mcr"],
        min_sechiba=first["min_sechiba"],
    )
    trace_resolv = jnp.asarray(residual["resolv"], dtype=bool)
    top_rhs_expected = jnp.asarray(first["mcr"])
    top_f_expected = jnp.asarray(1.0)
    top_g1_expected = jnp.asarray(0.0)

    errors = {
        "resolv": float(jnp.max(jnp.abs(trace_resolv.astype(jnp.int32) - resolv_expected.astype(jnp.int32)))),
        "top_rhs": float(jnp.abs(jnp.asarray(residual["rhs"][0]) - top_rhs_expected)),
        "top_f": float(jnp.abs(jnp.asarray(residual["tmat_f"][0]) - top_f_expected)),
        "top_g1": float(jnp.abs(jnp.asarray(residual["tmat_g1"][0]) - top_g1_expected)),
    }
    if not bool(resolv_expected):
        errors["inactive_top_mcl_preserved"] = float(
            jnp.abs(
                jnp.asarray(residual["mcl_after_residual_solve"][0])
                - jnp.asarray(first["mcl_top_after_first_solve"])
            )
        )
    else:
        errors["active_top_mcl_equals_mcr"] = float(
            jnp.abs(jnp.asarray(residual["mcl_after_residual_solve"][0]) - top_rhs_expected)
        )

    return HydrolAltResidualCheckResult(
        resolv_expected=jnp.asarray(resolv_expected),
        residual_top_rhs_expected=top_rhs_expected,
        residual_top_f_expected=top_f_expected,
        residual_top_g1_expected=top_g1_expected,
        max_abs_errors=errors,
        requires_trace=tuple(slice_record.requires_trace),
    )


def _trace_dz_mm_from_right_geometry(ep, gp):
    """Recover `dz` from emitted implicit-matrix geometric terms.

    Fortran provenance: `hydrol_soil_setup`, lines 8506-8565 with `w_time=1`
    makes `ep/fp/gp` pure geometry, but the 2026-06-23 trace did not emit
    `dz_mm`. For this trace-only adapter we recover `dz` from the emitted
    right-hand matrix records: top `gp(1)=dz(2)/8`, mid
    `ep(jsl)=dz(jsl)/8`, and bottom `ep(nslm)=dz(nslm)/8`. This is blocked
    from becoming general state wiring until `dz_mm` is emitted directly.
    """

    ep = jnp.asarray(ep)
    gp = jnp.asarray(gp)
    dz_mm = jnp.zeros_like(ep)
    dz_mm = dz_mm.at[1].set(8.0 * gp[0])
    for jsl in range(2, int(ep.shape[0])):
        dz_mm = dz_mm.at[jsl].set(8.0 * ep[jsl])
    return dz_mm


def _as_1d(name, value):
    array = jnp.asarray(value)
    if array.ndim == 0:
        array = array[None]
    if array.ndim != 1:
        raise ValueError(f"{name} must be a scalar or one-dimensional array")
    return array


def _as_2d(name, value):
    array = jnp.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array with shape (npts, ncols)")
    return array


def _as_3d(name, value):
    array = jnp.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a three-dimensional array")
    return array


def _as_4d(name, value):
    array = jnp.asarray(value)
    if array.ndim != 4:
        raise ValueError(f"{name} must be a four-dimensional array")
    return array


def _require_same_npts(*arrays):
    npts = arrays[0].shape[0]
    for array in arrays[1:]:
        if array.shape[0] != npts:
            raise ValueError("all inputs must agree on the npts axis")
