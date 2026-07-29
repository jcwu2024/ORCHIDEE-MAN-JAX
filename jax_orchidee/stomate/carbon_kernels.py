"""Independent STOMATE daily carbon kernels for Phase 1B.

The functions here stop before the full STOMATE daily loop. They implement
source-closed algebra only; callers must provide state that is produced by
unimplemented upstream routines such as phenology, season, and allocation
stress.
"""

from __future__ import annotations

from typing import NamedTuple

from jax import config, jit

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


# Fortran pool indices are 1-based in `constantes_var.f90`, lines 196-208.
ILEAF = 0
ISAPABOVE = 1
ISAPBELOW = 2
IHEARTABOVE = 3
IHEARTBELOW = 4
IROOT = 5
IFRUIT = 6
ICARBRES = 7
IAGRSAPST = 8
IAGRSAPPN = 9
IAGRHRTST = 10
IAGRHRTPN = 11
NPARTS = 12
ICARBON = 0
IABOVE = 0
IBELOW = 1
NLEVS = 2
NLEAFAGES = 4
IMETABOLIC = 0
ISTRUCTURAL = 1
NLITT = 2
IACTIVE = 0
ISLOW = 1
IPASSIVE = 2
NCARB = 3
IWPLCC = 0
IWPHAR = 1
NWP = 2
IMETABO = 0
ISTRABO = 1
IMETBEL = 2
ISTRBEL = 3
IACT = 4
ISLO = 5
IPAS = 6
NPOOL = 7
SENESCENCE_NONE = 0
SENESCENCE_CROP = 1
SENESCENCE_COLD = 2
SENESCENCE_DRY = 3
SENESCENCE_MIXED = 4

ZERO_CELSIUS = 273.15
ONE_YEAR_DAYS = 365.0
DEFAULT_MAINT_RESP_MIN_VMAX = 0.3
DEFAULT_MAINT_RESP_COEFF = 1.4
DEFAULT_UNDEF = -9999.0


def _host_bool_tuple_or_none(values) -> tuple[bool, ...] | None:
    try:
        array = np.asarray(values, dtype=bool)
    except Exception:
        return None
    if array.ndim != 1:
        return None
    return tuple(bool(value) for value in array)
DEFAULT_CN = (40.0,) * NPARTS
DEFAULT_LC = (0.22, 0.35, 0.35, 0.35, 0.35, 0.22, 0.22, 0.22, 0.35, 0.35, 0.35, 0.35)
DEFAULT_FRAC_SOIL_VALUES = {
    (ISTRUCTURAL, IACTIVE, IABOVE): 0.55,
    (ISTRUCTURAL, IACTIVE, IBELOW): 0.45,
    (ISTRUCTURAL, ISLOW, IABOVE): 0.7,
    (ISTRUCTURAL, ISLOW, IBELOW): 0.7,
    (IMETABOLIC, IACTIVE, IABOVE): 0.45,
    (IMETABOLIC, IACTIVE, IBELOW): 0.45,
}
PUMPED_BIOMASS_PARTS = (
    ILEAF,
    ISAPABOVE,
    ISAPBELOW,
    IROOT,
    IFRUIT,
    ICARBRES,
    IAGRSAPST,
    IAGRSAPPN,
)
WOOD_BIOMASS_PARTS = (
    ISAPABOVE,
    ISAPBELOW,
    IHEARTABOVE,
    IHEARTBELOW,
    IAGRSAPST,
    IAGRSAPPN,
    IAGRHRTST,
    IAGRHRTPN,
)


class LittercalcEntryPrep(NamedTuple):
    """Local arrays prepared before the ``littercalc_leak`` call."""

    soil_mc_32l: jnp.ndarray
    turnover_littercalc: jnp.ndarray
    bm_to_littercalc: jnp.ndarray


class LittercalcStaticFactors(NamedTuple):
    """First-call and per-step litter constants used by ``littercalc_leak``."""

    litterfrac: jnp.ndarray
    frac_soil: jnp.ndarray
    litter_tau: jnp.ndarray
    carbon_tau: jnp.ndarray


class LittercalcActivityFactors(NamedTuple):
    """Per-PFT/layer activity factors for metabolic and structural litter."""

    fbact_met: jnp.ndarray
    fbact_str: jnp.ndarray


class LitterIncrementResult(NamedTuple):
    """Litter pool increments before decomposition."""

    litter_inc_above: jnp.ndarray
    litter_inc_below: jnp.ndarray
    litter_inc_pft_above: jnp.ndarray
    litter_inc_pft_below: jnp.ndarray
    dead_leaves: jnp.ndarray
    lignin_struc_inc_above: jnp.ndarray
    lignin_struc_inc_below: jnp.ndarray


class LitterPoolUpdateResult(NamedTuple):
    """Litter pools and lignin fractions after section 2 additions."""

    litter_above: jnp.ndarray
    litter_below: jnp.ndarray
    lignin_struc_above: jnp.ndarray
    lignin_struc_below: jnp.ndarray
    litterpart: jnp.ndarray
    increments: LitterIncrementResult


class ProductPoolAgingResult(NamedTuple):
    """Product pools after the LCC/harvest annual aging update."""

    prod10: jnp.ndarray
    prod100: jnp.ndarray
    flux10: jnp.ndarray
    flux100: jnp.ndarray
    convflux: jnp.ndarray
    cflux_prod10: jnp.ndarray
    cflux_prod100: jnp.ndarray


class LcchangeMainLeakResult(NamedTuple):
    """Ordinary non-age-class ``lcchange_main`` state after one net-LCC step."""

    veget_max: jnp.ndarray
    biomass: jnp.ndarray
    ind: jnp.ndarray
    age: jnp.ndarray
    pft_present: jnp.ndarray
    senescence: jnp.ndarray
    when_growthinit: jnp.ndarray
    everywhere: jnp.ndarray
    co2_to_bm: jnp.ndarray
    bm_to_litter: jnp.ndarray
    turnover_daily: jnp.ndarray
    cn_ind: jnp.ndarray
    prod10: jnp.ndarray
    prod100: jnp.ndarray
    flux10: jnp.ndarray
    flux100: jnp.ndarray
    convflux: jnp.ndarray
    cflux_prod10: jnp.ndarray
    cflux_prod100: jnp.ndarray
    leaf_frac: jnp.ndarray
    npp_longterm: jnp.ndarray
    lm_lastyearmax: jnp.ndarray
    litter_avail: jnp.ndarray
    litter_not_avail: jnp.ndarray
    carbon: jnp.ndarray
    deepC_a: jnp.ndarray
    deepC_s: jnp.ndarray
    deepC_p: jnp.ndarray
    fuel_1hr: jnp.ndarray
    fuel_10hr: jnp.ndarray
    fuel_100hr: jnp.ndarray
    fuel_1000hr: jnp.ndarray
    litter_above: jnp.ndarray
    litter_below: jnp.ndarray
    carbon_32l: jnp.ndarray
    doc: jnp.ndarray


class LcchangeDeffireResult(NamedTuple):
    """Non-age-class ``lcchange_deffire`` state after deforestation fire LCC."""

    veget_max: jnp.ndarray
    veget_max_new: jnp.ndarray
    biomass: jnp.ndarray
    ind: jnp.ndarray
    age: jnp.ndarray
    pft_present: jnp.ndarray
    senescence: jnp.ndarray
    when_growthinit: jnp.ndarray
    everywhere: jnp.ndarray
    co2_to_bm: jnp.ndarray
    bm_to_litter: jnp.ndarray
    turnover_daily: jnp.ndarray
    cn_ind: jnp.ndarray
    prod10: jnp.ndarray
    prod100: jnp.ndarray
    flux10: jnp.ndarray
    flux100: jnp.ndarray
    convflux: jnp.ndarray
    cflux_prod10: jnp.ndarray
    cflux_prod100: jnp.ndarray
    leaf_frac: jnp.ndarray
    npp_longterm: jnp.ndarray
    lm_lastyearmax: jnp.ndarray
    litter: jnp.ndarray
    litter_avail: jnp.ndarray
    litter_not_avail: jnp.ndarray
    carbon: jnp.ndarray
    deepC_a: jnp.ndarray
    deepC_s: jnp.ndarray
    deepC_p: jnp.ndarray
    fuel_1hr: jnp.ndarray
    fuel_10hr: jnp.ndarray
    fuel_100hr: jnp.ndarray
    fuel_1000hr: jnp.ndarray
    deflitsup_total: jnp.ndarray
    defbiosup_total: jnp.ndarray


class ForestHarvestLegacyResult(NamedTuple):
    """Forest-clearing legacy fluxes before product-pool aging."""

    bm_to_litter_pro: jnp.ndarray
    convflux: jnp.ndarray
    prod10: jnp.ndarray
    prod100: jnp.ndarray
    above: jnp.ndarray


class ForestHarvestProxyPoolsResult(NamedTuple):
    """Forest-clearing litter/fuel/lignin proxy pools."""

    litter_pro: jnp.ndarray
    fuel_1hr_pro: jnp.ndarray
    fuel_10hr_pro: jnp.ndarray
    fuel_100hr_pro: jnp.ndarray
    fuel_1000hr_pro: jnp.ndarray
    lignin_content_pro: jnp.ndarray


class HerbHarvestResult(NamedTuple):
    """Herbaceous-clearing biomass-to-litter proxy transfer."""

    bm_to_litter_pro: jnp.ndarray


class ProxyPftInitializationResult(NamedTuple):
    """New youngest-age-class proxy PFT initialized during LCC."""

    biomass_pro: jnp.ndarray
    co2_to_bm_pro: jnp.ndarray
    ind_pro: jnp.ndarray
    age_pro: jnp.ndarray
    senescence_pro: jnp.ndarray
    pft_present_pro: jnp.ndarray
    lm_lastyearmax_pro: jnp.ndarray
    everywhere_pro: jnp.ndarray
    npp_longterm_pro: jnp.ndarray
    leaf_frac_pro: jnp.ndarray
    leaf_age_pro: jnp.ndarray


class SapTakeResult(NamedTuple):
    """Existing biomass after sapling demand is taken from age classes."""

    biomass: jnp.ndarray
    co2_to_bm_pro: jnp.ndarray
    biomass_total: jnp.ndarray


class LegacyScalarPoolsResult(NamedTuple):
    """Fraction-weighted legacy pools collected for a proxy PFT."""

    veget_max_pro: jnp.ndarray
    bm_to_litter_pro: jnp.ndarray
    carbon_pro: jnp.ndarray
    deepC_a_pro: jnp.ndarray
    deepC_s_pro: jnp.ndarray
    deepC_p_pro: jnp.ndarray
    co2_to_bm_pro: jnp.ndarray
    gpp_daily_pro: jnp.ndarray
    npp_daily_pro: jnp.ndarray
    resp_maint_pro: jnp.ndarray
    resp_growth_pro: jnp.ndarray
    resp_hetero_pro: jnp.ndarray
    co2_fire_pro: jnp.ndarray
    lignin_struc_pro: jnp.ndarray


class LccLegacyCollectionResult(NamedTuple):
    """Collected source-PFT legacy state for one receiving MTC."""

    veget_max_pro: jnp.ndarray
    carbon_pro: jnp.ndarray
    lignin_struc_pro: jnp.ndarray
    litter_pro: jnp.ndarray
    deepC_a_pro: jnp.ndarray
    deepC_s_pro: jnp.ndarray
    deepC_p_pro: jnp.ndarray
    fuel_1hr_pro: jnp.ndarray
    fuel_10hr_pro: jnp.ndarray
    fuel_100hr_pro: jnp.ndarray
    fuel_1000hr_pro: jnp.ndarray
    bm_to_litter_pro: jnp.ndarray
    co2_to_bm_pro: jnp.ndarray
    gpp_daily_pro: jnp.ndarray
    npp_daily_pro: jnp.ndarray
    resp_maint_pro: jnp.ndarray
    resp_growth_pro: jnp.ndarray
    resp_hetero_pro: jnp.ndarray
    co2_fire_pro: jnp.ndarray
    convflux: jnp.ndarray
    prod10: jnp.ndarray
    prod100: jnp.ndarray


class EmptyPftResult(NamedTuple):
    """State after a PFT slot is emptied by LCC or age-class movement."""

    veget_max: jnp.ndarray
    biomass: jnp.ndarray
    ind: jnp.ndarray
    carbon: jnp.ndarray
    litter: jnp.ndarray
    lignin_struc: jnp.ndarray
    bm_to_litter: jnp.ndarray
    deepC_a: jnp.ndarray
    deepC_s: jnp.ndarray
    deepC_p: jnp.ndarray
    fuel_1hr: jnp.ndarray
    fuel_10hr: jnp.ndarray
    fuel_100hr: jnp.ndarray
    fuel_1000hr: jnp.ndarray
    gpp_daily: jnp.ndarray
    npp_daily: jnp.ndarray
    gpp_week: jnp.ndarray
    npp_longterm: jnp.ndarray
    co2_to_bm: jnp.ndarray
    resp_maint: jnp.ndarray
    resp_growth: jnp.ndarray
    resp_hetero: jnp.ndarray
    lm_lastyearmax: jnp.ndarray
    leaf_frac: jnp.ndarray
    leaf_age: jnp.ndarray
    age: jnp.ndarray
    everywhere: jnp.ndarray
    pft_present: jnp.ndarray
    when_growthinit: jnp.ndarray
    senescence: jnp.ndarray
    gdd_from_growthinit: jnp.ndarray
    gdd_midwinter: jnp.ndarray
    time_hum_min: jnp.ndarray
    gdd_m5_dormance: jnp.ndarray
    ncd_dormance: jnp.ndarray
    moiavail_month: jnp.ndarray
    moiavail_week: jnp.ndarray
    ngd_minus5: jnp.ndarray


class IncomingProxyPftResult(NamedTuple):
    """State after an incoming proxy PFT is merged into its target slot."""

    veget_max: jnp.ndarray
    carbon: jnp.ndarray
    litter: jnp.ndarray
    lignin_struc: jnp.ndarray
    bm_to_litter: jnp.ndarray
    deepC_a: jnp.ndarray
    deepC_s: jnp.ndarray
    deepC_p: jnp.ndarray
    fuel_1hr: jnp.ndarray
    fuel_10hr: jnp.ndarray
    fuel_100hr: jnp.ndarray
    fuel_1000hr: jnp.ndarray
    biomass: jnp.ndarray
    co2_to_bm: jnp.ndarray
    npp_longterm: jnp.ndarray
    ind: jnp.ndarray
    lm_lastyearmax: jnp.ndarray
    age: jnp.ndarray
    everywhere: jnp.ndarray
    leaf_frac: jnp.ndarray
    leaf_age: jnp.ndarray
    pft_present: jnp.ndarray
    senescence: jnp.ndarray
    gpp_daily: jnp.ndarray
    npp_daily: jnp.ndarray
    resp_maint: jnp.ndarray
    resp_growth: jnp.ndarray
    resp_hetero: jnp.ndarray
    co2_fire: jnp.ndarray


class LccAgeClassIndices(NamedTuple):
    """PFT indices grouped for gross LCC old-to-young allocation."""

    indall_tree: jnp.ndarray
    indold_tree: jnp.ndarray
    indagec_tree: jnp.ndarray
    indall_grass: jnp.ndarray
    indold_grass: jnp.ndarray
    indagec_grass: jnp.ndarray
    indall_pasture: jnp.ndarray
    indold_pasture: jnp.ndarray
    indagec_pasture: jnp.ndarray
    indall_crop: jnp.ndarray
    indold_crop: jnp.ndarray
    indagec_crop: jnp.ndarray


class LccCoverResult(NamedTuple):
    """MTC and age-class cover fractions used by gross LCC."""

    veget_mtc: jnp.ndarray
    vegagec_tree: jnp.ndarray
    vegagec_grass: jnp.ndarray
    vegagec_pasture: jnp.ndarray
    vegagec_crop: jnp.ndarray


class CrossGiveReceiveResult(NamedTuple):
    """Gross-LCC donor and receiver allocation state."""

    veget_max: jnp.ndarray
    glcc_pft: jnp.ndarray
    glcc_pftmtc: jnp.ndarray
    glcc_pft_tmp: jnp.ndarray


class TypeConversionResult(NamedTuple):
    """State after one gross-LCC type-conversion transition is allocated."""

    vegagec_donor: jnp.ndarray
    veget_max: jnp.ndarray
    glcc_pft: jnp.ndarray
    glcc_pftmtc: jnp.ndarray
    glcc_pft_tmp: jnp.ndarray
    glcc_remain: jnp.ndarray


class GlccCompensationFullResult(NamedTuple):
    """State after one gross-LCC feasibility compensation target pass."""

    veget_4veg: jnp.ndarray
    glcc_real: jnp.ndarray
    glcc_def: jnp.ndarray
    incre_deficit: jnp.ndarray


class HarvestSameMtcResult(NamedTuple):
    """Forestry-harvest loss remapped to each tree PFT's own MTC."""

    glcc_pft_tmp: jnp.ndarray
    glcc_pftmtc: jnp.ndarray


class PrimaryNetLccSequenceResult(NamedTuple):
    """Gross-LCC primary-shift plus net-LCC transition allocation state."""

    glcc_real: jnp.ndarray
    glcc_remain: jnp.ndarray
    veget_max: jnp.ndarray
    glcc_pft: jnp.ndarray
    glcc_pftmtc: jnp.ndarray
    glcc_pft_tmp: jnp.ndarray
    vegagec_tree: jnp.ndarray
    vegagec_grass: jnp.ndarray
    vegagec_pasture: jnp.ndarray
    vegagec_crop: jnp.ndarray
    incre_deficit: jnp.ndarray


class SecondaryShiftLccSequenceResult(NamedTuple):
    """Gross-LCC secondary-agriculture shifting allocation state."""

    glcc_second_shift_remain: jnp.ndarray
    glcc_remain: jnp.ndarray
    veget_max: jnp.ndarray
    glcc_pft: jnp.ndarray
    glcc_pftmtc: jnp.ndarray
    glcc_pft_tmp: jnp.ndarray
    incre_deficit: jnp.ndarray
    cover_after_first_pass: LccCoverResult


class ForestryHarvestSequenceResult(NamedTuple):
    """Gross-LCC forestry harvest allocation state."""

    veget_max: jnp.ndarray
    glcc_pft: jnp.ndarray
    glcc_pftmtc: jnp.ndarray
    glcc_pft_tmp: jnp.ndarray
    glcc_remain: jnp.ndarray
    fhmatrix_remain_a: jnp.ndarray
    fhmatrix_remain_b: jnp.ndarray
    deficit_pf2yf_final: jnp.ndarray
    deficit_sf2yf_final: jnp.ndarray
    cover_after_secondary: LccCoverResult
    cover_after_compensation: LccCoverResult


class GrossFirstdayLccAllocationResult(NamedTuple):
    """Gross-LCC first-day fraction allocation outputs."""

    veget_max: jnp.ndarray
    glcc_real: jnp.ndarray
    incre_deficit: jnp.ndarray
    glcc_pft: jnp.ndarray
    glcc_pftmtc: jnp.ndarray
    glcc_pft_tmp: jnp.ndarray
    harvest: ForestryHarvestSequenceResult
    secondary_shift: SecondaryShiftLccSequenceResult
    primary_net: PrimaryNetLccSequenceResult


class SingleAgeGrossFirstdayLccAllocationResult(NamedTuple):
    """SingleAgeClass gross-LCC first-day fraction allocation outputs."""

    veget_max: jnp.ndarray
    glcc_real: jnp.ndarray
    glcc_def: jnp.ndarray
    glcc_remain: jnp.ndarray
    incre_deficit: jnp.ndarray
    glcc_pft: jnp.ndarray
    glcc_pftmtc: jnp.ndarray
    glcc_pft_tmp: jnp.ndarray
    hmatrix_real: jnp.ndarray
    deficit_pf2yf_final: jnp.ndarray
    deficit_sf2yf_final: jnp.ndarray
    pf2yf_compen_sf2yf: jnp.ndarray
    sf2yf_compen_pf2yf: jnp.ndarray


class LccOutgoingFractionResult(NamedTuple):
    """Source PFT fractions after one receiver-MTC outgoing subtraction."""

    veget_max: jnp.ndarray
    exhausted: jnp.ndarray


class LccApplyCollectedProxyResult(NamedTuple):
    """Target-PFT state after collected LCC proxy state is applied."""

    state: IncomingProxyPftResult
    exhausted_sources: jnp.ndarray
    veget_max_after_source_subtract: jnp.ndarray
    biomass_pro: jnp.ndarray
    co2_to_bm_pro_after_sap_take: jnp.ndarray


class LccReceiverMtcStepResult(NamedTuple):
    """State after one ``gross_glcchange_fh`` receiver-MTC application."""

    state: IncomingProxyPftResult
    convflux: jnp.ndarray
    prod10: jnp.ndarray
    prod100: jnp.ndarray
    exhausted_sources: jnp.ndarray
    collection: LccLegacyCollectionResult
    biomass_pro: jnp.ndarray
    co2_to_bm_pro_after_sap_take: jnp.ndarray
    gpp_week: jnp.ndarray
    when_growthinit: jnp.ndarray
    gdd_from_growthinit: jnp.ndarray
    gdd_midwinter: jnp.ndarray
    time_hum_min: jnp.ndarray
    gdd_m5_dormance: jnp.ndarray
    ncd_dormance: jnp.ndarray
    moiavail_month: jnp.ndarray
    moiavail_week: jnp.ndarray
    ngd_minus5: jnp.ndarray


class LccAllReceiversApplyResult(NamedTuple):
    """State after all active receiver MTCs and product-pool aging."""

    state: IncomingProxyPftResult
    prod10: jnp.ndarray
    prod100: jnp.ndarray
    flux10: jnp.ndarray
    flux100: jnp.ndarray
    convflux: jnp.ndarray
    cflux_prod10: jnp.ndarray
    cflux_prod100: jnp.ndarray
    gpp_week: jnp.ndarray
    when_growthinit: jnp.ndarray
    gdd_from_growthinit: jnp.ndarray
    gdd_midwinter: jnp.ndarray
    time_hum_min: jnp.ndarray
    gdd_m5_dormance: jnp.ndarray
    ncd_dormance: jnp.ndarray
    moiavail_month: jnp.ndarray
    moiavail_week: jnp.ndarray
    ngd_minus5: jnp.ndarray


class LccGrossGlcchangeStepResult(NamedTuple):
    """Composed gross-LCC allocation and state-application result."""

    allocation: GrossFirstdayLccAllocationResult | SingleAgeGrossFirstdayLccAllocationResult
    applied: LccAllReceiversApplyResult


class FuelPools(NamedTuple):
    """SPITFIRE aboveground fuel pools and their total."""

    fuel_1hr: jnp.ndarray
    fuel_10hr: jnp.ndarray
    fuel_100hr: jnp.ndarray
    fuel_1000hr: jnp.ndarray
    fuel_total: jnp.ndarray


class LitterAbovegroundDecompositionResult(NamedTuple):
    """Aboveground litter decay products from ``littercalc_leak``."""

    litter_above: jnp.ndarray
    dead_leaves: jnp.ndarray
    resp_hetero_litter: jnp.ndarray
    resp_hetero_flood: jnp.ndarray
    soilcarbon_input_doc: jnp.ndarray
    floodcarbon_input: jnp.ndarray
    qd_structural: jnp.ndarray
    qd_metabolic: jnp.ndarray
    qd_flood_structural: jnp.ndarray
    qd_flood_metabolic: jnp.ndarray


class LitterBelowgroundDecompositionResult(NamedTuple):
    """Belowground litter decay products from ``littercalc_leak``."""

    litter_below: jnp.ndarray
    resp_hetero_litter: jnp.ndarray
    resp_hetero_flood: jnp.ndarray
    soilcarbon_input_doc: jnp.ndarray
    floodcarbon_input: jnp.ndarray
    qd_structural: jnp.ndarray
    qd_metabolic: jnp.ndarray
    qd_flood_structural: jnp.ndarray
    qd_flood_metabolic: jnp.ndarray


class LittercalcLeakCoreResult(NamedTuple):
    """Composed ``littercalc_leak`` core with supplied environmental controls."""

    litter_above: jnp.ndarray
    litter_below: jnp.ndarray
    lignin_struc_above: jnp.ndarray
    lignin_struc_below: jnp.ndarray
    litterpart: jnp.ndarray
    dead_leaves: jnp.ndarray
    deadleaf_cover: jnp.ndarray
    resp_hetero_litter: jnp.ndarray
    resp_hetero_flood: jnp.ndarray
    soilcarbon_input_doc: jnp.ndarray
    floodcarbon_input: jnp.ndarray
    fuel: FuelPools
    increments: LitterIncrementResult


class LitterAbovegroundControls(NamedTuple):
    """Aboveground temperature/moisture controls for ``littercalc_leak``."""

    control_temp_above: jnp.ndarray
    control_moist_above: jnp.ndarray
    soilhum_decomp: jnp.ndarray
    soil_mc_top_by_pft: jnp.ndarray


class MaintenanceRespirationResult(NamedTuple):
    """Maintenance respiration and intermediate LAI/coefficient arrays."""

    lai: jnp.ndarray
    t_root: jnp.ndarray
    coeff_maint: jnp.ndarray
    resp_maint_part: jnp.ndarray


def littercalc_litter_fractions(
    cn=DEFAULT_CN,
    lc=DEFAULT_LC,
    *,
    metabolic_ref_frac=0.85,
    metabolic_ln_ratio=0.018,
):
    """Compute biomass-part fractions entering metabolic/structural litter.

    Fortran provenance: ``src_stomate/stomate_litter.f90``, subroutine
    ``littercalc_leak``, first-call table lines 2010-2016 and per-pixel table
    lines 2101-2104. Constants provenance: ``src_parameters/constantes_var.f90``,
    lines 1211-1225.
    """

    cn = jnp.asarray(cn)
    lc = jnp.asarray(lc)
    if cn.shape[0] != NPARTS or lc.shape[0] != NPARTS:
        raise ValueError(f"cn and lc must have length {NPARTS}")
    metabolic = metabolic_ref_frac - metabolic_ln_ratio * lc * cn
    structural = 1.0 - metabolic
    return jnp.stack([metabolic, structural], axis=-1)


def littercalc_frac_soil(
    *,
    frac_soil_struct_aa=0.55,
    frac_soil_struct_ab=0.45,
    frac_soil_struct_sa=0.7,
    frac_soil_struct_sb=0.7,
    frac_soil_metab_aa=0.45,
    frac_soil_metab_ab=0.45,
):
    """Build the litter-to-soil-carbon routing table.

    Fortran provenance: ``src_stomate/stomate_litter.f90``,
    ``littercalc_leak`` lines 2018-2025. Constant defaults come from
    ``src_parameters/constantes_var.f90`` lines 1199-1209.
    """

    table = jnp.zeros((NLITT, NCARB, NLEVS), dtype=jnp.float64)
    table = table.at[ISTRUCTURAL, IACTIVE, IABOVE].set(frac_soil_struct_aa)
    table = table.at[ISTRUCTURAL, IACTIVE, IBELOW].set(frac_soil_struct_ab)
    table = table.at[ISTRUCTURAL, ISLOW, IABOVE].set(frac_soil_struct_sa)
    table = table.at[ISTRUCTURAL, ISLOW, IBELOW].set(frac_soil_struct_sb)
    table = table.at[IMETABOLIC, IACTIVE, IABOVE].set(frac_soil_metab_aa)
    table = table.at[IMETABOLIC, IACTIVE, IBELOW].set(frac_soil_metab_ab)
    return table


def littercalc_turnover_times(
    *,
    tau_metabolic=0.066,
    tau_struct=0.245,
    carbon_tau_iactive=0.149,
    carbon_tau_islow=5.48,
    carbon_tau_ipassive=241.0,
    one_year=ONE_YEAR_DAYS,
):
    """Return litter and soil-carbon residence times in days.

    Fortran provenance: ``src_stomate/stomate_litter.f90``,
    ``littercalc_leak`` lines 1955-1961. Constant defaults:
    ``src_parameters/constantes_var.f90`` lines 1226-1229 and 1350-1354.
    """

    litter_tau = jnp.zeros((NLITT,), dtype=jnp.float64)
    litter_tau = litter_tau.at[IMETABOLIC].set(tau_metabolic * one_year)
    litter_tau = litter_tau.at[ISTRUCTURAL].set(tau_struct * one_year)
    carbon_tau = jnp.zeros((NCARB,), dtype=jnp.float64)
    carbon_tau = carbon_tau.at[IACTIVE].set(carbon_tau_iactive * one_year)
    carbon_tau = carbon_tau.at[ISLOW].set(carbon_tau_islow * one_year)
    carbon_tau = carbon_tau.at[IPASSIVE].set(carbon_tau_ipassive * one_year)
    return litter_tau, carbon_tau


def littercalc_static_factors(cn=DEFAULT_CN, lc=DEFAULT_LC) -> LittercalcStaticFactors:
    """Group source-closed first-call/per-call litter constants."""

    litter_tau, carbon_tau = littercalc_turnover_times()
    return LittercalcStaticFactors(
        litterfrac=littercalc_litter_fractions(cn, lc),
        frac_soil=littercalc_frac_soil(),
        litter_tau=litter_tau,
        carbon_tau=carbon_tau,
    )


def littercalc_activity_factors(fbact, lignin_struc_below, *, litter_struct_coef=3.0) -> LittercalcActivityFactors:
    """Transform ``microactem`` controls into litter activity factors.

    Fortran provenance: ``src_stomate/stomate_litter.f90``,
    ``littercalc_leak`` lines 1950-1968. ``fbact_met`` copies all
    PFT/layer controls scaled by active/metabolic tau, while
    ``fbact_str`` additionally applies the structural lignin exponential.
    """

    fbact = jnp.asarray(fbact)
    lignin_struc_below = jnp.asarray(lignin_struc_below)
    if fbact.ndim != 3:
        raise ValueError("fbact must have shape (npts, ndeep, nvm)")
    if lignin_struc_below.shape != (fbact.shape[0], fbact.shape[2], fbact.shape[1]):
        raise ValueError("lignin_struc_below must have shape (npts, nvm, ndeep)")
    litter_tau, carbon_tau = littercalc_turnover_times()
    active_to_metabolic = carbon_tau[IACTIVE] / litter_tau[IMETABOLIC]
    active_to_structural = carbon_tau[IACTIVE] / litter_tau[ISTRUCTURAL]
    fbact_met = fbact * active_to_metabolic
    lignin_layer_pft = jnp.swapaxes(lignin_struc_below, 1, 2)
    fbact_str = fbact * active_to_structural * jnp.exp(-litter_struct_coef * lignin_layer_pft)
    return LittercalcActivityFactors(fbact_met=fbact_met, fbact_str=fbact_str)


def root_profile_layer_weights(z_soil, rprof, *, nslm: int | None = None):
    """Compute Fortran root-profile weights over soil layers.

    Fortran provenance: ``src_stomate/stomate_litter.f90``,
    ``littercalc_leak`` lines 2174-2190 and 2225-2240. ``z_soil`` follows the
    Fortran boundary convention with length ``nslm + 1`` and ``z_soil[0] = 0``.
    """

    z_soil = jnp.asarray(z_soil)
    rprof = jnp.asarray(rprof)
    if z_soil.ndim != 1:
        raise ValueError("z_soil must be one-dimensional")
    if nslm is None:
        nslm = z_soil.shape[0] - 1
    if z_soil.shape[0] < nslm + 1:
        raise ValueError("z_soil length must be at least nslm + 1")
    rpc = 1.0 / (1.0 - jnp.exp(-z_soil[nslm] / rprof))
    return rpc[..., None] * (
        jnp.exp(-z_soil[:nslm] / rprof[..., None])
        - jnp.exp(-z_soil[1 : nslm + 1] / rprof[..., None])
    )


def littercalc_litter_increments(
    bm_to_litter,
    turnover,
    rprof,
    z_soil,
    dead_leaves,
    *,
    litterfrac=None,
    lc=DEFAULT_LC,
    nslm: int | None = None,
    ndeep: int | None = None,
) -> LitterIncrementResult:
    """Compute section-2 litter additions before pools are updated.

    Fortran provenance: ``src_stomate/stomate_litter.f90``,
    ``littercalc_leak`` lines 2138-2243. Aboveground receives leaf,
    above-wood, fruit, reserve, and AGR above parts; belowground receives
    sap-below, heart-below, and root through the root profile.
    """

    bm_to_litter = jnp.asarray(bm_to_litter)
    turnover = jnp.asarray(turnover)
    rprof = jnp.asarray(rprof)
    dead_leaves = jnp.asarray(dead_leaves)
    lc = jnp.asarray(lc)
    if bm_to_litter.shape != turnover.shape:
        raise ValueError("bm_to_litter and turnover must have the same shape")
    if bm_to_litter.ndim != 4 or bm_to_litter.shape[2] != NPARTS:
        raise ValueError("bm_to_litter must have shape (npts, nvm, nparts, nelements)")
    npts, nvm, _, nelements = bm_to_litter.shape
    if nslm is None:
        nslm = jnp.asarray(z_soil).shape[0] - 1
    if ndeep is None:
        ndeep = nslm
    if ndeep < nslm:
        raise ValueError("ndeep must be greater than or equal to nslm")
    if litterfrac is None:
        litterfrac = littercalc_litter_fractions(lc=lc)
    litterfrac = jnp.asarray(litterfrac)
    if litterfrac.shape != (NPARTS, NLITT):
        raise ValueError(f"litterfrac must have shape ({NPARTS}, {NLITT})")
    if dead_leaves.shape != (npts, nvm, NLITT):
        raise ValueError("dead_leaves must have shape (npts, nvm, nlitt)")

    flux = bm_to_litter + turnover
    above_parts = jnp.asarray(
        [ILEAF, ISAPABOVE, IHEARTABOVE, IFRUIT, ICARBRES, IAGRSAPST, IAGRSAPPN, IAGRHRTST, IAGRHRTPN]
    )
    below_parts = jnp.asarray([ISAPBELOW, IHEARTBELOW, IROOT])
    weights = root_profile_layer_weights(z_soil, rprof, nslm=nslm)

    flux_carbon = flux[:, :, :, ICARBON : ICARBON + 1]
    above_carbon = jnp.einsum("pk,njpe->njke", litterfrac[above_parts, :], flux_carbon[:, :, above_parts, :])
    above_by_litt = jnp.zeros((npts, nvm, NLITT, nelements), dtype=above_carbon.dtype)
    above_by_litt = above_by_litt.at[:, :, :, ICARBON : ICARBON + 1].set(above_carbon)
    below_part_flux = jnp.einsum("pk,njpe->njke", litterfrac[below_parts, :], flux_carbon[:, :, below_parts, :])
    below_by_litt_nslm = below_part_flux[:, :, :, None, :] * weights[:, :, None, :, None]
    if ndeep > nslm:
        below_pad = jnp.zeros((npts, nvm, NLITT, ndeep - nslm, nelements), dtype=below_by_litt_nslm.dtype)
        below_by_litt = jnp.concatenate([below_by_litt_nslm, below_pad], axis=3)
    else:
        below_by_litt = below_by_litt_nslm

    active_pft = jnp.arange(nvm) > 0
    above_by_litt = jnp.where(active_pft[None, :, None, None], above_by_litt, 0.0)
    below_by_litt = jnp.where(active_pft[None, :, None, None, None], below_by_litt, 0.0)

    litter_inc_above = jnp.transpose(above_by_litt, (0, 2, 1, 3))
    litter_inc_below = jnp.transpose(below_by_litt, (0, 2, 1, 3, 4))

    dead_leaf_add = flux[:, :, ILEAF, ICARBON][:, :, None] * litterfrac[ILEAF, :][None, None, :]
    dead_leaves_updated = dead_leaves + jnp.where(active_pft[None, :, None], dead_leaf_add, 0.0)

    lignin_above = jnp.sum(lc[above_parts][None, None, :, None] * flux[:, :, above_parts, :], axis=(2, 3))
    lignin_below_part = jnp.sum(lc[below_parts][None, None, :, None] * flux[:, :, below_parts, :], axis=(2, 3))
    lignin_above = jnp.where(active_pft[None, :], lignin_above, 0.0)
    lignin_below_nslm = lignin_below_part[:, :, None] * weights
    if ndeep > nslm:
        lignin_pad = jnp.zeros((npts, nvm, ndeep - nslm), dtype=lignin_below_nslm.dtype)
        lignin_below = jnp.concatenate([lignin_below_nslm, lignin_pad], axis=2)
    else:
        lignin_below = lignin_below_nslm
    lignin_below = jnp.where(active_pft[None, :, None], lignin_below, 0.0)

    return LitterIncrementResult(
        litter_inc_above=litter_inc_above,
        litter_inc_below=litter_inc_below,
        litter_inc_pft_above=above_by_litt,
        litter_inc_pft_below=below_by_litt,
        dead_leaves=dead_leaves_updated,
        lignin_struc_inc_above=lignin_above,
        lignin_struc_inc_below=lignin_below,
    )


def update_lignin_fraction(old_lignin, old_structural_litter, lignin_increment, structural_increment, *, min_stomate=0.0):
    """Update structural litter lignin fraction after adding fresh litter.

    Fortran provenance: ``src_stomate/stomate_litter.f90``,
    ``littercalc_leak`` lines 2353-2391. The lignin increment is first capped
    by the structural litter increment, then mixed with the previous stock.
    """

    old_lignin = jnp.asarray(old_lignin)
    old_structural_litter = jnp.asarray(old_structural_litter)
    lignin_increment = jnp.minimum(jnp.asarray(lignin_increment), jnp.asarray(structural_increment))
    new_structural_litter = old_structural_litter + structural_increment
    active = new_structural_litter > min_stomate
    safe_structural_litter = jnp.where(active, new_structural_litter, 1.0)
    return jnp.where(
        active,
        (old_lignin * old_structural_litter + lignin_increment) / safe_structural_litter,
        0.0,
    )


def littercalc_apply_pool_increments(
    litter_above,
    litter_below,
    lignin_struc_above,
    lignin_struc_below,
    litterpart,
    increments: LitterIncrementResult,
    *,
    min_stomate=0.0,
) -> LitterPoolUpdateResult:
    """Apply litter additions, lignin mixing, and aboveground litterpart update.

    Fortran provenance: ``src_stomate/stomate_litter.f90``,
    ``littercalc_leak`` lines 2126-2136, 2322-2325, 2353-2405.
    """

    litter_above = jnp.asarray(litter_above)
    litter_below = jnp.asarray(litter_below)
    lignin_struc_above = jnp.asarray(lignin_struc_above)
    lignin_struc_below = jnp.asarray(lignin_struc_below)
    litterpart = jnp.asarray(litterpart)
    old_struc_above = litter_above[:, ISTRUCTURAL, :, ICARBON]
    old_struc_below = litter_below[:, ISTRUCTURAL, :, :, ICARBON]
    litter_above_new = litter_above + increments.litter_inc_above
    litter_below_new = litter_below + increments.litter_inc_below

    lignin_above_new = update_lignin_fraction(
        lignin_struc_above,
        old_struc_above,
        increments.lignin_struc_inc_above,
        increments.litter_inc_above[:, ISTRUCTURAL, :, ICARBON],
        min_stomate=min_stomate,
    )
    lignin_below_new = update_lignin_fraction(
        lignin_struc_below,
        old_struc_below,
        increments.lignin_struc_inc_below,
        increments.litter_inc_below[:, ISTRUCTURAL, :, :, ICARBON],
        min_stomate=min_stomate,
    )

    numerator = jnp.swapaxes(increments.litter_inc_pft_above[:, :, :, ICARBON], 1, 2)
    denom = litter_above_new[:, :, :, ICARBON]
    active_litter = denom > min_stomate
    safe_denom = jnp.where(active_litter, denom, 1.0)
    litterpart_new = jnp.where(
        active_litter,
        (
            jnp.swapaxes(litterpart, 1, 2)
            * litter_above[:, :, :, ICARBON]
            + numerator
        )
        / safe_denom,
        0.0,
    )
    litterpart_new = jnp.swapaxes(litterpart_new, 1, 2)

    return LitterPoolUpdateResult(
        litter_above=litter_above_new,
        litter_below=litter_below_new,
        lignin_struc_above=lignin_above_new,
        lignin_struc_below=lignin_below_new,
        litterpart=litterpart_new,
        increments=increments,
    )


def littercalc_add_aboveground_fuel(
    fuel_1hr,
    fuel_10hr,
    fuel_100hr,
    fuel_1000hr,
    bm_to_litter,
    turnover,
    *,
    litterfrac=None,
) -> FuelPools:
    """Add aboveground litter inputs to SPITFIRE fuel-size classes.

    Fortran provenance: ``src_stomate/stomate_litter.f90``,
    ``littercalc_leak`` lines 2245-2317. This fuel path uses saved
    ``litterfrac`` rather than the per-pixel ``litterfrac_pxl`` and excludes
    belowground sapwood, heartwood, and root inputs.
    """

    fuel_1hr = jnp.asarray(fuel_1hr)
    fuel_10hr = jnp.asarray(fuel_10hr)
    fuel_100hr = jnp.asarray(fuel_100hr)
    fuel_1000hr = jnp.asarray(fuel_1000hr)
    bm_to_litter = jnp.asarray(bm_to_litter)
    turnover = jnp.asarray(turnover)
    if bm_to_litter.shape != turnover.shape:
        raise ValueError("bm_to_litter and turnover must have the same shape")
    if fuel_1hr.shape != fuel_10hr.shape or fuel_1hr.shape != fuel_100hr.shape or fuel_1hr.shape != fuel_1000hr.shape:
        raise ValueError("all fuel pools must have the same shape")
    if fuel_1hr.shape[:3] != (bm_to_litter.shape[0], bm_to_litter.shape[1], NLITT):
        raise ValueError("fuel pools must have shape (npts, nvm, nlitt, nelements)")
    if litterfrac is None:
        litterfrac = littercalc_litter_fractions()
    litterfrac = jnp.asarray(litterfrac)
    flux = bm_to_litter + turnover
    nvm = bm_to_litter.shape[1]
    active_pft = (jnp.arange(nvm) > 0)[None, :, None, None]

    flux_carbon = flux[:, :, :, ICARBON : ICARBON + 1]
    leaf_fruit = jnp.einsum(
        "pk,njpe->njke",
        litterfrac[jnp.asarray([ILEAF, IFRUIT]), :],
        flux_carbon[:, :, [ILEAF, IFRUIT], :],
    )
    above_wood = jnp.einsum(
        "pk,njpe->njke",
        litterfrac[jnp.asarray([ISAPABOVE, IHEARTABOVE, ICARBRES, IAGRSAPST, IAGRSAPPN, IAGRHRTST, IAGRHRTPN]), :],
        flux_carbon[:, :, [ISAPABOVE, IHEARTABOVE, ICARBRES, IAGRSAPST, IAGRSAPPN, IAGRHRTST, IAGRHRTPN], :],
    )
    fuel_1hr_new = fuel_1hr.at[:, :, :, ICARBON : ICARBON + 1].add(
        jnp.where(active_pft, leaf_fruit + 0.045 * above_wood, 0.0)
    )
    fuel_10hr_new = fuel_10hr.at[:, :, :, ICARBON : ICARBON + 1].add(jnp.where(active_pft, 0.075 * above_wood, 0.0))
    fuel_100hr_new = fuel_100hr.at[:, :, :, ICARBON : ICARBON + 1].add(jnp.where(active_pft, 0.210 * above_wood, 0.0))
    fuel_1000hr_new = fuel_1000hr.at[:, :, :, ICARBON : ICARBON + 1].add(jnp.where(active_pft, 0.670 * above_wood, 0.0))
    fuel_total = fuel_1hr_new + fuel_10hr_new + fuel_100hr_new + fuel_1000hr_new
    return FuelPools(
        fuel_1hr=fuel_1hr_new,
        fuel_10hr=fuel_10hr_new,
        fuel_100hr=fuel_100hr_new,
        fuel_1000hr=fuel_1000hr_new,
        fuel_total=fuel_total,
    )


def sync_aboveground_fuel_after_decomposition(
    fuel_1hr,
    fuel_10hr,
    fuel_100hr,
    fuel_1000hr,
    litter_above_pool,
    qd,
    *,
    min_stomate=0.0,
) -> FuelPools:
    """Synchronize one aboveground fuel pool with post-decay litter stock.

    Fortran provenance: ``src_stomate/stomate_litter.f90``,
    ``littercalc_leak`` structural block lines 2601-2639 and metabolic block
    lines 2678-2715. The function operates on one litter-pool slice with
    shape ``(npts, nvm, nelements)``.
    """

    fuel_1hr = jnp.asarray(fuel_1hr)
    fuel_10hr = jnp.asarray(fuel_10hr)
    fuel_100hr = jnp.asarray(fuel_100hr)
    fuel_1000hr = jnp.asarray(fuel_1000hr)
    litter_above_pool = jnp.asarray(litter_above_pool)
    qd = jnp.asarray(qd)
    fuel_total = fuel_1hr + fuel_10hr + fuel_100hr + fuel_1000hr
    qd_carbon = qd[:, :, ICARBON]
    positive = fuel_total[:, :, ICARBON] > min_stomate
    safe_fuel_total = jnp.where(positive, fuel_total[:, :, ICARBON], 1.0)
    ratio_1hr_c = jnp.where(positive, fuel_1hr[:, :, ICARBON] / safe_fuel_total, 0.0)
    ratio_10hr_c = jnp.where(positive, fuel_10hr[:, :, ICARBON] / safe_fuel_total, 0.0)
    ratio_100hr_c = jnp.where(positive, fuel_100hr[:, :, ICARBON] / safe_fuel_total, 0.0)
    ratio_1000hr_c = jnp.where(positive, fuel_1000hr[:, :, ICARBON] / safe_fuel_total, 0.0)

    fuel_1hr_after_qd = fuel_1hr.at[:, :, ICARBON].add(-qd_carbon * ratio_1hr_c)
    fuel_10hr_after_qd = fuel_10hr.at[:, :, ICARBON].add(-qd_carbon * ratio_10hr_c)
    fuel_100hr_after_qd = fuel_100hr.at[:, :, ICARBON].add(-qd_carbon * ratio_100hr_c)
    fuel_1000hr_after_qd = fuel_1000hr.at[:, :, ICARBON].add(-qd_carbon * ratio_1000hr_c)
    total_after_qd = fuel_1hr_after_qd + fuel_10hr_after_qd + fuel_100hr_after_qd + fuel_1000hr_after_qd

    diff_frac = litter_above_pool[:, :, ICARBON] - total_after_qd[:, :, ICARBON]
    rescale = (jnp.abs(diff_frac) > min_stomate) & (total_after_qd[:, :, ICARBON] > min_stomate)
    safe_total_after_qd = jnp.where(rescale, total_after_qd[:, :, ICARBON], 1.0)
    scale = jnp.where(rescale, 1.0 + diff_frac / safe_total_after_qd, 1.0)
    fuel_1hr_new = fuel_1hr_after_qd.at[:, :, ICARBON].set(fuel_1hr_after_qd[:, :, ICARBON] * scale)
    fuel_10hr_new = fuel_10hr_after_qd.at[:, :, ICARBON].set(fuel_10hr_after_qd[:, :, ICARBON] * scale)
    fuel_100hr_new = fuel_100hr_after_qd.at[:, :, ICARBON].set(fuel_100hr_after_qd[:, :, ICARBON] * scale)
    fuel_1000hr_new = fuel_1000hr_after_qd.at[:, :, ICARBON].set(fuel_1000hr_after_qd[:, :, ICARBON] * scale)
    return FuelPools(
        fuel_1hr=fuel_1hr_new,
        fuel_10hr=fuel_10hr_new,
        fuel_100hr=fuel_100hr_new,
        fuel_1000hr=fuel_1000hr_new,
        fuel_total=fuel_1hr_new + fuel_10hr_new + fuel_100hr_new + fuel_1000hr_new,
    )


def deadleaf_cover_from_litter(dead_leaves, veget_max, sla_calc):
    """Compute dead-leaf soil cover from dead leaves and dynamic SLA.

    Fortran provenance: ``src_stomate/stomate_litter.f90``, subroutine
    ``deadleaf``, signature and inputs lines 1314-1335, dead LAI summation
    lines 1345-1356, and cover formula line 1360.
    """

    dead_leaves = jnp.asarray(dead_leaves)
    veget_max = jnp.asarray(veget_max)
    sla_calc = jnp.asarray(sla_calc)
    if dead_leaves.ndim != 3 or dead_leaves.shape[2] != NLITT:
        raise ValueError("dead_leaves must have shape (npts, nvm, nlitt)")
    if veget_max.shape != dead_leaves.shape[:2] or sla_calc.shape != dead_leaves.shape[:2]:
        raise ValueError("veget_max and sla_calc must have shape (npts, nvm)")
    dead_lai = jnp.sum((dead_leaves[:, :, IMETABOLIC] + dead_leaves[:, :, ISTRUCTURAL]) * sla_calc * veget_max, axis=1)
    return 1.0 - jnp.exp(-0.5 * dead_lai)


def litter_control_moisture(moist_in, *, moist_coeff=(1.1, 2.4, 0.29), moistcont_min=0.25):
    """Compute the ordinary ORCHIDEE litter moisture-control factor.

    Fortran provenance: ``src_stomate/stomate_litter.f90``, function
    ``control_moist_func``, lines 1396-1418. Constant defaults are from
    ``src_parameters/constantes_var.f90`` lines 1236-1239.
    """

    moist_in = jnp.asarray(moist_in)
    coeff = jnp.asarray(moist_coeff)
    result = -coeff[0] * moist_in * moist_in + coeff[1] * moist_in - coeff[2]
    return jnp.maximum(moistcont_min, jnp.minimum(1.0, result))


def litter_control_moisture_moyano(moist_in, zz_coef_deep, bulk_dens, clay, carbon_32l, veget_max, *, nummoist=120):
    """Compute the Moyano moisture-control branch used by ``littercalc_leak``.

    Fortran provenance: ``src_stomate/stomate_litter.f90``, function
    ``control_moist_func_moyano``, lines 2904-3037. Inputs preserve the
    Fortran carbon axis order ``carbon_32l[npts,ncarb,nvm,ndeep]``.
    """

    moist_in = jnp.asarray(moist_in)
    zz_coef_deep = jnp.asarray(zz_coef_deep)
    bulk_dens = jnp.asarray(bulk_dens)
    clay = jnp.asarray(clay)
    carbon_32l = jnp.asarray(carbon_32l)
    veget_max = jnp.asarray(veget_max)
    if carbon_32l.ndim != 4 or carbon_32l.shape[1] != NCARB:
        raise ValueError("carbon_32l must have shape (npts, ncarb, nvm, ndeep)")
    npts, _, nvm, ndeep = carbon_32l.shape
    if zz_coef_deep.shape[0] != ndeep:
        raise ValueError("zz_coef_deep length must match carbon_32l ndeep")
    if bulk_dens.shape[0] != npts or clay.shape[0] != npts or moist_in.shape[0] != npts:
        raise ValueError("moist_in, bulk_dens, and clay must have shape (npts,)")
    if veget_max.shape != (npts, nvm):
        raise ValueError("veget_max must have shape (npts, nvm)")

    zf = jnp.concatenate([jnp.asarray([0.0], dtype=zz_coef_deep.dtype), zz_coef_deep])
    layer_fraction = (zf[1:] - zf[:-1]) / zf[-1]
    soil_carbon = carbon_32l[:, IACTIVE, :, :] + carbon_32l[:, ISLOW, :, :] + carbon_32l[:, IPASSIVE, :, :]
    total_soc = jnp.sum(soil_carbon * veget_max[:, :, None] * layer_fraction[None, None, :], axis=(1, 2)) / bulk_dens

    k = jnp.arange(1, nummoist + 1, dtype=moist_in.dtype)
    candidate_moist = k / 100.0
    prsr = (
        1.11066
        - 0.83344 * candidate_moist[None, :]
        + 1.48095 * candidate_moist[None, :] ** 2
        - 1.02959 * candidate_moist[None, :] ** 3
        + 0.07995 * clay[:, None]
        + 1.27892 * total_soc[:, None]
    )
    sr = jnp.cumprod(prsr, axis=1)
    moistfunc = sr / jnp.max(sr, axis=1, keepdims=True)
    max_index = jnp.argmax(moistfunc, axis=1)
    before_or_at_peak = jnp.arange(nummoist)[None, :] <= max_index[:, None]
    peak_values = jnp.where(before_or_at_peak, moistfunc, jnp.inf)
    peak_min = jnp.min(peak_values, axis=1, keepdims=True)
    shifted = jnp.where(before_or_at_peak, moistfunc - peak_min, moistfunc)
    shifted_peak = jnp.where(before_or_at_peak, shifted, -jnp.inf)
    peak_max = jnp.max(shifted_peak, axis=1, keepdims=True)
    rescaled = jnp.where(before_or_at_peak, shifted / peak_max, moistfunc)

    index_fortran = jnp.rint(moist_in * 100.0).astype(jnp.int32)
    clipped_index = jnp.clip(index_fortran, 1, nummoist) - 1
    gathered = jnp.take_along_axis(rescaled, clipped_index[:, None], axis=1)[:, 0]
    return jnp.where(index_fortran > 0, gathered, 0.0)


def litter_control_temperature(
    temp_in,
    frozen_respiration_func: int,
    *,
    soil_q10=0.69,
    tsoil_ref=30.0,
    q10=10.0,
):
    """Compute the ORCHIDEE litter temperature-control factor.

    Fortran provenance: ``src_stomate/stomate_litter.f90``, function
    ``control_temp_func``, lines 1451-1511. Constant defaults are from
    ``src_parameters/constantes_var.f90`` lines 1190 and 1230-1233.
    """

    temp = jnp.asarray(temp_in)
    if int(frozen_respiration_func) == 0:
        result = jnp.exp(soil_q10 * (temp - (ZERO_CELSIUS + tsoil_ref)) / q10)
    elif int(frozen_respiration_func) == 1:
        normal = jnp.exp(0.69 * (temp - (ZERO_CELSIUS + 30.0)) / 10.0)
        at_zero = jnp.exp(0.69 * (ZERO_CELSIUS - (ZERO_CELSIUS + 30.0)) / 10.0)
        result = jnp.where(temp > ZERO_CELSIUS, normal, jnp.where(temp > ZERO_CELSIUS - 1.0, (temp - (ZERO_CELSIUS - 1.0)) * at_zero, 0.0))
    elif int(frozen_respiration_func) == 2:
        normal = jnp.exp(0.69 * (temp - (ZERO_CELSIUS + 30.0)) / 10.0)
        at_zero = jnp.exp(0.69 * (ZERO_CELSIUS - (ZERO_CELSIUS + 30.0)) / 10.0)
        result = jnp.where(temp > ZERO_CELSIUS, normal, jnp.where(temp > ZERO_CELSIUS - 3.0, ((temp - (ZERO_CELSIUS - 3.0)) / 3.0) * at_zero, 0.0))
    elif int(frozen_respiration_func) == 3:
        normal = jnp.exp(0.69 * (temp - (ZERO_CELSIUS + 30.0)) / 10.0)
        frozen = jnp.exp(4.605 * (temp - ZERO_CELSIUS) / 10.0) * jnp.exp(0.69 * (-30.0) / 10.0)
        result = jnp.where(temp > ZERO_CELSIUS, normal, frozen)
    elif int(frozen_respiration_func) == 4:
        normal = jnp.exp(0.69 * (temp - (ZERO_CELSIUS + 30.0)) / 10.0)
        frozen = jnp.exp(6.908 * (temp - ZERO_CELSIUS) / 10.0) * jnp.exp(0.69 * (-30.0) / 10.0)
        result = jnp.where(temp > ZERO_CELSIUS, normal, frozen)
    else:
        raise ValueError(f"frozen_respiration_func not in Fortran control_temp_func list: {frozen_respiration_func}")
    return jnp.maximum(jnp.minimum(1.0, result), 0.0)


def littercalc_aboveground_controls(
    tsurf,
    soil_mc,
    z_soil,
    pref_soil_veg,
    *,
    frozen_respiration_func: int,
    moist_func_moyano: bool = False,
    zz_coef_deep=None,
    bulk_dens=None,
    clay=None,
    carbon_32l=None,
    veget_max=None,
):
    """Prepare ordinary aboveground controls before ``littercalc_leak`` decay.

    Fortran provenance: ``src_stomate/stomate_litter.f90``,
    ``littercalc_leak`` lines 2408-2414 for temperature and lines 2482-2497
    for moisture control, including the Moyano branch at lines 2491-2493 when
    its source inputs are supplied.
    """

    tsurf = jnp.asarray(tsurf)
    soil_mc = jnp.asarray(soil_mc)
    z_soil = jnp.asarray(z_soil)
    pref_soil_veg = jnp.asarray(pref_soil_veg, dtype=jnp.int32)
    if soil_mc.ndim != 3:
        raise ValueError("soil_mc must have shape (npts, nslm, nstm)")
    if z_soil.shape[0] < 5:
        raise ValueError("z_soil must include boundaries 0..4 for aboveground moisture control")
    if pref_soil_veg.ndim != 1:
        raise ValueError("pref_soil_veg must have shape (nvm,) with zero-based soil tile indices")
    npts = soil_mc.shape[0]
    nvm = pref_soil_veg.shape[0]
    tile = pref_soil_veg
    soil_mc_by_pft = soil_mc[:, :, tile]
    weights = jnp.asarray(
        [
            z_soil[1] / z_soil[4],
            (z_soil[2] - z_soil[1]) / z_soil[4],
            (z_soil[3] - z_soil[2]) / z_soil[4],
            (z_soil[4] - z_soil[3]) / z_soil[4],
        ]
    )
    soilhum_decomp = jnp.sum(soil_mc_by_pft[:, :4, :] * weights[None, :, None], axis=1)
    if moist_func_moyano:
        missing = [
            name
            for name, value in (
                ("zz_coef_deep", zz_coef_deep),
                ("bulk_dens", bulk_dens),
                ("clay", clay),
                ("carbon_32l", carbon_32l),
                ("veget_max", veget_max),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"MOIST_FUNC_MOYANO requires {', '.join(missing)}")
        control_moist_above = jnp.swapaxes(
            jnp.stack(
                [
                    litter_control_moisture_moyano(
                        soilhum_decomp[:, m],
                        zz_coef_deep,
                        bulk_dens,
                        clay,
                        carbon_32l,
                        veget_max,
                    )
                    for m in range(nvm)
                ],
                axis=0,
            ),
            0,
            1,
        )
    else:
        control_moist_above = litter_control_moisture(soilhum_decomp)
    temp = litter_control_temperature(tsurf, frozen_respiration_func)
    control_temp_above = jnp.repeat(temp[:, None], NLITT, axis=1)
    if control_temp_above.shape != (npts, NLITT) or control_moist_above.shape != (npts, nvm):
        raise ValueError("computed aboveground controls have inconsistent shape")
    return LitterAbovegroundControls(
        control_temp_above=control_temp_above,
        control_moist_above=control_moist_above,
        soilhum_decomp=soilhum_decomp,
        soil_mc_top_by_pft=soil_mc_by_pft[:, 0, :],
    )


def decompose_aboveground_litter_leak(
    litter_above,
    dead_leaves,
    lignin_struc_above,
    control_temp_above,
    control_moist_above,
    soil_mc_top_by_pft,
    poor_soils,
    flood_frac,
    *,
    dt_days,
    frac_soil=None,
    litter_tau=None,
    litter_struct_coef=3.0,
    ndeep=32,
):
    """Decompose aboveground structural and metabolic litter.

    Fortran provenance: ``src_stomate/stomate_litter.f90``,
    ``littercalc_leak`` lines 2563-2599 for structural decay and lines
    2652-2676 for metabolic decay. This helper returns flux arrays before the
    SPITFIRE fuel-size synchronization in lines 2600-2649 and 2678-2725.
    """

    litter_above = jnp.asarray(litter_above)
    dead_leaves = jnp.asarray(dead_leaves)
    lignin_struc_above = jnp.asarray(lignin_struc_above)
    control_temp_above = jnp.asarray(control_temp_above)
    control_moist_above = jnp.asarray(control_moist_above)
    soil_mc_top_by_pft = jnp.asarray(soil_mc_top_by_pft)
    poor_soils = jnp.asarray(poor_soils)
    flood_frac = jnp.asarray(flood_frac)
    if litter_above.ndim != 4 or litter_above.shape[1] != NLITT:
        raise ValueError("litter_above must have shape (npts, nlitt, nvm, nelements)")
    npts, _, nvm, nelements = litter_above.shape
    if dead_leaves.shape != (npts, nvm, NLITT):
        raise ValueError("dead_leaves must have shape (npts, nvm, nlitt)")
    if frac_soil is None:
        frac_soil = littercalc_frac_soil()
    if litter_tau is None:
        litter_tau, _ = littercalc_turnover_times()
    frac_soil = jnp.asarray(frac_soil)
    litter_tau = jnp.asarray(litter_tau)

    active = soil_mc_top_by_pft > 0.0
    flood = flood_frac[:, None]
    dry = 1.0 - flood
    poor = 1.0 + poor_soils[:, None]
    lignin = lignin_struc_above

    fd_struct = jnp.where(
        active,
        dt_days
        / (litter_tau[ISTRUCTURAL] * poor)
        * control_temp_above[:, ISTRUCTURAL, None]
        * control_moist_above
        * jnp.exp(-litter_struct_coef * lignin),
        0.0,
    )
    qd_struct = litter_above[:, ISTRUCTURAL, :, :] * fd_struct[:, :, None] * dry[:, :, None]
    qd_flood_struct = litter_above[:, ISTRUCTURAL, :, :] * fd_struct[:, :, None] * flood[:, :, None] / 3.0

    litter_after_struct = litter_above.at[:, ISTRUCTURAL, :, :].add(-(qd_struct + qd_flood_struct))
    dead_factor_struct = 1.0 - fd_struct * dry - fd_struct * flood / 3.0
    dead_after_struct = dead_leaves.at[:, :, ISTRUCTURAL].multiply(dead_factor_struct)

    fd_met = jnp.where(
        active,
        dt_days / (litter_tau[IMETABOLIC] * poor) * control_temp_above[:, IMETABOLIC, None] * control_moist_above,
        0.0,
    )
    qd_met = litter_after_struct[:, IMETABOLIC, :, :] * fd_met[:, :, None] * dry[:, :, None]
    qd_flood_met = litter_after_struct[:, IMETABOLIC, :, :] * fd_met[:, :, None] * flood[:, :, None] / 3.0

    litter_updated = litter_after_struct.at[:, IMETABOLIC, :, :].add(-(qd_met + qd_flood_met))
    dead_factor_met = 1.0 - fd_met * dry - fd_met * flood / 3.0
    dead_updated = dead_after_struct.at[:, :, IMETABOLIC].multiply(dead_factor_met)

    resp_hetero_litter = jnp.zeros((npts, nvm, NLEVS), dtype=litter_above.dtype)
    resp_hetero_flood = jnp.zeros((npts, nvm), dtype=litter_above.dtype)
    soilcarbon_input_doc = jnp.zeros((npts, nvm, ndeep, NPOOL, nelements), dtype=litter_above.dtype)
    floodcarbon_input = jnp.zeros((npts, nvm, NPOOL, nelements), dtype=litter_above.dtype)

    qd_struct_c = qd_struct[:, :, ICARBON]
    qd_flood_struct_c = qd_flood_struct[:, :, ICARBON]
    resp_struct = (
        (1.0 - frac_soil[ISTRUCTURAL, IACTIVE, IABOVE]) * qd_struct_c * (1.0 - lignin)
        + (1.0 - frac_soil[ISTRUCTURAL, ISLOW, IABOVE]) * qd_struct_c * lignin
    ) / dt_days
    resp_flood_struct = (
        (1.0 - frac_soil[ISTRUCTURAL, IACTIVE, IABOVE]) * qd_flood_struct_c * (1.0 - lignin)
        + (1.0 - frac_soil[ISTRUCTURAL, ISLOW, IABOVE]) * qd_flood_struct_c * lignin
    ) / dt_days
    resp_hetero_litter = resp_hetero_litter.at[:, :, IABOVE].add(resp_struct)
    resp_hetero_flood = resp_hetero_flood + resp_flood_struct
    soil_doc_struct = (
        frac_soil[ISTRUCTURAL, IACTIVE, IABOVE] * (1.0 - lignin[:, :, None]) * qd_struct
        + frac_soil[ISTRUCTURAL, ISLOW, IABOVE] * lignin[:, :, None] * qd_struct
    ) / dt_days
    soilcarbon_input_doc = soilcarbon_input_doc.at[:, :, 0, ISTRABO, :].set(soil_doc_struct)
    floodcarbon_input = floodcarbon_input.at[:, :, IACT, :].set(
        frac_soil[ISTRUCTURAL, IACTIVE, IABOVE] * (1.0 - lignin[:, :, None]) * qd_flood_struct / dt_days
    )
    floodcarbon_input = floodcarbon_input.at[:, :, ISLO, :].set(
        frac_soil[ISTRUCTURAL, ISLOW, IABOVE] * lignin[:, :, None] * qd_flood_struct / dt_days
    )

    qd_met_c = qd_met[:, :, ICARBON]
    qd_flood_met_c = qd_flood_met[:, :, ICARBON]
    resp_hetero_litter = resp_hetero_litter.at[:, :, IABOVE].add(
        (1.0 - frac_soil[IMETABOLIC, IACTIVE, IABOVE]) * qd_met_c / dt_days
    )
    resp_hetero_flood = resp_hetero_flood + (1.0 - frac_soil[IMETABOLIC, IACTIVE, IABOVE]) * qd_flood_met_c / dt_days
    soilcarbon_input_doc = soilcarbon_input_doc.at[:, :, 0, IMETABO, :].set(
        frac_soil[IMETABOLIC, IACTIVE, IABOVE] * qd_met / dt_days
    )
    floodcarbon_input = floodcarbon_input.at[:, :, IACT, :].add(
        frac_soil[IMETABOLIC, IACTIVE, IABOVE] * qd_flood_met / dt_days
    )

    return LitterAbovegroundDecompositionResult(
        litter_above=litter_updated,
        dead_leaves=dead_updated,
        resp_hetero_litter=resp_hetero_litter,
        resp_hetero_flood=resp_hetero_flood,
        soilcarbon_input_doc=soilcarbon_input_doc,
        floodcarbon_input=floodcarbon_input,
        qd_structural=qd_struct,
        qd_metabolic=qd_met,
        qd_flood_structural=qd_flood_struct,
        qd_flood_metabolic=qd_flood_met,
    )


def decompose_belowground_litter_leak(
    litter_below,
    lignin_struc_below,
    fbact_met,
    fbact_str,
    flood_frac,
    *,
    dt_days,
    frac_soil=None,
    sro_bottom=5,
):
    """Decompose belowground structural and metabolic litter.

    Fortran provenance: ``src_stomate/stomate_litter.f90``,
    ``littercalc_leak`` lines 2731-2768 for structural belowground decay and
    lines 2770-2794 for metabolic belowground decay. The ``sro_bottom`` branch
    follows Fortran's 1-based ``l .GT. sro_bottom`` condition.
    """

    litter_below = jnp.asarray(litter_below)
    lignin_struc_below = jnp.asarray(lignin_struc_below)
    fbact_met = jnp.asarray(fbact_met)
    fbact_str = jnp.asarray(fbact_str)
    flood_frac = jnp.asarray(flood_frac)
    if litter_below.ndim != 5 or litter_below.shape[1] != NLITT:
        raise ValueError("litter_below must have shape (npts, nlitt, nvm, ndeep, nelements)")
    npts, _, nvm, ndeep, nelements = litter_below.shape
    if lignin_struc_below.shape != (npts, nvm, ndeep):
        raise ValueError("lignin_struc_below must have shape (npts, nvm, ndeep)")
    if fbact_met.shape != (npts, ndeep, nvm) or fbact_str.shape != (npts, ndeep, nvm):
        raise ValueError("fbact_met and fbact_str must have shape (npts, ndeep, nvm)")
    if frac_soil is None:
        frac_soil = littercalc_frac_soil()
    frac_soil = jnp.asarray(frac_soil)

    flood = flood_frac[:, None, None]
    dry = 1.0 - flood
    lignin = lignin_struc_below
    fbact_str_pft_layer = jnp.swapaxes(fbact_str, 1, 2)
    fbact_met_pft_layer = jnp.swapaxes(fbact_met, 1, 2)

    qd_struct = litter_below[:, ISTRUCTURAL, :, :, :] * fbact_str_pft_layer[:, :, :, None] * dry[:, :, :, None]
    qd_flood_struct = (
        litter_below[:, ISTRUCTURAL, :, :, :] * fbact_str_pft_layer[:, :, :, None] * flood[:, :, :, None] / 3.0
    )
    litter_after_struct = litter_below.at[:, ISTRUCTURAL, :, :, :].add(-(qd_struct + qd_flood_struct))

    qd_met = litter_after_struct[:, IMETABOLIC, :, :, :] * fbact_met_pft_layer[:, :, :, None] * dry[:, :, :, None]
    qd_flood_met = (
        litter_after_struct[:, IMETABOLIC, :, :, :] * fbact_met_pft_layer[:, :, :, None] * flood[:, :, :, None] / 3.0
    )
    litter_updated = litter_after_struct.at[:, IMETABOLIC, :, :, :].add(-(qd_met + qd_flood_met))

    resp_hetero_litter = jnp.zeros((npts, nvm, NLEVS), dtype=litter_below.dtype)
    resp_hetero_flood = jnp.zeros((npts, nvm), dtype=litter_below.dtype)
    soilcarbon_input_doc = jnp.zeros((npts, nvm, ndeep, NPOOL, nelements), dtype=litter_below.dtype)
    floodcarbon_input = jnp.zeros((npts, nvm, NPOOL, nelements), dtype=litter_below.dtype)

    qd_struct_c = qd_struct[:, :, :, ICARBON]
    qd_flood_struct_c = qd_flood_struct[:, :, :, ICARBON]
    resp_struct = (
        (1.0 - frac_soil[ISTRUCTURAL, IACTIVE, IBELOW]) * qd_struct_c * (1.0 - lignin)
        + (1.0 - frac_soil[ISTRUCTURAL, ISLOW, IBELOW]) * qd_struct_c * lignin
    ) / dt_days
    resp_flood_struct = (
        (1.0 - frac_soil[ISTRUCTURAL, IACTIVE, IBELOW]) * qd_flood_struct_c * (1.0 - lignin)
        + (1.0 - frac_soil[ISTRUCTURAL, ISLOW, IBELOW]) * qd_flood_struct_c * lignin
    ) / dt_days
    resp_hetero_litter = resp_hetero_litter.at[:, :, IBELOW].add(jnp.sum(resp_struct, axis=2))
    resp_hetero_flood = resp_hetero_flood + jnp.sum(resp_flood_struct, axis=2)

    deep_doc_mask = (jnp.arange(ndeep) >= sro_bottom)[None, None, :, None]
    struct_doc_qd = jnp.where(deep_doc_mask, qd_struct + qd_flood_struct, qd_struct)
    soil_doc_struct = (
        frac_soil[ISTRUCTURAL, IACTIVE, IBELOW] * (1.0 - lignin[:, :, :, None]) * struct_doc_qd
        + frac_soil[ISTRUCTURAL, ISLOW, IBELOW] * lignin[:, :, :, None] * struct_doc_qd
    ) / dt_days
    soilcarbon_input_doc = soilcarbon_input_doc.at[:, :, :, ISTRBEL, :].set(soil_doc_struct)
    shallow_doc_mask = ~deep_doc_mask
    floodcarbon_input = floodcarbon_input.at[:, :, IACT, :].add(
        jnp.sum(
            jnp.where(
                shallow_doc_mask,
                frac_soil[ISTRUCTURAL, IACTIVE, IBELOW]
                * (1.0 - lignin[:, :, :, None])
                * qd_flood_struct
                / dt_days,
                0.0,
            ),
            axis=2,
        )
    )
    floodcarbon_input = floodcarbon_input.at[:, :, ISLO, :].add(
        jnp.sum(
            jnp.where(
                shallow_doc_mask,
                frac_soil[ISTRUCTURAL, ISLOW, IBELOW] * lignin[:, :, :, None] * qd_flood_struct / dt_days,
                0.0,
            ),
            axis=2,
        )
    )

    qd_met_c = qd_met[:, :, :, ICARBON]
    qd_flood_met_c = qd_flood_met[:, :, :, ICARBON]
    resp_hetero_litter = resp_hetero_litter.at[:, :, IBELOW].add(
        jnp.sum((1.0 - frac_soil[IMETABOLIC, IACTIVE, IBELOW]) * qd_met_c / dt_days, axis=2)
    )
    resp_hetero_flood = resp_hetero_flood + jnp.sum(
        (1.0 - frac_soil[IMETABOLIC, IACTIVE, IBELOW]) * qd_flood_met_c / dt_days,
        axis=2,
    )
    met_doc_qd = jnp.where(deep_doc_mask, qd_met + qd_flood_met, qd_met)
    soilcarbon_input_doc = soilcarbon_input_doc.at[:, :, :, IMETBEL, :].set(
        frac_soil[IMETABOLIC, IACTIVE, IBELOW] * met_doc_qd / dt_days
    )
    floodcarbon_input = floodcarbon_input.at[:, :, IACT, :].add(
        jnp.sum(
            jnp.where(
                shallow_doc_mask,
                frac_soil[IMETABOLIC, IACTIVE, IBELOW] * qd_flood_met / dt_days,
                0.0,
            ),
            axis=2,
        )
    )

    return LitterBelowgroundDecompositionResult(
        litter_below=litter_updated,
        resp_hetero_litter=resp_hetero_litter,
        resp_hetero_flood=resp_hetero_flood,
        soilcarbon_input_doc=soilcarbon_input_doc,
        floodcarbon_input=floodcarbon_input,
        qd_structural=qd_struct,
        qd_metabolic=qd_met,
        qd_flood_structural=qd_flood_struct,
        qd_flood_metabolic=qd_flood_met,
    )


def littercalc_leak_core_with_controls(
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
    fbact,
    poor_soils,
    flood_frac,
    *,
    dt_days,
    control_temp_above=None,
    control_moist_above=None,
    soil_mc_top_by_pft=None,
    tsurf=None,
    soil_mc=None,
    z_soil_for_controls=None,
    pref_soil_veg=None,
    frozen_respiration_func: int | None = None,
    moist_func_moyano: bool = False,
    moyano_zz_coef_deep=None,
    moyano_bulk_dens=None,
    moyano_clay=None,
    moyano_carbon_32l=None,
    moyano_veget_max=None,
    nslm: int | None = None,
    ndeep: int | None = None,
    sro_bottom=5,
    litterfrac=None,
):
    """Compose the source-closed litter leak core with explicit controls.

    Fortran provenance: ``src_stomate/stomate_litter.f90``, subroutine
    ``littercalc_leak``. This wrapper composes the audited helper sections:
    activity factors lines 1950-1968, litter additions and lignin mixing lines
    2101-2405, aboveground decay lines 2563-2725, belowground decay lines
    2731-2794, and deadleaf cover line 2799 via subroutine ``deadleaf`` lines
    1314-1360.

    The wrapper either accepts explicit ``control_temp_above``,
    ``control_moist_above``, and ``soil_mc_top_by_pft`` inputs or computes the
    ordinary non-Moyano aboveground controls from ``tsurf``, ``soil_mc``,
    ``z_soil_for_controls``, ``pref_soil_veg``, and
    ``frozen_respiration_func``. It does not implement the crop interpolation
    blocks in lines 2417-2479 and 2499-2560 or the mass-balance/output
    bookkeeping in lines 2801-2869.
    """

    litter_above = jnp.asarray(litter_above)
    litter_below = jnp.asarray(litter_below)
    if litter_below.ndim != 5:
        raise ValueError("litter_below must have shape (npts, nlitt, nvm, ndeep, nelements)")
    if ndeep is None:
        ndeep = litter_below.shape[3]
    if nslm is None:
        nslm = jnp.asarray(z_soil).shape[0] - 1
    if litterfrac is None:
        litterfrac = littercalc_litter_fractions()
    if control_temp_above is None or control_moist_above is None or soil_mc_top_by_pft is None:
        missing_controls = [
            name
            for name, value in (
                ("tsurf", tsurf),
                ("soil_mc", soil_mc),
                ("z_soil_for_controls", z_soil_for_controls),
                ("pref_soil_veg", pref_soil_veg),
                ("frozen_respiration_func", frozen_respiration_func),
            )
            if value is None
        ]
        if missing_controls:
            joined = ", ".join(missing_controls)
            raise ValueError(f"explicit controls or aboveground control inputs are required; missing {joined}")
        controls = littercalc_aboveground_controls(
            tsurf,
            soil_mc,
            z_soil_for_controls,
            pref_soil_veg,
            frozen_respiration_func=int(frozen_respiration_func),
            moist_func_moyano=moist_func_moyano,
            zz_coef_deep=moyano_zz_coef_deep,
            bulk_dens=moyano_bulk_dens,
            clay=moyano_clay,
            carbon_32l=moyano_carbon_32l,
            veget_max=moyano_veget_max,
        )
        control_temp_above = controls.control_temp_above
        control_moist_above = controls.control_moist_above
        soil_mc_top_by_pft = controls.soil_mc_top_by_pft

    activity = littercalc_activity_factors(fbact, lignin_struc_below)
    increments = littercalc_litter_increments(
        bm_to_litter,
        turnover,
        rprof,
        z_soil,
        dead_leaves,
        litterfrac=litterfrac,
        nslm=nslm,
        ndeep=ndeep,
    )
    pools = littercalc_apply_pool_increments(
        litter_above,
        litter_below,
        lignin_struc_above,
        lignin_struc_below,
        litterpart,
        increments,
    )
    fuel_added = littercalc_add_aboveground_fuel(
        fuel_1hr,
        fuel_10hr,
        fuel_100hr,
        fuel_1000hr,
        bm_to_litter,
        turnover,
        litterfrac=litterfrac,
    )
    above = decompose_aboveground_litter_leak(
        pools.litter_above,
        pools.increments.dead_leaves,
        pools.lignin_struc_above,
        control_temp_above,
        control_moist_above,
        soil_mc_top_by_pft,
        poor_soils,
        flood_frac,
        dt_days=dt_days,
        ndeep=ndeep,
    )
    fuel_struct = sync_aboveground_fuel_after_decomposition(
        fuel_added.fuel_1hr[:, :, ISTRUCTURAL, :],
        fuel_added.fuel_10hr[:, :, ISTRUCTURAL, :],
        fuel_added.fuel_100hr[:, :, ISTRUCTURAL, :],
        fuel_added.fuel_1000hr[:, :, ISTRUCTURAL, :],
        above.litter_above[:, ISTRUCTURAL, :, :].transpose((0, 1, 2)),
        above.qd_structural,
    )
    fuel_met = sync_aboveground_fuel_after_decomposition(
        fuel_added.fuel_1hr[:, :, IMETABOLIC, :],
        fuel_added.fuel_10hr[:, :, IMETABOLIC, :],
        fuel_added.fuel_100hr[:, :, IMETABOLIC, :],
        fuel_added.fuel_1000hr[:, :, IMETABOLIC, :],
        above.litter_above[:, IMETABOLIC, :, :].transpose((0, 1, 2)),
        above.qd_metabolic,
    )
    fuel_1hr_new = fuel_added.fuel_1hr.at[:, :, ISTRUCTURAL, :].set(fuel_struct.fuel_1hr)
    fuel_1hr_new = fuel_1hr_new.at[:, :, IMETABOLIC, :].set(fuel_met.fuel_1hr)
    fuel_10hr_new = fuel_added.fuel_10hr.at[:, :, ISTRUCTURAL, :].set(fuel_struct.fuel_10hr)
    fuel_10hr_new = fuel_10hr_new.at[:, :, IMETABOLIC, :].set(fuel_met.fuel_10hr)
    fuel_100hr_new = fuel_added.fuel_100hr.at[:, :, ISTRUCTURAL, :].set(fuel_struct.fuel_100hr)
    fuel_100hr_new = fuel_100hr_new.at[:, :, IMETABOLIC, :].set(fuel_met.fuel_100hr)
    fuel_1000hr_new = fuel_added.fuel_1000hr.at[:, :, ISTRUCTURAL, :].set(fuel_struct.fuel_1000hr)
    fuel_1000hr_new = fuel_1000hr_new.at[:, :, IMETABOLIC, :].set(fuel_met.fuel_1000hr)
    fuel = FuelPools(
        fuel_1hr=fuel_1hr_new,
        fuel_10hr=fuel_10hr_new,
        fuel_100hr=fuel_100hr_new,
        fuel_1000hr=fuel_1000hr_new,
        fuel_total=fuel_1hr_new + fuel_10hr_new + fuel_100hr_new + fuel_1000hr_new,
    )

    below = decompose_belowground_litter_leak(
        pools.litter_below,
        pools.lignin_struc_below,
        activity.fbact_met,
        activity.fbact_str,
        flood_frac,
        dt_days=dt_days,
        sro_bottom=sro_bottom,
    )
    deadleaf_cover = deadleaf_cover_from_litter(above.dead_leaves, veget_max, sla_calc)
    return LittercalcLeakCoreResult(
        litter_above=above.litter_above,
        litter_below=below.litter_below,
        lignin_struc_above=pools.lignin_struc_above,
        lignin_struc_below=pools.lignin_struc_below,
        litterpart=pools.litterpart,
        dead_leaves=above.dead_leaves,
        deadleaf_cover=deadleaf_cover,
        resp_hetero_litter=above.resp_hetero_litter + below.resp_hetero_litter,
        resp_hetero_flood=above.resp_hetero_flood + below.resp_hetero_flood,
        soilcarbon_input_doc=above.soilcarbon_input_doc + below.soilcarbon_input_doc,
        floodcarbon_input=above.floodcarbon_input + below.floodcarbon_input,
        fuel=fuel,
        increments=increments,
    )


littercalc_leak_core_with_controls_jit = jit(
    littercalc_leak_core_with_controls,
    static_argnames=(
        "frozen_respiration_func",
        "moist_func_moyano",
        "nslm",
        "ndeep",
        "sro_bottom",
    ),
)


def stomate_soil_mc_32l(soil_mc, *, ndeep: int = 32, nslm: int = 11):
    """Extend STOMATE soil moisture from hydrology layers to deep layers.

    Fortran provenance: ``src_stomate/stomate.f90``, subroutine
    ``stomate_main``, lines 3270-3275. The first ``nslm`` layers copy
    ``soil_mc`` and layers ``nslm+1..ndeep`` repeat hydrology layer ``nslm``.
    Input axes follow Fortran call-boundary order ``(npts, nslm, nstm)``.
    """

    soil_mc = jnp.asarray(soil_mc)
    if soil_mc.ndim != 3:
        raise ValueError("soil_mc must have shape (npts, nslm, nstm)")
    if soil_mc.shape[1] != nslm:
        raise ValueError(f"soil_mc layer axis must have length {nslm}")
    if ndeep < nslm:
        raise ValueError("ndeep must be greater than or equal to nslm")
    if ndeep == nslm:
        return soil_mc
    repeated = jnp.repeat(soil_mc[:, -1:, :], ndeep - nslm, axis=1)
    return jnp.concatenate([soil_mc, repeated], axis=1)


def stomate_littercalc_scaled_inputs(turnover_daily, bm_to_litter, *, dt_sechiba, one_day=86400.0):
    """Scale turnover and biomass-to-litter fields for ``littercalc_leak``.

    Fortran provenance: ``src_stomate/stomate.f90``, subroutine
    ``stomate_main``, lines 3288-3289:
    ``turnover_littercalc = turnover_daily * dt_sechiba / one_day`` and
    ``bm_to_littercalc = bm_to_litter * dt_sechiba / one_day``.
    """

    scale = jnp.asarray(dt_sechiba) / jnp.asarray(one_day)
    return jnp.asarray(turnover_daily) * scale, jnp.asarray(bm_to_litter) * scale


def stomate_littercalc_entry_prep(soil_mc, turnover_daily, bm_to_litter, *, dt_sechiba, ndeep: int = 32, nslm: int = 11):
    """Prepare exact local inputs immediately before ``littercalc_leak``.

    Fortran provenance: ``src_stomate/stomate.f90``, subroutine
    ``stomate_main``, lines 3270-3293. This helper stops before
    ``littercalc_leak`` and does not compute litter or soil-carbon process
    outputs.
    """

    turnover_littercalc, bm_to_littercalc = stomate_littercalc_scaled_inputs(
        turnover_daily,
        bm_to_litter,
        dt_sechiba=dt_sechiba,
    )
    return LittercalcEntryPrep(
        soil_mc_32l=stomate_soil_mc_32l(soil_mc, ndeep=ndeep, nslm=nslm),
        turnover_littercalc=turnover_littercalc,
        bm_to_littercalc=bm_to_littercalc,
    )


class AGRAllocationResult(NamedTuple):
    """Closed allocation fractions for the non-crop AGR split."""

    f_alloc: jnp.ndarray
    carb_rescale: jnp.ndarray


class AllocationStepResult(NamedTuple):
    """State and allocation fractions after ``stomate_alloc::alloc``."""

    biomass: jnp.ndarray
    leaf_age: jnp.ndarray
    leaf_frac: jnp.ndarray
    f_alloc: jnp.ndarray
    lai_around: jnp.ndarray
    limit_l: jnp.ndarray
    limit_w: jnp.ndarray
    limit_n: jnp.ndarray
    limit_w_or_n: jnp.ndarray
    l_to_lsr: jnp.ndarray
    s_to_lsr: jnp.ndarray
    r_to_lsr: jnp.ndarray
    alloc_sap_above: jnp.ndarray
    transloc_leaf: jnp.ndarray
    carb_rescale: jnp.ndarray


class NPPUpdateResult(NamedTuple):
    """Closed `npp_calc` outputs before leaf-age bookkeeping."""

    biomass: jnp.ndarray
    bm_alloc: jnp.ndarray
    resp_maint: jnp.ndarray
    resp_growth: jnp.ndarray
    npp: jnp.ndarray
    # Fortran `lm_old` is saved after maintenance tissue pumping and before
    # allocation; downstream leaf-age bookkeeping must use this boundary.
    biomass_before_alloc: jnp.ndarray


class NPPAgeSLAResult(NamedTuple):
    """Leaf-age, SLA, and plant-age state after ``npp_calc`` bookkeeping."""

    leaf_age: jnp.ndarray
    leaf_frac: jnp.ndarray
    age: jnp.ndarray
    sla_age1: jnp.ndarray
    sla_calc: jnp.ndarray
    leaf_age_weighted: jnp.ndarray


class TurnoverResult(NamedTuple):
    """State after ``stomate_turnover::turn`` local turnover processes."""

    turnover: jnp.ndarray
    senescence: jnp.ndarray
    c_export: jnp.ndarray
    leaf_age: jnp.ndarray
    leaf_frac: jnp.ndarray
    age: jnp.ndarray
    lai: jnp.ndarray
    biomass: jnp.ndarray
    turnover_time: jnp.ndarray
    leaf_meanage: jnp.ndarray
    leaf_age_crit: jnp.ndarray


class GapMortalityResult(NamedTuple):
    """State after ``lpj_gap::gap`` mortality and biomass transfer."""

    biomass: jnp.ndarray
    ind: jnp.ndarray
    bm_to_litter: jnp.ndarray
    mortality: jnp.ndarray
    mortality_fraction: jnp.ndarray
    delta_biomass: jnp.ndarray
    vigour: jnp.ndarray
    availability: jnp.ndarray


class KillResult(NamedTuple):
    """State after ``lpj_kill::kill`` eliminated invalid PFTs."""

    pft_present: jnp.ndarray
    cn_ind: jnp.ndarray
    ind: jnp.ndarray
    biomass: jnp.ndarray
    senescence: jnp.ndarray
    rip_time: jnp.ndarray
    age: jnp.ndarray
    leaf_age: jnp.ndarray
    leaf_frac: jnp.ndarray
    npp_longterm: jnp.ndarray
    when_growthinit: jnp.ndarray
    everywhere: jnp.ndarray
    veget_max: jnp.ndarray
    bm_to_litter: jnp.ndarray
    was_killed: jnp.ndarray


class CrownResult(NamedTuple):
    """Individual wood mass, crown area, and vegetation height."""

    woodmass_ind: jnp.ndarray
    cn_ind: jnp.ndarray
    height: jnp.ndarray
    veget_max: jnp.ndarray


class PrescribeResult(NamedTuple):
    """State after ``stomate_prescribe::prescribe``."""

    pft_present: jnp.ndarray
    everywhere: jnp.ndarray
    when_growthinit: jnp.ndarray
    biomass: jnp.ndarray
    leaf_frac: jnp.ndarray
    ind: jnp.ndarray
    cn_ind: jnp.ndarray
    co2_to_bm: jnp.ndarray


class ConstraintsResult(NamedTuple):
    """Climate adaptation and regeneration state after ``lpj_constraints``."""

    adapted: jnp.ndarray
    regenerate: jnp.ndarray
    regenerate_min: jnp.ndarray


class PhenologyResult(NamedTuple):
    """State after the audited ``stomate_phenology::phenology`` path."""

    biomass: jnp.ndarray
    leaf_frac: jnp.ndarray
    leaf_age: jnp.ndarray
    when_growthinit: jnp.ndarray
    co2_to_bm: jnp.ndarray
    begin_leaves: jnp.ndarray
    allow_initpheno: jnp.ndarray
    gdd_midwinter: jnp.ndarray | None = None


class LightCompetitionResult(NamedTuple):
    """State after ``lpj_light::light`` competition and light mortality."""

    ind: jnp.ndarray
    biomass: jnp.ndarray
    veget_lastlight: jnp.ndarray
    bm_to_litter: jnp.ndarray
    mortality: jnp.ndarray
    light_death: jnp.ndarray
    fpc_nat: jnp.ndarray


class EstablishmentRatesResult(NamedTuple):
    """Individual establishment increments before sapling biomass accounting."""

    d_ind: jnp.ndarray
    ind_estab: jnp.ndarray
    everywhere: jnp.ndarray
    fpc_nat: jnp.ndarray
    estab_rate_max_tree: jnp.ndarray
    estab_rate_max_grass: jnp.ndarray
    sumfpc: jnp.ndarray
    sumfpc_wood: jnp.ndarray
    spacefight_grass: jnp.ndarray
    fracnat: jnp.ndarray


class EstablishmentBiomassResult(NamedTuple):
    """State after sapling biomass accounting in ``lpj_establish``."""

    ind: jnp.ndarray
    biomass: jnp.ndarray
    leaf_age: jnp.ndarray
    leaf_frac: jnp.ndarray
    age: jnp.ndarray
    co2_to_bm: jnp.ndarray
    woodmass_ind: jnp.ndarray
    bm_to_litter: jnp.ndarray


class PftInoutResult(NamedTuple):
    """State after ``lpj_pftinout::pftinout`` PFT introduction/elimination."""

    veget_max: jnp.ndarray
    biomass: jnp.ndarray
    ind: jnp.ndarray
    age: jnp.ndarray
    leaf_frac: jnp.ndarray
    npp_longterm: jnp.ndarray
    lm_lastyearmax: jnp.ndarray
    senescence: jnp.ndarray
    pft_present: jnp.ndarray
    everywhere: jnp.ndarray
    when_growthinit: jnp.ndarray
    need_adjacent: jnp.ndarray
    rip_time: jnp.ndarray
    co2_to_bm: jnp.ndarray
    avail_tree: jnp.ndarray
    avail_grass: jnp.ndarray
    can_introduce: jnp.ndarray


class CoverResult(NamedTuple):
    """State after ordinary ``lpj_cover::cover`` redistribution."""

    veget_max: jnp.ndarray
    lai: jnp.ndarray
    litter: jnp.ndarray
    litter_avail: jnp.ndarray
    litter_not_avail: jnp.ndarray
    carbon: jnp.ndarray
    biomass: jnp.ndarray
    fuel_1hr: jnp.ndarray
    fuel_10hr: jnp.ndarray
    fuel_100hr: jnp.ndarray
    fuel_1000hr: jnp.ndarray
    turnover_daily: jnp.ndarray
    bm_to_litter: jnp.ndarray
    co2_to_bm: jnp.ndarray
    co2_fire: jnp.ndarray
    resp_hetero: jnp.ndarray
    resp_maint: jnp.ndarray
    resp_growth: jnp.ndarray
    gpp_daily: jnp.ndarray
    deepC_a: jnp.ndarray
    deepC_s: jnp.ndarray
    deepC_p: jnp.ndarray
    co2flux_old: jnp.ndarray
    co2flux_new: jnp.ndarray
    tcarbon: jnp.ndarray


class LpjCoverPeatResult(NamedTuple):
    """State after ``stomate_lpj::lpj_cover_peat`` dynamic-peat cover update."""

    veget_max: jnp.ndarray
    cn_ind: jnp.ndarray
    ind: jnp.ndarray
    biomass: jnp.ndarray
    litter: jnp.ndarray
    litter_avail: jnp.ndarray
    litter_not_avail: jnp.ndarray
    carbon: jnp.ndarray
    fuel_1hr: jnp.ndarray
    fuel_10hr: jnp.ndarray
    fuel_100hr: jnp.ndarray
    fuel_1000hr: jnp.ndarray
    turnover_daily: jnp.ndarray
    bm_to_litter: jnp.ndarray
    co2_to_bm: jnp.ndarray
    co2_fire: jnp.ndarray
    resp_hetero: jnp.ndarray
    resp_maint: jnp.ndarray
    resp_growth: jnp.ndarray
    gpp_daily: jnp.ndarray
    deepC_a: jnp.ndarray
    deepC_s: jnp.ndarray
    deepC_p: jnp.ndarray
    age: jnp.ndarray
    pft_present: jnp.ndarray
    senescence: jnp.ndarray
    when_growthinit: jnp.ndarray
    everywhere: jnp.ndarray
    leaf_frac: jnp.ndarray
    lm_lastyearmax: jnp.ndarray
    npp_longterm: jnp.ndarray
    carbon_save: jnp.ndarray
    deepC_a_save: jnp.ndarray
    deepC_s_save: jnp.ndarray
    deepC_p_save: jnp.ndarray
    delta_fsave: jnp.ndarray
    co2flux_old: jnp.ndarray
    co2flux_new: jnp.ndarray
    delta_fpeat: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::lpj_cover_peat lines 2583-3337",
    )


class StomateLpjCoverDispatchResult(NamedTuple):
    """Top-level ``StomateLpj`` section-13 cover branch selection."""

    cover: CoverResult | None
    peat_cover: LpjCoverPeatResult | None
    update_peatfrac: bool
    done_update_peatfrac: bool
    used_peat_cover: bool
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj lines 1358-1380",
    )


class AgripeatAdjustedFractionsResult(NamedTuple):
    """Vegetation fractions after ``lcchange::agripeat_adjust_fractions``."""

    veget_max_adjusted: jnp.ndarray
    sum_veget_natold: jnp.ndarray
    sum_veget_natnew: jnp.ndarray
    sum_crops_new: jnp.ndarray
    sum_peat_crops_old: jnp.ndarray
    sum_peat_crops_new: jnp.ndarray
    sumpeat_old: jnp.ndarray
    mass_error: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_lcchange.f90::agripeat_adjust_fractions lines 1640-1863",
    )


class AgripeatRedistributionResult(NamedTuple):
    """Agripeat litter, biomass-to-litter, carbon, and product-entry updates."""

    litter: jnp.ndarray
    litter_avail: jnp.ndarray
    litter_not_avail: jnp.ndarray
    bm_to_litter: jnp.ndarray
    carbon: jnp.ndarray
    deepC_a: jnp.ndarray
    deepC_s: jnp.ndarray
    deepC_p: jnp.ndarray
    convflux: jnp.ndarray
    prod10_entry: jnp.ndarray
    prod100_entry: jnp.ndarray
    delta_nat_peat: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_lcchange.f90::lcchange_main_agripeat lines 1342-1558",
    )


class AgripeatFinalizeResult(NamedTuple):
    """Agripeat product-pool aging and fuel rebalance outputs."""

    prod10: jnp.ndarray
    prod100: jnp.ndarray
    flux10: jnp.ndarray
    flux100: jnp.ndarray
    convflux: jnp.ndarray
    cflux_prod10: jnp.ndarray
    cflux_prod100: jnp.ndarray
    fuel_1hr: jnp.ndarray
    fuel_10hr: jnp.ndarray
    fuel_100hr: jnp.ndarray
    fuel_1000hr: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_lcchange.f90::lcchange_main_agripeat lines 1561-1613",
    )


class LcchangeMainAgripeatResult(NamedTuple):
    """State after ``lcchange_main_agripeat`` hard-coded PFT14-16 path."""

    veget_max_adjusted: jnp.ndarray
    biomass: jnp.ndarray
    ind: jnp.ndarray
    age: jnp.ndarray
    pft_present: jnp.ndarray
    senescence: jnp.ndarray
    when_growthinit: jnp.ndarray
    everywhere: jnp.ndarray
    co2_to_bm: jnp.ndarray
    bm_to_litter: jnp.ndarray
    turnover_daily: jnp.ndarray
    cn_ind: jnp.ndarray
    prod10: jnp.ndarray
    prod100: jnp.ndarray
    flux10: jnp.ndarray
    flux100: jnp.ndarray
    convflux: jnp.ndarray
    cflux_prod10: jnp.ndarray
    cflux_prod100: jnp.ndarray
    leaf_frac: jnp.ndarray
    npp_longterm: jnp.ndarray
    lm_lastyearmax: jnp.ndarray
    litter: jnp.ndarray
    litter_avail: jnp.ndarray
    litter_not_avail: jnp.ndarray
    carbon: jnp.ndarray
    deepC_a: jnp.ndarray
    deepC_s: jnp.ndarray
    deepC_p: jnp.ndarray
    fuel_1hr: jnp.ndarray
    fuel_10hr: jnp.ndarray
    fuel_100hr: jnp.ndarray
    fuel_1000hr: jnp.ndarray
    adjusted_fractions: AgripeatAdjustedFractionsResult
    redistribution: AgripeatRedistributionResult
    finalize: AgripeatFinalizeResult
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_lcchange.f90::lcchange_main_agripeat lines 1113-1632",
    )


class VmaxResult(NamedTuple):
    """Leaf-age state and Vcmax after ``stomate_vmax::vmax``."""

    leaf_age: jnp.ndarray
    leaf_frac: jnp.ndarray
    vcmax: jnp.ndarray
    leaf_efficiency: jnp.ndarray


class HarvestAgriResult(NamedTuple):
    """Crop harvest turnover reduction from ``StomateLpj::harvest``."""

    bm_to_litter: jnp.ndarray
    turnover_daily: jnp.ndarray
    harvest_above: jnp.ndarray
    above_old: jnp.ndarray


class GrasslandRoleSelectionResult(NamedTuple):
    """PFT roles used by ``main_grassland_management``."""

    enabled: bool
    mauto_c3: int
    mcut_c3: int
    mgraze_c3: int
    mnatural_c3: int
    mauto_c4: int
    mcut_c4: int
    mgraze_c4: int
    mnatural_c4: int
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj lines 1321-1353 gates main_grassland_management with enable_grazing",
        "fortran_source/ORCHIDEE/src_stomate/grassland_management.f90::main_grassland_management lines 970-1037 selects managed/cut/grazed/natural C3/C4 role PFTs",
    )


class GrasslandCuttingSpaResult(NamedTuple):
    """State after ``grassland_cutting::cutting_spa`` for active cuts."""

    biomass: jnp.ndarray
    devstage: jnp.ndarray
    regcount: jnp.ndarray
    wshcutinit: jnp.ndarray
    gmean: jnp.ndarray
    wc_frac: jnp.ndarray
    wnapo: jnp.ndarray
    wnsym: jnp.ndarray
    wgn: jnp.ndarray
    tasum: jnp.ndarray
    tgrowth: jnp.ndarray
    loss: jnp.ndarray
    lossc: jnp.ndarray
    lossn: jnp.ndarray
    tlossstart: jnp.ndarray
    lai: jnp.ndarray
    tcut: jnp.ndarray
    tcut_modif: jnp.ndarray
    wshtotsum: jnp.ndarray
    controle_azote_sum: jnp.ndarray
    wshtotsumprev: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/grassland_management.f90::main_grassland_management lines 1872-1954",
        "fortran_source/ORCHIDEE/src_stomate/grassland_cutting.f90::cutting_spa lines 48-304",
    )


class LitterAvailabilityResult(NamedTuple):
    """Aboveground litter available and unavailable pools."""

    litter_avail: jnp.ndarray
    litter_not_avail: jnp.ndarray


class StomateLpjOutputDiagnostics(NamedTuple):
    """Output-diagnostic pools prepared at the end of ``StomateLpj``."""

    tot_litter_carb: jnp.ndarray
    tot_soil_carb: jnp.ndarray
    tot_litter_soil_carb: jnp.ndarray
    tot_live_biomass: jnp.ndarray
    tot_turnover: jnp.ndarray
    tot_bm_to_litter: jnp.ndarray
    carb_mass_total: jnp.ndarray
    carb_mass_variation: jnp.ndarray
    carbon_32l_pftmean: jnp.ndarray
    carbon_32l_conct: jnp.ndarray
    free_doc: jnp.ndarray
    free_doc_stock: jnp.ndarray
    adsorbed_doc: jnp.ndarray
    f_veg_litter: jnp.ndarray


def root_temperature(stempdiag, z_soil, rprof):
    """Convolve soil temperature over the Fortran root profile.

    Fortran provenance: `src_stomate/stomate_resp.f90`, subroutine
    `maint_respiration`, lines 203-232.
    """

    stempdiag = jnp.asarray(stempdiag)
    z_soil = jnp.asarray(z_soil)
    rprof = jnp.asarray(rprof)
    if z_soil.ndim != 1:
        raise ValueError("z_soil must be one-dimensional")
    if stempdiag.shape[-1] != z_soil.shape[0] - 1:
        raise ValueError("stempdiag last axis must match len(z_soil) - 1")

    rpc = 1.0 / (1.0 - jnp.exp(-z_soil[-1] / rprof))
    weights = rpc[..., None] * (
        jnp.exp(-z_soil[:-1] / rprof[..., None]) - jnp.exp(-z_soil[1:] / rprof[..., None])
    )
    return jnp.sum(stempdiag * weights, axis=-1)


def set_lai_from_biomass(biomass, sla_calc):
    """Compute LAI from leaf biomass and dynamic SLA.

    Fortran provenance: `src_stomate/stomate_lai.f90`, subroutine `setlai`,
    lines 58-88. Equivalent inline non-crop update appears in
    `src_stomate/stomate.f90`, subroutine `stomate_main`, lines 4200-4208.
    Bare soil/PFT1 is forced to zero.
    """

    biomass = jnp.asarray(biomass)
    sla_calc = jnp.asarray(sla_calc)
    if biomass.ndim != 4:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    lai = biomass[:, :, ILEAF, ICARBON] * sla_calc
    return lai.at[:, 0].set(0.0)


def _maintenance_respiration_core(
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
    dt_sechiba_days,
    min_stomate,
    maint_resp_min_vmax,
    maint_resp_coeff,
):
    t_root = root_temperature(stempdiag[:, None, :], z_soil, rprof)
    lai = set_lai_from_biomass(biomass, sla_calc)

    npts, nvm, _, _ = biomass.shape
    t_maint = jnp.zeros((npts, nvm, NPARTS), dtype=biomass.dtype)
    above_parts = jnp.asarray(
        [ILEAF, ISAPABOVE, IHEARTABOVE, IFRUIT, IAGRSAPST, IAGRSAPPN, IAGRHRTST, IAGRHRTPN]
    )
    below_parts = jnp.asarray([ISAPBELOW, IHEARTBELOW, IROOT])
    t_maint = t_maint.at[:, :, above_parts].set(t2m[:, None, None])
    t_maint = t_maint.at[:, :, below_parts].set(t_root[:, :, None])
    reserve_temp = jnp.where(is_tree[None, :], t2m[:, None], t_root)
    t_maint = t_maint.at[:, :, ICARBRES].set(reserve_temp)

    tl = t2m_longterm - ZERO_CELSIUS
    slope = (
        maint_resp_slope[None, :, 0]
        + tl[:, None] * maint_resp_slope[None, :, 1]
        + tl[:, None] * tl[:, None] * maint_resp_slope[None, :, 2]
    )
    coeff_maint = jnp.maximum(
        (coeff_maint_zero[None, :, :] * dt_sechiba_days)
        * (1.0 + slope[:, :, None] * (t_maint - ZERO_CELSIUS)),
        0.0,
    )

    plain_resp = coeff_maint * biomass[:, :, :, ICARBON]
    leaf_factor = (
        maint_resp_min_vmax * lai
        + maint_resp_coeff * (1.0 - jnp.exp(-ext_coeff[None, :] * lai))
    ) / jnp.where(lai > min_stomate, lai, 1.0)
    leaf_resp = coeff_maint[:, :, ILEAF] * biomass[:, :, ILEAF, ICARBON] * leaf_factor
    leaf_resp = jnp.where(
        (biomass[:, :, ILEAF, ICARBON] > min_stomate) & (lai > min_stomate),
        leaf_resp,
        0.0,
    )
    resp_maint_part = plain_resp.at[:, :, ILEAF].set(leaf_resp)
    resp_maint_part = resp_maint_part.at[:, 0, :].set(0.0)
    return MaintenanceRespirationResult(
        lai=lai,
        t_root=t_root,
        coeff_maint=coeff_maint,
        resp_maint_part=resp_maint_part,
    )


maintenance_respiration_core_jit = jit(_maintenance_respiration_core)


def maintenance_respiration(
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
    *,
    dt_sechiba_days=1.0,
    min_stomate=1.0e-8,
    maint_resp_min_vmax=DEFAULT_MAINT_RESP_MIN_VMAX,
    maint_resp_coeff=DEFAULT_MAINT_RESP_COEFF,
) -> MaintenanceRespirationResult:
    """Compute maintenance respiration by PFT and biomass part.

    Fortran provenance: `src_stomate/stomate_resp.f90`, subroutine
    `maint_respiration`, inputs/local declarations lines 122-170, root
    temperature lines 203-232, part temperatures/coefficient lines 234-303,
    LAI and respiration lines 319-376. Constants provenance:
    `src_parameters/constantes_var.f90`, pool indices lines 196-208 and
    `maint_resp_min_vmax`/`maint_resp_coeff` lines 1315-1318.

    Inputs preserve Fortran axes: `biomass[npts, nvm, nparts, nelements]`.
    Bare soil/PFT1 is forced to zero as in Fortran.
    """

    biomass = jnp.asarray(biomass)
    t2m = jnp.asarray(t2m)
    t2m_longterm = jnp.asarray(t2m_longterm)
    stempdiag = jnp.asarray(stempdiag)
    rprof = jnp.asarray(rprof)
    sla_calc = jnp.asarray(sla_calc)
    coeff_maint_zero = jnp.asarray(coeff_maint_zero)
    maint_resp_slope = jnp.asarray(maint_resp_slope)
    ext_coeff = jnp.asarray(ext_coeff)
    is_tree = jnp.asarray(is_tree, dtype=bool)

    if biomass.ndim != 4:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    npts, nvm, nparts, _ = biomass.shape
    if nparts != NPARTS:
        raise ValueError(f"biomass pool axis must have length {NPARTS}")

    return _maintenance_respiration_core(
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
        dt_sechiba_days,
        min_stomate,
        maint_resp_min_vmax,
        maint_resp_coeff,
    )


def agr_allocation_split(
    leaf_biomass,
    reserve_biomass,
    sla_calc,
    senescence,
    l_to_lsr,
    s_to_lsr,
    r_to_lsr,
    alloc_sap_above,
    alloc_agr_st,
    alloc_agr_pn,
    lai_max,
    *,
    f_fruit=0.025,
    ecureuil=0.0,
    min_stomate=0.0,
) -> AGRAllocationResult:
    """Compute the closed non-crop allocation fractions including AGR split.

    Fortran provenance: `src_stomate/stomate_alloc.f90`, subroutine `alloc`,
    signature/output declarations lines 146-152 and 187-201; allocation
    fraction equations lines 760-834, especially lines 798-817 for reserve
    rescaling and AGR stilt/pneumatophore split.

    This helper deliberately does not compute `l_to_lsr`, `s_to_lsr`,
    `r_to_lsr`, or `alloc_sap_above`; those require the season/stress state in
    `stomate_alloc.f90` lines 335-759 and must be supplied by audited upstream
    kernels.
    """

    leaf_biomass = jnp.asarray(leaf_biomass)
    reserve_biomass = jnp.asarray(reserve_biomass)
    sla_calc = jnp.asarray(sla_calc)
    senescence = jnp.asarray(senescence, dtype=bool)
    l_to_lsr = jnp.asarray(l_to_lsr)
    s_to_lsr = jnp.asarray(s_to_lsr)
    r_to_lsr = jnp.asarray(r_to_lsr)
    alloc_sap_above = jnp.asarray(alloc_sap_above)
    alloc_agr_st = jnp.asarray(alloc_agr_st)
    alloc_agr_pn = jnp.asarray(alloc_agr_pn)
    lai_max = jnp.asarray(lai_max)

    f_alloc = jnp.zeros(leaf_biomass.shape + (NPARTS,), dtype=jnp.result_type(leaf_biomass, float))
    carb_rescale = jnp.where(
        reserve_biomass * sla_calc < 2.0 * lai_max,
        1.0 / (1.0 + ecureuil * (l_to_lsr + r_to_lsr)),
        1.0,
    )
    active = leaf_biomass > min_stomate
    growing = active & (~senescence)
    remaining = (1.0 - f_fruit) * carb_rescale

    f_alloc = f_alloc.at[..., ICARBRES].set(1.0)
    f_alloc = f_alloc.at[..., IFRUIT].set(jnp.where(growing, f_fruit, 0.0))
    f_alloc = f_alloc.at[..., ILEAF].set(jnp.where(growing, l_to_lsr * remaining, 0.0))
    f_alloc = f_alloc.at[..., ISAPABOVE].set(
        jnp.where(growing, s_to_lsr * alloc_sap_above * remaining * (1.0 - alloc_agr_st - alloc_agr_pn), 0.0)
    )
    f_alloc = f_alloc.at[..., IAGRSAPST].set(
        jnp.where(growing, s_to_lsr * alloc_sap_above * remaining * alloc_agr_st, 0.0)
    )
    f_alloc = f_alloc.at[..., IAGRSAPPN].set(
        jnp.where(growing, s_to_lsr * alloc_sap_above * remaining * alloc_agr_pn, 0.0)
    )
    f_alloc = f_alloc.at[..., ISAPBELOW].set(
        jnp.where(growing, s_to_lsr * (1.0 - alloc_sap_above) * remaining, 0.0)
    )
    f_alloc = f_alloc.at[..., IROOT].set(jnp.where(growing, r_to_lsr * remaining, 0.0))
    f_alloc = f_alloc.at[..., ICARBRES].set(
        jnp.where(senescence & active, 1.0, jnp.where(growing, (1.0 - carb_rescale) * (1.0 - f_fruit), 1.0))
    )
    return AGRAllocationResult(f_alloc=f_alloc, carb_rescale=carb_rescale)


def allocation_step(
    *,
    lai,
    veget_max,
    senescence,
    when_growthinit,
    moiavail_week,
    tsoil_month,
    soilhum_month,
    biomass,
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
    when_growthinit_cut=None,
    is_grassland_manag=None,
    dt_days=1.0,
    f_fruit=0.1,
    ecureuil=None,
    alloc_sap_above_grass=1.0,
    min_l_to_lsr=0.2,
    max_l_to_lsr=0.6,
    z_nitrogen=0.2,
    nlim_tref=25.0,
    nlim_q10=10.0,
    reserve_time_tree=30.0,
    reserve_time_grass=20.0,
    reserve_time_cut=20.0,
    lai_happy_cut=0.25,
    tau_leafinit_cut=10.0,
    max_possible_lai=200.0,
    min_stomate=0.0,
) -> AllocationStepResult:
    """Run the source-backed non-crop allocation path.

    Fortran provenance: ``src_stomate/stomate_alloc.f90``, subroutine
    ``alloc``. This covers first/every-call initialization lines 280-332,
    nitrogen proxy convolution lines 335-395, natural/agricultural LAI context
    lines 398-445, reserve use and leaf-age update lines 452-602, and the
    non-``ok_LAIdev`` allocation fractions lines 604-817. The STICS-LAIdev crop
    branch is not derived beyond preserving the source default
    ``f_alloc(:,:,icarbres)=1``.
    """

    lai = jnp.asarray(lai)
    veget_max = jnp.asarray(veget_max)
    senescence = jnp.asarray(senescence, dtype=bool)
    when_growthinit = jnp.asarray(when_growthinit)
    moiavail_week = jnp.asarray(moiavail_week)
    tsoil_month = jnp.asarray(tsoil_month)
    soilhum_month = jnp.asarray(soilhum_month)
    biomass = jnp.asarray(biomass)
    age = jnp.asarray(age)
    leaf_age = jnp.asarray(leaf_age)
    leaf_frac = jnp.asarray(leaf_frac)
    z_soil = jnp.asarray(z_soil)
    sla_calc = jnp.asarray(sla_calc)
    natural = jnp.asarray(natural, dtype=bool)
    pasture = jnp.asarray(pasture, dtype=bool)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    ok_LAIdev = jnp.asarray(ok_LAIdev, dtype=bool)
    r0 = jnp.asarray(r0)
    s0 = jnp.asarray(s0)
    ext_coeff = jnp.asarray(ext_coeff)
    lai_max = jnp.asarray(lai_max)
    lai_max_to_happy = jnp.asarray(lai_max_to_happy)
    tau_leafinit = jnp.asarray(tau_leafinit)
    alloc_min = jnp.asarray(alloc_min)
    alloc_max = jnp.asarray(alloc_max)
    demi_alloc = jnp.asarray(demi_alloc)
    alloc_agr_st = jnp.asarray(alloc_agr_st)
    alloc_agr_pn = jnp.asarray(alloc_agr_pn)
    if ecureuil is None:
        ecureuil = jnp.zeros_like(r0)
    else:
        ecureuil = jnp.asarray(ecureuil)

    if biomass.ndim != 4 or biomass.shape[2] != NPARTS:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    npts, nvm, _, _ = biomass.shape
    if lai.shape != (npts, nvm) or veget_max.shape != (npts, nvm):
        raise ValueError("lai and veget_max must have shape (npts, nvm)")
    if leaf_age.shape != (npts, nvm, NLEAFAGES) or leaf_frac.shape != (npts, nvm, NLEAFAGES):
        raise ValueError("leaf_age and leaf_frac must have shape (npts, nvm, nleafages)")
    if tsoil_month.shape != soilhum_month.shape or tsoil_month.shape[0] != npts:
        raise ValueError("tsoil_month and soilhum_month must have matching (npts, nslm) shape")
    if z_soil.ndim != 1 or z_soil.shape[0] != tsoil_month.shape[1] + 1:
        raise ValueError("z_soil must have length nslm + 1")

    active_pft = jnp.arange(nvm) > 0
    l0 = 1.0 - r0 - s0

    rpc = 1.0 / (1.0 - jnp.exp(-z_soil[-1] / z_nitrogen))
    nitrogen_weights = rpc * (
        jnp.exp(-z_soil[:-1] / z_nitrogen) - jnp.exp(-z_soil[1:] / z_nitrogen)
    )
    t_nitrogen = jnp.sum(tsoil_month * nitrogen_weights[None, :], axis=1)
    h_nitrogen = jnp.sum(soilhum_month * nitrogen_weights[None, :], axis=1)

    natural_not_pasture = natural & (~pasture)
    veget_max_nat = jnp.where(natural_not_pasture[None, :], veget_max, 0.0)
    lai_nat = jnp.sum(veget_max_nat * lai, axis=1)
    lai_around = jnp.where(natural_not_pasture[None, :], lai_nat[:, None], lai)
    lai_happy = lai_max * lai_max_to_happy

    lm_old = biomass[:, :, ILEAF, ICARBON]
    reserve_time = jnp.where(is_tree, reserve_time_tree, reserve_time_grass)
    reserve_active = (
        active_pft[None, :]
        & (~ok_LAIdev[None, :])
        & (biomass[:, :, ILEAF, ICARBON] > 0.0)
        & (~senescence)
        & (lai < lai_happy[None, :])
        & (when_growthinit < reserve_time[None, :])
    )
    safe_reserve_tau = jnp.where(reserve_active, tau_leafinit[None, :], 1.0)
    safe_reserve_sla = jnp.where(reserve_active, sla_calc, 1.0)
    reserve_demand = (
        2.0
        * dt_days
        / safe_reserve_tau
        * lai_happy[None, :]
        / safe_reserve_sla
    )
    use_reserve = jnp.where(reserve_active, jnp.minimum(biomass[:, :, ICARBRES, ICARBON], reserve_demand), 0.0)
    safe_leaf_root_share = jnp.where(
        reserve_active,
        (l0 + r0)[None, :],
        1.0,
    )
    leaf_share = l0[None, :] / safe_leaf_root_share
    transloc_leaf = leaf_share * use_reserve
    biomass = biomass.at[:, :, ILEAF, ICARBON].add(transloc_leaf)
    biomass = biomass.at[:, :, IROOT, ICARBON].add(use_reserve - transloc_leaf)
    biomass = biomass.at[:, :, ICARBRES, ICARBON].add(-use_reserve)

    if is_grassland_manag is not None:
        is_grassland_manag = jnp.asarray(is_grassland_manag, dtype=bool)
        if when_growthinit_cut is None:
            raise ValueError("when_growthinit_cut is required with is_grassland_manag")
        when_growthinit_cut = jnp.asarray(when_growthinit_cut)
        grass_mask = active_pft[None, :] & is_grassland_manag[None, :]
        reset_cut_age = grass_mask & (when_growthinit == 0.0)
        leaf_age = leaf_age.at[:, :, 1].set(jnp.where(reset_cut_age, 10.0, leaf_age[:, :, 1]))
        leaf_age = leaf_age.at[:, :, 2].set(jnp.where(reset_cut_age, 20.0, leaf_age[:, :, 2]))
        leaf_age = leaf_age.at[:, :, 3].set(jnp.where(reset_cut_age, 30.0, leaf_age[:, :, 3]))

        cut_active = (
            grass_mask
            & (biomass[:, :, ILEAF, ICARBON] > 0.0)
            & (when_growthinit_cut < reserve_time_cut)
            & (lai < lai_happy[None, :])
        )
        safe_cut_tau = jnp.where(cut_active, tau_leafinit_cut, 1.0)
        safe_cut_sla = jnp.where(cut_active, sla_calc, 1.0)
        cut_demand = (
            2.0 * dt_days / safe_cut_tau * lai_happy_cut / safe_cut_sla
        )
        use_cut = jnp.where(cut_active, jnp.minimum(biomass[:, :, ICARBRES, ICARBON], cut_demand), 0.0)
        biomass = biomass.at[:, :, ILEAF, ICARBON].add(use_cut)
        biomass = biomass.at[:, :, ICARBRES, ICARBON].add(-use_cut)
        transloc_leaf = jnp.where(cut_active, use_cut, transloc_leaf)

    leaf_mass_young = leaf_frac[:, :, 0] * lm_old + transloc_leaf
    youngest_update = (transloc_leaf > min_stomate) & (leaf_mass_young > min_stomate) & active_pft[None, :]
    leaf_age0 = jnp.maximum(
        0.0,
        leaf_age[:, :, 0] * (leaf_mass_young - transloc_leaf) / jnp.where(leaf_mass_young != 0.0, leaf_mass_young, 1.0),
    )
    leaf_age = leaf_age.at[:, :, 0].set(jnp.where(youngest_update, leaf_age0, leaf_age[:, :, 0]))
    leaf_mass = biomass[:, :, ILEAF, ICARBON]
    has_leaf = (leaf_mass > min_stomate) & active_pft[None, :]
    safe_leaf_mass = jnp.where(leaf_mass != 0.0, leaf_mass, 1.0)
    leaf_frac0 = jnp.where(has_leaf, leaf_mass_young / safe_leaf_mass, leaf_frac[:, :, 0])
    leaf_frac_tail = jnp.where(
        has_leaf[:, :, None],
        leaf_frac[:, :, 1:] * lm_old[:, :, None] / safe_leaf_mass[:, :, None],
        leaf_frac[:, :, 1:],
    )
    leaf_frac = jnp.concatenate((leaf_frac0[:, :, None], leaf_frac_tail), axis=2)

    noncrop = active_pft[None, :] & (~ok_LAIdev[None, :])
    has_alloc_leaf = (biomass[:, :, ILEAF, ICARBON] > min_stomate) & noncrop
    tree_alloc = alloc_min[None, :] + (alloc_max[None, :] - alloc_min[None, :]) * (
        1.0 - jnp.exp(-age / demi_alloc[None, :])
    )
    alloc_sap_above = jnp.where(is_tree[None, :], tree_alloc, alloc_sap_above_grass)

    raw_limit_l = jnp.where(
        lai_around < max_possible_lai,
        jnp.maximum(0.1, jnp.exp(-ext_coeff[None, :] * lai_around)),
        0.1,
    )
    raw_limit_w = jnp.maximum(0.1, jnp.minimum(1.0, moiavail_week))
    limit_n_hum = jnp.maximum(0.5, jnp.minimum(1.0, h_nitrogen))[:, None]
    limit_n_temp = 2.0 ** ((t_nitrogen[:, None] - ZERO_CELSIUS - nlim_tref) / nlim_q10)
    limit_n_temp = jnp.maximum(0.1, jnp.minimum(1.0, limit_n_temp))
    raw_limit_n = jnp.maximum(0.1, jnp.minimum(1.0, limit_n_hum * limit_n_temp))
    raw_limit_n = jnp.broadcast_to(raw_limit_n, (npts, nvm))
    raw_limit_w_or_n = jnp.minimum(raw_limit_w, raw_limit_n)

    r_to_lsr = jnp.maximum(
        0.15,
        r0[None, :] * 3.0 * raw_limit_l / (raw_limit_l + 2.0 * raw_limit_w_or_n),
    )
    s_to_lsr = s0[None, :] * 3.0 * raw_limit_w_or_n / (2.0 * raw_limit_l + raw_limit_w_or_n)
    l_to_lsr = 1.0 - r_to_lsr - s_to_lsr
    l_to_lsr = jnp.maximum(min_l_to_lsr, jnp.minimum(max_l_to_lsr, l_to_lsr))
    r_to_lsr = 1.0 - l_to_lsr - s_to_lsr
    over_lai = has_alloc_leaf & (lai > lai_max[None, :])
    s_to_lsr = jnp.where(over_lai, s_to_lsr + l_to_lsr, s_to_lsr)
    l_to_lsr = jnp.where(over_lai, 0.0, l_to_lsr)

    l_to_lsr = jnp.where(has_alloc_leaf, l_to_lsr, 0.0)
    s_to_lsr = jnp.where(has_alloc_leaf, s_to_lsr, 0.0)
    r_to_lsr = jnp.where(has_alloc_leaf, r_to_lsr, 0.0)
    limit_l = jnp.where(has_alloc_leaf, raw_limit_l, 0.0)
    limit_w = jnp.where(has_alloc_leaf, raw_limit_w, 0.0)
    limit_n = jnp.where(has_alloc_leaf, raw_limit_n, 0.0)
    limit_w_or_n = jnp.where(has_alloc_leaf, raw_limit_w_or_n, 0.0)

    split = agr_allocation_split(
        leaf_biomass=biomass[:, :, ILEAF, ICARBON],
        reserve_biomass=biomass[:, :, ICARBRES, ICARBON],
        sla_calc=sla_calc,
        senescence=senescence,
        l_to_lsr=l_to_lsr,
        s_to_lsr=s_to_lsr,
        r_to_lsr=r_to_lsr,
        alloc_sap_above=alloc_sap_above,
        alloc_agr_st=alloc_agr_st,
        alloc_agr_pn=alloc_agr_pn,
        lai_max=lai_max,
        f_fruit=f_fruit,
        ecureuil=ecureuil[None, :],
        min_stomate=min_stomate,
    )
    default_f_alloc = jnp.zeros((npts, nvm, NPARTS), dtype=biomass.dtype).at[:, :, ICARBRES].set(1.0)
    f_alloc = jnp.where(noncrop[:, :, None], split.f_alloc, default_f_alloc)
    f_alloc = f_alloc.at[:, 0, :].set(default_f_alloc[:, 0, :])

    return AllocationStepResult(
        biomass=biomass,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        f_alloc=f_alloc,
        lai_around=lai_around,
        limit_l=limit_l,
        limit_w=limit_w,
        limit_n=limit_n,
        limit_w_or_n=limit_w_or_n,
        l_to_lsr=l_to_lsr,
        s_to_lsr=s_to_lsr,
        r_to_lsr=r_to_lsr,
        alloc_sap_above=alloc_sap_above,
        transloc_leaf=transloc_leaf,
        carb_rescale=split.carb_rescale,
    )


def prescribe_step(
    *,
    veget_max,
    dt_days,
    pft_present,
    everywhere,
    when_growthinit,
    biomass,
    leaf_frac,
    ind,
    cn_ind,
    co2_to_bm,
    natural,
    pasture,
    is_tree,
    bm_sapl,
    maxdia,
    pheno_is_none,
    ok_dgvm=False,
    lpj_gap_const_mort=True,
    firstcall=True,
    stomate_restart_none=False,
    min_stomate=0.0,
    large_value=1.0e33,
    bm_sapl_rescale=1.0,
    pipe_density=2.0e5,
    pipe_tune1=100.0,
    pipe_tune2=40.0,
    pipe_tune3=0.5,
    pipe_tune_exp_coeff=1.6,
) -> PrescribeResult:
    """Update prescribed/static vegetation state.

    Fortran provenance: ``src_stomate/stomate_prescribe.f90``,
    subroutine ``prescribe``. Crown area/density update follows lines 126-248.
    Cold-start first-call biomass/PFT initialization follows lines 257-346.
    The ``stom_restname_in == 'NONE'`` gate is represented by
    ``stomate_restart_none`` and must be enabled explicitly.  The default is
    restart-backed because production paper runs enter through a STOMATE
    restart; treating an omitted wiring flag as ``NONE`` would inject sapling
    biomass into otherwise empty, positive-cover restart cells.
    """

    natural_host = _host_bool_tuple_or_none(natural)
    pasture_host = _host_bool_tuple_or_none(pasture)
    is_tree_host = _host_bool_tuple_or_none(is_tree)
    pheno_is_none_host = _host_bool_tuple_or_none(pheno_is_none)

    veget_max = jnp.asarray(veget_max)
    pft_present = jnp.asarray(pft_present, dtype=bool)
    everywhere = jnp.asarray(everywhere)
    when_growthinit = jnp.asarray(when_growthinit)
    biomass = jnp.asarray(biomass)
    leaf_frac = jnp.asarray(leaf_frac)
    ind = jnp.asarray(ind)
    cn_ind = jnp.asarray(cn_ind)
    co2_to_bm = jnp.asarray(co2_to_bm)
    natural = jnp.asarray(natural, dtype=bool)
    pasture = jnp.asarray(pasture, dtype=bool)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    bm_sapl = jnp.asarray(bm_sapl)
    maxdia = jnp.asarray(maxdia)
    pheno_is_none = jnp.asarray(pheno_is_none, dtype=bool)

    if biomass.ndim != 4 or biomass.shape[2] != NPARTS:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    npts, nvm, _, _ = biomass.shape
    if veget_max.shape != (npts, nvm):
        raise ValueError("veget_max must have shape (npts, nvm)")
    if leaf_frac.shape != (npts, nvm, NLEAFAGES):
        raise ValueError("leaf_frac must have shape (npts, nvm, nleafages)")
    if bm_sapl.shape[0] != nvm or bm_sapl.shape[1] != NPARTS or bm_sapl.shape[2] <= ICARBON:
        raise ValueError("bm_sapl must have shape (nvm, nparts, nelements)")

    wood_parts = jnp.asarray(WOOD_BIOMASS_PARTS)
    active_pft = jnp.arange(nvm) > 0
    static_or_agri = (((not ok_dgvm) and lpj_gap_const_mort) | ((~natural) | pasture)) & active_pft
    positive_cover = veget_max > 0.0
    safe_maxdia = jnp.where(is_tree, maxdia, 1.0)
    woodmass = jnp.sum(biomass[:, :, wood_parts, ICARBON], axis=2) * veget_max
    critical = pipe_density * jnp.pi / 4.0 * pipe_tune2 * safe_maxdia[None, :] ** (2.0 + pipe_tune3)
    provisional_ind = woodmass / critical
    woodmass_ind = woodmass / jnp.where(provisional_ind != 0.0, provisional_ind, 1.0)
    dia = (woodmass_ind / (pipe_density * jnp.pi / 4.0 * pipe_tune2)) ** (1.0 / (2.0 + pipe_tune3))
    cn_tree = pipe_tune1 * jnp.minimum(safe_maxdia[None, :], dia) ** pipe_tune_exp_coeff
    denom = pipe_tune1 * (woodmass / (pipe_density * jnp.pi / 4.0 * pipe_tune2)) ** (
        pipe_tune_exp_coeff / (2.0 + pipe_tune3)
    )
    recalculated_ind = (veget_max / jnp.where(denom != 0.0, denom, 1.0)) ** (
        1.0 / (1.0 - (pipe_tune_exp_coeff / (2.0 + pipe_tune3)))
    )
    woodmass_ind2 = woodmass / jnp.where(recalculated_ind != 0.0, recalculated_ind, 1.0)
    dia2 = (woodmass_ind2 / (pipe_density * jnp.pi / 4.0 * pipe_tune2)) ** (1.0 / (2.0 + pipe_tune3))
    cn_tree2 = pipe_tune1 * jnp.minimum(safe_maxdia[None, :], dia2) ** pipe_tune_exp_coeff
    cn_positive = jnp.where(cn_tree * provisional_ind > 1.002 * veget_max, cn_tree, cn_tree2)
    cn_zero_wood = pipe_tune1 * safe_maxdia[None, :] ** pipe_tune_exp_coeff
    cn_tree_value = jnp.where(woodmass > min_stomate, cn_positive, cn_zero_wood)
    cn_candidate = jnp.where(is_tree[None, :], cn_tree_value, 1.0)
    cn_static = jnp.where(positive_cover, cn_candidate, 0.0)
    cn_ind = jnp.where(static_or_agri[None, :], cn_static, cn_ind)
    ind_static = jnp.where(positive_cover, veget_max / jnp.where(cn_ind != 0.0, cn_ind, 1.0), 0.0)
    ind = jnp.where(static_or_agri[None, :], ind_static, ind)

    if firstcall and stomate_restart_none:
        cold_static = (not ok_dgvm) | ((~natural) | pasture)
        cold_static_host = None
        if natural_host is not None and pasture_host is not None:
            cold_static_host = tuple(
                (not ok_dgvm) or ((not natural_host[pft]) or pasture_host[pft])
                for pft in range(nvm)
            )
        total_biomass = jnp.sum(biomass[:, :, :, ICARBON], axis=2)
        for pft in range(1, nvm):
            cold_static_pft = (
                cold_static_host[pft]
                if cold_static_host is not None
                else bool(cold_static[pft])
            )
            if cold_static_pft:
                positive_cover = veget_max[:, pft] > min_stomate
                empty = total_biomass[:, pft] <= min_stomate
                is_tree_pft = (
                    is_tree_host[pft]
                    if is_tree_host is not None
                    else bool(is_tree[pft])
                )
                if is_tree_pft:
                    init_tree = positive_cover & empty
                    sapling = bm_sapl_rescale * bm_sapl[None, pft, :, :] * ind[:, pft, None, None] / jnp.where(
                        positive_cover[:, None, None],
                        veget_max[:, pft, None, None],
                        1.0,
                    )
                    biomass = biomass.at[:, pft, :, :].set(jnp.where(init_tree[:, None, None], sapling, biomass[:, pft, :, :]))
                    leaf_frac = leaf_frac.at[:, pft, :].set(jnp.where(init_tree[:, None], 0.0, leaf_frac[:, pft, :]))
                    leaf_frac = leaf_frac.at[:, pft, 0].set(jnp.where(init_tree, 1.0, leaf_frac[:, pft, 0]))
                    when_growthinit = when_growthinit.at[:, pft].set(jnp.where(init_tree, large_value, when_growthinit[:, pft]))
                    seasonal_pft = (
                        not pheno_is_none_host[pft]
                        if pheno_is_none_host is not None
                        else ~pheno_is_none[pft]
                    )
                    seasonal = init_tree & seasonal_pft
                    biomass = biomass.at[:, pft, ILEAF, ICARBON].set(jnp.where(seasonal, 0.0, biomass[:, pft, ILEAF, ICARBON]))
                    leaf_frac = leaf_frac.at[:, pft, 0].set(jnp.where(seasonal, 0.0, leaf_frac[:, pft, 0]))
                    init_sum = jnp.sum(biomass[:, pft, :, ICARBON], axis=1)
                    co2_to_bm = co2_to_bm.at[:, pft].add(jnp.where(init_tree, init_sum / dt_days, 0.0))
                else:
                    init_grass = positive_cover & empty
                    reserve = bm_sapl[pft, ICARBRES, :] * ind[:, pft, None] / jnp.where(
                        positive_cover[:, None],
                        veget_max[:, pft, None],
                        1.0,
                    )
                    biomass = biomass.at[:, pft, ICARBRES, :].set(
                        jnp.where(init_grass[:, None], reserve, biomass[:, pft, ICARBRES, :])
                    )
                    leaf_frac = leaf_frac.at[:, pft, :].set(jnp.where(init_grass[:, None], 0.0, leaf_frac[:, pft, :]))
                    leaf_frac = leaf_frac.at[:, pft, 0].set(jnp.where(init_grass, 1.0, leaf_frac[:, pft, 0]))
                    when_growthinit = when_growthinit.at[:, pft].set(jnp.where(init_grass, large_value, when_growthinit[:, pft]))
                    co2_to_bm = co2_to_bm.at[:, pft].add(jnp.where(init_grass, biomass[:, pft, ICARBRES, ICARBON] / dt_days, 0.0))

                pft_present = pft_present.at[:, pft].set(jnp.where(positive_cover, True, pft_present[:, pft]))
                everywhere = everywhere.at[:, pft].set(jnp.where(positive_cover, 1.0, everywhere[:, pft]))

    return PrescribeResult(
        pft_present=pft_present,
        everywhere=everywhere,
        when_growthinit=when_growthinit,
        biomass=biomass,
        leaf_frac=leaf_frac,
        ind=ind,
        cn_ind=cn_ind,
        co2_to_bm=co2_to_bm,
    )


def constraints_step(
    *,
    t2m_month,
    t2m_min_daily,
    when_growthinit,
    adapted,
    regenerate,
    tseason,
    natural,
    is_tree,
    is_peat,
    pheno_is_none,
    tmin_crit,
    tcm_crit,
    dt_days=1.0,
    agriculture=True,
    too_long=5.0,
    undef=DEFAULT_UNDEF,
    large_value=1.0e33,
    one_year=ONE_YEAR_DAYS,
    grow_limit=7.0 + ZERO_CELSIUS,
) -> ConstraintsResult:
    """Update climatic adaptation and regeneration memory.

    Fortran provenance: ``src_stomate/lpj_constraints.f90``, subroutine
    ``constraints``. Signature and declarations are lines 99-121,
    initialization and ``regenerate_min`` are lines 133-157, PFT mask and
    climate-memory updates are lines 160-231, and history writes are lines
    233-242. Constants come from ``src_parameters/constantes_var.f90``:
    ``undef`` lines 168-172, ``large_value`` line 178, ``agriculture`` lines
    514-515, ``too_long`` lines 916-918, and ``ZeroCelsius`` line 372.

    ``t2m_min_daily`` is accepted to preserve the Fortran call/history
    contract, but the source routine only writes it to history and does not
    use it in the adaptation equations.
    """

    natural_host = _host_bool_tuple_or_none(natural)
    is_tree_host = _host_bool_tuple_or_none(is_tree)
    is_peat_host = _host_bool_tuple_or_none(is_peat)
    pheno_is_none_host = _host_bool_tuple_or_none(pheno_is_none)
    tmin_crit_host = None
    tcm_crit_host = None
    try:
        tmin_crit_host = tuple(float(value) for value in np.asarray(tmin_crit, dtype=float))
        tcm_crit_host = tuple(float(value) for value in np.asarray(tcm_crit, dtype=float))
    except Exception:
        pass

    t2m_month = jnp.asarray(t2m_month)
    t2m_min_daily = jnp.asarray(t2m_min_daily)
    when_growthinit = jnp.asarray(when_growthinit)
    adapted = jnp.asarray(adapted)
    regenerate = jnp.asarray(regenerate)
    tseason = jnp.asarray(tseason)
    natural = jnp.asarray(natural, dtype=bool)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    is_peat = jnp.asarray(is_peat, dtype=bool)
    pheno_is_none = jnp.asarray(pheno_is_none, dtype=bool)
    tmin_crit = jnp.asarray(tmin_crit)
    tcm_crit = jnp.asarray(tcm_crit)

    npts, nvm = adapted.shape
    if regenerate.shape != (npts, nvm) or when_growthinit.shape != (npts, nvm):
        raise ValueError("adapted, regenerate, and when_growthinit must have shape (npts, nvm)")
    if t2m_month.shape != (npts,) or t2m_min_daily.shape != (npts,) or tseason.shape != (npts,):
        raise ValueError("t2m_month, t2m_min_daily, and tseason must have shape (npts,)")
    for name, value in {
        "natural": natural,
        "is_tree": is_tree,
        "is_peat": is_peat,
        "pheno_is_none": pheno_is_none,
        "tmin_crit": tmin_crit,
        "tcm_crit": tcm_crit,
    }.items():
        if value.shape != (nvm,):
            raise ValueError(f"{name} must have shape (nvm,)")

    tau_adapt = jnp.asarray(one_year, dtype=adapted.dtype)
    tau_regenerate = jnp.asarray(one_year, dtype=regenerate.dtype)
    regenerate_min = jnp.exp(-too_long * one_year / tau_regenerate)
    memory_adapt = (tau_adapt - dt_days) / tau_adapt
    memory_regenerate = (tau_regenerate - dt_days) / tau_regenerate

    for pft in range(1, nvm):
        natural_pft = natural_host[pft] if natural_host is not None else bool(natural[pft])
        is_peat_pft = is_peat_host[pft] if is_peat_host is not None else bool(is_peat[pft])
        is_tree_pft = is_tree_host[pft] if is_tree_host is not None else bool(is_tree[pft])
        pheno_is_none_pft = (
            pheno_is_none_host[pft]
            if pheno_is_none_host is not None
            else bool(pheno_is_none[pft])
        )
        allowed = natural_pft or bool(agriculture) or is_peat_pft
        if allowed:
            adapted_pft = adapted[:, pft]
            regenerate_pft = regenerate[:, pft]
            no_tmin_threshold = (
                (tmin_crit_host[pft] == undef) or (not np.isfinite(tmin_crit_host[pft]))
                if tmin_crit_host is not None
                else bool((tmin_crit[pft] == undef) | (~jnp.isfinite(tmin_crit[pft])))
            )
            no_tcm_threshold = (
                (tcm_crit_host[pft] == undef) or (not np.isfinite(tcm_crit_host[pft]))
                if tcm_crit_host is not None
                else bool((tcm_crit[pft] == undef) | (~jnp.isfinite(tcm_crit[pft])))
            )

            if no_tmin_threshold:
                adapted_pft = jnp.ones_like(adapted_pft)

            if is_tree_pft and not pheno_is_none_pft:
                no_growthinit = (when_growthinit[:, pft] > too_long * one_year) & (
                    when_growthinit[:, pft] < large_value
                )
                adapted_pft = jnp.where(no_growthinit, 0.0, adapted_pft)

            if is_tree_pft:
                adapted_pft = jnp.where(tseason < grow_limit, 0.0, adapted_pft)

            adapted_pft = 1.0 - (1.0 - adapted_pft) * memory_adapt

            if no_tcm_threshold:
                regenerate_pft = jnp.ones_like(regenerate_pft)
            else:
                regenerate_pft = jnp.where(t2m_month <= tcm_crit[pft], 1.0, regenerate_pft)
                regenerate_pft = regenerate_pft * memory_regenerate

            adapted_pft = jnp.where(regenerate_pft <= regenerate_min, 0.0, adapted_pft)
            adapted = adapted.at[:, pft].set(adapted_pft)
            regenerate = regenerate.at[:, pft].set(regenerate_pft)
        else:
            adapted = adapted.at[:, pft].set(0.0)
            regenerate = regenerate.at[:, pft].set(0.0)

    return ConstraintsResult(
        adapted=adapted,
        regenerate=regenerate,
        regenerate_min=regenerate_min,
    )


def phenology_none_step(
    *,
    pft_present,
    when_growthinit,
    biomass,
    leaf_frac,
    leaf_age,
    co2_to_bm,
    dt_days=1.0,
    min_growthinit_time=300.0,
) -> PhenologyResult:
    """Run the numeric ``pheno_model == 'none'`` phenology scheduling path.

    Fortran provenance: ``src_stomate/stomate_phenology.f90::phenology``
    lines 319-334 computes ``allow_initpheno`` from ``when_growthinit``, lines
    345-348 increments ``when_growthinit`` by ``dt``, lines 356-426 keep
    ``begin_leaves`` false for the ``none`` onset model, and lines 455-563 do
    not change biomass or leaf-age state when no leaves begin.
    """

    pft_present = jnp.asarray(pft_present, dtype=bool)
    when_growthinit = jnp.asarray(when_growthinit)
    biomass = jnp.asarray(biomass)
    leaf_frac = jnp.asarray(leaf_frac)
    leaf_age = jnp.asarray(leaf_age)
    co2_to_bm = jnp.asarray(co2_to_bm)
    npts, nvm = pft_present.shape
    active_pft = jnp.arange(nvm) > 0
    allow_initpheno = (when_growthinit > min_growthinit_time) & active_pft[None, :]
    begin_leaves = jnp.zeros((npts, nvm), dtype=bool)
    return PhenologyResult(
        biomass=biomass,
        leaf_frac=leaf_frac,
        leaf_age=leaf_age,
        when_growthinit=when_growthinit + dt_days,
        co2_to_bm=co2_to_bm,
        begin_leaves=begin_leaves,
        allow_initpheno=allow_initpheno,
    )


def phenology_step(
    *,
    pft_present,
    when_growthinit,
    biomass,
    leaf_frac,
    leaf_age,
    co2_to_bm,
    pheno_model,
    dt_days=1.0,
    min_growthinit_time=300.0,
    active_pft_mask=None,
    t2m_month=None,
    t2m_week=None,
    t2m_longterm=None,
    time_hum_min=None,
    maxmoiavail_lastyear=None,
    minmoiavail_lastyear=None,
    moiavail_month=None,
    moiavail_week=None,
    hum_frac=None,
    hum_min_time=None,
    gdd_m5_dormance=None,
    pheno_gdd_crit=None,
    pheno_moigdd_t_crit=None,
    ngd_minus5=None,
    ngd_crit=None,
    ncd_dormance=None,
    gdd_midwinter=None,
    ncdgdd_temp=None,
    gddncd_ref=603.0,
    gddncd_curve=0.0091,
    gddncd_offset=64.0,
    is_tree=None,
    lai_initmin=None,
    sla_calc=None,
    ok_laidev=None,
    pdlai=None,
    slai=None,
    deltai=None,
    ssla=None,
    always_init=False,
    moiavail_always_tree=1.0,
    moiavail_always_grass=0.6,
    zero_celsius=273.15,
    t_always_add=10.0,
    undef=DEFAULT_UNDEF,
) -> PhenologyResult:
    """Run source-backed ``stomate_phenology::phenology`` onset paths.

    Fortran provenance: ``src_stomate/stomate_phenology.f90``, subroutine
    ``phenology``. The implementation covers the common scheduling shell:
    ``allow_initpheno`` from ``when_growthinit`` lines 319-334,
    ``when_growthinit = when_growthinit + dt`` lines 345-348,
    default ``begin_leaves = .FALSE.`` lines 356-367, ``pheno_model == 'none'``
    lines 371-426, the ``pheno_hum`` onset condition lines 650-789, the
    ``pheno_moi`` onset condition lines 838-958, ``pheno_humgdd`` lines
    1029-1197, ``pheno_moigdd`` lines 1261-1456, the ``pheno_ncdgdd`` onset
    condition and ``gdd_midwinter`` reset lines 1520-1615, the ``pheno_ngd``
    onset condition lines 1668-1750, ``pheno_moi_C4`` lines 1757-1902,
    ``pheno_siggdd`` lines 1910-2061, and the non-``ok_LAIdev`` and
    crop-STICS ``ok_LAIdev`` leaf-growth/reset block lines 455-563. Constants
    provenance: ``src_parameters/constantes_var.f90`` lines 1171-1174 and
    1274-1287, plus ``gddncd_*`` lines 1291-1296.

    Active PFTs using onset models other than ``none``, ``hum``, ``moi``,
    ``humgdd``, ``moigdd``, ``ncdgdd``, ``ngd``, ``moi_C4``, or ``siggdd`` are rejected
    until their corresponding Fortran subroutines are ported. Inactive non-
    supported PFTs may be present in the parameter table so a PFT14-only run can
    remain PFT-extensible without silently executing missing logic.
    """

    active_pft_mask_host = _host_bool_tuple_or_none(active_pft_mask) if active_pft_mask is not None else None
    pft_present = jnp.asarray(pft_present, dtype=bool)
    when_growthinit = jnp.asarray(when_growthinit)
    biomass = jnp.asarray(biomass)
    leaf_frac = jnp.asarray(leaf_frac)
    leaf_age = jnp.asarray(leaf_age)
    co2_to_bm = jnp.asarray(co2_to_bm)
    gdd_midwinter_out = None if gdd_midwinter is None else jnp.asarray(gdd_midwinter)

    if biomass.ndim != 4 or biomass.shape[2] != NPARTS:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    npts, nvm, _, _ = biomass.shape
    if pft_present.shape != (npts, nvm) or when_growthinit.shape != (npts, nvm) or co2_to_bm.shape != (npts, nvm):
        raise ValueError("pft_present, when_growthinit, and co2_to_bm must have shape (npts, nvm)")
    if leaf_frac.shape != (npts, nvm, NLEAFAGES) or leaf_age.shape != (npts, nvm, NLEAFAGES):
        raise ValueError("leaf_frac and leaf_age must have shape (npts, nvm, nleafages)")
    if len(pheno_model) != nvm:
        raise ValueError("pheno_model must have length nvm")

    models = tuple(str(model).strip() for model in pheno_model)
    if active_pft_mask is None:
        active_pft_mask_arr = jnp.any(pft_present, axis=0)
    else:
        active_pft_mask_arr = jnp.asarray(active_pft_mask, dtype=bool)
        if active_pft_mask_arr.shape != (nvm,):
            raise ValueError("active_pft_mask must have shape (nvm,)")

    supported_models = {"none", "hum", "moi", "humgdd", "moigdd", "ncdgdd", "ngd", "moi_C4", "siggdd"}
    if active_pft_mask_host is not None:
        unsupported_active = tuple(
            pft for pft in range(1, nvm) if active_pft_mask_host[pft] and models[pft] not in supported_models
        )
    else:
        unsupported_active = tuple(
            pft for pft in range(1, nvm) if bool(active_pft_mask_arr[pft]) and models[pft] not in supported_models
        )
    if unsupported_active:
        unsupported_desc = ", ".join(f"PFT{pft + 1}:{models[pft]}" for pft in unsupported_active)
        raise NotImplementedError(
            "phenology_step only implements audited pheno_model='none', 'hum', 'moi', 'humgdd', 'moigdd', 'ncdgdd', 'ngd', 'moi_C4', and 'siggdd' paths for active PFTs; "
            f"unsupported active onset models: {unsupported_desc}"
        )

    active_pft = jnp.arange(nvm) > 0
    allow_initpheno = (when_growthinit > min_growthinit_time) & active_pft[None, :]
    begin_leaves = jnp.zeros((npts, nvm), dtype=bool)
    when_growthinit = when_growthinit + dt_days

    def active_model_pfts(model: str) -> tuple[int, ...]:
        if active_pft_mask_host is not None:
            return tuple(pft for pft in range(1, nvm) if active_pft_mask_host[pft] and models[pft] == model)
        return tuple(pft for pft in range(1, nvm) if bool(active_pft_mask_arr[pft]) and models[pft] == model)

    active_hum = active_model_pfts("hum")
    active_moi = active_model_pfts("moi")
    active_humgdd = active_model_pfts("humgdd")
    active_moigdd = active_model_pfts("moigdd")
    active_ncdgdd = active_model_pfts("ncdgdd")
    active_ngd = active_model_pfts("ngd")
    active_moi_c4 = active_model_pfts("moi_C4")
    active_siggdd = active_model_pfts("siggdd")
    active_onset = (
        active_hum
        + active_moi
        + active_humgdd
        + active_moigdd
        + active_ncdgdd
        + active_ngd
        + active_moi_c4
        + active_siggdd
    )
    if active_onset:
        missing_leaf = tuple(
            name
            for name, value in {
                "is_tree": is_tree,
                "lai_initmin": lai_initmin,
                "sla_calc": sla_calc,
            }.items()
            if value is None
        )
        if missing_leaf:
            raise ValueError(f"active phenology onset requires explicit source-backed inputs: {', '.join(missing_leaf)}")
        is_tree = jnp.asarray(is_tree, dtype=bool)
        lai_initmin = jnp.asarray(lai_initmin)
        sla_calc = jnp.asarray(sla_calc)
        if ok_laidev is None:
            ok_laidev = jnp.zeros((nvm,), dtype=bool)
        else:
            ok_laidev = jnp.asarray(ok_laidev, dtype=bool)
        if sla_calc.shape != (npts, nvm):
            raise ValueError("phenology sla_calc must have shape (npts, nvm)")
        if is_tree.shape != (nvm,) or lai_initmin.shape != (nvm,) or ok_laidev.shape != (nvm,):
            raise ValueError("phenology PFT arrays must have shape (nvm,)")
        active_ok_laidev = tuple(pft for pft in active_onset if bool(ok_laidev[pft]))
        if active_ok_laidev:
            missing_crop = tuple(
                name
                for name, value in {
                    "deltai": deltai,
                    "ssla": ssla,
                }.items()
                if value is None
            )
            if missing_crop:
                raise ValueError(
                    "active ok_LAIdev phenology onset requires explicit source-backed inputs: "
                    + ", ".join(missing_crop)
                )
            deltai = jnp.asarray(deltai)
            ssla = jnp.asarray(ssla)
            if deltai.shape != (npts, nvm) or ssla.shape != (npts, nvm):
                raise ValueError("phenology deltai and ssla must have shape (npts, nvm)")

    def _apply_non_lai_leaf_onset(pft, begin_pft):
        nonlocal biomass, leaf_frac, leaf_age, when_growthinit, co2_to_bm
        begin_leaves_local = begin_pft
        if bool(ok_laidev[pft]):
            bm_use = jnp.where(begin_leaves_local, deltai[:, pft] / ssla[:, pft] * 2.0 * 10000.0, 0.0)
            age_reset = begin_leaves_local
        else:
            lm_min = lai_initmin[pft] / sla_calc[:, pft]
            low_leaf = begin_leaves_local & (biomass[:, pft, ILEAF, ICARBON] < lm_min)
            bm_wanted = 2.0 * lm_min
            reserve = biomass[:, pft, ICARBRES, ICARBON]
            if bool(always_init):
                reserve_short = low_leaf & (reserve < bm_wanted)
                co2_to_bm = co2_to_bm.at[:, pft].add(
                    jnp.where(reserve_short, (bm_wanted - reserve) / dt_days, 0.0)
                )
                reserve = jnp.where(reserve_short, bm_wanted, reserve)
                biomass = biomass.at[:, pft, ICARBRES, ICARBON].set(reserve)
            bm_use = jnp.where(low_leaf, jnp.minimum(reserve, bm_wanted), 0.0)
            age_reset = low_leaf
        biomass = biomass.at[:, pft, ILEAF, ICARBON].add(bm_use / 2.0)
        biomass = biomass.at[:, pft, IROOT, ICARBON].add(bm_use / 2.0)
        biomass = biomass.at[:, pft, ICARBRES, ICARBON].add(-bm_use)
        when_growthinit = when_growthinit.at[:, pft].set(jnp.where(begin_leaves_local, 0.0, when_growthinit[:, pft]))
        leaf_frac = leaf_frac.at[:, pft, 0].set(jnp.where(age_reset, 1.0, leaf_frac[:, pft, 0]))
        for leaf_class in range(1, NLEAFAGES):
            leaf_frac = leaf_frac.at[:, pft, leaf_class].set(
                jnp.where(age_reset, 0.0, leaf_frac[:, pft, leaf_class])
            )
        for leaf_class in range(NLEAFAGES):
            leaf_age = leaf_age.at[:, pft, leaf_class].set(
                jnp.where(age_reset, 0.0, leaf_age[:, pft, leaf_class])
            )

    if active_hum:
        missing = tuple(
            name
            for name, value in {
                "maxmoiavail_lastyear": maxmoiavail_lastyear,
                "minmoiavail_lastyear": minmoiavail_lastyear,
                "moiavail_month": moiavail_month,
                "moiavail_week": moiavail_week,
                "hum_frac": hum_frac,
            }.items()
            if value is None
        )
        if missing:
            raise ValueError(f"active pheno_model='hum' requires explicit source-backed inputs: {', '.join(missing)}")
        maxmoiavail_lastyear = jnp.asarray(maxmoiavail_lastyear)
        minmoiavail_lastyear = jnp.asarray(minmoiavail_lastyear)
        moiavail_month = jnp.asarray(moiavail_month)
        moiavail_week = jnp.asarray(moiavail_week)
        hum_frac = jnp.asarray(hum_frac)
        if (
            maxmoiavail_lastyear.shape != (npts, nvm)
            or minmoiavail_lastyear.shape != (npts, nvm)
            or moiavail_month.shape != (npts, nvm)
            or moiavail_week.shape != (npts, nvm)
        ):
            raise ValueError("hum phenology daily arrays must have shape (npts, nvm)")
        if hum_frac.shape != (nvm,):
            raise ValueError("hum phenology PFT arrays must have shape (nvm,)")

        for pft in active_hum:
            if bool(hum_frac[pft] == undef) or bool(~jnp.isfinite(hum_frac[pft])):
                raise ValueError(f"hum_frac is undefined for active hum phenology PFT{pft + 1}")
            threshold = moiavail_always_tree if bool(is_tree[pft]) else moiavail_always_grass
            availability_crit = minmoiavail_lastyear[:, pft] + hum_frac[pft] * (
                maxmoiavail_lastyear[:, pft] - minmoiavail_lastyear[:, pft]
            )
            begin_pft = (
                pft_present[:, pft]
                & allow_initpheno[:, pft]
                & (
                    ((moiavail_week[:, pft] >= availability_crit) & (moiavail_month[:, pft] < moiavail_week[:, pft]))
                    | (moiavail_month[:, pft] >= threshold)
                )
            )
            begin_leaves = begin_leaves.at[:, pft].set(begin_pft)
            _apply_non_lai_leaf_onset(pft, begin_pft)

    if active_moi:
        missing = tuple(
            name
            for name, value in {
                "time_hum_min": time_hum_min,
                "moiavail_month": moiavail_month,
                "moiavail_week": moiavail_week,
                "hum_min_time": hum_min_time,
            }.items()
            if value is None
        )
        if missing:
            raise ValueError(f"active pheno_model='moi' requires explicit source-backed inputs: {', '.join(missing)}")
        time_hum_min = jnp.asarray(time_hum_min)
        moiavail_month = jnp.asarray(moiavail_month)
        moiavail_week = jnp.asarray(moiavail_week)
        hum_min_time = jnp.asarray(hum_min_time)
        if (
            time_hum_min.shape != (npts, nvm)
            or moiavail_month.shape != (npts, nvm)
            or moiavail_week.shape != (npts, nvm)
        ):
            raise ValueError("moi phenology daily arrays must have shape (npts, nvm)")
        if hum_min_time.shape != (nvm,):
            raise ValueError("moi phenology PFT arrays must have shape (nvm,)")

        for pft in active_moi:
            if bool(hum_min_time[pft] == undef) or bool(~jnp.isfinite(hum_min_time[pft])):
                raise ValueError(f"hum_min_time is undefined for active moi phenology PFT{pft + 1}")
            threshold = moiavail_always_tree if bool(is_tree[pft]) else moiavail_always_grass
            begin_pft = (
                pft_present[:, pft]
                & allow_initpheno[:, pft]
                & (
                    ((moiavail_week[:, pft] > moiavail_month[:, pft]) & (time_hum_min[:, pft] > hum_min_time[pft]))
                    | (moiavail_month[:, pft] >= threshold)
                )
            )
            begin_leaves = begin_leaves.at[:, pft].set(begin_pft)
            _apply_non_lai_leaf_onset(pft, begin_pft)

    active_gdd_moist = active_humgdd + active_moigdd + active_moi_c4 + active_siggdd
    active_moi_gdd = active_moigdd + active_moi_c4 + active_siggdd
    if active_gdd_moist:
        missing = tuple(
            name
            for name, value in {
                "gdd_m5_dormance": gdd_m5_dormance,
                "pheno_gdd_crit": pheno_gdd_crit,
                "t2m_longterm": t2m_longterm,
                "t2m_month": t2m_month,
                "t2m_week": t2m_week,
                "moiavail_month": moiavail_month,
                "moiavail_week": moiavail_week,
            }.items()
            if value is None
        )
        if active_humgdd:
            missing += tuple(
                name
                for name, value in {
                    "maxmoiavail_lastyear": maxmoiavail_lastyear,
                    "minmoiavail_lastyear": minmoiavail_lastyear,
                    "hum_frac": hum_frac,
                }.items()
                if value is None
            )
        if active_moi_gdd:
            missing += tuple(
                name
                for name, value in {
                    "time_hum_min": time_hum_min,
                    "hum_min_time": hum_min_time,
                }.items()
                if value is None
            )
        if missing:
            raise ValueError(
                "active pheno_model='humgdd'/'moigdd'/'moi_C4'/'siggdd' requires explicit source-backed inputs: "
                + ", ".join(dict.fromkeys(missing))
            )
        gdd_m5_dormance = jnp.asarray(gdd_m5_dormance)
        pheno_gdd_crit = jnp.asarray(pheno_gdd_crit)
        t2m_longterm = jnp.asarray(t2m_longterm)
        t2m_month = jnp.asarray(t2m_month)
        t2m_week = jnp.asarray(t2m_week)
        moiavail_month = jnp.asarray(moiavail_month)
        moiavail_week = jnp.asarray(moiavail_week)
        if gdd_m5_dormance.shape != (npts, nvm) or moiavail_month.shape != (npts, nvm) or moiavail_week.shape != (npts, nvm):
            raise ValueError("humgdd/moigdd phenology daily PFT arrays must have shape (npts, nvm)")
        if pheno_gdd_crit.shape != (nvm, 3) or t2m_longterm.shape != (npts,) or t2m_month.shape != (npts,) or t2m_week.shape != (npts,):
            raise ValueError("humgdd/moigdd pheno_gdd_crit must have shape (nvm, 3), and temperature arrays must have shape (npts,)")

        t_always = zero_celsius + t_always_add
        tl = t2m_longterm - zero_celsius
        temp_condition = (t2m_week > t2m_month) | (t2m_month > t_always)

        def _gdd_threshold(pft):
            if bool(jnp.any(pheno_gdd_crit[pft, :] == undef)) or bool(jnp.any(~jnp.isfinite(pheno_gdd_crit[pft, :]))):
                raise ValueError(f"pheno_gdd_crit is undefined for active phenology PFT{pft + 1}")
            return pheno_gdd_crit[pft, 0] + tl * pheno_gdd_crit[pft, 1] + tl * tl * pheno_gdd_crit[pft, 2]

        if active_humgdd:
            maxmoiavail_lastyear = jnp.asarray(maxmoiavail_lastyear)
            minmoiavail_lastyear = jnp.asarray(minmoiavail_lastyear)
            hum_frac = jnp.asarray(hum_frac)
            if maxmoiavail_lastyear.shape != (npts, nvm) or minmoiavail_lastyear.shape != (npts, nvm):
                raise ValueError("humgdd phenology moisture-history arrays must have shape (npts, nvm)")
            if hum_frac.shape != (nvm,):
                raise ValueError("humgdd phenology hum_frac must have shape (nvm,)")
            for pft in active_humgdd:
                if bool(hum_frac[pft] == undef) or bool(~jnp.isfinite(hum_frac[pft])):
                    raise ValueError(f"hum_frac is undefined for active humgdd phenology PFT{pft + 1}")
                gdd_crit = _gdd_threshold(pft)
                moiavail_always = moiavail_always_tree if bool(is_tree[pft]) else moiavail_always_grass
                moiavail_crit = minmoiavail_lastyear[:, pft] + hum_frac[pft] * (
                    maxmoiavail_lastyear[:, pft] - minmoiavail_lastyear[:, pft]
                )
                moisture_condition = (
                    ((moiavail_week[:, pft] >= moiavail_crit) & (moiavail_month[:, pft] < moiavail_crit))
                    | (moiavail_month[:, pft] >= moiavail_always)
                )
                begin_pft = (
                    pft_present[:, pft]
                    & allow_initpheno[:, pft]
                    & (gdd_m5_dormance[:, pft] != undef)
                    & (gdd_m5_dormance[:, pft] >= gdd_crit)
                    & temp_condition
                    & moisture_condition
                )
                begin_leaves = begin_leaves.at[:, pft].set(begin_pft)
                _apply_non_lai_leaf_onset(pft, begin_pft)

        if active_moi_gdd:
            active_moigdd_ok = tuple(pft for pft in active_moigdd if bool(ok_laidev[pft]))
            if active_moigdd_ok:
                missing_crop_onset = tuple(
                    name
                    for name, value in {
                        "pdlai": pdlai,
                        "slai": slai,
                    }.items()
                    if value is None
                )
                if missing_crop_onset:
                    raise ValueError(
                        "active moigdd ok_LAIdev phenology onset requires explicit source-backed inputs: "
                        + ", ".join(missing_crop_onset)
                    )
                pdlai = jnp.asarray(pdlai)
                slai = jnp.asarray(slai)
                if pdlai.shape != (npts, nvm) or slai.shape != (npts, nvm):
                    raise ValueError("moigdd phenology pdlai and slai must have shape (npts, nvm)")
            time_hum_min = jnp.asarray(time_hum_min)
            hum_min_time = jnp.asarray(hum_min_time)
            if time_hum_min.shape != (npts, nvm):
                raise ValueError("moigdd phenology time_hum_min must have shape (npts, nvm)")
            if hum_min_time.shape != (nvm,):
                raise ValueError("moigdd phenology hum_min_time must have shape (nvm,)")

        if active_moigdd or active_siggdd:
            if pheno_moigdd_t_crit is None:
                pheno_moigdd_t_crit = jnp.ones((nvm,), dtype=pheno_gdd_crit.dtype) * undef
            else:
                pheno_moigdd_t_crit = jnp.asarray(pheno_moigdd_t_crit)
            if pheno_moigdd_t_crit.shape != (nvm,):
                raise ValueError("moigdd/siggdd phenology pheno_moigdd_t_crit must have shape (nvm,)")

        def _moi_gdd_moisture_condition(pft):
            moiavail_always = moiavail_always_tree if bool(is_tree[pft]) else moiavail_always_grass
            return (
                ((time_hum_min[:, pft] > hum_min_time[pft]) & (moiavail_week[:, pft] > moiavail_month[:, pft]))
                | (moiavail_month[:, pft] >= moiavail_always)
            )

        if active_moigdd:
            for pft in active_moigdd:
                if bool(hum_min_time[pft] == undef) or bool(~jnp.isfinite(hum_min_time[pft])):
                    raise ValueError(f"hum_min_time is undefined for active moigdd phenology PFT{pft + 1}")
                gdd_crit = _gdd_threshold(pft)
                temp_crit_ok = (pheno_moigdd_t_crit[pft] == undef) | (
                    t2m_month > (zero_celsius + pheno_moigdd_t_crit[pft])
                )
                if bool(ok_laidev[pft]):
                    begin_pft = ((slai[:, pft] - pdlai[:, pft]) > 0.0) & (pdlai[:, pft] == 0.0)
                else:
                    begin_pft = (
                        pft_present[:, pft]
                        & allow_initpheno[:, pft]
                        & (gdd_m5_dormance[:, pft] != undef)
                        & (gdd_m5_dormance[:, pft] >= gdd_crit)
                        & temp_condition
                        & _moi_gdd_moisture_condition(pft)
                        & temp_crit_ok
                    )
                begin_leaves = begin_leaves.at[:, pft].set(begin_pft)
                _apply_non_lai_leaf_onset(pft, begin_pft)

        if active_moi_c4:
            for pft in active_moi_c4:
                if bool(hum_min_time[pft] == undef) or bool(~jnp.isfinite(hum_min_time[pft])):
                    raise ValueError(f"hum_min_time is undefined for active moi_C4 phenology PFT{pft + 1}")
                gdd_crit = _gdd_threshold(pft)
                begin_pft = (
                    pft_present[:, pft]
                    & allow_initpheno[:, pft]
                    & (gdd_m5_dormance[:, pft] != undef)
                    & (gdd_m5_dormance[:, pft] >= gdd_crit)
                    & temp_condition
                    & _moi_gdd_moisture_condition(pft)
                    & (t2m_month > (zero_celsius + 22.0))
                )
                begin_leaves = begin_leaves.at[:, pft].set(begin_pft)
                _apply_non_lai_leaf_onset(pft, begin_pft)

        if active_siggdd:
            for pft in active_siggdd:
                if bool(hum_min_time[pft] == undef) or bool(~jnp.isfinite(hum_min_time[pft])):
                    raise ValueError(f"hum_min_time is undefined for active siggdd phenology PFT{pft + 1}")
                _gdd_threshold(pft)
                gdd_crit = 1.92717865e5 / (1.0 + jnp.exp(-8.13142588e-2 * (tl - 8.78682785e1)))
                temp_crit_ok = (pheno_moigdd_t_crit[pft] == undef) | (
                    t2m_month > (zero_celsius + pheno_moigdd_t_crit[pft])
                )
                begin_pft = (
                    pft_present[:, pft]
                    & allow_initpheno[:, pft]
                    & (gdd_m5_dormance[:, pft] != undef)
                    & (gdd_m5_dormance[:, pft] >= gdd_crit)
                    & temp_condition
                    & _moi_gdd_moisture_condition(pft)
                    & temp_crit_ok
                )
                begin_leaves = begin_leaves.at[:, pft].set(begin_pft)
                _apply_non_lai_leaf_onset(pft, begin_pft)

    if active_ncdgdd:
        missing = tuple(
            name
            for name, value in {
                "ncd_dormance": ncd_dormance,
                "gdd_midwinter": gdd_midwinter,
                "ncdgdd_temp": ncdgdd_temp,
                "t2m_month": t2m_month,
                "t2m_week": t2m_week,
            }.items()
            if value is None
        )
        if missing:
            raise ValueError(f"active pheno_model='ncdgdd' requires explicit source-backed inputs: {', '.join(missing)}")
        ncd_dormance = jnp.asarray(ncd_dormance)
        gdd_midwinter_out = jnp.asarray(gdd_midwinter_out)
        ncdgdd_temp = jnp.asarray(ncdgdd_temp)
        t2m_month = jnp.asarray(t2m_month)
        t2m_week = jnp.asarray(t2m_week)
        if ncd_dormance.shape != (npts, nvm) or gdd_midwinter_out.shape != (npts, nvm):
            raise ValueError("ncdgdd phenology ncd_dormance and gdd_midwinter must have shape (npts, nvm)")
        if ncdgdd_temp.shape != (nvm,) or t2m_month.shape != (npts,) or t2m_week.shape != (npts,):
            raise ValueError("ncdgdd phenology PFT arrays must have shape (nvm), and temperature arrays must have shape (npts,)")

        for pft in active_ncdgdd:
            if bool(ncdgdd_temp[pft] == undef) or bool(~jnp.isfinite(ncdgdd_temp[pft])):
                raise ValueError(f"ncdgdd_temp is undefined for active ncdgdd phenology PFT{pft + 1}")
            valid_memory = (gdd_midwinter_out[:, pft] != undef) & (ncd_dormance[:, pft] != undef)
            gdd_min = gddncd_ref / jnp.exp(gddncd_curve * ncd_dormance[:, pft]) - gddncd_offset
            begin_pft = (
                pft_present[:, pft]
                & allow_initpheno[:, pft]
                & valid_memory
                & (gdd_midwinter_out[:, pft] >= gdd_min)
                & (t2m_week > t2m_month)
            )
            begin_leaves = begin_leaves.at[:, pft].set(begin_pft)
            gdd_midwinter_out = gdd_midwinter_out.at[:, pft].set(
                jnp.where(begin_pft, undef, gdd_midwinter_out[:, pft])
            )
            _apply_non_lai_leaf_onset(pft, begin_pft)

    if active_ngd:
        missing = tuple(
            name
            for name, value in {
                "ngd_minus5": ngd_minus5,
                "ngd_crit": ngd_crit,
                "t2m_month": t2m_month,
                "t2m_week": t2m_week,
            }.items()
            if value is None
        )
        if missing:
            raise ValueError(f"active pheno_model='ngd' requires explicit source-backed inputs: {', '.join(missing)}")
        ngd_minus5 = jnp.asarray(ngd_minus5)
        ngd_crit = jnp.asarray(ngd_crit)
        t2m_month = jnp.asarray(t2m_month)
        t2m_week = jnp.asarray(t2m_week)
        if ngd_minus5.shape != (npts, nvm):
            raise ValueError("ngd phenology ngd_minus5 must have shape (npts, nvm)")
        if ngd_crit.shape != (nvm,) or t2m_month.shape != (npts,) or t2m_week.shape != (npts,):
            raise ValueError("ngd phenology ngd_crit must have shape (nvm), and temperature arrays must have shape (npts,)")

        for pft in active_ngd:
            if bool(ngd_crit[pft] == undef) or bool(~jnp.isfinite(ngd_crit[pft])):
                raise ValueError(f"ngd_crit is undefined for active ngd phenology PFT{pft + 1}")
            begin_pft = (
                pft_present[:, pft]
                & allow_initpheno[:, pft]
                & (ngd_minus5[:, pft] >= ngd_crit[pft])
                & (t2m_week > t2m_month)
            )
            begin_leaves = begin_leaves.at[:, pft].set(begin_pft)
            _apply_non_lai_leaf_onset(pft, begin_pft)

    return PhenologyResult(
        biomass=biomass,
        leaf_frac=leaf_frac,
        leaf_age=leaf_age,
        when_growthinit=when_growthinit,
        co2_to_bm=co2_to_bm,
        begin_leaves=begin_leaves,
        allow_initpheno=allow_initpheno,
        gdd_midwinter=gdd_midwinter_out,
    )


def npp_closed_update(
    biomass,
    gpp,
    f_alloc,
    resp_maint_part,
    pft_present,
    frac_growthresp,
    *,
    dt_days=1.0,
    tax_max=0.8,
    min_stomate=0.0,
) -> NPPUpdateResult:
    """Apply closed `npp_calc` biomass, respiration, and NPP algebra.

    Fortran provenance: `src_stomate/stomate_npp.f90`, subroutine `npp_calc`,
    signature and input/output declarations lines 116-180; local tax variables
    lines 230-247; initialization and GPP-to-allocatable biomass lines
    280-295; maintenance subtraction/tissue pump lines 299-386; allocation,
    growth respiration, biomass update, negative-biomass guard, and NPP lines
    447-531.

    This kernel requires audited `f_alloc` and `resp_maint_part` from upstream
    allocation and maintenance-respiration kernels. It intentionally stops
    before leaf-age/SLA bookkeeping in lines 544-725.
    """

    biomass = jnp.asarray(biomass)
    gpp = jnp.asarray(gpp)
    f_alloc = jnp.asarray(f_alloc)
    resp_maint_part = jnp.asarray(resp_maint_part)
    pft_present = jnp.asarray(pft_present, dtype=bool)
    frac_growthresp = jnp.asarray(frac_growthresp)
    fortran_pft = jnp.ones_like(pft_present, dtype=bool).at[:, 0].set(False)

    resp_maint = jnp.sum(jnp.where(pft_present[:, :, None], resp_maint_part, 0.0), axis=2)
    resp_maint = jnp.where(fortran_pft, resp_maint, 0.0)
    bm_alloc_tot = gpp * dt_days
    bm_tax_max = tax_max * bm_alloc_tot

    enough = (bm_alloc_tot > 0.0) & ((resp_maint * dt_days) < bm_tax_max)
    over_tax = (bm_alloc_tot > 0.0) & (~enough)
    no_alloc = bm_alloc_tot <= 0.0
    bm_after_maint = jnp.where(enough, bm_alloc_tot - resp_maint * dt_days, bm_alloc_tot)
    bm_after_maint = jnp.where(over_tax, bm_alloc_tot - bm_tax_max, bm_after_maint)
    bm_after_maint = jnp.where(no_alloc, 0.0, bm_after_maint)
    bm_after_maint = jnp.where(fortran_pft, bm_after_maint, 0.0)

    bm_pump = jnp.where(over_tax & fortran_pft, resp_maint * dt_days - bm_tax_max, 0.0)
    maintenance_active = resp_maint[:, :, None] != 0.0
    safe_resp_maint = jnp.where(maintenance_active, resp_maint[:, :, None], 1.0)
    maint_fraction = jnp.where(maintenance_active, resp_maint_part / safe_resp_maint, 0.0)
    pump_delta = jnp.zeros_like(maint_fraction).at[:, :, jnp.asarray(PUMPED_BIOMASS_PARTS)].set(
        -bm_pump[:, :, None] * maint_fraction[:, :, jnp.asarray(PUMPED_BIOMASS_PARTS)]
    )
    biomass_after_pump = biomass.at[:, :, :, ICARBON].add(pump_delta)
    reserve_no_alloc_delta = jnp.where(no_alloc & fortran_pft, bm_alloc_tot - resp_maint * dt_days, 0.0)
    biomass_after_pump = biomass_after_pump.at[:, :, ICARBRES, ICARBON].add(reserve_no_alloc_delta)

    bm_alloc_carbon = f_alloc * bm_after_maint[:, :, None]
    bm_alloc_carbon = jnp.where(fortran_pft[:, :, None], bm_alloc_carbon, 0.0)
    positive_alloc = bm_alloc_carbon > 0.0
    resp_growth_part = jnp.where(
        positive_alloc,
        frac_growthresp[None, :, None] * bm_alloc_carbon / dt_days,
        0.0,
    )
    bm_alloc_carbon = jnp.where(positive_alloc, (1.0 - frac_growthresp[None, :, None]) * bm_alloc_carbon, bm_alloc_carbon)
    resp_growth = jnp.sum(resp_growth_part, axis=2)

    bm_alloc = jnp.zeros_like(biomass).at[:, :, :, ICARBON].set(bm_alloc_carbon)
    # Fortran saves `lm_old` after maintenance tissue pumping and before
    # adding `bm_alloc`; leaf-age fractions must use this same boundary.
    biomass_before_alloc = biomass_after_pump
    biomass_updated = biomass_before_alloc + bm_alloc

    negative = (biomass_updated[:, :, :, ICARBON] < 0.0) & fortran_pft[:, :, None]
    bm_create = jnp.where(negative, min_stomate - biomass_updated[:, :, :, ICARBON], 0.0)
    biomass_updated = biomass_updated.at[:, :, :, ICARBON].add(bm_create)
    resp_maint = resp_maint - jnp.sum(bm_create, axis=2) / dt_days

    npp = gpp - resp_growth - resp_maint
    npp = npp.at[:, 0].set(0.0)
    resp_maint = resp_maint.at[:, 0].set(0.0)
    resp_growth = resp_growth.at[:, 0].set(0.0)
    bm_alloc = bm_alloc.at[:, 0, :, :].set(0.0)
    return NPPUpdateResult(
        biomass=biomass_updated,
        bm_alloc=bm_alloc,
        resp_maint=resp_maint,
        resp_growth=resp_growth,
        npp=npp,
        biomass_before_alloc=biomass_before_alloc,
    )


def npp_leaf_age_sla_age_update(
    biomass,
    biomass_old,
    bm_alloc,
    leaf_age,
    leaf_frac,
    age,
    pft_present,
    is_tree,
    sla_age1,
    sla_calc,
    sla_max,
    sla_min,
    *,
    dt_days,
    min_stomate=0.0,
    one_year=ONE_YEAR_DAYS,
) -> NPPAgeSLAResult:
    """Apply ``npp_calc`` leaf age, SLA, and whole-plant age bookkeeping.

    Fortran provenance: ``src_stomate/stomate_npp.f90``, subroutine
    ``npp_calc`` lines 490 and 544-672. ``biomass_old`` must be the Fortran
    ``lm_old`` boundary: after maintenance respiration has possibly pumped
    tissue biomass, and before adding ``bm_alloc``. This covers non-crop source
    code shared by the PFT14 path; crop-specific STICS variables are
    intentionally not derived here.
    """

    biomass = jnp.asarray(biomass)
    biomass_old = jnp.asarray(biomass_old)
    bm_alloc = jnp.asarray(bm_alloc)
    leaf_age = jnp.asarray(leaf_age)
    leaf_frac = jnp.asarray(leaf_frac)
    age = jnp.asarray(age)
    pft_present = jnp.asarray(pft_present, dtype=bool)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    sla_age1 = jnp.asarray(sla_age1)
    sla_calc = jnp.asarray(sla_calc)
    sla_max = jnp.asarray(sla_max)
    sla_min = jnp.asarray(sla_min)

    if biomass.ndim != 4 or biomass.shape[2] != NPARTS:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    npts, nvm, _, _ = biomass.shape
    if leaf_age.shape != (npts, nvm, NLEAFAGES) or leaf_frac.shape != (npts, nvm, NLEAFAGES):
        raise ValueError("leaf_age and leaf_frac must have shape (npts, nvm, nleafages)")
    if age.shape != (npts, nvm):
        raise ValueError("age must have shape (npts, nvm)")

    active_pft = jnp.arange(nvm) > 0
    lm_old = biomass_old[:, :, ILEAF, ICARBON]
    leaf_alloc = bm_alloc[:, :, ILEAF, ICARBON]
    leaf_mass_young = leaf_frac[:, :, 0] * lm_old + leaf_alloc
    youngest_update = (leaf_alloc > 0.0) & (leaf_mass_young > 0.0) & active_pft[None, :]
    leaf_age0 = jnp.maximum(
        0.0,
        leaf_age[:, :, 0] * (leaf_mass_young - leaf_alloc) / jnp.where(leaf_mass_young != 0.0, leaf_mass_young, 1.0),
    )
    leaf_age = leaf_age.at[:, :, 0].set(jnp.where(youngest_update, leaf_age0, leaf_age[:, :, 0]))

    leaf_mass = biomass[:, :, ILEAF, ICARBON]
    has_leaf = (leaf_mass > min_stomate) & active_pft[None, :]
    safe_leaf_mass = jnp.where(leaf_mass != 0.0, leaf_mass, 1.0)
    leaf_frac0 = jnp.where(has_leaf, leaf_mass_young / safe_leaf_mass, leaf_frac[:, :, 0])
    leaf_frac_tail = jnp.where(
        has_leaf[:, :, None],
        leaf_frac[:, :, 1:] * lm_old[:, :, None] / safe_leaf_mass[:, :, None],
        leaf_frac[:, :, 1:],
    )
    leaf_frac = jnp.concatenate((leaf_frac0[:, :, None], leaf_frac_tail), axis=2)

    sla_update = youngest_update
    new_sla_age1 = (
        sla_age1 * (leaf_mass_young - leaf_alloc) + sla_max[None, :] * leaf_alloc
    ) / jnp.where(leaf_mass_young != 0.0, leaf_mass_young, 1.0)
    sla_age1 = jnp.where(sla_update, new_sla_age1, sla_age1)
    sla_age2 = sla_max * 0.9
    sla_age3 = sla_max * 0.85
    sla_age4 = sla_max * 0.8
    sla_combined = (
        sla_age1 * leaf_frac[:, :, 0]
        + sla_age2[None, :] * leaf_frac[:, :, 1]
        + sla_age3[None, :] * leaf_frac[:, :, 2]
        + sla_age4[None, :] * leaf_frac[:, :, 3]
    )
    sla_calc = jnp.where(sla_update, sla_combined, sla_calc)

    leaf_age_weighted = jnp.sum(leaf_age * leaf_frac, axis=2)
    sla_calc = jnp.where(sla_calc > sla_max[None, :], sla_max[None, :], sla_calc)
    sla_calc = jnp.where(leaf_age_weighted < 5.0, sla_max[None, :], sla_calc)
    sla_calc = jnp.where(sla_calc < sla_min[None, :], sla_min[None, :], sla_calc)

    age = jnp.where(pft_present, age + dt_days / one_year, 0.0)
    bm_new = (
        biomass[:, :, ILEAF, ICARBON]
        + biomass[:, :, ISAPABOVE, ICARBON]
        + biomass[:, :, IROOT, ICARBON]
        + biomass[:, :, IFRUIT, ICARBON]
    )
    bm_add = (
        bm_alloc[:, :, ILEAF, ICARBON]
        + bm_alloc[:, :, ISAPABOVE, ICARBON]
        + bm_alloc[:, :, IROOT, ICARBON]
        + bm_alloc[:, :, IFRUIT, ICARBON]
    )
    age_rescale = (bm_new > 0.0) & (bm_add > 0.0) & (~is_tree[None, :]) & active_pft[None, :]
    safe_bm_new = jnp.where(age_rescale, bm_new, 1.0)
    age = jnp.where(
        age_rescale,
        age * (bm_new - bm_add) / safe_bm_new,
        age,
    )

    return NPPAgeSLAResult(
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        age=age,
        sla_age1=sla_age1,
        sla_calc=sla_calc,
        leaf_age_weighted=leaf_age_weighted,
    )


def turnover_leaf_mean_age(leaf_age, leaf_frac):
    """Recalculate mean leaf age from class ages and fractions.

    Fortran provenance: ``src_stomate/stomate_turnover.f90``, subroutine
    ``turn``, lines 279-292.
    """

    leaf_age = jnp.asarray(leaf_age)
    leaf_frac = jnp.asarray(leaf_frac)
    if leaf_age.shape != leaf_frac.shape or leaf_age.shape[-1] != NLEAFAGES:
        raise ValueError("leaf_age and leaf_frac must have matching (..., nleafages) shape")
    return jnp.sum(leaf_age * leaf_frac, axis=-1)


def turnover_critical_leaf_age(t2m_longterm, is_tree, natural, leafagecrit, *, leaf_age_crit_coeff=(1.5, 0.75, 10.0), leaf_age_crit_tref=20.0):
    """Compute the critical leaf age used by aging turnover.

    Fortran provenance: ``src_stomate/stomate_turnover.f90``, subroutine
    ``turn``, lines 577-597. Constants provenance:
    ``src_parameters/constantes_var.f90`` lines 1410-1413.
    """

    t2m_longterm = jnp.asarray(t2m_longterm)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    natural = jnp.asarray(natural, dtype=bool)
    leafagecrit = jnp.asarray(leafagecrit)
    coeff = jnp.asarray(leaf_age_crit_coeff)
    tree_or_crop = is_tree[None, :] | (~natural[None, :])
    grass_crit = jnp.minimum(
        leafagecrit[None, :] * coeff[0],
        jnp.maximum(
            leafagecrit[None, :] * coeff[1],
            leafagecrit[None, :] - coeff[2] * (t2m_longterm[:, None] - ZERO_CELSIUS - leaf_age_crit_tref),
        ),
    )
    return jnp.where(tree_or_crop, leafagecrit[None, :], grass_crit)


def turnover_senescence_flags(
    *,
    biomass,
    leaf_meanage,
    maxmoiavail_lastyear,
    minmoiavail_lastyear,
    moiavail_week,
    t2m_longterm,
    t2m_month,
    t2m_week,
    gdd_from_growthinit,
    lai,
    nrec,
    senescence_type,
    is_tree,
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
):
    """Evaluate climatic senescence and crop/grass turnover time.

    Fortran provenance: ``src_stomate/stomate_turnover.f90``, subroutine
    ``turn``, lines 307-478. PFT parameter provenance:
    ``src_parameters/pft_parameters.f90`` lines 629-653, 666 and 4490-4662.
    """

    biomass = jnp.asarray(biomass)
    leaf_meanage = jnp.asarray(leaf_meanage)
    maxmoiavail_lastyear = jnp.asarray(maxmoiavail_lastyear)
    minmoiavail_lastyear = jnp.asarray(minmoiavail_lastyear)
    moiavail_week = jnp.asarray(moiavail_week)
    t2m_longterm = jnp.asarray(t2m_longterm)
    t2m_month = jnp.asarray(t2m_month)
    t2m_week = jnp.asarray(t2m_week)
    gdd_from_growthinit = jnp.asarray(gdd_from_growthinit)
    lai = jnp.asarray(lai)
    nrec = jnp.asarray(nrec)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    is_grassland_manag = jnp.asarray(is_grassland_manag, dtype=bool)
    ok_laidev = jnp.asarray(ok_laidev, dtype=bool)
    min_leaf_age_for_senescence = jnp.asarray(min_leaf_age_for_senescence)
    gdd_senescence = jnp.asarray(gdd_senescence)
    senescence_temp = jnp.asarray(senescence_temp)
    hum_frac = jnp.asarray(hum_frac)
    senescence_hum = jnp.asarray(senescence_hum)
    nosenescence_hum = jnp.asarray(nosenescence_hum)
    max_turnover_time = jnp.asarray(max_turnover_time)
    min_turnover_time = jnp.asarray(min_turnover_time)
    leaffall = jnp.asarray(leaffall)
    lai_max = jnp.asarray(lai_max)

    npts, nvm = leaf_meanage.shape
    if biomass.shape[:2] != (npts, nvm):
        raise ValueError("biomass and leaf_meanage must share (npts, nvm)")

    senescence_type = jnp.asarray(senescence_type)
    senescence = jnp.zeros((npts, nvm), dtype=bool)
    turnover_time = jnp.zeros((npts, nvm), dtype=leaf_meanage.dtype)
    active_leaf = biomass[:, :, ILEAF, ICARBON] > 0.0
    old_enough = leaf_meanage > min_leaf_age_for_senescence[None, :]
    eligible = active_leaf & old_enough

    tl = t2m_longterm - ZERO_CELSIUS
    t_crit = (
        ZERO_CELSIUS
        + senescence_temp[None, :, 0]
        + tl[:, None] * senescence_temp[None, :, 1]
        + tl[:, None] * tl[:, None] * senescence_temp[None, :, 2]
    )
    cold = (t2m_month[:, None] < t_crit) & (t2m_week[:, None] < t2m_month[:, None])
    moiavail_crit = jnp.minimum(
        jnp.maximum(
            minmoiavail_lastyear + hum_frac[None, :] * (maxmoiavail_lastyear - minmoiavail_lastyear),
            senescence_hum[None, :],
        ),
        nosenescence_hum[None, :],
    )
    dry = moiavail_week < moiavail_crit

    stype_crop = senescence_type == SENESCENCE_CROP
    stype_cold = senescence_type == SENESCENCE_COLD
    stype_dry = senescence_type == SENESCENCE_DRY
    stype_mixed = senescence_type == SENESCENCE_MIXED

    senescence = jnp.where(stype_crop[None, :], eligible & (gdd_from_growthinit > gdd_senescence[None, :]), senescence)
    senescence = jnp.where(stype_cold[None, :], eligible & cold, senescence)
    senescence = jnp.where(stype_dry[None, :], eligible & dry, senescence)
    mixed_tree = stype_mixed[None, :] & is_tree[None, :]
    senescence = jnp.where(mixed_tree, eligible & (dry | cold), senescence)

    grass_mixed = stype_mixed[None, :] & (~is_tree[None, :])
    turnover_time = jnp.where(grass_mixed, max_turnover_time[None, :], turnover_time)
    moisture_time = max_turnover_time[None, :] * (1.0 - (1.0 - (moiavail_week / moiavail_crit)) ** 2)
    turnover_time = jnp.where(grass_mixed & eligible & dry, moisture_time, turnover_time)
    turnover_time = jnp.where(grass_mixed & (turnover_time < min_turnover_time[None, :]), min_turnover_time[None, :], turnover_time)
    grass_cold = eligible & cold & ((t2m_month[:, None] < t_crit) & (lai > lai_max[None, :] / 4.0) | (t2m_month[:, None] < ZERO_CELSIUS))
    turnover_time = jnp.where(grass_mixed & grass_cold, leaffall[None, :], turnover_time)
    turnover_time = jnp.where(grass_mixed & is_grassland_manag[None, :] & (lai < 0.5), max_turnover_time[None, :], turnover_time)
    turnover_time = jnp.where(
        grass_mixed & is_grassland_manag[None, :] & (lai > 2.55),
        jnp.maximum(45.0, 85.0 - lai * 10.0),
        turnover_time,
    )

    crop_harvest = ok_laidev[None, :] & (nrec > 0)
    senescence = jnp.where(crop_harvest, True, senescence)
    turnover_time = jnp.where(ok_laidev[None, :], jnp.where(crop_harvest, 1.0, max_turnover_time[None, :]), turnover_time)

    senescence = senescence.at[:, 0].set(False)
    turnover_time = turnover_time.at[:, 0].set(0.0)
    return senescence, turnover_time, moiavail_crit, t_crit


def turnover_climatic_biomass_loss(
    biomass,
    turnover,
    senescence,
    turnover_time,
    nrec,
    *,
    senescence_type,
    is_tree,
    ok_laidev,
    leaffall,
    max_turnover_time,
    prc_residual=0.5,
    dt_days=1.0,
):
    """Apply climatic/crop senescence biomass loss and crop export.

    Fortran provenance: ``src_stomate/stomate_turnover.f90``, subroutine
    ``turn``, lines 480-553. ``prc_residual`` provenance:
    ``src_parameters/constantes_var.f90`` lines 1611-1612.
    """

    biomass = jnp.asarray(biomass)
    turnover = jnp.asarray(turnover)
    senescence = jnp.asarray(senescence, dtype=bool)
    turnover_time = jnp.asarray(turnover_time)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    ok_laidev = jnp.asarray(ok_laidev, dtype=bool)
    leaffall = jnp.asarray(leaffall)
    max_turnover_time = jnp.asarray(max_turnover_time)
    senescence_type = jnp.asarray(senescence_type)

    npts, nvm = senescence.shape
    c_export = jnp.zeros((npts, nvm), dtype=biomass.dtype)
    active_pft = jnp.arange(nvm) > 0

    senescence_parts = jnp.asarray((ILEAF, ISAPABOVE, IROOT, IFRUIT))
    tree_senes = senescence & is_tree[None, :] & active_pft[None, :]
    turnover_carbon = turnover[:, :, senescence_parts, ICARBON]
    biomass_carbon = biomass[:, :, senescence_parts, ICARBON]
    leaffall_loss = biomass_carbon * dt_days / leaffall[None, :, None]
    tree_part_mask = jnp.asarray((True, False, True, False))
    turnover_carbon = jnp.where(tree_senes[:, :, None] & tree_part_mask[None, None, :], leaffall_loss, turnover_carbon)

    laidev_active = ok_laidev[None, :] & (turnover_time < max_turnover_time[None, :]) & active_pft[None, :]
    laidev_loss = biomass_carbon * dt_days / jnp.where(laidev_active, turnover_time, 1.0)[:, :, None]
    laidev_mask = (~is_tree[None, :]) & laidev_active & active_pft[None, :]
    turnover_carbon = jnp.where(laidev_mask[:, :, None], laidev_loss, turnover_carbon)

    crop_senes = (
        (~is_tree[None, :])
        & (~ok_laidev[None, :])
        & (senescence_type[None, :] == SENESCENCE_CROP)
        & senescence
        & active_pft[None, :]
    )
    turnover_carbon = jnp.where(crop_senes[:, :, None], leaffall_loss, turnover_carbon)

    grass_time = (
        (~is_tree[None, :])
        & (~ok_laidev[None, :])
        & (senescence_type[None, :] != SENESCENCE_CROP)
        & (senescence_type[None, :] != SENESCENCE_NONE)
        & (turnover_time < max_turnover_time[None, :])
        & active_pft[None, :]
    )
    grass_loss = biomass_carbon * dt_days / jnp.where(grass_time, turnover_time, 1.0)[:, :, None]
    turnover_carbon = jnp.where(grass_time[:, :, None], grass_loss, turnover_carbon)
    turnover = turnover.at[:, :, senescence_parts, ICARBON].set(turnover_carbon)
    biomass = biomass.at[:, :, senescence_parts, ICARBON].add(-turnover_carbon)

    c_export = jnp.where(
        ok_laidev[None, :],
        turnover[:, :, IFRUIT, ICARBON] + (1.0 - prc_residual) * (turnover[:, :, ILEAF, ICARBON] + turnover[:, :, ISAPABOVE, ICARBON]),
        c_export,
    )
    turnover = turnover.at[:, :, IFRUIT, ICARBON].set(jnp.where(ok_laidev[None, :], 0.0, turnover[:, :, IFRUIT, ICARBON]))
    turnover = turnover.at[:, :, ILEAF, ICARBON].set(
        jnp.where(ok_laidev[None, :], prc_residual * turnover[:, :, ILEAF, ICARBON], turnover[:, :, ILEAF, ICARBON])
    )
    turnover = turnover.at[:, :, ISAPABOVE, ICARBON].set(
        jnp.where(ok_laidev[None, :], prc_residual * turnover[:, :, ISAPABOVE, ICARBON], turnover[:, :, ISAPABOVE, ICARBON])
    )
    turnover = turnover.at[:, 0, :, :].set(0.0)
    c_export = c_export.at[:, 0].set(0.0)
    return biomass, turnover, c_export


def turnover_leaf_age_fall(
    biomass,
    turnover,
    leaf_age,
    leaf_frac,
    t2m_longterm,
    *,
    is_tree,
    natural,
    ok_laidev,
    leafagecrit,
    leaf_age_crit_coeff=(1.5, 0.75, 10.0),
    leaf_age_crit_tref=20.0,
    dt_days=1.0,
):
    """Apply leaf-age turnover and update leaf age fractions.

    Fortran provenance: ``src_stomate/stomate_turnover.f90``, subroutine
    ``turn``, lines 569-665. Constants provenance:
    ``src_parameters/constantes_var.f90`` lines 1410-1413.
    """

    biomass = jnp.asarray(biomass)
    turnover = jnp.asarray(turnover)
    leaf_age = jnp.asarray(leaf_age)
    leaf_frac = jnp.asarray(leaf_frac)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    ok_laidev = jnp.asarray(ok_laidev, dtype=bool)

    npts, nvm, _, _ = biomass.shape
    lm_old = biomass[:, :, ILEAF, ICARBON]
    delta_lm = jnp.zeros((npts, nvm, NLEAFAGES), dtype=biomass.dtype)
    leaf_age_crit = turnover_critical_leaf_age(
        t2m_longterm,
        is_tree,
        natural,
        leafagecrit,
        leaf_age_crit_coeff=leaf_age_crit_coeff,
        leaf_age_crit_tref=leaf_age_crit_tref,
    )

    active_pft = jnp.arange(nvm) > 0
    do_age_turn = (~ok_laidev)[None, :] & active_pft[None, :]
    for leaf_class in range(NLEAFAGES):
        rate_active = do_age_turn & (
            leaf_age[:, :, leaf_class] > leaf_age_crit / 2.0
        )
        safe_leaf_age = jnp.where(
            rate_active,
            leaf_age[:, :, leaf_class],
            1.0,
        )
        safe_leaf_age_crit = jnp.where(
            rate_active,
            leaf_age_crit,
            1.0,
        )
        rate = jnp.where(
            rate_active,
            jnp.minimum(
                0.99,
                dt_days
                / (
                    safe_leaf_age_crit
                    * (safe_leaf_age_crit / safe_leaf_age) ** 4
                ),
            ),
            0.0,
        )

        dturnover = biomass[:, :, ILEAF, ICARBON] * leaf_frac[:, :, leaf_class] * rate
        turnover = turnover.at[:, :, ILEAF, ICARBON].add(dturnover)
        biomass = biomass.at[:, :, ILEAF, ICARBON].add(-dturnover)
        delta_lm = delta_lm.at[:, :, leaf_class].set(-dturnover)

        dturnover = biomass[:, :, IROOT, ICARBON] * leaf_frac[:, :, leaf_class] * rate
        turnover = turnover.at[:, :, IROOT, ICARBON].add(dturnover)
        biomass = biomass.at[:, :, IROOT, ICARBON].add(-dturnover)

        dturnover = biomass[:, :, IFRUIT, ICARBON] * leaf_frac[:, :, leaf_class] * rate
        turnover = turnover.at[:, :, IFRUIT, ICARBON].add(dturnover)
        biomass = biomass.at[:, :, IFRUIT, ICARBON].add(-dturnover)

        dturnover = biomass[:, :, ISAPABOVE, ICARBON] * leaf_frac[:, :, leaf_class] * rate
        grass_age_turn = do_age_turn & (~is_tree[None, :])
        turnover = turnover.at[:, :, ISAPABOVE, ICARBON].add(jnp.where(grass_age_turn, dturnover, 0.0))
        biomass = biomass.at[:, :, ISAPABOVE, ICARBON].add(jnp.where(grass_age_turn, -dturnover, 0.0))

    leaf_biomass = biomass[:, :, ILEAF, ICARBON]
    updated_frac = (leaf_frac * lm_old[:, :, None] + delta_lm) / jnp.where(leaf_biomass[:, :, None] != 0.0, leaf_biomass[:, :, None], 1.0)
    leaf_frac = jnp.where(leaf_biomass[:, :, None] > 0.0, updated_frac, 0.0)
    leaf_frac = leaf_frac.at[:, 0, :].set(0.0)
    turnover = turnover.at[:, 0, :, :].set(0.0)
    return biomass, turnover, leaf_frac, leaf_age_crit


def turnover_shed_remaining_leaves(
    biomass,
    turnover,
    leaf_age,
    leaf_frac,
    leaf_meanage,
    lai,
    senescence,
    *,
    is_tree,
    ok_laidev,
    senescence_type,
    lai_initmin,
    sla_calc,
):
    """Drop remaining low leaf mass during senescence and reset leaf classes.

    Fortran provenance: ``src_stomate/stomate_turnover.f90``, subroutine
    ``turn``, lines 695-779. ``lai_initmin`` provenance:
    ``src_stomate/stomate_data.f90`` lines 585-591 and
    ``src_parameters/constantes_var.f90`` lines 1171-1174.
    """

    is_tree = jnp.asarray(is_tree, dtype=bool)
    ok_laidev = jnp.asarray(ok_laidev, dtype=bool)
    senescence_type = jnp.asarray(senescence_type)
    lai_initmin = jnp.asarray(lai_initmin)
    sla_calc = jnp.asarray(sla_calc)

    active_leaf = biomass[:, :, ILEAF, ICARBON] > 0.0
    low_leaf = biomass[:, :, ILEAF, ICARBON] < (lai_initmin[None, :] / 2.0) / sla_calc
    tree_shed = is_tree[None, :] & (senescence_type[None, :] != SENESCENCE_NONE) & (~ok_laidev[None, :])
    grass_shed = (~is_tree[None, :]) & (~ok_laidev[None, :])
    shed_rest = active_leaf & senescence & low_leaf & (tree_shed | grass_shed)

    grass_mask = shed_rest & (~is_tree[None, :])
    turnover = turnover.at[:, :, ILEAF, ICARBON].add(jnp.where(shed_rest, biomass[:, :, ILEAF, ICARBON], 0.0))
    turnover = turnover.at[:, :, IROOT, ICARBON].add(jnp.where(shed_rest, biomass[:, :, IROOT, ICARBON], 0.0))
    turnover = turnover.at[:, :, IFRUIT, ICARBON].add(jnp.where(shed_rest, biomass[:, :, IFRUIT, ICARBON], 0.0))
    turnover = turnover.at[:, :, ISAPABOVE, ICARBON].add(jnp.where(grass_mask, biomass[:, :, ISAPABOVE, ICARBON], 0.0))

    biomass = biomass.at[:, :, ILEAF, ICARBON].set(jnp.where(shed_rest, 0.0, biomass[:, :, ILEAF, ICARBON]))
    biomass = biomass.at[:, :, IROOT, ICARBON].set(jnp.where(shed_rest, 0.0, biomass[:, :, IROOT, ICARBON]))
    biomass = biomass.at[:, :, IFRUIT, ICARBON].set(jnp.where(shed_rest, 0.0, biomass[:, :, IFRUIT, ICARBON]))
    biomass = biomass.at[:, :, ISAPABOVE, ICARBON].set(jnp.where(grass_mask, 0.0, biomass[:, :, ISAPABOVE, ICARBON]))
    leaf_meanage = jnp.where(shed_rest, 0.0, leaf_meanage)
    lai = jnp.where(shed_rest, 0.0, lai)
    leaf_age = jnp.where(shed_rest[:, :, None], 0.0, leaf_age)
    leaf_frac = jnp.where(shed_rest[:, :, None], 0.0, leaf_frac)

    turnover = turnover.at[:, 0, :, :].set(0.0)
    return biomass, turnover, leaf_age, leaf_frac, leaf_meanage, lai, shed_rest


def turnover_herbivory(
    biomass,
    turnover,
    herbivores,
    *,
    is_tree,
    ok_herbivores=False,
    dt_days=1.0,
):
    """Apply optional herbivore consumption from leaves/fruits/stalks.

    Fortran provenance: ``src_stomate/stomate_turnover.f90``, subroutine
    ``turn``, lines 781-839.
    """

    if not ok_herbivores:
        return biomass, turnover

    herbivores = jnp.asarray(herbivores)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    valid = (biomass[:, :, ILEAF, ICARBON] > 0.0) & (herbivores > 0.0)
    for part in (ILEAF, IFRUIT):
        loss = biomass[:, :, part, ICARBON] * dt_days / herbivores
        turnover = turnover.at[:, :, part, ICARBON].add(jnp.where(valid, loss, 0.0))
        biomass = biomass.at[:, :, part, ICARBON].add(jnp.where(valid, -loss, 0.0))
    loss = biomass[:, :, ISAPABOVE, ICARBON] * dt_days / herbivores
    grass_valid = valid & (~is_tree[None, :])
    turnover = turnover.at[:, :, ISAPABOVE, ICARBON].add(jnp.where(grass_valid, loss, 0.0))
    biomass = biomass.at[:, :, ISAPABOVE, ICARBON].add(jnp.where(grass_valid, -loss, 0.0))
    turnover = turnover.at[:, 0, :, :].set(0.0)
    return biomass, turnover


def turnover_tree_fruit_and_sapwood(
    biomass,
    turnover,
    age,
    *,
    is_tree,
    tau_fruit,
    tau_sap,
    ok_dgvm=False,
    dt_days=1.0,
):
    """Apply tree fruit turnover and sapwood-to-heartwood conversion.

    Fortran provenance: ``src_stomate/stomate_turnover.f90``, subroutine
    ``turn``, lines 841-925. PFT parameter provenance:
    ``src_parameters/pft_parameters.f90`` lines 631-633 and 4506-4522.
    """

    biomass = jnp.asarray(biomass)
    turnover = jnp.asarray(turnover)
    age = jnp.asarray(age)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    tau_fruit = jnp.asarray(tau_fruit)
    tau_sap = jnp.asarray(tau_sap)
    tree = is_tree[None, :]

    safe_tau_fruit = jnp.where(is_tree, tau_fruit, 1.0)
    fruit_loss = (
        biomass[:, :, IFRUIT, :]
        * dt_days
        / safe_tau_fruit[None, :, None]
    )
    turnover = turnover.at[:, :, IFRUIT, :].add(jnp.where(tree[:, :, None], fruit_loss, 0.0))
    biomass = biomass.at[:, :, IFRUIT, :].add(jnp.where(tree[:, :, None], -fruit_loss, 0.0))

    hw_old = (
        biomass[:, :, IHEARTABOVE, ICARBON]
        + biomass[:, :, IHEARTBELOW, ICARBON]
        + biomass[:, :, IAGRHRTST, ICARBON]
        + biomass[:, :, IAGRHRTPN, ICARBON]
    )
    sap_parts = jnp.asarray((ISAPABOVE, ISAPBELOW, IAGRSAPST, IAGRSAPPN))
    heart_parts = jnp.asarray((IHEARTABOVE, IHEARTBELOW, IAGRHRTST, IAGRHRTPN))
    safe_tau_sap = jnp.where(is_tree, tau_sap, 1.0)
    sapconv = (
        biomass[:, :, sap_parts, :]
        * dt_days
        / safe_tau_sap[None, :, None, None]
    )
    sap_delta = jnp.where(tree[:, :, None, None], sapconv, 0.0)
    biomass = biomass.at[:, :, sap_parts, :].add(-sap_delta)
    biomass = biomass.at[:, :, heart_parts, :].add(sap_delta)

    hw_new = (
        biomass[:, :, IHEARTABOVE, ICARBON]
        + biomass[:, :, IHEARTBELOW, ICARBON]
        + biomass[:, :, IAGRHRTST, ICARBON]
        + biomass[:, :, IAGRHRTPN, ICARBON]
    )
    age_update = (~ok_dgvm) & tree & (hw_new > 0.0)
    safe_hw_new = jnp.where(age_update, hw_new, 1.0)
    age = jnp.where(
        age_update,
        age * hw_old / safe_hw_new,
        age,
    )
    turnover = turnover.at[:, 0, :, :].set(0.0)
    return biomass, turnover, age


def turnover_step(
    *,
    pft_present,
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
    biomass,
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
    prc_residual=0.5,
    ok_herbivores=False,
    ok_dgvm=False,
    dt_days=1.0,
) -> TurnoverResult:
    """Compose audited local sections of ``stomate_turnover::turn``.

    Fortran provenance: ``src_stomate/stomate_turnover.f90``, subroutine
    ``turn`` lines 169-947, excluding only diagnostic output calls. The
    caller supplies PFT parameter arrays exactly as configured by
    ``src_parameters/pft_parameters.f90``.
    """

    biomass = jnp.asarray(biomass)
    leaf_age = jnp.asarray(leaf_age)
    leaf_frac = jnp.asarray(leaf_frac)
    age = jnp.asarray(age)
    lai = jnp.asarray(lai)
    turnover_time = jnp.asarray(turnover_time)
    pft_present = jnp.asarray(pft_present, dtype=bool)
    veget_max = jnp.asarray(veget_max)

    npts, nvm, nparts, nelements = biomass.shape
    if nparts != NPARTS:
        raise ValueError(f"biomass part axis must have length {NPARTS}")
    if leaf_age.shape != (npts, nvm, NLEAFAGES) or leaf_frac.shape != (npts, nvm, NLEAFAGES):
        raise ValueError("leaf_age and leaf_frac must have shape (npts, nvm, nleafages)")
    if pft_present.shape != (npts, nvm) or veget_max.shape != (npts, nvm):
        raise ValueError("pft_present and veget_max must have shape (npts, nvm)")

    turnover = jnp.zeros_like(biomass)
    leaf_meanage = turnover_leaf_mean_age(leaf_age, leaf_frac)
    senescence, turnover_time_from_senescence, _, _ = turnover_senescence_flags(
        biomass=biomass,
        leaf_meanage=leaf_meanage,
        maxmoiavail_lastyear=maxmoiavail_lastyear,
        minmoiavail_lastyear=minmoiavail_lastyear,
        moiavail_week=moiavail_week,
        t2m_longterm=t2m_longterm,
        t2m_month=t2m_month,
        t2m_week=t2m_week,
        gdd_from_growthinit=gdd_from_growthinit,
        lai=lai,
        nrec=nrec,
        senescence_type=senescence_type,
        is_tree=is_tree,
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
    )
    turnover_time = jnp.where(turnover_time_from_senescence != 0.0, turnover_time_from_senescence, turnover_time)
    biomass, turnover, c_export = turnover_climatic_biomass_loss(
        biomass,
        turnover,
        senescence,
        turnover_time,
        nrec,
        senescence_type=senescence_type,
        is_tree=is_tree,
        ok_laidev=ok_laidev,
        leaffall=leaffall,
        max_turnover_time=max_turnover_time,
        prc_residual=prc_residual,
        dt_days=dt_days,
    )
    biomass, turnover, leaf_frac, leaf_age_crit = turnover_leaf_age_fall(
        biomass,
        turnover,
        leaf_age,
        leaf_frac,
        t2m_longterm,
        is_tree=is_tree,
        natural=natural,
        ok_laidev=ok_laidev,
        leafagecrit=leafagecrit,
        dt_days=dt_days,
    )
    biomass, turnover, leaf_age, leaf_frac, leaf_meanage, lai, _ = turnover_shed_remaining_leaves(
        biomass,
        turnover,
        leaf_age,
        leaf_frac,
        leaf_meanage,
        lai,
        senescence,
        is_tree=is_tree,
        ok_laidev=ok_laidev,
        senescence_type=senescence_type,
        lai_initmin=lai_initmin,
        sla_calc=sla_calc,
    )
    biomass, turnover = turnover_herbivory(
        biomass,
        turnover,
        herbivores,
        is_tree=is_tree,
        ok_herbivores=ok_herbivores,
        dt_days=dt_days,
    )
    biomass, turnover, age = turnover_tree_fruit_and_sapwood(
        biomass,
        turnover,
        age,
        is_tree=is_tree,
        tau_fruit=tau_fruit,
        tau_sap=tau_sap,
        ok_dgvm=ok_dgvm,
        dt_days=dt_days,
    )

    pft_mask = (jnp.arange(nvm) > 0)[None, :] & pft_present
    senescence = jnp.where(pft_mask, senescence, False)
    turnover = jnp.where(pft_mask[:, :, None, None], turnover, 0.0)
    c_export = jnp.where(pft_mask, c_export, 0.0)
    return TurnoverResult(
        turnover=turnover,
        senescence=senescence,
        c_export=c_export,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        age=age,
        lai=lai,
        biomass=biomass,
        turnover_time=turnover_time,
        leaf_meanage=leaf_meanage,
        leaf_age_crit=leaf_age_crit,
    )


def gap_mortality_step(
    *,
    npp_longterm,
    turnover_longterm,
    lm_lastyearmax,
    pft_present,
    biomass,
    ind,
    bm_to_litter,
    t2m_min_daily,
    tmin_spring_time,
    sla_calc,
    natural,
    is_tree,
    pasture,
    availability_fact,
    residence_time,
    tmin_crit,
    leaf_tab,
    pheno_type,
    dt_days=1.0,
    lpj_gap_const_mort=False,
    ok_dgvm=False,
    min_stomate=0.0,
    min_avail=0.01,
    ref_greff=0.035,
    npp_longterm_init=10.0,
    coldness_mort=0.01,
    frost_damage_limit=273.15,
    spring_days_max=40,
) -> GapMortalityResult:
    """Apply ``lpj_gap::gap`` mortality and biomass transfer to litter.

    Fortran provenance: ``src_stomate/lpj_gap.f90``, subroutine ``gap``,
    lines 117-364. PFT parameter provenance:
    ``src_parameters/pft_parameters.f90`` lines 530, 658-659, 4161 and
    ``src_parameters/constantes_mtc.f90`` lines 178-181 and 786-793.
    Scalar constant provenance: ``src_parameters/constantes_var.f90`` lines
    155, 976-1009.

    ``mortality_fraction`` is the per-step fraction used to update biomass
    before Fortran line 355 divides by ``dt``. ``mortality`` matches the
    subroutine output after that final division.
    """

    npp_longterm = jnp.asarray(npp_longterm)
    turnover_longterm = jnp.asarray(turnover_longterm)
    lm_lastyearmax = jnp.asarray(lm_lastyearmax)
    pft_present = jnp.asarray(pft_present, dtype=bool)
    biomass = jnp.asarray(biomass)
    ind = jnp.asarray(ind)
    bm_to_litter = jnp.asarray(bm_to_litter)
    t2m_min_daily = jnp.asarray(t2m_min_daily)
    tmin_spring_time = jnp.asarray(tmin_spring_time)
    sla_calc = jnp.asarray(sla_calc)
    natural = jnp.asarray(natural, dtype=bool)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    pasture = jnp.asarray(pasture, dtype=bool)
    availability_fact = jnp.asarray(availability_fact)
    residence_time = jnp.asarray(residence_time)
    tmin_crit = jnp.asarray(tmin_crit)
    leaf_tab = jnp.asarray(leaf_tab)
    pheno_type = jnp.asarray(pheno_type)

    if biomass.ndim != 4 or biomass.shape[2] != NPARTS:
        raise ValueError(f"biomass must have shape (npts, nvm, {NPARTS}, nelements)")
    if bm_to_litter.shape != biomass.shape:
        raise ValueError("bm_to_litter must have the same shape as biomass")
    npts, nvm, _, _ = biomass.shape
    if npp_longterm.shape != (npts, nvm) or pft_present.shape != (npts, nvm):
        raise ValueError("npp_longterm and pft_present must have shape (npts, nvm)")

    active_pft = jnp.arange(nvm) > 0
    active = active_pft[None, :] & natural[None, :] & pft_present
    tree_active = active & is_tree[None, :]
    grass_active = active & (~is_tree[None, :])

    turnover_sum = (
        turnover_longterm[:, :, ILEAF, ICARBON]
        + turnover_longterm[:, :, IROOT, ICARBON]
        + turnover_longterm[:, :, IFRUIT, ICARBON]
        + turnover_longterm[:, :, ISAPABOVE, ICARBON]
        + turnover_longterm[:, :, ISAPBELOW, ICARBON]
        + turnover_longterm[:, :, IAGRSAPST, ICARBON]
        + turnover_longterm[:, :, IAGRSAPPN, ICARBON]
    )
    valid_vigour = tree_active & (lm_lastyearmax > min_stomate)
    delta_biomass = jnp.where(valid_vigour, jnp.maximum(npp_longterm - turnover_sum, 0.0), 0.0)
    safe_vigour_denominator = jnp.where(
        valid_vigour,
        lm_lastyearmax * sla_calc,
        1.0,
    )
    vigour = jnp.where(
        valid_vigour,
        delta_biomass / safe_vigour_denominator,
        0.0,
    )
    availability = availability_fact[None, :] / (1.0 + ref_greff * vigour)

    mortality_fraction = jnp.zeros((npts, nvm), dtype=biomass.dtype)
    if lpj_gap_const_mort:
        safe_residence_time = jnp.where(
            tree_active,
            residence_time[None, :],
            1.0,
        )
        constant = dt_days / (safe_residence_time * ONE_YEAR_DAYS)
        mortality_fraction = jnp.where(tree_active, constant, mortality_fraction)
    else:
        growth = jnp.maximum(min_avail, availability) * dt_days / ONE_YEAR_DAYS
        mortality_fraction = jnp.where(tree_active, growth, mortality_fraction)

    dgvm_low_npp = pft_present & (npp_longterm < (npp_longterm_init - 1.0))
    mortality_fraction = jnp.where(ok_dgvm & tree_active & dgvm_low_npp, 1.0, mortality_fraction)

    has_tmin_crit = jnp.isfinite(tmin_crit)
    frost_sensitive = ok_dgvm & tree_active & has_tmin_crit[None, :] & (t2m_min_daily[:, None] < tmin_crit[None, :])
    frost_added = coldness_mort * (tmin_crit[None, :] - t2m_min_daily[:, None]) + mortality_fraction
    mortality_fraction = jnp.where(frost_sensitive, jnp.minimum(1.0, frost_added), mortality_fraction)

    spring_sensitive = (
        ok_dgvm
        & tree_active
        & (leaf_tab[None, :] == 1)
        & (pheno_type[None, :] == 2)
        & (tmin_spring_time > 0.0)
        & (tmin_spring_time < spring_days_max + 1)
        & (t2m_min_daily[:, None] < frost_damage_limit)
    )
    spring_added = (
        0.01
        * (frost_damage_limit - t2m_min_daily[:, None])
        * tmin_spring_time
        / spring_days_max
        + mortality_fraction
    )
    mortality_fraction = jnp.where(spring_sensitive, jnp.minimum(1.0, spring_added), mortality_fraction)

    grass_dgvm = ok_dgvm & grass_active & natural[None, :] & (~pasture[None, :]) & dgvm_low_npp
    mortality_fraction = jnp.where(grass_dgvm, 1.0, mortality_fraction)
    mortality_fraction = jnp.where(active, mortality_fraction, 0.0)

    dmortality = mortality_fraction[:, :, None, None] * biomass
    bm_to_litter = bm_to_litter + dmortality
    biomass = biomass - dmortality
    ind = jnp.where(ok_dgvm & pft_present, ind * (1.0 - mortality_fraction), ind)

    mortality = mortality_fraction / dt_days
    mortality = mortality.at[:, 0].set(0.0)
    mortality_fraction = mortality_fraction.at[:, 0].set(0.0)
    return GapMortalityResult(
        biomass=biomass,
        ind=ind,
        bm_to_litter=bm_to_litter,
        mortality=mortality,
        mortality_fraction=mortality_fraction,
        delta_biomass=delta_biomass,
        vigour=vigour,
        availability=availability,
    )


def kill_pfts_step(
    *,
    lm_lastyearmax,
    ind,
    pft_present,
    cn_ind,
    biomass,
    senescence,
    rip_time,
    age,
    leaf_age,
    leaf_frac,
    npp_longterm,
    when_growthinit,
    everywhere,
    veget_max,
    bm_to_litter,
    natural,
    pasture,
    is_tree,
    ok_dgvm=False,
    lpj_gap_const_mort=False,
    min_stomate=0.0,
    large_value=1.0e33,
) -> KillResult:
    """Apply ``lpj_kill::kill`` PFT elimination bookkeeping.

    Fortran provenance: ``src_stomate/lpj_kill.f90``, subroutine ``kill``,
    lines 63-272. Constants provenance:
    ``src_parameters/constantes_var.f90`` lines 177-178.

    The Fortran ``lai`` argument is intentionally absent because lines 238-240
    leave the LAI reset commented out; this kernel does not mutate LAI.
    """

    lm_lastyearmax = jnp.asarray(lm_lastyearmax)
    ind = jnp.asarray(ind)
    pft_present = jnp.asarray(pft_present, dtype=bool)
    cn_ind = jnp.asarray(cn_ind)
    biomass = jnp.asarray(biomass)
    senescence = jnp.asarray(senescence, dtype=bool)
    rip_time = jnp.asarray(rip_time)
    age = jnp.asarray(age)
    leaf_age = jnp.asarray(leaf_age)
    leaf_frac = jnp.asarray(leaf_frac)
    npp_longterm = jnp.asarray(npp_longterm)
    when_growthinit = jnp.asarray(when_growthinit)
    everywhere = jnp.asarray(everywhere)
    veget_max = jnp.asarray(veget_max)
    bm_to_litter = jnp.asarray(bm_to_litter)
    natural = jnp.asarray(natural, dtype=bool)
    pasture = jnp.asarray(pasture, dtype=bool)
    is_tree = jnp.asarray(is_tree, dtype=bool)

    if biomass.ndim != 4 or biomass.shape[2] != NPARTS:
        raise ValueError(f"biomass must have shape (npts, nvm, {NPARTS}, nelements)")
    if bm_to_litter.shape != biomass.shape:
        raise ValueError("bm_to_litter must have the same shape as biomass")
    npts, nvm, _, _ = biomass.shape
    if leaf_age.shape != (npts, nvm, NLEAFAGES) or leaf_frac.shape != (npts, nvm, NLEAFAGES):
        raise ValueError("leaf_age and leaf_frac must have shape (npts, nvm, nleafages)")

    active_pft = jnp.arange(nvm) > 0
    kill_candidate = active_pft[None, :] & natural[None, :] & (~pasture[None, :]) & pft_present
    dgvm_kill = kill_candidate & ((ind < min_stomate) | (lm_lastyearmax < min_stomate))
    stomate_kill = kill_candidate & (
        (biomass[:, :, ICARBRES, ICARBON] <= 0.0)
        | (biomass[:, :, IROOT, ICARBON] < -min_stomate)
        | (biomass[:, :, ILEAF, ICARBON] < -min_stomate)
    ) & (ind > 0.0)
    was_killed = jnp.where(ok_dgvm, dgvm_kill, stomate_kill)

    npp_longterm = jnp.where(
        was_killed & (~is_tree[None, :]) & (~lpj_gap_const_mort),
        500.0,
        npp_longterm,
    )

    bm_to_litter = bm_to_litter + jnp.where(was_killed[:, :, None, None], biomass, 0.0)
    biomass = jnp.where(was_killed[:, :, None, None], 0.0, biomass)

    pft_present = jnp.where(ok_dgvm & was_killed, False, pft_present)
    veget_max = jnp.where(ok_dgvm & was_killed, 0.0, veget_max)
    rip_time = jnp.where(ok_dgvm & was_killed, 0.0, rip_time)

    ind = jnp.where(was_killed, 0.0, ind)
    cn_ind = jnp.where(was_killed, 0.0, cn_ind)
    senescence = jnp.where(was_killed, False, senescence)
    age = jnp.where(was_killed, 0.0, age)
    when_growthinit = jnp.where(was_killed, large_value, when_growthinit)
    everywhere = jnp.where(was_killed, 0.0, everywhere)
    leaf_age = jnp.where(was_killed[:, :, None], 0.0, leaf_age)
    leaf_frac = jnp.where(was_killed[:, :, None], 0.0, leaf_frac)
    was_killed = was_killed.at[:, 0].set(False)

    return KillResult(
        pft_present=pft_present,
        cn_ind=cn_ind,
        ind=ind,
        biomass=biomass,
        senescence=senescence,
        rip_time=rip_time,
        age=age,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        npp_longterm=npp_longterm,
        when_growthinit=when_growthinit,
        everywhere=everywhere,
        veget_max=veget_max,
        bm_to_litter=bm_to_litter,
        was_killed=was_killed,
    )


def crown_woodmass_ind(biomass, ind, veget_max, *, ok_dgvm=False, min_stomate=0.0):
    """Compute individual wood mass before ``lpj_crown::crown``.

    Fortran provenance: ``src_stomate/stomate_lpj.f90`` lines 1150-1170.
    In DGVM mode the biomass sum is multiplied by ``veget_max`` before
    division by ``ind``; outside DGVM it is not.
    """

    biomass = jnp.asarray(biomass)
    ind = jnp.asarray(ind)
    veget_max = jnp.asarray(veget_max)
    wood = (
        biomass[:, :, ISAPABOVE, ICARBON]
        + biomass[:, :, ISAPBELOW, ICARBON]
        + biomass[:, :, IHEARTABOVE, ICARBON]
        + biomass[:, :, IHEARTBELOW, ICARBON]
        + biomass[:, :, IAGRSAPST, ICARBON]
        + biomass[:, :, IAGRSAPPN, ICARBON]
        + biomass[:, :, IAGRHRTST, ICARBON]
        + biomass[:, :, IAGRHRTPN, ICARBON]
    )
    numerator = jnp.where(ok_dgvm, wood * veget_max, wood)
    return jnp.where(ind > min_stomate, numerator / ind, 0.0)


def crown_step(
    *,
    pft_present,
    ind,
    biomass,
    woodmass_ind,
    veget_max,
    height,
    is_tree,
    natural,
    maxdia,
    pipe_tune1=100.0,
    pipe_tune2=40.0,
    pipe_tune3=0.5,
    pipe_density=2.0e5,
    pipe_tune_exp_coeff=1.6,
    min_stomate=0.0,
) -> CrownResult:
    """Apply ``lpj_crown::crown`` to compute crown area and height.

    Fortran provenance: ``src_stomate/lpj_crown.f90``, subroutine ``crown``,
    lines 78-201. Pipe constants provenance:
    ``src_parameters/constantes_var.f90`` lines 1087-1100.

    The commented Fortran veget_max update at lines 187-197 is intentionally
    not applied.
    """

    pft_present = jnp.asarray(pft_present, dtype=bool)
    ind = jnp.asarray(ind)
    biomass = jnp.asarray(biomass)
    woodmass_ind = jnp.asarray(woodmass_ind)
    veget_max = jnp.asarray(veget_max)
    height = jnp.asarray(height)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    natural = jnp.asarray(natural, dtype=bool)
    maxdia = jnp.asarray(maxdia)

    npts, nvm = pft_present.shape
    if biomass.shape[:2] != (npts, nvm):
        raise ValueError("biomass must share pft_present (npts, nvm)")

    cn_ind = jnp.zeros((npts, nvm), dtype=woodmass_ind.dtype)
    active_pft = jnp.arange(nvm) > 0
    tree_mask = active_pft[None, :] & pft_present & is_tree[None, :] & natural[None, :] & (woodmass_ind > min_stomate)
    dia = (woodmass_ind / (pipe_density * jnp.pi / 4.0 * pipe_tune2)) ** (1.0 / (2.0 + pipe_tune3))
    new_height = pipe_tune2 * (dia**pipe_tune3)
    tree_cn = pipe_tune1 * jnp.minimum(dia, maxdia[None, :]) ** pipe_tune_exp_coeff
    height = jnp.where(tree_mask, new_height, height)
    cn_ind = jnp.where(tree_mask, tree_cn, cn_ind)

    grass_mask = active_pft[None, :] & pft_present & (~is_tree[None, :])
    cn_ind = jnp.where(grass_mask, 1.0, cn_ind)
    cn_ind = cn_ind.at[:, 0].set(0.0)
    return CrownResult(
        woodmass_ind=woodmass_ind,
        cn_ind=cn_ind,
        height=height,
        veget_max=veget_max,
    )


def light_competition_step(
    *,
    veget_max,
    fpc_max,
    pft_present,
    cn_ind,
    lai,
    maxfpc_lastyear,
    lm_lastyearmax,
    ind,
    biomass,
    veget_lastlight,
    bm_to_litter,
    mortality,
    sla_calc,
    natural,
    pasture,
    is_tree,
    ext_coeff,
    dt_days=1.0,
    ok_dgvm=False,
    fpc_crit=0.95,
    min_cover=0.05,
    min_stomate=0.0,
    annual_increase=True,
    val_exp=999999.0,
) -> LightCompetitionResult:
    """Apply ``lpj_light::light`` competition.

    Fortran provenance: ``src_stomate/lpj_light.f90``, subroutine ``light``,
    lines 103-648. Scalar constants provenance:
    ``src_parameters/constantes_var.f90`` lines 986-993 and 1108-1109.
    PFT parameter provenance for ``ext_coeff``:
    ``src_parameters/pft_parameters.f90`` lines 337-338 and 3416.
    """

    veget_max = jnp.asarray(veget_max)
    fpc_max = jnp.asarray(fpc_max)
    pft_present = jnp.asarray(pft_present, dtype=bool)
    cn_ind = jnp.asarray(cn_ind)
    lai = jnp.asarray(lai)
    maxfpc_lastyear = jnp.asarray(maxfpc_lastyear)
    lm_lastyearmax = jnp.asarray(lm_lastyearmax)
    ind = jnp.asarray(ind)
    biomass = jnp.asarray(biomass)
    veget_lastlight = jnp.asarray(veget_lastlight)
    bm_to_litter = jnp.asarray(bm_to_litter)
    mortality = jnp.asarray(mortality)
    sla_calc = jnp.asarray(sla_calc)
    natural = jnp.asarray(natural, dtype=bool)
    pasture = jnp.asarray(pasture, dtype=bool)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    ext_coeff = jnp.asarray(ext_coeff)

    if biomass.shape != bm_to_litter.shape:
        raise ValueError("biomass and bm_to_litter must have the same shape")
    npts, nvm, _, _ = biomass.shape
    fpc_nat = jnp.zeros((npts, nvm), dtype=biomass.dtype).at[:, 0].set(1.0)
    light_death = jnp.zeros((npts, nvm), dtype=biomass.dtype)

    if ok_dgvm:
        fracnat = jnp.ones((npts,), dtype=biomass.dtype)
        for pft in range(1, nvm):
            if (not bool(natural[pft])) or bool(pasture[pft]):
                fracnat = fracnat - veget_max[:, pft]

        for pft in range(1, nvm):
            if bool(natural[pft]) and not bool(pasture[pft]):
                mask = fracnat >= min_stomate
                fpc_val = cn_ind[:, pft] * ind[:, pft] / jnp.where(mask, fracnat, 1.0)
                fpc_nat = fpc_nat.at[:, pft].set(jnp.where(mask, fpc_val, fpc_nat[:, pft]))
            else:
                fpc_nat = fpc_nat.at[:, pft].set(0.0)

        sumfpc = jnp.sum(fpc_nat[:, 1:], axis=1)
        for point in range(npts):
            if bool(sumfpc[point] > fpc_crit):
                if annual_increase:
                    deltafpc = jnp.maximum(fpc_nat[point, :] - maxfpc_lastyear[point, :], 0.0)
                else:
                    deltafpc = jnp.maximum(fpc_nat[point, :] - veget_lastlight[point, :], 0.0)

                sumfpc_wood = 0.0
                sumdelta_fpc_wood = 0.0
                sumfpc_grass = 0.0
                for pft in range(1, nvm):
                    if bool(natural[pft]) and not bool(pasture[pft]):
                        if bool(is_tree[pft]):
                            sumfpc_wood = sumfpc_wood + fpc_nat[point, pft]
                            sumdelta_fpc_wood = sumdelta_fpc_wood + deltafpc[pft]
                        else:
                            sumfpc_grass = sumfpc_grass + fpc_nat[point, pft]

                survive = jnp.ones((nvm,), dtype=biomass.dtype)
                for pft in range(1, nvm):
                    if bool(pft_present[point, pft]) and bool(natural[pft]) and not bool(pasture[pft]):
                        if bool(is_tree[pft]):
                            if bool((sumfpc_wood >= fpc_crit) & (fpc_nat[point, pft] > min_stomate) & (sumdelta_fpc_wood > min_stomate)):
                                kept = (
                                    fpc_nat[point, pft]
                                    - (sumfpc_wood - fpc_crit) * deltafpc[pft] / sumdelta_fpc_wood
                                ) / fpc_nat[point, pft]
                                reduct = 1.0 - jnp.minimum(kept, 1.0)
                            else:
                                reduct = 0.0
                        else:
                            fpc_wood_limit = jnp.minimum(fpc_crit, sumfpc_wood)
                            if bool((sumfpc_grass >= 1.0 - fpc_wood_limit) & (sumfpc_grass >= min_stomate)):
                                reduct = (
                                    (sumfpc_grass - 1.0 + fpc_wood_limit)
                                    * fpc_nat[point, pft]
                                    / sumfpc_grass
                                )
                            else:
                                reduct = 0.0
                        survive = survive.at[pft].set(1.0 - reduct)

                for pft in range(1, nvm):
                    if bool(pft_present[point, pft]) and bool(natural[pft]) and not bool(pasture[pft]):
                        loss_fraction = 1.0 - survive[pft]
                        bm_to_litter = bm_to_litter.at[point, pft, :, :].add(biomass[point, pft, :, :] * loss_fraction)
                        biomass = biomass.at[point, pft, :, :].set(biomass[point, pft, :, :] * survive[pft])
                        ind = ind.at[point, pft].set(ind[point, pft] * survive[pft])
                        light_death = light_death.at[point, pft].set(loss_fraction / dt_days)

        for pft in range(1, nvm):
            if bool(natural[pft]) and not bool(pasture[pft]):
                if bool(is_tree[pft]):
                    fpc_factor = jnp.where(
                        lai[:, pft] == val_exp,
                        1.0,
                        jnp.maximum(
                            1.0 - jnp.exp(-lm_lastyearmax[:, pft] * sla_calc[:, pft] * ext_coeff[pft]),
                            min_cover,
                        ),
                    )
                else:
                    fpc_factor = jnp.where(
                        lai[:, pft] == val_exp,
                        1.0,
                        1.0 - jnp.exp(-lm_lastyearmax[:, pft] * sla_calc[:, pft] * ext_coeff[pft]),
                    )
                veget_lastlight = veget_lastlight.at[:, pft].set(cn_ind[:, pft] * ind[:, pft] * fpc_factor)
            else:
                veget_lastlight = veget_lastlight.at[:, pft].set(0.0)
    else:
        for pft in range(1, nvm):
            if bool(natural[pft]) and not bool(pasture[pft]):
                cover = ind[:, pft] * cn_ind[:, pft]
                lai_ind = jnp.where(
                    cover > min_stomate,
                    sla_calc[:, pft] * lm_lastyearmax[:, pft] / cover,
                    0.0,
                )
                fpc_pft = cover * jnp.maximum(1.0 - jnp.exp(-ext_coeff[pft] * lai_ind), min_cover)
                death = jnp.where(fpc_pft > fpc_max[:, pft], jnp.minimum(1.0, 1.0 - fpc_max[:, pft] / fpc_pft), 0.0)
                bm_to_litter = bm_to_litter.at[:, pft, :, :].add(death[:, None, None] * biomass[:, pft, :, :])
                biomass = biomass.at[:, pft, :, :].add(-death[:, None, None] * biomass[:, pft, :, :])
                ind = ind.at[:, pft].add(-death * ind[:, pft])
                light_death = light_death.at[:, pft].set(death / dt_days)
                fpc_nat = fpc_nat.at[:, pft].set(fpc_pft)

    light_death = light_death.at[:, 0].set(0.0)
    mortality = mortality.at[:, 0].set(0.0)
    return LightCompetitionResult(
        ind=ind,
        biomass=biomass,
        veget_lastlight=veget_lastlight,
        bm_to_litter=bm_to_litter,
        mortality=mortality,
        light_death=light_death,
        fpc_nat=fpc_nat,
    )


def pftinout_step(
    *,
    adapted,
    regenerate,
    neighbours,
    veget_max,
    biomass,
    ind,
    cn_ind,
    age,
    leaf_frac,
    npp_longterm,
    lm_lastyearmax,
    senescence,
    pft_present,
    everywhere,
    when_growthinit,
    need_adjacent,
    rip_time,
    co2_to_bm,
    sla_calc,
    bm_sapl,
    natural,
    pasture,
    is_tree,
    ext_coeff,
    dt_days=1.0,
    treat_expansion=False,
    min_stomate=0.0,
    min_avail=0.01,
    fpc_crit=0.95,
    adapted_crit=1.0 - 1.0 / jnp.e,
    regenerate_crit=1.0 / jnp.e,
    rip_time_min=1.25,
    ind_0=0.02,
    npp_longterm_init=10.0,
    everywhere_init=0.05,
    large_value=1.0e33,
) -> PftInoutResult:
    """Introduce and eliminate PFTs from a pixel.

    Fortran provenance: ``src_stomate/lpj_pftinout.f90``, subroutine
    ``pftinout`` lines 100-506. This covers woody FPC/availability lines
    211-243, ``RIP_time`` update line 247, agricultural grass initialization
    lines 251-288, natural-PFT elimination lines 294-310, introduction and
    neighbour gating lines 314-484, and ``need_adjacent`` writeback lines
    490-498. It does not implement the agricultural-tree fatal branch except
    to reject it explicitly.
    """

    adapted = jnp.asarray(adapted)
    regenerate = jnp.asarray(regenerate)
    neighbours = jnp.asarray(neighbours)
    veget_max = jnp.asarray(veget_max)
    biomass = jnp.asarray(biomass)
    ind = jnp.asarray(ind)
    cn_ind = jnp.asarray(cn_ind)
    age = jnp.asarray(age)
    leaf_frac = jnp.asarray(leaf_frac)
    npp_longterm = jnp.asarray(npp_longterm)
    lm_lastyearmax = jnp.asarray(lm_lastyearmax)
    senescence = jnp.asarray(senescence, dtype=bool)
    pft_present = jnp.asarray(pft_present, dtype=bool)
    everywhere = jnp.asarray(everywhere)
    when_growthinit = jnp.asarray(when_growthinit)
    need_adjacent = jnp.asarray(need_adjacent, dtype=bool)
    rip_time = jnp.asarray(rip_time)
    co2_to_bm = jnp.asarray(co2_to_bm)
    sla_calc = jnp.asarray(sla_calc)
    bm_sapl = jnp.asarray(bm_sapl)
    natural = jnp.asarray(natural, dtype=bool)
    pasture = jnp.asarray(pasture, dtype=bool)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    ext_coeff = jnp.asarray(ext_coeff)

    npts, nvm = veget_max.shape
    if neighbours.shape != (npts, 8):
        raise ValueError("lpj_pftinout expects NbNeighb == 8 with neighbours shaped (npts, 8)")
    for name, arr in (
        ("adapted", adapted),
        ("regenerate", regenerate),
        ("ind", ind),
        ("cn_ind", cn_ind),
        ("age", age),
        ("npp_longterm", npp_longterm),
        ("lm_lastyearmax", lm_lastyearmax),
        ("senescence", senescence),
        ("pft_present", pft_present),
        ("everywhere", everywhere),
        ("when_growthinit", when_growthinit),
        ("need_adjacent", need_adjacent),
        ("rip_time", rip_time),
        ("co2_to_bm", co2_to_bm),
        ("sla_calc", sla_calc),
    ):
        if arr.shape != (npts, nvm):
            raise ValueError(f"{name} must have shape (npts, nvm)")
    if biomass.ndim != 4 or biomass.shape[:3] != (npts, nvm, NPARTS):
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    if bm_sapl.shape[0] != nvm or bm_sapl.shape[1] != NPARTS or bm_sapl.shape[2] != biomass.shape[3]:
        raise ValueError("bm_sapl must have shape (nvm, nparts, nelements)")
    if leaf_frac.shape != (npts, nvm, NLEAFAGES):
        raise ValueError("leaf_frac must have shape (npts, nvm, nleafages)")
    for name, arr in (("natural", natural), ("pasture", pasture), ("is_tree", is_tree), ("ext_coeff", ext_coeff)):
        if arr.shape != (nvm,):
            raise ValueError(f"{name} must have shape (nvm,)")

    agri_tree = tuple(pft for pft in range(1, nvm) if bool((~natural[pft] | pasture[pft]) & is_tree[pft]))
    if agri_tree:
        raise NotImplementedError("lpj_pftinout agricultural-tree branch is a Fortran fatal error and is not supported")

    fracnat = jnp.ones((npts,), dtype=veget_max.dtype)
    for pft in range(1, nvm):
        if (not bool(natural[pft])) or bool(pasture[pft]):
            fracnat = fracnat - veget_max[:, pft]

    sumfrac_wood = jnp.zeros((npts,), dtype=veget_max.dtype)
    for pft in range(1, nvm):
        if bool(natural[pft]) and bool(is_tree[pft]):
            fpc_term = cn_ind[:, pft] * ind[:, pft] / jnp.where(fracnat > min_stomate, fracnat, 1.0)
            fpc_term = fpc_term * (1.0 - jnp.exp(-lm_lastyearmax[:, pft] * sla_calc[:, pft] * ext_coeff[pft]))
            sumfrac_wood = sumfrac_wood + jnp.where(fracnat > min_stomate, fpc_term, 0.0)

    avail_grass = jnp.maximum(1.0 - sumfrac_wood, min_avail)
    avail_tree = jnp.maximum(fpc_crit - sumfrac_wood, min_avail)
    rip_time = rip_time + dt_days / ONE_YEAR_DAYS

    for pft in range(1, nvm):
        if (not bool(natural[pft])) or bool(pasture[pft]):
            init_agri = (veget_max[:, pft] > min_stomate) & (~pft_present[:, pft])
            ind = ind.at[:, pft].set(jnp.where(init_agri, veget_max[:, pft], ind[:, pft]))
            sapling = bm_sapl[pft, :, :] * ind[:, pft, None, None] / jnp.where(
                veget_max[:, pft, None, None] > min_stomate,
                veget_max[:, pft, None, None],
                1.0,
            )
            biomass = biomass.at[:, pft, :, :].set(jnp.where(init_agri[:, None, None], sapling, biomass[:, pft, :, :]))
            co2_to_bm = co2_to_bm.at[:, pft].add(jnp.where(init_agri, jnp.sum(sapling[:, :, ICARBON], axis=1) / dt_days, 0.0))
            pft_present = pft_present.at[:, pft].set(jnp.where(init_agri, True, pft_present[:, pft]))
            everywhere = everywhere.at[:, pft].set(jnp.where(init_agri, 1.0, everywhere[:, pft]))
            senescence = senescence.at[:, pft].set(jnp.where(init_agri, False, senescence[:, pft]))
            age = age.at[:, pft].set(jnp.where(init_agri, 0.0, age[:, pft]))

    for pft in range(1, nvm):
        if bool(natural[pft]) and not bool(pasture[pft]):
            kill_climate = pft_present[:, pft] & (adapted[:, pft] < adapted_crit)
            ind = ind.at[:, pft].set(jnp.where(kill_climate, 0.0, ind[:, pft]))

    can_introduce = jnp.zeros((npts, nvm), dtype=bool)
    cardinal_neighbour_cols = (0, 2, 4, 6)
    for pft in range(1, nvm):
        if bool(natural[pft]) and not bool(pasture[pft]):
            climate_ok = (~pft_present[:, pft]) & (adapted[:, pft] > adapted_crit) & (regenerate[:, pft] > regenerate_crit)
            if bool(jnp.any(need_adjacent[:, pft])):
                neighbour_present = jnp.zeros((npts,), dtype=bool)
                for point in range(npts):
                    n_present = 0
                    for col in cardinal_neighbour_cols:
                        neigh = int(neighbours[point, col])
                        if neigh > 0 and bool(everywhere[neigh - 1, pft] >= 1.0 - min_stomate):
                            n_present += 1
                    neighbour_present = neighbour_present.at[point].set(n_present > 0)
                intro = climate_ok & jnp.where(need_adjacent[:, pft], neighbour_present, True)
            else:
                intro = climate_ok
            intro = intro & (rip_time[:, pft] >= rip_time_min)
            can_introduce = can_introduce.at[:, pft].set(intro)
            avail = jnp.where(bool(is_tree[pft]), avail_tree, avail_grass)
            new_ind = ind_0 * (dt_days / ONE_YEAR_DAYS) * avail
            ind = ind.at[:, pft].set(jnp.where(intro, new_ind, ind[:, pft]))

            sapling_no_cover = bm_sapl[pft, :, :] * new_ind[:, None, None]
            sapling_with_cover = bm_sapl[pft, :, :] * new_ind[:, None, None] / jnp.where(
                veget_max[:, pft, None, None] > min_stomate,
                veget_max[:, pft, None, None],
                1.0,
            )
            sapling = jnp.where(veget_max[:, pft, None, None] > min_stomate, sapling_with_cover, sapling_no_cover)
            biomass = biomass.at[:, pft, :, :].set(jnp.where(intro[:, None, None], sapling, biomass[:, pft, :, :]))
            co2_to_bm = co2_to_bm.at[:, pft].add(jnp.where(intro, jnp.sum(sapling[:, :, ICARBON], axis=1) / dt_days, 0.0))
            pft_present = pft_present.at[:, pft].set(jnp.where(intro, True, pft_present[:, pft]))
            senescence = senescence.at[:, pft].set(jnp.where(intro, False, senescence[:, pft]))
            when_growthinit = when_growthinit.at[:, pft].set(jnp.where(intro, large_value, when_growthinit[:, pft]))
            age = age.at[:, pft].set(jnp.where(intro, 0.0, age[:, pft]))
            leaf_frac = leaf_frac.at[:, pft, 0].set(jnp.where(intro, 1.0, leaf_frac[:, pft, 0]))
            npp_longterm = npp_longterm.at[:, pft].set(jnp.where(intro, npp_longterm_init, npp_longterm[:, pft]))
            lm_lastyearmax = lm_lastyearmax.at[:, pft].set(jnp.where(intro, bm_sapl[pft, ILEAF, ICARBON] * new_ind, lm_lastyearmax[:, pft]))
            everywhere_value = everywhere_init if treat_expansion else 1.0
            everywhere = everywhere.at[:, pft].set(jnp.where(intro, everywhere_value, everywhere[:, pft]))

    need_adjacent = jnp.where(pft_present, False, need_adjacent)

    return PftInoutResult(
        veget_max=veget_max,
        biomass=biomass,
        ind=ind,
        age=age,
        leaf_frac=leaf_frac,
        npp_longterm=npp_longterm,
        lm_lastyearmax=lm_lastyearmax,
        senescence=senescence,
        pft_present=pft_present,
        everywhere=everywhere,
        when_growthinit=when_growthinit,
        need_adjacent=need_adjacent,
        rip_time=rip_time,
        co2_to_bm=co2_to_bm,
        avail_tree=avail_tree,
        avail_grass=avail_grass,
        can_introduce=can_introduce,
    )


def establishment_rates_step(
    *,
    veget_max,
    pft_present,
    regenerate,
    neighbours,
    resolution,
    need_adjacent,
    herbivores,
    precip_annual,
    gdd0,
    lm_lastyearmax,
    cn_ind,
    lai,
    avail_tree,
    avail_grass,
    npp_longterm,
    ind,
    everywhere,
    sla_calc,
    natural,
    pasture,
    is_tree,
    ext_coeff,
    dt_days=1.0,
    ok_dgvm=False,
    ok_herbivores=False,
    treat_expansion=False,
    migrate=None,
    min_stomate=0.0,
    val_exp=999999.0,
    estab_max_tree=0.12,
    estab_max_grass=0.12,
    establish_scal_fact=5.0,
    max_tree_coverage=0.98,
    ind_0_estab=0.2,
    precip_crit=100.0,
    gdd_crit_estab=150.0,
    regenerate_crit=1.0 / jnp.e,
    min_cover=0.05,
) -> EstablishmentRatesResult:
    """Compute ``lpj_establish::establish`` establishment rates.

    This covers the establishment-rate sections before sapling biomass
    accounting. Fortran provenance: ``src_stomate/lpj_establish.f90``,
    subroutine ``establish``, lines 100-568 and 838-845. Constants come from
    ``src_parameters/constantes_var.f90`` lines 473-474, 922-938, and
    1104-1109, plus ``src_stomate/stomate_data.f90`` line 70.
    """

    veget_max = jnp.asarray(veget_max)
    pft_present = jnp.asarray(pft_present, dtype=bool)
    regenerate = jnp.asarray(regenerate)
    neighbours = jnp.asarray(neighbours)
    resolution = jnp.asarray(resolution)
    need_adjacent = jnp.asarray(need_adjacent, dtype=bool)
    herbivores = jnp.asarray(herbivores)
    precip_annual = jnp.asarray(precip_annual)
    gdd0 = jnp.asarray(gdd0)
    lm_lastyearmax = jnp.asarray(lm_lastyearmax)
    cn_ind = jnp.asarray(cn_ind)
    lai = jnp.asarray(lai)
    avail_tree = jnp.asarray(avail_tree)
    avail_grass = jnp.asarray(avail_grass)
    npp_longterm = jnp.asarray(npp_longterm)
    ind = jnp.asarray(ind)
    everywhere = jnp.asarray(everywhere)
    sla_calc = jnp.asarray(sla_calc)
    natural = jnp.asarray(natural, dtype=bool)
    pasture = jnp.asarray(pasture, dtype=bool)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    ext_coeff = jnp.asarray(ext_coeff)

    npts, nvm = ind.shape
    if neighbours.shape != (npts, 8):
        raise ValueError("lpj_establish expects NbNeighb == 8 with neighbours shaped (npts, 8)")
    if resolution.shape != (npts, 2):
        raise ValueError("resolution must have shape (npts, 2)")

    fpc_nat = jnp.zeros((npts, nvm), dtype=ind.dtype)
    sumfpc = jnp.zeros((npts,), dtype=ind.dtype)
    sumfpc_wood = jnp.zeros((npts,), dtype=ind.dtype)
    spacefight_grass = jnp.zeros((npts,), dtype=ind.dtype)
    d_ind = jnp.zeros((npts, nvm), dtype=ind.dtype)
    estab_rate_max_tree_arr = jnp.zeros((npts,), dtype=ind.dtype)
    estab_rate_max_grass_arr = jnp.zeros((npts,), dtype=ind.dtype)
    fracnat = jnp.ones((npts,), dtype=ind.dtype)

    if ok_dgvm:
        for pft in range(1, nvm):
            if (not bool(natural[pft])) or bool(pasture[pft]):
                fracnat = fracnat - veget_max[:, pft]

        for pft in range(1, nvm):
            if bool(natural[pft]) and not bool(pasture[pft]):
                fpc_no_lai = cn_ind[:, pft] * ind[:, pft] / jnp.where(fracnat > min_stomate, fracnat, 1.0)
                lai_factor = 1.0 - jnp.exp(-lm_lastyearmax[:, pft] * sla_calc[:, pft] * ext_coeff[pft])
                fpc_with_lai = fpc_no_lai * lai_factor
                fpc_pft = jnp.where(lai[:, pft] == val_exp, fpc_no_lai, fpc_with_lai)
                fpc_pft = jnp.where(fracnat > min_stomate, fpc_pft, 0.0)
                fpc_nat = fpc_nat.at[:, pft].set(fpc_pft)
                sumfpc = sumfpc + jnp.where(pft_present[:, pft], fpc_pft, 0.0)
            else:
                fpc_nat = fpc_nat.at[:, pft].set(0.0)

        for pft in range(1, nvm):
            if bool(is_tree[pft]) and bool(natural[pft]):
                sumfpc_wood = sumfpc_wood + jnp.where(pft_present[:, pft], fpc_nat[:, pft], 0.0)
            if (not bool(is_tree[pft])) and bool(natural[pft]) and not bool(pasture[pft]):
                spacefight_grass = spacefight_grass + jnp.where(pft_present[:, pft], everywhere[:, pft], 0.0)

        climate_ok = (precip_annual >= precip_crit) & (gdd0 >= gdd_crit_estab)
        estab_tree_climate = jnp.where(climate_ok, estab_max_tree, 0.0)
        estab_grass_climate = jnp.where(climate_ok, estab_max_grass, 0.0)
        tree_factor = (1.0 - jnp.exp(-establish_scal_fact * (1.0 - sumfpc_wood))) * (1.0 - sumfpc_wood)
        estab_rate_max_tree_arr = estab_tree_climate * tree_factor
        estab_rate_max_grass_arr = jnp.maximum(jnp.minimum(estab_grass_climate, max_tree_coverage - sumfpc), 0.0)

        grass_productivity = jnp.ones((npts,), dtype=ind.dtype) * min_stomate
        for pft in range(1, nvm):
            if bool(natural[pft]) and (not bool(is_tree[pft])) and not bool(pasture[pft]):
                grass_productivity = grass_productivity + npp_longterm[:, pft] * lm_lastyearmax[:, pft] * sla_calc[:, pft]

        if treat_expansion:
            if migrate is None:
                raise ValueError("treat_expansion=True requires explicit migrate speeds")
            migrate = jnp.asarray(migrate)
            for pft in range(1, nvm):
                if bool(natural[pft]) and not bool(pasture[pft]):
                    for point in range(npts):
                        expandable = (
                            bool(pft_present[point, pft])
                            and bool(everywhere[point, pft] < 1.0)
                            and bool(regenerate[point, pft] > regenerate_crit)
                        )
                        if expandable:
                            nfrontx = 0
                            east = int(neighbours[point, 2])
                            west = int(neighbours[point, 6])
                            if east > 0 and bool(everywhere[east - 1, pft] > 1.0 - min_stomate):
                                nfrontx += 1
                            if west > 0 and bool(everywhere[west - 1, pft] > 1.0 - min_stomate):
                                nfrontx += 1
                            nfronty = 0
                            north = int(neighbours[point, 0])
                            south = int(neighbours[point, 4])
                            if north > 0 and bool(everywhere[north - 1, pft] > 1.0 - min_stomate):
                                nfronty += 1
                            if south > 0 and bool(everywhere[south - 1, pft] > 1.0 - min_stomate):
                                nfronty += 1
                            expanded = everywhere[point, pft] + migrate[pft] * dt_days / ONE_YEAR_DAYS * (
                                nfrontx / resolution[point, 0] + nfronty / resolution[point, 1]
                            )
                            if not bool(need_adjacent[point, pft]):
                                expanded = expanded + migrate[pft] * dt_days / ONE_YEAR_DAYS * 2.0 * jnp.sqrt(
                                    jnp.pi * expanded / (resolution[point, 0] * resolution[point, 1])
                                )
                            everywhere = everywhere.at[point, pft].set(jnp.minimum(expanded, 1.0))

        for pft in range(1, nvm):
            if bool(natural[pft]) and not bool(pasture[pft]):
                regen_present = pft_present[:, pft] & (regenerate[:, pft] > regenerate_crit)
                if bool(is_tree[pft]):
                    d_val = estab_rate_max_tree_arr * everywhere[:, pft] * avail_tree * dt_days / ONE_YEAR_DAYS
                    d_ind = d_ind.at[:, pft].set(jnp.where(regen_present, d_val, d_ind[:, pft]))
                else:
                    grass_ok = regen_present & (grass_productivity > min_stomate) & (spacefight_grass > min_stomate)
                    grass_weight = jnp.maximum(
                        min_stomate,
                        npp_longterm[:, pft] * lm_lastyearmax[:, pft] * sla_calc[:, pft] / grass_productivity,
                    )
                    d_val = (
                        estab_rate_max_grass_arr
                        * everywhere[:, pft]
                        / jnp.where(spacefight_grass > min_stomate, spacefight_grass, 1.0)
                        * grass_weight
                        * fracnat
                        * dt_days
                        / ONE_YEAR_DAYS
                    )
                    d_ind = d_ind.at[:, pft].set(jnp.where(grass_ok, d_val, d_ind[:, pft]))
    else:
        for pft in range(1, nvm):
            cover = ind[:, pft] * cn_ind[:, pft]
            lai_ind = jnp.where(
                cover > min_stomate,
                sla_calc[:, pft] * lm_lastyearmax[:, pft] / cover,
                0.0,
            )
            if bool(natural[pft]) and bool(is_tree[pft]):
                fpc_pft = jnp.minimum(
                    1.0,
                    cn_ind[:, pft] * ind[:, pft] * jnp.maximum(1.0 - jnp.exp(-ext_coeff[pft] * lai_ind), min_cover),
                )
                fpc_nat = fpc_nat.at[:, pft].set(fpc_pft)
                active = (veget_max[:, pft] > min_stomate) & (ind[:, pft] <= 2.0)
                factor = (1.0 - jnp.exp(-establish_scal_fact * (1.0 - fpc_pft))) * (1.0 - fpc_pft)
                d_val = jnp.maximum(0.0, estab_max_tree * factor * dt_days / ONE_YEAR_DAYS)
                d_val = jnp.where(active, d_val, 0.0)
                d_val = jnp.where((veget_max[:, pft] > min_stomate) & (ind[:, pft] == 0.0), ind_0_estab, d_val)
                d_ind = d_ind.at[:, pft].set(d_val)
                estab_rate_max_tree_arr = jnp.maximum(estab_rate_max_tree_arr, jnp.where(active, estab_max_tree * factor, 0.0))
            elif bool(natural[pft]) and (not bool(is_tree[pft])) and not bool(pasture[pft]):
                fpc_pft = cn_ind[:, pft] * ind[:, pft] * jnp.maximum(1.0 - jnp.exp(-ext_coeff[pft] * lai_ind), min_cover)
                fpc_nat = fpc_nat.at[:, pft].set(jnp.where(veget_max[:, pft] > min_stomate, fpc_pft, 0.0))
                d_val = jnp.maximum(0.0, (1.0 - fpc_pft) * dt_days / ONE_YEAR_DAYS)
                d_val = jnp.where(veget_max[:, pft] > min_stomate, d_val, 0.0)
                d_val = jnp.where((veget_max[:, pft] > min_stomate) & (ind[:, pft] == 0.0), ind_0_estab, d_val)
                d_ind = d_ind.at[:, pft].set(d_val)

    if ok_herbivores:
        for pft in range(1, nvm):
            if bool(natural[pft]) and not bool(pasture[pft]):
                d_ind = d_ind.at[:, pft].set(d_ind[:, pft] * jnp.exp(-(ONE_YEAR_DAYS / 2.0) / herbivores[:, pft]))

    d_ind = d_ind.at[:, 0].set(0.0)
    ind_estab = d_ind / dt_days
    ind_estab = ind_estab.at[:, 0].set(0.0)
    return EstablishmentRatesResult(
        d_ind=d_ind,
        ind_estab=ind_estab,
        everywhere=everywhere,
        fpc_nat=fpc_nat,
        estab_rate_max_tree=estab_rate_max_tree_arr,
        estab_rate_max_grass=estab_rate_max_grass_arr,
        sumfpc=sumfpc,
        sumfpc_wood=sumfpc_wood,
        spacefight_grass=spacefight_grass,
        fracnat=fracnat,
    )


def establishment_biomass_step(
    *,
    d_ind,
    ind,
    biomass,
    leaf_age,
    leaf_frac,
    age,
    co2_to_bm,
    veget_max,
    woodmass_ind,
    mortality,
    bm_to_litter,
    bm_sapl,
    npp_longterm,
    natural,
    pasture,
    is_tree,
    maxdia,
    dt_days=1.0,
    ok_dgvm=False,
    min_stomate=0.0,
    pipe_density=2.0e5,
    pipe_tune1=100.0,
    pipe_tune2=40.0,
    pipe_tune3=0.5,
    pipe_tune_exp_coeff=1.6,
) -> EstablishmentBiomassResult:
    """Apply sapling biomass accounting from ``lpj_establish::establish``.

    Fortran provenance: ``src_stomate/lpj_establish.f90``, subroutine
    ``establish``, lines 563-824. This helper starts after ``d_ind`` has been
    computed by the establishment-rate sections and preserves the Fortran carbon
    bookkeeping, leaf age/fraction updates, age dilution, individual increment,
    and ``woodmass_ind`` update. ``vn`` is local in Fortran and is not written
    back here.
    """

    d_ind = jnp.asarray(d_ind)
    ind = jnp.asarray(ind)
    biomass = jnp.asarray(biomass)
    leaf_age = jnp.asarray(leaf_age)
    leaf_frac = jnp.asarray(leaf_frac)
    age = jnp.asarray(age)
    co2_to_bm = jnp.asarray(co2_to_bm)
    veget_max = jnp.asarray(veget_max)
    woodmass_ind = jnp.asarray(woodmass_ind)
    mortality = jnp.asarray(mortality)
    bm_to_litter = jnp.asarray(bm_to_litter)
    bm_sapl = jnp.asarray(bm_sapl)
    npp_longterm = jnp.asarray(npp_longterm)
    natural = jnp.asarray(natural, dtype=bool)
    pasture = jnp.asarray(pasture, dtype=bool)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    maxdia = jnp.asarray(maxdia)

    if biomass.shape != bm_to_litter.shape:
        raise ValueError("biomass and bm_to_litter must have the same shape")
    npts, nvm, nparts, nelements = biomass.shape
    if nparts != NPARTS or nelements <= ICARBON:
        raise ValueError("biomass must have ORCHIDEE biomass part layout and carbon element")
    if leaf_age.shape != (npts, nvm, NLEAFAGES) or leaf_frac.shape != (npts, nvm, NLEAFAGES):
        raise ValueError("leaf_age and leaf_frac must have shape (npts, nvm, nleafages)")
    if bm_sapl.shape[0] != nvm or bm_sapl.shape[1] != NPARTS or bm_sapl.shape[2] <= ICARBON:
        raise ValueError("bm_sapl must have shape (nvm, nparts, nelements)")

    wood_parts = jnp.asarray(WOOD_BIOMASS_PARTS)
    veget_max_tree = jnp.zeros((npts,), dtype=biomass.dtype)
    nbtree = 0
    for pft in range(nvm):
        if bool(is_tree[pft]):
            veget_max_tree = veget_max_tree + veget_max[:, pft]
            nbtree += 1
    if nbtree == 0:
        nbtree = 1

    for pft in range(1, nvm):
        if bool(natural[pft]) and not bool(pasture[pft]):
            leaf_mass_young = leaf_frac[:, pft, 0] * biomass[:, pft, ILEAF, ICARBON]
            total_bm_c = jnp.sum(biomass[:, pft, :, ICARBON], axis=1)

            wood_existing = jnp.sum(biomass[:, pft, wood_parts, ICARBON], axis=1)
            sapling_wood = jnp.sum(bm_sapl[pft, wood_parts, ICARBON])
            active_after = d_ind[:, pft] + ind[:, pft] > min_stomate
            if bool(is_tree[pft]):
                if ok_dgvm:
                    first_or_no_cover = (total_bm_c <= min_stomate) | (veget_max[:, pft] <= min_stomate)
                    wood_new_first = (wood_existing * veget_max[:, pft] + sapling_wood * d_ind[:, pft]) / jnp.where(
                        active_after, ind[:, pft] + d_ind[:, pft], 1.0
                    )
                    wood_new_existing = wood_existing * veget_max[:, pft] / jnp.where(
                        active_after, ind[:, pft] + d_ind[:, pft], 1.0
                    )
                else:
                    first_or_no_cover = total_bm_c <= min_stomate
                    wood_new_first = (wood_existing + sapling_wood * d_ind[:, pft]) / jnp.where(
                        active_after, ind[:, pft] + d_ind[:, pft], 1.0
                    )
                    wood_new_existing = wood_existing / jnp.where(active_after, ind[:, pft] + d_ind[:, pft], 1.0)
                wood_new = jnp.where(first_or_no_cover, wood_new_first, wood_new_existing)
                woodmass_ind = woodmass_ind.at[:, pft].set(jnp.where(active_after, wood_new, woodmass_ind[:, pft]))

                dia = (woodmass_ind[:, pft] / (pipe_density * jnp.pi / 4.0 * pipe_tune2)) ** (1.0 / (2.0 + pipe_tune3))
                _vn = (ind[:, pft] + d_ind[:, pft]) * pipe_tune1 * jnp.minimum(dia, maxdia[pft]) ** pipe_tune_exp_coeff
            else:
                _vn = ind[:, pft] + d_ind[:, pft] if ok_dgvm else jnp.ones((npts,), dtype=biomass.dtype)

            total_bm_sapl = jnp.zeros((npts,), dtype=biomass.dtype)
            total_bm_sapl_non = jnp.zeros((npts,), dtype=biomass.dtype)
            existing_with_cover = (d_ind[:, pft] > min_stomate) & (total_bm_c > min_stomate) & (veget_max[:, pft] > min_stomate)
            for part in range(NPARTS):
                total_bm_sapl = total_bm_sapl + jnp.where(
                    existing_with_cover,
                    bm_sapl[pft, part, ICARBON] * d_ind[:, pft] / veget_max[:, pft],
                    0.0,
                )
                total_bm_sapl_non = total_bm_sapl_non + jnp.where(
                    existing_with_cover,
                    bm_sapl[pft, part, ICARBON] * (ind[:, pft] + d_ind[:, pft]) * mortality[:, pft] / veget_max[:, pft],
                    0.0,
                )

            biomass_old = biomass[:, pft, :, :]
            for part in range(NPARTS):
                sapling_part = bm_sapl[pft, part, ICARBON]
                first_estab = (d_ind[:, pft] > min_stomate) & (total_bm_c <= min_stomate) & (veget_max[:, pft] > min_stomate)
                bm_new_first = d_ind[:, pft] * sapling_part / jnp.where(veget_max[:, pft] > min_stomate, veget_max[:, pft], 1.0)
                biomass = biomass.at[:, pft, part, ICARBON].add(jnp.where(first_estab, bm_new_first, 0.0))

                sparse_tree = (veget_max_tree > 0.1) & (veget_max[:, pft] < veget_max_tree / nbtree)
                bm_non_first_sparse_initial = jnp.minimum(
                    biomass[:, pft, part, ICARBON] + bm_to_litter[:, pft, part, ICARBON],
                    (ind[:, pft] + d_ind[:, pft])
                    * mortality[:, pft]
                    * sapling_part
                    / jnp.where(veget_max[:, pft] > min_stomate, veget_max[:, pft], 1.0),
                )
                bm_eff_first = jnp.minimum(npp_longterm[:, pft] / ONE_YEAR_DAYS, bm_new_first - bm_non_first_sparse_initial)
                bm_non_first_sparse = jnp.minimum(
                    biomass[:, pft, part, ICARBON] + bm_to_litter[:, pft, part, ICARBON],
                    bm_new_first - bm_eff_first,
                )
                bm_non_first_dense = jnp.minimum(
                    bm_to_litter[:, pft, part, ICARBON],
                    (ind[:, pft] + d_ind[:, pft])
                    * mortality[:, pft]
                    * sapling_part
                    / jnp.where(veget_max[:, pft] > min_stomate, veget_max[:, pft], 1.0),
                )
                bm_non_first = jnp.where(sparse_tree, bm_non_first_sparse, bm_non_first_dense)
                co2_first = jnp.where(sparse_tree, bm_new_first - bm_non_first, (bm_new_first - bm_non_first) / dt_days)
                co2_to_bm = co2_to_bm.at[:, pft].add(jnp.where(first_estab, co2_first, 0.0))
                deficit_first = jnp.maximum(bm_non_first - bm_to_litter[:, pft, part, ICARBON], 0.0)
                biomass = biomass.at[:, pft, part, ICARBON].add(jnp.where(first_estab & sparse_tree, -deficit_first, 0.0))
                litter_reduction_first = jnp.where(sparse_tree, jnp.minimum(bm_to_litter[:, pft, part, ICARBON], bm_non_first), bm_non_first)
                bm_to_litter = bm_to_litter.at[:, pft, part, ICARBON].add(jnp.where(first_estab, -litter_reduction_first, 0.0))

                existing_estab = (d_ind[:, pft] > min_stomate) & (total_bm_c > min_stomate)
                bm_new_existing = total_bm_sapl * biomass_old[:, part, ICARBON] / jnp.where(total_bm_c > min_stomate, total_bm_c, 1.0)
                biomass = biomass.at[:, pft, part, ICARBON].add(jnp.where(existing_estab, bm_new_existing, 0.0))

                bm_non_existing_sparse_initial = jnp.minimum(
                    biomass[:, pft, part, ICARBON] + bm_to_litter[:, pft, part, ICARBON],
                    total_bm_sapl_non * biomass_old[:, part, ICARBON] / jnp.where(total_bm_c > min_stomate, total_bm_c, 1.0),
                )
                bm_eff_existing = jnp.minimum(npp_longterm[:, pft] / ONE_YEAR_DAYS, bm_new_existing - bm_non_existing_sparse_initial)
                bm_non_existing_sparse = jnp.maximum(
                    0.0,
                    jnp.minimum(
                        biomass[:, pft, part, ICARBON] + bm_to_litter[:, pft, part, ICARBON] - min_stomate,
                        bm_new_existing - bm_eff_existing,
                    ),
                )
                bm_non_existing_dense = jnp.minimum(
                    bm_to_litter[:, pft, part, ICARBON],
                    total_bm_sapl_non * biomass_old[:, part, ICARBON] / jnp.where(total_bm_c > min_stomate, total_bm_c, 1.0),
                )
                bm_non_existing = jnp.where(sparse_tree, bm_non_existing_sparse, bm_non_existing_dense)
                co2_existing = jnp.where(sparse_tree, bm_new_existing - bm_non_existing, (bm_new_existing - bm_non_existing) / dt_days)
                co2_to_bm = co2_to_bm.at[:, pft].add(jnp.where(existing_estab, co2_existing, 0.0))
                deficit_existing = jnp.maximum(bm_non_existing - bm_to_litter[:, pft, part, ICARBON], 0.0)
                biomass = biomass.at[:, pft, part, ICARBON].add(jnp.where(existing_estab & sparse_tree, -deficit_existing, 0.0))
                litter_reduction_existing = jnp.where(
                    sparse_tree,
                    jnp.minimum(bm_to_litter[:, pft, part, ICARBON], bm_non_existing),
                    bm_non_existing,
                )
                bm_to_litter = bm_to_litter.at[:, pft, part, ICARBON].add(jnp.where(existing_estab, -litter_reduction_existing, 0.0))

            leaf_add = d_ind[:, pft] * bm_sapl[pft, ILEAF, ICARBON]
            age_update = leaf_add > min_stomate
            leaf_age_new0 = leaf_age[:, pft, 0] * leaf_mass_young / jnp.where(leaf_mass_young + leaf_add != 0.0, leaf_mass_young + leaf_add, 1.0)
            leaf_age = leaf_age.at[:, pft, 0].set(jnp.where(age_update, leaf_age_new0, leaf_age[:, pft, 0]))
            leaf_mass_young = leaf_mass_young + leaf_add
            leaf_biomass = biomass[:, pft, ILEAF, ICARBON]
            has_leaf = leaf_biomass > min_stomate
            leaf_frac = leaf_frac.at[:, pft, 0].set(jnp.where(has_leaf, leaf_mass_young / leaf_biomass, leaf_frac[:, pft, 0]))
            for leaf_class in range(1, NLEAFAGES):
                updated_frac = leaf_frac[:, pft, leaf_class] * (leaf_biomass + leaf_add) / jnp.where(leaf_biomass != 0.0, leaf_biomass, 1.0)
                leaf_frac = leaf_frac.at[:, pft, leaf_class].set(jnp.where(has_leaf, updated_frac, leaf_frac[:, pft, leaf_class]))

            active_estab = d_ind[:, pft] > min_stomate
            age = age.at[:, pft].set(jnp.where(active_estab, age[:, pft] * ind[:, pft] / (ind[:, pft] + d_ind[:, pft]), age[:, pft]))
            ind = ind.at[:, pft].set(jnp.where(active_estab, ind[:, pft] + d_ind[:, pft], ind[:, pft]))

    return EstablishmentBiomassResult(
        ind=ind,
        biomass=biomass,
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        age=age,
        co2_to_bm=co2_to_bm,
        woodmass_ind=woodmass_ind,
        bm_to_litter=bm_to_litter,
    )


def cover_step(
    *,
    cn_ind,
    ind,
    biomass,
    veget_max,
    veget_max_old,
    lai,
    litter,
    litter_avail,
    litter_not_avail,
    carbon,
    fuel_1hr,
    fuel_10hr,
    fuel_100hr,
    fuel_1000hr,
    turnover_daily,
    bm_to_litter,
    co2_to_bm,
    co2_fire,
    resp_hetero,
    resp_maint,
    resp_growth,
    gpp_daily,
    deepC_a,
    deepC_s,
    deepC_p,
    natural,
    pasture,
    is_peat,
    is_grassland_manag=None,
    is_grassland_grazed=None,
    ok_dgvm=False,
    ok_pc=False,
    min_stomate=0.0,
) -> CoverResult:
    """Apply ordinary ``lpj_cover::cover`` vegetation-cover redistribution.

    Fortran provenance: ``src_stomate/lpj_cover.f90``, subroutine ``cover``,
    lines 73-387. This is the ordinary non-peat branch called by
    ``stomate_lpj.f90`` lines 1372-1379; ``lpj_cover_peat`` is a separate
    branch and is not folded into this helper.
    """

    cn_ind = jnp.asarray(cn_ind)
    ind = jnp.asarray(ind)
    biomass = jnp.asarray(biomass)
    veget_max = jnp.asarray(veget_max)
    veget_max_old = jnp.asarray(veget_max_old)
    lai = jnp.asarray(lai)
    litter = jnp.asarray(litter)
    litter_avail = jnp.asarray(litter_avail)
    litter_not_avail = jnp.asarray(litter_not_avail)
    carbon = jnp.asarray(carbon)
    fuel_1hr = jnp.asarray(fuel_1hr)
    fuel_10hr = jnp.asarray(fuel_10hr)
    fuel_100hr = jnp.asarray(fuel_100hr)
    fuel_1000hr = jnp.asarray(fuel_1000hr)
    turnover_daily = jnp.asarray(turnover_daily)
    bm_to_litter = jnp.asarray(bm_to_litter)
    co2_to_bm = jnp.asarray(co2_to_bm)
    co2_fire = jnp.asarray(co2_fire)
    resp_hetero = jnp.asarray(resp_hetero)
    resp_maint = jnp.asarray(resp_maint)
    resp_growth = jnp.asarray(resp_growth)
    gpp_daily = jnp.asarray(gpp_daily)
    deepC_a = jnp.asarray(deepC_a)
    deepC_s = jnp.asarray(deepC_s)
    deepC_p = jnp.asarray(deepC_p)
    natural = jnp.asarray(natural, dtype=bool)
    pasture = jnp.asarray(pasture, dtype=bool)
    is_peat = jnp.asarray(is_peat, dtype=bool)
    if is_grassland_manag is None:
        is_grassland_manag = jnp.zeros((veget_max.shape[1],), dtype=bool)
    else:
        is_grassland_manag = jnp.asarray(is_grassland_manag, dtype=bool)
    if is_grassland_grazed is None:
        is_grassland_grazed = jnp.zeros((veget_max.shape[1],), dtype=bool)
    else:
        is_grassland_grazed = jnp.asarray(is_grassland_grazed, dtype=bool)

    npts, nvm = veget_max.shape
    if biomass.shape[:2] != (npts, nvm):
        raise ValueError("biomass must share (npts, nvm) with veget_max")
    co2flux_old = resp_maint + resp_growth + resp_hetero + co2_fire - co2_to_bm - gpp_daily
    co2flux_new = co2flux_old
    tcarbon = (
        jnp.sum(biomass[:, :, :, ICARBON], axis=2)
        + jnp.sum(carbon, axis=1)
        + jnp.sum(litter[:, :, :, :, ICARBON], axis=(1, 3))
        + jnp.sum(turnover_daily[:, :, :, ICARBON], axis=2)
        + jnp.sum(bm_to_litter[:, :, :, ICARBON], axis=2)
    )

    if not ok_dgvm:
        return CoverResult(
            veget_max=veget_max,
            lai=lai,
            litter=litter,
            litter_avail=litter_avail,
            litter_not_avail=litter_not_avail,
            carbon=carbon,
            biomass=biomass,
            fuel_1hr=fuel_1hr,
            fuel_10hr=fuel_10hr,
            fuel_100hr=fuel_100hr,
            fuel_1000hr=fuel_1000hr,
            turnover_daily=turnover_daily,
            bm_to_litter=bm_to_litter,
            co2_to_bm=co2_to_bm,
            co2_fire=co2_fire,
            resp_hetero=resp_hetero,
            resp_maint=resp_maint,
            resp_growth=resp_growth,
            gpp_daily=gpp_daily,
            deepC_a=deepC_a,
            deepC_s=deepC_s,
            deepC_p=deepC_p,
            co2flux_old=co2flux_old,
            co2flux_new=co2flux_new,
            tcarbon=tcarbon,
        )

    frac_nat = jnp.ones((npts,), dtype=veget_max.dtype)
    sum_veget_natveg = jnp.zeros((npts,), dtype=veget_max.dtype)
    veget_max = veget_max.at[:, 0].set(1.0)
    for pft in range(1, nvm):
        if bool(natural[pft]) and not bool(pasture[pft]) and not bool(is_peat[pft]):
            new_cover = ind[:, pft] * cn_ind[:, pft]
            veget_max = veget_max.at[:, pft].set(new_cover)
            sum_veget_natveg = sum_veget_natveg + new_cover
        else:
            frac_nat = frac_nat - veget_max[:, pft]

    scale_mask = (sum_veget_natveg > frac_nat) & (frac_nat > min_stomate)
    for pft in range(1, nvm):
        if bool(natural[pft]) and not bool(pasture[pft]) and not bool(is_peat[pft]):
            scaled = veget_max[:, pft] * frac_nat / jnp.where(sum_veget_natveg != 0.0, sum_veget_natveg, 1.0)
            veget_max = veget_max.at[:, pft].set(jnp.where(scale_mask, scaled, veget_max[:, pft]))

    bare = jnp.ones((npts,), dtype=veget_max.dtype)
    for pft in range(1, nvm):
        bare = bare - veget_max[:, pft]
    veget_max = veget_max.at[:, 0].set(jnp.maximum(bare, 0.0))

    carbon_old = carbon
    co2flux_old = resp_maint + resp_growth + resp_hetero + co2_fire - co2_to_bm - gpp_daily
    co2flux_new = co2flux_old
    tcarbon = (
        jnp.sum(biomass[:, :, :, ICARBON], axis=2)
        + jnp.sum(carbon, axis=1)
        + jnp.sum(litter[:, :, :, :, ICARBON], axis=(1, 3))
        + jnp.sum(turnover_daily[:, :, :, ICARBON], axis=2)
        + jnp.sum(bm_to_litter[:, :, :, ICARBON], axis=2)
    )

    for point in range(npts):
        delta_veg = veget_max[point, :] - veget_max_old[point, :]
        negative = delta_veg < -min_stomate
        delta_veg_sum = jnp.sum(jnp.where(negative, delta_veg, 0.0))

        dilu_lit = jnp.zeros_like(litter[point, :, 0, :, :])
        dilu_f1hr = jnp.zeros_like(fuel_1hr[point, 0, :, :])
        dilu_f10hr = jnp.zeros_like(fuel_10hr[point, 0, :, :])
        dilu_f100hr = jnp.zeros_like(fuel_100hr[point, 0, :, :])
        dilu_f1000hr = jnp.zeros_like(fuel_1000hr[point, 0, :, :])
        dilu_soil_carbon = jnp.zeros_like(carbon[point, :, 0])
        dilu_tcarbon = 0.0
        dilu_turnover_daily = jnp.zeros_like(turnover_daily[point, 0, :, :])
        dilu_bm_to_litter = jnp.zeros_like(bm_to_litter[point, 0, :, :])
        dilu_co2flux_new = 0.0
        dilu_gpp_daily = 0.0
        dilu_resp_growth = 0.0
        dilu_resp_maint = 0.0
        dilu_resp_hetero = 0.0
        dilu_co2_to_bm = 0.0
        dilu_co2_fire = 0.0

        if bool(delta_veg_sum < 0.0):
            for pft in range(nvm):
                if bool(delta_veg[pft] < -min_stomate):
                    weight = delta_veg[pft] / delta_veg_sum
                    dilu_lit = dilu_lit + weight * litter[point, :, pft, :, :]
                    dilu_f1hr = dilu_f1hr + weight * fuel_1hr[point, pft, :, :]
                    dilu_f10hr = dilu_f10hr + weight * fuel_10hr[point, pft, :, :]
                    dilu_f100hr = dilu_f100hr + weight * fuel_100hr[point, pft, :, :]
                    dilu_f1000hr = dilu_f1000hr + weight * fuel_1000hr[point, pft, :, :]
                    dilu_soil_carbon = dilu_soil_carbon + weight * carbon[point, :, pft]
                    dilu_tcarbon = dilu_tcarbon + weight * tcarbon[point, pft]
                    dilu_turnover_daily = dilu_turnover_daily + weight * turnover_daily[point, pft, :, :]
                    dilu_bm_to_litter = dilu_bm_to_litter + weight * bm_to_litter[point, pft, :, :]
                    dilu_co2flux_new = dilu_co2flux_new + weight * co2flux_old[point, pft]
                    dilu_gpp_daily = dilu_gpp_daily + weight * gpp_daily[point, pft]
                    dilu_resp_growth = dilu_resp_growth + weight * resp_growth[point, pft]
                    dilu_resp_maint = dilu_resp_maint + weight * resp_maint[point, pft]
                    dilu_resp_hetero = dilu_resp_hetero + weight * resp_hetero[point, pft]
                    dilu_co2_to_bm = dilu_co2_to_bm + weight * co2_to_bm[point, pft]
                    dilu_co2_fire = dilu_co2_fire + weight * co2_fire[point, pft]

        for pft in range(nvm):
            if bool(delta_veg[pft] > min_stomate):
                denom = veget_max[point, pft]
                litter = litter.at[point, :, pft, :, :].set(
                    (litter[point, :, pft, :, :] * veget_max_old[point, pft] + dilu_lit * delta_veg[pft]) / denom
                )
                fuel_1hr = fuel_1hr.at[point, pft, :, :].set(
                    (fuel_1hr[point, pft, :, :] * veget_max_old[point, pft] + dilu_f1hr * delta_veg[pft]) / denom
                )
                fuel_10hr = fuel_10hr.at[point, pft, :, :].set(
                    (fuel_10hr[point, pft, :, :] * veget_max_old[point, pft] + dilu_f10hr * delta_veg[pft]) / denom
                )
                fuel_100hr = fuel_100hr.at[point, pft, :, :].set(
                    (fuel_100hr[point, pft, :, :] * veget_max_old[point, pft] + dilu_f100hr * delta_veg[pft]) / denom
                )
                fuel_1000hr = fuel_1000hr.at[point, pft, :, :].set(
                    (fuel_1000hr[point, pft, :, :] * veget_max_old[point, pft] + dilu_f1000hr * delta_veg[pft]) / denom
                )
                if bool(is_grassland_manag[pft]) and bool(is_grassland_grazed[pft]):
                    litter_avail = litter_avail.at[point, :, pft].set(litter_avail[point, :, pft] * veget_max_old[point, pft] / denom)
                    litter_not_avail = litter_not_avail.at[point, :, pft].set(litter[point, :, pft, IABOVE, ICARBON] - litter_avail[point, :, pft])

                carbon = carbon.at[point, :, pft].set(
                    (carbon[point, :, pft] * veget_max_old[point, pft] + dilu_soil_carbon * delta_veg[pft]) / denom
                )
                if ok_pc:
                    active_old = carbon_old[point, IACTIVE, pft]
                    slow_old = carbon_old[point, ISLOW, pft]
                    passive_old = carbon_old[point, IPASSIVE, pft]
                    deepC_a = deepC_a.at[point, :, pft].set(
                        jnp.where(active_old > min_stomate, deepC_a[point, :, pft] * carbon[point, IACTIVE, pft] / active_old, deepC_a[point, :, pft])
                    )
                    deepC_s = deepC_s.at[point, :, pft].set(
                        jnp.where(slow_old > min_stomate, deepC_s[point, :, pft] * carbon[point, ISLOW, pft] / slow_old, deepC_s[point, :, pft])
                    )
                    deepC_p = deepC_p.at[point, :, pft].set(
                        jnp.where(passive_old > min_stomate, deepC_p[point, :, pft] * carbon[point, IPASSIVE, pft] / passive_old, deepC_p[point, :, pft])
                    )

                tcarbon = tcarbon.at[point, pft].set((tcarbon[point, pft] * veget_max_old[point, pft] + dilu_tcarbon * delta_veg[pft]) / denom)
                turnover_daily = turnover_daily.at[point, pft, :, :].set(
                    (turnover_daily[point, pft, :, :] * veget_max_old[point, pft] + dilu_turnover_daily * delta_veg[pft]) / denom
                )
                bm_to_litter = bm_to_litter.at[point, pft, :, :].set(
                    (bm_to_litter[point, pft, :, :] * veget_max_old[point, pft] + dilu_bm_to_litter * delta_veg[pft]) / denom
                )
                co2flux_new = co2flux_new.at[point, pft].set((co2flux_old[point, pft] * veget_max_old[point, pft] + dilu_co2flux_new * delta_veg[pft]) / denom)
                gpp_daily = gpp_daily.at[point, pft].set((gpp_daily[point, pft] * veget_max_old[point, pft] + dilu_gpp_daily * delta_veg[pft]) / denom)
                resp_growth = resp_growth.at[point, pft].set((resp_growth[point, pft] * veget_max_old[point, pft] + dilu_resp_growth * delta_veg[pft]) / denom)
                resp_maint = resp_maint.at[point, pft].set((resp_maint[point, pft] * veget_max_old[point, pft] + dilu_resp_maint * delta_veg[pft]) / denom)
                resp_hetero = resp_hetero.at[point, pft].set((resp_hetero[point, pft] * veget_max_old[point, pft] + dilu_resp_hetero * delta_veg[pft]) / denom)
                co2_to_bm = co2_to_bm.at[point, pft].set((co2_to_bm[point, pft] * veget_max_old[point, pft] + dilu_co2_to_bm * delta_veg[pft]) / denom)
                co2_fire = co2_fire.at[point, pft].set((co2_fire[point, pft] * veget_max_old[point, pft] + dilu_co2_fire * delta_veg[pft]) / denom)

            if bool(veget_max[point, pft] > min_stomate):
                biomass = biomass.at[point, pft, :, :].set(biomass[point, pft, :, :] * veget_max_old[point, pft] / veget_max[point, pft])

    return CoverResult(
        veget_max=veget_max,
        lai=lai,
        litter=litter,
        litter_avail=litter_avail,
        litter_not_avail=litter_not_avail,
        carbon=carbon,
        biomass=biomass,
        fuel_1hr=fuel_1hr,
        fuel_10hr=fuel_10hr,
        fuel_100hr=fuel_100hr,
        fuel_1000hr=fuel_1000hr,
        turnover_daily=turnover_daily,
        bm_to_litter=bm_to_litter,
        co2_to_bm=co2_to_bm,
        co2_fire=co2_fire,
        resp_hetero=resp_hetero,
        resp_maint=resp_maint,
        resp_growth=resp_growth,
        gpp_daily=gpp_daily,
        deepC_a=deepC_a,
        deepC_s=deepC_s,
        deepC_p=deepC_p,
        co2flux_old=co2flux_old,
        co2flux_new=co2flux_new,
        tcarbon=tcarbon,
    )


def lpj_cover_peat_step(
    *,
    cn_ind,
    ind,
    biomass,
    veget_max_new,
    veget_max,
    veget_max_old,
    litter,
    litter_avail,
    litter_not_avail,
    carbon,
    fuel_1hr,
    fuel_10hr,
    fuel_100hr,
    fuel_1000hr,
    turnover_daily,
    bm_to_litter,
    co2_to_bm,
    co2_fire,
    resp_hetero,
    resp_maint,
    resp_growth,
    gpp_daily,
    deepC_a,
    deepC_s,
    deepC_p,
    dt_days,
    age,
    pft_present,
    senescence,
    when_growthinit,
    everywhere,
    leaf_frac,
    lm_lastyearmax,
    npp_longterm,
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
    is_grassland_manag=None,
    is_grassland_grazed=None,
    ok_dgvm=False,
    ok_pc=False,
    min_stomate=1.0e-8,
    min_vegfrac=1.0e-6,
    npp_longterm_init=10.0,
    large_value=1.0e33,
) -> LpjCoverPeatResult:
    """Apply dynamic-peat cover redistribution from ``lpj_cover_peat``.

    Fortran provenance: ``src_stomate/stomate_lpj.f90``, subroutine
    ``lpj_cover_peat``, lines 2583-3337. This implements the
    source dynamic-peat branch: peat fraction target application, initial peat
    establishment, non-peat fraction adjustment, biomass/flux dilution,
    peat ``carbon_save``/``delta_fsave`` bookkeeping, and the ``OK_PC``
    vertically resolved ``deepC_*`` save/scale clauses.
    """

    cn_ind = jnp.asarray(cn_ind)
    ind = jnp.asarray(ind)
    biomass = jnp.asarray(biomass)
    veget_max_new = jnp.asarray(veget_max_new)
    veget_max = jnp.asarray(veget_max)
    veget_max_old = jnp.asarray(veget_max_old)
    litter = jnp.asarray(litter)
    litter_avail = jnp.asarray(litter_avail)
    litter_not_avail = jnp.asarray(litter_not_avail)
    carbon = jnp.asarray(carbon)
    fuel_1hr = jnp.asarray(fuel_1hr)
    fuel_10hr = jnp.asarray(fuel_10hr)
    fuel_100hr = jnp.asarray(fuel_100hr)
    fuel_1000hr = jnp.asarray(fuel_1000hr)
    turnover_daily = jnp.asarray(turnover_daily)
    bm_to_litter = jnp.asarray(bm_to_litter)
    co2_to_bm = jnp.asarray(co2_to_bm)
    co2_fire = jnp.asarray(co2_fire)
    resp_hetero = jnp.asarray(resp_hetero)
    resp_maint = jnp.asarray(resp_maint)
    resp_growth = jnp.asarray(resp_growth)
    gpp_daily = jnp.asarray(gpp_daily)
    deepC_a = jnp.asarray(deepC_a)
    deepC_s = jnp.asarray(deepC_s)
    deepC_p = jnp.asarray(deepC_p)
    age = jnp.asarray(age)
    pft_present = jnp.asarray(pft_present, dtype=bool)
    senescence = jnp.asarray(senescence, dtype=bool)
    when_growthinit = jnp.asarray(when_growthinit)
    everywhere = jnp.asarray(everywhere)
    leaf_frac = jnp.asarray(leaf_frac)
    lm_lastyearmax = jnp.asarray(lm_lastyearmax)
    npp_longterm = jnp.asarray(npp_longterm)
    carbon_save = jnp.asarray(carbon_save)
    deepC_a_save = jnp.asarray(deepC_a_save)
    deepC_s_save = jnp.asarray(deepC_s_save)
    deepC_p_save = jnp.asarray(deepC_p_save)
    delta_fsave = jnp.asarray(delta_fsave)
    liqwt_max_lastyear = jnp.asarray(liqwt_max_lastyear)
    natural = jnp.asarray(natural, dtype=bool)
    pasture = jnp.asarray(pasture, dtype=bool)
    is_peat = jnp.asarray(is_peat, dtype=bool)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    bm_sapl = jnp.asarray(bm_sapl)
    cn_sapl = jnp.asarray(cn_sapl)
    min_stomate = jnp.asarray(min_stomate, dtype=veget_max.dtype)
    min_vegfrac = jnp.asarray(min_vegfrac, dtype=veget_max.dtype)
    npp_longterm_init = jnp.asarray(npp_longterm_init, dtype=veget_max.dtype)
    large_value = jnp.asarray(large_value, dtype=veget_max.dtype)
    dt_days = jnp.asarray(dt_days, dtype=veget_max.dtype)

    if veget_max.ndim != 2:
        raise ValueError("veget_max must have shape (npts, nvm)")
    npts, nvm = veget_max.shape
    if veget_max_new.shape != (npts, nvm) or veget_max_old.shape != (npts, nvm):
        raise ValueError("veget_max_new and veget_max_old must share veget_max shape")
    scalar_shape = (npts, nvm)
    for name, array in (
        ("cn_ind", cn_ind),
        ("ind", ind),
        ("co2_to_bm", co2_to_bm),
        ("co2_fire", co2_fire),
        ("resp_hetero", resp_hetero),
        ("resp_maint", resp_maint),
        ("resp_growth", resp_growth),
        ("gpp_daily", gpp_daily),
        ("age", age),
        ("pft_present", pft_present),
        ("senescence", senescence),
        ("when_growthinit", when_growthinit),
        ("everywhere", everywhere),
        ("lm_lastyearmax", lm_lastyearmax),
        ("npp_longterm", npp_longterm),
    ):
        if array.shape != scalar_shape:
            raise ValueError(f"{name} must have shape (npts, nvm)")
    if biomass.ndim != 4 or biomass.shape[:2] != (npts, nvm) or biomass.shape[2] != NPARTS:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    nelements = biomass.shape[3]
    if turnover_daily.shape != biomass.shape or bm_to_litter.shape != biomass.shape:
        raise ValueError("turnover_daily and bm_to_litter must share biomass shape")
    if litter.shape != (npts, NLITT, nvm, NLEVS, nelements):
        raise ValueError("litter must have shape (npts, nlitt, nvm, nlevs, nelements)")
    if litter_avail.shape != (npts, NLITT, nvm) or litter_not_avail.shape != (npts, NLITT, nvm):
        raise ValueError("litter_avail and litter_not_avail must have shape (npts, nlitt, nvm)")
    if carbon.shape != (npts, NCARB, nvm) or carbon_save.shape != (npts, NCARB, nvm):
        raise ValueError("carbon and carbon_save must have shape (npts, ncarb, nvm)")
    if deepC_a.shape != deepC_s.shape or deepC_a.shape != deepC_p.shape or deepC_a.shape[:1] != (npts,) or deepC_a.shape[2] != nvm:
        raise ValueError("deepC_* must have shape (npts, ndeep, nvm)")
    ndeep = deepC_a.shape[1]
    if deepC_a_save.shape != (npts, ndeep) or deepC_s_save.shape != (npts, ndeep) or deepC_p_save.shape != (npts, ndeep):
        raise ValueError("deepC_*_save must have shape (npts, ndeep)")
    if delta_fsave.shape != (npts,) or liqwt_max_lastyear.shape != (npts,):
        raise ValueError("delta_fsave and liqwt_max_lastyear must have shape (npts,)")
    fuel_shape = (npts, nvm, NLITT, nelements)
    if fuel_1hr.shape != fuel_shape or fuel_10hr.shape != fuel_shape or fuel_100hr.shape != fuel_shape or fuel_1000hr.shape != fuel_shape:
        raise ValueError("fuel arrays must have shape (npts, nvm, nlitt, nelements)")
    if leaf_frac.shape != (npts, nvm, NLEAFAGES):
        raise ValueError("leaf_frac must have shape (npts, nvm, nleafages)")
    if bm_sapl.shape != (nvm, NPARTS, nelements):
        raise ValueError("bm_sapl must have shape (nvm, nparts, nelements)")
    for name, array in (
        ("natural", natural),
        ("pasture", pasture),
        ("is_peat", is_peat),
        ("is_tree", is_tree),
        ("cn_sapl", cn_sapl),
    ):
        if array.shape != (nvm,):
            raise ValueError(f"{name} must have shape (nvm,)")
    peat_indices = [pft for pft in range(nvm) if bool(is_peat[pft])]
    if len(peat_indices) != 1:
        raise NotImplementedError("lpj_cover_peat_step follows the paper-source single peat PFT path")
    peat_pft = peat_indices[0]
    if is_grassland_manag is None:
        is_grassland_manag = jnp.zeros((nvm,), dtype=bool)
    else:
        is_grassland_manag = jnp.asarray(is_grassland_manag, dtype=bool)
    if is_grassland_grazed is None:
        is_grassland_grazed = jnp.zeros((nvm,), dtype=bool)
    else:
        is_grassland_grazed = jnp.asarray(is_grassland_grazed, dtype=bool)
    if is_grassland_manag.shape != (nvm,) or is_grassland_grazed.shape != (nvm,):
        raise ValueError("is_grassland_manag and is_grassland_grazed must have shape (nvm,)")

    carbon_old = carbon
    carbon_save_tmp = jnp.zeros_like(carbon_save)

    def _scale_deep_with_carbon(pool, point, pft, reference):
        nonlocal deepC_a, deepC_s, deepC_p
        if not bool(ok_pc):
            return
        if pool == IACTIVE and bool(reference[IACTIVE] > min_stomate):
            deepC_a = deepC_a.at[point, :, pft].set(deepC_a[point, :, pft] * carbon[point, IACTIVE, pft] / reference[IACTIVE])
        elif pool == ISLOW and bool(reference[ISLOW] > min_stomate):
            deepC_s = deepC_s.at[point, :, pft].set(deepC_s[point, :, pft] * carbon[point, ISLOW, pft] / reference[ISLOW])
        elif pool == IPASSIVE and bool(reference[IPASSIVE] > min_stomate):
            deepC_p = deepC_p.at[point, :, pft].set(deepC_p[point, :, pft] * carbon[point, IPASSIVE, pft] / reference[IPASSIVE])

    def _scale_all_deep_with_carbon(point, pft, reference):
        for pool in (IACTIVE, ISLOW, IPASSIVE):
            _scale_deep_with_carbon(pool, point, pft, reference)

    if bool(ok_dgvm):
        frac_nat = jnp.ones((npts,), dtype=veget_max.dtype)
        sum_veget_natveg = jnp.zeros((npts,), dtype=veget_max.dtype)
        veget_max = veget_max.at[:, 0].set(1.0)
        for pft in range(1, nvm):
            if bool(natural[pft]) and not bool(pasture[pft]) and not bool(is_peat[pft]):
                new_cover = ind[:, pft] * cn_ind[:, pft]
                veget_max = veget_max.at[:, pft].set(new_cover)
                sum_veget_natveg = sum_veget_natveg + new_cover
            else:
                frac_nat = frac_nat - veget_max[:, pft]
        scale_mask = (sum_veget_natveg > frac_nat) & (frac_nat > min_stomate)
        for pft in range(1, nvm):
            if bool(natural[pft]) and not bool(pasture[pft]) and not bool(is_peat[pft]):
                scaled = veget_max[:, pft] * frac_nat / jnp.where(sum_veget_natveg != 0.0, sum_veget_natveg, 1.0)
                veget_max = veget_max.at[:, pft].set(jnp.where(scale_mask, scaled, veget_max[:, pft]))
        veget_max = veget_max.at[:, 0].set(jnp.maximum(1.0 - jnp.sum(veget_max[:, 1:], axis=1), 0.0))
        veget_tmp = veget_max

        for point in range(npts):
            delta_target = veget_max_new[point, :] - veget_tmp[point, :]
            for pft in peat_indices:
                target = veget_max_new[point, pft]
                current = veget_tmp[point, pft]
                if bool(delta_target[pft] < -min_stomate):
                    veget_max = veget_max.at[point, pft].set(jnp.where(target > min_stomate, target, 0.0))
                elif bool(delta_target[pft] > min_stomate):
                    if bool(current <= 0.0) or bool(liqwt_max_lastyear[point] > 0.6):
                        veget_max = veget_max.at[point, pft].set(target)
                if bool(veget_max[point, pft] - current > min_stomate) and bool(current <= 0.0):
                    new_cn = jnp.where(is_tree[pft], cn_sapl[pft], 1.0)
                    cn_ind = cn_ind.at[point, pft].set(new_cn)
                    ind = ind.at[point, pft].set(veget_max[point, pft] / new_cn)
                    pft_present = pft_present.at[point, pft].set(True)
                    everywhere = everywhere.at[point, pft].set(1.0)
                    senescence = senescence.at[point, pft].set(False)
                    age = age.at[point, pft].set(0.0)
                    when_growthinit = when_growthinit.at[point, pft].set(large_value)
                    leaf_frac = leaf_frac.at[point, pft, 0].set(1.0)
                    npp_longterm = npp_longterm.at[point, pft].set(npp_longterm_init)
                    lm_lastyearmax = lm_lastyearmax.at[point, pft].set(bm_sapl[pft, ILEAF, ICARBON] * ind[point, pft])

            sum_veg = jnp.sum(veget_tmp[point, :])
            sumvpeat = jnp.sum(jnp.where(is_peat, veget_max[point, :], 0.0))
            sumvpeat_old = jnp.sum(jnp.where(is_peat, veget_tmp[point, :], 0.0))
            if bool(sumvpeat > sumvpeat_old):
                denom = sum_veg - sumvpeat_old
                rapport = (sum_veg - sumvpeat) / jnp.where(denom != 0.0, denom, 1.0)
                for pft in range(nvm):
                    if bool(natural[pft]) and not bool(is_peat[pft]):
                        veget_max = veget_max.at[point, pft].set(veget_tmp[point, pft] * rapport)
            else:
                veget_max = veget_max.at[point, 0].set(veget_tmp[point, 0] + sumvpeat_old - sumvpeat)
    else:
        for point in range(npts):
            delta_target = veget_max_new[point, :] - veget_max_old[point, :]
            for pft in peat_indices:
                target = veget_max_new[point, pft]
                current = veget_max_old[point, pft]
                if bool(delta_target[pft] < -min_stomate):
                    veget_max = veget_max.at[point, pft].set(jnp.where(target > min_stomate, target, 0.0))
                elif bool(delta_target[pft] > min_stomate):
                    if bool(current <= 0.0) or bool(liqwt_max_lastyear[point] > 0.6):
                        veget_max = veget_max.at[point, pft].set(target)

        sum_veget_natveg = jnp.zeros((npts,), dtype=veget_max.dtype)
        for pft in range(nvm):
            if bool(natural[pft]) and not bool(pasture[pft]) and not bool(is_peat[pft]):
                sum_veget_natveg = sum_veget_natveg + veget_max_new[:, pft]
        for point in range(npts):
            if nvm > 11:
                veget_max = veget_max.at[point, 11].set(veget_max_new[point, 11])
            if nvm > 12:
                veget_max = veget_max.at[point, 12].set(veget_max_new[point, 12])
            peat_cover = veget_max[point, peat_pft]
            if bool(peat_cover <= sum_veget_natveg[point]):
                if bool(sum_veget_natveg[point] > min_stomate):
                    for pft in range(nvm):
                        if bool(natural[pft]) and not bool(pasture[pft]) and not bool(is_peat[pft]):
                            adjusted = veget_max_new[point, pft] - veget_max_new[point, pft] / sum_veget_natveg[point] * peat_cover
                            veget_max = veget_max.at[point, pft].set(adjusted)
                else:
                    veget_max = veget_max.at[point, peat_pft].set(0.0)
                    upper = min(13, nvm)
                    veget_max = veget_max.at[point, 0:upper].set(veget_max_new[point, 0:upper])
                    if nvm > 14:
                        veget_max = veget_max.at[point, 14:nvm].set(veget_max_new[point, 14:nvm])
            else:
                veget_max = veget_max.at[point, peat_pft].set(sum_veget_natveg[point])
                for pft in range(nvm):
                    if bool(natural[pft]) and not bool(pasture[pft]) and not bool(is_peat[pft]):
                        veget_max = veget_max.at[point, pft].set(0.0)
                if nvm > 14:
                    veget_max = veget_max.at[point, 14:nvm].set(veget_max_new[point, 14:nvm])

        for point in range(npts):
            for pft in peat_indices:
                if bool(veget_max[point, pft] - veget_max_old[point, pft] > min_stomate) and bool(veget_max_old[point, pft] <= 0.0):
                    new_cn = jnp.where(is_tree[pft], cn_sapl[pft], 1.0)
                    cn_ind = cn_ind.at[point, pft].set(new_cn)
                    ind = ind.at[point, pft].set(veget_max[point, pft] / new_cn)
                    pft_present = pft_present.at[point, pft].set(True)
                    everywhere = everywhere.at[point, pft].set(1.0)
                    senescence = senescence.at[point, pft].set(False)
                    age = age.at[point, pft].set(0.0)
                    when_growthinit = when_growthinit.at[point, pft].set(large_value)
                    leaf_frac = leaf_frac.at[point, pft, 0].set(1.0)
                    npp_longterm = npp_longterm.at[point, pft].set(npp_longterm_init)
                    lm_lastyearmax = lm_lastyearmax.at[point, pft].set(bm_sapl[pft, ILEAF, ICARBON] * ind[point, pft])

    for point in range(npts):
        small = veget_max[point, :] < min_vegfrac
        veget_max = veget_max.at[point, :].set(jnp.where(small, 0.0, veget_max[point, :]))
        total = jnp.sum(veget_max[point, :])
        veget_max = veget_max.at[point, :].set(veget_max[point, :] / jnp.where(total != 0.0, total, 1.0))

    delta_fpeat_all = jnp.zeros((npts,), dtype=veget_max.dtype)
    co2flux_old = jnp.zeros((npts, nvm), dtype=veget_max.dtype)
    co2flux_new = jnp.zeros((npts, nvm), dtype=veget_max.dtype)

    for point in range(npts):
        delta_veg = veget_max[point, :] - veget_max_old[point, :]
        delta_veg_sum = jnp.sum(jnp.where(delta_veg < 0.0, delta_veg, 0.0))
        delta_fpeat = delta_veg[peat_pft]
        delta_fpeat_all = delta_fpeat_all.at[point].set(delta_fpeat)
        exp_nat_sum = jnp.sum(jnp.where((~is_peat) & (delta_veg > min_stomate), delta_veg, 0.0))

        bm_new = jnp.zeros((NPARTS, nelements), dtype=veget_max.dtype)
        if bool(delta_fpeat > min_stomate):
            bm_new = delta_fpeat * bm_sapl[peat_pft, :, :]
        for pft in range(nvm):
            if bool(is_peat[pft]):
                if bool(veget_max[point, pft] > min_stomate):
                    co2_to_bm = co2_to_bm.at[point, pft].add(jnp.sum(bm_new[:, ICARBON]) / (dt_days * veget_max[point, pft]))
                    biomass = biomass.at[point, pft, :, :].set(
                        (biomass[point, pft, :, :] * veget_max_old[point, pft] + bm_new) / veget_max[point, pft]
                    )
            elif bool(veget_max[point, pft] > min_stomate):
                biomass = biomass.at[point, pft, :, :].set(
                    biomass[point, pft, :, :] * veget_max_old[point, pft] / veget_max[point, pft]
                )

        for pft in range(nvm):
            flux = resp_maint[point, pft] + resp_growth[point, pft] + resp_hetero[point, pft] + co2_fire[point, pft]
            flux = flux - co2_to_bm[point, pft] - gpp_daily[point, pft]
            co2flux_old = co2flux_old.at[point, pft].set(flux)
            co2flux_new = co2flux_new.at[point, pft].set(flux)

        dilu_lit = jnp.zeros((NLITT, NLEVS, nelements), dtype=veget_max.dtype)
        dilu_f1hr = jnp.zeros((NLITT, nelements), dtype=veget_max.dtype)
        dilu_f10hr = jnp.zeros((NLITT, nelements), dtype=veget_max.dtype)
        dilu_f100hr = jnp.zeros((NLITT, nelements), dtype=veget_max.dtype)
        dilu_f1000hr = jnp.zeros((NLITT, nelements), dtype=veget_max.dtype)
        dilu_turnover = jnp.zeros((NPARTS, nelements), dtype=veget_max.dtype)
        dilu_bm_to_litter = jnp.zeros((NPARTS, nelements), dtype=veget_max.dtype)
        dilu_co2flux = jnp.asarray(0.0, dtype=veget_max.dtype)
        dilu_gpp = jnp.asarray(0.0, dtype=veget_max.dtype)
        dilu_resp_growth = jnp.asarray(0.0, dtype=veget_max.dtype)
        dilu_resp_maint = jnp.asarray(0.0, dtype=veget_max.dtype)
        dilu_resp_hetero = jnp.asarray(0.0, dtype=veget_max.dtype)
        dilu_co2_to_bm = jnp.asarray(0.0, dtype=veget_max.dtype)
        dilu_co2_fire = jnp.asarray(0.0, dtype=veget_max.dtype)
        dilu_carbon = jnp.zeros((NCARB, nvm), dtype=veget_max.dtype)

        if bool(delta_veg_sum < -min_stomate):
            for pft in range(nvm):
                if bool(delta_veg[pft] < -min_stomate):
                    weight = delta_veg[pft] / delta_veg_sum
                    dilu_lit = dilu_lit + weight * litter[point, :, pft, :, :]
                    dilu_f1hr = dilu_f1hr + weight * fuel_1hr[point, pft, :, :]
                    dilu_f10hr = dilu_f10hr + weight * fuel_10hr[point, pft, :, :]
                    dilu_f100hr = dilu_f100hr + weight * fuel_100hr[point, pft, :, :]
                    dilu_f1000hr = dilu_f1000hr + weight * fuel_1000hr[point, pft, :, :]
                    dilu_turnover = dilu_turnover + weight * turnover_daily[point, pft, :, :]
                    dilu_bm_to_litter = dilu_bm_to_litter + weight * bm_to_litter[point, pft, :, :]
                    dilu_co2flux = dilu_co2flux + weight * co2flux_old[point, pft]
                    dilu_gpp = dilu_gpp + weight * gpp_daily[point, pft]
                    dilu_resp_growth = dilu_resp_growth + weight * resp_growth[point, pft]
                    dilu_resp_maint = dilu_resp_maint + weight * resp_maint[point, pft]
                    dilu_resp_hetero = dilu_resp_hetero + weight * resp_hetero[point, pft]
                    dilu_co2_to_bm = dilu_co2_to_bm + weight * co2_to_bm[point, pft]
                    dilu_co2_fire = dilu_co2_fire + weight * co2_fire[point, pft]

        for pft in range(nvm):
            if bool(delta_veg[pft] > min_stomate):
                denom = veget_max[point, pft]
                litter = litter.at[point, :, pft, :, :].set(
                    (litter[point, :, pft, :, :] * veget_max_old[point, pft] + dilu_lit * delta_veg[pft]) / denom
                )
                fuel_1hr = fuel_1hr.at[point, pft, :, :].set(
                    (fuel_1hr[point, pft, :, :] * veget_max_old[point, pft] + dilu_f1hr * delta_veg[pft]) / denom
                )
                fuel_10hr = fuel_10hr.at[point, pft, :, :].set(
                    (fuel_10hr[point, pft, :, :] * veget_max_old[point, pft] + dilu_f10hr * delta_veg[pft]) / denom
                )
                fuel_100hr = fuel_100hr.at[point, pft, :, :].set(
                    (fuel_100hr[point, pft, :, :] * veget_max_old[point, pft] + dilu_f100hr * delta_veg[pft]) / denom
                )
                fuel_1000hr = fuel_1000hr.at[point, pft, :, :].set(
                    (fuel_1000hr[point, pft, :, :] * veget_max_old[point, pft] + dilu_f1000hr * delta_veg[pft]) / denom
                )
                if bool(is_grassland_manag[pft]) and bool(is_grassland_grazed[pft]):
                    litter_avail = litter_avail.at[point, :, pft].set(litter_avail[point, :, pft] * veget_max_old[point, pft] / denom)
                    litter_not_avail = litter_not_avail.at[point, :, pft].set(
                        litter[point, :, pft, IABOVE, ICARBON] - litter_avail[point, :, pft]
                    )
                turnover_daily = turnover_daily.at[point, pft, :, :].set(
                    (turnover_daily[point, pft, :, :] * veget_max_old[point, pft] + dilu_turnover * delta_veg[pft]) / denom
                )
                bm_to_litter = bm_to_litter.at[point, pft, :, :].set(
                    (bm_to_litter[point, pft, :, :] * veget_max_old[point, pft] + dilu_bm_to_litter * delta_veg[pft]) / denom
                )
                co2flux_new = co2flux_new.at[point, pft].set(
                    (co2flux_old[point, pft] * veget_max_old[point, pft] + dilu_co2flux * delta_veg[pft]) / denom
                )
                gpp_daily = gpp_daily.at[point, pft].set((gpp_daily[point, pft] * veget_max_old[point, pft] + dilu_gpp * delta_veg[pft]) / denom)
                resp_growth = resp_growth.at[point, pft].set(
                    (resp_growth[point, pft] * veget_max_old[point, pft] + dilu_resp_growth * delta_veg[pft]) / denom
                )
                resp_maint = resp_maint.at[point, pft].set(
                    (resp_maint[point, pft] * veget_max_old[point, pft] + dilu_resp_maint * delta_veg[pft]) / denom
                )
                resp_hetero = resp_hetero.at[point, pft].set(
                    (resp_hetero[point, pft] * veget_max_old[point, pft] + dilu_resp_hetero * delta_veg[pft]) / denom
                )
                co2_fire = co2_fire.at[point, pft].set(
                    (co2_fire[point, pft] * veget_max_old[point, pft] + dilu_co2_fire * delta_veg[pft]) / denom
                )
                co2_to_bm = co2_to_bm.at[point, pft].set(
                    (co2_to_bm[point, pft] * veget_max_old[point, pft] + dilu_co2_to_bm * delta_veg[pft]) / denom
                )

        for pft in range(nvm):
            if bool(delta_veg[pft] <= 0.0):
                if bool(is_peat[pft]):
                    save_tmp = carbon_old[point, :, pft] * (-delta_veg[pft])
                    carbon_save_tmp = carbon_save_tmp.at[point, :, pft].set(save_tmp)
                    carbon_save = carbon_save.at[point, :, pft].add(save_tmp)
                    delta_fsave = delta_fsave.at[point].add(-delta_veg[pft])
                    if bool(ok_pc):
                        deepC_a_save = deepC_a_save.at[point, :].add(deepC_a[point, :, pft] * (-delta_veg[pft]))
                        deepC_s_save = deepC_s_save.at[point, :].add(deepC_s[point, :, pft] * (-delta_veg[pft]))
                        deepC_p_save = deepC_p_save.at[point, :].add(deepC_p[point, :, pft] * (-delta_veg[pft]))
                elif bool(delta_veg[pft] < -min_stomate):
                    dilu_carbon = dilu_carbon.at[:, pft].set(-delta_veg[pft] * carbon_old[point, :, pft])

        if bool(delta_fpeat <= 0.0):
            for pft in range(nvm):
                if bool(delta_veg[pft] > min_stomate) and (not bool(is_peat[pft])) and bool(exp_nat_sum > min_stomate):
                    received = jnp.sum(dilu_carbon, axis=1) * delta_veg[pft] / exp_nat_sum
                    received = received + carbon_save_tmp[point, :, peat_pft] * delta_veg[pft] / exp_nat_sum
                    carbon = carbon.at[point, :, pft].set(
                        (carbon_old[point, :, pft] * veget_max_old[point, pft] + received) / veget_max[point, pft]
                    )
                    _scale_all_deep_with_carbon(point, pft, carbon_old[point, :, pft])

        if bool(delta_fpeat > 0.0):
            if bool(delta_fpeat <= delta_fsave[point]) and bool(delta_fsave[point] > 0.0):
                ratio = delta_fpeat / delta_fsave[point]
                carbon_obtain = jnp.minimum(1.0, ratio) * carbon_save[point, :, peat_pft]
                carbon_save = carbon_save.at[point, :, peat_pft].set(jnp.maximum(1.0 - ratio, 0.0) * carbon_save[point, :, peat_pft])
                delta_fsave = delta_fsave.at[point].set(jnp.maximum(delta_fsave[point] - delta_fpeat, 0.0))
                if bool(ok_pc):
                    post_ratio = delta_fpeat / delta_fsave[point]
                    factor = jnp.maximum(1.0 - post_ratio, 0.0)
                    deepC_a_save = deepC_a_save.at[point, :].set(factor * deepC_a_save[point, :])
                    deepC_s_save = deepC_s_save.at[point, :].set(factor * deepC_s_save[point, :])
                    deepC_p_save = deepC_p_save.at[point, :].set(factor * deepC_p_save[point, :])
            else:
                diff_fpeat = delta_fpeat - delta_fsave[point]
                nat_sum = jnp.asarray(0.0, dtype=veget_max.dtype)
                for pft in range(nvm):
                    if bool(natural[pft]) and bool(veget_max_old[point, pft] > min_stomate) and not bool(is_peat[pft]):
                        nat_sum = nat_sum + veget_max_old[point, pft]
                carbon_obtain = carbon_save[point, :, peat_pft]
                if bool(nat_sum > min_stomate):
                    for pft in range(nvm):
                        if bool(natural[pft]) and bool(veget_max_old[point, pft] > min_stomate) and not bool(is_peat[pft]):
                            carbon_obtain = carbon_obtain + diff_fpeat * (veget_max_old[point, pft] / nat_sum) * carbon_old[point, :, pft]
                carbon_save = carbon_save.at[point, :, peat_pft].set(0.0)
                delta_fsave = delta_fsave.at[point].set(0.0)
                if bool(ok_pc):
                    deepC_a_save = deepC_a_save.at[point, :].set(0.0)
                    deepC_s_save = deepC_s_save.at[point, :].set(0.0)
                    deepC_p_save = deepC_p_save.at[point, :].set(0.0)

            for pft in peat_indices:
                carbon = carbon.at[point, :, pft].set(
                    (carbon_old[point, :, pft] * veget_max_old[point, pft] + carbon_obtain) / veget_max[point, pft]
                )
                _scale_all_deep_with_carbon(point, pft, carbon_old[point, :, pft])
            carbon_tmp = carbon[point, :, :]
            for pool in range(NCARB):
                dilu_pool_sum = jnp.sum(dilu_carbon[pool, :])
                if bool(dilu_pool_sum >= carbon_obtain[pool]):
                    if bool(jnp.any(delta_veg[: min(13, nvm)] > min_stomate)):
                        for pft in range(nvm):
                            if bool(delta_veg[pft] > min_stomate) and bool(natural[pft]) and bool(exp_nat_sum > min_stomate) and not bool(is_peat[pft]):
                                rest = (dilu_pool_sum - carbon_obtain[pool]) * delta_veg[pft] / exp_nat_sum
                                carbon = carbon.at[point, pool, pft].set(
                                    (carbon_old[point, pool, pft] * veget_max_old[point, pft] + rest) / veget_max[point, pft]
                                )
                                _scale_deep_with_carbon(pool, point, pft, carbon_old[point, :, pft])
                    else:
                        for pft in peat_indices:
                            carbon = carbon.at[point, pool, pft].add((dilu_pool_sum - carbon_obtain[pool]) / veget_max[point, pft])
                            _scale_deep_with_carbon(pool, point, pft, carbon_tmp[:, pft])
                else:
                    for pft in range(nvm):
                        if bool(delta_veg[pft] > min_stomate) and bool(natural[pft]) and not bool(is_peat[pft]):
                            carbon = carbon.at[point, pool, pft].set(
                                carbon_old[point, pool, pft] * veget_max_old[point, pft] / veget_max[point, pft]
                            )
                    excess = dilu_pool_sum - carbon_obtain[pool]
                    nat_sum_c = jnp.asarray(0.0, dtype=veget_max.dtype)
                    for pft in range(nvm):
                        if bool(natural[pft]) and not bool(is_peat[pft]) and bool(veget_max[point, pft] > min_stomate):
                            nat_sum_c = nat_sum_c + veget_max[point, pft] * carbon[point, pool, pft]
                    for pft in range(nvm):
                        if (
                            bool(natural[pft])
                            and not bool(is_peat[pft])
                            and bool(veget_max[point, pft] > min_stomate)
                            and bool(excess < 0.0)
                            and bool(nat_sum_c > 0.0)
                        ):
                            excess_tmp = jnp.maximum(
                                -carbon[point, pool, pft] * veget_max[point, pft],
                                excess * veget_max[point, pft] * carbon[point, pool, pft] / nat_sum_c,
                            )
                            nat_sum_c = nat_sum_c - veget_max[point, pft] * carbon[point, pool, pft]
                            carbon = carbon.at[point, pool, pft].set(
                                (carbon[point, pool, pft] * veget_max[point, pft] + excess_tmp) / veget_max[point, pft]
                            )
                            _scale_deep_with_carbon(pool, point, pft, carbon_old[point, :, pft])
                            excess = excess - excess_tmp
                    if bool(excess < 0.0):
                        for pft in peat_indices:
                            carbon = carbon.at[point, pool, pft].set(
                                (carbon[point, pool, pft] * veget_max[point, pft] + excess) / veget_max[point, pft]
                            )
                            _scale_deep_with_carbon(pool, point, pft, carbon_tmp[:, pft])
            _ = carbon_tmp

    return LpjCoverPeatResult(
        veget_max=veget_max,
        cn_ind=cn_ind,
        ind=ind,
        biomass=biomass,
        litter=litter,
        litter_avail=litter_avail,
        litter_not_avail=litter_not_avail,
        carbon=carbon,
        fuel_1hr=fuel_1hr,
        fuel_10hr=fuel_10hr,
        fuel_100hr=fuel_100hr,
        fuel_1000hr=fuel_1000hr,
        turnover_daily=turnover_daily,
        bm_to_litter=bm_to_litter,
        co2_to_bm=co2_to_bm,
        co2_fire=co2_fire,
        resp_hetero=resp_hetero,
        resp_maint=resp_maint,
        resp_growth=resp_growth,
        gpp_daily=gpp_daily,
        deepC_a=deepC_a,
        deepC_s=deepC_s,
        deepC_p=deepC_p,
        age=age,
        pft_present=pft_present,
        senescence=senescence,
        when_growthinit=when_growthinit,
        everywhere=everywhere,
        leaf_frac=leaf_frac,
        lm_lastyearmax=lm_lastyearmax,
        npp_longterm=npp_longterm,
        carbon_save=carbon_save,
        deepC_a_save=deepC_a_save,
        deepC_s_save=deepC_s_save,
        deepC_p_save=deepC_p_save,
        delta_fsave=delta_fsave,
        co2flux_old=co2flux_old,
        co2flux_new=co2flux_new,
        delta_fpeat=delta_fpeat_all,
    )


def stomate_lpj_cover_dispatch_step(
    *,
    update_peatfrac,
    done_update_peatfrac=False,
    peat_cover_inputs=None,
    cover_inputs=None,
) -> StomateLpjCoverDispatchResult:
    """Dispatch ``StomateLpj`` section 13 to peat or ordinary cover.

    Fortran provenance: ``src_stomate/stomate_lpj.f90::StomateLpj`` lines
    1358-1380. When ``update_peatfrac`` is true the source calls
    ``lpj_cover_peat``, then clears ``update_peatfrac`` and sets
    ``done_update_peatfrac``. Otherwise it calls ordinary ``lpj_cover::cover``.
    """

    if bool(update_peatfrac):
        if peat_cover_inputs is None:
            raise ValueError("update_peatfrac=True requires explicit peat_cover_inputs")
        peat_cover = lpj_cover_peat_step(**peat_cover_inputs)
        return StomateLpjCoverDispatchResult(
            cover=None,
            peat_cover=peat_cover,
            update_peatfrac=False,
            done_update_peatfrac=True,
            used_peat_cover=True,
        )
    if cover_inputs is None:
        raise ValueError("update_peatfrac=False requires explicit cover_inputs")
    cover = cover_step(**cover_inputs)
    return StomateLpjCoverDispatchResult(
        cover=cover,
        peat_cover=None,
        update_peatfrac=False,
        done_update_peatfrac=bool(done_update_peatfrac),
        used_peat_cover=False,
    )


def agripeat_adjust_fractions_step(
    *,
    veget_max_new,
    veget_max_old,
    natural,
    pasture,
    is_peat,
    pft_to_mtc,
    agri_peat_prop=False,
    agri_peat_mincrop=False,
    agri_peat_maxcrop=False,
    min_stomate=1.0e-8,
) -> AgripeatAdjustedFractionsResult:
    """Apply the hard-coded agricultural-peat vegetation-fraction adjustment.

    Fortran provenance: ``src_stomate/stomate_lcchange.f90``, subroutine
    ``agripeat_adjust_fractions``, lines 1640-1863. The source is explicitly
    hard-coded for PFT12/13 crops, PFT14 natural peat, and PFT15/16 peat crops;
    this helper keeps that boundary explicit instead of generalizing it silently.
    """

    veget_max_new = jnp.asarray(veget_max_new)
    veget_max_old = jnp.asarray(veget_max_old)
    natural = jnp.asarray(natural, dtype=bool)
    pasture = jnp.asarray(pasture, dtype=bool)
    is_peat = jnp.asarray(is_peat, dtype=bool)
    pft_to_mtc = jnp.asarray(pft_to_mtc, dtype=jnp.int32)
    min_stomate = jnp.asarray(min_stomate, dtype=veget_max_new.dtype)

    if veget_max_new.ndim != 2:
        raise ValueError("veget_max_new must have shape (npts, nvm)")
    npts, nvm = veget_max_new.shape
    if veget_max_old.shape != (npts, nvm):
        raise ValueError("veget_max_old must share veget_max_new shape")
    if nvm != 16:
        raise NotImplementedError("agripeat_adjust_fractions_step follows the source hard-coded nvm=16 PFT12-16 path")
    for name, array in (
        ("natural", natural),
        ("pasture", pasture),
        ("is_peat", is_peat),
        ("pft_to_mtc", pft_to_mtc),
    ):
        if array.shape != (nvm,):
            raise ValueError(f"{name} must have shape (nvm,)")
    if not bool(is_peat[13]):
        raise ValueError("source hard-coded agripeat path requires PFT14 (index 13) to be marked as peat")

    veget_max_adjusted = veget_max_new.copy()
    sum_veget_natold = jnp.zeros((npts,), dtype=veget_max_new.dtype)
    sum_veget_natnew = jnp.zeros((npts,), dtype=veget_max_new.dtype)
    sum_veget_nattmp = jnp.zeros((npts,), dtype=veget_max_new.dtype)
    sum_peat_crops_old = jnp.zeros((npts,), dtype=veget_max_new.dtype)
    sum_peat_crops_new = jnp.zeros((npts,), dtype=veget_max_new.dtype)
    sum_crops_new = jnp.zeros((npts,), dtype=veget_max_new.dtype)
    sumpeat_old = jnp.zeros((npts,), dtype=veget_max_new.dtype)

    nat_old_mask = (natural & (~pasture)) | is_peat
    peat_crop_mask = (pft_to_mtc == 16) | (pft_to_mtc == 17)
    nat_not_pasture_mask = natural & (~pasture)

    sum_veget_natold = jnp.sum(jnp.where(nat_old_mask[None, :], veget_max_old, 0.0), axis=1)
    sum_peat_crops_old = jnp.sum(jnp.where(peat_crop_mask[None, :], veget_max_old, 0.0), axis=1)
    for pft in range(nvm):
        if bool(is_peat[pft]):
            sumpeat_old = veget_max_old[:, pft] + sum_peat_crops_old

    if bool(agri_peat_prop):
        veget_max_tmp = jnp.zeros_like(veget_max_new)
        veget_max_tmp = veget_max_tmp.at[:, 0:13].set(veget_max_new[:, 0:13] * (1.0 - sumpeat_old[:, None]))
        veget_max_tmp = veget_max_tmp.at[:, 14].set(sumpeat_old * veget_max_new[:, 11])
        veget_max_tmp = veget_max_tmp.at[:, 15].set(sumpeat_old * veget_max_new[:, 12])
        veget_max_tmp = veget_max_tmp.at[:, 13].set(sumpeat_old - veget_max_tmp[:, 14] - veget_max_tmp[:, 15])

        for point in range(npts):
            if bool((sumpeat_old[point] > 0.0) & (sum_peat_crops_old[point] == 0.0)):
                veget_max_adjusted = veget_max_adjusted.at[point, :].set(veget_max_tmp[point, :])

            if bool(sum_peat_crops_old[point] > 0.0):
                for pft in range(nvm):
                    if bool(nat_not_pasture_mask[pft]):
                        sum_veget_natnew = sum_veget_natnew.at[point].add(veget_max_new[point, pft])
                if bool(sum_veget_natnew[point] <= sum_veget_natold[point]):
                    veget_max_adjusted = veget_max_adjusted.at[point, :].set(veget_max_tmp[point, :])
                else:
                    veget_max_adjusted = veget_max_adjusted.at[point, 13].set(veget_max_old[point, 13])
                    sum_veget_nattmp = sum_veget_nattmp.at[point].set(sum_veget_natnew[point] - veget_max_old[point, 13])
                    for pft in range(nvm):
                        if bool(nat_not_pasture_mask[pft]):
                            adjusted = veget_max_new[point, pft] * sum_veget_nattmp[point] / sum_veget_natnew[point]
                            veget_max_adjusted = veget_max_adjusted.at[point, pft].set(adjusted)
                    veget_max_adjusted = veget_max_adjusted.at[point, 14].set(veget_max_tmp[point, 14])
                    veget_max_adjusted = veget_max_adjusted.at[point, 15].set(veget_max_tmp[point, 15])
                    veget_max_adjusted = veget_max_adjusted.at[point, 11].set(veget_max_tmp[point, 11])
                    veget_max_adjusted = veget_max_adjusted.at[point, 12].set(veget_max_tmp[point, 12])

    if bool(agri_peat_mincrop):
        for point in range(npts):
            for pft in range(nvm):
                if bool(nat_not_pasture_mask[pft]):
                    sum_veget_natnew = sum_veget_natnew.at[point].add(veget_max_new[point, pft])
            sum_crops_new = sum_crops_new.at[point].set(veget_max_new[point, 11] + veget_max_new[point, 12])

            if bool(sum_veget_natnew[point] >= veget_max_old[point, 13]):
                veget_max_adjusted = veget_max_adjusted.at[point, 13].set(veget_max_old[point, 13])
                sum_veget_nattmp = sum_veget_nattmp.at[point].set(sum_veget_natnew[point] - veget_max_adjusted[point, 13])
                for pft in range(nvm):
                    if bool(nat_not_pasture_mask[pft]) and bool(sum_veget_natnew[point] > min_stomate):
                        adjusted = veget_max_new[point, pft] * sum_veget_nattmp[point] / sum_veget_natnew[point]
                        veget_max_adjusted = veget_max_adjusted.at[point, pft].set(adjusted)

                peat_crops = jnp.minimum(sum_crops_new[point], sumpeat_old[point] - veget_max_adjusted[point, 13])
                sum_peat_crops_new = sum_peat_crops_new.at[point].set(peat_crops)
                if bool(sum_crops_new[point] > min_stomate):
                    veget_max_adjusted = veget_max_adjusted.at[point, 14].set(jnp.maximum(0.0, peat_crops * veget_max_new[point, 11] / sum_crops_new[point]))
                    veget_max_adjusted = veget_max_adjusted.at[point, 15].set(jnp.maximum(0.0, peat_crops * veget_max_new[point, 12] / sum_crops_new[point]))
                veget_max_adjusted = veget_max_adjusted.at[point, 11].set(veget_max_new[point, 11] - veget_max_adjusted[point, 14])
                veget_max_adjusted = veget_max_adjusted.at[point, 12].set(veget_max_new[point, 12] - veget_max_adjusted[point, 15])
            else:
                veget_max_adjusted = veget_max_adjusted.at[point, 13].set(sum_veget_natnew[point])
                for pft in range(nvm):
                    if bool(nat_not_pasture_mask[pft]):
                        veget_max_adjusted = veget_max_adjusted.at[point, pft].set(0.0)

                peat_crops = sumpeat_old[point] - veget_max_adjusted[point, 13]
                sum_peat_crops_new = sum_peat_crops_new.at[point].set(peat_crops)
                if bool(sum_crops_new[point] > min_stomate):
                    veget_max_adjusted = veget_max_adjusted.at[point, 14].set(peat_crops * veget_max_new[point, 11] / sum_crops_new[point])
                    veget_max_adjusted = veget_max_adjusted.at[point, 15].set(peat_crops * veget_max_new[point, 12] / sum_crops_new[point])
                veget_max_adjusted = veget_max_adjusted.at[point, 11].set(veget_max_new[point, 11] - veget_max_adjusted[point, 14])
                veget_max_adjusted = veget_max_adjusted.at[point, 12].set(veget_max_new[point, 12] - veget_max_adjusted[point, 15])

    if bool(agri_peat_maxcrop):
        for point in range(npts):
            for pft in range(nvm):
                if bool(nat_not_pasture_mask[pft]):
                    sum_veget_natnew = sum_veget_natnew.at[point].add(veget_max_new[point, pft])
            sum_crops_new = sum_crops_new.at[point].set(veget_max_new[point, 11] + veget_max_new[point, 12])

            if bool(sum_crops_new[point] >= sumpeat_old[point]):
                if bool(sum_crops_new[point] > min_stomate):
                    veget_max_adjusted = veget_max_adjusted.at[point, 14].set(sumpeat_old[point] * veget_max_new[point, 11] / sum_crops_new[point])
                    veget_max_adjusted = veget_max_adjusted.at[point, 15].set(sumpeat_old[point] * veget_max_new[point, 12] / sum_crops_new[point])
                veget_max_adjusted = veget_max_adjusted.at[point, 13].set(0.0)
                veget_max_adjusted = veget_max_adjusted.at[point, 11].set(veget_max_new[point, 11] - veget_max_adjusted[point, 14])
                veget_max_adjusted = veget_max_adjusted.at[point, 12].set(veget_max_new[point, 12] - veget_max_adjusted[point, 15])
                for pft in range(nvm):
                    if bool(nat_not_pasture_mask[pft]):
                        veget_max_adjusted = veget_max_adjusted.at[point, pft].set(veget_max_new[point, pft])
            else:
                veget_max_adjusted = veget_max_adjusted.at[point, 14].set(veget_max_new[point, 11])
                veget_max_adjusted = veget_max_adjusted.at[point, 15].set(veget_max_new[point, 12])
                veget_max_adjusted = veget_max_adjusted.at[point, 11].set(0.0)
                veget_max_adjusted = veget_max_adjusted.at[point, 12].set(0.0)
                peat_remaining = sumpeat_old[point] - veget_max_adjusted[point, 14] - veget_max_adjusted[point, 15]
                veget_max_adjusted = veget_max_adjusted.at[point, 13].set(jnp.minimum(peat_remaining, veget_max_old[point, 13]))
                sum_veget_nattmp = sum_veget_nattmp.at[point].set(sum_veget_natnew[point] - veget_max_adjusted[point, 13])
                for pft in range(nvm):
                    if bool(nat_not_pasture_mask[pft]):
                        adjusted = veget_max_new[point, pft] * sum_veget_nattmp[point] / sum_veget_natnew[point]
                        veget_max_adjusted = veget_max_adjusted.at[point, pft].set(adjusted)

    mass_error = jnp.sum(veget_max_adjusted, axis=1) - jnp.sum(veget_max_new, axis=1)
    return AgripeatAdjustedFractionsResult(
        veget_max_adjusted=veget_max_adjusted,
        sum_veget_natold=sum_veget_natold,
        sum_veget_natnew=sum_veget_natnew,
        sum_crops_new=sum_crops_new,
        sum_peat_crops_old=sum_peat_crops_old,
        sum_peat_crops_new=sum_peat_crops_new,
        sumpeat_old=sumpeat_old,
        mass_error=jnp.where(jnp.abs(mass_error) > min_stomate, mass_error, 0.0),
    )


def lcchange_agripeat_redistribution_step(
    *,
    veget_max_adjusted,
    veget_max_old,
    biomass,
    litter,
    litter_avail,
    litter_not_avail,
    bm_to_litter,
    carbon,
    deepC_a,
    deepC_s,
    deepC_p,
    coeff_lcchange_1,
    coeff_lcchange_10,
    coeff_lcchange_100,
    is_peat,
    is_grassland_manag=None,
    is_grassland_grazed=None,
    ok_pc=False,
    min_stomate=1.0e-8,
) -> AgripeatRedistributionResult:
    """Redistribute agripeat litter/biomass/carbon after adjusted fractions.

    Fortran provenance: ``src_stomate/stomate_lcchange.f90``,
    ``lcchange_main_agripeat`` lines 1342-1558. This is the local carbon and
    litter redistribution kernel after ``agripeat_adjust_fractions`` has already
    produced ``veget_max_adjusted``. Product-pool aging and fuel rebalancing are
    separate source blocks and remain outside this helper.
    """

    veget_max_adjusted = jnp.asarray(veget_max_adjusted)
    veget_max_old = jnp.asarray(veget_max_old)
    biomass = jnp.asarray(biomass)
    litter = jnp.asarray(litter)
    litter_avail = jnp.asarray(litter_avail)
    litter_not_avail = jnp.asarray(litter_not_avail)
    bm_to_litter = jnp.asarray(bm_to_litter)
    carbon = jnp.asarray(carbon)
    deepC_a = jnp.asarray(deepC_a)
    deepC_s = jnp.asarray(deepC_s)
    deepC_p = jnp.asarray(deepC_p)
    coeff_lcchange_1 = jnp.asarray(coeff_lcchange_1)
    coeff_lcchange_10 = jnp.asarray(coeff_lcchange_10)
    coeff_lcchange_100 = jnp.asarray(coeff_lcchange_100)
    is_peat = jnp.asarray(is_peat, dtype=bool)
    min_stomate = jnp.asarray(min_stomate, dtype=veget_max_adjusted.dtype)

    if veget_max_adjusted.ndim != 2:
        raise ValueError("veget_max_adjusted must have shape (npts, nvm)")
    npts, nvm = veget_max_adjusted.shape
    if nvm != 16:
        raise NotImplementedError("lcchange_agripeat_redistribution_step follows source hard-coded PFT14-16 agripeat path")
    if veget_max_old.shape != (npts, nvm):
        raise ValueError("veget_max_old must share veget_max_adjusted shape")
    if biomass.ndim != 4 or biomass.shape[:2] != (npts, nvm) or biomass.shape[2] != NPARTS:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    nelements = biomass.shape[3]
    if litter.shape != (npts, NLITT, nvm, NLEVS, nelements):
        raise ValueError("litter must have shape (npts, nlitt, nvm, nlevs, nelements)")
    if litter_avail.shape != (npts, NLITT, nvm) or litter_not_avail.shape != (npts, NLITT, nvm):
        raise ValueError("litter_avail and litter_not_avail must have shape (npts, nlitt, nvm)")
    if bm_to_litter.shape != biomass.shape:
        raise ValueError("bm_to_litter must share biomass shape")
    if carbon.shape != (npts, NCARB, nvm):
        raise ValueError("carbon must have shape (npts, ncarb, nvm)")
    if deepC_a.shape != deepC_s.shape or deepC_a.shape != deepC_p.shape or deepC_a.shape[:1] != (npts,) or deepC_a.shape[2] != nvm:
        raise ValueError("deepC_* must have shape (npts, ndeep, nvm)")
    if is_peat.shape != (nvm,) or not bool(is_peat[13]):
        raise ValueError("source hard-coded agripeat path requires is_peat shape (nvm,) with PFT14 marked")
    for name, coeff in (
        ("coeff_lcchange_1", coeff_lcchange_1),
        ("coeff_lcchange_10", coeff_lcchange_10),
        ("coeff_lcchange_100", coeff_lcchange_100),
    ):
        if coeff.shape != (nvm,):
            raise ValueError(f"{name} must have shape (nvm,)")
    if is_grassland_manag is None:
        is_grassland_manag = jnp.zeros((nvm,), dtype=bool)
    else:
        is_grassland_manag = jnp.asarray(is_grassland_manag, dtype=bool)
    if is_grassland_grazed is None:
        is_grassland_grazed = jnp.zeros((nvm,), dtype=bool)
    else:
        is_grassland_grazed = jnp.asarray(is_grassland_grazed, dtype=bool)
    if is_grassland_manag.shape != (nvm,) or is_grassland_grazed.shape != (nvm,):
        raise ValueError("is_grassland_manag and is_grassland_grazed must have shape (nvm,)")

    convflux = jnp.zeros((npts,), dtype=veget_max_adjusted.dtype)
    prod10_entry = jnp.zeros((npts,), dtype=veget_max_adjusted.dtype)
    prod100_entry = jnp.zeros((npts,), dtype=veget_max_adjusted.dtype)
    delta_nat_peat_all = veget_max_adjusted[:, 13] - veget_max_old[:, 13]
    litter_parts = (ISAPBELOW, IHEARTBELOW, IROOT, IFRUIT, ICARBRES, ILEAF)

    for point in range(npts):
        delta_veg = veget_max_adjusted[point, :] - veget_max_old[point, :]
        delta_nat_peat = delta_veg[13]
        dilu_lit = jnp.zeros((NLITT, NLEVS, nelements), dtype=veget_max_adjusted.dtype)
        dilu_lit_peat = jnp.zeros((NLITT, NLEVS, nelements), dtype=veget_max_adjusted.dtype)
        biomass_loss = jnp.zeros((NPARTS, nelements), dtype=veget_max_adjusted.dtype)
        biomass_loss_peat = jnp.zeros((NPARTS, nelements), dtype=veget_max_adjusted.dtype)
        dilu_soil_carbon = jnp.zeros((NCARB,), dtype=veget_max_adjusted.dtype)
        dilu_deep_a = jnp.zeros((deepC_a.shape[1],), dtype=veget_max_adjusted.dtype)
        dilu_deep_s = jnp.zeros((deepC_a.shape[1],), dtype=veget_max_adjusted.dtype)
        dilu_deep_p = jnp.zeros((deepC_a.shape[1],), dtype=veget_max_adjusted.dtype)
        delta_a = jnp.zeros((deepC_a.shape[1],), dtype=veget_max_adjusted.dtype)
        delta_s = jnp.zeros((deepC_a.shape[1],), dtype=veget_max_adjusted.dtype)
        delta_p = jnp.zeros((deepC_a.shape[1],), dtype=veget_max_adjusted.dtype)
        delta_carbon = jnp.zeros((NCARB,), dtype=veget_max_adjusted.dtype)

        def _add_product_entry(pft, delta):
            nonlocal convflux, prod10_entry, prod100_entry
            above = biomass[point, pft, ISAPABOVE, ICARBON] + biomass[point, pft, IHEARTABOVE, ICARBON]
            convflux = convflux.at[point].add(-(coeff_lcchange_1[pft] * above * delta))
            prod10_entry = prod10_entry.at[point].add(-(coeff_lcchange_10[pft] * above * delta))
            prod100_entry = prod100_entry.at[point].add(-(coeff_lcchange_100[pft] * above * delta))

        def _set_expanding_pft_from_buffers(pft, source_lit, source_bm, source_carbon, source_a, source_s, source_p):
            nonlocal litter, litter_avail, litter_not_avail, bm_to_litter, carbon, deepC_a, deepC_s, deepC_p
            old_cover = veget_max_old[point, pft]
            new_cover = veget_max_adjusted[point, pft]
            litter = litter.at[point, :, pft, :, :].set((litter[point, :, pft, :, :] * old_cover + source_lit) / new_cover)
            if bool(is_grassland_manag[pft]) and bool(is_grassland_grazed[pft]):
                litter_avail = litter_avail.at[point, :, pft].set(litter_avail[point, :, pft] * old_cover / new_cover)
                litter_not_avail = litter_not_avail.at[point, :, pft].set(litter[point, :, pft, IABOVE, ICARBON] - litter_avail[point, :, pft])
            for part in litter_parts:
                bm_to_litter = bm_to_litter.at[point, pft, part, :].set(
                    (bm_to_litter[point, pft, part, :] * old_cover + source_bm[part, :]) / new_cover
                )
            carbon = carbon.at[point, :, pft].set((carbon[point, :, pft] * old_cover + source_carbon) / new_cover)
            if bool(ok_pc):
                deepC_a = deepC_a.at[point, :, pft].set((deepC_a[point, :, pft] * old_cover + source_a) / new_cover)
                deepC_s = deepC_s.at[point, :, pft].set((deepC_s[point, :, pft] * old_cover + source_s) / new_cover)
                deepC_p = deepC_p.at[point, :, pft].set((deepC_p[point, :, pft] * old_cover + source_p) / new_cover)

        if bool(delta_nat_peat < -min_stomate):
            for pft in range(nvm):
                if bool(is_peat[pft]):
                    delta = delta_veg[pft]
                    biomass_loss_peat = biomass_loss_peat - biomass[point, pft, :, :] * delta
                    dilu_lit_peat = dilu_lit_peat - delta * litter[point, :, pft, :, :]
                    delta_a = delta_a - deepC_a[point, :, pft] * delta
                    delta_s = delta_s - deepC_s[point, :, pft] * delta
                    delta_p = delta_p - deepC_p[point, :, pft] * delta
                    delta_carbon = delta_carbon - carbon[point, :, pft] * delta
                    _add_product_entry(pft, delta)

            if bool((delta_veg[14] > min_stomate) & (delta_veg[15] > min_stomate)):
                for pft in (14, 15):
                    ratio = delta_veg[pft] / delta_nat_peat
                    _set_expanding_pft_from_buffers(
                        pft,
                        -dilu_lit_peat * ratio,
                        -biomass_loss_peat * ratio,
                        -delta_carbon * ratio,
                        -delta_a * ratio,
                        -delta_s * ratio,
                        -delta_p * ratio,
                    )
            else:
                for pft in (14, 15):
                    if bool(delta_veg[pft] <= 0.0):
                        delta = delta_veg[pft]
                        biomass_loss_peat = biomass_loss_peat - biomass[point, pft, :, :] * delta
                        dilu_lit_peat = dilu_lit_peat - delta * litter[point, :, pft, :, :]
                        delta_a = delta_a - deepC_a[point, :, pft] * delta
                        delta_s = delta_s - deepC_s[point, :, pft] * delta
                        delta_p = delta_p - deepC_p[point, :, pft] * delta
                        delta_carbon = delta_carbon - carbon[point, :, pft] * delta
                        _add_product_entry(pft, delta)
                for pft in (14, 15):
                    if bool(delta_veg[pft] > 0.0):
                        _set_expanding_pft_from_buffers(pft, dilu_lit_peat, biomass_loss_peat, delta_carbon, delta_a, delta_s, delta_p)

            delta_veg_sum = jnp.sum(jnp.where(delta_veg[0:13] < 0.0, delta_veg[0:13], 0.0))
            for pft in range(13):
                if bool(delta_veg[pft] < -min_stomate):
                    dilu_lit = dilu_lit + delta_veg[pft] * litter[point, :, pft, :, :] / delta_veg_sum
                    biomass_loss = biomass_loss + biomass[point, pft, :, :] * delta_veg[pft] / delta_veg_sum
                    dilu_soil_carbon = dilu_soil_carbon + delta_veg[pft] * carbon[point, :, pft] / delta_veg_sum
                    if bool(ok_pc):
                        dilu_deep_a = dilu_deep_a + delta_veg[pft] * deepC_a[point, :, pft] / delta_veg_sum
                        dilu_deep_s = dilu_deep_s + delta_veg[pft] * deepC_s[point, :, pft] / delta_veg_sum
                        dilu_deep_p = dilu_deep_p + delta_veg[pft] * deepC_p[point, :, pft] / delta_veg_sum
            for pft in range(13):
                if bool(delta_veg[pft] > min_stomate):
                    _set_expanding_pft_from_buffers(
                        pft,
                        dilu_lit * delta_veg[pft],
                        biomass_loss * delta_veg[pft],
                        dilu_soil_carbon * delta_veg[pft],
                        dilu_deep_a * delta_veg[pft],
                        dilu_deep_s * delta_veg[pft],
                        dilu_deep_p * delta_veg[pft],
                    )
                else:
                    _add_product_entry(pft, delta_veg[pft])
        else:
            delta_veg_sum = jnp.sum(jnp.where(delta_veg < 0.0, delta_veg, 0.0))
            for pft in range(nvm):
                if bool(delta_veg[pft] < -min_stomate):
                    dilu_lit = dilu_lit + delta_veg[pft] * litter[point, :, pft, :, :] / delta_veg_sum
                    biomass_loss = biomass_loss + biomass[point, pft, :, :] * delta_veg[pft] / delta_veg_sum
                    dilu_soil_carbon = dilu_soil_carbon + delta_veg[pft] * carbon[point, :, pft] / delta_veg_sum
                    if bool(ok_pc):
                        dilu_deep_a = dilu_deep_a + delta_veg[pft] * deepC_a[point, :, pft] / delta_veg_sum
                        dilu_deep_s = dilu_deep_s + delta_veg[pft] * deepC_s[point, :, pft] / delta_veg_sum
                        dilu_deep_p = dilu_deep_p + delta_veg[pft] * deepC_p[point, :, pft] / delta_veg_sum
            for pft in range(nvm):
                if bool(delta_veg[pft] > min_stomate):
                    _set_expanding_pft_from_buffers(
                        pft,
                        dilu_lit * delta_veg[pft],
                        biomass_loss * delta_veg[pft],
                        dilu_soil_carbon * delta_veg[pft],
                        dilu_deep_a * delta_veg[pft],
                        dilu_deep_s * delta_veg[pft],
                        dilu_deep_p * delta_veg[pft],
                    )
                else:
                    _add_product_entry(pft, delta_veg[pft])

    return AgripeatRedistributionResult(
        litter=litter,
        litter_avail=litter_avail,
        litter_not_avail=litter_not_avail,
        bm_to_litter=bm_to_litter,
        carbon=carbon,
        deepC_a=deepC_a,
        deepC_s=deepC_s,
        deepC_p=deepC_p,
        convflux=convflux,
        prod10_entry=prod10_entry,
        prod100_entry=prod100_entry,
        delta_nat_peat=delta_nat_peat_all,
    )


def lcchange_agripeat_finalize_step(
    *,
    prod10,
    prod100,
    flux10,
    flux100,
    convflux,
    cflux_prod10,
    cflux_prod100,
    litter,
    fuel_1hr,
    fuel_10hr,
    fuel_100hr,
    fuel_1000hr,
    dt_days=1.0,
    one_year=ONE_YEAR_DAYS,
    min_stomate=1.0e-8,
) -> AgripeatFinalizeResult:
    """Age agripeat product pools and rebalance fuel pools from litter.

    Fortran provenance: ``src_stomate/stomate_lcchange.f90``,
    ``lcchange_main_agripeat`` lines 1561-1613. The product-pool loop is the
    same source algebra as the generic LCC product-pool aging helper; the fuel
    block preserves existing fuel-type fractions where total fuel is nonzero
    and uses 0.25 for each class otherwise.
    """

    prod10 = jnp.asarray(prod10)
    prod100 = jnp.asarray(prod100)
    flux10 = jnp.asarray(flux10)
    flux100 = jnp.asarray(flux100)
    convflux = jnp.asarray(convflux)
    cflux_prod10 = jnp.asarray(cflux_prod10)
    cflux_prod100 = jnp.asarray(cflux_prod100)
    litter = jnp.asarray(litter)
    fuel_1hr = jnp.asarray(fuel_1hr)
    fuel_10hr = jnp.asarray(fuel_10hr)
    fuel_100hr = jnp.asarray(fuel_100hr)
    fuel_1000hr = jnp.asarray(fuel_1000hr)
    min_stomate = jnp.asarray(min_stomate, dtype=prod10.dtype)

    if prod10.ndim != 2 or prod10.shape[1] != 11:
        raise ValueError("prod10 must have shape (npts, 11), mapping Fortran 0:10")
    if prod100.ndim != 2 or prod100.shape[1] != 101:
        raise ValueError("prod100 must have shape (npts, 101), mapping Fortran 0:100")
    npts = prod10.shape[0]
    if prod100.shape[0] != npts or flux10.shape != (npts, 10) or flux100.shape != (npts, 100):
        raise ValueError("product and flux pools must share npts and Fortran pool lengths")
    if convflux.shape != (npts,) or cflux_prod10.shape != (npts,) or cflux_prod100.shape != (npts,):
        raise ValueError("convflux and cflux arrays must have shape (npts,)")
    if litter.ndim != 5 or litter.shape[0] != npts or litter.shape[1] != NLITT:
        raise ValueError("litter must have shape (npts, nlitt, nvm, nlevs, nelements)")
    nvm = litter.shape[2]
    nelements = litter.shape[4]
    fuel_shape = (npts, nvm, NLITT, nelements)
    if fuel_1hr.shape != fuel_shape or fuel_10hr.shape != fuel_shape or fuel_100hr.shape != fuel_shape or fuel_1000hr.shape != fuel_shape:
        raise ValueError("fuel arrays must have shape (npts, nvm, nlitt, nelements)")

    aged = product_pool_aging_step(
        prod10=prod10[:, :, None],
        prod100=prod100[:, :, None],
        flux10=flux10[:, :, None],
        flux100=flux100[:, :, None],
        convflux=convflux[:, None],
        cflux_prod10=cflux_prod10[:, None],
        cflux_prod100=cflux_prod100[:, None],
        dt_days=dt_days,
        one_year=one_year,
    )

    fuel_all = fuel_1hr + fuel_10hr + fuel_100hr + fuel_1000hr
    fuel_1hr_frac = jnp.where(fuel_all > min_stomate, fuel_1hr / fuel_all, 0.25)
    fuel_10hr_frac = jnp.where(fuel_all > min_stomate, fuel_10hr / fuel_all, 0.25)
    fuel_100hr_frac = jnp.where(fuel_all > min_stomate, fuel_100hr / fuel_all, 0.25)
    fuel_1000hr_frac = jnp.where(fuel_all > min_stomate, fuel_1000hr / fuel_all, 0.25)
    above_litter_by_pft = jnp.swapaxes(litter[:, :, :, IABOVE, :], 1, 2)

    return AgripeatFinalizeResult(
        prod10=aged.prod10[:, :, 0],
        prod100=aged.prod100[:, :, 0],
        flux10=aged.flux10[:, :, 0],
        flux100=aged.flux100[:, :, 0],
        convflux=aged.convflux[:, 0],
        cflux_prod10=aged.cflux_prod10[:, 0],
        cflux_prod100=aged.cflux_prod100[:, 0],
        fuel_1hr=above_litter_by_pft * fuel_1hr_frac,
        fuel_10hr=above_litter_by_pft * fuel_10hr_frac,
        fuel_100hr=above_litter_by_pft * fuel_100hr_frac,
        fuel_1000hr=above_litter_by_pft * fuel_1000hr_frac,
    )


def lcchange_main_agripeat_step(
    *,
    dt_days,
    veget_max,
    veget_max_old,
    biomass,
    ind,
    age,
    pft_present,
    senescence,
    when_growthinit,
    everywhere,
    co2_to_bm,
    bm_to_litter,
    turnover_daily,
    bm_sapl,
    cn_ind,
    flux10,
    flux100,
    prod10,
    prod100,
    leaf_frac,
    npp_longterm,
    lm_lastyearmax,
    litter,
    litter_avail,
    litter_not_avail,
    carbon,
    deepC_a,
    deepC_s,
    deepC_p,
    fuel_1hr,
    fuel_10hr,
    fuel_100hr,
    fuel_1000hr,
    natural,
    pasture,
    is_peat,
    is_tree,
    pft_to_mtc,
    coeff_lcchange_1,
    coeff_lcchange_10,
    coeff_lcchange_100,
    cn_sapl,
    is_grassland_manag=None,
    is_grassland_grazed=None,
    agri_peat_prop=False,
    agri_peat_mincrop=False,
    agri_peat_maxcrop=False,
    ok_pc=False,
    min_stomate=1.0e-8,
    one_year=ONE_YEAR_DAYS,
    npp_longterm_init=10.0,
    large_value=1.0e33,
    undef=DEFAULT_UNDEF,
) -> LcchangeMainAgripeatResult:
    """Compose the hard-coded agricultural-peat LCC subroutine.

    Fortran provenance: ``src_stomate/stomate_lcchange.f90``,
    ``lcchange_main_agripeat`` lines 1113-1632. This wrapper follows the source
    order: adjust hard-coded PFT12-16 fractions, initialize expanding PFTs,
    clear exhausted PFTs, redistribute agripeat carbon/litter/product entries,
    then age product pools and rebalance fuel. It does not generalize beyond the
    source hard-coded 16-PFT agricultural peat layout.
    """

    veget_max = jnp.asarray(veget_max)
    veget_max_old = jnp.asarray(veget_max_old)
    biomass = jnp.asarray(biomass)
    ind = jnp.asarray(ind)
    age = jnp.asarray(age)
    pft_present = jnp.asarray(pft_present, dtype=bool)
    senescence = jnp.asarray(senescence, dtype=bool)
    when_growthinit = jnp.asarray(when_growthinit)
    everywhere = jnp.asarray(everywhere)
    co2_to_bm = jnp.asarray(co2_to_bm)
    bm_to_litter = jnp.asarray(bm_to_litter)
    turnover_daily = jnp.asarray(turnover_daily)
    bm_sapl = jnp.asarray(bm_sapl)
    cn_ind = jnp.asarray(cn_ind)
    flux10 = jnp.asarray(flux10)
    flux100 = jnp.asarray(flux100)
    prod10 = jnp.asarray(prod10)
    prod100 = jnp.asarray(prod100)
    leaf_frac = jnp.asarray(leaf_frac)
    npp_longterm = jnp.asarray(npp_longterm)
    lm_lastyearmax = jnp.asarray(lm_lastyearmax)
    litter = jnp.asarray(litter)
    litter_avail = jnp.asarray(litter_avail)
    litter_not_avail = jnp.asarray(litter_not_avail)
    carbon = jnp.asarray(carbon)
    deepC_a = jnp.asarray(deepC_a)
    deepC_s = jnp.asarray(deepC_s)
    deepC_p = jnp.asarray(deepC_p)
    fuel_1hr = jnp.asarray(fuel_1hr)
    fuel_10hr = jnp.asarray(fuel_10hr)
    fuel_100hr = jnp.asarray(fuel_100hr)
    fuel_1000hr = jnp.asarray(fuel_1000hr)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    cn_sapl = jnp.asarray(cn_sapl)
    min_stomate = jnp.asarray(min_stomate, dtype=veget_max.dtype)
    dt_days = jnp.asarray(dt_days, dtype=veget_max.dtype)
    one_year = jnp.asarray(one_year, dtype=veget_max.dtype)
    npp_longterm_init = jnp.asarray(npp_longterm_init, dtype=veget_max.dtype)
    large_value = jnp.asarray(large_value, dtype=veget_max.dtype)
    undef = jnp.asarray(undef, dtype=veget_max.dtype)

    if veget_max.ndim != 2:
        raise ValueError("veget_max must have shape (npts, nvm)")
    npts, nvm = veget_max.shape
    if nvm != 16:
        raise NotImplementedError("lcchange_main_agripeat_step follows source hard-coded nvm=16 PFT12-16 path")
    if veget_max_old.shape != (npts, nvm):
        raise ValueError("veget_max_old must share veget_max shape")
    scalar_shape = (npts, nvm)
    for name, arr in (
        ("ind", ind),
        ("age", age),
        ("pft_present", pft_present),
        ("senescence", senescence),
        ("when_growthinit", when_growthinit),
        ("everywhere", everywhere),
        ("co2_to_bm", co2_to_bm),
        ("cn_ind", cn_ind),
        ("npp_longterm", npp_longterm),
        ("lm_lastyearmax", lm_lastyearmax),
    ):
        if arr.shape != scalar_shape:
            raise ValueError(f"{name} must have shape (npts, nvm)")
    if biomass.ndim != 4 or biomass.shape[:2] != (npts, nvm) or biomass.shape[2] != NPARTS:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    nelements = biomass.shape[3]
    if bm_to_litter.shape != biomass.shape or turnover_daily.shape != biomass.shape:
        raise ValueError("bm_to_litter and turnover_daily must share biomass shape")
    if bm_sapl.shape != (nvm, NPARTS, nelements):
        raise ValueError("bm_sapl must have shape (nvm, nparts, nelements)")
    if leaf_frac.shape != (npts, nvm, NLEAFAGES):
        raise ValueError("leaf_frac must have shape (npts, nvm, nleafages)")
    if litter.shape != (npts, NLITT, nvm, NLEVS, nelements):
        raise ValueError("litter must have shape (npts, nlitt, nvm, nlevs, nelements)")
    if litter_avail.shape != (npts, NLITT, nvm) or litter_not_avail.shape != (npts, NLITT, nvm):
        raise ValueError("litter_avail and litter_not_avail must have shape (npts, nlitt, nvm)")
    if carbon.shape != (npts, NCARB, nvm):
        raise ValueError("carbon must have shape (npts, ncarb, nvm)")
    if deepC_a.shape != deepC_s.shape or deepC_a.shape != deepC_p.shape or deepC_a.shape[:1] != (npts,) or deepC_a.shape[2] != nvm:
        raise ValueError("deepC_* must have shape (npts, ndeep, nvm)")
    if flux10.shape != (npts, 10) or flux100.shape != (npts, 100):
        raise ValueError("flux10/flux100 must have shapes (npts,10)/(npts,100)")
    if prod10.shape != (npts, 11) or prod100.shape != (npts, 101):
        raise ValueError("prod10/prod100 must have shapes (npts,11)/(npts,101)")
    fuel_shape = (npts, nvm, NLITT, nelements)
    if fuel_1hr.shape != fuel_shape or fuel_10hr.shape != fuel_shape or fuel_100hr.shape != fuel_shape or fuel_1000hr.shape != fuel_shape:
        raise ValueError("fuel arrays must have shape (npts, nvm, nlitt, nelements)")
    for name, arr in (
        ("is_tree", is_tree),
        ("cn_sapl", cn_sapl),
    ):
        if arr.shape != (nvm,):
            raise ValueError(f"{name} must have shape (nvm,)")

    adjusted = agripeat_adjust_fractions_step(
        veget_max_new=veget_max,
        veget_max_old=veget_max_old,
        natural=natural,
        pasture=pasture,
        is_peat=is_peat,
        pft_to_mtc=pft_to_mtc,
        agri_peat_prop=agri_peat_prop,
        agri_peat_mincrop=agri_peat_mincrop,
        agri_peat_maxcrop=agri_peat_maxcrop,
        min_stomate=min_stomate,
    )
    veget_max_adjusted = adjusted.veget_max_adjusted

    prod10 = prod10.at[:, 0].set(0.0)
    prod100 = prod100.at[:, 0].set(0.0)

    for point in range(npts):
        delta_veg = veget_max_adjusted[point, :] - veget_max_old[point, :]
        for pft in range(1, nvm):
            if bool(delta_veg[pft] > min_stomate):
                if bool(veget_max_old[point, pft] < min_stomate):
                    new_cn = jnp.where(is_tree[pft], cn_sapl[pft], 1.0)
                    cn_ind = cn_ind.at[point, pft].set(new_cn)
                    ind = ind.at[point, pft].set(delta_veg[pft] / new_cn)
                    pft_present = pft_present.at[point, pft].set(True)
                    everywhere = everywhere.at[point, pft].set(1.0)
                    senescence = senescence.at[point, pft].set(False)
                    age = age.at[point, pft].set(0.0)
                    when_growthinit = when_growthinit.at[point, pft].set(large_value)
                    leaf_frac = leaf_frac.at[point, pft, 0].set(1.0)
                    npp_longterm = npp_longterm.at[point, pft].set(npp_longterm_init)
                    lm_lastyearmax = lm_lastyearmax.at[point, pft].set(bm_sapl[pft, ILEAF, ICARBON] * ind[point, pft])
                delta_ind = jnp.where(cn_ind[point, pft] > min_stomate, delta_veg[pft] / cn_ind[point, pft], 0.0)
                for part in range(NPARTS):
                    for element in range(nelements):
                        bm_new = delta_ind * bm_sapl[pft, part, element]
                        if bool(veget_max_old[point, pft] > min_stomate):
                            over_existing = (bm_new / delta_veg[pft]) > biomass[point, pft, part, element]
                            bm_new = jnp.where(over_existing, biomass[point, pft, part, element] * delta_veg[pft], bm_new)
                        biomass = biomass.at[point, pft, part, element].set(
                            (biomass[point, pft, part, element] * veget_max_old[point, pft] + bm_new) / veget_max_adjusted[point, pft]
                        )
                        if element == ICARBON:
                            co2_to_bm = co2_to_bm.at[point, pft].add(bm_new * dt_days / (one_year * veget_max_adjusted[point, pft]))
            else:
                if bool(veget_max_adjusted[point, pft] < min_stomate):
                    ind = ind.at[point, pft].set(0.0)
                    biomass = biomass.at[point, pft, :, :].set(0.0)
                    pft_present = pft_present.at[point, pft].set(False)
                    senescence = senescence.at[point, pft].set(False)
                    age = age.at[point, pft].set(0.0)
                    when_growthinit = when_growthinit.at[point, pft].set(undef)
                    everywhere = everywhere.at[point, pft].set(0.0)
                    carbon = carbon.at[point, :, pft].set(0.0)
                    litter = litter.at[point, :, pft, :, :].set(0.0)
                    bm_to_litter = bm_to_litter.at[point, pft, :, :].set(0.0)
                    turnover_daily = turnover_daily.at[point, pft, :, :].set(0.0)
                    if bool(ok_pc):
                        deepC_a = deepC_a.at[point, :, pft].set(0.0)
                        deepC_s = deepC_s.at[point, :, pft].set(0.0)
                        deepC_p = deepC_p.at[point, :, pft].set(0.0)

    redistribution = lcchange_agripeat_redistribution_step(
        veget_max_adjusted=veget_max_adjusted,
        veget_max_old=veget_max_old,
        biomass=biomass,
        litter=litter,
        litter_avail=litter_avail,
        litter_not_avail=litter_not_avail,
        bm_to_litter=bm_to_litter,
        carbon=carbon,
        deepC_a=deepC_a,
        deepC_s=deepC_s,
        deepC_p=deepC_p,
        coeff_lcchange_1=coeff_lcchange_1,
        coeff_lcchange_10=coeff_lcchange_10,
        coeff_lcchange_100=coeff_lcchange_100,
        is_peat=is_peat,
        is_grassland_manag=is_grassland_manag,
        is_grassland_grazed=is_grassland_grazed,
        ok_pc=ok_pc,
        min_stomate=min_stomate,
    )
    prod10 = prod10.at[:, 0].set(redistribution.prod10_entry)
    prod100 = prod100.at[:, 0].set(redistribution.prod100_entry)
    cflux_prod10 = jnp.zeros((npts,), dtype=veget_max.dtype)
    cflux_prod100 = jnp.zeros((npts,), dtype=veget_max.dtype)

    finalized = lcchange_agripeat_finalize_step(
        prod10=prod10,
        prod100=prod100,
        flux10=flux10,
        flux100=flux100,
        convflux=redistribution.convflux,
        cflux_prod10=cflux_prod10,
        cflux_prod100=cflux_prod100,
        litter=redistribution.litter,
        fuel_1hr=fuel_1hr,
        fuel_10hr=fuel_10hr,
        fuel_100hr=fuel_100hr,
        fuel_1000hr=fuel_1000hr,
        dt_days=dt_days,
        one_year=one_year,
        min_stomate=min_stomate,
    )

    return LcchangeMainAgripeatResult(
        veget_max_adjusted=veget_max_adjusted,
        biomass=biomass,
        ind=ind,
        age=age,
        pft_present=pft_present,
        senescence=senescence,
        when_growthinit=when_growthinit,
        everywhere=everywhere,
        co2_to_bm=co2_to_bm,
        bm_to_litter=redistribution.bm_to_litter,
        turnover_daily=turnover_daily,
        cn_ind=cn_ind,
        prod10=finalized.prod10,
        prod100=finalized.prod100,
        flux10=finalized.flux10,
        flux100=finalized.flux100,
        convflux=finalized.convflux,
        cflux_prod10=finalized.cflux_prod10,
        cflux_prod100=finalized.cflux_prod100,
        leaf_frac=leaf_frac,
        npp_longterm=npp_longterm,
        lm_lastyearmax=lm_lastyearmax,
        litter=redistribution.litter,
        litter_avail=redistribution.litter_avail,
        litter_not_avail=redistribution.litter_not_avail,
        carbon=redistribution.carbon,
        deepC_a=redistribution.deepC_a,
        deepC_s=redistribution.deepC_s,
        deepC_p=redistribution.deepC_p,
        fuel_1hr=finalized.fuel_1hr,
        fuel_10hr=finalized.fuel_10hr,
        fuel_100hr=finalized.fuel_100hr,
        fuel_1000hr=finalized.fuel_1000hr,
        adjusted_fractions=adjusted,
        redistribution=redistribution,
        finalize=finalized,
    )


def vmax_step(
    *,
    leaf_age,
    leaf_frac,
    vcmax25,
    n_limfert,
    leaf_timecst,
    leafagecrit,
    pheno_type,
    leaf_tab,
    ok_laidev,
    dt_days=1.0,
    ok_dgvm=False,
    ok_nlim=False,
    vmax_offset=0.3,
    leafage_firstmax=0.03,
    leafage_lastmax=0.5,
    leafage_old=1.0,
    min_stomate=0.0,
) -> VmaxResult:
    """Advance leaf ages and compute ``vcmax`` from ``stomate_vmax::vmax``.

    Fortran provenance: ``src_stomate/stomate_vmax.f90``, subroutine
    ``vmax``, lines 105-363. Scalar defaults come from
    ``src_parameters/constantes_var.f90`` lines 1417-1431; PFT parameters
    ``Vcmax25``, ``leaf_timecst``, ``leafagecrit``, ``pheno_type``,
    ``leaf_tab``, and ``ok_LAIdev`` are allocated/read in
    ``src_parameters/pft_parameters_var.f90`` and
    ``src_parameters/pft_parameters.f90``.
    """

    leaf_age = jnp.asarray(leaf_age)
    leaf_frac = jnp.asarray(leaf_frac)
    vcmax25 = jnp.asarray(vcmax25)
    n_limfert = jnp.asarray(n_limfert)
    leaf_timecst = jnp.asarray(leaf_timecst)
    leafagecrit = jnp.asarray(leafagecrit)
    pheno_type = jnp.asarray(pheno_type)
    leaf_tab = jnp.asarray(leaf_tab)
    ok_laidev = jnp.asarray(ok_laidev, dtype=bool)

    if leaf_age.shape != leaf_frac.shape or leaf_age.ndim != 3 or leaf_age.shape[2] != NLEAFAGES:
        raise ValueError("leaf_age and leaf_frac must have shape (npts, nvm, nleafages)")
    npts, nvm, _ = leaf_age.shape
    if n_limfert.shape != (npts, nvm):
        raise ValueError("n_limfert must have shape (npts, nvm)")
    for name, arr in {
        "vcmax25": vcmax25,
        "leaf_timecst": leaf_timecst,
        "leafagecrit": leafagecrit,
        "pheno_type": pheno_type,
        "leaf_tab": leaf_tab,
        "ok_laidev": ok_laidev,
    }.items():
        if arr.shape != (nvm,):
            raise ValueError(f"{name} must have shape (nvm,)")

    pft_active_axis = (jnp.arange(nvm) > 0)[None, :, None]
    positive_fraction = (leaf_frac > min_stomate) & pft_active_axis
    leaf_age = jnp.where(positive_fraction, leaf_age + dt_days, leaf_age)

    zero_leaf_class = jnp.zeros_like(leaf_frac[:, :, :1])
    safe_leaf_timecst = jnp.where(
        pft_active_axis,
        leaf_timecst[None, :, None],
        1.0,
    )
    d_leaf_frac = jnp.concatenate(
        (
            zero_leaf_class,
            leaf_frac[:, :, : NLEAFAGES - 1]
            * dt_days
            / safe_leaf_timecst,
        ),
        axis=2,
    )
    d_leaf_frac = jnp.where(pft_active_axis, d_leaf_frac, 0.0)

    denom_mid = leaf_frac[:, :, 1 : NLEAFAGES - 1] + d_leaf_frac[:, :, 1 : NLEAFAGES - 1] - d_leaf_frac[:, :, 2:NLEAFAGES]
    update_mid = (
        d_leaf_frac[:, :, 1 : NLEAFAGES - 1] > min_stomate
    )
    safe_denom_mid = jnp.where(update_mid, denom_mid, 1.0)
    updated_mid = (
        (leaf_frac[:, :, 1 : NLEAFAGES - 1] - d_leaf_frac[:, :, 2:NLEAFAGES])
        * leaf_age[:, :, 1 : NLEAFAGES - 1]
        + d_leaf_frac[:, :, 1 : NLEAFAGES - 1] * leaf_age[:, :, : NLEAFAGES - 2]
    ) / safe_denom_mid
    denom_last = leaf_frac[:, :, NLEAFAGES - 1] + d_leaf_frac[:, :, NLEAFAGES - 1]
    update_last = d_leaf_frac[:, :, NLEAFAGES - 1] > min_stomate
    safe_denom_last = jnp.where(update_last, denom_last, 1.0)
    updated_last = (
        leaf_frac[:, :, NLEAFAGES - 1] * leaf_age[:, :, NLEAFAGES - 1]
        + d_leaf_frac[:, :, NLEAFAGES - 1] * leaf_age[:, :, NLEAFAGES - 2]
    ) / safe_denom_last
    updated_tail = jnp.concatenate((updated_mid, updated_last[:, :, None]), axis=2)
    update_tail = jnp.concatenate(
        (update_mid, update_last[:, :, None]),
        axis=2,
    )
    leaf_age = jnp.concatenate(
        (
            leaf_age[:, :, :1],
            jnp.where(update_tail, updated_tail, leaf_age[:, :, 1:NLEAFAGES]),
        ),
        axis=2,
    )

    updated_frac = jnp.stack(
        (
            leaf_frac[:, :, 0] - d_leaf_frac[:, :, 1],
            leaf_frac[:, :, 1] + d_leaf_frac[:, :, 1] - d_leaf_frac[:, :, 2],
            leaf_frac[:, :, 2] + d_leaf_frac[:, :, 2] - d_leaf_frac[:, :, 3],
            leaf_frac[:, :, 3] + d_leaf_frac[:, :, 3],
        ),
        axis=2,
    )
    updated_frac = jnp.maximum(0.0, updated_frac)
    sumfrac = jnp.sum(updated_frac, axis=2)
    normalized = updated_frac / jnp.where(sumfrac[:, :, None] != 0.0, sumfrac[:, :, None], 1.0)
    updated_frac = jnp.where(sumfrac[:, :, None] > min_stomate, normalized, 0.0)
    leaf_frac = jnp.where(pft_active_axis, updated_frac, leaf_frac)

    evergreen_dgvm = jnp.asarray(ok_dgvm, dtype=bool) & (pheno_type == 1) & (leaf_tab == 2)
    efficiency_active = pft_active_axis & (
        ~evergreen_dgvm[None, :, None]
    )
    safe_leafagecrit = jnp.where(
        efficiency_active,
        leafagecrit[None, :, None],
        1.0,
    )
    rel_age = leaf_age / safe_leafagecrit
    leaf_efficiency = jnp.maximum(
        vmax_offset,
        jnp.minimum(
            1.0,
            jnp.minimum(
                vmax_offset + (1.0 - vmax_offset) * rel_age / leafage_firstmax,
                1.0 - (1.0 - vmax_offset) * (rel_age - leafage_lastmax) / (leafage_old - leafage_lastmax),
            ),
        ),
    )
    leaf_efficiency_all = jnp.where(
        efficiency_active,
        leaf_efficiency,
        0.0,
    )
    if bool(ok_nlim):
        n_factor = n_limfert
    else:
        n_factor = jnp.where(ok_laidev[None, :], n_limfert, 1.0)
    weighted_vcmax = jnp.sum(
        vcmax25[None, :, None] * n_factor[:, :, None] * leaf_efficiency * leaf_frac,
        axis=2,
    )
    vcmax = jnp.where(evergreen_dgvm[None, :], vcmax25[None, :], weighted_vcmax)
    vcmax = jnp.where(pft_active_axis[:, :, 0], vcmax, 0.0)

    return VmaxResult(
        leaf_age=leaf_age,
        leaf_frac=leaf_frac,
        vcmax=vcmax,
        leaf_efficiency=leaf_efficiency_all,
    )


class GrasslandUserCutScheduleResult(NamedTuple):
    """User-defined grassland cut flags before ``cutting_spa`` execution."""

    flag_cutting_by_stocking: jnp.ndarray
    tcut_verif: jnp.ndarray
    compt_cut: jnp.ndarray
    when_growthinit_cut: jnp.ndarray
    lm_before: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/grassland_management.f90::main_grassland_management lines 1872-1897",
    )


class GrasslandAutoCutScheduleResult(NamedTuple):
    """Auto-fauche cut flags for one source scheduling test."""

    flag_cutting: jnp.ndarray
    countschedule: jnp.ndarray
    compt_cut: jnp.ndarray
    when_growthinit_cut: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/grassland_management.f90::main_grassland_management lines 1975-1993, 2075-2089, 2177-2211, 2296-2314",
    )


class GrasslandPreAnimalStatusResult(NamedTuple):
    """Grassland development and regrowth state before animal/cutting modules."""

    devstage: jnp.ndarray
    tgrowth: jnp.ndarray
    tcut0: jnp.ndarray
    tacumm: jnp.ndarray
    tsoilcumm: jnp.ndarray
    tacummprev: jnp.ndarray
    tsoilcummprev: jnp.ndarray
    tamean1: jnp.ndarray
    tamean2: jnp.ndarray
    tamean3: jnp.ndarray
    tamean4: jnp.ndarray
    tamean5: jnp.ndarray
    tamean6: jnp.ndarray
    tameand: jnp.ndarray
    tameanw: jnp.ndarray
    tsoilmeand: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/grassland_management.f90::Main_appl_pre_animal lines 2553-2590",
        "fortran_source/ORCHIDEE/src_stomate/grassland_management.f90::cal_devstage lines 2596-2670",
        "fortran_source/ORCHIDEE/src_stomate/grassland_management.f90::cal_tgrowth lines 2673-2722",
        "fortran_source/ORCHIDEE/src_stomate/grassland_fonctions.f90::Euler_funct lines 44-58",
    )


def grassland_user_cut_schedule_step(
    *,
    tjulian,
    dt,
    tcut,
    tcut_verif,
    compt_cut,
    when_growthinit_cut,
    biomass,
    is_grassland_manag,
    lm_before=None,
    f_autogestion=0,
    f_postauto=0,
) -> GrasslandUserCutScheduleResult:
    """Schedule user-defined grassland cuts before ``cutting_spa`` execution.

    Fortran provenance: ``src_stomate/grassland_management.f90``,
    ``main_grassland_management`` lines 1872-1897. The source enters this
    block only when ``f_autogestion == 0`` and ``f_postauto == 0``, increments
    ``when_growthinit_cut`` by ``dt``, saves leaf carbon in ``lm_before``, then
    loops over stocking classes and managed non-bare-soil PFTs to mark cuts
    whose ``tjulian`` is within ``[tcut, tcut + 0.9]`` and not yet verified.
    It resets ``flag_cutting`` for each stocking class before the later
    ``cutting_spa`` call; this helper therefore returns one flag matrix per
    stocking class and does not execute the cut itself.
    """

    tcut = jnp.asarray(tcut)
    tcut_verif = jnp.asarray(tcut_verif, dtype=bool)
    compt_cut = jnp.asarray(compt_cut)
    when_growthinit_cut = jnp.asarray(when_growthinit_cut)
    biomass = jnp.asarray(biomass)
    is_grassland_manag = jnp.asarray(is_grassland_manag, dtype=bool)
    if lm_before is None:
        lm_before_arr = jnp.zeros_like(when_growthinit_cut, dtype=biomass.dtype)
    else:
        lm_before_arr = jnp.asarray(lm_before)

    if tcut.ndim != 3:
        raise ValueError("tcut must have shape (npts, nvm, nstocking)")
    npts, nvm, nstocking = tcut.shape
    if tcut_verif.shape != (npts, nvm, nstocking):
        raise ValueError("tcut_verif must have shape (npts, nvm, nstocking)")
    if compt_cut.shape != (npts, nvm):
        raise ValueError("compt_cut must have shape (npts, nvm)")
    if when_growthinit_cut.shape != (npts, nvm):
        raise ValueError("when_growthinit_cut must have shape (npts, nvm)")
    if lm_before_arr.shape != (npts, nvm):
        raise ValueError("lm_before must have shape (npts, nvm)")
    if is_grassland_manag.shape != (nvm,):
        raise ValueError("is_grassland_manag must have shape (nvm,)")
    if biomass.ndim != 4 or biomass.shape[0] != npts or biomass.shape[1] != nvm:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")

    flag_by_stocking = jnp.zeros((nstocking, npts, nvm), dtype=jnp.int32)
    if int(f_autogestion) != 0 or int(f_postauto) != 0:
        return GrasslandUserCutScheduleResult(
            flag_cutting_by_stocking=flag_by_stocking,
            tcut_verif=tcut_verif,
            compt_cut=compt_cut,
            when_growthinit_cut=when_growthinit_cut,
            lm_before=lm_before_arr,
        )

    when_growthinit_cut = when_growthinit_cut + dt
    lm_before_arr = biomass[:, :, ILEAF, ICARBON]

    for stocking in range(nstocking):
        flag_cutting = jnp.zeros((npts, nvm), dtype=jnp.int32)
        for pft in range(1, nvm):
            if bool(is_grassland_manag[pft]) and bool(jnp.any(~tcut_verif[:, pft, stocking])):
                due = (
                    (tjulian >= tcut[:, pft, stocking])
                    & (tjulian <= tcut[:, pft, stocking] + 0.9)
                    & (~tcut_verif[:, pft, stocking])
                )
                due_int = due.astype(compt_cut.dtype)
                tcut_verif = tcut_verif.at[:, pft, stocking].set(jnp.where(due, True, tcut_verif[:, pft, stocking]))
                flag_cutting = flag_cutting.at[:, pft].set(jnp.where(due, 1, flag_cutting[:, pft]))
                compt_cut = compt_cut.at[:, pft].set(compt_cut[:, pft] + due_int)
                when_growthinit_cut = when_growthinit_cut.at[:, pft].set(
                    jnp.where(due, 0.0, when_growthinit_cut[:, pft])
                )
        flag_by_stocking = flag_by_stocking.at[stocking].set(flag_cutting)

    return GrasslandUserCutScheduleResult(
        flag_cutting_by_stocking=flag_by_stocking,
        tcut_verif=tcut_verif,
        compt_cut=compt_cut,
        when_growthinit_cut=when_growthinit_cut,
        lm_before=lm_before_arr,
    )


def grassland_pre_animal_status_step(
    *,
    dt,
    tjulian,
    t2m_daily,
    tsoil,
    new_day,
    new_year,
    regcount,
    tcut,
    devstage,
    tgrowth,
    tcut0,
    tacumm,
    tsoilcumm,
    tacummprev,
    tsoilcummprev,
    tamean1,
    tamean2,
    tamean3,
    tamean4,
    tamean5,
    tamean6,
    tameand,
    tameanw,
    tsoilmeand,
    trep,
    tbase,
    tasumrep,
) -> GrasslandPreAnimalStatusResult:
    """Update grassland ``devstage`` and ``tgrowth`` from explicit SAVE state.

    Fortran provenance: ``src_stomate/grassland_management.f90``,
    ``Main_appl_pre_animal`` lines 2553-2590 calls ``cal_devstage`` and
    ``cal_tgrowth``. ``cal_devstage`` lines 2596-2670 integrates air/soil
    temperature with ``Euler_funct`` and advances development stage from the
    rolling seven-day air-temperature memory. ``cal_tgrowth`` lines 2673-2722
    records the first positive-development day in ``tcut0`` and derives days
    since first growth or since the previous cut. The temperature thresholds
    are grassland-management parameters and are explicit inputs here.
    """

    t2m_daily = jnp.asarray(t2m_daily)
    tsoil = jnp.asarray(tsoil)
    regcount = jnp.asarray(regcount)
    tcut = jnp.asarray(tcut)
    devstage = jnp.asarray(devstage)
    tgrowth = jnp.asarray(tgrowth)
    tcut0 = jnp.asarray(tcut0)
    tacumm = jnp.asarray(tacumm)
    tsoilcumm = jnp.asarray(tsoilcumm)
    tacummprev = jnp.asarray(tacummprev)
    tsoilcummprev = jnp.asarray(tsoilcummprev)
    tamean1 = jnp.asarray(tamean1)
    tamean2 = jnp.asarray(tamean2)
    tamean3 = jnp.asarray(tamean3)
    tamean4 = jnp.asarray(tamean4)
    tamean5 = jnp.asarray(tamean5)
    tamean6 = jnp.asarray(tamean6)
    tameand = jnp.asarray(tameand)
    tameanw = jnp.asarray(tameanw)
    tsoilmeand = jnp.asarray(tsoilmeand)

    if devstage.ndim != 2:
        raise ValueError("devstage must have shape (npts, nvm)")
    npts, nvm = devstage.shape
    if t2m_daily.shape != (npts,) or tsoil.shape != (npts,):
        raise ValueError("t2m_daily and tsoil must have shape (npts,)")
    for name, arr in (
        ("regcount", regcount),
        ("tgrowth", tgrowth),
        ("tcut0", tcut0),
    ):
        if arr.shape != (npts, nvm):
            raise ValueError(f"{name} must have shape (npts, nvm)")
    if tcut.ndim != 3 or tcut.shape[:2] != (npts, nvm):
        raise ValueError("tcut must have shape (npts, nvm, nstocking)")
    for name, arr in (
        ("tacumm", tacumm),
        ("tsoilcumm", tsoilcumm),
        ("tacummprev", tacummprev),
        ("tsoilcummprev", tsoilcummprev),
        ("tamean1", tamean1),
        ("tamean2", tamean2),
        ("tamean3", tamean3),
        ("tamean4", tamean4),
        ("tamean5", tamean5),
        ("tamean6", tamean6),
        ("tameand", tameand),
        ("tameanw", tameanw),
        ("tsoilmeand", tsoilmeand),
    ):
        if arr.shape != (npts,):
            raise ValueError(f"{name} must have shape (npts,)")

    if bool(new_year):
        tcut0 = jnp.zeros_like(tcut0)

    tacumm = tacumm + dt * t2m_daily
    tsoilcumm = tsoilcumm + dt * tsoil

    if bool(new_day):
        old_tamean2 = tamean2
        old_tamean3 = tamean3
        old_tamean4 = tamean4
        old_tamean5 = tamean5
        old_tamean6 = tamean6
        old_tameand = tameand
        tamean1 = old_tamean2
        tamean2 = old_tamean3
        tamean3 = old_tamean4
        tamean4 = old_tamean5
        tamean5 = old_tamean6
        tamean6 = old_tameand
        tameand = tacumm - tacummprev
        tacummprev = tacumm
        tameanw = (tamean1 + tamean2 + tamean3 + tamean4 + tamean5 + tamean6 + tameand) / 7.0
        tsoilmeand = tsoilcumm - tsoilcummprev
        tsoilcummprev = tsoilcumm

        dev_increment = jnp.maximum(0.0, tameand - tbase) / tasumrep
        for pft in range(1, nvm):
            start = (devstage[:, pft] <= 0.0) & ((tameanw > trep) | (regcount[:, pft] == 2))
            grow = (devstage[:, pft] > 0.0) & (tsoilmeand > tbase) & (devstage[:, pft] < 2.0)
            updated = jnp.where(start, dev_increment, jnp.where(grow, devstage[:, pft] + dev_increment, devstage[:, pft]))
            devstage = devstage.at[:, pft].set(updated)

    if bool(new_year):
        devstage = jnp.zeros_like(devstage)

    if bool(new_day):
        for pft in range(1, nvm):
            first_growth = (devstage[:, pft] > 0.0) & (tcut0[:, pft] <= 0.0)
            tcut0 = tcut0.at[:, pft].set(jnp.where(first_growth, float(tjulian), tcut0[:, pft]))
            reg = regcount[:, pft]
            first_cut_wait = (reg == 1) & (tcut0[:, pft] <= 0.0)
            first_cut_growth = reg == 1
            if bool(jnp.any(reg < 1)):
                raise ValueError("regcount must be at least 1 for grassland tgrowth")
            previous_cut_index = reg.astype(jnp.int32) - 2
            if bool(jnp.any(previous_cut_index >= tcut.shape[2])):
                raise ValueError("regcount points outside tcut cut axis")
            clipped_index = jnp.maximum(previous_cut_index, 0)
            previous_cut = jnp.take_along_axis(tcut[:, pft, :], clipped_index[:, None], axis=1)[:, 0]
            from_first_growth = jnp.asarray(tjulian, dtype=tgrowth.dtype) - tcut0[:, pft]
            from_previous_cut = jnp.asarray(tjulian, dtype=tgrowth.dtype) - previous_cut
            updated_growth = jnp.where(first_cut_wait, 0.0, jnp.where(first_cut_growth, from_first_growth, from_previous_cut))
            tgrowth = tgrowth.at[:, pft].set(updated_growth)

    if bool(new_year):
        tgrowth = jnp.zeros_like(tgrowth)

    return GrasslandPreAnimalStatusResult(
        devstage=devstage,
        tgrowth=tgrowth,
        tcut0=tcut0,
        tacumm=tacumm,
        tsoilcumm=tsoilcumm,
        tacummprev=tacummprev,
        tsoilcummprev=tsoilcummprev,
        tamean1=tamean1,
        tamean2=tamean2,
        tamean3=tamean3,
        tamean4=tamean4,
        tamean5=tamean5,
        tamean6=tamean6,
        tameand=tameand,
        tameanw=tameanw,
        tsoilmeand=tsoilmeand,
    )


def grassland_auto_cut_schedule_step(
    *,
    dt,
    new_day,
    phase,
    countschedule,
    compt_cut,
    when_growthinit_cut,
    nanimal,
    cuttingend,
    tgrowth,
    gmean,
    lai,
    devstage,
    gmeanslope,
    mugmean,
    tasum,
    is_grassland_manag,
    is_grassland_cut,
    is_grassland_grazed,
    f_autogestion,
    f_postauto,
    mcut_c3=None,
    mcut_c4=None,
    tgrowthmin=45.0,
    devstagemin=1.0,
    gmeansloperel=0.05,
    tasumrep=500.0,
) -> GrasslandAutoCutScheduleResult:
    """Schedule one auto-fauche cutting-test pass without executing the cut.

    Fortran provenance: ``src_stomate/grassland_management.f90``,
    ``main_grassland_management`` lines 1975-1993 and 2075-2089 cover the
    ``f_autogestion == 1`` automatic-management PFT loop. Lines 2177-2211 and
    2296-2314 cover the postauto/managed-cut C3/C4 role path. The first test
    is selected with ``phase="growth"`` and applies the growth-slope criterion;
    the second test is selected with ``phase="tasum"`` and applies the
    accumulated-temperature criterion. The Fortran source executes
    ``cutting_spa`` between these tests, so this helper intentionally returns
    only the flags and local scheduler state for one pass.
    """

    countschedule = jnp.asarray(countschedule)
    compt_cut = jnp.asarray(compt_cut)
    when_growthinit_cut = jnp.asarray(when_growthinit_cut)
    nanimal = jnp.asarray(nanimal)
    cuttingend = jnp.asarray(cuttingend)
    tgrowth = jnp.asarray(tgrowth)
    gmean = jnp.asarray(gmean)
    lai = jnp.asarray(lai)
    devstage = jnp.asarray(devstage)
    gmeanslope = jnp.asarray(gmeanslope)
    mugmean = jnp.asarray(mugmean)
    tasum = jnp.asarray(tasum)
    is_grassland_manag = jnp.asarray(is_grassland_manag, dtype=bool)
    is_grassland_cut = jnp.asarray(is_grassland_cut, dtype=bool)
    is_grassland_grazed = jnp.asarray(is_grassland_grazed, dtype=bool)

    if countschedule.ndim != 2:
        raise ValueError("countschedule must have shape (npts, nvm)")
    npts, nvm = countschedule.shape
    for name, arr in (
        ("compt_cut", compt_cut),
        ("when_growthinit_cut", when_growthinit_cut),
        ("cuttingend", cuttingend),
        ("tgrowth", tgrowth),
        ("lai", lai),
        ("devstage", devstage),
        ("gmeanslope", gmeanslope),
        ("mugmean", mugmean),
        ("tasum", tasum),
    ):
        if arr.shape != (npts, nvm):
            raise ValueError(f"{name} must have shape (npts, nvm)")
    if nanimal.ndim != 3 or nanimal.shape[:2] != (npts, nvm):
        raise ValueError("nanimal must have shape (npts, nvm, nstocking)")
    if gmean.ndim != 3 or gmean.shape[:2] != (npts, nvm):
        raise ValueError("gmean must have shape (npts, nvm, ngmean)")
    for name, arr in (
        ("is_grassland_manag", is_grassland_manag),
        ("is_grassland_cut", is_grassland_cut),
        ("is_grassland_grazed", is_grassland_grazed),
    ):
        if arr.shape != (nvm,):
            raise ValueError(f"{name} must have shape (nvm,)")
    if phase not in {"growth", "tasum"}:
        raise ValueError("phase must be 'growth' or 'tasum'")

    flag_cutting = jnp.zeros((npts, nvm), dtype=jnp.int32)
    if not bool(new_day):
        return GrasslandAutoCutScheduleResult(
            flag_cutting=flag_cutting,
            countschedule=countschedule,
            compt_cut=compt_cut,
            when_growthinit_cut=when_growthinit_cut,
        )

    auto_loop = int(f_postauto) == 0 and int(f_autogestion) == 1
    role_loop = int(f_postauto) == 1 or int(f_autogestion) in (3, 4) or int(f_postauto) >= 2
    if not (auto_loop or role_loop):
        return GrasslandAutoCutScheduleResult(
            flag_cutting=flag_cutting,
            countschedule=countschedule,
            compt_cut=compt_cut,
            when_growthinit_cut=when_growthinit_cut,
        )

    if phase == "growth":
        when_growthinit_cut = when_growthinit_cut + dt

    if auto_loop:
        active_pfts = [
            pft
            for pft in range(1, nvm)
            if bool(is_grassland_manag[pft]) and not bool(is_grassland_cut[pft]) and not bool(is_grassland_grazed[pft])
        ]
    else:
        if mcut_c3 is None or mcut_c4 is None:
            raise ValueError("mcut_c3 and mcut_c4 are required for postauto/autogestion role auto-fauche")
        active_pfts = [int(mcut_c3), int(mcut_c4)]

    for pft in active_pfts:
        if pft < 0 or pft >= nvm:
            raise ValueError("auto-fauche PFT index outside 0..nvm-1")
        if phase == "growth":
            due = (
                (nanimal[:, pft, 0] == 0.0)
                & (cuttingend[:, pft] == 0)
                & (countschedule[:, pft] == 1)
                & (tgrowth[:, pft] >= tgrowthmin)
                & (gmean[:, pft, -1] > 0.0)
                & (lai[:, pft] >= 2.5)
                & (devstage[:, pft] > devstagemin)
                & (gmeanslope[:, pft] < gmeansloperel * mugmean[:, pft])
            )
        else:
            due = (
                (countschedule[:, pft] == 1)
                & (nanimal[:, pft, 0] == 0.0)
                & (devstage[:, pft] < 2.0)
                & (tasum[:, pft] >= tasumrep)
                & (lai[:, pft] >= 2.5)
            )
        flag_cutting = flag_cutting.at[:, pft].set(jnp.where(due, 1, flag_cutting[:, pft]))
        countschedule = countschedule.at[:, pft].set(countschedule[:, pft] + due.astype(countschedule.dtype))
        compt_cut = compt_cut.at[:, pft].set(compt_cut[:, pft] + due.astype(compt_cut.dtype))
        when_growthinit_cut = when_growthinit_cut.at[:, pft].set(jnp.where(due, 0.0, when_growthinit_cut[:, pft]))

    return GrasslandAutoCutScheduleResult(
        flag_cutting=flag_cutting,
        countschedule=countschedule,
        compt_cut=compt_cut,
        when_growthinit_cut=when_growthinit_cut,
    )


def harvest_agri_step(
    *,
    veget_max,
    bm_to_litter,
    turnover_daily,
    natural,
    is_peat,
    frac_turnover_daily=0.55,
) -> HarvestAgriResult:
    """Apply agricultural harvest turnover reduction from ``StomateLpj``.

    Fortran provenance: ``src_stomate/stomate_lpj.f90``, subroutine
    ``harvest``, lines 2502-2566. The controlling flag is
    ``HARVEST_AGRI``/``harvest_agri`` in ``src_parameters/constantes.f90``
    lines 114-121 and ``src_parameters/constantes_var.f90`` lines 477-478.
    ``frac_turnover_daily`` default comes from
    ``src_parameters/constantes_var.f90`` lines 1253-1254. Although
    ``bm_to_litter`` is an inout argument in Fortran, this subroutine does not
    update it.
    """

    veget_max = jnp.asarray(veget_max)
    bm_to_litter = jnp.asarray(bm_to_litter)
    turnover_daily = jnp.asarray(turnover_daily)
    natural = jnp.asarray(natural, dtype=bool)
    is_peat = jnp.asarray(is_peat, dtype=bool)

    if turnover_daily.ndim != 4:
        raise ValueError("turnover_daily must have shape (npts, nvm, nparts, nelements)")
    npts, nvm = turnover_daily.shape[:2]
    if veget_max.shape != (npts, nvm):
        raise ValueError("veget_max must have shape (npts, nvm)")
    if natural.shape != (nvm,) or is_peat.shape != (nvm,):
        raise ValueError("natural and is_peat must have shape (nvm,)")

    harvest_parts = jnp.asarray(
        [
            ILEAF,
            ISAPABOVE,
            IHEARTABOVE,
            IFRUIT,
            ICARBRES,
            ISAPBELOW,
            IHEARTBELOW,
            IROOT,
            IAGRSAPST,
            IAGRSAPPN,
            IAGRHRTST,
            IAGRHRTPN,
        ]
    )
    above_old = jnp.zeros((npts, nvm), dtype=turnover_daily.dtype)
    harvest_above = jnp.zeros((npts,), dtype=turnover_daily.dtype)
    for pft in range(nvm):
        if (not bool(natural[pft])) and (not bool(is_peat[pft])):
            old = jnp.sum(turnover_daily[:, pft, harvest_parts, ICARBON], axis=1)
            above_old = above_old.at[:, pft].set(old)
            turnover_daily = turnover_daily.at[:, pft, harvest_parts, ICARBON].set(
                turnover_daily[:, pft, harvest_parts, ICARBON] * frac_turnover_daily
            )
            harvest_above = harvest_above + veget_max[:, pft] * old * (1.0 - frac_turnover_daily)

    return HarvestAgriResult(
        bm_to_litter=bm_to_litter,
        turnover_daily=turnover_daily,
        harvest_above=harvest_above,
        above_old=above_old,
    )


def grassland_role_selection_step(
    *,
    enable_grazing,
    is_grassland_manag,
    is_grassland_cut,
    is_grassland_grazed,
    is_c4,
    is_tree,
    natural,
    reject_missing=True,
) -> GrasslandRoleSelectionResult:
    """Select grassland management role PFTs before the large grazing driver.

    Fortran provenance: ``src_stomate/stomate_lpj.f90::StomateLpj`` lines
    1321-1353 calls ``main_grassland_management`` only when
    ``enable_grazing`` is true. ``src_stomate/grassland_management.f90``
    lines 970-1037 select the C3/C4 auto/cut/grazed/natural PFT roles from
    `is_grassland_*`, `is_c4`, `is_tree`, and `natural`. The original source
    maps missing roles to bare-soil PFT 1; this JAX helper rejects missing
    roles by default so unsupported management setups do not silently proceed.
    Returned indices are Python zero-based.
    """

    arrays = {
        "is_grassland_manag": jnp.asarray(is_grassland_manag, dtype=bool),
        "is_grassland_cut": jnp.asarray(is_grassland_cut, dtype=bool),
        "is_grassland_grazed": jnp.asarray(is_grassland_grazed, dtype=bool),
        "is_c4": jnp.asarray(is_c4, dtype=bool),
        "is_tree": jnp.asarray(is_tree, dtype=bool),
        "natural": jnp.asarray(natural, dtype=bool),
    }
    shapes = {name: arr.shape for name, arr in arrays.items()}
    if len(set(shapes.values())) != 1 or next(iter(shapes.values())) == ():
        raise ValueError("grassland role masks must all have matching shape (nvm,)")
    nvm = arrays["natural"].shape[0]

    def _empty():
        return GrasslandRoleSelectionResult(
            enabled=False,
            mauto_c3=-1,
            mcut_c3=-1,
            mgraze_c3=-1,
            mnatural_c3=-1,
            mauto_c4=-1,
            mcut_c4=-1,
            mgraze_c4=-1,
            mnatural_c4=-1,
        )

    if not bool(enable_grazing):
        return _empty()

    def _select(mask, role_name):
        candidates = [pft for pft in range(1, nvm) if bool(mask[pft])]
        if not candidates:
            if bool(reject_missing):
                raise ValueError(f"missing grassland management role: {role_name}")
            return 0
        return candidates[-1]

    manag = arrays["is_grassland_manag"]
    cut = arrays["is_grassland_cut"]
    grazed = arrays["is_grassland_grazed"]
    c4 = arrays["is_c4"]
    tree = arrays["is_tree"]
    nat = arrays["natural"]
    herb = ~tree
    auto = manag & (~cut) & (~grazed) & herb
    cut_only = cut & (~grazed) & herb
    graze_only = manag & grazed & (~cut) & herb
    natural_unmanaged = (~manag) & (~grazed) & (~cut) & herb & nat

    return GrasslandRoleSelectionResult(
        enabled=True,
        mauto_c3=_select(auto & (~c4), "mauto_C3"),
        mcut_c3=_select(cut_only & (~c4), "mcut_C3"),
        mgraze_c3=_select(graze_only & (~c4), "mgraze_C3"),
        mnatural_c3=_select(natural_unmanaged & (~c4), "mnatural_C3"),
        mauto_c4=_select(auto & c4, "mauto_C4"),
        mcut_c4=_select(cut_only & c4, "mcut_C4"),
        mgraze_c4=_select(graze_only & c4, "mgraze_C4"),
        mnatural_c4=_select(natural_unmanaged & c4, "mnatural_C4"),
    )


def grassland_cutting_spa_step(
    *,
    tjulian,
    flag_cutting,
    wshtotcutinit,
    lcutinit,
    wsh,
    wshtot,
    wr,
    c,
    n,
    napo,
    nsym,
    fn,
    nel,
    biomass,
    devstage,
    regcount,
    gmean,
    tasum,
    tgrowth,
    lai,
    tcut,
    tcut_modif,
    wshtotsum,
    controle_azote_sum,
    wshtotsumprev,
    f_autogestion=0,
    f_postauto=0,
    c_to_dm=0.45,
    mc=28.5,
    mn=62.0,
    devsecond=0.77,
    yieldloss=0.05,
) -> GrasslandCuttingSpaResult:
    """Execute the source ``cutting_spa`` update for flagged grassland cuts.

    Fortran provenance: ``src_stomate/grassland_management.f90`` lines
    1872-1954 build ``flag_cutting`` for user cuts and call
    ``grassland_cutting::cutting_spa``. The cutting kernel is in
    ``src_stomate/grassland_cutting.f90`` lines 48-304. This helper starts at
    the already-audited ``flag_cutting`` boundary and applies the source
    biomass, regrowth-yield, loss, regcount, and saved ``wshtotsumprev``
    updates. It does not implement auto-fauche scheduling or animal modules.
    """

    flag_cutting = jnp.asarray(flag_cutting)
    wshtotcutinit = jnp.asarray(wshtotcutinit)
    _ = jnp.asarray(lcutinit)
    wsh = jnp.asarray(wsh)
    wshtot = jnp.asarray(wshtot)
    wr = jnp.asarray(wr)
    c = jnp.asarray(c)
    n = jnp.asarray(n)
    napo = jnp.asarray(napo)
    nsym = jnp.asarray(nsym)
    fn = jnp.asarray(fn)
    _ = jnp.asarray(nel)
    biomass = jnp.asarray(biomass)
    devstage = jnp.asarray(devstage)
    regcount = jnp.asarray(regcount)
    gmean = jnp.asarray(gmean)
    tasum = jnp.asarray(tasum)
    tgrowth = jnp.asarray(tgrowth)
    lai = jnp.asarray(lai)
    tcut = jnp.asarray(tcut)
    tcut_modif = jnp.asarray(tcut_modif)
    wshtotsum = jnp.asarray(wshtotsum)
    controle_azote_sum = jnp.asarray(controle_azote_sum)
    wshtotsumprev = jnp.asarray(wshtotsumprev)
    c_to_dm = jnp.asarray(c_to_dm, dtype=biomass.dtype)
    mc = jnp.asarray(mc, dtype=biomass.dtype)
    mn = jnp.asarray(mn, dtype=biomass.dtype)
    devsecond = jnp.asarray(devsecond, dtype=biomass.dtype)
    yieldloss = jnp.asarray(yieldloss, dtype=biomass.dtype)
    tjulian_value = jnp.asarray(tjulian, dtype=biomass.dtype)

    if biomass.ndim != 4 or biomass.shape[2] != NPARTS or biomass.shape[3] <= ICARBON:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    npts, nvm = biomass.shape[:2]
    scalar_shape = (npts, nvm)
    for name, array in (
        ("flag_cutting", flag_cutting),
        ("wsh", wsh),
        ("wshtot", wshtot),
        ("wr", wr),
        ("c", c),
        ("n", n),
        ("napo", napo),
        ("nsym", nsym),
        ("fn", fn),
        ("devstage", devstage),
        ("regcount", regcount),
        ("tasum", tasum),
        ("tgrowth", tgrowth),
        ("lai", lai),
        ("wshtotsum", wshtotsum),
        ("controle_azote_sum", controle_azote_sum),
        ("wshtotsumprev", wshtotsumprev),
    ):
        if array.shape != scalar_shape:
            raise ValueError(f"{name} must have shape (npts, nvm)")
    if wshtotcutinit.ndim != 3 or wshtotcutinit.shape[:2] != (npts, nvm):
        raise ValueError("wshtotcutinit must have shape (npts, nvm, ncut)")
    ncut = wshtotcutinit.shape[2]
    if gmean.ndim != 3 or gmean.shape[:2] != (npts, nvm):
        raise ValueError("gmean must have shape (npts, nvm, ngmean)")
    if tcut.shape != (npts, nvm, ncut) or tcut_modif.shape != (npts, nvm, ncut):
        raise ValueError("tcut and tcut_modif must have shape (npts, nvm, ncut)")

    wshcutinit = jnp.zeros((npts, nvm), dtype=biomass.dtype)
    wc_frac = jnp.zeros((npts, nvm), dtype=biomass.dtype)
    wnapo = jnp.zeros((npts, nvm), dtype=biomass.dtype)
    wnsym = jnp.zeros((npts, nvm), dtype=biomass.dtype)
    wgn = jnp.zeros((npts, nvm), dtype=biomass.dtype)
    loss = jnp.zeros((npts, nvm), dtype=biomass.dtype)
    lossc = jnp.zeros((npts, nvm), dtype=biomass.dtype)
    lossn = jnp.zeros((npts, nvm), dtype=biomass.dtype)
    tlossstart = jnp.zeros((npts, nvm), dtype=biomass.dtype)

    for pft in range(1, nvm):
        for point in range(npts):
            if not bool(flag_cutting[point, pft] == 1):
                continue
            cut_index = int(regcount[point, pft])
            if cut_index < 0 or cut_index >= ncut:
                raise ValueError("regcount must address wshtotcutinit/tcut cut axis for active cuts")
            mass_factor = 1.0 + (mc / 12.0) * c[point, pft] + (mn / 14.0) * n[point, pft]
            wlam = (biomass[point, pft, ILEAF, ICARBON] / (1000.0 * c_to_dm)) / mass_factor
            wst = (biomass[point, pft, ISAPABOVE, ICARBON] / (1000.0 * c_to_dm)) / mass_factor
            wear = (biomass[point, pft, IFRUIT, ICARBON] / (1000.0 * c_to_dm)) / mass_factor
            _ = wear
            cut_init = jnp.minimum(wsh[point, pft], wshtotcutinit[point, pft, cut_index] / mass_factor)
            wshcutinit = wshcutinit.at[point, pft].set(cut_init)
            shoot_sum = wlam + wst
            proportion_wsh = jnp.where(jnp.abs(shoot_sum) > 10.0e-15, cut_init / shoot_sum, 1.0)
            wlam = proportion_wsh * wlam
            wst = proportion_wsh * wst
            wc_frac = wc_frac.at[point, pft].set(c[point, pft] * (wr[point, pft] + cut_init))
            wnapo = wnapo.at[point, pft].set(napo[point, pft] * (wr[point, pft] + cut_init))
            wnsym = wnsym.at[point, pft].set(nsym[point, pft] * (wr[point, pft] + cut_init))
            wgn = wgn.at[point, pft].set(fn[point, pft] * (wr[point, pft] + cut_init))
            w_postecut = wlam + wst
            wlam = 0.1 * w_postecut
            wst = 0.9 * w_postecut
            devstage = devstage.at[point, pft].set(
                jnp.where((devstage[point, pft] < devsecond) & (regcount[point, pft] == 1), 0.0, 2.0)
            )
            tasum = tasum.at[point, pft].set(0.0)
            tgrowth = tgrowth.at[point, pft].set(0.0)
            loss_value = jnp.maximum(0.0, wshtot[point, pft] - wshtotcutinit[point, pft, cut_index]) * yieldloss
            loss = loss.at[point, pft].set(loss_value)
            lossc = lossc.at[point, pft].set(loss_value * c_to_dm)
            lossn = lossn.at[point, pft].set(loss_value * (n[point, pft] + fn[point, pft]))
            tlossstart = tlossstart.at[point, pft].set(tjulian_value)
            if (int(f_autogestion) == 1) or (int(f_postauto) != 0) or (int(f_autogestion) == 3) or (int(f_autogestion) == 4):
                tcut = tcut.at[point, pft, cut_index].set(tjulian_value)
                tcut_modif = tcut_modif.at[point, pft, cut_index].set(tjulian_value)
            gmean = gmean.at[point, pft, :].set(0.0)
            regcount = regcount.at[point, pft].set(regcount[point, pft] + 1)
            wshtotreg = jnp.maximum(0.0, wshtot[point, pft] - wshtotcutinit[point, pft, cut_index]) * (1.0 - yieldloss)
            wshtotsum = wshtotsum.at[point, pft].add(wshtotreg)
            biomass = biomass.at[point, pft, ILEAF, ICARBON].set((wlam * 1000.0 * c_to_dm) * mass_factor)
            biomass = biomass.at[point, pft, ISAPABOVE, ICARBON].set((wst * 1000.0 * c_to_dm) * mass_factor)
            biomass = biomass.at[point, pft, IFRUIT, ICARBON].set(0.0)

    if int(f_autogestion) < 2:
        controle_azote_sum = controle_azote_sum + wshtotsum - wshtotsumprev
    wshtotsumprev = wshtotsum

    return GrasslandCuttingSpaResult(
        biomass=biomass,
        devstage=devstage,
        regcount=regcount,
        wshcutinit=wshcutinit,
        gmean=gmean,
        wc_frac=wc_frac,
        wnapo=wnapo,
        wnsym=wnsym,
        wgn=wgn,
        tasum=tasum,
        tgrowth=tgrowth,
        loss=loss,
        lossc=lossc,
        lossn=lossn,
        tlossstart=tlossstart,
        lai=lai,
        tcut=tcut,
        tcut_modif=tcut_modif,
        wshtotsum=wshtotsum,
        controle_azote_sum=controle_azote_sum,
        wshtotsumprev=wshtotsumprev,
    )


def litter_availability_fraction_step(
    *,
    litter_above_carbon,
    litter_not_avail,
    is_tree,
    natural,
    is_grassland_manag,
    is_grassland_grazed,
    is_grassland_cut,
    do_slow=True,
):
    """Compute the aboveground litter fraction available to grazing.

    Fortran provenance: ``src_stomate/stomate_litter.f90``,
    ``littercalc_leak`` lines 1916 and 2330-2347; the older non-LEAK
    ``littercalc`` path has the same branch at lines 698-718. Fortran leaves
    bare soil/PFT index 1 at zero and loops over vegetation PFTs only.
    """

    litter_above_carbon = jnp.asarray(litter_above_carbon)
    litter_not_avail = jnp.asarray(litter_not_avail)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    natural = jnp.asarray(natural, dtype=bool)
    is_grassland_manag = jnp.asarray(is_grassland_manag, dtype=bool)
    is_grassland_grazed = jnp.asarray(is_grassland_grazed, dtype=bool)
    is_grassland_cut = jnp.asarray(is_grassland_cut, dtype=bool)

    if litter_above_carbon.ndim != 3 or litter_above_carbon.shape[1] != NLITT:
        raise ValueError("litter_above_carbon must have shape (npts, nlitt, nvm)")
    npts, _, nvm = litter_above_carbon.shape
    if litter_not_avail.shape != (npts, NLITT, nvm):
        raise ValueError("litter_not_avail must have shape (npts, nlitt, nvm)")
    for name, arr in (
        ("is_tree", is_tree),
        ("natural", natural),
        ("is_grassland_manag", is_grassland_manag),
        ("is_grassland_grazed", is_grassland_grazed),
        ("is_grassland_cut", is_grassland_cut),
    ):
        if arr.shape != (nvm,):
            raise ValueError(f"{name} must have shape (nvm,)")

    litter_avail_frac = jnp.zeros((npts, NLITT, nvm), dtype=litter_above_carbon.dtype)
    if not do_slow:
        return litter_avail_frac

    for pft in range(1, nvm):
        grazing_litter_branch = (
            (bool(is_grassland_manag[pft]) and bool(is_grassland_grazed[pft]))
            or (
                (not bool(is_tree[pft]))
                and bool(natural[pft])
                and (not bool(is_grassland_cut[pft]))
                and (not bool(is_grassland_grazed[pft]))
            )
        )
        if grazing_litter_branch:
            litter_pft = litter_above_carbon[:, :, pft]
            not_avail_pft = litter_not_avail[:, :, pft]
            frac = jnp.where(
                (litter_pft >= not_avail_pft) & (litter_pft > 0.0),
                (litter_pft - not_avail_pft) / litter_pft,
                0.0,
            )
            litter_avail_frac = litter_avail_frac.at[:, :, pft].set(frac)
        else:
            litter_avail_frac = litter_avail_frac.at[:, :, pft].set(1.0)

    return litter_avail_frac


def stomate_lpj_litter_availability_from_fraction(
    *,
    litter_above_carbon,
    litter_avail_frac,
) -> LitterAvailabilityResult:
    """Write available/unavailable aboveground litter from availability fraction.

    Fortran provenance: ``src_stomate/stomate_lpj.f90``, subroutine
    ``StomateLpj``, post-fire grazing-litter update lines 1219-1222.
    """

    litter_above_carbon = jnp.asarray(litter_above_carbon)
    litter_avail_frac = jnp.asarray(litter_avail_frac)
    if litter_above_carbon.ndim != 3 or litter_above_carbon.shape[1] != NLITT:
        raise ValueError("litter_above_carbon must have shape (npts, nlitt, nvm)")
    if litter_avail_frac.shape != litter_above_carbon.shape:
        raise ValueError("litter_avail_frac must share shape with litter_above_carbon")

    litter_avail = litter_above_carbon * litter_avail_frac
    return LitterAvailabilityResult(
        litter_avail=litter_avail,
        litter_not_avail=litter_above_carbon - litter_avail,
    )


def product_pool_aging_step(
    *,
    prod10,
    prod100,
    flux10,
    flux100,
    convflux,
    cflux_prod10=0.0,
    cflux_prod100=0.0,
    dt_days=1.0,
    one_year=ONE_YEAR_DAYS,
) -> ProductPoolAgingResult:
    """Age LCC/harvest product pools and scale released fluxes.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``gross_glcchange_fh``, product-pool aging lines 1669-1697; the same loop
    appears in ``stomate_glcchange_SinAgeC_fh.f90`` lines 834-862. This helper
    implements only the product-pool aging block, not the preceding land-cover
    transition or harvest allocation routines.
    """

    prod10 = jnp.asarray(prod10)
    prod100 = jnp.asarray(prod100)
    flux10 = jnp.asarray(flux10)
    flux100 = jnp.asarray(flux100)
    convflux = jnp.asarray(convflux)
    cflux_prod10 = jnp.asarray(cflux_prod10, dtype=prod10.dtype)
    cflux_prod100 = jnp.asarray(cflux_prod100, dtype=prod100.dtype)
    dt_days = jnp.asarray(dt_days, dtype=prod10.dtype)
    one_year = jnp.asarray(one_year, dtype=prod10.dtype)

    if prod10.ndim != 3 or prod10.shape[1] != 11:
        raise ValueError("prod10 must have shape (npts, 11, nwp), mapping Fortran 0:10")
    if prod100.ndim != 3 or prod100.shape[1] != 101:
        raise ValueError("prod100 must have shape (npts, 101, nwp), mapping Fortran 0:100")
    npts, _, nwp = prod10.shape
    if prod100.shape[0] != npts or prod100.shape[2] != nwp:
        raise ValueError("prod10 and prod100 must share npts and nwp axes")
    if flux10.shape != (npts, 10, nwp):
        raise ValueError("flux10 must have shape (npts, 10, nwp), mapping Fortran 1:10")
    if flux100.shape != (npts, 100, nwp):
        raise ValueError("flux100 must have shape (npts, 100, nwp), mapping Fortran 1:100")
    if convflux.shape != (npts, nwp):
        raise ValueError("convflux must have shape (npts, nwp)")
    if cflux_prod10.shape not in ((), (npts, nwp)):
        raise ValueError("cflux_prod10 must be scalar or have shape (npts, nwp)")
    if cflux_prod100.shape not in ((), (npts, nwp)):
        raise ValueError("cflux_prod100 must be scalar or have shape (npts, nwp)")

    new_prod10 = prod10
    aged_prod10 = prod10[:, 1:10, :] - flux10[:, :9, :]
    new_prod10 = new_prod10.at[:, 2:11, :].set(jnp.where(aged_prod10 < 1.0, 0.0, aged_prod10))
    new_prod10 = new_prod10.at[:, 1, :].set(prod10[:, 0, :])
    new_prod10 = new_prod10.at[:, 0, :].set(0.0)

    new_flux10 = flux10
    new_flux10 = new_flux10.at[:, 1:10, :].set(flux10[:, :9, :])
    new_flux10 = new_flux10.at[:, 0, :].set(0.1 * prod10[:, 0, :])

    new_prod100 = prod100
    aged_prod100 = prod100[:, 1:100, :] - flux100[:, :99, :]
    new_prod100 = new_prod100.at[:, 2:101, :].set(jnp.where(aged_prod100 < 1.0, 0.0, aged_prod100))
    new_prod100 = new_prod100.at[:, 1, :].set(prod100[:, 0, :])
    new_prod100 = new_prod100.at[:, 0, :].set(0.0)

    new_flux100 = flux100
    new_flux100 = new_flux100.at[:, 1:100, :].set(flux100[:, :99, :])
    new_flux100 = new_flux100.at[:, 0, :].set(0.01 * prod100[:, 0, :])

    scale = dt_days / one_year
    scaled_cflux_prod10 = (cflux_prod10 + jnp.sum(flux10, axis=1)) * scale
    scaled_cflux_prod100 = (cflux_prod100 + jnp.sum(flux100, axis=1)) * scale

    return ProductPoolAgingResult(
        prod10=new_prod10,
        prod100=new_prod100,
        flux10=new_flux10,
        flux100=new_flux100,
        convflux=convflux * scale,
        cflux_prod10=scaled_cflux_prod10,
        cflux_prod100=scaled_cflux_prod100,
    )


def lcchange_main_leak_step(
    *,
    dt_days,
    veget_max,
    veget_max_old,
    biomass,
    ind,
    age,
    pft_present,
    senescence,
    when_growthinit,
    everywhere,
    co2_to_bm,
    bm_to_litter,
    turnover_daily,
    bm_sapl,
    cn_ind,
    flux10,
    flux100,
    prod10,
    prod100,
    leaf_frac,
    npp_longterm,
    lm_lastyearmax,
    litter_avail,
    litter_not_avail,
    carbon,
    deepC_a,
    deepC_s,
    deepC_p,
    fuel_1hr,
    fuel_10hr,
    fuel_100hr,
    fuel_1000hr,
    litter_above,
    litter_below,
    carbon_32l,
    doc,
    cn_sapl,
    is_tree,
    coeff_lcchange_1,
    coeff_lcchange_10,
    coeff_lcchange_100,
    is_grassland_manag=None,
    is_grassland_grazed=None,
    ok_pc=False,
    min_stomate=1.0e-8,
    one_year=ONE_YEAR_DAYS,
    npp_longterm_init=10.0,
    large_value=1.0e33,
    undef=DEFAULT_UNDEF,
) -> LcchangeMainLeakResult:
    """Apply ordinary non-age-class net land-cover change with MICT leak pools.

    Fortran provenance: ``src_stomate/stomate_lcchange.f90``, subroutine
    ``lcchange_main``, lines 79-498. This is the ordinary
    ``use_age_class=.FALSE.``, ``allow_deforest_fire=.FALSE.``,
    ``agri_peat=.FALSE.`` path. When ``ok_pc`` is true, the vertically
    resolved deep-carbon branch follows lines 259-267, 356-363, and 421-424;
    otherwise the paper PFT14 MICT-leak path uses ``litter_above``,
    ``litter_below``, ``carbon_32l``, and ``DOC``.
    """

    veget_max = jnp.asarray(veget_max)
    veget_max_old = jnp.asarray(veget_max_old)
    biomass = jnp.asarray(biomass)
    ind = jnp.asarray(ind)
    age = jnp.asarray(age)
    pft_present = jnp.asarray(pft_present, dtype=bool)
    senescence = jnp.asarray(senescence, dtype=bool)
    when_growthinit = jnp.asarray(when_growthinit)
    everywhere = jnp.asarray(everywhere)
    co2_to_bm = jnp.asarray(co2_to_bm)
    bm_to_litter = jnp.asarray(bm_to_litter)
    turnover_daily = jnp.asarray(turnover_daily)
    bm_sapl = jnp.asarray(bm_sapl)
    cn_ind = jnp.asarray(cn_ind)
    flux10 = jnp.asarray(flux10)
    flux100 = jnp.asarray(flux100)
    prod10 = jnp.asarray(prod10)
    prod100 = jnp.asarray(prod100)
    leaf_frac = jnp.asarray(leaf_frac)
    npp_longterm = jnp.asarray(npp_longterm)
    lm_lastyearmax = jnp.asarray(lm_lastyearmax)
    litter_avail = jnp.asarray(litter_avail)
    litter_not_avail = jnp.asarray(litter_not_avail)
    carbon = jnp.asarray(carbon)
    deepC_a = jnp.asarray(deepC_a)
    deepC_s = jnp.asarray(deepC_s)
    deepC_p = jnp.asarray(deepC_p)
    fuel_1hr = jnp.asarray(fuel_1hr)
    fuel_10hr = jnp.asarray(fuel_10hr)
    fuel_100hr = jnp.asarray(fuel_100hr)
    fuel_1000hr = jnp.asarray(fuel_1000hr)
    litter_above = jnp.asarray(litter_above)
    litter_below = jnp.asarray(litter_below)
    carbon_32l = jnp.asarray(carbon_32l)
    doc = jnp.asarray(doc)
    cn_sapl = jnp.asarray(cn_sapl)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    coeff_lcchange_1 = jnp.asarray(coeff_lcchange_1)
    coeff_lcchange_10 = jnp.asarray(coeff_lcchange_10)
    coeff_lcchange_100 = jnp.asarray(coeff_lcchange_100)
    min_stomate = jnp.asarray(min_stomate, dtype=veget_max.dtype)
    dt_days = jnp.asarray(dt_days, dtype=veget_max.dtype)
    one_year = jnp.asarray(one_year, dtype=veget_max.dtype)
    npp_longterm_init = jnp.asarray(npp_longterm_init, dtype=veget_max.dtype)
    large_value = jnp.asarray(large_value, dtype=veget_max.dtype)
    undef = jnp.asarray(undef, dtype=veget_max.dtype)

    if veget_max.ndim != 2:
        raise ValueError("veget_max must have shape (npts, nvm)")
    npts, nvm = veget_max.shape
    if veget_max_old.shape != (npts, nvm):
        raise ValueError("veget_max_old must share veget_max shape")
    if biomass.ndim != 4 or biomass.shape[:2] != (npts, nvm) or biomass.shape[2] != NPARTS:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    nelements = biomass.shape[3]
    scalar_shape = (npts, nvm)
    for name, array in (
        ("ind", ind),
        ("age", age),
        ("pft_present", pft_present),
        ("senescence", senescence),
        ("when_growthinit", when_growthinit),
        ("everywhere", everywhere),
        ("co2_to_bm", co2_to_bm),
        ("cn_ind", cn_ind),
        ("npp_longterm", npp_longterm),
        ("lm_lastyearmax", lm_lastyearmax),
    ):
        if array.shape != scalar_shape:
            raise ValueError(f"{name} must have shape (npts, nvm)")
    if bm_to_litter.shape != biomass.shape or turnover_daily.shape != biomass.shape:
        raise ValueError("bm_to_litter and turnover_daily must share biomass shape")
    if bm_sapl.shape != (nvm, NPARTS, nelements):
        raise ValueError("bm_sapl must have shape (nvm, nparts, nelements)")
    if flux10.shape != (npts, 10) or flux100.shape != (npts, 100):
        raise ValueError("flux10/flux100 must have shapes (npts, 10)/(npts, 100)")
    if prod10.shape != (npts, 11) or prod100.shape != (npts, 101):
        raise ValueError("prod10/prod100 must have shapes (npts, 11)/(npts, 101)")
    if leaf_frac.shape != (npts, nvm, NLEAFAGES):
        raise ValueError("leaf_frac must have shape (npts, nvm, nleafages)")
    if litter_avail.shape != (npts, NLITT, nvm) or litter_not_avail.shape != (npts, NLITT, nvm):
        raise ValueError("litter_avail and litter_not_avail must have shape (npts, nlitt, nvm)")
    if carbon.shape != (npts, NCARB, nvm):
        raise ValueError("carbon must have shape (npts, ncarb, nvm)")
    if deepC_a.shape != deepC_s.shape or deepC_a.shape != deepC_p.shape or deepC_a.shape[:1] != (npts,) or deepC_a.shape[2] != nvm:
        raise ValueError("deepC_* arrays must have shape (npts, ndeep, nvm)")
    ndeep = deepC_a.shape[1]
    fuel_shape = (npts, nvm, NLITT, nelements)
    if fuel_1hr.shape != fuel_shape or fuel_10hr.shape != fuel_shape or fuel_100hr.shape != fuel_shape or fuel_1000hr.shape != fuel_shape:
        raise ValueError("fuel arrays must have shape (npts, nvm, nlitt, nelements)")
    if litter_above.shape != (npts, NLITT, nvm, nelements):
        raise ValueError("litter_above must have shape (npts, nlitt, nvm, nelements)")
    if litter_below.shape != (npts, NLITT, nvm, ndeep, nelements):
        raise ValueError("litter_below must have shape (npts, nlitt, nvm, ndeep, nelements)")
    if carbon_32l.shape != (npts, NCARB, nvm, ndeep):
        raise ValueError("carbon_32l must have shape (npts, ncarb, nvm, ndeep)")
    if doc.ndim != 6 or doc.shape[0] != npts or doc.shape[1] != nvm or doc.shape[2] != ndeep or doc.shape[5] != nelements:
        raise ValueError("doc must have shape (npts, nvm, ndeep, ndoc, npool, nelements)")
    ndoc, npool = doc.shape[3], doc.shape[4]
    for name, array in (
        ("cn_sapl", cn_sapl),
        ("is_tree", is_tree),
        ("coeff_lcchange_1", coeff_lcchange_1),
        ("coeff_lcchange_10", coeff_lcchange_10),
        ("coeff_lcchange_100", coeff_lcchange_100),
    ):
        if array.shape != (nvm,):
            raise ValueError(f"{name} must have shape (nvm,)")
    if is_grassland_manag is None:
        is_grassland_manag = jnp.zeros((nvm,), dtype=bool)
    else:
        is_grassland_manag = jnp.asarray(is_grassland_manag, dtype=bool)
    if is_grassland_grazed is None:
        is_grassland_grazed = jnp.zeros((nvm,), dtype=bool)
    else:
        is_grassland_grazed = jnp.asarray(is_grassland_grazed, dtype=bool)
    if is_grassland_manag.shape != (nvm,) or is_grassland_grazed.shape != (nvm,):
        raise ValueError("is_grassland_manag and is_grassland_grazed must have shape (nvm,)")

    prod10 = prod10.at[:, 0].set(0.0)
    prod100 = prod100.at[:, 0].set(0.0)
    convflux = jnp.zeros((npts,), dtype=veget_max.dtype)
    cflux_prod10 = jnp.zeros((npts,), dtype=veget_max.dtype)
    cflux_prod100 = jnp.zeros((npts,), dtype=veget_max.dtype)

    woody_above_parts = jnp.asarray([ISAPABOVE, IHEARTABOVE, IAGRSAPST, IAGRSAPPN, IAGRHRTST, IAGRHRTPN])
    litter_parts = (ISAPBELOW, IHEARTBELOW, IROOT, IFRUIT, ICARBRES, ILEAF)

    for point in range(npts):
        delta_veg = veget_max[point, :] - veget_max_old[point, :]
        negative = jnp.where(delta_veg < 0.0, delta_veg, 0.0)
        delta_veg_sum = jnp.sum(negative)

        dilu_lit_above = jnp.zeros((NLITT, nelements), dtype=veget_max.dtype)
        dilu_lit_below = jnp.zeros((NLITT, ndeep, nelements), dtype=veget_max.dtype)
        dilu_soil_carbon = jnp.zeros((NCARB, ndeep), dtype=veget_max.dtype)
        dilu_deep_a = jnp.zeros((ndeep,), dtype=veget_max.dtype)
        dilu_deep_s = jnp.zeros((ndeep,), dtype=veget_max.dtype)
        dilu_deep_p = jnp.zeros((ndeep,), dtype=veget_max.dtype)
        dilu_doc = jnp.zeros((ndeep, npool, ndoc), dtype=veget_max.dtype)
        biomass_loss = jnp.zeros((NPARTS, nelements), dtype=veget_max.dtype)

        for pft in range(1, nvm):
            if bool(delta_veg[pft] < -min_stomate):
                frac_loss = delta_veg[pft] / delta_veg_sum
                dilu_lit_above = dilu_lit_above + litter_above[point, :, pft, :] * frac_loss
                dilu_lit_below = dilu_lit_below + litter_below[point, :, pft, :, :] * frac_loss
                biomass_loss = biomass_loss + biomass[point, pft, :, :] * frac_loss
                if bool(ok_pc):
                    dilu_deep_a = dilu_deep_a + deepC_a[point, :, pft] * frac_loss
                    dilu_deep_s = dilu_deep_s + deepC_s[point, :, pft] * frac_loss
                    dilu_deep_p = dilu_deep_p + deepC_p[point, :, pft] * frac_loss
                else:
                    dilu_soil_carbon = dilu_soil_carbon + carbon_32l[point, :, pft, :] * frac_loss
                    for pool in range(npool):
                        for doc_idx in range(ndoc):
                            dilu_doc = dilu_doc.at[:, pool, doc_idx].add(
                                doc[point, pft, :, doc_idx, pool, ICARBON] * frac_loss
                            )

        for pft in range(1, nvm):
            old_cover = veget_max_old[point, pft]
            new_cover = veget_max[point, pft]
            delta = delta_veg[pft]
            if bool(delta > min_stomate):
                if bool(old_cover < min_stomate):
                    new_cn = jnp.where(is_tree[pft], cn_sapl[pft], 1.0)
                    cn_ind = cn_ind.at[point, pft].set(new_cn)
                    ind = ind.at[point, pft].set(delta / new_cn)
                    pft_present = pft_present.at[point, pft].set(True)
                    everywhere = everywhere.at[point, pft].set(1.0)
                    senescence = senescence.at[point, pft].set(False)
                    age = age.at[point, pft].set(0.0)
                    when_growthinit = when_growthinit.at[point, pft].set(large_value)
                    leaf_frac = leaf_frac.at[point, pft, 0].set(1.0)
                    npp_longterm = npp_longterm.at[point, pft].set(npp_longterm_init)
                    lm_lastyearmax = lm_lastyearmax.at[point, pft].set(bm_sapl[pft, ILEAF, ICARBON] * ind[point, pft])

                delta_ind = jnp.where(cn_ind[point, pft] > min_stomate, delta / cn_ind[point, pft], 0.0)
                for part in range(NPARTS):
                    for element in range(nelements):
                        bm_new = delta_ind * bm_sapl[pft, part, element]
                        if bool(old_cover > min_stomate):
                            over = (bm_new / delta) > biomass[point, pft, part, element]
                            bm_new = jnp.where(over, biomass[point, pft, part, element] * delta, bm_new)
                        biomass = biomass.at[point, pft, part, element].set(
                            (biomass[point, pft, part, element] * old_cover + bm_new) / new_cover
                        )
                        if element == ICARBON:
                            co2_to_bm = co2_to_bm.at[point, pft].add(bm_new * dt_days / (one_year * new_cover))

                litter_above = litter_above.at[point, :, pft, :].set(
                    (litter_above[point, :, pft, :] * old_cover + dilu_lit_above * delta) / new_cover
                )
                litter_below = litter_below.at[point, :, pft, :, :].set(
                    (litter_below[point, :, pft, :, :] * old_cover + dilu_lit_below * delta) / new_cover
                )
                if bool(is_grassland_manag[pft]) and bool(is_grassland_grazed[pft]):
                    litter_avail = litter_avail.at[point, :, pft].set(litter_avail[point, :, pft] * old_cover / new_cover)
                    litter_not_avail = litter_not_avail.at[point, :, pft].set(
                        litter_above[point, :, pft, ICARBON] - litter_avail[point, :, pft]
                    )

                if bool(ok_pc):
                    deepC_a = deepC_a.at[point, :, pft].set(
                        (deepC_a[point, :, pft] * old_cover + dilu_deep_a * delta) / new_cover
                    )
                    deepC_s = deepC_s.at[point, :, pft].set(
                        (deepC_s[point, :, pft] * old_cover + dilu_deep_s * delta) / new_cover
                    )
                    deepC_p = deepC_p.at[point, :, pft].set(
                        (deepC_p[point, :, pft] * old_cover + dilu_deep_p * delta) / new_cover
                    )
                else:
                    carbon_32l = carbon_32l.at[point, :, pft, :].set(
                        (carbon_32l[point, :, pft, :] * old_cover + dilu_soil_carbon * delta) / new_cover
                    )
                    for pool in range(npool):
                        for doc_idx in range(ndoc):
                            doc = doc.at[point, pft, :, doc_idx, pool, ICARBON].set(
                                (doc[point, pft, :, doc_idx, pool, ICARBON] * new_cover + dilu_doc[:, pool, doc_idx] * delta)
                                / new_cover
                            )

                for element in range(nelements):
                    for part in litter_parts:
                        bm_to_litter = bm_to_litter.at[point, pft, part, element].set(
                            (bm_to_litter[point, pft, part, element] * old_cover + biomass_loss[part, element] * delta)
                            / new_cover
                        )
                age = age.at[point, pft].set(age[point, pft] * old_cover / new_cover)
            else:
                above = jnp.sum(biomass[point, pft, woody_above_parts, ICARBON])
                convflux = convflux.at[point].add(-(coeff_lcchange_1[pft] * above * delta))
                prod10 = prod10.at[point, 0].add(-(coeff_lcchange_10[pft] * above * delta))
                prod100 = prod100.at[point, 0].add(-(coeff_lcchange_100[pft] * above * delta))

                if bool(new_cover < min_stomate):
                    ind = ind.at[point, pft].set(0.0)
                    biomass = biomass.at[point, pft, :, :].set(0.0)
                    pft_present = pft_present.at[point, pft].set(False)
                    senescence = senescence.at[point, pft].set(False)
                    age = age.at[point, pft].set(0.0)
                    when_growthinit = when_growthinit.at[point, pft].set(undef)
                    everywhere = everywhere.at[point, pft].set(0.0)
                    carbon = carbon.at[point, :, pft].set(0.0)
                    bm_to_litter = bm_to_litter.at[point, pft, :, :].set(0.0)
                    turnover_daily = turnover_daily.at[point, pft, :, :].set(0.0)
                    if bool(ok_pc):
                        deepC_a = deepC_a.at[point, :, pft].set(0.0)
                        deepC_s = deepC_s.at[point, :, pft].set(0.0)
                        deepC_p = deepC_p.at[point, :, pft].set(0.0)

    aged = product_pool_aging_step(
        prod10=prod10[:, :, None],
        prod100=prod100[:, :, None],
        flux10=flux10[:, :, None],
        flux100=flux100[:, :, None],
        convflux=convflux[:, None],
        cflux_prod10=cflux_prod10[:, None],
        cflux_prod100=cflux_prod100[:, None],
        dt_days=dt_days,
        one_year=one_year,
    )

    fuel_all = fuel_1hr + fuel_10hr + fuel_100hr + fuel_1000hr
    fuel_1hr_frac = jnp.where(fuel_all > min_stomate, fuel_1hr / fuel_all, 0.25)
    fuel_10hr_frac = jnp.where(fuel_all > min_stomate, fuel_10hr / fuel_all, 0.25)
    fuel_100hr_frac = jnp.where(fuel_all > min_stomate, fuel_100hr / fuel_all, 0.25)
    fuel_1000hr_frac = jnp.where(fuel_all > min_stomate, fuel_1000hr / fuel_all, 0.25)
    above_litter_by_pft = jnp.swapaxes(litter_above, 1, 2)
    fuel_1hr = above_litter_by_pft * fuel_1hr_frac
    fuel_10hr = above_litter_by_pft * fuel_10hr_frac
    fuel_100hr = above_litter_by_pft * fuel_100hr_frac
    fuel_1000hr = above_litter_by_pft * fuel_1000hr_frac

    return LcchangeMainLeakResult(
        veget_max=veget_max,
        biomass=biomass,
        ind=ind,
        age=age,
        pft_present=pft_present,
        senescence=senescence,
        when_growthinit=when_growthinit,
        everywhere=everywhere,
        co2_to_bm=co2_to_bm,
        bm_to_litter=bm_to_litter,
        turnover_daily=turnover_daily,
        cn_ind=cn_ind,
        prod10=aged.prod10[:, :, 0],
        prod100=aged.prod100[:, :, 0],
        flux10=aged.flux10[:, :, 0],
        flux100=aged.flux100[:, :, 0],
        convflux=aged.convflux[:, 0],
        cflux_prod10=aged.cflux_prod10[:, 0],
        cflux_prod100=aged.cflux_prod100[:, 0],
        leaf_frac=leaf_frac,
        npp_longterm=npp_longterm,
        lm_lastyearmax=lm_lastyearmax,
        litter_avail=litter_avail,
        litter_not_avail=litter_not_avail,
        carbon=carbon,
        deepC_a=deepC_a,
        deepC_s=deepC_s,
        deepC_p=deepC_p,
        fuel_1hr=fuel_1hr,
        fuel_10hr=fuel_10hr,
        fuel_100hr=fuel_100hr,
        fuel_1000hr=fuel_1000hr,
        litter_above=litter_above,
        litter_below=litter_below,
        carbon_32l=carbon_32l,
        doc=doc,
    )


def lcchange_deffire_step(
    *,
    dt_days,
    veget_max,
    veget_max_new,
    biomass,
    ind,
    age,
    pft_present,
    senescence,
    when_growthinit,
    everywhere,
    co2_to_bm,
    bm_to_litter,
    turnover_daily,
    bm_sapl,
    cn_ind,
    flux10,
    flux100,
    prod10,
    prod100,
    leaf_frac,
    npp_longterm,
    lm_lastyearmax,
    litter,
    litter_avail,
    litter_not_avail,
    carbon,
    deepC_a,
    deepC_s,
    deepC_p,
    fuel_1hr,
    fuel_10hr,
    fuel_100hr,
    fuel_1000hr,
    lcc,
    bafrac_deforest_accu,
    emideforest_litter_accu,
    emideforest_biomass_accu,
    deflitsup_total,
    defbiosup_total,
    cn_sapl,
    is_tree,
    is_grassland_manag=None,
    is_grassland_grazed=None,
    ok_pc=False,
    min_stomate=1.0e-8,
    one_year=ONE_YEAR_DAYS,
    npp_longterm_init=10.0,
    large_value=1.0e33,
    undef=DEFAULT_UNDEF,
) -> LcchangeDeffireResult:
    """Apply non-age-class net LCC with deforestation-fire emissions.

    Fortran provenance: ``src_stomate/stomate_lcchange.f90``, subroutine
    ``lcchange_deffire``, lines 502-1059. Fire emissions are subtracted from
    deforested tree litter/biomass before dilution, remaining above wood enters
    product pools, product pools age, fuel pools are rebalanced from
    aboveground litter, and ``veget_max`` is assigned to ``veget_max_new``.
    The ``OK_PC=.TRUE.`` branch follows lines 744-750 and 834-840 by
    diluting vertically resolved ``deepC_a/s/p`` instead of legacy
    ``carbon``.
    """

    veget_max = jnp.asarray(veget_max)
    veget_max_new = jnp.asarray(veget_max_new)
    biomass = jnp.asarray(biomass)
    ind = jnp.asarray(ind)
    age = jnp.asarray(age)
    pft_present = jnp.asarray(pft_present, dtype=bool)
    senescence = jnp.asarray(senescence, dtype=bool)
    when_growthinit = jnp.asarray(when_growthinit)
    everywhere = jnp.asarray(everywhere)
    co2_to_bm = jnp.asarray(co2_to_bm)
    bm_to_litter = jnp.asarray(bm_to_litter)
    turnover_daily = jnp.asarray(turnover_daily)
    bm_sapl = jnp.asarray(bm_sapl)
    cn_ind = jnp.asarray(cn_ind)
    flux10 = jnp.asarray(flux10)
    flux100 = jnp.asarray(flux100)
    prod10 = jnp.asarray(prod10)
    prod100 = jnp.asarray(prod100)
    leaf_frac = jnp.asarray(leaf_frac)
    npp_longterm = jnp.asarray(npp_longterm)
    lm_lastyearmax = jnp.asarray(lm_lastyearmax)
    litter = jnp.asarray(litter)
    litter_avail = jnp.asarray(litter_avail)
    litter_not_avail = jnp.asarray(litter_not_avail)
    carbon = jnp.asarray(carbon)
    deepC_a = jnp.asarray(deepC_a)
    deepC_s = jnp.asarray(deepC_s)
    deepC_p = jnp.asarray(deepC_p)
    fuel_1hr = jnp.asarray(fuel_1hr)
    fuel_10hr = jnp.asarray(fuel_10hr)
    fuel_100hr = jnp.asarray(fuel_100hr)
    fuel_1000hr = jnp.asarray(fuel_1000hr)
    lcc = jnp.asarray(lcc)
    bafrac_deforest_accu = jnp.asarray(bafrac_deforest_accu)
    emideforest_litter_accu = jnp.asarray(emideforest_litter_accu)
    emideforest_biomass_accu = jnp.asarray(emideforest_biomass_accu)
    deflitsup_total = jnp.asarray(deflitsup_total)
    defbiosup_total = jnp.asarray(defbiosup_total)
    cn_sapl = jnp.asarray(cn_sapl)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    min_stomate = jnp.asarray(min_stomate, dtype=veget_max.dtype)
    dt_days = jnp.asarray(dt_days, dtype=veget_max.dtype)
    one_year = jnp.asarray(one_year, dtype=veget_max.dtype)
    npp_longterm_init = jnp.asarray(npp_longterm_init, dtype=veget_max.dtype)
    large_value = jnp.asarray(large_value, dtype=veget_max.dtype)
    undef = jnp.asarray(undef, dtype=veget_max.dtype)

    if veget_max.ndim != 2:
        raise ValueError("veget_max must have shape (npts, nvm)")
    npts, nvm = veget_max.shape
    if veget_max_new.shape != (npts, nvm):
        raise ValueError("veget_max_new must share veget_max shape")
    if biomass.ndim != 4 or biomass.shape[:2] != (npts, nvm) or biomass.shape[2] != NPARTS:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    nelements = biomass.shape[3]
    if bm_to_litter.shape != biomass.shape or turnover_daily.shape != biomass.shape:
        raise ValueError("bm_to_litter and turnover_daily must share biomass shape")
    if bm_sapl.shape != (nvm, NPARTS, nelements):
        raise ValueError("bm_sapl must have shape (nvm, nparts, nelements)")
    scalar_shape = (npts, nvm)
    for name, array in (
        ("ind", ind),
        ("age", age),
        ("pft_present", pft_present),
        ("senescence", senescence),
        ("when_growthinit", when_growthinit),
        ("everywhere", everywhere),
        ("co2_to_bm", co2_to_bm),
        ("cn_ind", cn_ind),
        ("npp_longterm", npp_longterm),
        ("lm_lastyearmax", lm_lastyearmax),
        ("lcc", lcc),
        ("bafrac_deforest_accu", bafrac_deforest_accu),
        ("deflitsup_total", deflitsup_total),
        ("defbiosup_total", defbiosup_total),
    ):
        if array.shape != scalar_shape:
            raise ValueError(f"{name} must have shape (npts, nvm)")
    if flux10.shape != (npts, 10) or flux100.shape != (npts, 100):
        raise ValueError("flux10/flux100 must have shapes (npts, 10)/(npts, 100)")
    if prod10.shape != (npts, 11) or prod100.shape != (npts, 101):
        raise ValueError("prod10/prod100 must have shapes (npts, 11)/(npts, 101)")
    if leaf_frac.shape != (npts, nvm, NLEAFAGES):
        raise ValueError("leaf_frac must have shape (npts, nvm, nleafages)")
    if litter.shape != (npts, NLITT, nvm, NLEVS, nelements):
        raise ValueError("litter must have shape (npts, nlitt, nvm, nlevs, nelements)")
    if litter_avail.shape != (npts, NLITT, nvm) or litter_not_avail.shape != (npts, NLITT, nvm):
        raise ValueError("litter_avail and litter_not_avail must have shape (npts, nlitt, nvm)")
    if carbon.shape != (npts, NCARB, nvm):
        raise ValueError("carbon must have shape (npts, ncarb, nvm)")
    if deepC_a.shape != deepC_s.shape or deepC_a.shape != deepC_p.shape or deepC_a.shape[:1] != (npts,) or deepC_a.shape[2] != nvm:
        raise ValueError("deepC_* arrays must have shape (npts, ndeep, nvm)")
    ndeep = deepC_a.shape[1]
    fuel_shape = (npts, nvm, NLITT, nelements)
    if fuel_1hr.shape != fuel_shape or fuel_10hr.shape != fuel_shape or fuel_100hr.shape != fuel_shape or fuel_1000hr.shape != fuel_shape:
        raise ValueError("fuel arrays must have shape (npts, nvm, nlitt, nelements)")
    if emideforest_litter_accu.shape != (npts, nvm, NLITT, nelements):
        raise ValueError("emideforest_litter_accu must have shape (npts, nvm, nlitt, nelements)")
    if emideforest_biomass_accu.shape != biomass.shape:
        raise ValueError("emideforest_biomass_accu must share biomass shape")
    if cn_sapl.shape != (nvm,) or is_tree.shape != (nvm,):
        raise ValueError("cn_sapl and is_tree must have shape (nvm,)")
    if is_grassland_manag is None:
        is_grassland_manag = jnp.zeros((nvm,), dtype=bool)
    else:
        is_grassland_manag = jnp.asarray(is_grassland_manag, dtype=bool)
    if is_grassland_grazed is None:
        is_grassland_grazed = jnp.zeros((nvm,), dtype=bool)
    else:
        is_grassland_grazed = jnp.asarray(is_grassland_grazed, dtype=bool)
    if is_grassland_manag.shape != (nvm,) or is_grassland_grazed.shape != (nvm,):
        raise ValueError("is_grassland_manag and is_grassland_grazed must have shape (nvm,)")

    prod10 = prod10.at[:, 0].set(0.0)
    prod100 = prod100.at[:, 0].set(0.0)
    convflux = jnp.zeros((npts,), dtype=veget_max.dtype)
    cflux_prod10 = jnp.zeros((npts,), dtype=veget_max.dtype)
    cflux_prod100 = jnp.zeros((npts,), dtype=veget_max.dtype)
    deforest_litter_surplus = jnp.zeros((npts, nvm, NLITT), dtype=veget_max.dtype)
    deforest_biomass_surplus = jnp.zeros((npts, nvm, NPARTS), dtype=veget_max.dtype)
    deforest_litter_deficit = jnp.zeros((npts, nvm, NLITT), dtype=veget_max.dtype)
    deforest_biomass_deficit = jnp.zeros((npts, nvm, NPARTS), dtype=veget_max.dtype)
    litter_parts = (ISAPBELOW, IHEARTBELOW, IROOT, IFRUIT, ICARBRES, ILEAF)
    woody_above_parts = jnp.asarray([ISAPABOVE, IHEARTABOVE, IAGRSAPST, IAGRSAPPN, IAGRHRTST, IAGRHRTPN])

    for point in range(npts):
        delta_veg = veget_max_new[point, :] - veget_max[point, :]
        delta_veg_sum = jnp.sum(jnp.where(delta_veg < 0.0, delta_veg, 0.0))
        dilu_lit = jnp.zeros((NLITT, NLEVS, nelements), dtype=veget_max.dtype)
        dilu_soil_carbon = jnp.zeros((NCARB,), dtype=veget_max.dtype)
        dilu_deep_a = jnp.zeros((ndeep,), dtype=veget_max.dtype)
        dilu_deep_s = jnp.zeros((ndeep,), dtype=veget_max.dtype)
        dilu_deep_p = jnp.zeros((ndeep,), dtype=veget_max.dtype)
        biomass_loss = jnp.zeros((NPARTS, nelements), dtype=veget_max.dtype)

        for pft in range(1, nvm):
            if bool(delta_veg[pft] < -min_stomate):
                if bool(is_tree[pft]):
                    lit_surplus = -delta_veg[pft] * litter[point, :, pft, IABOVE, ICARBON] - emideforest_litter_accu[point, pft, :, ICARBON]
                    deforest_litter_surplus = deforest_litter_surplus.at[point, pft, :].set(lit_surplus)
                    for ilit in range(NLITT):
                        if bool(lit_surplus[ilit] < 0.0):
                            not_enough_cover = bool(veget_max_new[point, pft] < min_stomate)
                            not_enough_pool = bool(litter[point, ilit, pft, IABOVE, ICARBON] * veget_max_new[point, pft] < -lit_surplus[ilit])
                            if not_enough_cover or not_enough_pool:
                                deforest_litter_deficit = deforest_litter_deficit.at[point, pft, ilit].set(lit_surplus[ilit])
                            else:
                                litter = litter.at[point, ilit, pft, IABOVE, ICARBON].set(
                                    (litter[point, ilit, pft, IABOVE, ICARBON] * veget_max_new[point, pft] + lit_surplus[ilit])
                                    / veget_max_new[point, pft]
                                )
                        else:
                            dilu_lit = dilu_lit.at[ilit, IABOVE, ICARBON].add(-lit_surplus[ilit])
                    dilu_lit = dilu_lit.at[:, IBELOW, :].add(delta_veg[pft] * litter[point, :, pft, IBELOW, :])

                    bio_surplus = -delta_veg[pft] * biomass[point, pft, :, ICARBON] - emideforest_biomass_accu[point, pft, :, ICARBON]
                    deforest_biomass_surplus = deforest_biomass_surplus.at[point, pft, :].set(bio_surplus)
                    for part in range(NPARTS):
                        if bool(bio_surplus[part] < 0.0):
                            not_enough_cover = bool(veget_max_new[point, pft] < min_stomate)
                            not_enough_pool = bool(biomass[point, pft, part, ICARBON] * veget_max_new[point, pft] < -bio_surplus[part])
                            if not_enough_cover or not_enough_pool:
                                deforest_biomass_deficit = deforest_biomass_deficit.at[point, pft, part].set(bio_surplus[part])
                            else:
                                biomass = biomass.at[point, pft, part, ICARBON].set(
                                    (biomass[point, pft, part, ICARBON] * veget_max_new[point, pft] + bio_surplus[part])
                                    / veget_max_new[point, pft]
                                )
                        else:
                            biomass_loss = biomass_loss.at[part, ICARBON].add(-bio_surplus[part])
                else:
                    dilu_lit = dilu_lit + delta_veg[pft] * litter[point, :, pft, :, :]
                    biomass_loss = biomass_loss + biomass[point, pft, :, :] * delta_veg[pft]

                if bool(ok_pc):
                    dilu_deep_a = dilu_deep_a + delta_veg[pft] * deepC_a[point, :, pft] / delta_veg_sum
                    dilu_deep_s = dilu_deep_s + delta_veg[pft] * deepC_s[point, :, pft] / delta_veg_sum
                    dilu_deep_p = dilu_deep_p + delta_veg[pft] * deepC_p[point, :, pft] / delta_veg_sum
                else:
                    dilu_soil_carbon = dilu_soil_carbon + delta_veg[pft] * carbon[point, :, pft] / delta_veg_sum

        if bool(delta_veg_sum < -min_stomate):
            biomass_loss = biomass_loss / delta_veg_sum
            dilu_lit = dilu_lit / delta_veg_sum

        for pft in range(1, nvm):
            old_cover = veget_max[point, pft]
            new_cover = veget_max_new[point, pft]
            delta = delta_veg[pft]
            if bool(delta > min_stomate):
                if bool(old_cover < min_stomate):
                    new_cn = jnp.where(is_tree[pft], cn_sapl[pft], 1.0)
                    cn_ind = cn_ind.at[point, pft].set(new_cn)
                    ind = ind.at[point, pft].set(delta / new_cn)
                    pft_present = pft_present.at[point, pft].set(True)
                    everywhere = everywhere.at[point, pft].set(1.0)
                    senescence = senescence.at[point, pft].set(False)
                    age = age.at[point, pft].set(0.0)
                    when_growthinit = when_growthinit.at[point, pft].set(large_value)
                    leaf_frac = leaf_frac.at[point, pft, 0].set(1.0)
                    npp_longterm = npp_longterm.at[point, pft].set(npp_longterm_init)
                    lm_lastyearmax = lm_lastyearmax.at[point, pft].set(bm_sapl[pft, ILEAF, ICARBON] * ind[point, pft])

                delta_ind = jnp.where(cn_ind[point, pft] > min_stomate, delta / cn_ind[point, pft], 0.0)
                for part in range(NPARTS):
                    for element in range(nelements):
                        bm_new = delta_ind * bm_sapl[pft, part, element]
                        if bool(old_cover > min_stomate):
                            over = (bm_new / delta) > biomass[point, pft, part, element]
                            bm_new = jnp.where(over, biomass[point, pft, part, element] * delta, bm_new)
                        biomass = biomass.at[point, pft, part, element].set(
                            (biomass[point, pft, part, element] * old_cover + bm_new) / new_cover
                        )
                        if element == ICARBON:
                            co2_to_bm = co2_to_bm.at[point, pft].add(bm_new * dt_days / (one_year * new_cover))

                litter = litter.at[point, :, pft, :, :].set(
                    (litter[point, :, pft, :, :] * old_cover + dilu_lit * delta) / new_cover
                )
                if bool(is_grassland_manag[pft]) and bool(is_grassland_grazed[pft]):
                    litter_avail = litter_avail.at[point, :, pft].set(litter_avail[point, :, pft] * old_cover / new_cover)
                    litter_not_avail = litter_not_avail.at[point, :, pft].set(
                        litter[point, :, pft, IABOVE, ICARBON] - litter_avail[point, :, pft]
                    )
                if bool(ok_pc):
                    deepC_a = deepC_a.at[point, :, pft].set(
                        (deepC_a[point, :, pft] * old_cover + dilu_deep_a * delta) / new_cover
                    )
                    deepC_s = deepC_s.at[point, :, pft].set(
                        (deepC_s[point, :, pft] * old_cover + dilu_deep_s * delta) / new_cover
                    )
                    deepC_p = deepC_p.at[point, :, pft].set(
                        (deepC_p[point, :, pft] * old_cover + dilu_deep_p * delta) / new_cover
                    )
                else:
                    carbon = carbon.at[point, :, pft].set(
                        (carbon[point, :, pft] * old_cover + dilu_soil_carbon * delta) / new_cover
                    )
                for element in range(nelements):
                    for part in litter_parts:
                        bm_to_litter = bm_to_litter.at[point, pft, part, element].add(
                            biomass_loss[part, element] * delta / new_cover
                        )
                age = age.at[point, pft].set(age[point, pft] * old_cover / new_cover)
            else:
                if bool(new_cover < min_stomate):
                    veget_max_new = veget_max_new.at[point, pft].set(0.0)
                    ind = ind.at[point, pft].set(0.0)
                    biomass = biomass.at[point, pft, :, :].set(0.0)
                    pft_present = pft_present.at[point, pft].set(False)
                    senescence = senescence.at[point, pft].set(False)
                    age = age.at[point, pft].set(0.0)
                    when_growthinit = when_growthinit.at[point, pft].set(undef)
                    everywhere = everywhere.at[point, pft].set(0.0)
                    carbon = carbon.at[point, :, pft].set(0.0)
                    litter = litter.at[point, :, pft, :, :].set(0.0)
                    litter_avail = litter_avail.at[point, :, pft].set(0.0)
                    litter_not_avail = litter_not_avail.at[point, :, pft].set(0.0)
                    bm_to_litter = bm_to_litter.at[point, pft, :, :].set(0.0)
                    turnover_daily = turnover_daily.at[point, pft, :, :].set(0.0)
                    deepC_a = deepC_a.at[point, :, pft].set(0.0)
                    deepC_s = deepC_s.at[point, :, pft].set(0.0)
                    deepC_p = deepC_p.at[point, :, pft].set(0.0)

        above = jnp.sum(biomass_loss[woody_above_parts, ICARBON]) * delta_veg_sum * -1.0
        convflux = convflux.at[point].set(
            jnp.sum(emideforest_biomass_accu[point, :, ISAPABOVE, ICARBON] + emideforest_biomass_accu[point, :, IHEARTABOVE, ICARBON])
        )
        prod10 = prod10.at[point, 0].set(0.4 * above)
        prod100 = prod100.at[point, 0].set(0.6 * above)

    aged = product_pool_aging_step(
        prod10=prod10[:, :, None],
        prod100=prod100[:, :, None],
        flux10=flux10[:, :, None],
        flux100=flux100[:, :, None],
        convflux=convflux[:, None],
        cflux_prod10=cflux_prod10[:, None],
        cflux_prod100=cflux_prod100[:, None],
        dt_days=dt_days,
        one_year=one_year,
    )

    fuel_all = fuel_1hr + fuel_10hr + fuel_100hr + fuel_1000hr
    fuel_1hr_frac = jnp.where(fuel_all > min_stomate, fuel_1hr / fuel_all, 0.25)
    fuel_10hr_frac = jnp.where(fuel_all > min_stomate, fuel_10hr / fuel_all, 0.25)
    fuel_100hr_frac = jnp.where(fuel_all > min_stomate, fuel_100hr / fuel_all, 0.25)
    fuel_1000hr_frac = jnp.where(fuel_all > min_stomate, fuel_1000hr / fuel_all, 0.25)
    above_litter_by_pft = jnp.swapaxes(litter[:, :, :, IABOVE, :], 1, 2)
    fuel_1hr = above_litter_by_pft * fuel_1hr_frac
    fuel_10hr = above_litter_by_pft * fuel_10hr_frac
    fuel_100hr = above_litter_by_pft * fuel_100hr_frac
    fuel_1000hr = above_litter_by_pft * fuel_1000hr_frac

    veget_max = veget_max_new
    deflitsup_total = jnp.sum(deforest_litter_surplus, axis=2)
    defbiosup_total = jnp.sum(deforest_biomass_surplus, axis=2)

    return LcchangeDeffireResult(
        veget_max=veget_max,
        veget_max_new=veget_max_new,
        biomass=biomass,
        ind=ind,
        age=age,
        pft_present=pft_present,
        senescence=senescence,
        when_growthinit=when_growthinit,
        everywhere=everywhere,
        co2_to_bm=co2_to_bm,
        bm_to_litter=bm_to_litter,
        turnover_daily=turnover_daily,
        cn_ind=cn_ind,
        prod10=aged.prod10[:, :, 0],
        prod100=aged.prod100[:, :, 0],
        flux10=aged.flux10[:, :, 0],
        flux100=aged.flux100[:, :, 0],
        convflux=aged.convflux[:, 0],
        cflux_prod10=aged.cflux_prod10[:, 0],
        cflux_prod100=aged.cflux_prod100[:, 0],
        leaf_frac=leaf_frac,
        npp_longterm=npp_longterm,
        lm_lastyearmax=lm_lastyearmax,
        litter=litter,
        litter_avail=litter_avail,
        litter_not_avail=litter_not_avail,
        carbon=carbon,
        deepC_a=deepC_a,
        deepC_s=deepC_s,
        deepC_p=deepC_p,
        fuel_1hr=fuel_1hr,
        fuel_10hr=fuel_10hr,
        fuel_100hr=fuel_100hr,
        fuel_1000hr=fuel_1000hr,
        deflitsup_total=deflitsup_total,
        defbiosup_total=defbiosup_total,
    )


def forest_harvest_legacy_step(
    *,
    biomass,
    bm_to_litter_pro,
    convflux,
    prod10,
    prod100,
    point,
    pft,
    frac,
    coeff_lcchange_1,
    coeff_lcchange_10,
    coeff_lcchange_100,
    allow_deforest_fire=False,
    deforest_biomass_remain=None,
) -> ForestHarvestLegacyResult:
    """Apply the forest-clearing product-pool and biomass-to-litter boundary.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``harvest_forest``, lines 944-980; the same branch appears in
    ``stomate_glcchange_SinAgeC_fh.f90`` lines 109-135. This helper implements
    only the legacy product-pool inputs and direct biomass-to-litter transfer,
    not the following litter/fuel/lignin proxy updates or full LCC dispatcher.
    """

    biomass = jnp.asarray(biomass)
    bm_to_litter_pro = jnp.asarray(bm_to_litter_pro)
    convflux = jnp.asarray(convflux)
    prod10 = jnp.asarray(prod10)
    prod100 = jnp.asarray(prod100)
    coeff_lcchange_1 = jnp.asarray(coeff_lcchange_1)
    coeff_lcchange_10 = jnp.asarray(coeff_lcchange_10)
    coeff_lcchange_100 = jnp.asarray(coeff_lcchange_100)
    frac = jnp.asarray(frac, dtype=biomass.dtype)
    point = int(point)
    pft = int(pft)

    if biomass.ndim != 4 or biomass.shape[2] < NPARTS:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    npts, nvm, _, nelements = biomass.shape
    if not (0 <= point < npts) or not (0 <= pft < nvm):
        raise ValueError("point and pft must address biomass axes")
    if bm_to_litter_pro.shape != (NPARTS, nelements):
        raise ValueError("bm_to_litter_pro must have shape (nparts, nelements)")
    if convflux.shape != (npts,):
        raise ValueError("convflux must have shape (npts,)")
    if prod10.shape != (npts, 11):
        raise ValueError("prod10 must have shape (npts, 11), mapping Fortran 0:10")
    if prod100.shape != (npts, 101):
        raise ValueError("prod100 must have shape (npts, 101), mapping Fortran 0:100")
    if coeff_lcchange_1.shape != (nvm,) or coeff_lcchange_10.shape != (nvm,) or coeff_lcchange_100.shape != (nvm,):
        raise ValueError("coeff_lcchange_* arrays must have shape (nvm,)")

    woody_above_parts = jnp.asarray(
        [ISAPABOVE, IHEARTABOVE, IAGRSAPST, IAGRSAPPN, IAGRHRTST, IAGRHRTPN]
    )
    litter_parts = jnp.asarray([ISAPBELOW, IHEARTBELOW, IROOT, IFRUIT, ICARBRES, ILEAF])

    if allow_deforest_fire:
        if deforest_biomass_remain is None:
            raise ValueError("deforest_biomass_remain is required when allow_deforest_fire is true")
        deforest_biomass_remain = jnp.asarray(deforest_biomass_remain)
        if deforest_biomass_remain.shape != biomass.shape:
            raise ValueError("deforest_biomass_remain must share biomass shape")
        above = jnp.sum(deforest_biomass_remain[point, pft, woody_above_parts, ICARBON])
        new_convflux = convflux
        new_prod10 = prod10.at[point, 0].add(0.4 * above)
        new_prod100 = prod100.at[point, 0].add(0.6 * above)
    else:
        above = jnp.sum(biomass[point, pft, woody_above_parts, ICARBON]) * frac
        new_convflux = convflux.at[point].add(coeff_lcchange_1[pft] * above)
        new_prod10 = prod10.at[point, 0].add(coeff_lcchange_10[pft] * above)
        new_prod100 = prod100.at[point, 0].add(coeff_lcchange_100[pft] * above)

    litter_increment = biomass[point, pft, litter_parts, :] * frac
    new_bm_to_litter_pro = bm_to_litter_pro.at[litter_parts, :].add(litter_increment)

    return ForestHarvestLegacyResult(
        bm_to_litter_pro=new_bm_to_litter_pro,
        convflux=new_convflux,
        prod10=new_prod10,
        prod100=new_prod100,
        above=above,
    )


def forest_harvest_proxy_pools_step(
    *,
    litter,
    fuel_1hr,
    fuel_10hr,
    fuel_100hr,
    fuel_1000hr,
    lignin_struc,
    litter_pro,
    fuel_1hr_pro,
    fuel_10hr_pro,
    fuel_100hr_pro,
    fuel_1000hr_pro,
    lignin_content_pro,
    point,
    pft,
    frac,
) -> ForestHarvestProxyPoolsResult:
    """Transfer cleared forest litter, fuel, and lignin to proxy pools.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``harvest_forest``, lines 982-990; same branch in
    ``stomate_glcchange_SinAgeC_fh.f90`` lines 147-155.
    """

    litter = jnp.asarray(litter)
    fuel_1hr = jnp.asarray(fuel_1hr)
    fuel_10hr = jnp.asarray(fuel_10hr)
    fuel_100hr = jnp.asarray(fuel_100hr)
    fuel_1000hr = jnp.asarray(fuel_1000hr)
    lignin_struc = jnp.asarray(lignin_struc)
    litter_pro = jnp.asarray(litter_pro)
    fuel_1hr_pro = jnp.asarray(fuel_1hr_pro)
    fuel_10hr_pro = jnp.asarray(fuel_10hr_pro)
    fuel_100hr_pro = jnp.asarray(fuel_100hr_pro)
    fuel_1000hr_pro = jnp.asarray(fuel_1000hr_pro)
    lignin_content_pro = jnp.asarray(lignin_content_pro)
    frac = jnp.asarray(frac, dtype=litter.dtype)
    point = int(point)
    pft = int(pft)

    if litter.ndim != 5 or litter.shape[1] != NLITT:
        raise ValueError("litter must have shape (npts, nlitt, nvm, nlevels, nelements)")
    npts, _, nvm, nlevels, nelements = litter.shape
    if not (0 <= point < npts) or not (0 <= pft < nvm):
        raise ValueError("point and pft must address litter axes")
    fuel_shape = (npts, nvm, NLITT, nelements)
    if fuel_1hr.shape != fuel_shape or fuel_10hr.shape != fuel_shape or fuel_100hr.shape != fuel_shape or fuel_1000hr.shape != fuel_shape:
        raise ValueError("fuel arrays must have shape (npts, nvm, nlitt, nelements)")
    if lignin_struc.shape != (npts, nvm, nlevels):
        raise ValueError("lignin_struc must have shape (npts, nvm, nlevels)")
    if litter_pro.shape != (NLITT, nlevels, nelements):
        raise ValueError("litter_pro must have shape (nlitt, nlevels, nelements)")
    fuel_pro_shape = (NLITT, nelements)
    if fuel_1hr_pro.shape != fuel_pro_shape or fuel_10hr_pro.shape != fuel_pro_shape or fuel_100hr_pro.shape != fuel_pro_shape or fuel_1000hr_pro.shape != fuel_pro_shape:
        raise ValueError("fuel_*_pro arrays must have shape (nlitt, nelements)")
    if lignin_content_pro.shape != (nlevels,):
        raise ValueError("lignin_content_pro must have shape (nlevels,)")

    new_litter_pro = litter_pro + litter[point, :, pft, :, :] * frac
    new_fuel_1hr_pro = fuel_1hr_pro + fuel_1hr[point, pft, :, :] * frac
    new_fuel_10hr_pro = fuel_10hr_pro + fuel_10hr[point, pft, :, :] * frac
    new_fuel_100hr_pro = fuel_100hr_pro + fuel_100hr[point, pft, :, :] * frac
    new_fuel_1000hr_pro = fuel_1000hr_pro + fuel_1000hr[point, pft, :, :] * frac
    new_lignin_content_pro = (
        lignin_content_pro
        + litter[point, ISTRUCTURAL, pft, :, ICARBON] * frac * lignin_struc[point, pft, :]
    )

    return ForestHarvestProxyPoolsResult(
        litter_pro=new_litter_pro,
        fuel_1hr_pro=new_fuel_1hr_pro,
        fuel_10hr_pro=new_fuel_10hr_pro,
        fuel_100hr_pro=new_fuel_100hr_pro,
        fuel_1000hr_pro=new_fuel_1000hr_pro,
        lignin_content_pro=new_lignin_content_pro,
    )


def harvest_herb_legacy_step(
    *,
    biomass,
    bm_to_litter_pro,
    point,
    pft,
    veget_frac,
) -> HerbHarvestResult:
    """Transfer cleared herbaceous biomass to the proxy litter input.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``harvest_herb``, lines 1002-1019; same branch in
    ``stomate_glcchange_SinAgeC_fh.f90`` lines 167-183.
    """

    biomass = jnp.asarray(biomass)
    bm_to_litter_pro = jnp.asarray(bm_to_litter_pro)
    veget_frac = jnp.asarray(veget_frac, dtype=biomass.dtype)
    point = int(point)
    pft = int(pft)

    if biomass.ndim != 4 or biomass.shape[2] < NPARTS:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    npts, nvm, _, nelements = biomass.shape
    if not (0 <= point < npts) or not (0 <= pft < nvm):
        raise ValueError("point and pft must address biomass axes")
    if bm_to_litter_pro.shape != (NPARTS, nelements):
        raise ValueError("bm_to_litter_pro must have shape (nparts, nelements)")

    return HerbHarvestResult(
        bm_to_litter_pro=bm_to_litter_pro + biomass[point, pft, :, :] * veget_frac,
    )


def initialize_proxy_pft_step(
    *,
    pft,
    veget_max_pro,
    co2_to_bm_pro,
    bm_sapl,
    cn_sapl,
    is_tree,
    npp_longterm_init,
) -> ProxyPftInitializationResult:
    """Initialize the proxy youngest-age-class PFT used by LCC.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``initialize_proxy_pft``, lines 1032-1108; same branch in
    ``stomate_glcchange_SinAgeC_fh.f90`` lines 197-273.
    """

    bm_sapl = jnp.asarray(bm_sapl)
    cn_sapl = jnp.asarray(cn_sapl)
    is_tree = jnp.asarray(is_tree)
    veget_max_pro = jnp.asarray(veget_max_pro, dtype=bm_sapl.dtype)
    co2_to_bm_pro = jnp.asarray(co2_to_bm_pro, dtype=bm_sapl.dtype)
    npp_longterm_init = jnp.asarray(npp_longterm_init, dtype=bm_sapl.dtype)
    pft = int(pft)

    if bm_sapl.ndim != 3 or bm_sapl.shape[1] != NPARTS or bm_sapl.shape[2] <= ICARBON:
        raise ValueError("bm_sapl must have shape (nvm, nparts, nelements)")
    nvm, _, _ = bm_sapl.shape
    if not (0 <= pft < nvm):
        raise ValueError("pft must address bm_sapl axis")
    if cn_sapl.shape != (nvm,) or is_tree.shape != (nvm,):
        raise ValueError("cn_sapl and is_tree must have shape (nvm,)")

    cn_ind = jnp.where(is_tree[pft], cn_sapl[pft], 1.0)
    ind = veget_max_pro / cn_ind
    biomass_pro = ind * bm_sapl[pft, :, :]
    co2_to_bm_pro = co2_to_bm_pro + jnp.sum(ind * bm_sapl[pft, :, ICARBON])

    leaf_frac_pro = jnp.zeros((NLEAFAGES,), dtype=bm_sapl.dtype).at[0].set(veget_max_pro)
    leaf_age_pro = jnp.zeros((NLEAFAGES,), dtype=bm_sapl.dtype).at[0].set(veget_max_pro)

    return ProxyPftInitializationResult(
        biomass_pro=biomass_pro,
        co2_to_bm_pro=co2_to_bm_pro,
        ind_pro=ind * veget_max_pro,
        age_pro=jnp.asarray(0.0, dtype=bm_sapl.dtype),
        senescence_pro=jnp.asarray(False),
        pft_present_pro=jnp.asarray(True),
        lm_lastyearmax_pro=bm_sapl[pft, ILEAF, ICARBON] * ind * veget_max_pro,
        everywhere_pro=veget_max_pro,
        npp_longterm_pro=npp_longterm_init * veget_max_pro,
        leaf_frac_pro=leaf_frac_pro,
        leaf_age_pro=leaf_age_pro,
    )


def sap_take_step(
    *,
    biomass,
    veget_max,
    biomass_pro,
    co2_to_bm_pro,
    point,
    age_class_pfts,
    min_stomate=1.0e-8,
) -> SapTakeResult:
    """Take proxy sapling biomass from existing age-class biomass when possible.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``sap_take``, lines 1118-1159; same branch in
    ``stomate_glcchange_SinAgeC_fh.f90`` lines 283-324.
    """

    biomass = jnp.asarray(biomass)
    veget_max = jnp.asarray(veget_max)
    biomass_pro = jnp.asarray(biomass_pro)
    co2_to_bm_pro = jnp.asarray(co2_to_bm_pro, dtype=biomass.dtype)
    age_class_indices = tuple(int(v) for v in age_class_pfts)
    age_class_pfts = jnp.asarray(age_class_indices, dtype=jnp.int32)
    min_stomate = jnp.asarray(min_stomate, dtype=biomass.dtype)
    point = int(point)

    if biomass.ndim != 4 or biomass.shape[2] != NPARTS:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    npts, nvm, _, nelements = biomass.shape
    if not (0 <= point < npts):
        raise ValueError("point must address biomass axis")
    if veget_max.shape != (npts, nvm):
        raise ValueError("veget_max must have shape (npts, nvm)")
    if biomass_pro.shape != (NPARTS, nelements):
        raise ValueError("biomass_pro must have shape (nparts, nelements)")
    if age_class_pfts.ndim != 1:
        raise ValueError("age_class_pfts must be a one-dimensional index array")

    selected_biomass = biomass[point, age_class_pfts, :, :]
    selected_veget = veget_max[point, age_class_pfts]
    active = selected_veget > min_stomate
    biomass_total = jnp.sum(
        selected_biomass * selected_veget[:, None, None] * active[:, None, None],
        axis=0,
    )

    new_biomass = biomass
    new_co2_to_bm_pro = co2_to_bm_pro
    for part in range(NPARTS):
        enough = biomass_total[part, ICARBON] > biomass_pro[part, ICARBON]
        new_co2_to_bm_pro = jnp.where(
            enough,
            new_co2_to_bm_pro - biomass_pro[part, ICARBON],
            new_co2_to_bm_pro,
        )
        total_c = biomass_total[part, ICARBON]
        for pft_value in age_class_indices:
            is_active = veget_max[point, pft_value] > min_stomate
            bm_org = biomass[point, pft_value, part, ICARBON] * veget_max[point, pft_value]
            bmpro_share = bm_org / jnp.where(total_c != 0.0, total_c, 1.0) * biomass_pro[part, ICARBON]
            updated = (bm_org - bmpro_share) / veget_max[point, pft_value]
            current = new_biomass[point, pft_value, part, ICARBON]
            new_biomass = new_biomass.at[point, pft_value, part, ICARBON].set(
                jnp.where(enough & is_active, updated, current)
            )

    return SapTakeResult(
        biomass=new_biomass,
        co2_to_bm_pro=new_co2_to_bm_pro,
        biomass_total=biomass_total,
    )


def collect_legacy_scalar_pools_step(
    *,
    glcc_frac,
    bm_to_litter,
    carbon,
    deepC_a,
    deepC_s,
    deepC_p,
    co2_to_bm,
    gpp_daily,
    npp_daily,
    resp_maint,
    resp_growth,
    resp_hetero,
    co2_fire,
    litter_pro,
    lignin_content_pro,
    point,
    bm_to_litter_pro_initial=0.0,
    min_stomate=1.0e-8,
) -> LegacyScalarPoolsResult:
    """Collect non-harvest legacy state into proxy PFT accumulators.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``collect_legacy_pft``, lines 1300 and 1337-1356; same block appears in
    ``stomate_glcchange_SinAgeC_fh.f90`` within ``collect_legacy_pft``.
    Harvest-specific product, litter, fuel, and lignin-content inputs are
    handled by the smaller harvest helpers before this accumulation step.
    """

    glcc_frac = jnp.asarray(glcc_frac)
    bm_to_litter = jnp.asarray(bm_to_litter)
    carbon = jnp.asarray(carbon)
    deepC_a = jnp.asarray(deepC_a)
    deepC_s = jnp.asarray(deepC_s)
    deepC_p = jnp.asarray(deepC_p)
    co2_to_bm = jnp.asarray(co2_to_bm)
    gpp_daily = jnp.asarray(gpp_daily)
    npp_daily = jnp.asarray(npp_daily)
    resp_maint = jnp.asarray(resp_maint)
    resp_growth = jnp.asarray(resp_growth)
    resp_hetero = jnp.asarray(resp_hetero)
    co2_fire = jnp.asarray(co2_fire)
    litter_pro = jnp.asarray(litter_pro)
    lignin_content_pro = jnp.asarray(lignin_content_pro)
    min_stomate = jnp.asarray(min_stomate, dtype=bm_to_litter.dtype)
    point = int(point)

    if bm_to_litter.ndim != 4 or bm_to_litter.shape[2] != NPARTS:
        raise ValueError("bm_to_litter must have shape (npts, nvm, nparts, nelements)")
    npts, nvm, _, nelements = bm_to_litter.shape
    if not (0 <= point < npts):
        raise ValueError("point must address bm_to_litter axis")
    if glcc_frac.shape != (nvm,):
        raise ValueError("glcc_frac must have shape (nvm,)")
    if carbon.shape[:1] != (npts,) or carbon.shape[2] != nvm:
        raise ValueError("carbon must have shape (npts, ncarb, nvm)")
    if deepC_a.shape != deepC_s.shape or deepC_a.shape != deepC_p.shape or deepC_a.shape[:1] != (npts,) or deepC_a.shape[2] != nvm:
        raise ValueError("deepC_* arrays must have shape (npts, ndeep, nvm)")
    scalar_shape = (npts, nvm)
    if (
        co2_to_bm.shape != scalar_shape
        or gpp_daily.shape != scalar_shape
        or npp_daily.shape != scalar_shape
        or resp_maint.shape != scalar_shape
        or resp_growth.shape != scalar_shape
        or resp_hetero.shape != scalar_shape
        or co2_fire.shape != scalar_shape
    ):
        raise ValueError("daily scalar arrays must have shape (npts, nvm)")
    if litter_pro.ndim != 3 or litter_pro.shape[0] != NLITT or litter_pro.shape[2] <= ICARBON:
        raise ValueError("litter_pro must have shape (nlitt, nlevels, nelements)")
    nlevels = litter_pro.shape[1]
    if lignin_content_pro.shape != (nlevels,):
        raise ValueError("lignin_content_pro must have shape (nlevels,)")

    if jnp.asarray(bm_to_litter_pro_initial).shape == ():
        bm_to_litter_pro = jnp.zeros((NPARTS, nelements), dtype=bm_to_litter.dtype) + bm_to_litter_pro_initial
    else:
        bm_to_litter_pro = jnp.asarray(bm_to_litter_pro_initial)
        if bm_to_litter_pro.shape != (NPARTS, nelements):
            raise ValueError("bm_to_litter_pro_initial must be scalar or shape (nparts, nelements)")

    frac = jnp.where(glcc_frac > 0.0, glcc_frac, 0.0)
    weights_parts = frac[:, None, None]
    weights = frac[:, None]

    bm_to_litter_pro = bm_to_litter_pro + jnp.sum(bm_to_litter[point, :, :, :] * weights_parts, axis=0)
    carbon_pro = jnp.sum(carbon[point, :, :] * weights.T, axis=1)
    deepC_a_pro = jnp.sum(deepC_a[point, :, :] * weights.T, axis=1)
    deepC_s_pro = jnp.sum(deepC_s[point, :, :] * weights.T, axis=1)
    deepC_p_pro = jnp.sum(deepC_p[point, :, :] * weights.T, axis=1)

    lignin_denominator = litter_pro[ISTRUCTURAL, :, ICARBON]
    lignin_struc_pro = jnp.where(
        lignin_denominator > min_stomate,
        lignin_content_pro / lignin_denominator,
        0.0,
    )

    return LegacyScalarPoolsResult(
        veget_max_pro=jnp.sum(frac),
        bm_to_litter_pro=bm_to_litter_pro,
        carbon_pro=carbon_pro,
        deepC_a_pro=deepC_a_pro,
        deepC_s_pro=deepC_s_pro,
        deepC_p_pro=deepC_p_pro,
        co2_to_bm_pro=jnp.sum(co2_to_bm[point, :] * frac),
        gpp_daily_pro=jnp.sum(gpp_daily[point, :] * frac),
        npp_daily_pro=jnp.sum(npp_daily[point, :] * frac),
        resp_maint_pro=jnp.sum(resp_maint[point, :] * frac),
        resp_growth_pro=jnp.sum(resp_growth[point, :] * frac),
        resp_hetero_pro=jnp.sum(resp_hetero[point, :] * frac),
        co2_fire_pro=jnp.sum(co2_fire[point, :] * frac),
        lignin_struc_pro=lignin_struc_pro,
    )


def empty_pft_lcc_step(
    *,
    point,
    pft,
    veget_max,
    biomass,
    ind,
    carbon,
    litter,
    lignin_struc,
    bm_to_litter,
    deepC_a,
    deepC_s,
    deepC_p,
    fuel_1hr,
    fuel_10hr,
    fuel_100hr,
    fuel_1000hr,
    gpp_daily,
    npp_daily,
    gpp_week,
    npp_longterm,
    co2_to_bm,
    resp_maint,
    resp_growth,
    resp_hetero,
    lm_lastyearmax,
    leaf_frac,
    leaf_age,
    age,
    everywhere,
    pft_present,
    when_growthinit,
    senescence,
    gdd_from_growthinit,
    gdd_midwinter,
    time_hum_min,
    gdd_m5_dormance,
    ncd_dormance,
    moiavail_month,
    moiavail_week,
    ngd_minus5,
) -> EmptyPftResult:
    """Empty one PFT slot after land-cover loss exhausts its fraction.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``empty_pft``, lines 2044-2179.
    """

    point = int(point)
    pft = int(pft)
    veget_max = jnp.asarray(veget_max)
    biomass = jnp.asarray(biomass)
    ind = jnp.asarray(ind)
    carbon = jnp.asarray(carbon)
    litter = jnp.asarray(litter)
    lignin_struc = jnp.asarray(lignin_struc)
    bm_to_litter = jnp.asarray(bm_to_litter)
    deepC_a = jnp.asarray(deepC_a)
    deepC_s = jnp.asarray(deepC_s)
    deepC_p = jnp.asarray(deepC_p)
    fuel_1hr = jnp.asarray(fuel_1hr)
    fuel_10hr = jnp.asarray(fuel_10hr)
    fuel_100hr = jnp.asarray(fuel_100hr)
    fuel_1000hr = jnp.asarray(fuel_1000hr)
    gpp_daily = jnp.asarray(gpp_daily)
    npp_daily = jnp.asarray(npp_daily)
    gpp_week = jnp.asarray(gpp_week)
    npp_longterm = jnp.asarray(npp_longterm)
    co2_to_bm = jnp.asarray(co2_to_bm)
    resp_maint = jnp.asarray(resp_maint)
    resp_growth = jnp.asarray(resp_growth)
    resp_hetero = jnp.asarray(resp_hetero)
    lm_lastyearmax = jnp.asarray(lm_lastyearmax)
    leaf_frac = jnp.asarray(leaf_frac)
    leaf_age = jnp.asarray(leaf_age)
    age = jnp.asarray(age)
    everywhere = jnp.asarray(everywhere)
    pft_present = jnp.asarray(pft_present)
    when_growthinit = jnp.asarray(when_growthinit)
    senescence = jnp.asarray(senescence)
    gdd_from_growthinit = jnp.asarray(gdd_from_growthinit)
    gdd_midwinter = jnp.asarray(gdd_midwinter)
    time_hum_min = jnp.asarray(time_hum_min)
    gdd_m5_dormance = jnp.asarray(gdd_m5_dormance)
    ncd_dormance = jnp.asarray(ncd_dormance)
    moiavail_month = jnp.asarray(moiavail_month)
    moiavail_week = jnp.asarray(moiavail_week)
    ngd_minus5 = jnp.asarray(ngd_minus5)

    return EmptyPftResult(
        veget_max=veget_max.at[point, pft].set(0.0),
        biomass=biomass.at[point, pft, :, :].set(0.0),
        ind=ind.at[point, pft].set(0.0),
        carbon=carbon.at[point, :, pft].set(0.0),
        litter=litter.at[point, :, pft, :, :].set(0.0),
        lignin_struc=lignin_struc.at[point, pft, :].set(0.0),
        bm_to_litter=bm_to_litter.at[point, pft, :, :].set(0.0),
        deepC_a=deepC_a.at[point, :, pft].set(0.0),
        deepC_s=deepC_s.at[point, :, pft].set(0.0),
        deepC_p=deepC_p.at[point, :, pft].set(0.0),
        fuel_1hr=fuel_1hr.at[point, pft, :, :].set(0.0),
        fuel_10hr=fuel_10hr.at[point, pft, :, :].set(0.0),
        fuel_100hr=fuel_100hr.at[point, pft, :, :].set(0.0),
        fuel_1000hr=fuel_1000hr.at[point, pft, :, :].set(0.0),
        gpp_daily=gpp_daily.at[point, pft].set(0.0),
        npp_daily=npp_daily.at[point, pft].set(0.0),
        gpp_week=gpp_week.at[point, pft].set(0.0),
        npp_longterm=npp_longterm.at[point, pft].set(0.0),
        co2_to_bm=co2_to_bm.at[point, pft].set(0.0),
        resp_maint=resp_maint.at[point, pft].set(0.0),
        resp_growth=resp_growth.at[point, pft].set(0.0),
        resp_hetero=resp_hetero.at[point, pft].set(0.0),
        lm_lastyearmax=lm_lastyearmax.at[point, pft].set(0.0),
        leaf_frac=leaf_frac.at[point, pft, :].set(0.0),
        leaf_age=leaf_age.at[point, pft, :].set(0.0),
        age=age.at[point, pft].set(0.0),
        everywhere=everywhere.at[point, pft].set(0.0),
        pft_present=pft_present.at[point, pft].set(False),
        when_growthinit=when_growthinit.at[point, pft].set(0.0),
        senescence=senescence.at[point, pft].set(False),
        gdd_from_growthinit=gdd_from_growthinit.at[point, pft].set(0.0),
        gdd_midwinter=gdd_midwinter.at[point, pft].set(0.0),
        time_hum_min=time_hum_min.at[point, pft].set(0.0),
        gdd_m5_dormance=gdd_m5_dormance.at[point, pft].set(0.0),
        ncd_dormance=ncd_dormance.at[point, pft].set(0.0),
        moiavail_month=moiavail_month.at[point, pft].set(0.0),
        moiavail_week=moiavail_week.at[point, pft].set(0.0),
        ngd_minus5=ngd_minus5.at[point, pft].set(0.0),
    )


def add_incoming_proxy_pft_step(
    *,
    point,
    pft,
    veget_max_pro,
    carbon_pro,
    litter_pro,
    lignin_struc_pro,
    bm_to_litter_pro,
    deepC_a_pro,
    deepC_s_pro,
    deepC_p_pro,
    fuel_1hr_pro,
    fuel_10hr_pro,
    fuel_100hr_pro,
    fuel_1000hr_pro,
    biomass_pro,
    co2_to_bm_pro,
    npp_longterm_pro,
    ind_pro,
    lm_lastyearmax_pro,
    age_pro,
    everywhere_pro,
    leaf_frac_pro,
    leaf_age_pro,
    pft_present_pro,
    senescence_pro,
    gpp_daily_pro,
    npp_daily_pro,
    resp_maint_pro,
    resp_growth_pro,
    resp_hetero_pro,
    co2_fire_pro,
    veget_max,
    carbon,
    litter,
    lignin_struc,
    bm_to_litter,
    deepC_a,
    deepC_s,
    deepC_p,
    fuel_1hr,
    fuel_10hr,
    fuel_100hr,
    fuel_1000hr,
    biomass,
    co2_to_bm,
    npp_longterm,
    ind,
    lm_lastyearmax,
    age,
    everywhere,
    leaf_frac,
    leaf_age,
    pft_present,
    senescence,
    gpp_daily,
    npp_daily,
    resp_maint,
    resp_growth,
    resp_hetero,
    co2_fire,
    min_stomate=1.0e-8,
) -> IncomingProxyPftResult:
    """Merge a collected incoming proxy PFT into the target youngest age class.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``add_incoming_proxy_pft``, lines 1751-2033.
    """

    point = int(point)
    pft = int(pft)
    veget_max_pro = jnp.asarray(veget_max_pro)
    min_stomate = jnp.asarray(min_stomate)
    carbon_pro = jnp.asarray(carbon_pro)
    litter_pro = jnp.asarray(litter_pro)
    lignin_struc_pro = jnp.asarray(lignin_struc_pro)
    bm_to_litter_pro = jnp.asarray(bm_to_litter_pro)
    deepC_a_pro = jnp.asarray(deepC_a_pro)
    deepC_s_pro = jnp.asarray(deepC_s_pro)
    deepC_p_pro = jnp.asarray(deepC_p_pro)
    fuel_1hr_pro = jnp.asarray(fuel_1hr_pro)
    fuel_10hr_pro = jnp.asarray(fuel_10hr_pro)
    fuel_100hr_pro = jnp.asarray(fuel_100hr_pro)
    fuel_1000hr_pro = jnp.asarray(fuel_1000hr_pro)
    biomass_pro = jnp.asarray(biomass_pro)
    co2_to_bm_pro = jnp.asarray(co2_to_bm_pro)
    npp_longterm_pro = jnp.asarray(npp_longterm_pro)
    ind_pro = jnp.asarray(ind_pro)
    lm_lastyearmax_pro = jnp.asarray(lm_lastyearmax_pro)
    age_pro = jnp.asarray(age_pro)
    everywhere_pro = jnp.asarray(everywhere_pro)
    leaf_frac_pro = jnp.asarray(leaf_frac_pro)
    leaf_age_pro = jnp.asarray(leaf_age_pro)
    gpp_daily_pro = jnp.asarray(gpp_daily_pro)
    npp_daily_pro = jnp.asarray(npp_daily_pro)
    resp_maint_pro = jnp.asarray(resp_maint_pro)
    resp_growth_pro = jnp.asarray(resp_growth_pro)
    resp_hetero_pro = jnp.asarray(resp_hetero_pro)
    co2_fire_pro = jnp.asarray(co2_fire_pro)

    veget_max = jnp.asarray(veget_max)
    carbon = jnp.asarray(carbon)
    litter = jnp.asarray(litter)
    lignin_struc = jnp.asarray(lignin_struc)
    bm_to_litter = jnp.asarray(bm_to_litter)
    deepC_a = jnp.asarray(deepC_a)
    deepC_s = jnp.asarray(deepC_s)
    deepC_p = jnp.asarray(deepC_p)
    fuel_1hr = jnp.asarray(fuel_1hr)
    fuel_10hr = jnp.asarray(fuel_10hr)
    fuel_100hr = jnp.asarray(fuel_100hr)
    fuel_1000hr = jnp.asarray(fuel_1000hr)
    biomass = jnp.asarray(biomass)
    co2_to_bm = jnp.asarray(co2_to_bm)
    npp_longterm = jnp.asarray(npp_longterm)
    ind = jnp.asarray(ind)
    lm_lastyearmax = jnp.asarray(lm_lastyearmax)
    age = jnp.asarray(age)
    everywhere = jnp.asarray(everywhere)
    leaf_frac = jnp.asarray(leaf_frac)
    leaf_age = jnp.asarray(leaf_age)
    pft_present = jnp.asarray(pft_present)
    senescence = jnp.asarray(senescence)
    gpp_daily = jnp.asarray(gpp_daily)
    npp_daily = jnp.asarray(npp_daily)
    resp_maint = jnp.asarray(resp_maint)
    resp_growth = jnp.asarray(resp_growth)
    resp_hetero = jnp.asarray(resp_hetero)
    co2_fire = jnp.asarray(co2_fire)

    veget_old = veget_max[point, pft]
    veget_total = veget_old + veget_max_pro
    litter_old = litter

    new_veget_max = veget_max.at[point, pft].set(veget_total)
    new_carbon = carbon.at[point, :, pft].set((veget_old * carbon[point, :, pft] + carbon_pro) / veget_total)
    new_deepC_a = deepC_a.at[point, :, pft].set((veget_old * deepC_a[point, :, pft] + deepC_a_pro) / veget_total)
    new_deepC_s = deepC_s.at[point, :, pft].set((veget_old * deepC_s[point, :, pft] + deepC_s_pro) / veget_total)
    new_deepC_p = deepC_p.at[point, :, pft].set((veget_old * deepC_p[point, :, pft] + deepC_p_pro) / veget_total)
    new_litter = litter.at[point, :, pft, :, :].set((veget_old * litter[point, :, pft, :, :] + litter_pro) / veget_total)
    new_fuel_1hr = fuel_1hr.at[point, pft, :, :].set((veget_old * fuel_1hr[point, pft, :, :] + fuel_1hr_pro) / veget_total)
    new_fuel_10hr = fuel_10hr.at[point, pft, :, :].set((veget_old * fuel_10hr[point, pft, :, :] + fuel_10hr_pro) / veget_total)
    new_fuel_100hr = fuel_100hr.at[point, pft, :, :].set((veget_old * fuel_100hr[point, pft, :, :] + fuel_100hr_pro) / veget_total)
    new_fuel_1000hr = fuel_1000hr.at[point, pft, :, :].set((veget_old * fuel_1000hr[point, pft, :, :] + fuel_1000hr_pro) / veget_total)

    structural_litter_c = new_litter[point, ISTRUCTURAL, pft, :, ICARBON]
    lignin_merged = (
        veget_old * litter_old[point, ISTRUCTURAL, pft, :, ICARBON] * lignin_struc[point, pft, :]
        + litter_pro[ISTRUCTURAL, :, ICARBON] * lignin_struc_pro
    ) / (veget_total * structural_litter_c)
    new_lignin_struc = lignin_struc.at[point, pft, :].set(
        jnp.where(structural_litter_c > min_stomate, lignin_merged, lignin_struc[point, pft, :])
    )

    new_bm_to_litter = bm_to_litter.at[point, pft, :, :].set((veget_old * bm_to_litter[point, pft, :, :] + bm_to_litter_pro) / veget_total)
    new_biomass = biomass.at[point, pft, :, :].set((biomass[point, pft, :, :] * veget_old + biomass_pro) / veget_total)
    new_co2_to_bm = co2_to_bm.at[point, pft].set((veget_old * co2_to_bm[point, pft] + co2_to_bm_pro) / veget_total)
    new_ind = ind.at[point, pft].set((ind[point, pft] * veget_old + ind_pro) / veget_total)
    new_lm_lastyearmax = lm_lastyearmax.at[point, pft].set((lm_lastyearmax[point, pft] * veget_old + lm_lastyearmax_pro) / veget_total)
    new_npp_longterm = npp_longterm.at[point, pft].set((veget_old * npp_longterm[point, pft] + npp_longterm_pro) / veget_total)
    new_leaf_frac = leaf_frac.at[point, pft, :].set((leaf_frac[point, pft, :] * veget_old + leaf_frac_pro) / veget_total)
    new_leaf_age = leaf_age.at[point, pft, :].set((leaf_age[point, pft, :] * veget_old + leaf_age_pro) / veget_total)
    new_age = age.at[point, pft].set((veget_old * age[point, pft] + age_pro) / veget_total)
    new_everywhere = everywhere.at[point, pft].set(jnp.maximum(everywhere[point, pft], everywhere_pro))

    new_gpp_daily = gpp_daily.at[point, pft].set((veget_old * gpp_daily[point, pft] + gpp_daily_pro) / veget_total)
    new_npp_daily = npp_daily.at[point, pft].set((veget_old * npp_daily[point, pft] + npp_daily_pro) / veget_total)
    new_resp_maint = resp_maint.at[point, pft].set((veget_old * resp_maint[point, pft] + resp_maint_pro) / veget_total)
    new_resp_growth = resp_growth.at[point, pft].set((veget_old * resp_growth[point, pft] + resp_growth_pro) / veget_total)
    new_resp_hetero = resp_hetero.at[point, pft].set((veget_old * resp_hetero[point, pft] + resp_hetero_pro) / veget_total)
    new_co2_fire = co2_fire.at[point, pft].set((veget_old * co2_fire[point, pft] + co2_fire_pro) / veget_total)

    return IncomingProxyPftResult(
        veget_max=new_veget_max,
        carbon=new_carbon,
        litter=new_litter,
        lignin_struc=new_lignin_struc,
        bm_to_litter=new_bm_to_litter,
        deepC_a=new_deepC_a,
        deepC_s=new_deepC_s,
        deepC_p=new_deepC_p,
        fuel_1hr=new_fuel_1hr,
        fuel_10hr=new_fuel_10hr,
        fuel_100hr=new_fuel_100hr,
        fuel_1000hr=new_fuel_1000hr,
        biomass=new_biomass,
        co2_to_bm=new_co2_to_bm,
        npp_longterm=new_npp_longterm,
        ind=new_ind,
        lm_lastyearmax=new_lm_lastyearmax,
        age=new_age,
        everywhere=new_everywhere,
        leaf_frac=new_leaf_frac,
        leaf_age=new_leaf_age,
        pft_present=pft_present.at[point, pft].set(pft_present_pro),
        senescence=senescence.at[point, pft].set(senescence_pro),
        gpp_daily=new_gpp_daily,
        npp_daily=new_npp_daily,
        resp_maint=new_resp_maint,
        resp_growth=new_resp_growth,
        resp_hetero=new_resp_hetero,
        co2_fire=new_co2_fire,
    )


def lcc_age_class_indices(
    *,
    start_index,
    nagec_pft,
    is_tree,
    natural,
    is_grassland_manag,
    nagec_tree,
    nagec_herb,
) -> LccAgeClassIndices:
    """Build gross-LCC category and age-class index groups.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``gross_glcc_firstday_fh``, lines 2311-2504. Inputs and returned indices
    are 0-based JAX/Python PFT indices; Fortran source indices are 1-based.
    """

    start_index = [int(v) for v in start_index]
    nagec_pft = [int(v) for v in nagec_pft]
    is_tree = [bool(v) for v in is_tree]
    natural = [bool(v) for v in natural]
    is_grassland_manag = [bool(v) for v in is_grassland_manag]

    groups = {
        "tree": {"all": [], "old": [], "multi": []},
        "grass": {"all": [], "old": [], "multi": []},
        "pasture": {"all": [], "old": [], "multi": []},
        "crop": {"all": [], "old": [], "multi": []},
    }

    for ivma in range(1, len(start_index)):
        staind = start_index[ivma]
        count = nagec_pft[ivma]
        if is_tree[staind]:
            key = "tree"
        elif is_grassland_manag[staind]:
            key = "pasture"
        elif natural[staind]:
            key = "grass"
        else:
            key = "crop"
        groups[key]["old"].append(staind + count - 1)
        groups[key]["all"].extend(range(staind, staind + count))
        if count > 1:
            groups[key]["multi"].append((staind, count))

    def _agec_matrix(key, default_age_count):
        multi = groups[key]["multi"]
        if not multi:
            values = [[v] for v in groups[key]["all"]]
            if not values:
                return jnp.zeros((0, 1), dtype=jnp.int32)
            return jnp.asarray(values, dtype=jnp.int32)
        rows = []
        for staind, _count in multi:
            rows.append([staind + default_age_count - j - 1 for j in range(1, default_age_count)])
        return jnp.asarray(rows, dtype=jnp.int32)

    return LccAgeClassIndices(
        indall_tree=jnp.asarray(groups["tree"]["all"], dtype=jnp.int32),
        indold_tree=jnp.asarray(groups["tree"]["old"], dtype=jnp.int32),
        indagec_tree=_agec_matrix("tree", int(nagec_tree)),
        indall_grass=jnp.asarray(groups["grass"]["all"], dtype=jnp.int32),
        indold_grass=jnp.asarray(groups["grass"]["old"], dtype=jnp.int32),
        indagec_grass=_agec_matrix("grass", int(nagec_herb)),
        indall_pasture=jnp.asarray(groups["pasture"]["all"], dtype=jnp.int32),
        indold_pasture=jnp.asarray(groups["pasture"]["old"], dtype=jnp.int32),
        indagec_pasture=_agec_matrix("pasture", int(nagec_herb)),
        indall_crop=jnp.asarray(groups["crop"]["all"], dtype=jnp.int32),
        indold_crop=jnp.asarray(groups["crop"]["old"], dtype=jnp.int32),
        indagec_crop=_agec_matrix("crop", int(nagec_herb)),
    )


def lcc_calc_cover_step(
    *,
    veget_max,
    start_index,
    nagec_pft,
    is_tree,
    natural,
    is_grassland_manag,
    nagec_tree,
    nagec_herb,
) -> LccCoverResult:
    """Aggregate PFT cover to MTC and old-to-young age-class cover.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``calc_cover``, lines 3319-3394. Inputs and indices are 0-based in the JAX
    implementation.
    """

    veget_max = jnp.asarray(veget_max)
    start_index = [int(v) for v in start_index]
    nagec_pft = [int(v) for v in nagec_pft]
    is_tree = [bool(v) for v in is_tree]
    natural = [bool(v) for v in natural]
    is_grassland_manag = [bool(v) for v in is_grassland_manag]
    nagec_tree = int(nagec_tree)
    nagec_herb = int(nagec_herb)

    if veget_max.ndim != 2:
        raise ValueError("veget_max must have shape (npts, nvm)")
    npts = veget_max.shape[0]
    nvmap = len(start_index)
    veget_mtc = jnp.zeros((npts, nvmap), dtype=veget_max.dtype)
    vegagec_tree = jnp.zeros((npts, nagec_tree), dtype=veget_max.dtype)
    vegagec_grass = jnp.zeros((npts, nagec_herb), dtype=veget_max.dtype)
    vegagec_pasture = jnp.zeros((npts, nagec_herb), dtype=veget_max.dtype)
    vegagec_crop = jnp.zeros((npts, nagec_herb), dtype=veget_max.dtype)

    for ivma in range(nvmap):
        staind = start_index[ivma]
        count = nagec_pft[ivma]
        veget_mtc = veget_mtc.at[:, ivma].set(jnp.sum(veget_max[:, staind : staind + count], axis=1))

    for ivma in range(1, nvmap):
        staind = start_index[ivma]
        count = nagec_pft[ivma]
        endind = staind + count - 1
        if is_tree[staind]:
            if count == 1:
                vegagec_tree = vegagec_tree.at[:, 0].add(veget_max[:, staind])
            else:
                for j in range(nagec_tree):
                    vegagec_tree = vegagec_tree.at[:, j].add(veget_max[:, endind - j])
        elif is_grassland_manag[staind]:
            if count == 1:
                vegagec_pasture = vegagec_pasture.at[:, 0].add(veget_max[:, staind])
            else:
                for j in range(nagec_herb):
                    vegagec_pasture = vegagec_pasture.at[:, j].add(veget_max[:, endind - j])
        elif natural[staind]:
            if count == 1:
                vegagec_grass = vegagec_grass.at[:, 0].add(veget_max[:, staind])
            else:
                for j in range(nagec_herb):
                    vegagec_grass = vegagec_grass.at[:, j].add(veget_max[:, endind - j])
        else:
            if count == 1:
                vegagec_crop = vegagec_crop.at[:, 0].add(veget_max[:, staind])
            else:
                for j in range(nagec_herb):
                    vegagec_crop = vegagec_crop.at[:, j].add(veget_max[:, endind - j])

    return LccCoverResult(
        veget_mtc=veget_mtc,
        vegagec_tree=vegagec_tree,
        vegagec_grass=vegagec_grass,
        vegagec_pasture=vegagec_pasture,
        vegagec_crop=vegagec_crop,
    )


def lcc_cross_give_receive_step(
    *,
    point,
    frac_used,
    veget_mtc,
    donor_pfts,
    receiver_agec_pfts,
    nagec_receive,
    agec_group,
    veget_max,
    glcc_pft,
    glcc_pftmtc,
    glcc_pft_tmp,
    min_stomate=1.0e-8,
) -> CrossGiveReceiveResult:
    """Allocate gross-LCC outgoing fraction among donor and receiver PFTs.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``cross_give_receive``, lines 3076-3152. Inputs and indices are 0-based in
    this JAX implementation.
    """

    point = int(point)
    frac_used = jnp.asarray(frac_used)
    veget_mtc = jnp.asarray(veget_mtc)
    donor_pfts = jnp.asarray([int(v) for v in donor_pfts], dtype=jnp.int32)
    receiver_agec_pfts = jnp.asarray(receiver_agec_pfts, dtype=jnp.int32)
    nagec_receive = int(nagec_receive)
    agec_group = jnp.asarray(agec_group, dtype=jnp.int32)
    veget_max = jnp.asarray(veget_max)
    glcc_pft = jnp.asarray(glcc_pft)
    glcc_pftmtc = jnp.asarray(glcc_pftmtc)
    glcc_pft_tmp = jnp.asarray(glcc_pft_tmp)
    min_stomate = jnp.asarray(min_stomate, dtype=veget_max.dtype)

    donor_total = jnp.sum(veget_max[point, donor_pfts])
    donor_tmp = veget_max[point, donor_pfts] / donor_total * frac_used
    new_glcc_pft_tmp = glcc_pft_tmp.at[point, donor_pfts].set(donor_tmp)
    new_glcc_pft = glcc_pft.at[point, donor_pfts].add(donor_tmp)
    new_veget_max = veget_max.at[point, donor_pfts].add(-donor_tmp)

    iyoung = 0 if nagec_receive == 1 else nagec_receive - 2
    receiver_pfts = receiver_agec_pfts[:, iyoung]
    receiver_mtc = agec_group[receiver_pfts]
    totalveg = jnp.sum(veget_mtc[point, receiver_mtc])
    receiver_count = receiver_pfts.shape[0]

    new_glcc_pftmtc = glcc_pftmtc
    for idx in range(receiver_count):
        mtc = int(receiver_mtc[idx])
        allocation = jnp.where(
            totalveg > min_stomate,
            donor_tmp * veget_mtc[point, mtc] / jnp.where(totalveg != 0.0, totalveg, 1.0),
            donor_tmp / receiver_count,
        )
        new_glcc_pftmtc = new_glcc_pftmtc.at[point, donor_pfts, mtc].set(allocation)

    return CrossGiveReceiveResult(
        veget_max=new_veget_max,
        glcc_pft=new_glcc_pft,
        glcc_pftmtc=new_glcc_pftmtc,
        glcc_pft_tmp=new_glcc_pft_tmp,
    )


def lcc_type_conversion_step(
    *,
    point,
    transition_index,
    glcc_real,
    veget_mtc,
    donor_old_pfts,
    donor_agec_pfts,
    receiver_agec_pfts,
    nagec_receive,
    agec_group,
    vegagec_donor,
    veget_max,
    glcc_pft,
    glcc_pftmtc,
    glcc_pft_tmp,
    glcc_remain,
    iagec_start,
    iagec_end,
    old_to_young=True,
    min_stomate=1.0e-8,
) -> TypeConversionResult:
    """Allocate one gross-LCC type conversion across donor age classes.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``type_conversion``, lines 3163-3299. The JAX implementation uses 0-based
    inclusive ``iagec_start``/``iagec_end`` bounds.
    """

    point = int(point)
    transition_index = int(transition_index)
    iagec_start = int(iagec_start)
    iagec_end = int(iagec_end)
    glcc_real = jnp.asarray(glcc_real)
    veget_mtc = jnp.asarray(veget_mtc)
    donor_agec_pfts = jnp.asarray(donor_agec_pfts, dtype=jnp.int32)
    receiver_agec_pfts = jnp.asarray(receiver_agec_pfts, dtype=jnp.int32)
    vegagec_donor = jnp.asarray(vegagec_donor)
    veget_max = jnp.asarray(veget_max)
    glcc_pft = jnp.asarray(glcc_pft)
    glcc_pftmtc = jnp.asarray(glcc_pftmtc)
    glcc_pft_tmp = jnp.asarray(glcc_pft_tmp)
    glcc_remain = jnp.asarray(glcc_remain)
    min_stomate = jnp.asarray(min_stomate, dtype=veget_max.dtype)

    frac_begin = glcc_real[point, transition_index]
    if old_to_young:
        age_range = range(iagec_start, iagec_end + 1)
    else:
        age_range = range(iagec_start, iagec_end - 1, -1)

    for iagec in age_range:
        available = vegagec_donor[point, iagec]
        frac_used = jnp.where(
            available > frac_begin,
            frac_begin,
            jnp.where(available > min_stomate, available, 0.0),
        )
        if bool(frac_used > min_stomate):
            donors = donor_old_pfts if iagec == 0 else donor_agec_pfts[:, iagec - 1]
            cross = lcc_cross_give_receive_step(
                point=point,
                frac_used=frac_used,
                veget_mtc=veget_mtc,
                donor_pfts=donors,
                receiver_agec_pfts=receiver_agec_pfts,
                nagec_receive=nagec_receive,
                agec_group=agec_group,
                veget_max=veget_max,
                glcc_pft=glcc_pft,
                glcc_pftmtc=glcc_pftmtc,
                glcc_pft_tmp=glcc_pft_tmp,
                min_stomate=min_stomate,
            )
            veget_max = cross.veget_max
            glcc_pft = cross.glcc_pft
            glcc_pftmtc = cross.glcc_pftmtc
            glcc_pft_tmp = cross.glcc_pft_tmp
            frac_begin = frac_begin - frac_used
            vegagec_donor = vegagec_donor.at[point, iagec].add(-frac_used)
            glcc_remain = glcc_remain.at[point, transition_index].add(-frac_used)

    return TypeConversionResult(
        vegagec_donor=vegagec_donor,
        veget_max=veget_max,
        glcc_pft=glcc_pft,
        glcc_pftmtc=glcc_pftmtc,
        glcc_pft_tmp=glcc_pft_tmp,
        glcc_remain=glcc_remain,
    )


def lcc_glcc_compensation_full_step(
    *,
    veget_4veg,
    glcc,
    glcc_real,
    glcc_def,
    incre_deficit,
    first_transition,
    first_source,
    second_transition,
    second_source,
    third_transition,
    third_source,
    target_slot,
    min_stomate=1.0e-8,
) -> GlccCompensationFullResult:
    """Apply one target-vegetation gross-LCC feasibility compensation pass.

    Fortran provenance: ``src_stomate/stomate_glcchange_SinAgeC_fh.f90``,
    subroutine ``glcc_compensation_full``, lines 2216-2395. The argument names
    are generalized from the Fortran dummy names ``p2c/g2c/f2c``: the
    compensation sequence is first-source, second-source, third-source, while
    the initial availability checks are applied to third, first, then second
    exactly as in the source.
    """

    veget_4veg = jnp.asarray(veget_4veg)
    glcc = jnp.asarray(glcc)
    glcc_real = jnp.asarray(glcc_real)
    glcc_def = jnp.asarray(glcc_def)
    incre_deficit = jnp.asarray(incre_deficit)
    first_transition = int(first_transition)
    first_source = int(first_source)
    second_transition = int(second_transition)
    second_source = int(second_source)
    third_transition = int(third_transition)
    third_source = int(third_source)
    target_slot = int(target_slot)
    min_stomate = jnp.asarray(min_stomate, dtype=veget_4veg.dtype)

    npts = veget_4veg.shape[0]
    for point in range(npts):
        source_value = veget_4veg[point, third_source]
        prescribed = glcc[point, third_transition]
        if bool(source_value > min_stomate):
            glcc_def = glcc_def.at[point, third_transition].set(source_value - prescribed)
            glcc_real = glcc_real.at[point, third_transition].set(
                jnp.where(source_value > prescribed, prescribed, source_value)
            )
        else:
            glcc_real = glcc_real.at[point, third_transition].set(0.0)
            glcc_def = glcc_def.at[point, third_transition].set(-prescribed)

        source_value = veget_4veg[point, first_source]
        prescribed = glcc[point, first_transition]
        if bool(source_value > min_stomate):
            glcc_def = glcc_def.at[point, first_transition].set(source_value - prescribed)
            glcc_real = glcc_real.at[point, first_transition].set(
                jnp.where(source_value > prescribed, prescribed, source_value)
            )
        else:
            glcc_real = glcc_real.at[point, first_transition].set(0.0)
            glcc_def = glcc_def.at[point, first_transition].set(-prescribed)

        source_value = veget_4veg[point, second_source]
        prescribed = glcc[point, second_transition]
        if bool(source_value > min_stomate):
            glcc_def = glcc_def.at[point, second_transition].set(source_value - prescribed)
            glcc_real = glcc_real.at[point, second_transition].set(
                jnp.where(source_value > prescribed, prescribed, source_value)
            )
        else:
            glcc_real = glcc_real.at[point, second_transition].set(0.0)
            glcc_def = glcc_def.at[point, second_transition].set(-prescribed)

        first_def = glcc_def[point, first_transition]
        second_def = glcc_def[point, second_transition]
        third_def = glcc_def[point, third_transition]
        tmpdef = first_def + second_def + third_def

        if bool(first_def < 0.0):
            if bool(second_def < 0.0):
                if bool(third_def < 0.0):
                    incre_deficit = incre_deficit.at[point, target_slot].set(tmpdef)
                elif bool(tmpdef >= min_stomate):
                    glcc_real = glcc_real.at[point, third_transition].set(
                        glcc_real[point, third_transition] - second_def - first_def
                    )
                else:
                    glcc_real = glcc_real.at[point, third_transition].set(veget_4veg[point, third_source])
                    incre_deficit = incre_deficit.at[point, target_slot].set(tmpdef)
            elif bool(third_def < 0.0):
                if bool(tmpdef >= min_stomate):
                    glcc_real = glcc_real.at[point, second_transition].set(
                        glcc_real[point, second_transition] - first_def - third_def
                    )
                else:
                    glcc_real = glcc_real.at[point, second_transition].set(veget_4veg[point, second_source])
                    incre_deficit = incre_deficit.at[point, target_slot].set(tmpdef)
            elif bool(tmpdef >= min_stomate):
                if bool((second_def + first_def) >= min_stomate):
                    glcc_real = glcc_real.at[point, second_transition].set(
                        glcc_real[point, second_transition] - first_def
                    )
                else:
                    glcc_real = glcc_real.at[point, second_transition].set(veget_4veg[point, second_source])
                    glcc_real = glcc_real.at[point, third_transition].set(
                        glcc_real[point, third_transition] - (first_def + second_def)
                    )
            else:
                glcc_real = glcc_real.at[point, second_transition].set(veget_4veg[point, second_source])
                glcc_real = glcc_real.at[point, third_transition].set(veget_4veg[point, third_source])
                incre_deficit = incre_deficit.at[point, target_slot].set(tmpdef)
        elif bool(second_def < 0.0):
            if bool(third_def < 0.0):
                if bool(tmpdef >= min_stomate):
                    glcc_real = glcc_real.at[point, first_transition].set(
                        glcc_real[point, first_transition] - second_def - third_def
                    )
                else:
                    incre_deficit = incre_deficit.at[point, target_slot].set(tmpdef)
                    glcc_real = glcc_real.at[point, first_transition].set(veget_4veg[point, first_source])
            elif bool(tmpdef >= min_stomate):
                if bool((first_def + second_def) >= min_stomate):
                    glcc_real = glcc_real.at[point, first_transition].set(
                        glcc_real[point, first_transition] - second_def
                    )
                else:
                    glcc_real = glcc_real.at[point, first_transition].set(veget_4veg[point, first_source])
                    glcc_real = glcc_real.at[point, third_transition].set(
                        glcc_real[point, third_transition] - (second_def + first_def)
                    )
            else:
                incre_deficit = incre_deficit.at[point, target_slot].set(tmpdef)
                glcc_real = glcc_real.at[point, first_transition].set(veget_4veg[point, first_source])
                glcc_real = glcc_real.at[point, third_transition].set(veget_4veg[point, third_source])
        elif bool(third_def < 0.0):
            if bool(tmpdef >= min_stomate):
                if bool((first_def + third_def) >= min_stomate):
                    glcc_real = glcc_real.at[point, first_transition].set(
                        glcc_real[point, first_transition] - third_def
                    )
                else:
                    glcc_real = glcc_real.at[point, first_transition].set(veget_4veg[point, first_source])
                    glcc_real = glcc_real.at[point, second_transition].set(
                        glcc_real[point, second_transition] - (third_def + first_def)
                    )
            else:
                incre_deficit = incre_deficit.at[point, target_slot].set(tmpdef)
                glcc_real = glcc_real.at[point, second_transition].set(veget_4veg[point, second_source])
                glcc_real = glcc_real.at[point, first_transition].set(veget_4veg[point, first_source])

        veget_4veg = veget_4veg.at[point, third_source].add(-glcc_real[point, third_transition])
        veget_4veg = veget_4veg.at[point, second_source].add(-glcc_real[point, second_transition])
        veget_4veg = veget_4veg.at[point, first_source].add(-glcc_real[point, first_transition])

    return GlccCompensationFullResult(
        veget_4veg=veget_4veg,
        glcc_real=glcc_real,
        glcc_def=glcc_def,
        incre_deficit=incre_deficit,
    )


def lcc_harvest_same_mtc_step(
    *,
    glcc_pft,
    glcc_pft_tmp,
    glcc_pftmtc,
    is_tree,
    pft_to_mtc,
) -> HarvestSameMtcResult:
    """Reset harvest receiver slots and map tree harvest loss to its own MTC.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``gross_glcc_firstday_fh``, lines 2662-2675.
    """

    glcc_pft = jnp.asarray(glcc_pft)
    glcc_pft_tmp = jnp.asarray(glcc_pft_tmp)
    glcc_pftmtc = jnp.asarray(glcc_pftmtc)
    is_tree = jnp.asarray(is_tree)
    pft_to_mtc = jnp.asarray(pft_to_mtc, dtype=jnp.int32)

    if glcc_pft.ndim != 2:
        raise ValueError("glcc_pft must have shape (npts, nvm)")
    npts, nvm = glcc_pft.shape
    if glcc_pft_tmp.shape != (npts, nvm):
        raise ValueError("glcc_pft_tmp must have shape (npts, nvm)")
    if glcc_pftmtc.ndim != 3 or glcc_pftmtc.shape[:2] != (npts, nvm):
        raise ValueError("glcc_pftmtc must have shape (npts, nvm, nvmap)")
    if is_tree.shape != (nvm,) or pft_to_mtc.shape != (nvm,):
        raise ValueError("is_tree and pft_to_mtc must have shape (nvm,)")

    new_glcc_pft_tmp = jnp.zeros_like(glcc_pft_tmp)
    new_glcc_pftmtc = jnp.zeros_like(glcc_pftmtc)
    for pft in range(nvm):
        new_glcc_pftmtc = new_glcc_pftmtc.at[:, pft, pft_to_mtc[pft]].set(
            jnp.where(is_tree[pft], glcc_pft[:, pft], 0.0)
        )

    return HarvestSameMtcResult(
        glcc_pft_tmp=new_glcc_pft_tmp,
        glcc_pftmtc=new_glcc_pftmtc,
    )


def lcc_primary_net_transition_sequence_step(
    *,
    glcc_primary_shift,
    glcc_net_lcc,
    veget_mtc,
    indices: LccAgeClassIndices,
    agec_group,
    vegagec_tree,
    vegagec_grass,
    vegagec_pasture,
    vegagec_crop,
    veget_max,
    glcc_pft,
    glcc_pftmtc,
    glcc_pft_tmp,
    incre_deficit=0.0,
    nagec_tree=1,
    nagec_herb=1,
    min_stomate=1.0e-8,
) -> PrimaryNetLccSequenceResult:
    """Apply primary-agriculture shift plus net-LCC transition allocation.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``gross_glcc_firstday_fh``, lines 2946-3057. This helper covers only the
    section where ``glccReal = glccPrimaryShift + glccNetLCC`` is allocated by
    the twelve `type_conversion` calls in Fortran order.
    """

    glcc_primary_shift = jnp.asarray(glcc_primary_shift)
    glcc_net_lcc = jnp.asarray(glcc_net_lcc)
    veget_mtc = jnp.asarray(veget_mtc)
    agec_group = jnp.asarray(agec_group, dtype=jnp.int32)
    vegagec_tree = jnp.asarray(vegagec_tree)
    vegagec_grass = jnp.asarray(vegagec_grass)
    vegagec_pasture = jnp.asarray(vegagec_pasture)
    vegagec_crop = jnp.asarray(vegagec_crop)
    veget_max = jnp.asarray(veget_max)
    glcc_pft = jnp.asarray(glcc_pft)
    glcc_pftmtc = jnp.asarray(glcc_pftmtc)
    glcc_pft_tmp = jnp.asarray(glcc_pft_tmp)
    if jnp.asarray(incre_deficit).shape == ():
        incre_deficit = jnp.zeros_like(glcc_primary_shift) + incre_deficit
    else:
        incre_deficit = jnp.asarray(incre_deficit)
    nagec_tree = int(nagec_tree)
    nagec_herb = int(nagec_herb)

    glcc_real = glcc_primary_shift + glcc_net_lcc
    glcc_remain = glcc_real
    npts = glcc_real.shape[0]

    transitions = (
        (2, "tree", "crop", nagec_tree, nagec_herb),
        (1, "tree", "pasture", nagec_tree, nagec_herb),
        (0, "tree", "grass", nagec_tree, nagec_herb),
        (5, "grass", "crop", nagec_herb, nagec_herb),
        (4, "grass", "pasture", nagec_herb, nagec_herb),
        (3, "grass", "tree", nagec_herb, nagec_tree),
        (8, "pasture", "crop", nagec_herb, nagec_herb),
        (7, "pasture", "grass", nagec_herb, nagec_herb),
        (6, "pasture", "tree", nagec_herb, nagec_tree),
        (11, "crop", "pasture", nagec_herb, nagec_herb),
        (10, "crop", "grass", nagec_herb, nagec_herb),
        (9, "crop", "tree", nagec_herb, nagec_tree),
    )

    def _donor_state(name):
        if name == "tree":
            return indices.indold_tree, indices.indagec_tree, vegagec_tree
        if name == "grass":
            return indices.indold_grass, indices.indagec_grass, vegagec_grass
        if name == "pasture":
            return indices.indold_pasture, indices.indagec_pasture, vegagec_pasture
        if name == "crop":
            return indices.indold_crop, indices.indagec_crop, vegagec_crop
        raise ValueError(f"unknown donor category {name}")

    def _receiver_agec(name):
        if name == "tree":
            return indices.indagec_tree
        if name == "grass":
            return indices.indagec_grass
        if name == "pasture":
            return indices.indagec_pasture
        if name == "crop":
            return indices.indagec_crop
        raise ValueError(f"unknown receiver category {name}")

    def _set_donor_cover(name, value):
        nonlocal vegagec_tree, vegagec_grass, vegagec_pasture, vegagec_crop
        if name == "tree":
            vegagec_tree = value
        elif name == "grass":
            vegagec_grass = value
        elif name == "pasture":
            vegagec_pasture = value
        elif name == "crop":
            vegagec_crop = value
        else:
            raise ValueError(f"unknown donor category {name}")

    for point in range(npts):
        for transition_index, donor_name, receiver_name, donor_nagec, receiver_nagec in transitions:
            donor_old, donor_agec, donor_cover = _donor_state(donor_name)
            result = lcc_type_conversion_step(
                point=point,
                transition_index=transition_index,
                glcc_real=glcc_real,
                veget_mtc=veget_mtc,
                donor_old_pfts=donor_old,
                donor_agec_pfts=donor_agec,
                receiver_agec_pfts=_receiver_agec(receiver_name),
                nagec_receive=receiver_nagec,
                agec_group=agec_group,
                vegagec_donor=donor_cover,
                veget_max=veget_max,
                glcc_pft=glcc_pft,
                glcc_pftmtc=glcc_pftmtc,
                glcc_pft_tmp=glcc_pft_tmp,
                glcc_remain=glcc_remain,
                iagec_start=0,
                iagec_end=donor_nagec - 1,
                old_to_young=True,
                min_stomate=min_stomate,
            )
            veget_max = result.veget_max
            glcc_pft = result.glcc_pft
            glcc_pftmtc = result.glcc_pftmtc
            glcc_pft_tmp = result.glcc_pft_tmp
            glcc_remain = result.glcc_remain
            _set_donor_cover(donor_name, result.vegagec_donor)

    return PrimaryNetLccSequenceResult(
        glcc_real=glcc_real,
        glcc_remain=glcc_remain,
        veget_max=veget_max,
        glcc_pft=glcc_pft,
        glcc_pftmtc=glcc_pftmtc,
        glcc_pft_tmp=glcc_pft_tmp,
        vegagec_tree=vegagec_tree,
        vegagec_grass=vegagec_grass,
        vegagec_pasture=vegagec_pasture,
        vegagec_crop=vegagec_crop,
        incre_deficit=incre_deficit - glcc_remain,
    )


def lcc_secondary_shift_transition_sequence_step(
    *,
    glcc_second_shift,
    veget_mtc,
    indices: LccAgeClassIndices,
    agec_group,
    vegagec_tree,
    vegagec_grass,
    vegagec_pasture,
    vegagec_crop,
    veget_max,
    glcc_pft,
    glcc_pftmtc,
    glcc_pft_tmp,
    start_index,
    nagec_pft,
    is_tree,
    natural,
    is_grassland_manag,
    incre_deficit=0.0,
    nagec_tree=1,
    nagec_herb=1,
    min_stomate=1.0e-8,
) -> SecondaryShiftLccSequenceResult:
    """Apply the two-pass secondary-agriculture shifting cultivation block.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``gross_glcc_firstday_fh``, lines 2678-2851. This implements the source
    sequence: first all twelve transitions with forest donation restricted to
    the configured secondary age class, then remaining forest transitions in a
    young-to-old pass, and finally ``IncreDeficit = -glccRemain`` for this block.
    """

    glcc_second_shift = jnp.asarray(glcc_second_shift)
    veget_mtc = jnp.asarray(veget_mtc)
    agec_group = jnp.asarray(agec_group, dtype=jnp.int32)
    vegagec_tree = jnp.asarray(vegagec_tree)
    vegagec_grass = jnp.asarray(vegagec_grass)
    vegagec_pasture = jnp.asarray(vegagec_pasture)
    vegagec_crop = jnp.asarray(vegagec_crop)
    veget_max = jnp.asarray(veget_max)
    glcc_pft = jnp.asarray(glcc_pft)
    glcc_pftmtc = jnp.asarray(glcc_pftmtc)
    glcc_pft_tmp = jnp.asarray(glcc_pft_tmp)
    if jnp.asarray(incre_deficit).shape == ():
        incre_deficit = jnp.zeros_like(glcc_second_shift) + incre_deficit
    else:
        incre_deficit = jnp.asarray(incre_deficit)
    nagec_tree = int(nagec_tree)
    nagec_herb = int(nagec_herb)

    glcc_remain = glcc_second_shift
    npts = glcc_second_shift.shape[0]

    transitions = (
        (2, "tree", "crop", nagec_tree, nagec_herb),
        (1, "tree", "pasture", nagec_tree, nagec_herb),
        (0, "tree", "grass", nagec_tree, nagec_herb),
        (5, "grass", "crop", nagec_herb, nagec_herb),
        (4, "grass", "pasture", nagec_herb, nagec_herb),
        (3, "grass", "tree", nagec_herb, nagec_tree),
        (8, "pasture", "crop", nagec_herb, nagec_herb),
        (7, "pasture", "grass", nagec_herb, nagec_herb),
        (6, "pasture", "tree", nagec_herb, nagec_tree),
        (11, "crop", "pasture", nagec_herb, nagec_herb),
        (10, "crop", "grass", nagec_herb, nagec_herb),
        (9, "crop", "tree", nagec_herb, nagec_tree),
    )

    def _donor_state(name):
        if name == "tree":
            return indices.indold_tree, indices.indagec_tree, vegagec_tree
        if name == "grass":
            return indices.indold_grass, indices.indagec_grass, vegagec_grass
        if name == "pasture":
            return indices.indold_pasture, indices.indagec_pasture, vegagec_pasture
        if name == "crop":
            return indices.indold_crop, indices.indagec_crop, vegagec_crop
        raise ValueError(f"unknown donor category {name}")

    def _receiver_agec(name):
        if name == "tree":
            return indices.indagec_tree
        if name == "grass":
            return indices.indagec_grass
        if name == "pasture":
            return indices.indagec_pasture
        if name == "crop":
            return indices.indagec_crop
        raise ValueError(f"unknown receiver category {name}")

    def _set_donor_cover(name, value):
        nonlocal vegagec_tree, vegagec_grass, vegagec_pasture, vegagec_crop
        if name == "tree":
            vegagec_tree = value
        elif name == "grass":
            vegagec_grass = value
        elif name == "pasture":
            vegagec_pasture = value
        elif name == "crop":
            vegagec_crop = value
        else:
            raise ValueError(f"unknown donor category {name}")

    forest_first_index = max(nagec_tree - 3, 0)
    for point in range(npts):
        for transition_index, donor_name, receiver_name, donor_nagec, receiver_nagec in transitions:
            donor_old, donor_agec, donor_cover = _donor_state(donor_name)
            if donor_name == "tree":
                start_age = forest_first_index
                end_age = forest_first_index
            else:
                start_age = 0
                end_age = donor_nagec - 1
            result = lcc_type_conversion_step(
                point=point,
                transition_index=transition_index,
                glcc_real=glcc_second_shift,
                veget_mtc=veget_mtc,
                donor_old_pfts=donor_old,
                donor_agec_pfts=donor_agec,
                receiver_agec_pfts=_receiver_agec(receiver_name),
                nagec_receive=receiver_nagec,
                agec_group=agec_group,
                vegagec_donor=donor_cover,
                veget_max=veget_max,
                glcc_pft=glcc_pft,
                glcc_pftmtc=glcc_pftmtc,
                glcc_pft_tmp=glcc_pft_tmp,
                glcc_remain=glcc_remain,
                iagec_start=start_age,
                iagec_end=end_age,
                old_to_young=True,
                min_stomate=min_stomate,
            )
            veget_max = result.veget_max
            glcc_pft = result.glcc_pft
            glcc_pftmtc = result.glcc_pftmtc
            glcc_pft_tmp = result.glcc_pft_tmp
            glcc_remain = result.glcc_remain
            _set_donor_cover(donor_name, result.vegagec_donor)

    glcc_second_shift_remain = glcc_remain
    cover_after_first = lcc_calc_cover_step(
        veget_max=veget_max,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        nagec_tree=nagec_tree,
        nagec_herb=nagec_herb,
    )
    veget_mtc = cover_after_first.veget_mtc
    vegagec_tree = cover_after_first.vegagec_tree

    forest_second_start = max(nagec_tree - 4, 0)
    for point in range(npts):
        for transition_index, receiver_name in ((2, "crop"), (1, "pasture"), (0, "grass")):
            result = lcc_type_conversion_step(
                point=point,
                transition_index=transition_index,
                glcc_real=glcc_second_shift_remain,
                veget_mtc=veget_mtc,
                donor_old_pfts=indices.indold_tree,
                donor_agec_pfts=indices.indagec_tree,
                receiver_agec_pfts=_receiver_agec(receiver_name),
                nagec_receive=nagec_herb,
                agec_group=agec_group,
                vegagec_donor=vegagec_tree,
                veget_max=veget_max,
                glcc_pft=glcc_pft,
                glcc_pftmtc=glcc_pftmtc,
                glcc_pft_tmp=glcc_pft_tmp,
                glcc_remain=glcc_remain,
                iagec_start=forest_second_start,
                iagec_end=0,
                old_to_young=False,
                min_stomate=min_stomate,
            )
            veget_max = result.veget_max
            glcc_pft = result.glcc_pft
            glcc_pftmtc = result.glcc_pftmtc
            glcc_pft_tmp = result.glcc_pft_tmp
            glcc_remain = result.glcc_remain
            vegagec_tree = result.vegagec_donor

    return SecondaryShiftLccSequenceResult(
        glcc_second_shift_remain=glcc_second_shift_remain,
        glcc_remain=glcc_remain,
        veget_max=veget_max,
        glcc_pft=glcc_pft,
        glcc_pftmtc=glcc_pftmtc,
        glcc_pft_tmp=glcc_pft_tmp,
        incre_deficit=-glcc_remain + incre_deficit,
        cover_after_first_pass=cover_after_first,
    )


def lcc_forestry_harvest_sequence_step(
    *,
    harvest_matrix,
    veget_mtc,
    indices: LccAgeClassIndices,
    agec_group,
    vegagec_tree,
    veget_max,
    glcc_pft,
    glcc_pftmtc,
    glcc_pft_tmp,
    start_index,
    nagec_pft,
    is_tree,
    natural,
    is_grassland_manag,
    pft_to_mtc,
    nagec_tree=1,
    nagec_herb=1,
    min_stomate=1.0e-8,
) -> ForestryHarvestSequenceResult:
    """Allocate forestry harvest loss before other gross-LCC transitions.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``gross_glcc_firstday_fh``, lines 2516-2675. This helper covers the
    `sf2yf` and `pf2yf` harvest allocation, secondary-deficit compensation from
    primary forest, primary-harvest deficit bookkeeping, and same-MTC harvest
    remapping.
    """

    harvest_matrix = jnp.asarray(harvest_matrix)
    veget_mtc = jnp.asarray(veget_mtc)
    agec_group = jnp.asarray(agec_group, dtype=jnp.int32)
    vegagec_tree = jnp.asarray(vegagec_tree)
    veget_max = jnp.asarray(veget_max)
    glcc_pft = jnp.asarray(glcc_pft)
    glcc_pftmtc = jnp.asarray(glcc_pftmtc)
    glcc_pft_tmp = jnp.asarray(glcc_pft_tmp)
    is_tree = jnp.asarray(is_tree)
    pft_to_mtc = jnp.asarray(pft_to_mtc, dtype=jnp.int32)
    nagec_tree = int(nagec_tree)
    nagec_herb = int(nagec_herb)
    npts = harvest_matrix.shape[0]
    pf2yf = 0
    sf2yf = 1

    glcc_remain = harvest_matrix
    deficit_pf2yf_final = jnp.zeros((npts,), dtype=harvest_matrix.dtype)
    deficit_sf2yf_final = jnp.zeros((npts,), dtype=harvest_matrix.dtype)

    first_start = 1 if nagec_tree > 1 else 0
    first_end = max(nagec_tree - 2, 0)
    for point in range(npts):
        result = lcc_type_conversion_step(
            point=point,
            transition_index=sf2yf,
            glcc_real=harvest_matrix,
            veget_mtc=veget_mtc,
            donor_old_pfts=indices.indold_tree,
            donor_agec_pfts=indices.indagec_tree,
            receiver_agec_pfts=indices.indagec_pasture,
            nagec_receive=nagec_herb,
            agec_group=agec_group,
            vegagec_donor=vegagec_tree,
            veget_max=veget_max,
            glcc_pft=glcc_pft,
            glcc_pftmtc=glcc_pftmtc,
            glcc_pft_tmp=glcc_pft_tmp,
            glcc_remain=glcc_remain,
            iagec_start=first_start,
            iagec_end=first_end,
            old_to_young=True,
            min_stomate=min_stomate,
        )
        veget_max = result.veget_max
        glcc_pft = result.glcc_pft
        glcc_pftmtc = result.glcc_pftmtc
        glcc_pft_tmp = result.glcc_pft_tmp
        glcc_remain = result.glcc_remain
        vegagec_tree = result.vegagec_donor

    fhmatrix_remain_a = glcc_remain
    cover_after_secondary = lcc_calc_cover_step(
        veget_max=veget_max,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        nagec_tree=nagec_tree,
        nagec_herb=nagec_herb,
    )
    veget_mtc = cover_after_secondary.veget_mtc
    vegagec_tree = cover_after_secondary.vegagec_tree

    for point in range(npts):
        if bool(fhmatrix_remain_a[point, sf2yf] > min_stomate):
            result = lcc_type_conversion_step(
                point=point,
                transition_index=sf2yf,
                glcc_real=fhmatrix_remain_a,
                veget_mtc=veget_mtc,
                donor_old_pfts=indices.indold_tree,
                donor_agec_pfts=indices.indagec_tree,
                receiver_agec_pfts=indices.indagec_pasture,
                nagec_receive=nagec_herb,
                agec_group=agec_group,
                vegagec_donor=vegagec_tree,
                veget_max=veget_max,
                glcc_pft=glcc_pft,
                glcc_pftmtc=glcc_pftmtc,
                glcc_pft_tmp=glcc_pft_tmp,
                glcc_remain=glcc_remain,
                iagec_start=0,
                iagec_end=0,
                old_to_young=True,
                min_stomate=min_stomate,
            )
            veget_max = result.veget_max
            glcc_pft = result.glcc_pft
            glcc_pftmtc = result.glcc_pftmtc
            glcc_pft_tmp = result.glcc_pft_tmp
            glcc_remain = result.glcc_remain
            vegagec_tree = result.vegagec_donor

    fhmatrix_remain_b = glcc_remain
    cover_after_compensation = lcc_calc_cover_step(
        veget_max=veget_max,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        nagec_tree=nagec_tree,
        nagec_herb=nagec_herb,
    )
    veget_mtc = cover_after_compensation.veget_mtc
    vegagec_tree = cover_after_compensation.vegagec_tree

    for point in range(npts):
        if bool(fhmatrix_remain_b[point, sf2yf] > min_stomate):
            deficit_sf2yf_final = deficit_sf2yf_final.at[point].set(-fhmatrix_remain_b[point, sf2yf])
            deficit_pf2yf_final = deficit_pf2yf_final.at[point].set(-fhmatrix_remain_b[point, pf2yf])
        else:
            result = lcc_type_conversion_step(
                point=point,
                transition_index=pf2yf,
                glcc_real=fhmatrix_remain_b,
                veget_mtc=veget_mtc,
                donor_old_pfts=indices.indold_tree,
                donor_agec_pfts=indices.indagec_tree,
                receiver_agec_pfts=indices.indagec_pasture,
                nagec_receive=nagec_herb,
                agec_group=agec_group,
                vegagec_donor=vegagec_tree,
                veget_max=veget_max,
                glcc_pft=glcc_pft,
                glcc_pftmtc=glcc_pftmtc,
                glcc_pft_tmp=glcc_pft_tmp,
                glcc_remain=glcc_remain,
                iagec_start=0,
                iagec_end=max(nagec_tree - 2, 0),
                old_to_young=True,
                min_stomate=min_stomate,
            )
            veget_max = result.veget_max
            glcc_pft = result.glcc_pft
            glcc_pftmtc = result.glcc_pftmtc
            glcc_pft_tmp = result.glcc_pft_tmp
            glcc_remain = result.glcc_remain
            vegagec_tree = result.vegagec_donor
        deficit_pf2yf_final = deficit_pf2yf_final.at[point].set(
            jnp.where(glcc_remain[point, pf2yf] > min_stomate, -glcc_remain[point, pf2yf], deficit_pf2yf_final[point])
        )

    remap = lcc_harvest_same_mtc_step(
        glcc_pft=glcc_pft,
        glcc_pft_tmp=glcc_pft_tmp,
        glcc_pftmtc=glcc_pftmtc,
        is_tree=is_tree,
        pft_to_mtc=pft_to_mtc,
    )

    return ForestryHarvestSequenceResult(
        veget_max=veget_max,
        glcc_pft=glcc_pft,
        glcc_pftmtc=remap.glcc_pftmtc,
        glcc_pft_tmp=remap.glcc_pft_tmp,
        glcc_remain=glcc_remain,
        fhmatrix_remain_a=fhmatrix_remain_a,
        fhmatrix_remain_b=fhmatrix_remain_b,
        deficit_pf2yf_final=deficit_pf2yf_final,
        deficit_sf2yf_final=deficit_sf2yf_final,
        cover_after_secondary=cover_after_secondary,
        cover_after_compensation=cover_after_compensation,
    )


def lcc_gross_firstday_allocation_step(
    *,
    veget_max_org,
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
    nagec_tree=1,
    nagec_herb=1,
    min_stomate=1.0e-8,
) -> GrossFirstdayLccAllocationResult:
    """Allocate gross-LCC first-day fraction losses and receiver MTCs.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``gross_glcc_firstday_fh``, lines 2193-3059, excluding the commented frozen
    DGVM compensation block. This helper returns the allocation state consumed
    later by `gross_glcchange_fh`; it does not apply biomass/carbon legacy
    transfers.
    """

    veget_max_org = jnp.asarray(veget_max_org)
    harvest_matrix = jnp.asarray(harvest_matrix)
    glcc_second_shift = jnp.asarray(glcc_second_shift)
    glcc_primary_shift = jnp.asarray(glcc_primary_shift)
    glcc_net_lcc = jnp.asarray(glcc_net_lcc)
    pft_to_mtc = jnp.asarray(pft_to_mtc, dtype=jnp.int32)
    agec_group = jnp.asarray(agec_group, dtype=jnp.int32)
    npts, nvm = veget_max_org.shape
    nvmap = len(start_index)
    nagec_tree = int(nagec_tree)
    nagec_herb = int(nagec_herb)

    indices = lcc_age_class_indices(
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        nagec_tree=nagec_tree,
        nagec_herb=nagec_herb,
    )
    cover = lcc_calc_cover_step(
        veget_max=veget_max_org,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        nagec_tree=nagec_tree,
        nagec_herb=nagec_herb,
    )
    glcc_pft = jnp.zeros((npts, nvm), dtype=veget_max_org.dtype)
    glcc_pft_tmp = jnp.zeros((npts, nvm), dtype=veget_max_org.dtype)
    glcc_pftmtc = jnp.zeros((npts, nvm, nvmap), dtype=veget_max_org.dtype)

    harvest = lcc_forestry_harvest_sequence_step(
        harvest_matrix=harvest_matrix,
        veget_mtc=cover.veget_mtc,
        indices=indices,
        agec_group=agec_group,
        vegagec_tree=cover.vegagec_tree,
        veget_max=veget_max_org,
        glcc_pft=glcc_pft,
        glcc_pftmtc=glcc_pftmtc,
        glcc_pft_tmp=glcc_pft_tmp,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        pft_to_mtc=pft_to_mtc,
        nagec_tree=nagec_tree,
        nagec_herb=nagec_herb,
        min_stomate=min_stomate,
    )

    cover_after_harvest = lcc_calc_cover_step(
        veget_max=harvest.veget_max,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        nagec_tree=nagec_tree,
        nagec_herb=nagec_herb,
    )
    secondary = lcc_secondary_shift_transition_sequence_step(
        glcc_second_shift=glcc_second_shift,
        veget_mtc=cover_after_harvest.veget_mtc,
        indices=indices,
        agec_group=agec_group,
        vegagec_tree=cover_after_harvest.vegagec_tree,
        vegagec_grass=cover_after_harvest.vegagec_grass,
        vegagec_pasture=cover_after_harvest.vegagec_pasture,
        vegagec_crop=cover_after_harvest.vegagec_crop,
        veget_max=harvest.veget_max,
        glcc_pft=harvest.glcc_pft,
        glcc_pftmtc=harvest.glcc_pftmtc,
        glcc_pft_tmp=harvest.glcc_pft_tmp,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        incre_deficit=0.0,
        nagec_tree=nagec_tree,
        nagec_herb=nagec_herb,
        min_stomate=min_stomate,
    )

    cover_after_secondary = lcc_calc_cover_step(
        veget_max=secondary.veget_max,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        nagec_tree=nagec_tree,
        nagec_herb=nagec_herb,
    )
    primary_net = lcc_primary_net_transition_sequence_step(
        glcc_primary_shift=glcc_primary_shift,
        glcc_net_lcc=glcc_net_lcc,
        veget_mtc=cover_after_secondary.veget_mtc,
        indices=indices,
        agec_group=agec_group,
        vegagec_tree=cover_after_secondary.vegagec_tree,
        vegagec_grass=cover_after_secondary.vegagec_grass,
        vegagec_pasture=cover_after_secondary.vegagec_pasture,
        vegagec_crop=cover_after_secondary.vegagec_crop,
        veget_max=secondary.veget_max,
        glcc_pft=secondary.glcc_pft,
        glcc_pftmtc=secondary.glcc_pftmtc,
        glcc_pft_tmp=secondary.glcc_pft_tmp,
        incre_deficit=secondary.incre_deficit,
        nagec_tree=nagec_tree,
        nagec_herb=nagec_herb,
        min_stomate=min_stomate,
    )

    return GrossFirstdayLccAllocationResult(
        veget_max=primary_net.veget_max,
        glcc_real=primary_net.glcc_real,
        incre_deficit=primary_net.incre_deficit,
        glcc_pft=primary_net.glcc_pft,
        glcc_pftmtc=primary_net.glcc_pftmtc,
        glcc_pft_tmp=primary_net.glcc_pft_tmp,
        harvest=harvest,
        secondary_shift=secondary,
        primary_net=primary_net,
    )


def lcc_gross_firstday_single_age_allocation_step(
    *,
    veget_max_org,
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
    nagec_tree=1,
    nagec_herb=1,
    min_stomate=1.0e-8,
) -> SingleAgeGrossFirstdayLccAllocationResult:
    """Allocate ``SingleAgeClass`` gross-LCC first-day losses and receiver MTCs.

    Fortran provenance: ``src_stomate/stomate_glcchange_SinAgeC_fh.f90``,
    subroutine ``gross_glcc_firstday_SinAgeC_fh``, lines 1358-1950. This covers
    the source-backed first-day allocation boundary: simplified forest-harvest
    aggregation/capping, same-MTC harvest remap, `glcc_compensation_full`, and
    the twelve `type_conversion` calls in Fortran order. It does not by itself
    apply the downstream legacy biomass/carbon transfers owned by
    ``gross_glcchange_SinAgeC_fh``.
    """

    veget_max_org = jnp.asarray(veget_max_org)
    harvest_matrix = jnp.asarray(harvest_matrix)
    glcc_second_shift = jnp.asarray(glcc_second_shift)
    glcc_primary_shift = jnp.asarray(glcc_primary_shift)
    glcc_net_lcc = jnp.asarray(glcc_net_lcc)
    pft_to_mtc = jnp.asarray(pft_to_mtc, dtype=jnp.int32)
    agec_group = jnp.asarray(agec_group, dtype=jnp.int32)
    npts, nvm = veget_max_org.shape
    nvmap = len(start_index)
    nagec_tree = int(nagec_tree)
    nagec_herb = int(nagec_herb)
    min_stomate = jnp.asarray(min_stomate, dtype=veget_max_org.dtype)

    indices = lcc_age_class_indices(
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        nagec_tree=nagec_tree,
        nagec_herb=nagec_herb,
    )
    cover = lcc_calc_cover_step(
        veget_max=veget_max_org,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        nagec_tree=nagec_tree,
        nagec_herb=nagec_herb,
    )

    veget_max = veget_max_org
    glcc_pft = jnp.zeros((npts, nvm), dtype=veget_max_org.dtype)
    glcc_pft_tmp = jnp.zeros((npts, nvm), dtype=veget_max_org.dtype)
    glcc_pftmtc = jnp.zeros((npts, nvm, nvmap), dtype=veget_max_org.dtype)
    deficit_pf2yf_final = jnp.zeros((npts,), dtype=veget_max_org.dtype)
    deficit_sf2yf_final = jnp.zeros((npts,), dtype=veget_max_org.dtype)
    pf2yf_compen_sf2yf = jnp.zeros((npts,), dtype=veget_max_org.dtype)
    sf2yf_compen_pf2yf = jnp.zeros((npts,), dtype=veget_max_org.dtype)

    pf2yf = 0
    sf2yf = 1
    hmatrix_real = jnp.zeros_like(harvest_matrix)
    hmatrix_real = hmatrix_real.at[:, pf2yf].set(harvest_matrix[:, pf2yf] + harvest_matrix[:, sf2yf])
    veget_tree = jnp.sum(cover.vegagec_tree, axis=1)
    harvest_over = veget_tree <= hmatrix_real[:, pf2yf]
    deficit_pf2yf_final = jnp.where(harvest_over, veget_tree - hmatrix_real[:, pf2yf], deficit_pf2yf_final)
    hmatrix_real = hmatrix_real.at[:, pf2yf].set(
        jnp.where(harvest_over, veget_tree, hmatrix_real[:, pf2yf])
    )

    harvest_remain = hmatrix_real
    for point in range(npts):
        harvest = lcc_type_conversion_step(
            point=point,
            transition_index=pf2yf,
            glcc_real=hmatrix_real,
            veget_mtc=cover.veget_mtc,
            donor_old_pfts=indices.indold_tree,
            donor_agec_pfts=indices.indagec_tree,
            receiver_agec_pfts=indices.indagec_crop,
            nagec_receive=nagec_herb,
            agec_group=agec_group,
            vegagec_donor=cover.vegagec_tree,
            veget_max=veget_max,
            glcc_pft=glcc_pft,
            glcc_pftmtc=glcc_pftmtc,
            glcc_pft_tmp=glcc_pft_tmp,
            glcc_remain=harvest_remain,
            iagec_start=0,
            iagec_end=0,
            old_to_young=True,
            min_stomate=min_stomate,
        )
        veget_max = harvest.veget_max
        glcc_pft = harvest.glcc_pft
        glcc_pftmtc = harvest.glcc_pftmtc
        glcc_pft_tmp = harvest.glcc_pft_tmp
        harvest_remain = harvest.glcc_remain
        cover = cover._replace(vegagec_tree=harvest.vegagec_donor)

    remap = lcc_harvest_same_mtc_step(
        glcc_pft=glcc_pft,
        glcc_pft_tmp=glcc_pft_tmp,
        glcc_pftmtc=glcc_pftmtc,
        is_tree=is_tree,
        pft_to_mtc=pft_to_mtc,
    )
    glcc_pft_tmp = remap.glcc_pft_tmp
    glcc_pftmtc = remap.glcc_pftmtc
    veget_max_tmp = veget_max

    cover_after_harvest = lcc_calc_cover_step(
        veget_max=veget_max,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        nagec_tree=nagec_tree,
        nagec_herb=nagec_herb,
    )
    veget_4veg = jnp.stack(
        (
            jnp.sum(cover_after_harvest.vegagec_tree, axis=1),
            jnp.sum(cover_after_harvest.vegagec_grass, axis=1),
            jnp.sum(cover_after_harvest.vegagec_pasture, axis=1),
            jnp.sum(cover_after_harvest.vegagec_crop, axis=1),
        ),
        axis=1,
    )
    glcc = glcc_second_shift + glcc_primary_shift + glcc_net_lcc
    glcc_real = jnp.zeros_like(glcc)
    glcc_def = jnp.zeros_like(glcc)
    incre_deficit = jnp.zeros((npts, 4), dtype=veget_max_org.dtype)

    compensation_calls = (
        (8, 2, 5, 1, 2, 0, 3),
        (4, 1, 11, 3, 1, 0, 2),
        (7, 2, 10, 3, 0, 0, 1),
        (9, 3, 6, 2, 3, 1, 0),
    )
    for first_transition, first_source, second_transition, second_source, third_transition, third_source, target in compensation_calls:
        comp = lcc_glcc_compensation_full_step(
            veget_4veg=veget_4veg,
            glcc=glcc,
            glcc_real=glcc_real,
            glcc_def=glcc_def,
            incre_deficit=incre_deficit,
            first_transition=first_transition,
            first_source=first_source,
            second_transition=second_transition,
            second_source=second_source,
            third_transition=third_transition,
            third_source=third_source,
            target_slot=target,
            min_stomate=min_stomate,
        )
        veget_4veg = comp.veget_4veg
        glcc_real = comp.glcc_real
        glcc_def = comp.glcc_def
        incre_deficit = comp.incre_deficit

    veget_max = veget_max_tmp
    cover_after_compensation = lcc_calc_cover_step(
        veget_max=veget_max,
        start_index=start_index,
        nagec_pft=nagec_pft,
        is_tree=is_tree,
        natural=natural,
        is_grassland_manag=is_grassland_manag,
        nagec_tree=nagec_tree,
        nagec_herb=nagec_herb,
    )
    veget_mtc = cover_after_compensation.veget_mtc
    vegagec_tree = cover_after_compensation.vegagec_tree
    vegagec_grass = cover_after_compensation.vegagec_grass
    vegagec_pasture = cover_after_compensation.vegagec_pasture
    vegagec_crop = cover_after_compensation.vegagec_crop
    glcc_remain = glcc_real

    transitions = (
        (2, "tree", "crop", nagec_tree, nagec_herb),
        (1, "tree", "pasture", nagec_tree, nagec_herb),
        (0, "tree", "grass", nagec_tree, nagec_herb),
        (5, "grass", "crop", nagec_herb, nagec_herb),
        (4, "grass", "pasture", nagec_herb, nagec_herb),
        (3, "grass", "tree", nagec_herb, nagec_tree),
        (8, "pasture", "crop", nagec_herb, nagec_herb),
        (7, "pasture", "grass", nagec_herb, nagec_herb),
        (6, "pasture", "tree", nagec_herb, nagec_tree),
        (11, "crop", "pasture", nagec_herb, nagec_herb),
        (10, "crop", "grass", nagec_herb, nagec_herb),
        (9, "crop", "tree", nagec_herb, nagec_tree),
    )

    def _donor_state(name):
        if name == "tree":
            return indices.indold_tree, indices.indagec_tree, vegagec_tree
        if name == "grass":
            return indices.indold_grass, indices.indagec_grass, vegagec_grass
        if name == "pasture":
            return indices.indold_pasture, indices.indagec_pasture, vegagec_pasture
        if name == "crop":
            return indices.indold_crop, indices.indagec_crop, vegagec_crop
        raise ValueError(f"unknown donor category {name}")

    def _receiver_agec(name):
        if name == "tree":
            return indices.indagec_tree
        if name == "grass":
            return indices.indagec_grass
        if name == "pasture":
            return indices.indagec_pasture
        if name == "crop":
            return indices.indagec_crop
        raise ValueError(f"unknown receiver category {name}")

    def _set_donor_cover(name, value):
        nonlocal vegagec_tree, vegagec_grass, vegagec_pasture, vegagec_crop
        if name == "tree":
            vegagec_tree = value
        elif name == "grass":
            vegagec_grass = value
        elif name == "pasture":
            vegagec_pasture = value
        elif name == "crop":
            vegagec_crop = value
        else:
            raise ValueError(f"unknown donor category {name}")

    for point in range(npts):
        for transition_index, donor_name, receiver_name, donor_nagec, receiver_nagec in transitions:
            donor_old, donor_agec, donor_cover = _donor_state(donor_name)
            result = lcc_type_conversion_step(
                point=point,
                transition_index=transition_index,
                glcc_real=glcc_real,
                veget_mtc=veget_mtc,
                donor_old_pfts=donor_old,
                donor_agec_pfts=donor_agec,
                receiver_agec_pfts=_receiver_agec(receiver_name),
                nagec_receive=receiver_nagec,
                agec_group=agec_group,
                vegagec_donor=donor_cover,
                veget_max=veget_max,
                glcc_pft=glcc_pft,
                glcc_pftmtc=glcc_pftmtc,
                glcc_pft_tmp=glcc_pft_tmp,
                glcc_remain=glcc_remain,
                iagec_start=0,
                iagec_end=donor_nagec - 1,
                old_to_young=True,
                min_stomate=min_stomate,
            )
            veget_max = result.veget_max
            glcc_pft = result.glcc_pft
            glcc_pftmtc = result.glcc_pftmtc
            glcc_pft_tmp = result.glcc_pft_tmp
            glcc_remain = result.glcc_remain
            _set_donor_cover(donor_name, result.vegagec_donor)

    return SingleAgeGrossFirstdayLccAllocationResult(
        veget_max=veget_max,
        glcc_real=glcc_real,
        glcc_def=glcc_def,
        glcc_remain=glcc_remain,
        incre_deficit=incre_deficit,
        glcc_pft=glcc_pft,
        glcc_pftmtc=glcc_pftmtc,
        glcc_pft_tmp=glcc_pft_tmp,
        hmatrix_real=hmatrix_real,
        deficit_pf2yf_final=deficit_pf2yf_final,
        deficit_sf2yf_final=deficit_sf2yf_final,
        pf2yf_compen_sf2yf=pf2yf_compen_sf2yf,
        sf2yf_compen_pf2yf=sf2yf_compen_pf2yf,
    )


def lcc_collect_legacy_pft_step(
    *,
    point,
    receiver_mtc,
    start_index,
    glcc_pftmtc,
    biomass,
    bm_to_litter,
    carbon,
    litter,
    deepC_a,
    deepC_s,
    deepC_p,
    fuel_1hr,
    fuel_10hr,
    fuel_100hr,
    fuel_1000hr,
    lignin_struc,
    co2_to_bm,
    gpp_daily,
    npp_daily,
    resp_maint,
    resp_growth,
    resp_hetero,
    co2_fire,
    convflux,
    prod10,
    prod100,
    is_tree,
    coeff_lcchange_1,
    coeff_lcchange_10,
    coeff_lcchange_100,
    allow_deforest_fire=False,
    deforest_biomass_remain=None,
    min_stomate=1.0e-8,
) -> LccLegacyCollectionResult:
    """Collect donor-PFT legacy state for one receiving meta-class.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``collect_legacy_pft``, lines 1169-1356. This follows the donor loop and
    tree/herbaceous harvest dispatch used by ``gross_glcchange_fh`` before the
    source-fraction subtraction and proxy merge.
    """

    glcc_pftmtc = jnp.asarray(glcc_pftmtc)
    biomass = jnp.asarray(biomass)
    bm_to_litter = jnp.asarray(bm_to_litter)
    carbon = jnp.asarray(carbon)
    litter = jnp.asarray(litter)
    deepC_a = jnp.asarray(deepC_a)
    deepC_s = jnp.asarray(deepC_s)
    deepC_p = jnp.asarray(deepC_p)
    fuel_1hr = jnp.asarray(fuel_1hr)
    fuel_10hr = jnp.asarray(fuel_10hr)
    fuel_100hr = jnp.asarray(fuel_100hr)
    fuel_1000hr = jnp.asarray(fuel_1000hr)
    lignin_struc = jnp.asarray(lignin_struc)
    co2_to_bm = jnp.asarray(co2_to_bm)
    gpp_daily = jnp.asarray(gpp_daily)
    npp_daily = jnp.asarray(npp_daily)
    resp_maint = jnp.asarray(resp_maint)
    resp_growth = jnp.asarray(resp_growth)
    resp_hetero = jnp.asarray(resp_hetero)
    co2_fire = jnp.asarray(co2_fire)
    convflux = jnp.asarray(convflux)
    prod10 = jnp.asarray(prod10)
    prod100 = jnp.asarray(prod100)
    is_tree = jnp.asarray(is_tree)
    coeff_lcchange_1 = jnp.asarray(coeff_lcchange_1)
    coeff_lcchange_10 = jnp.asarray(coeff_lcchange_10)
    coeff_lcchange_100 = jnp.asarray(coeff_lcchange_100)
    min_stomate = jnp.asarray(min_stomate, dtype=biomass.dtype)
    point = int(point)
    receiver_mtc = int(receiver_mtc)
    start_index = tuple(int(v) for v in start_index)

    if glcc_pftmtc.ndim != 3:
        raise ValueError("glcc_pftmtc must have shape (npts, nvm, nvmap)")
    npts, nvm, nvmap = glcc_pftmtc.shape
    if not (0 <= point < npts) or not (0 <= receiver_mtc < nvmap):
        raise ValueError("point and receiver_mtc must address glcc_pftmtc axes")
    if len(start_index) != nvmap:
        raise ValueError("start_index length must match the receiver MTC axis")
    target_pft = start_index[receiver_mtc]
    if not (0 <= target_pft < nvm):
        raise ValueError("start_index[receiver_mtc] must address the PFT axis")
    if biomass.ndim != 4 or biomass.shape[:2] != (npts, nvm) or biomass.shape[2] != NPARTS:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    nelements = biomass.shape[3]
    if bm_to_litter.shape != biomass.shape:
        raise ValueError("bm_to_litter must share biomass shape")
    if carbon.shape[:1] != (npts,) or carbon.shape[2] != nvm:
        raise ValueError("carbon must have shape (npts, ncarb, nvm)")
    if litter.ndim != 5 or litter.shape[0] != npts or litter.shape[1] != NLITT or litter.shape[2] != nvm:
        raise ValueError("litter must have shape (npts, nlitt, nvm, nlevels, nelements)")
    nlevels = litter.shape[3]
    if litter.shape[4] != nelements:
        raise ValueError("litter and biomass must share the element axis")
    deep_shape = deepC_a.shape
    if deepC_s.shape != deep_shape or deepC_p.shape != deep_shape or deep_shape[:1] != (npts,) or deep_shape[2] != nvm:
        raise ValueError("deepC_* arrays must have shape (npts, ndeep, nvm)")
    fuel_shape = (npts, nvm, NLITT, nelements)
    if fuel_1hr.shape != fuel_shape or fuel_10hr.shape != fuel_shape or fuel_100hr.shape != fuel_shape or fuel_1000hr.shape != fuel_shape:
        raise ValueError("fuel arrays must have shape (npts, nvm, nlitt, nelements)")
    if lignin_struc.shape != (npts, nvm, nlevels):
        raise ValueError("lignin_struc must have shape (npts, nvm, nlevels)")
    scalar_shape = (npts, nvm)
    if (
        co2_to_bm.shape != scalar_shape
        or gpp_daily.shape != scalar_shape
        or npp_daily.shape != scalar_shape
        or resp_maint.shape != scalar_shape
        or resp_growth.shape != scalar_shape
        or resp_hetero.shape != scalar_shape
        or co2_fire.shape != scalar_shape
    ):
        raise ValueError("daily scalar arrays must have shape (npts, nvm)")
    if convflux.ndim != 2 or convflux.shape[0] != npts or convflux.shape[1] < NWP:
        raise ValueError("convflux must have shape (npts, nwp>=2)")
    if prod10.shape != (npts, 11, convflux.shape[1]):
        raise ValueError("prod10 must have shape (npts, 11, nwp)")
    if prod100.shape != (npts, 101, convflux.shape[1]):
        raise ValueError("prod100 must have shape (npts, 101, nwp)")
    if is_tree.shape != (nvm,):
        raise ValueError("is_tree must have shape (nvm,)")
    if coeff_lcchange_1.shape != (nvm,) or coeff_lcchange_10.shape != (nvm,) or coeff_lcchange_100.shape != (nvm,):
        raise ValueError("coeff_lcchange_* arrays must have shape (nvm,)")
    if allow_deforest_fire:
        if deforest_biomass_remain is None:
            raise ValueError("deforest_biomass_remain is required when allow_deforest_fire is true")
        deforest_biomass_remain = jnp.asarray(deforest_biomass_remain)
        if deforest_biomass_remain.shape != biomass.shape:
            raise ValueError("deforest_biomass_remain must share biomass shape")

    litter_pro = jnp.zeros((NLITT, nlevels, nelements), dtype=biomass.dtype)
    fuel_1hr_pro = jnp.zeros((NLITT, nelements), dtype=biomass.dtype)
    fuel_10hr_pro = jnp.zeros((NLITT, nelements), dtype=biomass.dtype)
    fuel_100hr_pro = jnp.zeros((NLITT, nelements), dtype=biomass.dtype)
    fuel_1000hr_pro = jnp.zeros((NLITT, nelements), dtype=biomass.dtype)
    lignin_content_pro = jnp.zeros((nlevels,), dtype=biomass.dtype)
    bm_to_litter_pro = jnp.zeros((NPARTS, nelements), dtype=biomass.dtype)

    glcc_frac = glcc_pftmtc[point, :, receiver_mtc]
    for donor_pft in range(nvm):
        frac = glcc_frac[donor_pft]
        if bool(frac > 0.0):
            if bool(is_tree[donor_pft]):
                product_pool = IWPHAR if bool(is_tree[target_pft]) else IWPLCC
                legacy = forest_harvest_legacy_step(
                    biomass=biomass,
                    bm_to_litter_pro=bm_to_litter_pro,
                    convflux=convflux[:, product_pool],
                    prod10=prod10[:, :, product_pool],
                    prod100=prod100[:, :, product_pool],
                    point=point,
                    pft=donor_pft,
                    frac=frac,
                    coeff_lcchange_1=coeff_lcchange_1,
                    coeff_lcchange_10=coeff_lcchange_10,
                    coeff_lcchange_100=coeff_lcchange_100,
                    allow_deforest_fire=allow_deforest_fire,
                    deforest_biomass_remain=deforest_biomass_remain,
                )
                bm_to_litter_pro = legacy.bm_to_litter_pro
                convflux = convflux.at[:, product_pool].set(legacy.convflux)
                prod10 = prod10.at[:, :, product_pool].set(legacy.prod10)
                prod100 = prod100.at[:, :, product_pool].set(legacy.prod100)
                proxy = forest_harvest_proxy_pools_step(
                    litter=litter,
                    fuel_1hr=fuel_1hr,
                    fuel_10hr=fuel_10hr,
                    fuel_100hr=fuel_100hr,
                    fuel_1000hr=fuel_1000hr,
                    lignin_struc=lignin_struc,
                    litter_pro=litter_pro,
                    fuel_1hr_pro=fuel_1hr_pro,
                    fuel_10hr_pro=fuel_10hr_pro,
                    fuel_100hr_pro=fuel_100hr_pro,
                    fuel_1000hr_pro=fuel_1000hr_pro,
                    lignin_content_pro=lignin_content_pro,
                    point=point,
                    pft=donor_pft,
                    frac=frac,
                )
                litter_pro = proxy.litter_pro
                fuel_1hr_pro = proxy.fuel_1hr_pro
                fuel_10hr_pro = proxy.fuel_10hr_pro
                fuel_100hr_pro = proxy.fuel_100hr_pro
                fuel_1000hr_pro = proxy.fuel_1000hr_pro
                lignin_content_pro = proxy.lignin_content_pro
            else:
                herb = harvest_herb_legacy_step(
                    biomass=biomass,
                    bm_to_litter_pro=bm_to_litter_pro,
                    point=point,
                    pft=donor_pft,
                    veget_frac=frac,
                )
                bm_to_litter_pro = herb.bm_to_litter_pro
                litter_pro = litter_pro + litter[point, :, donor_pft, :, :] * frac
                fuel_1hr_pro = fuel_1hr_pro + fuel_1hr[point, donor_pft, :, :] * frac
                fuel_10hr_pro = fuel_10hr_pro + fuel_10hr[point, donor_pft, :, :] * frac
                fuel_100hr_pro = fuel_100hr_pro + fuel_100hr[point, donor_pft, :, :] * frac
                fuel_1000hr_pro = fuel_1000hr_pro + fuel_1000hr[point, donor_pft, :, :] * frac
                lignin_content_pro = (
                    lignin_content_pro
                    + litter[point, ISTRUCTURAL, donor_pft, :, ICARBON]
                    * lignin_struc[point, donor_pft, :]
                    * frac
                )

    legacy_scalars = collect_legacy_scalar_pools_step(
        glcc_frac=glcc_frac,
        bm_to_litter=bm_to_litter,
        carbon=carbon,
        deepC_a=deepC_a,
        deepC_s=deepC_s,
        deepC_p=deepC_p,
        co2_to_bm=co2_to_bm,
        gpp_daily=gpp_daily,
        npp_daily=npp_daily,
        resp_maint=resp_maint,
        resp_growth=resp_growth,
        resp_hetero=resp_hetero,
        co2_fire=co2_fire,
        litter_pro=litter_pro,
        lignin_content_pro=lignin_content_pro,
        point=point,
        bm_to_litter_pro_initial=bm_to_litter_pro,
        min_stomate=min_stomate,
    )

    return LccLegacyCollectionResult(
        veget_max_pro=legacy_scalars.veget_max_pro,
        carbon_pro=legacy_scalars.carbon_pro,
        lignin_struc_pro=legacy_scalars.lignin_struc_pro,
        litter_pro=litter_pro,
        deepC_a_pro=legacy_scalars.deepC_a_pro,
        deepC_s_pro=legacy_scalars.deepC_s_pro,
        deepC_p_pro=legacy_scalars.deepC_p_pro,
        fuel_1hr_pro=fuel_1hr_pro,
        fuel_10hr_pro=fuel_10hr_pro,
        fuel_100hr_pro=fuel_100hr_pro,
        fuel_1000hr_pro=fuel_1000hr_pro,
        bm_to_litter_pro=legacy_scalars.bm_to_litter_pro,
        co2_to_bm_pro=legacy_scalars.co2_to_bm_pro,
        gpp_daily_pro=legacy_scalars.gpp_daily_pro,
        npp_daily_pro=legacy_scalars.npp_daily_pro,
        resp_maint_pro=legacy_scalars.resp_maint_pro,
        resp_growth_pro=legacy_scalars.resp_growth_pro,
        resp_hetero_pro=legacy_scalars.resp_hetero_pro,
        co2_fire_pro=legacy_scalars.co2_fire_pro,
        convflux=convflux,
        prod10=prod10,
        prod100=prod100,
    )


def lcc_subtract_outgoing_fractions_step(
    *,
    veget_max,
    glcc_pftmtc,
    point,
    receiver_mtc,
    min_stomate=1.0e-8,
) -> LccOutgoingFractionResult:
    """Subtract outgoing source fractions for one receiving MTC.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``gross_glcchange_fh``, lines 1604-1626. The returned ``exhausted`` mask
    identifies PFTs for which Fortran immediately calls ``empty_pft``.
    """

    veget_max = jnp.asarray(veget_max)
    glcc_pftmtc = jnp.asarray(glcc_pftmtc)
    point = int(point)
    receiver_mtc = int(receiver_mtc)
    min_stomate = jnp.asarray(min_stomate, dtype=veget_max.dtype)

    if veget_max.ndim != 2:
        raise ValueError("veget_max must have shape (npts, nvm)")
    if glcc_pftmtc.ndim != 3 or glcc_pftmtc.shape[:2] != veget_max.shape:
        raise ValueError("glcc_pftmtc must have shape (npts, nvm, nvmap)")

    outgoing = glcc_pftmtc[point, :, receiver_mtc]
    active_loss = outgoing > min_stomate
    updated_row = jnp.where(active_loss, veget_max[point, :] - outgoing, veget_max[point, :])
    new_veget_max = veget_max.at[point, :].set(updated_row)
    exhausted = active_loss & (updated_row < min_stomate)

    return LccOutgoingFractionResult(
        veget_max=new_veget_max,
        exhausted=exhausted,
    )


def lcc_apply_collected_proxy_to_target_step(
    *,
    point,
    receiver_mtc,
    target_pft,
    glcc_pftmtc,
    age_class_pfts,
    veget_max_pro,
    carbon_pro,
    litter_pro,
    lignin_struc_pro,
    bm_to_litter_pro,
    deepC_a_pro,
    deepC_s_pro,
    deepC_p_pro,
    fuel_1hr_pro,
    fuel_10hr_pro,
    fuel_100hr_pro,
    fuel_1000hr_pro,
    co2_to_bm_pro,
    gpp_daily_pro,
    npp_daily_pro,
    resp_maint_pro,
    resp_growth_pro,
    resp_hetero_pro,
    co2_fire_pro,
    bm_sapl,
    cn_sapl,
    is_tree,
    npp_longterm_init,
    veget_max,
    carbon,
    litter,
    lignin_struc,
    bm_to_litter,
    deepC_a,
    deepC_s,
    deepC_p,
    fuel_1hr,
    fuel_10hr,
    fuel_100hr,
    fuel_1000hr,
    biomass,
    co2_to_bm,
    npp_longterm,
    ind,
    lm_lastyearmax,
    age,
    everywhere,
    leaf_frac,
    leaf_age,
    pft_present,
    senescence,
    gpp_daily,
    npp_daily,
    resp_maint,
    resp_growth,
    resp_hetero,
    co2_fire,
    min_stomate=1.0e-8,
) -> LccApplyCollectedProxyResult:
    """Apply one collected incoming proxy PFT to its youngest target PFT.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``gross_glcchange_fh``, lines 1604-1658, with called subroutines
    ``initialize_proxy_pft`` lines 1032-1108, ``sap_take`` lines 1118-1159,
    and ``add_incoming_proxy_pft`` lines 1751-2033. Source-PFT clearing is
    reported as a mask and remains delegated to ``empty_pft_lcc_step``.
    """

    source_subtract = lcc_subtract_outgoing_fractions_step(
        veget_max=veget_max,
        glcc_pftmtc=glcc_pftmtc,
        point=point,
        receiver_mtc=receiver_mtc,
        min_stomate=min_stomate,
    )
    proxy_init = initialize_proxy_pft_step(
        pft=target_pft,
        veget_max_pro=veget_max_pro,
        co2_to_bm_pro=co2_to_bm_pro,
        bm_sapl=bm_sapl,
        cn_sapl=cn_sapl,
        is_tree=is_tree,
        npp_longterm_init=npp_longterm_init,
    )
    sap_taken = sap_take_step(
        biomass=biomass,
        veget_max=source_subtract.veget_max,
        biomass_pro=proxy_init.biomass_pro,
        co2_to_bm_pro=proxy_init.co2_to_bm_pro,
        point=point,
        age_class_pfts=age_class_pfts,
        min_stomate=min_stomate,
    )
    merged = add_incoming_proxy_pft_step(
        point=point,
        pft=target_pft,
        veget_max_pro=veget_max_pro,
        carbon_pro=carbon_pro,
        litter_pro=litter_pro,
        lignin_struc_pro=lignin_struc_pro,
        bm_to_litter_pro=bm_to_litter_pro,
        deepC_a_pro=deepC_a_pro,
        deepC_s_pro=deepC_s_pro,
        deepC_p_pro=deepC_p_pro,
        fuel_1hr_pro=fuel_1hr_pro,
        fuel_10hr_pro=fuel_10hr_pro,
        fuel_100hr_pro=fuel_100hr_pro,
        fuel_1000hr_pro=fuel_1000hr_pro,
        biomass_pro=proxy_init.biomass_pro,
        co2_to_bm_pro=sap_taken.co2_to_bm_pro,
        npp_longterm_pro=proxy_init.npp_longterm_pro,
        ind_pro=proxy_init.ind_pro,
        lm_lastyearmax_pro=proxy_init.lm_lastyearmax_pro,
        age_pro=proxy_init.age_pro,
        everywhere_pro=proxy_init.everywhere_pro,
        leaf_frac_pro=proxy_init.leaf_frac_pro,
        leaf_age_pro=proxy_init.leaf_age_pro,
        pft_present_pro=proxy_init.pft_present_pro,
        senescence_pro=proxy_init.senescence_pro,
        gpp_daily_pro=gpp_daily_pro,
        npp_daily_pro=npp_daily_pro,
        resp_maint_pro=resp_maint_pro,
        resp_growth_pro=resp_growth_pro,
        resp_hetero_pro=resp_hetero_pro,
        co2_fire_pro=co2_fire_pro,
        veget_max=source_subtract.veget_max,
        carbon=carbon,
        litter=litter,
        lignin_struc=lignin_struc,
        bm_to_litter=bm_to_litter,
        deepC_a=deepC_a,
        deepC_s=deepC_s,
        deepC_p=deepC_p,
        fuel_1hr=fuel_1hr,
        fuel_10hr=fuel_10hr,
        fuel_100hr=fuel_100hr,
        fuel_1000hr=fuel_1000hr,
        biomass=sap_taken.biomass,
        co2_to_bm=co2_to_bm,
        npp_longterm=npp_longterm,
        ind=ind,
        lm_lastyearmax=lm_lastyearmax,
        age=age,
        everywhere=everywhere,
        leaf_frac=leaf_frac,
        leaf_age=leaf_age,
        pft_present=pft_present,
        senescence=senescence,
        gpp_daily=gpp_daily,
        npp_daily=npp_daily,
        resp_maint=resp_maint,
        resp_growth=resp_growth,
        resp_hetero=resp_hetero,
        co2_fire=co2_fire,
        min_stomate=min_stomate,
    )

    return LccApplyCollectedProxyResult(
        state=merged,
        exhausted_sources=source_subtract.exhausted,
        veget_max_after_source_subtract=source_subtract.veget_max,
        biomass_pro=proxy_init.biomass_pro,
        co2_to_bm_pro_after_sap_take=sap_taken.co2_to_bm_pro,
    )


def lcc_receiver_mtc_step(
    *,
    point,
    receiver_mtc,
    start_index,
    nagec_pft,
    glcc_pftmtc,
    biomass,
    bm_to_litter,
    carbon,
    litter,
    deepC_a,
    deepC_s,
    deepC_p,
    fuel_1hr,
    fuel_10hr,
    fuel_100hr,
    fuel_1000hr,
    lignin_struc,
    co2_to_bm,
    gpp_daily,
    npp_daily,
    resp_maint,
    resp_growth,
    resp_hetero,
    co2_fire,
    convflux,
    prod10,
    prod100,
    is_tree,
    coeff_lcchange_1,
    coeff_lcchange_10,
    coeff_lcchange_100,
    bm_sapl,
    cn_sapl,
    npp_longterm_init,
    veget_max,
    npp_longterm,
    ind,
    lm_lastyearmax,
    age,
    everywhere,
    leaf_frac,
    leaf_age,
    pft_present,
    senescence,
    gpp_week,
    when_growthinit,
    gdd_from_growthinit,
    gdd_midwinter,
    time_hum_min,
    gdd_m5_dormance,
    ncd_dormance,
    moiavail_month,
    moiavail_week,
    ngd_minus5,
    allow_deforest_fire=False,
    deforest_biomass_remain=None,
    min_stomate=1.0e-8,
) -> LccReceiverMtcStepResult:
    """Apply one active receiver MTC in the gross-LCC sequence.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``, subroutine
    ``gross_glcchange_fh``, lines 1587-1658. This composes
    ``collect_legacy_pft``, source fraction subtraction, ``empty_pft``,
    ``initialize_proxy_pft``, ``sap_take``, and ``add_incoming_proxy_pft`` for
    one receiving meta-class.
    """

    point = int(point)
    receiver_mtc = int(receiver_mtc)
    start_index = tuple(int(v) for v in start_index)
    nagec_pft = tuple(int(v) for v in nagec_pft)
    if len(start_index) != len(nagec_pft):
        raise ValueError("start_index and nagec_pft must describe the same MTC axis")
    if not (0 <= receiver_mtc < len(start_index)):
        raise ValueError("receiver_mtc must address start_index")
    target_pft = start_index[receiver_mtc]
    age_class_pfts = tuple(range(target_pft, target_pft + nagec_pft[receiver_mtc]))

    collection = lcc_collect_legacy_pft_step(
        point=point,
        receiver_mtc=receiver_mtc,
        start_index=start_index,
        glcc_pftmtc=glcc_pftmtc,
        biomass=biomass,
        bm_to_litter=bm_to_litter,
        carbon=carbon,
        litter=litter,
        deepC_a=deepC_a,
        deepC_s=deepC_s,
        deepC_p=deepC_p,
        fuel_1hr=fuel_1hr,
        fuel_10hr=fuel_10hr,
        fuel_100hr=fuel_100hr,
        fuel_1000hr=fuel_1000hr,
        lignin_struc=lignin_struc,
        co2_to_bm=co2_to_bm,
        gpp_daily=gpp_daily,
        npp_daily=npp_daily,
        resp_maint=resp_maint,
        resp_growth=resp_growth,
        resp_hetero=resp_hetero,
        co2_fire=co2_fire,
        convflux=convflux,
        prod10=prod10,
        prod100=prod100,
        is_tree=is_tree,
        coeff_lcchange_1=coeff_lcchange_1,
        coeff_lcchange_10=coeff_lcchange_10,
        coeff_lcchange_100=coeff_lcchange_100,
        allow_deforest_fire=allow_deforest_fire,
        deforest_biomass_remain=deforest_biomass_remain,
        min_stomate=min_stomate,
    )

    source_subtract = lcc_subtract_outgoing_fractions_step(
        veget_max=veget_max,
        glcc_pftmtc=glcc_pftmtc,
        point=point,
        receiver_mtc=receiver_mtc,
        min_stomate=min_stomate,
    )

    veget_max_curr = source_subtract.veget_max
    biomass_curr = jnp.asarray(biomass)
    ind_curr = jnp.asarray(ind)
    carbon_curr = jnp.asarray(carbon)
    litter_curr = jnp.asarray(litter)
    lignin_struc_curr = jnp.asarray(lignin_struc)
    bm_to_litter_curr = jnp.asarray(bm_to_litter)
    deepC_a_curr = jnp.asarray(deepC_a)
    deepC_s_curr = jnp.asarray(deepC_s)
    deepC_p_curr = jnp.asarray(deepC_p)
    fuel_1hr_curr = jnp.asarray(fuel_1hr)
    fuel_10hr_curr = jnp.asarray(fuel_10hr)
    fuel_100hr_curr = jnp.asarray(fuel_100hr)
    fuel_1000hr_curr = jnp.asarray(fuel_1000hr)
    gpp_daily_curr = jnp.asarray(gpp_daily)
    npp_daily_curr = jnp.asarray(npp_daily)
    gpp_week_curr = jnp.asarray(gpp_week)
    npp_longterm_curr = jnp.asarray(npp_longterm)
    co2_to_bm_curr = jnp.asarray(co2_to_bm)
    resp_maint_curr = jnp.asarray(resp_maint)
    resp_growth_curr = jnp.asarray(resp_growth)
    resp_hetero_curr = jnp.asarray(resp_hetero)
    lm_lastyearmax_curr = jnp.asarray(lm_lastyearmax)
    leaf_frac_curr = jnp.asarray(leaf_frac)
    leaf_age_curr = jnp.asarray(leaf_age)
    age_curr = jnp.asarray(age)
    everywhere_curr = jnp.asarray(everywhere)
    pft_present_curr = jnp.asarray(pft_present)
    when_growthinit_curr = jnp.asarray(when_growthinit)
    senescence_curr = jnp.asarray(senescence)
    gdd_from_growthinit_curr = jnp.asarray(gdd_from_growthinit)
    gdd_midwinter_curr = jnp.asarray(gdd_midwinter)
    time_hum_min_curr = jnp.asarray(time_hum_min)
    gdd_m5_dormance_curr = jnp.asarray(gdd_m5_dormance)
    ncd_dormance_curr = jnp.asarray(ncd_dormance)
    moiavail_month_curr = jnp.asarray(moiavail_month)
    moiavail_week_curr = jnp.asarray(moiavail_week)
    ngd_minus5_curr = jnp.asarray(ngd_minus5)

    for donor_pft in range(jnp.asarray(glcc_pftmtc).shape[1]):
        if bool(source_subtract.exhausted[donor_pft]):
            emptied = empty_pft_lcc_step(
                point=point,
                pft=donor_pft,
                veget_max=veget_max_curr,
                biomass=biomass_curr,
                ind=ind_curr,
                carbon=carbon_curr,
                litter=litter_curr,
                lignin_struc=lignin_struc_curr,
                bm_to_litter=bm_to_litter_curr,
                deepC_a=deepC_a_curr,
                deepC_s=deepC_s_curr,
                deepC_p=deepC_p_curr,
                fuel_1hr=fuel_1hr_curr,
                fuel_10hr=fuel_10hr_curr,
                fuel_100hr=fuel_100hr_curr,
                fuel_1000hr=fuel_1000hr_curr,
                gpp_daily=gpp_daily_curr,
                npp_daily=npp_daily_curr,
                gpp_week=gpp_week_curr,
                npp_longterm=npp_longterm_curr,
                co2_to_bm=co2_to_bm_curr,
                resp_maint=resp_maint_curr,
                resp_growth=resp_growth_curr,
                resp_hetero=resp_hetero_curr,
                lm_lastyearmax=lm_lastyearmax_curr,
                leaf_frac=leaf_frac_curr,
                leaf_age=leaf_age_curr,
                age=age_curr,
                everywhere=everywhere_curr,
                pft_present=pft_present_curr,
                when_growthinit=when_growthinit_curr,
                senescence=senescence_curr,
                gdd_from_growthinit=gdd_from_growthinit_curr,
                gdd_midwinter=gdd_midwinter_curr,
                time_hum_min=time_hum_min_curr,
                gdd_m5_dormance=gdd_m5_dormance_curr,
                ncd_dormance=ncd_dormance_curr,
                moiavail_month=moiavail_month_curr,
                moiavail_week=moiavail_week_curr,
                ngd_minus5=ngd_minus5_curr,
            )
            veget_max_curr = emptied.veget_max
            biomass_curr = emptied.biomass
            ind_curr = emptied.ind
            carbon_curr = emptied.carbon
            litter_curr = emptied.litter
            lignin_struc_curr = emptied.lignin_struc
            bm_to_litter_curr = emptied.bm_to_litter
            deepC_a_curr = emptied.deepC_a
            deepC_s_curr = emptied.deepC_s
            deepC_p_curr = emptied.deepC_p
            fuel_1hr_curr = emptied.fuel_1hr
            fuel_10hr_curr = emptied.fuel_10hr
            fuel_100hr_curr = emptied.fuel_100hr
            fuel_1000hr_curr = emptied.fuel_1000hr
            gpp_daily_curr = emptied.gpp_daily
            npp_daily_curr = emptied.npp_daily
            gpp_week_curr = emptied.gpp_week
            npp_longterm_curr = emptied.npp_longterm
            co2_to_bm_curr = emptied.co2_to_bm
            resp_maint_curr = emptied.resp_maint
            resp_growth_curr = emptied.resp_growth
            resp_hetero_curr = emptied.resp_hetero
            lm_lastyearmax_curr = emptied.lm_lastyearmax
            leaf_frac_curr = emptied.leaf_frac
            leaf_age_curr = emptied.leaf_age
            age_curr = emptied.age
            everywhere_curr = emptied.everywhere
            pft_present_curr = emptied.pft_present
            when_growthinit_curr = emptied.when_growthinit
            senescence_curr = emptied.senescence
            gdd_from_growthinit_curr = emptied.gdd_from_growthinit
            gdd_midwinter_curr = emptied.gdd_midwinter
            time_hum_min_curr = emptied.time_hum_min
            gdd_m5_dormance_curr = emptied.gdd_m5_dormance
            ncd_dormance_curr = emptied.ncd_dormance
            moiavail_month_curr = emptied.moiavail_month
            moiavail_week_curr = emptied.moiavail_week
            ngd_minus5_curr = emptied.ngd_minus5

    proxy_init = initialize_proxy_pft_step(
        pft=target_pft,
        veget_max_pro=collection.veget_max_pro,
        co2_to_bm_pro=collection.co2_to_bm_pro,
        bm_sapl=bm_sapl,
        cn_sapl=cn_sapl,
        is_tree=is_tree,
        npp_longterm_init=npp_longterm_init,
    )
    sap_taken = sap_take_step(
        biomass=biomass_curr,
        veget_max=veget_max_curr,
        biomass_pro=proxy_init.biomass_pro,
        co2_to_bm_pro=proxy_init.co2_to_bm_pro,
        point=point,
        age_class_pfts=age_class_pfts,
        min_stomate=min_stomate,
    )
    merged = add_incoming_proxy_pft_step(
        point=point,
        pft=target_pft,
        veget_max_pro=collection.veget_max_pro,
        carbon_pro=collection.carbon_pro,
        litter_pro=collection.litter_pro,
        lignin_struc_pro=collection.lignin_struc_pro,
        bm_to_litter_pro=collection.bm_to_litter_pro,
        deepC_a_pro=collection.deepC_a_pro,
        deepC_s_pro=collection.deepC_s_pro,
        deepC_p_pro=collection.deepC_p_pro,
        fuel_1hr_pro=collection.fuel_1hr_pro,
        fuel_10hr_pro=collection.fuel_10hr_pro,
        fuel_100hr_pro=collection.fuel_100hr_pro,
        fuel_1000hr_pro=collection.fuel_1000hr_pro,
        biomass_pro=proxy_init.biomass_pro,
        co2_to_bm_pro=sap_taken.co2_to_bm_pro,
        npp_longterm_pro=proxy_init.npp_longterm_pro,
        ind_pro=proxy_init.ind_pro,
        lm_lastyearmax_pro=proxy_init.lm_lastyearmax_pro,
        age_pro=proxy_init.age_pro,
        everywhere_pro=proxy_init.everywhere_pro,
        leaf_frac_pro=proxy_init.leaf_frac_pro,
        leaf_age_pro=proxy_init.leaf_age_pro,
        pft_present_pro=proxy_init.pft_present_pro,
        senescence_pro=proxy_init.senescence_pro,
        gpp_daily_pro=collection.gpp_daily_pro,
        npp_daily_pro=collection.npp_daily_pro,
        resp_maint_pro=collection.resp_maint_pro,
        resp_growth_pro=collection.resp_growth_pro,
        resp_hetero_pro=collection.resp_hetero_pro,
        co2_fire_pro=collection.co2_fire_pro,
        veget_max=veget_max_curr,
        carbon=carbon_curr,
        litter=litter_curr,
        lignin_struc=lignin_struc_curr,
        bm_to_litter=bm_to_litter_curr,
        deepC_a=deepC_a_curr,
        deepC_s=deepC_s_curr,
        deepC_p=deepC_p_curr,
        fuel_1hr=fuel_1hr_curr,
        fuel_10hr=fuel_10hr_curr,
        fuel_100hr=fuel_100hr_curr,
        fuel_1000hr=fuel_1000hr_curr,
        biomass=sap_taken.biomass,
        co2_to_bm=co2_to_bm_curr,
        npp_longterm=npp_longterm_curr,
        ind=ind_curr,
        lm_lastyearmax=lm_lastyearmax_curr,
        age=age_curr,
        everywhere=everywhere_curr,
        leaf_frac=leaf_frac_curr,
        leaf_age=leaf_age_curr,
        pft_present=pft_present_curr,
        senescence=senescence_curr,
        gpp_daily=gpp_daily_curr,
        npp_daily=npp_daily_curr,
        resp_maint=resp_maint_curr,
        resp_growth=resp_growth_curr,
        resp_hetero=resp_hetero_curr,
        co2_fire=co2_fire,
        min_stomate=min_stomate,
    )

    return LccReceiverMtcStepResult(
        state=merged,
        convflux=collection.convflux,
        prod10=collection.prod10,
        prod100=collection.prod100,
        exhausted_sources=source_subtract.exhausted,
        collection=collection,
        biomass_pro=proxy_init.biomass_pro,
        co2_to_bm_pro_after_sap_take=sap_taken.co2_to_bm_pro,
        gpp_week=gpp_week_curr,
        when_growthinit=when_growthinit_curr,
        gdd_from_growthinit=gdd_from_growthinit_curr,
        gdd_midwinter=gdd_midwinter_curr,
        time_hum_min=time_hum_min_curr,
        gdd_m5_dormance=gdd_m5_dormance_curr,
        ncd_dormance=ncd_dormance_curr,
        moiavail_month=moiavail_month_curr,
        moiavail_week=moiavail_week_curr,
        ngd_minus5=ngd_minus5_curr,
    )


def lcc_apply_all_receiver_mtcs_step(
    *,
    glcc_pftmtc,
    start_index,
    nagec_pft,
    biomass,
    bm_to_litter,
    carbon,
    litter,
    deepC_a,
    deepC_s,
    deepC_p,
    fuel_1hr,
    fuel_10hr,
    fuel_100hr,
    fuel_1000hr,
    lignin_struc,
    co2_to_bm,
    gpp_daily,
    npp_daily,
    resp_maint,
    resp_growth,
    resp_hetero,
    co2_fire,
    convflux,
    prod10,
    prod100,
    flux10,
    flux100,
    cflux_prod10,
    cflux_prod100,
    is_tree,
    coeff_lcchange_1,
    coeff_lcchange_10,
    coeff_lcchange_100,
    bm_sapl,
    cn_sapl,
    npp_longterm_init,
    veget_max,
    npp_longterm,
    ind,
    lm_lastyearmax,
    age,
    everywhere,
    leaf_frac,
    leaf_age,
    pft_present,
    senescence,
    gpp_week,
    when_growthinit,
    gdd_from_growthinit,
    gdd_midwinter,
    time_hum_min,
    gdd_m5_dormance,
    ncd_dormance,
    moiavail_month,
    moiavail_week,
    ngd_minus5,
    dt_days,
    one_year=ONE_YEAR_DAYS,
    allow_deforest_fire=False,
    deforest_biomass_remain=None,
    min_stomate=1.0e-8,
) -> LccAllReceiversApplyResult:
    """Apply all active receiver MTCs and age product pools.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``,
    ``gross_glcchange_fh`` lines 1571-1658 for the receiver loop and
    lines 1669-1697 for product-pool aging. The caller must provide
    ``glcc_pftmtc`` from the source-backed first-day allocation helper.
    """

    glcc_pftmtc = jnp.asarray(glcc_pftmtc)
    if glcc_pftmtc.ndim != 3:
        raise ValueError("glcc_pftmtc must have shape (npts, nvm, nvmap)")
    npts, _, nvmap = glcc_pftmtc.shape

    veget_max_curr = jnp.asarray(veget_max)
    biomass_curr = jnp.asarray(biomass)
    bm_to_litter_curr = jnp.asarray(bm_to_litter)
    carbon_curr = jnp.asarray(carbon)
    litter_curr = jnp.asarray(litter)
    deepC_a_curr = jnp.asarray(deepC_a)
    deepC_s_curr = jnp.asarray(deepC_s)
    deepC_p_curr = jnp.asarray(deepC_p)
    fuel_1hr_curr = jnp.asarray(fuel_1hr)
    fuel_10hr_curr = jnp.asarray(fuel_10hr)
    fuel_100hr_curr = jnp.asarray(fuel_100hr)
    fuel_1000hr_curr = jnp.asarray(fuel_1000hr)
    lignin_struc_curr = jnp.asarray(lignin_struc)
    co2_to_bm_curr = jnp.asarray(co2_to_bm)
    gpp_daily_curr = jnp.asarray(gpp_daily)
    npp_daily_curr = jnp.asarray(npp_daily)
    resp_maint_curr = jnp.asarray(resp_maint)
    resp_growth_curr = jnp.asarray(resp_growth)
    resp_hetero_curr = jnp.asarray(resp_hetero)
    co2_fire_curr = jnp.asarray(co2_fire)
    convflux_curr = jnp.zeros_like(jnp.asarray(convflux))
    prod10_curr = jnp.asarray(prod10).at[:, 0, :].set(0.0)
    prod100_curr = jnp.asarray(prod100).at[:, 0, :].set(0.0)
    npp_longterm_curr = jnp.asarray(npp_longterm)
    ind_curr = jnp.asarray(ind)
    lm_lastyearmax_curr = jnp.asarray(lm_lastyearmax)
    age_curr = jnp.asarray(age)
    everywhere_curr = jnp.asarray(everywhere)
    leaf_frac_curr = jnp.asarray(leaf_frac)
    leaf_age_curr = jnp.asarray(leaf_age)
    pft_present_curr = jnp.asarray(pft_present)
    senescence_curr = jnp.asarray(senescence)
    gpp_week_curr = jnp.asarray(gpp_week)
    when_growthinit_curr = jnp.asarray(when_growthinit)
    gdd_from_growthinit_curr = jnp.asarray(gdd_from_growthinit)
    gdd_midwinter_curr = jnp.asarray(gdd_midwinter)
    time_hum_min_curr = jnp.asarray(time_hum_min)
    gdd_m5_dormance_curr = jnp.asarray(gdd_m5_dormance)
    ncd_dormance_curr = jnp.asarray(ncd_dormance)
    moiavail_month_curr = jnp.asarray(moiavail_month)
    moiavail_week_curr = jnp.asarray(moiavail_week)
    ngd_minus5_curr = jnp.asarray(ngd_minus5)

    last_state = IncomingProxyPftResult(
        veget_max=veget_max_curr,
        carbon=carbon_curr,
        litter=litter_curr,
        lignin_struc=lignin_struc_curr,
        bm_to_litter=bm_to_litter_curr,
        deepC_a=deepC_a_curr,
        deepC_s=deepC_s_curr,
        deepC_p=deepC_p_curr,
        fuel_1hr=fuel_1hr_curr,
        fuel_10hr=fuel_10hr_curr,
        fuel_100hr=fuel_100hr_curr,
        fuel_1000hr=fuel_1000hr_curr,
        biomass=biomass_curr,
        co2_to_bm=co2_to_bm_curr,
        npp_longterm=npp_longterm_curr,
        ind=ind_curr,
        lm_lastyearmax=lm_lastyearmax_curr,
        age=age_curr,
        everywhere=everywhere_curr,
        leaf_frac=leaf_frac_curr,
        leaf_age=leaf_age_curr,
        pft_present=pft_present_curr,
        senescence=senescence_curr,
        gpp_daily=gpp_daily_curr,
        npp_daily=npp_daily_curr,
        resp_maint=resp_maint_curr,
        resp_growth=resp_growth_curr,
        resp_hetero=resp_hetero_curr,
        co2_fire=co2_fire_curr,
    )

    for point in range(npts):
        for receiver_mtc in range(1, nvmap):
            if bool(jnp.sum(glcc_pftmtc[point, :, receiver_mtc]) > min_stomate):
                step = lcc_receiver_mtc_step(
                    point=point,
                    receiver_mtc=receiver_mtc,
                    start_index=start_index,
                    nagec_pft=nagec_pft,
                    glcc_pftmtc=glcc_pftmtc,
                    biomass=last_state.biomass,
                    bm_to_litter=last_state.bm_to_litter,
                    carbon=last_state.carbon,
                    litter=last_state.litter,
                    deepC_a=last_state.deepC_a,
                    deepC_s=last_state.deepC_s,
                    deepC_p=last_state.deepC_p,
                    fuel_1hr=last_state.fuel_1hr,
                    fuel_10hr=last_state.fuel_10hr,
                    fuel_100hr=last_state.fuel_100hr,
                    fuel_1000hr=last_state.fuel_1000hr,
                    lignin_struc=last_state.lignin_struc,
                    co2_to_bm=last_state.co2_to_bm,
                    gpp_daily=last_state.gpp_daily,
                    npp_daily=last_state.npp_daily,
                    resp_maint=last_state.resp_maint,
                    resp_growth=last_state.resp_growth,
                    resp_hetero=last_state.resp_hetero,
                    co2_fire=last_state.co2_fire,
                    convflux=convflux_curr,
                    prod10=prod10_curr,
                    prod100=prod100_curr,
                    is_tree=is_tree,
                    coeff_lcchange_1=coeff_lcchange_1,
                    coeff_lcchange_10=coeff_lcchange_10,
                    coeff_lcchange_100=coeff_lcchange_100,
                    bm_sapl=bm_sapl,
                    cn_sapl=cn_sapl,
                    npp_longterm_init=npp_longterm_init,
                    veget_max=last_state.veget_max,
                    npp_longterm=last_state.npp_longterm,
                    ind=last_state.ind,
                    lm_lastyearmax=last_state.lm_lastyearmax,
                    age=last_state.age,
                    everywhere=last_state.everywhere,
                    leaf_frac=last_state.leaf_frac,
                    leaf_age=last_state.leaf_age,
                    pft_present=last_state.pft_present,
                    senescence=last_state.senescence,
                    gpp_week=gpp_week_curr,
                    when_growthinit=when_growthinit_curr,
                    gdd_from_growthinit=gdd_from_growthinit_curr,
                    gdd_midwinter=gdd_midwinter_curr,
                    time_hum_min=time_hum_min_curr,
                    gdd_m5_dormance=gdd_m5_dormance_curr,
                    ncd_dormance=ncd_dormance_curr,
                    moiavail_month=moiavail_month_curr,
                    moiavail_week=moiavail_week_curr,
                    ngd_minus5=ngd_minus5_curr,
                    allow_deforest_fire=allow_deforest_fire,
                    deforest_biomass_remain=deforest_biomass_remain,
                    min_stomate=min_stomate,
                )
                last_state = step.state
                convflux_curr = step.convflux
                prod10_curr = step.prod10
                prod100_curr = step.prod100
                gpp_week_curr = step.gpp_week
                when_growthinit_curr = step.when_growthinit
                gdd_from_growthinit_curr = step.gdd_from_growthinit
                gdd_midwinter_curr = step.gdd_midwinter
                time_hum_min_curr = step.time_hum_min
                gdd_m5_dormance_curr = step.gdd_m5_dormance
                ncd_dormance_curr = step.ncd_dormance
                moiavail_month_curr = step.moiavail_month
                moiavail_week_curr = step.moiavail_week
                ngd_minus5_curr = step.ngd_minus5

    aged = product_pool_aging_step(
        prod10=prod10_curr,
        prod100=prod100_curr,
        flux10=flux10,
        flux100=flux100,
        convflux=convflux_curr,
        cflux_prod10=jnp.zeros_like(jnp.asarray(cflux_prod10)),
        cflux_prod100=jnp.zeros_like(jnp.asarray(cflux_prod100)),
        dt_days=dt_days,
        one_year=one_year,
    )

    return LccAllReceiversApplyResult(
        state=last_state,
        prod10=aged.prod10,
        prod100=aged.prod100,
        flux10=aged.flux10,
        flux100=aged.flux100,
        convflux=aged.convflux,
        cflux_prod10=aged.cflux_prod10,
        cflux_prod100=aged.cflux_prod100,
        gpp_week=gpp_week_curr,
        when_growthinit=when_growthinit_curr,
        gdd_from_growthinit=gdd_from_growthinit_curr,
        gdd_midwinter=gdd_midwinter_curr,
        time_hum_min=time_hum_min_curr,
        gdd_m5_dormance=gdd_m5_dormance_curr,
        ncd_dormance=ncd_dormance_curr,
        moiavail_month=moiavail_month_curr,
        moiavail_week=moiavail_week_curr,
        ngd_minus5=ngd_minus5_curr,
    )


def lcc_gross_glcchange_step(
    *,
    veget_max_org,
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
    biomass,
    bm_to_litter,
    carbon,
    litter,
    deepC_a,
    deepC_s,
    deepC_p,
    fuel_1hr,
    fuel_10hr,
    fuel_100hr,
    fuel_1000hr,
    lignin_struc,
    co2_to_bm,
    gpp_daily,
    npp_daily,
    resp_maint,
    resp_growth,
    resp_hetero,
    co2_fire,
    convflux,
    prod10,
    prod100,
    flux10,
    flux100,
    cflux_prod10,
    cflux_prod100,
    coeff_lcchange_1,
    coeff_lcchange_10,
    coeff_lcchange_100,
    bm_sapl,
    cn_sapl,
    npp_longterm_init,
    npp_longterm,
    ind,
    lm_lastyearmax,
    age,
    everywhere,
    leaf_frac,
    leaf_age,
    pft_present,
    senescence,
    gpp_week,
    when_growthinit,
    gdd_from_growthinit,
    gdd_midwinter,
    time_hum_min,
    gdd_m5_dormance,
    ncd_dormance,
    moiavail_month,
    moiavail_week,
    ngd_minus5,
    dt_days,
    one_year=ONE_YEAR_DAYS,
    nagec_tree=1,
    nagec_herb=1,
    single_age_class=False,
    allow_deforest_fire=False,
    deforest_biomass_remain=None,
    min_stomate=1.0e-8,
) -> LccGrossGlcchangeStepResult:
    """Compose gross-LCC first-day allocation with state application.

    Fortran provenance: ``src_stomate/stomate_glcchange_fh.f90``,
    ``gross_glcchange_fh`` lines 1565-1697, with
    ``gross_glcc_firstday_fh`` lines 2193-3059 providing the
    ``glcc_pftmtc`` allocation consumed by the receiver loop. When
    ``allow_deforest_fire`` is true, ``stomate_lpj.f90`` lines 835-859 build
    the deforestation proxy from tree-to-non-tree ``glcc_pftmtc`` before this
    call; this helper reproduces that boundary if the caller did not already
    supply ``deforest_biomass_remain``.
    """

    allocation_fn = (
        lcc_gross_firstday_single_age_allocation_step
        if bool(single_age_class)
        else lcc_gross_firstday_allocation_step
    )
    allocation = allocation_fn(
        veget_max_org=veget_max_org,
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
        nagec_tree=nagec_tree,
        nagec_herb=nagec_herb,
        min_stomate=min_stomate,
    )
    if bool(allow_deforest_fire) and deforest_biomass_remain is None:
        glcc_pftmtc = jnp.asarray(allocation.glcc_pftmtc)
        biomass_arr = jnp.asarray(biomass)
        is_tree_arr = jnp.asarray(is_tree, dtype=bool)
        start_index_arr = tuple(int(v) for v in start_index)
        lcc = jnp.zeros(biomass_arr.shape[:2], dtype=biomass_arr.dtype)
        for receiver_mtc, target_pft in enumerate(start_index_arr):
            tree_to_not_tree = is_tree_arr & (~bool(is_tree_arr[target_pft]))
            lcc = lcc + jnp.where(tree_to_not_tree[None, :], glcc_pftmtc[:, :, receiver_mtc], 0.0)
        deforest_biomass_remain = biomass_arr * lcc[:, :, None, None]
    applied = lcc_apply_all_receiver_mtcs_step(
        glcc_pftmtc=allocation.glcc_pftmtc,
        start_index=start_index,
        nagec_pft=nagec_pft,
        biomass=biomass,
        bm_to_litter=bm_to_litter,
        carbon=carbon,
        litter=litter,
        deepC_a=deepC_a,
        deepC_s=deepC_s,
        deepC_p=deepC_p,
        fuel_1hr=fuel_1hr,
        fuel_10hr=fuel_10hr,
        fuel_100hr=fuel_100hr,
        fuel_1000hr=fuel_1000hr,
        lignin_struc=lignin_struc,
        co2_to_bm=co2_to_bm,
        gpp_daily=gpp_daily,
        npp_daily=npp_daily,
        resp_maint=resp_maint,
        resp_growth=resp_growth,
        resp_hetero=resp_hetero,
        co2_fire=co2_fire,
        convflux=convflux,
        prod10=prod10,
        prod100=prod100,
        flux10=flux10,
        flux100=flux100,
        cflux_prod10=cflux_prod10,
        cflux_prod100=cflux_prod100,
        is_tree=is_tree,
        coeff_lcchange_1=coeff_lcchange_1,
        coeff_lcchange_10=coeff_lcchange_10,
        coeff_lcchange_100=coeff_lcchange_100,
        bm_sapl=bm_sapl,
        cn_sapl=cn_sapl,
        npp_longterm_init=npp_longterm_init,
        veget_max=veget_max_org,
        npp_longterm=npp_longterm,
        ind=ind,
        lm_lastyearmax=lm_lastyearmax,
        age=age,
        everywhere=everywhere,
        leaf_frac=leaf_frac,
        leaf_age=leaf_age,
        pft_present=pft_present,
        senescence=senescence,
        gpp_week=gpp_week,
        when_growthinit=when_growthinit,
        gdd_from_growthinit=gdd_from_growthinit,
        gdd_midwinter=gdd_midwinter,
        time_hum_min=time_hum_min,
        gdd_m5_dormance=gdd_m5_dormance,
        ncd_dormance=ncd_dormance,
        moiavail_month=moiavail_month,
        moiavail_week=moiavail_week,
        ngd_minus5=ngd_minus5,
        dt_days=dt_days,
        one_year=one_year,
        allow_deforest_fire=allow_deforest_fire,
        deforest_biomass_remain=deforest_biomass_remain,
        min_stomate=min_stomate,
    )

    return LccGrossGlcchangeStepResult(
        allocation=allocation,
        applied=applied,
    )


def stomate_lpj_output_diagnostics(
    *,
    biomass,
    turnover_daily,
    bm_to_litter,
    litter_above,
    litter_below,
    carbon_32l,
    doc,
    veget_max,
    z_soil,
    zf_soil,
    carb_mass_total_old,
    prod10_total=0.0,
    prod100_total=0.0,
    contfrac=1.0,
    one_day=86400.0,
) -> StomateLpjOutputDiagnostics:
    """Compute end-of-``StomateLpj`` diagnostic pools before history output.

    Fortran provenance: ``src_stomate/stomate_lpj.f90``, subroutine
    ``StomateLpj``, lines 1578-1663 and XIOS/modelout preparation lines
    1809-1880 and 1917-1918. This helper preserves the source loop structure,
    including the repeated product-pool addition in the ``carb_mass_total``
    loop.
    """

    biomass = jnp.asarray(biomass)
    turnover_daily = jnp.asarray(turnover_daily)
    bm_to_litter = jnp.asarray(bm_to_litter)
    litter_above = jnp.asarray(litter_above)
    litter_below = jnp.asarray(litter_below)
    carbon_32l = jnp.asarray(carbon_32l)
    doc = jnp.asarray(doc)
    veget_max = jnp.asarray(veget_max)
    z_soil = jnp.asarray(z_soil)
    zf_soil = jnp.asarray(zf_soil)
    carb_mass_total_old = jnp.asarray(carb_mass_total_old)
    prod10_total = jnp.asarray(prod10_total)
    prod100_total = jnp.asarray(prod100_total)
    contfrac = jnp.asarray(contfrac)
    one_day = jnp.asarray(one_day)

    if biomass.ndim != 4:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    npts, nvm = biomass.shape[:2]
    ndeep = carbon_32l.shape[3]
    if veget_max.shape != (npts, nvm):
        raise ValueError("veget_max must have shape (npts, nvm)")
    nslm = z_soil.shape[0]
    if nslm == 0 or zf_soil.shape[0] < ndeep + 1:
        raise ValueError("z_soil must cover nslm layers and zf_soil must cover ndeep layer interfaces")

    pft_mask = (jnp.arange(nvm) > 0).astype(biomass.dtype)
    litter_below_sum = (
        litter_below[:, ISTRUCTURAL, :, :, ICARBON]
        + litter_below[:, IMETABOLIC, :, :, ICARBON]
    )
    litter_above_sum = (
        litter_above[:, ISTRUCTURAL, :, ICARBON]
        + litter_above[:, IMETABOLIC, :, ICARBON]
    )
    surface_layer = (jnp.arange(ndeep) == 0).astype(biomass.dtype)
    tot_litter_carb = (
        litter_below_sum
        + litter_above_sum[:, :, None] * surface_layer[None, None, :]
    ) * pft_mask[None, :, None]
    tot_soil_carb = (
        carbon_32l[:, IACTIVE, :, :]
        + carbon_32l[:, ISLOW, :, :]
        + carbon_32l[:, IPASSIVE, :, :]
        + jnp.sum(doc[:, :, :, :, :, ICARBON], axis=(3, 4))
    ) * pft_mask[None, :, None]

    tot_litter_soil_carb = tot_litter_carb + tot_soil_carb
    parts = jnp.asarray(
        [
            ILEAF,
            ISAPABOVE,
            ISAPBELOW,
            IHEARTABOVE,
            IHEARTBELOW,
            IROOT,
            IFRUIT,
            ICARBRES,
            IAGRSAPST,
            IAGRSAPPN,
            IAGRHRTST,
            IAGRHRTPN,
        ]
    )
    tot_live_biomass = jnp.sum(biomass[:, :, parts, :], axis=2)
    tot_turnover = jnp.sum(turnover_daily[:, :, parts, :], axis=2)
    tot_bm_to_litter = jnp.sum(bm_to_litter[:, :, parts, :], axis=2)

    product_total = prod10_total + prod100_total
    z_ratio = z_soil[:nslm] / z_soil[nslm - 1]
    live_mass_term = tot_live_biomass[:, :, ICARBON] * veget_max
    litter_soil_mass_term = (
        (tot_litter_carb[:, :, :nslm] + tot_soil_carb[:, :, :nslm])
        * veget_max[:, :, None]
        * z_ratio[None, None, :]
    )
    carb_mass_total = (
        carb_mass_total_old
        + jnp.sum(live_mass_term[:, :, None] + litter_soil_mass_term, axis=(1, 2))
        + product_total * (nvm * nslm)
    )
    carb_mass_variation = carb_mass_total - carb_mass_total_old

    carbon_32l_pftmean = jnp.sum(carbon_32l * veget_max[:, None, :, None], axis=2)
    layer_thickness = zf_soil[1 : ndeep + 1] - zf_soil[:ndeep]
    carbon_32l_conct = carbon_32l_pftmean / layer_thickness[None, None, :]

    free_doc_stock_by_pft = jnp.sum(doc[:, :, :, 0, :, ICARBON], axis=3)
    adsorbed_doc_stock_by_pft = jnp.sum(doc[:, :, :, 1, :, ICARBON], axis=3)
    free_doc_stock = jnp.sum(free_doc_stock_by_pft * veget_max[:, :, None], axis=1)
    adsorbed_doc_stock = jnp.sum(adsorbed_doc_stock_by_pft * veget_max[:, :, None], axis=1)
    free_doc_thickness = layer_thickness.at[0].set(z_soil[0])
    free_doc = free_doc_stock / free_doc_thickness[None, :]
    adsorbed_doc = adsorbed_doc_stock / layer_thickness[None, :]

    f_veg_litter = jnp.sum((tot_bm_to_litter[:, :, ICARBON] + tot_turnover[:, :, ICARBON]) * veget_max, axis=1) / 1.0e3 / one_day * contfrac

    return StomateLpjOutputDiagnostics(
        tot_litter_carb=tot_litter_carb,
        tot_soil_carb=tot_soil_carb,
        tot_litter_soil_carb=tot_litter_soil_carb,
        tot_live_biomass=tot_live_biomass,
        tot_turnover=tot_turnover,
        tot_bm_to_litter=tot_bm_to_litter,
        carb_mass_total=carb_mass_total,
        carb_mass_variation=carb_mass_variation,
        carbon_32l_pftmean=carbon_32l_pftmean,
        carbon_32l_conct=carbon_32l_conct,
        free_doc=free_doc,
        free_doc_stock=free_doc_stock,
        adsorbed_doc=adsorbed_doc,
        f_veg_litter=f_veg_litter,
    )
