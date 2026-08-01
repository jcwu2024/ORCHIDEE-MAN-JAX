"""Source-closed helper kernels for STOMATE soilcarbon leak path."""

from __future__ import annotations

from typing import NamedTuple

from jax import config, core, jit

config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jax_orchidee.ad_primitives import source_sqrt_with_finite_zero_tangent

from jax_orchidee.stomate.carbon_kernels import (
    IACTIVE,
    IACT,
    IBELOW,
    ICARBON,
    ILEAF,
    IMETABO,
    IMETBEL,
    IROOT,
    IPASSIVE,
    IPAS,
    ISTRABO,
    ISTRBEL,
    ISLO,
    ISLOW,
    NCARB,
    NLITT,
    NPOOL,
    ZERO_CELSIUS,
)

IFREE = 0
IADSORBED = 1
NDOC = 2
# Fortran: src_parameters/constantes_var.f90 lines 268-272 (zero-based here).
IRUNOFF = 0
IFLOODED = 1
IDRAINAGE = 2
NEXP = 3
IH2O = 0
IDOCL = 1
IDOCR = 2
ICO2AQ = 3
NFLOW = 10
DIFFO2_AIR = 1.596e-5
DIFFO2_WATER = 1.596e-9
DIFFCH4_AIR = 1.702e-5
DIFFCH4_WATER = 2.0e-9
O2_SURF = 0.209
CH4_SURF = 1700.0e-9
BUNSEN_O2 = 0.038
BUNSEN_CH4 = 0.043
RHO_ICE = 920.0
AVM = 0.01
TETASAT = 0.5
RR_GAS = 8.314
HMIN_TCALC = 0.001
EBUTHR = 0.9
MIN_STOMATE = 1.0e-8


class AltCalcDocResult(NamedTuple):
    """Active-layer state from ``altcalc_DOC``."""

    alt: jnp.ndarray
    alt_ind: jnp.ndarray
    altmax: jnp.ndarray
    altmax_ind: jnp.ndarray
    altmax_lastyear: jnp.ndarray
    altmax_ind_lastyear: jnp.ndarray


class TfDocInputs(NamedTuple):
    """TF-DOC canopy/ground deposition terms."""

    dry_dep_canopy: jnp.ndarray
    doc_precip2ground: jnp.ndarray
    doc_precip2canopy: jnp.ndarray
    doc_canopy2ground: jnp.ndarray
    bio_frac: jnp.ndarray


class SoilcarbonDocExportAggregate(NamedTuple):
    """Aggregated DOC/DIC export produced after ``soilcarbon_leak``."""

    doc_exp_agg: jnp.ndarray
    doc_exp_b: jnp.ndarray
    soil_resp_modif: jnp.ndarray


class SoilcarbonActivityFactors(NamedTuple):
    """DOC and POC decomposition activity factors."""

    fbact_doc_labile: jnp.ndarray
    fbact_doc_refractory: jnp.ndarray
    fbact_npool: jnp.ndarray
    fbact_ncarb: jnp.ndarray


class TfDocGroundFluxes(NamedTuple):
    """TF-DOC ground/flood deposition and canopy storage after drip."""

    doc_canopy2ground: jnp.ndarray
    wet_dep_ground: jnp.ndarray
    wet_dep_flood: jnp.ndarray
    interception_storage: jnp.ndarray


class SoilcarbonDocInputResult(NamedTuple):
    """DOC pools after litter, routing, and wet-deposition inputs."""

    doc: jnp.ndarray
    wet_dep_ground: jnp.ndarray
    wet_dep_flood: jnp.ndarray
    interception_storage: jnp.ndarray


class SoilcarbonLomResult(NamedTuple):
    """Total litter and labile organic matter used by priming."""

    litter_tot: jnp.ndarray
    lom: jnp.ndarray


class SoilcarbonDecompositionResult(NamedTuple):
    """POC/DOC decomposition and pool updates through section 2.3.4."""

    carbon_32l: jnp.ndarray
    doc: jnp.ndarray
    resp_hetero_soil: jnp.ndarray
    resp_flood_soil: jnp.ndarray
    fluxtot: jnp.ndarray
    fluxtot_flood: jnp.ndarray
    fluxtot_doc: jnp.ndarray
    fluxtot_doc_flood: jnp.ndarray
    litter_tot: jnp.ndarray
    lom: jnp.ndarray


class SoilcarbonAdsorptionResult(NamedTuple):
    """DOC after free/adsorbed equilibrium plus distribution coefficients."""

    doc: jnp.ndarray
    kd: jnp.ndarray
    doc_re: jnp.ndarray


class SoilcarbonWaterTransportResult(NamedTuple):
    """DOC after water-flux transport plus layer DOC fluxes."""

    doc: jnp.ndarray
    doc_flux: jnp.ndarray


class SoilcarbonDiffusionResult(NamedTuple):
    """DOC after diffusion plus inter-layer diffusion fluxes."""

    doc: jnp.ndarray
    doc_flux_diff: jnp.ndarray


class SoilcarbonLeakExportResult(NamedTuple):
    """DOC export diagnostics and DOC pools after runoff/drain subtraction."""

    doc: jnp.ndarray
    doc_exp: jnp.ndarray
    doc_run: jnp.ndarray
    doc_drain: jnp.ndarray
    doc_flood: jnp.ndarray
    doc_run_2_peat: jnp.ndarray
    fastr_corr: jnp.ndarray
    soil_doc_corr: jnp.ndarray


class SoilcarbonLeakCoreResult(NamedTuple):
    """Composed source-order OK_LEAK soilcarbon core through DOC export."""

    carbon_32l: jnp.ndarray
    doc: jnp.ndarray
    litter_below: jnp.ndarray
    doc_exp: jnp.ndarray
    resp_hetero_soil: jnp.ndarray
    resp_flood_soil: jnp.ndarray
    fluxtot: jnp.ndarray
    fluxtot_flood: jnp.ndarray
    fluxtot_doc: jnp.ndarray
    fluxtot_doc_flood: jnp.ndarray
    litter_tot: jnp.ndarray
    lom: jnp.ndarray
    kd: jnp.ndarray
    doc_flux: jnp.ndarray
    doc_flux_diff: jnp.ndarray
    doc_run: jnp.ndarray
    doc_drain: jnp.ndarray
    doc_flood: jnp.ndarray
    fastr_corr: jnp.ndarray
    soil_doc_corr: jnp.ndarray
    cryoturbation_coefficients: SoilcarbonCryoturbationCoefficients | None
    perma_peat: SoilcarbonPermaPeatResult | None


class SoilcarbonResult(NamedTuple):
    """State produced by the legacy three-pool ``soilcarbon`` routine."""

    carbon: jnp.ndarray
    resp_hetero_soil: jnp.ndarray
    matrix_a: jnp.ndarray
    carbon_tau: jnp.ndarray
    firstcall_soilcarbon: bool
    height_acro: jnp.ndarray | None
    height_cato: jnp.ndarray | None
    carbon_acro: jnp.ndarray | None
    carbon_cato: jnp.ndarray | None
    tcarbon_acro: jnp.ndarray | None
    tcarbon_cato: jnp.ndarray | None
    resp_acro_oxic: jnp.ndarray | None
    resp_acro_anoxic: jnp.ndarray | None
    resp_cato: jnp.ndarray | None
    acro_to_cato: jnp.ndarray | None
    litter_to_acro: jnp.ndarray | None


class SoilcarbonCryoturbationCoefficients(NamedTuple):
    """Saved coefficients used by ``cryoturbate_doc_POC`` diffuse action."""

    cryoturb_location: jnp.ndarray
    bioturb_location: jnp.ndarray
    cryoturbation_depth: jnp.ndarray
    diff_k: jnp.ndarray
    xc_cryoturb: jnp.ndarray
    xd_cryoturb: jnp.ndarray
    alpha_c: jnp.ndarray
    beta_c: jnp.ndarray
    alpha_doc: jnp.ndarray
    beta_doc: jnp.ndarray
    alpha_litter_below: jnp.ndarray
    beta_litter_below: jnp.ndarray


class SoilcarbonCryoturbationDiffuseResult(NamedTuple):
    """Carbon, DOC, and belowground litter after cryoturbation diffusion."""

    carbon_32l: jnp.ndarray
    doc: jnp.ndarray
    litter_below: jnp.ndarray


class SoilcarbonPermaPeatResult(NamedTuple):
    """PERMA_PEAT carbon redistribution diagnostics and state."""

    carbon_32l: jnp.ndarray
    deepc_peat: jnp.ndarray
    deepc_pt: jnp.ndarray
    peat_olt: jnp.ndarray


class DeepCarbonVerticalIntegralResult(NamedTuple):
    """Vertically integrated OK_PC deep-carbon pools."""

    carbon: jnp.ndarray
    carbon_surf: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90::calc_vert_int_soil_carbon lines 4425-4463",
    )


class DeepCarbonInputResult(NamedTuple):
    """Deep-carbon pools and vertical litter-input density after ``carbinput``."""

    deepC_a: jnp.ndarray
    deepC_s: jnp.ndarray
    deepC_p: jnp.ndarray
    dc_litter_z: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90::carbinput lines 3184-3410",
    )


class DeepCarbonPermafrostDecompResult(NamedTuple):
    """Deep-carbon pools and gas/delta diagnostics after decomposition."""

    deepC_a: jnp.ndarray
    deepC_s: jnp.ndarray
    deepC_p: jnp.ndarray
    O2_soil: jnp.ndarray
    CH4_soil: jnp.ndarray | None
    deltaC1_a: jnp.ndarray
    deltaC1_s: jnp.ndarray
    deltaC1_p: jnp.ndarray
    deltaCH4: jnp.ndarray
    deltaCH4g: jnp.ndarray
    deltaC2: jnp.ndarray
    deltaC3: jnp.ndarray
    nadd_soil: jnp.ndarray
    perma_peat: SoilcarbonPermaPeatResult | None
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90::permafrost_decomp lines 3920-4403",
    )


class DeepCarbonNonMethaneCoreResult(NamedTuple):
    """Composed OK_PC deep-carbon non-methane core state."""

    deepC_a: jnp.ndarray
    deepC_s: jnp.ndarray
    deepC_p: jnp.ndarray
    carbon: jnp.ndarray
    carbon_surf: jnp.ndarray
    resp_hetero_soil: jnp.ndarray
    sfluxCH4: jnp.ndarray | None
    heat_Zimov: jnp.ndarray
    O2_soil: jnp.ndarray
    CH4_soil: jnp.ndarray | None
    dc_litter_z: jnp.ndarray
    decomposition: DeepCarbonPermafrostDecompResult
    cryoturbation_coefficients: DeepCarbonCryoturbationCoefficients | None
    Tref: jnp.ndarray | None = None
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90::deep_carbcycle lines 953-1077",
    )


class DeepCarbonYedomaResetResult(NamedTuple):
    """Deep-carbon pools after optional yedoma initialization reset."""

    deepC_a: jnp.ndarray
    deepC_s: jnp.ndarray
    deepC_p: jnp.ndarray
    yedoma: jnp.ndarray
    yedoma_depth_index: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90::initialize_yedoma_carbonstocks lines 3014-3165",
    )


class DeepCarbonRootDepthResult(NamedTuple):
    """Rooting-depth state bounded by the previous-year active layer."""

    z_root: jnp.ndarray
    rootlev: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90::deep_carbcycle lines 945-951",
    )


class DeepCarbonCryoturbationCoefficients(NamedTuple):
    """Saved OK_PC deep-carbon cryoturbation coefficients."""

    cryoturb_location: jnp.ndarray
    bioturb_location: jnp.ndarray
    cryoturbation_depth: jnp.ndarray
    diff_k: jnp.ndarray
    xc_cryoturb: jnp.ndarray
    xd_cryoturb: jnp.ndarray
    alpha_a: jnp.ndarray
    beta_a: jnp.ndarray
    alpha_s: jnp.ndarray
    beta_s: jnp.ndarray
    alpha_p: jnp.ndarray
    beta_p: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90::cryoturbate lines 3431-3916",
    )


class DeepCarbonCryoturbationDiffuseResult(NamedTuple):
    """Deep-carbon pools after OK_PC cryoturbation diffusion."""

    deepC_a: jnp.ndarray
    deepC_s: jnp.ndarray
    deepC_p: jnp.ndarray
    surfC_totake_a: jnp.ndarray
    surfC_totake_s: jnp.ndarray
    surfC_totake_p: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90::cryoturbate lines 3546-3628",
    )


class DeepCarbonSnowGeometryResult(NamedTuple):
    """Snow full/intermediate levels used by OK_PC gas diffusion."""

    zi_snow: jnp.ndarray
    zf_snow: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90::snowlevels lines 2708-2802",
    )


class DeepCarbonSnowInterpolResult(NamedTuple):
    """Snow O2/CH4 interpolated after a snow-grid update."""

    snowO2: jnp.ndarray
    snowCH4: jnp.ndarray
    zi_snow: jnp.ndarray
    zf_snow: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90::snow_interpol lines 2806-2953",
    )


class DeepCarbonGasDiffusionPropertiesResult(NamedTuple):
    """Gas porosity and diffusivity inputs for OK_PC methane/O2 diffusion."""

    airvol_snow: jnp.ndarray
    totporO2_snow: jnp.ndarray
    totporCH4_snow: jnp.ndarray
    diffO2_snow: jnp.ndarray
    diffCH4_snow: jnp.ndarray
    airvol_soil: jnp.ndarray
    totporO2_soil: jnp.ndarray
    totporCH4_soil: jnp.ndarray
    diffO2_soil: jnp.ndarray
    diffCH4_soil: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90::get_gasdiff lines 2098-2223",
    )


class DeepCarbonGasDiffusionCoefficients(NamedTuple):
    """Saved tridiagonal coefficients used by ``soil_gasdiff_diff``."""

    alphaO2_soil: jnp.ndarray
    betaO2_soil: jnp.ndarray
    alphaCH4_soil: jnp.ndarray
    betaCH4_soil: jnp.ndarray
    alphaO2_snow: jnp.ndarray
    betaO2_snow: jnp.ndarray
    alphaCH4_snow: jnp.ndarray
    betaCH4_snow: jnp.ndarray
    mu_soil: jnp.ndarray
    mu_snow: jnp.ndarray
    zi_coeff_snow: jnp.ndarray
    zf_coeff_snow: jnp.ndarray
    snow_height_mask: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90::soil_gasdiff_coeff lines 1663-1969",
    )


class DeepCarbonGasDiffusionStateResult(NamedTuple):
    """Snow/soil O2 and CH4 after applying saved gas-diffusion coefficients."""

    O2_snow: jnp.ndarray
    CH4_snow: jnp.ndarray
    O2_soil: jnp.ndarray
    CH4_soil: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90::soil_gasdiff_diff lines 1973-2094",
    )


class DeepCarbonPlantTransportResult(NamedTuple):
    """Soil O2/CH4 and flux after plant-mediated methane transport."""

    CH4_soil: jnp.ndarray
    O2_soil: jnp.ndarray
    Tref: jnp.ndarray
    flupmt: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90::traMplan lines 2227-2368",
    )


class DeepCarbonEbullitionResult(NamedTuple):
    """Soil CH4 and ebullition flux after the OK_PC ebullition step."""

    CH4_soil: jnp.ndarray
    febul: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90::ebullition lines 2372-2448",
    )


def altcalc_doc(
    tprof,
    zprof,
    altmax,
    altmax_ind,
    altmax_lastyear,
    altmax_ind_lastyear,
    veget_mask,
    *,
    firstcall: bool,
    newaltcalc: bool = False,
    dayno: int = 1,
    soilc_isspinup: bool = False,
) -> AltCalcDocResult:
    """Compute active-layer thickness/index for the DOC leak path.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``, subroutine
    ``altcalc_DOC``, lines 2438-2578. Indices returned here are zero-based
    counts matching the Fortran values ``iz-1``/``ndeep`` numerically.
    """

    tprof = jnp.asarray(tprof)
    zprof = jnp.asarray(zprof)
    altmax = jnp.asarray(altmax)
    altmax_ind = jnp.asarray(altmax_ind)
    altmax_lastyear = jnp.asarray(altmax_lastyear)
    altmax_ind_lastyear = jnp.asarray(altmax_ind_lastyear)
    veget_mask = jnp.asarray(veget_mask, dtype=bool)
    if tprof.ndim != 3:
        raise ValueError("tprof must have shape (npts, ndeep, nvm)")
    npts, ndeep, nvm = tprof.shape
    if zprof.shape[0] != ndeep:
        raise ValueError("zprof length must match tprof ndeep")
    if veget_mask.shape != (npts, nvm):
        raise ValueError("veget_mask must have shape (npts, nvm)")

    if firstcall:
        thawed_or_above = altmax[:, :, None] >= zprof[None, None, :]
        new_altmax_ind = jnp.sum(thawed_or_above & veget_mask[:, :, None], axis=2)
        return AltCalcDocResult(
            alt=jnp.zeros_like(altmax),
            alt_ind=jnp.zeros_like(altmax_ind),
            altmax=altmax,
            altmax_ind=new_altmax_ind,
            altmax_lastyear=altmax,
            altmax_ind_lastyear=new_altmax_ind,
        )

    if not newaltcalc:
        thawed = tprof > ZERO_CELSIUS
        first_frozen = jnp.argmax(~thawed, axis=1) + 1
        all_thawed = jnp.all(thawed, axis=1)
        iz_fortran = jnp.where(all_thawed, ndeep, first_frozen)
        alt_ind = jnp.where(iz_fortran == 1, 0, iz_fortran - 1)
        alt = jnp.where(iz_fortran == 1, 0.0, zprof[alt_ind - 1])
        alt = jnp.where(veget_mask, alt, 0.0)
        alt_ind = jnp.where(veget_mask, alt_ind, 0)
    else:
        thawed = tprof > ZERO_CELSIUS
        bottom_thawed = thawed[:, -1, :]
        alt = jnp.full_like(altmax, zprof[-1])
        alt_ind = jnp.where(bottom_thawed, ndeep, 0)
        inalt = jnp.zeros((npts, nvm), dtype=bool)
        # Preserve the Fortran bottom-to-top scan: a frozen layer closes a
        # deeper thawed segment, so a later shallow thawed segment replaces it.
        for lev in range(ndeep - 1, 0, -1):
            layer_thawed = thawed[:, lev - 1, :]
            begin = layer_thawed & ~inalt & ~bottom_thawed
            alt = jnp.where(begin, zprof[lev - 1], alt)
            alt_ind = jnp.where(begin, lev, alt_ind)
            inalt = jnp.where(
                (~layer_thawed) & inalt & ~bottom_thawed,
                False,
                inalt | begin,
            )
        alt = jnp.where(~inalt & ~bottom_thawed, 0.0, alt)
        alt_ind = jnp.where(~inalt & ~bottom_thawed, 0, alt_ind)

    update = (alt > altmax) & veget_mask
    new_altmax = jnp.where(update, alt, altmax)
    new_altmax_ind = jnp.where(update, alt_ind, altmax_ind)
    if soilc_isspinup:
        new_lastyear = new_altmax
        new_lastyear_ind = new_altmax_ind
    else:
        is_second_day = jnp.asarray(dayno) == 2
        new_lastyear = jnp.where(is_second_day, new_altmax, altmax_lastyear)
        new_lastyear_ind = jnp.where(
            is_second_day,
            new_altmax_ind,
            altmax_ind_lastyear,
        )
        new_altmax = jnp.where(is_second_day, alt, new_altmax)
        new_altmax_ind = jnp.where(is_second_day, alt_ind, new_altmax_ind)
    return AltCalcDocResult(
        alt=alt,
        alt_ind=alt_ind,
        altmax=new_altmax,
        altmax_ind=new_altmax_ind,
        altmax_lastyear=new_lastyear,
        altmax_ind_lastyear=new_lastyear_ind,
    )


def deep_carbon_altcalc_step(*args, **kwargs) -> AltCalcDocResult:
    """Compute OK_PC active-layer thickness/index with the shared source logic.

    Fortran provenance: ``src_stomate/stomate_permafrost_soilcarbon.f90``,
    subroutine ``altcalc``, lines 1271-1419. The arithmetic is the same
    active-layer algorithm as ``altcalc_DOC`` already used for OK_LEAK; this
    wrapper gives the OK_PC/deep_carbcycle path an explicit source boundary.
    """

    return altcalc_doc(*args, **kwargs)


def deep_carbon_root_depth_step(altmax_lastyear, altmax_ind_lastyear, veget_mask, *, z_root_max) -> DeepCarbonRootDepthResult:
    """Apply the source root-depth cap used before OK_PC carbon input.

    Fortran provenance: ``src_stomate/stomate_permafrost_soilcarbon.f90``,
    ``deep_carbcycle`` lines 945-951.
    """

    altmax = jnp.asarray(altmax_lastyear)
    altind = jnp.asarray(altmax_ind_lastyear)
    mask = jnp.asarray(veget_mask, dtype=bool)
    if altmax.ndim != 2:
        raise ValueError("altmax_lastyear must have shape (npts, nvm)")
    if altind.shape != altmax.shape or mask.shape != altmax.shape:
        raise ValueError("altmax_ind_lastyear and veget_mask must match altmax_lastyear shape")
    use_active_layer = (altmax < z_root_max) & mask
    z_root = jnp.where(mask, jnp.where(use_active_layer, altmax, z_root_max), 0.0)
    rootlev = jnp.where(mask, altind, 0)
    return DeepCarbonRootDepthResult(z_root=z_root, rootlev=rootlev)


def deep_carbon_cryoturbation_coefficients(
    altmax_ind,
    deepC_a,
    deepC_s,
    deepC_p,
    altmax_lastyear,
    fixed_cryoturbation_depth,
    veget_mask,
    zi_soil,
    zf_soil,
    *,
    dt_seconds,
    diff_k_const,
    bio_diff_k_const,
    use_new_cryoturbation: bool = False,
    cryoturbation_method: int = 4,
    max_cryoturb_alt=3.0,
    min_cryoturb_alt=0.01,
    use_fixed_cryoturbation_depth: bool = False,
    bioturbation_depth=2.0,
) -> DeepCarbonCryoturbationCoefficients:
    """Compute saved coefficients for OK_PC ``cryoturbate``.

    Fortran provenance: ``src_stomate/stomate_permafrost_soilcarbon.f90``,
    subroutine ``cryoturbate``, action ``coefficients``, lines 3639-3898.
    ``altmax_ind`` uses the source numeric convention: a Fortran layer count.
    """

    alt_ind = jnp.asarray(altmax_ind)
    deep_a = jnp.asarray(deepC_a)
    deep_s = jnp.asarray(deepC_s)
    deep_p = jnp.asarray(deepC_p)
    alt_last = jnp.asarray(altmax_lastyear)
    fixed_depth = jnp.asarray(fixed_cryoturbation_depth)
    mask = jnp.asarray(veget_mask, dtype=bool)
    zi = jnp.asarray(zi_soil)
    zf = jnp.asarray(zf_soil)
    if deep_a.ndim != 3:
        raise ValueError("deepC_a must have shape (npts, ndeep, nvm)")
    if deep_s.shape != deep_a.shape or deep_p.shape != deep_a.shape:
        raise ValueError("deepC_s and deepC_p must match deepC_a shape")
    npts, ndeep, nvm = deep_a.shape
    if alt_ind.shape != (npts, nvm) or alt_last.shape != (npts, nvm) or fixed_depth.shape != (npts, nvm):
        raise ValueError("altmax_ind, altmax_lastyear, and fixed_cryoturbation_depth must have shape (npts, nvm)")
    if mask.shape != (npts, nvm):
        raise ValueError("veget_mask must have shape (npts, nvm)")
    if zi.shape != (ndeep,):
        raise ValueError("zi_soil must have shape (ndeep,)")
    if zf.shape != (ndeep + 1,):
        raise ValueError("zf_soil must have length ndeep + 1")
    if ndeep < 2:
        raise ValueError("cryoturbate requires at least two deep soil layers")

    dtype = jnp.result_type(deep_a, deep_s, deep_p, zi)
    layer_depths = zi
    layer_no = jnp.arange(1, ndeep + 1)
    cryo_location = (alt_last < max_cryoturb_alt) & (alt_last >= min_cryoturb_alt) & mask
    bio_location = (alt_last >= max_cryoturb_alt) & mask
    cryo_depth = jnp.where(use_fixed_cryoturbation_depth, fixed_depth, alt_last)
    diff_k = jnp.zeros((npts, ndeep, nvm), dtype=dtype)

    for ip in range(npts):
        for iv in range(nvm):
            if bool(cryo_location[ip, iv]):
                depth = cryo_depth[ip, iv]
                if bool(use_new_cryoturbation):
                    if int(cryoturbation_method) == 1:
                        profile = jnp.where(
                            layer_depths <= depth,
                            diff_k_const,
                            diff_k_const * (1.0 - jnp.maximum(jnp.minimum((layer_depths / depth) - 1.0, 1.0), 0.0)),
                        )
                    elif int(cryoturbation_method) == 2:
                        profile = jnp.where(
                            layer_depths <= depth,
                            diff_k_const,
                            diff_k_const * jnp.exp(-jnp.maximum((layer_depths / depth) - 1.0, 0.0)),
                        )
                    elif int(cryoturbation_method) == 3:
                        profile = diff_k_const * jnp.exp(-(layer_depths / depth))
                    elif int(cryoturbation_method) == 4:
                        profile = jnp.where(
                            layer_depths <= depth,
                            diff_k_const,
                            diff_k_const
                            * (
                                1.0
                                - jnp.maximum(
                                    jnp.minimum((layer_depths - depth) / (2.0 * depth), 1.0),
                                    0.0,
                                )
                            ),
                        )
                        profile = jnp.where(layer_depths > max_cryoturb_alt, 0.0, profile)
                    elif int(cryoturbation_method) == 5:
                        profile = jnp.where(
                            layer_depths <= depth,
                            diff_k_const,
                            diff_k_const
                            * (
                                1.0
                                - jnp.maximum(
                                    jnp.minimum((layer_depths - depth) / (3.0 - depth), 1.0),
                                    0.0,
                                )
                            ),
                        )
                    else:
                        raise ValueError("cryoturbation_method must be one of 1, 2, 3, 4, 5")
                else:
                    aind = int(alt_ind[ip, iv])
                    if aind < 1 or aind + 2 > ndeep:
                        raise ValueError("old OK_PC cryoturbation requires 1 <= altmax_ind <= ndeep-2")
                    profile = jnp.where(layer_no <= aind, diff_k_const, 0.0)
                    profile = profile.at[aind].set(diff_k_const / 10.0)
                    profile = profile.at[aind + 1].set(diff_k_const / 100.0)
                diff_k = diff_k.at[ip, :, iv].set(profile)
            elif bool(bio_location[ip, iv]):
                diff_k = diff_k.at[ip, :, iv].set(jnp.where(layer_depths <= bioturbation_depth, bio_diff_k_const, 0.0))

    active = cryo_location | bio_location
    xc = jnp.zeros((npts, ndeep, nvm), dtype=dtype)
    xd = jnp.zeros((npts, ndeep, nvm), dtype=dtype)
    thickness = zf[1:] - zf[:-1]
    xc = jnp.where(active[:, None, :], thickness[None, :, None] / dt_seconds, xc)
    interface_thickness = zi[1:] - zi[:-1]
    xd_inner = diff_k[:, : ndeep - 1, :] / interface_thickness[None, :, None]
    xd = xd.at[:, : ndeep - 1, :].set(jnp.where(active[:, None, :], xd_inner, xd[:, : ndeep - 1, :]))

    alpha_a = jnp.zeros((npts, ndeep, nvm), dtype=dtype)
    alpha_s = jnp.zeros_like(alpha_a)
    alpha_p = jnp.zeros_like(alpha_a)
    beta_a = jnp.zeros_like(deep_a)
    beta_s = jnp.zeros_like(deep_s)
    beta_p = jnp.zeros_like(deep_p)

    xe_a = jnp.where(active, xc[:, ndeep - 1, :] + xd[:, ndeep - 2, :], 1.0)
    xe_s = xe_a
    xe_p = xe_a
    alpha_a = alpha_a.at[:, ndeep - 2, :].set(jnp.where(active, xd[:, ndeep - 2, :] / xe_a, 0.0))
    alpha_s = alpha_s.at[:, ndeep - 2, :].set(jnp.where(active, xd[:, ndeep - 2, :] / xe_s, 0.0))
    alpha_p = alpha_p.at[:, ndeep - 2, :].set(jnp.where(active, xd[:, ndeep - 2, :] / xe_p, 0.0))
    beta_a = beta_a.at[:, ndeep - 2, :].set(jnp.where(active, xc[:, ndeep - 1, :] * deep_a[:, ndeep - 1, :] / xe_a, 0.0))
    beta_s = beta_s.at[:, ndeep - 2, :].set(jnp.where(active, xc[:, ndeep - 1, :] * deep_s[:, ndeep - 1, :] / xe_s, 0.0))
    beta_p = beta_p.at[:, ndeep - 2, :].set(jnp.where(active, xc[:, ndeep - 1, :] * deep_p[:, ndeep - 1, :] / xe_p, 0.0))

    for layer in range(ndeep - 3, -1, -1):
        xe_a = jnp.where(
            active,
            xc[:, layer + 1, :] + (1.0 - alpha_a[:, layer + 1, :]) * xd[:, layer + 1, :] + xd[:, layer, :],
            1.0,
        )
        xe_s = jnp.where(
            active,
            xc[:, layer + 1, :] + (1.0 - alpha_s[:, layer + 1, :]) * xd[:, layer + 1, :] + xd[:, layer, :],
            1.0,
        )
        xe_p = jnp.where(
            active,
            xc[:, layer + 1, :] + (1.0 - alpha_s[:, layer + 1, :]) * xd[:, layer + 1, :] + xd[:, layer, :],
            1.0,
        )
        alpha_a = alpha_a.at[:, layer, :].set(jnp.where(active, xd[:, layer, :] / xe_a, 0.0))
        alpha_s = alpha_s.at[:, layer, :].set(jnp.where(active, xd[:, layer, :] / xe_s, 0.0))
        alpha_p = alpha_p.at[:, layer, :].set(jnp.where(active, xd[:, layer, :] / xe_p, 0.0))
        beta_a = beta_a.at[:, layer, :].set(
            jnp.where(
                active,
                (xc[:, layer + 1, :] * deep_a[:, layer + 1, :] + xd[:, layer + 1, :] * beta_a[:, layer + 1, :]) / xe_a,
                0.0,
            )
        )
        beta_s = beta_s.at[:, layer, :].set(
            jnp.where(
                active,
                (xc[:, layer + 1, :] * deep_s[:, layer + 1, :] + xd[:, layer + 1, :] * beta_s[:, layer + 1, :]) / xe_s,
                0.0,
            )
        )
        beta_p = beta_p.at[:, layer, :].set(
            jnp.where(
                active,
                (xc[:, layer + 1, :] * deep_p[:, layer + 1, :] + xd[:, layer + 1, :] * beta_p[:, layer + 1, :]) / xe_p,
                0.0,
            )
        )

    return DeepCarbonCryoturbationCoefficients(
        cryoturb_location=cryo_location,
        bioturb_location=bio_location,
        cryoturbation_depth=cryo_depth,
        diff_k=diff_k,
        xc_cryoturb=xc,
        xd_cryoturb=xd,
        alpha_a=alpha_a,
        beta_a=beta_a,
        alpha_s=alpha_s,
        beta_s=beta_s,
        alpha_p=alpha_p,
        beta_p=beta_p,
    )


def _deep_carbon_cryoturbation_diffuse_one(pool, alpha, beta, active, altmax_ind, zf_soil, zi_soil):
    work = jnp.asarray(pool)
    alpha_arr = jnp.asarray(alpha)
    beta_arr = jnp.asarray(beta)
    active_arr = jnp.asarray(active, dtype=bool)
    alt_ind = jnp.asarray(altmax_ind)
    zf = jnp.asarray(zf_soil)
    zi = jnp.asarray(zi_soil)
    npts, ndeep, nvm = work.shape
    thickness = zf[1:] - zf[:-1]
    mu_soil = zi[0] / (zi[1] - zi[0])
    surfC_totake = jnp.zeros((npts, nvm), dtype=work.dtype)
    for ip in range(npts):
        for iv in range(nvm):
            if bool(active_arr[ip, iv]):
                aind = int(alt_ind[ip, iv])
                if aind < 1 or aind > ndeep:
                    raise ValueError("active OK_PC cryoturbation requires 1 <= altmax_ind <= ndeep")
                old_profile = work[ip, :, iv]
                old_total = jnp.sum(old_profile * thickness)
                new_profile = old_profile
                top = (new_profile[0] + mu_soil * beta_arr[ip, 0, iv]) / (1.0 + mu_soil * (1.0 - alpha_arr[ip, 0, iv]))
                new_profile = new_profile.at[0].set(top)
                for layer in range(1, ndeep):
                    new_profile = new_profile.at[layer].set(
                        alpha_arr[ip, layer - 1, iv] * new_profile[layer - 1] + beta_arr[ip, layer - 1, iv]
                    )
                new_total = jnp.sum(new_profile * thickness)
                denom = zf[aind] - zf[0]
                surf = (new_total - old_total) / denom
                corrected = new_profile.at[:aind].add(-surf)
                has_negative = bool(jnp.any(corrected[:aind] < 0.0))
                if has_negative:
                    corrected = new_profile
                    if bool(new_total > 0.0):
                        corrected = corrected * old_total / new_total
                work = work.at[ip, :, iv].set(corrected)
                surfC_totake = surfC_totake.at[ip, iv].set(surf)
    return work, surfC_totake


def deep_carbon_cryoturbation_diffuse(
    deepC_a,
    deepC_s,
    deepC_p,
    coefficients: DeepCarbonCryoturbationCoefficients,
    altmax_ind,
    zi_soil,
    zf_soil,
) -> DeepCarbonCryoturbationDiffuseResult:
    """Apply saved OK_PC ``cryoturbate`` coefficients to deep-carbon pools.

    Fortran provenance: ``src_stomate/stomate_permafrost_soilcarbon.f90``,
    subroutine ``cryoturbate``, action ``diffuse``, lines 3546-3628. Unlike
    ``cryoturbate_doc_POC``, this source path applies the active-layer surface
    correction and negative-value fallback.
    """

    deep_a = jnp.asarray(deepC_a)
    deep_s = jnp.asarray(deepC_s)
    deep_p = jnp.asarray(deepC_p)
    alt_ind = jnp.asarray(altmax_ind)
    zi = jnp.asarray(zi_soil)
    zf = jnp.asarray(zf_soil)
    if deep_a.ndim != 3:
        raise ValueError("deepC_a must have shape (npts, ndeep, nvm)")
    if deep_s.shape != deep_a.shape or deep_p.shape != deep_a.shape:
        raise ValueError("deepC_s and deepC_p must match deepC_a shape")
    npts, ndeep, nvm = deep_a.shape
    if alt_ind.shape != (npts, nvm):
        raise ValueError("altmax_ind must have shape (npts, nvm)")
    if zi.shape != (ndeep,) or zf.shape != (ndeep + 1,):
        raise ValueError("zi_soil/zf_soil shapes must match deep-carbon layers")
    active = coefficients.cryoturb_location | coefficients.bioturb_location
    next_a, surf_a = _deep_carbon_cryoturbation_diffuse_one(
        deep_a, coefficients.alpha_a, coefficients.beta_a, active, alt_ind, zf, zi
    )
    next_s, surf_s = _deep_carbon_cryoturbation_diffuse_one(
        deep_s, coefficients.alpha_s, coefficients.beta_s, active, alt_ind, zf, zi
    )
    next_p, surf_p = _deep_carbon_cryoturbation_diffuse_one(
        deep_p, coefficients.alpha_p, coefficients.beta_p, active, alt_ind, zf, zi
    )
    return DeepCarbonCryoturbationDiffuseResult(next_a, next_s, next_p, surf_a, surf_s, surf_p)


def deep_carbon_cryoturbation_cycle(
    deepC_a,
    deepC_s,
    deepC_p,
    previous_coefficients,
    altmax_ind,
    altmax_lastyear,
    fixed_cryoturbation_depth,
    veget_mask,
    zi_soil,
    zf_soil,
    *,
    dt_seconds,
    diff_k_const,
    bio_diff_k_const,
    use_new_cryoturbation: bool = False,
    cryoturbation_method: int = 4,
    max_cryoturb_alt=3.0,
    min_cryoturb_alt=0.01,
    use_fixed_cryoturbation_depth: bool = False,
    bioturbation_depth=2.0,
) -> tuple[DeepCarbonCryoturbationDiffuseResult, DeepCarbonCryoturbationCoefficients]:
    """Run the source-order OK_PC cryoturbation diffuse/coefficients pair.

    Fortran provenance: ``deep_carbcycle`` calls ``cryoturbate(...,
    'diffuse', ...)`` before carbon input and ``cryoturbate(...,
    'coefficients', ...)`` after respiration aggregation, with the action
    internals in ``cryoturbate`` lines 3431-3916.
    """

    if previous_coefficients is None:
        diffuse = DeepCarbonCryoturbationDiffuseResult(
            jnp.asarray(deepC_a),
            jnp.asarray(deepC_s),
            jnp.asarray(deepC_p),
            jnp.zeros_like(jnp.asarray(altmax_lastyear)),
            jnp.zeros_like(jnp.asarray(altmax_lastyear)),
            jnp.zeros_like(jnp.asarray(altmax_lastyear)),
        )
    else:
        diffuse = deep_carbon_cryoturbation_diffuse(
            deepC_a,
            deepC_s,
            deepC_p,
            previous_coefficients,
            altmax_ind,
            zi_soil,
            zf_soil,
        )
    new_coefficients = deep_carbon_cryoturbation_coefficients(
        altmax_ind,
        diffuse.deepC_a,
        diffuse.deepC_s,
        diffuse.deepC_p,
        altmax_lastyear,
        fixed_cryoturbation_depth,
        veget_mask,
        zi_soil,
        zf_soil,
        dt_seconds=dt_seconds,
        diff_k_const=diff_k_const,
        bio_diff_k_const=bio_diff_k_const,
        use_new_cryoturbation=use_new_cryoturbation,
        cryoturbation_method=cryoturbation_method,
        max_cryoturb_alt=max_cryoturb_alt,
        min_cryoturb_alt=min_cryoturb_alt,
        use_fixed_cryoturbation_depth=use_fixed_cryoturbation_depth,
        bioturbation_depth=bioturbation_depth,
    )
    return diffuse, new_coefficients


def deep_carbon_snowlevels_step(snowdz, veget_max) -> DeepCarbonSnowGeometryResult:
    """Build OK_PC snow intermediate/full levels from snow-layer thickness.

    Fortran provenance: ``src_stomate/stomate_permafrost_soilcarbon.f90``,
    subroutine ``snowlevels``, lines 2708-2802. The source copies the same
    ``snowdz`` profile to every PFT and does not mask by vegetation.
    """

    snowdz_arr = jnp.asarray(snowdz)
    veget = jnp.asarray(veget_max)
    if snowdz_arr.ndim != 2:
        raise ValueError("snowdz must have shape (npts, nsnow)")
    npts, nsnow = snowdz_arr.shape
    if veget.ndim != 2 or veget.shape[0] != npts:
        raise ValueError("veget_max must have shape (npts, nvm)")
    nvm = veget.shape[1]
    snowdz_pft = jnp.repeat(snowdz_arr[:, :, None], nvm, axis=2)
    zf = jnp.zeros((npts, nsnow + 1, nvm), dtype=snowdz_arr.dtype)
    zi = jnp.zeros((npts, nsnow, nvm), dtype=snowdz_arr.dtype)
    cumulative = jnp.cumsum(snowdz_pft, axis=1)
    zf = zf.at[:, 1:, :].set(cumulative)
    zi = zi.at[:, 0, :].set(snowdz_pft[:, 0, :] / 2.0)
    if nsnow > 1:
        zi = zi.at[:, 1:, :].set(zf[:, 1:-1, :] + snowdz_pft[:, 1:, :] / 2.0)
    return DeepCarbonSnowGeometryResult(zi_snow=zi, zf_snow=zf)


def deep_carbon_snow_interpol_step(
    snowO2,
    snowCH4,
    zi_snow,
    zf_snow,
    veget_max,
    snowdz,
    veget_mask,
    *,
    min_stomate=MIN_STOMATE,
) -> DeepCarbonSnowInterpolResult:
    """Interpolate snow O2/CH4 after changing the snow vertical grid.

    Fortran provenance: ``src_stomate/stomate_permafrost_soilcarbon.f90``,
    subroutine ``snow_interpol``, lines 2806-2953.
    """

    o2 = jnp.asarray(snowO2)
    ch4 = jnp.asarray(snowCH4)
    old_zi = jnp.asarray(zi_snow)
    old_zf = jnp.asarray(zf_snow)
    mask = jnp.asarray(veget_mask, dtype=bool)
    if o2.ndim != 3:
        raise ValueError("snowO2 must have shape (npts, nsnow, nvm)")
    if ch4.shape != o2.shape or old_zi.shape != o2.shape:
        raise ValueError("snowCH4 and zi_snow must match snowO2 shape")
    npts, nsnow, nvm = o2.shape
    if old_zf.shape != (npts, nsnow + 1, nvm):
        raise ValueError("zf_snow must have shape (npts, nsnow + 1, nvm)")
    if mask.shape != (npts, nvm):
        raise ValueError("veget_mask must have shape (npts, nvm)")
    if nsnow < 2:
        raise ValueError("snow_interpol requires at least two snow layers")

    geometry = deep_carbon_snowlevels_step(snowdz, veget_max)
    new_o2 = o2
    new_ch4 = ch4
    for ip in range(npts):
        for iv in range(nvm):
            if bool(mask[ip, iv]):
                for il in range(nsnow):
                    isnow = -1
                    for ill in range(nsnow - 1, -1, -1):
                        if bool(old_zi[ip, ill, iv] > geometry.zi_snow[ip, il, iv]):
                            isnow = ill + 1
                    if isnow == 1:
                        i1, i2 = 0, 1
                    elif isnow == -1:
                        i1, i2 = nsnow - 2, nsnow - 1
                    else:
                        i1, i2 = isnow - 2, isnow - 1
                    dzio = old_zi[ip, i2, iv] - old_zi[ip, i1, iv]
                    if bool(dzio > min_stomate):
                        weight = (geometry.zi_snow[ip, il, iv] - old_zi[ip, i1, iv]) / dzio
                        new_o2 = new_o2.at[ip, il, iv].set(o2[ip, i1, iv] + weight * (o2[ip, i2, iv] - o2[ip, i1, iv]))
                        new_ch4 = new_ch4.at[ip, il, iv].set(ch4[ip, i1, iv] + weight * (ch4[ip, i2, iv] - ch4[ip, i1, iv]))
                    else:
                        new_o2 = new_o2.at[ip, il, iv].set(o2[ip, i1, iv])
                        new_ch4 = new_ch4.at[ip, il, iv].set(ch4[ip, i1, iv])
    return DeepCarbonSnowInterpolResult(new_o2, new_ch4, geometry.zi_snow, geometry.zf_snow)


def deep_carbon_gasdiff_properties_step(
    hslong,
    snowrho,
    veget_mask,
    *,
    rho_ice=RHO_ICE,
    diffO2_air=DIFFO2_AIR,
    diffCH4_air=DIFFCH4_AIR,
    diffO2_w=DIFFO2_WATER,
    diffCH4_w=DIFFCH4_WATER,
    BunsenO2=BUNSEN_O2,
    BunsenCH4=BUNSEN_CH4,
    tetasat=TETASAT,
    avm=AVM,
) -> DeepCarbonGasDiffusionPropertiesResult:
    """Compute OK_PC snow/soil gas porosity and diffusivity.

    Fortran provenance: ``src_stomate/stomate_permafrost_soilcarbon.f90``,
    subroutine ``get_gasdiff``, lines 2098-2223. Inputs unused by the source
    algebra in this block (``snow``, ``tprof``, ``z_organic``) are intentionally
    excluded.
    """

    hum = jnp.asarray(hslong)
    snow_density = jnp.asarray(snowrho)
    mask = jnp.asarray(veget_mask, dtype=bool)
    if hum.ndim != 3:
        raise ValueError("hslong must have shape (npts, ndeep, nvm)")
    npts, ndeep, nvm = hum.shape
    if snow_density.ndim != 2 or snow_density.shape[0] != npts:
        raise ValueError("snowrho must have shape (npts, nsnow)")
    if mask.shape != (npts, nvm):
        raise ValueError("veget_mask must have shape (npts, nvm)")
    nsnow = snow_density.shape[1]
    density_snow = jnp.repeat(snow_density[:, :, None], nvm, axis=2)
    porosity_snow = 1.0 - density_snow / rho_ice
    tortuosity_snow = porosity_snow ** (1.0 / 3.0)
    diff_o2_snow = diffO2_air * porosity_snow * tortuosity_snow
    diff_ch4_snow = diffCH4_air * porosity_snow * tortuosity_snow
    airvol_snow = jnp.maximum(porosity_snow, avm)
    totpor_o2_snow = airvol_snow
    totpor_ch4_snow = airvol_snow

    eps = jnp.asarray(jnp.finfo(jnp.float32).eps, dtype=hum.dtype)
    porosity_soil = jnp.ones_like(hum) * jnp.asarray(tetasat, dtype=hum.dtype)
    tortuosity_soil = jnp.ones_like(hum) * (2.0 / 3.0)
    airvol_soil = porosity_soil * (1.0 - hum)
    totpor_o2_soil = airvol_soil + porosity_soil * BunsenO2 * hum
    totpor_ch4_soil = airvol_soil + porosity_soil * BunsenCH4 * hum
    diff_o2_soil = (diffO2_air * airvol_soil + diffO2_w * BunsenO2 * hum * porosity_soil) * tortuosity_soil
    diff_ch4_soil = (diffCH4_air * airvol_soil + diffCH4_w * BunsenCH4 * hum * porosity_soil) * tortuosity_soil
    inactive = ~mask[:, None, :]
    airvol_soil = jnp.where(inactive, eps, airvol_soil)
    totpor_o2_soil = jnp.where(inactive, eps, totpor_o2_soil)
    totpor_ch4_soil = jnp.where(inactive, eps, totpor_ch4_soil)
    diff_o2_soil = jnp.where(inactive, eps, diff_o2_soil)
    diff_ch4_soil = jnp.where(inactive, eps, diff_ch4_soil)
    return DeepCarbonGasDiffusionPropertiesResult(
        airvol_snow=airvol_snow,
        totporO2_snow=totpor_o2_snow,
        totporCH4_snow=totpor_ch4_snow,
        diffO2_snow=diff_o2_snow,
        diffCH4_snow=diff_ch4_snow,
        airvol_soil=airvol_soil,
        totporO2_soil=totpor_o2_soil,
        totporCH4_soil=totpor_ch4_soil,
        diffO2_soil=diff_o2_soil,
        diffCH4_soil=diff_ch4_soil,
    )


def deep_carbon_soil_gasdiff_coefficients(
    O2_snow,
    CH4_snow,
    diffO2_snow,
    diffCH4_snow,
    totporO2_snow,
    totporCH4_snow,
    O2_soil,
    CH4_soil,
    diffO2_soil,
    diffCH4_soil,
    totporO2_soil,
    totporCH4_soil,
    zi_snow,
    zf_snow,
    zi_soil,
    zf_soil,
    veget_mask,
    heights_snow,
    *,
    dt_seconds,
    hmin_tcalc=HMIN_TCALC,
) -> DeepCarbonGasDiffusionCoefficients:
    """Compute saved coefficients for OK_PC snow/soil gas diffusion."""

    o2_snow = jnp.asarray(O2_snow)
    ch4_snow = jnp.asarray(CH4_snow)
    do2_snow = jnp.asarray(diffO2_snow)
    dch4_snow = jnp.asarray(diffCH4_snow)
    tp_o2_snow = jnp.asarray(totporO2_snow)
    tp_ch4_snow = jnp.asarray(totporCH4_snow)
    o2_soil = jnp.asarray(O2_soil)
    ch4_soil = jnp.asarray(CH4_soil)
    do2_soil = jnp.asarray(diffO2_soil)
    dch4_soil = jnp.asarray(diffCH4_soil)
    tp_o2_soil = jnp.asarray(totporO2_soil)
    tp_ch4_soil = jnp.asarray(totporCH4_soil)
    zi_snow_arr = jnp.asarray(zi_snow)
    zf_snow_arr = jnp.asarray(zf_snow)
    zi = jnp.asarray(zi_soil)
    zf = jnp.asarray(zf_soil)
    mask = jnp.asarray(veget_mask, dtype=bool)
    snow_height = jnp.asarray(heights_snow)
    if o2_soil.ndim != 3:
        raise ValueError("O2_soil must have shape (npts, ndeep, nvm)")
    npts, ndeep, nvm = o2_soil.shape
    if o2_snow.ndim != 3:
        raise ValueError("O2_snow must have shape (npts, nsnow, nvm)")
    nsnow = o2_snow.shape[1]
    if nsnow < 2 or ndeep < 2:
        raise ValueError("soil_gasdiff_coeff requires at least two snow and soil layers")
    for name, arr, shape in (
        ("CH4_snow", ch4_snow, o2_snow.shape),
        ("diffO2_snow", do2_snow, o2_snow.shape),
        ("diffCH4_snow", dch4_snow, o2_snow.shape),
        ("totporO2_snow", tp_o2_snow, o2_snow.shape),
        ("totporCH4_snow", tp_ch4_snow, o2_snow.shape),
        ("CH4_soil", ch4_soil, o2_soil.shape),
        ("diffO2_soil", do2_soil, o2_soil.shape),
        ("diffCH4_soil", dch4_soil, o2_soil.shape),
        ("totporO2_soil", tp_o2_soil, o2_soil.shape),
        ("totporCH4_soil", tp_ch4_soil, o2_soil.shape),
        ("zi_snow", zi_snow_arr, o2_snow.shape),
    ):
        if arr.shape != shape:
            raise ValueError(f"{name} has incompatible shape")
    if zf_snow_arr.shape != (npts, nsnow + 1, nvm):
        raise ValueError("zf_snow must have shape (npts, nsnow + 1, nvm)")
    if zi.shape != (ndeep,) or zf.shape != (ndeep + 1,):
        raise ValueError("zi_soil/zf_soil shapes must match soil layers")
    if mask.shape != (npts, nvm) or snow_height.shape != (npts, nvm):
        raise ValueError("veget_mask and heights_snow must have shape (npts, nvm)")

    dtype = jnp.result_type(o2_soil, o2_snow, zi)
    active = mask
    snow_active = (snow_height > hmin_tcalc) & mask
    xc_o2_soil = jnp.zeros((npts, ndeep, nvm), dtype=dtype)
    xd_o2_soil = jnp.zeros_like(xc_o2_soil)
    xc_ch4_soil = jnp.zeros_like(xc_o2_soil)
    xd_ch4_soil = jnp.zeros_like(xc_o2_soil)
    soil_thickness = zf[1:] - zf[:-1]
    xc_o2_soil = jnp.where(active[:, None, :], soil_thickness[None, :, None] * tp_o2_soil / dt_seconds, xc_o2_soil)
    xc_ch4_soil = jnp.where(active[:, None, :], soil_thickness[None, :, None] * tp_ch4_soil / dt_seconds, xc_ch4_soil)
    xd_o2_soil = xd_o2_soil.at[:, : ndeep - 1, :].set(
        jnp.where(active[:, None, :], do2_soil[:, : ndeep - 1, :] / (zi[1:] - zi[:-1])[None, :, None], 0.0)
    )
    xd_ch4_soil = xd_ch4_soil.at[:, : ndeep - 1, :].set(
        jnp.where(active[:, None, :], dch4_soil[:, : ndeep - 1, :] / (zi[1:] - zi[:-1])[None, :, None], 0.0)
    )
    xd_o2_soil = xd_o2_soil.at[:, ndeep - 1, :].set(jnp.where(active, do2_soil[:, ndeep - 1, :] / (zf[-1] - zi[-1]), 0.0))
    xd_ch4_soil = xd_ch4_soil.at[:, ndeep - 1, :].set(jnp.where(active, dch4_soil[:, ndeep - 1, :] / (zf[-1] - zi[-1]), 0.0))

    xc_o2_snow = jnp.zeros((npts, nsnow, nvm), dtype=dtype)
    xd_o2_snow = jnp.zeros_like(xc_o2_snow)
    xc_ch4_snow = jnp.zeros_like(xc_o2_snow)
    xd_ch4_snow = jnp.zeros_like(xc_o2_snow)
    snow_thickness = zf_snow_arr[:, 1:, :] - zf_snow_arr[:, :-1, :]
    xc_o2_snow = jnp.where(snow_active[:, None, :], snow_thickness * tp_o2_snow / dt_seconds, xc_o2_snow)
    xc_ch4_snow = jnp.where(snow_active[:, None, :], snow_thickness * tp_ch4_snow / dt_seconds, xc_ch4_snow)
    xd_o2_snow = xd_o2_snow.at[:, : nsnow - 1, :].set(
        jnp.where(snow_active[:, None, :], do2_snow[:, : nsnow - 1, :] / (zi_snow_arr[:, 1:, :] - zi_snow_arr[:, :-1, :]), 0.0)
    )
    xd_ch4_snow = xd_ch4_snow.at[:, : nsnow - 1, :].set(
        jnp.where(snow_active[:, None, :], dch4_snow[:, : nsnow - 1, :] / (zi_snow_arr[:, 1:, :] - zi_snow_arr[:, :-1, :]), 0.0)
    )
    xd_o2_snow = xd_o2_snow.at[:, nsnow - 1, :].set(
        jnp.where(snow_active, do2_snow[:, nsnow - 1, :] / (zi[0] + zf_snow_arr[:, nsnow, :] - zi_snow_arr[:, nsnow - 1, :]), 0.0)
    )
    xd_ch4_snow = xd_ch4_snow.at[:, nsnow - 1, :].set(
        jnp.where(snow_active, dch4_snow[:, nsnow - 1, :] / (zi[0] + zf_snow_arr[:, nsnow, :] - zi_snow_arr[:, nsnow - 1, :]), 0.0)
    )
    mu_soil = zi[0] / (zi[1] - zi[0])
    mu_snow = jnp.where(snow_active, zi_snow_arr[:, 0, :] / (zi_snow_arr[:, 1, :] - zi_snow_arr[:, 0, :]), 0.5)

    alpha_o2_soil = jnp.zeros_like(xc_o2_soil)
    beta_o2_soil = jnp.zeros_like(xc_o2_soil)
    alpha_ch4_soil = jnp.zeros_like(xc_o2_soil)
    beta_ch4_soil = jnp.zeros_like(xc_o2_soil)
    xe = jnp.where(active, xc_o2_soil[:, ndeep - 1, :] + xd_o2_soil[:, ndeep - 2, :], 1.0)
    alpha_o2_soil = alpha_o2_soil.at[:, ndeep - 2, :].set(jnp.where(active, xd_o2_soil[:, ndeep - 2, :] / xe, 0.0))
    beta_o2_soil = beta_o2_soil.at[:, ndeep - 2, :].set(jnp.where(active, xc_o2_soil[:, ndeep - 1, :] * o2_soil[:, ndeep - 1, :] / xe, 0.0))
    xe = jnp.where(active, xc_ch4_soil[:, ndeep - 1, :] + xd_ch4_soil[:, ndeep - 2, :], 1.0)
    alpha_ch4_soil = alpha_ch4_soil.at[:, ndeep - 2, :].set(jnp.where(active, xd_ch4_soil[:, ndeep - 2, :] / xe, 0.0))
    beta_ch4_soil = beta_ch4_soil.at[:, ndeep - 2, :].set(jnp.where(active, xc_ch4_soil[:, ndeep - 1, :] * ch4_soil[:, ndeep - 1, :] / xe, 0.0))
    for layer in range(ndeep - 3, -1, -1):
        xe = jnp.where(active, xc_o2_soil[:, layer + 1, :] + (1.0 - alpha_o2_soil[:, layer + 1, :]) * xd_o2_soil[:, layer + 1, :] + xd_o2_soil[:, layer, :], 1.0)
        alpha_o2_soil = alpha_o2_soil.at[:, layer, :].set(jnp.where(active, xd_o2_soil[:, layer, :] / xe, 0.0))
        beta_o2_soil = beta_o2_soil.at[:, layer, :].set(jnp.where(active, (xc_o2_soil[:, layer + 1, :] * o2_soil[:, layer + 1, :] + xd_o2_soil[:, layer + 1, :] * beta_o2_soil[:, layer + 1, :]) / xe, 0.0))
        xe = jnp.where(active, xc_ch4_soil[:, layer + 1, :] + (1.0 - alpha_ch4_soil[:, layer + 1, :]) * xd_ch4_soil[:, layer + 1, :] + xd_ch4_soil[:, layer, :], 1.0)
        alpha_ch4_soil = alpha_ch4_soil.at[:, layer, :].set(jnp.where(active, xd_ch4_soil[:, layer, :] / xe, 0.0))
        beta_ch4_soil = beta_ch4_soil.at[:, layer, :].set(jnp.where(active, (xc_ch4_soil[:, layer + 1, :] * ch4_soil[:, layer + 1, :] + xd_ch4_soil[:, layer + 1, :] * beta_ch4_soil[:, layer + 1, :]) / xe, 0.0))

    alpha_o2_snow = jnp.ones_like(xc_o2_snow)
    beta_o2_snow = jnp.zeros_like(xc_o2_snow)
    alpha_ch4_snow = jnp.ones_like(xc_o2_snow)
    beta_ch4_snow = jnp.zeros_like(xc_o2_snow)
    xe = jnp.where(snow_active, xc_o2_soil[:, 0, :] + (1.0 - alpha_o2_soil[:, 0, :]) * xd_o2_soil[:, 0, :] + xd_o2_snow[:, nsnow - 1, :], 1.0)
    alpha_o2_snow = alpha_o2_snow.at[:, nsnow - 1, :].set(jnp.where(snow_active, xd_o2_snow[:, nsnow - 1, :] / xe, alpha_o2_snow[:, nsnow - 1, :]))
    beta_o2_snow = beta_o2_snow.at[:, nsnow - 1, :].set(jnp.where(snow_active, (xc_o2_soil[:, 0, :] * o2_soil[:, 0, :] + xd_o2_soil[:, 0, :] * beta_o2_soil[:, 0, :]) / xe, 0.0))
    xe = jnp.where(snow_active, xc_ch4_soil[:, 0, :] + (1.0 - alpha_ch4_soil[:, 0, :]) * xd_ch4_soil[:, 0, :] + xd_ch4_snow[:, nsnow - 1, :], 1.0)
    alpha_ch4_snow = alpha_ch4_snow.at[:, nsnow - 1, :].set(jnp.where(snow_active, xd_ch4_snow[:, nsnow - 1, :] / xe, alpha_ch4_snow[:, nsnow - 1, :]))
    beta_ch4_snow = beta_ch4_snow.at[:, nsnow - 1, :].set(jnp.where(snow_active, (xc_ch4_soil[:, 0, :] * ch4_soil[:, 0, :] + xd_ch4_soil[:, 0, :] * beta_ch4_soil[:, 0, :]) / xe, 0.0))
    for layer in range(nsnow - 2, -1, -1):
        xe = jnp.where(snow_active, xc_o2_snow[:, layer + 1, :] + (1.0 - alpha_o2_snow[:, layer + 1, :]) * xd_o2_snow[:, layer + 1, :] + xd_o2_snow[:, layer, :], 1.0)
        alpha_o2_snow = alpha_o2_snow.at[:, layer, :].set(jnp.where(snow_active, xd_o2_snow[:, layer, :] / xe, alpha_o2_snow[:, layer, :]))
        beta_o2_snow = beta_o2_snow.at[:, layer, :].set(jnp.where(snow_active, (xc_o2_snow[:, layer + 1, :] * o2_snow[:, layer + 1, :] + xd_o2_snow[:, layer + 1, :] * beta_o2_snow[:, layer + 1, :]) / xe, 0.0))
        xe = jnp.where(snow_active, xc_ch4_snow[:, layer + 1, :] + (1.0 - alpha_ch4_snow[:, layer + 1, :]) * xd_ch4_snow[:, layer + 1, :] + xd_ch4_snow[:, layer, :], 1.0)
        alpha_ch4_snow = alpha_ch4_snow.at[:, layer, :].set(jnp.where(snow_active, xd_ch4_snow[:, layer, :] / xe, alpha_ch4_snow[:, layer, :]))
        beta_ch4_snow = beta_ch4_snow.at[:, layer, :].set(jnp.where(snow_active, (xc_ch4_snow[:, layer + 1, :] * ch4_snow[:, layer + 1, :] + xd_ch4_snow[:, layer + 1, :] * beta_ch4_snow[:, layer + 1, :]) / xe, 0.0))

    return DeepCarbonGasDiffusionCoefficients(
        alphaO2_soil=alpha_o2_soil,
        betaO2_soil=beta_o2_soil,
        alphaCH4_soil=alpha_ch4_soil,
        betaCH4_soil=beta_ch4_soil,
        alphaO2_snow=alpha_o2_snow,
        betaO2_snow=beta_o2_snow,
        alphaCH4_snow=alpha_ch4_snow,
        betaCH4_snow=beta_ch4_snow,
        mu_soil=mu_soil,
        mu_snow=mu_snow,
        zi_coeff_snow=zi_snow_arr,
        zf_coeff_snow=zf_snow_arr,
        snow_height_mask=snow_active,
    )


def deep_carbon_soil_gasdiff_diffuse(
    O2_snow,
    CH4_snow,
    O2_soil,
    CH4_soil,
    coefficients: DeepCarbonGasDiffusionCoefficients,
    psol,
    tsurf,
    veget_mask,
    *,
    O2_surf=O2_SURF,
    CH4_surf=CH4_SURF,
    wO2=32.0,
    wCH4=16.0,
    rr=RR_GAS,
) -> DeepCarbonGasDiffusionStateResult:
    """Apply saved OK_PC gas-diffusion coefficients to snow/soil O2 and CH4."""

    o2_snow = jnp.asarray(O2_snow)
    ch4_snow = jnp.asarray(CH4_snow)
    o2_soil = jnp.asarray(O2_soil)
    ch4_soil = jnp.asarray(CH4_soil)
    pressure = jnp.asarray(psol)
    surface_temp = jnp.asarray(tsurf)
    mask = jnp.asarray(veget_mask, dtype=bool)
    npts, nsnow, nvm = o2_snow.shape
    ndeep = o2_soil.shape[1]
    if ch4_snow.shape != o2_snow.shape or o2_soil.shape != (npts, ndeep, nvm) or ch4_soil.shape != o2_soil.shape:
        raise ValueError("snow/soil gas arrays have incompatible shapes")
    if pressure.shape != (npts,) or surface_temp.shape != (npts,) or mask.shape != (npts, nvm):
        raise ValueError("psol, tsurf, and veget_mask shapes are incompatible")
    o2sa = pressure[:, None] / (rr * surface_temp[:, None]) * O2_surf * wO2
    ch4sa = pressure[:, None] / (rr * surface_temp[:, None]) * CH4_surf * wCH4
    snowtop = coefficients.snow_height_mask
    no_snow = (~snowtop) & mask
    with_snow = snowtop & mask

    o2_snow = o2_snow.at[:, 0, :].set(jnp.where(no_snow, o2sa, o2_snow[:, 0, :]))
    ch4_snow = ch4_snow.at[:, 0, :].set(jnp.where(no_snow, ch4sa, ch4_snow[:, 0, :]))
    o2_top = (o2sa + coefficients.mu_soil * coefficients.betaO2_soil[:, 0, :]) / (
        1.0 + coefficients.mu_soil * (1.0 - coefficients.alphaO2_soil[:, 0, :])
    )
    ch4_top = (ch4sa + coefficients.mu_soil * coefficients.betaCH4_soil[:, 0, :]) / (
        1.0 + coefficients.mu_soil * (1.0 - coefficients.alphaCH4_soil[:, 0, :])
    )
    o2_soil = o2_soil.at[:, 0, :].set(jnp.where(no_snow, o2_top, o2_soil[:, 0, :]))
    ch4_soil = ch4_soil.at[:, 0, :].set(jnp.where(no_snow, ch4_top, ch4_soil[:, 0, :]))

    snow_o2_top = (o2sa + coefficients.mu_snow * coefficients.betaO2_snow[:, 0, :]) / (
        1.0 + coefficients.mu_snow * (1.0 - coefficients.alphaO2_snow[:, 0, :])
    )
    snow_ch4_top = (ch4sa + coefficients.mu_snow * coefficients.betaCH4_snow[:, 0, :]) / (
        1.0 + coefficients.mu_snow * (1.0 - coefficients.alphaCH4_snow[:, 0, :])
    )
    o2_snow = o2_snow.at[:, 0, :].set(jnp.where(with_snow, snow_o2_top, o2_snow[:, 0, :]))
    ch4_snow = ch4_snow.at[:, 0, :].set(jnp.where(with_snow, snow_ch4_top, ch4_snow[:, 0, :]))
    o2_soil = o2_soil.at[:, 0, :].set(
        jnp.where(with_snow, coefficients.alphaO2_snow[:, nsnow - 1, :] * o2_snow[:, nsnow - 1, :] + coefficients.betaO2_snow[:, nsnow - 1, :], o2_soil[:, 0, :])
    )
    ch4_soil = ch4_soil.at[:, 0, :].set(
        jnp.where(with_snow, coefficients.alphaCH4_snow[:, nsnow - 1, :] * ch4_snow[:, nsnow - 1, :] + coefficients.betaCH4_snow[:, nsnow - 1, :], ch4_soil[:, 0, :])
    )
    for layer in range(1, nsnow):
        o2_snow = o2_snow.at[:, layer, :].set(
            jnp.where(mask, coefficients.alphaO2_snow[:, layer - 1, :] * o2_snow[:, layer - 1, :] + coefficients.betaO2_snow[:, layer - 1, :], o2_snow[:, layer, :])
        )
        ch4_snow = ch4_snow.at[:, layer, :].set(
            jnp.where(mask, coefficients.alphaCH4_snow[:, layer - 1, :] * ch4_snow[:, layer - 1, :] + coefficients.betaCH4_snow[:, layer - 1, :], ch4_snow[:, layer, :])
        )
    for layer in range(1, ndeep):
        o2_soil = o2_soil.at[:, layer, :].set(
            jnp.where(mask, coefficients.alphaO2_soil[:, layer - 1, :] * o2_soil[:, layer - 1, :] + coefficients.betaO2_soil[:, layer - 1, :], o2_soil[:, layer, :])
        )
        ch4_soil = ch4_soil.at[:, layer, :].set(
            jnp.where(mask, coefficients.alphaCH4_soil[:, layer - 1, :] * ch4_soil[:, layer - 1, :] + coefficients.betaCH4_soil[:, layer - 1, :], ch4_soil[:, layer, :])
        )
    return DeepCarbonGasDiffusionStateResult(o2_snow, ch4_snow, o2_soil, ch4_soil)


def deep_carbon_plant_transport_step(
    CH4_soil,
    O2_soil,
    totporCH4_soil,
    totporO2_soil,
    z_root,
    rootlev,
    Tref,
    hslong,
    zi_soil,
    zf_soil,
    tprof,
    veget_mask,
    *,
    time_step_seconds,
    Tgr,
    refdep,
    zero_celsius=ZERO_CELSIUS,
    wO2=32.0,
    wCH4=16.0,
) -> DeepCarbonPlantTransportResult:
    """Apply plant-mediated methane transport from OK_PC ``traMplan``."""

    ch4 = jnp.asarray(CH4_soil)
    o2 = jnp.asarray(O2_soil)
    tp_ch4 = jnp.asarray(totporCH4_soil)
    tp_o2 = jnp.asarray(totporO2_soil)
    zroot = jnp.asarray(z_root)
    root_level = jnp.asarray(rootlev)
    tref = jnp.asarray(Tref)
    hum = jnp.asarray(hslong)
    zi = jnp.asarray(zi_soil)
    zf = jnp.asarray(zf_soil)
    temp = jnp.asarray(tprof)
    mask = jnp.asarray(veget_mask, dtype=bool)
    if ch4.ndim != 3:
        raise ValueError("CH4_soil must have shape (npts, ndeep, nvm)")
    npts, ndeep, nvm = ch4.shape
    if o2.shape != ch4.shape or tp_ch4.shape != ch4.shape or tp_o2.shape != ch4.shape or hum.shape != ch4.shape or temp.shape != ch4.shape:
        raise ValueError("soil gas, porosity, humidity, and tprof arrays must share shape")
    if zroot.shape != (npts, nvm) or root_level.shape != (npts, nvm) or tref.shape != (npts, nvm) or mask.shape != (npts, nvm):
        raise ValueError("z_root, rootlev, Tref, and veget_mask must have shape (npts, nvm)")
    if zi.shape != (ndeep,) or zf.shape != (ndeep + 1,):
        raise ValueError("zi_soil/zf_soil shapes must match soil layers")
    ref_candidates = [idx for idx in range(ndeep) if bool(zi[idx] > refdep)]
    if ref_candidates:
        reflev_fortran = ref_candidates[0]
        if reflev_fortran == 0:
            raise ValueError("traMplan source would select reflev=0; choose refdep below the first soil level")
        ref_index = reflev_fortran - 1
    else:
        ref_index = ndeep - 1
    tref = jnp.where(mask, temp[:, ref_index, :] - zero_celsius, tref)
    tmat = Tgr + 10.0
    fgrow = jnp.where(
        tref <= Tgr,
        0.0,
        jnp.where(tref >= tmat, 4.0, 4.0 * (1.0 - ((tmat - tref) / (tmat - Tgr)) ** 2)),
    )
    flupmt = jnp.zeros((npts, nvm), dtype=ch4.dtype)
    for ip in range(npts):
        for iv in range(nvm):
            if bool((zroot[ip, iv] > 0.0) & mask[ip, iv]):
                for layer in range(int(root_level[ip, iv])):
                    froot = jnp.maximum(2.0 * (zroot[ip, iv] - zi[layer]) / zroot[ip, iv], 0.0)
                    dch4 = 0.01 * 10.0 * froot * fgrow[ip, iv] * hum[ip, layer, iv] * ch4[ip, layer, iv]
                    dch4 = jnp.where(dch4 < 0.0, 0.0, dch4)
                    dch4 = jnp.minimum(dch4, ch4[ip, layer, iv])
                    ch4 = ch4.at[ip, layer, iv].add(-dch4)
                    do2 = dch4 * 0.5 * wO2 / wCH4 * tp_ch4[ip, layer, iv] / tp_o2[ip, layer, iv]
                    o2 = o2.at[ip, layer, iv].set(jnp.where(do2 < o2[ip, layer, iv], o2[ip, layer, iv] - do2, o2[ip, layer, iv]))
                    flupmt = flupmt.at[ip, iv].add(
                        dch4 * tp_ch4[ip, layer, iv] / time_step_seconds * 0.5 * (zf[layer + 1] - zf[layer])
                    )
    return DeepCarbonPlantTransportResult(CH4_soil=ch4, O2_soil=o2, Tref=tref, flupmt=flupmt)


def deep_carbon_ebullition_step(
    CH4_soil,
    totporCH4_soil,
    hslong,
    zi_soil,
    zf_soil,
    veget_mask,
    *,
    time_step_seconds,
    BunsenCH4=BUNSEN_CH4,
    ebuthr=EBUTHR,
) -> DeepCarbonEbullitionResult:
    """Apply the OK_PC methane ebullition threshold/removal step."""

    ch4 = jnp.asarray(CH4_soil)
    totpor = jnp.asarray(totporCH4_soil)
    hum = jnp.asarray(hslong)
    zi = jnp.asarray(zi_soil)
    zf = jnp.asarray(zf_soil)
    mask = jnp.asarray(veget_mask, dtype=bool)
    if ch4.ndim != 3:
        raise ValueError("CH4_soil must have shape (npts, ndeep, nvm)")
    npts, ndeep, nvm = ch4.shape
    if totpor.shape != ch4.shape or hum.shape != ch4.shape:
        raise ValueError("totporCH4_soil and hslong must match CH4_soil")
    if zi.shape != (ndeep,) or zf.shape != (ndeep + 1,):
        raise ValueError("zi_soil/zf_soil shapes must match soil layers")
    if mask.shape != (npts, nvm):
        raise ValueError("veget_mask must have shape (npts, nvm)")
    febul = jnp.zeros((npts, nvm), dtype=ch4.dtype)
    threshold = 12.0e-3 / BunsenCH4
    eps = jnp.asarray(jnp.finfo(jnp.float32).eps, dtype=ch4.dtype)
    for ip in range(npts):
        for iv in range(nvm):
            if bool(mask[ip, iv]) and bool(hum[ip, 0, iv] > ebuthr):
                for layer in range(ndeep - 1, -1, -1):
                    ch4d = ch4[ip, layer, iv] - threshold
                    if bool(ch4d > eps):
                        ch4 = ch4.at[ip, layer, iv].add(-ch4d)
                        febul = febul.at[ip, iv].add(ch4d * totpor[ip, layer, iv] * (zf[layer + 1] - zf[layer]) / time_step_seconds)
    return DeepCarbonEbullitionResult(CH4_soil=ch4, febul=febul)


def soilcarbon_leak_tf_doc_inputs(
    precip2ground,
    precip2canopy,
    veget_max,
    biomass,
    is_tree,
    *,
    ok_tf_doc: bool,
    dt_days,
    conc_doc_rain=3.02,
    doc_incr_per_leaf_m2=0.00092,
) -> TfDocInputs:
    """Compute throughfall/wet/dry DOC inputs for ``soilcarbon_leak``.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``,
    ``soilcarbon_leak`` lines 1206-1233. Non-carbon elements remain zero.
    """

    precip2ground = jnp.asarray(precip2ground)
    precip2canopy = jnp.asarray(precip2canopy)
    veget_max = jnp.asarray(veget_max)
    biomass = jnp.asarray(biomass)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    npts, nvm = veget_max.shape
    nelements = biomass.shape[-1]
    dry_dep_canopy = jnp.zeros((npts, nvm, nelements), dtype=biomass.dtype)
    doc_precip2ground = jnp.zeros_like(dry_dep_canopy)
    doc_precip2canopy = jnp.zeros_like(dry_dep_canopy)
    doc_canopy2ground = jnp.zeros_like(dry_dep_canopy)
    bio_frac = jnp.sum(veget_max[:, 1:], axis=1)
    if not ok_tf_doc:
        return TfDocInputs(dry_dep_canopy, doc_precip2ground, doc_precip2canopy, doc_canopy2ground, bio_frac)

    active_pft = jnp.arange(nvm) > 0
    doc_precip2canopy = doc_precip2canopy.at[:, :, ICARBON].set(
        jnp.where(active_pft[None, :], precip2canopy * conc_doc_rain * 1e-3, 0.0)
    )
    doc_precip2ground = doc_precip2ground.at[:, :, ICARBON].set(
        jnp.where(active_pft[None, :], precip2ground * conc_doc_rain * 1e-3, 0.0)
    )
    dry = doc_incr_per_leaf_m2 * dt_days * biomass[:, :, ILEAF, ICARBON] * veget_max
    dry_dep_canopy = dry_dep_canopy.at[:, :, ICARBON].set(
        jnp.where(is_tree[None, :] & (veget_max > 0.0), dry, 0.0)
    )
    return TfDocInputs(dry_dep_canopy, doc_precip2ground, doc_precip2canopy, doc_canopy2ground, bio_frac)


def soilcarbon_cryoturbation_coefficients(
    altmax_ind,
    carbon_32l,
    doc,
    litter_below,
    altmax_lastyear,
    fixed_cryoturbation_depth,
    veget_mask,
    zi_soil,
    zf_soil_b,
    *,
    dt_seconds,
    diff_k_const,
    bio_diff_k_const,
    use_new_cryoturbation: bool = False,
    cryoturbation_method: int = 4,
    max_cryoturb_alt=3.0,
    min_cryoturb_alt=0.01,
    use_fixed_cryoturbation_depth: bool = False,
    bioturbation_depth=2.0,
) -> SoilcarbonCryoturbationCoefficients:
    """Compute ``cryoturbate_doc_POC`` saved diffusion coefficients.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``, subroutine
    ``cryoturbate_doc_POC``, action ``coefficients``, lines 3123-3329.
    ``altmax_ind`` uses the Fortran numeric convention: one-based layer count
    values from ``altcalc_DOC``, not Python zero-based array indices.
    ``zi_soil`` follows the source ``zi_soil(1:ndeep)`` vector; only
    ``zf_soil_b`` carries a zero upper boundary.
    """

    altmax_ind = jnp.asarray(altmax_ind)
    carbon_32l = jnp.asarray(carbon_32l)
    doc = jnp.asarray(doc)
    litter_below = jnp.asarray(litter_below)
    altmax_lastyear = jnp.asarray(altmax_lastyear)
    fixed_cryoturbation_depth = jnp.asarray(fixed_cryoturbation_depth)
    veget_mask = jnp.asarray(veget_mask, dtype=bool)
    zi_soil = jnp.asarray(zi_soil)
    zf_soil_b = jnp.asarray(zf_soil_b)

    if carbon_32l.ndim != 4:
        raise ValueError("carbon_32l must have shape (npts, ncarb, nvm, ndeep)")
    npts, ncarb, nvm, ndeep = carbon_32l.shape
    if doc.shape[:5] != (npts, nvm, ndeep, NDOC, NPOOL):
        raise ValueError("doc must have shape (npts, nvm, ndeep, ndoc, npool, nelements)")
    if litter_below.shape[:4] != (npts, NLITT, nvm, ndeep):
        raise ValueError("litter_below must have shape (npts, nlitt, nvm, ndeep, nelements)")
    if altmax_ind.shape != (npts, nvm):
        raise ValueError("altmax_ind must have shape (npts, nvm)")
    if altmax_lastyear.shape != (npts, nvm) or fixed_cryoturbation_depth.shape != (npts, nvm):
        raise ValueError("altmax_lastyear and fixed_cryoturbation_depth must have shape (npts, nvm)")
    if veget_mask.shape != (npts, nvm):
        raise ValueError("veget_mask must have shape (npts, nvm)")
    if zi_soil.shape[0] != ndeep:
        raise ValueError("zi_soil must have length ndeep to match Fortran zi_soil(1:ndeep)")
    if zf_soil_b.shape[0] != ndeep + 1:
        raise ValueError("zf_soil_b must have length ndeep + 1 to match Fortran zf_soil_B(0:ndeep)")

    dtype = jnp.result_type(carbon_32l, doc, litter_below, zi_soil)
    layer_depths = zi_soil
    layer_no = jnp.arange(1, ndeep + 1)
    cryoturb_location = (altmax_lastyear < max_cryoturb_alt) & (altmax_lastyear >= min_cryoturb_alt) & veget_mask
    bioturb_location = (altmax_lastyear >= max_cryoturb_alt) & veget_mask
    cryoturbation_depth = jnp.where(use_fixed_cryoturbation_depth, fixed_cryoturbation_depth, altmax_lastyear)

    diff_k = jnp.zeros((npts, ndeep, nvm), dtype=dtype)
    for ip in range(npts):
        for iv in range(nvm):
            if bool(cryoturb_location[ip, iv]):
                depth = cryoturbation_depth[ip, iv]
                if use_new_cryoturbation:
                    if int(cryoturbation_method) == 1:
                        profile = jnp.where(
                            layer_depths <= depth,
                            diff_k_const,
                            diff_k_const * (1.0 - jnp.maximum(jnp.minimum((layer_depths / depth) - 1.0, 1.0), 0.0)),
                        )
                    elif int(cryoturbation_method) == 2:
                        profile = jnp.where(
                            layer_depths <= depth,
                            diff_k_const,
                            diff_k_const * jnp.exp(-jnp.maximum((layer_depths / depth) - 1.0, 0.0)),
                        )
                    elif int(cryoturbation_method) == 3:
                        profile = diff_k_const * jnp.exp(-(layer_depths / depth))
                    elif int(cryoturbation_method) == 4:
                        profile = jnp.where(
                            layer_depths <= depth,
                            diff_k_const,
                            diff_k_const
                            * (
                                1.0
                                - jnp.maximum(
                                    jnp.minimum((layer_depths - depth) / (2.0 * depth), 1.0),
                                    0.0,
                                )
                            ),
                        )
                        profile = jnp.where(layer_depths > max_cryoturb_alt, 0.0, profile)
                    elif int(cryoturbation_method) == 5:
                        profile = jnp.where(
                            layer_depths <= depth,
                            diff_k_const,
                            diff_k_const
                            * (
                                1.0
                                - jnp.maximum(
                                    jnp.minimum((layer_depths - depth) / (3.0 - depth), 1.0),
                                    0.0,
                                )
                            ),
                        )
                    else:
                        raise ValueError("cryoturbation_method must be one of 1, 2, 3, 4, 5")
                else:
                    aind = int(altmax_ind[ip, iv])
                    if aind < ndeep - 3:
                        profile = jnp.where(layer_no <= aind, diff_k_const, 0.0)
                        profile = profile.at[aind].set(diff_k_const / 10.0)
                        profile = profile.at[aind + 1].set(diff_k_const / 100.0)
                    else:
                        profile = jnp.where(layer_no <= aind - 3, diff_k_const, 0.0)
                        profile = profile.at[aind - 3].set(diff_k_const / 10.0)
                        profile = profile.at[aind - 2].set(diff_k_const / 100.0)
                diff_k = diff_k.at[ip, :, iv].set(profile)
            elif bool(bioturb_location[ip, iv]):
                diff_k = diff_k.at[ip, :, iv].set(jnp.where(layer_depths <= bioturbation_depth, bio_diff_k_const, 0.0))

    active = cryoturb_location | bioturb_location
    xc = jnp.zeros((npts, ndeep, nvm), dtype=dtype)
    xd = jnp.zeros((npts, ndeep, nvm), dtype=dtype)
    layer_thickness = zf_soil_b[1:] - zf_soil_b[:-1]
    xc = jnp.where(active[:, None, :], layer_thickness[None, :, None] / dt_seconds, xc)
    if ndeep > 1:
        interface_thickness = zi_soil[1:] - zi_soil[:-1]
        xd_inner = diff_k[:, : ndeep - 1, :] / interface_thickness[None, :, None]
        xd = xd.at[:, : ndeep - 1, :].set(jnp.where(active[:, None, :], xd_inner, xd[:, : ndeep - 1, :]))

    alpha_layer = jnp.zeros((npts, ndeep, nvm), dtype=dtype)
    beta_c = jnp.zeros((npts, ncarb, nvm, ndeep), dtype=dtype)
    beta_doc = jnp.zeros((npts, nvm, ndeep, NDOC, NPOOL), dtype=dtype)
    beta_litter = jnp.zeros((npts, NLITT, nvm, ndeep), dtype=dtype)
    if ndeep > 1:
        xe = jnp.where(active, xc[:, ndeep - 1, :] + xd[:, ndeep - 2, :], 1.0)
        alpha_bottom = xd[:, ndeep - 2, :] / xe
        alpha_layer = alpha_layer.at[:, ndeep - 2, :].set(jnp.where(active, alpha_bottom, 0.0))
        beta_c = beta_c.at[:, :, :, ndeep - 2].set(
            jnp.where(active[:, None, :], xc[:, None, ndeep - 1, :] * carbon_32l[:, :, :, ndeep - 1] / xe[:, None, :], 0.0)
        )
        beta_doc = beta_doc.at[:, :, ndeep - 2, :, :].set(
            jnp.where(
                active[:, :, None, None],
                xc[:, ndeep - 1, :, None, None] * doc[:, :, ndeep - 1, :, :, ICARBON] / xe[:, :, None, None],
                0.0,
            )
        )
        beta_litter = beta_litter.at[:, :, :, ndeep - 2].set(
            jnp.where(
                active[:, None, :],
                xc[:, None, ndeep - 1, :] * litter_below[:, :, :, ndeep - 1, ICARBON] / xe[:, None, :],
                0.0,
            )
        )

        for layer in range(ndeep - 3, -1, -1):
            xe = jnp.where(
                active,
                xc[:, layer + 1, :] + (1.0 - alpha_layer[:, layer + 1, :]) * xd[:, layer + 1, :] + xd[:, layer, :],
                1.0,
            )
            alpha_here = xd[:, layer, :] / xe
            alpha_layer = alpha_layer.at[:, layer, :].set(jnp.where(active, alpha_here, 0.0))
            beta_c = beta_c.at[:, :, :, layer].set(
                jnp.where(
                    active[:, None, :],
                    (
                        xc[:, None, layer + 1, :] * carbon_32l[:, :, :, layer + 1]
                        + xd[:, None, layer + 1, :] * beta_c[:, :, :, layer + 1]
                    )
                    / xe[:, None, :],
                    0.0,
                )
            )
            beta_doc = beta_doc.at[:, :, layer, :, :].set(
                jnp.where(
                    active[:, :, None, None],
                    (
                        xc[:, layer + 1, :, None, None] * doc[:, :, layer + 1, :, :, ICARBON]
                        + xd[:, layer + 1, :, None, None] * beta_doc[:, :, layer + 1, :, :]
                    )
                    / xe[:, :, None, None],
                    0.0,
                )
            )
            beta_litter = beta_litter.at[:, :, :, layer].set(
                jnp.where(
                    active[:, None, :],
                    (
                        xc[:, None, layer + 1, :] * litter_below[:, :, :, layer + 1, ICARBON]
                        + xd[:, None, layer + 1, :] * beta_litter[:, :, :, layer + 1]
                    )
                    / xe[:, None, :],
                    0.0,
                )
            )

    alpha_by_pft = jnp.transpose(alpha_layer, (0, 2, 1))
    alpha_c = jnp.repeat(alpha_by_pft[:, None, :, :], ncarb, axis=1)
    alpha_doc = jnp.repeat(alpha_by_pft[:, :, :, None, None], NDOC, axis=3)
    alpha_doc = jnp.repeat(alpha_doc, NPOOL, axis=4)
    alpha_litter = jnp.repeat(alpha_by_pft[:, None, :, :], NLITT, axis=1)
    return SoilcarbonCryoturbationCoefficients(
        cryoturb_location=cryoturb_location,
        bioturb_location=bioturb_location,
        cryoturbation_depth=cryoturbation_depth,
        diff_k=diff_k,
        xc_cryoturb=xc,
        xd_cryoturb=xd,
        alpha_c=alpha_c,
        beta_c=beta_c,
        alpha_doc=alpha_doc,
        beta_doc=beta_doc,
        alpha_litter_below=alpha_litter,
        beta_litter_below=beta_litter,
    )


def soilcarbon_cryoturbation_diffuse(
    carbon_32l,
    doc,
    litter_below,
    coefficients: SoilcarbonCryoturbationCoefficients,
    zi_soil,
) -> SoilcarbonCryoturbationDiffuseResult:
    """Apply ``cryoturbate_doc_POC`` action ``diffuse`` with saved coefficients.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``,
    ``cryoturbate_doc_POC`` lines 2881-3112. The commented active-layer mass
    correction block in the source is intentionally not applied.
    """

    carbon_work = jnp.asarray(carbon_32l)
    doc_work = jnp.asarray(doc)
    litter_work = jnp.asarray(litter_below)
    zi_soil = jnp.asarray(zi_soil)
    active = coefficients.cryoturb_location | coefficients.bioturb_location
    npts, ncarb, nvm, ndeep = carbon_work.shape
    if zi_soil.shape[0] != ndeep:
        raise ValueError("zi_soil must have length ndeep")
    mu_soil = zi_soil[0] / (zi_soil[1] - zi_soil[0])

    for ip in range(npts):
        for iv in range(nvm):
            if bool(active[ip, iv]):
                top_c = (carbon_work[ip, :, iv, 0] + mu_soil * coefficients.beta_c[ip, :, iv, 0]) / (
                    1.0 + mu_soil * (1.0 - coefficients.alpha_c[ip, :, iv, 0])
                )
                carbon_work = carbon_work.at[ip, :, iv, 0].set(top_c)
                top_doc = (
                    doc_work[ip, iv, 0, :, :, ICARBON] + mu_soil * coefficients.beta_doc[ip, iv, 0, :, :]
                ) / (1.0 + mu_soil * (1.0 - coefficients.alpha_doc[ip, iv, 0, :, :]))
                doc_work = doc_work.at[ip, iv, 0, :, :, ICARBON].set(top_doc)
                top_litter = (
                    litter_work[ip, :, iv, 0, ICARBON] + mu_soil * coefficients.beta_litter_below[ip, :, iv, 0]
                ) / (1.0 + mu_soil * (1.0 - coefficients.alpha_litter_below[ip, :, iv, 0]))
                litter_work = litter_work.at[ip, :, iv, 0, ICARBON].set(top_litter)
                for layer in range(1, ndeep):
                    next_c = (
                        coefficients.alpha_c[ip, :, iv, layer - 1] * carbon_work[ip, :, iv, layer - 1]
                        + coefficients.beta_c[ip, :, iv, layer - 1]
                    )
                    carbon_work = carbon_work.at[ip, :, iv, layer].set(next_c)
                    next_doc = (
                        coefficients.alpha_doc[ip, iv, layer - 1, :, :] * doc_work[ip, iv, layer - 1, :, :, ICARBON]
                        + coefficients.beta_doc[ip, iv, layer - 1, :, :]
                    )
                    doc_work = doc_work.at[ip, iv, layer, :, :, ICARBON].set(next_doc)
                    next_litter = (
                        coefficients.alpha_litter_below[ip, :, iv, layer - 1]
                        * litter_work[ip, :, iv, layer - 1, ICARBON]
                        + coefficients.beta_litter_below[ip, :, iv, layer - 1]
                    )
                    litter_work = litter_work.at[ip, :, iv, layer, ICARBON].set(next_litter)

    return SoilcarbonCryoturbationDiffuseResult(carbon_work, doc_work, litter_work)


def soilcarbon_cryoturbation_cycle(
    carbon_32l,
    doc,
    litter_below,
    previous_coefficients,
    altmax_ind,
    altmax_lastyear,
    fixed_cryoturbation_depth,
    veget_mask,
    zi_soil,
    zf_soil_b,
    *,
    dt_seconds,
    diff_k_const,
    bio_diff_k_const,
    use_new_cryoturbation: bool = False,
    cryoturbation_method: int = 4,
    max_cryoturb_alt=3.0,
    min_cryoturb_alt=0.01,
    use_fixed_cryoturbation_depth: bool = False,
    bioturbation_depth=2.0,
) -> tuple[SoilcarbonCryoturbationDiffuseResult, SoilcarbonCryoturbationCoefficients]:
    """Run the source-order cryoturbation diffuse/coefficients pair.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``,
    ``soilcarbon_leak`` lines 1348-1368 and ``cryoturbate_doc_POC`` lines
    2881-3329. The source converts pools from layer-integrated stocks to
    concentration, applies the saved ``diffuse`` coefficients when available,
    recomputes ``coefficients`` from the post-diffuse concentration state, then
    converts pools back to layer-integrated stocks.
    """

    carbon_32l = jnp.asarray(carbon_32l)
    doc = jnp.asarray(doc)
    litter_below = jnp.asarray(litter_below)
    zf_soil_b = jnp.asarray(zf_soil_b)
    ndeep = carbon_32l.shape[3]
    if zf_soil_b.shape[0] != ndeep + 1:
        raise ValueError("zf_soil_b must have length ndeep + 1")
    thickness = zf_soil_b[1:] - zf_soil_b[:-1]
    carbon_conc = carbon_32l / thickness[None, None, None, :]
    doc_conc = doc / thickness[None, None, :, None, None, None]
    litter_conc = litter_below / thickness[None, None, None, :, None]

    if previous_coefficients is not None:
        diffuse = soilcarbon_cryoturbation_diffuse(carbon_conc, doc_conc, litter_conc, previous_coefficients, zi_soil)
        carbon_conc = diffuse.carbon_32l
        doc_conc = diffuse.doc
        litter_conc = diffuse.litter_below

    new_coefficients = soilcarbon_cryoturbation_coefficients(
        altmax_ind,
        carbon_conc,
        doc_conc,
        litter_conc,
        altmax_lastyear,
        fixed_cryoturbation_depth,
        veget_mask,
        zi_soil,
        zf_soil_b,
        dt_seconds=dt_seconds,
        diff_k_const=diff_k_const,
        bio_diff_k_const=bio_diff_k_const,
        use_new_cryoturbation=use_new_cryoturbation,
        cryoturbation_method=cryoturbation_method,
        max_cryoturb_alt=max_cryoturb_alt,
        min_cryoturb_alt=min_cryoturb_alt,
        use_fixed_cryoturbation_depth=use_fixed_cryoturbation_depth,
        bioturbation_depth=bioturbation_depth,
    )
    result = SoilcarbonCryoturbationDiffuseResult(
        carbon_32l=carbon_conc * thickness[None, None, None, :],
        doc=doc_conc * thickness[None, None, :, None, None, None],
        litter_below=litter_conc * thickness[None, None, None, :, None],
    )
    return result, new_coefficients


def deep_carbon_vertical_integral_step(
    deepC_a,
    deepC_s,
    deepC_p,
    zf_soil,
    veget_mask,
    *,
    maxdepth: float = 2.0,
) -> DeepCarbonVerticalIntegralResult:
    """Integrate OK_PC deep carbon pools over full and surface depths.

    Fortran provenance: ``src_stomate/stomate_permafrost_soilcarbon.f90``,
    subroutine ``calc_vert_int_soil_carbon``, lines 4425-4463. The source
    ``carbon_surf`` comment says 1 m, but the executable parameter is
    ``maxdepth=2.`` and is preserved here.
    """

    deep_a = jnp.asarray(deepC_a)
    deep_s = jnp.asarray(deepC_s)
    deep_p = jnp.asarray(deepC_p)
    zf = jnp.asarray(zf_soil)
    mask = jnp.asarray(veget_mask, dtype=bool)
    if deep_a.ndim != 3:
        raise ValueError("deepC_a must have shape (npts, ndeep, nvm)")
    if deep_s.shape != deep_a.shape or deep_p.shape != deep_a.shape:
        raise ValueError("deepC_s and deepC_p must match deepC_a shape")
    npts, ndeep, nvm = deep_a.shape
    if zf.shape != (ndeep + 1,):
        raise ValueError("zf_soil must have length ndeep + 1")
    if mask.shape != (npts, nvm):
        raise ValueError("veget_mask must have shape (npts, nvm)")

    layer_thickness = zf[1:] - zf[:-1]
    active = jnp.sum(deep_a * layer_thickness[None, :, None], axis=1)
    slow = jnp.sum(deep_s * layer_thickness[None, :, None], axis=1)
    passive = jnp.sum(deep_p * layer_thickness[None, :, None], axis=1)
    carbon = jnp.zeros((npts, NCARB, nvm), dtype=deep_a.dtype)
    carbon = carbon.at[:, IACTIVE, :].set(jnp.where(mask, active, 0.0))
    carbon = carbon.at[:, ISLOW, :].set(jnp.where(mask, slow, 0.0))
    carbon = carbon.at[:, IPASSIVE, :].set(jnp.where(mask, passive, 0.0))

    surface_thickness = jnp.where(
        zf[:-1] < maxdepth,
        jnp.minimum(maxdepth, zf[1:]) - zf[:-1],
        0.0,
    )
    surf_active = jnp.sum(deep_a * surface_thickness[None, :, None], axis=1)
    surf_slow = jnp.sum(deep_s * surface_thickness[None, :, None], axis=1)
    surf_passive = jnp.sum(deep_p * surface_thickness[None, :, None], axis=1)
    carbon_surf = jnp.zeros((npts, NCARB, nvm), dtype=deep_a.dtype)
    carbon_surf = carbon_surf.at[:, IACTIVE, :].set(jnp.where(mask, surf_active, 0.0))
    carbon_surf = carbon_surf.at[:, ISLOW, :].set(jnp.where(mask, surf_slow, 0.0))
    carbon_surf = carbon_surf.at[:, IPASSIVE, :].set(jnp.where(mask, surf_passive, 0.0))

    return DeepCarbonVerticalIntegralResult(carbon=carbon, carbon_surf=carbon_surf)


def deep_carbon_input_step(
    deepC_a,
    deepC_s,
    deepC_p,
    soilc_in,
    z_root,
    altmax,
    veget_mask,
    zi_soil,
    zf_soil,
    rprof,
    *,
    time_step_seconds,
    no_pfrost_decomp: bool = False,
    new_carbinput_intdepzlit: bool = False,
    correct_carboninput_vertprof: bool = True,
    finerootdepthratio=0.5,
    altrootratio=0.5,
    maxaltmax=2.0,
    one_day=86400.0,
    epsilon=None,
) -> DeepCarbonInputResult:
    """Distribute legacy daily carbon input into OK_PC deep-carbon layers.

    Fortran provenance: ``src_stomate/stomate_permafrost_soilcarbon.f90``,
    subroutine ``carbinput``, lines 3184-3410. This covers the process algebra
    and state writeback; diagnostic file output under the source ``check`` flag
    is intentionally excluded from the model kernel.
    """

    deep_a = jnp.asarray(deepC_a)
    deep_s = jnp.asarray(deepC_s)
    deep_p = jnp.asarray(deepC_p)
    soilc = jnp.asarray(soilc_in)
    zroot = jnp.asarray(z_root)
    alt = jnp.asarray(altmax)
    mask = jnp.asarray(veget_mask, dtype=bool)
    zi = jnp.asarray(zi_soil)
    zf = jnp.asarray(zf_soil)
    rprof_arr = jnp.asarray(rprof)
    if deep_a.ndim != 3:
        raise ValueError("deepC_a must have shape (npts, ndeep, nvm)")
    if deep_s.shape != deep_a.shape or deep_p.shape != deep_a.shape:
        raise ValueError("deepC_s and deepC_p must match deepC_a shape")
    npts, ndeep, nvm = deep_a.shape
    if soilc.shape != (npts, NCARB, nvm):
        raise ValueError("soilc_in must have shape (npts, NCARB, nvm)")
    if zroot.shape != (npts, nvm) or alt.shape != (npts, nvm) or rprof_arr.shape != (npts, nvm):
        raise ValueError("z_root, altmax, and rprof must have shape (npts, nvm)")
    if mask.shape != (npts, nvm):
        raise ValueError("veget_mask must have shape (npts, nvm)")
    if zi.shape != (ndeep,):
        raise ValueError("zi_soil must have shape (ndeep,)")
    if zf.shape != (ndeep + 1,):
        raise ValueError("zf_soil must have length ndeep + 1")
    eps = jnp.asarray(jnp.finfo(jnp.float32).eps if epsilon is None else epsilon, dtype=deep_a.dtype)

    dc_litter_z = jnp.zeros((npts, NCARB, ndeep, nvm), dtype=deep_a.dtype)
    if bool(no_pfrost_decomp):
        return DeepCarbonInputResult(deepC_a=deep_a, deepC_s=deep_s, deepC_p=deep_p, dc_litter_z=dc_litter_z)

    soilc_in_ts = soilc * jnp.asarray(time_step_seconds, dtype=soilc.dtype) / jnp.asarray(one_day, dtype=soilc.dtype)
    if bool(new_carbinput_intdepzlit):
        z_lit = jnp.minimum(rprof_arr * finerootdepthratio, alt * altrootratio)
        intdep = jnp.minimum(alt, maxaltmax)
    else:
        z_lit = zroot
        intdep = zroot
    min_depth = zi[1]
    intdep = jnp.where(intdep < min_depth, min_depth + eps, intdep)
    z_lit = jnp.where(z_lit < min_depth, min_depth, z_lit)

    layer_active = (zi[None, :, None] < intdep[:, None, :]) & mask[:, None, :]
    denominator = z_lit[:, None, :] * (1.0 - jnp.exp(-intdep[:, None, :] / z_lit[:, None, :]))
    profile = jnp.exp(-zi[None, :, None] / z_lit[:, None, :]) / denominator
    dc_litter_z = jnp.where(layer_active[:, None, :, :], soilc_in_ts[:, :, None, :] * profile[:, None, :, :], 0.0)

    thickness = zf[1:] - zf[:-1]
    if bool(correct_carboninput_vertprof):
        dc_litter = jnp.sum(dc_litter_z * thickness[None, None, :, None], axis=2)
        correction = jnp.where((dc_litter > eps) & mask[:, None, :], soilc_in_ts / dc_litter, 0.0)
        dc_litter_z = jnp.where(mask[:, None, None, :], correction[:, :, None, :] * dc_litter_z, dc_litter_z)

    deep_a = deep_a + jnp.where(mask[:, None, :], dc_litter_z[:, IACTIVE, :, :], 0.0)
    deep_s = deep_s + jnp.where(mask[:, None, :], dc_litter_z[:, ISLOW, :, :], 0.0)
    deep_p = deep_p + jnp.where(mask[:, None, :], dc_litter_z[:, IPASSIVE, :, :], 0.0)
    return DeepCarbonInputResult(deepC_a=deep_a, deepC_s=deep_s, deepC_p=deep_p, dc_litter_z=dc_litter_z)


def deep_carbon_permafrost_decomp_oxic_step(
    deepC_a,
    deepC_s,
    deepC_p,
    fbact_out,
    O2_soil,
    totporO2_soil,
    clay,
    veget_mask,
    zf_soil,
    *,
    time_step_seconds,
    airvol_soil=None,
    oxlim: bool = False,
    no_pfrost_decomp: bool = False,
    ok_methane: bool = False,
    CH4_soil=None,
    totporCH4_soil=None,
    hslong=None,
    tprof=None,
    tau_CH4troph=432000.0,
    fbactratio=9.0,
    O2m=3.0,
    MG_useallCpools: bool = True,
    fslow=37.0,
    fpassive=1617.45,
    wC=12.0,
    wO2=32.0,
    wCH4=16.0,
    BunsenO2=0.038,
    perma_peat: bool = False,
    cmax_peat=None,
    is_peat=None,
    frac1=0.95,
    frac2=0.05,
    min_stomate=0.0,
) -> DeepCarbonPermafrostDecompResult:
    """Run the source ``permafrost_decomp`` carbon-pool update.

    Fortran provenance: ``src_stomate/stomate_permafrost_soilcarbon.f90``,
    subroutine ``permafrost_decomp``, lines 3920-4403. This kernel covers
    active/slow/passive residence-time scaling, optional O2 limitation, O2
    drawdown, ``deltaC1_*`` diagnostics, optional methanogenesis/trophy,
    reservoir transfers, and optional PERMA_PEAT redistribution.
    """

    deep_a = jnp.asarray(deepC_a)
    deep_s = jnp.asarray(deepC_s)
    deep_p = jnp.asarray(deepC_p)
    fbact = jnp.asarray(fbact_out)
    o2 = jnp.asarray(O2_soil)
    totpor_o2 = jnp.asarray(totporO2_soil)
    clay_arr = jnp.asarray(clay)
    mask = jnp.asarray(veget_mask, dtype=bool)
    zf = jnp.asarray(zf_soil)
    if deep_a.ndim != 3:
        raise ValueError("deepC_a must have shape (npts, ndeep, nvm)")
    if deep_s.shape != deep_a.shape or deep_p.shape != deep_a.shape:
        raise ValueError("deepC_s and deepC_p must match deepC_a shape")
    if fbact.shape != deep_a.shape or o2.shape != deep_a.shape or totpor_o2.shape != deep_a.shape:
        raise ValueError("fbact_out, O2_soil, and totporO2_soil must match deepC_a shape")
    npts, ndeep, nvm = deep_a.shape
    if clay_arr.shape != (npts,):
        raise ValueError("clay must have shape (npts,)")
    if mask.shape != (npts, nvm):
        raise ValueError("veget_mask must have shape (npts, nvm)")
    if zf.shape != (ndeep + 1,):
        raise ValueError("zf_soil must have length ndeep + 1")
    if bool(oxlim):
        if airvol_soil is None:
            raise ValueError("airvol_soil is required when oxlim=True")
        airvol = jnp.asarray(airvol_soil)
        if airvol.shape != deep_a.shape:
            raise ValueError("airvol_soil must match deepC_a shape")
    else:
        airvol = jnp.ones_like(deep_a)
    if bool(ok_methane):
        required = {
            "CH4_soil": CH4_soil,
            "totporCH4_soil": totporCH4_soil,
            "hslong": hslong,
            "tprof": tprof,
        }
        missing = tuple(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(f"ok_methane requires explicit {', '.join(missing)}")
        ch4 = jnp.asarray(CH4_soil)
        totpor_ch4 = jnp.asarray(totporCH4_soil)
        hslong_arr = jnp.asarray(hslong)
        tprof_arr = jnp.asarray(tprof)
        if ch4.shape != deep_a.shape or totpor_ch4.shape != deep_a.shape or hslong_arr.shape != deep_a.shape or tprof_arr.shape != deep_a.shape:
            raise ValueError("CH4_soil, totporCH4_soil, hslong, and tprof must match deepC_a shape")
    else:
        ch4 = None
        totpor_ch4 = None
        hslong_arr = None
        tprof_arr = None

    zero = jnp.zeros_like(deep_a)
    if bool(no_pfrost_decomp):
        return DeepCarbonPermafrostDecompResult(
            deepC_a=deep_a,
            deepC_s=deep_s,
            deepC_p=deep_p,
            O2_soil=o2,
            CH4_soil=ch4,
            deltaC1_a=zero,
            deltaC1_s=zero,
            deltaC1_p=zero,
            deltaCH4=zero,
            deltaCH4g=zero,
            deltaC2=zero,
            deltaC3=zero,
            nadd_soil=zero,
            perma_peat=None,
        )

    active = mask[:, None, :]
    timestep = jnp.asarray(time_step_seconds, dtype=deep_a.dtype)
    fbact_a = jnp.maximum(fbact, timestep)
    fbact_s = fbact_a * jnp.asarray(fslow, dtype=deep_a.dtype)
    fbact_p = fbact_a * jnp.asarray(fpassive, dtype=deep_a.dtype)
    active_clay = 1.0 - 0.75 * clay_arr[:, None, None]
    frac_a_to_p = jnp.asarray(0.004, dtype=deep_a.dtype)
    frac_a_to_s = 1.0 - (0.85 - 0.68 * clay_arr[:, None, None]) - frac_a_to_p
    fr_a = 0.85 - 0.68 * clay_arr[:, None, None]
    fr_s = jnp.asarray(0.55, dtype=deep_a.dtype)
    fr_p = jnp.asarray(0.55, dtype=deep_a.dtype)

    def _decay(pool, residence, o2_current, *, apply_active_clay=False):
        raw = pool * timestep / residence
        if bool(oxlim):
            o2_cap = o2_current * airvol * jnp.asarray(wC / wO2, dtype=deep_a.dtype)
            raw = jnp.minimum(raw, o2_cap)
        if apply_active_clay:
            raw = raw * active_clay
        return jnp.where(active, raw, 0.0)

    dC_a = _decay(deep_a, fbact_a, o2, apply_active_clay=True)
    o2_after_a = jnp.maximum(o2 - (wO2 / wC) * dC_a * fr_a / totpor_o2, 0.0)
    dC_s = _decay(deep_s, fbact_s, o2_after_a)
    o2_after_s = jnp.maximum(o2_after_a - (wO2 / wC) * dC_s * fr_s / totpor_o2, 0.0)
    dC_p = _decay(deep_p, fbact_p, o2_after_s)
    o2_after_p = jnp.maximum(o2_after_s - (wO2 / wC) * dC_p * fr_p / totpor_o2, 0.0)
    o2_after_oxic = jnp.where(active, o2_after_p, o2)

    a_to_p = frac_a_to_p * dC_a
    a_to_s = frac_a_to_s * dC_a
    s_to_a = 0.42 * dC_s
    s_to_p = 0.03 * dC_s
    p_to_a = 0.45 * dC_p
    p_to_s = 0.0 * dC_p
    deep_a_work = jnp.where(active, deep_a - dC_a, deep_a)
    deep_s_work = jnp.where(active, deep_s - dC_s, deep_s)
    deep_p_work = jnp.where(active, deep_p - dC_p, deep_p)
    deltaCH4 = zero
    deltaCH4g = zero
    deltaC2 = zero
    deltaC3 = zero
    nadd_soil = zero
    o2_final = o2_after_oxic
    ch4_final = ch4
    if bool(ok_methane):
        fbactCH4_a = fbact_a * jnp.asarray(fbactratio, dtype=deep_a.dtype)
        huge = jnp.asarray(jnp.finfo(deep_a.dtype).max / 16.0, dtype=deep_a.dtype)
        fbactCH4_s = jnp.where(MG_useallCpools, fbact_s * jnp.asarray(fbactratio, dtype=deep_a.dtype), huge)
        fbactCH4_p = jnp.where(MG_useallCpools, fbact_p * jnp.asarray(fbactratio, dtype=deep_a.dtype), huge)
        meth_lim = jnp.exp(-o2_after_oxic * (1.0 + hslong_arr * (BunsenO2 - 1.0)) / O2m)
        dC_mg_a = jnp.where(active, deep_a_work * timestep / fbactCH4_a * meth_lim * active_clay, 0.0)
        dCH4g_a = dC_mg_a * fr_a * wCH4 / wC / totpor_ch4
        deep_a_work = jnp.where(active, deep_a_work - dC_mg_a, deep_a_work)
        ch4_after_mg = jnp.where(active, ch4 + dCH4g_a, ch4)
        deltaCH4g = deltaCH4g + dCH4g_a
        deltaC2 = deltaC2 + dC_mg_a * fr_a
        nadd_soil = nadd_soil + dCH4g_a / wCH4
        a_to_p = a_to_p + frac_a_to_p * dC_mg_a
        a_to_s = a_to_s + frac_a_to_s * dC_mg_a

        dC_mg_s = jnp.where(active, deep_s_work * timestep / fbactCH4_s * meth_lim, 0.0)
        dC_mg_s = jnp.where(MG_useallCpools, dC_mg_s, 0.0)
        dCH4g_s = dC_mg_s * fr_s * wCH4 / wC / totpor_ch4
        deep_s_work = jnp.where(active, deep_s_work - dC_mg_s, deep_s_work)
        ch4_after_mg = jnp.where(active, ch4_after_mg + dCH4g_s, ch4_after_mg)
        deltaCH4g = deltaCH4g + dCH4g_s
        deltaC2 = deltaC2 + dC_mg_s * fr_s
        nadd_soil = nadd_soil + dCH4g_s / wCH4
        s_to_p = s_to_p + 0.03 * dC_mg_s
        s_to_a = s_to_a + 0.42 * dC_mg_s

        dC_mg_p = jnp.where(active, deep_p_work * timestep / fbactCH4_p * meth_lim, 0.0)
        dC_mg_p = jnp.where(MG_useallCpools, dC_mg_p, 0.0)
        dCH4g_p = dC_mg_p * fr_p * wCH4 / wC / totpor_ch4
        deep_p_work = jnp.where(active, deep_p_work - dC_mg_p, deep_p_work)
        ch4_after_mg = jnp.where(active, ch4_after_mg + dCH4g_p, ch4_after_mg)
        deltaCH4g = deltaCH4g + dCH4g_p
        deltaC2 = deltaC2 + dC_mg_p * fr_p
        nadd_soil = nadd_soil + dCH4g_p / wCH4
        p_to_s = p_to_s + 0.0 * dC_mg_p
        p_to_a = p_to_a + 0.45 * dC_mg_p

        temp_c = tprof_arr - ZERO_CELSIUS
        troph_active = active & (temp_c >= 0.0)
        dCH4m = o2_after_oxic / 2.0 * wCH4 / wO2 * totpor_o2 / totpor_ch4
        dCH4_trophy = jnp.minimum(ch4_after_mg * timestep / jnp.maximum(tau_CH4troph, timestep), dCH4m)
        dCH4_trophy = jnp.where(troph_active, dCH4_trophy, 0.0)
        ch4_final = jnp.where(active, ch4_after_mg - dCH4_trophy, ch4_after_mg)
        dO2_trophy = 2.0 * dCH4_trophy * wO2 / wCH4 * totpor_ch4 / totpor_o2
        o2_final = jnp.where(active, jnp.maximum(o2_after_oxic - dO2_trophy, 0.0), o2_after_oxic)
        deltaCH4 = dCH4_trophy
        deltaC3 = dCH4_trophy / wCH4 * wC * totpor_ch4
        nadd_soil = nadd_soil - 2.0 * dCH4_trophy / wCH4

    deep_a_new = jnp.where(active, deep_a_work + s_to_a + p_to_a, deep_a)
    deep_s_new = jnp.where(active, deep_s_work + a_to_s + p_to_s, deep_s)
    deep_p_new = jnp.where(active, deep_p_work + a_to_p + s_to_p, deep_p)

    perma_result = None
    if bool(perma_peat):
        if cmax_peat is None or is_peat is None:
            raise ValueError("perma_peat requires explicit cmax_peat and is_peat")
        thickness = zf[1:] - zf[:-1]
        carbon_amount = jnp.stack(
            (
                (deep_a_new * thickness[None, :, None]).transpose((0, 2, 1)),
                (deep_s_new * thickness[None, :, None]).transpose((0, 2, 1)),
                (deep_p_new * thickness[None, :, None]).transpose((0, 2, 1)),
            ),
            axis=1,
        )
        perma_result = soilcarbon_perma_peat_redistribute(
            carbon_amount,
            cmax_peat,
            zf,
            is_peat,
            mask,
            frac1=frac1,
            frac2=frac2,
            min_stomate=min_stomate,
        )
        deep_a_new = (perma_result.carbon_32l[:, IACTIVE, :, :] / thickness[None, None, :]).transpose((0, 2, 1))
        deep_s_new = (perma_result.carbon_32l[:, ISLOW, :, :] / thickness[None, None, :]).transpose((0, 2, 1))
        deep_p_new = (perma_result.carbon_32l[:, IPASSIVE, :, :] / thickness[None, None, :]).transpose((0, 2, 1))

    return DeepCarbonPermafrostDecompResult(
        deepC_a=deep_a_new,
        deepC_s=deep_s_new,
        deepC_p=deep_p_new,
        O2_soil=o2_final,
        CH4_soil=ch4_final,
        deltaC1_a=jnp.where(active, dC_a * fr_a, 0.0),
        deltaC1_s=jnp.where(active, dC_s * fr_s, 0.0),
        deltaC1_p=jnp.where(active, dC_p * fr_p, 0.0),
        deltaCH4=jnp.where(active, deltaCH4, 0.0),
        deltaCH4g=jnp.where(active, deltaCH4g, 0.0),
        deltaC2=jnp.where(active, deltaC2, 0.0),
        deltaC3=jnp.where(active, deltaC3, 0.0),
        nadd_soil=jnp.where(active, nadd_soil, 0.0),
        perma_peat=perma_result,
    )


def deep_carbon_yedoma_reset_step(
    deepC_a,
    deepC_s,
    deepC_p,
    zz_deep,
    altmax_ind,
    veget_mask,
    *,
    yedoma_depth,
    yedoma_cinit_act,
    yedoma_cinit_slo,
    yedoma_cinit_pas,
    yedoma_map_filename: str = "NONE",
    yedoma_values=None,
) -> DeepCarbonYedomaResetResult:
    """Apply the source yedoma carbon-stock reset after map values are known.

    Fortran provenance: ``src_stomate/stomate_permafrost_soilcarbon.f90``,
    subroutine ``initialize_yedoma_carbonstocks``, lines 3014-3165. The NetCDF
    nearest-neighbor file read is an IO boundary; this kernel supports the
    source ``NONE``/``EVERYWHERE`` modes directly and accepts explicit resolved
    per-grid-cell yedoma values for file-backed cases.
    """

    deep_a = jnp.asarray(deepC_a)
    deep_s = jnp.asarray(deepC_s)
    deep_p = jnp.asarray(deepC_p)
    zz = jnp.asarray(zz_deep)
    alt_ind = jnp.asarray(altmax_ind)
    mask = jnp.asarray(veget_mask, dtype=bool)
    if deep_a.ndim != 3:
        raise ValueError("deepC_a must have shape (npts, ndeep, nvm)")
    if deep_s.shape != deep_a.shape or deep_p.shape != deep_a.shape:
        raise ValueError("deepC_s and deepC_p must match deepC_a shape")
    npts, ndeep, nvm = deep_a.shape
    if zz.shape != (ndeep,):
        raise ValueError("zz_deep must have shape (ndeep,)")
    if alt_ind.shape != (npts, nvm):
        raise ValueError("altmax_ind must have shape (npts, nvm)")
    if mask.shape != (npts, nvm):
        raise ValueError("veget_mask must have shape (npts, nvm)")
    if yedoma_values is None:
        if yedoma_map_filename == "NONE":
            yedoma = jnp.zeros((npts,), dtype=deep_a.dtype)
        elif yedoma_map_filename == "EVERYWHERE":
            yedoma = jnp.ones((npts,), dtype=deep_a.dtype)
        else:
            raise ValueError("file-backed yedoma maps require explicit resolved yedoma_values")
    else:
        yedoma = jnp.asarray(yedoma_values, dtype=deep_a.dtype)
        if yedoma.shape != (npts,):
            raise ValueError("yedoma_values must have shape (npts,)")

    layer_no = jnp.arange(1, ndeep + 1)
    yedoma_depth_index = jnp.sum(zz <= yedoma_depth)
    in_reset_depth = layer_no[None, :, None] <= yedoma_depth_index
    below_or_at_alt = layer_no[None, :, None] >= alt_ind[:, None, :]
    pft_nonzero = jnp.arange(nvm)[None, None, :] >= 1
    active = in_reset_depth & pft_nonzero & mask[:, None, :]
    has_yedoma = yedoma[:, None, None] > 0.0
    set_yedoma = active & has_yedoma & below_or_at_alt
    set_zero = active & (~set_yedoma)
    deep_a = jnp.where(set_yedoma, jnp.asarray(yedoma_cinit_act, dtype=deep_a.dtype), jnp.where(set_zero, 0.0, deep_a))
    deep_s = jnp.where(set_yedoma, jnp.asarray(yedoma_cinit_slo, dtype=deep_s.dtype), jnp.where(set_zero, 0.0, deep_s))
    deep_p = jnp.where(set_yedoma, jnp.asarray(yedoma_cinit_pas, dtype=deep_p.dtype), jnp.where(set_zero, 0.0, deep_p))
    return DeepCarbonYedomaResetResult(
        deepC_a=deep_a,
        deepC_s=deep_s,
        deepC_p=deep_p,
        yedoma=yedoma,
        yedoma_depth_index=yedoma_depth_index,
    )


def deep_carbon_nonmethane_core_step(
    deepC_a,
    deepC_s,
    deepC_p,
    soilc_in,
    fbact_out,
    O2_soil,
    totporO2_soil,
    clay,
    veget_mask,
    zi_soil,
    zf_soil,
    z_root,
    altmax_lastyear,
    rprof,
    *,
    time_step_seconds,
    airvol_soil=None,
    oxlim: bool = False,
    no_pfrost_decomp: bool = False,
    new_carbinput_intdepzlit: bool = False,
    correct_carboninput_vertprof: bool = True,
    perma_peat: bool = False,
    cmax_peat=None,
    is_peat=None,
    frac1=0.95,
    frac2=0.05,
    min_stomate=0.0,
    one_day=86400.0,
    ok_methane: bool = False,
    CH4_soil=None,
    totporCH4_soil=None,
    hslong=None,
    tprof=None,
    tau_CH4troph=432000.0,
    fbactratio=9.0,
    O2m=3.0,
    MG_useallCpools: bool = True,
    firstcall: bool = True,
    enable_plant_transport: bool = True,
    enable_ebullition: bool = True,
    Tref=None,
    rootlev=None,
    Tgr=0.0,
    refdep=0.75,
    ok_cryoturb: bool = False,
    cryoturbation_coefficients: DeepCarbonCryoturbationCoefficients | None = None,
    altmax_ind=None,
    fixed_cryoturbation_depth=None,
    cryoturbation_diff_k_const=0.001,
    cryoturbation_bio_diff_k_const=0.0001,
    use_new_cryoturbation: bool = False,
    cryoturbation_method: int = 4,
    max_cryoturb_alt=3.0,
    min_cryoturb_alt=0.01,
    use_fixed_cryoturbation_depth: bool = False,
    bioturbation_depth=2.0,
    heat_co2_act=40.0e6,
    heat_co2_slo=30.0e6,
    heat_co2_pas=10.0e6,
    heat_ch4_gen=0.0,
    heat_ch4_troph=0.0,
) -> DeepCarbonNonMethaneCoreResult:
    """Compose the source-order non-methane OK_PC deep-carbon core.

    Fortran provenance: ``src_stomate/stomate_permafrost_soilcarbon.f90``,
    ``deep_carbcycle`` lines 953-1077. This is the local carbon core after
    active-layer/root-depth and gas-diffusion state have been supplied:
    optional ``cryoturbate`` diffuse -> ``carbinput`` -> ``permafrost_decomp``
    -> optional methane plant transport/ebullition -> respiration/CH4 flux
    aggregation -> optional ``cryoturbate`` coefficients ->
    ``calc_vert_int_soil_carbon``.
    """

    deep_start_a = jnp.asarray(deepC_a)
    deep_start_s = jnp.asarray(deepC_s)
    deep_start_p = jnp.asarray(deepC_p)
    new_cryoturbation_coefficients = cryoturbation_coefficients
    if bool(ok_cryoturb):
        required = {
            "altmax_ind": altmax_ind,
            "fixed_cryoturbation_depth": fixed_cryoturbation_depth,
        }
        missing = tuple(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(f"ok_cryoturb requires explicit {', '.join(missing)}")
        if cryoturbation_coefficients is not None:
            diffuse = deep_carbon_cryoturbation_diffuse(
                deep_start_a,
                deep_start_s,
                deep_start_p,
                cryoturbation_coefficients,
                altmax_ind,
                zi_soil,
                zf_soil,
            )
            deep_start_a = diffuse.deepC_a
            deep_start_s = diffuse.deepC_s
            deep_start_p = diffuse.deepC_p

    carbon_input = deep_carbon_input_step(
        deep_start_a,
        deep_start_s,
        deep_start_p,
        soilc_in,
        z_root,
        altmax_lastyear,
        veget_mask,
        zi_soil,
        zf_soil,
        rprof,
        time_step_seconds=time_step_seconds,
        no_pfrost_decomp=no_pfrost_decomp,
        new_carbinput_intdepzlit=new_carbinput_intdepzlit,
        correct_carboninput_vertprof=correct_carboninput_vertprof,
        one_day=one_day,
    )
    ch4_initial = None
    if bool(ok_methane):
        required = {
            "CH4_soil": CH4_soil,
            "totporCH4_soil": totporCH4_soil,
            "hslong": hslong,
            "tprof": tprof,
        }
        missing = tuple(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(f"ok_methane requires explicit {', '.join(missing)}")
        ch4_initial = jnp.asarray(CH4_soil)
    decomp = deep_carbon_permafrost_decomp_oxic_step(
        carbon_input.deepC_a,
        carbon_input.deepC_s,
        carbon_input.deepC_p,
        fbact_out,
        O2_soil,
        totporO2_soil,
        clay,
        veget_mask,
        zf_soil,
        time_step_seconds=time_step_seconds,
        airvol_soil=airvol_soil,
        oxlim=oxlim,
        no_pfrost_decomp=no_pfrost_decomp,
        ok_methane=ok_methane,
        CH4_soil=CH4_soil,
        totporCH4_soil=totporCH4_soil,
        hslong=hslong,
        tprof=tprof,
        tau_CH4troph=tau_CH4troph,
        fbactratio=fbactratio,
        O2m=O2m,
        MG_useallCpools=MG_useallCpools,
        perma_peat=perma_peat,
        cmax_peat=cmax_peat,
        is_peat=is_peat,
        frac1=frac1,
        frac2=frac2,
        min_stomate=min_stomate,
    )
    o2_after = decomp.O2_soil
    ch4_after = decomp.CH4_soil
    tref_after = None if Tref is None else jnp.asarray(Tref)
    if bool(ok_methane) and not bool(firstcall):
        if bool(enable_plant_transport):
            if Tref is None:
                raise ValueError("enable_plant_transport requires explicit Tref")
            if rootlev is None:
                raise ValueError("enable_plant_transport requires explicit rootlev")
            plant = deep_carbon_plant_transport_step(
                ch4_after,
                o2_after,
                totporCH4_soil,
                totporO2_soil,
                z_root,
                rootlev,
                Tref,
                hslong,
                zi_soil,
                zf_soil,
                tprof,
                veget_mask,
                time_step_seconds=time_step_seconds,
                Tgr=Tgr,
                refdep=refdep,
            )
            ch4_after = plant.CH4_soil
            o2_after = plant.O2_soil
            tref_after = plant.Tref
        if bool(enable_ebullition):
            ebullition = deep_carbon_ebullition_step(
                ch4_after,
                totporCH4_soil,
                hslong,
                zi_soil,
                zf_soil,
                veget_mask,
                time_step_seconds=time_step_seconds,
            )
            ch4_after = ebullition.CH4_soil
    if totporCH4_soil is None:
        methane_porosity = jnp.zeros_like(decomp.deltaCH4)
    else:
        methane_porosity = jnp.asarray(totporCH4_soil)
    heat_Zimov = (
        jnp.asarray(heat_co2_act, dtype=decomp.deltaC1_a.dtype) * 1.0e-3 * decomp.deltaC1_a
        + jnp.asarray(heat_co2_slo, dtype=decomp.deltaC1_s.dtype) * 1.0e-3 * decomp.deltaC1_s
        + jnp.asarray(heat_co2_pas, dtype=decomp.deltaC1_p.dtype) * 1.0e-3 * decomp.deltaC1_p
        + jnp.asarray(heat_ch4_gen, dtype=decomp.deltaC2.dtype) * 1.0e-3 * decomp.deltaC2
        + jnp.asarray(heat_ch4_troph, dtype=decomp.deltaCH4.dtype) * 1.0e-3 * decomp.deltaCH4 * methane_porosity
    ) / jnp.asarray(time_step_seconds, dtype=decomp.deltaC1_a.dtype)
    heat_Zimov = jnp.where(jnp.asarray(veget_mask, dtype=bool)[:, None, :], heat_Zimov, 0.0)
    thickness = jnp.asarray(zf_soil)[1:] - jnp.asarray(zf_soil)[:-1]
    dC1i = jnp.sum((decomp.deltaC1_a + decomp.deltaC1_s + decomp.deltaC1_p) * thickness[None, :, None], axis=1)
    sfluxCH4 = None
    if bool(ok_methane):
        mt = jnp.sum(decomp.deltaCH4 * jnp.asarray(totporCH4_soil) * thickness[None, :, None], axis=1)
        mg = jnp.sum(decomp.deltaCH4g * jnp.asarray(totporCH4_soil) * thickness[None, :, None], axis=1)
        ch4i = jnp.sum(ch4_after * jnp.asarray(totporCH4_soil) * thickness[None, :, None], axis=1)
        ch4ii = jnp.sum(ch4_initial * jnp.asarray(totporCH4_soil) * thickness[None, :, None], axis=1)
        resp_source = dC1i + mt * (12.0 / 16.0)
        sfluxCH4 = (ch4ii - ch4i + mg - mt) * jnp.asarray(one_day, dtype=dC1i.dtype) / jnp.asarray(time_step_seconds, dtype=dC1i.dtype)
        sfluxCH4 = jnp.where(jnp.asarray(veget_mask, dtype=bool), sfluxCH4, 0.0)
    else:
        resp_source = dC1i
    resp_hetero_soil = resp_source * jnp.asarray(one_day, dtype=dC1i.dtype) / jnp.asarray(time_step_seconds, dtype=dC1i.dtype)
    resp_hetero_soil = jnp.where(jnp.asarray(veget_mask, dtype=bool), resp_hetero_soil, 0.0)
    if bool(ok_cryoturb):
        new_cryoturbation_coefficients = deep_carbon_cryoturbation_coefficients(
            altmax_ind,
            decomp.deepC_a,
            decomp.deepC_s,
            decomp.deepC_p,
            altmax_lastyear,
            fixed_cryoturbation_depth,
            veget_mask,
            zi_soil,
            zf_soil,
            dt_seconds=time_step_seconds,
            diff_k_const=cryoturbation_diff_k_const,
            bio_diff_k_const=cryoturbation_bio_diff_k_const,
            use_new_cryoturbation=use_new_cryoturbation,
            cryoturbation_method=cryoturbation_method,
            max_cryoturb_alt=max_cryoturb_alt,
            min_cryoturb_alt=min_cryoturb_alt,
            use_fixed_cryoturbation_depth=use_fixed_cryoturbation_depth,
            bioturbation_depth=bioturbation_depth,
        )
    vertical = deep_carbon_vertical_integral_step(
        decomp.deepC_a,
        decomp.deepC_s,
        decomp.deepC_p,
        zf_soil,
        veget_mask,
    )
    return DeepCarbonNonMethaneCoreResult(
        deepC_a=decomp.deepC_a,
        deepC_s=decomp.deepC_s,
        deepC_p=decomp.deepC_p,
        carbon=vertical.carbon,
        carbon_surf=vertical.carbon_surf,
        resp_hetero_soil=resp_hetero_soil,
        sfluxCH4=sfluxCH4,
        heat_Zimov=heat_Zimov,
        O2_soil=o2_after,
        CH4_soil=ch4_after,
        dc_litter_z=carbon_input.dc_litter_z,
        decomposition=decomp,
        cryoturbation_coefficients=new_cryoturbation_coefficients,
        Tref=tref_after,
    )


def soilcarbon_perma_peat_cmax(peat_bulk_density, zf_soil_b) -> jnp.ndarray:
    """Compute maximum peat carbon content per layer.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``, subroutine
    ``soilcarbon_leak``, lines 1078-1087. ``peat_bulk_density`` is in g cm-3
    and the returned ``Cmax`` is in g m-2 per layer.
    """

    peat_bd = jnp.asarray(peat_bulk_density)
    zf = jnp.asarray(zf_soil_b)
    if zf.shape[0] != peat_bd.shape[0] + 1:
        raise ValueError("zf_soil_b must have length ndeep + 1")
    peat_soc = (1.0 / ((0.4 * peat_bd + 0.13) ** 2.19)) * 0.01
    return peat_bd * 1.0e6 * peat_soc * (zf[1:] - zf[:-1])


def soilcarbon_perma_peat_redistribute(
    carbon_32l,
    cmax,
    zf_soil_b,
    is_peat,
    veget_mask,
    *,
    frac1=0.95,
    frac2=0.05,
    min_stomate=0.0,
) -> SoilcarbonPermaPeatResult:
    """Apply the ``PERMA_PEAT`` soilcarbon redistribution branch.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``,
    ``soilcarbon_leak`` lines 1375-1437, with default ``frac1``/``frac2`` from
    ``src_parameters/constantes_var.f90`` lines 1390-1391. The loop is kept
    sequential because each layer can add carbon to the next layer before that
    next layer is tested against ``Cmax``.
    """

    carbon = jnp.asarray(carbon_32l)
    cmax = jnp.asarray(cmax)
    zf = jnp.asarray(zf_soil_b)
    is_peat = jnp.asarray(is_peat, dtype=bool)
    veget_mask = jnp.asarray(veget_mask, dtype=bool)
    npts, ncarb, nvm, ndeep = carbon.shape
    if ncarb != NCARB:
        raise ValueError("carbon_32l carbon-pool axis must have length NCARB")
    if cmax.shape != (ndeep,):
        raise ValueError("cmax must have shape (ndeep,)")
    if zf.shape != (ndeep + 1,):
        raise ValueError("zf_soil_b must have length ndeep + 1")
    if is_peat.shape != (nvm,):
        raise ValueError("is_peat must have shape (nvm,)")
    if veget_mask.shape != (npts, nvm):
        raise ValueError("veget_mask must have shape (npts, nvm)")

    active = veget_mask & is_peat[None, :]
    deepc_peat = jnp.where(active[:, None, :], jnp.sum(carbon, axis=1).transpose((0, 2, 1)), 0.0)
    deepc_pt = jnp.zeros((npts, ndeep, nvm), dtype=carbon.dtype)
    layer_thickness = zf[1:] - zf[:-1]
    for layer in range(ndeep - 1):
        over = active & (deepc_peat[:, layer, :] > frac1 * cmax[layer])
        excess = deepc_peat[:, layer, :] * frac2
        max_vsreal = jnp.where(deepc_peat[:, layer, :] != 0.0, (deepc_peat[:, layer, :] - excess) / deepc_peat[:, layer, :], 1.0)
        carbon = carbon.at[:, IACTIVE, :, layer].set(jnp.where(over, carbon[:, IACTIVE, :, layer] * max_vsreal, carbon[:, IACTIVE, :, layer]))
        carbon = carbon.at[:, ISLOW, :, layer].set(jnp.where(over, carbon[:, ISLOW, :, layer] * max_vsreal, carbon[:, ISLOW, :, layer]))
        carbon = carbon.at[:, IPASSIVE, :, layer].set(
            jnp.where(over, carbon[:, IPASSIVE, :, layer] * max_vsreal, carbon[:, IPASSIVE, :, layer])
        )
        deepc_peat = deepc_peat.at[:, layer, :].set(jnp.where(over, deepc_peat[:, layer, :] - excess, deepc_peat[:, layer, :]))
        trans_flux = excess * layer_thickness[layer] / layer_thickness[layer + 1]
        safe_deep = jnp.where(deepc_peat[:, layer, :] != 0.0, deepc_peat[:, layer, :], 1.0)
        for pool in (IACTIVE, ISLOW, IPASSIVE):
            add = carbon[:, pool, :, layer] / safe_deep * trans_flux
            carbon = carbon.at[:, pool, :, layer + 1].add(jnp.where(over, add, 0.0))
        deepc_peat = deepc_peat.at[:, layer + 1, :].add(jnp.where(over, trans_flux, 0.0))
        deepc_pt = deepc_pt.at[:, layer, :].set(jnp.where(active, deepc_peat[:, layer, :] / layer_thickness[layer], deepc_pt[:, layer, :]))

    positive = deepc_peat > min_stomate
    contiguous_positive = jnp.cumprod(positive.astype(jnp.int32), axis=1).astype(bool)
    valid_peat_layers = contiguous_positive & active[:, None, :]
    cthick = layer_thickness[None, :, None] * deepc_peat / cmax[None, :, None]
    layer_olt = zf[:-1][None, :, None] + cthick
    layer_olt = layer_olt.at[:, -1, :].set(jnp.minimum(zf[-1], layer_olt[:, -1, :]))
    valid_count = jnp.sum(valid_peat_layers.astype(jnp.int32), axis=1)
    last_valid = jnp.maximum(valid_count - 1, 0)
    peat_olt_candidate = jnp.take_along_axis(layer_olt, last_valid[:, None, :], axis=1)[:, 0, :]
    peat_olt = jnp.where(valid_count > 0, peat_olt_candidate, 0.0)

    return SoilcarbonPermaPeatResult(
        carbon_32l=carbon,
        deepc_peat=deepc_peat,
        deepc_pt=deepc_pt,
        peat_olt=peat_olt,
    )


def soilcarbon_leak_activity_factors(
    fbact_doc,
    fbact,
    *,
    carbon_tau_iactive=0.149,
    carbon_tau_islow=5.48,
    carbon_tau_ipassive=241.0,
    doc_tau_labile=1.3,
    doc_tau_stable=60.4,
    one_year=365.0,
) -> SoilcarbonActivityFactors:
    """Build DOC/POC activity-factor arrays for ``soilcarbon_leak``.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``,
    ``soilcarbon_leak`` lines 1176-1193. Constants provenance:
    ``src_parameters/constantes_var.f90`` lines 1350-1368.
    """

    fbact_doc = jnp.asarray(fbact_doc)
    fbact = jnp.asarray(fbact)
    if fbact_doc.ndim != 3 or fbact.ndim != 3:
        raise ValueError("fbact_doc and fbact must have shape (npts, ndeep, nvm)")
    if fbact_doc.shape != fbact.shape:
        raise ValueError("fbact_doc and fbact must share shape")
    npts, ndeep, nvm = fbact.shape
    fbact_doc_labile = fbact_doc * (carbon_tau_iactive * one_year / doc_tau_labile)
    fbact_doc_refractory = fbact_doc * (carbon_tau_iactive * one_year / doc_tau_stable)
    fbact_npool = jnp.zeros((npts, ndeep, nvm, NPOOL), dtype=fbact.dtype)
    for pool in (IMETABO, IMETBEL, ISTRABO, ISTRBEL, IACT):
        fbact_npool = fbact_npool.at[:, :, :, pool].set(fbact_doc_labile)
    for pool in (IPAS, ISLO):
        fbact_npool = fbact_npool.at[:, :, :, pool].set(fbact_doc_refractory)

    fbact_ncarb = jnp.zeros((npts, ndeep, nvm, NCARB), dtype=fbact.dtype)
    fbact_ncarb = fbact_ncarb.at[:, :, :, IACTIVE].set(fbact)
    fbact_ncarb = fbact_ncarb.at[:, :, :, ISLOW].set(fbact * (carbon_tau_iactive / carbon_tau_islow))
    fbact_ncarb = fbact_ncarb.at[:, :, :, IPASSIVE].set(fbact * (carbon_tau_iactive / carbon_tau_ipassive))
    return SoilcarbonActivityFactors(
        fbact_doc_labile=fbact_doc_labile,
        fbact_doc_refractory=fbact_doc_refractory,
        fbact_npool=fbact_npool,
        fbact_ncarb=fbact_ncarb,
    )


def soilcarbon_leak_frac_carb(
    clay,
    *,
    metabolic_ref_frac=0.85,
    active_to_pass_clay_frac=0.68,
) -> jnp.ndarray:
    """Return the ``frac_carb`` table used in the OK_LEAK soilcarbon path.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``,
    ``soilcarbon_leak`` lines 1289-1302. This leak-path table differs from the
    older ``soilcarbon`` routine table earlier in the same file.
    """

    clay = jnp.asarray(clay)
    if clay.ndim != 1:
        raise ValueError("clay must have shape (npts,)")
    frac = jnp.zeros((clay.shape[0], NCARB, NCARB), dtype=clay.dtype)
    frac = frac.at[:, IACTIVE, IPASSIVE].set(0.004 / (1.0 - metabolic_ref_frac + active_to_pass_clay_frac * clay))
    frac = frac.at[:, IACTIVE, ISLOW].set(1.0 - frac[:, IACTIVE, IPASSIVE])
    frac = frac.at[:, ISLOW, IACTIVE].set(0.93)
    frac = frac.at[:, ISLOW, IPASSIVE].set(1.0 - frac[:, ISLOW, IACTIVE])
    frac = frac.at[:, IPASSIVE, IACTIVE].set(1.0)
    frac = frac.at[:, IPASSIVE, ISLOW].set(1.0 - frac[:, IPASSIVE, IACTIVE])
    return frac


def soilcarbon_leak_tf_doc_ground_fluxes(
    doc_precip2ground,
    doc_precip2canopy,
    dry_dep_canopy,
    interception_storage,
    canopy2ground,
    veget_max,
    flood_frac,
    *,
    conc_doc_max=100.0,
) -> TfDocGroundFluxes:
    """Move TF-DOC from canopy to ground/flood and update storage.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``,
    ``soilcarbon_leak`` lines 1495-1525. Only carbon is non-zero in the source
    path; non-carbon elements are carried through unchanged except storage
    updates are limited to the carbon element.
    """

    doc_precip2ground = jnp.asarray(doc_precip2ground)
    doc_precip2canopy = jnp.asarray(doc_precip2canopy)
    dry_dep_canopy = jnp.asarray(dry_dep_canopy)
    interception_storage = jnp.asarray(interception_storage)
    canopy2ground = jnp.asarray(canopy2ground)
    veget_max = jnp.asarray(veget_max)
    flood_frac = jnp.asarray(flood_frac)
    if interception_storage.shape != doc_precip2ground.shape:
        raise ValueError("interception_storage and DOC precipitation arrays must share shape")
    npts, nvm, nelements = interception_storage.shape
    if canopy2ground.shape != (npts, nvm) or veget_max.shape != (npts, nvm):
        raise ValueError("canopy2ground and veget_max must have shape (npts, nvm)")

    doc_canopy2ground = jnp.zeros_like(interception_storage)
    possible = canopy2ground * conc_doc_max * 1e-3
    active = (jnp.arange(nvm) > 0)[None, :] & (veget_max > 0.0)
    doc_canopy2ground = doc_canopy2ground.at[:, :, ICARBON].set(
        jnp.where(active, jnp.minimum(possible, interception_storage[:, :, ICARBON]), 0.0)
    )
    wet = doc_precip2ground + doc_canopy2ground
    wet_dep_flood = wet * flood_frac[:, None, None]
    wet_dep_ground = wet * (1.0 - flood_frac[:, None, None])
    storage_c = jnp.where(
        active,
        interception_storage[:, :, ICARBON]
        + dry_dep_canopy[:, :, ICARBON]
        + doc_precip2canopy[:, :, ICARBON]
        - doc_canopy2ground[:, :, ICARBON],
        0.0,
    )
    storage = interception_storage.at[:, :, ICARBON].set(storage_c)
    if nelements > 1:
        inactive = ~active
        storage = storage.at[:, :, 1:].set(jnp.where(inactive[:, :, None], 0.0, interception_storage[:, :, 1:]))
    return TfDocGroundFluxes(
        doc_canopy2ground=doc_canopy2ground,
        wet_dep_ground=wet_dep_ground,
        wet_dep_flood=wet_dep_flood,
        interception_storage=storage,
    )


def soilcarbon_leak_doc_inputs(
    doc,
    soilcarbon_input_doc,
    doc_to_topsoil,
    doc_to_subsoil,
    wet_dep_ground,
    veget_max,
    *,
    dt_days,
    bio_frac=None,
    nslm: int | None = None,
) -> jnp.ndarray:
    """Update free DOC with litter, topsoil/subsoil, and wet-deposition inputs.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``,
    ``soilcarbon_leak`` lines 1528-1557. Source updates only ``icarbon``.
    """

    doc = jnp.asarray(doc)
    soilcarbon_input_doc = jnp.asarray(soilcarbon_input_doc)
    doc_to_topsoil = jnp.asarray(doc_to_topsoil)
    doc_to_subsoil = jnp.asarray(doc_to_subsoil)
    wet_dep_ground = jnp.asarray(wet_dep_ground)
    veget_max = jnp.asarray(veget_max)
    if doc.ndim != 6 or doc.shape[3] != NDOC or doc.shape[4] != NPOOL:
        raise ValueError("doc must have shape (npts, nvm, ndeep, ndoc, npool, nelements)")
    npts, nvm, ndeep, _, _, nelements = doc.shape
    if soilcarbon_input_doc.shape != (npts, nvm, ndeep, NPOOL, nelements):
        raise ValueError("soilcarbon_input_doc must have shape (npts, nvm, ndeep, npool, nelements)")
    if doc_to_topsoil.shape[0] != npts or doc_to_subsoil.shape[0] != npts:
        raise ValueError("DOC_to_topsoil and DOC_to_subsoil must start with npts")
    if nslm is None:
        nslm = ndeep
    subsoil_layer = int(nslm) - 1
    if subsoil_layer < 0 or subsoil_layer >= ndeep:
        raise ValueError("nslm must select a valid zero-based DOC layer")
    if bio_frac is None:
        bio_frac = jnp.sum(veget_max[:, 1:], axis=1)
    bio_frac = jnp.asarray(bio_frac)

    active_pft = jnp.arange(nvm) > 0
    carbon_inputs = soilcarbon_input_doc[:, :, :, :, ICARBON] * dt_days
    litter_inputs = jnp.zeros_like(carbon_inputs)
    for pool in (IMETABO, ISTRABO, IMETBEL, ISTRBEL):
        litter_inputs = litter_inputs.at[:, :, :, pool].set(carbon_inputs[:, :, :, pool])
    doc = doc.at[:, :, :, IFREE, :, ICARBON].add(
        jnp.where(active_pft[None, :, None, None], litter_inputs, 0.0)
    )

    has_veg = (veget_max > 0.0) & active_pft[None, :]
    safe_bio = jnp.where(bio_frac > 0.0, bio_frac, 1.0)
    lab_top = doc_to_topsoil[:, IDOCL] * dt_days / safe_bio
    lab_sub = doc_to_subsoil[:, IDOCL] * dt_days / safe_bio
    ref_top = doc_to_topsoil[:, IDOCR] * dt_days / safe_bio
    ref_sub = doc_to_subsoil[:, IDOCR] * dt_days / safe_bio
    wet_half = 0.5 * wet_dep_ground[:, :, ICARBON] / jnp.where(veget_max > 0.0, veget_max, 1.0)
    doc = doc.at[:, :, 0, IFREE, IACT, ICARBON].add(jnp.where(has_veg, lab_top[:, None] + wet_half, 0.0))
    doc = doc.at[:, :, subsoil_layer, IFREE, IACT, ICARBON].add(jnp.where(has_veg, lab_sub[:, None], 0.0))
    doc = doc.at[:, :, 0, IFREE, ISLO, ICARBON].add(jnp.where(has_veg, ref_top[:, None] + wet_half, 0.0))
    doc = doc.at[:, :, subsoil_layer, IFREE, ISLO, ICARBON].add(jnp.where(has_veg, ref_sub[:, None], 0.0))
    return doc


soilcarbon_leak_apply_doc_inputs = soilcarbon_leak_doc_inputs


def soilcarbon_leak_litter_tot_lom(
    litter_above,
    litter_below,
    carbon_32l,
    doc,
    *,
    nslm: int | None = None,
) -> SoilcarbonLomResult:
    """Build ``litter_tot`` and per-carbon-pool LOM for priming.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``,
    ``soilcarbon_leak`` lines 1563-1590.
    """

    litter_above = jnp.asarray(litter_above)
    litter_below = jnp.asarray(litter_below)
    carbon_32l = jnp.asarray(carbon_32l)
    doc = jnp.asarray(doc)
    npts, nvm, ndeep = carbon_32l.shape[0], carbon_32l.shape[2], carbon_32l.shape[3]
    if nslm is None:
        nslm = ndeep
    nslm = int(nslm)
    if nslm < 0 or nslm > ndeep:
        raise ValueError("nslm must be between 0 and ndeep")
    litter_tot = jnp.zeros((npts, nvm, ndeep), dtype=carbon_32l.dtype)
    surface = litter_above[:, 0, :, ICARBON] + litter_above[:, 1, :, ICARBON]
    litter_tot = litter_tot.at[:, :, 0].set(surface)
    below = litter_below[:, 0, :, :, ICARBON] + litter_below[:, 1, :, :, ICARBON]
    litter_tot = litter_tot.at[:, :, :nslm].add(below[:, :, :nslm])
    litter_tot = litter_tot.at[:, 0, :].set(0.0)

    litter_doc = (
        doc[:, :, :, IFREE, IMETABO, ICARBON]
        + doc[:, :, :, IFREE, ISTRABO, ICARBON]
        + doc[:, :, :, IFREE, IMETBEL, ICARBON]
        + doc[:, :, :, IFREE, ISTRBEL, ICARBON]
    )
    lom = jnp.zeros((npts, nvm, ndeep, NCARB), dtype=carbon_32l.dtype)
    lom = lom.at[:, :, :, IACTIVE].set(litter_tot + litter_doc)
    lom = lom.at[:, :, :, ISLOW].set(litter_tot + carbon_32l[:, IACTIVE, :, :] + doc[:, :, :, IFREE, IACT, ICARBON] + litter_doc)
    lom = lom.at[:, :, :, IPASSIVE].set(
        litter_tot
        + carbon_32l[:, IACTIVE, :, :]
        + carbon_32l[:, ISLOW, :, :]
        + doc[:, :, :, IFREE, IACT, ICARBON]
        + doc[:, :, :, IFREE, ISLO, ICARBON]
        + litter_doc
    )
    return SoilcarbonLomResult(litter_tot=litter_tot, lom=lom)


soilcarbon_leak_litter_lom = soilcarbon_leak_litter_tot_lom


def _pft_factor(nvm, natural, is_peat, is_c4, flux_tot_coeff):
    natural = jnp.asarray(natural, dtype=bool)
    is_peat = jnp.asarray(is_peat, dtype=bool)
    is_c4 = jnp.asarray(is_c4, dtype=bool)
    pft_factor = jnp.ones((nvm,), dtype=jnp.asarray(flux_tot_coeff).dtype)
    crop_c3 = (~natural) & (~is_peat) & (~is_c4)
    crop_c4 = (~natural) & (~is_peat) & is_c4
    pft_factor = jnp.where(crop_c3, flux_tot_coeff[0], pft_factor)
    pft_factor = jnp.where(crop_c4, flux_tot_coeff[1], pft_factor)
    return pft_factor


def soilcarbon_leak_decompose_poc_doc(
    carbon_32l,
    doc,
    lom,
    lignin_struc_above,
    lignin_struc_below,
    fbact_npool,
    fbact_ncarb,
    frac_carb,
    soil_mc_32l,
    pref_soil_veg,
    flood_frac,
    natural,
    is_c4,
    is_peat,
    clay,
    *,
    dt_days,
    priming: bool = True,
    priming_param=(493.66, 194.03, 136.54),
    flux_tot_coeff=(1.2, 1.4, 0.75),
    cue=0.3,
    sro_bottom: int | None = None,
) -> SoilcarbonDecompositionResult:
    """Run POC/DOC decomposition and update pools through ``soilcarbon_leak`` 2.3.4.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``,
    ``soilcarbon_leak`` lines 1605-1785. ``sro_bottom`` is zero-based/count-like:
    flood-generated POC flux is added back to DOC only for layers whose Python
    index is greater than or equal to this value, matching Fortran
    ``IF (l .GT. sro_bottom)``.
    """

    carbon_32l = jnp.asarray(carbon_32l)
    doc = jnp.asarray(doc)
    lom = jnp.asarray(lom)
    frac_carb = jnp.asarray(frac_carb)
    soil_mc_32l = jnp.asarray(soil_mc_32l)
    pref_soil_veg = jnp.asarray(pref_soil_veg, dtype=jnp.int32)
    flood_frac = jnp.asarray(flood_frac)
    clay = jnp.asarray(clay)
    npts, ncarb, nvm, ndeep = carbon_32l.shape
    if ncarb != NCARB:
        raise ValueError("carbon_32l carbon-pool axis must have length NCARB")
    if doc.shape[:5] != (npts, nvm, ndeep, NDOC, NPOOL):
        raise ValueError("doc must have shape (npts, nvm, ndeep, ndoc, npool, nelements)")
    if fbact_npool.shape != (npts, ndeep, nvm, NPOOL):
        raise ValueError("fbact_npool must have shape (npts, ndeep, nvm, npool)")
    if fbact_ncarb.shape != (npts, ndeep, nvm, NCARB):
        raise ValueError("fbact_ncarb must have shape (npts, ndeep, nvm, ncarb)")
    if lom.shape != (npts, nvm, ndeep, NCARB):
        raise ValueError("lom must have shape (npts, nvm, ndeep, ncarb)")
    if frac_carb.shape != (npts, NCARB, NCARB):
        raise ValueError("frac_carb must have shape (npts, ncarb, ncarb)")
    if sro_bottom is None:
        sro_bottom = ndeep
    sro_bottom = int(sro_bottom)
    if sro_bottom < 0 or sro_bottom > ndeep:
        raise ValueError("sro_bottom must be between 0 and ndeep")

    pft_factor = _pft_factor(nvm, natural, is_peat, is_c4, jnp.asarray(flux_tot_coeff))
    pref = pref_soil_veg
    pref_min = jnp.min(pref)
    if not isinstance(pref_min, core.Tracer) and bool(pref_min < 0):
        raise ValueError("pref_soil_veg must use zero-based soil tile indices")
    moisture = jnp.take(soil_mc_32l, pref, axis=2).transpose((0, 2, 1))
    active_water = moisture > 0.0
    dry = 1.0 - flood_frac[:, None, None]
    flood = flood_frac[:, None, None] / 3.0
    carbon_by_pft = carbon_32l.transpose((0, 2, 3, 1))
    fbact_c = fbact_ncarb.transpose((0, 2, 1, 3))
    if priming:
        priming_term = 1.0 - jnp.exp(-jnp.asarray(priming_param)[None, None, None, :] * lom)
    else:
        priming_term = 1.0
    base = dt_days * fbact_c * carbon_by_pft * priming_term * pft_factor[None, :, None, None]
    fluxtot_by_pft = jnp.where(active_water[:, :, :, None], base * dry[:, :, :, None], 0.0)
    fluxtot_flood_by_pft = jnp.where(active_water[:, :, :, None], base * flood[:, :, :, None], 0.0)
    active_clay = 1.0 - jnp.asarray(flux_tot_coeff)[2] * clay
    fluxtot_by_pft = fluxtot_by_pft.at[:, :, :, IACTIVE].multiply(active_clay[:, None, None])
    fluxtot_flood_by_pft = fluxtot_flood_by_pft.at[:, :, :, IACTIVE].multiply(active_clay[:, None, None])
    fluxtot = fluxtot_by_pft.transpose((0, 3, 1, 2))[:, :, None, :]
    fluxtot_flood = fluxtot_flood_by_pft.transpose((0, 3, 1, 2))[:, :, None, :]

    cue_arr = jnp.asarray(cue, dtype=carbon_32l.dtype)
    resp_hetero_soil = jnp.sum((1.0 - cue_arr) * fluxtot_by_pft, axis=(2, 3)) / dt_days
    resp_flood_soil = jnp.sum((1.0 - cue_arr) * fluxtot_flood_by_pft, axis=(2, 3)) / dt_days

    doc_free_c = doc[:, :, :, IFREE, :, ICARBON]
    fbact_pool = fbact_npool.transpose((0, 2, 1, 3))
    fluxtot_doc = jnp.where(active_water[:, :, :, None], dt_days * fbact_pool * dry[:, :, :, None] * doc_free_c, 0.0)
    fluxtot_doc_flood = jnp.where(active_water[:, :, :, None], dt_days * fbact_pool * flood[:, :, :, None] * doc_free_c, 0.0)
    resp_hetero_soil = resp_hetero_soil + jnp.sum((1.0 - cue_arr) * fluxtot_doc, axis=(2, 3)) / dt_days
    resp_flood_soil = resp_flood_soil + jnp.sum((1.0 - cue_arr) * fluxtot_doc_flood, axis=(2, 3)) / dt_days

    total_doc_flux = fluxtot_doc + fluxtot_doc_flood
    lignin_above = jnp.asarray(lignin_struc_above)
    lignin_below = jnp.asarray(lignin_struc_below)
    def frac_from_to(source, target):
        return frac_carb[:, source, target][:, None, None]

    active_gain = (
        frac_from_to(IPASSIVE, IACTIVE) * total_doc_flux[:, :, :, IPAS]
        + frac_from_to(ISLOW, IACTIVE) * total_doc_flux[:, :, :, ISLO]
        + total_doc_flux[:, :, :, IMETBEL]
        + total_doc_flux[:, :, :, ISTRBEL] * (1.0 - lignin_below)
        + total_doc_flux[:, :, :, IMETABO]
        + total_doc_flux[:, :, :, ISTRABO] * (1.0 - lignin_above[:, :, None])
    )
    slow_gain = (
        frac_from_to(IPASSIVE, ISLOW) * total_doc_flux[:, :, :, IPAS]
        + frac_from_to(IACTIVE, ISLOW) * total_doc_flux[:, :, :, IACT]
        + total_doc_flux[:, :, :, ISTRBEL] * lignin_below
        + total_doc_flux[:, :, :, ISTRABO] * lignin_above[:, :, None]
    )
    passive_gain = (
        frac_from_to(IACTIVE, IPASSIVE) * total_doc_flux[:, :, :, IACT]
        + frac_from_to(ISLOW, IPASSIVE) * total_doc_flux[:, :, :, ISLO]
    )
    carbon_by_pft = carbon_by_pft.at[:, :, :, IACTIVE].add(cue_arr * active_gain)
    carbon_by_pft = carbon_by_pft.at[:, :, :, ISLOW].add(cue_arr * slow_gain)
    carbon_by_pft = carbon_by_pft.at[:, :, :, IPASSIVE].add(cue_arr * passive_gain)
    carbon_by_pft = carbon_by_pft - fluxtot_by_pft - fluxtot_flood_by_pft
    carbon_updated = carbon_by_pft.transpose((0, 3, 1, 2))

    poc_doc_gain = jnp.zeros_like(doc_free_c)
    upper_mask = (jnp.arange(ndeep) < sro_bottom)[None, None, :]
    add_flood_to_doc = ~upper_mask
    poc_active = cue_arr * (fluxtot_by_pft[:, :, :, IACTIVE] + jnp.where(add_flood_to_doc, fluxtot_flood_by_pft[:, :, :, IACTIVE], 0.0))
    poc_slow = cue_arr * (fluxtot_by_pft[:, :, :, ISLOW] + jnp.where(add_flood_to_doc, fluxtot_flood_by_pft[:, :, :, ISLOW], 0.0))
    poc_passive = cue_arr * (fluxtot_by_pft[:, :, :, IPASSIVE] + jnp.where(add_flood_to_doc, fluxtot_flood_by_pft[:, :, :, IPASSIVE], 0.0))
    poc_doc_gain = poc_doc_gain.at[:, :, :, IACT].add(poc_active)
    poc_doc_gain = poc_doc_gain.at[:, :, :, ISLO].add(poc_slow)
    poc_doc_gain = poc_doc_gain.at[:, :, :, IPAS].add(poc_passive)
    doc_free_c = doc_free_c + poc_doc_gain - total_doc_flux
    doc_updated = doc.at[:, :, :, IFREE, :, ICARBON].set(doc_free_c)
    return SoilcarbonDecompositionResult(
        carbon_32l=carbon_updated,
        doc=doc_updated,
        resp_hetero_soil=resp_hetero_soil,
        resp_flood_soil=resp_flood_soil,
        fluxtot=fluxtot,
        fluxtot_flood=fluxtot_flood,
        fluxtot_doc=fluxtot_doc,
        fluxtot_doc_flood=fluxtot_doc_flood,
        litter_tot=jnp.zeros((npts, nvm, ndeep), dtype=carbon_32l.dtype),
        lom=lom,
    )


def soilcarbon_leak_decompose_update(
    carbon_32l,
    doc,
    litter_above,
    litter_below,
    lignin_struc_above,
    lignin_struc_below,
    fbact_npool,
    fbact_ncarb,
    soil_mc_32l,
    pref_soil_veg,
    flood_frac,
    clay,
    natural,
    is_peat,
    is_c4,
    *,
    dt_days,
    priming: bool = True,
    priming_param=(493.66, 194.03, 136.54),
    flux_tot_coeff=(1.2, 1.4, 0.75),
    cue=0.3,
    nslm: int | None = None,
    sro_bottom: int | None = None,
) -> SoilcarbonDecompositionResult:
    """Build LOM and run POC/DOC decomposition through section 2.3.4.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``,
    ``soilcarbon_leak`` lines 1563-1785. This helper intentionally stops before
    adsorption, vertical DOC transport, diffusion, and export (lines
    1789-2303).
    """

    lom_result = soilcarbon_leak_litter_tot_lom(
        litter_above,
        litter_below,
        carbon_32l,
        doc,
        nslm=nslm,
    )
    result = soilcarbon_leak_decompose_poc_doc(
        carbon_32l,
        doc,
        lom_result.lom,
        lignin_struc_above,
        lignin_struc_below,
        fbact_npool,
        fbact_ncarb,
        soilcarbon_leak_frac_carb(clay),
        soil_mc_32l,
        pref_soil_veg,
        flood_frac,
        natural,
        is_c4,
        is_peat,
        clay,
        dt_days=dt_days,
        priming=priming,
        priming_param=priming_param,
        flux_tot_coeff=flux_tot_coeff,
        cue=cue,
        sro_bottom=sro_bottom,
    )
    return result._replace(litter_tot=lom_result.litter_tot, lom=lom_result.lom)


def soilcarbon_leak_adsorption_desorption(
    doc,
    clay,
    bulk_dens,
    z_soil,
    *,
    nslm: int | None = None,
    kd_ads=0.00805,
) -> SoilcarbonAdsorptionResult:
    """Equilibrate free and adsorbed DOC with the source Kd formula.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``,
    ``soilcarbon_leak`` lines 1789-1838. ``z_soil`` follows the Fortran
    boundary convention ``z_soil(0:nslm)`` with Python length ``nslm + 1``.
    Source logic updates all elements and DOC pools for every PFT/layer.
    """

    doc = jnp.asarray(doc)
    clay = jnp.asarray(clay)
    bulk_dens = jnp.asarray(bulk_dens)
    z_soil = jnp.asarray(z_soil)
    if doc.ndim != 6 or doc.shape[3] != NDOC or doc.shape[4] != NPOOL:
        raise ValueError("doc must have shape (npts, nvm, ndeep, ndoc, npool, nelements)")
    npts, nvm, ndeep, _, _, nelements = doc.shape
    if clay.shape != (npts,) or bulk_dens.shape != (npts,):
        raise ValueError("clay and bulk_dens must have shape (npts,)")
    if nslm is None:
        nslm = min(ndeep, z_soil.shape[0] - 1)
    nslm = int(nslm)
    if z_soil.ndim != 1 or z_soil.shape[0] < nslm + 1:
        raise ValueError("z_soil must be one-dimensional with length at least nslm + 1")
    if nslm < 1 or nslm > ndeep:
        raise ValueError("nslm must be between 1 and ndeep")

    # Fortran initializes every deep layer from kd_ads, then replaces only
    # layers 1:nslm with the depth-dependent statistical relationship.
    kd = jnp.full((npts, ndeep), jnp.asarray(kd_ads, dtype=doc.dtype), dtype=doc.dtype)
    kd_surface = 10.0 ** (-3.1 + 0.2 * jnp.log10(clay * 100.0) + jnp.log10(z_soil[1] * 100.0 / 2.0))
    kd = kd.at[:, 0].set(kd_surface)
    if nslm > 1:
        layer_mid = (z_soil[2 : nslm + 1] + z_soil[1:nslm]) / 2.0
        kd_deep = 10.0 ** (-3.1 + 0.2 * jnp.log10(clay[:, None] * 100.0) + jnp.log10(layer_mid[None, :]))
        kd = kd.at[:, 1:nslm].set(kd_deep)
    kd = jnp.clip(kd, 10.0 ** -3.2, 10.0 ** -1.5)

    total = doc[:, :, :, IFREE, :, :] + doc[:, :, :, IADSORBED, :, :]
    eq_frac = (kd[:, None, :, None, None] * bulk_dens[:, None, None, None, None]) / (
        kd[:, None, :, None, None] * bulk_dens[:, None, None, None, None] + 1.0
    )
    doc_re = eq_frac * total
    old_free = doc[:, :, :, IFREE, :, :]
    old_ads = doc[:, :, :, IADSORBED, :, :]
    delta = doc_re - old_ads
    adsorb_all_free = (delta > 0.0) & (jnp.abs(delta) > old_free)
    desorb_all_ads = (delta < 0.0) & (jnp.abs(delta) > old_ads)
    normal = (delta != 0.0) & (~adsorb_all_free) & (~desorb_all_ads)
    new_free = jnp.where(
        adsorb_all_free,
        0.0,
        jnp.where(desorb_all_ads, old_free + old_ads, jnp.where(normal, old_free - delta, old_free)),
    )
    new_ads = jnp.where(
        adsorb_all_free,
        old_ads + old_free,
        jnp.where(desorb_all_ads, 0.0, jnp.where(normal, old_ads + delta, old_ads)),
    )
    doc = doc.at[:, :, :, IFREE, :, :].set(new_free)
    doc = doc.at[:, :, :, IADSORBED, :, :].set(new_ads)
    return SoilcarbonAdsorptionResult(doc=doc, kd=kd, doc_re=doc_re)


def soilcarbon_leak_water_transport(
    doc,
    soil_mc,
    soil_mc_32l,
    wat_flux,
    pref_soil_veg,
    zf_soil_b,
    flux_red,
    *,
    nslm: int | None = None,
) -> SoilcarbonWaterTransportResult:
    """Move free DOC vertically with hydrological water fluxes.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``,
    ``soilcarbon_leak`` lines 1840-1968. ``zf_soil_b`` follows the Fortran
    boundary convention ``zf_soil_B(0:ndeep)`` with Python length
    ``ndeep + 1``; ``pref_soil_veg`` is zero-based.
    """

    doc = jnp.asarray(doc)
    soil_mc = jnp.asarray(soil_mc)
    soil_mc_32l = jnp.asarray(soil_mc_32l)
    wat_flux = jnp.asarray(wat_flux)
    pref_soil_veg = jnp.asarray(pref_soil_veg, dtype=jnp.int32)
    zf_soil_b = jnp.asarray(zf_soil_b)
    flux_red = jnp.asarray(flux_red)
    if doc.ndim != 6 or doc.shape[3] != NDOC or doc.shape[4] != NPOOL:
        raise ValueError("doc must have shape (npts, nvm, ndeep, ndoc, npool, nelements)")
    npts, nvm, ndeep, _, _, nelements = doc.shape
    if soil_mc.ndim != 3 or soil_mc_32l.ndim != 3 or wat_flux.ndim != 3:
        raise ValueError("soil_mc, soil_mc_32l, and wat_flux must have shape (npts, nlayer, nstm)")
    if soil_mc.shape[0] != npts or soil_mc_32l.shape[0] != npts or wat_flux.shape[0] != npts:
        raise ValueError("soil and water arrays must share doc npts")
    if soil_mc_32l.shape[1] != ndeep:
        raise ValueError("soil_mc_32l layer axis must match doc ndeep")
    if pref_soil_veg.shape != (nvm,):
        raise ValueError("pref_soil_veg must have shape (nvm,)")
    if flux_red.shape != (npts,):
        raise ValueError("flux_red must have shape (npts,)")
    if nslm is None:
        nslm = soil_mc.shape[1]
    nslm = int(nslm)
    if nslm < 2 or nslm > ndeep or soil_mc.shape[1] < nslm or wat_flux.shape[1] < nslm:
        raise ValueError("nslm must be at least 2 and fit doc, soil_mc, and wat_flux")
    if zf_soil_b.ndim != 1 or zf_soil_b.shape[0] < ndeep + 1:
        raise ValueError("zf_soil_b must have length at least ndeep + 1")

    soil_pref = jnp.take(soil_mc[:, :nslm, :], pref_soil_veg, axis=2).transpose((0, 2, 1))
    soil32_pref = jnp.take(soil_mc_32l, pref_soil_veg, axis=2).transpose((0, 2, 1))
    wat_pref = jnp.take(wat_flux[:, :nslm, :], pref_soil_veg, axis=2).transpose((0, 2, 1))
    layer_thick = zf_soil_b[1 : ndeep + 1] - zf_soil_b[:ndeep]

    doc_conc = doc
    first_denom = zf_soil_b[1] * soil_pref[:, :, 0]
    first_free = doc_conc[:, :, 0, IFREE, :, :]
    first_free = jnp.where(
        soil_pref[:, :, 0, None, None] > 0.0,
        first_free / first_denom[:, :, None, None],
        first_free,
    )
    doc_conc = doc_conc.at[:, :, 0, IFREE, :, :].set(first_free)
    if ndeep > 1:
        denom = layer_thick[1:][None, None, :, None, None] * soil32_pref[:, :, 1:, None, None]
        deeper_free = doc_conc[:, :, 1:, IFREE, :, :]
        deeper_free = jnp.where(soil32_pref[:, :, 1:, None, None] > 0.0, deeper_free / denom, deeper_free)
        doc_conc = doc_conc.at[:, :, 1:, IFREE, :, :].set(deeper_free)

    doc_flux = jnp.zeros((npts, nvm, ndeep, NPOOL, nelements), dtype=doc.dtype)
    for layer in range(nslm - 1):
        positive = (wat_pref[:, :, layer] > 0.0) & (soil_pref[:, :, layer] > 0.0)
        negative = (wat_pref[:, :, layer] < 0.0) & (soil_pref[:, :, layer] > 0.0)
        down = doc_conc[:, :, layer, IFREE, :, :] * wat_pref[:, :, layer, None, None] * 1.0e-3 * flux_red[:, None, None, None]
        up = doc_conc[:, :, layer + 1, IFREE, :, :] * jnp.abs(wat_pref[:, :, layer, None, None]) * 1.0e-3 * flux_red[:, None, None, None]
        flux = jnp.where(positive[:, :, None, None], down, jnp.where(negative[:, :, None, None], up, 0.0))
        doc_flux = doc_flux.at[:, :, layer, :, :].set(flux)

    last_positive = (wat_pref[:, :, nslm - 1] > 0.0) & (soil_pref[:, :, nslm - 1] > 0.0)
    last_flux = (
        doc_conc[:, :, nslm - 1, IFREE, :, :]
        * wat_pref[:, :, nslm - 1, None, None]
        * 1.0e-3
        * flux_red[:, None, None, None]
    )
    doc_flux = doc_flux.at[:, :, nslm - 1, :, :].set(jnp.where(last_positive[:, :, None, None], last_flux, 0.0))

    doc_area = doc_conc
    first_free = doc_area[:, :, 0, IFREE, :, :]
    first_free = jnp.where(
        soil32_pref[:, :, 0, None, None] > 0.0,
        first_free * soil32_pref[:, :, 0, None, None] * zf_soil_b[1],
        first_free,
    )
    doc_area = doc_area.at[:, :, 0, IFREE, :, :].set(first_free)
    if ndeep > 1:
        deeper_free = doc_area[:, :, 1:, IFREE, :, :]
        deeper_free = jnp.where(
            soil32_pref[:, :, 1:, None, None] > 0.0,
            deeper_free * soil32_pref[:, :, 1:, None, None] * layer_thick[1:][None, None, :, None, None],
            deeper_free,
        )
        doc_area = doc_area.at[:, :, 1:, IFREE, :, :].set(deeper_free)

    old = doc_area[:, :, :, IFREE, :, :]
    free = old
    wf0 = wat_pref[:, :, 0]
    first = jnp.where(
        wf0[:, :, None, None] > 0.0,
        old[:, :, 0, :, :] - jnp.minimum(old[:, :, 0, :, :], doc_flux[:, :, 0, :, :]),
        jnp.where(
            wf0[:, :, None, None] < 0.0,
            old[:, :, 0, :, :] + jnp.minimum(old[:, :, 1, :, :], doc_flux[:, :, 0, :, :]),
            old[:, :, 0, :, :],
        ),
    )
    free = free.at[:, :, 0, :, :].set(first)

    for layer in range(1, nslm - 1):
        wf_layer = wat_pref[:, :, layer]
        wf_above = wat_pref[:, :, layer - 1]
        old_l = old[:, :, layer, :, :]
        both_down = old_l - jnp.minimum(doc_flux[:, :, layer, :, :], old_l) + jnp.minimum(
            doc_flux[:, :, layer - 1, :, :], old[:, :, layer - 1, :, :]
        )
        both_up = old_l + jnp.minimum(doc_flux[:, :, layer, :, :], old[:, :, layer + 1, :, :]) - jnp.minimum(
            doc_flux[:, :, layer - 1, :, :], old_l
        )
        diverge = old_l - jnp.minimum(doc_flux[:, :, layer, :, :], old_l) - jnp.minimum(doc_flux[:, :, layer - 1, :, :], old_l)
        converge = old_l + jnp.minimum(doc_flux[:, :, layer, :, :], old[:, :, layer + 1, :, :]) + jnp.minimum(
            doc_flux[:, :, layer - 1, :, :], old[:, :, layer - 1, :, :]
        )
        updated = jnp.where(
            ((wf_layer > 0.0) & (wf_above > 0.0))[:, :, None, None],
            both_down,
            jnp.where(
                ((wf_layer < 0.0) & (wf_above < 0.0))[:, :, None, None],
                both_up,
                jnp.where(
                    ((wf_layer > 0.0) & (wf_above < 0.0))[:, :, None, None],
                    diverge,
                    jnp.where(((wf_layer < 0.0) & (wf_above > 0.0))[:, :, None, None], converge, old_l),
                ),
            ),
        )
        free = free.at[:, :, layer, :, :].set(updated)

    wf_above_last = wat_pref[:, :, nslm - 2]
    last = jnp.where(
        wf_above_last[:, :, None, None] > 0.0,
        old[:, :, nslm - 1, :, :] + jnp.minimum(doc_flux[:, :, nslm - 2, :, :], old[:, :, nslm - 2, :, :]),
        jnp.where(
            wf_above_last[:, :, None, None] < 0.0,
            old[:, :, nslm - 1, :, :] - jnp.minimum(doc_flux[:, :, nslm - 2, :, :], old[:, :, nslm - 1, :, :]),
            old[:, :, nslm - 1, :, :],
        ),
    )
    free = free.at[:, :, nslm - 1, :, :].set(last)
    doc_updated = doc_area.at[:, :, :, IFREE, :, :].set(free)
    return SoilcarbonWaterTransportResult(doc=doc_updated, doc_flux=doc_flux)


def soilcarbon_leak_doc_diffusion(
    doc,
    tprof,
    zf_soil_b,
    dif_doc,
    *,
    min_stomate=1.0e-8,
) -> SoilcarbonDiffusionResult:
    """Diffuse free DOC between adjacent soil layers.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``,
    ``soilcarbon_leak`` lines 1970-2050. The temperature cutoff is the source
    threshold ``272.15 K`` on both adjacent layers.
    """

    doc = jnp.asarray(doc)
    tprof = jnp.asarray(tprof)
    zf_soil_b = jnp.asarray(zf_soil_b)
    dif_doc = jnp.asarray(dif_doc)
    if doc.ndim != 6 or doc.shape[3] != NDOC or doc.shape[4] != NPOOL:
        raise ValueError("doc must have shape (npts, nvm, ndeep, ndoc, npool, nelements)")
    npts, nvm, ndeep, _, _, nelements = doc.shape
    if ndeep < 2:
        raise ValueError("doc diffusion needs at least two layers")
    if tprof.shape != (npts, ndeep, nvm):
        raise ValueError("tprof must have shape (npts, ndeep, nvm)")
    if zf_soil_b.ndim != 1 or zf_soil_b.shape[0] < ndeep + 1:
        raise ValueError("zf_soil_b must have length at least ndeep + 1")
    if dif_doc.shape != (npts,):
        raise ValueError("dif_doc must have shape (npts,)")

    old = doc[:, :, :, IFREE, :, :]
    thickness = zf_soil_b[1 : ndeep + 1] - zf_soil_b[:ndeep]
    concentration = old / thickness[None, None, :, None, None]
    doc_flux_diff = jnp.zeros((npts, nvm, ndeep, NPOOL, nelements), dtype=doc.dtype)
    temp_pft = tprof.transpose((0, 2, 1))
    for layer in range(ndeep - 1):
        warm = (temp_pft[:, :, layer] > 272.15) & (temp_pft[:, :, layer + 1] > 272.15)
        flux = (
            dif_doc[:, None, None, None]
            * jnp.abs(concentration[:, :, layer, :, :] - concentration[:, :, layer + 1, :, :])
            / thickness[layer + 1]
        )
        doc_flux_diff = doc_flux_diff.at[:, :, layer, :, :].set(jnp.where(warm[:, :, None, None], flux, 0.0))

    for layer in range(ndeep - 2):
        c0 = concentration[:, :, layer, :, :]
        c1 = concentration[:, :, layer + 1, :, :]
        c2 = concentration[:, :, layer + 2, :, :]
        f0 = doc_flux_diff[:, :, layer, :, :]
        f1 = doc_flux_diff[:, :, layer + 1, :, :]
        peak_limited = (c0 < c1) & (c2 < c1) & ((f0 + f1) > old[:, :, layer + 1, :, :])
        denom = jnp.where((f0 + f1) != 0.0, f0 + f1, 1.0)
        f0_peak = old[:, :, layer + 1, :, :] * (f0 / denom)
        f1_peak = old[:, :, layer + 1, :, :] * (f1 / denom)
        branch2 = (c0 >= c1) & (c2 < c1) & ((old[:, :, layer + 1, :, :] + f0 - f1) <= min_stomate)
        branch3 = (c0 < c1) & (c2 >= c1) & ((old[:, :, layer + 1, :, :] - f0 + f1) <= min_stomate)
        new_f0 = jnp.where(peak_limited, f0_peak, jnp.where(branch3, old[:, :, layer + 1, :, :] + f1, f0))
        new_f1 = jnp.where(peak_limited, f1_peak, jnp.where(branch2, old[:, :, layer + 1, :, :] + f0, f1))
        doc_flux_diff = doc_flux_diff.at[:, :, layer, :, :].set(new_f0)
        doc_flux_diff = doc_flux_diff.at[:, :, layer + 1, :, :].set(new_f1)

    free = old
    buffer = jnp.zeros_like(old)
    for layer in range(ndeep - 1):
        c0 = concentration[:, :, layer, :, :]
        c1 = concentration[:, :, layer + 1, :, :]
        f = doc_flux_diff[:, :, layer, :, :]
        old0 = old[:, :, layer, :, :]
        old1 = old[:, :, layer + 1, :, :]
        buf0 = buffer[:, :, layer, :, :]
        up_limited = (c0 < c1) & ((f - old1) >= min_stomate)
        up_normal = (c0 < c1) & ((old1 - f) > min_stomate)
        down_limited = (c0 > c1) & ((f - old0) >= min_stomate)
        down_normal = (c0 > c1) & ((old0 - f) > min_stomate)
        new0 = jnp.where(
            up_limited,
            old0 + old1 + buf0,
            jnp.where(
                up_normal,
                old0 + f + buf0,
                jnp.where(down_limited, buf0, jnp.where(down_normal, old0 - f + buf0, old0)),
            ),
        )
        new1 = jnp.where(
            up_limited,
            0.0,
            jnp.where(
                up_normal,
                old1 - f,
                jnp.where(down_limited, old1 + old0, jnp.where(down_normal, old1 + f, old1)),
            ),
        )
        new_buf1 = jnp.where(
            up_limited,
            -old1,
            jnp.where(up_normal, -f, jnp.where(down_limited, old0, jnp.where(down_normal, f, buffer[:, :, layer + 1, :, :]))),
        )
        free = free.at[:, :, layer, :, :].set(new0)
        free = free.at[:, :, layer + 1, :, :].set(new1)
        buffer = buffer.at[:, :, layer + 1, :, :].set(new_buf1)

    doc_updated = doc.at[:, :, :, IFREE, :, :].set(free)
    return SoilcarbonDiffusionResult(doc=doc_updated, doc_flux_diff=doc_flux_diff)


def _remap_litter_doc_export(export_pool, pool: int, lignin_above, lignin_below):
    """Apply the source litter-pool remap for one just-updated pool."""

    if pool == IMETABO:
        export_pool = export_pool.at[:, IACT, :].add(export_pool[:, IMETABO, :])
        export_pool = export_pool.at[:, IMETABO, :].set(0.0)
    elif pool == ISTRABO:
        source = export_pool[:, ISTRABO, :]
        export_pool = export_pool.at[:, IACT, :].add(source * (1.0 - lignin_above[:, None]))
        export_pool = export_pool.at[:, ISLO, :].add(source * lignin_above[:, None])
        export_pool = export_pool.at[:, ISTRABO, :].set(0.0)
    elif pool == ISTRBEL:
        source = export_pool[:, ISTRBEL, :]
        export_pool = export_pool.at[:, IACT, :].add(source * (1.0 - lignin_below[:, None]))
        export_pool = export_pool.at[:, ISLO, :].add(source * lignin_below[:, None])
        export_pool = export_pool.at[:, ISTRBEL, :].set(0.0)
    elif pool == IMETBEL:
        export_pool = export_pool.at[:, IACT, :].add(export_pool[:, IMETBEL, :])
        export_pool = export_pool.at[:, IMETBEL, :].set(0.0)
    return export_pool


def _sqrt_with_finite_zero_tangent(value):
    """Keep the source primal while choosing the zero subgradient at zero."""

    return source_sqrt_with_finite_zero_tangent(jnp.asarray(value))


def soilcarbon_leak_doc_export(
    doc,
    soilwater_31mm,
    runoff_per_soil,
    drainage_per_soil,
    runoff2peat,
    fastr,
    flux_red,
    pref_soil_veg,
    veget_max,
    wet_dep_flood,
    floodcarbon_input,
    fluxtot_flood,
    lignin_struc_above,
    lignin_struc_below,
    soil_mc,
    soil_mc_32l,
    zf_soil_b,
    z_soil,
    is_peat,
    *,
    dt_days,
    cue_coef=0.3,
    sro_bottom=5,
    nslm: int | None = None,
    fastr_ref=25.0,
    docexp_max=20.0,
    min_stomate=1.0e-8,
) -> SoilcarbonLeakExportResult:
    """Compute DOC runoff, drainage, and flood export for ``soilcarbon_leak``.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``,
    ``soilcarbon_leak`` lines 2053-2303. ``pref_soil_veg`` is zero-based.
    """

    doc = jnp.asarray(doc)
    soilwater_31mm = jnp.asarray(soilwater_31mm)
    runoff_per_soil = jnp.asarray(runoff_per_soil)
    drainage_per_soil = jnp.asarray(drainage_per_soil)
    runoff2peat = jnp.asarray(runoff2peat)
    fastr = jnp.asarray(fastr)
    flux_red = jnp.asarray(flux_red)
    pref_soil_veg = jnp.asarray(pref_soil_veg, dtype=jnp.int32)
    veget_max = jnp.asarray(veget_max)
    wet_dep_flood = jnp.asarray(wet_dep_flood)
    floodcarbon_input = jnp.asarray(floodcarbon_input)
    fluxtot_flood = jnp.asarray(fluxtot_flood)
    lignin_struc_above = jnp.asarray(lignin_struc_above)
    lignin_struc_below = jnp.asarray(lignin_struc_below)
    soil_mc = jnp.asarray(soil_mc)
    soil_mc_32l = jnp.asarray(soil_mc_32l)
    zf_soil_b = jnp.asarray(zf_soil_b)
    z_soil = jnp.asarray(z_soil)
    is_peat = jnp.asarray(is_peat, dtype=bool)
    if doc.ndim != 6 or doc.shape[3] != NDOC or doc.shape[4] != NPOOL:
        raise ValueError("doc must have shape (npts, nvm, ndeep, ndoc, npool, nelements)")
    npts, nvm, ndeep, _, _, nelements = doc.shape
    if veget_max.shape != (npts, nvm) or wet_dep_flood.shape != (npts, nvm, nelements):
        raise ValueError("veget_max and wet_dep_flood must match doc npts/nvm/nelements")
    if floodcarbon_input.shape != (npts, nvm, NPOOL, nelements):
        raise ValueError("floodcarbon_input must have shape (npts, nvm, npool, nelements)")
    if fluxtot_flood.shape != (npts, NCARB, nelements, nvm, ndeep):
        raise ValueError("fluxtot_flood must have shape (npts, ncarb, nelements, nvm, ndeep)")
    if lignin_struc_above.shape != (npts, nvm) or lignin_struc_below.shape != (npts, nvm, ndeep):
        raise ValueError("lignin arrays must match doc dimensions")
    if soil_mc.ndim != 3 or soil_mc_32l.ndim != 3:
        raise ValueError("soil_mc and soil_mc_32l must have shape (npts, nlayer, nstm)")
    if nslm is None:
        nslm = soil_mc.shape[1]
    nslm = int(nslm)
    sro_bottom = int(sro_bottom)
    if nslm < 1 or nslm > ndeep or sro_bottom < 1 or sro_bottom > ndeep:
        raise ValueError("nslm and sro_bottom must select valid one-based layer counts")
    if zf_soil_b.ndim != 1 or zf_soil_b.shape[0] < ndeep + 1 or z_soil.ndim != 1 or z_soil.shape[0] < nslm + 1:
        raise ValueError("zf_soil_b and z_soil must include their zero boundary")
    if pref_soil_veg.shape != (nvm,) or is_peat.shape != (nvm,):
        raise ValueError("pref_soil_veg and is_peat must have shape (nvm,)")

    cue_coef = jnp.asarray(cue_coef, dtype=doc.dtype)
    if cue_coef.ndim == 0:
        cue_coef = jnp.full((npts,), cue_coef, dtype=doc.dtype)
    if cue_coef.shape != (npts,):
        raise ValueError("cue_coef must be scalar or have shape (npts,)")

    fastr_corr = jnp.maximum(
        _sqrt_with_finite_zero_tangent(fastr)
        / jnp.sqrt(jnp.asarray(fastr_ref, dtype=doc.dtype)),
        0.0,
    )
    doc_run = jnp.zeros((npts, nvm, NPOOL, nelements), dtype=doc.dtype)
    doc_drain = jnp.zeros_like(doc_run)
    doc_flood = jnp.zeros_like(doc_run)
    doc_exp = jnp.zeros((npts, nvm, NEXP, NPOOL, nelements), dtype=doc.dtype)
    doc_run_2_peat_all = jnp.zeros((npts, nvm, NPOOL, ndeep, nelements), dtype=doc.dtype)
    soil_doc_corr_all = jnp.zeros((npts, nvm), dtype=doc.dtype)
    doc_work = doc

    for pft in range(nvm):
        tile = pref_soil_veg[pft]
        soilwater = soilwater_31mm[:, tile]
        soil_doc_31mm = jnp.sum(doc_work[:, pft, :sro_bottom, IFREE, :, :], axis=(1, 2, 3))
        concentration = soil_doc_31mm * flux_red * fastr_corr / jnp.where(soilwater > min_stomate, soilwater, 1.0)
        soil_doc_corr = jnp.where(
            (soilwater > min_stomate) & (concentration > docexp_max),
            docexp_max / concentration,
            1.0,
        )
        soil_doc_corr = soil_doc_corr * flux_red * fastr_corr
        soil_doc_corr_all = soil_doc_corr_all.at[:, pft].set(soil_doc_corr)

        runoff2peat_factor = soil_doc_corr * runoff2peat[:, tile] * 1.0e-3 / jnp.where(soilwater > 0.0, soilwater, 1.0)
        doc_run_2_peat = jnp.zeros((npts, NPOOL, ndeep, nelements), dtype=doc.dtype)
        for layer in range(sro_bottom):
            loss = jnp.minimum(
                doc_work[:, pft, layer, IFREE, :, :],
                runoff2peat_factor[:, None, None] * doc_work[:, pft, layer, IFREE, :, :],
            )
            loss = jnp.where((soilwater > 0.0)[:, None, None] & is_peat[pft], loss, 0.0)
            doc_run_2_peat = doc_run_2_peat.at[:, :, layer, :].add(loss)
            doc_work = doc_work.at[:, pft, layer, IFREE, :, :].add(loss)
        doc_run_2_peat_all = doc_run_2_peat_all.at[:, pft, :, :, :].set(doc_run_2_peat)

        runoff_factor = soil_doc_corr * runoff_per_soil[:, tile] * 1.0e-3 / jnp.where(soilwater > 0.0, soilwater, 1.0)
        run_pft = doc_run[:, pft, :, :]
        for pool in range(NPOOL):
            for layer in range(sro_bottom):
                loss = jnp.minimum(
                    doc_work[:, pft, layer, IFREE, pool, :],
                    runoff_factor[:, None] * doc_work[:, pft, layer, IFREE, pool, :],
                )
                loss = jnp.where((soilwater > 0.0)[:, None], loss, 0.0)
                run_pft = run_pft.at[:, pool, :].add(loss)
                run_pft = _remap_litter_doc_export(
                    run_pft,
                    pool,
                    lignin_struc_above[:, pft],
                    lignin_struc_below[:, pft, layer],
                )
        doc_run = doc_run.at[:, pft, :, :].set(run_pft)

        drain_pft = doc_drain[:, pft, :, :]
        last = nslm - 1
        drain_denom_export = (zf_soil_b[nslm] - zf_soil_b[nslm - 1]) * soil_mc_32l[:, last, tile]
        drain_factor_export = drainage_per_soil[:, tile] * 1.0e-3 / jnp.where(drain_denom_export != 0.0, drain_denom_export, 1.0)
        for pool in range(NPOOL):
            loss = jnp.minimum(
                doc_work[:, pft, last, IFREE, pool, :],
                doc_work[:, pft, last, IFREE, pool, :] * drain_factor_export[:, None],
            )
            loss = jnp.where((soil_mc[:, last, tile] > 0.0)[:, None], loss, 0.0)
            drain_pft = drain_pft.at[:, pool, :].add(loss)
            drain_pft = _remap_litter_doc_export(
                drain_pft,
                pool,
                lignin_struc_above[:, pft],
                lignin_struc_below[:, pft, last],
            )
        doc_drain = doc_drain.at[:, pft, :, :].set(drain_pft)

        flood_pft = doc_flood[:, pft, :, :]
        has_veg = veget_max[:, pft] > 0.0
        safe_veg = jnp.where(has_veg, veget_max[:, pft], 1.0)
        flood_pft = flood_pft.at[:, IACT, :].add(jnp.where(has_veg[:, None], wet_dep_flood[:, pft, :] * 0.5 / safe_veg[:, None], 0.0))
        flood_pft = flood_pft.at[:, ISLO, :].add(jnp.where(has_veg[:, None], wet_dep_flood[:, pft, :] * 0.5 / safe_veg[:, None], 0.0))
        flood_pft = flood_pft.at[:, IACT, :].add(jnp.where(has_veg[:, None], floodcarbon_input[:, pft, IACT, :] * dt_days, 0.0))
        flood_pft = flood_pft.at[:, ISLO, :].add(jnp.where(has_veg[:, None], floodcarbon_input[:, pft, ISLO, :] * dt_days, 0.0))
        for layer in range(sro_bottom):
            flood_pft = flood_pft.at[:, ISLO, :].add(
                jnp.where(has_veg[:, None], cue_coef[:, None] * fluxtot_flood[:, ISLOW, :, pft, layer], 0.0)
            )
            flood_pft = flood_pft.at[:, IPAS, :].add(
                jnp.where(has_veg[:, None], cue_coef[:, None] * fluxtot_flood[:, IPASSIVE, :, pft, layer], 0.0)
            )
            flood_pft = flood_pft.at[:, IACT, :].add(
                jnp.where(has_veg[:, None], cue_coef[:, None] * fluxtot_flood[:, IACTIVE, :, pft, layer], 0.0)
            )
        doc_flood = doc_flood.at[:, pft, :, :].set(flood_pft)

        for layer in range(sro_bottom):
            runoff_loss = jnp.minimum(
                doc_work[:, pft, layer, IFREE, :, :],
                runoff_factor[:, None, None] * doc_work[:, pft, layer, IFREE, :, :],
            )
            runoff_loss = jnp.where((soilwater > 0.0)[:, None, None], runoff_loss, 0.0)
            doc_work = doc_work.at[:, pft, layer, IFREE, :, :].add(-runoff_loss)

        drain_denom_subtract = (z_soil[nslm] - z_soil[nslm - 1]) * soil_mc[:, last, tile]
        drain_factor_subtract = drainage_per_soil[:, tile] * 1.0e-3 / jnp.where(drain_denom_subtract != 0.0, drain_denom_subtract, 1.0)
        drain_loss = jnp.minimum(
            doc_work[:, pft, last, IFREE, :, :],
            doc_work[:, pft, last, IFREE, :, :] * drain_factor_subtract[:, None, None],
        )
        drain_loss = jnp.where((soil_mc[:, last, tile] > 0.0)[:, None, None], drain_loss, 0.0)
        doc_work = doc_work.at[:, pft, last, IFREE, :, :].add(-drain_loss)

    doc_exp = doc_exp.at[:, :, IRUNOFF, :, :].set(doc_run / dt_days)
    doc_exp = doc_exp.at[:, :, IFLOODED, :, :].set(doc_flood / dt_days)
    doc_exp = doc_exp.at[:, :, IDRAINAGE, :, :].set(doc_drain / dt_days)
    return SoilcarbonLeakExportResult(
        doc=doc_work,
        doc_exp=doc_exp,
        doc_run=doc_run,
        doc_drain=doc_drain,
        doc_flood=doc_flood,
        doc_run_2_peat=doc_run_2_peat_all,
        fastr_corr=fastr_corr,
        soil_doc_corr=soil_doc_corr_all,
    )


def soilcarbon_leak_core_step(
    carbon_32l,
    doc,
    soilcarbon_input_doc,
    doc_to_topsoil,
    doc_to_subsoil,
    wet_dep_ground,
    wet_dep_flood,
    litter_above,
    litter_below,
    lignin_struc_above,
    lignin_struc_below,
    fbact_doc,
    fbact,
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
    floodcarbon_input,
    tprof,
    clay,
    bulk_dens,
    z_soil,
    zf_soil_b,
    flux_red,
    natural,
    is_peat,
    is_c4,
    *,
    dt_days,
    dif_doc=None,
    cue=0.3,
    nslm: int | None = None,
    sro_bottom=5,
    priming: bool = True,
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
    min_stomate=0.0,
) -> SoilcarbonLeakCoreResult:
    """Run the source-backed ``soilcarbon_leak`` core in Fortran order.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``,
    ``soilcarbon_leak`` lines 1176-1193, 1348-1368, 1289-1302, and
    1528-2303. This composition excludes the still-separate TF-DOC canopy
    transfer. Cryoturbation and PERMA_PEAT are only executed when their
    explicit state is supplied by the caller.
    """

    if nslm is None:
        nslm = jnp.asarray(soil_mc).shape[1]
    nslm = int(nslm)
    if dif_doc is None:
        dif_doc = jnp.full((jnp.asarray(doc).shape[0],), 1.0e-5 * dt_days, dtype=jnp.asarray(doc).dtype)

    new_cryoturbation_coefficients = cryoturbation_coefficients
    if ok_cryoturb:
        required = {
            "altmax_ind": altmax_ind,
            "altmax_lastyear": altmax_lastyear,
            "fixed_cryoturbation_depth": fixed_cryoturbation_depth,
            "veget_mask": veget_mask,
            "zi_soil": zi_soil,
        }
        missing = tuple(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(f"ok_cryoturb requires explicit {', '.join(missing)}")
        cryo_result, new_cryoturbation_coefficients = soilcarbon_cryoturbation_cycle(
            carbon_32l,
            doc,
            litter_below,
            cryoturbation_coefficients,
            altmax_ind,
            altmax_lastyear,
            fixed_cryoturbation_depth,
            veget_mask,
            zi_soil,
            zf_soil_b,
            dt_seconds=dt_days * 86400.0,
            diff_k_const=cryoturbation_diff_k_in / (86400.0 * 365.0),
            bio_diff_k_const=bioturbation_diff_k_in / (86400.0 * 365.0),
            use_new_cryoturbation=use_new_cryoturbation,
            cryoturbation_method=cryoturbation_method,
            max_cryoturb_alt=max_cryoturb_alt,
            min_cryoturb_alt=min_cryoturb_alt,
            use_fixed_cryoturbation_depth=use_fixed_cryoturbation_depth,
            bioturbation_depth=bioturbation_depth,
        )
        carbon_32l = cryo_result.carbon_32l
        doc = cryo_result.doc
        litter_below = cryo_result.litter_below

    perma_peat_result = None
    if perma_peat:
        required = {
            "cmax_peat": cmax_peat,
            "perma_peat_veget_mask": perma_peat_veget_mask,
        }
        missing = tuple(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(f"perma_peat requires explicit {', '.join(missing)}")
        perma_peat_result = soilcarbon_perma_peat_redistribute(
            carbon_32l,
            cmax_peat,
            zf_soil_b,
            is_peat,
            perma_peat_veget_mask,
            frac1=frac1,
            frac2=frac2,
            min_stomate=min_stomate,
        )
        carbon_32l = perma_peat_result.carbon_32l

    controls = soilcarbon_leak_activity_factors(fbact_doc, fbact)
    doc_after_inputs = soilcarbon_leak_doc_inputs(
        doc,
        soilcarbon_input_doc,
        doc_to_topsoil,
        doc_to_subsoil,
        wet_dep_ground,
        veget_max,
        dt_days=dt_days,
        nslm=nslm,
    )
    decomp = soilcarbon_leak_decompose_update(
        carbon_32l,
        doc_after_inputs,
        litter_above,
        litter_below,
        lignin_struc_above,
        lignin_struc_below,
        controls.fbact_npool,
        controls.fbact_ncarb,
        soil_mc_32l,
        pref_soil_veg,
        flood_frac,
        clay,
        natural,
        is_peat,
        is_c4,
        dt_days=dt_days,
        cue=cue,
        priming=priming,
        nslm=nslm,
        sro_bottom=sro_bottom,
    )
    ads = soilcarbon_leak_adsorption_desorption(decomp.doc, clay, bulk_dens, z_soil, nslm=nslm)
    water = soilcarbon_leak_water_transport(
        ads.doc,
        soil_mc,
        soil_mc_32l,
        wat_flux,
        pref_soil_veg,
        zf_soil_b,
        flux_red,
        nslm=nslm,
    )
    diffusion = soilcarbon_leak_doc_diffusion(water.doc, tprof, zf_soil_b, dif_doc)
    export = soilcarbon_leak_doc_export(
        diffusion.doc,
        soilwater_31mm,
        runoff_per_soil,
        drainage_per_soil,
        runoff2peat,
        fastr,
        flux_red,
        pref_soil_veg,
        veget_max,
        wet_dep_flood,
        floodcarbon_input,
        decomp.fluxtot_flood,
        lignin_struc_above,
        lignin_struc_below,
        soil_mc,
        soil_mc_32l,
        zf_soil_b,
        z_soil,
        is_peat,
        dt_days=dt_days,
        cue_coef=cue,
        sro_bottom=sro_bottom,
        nslm=nslm,
    )
    return SoilcarbonLeakCoreResult(
        carbon_32l=decomp.carbon_32l,
        doc=export.doc,
        litter_below=litter_below,
        doc_exp=export.doc_exp,
        resp_hetero_soil=decomp.resp_hetero_soil,
        resp_flood_soil=decomp.resp_flood_soil,
        fluxtot=decomp.fluxtot,
        fluxtot_flood=decomp.fluxtot_flood,
        fluxtot_doc=decomp.fluxtot_doc,
        fluxtot_doc_flood=decomp.fluxtot_doc_flood,
        litter_tot=decomp.litter_tot,
        lom=decomp.lom,
        kd=ads.kd,
        doc_flux=water.doc_flux,
        doc_flux_diff=diffusion.doc_flux_diff,
        doc_run=export.doc_run,
        doc_drain=export.doc_drain,
        doc_flood=export.doc_flood,
        fastr_corr=export.fastr_corr,
        soil_doc_corr=export.soil_doc_corr,
        cryoturbation_coefficients=new_cryoturbation_coefficients,
        perma_peat=perma_peat_result,
    )


SOILCARBON_LEAK_CORE_STEP_JIT_STATIC_ARGNAMES = (
    "nslm",
    "sro_bottom",
    "priming",
    "ok_cryoturb",
    "use_new_cryoturbation",
    "cryoturbation_method",
    "use_fixed_cryoturbation_depth",
    "perma_peat",
)


soilcarbon_leak_core_step_jit = jit(
    soilcarbon_leak_core_step,
    static_argnames=SOILCARBON_LEAK_CORE_STEP_JIT_STATIC_ARGNAMES,
)


def soilcarbon_leak_doc_export_aggregate(
    doc_exp,
    veget_max,
    resp_hetero_litter,
    resp_hetero_soil,
    resp_hetero_flood,
    resp_flood_soil,
    resp_maint_part_radia,
    flood_root_radia,
    runoff_per_soil,
    drainage_per_soil,
    pref_soil_veg,
    flood_frac,
    *,
    dt_days,
    min_sechiba=0.0,
) -> SoilcarbonDocExportAggregate:
    """Aggregate ``DOC_EXP`` and DIC terms after ``soilcarbon_leak``.

    Fortran provenance: ``src_stomate/stomate.f90``, subroutine
    ``stomate_main``, lines 3415-3458.
    """

    doc_exp = jnp.asarray(doc_exp)
    veget_max = jnp.asarray(veget_max)
    npts, nvm, _, _, nelements = doc_exp.shape
    doc_exp_agg = jnp.zeros((npts, NEXP, ICO2AQ + 1), dtype=doc_exp.dtype)
    doc_exp_b = jnp.zeros((npts, nvm, NEXP, ICO2AQ + 1, nelements), dtype=doc_exp.dtype)

    active_pft = jnp.arange(nvm) > 0
    labile = jnp.nan_to_num(doc_exp[:, :, :, : IACT + 1, ICARBON], nan=0.0)
    refractory = jnp.nan_to_num(doc_exp[:, :, :, ISLO : IPAS + 1, ICARBON], nan=0.0)
    labile_sum = jnp.sum(labile, axis=3)
    refractory_sum = jnp.sum(refractory, axis=3)
    weighted = veget_max[:, :, None] * active_pft[None, :, None]
    doc_exp_agg = doc_exp_agg.at[:, :, IDOCL].set(jnp.sum(labile_sum * weighted, axis=1) * dt_days)
    doc_exp_agg = doc_exp_agg.at[:, :, IDOCR].set(jnp.sum(refractory_sum * weighted, axis=1) * dt_days)
    doc_exp_b = doc_exp_b.at[:, :, :, IDOCL, ICARBON].set(labile_sum * weighted)
    doc_exp_b = doc_exp_b.at[:, :, :, IDOCR, ICARBON].set(refractory_sum * weighted)

    dry_fraction = 1.0 - flood_frac[:, None]
    soil_resp_modif = jnp.where(
        dry_fraction > min_sechiba,
        (
            resp_hetero_litter[:, :, IBELOW]
            + resp_hetero_soil
            + resp_maint_part_radia[:, :, IROOT]
        )
        / (4.25 * dry_fraction * dt_days),
        0.0,
    )
    tile = jnp.asarray(pref_soil_veg, dtype=jnp.int32)
    runoff_by_pft = runoff_per_soil[:, tile]
    drainage_by_pft = drainage_per_soil[:, tile]
    dic_runoff = runoff_by_pft * 20e-4 * veget_max * soil_resp_modif
    dic_drain = drainage_by_pft * 20e-3 * veget_max * soil_resp_modif
    dic_flood = (resp_hetero_flood + resp_flood_soil + flood_root_radia) * veget_max
    doc_exp_agg = doc_exp_agg.at[:, IRUNOFF, ICO2AQ].add(jnp.sum(jnp.where(active_pft[None, :], dic_runoff, 0.0), axis=1))
    doc_exp_agg = doc_exp_agg.at[:, IDRAINAGE, ICO2AQ].add(jnp.sum(jnp.where(active_pft[None, :], dic_drain, 0.0), axis=1))
    doc_exp_agg = doc_exp_agg.at[:, IFLOODED, ICO2AQ].add(jnp.sum(jnp.where(active_pft[None, :], dic_flood, 0.0), axis=1))
    return SoilcarbonDocExportAggregate(doc_exp_agg=doc_exp_agg, doc_exp_b=doc_exp_b, soil_resp_modif=soil_resp_modif)


def soilcarbon_pft14_owner(
    dt,
    clay,
    soilcarbon_input,
    control_temp,
    control_moist,
    carbon,
    matrix_a,
    *,
    natural,
    is_peat,
    is_c4,
    firstcall_soilcarbon=True,
    carbon_tau=None,
    ok_leak=False,
    spinup_analytic=False,
    ok_peat=False,
    height_acro=None,
    carbon_acro=None,
    carbon_cato=None,
    wtp_peat=None,
    one_year=365.0,
    below_index=IBELOW,
    metabolic_ref_frac=0.85,
    active_to_pass_clay_frac=0.68,
    frac_carb_ap=0.004,
    frac_carb_sa=0.42,
    frac_carb_sp=0.03,
    frac_carb_pa=0.45,
    frac_carb_ps=0.0,
    carbon_tau_iactive=0.149,
    carbon_tau_islow=5.48,
    carbon_tau_ipassive=241.0,
    flux_tot_coeff=(1.2, 1.4, 0.75),
    wtd_min=300.0,
    p_a=3.5e4,
    p_c=9.1e4,
    cf_a=0.50,
    cf_c=0.52,
    v_ratio=0.35,
    ka_ini=0.067,
    kp_ini=1.91e-2,
    kc_ini=3.35e-5,
) -> SoilcarbonResult:
    """Execute ``stomate_soilcarbon::soilcarbon`` in source order.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``, subroutine
    ``soilcarbon``, lines 212-632. This owns first-call SAVE state (312-339),
    three-pool fluxes (346-436), analytical spinup (442-572), and the optional
    legacy peat pools (576-630). The paper ``OK_LEAK=y`` dispatches to
    ``soilcarbon_leak`` before this alternate routine; callers must therefore
    select this owner explicitly rather than treating it as the paper path.
    """

    clay = jnp.asarray(clay)
    soilcarbon_input = jnp.asarray(soilcarbon_input)
    control_temp = jnp.asarray(control_temp)
    control_moist = jnp.asarray(control_moist)
    carbon = jnp.asarray(carbon)
    matrix_a = jnp.asarray(matrix_a)
    natural = jnp.asarray(natural, dtype=bool)
    is_peat = jnp.asarray(is_peat, dtype=bool)
    is_c4 = jnp.asarray(is_c4, dtype=bool)

    if bool(ok_leak):
        raise ValueError(
            "OK_LEAK=y selects soilcarbon_leak, so legacy soilcarbon is unreachable "
            "(stomate.f90 stomate_main lines 3384-3522)"
        )

    if carbon.ndim != 3 or carbon.shape[1] != NCARB:
        raise ValueError("carbon must have shape (npts, 3, nvm)")
    npts, _, nvm = carbon.shape
    if clay.shape != (npts,):
        raise ValueError("clay must have shape (npts,)")
    if soilcarbon_input.shape != carbon.shape:
        raise ValueError("soilcarbon_input must match carbon shape")
    if control_temp.ndim != 2 or control_temp.shape[0] != npts:
        raise ValueError("control_temp must have shape (npts, nlevs)")
    if control_moist.shape != control_temp.shape:
        raise ValueError("control_moist must match control_temp shape")
    if not 0 <= int(below_index) < control_temp.shape[1]:
        raise ValueError("below_index is outside the control arrays")
    if natural.shape != (nvm,) or is_peat.shape != (nvm,) or is_c4.shape != (nvm,):
        raise ValueError("natural, is_peat, and is_c4 must have shape (nvm,)")
    if matrix_a.ndim != 4 or matrix_a.shape[:2] != (npts, nvm) or matrix_a.shape[2] != matrix_a.shape[3]:
        raise ValueError("matrix_a must have shape (npts, nvm, nbpools, nbpools)")
    if matrix_a.shape[2] < 7:
        raise ValueError("matrix_a needs the seven Fortran carbon pools")
    if bool(is_peat[0]):
        raise ValueError("bare-soil PFT cannot be peat (soilcarbon lines 360 and 606)")

    dtype = jnp.result_type(carbon, soilcarbon_input, clay, jnp.float64)
    dt = jnp.asarray(dt, dtype=dtype)
    one_year = jnp.asarray(one_year, dtype=dtype)
    if bool(firstcall_soilcarbon):
        carbon_tau = jnp.asarray(
            (carbon_tau_iactive, carbon_tau_islow, carbon_tau_ipassive), dtype=dtype
        ) * one_year
    else:
        if carbon_tau is None:
            raise ValueError(
                "carbon_tau SAVE state is required after first call "
                "(stomate_soilcarbon.f90 soilcarbon lines 312-339)"
            )
        carbon_tau = jnp.asarray(carbon_tau, dtype=dtype)
        if carbon_tau.shape != (NCARB,):
            raise ValueError("carbon_tau must have shape (3,)")

    frac_carb = jnp.zeros((npts, NCARB, NCARB), dtype=dtype)
    frac_carb = frac_carb.at[:, IACTIVE, IPASSIVE].set(frac_carb_ap)
    frac_carb = frac_carb.at[:, IACTIVE, ISLOW].set(
        1.0 - (metabolic_ref_frac - active_to_pass_clay_frac * clay) - frac_carb_ap
    )
    frac_carb = frac_carb.at[:, ISLOW, IACTIVE].set(frac_carb_sa)
    frac_carb = frac_carb.at[:, ISLOW, IPASSIVE].set(frac_carb_sp)
    frac_carb = frac_carb.at[:, IPASSIVE, IACTIVE].set(frac_carb_pa)
    frac_carb = frac_carb.at[:, IPASSIVE, ISLOW].set(frac_carb_ps)
    frac_resp = 1.0 - frac_carb[:, :, IACTIVE] - frac_carb[:, :, ISLOW] - frac_carb[:, :, IPASSIVE]

    carbon = carbon + soilcarbon_input * dt
    resp_hetero_soil = jnp.zeros((npts, nvm), dtype=dtype)
    moisture_temperature = control_moist[:, int(below_index)] * control_temp[:, int(below_index)]
    coeff = jnp.asarray(flux_tot_coeff, dtype=dtype)

    for m in range(1, nvm):
        if bool(is_peat[m]):
            continue
        if bool(natural[m]):
            pft_factor = jnp.asarray(1.0, dtype=dtype)
        elif not bool(is_c4[m]):
            pft_factor = coeff[0]
        else:
            pft_factor = coeff[1]

        flux_total = []
        flux = jnp.zeros((npts, NCARB, NCARB), dtype=dtype)
        for source_pool in range(NCARB):
            total = dt / carbon_tau[source_pool] * carbon[:, source_pool, m] * moisture_temperature * pft_factor
            if source_pool == IACTIVE:
                total = total * (1.0 - coeff[2] * clay)
            carbon = carbon.at[:, source_pool, m].add(-total)
            flux_total.append(total)
            for destination_pool in range(NCARB):
                flux = flux.at[:, source_pool, destination_pool].set(
                    frac_carb[:, source_pool, destination_pool] * total
                )

        flux_total = jnp.stack(flux_total, axis=1)
        resp_hetero_soil = resp_hetero_soil.at[:, m].set(
            (
                frac_resp[:, IACTIVE] * flux_total[:, IACTIVE]
                + frac_resp[:, ISLOW] * flux_total[:, ISLOW]
                + frac_resp[:, IPASSIVE] * flux_total[:, IPASSIVE]
            ) / dt
        )
        for destination_pool in range(NCARB):
            carbon = carbon.at[:, destination_pool, m].add(
                flux[:, IACTIVE, destination_pool]
                + flux[:, IPASSIVE, destination_pool]
                + flux[:, ISLOW, destination_pool]
            )

    if bool(spinup_analytic):
        active_pool, slow_pool, passive_pool = 4, 5, 6
        for m in range(1, nvm):
            base = dt * moisture_temperature
            active_clay = 1.0 - coeff[2] * clay
            values = (
                (active_pool, active_pool, -base / carbon_tau[IACTIVE] * active_clay),
                (active_pool, slow_pool, frac_carb[:, ISLOW, IACTIVE] * base / carbon_tau[ISLOW]),
                (active_pool, passive_pool, frac_carb[:, IPASSIVE, IACTIVE] * base / carbon_tau[IPASSIVE]),
                (slow_pool, active_pool, frac_carb[:, IACTIVE, ISLOW] * base / carbon_tau[IACTIVE] * active_clay),
                (slow_pool, slow_pool, -base / carbon_tau[ISLOW]),
                (passive_pool, active_pool, frac_carb[:, IACTIVE, IPASSIVE] * base / carbon_tau[IACTIVE] * active_clay),
                (passive_pool, slow_pool, frac_carb[:, ISLOW, IPASSIVE] * base / carbon_tau[ISLOW]),
                (passive_pool, passive_pool, -base / carbon_tau[IPASSIVE]),
            )
            crop_factor = jnp.asarray(1.0, dtype=dtype)
            if not bool(natural[m]):
                crop_factor = coeff[1] if bool(is_c4[m]) else coeff[0]
            for destination_pool, source_pool, value in values:
                matrix_a = matrix_a.at[:, m, destination_pool, source_pool].set(value * crop_factor)
        diagonal = jnp.arange(matrix_a.shape[2])
        matrix_a = matrix_a.at[:, :, diagonal, diagonal].add(1.0)

    peat_outputs = (None,) * 11
    if bool(ok_peat):
        required = {
            "height_acro": height_acro,
            "carbon_acro": carbon_acro,
            "carbon_cato": carbon_cato,
            "wtp_peat": wtp_peat,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "OK_PEAT requires " + ", ".join(missing)
                + " (stomate_soilcarbon.f90 soilcarbon lines 576-630)"
            )
        height_acro = jnp.asarray(height_acro, dtype=dtype)
        carbon_acro = jnp.asarray(carbon_acro, dtype=dtype)
        carbon_cato = jnp.asarray(carbon_cato, dtype=dtype)
        wtp_peat = jnp.asarray(wtp_peat, dtype=dtype)
        if height_acro.shape != (npts,) or wtp_peat.shape != (npts,):
            raise ValueError("height_acro and wtp_peat must have shape (npts,)")
        if carbon_acro.shape != (npts, nvm) or carbon_cato.shape != (npts, nvm):
            raise ValueError("carbon_acro and carbon_cato must have shape (npts, nvm)")

        active_nonpeat = (~is_peat) & (jnp.arange(nvm) > 0)
        carbon_acro = jnp.where(active_nonpeat[None, :], 0.0, carbon_acro)
        carbon_cato = jnp.where(active_nonpeat[None, :], 0.0, carbon_cato)
        ka = control_temp[:, int(below_index)] * ka_ini * dt / one_year
        kp = control_temp[:, int(below_index)] * kp_ini * dt / one_year
        kc = kc_ini * dt / one_year
        wtd = (jnp.asarray(wtd_min, dtype=dtype) - wtp_peat) * 0.001
        safe_height = jnp.where(height_acro != 0.0, height_acro, 1.0)
        b = jnp.where(wtd <= 0.0, 1.0, jnp.where(wtd < height_acro, (height_acro - wtd) / safe_height, 0.0))

        resp_acro_oxic = jnp.zeros((npts, nvm), dtype=dtype)
        resp_acro_anoxic = jnp.zeros((npts, nvm), dtype=dtype)
        resp_cato = jnp.zeros((npts, nvm), dtype=dtype)
        acro_to_cato = jnp.zeros((npts, nvm), dtype=dtype)
        litter_to_acro = jnp.zeros((npts, nvm), dtype=dtype)
        tcarbon_acro = jnp.zeros((npts,), dtype=dtype)
        tcarbon_cato = jnp.zeros((npts,), dtype=dtype)
        height_cato = jnp.zeros((npts,), dtype=dtype)
        for m in range(1, nvm):
            if not bool(is_peat[m]):
                continue
            carbon = carbon.at[:, :, m].set(0.0)
            oxic = b * ka * carbon_acro[:, m]
            anoxic = (1.0 - b) * v_ratio * ka * carbon_acro[:, m]
            to_cato = kp * carbon_acro[:, m]
            cato_resp = kc * carbon_cato[:, m]
            litter = (soilcarbon_input[:, IACTIVE, m] + soilcarbon_input[:, ISLOW, m]) * dt
            resp_acro_oxic = resp_acro_oxic.at[:, m].set(oxic)
            resp_acro_anoxic = resp_acro_anoxic.at[:, m].set(anoxic)
            acro_to_cato = acro_to_cato.at[:, m].set(to_cato)
            resp_cato = resp_cato.at[:, m].set(cato_resp)
            litter_to_acro = litter_to_acro.at[:, m].set(litter)
            resp_hetero_soil = resp_hetero_soil.at[:, m].set((oxic + anoxic + cato_resp) / dt)
            carbon_acro = carbon_acro.at[:, m].add(litter - to_cato - oxic - anoxic)
            carbon_cato = carbon_cato.at[:, m].add(to_cato - cato_resp)
            tcarbon_acro = tcarbon_acro + carbon_acro[:, m]
            tcarbon_cato = tcarbon_cato + carbon_cato[:, m]
            height_acro = tcarbon_acro / (p_a * cf_a)
            height_cato = tcarbon_cato / (p_c * cf_c)
        peat_outputs = (
            height_acro, height_cato, carbon_acro, carbon_cato, tcarbon_acro,
            tcarbon_cato, resp_acro_oxic, resp_acro_anoxic, resp_cato,
            acro_to_cato, litter_to_acro,
        )

    return SoilcarbonResult(
        carbon=carbon,
        resp_hetero_soil=resp_hetero_soil,
        matrix_a=matrix_a,
        carbon_tau=carbon_tau,
        firstcall_soilcarbon=False,
        height_acro=peat_outputs[0],
        height_cato=peat_outputs[1],
        carbon_acro=peat_outputs[2],
        carbon_cato=peat_outputs[3],
        tcarbon_acro=peat_outputs[4],
        tcarbon_cato=peat_outputs[5],
        resp_acro_oxic=peat_outputs[6],
        resp_acro_anoxic=peat_outputs[7],
        resp_cato=peat_outputs[8],
        acro_to_cato=peat_outputs[9],
        litter_to_acro=peat_outputs[10],
    )
