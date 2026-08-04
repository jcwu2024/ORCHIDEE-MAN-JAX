"""Small source-backed DIFFUCO kernels for the PFT14 paper-case path.

These helpers intentionally cover only closed-form algebra audited from
``src_sechiba/diffuco.f90``. They do not run ``diffuco_main`` and they do not
infer untraced mangrove control internals from bridge outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, NamedTuple

from jax import config, core, jit

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jax_orchidee.driver.init import parse_run_def_bool, parse_run_def_float, parse_run_def_indexed_bool, parse_run_def_indexed_float
from jax_orchidee.driver.restart import read_restart_fields
from jax_orchidee.parameters.vertical_soil import vertical_soil_init
from jax_orchidee.sechiba.condveg import condveg_frac_snow

from jax_orchidee.sechiba.enerbil import (
    driver_swnet_from_swdown_albedo,
    qsat_moisture_qsatcalc,
    sechiba_air_density_from_pb_temp_air,
)
from jax_orchidee.stomate.carbon_kernels import (
    IAGRSAPPN,
    IAGRSAPST,
    IAGRHRTPN,
    IAGRHRTST,
    ICARBON,
)


MIN_SECHIBA = 1.0e-8
TP_00 = 273.15
PB_STD = 1013.0
UNDEF_SECHIBA = 1.0e20
CP_AIR = 1004.675
CTE_GRAV = 9.80665
MSMLR_AIR = 28.964e-3
MSMLR_H2O = 18.02e-3
RET_V = MSMLR_AIR / MSMLR_H2O - 1.0
RV_TMP2 = (4.0 * MSMLR_AIR) / (3.5 * MSMLR_H2O) - 1.0
CEPDU2 = 0.1**2
CT_KARMAN = 0.41
LOUIS_CB = 5.0
LOUIS_CC = 5.0
LOUIS_CD = 5.0
RATIO_H2O_TO_CO2 = 1.6
MOL_TO_M_1 = 0.0244
TETENS_1 = 0.622
TETENS_2 = 0.378
RG_TO_PAR = 0.5
W_TO_MOL = 4.6
DEFAULT_NLAI = 20
DEFAULT_LAIMAX = 12.0
DEFAULT_LAI_LEVEL_DEPTH = 0.15

DIFFUCO_INITIALIZE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_initialize lines 127-233",
    "fortran_source/ORCHIDEE/src_sechiba/sechiba_io_p.f90::r30setvar_p lines 485-508",
)


class DiffucoInitializeBoundaryError(NotImplementedError):
    """A reachable initializer branch is owned by an unported external module."""


@dataclass(frozen=True)
class DiffucoInitializeResult:
    """State written by the PFT14 path through ``diffuco_initialize``."""

    rstruct: jnp.ndarray
    q_cdrag_pft: jnp.ndarray | None
    leaf_ci: jnp.ndarray
    wind: None
    control_salinity: None
    control_inudate: None
    ldq_cdrag_from_gcm: bool
    provenance: tuple[str, ...] = DIFFUCO_INITIALIZE_PROVENANCE

    def as_payload(self) -> dict[str, object]:
        payload = {
            "rstruct": self.rstruct,
            "leaf_ci": self.leaf_ci,
            "ldq_cdrag_from_gcm": self.ldq_cdrag_from_gcm,
        }
        # Lines 178-180 make no assignment when the restart contains only
        # val_exp. Do not manufacture an allocation value in that case.
        if self.q_cdrag_pft is not None:
            payload["q_cdrag_pft"] = self.q_cdrag_pft
        return payload


def _diffuco_restart_array(restart_fields, name, shape, *, val_exp):
    value = restart_fields.get(name)
    if value is None:
        return jnp.full(shape, val_exp, dtype=jnp.float64)
    value = jnp.asarray(value, dtype=jnp.float64)
    if value.shape != shape:
        raise ValueError(f"restart {name} must have shape {shape}, got {value.shape}")
    return value


def diffuco_initialize(
    *,
    q_cdrag,
    rstruct_const,
    nlai: int,
    restart_fields: Mapping[str, object] | None = None,
    cdrag_from_gcm: bool | None = None,
    diffuco_leafci: float = 233.0,
    ok_co2: bool = True,
    ok_bvoc: bool = False,
    val_exp: float = 999999.0,
) -> DiffucoInitializeResult:
    """Execute the source-order PFT14 initialization owned by DIFFUCO.

    Fortran provenance: ``diffuco.f90::diffuco_initialize`` lines 127-233.
    Restart reads are explicit mappings. The paper path has ``OK_CO2=y`` and
    ``CHEMISTRY_BVOC=n``. The BVOC branch at lines 229-231 is rejected because
    it structurally transfers ownership to ``chemistry_initialize``.
    """

    if not bool(ok_co2):
        raise DiffucoInitializeBoundaryError(
            "ok_co2=False is outside the audited PFT14 paper path (diffuco_initialize lines 185-189, 212-227)"
        )
    if bool(ok_bvoc):
        raise DiffucoInitializeBoundaryError(
            "ok_bvoc=True requires external chemistry_initialize (diffuco_initialize lines 229-231)"
        )
    nlai = int(nlai)
    if nlai < 1:
        raise ValueError("nlai must be positive")
    q_cdrag = jnp.asarray(q_cdrag, dtype=jnp.float64)
    rstruct_const = jnp.asarray(rstruct_const, dtype=jnp.float64)
    if q_cdrag.ndim != 1:
        raise ValueError("q_cdrag must have shape (npts,)")
    if rstruct_const.ndim != 1 or rstruct_const.shape[0] < 14:
        raise ValueError("rstruct_const must have shape (nvm,) with nvm >= 14")
    npts, nvm = q_cdrag.shape[0], rstruct_const.shape[0]
    restart_fields = {} if restart_fields is None else dict(restart_fields)

    # Lines 167-172: the run.def value, when supplied, overrides the value
    # inferred from the incoming continuous q_cdrag field.
    inferred_cdrag = abs(float(jnp.max(q_cdrag))) > float(jnp.finfo(jnp.float64).eps)
    ldq_cdrag_from_gcm = inferred_cdrag if cdrag_from_gcm is None else bool(cdrag_from_gcm)

    temppft = _diffuco_restart_array(restart_fields, "cdrag_pft", (npts, nvm), val_exp=val_exp)
    q_cdrag_pft = None
    if bool(jnp.min(temppft) < jnp.max(temppft)) or float(jnp.max(temppft)) != float(val_exp):
        q_cdrag_pft = temppft

    rstruct = _diffuco_restart_array(restart_fields, "rstruct", (npts, nvm), val_exp=val_exp)
    if bool(jnp.all(rstruct == val_exp)):
        rstruct = jnp.broadcast_to(rstruct_const[None, :], (npts, nvm))

    leaf_ci = _diffuco_restart_array(restart_fields, "leaf_ci", (npts, nvm, nlai), val_exp=val_exp)
    if bool(jnp.all(leaf_ci == val_exp)):
        leaf_ci = jnp.full((npts, nvm, nlai), diffuco_leafci, dtype=jnp.float64)

    # Lines 191-198 allocate these module arrays but do not assign values.
    # None records that no scientific value exists at this boundary.
    return DiffucoInitializeResult(
        rstruct=rstruct,
        q_cdrag_pft=q_cdrag_pft,
        leaf_ci=leaf_ci,
        wind=None,
        control_salinity=None,
        control_inudate=None,
        ldq_cdrag_from_gcm=ldq_cdrag_from_gcm,
    )
ALPHA_LL_MTC = (
    -9999.0,
    0.3,
    0.3,
    0.3,
    0.3,
    0.3,
    0.3,
    0.3,
    0.3,
    0.3,
    0.3,
    0.3,
    0.3,
    0.3,
    0.3,
)
GB_REF = 0.04
RR_GAS = 8.314
DEW_VEG_POLY_COEFF = (
    0.887773,
    0.205673,
    0.110112,
    0.014843,
    0.000824,
    0.000017,
)
HUMCSTE_REF4M = (
    5.0,
    0.4,
    0.4,
    1.0,
    0.8,
    0.8,
    1.0,
    1.0,
    0.8,
    4.0,
    1.0,
    4.0,
    1.0,
    4.0,
    10.0,
    4.0,
    1.0,
    0.4,
)
HUMCSTE_REF2M = (
    5.0,
    0.8,
    0.8,
    1.0,
    0.8,
    0.8,
    1.0,
    1.0,
    0.8,
    4.0,
    4.0,
    4.0,
    4.0,
    4.0,
    10.0,
    4.0,
    4.0,
    0.8,
)


@dataclass(frozen=True)
class DiffucoInputCoverage:
    """Coverage status for one DIFFUCO inundation-control input."""

    status: str
    provenance: tuple[str, ...]
    detail: str = ""


@dataclass(frozen=True)
class DiffucoControlInundationInputCoverage:
    """Read-only coverage report for the PFT14 inundation-control input chain.

    This is an audit contract, not a value builder. A field is marked covered
    only when the caller has an explicit full input source for the DIFFUCO
    call boundary, or constructible when Fortran gives a direct algebraic
    mapping from a supplied upstream value.
    """

    z_soil: DiffucoInputCoverage
    rprof: DiffucoInputCoverage
    tide_height: DiffucoInputCoverage
    biomass: DiffucoInputCoverage

    @property
    def complete(self) -> bool:
        return not self.missing_inputs

    @property
    def covered_inputs(self) -> tuple[str, ...]:
        fields = {
            "z_soil": self.z_soil,
            "rprof": self.rprof,
            "tide_height": self.tide_height,
            "biomass": self.biomass,
        }
        return tuple(name for name, field in fields.items() if field.status in {"covered", "constructible"})

    @property
    def missing_inputs(self) -> tuple[str, ...]:
        fields = {
            "z_soil": self.z_soil,
            "rprof": self.rprof,
            "tide_height": self.tide_height,
            "biomass": self.biomass,
        }
        return tuple(name for name, field in fields.items() if field.status == "missing")


class DiffucoCombinedBetaResult(NamedTuple):
    """Final beta bundle after the `diffuco_comb` near-zero consistency rule."""

    valpha: jnp.ndarray
    vbeta: jnp.ndarray
    vbeta2: jnp.ndarray
    vbeta3: jnp.ndarray
    vbeta4: jnp.ndarray


class DiffucoBareCWRRBetaResult(NamedTuple):
    """Bare-soil beta outputs from the active CWRR branch."""

    vbeta4: jnp.ndarray
    vbeta4_pft: jnp.ndarray


class DiffucoInterResult(NamedTuple):
    """Vegetation interception beta outputs from ``diffuco_inter``."""

    vbeta2: jnp.ndarray
    vbeta23: jnp.ndarray
    zqsvegrap: jnp.ndarray
    ziltest: jnp.ndarray
    zrapp: jnp.ndarray
    active: jnp.ndarray
    limited_by_storage: jnp.ndarray


class DiffucoSnowBetaResult(NamedTuple):
    """Snow sublimation beta outputs from ``diffuco_snow``."""

    vbeta1: jnp.ndarray
    vegetation_base: jnp.ndarray
    vegetation_subtest: jnp.ndarray
    vegetation_zrapp: jnp.ndarray
    nobio_add: jnp.ndarray
    nobio_subtest: jnp.ndarray
    nobio_zrapp: jnp.ndarray


class DiffucoFloodBetaResult(NamedTuple):
    """Floodplain evaporation beta outputs from ``diffuco_flood``."""

    vbeta5: jnp.ndarray
    base: jnp.ndarray
    subtest: jnp.ndarray
    zrapp: jnp.ndarray


class DiffucoDragBoundaryResult(NamedTuple):
    """Drag fields and aerodynamic resistance before DIFFUCO beta kernels."""

    q_cdrag: jnp.ndarray
    q_cdrag_pft: jnp.ndarray
    raero: jnp.ndarray


class DiffucoAeroResult(NamedTuple):
    """Local ``diffuco_aero`` drag fields and audited intermediates."""

    q_cdrag: jnp.ndarray
    q_cdrag_pft: jnp.ndarray
    raero: jnp.ndarray
    speed: jnp.ndarray
    zri: jnp.ndarray
    zri_pft: jnp.ndarray
    cd_neut: jnp.ndarray
    cd_neut_pft: jnp.ndarray
    cd_tmp: jnp.ndarray
    cd_tmp_pft: jnp.ndarray
    snowfact: jnp.ndarray


class DiffucoCombResult(NamedTuple):
    """Full explicit ``diffuco_comb`` beta/humidity state."""

    valpha: jnp.ndarray
    vbeta: jnp.ndarray
    vbeta_pft: jnp.ndarray
    vbeta1: jnp.ndarray
    vbeta2: jnp.ndarray
    vbeta3: jnp.ndarray
    vbeta4: jnp.ndarray
    vbeta4_pft: jnp.ndarray
    humrel: jnp.ndarray
    toveg: jnp.ndarray
    tosnow: jnp.ndarray


class DiffucoTransCO2ActivityResult(NamedTuple):
    """Per-PFT activity diagnostics before the FvCB solve."""

    lai_active: jnp.ndarray
    assimilate: jnp.ndarray
    zqsvegrap: jnp.ndarray
    water_lim: jnp.ndarray


class DiffucoTransCO2OutputResult(NamedTuple):
    """Post-FvCB ``diffuco_trans_co2`` output fields."""

    gsmean: jnp.ndarray
    cimean: jnp.ndarray
    gpp: jnp.ndarray
    rveget: jnp.ndarray
    rstruct: jnp.ndarray
    vbeta3: jnp.ndarray
    vbeta3pot: jnp.ndarray
    gstot_m_per_s: jnp.ndarray
    gstop_m_per_s: jnp.ndarray
    cresist: jnp.ndarray


class DiffucoLAILightTable(NamedTuple):
    """LAI discretization and Beer light extinction table."""

    laitab: jnp.ndarray
    light: jnp.ndarray


class DiffucoVPDBoundaryResult(NamedTuple):
    """VPD and boundary-layer conductance inputs for FvCB."""

    air_relhum: jnp.ndarray
    vpd: jnp.ndarray
    fvpd: jnp.ndarray
    gb_h2o: jnp.ndarray
    gb_co2: jnp.ndarray


class DiffucoPhotoTemperatureResult(NamedTuple):
    """Temperature response intermediates before the FvCB layer loop."""

    t_kmc: jnp.ndarray
    t_kmo: jnp.ndarray
    t_sco: jnp.ndarray
    t_gamma_star: jnp.ndarray
    t_rd: jnp.ndarray
    s_jmax_acclim_temp: jnp.ndarray
    t_jmax: jnp.ndarray
    s_vcmax_acclim_temp: jnp.ndarray
    t_vcmax: jnp.ndarray
    t_gm: jnp.ndarray
    vc: jnp.ndarray
    vj: jnp.ndarray
    gm: jnp.ndarray
    g0var: jnp.ndarray
    kmc: jnp.ndarray
    kmo: jnp.ndarray
    sco: jnp.ndarray
    gamma_star: jnp.ndarray
    low_gamma_star: jnp.ndarray


class DiffucoC3AssimilationLayerResult(NamedTuple):
    """C3 scalar/layer Yin-FvCB assimilation result."""

    assimi: jnp.ndarray
    cc: jnp.ndarray
    leaf_ci: jnp.ndarray
    ci_star: jnp.ndarray
    gs: jnp.ndarray
    info_limitphoto: jnp.ndarray
    x1: jnp.ndarray
    x2: jnp.ndarray
    p: jnp.ndarray
    q: jnp.ndarray
    r: jnp.ndarray
    qq: jnp.ndarray
    uu: jnp.ndarray
    psi: jnp.ndarray
    root_vc: jnp.ndarray
    root_j: jnp.ndarray
    valid_vc: jnp.ndarray
    valid_j: jnp.ndarray
    p_vc: jnp.ndarray
    q_vc: jnp.ndarray
    r_vc: jnp.ndarray
    qq_vc: jnp.ndarray
    uu_vc: jnp.ndarray
    psi_vc: jnp.ndarray
    p_j: jnp.ndarray
    q_j: jnp.ndarray
    r_j: jnp.ndarray
    qq_j: jnp.ndarray
    uu_j: jnp.ndarray
    psi_j: jnp.ndarray
    fallback: jnp.ndarray


class DiffucoC3CanopyResult(NamedTuple):
    """C3 PFT canopy-layer integration intermediates before output conversion."""

    vc2: jnp.ndarray
    vj2: jnp.ndarray
    rd: jnp.ndarray
    jj: jnp.ndarray
    iabs: jnp.ndarray
    jmax: jnp.ndarray
    assimi: jnp.ndarray
    cc: jnp.ndarray
    leaf_ci: jnp.ndarray
    gs: jnp.ndarray
    info_limitphoto: jnp.ndarray
    assimtot: jnp.ndarray
    rdtot: jnp.ndarray
    gstot: jnp.ndarray
    leaf_gs_top: jnp.ndarray
    ilai: jnp.ndarray
    laisum: jnp.ndarray
    cim: jnp.ndarray
    calculate: jnp.ndarray
    fallback: jnp.ndarray


class DiffucoTransCO2C3PFTResult(NamedTuple):
    """Source-backed one-PFT C3 ``diffuco_trans_co2`` closure."""

    activity: DiffucoTransCO2ActivityResult
    lai_light: DiffucoLAILightTable
    vpd_boundary: DiffucoVPDBoundaryResult
    photo_temperature: DiffucoPhotoTemperatureResult
    canopy: DiffucoC3CanopyResult
    controlled_assimtot: jnp.ndarray
    output: DiffucoTransCO2OutputResult


class DiffucoPFT14C3BetaClosureResult(NamedTuple):
    """PFT14 C3 DIFFUCO beta-chain closure through ``diffuco_comb``."""

    trans_co2: DiffucoTransCO2C3PFTResult
    vbeta3: jnp.ndarray
    vbeta3pot: jnp.ndarray
    bare: DiffucoBareCWRRBetaResult
    comb: DiffucoCombResult


class DiffucoPFT14C3ChainWithInterResult(NamedTuple):
    """PFT14 C3 DIFFUCO chain with local ``diffuco_inter`` outputs."""

    inter: DiffucoInterResult
    trans_co2: DiffucoTransCO2C3PFTResult
    closure: DiffucoPFT14C3BetaClosureResult


class DiffucoPFT14C3BetaProcessChainResult(NamedTuple):
    """PFT14 C3 DIFFUCO beta chain with snow, flood, and interception kernels."""

    snow: "DiffucoSnowBetaResult"
    flood: "DiffucoFloodBetaResult"
    inter: DiffucoInterResult
    trans_co2: DiffucoTransCO2C3PFTResult
    closure: DiffucoPFT14C3BetaClosureResult


class DiffucoPFT14LocalProcessBoundaryResult(NamedTuple):
    """PFT14 DIFFUCO local process boundary from drag through beta closure."""

    drag: DiffucoDragBoundaryResult
    qsatt: jnp.ndarray
    process_chain: DiffucoPFT14C3BetaProcessChainResult


class DiffucoPFT14LocalEnerbilPrecallResult(NamedTuple):
    """Local DIFFUCO process boundary plus full-array ENERBIL payload."""

    boundary: DiffucoPFT14LocalProcessBoundaryResult
    payload: "DiffucoPFT14AfterMainPayload"


class DiffucoPFT14FirstStepLocalEnerbilPrecallAssembly(NamedTuple):
    """First-step PFT14 DIFFUCO execution result assembled from pre-call state."""

    kwargs: Mapping[str, object]
    result: DiffucoPFT14LocalEnerbilPrecallResult
    passthrough_fields: tuple[str, ...]
    parameter_inputs: tuple[str, ...]
    provenance: tuple[str, ...]


class DiffucoPFT14AfterMainPayload(NamedTuple):
    """Source-backed subset of the ``after_diffuco_main`` boundary payload."""

    payload: dict[str, object]
    local_fields: tuple[str, ...]
    passthrough_fields: tuple[str, ...]
    provenance: tuple[str, ...]


DIFFUCO_FIRST_STEP_PRECALL_RESTART_FIELDS = (
    "temp_sol",
    "temp_sol_pft",
    "qsurf",
    "snow",
    "snowdz",
    "snowrho",
    "snowtemp",
    "snow_age",
    "snow_nobio",
    "snow_nobio_age",
    "frac_nobio",
    "veget",
    "veget_max",
    "lai",
    "qsintveg",
    "rstruct",
    "roughheight",
    "roughheight_pft",
    "z0h",
    "z0m",
    "evapot",
    "evapot_corr",
    "humrel",
    "evap_bare_lim",
    "drysoil_frac",
    "soilalbedo_bg",
    "height",
)

DIFFUCO_FIRST_STEP_PRECALL_DRIVER_FIELDS = (
    "u",
    "v",
    "zlev",
    "temp_air",
    "qair",
    "pb",
    "swdown",
    "ccanopy",
    "precip_rain",
    "lalo",
    "neighbours",
    "resolution",
)

DIFFUCO_FIRST_STEP_PRECALL_DERIVED_FIELDS = (
    "rau",
    "swnet",
    "rprof",
    "totfrac_nobio",
    "tot_bare_soil",
    "qsintmax",
    "assim_param",
    "temp_growth",
)

DIFFUCO_FIRST_STEP_PRECALL_OPTIONAL_EXACT_FIELDS = (
    "salinity",
    "tide_height",
    "biomass",
    "control_salinity",
    "control_inudate",
)

DIFFUCO_FIRST_STEP_LOCAL_PROCESS_REQUIRED_FIELDS = (
    "u",
    "v",
    "zlev",
    "z0h",
    "z0m",
    "roughheight",
    "roughheight_pft",
    "temp_sol",
    "temp_sol_pft",
    "temp_air",
    "qsurf",
    "qair",
    "pb",
    "rau",
    "humrel",
    "veget",
    "veget_max",
    "lai",
    "qsintveg",
    "qsintmax",
    "rstruct",
    "snow",
    "frac_nobio",
    "totfrac_nobio",
    "snow_nobio",
    "frac_snow_veg",
    "frac_snow_nobio",
    "evapot",
    "evapot_corr",
    "flood_frac",
    "flood_res",
    "tot_bare_soil",
    "evap_bare_lim",
    "assim_param",
    "temp_growth",
    "control_salinity",
    "control_inudate",
)


@dataclass(frozen=True)
class DiffucoFirstStepRestartState:
    """Restart-backed fields passed into first-step ``diffuco_main``."""

    path: Path
    temp_sol: np.ndarray
    temp_sol_pft: np.ndarray
    qsurf: np.ndarray
    snow: np.ndarray
    snowdz: np.ndarray
    snowrho: np.ndarray
    snowtemp: np.ndarray
    snow_age: np.ndarray
    snow_nobio: np.ndarray
    snow_nobio_age: np.ndarray
    frac_nobio: np.ndarray
    veget: np.ndarray
    veget_max: np.ndarray
    lai: np.ndarray
    qsintveg: np.ndarray
    rstruct: np.ndarray
    roughheight: np.ndarray
    roughheight_pft: np.ndarray
    z0h: np.ndarray
    z0m: np.ndarray
    evapot: np.ndarray
    evapot_corr: np.ndarray
    humrel: np.ndarray
    evap_bare_lim: np.ndarray
    drysoil_frac: np.ndarray
    soilalbedo_bg: np.ndarray
    height: np.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 984-1005",
        "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_main lines 306-387",
        "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_initialize lines 241-286",
        "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_initialize lines 246-275",
        "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90 restart reads lines 2705-2724, 2773-2776, and 3064-3090",
        "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90::explicitsnow_initialize lines 73-82",
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_init lines 1581-1601 and 1685-1707",
    )

    def as_payload(self) -> dict[str, object]:
        """Return the fields by Fortran DIFFUCO argument names."""

        return {
            name: getattr(self, name)
            for name in DIFFUCO_FIRST_STEP_PRECALL_RESTART_FIELDS
        }


class DiffucoFirstStepPrecallAssembly(NamedTuple):
    """Exact first-step DIFFUCO pre-call payload plus still-missing fields."""

    payload: Mapping[str, object]
    restart_inputs: tuple[str, ...]
    driver_inputs: tuple[str, ...]
    source_kernel_inputs: tuple[str, ...]
    control_inputs: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    missing_local_process_inputs: tuple[str, ...]
    provenance: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True only when every local DIFFUCO process input is exact."""

        return not self.missing_local_process_inputs


@dataclass(frozen=True)
class ControlInundateInputs:
    """Explicit state required by the source-backed PFT14 inundation control.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_main``, lines 470-622. ``z_soil`` is the already-built
    ``(/0, diaglev(1:nslm)/)`` depth vector from lines 470-474; ``rprof``,
    ``tide_height``, and ``biomass`` are upstream state passed to
    ``diffuco_main``. This contract does not use ``after_diffuco_main`` trace
    outputs to infer the internal mangrove control.
    """

    z_soil: object
    rprof: object
    tide_height: object
    biomass: object
    pft_index: int = 13


class DiffucoControlInundationAssembly(NamedTuple):
    """Exact first-step input assembly for PFT14 inundation control."""

    inputs: ControlInundateInputs | None
    result: "MangroveControlInundationResult | None"
    payload: Mapping[str, object]
    missing_inputs: tuple[str, ...]
    source_inputs: tuple[str, ...]
    provenance: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True only when ``control_inudate`` was computed from exact inputs."""

        return self.inputs is not None and self.result is not None and not self.missing_inputs


class CWRRVerticalSoilGrid(NamedTuple):
    """CWRR vertical-soil grid arrays built by ``vertical_soil_init``."""

    znh: jnp.ndarray
    dnh: jnp.ndarray
    dlh: jnp.ndarray
    diaglev: jnp.ndarray
    znt: jnp.ndarray
    zlt: jnp.ndarray
    dlt: jnp.ndarray
    dz1: jnp.ndarray
    dz5: jnp.ndarray
    lambda_thermal: float


class DiffucoControlSalinityAssembly(NamedTuple):
    """Exact first-step input assembly for PFT14 salinity control."""

    payload: Mapping[str, object]
    missing_inputs: tuple[str, ...]
    source_inputs: tuple[str, ...]
    provenance: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True only when ``control_salinity`` was computed from exact salinity."""

        return not self.missing_inputs and "control_salinity" in self.payload


class AGRRootState(NamedTuple):
    """Above-ground root biomass and heights for stilt and pneumatophore pools."""

    agb_agr_st: jnp.ndarray
    agb_agr_pn: jnp.ndarray
    h_agr_st: jnp.ndarray
    h_agr_pn: jnp.ndarray


class RootVentilationResult(NamedTuple):
    """Ventilated root fractions by AGR type and combined Fortran fraction."""

    frac_root_ven_st: jnp.ndarray
    frac_root_ven_pn: jnp.ndarray
    frac_root_ventilate: jnp.ndarray


class MangroveControlInundationResult(NamedTuple):
    """PFT14 inundation-control result with source-line intermediates."""

    control_inudate: jnp.ndarray
    frac_root_soil: jnp.ndarray
    n_soil_inudate: jnp.ndarray
    frac_root_inudate: jnp.ndarray
    agr: AGRRootState
    ventilation: RootVentilationResult
    frac_root_anoxia: jnp.ndarray


def z_soil_from_diaglev(diaglev):
    """Build the DIFFUCO soil-boundary vector from supplied ``diaglev``.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_main``, lines 470-474 allocate ``z_soil(0:nslm)``, set
    ``z_soil(0)=zero``, and copy ``diaglev(1:nslm)``. ``diaglev`` is allocated
    and populated in ``src_parameters/control.f90``, subroutine
    ``control_initialize``, lines 584-603. This helper does not derive
    ``diaglev`` values.
    """

    diaglev = jnp.asarray(diaglev, dtype=jnp.float64)
    if diaglev.ndim != 1:
        raise ValueError("diaglev must be a 1D vector with shape (nslm,)")
    if diaglev.shape[0] == 0:
        raise ValueError("diaglev must contain at least one soil layer")
    return jnp.concatenate((jnp.asarray([0.0], dtype=jnp.float64), diaglev))


def cwrr_diaglev_from_vertical_soil_params(
    *,
    depth_max_h,
    depth_max_t,
    depth_topthickness,
    depth_cstthickness,
    depth_geom,
    ratio_geom_below,
    nblayermax=500,
):
    """Build CWRR ``diaglev`` from explicit vertical-soil parameters.

    Fortran provenance: ``src_parameters/vertical_soil.f90``, subroutine
    ``vertical_soil_init``, lines 185-263 read ``DEPTH_MAX_H``,
    ``DEPTH_MAX_T``, ``DEPTH_TOPTHICK``, ``DEPTH_CSTTHICK``, ``DEPTH_GEOM``,
    and ``RATIO_GEOM_BELOW``; lines 268-281 build hydrology node depths;
    lines 331-363 select ``nslm`` and force the last hydrology node to
    ``zmaxh``; lines 374-444 build ``znt``. ``src_parameters/control.f90``,
    subroutine ``control_initialize``, lines 593-595 set
    ``diaglev=znt(1:nslm)`` for the CWRR branch.

    The caller must pass the active run parameters, for example from a local
    ``used_run.def`` audit. This helper does not read files and does not
    substitute missing runtime values.
    """

    geometry = vertical_soil_init(
        {
            "DEPTH_MAX_H": depth_max_h,
            "DEPTH_MAX_T": depth_max_t,
            "DEPTH_TOPTHICK": depth_topthickness,
            "DEPTH_CSTTHICK": depth_cstthickness,
            "DEPTH_GEOM": depth_geom,
            "RATIO_GEOM_BELOW": ratio_geom_below,
        },
        nblayermax=nblayermax,
    )
    return jnp.asarray(geometry.znt[: geometry.nslm], dtype=jnp.float64)


def cwrr_vertical_soil_grid_from_params(
    *,
    depth_max_h,
    depth_max_t,
    depth_topthickness,
    depth_cstthickness,
    depth_geom,
    ratio_geom_below,
    nblayermax=500,
) -> CWRRVerticalSoilGrid:
    """Build HYDROL and THERMOSOIL CWRR vertical grids from run parameters.

    Fortran provenance: ``src_parameters/vertical_soil.f90``,
    ``vertical_soil_init`` lines 185-263 read the active parameters; lines
    268-363 build HYDROL ``znh``, ``dnh``, and ``dlh``; lines 377-447 build
    THERMOSOIL ``znt``/``zlt``/``dlt``. ``control.f90`` lines 584-603 copy
    ``znt(1:nslm)`` into ``diaglev`` for CWRR.
    """

    geometry = vertical_soil_init(
        {
            "DEPTH_MAX_H": depth_max_h,
            "DEPTH_MAX_T": depth_max_t,
            "DEPTH_TOPTHICK": depth_topthickness,
            "DEPTH_CSTTHICK": depth_cstthickness,
            "DEPTH_GEOM": depth_geom,
            "RATIO_GEOM_BELOW": ratio_geom_below,
        },
        nblayermax=nblayermax,
    )
    znt = geometry.znt
    zlt = geometry.zlt
    dlt = geometry.dlt
    ngrnd = geometry.ngrnd
    dz1 = [1.0 / (znt[i + 1] - znt[i]) for i in range(ngrnd - 1)]
    dz5 = [(zlt[i] - znt[i]) * dz1[i] for i in range(ngrnd - 1)]
    lambda_thermal = znt[0] * dz1[0]

    return CWRRVerticalSoilGrid(
        znh=jnp.asarray(geometry.znh, dtype=jnp.float64),
        dnh=jnp.asarray(geometry.dnh, dtype=jnp.float64),
        dlh=jnp.asarray(geometry.dlh, dtype=jnp.float64),
        diaglev=jnp.asarray(znt[: geometry.nslm], dtype=jnp.float64),
        znt=jnp.asarray(znt, dtype=jnp.float64),
        zlt=jnp.asarray(zlt, dtype=jnp.float64),
        dlt=jnp.asarray(dlt, dtype=jnp.float64),
        dz1=jnp.asarray(dz1, dtype=jnp.float64),
        dz5=jnp.asarray(dz5, dtype=jnp.float64),
        lambda_thermal=float(lambda_thermal),
    )


def z_soil_from_cwrr_vertical_soil_params(**kwargs):
    """Build DIFFUCO ``z_soil`` from explicit CWRR vertical-soil parameters.

    Fortran provenance combines ``cwrr_diaglev_from_vertical_soil_params`` and
    ``src_sechiba/diffuco.f90``, subroutine ``diffuco_main``, lines 470-474.
    """

    return z_soil_from_diaglev(cwrr_diaglev_from_vertical_soil_params(**kwargs))


def _infer_control_inundation_npts(*, tide_height=None, biomass=None, npts: int | None = None) -> int:
    inferred: list[int] = []
    if npts is not None:
        inferred.append(int(npts))
    if tide_height is not None:
        tide_arr = np.asarray(tide_height)
        if tide_arr.ndim < 1:
            raise ValueError("tide_height must carry a land-point dimension")
        inferred.append(int(tide_arr.shape[0]))
    if biomass is not None:
        biomass_arr = np.asarray(biomass)
        if biomass_arr.ndim < 1:
            raise ValueError("biomass must carry a land-point dimension")
        inferred.append(int(biomass_arr.shape[0]))
    if not inferred:
        return 1
    first = inferred[0]
    if first < 1:
        raise ValueError("npts must be positive")
    if any(value != first for value in inferred):
        raise ValueError("npts, tide_height, and biomass must share the same land-point dimension")
    return first


def assemble_pft14_control_inundation_first_step(
    *,
    depth_max_h,
    depth_max_t,
    depth_topthickness,
    depth_cstthickness,
    depth_geom,
    ratio_geom_below,
    pft_to_mtc,
    hydrol_humcste,
    tide_height=None,
    biomass=None,
    control_inudate_min=None,
    agb_agr_ven_all_st=None,
    agb_agr_ven_all_pn=None,
    h_agr_max_st=None,
    h_agr_max_pn=None,
    npts: int | None = None,
    pft_index: int = 13,
) -> DiffucoControlInundationAssembly:
    """Assemble and optionally run first-step PFT14 inundation control.

    Fortran provenance: ``sechiba_main`` lines 984-1005 builds ``rprof`` from
    ``humcste_use`` and calls ``diffuco_main``; ``diffuco_main`` lines 470-622
    builds ``z_soil`` and computes the PFT14 mangrove inundation control from
    ``z_soil``, ``rprof``, full ``tide_height``, and ``biomass``. Vertical soil
    parameters follow ``vertical_soil_init`` lines 185-263 and 331-444.

    The function returns partial exact inputs when tide or biomass is absent.
    It computes ``control_inudate`` only when every required field and scalar
    parameter is explicitly supplied.
    """

    z_soil = z_soil_from_cwrr_vertical_soil_params(
        depth_max_h=depth_max_h,
        depth_max_t=depth_max_t,
        depth_topthickness=depth_topthickness,
        depth_cstthickness=depth_cstthickness,
        depth_geom=depth_geom,
        ratio_geom_below=ratio_geom_below,
    )
    npts_in = _infer_control_inundation_npts(tide_height=tide_height, biomass=biomass, npts=npts)
    pft_to_mtc = jnp.asarray(pft_to_mtc, dtype=jnp.int32)
    humcste = humcste_from_pft_to_mtc(
        pft_to_mtc,
        zmaxh=depth_max_h,
        hydrol_humcste=hydrol_humcste,
    )
    humcste_use = humcste_use_from_humcste(humcste, npts=npts_in)
    rprof = rprof_from_humcste_use(humcste_use)

    payload: dict[str, object] = {
        "z_soil": z_soil,
        "humcste": humcste,
        "humcste_use": humcste_use,
        "rprof": rprof,
    }
    source_inputs = ["z_soil", "humcste", "humcste_use", "rprof"]
    missing: list[str] = []
    if tide_height is None:
        missing.append("tide_height")
    else:
        payload["tide_height"] = tide_height
        source_inputs.append("tide_height")
    if biomass is None:
        missing.append("biomass")
    else:
        payload["biomass"] = biomass
        source_inputs.append("biomass")

    scalar_values = {
        "control_inudate_min": control_inudate_min,
        "agb_agr_ven_all_st": agb_agr_ven_all_st,
        "agb_agr_ven_all_pn": agb_agr_ven_all_pn,
        "h_agr_max_st": h_agr_max_st,
        "h_agr_max_pn": h_agr_max_pn,
    }
    for name, value in scalar_values.items():
        if value is None:
            missing.append(name)

    inputs = None
    result = None
    if not missing:
        inputs = ControlInundateInputs(
            z_soil=z_soil,
            rprof=rprof,
            tide_height=tide_height,
            biomass=biomass,
            pft_index=pft_index,
        )
        result = mangrove_control_inundation(
            inputs,
            control_inudate_min=control_inudate_min,
            agb_agr_ven_all_st=agb_agr_ven_all_st,
            agb_agr_ven_all_pn=agb_agr_ven_all_pn,
            h_agr_max_st=h_agr_max_st,
            h_agr_max_pn=h_agr_max_pn,
        )
        payload["control_inudate"] = result.control_inudate
        source_inputs.append("control_inudate")

    return DiffucoControlInundationAssembly(
        inputs=inputs,
        result=result,
        payload=payload,
        missing_inputs=tuple(dict.fromkeys(missing)),
        source_inputs=tuple(dict.fromkeys(source_inputs)),
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 984-1005",
            "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_main lines 470-622",
            "fortran_source/ORCHIDEE/src_parameters/vertical_soil.f90::vertical_soil_init lines 185-263, 331-444",
            "fortran_source/ORCHIDEE/src_parameters/pft_parameters.f90 lines 111-123, 265-273",
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_init lines 4093-4107",
        ),
        notes=(
            "Full tide_height(kjpindex,itimetide) is required; a scalar tide sample is not enough.",
            "First-step biomass must be the STOMATE_RESTART_FILEIN state or an explicit DIFFUCO call-boundary array.",
            "This assembly covers only the inundation-control subchain, not DIFFUCO drag, snow/flood/interception beta, or trans_co2.",
        ),
    )


def humcste_from_pft_to_mtc(pft_to_mtc, *, zmaxh, hydrol_humcste=None):
    """Return ``humcste`` for explicit PFT-to-MTC mapping and root depth.

    Fortran provenance: ``src_parameters/pft_parameters.f90``, subroutine
    ``pft_parameters_main``, lines 111-123 initialize/read ``pft_to_mtc`` and
    lines 265-273 choose ``humcste_ref2m`` or ``humcste_ref4m`` from
    ``src_parameters/constantes_mtc.f90`` lines 203-213. Optional
    ``hydrol_humcste`` mirrors ``config_stomate_pft_parameters`` lines
    3604-3610, where ``getin_p('HYDROL_HUMCSTE', humcste)`` may override the
    initialized vector.
    """

    pft_to_mtc = jnp.asarray(pft_to_mtc, dtype=jnp.int32)
    if pft_to_mtc.ndim != 1:
        raise ValueError("pft_to_mtc must be a one-dimensional vector")
    if pft_to_mtc.shape[0] == 0:
        raise ValueError("pft_to_mtc must contain at least one PFT")
    if not isinstance(pft_to_mtc, core.Tracer) and (
        bool(jnp.any(pft_to_mtc < 1))
        or bool(jnp.any(pft_to_mtc > len(HUMCSTE_REF2M)))
    ):
        raise ValueError("pft_to_mtc contains an MTC index outside the source table")

    if hydrol_humcste is not None:
        humcste = jnp.asarray(hydrol_humcste, dtype=jnp.float64)
        if humcste.shape != pft_to_mtc.shape:
            raise ValueError("hydrol_humcste override must match pft_to_mtc shape")
        return humcste

    table = HUMCSTE_REF4M if float(zmaxh) == 4.0 else HUMCSTE_REF2M
    return jnp.asarray(table, dtype=jnp.float64)[pft_to_mtc - 1]


def humcste_use_from_humcste(humcste, *, npts):
    """Broadcast ``humcste`` to the ``humcste_use(kjpindex,nvm)`` state.

    Fortran provenance: ``src_sechiba/hydrol.f90``, subroutine
    ``hydrol_init``, lines 4093-4107 loop over grid points and PFTs and assign
    ``humcste_use(ji,jv)=humcste(jv)``. The near-surface permafrost tree
    rewrite in lines 4107-4110 is commented out in the source.
    """

    humcste = jnp.asarray(humcste, dtype=jnp.float64)
    if humcste.ndim != 1:
        raise ValueError("humcste must be a one-dimensional vector")
    npts = int(npts)
    if npts <= 0:
        raise ValueError("npts must be positive")
    return jnp.broadcast_to(humcste, (npts, humcste.shape[0]))


def rprof_from_humcste_use(humcste_use):
    """Build ``rprof`` from supplied ``humcste_use`` state.

    Fortran provenance: ``src_sechiba/sechiba.f90``, subroutine
    ``sechiba_main``, lines 984-990 set ``rprof(:,jv)=1./humcste_use(:,jv)``
    before calling ``diffuco_main`` at lines 997-1005. ``humcste_use`` is
    assigned in ``src_sechiba/hydrol.f90``, subroutine ``hydrol_init``, lines
    4093-4107 from ``humcste``. This helper requires explicit
    ``humcste_use`` input and does not infer missing PFT parameters.
    """

    humcste_use = jnp.asarray(humcste_use, dtype=jnp.float64)
    if humcste_use.ndim != 2:
        raise ValueError("humcste_use must have shape (npts, nvm)")
    return 1.0 / humcste_use


def diffuco_control_inundation_input_coverage(
    *,
    diaglev_available=False,
    diaglev_source_available=False,
    humcste_use_available=False,
    humcste_source_available=False,
    pft14_humcste_parameter_available=False,
    full_tide_height_available=False,
    after_diffuco_tide_height_1_available=False,
    diffuco_call_biomass_available=False,
    first_step_cold_start_biomass_zeroed=False,
    first_step_stomate_start_biomass_available=False,
    restart_biomass_available=False,
    stomate_restart_output_biomass_available=False,
) -> DiffucoControlInundationInputCoverage:
    """Report audited coverage for ``mangrove_control_inundation`` inputs.

    Fortran provenance: ``src_sechiba/sechiba.f90`` lines 984-1005 build
    ``rprof`` and call ``diffuco_main`` with ``salinity``, ``rprof``,
    ``tide_height``, and ``biomass``. ``src_sechiba/diffuco.f90`` lines
    470-622 consume ``z_soil``, ``rprof``, ``tide_height``, and ``biomass`` to
    compute ``control_inudate``. Trace provenance for available tide values:
    ``src_sechiba/slowproc.f90`` lines 2531-2534 and 6029-6042 read the full
    tide field; the local fixed-format driver adapter preserves all tide time
    slots. The bridge ``after_diffuco_main`` trace exposes only
    ``tide_height_1`` and must not be treated as the full control input.

    Biomass coverage is deliberately narrower than restart readability.
    ``src_sechiba/sechiba.f90`` lines 602-617 call ``slowproc_initialize``
    before the first ``sechiba_main`` step; lines 984-1005 call
    ``diffuco_main`` before same-step ``slowproc_main``/``stomate_main`` at
    lines 1184-1216. Therefore first-step DIFFUCO biomass is the state left by
    initialization, not same-step STOMATE output. It is covered only when the
    caller has either the explicit DIFFUCO call-boundary array, a cold-start
    proof that ``stomate_io.readstart`` fell back to zero, or the exact
    ``STOMATE_RESTART_FILEIN`` start file used by ``stomate_initialize``.
    """

    z_status = "constructible" if (diaglev_available or diaglev_source_available) else "missing"
    if diaglev_available:
        z_detail = "diaglev supplied; z_soil is exactly [0, diaglev(1:nslm)]."
    elif diaglev_source_available:
        z_detail = (
            "CWRR vertical-soil parameters are available; diaglev is constructible as znt(1:nslm), "
            "then z_soil is [0, diaglev]."
        )
    else:
        z_detail = "No local full diaglev value/trace was identified; only the construction rule is covered."

    rprof_status = "constructible" if (humcste_use_available or humcste_source_available) else "missing"
    if humcste_use_available:
        rprof_detail = "humcste_use supplied; rprof is exactly 1/humcste_use."
    elif humcste_source_available:
        rprof_detail = (
            "humcste is constructible from PFT_TO_MTC/HYDROL_HUMCSTE source parameters; "
            "humcste_use broadcasts humcste over kjpindex and rprof is 1/humcste_use."
        )
    elif pft14_humcste_parameter_available:
        rprof_detail = (
            "A humcste parameter alone is not the DIFFUCO call-boundary rprof array; "
            "the audited input is humcste_use(kjpindex,nvm)."
        )
    else:
        rprof_detail = "No explicit humcste_use or source humcste/PFT_TO_MTC parameters were provided to this coverage call."

    tide_status = "covered" if full_tide_height_available else "missing"
    if full_tide_height_available:
        tide_detail = "Full tide_height(kjpindex,itimetide) is available from the driver/static tide trace chain."
    elif after_diffuco_tide_height_1_available:
        tide_detail = "Only after_diffuco_main tide_height_1 is available; control_inudate needs all itimetide slots."
    else:
        tide_detail = "No full tide_height(kjpindex,itimetide) source was identified."

    biomass_status = (
        "covered"
        if (
            diffuco_call_biomass_available
            or first_step_cold_start_biomass_zeroed
            or first_step_stomate_start_biomass_available
        )
        else "missing"
    )
    if diffuco_call_biomass_available:
        biomass_detail = "DIFFUCO call-boundary biomass(kjpindex,nvm,nparts,nelements) is available."
    elif first_step_cold_start_biomass_zeroed:
        biomass_detail = (
            "First-step DIFFUCO biomass is covered for a cold start: "
            "stomate_io.readstart reads missing biomass as val_exp and then sets the full array to zero "
            "before the first sechiba_main/diffuco_main call."
        )
    elif first_step_stomate_start_biomass_available:
        biomass_detail = (
            "First-step DIFFUCO biomass is covered by the exact STOMATE_RESTART_FILEIN start file read "
            "during slowproc_initialize/stomate_initialize before sechiba_main calls diffuco_main."
        )
    elif restart_biomass_available:
        biomass_detail = (
            "A STOMATE restart biomass array can be read locally, but this flag does not identify whether it is "
            "the STOMATE_RESTART_FILEIN start file used before this first diffuco_main call."
        )
    elif stomate_restart_output_biomass_available:
        biomass_detail = (
            "STOMATE restart output biomass is end-of-run saved state from stomate_finalize/restput_p, "
            "not the first-step DIFFUCO call-boundary state unless the run script later moves it to "
            "STOMATE_RESTART_FILEIN for the next run."
        )
    else:
        biomass_detail = "No DIFFUCO call-boundary biomass trace/source was identified."

    return DiffucoControlInundationInputCoverage(
        z_soil=DiffucoInputCoverage(
            z_status,
            (
                "src_sechiba/diffuco.f90::diffuco_main lines 470-474",
                "src_parameters/control.f90::control_initialize lines 584-603",
            ),
            z_detail,
        ),
        rprof=DiffucoInputCoverage(
            rprof_status,
            (
                "src_sechiba/sechiba.f90::sechiba_main lines 984-990",
                "src_sechiba/hydrol.f90::hydrol_init lines 4093-4107",
                "src_parameters/pft_parameters.f90::config_stomate_pft_parameters lines 265-273 and 3604-3610",
            ),
            rprof_detail,
        ),
        tide_height=DiffucoInputCoverage(
            tide_status,
            (
                "src_sechiba/slowproc.f90::slowproc_main lines 2531-2534",
                "src_sechiba/slowproc.f90::slowproc_read_data lines 6029-6042",
                "outputs/server_1961_trace_full_20260623/traces/orchjax_slowproc_read_data_trace.txt",
                "outputs/server_1961_bridge_trace_20260624/traces/orchjax_sechiba_bridge_diffuco_trace.txt:tide_height_1 only",
            ),
            tide_detail,
        ),
        biomass=DiffucoInputCoverage(
            biomass_status,
            (
                "src_sechiba/sechiba.f90::sechiba_init lines 2383-2388",
                "src_sechiba/sechiba.f90::sechiba_initialize lines 602-617",
                "src_sechiba/sechiba.f90::sechiba_main lines 984-1005 and 1184-1216",
                "src_sechiba/slowproc.f90::slowproc_initialize lines 257-293",
                "src_stomate/stomate.f90::stomate_initialize lines 1309-1368",
                "src_stomate/stomate_io.f90::readstart lines 919-922",
                "src_stomate/stomate_io.f90::writerestart lines 2631-2632",
                "src_sechiba/diffuco.f90::diffuco_main lines 386-387 and 532-535",
                "jax_orchidee/stomate/reference.py::read_restart_biomass_carbon restart-reader boundary",
            ),
            biomass_detail,
        ),
    )


def mangrove_control_salinity(salinity, *, control_salinity_min):
    """Return the mangrove salinity control scalar.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_main``, lines 453-462 compute
    ``-0.0228*salinity**2 + 1.37*salinity - 19.6`` and clip it to
    ``control_salinity_min``. The configuration key is read in
    ``src_parameters/constantes.f90`` lines 757-763.
    """

    salinity = jnp.asarray(salinity, dtype=jnp.float64)
    control = -0.0228 * salinity**2.0 + 1.37 * salinity - 19.6
    return jnp.maximum(control, jnp.asarray(control_salinity_min, dtype=jnp.float64))


def root_fraction_by_soil_layer(z_soil, rprof):
    """Return normalized root fractions for each soil layer.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_main``, lines 477-490 compute ``rpc = 1/(1-exp(-zmax/rprof))``
    and per-layer fractions
    ``rpc*(exp(-z_soil(jl-1)/rprof)-exp(-z_soil(jl)/rprof))``. The same
    formula is noted as modified from ``src_stomate/stomate_resp.f90`` lines
    183-210.
    """

    z_soil = jnp.asarray(z_soil, dtype=jnp.float64)
    rprof = jnp.asarray(rprof, dtype=jnp.float64)
    if z_soil.ndim != 1:
        raise ValueError("z_soil must be a 1D vector with shape (nslm + 1,)")

    rpc = 1.0 / (1.0 - jnp.exp(-z_soil[-1] / rprof))
    return rpc[..., None] * (
        jnp.exp(-z_soil[:-1] / rprof[..., None])
        - jnp.exp(-z_soil[1:] / rprof[..., None])
    )


def assemble_pft14_control_salinity_first_step(
    *,
    salinity=None,
    control_salinity_min=None,
) -> DiffucoControlSalinityAssembly:
    """Assemble and optionally run first-step PFT14 salinity control.

    Fortran provenance: ``sechiba_main`` lines 984-1005 passes ``salinity`` to
    ``diffuco_main``. ``diffuco_main`` lines 470-486 initializes the PFT14
    control and ``diffuco_trans_co2`` lines 2834-2844 applies
    ``control_salinity`` to assimilation before GPP conversion. The quadratic
    salinity response is implemented by ``mangrove_control_salinity``.
    """

    payload: dict[str, object] = {}
    source_inputs: list[str] = []
    missing: list[str] = []
    if salinity is None:
        missing.append("salinity")
    else:
        payload["salinity"] = salinity
        source_inputs.append("salinity")
    if control_salinity_min is None:
        missing.append("control_salinity_min")

    if not missing:
        payload["control_salinity"] = mangrove_control_salinity(
            salinity,
            control_salinity_min=control_salinity_min,
        )
        source_inputs.append("control_salinity")

    return DiffucoControlSalinityAssembly(
        payload=payload,
        missing_inputs=tuple(dict.fromkeys(missing)),
        source_inputs=tuple(dict.fromkeys(source_inputs)),
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 984-1005",
            "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_main lines 470-486",
            "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_trans_co2 lines 2834-2844",
        ),
        notes=(
            "This assembly covers only the salinity-control subchain.",
            "The salinity value must be the same land-point field passed into diffuco_main; no climatology or scalar fallback is used.",
        ),
    )


def soil_inundated_layer_count(tide_height, z_soil, nslm=None):
    """Count inundated soil boundaries from tide height and clip to ``nslm-1``.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_main``, lines 492-506 set ``tide_height_neg = -tide_height``,
    count ``tide_height_neg > z_soil``, then cap counts greater than or equal
    to ``nslm`` at ``nslm-1``.
    """

    tide_height = jnp.asarray(tide_height, dtype=jnp.float64)
    z_soil = jnp.asarray(z_soil, dtype=jnp.float64)
    if z_soil.ndim != 1:
        raise ValueError("z_soil must be a 1D vector with shape (nslm + 1,)")

    if nslm is None:
        nslm = int(z_soil.shape[0] - 1)
    n_soil_inudate = jnp.sum((-tide_height)[..., None] > z_soil, axis=-1)
    return jnp.minimum(n_soil_inudate, jnp.asarray(nslm - 1, dtype=n_soil_inudate.dtype))


def inundated_root_fraction(frac_root_soil, n_soil_inudate):
    """Sum root fractions in layers at or below the counted inundation depth.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_main``, lines 508-522 assign
    ``sum(frac_root_soil(...,(n_soil_inudate+1):nslm))`` for every grid, PFT,
    and tide step.
    """

    frac_root_soil = jnp.asarray(frac_root_soil, dtype=jnp.float64)
    n_soil_inudate = jnp.asarray(n_soil_inudate)
    nslm = frac_root_soil.shape[-1]
    layer_index = jnp.arange(nslm)
    mask = layer_index >= n_soil_inudate[..., None]
    return jnp.sum(frac_root_soil[:, None, :] * mask, axis=-1)


def aboveground_root_state(
    biomass,
    *,
    pft_index=13,
    h_agr_max_st,
    h_agr_max_pn,
):
    """Return AGR biomass pools and stilt/pneumatophore heights.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_main``, lines 525-539 sum ``iagrsapst+iagrhrtst`` and
    ``iagrsappn+iagrhrtpn`` biomass pools and apply the two height formulas.
    Biomass pool indices are defined in ``src_parameters/constantes_var.f90``
    lines 204-207 and ``icarbon`` at line 233. Config defaults for the height
    caps are in ``constantes_var.f90`` lines 802-804 and read by
    ``constantes.f90`` lines 791-805.
    """

    biomass = jnp.asarray(biomass, dtype=jnp.float64)
    agb_agr_st = biomass[:, pft_index, IAGRSAPST, ICARBON] + biomass[:, pft_index, IAGRHRTST, ICARBON]
    agb_agr_pn = biomass[:, pft_index, IAGRSAPPN, ICARBON] + biomass[:, pft_index, IAGRHRTPN, ICARBON]
    h_agr_st = jnp.minimum(
        (10.0 ** ((jnp.log10(agb_agr_st * 0.02 * 1000.0 / 1000.0) + 2.945) / 2.546) / 3.1415 / 100.0)
        * 7.56
        + 0.5,
        jnp.asarray(h_agr_max_st, dtype=jnp.float64),
    )
    h_agr_pn = jnp.minimum(
        agb_agr_pn * 0.02 * 0.085,
        jnp.asarray(h_agr_max_pn, dtype=jnp.float64),
    )
    return AGRRootState(
        agb_agr_st=agb_agr_st,
        agb_agr_pn=agb_agr_pn,
        h_agr_st=h_agr_st,
        h_agr_pn=h_agr_pn,
    )


def root_ventilation_fraction(
    tide_height,
    agr,
    *,
    agb_agr_ven_all_st,
    agb_agr_ven_all_pn,
):
    """Return stilt, pneumatophore, and combined ventilated root fractions.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_main``, lines 541-572 clip positive tide height at zero, compute
    each AGR-type ventilation as biomass saturation times exposed height
    fraction, and combine the two fractions as
    ``MIN(0.5*frac_root_ven_st + 0.5*frac_root_ven_pn, 1.0)``. The biomass
    thresholds are configured by ``constantes.f90`` lines 773-789.
    """

    tide_height = jnp.asarray(tide_height, dtype=jnp.float64)
    tide_height_pos = jnp.maximum(tide_height, 0.0)

    h_agr_st = agr.h_agr_st[:, None]
    h_agr_pn = agr.h_agr_pn[:, None]
    frac_root_ven_st = jnp.where(
        h_agr_st <= 0.0,
        0.0,
        jnp.minimum(agr.agb_agr_st[:, None] / jnp.asarray(agb_agr_ven_all_st, dtype=jnp.float64), 1.0)
        * jnp.maximum(h_agr_st - tide_height_pos, 0.0)
        / h_agr_st,
    )
    frac_root_ven_pn = jnp.where(
        h_agr_pn <= 0.0,
        0.0,
        jnp.minimum(agr.agb_agr_pn[:, None] / jnp.asarray(agb_agr_ven_all_pn, dtype=jnp.float64), 1.0)
        * jnp.maximum(h_agr_pn - tide_height_pos, 0.0)
        / h_agr_pn,
    )
    return RootVentilationResult(
        frac_root_ven_st=frac_root_ven_st,
        frac_root_ven_pn=frac_root_ven_pn,
        frac_root_ventilate=jnp.minimum(0.5 * frac_root_ven_st + 0.5 * frac_root_ven_pn, 1.0),
    )


def mangrove_control_inundation(
    inputs: ControlInundateInputs,
    *,
    control_inudate_min,
    agb_agr_ven_all_st,
    agb_agr_ven_all_pn,
    h_agr_max_st,
    h_agr_max_pn,
):
    """Compute the source-equivalent PFT14 inundation control from explicit state.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_main``, lines 470-622. This helper implements the closed PFT14
    path used by lines 578-584; callers must provide upstream ``diaglev`` as
    ``z_soil``, ``rprof``, ``tide_height``, and ``biomass`` state.
    """

    rprof = jnp.asarray(inputs.rprof, dtype=jnp.float64)
    z_soil = jnp.asarray(inputs.z_soil, dtype=jnp.float64)
    tide_height = jnp.asarray(inputs.tide_height, dtype=jnp.float64)

    frac_root_soil = root_fraction_by_soil_layer(z_soil, rprof[:, inputs.pft_index])
    n_soil_inudate = soil_inundated_layer_count(tide_height, z_soil, nslm=frac_root_soil.shape[-1])
    frac_root_inudate = inundated_root_fraction(frac_root_soil, n_soil_inudate)
    agr = aboveground_root_state(
        inputs.biomass,
        pft_index=inputs.pft_index,
        h_agr_max_st=h_agr_max_st,
        h_agr_max_pn=h_agr_max_pn,
    )
    ventilation = root_ventilation_fraction(
        tide_height,
        agr,
        agb_agr_ven_all_st=agb_agr_ven_all_st,
        agb_agr_ven_all_pn=agb_agr_ven_all_pn,
    )
    frac_root_anoxia = jnp.maximum(frac_root_inudate - ventilation.frac_root_ventilate, 0.0)
    control_inudate = 1.0 - (
        (1.0 - jnp.asarray(control_inudate_min, dtype=jnp.float64))
        * jnp.mean(frac_root_anoxia, axis=-1)
    )

    return MangroveControlInundationResult(
        control_inudate=control_inudate,
        frac_root_soil=frac_root_soil,
        n_soil_inudate=n_soil_inudate,
        frac_root_inudate=frac_root_inudate,
        agr=agr,
        ventilation=ventilation,
        frac_root_anoxia=frac_root_anoxia,
    )


def diffuco_apply_pft14_assimtot_controls(
    assimtot,
    *,
    control_salinity,
    control_inudate,
    jv=14,
):
    """Apply the PFT14 mangrove controls to ``assimtot``.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_trans_co2``, lines 2834-2844 multiply ``assimtot`` by
    ``control_salinity`` and ``control_inudate`` only when ``jv == 14``. This
    mutation happens before `cimean` and `gpp` are computed at lines
    2884-2897, so callers should pass the returned value into the output
    conversion layer rather than treating the controls as GPP-only.
    """

    assimtot = jnp.asarray(assimtot, dtype=jnp.float64)
    if jv == 14:
        return (
            assimtot
            * jnp.asarray(control_salinity, dtype=jnp.float64)
            * jnp.asarray(control_inudate, dtype=jnp.float64)
        )
    return assimtot


def pft14_gpp_from_assimilation(
    assimtot,
    *,
    veget_max,
    dt_sechiba,
    control_salinity,
    control_inudate,
    jv=14,
):
    """Apply PFT14 controls to assimilation and convert it to GPP.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_trans_co2``, lines 2834-2844 mutate ``assimtot`` for PFT14; lines
    2896-2897 assign ``gpp = assimtot*12e-6*veget_max*dt_sechiba``.
    """

    veget_max = jnp.asarray(veget_max, dtype=jnp.float64)
    assimtot = diffuco_apply_pft14_assimtot_controls(
        assimtot,
        control_salinity=control_salinity,
        control_inudate=control_inudate,
        jv=jv,
    )

    return assimtot * 12.0e-6 * veget_max * jnp.asarray(dt_sechiba, dtype=jnp.float64)


def _require_first_step_diffuco_restart_shapes(fields: Mapping[str, np.ndarray]) -> None:
    """Validate the restart axes needed by first-step DIFFUCO."""

    npts = int(np.asarray(fields["temp_sol"]).shape[0])
    nvm = int(np.asarray(fields["veget"]).shape[1])
    nnobio = int(np.asarray(fields["frac_nobio"]).shape[1])
    for name in (
        "temp_sol",
        "qsurf",
        "snow",
        "roughheight",
        "z0h",
        "z0m",
        "evapot",
        "evapot_corr",
        "evap_bare_lim",
        "drysoil_frac",
        "snow_age",
    ):
        if np.asarray(fields[name]).shape != (npts,):
            raise ValueError(f"{name} must have shape (npts,)")
    for name in (
        "temp_sol_pft",
        "veget",
        "veget_max",
        "lai",
        "qsintveg",
        "rstruct",
        "roughheight_pft",
        "humrel",
        "height",
    ):
        if np.asarray(fields[name]).shape != (npts, nvm):
            raise ValueError(f"{name} must have shape (npts, nvm)")
    for name in ("snow_nobio", "snow_nobio_age", "frac_nobio"):
        if np.asarray(fields[name]).shape != (npts, nnobio):
            raise ValueError(f"{name} must have shape (npts, nnobio)")
    for name in ("snowdz", "snowrho", "snowtemp"):
        arr = np.asarray(fields[name])
        if arr.ndim != 2 or arr.shape[0] != npts:
            raise ValueError(f"{name} must have shape (npts, nsnow)")
    if np.asarray(fields["soilalbedo_bg"]).shape != (npts, 2):
        raise ValueError("soilalbedo_bg must have shape (npts, 2)")


def read_diffuco_first_step_restart_state(
    path: str | Path,
    *,
    source_pft_layout=None,
    target_pft_layout=None,
) -> DiffucoFirstStepRestartState:
    """Read exact SECHIBA restart fields used before first-step DIFFUCO.

    Fortran provenance follows the initialization reads that populate the
    arrays later passed by ``sechiba_main`` to ``diffuco_main`` at lines
    997-1005. This reader requires the variables to exist in the restart and
    does not apply cold-start ``setvar_p`` fallbacks.
    """

    fields = read_restart_fields(
        path,
        DIFFUCO_FIRST_STEP_PRECALL_RESTART_FIELDS,
        source_pft_layout=source_pft_layout,
        target_pft_layout=target_pft_layout,
    )
    _require_first_step_diffuco_restart_shapes(fields)
    return DiffucoFirstStepRestartState(
        path=Path(path),
        temp_sol=fields["temp_sol"],
        temp_sol_pft=fields["temp_sol_pft"],
        qsurf=fields["qsurf"],
        snow=fields["snow"],
        snowdz=fields["snowdz"],
        snowrho=fields["snowrho"],
        snowtemp=fields["snowtemp"],
        snow_age=fields["snow_age"],
        snow_nobio=fields["snow_nobio"],
        snow_nobio_age=fields["snow_nobio_age"],
        frac_nobio=fields["frac_nobio"],
        veget=fields["veget"],
        veget_max=fields["veget_max"],
        lai=fields["lai"],
        qsintveg=fields["qsintveg"],
        rstruct=fields["rstruct"],
        roughheight=fields["roughheight"],
        roughheight_pft=fields["roughheight_pft"],
        z0h=fields["z0h"],
        z0m=fields["z0m"],
        evapot=fields["evapot"],
        evapot_corr=fields["evapot_corr"],
        humrel=fields["humrel"],
        evap_bare_lim=fields["evap_bare_lim"],
        drysoil_frac=fields["drysoil_frac"],
        soilalbedo_bg=fields["soilalbedo_bg"],
        height=fields["height"],
    )


def assemble_diffuco_first_step_precall_payload(
    *,
    driver_or_intersurf_payload: Mapping[str, object],
    restart_payload: Mapping[str, object],
    driver_albedo_payload: Mapping[str, object] | None = None,
    slowproc_derivvar_payload: Mapping[str, object] | None = None,
    diffuco_control_salinity: DiffucoControlSalinityAssembly | None = None,
    diffuco_control_inundation: DiffucoControlInundationAssembly | None = None,
    ok_explicitsnow: bool | None = None,
    river_routing: bool | None = None,
    nbp_glo: int | None = None,
) -> DiffucoFirstStepPrecallAssembly:
    """Assemble exact first-step ``diffuco_main`` pre-call fields.

    Fortran provenance: ``sechiba_main`` lines 984-1005 builds ``rprof`` and
    calls ``diffuco_main``; ``diffuco_main`` lines 306-387 declare the driver,
    restart, SLOWPROC, HYDROL, CONDVEG, and mangrove-control inputs consumed by
    local DIFFUCO kernels. The helper materializes only fields with explicit
    upstream state or source-backed local algebra and returns all remaining
    inputs as missing.
    """

    aliases = {
        "tair": "temp_air",
        "psurf": "pb",
        "height_lev1": "zlev",
        "Height_Lev1": "zlev",
        "Tair": "temp_air",
        "PSurf": "pb",
        "Qair": "qair",
        "Wind_E": "u",
        "Wind_N": "v",
        "Rainf": "precip_rain",
        "SWdown": "swdown",
    }
    alias_updates = {
        target: driver_or_intersurf_payload[source]
        for source, target in aliases.items()
        if target not in driver_or_intersurf_payload and source in driver_or_intersurf_payload
    }
    driver = driver_or_intersurf_payload if not alias_updates else {**driver_or_intersurf_payload, **alias_updates}
    restart = restart_payload
    albedo = {} if driver_albedo_payload is None else driver_albedo_payload
    derivvar = {} if slowproc_derivvar_payload is None else slowproc_derivvar_payload

    payload: dict[str, object] = {}
    restart_inputs: list[str] = []
    driver_inputs: list[str] = []
    source_kernel_inputs: list[str] = []
    control_inputs: list[str] = []

    def as_runtime_array(value):
        return jnp.asarray(value) if isinstance(value, core.Tracer) else np.asarray(value)

    def add(name: str, value, bucket: list[str]) -> None:
        payload[name] = as_runtime_array(value)
        bucket.append(name)

    for field in DIFFUCO_FIRST_STEP_PRECALL_RESTART_FIELDS:
        if field in restart:
            add(field, restart[field], restart_inputs)

    for field in DIFFUCO_FIRST_STEP_PRECALL_DRIVER_FIELDS:
        if field in driver and driver[field] is not None:
            add(field, driver[field], driver_inputs)

    if "rau" not in payload and {"pb", "temp_air"} <= payload.keys():
        add(
            "rau",
            as_runtime_array(sechiba_air_density_from_pb_temp_air(payload["pb"], payload["temp_air"])),
            source_kernel_inputs,
        )
    if "swnet" not in payload and "swdown" in payload and "albedo" in albedo:
        add(
            "swnet",
            as_runtime_array(driver_swnet_from_swdown_albedo(swdown=payload["swdown"], albedo=albedo["albedo"])),
            source_kernel_inputs,
        )
    elif "swnet" not in payload and "swdown" in payload and {"albedo_vis", "albedo_nir"} <= albedo.keys():
        albedo_vis = as_runtime_array(albedo["albedo_vis"])
        albedo_nir = as_runtime_array(albedo["albedo_nir"])
        stack = jnp.stack if any(isinstance(value, core.Tracer) for value in (albedo_vis, albedo_nir)) else np.stack
        two_band = stack((albedo_vis, albedo_nir), axis=1)
        add(
            "swnet",
            as_runtime_array(driver_swnet_from_swdown_albedo(swdown=payload["swdown"], albedo=two_band)),
            source_kernel_inputs,
        )
    if "frac_nobio" in payload and "totfrac_nobio" not in payload:
        frac_nobio = as_runtime_array(payload["frac_nobio"])
        add("totfrac_nobio", frac_nobio.sum(axis=1), source_kernel_inputs)
    if {"veget", "veget_max"} <= payload.keys() and "tot_bare_soil" not in payload:
        veget = as_runtime_array(payload["veget"])
        veget_max = as_runtime_array(payload["veget_max"])
        add(
            "tot_bare_soil",
            veget_max[:, 0] + (veget_max[:, 1:] - veget[:, 1:]).sum(axis=1),
            source_kernel_inputs,
        )
    if (
        ok_explicitsnow is not None
        and {"snow", "snow_nobio", "snowrho", "snowdz"} <= payload.keys()
        and ("frac_snow_veg" not in payload or "frac_snow_nobio" not in payload)
    ):
        snow_fraction = condveg_frac_snow(
            snow=payload["snow"],
            snow_nobio=payload["snow_nobio"],
            snowrho=payload["snowrho"],
            snowdz=payload["snowdz"],
            ok_explicitsnow=ok_explicitsnow,
        )
        add("frac_snow_veg", as_runtime_array(snow_fraction.frac_snow_veg), source_kernel_inputs)
        add("frac_snow_nobio", as_runtime_array(snow_fraction.frac_snow_nobio), source_kernel_inputs)
    if river_routing is not None and nbp_glo is not None:
        routing_active = bool(river_routing) and int(nbp_glo) > 1
        if not routing_active:
            if "temp_sol" in payload:
                temp_sol = as_runtime_array(payload["temp_sol"])
                zeros = jnp.zeros_like(temp_sol) if isinstance(temp_sol, core.Tracer) else np.zeros_like(temp_sol, dtype=np.float64)
            else:
                zeros = np.zeros((int(nbp_glo),), dtype=np.float64)
            add("flood_frac", zeros, source_kernel_inputs)
            add("flood_res", zeros, source_kernel_inputs)

    for field in ("qsintmax", "assim_param", "height", "temp_growth"):
        if field in derivvar:
            add(field, derivvar[field], source_kernel_inputs)

    for assembly in (diffuco_control_salinity, diffuco_control_inundation):
        if assembly is None:
            continue
        for field in DIFFUCO_FIRST_STEP_PRECALL_OPTIONAL_EXACT_FIELDS:
            if field in assembly.payload:
                add(field, assembly.payload[field], control_inputs)

    missing_inputs = tuple(
        field
        for field in (
            *DIFFUCO_FIRST_STEP_PRECALL_RESTART_FIELDS,
            *DIFFUCO_FIRST_STEP_PRECALL_DRIVER_FIELDS,
            *DIFFUCO_FIRST_STEP_PRECALL_DERIVED_FIELDS,
            *DIFFUCO_FIRST_STEP_PRECALL_OPTIONAL_EXACT_FIELDS,
        )
        if field not in payload
    )
    missing_local = tuple(field for field in DIFFUCO_FIRST_STEP_LOCAL_PROCESS_REQUIRED_FIELDS if field not in payload)
    return DiffucoFirstStepPrecallAssembly(
        payload=payload,
        restart_inputs=tuple(dict.fromkeys(restart_inputs)),
        driver_inputs=tuple(dict.fromkeys(driver_inputs)),
        source_kernel_inputs=tuple(dict.fromkeys(source_kernel_inputs)),
        control_inputs=tuple(dict.fromkeys(control_inputs)),
        missing_inputs=tuple(dict.fromkeys(missing_inputs)),
        missing_local_process_inputs=tuple(dict.fromkeys(missing_local)),
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 984-1005",
            "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_main lines 306-710",
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_var_init lines 3069-3092",
            "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 1139-1164",
            "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_derivvar lines 2625-2673",
            "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 1116-1121",
            "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_frac_snow lines 858-898",
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_initialize lines 748-768 and sechiba_main lines 1226-1248",
        ),
        notes=(
            "This is a pre-call assembly, not a DIFFUCO execution result.",
            "Missing frac_snow_veg/frac_snow_nobio must come from exact CONDVEG snow-fraction logic.",
            "flood_frac/flood_res are zero only for the audited no-routing branch; active routing requires routing state.",
            "The local DIFFUCO process is not marked ok until all missing_local_process_inputs are present.",
        ),
    )


def alpha_ll_for_pft_from_run_def(run_def_values: Mapping[str, str], *, pft_fortran_index: int) -> float:
    """Return ``alpha_LL`` for one PFT through the Fortran MTC mapping.

    Fortran provenance: ``src_parameters/pft_parameters.f90`` lines 117-123
    read ``PFT_TO_MTC`` and lines 333 and 484 set
    ``alpha_LL(:)=alpha_LL_mtc(pft_to_mtc(:))``. The MTC constants are in
    ``src_parameters/constantes_mtc.f90`` lines 440-442.
    """

    mtc_index = int(parse_run_def_indexed_float(dict(run_def_values), "PFT_TO_MTC", pft_fortran_index))
    if mtc_index <= 0 or mtc_index > len(ALPHA_LL_MTC):
        raise ValueError(f"PFT_TO_MTC for PFT {pft_fortran_index} selects invalid MTC {mtc_index}")
    return float(ALPHA_LL_MTC[mtc_index - 1])


def pft14_trans_co2_parameter_inputs_from_run_def(
    run_def_values: Mapping[str, str],
    *,
    pft_fortran_index: int = 14,
) -> dict[str, object]:
    """Build the source-backed DIFFUCO C3 parameter set for PFT14.

    The indexed values come from ``used_run.def`` entries consumed by
    ``getin_p`` in ``src_parameters/pft_parameters.f90``. ``alpha_ll`` is not a
    run.def key in this source branch; it is derived from the Fortran
    ``PFT_TO_MTC`` to ``alpha_LL_mtc`` mapping.
    """

    values = dict(run_def_values)
    return {
        "tphoto_min": parse_run_def_indexed_float(values, "TPHOTO_MIN", pft_fortran_index),
        "tphoto_max": parse_run_def_indexed_float(values, "TPHOTO_MAX", pft_fortran_index),
        "ok_laidev": parse_run_def_indexed_bool(values, "OK_LAIDEV", pft_fortran_index),
        "ext_coeff": parse_run_def_indexed_float(values, "EXT_COEFF", pft_fortran_index),
        "a1": parse_run_def_indexed_float(values, "A1", pft_fortran_index),
        "b1": parse_run_def_indexed_float(values, "B1", pft_fortran_index),
        "stress_gs": parse_run_def_indexed_float(values, "STRESS_GS", pft_fortran_index),
        "e_kmc": parse_run_def_indexed_float(values, "E_KMC", pft_fortran_index),
        "e_kmo": parse_run_def_indexed_float(values, "E_KMO", pft_fortran_index),
        "e_sco": parse_run_def_indexed_float(values, "E_SCO", pft_fortran_index),
        "e_gamma_star": parse_run_def_indexed_float(values, "E_GAMMA_STAR", pft_fortran_index),
        "e_rd": parse_run_def_indexed_float(values, "E_RD", pft_fortran_index),
        "e_jmax": parse_run_def_indexed_float(values, "E_JMAX", pft_fortran_index),
        "d_jmax": parse_run_def_indexed_float(values, "D_JMAX", pft_fortran_index),
        "asj": parse_run_def_indexed_float(values, "ASJ", pft_fortran_index),
        "bsj": parse_run_def_indexed_float(values, "BSJ", pft_fortran_index),
        "e_vcmax": parse_run_def_indexed_float(values, "E_VCMAX", pft_fortran_index),
        "d_vcmax": parse_run_def_indexed_float(values, "D_VCMAX", pft_fortran_index),
        "asv": parse_run_def_indexed_float(values, "ASV", pft_fortran_index),
        "bsv": parse_run_def_indexed_float(values, "BSV", pft_fortran_index),
        "e_gm": parse_run_def_indexed_float(values, "E_GM", pft_fortran_index),
        "d_gm": parse_run_def_indexed_float(values, "D_GM", pft_fortran_index),
        "s_gm": parse_run_def_indexed_float(values, "S_GM", pft_fortran_index),
        "arjv": parse_run_def_indexed_float(values, "ARJV", pft_fortran_index),
        "brjv": parse_run_def_indexed_float(values, "BRJV", pft_fortran_index),
        "gm25": parse_run_def_indexed_float(values, "GM25", pft_fortran_index),
        "stress_gm": parse_run_def_indexed_float(values, "STRESS_GM", pft_fortran_index),
        "g0": parse_run_def_indexed_float(values, "G0", pft_fortran_index),
        "kmc25": parse_run_def_indexed_float(values, "KMC25", pft_fortran_index),
        "kmo25": parse_run_def_indexed_float(values, "KMO25", pft_fortran_index),
        "sco25": parse_run_def_indexed_float(values, "SCO25", pft_fortran_index),
        "gamma_star25": parse_run_def_indexed_float(values, "gamma_star25", pft_fortran_index),
        "alpha_ll": alpha_ll_for_pft_from_run_def(values, pft_fortran_index=pft_fortran_index),
        "theta": parse_run_def_indexed_float(values, "THETA", pft_fortran_index),
        "stress_vcmax": parse_run_def_indexed_float(values, "STRESS_VCMAX", pft_fortran_index),
        "rveg_pft": parse_run_def_indexed_float(values, "RVEG_PFT", pft_fortran_index),
        "rstruct_const": parse_run_def_indexed_float(values, "RSTRUCT_CONST", pft_fortran_index),
        "lai_light": _cached_diffuco_lai_light_table(
            parse_run_def_indexed_float(values, "EXT_COEFF", pft_fortran_index),
            DEFAULT_NLAI,
            parse_run_def_float(values, "LAIMAX"),
            parse_run_def_float(values, "LAI_LEVEL_DEPTH"),
        ),
    }


def _require_precall_payload_arrays(
    payload: Mapping[str, object],
    fields: tuple[str, ...],
) -> dict[str, np.ndarray]:
    missing = tuple(field for field in fields if field not in payload)
    if missing:
        raise ValueError(f"missing DIFFUCO first-step pre-call fields: {missing}")
    return {
        field: jnp.asarray(payload[field]) if isinstance(payload[field], core.Tracer) else np.asarray(payload[field])
        for field in fields
    }


def assemble_pft14_local_enerbil_precall_kwargs_from_first_step_precall(
    precall: DiffucoFirstStepPrecallAssembly,
    *,
    run_def_values: Mapping[str, str],
    pft_index: int = 13,
    ldq_cdrag_from_gcm: bool = False,
    static_kwargs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build local PFT14 DIFFUCO kwargs from exact first-step pre-call state.

    Fortran provenance: this adapter follows ``diffuco_main`` lines 629-710
    and the ``ok_co2`` PFT14 ``diffuco_trans_co2`` path. It refuses incomplete
    pre-call assemblies and does not use after-DIFFUCO bridge values.
    """

    if not precall.ok:
        raise ValueError(f"DIFFUCO first-step pre-call is incomplete: {precall.missing_local_process_inputs}")

    values: dict[str, str] | None = None

    def _run_def_values() -> dict[str, str]:
        nonlocal values
        if values is None:
            values = dict(run_def_values)
        return values

    payload = precall.payload
    arrays = _require_precall_payload_arrays(
        payload,
        (
            "u",
            "v",
            "zlev",
            "z0h",
            "z0m",
            "roughheight",
            "roughheight_pft",
            "temp_sol",
            "temp_sol_pft",
            "temp_air",
            "qsurf",
            "qair",
            "pb",
            "rau",
            "humrel",
            "veget",
            "veget_max",
            "lai",
            "qsintveg",
            "qsintmax",
            "rstruct",
            "snow",
            "frac_nobio",
            "totfrac_nobio",
            "snow_nobio",
            "frac_snow_veg",
            "frac_snow_nobio",
            "evapot",
            "evapot_corr",
            "flood_frac",
            "flood_res",
            "tot_bare_soil",
            "evap_bare_lim",
            "assim_param",
            "temp_growth",
            "swdown",
            "ccanopy",
            "control_salinity",
            "control_inudate",
        ),
    )
    humrel = arrays["humrel"]
    if humrel.ndim != 2:
        raise ValueError("humrel must have shape (npts, nvm)")
    if int(pft_index) < 0 or int(pft_index) >= humrel.shape[1]:
        raise ValueError("pft_index must select a valid PFT column")

    npts, nvm = humrel.shape
    pft_fortran_index = int(pft_index) + 1
    assim_param = arrays["assim_param"]
    if assim_param.ndim != 3 or assim_param.shape[0] != npts or assim_param.shape[1] != nvm:
        raise ValueError("assim_param must have shape (npts, nvm, npco2)")
    vcmax = assim_param[:, int(pft_index), 0]
    static = {} if static_kwargs is None else static_kwargs
    if "rstruct_const" in static:
        rstruct_const_source = static["rstruct_const"]
    elif "trans_co2_pft_params" in static and "rstruct_const" in static["trans_co2_pft_params"]:
        rstruct_const_source = static["trans_co2_pft_params"]["rstruct_const"]
    else:
        values = _run_def_values()
        rstruct_const_source = [
            parse_run_def_indexed_float(values, "RSTRUCT_CONST", index)
            for index in range(1, nvm + 1)
        ]
    rstruct_const = (
        jnp.asarray(rstruct_const_source, dtype=jnp.float64)
        if isinstance(rstruct_const_source, core.Tracer)
        else np.asarray(rstruct_const_source, dtype=np.float64)
    )
    ca = arrays["ccanopy"]
    if "trans_co2_pft_params" in static:
        pft_params_source = static["trans_co2_pft_params"]
    else:
        pft_params_source = pft14_trans_co2_parameter_inputs_from_run_def(
            _run_def_values(),
            pft_fortran_index=pft_fortran_index,
        )
    pft_params = dict(pft_params_source)
    dt_sechiba = float(static["dt_sechiba"]) if "dt_sechiba" in static else parse_run_def_float(_run_def_values(), "DT_SECHIBA")
    laimax = float(static["laimax"]) if "laimax" in static else parse_run_def_float(_run_def_values(), "LAIMAX")
    lai_level_depth = (
        float(static["lai_level_depth"])
        if "lai_level_depth" in static
        else parse_run_def_float(_run_def_values(), "LAI_LEVEL_DEPTH")
    )
    min_wind = float(static["min_wind"]) if "min_wind" in static else parse_run_def_float(_run_def_values(), "MIN_WIND")
    snowcri = float(static["snowcri"]) if "snowcri" in static else parse_run_def_float(_run_def_values(), "SNOWCRI")
    ok_snowfact = bool(static["ok_snowfact"]) if "ok_snowfact" in static else parse_run_def_bool(_run_def_values()["OK_SNOWFACT"])
    rough_dyn = bool(static["rough_dyn"]) if "rough_dyn" in static else parse_run_def_bool(_run_def_values()["ROUGH_DYN"])
    if "ok_laidev" in static:
        ok_laidev_source = static["ok_laidev"]
    else:
        values = _run_def_values()
        ok_laidev_source = [parse_run_def_indexed_bool(values, "OK_LAIDEV", index) for index in range(1, nvm + 1)]
    ok_laidev = np.asarray(ok_laidev_source, dtype=bool)
    trans_co2_inputs = {
        **pft_params,
        "swdown": arrays["swdown"],
        "qsurf": arrays["qsurf"],
        "t2m": arrays["temp_air"],
        "pb": arrays["pb"],
        "wind": (
            jnp.sqrt(arrays["u"] * arrays["u"] + arrays["v"] * arrays["v"])
            if any(isinstance(arrays[name], core.Tracer) for name in ("u", "v"))
            else np.sqrt(arrays["u"] * arrays["u"] + arrays["v"] * arrays["v"])
        ),
        "temp_growth": arrays["temp_growth"],
        "ca": ca,
        "vcmax": vcmax,
        "control_salinity": arrays["control_salinity"],
        "control_inudate": arrays["control_inudate"],
        "dt_sechiba": dt_sechiba,
        "nlai": DEFAULT_NLAI,
        "laimax": laimax,
        "lai_level_depth": lai_level_depth,
        "min_wind": min_wind,
    }
    xp = jnp if isinstance(ca, core.Tracer) else np
    pft_backgrounds = {
        "gpp": xp.zeros((npts, nvm), dtype=np.float64),
        "gsmean": xp.zeros((npts, nvm), dtype=np.float64),
        "rveget": xp.full((npts, nvm), UNDEF_SECHIBA, dtype=np.float64),
        "rstruct": xp.broadcast_to(rstruct_const[None, :], (npts, nvm)),
        "cimean": xp.broadcast_to(ca[:, None], (npts, nvm)),
    }
    vbeta3_background = xp.zeros((npts, nvm), dtype=np.float64)
    passthrough = {
        name: payload[name]
        for name in (
            "swnet",
            "rau",
            "temp_sol",
            "temp_sol_pft",
            "qsurf",
            "evapot",
            "evapot_corr",
            "snowdz",
            "veget",
            "veget_max",
            "lai",
            "height",
        )
        if name in payload
    }

    return {
        "pft_index": int(pft_index),
        "trans_co2_inputs": trans_co2_inputs,
        "ldq_cdrag_from_gcm": bool(ldq_cdrag_from_gcm),
        "u": arrays["u"],
        "v": arrays["v"],
        "zlev": arrays["zlev"],
        "z0h": arrays["z0h"],
        "z0m": arrays["z0m"],
        "roughheight": arrays["roughheight"],
        "roughheight_pft": arrays["roughheight_pft"],
        "temp_sol": arrays["temp_sol"],
        "temp_sol_pft": arrays["temp_sol_pft"],
        "temp_air": arrays["temp_air"],
        "qsurf": arrays["qsurf"],
        "qair": arrays["qair"],
        "pb": arrays["pb"],
        "rau": arrays["rau"],
        "humrel": arrays["humrel"],
        "veget": arrays["veget"],
        "veget_max": arrays["veget_max"],
        "lai": arrays["lai"],
        "qsintveg": arrays["qsintveg"],
        "qsintmax": arrays["qsintmax"],
        "rstruct": arrays["rstruct"],
        "ok_laidev": ok_laidev,
        "snow": arrays["snow"],
        "frac_nobio": arrays["frac_nobio"],
        "totfrac_nobio": arrays["totfrac_nobio"],
        "snow_nobio": arrays["snow_nobio"],
        "frac_snow_veg": arrays["frac_snow_veg"],
        "frac_snow_nobio": arrays["frac_snow_nobio"],
        "evapot": arrays["evapot"],
        "evapot_corr": arrays["evapot_corr"],
        "flood_frac": arrays["flood_frac"],
        "flood_res": arrays["flood_res"],
        "tot_bare_soil": arrays["tot_bare_soil"],
        "evap_bare_lim": arrays["evap_bare_lim"],
        "vbeta3_background": vbeta3_background,
        "dt_sechiba": dt_sechiba,
        "min_wind": min_wind,
        "snowcri": snowcri,
        "ok_snowfact": ok_snowfact,
        "rough_dyn": rough_dyn,
        "pft_output_backgrounds": pft_backgrounds,
        "passthrough": passthrough,
    }


def run_pft14_local_enerbil_precall_from_first_step_precall(
    precall: DiffucoFirstStepPrecallAssembly,
    *,
    run_def_values: Mapping[str, str],
    pft_index: int = 13,
    ldq_cdrag_from_gcm: bool = False,
    static_kwargs: Mapping[str, object] | None = None,
    use_jit: bool = False,
) -> DiffucoPFT14FirstStepLocalEnerbilPrecallAssembly:
    """Execute local first-step DIFFUCO and return the ENERBIL payload."""

    kwargs = assemble_pft14_local_enerbil_precall_kwargs_from_first_step_precall(
        precall,
        run_def_values=run_def_values,
        pft_index=pft_index,
        ldq_cdrag_from_gcm=ldq_cdrag_from_gcm,
        static_kwargs=static_kwargs,
    )
    kwargs["use_jit"] = bool(use_jit)
    result = diffuco_pft14_local_enerbil_precall_explicit(**kwargs)
    return DiffucoPFT14FirstStepLocalEnerbilPrecallAssembly(
        kwargs=kwargs,
        result=result,
        passthrough_fields=tuple(kwargs["passthrough"].keys()),
        parameter_inputs=(
            "used_run.def: DT_SECHIBA, MIN_WIND, SNOWCRI, OK_SNOWFACT, ROUGH_DYN, LAIMAX, LAI_LEVEL_DEPTH",
            "used_run.def indexed PFT parameters for DIFFUCO C3 photosynthesis",
            "PFT_TO_MTC plus constantes_mtc.f90 alpha_LL_mtc",
        ),
        provenance=(
            *precall.provenance,
            "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_main lines 629-710",
            "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_trans_co2 lines 2284-2969",
            "fortran_source/ORCHIDEE/src_parameters/pft_parameters.f90 lines 117-123, 333, 484",
            "fortran_source/ORCHIDEE/src_parameters/constantes_mtc.f90 lines 440-442",
        ),
    )


def diffuco_trans_co2_activity(
    *,
    swdown,
    humrel,
    veget,
    veget_max,
    lai,
    qsintveg,
    qsintmax,
    temp_growth,
    tphoto_min,
    tphoto_max,
    ok_laidev,
    min_sechiba=MIN_SECHIBA,
):
    """Return source-order activity and water-limitation inputs for CO2 transfer.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_trans_co2``, lines 2338-2384 classify points where
    photosynthesis is calculated for each PFT; lines 2393-2400 compute
    ``zqsvegrap`` and ``water_lim``. This helper stops before the FvCB/Yin
    photosynthesis algebra at line 2411 and does not synthesize assimilation
    or conductance.
    """

    swdown = jnp.asarray(swdown, dtype=jnp.float64)
    humrel = jnp.asarray(humrel, dtype=jnp.float64)
    veget = jnp.asarray(veget, dtype=jnp.float64)
    veget_max = jnp.asarray(veget_max, dtype=jnp.float64)
    lai = jnp.asarray(lai, dtype=jnp.float64)
    qsintveg = jnp.asarray(qsintveg, dtype=jnp.float64)
    qsintmax = jnp.asarray(qsintmax, dtype=jnp.float64)
    temp_growth = jnp.asarray(temp_growth, dtype=jnp.float64)
    tphoto_min = jnp.asarray(tphoto_min, dtype=jnp.float64)
    tphoto_max = jnp.asarray(tphoto_max, dtype=jnp.float64)
    ok_laidev = jnp.asarray(ok_laidev, dtype=bool)
    if humrel.ndim != 2:
        raise ValueError("humrel must have shape (npts, nvm)")
    pft_shape = humrel.shape
    for name, value in (
        ("veget", veget),
        ("veget_max", veget_max),
        ("lai", lai),
        ("qsintveg", qsintveg),
        ("qsintmax", qsintmax),
    ):
        if value.shape != pft_shape:
            raise ValueError(f"{name} must have shape {pft_shape}")
    for name, value in (("swdown", swdown), ("temp_growth", temp_growth)):
        if value.ndim != 1 or value.shape[0] != pft_shape[0]:
            raise ValueError(f"{name} must have shape (npts,)")
    for name, value in (("tphoto_min", tphoto_min), ("tphoto_max", tphoto_max), ("ok_laidev", ok_laidev)):
        if value.ndim != 1 or value.shape[0] != pft_shape[1]:
            raise ValueError(f"{name} must have shape (nvm,)")

    lai_threshold = jnp.where(ok_laidev, 0.001, 0.01)
    lai_active = (lai > lai_threshold[None, :]) & (veget_max > min_sechiba)
    assimilate = (
        lai_active
        & (veget > min_sechiba)
        & (swdown[:, None] > min_sechiba)
        & (humrel > min_sechiba)
        & (temp_growth[:, None] > tphoto_min[None, :])
        & (temp_growth[:, None] < tphoto_max[None, :])
    )
    zqsvegrap = jnp.where(
        qsintmax > min_sechiba,
        jnp.maximum(0.0, qsintveg / jnp.where(qsintmax > min_sechiba, qsintmax, 1.0)),
        0.0,
    )
    water_lim = humrel
    return DiffucoTransCO2ActivityResult(
        lai_active=lai_active,
        assimilate=assimilate,
        zqsvegrap=zqsvegrap,
        water_lim=water_lim,
    )


def diffuco_co2_downregulated_vcmax(
    assim_vcmax,
    ca,
    *,
    downregulation_co2,
    downregulation_co2_coeff,
    downregulation_co2_baselevel,
):
    """Apply ``diffuco_trans_co2`` lines 2240-2246 to Vcmax parameters."""

    assim_vcmax = jnp.asarray(assim_vcmax, dtype=jnp.float64)
    ca = jnp.asarray(ca, dtype=jnp.float64)
    coeff = jnp.asarray(downregulation_co2_coeff, dtype=jnp.float64)
    if assim_vcmax.ndim != 2 or ca.shape != (assim_vcmax.shape[0],):
        raise ValueError("assim_vcmax must be (npts,nvm) and ca must be (npts,)")
    if coeff.shape != (assim_vcmax.shape[1],):
        raise ValueError("downregulation_co2_coeff must have shape (nvm,)")
    if not downregulation_co2:
        return assim_vcmax
    return assim_vcmax * (
        1.0
        - coeff[None, :]
        * jnp.log(ca[:, None] / jnp.asarray(downregulation_co2_baselevel))
    )


def diffuco_lai_light_table(
    ext_coeff,
    *,
    nlai=DEFAULT_NLAI,
    laimax=DEFAULT_LAIMAX,
    lai_level_depth=DEFAULT_LAI_LEVEL_DEPTH,
):
    """Build ``diffuco_trans_co2`` LAI levels and Beer light fractions.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_trans_co2``, lines 2214-2234. Defaults for ``nlai``,
    ``laimax``, and ``lai_level_depth`` come from
    ``src_parameters/constantes_var.f90`` lines 779, 783, and 809. ``ext_coeff``
    is a caller-supplied PFT parameter from ``pft_parameters``.
    """

    ext_coeff = jnp.asarray(ext_coeff, dtype=jnp.float64)
    if ext_coeff.ndim != 1:
        raise ValueError("ext_coeff must have shape (nvm,)")
    nlai = int(nlai)
    if nlai <= 0:
        raise ValueError("nlai must be positive")
    laimax = jnp.asarray(laimax, dtype=jnp.float64)
    depth = jnp.asarray(lai_level_depth, dtype=jnp.float64)
    jl = jnp.arange(nlai + 1, dtype=jnp.float64)
    denom = jnp.exp(depth * jnp.asarray(nlai, dtype=jnp.float64)) - 1.0
    laitab = laimax * (jnp.exp(depth * jl) - 1.0) / denom
    light = jnp.exp(-ext_coeff[:, None] * laitab[:-1][None, :])
    return DiffucoLAILightTable(laitab=laitab, light=light)


@lru_cache(maxsize=64)
def _cached_diffuco_lai_light_table(
    ext_coeff: float,
    nlai: int,
    laimax: float,
    lai_level_depth: float,
) -> DiffucoLAILightTable:
    return diffuco_lai_light_table(
        np.asarray([float(ext_coeff)], dtype=np.float64),
        nlai=int(nlai),
        laimax=float(laimax),
        lai_level_depth=float(lai_level_depth),
    )


def diffuco_vpd_boundary_conductance(
    *,
    qsurf,
    qsatt,
    t2m,
    pb,
    water_lim,
    a1,
    b1,
    stress_gs,
    tetens_1=TETENS_1,
    tetens_2=TETENS_2,
    gb_ref=GB_REF,
    tp_00=TP_00,
    pb_std=PB_STD,
    ratio_h2o_to_co2=RATIO_H2O_TO_CO2,
    min_sechiba=MIN_SECHIBA,
):
    """Compute VPD, ``fvpd``, and boundary-layer conductance inputs.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_trans_co2``, lines 2267-2276 compute ``air_relhum`` and VPD from
    explicit ``qsatt``; lines 2486-2497 compute ``fvpd``, ``gb_h2o``, and
    ``gb_co2``. This helper does not run ``qsatcalc``; callers must pass exact
    saturated humidity from a source-backed qsat helper or trace.
    """

    qsurf = jnp.asarray(qsurf, dtype=jnp.float64)
    qsatt = jnp.asarray(qsatt, dtype=jnp.float64)
    t2m = jnp.asarray(t2m, dtype=jnp.float64)
    pb = jnp.asarray(pb, dtype=jnp.float64)
    water_lim = jnp.asarray(water_lim, dtype=jnp.float64)
    a1 = jnp.asarray(a1, dtype=jnp.float64)
    b1 = jnp.asarray(b1, dtype=jnp.float64)
    stress_gs = jnp.asarray(stress_gs, dtype=jnp.float64)
    if water_lim.ndim != 2:
        raise ValueError("water_lim must have shape (npts, nvm)")
    pft_shape = water_lim.shape
    for name, value in (("qsurf", qsurf), ("qsatt", qsatt), ("t2m", t2m), ("pb", pb)):
        if value.ndim != 1 or value.shape[0] != pft_shape[0]:
            raise ValueError(f"{name} must have shape (npts,)")
    for name, value in (("a1", a1), ("b1", b1), ("stress_gs", stress_gs)):
        if value.ndim != 1 or value.shape[0] != pft_shape[1]:
            raise ValueError(f"{name} must have shape (nvm,)")

    vapor_air = qsurf * pb / (tetens_1 + qsurf * tetens_2)
    vapor_sat = qsatt * pb / (tetens_1 + qsatt * tetens_2)
    air_relhum = vapor_air / vapor_sat
    vpd = (vapor_sat - vapor_air) / 10.0
    bounded = jnp.minimum(
        1.0 - min_sechiba,
        jnp.maximum(min_sechiba, a1[None, :] - b1[None, :] * vpd[:, None]),
    )
    fvpd = (1.0 / (1.0 / bounded - 1.0)) * jnp.maximum(1.0 - stress_gs[None, :], water_lim)
    gb_h2o = jnp.asarray(gb_ref, dtype=jnp.float64) * 44.6 * (tp_00 / t2m) * (pb / pb_std)
    gb_co2 = gb_h2o / ratio_h2o_to_co2
    return DiffucoVPDBoundaryResult(
        air_relhum=air_relhum,
        vpd=vpd,
        fvpd=fvpd,
        gb_h2o=gb_h2o,
        gb_co2=gb_co2,
    )


def diffuco_arrhenius(temp, ref_temp, energy_act, *, rr=RR_GAS):
    """Fortran ``Arrhenius`` helper used by ``diffuco_trans_co2``.

    Provenance: ``src_sechiba/diffuco.f90`` function ``Arrhenius``, lines
    3356-3370.
    """

    temp = jnp.asarray(temp, dtype=jnp.float64)
    return jnp.exp(((temp - ref_temp) * energy_act) / (ref_temp * rr * temp))


def diffuco_arrhenius_modified(temp, ref_temp, energy_act, energy_deact, entropy, *, rr=RR_GAS):
    """Fortran modified Arrhenius helper with scalar or vector entropy.

    Provenance: ``src_sechiba/diffuco.f90`` functions
    ``Arrhenius_modified_1d`` and ``Arrhenius_modified_0d``, lines 3372-3412.
    """

    temp = jnp.asarray(temp, dtype=jnp.float64)
    entropy = jnp.asarray(entropy, dtype=jnp.float64)
    return (
        diffuco_arrhenius(temp, ref_temp, energy_act, rr=rr)
        * (1.0 + jnp.exp((ref_temp * entropy - energy_deact) / (ref_temp * rr)))
        / (1.0 + jnp.exp((temp * entropy - energy_deact) / (rr * temp)))
    )


def diffuco_photo_temperature_response(
    *,
    t2m,
    temp_growth,
    vcmax,
    water_lim,
    e_kmc,
    e_kmo,
    e_sco,
    e_gamma_star,
    e_rd,
    e_jmax,
    d_jmax,
    asj,
    bsj,
    e_vcmax,
    d_vcmax,
    asv,
    bsv,
    e_gm,
    d_gm,
    s_gm,
    arjv,
    brjv,
    gm25,
    stress_gm,
    stress_gs,
    g0,
    kmc25,
    kmo25,
    sco25,
    gamma_star25,
    ref_temp=298.0,
):
    """Compute temperature response inputs for ``diffuco_trans_co2``.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_trans_co2``, lines 2427-2484, using Arrhenius helpers at lines
    3356-3412. PFT parameters are explicit scalar inputs for the active PFT;
    ``vcmax`` and ``water_lim`` are per-point vectors already produced by the
    caller.
    """

    t2m = jnp.asarray(t2m, dtype=jnp.float64)
    temp_growth = jnp.asarray(temp_growth, dtype=jnp.float64)
    vcmax = jnp.asarray(vcmax, dtype=jnp.float64)
    water_lim = jnp.asarray(water_lim, dtype=jnp.float64)
    if not (t2m.ndim == temp_growth.ndim == vcmax.ndim == water_lim.ndim == 1):
        raise ValueError("t2m, temp_growth, vcmax, and water_lim must be one-dimensional")
    if not (t2m.shape == temp_growth.shape == vcmax.shape == water_lim.shape):
        raise ValueError("t2m, temp_growth, vcmax, and water_lim must share shape")

    growth_clamped = jnp.maximum(11.0, jnp.minimum(temp_growth, 35.0))
    t_kmc = diffuco_arrhenius(t2m, ref_temp, e_kmc)
    t_kmo = diffuco_arrhenius(t2m, ref_temp, e_kmo)
    t_sco = diffuco_arrhenius(t2m, ref_temp, e_sco)
    t_gamma_star = diffuco_arrhenius(t2m, ref_temp, e_gamma_star)
    t_rd = diffuco_arrhenius(t2m, ref_temp, e_rd)
    s_jmax = asj + bsj * growth_clamped
    t_jmax = diffuco_arrhenius_modified(t2m, ref_temp, e_jmax, d_jmax, s_jmax)
    s_vcmax = asv + bsv * growth_clamped
    t_vcmax = diffuco_arrhenius_modified(t2m, ref_temp, e_vcmax, d_vcmax, s_vcmax)
    t_gm = diffuco_arrhenius_modified(t2m, ref_temp, e_gm, d_gm, s_gm)

    vc = vcmax * t_vcmax
    vj = (arjv + brjv * growth_clamped) * vcmax * t_jmax
    gm = gm25 * t_gm * jnp.maximum(1.0 - stress_gm, water_lim)
    g0var = g0 * jnp.maximum(1.0 - stress_gs, water_lim)
    kmc = kmc25 * t_kmc
    kmo = kmo25 * t_kmo
    sco = sco25 * t_sco
    gamma_star = gamma_star25 * t_gamma_star
    low_gamma_star = 0.5 / sco
    return DiffucoPhotoTemperatureResult(
        t_kmc=t_kmc,
        t_kmo=t_kmo,
        t_sco=t_sco,
        t_gamma_star=t_gamma_star,
        t_rd=t_rd,
        s_jmax_acclim_temp=s_jmax,
        t_jmax=t_jmax,
        s_vcmax_acclim_temp=s_vcmax,
        t_vcmax=t_vcmax,
        t_gm=t_gm,
        vc=vc,
        vj=vj,
        gm=gm,
        g0var=g0var,
        kmc=kmc,
        kmo=kmo,
        sco=sco,
        gamma_star=gamma_star,
        low_gamma_star=low_gamma_star,
    )


def _yin_cubic_lowest_root(p, q, r, *, root_phase=0.0):
    p = jnp.asarray(p, dtype=jnp.float64)
    q = jnp.asarray(q, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    qq = ((p**2.0) - 3.0 * q) / 9.0
    uu = (2.0 * (p**3.0) - 9.0 * p * q + 27.0 * r) / 54.0
    ratio = uu / (qq**1.5)
    valid = (qq >= 0.0) & (jnp.abs(ratio) <= 1.0)
    psi = jnp.arccos(jnp.clip(ratio, -1.0, 1.0))
    root = -2.0 * jnp.sqrt(jnp.maximum(qq, 0.0)) * jnp.cos((psi + root_phase) / 3.0) - p / 3.0
    return root, valid, qq, uu, psi


def diffuco_c3_assimilation_yin_layer(
    *,
    vc2,
    jj,
    rd,
    ca,
    gamma_star,
    kmc,
    kmo,
    sco,
    gm,
    gb_co2,
    g0var,
    fvpd,
    min_sechiba=MIN_SECHIBA,
):
    """Solve the C3 Yin/FvCB scalar layer equations.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_trans_co2``, C3 branch lines 2705-2785. The two limiting cases
    are evaluated in Fortran order. If the second candidate is not lower than
    the first, the state is restored to the Vc-limited ``x1/x2`` branch as in
    lines 2749-2757.
    """

    vc2 = jnp.asarray(vc2, dtype=jnp.float64)
    jj = jnp.asarray(jj, dtype=jnp.float64)
    rd = jnp.asarray(rd, dtype=jnp.float64)
    ca = jnp.asarray(ca, dtype=jnp.float64)
    gamma_star = jnp.asarray(gamma_star, dtype=jnp.float64)
    kmc = jnp.asarray(kmc, dtype=jnp.float64)
    kmo = jnp.asarray(kmo, dtype=jnp.float64)
    sco = jnp.asarray(sco, dtype=jnp.float64)
    gm = jnp.asarray(gm, dtype=jnp.float64)
    gb_co2 = jnp.asarray(gb_co2, dtype=jnp.float64)
    g0var = jnp.asarray(g0var, dtype=jnp.float64)
    fvpd = jnp.asarray(fvpd, dtype=jnp.float64)
    shape = jnp.broadcast_shapes(
        vc2.shape,
        jj.shape,
        rd.shape,
        ca.shape,
        gamma_star.shape,
        kmc.shape,
        kmo.shape,
        sco.shape,
        gm.shape,
        gb_co2.shape,
        g0var.shape,
        fvpd.shape,
    )
    vc2, jj, rd, ca, gamma_star, kmc, kmo, sco, gm, gb_co2, g0var, fvpd = [
        jnp.broadcast_to(value, shape)
        for value in (vc2, jj, rd, ca, gamma_star, kmc, kmo, sco, gm, gb_co2, g0var, fvpd)
    ]

    def candidate(x1, x2):
        a = g0var * (x2 + gamma_star) + (g0var / gm + fvpd) * (x1 - rd)
        b = ca * (x1 - rd) - gamma_star * x1 - rd * x2
        c = ca + x2 + (1.0 / gm + 1.0 / gb_co2) * (x1 - rd)
        d = x2 + gamma_star + (x1 - rd) / gm
        m = 1.0 / gm + (g0var / gm + fvpd) * (1.0 / gm + 1.0 / gb_co2)
        p = -(d + (x1 - rd) / gm + a * (1.0 / gm + 1.0 / gb_co2) + (g0var / gm + fvpd) * c) / m
        q = (d * (x1 - rd) + a * c + (g0var / gm + fvpd) * b) / m
        r = -a * b / m
        root, valid, qq, uu, psi = _yin_cubic_lowest_root(p, q, r)
        return root, valid, p, q, r, qq, uu, psi

    x1_vc = vc2
    x2_vc = kmc * (1.0 + 2.0 * gamma_star * sco / kmo)
    root_vc, valid_vc, p_vc, q_vc, r_vc, qq_vc, uu_vc, psi_vc = candidate(x1_vc, x2_vc)

    x1_j = jj / 4.0
    x2_j = 2.0 * gamma_star
    root_j, valid_j, p_j, q_j, r_j, qq_j, uu_j, psi_j = candidate(x1_j, x2_j)

    a1_after_vc = jnp.where(valid_vc & (root_vc < 9999.0), root_vc, 9999.0)
    info_after_vc = jnp.where(valid_vc & (root_vc < 9999.0), 2.0, 0.0)

    use_j = valid_j & (root_j < a1_after_vc)
    reset_to_vc = valid_j & (~use_j)
    a1 = jnp.where(use_j, root_j, a1_after_vc)
    x1 = jnp.where(reset_to_vc, x1_vc, x1_j)
    x2 = jnp.where(reset_to_vc, x2_vc, x2_j)
    p = p_j
    q = q_j
    r = r_j
    qq = qq_j
    uu = uu_j
    psi = psi_j
    info = jnp.where(reset_to_vc, 1.0, info_after_vc)

    valid_any = valid_vc | valid_j
    fallback = (~valid_any) | (a1 == 9999.0) | (a1 < -rd)
    assimi = jnp.where(fallback, -rd, a1)
    cc = (gamma_star * x1 + (assimi + rd) * x2) / jnp.maximum(min_sechiba, x1 - (assimi + rd))
    leaf_ci = cc + assimi / gm
    ci_star = gamma_star - rd / gm
    gs = jnp.where(
        jnp.abs(assimi + rd) < min_sechiba,
        g0var,
        g0var + (assimi + rd) / (leaf_ci - ci_star) * fvpd,
    )
    return DiffucoC3AssimilationLayerResult(
        assimi=assimi,
        cc=cc,
        leaf_ci=leaf_ci,
        ci_star=ci_star,
        gs=gs,
        info_limitphoto=info,
        x1=x1,
        x2=x2,
        p=p,
        q=q,
        r=r,
        qq=qq,
        uu=uu,
        psi=psi,
        root_vc=root_vc,
        root_j=root_j,
        valid_vc=valid_vc,
        valid_j=valid_j,
        p_vc=p_vc,
        q_vc=q_vc,
        r_vc=r_vc,
        qq_vc=qq_vc,
        uu_vc=uu_vc,
        psi_vc=psi_vc,
        p_j=p_j,
        q_j=q_j,
        r_j=r_j,
        qq_j=qq_j,
        uu_j=uu_j,
        psi_j=psi_j,
        fallback=fallback,
    )


def diffuco_c3_canopy_layer_integrals(
    *,
    assimilate,
    lai,
    laitab,
    light,
    vc,
    vj,
    vcmax,
    t_rd,
    water_lim,
    swdown,
    ca,
    gamma_star,
    kmc,
    kmo,
    sco,
    gm,
    gb_co2,
    g0var,
    fvpd,
    ext_coeff,
    alpha_ll,
    theta,
    stress_vcmax,
    w_to_mol=W_TO_MOL,
    rg_to_par=RG_TO_PAR,
    min_sechiba=MIN_SECHIBA,
):
    """Integrate the C3 Yin/FvCB layer solver over fixed LAI levels.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_trans_co2``, lines 2503-2558 build the active layer mask,
    nitrogen-scaled ``Vc2``/``Vj2``/``Rd``, absorbed irradiance, and ``JJ``;
    C3 assimilation is solved at lines 2705-2785; lines 2788-2832 keep
    ``leaf_gs_top`` and accumulate ``assimtot``, ``Rdtot``, ``gstot``, and
    ``ilai``; lines 2861-2872 compute ``cim`` and ``laisum`` from ``leaf_ci``.
    This helper covers one active C3 PFT vector and leaves PFT14 salinity and
    inundation controls to the caller before output conversion.
    """

    assimilate = jnp.asarray(assimilate, dtype=bool)
    lai = jnp.asarray(lai, dtype=jnp.float64)
    laitab = jnp.asarray(laitab, dtype=jnp.float64)
    light = jnp.asarray(light, dtype=jnp.float64)
    vc = jnp.asarray(vc, dtype=jnp.float64)
    vj = jnp.asarray(vj, dtype=jnp.float64)
    vcmax = jnp.asarray(vcmax, dtype=jnp.float64)
    t_rd = jnp.asarray(t_rd, dtype=jnp.float64)
    water_lim = jnp.asarray(water_lim, dtype=jnp.float64)
    swdown = jnp.asarray(swdown, dtype=jnp.float64)
    ca = jnp.asarray(ca, dtype=jnp.float64)
    gamma_star = jnp.asarray(gamma_star, dtype=jnp.float64)
    kmc = jnp.asarray(kmc, dtype=jnp.float64)
    kmo = jnp.asarray(kmo, dtype=jnp.float64)
    sco = jnp.asarray(sco, dtype=jnp.float64)
    gm = jnp.asarray(gm, dtype=jnp.float64)
    gb_co2 = jnp.asarray(gb_co2, dtype=jnp.float64)
    g0var = jnp.asarray(g0var, dtype=jnp.float64)
    fvpd = jnp.asarray(fvpd, dtype=jnp.float64)
    ext_coeff = jnp.asarray(ext_coeff, dtype=jnp.float64)
    alpha_ll = jnp.asarray(alpha_ll, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    stress_vcmax = jnp.asarray(stress_vcmax, dtype=jnp.float64)

    if assimilate.ndim != 1:
        raise ValueError("assimilate must have shape (npts,)")
    npts = assimilate.shape[0]
    for name, value in (
        ("lai", lai),
        ("vc", vc),
        ("vj", vj),
        ("vcmax", vcmax),
        ("t_rd", t_rd),
        ("water_lim", water_lim),
        ("swdown", swdown),
        ("ca", ca),
        ("gamma_star", gamma_star),
        ("kmc", kmc),
        ("kmo", kmo),
        ("sco", sco),
        ("gm", gm),
        ("gb_co2", gb_co2),
        ("g0var", g0var),
        ("fvpd", fvpd),
    ):
        if value.ndim != 1 or value.shape[0] != npts:
            raise ValueError(f"{name} must have shape (npts,)")
    if laitab.ndim != 1 or laitab.shape[0] < 2:
        raise ValueError("laitab must have shape (nlai + 1,) with at least one layer")
    nlai = laitab.shape[0] - 1
    if light.ndim != 1 or light.shape[0] != nlai:
        raise ValueError("light must have shape (nlai,)")

    layer_active = assimilate[:, None] & (laitab[:-1][None, :] <= lai[:, None])
    n_vcmax = 1.0 - 0.7 * (1.0 - light)
    stress_factor = jnp.maximum(1.0 - stress_vcmax, water_lim)
    vc2 = vc[:, None] * n_vcmax[None, :] * stress_factor[:, None]
    vj2 = vj[:, None] * n_vcmax[None, :] * stress_factor[:, None]
    rd = vcmax[:, None] * n_vcmax[None, :] * 0.01 * t_rd[:, None] * stress_factor[:, None]
    iabs = swdown[:, None] * w_to_mol * rg_to_par * ext_coeff * light[None, :]
    jmax = vj2
    radical = (alpha_ll * iabs + jmax) ** 2.0 - 4.0 * theta * jmax * alpha_ll * iabs
    jj = (alpha_ll * iabs + jmax - jnp.sqrt(jnp.maximum(radical, 0.0))) / (2.0 * theta)

    layer = diffuco_c3_assimilation_yin_layer(
        vc2=vc2,
        jj=jj,
        rd=rd,
        ca=ca[:, None],
        gamma_star=gamma_star[:, None],
        kmc=kmc[:, None],
        kmo=kmo[:, None],
        sco=sco[:, None],
        gm=gm[:, None],
        gb_co2=gb_co2[:, None],
        g0var=g0var[:, None],
        fvpd=fvpd[:, None],
        min_sechiba=min_sechiba,
    )

    layer_width = laitab[1:] - laitab[:-1]
    assimi = jnp.where(layer_active, layer.assimi, 0.0)
    rd_active = jnp.where(layer_active, rd, 0.0)
    gs = jnp.where(layer_active, layer.gs, 0.0)
    leaf_ci = jnp.where(layer_active, layer.leaf_ci, ca[:, None])
    cc = jnp.where(layer_active, layer.cc, 0.0)
    info = jnp.where(layer_active, layer.info_limitphoto, 0.0)
    fallback = layer_active & layer.fallback

    assimtot = jnp.sum(assimi * layer_width[None, :], axis=1)
    rdtot = jnp.sum(rd_active * layer_width[None, :], axis=1)
    gstot = jnp.sum(gs * layer_width[None, :], axis=1)
    leaf_gs_top = jnp.where(layer_active[:, 0], gs[:, 0], 0.0)
    ilai = jnp.maximum(1, jnp.sum(layer_active, axis=1)).astype(jnp.int32)
    laisum = jnp.sum(jnp.where(laitab[:-1][None, :] <= lai[:, None], layer_width[None, :], 0.0), axis=1)
    cim_sum = jnp.sum(
        jnp.where(laitab[:-1][None, :] <= lai[:, None], leaf_ci * layer_width[None, :], 0.0),
        axis=1,
    )
    cim = jnp.where(laisum > 0.0, cim_sum / jnp.where(laisum > 0.0, laisum, 1.0), 0.0)

    return DiffucoC3CanopyResult(
        vc2=vc2,
        vj2=vj2,
        rd=rd,
        jj=jj,
        iabs=iabs,
        jmax=jmax,
        assimi=assimi,
        cc=cc,
        leaf_ci=leaf_ci,
        gs=gs,
        info_limitphoto=info,
        assimtot=assimtot,
        rdtot=rdtot,
        gstot=gstot,
        leaf_gs_top=leaf_gs_top,
        ilai=ilai,
        laisum=laisum,
        cim=cim,
        calculate=layer_active,
        fallback=fallback,
    )


def diffuco_trans_co2_outputs_from_fvcb(
    *,
    assimilate,
    assimtot,
    rdtot,
    gstot,
    leaf_gs_top,
    gamma_star,
    fvpd,
    g0var,
    laisum,
    ilai,
    laitab,
    veget_max,
    humrel,
    zqsvegrap,
    vbeta23,
    t2m,
    pb,
    wind,
    q_cdrag,
    q_cdrag_pft,
    rveg_pft,
    ca,
    rstruct_const,
    ok_laidev,
    dt_sechiba,
    min_wind=0.1,
    min_sechiba=MIN_SECHIBA,
    undef_sechiba=1.0e20,
    tp_00=TP_00,
    pb_std=PB_STD,
    ratio_h2o_to_co2=RATIO_H2O_TO_CO2,
    mol_to_m_1=MOL_TO_M_1,
):
    """Convert FvCB intermediates to ``diffuco_trans_co2`` output fields.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_trans_co2``, lines 2884-2897 assign ``gsmean``, ``cimean``, and
    ``gpp``; lines 2915-2931 convert conductances to m/s and compute
    ``rveget``/``rstruct``; lines 2935-2969 compute ``speed``, ``cresist``,
    ``vbeta3``, and ``vbeta3pot``. Lines 2284-2288 and 2327-2328 initialize
    inactive columns to ``rveget=undef_sechiba``, ``rstruct=rstruct_const``,
    and ``cimean=Ca``; those source defaults are preserved here. This helper
    assumes the FvCB/Yin loop has already produced explicit intermediates; it
    does not compute assimilation.
    """

    assimilate = jnp.asarray(assimilate, dtype=bool)
    assimtot = jnp.asarray(assimtot, dtype=jnp.float64)
    rdtot = jnp.asarray(rdtot, dtype=jnp.float64)
    gstot = jnp.asarray(gstot, dtype=jnp.float64)
    leaf_gs_top = jnp.asarray(leaf_gs_top, dtype=jnp.float64)
    gamma_star = jnp.asarray(gamma_star, dtype=jnp.float64)
    fvpd = jnp.asarray(fvpd, dtype=jnp.float64)
    g0var = jnp.asarray(g0var, dtype=jnp.float64)
    laisum = jnp.asarray(laisum, dtype=jnp.float64)
    ilai = jnp.asarray(ilai, dtype=jnp.int32)
    laitab = jnp.asarray(laitab, dtype=jnp.float64)
    veget_max = jnp.asarray(veget_max, dtype=jnp.float64)
    humrel = jnp.asarray(humrel, dtype=jnp.float64)
    zqsvegrap = jnp.asarray(zqsvegrap, dtype=jnp.float64)
    vbeta23 = jnp.asarray(vbeta23, dtype=jnp.float64)
    t2m = jnp.asarray(t2m, dtype=jnp.float64)
    pb = jnp.asarray(pb, dtype=jnp.float64)
    wind = jnp.asarray(wind, dtype=jnp.float64)
    q_cdrag = jnp.asarray(q_cdrag, dtype=jnp.float64)
    q_cdrag_pft = jnp.asarray(q_cdrag_pft, dtype=jnp.float64)
    rveg_pft = jnp.asarray(rveg_pft, dtype=jnp.float64)
    ca = jnp.asarray(ca, dtype=jnp.float64)
    rstruct_const = jnp.asarray(rstruct_const, dtype=jnp.float64)
    ok_laidev = jnp.asarray(ok_laidev, dtype=bool)

    if assimilate.ndim != 2:
        raise ValueError("assimilate must have shape (npts, nvm)")
    pft_shape = assimilate.shape
    for name, value in (
        ("assimtot", assimtot),
        ("rdtot", rdtot),
        ("gstot", gstot),
        ("leaf_gs_top", leaf_gs_top),
        ("gamma_star", gamma_star),
        ("fvpd", fvpd),
        ("g0var", g0var),
        ("laisum", laisum),
        ("ilai", ilai),
        ("veget_max", veget_max),
        ("humrel", humrel),
        ("zqsvegrap", zqsvegrap),
        ("vbeta23", vbeta23),
        ("q_cdrag_pft", q_cdrag_pft),
    ):
        if value.shape != pft_shape:
            raise ValueError(f"{name} must have shape {pft_shape}")
    for name, value in (("t2m", t2m), ("pb", pb), ("wind", wind), ("q_cdrag", q_cdrag), ("ca", ca)):
        if value.ndim != 1 or value.shape[0] != pft_shape[0]:
            raise ValueError(f"{name} must have shape (npts,)")
    for name, value in (("rveg_pft", rveg_pft), ("rstruct_const", rstruct_const), ("ok_laidev", ok_laidev)):
        if value.ndim != 1 or value.shape[0] != pft_shape[1]:
            raise ValueError(f"{name} must have shape (nvm,)")
    if laitab.ndim != 1:
        raise ValueError("laitab must be one-dimensional")
    ilai_low = jnp.any(ilai < 1)
    ilai_high = jnp.any(ilai >= laitab.shape[0])
    if not isinstance(ilai_low, core.Tracer) and (bool(ilai_low) or bool(ilai_high)):
        raise ValueError("ilai must index a valid Fortran LAI upper bound in laitab")

    gsmean = jnp.where(assimilate, gstot, 0.0)
    denom = gsmean - g0var * laisum
    cimean_active = jnp.where(
        jnp.abs(denom) > min_sechiba,
        fvpd * (assimtot + rdtot) / jnp.where(jnp.abs(denom) > min_sechiba, denom, 1.0) + gamma_star,
        gamma_star,
    )
    cimean = jnp.where(assimilate, cimean_active, ca[:, None])
    gpp = jnp.where(assimilate, assimtot * 12.0e-6 * veget_max * jnp.asarray(dt_sechiba, dtype=jnp.float64), 0.0)

    conductance_scale = mol_to_m_1 * (t2m[:, None] / tp_00) * (pb_std / pb[:, None]) * ratio_h2o_to_co2
    gstot_m = conductance_scale * gstot
    gstop_m = conductance_scale * leaf_gs_top * laitab[ilai]
    rveget_active = 1.0 / gstop_m
    rstruct_active = jnp.maximum(1.0 / gstot_m - rveget_active, min_sechiba)
    rveget = jnp.where(assimilate, rveget_active, jnp.asarray(undef_sechiba, dtype=jnp.float64))
    rstruct = jnp.where(assimilate, rstruct_active, rstruct_const[None, :])

    speed = jnp.maximum(min_wind, wind)
    drag = jnp.where(ok_laidev[None, :], q_cdrag_pft, q_cdrag[:, None])
    cresist = 1.0 / (1.0 + speed[:, None] * drag * (rveg_pft[None, :] * (rveget + rstruct)))
    vbeta3_active = jnp.where(
        humrel >= min_sechiba,
        veget_max * (1.0 - zqsvegrap) * cresist
        + jnp.minimum(vbeta23, veget_max * zqsvegrap * cresist),
        0.0,
    )
    vbeta3 = jnp.where(assimilate, vbeta3_active, 0.0)
    vbeta3pot = jnp.where(assimilate, jnp.maximum(0.0, veget_max * cresist), 0.0)

    return DiffucoTransCO2OutputResult(
        gsmean=gsmean,
        cimean=cimean,
        gpp=gpp,
        rveget=rveget,
        rstruct=rstruct,
        vbeta3=vbeta3,
        vbeta3pot=vbeta3pot,
        gstot_m_per_s=jnp.where(assimilate, gstot_m, 0.0),
        gstop_m_per_s=jnp.where(assimilate, gstop_m, 0.0),
        cresist=jnp.where(assimilate, cresist, 0.0),
    )


def diffuco_trans_co2_c3_pft_explicit(
    *,
    swdown,
    qsurf,
    qsatt,
    t2m,
    pb,
    wind,
    temp_growth,
    ca,
    vcmax,
    humrel,
    veget,
    veget_max,
    lai,
    qsintveg,
    qsintmax,
    vbeta23,
    q_cdrag,
    q_cdrag_pft,
    tphoto_min,
    tphoto_max,
    ok_laidev,
    ext_coeff,
    a1,
    b1,
    stress_gs,
    e_kmc,
    e_kmo,
    e_sco,
    e_gamma_star,
    e_rd,
    e_jmax,
    d_jmax,
    asj,
    bsj,
    e_vcmax,
    d_vcmax,
    asv,
    bsv,
    e_gm,
    d_gm,
    s_gm,
    arjv,
    brjv,
    gm25,
    stress_gm,
    g0,
    kmc25,
    kmo25,
    sco25,
    gamma_star25,
    alpha_ll,
    theta,
    stress_vcmax,
    rveg_pft,
    rstruct_const,
    dt_sechiba,
    control_salinity,
    control_inudate,
    lai_light=None,
    jv_fortran=14,
    nlai=DEFAULT_NLAI,
    laimax=DEFAULT_LAIMAX,
    lai_level_depth=DEFAULT_LAI_LEVEL_DEPTH,
    min_wind=0.1,
    min_sechiba=MIN_SECHIBA,
):
    """Run the closed C3 branch of ``diffuco_trans_co2`` for one PFT.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_trans_co2``. This wrapper preserves the source order for one C3
    PFT: activity and wet-canopy inputs from lines 2338-2407; temperature,
    VPD, and boundary conductance from lines 2427-2497; fixed-LAI C3
    assimilation and canopy integration from lines 2503-2872; PFT14
    salinity/inundation control from lines 2834-2844; and conductance,
    resistance, GPP, and transpiration beta outputs from lines 2884-2969.

    The caller supplies exact PFT parameters and same-step environmental
    inputs. This helper does not compute `qsatt`, `humrel`, `vbeta23`, or
    mangrove controls from nearby outputs.
    """

    humrel = jnp.asarray(humrel, dtype=jnp.float64)
    veget = jnp.asarray(veget, dtype=jnp.float64)
    veget_max = jnp.asarray(veget_max, dtype=jnp.float64)
    lai = jnp.asarray(lai, dtype=jnp.float64)
    qsintveg = jnp.asarray(qsintveg, dtype=jnp.float64)
    qsintmax = jnp.asarray(qsintmax, dtype=jnp.float64)
    vbeta23 = jnp.asarray(vbeta23, dtype=jnp.float64)
    q_cdrag_pft = jnp.asarray(q_cdrag_pft, dtype=jnp.float64)
    if humrel.ndim != 1:
        raise ValueError("one-PFT humrel must have shape (npts,)")
    npts = humrel.shape[0]
    for name, value in (
        ("veget", veget),
        ("veget_max", veget_max),
        ("lai", lai),
        ("qsintveg", qsintveg),
        ("qsintmax", qsintmax),
        ("vbeta23", vbeta23),
        ("q_cdrag_pft", q_cdrag_pft),
    ):
        if value.ndim != 1 or value.shape[0] != npts:
            raise ValueError(f"{name} must have shape (npts,)")

    pft_humrel = humrel[:, None]
    pft_veget = veget[:, None]
    pft_veget_max = veget_max[:, None]
    pft_lai = lai[:, None]
    pft_qsintveg = qsintveg[:, None]
    pft_qsintmax = qsintmax[:, None]

    activity = diffuco_trans_co2_activity(
        swdown=swdown,
        humrel=pft_humrel,
        veget=pft_veget,
        veget_max=pft_veget_max,
        lai=pft_lai,
        qsintveg=pft_qsintveg,
        qsintmax=pft_qsintmax,
        temp_growth=temp_growth,
        tphoto_min=jnp.asarray([tphoto_min], dtype=jnp.float64),
        tphoto_max=jnp.asarray([tphoto_max], dtype=jnp.float64),
        ok_laidev=jnp.asarray([ok_laidev], dtype=bool),
        min_sechiba=min_sechiba,
    )
    if lai_light is None:
        lai_light = diffuco_lai_light_table(
            jnp.asarray([ext_coeff], dtype=jnp.float64),
            nlai=nlai,
            laimax=laimax,
            lai_level_depth=lai_level_depth,
        )
    vpd_boundary = diffuco_vpd_boundary_conductance(
        qsurf=qsurf,
        qsatt=qsatt,
        t2m=t2m,
        pb=pb,
        water_lim=activity.water_lim,
        a1=jnp.asarray([a1], dtype=jnp.float64),
        b1=jnp.asarray([b1], dtype=jnp.float64),
        stress_gs=jnp.asarray([stress_gs], dtype=jnp.float64),
        min_sechiba=min_sechiba,
    )
    photo_temperature = diffuco_photo_temperature_response(
        t2m=t2m,
        temp_growth=temp_growth,
        vcmax=vcmax,
        water_lim=activity.water_lim[:, 0],
        e_kmc=e_kmc,
        e_kmo=e_kmo,
        e_sco=e_sco,
        e_gamma_star=e_gamma_star,
        e_rd=e_rd,
        e_jmax=e_jmax,
        d_jmax=d_jmax,
        asj=asj,
        bsj=bsj,
        e_vcmax=e_vcmax,
        d_vcmax=d_vcmax,
        asv=asv,
        bsv=bsv,
        e_gm=e_gm,
        d_gm=d_gm,
        s_gm=s_gm,
        arjv=arjv,
        brjv=brjv,
        gm25=gm25,
        stress_gm=stress_gm,
        stress_gs=stress_gs,
        g0=g0,
        kmc25=kmc25,
        kmo25=kmo25,
        sco25=sco25,
        gamma_star25=gamma_star25,
    )
    canopy = diffuco_c3_canopy_layer_integrals(
        assimilate=activity.assimilate[:, 0],
        lai=lai,
        laitab=lai_light.laitab,
        light=lai_light.light[0],
        vc=photo_temperature.vc,
        vj=photo_temperature.vj,
        vcmax=vcmax,
        t_rd=photo_temperature.t_rd,
        water_lim=activity.water_lim[:, 0],
        swdown=swdown,
        ca=ca,
        gamma_star=photo_temperature.gamma_star,
        kmc=photo_temperature.kmc,
        kmo=photo_temperature.kmo,
        sco=photo_temperature.sco,
        gm=photo_temperature.gm,
        gb_co2=vpd_boundary.gb_co2,
        g0var=photo_temperature.g0var,
        fvpd=vpd_boundary.fvpd[:, 0],
        ext_coeff=ext_coeff,
        alpha_ll=alpha_ll,
        theta=theta,
        stress_vcmax=stress_vcmax,
        min_sechiba=min_sechiba,
    )
    controlled_assimtot = diffuco_apply_pft14_assimtot_controls(
        canopy.assimtot,
        control_salinity=control_salinity,
        control_inudate=control_inudate,
        jv=jv_fortran,
    )
    output = diffuco_trans_co2_outputs_from_fvcb(
        assimilate=activity.assimilate,
        assimtot=controlled_assimtot[:, None],
        rdtot=canopy.rdtot[:, None],
        gstot=canopy.gstot[:, None],
        leaf_gs_top=canopy.leaf_gs_top[:, None],
        gamma_star=photo_temperature.gamma_star[:, None],
        fvpd=vpd_boundary.fvpd,
        g0var=photo_temperature.g0var[:, None],
        laisum=canopy.laisum[:, None],
        ilai=canopy.ilai[:, None],
        laitab=lai_light.laitab,
        veget_max=pft_veget_max,
        humrel=pft_humrel,
        zqsvegrap=activity.zqsvegrap,
        vbeta23=vbeta23[:, None],
        t2m=t2m,
        pb=pb,
        wind=wind,
        q_cdrag=q_cdrag,
        q_cdrag_pft=q_cdrag_pft[:, None],
        rveg_pft=jnp.asarray([rveg_pft], dtype=jnp.float64),
        ca=ca,
        rstruct_const=jnp.asarray([rstruct_const], dtype=jnp.float64),
        ok_laidev=jnp.asarray([ok_laidev], dtype=bool),
        dt_sechiba=dt_sechiba,
        min_wind=min_wind,
        min_sechiba=min_sechiba,
    )
    return DiffucoTransCO2C3PFTResult(
        activity=activity,
        lai_light=lai_light,
        vpd_boundary=vpd_boundary,
        photo_temperature=photo_temperature,
        canopy=canopy,
        controlled_assimtot=controlled_assimtot,
        output=output,
    )


def diffuco_pft14_c3_beta_closure_explicit(
    *,
    trans_co2,
    pft_index,
    humrel,
    qair,
    temp_air,
    qsatt,
    veget,
    veget_max,
    lai,
    tot_bare_soil,
    evap_bare_lim,
    vbeta1,
    vbeta2,
    vbeta3_background,
    qsintmax,
    min_sechiba=MIN_SECHIBA,
):
    """Close the audited PFT14 C3 DIFFUCO beta chain.

    Fortran provenance: ``diffuco_main`` calls ``diffuco_trans_co2`` at
    ``src_sechiba/diffuco.f90`` lines 665-671, ``diffuco_bare`` at lines
    698-700, and ``diffuco_comb`` at lines 709-710. This helper inserts the
    closed PFT14 C3 transpiration beta into explicit full-PFT beta arrays,
    then applies the source-backed CWRR bare-soil beta and final combination
    algebra. Other PFT columns are not inferred; the caller must supply their
    current `vbeta3_background` values.
    """

    pft_index = int(pft_index)
    humrel = jnp.asarray(humrel, dtype=jnp.float64)
    veget = jnp.asarray(veget, dtype=jnp.float64)
    veget_max = jnp.asarray(veget_max, dtype=jnp.float64)
    lai = jnp.asarray(lai, dtype=jnp.float64)
    vbeta2 = jnp.asarray(vbeta2, dtype=jnp.float64)
    vbeta3_background = jnp.asarray(vbeta3_background, dtype=jnp.float64)
    if humrel.ndim != 2:
        raise ValueError("humrel must have shape (npts, nvm)")
    if pft_index < 0 or pft_index >= humrel.shape[1]:
        raise ValueError("pft_index must select a valid PFT column")
    for name, value in (
        ("veget", veget),
        ("veget_max", veget_max),
        ("lai", lai),
        ("vbeta2", vbeta2),
        ("vbeta3_background", vbeta3_background),
    ):
        if value.shape != humrel.shape:
            raise ValueError(f"{name} must have shape {humrel.shape}")

    active_vbeta3 = jnp.asarray(trans_co2.output.vbeta3, dtype=jnp.float64)[:, 0]
    active_vbeta3pot = jnp.asarray(trans_co2.output.vbeta3pot, dtype=jnp.float64)[:, 0]
    pft_mask = jnp.arange(humrel.shape[1]) == pft_index
    vbeta3 = jnp.where(pft_mask[None, :], active_vbeta3[:, None], vbeta3_background)
    vbeta3pot = jnp.where(pft_mask[None, :], active_vbeta3pot[:, None], jnp.zeros_like(vbeta3_background))

    bare = diffuco_bare_cwrr_beta(
        evap_bare_lim=evap_bare_lim,
        vbeta2=vbeta2,
        vbeta3=vbeta3,
        veget_max=veget_max,
    )

    comb = diffuco_comb_explicit(
        humrel=humrel,
        qair=qair,
        temp_air=temp_air,
        qsatt=qsatt,
        veget=veget,
        veget_max=veget_max,
        lai=lai,
        tot_bare_soil=tot_bare_soil,
        vbeta1=vbeta1,
        vbeta2=vbeta2,
        vbeta3=vbeta3,
        vbeta4=bare.vbeta4,
        vbeta4_pft=bare.vbeta4_pft,
        qsintmax=qsintmax,
        min_sechiba=min_sechiba,
    )
    return DiffucoPFT14C3BetaClosureResult(
        trans_co2=trans_co2,
        vbeta3=vbeta3,
        vbeta3pot=vbeta3pot,
        bare=bare,
        comb=comb,
    )


DIFFUCO_PFT14_AFTER_MAIN_PAYLOAD_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 997-1005",
    "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_main lines 668-710",
    "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_trans_co2 lines 2884-2976",
    "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_bare lines 1670-1685",
    "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_comb lines 3070-3271",
)

DIFFUCO_PFT14_AFTER_MAIN_LOCAL_FIELDS = (
    "gpp",
    "gsmean",
    "rveget",
    "rstruct",
    "cimean",
    "vbeta3",
    "vbeta3pot",
    "valpha",
    "vbeta",
    "vbeta_pft",
    "vbeta1",
    "vbeta2",
    "vbeta4",
    "vbeta4_pft",
    "vbeta5",
    "humrel",
)


def _diffuco_pft_column(array, pft_index: int, name: str):
    value = jnp.asarray(array, dtype=jnp.float64)
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape (npts, nvm)")
    if pft_index < 0 or pft_index >= value.shape[1]:
        raise ValueError(f"pft_index must select a valid {name} column")
    return value[:, pft_index]


@lru_cache(maxsize=32)
def _diffuco_pft_column_mask(nvm: int, pft_index: int):
    return jnp.arange(int(nvm)) == int(pft_index)


def _diffuco_insert_pft_column(background, column, pft_index: int, name: str):
    base = jnp.asarray(background, dtype=jnp.float64)
    col = jnp.asarray(column, dtype=jnp.float64)
    if base.ndim != 2:
        raise ValueError(f"{name} background must have shape (npts, nvm)")
    if col.ndim != 2 or col.shape[1] != 1:
        raise ValueError(f"{name} column must have shape (npts, 1)")
    if base.shape[0] != col.shape[0]:
        raise ValueError(f"{name} background and column must share npts")
    if pft_index < 0 or pft_index >= base.shape[1]:
        raise ValueError(f"pft_index must select a valid {name} background column")
    pft_mask = _diffuco_pft_column_mask(base.shape[1], int(pft_index))
    return jnp.where(pft_mask[None, :], col, base)


def _diffuco_validate_pft_column_insert(background, column, pft_index: int, name: str) -> None:
    base = jnp.asarray(background)
    col = jnp.asarray(column)
    if base.ndim != 2:
        raise ValueError(f"{name} background must have shape (npts, nvm)")
    if col.ndim != 2 or col.shape[1] != 1:
        raise ValueError(f"{name} column must have shape (npts, 1)")
    if base.shape[0] != col.shape[0]:
        raise ValueError(f"{name} background and column must share npts")
    if pft_index < 0 or pft_index >= base.shape[1]:
        raise ValueError(f"pft_index must select a valid {name} background column")


def _diffuco_insert_pft_output_columns(
    gpp_background,
    gsmean_background,
    rveget_background,
    rstruct_background,
    cimean_background,
    gpp_column,
    gsmean_column,
    rveget_column,
    rstruct_column,
    cimean_column,
    *,
    pft_index: int,
):
    nvm = jnp.asarray(gpp_background).shape[1]
    pft_mask = _diffuco_pft_column_mask(nvm, int(pft_index))[None, :]
    return (
        jnp.where(pft_mask, jnp.asarray(gpp_column, dtype=jnp.float64), jnp.asarray(gpp_background, dtype=jnp.float64)),
        jnp.where(pft_mask, jnp.asarray(gsmean_column, dtype=jnp.float64), jnp.asarray(gsmean_background, dtype=jnp.float64)),
        jnp.where(pft_mask, jnp.asarray(rveget_column, dtype=jnp.float64), jnp.asarray(rveget_background, dtype=jnp.float64)),
        jnp.where(pft_mask, jnp.asarray(rstruct_column, dtype=jnp.float64), jnp.asarray(rstruct_background, dtype=jnp.float64)),
        jnp.where(pft_mask, jnp.asarray(cimean_column, dtype=jnp.float64), jnp.asarray(cimean_background, dtype=jnp.float64)),
    )


_diffuco_insert_pft_output_columns_jit = jit(
    _diffuco_insert_pft_output_columns,
    static_argnames=("pft_index",),
)


def diffuco_pft14_after_main_payload(
    closure: DiffucoPFT14C3BetaClosureResult | DiffucoPFT14C3BetaProcessChainResult,
    *,
    pft_index: int,
    passthrough: Mapping[str, object] | None = None,
) -> DiffucoPFT14AfterMainPayload:
    """Assemble the source-backed PFT14 DIFFUCO after-main boundary subset.

    Fortran provenance: ``sechiba_main`` calls ``diffuco_main`` at
    ``src_sechiba/sechiba.f90`` lines 997-1005. Inside ``diffuco_main``, the
    active PFT14 C3 path calls ``diffuco_trans_co2`` at
    ``src_sechiba/diffuco.f90`` lines 668-671, ``diffuco_bare`` at lines
    698-700, and ``diffuco_comb`` at lines 709-710. This helper exports only
    fields computed by ``diffuco_pft14_c3_beta_closure_explicit`` plus explicit
    pass-through fields supplied by the caller; it does not infer driver,
    vegetation, drag, flood, snow, or upstream state.
    """

    pft_index = int(pft_index)
    flood = closure.flood if isinstance(closure, DiffucoPFT14C3BetaProcessChainResult) else None
    closure = closure.closure if isinstance(closure, DiffucoPFT14C3BetaProcessChainResult) else closure
    output = closure.trans_co2.output
    local_payload: dict[str, object] = {
        "gpp": _diffuco_pft_column(output.gpp, 0, "trans_co2.output.gpp"),
        "gsmean": _diffuco_pft_column(output.gsmean, 0, "trans_co2.output.gsmean"),
        "rveget": _diffuco_pft_column(output.rveget, 0, "trans_co2.output.rveget"),
        "rstruct": _diffuco_pft_column(output.rstruct, 0, "trans_co2.output.rstruct"),
        "cimean": _diffuco_pft_column(output.cimean, 0, "trans_co2.output.cimean"),
        "vbeta3": _diffuco_pft_column(closure.comb.vbeta3, pft_index, "vbeta3"),
        "vbeta3pot": _diffuco_pft_column(closure.vbeta3pot, pft_index, "vbeta3pot"),
        "valpha": jnp.asarray(closure.comb.valpha, dtype=jnp.float64),
        "vbeta": jnp.asarray(closure.comb.vbeta, dtype=jnp.float64),
        "vbeta_pft": _diffuco_pft_column(closure.comb.vbeta_pft, pft_index, "vbeta_pft"),
        "vbeta1": jnp.asarray(closure.comb.vbeta1, dtype=jnp.float64),
        "vbeta2": _diffuco_pft_column(closure.comb.vbeta2, pft_index, "vbeta2"),
        "vbeta4": jnp.asarray(closure.comb.vbeta4, dtype=jnp.float64),
        "vbeta4_pft": _diffuco_pft_column(closure.comb.vbeta4_pft, pft_index, "vbeta4_pft"),
        "humrel": _diffuco_pft_column(closure.comb.humrel, pft_index, "humrel"),
    }
    if flood is not None:
        local_payload["vbeta5"] = jnp.asarray(flood.vbeta5, dtype=jnp.float64)
    payload = dict(local_payload)
    passthrough_fields: tuple[str, ...] = ()
    if passthrough is not None:
        overlap = set(payload).intersection(passthrough)
        if overlap:
            raise ValueError(f"pass-through fields overlap local DIFFUCO outputs: {tuple(sorted(overlap))}")
        payload.update(passthrough)
        passthrough_fields = tuple(passthrough.keys())

    return DiffucoPFT14AfterMainPayload(
        payload=payload,
        local_fields=tuple(local_payload.keys()),
        passthrough_fields=passthrough_fields,
        provenance=DIFFUCO_PFT14_AFTER_MAIN_PAYLOAD_PROVENANCE,
    )


DIFFUCO_PFT14_ENERBIL_PRECALL_LOCAL_FIELDS = DIFFUCO_PFT14_AFTER_MAIN_LOCAL_FIELDS
DIFFUCO_PFT14_ENERBIL_PRECALL_PROVENANCE = (
    *DIFFUCO_PFT14_AFTER_MAIN_PAYLOAD_PROVENANCE,
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1013-1019",
)


def diffuco_pft14_enerbil_precall_payload(
    closure: DiffucoPFT14C3BetaClosureResult | DiffucoPFT14C3BetaProcessChainResult,
    *,
    pft_index: int,
    pft_output_backgrounds: Mapping[str, object],
    passthrough: Mapping[str, object] | None = None,
) -> DiffucoPFT14AfterMainPayload:
    """Assemble full-array DIFFUCO fields for the ENERBIL pre-call boundary.

    Fortran provenance is the same DIFFUCO source path as
    ``diffuco_pft14_after_main_payload``, with the full-array consumer boundary
    at ``src_sechiba/sechiba.f90`` lines 1013-1019. ``gpp``, ``gsmean``,
    ``rveget``, ``rstruct``, and ``cimean`` are produced locally only for the
    active PFT14 column; all other PFT columns must be supplied explicitly via
    ``pft_output_backgrounds`` and are not inferred here.
    """

    pft_index = int(pft_index)
    flood = closure.flood if isinstance(closure, DiffucoPFT14C3BetaProcessChainResult) else None
    closure = closure.closure if isinstance(closure, DiffucoPFT14C3BetaProcessChainResult) else closure
    output = closure.trans_co2.output
    required_backgrounds = ("gpp", "gsmean", "rveget", "rstruct", "cimean")
    missing = tuple(name for name in required_backgrounds if name not in pft_output_backgrounds)
    if missing:
        raise ValueError(f"missing PFT output backgrounds: {missing}")

    for name, column in (
        ("gpp", output.gpp),
        ("gsmean", output.gsmean),
        ("rveget", output.rveget),
        ("rstruct", output.rstruct),
        ("cimean", output.cimean),
    ):
        _diffuco_validate_pft_column_insert(
            pft_output_backgrounds[name],
            column,
            pft_index,
            name,
        )
    gpp, gsmean, rveget, rstruct, cimean = _diffuco_insert_pft_output_columns_jit(
        pft_output_backgrounds["gpp"],
        pft_output_backgrounds["gsmean"],
        pft_output_backgrounds["rveget"],
        pft_output_backgrounds["rstruct"],
        pft_output_backgrounds["cimean"],
        output.gpp,
        output.gsmean,
        output.rveget,
        output.rstruct,
        output.cimean,
        pft_index=pft_index,
    )
    local_payload: dict[str, object] = {
        "gpp": gpp,
        "gsmean": gsmean,
        "rveget": rveget,
        "rstruct": rstruct,
        "cimean": cimean,
        "vbeta3": jnp.asarray(closure.comb.vbeta3, dtype=jnp.float64),
        "vbeta3pot": jnp.asarray(closure.vbeta3pot, dtype=jnp.float64),
        "valpha": jnp.asarray(closure.comb.valpha, dtype=jnp.float64),
        "vbeta": jnp.asarray(closure.comb.vbeta, dtype=jnp.float64),
        "vbeta_pft": jnp.asarray(closure.comb.vbeta_pft, dtype=jnp.float64),
        "vbeta1": jnp.asarray(closure.comb.vbeta1, dtype=jnp.float64),
        "vbeta2": jnp.asarray(closure.comb.vbeta2, dtype=jnp.float64),
        "vbeta4": jnp.asarray(closure.comb.vbeta4, dtype=jnp.float64),
        "vbeta4_pft": jnp.asarray(closure.comb.vbeta4_pft, dtype=jnp.float64),
        "humrel": jnp.asarray(closure.comb.humrel, dtype=jnp.float64),
    }
    if flood is not None:
        local_payload["vbeta5"] = jnp.asarray(flood.vbeta5, dtype=jnp.float64)

    payload = dict(local_payload)
    passthrough_fields: tuple[str, ...] = ()
    if passthrough is not None:
        overlap = set(payload).intersection(passthrough)
        if overlap:
            raise ValueError(f"pass-through fields overlap local DIFFUCO outputs: {tuple(sorted(overlap))}")
        payload.update(passthrough)
        passthrough_fields = tuple(passthrough.keys())

    return DiffucoPFT14AfterMainPayload(
        payload=payload,
        local_fields=tuple(local_payload.keys()),
        passthrough_fields=passthrough_fields,
        provenance=DIFFUCO_PFT14_ENERBIL_PRECALL_PROVENANCE,
    )


def diffuco_inter_explicit(
    *,
    qair,
    qsatt,
    rau,
    u,
    v,
    q_cdrag,
    q_cdrag_pft,
    humrel,
    veget,
    qsintveg,
    qsintmax,
    rstruct,
    ok_laidev,
    dt_sechiba,
    min_wind=0.1,
    min_sechiba=MIN_SECHIBA,
):
    """Compute ``diffuco_inter`` interception and wetted-foliage betas.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_inter``, lines 1426-1553. The implemented source path initializes
    ``vbeta2`` and ``vbeta23`` to zero, loops over ``jv = 2,nvm``, computes the
    wetted vegetation fraction ``zqsvegrap`` at lines 1477-1484, chooses
    grid/PFT drag according to ``ok_LAIdev`` at lines 1494-1501, and applies
    the leaf-storage limiter at lines 1510-1535.
    """

    qair = jnp.asarray(qair, dtype=jnp.float64)
    qsatt = jnp.asarray(qsatt, dtype=jnp.float64)
    rau = jnp.asarray(rau, dtype=jnp.float64)
    u = jnp.asarray(u, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)
    q_cdrag = jnp.asarray(q_cdrag, dtype=jnp.float64)
    q_cdrag_pft = jnp.asarray(q_cdrag_pft, dtype=jnp.float64)
    humrel = jnp.asarray(humrel, dtype=jnp.float64)
    veget = jnp.asarray(veget, dtype=jnp.float64)
    qsintveg = jnp.asarray(qsintveg, dtype=jnp.float64)
    qsintmax = jnp.asarray(qsintmax, dtype=jnp.float64)
    rstruct = jnp.asarray(rstruct, dtype=jnp.float64)
    ok_laidev = jnp.asarray(ok_laidev, dtype=bool)
    dt_sechiba = jnp.asarray(dt_sechiba, dtype=jnp.float64)

    if humrel.ndim != 2:
        raise ValueError("humrel must have shape (npts, nvm)")
    pft_shape = humrel.shape
    for name, value in (
        ("q_cdrag_pft", q_cdrag_pft),
        ("veget", veget),
        ("qsintveg", qsintveg),
        ("qsintmax", qsintmax),
        ("rstruct", rstruct),
    ):
        if value.shape != pft_shape:
            raise ValueError(f"{name} must have shape {pft_shape}")
    for name, value in (
        ("qair", qair),
        ("qsatt", qsatt),
        ("rau", rau),
        ("u", u),
        ("v", v),
        ("q_cdrag", q_cdrag),
    ):
        if value.ndim != 1 or value.shape[0] != pft_shape[0]:
            raise ValueError(f"{name} must have shape (npts,)")
    if ok_laidev.ndim != 1 or ok_laidev.shape[0] != pft_shape[1]:
        raise ValueError("ok_laidev must have shape (nvm,)")

    speed = jnp.maximum(jnp.asarray(min_wind, dtype=jnp.float64), jnp.sqrt(u * u + v * v))
    zqsvegrap = jnp.where(
        qsintmax > min_sechiba,
        jnp.maximum(0.0, qsintveg / jnp.where(qsintmax > min_sechiba, qsintmax, 1.0)),
        0.0,
    )
    active = (veget > min_sechiba) & (qsintveg > 0.0)
    drag = jnp.where(ok_laidev[None, :], q_cdrag_pft, q_cdrag[:, None])
    pft0_mask = jnp.arange(humrel.shape[1]) == 0
    initial_vbeta2 = veget * zqsvegrap / (1.0 + speed[:, None] * drag * rstruct)
    initial_vbeta2 = jnp.where(active, initial_vbeta2, 0.0)
    initial_vbeta2 = jnp.where(pft0_mask[None, :], 0.0, initial_vbeta2)

    ziltest = dt_sechiba * initial_vbeta2 * speed[:, None] * q_cdrag[:, None] * rau[:, None] * (
        qsatt[:, None] - qair[:, None]
    )
    limiter_active = active & (ziltest > 0.0)
    zrapp = jnp.where(limiter_active, qsintveg / jnp.where(limiter_active, ziltest, 1.0), 0.0)
    limited_by_storage = limiter_active & (zrapp < 1.0)
    positive_humrel = humrel >= min_sechiba
    vbeta23 = jnp.where(
        limited_by_storage & positive_humrel,
        jnp.maximum(initial_vbeta2 - initial_vbeta2 * zrapp, 0.0),
        0.0,
    )
    vbeta2 = jnp.where(limited_by_storage, initial_vbeta2 * zrapp, initial_vbeta2)
    vbeta2 = jnp.where(pft0_mask[None, :], 0.0, vbeta2)
    vbeta23 = jnp.where(pft0_mask[None, :], 0.0, vbeta23)

    return DiffucoInterResult(
        vbeta2=vbeta2,
        vbeta23=vbeta23,
        zqsvegrap=zqsvegrap,
        ziltest=ziltest,
        zrapp=zrapp,
        active=jnp.where(pft0_mask[None, :], False, active),
        limited_by_storage=jnp.where(pft0_mask[None, :], False, limited_by_storage),
    )


def diffuco_pft14_c3_chain_with_inter_explicit(
    *,
    pft_index: int,
    trans_co2_inputs: Mapping[str, object],
    qair,
    qsatt,
    temp_air,
    rau,
    u,
    v,
    q_cdrag,
    q_cdrag_pft,
    humrel,
    veget,
    veget_max,
    lai,
    qsintveg,
    qsintmax,
    rstruct,
    ok_laidev,
    tot_bare_soil,
    evap_bare_lim,
    vbeta1,
    vbeta3_background,
    dt_sechiba,
    min_wind=0.1,
    min_sechiba=MIN_SECHIBA,
) -> DiffucoPFT14C3ChainWithInterResult:
    """Run the local PFT14 C3 DIFFUCO chain starting at ``diffuco_inter``.

    Fortran provenance: ``src_sechiba/diffuco.f90::diffuco_main`` calls
    ``diffuco_inter`` at lines 659-661, ``diffuco_trans_co2`` at lines
    668-671, ``diffuco_bare`` at lines 698-700, and ``diffuco_comb`` at lines
    709-710. This wrapper uses local ``diffuco_inter`` outputs for ``vbeta2``
    and ``vbeta23``; all meteorological, vegetation, drag, canopy-water,
    resistance, and parameter inputs remain explicit.
    """

    pft_index = int(pft_index)
    humrel = jnp.asarray(humrel, dtype=jnp.float64)
    veget = jnp.asarray(veget, dtype=jnp.float64)
    veget_max = jnp.asarray(veget_max, dtype=jnp.float64)
    lai = jnp.asarray(lai, dtype=jnp.float64)
    qsintveg = jnp.asarray(qsintveg, dtype=jnp.float64)
    qsintmax = jnp.asarray(qsintmax, dtype=jnp.float64)
    q_cdrag_pft = jnp.asarray(q_cdrag_pft, dtype=jnp.float64)
    ok_laidev = jnp.asarray(ok_laidev, dtype=bool)
    ok_laidev_pft = ok_laidev[pft_index]
    if humrel.ndim != 2:
        raise ValueError("humrel must have shape (npts, nvm)")
    if pft_index < 0 or pft_index >= humrel.shape[1]:
        raise ValueError("pft_index must select a valid PFT column")
    for name, value in (
        ("veget", veget),
        ("veget_max", veget_max),
        ("lai", lai),
        ("qsintveg", qsintveg),
        ("qsintmax", qsintmax),
        ("q_cdrag_pft", q_cdrag_pft),
    ):
        if value.shape != humrel.shape:
            raise ValueError(f"{name} must have shape {humrel.shape}")
    if ok_laidev.ndim != 1 or ok_laidev.shape[0] != humrel.shape[1]:
        raise ValueError("ok_laidev must have shape (nvm,)")

    inter = diffuco_inter_explicit(
        qair=qair,
        qsatt=qsatt,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        q_cdrag_pft=q_cdrag_pft,
        humrel=humrel,
        veget=veget,
        qsintveg=qsintveg,
        qsintmax=qsintmax,
        rstruct=rstruct,
        ok_laidev=ok_laidev,
        dt_sechiba=dt_sechiba,
        min_wind=min_wind,
        min_sechiba=min_sechiba,
    )

    trans_inputs = dict(trans_co2_inputs)
    trans_qsatt = diffuco_trans_co2_qsatt_explicit(t2m=trans_inputs["t2m"], pb=trans_inputs["pb"])
    trans_inputs.update(
        {
            "qsatt": trans_qsatt,
            "humrel": humrel[:, pft_index],
            "veget": veget[:, pft_index],
            "veget_max": veget_max[:, pft_index],
            "lai": lai[:, pft_index],
            "qsintveg": qsintveg[:, pft_index],
            "qsintmax": qsintmax[:, pft_index],
            "vbeta23": inter.vbeta23[:, pft_index],
            "q_cdrag": q_cdrag,
            "q_cdrag_pft": q_cdrag_pft[:, pft_index],
            "ok_laidev": ok_laidev_pft,
            "dt_sechiba": dt_sechiba,
            "min_wind": min_wind,
            "min_sechiba": min_sechiba,
        }
    )
    trans = diffuco_trans_co2_c3_pft_explicit(**trans_inputs)

    closure = diffuco_pft14_c3_beta_closure_explicit(
        trans_co2=trans,
        pft_index=pft_index,
        humrel=humrel,
        qair=qair,
        temp_air=temp_air,
        qsatt=qsatt,
        veget=veget,
        veget_max=veget_max,
        lai=lai,
        tot_bare_soil=tot_bare_soil,
        evap_bare_lim=evap_bare_lim,
        vbeta1=vbeta1,
        vbeta2=inter.vbeta2,
        vbeta3_background=vbeta3_background,
        qsintmax=qsintmax,
        min_sechiba=min_sechiba,
    )
    return DiffucoPFT14C3ChainWithInterResult(inter=inter, trans_co2=trans, closure=closure)


def diffuco_coupled_q_cdrag_pft(q_cdrag, *, nvm: int):
    """Broadcast grid drag to all PFTs for the coupled/GCM drag branch.

    Fortran provenance: ``src_sechiba/diffuco.f90::diffuco_main`` lines
    633-637. When ``ldq_cdrag_from_gcm`` is true, ORCHIDEE cannot distinguish
    drag by PFT and assigns ``q_cdrag_pft(:,jv) = q_cdrag`` for every PFT.
    """

    q_cdrag = jnp.asarray(q_cdrag, dtype=jnp.float64)
    if q_cdrag.ndim != 1:
        raise ValueError("q_cdrag must have shape (npts,)")
    nvm = int(nvm)
    if nvm <= 0:
        raise ValueError("nvm must be positive")
    return jnp.broadcast_to(q_cdrag[:, None], (q_cdrag.shape[0], nvm))


def diffuco_raerod_explicit(*, u, v, q_cdrag, min_wind=0.1):
    """Compute diagnostic aerodynamic resistance ``raero``.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_raerod``, lines 3312-3353. The source computes
    ``speed = MAX(min_wind, wind(ji))`` after ``diffuco_main`` has assigned
    ``wind = SQRT(u*u+v*v)``, then ``raero = 1/(q_cdrag*speed)``.
    """

    u = jnp.asarray(u, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)
    q_cdrag = jnp.asarray(q_cdrag, dtype=jnp.float64)
    if u.ndim != 1:
        raise ValueError("u must have shape (npts,)")
    for name, value in (("v", v), ("q_cdrag", q_cdrag)):
        if value.ndim != 1 or value.shape[0] != u.shape[0]:
            raise ValueError(f"{name} must have shape (npts,)")
    speed = jnp.maximum(jnp.asarray(min_wind, dtype=jnp.float64), jnp.sqrt(u * u + v * v))
    return 1.0 / (q_cdrag * speed)


def diffuco_coupled_drag_boundary_explicit(*, u, v, q_cdrag, nvm: int, min_wind=0.1) -> DiffucoDragBoundaryResult:
    """Return coupled-branch drag fields before DIFFUCO beta kernels.

    Fortran provenance: ``src_sechiba/diffuco.f90::diffuco_main`` lines
    629-641 for the ``ldq_cdrag_from_gcm`` branch and
    ``diffuco_raerod`` lines 3312-3353 for ``raero``. This helper does not
    implement ``diffuco_aero``; callers should use it only when drag is supplied
    by the atmospheric driver/GCM branch.
    """

    q_cdrag = jnp.asarray(q_cdrag, dtype=jnp.float64)
    q_cdrag_pft = diffuco_coupled_q_cdrag_pft(q_cdrag, nvm=nvm)
    raero = diffuco_raerod_explicit(u=u, v=v, q_cdrag=q_cdrag, min_wind=min_wind)
    return DiffucoDragBoundaryResult(q_cdrag=q_cdrag, q_cdrag_pft=q_cdrag_pft, raero=raero)


def diffuco_drag_boundary_explicit(
    *,
    ldq_cdrag_from_gcm: bool,
    u,
    v,
    q_cdrag=None,
    nvm: int | None = None,
    zlev=None,
    z0h=None,
    z0m=None,
    roughheight=None,
    roughheight_pft=None,
    temp_sol=None,
    temp_sol_pft=None,
    temp_air=None,
    qsurf=None,
    qair=None,
    snow=None,
    ok_laidev=None,
    min_wind=0.1,
    snowcri=1.5,
    ok_snowfact=False,
    rough_dyn=False,
) -> DiffucoDragBoundaryResult:
    """Select the DIFFUCO drag branch using the explicit Fortran flag.

    Fortran provenance: ``src_sechiba/diffuco.f90::diffuco_main`` lines
    629-641. If ``ldq_cdrag_from_gcm`` is true, ORCHIDEE broadcasts the
    incoming grid-cell ``q_cdrag`` to every PFT and then calls
    ``diffuco_raerod``. Otherwise it calls ``diffuco_aero`` to compute
    ``q_cdrag``/``q_cdrag_pft`` locally, followed by ``diffuco_raerod``.
    """

    if bool(ldq_cdrag_from_gcm):
        if q_cdrag is None or nvm is None:
            raise ValueError("q_cdrag and nvm are required when ldq_cdrag_from_gcm is true")
        return diffuco_coupled_drag_boundary_explicit(u=u, v=v, q_cdrag=q_cdrag, nvm=nvm, min_wind=min_wind)

    required = {
        "zlev": zlev,
        "z0h": z0h,
        "z0m": z0m,
        "roughheight": roughheight,
        "roughheight_pft": roughheight_pft,
        "temp_sol": temp_sol,
        "temp_sol_pft": temp_sol_pft,
        "temp_air": temp_air,
        "qsurf": qsurf,
        "qair": qair,
        "snow": snow,
        "ok_laidev": ok_laidev,
    }
    missing = tuple(name for name, value in required.items() if value is None)
    if missing:
        raise ValueError(f"missing diffuco_aero inputs when ldq_cdrag_from_gcm is false: {missing}")
    aero = diffuco_aero_explicit(
        u=u,
        v=v,
        zlev=zlev,
        z0h=z0h,
        z0m=z0m,
        roughheight=roughheight,
        roughheight_pft=roughheight_pft,
        temp_sol=temp_sol,
        temp_sol_pft=temp_sol_pft,
        temp_air=temp_air,
        qsurf=qsurf,
        qair=qair,
        snow=snow,
        ok_laidev=ok_laidev,
        min_wind=min_wind,
        snowcri=snowcri,
        ok_snowfact=ok_snowfact,
        rough_dyn=rough_dyn,
    )
    return DiffucoDragBoundaryResult(q_cdrag=aero.q_cdrag, q_cdrag_pft=aero.q_cdrag_pft, raero=aero.raero)


def diffuco_aero_explicit(
    *,
    u,
    v,
    zlev,
    z0h,
    z0m,
    roughheight,
    roughheight_pft,
    temp_sol,
    temp_sol_pft,
    temp_air,
    qsurf,
    qair,
    snow,
    ok_laidev,
    min_wind=0.1,
    snowcri=1.5,
    ok_snowfact=False,
    rough_dyn=False,
    cp_air=CP_AIR,
    cte_grav=CTE_GRAV,
    retv=RET_V,
    rvtmp2=RV_TMP2,
    cepdu2=CEPDU2,
    ct_karman=CT_KARMAN,
    louis_cb=LOUIS_CB,
    louis_cc=LOUIS_CC,
    louis_cd=LOUIS_CD,
) -> DiffucoAeroResult:
    """Compute local surface drag using the Louis stability scheme.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_aero``, lines 940-1144. Constants are from
    ``src_parameters/constantes_var.f90`` lines 378-410 and 460-462.
    ``ok_snowfact`` follows ``src_parameters/constantes_soil.f90`` lines
    406-420; it is explicit here because its default depends on configured
    soil-freezing options.
    """

    u = jnp.asarray(u, dtype=jnp.float64)
    if u.ndim != 1:
        raise ValueError("u must have shape (npts,)")
    npts = u.shape[0]

    one_d_inputs = (
        ("v", v),
        ("zlev", zlev),
        ("z0h", z0h),
        ("z0m", z0m),
        ("roughheight", roughheight),
        ("temp_sol", temp_sol),
        ("temp_air", temp_air),
        ("qsurf", qsurf),
        ("qair", qair),
        ("snow", snow),
    )
    converted = {}
    for name, value in one_d_inputs:
        arr = jnp.asarray(value, dtype=jnp.float64)
        if arr.ndim != 1 or arr.shape[0] != npts:
            raise ValueError(f"{name} must have shape (npts,)")
        converted[name] = arr

    v = converted["v"]
    zlev = converted["zlev"]
    z0h = converted["z0h"]
    z0m = converted["z0m"]
    roughheight = converted["roughheight"]
    temp_sol = converted["temp_sol"]
    temp_air = converted["temp_air"]
    qsurf = converted["qsurf"]
    qair = converted["qair"]
    snow = converted["snow"]

    roughheight_pft = jnp.asarray(roughheight_pft, dtype=jnp.float64)
    temp_sol_pft = jnp.asarray(temp_sol_pft, dtype=jnp.float64)
    if roughheight_pft.ndim != 2 or roughheight_pft.shape[0] != npts:
        raise ValueError("roughheight_pft must have shape (npts, nvm)")
    pft_shape = roughheight_pft.shape
    if temp_sol_pft.shape != pft_shape:
        raise ValueError(f"temp_sol_pft must have shape {pft_shape}")
    ok_laidev = jnp.asarray(ok_laidev, dtype=bool)
    if ok_laidev.ndim != 1 or ok_laidev.shape[0] != pft_shape[1]:
        raise ValueError("ok_laidev must have shape (nvm,)")

    min_wind = jnp.asarray(min_wind, dtype=jnp.float64)
    snowcri = jnp.asarray(snowcri, dtype=jnp.float64)
    cp_air = jnp.asarray(cp_air, dtype=jnp.float64)
    cte_grav = jnp.asarray(cte_grav, dtype=jnp.float64)
    retv = jnp.asarray(retv, dtype=jnp.float64)
    rvtmp2 = jnp.asarray(rvtmp2, dtype=jnp.float64)
    cepdu2 = jnp.asarray(cepdu2, dtype=jnp.float64)
    ct_karman = jnp.asarray(ct_karman, dtype=jnp.float64)
    louis_cb = jnp.asarray(louis_cb, dtype=jnp.float64)
    louis_cc = jnp.asarray(louis_cc, dtype=jnp.float64)
    louis_cd = jnp.asarray(louis_cd, dtype=jnp.float64)

    speed_raw = jnp.sqrt(u * u + v * v)
    speed = speed_raw
    zg = zlev * cte_grav
    zdphi = zg / cp_air
    ztvd = (temp_air + zdphi / (1.0 + rvtmp2 * qair)) * (1.0 + retv * qair)
    ztvs = temp_sol * (1.0 + retv * qsurf)

    pft_active = ok_laidev[None, :]
    loop_pft = jnp.arange(pft_shape[1])[None, :] >= 1
    pft_compute = pft_active & loop_pft
    safe_temp_sol_pft = jnp.where(pft_compute, temp_sol_pft, temp_sol[:, None])
    safe_roughheight_pft = jnp.where(
        pft_compute,
        roughheight_pft,
        roughheight[:, None],
    )
    ztvs_pft = safe_temp_sol_pft * (1.0 + retv * qsurf[:, None])

    zdu2 = jnp.maximum(cepdu2, speed * speed)
    zri = zg * (ztvd - ztvs) / (zdu2 * ztvd)
    zri = jnp.maximum(jnp.minimum(zri, 5.0), -5.0)

    zri_pft_raw = zg[:, None] * (ztvd[:, None] - ztvs_pft) / (zdu2[:, None] * ztvd[:, None])
    zri_pft_raw = jnp.maximum(jnp.minimum(zri_pft_raw, 5.0), -5.0)
    zri_pft = jnp.where(pft_compute, zri_pft_raw, zri[:, None])

    snowfact = jnp.where((snow > snowcri) & bool(ok_snowfact) & (not bool(rough_dyn)), 10.0, 1.0)

    cd_neut = ct_karman**2 / (
        jnp.log((zlev + roughheight) / z0m) * jnp.log((zlev + roughheight) / z0h)
    )
    cd_neut_pft = ct_karman**2 / (
        jnp.log((zlev[:, None] + safe_roughheight_pft) / z0m[:, None])
        * jnp.log((zlev[:, None] + safe_roughheight_pft) / z0h[:, None])
    )

    grid_stable = zri >= 0.0
    zscf_stable = jnp.sqrt(1.0 + louis_cd * jnp.abs(zri))
    cd_tmp_stable = cd_neut / (1.0 + 3.0 * louis_cb * zri * zscf_stable)
    zscf_unstable = 1.0 / (
        1.0
        + 3.0
        * louis_cb
        * louis_cc
        * cd_neut
        * jnp.sqrt(jnp.abs(zri) * ((zlev + roughheight) / z0m / snowfact))
    )
    cd_tmp_unstable = cd_neut * (1.0 - 3.0 * louis_cb * zri * zscf_unstable)
    cd_tmp = jnp.where(grid_stable, cd_tmp_stable, cd_tmp_unstable)

    pft_stable = zri_pft >= 0.0
    zscf_pft_stable = jnp.sqrt(1.0 + louis_cd * jnp.abs(zri_pft))
    cd_tmp_pft_stable = cd_neut_pft / (1.0 + 3.0 * louis_cb * zri_pft * zscf_pft_stable)
    zscf_pft_unstable = 1.0 / (
        1.0
        + 3.0
        * louis_cb
        * louis_cc
        * cd_neut_pft
        * jnp.sqrt(
            jnp.abs(zri_pft)
            * ((zlev[:, None] + safe_roughheight_pft) / z0m[:, None] / snowfact[:, None])
        )
    )
    cd_tmp_pft_unstable = cd_neut_pft * (1.0 - 3.0 * louis_cb * zri_pft * zscf_pft_unstable)
    cd_tmp_pft_candidate = jnp.where(pft_stable, cd_tmp_pft_stable, cd_tmp_pft_unstable)
    cd_tmp_pft = jnp.where(pft_compute, cd_tmp_pft_candidate, cd_tmp[:, None])
    cd_neut_pft = jnp.where(loop_pft, cd_neut_pft, cd_neut[:, None])

    drag_floor = 1.0e-4 / jnp.maximum(speed, min_wind)
    q_cdrag = jnp.maximum(cd_tmp, drag_floor)
    q_cdrag_pft_candidate = jnp.maximum(cd_tmp_pft, drag_floor[:, None])
    q_cdrag_pft = jnp.where(pft_compute, q_cdrag_pft_candidate, q_cdrag[:, None])
    raero = diffuco_raerod_explicit(u=u, v=v, q_cdrag=q_cdrag, min_wind=min_wind)

    return DiffucoAeroResult(
        q_cdrag=q_cdrag,
        q_cdrag_pft=q_cdrag_pft,
        raero=raero,
        speed=speed,
        zri=zri,
        zri_pft=zri_pft,
        cd_neut=cd_neut,
        cd_neut_pft=cd_neut_pft,
        cd_tmp=cd_tmp,
        cd_tmp_pft=cd_tmp_pft,
        snowfact=snowfact,
    )


def diffuco_qsatt_explicit(*, temp_sol, pb):
    """Compute DIFFUCO surface saturated humidity ``qsatt``.

    Fortran provenance: ``src_sechiba/diffuco.f90::diffuco_main`` lines
    644-646 call ``qsatcalc(kjpindex,temp_sol,pb,qsatt)`` before snow, flood,
    interception, transpiration, and combination betas. The table-interpolation
    source kernel is shared with ENERBIL as ``qsat_moisture_qsatcalc``.
    """

    return qsat_moisture_qsatcalc(temp_sol, pb)


def diffuco_trans_co2_qsatt_explicit(*, t2m, pb):
    """Compute the saturated humidity used inside ``diffuco_trans_co2``.

    Fortran provenance: ``src_sechiba/diffuco.f90::diffuco_trans_co2`` lines
    2266-2276 call ``qsatcalc(kjpindex, t2m, pb, qsatt)`` before computing
    ``air_relhum`` and ``VPD``. This is intentionally separate from the
    surface ``qsatt`` computed in ``diffuco_main`` lines 644-646 from
    ``temp_sol`` for snow, flood, interception, and combination betas.
    """

    return qsat_moisture_qsatcalc(t2m, pb)


def diffuco_snow_beta_explicit(
    *,
    qair,
    qsatt,
    rau,
    u,
    v,
    q_cdrag,
    snow,
    frac_nobio,
    totfrac_nobio,
    snow_nobio,
    frac_snow_veg,
    frac_snow_nobio,
    dt_sechiba,
    min_wind=0.1,
) -> DiffucoSnowBetaResult:
    """Compute snow sublimation beta ``vbeta1``.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_snow``, lines 1177-1281. Vegetated snow sublimation is computed
    at lines 1216-1233; non-biological surface additions and their snow-mass
    limiter are computed at lines 1238-1274.
    """

    qair = jnp.asarray(qair, dtype=jnp.float64)
    qsatt = jnp.asarray(qsatt, dtype=jnp.float64)
    rau = jnp.asarray(rau, dtype=jnp.float64)
    u = jnp.asarray(u, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)
    q_cdrag = jnp.asarray(q_cdrag, dtype=jnp.float64)
    snow = jnp.asarray(snow, dtype=jnp.float64)
    frac_nobio = jnp.asarray(frac_nobio, dtype=jnp.float64)
    totfrac_nobio = jnp.asarray(totfrac_nobio, dtype=jnp.float64)
    snow_nobio = jnp.asarray(snow_nobio, dtype=jnp.float64)
    frac_snow_veg = jnp.asarray(frac_snow_veg, dtype=jnp.float64)
    frac_snow_nobio = jnp.asarray(frac_snow_nobio, dtype=jnp.float64)
    dt_sechiba = jnp.asarray(dt_sechiba, dtype=jnp.float64)

    if frac_nobio.ndim != 2:
        raise ValueError("frac_nobio must have shape (npts, nnobio)")
    shape = frac_nobio.shape
    for name, value in (("snow_nobio", snow_nobio), ("frac_snow_nobio", frac_snow_nobio)):
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    for name, value in (
        ("qair", qair),
        ("qsatt", qsatt),
        ("rau", rau),
        ("u", u),
        ("v", v),
        ("q_cdrag", q_cdrag),
        ("snow", snow),
        ("totfrac_nobio", totfrac_nobio),
        ("frac_snow_veg", frac_snow_veg),
    ):
        if value.ndim != 1 or value.shape[0] != shape[0]:
            raise ValueError(f"{name} must have shape (npts,)")

    speed = jnp.maximum(jnp.asarray(min_wind, dtype=jnp.float64), jnp.sqrt(u * u + v * v))
    vegetation_base = (1.0 - totfrac_nobio) * frac_snow_veg
    vegetation_subtest = (
        dt_sechiba
        * vegetation_base
        * speed
        * q_cdrag
        * rau
        * (qsatt - qair)
    )
    vegetation_limiter = vegetation_subtest > 0.0
    vegetation_zrapp = jnp.where(
        vegetation_limiter,
        snow / jnp.where(vegetation_limiter, vegetation_subtest, 1.0),
        0.0,
    )
    vegetation_vbeta1 = jnp.where(
        vegetation_limiter & (vegetation_zrapp < 1.0),
        vegetation_base * vegetation_zrapp,
        vegetation_base,
    )

    nobio_base = frac_nobio * frac_snow_nobio
    nobio_subtest = (
        dt_sechiba
        * nobio_base
        * speed[:, None]
        * q_cdrag[:, None]
        * rau[:, None]
        * (qsatt[:, None] - qair[:, None])
    )
    nobio_limiter = nobio_subtest > 0.0
    nobio_zrapp = jnp.where(
        nobio_limiter,
        snow_nobio / jnp.where(nobio_limiter, nobio_subtest, 1.0),
        0.0,
    )
    nobio_add = jnp.where(nobio_limiter & (nobio_zrapp < 1.0), nobio_base * nobio_zrapp, nobio_base)
    vbeta1 = vegetation_vbeta1 + jnp.sum(nobio_add, axis=1)
    return DiffucoSnowBetaResult(
        vbeta1=vbeta1,
        vegetation_base=vegetation_base,
        vegetation_subtest=vegetation_subtest,
        vegetation_zrapp=vegetation_zrapp,
        nobio_add=nobio_add,
        nobio_subtest=nobio_subtest,
        nobio_zrapp=nobio_zrapp,
    )


def diffuco_flood_beta_explicit(
    *,
    qair,
    qsatt,
    rau,
    u,
    v,
    q_cdrag,
    evapot,
    evapot_corr,
    flood_frac,
    flood_res,
    dt_sechiba,
    min_wind=0.1,
    min_sechiba=MIN_SECHIBA,
) -> DiffucoFloodBetaResult:
    """Compute floodplain beta ``vbeta5``.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_flood``, lines 1301-1356. The base floodplain beta is assigned
    from ``flood_frac * evapot_corr / evapot`` when ``evapot`` is positive at
    lines 1330-1334, then limited by ``flood_res`` at lines 1339-1349.
    """

    qair = jnp.asarray(qair, dtype=jnp.float64)
    qsatt = jnp.asarray(qsatt, dtype=jnp.float64)
    rau = jnp.asarray(rau, dtype=jnp.float64)
    u = jnp.asarray(u, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)
    q_cdrag = jnp.asarray(q_cdrag, dtype=jnp.float64)
    evapot = jnp.asarray(evapot, dtype=jnp.float64)
    evapot_corr = jnp.asarray(evapot_corr, dtype=jnp.float64)
    flood_frac = jnp.asarray(flood_frac, dtype=jnp.float64)
    flood_res = jnp.asarray(flood_res, dtype=jnp.float64)
    dt_sechiba = jnp.asarray(dt_sechiba, dtype=jnp.float64)
    if qair.ndim != 1:
        raise ValueError("qair must have shape (npts,)")
    for name, value in (
        ("qsatt", qsatt),
        ("rau", rau),
        ("u", u),
        ("v", v),
        ("q_cdrag", q_cdrag),
        ("evapot", evapot),
        ("evapot_corr", evapot_corr),
        ("flood_frac", flood_frac),
        ("flood_res", flood_res),
    ):
        if value.ndim != 1 or value.shape[0] != qair.shape[0]:
            raise ValueError(f"{name} must have shape (npts,)")

    base = jnp.where(evapot > min_sechiba, flood_frac * evapot_corr / evapot, flood_frac)
    speed = jnp.maximum(jnp.asarray(min_wind, dtype=jnp.float64), jnp.sqrt(u * u + v * v))
    subtest = dt_sechiba * base * speed * q_cdrag * rau * (qsatt - qair)
    limiter = subtest > 0.0
    zrapp = jnp.where(limiter, flood_res / jnp.where(limiter, subtest, 1.0), 0.0)
    vbeta5 = jnp.where(limiter & (zrapp < 1.0), base * zrapp, base)
    return DiffucoFloodBetaResult(vbeta5=vbeta5, base=base, subtest=subtest, zrapp=zrapp)


def diffuco_pft14_c3_beta_process_chain_explicit(
    *,
    pft_index: int,
    trans_co2_inputs: Mapping[str, object],
    qair,
    qsatt=None,
    temp_sol=None,
    pb=None,
    temp_air,
    rau,
    u,
    v,
    q_cdrag,
    q_cdrag_pft,
    humrel,
    veget,
    veget_max,
    lai,
    qsintveg,
    qsintmax,
    rstruct,
    ok_laidev,
    snow,
    frac_nobio,
    totfrac_nobio,
    snow_nobio,
    frac_snow_veg,
    frac_snow_nobio,
    evapot,
    evapot_corr,
    flood_frac,
    flood_res,
    tot_bare_soil,
    evap_bare_lim,
    vbeta3_background,
    dt_sechiba,
    min_wind=0.1,
    min_sechiba=MIN_SECHIBA,
) -> DiffucoPFT14C3BetaProcessChainResult:
    """Run local DIFFUCO beta processes through the PFT14 C3 closure.

    Fortran provenance: ``src_sechiba/diffuco.f90::diffuco_main`` calls
    ``diffuco_snow`` at lines 649-651, ``diffuco_flood`` at lines 654-655,
    ``diffuco_inter`` at lines 659-661, ``diffuco_trans_co2`` at lines
    668-671, ``diffuco_bare`` at lines 698-700, and ``diffuco_comb`` at lines
    709-710. This wrapper locally computes ``vbeta1``, ``vbeta5``,
    ``vbeta2``, and ``vbeta23`` while keeping all upstream state explicit.
    """

    if qsatt is None:
        if temp_sol is None or pb is None:
            raise ValueError("qsatt is required unless temp_sol and pb are supplied for qsatcalc")
        qsatt = diffuco_qsatt_explicit(temp_sol=temp_sol, pb=pb)

    snow_result = diffuco_snow_beta_explicit(
        qair=qair,
        qsatt=qsatt,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        snow=snow,
        frac_nobio=frac_nobio,
        totfrac_nobio=totfrac_nobio,
        snow_nobio=snow_nobio,
        frac_snow_veg=frac_snow_veg,
        frac_snow_nobio=frac_snow_nobio,
        dt_sechiba=dt_sechiba,
        min_wind=min_wind,
    )
    flood_result = diffuco_flood_beta_explicit(
        qair=qair,
        qsatt=qsatt,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        evapot=evapot,
        evapot_corr=evapot_corr,
        flood_frac=flood_frac,
        flood_res=flood_res,
        dt_sechiba=dt_sechiba,
        min_wind=min_wind,
        min_sechiba=min_sechiba,
    )
    chain = diffuco_pft14_c3_chain_with_inter_explicit(
        pft_index=pft_index,
        trans_co2_inputs=trans_co2_inputs,
        qair=qair,
        qsatt=qsatt,
        temp_air=temp_air,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        q_cdrag_pft=q_cdrag_pft,
        humrel=humrel,
        veget=veget,
        veget_max=veget_max,
        lai=lai,
        qsintveg=qsintveg,
        qsintmax=qsintmax,
        rstruct=rstruct,
        ok_laidev=ok_laidev,
        tot_bare_soil=tot_bare_soil,
        evap_bare_lim=evap_bare_lim,
        vbeta1=snow_result.vbeta1,
        vbeta3_background=vbeta3_background,
        dt_sechiba=dt_sechiba,
        min_wind=min_wind,
        min_sechiba=min_sechiba,
    )
    return DiffucoPFT14C3BetaProcessChainResult(
        snow=snow_result,
        flood=flood_result,
        inter=chain.inter,
        trans_co2=chain.trans_co2,
        closure=chain.closure,
    )


def diffuco_pft14_local_process_boundary_explicit(
    *,
    pft_index: int,
    trans_co2_inputs: Mapping[str, object],
    ldq_cdrag_from_gcm: bool,
    u,
    v,
    q_cdrag=None,
    nvm: int | None = None,
    zlev,
    z0h=None,
    z0m=None,
    roughheight=None,
    roughheight_pft=None,
    temp_sol,
    temp_sol_pft=None,
    temp_air,
    qsurf,
    qair,
    pb,
    rau,
    humrel,
    veget,
    veget_max,
    lai,
    qsintveg,
    qsintmax,
    rstruct,
    ok_laidev,
    snow,
    frac_nobio,
    totfrac_nobio,
    snow_nobio,
    frac_snow_veg,
    frac_snow_nobio,
    evapot,
    evapot_corr,
    flood_frac,
    flood_res,
    tot_bare_soil,
    evap_bare_lim,
    vbeta3_background,
    dt_sechiba,
    min_wind=0.1,
    snowcri=1.5,
    ok_snowfact=False,
    rough_dyn=False,
    min_sechiba=MIN_SECHIBA,
) -> DiffucoPFT14LocalProcessBoundaryResult:
    """Run the current local PFT14 DIFFUCO boundary in Fortran source order.

    Fortran provenance: ``src_sechiba/diffuco.f90::diffuco_main`` lines
    629-646 compute/broadcast drag and ``qsatt`` before the local beta and
    PFT14 C3 path implemented by ``diffuco_pft14_c3_beta_process_chain_explicit``
    from lines 649-710. This function keeps all upstream state explicit and
    does not infer missing vegetation, hydrology, or driver fields.
    """

    if bool(ldq_cdrag_from_gcm):
        if nvm is None:
            ok_laidev_arr = jnp.asarray(ok_laidev, dtype=bool)
            if ok_laidev_arr.ndim != 1:
                raise ValueError("nvm is required when ok_laidev is not a one-dimensional PFT mask")
            nvm = int(ok_laidev_arr.shape[0])
        drag = diffuco_drag_boundary_explicit(
            ldq_cdrag_from_gcm=True,
            u=u,
            v=v,
            q_cdrag=q_cdrag,
            nvm=nvm,
            min_wind=min_wind,
        )
    else:
        drag = diffuco_drag_boundary_explicit(
            ldq_cdrag_from_gcm=False,
            u=u,
            v=v,
            zlev=zlev,
            z0h=z0h,
            z0m=z0m,
            roughheight=roughheight,
            roughheight_pft=roughheight_pft,
            temp_sol=temp_sol,
            temp_sol_pft=temp_sol_pft,
            temp_air=temp_air,
            qsurf=qsurf,
            qair=qair,
            snow=snow,
            ok_laidev=ok_laidev,
            min_wind=min_wind,
            snowcri=snowcri,
            ok_snowfact=ok_snowfact,
            rough_dyn=rough_dyn,
        )

    qsatt = diffuco_qsatt_explicit(temp_sol=temp_sol, pb=pb)
    process_chain = diffuco_pft14_c3_beta_process_chain_explicit(
        pft_index=pft_index,
        trans_co2_inputs=trans_co2_inputs,
        qair=qair,
        qsatt=qsatt,
        temp_air=temp_air,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=drag.q_cdrag,
        q_cdrag_pft=drag.q_cdrag_pft,
        humrel=humrel,
        veget=veget,
        veget_max=veget_max,
        lai=lai,
        qsintveg=qsintveg,
        qsintmax=qsintmax,
        rstruct=rstruct,
        ok_laidev=ok_laidev,
        snow=snow,
        frac_nobio=frac_nobio,
        totfrac_nobio=totfrac_nobio,
        snow_nobio=snow_nobio,
        frac_snow_veg=frac_snow_veg,
        frac_snow_nobio=frac_snow_nobio,
        evapot=evapot,
        evapot_corr=evapot_corr,
        flood_frac=flood_frac,
        flood_res=flood_res,
        tot_bare_soil=tot_bare_soil,
        evap_bare_lim=evap_bare_lim,
        vbeta3_background=vbeta3_background,
        dt_sechiba=dt_sechiba,
        min_wind=min_wind,
        min_sechiba=min_sechiba,
    )
    return DiffucoPFT14LocalProcessBoundaryResult(
        drag=drag,
        qsatt=qsatt,
        process_chain=process_chain,
    )


DIFFUCO_PFT14_LOCAL_BOUNDARY_JIT_STATIC_ARGNAMES = (
    "pft_index",
    "ldq_cdrag_from_gcm",
    "nvm",
    "ok_snowfact",
    "rough_dyn",
)


_diffuco_pft14_local_process_boundary_jit = jit(
    diffuco_pft14_local_process_boundary_explicit,
    static_argnames=DIFFUCO_PFT14_LOCAL_BOUNDARY_JIT_STATIC_ARGNAMES,
)


def diffuco_pft14_local_enerbil_precall_explicit(
    *,
    pft_output_backgrounds: Mapping[str, object],
    passthrough: Mapping[str, object] | None = None,
    use_jit: bool = False,
    **boundary_kwargs,
) -> DiffucoPFT14LocalEnerbilPrecallResult:
    """Run local PFT14 DIFFUCO processes and assemble the ENERBIL payload.

    Fortran provenance: ``sechiba_main`` consumes the post-DIFFUCO fields at
    ``src_sechiba/sechiba.f90`` lines 1013-1019 after ``diffuco_main`` lines
    629-710. ``q_cdrag``, ``q_cdrag_pft``, and ``raero`` are locally generated
    boundary fields here unless the explicit GCM branch supplies ``q_cdrag``.
    """

    pft_index = int(boundary_kwargs["pft_index"])
    boundary_runner = _diffuco_pft14_local_process_boundary_jit if bool(use_jit) else diffuco_pft14_local_process_boundary_explicit
    boundary = boundary_runner(**boundary_kwargs)
    caller_passthrough = {} if passthrough is None else dict(passthrough)
    generated_passthrough = {
        "q_cdrag": boundary.drag.q_cdrag,
        "q_cdrag_pft": boundary.drag.q_cdrag_pft,
        "raero": boundary.drag.raero,
        "qsatt": boundary.qsatt,
    }
    overlap = set(caller_passthrough).intersection(generated_passthrough)
    if overlap:
        raise ValueError(f"pass-through fields overlap local DIFFUCO boundary fields: {tuple(sorted(overlap))}")
    payload = diffuco_pft14_enerbil_precall_payload(
        boundary.process_chain,
        pft_index=pft_index,
        pft_output_backgrounds=pft_output_backgrounds,
        passthrough={**caller_passthrough, **generated_passthrough},
    )
    return DiffucoPFT14LocalEnerbilPrecallResult(boundary=boundary, payload=payload)


def diffuco_bare_cwrr_beta(evap_bare_lim, vbeta2, vbeta3, veget_max):
    """Compute CWRR bare-soil evaporation beta outputs.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_bare``, CWRR branch lines 1670-1685; ``diffuco_main`` calls this
    subroutine at lines 698-700 after ``diffuco_trans_co2`` and
    ``diffuco_inter`` have supplied ``vbeta3`` and ``vbeta2``. The commented
    PFT-specific ``evap_bare_lim_pft`` path is not active in this source; both
    grid and PFT bare betas use scalar ``evap_bare_lim``.
    """

    evap_bare_lim = jnp.asarray(evap_bare_lim, dtype=jnp.float64)
    vbeta2 = jnp.asarray(vbeta2, dtype=jnp.float64)
    vbeta3 = jnp.asarray(vbeta3, dtype=jnp.float64)
    veget_max = jnp.asarray(veget_max, dtype=jnp.float64)
    if evap_bare_lim.ndim != 1:
        raise ValueError("evap_bare_lim must have shape (npts,)")
    if vbeta2.ndim != 2 or vbeta3.ndim != 2 or veget_max.ndim != 2:
        raise ValueError("vbeta2, vbeta3, and veget_max must have shape (npts, nvm)")
    if not (vbeta2.shape == vbeta3.shape == veget_max.shape):
        raise ValueError("vbeta2, vbeta3, and veget_max must share shape")
    if evap_bare_lim.shape[0] != vbeta2.shape[0]:
        raise ValueError("evap_bare_lim and PFT beta arrays must share npts")

    occupied = veget_max > 0.0
    vbeta4 = jnp.minimum(evap_bare_lim, 1.0 - jnp.sum(vbeta2 + vbeta3, axis=1))
    vbeta4_pft = jnp.where(
        occupied,
        jnp.minimum(evap_bare_lim[:, None], veget_max - (vbeta2 + vbeta3)),
        0.0,
    )
    return DiffucoBareCWRRBetaResult(vbeta4=vbeta4, vbeta4_pft=vbeta4_pft)


def diffuco_bare_non_cwrr_beta(
    *, u, v, q_cdrag, rsol, tot_bare_soil, nvm, min_wind=0.1, min_sechiba=MIN_SECHIBA
):
    """Compute the legacy non-CWRR bare-soil beta from ``diffuco_bare`` 1634-1663."""

    u = jnp.asarray(u, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)
    q_cdrag = jnp.asarray(q_cdrag, dtype=jnp.float64)
    rsol = jnp.asarray(rsol, dtype=jnp.float64)
    tot_bare_soil = jnp.asarray(tot_bare_soil, dtype=jnp.float64)
    if not (u.shape == v.shape == q_cdrag.shape == rsol.shape == tot_bare_soil.shape):
        raise ValueError("non-CWRR bare-soil inputs must share shape (npts,)")
    speed = jnp.maximum(min_wind, jnp.sqrt(u * u + v * v))
    active = tot_bare_soil >= min_sechiba
    vbeta4 = jnp.where(
        active,
        tot_bare_soil / (1.0 + speed * q_cdrag * rsol),
        0.0,
    )
    return DiffucoBareCWRRBetaResult(
        vbeta4=vbeta4,
        vbeta4_pft=jnp.zeros((u.shape[0], int(nvm)), dtype=jnp.float64),
    )


def diffuco_comb_final_beta_bundle_no_dew(vbeta2, vbeta3, vbeta4, *, qsatt=None, qair=None, min_sechiba=MIN_SECHIBA):
    """Guarded final ``diffuco_comb`` beta bundle for proven no-dew states.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_comb``, lines 3126-3254 modify ``vbeta1``, ``vbeta2``,
    ``vbeta3``, ``vbeta4``, ``vbeta4_pft``, ``humrel``, and ``vbeta_pft`` when
    ``qsatt < qair``. This helper therefore accepts the final bundle shortcut
    only when callers provide explicit proof that the dew branch is inactive.
    """

    if qsatt is None or qair is None:
        raise ValueError("qsatt and qair are required to prove diffuco_comb dew branch is inactive")
    qsatt = jnp.asarray(qsatt, dtype=jnp.float64)
    qair = jnp.asarray(qair, dtype=jnp.float64)
    if qsatt.ndim != 1 or qair.ndim != 1 or qsatt.shape != qair.shape:
        raise ValueError("qsatt and qair must share shape (npts,)")
    if bool(jnp.any(qsatt < qair)):
        raise ValueError("diffuco_comb dew branch is active; use a full source-backed dew implementation")
    return diffuco_comb_final_beta_bundle(vbeta2, vbeta3, vbeta4, min_sechiba=min_sechiba)


def diffuco_dew_vegetation_coefficient(lai, *, dew_veg_poly_coeff=DEW_VEG_POLY_COEFF, min_sechiba=MIN_SECHIBA):
    """Return the ``diffuco_comb`` warm-dew vegetation interception coefficient.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_comb``, lines 3197-3210. Default coefficients come from
    ``src_parameters/constantes_var.f90`` lines 812-813 and may be overridden
    by callers that audited a different ``DEW_VEG_POLY_COEFF`` run setting.
    """

    lai = jnp.asarray(lai, dtype=jnp.float64)
    coeff = jnp.asarray(dew_veg_poly_coeff, dtype=jnp.float64)
    if coeff.shape != (6,):
        raise ValueError("dew_veg_poly_coeff must contain six Fortran coefficients")
    polynomial = (
        coeff[5] * lai**5
        - coeff[4] * lai**4
        + coeff[3] * lai**3
        - coeff[2] * lai**2
        + coeff[1] * lai
        + coeff[0]
    )
    return jnp.where(lai > min_sechiba, jnp.where(lai > 1.5, polynomial, 1.0), 0.0)


def diffuco_comb_explicit(
    *,
    humrel,
    qair,
    temp_air,
    qsatt,
    veget,
    veget_max,
    lai,
    tot_bare_soil,
    vbeta1,
    vbeta2,
    vbeta3,
    vbeta4,
    vbeta4_pft,
    qsintmax,
    dew_veg_poly_coeff=DEW_VEG_POLY_COEFF,
    tp_00=TP_00,
    min_sechiba=MIN_SECHIBA,
):
    """Run the source-backed ``diffuco_comb`` beta combination algebra.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_comb``, lines 3070-3108 declare inputs/inouts; lines 3121-3122
    set ``valpha``; lines 3126-3149 classify warm dew versus freezing dew;
    lines 3157-3183 rewrite soil/snow betas; lines 3189-3228 rewrite
    interception; lines 3235-3254 zero transpiration and bare-soil
    interception under saturated air; lines 3256-3267 compute final ``vbeta``
    and clear tiny totals.

    ``qsatt`` is explicit here rather than recomputed, because the Fortran call
    obtains it through ``qsatcalc`` at line 3126 and callers may already have an
    audited ENERBIL/DIFFUCO saturation source.
    """

    humrel = jnp.asarray(humrel, dtype=jnp.float64)
    qair = jnp.asarray(qair, dtype=jnp.float64)
    temp_air = jnp.asarray(temp_air, dtype=jnp.float64)
    qsatt = jnp.asarray(qsatt, dtype=jnp.float64)
    veget = jnp.asarray(veget, dtype=jnp.float64)
    veget_max = jnp.asarray(veget_max, dtype=jnp.float64)
    lai = jnp.asarray(lai, dtype=jnp.float64)
    tot_bare_soil = jnp.asarray(tot_bare_soil, dtype=jnp.float64)
    vbeta1 = jnp.asarray(vbeta1, dtype=jnp.float64)
    vbeta2 = jnp.asarray(vbeta2, dtype=jnp.float64)
    vbeta3 = jnp.asarray(vbeta3, dtype=jnp.float64)
    vbeta4 = jnp.asarray(vbeta4, dtype=jnp.float64)
    vbeta4_pft = jnp.asarray(vbeta4_pft, dtype=jnp.float64)
    qsintmax = jnp.asarray(qsintmax, dtype=jnp.float64)

    if humrel.ndim != 2:
        raise ValueError("humrel must have shape (npts, nvm)")
    pft_shape = humrel.shape
    for name, value in (
        ("veget", veget),
        ("veget_max", veget_max),
        ("lai", lai),
        ("vbeta2", vbeta2),
        ("vbeta3", vbeta3),
        ("vbeta4_pft", vbeta4_pft),
        ("qsintmax", qsintmax),
    ):
        if value.shape != pft_shape:
            raise ValueError(f"{name} must have shape {pft_shape}")
    for name, value in (
        ("qair", qair),
        ("temp_air", temp_air),
        ("qsatt", qsatt),
        ("tot_bare_soil", tot_bare_soil),
        ("vbeta1", vbeta1),
        ("vbeta4", vbeta4),
    ):
        if value.ndim != 1 or value.shape[0] != pft_shape[0]:
            raise ValueError(f"{name} must have shape (npts,)")

    toveg = (qsatt < qair) & (temp_air > tp_00)
    tosnow = (qsatt < qair) & (temp_air <= tp_00)
    valpha = jnp.ones_like(qair)
    vbeta_pft = jnp.zeros_like(humrel)

    vbeta1_out = jnp.where(toveg, 0.0, jnp.where(tosnow, 1.0, vbeta1))
    vbeta4_out = jnp.where(toveg, tot_bare_soil, jnp.where(tosnow, 0.0, vbeta4))
    vbeta_pft = jnp.where(
        toveg[:, None],
        jnp.where(tot_bare_soil[:, None] > 0.0, vbeta4_pft, 0.0),
        vbeta_pft,
    )
    vbeta_pft = jnp.where(tosnow[:, None], veget_max, vbeta_pft)

    coeff_dew_veg = diffuco_dew_vegetation_coefficient(
        lai,
        dew_veg_poly_coeff=dew_veg_poly_coeff,
        min_sechiba=min_sechiba,
    )
    warm_dew_interception = jnp.where(
        qsintmax > min_sechiba,
        coeff_dew_veg * veget,
        0.0,
    )
    pft0_mask = jnp.arange(humrel.shape[1]) == 0
    warm_dew_bare = jnp.where(qsintmax[:, 0] > min_sechiba, coeff_dew_veg[:, 0] * tot_bare_soil, 0.0)
    warm_dew_interception = jnp.where(pft0_mask[None, :], warm_dew_bare[:, None], warm_dew_interception)
    vbeta2_out = jnp.where(toveg[:, None], warm_dew_interception, vbeta2)
    vbeta_pft = jnp.where(toveg[:, None], vbeta_pft + vbeta2_out, vbeta_pft)
    vbeta2_out = jnp.where(tosnow[:, None], 0.0, vbeta2_out)

    saturated_air = qsatt < qair
    vbeta3_out = jnp.where(saturated_air[:, None], 0.0, vbeta3)
    humrel_out = jnp.where(saturated_air[:, None], 0.0, humrel)
    vbeta2_out = jnp.where(pft0_mask[None, :] & saturated_air[:, None], 0.0, vbeta2_out)

    vbeta = vbeta4_out + jnp.sum(vbeta2_out, axis=1) + jnp.sum(vbeta3_out, axis=1)
    tiny = vbeta < min_sechiba
    vbeta = jnp.where(tiny, 0.0, vbeta)
    vbeta4_out = jnp.where(tiny, 0.0, vbeta4_out)
    vbeta2_out = jnp.where(tiny[:, None], 0.0, vbeta2_out)
    vbeta3_out = jnp.where(tiny[:, None], 0.0, vbeta3_out)

    return DiffucoCombResult(
        valpha=valpha,
        vbeta=vbeta,
        vbeta_pft=vbeta_pft,
        vbeta1=vbeta1_out,
        vbeta2=vbeta2_out,
        vbeta3=vbeta3_out,
        vbeta4=vbeta4_out,
        vbeta4_pft=vbeta4_pft,
        humrel=humrel_out,
        toveg=toveg,
        tosnow=tosnow,
    )


def diffuco_comb_final_beta_bundle(vbeta2, vbeta3, vbeta4, *, min_sechiba=MIN_SECHIBA):
    """Return the final ``diffuco_comb`` valpha/vbeta consistency bundle.

    Fortran provenance: ``src_sechiba/diffuco.f90``, subroutine
    ``diffuco_comb``, lines 3121-3122 set ``valpha(:)=un``; lines 3256-3267
    set ``vbeta = vbeta4 + SUM(vbeta2(:)) + SUM(vbeta3(:))`` and, when
    ``vbeta < min_sechiba``, reset ``vbeta``, ``vbeta4``, ``vbeta2``, and
    ``vbeta3`` to zero. ``min_sechiba`` is defined in
    ``src_parameters/constantes_var.f90`` line 174.
    """

    vbeta2 = jnp.asarray(vbeta2, dtype=jnp.float64)
    vbeta3 = jnp.asarray(vbeta3, dtype=jnp.float64)
    vbeta4 = jnp.asarray(vbeta4, dtype=jnp.float64)

    vbeta = vbeta4 + jnp.sum(vbeta2, axis=-1) + jnp.sum(vbeta3, axis=-1)
    is_tiny = vbeta < jnp.asarray(min_sechiba, dtype=jnp.float64)
    row_is_tiny = jnp.expand_dims(is_tiny, axis=-1)

    return DiffucoCombinedBetaResult(
        valpha=jnp.ones_like(vbeta),
        vbeta=jnp.where(is_tiny, 0.0, vbeta),
        vbeta2=jnp.where(row_is_tiny, 0.0, vbeta2),
        vbeta3=jnp.where(row_is_tiny, 0.0, vbeta3),
        vbeta4=jnp.where(is_tiny, 0.0, vbeta4),
    )


def diffuco_trans_source_routed(**kwargs):
    """Production entry point for ``diffuco_trans`` lines 1820-1910."""
    from .science_completion import diffuco_trans_source_routed as owner
    return owner(**kwargs)


def diffuco_main_source_routed(**kwargs):
    """Production entry point for completed ``diffuco_main`` routing."""
    from .science_completion import diffuco_main_source_routed as owner
    return owner(**kwargs)


def diffuco_finalize_restart_packet(**kwargs):
    """Production entry point for ``diffuco_finalize`` restart selection."""
    from .science_completion import diffuco_finalize_restart_packet as owner
    return owner(**kwargs)
