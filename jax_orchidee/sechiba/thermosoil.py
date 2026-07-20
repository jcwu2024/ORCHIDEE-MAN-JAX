"""Source-backed THERMOSOIL coefficient kernels.

This module starts with the explicit-input part of ``thermosoil_coef``. The
soil thermal properties ``pcapa`` and ``pkappa`` are inputs here because their
producer, ``thermosoil_getdiff*``, is a separate process path and must be
ported from source before it can be trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, NamedTuple

from jax import config, core, jit, tree_util

config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from jax_orchidee.sechiba.hydrol_thermosoil_completion import (  # noqa: E402
    read_refsocfile as _read_refsocfile,
)


PHIGEOTH = 0.057
MIN_SECHIBA = 1.0e-8
ZERO_CELSIUS = 273.15
MILLE = 1000.0
SO_CAPA_DRY = 1.80e6
SO_CAPA_WET = 3.03e6
SO_CAPA_DRY_ORG = 2.5e6
SO_COND_DRY = 0.40
SO_COND_WET = 1.89
COND_DRY_ORG = 0.05
COND_SOLID = 2.32
COND_SOLID_ORG = 0.25
THKQTZ = 7.7
THKICE = 2.2
THKW = 0.57
WATER_CAPA = 4.18e6
CAPA_ICE = 2.228e3
RHO_ICE = 920.0
RHO_WATER = 1000.0
LHF = 0.3336e6
POROS = 0.41
FR_DT = 2.0
SN_COND = 0.3
SN_DENS = 330.0
SN_CAPA = 2100.0 * SN_DENS
BRK_CAPA = 2.0e6
BRK_COND = 3.0
XCI = 2.106e3
SOILC_MAX = 130000.0
ZSNOWTHRMCOND1 = 0.02
ZSNOWTHRMCOND2 = 2.5e-6
ZSNOWTHRMCOND_AVAP = -0.06023
ZSNOWTHRMCOND_BVAP = -2.5425
ZSNOWTHRMCOND_CVAP = -289.99
XP00 = 1.0e5
PSNOWDZMIN = 1.0e-4

SMCMAX_USDA = (
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
QZ_USDA = (
    0.92,
    0.82,
    0.60,
    0.25,
    0.10,
    0.40,
    0.60,
    0.10,
    0.35,
    0.52,
    0.10,
    0.25,
)
SO_CAPA_DRY_NS_USDA = (
    1.47e6,
    1.41e6,
    1.34e6,
    1.27e6,
    1.21e6,
    1.21e6,
    1.18e6,
    1.32e6,
    1.23e6,
    1.18e6,
    1.15e6,
    1.09e6,
)

THERMOSOIL_COEF_SOIL_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_coef lines 1386-1427",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_coef lines 1471-1494",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_coef lines 1518-1628",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_coef lines 1722-1725",
)
THERMOSOIL_PROFILE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_profile lines 1761-1851",
)
THERMOSOIL_DIAGLEV_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_diaglev lines 3484-3513",
)
THERMOSOIL_ENERGY_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_energy lines 2420-2462",
)
THERMOSOIL_HUMLEV_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_humlev lines 2258-2399",
)
THERMOSOIL_COND_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_cond lines 1883-1972",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_cond_pft lines 2007-2127",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_cond_nopft lines 2129-2231",
)
THERMOSOIL_GETDIFF_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_getdiff lines 2566-2863",
)
THERMOSOIL_GETDIFF_THINSNOW_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_getdiff_thinsnow lines 3692-3788",
)
THERMOSOIL_GETDIFF_OLD_SNOW_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_getdiff_old_thermix_with_snow lines 2884-3031",
)
THERMOSOIL_COEF_SNOW_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_coef lines 1633-1721",
)
THERMOSOIL_COEF_NO_EXPLICIT_SNOW_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_coef lines 1499-1507",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_coef lines 1518-1628",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_coef lines 1633-1725",
)
THERMOSOIL_MAIN_FINAL_STATE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_main lines 1011-1032",
)
THERMOSOIL_RESTART_STATE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_initialize lines 668-684 and 735-762",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_finalize lines 1051-1136",
)
THERMOSOIL_ORGANIC_FRACTION_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_getdiff lines 2624-2689",
)
THERMOSOIL_VERTICAL_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_parameters/vertical_soil.f90 lines 427-454",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_var_init lines 1295-1299",
)
THERMOSOIL_CONSTANT_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_parameters/constantes_var.f90 lines 158-159,577",
    "fortran_source/ORCHIDEE/src_parameters/pft_parameters.f90 lines 666,3459",
)
THERMOSOIL_INITIAL_PTN_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_initialize lines 527-560",
)
THERMOSOIL_COLD_START_COEF_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_initialize lines 563-591",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_initialize lines 719-735",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_var_init lines 1239-1336",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_humlev lines 2258-2399",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_getdiff lines 2566-2863",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_coef lines 1386-1725",
)
THERMOSOIL_INITIALIZE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_initialize lines 327-726",
)
THERMOSOIL_EXTERNAL_IO_BOUNDARIES = (
    (
        "read_refsocfile",
        "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::read_refSOCfile lines 3227-3405",
        "structured NetCDF read and aggregate_p callback routed by read_refsocfile",
    ),
)


def thermosoil_read_refsocfile(**kwargs):
    """Route ``read_refSOCfile`` to its structured IO and aggregation owner."""

    return _read_refsocfile(**kwargs)


class ThermosoilCoefSoilResult(NamedTuple):
    """No explicit-snow surface coefficient subset of ``thermosoil_coef``."""

    cgrnd: jnp.ndarray
    dgrnd: jnp.ndarray
    soilcap: jnp.ndarray
    soilcap_pft: jnp.ndarray
    soilflx: jnp.ndarray
    soilflx_pft: jnp.ndarray
    soilcap_pft_nosnow: jnp.ndarray
    soilflx_pft_nosnow: jnp.ndarray
    zdz1: jnp.ndarray
    zdz2: jnp.ndarray
    cgrnd_soil: jnp.ndarray
    dgrnd_soil: jnp.ndarray
    zdz1_soil: jnp.ndarray
    zdz2_soil: jnp.ndarray
    provenance: tuple[str, ...] = THERMOSOIL_COEF_SOIL_PROVENANCE


class ThermosoilProfileResult(NamedTuple):
    """Updated soil-temperature profile and diagnostic levels."""

    ptn: jnp.ndarray
    stempdiag: jnp.ndarray
    provenance: tuple[str, ...] = (*THERMOSOIL_PROFILE_PROVENANCE, *THERMOSOIL_DIAGLEV_PROVENANCE)


class ThermosoilEnergyResult(NamedTuple):
    """Energy diagnostic state update from ``thermosoil_energy``."""

    surfheat_incr: jnp.ndarray
    coldcont_incr: jnp.ndarray
    ptn_beg: jnp.ndarray
    temp_sol_beg: jnp.ndarray
    provenance: tuple[str, ...] = THERMOSOIL_ENERGY_PROVENANCE


class ThermosoilHumlevResult(NamedTuple):
    """HYDROL moisture fields interpolated to THERMOSOIL levels."""

    mc_layt: jnp.ndarray
    mcl_layt: jnp.ndarray
    tmc_layt: jnp.ndarray
    mc_layt_pft: jnp.ndarray
    mcl_layt_pft: jnp.ndarray
    tmc_layt_pft: jnp.ndarray
    shum_ngrnd_perma: jnp.ndarray
    provenance: tuple[str, ...] = THERMOSOIL_HUMLEV_PROVENANCE


class ThermosoilCondResult(NamedTuple):
    """Thermal conductivity and diagnostics from ``thermosoil_cond*``."""

    cnd: jnp.ndarray
    ake: jnp.ndarray
    thksat: jnp.ndarray
    thkdry: jnp.ndarray
    thks: jnp.ndarray
    satratio: jnp.ndarray
    provenance: tuple[str, ...] = THERMOSOIL_COND_PROVENANCE


class ThermosoilGetdiffResult(NamedTuple):
    """THERMOSOIL thermal properties from active explicit-snow branch."""

    pcapa: jnp.ndarray
    pcapa_en: jnp.ndarray
    pkappa: jnp.ndarray
    profil_froz: jnp.ndarray
    pcappa_supp: jnp.ndarray
    pcapa_snow: jnp.ndarray
    pkappa_snow: jnp.ndarray
    poros_net: jnp.ndarray
    zx1: jnp.ndarray
    zx2: jnp.ndarray
    provenance: tuple[str, ...] = (*THERMOSOIL_GETDIFF_PROVENANCE, *THERMOSOIL_COND_PROVENANCE)


class ThermosoilThinSnowResult(NamedTuple):
    """First-layer thin-snow overwrite applied after ``thermosoil_getdiff``."""

    pcapa: jnp.ndarray
    pcapa_en: jnp.ndarray
    pkappa: jnp.ndarray
    profil_froz: jnp.ndarray
    snow_fraction: jnp.ndarray
    soil_fraction: jnp.ndarray
    provenance: tuple[str, ...] = THERMOSOIL_GETDIFF_THINSNOW_PROVENANCE


class ThermosoilWetnessLongResult(NamedTuple):
    """Long-term thawed-soil wetness state from ``thermosoil_wlupdate``."""

    hsdlong: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_wlupdate lines 3648-3672",
    )


class ThermosoilZimovHeatResult(NamedTuple):
    """Temperature and PFT-mean state after decomposition heating."""

    ptn: jnp.ndarray
    ptn_pftmean: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::add_heat_Zimov lines 3426-3463",
    )


class ThermosoilCoefExplicitSnowResult(NamedTuple):
    """``thermosoil_coef`` result with explicit-snow coefficient blend."""

    soil: ThermosoilCoefSoilResult
    lambda_snow: jnp.ndarray
    cgrnd_snow: jnp.ndarray
    dgrnd_snow: jnp.ndarray
    snowcap: jnp.ndarray
    snowflx: jnp.ndarray
    dz1_snow: jnp.ndarray
    dz2_snow: jnp.ndarray
    zdz1_snow: jnp.ndarray
    zdz2_snow: jnp.ndarray
    soilcap: jnp.ndarray
    soilflx: jnp.ndarray
    provenance: tuple[str, ...] = (*THERMOSOIL_COEF_SOIL_PROVENANCE, *THERMOSOIL_COEF_SNOW_PROVENANCE)


class ThermosoilFinalStateResult(NamedTuple):
    """Final THERMOSOIL state handed to snow, diagnostics, and slow processes."""

    ptn_pftmean: jnp.ndarray
    pkappa_pftmean: jnp.ndarray
    gtemp: jnp.ndarray
    ptnlev1: jnp.ndarray
    deephum_prof: jnp.ndarray
    deeptemp_prof: jnp.ndarray
    provenance: tuple[str, ...] = THERMOSOIL_MAIN_FINAL_STATE_PROVENANCE


class ThermosoilRestartState(NamedTuple):
    """THERMOSOIL recurrence state read from a SECHIBA restart file."""

    path: Path
    ptn: np.ndarray
    cgrnd: np.ndarray
    dgrnd: np.ndarray
    cgrnd_snow: np.ndarray
    dgrnd_snow: np.ndarray
    lambda_snow: np.ndarray
    gtemp: np.ndarray
    refsoc: np.ndarray
    shum_ngrnd_permalong: np.ndarray
    e_soil_lat: np.ndarray
    provenance: tuple[str, ...] = THERMOSOIL_RESTART_STATE_PROVENANCE

    def as_profile_payload(self) -> dict[str, np.ndarray]:
        """Expose fields consumed by THERMOSOIL profile kernels."""

        return {
            "ptn": self.ptn,
            "cgrnd": self.cgrnd,
            "dgrnd": self.dgrnd,
            "cgrnd_snow": self.cgrnd_snow,
            "dgrnd_snow": self.dgrnd_snow,
            "lambda_snow": self.lambda_snow,
        }


class ThermosoilRecurrenceState(NamedTuple):
    """THERMOSOIL state consumed before profile/energy on the next step."""

    ptn: jnp.ndarray
    cgrnd: jnp.ndarray
    dgrnd: jnp.ndarray
    cgrnd_snow: jnp.ndarray
    dgrnd_snow: jnp.ndarray
    lambda_snow: jnp.ndarray
    pcapa_en: jnp.ndarray
    temp_sol_beg: jnp.ndarray
    soilcap: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_main lines 905-913",
        "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_main lines 995-1024",
    )


class ThermosoilExplicitTransitionResult(NamedTuple):
    """Source-ordered THERMOSOIL step and explicit next-step recurrence."""

    step: "ThermosoilExplicitStepResult"
    next_state: ThermosoilRecurrenceState


class ThermosoilFirstStepModuleClosure(NamedTuple):
    """First-step THERMOSOIL execution and downstream boundary payloads."""

    module: ThermosoilExplicitStepResult | ThermosoilNoExplicitSnowStepResult | None
    boundary_payload: Mapping[str, object]
    downstream_payload: Mapping[str, object]
    missing_inputs: tuple[str, ...]
    provenance: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True only when THERMOSOIL main subset and downstream payload ran."""

        return not self.missing_inputs and self.module is not None


class ThermosoilExplicitStepResult(NamedTuple):
    """Composed explicit-input subset of ``thermosoil_main``."""

    humlev: ThermosoilHumlevResult
    profile: ThermosoilProfileResult
    energy: ThermosoilEnergyResult
    getdiff: ThermosoilGetdiffResult
    coef: ThermosoilCoefExplicitSnowResult
    final: ThermosoilFinalStateResult
    provenance: tuple[str, ...] = (
        *THERMOSOIL_HUMLEV_PROVENANCE,
        *THERMOSOIL_PROFILE_PROVENANCE,
        *THERMOSOIL_ENERGY_PROVENANCE,
        *THERMOSOIL_GETDIFF_PROVENANCE,
        *THERMOSOIL_COEF_SOIL_PROVENANCE,
        *THERMOSOIL_COEF_SNOW_PROVENANCE,
        *THERMOSOIL_MAIN_FINAL_STATE_PROVENANCE,
    )


class ThermosoilNoExplicitSnowStepResult(NamedTuple):
    """Composed ``thermosoil_main`` subset for ``ok_explicitsnow=false``."""

    humlev: ThermosoilHumlevResult
    profile: ThermosoilProfileResult
    energy: ThermosoilEnergyResult
    getdiff: ThermosoilGetdiffResult
    coef: ThermosoilCoefExplicitSnowResult
    final: ThermosoilFinalStateResult
    provenance: tuple[str, ...] = (
        *THERMOSOIL_HUMLEV_PROVENANCE,
        *THERMOSOIL_PROFILE_PROVENANCE,
        *THERMOSOIL_ENERGY_PROVENANCE,
        *THERMOSOIL_GETDIFF_OLD_SNOW_PROVENANCE,
        *THERMOSOIL_COEF_NO_EXPLICIT_SNOW_PROVENANCE,
        *THERMOSOIL_MAIN_FINAL_STATE_PROVENANCE,
    )


def _register_provenance_namedtuple_pytree(cls, child_fields: tuple[str, ...]) -> None:
    """Keep provenance metadata out of compiled JAX result leaves."""

    def flatten(value):
        return tuple(getattr(value, field) for field in child_fields), value.provenance

    def unflatten(provenance, children):
        return cls(**dict(zip(child_fields, children)), provenance=provenance)

    tree_util.register_pytree_node(cls, flatten, unflatten)


_register_provenance_namedtuple_pytree(
    ThermosoilCoefSoilResult,
    (
        "cgrnd",
        "dgrnd",
        "soilcap",
        "soilcap_pft",
        "soilflx",
        "soilflx_pft",
        "soilcap_pft_nosnow",
        "soilflx_pft_nosnow",
        "zdz1",
        "zdz2",
        "cgrnd_soil",
        "dgrnd_soil",
        "zdz1_soil",
        "zdz2_soil",
    ),
)
_register_provenance_namedtuple_pytree(
    ThermosoilProfileResult,
    ("ptn", "stempdiag"),
)
_register_provenance_namedtuple_pytree(
    ThermosoilEnergyResult,
    ("surfheat_incr", "coldcont_incr", "ptn_beg", "temp_sol_beg"),
)
_register_provenance_namedtuple_pytree(
    ThermosoilHumlevResult,
    ("mc_layt", "mcl_layt", "tmc_layt", "mc_layt_pft", "mcl_layt_pft", "tmc_layt_pft", "shum_ngrnd_perma"),
)
_register_provenance_namedtuple_pytree(
    ThermosoilCondResult,
    ("cnd", "ake", "thksat", "thkdry", "thks", "satratio"),
)
_register_provenance_namedtuple_pytree(
    ThermosoilGetdiffResult,
    ("pcapa", "pcapa_en", "pkappa", "profil_froz", "pcappa_supp", "pcapa_snow", "pkappa_snow", "poros_net", "zx1", "zx2"),
)
_register_provenance_namedtuple_pytree(
    ThermosoilCoefExplicitSnowResult,
    (
        "soil",
        "lambda_snow",
        "cgrnd_snow",
        "dgrnd_snow",
        "snowcap",
        "snowflx",
        "dz1_snow",
        "dz2_snow",
        "zdz1_snow",
        "zdz2_snow",
        "soilcap",
        "soilflx",
    ),
)
_register_provenance_namedtuple_pytree(
    ThermosoilFinalStateResult,
    ("ptn_pftmean", "pkappa_pftmean", "gtemp", "ptnlev1", "deephum_prof", "deeptemp_prof"),
)
_register_provenance_namedtuple_pytree(
    ThermosoilExplicitStepResult,
    ("humlev", "profile", "energy", "getdiff", "coef", "final"),
)
_register_provenance_namedtuple_pytree(
    ThermosoilNoExplicitSnowStepResult,
    ("humlev", "profile", "energy", "getdiff", "coef", "final"),
)


class ThermosoilColdStartCoefClosure(NamedTuple):
    """No-restart coefficient initialization from ``thermosoil_initialize``."""

    humlev: ThermosoilHumlevResult | None
    getdiff: ThermosoilGetdiffResult | None
    coef: ThermosoilCoefExplicitSnowResult | None
    ptn_pftmean: jnp.ndarray | None
    stempdiag: jnp.ndarray | None
    missing_inputs: tuple[str, ...]
    provenance: tuple[str, ...] = THERMOSOIL_COLD_START_COEF_PROVENANCE
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.missing_inputs and self.humlev is not None and self.getdiff is not None and self.coef is not None


class ThermosoilInitializeStaticInputs(NamedTuple):
    """File-backed fields kept outside the pure initialization transition."""

    ptn: np.ndarray | None
    refsoc: np.ndarray | None
    metadata: Mapping[str, np.ndarray]
    provenance: tuple[str, ...] = THERMOSOIL_INITIALIZE_PROVENANCE


class ThermosoilInitializeResult(NamedTuple):
    """PFT14-reachable state written by ``thermosoil_initialize``."""

    ptn: jnp.ndarray
    refsoc: jnp.ndarray
    ptn_beg: jnp.ndarray
    temp_sol_beg: jnp.ndarray
    shum_ngrnd_perma: jnp.ndarray
    shum_ngrnd_permalong: jnp.ndarray
    ptn_pftmean: jnp.ndarray
    stempdiag: jnp.ndarray
    profil_froz: jnp.ndarray
    pcappa_supp: jnp.ndarray
    e_soil_lat: jnp.ndarray | None
    gtemp: jnp.ndarray
    veget_max_bg: jnp.ndarray
    veget_mask_2d: jnp.ndarray
    veget_mask_real: jnp.ndarray
    cgrnd: jnp.ndarray
    dgrnd: jnp.ndarray
    cgrnd_snow: jnp.ndarray
    dgrnd_snow: jnp.ndarray
    lambda_snow: jnp.ndarray
    soilcap: jnp.ndarray
    soilcap_pft: jnp.ndarray
    soilflx: jnp.ndarray
    soilflx_pft: jnp.ndarray
    calculate_coef: bool
    brk_flag: int
    satsoil: bool
    ok_zimov: bool
    ok_shum_ngrnd_permalong: bool
    ptn_source: str
    refsoc_source: str
    provenance: tuple[str, ...] = THERMOSOIL_INITIALIZE_PROVENANCE


@dataclass(frozen=True)
class ThermosoilCoefCoverage:
    """Audit status for the currently ported ``thermosoil_coef`` subset."""

    ok: bool
    covered_fields: tuple[str, ...]
    missing_processes: tuple[str, ...]
    branch: str
    provenance: tuple[str, ...]
    notes: tuple[str, ...] = ()


def _as_float64(value):
    return jnp.asarray(value, dtype=jnp.float64)


def _validate_shapes(
    *,
    ptn,
    pcapa,
    pkappa,
    temp_sol_new,
    temp_sol_new_pft,
    veget_max,
    ok_laidev,
    dlt,
    dz1,
    veget_mask_2d,
):
    if ptn.ndim != 3:
        raise ValueError("ptn must have shape (npts, ngrnd, nvm)")
    if pcapa.shape != ptn.shape or pkappa.shape != ptn.shape:
        raise ValueError("pcapa and pkappa must match ptn shape")
    npts, ngrnd, nvm = ptn.shape
    if ngrnd < 2:
        raise ValueError("ngrnd must be at least 2")
    if temp_sol_new.shape != (npts,):
        raise ValueError("temp_sol_new must have shape (npts,)")
    if temp_sol_new_pft.shape != (npts, nvm):
        raise ValueError("temp_sol_new_pft must have shape (npts, nvm)")
    if veget_max.shape != (npts, nvm):
        raise ValueError("veget_max must have shape (npts, nvm)")
    if ok_laidev.shape != (nvm,):
        raise ValueError("ok_laidev must have shape (nvm,)")
    if dlt.shape != (ngrnd,):
        raise ValueError("dlt must have shape (ngrnd,)")
    if dz1.shape != (ngrnd - 1,):
        raise ValueError("dz1 must have shape (ngrnd - 1,)")
    if veget_mask_2d is not None and veget_mask_2d.shape != (npts, nvm):
        raise ValueError("veget_mask_2d must have shape (npts, nvm)")


def thermosoil_veget_max_bg(veget_max):
    """Build ``veget_max_bg`` with bare soil as residual PFT 1.

    Fortran provenance: ``thermosoil.f90::thermosoil_main`` lines 886-887
    and ``thermosoil_diaglev`` lines 3503-3504.
    """

    veget_max = _as_float64(veget_max)
    if veget_max.ndim != 2:
        raise ValueError("veget_max must have shape (npts, nvm)")
    npts, nvm = veget_max.shape
    if nvm < 1:
        raise ValueError("veget_max must contain at least one PFT")
    bg_first = jnp.maximum(1.0 - jnp.sum(veget_max[:, 1:], axis=1), 0.0)
    if nvm == 1:
        return bg_first[:, None]
    return jnp.concatenate((bg_first[:, None], veget_max[:, 1:]), axis=1)


THERMOSOIL_RESTART_FIELDS = (
    "ptn",
    "cgrnd",
    "dgrnd",
    "cgrnd_snow",
    "dgrnd_snow",
    "lambda_snow",
    "gtemp",
    "refSOC",
    "shum_ngrnd_prmlng",
    "e_soil_lat",
)

THERMOSOIL_INITIALIZE_RESTART_FIELDS = (
    "ptn",
    "refSOC",
    "shum_ngrnd_prmlng",
    "shum_ngrnd_perma",
    "e_soil_lat",
    "gtemp",
    "soilcap",
    "soilcap_pft",
    "soilflx",
    "soilflx_pft",
    "cgrnd",
    "dgrnd",
    "cgrnd_snow",
    "dgrnd_snow",
    "lambda_snow",
)


def _drop_restart_time(values: xr.DataArray) -> xr.DataArray:
    if "time" in values.dims:
        return values.isel(time=0)
    return values


def _flatten_thermosoil_restart_variable(values: xr.DataArray) -> np.ndarray:
    """Normalize THERMOSOIL restart layouts to model-point-major axes."""

    values = _drop_restart_time(values)
    dims = values.dims
    if dims == ("l_d", "z_f", "y", "x"):
        arr = np.asarray(values.transpose("y", "x", "z_f", "l_d").values)
        return arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
    if dims == ("l_d", "z_g", "y", "x"):
        arr = np.asarray(values.transpose("y", "x", "z_g", "l_d").values)
        return arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
    if dims == ("z_d", "y", "x"):
        arr = np.asarray(values.transpose("y", "x", "z_d").values)
        return arr.reshape((-1, arr.shape[-1]))
    if dims == ("y", "x"):
        return np.asarray(values.transpose("y", "x").values).reshape(-1)
    if "y" in dims and "x" in dims:
        other_dims = tuple(dim for dim in dims if dim not in ("y", "x"))
        arr = np.asarray(values.transpose("y", "x", *other_dims).values)
        return arr.reshape((-1, *arr.shape[2:]))
    return np.asarray(values.values)


def read_thermosoil_restart_state(path: str | Path) -> ThermosoilRestartState:
    """Read exact THERMOSOIL recurrence state from ``sechiba_start.nc``.

    Fortran provenance: ``thermosoil_initialize`` reads these fields before
    the first ``thermosoil_main`` call, and ``thermosoil_finalize`` writes the
    corresponding recurrence state back to restart. Missing variables raise
    instead of falling back to coefficient recomputation or tuned constants.
    """

    path = Path(path)
    with xr.open_dataset(path, decode_times=False) as ds:
        missing = [name for name in THERMOSOIL_RESTART_FIELDS if name not in ds.variables]
        if missing:
            raise KeyError(f"{path.name} is missing THERMOSOIL restart variables: {missing}")
        fields = {
            name: _flatten_thermosoil_restart_variable(ds[name])
            for name in THERMOSOIL_RESTART_FIELDS
        }
    return ThermosoilRestartState(
        path=path,
        ptn=fields["ptn"],
        cgrnd=fields["cgrnd"],
        dgrnd=fields["dgrnd"],
        cgrnd_snow=fields["cgrnd_snow"],
        dgrnd_snow=fields["dgrnd_snow"],
        lambda_snow=fields["lambda_snow"],
        gtemp=fields["gtemp"],
        refsoc=fields["refSOC"],
        shum_ngrnd_permalong=fields["shum_ngrnd_prmlng"],
        e_soil_lat=fields["e_soil_lat"],
    )


def read_thermosoil_initialize_restart_fields(path: str | Path) -> dict[str, np.ndarray]:
    """Read the optional restart fields consumed by ``thermosoil_initialize``.

    Unlike :func:`read_thermosoil_restart_state`, this boundary preserves the
    Fortran missing-variable decision: absent fields are omitted so the pure
    transition can apply the source default or recompute the complete
    coefficient group. Present fields are normalized to model-point-major
    axes; no values are synthesized here.

    Fortran provenance: ``thermosoil.f90::thermosoil_initialize`` lines
    531-596 and 654-714.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"THERMOSOIL restart file does not exist: {path}")
    with xr.open_dataset(path, decode_times=False) as ds:
        return {
            name: _flatten_thermosoil_restart_variable(ds[name])
            for name in THERMOSOIL_INITIALIZE_RESTART_FIELDS
            if name in ds.variables
        }


def read_thermosoil_initialize_static_inputs(
    config_path: str | Path,
    *,
    lalo,
    resolution_m,
    ngrnd: int,
    nvm: int,
    read_reftemp: bool,
    read_refsoc: bool,
) -> ThermosoilInitializeStaticInputs:
    """Resolve external THERMOSOIL initialization fields with driver readers.

    This is the only static-file boundary in the initialization API. It reuses
    the audited driver overlap readers and fails on absent configuration,
    geometry, files, or variables rather than replacing them with constants.

    Fortran provenance: ``thermosoil.f90::thermosoil_initialize`` lines
    536-567; ``thermosoil_read_reftempfile`` lines 3167-3214; and
    ``read_refSOCfile`` lines 3278-3392.
    """

    from jax_orchidee.driver.static import (  # Local import avoids a driver/SECHIBA import cycle.
        read_paper_thermosoil_refsoc_static_field,
        read_paper_thermosoil_reftemp_static_field,
    )

    metadata: dict[str, np.ndarray] = {}
    ptn = None
    refsoc = None
    if bool(read_reftemp):
        ptn, reftemp_meta = read_paper_thermosoil_reftemp_static_field(
            config_path,
            lalo=lalo,
            resolution_m=resolution_m,
            ngrnd=ngrnd,
            nvm=nvm,
        )
        metadata.update(reftemp_meta)
    if bool(read_refsoc):
        refsoc, refsoc_meta = read_paper_thermosoil_refsoc_static_field(
            config_path,
            lalo=lalo,
            resolution_m=resolution_m,
        )
        metadata.update(refsoc_meta)
    return ThermosoilInitializeStaticInputs(ptn=ptn, refsoc=refsoc, metadata=metadata)


def thermosoil_initial_ptn_constant(*, kjpindex, ngrnd, nvm, thermosoil_tpro=280.0, dtype=jnp.float64):
    """Build cold-start ``ptn`` when no restart/reftemp field is used.

    Fortran provenance: ``thermosoil_initialize`` reads restart ``ptn`` at
    lines 532-537. If it is absent and ``read_reftemp`` is false, lines
    548-560 call ``setvar_p(ptn, val_exp, 'THERMOSOIL_TPRO', 280.)``. This
    helper covers only that constant fallback; reftemp-file interpolation must
    be supplied by a separate source-backed reader.
    """

    kjpindex = int(kjpindex)
    ngrnd = int(ngrnd)
    nvm = int(nvm)
    if kjpindex < 1 or ngrnd < 1 or nvm < 1:
        raise ValueError("kjpindex, ngrnd, and nvm must be positive")
    return jnp.full((kjpindex, ngrnd, nvm), float(thermosoil_tpro), dtype=dtype)


def thermosoil_cold_start_coef_closure(
    *,
    ptn,
    moisture,
    temp_sol_new,
    temp_sol_new_pft,
    snowdz,
    snowrho,
    snowtemp,
    njsc,
    veget_max,
    pb,
    dlt,
    dz1,
    zlt,
    znt,
    dz5,
    dt_sechiba,
    frac_snow_veg,
    frac_snow_nobio,
    totfrac_nobio,
    refsoc=None,
    soilc_total=None,
    zx1=None,
    use_refSOC=True,
    use_soilc_tempdiff=True,
    ok_laidev=None,
    veget_mask_2d=None,
    satsoil=False,
    shum_ngrnd_permalong=None,
    ok_freeze_thermix=True,
    brk_flag=0,
) -> ThermosoilColdStartCoefClosure:
    """Compute no-restart THERMOSOIL coefficients when organic input is sourced.

    ``thermosoil_initialize`` recomputes ``soilcap*``, ``soilflx*``,
    ``cgrnd/dgrnd``, ``cgrnd_snow/dgrnd_snow``, and ``lambda_snow`` when the
    coefficient restart fields are absent. In the paper configuration
    ``USE_SOILC_TEMPDIFF`` and ``use_refSOC`` are true, so this helper refuses
    to fabricate the organic fraction: callers must provide ``refSOC``,
    ``soilc_total``, or an already audited ``zx1``.
    """

    ptn = _as_float64(ptn)
    if ptn.ndim != 3:
        raise ValueError("ptn must have shape (npts, ngrnd, nvm)")
    npts, ngrnd, nvm = ptn.shape
    missing: list[str] = []
    organic_inputs = sum(value is not None for value in (refsoc, soilc_total, zx1))
    if bool(use_soilc_tempdiff) and organic_inputs == 0:
        missing.append("refSOC_or_soilc_total_or_zx1_for_USE_SOILC_TEMPDIFF")
    if bool(use_refSOC) and refsoc is None and soilc_total is None and zx1 is None:
        missing.append("refSOC_non_restart_initialization")
    if organic_inputs > 1:
        raise ValueError("provide at most one of refsoc, soilc_total, or zx1")
    if missing:
        return ThermosoilColdStartCoefClosure(
            humlev=None,
            getdiff=None,
            coef=None,
            ptn_pftmean=None,
            stempdiag=None,
            missing_inputs=tuple(missing),
            notes=(
                "No coefficient defaults were computed because the active paper path requires sourced organic-carbon thermal fractions.",
            ),
        )

    if zx1 is None:
        if bool(use_soilc_tempdiff):
            zx1, _ = thermosoil_soilc_tempdiff_fraction(
                soilc_total=soilc_total,
                refsoc=refsoc,
                nvm=nvm,
            )
        else:
            zx1 = jnp.zeros_like(ptn)
    else:
        zx1 = _as_float64(zx1)
        if zx1.shape != ptn.shape:
            raise ValueError("zx1 must match ptn shape")

    humlev = thermosoil_humlev(
        shumdiag_perma=moisture.shumdiag_perma,
        mc_layh=moisture.mc_layh,
        mcl_layh=moisture.mcl_layh,
        tmc_layh=moisture.tmc_layh,
        mc_layh_pft=moisture.mc_layh_pft,
        mcl_layh_pft=moisture.mcl_layh_pft,
        tmc_layh_pft=moisture.tmc_layh_pft,
        znt=znt,
        zlt=zlt,
        dz5=dz5,
        veget_mask_2d=veget_mask_2d,
        satsoil=satsoil,
    )
    getdiff_humidity = humlev.shum_ngrnd_perma
    if shum_ngrnd_permalong is not None:
        getdiff_humidity = _as_float64(shum_ngrnd_permalong)
        if getdiff_humidity.shape != ptn.shape:
            raise ValueError("shum_ngrnd_permalong must match ptn shape")
    getdiff = thermosoil_getdiff_explicit(
        ptn=ptn,
        njsc=njsc,
        veget_max=veget_max,
        shum_ngrnd_permalong=getdiff_humidity,
        mc_layt=humlev.mc_layt,
        tmc_layt_pft=humlev.tmc_layt_pft,
        snowrho=snowrho,
        snowtemp=snowtemp,
        pb=pb,
        dlt=dlt,
        zx1=zx1,
        ok_laidev=ok_laidev,
        ok_freeze_thermix=ok_freeze_thermix,
        brk_flag=brk_flag,
        veget_mask_2d=veget_mask_2d,
        nslm=moisture.shumdiag_perma.shape[1],
    )
    thin_snow = thermosoil_getdiff_thinsnow(
        ptn=ptn,
        shum_ngrnd_permalong=getdiff_humidity,
        snowdz=snowdz,
        pcapa=getdiff.pcapa,
        pcapa_en=getdiff.pcapa_en,
        pkappa=getdiff.pkappa,
        profil_froz=getdiff.profil_froz,
        veget_mask_2d=(
            jnp.ones((npts, nvm), dtype=bool) if veget_mask_2d is None else veget_mask_2d
        ),
        zlt=zlt,
    )
    # thermosoil_coef lines 1499-1504 immediately rerun getdiff without the
    # commented thin-snow call, overwriting these transient var-init writes.
    getdiff = getdiff._replace(
        provenance=(*getdiff.provenance, *thin_snow.provenance),
    )
    ptn_pftmean = thermosoil_final_state(
        ptn=ptn,
        pkappa=getdiff.pkappa,
        veget_max=veget_max,
        shum_ngrnd_permalong=humlev.shum_ngrnd_perma,
    ).ptn_pftmean
    coef = thermosoil_coef_explicit_snow(
        ptn=ptn,
        ptn_pftmean=ptn_pftmean,
        pcapa=getdiff.pcapa,
        pkappa=getdiff.pkappa,
        pcapa_snow=getdiff.pcapa_snow,
        pkappa_snow=getdiff.pkappa_snow,
        snowdz=snowdz,
        snowtemp=snowtemp,
        temp_sol_new=temp_sol_new,
        temp_sol_new_pft=temp_sol_new_pft,
        veget_max=veget_max,
        ok_laidev=ok_laidev if ok_laidev is not None else jnp.zeros((nvm,), dtype=bool),
        dlt=dlt,
        dz1=dz1,
        zlt=zlt,
        dt_sechiba=dt_sechiba,
        lambda_thermal=float(_as_float64(znt)[0] * _as_float64(dz1)[0]),
        frac_snow_veg=frac_snow_veg,
        frac_snow_nobio=frac_snow_nobio,
        totfrac_nobio=totfrac_nobio,
        veget_mask_2d=veget_mask_2d,
    )
    stempdiag = thermosoil_diaglev_from_ptn(ptn=ptn, veget_max=veget_max, nslm=moisture.shumdiag_perma.shape[1])
    return ThermosoilColdStartCoefClosure(
        humlev=humlev,
        getdiff=getdiff,
        coef=coef,
        ptn_pftmean=ptn_pftmean,
        stempdiag=stempdiag,
        missing_inputs=(),
        notes=(
            "Computes the no-restart coefficient path only after organic-carbon thermal input is explicitly supplied.",
        ),
    )


def thermosoil_rotation_update(*, matrix_rot, old_veget_max, ptn, cgrnd, dgrnd, min_sechiba=1.0e-8):
    """Rotate thermal PFT stores from ``thermosoil_rotation_update`` 3790-3858."""
    rot = jnp.asarray(matrix_rot)
    old = jnp.asarray(old_veget_max)
    ptn_out, cgrnd_out, dgrnd_out = jnp.asarray(ptn), jnp.asarray(cgrnd), jnp.asarray(dgrnd)
    if rot.ndim != 2 or rot.shape[0] != rot.shape[1] or old.shape != (rot.shape[0],):
        raise ValueError("rotation matrix and old_veget_max must share PFT dimension")
    max_new = old - jnp.sum(old[:, None] * rot, axis=1) + jnp.sum(old[:, None] * rot, axis=0)
    for target in range(rot.shape[1]):
        incoming = rot[:, target]
        if bool(jnp.sum(incoming) > min_sechiba):
            retained = old[target] * (1.0 - jnp.sum(rot[target, :]))
            weights = retained * jnp.eye(rot.shape[0], dtype=rot.dtype)[target] + old * incoming
            ptn_out = ptn_out.at[..., target].set(jnp.sum(ptn[..., :] * weights, axis=-1) / max_new[target])
            cgrnd_out = cgrnd_out.at[..., target].set(jnp.sum(cgrnd[..., :] * weights, axis=-1) / max_new[target])
            dgrnd_out = dgrnd_out.at[..., target].set(jnp.sum(dgrnd[..., :] * weights, axis=-1) / max_new[target])
    return ptn_out, cgrnd_out, dgrnd_out


def _thermosoil_restart_value(restart_fields, name, shape, *, val_exp):
    value = restart_fields.get(name)
    if value is None:
        return None
    value = np.asarray(value)
    if value.shape != shape:
        raise ValueError(f"restart {name} must have shape {shape}, got {value.shape}")
    missing = value == float(val_exp)
    if np.any(missing) and not np.all(missing):
        raise ValueError(f"restart {name} mixes val_exp and initialized values")
    if np.all(missing):
        return None
    if not np.all(np.isfinite(value)):
        raise ValueError(f"restart {name} contains non-finite values")
    return _as_float64(value)


def thermosoil_initialize(
    *,
    ngrnd: int,
    nscm: int,
    temp_sol_new,
    veget_max,
    coefficient_inputs: Mapping[str, object],
    restart_fields: Mapping[str, object] | None = None,
    external_ptn=None,
    external_refsoc=None,
    read_reftemp=False,
    thermosoil_tpro=280.0,
    use_refSOC=True,
    use_soilc_tempdiff=True,
    use_toporganiclayer_tempdiff=False,
    ok_freeze_thermix=True,
    ok_pc=True,
    ok_leak=False,
    ok_Ecorr=False,
    ok_wetdiaglong=False,
    ok_zimov=False,
    satsoil=False,
    bedrock_flag=0,
    val_exp=999999.0,
) -> ThermosoilInitializeResult:
    """Execute the PFT14-reachable ``thermosoil_initialize`` transition.

    File access is intentionally excluded: ``restart_fields`` must come from
    :func:`read_thermosoil_initialize_restart_fields`, while reference
    temperature and reference SOC must come from
    :func:`read_thermosoil_initialize_static_inputs`. The transition follows
    Fortran's restart/default ordering and recomputes the *whole* coefficient
    group when any one group member is absent.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90``
    subroutine ``thermosoil_initialize`` lines 327-726.
    """

    ngrnd = int(ngrnd)
    nscm = int(nscm)
    if ngrnd < 2:
        raise ValueError("ngrnd must be at least 2")
    if nscm != 12:
        raise ValueError("only the source-backed USDA nscm=12 thermal tables are implemented")
    restart_fields = {} if restart_fields is None else dict(restart_fields)
    veget_max = _as_float64(veget_max)
    temp_sol_new = _as_float64(temp_sol_new)
    if veget_max.ndim != 2:
        raise ValueError("veget_max must have shape (npts, nvm)")
    npts, nvm = veget_max.shape
    if temp_sol_new.shape != (npts,):
        raise ValueError("temp_sol_new must have shape (npts,)")

    ptn = _thermosoil_restart_value(restart_fields, "ptn", (npts, ngrnd, nvm), val_exp=val_exp)
    if ptn is None:
        if bool(read_reftemp):
            if external_ptn is None:
                raise ValueError("READ_REFTEMP requires ptn from the driver static reader")
            ptn = _as_float64(external_ptn)
            if ptn.shape != (npts, ngrnd, nvm):
                raise ValueError(f"external reference-temperature ptn must have shape {(npts, ngrnd, nvm)}")
            ptn_source = "external_reference_temperature"
        else:
            if external_ptn is not None:
                raise ValueError("external_ptn was supplied while READ_REFTEMP is false")
            ptn = thermosoil_initial_ptn_constant(
                kjpindex=npts,
                ngrnd=ngrnd,
                nvm=nvm,
                thermosoil_tpro=thermosoil_tpro,
            )
            ptn_source = "THERMOSOIL_TPRO"
    else:
        ptn_source = "restart"

    refsoc = _thermosoil_restart_value(restart_fields, "refSOC", (npts, ngrnd), val_exp=val_exp)
    if refsoc is None:
        if bool(use_soilc_tempdiff) and bool(use_refSOC):
            if external_refsoc is None:
                raise ValueError("use_refSOC requires refSOC from restart or the driver static reader")
            refsoc = _as_float64(external_refsoc)
            if refsoc.shape != (npts, ngrnd):
                raise ValueError(f"external refSOC must have shape {(npts, ngrnd)}")
            refsoc_source = "external_refSOC"
        else:
            if external_refsoc is not None:
                raise ValueError("external_refsoc was supplied while the refSOC path is inactive")
            refsoc = jnp.zeros((npts, ngrnd), dtype=jnp.float64)
            refsoc_source = "inactive_zero_allocation"
    else:
        refsoc_source = "restart"

    bedrock_flag = int(bedrock_flag)
    if bedrock_flag not in (0, 1):
        raise ValueError("BEDROCK_FLAG must be 0 or 1")
    long_humidity_enabled = bool(ok_wetdiaglong) or (
        (bool(ok_freeze_thermix) and bool(ok_pc)) or bool(ok_leak)
    )
    shum_perma = _thermosoil_restart_value(
        restart_fields,
        "shum_ngrnd_perma",
        (npts, ngrnd, nvm),
        val_exp=val_exp,
    )
    if shum_perma is None:
        shum_perma = jnp.ones((npts, ngrnd, nvm), dtype=jnp.float64)
    if long_humidity_enabled:
        shum_long = _thermosoil_restart_value(
            restart_fields,
            "shum_ngrnd_prmlng",
            (npts, ngrnd, nvm),
            val_exp=val_exp,
        )
        if shum_long is None:
            shum_long = jnp.ones((npts, ngrnd, nvm), dtype=jnp.float64)
    else:
        shum_long = jnp.zeros((npts, ngrnd, nvm), dtype=jnp.float64)

    coefficient_inputs = dict(coefficient_inputs)
    required = {
        "moisture",
        "temp_sol_new_pft",
        "snowdz",
        "snowrho",
        "snowtemp",
        "njsc",
        "pb",
        "dlt",
        "dz1",
        "zlt",
        "znt",
        "dz5",
        "dt_sechiba",
        "frac_snow_veg",
        "frac_snow_nobio",
        "totfrac_nobio",
    }
    missing = sorted(required.difference(coefficient_inputs))
    if missing:
        raise KeyError(f"coefficient_inputs is missing source inputs: {missing}")
    if bool(use_soilc_tempdiff) and bool(use_toporganiclayer_tempdiff):
        use_toporganiclayer_tempdiff = False
    if bool(use_toporganiclayer_tempdiff):
        if "zx1" not in coefficient_inputs:
            if "organic_layer_thick" not in coefficient_inputs:
                raise ValueError("USE_TOPORGANICLAYER_TEMPDIFF requires organic_layer_thick or explicit zx1")
            coefficient_inputs["zx1"], _ = thermosoil_toporganiclayer_fraction(
                organic_layer_thick=coefficient_inputs.pop("organic_layer_thick"),
                zlt=coefficient_inputs["zlt"],
                nvm=nvm,
            )
    else:
        coefficient_inputs.pop("organic_layer_thick", None)

    coefficient_inputs.pop("refsoc", None)
    coefficient_inputs.pop("use_refSOC", None)
    coefficient_inputs.pop("use_soilc_tempdiff", None)
    initialized = thermosoil_cold_start_coef_closure(
        ptn=ptn,
        temp_sol_new=temp_sol_new,
        veget_max=veget_max,
        refsoc=refsoc if bool(use_soilc_tempdiff) and bool(use_refSOC) else None,
        use_refSOC=use_refSOC,
        use_soilc_tempdiff=use_soilc_tempdiff,
        satsoil=satsoil,
        shum_ngrnd_permalong=shum_long,
        ok_freeze_thermix=ok_freeze_thermix,
        brk_flag=bedrock_flag,
        **coefficient_inputs,
    )
    if not initialized.ok or initialized.getdiff is None or initialized.coef is None or initialized.stempdiag is None:
        raise ValueError(f"thermosoil coefficient initialization is incomplete: {initialized.missing_inputs}")

    snowdz = np.asarray(coefficient_inputs["snowdz"])
    if snowdz.ndim != 2 or snowdz.shape[0] != npts:
        raise ValueError("snowdz must have shape (npts, nsnow)")
    nsnow = snowdz.shape[1]
    coefficient_shapes = {
        "soilcap": (npts,),
        "soilcap_pft": (npts, nvm),
        "soilflx": (npts,),
        "soilflx_pft": (npts, nvm),
        "cgrnd": (npts, ngrnd - 1, nvm),
        "dgrnd": (npts, ngrnd - 1, nvm),
        "cgrnd_snow": (npts, nsnow),
        "dgrnd_snow": (npts, nsnow),
        "lambda_snow": (npts,),
    }
    restart_coefficients = {
        name: _thermosoil_restart_value(restart_fields, name, shape, val_exp=val_exp)
        for name, shape in coefficient_shapes.items()
    }
    calculate_coef = any(value is None for value in restart_coefficients.values())
    computed = initialized.coef
    coefficients = {
        "soilcap": computed.soilcap,
        "soilcap_pft": computed.soil.soilcap_pft,
        "soilflx": computed.soilflx,
        "soilflx_pft": computed.soil.soilflx_pft,
        "cgrnd": computed.soil.cgrnd,
        "dgrnd": computed.soil.dgrnd,
        "cgrnd_snow": computed.cgrnd_snow,
        "dgrnd_snow": computed.dgrnd_snow,
        "lambda_snow": computed.lambda_snow,
    }
    if not calculate_coef:
        coefficients = restart_coefficients

    final_getdiff = initialized.getdiff
    if not calculate_coef:
        thin_snow = thermosoil_getdiff_thinsnow(
            ptn=ptn,
            shum_ngrnd_permalong=shum_long,
            snowdz=coefficient_inputs["snowdz"],
            pcapa=final_getdiff.pcapa,
            pcapa_en=final_getdiff.pcapa_en,
            pkappa=final_getdiff.pkappa,
            profil_froz=final_getdiff.profil_froz,
            veget_mask_2d=jnp.ones((npts, nvm), dtype=bool),
            zlt=coefficient_inputs["zlt"],
        )
        final_getdiff = final_getdiff._replace(
            pcapa=thin_snow.pcapa,
            pcapa_en=thin_snow.pcapa_en,
            pkappa=thin_snow.pkappa,
            profil_froz=thin_snow.profil_froz,
            provenance=(*final_getdiff.provenance, *thin_snow.provenance),
        )

    veget_max_bg = thermosoil_veget_max_bg(veget_max)
    veget_mask_2d = jnp.ones((npts, nvm), dtype=bool)
    gtemp = _thermosoil_restart_value(restart_fields, "gtemp", (npts,), val_exp=val_exp)
    if gtemp is None:
        gtemp = jnp.zeros((npts,), dtype=jnp.float64)
    e_soil_lat = None
    if bool(ok_Ecorr):
        e_soil_lat = _thermosoil_restart_value(restart_fields, "e_soil_lat", (npts, nvm), val_exp=val_exp)
        if e_soil_lat is None:
            e_soil_lat = jnp.zeros((npts, nvm), dtype=jnp.float64)

    return ThermosoilInitializeResult(
        ptn=ptn,
        refsoc=refsoc,
        ptn_beg=ptn,
        temp_sol_beg=temp_sol_new,
        shum_ngrnd_perma=initialized.humlev.shum_ngrnd_perma,
        shum_ngrnd_permalong=shum_long,
        ptn_pftmean=initialized.ptn_pftmean,
        stempdiag=initialized.stempdiag,
        profil_froz=final_getdiff.profil_froz,
        pcappa_supp=final_getdiff.pcappa_supp,
        e_soil_lat=e_soil_lat,
        gtemp=gtemp,
        veget_max_bg=veget_max_bg,
        veget_mask_2d=veget_mask_2d,
        veget_mask_real=veget_mask_2d.astype(jnp.float64),
        cgrnd=coefficients["cgrnd"],
        dgrnd=coefficients["dgrnd"],
        cgrnd_snow=coefficients["cgrnd_snow"],
        dgrnd_snow=coefficients["dgrnd_snow"],
        lambda_snow=coefficients["lambda_snow"],
        soilcap=coefficients["soilcap"],
        soilcap_pft=coefficients["soilcap_pft"],
        soilflx=coefficients["soilflx"],
        soilflx_pft=coefficients["soilflx_pft"],
        calculate_coef=calculate_coef,
        brk_flag=bedrock_flag,
        satsoil=bool(satsoil),
        ok_zimov=bool(ok_zimov),
        ok_shum_ngrnd_permalong=long_humidity_enabled,
        ptn_source=ptn_source,
        refsoc_source=refsoc_source,
    )


def thermosoil_finalize_restart_packet(
    *,
    ptn,
    refsoc,
    shum_ngrnd_perma,
    cgrnd,
    dgrnd,
    gtemp,
    soilcap,
    soilcap_pft,
    soilflx,
    soilflx_pft,
    cgrnd_snow,
    dgrnd_snow,
    lambda_snow,
    ok_shum_ngrnd_permalong=True,
    shum_ngrnd_permalong=None,
    ok_Ecorr=False,
    e_soil_lat=None,
) -> dict[str, jnp.ndarray]:
    """Build the exact scientific restart payload written by THERMOSOIL.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90``,
    subroutine ``thermosoil_finalize``, lines 1065-1142. This pure boundary
    owns field selection and shapes; ``restput_p`` scatter/NetCDF persistence
    remains the driver restart writer's responsibility.
    """

    ptn = _as_float64(ptn)
    if ptn.ndim != 3:
        raise ValueError("ptn must have shape (npts, ngrnd, nvm)")
    npts, ngrnd, nvm = ptn.shape
    refsoc = _as_float64(refsoc)
    shum_perma = _as_float64(shum_ngrnd_perma)
    cgrnd = _as_float64(cgrnd)
    dgrnd = _as_float64(dgrnd)
    gtemp = _as_float64(gtemp)
    soilcap = _as_float64(soilcap)
    soilcap_pft = _as_float64(soilcap_pft)
    soilflx = _as_float64(soilflx)
    soilflx_pft = _as_float64(soilflx_pft)
    cgrnd_snow = _as_float64(cgrnd_snow)
    dgrnd_snow = _as_float64(dgrnd_snow)
    lambda_snow = _as_float64(lambda_snow)
    expected = {
        "refSOC": (refsoc, (npts, ngrnd)),
        "shum_ngrnd_perma": (shum_perma, (npts, ngrnd, nvm)),
        "cgrnd": (cgrnd, (npts, ngrnd - 1, nvm)),
        "dgrnd": (dgrnd, (npts, ngrnd - 1, nvm)),
        "gtemp": (gtemp, (npts,)),
        "soilcap": (soilcap, (npts,)),
        "soilcap_pft": (soilcap_pft, (npts, nvm)),
        "soilflx": (soilflx, (npts,)),
        "soilflx_pft": (soilflx_pft, (npts, nvm)),
        "lambda_snow": (lambda_snow, (npts,)),
    }
    for name, (value, shape) in expected.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if cgrnd_snow.ndim != 2 or cgrnd_snow.shape[0] != npts:
        raise ValueError("cgrnd_snow must have shape (npts, nsnow)")
    if dgrnd_snow.shape != cgrnd_snow.shape:
        raise ValueError("dgrnd_snow must match cgrnd_snow")

    packet = {
        "ptn": ptn,
        "refSOC": refsoc,
        "shum_ngrnd_perma": shum_perma,
        "cgrnd": cgrnd,
        "dgrnd": dgrnd,
        "gtemp": gtemp,
        "soilcap": soilcap,
        "soilcap_pft": soilcap_pft,
        "soilflx": soilflx,
        "soilflx_pft": soilflx_pft,
        "cgrnd_snow": cgrnd_snow,
        "dgrnd_snow": dgrnd_snow,
        "lambda_snow": lambda_snow,
    }
    if bool(ok_shum_ngrnd_permalong):
        if shum_ngrnd_permalong is None:
            raise ValueError("active OK_WETDIAGLONG requires shum_ngrnd_permalong")
        shum_long = _as_float64(shum_ngrnd_permalong)
        if shum_long.shape != (npts, ngrnd, nvm):
            raise ValueError("shum_ngrnd_permalong must match ptn")
        packet["shum_ngrnd_prmlng"] = shum_long
    elif shum_ngrnd_permalong is not None:
        raise ValueError("shum_ngrnd_permalong supplied while its restart branch is inactive")
    if bool(ok_Ecorr):
        if e_soil_lat is None:
            raise ValueError("active ok_Ecorr requires e_soil_lat")
        e_soil_lat = _as_float64(e_soil_lat)
        if e_soil_lat.shape != (npts, nvm):
            raise ValueError("e_soil_lat must have shape (npts, nvm)")
        packet["e_soil_lat"] = e_soil_lat
    elif e_soil_lat is not None:
        raise ValueError("e_soil_lat supplied while ok_Ecorr is false")
    return packet


def thermosoil_diaglev_from_ptn(*, ptn, veget_max, nslm=None):
    """Aggregate PFT soil temperatures onto the diagnostic profile.

    Fortran provenance: ``thermosoil.f90::thermosoil_diaglev`` lines
    3503-3510. The active CWRR branch uses diagnostic levels already aligned
    with the first ``nslm`` thermal levels.
    """

    ptn = _as_float64(ptn)
    if ptn.ndim != 3:
        raise ValueError("ptn must have shape (npts, ngrnd, nvm)")
    veget_max_bg = thermosoil_veget_max_bg(veget_max)
    if veget_max_bg.shape != (ptn.shape[0], ptn.shape[2]):
        raise ValueError("veget_max must have shape (npts, nvm) matching ptn")
    nslm = ptn.shape[1] if nslm is None else int(nslm)
    if nslm < 1 or nslm > ptn.shape[1]:
        raise ValueError("nslm must be between 1 and ngrnd")
    return jnp.sum(ptn[:, :nslm, :] * veget_max_bg[:, None, :], axis=2)


def thermosoil_profile_no_explicit_snow(
    *,
    ptn,
    cgrnd,
    dgrnd,
    temp_sol_new,
    temp_sol_new_pft,
    veget_max,
    ok_laidev,
    lambda_thermal,
    veget_mask_2d=None,
    nslm=None,
) -> ThermosoilProfileResult:
    """Update ``ptn`` using the standard no-explicit-snow profile equations.

    Fortran provenance: ``thermosoil.f90::thermosoil_profile`` lines
    1804-1835, followed by ``thermosoil_diaglev`` lines 3503-3510. This
    excludes the explicit-snow branch at lines 1807-1814.
    """

    ptn = _as_float64(ptn)
    cgrnd = _as_float64(cgrnd)
    dgrnd = _as_float64(dgrnd)
    temp_sol_new = _as_float64(temp_sol_new)
    temp_sol_new_pft = _as_float64(temp_sol_new_pft)
    veget_max = _as_float64(veget_max)
    ok_laidev = jnp.asarray(ok_laidev, dtype=bool)
    mask = None if veget_mask_2d is None else jnp.asarray(veget_mask_2d, dtype=bool)
    if ptn.ndim != 3:
        raise ValueError("ptn must have shape (npts, ngrnd, nvm)")
    npts, ngrnd, nvm = ptn.shape
    if cgrnd.shape != (npts, ngrnd - 1, nvm) or dgrnd.shape != (npts, ngrnd - 1, nvm):
        raise ValueError("cgrnd and dgrnd must have shape (npts, ngrnd - 1, nvm)")
    if temp_sol_new.shape != (npts,):
        raise ValueError("temp_sol_new must have shape (npts,)")
    if temp_sol_new_pft.shape != (npts, nvm):
        raise ValueError("temp_sol_new_pft must have shape (npts, nvm)")
    if veget_max.shape != (npts, nvm):
        raise ValueError("veget_max must have shape (npts, nvm)")
    if ok_laidev.shape != (nvm,):
        raise ValueError("ok_laidev must have shape (nvm,)")
    if mask is not None and mask.shape != (npts, nvm):
        raise ValueError("veget_mask_2d must have shape (npts, nvm)")
    if mask is None:
        mask = jnp.ones((npts, nvm), dtype=bool)

    lamb = jnp.asarray(lambda_thermal, dtype=jnp.float64)
    active_laidev = ok_laidev[None, :] & (veget_max > 0.0)
    surface_temp = jnp.where(active_laidev, temp_sol_new_pft, temp_sol_new[:, None])
    ptn_updated = ptn
    layer1 = (lamb * cgrnd[:, 0, :] + surface_temp) / (lamb * (1.0 - dgrnd[:, 0, :]) + 1.0)
    ptn_updated = ptn_updated.at[:, 0, :].set(jnp.where(mask, layer1, ptn_updated[:, 0, :]))
    for jg in range(ngrnd - 1):
        next_layer = cgrnd[:, jg, :] + dgrnd[:, jg, :] * ptn_updated[:, jg, :]
        ptn_updated = ptn_updated.at[:, jg + 1, :].set(jnp.where(mask, next_layer, ptn_updated[:, jg + 1, :]))
    stempdiag = thermosoil_diaglev_from_ptn(ptn=ptn_updated, veget_max=veget_max, nslm=nslm)
    return ThermosoilProfileResult(ptn=ptn_updated, stempdiag=stempdiag)


def thermosoil_profile_explicit_snow(
    *,
    ptn,
    cgrnd,
    dgrnd,
    cgrnd_snow,
    dgrnd_snow,
    temp_sol_new,
    snowtemp,
    veget_max,
    frac_snow_veg,
    frac_snow_nobio,
    totfrac_nobio,
    veget_mask_2d=None,
    nslm=None,
) -> ThermosoilProfileResult:
    """Update ``ptn`` with the active explicit-snow profile equations.

    Fortran provenance: ``thermosoil.f90::thermosoil_profile`` lines
    1807-1814 compute the first soil layer from previous snow coefficients;
    lines 1827-1835 propagate deeper soil layers with ``cgrnd``/``dgrnd``.
    """

    ptn = _as_float64(ptn)
    cgrnd = _as_float64(cgrnd)
    dgrnd = _as_float64(dgrnd)
    cgrnd_snow = _as_float64(cgrnd_snow)
    dgrnd_snow = _as_float64(dgrnd_snow)
    temp_sol_new = _as_float64(temp_sol_new)
    snowtemp = _as_float64(snowtemp)
    veget_max = _as_float64(veget_max)
    frac_snow_veg = _as_float64(frac_snow_veg)
    frac_snow_nobio = _as_float64(frac_snow_nobio)
    totfrac_nobio = _as_float64(totfrac_nobio)
    mask = None if veget_mask_2d is None else jnp.asarray(veget_mask_2d, dtype=bool)
    if ptn.ndim != 3:
        raise ValueError("ptn must have shape (npts, ngrnd, nvm)")
    npts, ngrnd, nvm = ptn.shape
    if cgrnd.shape != (npts, ngrnd - 1, nvm) or dgrnd.shape != (npts, ngrnd - 1, nvm):
        raise ValueError("cgrnd and dgrnd must have shape (npts, ngrnd - 1, nvm)")
    if cgrnd_snow.ndim != 2 or dgrnd_snow.shape != cgrnd_snow.shape or cgrnd_snow.shape[0] != npts:
        raise ValueError("cgrnd_snow and dgrnd_snow must share shape (npts, nsnow)")
    if snowtemp.shape != cgrnd_snow.shape:
        raise ValueError("snowtemp must match cgrnd_snow shape")
    if temp_sol_new.shape != (npts,):
        raise ValueError("temp_sol_new must have shape (npts,)")
    if veget_max.shape != (npts, nvm):
        raise ValueError("veget_max must have shape (npts, nvm)")
    if frac_snow_veg.shape != (npts,) or totfrac_nobio.shape != (npts,):
        raise ValueError("frac_snow_veg and totfrac_nobio must have shape (npts,)")
    if frac_snow_nobio.ndim != 2 or frac_snow_nobio.shape[0] != npts:
        raise ValueError("frac_snow_nobio must have shape (npts, nnobio)")
    if mask is None:
        mask = jnp.ones((npts, nvm), dtype=bool)
    elif mask.shape != (npts, nvm):
        raise ValueError("veget_mask_2d must have shape (npts, nvm)")

    snow_weight = frac_snow_veg * (1.0 - totfrac_nobio)
    nobio_weight = jnp.sum(frac_snow_nobio, axis=1) * totfrac_nobio
    temp_sol_eff = (
        snowtemp[:, -1] * snow_weight
        + temp_sol_new * nobio_weight
        + temp_sol_new * (1.0 - (snow_weight + nobio_weight))
    )
    ptn_updated = ptn
    layer1 = cgrnd_snow[:, -1, None] + dgrnd_snow[:, -1, None] * temp_sol_eff[:, None]
    ptn_updated = ptn_updated.at[:, 0, :].set(jnp.where(mask, layer1, ptn_updated[:, 0, :]))
    for jg in range(ngrnd - 1):
        next_layer = cgrnd[:, jg, :] + dgrnd[:, jg, :] * ptn_updated[:, jg, :]
        ptn_updated = ptn_updated.at[:, jg + 1, :].set(jnp.where(mask, next_layer, ptn_updated[:, jg + 1, :]))
    stempdiag = thermosoil_diaglev_from_ptn(ptn=ptn_updated, veget_max=veget_max, nslm=nslm)
    return ThermosoilProfileResult(ptn=ptn_updated, stempdiag=stempdiag)


def thermosoil_energy_diagnostics(*, temp_sol_new, temp_sol_beg, soilcap, pcapa_en, veget_max, ptn, sn_capa):
    """Compute THERMOSOIL surface energy diagnostics and begin-state update.

    Fortran provenance: ``thermosoil.f90::thermosoil_energy`` lines
    2442-2460. ``pcapa_en`` is an explicit input because it is produced by
    the same audited thermal-property path as ``pcapa``.
    """

    temp_sol_new = _as_float64(temp_sol_new)
    temp_sol_beg = _as_float64(temp_sol_beg)
    soilcap = _as_float64(soilcap)
    pcapa_en = _as_float64(pcapa_en)
    veget_max = _as_float64(veget_max)
    ptn = _as_float64(ptn)
    if ptn.ndim != 3:
        raise ValueError("ptn must have shape (npts, ngrnd, nvm)")
    npts, _, nvm = ptn.shape
    if temp_sol_new.shape != (npts,) or temp_sol_beg.shape != (npts,) or soilcap.shape != (npts,):
        raise ValueError("temp_sol_new, temp_sol_beg, and soilcap must have shape (npts,)")
    if pcapa_en.shape != ptn.shape:
        raise ValueError("pcapa_en must match ptn shape")
    if veget_max.shape != (npts, nvm):
        raise ValueError("veget_max must have shape (npts, nvm)")

    delta_energy = soilcap * (temp_sol_new - temp_sol_beg)
    snow_like = jnp.sum(pcapa_en[:, 0, :] * veget_max, axis=1) <= jnp.asarray(sn_capa, dtype=jnp.float64)
    coldcont_incr = jnp.where(snow_like, delta_energy, 0.0)
    surfheat_incr = jnp.where(snow_like, 0.0, delta_energy)
    return ThermosoilEnergyResult(
        surfheat_incr=surfheat_incr,
        coldcont_incr=coldcont_incr,
        ptn_beg=ptn,
        temp_sol_beg=temp_sol_new,
    )


def thermosoil_wlupdate(
    *,
    ptn,
    hsd,
    hsdlong,
    veget_mask_2d,
    dt_sechiba,
    ndeep=None,
    zero_celsius=ZERO_CELSIUS,
    fr_dt=FR_DT,
    tau_freezesoil=30.0 * 86400.0,
) -> ThermosoilWetnessLongResult:
    """Update long-term wetness only in thawed, active soil.

    Fortran provenance: ``thermosoil.f90::thermosoil_wlupdate`` lines
    3648-3672. The source loops over ``1:ndeep`` and leaves all other state
    unchanged.
    """

    ptn = _as_float64(ptn)
    hsd = _as_float64(hsd)
    hsdlong = _as_float64(hsdlong)
    mask = jnp.asarray(veget_mask_2d, dtype=bool)
    if ptn.ndim != 3:
        raise ValueError("ptn must have shape (npts, ngrnd, nvm)")
    if hsd.shape != ptn.shape or hsdlong.shape != ptn.shape:
        raise ValueError("hsd and hsdlong must match ptn shape")
    if mask.shape != (ptn.shape[0], ptn.shape[2]):
        raise ValueError("veget_mask_2d must have shape (npts, nvm)")
    ndeep = ptn.shape[1] if ndeep is None else int(ndeep)
    if ndeep < 0 or ndeep > ptn.shape[1]:
        raise ValueError("ndeep must be between zero and ngrnd")
    updated = (hsd * dt_sechiba + hsdlong * (tau_freezesoil - dt_sechiba)) / tau_freezesoil
    thawed = (ptn > zero_celsius + fr_dt / 2.0) & mask[:, None, :]
    depth_mask = jnp.arange(ptn.shape[1])[None, :, None] < ndeep
    return ThermosoilWetnessLongResult(hsdlong=jnp.where(thawed & depth_mask, updated, hsdlong))


def thermosoil_explicit_snow_surface_clamp(
    *, temp_sol_new, snowdz, ok_explicitsnow=True, tp_00=ZERO_CELSIUS
):
    """Clamp snow-covered surface temperature to the melting point.

    Fortran provenance: ``thermosoil.f90::thermosoil_main`` lines 1041-1050.
    Snow-free points and runs without the explicit snow scheme are unchanged.
    """

    temp_sol_new = _as_float64(temp_sol_new)
    snowdz = _as_float64(snowdz)
    if temp_sol_new.ndim != 1:
        raise ValueError("temp_sol_new must have shape (npts,)")
    if snowdz.ndim != 2 or snowdz.shape[0] != temp_sol_new.shape[0]:
        raise ValueError("snowdz must have shape (npts, nsnow)")
    if not bool(ok_explicitsnow):
        return temp_sol_new
    covered = jnp.sum(snowdz, axis=1) > 0.0
    return jnp.where(
        covered & (temp_sol_new >= tp_00),
        jnp.asarray(tp_00, dtype=jnp.float64),
        temp_sol_new,
    )


def add_heat_zimov(
    *,
    ptn,
    heat_zimov,
    pcapa,
    dlt,
    dt_sechiba,
    veget_mask_2d,
    veget_max_bg,
) -> ThermosoilZimovHeatResult:
    """Apply decomposition heat and recompute the background PFT mean.

    Fortran provenance: ``thermosoil.f90::add_heat_Zimov`` lines 3426-3463.
    """

    ptn = _as_float64(ptn)
    heat_zimov = _as_float64(heat_zimov)
    pcapa = _as_float64(pcapa)
    dlt = _as_float64(dlt)
    mask = jnp.asarray(veget_mask_2d, dtype=bool)
    veget_max_bg = _as_float64(veget_max_bg)
    if ptn.ndim != 3:
        raise ValueError("ptn must have shape (npts, ngrnd, nvm)")
    if heat_zimov.shape != ptn.shape or pcapa.shape != ptn.shape:
        raise ValueError("heat_zimov and pcapa must match ptn shape")
    if dlt.shape != (ptn.shape[1],):
        raise ValueError("dlt must have shape (ngrnd,)")
    expected_pft_shape = (ptn.shape[0], ptn.shape[2])
    if mask.shape != expected_pft_shape or veget_max_bg.shape != expected_pft_shape:
        raise ValueError("veget_mask_2d and veget_max_bg must have shape (npts, nvm)")
    increment = heat_zimov * dt_sechiba / (pcapa * dlt[None, :, None])
    ptn_updated = jnp.where(mask[:, None, :], ptn + increment, ptn)
    ptn_pftmean = jnp.sum(ptn_updated * veget_max_bg[:, None, :], axis=2)
    return ThermosoilZimovHeatResult(ptn=ptn_updated, ptn_pftmean=ptn_pftmean)


def thermosoil_humlev(
    *,
    shumdiag_perma,
    mc_layh,
    mcl_layh,
    tmc_layh,
    mc_layh_pft,
    mcl_layh_pft,
    tmc_layh_pft,
    znt,
    zlt,
    dz5,
    veget_mask_2d=None,
    satsoil=False,
    min_sechiba=MIN_SECHIBA,
) -> ThermosoilHumlevResult:
    """Put HYDROL moisture diagnostics on THERMOSOIL thermal levels.

    Fortran provenance: ``thermosoil.f90::thermosoil_humlev`` lines
    2294-2395. The commented deep-permafrost ``update_deep_soil_moisture``
    call at lines 2388-2390 is intentionally not active here, matching source.
    """

    shumdiag_perma = _as_float64(shumdiag_perma)
    mc_layh = _as_float64(mc_layh)
    mcl_layh = _as_float64(mcl_layh)
    tmc_layh = _as_float64(tmc_layh)
    mc_layh_pft = _as_float64(mc_layh_pft)
    mcl_layh_pft = _as_float64(mcl_layh_pft)
    tmc_layh_pft = _as_float64(tmc_layh_pft)
    znt = _as_float64(znt)
    zlt = _as_float64(zlt)
    dz5 = _as_float64(dz5)
    mask = None if veget_mask_2d is None else jnp.asarray(veget_mask_2d, dtype=bool)

    if shumdiag_perma.ndim != 2:
        raise ValueError("shumdiag_perma must have shape (npts, nslm)")
    npts, nslm = shumdiag_perma.shape
    if nslm < 2:
        raise ValueError("nslm must be at least 2 for thermosoil_humlev interpolation")
    if mc_layh.shape != (npts, nslm) or mcl_layh.shape != (npts, nslm) or tmc_layh.shape != (npts, nslm):
        raise ValueError("mc_layh, mcl_layh, and tmc_layh must have shape (npts, nslm)")
    if mc_layh_pft.ndim != 3:
        raise ValueError("mc_layh_pft must have shape (npts, nslm, nvm)")
    npts_pft, nslm_pft, nvm = mc_layh_pft.shape
    if (npts_pft, nslm_pft) != (npts, nslm):
        raise ValueError("PFT moisture fields must share (npts, nslm)")
    if mcl_layh_pft.shape != (npts, nslm, nvm) or tmc_layh_pft.shape != (npts, nslm, nvm):
        raise ValueError("PFT moisture fields must have shape (npts, nslm, nvm)")
    if znt.ndim != 1 or zlt.ndim != 1:
        raise ValueError("znt and zlt must be 1D vertical vectors")
    ngrnd = int(znt.shape[0])
    if zlt.shape != (ngrnd,):
        raise ValueError("zlt must have shape (ngrnd,)")
    if ngrnd < nslm:
        raise ValueError("ngrnd must be at least nslm")
    if dz5.shape[0] < max(ngrnd - 1, nslm - 1):
        raise ValueError("dz5 must contain at least ngrnd - 1 entries")
    if mask is None:
        mask = jnp.ones((npts, nvm), dtype=bool)
    elif mask.shape != (npts, nvm):
        raise ValueError("veget_mask_2d must have shape (npts, nvm)")

    mc_layt = jnp.zeros((npts, ngrnd), dtype=jnp.float64)
    mcl_layt = jnp.zeros((npts, ngrnd), dtype=jnp.float64)
    tmc_layt = jnp.zeros((npts, ngrnd), dtype=jnp.float64)
    mc_layt_pft = jnp.zeros((npts, ngrnd, nvm), dtype=jnp.float64)
    mcl_layt_pft = jnp.zeros((npts, ngrnd, nvm), dtype=jnp.float64)
    tmc_layt_pft = jnp.zeros((npts, ngrnd, nvm), dtype=jnp.float64)

    for jd in range(nslm):
        if jd == 0:
            mc_val = mc_layh[:, jd]
            mcl_val = mcl_layh[:, jd]
            mc_pft_val = jnp.maximum(mc_layh_pft[:, jd, :], min_sechiba)
            mcl_pft_val = jnp.maximum(mcl_layh_pft[:, jd, :], min_sechiba)
        elif jd == 1:
            w_prev = (znt[jd] - zlt[jd - 1]) / znt[jd]
            w_curr = zlt[jd - 1] / znt[jd]
            mc_val = mc_layh[:, jd - 1] * w_prev + mc_layh[:, jd] * w_curr
            mcl_val = mcl_layh[:, jd - 1] * w_prev + mcl_layh[:, jd] * w_curr
            mc_pft_val = jnp.maximum(
                mc_layh_pft[:, jd - 1, :] * w_prev + mc_layh_pft[:, jd, :] * w_curr,
                min_sechiba,
            )
            mcl_pft_val = jnp.maximum(
                mcl_layh_pft[:, jd - 1, :] * w_prev + mcl_layh_pft[:, jd, :] * w_curr,
                min_sechiba,
            )
        elif jd == nslm - 1:
            w_prev = (zlt[jd] - zlt[jd - 1]) / (zlt[jd] - znt[jd - 1])
            w_curr = (zlt[jd - 1] - znt[jd - 1]) / (zlt[jd] - znt[jd - 1])
            mc_val = mc_layh[:, jd - 1] * w_prev + mc_layh[:, jd] * w_curr
            mcl_val = mcl_layh[:, jd - 1] * w_prev + mcl_layh[:, jd] * w_curr
            mc_pft_val = jnp.maximum(
                mc_layh_pft[:, jd - 1, :] * w_prev + mc_layh_pft[:, jd, :] * w_curr,
                min_sechiba,
            )
            mcl_pft_val = jnp.maximum(
                mcl_layh_pft[:, jd - 1, :] * w_prev + mcl_layh_pft[:, jd, :] * w_curr,
                min_sechiba,
            )
        else:
            w_curr = dz5[jd - 1]
            mc_val = mc_layh[:, jd - 1] * (1.0 - w_curr) + mc_layh[:, jd] * w_curr
            mcl_val = mcl_layh[:, jd - 1] * (1.0 - w_curr) + mcl_layh[:, jd] * w_curr
            mc_pft_val = jnp.maximum(
                mc_layh_pft[:, jd - 1, :] * (1.0 - w_curr) + mc_layh_pft[:, jd, :] * w_curr,
                min_sechiba,
            )
            mcl_pft_val = jnp.maximum(
                mcl_layh_pft[:, jd - 1, :] * (1.0 - w_curr) + mcl_layh_pft[:, jd, :] * w_curr,
                min_sechiba,
            )
        mc_layt = mc_layt.at[:, jd].set(mc_val)
        mcl_layt = mcl_layt.at[:, jd].set(mcl_val)
        tmc_layt = tmc_layt.at[:, jd].set(tmc_layh[:, jd])
        mc_layt_pft = mc_layt_pft.at[:, jd, :].set(mc_pft_val)
        mcl_layt_pft = mcl_layt_pft.at[:, jd, :].set(mcl_pft_val)
        tmc_layt_pft = tmc_layt_pft.at[:, jd, :].set(tmc_layh_pft[:, jd, :])

    for jd in range(nslm, ngrnd):
        mc_layt = mc_layt.at[:, jd].set(mc_layh[:, nslm - 1])
        mcl_layt = mcl_layt.at[:, jd].set(mcl_layh[:, nslm - 1])
        tmc_layt = tmc_layt.at[:, jd].set(tmc_layh[:, nslm - 1])
        mc_layt_pft = mc_layt_pft.at[:, jd, :].set(mc_layh_pft[:, nslm - 1, :])
        mcl_layt_pft = mcl_layt_pft.at[:, jd, :].set(mcl_layh_pft[:, nslm - 1, :])
        tmc_layt_pft = tmc_layt_pft.at[:, jd, :].set(tmc_layh_pft[:, nslm - 1, :])

    if satsoil:
        shum_ngrnd_perma = jnp.ones((npts, ngrnd, nvm), dtype=jnp.float64)
    else:
        shum_ngrnd_perma = jnp.zeros((npts, ngrnd, nvm), dtype=jnp.float64)
        for jd in range(nslm):
            shum_ngrnd_perma = shum_ngrnd_perma.at[:, jd, :].set(
                jnp.where(mask, shumdiag_perma[:, jd, None], 0.0)
            )
        for jd in range(nslm, ngrnd):
            shum_ngrnd_perma = shum_ngrnd_perma.at[:, jd, :].set(
                jnp.where(mask, shumdiag_perma[:, nslm - 1, None], 0.0)
            )

    return ThermosoilHumlevResult(
        mc_layt=mc_layt,
        mcl_layt=mcl_layt,
        tmc_layt=tmc_layt,
        mc_layt_pft=mc_layt_pft,
        mcl_layt_pft=mcl_layt_pft,
        tmc_layt_pft=tmc_layt_pft,
        shum_ngrnd_perma=shum_ngrnd_perma,
    )


def _soil_class_values(njsc, table):
    njsc = jnp.asarray(njsc, dtype=jnp.int32)
    values = _as_float64(table)
    if njsc.ndim != 1:
        raise ValueError("njsc must have shape (npts,) with Fortran 1-based soil classes")
    if not isinstance(njsc, core.Tracer) and (
        bool(jnp.any(njsc < 1)) or bool(jnp.any(njsc > values.shape[0]))
    ):
        raise ValueError("njsc contains a soil class outside the supplied table")
    return values[njsc - 1]


def thermosoil_cond(
    *,
    njsc,
    smc,
    sh2o,
    qz=QZ_USDA,
    smcmax=SMCMAX_USDA,
    min_sechiba=MIN_SECHIBA,
    thkqtz=THKQTZ,
    thkice=THKICE,
    thkw=THKW,
) -> ThermosoilCondResult:
    """Compute soil thermal conductivity without organic/PFT corrections.

    Fortran provenance: ``thermosoil.f90::thermosoil_cond`` lines
    1883-1972. ``njsc`` is the Fortran 1-based dominant soil class.
    """

    smc = _as_float64(smc)
    sh2o = _as_float64(sh2o)
    if smc.ndim != 2:
        raise ValueError("smc must have shape (npts, ngrnd)")
    if sh2o.shape != smc.shape:
        raise ValueError("sh2o must match smc shape")
    npts, ngrnd = smc.shape
    smcmax_point = _soil_class_values(njsc, smcmax)
    qz_point = _soil_class_values(njsc, qz)
    if smcmax_point.shape != (npts,):
        raise ValueError("njsc must have shape (npts,)")

    gammd = (1.0 - smcmax_point) * 2700.0
    thkdry = (0.135 * gammd + 64.7) / (2700.0 - 0.947 * gammd)
    thko = jnp.where(qz_point > 0.2, 2.0, 3.0)
    thks = (jnp.asarray(thkqtz, dtype=jnp.float64) ** qz_point) * (thko ** (1.0 - qz_point))

    satratio = smc / smcmax_point[:, None]
    positive_smc = smc > min_sechiba
    xunfroz = jnp.where(positive_smc, sh2o / smc, 0.0)
    xu = xunfroz * smcmax_point[:, None]
    thksat = jnp.where(
        positive_smc,
        thks[:, None] ** (1.0 - smcmax_point[:, None])
        * jnp.asarray(thkice, dtype=jnp.float64) ** (smcmax_point[:, None] - xu)
        * jnp.asarray(thkw, dtype=jnp.float64) ** xu,
        0.0,
    )
    frozen = (sh2o + 0.0005) < smc
    ake_unfrozen = jnp.where(
        satratio > 0.1,
        jnp.log10(satratio) + 1.0,
        jnp.where(satratio > 0.05, 0.7 * jnp.log10(satratio) + 1.0, 0.0),
    )
    ake = jnp.where(frozen, satratio, ake_unfrozen)
    cnd = ake * (thksat - thkdry[:, None]) + thkdry[:, None]
    return ThermosoilCondResult(
        cnd=cnd,
        ake=ake,
        thksat=thksat,
        thkdry=jnp.broadcast_to(thkdry[:, None], (npts, ngrnd)),
        thks=jnp.broadcast_to(thks[:, None], (npts, ngrnd)),
        satratio=satratio,
    )


def thermosoil_soilc_tempdiff_fraction(*, soilc_total=None, refsoc=None, nvm=None, soilc_max=SOILC_MAX):
    """Build organic/mineral fractions for ``USE_SOILC_TEMPDIFF``.

    Fortran provenance: ``thermosoil.f90::thermosoil_getdiff`` lines
    2673-2689. With ``use_refSOC`` true the source uses ``refSOC(ji,jg)`` for
    every PFT; otherwise it uses ``soilc_total(ji,jg,jv)``. The returned
    ``zx1`` is the organic fraction and ``zx2=1-zx1``.
    """

    if (soilc_total is None) == (refsoc is None):
        raise ValueError("provide exactly one of soilc_total or refsoc")
    if refsoc is not None:
        refsoc = _as_float64(refsoc)
        if refsoc.ndim != 2:
            raise ValueError("refsoc must have shape (npts, ngrnd)")
        nvm = 1 if nvm is None else int(nvm)
        if nvm < 1:
            raise ValueError("nvm must be at least 1")
        zx1_2d = jnp.minimum(refsoc / jnp.asarray(soilc_max, dtype=jnp.float64), 1.0)
        zx1 = jnp.broadcast_to(zx1_2d[:, :, None], (refsoc.shape[0], refsoc.shape[1], nvm))
    else:
        soilc_total = _as_float64(soilc_total)
        if soilc_total.ndim != 3:
            raise ValueError("soilc_total must have shape (npts, ngrnd, nvm)")
        zx1 = jnp.minimum(soilc_total / jnp.asarray(soilc_max, dtype=jnp.float64), 1.0)
    zx2 = 1.0 - zx1
    return zx1, zx2


def thermosoil_toporganiclayer_fraction(*, organic_layer_thick, zlt, nvm):
    """Build organic/mineral fractions for ``USE_TOPORGANICLAYER_TEMPDIFF``.

    Fortran provenance: ``thermosoil.f90::thermosoil_getdiff`` lines
    2632-2671. The source contains an apparent loop typo for copying levels to
    PFTs; this helper applies the intended same organic fraction to all PFTs.
    """

    organic_layer_thick = _as_float64(organic_layer_thick)
    zlt = _as_float64(zlt)
    if organic_layer_thick.ndim != 1:
        raise ValueError("organic_layer_thick must have shape (npts,)")
    if zlt.ndim != 1 or zlt.shape[0] < 1:
        raise ValueError("zlt must be a non-empty 1D vector")
    npts = organic_layer_thick.shape[0]
    ngrnd = zlt.shape[0]
    nvm = int(nvm)
    if nvm < 1:
        raise ValueError("nvm must be at least 1")
    zx1_2d = jnp.zeros((npts, ngrnd), dtype=jnp.float64)
    first = jnp.where(
        organic_layer_thick > zlt[0],
        1.0,
        jnp.where(organic_layer_thick > 0.0, organic_layer_thick / zlt[0], 0.0),
    )
    zx1_2d = zx1_2d.at[:, 0].set(first)
    for jg in range(1, ngrnd):
        value = jnp.where(
            organic_layer_thick > zlt[jg],
            1.0,
            jnp.where(
                organic_layer_thick > zlt[jg - 1],
                (organic_layer_thick - zlt[jg - 1]) / (zlt[jg] - zlt[jg - 1]),
                0.0,
            ),
        )
        zx1_2d = zx1_2d.at[:, jg].set(value)
    zx1 = jnp.broadcast_to(zx1_2d[:, :, None], (npts, ngrnd, nvm))
    return zx1, 1.0 - zx1


def thermosoil_cond_pft(
    *,
    njsc,
    smc,
    sh2o,
    zx1,
    zx2,
    porosnet,
    qz=QZ_USDA,
    smcmax=SMCMAX_USDA,
    min_sechiba=MIN_SECHIBA,
    cond_dry_org=COND_DRY_ORG,
    cond_solid_org=COND_SOLID_ORG,
    thkqtz=THKQTZ,
    thkice=THKICE,
    thkw=THKW,
) -> ThermosoilCondResult:
    """Compute PFT-resolved thermal conductivity with organic/mineral fractions.

    Fortran provenance: ``thermosoil.f90::thermosoil_cond_pft`` lines
    2007-2127. The source uses grid-cell ``smc``/``sh2o`` and PFT-specific
    ``zx1``/``zx2``/``porosnet``.
    """

    smc = _as_float64(smc)
    sh2o = _as_float64(sh2o)
    zx1 = _as_float64(zx1)
    zx2 = _as_float64(zx2)
    porosnet = _as_float64(porosnet)
    if smc.ndim != 2:
        raise ValueError("smc must have shape (npts, ngrnd)")
    if sh2o.shape != smc.shape:
        raise ValueError("sh2o must match smc shape")
    if zx1.ndim != 3:
        raise ValueError("zx1 must have shape (npts, ngrnd, nvm)")
    npts, ngrnd = smc.shape
    if zx1.shape[:2] != (npts, ngrnd) or zx2.shape != zx1.shape or porosnet.shape != zx1.shape:
        raise ValueError("zx1, zx2, and porosnet must share shape (npts, ngrnd, nvm)")
    smcmax_point = _soil_class_values(njsc, smcmax)
    qz_point = _soil_class_values(njsc, qz)

    gammd = (1.0 - smcmax_point) * 2700.0
    thkdry_min = (0.135 * gammd + 64.7) / (2700.0 - 0.947 * gammd)
    thkdry = zx1 * cond_dry_org + zx2 * thkdry_min[:, None, None]
    thko = jnp.where(qz_point > 0.2, 2.0, 3.0)
    thks_min = (jnp.asarray(thkqtz, dtype=jnp.float64) ** qz_point) * (thko ** (1.0 - qz_point))
    thks = zx1 * cond_solid_org + zx2 * thks_min[:, None, None]

    satratio = smc[:, :, None] / porosnet
    positive_smc = smc > min_sechiba
    xunfroz = jnp.where(positive_smc, sh2o / smc, 0.0)
    xu = xunfroz[:, :, None] * porosnet
    thksat = jnp.where(
        positive_smc[:, :, None],
        thks ** (1.0 - porosnet)
        * jnp.asarray(thkice, dtype=jnp.float64) ** (porosnet - xu)
        * jnp.asarray(thkw, dtype=jnp.float64) ** xu,
        0.0,
    )
    frozen = ((sh2o + 0.0005) < smc)[:, :, None]
    ake_unfrozen = jnp.where(
        satratio > 0.1,
        jnp.log10(satratio) + 1.0,
        jnp.where(satratio > 0.05, 0.7 * jnp.log10(satratio) + 1.0, 0.0),
    )
    ake = jnp.where(frozen, satratio, ake_unfrozen)
    cnd = ake * (thksat - thkdry) + thkdry
    return ThermosoilCondResult(
        cnd=cnd,
        ake=ake,
        thksat=thksat,
        thkdry=thkdry,
        thks=thks,
        satratio=satratio,
    )


def thermosoil_cond_nopft(
    *,
    njsc,
    smc,
    sh2o,
    zx1,
    zx2,
    porosnet,
    qz=QZ_USDA,
    smcmax=SMCMAX_USDA,
    min_sechiba=MIN_SECHIBA,
    cond_dry_org=COND_DRY_ORG,
    cond_solid_org=COND_SOLID_ORG,
    thkqtz=THKQTZ,
    thkice=THKICE,
    thkw=THKW,
) -> ThermosoilCondResult:
    """Compute the non-PFT organic/mineral conductivity path.

    Fortran provenance: ``thermosoil.f90::thermosoil_cond_nopft`` lines
    2129-2231. Unlike ``thermosoil_cond``, porosity and organic fractions vary
    by thermal layer; unlike ``thermosoil_cond_pft``, there is no PFT axis.
    """

    smc = _as_float64(smc)
    sh2o = _as_float64(sh2o)
    zx1 = _as_float64(zx1)
    zx2 = _as_float64(zx2)
    porosnet = _as_float64(porosnet)
    if smc.ndim != 2:
        raise ValueError("smc must have shape (npts, ngrnd)")
    if sh2o.shape != smc.shape:
        raise ValueError("sh2o must match smc shape")
    if zx1.shape != smc.shape or zx2.shape != smc.shape or porosnet.shape != smc.shape:
        raise ValueError("zx1, zx2, and porosnet must match smc shape")
    npts, ngrnd = smc.shape
    smcmax_point = _soil_class_values(njsc, smcmax)
    qz_point = _soil_class_values(njsc, qz)
    if smcmax_point.shape != (npts,):
        raise ValueError("njsc must have shape (npts,)")

    gammd = (1.0 - smcmax_point) * 2700.0
    thkdry_min = (0.135 * gammd + 64.7) / (2700.0 - 0.947 * gammd)
    thkdry = zx1 * cond_dry_org + zx2 * thkdry_min[:, None]
    thko = jnp.where(qz_point > 0.2, 2.0, 3.0)
    thks_min = (jnp.asarray(thkqtz, dtype=jnp.float64) ** qz_point) * (thko ** (1.0 - qz_point))
    thks = zx1 * cond_solid_org + zx2 * thks_min[:, None]

    satratio = smc / porosnet
    positive_smc = smc > min_sechiba
    xunfroz = jnp.where(positive_smc, sh2o / smc, 0.0)
    xu = xunfroz * porosnet
    thksat = jnp.where(
        positive_smc,
        thks ** (1.0 - porosnet) * thkice ** (porosnet - xu) * thkw**xu,
        0.0,
    )
    frozen = (sh2o + 0.0005) < smc
    ake_unfrozen = jnp.where(
        satratio > 0.1,
        jnp.log10(satratio) + 1.0,
        jnp.where(satratio > 0.05, 0.7 * jnp.log10(satratio) + 1.0, 0.0),
    )
    ake = jnp.where(frozen, satratio, ake_unfrozen)
    cnd = ake * (thksat - thkdry) + thkdry
    return ThermosoilCondResult(
        cnd=cnd,
        ake=ake,
        thksat=thksat,
        thkdry=thkdry,
        thks=thks,
        satratio=satratio,
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_cond_nopft lines 2129-2231",
        ),
    )


def thermosoil_getdiff_explicit(
    *,
    ptn,
    njsc,
    veget_max,
    shum_ngrnd_permalong,
    mc_layt,
    tmc_layt_pft,
    snowrho,
    snowtemp,
    pb,
    dlt,
    zx1=None,
    qz=QZ_USDA,
    smcmax=SMCMAX_USDA,
    so_capa_dry_ns=SO_CAPA_DRY_NS_USDA,
    ok_laidev=None,
    ok_freeze_thermix=True,
    brk_flag=0,
    veget_mask_2d=None,
    fr_dt=FR_DT,
    zero_celsius=ZERO_CELSIUS,
    so_capa_dry=SO_CAPA_DRY,
    so_capa_dry_org=SO_CAPA_DRY_ORG,
    water_capa=WATER_CAPA,
    so_capa_wet=SO_CAPA_WET,
    so_capa_ice=None,
    poros=POROS,
    poros_org=0.92,
    lhf=LHF,
    rho_water=RHO_WATER,
    mille=MILLE,
    brk_capa=BRK_CAPA,
    brk_cond=BRK_COND,
    xci=XCI,
    zsnowthrmcond1=ZSNOWTHRMCOND1,
    zsnowthrmcond2=ZSNOWTHRMCOND2,
    zsnowthrmcond_avap=ZSNOWTHRMCOND_AVAP,
    zsnowthrmcond_bvap=ZSNOWTHRMCOND_BVAP,
    zsnowthrmcond_cvap=ZSNOWTHRMCOND_CVAP,
    xp00=XP00,
    min_sechiba=MIN_SECHIBA,
    nslm=None,
) -> ThermosoilGetdiffResult:
    """Compute active explicit-snow THERMOSOIL thermal properties.

    Fortran provenance: ``thermosoil.f90::thermosoil_getdiff`` lines
    2566-2863. The organic/mineral fraction ``zx1`` is explicit input here:
    in the paper run ``USE_SOILC_TEMPDIFF=y`` derives it from ``soilc_total``
    or ``refSOC`` at lines 2673-2685, which is an upstream data path rather
    than a tunable parameter.
    """

    ptn = _as_float64(ptn)
    shum_ngrnd_permalong = _as_float64(shum_ngrnd_permalong)
    veget_max = _as_float64(veget_max)
    mc_layt = _as_float64(mc_layt)
    tmc_layt_pft = _as_float64(tmc_layt_pft)
    snowrho = _as_float64(snowrho)
    snowtemp = _as_float64(snowtemp)
    pb = _as_float64(pb)
    dlt = _as_float64(dlt)
    if ptn.ndim != 3:
        raise ValueError("ptn must have shape (npts, ngrnd, nvm)")
    npts, ngrnd, nvm = ptn.shape
    if shum_ngrnd_permalong.shape != ptn.shape:
        raise ValueError("shum_ngrnd_permalong must match ptn shape")
    if veget_max.shape != (npts, nvm):
        raise ValueError("veget_max must have shape (npts, nvm)")
    if mc_layt.shape != (npts, ngrnd):
        raise ValueError("mc_layt must have shape (npts, ngrnd)")
    if tmc_layt_pft.shape != (npts, ngrnd, nvm):
        raise ValueError("tmc_layt_pft must have shape (npts, ngrnd, nvm)")
    if dlt.shape != (ngrnd,):
        raise ValueError("dlt must have shape (ngrnd,)")
    if snowrho.shape != snowtemp.shape or snowrho.ndim != 2 or snowrho.shape[0] != npts:
        raise ValueError("snowrho and snowtemp must share shape (npts, nsnow)")
    if pb.shape != (npts,):
        raise ValueError("pb must have shape (npts,)")
    if ok_laidev is None:
        ok_laidev = jnp.zeros((nvm,), dtype=bool)
    else:
        ok_laidev = jnp.asarray(ok_laidev, dtype=bool)
        if ok_laidev.shape != (nvm,):
            raise ValueError("ok_laidev must have shape (nvm,)")
    mask = jnp.ones((npts, nvm), dtype=bool) if veget_mask_2d is None else jnp.asarray(veget_mask_2d, dtype=bool)
    if mask.shape != (npts, nvm):
        raise ValueError("veget_mask_2d must have shape (npts, nvm)")
    if zx1 is None:
        zx1 = jnp.zeros_like(ptn)
    else:
        zx1 = _as_float64(zx1)
        if zx1.shape != ptn.shape:
            raise ValueError("zx1 must match ptn shape")
    zx2 = 1.0 - zx1

    so_capa_ice = (
        jnp.asarray(so_capa_dry, dtype=jnp.float64)
        + jnp.asarray(poros, dtype=jnp.float64) * jnp.asarray(CAPA_ICE, dtype=jnp.float64) * jnp.asarray(RHO_ICE, dtype=jnp.float64)
        if so_capa_ice is None
        else jnp.asarray(so_capa_ice, dtype=jnp.float64)
    )
    so_capa_dry_ns_point = _soil_class_values(njsc, so_capa_dry_ns)
    smcmax_point = _soil_class_values(njsc, smcmax)

    poros_net = zx1 * poros_org + zx2 * smcmax_point[:, None, None]
    so_capa_dry_net = zx1 * so_capa_dry_org + zx2 * so_capa_dry_ns_point[:, None, None]
    tmc_layt_pft_tmp = jnp.maximum(tmc_layt_pft, min_sechiba)
    pcapa_tmp = so_capa_dry_net * (1.0 - poros_net) + water_capa * mc_layt[:, :, None]
    pcapa_pft_tmp = so_capa_dry_net + water_capa * tmc_layt_pft_tmp / mille / dlt[None, :, None]

    lower = zero_celsius - fr_dt / 2.0
    upper = zero_celsius + fr_dt / 2.0
    frozen = ptn < lower
    unfrozen = ptn > upper
    transition = ~(frozen | unfrozen)
    xx = (ptn - lower) / fr_dt
    pcappa_supp = jnp.where(
        transition,
        shum_ngrnd_permalong * lhf * rho_water / fr_dt,
        0.0,
    )
    nslm = min(mc_layt.shape[1], ngrnd) if nslm is None else int(nslm)
    if nslm < 1 or nslm > ngrnd:
        raise ValueError("nslm must be between 1 and ngrnd")
    if nslm < ngrnd:
        deep = jnp.arange(ngrnd)[None, :, None] >= nslm
        pcappa_supp = jnp.where(deep, 0.0, pcappa_supp)
    profil_froz = jnp.where(frozen, 1.0, jnp.where(unfrozen, 0.0, 1.0 - xx))

    laidev = ok_laidev[None, None, :]
    pcapa_frozen = jnp.where(
        laidev,
        so_capa_dry_net + so_capa_ice * tmc_layt_pft_tmp / mille / dlt[None, :, None],
        so_capa_dry_net * (1.0 - poros_net) + so_capa_ice * mc_layt[:, :, None],
    )
    pcapa_unfrozen = jnp.where(laidev, pcapa_pft_tmp, pcapa_tmp)
    pcapa_transition = jnp.where(
        laidev,
        so_capa_dry_net
        + water_capa * tmc_layt_pft_tmp / mille / dlt[None, :, None] * xx
        + so_capa_ice * tmc_layt_pft_tmp / mille / dlt[None, :, None] * (1.0 - xx),
        so_capa_dry_net * (1.0 - poros_net)
        + water_capa * mc_layt[:, :, None] * xx
        + so_capa_ice * mc_layt[:, :, None] * (1.0 - xx)
        + pcappa_supp,
    )
    pcapa = jnp.where(frozen, pcapa_frozen, jnp.where(unfrozen, pcapa_unfrozen, pcapa_transition))
    pcapa = jnp.where(mask[:, None, :], pcapa, 0.0)
    profil_froz = jnp.where(mask[:, None, :], profil_froz, 0.0)
    pcappa_supp = jnp.where(mask[:, None, :], pcappa_supp, 0.0)
    if not ok_freeze_thermix:
        pcapa = jnp.where(laidev, pcapa_pft_tmp, pcapa_tmp)
        profil_froz = jnp.zeros_like(pcapa)
        pcappa_supp = jnp.zeros_like(pcapa)

    pcapa_en = pcapa
    profil_max = jnp.max(profil_froz, axis=2)
    profil_min = jnp.min(profil_froz, axis=2)
    profil_differs = jnp.any(jnp.abs(profil_max - profil_min) > 0.0)
    if not isinstance(profil_differs, core.Tracer) and bool(profil_differs):
        raise ValueError("profil_froz differs across PFTs; Fortran aborts this path")
    profil_froz_mean = profil_min
    tmp = mc_layt * (1.0 - profil_froz_mean)
    cond_pft = thermosoil_cond_pft(
        njsc=njsc,
        smc=mc_layt,
        sh2o=tmp,
        zx1=zx1,
        zx2=zx2,
        porosnet=poros_net,
        qz=qz,
        smcmax=smcmax,
        min_sechiba=min_sechiba,
    )
    cond_nopft = thermosoil_cond_nopft(
        njsc=njsc,
        smc=mc_layt,
        sh2o=tmp,
        zx1=zx1[:, :, 0],
        zx2=zx2[:, :, 0],
        porosnet=poros_net[:, :, 0],
        qz=qz,
        smcmax=smcmax,
        min_sechiba=min_sechiba,
    )
    pkappa_nopft = jnp.broadcast_to(cond_nopft.cnd[:, :, None], (npts, ngrnd, nvm))
    pkappa = jnp.where(jnp.any(ok_laidev), cond_pft.cnd, pkappa_nopft)
    if int(brk_flag) == 1:
        pcapa = pcapa.at[:, ngrnd - 2 : ngrnd, :].set(brk_capa)
        pcapa_en = pcapa_en.at[:, ngrnd - 2 : ngrnd, :].set(brk_capa)

    pcapa_snow = snowrho * xci
    pkappa_snow = (zsnowthrmcond1 + zsnowthrmcond2 * snowrho * snowrho) + jnp.maximum(
        0.0,
        (zsnowthrmcond_avap + (zsnowthrmcond_bvap / (snowtemp + zsnowthrmcond_cvap))) * (xp00 / (pb[:, None] * 100.0)),
    )
    return ThermosoilGetdiffResult(
        pcapa=pcapa,
        pcapa_en=pcapa_en,
        pkappa=pkappa,
        profil_froz=profil_froz,
        pcappa_supp=pcappa_supp,
        pcapa_snow=pcapa_snow,
        pkappa_snow=pkappa_snow,
        poros_net=poros_net,
        zx1=zx1,
        zx2=zx2,
    )


def thermosoil_getdiff_thinsnow(
    *,
    ptn,
    shum_ngrnd_permalong,
    snowdz,
    pcapa,
    pcapa_en,
    pkappa,
    profil_froz,
    veget_mask_2d,
    zlt,
    zero_celsius=ZERO_CELSIUS,
    fr_dt=FR_DT,
    so_capa_dry=SO_CAPA_DRY,
    so_capa_wet=SO_CAPA_WET,
    so_cond_dry=SO_COND_DRY,
    so_cond_wet=SO_COND_WET,
    sn_capa=SN_CAPA,
    sn_cond=SN_COND,
) -> ThermosoilThinSnowResult:
    """Overlay the source thin-snow correction on the first thermal layer.

    Fortran provenance: ``thermosoil.f90::thermosoil_getdiff_thinsnow`` lines
    3692-3788. Only points with total snow depth in ``(0, 0.01]`` are
    modified, and only thermal level one is touched. Arrays from the preceding
    ``thermosoil_getdiff`` call are explicit inputs so unaffected values retain
    their source-order state.
    """

    ptn = _as_float64(ptn)
    shum = _as_float64(shum_ngrnd_permalong)
    snowdz = _as_float64(snowdz)
    pcapa = _as_float64(pcapa)
    pcapa_en = _as_float64(pcapa_en)
    pkappa = _as_float64(pkappa)
    profil_froz = _as_float64(profil_froz)
    mask = jnp.asarray(veget_mask_2d, dtype=bool)
    zlt = _as_float64(zlt)
    if ptn.ndim != 3 or ptn.shape[1] < 1:
        raise ValueError("ptn must have shape (npts, ngrnd, nvm) with ngrnd >= 1")
    if shum.shape != ptn.shape:
        raise ValueError("shum_ngrnd_permalong must match ptn shape")
    if pcapa.shape != ptn.shape or pcapa_en.shape != ptn.shape:
        raise ValueError("pcapa and pcapa_en must match ptn shape")
    if pkappa.shape != ptn.shape or profil_froz.shape != ptn.shape:
        raise ValueError("pkappa and profil_froz must match ptn shape")
    if snowdz.ndim != 2 or snowdz.shape[0] != ptn.shape[0]:
        raise ValueError("snowdz must have shape (npts, nsnow)")
    if mask.shape != (ptn.shape[0], ptn.shape[2]):
        raise ValueError("veget_mask_2d must have shape (npts, nvm)")
    if zlt.ndim != 1 or zlt.shape[0] < 1:
        raise ValueError("zlt must be a non-empty one-dimensional array")

    snow_h = jnp.sum(snowdz, axis=1)
    thin = (snow_h <= 0.01) & (snow_h > 0.0)
    full_snow = snow_h > zlt[0]
    snow_fraction = jnp.where(full_snow, 1.0, snow_h / zlt[0])
    soil_fraction = jnp.where(full_snow, 0.0, (zlt[0] - snow_h) / zlt[0])
    active = thin[:, None] & mask

    lower = zero_celsius - fr_dt / 2.0
    upper = zero_celsius + fr_dt / 2.0
    first_ptn = ptn[:, 0, :]
    x = (first_ptn - lower) / fr_dt
    first_frozen = jnp.where(first_ptn < lower, 1.0, jnp.where(first_ptn > upper, 0.0, 1.0 - x))
    base_pcapa = so_capa_dry + shum[:, 0, :] * (so_capa_wet - so_capa_dry)
    first_pcapa = snow_fraction[:, None] * sn_capa + soil_fraction[:, None] * base_pcapa
    first_pcapa_en = jnp.where(snow_fraction[:, None] > 0.0, sn_capa, first_pcapa)
    base_pkappa = so_cond_dry + shum[:, 0, :] * (so_cond_wet - so_cond_dry)
    first_pkappa = 1.0 / (
        snow_fraction[:, None] / sn_cond + soil_fraction[:, None] / base_pkappa
    )

    pcapa = pcapa.at[:, 0, :].set(jnp.where(active, first_pcapa, pcapa[:, 0, :]))
    pcapa_en = pcapa_en.at[:, 0, :].set(jnp.where(active, first_pcapa_en, pcapa_en[:, 0, :]))
    pkappa = pkappa.at[:, 0, :].set(jnp.where(active, first_pkappa, pkappa[:, 0, :]))
    profil_froz = profil_froz.at[:, 0, :].set(jnp.where(active, first_frozen, profil_froz[:, 0, :]))
    return ThermosoilThinSnowResult(
        pcapa=pcapa,
        pcapa_en=pcapa_en,
        pkappa=pkappa,
        profil_froz=profil_froz,
        snow_fraction=jnp.where(thin, snow_fraction, 0.0),
        soil_fraction=jnp.where(thin, soil_fraction, 0.0),
    )


def thermosoil_getdiff_old_thermix_with_snow(
    *,
    snow,
    njsc,
    mc_layt,
    mcl_layt,
    tmc_layt,
    mc_layt_pft,
    mcl_layt_pft,
    tmc_layt_pft,
    dlt,
    zlt,
    ok_laidev,
    qz=QZ_USDA,
    smcmax=SMCMAX_USDA,
    so_capa_dry_ns=SO_CAPA_DRY_NS_USDA,
    water_capa=WATER_CAPA,
    mille=MILLE,
    min_sechiba=MIN_SECHIBA,
    sn_dens=SN_DENS,
    sn_capa=SN_CAPA,
    sn_cond=SN_COND,
    so_capa_dry=SO_CAPA_DRY,
    so_cond_dry=SO_COND_DRY,
    brk_flag=0,
    brk_capa=BRK_CAPA,
    brk_cond=BRK_COND,
) -> ThermosoilGetdiffResult:
    """Compute the old non-explicit-snow thermal properties.

    Fortran provenance:
    ``thermosoil.f90::thermosoil_getdiff_old_thermix_with_snow`` lines
    2884-3031. This is the branch selected by ``thermosoil_coef`` when
    ``ok_explicitsnow=false`` at lines 1499-1507.
    """

    snow = _as_float64(snow)
    mc_layt = _as_float64(mc_layt)
    mcl_layt = _as_float64(mcl_layt)
    tmc_layt = _as_float64(tmc_layt)
    mc_layt_pft = _as_float64(mc_layt_pft)
    mcl_layt_pft = _as_float64(mcl_layt_pft)
    tmc_layt_pft = _as_float64(tmc_layt_pft)
    dlt = _as_float64(dlt)
    zlt = _as_float64(zlt)
    ok_laidev = jnp.asarray(ok_laidev, dtype=bool)
    if mc_layt.ndim != 2:
        raise ValueError("mc_layt must have shape (npts, ngrnd)")
    npts, ngrnd = mc_layt.shape
    if snow.shape != (npts,):
        raise ValueError("snow must have shape (npts,)")
    if mcl_layt.shape != (npts, ngrnd) or tmc_layt.shape != (npts, ngrnd):
        raise ValueError("mcl_layt and tmc_layt must match mc_layt")
    if mc_layt_pft.ndim != 3 or mc_layt_pft.shape[:2] != (npts, ngrnd):
        raise ValueError("mc_layt_pft must have shape (npts, ngrnd, nvm)")
    nvm = mc_layt_pft.shape[2]
    if mcl_layt_pft.shape != (npts, ngrnd, nvm) or tmc_layt_pft.shape != (npts, ngrnd, nvm):
        raise ValueError("PFT moisture arrays must share shape (npts, ngrnd, nvm)")
    if dlt.shape != (ngrnd,) or zlt.shape[0] < ngrnd:
        raise ValueError("dlt and zlt must cover all thermal levels")
    if ok_laidev.shape != (nvm,):
        raise ValueError("ok_laidev must have shape (nvm,)")

    so_capa_dry_ns_point = _soil_class_values(njsc, so_capa_dry_ns)
    smcmax_point = _soil_class_values(njsc, smcmax)
    mcs_tmp = jnp.broadcast_to(smcmax_point[:, None], (npts, ngrnd))
    pcapa_tmp = so_capa_dry_ns_point[:, None] + water_capa * tmc_layt / mille / dlt[None, :]
    pcapa_wet = so_capa_dry_ns_point[:, None] + water_capa * smcmax_point[:, None]

    mc_layt_pft_tmp = jnp.maximum(mc_layt_pft, min_sechiba)
    mcl_layt_pft_tmp = jnp.maximum(mcl_layt_pft, min_sechiba)
    tmc_layt_pft_tmp = jnp.maximum(tmc_layt_pft, min_sechiba)
    pcapa_pft_tmp = jnp.where(
        ok_laidev[None, None, :],
        so_capa_dry_ns_point[:, None, None] + water_capa * tmc_layt_pft_tmp / mille / dlt[None, :, None],
        pcapa_tmp[:, :, None],
    )

    pkappa_pft_layers = []
    for jv in range(nvm):
        pkappa_pft_layers.append(
            thermosoil_cond(
                njsc=njsc,
                smc=mc_layt_pft_tmp[:, :, jv],
                sh2o=mcl_layt_pft_tmp[:, :, jv],
                qz=qz,
                smcmax=smcmax,
                min_sechiba=min_sechiba,
            ).cnd
        )
    pkappa_pft_tmp = jnp.stack(pkappa_pft_layers, axis=2)
    pkappa_tmp = thermosoil_cond(
        njsc=njsc,
        smc=mc_layt,
        sh2o=mcl_layt,
        qz=qz,
        smcmax=smcmax,
        min_sechiba=min_sechiba,
    ).cnd
    pkappa_wet = thermosoil_cond(
        njsc=njsc,
        smc=mcs_tmp,
        sh2o=mcs_tmp,
        qz=qz,
        smcmax=smcmax,
        min_sechiba=min_sechiba,
    ).cnd

    pcapa = jnp.zeros((npts, ngrnd, nvm), dtype=jnp.float64)
    pcapa_en = jnp.zeros_like(pcapa)
    pkappa = jnp.zeros_like(pcapa)
    snow_h = snow / jnp.asarray(sn_dens, dtype=jnp.float64)
    sn_capa = jnp.asarray(sn_capa, dtype=jnp.float64)
    sn_cond = jnp.asarray(sn_cond, dtype=jnp.float64)

    def set_layer(layer, lower_depth):
        full = snow_h > zlt[layer]
        partial = snow_h > lower_depth
        denom = zlt[layer] - lower_depth
        zx1 = jnp.where(denom > 0.0, (snow_h - lower_depth) / denom, 0.0)
        zx2 = jnp.where(denom > 0.0, (zlt[layer] - snow_h) / denom, 1.0)
        mixed_pcapa = zx1[:, None] * sn_capa + zx2[:, None] * pcapa_wet[:, layer, None]
        mixed_pkappa = 1.0 / (zx1[:, None] / sn_cond + zx2[:, None] / pkappa_wet[:, layer, None])
        pcapa_base = pcapa_pft_tmp[:, layer, :]
        pkappa_base = jnp.where(ok_laidev[None, :], pkappa_pft_tmp[:, layer, :], pkappa_tmp[:, layer, None])
        pcapa_layer = jnp.where(full[:, None], sn_capa, jnp.where(partial[:, None], mixed_pcapa, pcapa_base))
        pcapa_en_layer = jnp.where(full[:, None] | partial[:, None], sn_capa, pcapa_base)
        pkappa_layer = jnp.where(full[:, None], sn_cond, jnp.where(partial[:, None], mixed_pkappa, pkappa_base))
        return pcapa_layer, pcapa_en_layer, pkappa_layer

    first_pcapa, first_pcapa_en, first_pkappa = set_layer(0, jnp.asarray(0.0, dtype=jnp.float64))
    pcapa = pcapa.at[:, 0, :].set(first_pcapa)
    pcapa_en = pcapa_en.at[:, 0, :].set(first_pcapa_en)
    pkappa = pkappa.at[:, 0, :].set(first_pkappa)
    for jg in range(1, max(1, ngrnd - 2)):
        layer_pcapa, layer_pcapa_en, layer_pkappa = set_layer(jg, zlt[jg - 1])
        pcapa = pcapa.at[:, jg, :].set(layer_pcapa)
        pcapa_en = pcapa_en.at[:, jg, :].set(layer_pcapa_en)
        pkappa = pkappa.at[:, jg, :].set(layer_pkappa)

    if ngrnd >= 2:
        pcapa = pcapa.at[:, ngrnd - 2 : ngrnd, :].set(so_capa_dry)
        pcapa_en = pcapa_en.at[:, ngrnd - 2 : ngrnd, :].set(so_capa_dry)
        pkappa = pkappa.at[:, ngrnd - 2 : ngrnd, :].set(so_cond_dry)
    if int(brk_flag) == 1:
        pcapa = pcapa.at[:, ngrnd - 2 : ngrnd, :].set(brk_capa)
        pcapa_en = pcapa_en.at[:, ngrnd - 2 : ngrnd, :].set(brk_capa)
        pkappa = pkappa.at[:, ngrnd - 2 : ngrnd, :].set(brk_cond)

    snow_props = jnp.zeros((npts, 0), dtype=jnp.float64)
    zeros = jnp.zeros_like(pcapa)
    return ThermosoilGetdiffResult(
        pcapa=pcapa,
        pcapa_en=pcapa_en,
        pkappa=pkappa,
        profil_froz=zeros,
        pcappa_supp=zeros,
        pcapa_snow=snow_props,
        pkappa_snow=snow_props,
        poros_net=jnp.broadcast_to(smcmax_point[:, None, None], (npts, ngrnd, nvm)),
        zx1=jnp.zeros_like(pcapa),
        zx2=jnp.ones_like(pcapa),
        provenance=(*THERMOSOIL_GETDIFF_OLD_SNOW_PROVENANCE, *THERMOSOIL_COND_PROVENANCE),
    )


def thermosoil_coef_soil_no_snow(
    *,
    ptn,
    pcapa,
    pkappa,
    temp_sol_new,
    temp_sol_new_pft,
    veget_max,
    ok_laidev,
    dlt,
    dz1,
    dt_sechiba,
    lambda_thermal,
    phigeoth=PHIGEOTH,
    veget_mask_2d=None,
) -> ThermosoilCoefSoilResult:
    """Compute the soil part of ``thermosoil_coef`` for ``ok_explicitsnow=false``.

    Fortran provenance: ``thermosoil.f90::thermosoil_coef`` lines 1518-1628
    and 1722-1725. The caller supplies ``ptn``, ``pcapa``, and ``pkappa`` as
    explicit inputs from the audited upstream thermal-property path. This
    function deliberately does not approximate ``thermosoil_getdiff`` or the
    explicit-snow blend at lines 1633-1721.
    """

    ptn = _as_float64(ptn)
    pcapa = _as_float64(pcapa)
    pkappa = _as_float64(pkappa)
    temp_sol_new = _as_float64(temp_sol_new)
    temp_sol_new_pft = _as_float64(temp_sol_new_pft)
    veget_max = _as_float64(veget_max)
    ok_laidev = jnp.asarray(ok_laidev, dtype=bool)
    dlt = _as_float64(dlt)
    dz1 = _as_float64(dz1)
    mask = None if veget_mask_2d is None else jnp.asarray(veget_mask_2d, dtype=bool)
    _validate_shapes(
        ptn=ptn,
        pcapa=pcapa,
        pkappa=pkappa,
        temp_sol_new=temp_sol_new,
        temp_sol_new_pft=temp_sol_new_pft,
        veget_max=veget_max,
        ok_laidev=ok_laidev,
        dlt=dlt,
        dz1=dz1,
        veget_mask_2d=mask,
    )

    npts, ngrnd, nvm = ptn.shape
    if mask is None:
        mask = jnp.ones((npts, nvm), dtype=bool)
    mask3 = mask[:, None, :]

    dt = jnp.asarray(dt_sechiba, dtype=jnp.float64)
    lamb = jnp.asarray(lambda_thermal, dtype=jnp.float64)
    phigeoth = jnp.asarray(phigeoth, dtype=jnp.float64)

    zdz2 = pcapa * dlt[None, :, None] / dt
    zdz1 = pkappa[:, : ngrnd - 1, :] * dz1[None, :, None]
    zdz2 = jnp.where(mask3, zdz2, 0.0)
    zdz1 = jnp.where(mask[:, None, :], zdz1, 0.0)

    cgrnd = jnp.zeros((npts, ngrnd - 1, nvm), dtype=jnp.float64)
    dgrnd = jnp.zeros((npts, ngrnd - 1, nvm), dtype=jnp.float64)

    bottom_den = zdz2[:, ngrnd - 1, :] + zdz1[:, ngrnd - 2, :]
    bottom_c = (phigeoth + zdz2[:, ngrnd - 1, :] * ptn[:, ngrnd - 1, :]) / bottom_den
    bottom_d = zdz1[:, ngrnd - 2, :] / bottom_den
    cgrnd = cgrnd.at[:, ngrnd - 2, :].set(jnp.where(mask, bottom_c, 0.0))
    dgrnd = dgrnd.at[:, ngrnd - 2, :].set(jnp.where(mask, bottom_d, 0.0))

    for jg in range(ngrnd - 2, 0, -1):
        z1 = 1.0 / (
            zdz2[:, jg, :]
            + zdz1[:, jg - 1, :]
            + zdz1[:, jg, :] * (1.0 - dgrnd[:, jg, :])
        )
        next_c = (ptn[:, jg, :] * zdz2[:, jg, :] + zdz1[:, jg, :] * cgrnd[:, jg, :]) * z1
        next_d = zdz1[:, jg - 1, :] * z1
        cgrnd = cgrnd.at[:, jg - 1, :].set(jnp.where(mask, next_c, 0.0))
        dgrnd = dgrnd.at[:, jg - 1, :].set(jnp.where(mask, next_d, 0.0))

    surface_base = zdz1[:, 0, :] * (cgrnd[:, 0, :] + (dgrnd[:, 0, :] - 1.0) * ptn[:, 0, :])
    surface_cap = dt * (zdz2[:, 0, :] + (1.0 - dgrnd[:, 0, :]) * zdz1[:, 0, :])
    surface_z1 = lamb * (1.0 - dgrnd[:, 0, :]) + 1.0
    soilcap_pft_nosnow = surface_cap / surface_z1
    soilflx_pft_nosnow = surface_base + soilcap_pft_nosnow * (
        ptn[:, 0, :] * surface_z1 - lamb * cgrnd[:, 0, :] - temp_sol_new[:, None]
    ) / dt

    pft_surface_temp = jnp.where(ok_laidev[None, :], temp_sol_new_pft, temp_sol_new[:, None])
    soilcap_pft = surface_cap / surface_z1
    soilflx_pft = surface_base + soilcap_pft * (
        ptn[:, 0, :] * surface_z1 - lamb * cgrnd[:, 0, :] - pft_surface_temp
    ) / dt

    soilcap_pft = jnp.where(mask, soilcap_pft, 0.0)
    soilflx_pft = jnp.where(mask, soilflx_pft, 0.0)
    soilcap_pft_nosnow = jnp.where(mask, soilcap_pft_nosnow, 0.0)
    soilflx_pft_nosnow = jnp.where(mask, soilflx_pft_nosnow, 0.0)

    weights = jnp.where(mask, veget_max, 0.0)
    soilcap = jnp.sum(soilcap_pft_nosnow * weights, axis=1)
    soilflx = jnp.sum(soilflx_pft_nosnow * weights, axis=1)
    cgrnd_soil = jnp.sum(cgrnd[:, 0, :] * weights, axis=1)
    dgrnd_soil = jnp.sum(dgrnd[:, 0, :] * weights, axis=1)
    zdz1_soil = jnp.sum(zdz1[:, 0, :] * weights, axis=1)
    zdz2_soil = jnp.sum(zdz2[:, 0, :] * weights, axis=1)

    return ThermosoilCoefSoilResult(
        cgrnd=cgrnd,
        dgrnd=dgrnd,
        soilcap=soilcap,
        soilcap_pft=soilcap_pft,
        soilflx=soilflx,
        soilflx_pft=soilflx_pft,
        soilcap_pft_nosnow=soilcap_pft_nosnow,
        soilflx_pft_nosnow=soilflx_pft_nosnow,
        zdz1=zdz1,
        zdz2=zdz2,
        cgrnd_soil=cgrnd_soil,
        dgrnd_soil=dgrnd_soil,
        zdz1_soil=zdz1_soil,
        zdz2_soil=zdz2_soil,
    )


def thermosoil_coef_explicit_snow(
    *,
    ptn,
    ptn_pftmean,
    pcapa,
    pkappa,
    pcapa_snow,
    pkappa_snow,
    snowdz,
    snowtemp,
    temp_sol_new,
    temp_sol_new_pft,
    veget_max,
    ok_laidev,
    dlt,
    dz1,
    zlt,
    dt_sechiba,
    lambda_thermal,
    frac_snow_veg,
    frac_snow_nobio,
    totfrac_nobio,
    psnowdzmin=PSNOWDZMIN,
    phigeoth=PHIGEOTH,
    veget_mask_2d=None,
) -> ThermosoilCoefExplicitSnowResult:
    """Compute ``thermosoil_coef`` with active explicit-snow blending.

    Fortran provenance: soil coefficients follow ``thermosoil_coef`` lines
    1518-1628; explicit snow coefficients and final grid-cell blending follow
    lines 1633-1721. ``ptn_pftmean`` is explicit because ``thermosoil_main``
    builds it from ``ptn`` and ``veget_max_bg`` before this call.
    """

    soil = thermosoil_coef_soil_no_snow(
        ptn=ptn,
        pcapa=pcapa,
        pkappa=pkappa,
        temp_sol_new=temp_sol_new,
        temp_sol_new_pft=temp_sol_new_pft,
        veget_max=veget_max,
        ok_laidev=ok_laidev,
        dlt=dlt,
        dz1=dz1,
        dt_sechiba=dt_sechiba,
        lambda_thermal=lambda_thermal,
        phigeoth=phigeoth,
        veget_mask_2d=veget_mask_2d,
    )
    ptn_pftmean = _as_float64(ptn_pftmean)
    pcapa_snow = _as_float64(pcapa_snow)
    pkappa_snow = _as_float64(pkappa_snow)
    snowdz = _as_float64(snowdz)
    snowtemp = _as_float64(snowtemp)
    temp_sol_new = _as_float64(temp_sol_new)
    zlt = _as_float64(zlt)
    frac_snow_veg = _as_float64(frac_snow_veg)
    frac_snow_nobio = _as_float64(frac_snow_nobio)
    totfrac_nobio = _as_float64(totfrac_nobio)
    npts = soil.soilcap.shape[0]
    if pcapa_snow.ndim != 2:
        raise ValueError("pcapa_snow must have shape (npts, nsnow)")
    if pkappa_snow.shape != pcapa_snow.shape or snowdz.shape != pcapa_snow.shape or snowtemp.shape != pcapa_snow.shape:
        raise ValueError("pcapa_snow, pkappa_snow, snowdz, and snowtemp must share shape (npts, nsnow)")
    if pcapa_snow.shape[0] != npts:
        raise ValueError("snow arrays must have npts matching soil inputs")
    nsnow = pcapa_snow.shape[1]
    if nsnow < 2:
        raise ValueError("explicit snow coefficient path requires at least two snow layers")
    if ptn_pftmean.shape[0] != npts or ptn_pftmean.ndim != 2 or ptn_pftmean.shape[1] < 1:
        raise ValueError("ptn_pftmean must have shape (npts, ngrnd)")
    if zlt.ndim != 1 or zlt.shape[0] < 1:
        raise ValueError("zlt must be a non-empty 1D vector")
    if frac_snow_veg.shape != (npts,) or totfrac_nobio.shape != (npts,):
        raise ValueError("frac_snow_veg and totfrac_nobio must have shape (npts,)")
    if frac_snow_nobio.ndim != 2 or frac_snow_nobio.shape[0] != npts:
        raise ValueError("frac_snow_nobio must have shape (npts, nnobio)")

    dt = jnp.asarray(dt_sechiba, dtype=jnp.float64)
    dz2_snow = jnp.maximum(snowdz, jnp.asarray(psnowdzmin, dtype=jnp.float64))
    dz1_snow = jnp.zeros_like(dz2_snow)
    for jg in range(nsnow - 1):
        dz1_snow = dz1_snow.at[:, jg].set(2.0 / (dz2_snow[:, jg + 1] + dz2_snow[:, jg]))
    lambda_snow = dz2_snow[:, 0] / 2.0 * dz1_snow[:, 0]
    zdz2_snow = pcapa_snow * dz2_snow / dt
    zdz1_snow = jnp.zeros_like(dz2_snow)
    for jg in range(nsnow - 1):
        zdz1_snow = zdz1_snow.at[:, jg].set(dz1_snow[:, jg] * pkappa_snow[:, jg])
    zdz1_snow = zdz1_snow.at[:, nsnow - 1].set(pkappa_snow[:, nsnow - 1] / (zlt[0] + dz2_snow[:, nsnow - 1] / 2.0))

    cgrnd_snow = jnp.zeros_like(dz2_snow)
    dgrnd_snow = jnp.zeros_like(dz2_snow)
    z1_bottom = soil.zdz2_soil + (1.0 - soil.dgrnd_soil) * soil.zdz1_soil + zdz1_snow[:, nsnow - 1]
    c_bottom = (soil.zdz2_soil * ptn_pftmean[:, 0] + soil.zdz1_soil * soil.cgrnd_soil) / z1_bottom
    d_bottom = zdz1_snow[:, nsnow - 1] / z1_bottom
    cgrnd_snow = cgrnd_snow.at[:, nsnow - 1].set(c_bottom)
    dgrnd_snow = dgrnd_snow.at[:, nsnow - 1].set(d_bottom)

    z1_next = zdz2_snow[:, nsnow - 1] + (1.0 - dgrnd_snow[:, nsnow - 1]) * zdz1_snow[:, nsnow - 1] + zdz1_snow[:, nsnow - 2]
    c_next = (
        zdz2_snow[:, nsnow - 1] * snowtemp[:, nsnow - 1]
        + zdz1_snow[:, nsnow - 1] * cgrnd_snow[:, nsnow - 1]
    ) / z1_next
    d_next = zdz1_snow[:, nsnow - 2] / z1_next
    cgrnd_snow = cgrnd_snow.at[:, nsnow - 2].set(c_next)
    dgrnd_snow = dgrnd_snow.at[:, nsnow - 2].set(d_next)

    for jg in range(nsnow - 2, 0, -1):
        z1 = 1.0 / (
            zdz2_snow[:, jg]
            + zdz1_snow[:, jg - 1]
            + zdz1_snow[:, jg] * (1.0 - dgrnd_snow[:, jg])
        )
        c_val = (snowtemp[:, jg] * zdz2_snow[:, jg] + zdz1_snow[:, jg] * cgrnd_snow[:, jg]) * z1
        d_val = zdz1_snow[:, jg - 1] * z1
        cgrnd_snow = cgrnd_snow.at[:, jg - 1].set(c_val)
        dgrnd_snow = dgrnd_snow.at[:, jg - 1].set(d_val)

    snowflx = zdz1_snow[:, 0] * (cgrnd_snow[:, 0] + (dgrnd_snow[:, 0] - 1.0) * snowtemp[:, 0])
    snowcap = dt * (zdz2_snow[:, 0] + (1.0 - dgrnd_snow[:, 0]) * zdz1_snow[:, 0])
    z1_snow_surface = lambda_snow * (1.0 - dgrnd_snow[:, 0]) + 1.0
    snowcap = snowcap / z1_snow_surface
    snowflx = snowflx + snowcap * (
        snowtemp[:, 0] * z1_snow_surface - lambda_snow * cgrnd_snow[:, 0] - temp_sol_new
    ) / dt

    snow_veg_weight = frac_snow_veg * (1.0 - totfrac_nobio)
    snow_nobio_weight = jnp.sum(frac_snow_nobio, axis=1) * totfrac_nobio
    nonsnow_weight = 1.0 - (snow_veg_weight + snow_nobio_weight)
    soilcap = snowcap * snow_veg_weight + soil.soilcap * snow_nobio_weight + soil.soilcap * nonsnow_weight
    soilflx = snowflx * snow_veg_weight + soil.soilflx * snow_nobio_weight + soil.soilflx * nonsnow_weight

    return ThermosoilCoefExplicitSnowResult(
        soil=soil,
        lambda_snow=lambda_snow,
        cgrnd_snow=cgrnd_snow,
        dgrnd_snow=dgrnd_snow,
        snowcap=snowcap,
        snowflx=snowflx,
        dz1_snow=dz1_snow,
        dz2_snow=dz2_snow,
        zdz1_snow=zdz1_snow,
        zdz2_snow=zdz2_snow,
        soilcap=soilcap,
        soilflx=soilflx,
    )


def thermosoil_coef_no_explicit_snow(
    *,
    ptn,
    pcapa,
    pkappa,
    temp_sol_new,
    temp_sol_new_pft,
    veget_max,
    ok_laidev,
    dlt,
    dz1,
    dt_sechiba,
    lambda_thermal,
    cgrnd_snow_template=None,
    dgrnd_snow_template=None,
    nsnow=0,
    phigeoth=PHIGEOTH,
    veget_mask_2d=None,
) -> ThermosoilCoefExplicitSnowResult:
    """Compute ``thermosoil_coef`` for ``ok_explicitsnow=false``.

    Fortran provenance: ``thermosoil_coef`` lines 1499-1507 select the old
    thermix snow path; lines 1633-1725 set ``lambda_snow=lambda``, zero snow
    recurrence coefficients/fluxes, and pass through no-snow soilcap/soilflx.
    """

    soil = thermosoil_coef_soil_no_snow(
        ptn=ptn,
        pcapa=pcapa,
        pkappa=pkappa,
        temp_sol_new=temp_sol_new,
        temp_sol_new_pft=temp_sol_new_pft,
        veget_max=veget_max,
        ok_laidev=ok_laidev,
        dlt=dlt,
        dz1=dz1,
        dt_sechiba=dt_sechiba,
        lambda_thermal=lambda_thermal,
        phigeoth=phigeoth,
        veget_mask_2d=veget_mask_2d,
    )
    npts = soil.soilcap.shape[0]
    if cgrnd_snow_template is not None:
        cgrnd_snow = jnp.zeros_like(_as_float64(cgrnd_snow_template))
        if cgrnd_snow.ndim != 2 or cgrnd_snow.shape[0] != npts:
            raise ValueError("cgrnd_snow_template must have shape (npts, nsnow)")
        snow_shape = cgrnd_snow.shape
    else:
        snow_shape = (npts, int(nsnow))
        cgrnd_snow = jnp.zeros(snow_shape, dtype=jnp.float64)
    if dgrnd_snow_template is not None:
        dgrnd_snow = jnp.zeros_like(_as_float64(dgrnd_snow_template))
        if dgrnd_snow.shape != snow_shape:
            raise ValueError("dgrnd_snow_template must match cgrnd_snow_template")
    else:
        dgrnd_snow = jnp.zeros(snow_shape, dtype=jnp.float64)
    snow_zeros = jnp.zeros(snow_shape, dtype=jnp.float64)
    return ThermosoilCoefExplicitSnowResult(
        soil=soil,
        lambda_snow=jnp.full((npts,), jnp.asarray(lambda_thermal, dtype=jnp.float64)),
        cgrnd_snow=cgrnd_snow,
        dgrnd_snow=dgrnd_snow,
        snowcap=jnp.zeros((npts,), dtype=jnp.float64),
        snowflx=jnp.zeros((npts,), dtype=jnp.float64),
        dz1_snow=snow_zeros,
        dz2_snow=snow_zeros,
        zdz1_snow=snow_zeros,
        zdz2_snow=snow_zeros,
        soilcap=soil.soilcap,
        soilflx=soil.soilflx,
        provenance=THERMOSOIL_COEF_NO_EXPLICIT_SNOW_PROVENANCE,
    )


def thermosoil_final_state(*, ptn, pkappa, veget_max, shum_ngrnd_permalong):
    """Build final PFT-mean fields and downstream diagnostic payloads.

    Fortran provenance: ``thermosoil.f90::thermosoil_main`` lines 1011-1032.
    The bare-soil residual uses the same ``veget_max_bg`` rule as lines
    886-887.
    """

    ptn = _as_float64(ptn)
    pkappa = _as_float64(pkappa)
    shum_ngrnd_permalong = _as_float64(shum_ngrnd_permalong)
    if ptn.ndim != 3:
        raise ValueError("ptn must have shape (npts, ngrnd, nvm)")
    if pkappa.shape != ptn.shape or shum_ngrnd_permalong.shape != ptn.shape:
        raise ValueError("pkappa and shum_ngrnd_permalong must match ptn shape")
    veget_max_bg = thermosoil_veget_max_bg(veget_max)
    if veget_max_bg.shape != (ptn.shape[0], ptn.shape[2]):
        raise ValueError("veget_max must have shape (npts, nvm) matching ptn")
    ptn_pftmean = jnp.sum(ptn * veget_max_bg[:, None, :], axis=2)
    pkappa_pftmean = jnp.sum(pkappa * veget_max_bg[:, None, :], axis=2)
    return ThermosoilFinalStateResult(
        ptn_pftmean=ptn_pftmean,
        pkappa_pftmean=pkappa_pftmean,
        gtemp=ptn_pftmean[:, 0],
        ptnlev1=ptn_pftmean[:, 0],
        deephum_prof=shum_ngrnd_permalong,
        deeptemp_prof=ptn,
    )


def thermosoil_explicit_step(
    *,
    ptn,
    cgrnd,
    dgrnd,
    cgrnd_snow,
    dgrnd_snow,
    temp_sol_new,
    temp_sol_new_pft,
    snowrho,
    snowtemp,
    snowdz,
    shumdiag_perma,
    mc_layh,
    mcl_layh,
    tmc_layh,
    mc_layh_pft,
    mcl_layh_pft,
    tmc_layh_pft,
    njsc,
    veget_max,
    pb,
    dlt,
    dz1,
    zlt,
    znt,
    dz5,
    dt_sechiba,
    lambda_thermal,
    frac_snow_veg,
    frac_snow_nobio,
    totfrac_nobio,
    temp_sol_beg,
    soilcap_initial,
    pcapa_en_previous=None,
    soilc_total=None,
    refsoc=None,
    zx1=None,
    ok_laidev=None,
    veget_mask_2d=None,
    satsoil=False,
    shum_ngrnd_permalong_previous=None,
    ok_shum_ngrnd_permalong=True,
) -> ThermosoilExplicitStepResult:
    """Run the implemented explicit-input THERMOSOIL main sequence.

    Fortran provenance: ``thermosoil_main`` calls ``thermosoil_humlev`` at
    lines 893-894, ``thermosoil_profile`` at lines 906-909,
    ``thermosoil_energy`` at line 913, recomputes ``thermosoil_coef`` at
    lines 1002-1009, and emits final PFT means at lines 1011-1032.

    This function does not source restart, refSOC, or forcing data; callers
    must provide those active-run inputs explicitly.

    ``pcapa_en_previous`` is the coefficient state used by
    ``thermosoil_energy`` before this step's coefficient update. Production
    recurrence should use ``thermosoil_explicit_transition``, which requires
    that state and writes the replacement back explicitly.
    """

    humlev = thermosoil_humlev(
        shumdiag_perma=shumdiag_perma,
        mc_layh=mc_layh,
        mcl_layh=mcl_layh,
        tmc_layh=tmc_layh,
        mc_layh_pft=mc_layh_pft,
        mcl_layh_pft=mcl_layh_pft,
        tmc_layh_pft=tmc_layh_pft,
        znt=znt,
        zlt=zlt,
        dz5=dz5,
        veget_mask_2d=veget_mask_2d,
        satsoil=satsoil,
    )
    shum_ngrnd_permalong = humlev.shum_ngrnd_perma
    if ok_shum_ngrnd_permalong and shum_ngrnd_permalong_previous is not None:
        shum_ngrnd_permalong = thermosoil_wlupdate(
            ptn=ptn,
            hsd=humlev.shum_ngrnd_perma,
            hsdlong=shum_ngrnd_permalong_previous,
            veget_mask_2d=veget_mask_2d,
            dt_sechiba=dt_sechiba,
        ).hsdlong
    profile = thermosoil_profile_explicit_snow(
        ptn=ptn,
        cgrnd=cgrnd,
        dgrnd=dgrnd,
        cgrnd_snow=cgrnd_snow,
        dgrnd_snow=dgrnd_snow,
        temp_sol_new=temp_sol_new,
        snowtemp=snowtemp,
        veget_max=veget_max,
        frac_snow_veg=frac_snow_veg,
        frac_snow_nobio=frac_snow_nobio,
        totfrac_nobio=totfrac_nobio,
        veget_mask_2d=veget_mask_2d,
        nslm=_as_float64(shumdiag_perma).shape[1],
    )
    if zx1 is None:
        if soilc_total is not None or refsoc is not None:
            zx1, _ = thermosoil_soilc_tempdiff_fraction(
                soilc_total=soilc_total,
                refsoc=refsoc,
                nvm=profile.ptn.shape[2],
            )
        else:
            zx1 = jnp.zeros_like(profile.ptn)
    getdiff = thermosoil_getdiff_explicit(
        ptn=profile.ptn,
        njsc=njsc,
        veget_max=veget_max,
        shum_ngrnd_permalong=shum_ngrnd_permalong,
        mc_layt=humlev.mc_layt,
        tmc_layt_pft=humlev.tmc_layt_pft,
        snowrho=snowrho,
        snowtemp=snowtemp,
        pb=pb,
        dlt=dlt,
        zx1=zx1,
        ok_laidev=ok_laidev,
        veget_mask_2d=veget_mask_2d,
    )
    energy_pcapa_en = getdiff.pcapa_en if pcapa_en_previous is None else pcapa_en_previous
    energy = thermosoil_energy_diagnostics(
        temp_sol_new=temp_sol_new,
        temp_sol_beg=temp_sol_beg,
        soilcap=soilcap_initial,
        pcapa_en=energy_pcapa_en,
        veget_max=veget_max,
        ptn=profile.ptn,
        sn_capa=SN_CAPA,
    )
    precoef_final = thermosoil_final_state(
        ptn=profile.ptn,
        pkappa=getdiff.pkappa,
        veget_max=veget_max,
        shum_ngrnd_permalong=shum_ngrnd_permalong,
    )
    coef = thermosoil_coef_explicit_snow(
        ptn=profile.ptn,
        ptn_pftmean=precoef_final.ptn_pftmean,
        pcapa=getdiff.pcapa,
        pkappa=getdiff.pkappa,
        pcapa_snow=getdiff.pcapa_snow,
        pkappa_snow=getdiff.pkappa_snow,
        snowdz=snowdz,
        snowtemp=snowtemp,
        temp_sol_new=temp_sol_new,
        temp_sol_new_pft=temp_sol_new_pft,
        veget_max=veget_max,
        ok_laidev=ok_laidev if ok_laidev is not None else jnp.zeros(profile.ptn.shape[2], dtype=bool),
        dlt=dlt,
        dz1=dz1,
        zlt=zlt,
        dt_sechiba=dt_sechiba,
        lambda_thermal=lambda_thermal,
        frac_snow_veg=frac_snow_veg,
        frac_snow_nobio=frac_snow_nobio,
        totfrac_nobio=totfrac_nobio,
        veget_mask_2d=veget_mask_2d,
    )
    final = thermosoil_final_state(
        ptn=profile.ptn,
        pkappa=getdiff.pkappa,
        veget_max=veget_max,
        shum_ngrnd_permalong=shum_ngrnd_permalong,
    )
    return ThermosoilExplicitStepResult(
        humlev=humlev,
        profile=profile,
        energy=energy,
        getdiff=getdiff,
        coef=coef,
        final=final,
    )


def thermosoil_explicit_transition(
    *,
    state: ThermosoilRecurrenceState,
    **step_inputs,
) -> ThermosoilExplicitTransitionResult:
    """Run ``thermosoil_main`` with the previous coefficient state explicit.

    Fortran provenance: ``thermosoil_main`` lines 905-913 use previous-step
    ``cgrnd``, ``dgrnd``, ``soilcap``, ``pcapa_en``, and ``temp_sol_beg`` for
    profile and energy diagnostics.  Only afterwards do lines 995-1009 call
    ``thermosoil_coef`` to produce coefficients for the next step.  This
    transition enforces that recurrence and writes every consumed state field
    back explicitly; it never substitutes same-step ``getdiff.pcapa_en``.
    """

    forbidden = tuple(
        name
        for name in (
            "ptn",
            "cgrnd",
            "dgrnd",
            "cgrnd_snow",
            "dgrnd_snow",
            "lambda_snow",
            "pcapa_en_previous",
            "temp_sol_beg",
            "soilcap_initial",
        )
        if name in step_inputs
    )
    if forbidden:
        raise ValueError(f"recurrence fields must come only from state: {forbidden}")
    step = thermosoil_explicit_step(
        **step_inputs,
        ptn=state.ptn,
        cgrnd=state.cgrnd,
        dgrnd=state.dgrnd,
        cgrnd_snow=state.cgrnd_snow,
        dgrnd_snow=state.dgrnd_snow,
        pcapa_en_previous=state.pcapa_en,
        temp_sol_beg=state.temp_sol_beg,
        soilcap_initial=state.soilcap,
    )
    next_state = ThermosoilRecurrenceState(
        ptn=step.profile.ptn,
        cgrnd=step.coef.soil.cgrnd,
        dgrnd=step.coef.soil.dgrnd,
        cgrnd_snow=step.coef.cgrnd_snow,
        dgrnd_snow=step.coef.dgrnd_snow,
        lambda_snow=step.coef.lambda_snow,
        pcapa_en=step.getdiff.pcapa_en,
        temp_sol_beg=step.energy.temp_sol_beg,
        soilcap=step.coef.soilcap,
    )
    return ThermosoilExplicitTransitionResult(step=step, next_state=next_state)


def thermosoil_no_explicit_snow_step(
    *,
    ptn,
    cgrnd,
    dgrnd,
    cgrnd_snow,
    dgrnd_snow,
    temp_sol_new,
    temp_sol_new_pft,
    snow,
    shumdiag_perma,
    mc_layh,
    mcl_layh,
    tmc_layh,
    mc_layh_pft,
    mcl_layh_pft,
    tmc_layh_pft,
    njsc,
    veget_max,
    dlt,
    dz1,
    zlt,
    znt,
    dz5,
    dt_sechiba,
    lambda_thermal,
    temp_sol_beg,
    soilcap_initial,
    pcapa_en_previous=None,
    ok_laidev=None,
    veget_mask_2d=None,
    satsoil=False,
    shum_ngrnd_permalong_previous=None,
    ok_shum_ngrnd_permalong=False,
) -> ThermosoilNoExplicitSnowStepResult:
    """Run the implemented ``thermosoil_main`` subset with bucket snow.

    Fortran provenance: ``thermosoil_main`` lines 893-1032 with
    ``thermosoil_profile`` no-explicit-snow lines 1804-1835,
    ``thermosoil_getdiff_old_thermix_with_snow`` lines 2884-3031, and
    ``thermosoil_coef`` no-explicit-snow lines 1499-1507 and 1633-1725.
    """

    ptn = _as_float64(ptn)
    if ptn.ndim != 3:
        raise ValueError("ptn must have shape (npts, ngrnd, nvm)")
    nvm = ptn.shape[2]
    if ok_laidev is None:
        ok_laidev = jnp.zeros(nvm, dtype=bool)
    humlev = thermosoil_humlev(
        shumdiag_perma=shumdiag_perma,
        mc_layh=mc_layh,
        mcl_layh=mcl_layh,
        tmc_layh=tmc_layh,
        mc_layh_pft=mc_layh_pft,
        mcl_layh_pft=mcl_layh_pft,
        tmc_layh_pft=tmc_layh_pft,
        znt=znt,
        zlt=zlt,
        dz5=dz5,
        veget_mask_2d=veget_mask_2d,
        satsoil=satsoil,
    )
    shum_ngrnd_permalong = humlev.shum_ngrnd_perma
    if ok_shum_ngrnd_permalong and shum_ngrnd_permalong_previous is not None:
        shum_ngrnd_permalong = thermosoil_wlupdate(
            ptn=ptn,
            hsd=humlev.shum_ngrnd_perma,
            hsdlong=shum_ngrnd_permalong_previous,
            veget_mask_2d=veget_mask_2d,
            dt_sechiba=dt_sechiba,
        ).hsdlong
    profile = thermosoil_profile_no_explicit_snow(
        ptn=ptn,
        cgrnd=cgrnd,
        dgrnd=dgrnd,
        temp_sol_new=temp_sol_new,
        temp_sol_new_pft=temp_sol_new_pft,
        veget_max=veget_max,
        ok_laidev=ok_laidev,
        lambda_thermal=lambda_thermal,
        veget_mask_2d=veget_mask_2d,
        nslm=_as_float64(shumdiag_perma).shape[1],
    )
    getdiff = thermosoil_getdiff_old_thermix_with_snow(
        snow=snow,
        njsc=njsc,
        mc_layt=humlev.mc_layt,
        mcl_layt=humlev.mcl_layt,
        tmc_layt=humlev.tmc_layt,
        mc_layt_pft=humlev.mc_layt_pft,
        mcl_layt_pft=humlev.mcl_layt_pft,
        tmc_layt_pft=humlev.tmc_layt_pft,
        dlt=dlt,
        zlt=zlt,
        ok_laidev=ok_laidev,
    )
    energy = thermosoil_energy_diagnostics(
        temp_sol_new=temp_sol_new,
        temp_sol_beg=temp_sol_beg,
        soilcap=soilcap_initial,
        pcapa_en=getdiff.pcapa_en if pcapa_en_previous is None else pcapa_en_previous,
        veget_max=veget_max,
        ptn=profile.ptn,
        sn_capa=SN_CAPA,
    )
    coef = thermosoil_coef_no_explicit_snow(
        ptn=profile.ptn,
        pcapa=getdiff.pcapa,
        pkappa=getdiff.pkappa,
        temp_sol_new=temp_sol_new,
        temp_sol_new_pft=temp_sol_new_pft,
        veget_max=veget_max,
        ok_laidev=ok_laidev,
        dlt=dlt,
        dz1=dz1,
        dt_sechiba=dt_sechiba,
        lambda_thermal=lambda_thermal,
        cgrnd_snow_template=cgrnd_snow,
        dgrnd_snow_template=dgrnd_snow,
        veget_mask_2d=veget_mask_2d,
    )
    final = thermosoil_final_state(
        ptn=profile.ptn,
        pkappa=getdiff.pkappa,
        veget_max=veget_max,
        shum_ngrnd_permalong=shum_ngrnd_permalong,
    )
    return ThermosoilNoExplicitSnowStepResult(
        humlev=humlev,
        profile=profile,
        energy=energy,
        getdiff=getdiff,
        coef=coef,
        final=final,
    )


THERMOSOIL_EXPLICIT_STEP_JIT_STATIC_ARGNAMES = (
    "satsoil",
    "ok_shum_ngrnd_permalong",
)


THERMOSOIL_FIRST_STEP_MODULE_CLOSURE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1093-1118",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_initialize lines 379-381 and 534-565",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_main lines 893-1032",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_var_init lines 1295-1299",
)


def run_thermosoil_first_step_module(
    *,
    moisture,
    thermosoil_restart: ThermosoilRestartState,
    enerbil_payload: Mapping[str, object],
    condveg_result,
    snowrho=None,
    snowtemp=None,
    snowdz=None,
    snow=None,
    njsc,
    veget_max,
    pb,
    totfrac_nobio,
    dlt,
    dz1,
    zlt,
    znt,
    dz5,
    dt_sechiba,
    ok_laidev,
    ok_explicitsnow=True,
    satsoil=False,
    use_jit=False,
    ok_shum_ngrnd_permalong=True,
) -> ThermosoilFirstStepModuleClosure:
    """Run first-step THERMOSOIL from strict HYDROL/CONDVEG/ENERBIL inputs.

    The caller supplies already executed same-step HYDROL moisture,
    CONDVEG snow fractions, ENERBIL surface temperatures, restart recurrence
    fields, and vertical-grid constants. No refSOC map interpolation is
    attempted here: ``thermosoil_initialize`` first reads restart ``refSOC`` at
    lines 534-565, and the paper restart contains that field.
    """

    required_enerbil = ("temp_sol_new", "temp_sol_new_pft")
    missing = [f"enerbil_payload:{name}" for name in required_enerbil if name not in enerbil_payload]
    if bool(ok_explicitsnow) and thermosoil_restart.refsoc is None:
        missing.append("thermosoil_restart:refSOC")
    if bool(ok_explicitsnow):
        for name, value in (("snowrho", snowrho), ("snowtemp", snowtemp), ("snowdz", snowdz)):
            if value is None:
                missing.append(name)
    elif snow is None:
        missing.append("snow")
    if missing:
        return ThermosoilFirstStepModuleClosure(
            module=None,
            boundary_payload={},
            downstream_payload={},
            missing_inputs=tuple(dict.fromkeys(missing)),
            provenance=THERMOSOIL_FIRST_STEP_MODULE_CLOSURE_PROVENANCE,
            notes=("THERMOSOIL first-step execution was skipped because strict inputs are missing.",),
        )

    from jax_orchidee.sechiba.coupling import (
        assemble_thermosoil_explicit_kwargs,
        assemble_thermosoil_no_explicit_snow_kwargs,
        thermosoil_downstream_payload,
    )

    veget_mask_2d = getattr(thermosoil_restart, "veget_mask_2d", None)
    if veget_mask_2d is None:
        veget_mask_2d = jnp.ones_like(_as_float64(veget_max), dtype=bool)

    if bool(ok_explicitsnow):
        lambda_thermal = znt[0] * dz1[0]
        if not isinstance(lambda_thermal, core.Tracer):
            lambda_thermal = float(lambda_thermal)
        boundary = assemble_thermosoil_explicit_kwargs(
            moisture=moisture,
            temp_sol_new=enerbil_payload["temp_sol_new"],
            temp_sol_new_pft=enerbil_payload["temp_sol_new_pft"],
            snowrho=snowrho,
            snowtemp=snowtemp,
            snowdz=snowdz,
            ptn=thermosoil_restart.ptn,
            cgrnd=thermosoil_restart.cgrnd,
            dgrnd=thermosoil_restart.dgrnd,
            cgrnd_snow=thermosoil_restart.cgrnd_snow,
            dgrnd_snow=thermosoil_restart.dgrnd_snow,
            lambda_snow=thermosoil_restart.lambda_snow,
            njsc=njsc,
            veget_max=veget_max,
            pb=pb,
            dlt=dlt,
            dz1=dz1,
            zlt=zlt,
            znt=znt,
            dz5=dz5,
            dt_sechiba=dt_sechiba,
            lambda_thermal=lambda_thermal,
            frac_snow_veg=condveg_result.frac_snow_veg,
            frac_snow_nobio=condveg_result.frac_snow_nobio,
            totfrac_nobio=totfrac_nobio,
            temp_sol_beg=getattr(thermosoil_restart, "temp_sol_beg", thermosoil_restart.gtemp),
            soilcap_initial=enerbil_payload["soilcap"],
            pcapa_en_previous=getattr(thermosoil_restart, "pcapa_en", None),
            refsoc=thermosoil_restart.refsoc,
            ok_laidev=ok_laidev,
            veget_mask_2d=veget_mask_2d,
            satsoil=satsoil,
            shum_ngrnd_permalong_previous=getattr(
                thermosoil_restart, "shum_ngrnd_permalong", None
            ),
            ok_shum_ngrnd_permalong=ok_shum_ngrnd_permalong,
        )
        if bool(use_jit):
            module = _thermosoil_explicit_step_jit(**boundary.kwargs)
        else:
            module = thermosoil_explicit_step(**boundary.kwargs)
    else:
        boundary = assemble_thermosoil_no_explicit_snow_kwargs(
            moisture=moisture,
            temp_sol_new=enerbil_payload["temp_sol_new"],
            temp_sol_new_pft=enerbil_payload["temp_sol_new_pft"],
            snow=snow,
            ptn=thermosoil_restart.ptn,
            cgrnd=thermosoil_restart.cgrnd,
            dgrnd=thermosoil_restart.dgrnd,
            cgrnd_snow=thermosoil_restart.cgrnd_snow,
            dgrnd_snow=thermosoil_restart.dgrnd_snow,
            lambda_snow=thermosoil_restart.lambda_snow,
            njsc=njsc,
            veget_max=veget_max,
            dlt=dlt,
            dz1=dz1,
            zlt=zlt,
            znt=znt,
            dz5=dz5,
            dt_sechiba=dt_sechiba,
            lambda_thermal=float(znt[0] * dz1[0]),
            temp_sol_beg=getattr(thermosoil_restart, "temp_sol_beg", thermosoil_restart.gtemp),
            soilcap_initial=enerbil_payload["soilcap"],
            pcapa_en_previous=getattr(thermosoil_restart, "pcapa_en", None),
            ok_laidev=ok_laidev,
            veget_mask_2d=veget_mask_2d,
            satsoil=satsoil,
            shum_ngrnd_permalong_previous=getattr(
                thermosoil_restart, "shum_ngrnd_permalong", None
            ),
            ok_shum_ngrnd_permalong=ok_shum_ngrnd_permalong,
        )
        if bool(use_jit):
            module = _thermosoil_no_explicit_snow_step_jit(**boundary.kwargs)
        else:
            module = thermosoil_no_explicit_snow_step(**boundary.kwargs)
    downstream = thermosoil_downstream_payload(module)
    return ThermosoilFirstStepModuleClosure(
        module=module,
        boundary_payload=boundary.advertised_payload,
        downstream_payload=downstream,
        missing_inputs=(),
        provenance=THERMOSOIL_FIRST_STEP_MODULE_CLOSURE_PROVENANCE,
        notes=(
            "Runs THERMOSOIL first-step profile, energy diagnostics, getdiff, coefficient update, and downstream payload from exact first-step sources.",
        ),
    )


_thermosoil_explicit_step_jit = jit(
    thermosoil_explicit_step,
    static_argnames=THERMOSOIL_EXPLICIT_STEP_JIT_STATIC_ARGNAMES,
)
_thermosoil_no_explicit_snow_step_jit = jit(
    thermosoil_no_explicit_snow_step,
    static_argnames=THERMOSOIL_EXPLICIT_STEP_JIT_STATIC_ARGNAMES,
)


def thermosoil_coef_soil_coverage() -> ThermosoilCoefCoverage:
    """Return the audited scope of the currently implemented coefficient path."""

    return ThermosoilCoefCoverage(
        ok=True,
        covered_fields=(
            "mc_layt",
            "mcl_layt",
            "tmc_layt",
            "shum_ngrnd_perma",
            "pcapa",
            "pcapa_en",
            "pkappa",
            "pcappa_supp",
            "profil_froz",
            "pcapa_snow",
            "pkappa_snow",
            "ptn",
            "stempdiag",
            "surfheat_incr",
            "coldcont_incr",
            "cgrnd",
            "dgrnd",
            "soilcap",
            "soilcap_pft",
            "soilflx",
            "soilflx_pft",
            "lambda_snow",
            "cgrnd_snow",
            "dgrnd_snow",
            "snowcap",
            "snowflx",
            "ptn_pftmean",
            "pkappa_pftmean",
            "gtemp",
            "ptnlev1",
            "deephum_prof",
            "deeptemp_prof",
            "zdz1",
            "zdz2",
            "zdz1_snow",
            "zdz2_snow",
        ),
        missing_processes=(
            "thermosoil_wlupdate long-term wetness smoothing when OK_WETDIAGLONG is active",
            "add_heat_Zimov microbial heat perturbation when ok_zimov is active",
            "full thermosoil_main orchestration with real restart/trace inputs",
        ),
        branch="active CWRR/OK_EXPLICITSNOW THERMOSOIL kernels except snow coefficient blend",
        provenance=(
            *THERMOSOIL_HUMLEV_PROVENANCE,
            *THERMOSOIL_PROFILE_PROVENANCE,
            *THERMOSOIL_DIAGLEV_PROVENANCE,
            *THERMOSOIL_ENERGY_PROVENANCE,
            *THERMOSOIL_GETDIFF_PROVENANCE,
            *THERMOSOIL_COND_PROVENANCE,
            *THERMOSOIL_ORGANIC_FRACTION_PROVENANCE,
            *THERMOSOIL_COEF_SOIL_PROVENANCE,
            *THERMOSOIL_COEF_SNOW_PROVENANCE,
            *THERMOSOIL_MAIN_FINAL_STATE_PROVENANCE,
            *THERMOSOIL_VERTICAL_PROVENANCE,
            *THERMOSOIL_CONSTANT_PROVENANCE,
        ),
        notes=(
            "USE_SOILC_TEMPDIFF organic fraction is explicit via refsoc/soilc_total; no organic state is inferred.",
            "The module-level orchestration still needs real restart/trace payloads for end-to-end parity.",
        ),
    )
