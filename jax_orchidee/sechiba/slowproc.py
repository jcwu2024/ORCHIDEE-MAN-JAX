"""Source-backed SLOWPROC boundary and vegetation-state kernels.

This module covers local algebra around ``slowproc_main`` without running the
full STOMATE process tree. Callers must provide explicit post-STOMATE state
when using the surface-update helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, NamedTuple

from jax import config, core, lax

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jax_orchidee.driver.restart import read_restart_fields
from jax_orchidee.sechiba.diffuco import humcste_from_pft_to_mtc, humcste_use_from_humcste
from jax_orchidee.stomate.main import StomateMainPFT14Switches, stomate_main_pft14_step


MIN_SECHIBA = 1.0e-8
MIN_VEGFRAC = 1.0e-6
VAL_EXP = 999999.0

SLOWPROC_MAIN_NOBIO_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 937-963",
)
SLOWPROC_VEGET_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_veget lines 2820-2925",
)
SLOWPROC_LAI_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_lai lines 2949-3062",
)
SLOWPROC_MAIN_SURFACE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 1103-1121",
)
SLOWPROC_RESTART_ENTRY_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_init lines 1685-1707",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_init lines 2149-2172",
)
SLOWPROC_DERIVVAR_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_derivvar lines 2625-2673",
)
SLOWPROC_NO_LCC_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_init lines 1449-1452",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_init lines 1518-1536",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_init lines 1603-1643",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 943-955",
    "paper run protocol: GLUC_USE_AGE_CLASS=n and effective veget_update=0 when IMPOSE_VEG=y",
)
SLOWPROC_FIRE_DISABLED_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_init lines 1509-1576",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_init lines 2281-2294 and 2345-2381",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 925-933",
    "reference/case_001_071 run.def: FIRE_DISABLE=y",
)
SLOWPROC_DYN_PEAT_DISABLED_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 616-646",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 850-918",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3005-3016 and 3618-3628",
    "reference/case_001_071 run.def: DYN_PEAT=n",
)
SLOWPROC_DYNAMIC_PEAT_FRACTION_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 566-650 updates FirstTsYear/FirstTsMonth/growth_day/peatC_ok/PWT memories",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 850-918 updates veget_max_new for dynamic peat",
)
SLOWPROC_THERMOSOIL_ENTRY_INIT_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_init lines 2704-2732",
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_init lines 2808-2809",
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1109-1118 and 1198-1201",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 991-994",
)
SLOWPROC_STATIC_ENTRY_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_soilt lines 2190-2216",
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_var_init lines 4093-4107",
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_soil lines 535-587",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 991-997",
)
SLOWPROC_EROSION_DAILY_ZERO_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_init lines 2461-2467",
    "fortran_source/ORCHIDEE/src_sechiba/erosion.f90::erosion_main lines 633-653",
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1214-1224",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 1011-1013",
)
SLOWPROC_INIT_PFT14_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_init lines 1404-1576",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_init lines 1579-1804",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_init lines 1822-2034",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_init lines 2133-2277",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_init lines 2281-2534",
    "reference/case_001_071 run.def: NVM=14, IMPOSE_VEG=y, PFT14=1, FIRE_DISABLE=y, DYN_PEAT=n",
)
SLOWPROC_MAIN_PFT14_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 556-675",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 685-918",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 925-1074",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 1078-1134",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_init lines 2495-2534 initializes salinity/tide module state carried unchanged by slowproc_main",
)
SLOWPROC_MAIN_PFT14_GAPS = (
    "lines 685-843: external vegetation-map, age-class land-use, and rotation readers are rejected by fixed paper switches",
    "lines 616-675 and 850-918: active dynamic-peat PWT/fraction branches are rejected by fixed paper switches",
    "lines 1037-1068 and 1131-1134: XIOS/history writes are returned as pure diagnostics; file side effects are not performed",
)

MCS_FAO = (0.41, 0.43, 0.41)
MCS_USDA = (
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

SLOWPROC_RESTART_STOMATE_ENTRY_FIELDS = (
    "lai",
    "height",
    "frac_age",
    "veget",
    "veget_max",
    "frac_nobio",
)


class SlowprocStomateNobioBoundary(NamedTuple):
    """Non-biological fractions passed from SLOWPROC into STOMATE."""

    totfrac_nobio_lastyear: jnp.ndarray
    totfrac_nobio_new: jnp.ndarray
    provenance: tuple[str, ...] = SLOWPROC_MAIN_NOBIO_PROVENANCE


class SlowprocVegetResult(NamedTuple):
    """Updated fractions from ``slowproc_veget``."""

    frac_nobio: jnp.ndarray
    veget_max: jnp.ndarray
    veget: jnp.ndarray
    totfrac_nobio: jnp.ndarray
    soiltile: jnp.ndarray
    provenance: tuple[str, ...] = SLOWPROC_VEGET_PROVENANCE


class SlowprocLaiResult(NamedTuple):
    """LAI from ``slowproc_lai`` for explicit monthly or temperature inputs."""

    lai: jnp.ndarray
    provenance: tuple[str, ...] = SLOWPROC_LAI_PROVENANCE
    notes: tuple[str, ...] = (
        "READ_LAI=True requires caller-supplied laimap already interpolated to the model land points.",
        "This helper does not read or regrid LAI files; slowproc_interlai remains an external source boundary.",
    )


class SlowprocSurfaceUpdateResult(NamedTuple):
    """Surface state produced by the closed part of ``slowproc_main``."""

    vegetation: SlowprocVegetResult
    tot_bare_soil: jnp.ndarray
    provenance: tuple[str, ...] = SLOWPROC_MAIN_SURFACE_PROVENANCE
    notes: tuple[str, ...] = (
        "This helper assumes caller-supplied post-STOMATE lai/veget_max state.",
        "It does not run stomate_main, slowproc_lai, slowproc_derivvar, or slowproc_checkveget.",
    )


class SlowprocSurfaceTransitionResult(NamedTuple):
    """One ``slowproc_main`` surface transition with its time gate exposed."""

    surface: SlowprocSurfaceUpdateResult
    do_slow: bool
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 654-663",
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 1078-1121",
    )


class SlowprocRestartEntryState(NamedTuple):
    """SECHIBA restart fields used before ``stomate_main``."""

    lai: np.ndarray
    height: np.ndarray
    frac_age: np.ndarray
    veget: np.ndarray
    veget_max: np.ndarray
    frac_nobio: np.ndarray
    provenance: tuple[str, ...] = SLOWPROC_RESTART_ENTRY_PROVENANCE


class SlowprocDerivvarResult(NamedTuple):
    """Closed outputs from ``slowproc_derivvar``."""

    qsintmax: jnp.ndarray
    deadleaf_cover: jnp.ndarray
    assim_param: jnp.ndarray
    height: jnp.ndarray
    temp_growth: jnp.ndarray
    provenance: tuple[str, ...] = SLOWPROC_DERIVVAR_PROVENANCE


class SlowprocNoLccEntryState(NamedTuple):
    """Explicit no land-cover-change state at the STOMATE entry boundary."""

    veget_max_new: jnp.ndarray
    vegetnew_firstday: jnp.ndarray
    totfrac_nobio_new: jnp.ndarray
    glccNetLCC: jnp.ndarray
    glccSecondShift: jnp.ndarray
    glccPrimaryShift: jnp.ndarray
    harvest_matrix: jnp.ndarray
    bound_spa: jnp.ndarray
    provenance: tuple[str, ...] = SLOWPROC_NO_LCC_PROVENANCE


class SlowprocFireDisabledEntryState(NamedTuple):
    """SPITFIRE inputs when the paper case disables fire."""

    lightn: jnp.ndarray
    popd: jnp.ndarray
    read_observed_ba: bool
    observed_ba: jnp.ndarray
    humign: jnp.ndarray
    read_cf_fine: bool
    cf_fine: jnp.ndarray
    read_cf_coarse: bool
    cf_coarse: jnp.ndarray
    read_ratio_flag: bool
    ratio_flag: jnp.ndarray
    read_ratio: bool
    ratio: jnp.ndarray
    provenance: tuple[str, ...] = SLOWPROC_FIRE_DISABLED_PROVENANCE


class SlowprocDynPeatDisabledEntryState(NamedTuple):
    """Peat-update inputs for the ``DYN_PEAT=n`` paper-case branch."""

    sat_duration: jnp.ndarray
    provenance: tuple[str, ...] = SLOWPROC_DYN_PEAT_DISABLED_PROVENANCE
    notes: tuple[str, ...] = (
        "sat_duration is a branch-inactive dummy for DYN_PEAT=n; peat update and fwet_series indexing are not executed.",
    )


class SlowprocDynamicPeatFractionResult(NamedTuple):
    """Dynamic peat monthly target cover from ``slowproc_main``."""

    veget_max_new: jnp.ndarray
    update_peatfrac: bool
    peatC_ok: jnp.ndarray
    growth_day: jnp.ndarray
    GSL: jnp.ndarray
    sat_duration: jnp.ndarray
    precipitation_thissummer: jnp.ndarray
    precipitation_lastsummer: jnp.ndarray
    peatPET_thisyear: jnp.ndarray
    peatPET_lastyear: jnp.ndarray
    provenance: tuple[str, ...] = SLOWPROC_DYNAMIC_PEAT_FRACTION_PROVENANCE


class SlowprocThermosoilEntryInitState(NamedTuple):
    """THERMOSOIL/permafrost state that is source-backed at STOMATE entry."""

    tdeep: jnp.ndarray
    hsdeep: jnp.ndarray
    heat_Zimov: jnp.ndarray
    zz_deep: jnp.ndarray
    zz_coef_deep: jnp.ndarray
    provenance: tuple[str, ...] = SLOWPROC_THERMOSOIL_ENTRY_INIT_PROVENANCE
    notes: tuple[str, ...] = (
        "sfluxCH4_deep, sfluxCO2_deep, thawed_humidity, depth_organic_soil, and soilc_total are deliberately absent here.",
        "They are allocated near this state but not source-backed by these initialization lines as valid STOMATE entry values.",
    )


class SlowprocStaticEntryState(NamedTuple):
    """Static SLOWPROC/HYDROL inputs forwarded to ``stomate_main``."""

    fc_grazing: jnp.ndarray
    humcste_use: jnp.ndarray
    provenance: tuple[str, ...] = SLOWPROC_STATIC_ENTRY_PROVENANCE


class SlowprocErosionDailyZeroEntryState(NamedTuple):
    """Daily erosion deposition inputs with explicit zero initialization/reset."""

    sed_deposition_d: jnp.ndarray
    poc_deposition_d: jnp.ndarray
    provenance: tuple[str, ...] = SLOWPROC_EROSION_DAILY_ZERO_PROVENANCE
    notes: tuple[str, ...] = (
        "erodepth is not included: the apparent sechiba_init zero assignment is inside the allocation-error block in this source snapshot.",
    )


class SlowprocColdStartVegetationEntryState(NamedTuple):
    """No-restart SLOWPROC vegetation state before the first STOMATE call."""

    lai: jnp.ndarray
    height: jnp.ndarray
    frac_age: jnp.ndarray
    vegetation: SlowprocVegetResult
    tot_bare_soil: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_init lines 2153-2170",
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_init lines 2249-2277",
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_veget lines 2858-2921",
    )


class SlowprocInitGapError(NotImplementedError):
    """An active ``slowproc_init`` source branch outside the owned closure."""


class SlowprocInitPeatMemoryState(NamedTuple):
    """Restart-or-default peat memories initialized even when ``DYN_PEAT=n``."""

    peatPET_lastyear: jnp.ndarray
    precipitation_lastsummer: jnp.ndarray
    precipitation_thissummer: jnp.ndarray
    peatPET_thisyear: jnp.ndarray
    growth_day: jnp.ndarray
    GSL: jnp.ndarray
    summerp_long: jnp.ndarray
    summerpet_long: jnp.ndarray
    peatC: jnp.ndarray
    peatC_ok: jnp.ndarray


class SlowprocInitPft14Result(NamedTuple):
    """Closed paper-case outputs of ``slowproc_init`` for single-point PFT14."""

    found_restart: bool
    veget_update: int
    lcanop: int
    qsintcst: float
    dt_stomate: float
    PWT_lim: float
    PC_lim: float
    sat_gsl: float
    lai: jnp.ndarray
    height: jnp.ndarray
    frac_age: jnp.ndarray
    veget: jnp.ndarray
    veget_max: jnp.ndarray
    frac_nobio: jnp.ndarray
    totfrac_nobio: jnp.ndarray
    soiltile: jnp.ndarray
    tot_bare_soil: jnp.ndarray
    reinf_slope: jnp.ndarray
    njsc: jnp.ndarray
    clayfraction: jnp.ndarray
    sandfraction: jnp.ndarray
    siltfraction: jnp.ndarray
    bulk_density: jnp.ndarray
    soil_ph: jnp.ndarray
    poor_soils: jnp.ndarray
    fc_grazing: jnp.ndarray
    salinity: jnp.ndarray
    tide_height: jnp.ndarray
    peat: SlowprocInitPeatMemoryState
    no_lcc: SlowprocNoLccEntryState
    fire: SlowprocFireDisabledEntryState
    provenance: tuple[str, ...] = SLOWPROC_INIT_PFT14_PROVENANCE
    notes: tuple[str, ...] = (
        "NetCDF interpolation is an explicit input boundary for soil, salinity, and tide data.",
        "Only the audited single-point, PFT14 paper configuration is accepted; other active branches raise SlowprocInitGapError.",
        "veget_year is omitted: with IMPOSE_VEG=y, lines 1603-1644 do not assign this INTENT(out) value in the source snapshot.",
    )


@dataclass(frozen=True)
class SlowprocMainPft14Switches:
    """Structural switches fixed by the audited paper target."""

    nvm: int = 14
    map_pft_format: bool = True
    veget_update: int = 0
    rotation_update: int = 0
    use_age_class: bool = False
    ok_rotate: bool = False
    ok_stomate: bool = True
    ok_dgvm: bool = False
    dyn_peat: bool = False
    dynpeat_pwt: bool = False
    dynpeat_pc: bool = False
    judge_pc: bool = False
    fire_disable: bool = True


class SlowprocMainCalendarState(NamedTuple):
    """Calendar flags and mutable memories after lines 556-675."""

    date: int
    first_ts_year: bool
    first_ts_month: bool
    last_ts_day: bool
    end_of_year: bool
    do_slow: bool
    update_peatfrac: bool
    growth_day: jnp.ndarray
    peatC_ok: jnp.ndarray


class SlowprocMainPft14Result(NamedTuple):
    """Complete fixed-switch PFT14 writeback from one ``slowproc_main`` call."""

    calendar: SlowprocMainCalendarState
    stomate: object
    vegetation_state: dict[str, object]
    soil_state: dict[str, object]
    salinity: jnp.ndarray
    tide_height: jnp.ndarray
    slow_state: dict[str, object]
    diagnostics: dict[str, object]
    provenance: tuple[str, ...] = SLOWPROC_MAIN_PFT14_PROVENANCE
    gaps: tuple[str, ...] = SLOWPROC_MAIN_PFT14_GAPS


def _as_float64(value):
    return jnp.asarray(value, dtype=jnp.float64)


def _validate_fraction_arrays(frac_nobio, veget_max) -> tuple[jnp.ndarray, jnp.ndarray]:
    frac_nobio = _as_float64(frac_nobio)
    veget_max = _as_float64(veget_max)
    if frac_nobio.ndim != 2:
        raise ValueError("frac_nobio must have shape (npts, nnobio)")
    if veget_max.ndim != 2:
        raise ValueError("veget_max must have shape (npts, nvm)")
    if frac_nobio.shape[0] != veget_max.shape[0]:
        raise ValueError("frac_nobio and veget_max must share npts")
    return frac_nobio, veget_max


def slowproc_totfrac_nobio(frac_nobio):
    """Sum non-biological fractions.

    Fortran provenance: ``slowproc.f90::slowproc_main`` lines 937-940 and
    ``slowproc_veget`` lines 2899-2902.
    """

    frac_nobio = _as_float64(frac_nobio)
    if frac_nobio.ndim != 2:
        raise ValueError("frac_nobio must have shape (npts, nnobio)")
    return jnp.sum(frac_nobio, axis=1)


def read_slowproc_restart_entry_state(
    path,
    *,
    source_pft_layout=None,
    target_pft_layout=None,
) -> SlowprocRestartEntryState:
    """Read SECHIBA restart state needed by ``slowproc_main`` before STOMATE.

    Fortran provenance: ``slowproc.f90::slowproc_init`` lines 1685-1707 reads
    ``lai``, ``height``, and ``frac_age`` from the SECHIBA restart. Lines
    2149-2172 define missing-value fallbacks, but this reader deliberately
    requires the restart variables to be present and does not synthesize those
    fallbacks.
    """

    fields = read_restart_fields(
        path,
        SLOWPROC_RESTART_STOMATE_ENTRY_FIELDS,
        source_pft_layout=source_pft_layout,
        target_pft_layout=target_pft_layout,
    )
    frac_age = fields["frac_age"]
    if frac_age.ndim != 3:
        raise ValueError("frac_age must have shape (npts, nleafages, nvm) in the SECHIBA restart")
    return SlowprocRestartEntryState(
        lai=fields["lai"],
        height=fields["height"],
        frac_age=np.transpose(frac_age, (0, 2, 1)),
        veget=fields["veget"],
        veget_max=fields["veget_max"],
        frac_nobio=fields["frac_nobio"],
    )


def slowproc_derivvar_explicit(
    *,
    veget,
    lai,
    vcmax_fix,
    height_presc,
    qsintcst,
) -> SlowprocDerivvarResult:
    """Run the closed ``slowproc_derivvar`` initialization formulas.

    Fortran provenance: ``slowproc.f90::slowproc_derivvar`` lines 2649-2673
    sets ``assim_param(:,:,ivcmax)=vcmax_fix``, zeroes ``deadleaf_cover``,
    sets ``height`` from ``height_presc``, computes ``qsintmax = qsintcst *
    veget * lai``, forces bare-soil ``qsintmax(:,1)=0``, and sets
    ``temp_growth=25``.
    """

    veget = _as_float64(veget)
    lai = _as_float64(lai)
    vcmax_fix = _as_float64(vcmax_fix)
    height_presc = _as_float64(height_presc)
    qsintcst = _as_float64(qsintcst)
    if veget.ndim != 2 or lai.shape != veget.shape:
        raise ValueError("veget and lai must share shape (npts, nvm)")
    npts, nvm = veget.shape
    if vcmax_fix.shape != (nvm,):
        raise ValueError("vcmax_fix must have shape (nvm,)")
    if height_presc.shape != (nvm,):
        raise ValueError("height_presc must have shape (nvm,)")

    assim_param = jnp.broadcast_to(vcmax_fix[None, :, None], (npts, nvm, 1))
    height = jnp.broadcast_to(height_presc[None, :], (npts, nvm))
    deadleaf_cover = jnp.zeros((npts,), dtype=veget.dtype)
    temp_growth = jnp.full((npts,), 25.0, dtype=veget.dtype)
    qsintmax = qsintcst * veget * lai
    qsintmax = qsintmax.at[:, 0].set(0.0)
    return SlowprocDerivvarResult(
        qsintmax=qsintmax,
        deadleaf_cover=deadleaf_cover,
        assim_param=assim_param,
        height=height,
        temp_growth=temp_growth,
    )


def slowproc_no_lcc_entry_state(
    *,
    veget_max,
    use_age_class,
    veget_update,
    map_pft_format=True,
    impose_veg=False,
    dtype=jnp.float64,
) -> SlowprocNoLccEntryState:
    """Build explicit no-LCC arrays passed from ``slowproc_main`` to STOMATE.

    This helper covers the paper-case branch where land-cover update is
    inactive after ``slowproc_init`` configuration and age-class GLUC is
    disabled. Fortran
    provenance: ``slowproc_init`` lines 1449-1452 initializes
    ``vegetnew_firstday`` to bare-soil one and other PFTs zero; lines
    1518-1536 initialize GLUC matrices and ``bound_spa`` to zero;
    lines 1603-1643 parse ``VEGET_UPDATE`` only when
    ``map_pft_format .AND. .NOT. impveg`` and otherwise set
    ``veget_update=0``;
    ``slowproc_main`` lines 943-955 sets ``totfrac_nobio_new`` to zero when
    ``do_now_stomate_lcchange`` is false.
    """

    if bool(use_age_class):
        raise ValueError("age-class LCC state must be supplied from GLUC sources")
    effective_update = 0
    if bool(map_pft_format) and not bool(impose_veg):
        text = str(veget_update).strip().upper().replace(" ", "")
        if text.endswith("Y"):
            text = text[:-1]
        effective_update = int(text or "0")
    if effective_update != 0:
        raise ValueError("nonzero vegetation update requires explicit LCC map state")
    veget_max = jnp.asarray(veget_max, dtype=dtype)
    if veget_max.ndim != 2:
        raise ValueError("veget_max must have shape (npts, nvm)")
    npts, nvm = veget_max.shape
    vegetnew_firstday = jnp.zeros((npts, nvm), dtype=dtype).at[:, 0].set(1.0)
    return SlowprocNoLccEntryState(
        veget_max_new=veget_max,
        vegetnew_firstday=vegetnew_firstday,
        totfrac_nobio_new=jnp.zeros((npts,), dtype=dtype),
        glccNetLCC=jnp.zeros((npts, 12), dtype=dtype),
        glccSecondShift=jnp.zeros((npts, 12), dtype=dtype),
        glccPrimaryShift=jnp.zeros((npts, 12), dtype=dtype),
        harvest_matrix=jnp.zeros((npts, 12), dtype=dtype),
        bound_spa=jnp.zeros((npts, nvm), dtype=dtype),
    )


def slowproc_fire_disabled_entry_state(
    *,
    kjpindex,
    fire_disable,
    dtype=jnp.float64,
) -> SlowprocFireDisabledEntryState:
    """Build explicit SPITFIRE no-op inputs for ``FIRE_DISABLE=y``.

    Fortran provenance: ``slowproc_init`` lines 1509-1576 allocates these
    arrays and initializes them to zero. Lines 2281-2294 show fire input data
    are read only when ``ok_stomate .AND. .NOT.disable_fire``. The paper case
    has ``FIRE_DISABLE=y``, so the initialized zero arrays and false read flags
    are the source-backed entry values used by ``slowproc_main`` lines 925-933.
    """

    if not bool(fire_disable):
        raise ValueError("fire-enabled runs require explicit SPITFIRE input data")
    kjpindex = int(kjpindex)
    zeros = jnp.zeros((kjpindex,), dtype=dtype)
    return SlowprocFireDisabledEntryState(
        lightn=zeros,
        popd=zeros,
        read_observed_ba=False,
        observed_ba=zeros,
        humign=zeros,
        read_cf_fine=False,
        cf_fine=zeros,
        read_cf_coarse=False,
        cf_coarse=zeros,
        read_ratio_flag=False,
        ratio_flag=zeros,
        read_ratio=False,
        ratio=zeros,
    )


def slowproc_dyn_peat_disabled_entry_state(
    *,
    kjpindex,
    dyn_peat,
    dtype=jnp.int32,
) -> SlowprocDynPeatDisabledEntryState:
    """Build branch-inactive peat entry input for ``DYN_PEAT=n``.

    Fortran provenance: ``slowproc_main`` lines 616-646 computes
    ``sat_duration`` only inside ``IF (dyn_peat)``. Lines 850-918 and
    ``stomate_main`` lines 3005-3016 and 3618-3628 consume peat-update state
    only inside dynamic-peat branches. For the paper-case ``DYN_PEAT=n`` this
    helper provides a shape-valid dummy while recording that the value is not a
    claimed peat-process result.
    """

    if bool(dyn_peat):
        raise ValueError("dynamic peat runs require explicit GSL-derived sat_duration")
    return SlowprocDynPeatDisabledEntryState(
        sat_duration=jnp.zeros((int(kjpindex),), dtype=dtype),
    )


def slowproc_dynamic_peat_fraction_step(
    *,
    veget_max,
    veget_max_new,
    fpeat,
    is_peat,
    month,
    day,
    sec,
    dt_sechiba,
    dyn_peat,
    dynpeat_PWT,
    dynpeat_PC,
    judge_pc=False,
    peatC=None,
    PC_lim=0.0,
    peatC_ok=None,
    temp_growth=None,
    growth_day=None,
    precipitation_thissummer=None,
    precipitation_lastsummer=None,
    peatPET_thisyear=None,
    peatPET_lastyear=None,
    precip_rain=None,
    precip_snow=None,
    peat_PET=None,
    GSL=None,
    sat_gsl=1.0,
    PWT_lim=0.0,
    min_vegfrac=MIN_VEGFRAC,
    min_stomate=0.0,
    ini_peat=1.0e-6,
) -> SlowprocDynamicPeatFractionResult:
    """Update monthly dynamic-peat target cover from ``slowproc_main``.

    Fortran provenance: ``src_sechiba/slowproc.f90::slowproc_main`` lines
    566-650 for calendar/PWT/PC memories and lines 850-918 for
    ``veget_max_new`` peat-PFT target updates. This helper stops before
    ``stomate_lpj.f90::lpj_cover_peat`` pool redistribution.
    """

    if not bool(dyn_peat):
        raise ValueError("dynamic peat inactive path is handled by slowproc_dyn_peat_disabled_entry_state")

    veget_max = _as_float64(veget_max)
    veget_max_new = _as_float64(veget_max_new)
    fpeat = _as_float64(fpeat)
    is_peat = jnp.asarray(is_peat, dtype=bool)
    if veget_max.ndim != 2 or veget_max_new.shape != veget_max.shape:
        raise ValueError("veget_max and veget_max_new must have shape (npts, nvm)")
    npts, nvm = veget_max.shape
    if fpeat.shape != (npts,) or is_peat.shape != (nvm,):
        raise ValueError("fpeat must have shape (npts,), and is_peat must have shape (nvm,)")

    def _require_vector(name, value):
        if value is None:
            raise ValueError(f"{name} is required for active dynamic peat")
        arr = _as_float64(value)
        if arr.shape != (npts,):
            raise ValueError(f"{name} must have shape (npts,)")
        return arr

    peatC_ok = _require_vector("peatC_ok", peatC_ok)
    growth_day = _require_vector("growth_day", growth_day)
    GSL = _require_vector("GSL", GSL)
    precipitation_thissummer = _require_vector("precipitation_thissummer", precipitation_thissummer)
    precipitation_lastsummer = _require_vector("precipitation_lastsummer", precipitation_lastsummer)
    peatPET_thisyear = _require_vector("peatPET_thisyear", peatPET_thisyear)
    peatPET_lastyear = _require_vector("peatPET_lastyear", peatPET_lastyear)
    temp_growth = _require_vector("temp_growth", temp_growth)

    first_ts_year = (sec == dt_sechiba) and (int(month) == 1) and (int(day) == 1)
    end_of_year = (sec == 0) and (int(month) == 1) and (int(day) == 1)
    last_ts_day = sec == 0
    first_ts_month = (sec == dt_sechiba) and (int(day) == 1)

    if first_ts_year:
        growth_day = jnp.zeros_like(growth_day)

    if (sec == dt_sechiba) and (int(month) == 1) and (int(day) == 2):
        if bool(dynpeat_PC):
            if bool(judge_pc):
                peatC = _require_vector("peatC", peatC)
                peatC_ok = jnp.where(peatC >= PC_lim, 1.0, 0.0)
        else:
            peatC_ok = jnp.ones_like(peatC_ok)

    if last_ts_day:
        growth_day = growth_day + jnp.where(temp_growth > 5.0, 1.0, 0.0)

    if bool(dynpeat_PWT):
        precip_rain = _require_vector("precip_rain", precip_rain)
        precip_snow = _require_vector("precip_snow", precip_snow)
        peat_PET = _require_vector("peat_PET", peat_PET)
        if 5 <= int(month) <= 9:
            precipitation_thissummer = precipitation_thissummer + precip_rain + precip_snow
            peatPET_thisyear = peatPET_thisyear + peat_PET
        if end_of_year:
            peatPET_lastyear = peatPET_thisyear
            peatPET_thisyear = jnp.zeros_like(peatPET_thisyear)
            precipitation_lastsummer = precipitation_thissummer
            precipitation_thissummer = jnp.zeros_like(precipitation_thissummer)
            GSL = growth_day / 365.0 * sat_gsl

    sat_duration = jnp.minimum(360.0, jnp.maximum(1.0, 360.0 - 30.0 * 12.0 * GSL)).astype(jnp.int32)
    update_peatfrac = bool(first_ts_month)
    if update_peatfrac:
        for pft in range(nvm):
            if not bool(is_peat[pft]):
                continue
            current = veget_max[:, pft]
            target = fpeat
            expand = target > current
            if bool(dynpeat_PWT):
                water_ok = (precipitation_lastsummer - peatPET_lastyear) >= PWT_lim
                first_initiation = current <= 0.0
                expand_value = jnp.where(
                    target > min_vegfrac,
                    jnp.where(water_ok, jnp.where(first_initiation | (peatC_ok > 0.0), target, current), current),
                    current,
                )
                contract_value = jnp.where(target > min_stomate, target, 0.0)
            else:
                expand_value = jnp.where(target > ini_peat, target, current)
                contract_value = jnp.where(target > min_stomate, target, 0.0)
            veget_max_new = veget_max_new.at[:, pft].set(jnp.where(expand, expand_value, contract_value))

    return SlowprocDynamicPeatFractionResult(
        veget_max_new=veget_max_new,
        update_peatfrac=update_peatfrac,
        peatC_ok=peatC_ok,
        growth_day=growth_day,
        GSL=GSL,
        sat_duration=sat_duration,
        precipitation_thissummer=precipitation_thissummer,
        precipitation_lastsummer=precipitation_lastsummer,
        peatPET_thisyear=peatPET_thisyear,
        peatPET_lastyear=peatPET_lastyear,
    )


def slowproc_thermosoil_entry_init_state(
    *,
    kjpindex,
    nvm,
    znt,
    zlt,
    dtype=jnp.float64,
) -> SlowprocThermosoilEntryInitState:
    """Build source-backed THERMOSOIL inputs at the SLOWPROC/STOMATE boundary.

    Fortran provenance: ``sechiba_init`` lines 2704-2715 initializes
    ``tdeep=250``, ``hsdeep=1``, and ``heat_Zimov=0``; lines 2808-2809 copy
    vertical vectors ``zz_deep=znt`` and ``zz_coef_deep=zlt``. ``sechiba_main``
    lines 1109-1118 lets ``thermosoil_main`` update these arrays before
    ``slowproc_main`` forwards them at lines 1198-1201. This helper therefore
    represents the source-backed initial boundary state; callers with an
    audited ``thermosoil_main`` result should pass that updated state instead.
    """

    kjpindex = int(kjpindex)
    nvm = int(nvm)
    znt = jnp.asarray(znt, dtype=dtype)
    zlt = jnp.asarray(zlt, dtype=dtype)
    if znt.ndim != 1 or zlt.ndim != 1:
        raise ValueError("znt and zlt must be one-dimensional vertical vectors")
    if znt.shape != zlt.shape:
        raise ValueError("znt and zlt must share shape")
    if kjpindex < 1 or nvm < 1:
        raise ValueError("kjpindex and nvm must be positive")
    shape = (kjpindex, int(znt.shape[0]), nvm)
    return SlowprocThermosoilEntryInitState(
        tdeep=jnp.full(shape, 250.0, dtype=dtype),
        hsdeep=jnp.ones(shape, dtype=dtype),
        heat_Zimov=jnp.zeros(shape, dtype=dtype),
        zz_deep=znt,
        zz_coef_deep=zlt,
    )


def slowproc_static_entry_state(
    *,
    njsc,
    soil_classif,
    pft_to_mtc,
    zmaxh,
    hydrol_humcste=None,
    dtype=jnp.float64,
) -> SlowprocStaticEntryState:
    """Build ``fc_grazing`` and ``humcste_use`` for the STOMATE entry.

    Fortran provenance: ``slowproc_soilt`` lines 2190-2216 selects
    ``fc_grazing`` from ``mcs_fao`` or ``mcs_usda`` using one-based ``njsc``.
    ``hydrol_var_init`` lines 4093-4107 copies PFT ``humcste`` into
    ``humcste_use(kjpindex,nvm)``, and ``slowproc_main`` lines 991-997 forwards
    both arrays to ``stomate_main``.
    """

    njsc = jnp.asarray(njsc, dtype=jnp.int32)
    if njsc.ndim != 1:
        raise ValueError("njsc must have shape (npts,)")
    key = str(soil_classif).strip().lower()
    if key in {"none", "zobler"}:
        table = jnp.asarray(MCS_FAO, dtype=dtype)
    elif key == "usda":
        table = jnp.asarray(MCS_USDA, dtype=dtype)
    else:
        raise ValueError("soil_classif must be one of 'none', 'zobler', or 'usda'")
    if bool(jnp.any(njsc < 1)) or bool(jnp.any(njsc > table.shape[0])):
        raise ValueError("njsc contains one-based soil class indices outside the selected soil table")

    humcste = humcste_from_pft_to_mtc(
        pft_to_mtc,
        zmaxh=zmaxh,
        hydrol_humcste=hydrol_humcste,
    )
    return SlowprocStaticEntryState(
        fc_grazing=table[njsc - 1],
        humcste_use=humcste_use_from_humcste(humcste, npts=int(njsc.shape[0])),
    )


def slowproc_erosion_daily_zero_entry_state(
    *,
    kjpindex,
    ncarb=3,
    dtype=jnp.float64,
) -> SlowprocErosionDailyZeroEntryState:
    """Build daily erosion deposition inputs with explicit zero sources.

    Fortran provenance: ``sechiba_init`` lines 2461-2467 initializes
    ``sed_deposition_d`` and ``poc_deposition_d`` to zero. ``erosion_main``
    lines 633-653 resets those daily accumulators to zero after output. This
    does not claim ``erodepth`` parity; that field must come from a real
    erosion process state or audited trace.
    """

    kjpindex = int(kjpindex)
    ncarb = int(ncarb)
    if kjpindex < 1 or ncarb < 1:
        raise ValueError("kjpindex and ncarb must be positive")
    return SlowprocErosionDailyZeroEntryState(
        sed_deposition_d=jnp.zeros((kjpindex,), dtype=dtype),
        poc_deposition_d=jnp.zeros((kjpindex, ncarb), dtype=dtype),
    )


def slowproc_stomate_nobio_boundary(
    *,
    frac_nobio_lastyear,
    do_now_stomate_lcchange,
    use_age_class,
    frac_nobio_new=None,
) -> SlowprocStomateNobioBoundary:
    """Compute the two non-biological fractions passed into ``stomate_main``.

    Fortran provenance: ``slowproc.f90::slowproc_main`` lines 937-963. If land
    cover change is inactive, ``totfrac_nobio_new`` is zero. If it is active
    and ``use_age_class`` is true, ``totfrac_nobio_new`` reuses last year's
    value; otherwise it is the sum of explicit ``frac_nobio_new``.
    """

    lastyear = slowproc_totfrac_nobio(frac_nobio_lastyear)
    if bool(do_now_stomate_lcchange):
        if bool(use_age_class):
            new = lastyear
        else:
            if frac_nobio_new is None:
                raise ValueError("frac_nobio_new is required when land-cover change is active without age classes")
            new = slowproc_totfrac_nobio(frac_nobio_new)
            if new.shape != lastyear.shape:
                raise ValueError("frac_nobio_new must share npts with frac_nobio_lastyear")
    else:
        new = jnp.zeros_like(lastyear)
    return SlowprocStomateNobioBoundary(
        totfrac_nobio_lastyear=lastyear,
        totfrac_nobio_new=new,
    )


def slowproc_veget_explicit(
    *,
    lai,
    frac_nobio,
    veget_max,
    pref_soil_veg,
    ext_coeff_vegetfrac,
    nstm: int,
    ok_dgvm=False,
    min_vegfrac=MIN_VEGFRAC,
    min_sechiba=MIN_SECHIBA,
) -> SlowprocVegetResult:
    """Run the source-backed fraction update from ``slowproc_veget``.

    Fortran provenance: ``slowproc.f90::slowproc_veget`` lines 2858-2921:
    small-fraction cleanup, normalization, LAI-to-``veget`` conversion,
    ``totfrac_nobio`` summation, and soil-tile aggregation with Fortran
    one-based ``pref_soil_veg`` indices.
    """

    lai = _as_float64(lai)
    frac_nobio, veget_max = _validate_fraction_arrays(frac_nobio, veget_max)
    if lai.shape != veget_max.shape:
        raise ValueError("lai must have the same shape as veget_max")
    npts, nvm = veget_max.shape
    pref_soil_veg = jnp.asarray(pref_soil_veg, dtype=jnp.int32)
    ext_coeff_vegetfrac = _as_float64(ext_coeff_vegetfrac)
    if pref_soil_veg.shape != (nvm,):
        raise ValueError("pref_soil_veg must have shape (nvm,)")
    if ext_coeff_vegetfrac.shape != (nvm,):
        raise ValueError("ext_coeff_vegetfrac must have shape (nvm,)")
    if nstm < 1:
        raise ValueError("nstm must be positive")
    if not isinstance(pref_soil_veg, core.Tracer) and (
        bool(jnp.any(pref_soil_veg < 1))
        or bool(jnp.any(pref_soil_veg > nstm))
    ):
        raise ValueError("pref_soil_veg must contain Fortran 1-based soil tile indices in 1..nstm")

    min_vegfrac = jnp.asarray(min_vegfrac, dtype=jnp.float64)
    min_sechiba = jnp.asarray(min_sechiba, dtype=jnp.float64)
    frac_sum = jnp.sum(frac_nobio, axis=1)
    frac_clean = jnp.where((frac_sum < min_vegfrac)[:, None], 0.0, frac_nobio)
    if not bool(ok_dgvm):
        veget_max_clean = jnp.where(veget_max < min_vegfrac, 0.0, veget_max)
    else:
        veget_max_clean = veget_max

    total = jnp.sum(frac_clean, axis=1) + jnp.sum(veget_max_clean, axis=1)
    if not isinstance(total, core.Tracer) and bool(jnp.any(total <= 0.0)):
        raise ValueError("frac_nobio + veget_max sum must be positive for every point")
    frac_norm = frac_clean / total[:, None]
    veget_max_norm = veget_max_clean / total[:, None]

    veget = veget_max_norm * (1.0 - jnp.exp(-lai * ext_coeff_vegetfrac[None, :]))
    veget = veget.at[:, 0].set(veget_max_norm[:, 0])

    totfrac_nobio = slowproc_totfrac_nobio(frac_norm)
    soiltile = jnp.zeros((npts, int(nstm)), dtype=jnp.float64)

    def add_pft(jv, current):
        jst = pref_soil_veg[jv] - 1
        return current.at[:, jst].add(veget_max_norm[:, jv])

    soiltile = lax.fori_loop(0, nvm, add_pft, soiltile)
    denom = 1.0 - totfrac_nobio
    soiltile = jnp.where(
        (totfrac_nobio < (1.0 - min_sechiba))[:, None],
        soiltile / denom[:, None],
        soiltile,
    )

    return SlowprocVegetResult(
        frac_nobio=frac_norm,
        veget_max=veget_max_norm,
        veget=veget,
        totfrac_nobio=totfrac_nobio,
        soiltile=soiltile,
    )


def _normalize_lai_type(label) -> str:
    text = str(label).strip().lower()
    if text == "mean":
        return "mean"
    if text == "inter":
        return "inter"
    raise ValueError(f"unsupported type_of_lai value {label!r}")


def slowproc_lai_explicit(
    *,
    read_lai: bool,
    type_of_lai,
    month: int,
    day: int,
    npts: int | None = None,
    laimap=None,
    stempdiag=None,
    lcanop: int | None = None,
    llaimax=None,
    llaimin=None,
    tempfunc_values=None,
) -> SlowprocLaiResult:
    """Compute ``slowproc_lai`` from explicit source inputs.

    Fortran provenance: ``slowproc.f90::slowproc_lai`` lines 2949-3062.
    The ``READ_LAI`` branch consumes a monthly ``laimap(kjpindex,nvm,12)``
    payload that must already have been produced by the source
    ``slowproc_interlai``/restart path. The non-``READ_LAI`` ``inter`` branch
    accepts explicit ``tempfunc_values`` to keep the temperature response
    source-owned rather than reimplemented here without its own audit.
    """

    type_array = tuple(_normalize_lai_type(label) for label in type_of_lai)
    nvm = len(type_array)
    month = int(month)
    day = int(day)
    if month < 1 or month > 12:
        raise ValueError("month must be a Fortran 1-based month in 1..12")
    if day < 1 or day > 31:
        raise ValueError("day must be a Fortran calendar day in 1..31")

    if bool(read_lai):
        if laimap is None:
            raise ValueError("READ_LAI=True requires explicit laimap")
        laimap_arr = jnp.asarray(laimap, dtype=jnp.float64)
        if laimap_arr.ndim != 3 or laimap_arr.shape[1:] != (nvm, 12):
            raise ValueError("laimap must have shape (npts,nvm,12)")
        npts_eff = int(laimap_arr.shape[0])
        lai = jnp.zeros((npts_eff, nvm), dtype=jnp.float64)
        for jv in range(1, nvm):
            kind = type_array[jv]
            if kind == "mean":
                lai = lai.at[:, jv].set(jnp.max(laimap_arr[:, jv, :], axis=1))
            else:
                mm = month - 1
                if month == 1:
                    if day <= 15:
                        left, right = 11, 0
                        weight = (day + 15.0) / 30.0
                    else:
                        left, right = 0, 1
                        weight = (day - 15.0) / 30.0
                elif month == 12:
                    if day <= 15:
                        left, right = 10, 11
                        weight = (day + 15.0) / 30.0
                    else:
                        left, right = 11, 0
                        weight = (day - 15.0) / 30.0
                else:
                    if day <= 15:
                        left, right = mm - 1, mm
                        weight = (day + 15.0) / 30.0
                    else:
                        left, right = mm, mm + 1
                        weight = (day - 15.0) / 30.0
                value = laimap_arr[:, jv, left] * (1.0 - weight) + laimap_arr[:, jv, right] * weight
                lai = lai.at[:, jv].set(value)
        return SlowprocLaiResult(lai=lai)

    if npts is None:
        if stempdiag is not None:
            npts_eff = int(jnp.asarray(stempdiag).shape[0])
        elif tempfunc_values is not None:
            npts_eff = int(jnp.asarray(tempfunc_values).shape[0])
        else:
            raise ValueError("READ_LAI=False requires npts, stempdiag, or tempfunc_values")
    else:
        npts_eff = int(npts)
    if npts_eff < 1:
        raise ValueError("npts must be positive")
    if llaimax is None or llaimin is None:
        raise ValueError("READ_LAI=False requires llaimax and llaimin")
    lmax = jnp.asarray(llaimax, dtype=jnp.float64)
    lmin = jnp.asarray(llaimin, dtype=jnp.float64)
    if lmax.shape != (nvm,) or lmin.shape != (nvm,):
        raise ValueError("llaimax and llaimin must have shape (nvm,)")

    if any(kind == "inter" for kind in type_array[1:]):
        if tempfunc_values is None:
            raise ValueError("READ_LAI=False inter branch requires explicit tempfunc_values")
        temp = jnp.asarray(tempfunc_values, dtype=jnp.float64)
        if temp.shape != (npts_eff,):
            raise ValueError("tempfunc_values must have shape (npts,)")
    else:
        temp = jnp.zeros((npts_eff,), dtype=jnp.float64)

    lai = jnp.zeros((npts_eff, nvm), dtype=jnp.float64)
    for jv in range(1, nvm):
        kind = type_array[jv]
        if kind == "mean":
            lai = lai.at[:, jv].set(0.5 * (lmax[jv] + lmin[jv]))
        else:
            lai = lai.at[:, jv].set(lmin[jv] + temp * (lmax[jv] - lmin[jv]))
    return SlowprocLaiResult(lai=lai)


def slowproc_tot_bare_soil(*, veget_max, veget):
    """Compute ``tot_bare_soil`` after SLOWPROC vegetation updates.

    Fortran provenance: ``slowproc.f90::slowproc_main`` lines 1116-1121:
    initialize with bare soil PFT1, then add ``veget_max - veget`` for PFTs
    2..nvm.
    """

    veget_max = _as_float64(veget_max)
    veget = _as_float64(veget)
    if veget_max.ndim != 2 or veget.shape != veget_max.shape:
        raise ValueError("veget_max and veget must share shape (npts, nvm)")
    return veget_max[:, 0] + jnp.sum(veget_max[:, 1:] - veget[:, 1:], axis=1)


def slowproc_surface_update_explicit(
    *,
    lai,
    frac_nobio,
    veget_max,
    pref_soil_veg,
    ext_coeff_vegetfrac,
    nstm: int,
    ok_dgvm=False,
    min_vegfrac=MIN_VEGFRAC,
    min_sechiba=MIN_SECHIBA,
) -> SlowprocSurfaceUpdateResult:
    """Run the closed post-STOMATE surface update in ``slowproc_main``.

    Fortran provenance: ``slowproc.f90::slowproc_main`` line 1103 calls
    ``slowproc_veget`` and lines 1116-1121 compute ``tot_bare_soil``. The
    caller must supply the exact ``lai`` and ``veget_max`` state at this point.
    """

    vegetation = slowproc_veget_explicit(
        lai=lai,
        frac_nobio=frac_nobio,
        veget_max=veget_max,
        pref_soil_veg=pref_soil_veg,
        ext_coeff_vegetfrac=ext_coeff_vegetfrac,
        nstm=nstm,
        ok_dgvm=ok_dgvm,
        min_vegfrac=min_vegfrac,
        min_sechiba=min_sechiba,
    )
    return SlowprocSurfaceUpdateResult(
        vegetation=vegetation,
        tot_bare_soil=slowproc_tot_bare_soil(
            veget_max=vegetation.veget_max,
            veget=vegetation.veget,
        ),
    )


def slowproc_surface_transition_explicit(
    *,
    do_slow: bool,
    lai,
    frac_nobio,
    veget_max,
    veget=None,
    totfrac_nobio=None,
    soiltile=None,
    pref_soil_veg=None,
    ext_coeff_vegetfrac=None,
    nstm: int | None = None,
    ok_dgvm=False,
    min_vegfrac=MIN_VEGFRAC,
    min_sechiba=MIN_SECHIBA,
) -> SlowprocSurfaceTransitionResult:
    """Apply the reachable ``do_slow`` gate around the surface update.

    Fortran provenance: ``slowproc_main`` lines 654-663 derive ``do_slow``;
    lines 1078-1114 update LAI/vegetation/interception state only when it is
    true, while lines 1116-1121 recompute ``tot_bare_soil`` every step.  The
    false branch therefore preserves all vegetation state exactly.  No
    missing previous state is reconstructed here.
    """

    if bool(do_slow):
        missing = tuple(
            name
            for name, value in (
                ("pref_soil_veg", pref_soil_veg),
                ("ext_coeff_vegetfrac", ext_coeff_vegetfrac),
                ("nstm", nstm),
            )
            if value is None
        )
        if missing:
            raise ValueError(f"do_slow=True requires surface-update inputs: {missing}")
        surface = slowproc_surface_update_explicit(
            lai=lai,
            frac_nobio=frac_nobio,
            veget_max=veget_max,
            pref_soil_veg=pref_soil_veg,
            ext_coeff_vegetfrac=ext_coeff_vegetfrac,
            nstm=nstm,
            ok_dgvm=ok_dgvm,
            min_vegfrac=min_vegfrac,
            min_sechiba=min_sechiba,
        )
        return SlowprocSurfaceTransitionResult(surface=surface, do_slow=True)

    missing = tuple(
        name
        for name, value in (
            ("veget", veget),
            ("totfrac_nobio", totfrac_nobio),
            ("soiltile", soiltile),
        )
        if value is None
    )
    if missing:
        raise ValueError(f"do_slow=False requires previous surface state: {missing}")
    frac_nobio_arr, veget_max_arr = _validate_fraction_arrays(frac_nobio, veget_max)
    lai_arr = _as_float64(lai)
    veget_arr = _as_float64(veget)
    totfrac_arr = _as_float64(totfrac_nobio)
    soiltile_arr = _as_float64(soiltile)
    npts = veget_max_arr.shape[0]
    if lai_arr.shape != veget_max_arr.shape or veget_arr.shape != veget_max_arr.shape:
        raise ValueError("lai, veget, and veget_max must share shape (npts, nvm)")
    if totfrac_arr.shape != (npts,) or soiltile_arr.ndim != 2 or soiltile_arr.shape[0] != npts:
        raise ValueError("totfrac_nobio and soiltile must match the previous land-point state")
    vegetation = SlowprocVegetResult(
        frac_nobio=frac_nobio_arr,
        veget_max=veget_max_arr,
        veget=veget_arr,
        totfrac_nobio=totfrac_arr,
        soiltile=soiltile_arr,
    )
    surface = SlowprocSurfaceUpdateResult(
        vegetation=vegetation,
        tot_bare_soil=slowproc_tot_bare_soil(veget_max=veget_max_arr, veget=veget_arr),
        notes=(
            "do_slow is false: vegetation state is carried unchanged and only tot_bare_soil is diagnosed.",
        ),
    )
    return SlowprocSurfaceTransitionResult(surface=surface, do_slow=False)


def validate_slowproc_main_pft14_switches(switches: SlowprocMainPft14Switches) -> None:
    """Reject source branches outside the fixed 250919 PFT14 target."""

    expected = SlowprocMainPft14Switches()
    mismatches = tuple(
        f"{name}={getattr(switches, name)!r} (required {getattr(expected, name)!r})"
        for name in expected.__dataclass_fields__
        if getattr(switches, name) != getattr(expected, name)
    )
    if mismatches:
        raise ValueError("unsupported slowproc_main structural switch: " + "; ".join(mismatches))


def _slowproc_main_post_stomate_vegetation(stomate, vegetation: dict[str, object], *, do_slow: bool):
    """Extract only fields explicitly produced by ``stomate_main_pft14_step``."""

    if not do_slow or getattr(stomate, "lpj", None) is None:
        return vegetation
    post = stomate.lpj.post_npp
    if post.lai_after_setlai is not None:
        vegetation["lai"] = post.lai_after_setlai
    if post.cover is not None:
        vegetation["veget_max"] = post.cover.veget_max
        vegetation["biomass"] = post.cover.biomass
    elif post.crown_after_establish is not None:
        vegetation["biomass"] = post.crown_after_establish.biomass
    else:
        vegetation["biomass"] = post.turnover.biomass
    final_leaf = post.vmax.leaf_frac if post.vmax is not None else post.turnover.leaf_frac
    vegetation["frac_age"] = final_leaf
    final_crown = post.crown_after_establish or post.crown_after_npp
    if final_crown is not None:
        vegetation["height"] = final_crown.height
    if post.vmax is not None:
        vegetation["assim_param"] = post.vmax.vcmax[:, :, None]
    season_memory = getattr(stomate, "season_memory", None)
    season_state = getattr(season_memory, "state", None)
    if season_state is not None and hasattr(season_state, "t2m_month"):
        vegetation["temp_growth"] = season_state.t2m_month - 273.15
    return vegetation


def slowproc_main_pft14_step(
    *,
    month: int,
    day: int,
    sec: float,
    dt_sechiba: float,
    dt_stomate: float,
    date: int,
    vegetation_state: Mapping[str, object],
    soil_state: Mapping[str, object],
    salinity,
    tide_height,
    slow_state: Mapping[str, object],
    pref_soil_veg,
    ext_coeff_vegetfrac,
    nstm: int,
    qsintcst: float,
    stomate_step_kwargs: Mapping[str, object] | None = None,
    stomate_result: object | None = None,
    switches: SlowprocMainPft14Switches | None = None,
    bulk_density_default: float = 1.65,
    one_day: float = 86400.0,
) -> SlowprocMainPft14Result:
    """Run ``slowproc_main`` lines 556-1134 for the fixed paper PFT14 graph.

    Runtime calendar branches, continuous values, and the landpoint dimension
    remain dynamic. Active vegetation-map, rotation, fire, and dynamic-peat
    structures are rejected. ``stomate_step_kwargs`` is passed to the existing
    exact STOMATE owner; ``stomate_result`` permits composition with a result
    already produced by that same owner, but the two boundaries are mutually
    exclusive.
    """

    fixed = switches or SlowprocMainPft14Switches()
    validate_slowproc_main_pft14_switches(fixed)
    if float(one_day) != 86400.0:
        raise ValueError("one_day is the fixed Fortran source constant 86400 seconds")
    if (stomate_step_kwargs is None) == (stomate_result is None):
        raise ValueError("provide exactly one of stomate_step_kwargs or stomate_result")

    vegetation = {name: jnp.asarray(value) for name, value in vegetation_state.items()}
    required_vegetation = (
        "lai",
        "frac_age",
        "height",
        "veget",
        "frac_nobio",
        "veget_max",
        "totfrac_nobio",
        "soiltile",
        "qsintmax",
        "temp_growth",
    )
    missing = tuple(name for name in required_vegetation if name not in vegetation)
    if missing:
        raise ValueError("vegetation_state requires " + ", ".join(missing))
    veget_max = _as_float64(vegetation["veget_max"])
    if veget_max.ndim != 2 or veget_max.shape[1] != fixed.nvm or veget_max.shape[0] < 1:
        raise ValueError("vegetation_state['veget_max'] must have shape (nland, 14)")
    nland = veget_max.shape[0]
    frac_nobio, _ = _validate_fraction_arrays(vegetation["frac_nobio"], veget_max)
    for name in ("lai", "height", "veget", "qsintmax"):
        if jnp.asarray(vegetation[name]).shape != veget_max.shape:
            raise ValueError(f"vegetation_state['{name}'] must have shape (nland, 14)")
    if jnp.asarray(vegetation["temp_growth"]).shape != (nland,):
        raise ValueError("vegetation_state['temp_growth'] must have shape (nland,)")
    salinity_arr = _as_float64(salinity)
    tide_arr = _as_float64(tide_height)
    if salinity_arr.shape != (nland,) or tide_arr.ndim != 2 or tide_arr.shape[0] != nland:
        raise ValueError("salinity and tide_height must have shapes (nland,) and (nland, ntide)")

    slow = {name: jnp.asarray(value) if name != "date" else int(value) for name, value in slow_state.items()}
    for name in ("growth_day", "peatC_ok"):
        if name not in slow or jnp.asarray(slow[name]).shape != (nland,):
            raise ValueError(f"slow_state['{name}'] must have shape (nland,)")

    # Source order: year reset, fixed dynpeat_PC Jan-2 write, then day-end growth.
    first_ts_year = (sec == dt_sechiba) and int(month) == 1 and int(day) == 1
    growth_day = _as_float64(slow["growth_day"])
    peatC_ok = _as_float64(slow["peatC_ok"])
    if first_ts_year:
        growth_day = jnp.zeros_like(growth_day)
    if (sec == dt_sechiba) and int(month) == 1 and int(day) == 2:
        # lines 572-574 execute independently of DYN_PEAT when dynpeat_PC is false.
        peatC_ok = jnp.ones_like(peatC_ok)
    last_ts_day = sec == 0
    end_of_year = last_ts_day and int(month) == 1 and int(day) == 1
    if last_ts_day:
        growth_day = growth_day + jnp.where(_as_float64(vegetation["temp_growth"]) > 5.0, 1.0, 0.0)
    do_slow = last_ts_day
    date_after = int(date)
    if do_slow:
        date_after += int(np.floor(float(dt_stomate) / float(one_day) + 0.5))

    no_lcc = slowproc_no_lcc_entry_state(
        veget_max=veget_max,
        use_age_class=False,
        veget_update=0,
        map_pft_format=True,
    )
    nobio = slowproc_stomate_nobio_boundary(
        frac_nobio_lastyear=frac_nobio,
        do_now_stomate_lcchange=False,
        use_age_class=False,
    )
    fire = slowproc_fire_disabled_entry_state(kjpindex=nland, fire_disable=True)

    if stomate_result is None:
        kwargs = dict(stomate_step_kwargs)
        entry_payload = dict(kwargs.pop("entry_payload"))
        entry_payload.update(
            date=date_after,
            lai=vegetation["lai"],
            frac_age=vegetation["frac_age"],
            height=vegetation["height"],
            veget=vegetation["veget"],
            veget_max=veget_max,
            veget_max_new=no_lcc.veget_max_new,
            vegetnew_firstday=no_lcc.vegetnew_firstday,
            totfrac_nobio_lastyear=nobio.totfrac_nobio_lastyear,
            totfrac_nobio_new=nobio.totfrac_nobio_new,
            lightn=fire.lightn,
            observed_ba=fire.observed_ba,
            cf_coarse=fire.cf_coarse,
            cf_fine=fire.cf_fine,
            ratio=fire.ratio,
            ratio_flag=fire.ratio_flag,
            popd=fire.popd,
            read_observed_ba=fire.read_observed_ba,
            humign=fire.humign,
            read_cf_fine=fire.read_cf_fine,
            read_cf_coarse=fire.read_cf_coarse,
            read_ratio_flag=fire.read_ratio_flag,
            read_ratio=fire.read_ratio,
        )
        for name, value in soil_state.items():
            entry_payload[name] = value
        owned = {"dt_sechiba", "dt_stomate", "do_slow", "switches"}
        overlap = tuple(sorted(owned.intersection(kwargs)))
        if overlap:
            raise ValueError("stomate_step_kwargs must not override " + ", ".join(overlap))
        stomate_result = stomate_main_pft14_step(
            entry_payload=entry_payload,
            dt_sechiba=dt_sechiba,
            dt_stomate=dt_stomate,
            do_slow=do_slow,
            switches=StomateMainPFT14Switches(),
            **kwargs,
        )

    vegetation = _slowproc_main_post_stomate_vegetation(stomate_result, vegetation, do_slow=do_slow)
    carbon_state = dict(getattr(stomate_result, "carbon_state", {}))
    for name in ("biomass", "litter_above", "litter_below", "carbon_32l", "DOC"):
        if name in carbon_state:
            vegetation[name] = carbon_state[name]

    surface = slowproc_surface_transition_explicit(
        do_slow=do_slow,
        lai=vegetation["lai"],
        frac_nobio=frac_nobio,
        veget_max=vegetation["veget_max"],
        veget=vegetation["veget"],
        totfrac_nobio=vegetation["totfrac_nobio"],
        soiltile=vegetation["soiltile"],
        pref_soil_veg=pref_soil_veg,
        ext_coeff_vegetfrac=ext_coeff_vegetfrac,
        nstm=nstm,
        ok_dgvm=False,
    )
    surface_state = surface.surface
    vegetation.update(
        frac_nobio=surface_state.vegetation.frac_nobio,
        veget_max=surface_state.vegetation.veget_max,
        veget=surface_state.vegetation.veget,
        totfrac_nobio=surface_state.vegetation.totfrac_nobio,
        soiltile=surface_state.vegetation.soiltile,
        tot_bare_soil=surface_state.tot_bare_soil,
    )
    if do_slow:
        vegetation["qsintmax"] = (
            _as_float64(qsintcst) * surface_state.vegetation.veget * _as_float64(vegetation["lai"])
        )
        vegetation["qsintmax"] = vegetation["qsintmax"].at[:, 0].set(0.0)

    soil = {name: jnp.asarray(value) for name, value in soil_state.items()}
    soil.update(carbon_state)
    for name in ("clayfraction", "siltfraction", "bulk_density"):
        if name not in soil or jnp.asarray(soil[name]).shape != (nland,):
            raise ValueError(f"soil_state['{name}'] must have shape (nland,)")
    bulk_density = _as_float64(soil["bulk_density"])
    soil["bulkdens"] = jnp.where(bulk_density > 0.0, bulk_density, float(bulk_density_default) * 1.0e3)
    clay = _as_float64(soil["clayfraction"])
    silt = _as_float64(soil["siltfraction"])
    soil["textfrac"] = jnp.stack((clay, silt, 1.0 - clay - silt), axis=1)

    slow.update(
        date=date_after,
        growth_day=growth_day,
        peatC_ok=peatC_ok,
        update_peatfrac=False,
    )
    slow.update(dict(getattr(stomate_result, "daily_state", {})))
    calendar = SlowprocMainCalendarState(
        date=date_after,
        first_ts_year=first_ts_year,
        first_ts_month=False,
        last_ts_day=last_ts_day,
        end_of_year=end_of_year,
        do_slow=do_slow,
        update_peatfrac=False,
        growth_day=growth_day,
        peatC_ok=peatC_ok,
    )
    diagnostics = dict(getattr(getattr(stomate_result, "output", None), "history", {}))
    diagnostics.update(
        tot_bare_soil=surface_state.tot_bare_soil,
        GSL=slow.get("GSL"),
        peatPET_lastyear=slow.get("peatPET_lastyear"),
        precipitation_lastsummer=slow.get("precipitation_lastsummer"),
        summerp_long=slow.get("summerp_long"),
        summerpet_long=slow.get("summerpet_long"),
    )
    return SlowprocMainPft14Result(
        calendar=calendar,
        stomate=stomate_result,
        vegetation_state=vegetation,
        soil_state=soil,
        salinity=salinity_arr,
        tide_height=tide_arr,
        slow_state=slow,
        diagnostics=diagnostics,
    )


def slowproc_cold_start_vegetation_entry_state(
    *,
    veget_max,
    frac_nobio,
    pref_soil_veg,
    ext_coeff_vegetfrac,
    height_presc,
    nstm: int,
    nleafages: int = 4,
    read_lai: bool = False,
    laimap=None,
    type_of_lai=None,
    month: int | None = None,
    day: int | None = None,
    ok_stomate: bool = True,
    val_exp: float = VAL_EXP,
    dtype=jnp.float64,
) -> SlowprocColdStartVegetationEntryState:
    """Build no-restart LAI/height/frac_age/veget state for SLOWPROC.

    For the active paper cold start, ``read_lai`` is false and STOMATE is
    enabled. ``slowproc_init`` first calls ``slowproc_veget`` in the imposed
    vegetation/no-restart branch while missing ``lai`` still has ``val_exp``;
    this yields full cover for present PFTs. It later sets missing ``lai`` to
    zero for STOMATE consistency, sets ``frac_age(:,:,1)=1``, and initializes
    missing ``height`` from ``SLOWPROC_HEIGHT``/``height_presc``.

    When ``read_lai`` is true, the caller must provide the exact same-case
    ``laimap`` already read from restart or produced by ``slowproc_interlai``.
    This function then applies ``slowproc_lai`` lines 2949-3062 and uses that
    LAI in the final ``slowproc_veget`` call, without inventing file
    interpolation.
    """

    veget_max = jnp.asarray(veget_max, dtype=dtype)
    frac_nobio = jnp.asarray(frac_nobio, dtype=dtype)
    height_presc = jnp.asarray(height_presc, dtype=dtype)
    if veget_max.ndim != 2:
        raise ValueError("veget_max must have shape (npts,nvm)")
    npts, nvm = veget_max.shape
    if height_presc.shape != (nvm,):
        raise ValueError("height_presc must have shape (nvm,)")
    nleafages = int(nleafages)
    if nleafages < 1:
        raise ValueError("nleafages must be positive")

    if bool(read_lai):
        if type_of_lai is None or month is None or day is None:
            raise ValueError("read_lai cold-start path requires type_of_lai, month, and day")
        lai = slowproc_lai_explicit(
            read_lai=True,
            type_of_lai=type_of_lai,
            month=month,
            day=day,
            laimap=laimap,
        ).lai.astype(dtype)
        slowproc_veget_lai = lai
    else:
        lai = jnp.zeros((npts, nvm), dtype=dtype)
        slowproc_veget_lai = jnp.full((npts, nvm), jnp.asarray(val_exp, dtype=dtype), dtype=dtype)
    frac_age = jnp.zeros((npts, nvm, nleafages), dtype=dtype).at[:, :, 0].set(1.0)
    height = jnp.broadcast_to(height_presc[None, :], (npts, nvm))
    vegetation = slowproc_veget_explicit(
        lai=slowproc_veget_lai,
        frac_nobio=frac_nobio,
        veget_max=veget_max,
        pref_soil_veg=pref_soil_veg,
        ext_coeff_vegetfrac=ext_coeff_vegetfrac,
        nstm=nstm,
    )
    return SlowprocColdStartVegetationEntryState(
        lai=lai,
        height=height,
        frac_age=frac_age,
        vegetation=vegetation,
        tot_bare_soil=slowproc_tot_bare_soil(
            veget_max=vegetation.veget_max,
            veget=vegetation.veget,
        ),
    )


def slowproc_init_pft14_explicit(
    *,
    restart: Mapping[str, object],
    veget_max_default,
    frac_nobio_default: float,
    height_presc,
    pref_soil_veg,
    ext_coeff_vegetfrac,
    nstm: int,
    diaglev,
    soil_boundary: Mapping[str, object] | None = None,
    salinity_data=None,
    tide_height_data=None,
    impose_veg: bool = True,
    impose_soilt: bool = False,
    ok_stomate: bool = True,
    ok_dgvm: bool = False,
    map_pft_format: bool = True,
    use_age_class: bool = False,
    veget_update="0Y",
    fire_disable: bool = True,
    dyn_peat: bool = False,
    hydrol_cwrr: bool = True,
    get_slope: bool = False,
    read_lai: bool = False,
    read_salinity: bool = True,
    read_tide: bool = True,
    soil_classif: str = "usda",
    nleafages: int = 4,
    val_exp: float = VAL_EXP,
    undef_sechiba: float = 1.0e20,
    undef_int: int = 999999999,
    precip_crit: float = 100.0,
    zcanop: float = 0.5,
    qsintcst: float = 0.1,
    dt_stomate: float = 86400.0,
    PWT_lim: float = 60.0,
    PC_lim: float = 50.0,
    sat_gsl: float = 1.0,
    active_pft_fortran: int = 14,
) -> SlowprocInitPft14Result:
    """Close the reachable paper-case ``slowproc_init`` state transition.

    ``restart`` uses the Fortran restart names normalized to model-point-first
    arrays. ``soil_boundary`` is the exact output boundary of
    ``slowproc_soilt`` and must contain its canonical ``soilclass``,
    ``clay_frac``, ``sand_frac``, ``silt_frac``, ``bulk_dens``, ``soil_ph``,
    and ``poor_soils`` outputs when the restart soil state triggers that call
    (legacy Fortran-style field aliases remain accepted). Salinity and
    tide arrays are likewise post-interpolation values, not approximations of
    ``slowproc_read_annual`` or ``slowproc_read_data``.

    Fortran provenance: ``slowproc.f90::slowproc_init`` lines 1404-2534.
    This owned closure is deliberately restricted to the paper protocol:
    only PFT14 occupied at each land point, imposed vegetation,
    STOMATE and CWRR enabled, no DGVM/age classes/fire/dynamic peat, and
    externally supplied active salinity/tide reads.
    """

    unsupported = []
    checks = (
        (impose_veg, "IMPOSE_VEG=n: map vegetation branch, lines 2037-2137"),
        (not impose_soilt, "IMPOSE_SOILT=y: prescribed soil branch, lines 1928-1995"),
        (ok_stomate, "OK_STOMATE=n: prescribed-LAI branch, lines 1908-1920"),
        (not ok_dgvm, "OK_DGVM=y: dynamic vegetation reset, lines 2096-2130"),
        (map_pft_format, "MAP_PFT_FORMAT=n: Olson interpolation branch, lines 2078-2094"),
        (not use_age_class, "GLUC_USE_AGE_CLASS=y: age-class map/LCC branch, lines 2043-2061 and 2308-2331"),
        (fire_disable, "FIRE_DISABLE=n: SPITFIRE file branches, lines 2281-2492"),
        (not dyn_peat, "DYN_PEAT=y: dynamic peat state is outside this init closure"),
        (hydrol_cwrr, "HYDROL_CWRR=n: reinfiltration state differs at lines 1648-1653"),
        (not get_slope, "GET_SLOPE=y: slope-map interpolation, lines 2230-2240"),
        (not read_lai, "READ_LAI=y: LAI-map interpolation, lines 1436-1442 and 2141-2154"),
        (read_salinity, "READ_SALINITY=n: paper run has an active read at lines 2494-2513"),
        (read_tide, "READ_TIDE=n: paper run has an active read at lines 2515-2534"),
        (str(soil_classif).strip().lower() == "usda", "non-USDA fc_grazing branch, lines 2201-2221"),
    )
    unsupported.extend(message for owned, message in checks if not bool(owned))
    if unsupported:
        raise SlowprocInitGapError("slowproc_init PFT14 gap: " + "; ".join(unsupported))

    diaglev = _as_float64(diaglev)
    if diaglev.ndim != 1 or diaglev.size < 1:
        raise ValueError("diaglev must be a non-empty one-dimensional array")
    zsoil = jnp.empty_like(diaglev).at[0].set(diaglev[0] / 2.0)
    if diaglev.size > 1:
        zsoil = zsoil.at[1:].set((diaglev[1:] + diaglev[:-1]) / 2.0)
    lcanop = int(jnp.argmin(jnp.abs(float(zcanop) - zsoil))) + 1
    if float(dt_stomate) <= 0.0:
        raise ValueError("dt_stomate must be positive")

    default_veg = _as_float64(veget_max_default)
    if default_veg.ndim == 1:
        default_veg = default_veg[None, :]
    if default_veg.ndim != 2 or default_veg.shape[0] < 1:
        raise ValueError("veget_max_default must have shape (nvm,) or (npts,nvm)")
    npts, nvm = map(int, default_veg.shape)
    active_pft = int(active_pft_fortran) - 1
    if active_pft < 0 or active_pft >= nvm:
        raise ValueError("active_pft_fortran must be a Fortran index inside 1..nvm")
    height_presc = _as_float64(height_presc)
    if height_presc.shape != (nvm,):
        raise ValueError("height_presc must have shape (nvm,)")
    inactive_pfts = np.arange(nvm) != active_pft
    if bool(jnp.any(default_veg[:, inactive_pfts] != 0.0)) or not bool(jnp.all(default_veg[:, active_pft] > 0.0)):
        raise SlowprocInitGapError("slowproc_init PFT14 gap: prescribed vegetation is not PFT14-only")

    def restart_array(name, shape, *, dtype=jnp.float64, missing=val_exp):
        value = restart.get(name)
        if value is None:
            return jnp.full(shape, missing, dtype=dtype)
        out = jnp.asarray(value, dtype=dtype)
        if out.shape != shape:
            raise ValueError(f"restart[{name!r}] must have shape {shape}")
        return out

    def is_all(array, value) -> bool:
        return bool(jnp.all(array == jnp.asarray(value, dtype=array.dtype)))

    veget_restart = restart_array("veget", (npts, nvm))
    veget_max_restart = restart_array("veget_max", (npts, nvm))
    frac_nobio_restart = restart_array("frac_nobio", (npts, 1))
    vegetation_missing = tuple(
        is_all(array, val_exp)
        for array in (veget_restart, veget_max_restart, frac_nobio_restart)
    )
    if any(vegetation_missing) and not all(vegetation_missing):
        raise SlowprocInitGapError(
            "slowproc_init PFT14 gap: mixed present/missing veget, veget_max, frac_nobio restart fields "
            "enter the partially recovered branch at lines 1581-1601 and 1881-1926"
        )
    found_restart = not any(vegetation_missing)

    lai_restart = restart_array("lai", (npts, nvm))
    height_restart = restart_array("height", (npts, nvm))
    frac_age_restart = restart_array("frac_age", (npts, nvm, int(nleafages)))
    lai_missing = is_all(lai_restart, val_exp)
    height_missing = is_all(height_restart, val_exp)

    if found_restart:
        if lai_missing or is_all(frac_age_restart, val_exp):
            raise SlowprocInitGapError(
                "slowproc_init PFT14 gap: vegetation restart is present but LAI/frac_age is absent; "
                "lines 2149-2164 couple frac_age fallback only to missing LAI"
            )
        veget = veget_restart
        veget_max = veget_max_restart
        frac_nobio = frac_nobio_restart
        lai = lai_restart
        frac_age = frac_age_restart
        height = jnp.broadcast_to(height_presc[None, :], (npts, nvm)) if height_missing else height_restart
    else:
        if not lai_missing or not height_missing or not is_all(frac_age_restart, val_exp):
            raise SlowprocInitGapError(
                "slowproc_init PFT14 gap: cold-start vegetation with pre-populated LAI/height/frac_age "
                "is not in the paper path"
            )
        frac_nobio_default = _as_float64(frac_nobio_default)
        if frac_nobio_default.ndim == 0:
            frac_nobio_default = jnp.full((npts, 1), frac_nobio_default, dtype=jnp.float64)
        elif frac_nobio_default.shape == (npts,):
            frac_nobio_default = frac_nobio_default[:, None]
        elif frac_nobio_default.shape != (npts, 1):
            raise ValueError("frac_nobio_default must be scalar, (npts,), or (npts,1)")
        cold = slowproc_cold_start_vegetation_entry_state(
            veget_max=default_veg,
            frac_nobio=frac_nobio_default,
            pref_soil_veg=pref_soil_veg,
            ext_coeff_vegetfrac=ext_coeff_vegetfrac,
            height_presc=height_presc,
            nstm=int(nstm),
            nleafages=int(nleafages),
            read_lai=False,
            ok_stomate=True,
            val_exp=val_exp,
        )
        veget = cold.vegetation.veget
        veget_max = cold.vegetation.veget_max
        frac_nobio = cold.vegetation.frac_nobio
        lai = cold.lai
        frac_age = cold.frac_age
        height = cold.height

    if bool(jnp.any(veget_max[:, inactive_pfts] != 0.0)) or bool(jnp.any(veget[:, inactive_pfts] != 0.0)):
        raise SlowprocInitGapError("slowproc_init PFT14 gap: restart activates a PFT other than PFT14")

    pref = jnp.asarray(pref_soil_veg, dtype=jnp.int32)
    if pref.shape != (nvm,) or int(nstm) < 1:
        raise ValueError("pref_soil_veg must have shape (nvm,) and nstm must be positive")
    if bool(jnp.any(pref < 1)) or bool(jnp.any(pref > int(nstm))):
        raise ValueError("pref_soil_veg must contain Fortran indices in 1..nstm")
    totfrac_nobio = jnp.sum(frac_nobio, axis=1)
    soiltile = jnp.zeros((npts, int(nstm)), dtype=jnp.float64)
    for jv in range(nvm):
        soiltile = soiltile.at[:, int(pref[jv]) - 1].add(veget_max[:, jv])
    vegetated = totfrac_nobio < (1.0 - MIN_SECHIBA)
    soiltile = jnp.where(
        vegetated[:, None],
        soiltile / jnp.where(vegetated[:, None], 1.0 - totfrac_nobio[:, None], 1.0),
        soiltile,
    )
    tot_bare_soil = slowproc_tot_bare_soil(veget_max=veget_max, veget=veget)

    njsc_restart = restart_array("njsc", (npts,), dtype=jnp.int32, missing=undef_int)
    clay_restart = restart_array("clay_frac", (npts,))
    sand_restart = restart_array("sand_frac", (npts,))
    bulk_restart = restart_array("bulk_dens", (npts,))
    ph_restart = restart_array("soil_ph", (npts,))
    poor_restart = restart_array("poor_soils", (npts,))
    soil_missing = (
        is_all(njsc_restart, undef_int)
        or is_all(clay_restart, val_exp)
        or is_all(sand_restart, val_exp)
        or is_all(bulk_restart, val_exp)
        or is_all(ph_restart, val_exp)
    )
    if soil_missing:
        if soil_boundary is None:
            raise SlowprocInitGapError(
                "slowproc_init PFT14 gap: restart soil state triggers slowproc_soilt at lines 2174-2193; "
                "provide the exact post-interpolation soil_boundary"
            )
        aliases = {
            "soilclass": ("soilclass",),
            "clayfraction": ("clay_frac", "clayfraction"),
            "sandfraction": ("sand_frac", "sandfraction"),
            "siltfraction": ("silt_frac", "siltfraction"),
            "bulk_density": ("bulk_dens", "bulk_density"),
            "soil_ph": ("soil_ph",),
            "poor_soils": ("poor_soils",),
        }
        missing_keys = tuple(name for name, names in aliases.items() if not any(key in soil_boundary for key in names))
        if missing_keys:
            raise ValueError(f"soil_boundary is missing fields: {missing_keys}")

        def soil_value(name):
            return soil_boundary[next(key for key in aliases[name] if key in soil_boundary)]

        soilclass = _as_float64(soil_value("soilclass"))
        if soilclass.shape != (npts, 12):
            raise ValueError("soil_boundary['soilclass'] must have shape (npts,12) for USDA")
        njsc = jnp.argmax(soilclass, axis=1).astype(jnp.int32) + 1
        clayfraction = _as_float64(soil_value("clayfraction"))
        sandfraction = _as_float64(soil_value("sandfraction"))
        siltfraction = _as_float64(soil_value("siltfraction"))
        bulk_density = _as_float64(soil_value("bulk_density"))
        soil_ph = _as_float64(soil_value("soil_ph"))
        poor_soils = _as_float64(soil_value("poor_soils"))
        for name, value in (
            ("clayfraction", clayfraction), ("sandfraction", sandfraction),
            ("siltfraction", siltfraction), ("bulk_density", bulk_density),
            ("soil_ph", soil_ph), ("poor_soils", poor_soils),
        ):
            if value.shape != (npts,):
                raise ValueError(f"soil_boundary[{name!r}] must have shape (npts,)")
    else:
        njsc = njsc_restart
        clayfraction = clay_restart
        sandfraction = sand_restart
        siltfraction = 1.0 - clayfraction - sandfraction
        bulk_density = bulk_restart
        soil_ph = ph_restart
        poor_soils = poor_restart
    if bool(jnp.any(njsc < 1)) or bool(jnp.any(njsc > len(MCS_USDA))):
        raise ValueError("njsc is outside the USDA soil-class table 1..12")
    fc_grazing = jnp.asarray(MCS_USDA, dtype=jnp.float64)[njsc - 1]

    reinf_slope = restart_array("reinf_slope", (npts,))
    if is_all(reinf_slope, val_exp):
        reinf_slope = jnp.zeros((npts,), dtype=jnp.float64)

    def external(name, value, ndim):
        if value is None:
            raise SlowprocInitGapError(
                f"slowproc_init PFT14 gap: active {name} NetCDF interpolation requires explicit data"
            )
        out = _as_float64(value)
        if out.ndim != ndim or out.shape[0] != npts:
            expected = "(npts,)" if ndim == 1 else "(npts,itimetide)"
            raise ValueError(f"{name} must have shape {expected}")
        return out

    salinity = external("salinity", salinity_data, 1)
    tide_height = external("tide_height", tide_height_data, 2)

    peat_defaults = {
        "peatPET_last": undef_sechiba,
        "precipitation_last": precip_crit,
        "precipitation_this": 0.0,
        "peatPET_this": 0.0,
        "growth_day": 0.0,
        "GSL": 1.0,
        "summerp_longterm": 0.0,
        "summerpet_longterm": undef_sechiba,
        "peatC": 0.0,
        "peatC_ok": 0.0,
    }
    peat_values = {}
    for name, default in peat_defaults.items():
        value = restart_array(name, (npts,))
        peat_values[name] = jnp.full((npts,), default, dtype=jnp.float64) if is_all(value, val_exp) else value
    peat = SlowprocInitPeatMemoryState(
        peatPET_lastyear=peat_values["peatPET_last"],
        precipitation_lastsummer=peat_values["precipitation_last"],
        precipitation_thissummer=peat_values["precipitation_this"],
        peatPET_thisyear=peat_values["peatPET_this"],
        growth_day=peat_values["growth_day"],
        GSL=peat_values["GSL"],
        summerp_long=peat_values["summerp_longterm"],
        summerpet_long=peat_values["summerpet_longterm"],
        peatC=peat_values["peatC"],
        peatC_ok=peat_values["peatC_ok"],
    )
    no_lcc = slowproc_no_lcc_entry_state(
        veget_max=veget_max,
        use_age_class=False,
        veget_update=veget_update,
        map_pft_format=True,
        impose_veg=True,
    )
    fire = slowproc_fire_disabled_entry_state(kjpindex=npts, fire_disable=True)
    return SlowprocInitPft14Result(
        found_restart=found_restart,
        veget_update=0,
        lcanop=lcanop,
        qsintcst=float(qsintcst),
        dt_stomate=float(dt_stomate),
        PWT_lim=float(PWT_lim),
        PC_lim=float(PC_lim),
        sat_gsl=float(sat_gsl),
        lai=lai,
        height=height,
        frac_age=frac_age,
        veget=veget,
        veget_max=veget_max,
        frac_nobio=frac_nobio,
        totfrac_nobio=totfrac_nobio,
        soiltile=soiltile,
        tot_bare_soil=tot_bare_soil,
        reinf_slope=reinf_slope,
        njsc=njsc,
        clayfraction=clayfraction,
        sandfraction=sandfraction,
        siltfraction=siltfraction,
        bulk_density=bulk_density,
        soil_ph=soil_ph,
        poor_soils=poor_soils,
        fc_grazing=fc_grazing,
        salinity=salinity,
        tide_height=tide_height,
        peat=peat,
        no_lcc=no_lcc,
        fire=fire,
    )


def slowproc_initialize_source_routed(**kwargs):
    """Production entry point for completed ``slowproc_initialize`` routing."""
    from .science_completion import slowproc_initialize_source_routed as owner
    return owner(**kwargs)


def slowproc_main_source_routed(**kwargs):
    """Production entry point for main routing plus source invariant checks."""
    from .science_completion import slowproc_main_source_routed as owner
    return owner(**kwargs)


def slowproc_main_history_routing(**kwargs):
    """Production entry point for the secondary carbon-history arm."""
    from .science_completion import slowproc_main_history_routing as owner

    return owner(**kwargs)


def slowproc_finalize_restart_packet(**kwargs):
    """Production entry point for ``slowproc_finalize`` field selection."""
    from .science_completion import slowproc_finalize_restart_packet as owner
    return owner(**kwargs)


def get_soilcorr_usda_source_routed(nusda=12):
    """Production entry point for the USDA texture correspondence table."""
    from .science_completion import get_soilcorr_usda_source_routed as owner
    return owner(nusda)


def slowproc_checkveget_source_routed(**kwargs):
    """Production entry point for all ``slowproc_checkveget`` invariants."""
    from .science_completion import slowproc_checkveget_source_routed as owner
    return owner(**kwargs)


def slowproc_change_frac_source_routed(**kwargs):
    """Production entry point for ``slowproc_change_frac`` state transition."""
    from .science_completion import slowproc_change_frac_source_routed as owner
    return owner(**kwargs)
