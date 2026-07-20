"""Small source-driven ENERBIL kernels.

This module intentionally implements only isolated algebra from
``enerbil_surftemp``, ``enerbil_flux``, ``enerbil_pottemp``, and
``enerbil_evapveg``. It is not a full ``enerbil_main`` port.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import NamedTuple

from jax import config, core, jit

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jax_orchidee.driver.restart import read_restart_fields


class EnerbilEvapvegGridFluxes(NamedTuple):
    """Grid-cell snow, bare-soil, and floodplain evaporation components."""

    vevapsno: jnp.ndarray
    vevapnu: jnp.ndarray
    vevapflo: jnp.ndarray


class EnerbilEvapvegPFTFluxes(NamedTuple):
    """PFT-resolved bare-soil, interception, and transpiration components."""

    vevapnu_pft: jnp.ndarray
    vevapwet: jnp.ndarray
    transpir: jnp.ndarray
    transpot: jnp.ndarray


class EnerbilFluxDiagnostics(NamedTuple):
    """Local no-snow-ablation diagnostics from ``enerbil_flux``."""

    qsurf: jnp.ndarray
    lwup: jnp.ndarray
    lwnet: jnp.ndarray
    tsol_rad: jnp.ndarray
    netrad: jnp.ndarray
    vevapp: jnp.ndarray
    fluxlat: jnp.ndarray
    fluxsubli: jnp.ndarray
    fluxsens: jnp.ndarray
    evapot: jnp.ndarray


class EnerbilFluxEvapotCorrResult(NamedTuple):
    """Milly/Penman correction from the second ``enerbil_flux`` pass."""

    correction: jnp.ndarray
    evapot_corr: jnp.ndarray


class EnerbilFluxExplicitSnowDiagnostics(NamedTuple):
    """Explicit-snow ``pgflux`` and ablation temperature increment."""

    pgflux: jnp.ndarray
    temp_sol_add: jnp.ndarray
    phpsnow: jnp.ndarray


class EnerbilFusionResult(NamedTuple):
    """Surface temperature after ``enerbil_fusion``."""

    temp_sol_new: jnp.ndarray
    temp_sol_new_pft: jnp.ndarray
    fusion: jnp.ndarray
    provenance: tuple[str, ...]
    notes: tuple[str, ...]


class EnerbilBeginDiagnostics(NamedTuple):
    """Explicit local diagnostics from ``enerbil_begin``."""

    psold: jnp.ndarray
    psold_pft: jnp.ndarray
    qsol_sat: jnp.ndarray
    qsol_sat_pft: jnp.ndarray
    pdqsold: jnp.ndarray
    pdqsold_pft: jnp.ndarray
    lwabs: jnp.ndarray
    netrad: jnp.ndarray
    netrad_pft: jnp.ndarray


class EnerbilSurftempResult(NamedTuple):
    """Explicit linearized surface-temperature solve from ``enerbil_surftemp``."""

    dtheta: jnp.ndarray
    dtheta_pft: jnp.ndarray
    psnew: jnp.ndarray
    psnew_pft: jnp.ndarray
    qsol_sat_new: jnp.ndarray
    qsol_sat_new_pft: jnp.ndarray
    temp_sol_new: jnp.ndarray
    temp_sol_new_pft: jnp.ndarray
    qair_new: jnp.ndarray
    epot_air_new: jnp.ndarray


class EnerbilPottempResult(NamedTuple):
    """Exact pass-through result from ``enerbil_pottemp``."""

    q_sol_pot: jnp.ndarray
    temp_sol_pot: jnp.ndarray


class EnerbilExplicitStepResult(NamedTuple):
    """Explicit-input ENERBIL local sequence result."""

    begin: EnerbilBeginDiagnostics
    surftemp: EnerbilSurftempResult
    pottemp: EnerbilPottempResult | None
    flux: EnerbilFluxDiagnostics
    evapot_corr: EnerbilFluxEvapotCorrResult
    explicit_snow: EnerbilFluxExplicitSnowDiagnostics | None
    evapveg_grid: EnerbilEvapvegGridFluxes
    evapveg_pft: EnerbilEvapvegPFTFluxes
    t2mdiag: jnp.ndarray | None


class EnerbilServerBridgeExplicitAssembly(NamedTuple):
    """Audited server bridge assembly result for first-step local ENERBIL."""

    step: EnerbilExplicitStepResult | None
    missing_inputs: tuple[str, ...]
    trace_input_diagnostics: tuple[str, ...]
    source_kernel_inputs: tuple[str, ...]
    after_boundary_input_diagnostics: tuple[str, ...]
    inputs: Mapping[str, np.ndarray]
    provenance: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True only when the explicit local step was actually assembled."""

        return self.step is not None and not self.missing_inputs


class EnerbilFirstStepPrecallAssembly(NamedTuple):
    """Exact first-step ENERBIL pre-call payload assembled from local sources."""

    payload: Mapping[str, np.ndarray]
    missing_inputs: tuple[str, ...]
    driver_inputs: tuple[str, ...]
    source_kernel_inputs: tuple[str, ...]
    restart_inputs: tuple[str, ...]
    diffuco_inputs: tuple[str, ...]
    provenance: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when all external inputs needed to run local ENERBIL are exact."""

        return not self.missing_external_inputs

    @property
    def missing_external_inputs(self) -> tuple[str, ...]:
        """Missing fields excluding local ENERBIL begin/surftemp outputs."""

        local_outputs = {
            "psold",
            "psold_pft",
            "qsol_sat",
            "qsol_sat_pft",
            "pdqsold",
            "pdqsold_pft",
            "lwabs",
            "netrad",
            "netrad_pft",
            "dtheta",
            "dtheta_pft",
            "psnew",
            "psnew_pft",
            "qsol_sat_new",
            "qsol_sat_new_pft",
            "temp_sol_new",
            "temp_sol_new_pft",
            "qair_new",
            "epot_air_new",
        }
        return tuple(field for field in self.missing_inputs if field not in local_outputs)


class EnerbilFluxInputContract(NamedTuple):
    """Coverage check for inputs needed before local ``enerbil_flux`` kernels."""

    ok: bool
    missing_inputs: tuple[str, ...]
    after_main_outputs_only: tuple[str, ...]
    provenance: tuple[str, ...]


class EnerbilFirstStepLocalAssembly(NamedTuple):
    """First-step ENERBIL local execution from an exact pre-call payload."""

    step: EnerbilExplicitStepResult
    payload: Mapping[str, object]
    source_kernel_inputs: tuple[str, ...]
    passthrough_inputs: tuple[str, ...]
    provenance: tuple[str, ...]


class EnerbilFirstStepInputCoverage(NamedTuple):
    """Coverage audit for the first-step ENERBIL input chain."""

    covered_by_trace: tuple[str, ...]
    covered_by_source_kernel: tuple[str, ...]
    missing: tuple[str, ...]
    after_boundary_outputs_only: tuple[str, ...]
    trace_sources: Mapping[str, str]
    source_kernel_sources: Mapping[str, str]
    provenance: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True only when every audited input has an exact source."""

        return not self.missing


class EnerbilSurfaceStateInputCoverage(NamedTuple):
    """Coverage audit for ENERBIL surface thermal state inputs."""

    covered_by_trace: tuple[str, ...]
    covered_by_source_kernel: tuple[str, ...]
    missing: tuple[str, ...]
    after_boundary_outputs_only: tuple[str, ...]
    trace_sources: Mapping[str, str]
    source_kernel_sources: Mapping[str, str]
    provenance: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True only when every audited surface-state input is exact."""

        return not self.missing


class EnerbilSwnetInputChainCoverage(NamedTuple):
    """Coverage audit for the driver ``albedo -> swnet`` input chain."""

    covered_by_trace: tuple[str, ...]
    covered_by_source_kernel: tuple[str, ...]
    missing: tuple[str, ...]
    missing_formula_inputs: tuple[str, ...]
    after_boundary_outputs_only: tuple[str, ...]
    trace_sources: Mapping[str, str]
    source_kernel_sources: Mapping[str, str]
    provenance: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True only when ``swnet`` has an exact pre-ENERBIL source."""

        return not self.missing


class EnerbilSoilThermalStateRestart(NamedTuple):
    """THERMOSOIL restart state consumed by first-step ``enerbil_main``."""

    path: Path
    soilcap: np.ndarray
    soilcap_pft: np.ndarray
    soilflx: np.ndarray
    soilflx_pft: np.ndarray
    provenance: tuple[str, ...]

    def as_coverage_payload(self) -> Mapping[str, np.ndarray]:
        """Expose restart fields by ENERBIL interface name."""

        return {
            "soilcap": self.soilcap,
            "soilcap_pft": self.soilcap_pft,
            "soilflx": self.soilflx,
            "soilflx_pft": self.soilflx_pft,
        }


class EnerbilColdStartSurfaceState(NamedTuple):
    """Cold-start ENERBIL surface state from restart-missing fallbacks."""

    temp_sol: jnp.ndarray
    temp_sol_pft: jnp.ndarray
    temp_sol_new: jnp.ndarray
    temp_sol_new_pft: jnp.ndarray
    qsurf: jnp.ndarray
    evapot: jnp.ndarray
    evapot_corr: jnp.ndarray
    tsol_rad: jnp.ndarray
    temp_sol_pot: jnp.ndarray
    q_sol_pot: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_initialize lines 241-292",
        "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_initialize lines 319-328",
    )

    def as_payload(self) -> dict[str, jnp.ndarray]:
        return {
            "temp_sol": self.temp_sol,
            "temp_sol_pft": self.temp_sol_pft,
            "temp_sol_new": self.temp_sol_new,
            "temp_sol_new_pft": self.temp_sol_new_pft,
            "qsurf": self.qsurf,
            "evapot": self.evapot,
            "evapot_corr": self.evapot_corr,
            "tsol_rad": self.tsol_rad,
            "temp_sol_pot": self.temp_sol_pot,
            "q_sol_pot": self.q_sol_pot,
        }


class DriverAlbedoRestart(NamedTuple):
    """Driver restart surface fields consumed before first-step ``sechiba_main``."""

    path: Path
    albedo: np.ndarray
    z0: np.ndarray | None
    provenance: tuple[str, ...]

    def as_coverage_payload(self) -> Mapping[str, np.ndarray]:
        """Expose restart fields needed by the driver ``swnet`` formula."""

        return {
            "albedo": self.albedo,
            "albedo_vis": self.albedo[:, 0],
            "albedo_nir": self.albedo[:, 1],
        }


CONDVEG_EMIS_UNITY = 1.0
CHALSU0 = 2.8345e6
CHALEV0 = 2.5008e6
C_STEFAN = 5.6697e-8
CP_AIR = 1004.675
TP_00 = 273.15
CTE_MOLR = 287.05
CTE_GRAV = 9.80665
PA_PAR_HPA = 100.0
MSMLR_AIR = 28.964e-03
MSMLR_H2O = 18.02e-03
MIN_SECHIBA = 1.0e-8


def enerbil_cold_start_surface_state(*, qair, nvm, enerbil_tsurf=280.0, dtype=jnp.float64) -> EnerbilColdStartSurfaceState:
    """Build ENERBIL restart-missing first-step surface state.

    Fortran provenance: ``enerbil_initialize`` lines 241-255 use
    ``ENERBIL_TSURF`` default 280 K for missing ``temp_sol`` and
    ``temp_sol_pft``; lines 259-261 copy them to ``temp_sol_new*``; lines
    265-267 set missing ``qsurf`` to ``qair``; lines 273-286 zero
    ``evapot`` and ``evapot_corr``; lines 290-292 set ``tsol_rad`` to
    ``temp_sol``; lines 319-328 initialize potential state from the same
    surface fields.
    """

    qair = jnp.asarray(qair, dtype=dtype)
    if qair.ndim != 1:
        raise ValueError("qair must have shape (npts,)")
    nvm = int(nvm)
    if nvm < 1:
        raise ValueError("nvm must be positive")
    npts = int(qair.shape[0])
    temp_sol = jnp.full((npts,), float(enerbil_tsurf), dtype=dtype)
    temp_sol_pft = jnp.full((npts, nvm), float(enerbil_tsurf), dtype=dtype)
    zeros = jnp.zeros((npts,), dtype=dtype)
    return EnerbilColdStartSurfaceState(
        temp_sol=temp_sol,
        temp_sol_pft=temp_sol_pft,
        temp_sol_new=temp_sol,
        temp_sol_new_pft=temp_sol_pft,
        qsurf=qair,
        evapot=zeros,
        evapot_corr=zeros,
        tsol_rad=temp_sol,
        temp_sol_pot=temp_sol,
        q_sol_pot=qair,
    )

ENERBIL_FLUX_REQUIRED_INPUTS = (
    "emis",
    "temp_sol",
    "rau",
    "u",
    "v",
    "q_cdrag",
    "vbeta",
    "valpha",
    "vbeta1",
    "vbeta5",
    "qair",
    "epot_air",
    "psnew",
    "qsol_sat_new",
    "temp_sol_new",
    "lwdown",
    "swnet",
    "dt_sechiba",
)

ENERBIL_AFTER_MAIN_OUTPUTS_ONLY = (
    "qsurf",
    "fluxsens",
    "fluxlat",
    "evapot",
    "evapot_corr",
    "temp_sol_new",
    "temp_sol_new_pft",
)

ENERBIL_SURFACE_STATE_INPUT_FIELDS = (
    "swnet",
    "emis",
    "soilflx",
    "soilflx_pft",
    "soilcap",
    "soilcap_pft",
    "temp_sol",
    "temp_sol_pft",
)

ENERBIL_FIRST_STEP_INPUT_CHAIN_FIELDS = (
    "zlev",
    "lwdown",
    "swdown",
    "swnet",
    "epot_air",
    "temp_air",
    "u",
    "v",
    "petAcoef",
    "petBcoef",
    "qair",
    "peqAcoef",
    "peqBcoef",
    "pb",
    "rau",
    "vbeta",
    "vbeta_pft",
    "valpha",
    "vbeta1",
    "vbeta2",
    "vbeta3",
    "vbeta3pot",
    "vbeta4",
    "vbeta4_pft",
    "vbeta5",
    "emis",
    "soilflx",
    "soilflx_pft",
    "soilcap",
    "soilcap_pft",
    "q_cdrag",
    "q_cdrag_pft",
    "veget_max",
    "humrel",
    "precip_rain",
    "snowdz",
    "temp_sol",
    "temp_sol_pft",
    "qsurf",
    "evapot",
    "evapot_corr",
    "psold",
    "psold_pft",
    "qsol_sat",
    "qsol_sat_pft",
    "pdqsold",
    "pdqsold_pft",
    "lwabs",
    "netrad",
    "netrad_pft",
    "dtheta",
    "dtheta_pft",
    "psnew",
    "psnew_pft",
    "qsol_sat_new",
    "qsol_sat_new_pft",
    "temp_sol_new",
    "temp_sol_new_pft",
    "qair_new",
    "epot_air_new",
)

ENERBIL_AFTER_DIFFUCO_PRECALL_FIELDS = frozenset(
    (
        "vbeta",
        "vbeta_pft",
        "valpha",
        "vbeta1",
        "vbeta2",
        "vbeta3",
        "vbeta3pot",
        "vbeta4",
        "vbeta4_pft",
        "vbeta5",
        "q_cdrag",
        "q_cdrag_pft",
        "humrel",
        "veget_max",
        "temp_sol",
        "temp_sol_pft",
        "qsurf",
        "evapot",
        "evapot_corr",
        "snowdz",
    )
)

ENERBIL_DRIVER_OR_INTERSURF_TRACE_FIELDS = frozenset(
    (
        "zlev",
        "height_lev1",
        "lwdown",
        "swdown",
        "temp_air",
        "u",
        "v",
        "qair",
        "pb",
        "epot_air",
        "Eair",
        "petAcoef",
        "petBcoef",
        "peqAcoef",
        "peqBcoef",
        "precip_rain",
    )
)

ENERBIL_BEGIN_INPUTS = ("temp_sol", "temp_sol_pft", "lwdown", "swnet", "pb", "emis")

ENERBIL_BEGIN_SOURCE_OUTPUTS = (
    "psold",
    "psold_pft",
    "qsol_sat",
    "qsol_sat_pft",
    "pdqsold",
    "pdqsold_pft",
    "lwabs",
    "netrad",
    "netrad_pft",
)

ENERBIL_SURFTEMP_INPUTS = (
    "psold",
    "psold_pft",
    "qsol_sat",
    "qsol_sat_pft",
    "pdqsold",
    "pdqsold_pft",
    "netrad",
    "netrad_pft",
    "emis",
    "epot_air",
    "petAcoef",
    "petBcoef",
    "qair",
    "peqAcoef",
    "peqBcoef",
    "soilflx",
    "soilflx_pft",
    "rau",
    "u",
    "v",
    "q_cdrag",
    "q_cdrag_pft",
    "vbeta",
    "vbeta_pft",
    "valpha",
    "vbeta1",
    "vbeta5",
    "soilcap",
    "soilcap_pft",
    "veget_max",
)

ENERBIL_NON_WATCHOUT_DRIVER_SOURCE_OUTPUTS = (
    "epot_air",
    "petAcoef",
    "petBcoef",
    "peqAcoef",
    "peqBcoef",
)

ENERBIL_SURFTEMP_SOURCE_OUTPUTS = (
    "dtheta",
    "dtheta_pft",
    "psnew",
    "psnew_pft",
    "qsol_sat_new",
    "qsol_sat_new_pft",
    "temp_sol_new",
    "temp_sol_new_pft",
    "qair_new",
    "epot_air_new",
)

ENERBIL_SOIL_THERMAL_STATE_FIELDS = (
    "soilcap",
    "soilcap_pft",
    "soilflx",
    "soilflx_pft",
)

ENERBIL_DRIVER_ALBEDO_FIELDS = ("albedo_vis", "albedo_nir")

ENERBIL_THERMOSOIL_RESTART_SOURCE = (
    "jax_orchidee.sechiba.enerbil.read_enerbil_soil_thermal_state_restart; "
    "Fortran src_sechiba/thermosoil.f90::thermosoil_initialize lines 668-684"
)

ENERBIL_DRIVER_SWNET_FROM_RESTART_ALBEDO_SOURCE = (
    "jax_orchidee.sechiba.enerbil.driver_swnet_from_swdown_albedo; "
    "Fortran src_driver/dim2_driver.f90 lines 1139-1163; "
    "src_driver/orchideedriver.f90 lines 598, 659"
)

ENERBIL_CONDVEG_INITIAL_EMIS_SOURCE = (
    "jax_orchidee.sechiba.enerbil.condveg_initialized_emis; "
    "Fortran src_sechiba/condveg.f90::condveg_initialize lines 260-269; "
    "src_parameters/constantes.f90 lines 622-678; "
    "src_parameters/constantes_var.f90 lines 670-707"
)


def qsat_moisture_qsatcalc(temp_in, pres_in):
    """Return saturated humidity using ORCHIDEE ``qsatcalc`` table logic.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90``,
    subroutine ``qsatcalc``, lines 79-179, and ``qsfrict_init``, lines
    547-589. Constants ``msmlr_h2o`` and ``msmlr_air`` are from
    ``src_parameters/constantes_var.f90`` lines 382-383.

    ``diag_qsat`` is a source-fixed true parameter. Fortran therefore clamps
    only the integer table index for out-of-range temperatures, while retaining
    the original temperature in the interpolation offset.
    """

    temp = _as_1d("temp_in", temp_in)
    pres = _as_1d("pres_in", pres_in)
    _require_same_npts(temp, pres)
    table = _qsfrict_table()
    jt = jnp.clip(temp.astype(jnp.int32), 100, 369)
    zz_f = temp - jt.astype(temp.dtype)
    zz_a = jnp.take(table, jt)
    zz_b = jnp.take(table, jt + 1)
    return ((zz_b - zz_a) * zz_f + zz_a) / pres


def qsat_moisture_dev_qsatcalc(temp_in, pres_in):
    """Return ``dev_qsatcalc`` derivative using ORCHIDEE table logic.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90``,
    subroutine ``dev_qsatcalc``, lines 311-408, and ``qsfrict_init``, lines
    547-589.
    """

    temp = _as_1d("temp_in", temp_in)
    pres = _as_1d("pres_in", pres_in)
    _require_same_npts(temp, pres)
    table = _qsfrict_table()
    jt = jnp.clip((temp + 0.5).astype(jnp.int32), 100, 369)
    zz_f = temp + 0.5 - jt.astype(temp.dtype)
    zz_a = jnp.take(table, jt - 1)
    zz_b = jnp.take(table, jt)
    zz_c = jnp.take(table, jt + 1)
    return ((zz_c - 2.0 * zz_b + zz_a) * (zz_f - 1.0) + zz_c - zz_b) / pres


def sechiba_air_density_from_pb_temp_air(
    pb,
    temp_air,
    *,
    pa_par_hpa=PA_PAR_HPA,
    cte_molr=CTE_MOLR,
):
    """Return ``rau`` from ``sechiba_var_init``.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/sechiba.f90``,
    subroutine ``sechiba_main`` calls ``sechiba_var_init`` at line 982;
    ``sechiba_var_init`` lines 3069-3092 compute ``rau(ji) = pa_par_hpa *
    pb(ji) / (cte_molr * temp_air(ji))``. Constants are from
    ``src_parameters/constantes_var.f90`` lines 379 and 394.
    """

    pb = _as_1d("pb", pb)
    temp_air = _as_1d("temp_air", temp_air)
    _require_same_npts(pb, temp_air)
    return (
        jnp.asarray(pa_par_hpa, dtype=pb.dtype)
        * pb
        / (jnp.asarray(cte_molr, dtype=pb.dtype) * temp_air)
    )


def dim2_driver_non_watchout_energy_coupling_inputs(
    temp_air,
    zlev,
    qair,
    *,
    cp_air=CP_AIR,
    cte_grav=CTE_GRAV,
):
    """Return non-WATCHOUT, non-relaxation driver ENERBIL coupling inputs.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_driver/dim2_driver.f90``
    lines 914-920 set ``eair_obs = cp_air*tair_obs + cte_grav*zlev_vec`` for
    non-WATCHOUT forcing. Lines 995-1003, in the non-relaxation branch, set
    ``petAcoef=0``, ``peqAcoef=0``, ``petBcoef=eair_obs``, and
    ``peqBcoef=qair_obs``. ``readdim2.f90`` lines 646-650 contains the same
    non-WATCHOUT potential-air-energy formula.

    The helper is intentionally path-specific. Callers must only use it after
    auditing that the driver path is non-WATCHOUT and non-relaxation; it is not
    a fallback for WATCHOUT forcing, which reads ``Eair`` and PET/PEQ fields
    from forcing files.
    """

    temp_air = _as_1d("temp_air", temp_air)
    zlev = _as_1d("zlev", zlev)
    qair = _as_1d("qair", qair)
    _require_same_npts(temp_air, zlev, qair)

    epot_air = jnp.asarray(cp_air, dtype=temp_air.dtype) * temp_air + jnp.asarray(
        cte_grav, dtype=temp_air.dtype
    ) * zlev
    zero = jnp.zeros_like(epot_air)
    return {
        "epot_air": epot_air,
        "petAcoef": zero,
        "petBcoef": epot_air,
        "peqAcoef": jnp.zeros_like(qair),
        "peqBcoef": qair,
    }


def enerbil_begin_local_diagnostics(
    *,
    temp_sol,
    temp_sol_pft,
    lwdown,
    swnet,
    pb,
    emis,
    ok_laidev,
    cp_air=CP_AIR,
    c_stefan=C_STEFAN,
):
    """Compute explicit local ``enerbil_begin`` diagnostics.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/enerbil.f90``,
    subroutine ``enerbil_begin``, lines 735-873. This covers ``psold`` lines
    780-787, ``qsol_sat`` lines 790-796, ``pdqsold`` lines 817-832,
    ``lwabs`` line 861, and ``netrad`` lines 867-870.

    The Fortran loop starts at one-based PFT 2. For array outputs that Fortran
    does not assign for PFT1, this helper returns ``nan`` in column 0 rather
    than inventing a process value. For non-``ok_LAIdev`` PFTs, only
    ``psold_pft`` is explicitly copied from grid-cell ``psold`` by lines
    781-787; qsat, derivative, and net radiation PFT branches are still
    computed over PFTs 2..nvm in the source.
    """

    temp_sol = _as_1d("temp_sol", temp_sol)
    temp_sol_pft = _as_2d("temp_sol_pft", temp_sol_pft)
    lwdown = _as_1d("lwdown", lwdown)
    swnet = _as_1d("swnet", swnet)
    pb = _as_1d("pb", pb)
    emis = _as_1d("emis", emis)
    ok_laidev = jnp.asarray(ok_laidev, dtype=bool)
    _require_same_npts(temp_sol, temp_sol_pft, lwdown, swnet, pb, emis)
    if ok_laidev.ndim != 1 or ok_laidev.shape[0] != temp_sol_pft.shape[1]:
        raise ValueError("ok_laidev must be a one-dimensional array with length nvm")

    cp_air = jnp.asarray(cp_air, dtype=temp_sol.dtype)
    c_stefan = jnp.asarray(c_stefan, dtype=temp_sol.dtype)

    psold = temp_sol * cp_air
    psold_formula_pft = temp_sol_pft * cp_air
    psold_pft = jnp.where(ok_laidev[None, :], psold_formula_pft, psold[:, None])

    qsol_sat = qsat_moisture_qsatcalc(temp_sol, pb)
    qsol_sat_all_pft = _pft_qsat_or_derivative(temp_sol_pft, pb, qsat_moisture_qsatcalc)
    qsol_sat_pft = _mask_unassigned_pft1(qsol_sat_all_pft)

    pdqsold = qsat_moisture_dev_qsatcalc(temp_sol, pb)
    pdqsold_all_pft = _pft_qsat_or_derivative(temp_sol_pft, pb, qsat_moisture_dev_qsatcalc)
    pdqsold_pft = _mask_unassigned_pft1(pdqsold_all_pft)

    lwabs = emis * lwdown
    netrad = lwdown + swnet - (emis * c_stefan * temp_sol**4 + (1.0 - emis) * lwdown)
    netrad_pft_all = (
        lwdown[:, None]
        + swnet[:, None]
        - (emis[:, None] * c_stefan * temp_sol_pft**4 + (1.0 - emis[:, None]) * lwdown[:, None])
    )
    netrad_pft = _mask_unassigned_pft1(netrad_pft_all)

    return EnerbilBeginDiagnostics(
        psold=psold,
        psold_pft=psold_pft,
        qsol_sat=qsol_sat,
        qsol_sat_pft=qsol_sat_pft,
        pdqsold=pdqsold,
        pdqsold_pft=pdqsold_pft,
        lwabs=lwabs,
        netrad=netrad,
        netrad_pft=netrad_pft,
    )


def enerbil_surftemp_qsol_sat_update(
    *,
    qsol_sat,
    pdqsold,
    dtheta,
    qsol_sat_pft=None,
    pdqsold_pft=None,
    dtheta_pft=None,
    ok_laidev=None,
    cp_air=CP_AIR,
):
    """Update saturated surface humidity after the surface-temperature solve.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/enerbil.f90``,
    subroutine ``enerbil_surftemp``, lines 1197 and 1204-1212. The grid-cell
    update is ``qsol_sat + (1/cp_air) * pdqsold * dtheta``. PFT updates are
    emitted only when all PFT inputs plus ``ok_laidev`` are provided; inactive
    PFTs copy the grid-cell update exactly as in lines 1209-1212.

    ``dtheta`` is an explicit input from the upstream implicit energy-balance
    solve (lines 1172-1179). This helper does not solve that balance.
    """

    qsol_sat = _as_1d("qsol_sat", qsol_sat)
    pdqsold = _as_1d("pdqsold", pdqsold)
    dtheta = _as_1d("dtheta", dtheta)
    _require_same_npts(qsol_sat, pdqsold, dtheta)

    zicp = jnp.asarray(1.0, dtype=qsol_sat.dtype) / jnp.asarray(cp_air, dtype=qsol_sat.dtype)
    qsol_sat_new = qsol_sat + zicp * pdqsold * dtheta

    pft_inputs = (qsol_sat_pft, pdqsold_pft, dtheta_pft, ok_laidev)
    if all(value is None for value in pft_inputs):
        return qsol_sat_new
    if any(value is None for value in pft_inputs):
        raise ValueError(
            "qsol_sat_pft, pdqsold_pft, dtheta_pft, and ok_laidev are required together"
        )

    qsol_sat_pft = _as_2d("qsol_sat_pft", qsol_sat_pft)
    pdqsold_pft = _as_2d("pdqsold_pft", pdqsold_pft)
    dtheta_pft = _as_2d("dtheta_pft", dtheta_pft)
    needs_pft_forcing = any(bool(value) for value in ok_laidev)
    ok_laidev = jnp.asarray(ok_laidev, dtype=bool)
    pft_shape = qsol_sat_pft.shape
    if pdqsold_pft.shape != pft_shape or dtheta_pft.shape != pft_shape:
        raise ValueError("qsol_sat_pft, pdqsold_pft, and dtheta_pft must have matching shape")
    if ok_laidev.ndim != 1 or ok_laidev.shape[0] != pft_shape[1]:
        raise ValueError("ok_laidev must be a one-dimensional array with length nvm")
    if pft_shape[0] != qsol_sat_new.shape[0]:
        raise ValueError("PFT inputs must agree with qsol_sat on the npts axis")

    qsol_sat_new_pft = qsol_sat_pft + zicp * pdqsold_pft * dtheta_pft
    return qsol_sat_new, jnp.where(ok_laidev[None, :], qsol_sat_new_pft, qsol_sat_new[:, None])


def enerbil_surftemp_explicit_solve(
    *,
    psold,
    psold_pft,
    qsol_sat,
    qsol_sat_pft,
    pdqsold,
    pdqsold_pft,
    netrad,
    netrad_pft,
    emis,
    epot_air,
    petAcoef,
    petBcoef,
    qair,
    peqAcoef,
    peqBcoef,
    soilflx,
    soilflx_pft,
    rau,
    u,
    v,
    q_cdrag,
    q_cdrag_pft,
    vbeta,
    vbeta_pft,
    valpha,
    vbeta1,
    vbeta5,
    soilcap,
    soilcap_pft,
    veget_max,
    ok_laidev,
    dt_sechiba,
    min_wind=0.1,
    cp_air=CP_AIR,
    chalsu0=CHALSU0,
    chalev0=CHALEV0,
    c_stefan=C_STEFAN,
):
    """Solve the linearized ``enerbil_surftemp`` state update from inputs.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/enerbil.f90``,
    subroutine ``enerbil_surftemp``, lines 927-1245. This helper covers wind
    and transfer resistances lines 999-1012, old sensible/latent fluxes lines
    1027-1068, sensitivity terms lines 1080-1135, ``sum_old``/``sum_sns`` and
    ``dtheta`` lines 1144-1179, state updates lines 1188-1212, and
    ``epot_air_new``/``qair_new`` lines 1220-1239.

    All upstream inputs must be supplied explicitly. In particular this helper
    consumes ``psold*``, ``qsol_sat*``, ``pdqsold*``, and ``netrad*`` from the
    audited ``enerbil_begin`` path; it does not infer them from after-call
    traces. ``zlev`` is an ``enerbil_surftemp`` dummy argument in this source
    version but is not used in lines 927-1245, so it is intentionally not an
    argument here.
    """

    psold = _as_1d("psold", psold)
    psold_pft = _as_2d("psold_pft", psold_pft)
    qsol_sat = _as_1d("qsol_sat", qsol_sat)
    qsol_sat_pft = _as_2d("qsol_sat_pft", qsol_sat_pft)
    pdqsold = _as_1d("pdqsold", pdqsold)
    pdqsold_pft = _as_2d("pdqsold_pft", pdqsold_pft)
    netrad = _as_1d("netrad", netrad)
    netrad_pft = _as_2d("netrad_pft", netrad_pft)
    emis = _as_1d("emis", emis)
    epot_air = _as_1d("epot_air", epot_air)
    petAcoef = _as_1d("petAcoef", petAcoef)
    petBcoef = _as_1d("petBcoef", petBcoef)
    qair = _as_1d("qair", qair)
    peqAcoef = _as_1d("peqAcoef", peqAcoef)
    peqBcoef = _as_1d("peqBcoef", peqBcoef)
    soilflx = _as_1d("soilflx", soilflx)
    soilflx_pft = _as_2d("soilflx_pft", soilflx_pft)
    rau = _as_1d("rau", rau)
    u = _as_1d("u", u)
    v = _as_1d("v", v)
    q_cdrag = _as_1d("q_cdrag", q_cdrag)
    q_cdrag_pft = _as_2d("q_cdrag_pft", q_cdrag_pft)
    vbeta = _as_1d("vbeta", vbeta)
    vbeta_pft = _as_2d("vbeta_pft", vbeta_pft)
    valpha = _as_1d("valpha", valpha)
    vbeta1 = _as_1d("vbeta1", vbeta1)
    vbeta5 = _as_1d("vbeta5", vbeta5)
    soilcap = _as_1d("soilcap", soilcap)
    soilcap_pft = _as_2d("soilcap_pft", soilcap_pft)
    veget_max = _as_2d("veget_max", veget_max)
    ok_laidev = jnp.asarray(ok_laidev, dtype=bool)

    _require_same_npts(
        psold,
        psold_pft,
        qsol_sat,
        qsol_sat_pft,
        pdqsold,
        pdqsold_pft,
        netrad,
        netrad_pft,
        emis,
        epot_air,
        petAcoef,
        petBcoef,
        qair,
        peqAcoef,
        peqBcoef,
        soilflx,
        soilflx_pft,
        rau,
        u,
        v,
        q_cdrag,
        q_cdrag_pft,
        vbeta,
        vbeta_pft,
        valpha,
        vbeta1,
        vbeta5,
        soilcap,
        soilcap_pft,
        veget_max,
    )

    pft_shape = psold_pft.shape
    for name, value in (
        ("qsol_sat_pft", qsol_sat_pft),
        ("pdqsold_pft", pdqsold_pft),
        ("netrad_pft", netrad_pft),
        ("soilflx_pft", soilflx_pft),
        ("q_cdrag_pft", q_cdrag_pft),
        ("vbeta_pft", vbeta_pft),
        ("soilcap_pft", soilcap_pft),
        ("veget_max", veget_max),
    ):
        if value.shape != pft_shape:
            raise ValueError(f"{name} must have shape {pft_shape}")
    if ok_laidev.ndim != 1 or ok_laidev.shape[0] != pft_shape[1]:
        raise ValueError("ok_laidev must be a one-dimensional array with length nvm")

    one = jnp.asarray(1.0)
    zero = jnp.asarray(0.0)
    cp_air = jnp.asarray(cp_air)
    zicp = one / cp_air
    chalsu0 = jnp.asarray(chalsu0)
    chalev0 = jnp.asarray(chalev0)
    c_stefan = jnp.asarray(c_stefan)
    dt = jnp.asarray(dt_sechiba)

    speed = _wind_speed(u, v, min_wind=min_wind)
    zikt = one / (rau * speed * q_cdrag)
    zikq = zikt
    zikt_pft = one / (rau[:, None] * speed[:, None] * q_cdrag_pft)
    zikq_pft = zikt_pft

    ok = ok_laidev[None, :]
    active_pft = ok & (veget_max > zero)

    sensfl_old = (petBcoef - psold) / (zikt - petAcoef)
    sensfl_old_pft_formula = (petBcoef[:, None] - psold_pft) / (zikt_pft - petAcoef[:, None])
    sensfl_old_pft = jnp.where(ok, sensfl_old_pft_formula, sensfl_old[:, None])

    snow_beta = vbeta1 * (one - vbeta5)
    snow_beta_pft = snow_beta[:, None]
    larsub_old = (
        chalsu0
        * snow_beta
        * (peqBcoef - qsol_sat)
        / (zikq - snow_beta * peqAcoef)
    )
    larsub_old_pft_formula = (
        chalsu0
        * snow_beta_pft
        * (peqBcoef[:, None] - qsol_sat_pft)
        / (zikq_pft - snow_beta_pft * peqAcoef[:, None])
    )
    larsub_old_pft = jnp.where(ok, larsub_old_pft_formula, larsub_old[:, None])

    evap_beta = (one - vbeta1) * (one - vbeta5) * vbeta
    lareva_old = (
        chalev0
        * evap_beta
        * (peqBcoef - valpha * qsol_sat)
        / (zikq - evap_beta * peqAcoef)
        + chalev0 * vbeta5 * (peqBcoef - qsol_sat) / (zikq - vbeta5 * peqAcoef)
    )
    vbeta_pft_fraction = jnp.where(veget_max > zero, vbeta_pft / veget_max, zero)
    evap_beta_pft = (one - vbeta1[:, None]) * (one - vbeta5[:, None]) * vbeta_pft_fraction
    lareva_old_pft_formula = (
        chalev0
        * evap_beta_pft
        * (peqBcoef[:, None] - valpha[:, None] * qsol_sat_pft)
        / (zikq_pft - evap_beta_pft * peqAcoef[:, None])
        + chalev0
        * vbeta5[:, None]
        * (peqBcoef[:, None] - qsol_sat_pft)
        / (zikq_pft - vbeta5[:, None] * peqAcoef[:, None])
    )
    lareva_old_pft = jnp.where(active_pft, lareva_old_pft_formula, lareva_old[:, None])

    netrad_sns = zicp * 4.0 * emis * c_stefan * (zicp * psold) ** 3
    netrad_sns_pft_formula = (
        zicp
        * 4.0
        * emis[:, None]
        * c_stefan
        * (zicp * psold_pft) ** 3
    )
    netrad_sns_pft = jnp.where(ok, netrad_sns_pft_formula, netrad_sns[:, None])

    sensfl_sns = one / (zikt - petAcoef)
    sensfl_sns_pft_formula = one / (zikt_pft - petAcoef[:, None])
    sensfl_sns_pft = jnp.where(ok, sensfl_sns_pft_formula, sensfl_sns[:, None])

    larsub_sns = (
        chalsu0
        * snow_beta
        * zicp
        * pdqsold
        / (zikq - snow_beta * peqAcoef)
    )
    larsub_sns_pft_formula = (
        chalsu0
        * snow_beta_pft
        * zicp
        * pdqsold_pft
        / (zikq_pft - snow_beta_pft * peqAcoef[:, None])
    )
    larsub_sns_pft = jnp.where(ok, larsub_sns_pft_formula, larsub_sns[:, None])

    evap_sns_beta = evap_beta * valpha + vbeta5
    lareva_sns = (
        chalev0
        * evap_sns_beta
        * zicp
        * pdqsold
        / (zikq - evap_sns_beta * peqAcoef)
    )
    evap_sns_beta_pft = evap_beta_pft * valpha[:, None] + vbeta5[:, None]
    lareva_sns_pft_formula = (
        chalev0
        * evap_sns_beta_pft
        * zicp
        * pdqsold_pft
        / (zikq_pft - evap_sns_beta_pft * peqAcoef[:, None])
    )
    lareva_sns_pft = jnp.where(active_pft, lareva_sns_pft_formula, lareva_sns[:, None])

    sum_old = netrad + sensfl_old + larsub_old + lareva_old + soilflx
    sum_old_pft_formula = (
        netrad_pft
        + sensfl_old_pft
        + larsub_old_pft
        + lareva_old_pft
        + soilflx_pft
    )
    sum_old_pft = jnp.where(ok, sum_old_pft_formula, sum_old[:, None])

    sum_sns = netrad_sns + sensfl_sns + larsub_sns + lareva_sns
    sum_sns_pft_formula = netrad_sns_pft + sensfl_sns_pft + larsub_sns_pft + lareva_sns_pft
    sum_sns_pft = jnp.where(ok, sum_sns_pft_formula, sum_sns[:, None])

    dtheta = dt * sum_old / (zicp * soilcap + dt * sum_sns)
    dtheta_pft_formula = dt * sum_old_pft / (zicp * soilcap_pft + dt * sum_sns_pft)
    dtheta_pft = jnp.where(active_pft, dtheta_pft_formula, dtheta[:, None])

    psnew = psold + dtheta
    qsol_sat_new = qsol_sat + zicp * pdqsold * dtheta
    temp_sol_new = psnew / cp_air

    psnew_pft_formula = psold_pft + dtheta_pft
    qsol_sat_new_pft_formula = qsol_sat_pft + zicp * pdqsold_pft * dtheta_pft
    temp_sol_new_pft_formula = psnew_pft_formula / cp_air
    psnew_pft = jnp.where(ok, psnew_pft_formula, psnew[:, None])
    qsol_sat_new_pft = jnp.where(ok, qsol_sat_new_pft_formula, qsol_sat_new[:, None])
    temp_sol_new_pft = jnp.where(ok, temp_sol_new_pft_formula, temp_sol_new[:, None])

    epot_air_new = zikt * (sensfl_old - sensfl_sns * dtheta) + psnew
    fevap = (lareva_old - lareva_sns * dtheta) + (larsub_old - larsub_sns * dtheta)
    qair_new_formula = (
        zikq
        / (
            chalsu0 * snow_beta
            + chalev0 * (evap_sns_beta)
        )
        * fevap
        + qsol_sat_new
    )
    qair_new = jnp.where(jnp.abs(fevap) < jnp.finfo(fevap.dtype).eps, qair, qair_new_formula)

    return EnerbilSurftempResult(
        dtheta=dtheta,
        dtheta_pft=dtheta_pft,
        psnew=psnew,
        psnew_pft=psnew_pft,
        qsol_sat_new=qsol_sat_new,
        qsol_sat_new_pft=qsol_sat_new_pft,
        temp_sol_new=temp_sol_new,
        temp_sol_new_pft=temp_sol_new_pft,
        qair_new=qair_new,
        epot_air_new=epot_air_new,
    )


def enerbil_flux_local_diagnostics(
    *,
    emis,
    temp_sol,
    rau,
    u,
    v,
    q_cdrag,
    vbeta,
    valpha,
    vbeta1,
    vbeta5,
    qair,
    epot_air,
    psnew,
    qsol_sat_new,
    temp_sol_new,
    lwdown,
    swnet,
    dt_sechiba,
    min_wind=0.1,
    chalsu0=CHALSU0,
    chalev0=CHALEV0,
    c_stefan=C_STEFAN,
):
    """Compute algebraic diagnostics from ``enerbil_flux`` before correction.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/enerbil.f90``,
    subroutine ``enerbil_flux``, lines 1445-1545. This covers wind/drag,
    longwave, ``qsurf``, net radiation, total evaporation, latent/sublimation
    fluxes, sensible heat flux, ``lwnet``, and raw ``evapot``. It intentionally
    excludes the explicit-snow ablation branch (lines 1553-1588) and the Milly
    ``evapot_corr`` correction (lines 1592-1639), both of which require
    additional upstream inputs.
    """

    emis = _as_1d("emis", emis)
    temp_sol = _as_1d("temp_sol", temp_sol)
    rau = _as_1d("rau", rau)
    u = _as_1d("u", u)
    v = _as_1d("v", v)
    q_cdrag = _as_1d("q_cdrag", q_cdrag)
    vbeta = _as_1d("vbeta", vbeta)
    valpha = _as_1d("valpha", valpha)
    vbeta1 = _as_1d("vbeta1", vbeta1)
    vbeta5 = _as_1d("vbeta5", vbeta5)
    qair = _as_1d("qair", qair)
    epot_air = _as_1d("epot_air", epot_air)
    psnew = _as_1d("psnew", psnew)
    qsol_sat_new = _as_1d("qsol_sat_new", qsol_sat_new)
    temp_sol_new = _as_1d("temp_sol_new", temp_sol_new)
    lwdown = _as_1d("lwdown", lwdown)
    swnet = _as_1d("swnet", swnet)
    _require_same_npts(
        emis,
        temp_sol,
        rau,
        u,
        v,
        q_cdrag,
        vbeta,
        valpha,
        vbeta1,
        vbeta5,
        qair,
        epot_air,
        psnew,
        qsol_sat_new,
        temp_sol_new,
        lwdown,
        swnet,
    )

    one = jnp.asarray(1.0, dtype=qsol_sat_new.dtype)
    zero = jnp.asarray(0.0, dtype=qsol_sat_new.dtype)
    four = jnp.asarray(4.0, dtype=qsol_sat_new.dtype)
    speed = _wind_speed(u, v, min_wind=min_wind)
    qc = speed * q_cdrag
    chalsu0 = jnp.asarray(chalsu0, dtype=qsol_sat_new.dtype)
    chalev0 = jnp.asarray(chalev0, dtype=qsol_sat_new.dtype)
    c_stefan = jnp.asarray(c_stefan, dtype=qsol_sat_new.dtype)

    lwup = (
        emis * c_stefan * temp_sol**4
        + four * emis * c_stefan * temp_sol**3 * (temp_sol_new - temp_sol)
        + (one - emis) * lwdown
    )
    tsol_rad = (lwup / (emis * c_stefan)) ** (one / four)

    snow_or_flood_beta = vbeta1 * (one - vbeta5) + vbeta5
    evap_beta = (one - vbeta1) * (one - vbeta5) * vbeta
    qsurf_raw = (snow_or_flood_beta + evap_beta * valpha) * qsol_sat_new
    qsurf = jnp.maximum(qsurf_raw, qair)

    netrad = lwdown + swnet - lwup
    vevapp = (
        jnp.asarray(dt_sechiba, dtype=qsol_sat_new.dtype)
        * rau
        * qc
        * (
            snow_or_flood_beta * (qsol_sat_new - qair)
            + evap_beta * (valpha * qsol_sat_new - qair)
        )
    )
    fluxsubli = chalsu0 * rau * qc * vbeta1 * (one - vbeta5) * (qsol_sat_new - qair)
    fluxlat = (
        fluxsubli
        + chalev0 * rau * qc * vbeta5 * (qsol_sat_new - qair)
        + chalev0 * rau * qc * evap_beta * (valpha * qsol_sat_new - qair)
    )
    fluxsens = rau * qc * (psnew - epot_air)
    lwnet = lwdown - lwup
    evapot = jnp.maximum(
        zero,
        jnp.asarray(dt_sechiba, dtype=qsol_sat_new.dtype)
        * rau
        * qc
        * (qsol_sat_new - qair),
    )

    return EnerbilFluxDiagnostics(
        qsurf=qsurf,
        lwup=lwup,
        lwnet=lwnet,
        tsol_rad=tsol_rad,
        netrad=netrad,
        vevapp=vevapp,
        fluxlat=fluxlat,
        fluxsubli=fluxsubli,
        fluxsens=fluxsens,
        evapot=evapot,
    )


def enerbil_flux_evapot_corr(
    *,
    flux: EnerbilFluxDiagnostics,
    emis,
    rau,
    u,
    v,
    q_cdrag,
    epot_air,
    psnew,
    pb,
    min_wind=0.1,
    min_sechiba=MIN_SECHIBA,
    chalev0=CHALEV0,
    c_stefan=C_STEFAN,
    cp_air=CP_AIR,
):
    """Compute the exact Milly/Penman correction from ``enerbil_flux``.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/enerbil.f90``,
    subroutine ``enerbil_flux``, lines 1592-1640. This is the second pass in
    the Fortran routine: it reuses the diagnosed ``evapot`` and ``vevapp``,
    recomputes wind drag with the same ``min_wind`` logic, applies the Milly
    correction factor, and returns ``evapot_corr``.
    """

    emis = _as_1d("emis", emis)
    rau = _as_1d("rau", rau)
    u = _as_1d("u", u)
    v = _as_1d("v", v)
    q_cdrag = _as_1d("q_cdrag", q_cdrag)
    epot_air = _as_1d("epot_air", epot_air)
    psnew = _as_1d("psnew", psnew)
    pb = _as_1d("pb", pb)
    evapot = _as_1d("evapot", flux.evapot)
    vevapp = _as_1d("vevapp", flux.vevapp)
    _require_same_npts(emis, rau, u, v, q_cdrag, epot_air, psnew, pb, evapot, vevapp)

    speed = _wind_speed(u, v, min_wind=min_wind)
    qc = speed * q_cdrag
    tair = epot_air / jnp.asarray(cp_air, dtype=epot_air.dtype)
    grad_qsat = qsat_moisture_dev_qsatcalc(tair, pb)
    safe_evapot = jnp.where(evapot != 0.0, evapot, jnp.ones_like(evapot))
    vevapp_over_evapot = vevapp / safe_evapot

    active = (evapot > 0.0) & ((psnew - epot_air) != 0.0)
    raw_correction = (
        4.0 * emis * c_stefan * tair**3
        + rau * qc * cp_air
        + chalev0 * rau * qc * grad_qsat * vevapp_over_evapot
    )
    correction = jnp.where(active, raw_correction, 0.0)
    correction = jnp.where(
        active & (jnp.abs(raw_correction) > min_sechiba),
        chalev0 * rau * qc * grad_qsat * (1.0 - vevapp_over_evapot) / raw_correction,
        correction,
    )
    correction = jnp.maximum(0.0, correction)
    evapot_corr = evapot / (1.0 + correction)
    return EnerbilFluxEvapotCorrResult(correction=correction, evapot_corr=evapot_corr)


def enerbil_pottemp_pass_through(*, q_sol_pot, temp_sol_pot):
    """Return the exact current ``enerbil_pottemp`` outputs.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/enerbil.f90``,
    subroutine ``enerbil_pottemp``, lines 1263-1317. The current source body
    initializes ``dtheta`` and ``fevap`` to zero and then adds them to the two
    outputs, so the outputs are returned unchanged here as an explicit
    source-backed helper.
    """

    q_sol_pot = _as_1d("q_sol_pot", q_sol_pot)
    temp_sol_pot = _as_1d("temp_sol_pot", temp_sol_pot)
    _require_same_npts(q_sol_pot, temp_sol_pot)
    return EnerbilPottempResult(q_sol_pot=q_sol_pot, temp_sol_pot=temp_sol_pot)


def enerbil_t2mdiag(temp_air):
    """Return the exact 2 m diagnostic air temperature from ``enerbil_t2mdiag``.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/enerbil.f90``,
    subroutine ``enerbil_t2mdiag``, lines 1993-2021. The source assigns
    ``t2mdiag(:) = temp_air(:)`` without additional algebra.
    """

    return jnp.asarray(temp_air)


def enerbil_flux_explicit_snow_diagnostics(
    *,
    flux: EnerbilFluxDiagnostics,
    emis,
    temp_sol,
    temp_sol_new,
    lwdown,
    swnet,
    rau,
    u,
    v,
    q_cdrag,
    vbeta,
    valpha,
    vbeta1,
    vbeta5,
    qair,
    epot_air,
    qsol_sat_new,
    pb,
    precip_rain,
    snowdz,
    temp_air,
    pgflux,
    soilcap,
    dt_sechiba,
    ok_explicitsnow=True,
    min_wind=0.1,
    chalsu0=CHALSU0,
    chalev0=CHALEV0,
    c_stefan=C_STEFAN,
    cp_air=CP_AIR,
    tp_00=TP_00,
):
    """Compute ``enerbil_flux`` explicit-snow ``pgflux``/``temp_sol_add``.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/enerbil.f90``,
    subroutine ``enerbil_flux``, lines 1428-1436 and 1554-1588. The branch
    first updates the ``pgflux`` inout with net energy into the snowpack when
    ``ok_explicitsnow`` is true, then caps snow-ablation surface temperature at
    ``tp_00`` and returns the extra melt-energy temperature increment in
    ``temp_sol_add``.
    """

    emis = _as_1d("emis", emis)
    temp_sol = _as_1d("temp_sol", temp_sol)
    temp_sol_new = _as_1d("temp_sol_new", temp_sol_new)
    lwdown = _as_1d("lwdown", lwdown)
    swnet = _as_1d("swnet", swnet)
    rau = _as_1d("rau", rau)
    u = _as_1d("u", u)
    v = _as_1d("v", v)
    q_cdrag = _as_1d("q_cdrag", q_cdrag)
    vbeta = _as_1d("vbeta", vbeta)
    valpha = _as_1d("valpha", valpha)
    vbeta1 = _as_1d("vbeta1", vbeta1)
    vbeta5 = _as_1d("vbeta5", vbeta5)
    qair = _as_1d("qair", qair)
    epot_air = _as_1d("epot_air", epot_air)
    qsol_sat_new = _as_1d("qsol_sat_new", qsol_sat_new)
    pb = _as_1d("pb", pb)
    precip_rain = _as_1d("precip_rain", precip_rain)
    snowdz = _as_2d("snowdz", snowdz)
    temp_air = _as_1d("temp_air", temp_air)
    pgflux = _as_1d("pgflux", pgflux)
    soilcap = _as_1d("soilcap", soilcap)
    _require_same_npts(
        emis,
        temp_sol,
        temp_sol_new,
        lwdown,
        swnet,
        rau,
        u,
        v,
        q_cdrag,
        vbeta,
        valpha,
        vbeta1,
        vbeta5,
        qair,
        epot_air,
        qsol_sat_new,
        pb,
        precip_rain,
        snowdz,
        temp_air,
        pgflux,
        soilcap,
        flux.netrad,
        flux.fluxsens,
        flux.fluxlat,
    )

    dtype = qsol_sat_new.dtype
    one = jnp.asarray(1.0, dtype=dtype)
    zero = jnp.asarray(0.0, dtype=dtype)
    four = jnp.asarray(4.0, dtype=dtype)
    dt = jnp.asarray(dt_sechiba, dtype=dtype)
    chalsu0 = jnp.asarray(chalsu0, dtype=dtype)
    chalev0 = jnp.asarray(chalev0, dtype=dtype)
    c_stefan = jnp.asarray(c_stefan, dtype=dtype)
    cp_air = jnp.asarray(cp_air, dtype=dtype)
    tp_00 = jnp.asarray(tp_00, dtype=dtype)

    speed = _wind_speed(u, v, min_wind=min_wind)
    qc = speed * q_cdrag
    phpsnow = precip_rain * jnp.asarray(4.218e3, dtype=dtype) * (
        jnp.maximum(tp_00, temp_air) - tp_00
    ) / dt
    pgflux_explicit = flux.netrad - flux.fluxsens - flux.fluxlat + phpsnow

    qsol_sat_tmp = qsat_moisture_qsatcalc(jnp.full_like(pb, tp_00), pb)
    lwup_tmp = (
        emis * c_stefan * temp_sol**4
        + four * emis * c_stefan * temp_sol**3 * (tp_00 - temp_sol)
        + (one - emis) * lwdown
    )
    netrad_tmp = lwdown + swnet - lwup_tmp
    fluxsens_tmp = rau * qc * cp_air * (tp_00 - epot_air / cp_air)
    fluxlat_tmp = (
        chalsu0 * rau * qc * vbeta1 * (one - vbeta5) * (qsol_sat_tmp - qair)
        + chalev0 * rau * qc * vbeta5 * (qsol_sat_tmp - qair)
        + chalev0
        * rau
        * qc
        * (one - vbeta1)
        * (one - vbeta5)
        * vbeta
        * (valpha * qsol_sat_tmp - qair)
    )
    zgflux = netrad_tmp - fluxsens_tmp - fluxlat_tmp + phpsnow

    active = (
        bool(ok_explicitsnow)
        & (temp_sol_new > tp_00)
        & (jnp.sum(snowdz, axis=1) > zero)
        & (soilcap > zero)
    )
    temp_sol_add = jnp.where(active, -(pgflux_explicit - zgflux) * dt / soilcap, zero)
    pgflux_ablation = jnp.where(active, zgflux, pgflux_explicit)
    pgflux_out = jnp.where(bool(ok_explicitsnow), pgflux_ablation, pgflux)
    phpsnow_out = jnp.where(bool(ok_explicitsnow), phpsnow, zero)

    return EnerbilFluxExplicitSnowDiagnostics(
        pgflux=pgflux_out,
        temp_sol_add=temp_sol_add,
        phpsnow=phpsnow_out,
    )


def enerbil_fusion_step(
    *,
    tot_melt,
    soilcap,
    soilcap_pft,
    snowdz,
    temp_sol_new,
    temp_sol_new_pft,
    ok_laidev,
    ok_explicitsnow,
    dt_sechiba=1800.0,
    chalfu0=CHALSU0 - CHALEV0,
    tp_00=TP_00,
) -> EnerbilFusionResult:
    """Apply the snow/ice-melt surface-temperature update.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/enerbil.f90``,
    subroutine ``enerbil_fusion``, lines 1881-1953. In normal
    ``sechiba_main`` scheduling this routine is called only when
    ``OK_EXPLICITSNOW=n`` (``sechiba.f90`` lines 1077-1081). The explicit-snow
    branch is still represented because it exists in the subroutine body.
    """

    tot_melt = _as_1d("tot_melt", tot_melt)
    soilcap = _as_1d("soilcap", soilcap)
    soilcap_pft = _as_2d("soilcap_pft", soilcap_pft)
    snowdz = _as_2d("snowdz", snowdz)
    temp_sol_new = _as_1d("temp_sol_new", temp_sol_new)
    temp_sol_new_pft = _as_2d("temp_sol_new_pft", temp_sol_new_pft)
    _require_same_npts(tot_melt, soilcap, soilcap_pft, snowdz, temp_sol_new, temp_sol_new_pft)
    if soilcap_pft.shape != temp_sol_new_pft.shape:
        raise ValueError("soilcap_pft and temp_sol_new_pft must share shape (npts,nvm)")
    ok_laidev = jnp.asarray(ok_laidev, dtype=bool)
    if ok_laidev.ndim != 1 or ok_laidev.shape[0] != soilcap_pft.shape[1]:
        raise ValueError("ok_laidev must have shape (nvm,)")

    dtype = temp_sol_new.dtype
    tp_00 = jnp.asarray(tp_00, dtype=dtype)
    chalfu0 = jnp.asarray(chalfu0, dtype=dtype)
    fusion = jnp.zeros_like(temp_sol_new)

    if bool(ok_explicitsnow):
        cap = (jnp.sum(snowdz, axis=1) > 0.0) & (temp_sol_new >= tp_00)
        capped_grid = jnp.where(cap, tp_00, temp_sol_new)
        capped_pft = jnp.where(cap[:, None], tp_00, temp_sol_new_pft)
        return EnerbilFusionResult(
            temp_sol_new=capped_grid,
            temp_sol_new_pft=capped_pft,
            fusion=fusion,
            provenance=(
                "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_fusion lines 1881-1953",
                "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1077-1081",
            ),
            notes=(
                "Explicit-snow branch caps temperatures when snow layers exist; sechiba_main normally skips this call when OK_EXPLICITSNOW=y.",
            ),
        )

    fusion = tot_melt * chalfu0 / jnp.asarray(dt_sechiba, dtype=dtype)
    melt_cooling = tot_melt * chalfu0
    out_temp = temp_sol_new - melt_cooling / soilcap
    laidev_cooling = temp_sol_new_pft - melt_cooling[:, None] / soilcap_pft
    out_pft = jnp.where(ok_laidev[None, :], laidev_cooling, out_temp[:, None])
    return EnerbilFusionResult(
        temp_sol_new=out_temp,
        temp_sol_new_pft=out_pft,
        fusion=fusion,
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_fusion lines 1881-1953",
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1077-1081",
        ),
        notes=("Bucket-snow OK_EXPLICITSNOW=n branch; PFT temperatures follow ok_LAIdev exactly.",),
    )


def enerbil_flux_inputs_contract(available_fields):
    """Check whether an upstream payload can drive ``enerbil_flux`` algebra.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/enerbil.f90``,
    subroutine ``enerbil_flux``, lines 1367-1415 declare the required inputs,
    and lines 1486-1545 consume ``valpha``, ``qsol_sat_new``, drag, beta,
    radiation, thermodynamic, and timestep fields. The current
    ``after_enerbil_main`` bridge trace is an after-boundary output trace, not
    the pre-call input state; it also does not cover DIFFUCO ``valpha``.
    """

    available = set(available_fields)
    missing = tuple(name for name in ENERBIL_FLUX_REQUIRED_INPUTS if name not in available)
    outputs_only = tuple(name for name in ENERBIL_AFTER_MAIN_OUTPUTS_ONLY if name in available)
    return EnerbilFluxInputContract(
        ok=not missing,
        missing_inputs=missing,
        after_main_outputs_only=outputs_only,
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_flux lines 1367-1415, 1486-1545",
            "jax_orchidee/sechiba/diffuco_bridge.py notes after_diffuco_main trace writes beta fields but not valpha",
        ),
    )


def driver_swnet_from_swdown_albedo(*, swdown, albedo):
    """Compute driver ``swnet`` from downward shortwave and two-band albedo.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_driver/orchideedriver.f90``,
    main driver loop, lines 598 and 659. The exact formula is
    ``swnet(:) = (1 - (albedo(:,1) + albedo(:,2))/2) * swdown(:)``.

    This helper is deliberately narrow: it requires the same-step two-band
    ``albedo`` that the driver used. Supplying only ``swdown`` is not enough
    to close ENERBIL ``swnet``.
    """

    swdown = _as_1d("swdown", swdown)
    albedo = _as_2d("albedo", albedo)
    _require_same_npts(swdown, albedo)
    if albedo.shape[1] != 2:
        raise ValueError("albedo must have shape (npts, 2)")
    return (1.0 - (albedo[:, 0] + albedo[:, 1]) / 2.0) * swdown


def condveg_initialized_emis(
    *,
    npts: int,
    impaze: bool,
    emis_scal: float | None = None,
):
    """Return the exact ``emis`` vector initialized by ``condveg_initialize``.

    Fortran provenance:

    * ``fortran_source/ORCHIDEE/src_sechiba/condveg.f90``,
      subroutine ``condveg_initialize``, lines 260-269: if ``impaze`` is true,
      ``emis(:)=emis_scal``; otherwise ``emis_scal=un`` and
      ``emis(:)=emis_scal``.
    * ``fortran_source/ORCHIDEE/src_parameters/constantes.f90`` lines 622-678:
      ``IMPOSE_AZE`` is read first; ``CONDVEG_EMIS`` is read only inside the
      true branch.
    * ``fortran_source/ORCHIDEE/src_parameters/constantes_var.f90`` lines
      670-707: defaults are ``impaze=.FALSE.`` and ``emis_scal=1.0``.

    The true ``IMPOSE_AZE`` branch requires the caller to provide the audited
    same-case ``CONDVEG_EMIS`` value. The false branch uses the Fortran
    assignment to ``un`` and does not depend on a config default.
    """

    if npts < 1:
        raise ValueError("npts must be positive")
    if impaze:
        if emis_scal is None:
            raise ValueError("emis_scal is required when impaze is True")
        value = jnp.asarray(emis_scal)
        if value.ndim != 0:
            raise ValueError("emis_scal must be scalar")
    else:
        value = jnp.asarray(CONDVEG_EMIS_UNITY)
    return jnp.full((int(npts),), value, dtype=value.dtype)


def assemble_enerbil_first_step_precall_payload(
    *,
    driver_or_intersurf_payload: Mapping[str, object],
    driver_albedo_payload: Mapping[str, object],
    soil_thermal_restart_payload: Mapping[str, object],
    after_diffuco_payload: Mapping[str, object] | None = None,
    condveg_impaze: bool = False,
    condveg_emis_scal: float | None = None,
    non_watchout_driver_executed: bool = True,
) -> EnerbilFirstStepPrecallAssembly:
    """Assemble exact first-step ``enerbil_main`` pre-call fields.

    Fortran provenance: ``sechiba_main`` passes the post-DIFFUCO fields and
    forcing state into ``enerbil_main`` at lines 1013-1019. First-step
    ``swnet`` is computed by the dim2 driver from current ``swdown`` and the
    two-band driver restart albedo at ``dim2_driver.f90`` lines 1139-1164.
    ``soilcap*`` and ``soilflx*`` come from ``thermosoil_initialize`` restart
    reads at ``thermosoil.f90`` lines 668-684. ``emis`` is the
    ``condveg_initialize`` state from ``condveg.f90`` lines 260-269.

    This helper never reads files, never uses after-ENERBIL outputs, and never
    supplies defaults for DIFFUCO or thermal surface state. Missing fields are
    returned as explicit ``missing_inputs``.
    """

    aliases = {
        "tair": "temp_air",
        "psurf": "pb",
        "height_lev1": "zlev",
        "Height_Lev1": "zlev",
        "Eair": "epot_air",
    }
    alias_updates = {
        target: driver_or_intersurf_payload[source]
        for source, target in aliases.items()
        if target not in driver_or_intersurf_payload and source in driver_or_intersurf_payload
    }
    driver = driver_or_intersurf_payload if not alias_updates else {**driver_or_intersurf_payload, **alias_updates}
    albedo = driver_albedo_payload
    thermal = soil_thermal_restart_payload
    diffuco = {} if after_diffuco_payload is None else after_diffuco_payload

    payload: dict[str, np.ndarray] = {}
    driver_inputs: list[str] = []
    source_kernel_inputs: list[str] = []
    restart_inputs: list[str] = []
    diffuco_inputs: list[str] = []

    def as_runtime_array(value):
        return jnp.asarray(value) if isinstance(value, core.Tracer) else np.asarray(value)

    def add(name: str, value, bucket: list[str]) -> None:
        payload[name] = as_runtime_array(value)
        bucket.append(name)

    def infer_npts() -> int:
        for value in tuple(payload.values()) + tuple(thermal.values()) + tuple(diffuco.values()):
            arr = as_runtime_array(value)
            if arr.ndim > 0 and arr.shape[0] > 0:
                return int(arr.shape[0])
        raise ValueError("cannot initialize ENERBIL emis without any grid-point field")

    for field in (
        "zlev",
        "lwdown",
        "swdown",
        "temp_air",
        "u",
        "v",
        "qair",
        "pb",
        "epot_air",
        "petAcoef",
        "petBcoef",
        "peqAcoef",
        "peqBcoef",
        "precip_rain",
    ):
        if field in driver:
            add(field, driver[field], driver_inputs)

    for field in ("swnet", "rau"):
        if field in diffuco:
            add(field, diffuco[field], source_kernel_inputs)

    if "swnet" not in payload and "swnet" in driver:
        add("swnet", driver["swnet"], driver_inputs)
    elif "swnet" not in payload and "swdown" in payload and "albedo" in albedo:
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

    if "rau" not in payload and {"pb", "temp_air"} <= payload.keys():
        add(
            "rau",
            as_runtime_array(sechiba_air_density_from_pb_temp_air(payload["pb"], payload["temp_air"])),
            source_kernel_inputs,
        )

    if non_watchout_driver_executed and {"temp_air", "zlev", "qair"} <= payload.keys():
        energy = dim2_driver_non_watchout_energy_coupling_inputs(
            payload["temp_air"],
            payload["zlev"],
            payload["qair"],
        )
        for field in ENERBIL_NON_WATCHOUT_DRIVER_SOURCE_OUTPUTS:
            if field not in payload:
                add(field, as_runtime_array(energy[field]), source_kernel_inputs)

    if "emis" not in payload:
        add(
            "emis",
            as_runtime_array(
                condveg_initialized_emis(
                    npts=infer_npts(),
                    impaze=condveg_impaze,
                    emis_scal=condveg_emis_scal,
                )
            ),
            source_kernel_inputs,
        )

    for field in ENERBIL_SOIL_THERMAL_STATE_FIELDS:
        if field in thermal:
            add(field, thermal[field], restart_inputs)

    for field in ENERBIL_AFTER_DIFFUCO_PRECALL_FIELDS:
        if field in diffuco and field not in payload:
            add(field, diffuco[field], diffuco_inputs)
    if "valpha" not in payload and {"vbeta2", "vbeta3", "vbeta4"} <= payload.keys():
        vbeta4 = as_runtime_array(payload["vbeta4"])
        add("valpha", jnp.ones_like(vbeta4) if isinstance(vbeta4, core.Tracer) else np.ones_like(vbeta4), source_kernel_inputs)

    missing = tuple(field for field in ENERBIL_FIRST_STEP_INPUT_CHAIN_FIELDS if field not in payload)
    return EnerbilFirstStepPrecallAssembly(
        payload=payload,
        missing_inputs=missing,
        driver_inputs=tuple(dict.fromkeys(driver_inputs)),
        source_kernel_inputs=tuple(dict.fromkeys(source_kernel_inputs)),
        restart_inputs=tuple(dict.fromkeys(restart_inputs)),
        diffuco_inputs=tuple(dict.fromkeys(diffuco_inputs)),
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 997-1019",
            "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_main lines 390-471",
            "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 914-920, 995-1003, 1139-1164",
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_var_init lines 3069-3092",
            "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_initialize lines 260-269",
            "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_initialize lines 668-684",
        ),
        notes=(
            "The payload is first-step only because it consumes driver_start and sechiba_start restart state.",
            "after_diffuco_payload is accepted only as the compatible pre-ENERBIL boundary; after-ENERBIL outputs are not used.",
            "Missing temp_sol*, q_cdrag*, vbeta*, qsurf, evapot, and evapot_corr must be produced by exact upstream DIFFUCO/surface-state kernels or compatible boundary truth.",
        ),
    )


def run_enerbil_first_step_local_from_precall(
    precall: EnerbilFirstStepPrecallAssembly,
    *,
    ok_laidev,
    dt_sechiba: float,
    ok_explicitsnow: bool,
    min_wind: float = 0.1,
    use_jit: bool = False,
) -> EnerbilFirstStepLocalAssembly:
    """Execute first-step local ENERBIL kernels from an exact pre-call payload.

    Fortran provenance follows ``enerbil_main`` lines 485-545:
    ``enerbil_begin``, ``enerbil_surftemp``, ``enerbil_flux``, and
    ``enerbil_evapveg``. The caller must provide a pre-call assembly whose
    missing inputs are empty; this helper does not source or patch fields.
    """

    payload = dict(precall.payload)
    required_external = tuple(
        dict.fromkeys(
            (
                *ENERBIL_BEGIN_INPUTS,
                "ok_laidev",
                "epot_air",
                "petAcoef",
                "petBcoef",
                "qair",
                "peqAcoef",
                "peqBcoef",
                "soilflx",
                "soilflx_pft",
                "rau",
                "u",
                "v",
                "q_cdrag",
                "q_cdrag_pft",
                "vbeta",
                "vbeta_pft",
                "valpha",
                "vbeta1",
                "vbeta2",
                "vbeta3",
                "vbeta3pot",
                "vbeta4",
                "vbeta4_pft",
                "vbeta5",
                "soilcap",
                "soilcap_pft",
                "veget_max",
            )
        )
    )
    missing_external = tuple(field for field in required_external if field != "ok_laidev" and field not in payload)
    if missing_external:
        raise ValueError(f"ENERBIL first-step external inputs are incomplete: {missing_external}")
    ok_laidev_arg = tuple(bool(value) for value in ok_laidev) if bool(use_jit) else ok_laidev
    step_kernel = _enerbil_explicit_local_step_jit if bool(use_jit) else enerbil_explicit_local_step
    step = step_kernel(
        temp_sol=payload["temp_sol"],
        temp_sol_pft=payload["temp_sol_pft"],
        lwdown=payload["lwdown"],
        swnet=payload["swnet"],
        pb=payload["pb"],
        emis=payload["emis"],
        ok_laidev=ok_laidev_arg,
        epot_air=payload["epot_air"],
        petAcoef=payload["petAcoef"],
        petBcoef=payload["petBcoef"],
        qair=payload["qair"],
        peqAcoef=payload["peqAcoef"],
        peqBcoef=payload["peqBcoef"],
        soilflx=payload["soilflx"],
        soilflx_pft=payload["soilflx_pft"],
        rau=payload["rau"],
        u=payload["u"],
        v=payload["v"],
        q_cdrag=payload["q_cdrag"],
        q_cdrag_pft=payload["q_cdrag_pft"],
        vbeta=payload["vbeta"],
        vbeta_pft=payload["vbeta_pft"],
        valpha=payload["valpha"],
        vbeta1=payload["vbeta1"],
        vbeta2=payload["vbeta2"],
        vbeta3=payload["vbeta3"],
        vbeta3pot=payload["vbeta3pot"],
        vbeta4=payload["vbeta4"],
        vbeta4_pft=payload["vbeta4_pft"],
        vbeta5=payload["vbeta5"],
        soilcap=payload["soilcap"],
        soilcap_pft=payload["soilcap_pft"],
        veget_max=payload["veget_max"],
        dt_sechiba=dt_sechiba,
        precip_rain=payload.get("precip_rain"),
        snowdz=payload.get("snowdz"),
        temp_air=payload.get("temp_air"),
        pgflux=payload.get("soilflx"),
        ok_explicitsnow=ok_explicitsnow,
        min_wind=min_wind,
    )

    local_payload = {
        "psold": step.begin.psold,
        "psold_pft": step.begin.psold_pft,
        "qsol_sat": step.begin.qsol_sat,
        "qsol_sat_pft": step.begin.qsol_sat_pft,
        "pdqsold": step.begin.pdqsold,
        "pdqsold_pft": step.begin.pdqsold_pft,
        "lwabs": step.begin.lwabs,
        "netrad": step.begin.netrad,
        "netrad_pft": step.begin.netrad_pft,
        "dtheta": step.surftemp.dtheta,
        "dtheta_pft": step.surftemp.dtheta_pft,
        "psnew": step.surftemp.psnew,
        "psnew_pft": step.surftemp.psnew_pft,
        "qsol_sat_new": step.surftemp.qsol_sat_new,
        "qsol_sat_new_pft": step.surftemp.qsol_sat_new_pft,
        "temp_sol_new": step.surftemp.temp_sol_new,
        "temp_sol_new_pft": step.surftemp.temp_sol_new_pft,
        "qair_new": step.surftemp.qair_new,
        "epot_air_new": step.surftemp.epot_air_new,
        "qsurf": step.flux.qsurf,
        "fluxsens": step.flux.fluxsens,
        "fluxlat": step.flux.fluxlat,
        "fluxsubli": step.flux.fluxsubli,
        "vevapp": step.flux.vevapp,
        "lwup": step.flux.lwup,
        "lwnet": step.flux.lwnet,
        "tsol_rad": step.flux.tsol_rad,
        "evapot": step.flux.evapot,
        "evapot_corr": step.evapot_corr.evapot_corr,
        "vevapsno": step.evapveg_grid.vevapsno,
        "vevapnu": step.evapveg_grid.vevapnu,
        "vevapflo": step.evapveg_grid.vevapflo,
        "vevapnu_pft": step.evapveg_pft.vevapnu_pft,
        "vevapwet": step.evapveg_pft.vevapwet,
        "transpir": step.evapveg_pft.transpir,
        "transpot": step.evapveg_pft.transpot,
    }
    if step.explicit_snow is not None:
        local_payload.update(
            {
                "pgflux": step.explicit_snow.pgflux,
                "temp_sol_add": step.explicit_snow.temp_sol_add,
                "phpsnow": step.explicit_snow.phpsnow,
            }
        )
    if step.t2mdiag is not None:
        local_payload["t2mdiag"] = step.t2mdiag

    return EnerbilFirstStepLocalAssembly(
        step=step,
        payload={**payload, **local_payload},
        source_kernel_inputs=tuple(local_payload.keys()),
        passthrough_inputs=tuple(payload.keys()),
        provenance=(
            *precall.provenance,
            "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_main lines 485-545",
            "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_begin lines 735-873",
            "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_surftemp lines 927-1245",
            "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_flux lines 1367-1651",
            "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_evapveg lines 1691-1841",
        ),
    )


ENERBIL_EXPLICIT_LOCAL_STEP_JIT_STATIC_ARGNAMES = (
    "ok_laidev",
    "ok_explicitsnow",
)


def _condveg_emis_initialization_is_exact(
    *,
    condveg_initialize_executed: bool,
    impaze: bool | None,
    emis_scal: float | None,
) -> bool:
    """Return whether ``condveg_initialize`` has enough audited inputs for emis."""

    if not condveg_initialize_executed or impaze is None:
        return False
    return (not impaze) or emis_scal is not None


def _driver_swnet_formula_is_exact(
    *,
    driver_or_intersurf: set[str],
    driver_albedo: set[str],
) -> bool:
    """Return whether exact driver inputs exist for ``swdown, albedo -> swnet``."""

    has_swdown = "swdown" in driver_or_intersurf
    has_two_band_albedo = "albedo" in driver_albedo or all(
        field in driver_albedo for field in ENERBIL_DRIVER_ALBEDO_FIELDS
    )
    return has_swdown and has_two_band_albedo


def read_driver_albedo_restart(path: str | Path) -> DriverAlbedoRestart:
    """Read exact driver restart albedo and roughness for first-step driver coupling.

    Fortran provenance:

    * ``dim2_driver.f90`` lines 1139-1152 read ``albedo_vis`` and
      ``albedo_nir`` from ``rest_id`` after ``intersurf_initialize_2d``.
    * ``dim2_driver.f90`` lines 1154-1180 read ``z0`` from the driver restart
      and recompute ``for_u/for_v`` before the first ``intersurf_main_2d``.
    * ``dim2_driver.f90`` lines 1161-1164 recompute ``for_swnet`` from the
      restart/initialize albedo and current ``swdown`` before the first
      ``intersurf_main_2d`` call at lines 1293-1307.
    * ``dim2_driver.f90`` lines 1421-1422 write those two albedo bands back to
      the driver restart for subsequent runs.

    This helper only reads the driver restart bands and `z0`. It does not use
    ``sechiba_start`` ``soilalbedo_bg`` or history-output ``alb_vis/alb_nir``
    as a pre-ENERBIL albedo substitute.
    """

    fields = read_restart_fields(path, ENERBIL_DRIVER_ALBEDO_FIELDS)
    try:
        z0_fields = read_restart_fields(path, ("z0",))
        z0 = z0_fields["z0"]
    except KeyError:
        z0 = None
    albedo = np.stack((fields["albedo_vis"], fields["albedo_nir"]), axis=1)
    return DriverAlbedoRestart(
        path=Path(path),
        albedo=albedo,
        z0=z0,
        provenance=(
            "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 1139-1180",
            "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 1293-1307, 1421-1422",
            "fortran_source/ORCHIDEE/src_driver/orchideedriver.f90 lines 598, 659",
        ),
    )


def read_enerbil_soil_thermal_state_restart(path: str | Path) -> EnerbilSoilThermalStateRestart:
    """Read exact THERMOSOIL restart fields for ENERBIL first-step inputs.

    Fortran provenance:

    * ``thermosoil.f90::thermosoil_initialize`` lines 668-684 reads
      ``soilcap``, ``soilcap_pft``, ``soilflx``, and ``soilflx_pft`` from
      restart before the first ``sechiba_main`` ENERBIL call.
    * ``thermosoil.f90::thermosoil_initialize`` lines 725-735 recomputes the
      coefficients with ``thermosoil_coef`` only if any restart coefficient is
      absent.
    * ``thermosoil.f90::thermosoil_main`` lines 998-1010 recomputes these
      fields after ENERBIL for next timestep use, and
      ``thermosoil_finalize`` lines 1129-1136 writes them back to restart.

    This helper reads the local restart state only. It does not fill missing
    variables from ``after_enerbil_main`` or approximate a missing restart with
    defaults.
    """

    fields = read_restart_fields(path, ENERBIL_SOIL_THERMAL_STATE_FIELDS)
    return EnerbilSoilThermalStateRestart(
        path=Path(path),
        soilcap=fields["soilcap"],
        soilcap_pft=fields["soilcap_pft"],
        soilflx=fields["soilflx"],
        soilflx_pft=fields["soilflx_pft"],
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_initialize lines 668-684, 725-735",
            "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_main lines 998-1010",
            "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_finalize lines 1129-1136",
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_initialize lines 719-730",
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1013-1019, 1109-1118",
        ),
    )


def enerbil_swnet_input_chain_coverage(
    *,
    driver_or_intersurf_payload: Mapping[str, object] | Iterable[str] | None = None,
    driver_albedo_payload: Mapping[str, object] | Iterable[str] | None = None,
    after_enerbil_payload: Mapping[str, object] | Iterable[str] | None = None,
    source_kernel_outputs: Iterable[str] = (),
) -> EnerbilSwnetInputChainCoverage:
    """Classify exact coverage for ENERBIL pre-call ``swnet``.

    Fortran provenance:

    * ``sechiba.f90::sechiba_main`` lines 889-891 receive ``swnet`` and
      ``swdown`` as distinct inputs; lines 997-1019 pass ``swnet`` through
      DIFFUCO and into ENERBIL before same-step ``condveg_main`` updates
      ``albedo`` at lines 1084-1090.
    * ``dim2_driver.f90`` lines 1139-1164 read driver restart
      ``albedo_vis/albedo_nir`` and recompute ``for_swnet`` from current
      ``swdown`` before first-step ``intersurf_main_2d``.
    * ``orchideedriver.f90`` lines 598 and 659 use the same two-band formula
      around ``sechiba_initialize``.

    This helper does not compute numeric ``swnet``. It only reports whether
    exact inputs exist for either a direct ``swnet`` trace/source or the audited
    driver formula.
    """

    trace_sources: dict[str, str] = {}
    source_kernel_sources: dict[str, str] = {}
    driver_or_intersurf = _field_set(driver_or_intersurf_payload)
    driver_albedo = _field_set(driver_albedo_payload)

    if "swdown" in driver_or_intersurf:
        trace_sources["swdown"] = "driver_forcing/intersurf_main"
    if "swnet" in driver_or_intersurf:
        trace_sources["swnet"] = "driver_or_intersurf_pre_sechiba"

    for field in ("albedo", *ENERBIL_DRIVER_ALBEDO_FIELDS):
        if field in driver_albedo:
            source_kernel_sources[field] = "driver restart/intersurf_initialize albedo"

    explicit_source_outputs = frozenset(str(field) for field in source_kernel_outputs)
    if "swnet" in explicit_source_outputs:
        source_kernel_sources["swnet"] = "caller-advertised audited source kernel"

    formula_ready = _driver_swnet_formula_is_exact(
        driver_or_intersurf=set(driver_or_intersurf),
        driver_albedo=set(driver_albedo),
    )
    if "swnet" not in trace_sources and "swnet" not in source_kernel_sources and formula_ready:
        source_kernel_sources["swnet"] = ENERBIL_DRIVER_SWNET_FROM_RESTART_ALBEDO_SOURCE

    missing_formula_inputs = tuple(
        field
        for field in ("swdown", *ENERBIL_DRIVER_ALBEDO_FIELDS)
        if not (
            field in trace_sources
            or field in source_kernel_sources
            or (field in ENERBIL_DRIVER_ALBEDO_FIELDS and "albedo" in source_kernel_sources)
        )
    )
    missing = () if "swnet" in trace_sources or "swnet" in source_kernel_sources else ("swnet",)

    after_enerbil = _field_set(after_enerbil_payload)
    after_boundary_outputs = tuple(
        field for field in ("swnet", "swdown", "albedo", *ENERBIL_DRIVER_ALBEDO_FIELDS) if field in after_enerbil
    )

    ordered_fields = ("swdown", "albedo", *ENERBIL_DRIVER_ALBEDO_FIELDS, "swnet")
    return EnerbilSwnetInputChainCoverage(
        covered_by_trace=tuple(field for field in ordered_fields if field in trace_sources),
        covered_by_source_kernel=tuple(
            field
            for field in ordered_fields
            if field not in trace_sources and field in source_kernel_sources
        ),
        missing=missing,
        missing_formula_inputs=missing_formula_inputs,
        after_boundary_outputs_only=after_boundary_outputs,
        trace_sources=trace_sources,
        source_kernel_sources=source_kernel_sources,
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 889-891, 997-1019, 1084-1090",
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_initialize lines 690-697",
            "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_initialize lines 302-305",
            "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_main lines 431-434",
            "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_albedo lines 612-839",
            "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 1139-1164, 1293-1307, 1421-1422",
            "fortran_source/ORCHIDEE/src_driver/orchideedriver.f90 lines 512, 598, 644-659, 685-697",
            "fortran_source/ORCHIDEE/src_driver/readdim2.f90 lines 1347-1363, 1602-1616",
        ),
        notes=(
            "swdown alone does not cover swnet; the driver formula also needs exact two-band albedo.",
            "For the dim2 driver restart path, first-step swnet uses albedo_vis/albedo_nir read from driver_start.nc after intersurf_initialize_2d, then recomputed before intersurf_main_2d.",
            "sechiba_main calls condveg_main after ENERBIL, so same-step condveg_main albedo is next-step/current-output state, not the ENERBIL pre-call albedo.",
            "sechiba_start.nc soilalbedo_bg is a background soil component, not the final two-band driver albedo required by the swnet formula.",
            "History/reference alb_vis/alb_nir outputs are after model calls and are not used as pre-call albedo unless independently tied to the driver restart state.",
        ),
    )


def enerbil_explicit_local_step(
    *,
    temp_sol,
    temp_sol_pft,
    lwdown,
    swnet,
    pb,
    emis,
    ok_laidev,
    epot_air,
    petAcoef,
    petBcoef,
    qair,
    peqAcoef,
    peqBcoef,
    soilflx,
    soilflx_pft,
    rau,
    u,
    v,
    q_cdrag,
    q_cdrag_pft,
    vbeta,
    vbeta_pft,
    valpha,
    vbeta1,
    vbeta2,
    vbeta3,
    vbeta3pot,
    vbeta4,
    vbeta4_pft,
    vbeta5,
    soilcap,
    soilcap_pft,
    veget_max,
    dt_sechiba,
    q_sol_pot=None,
    temp_sol_pot=None,
    precip_rain=None,
    snowdz=None,
    temp_air=None,
    pgflux=None,
    ok_explicitsnow=True,
    min_wind=0.1,
):
    """Run the closed ENERBIL local kernels with explicit upstream inputs.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/enerbil.f90``,
    subroutine ``enerbil_main`` lines 485-545. The sequence is
    ``enerbil_begin`` (lines 485-489), ``enerbil_surftemp`` (lines 507-510),
    ``enerbil_pottemp`` (lines 523-526), ``enerbil_flux`` (lines 534-538),
    ``enerbil_evapveg`` (lines 541-545), and ``enerbil_t2mdiag`` (lines
    547-551).

    This is a process-order adapter only. It does not source, default, or
    approximate any ENERBIL pre-call input; callers must supply the exact
    source-backed fields. ``pgflux`` and ``temp_sol_add`` are returned only
    when all explicit-snow branch inputs are provided. The separate source
    helpers ``enerbil_pottemp_pass_through``, ``enerbil_flux_evapot_corr``,
    and ``enerbil_t2mdiag`` remain available for the rest of the ENERBIL call
    order.
    """

    begin = enerbil_begin_local_diagnostics(
        temp_sol=temp_sol,
        temp_sol_pft=temp_sol_pft,
        lwdown=lwdown,
        swnet=swnet,
        pb=pb,
        emis=emis,
        ok_laidev=ok_laidev,
    )
    surftemp = enerbil_surftemp_explicit_solve(
        psold=begin.psold,
        psold_pft=begin.psold_pft,
        qsol_sat=begin.qsol_sat,
        qsol_sat_pft=begin.qsol_sat_pft,
        pdqsold=begin.pdqsold,
        pdqsold_pft=begin.pdqsold_pft,
        netrad=begin.netrad,
        netrad_pft=begin.netrad_pft,
        emis=emis,
        epot_air=epot_air,
        petAcoef=petAcoef,
        petBcoef=petBcoef,
        qair=qair,
        peqAcoef=peqAcoef,
        peqBcoef=peqBcoef,
        soilflx=soilflx,
        soilflx_pft=soilflx_pft,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        q_cdrag_pft=q_cdrag_pft,
        vbeta=vbeta,
        vbeta_pft=vbeta_pft,
        valpha=valpha,
        vbeta1=vbeta1,
        vbeta5=vbeta5,
        soilcap=soilcap,
        soilcap_pft=soilcap_pft,
        veget_max=veget_max,
        ok_laidev=ok_laidev,
        dt_sechiba=dt_sechiba,
        min_wind=min_wind,
    )
    pottemp_inputs = (q_sol_pot, temp_sol_pot)
    if all(value is not None for value in pottemp_inputs):
        pottemp = enerbil_pottemp_pass_through(
            q_sol_pot=q_sol_pot,
            temp_sol_pot=temp_sol_pot,
        )
    else:
        pottemp = None
    flux = enerbil_flux_local_diagnostics(
        emis=emis,
        temp_sol=temp_sol,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        vbeta=vbeta,
        valpha=valpha,
        vbeta1=vbeta1,
        vbeta5=vbeta5,
        qair=surftemp.qair_new,
        epot_air=surftemp.epot_air_new,
        psnew=surftemp.psnew,
        qsol_sat_new=surftemp.qsol_sat_new,
        temp_sol_new=surftemp.temp_sol_new,
        lwdown=lwdown,
        swnet=swnet,
        dt_sechiba=dt_sechiba,
        min_wind=min_wind,
    )
    evapot_corr = enerbil_flux_evapot_corr(
        flux=flux,
        emis=emis,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        epot_air=surftemp.epot_air_new,
        psnew=surftemp.psnew,
        pb=pb,
        min_wind=min_wind,
    )
    explicit_snow_inputs = (precip_rain, snowdz, temp_air, pgflux)
    if all(value is not None for value in explicit_snow_inputs):
        explicit_snow = enerbil_flux_explicit_snow_diagnostics(
            flux=flux,
            emis=emis,
            temp_sol=temp_sol,
            temp_sol_new=surftemp.temp_sol_new,
            lwdown=lwdown,
            swnet=swnet,
            rau=rau,
            u=u,
            v=v,
            q_cdrag=q_cdrag,
            vbeta=vbeta,
            valpha=valpha,
            vbeta1=vbeta1,
            vbeta5=vbeta5,
            qair=surftemp.qair_new,
            epot_air=surftemp.epot_air_new,
            qsol_sat_new=surftemp.qsol_sat_new,
            pb=pb,
            precip_rain=precip_rain,
            snowdz=snowdz,
            temp_air=temp_air,
            pgflux=pgflux,
            soilcap=soilcap,
            dt_sechiba=dt_sechiba,
            ok_explicitsnow=ok_explicitsnow,
            min_wind=min_wind,
        )
    else:
        explicit_snow = None
    evapveg_grid = enerbil_evapveg_grid_fluxes(
        vbeta1=vbeta1,
        vbeta2=vbeta2,
        vbeta3=vbeta3,
        vbeta4=vbeta4,
        vbeta5=vbeta5,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        qair=surftemp.qair_new,
        qsol_sat_new=surftemp.qsol_sat_new,
        dt_sechiba=dt_sechiba,
        min_wind=min_wind,
    )
    evapveg_pft = enerbil_evapveg_pft_fluxes(
        vbeta1=vbeta1,
        vbeta2=vbeta2,
        vbeta3=vbeta3,
        vbeta3pot=vbeta3pot,
        vbeta4_pft=vbeta4_pft,
        vbeta5=vbeta5,
        veget_max=veget_max,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        q_cdrag_pft=q_cdrag_pft,
        qair=surftemp.qair_new,
        qsol_sat_new=surftemp.qsol_sat_new,
        qsol_sat_new_pft=surftemp.qsol_sat_new_pft,
        ok_laidev=ok_laidev,
        dt_sechiba=dt_sechiba,
        min_wind=min_wind,
    )
    return EnerbilExplicitStepResult(
        begin=begin,
        surftemp=surftemp,
        pottemp=pottemp,
        flux=flux,
        evapot_corr=evapot_corr,
        explicit_snow=explicit_snow,
        evapveg_grid=evapveg_grid,
        evapveg_pft=evapveg_pft,
        t2mdiag=enerbil_t2mdiag(temp_air) if temp_air is not None else None,
    )


_enerbil_explicit_local_step_jit = jit(
    enerbil_explicit_local_step,
    static_argnames=ENERBIL_EXPLICIT_LOCAL_STEP_JIT_STATIC_ARGNAMES,
)


def assemble_enerbil_explicit_local_step_from_server_bridge(
    *,
    after_diffuco_payload: Mapping[str, object],
    driver_or_intersurf_payload: Mapping[str, object],
    after_enerbil_payload: Mapping[str, object],
    soil_thermal_trace_input_payload: Mapping[str, object] | None = None,
    swnet_trace_input_payload: Mapping[str, object] | None = None,
    pottemp_trace_input_payload: Mapping[str, object] | None = None,
    ok_laidev: Iterable[bool] | None = None,
    dt_sechiba: float = 1800.0,
    zlev: float = 2.0,
    condveg_impaze: bool = False,
) -> EnerbilServerBridgeExplicitAssembly:
    """Assemble the explicit local ENERBIL step from one server bridge truth.

    Fortran provenance: ``sechiba.f90::sechiba_main`` calls
    ``diffuco_main`` before ``enerbil_main`` at lines 1013-1019, then calls
    ``thermosoil_main`` later at lines 1109-1118. ``enerbil_main`` declares
    ``soilflx_pft``, ``soilflx``, ``soilcap``, ``soilcap_pft``,
    ``q_cdrag``, and ``q_cdrag_pft`` as ``INTENT(in)`` at lines 433-438.
    ``thermosoil_main`` lines 995-1010 call ``thermosoil_coef`` to prepare
    ``soilcap*``/``soilflx*`` for the next ENERBIL timestep.

    This helper is for the server cold-start bridge run only. It never reads
    local restart files and never treats ``after_enerbil_main`` as an ENERBIL
    source kernel. If ``soilcap``/``soilcap_pft`` are supplied from
    ``after_enerbil_main``, they are explicitly recorded as trace input
    diagnostics because they are ``INTENT(in)`` in ENERBIL. Missing
    ``swnet`` or ``soilflx*`` stops assembly rather than triggering a default.
    """

    pft_count = int(after_diffuco_payload.get("jv", 14))
    pft_index = pft_count - 1

    inputs: dict[str, np.ndarray] = {}
    trace_input_diagnostics: list[str] = []
    source_kernel_inputs: list[str] = []
    after_boundary_input_diagnostics: list[str] = []

    def grid(value) -> np.ndarray:
        return np.asarray([value], dtype=np.float64)

    def pft14(value, *, fill=0.0) -> np.ndarray:
        data = np.full((1, pft_count), fill, dtype=np.float64)
        data[0, pft_index] = value
        return data

    def add_trace(name: str, value: np.ndarray) -> None:
        inputs[name] = value
        trace_input_diagnostics.append(name)

    def add_source(name: str, value: np.ndarray) -> None:
        inputs[name] = value
        source_kernel_inputs.append(name)

    for field in (
        "temp_sol",
        "q_cdrag",
        "vbeta",
        "valpha",
        "vbeta1",
        "vbeta4",
        "vbeta5",
        "emis",
    ):
        if field in after_diffuco_payload:
            add_trace(field, grid(after_diffuco_payload[field]))
    for field in (
        "temp_sol_pft",
        "q_cdrag_pft",
        "vbeta_pft",
        "vbeta2",
        "vbeta3",
        "vbeta3pot",
        "vbeta4_pft",
        "veget_max",
    ):
        if field in after_diffuco_payload:
            fill = after_diffuco_payload["temp_sol"] if field == "temp_sol_pft" else 0.0
            if field == "q_cdrag_pft":
                fill = after_diffuco_payload["q_cdrag"]
            add_trace(field, pft14(after_diffuco_payload[field], fill=fill))

    for field in (
        "lwdown",
        "pb",
        "qair",
        "u",
        "v",
        "temp_air",
        "rau",
        "epot_air",
        "petAcoef",
        "petBcoef",
        "peqAcoef",
        "peqBcoef",
    ):
        if field in driver_or_intersurf_payload:
            add_trace(field, grid(driver_or_intersurf_payload[field]))
    if "precip_rain" in driver_or_intersurf_payload:
        add_trace("precip_rain", grid(driver_or_intersurf_payload["precip_rain"]))
    if "snowdz_1" in driver_or_intersurf_payload:
        add_trace("snowdz", np.asarray([[driver_or_intersurf_payload["snowdz_1"]]], dtype=np.float64))
    elif "snowdz" in driver_or_intersurf_payload:
        add_trace("snowdz", np.asarray([[driver_or_intersurf_payload["snowdz"]]], dtype=np.float64))
    if "pgflux" in driver_or_intersurf_payload:
        add_trace("pgflux", grid(driver_or_intersurf_payload["pgflux"]))

    if swnet_trace_input_payload is not None and "swnet" in swnet_trace_input_payload:
        add_trace("swnet", grid(swnet_trace_input_payload["swnet"]))

    if pottemp_trace_input_payload is not None:
        for field in ("q_sol_pot", "temp_sol_pot"):
            if field in pottemp_trace_input_payload:
                add_trace(field, grid(pottemp_trace_input_payload[field]))

    if soil_thermal_trace_input_payload is not None:
        for field in ENERBIL_SOIL_THERMAL_STATE_FIELDS:
            if field in soil_thermal_trace_input_payload:
                value = soil_thermal_trace_input_payload[field]
                array = pft14(value) if field.endswith("_pft") else grid(value)
                add_trace(field, array)

    for field in ("soilcap", "soilcap_pft"):
        if (
            field not in inputs
            and field in after_enerbil_payload
            and after_enerbil_payload.get("kjit") == after_diffuco_payload.get("kjit")
        ):
            value = after_enerbil_payload[field]
            inputs[field] = pft14(value) if field.endswith("_pft") else grid(value)
            trace_input_diagnostics.append(field)
            after_boundary_input_diagnostics.append(field)

    if "rau" not in inputs and {"pb", "temp_air"} <= inputs.keys():
        add_source(
            "rau",
            np.asarray(sechiba_air_density_from_pb_temp_air(inputs["pb"], inputs["temp_air"])),
        )
    if {"temp_air", "qair"} <= inputs.keys():
        energy = dim2_driver_non_watchout_energy_coupling_inputs(
            inputs["temp_air"],
            grid(zlev),
            inputs["qair"],
        )
        for field in ENERBIL_NON_WATCHOUT_DRIVER_SOURCE_OUTPUTS:
            if field not in inputs:
                add_source(field, np.asarray(energy[field]))
    if "emis" not in inputs and not condveg_impaze:
        add_source("emis", np.asarray(condveg_initialized_emis(npts=1, impaze=False)))

    if "valpha" not in inputs and {"vbeta2", "vbeta3", "vbeta4"} <= inputs.keys():
        add_source("valpha", np.ones_like(inputs["vbeta4"]))

    required = (
        "temp_sol",
        "temp_sol_pft",
        "lwdown",
        "swnet",
        "pb",
        "emis",
        "epot_air",
        "petAcoef",
        "petBcoef",
        "qair",
        "peqAcoef",
        "peqBcoef",
        "soilflx",
        "soilflx_pft",
        "rau",
        "u",
        "v",
        "q_cdrag",
        "q_cdrag_pft",
        "vbeta",
        "vbeta_pft",
        "valpha",
        "vbeta1",
        "vbeta2",
        "vbeta3",
        "vbeta3pot",
        "vbeta4",
        "vbeta4_pft",
        "vbeta5",
        "soilcap",
        "soilcap_pft",
        "veget_max",
    )
    missing = tuple(field for field in required if field not in inputs)
    step = None
    if not missing:
        if ok_laidev is None:
            ok_laidev_array = np.zeros(pft_count, dtype=bool)
        else:
            ok_laidev_array = np.asarray(tuple(ok_laidev), dtype=bool)
            if ok_laidev_array.shape != (pft_count,):
                raise ValueError(f"ok_laidev must have length {pft_count}")
        step = enerbil_explicit_local_step(
            temp_sol=inputs["temp_sol"],
            temp_sol_pft=inputs["temp_sol_pft"],
            lwdown=inputs["lwdown"],
            swnet=inputs["swnet"],
            pb=inputs["pb"],
            emis=inputs["emis"],
            ok_laidev=ok_laidev_array,
            epot_air=inputs["epot_air"],
            petAcoef=inputs["petAcoef"],
            petBcoef=inputs["petBcoef"],
            qair=inputs["qair"],
            peqAcoef=inputs["peqAcoef"],
            peqBcoef=inputs["peqBcoef"],
            soilflx=inputs["soilflx"],
            soilflx_pft=inputs["soilflx_pft"],
            rau=inputs["rau"],
            u=inputs["u"],
            v=inputs["v"],
            q_cdrag=inputs["q_cdrag"],
            q_cdrag_pft=inputs["q_cdrag_pft"],
            vbeta=inputs["vbeta"],
            vbeta_pft=inputs["vbeta_pft"],
            valpha=inputs["valpha"],
            vbeta1=inputs["vbeta1"],
            vbeta2=inputs["vbeta2"],
            vbeta3=inputs["vbeta3"],
            vbeta3pot=inputs["vbeta3pot"],
            vbeta4=inputs["vbeta4"],
            vbeta4_pft=inputs["vbeta4_pft"],
            vbeta5=inputs["vbeta5"],
            soilcap=inputs["soilcap"],
            soilcap_pft=inputs["soilcap_pft"],
            veget_max=inputs["veget_max"],
            dt_sechiba=dt_sechiba,
            q_sol_pot=inputs.get("q_sol_pot"),
            temp_sol_pot=inputs.get("temp_sol_pot"),
            precip_rain=inputs.get("precip_rain"),
            snowdz=inputs.get("snowdz"),
            temp_air=inputs.get("temp_air"),
            pgflux=inputs.get("pgflux"),
        )

    return EnerbilServerBridgeExplicitAssembly(
        step=step,
        missing_inputs=missing,
        trace_input_diagnostics=tuple(dict.fromkeys(trace_input_diagnostics)),
        source_kernel_inputs=tuple(dict.fromkeys(source_kernel_inputs)),
        after_boundary_input_diagnostics=tuple(dict.fromkeys(after_boundary_input_diagnostics)),
        inputs=inputs,
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_main lines 433-438",
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1013-1019, 1109-1118",
            "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_main lines 995-1010",
            "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_coef lines 1386-1427",
            "outputs/server_1961_bridge_trace_20260624/traces/orchjax_sechiba_bridge_diffuco_trace.txt:after_diffuco_main",
            "outputs/server_1961_bridge_trace_20260624/traces/orchjax_intersurf_main_trace.txt:main",
            "outputs/server_1961_bridge_trace_20260624/traces/orchjax_sechiba_bridge_enerbil_trace.txt:after_enerbil_main",
        ),
        notes=(
            "This assembly uses only server bridge runtime truth; it does not read reference/case_001_071 restart files.",
            "soilcap and soilcap_pft from after_enerbil_main are marked as trace input diagnostics because enerbil_main declares them INTENT(in), not because a THERMOSOIL source port exists.",
            "THERMOSOIL source-kernel closure is not claimed; thermosoil_main updates the coefficients after ENERBIL for the next timestep.",
            "Missing swnet or soilflx* remain missing trace inputs and are not fabricated from swdown, restart files, or tuned constants.",
        ),
    )


def enerbil_surface_state_input_coverage(
    *,
    after_diffuco_payload: Mapping[str, object] | Iterable[str] | None = None,
    driver_or_intersurf_payload: Mapping[str, object] | Iterable[str] | None = None,
    driver_albedo_payload: Mapping[str, object] | Iterable[str] | None = None,
    after_enerbil_payload: Mapping[str, object] | Iterable[str] | None = None,
    soil_thermal_restart_payload: Mapping[str, object] | Iterable[str] | None = None,
    source_kernel_outputs: Iterable[str] = (),
    condveg_initialize_executed: bool = False,
    condveg_impaze: bool | None = None,
    condveg_emis_scal: float | None = None,
) -> EnerbilSurfaceStateInputCoverage:
    """Classify exact coverage for ``enerbil_begin/surftemp`` surface inputs.

    The audited fields are ``swnet``, ``emis``, ``soilflx*``, ``soilcap*``,
    and ``temp_sol*``. This helper never back-fills values from
    ``after_enerbil_main``: that trace is after the ENERBIL boundary and can
    verify outputs, but it is not a same-call precondition source.

    Fortran provenance:

    * ``enerbil.f90::enerbil_main`` lines 410-438 declare these as ENERBIL
      inputs/inout state; lines 488-509 pass them into ``enerbil_begin`` and
      ``enerbil_surftemp``.
    * ``sechiba.f90::sechiba_main`` lines 997-1019 place ``after_diffuco``
      before ENERBIL, so ``temp_sol``/``temp_sol_pft`` traced there are
      compatible pre-call state.
    * ``condveg.f90::condveg_initialize`` lines 260-269 and
      ``condveg_main`` lines 402-403 set ``emis`` from ``emis_scal``.
    * ``thermosoil.f90::thermosoil_initialize`` lines 668-684 reads
      ``soilcap*``/``soilflx*`` from restart, and lines 725-735 call
      ``thermosoil_coef`` if restart values are missing; ``thermosoil_coef``
      declares these outputs at lines 1423-1427 and assigns them at
      lines 1491-1494, 1567-1581, and 1715-1725.
    * ``sechiba.f90::sechiba_end`` lines 3114-3135 swaps
      ``temp_sol_new*`` to next-step ``temp_sol*``.
    * ``dim2_driver.f90`` lines 1139-1164 and ``orchideedriver.f90`` lines
      598 and 659 compute ``swnet`` exactly from same-step ``swdown`` and
      two-band ``albedo``.
    """

    trace_sources: dict[str, str] = {}
    source_kernel_sources: dict[str, str] = {}

    after_diffuco = _field_set(after_diffuco_payload)
    for field in ("temp_sol", "temp_sol_pft"):
        if field in after_diffuco:
            trace_sources[field] = "after_diffuco_main"

    driver_or_intersurf = _field_set(driver_or_intersurf_payload)
    if "swnet" in driver_or_intersurf:
        trace_sources["swnet"] = "driver_or_intersurf_pre_sechiba"

    driver_albedo = _field_set(driver_albedo_payload)
    if "swnet" not in trace_sources and _driver_swnet_formula_is_exact(
        driver_or_intersurf=set(driver_or_intersurf),
        driver_albedo=set(driver_albedo),
    ):
        source_kernel_sources["swnet"] = ENERBIL_DRIVER_SWNET_FROM_RESTART_ALBEDO_SOURCE

    explicit_source_outputs = frozenset(str(field) for field in source_kernel_outputs)
    for field in ENERBIL_SURFACE_STATE_INPUT_FIELDS:
        if field in explicit_source_outputs:
            source_kernel_sources[field] = "caller-advertised audited source kernel"

    soil_thermal_restart = _field_set(soil_thermal_restart_payload)
    for field in ENERBIL_SOIL_THERMAL_STATE_FIELDS:
        if field in soil_thermal_restart and field in ENERBIL_SURFACE_STATE_INPUT_FIELDS:
            source_kernel_sources[field] = ENERBIL_THERMOSOIL_RESTART_SOURCE

    if (
        "emis" not in trace_sources
        and "emis" not in source_kernel_sources
        and _condveg_emis_initialization_is_exact(
            condveg_initialize_executed=condveg_initialize_executed,
            impaze=condveg_impaze,
            emis_scal=condveg_emis_scal,
        )
    ):
        source_kernel_sources["emis"] = ENERBIL_CONDVEG_INITIAL_EMIS_SOURCE

    after_enerbil = _field_set(after_enerbil_payload)
    after_boundary_outputs = tuple(
        field
        for field in ENERBIL_SURFACE_STATE_INPUT_FIELDS
        if field in after_enerbil and field not in trace_sources and field not in source_kernel_sources
    )

    covered_by_trace = tuple(
        field for field in ENERBIL_SURFACE_STATE_INPUT_FIELDS if field in trace_sources
    )
    covered_by_source_kernel = tuple(
        field
        for field in ENERBIL_SURFACE_STATE_INPUT_FIELDS
        if field not in trace_sources and field in source_kernel_sources
    )
    missing = tuple(
        field
        for field in ENERBIL_SURFACE_STATE_INPUT_FIELDS
        if field not in trace_sources and field not in source_kernel_sources
    )

    return EnerbilSurfaceStateInputCoverage(
        covered_by_trace=covered_by_trace,
        covered_by_source_kernel=covered_by_source_kernel,
        missing=missing,
        after_boundary_outputs_only=after_boundary_outputs,
        trace_sources=trace_sources,
        source_kernel_sources=source_kernel_sources,
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_main lines 410-438, 488-509",
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 997-1019, 1085-1118, 1409-1410",
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_end lines 3114-3135",
            "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_initialize lines 260-269",
            "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_main lines 402-403",
            "fortran_source/ORCHIDEE/src_parameters/constantes.f90::config_sechiba_parameters lines 622-678",
            "fortran_source/ORCHIDEE/src_parameters/constantes_var.f90 lines 670-707",
            "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_initialize lines 668-684, 725-735",
            "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_main lines 998-1010",
            "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_finalize lines 1129-1136",
            "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_coef lines 1386-1427, 1491-1494, 1567-1581, 1715-1725",
            "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 1139-1164, 1421-1422",
            "fortran_source/ORCHIDEE/src_driver/orchideedriver.f90 lines 598, 659",
        ),
        notes=(
            "after_diffuco_main is after DIFFUCO and before ENERBIL, so temp_sol/temp_sol_pft there are compatible ENERBIL pre-call state.",
            "after_enerbil_main is after the ENERBIL boundary; its soilcap*, soilflx*, temp_sol*, or output diagnostics are not used to close this pre-call audit.",
            "swdown alone does not cover swnet; swnet needs a direct trace or the exact driver formula with same-step two-band albedo from driver restart/initialization state.",
            "emis is source-constructible from CONDVEG initialization only when condveg_initialize and the IMPOSE_AZE/CONDVEG_EMIS path are audited for the case.",
            "same-step condveg_main is after ENERBIL in sechiba_main and is not used to back-fill current ENERBIL emis.",
            "soilcap*/soilflx* are covered for first-step ENERBIL when read from the same local sechiba_start restart consumed by thermosoil_initialize.",
            "Runtime thermosoil_main updates soilcap*/soilflx* after ENERBIL for the next timestep; after_thermosoil state is not same-call pre-ENERBIL state unless the run-order target is the following timestep.",
            "If restart coefficients are absent, exact coverage requires a same-case thermosoil_coef port or trace of the thermosoil_initialize coefficient output, not defaults.",
        ),
    )


def enerbil_first_step_input_coverage(
    *,
    after_diffuco_payload: Mapping[str, object] | Iterable[str] | None = None,
    driver_or_intersurf_payload: Mapping[str, object] | Iterable[str] | None = None,
    driver_albedo_payload: Mapping[str, object] | Iterable[str] | None = None,
    after_enerbil_payload: Mapping[str, object] | Iterable[str] | None = None,
    soil_thermal_restart_payload: Mapping[str, object] | Iterable[str] | None = None,
    source_kernel_outputs: Iterable[str] = (),
    sechiba_var_init_executed: bool = False,
    non_watchout_driver_executed: bool = False,
    diffuco_comb_executed: bool = False,
    enerbil_begin_executed: bool = False,
    enerbil_surftemp_executed: bool = False,
    condveg_initialize_executed: bool = False,
    condveg_impaze: bool | None = None,
    condveg_emis_scal: float | None = None,
) -> EnerbilFirstStepInputCoverage:
    """Classify exact input coverage before first-step ``enerbil_main`` use.

    This is an audit helper only: it never computes, defaults, broadcasts, or
    back-fills process values. It distinguishes three exact sources:

    * ``covered_by_trace``: field name is present in an audited trace at a
      compatible boundary.
    * ``covered_by_source_kernel``: a local source-backed helper can supply the
      field, but only under its stated preconditions.
    * ``missing``: no exact trace or source-kernel output has been advertised.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/sechiba.f90``,
    subroutine ``sechiba_main``, lines 997-1019 show ``diffuco_main`` running
    immediately before ``enerbil_main`` and passing beta/drag fields onward.
    ``enerbil.f90::enerbil_main`` lines 390-471 declare the ENERBIL process
    boundary; lines 507-545 call ``enerbil_surftemp``, ``enerbil_flux``, and
    ``enerbil_evapveg``. ``diffuco.f90::diffuco_comb`` lines 3121-3122 set
    ``valpha(:)=un`` and lines 3256-3267 enforce the final beta bundle.

    ``diffuco_comb_executed=True`` means only that callers have audited that
    the same-step ``diffuco_comb`` path ran and that the supplied beta fields
    satisfy its contract. In that case this helper marks ``valpha`` as covered
    by ``diffuco_comb_explicit`` or its proven no-dew final-bundle shortcut.
    It is not a global default.

    ``sechiba_var_init_executed=True`` marks ``rau`` source-covered only when
    exact ``pb`` and ``temp_air`` are already covered, matching
    ``sechiba_var_init``.

    ``non_watchout_driver_executed=True`` marks the non-WATCHOUT,
    non-relaxation driver energy/coupling bundle covered only when exact
    ``temp_air``, ``zlev``, and ``qair`` are already covered. It is not a
    WATCHOUT fallback.

    ``enerbil_begin_executed=True`` marks ``enerbil_begin`` outputs covered
    only when all of its explicit inputs are already covered. That covers
    ``psold*``, ``qsol_sat*``, ``pdqsold*``, ``lwabs``, and ``netrad*``; it
    does not solve ``enerbil_surftemp`` or infer ``swnet`` from ``swdown``.

    ``enerbil_surftemp_executed=True`` marks ``enerbil_surftemp`` outputs
    covered only when every explicit input consumed by
    ``enerbil_surftemp_explicit_solve`` is already covered. It does not source
    missing aerodynamic coefficients, heat fluxes, ``swnet``, or emissivity.

    ``condveg_initialize_executed=True`` can mark ``emis`` covered only when
    the ``IMPOSE_AZE`` branch is explicit. If ``IMPOSE_AZE`` is false,
    ``condveg_initialize`` sets ``emis_scal=un`` and ``emis(:)=emis_scal``.
    If ``IMPOSE_AZE`` is true, the same-case ``CONDVEG_EMIS`` value must be
    supplied as ``condveg_emis_scal``. This does not use same-step
    ``condveg_main``, which is called after ENERBIL.
    """

    trace_sources: dict[str, str] = {}
    source_kernel_sources: dict[str, str] = {}

    after_diffuco = _field_set(after_diffuco_payload)
    for field in ENERBIL_AFTER_DIFFUCO_PRECALL_FIELDS:
        if field in after_diffuco:
            trace_sources[field] = "after_diffuco_main"

    driver_or_intersurf = _field_set(driver_or_intersurf_payload)
    driver_or_intersurf = set(driver_or_intersurf)
    if "tair" in driver_or_intersurf:
        driver_or_intersurf.add("temp_air")
    if "psurf" in driver_or_intersurf:
        driver_or_intersurf.add("pb")
    if "Height_Lev1" in driver_or_intersurf or "height_lev1" in driver_or_intersurf:
        driver_or_intersurf.add("zlev")
    if "Eair" in driver_or_intersurf:
        driver_or_intersurf.add("epot_air")
    for field in ENERBIL_DRIVER_OR_INTERSURF_TRACE_FIELDS:
        if field in driver_or_intersurf:
            canonical = {"height_lev1": "zlev", "Eair": "epot_air"}.get(field, field)
            trace_sources[canonical] = "driver_forcing/intersurf_main"

    driver_albedo = _field_set(driver_albedo_payload)
    if "swnet" not in trace_sources and _driver_swnet_formula_is_exact(
        driver_or_intersurf=driver_or_intersurf,
        driver_albedo=set(driver_albedo),
    ):
        source_kernel_sources["swnet"] = ENERBIL_DRIVER_SWNET_FROM_RESTART_ALBEDO_SOURCE

    explicit_source_outputs = frozenset(str(field) for field in source_kernel_outputs)
    for field in ENERBIL_FIRST_STEP_INPUT_CHAIN_FIELDS:
        if field in explicit_source_outputs:
            source_kernel_sources[field] = "caller-advertised audited source kernel"

    soil_thermal_restart = _field_set(soil_thermal_restart_payload)
    for field in ENERBIL_SOIL_THERMAL_STATE_FIELDS:
        if field in soil_thermal_restart and field in ENERBIL_FIRST_STEP_INPUT_CHAIN_FIELDS:
            source_kernel_sources[field] = ENERBIL_THERMOSOIL_RESTART_SOURCE

    if (
        "emis" not in trace_sources
        and "emis" not in source_kernel_sources
        and _condveg_emis_initialization_is_exact(
            condveg_initialize_executed=condveg_initialize_executed,
            impaze=condveg_impaze,
            emis_scal=condveg_emis_scal,
        )
    ):
        source_kernel_sources["emis"] = ENERBIL_CONDVEG_INITIAL_EMIS_SOURCE

    diffuco_comb_beta_inputs = ("vbeta2", "vbeta3", "vbeta4")
    if (
        diffuco_comb_executed
        and "valpha" not in trace_sources
        and all(field in trace_sources for field in diffuco_comb_beta_inputs)
    ):
        source_kernel_sources["valpha"] = (
            "jax_orchidee.sechiba.diffuco.diffuco_comb_explicit or "
            "diffuco_comb_final_beta_bundle_no_dew; Fortran "
            "src_sechiba/diffuco.f90::diffuco_comb lines 3121-3267"
        )

    if (
        sechiba_var_init_executed
        and "rau" not in trace_sources
        and all(field in trace_sources or field in source_kernel_sources for field in ("pb", "temp_air"))
    ):
        source_kernel_sources["rau"] = (
            "jax_orchidee.sechiba.enerbil.sechiba_air_density_from_pb_temp_air; "
            "Fortran src_sechiba/sechiba.f90::sechiba_var_init lines 3069-3092"
        )

    if (
        non_watchout_driver_executed
        and all(field in trace_sources or field in source_kernel_sources for field in ("temp_air", "zlev", "qair"))
    ):
        for field in ENERBIL_NON_WATCHOUT_DRIVER_SOURCE_OUTPUTS:
            source_kernel_sources.setdefault(
                field,
                (
                    "jax_orchidee.sechiba.enerbil.dim2_driver_non_watchout_energy_coupling_inputs; "
                    "Fortran src_driver/dim2_driver.f90 lines 914-920, 995-1003"
                ),
            )

    if (
        enerbil_begin_executed
        and all(
            field in trace_sources or field in source_kernel_sources
            for field in ENERBIL_BEGIN_INPUTS
        )
    ):
        for field in ENERBIL_BEGIN_SOURCE_OUTPUTS:
            source_kernel_sources.setdefault(
                field,
                (
                    "jax_orchidee.sechiba.enerbil.enerbil_begin_local_diagnostics; "
                    "Fortran src_sechiba/enerbil.f90::enerbil_begin lines 735-873"
                ),
            )

    if (
        enerbil_surftemp_executed
        and all(
            field in trace_sources or field in source_kernel_sources
            for field in ENERBIL_SURFTEMP_INPUTS
        )
    ):
        for field in ENERBIL_SURFTEMP_SOURCE_OUTPUTS:
            source_kernel_sources.setdefault(
                field,
                (
                    "jax_orchidee.sechiba.enerbil.enerbil_surftemp_explicit_solve; "
                    "Fortran src_sechiba/enerbil.f90::enerbil_surftemp lines 927-1245"
                ),
            )

    covered_by_trace = tuple(
        field for field in ENERBIL_FIRST_STEP_INPUT_CHAIN_FIELDS if field in trace_sources
    )
    covered_by_source_kernel = tuple(
        field
        for field in ENERBIL_FIRST_STEP_INPUT_CHAIN_FIELDS
        if field not in trace_sources and field in source_kernel_sources
    )
    missing = tuple(
        field
        for field in ENERBIL_FIRST_STEP_INPUT_CHAIN_FIELDS
        if field not in trace_sources and field not in source_kernel_sources
    )
    after_enerbil = _field_set(after_enerbil_payload)
    outputs_only = tuple(
        field
        for field in ENERBIL_AFTER_MAIN_OUTPUTS_ONLY
        if field in after_enerbil and field not in trace_sources and field not in source_kernel_sources
    )

    return EnerbilFirstStepInputCoverage(
        covered_by_trace=covered_by_trace,
        covered_by_source_kernel=covered_by_source_kernel,
        missing=missing,
        after_boundary_outputs_only=outputs_only,
        trace_sources=trace_sources,
        source_kernel_sources=source_kernel_sources,
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 997-1019",
            "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_main lines 390-471",
            "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_begin lines 735-873",
            "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90::qsatcalc/dev_qsatcalc lines 79-179, 311-408, 547-589",
            "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_surftemp lines 927-1245",
            "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_surftemp/flux/evapveg lines 507-545",
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_var_init lines 3069-3092",
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_initialize lines 719-730",
            "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 914-920, 995-1003",
            "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 1139-1164, 1421-1422",
            "fortran_source/ORCHIDEE/src_driver/readdim2.f90 lines 646-650, 1833-1857",
            "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_comb lines 3121-3122, 3256-3267",
            "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_initialize lines 668-684, 725-735",
            "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_main lines 998-1010",
            "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_finalize lines 1129-1136",
            "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_initialize lines 260-269",
            "fortran_source/ORCHIDEE/src_parameters/constantes.f90 lines 622-678",
            "fortran_source/ORCHIDEE/src_parameters/constantes_var.f90 lines 670-707",
            "outputs/server_1961_bridge_trace_20260624/traces/orchjax_sechiba_bridge_diffuco_trace.txt:after_diffuco_main",
            "outputs/server_1961_bridge_trace_20260624/traces/orchjax_sechiba_bridge_enerbil_trace.txt:after_enerbil_main output boundary",
            "outputs/server_1961_trace_full_20260623/traces/orchjax_driver_forcing_trace.txt:first_forcing",
            "outputs/server_1961_trace_full_20260623/traces/orchjax_intersurf_main_trace.txt:main",
        ),
        notes=(
            "after_diffuco_main is after DIFFUCO and before ENERBIL, so beta/drag fields there can cover ENERBIL pre-call inputs.",
            "after_enerbil_main fields, including soilcap*/soilflx* diagnostics written there, are not used as same-call ENERBIL pre-call inputs.",
            "swdown coverage from driver/intersurf does not cover swnet unless exact same-step two-band albedo from the driver restart/initialization state is also supplied; no shortwave-net approximation is made.",
            "zlev is an enerbil_surftemp interface input in this source, but the local enerbil_surftemp solve body does not consume zlev.",
            "sechiba_var_init can close rau from exact pb and temp_air; rau is not read from the current intersurf trace.",
            "non-WATCHOUT, non-relaxation driver code can close epot_air and PET/PEQ coefficients only after that driver branch is explicitly audited.",
            "WATCHOUT forcing reads Eair and PET/PEQ fields from forcing files; current driver/intersurf traces do not contain those columns.",
            "soilcap*/soilflx* first-step ENERBIL inputs are exact when supplied from the same sechiba_start restart read by thermosoil_initialize.",
            "thermosoil_main updates soilcap*/soilflx* after ENERBIL and writes them at finalize for restart/next-step use; after_enerbil_main remains after-boundary only.",
            "enerbil_begin can close old-state qsat/psold/netrad inputs only after temp_sol/temp_sol_pft/lwdown/swnet/pb/emis are exact.",
            "qsol_sat_new, psnew, dtheta, qair_new, and epot_air_new remain missing unless a source-backed enerbil_surftemp solve advertises them.",
            "condveg_initialize can close first-step emis from audited IMPOSE_AZE/CONDVEG_EMIS state; same-step condveg_main is after ENERBIL and is not a current-step source.",
            "same-step condveg_main updates albedo after ENERBIL; first-step swnet uses driver albedo state prepared before sechiba_main.",
        ),
    )


def enerbil_evapveg_grid_fluxes(
    *,
    vbeta1,
    vbeta2,
    vbeta3,
    vbeta4,
    vbeta5,
    rau,
    u,
    v,
    q_cdrag,
    qair,
    qsol_sat_new,
    dt_sechiba,
    min_wind=0.1,
):
    """Compute grid-cell ``enerbil_evapveg`` evaporation components.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/enerbil.f90``,
    subroutine ``enerbil_evapveg``, lines 1738-1787. The wind lower bound
    follows ``src_parameters/constantes_var.f90`` line 460 and
    ``src_parameters/constantes.f90`` lines 447-453; ``dt_sechiba`` is the
    caller-supplied control timestep from ``src_parameters/control.f90`` lines
    48-57.

    This kernel requires ``qsol_sat_new`` from the preceding
    ``enerbil_surftemp`` solve as an explicit input. It does not infer it from
    after-call traces or diagnostics.
    """

    vbeta1 = _as_1d("vbeta1", vbeta1)
    vbeta2 = _as_2d("vbeta2", vbeta2)
    vbeta3 = _as_2d("vbeta3", vbeta3)
    vbeta4 = _as_1d("vbeta4", vbeta4)
    vbeta5 = _as_1d("vbeta5", vbeta5)
    rau = _as_1d("rau", rau)
    u = _as_1d("u", u)
    v = _as_1d("v", v)
    q_cdrag = _as_1d("q_cdrag", q_cdrag)
    qair = _as_1d("qair", qair)
    qsol_sat_new = _as_1d("qsol_sat_new", qsol_sat_new)
    _require_same_npts(
        vbeta1,
        vbeta2,
        vbeta3,
        vbeta4,
        vbeta5,
        rau,
        u,
        v,
        q_cdrag,
        qair,
        qsol_sat_new,
    )
    if vbeta2.shape != vbeta3.shape:
        raise ValueError("vbeta2 and vbeta3 must have matching shape (npts, nvm)")

    speed = _wind_speed(u, v, min_wind=min_wind)
    exchange = jnp.asarray(dt_sechiba) * rau * speed * q_cdrag * (qsol_sat_new - qair)
    one = jnp.asarray(1.0, dtype=exchange.dtype)

    return EnerbilEvapvegGridFluxes(
        vevapsno=(one - vbeta5) * vbeta1 * exchange,
        vevapnu=(one - vbeta1) * (one - vbeta5) * vbeta4 * exchange,
        vevapflo=vbeta5 * (one - jnp.sum(vbeta2, axis=1) - jnp.sum(vbeta3, axis=1)) * exchange,
    )


def enerbil_evapveg_pft_fluxes(
    *,
    vbeta1,
    vbeta2,
    vbeta3,
    vbeta3pot,
    vbeta4_pft,
    vbeta5,
    veget_max,
    rau,
    u,
    v,
    q_cdrag,
    q_cdrag_pft=None,
    qair,
    qsol_sat_new,
    qsol_sat_new_pft=None,
    ok_laidev,
    dt_sechiba,
    min_wind=0.1,
):
    """Compute PFT-resolved ``enerbil_evapveg`` evaporation components.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/enerbil.f90``,
    subroutine ``enerbil_evapveg``, lines 1768-1781 and 1807-1833. The
    ``ok_laidev`` branch consumes PFT-specific drag and saturated humidity
    produced by ``enerbil_surftemp`` lines 1204-1212; non-``ok_laidev`` PFTs
    use grid-cell ``q_cdrag`` and ``qsol_sat_new`` exactly as in the Fortran.
    The wind lower bound follows ``src_parameters/constantes_var.f90`` line
    460 and ``src_parameters/constantes.f90`` lines 447-453; ``dt_sechiba`` is
    caller supplied from ``src_parameters/control.f90`` lines 48-57.

    If any PFT has ``ok_laidev=True``, callers must provide
    ``q_cdrag_pft`` and ``qsol_sat_new_pft``. This explicit contract prevents
    silently filling DIFFUCO/``enerbil_surftemp`` coverage gaps.
    """

    vbeta1 = _as_1d("vbeta1", vbeta1)
    vbeta2 = _as_2d("vbeta2", vbeta2)
    vbeta3 = _as_2d("vbeta3", vbeta3)
    vbeta3pot = _as_2d("vbeta3pot", vbeta3pot)
    vbeta4_pft = _as_2d("vbeta4_pft", vbeta4_pft)
    vbeta5 = _as_1d("vbeta5", vbeta5)
    veget_max = _as_2d("veget_max", veget_max)
    rau = _as_1d("rau", rau)
    u = _as_1d("u", u)
    v = _as_1d("v", v)
    q_cdrag = _as_1d("q_cdrag", q_cdrag)
    qair = _as_1d("qair", qair)
    qsol_sat_new = _as_1d("qsol_sat_new", qsol_sat_new)
    needs_pft_forcing = any(bool(value) for value in ok_laidev)
    ok_laidev = jnp.asarray(ok_laidev, dtype=bool)

    pft_shape = vbeta2.shape
    for name, value in (
        ("vbeta3", vbeta3),
        ("vbeta3pot", vbeta3pot),
        ("vbeta4_pft", vbeta4_pft),
        ("veget_max", veget_max),
    ):
        if value.shape != pft_shape:
            raise ValueError(f"{name} must have shape {pft_shape}")
    if ok_laidev.ndim != 1 or ok_laidev.shape[0] != pft_shape[1]:
        raise ValueError("ok_laidev must be a one-dimensional array with length nvm")

    _require_same_npts(vbeta1, vbeta2, vbeta5, rau, u, v, q_cdrag, qair, qsol_sat_new)

    if needs_pft_forcing and (q_cdrag_pft is None or qsol_sat_new_pft is None):
        raise ValueError(
            "q_cdrag_pft and qsol_sat_new_pft are required when any ok_laidev PFT is active"
        )
    if q_cdrag_pft is None:
        q_cdrag_pft = jnp.broadcast_to(q_cdrag[:, None], pft_shape)
    else:
        q_cdrag_pft = _as_2d("q_cdrag_pft", q_cdrag_pft)
    if qsol_sat_new_pft is None:
        qsol_sat_new_pft = jnp.broadcast_to(qsol_sat_new[:, None], pft_shape)
    else:
        qsol_sat_new_pft = _as_2d("qsol_sat_new_pft", qsol_sat_new_pft)
    if q_cdrag_pft.shape != pft_shape:
        raise ValueError("q_cdrag_pft must have shape (npts, nvm)")
    if qsol_sat_new_pft.shape != pft_shape:
        raise ValueError("qsol_sat_new_pft must have shape (npts, nvm)")

    speed = _wind_speed(u, v, min_wind=min_wind)
    one = jnp.asarray(1.0, dtype=qsol_sat_new.dtype)
    grid_xx = (
        jnp.asarray(dt_sechiba)
        * (one - vbeta1)
        * (qsol_sat_new - qair)
        * rau
        * speed
        * q_cdrag
    )
    pft_xx = (
        jnp.asarray(dt_sechiba)
        * (one - vbeta1[:, None])
        * (qsol_sat_new_pft - qair[:, None])
        * rau[:, None]
        * speed[:, None]
        * q_cdrag_pft
    )
    xxtemp = jnp.where(ok_laidev[None, :], pft_xx, grid_xx[:, None])

    vevapnu_grid_branch = (
        (one - vbeta1[:, None])
        * (one - vbeta5[:, None])
        * vbeta4_pft
        * jnp.asarray(dt_sechiba)
        * rau[:, None]
        * speed[:, None]
        * q_cdrag[:, None]
        * (qsol_sat_new[:, None] - qair[:, None])
    )
    vevapnu_pft_branch = (
        (one - vbeta1[:, None])
        * (one - vbeta5[:, None])
        * vbeta4_pft
        * jnp.asarray(dt_sechiba)
        * rau[:, None]
        * speed[:, None]
        * q_cdrag_pft
        * (qsol_sat_new_pft - qair[:, None])
    )
    vevapnu_pft = jnp.where(ok_laidev[None, :], vevapnu_pft_branch, vevapnu_grid_branch)
    vevapnu_pft = jnp.where(veget_max > 0.0, vevapnu_pft, 0.0)

    return EnerbilEvapvegPFTFluxes(
        vevapnu_pft=vevapnu_pft,
        vevapwet=xxtemp * vbeta2,
        transpir=xxtemp * vbeta3,
        transpot=xxtemp * vbeta3pot,
    )


def _wind_speed(u, v, *, min_wind):
    return jnp.maximum(jnp.asarray(min_wind), jnp.sqrt(u * u + v * v))


def _qsfrict_table():
    temp = jnp.arange(371, dtype=jnp.float64)
    zrapp = jnp.asarray(MSMLR_H2O / MSMLR_AIR, dtype=jnp.float64)
    zcorr = jnp.asarray(0.00320991, dtype=jnp.float64)
    solid = zrapp * 10.0 ** (
        2.07023
        - zcorr * temp
        - 2484.896 / temp
        + 3.56654 * jnp.log10(temp)
    )
    liquid = zrapp * 10.0 ** (
        23.8319
        - 2948.964 / temp
        - 5.028 * jnp.log10(temp)
        - 29810.16 * jnp.exp(-0.0699382 * temp)
        + 25.21935 * jnp.exp(-2999.924 / temp)
    )
    table = jnp.where(temp < 273.0, solid, liquid)
    return table.at[:101].set(0.0)


def _pft_qsat_or_derivative(temp_sol_pft, pb, kernel):
    values = [kernel(temp_sol_pft[:, jv], pb) for jv in range(temp_sol_pft.shape[1])]
    return jnp.stack(values, axis=1)


def _mask_unassigned_pft1(array):
    if array.shape[1] == 0:
        return array
    nan_column = jnp.full_like(array[:, :1], jnp.nan)
    return jnp.concatenate((nan_column, array[:, 1:]), axis=1)


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
        raise ValueError(f"{name} must be a two-dimensional array with shape (npts, nvm)")
    return array


def _field_set(payload):
    if payload is None:
        return frozenset()
    if isinstance(payload, Mapping):
        return frozenset(str(key) for key in payload.keys())
    return frozenset(str(item) for item in payload)


def _require_same_npts(*arrays):
    npts = arrays[0].shape[0]
    for array in arrays[1:]:
        if array.shape[0] != npts:
            raise ValueError("all inputs must agree on the npts axis")


def enerbil_flux_source_routed(**kwargs):
    """Production entry point composing all ``enerbil_flux`` process spans."""
    from .science_completion import enerbil_flux_source_routed as owner
    return owner(**kwargs)


def enerbil_finalize_restart_packet(**kwargs):
    """Production wrapper for the complete ENERBIL finalize packet."""

    from .science_completion import enerbil_finalize_restart_packet as owner

    return owner(**kwargs)
