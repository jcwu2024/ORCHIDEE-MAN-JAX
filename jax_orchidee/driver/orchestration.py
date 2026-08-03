"""Strict driver-to-process orchestration scaffolds for the paper case."""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, replace
from functools import lru_cache
from types import SimpleNamespace
from pathlib import Path
from typing import Mapping, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from jax_orchidee.coupled import (
    PaperCaseStomateBundleSourceKwargs,
    StomateOkLeakBoundaryInputs,
    StomateRestartInputBundles,
    stomate_pre_step_ok_leak_boundary_inputs,
    stomate_same_step_ok_leak_boundary_inputs,
    paper_case_stomate_boundary_input_kwargs,
    paper_case_stomate_static_input_kwargs,
    paper_case_stomate_bundle_source_kwargs,
    stomate_restart_input_bundles,
)

# Fortran provenance: fortran_source/ORCHIDEE/src_parameters/constantes_soil_var.f90
# lines 182-185, parameter peat_bulk_density(ndeep), median core measurement.
PAPER_CASE_PEAT_BULK_DENSITY = np.asarray(
    [
        0.063,
        0.063,
        0.063,
        0.047,
        0.064,
        0.066,
        0.079,
        0.091,
        0.101,
        0.109,
        0.109,
        0.103,
        0.112,
        0.110,
        0.104,
        0.073,
        0.092,
        0.092,
        0.092,
        0.092,
        0.092,
        0.092,
        0.092,
        0.092,
        0.092,
        0.092,
        0.092,
        0.092,
        0.092,
        0.092,
        0.092,
        0.092,
    ],
    dtype=np.float64,
)

# The paper run writes `PEAT_HYDRO=TRUE` in used_run.def, but the audited
# HYDROL coefficient trace for the active PFT14 soil tile records
# `peat_hydro=F`/`branch_peat=F` (traces/hydrol_1961_v9, hydrol.f90 branches
# at lines 5902 and 8259). Keep this as an explicit paper-case execution fact
# instead of using the run.def switch as the HYDROL soil-coefficient branch.
# The same HYDROL-local SAVE variable is used by coefficient, prognostic,
# water-stress, and dynamic-root branches. It is distinct from the same-named
# pft_parameters variable that does read run.def.
PAPER_1961_HYDROL_SOIL_PEAT_HYDRO = False
PAPER_MANGROVE_PFT_ID = "mangrove_pft14"

from jax_orchidee.driver.bundle import (
    DriverStepBundle,
    ccanopy_from_co2,
    load_paper_1961_first_step_bundle,
    load_paper_1961_step_bundle,
    paper_forcing_nb_spread,
    paper_forcing_split_from_config,
)
from jax_orchidee.driver.fast_state import (
    DriverFastStateBundle,
    fast_state_from_previous_fields,
    fast_state_from_previous_packet,
    previous_packet_from_fast_state,
)
from jax_orchidee.driver.domain import load_case_config, read_annual_co2, read_forcing_model_step_cached, read_forcing_step
from jax_orchidee.driver.dim2 import CP_AIR, GRAVITY
from jax_orchidee.driver.init import (
    RunScalars,
    initialize_imposed_vegetation_state,
    parse_run_def,
    parse_run_def_bool,
    parse_run_def_float,
    parse_run_def_indexed_selection,
    parse_run_def_indexed_vector,
    parse_run_def_int,
    read_run_scalars,
)
from jax_orchidee.driver.restart_state import (
    COLD_START_INITIAL_STATE_MODE,
    REFERENCE_CASE_INITIAL_STATE_MODE,
    ReferenceCaseFirstStepRestartState,
    reference_case_first_step_restart_state,
)
from jax_orchidee.driver.sechiba_boundary import (
    IntersurfFirstStepPayload,
    build_intersurf_first_step_payload,
    intersurf_payload_with_driver_wind_z0,
)
from jax_orchidee.driver.static import (
    read_paper_hydrol_refsoc_1d_static_field,
    read_paper_condveg_background_soilalbedo,
    read_paper_thermosoil_reftemp_static_field,
    read_paper_thermosoil_refsoc_static_field,
)
from jax_orchidee.driver.stomate_boundary import (
    PaperFirstStepStomateBoundary,
    default_used_run_def_path,
    paper_case_cwrr_vertical_soil_grid_from_used_run_def,
    paper_1961_first_step_stomate_boundary,
    paper_case_vertical_grids_from_used_run_def,
)
from jax_orchidee.driver.trace import StaticTraceFields
from jax_orchidee.sechiba.diffuco import (
    ControlInundateInputs,
    DiffucoControlInundationAssembly,
    DiffucoControlInundationInputCoverage,
    DiffucoControlSalinityAssembly,
    DiffucoFirstStepPrecallAssembly,
    DIFFUCO_FIRST_STEP_PRECALL_RESTART_FIELDS,
    DiffucoPFT14FirstStepLocalEnerbilPrecallAssembly,
    assemble_diffuco_first_step_precall_payload,
    assemble_pft14_control_inundation_first_step,
    assemble_pft14_control_salinity_first_step,
    diffuco_control_inundation_input_coverage,
    mangrove_control_inundation,
    pft14_trans_co2_parameter_inputs_from_run_def,
    run_pft14_local_enerbil_precall_from_first_step_precall,
)
from jax_orchidee.sechiba.restart_io import (
    SECHIBA_RESTART_COMPONENT_FIELDS,
    SECHIBA_RESTART_TO_SOURCE_NAMES,
    sechiba_finalize_source_state_from_restart,
)
from jax_orchidee.sechiba.restart_lifecycle import (
    transition_sechiba_finalize_source_state,
)
from jax_orchidee.sechiba.condveg import CondvegFirstStepModuleClosure, condveg_main_minimal, run_condveg_first_step_module
from jax_orchidee.sechiba.condveg import condveg_albedo_explicit, condveg_frac_snow
from jax_orchidee.sechiba.enerbil import (
    EnerbilFirstStepInputCoverage,
    EnerbilFirstStepLocalAssembly,
    EnerbilFirstStepPrecallAssembly,
    assemble_enerbil_first_step_precall_payload,
    driver_swnet_from_swdown_albedo,
    enerbil_cold_start_surface_state,
    enerbil_first_step_input_coverage,
    run_enerbil_first_step_local_from_precall,
)
from jax_orchidee.sechiba.first_step import SechibaFirstStepInputCoverage, first_step_sechiba_input_coverage
from jax_orchidee.sechiba.hydrol import (
    HydrolRuntimeStaticTables,
    HydrolFirstStepModuleClosure,
    HydrolFirstStepPrecallAssembly,
    MineralCWRRTables,
    assemble_hydrol_first_step_precall_payload,
    hydrol_cold_start_thermosoil_moisture_inputs,
    hydrol_cold_start_state,
    hydrol_cold_start_waterbal_begin_state,
    hydrol_layer_thickness_from_zz_mm,
    hydrol_runtime_static_tables_from_payload,
    hydrol_static_precall_template_from_slowproc,
    run_hydrol_first_step_module_from_precall,
)
from jax_orchidee.sechiba.slowproc import (
    slowproc_derivvar_explicit,
    slowproc_cold_start_vegetation_entry_state,
    slowproc_dyn_peat_disabled_entry_state,
    slowproc_erosion_daily_zero_entry_state,
    slowproc_fire_disabled_entry_state,
    slowproc_init_pft14_explicit,
    slowproc_no_lcc_entry_state,
    slowproc_static_entry_state,
    slowproc_surface_update_explicit,
    slowproc_thermosoil_entry_init_state,
)
from jax_orchidee.sechiba.thermosoil import (
    ThermosoilFirstStepModuleClosure,
    run_thermosoil_first_step_module,
    thermosoil_cold_start_coef_closure,
    thermosoil_initial_ptn_constant,
)
from jax_orchidee.stomate.daily import (
    DailyCarbonBoundary,
    StomateFirstStepDailyAccumulation,
    StomateDailyProcessFold,
    prepare_daily_carbon_inputs,
    reset_daily_on_slow,
    sum_resp_maint_radia,
    stomate_daily_process_fold_from_entries,
    stomate_first_step_daily_accumulation_from_entry,
    stomate_maintenance_respiration_parts_from_entries,
    stomate_maintenance_respiration_parts_from_stacks,
)
from jax_orchidee.stomate.carbon_kernels import (
    littercalc_aboveground_controls,
    stomate_littercalc_entry_prep,
    stomate_soil_mc_32l,
)
from jax_orchidee.stomate.entry import (
    StomateMainPayloadAssembly,
    assemble_stomate_main_payload,
    stomate_diffuco_entry_source,
    stomate_driver_entry_source,
    stomate_enerbil_entry_source,
    stomate_erosion_disabled_erodepth_entry_source,
    stomate_hydrol_entry_source,
    stomate_no_routing_entry_source,
    stomate_restart_entry_source,
    stomate_slowproc_derivvar_entry_source,
    stomate_slowproc_dyn_peat_disabled_entry_source,
    stomate_slowproc_erosion_daily_zero_entry_source,
    stomate_slowproc_fire_disabled_entry_source,
    stomate_slowproc_no_lcc_entry_source,
    stomate_slowproc_restart_entry_source,
    stomate_slowproc_static_entry_source,
    stomate_slowproc_thermosoil_entry_source,
    stomate_snow_entry_source,
    stomate_thermosoil_entry_source,
    stomate_thermosoil_static_entry_source,
)
from jax_orchidee.stomate.integration import (
    ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult,
    ExplicitOkLeakFromPostNppResult,
    ExplicitStomateLpjOutputsResult,
    stomate_ok_leak_explicit,
    stomate_ok_leak_from_post_npp_explicit,
    stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit,
    stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_outer_compiled,
    stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_static_jit,
    stomate_lpj_outputs_from_post_npp_ok_leak_explicit,
)
from jax_orchidee.stomate.permafrost import stomate_permafrost_decomposition_controls
from jax_orchidee.stomate.reference import (
    StomateRestartEntryState,
    StomateRestartSeasonState,
    stomate_cold_start_daily_accumulator_state,
    stomate_cold_start_entry_state,
    stomate_cold_start_season_state,
)
from jax_orchidee.stomate.restart_io import (
    STOMATE_FIXED_PATH_CARRY_FIELDS,
    stomate_fixed_path_carry_state,
)
from jax_orchidee.stomate.soilcarbon_kernels import soilcarbon_leak_tf_doc_inputs
from jax_orchidee.stomate.season import (
    SeasonMemoryState,
    season_surface_temperature_memory_step,
    season_update_gdd_init_date,
)


DRIVER_TIMESTEP_ORCHESTRATION_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 802-908",
    "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 1111-1125 and 1293-1305",
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 997-1224",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 973-1023",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 2409-2448 and 2899-3522",
)


def _hydrol_current_diffuco_payload(
    precall_payload: Mapping[str, object] | None,
    after_diffuco_payload: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return current DIFFUCO state while preserving pass-through fields.

    Fortran provenance: `sechiba.f90::sechiba_main` lines 997-1090 passes the
    same state through `diffuco_main`, `enerbil_main`, then `hydrol_main`.
    Fields updated by local DIFFUCO override the pre-call payload; unchanged
    HYDROL inputs such as snow, flood, and canopy storage remain pass-throughs.
    """

    payload = dict(precall_payload or {})
    payload.update(dict(after_diffuco_payload or {}))
    return payload


@dataclass(frozen=True)
class DriverRuntimeScalars:
    """Run-definition scalars needed before a strict timestep loop can run."""

    dt_sechiba: float
    dt_stomate: float
    stomate_ok_stomate: bool
    stomate_ok_dgvm: bool
    provenance: tuple[str, ...] = (
        "outputs/server_1961_trace_full_20260623/run/used_run.def lines 57-59, 201-207, and 4653-4655",
        "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 287-299 and 653-659",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 1844-1846 and 1912",
    )

    @property
    def dt_days(self) -> float:
        return self.dt_stomate / 86400.0


@dataclass(frozen=True)
class Paper1961PreparedDriverContext:
    """Reusable paper-case constants for strict driver-loop execution.

    This object is an orchestration cache only: it stores values read from the
    audited ``used_run.def`` and source-backed vertical-grid constructors so a
    day loop does not reparse static inputs on every half-hour step.
    """

    config_path: Path
    run_def_path: Path
    reference_run_dir: Path | None
    run_def_values: dict[str, str]
    domain_override: dict[str, float] | None
    runtime: DriverRuntimeScalars
    cwrr_grid: object
    diaglev: np.ndarray
    zlt: np.ndarray
    znt: np.ndarray
    run_scalars: RunScalars
    nflow: int
    nvm: int
    forcing_split: int
    forcing_nb_spread: int
    ok_laidev: tuple[bool, ...]
    ext_coeff: np.ndarray
    ext_coeff_vegetfrac: np.ndarray
    hydrol_humcste: np.ndarray
    hydrol_throughfall_by_pft: np.ndarray
    hydrol_cwrr_ks: np.ndarray
    hydrol_zz_mm: np.ndarray
    hydrol_dz_mm: np.ndarray
    hydrol_dh_mm: np.ndarray
    hydrol_reinf_slope: tuple[float, ...]
    dt_sechiba: float
    dt_sechiba_days: float
    min_wind: float
    ok_explicitsnow: bool
    rough_dyn: bool
    impaze: bool
    peat_hydro: bool
    hydrol_soil_peat_hydro: bool
    stomate_ok_dgvm: bool
    ok_freeze_cwrr: bool
    ok_thermodynamical_freezing: bool
    hydrol_fr_dt: float
    hydrol_froz_frac_corr: float
    hydrol_smtot_corr: float
    hydrol_max_froz_hydro: float
    hydrol_depth_max_h: float
    hydrol_do_ponds: bool
    hydrol_ok_pc: bool
    hydrol_ok_leak: bool
    hydrol_ok_ru2peat: bool
    hydrol_ok_wt_ab: bool
    hydrol_tides: bool
    hydrol_agri_peat: bool
    hydrol_max_wt_ab: float
    hydrol_dyn_nroot_larix: bool
    thermosoil_satsoil: bool
    river_routing: bool
    topmodel_new: bool
    erosion_module: bool
    fire_disable: bool
    dyn_peat: bool
    peat_occur: bool
    gluc_use_age_class: bool
    pft_to_mtc: np.ndarray
    vcmax_fix: np.ndarray
    slowproc_height: np.ndarray
    sechiba_qsint: float
    control_salinity_min: float
    control_inudate_min: float
    depth_max_t: float
    depth_topthickness: float
    depth_cstthickness: float
    depth_geom: float
    ratio_geom_below: float
    agb_agr_ven_all_st: float
    agb_agr_ven_all_pn: float
    h_agr_max_st: float
    h_agr_max_pn: float
    diffuco_inundation_static: DiffucoControlInundationAssembly
    diffuco_pft14_static_kwargs: dict[str, object]
    stomate_static: object | None = None
    stomate_boundary: object | None = None
    first_step_bundle: object | None = None
    first_step_restart_state: object | None = None
    first_step_stomate_boundary: object | None = None
    provenance: tuple[str, ...] = (
        "outputs/server_1961_trace_full_20260623/run/used_run.def parsed once for driver-loop constants",
        "fortran_source/ORCHIDEE/src_parameters/vertical_soil.f90::vertical_soil_init lines 175-454",
        "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 653-659 and 1111-1125",
    )


def _domain_override_from_run_def_values(values: Mapping[str, str]) -> dict[str, float] | None:
    """Return driver domain limits from run.def when all active keys exist.

    Fortran provenance: `dim2_driver.f90` lines 147-175 reads `LIMIT_WEST`,
    `LIMIT_EAST`, `LIMIT_NORTH`, and `LIMIT_SOUTH`; `readdim2.f90::domain_size`
    lines 2263-2341 uses those limits to select the forcing zoom.
    """

    key_map = {
        "LIMIT_WEST": "west",
        "LIMIT_EAST": "east",
        "LIMIT_SOUTH": "south",
        "LIMIT_NORTH": "north",
    }
    if not all(key in values for key in key_map):
        return None
    return {target: parse_run_def_float(values, source) for source, target in key_map.items()}


def _effective_thermosoil_wetdiaglong(values: Mapping[str, str]) -> bool:
    """Return Fortran's initialized long-term-humidity switch.

    ``thermosoil_initialize`` first reads ``OK_WETDIAGLONG`` and then promotes
    the switch when either the freezing/peat-carbon path or ``OK_LEAK`` is
    active. The production driver must use this initialized state rather than
    forwarding the raw run.def value.

    Fortran provenance: ``src_sechiba/thermosoil.f90::thermosoil_initialize``
    lines 318-329.
    """

    configured = parse_run_def_bool(values.get("OK_WETDIAGLONG", "FALSE"))
    freeze_default = parse_run_def_bool(values.get("OK_FREEZE", "FALSE"))
    freeze_thermix = parse_run_def_bool(
        values.get("OK_FREEZE_THERMIX", "TRUE" if freeze_default else "FALSE")
    )
    ok_pc = parse_run_def_bool(values.get("OK_PC", "FALSE"))
    ok_leak = parse_run_def_bool(values.get("OK_LEAK", "FALSE"))
    return bool(configured or (freeze_thermix and ok_pc) or ok_leak)


def _validate_stomate_main_supported_switches(values: Mapping[str, str]) -> None:
    """Reject full-driver STOMATE switch combinations rejected by Fortran."""

    ok_stomate = parse_run_def_bool(values.get("STOMATE_OK_STOMATE", "TRUE"))
    ok_leak = parse_run_def_bool(values.get("OK_LEAK", "FALSE"))
    if ok_stomate and not ok_leak:
        raise NotImplementedError(
            "Fortran stomate_main rejects full-driver STOMATE with OK_LEAK=n "
            "(src_stomate/stomate.f90 lines 3093-3127). OK_PC/deep_carbcycle "
            "kernels remain source-local micro-cases, but the production "
            "driver path must not continue past this source fatal guard."
        )


def _paper_mangrove_pft_index(run_scalars: RunScalars) -> int:
    """Resolve the paper mangrove execution slot from stable PFT identity."""

    return run_scalars.pft_layout.index_for_id(PAPER_MANGROVE_PFT_ID)


def _paper_mangrove_fortran_pft_id(run_scalars: RunScalars) -> int:
    """Resolve the source parameter row for the paper mangrove capability."""

    index = _paper_mangrove_pft_index(run_scalars)
    return int(run_scalars.pft_layout.entries[index].fortran_pft_id)


def _runtime_fortran_pft_ids(run_scalars: RunScalars, *, nvm: int | None = None) -> tuple[int, ...]:
    """Return source parameter rows after validating the dense execution axis."""

    fortran_pft_ids = tuple(int(value) for value in run_scalars.fortran_pft_ids)
    expected = int(run_scalars.nvm)
    if len(fortran_pft_ids) != expected:
        raise ValueError("run-scalars PFT identities disagree with NVM")
    if nvm is not None and int(nvm) != expected:
        raise ValueError("model-state PFT axis disagrees with the selected stable PFT layout")
    return fortran_pft_ids


def _select_run_def_pft_vector(
    values: dict[str, str],
    name: str,
    run_scalars: RunScalars,
    *,
    nvm: int | None = None,
    dtype=np.float64,
) -> np.ndarray:
    """Read a PFT-indexed run.def vector in stable execution-layout order."""

    return parse_run_def_indexed_selection(
        values,
        name,
        _runtime_fortran_pft_ids(run_scalars, nvm=nvm),
        dtype=dtype,
    )


def _validated_prepared_pft_axis(
    values,
    *,
    context: Paper1961PreparedDriverContext,
    nvm: int,
    name: str,
):
    """Reject positional truncation when prepared arrays and state layouts differ."""

    _runtime_fortran_pft_ids(context.run_scalars, nvm=nvm)
    if len(values) != int(nvm):
        raise ValueError(f"prepared {name} PFT axis disagrees with the selected stable PFT layout")
    return values


@lru_cache(maxsize=8)
def _prepare_paper_1961_driver_context_cached(
    config_path: str,
    run_def_path: str,
    reference_run_dir: str | None,
) -> Paper1961PreparedDriverContext:
    values = parse_run_def(run_def_path)
    _validate_stomate_main_supported_switches(values)
    domain_override = _domain_override_from_run_def_values(values)
    runtime = DriverRuntimeScalars(
        dt_sechiba=parse_run_def_float(values, "DT_SECHIBA"),
        dt_stomate=parse_run_def_float(values, "DT_STOMATE"),
        stomate_ok_stomate=parse_run_def_bool(values["STOMATE_OK_STOMATE"]),
        stomate_ok_dgvm=parse_run_def_bool(values["STOMATE_OK_DGVM"]),
    )
    run_scalars = read_run_scalars(config_path, run_def_path=run_def_path)
    nvm = int(run_scalars.nvm)
    if parse_run_def_int(values, "NVM") != nvm:
        raise ValueError("runtime run.def NVM disagrees with the selected stable PFT layout")
    fortran_pft_ids = tuple(int(value) for value in run_scalars.fortran_pft_ids)
    mangrove_pft_index = _paper_mangrove_pft_index(run_scalars)
    mangrove_fortran_pft_id = _paper_mangrove_fortran_pft_id(run_scalars)
    cwrr_grid = paper_case_cwrr_vertical_soil_grid_from_used_run_def(run_def_path)
    diaglev, zlt, znt = paper_case_vertical_grids_from_used_run_def(run_def_path)
    dt_sechiba = runtime.dt_sechiba
    ext_coeff = parse_run_def_indexed_selection(values, "EXT_COEFF", fortran_pft_ids)
    ext_coeff_vegetfrac = parse_run_def_indexed_selection(
        values, "EXT_COEFF_VEGETFRAC", fortran_pft_ids
    )
    hydrol_humcste = parse_run_def_indexed_selection(
        values, "HYDROL_HUMCSTE", fortran_pft_ids
    )
    pft_to_mtc = parse_run_def_indexed_selection(
        values, "PFT_TO_MTC", fortran_pft_ids, dtype=int
    )
    if not np.array_equal(pft_to_mtc, run_scalars.pft_to_mtc):
        raise ValueError("runtime PFT_TO_MTC disagrees with the selected stable PFT layout")
    vcmax_fix = parse_run_def_indexed_selection(values, "VCMAX_FIX", fortran_pft_ids)
    slowproc_height = parse_run_def_indexed_selection(
        values, "SLOWPROC_HEIGHT", fortran_pft_ids
    )
    first_step_bundle = load_paper_1961_first_step_bundle(
        config_path,
        run_def_path=run_def_path,
        reference_run_dir=reference_run_dir,
        domain_override=domain_override,
    )
    forcing_config = load_case_config(config_path)
    forcing_split = paper_forcing_split_from_config(forcing_config, first_step_bundle.run_scalars)
    first_step_restart_state = reference_case_first_step_restart_state(
        config_path,
        root=Path(config_path).resolve().parents[1],
        run_dir=reference_run_dir,
        run_def_path=run_def_path,
    )
    first_step_stomate_boundary = paper_1961_first_step_stomate_boundary(
        config_path,
        bundle=first_step_bundle,
        used_run_def_path=Path(run_def_path),
        root=Path(config_path).resolve().parents[1],
        run_dir=reference_run_dir,
    )
    stomate_static = paper_case_stomate_static_input_kwargs(config_path, used_run_def=Path(run_def_path))
    stomate_boundary = paper_case_stomate_boundary_input_kwargs(
        config_path,
        parameters=stomate_static.parameters,
        npts=int(np.asarray(first_step_bundle.domain.lalo).shape[0]),
        used_run_def=Path(run_def_path),
    )
    return Paper1961PreparedDriverContext(
        config_path=Path(config_path),
        run_def_path=Path(run_def_path),
        reference_run_dir=None if reference_run_dir is None else Path(reference_run_dir),
        run_def_values=values,
        domain_override=domain_override,
        runtime=runtime,
        cwrr_grid=cwrr_grid,
        diaglev=diaglev,
        zlt=zlt,
        znt=znt,
        run_scalars=first_step_bundle.run_scalars,
        nflow=parse_run_def_int(values, "NSTM"),
        nvm=nvm,
        forcing_split=forcing_split,
        forcing_nb_spread=paper_forcing_nb_spread(forcing_split),
        ok_laidev=tuple(
            parse_run_def_bool(values[f"OK_LAIDEV__{fortran_id:05d}"])
            for fortran_id in fortran_pft_ids
        ),
        ext_coeff=ext_coeff,
        ext_coeff_vegetfrac=ext_coeff_vegetfrac,
        hydrol_humcste=hydrol_humcste,
        hydrol_throughfall_by_pft=parse_run_def_indexed_selection(
            values, "PERCENT_THROUGHFALL_PFT", fortran_pft_ids
        ),
        hydrol_cwrr_ks=parse_run_def_indexed_vector(values, "CWRR_KS", 12),
        hydrol_zz_mm=cwrr_grid.znh * 1000.0,
        hydrol_dz_mm=cwrr_grid.dnh * 1000.0,
        hydrol_dh_mm=cwrr_grid.dlh * 1000.0,
        # slowproc.f90:2021-2022 supplies 0.0 when SLOPE is absent.
        hydrol_reinf_slope=(float(values.get("SLOPE", "0.0")),),
        dt_sechiba=dt_sechiba,
        dt_sechiba_days=dt_sechiba / 86400.0,
        min_wind=parse_run_def_float(values, "MIN_WIND"),
        ok_explicitsnow=parse_run_def_bool(values["OK_EXPLICITSNOW"]),
        rough_dyn=parse_run_def_bool(values["ROUGH_DYN"]),
        impaze=parse_run_def_bool(values["IMPOSE_AZE"]),
        peat_hydro=parse_run_def_bool(values["PEAT_HYDRO"]),
        hydrol_soil_peat_hydro=PAPER_1961_HYDROL_SOIL_PEAT_HYDRO,
        stomate_ok_dgvm=parse_run_def_bool(values["STOMATE_OK_DGVM"]),
        ok_freeze_cwrr=parse_run_def_bool(values["OK_FREEZE_CWRR"]),
        ok_thermodynamical_freezing=parse_run_def_bool(values["OK_THERMODYNAMICAL_FREEZING"]),
        hydrol_fr_dt=parse_run_def_float(values, "FR_DT"),
        hydrol_froz_frac_corr=parse_run_def_float(values, "FROZ_FRAC_CORR"),
        hydrol_smtot_corr=parse_run_def_float(values, "SMTOT_CORR"),
        hydrol_max_froz_hydro=parse_run_def_float(values, "MAX_FROZ_HYDRO"),
        hydrol_depth_max_h=parse_run_def_float(values, "DEPTH_MAX_H"),
        hydrol_do_ponds=parse_run_def_bool(values["DO_PONDS"]),
        hydrol_ok_pc=parse_run_def_bool(values["OK_PC"]),
        hydrol_ok_leak=parse_run_def_bool(values["OK_LEAK"]),
        hydrol_ok_ru2peat=parse_run_def_bool(values["OK_RU2PEAT"]),
        hydrol_ok_wt_ab=parse_run_def_bool(values["OK_WT_AB"]),
        hydrol_tides=parse_run_def_bool(values["TIDES"]),
        hydrol_agri_peat=parse_run_def_bool(values["AGRI_PEAT"]),
        hydrol_max_wt_ab=parse_run_def_float(values, "max_wt_ab") if "max_wt_ab" in values else 100.0,
        hydrol_dyn_nroot_larix=parse_run_def_bool(values["dyn_nroot_larix"]),
        thermosoil_satsoil=parse_run_def_bool(values["satsoil"]),
        river_routing=parse_run_def_bool(values["RIVER_ROUTING"]),
        topmodel_new=parse_run_def_bool(values["TOPMODEL_NEW"]),
        erosion_module=parse_run_def_bool(values["EROSION_MODULE"]),
        fire_disable=parse_run_def_bool(values["FIRE_DISABLE"]),
        dyn_peat=parse_run_def_bool(values["DYN_PEAT"]),
        peat_occur=parse_run_def_bool(values["PEAT_OCCUR"]),
        gluc_use_age_class=parse_run_def_bool(values["GLUC_USE_AGE_CLASS"]),
        pft_to_mtc=pft_to_mtc,
        vcmax_fix=vcmax_fix,
        slowproc_height=slowproc_height,
        sechiba_qsint=parse_run_def_float(values, "SECHIBA_QSINT"),
        control_salinity_min=parse_run_def_float(values, "CONTROL_SALINITY_MIN"),
        control_inudate_min=parse_run_def_float(values, "CONTROL_INUDATE_MIN"),
        depth_max_t=parse_run_def_float(values, "DEPTH_MAX_T"),
        depth_topthickness=parse_run_def_float(values, "DEPTH_TOPTHICK"),
        depth_cstthickness=parse_run_def_float(values, "DEPTH_CSTTHICK"),
        depth_geom=parse_run_def_float(values, "DEPTH_GEOM"),
        ratio_geom_below=parse_run_def_float(values, "RATIO_GEOM_BELOW"),
        agb_agr_ven_all_st=parse_run_def_float(values, "AGB_AGR_VEN_ALL_ST"),
        agb_agr_ven_all_pn=parse_run_def_float(values, "AGB_AGR_VEN_ALL_PN"),
        h_agr_max_st=parse_run_def_float(values, "H_AGR_MAX_ST"),
        h_agr_max_pn=parse_run_def_float(values, "H_AGR_MAX_PN"),
        diffuco_inundation_static=assemble_pft14_control_inundation_first_step(
            depth_max_h=parse_run_def_float(values, "DEPTH_MAX_H"),
            depth_max_t=parse_run_def_float(values, "DEPTH_MAX_T"),
            depth_topthickness=parse_run_def_float(values, "DEPTH_TOPTHICK"),
            depth_cstthickness=parse_run_def_float(values, "DEPTH_CSTTHICK"),
            depth_geom=parse_run_def_float(values, "DEPTH_GEOM"),
            ratio_geom_below=parse_run_def_float(values, "RATIO_GEOM_BELOW"),
            pft_to_mtc=pft_to_mtc,
            hydrol_humcste=hydrol_humcste,
            pft_index=mangrove_pft_index,
        ),
        diffuco_pft14_static_kwargs={
            "rstruct_const": parse_run_def_indexed_selection(
                values, "RSTRUCT_CONST", fortran_pft_ids
            ),
            "trans_co2_pft_params": pft14_trans_co2_parameter_inputs_from_run_def(
                values,
                pft_fortran_index=mangrove_fortran_pft_id,
            ),
            "dt_sechiba": dt_sechiba,
            "laimax": parse_run_def_float(values, "LAIMAX"),
            "lai_level_depth": parse_run_def_float(values, "LAI_LEVEL_DEPTH"),
            "min_wind": parse_run_def_float(values, "MIN_WIND"),
            "snowcri": parse_run_def_float(values, "SNOWCRI"),
            "ok_snowfact": parse_run_def_bool(values["OK_SNOWFACT"]),
            "rough_dyn": parse_run_def_bool(values["ROUGH_DYN"]),
            "ok_laidev": tuple(
                parse_run_def_bool(values[f"OK_LAIDEV__{fortran_id:05d}"])
                for fortran_id in fortran_pft_ids
            ),
        },
        stomate_static=stomate_static,
        stomate_boundary=stomate_boundary,
        first_step_bundle=first_step_bundle,
        first_step_restart_state=first_step_restart_state,
        first_step_stomate_boundary=first_step_stomate_boundary,
    )


def prepare_paper_1961_driver_context(
    config_path: str | Path,
    *,
    used_run_def_path: str | Path | None = None,
    reference_run_dir: str | Path | None = None,
) -> Paper1961PreparedDriverContext:
    """Prepare reusable constants for a source-backed paper-case day loop."""

    config_resolved = Path(config_path).resolve()
    run_def_path = Path(used_run_def_path) if used_run_def_path is not None else default_used_run_def_path(config_resolved)
    run_dir_key = None if reference_run_dir is None else str(Path(reference_run_dir).resolve())
    return _prepare_paper_1961_driver_context_cached(str(config_resolved), str(run_def_path.resolve()), run_dir_key)


def _paper_1961_step_bundle_from_context(
    context: Paper1961PreparedDriverContext,
    *,
    year: int,
    tstep: int,
) -> DriverStepBundle:
    """Build a step bundle by reusing prepared static/domain fields."""

    first = context.first_step_bundle
    if first is None:
        return load_paper_1961_step_bundle(
            context.config_path,
            year=year,
            tstep=tstep,
            run_def_path=context.run_def_path,
            reference_run_dir=context.reference_run_dir,
            domain_override=context.domain_override,
        )
    forcing = read_forcing_model_step_cached(
        context.config_path,
        domain=first.domain,
        year=year,
        model_tstep=tstep,
        split=context.forcing_split,
        nb_spread=context.forcing_nb_spread,
    )
    co2_ppm = read_annual_co2(context.config_path, year)
    return DriverStepBundle(
        year=year,
        tstep=int(tstep),
        domain=first.domain,
        forcing=forcing,
        run_scalars=first.run_scalars,
        vegetation=first.vegetation,
        co2_ppm=co2_ppm,
        ccanopy=ccanopy_from_co2(co2_ppm, first.domain.nbindex),
        water_table=first.water_table,
        restart_anchors=first.restart_anchors,
        static_trace_fields=first.static_trace_fields,
        missing_geometry_fields=first.missing_geometry_fields,
        missing_static_fields=first.missing_static_fields,
    )



@dataclass(frozen=True)
class ColdStartFirstStepStateCoverage:
    """Source-backed cold-start initialization coverage before first SECHIBA step."""

    mode: str
    covered_by_component: dict[str, tuple[str, ...]]
    missing_by_component: dict[str, tuple[str, ...]]
    initialized_payloads: dict[str, object]
    provenance: tuple[str, ...] = (
        "configs/orchidee_man_250919.yaml minimal_single_case.cold_start and restart settings",
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_initialize lines 602-617",
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_initialize lines 257-293",
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_initialize lines 1881-1926",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_initialize lines 1309-1368",
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 919-922",
        "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_initialize lines 668-684 and 725-735",
    )
    notes: tuple[str, ...] = (
        "This is a coverage contract for RESTART_FILEIN=NONE/SECHIBA_restart_in=NONE/STOMATE_RESTART_FILEIN=NONE.",
        "It exposes exact source-backed initialization fields and keeps non-restart thermal coefficients as gaps until thermosoil_coef is ported or traced.",
    )

    @property
    def ready_for_cold_start_first_step(self) -> bool:
        return not any(self.missing_by_component.values())

    @property
    def missing_components(self) -> tuple[str, ...]:
        return tuple(
            f"{component}:{field}"
            for component, fields in self.missing_by_component.items()
            for field in fields
        )


def _thermosoil_restart_namespace_from_cold_start(
    *, ptn, coef, refsoc, temp_sol_beg, shum_ngrnd_permalong, e_soil_lat=None
):
    """Expose no-restart THERMOSOIL recurrence fields by restart-state names."""

    if coef is None or not coef.ok:
        return None
    ptn_array = np.asarray(ptn)
    return SimpleNamespace(
        ptn=ptn,
        cgrnd=coef.coef.soil.cgrnd,
        dgrnd=coef.coef.soil.dgrnd,
        cgrnd_snow=coef.coef.cgrnd_snow,
        dgrnd_snow=coef.coef.dgrnd_snow,
        lambda_snow=coef.coef.lambda_snow,
        gtemp=np.zeros((ptn_array.shape[0],), dtype=np.float64),
        temp_sol_beg=np.asarray(temp_sol_beg, dtype=np.float64),
        pcapa_en=coef.getdiff.pcapa_en,
        refsoc=refsoc,
        shum_ngrnd_permalong=np.asarray(shum_ngrnd_permalong, dtype=np.float64),
        e_soil_lat=None if e_soil_lat is None else np.asarray(e_soil_lat, dtype=np.float64),
        veget_mask_2d=np.ones((ptn_array.shape[0], ptn_array.shape[2]), dtype=bool),
    )


def _previous_step_state_from_cold_start_payloads(payloads: dict[str, object]) -> DriverPreviousStepStatePacket:
    """Extract source-backed next-step state from closed cold-start payloads."""

    fields = _empty_previous_step_state_fields()
    _add_enerbil_local_to_previous_fields(fields, payloads.get("enerbil_first_step_local"))
    _add_diffuco_local_to_previous_fields(fields, payloads.get("diffuco_first_step_local_enerbil_precall"))
    _add_diffuco_saved_controls_to_previous_fields(
        fields,
        control_salinity=payloads.get("diffuco_control_salinity"),
        control_inundation=payloads.get("diffuco_control_inundation"),
        precall=payloads.get("diffuco_first_step_precall"),
    )
    _add_thermosoil_to_previous_fields(fields, payloads.get("thermosoil_first_step_module"))
    if payloads.get("thermosoil_e_soil_lat") is not None:
        fields["thermosoil_previous_step_state"]["e_soil_lat"] = payloads["thermosoil_e_soil_lat"]
    _add_condveg_to_previous_fields(fields, payloads.get("condveg_first_step_module"))
    _add_hydrol_to_previous_fields(fields, payloads.get("hydrol_first_step_module"))

    slowproc = payloads.get("slowproc_cold_start_vegetation")
    derivvar = payloads.get("slowproc_derivvar")
    stomate = payloads.get("stomate_cold_start_entry_state")
    static = payloads.get("slowproc_static")
    if slowproc is not None:
        for name, value in (
            ("veget", slowproc.vegetation.veget),
            ("veget_max", slowproc.vegetation.veget_max),
            ("lai", slowproc.lai),
            ("frac_nobio", slowproc.vegetation.frac_nobio),
            ("totfrac_nobio", slowproc.vegetation.totfrac_nobio),
            ("height", slowproc.height),
            ("frac_age", slowproc.frac_age),
            ("soiltile", slowproc.vegetation.soiltile),
            ("tot_bare_soil", slowproc.tot_bare_soil),
        ):
            fields["slowproc_stomate_previous_step_state"][name] = value
            if name not in {"frac_age", "soiltile"}:
                fields["diffuco_previous_step_state"][name] = value
    if static is not None:
        fields["slowproc_stomate_previous_step_state"]["fc_grazing"] = static.fc_grazing
        fields["slowproc_stomate_previous_step_state"]["humcste_use"] = static.humcste_use
    if derivvar is not None:
        for name in ("assim_param", "deadleaf_cover", "temp_growth"):
            fields["slowproc_stomate_previous_step_state"][name] = getattr(derivvar, name)
    if stomate is not None:
        for name in (
            "biomass",
            "resp_maint_part",
            "ind",
            "adapted",
            "regenerate",
            "when_growthinit",
            "everywhere",
            "PFTpresent",
            "veget_lastlight",
            "need_adjacent",
            "senescence",
            "begin_leaves",
            "leaf_frac",
            "leaf_age",
            "npp_longterm",
            "lm_lastyearmax",
            "lm_thisyearmax",
            "age",
            "sla_calc",
            "turnover_time",
            "RIP_time",
            "time_lowgpp",
            "time_hum_min",
            "gdd_midwinter",
            "ncd_dormance",
            "ngd_minus5",
            "litter_above",
            "litter_below",
            "litterpart",
            "dead_leaves",
            "carbon",
            "litter",
            "lignin_struc",
            "prod10",
            "prod100",
            "flux10",
            "flux100",
            "fuel_1hr",
            "fuel_10hr",
            "fuel_100hr",
            "fuel_1000hr",
            "carbon_32l",
            "DOC",
            "lignin_struc_above",
            "lignin_struc_below",
            "deepC_a",
            "deepC_s",
            "deepC_p",
            "soilc_total",
            "thawed_humidity",
            "depth_organic_soil",
            "altmax",
            "fpeat",
        ):
            if hasattr(stomate, name):
                fields["slowproc_stomate_previous_step_state"][name] = getattr(stomate, name)
    daily = payloads.get("stomate_cold_start_daily_accumulators")
    if daily is not None:
        fields["slowproc_stomate_previous_step_state"]["daily_accumulators"] = daily._asdict()
    season = payloads.get("stomate_cold_start_season_state")
    if season is not None:
        for name, value in season._asdict().items():
            fields["slowproc_stomate_previous_step_state"][name] = value
    cold_entry = payloads.get("stomate_cold_start_entry_state")
    forcing_t2m = payloads.get("forcing_t2m_for_stomate")
    if cold_entry is not None and forcing_t2m is not None:
        npts, nvm = np.asarray(cold_entry.gpp_daily).shape
        ndeep = np.asarray(cold_entry.carbon_32l).shape[3]
        t2m = np.asarray(forcing_t2m)
        fields["slowproc_stomate_previous_step_state"].update(
            {
                # stomate_io.f90::readstart lines 1387-1391, 1474-1478,
                # 1535-1538, and 1629-1633 cold-start fallbacks.
                "tsurf_year": t2m.copy(),
                "t2m_14": t2m.copy(),
                "deepC_peat": np.zeros((npts, ndeep, nvm), dtype=np.float64),
                # sechiba.f90::sechiba_init lines 2432-2437 initializes the
                # slowproc/stomate_initialize INTENT(out) object to zero.
                "depth_deepsoil": np.zeros((npts, nvm), dtype=np.float64),
                "resp_hetero": np.zeros((npts, nvm), dtype=np.float64),
                "dt_days_read": float(season.dt_days_read),
            }
        )
    if "soilalbedo_bg" not in fields["diffuco_previous_step_state"] and payloads.get("condveg_soilalbedo_bg") is not None:
        fields["diffuco_previous_step_state"]["soilalbedo_bg"] = payloads["condveg_soilalbedo_bg"]
    condveg_albedo = payloads.get("condveg_albedo")
    if condveg_albedo is not None and getattr(condveg_albedo, "albedo", None) is not None:
        fields.setdefault("driver_previous_step_state", {})["albedo"] = condveg_albedo.albedo
    if payloads.get("thermosoil_refSOC") is not None:
        fields["thermosoil_previous_step_state"]["refSOC"] = payloads["thermosoil_refSOC"]
    if payloads.get("hydrol_refSOC_1d") is not None:
        fields["hydrol_previous_step_state"]["refSOC_1d"] = payloads["hydrol_refSOC_1d"]
    hydrol_cold = payloads.get("hydrol_cold_start_state")
    if hydrol_cold is not None:
        for name in (
            "free_drain_coef",
            "zwt_force",
            "fwet_out",
            "fwet_new",
            "liqwt_ratio",
            "wtp",
            "run2peat",
            "run2man",
            "wt_ab",
            "wt_ab_tide",
        ):
            if name not in fields["hydrol_previous_step_state"]:
                fields["hydrol_previous_step_state"][name] = getattr(hydrol_cold, name)

    # Seed the exact 93-field SECHIBA finalize lifetime before applying the
    # first-step component updates. Values marked None here are never carried:
    # _sechiba_finalize_state_from_components must overwrite them.
    expected_finalize = {
        SECHIBA_RESTART_TO_SOURCE_NAMES.get(name, name)
        for component_fields in SECHIBA_RESTART_COMPONENT_FIELDS.values()
        for name in component_fields
    }
    carry: dict[str, object] = {}
    enerbil_surface = payloads.get("enerbil_surface_state")
    slowproc_full = payloads.get("slowproc_init_pft14")
    soilalbedo_bg = payloads.get("condveg_soilalbedo_bg")
    refsoc = payloads.get("thermosoil_refSOC")
    if enerbil_surface is not None:
        carry.update(
            q_sol_pot=enerbil_surface.q_sol_pot,
            temp_sol_pot=enerbil_surface.temp_sol_pot,
        )
    if hydrol_cold is not None:
        carry.update(
            free_drain_coef=hydrol_cold.free_drain_coef,
            zwt_force=hydrol_cold.zwt_force,
            fwet_new=hydrol_cold.fwet_new,
            resdist=hydrol_cold.resdist,
            vegtot_old=hydrol_cold.vegtot_old,
        )
    if soilalbedo_bg is not None:
        carry["soilalb_bg"] = soilalbedo_bg
    if refsoc is not None:
        carry["refsoc"] = refsoc
    diffuco_local = payloads.get("diffuco_first_step_local_enerbil_precall")
    if diffuco_local is not None:
        active_leaf_ci = diffuco_local.result.boundary.process_chain.trans_co2.canopy.leaf_ci
        npts, nlai = np.asarray(active_leaf_ci).shape
        if slowproc_full is None:
            raise ValueError("DIFFUCO cold-start state requires explicit slowproc PFT axes")
        nvm = np.asarray(slowproc_full.veget).shape[1]
        # diffuco_initialize lines 212-225 applies DIFFUCO_LEAFCI=233 when
        # the cold-start restart field is absent; trans_co2 then overwrites
        # the active PFT14 slice during the first production step.
        carry["leaf_ci"] = np.full((npts, nvm, nlai), 233.0, dtype=np.float64)
    if slowproc_full is not None:
        peat = slowproc_full.peat
        carry.update(
            veget=slowproc_full.veget,
            veget_max=slowproc_full.veget_max,
            lai=slowproc_full.lai,
            frac_nobio=slowproc_full.frac_nobio,
            frac_age=slowproc_full.frac_age,
            njsc=slowproc_full.njsc,
            reinf_slope=slowproc_full.reinf_slope,
            clayfraction=slowproc_full.clayfraction,
            sandfraction=slowproc_full.sandfraction,
            height=slowproc_full.height,
            # With IMPOSE_VEG=y, slowproc_init never defines veget_year and
            # the same branch never reads it. Use a stable serialization-only
            # sentinel; this field is outside the defined scientific contract.
            veget_year=np.zeros(
                (np.asarray(slowproc_full.veget).shape[0],), dtype=np.int64
            ),
            peatPET_lastyear=peat.peatPET_lastyear,
            growth_day=peat.growth_day,
            GSL=peat.GSL,
            peatPET_thisyear=peat.peatPET_thisyear,
            precipitation_lastsummer=peat.precipitation_lastsummer,
            precipitation_thissummer=peat.precipitation_thissummer,
            summerpet_long=peat.summerpet_long,
            summerp_long=peat.summerp_long,
            peatC=peat.peatC,
            peatC_ok=peat.peatC_ok,
            soil_ph=slowproc_full.soil_ph,
            poor_soils=slowproc_full.poor_soils,
            bulk_density=slowproc_full.bulk_density,
        )
        carry["e_soil_lat"] = np.zeros_like(np.asarray(slowproc_full.veget), dtype=np.float64)
    missing_carry = _SECHIBA_HALF_HOUR_CARRY_FIELDS - set(carry)
    if missing_carry:
        raise ValueError(f"cold-start SECHIBA finalize carry fields missing: {sorted(missing_carry)}")
    seed = {name: None for name in expected_finalize}
    seed.update(carry)
    fields["sechiba_finalize_state"] = _sechiba_finalize_state_from_components(
        previous=seed,
        enerbil_local=payloads["enerbil_first_step_local"],
        diffuco_local=payloads["diffuco_first_step_local_enerbil_precall"],
        thermosoil_module=payloads["thermosoil_first_step_module"],
        condveg_module=payloads["condveg_first_step_module"],
        hydrol_module=payloads["hydrol_first_step_module"],
        slowproc_updates=carry,
    )
    unset_finalize = tuple(
        sorted(name for name, value in fields["sechiba_finalize_state"].items() if value is None)
    )
    if len(fields["sechiba_finalize_state"]) != 93 or unset_finalize:
        raise ValueError(
            "cold-start SECHIBA finalize state did not close all 93 fields: "
            f"unset={unset_finalize}"
        )
    return _packet_from_previous_step_fields(tstep=0, fields=fields)


def _previous_step_state_from_cold_start_coverage(coverage: ColdStartFirstStepStateCoverage) -> DriverPreviousStepStatePacket | None:
    """Extract source-backed next-step state from a closed cold-start first step."""

    return _previous_step_state_from_cold_start_payloads(coverage.initialized_payloads)


@dataclass(frozen=True)
class DriverTimestepScaffold:
    """Source-backed boundary objects for one timestep, plus explicit gaps."""

    step: DriverStepBundle
    payload: IntersurfFirstStepPayload
    runtime: DriverRuntimeScalars
    stomate_entry_assembly: StomateMainPayloadAssembly
    diffuco_control_coverage: DiffucoControlInundationInputCoverage | None
    diffuco_control_salinity: DiffucoControlSalinityAssembly | None
    diffuco_control_assembly: DiffucoControlInundationAssembly | None
    diffuco_first_step_precall: DiffucoFirstStepPrecallAssembly | None
    diffuco_first_step_local_enerbil_precall: DiffucoPFT14FirstStepLocalEnerbilPrecallAssembly | None
    enerbil_first_step_coverage: EnerbilFirstStepInputCoverage | None
    enerbil_first_step_precall: EnerbilFirstStepPrecallAssembly | None
    enerbil_first_step_local: EnerbilFirstStepLocalAssembly | None
    hydrol_first_step_precall: HydrolFirstStepPrecallAssembly | None
    hydrol_first_step_module: HydrolFirstStepModuleClosure | None
    condveg_first_step_module: CondvegFirstStepModuleClosure | None
    thermosoil_first_step_module: ThermosoilFirstStepModuleClosure | None
    sechiba_first_step_coverage: SechibaFirstStepInputCoverage | None
    stomate_first_step_daily_accumulation: StomateFirstStepDailyAccumulation | None
    stomate_bundle_source: PaperCaseStomateBundleSourceKwargs | None
    stomate_restart_input_bundles: StomateRestartInputBundles | None
    first_step_restart_state: ReferenceCaseFirstStepRestartState | None
    stomate_pre_step: PaperFirstStepStomateBoundary | None
    initial_state_mode: str
    covered_components: tuple[str, ...]
    missing_components: tuple[str, ...]
    notes: tuple[str, ...]
    provenance: tuple[str, ...] = DRIVER_TIMESTEP_ORCHESTRATION_PROVENANCE

    @property
    def ready_to_run_processes(self) -> bool:
        return not self.missing_components


STOMATE_MAIN_CHAIN_BUNDLE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90 lines 880-951 initializes cn_ind and calls prescribe",
    "fortran_source/ORCHIDEE/src_stomate/stomate_prescribe.f90::prescribe lines 126-248 updates cn_ind/ind for static vegetation",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90 lines 656-662 updates do_slow scheduling",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90 lines 595-607 sets EndOfYear on the last SECHIBA timestep of the paper-case noleap year",
    "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 501-505 and 613-637 reads dt_days/date/tau_longterm",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 4221-4251 passes EndOfYear into season before StomateLpj",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90 line 4360 uses MODULO(date,365) as the STICS day-of-year boundary",
)


def _paper_first_step_do_slow(tstep: int, runtime: DriverRuntimeScalars) -> bool:
    """Return the first-step ``do_slow`` flag from the slowproc schedule."""

    steps_per_stomate = int(round(runtime.dt_stomate / runtime.dt_sechiba))
    if steps_per_stomate <= 0:
        raise ValueError("dt_stomate / dt_sechiba must be positive")
    return (int(tstep) + 1) % steps_per_stomate == 0


def _paper_end_of_year(tstep: int, runtime: DriverRuntimeScalars) -> bool:
    """Return the paper-case ``EndOfYear`` flag at a ``do_slow`` boundary."""

    steps_per_stomate = int(round(runtime.dt_stomate / runtime.dt_sechiba))
    if steps_per_stomate <= 0:
        raise ValueError("dt_stomate / dt_sechiba must be positive")
    steps_per_year = 365 * steps_per_stomate
    return (int(tstep) + 1) % steps_per_year == 0


def _stomate_lpj_initial_cn_ind(state, *, dtype=np.float64):
    """Local ``StomateLpj`` initializes ``cn_ind`` to zero before ``prescribe``."""

    shape = state.ind.shape
    return np.zeros(shape, dtype=dtype)


def _stomate_restart_day_of_year(date: int) -> int:
    """Return the STOMATE year-day scalar used by current paper-case STOMATE code."""

    return int(date) % 365


def _paper_first_step_stomate_restart_input_bundles(
    *,
    config_path: str | Path,
    run_def_path: Path,
    restart_state: ReferenceCaseFirstStepRestartState,
    runtime: DriverRuntimeScalars,
    tstep: int,
    daily_fields: dict[str, object] | None = None,
    run_def_values: dict[str, str] | None = None,
) -> tuple[PaperCaseStomateBundleSourceKwargs, StomateRestartInputBundles]:
    """Build explicit STOMATE chain dictionaries at a ``do_slow`` boundary.

    ``daily_fields`` should come from the ordered STOMATE daily accumulator
    fold once the preceding SECHIBA steps are model-produced. If it is omitted,
    the restart-backed daily state remains visible in the bundle rather than
    being silently replaced by synthetic values.
    """

    do_slow = _paper_first_step_do_slow(tstep, runtime)
    if not do_slow:
        raise ValueError("STOMATE main-chain bundles are only valid at a do_slow boundary")

    season_state = restart_state.stomate_readstart.season_state
    daily_state = restart_state.stomate_readstart.daily_state
    source_daily_state = (
        SimpleNamespace(**{**daily_state._asdict(), **daily_fields})
        if daily_fields is not None
        else daily_state
    )
    source = paper_case_stomate_bundle_source_kwargs(
        config_path=config_path,
        season=season_state,
        daily=source_daily_state,
        entry_state=restart_state.stomate,
        slowproc_state=restart_state.slowproc,
        veget=restart_state.slowproc.veget,
        veget_max=restart_state.slowproc.veget_max,
        dt_days=runtime.dt_days,
        julian_diff=_stomate_restart_day_of_year(season_state.date),
        end_of_year=_paper_end_of_year(tstep, runtime),
        firstcall_season=True,
        used_run_def=run_def_path,
        run_def_values=run_def_values,
    )
    bundles = stomate_restart_input_bundles(
        state=restart_state.stomate,
        veget_max=restart_state.slowproc.veget_max,
        lai=restart_state.slowproc.lai,
        cn_ind=_stomate_lpj_initial_cn_ind(restart_state.stomate),
        dt_days=runtime.dt_days,
        dt_sechiba=runtime.dt_sechiba,
        dt_stomate=runtime.dt_stomate,
        do_slow=do_slow,
        stomate_restart_none=False,
        daily_fields=daily_fields,
        **source.kwargs,
    )
    return source, bundles


def _paper_cold_start_stomate_input_bundles(
    *,
    config_path: str | Path,
    run_def_path: Path,
    first_step_coverage: ColdStartFirstStepStateCoverage,
    runtime: DriverRuntimeScalars,
    tstep: int,
    daily_fold: StomateDailyProcessFold | None,
    run_def_values: dict[str, str] | None = None,
) -> tuple[PaperCaseStomateBundleSourceKwargs | None, StomateRestartInputBundles | None, tuple[DriverDayStateGap, ...]]:
    """Build cold-start STOMATE daily-carbon bundles from no-restart state."""

    if daily_fold is None or not daily_fold.ok:
        return None, None, (
            DriverDayStateGap(
                component="cold_start_daily_process_fold",
                fields=("daily_process_fold",),
                provenance=("fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 3187-3267",),
            ),
        )
    do_slow = _paper_first_step_do_slow(tstep, runtime)
    if not do_slow:
        raise ValueError("cold-start STOMATE bundles are only valid at a do_slow boundary")
    payloads = first_step_coverage.initialized_payloads
    required = (
        "stomate_cold_start_entry_state",
        "stomate_cold_start_season_state",
        "stomate_cold_start_daily_accumulators",
        "slowproc_cold_start_vegetation",
    )
    missing = tuple(name for name in required if name not in payloads)
    previous_state = payloads.get("cold_start_first_step_previous_state")
    if previous_state is None:
        missing = (*missing, "cold_start_first_step_previous_state")
    if missing:
        return None, None, (
            DriverDayStateGap(
                component="cold_start_stomate_input_bundles",
                fields=missing,
                provenance=first_step_coverage.provenance,
            ),
        )
    entry_state = payloads["stomate_cold_start_entry_state"]
    season_state = payloads["stomate_cold_start_season_state"]
    slowproc_state = previous_state.fields_by_component["slowproc_stomate_previous_step_state"]
    veget = payloads["slowproc_cold_start_vegetation"].vegetation.veget
    veget_max = payloads["slowproc_cold_start_vegetation"].vegetation.veget_max
    daily_state = SimpleNamespace(**daily_fold.daily_fields)
    source = paper_case_stomate_bundle_source_kwargs(
        config_path=config_path,
        season=season_state,
        daily=daily_state,
        entry_state=entry_state,
        slowproc_state=SimpleNamespace(
            lai=slowproc_state["lai"],
            height=slowproc_state["height"],
            frac_age=slowproc_state["frac_age"],
            veget=veget,
            veget_max=veget_max,
        ),
        veget=veget,
        veget_max=veget_max,
        dt_days=runtime.dt_days,
        julian_diff=_stomate_restart_day_of_year(season_state.date),
        end_of_year=_paper_end_of_year(tstep, runtime),
        firstcall_season=True,
        used_run_def=run_def_path,
        run_def_values=run_def_values,
    )
    bundles = stomate_restart_input_bundles(
        state=entry_state,
        veget_max=veget_max,
        lai=slowproc_state["lai"],
        cn_ind=_stomate_lpj_initial_cn_ind(entry_state),
        dt_days=runtime.dt_days,
        dt_sechiba=runtime.dt_sechiba,
        dt_stomate=runtime.dt_stomate,
        do_slow=do_slow,
        stomate_restart_none=True,
        daily_fields=daily_fold.daily_fields,
        **source.kwargs,
    )
    return source, bundles, ()


LATER_DAY_STOMATE_ENTRY_STATE_OVERRIDES = (
    "biomass",
    "resp_maint_part",
    "gpp_daily",
    "npp_daily",
    "turnover_daily",
    "resp_maint",
    "resp_growth",
    "leaf_age",
    "leaf_frac",
    "age",
    "sla_calc",
    "pft_present",
    "veget_lastlight",
    "need_adjacent",
    "ind",
    "adapted",
    "regenerate",
    "npp_longterm",
    "turnover_longterm",
    "lm_lastyearmax",
    "turnover_time",
    "senescence",
    "when_growthinit",
    "co2_to_bm",
    "everywhere",
    "bm_to_litter",
    "carb_mass_total",
    "rip_time",
    "assim_param",
    "altmax",
    "fixed_cryoturbation_depth",
    "fpeat",
    "litter_above",
    "litter_below",
    "litterpart",
    "dead_leaves",
    "carbon",
    "litter",
    "lignin_struc",
    "prod10",
    "prod100",
    "flux10",
    "flux100",
    "fuel_1hr",
    "fuel_10hr",
    "fuel_100hr",
    "fuel_1000hr",
    "carbon_32l",
    "DOC",
    "interception_storage",
    "lignin_struc_above",
    "lignin_struc_below",
    "deepC_a",
    "deepC_s",
    "deepC_p",
    "soilc_total",
    "thawed_humidity",
    "depth_organic_soil",
)

# State carried unchanged through each half-hour SECHIBA transition. Daily
# STOMATE/season fields omitted here remain available from the day-start packet
# and are replaced at the day-end boundary.
STOMATE_HALF_HOUR_CARRY_FIELDS = (
    "veget", "veget_max", "lai", "frac_nobio", "height", "frac_age",
    "assim_param", "altmax", "fpeat", "pft_present", "veget_lastlight",
    "need_adjacent", "everywhere", "when_growthinit", "biomass", "leaf_frac",
    "leaf_age", "age", "sla_calc", "ind", "cn_ind", "co2_to_bm", "adapted",
    "regenerate", "npp_longterm", "turnover_daily", "bm_to_litter",
    "senescence", "turnover_time", "rip_time", "gpp_daily", "npp_daily",
    "resp_maint", "resp_growth", "litter_above", "litter_below", "litterpart",
    "dead_leaves", "carbon", "litter", "lignin_struc", "prod10", "prod100",
    "flux10", "flux100", "fuel_1hr", "fuel_10hr", "fuel_100hr",
    "fuel_1000hr", "carbon_32l", "DOC", "interception_storage",
    "carb_mass_total", "soiltile", "totfrac_nobio", "tot_bare_soil",
    "lignin_struc_above", "lignin_struc_below", "deepC_a", "deepC_s",
    "deepC_p", "soilc_total", "thawed_humidity", "depth_organic_soil",
    "resp_maint_part", "daily_accumulators", "t2m_longterm",
    "dt_days_read", "resp_hetero", "tsurf_year", "deepC_peat",
    "depth_deepsoil", "t2m_14",
    *STOMATE_FIXED_PATH_CARRY_FIELDS,
)


def _write_maintenance_lai_to_previous_fields(
    fields: dict[str, dict[str, object]],
    prior_slowproc: Mapping[str, object],
    ok_laidev: tuple[bool, ...] | None,
) -> None:
    """Apply ``maint_respiration``'s LAI output before the next SECHIBA step.

    Fortran provenance: ``stomate_resp.f90::maint_respiration`` lines 317-327
    sets bare-soil LAI to zero and, where ``ok_LAIdev`` is false, writes
    ``biomass(:,:,ileaf,icarbon) * sla_calc``. ``sechiba.f90`` then returns
    that state to the next half-hour ``diffuco_main`` call.
    """

    if ok_laidev is None or not all(name in prior_slowproc for name in ("lai", "biomass", "sla_calc")):
        return
    previous_lai = jnp.asarray(prior_slowproc["lai"])
    biomass = jnp.asarray(prior_slowproc["biomass"])
    sla_calc = jnp.asarray(prior_slowproc["sla_calc"])
    switches = jnp.asarray(ok_laidev, dtype=bool)
    if biomass.ndim != 4 or previous_lai.shape != biomass.shape[:2] or sla_calc.shape != biomass.shape[:2]:
        raise ValueError("maintenance LAI writeback requires matching lai, biomass, and sla_calc shapes")
    if switches.shape != (biomass.shape[1],):
        raise ValueError("ok_laidev must match the PFT axis")
    derived = biomass[:, :, 0, 0] * sla_calc
    derived = derived.at[:, 0].set(0.0)
    lai = jnp.where(switches[None, :], previous_lai, derived)
    fields["slowproc_stomate_previous_step_state"]["lai"] = lai
    fields["diffuco_previous_step_state"]["lai"] = lai


def _paper_later_day_stomate_entry_state_from_previous(
    *,
    template: StomateRestartEntryState,
    previous_state: DriverPreviousStepStatePacket,
) -> tuple[StomateRestartEntryState | None, tuple[DriverDayStateGap, ...]]:
    """Overlay model-produced STOMATE state on the restart-entry template.

    The template is used only for fields not yet produced by the current
    explicit chain. Missing override fields remain explicit gaps so later
    development can remove the remaining restart-backed tail state one process
    at a time.
    """

    slowproc = previous_state.fields_by_component.get("slowproc_stomate_previous_step_state", {})
    missing = tuple(name for name in LATER_DAY_STOMATE_ENTRY_STATE_OVERRIDES if name not in slowproc)
    if missing:
        return None, (
            DriverDayStateGap(
                component="later_day_stomate_entry_state",
                fields=missing,
                provenance=(
                    "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 4225-4364 updates STOMATE state",
                    "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj lines 1070-1557 updates daily carbon state",
                ),
                notes=("Later-day STOMATE state must come from the previous model-produced day, not the initial restart.",),
            ),
        )
    updates = {name: slowproc[name] for name in LATER_DAY_STOMATE_ENTRY_STATE_OVERRIDES}
    return template._replace(**updates), ()


def _paper_later_day_stomate_season_state_from_previous(
    *,
    template: StomateRestartSeasonState,
    previous_state: DriverPreviousStepStatePacket,
) -> tuple[StomateRestartSeasonState | None, tuple[DriverDayStateGap, ...]]:
    """Overlay model-produced season memory on the restart season template."""

    slowproc = previous_state.fields_by_component.get("slowproc_stomate_previous_step_state", {})
    missing = tuple(name for name in STOMATE_DAY_SEASON_STATE_FIELDS if name not in slowproc)
    if missing:
        return None, (
            DriverDayStateGap(
                component="later_day_stomate_season_memory",
                fields=missing,
                provenance=(
                    "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 4221-4251 calls season before StomateLpj",
                    "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 577-772 updates season memory",
                ),
                notes=("Later-day STOMATE season memory must come from the previous model-produced day, not the initial restart.",),
            ),
        )
    updates = {name: slowproc[name] for name in STOMATE_DAY_SEASON_STATE_FIELDS}
    return template._replace(**updates), ()


def _paper_later_day_stomate_input_bundles(
    *,
    config_path: str | Path,
    run_def_path: Path,
    restart_state: ReferenceCaseFirstStepRestartState,
    previous_state: DriverPreviousStepStatePacket,
    runtime: DriverRuntimeScalars,
    tstep: int,
    daily_fold: StomateDailyProcessFold | None,
    firstcall_season: bool = False,
    run_def_values: dict[str, str] | None = None,
    model_day_number=None,
    stomate_parameter_overrides: Mapping[str, object] | None = None,
    stomate_restart_template: StomateRestartEntryState | None = None,
    stomate_season_template: StomateRestartSeasonState | None = None,
) -> tuple[PaperCaseStomateBundleSourceKwargs | None, StomateRestartInputBundles | None, tuple[DriverDayStateGap, ...]]:
    """Build later-day STOMATE bundles without aliasing restart dynamics."""

    if daily_fold is None or not daily_fold.ok:
        return None, None, (
            DriverDayStateGap(
                component="later_day_daily_process_fold",
                fields=("daily_process_fold",),
                provenance=("fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 3187-3267",),
            ),
        )
    do_slow = _paper_first_step_do_slow(tstep, runtime)
    if not do_slow:
        raise ValueError("later-day STOMATE bundles are only valid at a do_slow boundary")
    entry_state, gaps = _paper_later_day_stomate_entry_state_from_previous(
        template=restart_state.stomate if stomate_restart_template is None else stomate_restart_template,
        previous_state=previous_state,
    )
    if gaps or entry_state is None:
        return None, None, gaps
    slowproc_state = previous_state.fields_by_component["slowproc_stomate_previous_step_state"]
    season_template = (
        restart_state.stomate_readstart.season_state
        if stomate_season_template is None
        else stomate_season_template
    )
    season_state, season_gaps = _paper_later_day_stomate_season_state_from_previous(
        template=season_template,
        previous_state=previous_state,
    )
    if season_gaps or season_state is None:
        return None, None, season_gaps
    daily_state = SimpleNamespace(**daily_fold.daily_fields)
    scientific_day = (
        int(tstep // int(round(runtime.dt_stomate / runtime.dt_sechiba))) + 1
        if model_day_number is None
        else model_day_number
    )
    source = paper_case_stomate_bundle_source_kwargs(
        config_path=config_path,
        season=season_state,
        daily=daily_state,
        entry_state=entry_state,
        slowproc_state=SimpleNamespace(
            lai=slowproc_state["lai"],
            height=slowproc_state["height"],
            frac_age=slowproc_state["frac_age"],
            veget=slowproc_state["veget"],
            veget_max=slowproc_state["veget_max"],
            frac_nobio=slowproc_state["frac_nobio"],
        ),
        veget=slowproc_state["veget"],
        veget_max=slowproc_state["veget_max"],
        dt_days=runtime.dt_days,
        julian_diff=scientific_day,
        end_of_year=jnp.equal(jnp.mod(jnp.asarray(scientific_day), 365), 0),
        firstcall_season=firstcall_season,
        used_run_def=run_def_path,
        run_def_values=run_def_values,
        parameter_overrides=stomate_parameter_overrides,
    )
    bundles = stomate_restart_input_bundles(
        state=entry_state,
        veget_max=slowproc_state["veget_max"],
        lai=slowproc_state["lai"],
        cn_ind=slowproc_state["cn_ind"],
        dt_days=runtime.dt_days,
        dt_sechiba=runtime.dt_sechiba,
        dt_stomate=runtime.dt_stomate,
        do_slow=do_slow,
        stomate_restart_none=False,
        daily_fields=daily_fold.daily_fields,
        **source.kwargs,
    )
    return source, bundles, ()


def _paper_ok_leak_pre_step_boundary_from_entry_state(
    *,
    config_path: str | Path,
    run_def_path: Path,
    run_def_values: dict[str, str],
    entry_state: StomateRestartEntryState,
    driver_payload: IntersurfFirstStepPayload,
    run_scalars,
    diaglev,
    deep_zlt,
    deep_znt,
    kjit: int,
    nflow: int,
    t2mdiag=None,
    temp_sol=None,
    final_entry_payload: dict[str, object] | None = None,
) -> StomateOkLeakBoundaryInputs:
    """Build OK_LEAK pre-step inputs from the active STOMATE entry state.

    Fortran provenance: ``stomate_main`` keeps these pools as inout state
    across daily calls (src_stomate/stomate.f90 lines 2409-2448 and
    3288-3489). Later days must therefore use the model-produced entry state,
    not the initial restart image.
    """

    driver_entry_kwargs = {} if final_entry_payload is None else dict(final_entry_payload)
    if t2mdiag is None:
        t2mdiag = driver_entry_kwargs.get("t2mdiag")
    if temp_sol is None:
        temp_sol = driver_entry_kwargs.get("temp_sol")
    return stomate_pre_step_ok_leak_boundary_inputs(
        state=entry_state,
        driver_payload=driver_payload,
        run_scalars=run_scalars,
        diaglev=diaglev,
        zz_coef_deep=deep_zlt,
        zz_deep=deep_znt,
        kjit=kjit,
        nflow=nflow,
        river_routing=parse_run_def_bool(run_def_values["RIVER_ROUTING"]),
        nbp_glo=driver_payload.kjpindex,
        t2mdiag=t2mdiag,
        temp_sol=temp_sol,
        min_wind=parse_run_def_float(run_def_values, "MIN_WIND"),
    ).boundary_inputs


def _paper_day_season_fields_from_bundle_source(
    source: PaperCaseStomateBundleSourceKwargs | None,
    *,
    season_state: StomateRestartSeasonState | None = None,
    prior_fields: Mapping[str, object] | None = None,
    dt_days: float | None = None,
    tsurf_daily=None,
    firstcall: bool = False,
    latitude=None,
    julian_diff=None,
) -> dict[str, object] | None:
    """Return the source-backed season state produced before daily carbon."""

    if source is None:
        return None
    memory = source.pre_step.memory.step.state._asdict()
    annual = source.pre_step.annual.step.state._asdict()
    biometeorology = source.pre_step.biometeorology.step.state._asdict()
    fields = {
        **{name: memory[name] for name in STOMATE_DAY_SEASON_MEMORY_FIELDS if name in memory},
        **{name: annual[name] for name in STOMATE_DAY_SEASON_ANNUAL_FIELDS if name in annual},
        **{name: biometeorology[name] for name in STOMATE_DAY_SEASON_BIOMETEOROLOGY_FIELDS if name in biometeorology},
    }
    if "turnover_longterm" in annual:
        fields["turnover_longterm"] = annual["turnover_longterm"]
    if "lm_lastyearmax" in annual:
        fields["lm_lastyearmax"] = annual["lm_lastyearmax"]
    if season_state is not None:
        fields["gdd_init_date"] = (
            season_update_gdd_init_date(
                season_state.gdd_init_date,
                latitude=latitude,
                julian_diff=julian_diff,
            )
            if latitude is not None and julian_diff is not None
            else season_state.gdd_init_date
        )
    prior = {} if prior_fields is None else prior_fields
    t2m_14_old = getattr(season_state, "t2m_14", prior.get("t2m_14"))
    tsurf_year_old = getattr(season_state, "tsurf_year", prior.get("tsurf_year"))
    if t2m_14_old is None or tsurf_year_old is None or dt_days is None:
        raise ValueError("season writeback requires prior t2m_14/tsurf_year and dt_days")

    previous_date = getattr(season_state, "date", prior.get("date", 0))
    fields["date"] = jnp.asarray(previous_date) + jnp.rint(
        jnp.asarray(dt_days)
    ).astype(jnp.int64)

    if tsurf_daily is None:
        raise ValueError("season writeback requires tsurf_daily")
    surface_temperature = season_surface_temperature_memory_step(
        tsurf_year=tsurf_year_old,
        t2m_14=t2m_14_old,
        tsurf_daily=tsurf_daily,
        t2m_daily=source.kwargs["t2m"],
        dt_days=dt_days,
        firstcall=firstcall,
    )
    fields["t2m_14"] = surface_temperature.t2m_14
    fields["tsurf_year"] = surface_temperature.tsurf_year
    return fields


@dataclass(frozen=True)
class DriverYearScaffold:
    """Preview of ordered timestep scaffolds for a year."""

    year: int
    preview_steps: tuple[DriverTimestepScaffold, ...]
    total_forcing_steps: int
    provenance: tuple[str, ...] = DRIVER_TIMESTEP_ORCHESTRATION_PROVENANCE

    @property
    def ready_to_run_processes(self) -> bool:
        return all(step.ready_to_run_processes for step in self.preview_steps)

    @property
    def missing_components(self) -> tuple[str, ...]:
        missing: list[str] = []
        for step in self.preview_steps:
            missing.extend(step.missing_components)
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class DriverDayStateGap:
    """Exact previous-step state packet needed before a day can be advanced."""

    component: str
    fields: tuple[str, ...]
    provenance: tuple[str, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DriverPreviousStepStatePacket:
    """Audited state fields produced by one completed driver timestep."""

    tstep: int
    fields_by_component: dict[str, dict[str, object]]
    provenance_by_component: dict[str, tuple[str, ...]]

    @property
    def covered_fields(self) -> dict[str, tuple[str, ...]]:
        return {
            component: tuple(fields.keys())
            for component, fields in self.fields_by_component.items()
        }

    @property
    def empty(self) -> bool:
        return not any(self.fields_by_component.values())


@dataclass(frozen=True)
class DriverDiffucoDayStaticCache:
    """DIFFUCO inputs that stay fixed through one STOMATE day."""

    nvm: int
    control_salinity: DiffucoControlSalinityAssembly
    control_inundation: DiffucoControlInundationAssembly
    slowproc_derivvar: object | None
    slowproc_derivvar_payload: dict[str, object]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 973-985 keeps slowproc state fixed between daily STOMATE updates",
        "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_main lines 445-622 computes PFT14 salinity/inundation controls only under firstcall_diffuco",
    )


@dataclass(frozen=True)
class DriverFirstDayStomateStatePacket:
    """Explicit STOMATE state available after the first do_slow day."""

    day_index: int
    fields: dict[str, object]
    provenance_by_field: dict[str, tuple[str, ...]]

    @property
    def covered_fields(self) -> tuple[str, ...]:
        return tuple(self.fields.keys())


@dataclass(frozen=True)
class DriverNextStepDiffucoPrecallScaffold:
    """DIFFUCO pre-call assembly for a later driver step from prior state."""

    tstep: int
    step: DriverStepBundle
    payload: IntersurfFirstStepPayload
    previous_state: DriverPreviousStepStatePacket
    diffuco_precall: DiffucoFirstStepPrecallAssembly
    slowproc_derivvar: object | None
    used_previous_state_fields: tuple[str, ...]
    static_passthrough_fields: tuple[str, ...]
    missing_previous_state_fields: tuple[str, ...]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 839-908 advances forcing steps",
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 997-1005 calls diffuco_main",
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_end lines 3114-3140 migrates temp_sol state",
    )

    @property
    def ready_for_local_diffuco(self) -> bool:
        return self.diffuco_precall.ok and not self.missing_previous_state_fields


@dataclass(frozen=True)
class DriverNextStepEnerbilPrecallScaffold:
    """ENERBIL pre-call assembly for a later driver step from prior state."""

    tstep: int
    step: DriverStepBundle
    payload: IntersurfFirstStepPayload
    previous_state: DriverPreviousStepStatePacket
    diffuco_precall: DiffucoFirstStepPrecallAssembly
    slowproc_derivvar: object | None
    diffuco_local_enerbil_precall: DiffucoPFT14FirstStepLocalEnerbilPrecallAssembly | None
    enerbil_precall: EnerbilFirstStepPrecallAssembly
    enerbil_local: EnerbilFirstStepLocalAssembly | None
    used_previous_state_fields: tuple[str, ...]
    missing_previous_state_fields: tuple[str, ...]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 997-1019",
        "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_main lines 390-545",
        "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_main lines 998-1032",
    )

    @property
    def ready_for_local_enerbil(self) -> bool:
        return self.diffuco_precall.ok and self.enerbil_precall.ok and not self.missing_previous_state_fields


@dataclass(frozen=True)
class DriverNextStepHydrolPrecallScaffold:
    """HYDROL pre-call assembly for a later driver step from prior state."""

    tstep: int
    enerbil: DriverNextStepEnerbilPrecallScaffold
    hydrol_precall: HydrolFirstStepPrecallAssembly | None
    hydrol_module: HydrolFirstStepModuleClosure | None
    condveg_module: CondvegFirstStepModuleClosure | None
    thermosoil_module: ThermosoilFirstStepModuleClosure | None
    missing_previous_state_fields: tuple[str, ...]

    @property
    def ready_for_local_hydrol(self) -> bool:
        return (
            self.enerbil.enerbil_local is not None
            and self.hydrol_precall is not None
            and self.hydrol_precall.ok
            and not self.missing_previous_state_fields
        )

    @property
    def ready_for_next_state(self) -> bool:
        return (
            self.ready_for_local_hydrol
            and self.hydrol_module is not None
            and self.hydrol_module.ok
            and self.condveg_module is not None
            and self.condveg_module.ok
            and self.thermosoil_module is not None
            and self.thermosoil_module.ok
        )


@dataclass(frozen=True)
class DriverRuntimeStepMetadata:
    """Small same-step constants needed after a compact half-hour step."""

    pref_soil_veg: object
    nstm: int
    is_tree: object
    is_peat: object
    lalo: object
    npts: int


@dataclass(frozen=True)
class DriverRuntimeStepResult:
    """Compact half-hour runtime step result."""

    tstep: int
    entry_payload: dict[str, object] | None
    next_state: object | None
    metadata: DriverRuntimeStepMetadata | None
    missing_components: tuple[str, ...]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 839-908 advances forcing steps",
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 997-1224 advances SECHIBA state",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 3187-3267 consumes same-step SECHIBA outputs",
    )

    @property
    def ok(self) -> bool:
        return self.entry_payload is not None and self.next_state is not None and self.metadata is not None and not self.missing_components


@dataclass(frozen=True)
class DriverRuntimeHalfHourInput:
    """State-independent inputs for one half-hour runtime step."""

    step: DriverStepBundle
    base_payload: IntersurfFirstStepPayload


class DriverCompiledHalfHourForcing(NamedTuple):
    """Dynamic forcing arrays consumed by one compiled SECHIBA step."""

    zlev: object
    zlevuv: object
    u: object
    v: object
    qair: object
    temp_air: object
    pb: object
    precip_rain: object
    precip_snow: object
    lwdown: object
    swdown: object
    ccanopy: object
    salinity: object
    tide_height: object


class DriverCompiledDiffucoDayInputs(NamedTuple):
    """Numeric DIFFUCO SAVE/slowproc inputs fixed within one STOMATE day."""

    control_salinity: object
    control_inudate: object
    qsintmax: object
    assim_param: object
    height: object
    temp_growth: object


class DriverCompiledDiffucoParameterValues(NamedTuple):
    """Continuous DIFFUCO parameters that may vary between landpoints."""

    trans_co2: dict[str, object]
    rstruct_const: object


class DriverCompiledStomateParameterValues(NamedTuple):
    """Continuous STOMATE PFT vectors supplied per compatible landpoint."""

    alloc_min: object
    residence_time: object
    vcmax25: object
    maint_resp_slope: object


class DriverCompiledHydrolTableArrays(NamedTuple):
    """Dynamic mineral CWRR table arrays for one landpoint batch."""

    mc_lin: object
    k_lin: object
    a_lin: object
    b_lin: object
    d_lin: object
    kfact: object
    nfact: object
    afact: object
    nvan_mod: object
    avan_mod: object


@dataclass(frozen=True)
class DriverRuntimeDayTransitionInputs:
    """Per-day inputs bound before the half-hour runtime transition."""

    year: int
    start: int
    steps_per_stomate: int
    run_def_path: Path
    day_start_bundle: DriverStepBundle
    hydrol_static_template: dict[str, object] | None
    hydrol_runtime_static_tables: object | None
    diffuco_day_static_cache: DriverDiffucoDayStaticCache | None
    half_hour_inputs: tuple[DriverRuntimeHalfHourInput, ...] = ()
    compiled_forcing_series: DriverCompiledHalfHourForcing | None = None
    compiled_base_payload_template: IntersurfFirstStepPayload | None = None


@dataclass(frozen=True)
class DriverRuntimeDayStepTransition:
    """Half-hour SECHIBA transition result for one STOMATE day."""

    completed_entry_payloads: tuple[dict[str, object], ...]
    current_state: object
    first_step_metadata: DriverRuntimeStepMetadata | None
    stopped_at_tstep: int | None
    state_gaps: tuple[DriverDayStateGap, ...]
    compiled_entry_stacks: Mapping[str, object] | None = None

    @property
    def ok(self) -> bool:
        return self.stopped_at_tstep is None and not self.state_gaps


@dataclass(frozen=True)
class DriverDayScaffold:
    """Strict first-day driver/STOMATE scheduling scaffold.

    This object expands the paper-case first STOMATE day in Fortran driver
    order without aliasing the initial restart as later prognostic state.
    """

    year: int
    start_tstep: int
    steps_per_stomate: int
    initial_state_mode: str
    preview_steps: tuple[DriverTimestepScaffold, ...]
    do_slow_flags: tuple[bool, ...]
    completed_entry_payloads: tuple[dict[str, object], ...]
    previous_step_state: DriverPreviousStepStatePacket | None
    stopped_at_tstep: int | None
    state_gaps: tuple[DriverDayStateGap, ...]
    daily_process_ready: bool
    daily_process_fold: StomateDailyProcessFold | None
    daily_carbon_boundary: DailyCarbonBoundary | None
    stomate_restart_input_bundles: StomateRestartInputBundles | None
    stomate_daily_carbon: ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult | None
    ok_leak_boundary_inputs: StomateOkLeakBoundaryInputs | None
    ok_leak_boundary_gaps: tuple[DriverDayStateGap, ...]
    stomate_ok_leak: ExplicitOkLeakFromPostNppResult | None
    stomate_outputs: ExplicitStomateLpjOutputsResult | None
    daily_reset_fields: dict[str, object] | None
    first_day_stomate_state: DriverFirstDayStomateStatePacket | None
    first_day_end_state: DriverPreviousStepStatePacket | None
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 839-908 advances forcing steps",
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90 lines 656-662 schedules do_slow",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 3187-3267 accumulates daily fields before do_slow",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 4225-4364 enters daily STOMATE processes only at do_slow",
        "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90 lines 1578-1663 and 1809-1880 prepares end-of-day diagnostics",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 4950-5000 resets daily accumulators after do_slow",
    )
    notes: tuple[str, ...] = (
        "This scaffold does not fabricate tstep>0 prognostic state from restart files.",
        "A complete first-day closure requires model-produced SECHIBA/STOMATE state after every preceding half-hour step.",
    )

    @property
    def ready_for_day_boundary(self) -> bool:
        return (
            self.daily_process_ready
            and self.stomate_restart_input_bundles is not None
            and self.stopped_at_tstep is None
            and not self.state_gaps
        )

    @property
    def ready_for_first_daily_carbon(self) -> bool:
        return self.ready_for_day_boundary and self.stomate_daily_carbon is not None

    @property
    def ready_for_ok_leak(self) -> bool:
        return self.ready_for_first_daily_carbon and self.stomate_ok_leak is not None and not self.ok_leak_boundary_gaps

    @property
    def ready_for_outputs(self) -> bool:
        return self.ready_for_ok_leak and self.stomate_outputs is not None

    @property
    def ready_for_first_day_end_state(self) -> bool:
        return (
            self.ready_for_outputs
            and self.daily_reset_fields is not None
            and self.first_day_stomate_state is not None
            and self.first_day_end_state is not None
        )

    @property
    def missing_components(self) -> tuple[str, ...]:
        missing: list[str] = []
        for step in self.preview_steps:
            missing.extend(step.missing_components)
        for gap in self.state_gaps:
            missing.append(gap.component)
            missing.extend(f"{gap.component}:{field}" for field in gap.fields)
        for gap in self.ok_leak_boundary_gaps:
            missing.append(gap.component)
            missing.extend(f"{gap.component}:{field}" for field in gap.fields)
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class DriverLaterDayScaffold:
    """Strict driver day advanced from a previous model-produced state packet."""

    year: int
    start_tstep: int
    steps_per_stomate: int
    initial_state_mode: str
    input_previous_state: DriverPreviousStepStatePacket
    completed_entry_payloads: tuple[dict[str, object], ...]
    previous_step_state: DriverPreviousStepStatePacket | None
    stopped_at_tstep: int | None
    state_gaps: tuple[DriverDayStateGap, ...]
    daily_process_ready: bool
    daily_process_fold: StomateDailyProcessFold | None
    stomate_restart_input_bundles: StomateRestartInputBundles | None
    stomate_bundle_gaps: tuple[DriverDayStateGap, ...]
    stomate_daily_carbon: ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult | None
    ok_leak_boundary_inputs: StomateOkLeakBoundaryInputs | None
    ok_leak_boundary_gaps: tuple[DriverDayStateGap, ...]
    stomate_ok_leak: ExplicitOkLeakFromPostNppResult | None
    stomate_outputs: ExplicitStomateLpjOutputsResult | None
    daily_reset_fields: dict[str, object] | None
    day_stomate_state: DriverFirstDayStomateStatePacket | None
    day_end_state: DriverPreviousStepStatePacket | None
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 839-908 advances forcing steps",
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 997-1224 advances SECHIBA state",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 3187-3267 accumulates daily STOMATE fields",
    )
    notes: tuple[str, ...] = (
        "This scaffold starts from an audited previous-day state packet; it does not reread restart as prognostic state.",
        "Daily STOMATE carbon is deliberately separate until all required dynamic STOMATE state is mapped from previous_state.",
    )

    @property
    def ready_for_day_boundary(self) -> bool:
        return (
            self.stopped_at_tstep is None
            and not self.state_gaps
            and self.daily_process_fold is not None
            and self.daily_process_ready
        )

    @property
    def ready_for_day_end_state(self) -> bool:
        return (
            self.ready_for_day_boundary
            and self.stomate_restart_input_bundles is not None
            and not self.stomate_bundle_gaps
            and self.stomate_daily_carbon is not None
            and self.ok_leak_boundary_inputs is not None
            and not self.ok_leak_boundary_gaps
            and self.stomate_ok_leak is not None
            and self.stomate_outputs is not None
            and self.daily_reset_fields is not None
            and self.day_stomate_state is not None
            and self.day_end_state is not None
        )

    @property
    def missing_components(self) -> tuple[str, ...]:
        missing: list[str] = []
        for gap in self.state_gaps:
            missing.append(gap.component)
            missing.extend(f"{gap.component}:{field}" for field in gap.fields)
        for gap in self.stomate_bundle_gaps:
            missing.append(gap.component)
            missing.extend(f"{gap.component}:{field}" for field in gap.fields)
        for gap in self.ok_leak_boundary_gaps:
            missing.append(gap.component)
            missing.extend(f"{gap.component}:{field}" for field in gap.fields)
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class DriverColdStartDayScaffold:
    """Trace-free first-day scaffold from Fortran no-restart initialization."""

    year: int
    start_tstep: int
    steps_per_stomate: int
    initial_state_mode: str
    first_step_coverage: ColdStartFirstStepStateCoverage
    completed_entry_payloads: tuple[dict[str, object], ...]
    completed_steps: int
    previous_step_state: DriverPreviousStepStatePacket | None
    stopped_at_tstep: int | None
    state_gaps: tuple[DriverDayStateGap, ...]
    daily_process_fold: StomateDailyProcessFold | None
    daily_process_ready: bool
    stomate_restart_input_bundles: StomateRestartInputBundles | None = None
    stomate_bundle_gaps: tuple[DriverDayStateGap, ...] = ()
    stomate_daily_carbon: ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult | None = None
    ok_leak_boundary_inputs: StomateOkLeakBoundaryInputs | None = None
    ok_leak_boundary_gaps: tuple[DriverDayStateGap, ...] = ()
    stomate_ok_leak: ExplicitOkLeakFromPostNppResult | None = None
    stomate_outputs: ExplicitStomateLpjOutputsResult | None = None
    daily_reset_fields: dict[str, object] | None = None
    first_day_stomate_state: DriverFirstDayStomateStatePacket | None = None
    first_day_end_state: DriverPreviousStepStatePacket | None = None
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_initialize lines 602-735",
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 997-1224",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 3187-3267",
    )
    notes: tuple[str, ...] = (
        "This scaffold starts from explicit no-restart initialization, not case_001_071 restart files.",
        "STOMATE daily carbon and day-end state remain separate until their cold-start input bundles are source-closed.",
    )

    @property
    def ready_for_day_boundary(self) -> bool:
        return (
            self.completed_steps == self.steps_per_stomate
            and self.stopped_at_tstep is None
            and not self.state_gaps
            and self.daily_process_ready
        )

    @property
    def ready_for_first_day_end_state(self) -> bool:
        return self.ready_for_day_boundary and self.first_day_end_state is not None

    @property
    def missing_components(self) -> tuple[str, ...]:
        missing: list[str] = []
        for gap in self.state_gaps:
            missing.append(gap.component)
            missing.extend(f"{gap.component}:{field}" for field in gap.fields)
        for gap in self.stomate_bundle_gaps:
            missing.append(gap.component)
            missing.extend(f"{gap.component}:{field}" for field in gap.fields)
        for gap in self.ok_leak_boundary_gaps:
            missing.append(gap.component)
            missing.extend(f"{gap.component}:{field}" for field in gap.fields)
        if self.ready_for_day_boundary and self.first_day_end_state is None:
            missing.append("cold_start_stomate_daily_carbon_and_day_end_state")
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class DriverMultiDayScaffold:
    """Strict consecutive-day driver scaffold advanced only by model state."""

    year: int
    requested_days: int
    steps_per_stomate: int
    initial_state_mode: str
    days: tuple[DriverDayScaffold | DriverLaterDayScaffold, ...]
    stopped_day_index: int | None
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 839-908 advances forcing steps",
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90 lines 656-662 schedules daily do_slow calls",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 3187-3267 and 4225-4364 advance daily STOMATE state",
    )
    notes: tuple[str, ...] = (
        "Each later day starts from the previous day_end_state packet; initial restart files are not reused as later prognostic state.",
    )

    @property
    def ready_for_requested_days(self) -> bool:
        return self.stopped_day_index is None and len(self.days) == self.requested_days

    @property
    def last_day_end_state(self) -> DriverPreviousStepStatePacket | None:
        if not self.days:
            return None
        last = self.days[-1]
        if isinstance(last, DriverDayScaffold):
            return last.first_day_end_state
        return last.day_end_state

    @property
    def missing_components(self) -> tuple[str, ...]:
        missing: list[str] = []
        for index, day in enumerate(self.days, start=1):
            if isinstance(day, DriverDayScaffold) and not day.ready_for_first_day_end_state:
                missing.extend(f"day_{index}:{item}" for item in day.missing_components)
                missing.append(f"day_{index}:day_end_state")
            if isinstance(day, DriverLaterDayScaffold) and not day.ready_for_day_end_state:
                missing.extend(f"day_{index}:{item}" for item in day.missing_components)
                missing.append(f"day_{index}:day_end_state")
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class DriverDailyModelout:
    """Modelout values produced at one completed STOMATE daily boundary."""

    day_index: int
    start_tstep: int
    end_tstep: int
    modelout_fields: dict[str, object]
    modelout: object
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 839-908 advances forcing steps",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 4225-4364 calls daily STOMATE at do_slow",
        "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90 lines 2200-2246 writes STOMATE history fields consumed by paper modelout",
    )


class DriverPreDailyTrainingBoundary(NamedTuple):
    """Numeric Teacher boundary retained by compiled sample-generation blocks."""

    half_hour_state_values: tuple[tuple[object, ...], ...]
    daily_fields: Mapping[str, object]
    ok_leak_updates: Mapping[str, object]
    ok_leak_driver_steps: object
    deepc_peat: object
    final_diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class DriverRuntimeDayResult:
    """Compact later-day runtime result for long sequence validation."""

    year: int
    day_index: int
    start_tstep: int
    steps_per_stomate: int
    daily_modelout: DriverDailyModelout | None
    day_end_state: DriverPreviousStepStatePacket | None
    stopped_at_tstep: int | None
    missing_components: tuple[str, ...]
    daily_process_fold: StomateDailyProcessFold | None = None
    pre_daily_training_boundary: DriverPreDailyTrainingBoundary | None = None
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 839-908 advances forcing steps",
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 997-1224 advances SECHIBA state",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 3187-3267 and 4225-4364 advance daily STOMATE state",
    )
    notes: tuple[str, ...] = (
        "Runtime day uses the same source-backed process functions as DriverLaterDayScaffold but retains only modelout, day-end state, and gaps.",
    )

    @property
    def ready_for_day_end_state(self) -> bool:
        return self.stopped_at_tstep is None and self.daily_modelout is not None and self.day_end_state is not None and not self.missing_components


@dataclass(frozen=True)
class DriverMultiDayModeloutRun:
    """Closed paper-case driver run through daily STOMATE modelout fields."""

    scaffold: DriverMultiDayScaffold
    daily_modelout: tuple[DriverDailyModelout, ...]

    @property
    def ready_for_requested_days(self) -> bool:
        return self.scaffold.ready_for_requested_days and len(self.daily_modelout) == self.scaffold.requested_days

    @property
    def missing_components(self) -> tuple[str, ...]:
        return self.scaffold.missing_components

    @property
    def last_day_end_state(self) -> DriverPreviousStepStatePacket | None:
        return self.scaffold.last_day_end_state


@dataclass(frozen=True)
class DriverLiteMultiDayModeloutRun:
    """Closed paper-case modelout run without retaining per-day scaffolds.

    The process order and source-backed functions are identical to
    ``DriverMultiDayModeloutRun``. This runtime form keeps only daily modelout,
    the last day-end state, and the first missing boundary, so long validation
    runs do not carry every audit scaffold in the result tree.
    """

    year: int
    requested_days: int
    steps_per_stomate: int
    initial_state_mode: str
    daily_modelout: tuple[DriverDailyModelout, ...]
    stopped_day_index: int | None
    _missing_components: tuple[str, ...]
    last_day_end_state: DriverPreviousStepStatePacket | None
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 839-908 advances forcing steps",
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90 lines 656-662 schedules daily do_slow calls",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 3187-3267 and 4225-4364 advance daily STOMATE state",
    )
    notes: tuple[str, ...] = (
        "Lite runtime keeps the same source-backed day functions but drops completed day scaffolds after extracting modelout and day-end state.",
    )

    @property
    def ready_for_requested_days(self) -> bool:
        return self.stopped_day_index is None and len(self.daily_modelout) == self.requested_days

    @property
    def missing_components(self) -> tuple[str, ...]:
        return self._missing_components


def _driver_day_modelout(day: DriverDayScaffold | DriverLaterDayScaffold, *, day_index: int) -> DriverDailyModelout | None:
    outputs = day.stomate_outputs
    if outputs is None or outputs.requires_trace:
        return None
    if isinstance(day, DriverDayScaffold):
        if not day.ready_for_first_day_end_state:
            return None
    elif not day.ready_for_day_end_state:
        return None
    return DriverDailyModelout(
        day_index=int(day_index),
        start_tstep=int(day.start_tstep),
        end_tstep=int(day.start_tstep + day.steps_per_stomate - 1),
        modelout_fields=dict(outputs.modelout_fields),
        modelout=outputs.modelout,
    )


def _driver_modelout_from_outputs(
    outputs: ExplicitStomateLpjOutputsResult | None,
    *,
    day_index: int,
    start_tstep: int,
    steps_per_stomate: int,
) -> DriverDailyModelout | None:
    if outputs is None or outputs.requires_trace:
        return None
    return DriverDailyModelout(
        day_index=int(day_index),
        start_tstep=int(start_tstep),
        end_tstep=int(start_tstep + steps_per_stomate - 1),
        modelout_fields=dict(outputs.modelout_fields),
        modelout=outputs.modelout,
    )


DAY1_PREVIOUS_STEP_STATE_GAPS = (
    DriverDayStateGap(
        component="diffuco_previous_step_state",
        fields=(
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
            "veget",
            "veget_max",
            "lai",
            "frac_nobio",
            "qsintveg",
            "humrel",
            "evap_bare_lim",
            "drysoil_frac",
            "height",
        ),
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90 lines 997-1005 passes current state into diffuco_main",
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90 lines 1226-1248/sechiba_end migrates new state for the next driver step",
        ),
    ),
    DriverDayStateGap(
        component="enerbil_previous_step_state",
        fields=(
            "temp_sol",
            "temp_sol_pft",
            "qsurf",
            "soilcap",
            "soilcap_pft",
            "soilflx",
            "soilflx_pft",
            "albedo",
            "roughness",
        ),
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_main lines 485-545",
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90 lines 1226-1248/sechiba_end exposes next-step surface state",
        ),
    ),
    DriverDayStateGap(
        component="hydrol_previous_step_state",
        fields=(
            "mc",
            "mcl",
            "qsintveg",
            "humrel",
            "humrelv",
            "us",
            "evap_bare_lim",
            "evap_bare_lim_ns",
            "runoff",
            "drainage",
            "flood_res",
            "fwet_new",
        ),
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_main lines 1206-1296",
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90 restart/state writes lines 2705-3090",
        ),
    ),
    DriverDayStateGap(
        component="thermosoil_previous_step_state",
        fields=(
            "ptn",
            "cgrnd",
            "dgrnd",
            "stempdiag",
            "tdeep",
            "hsdeep",
            "soilcap",
            "soilflx",
        ),
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_main lines 893-1032",
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90 lines 1109-1118 and 1226-1248",
        ),
    ),
    DriverDayStateGap(
        component="slowproc_stomate_previous_step_state",
        fields=(
            "veget",
            "veget_max",
            "lai",
            "biomass",
            "ind",
            "carbon_32l",
            "litter_above",
            "litter_below",
            "litterpart",
            "dead_leaves",
            "carbon",
            "litter",
            "lignin_struc",
            "prod10",
            "prod100",
            "flux10",
            "flux100",
            "fuel_1hr",
            "fuel_10hr",
            "fuel_100hr",
            "fuel_1000hr",
            "resp_maint_part",
            "daily_accumulators",
        ),
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90 lines 973-1121",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 3187-3267 and 4225-4364",
        ),
    ),
)


YEAR_HANDOFF_STATE_GAPS = (
    *DAY1_PREVIOUS_STEP_STATE_GAPS,
    DriverDayStateGap(
        component="driver_previous_step_state",
        fields=("albedo",),
        provenance=(
            "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 1144-1180 reuses previous surface state for the next main call",
            "fortran_source/ORCHIDEE/src_sechiba/condveg.f90 writes albedo for driver restart/state handoff",
        ),
        notes=("Needed for a restart-style next-year first step without falling back to initial z0/albedo assumptions.",),
    ),
    DriverDayStateGap(
        component="slowproc_stomate_previous_step_state",
        fields=(
            "pft_present",
            "everywhere",
            "when_growthinit",
            "leaf_frac",
            "leaf_age",
            "age",
            "sla_calc",
            "cn_ind",
            "co2_to_bm",
            "adapted",
            "regenerate",
            "npp_longterm",
            "turnover_longterm",
            "lm_lastyearmax",
            "turnover_daily",
            "bm_to_litter",
            "senescence",
            "turnover_time",
            "rip_time",
            "npp_daily",
            "resp_maint",
            "resp_growth",
            "litterpart",
            "dead_leaves",
            "fuel_1hr",
            "fuel_10hr",
            "fuel_100hr",
            "fuel_1000hr",
            "lignin_struc_above",
            "lignin_struc_below",
            "soilc_total",
            "t2m_longterm",
        ),
        provenance=(
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 2409-2448 and 4225-4364 carry STOMATE state across calls",
            "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90 writes STOMATE restart fields for yearly rollover",
        ),
        notes=("Year-to-year restart needs the broader mutable STOMATE state, not only daily modelout fields.",),
    ),
)


def _filter_day_state_gaps(
    gaps: tuple[DriverDayStateGap, ...],
    state: DriverPreviousStepStatePacket | None,
) -> tuple[DriverDayStateGap, ...]:
    if state is None:
        return gaps
    filtered: list[DriverDayStateGap] = []
    for gap in gaps:
        covered = set(state.fields_by_component.get(gap.component, ()))
        remaining = tuple(field for field in gap.fields if field not in covered)
        if remaining:
            filtered.append(
                DriverDayStateGap(
                    component=gap.component,
                    fields=remaining,
                    provenance=gap.provenance,
                    notes=gap.notes,
                )
            )
    return tuple(filtered)


def driver_year_handoff_state_gaps(
    state: DriverPreviousStepStatePacket | None,
) -> tuple[DriverDayStateGap, ...]:
    """Return missing fields before a day-end packet can seed a new model year.

    This is a restart contract check only. It does not write NetCDF restart
    files and it does not rebase the driver timestep counter.
    """

    return _filter_day_state_gaps(YEAR_HANDOFF_STATE_GAPS, state)


def rebase_driver_state_for_year_start(
    state: DriverPreviousStepStatePacket,
) -> DriverPreviousStepStatePacket:
    """Return the same dynamic state with a driver counter suitable for tstep 0.

    Fortran yearly scripts restart the executable with new forcing-year
    counters while carrying restart state forward. The local runtime still
    checks ``previous_state.tstep == tstep - 1``; rebasing to ``-1`` makes that
    contract explicit for a future next-year first-step runner.
    """

    fields_by_component = {
        component: dict(fields)
        for component, fields in state.fields_by_component.items()
    }
    hydrol_fields = fields_by_component.get("hydrol_previous_step_state")
    if hydrol_fields is not None:
        # Fortran hydrol_end writes `us` but not dynamic `nroot` to restart
        # (hydrol.f90 lines 1730 and 1727-1748). A new executable year
        # therefore recomputes nroot in hydrol_var_init before the first
        # hydrol_soil diagnostic update.
        hydrol_fields.pop("nroot", None)

    return DriverPreviousStepStatePacket(
        tstep=-1,
        fields_by_component=fields_by_component,
        provenance_by_component={
            component: (
                *provenance,
                "local restart rebase: yearly driver counter resets before the next forcing year",
                "local restart rebase: HYDROL dynamic nroot is not a Fortran restart field and is reinitialized for the new executable year",
            )
            for component, provenance in state.provenance_by_component.items()
        },
    )


def _empty_previous_step_state_fields() -> dict[str, dict[str, object]]:
    return {
        "diffuco_previous_step_state": {},
        "enerbil_previous_step_state": {},
        "hydrol_previous_step_state": {},
        "thermosoil_previous_step_state": {},
        "slowproc_stomate_previous_step_state": {},
        "sechiba_finalize_state": {},
    }


def _previous_step_state_provenance() -> dict[str, tuple[str, ...]]:
    return {
        "diffuco_previous_step_state": (
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_end lines 3114-3140",
            "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_main lines 456-467 and 927-1651",
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_main lines 1041-1063 and 9106-9155",
            "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 1103-1128",
        ),
        "enerbil_previous_step_state": (
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_end lines 3114-3140",
            "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_main lines 998-1032 and 1103-1139",
            "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_main lines 398-435",
        ),
        "hydrol_previous_step_state": (
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_main lines 1206-1296",
            "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_soil lines 5405-6815",
        ),
        "thermosoil_previous_step_state": (
            "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_main lines 893-1032",
            "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_finalize lines 1103-1139",
        ),
        "slowproc_stomate_previous_step_state": (
            "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90 lines 973-1121",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 3187-3267",
        ),
        "driver_previous_step_state": (
            "fortran_source/ORCHIDEE/src_driver/orchideedriver.f90 lines 598-659",
            "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 1139-1164",
            "fortran_source/ORCHIDEE/src_sechiba/condveg.f90 lines 612-839 returns two-band albedo to the driver state",
        ),
        "sechiba_finalize_state": (
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_initialize lines 561-679",
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_finalize lines 1800-1923",
        ),
    }


def _previous_state_component_fields(previous_state: object, component: str) -> Mapping[str, object]:
    component_fields = getattr(previous_state, "component_fields", None)
    if component_fields is not None:
        return component_fields(component)
    return previous_state.fields_by_component.get(component, {})


def _driver_wind_z0_from_previous_state(previous_state: DriverPreviousStepStatePacket, *, fallback: float = 0.1) -> np.ndarray | float:
    """Return the driver roughness used to lower later-step wind.

    Fortran provenance: `dim2_driver.f90`, lines 1144-1180 recompute
    `for_u/for_v` with the driver-side `z0` after the first
    `intersurf_initialize_2d`; subsequent iterations use the `z0`/`z0m`
    returned by `intersurf_main_2d` through `sechiba.f90`, lines 1415-1416,
    and `intersurf.f90`, lines 647-674.
    """

    diffuco_state = _previous_state_component_fields(previous_state, "diffuco_previous_step_state")
    if "z0m" in diffuco_state:
        return diffuco_state["z0m"]
    return float(fallback)


def _packet_from_previous_step_fields(*, tstep: int, fields: dict[str, dict[str, object]]) -> DriverPreviousStepStatePacket:
    provenance = _previous_step_state_provenance()
    return DriverPreviousStepStatePacket(
        tstep=int(tstep),
        fields_by_component={key: value for key, value in fields.items() if value},
        provenance_by_component={
            key: provenance[key]
            for key, value in fields.items()
            if value
        },
    )


def _add_enerbil_local_to_previous_fields(fields: dict[str, dict[str, object]], enerbil_local: EnerbilFirstStepLocalAssembly | None) -> None:
    if enerbil_local is None:
        return
    payload = enerbil_local.payload
    for source_name, target_name in (
        ("temp_sol_new", "temp_sol"),
        ("temp_sol_new_pft", "temp_sol_pft"),
        ("qsurf", "qsurf"),
        ("evapot", "evapot"),
        ("evapot_corr", "evapot_corr"),
    ):
        if source_name in payload:
            fields["diffuco_previous_step_state"][target_name] = payload[source_name]
            if target_name in {"temp_sol", "temp_sol_pft", "qsurf"}:
                fields["enerbil_previous_step_state"][target_name] = payload[source_name]
    driver = fields.setdefault("driver_previous_step_state", {})
    for name in ("fluxsens", "vevapp", "petAcoef", "petBcoef", "peqAcoef", "peqBcoef"):
        driver[name] = payload[name]
    driver.update(
        {
            "old_zlev": payload["zlev"],
            "old_qair": payload["qair"],
            "old_eair": CP_AIR * payload["temp_air"] + GRAVITY * payload["zlev"],
            "rau_old": payload["rau"],
        }
    )


def _add_diffuco_local_to_previous_fields(
    fields: dict[str, dict[str, object]],
    diffuco_local: DiffucoPFT14FirstStepLocalEnerbilPrecallAssembly | None,
) -> None:
    if diffuco_local is None:
        return
    diffuco_payload = diffuco_local.result.payload.payload
    for name in ("gpp", "gsmean", "rveget", "rstruct", "cimean"):
        if name in diffuco_payload:
            fields["diffuco_previous_step_state"][name] = diffuco_payload[name]


def _add_diffuco_saved_controls_to_previous_fields(
    fields: dict[str, dict[str, object]],
    *,
    control_salinity: DiffucoControlSalinityAssembly | None = None,
    control_inundation: DiffucoControlInundationAssembly | None = None,
    precall: object | None = None,
) -> None:
    """Persist DIFFUCO firstcall controls with Fortran SAVE lifetime."""

    target = fields["diffuco_previous_step_state"]
    payloads: list[Mapping[str, object]] = []
    if control_salinity is not None:
        payloads.append(control_salinity.payload)
    if control_inundation is not None:
        payloads.append(control_inundation.payload)
    if precall is not None:
        payloads.append(precall if isinstance(precall, Mapping) else getattr(precall, "payload", {}))
    for payload in payloads:
        for name in ("control_salinity", "control_inudate"):
            if name in payload and name not in target:
                target[name] = payload[name]


def _add_thermosoil_to_previous_fields(fields: dict[str, dict[str, object]], therm: ThermosoilFirstStepModuleClosure | None) -> None:
    if therm is None or not therm.ok:
        return
    module = therm.module
    downstream = dict(therm.downstream_payload)
    fields["thermosoil_previous_step_state"]["ptn"] = module.profile.ptn
    for name in ("cgrnd", "dgrnd", "soilcap", "soilflx"):
        if name in downstream:
            fields["thermosoil_previous_step_state"][name] = downstream[name]
            fields["enerbil_previous_step_state"][name] = downstream[name]
    for name in ("pcapa_en", "temp_sol_beg"):
        if name in downstream:
            fields["thermosoil_previous_step_state"][name] = downstream[name]
    for name in ("stempdiag", "deeptemp_prof", "deephum_prof", "shum_ngrnd_permalong"):
        if name in downstream:
            target = {"deeptemp_prof": "tdeep", "deephum_prof": "hsdeep"}.get(name, name)
            fields["thermosoil_previous_step_state"][target] = downstream[name]
    for name in ("cgrnd_snow", "dgrnd_snow", "lambda_snow", "gtemp"):
        if name in downstream:
            fields["thermosoil_previous_step_state"][name] = downstream[name]
    if "soilcap_pft" in downstream:
        fields["enerbil_previous_step_state"]["soilcap_pft"] = downstream["soilcap_pft"]
    if "soilflx_pft" in downstream:
        fields["enerbil_previous_step_state"]["soilflx_pft"] = downstream["soilflx_pft"]


def _add_condveg_to_previous_fields(fields: dict[str, dict[str, object]], condveg_module: CondvegFirstStepModuleClosure | None) -> None:
    if condveg_module is None or not condveg_module.ok:
        return
    condveg = condveg_module.module
    if condveg is not None:
        for name in ("z0m", "z0h", "roughheight", "roughheight_pft"):
            fields["diffuco_previous_step_state"][name] = getattr(condveg, name)
        fields["enerbil_previous_step_state"]["roughness"] = condveg.roughheight
        if condveg.albedo is not None:
            fields.setdefault("driver_previous_step_state", {})["albedo"] = condveg.albedo
            fields["driver_previous_step_state"]["albedo_vis"] = condveg.albedo[:, 0]
            fields["driver_previous_step_state"]["albedo_nir"] = condveg.albedo[:, 1]
            fields["driver_previous_step_state"]["z0"] = condveg.z0m
            fields["enerbil_previous_step_state"]["albedo"] = condveg.albedo


def _add_hydrol_to_previous_fields(fields: dict[str, dict[str, object]], hydrol: HydrolFirstStepModuleClosure | None) -> None:
    if hydrol is None or not hydrol.ok:
        return
    fields["hydrol_previous_step_state"]["mc"] = hydrol.module.soil.mc
    fields["hydrol_previous_step_state"]["mcl"] = hydrol.module.soil.mcl
    fields["hydrol_previous_step_state"]["water2infilt"] = hydrol.module.soil.water2infilt
    fields["hydrol_previous_step_state"]["ae_ns"] = hydrol.module.split.ae_ns
    fields["hydrol_previous_step_state"]["qsintveg"] = hydrol.module.canop.qsintveg
    fields["diffuco_previous_step_state"]["qsintveg"] = hydrol.module.canop.qsintveg
    for name in ("humrel", "humrelv", "us", "evap_bare_lim", "evap_bare_lim_ns", "fwet_new"):
        value = getattr(hydrol.diagnostics, name, None)
        if value is not None:
            fields["hydrol_previous_step_state"][name] = value
            if name in {"evap_bare_lim", "humrel"}:
                fields["diffuco_previous_step_state"][name] = value
    if getattr(hydrol.diagnostics, "nroot", None) is not None:
        fields["hydrol_previous_step_state"]["nroot"] = hydrol.diagnostics.nroot
    if getattr(hydrol.diagnostics, "shumdiag_peat", None) is not None:
        fields["hydrol_previous_step_state"]["shumdiag_peat"] = hydrol.diagnostics.shumdiag_peat
    if getattr(hydrol.diagnostics, "drysoil_frac", None) is not None:
        fields["diffuco_previous_step_state"]["drysoil_frac"] = hydrol.diagnostics.drysoil_frac
    fields["hydrol_previous_step_state"]["drainage"] = hydrol.outputs.drainage_per_soil
    fields["hydrol_previous_step_state"]["runoff"] = hydrol.outputs.runoff_per_soil
    fields["hydrol_previous_step_state"]["soil_mc"] = hydrol.diagnostics.mc_layh_s
    fields["hydrol_previous_step_state"]["wat_flux"] = hydrol.outputs.wat_flux
    fields["hydrol_previous_step_state"]["runoff_per_soil"] = hydrol.outputs.runoff_per_soil
    fields["hydrol_previous_step_state"]["drainage_per_soil"] = hydrol.outputs.drainage_per_soil
    fields["hydrol_previous_step_state"]["runoff2peat"] = hydrol.outputs.runoff2peat
    fields["hydrol_previous_step_state"]["canopy2ground"] = hydrol.outputs.canopy2ground
    fields["hydrol_previous_step_state"]["precip2ground"] = hydrol.outputs.precip2ground
    fields["hydrol_previous_step_state"]["precip2canopy"] = hydrol.outputs.precip2canopy
    fields["hydrol_previous_step_state"]["flood_res"] = hydrol.module.flood.flood_res
    fields["hydrol_previous_step_state"]["run2peat"] = hydrol.module.soil.run2peat
    fields["hydrol_previous_step_state"]["run2man"] = hydrol.module.soil.run2man
    fields["hydrol_previous_step_state"]["wt_ab"] = hydrol.module.soil.wt_ab
    fields["hydrol_previous_step_state"]["wt_ab_tide"] = hydrol.module.soil.wt_ab_tide
    for name in ("njsc", "profil_froz", "temp_hydro", "stempdiag"):
        if name in hydrol.module_inputs:
            target = "profil_froz_hydro_ns" if name == "profil_froz" else name
            fields["hydrol_previous_step_state"][target] = hydrol.module_inputs[name]
    if hydrol.snow_state is not None:
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
        ):
            value = getattr(hydrol.snow_state, name)
            fields["hydrol_previous_step_state"][name] = value
            fields["diffuco_previous_step_state"][name] = value
        fields["enerbil_previous_step_state"]["snowdz"] = hydrol.snow_state.snowdz
    snow_step = getattr(hydrol, "snow_step", None)
    if snow_step is not None:
        for name in ("snowliq", "subsnownobio", "grndflux", "snowmelt", "temp_sol_add"):
            fields["hydrol_previous_step_state"][name] = getattr(snow_step, name)
        fields["enerbil_previous_step_state"]["temp_sol_add"] = snow_step.temp_sol_add
    elif hydrol.snow_state is not None:
        # explicitsnow.f90 lines 219-226 and 242-254 reset these fields on the
        # audited no-snow path. Persist them so later snowfall can enter the
        # full explicit-snow branch with exact previous-step state.
        snow = hydrol.snow_state.snow
        zeros_like = jnp.zeros_like if isinstance(snow, jax.core.Tracer) else np.zeros_like
        fields["hydrol_previous_step_state"]["snowliq"] = zeros_like(hydrol.snow_state.snowdz)
        fields["hydrol_previous_step_state"]["subsnownobio"] = zeros_like(hydrol.snow_state.snow_nobio)
        fields["hydrol_previous_step_state"]["grndflux"] = zeros_like(snow)
        fields["hydrol_previous_step_state"]["snowmelt"] = zeros_like(snow)
        fields["hydrol_previous_step_state"]["temp_sol_add"] = zeros_like(snow)
        fields["enerbil_previous_step_state"]["temp_sol_add"] = zeros_like(snow)
    if "soilflxresid" in hydrol.module_inputs:
        fields["hydrol_previous_step_state"]["soilflxresid"] = hydrol.module_inputs["soilflxresid"]
    elif hydrol.snow_state is not None:
        snow = hydrol.snow_state.snow
        zeros_like = jnp.zeros_like if isinstance(snow, jax.core.Tracer) else np.zeros_like
        fields["hydrol_previous_step_state"]["soilflxresid"] = zeros_like(snow)


_SECHIBA_SLOWPROC_FINALIZE_SOURCE_FIELDS = frozenset(
    SECHIBA_RESTART_TO_SOURCE_NAMES.get(name, name)
    for name in SECHIBA_RESTART_COMPONENT_FIELDS["slowproc"]
)
_SECHIBA_HALF_HOUR_CARRY_FIELDS = frozenset(
    {
        "q_sol_pot",
        "temp_sol_pot",
        "free_drain_coef",
        "zwt_force",
        "resdist",
        "vegtot_old",
        "soilalb_bg",
        "refsoc",
        "e_soil_lat",
        *_SECHIBA_SLOWPROC_FINALIZE_SOURCE_FIELDS,
    }
)


def _updated_leaf_ci(previous_leaf_ci, diffuco_local):
    boundary = diffuco_local.result.boundary
    pft_index = int(diffuco_local.kwargs["pft_index"])
    active_leaf_ci = boundary.process_chain.trans_co2.canopy.leaf_ci
    previous = jnp.asarray(previous_leaf_ci)
    if active_leaf_ci.shape != (previous.shape[0], previous.shape[2]):
        raise ValueError("PFT14 active leaf_ci must match restart (npts,nlai) axes")
    return previous.at[:, pft_index, :].set(active_leaf_ci)


def _sechiba_finalize_state_from_components(
    *,
    previous: Mapping[str, object],
    enerbil_local: EnerbilFirstStepLocalAssembly,
    diffuco_local: DiffucoPFT14FirstStepLocalEnerbilPrecallAssembly,
    thermosoil_module: ThermosoilFirstStepModuleClosure,
    condveg_module: CondvegFirstStepModuleClosure,
    hydrol_module: HydrolFirstStepModuleClosure,
    slowproc_updates: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build all 93 SECHIBA finalize inputs after one production step."""

    if not (
        thermosoil_module.ok and condveg_module.ok and hydrol_module.ok
    ):
        raise ValueError("SECHIBA finalize transition requires all physical component owners")
    e = enerbil_local
    d = diffuco_local
    h = hydrol_module
    t = thermosoil_module
    c = condveg_module
    snow = h.snow_state
    diag = h.diagnostics
    module = h.module
    downstream = t.downstream_payload
    ep = e.payload
    updates = {
        "rstruct": d.result.payload.payload["rstruct"],
        "q_cdrag_pft": d.result.boundary.drag.q_cdrag_pft,
        "leaf_ci": _updated_leaf_ci(previous["leaf_ci"], d),
        "temp_sol": ep["temp_sol_new"],
        "temp_sol_pft": ep["temp_sol_new_pft"],
        "qsurf": ep["qsurf"],
        "evapot": ep["evapot"],
        "evapot_corr": ep["evapot_corr"],
        "tsol_rad": ep["tsol_rad"],
        "vevapp": ep["vevapp"],
        "fluxlat": ep["fluxlat"],
        "fluxsens": ep["fluxsens"],
        "mc": module.soil.mc,
        "mcl": module.soil.mcl,
        "us": diag.us,
        "water2infilt": module.soil.water2infilt,
        "ae_ns": diag.ae_ns,
        "vegstress": diag.vegstress,
        "snow": snow.snow,
        "snow_age": snow.snow_age,
        "snow_nobio": snow.snow_nobio,
        "snow_nobio_age": snow.snow_nobio_age,
        "qsintveg": module.canop.qsintveg,
        "evap_bare_lim_ns": diag.evap_bare_lim_ns,
        "evap_bare_lim": diag.evap_bare_lim,
        "drysoil_frac": diag.drysoil_frac,
        "humrel": diag.humrel,
        "fwet_out": jnp.zeros_like(snow.snow),
        "run2peat": module.soil.run2peat,
        "wt_ab": module.soil.wt_ab,
        "wtp": diag.wtp,
        "fwet_new": diag.fwet_new if diag.fwet_new is not None else previous["fwet_new"],
        "liqwt_ratio": diag.liqwt_ratio,
        "wt_ab_tide": module.soil.wt_ab_tide,
        "run2man": module.soil.run2man,
        "tot_watveg_beg": jnp.sum(module.canop.qsintveg, axis=1),
        "tot_watsoil_beg": diag.humtot,
        "snow_beg": snow.snow + jnp.sum(snow.snow_nobio, axis=1),
        "snowrho": snow.snowrho,
        "snowtemp": snow.snowtemp,
        "snowdz": snow.snowdz,
        "snowheat": snow.snowheat,
        "snowgrain": snow.snowgrain,
        "z0m": c.module.z0m,
        "z0h": c.module.z0h,
        "roughheight": c.module.roughheight,
        "roughheight_pft": c.module.roughheight_pft,
        "ptn": t.module.profile.ptn,
        "shum_ngrnd_permalong": downstream["shum_ngrnd_permalong"],
        "shum_ngrnd_perma": t.module.humlev.shum_ngrnd_perma,
        "cgrnd": downstream["cgrnd"],
        "dgrnd": downstream["dgrnd"],
        "gtemp": downstream["gtemp"],
        "soilcap": downstream["soilcap"],
        "soilcap_pft": downstream["soilcap_pft"],
        "soilflx": downstream["soilflx"],
        "soilflx_pft": downstream["soilflx_pft"],
        "cgrnd_snow": downstream["cgrnd_snow"],
        "dgrnd_snow": downstream["dgrnd_snow"],
        "lambda_snow": downstream["lambda_snow"],
    }
    pottemp = e.step.pottemp
    carry = set(_SECHIBA_HALF_HOUR_CARRY_FIELDS)
    if pottemp is not None:
        updates["q_sol_pot"] = pottemp.q_sol_pot
        updates["temp_sol_pot"] = pottemp.temp_sol_pot
        carry -= {"q_sol_pot", "temp_sol_pot"}
    if slowproc_updates is not None:
        for name in _SECHIBA_SLOWPROC_FINALIZE_SOURCE_FIELDS:
            if name in slowproc_updates:
                updates[name] = slowproc_updates[name]
                carry.discard(name)
    update_provenance = {
        name: (
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 893-1128",
        )
        for name in updates
    }
    carry_provenance = {
        name: (
            "fortran_source/ORCHIDEE/src_sechiba component SAVE/static state; owner audited in sechiba_restart_field_owners.json",
        )
        for name in carry
    }
    return dict(
        transition_sechiba_finalize_source_state(
            previous,
            updates=updates,
            carried_fields=tuple(carry),
            update_provenance=update_provenance,
            carry_provenance=carry_provenance,
        ).fields
    )


def _sechiba_finalize_state_after_slowproc(
    previous: Mapping[str, object], slowproc: Mapping[str, object]
) -> dict[str, object]:
    updates = {
        name: slowproc[name]
        for name in _SECHIBA_SLOWPROC_FINALIZE_SOURCE_FIELDS
        if name in slowproc
    }
    carried = tuple(sorted(set(previous) - set(updates)))
    return dict(
        transition_sechiba_finalize_source_state(
            previous,
            updates=updates,
            carried_fields=carried,
            update_provenance={
                name: (
                    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 973-1128",
                )
                for name in updates
            },
            carry_provenance={
                name: (
                    "day-end handoff after the last SECHIBA half-hour; no later owner writes this field",
                )
                for name in carried
            },
        ).fields
    )


def driver_previous_step_state_from_timestep_scaffold(
    scaffold: DriverTimestepScaffold,
    *,
    ok_laidev: tuple[bool, ...] | None = None,
) -> DriverPreviousStepStatePacket:
    """Extract only source-backed next-step state produced by a timestep.

    Fortran provenance is attached per component. The helper intentionally
    leaves absent any field not directly available from the completed local
    module results or an audited Fortran state migration.
    """

    fields = _empty_previous_step_state_fields()
    _add_enerbil_local_to_previous_fields(fields, scaffold.enerbil_first_step_local)
    _add_diffuco_local_to_previous_fields(fields, scaffold.diffuco_first_step_local_enerbil_precall)
    _add_diffuco_saved_controls_to_previous_fields(
        fields,
        control_salinity=scaffold.diffuco_control_salinity,
        control_inundation=scaffold.diffuco_control_assembly,
        precall=scaffold.diffuco_first_step_precall,
    )
    _add_thermosoil_to_previous_fields(fields, scaffold.thermosoil_first_step_module)
    _add_condveg_to_previous_fields(fields, scaffold.condveg_first_step_module)
    _add_hydrol_to_previous_fields(fields, scaffold.hydrol_first_step_module)

    if scaffold.first_step_restart_state is not None:
        slowproc = scaffold.first_step_restart_state.slowproc
        stomate = scaffold.first_step_restart_state.stomate
        if scaffold.diffuco_first_step_precall is not None and "height" in scaffold.diffuco_first_step_precall.payload:
            fields["diffuco_previous_step_state"]["height"] = scaffold.diffuco_first_step_precall.payload["height"]
        fields["diffuco_previous_step_state"]["soilalbedo_bg"] = scaffold.first_step_restart_state.diffuco_precall.soilalbedo_bg
        for name in ("free_drain_coef", "zwt_force", "wt_ab", "wt_ab_tide", "njsc", "fwet_new"):
            value = getattr(scaffold.first_step_restart_state.hydrol, name, None)
            if value is not None and name not in fields["hydrol_previous_step_state"]:
                fields["hydrol_previous_step_state"][name] = value
        if (
            scaffold.first_step_restart_state.thermosoil.refsoc is not None
            and "refSOC" not in fields["thermosoil_previous_step_state"]
        ):
            fields["thermosoil_previous_step_state"]["refSOC"] = scaffold.first_step_restart_state.thermosoil.refsoc
        for name in ("veget", "veget_max", "lai", "frac_nobio", "height", "frac_age"):
            fields["slowproc_stomate_previous_step_state"][name] = getattr(slowproc, name)
            if name != "frac_age":
                fields["diffuco_previous_step_state"][name] = getattr(slowproc, name)
        for name in (
            "assim_param",
            "altmax",
            "fpeat",
            "biomass",
            "ind",
            "veget_lastlight",
            "need_adjacent",
            "litter_above",
            "litter_below",
            "litterpart",
            "dead_leaves",
            "carbon",
            "litter",
            "lignin_struc",
            "prod10",
            "prod100",
            "flux10",
            "flux100",
            "fuel_1hr",
            "fuel_10hr",
            "fuel_100hr",
            "fuel_1000hr",
            "carbon_32l",
            "DOC",
            "lignin_struc_above",
            "lignin_struc_below",
            "deepC_a",
            "deepC_s",
            "deepC_p",
            "soilc_total",
            "thawed_humidity",
            "depth_organic_soil",
            "resp_maint_part",
        ):
            if hasattr(stomate, name):
                fields["slowproc_stomate_previous_step_state"][name] = getattr(stomate, name)
        daily = scaffold.stomate_first_step_daily_accumulation
        if daily is not None and daily.ok:
            fields["slowproc_stomate_previous_step_state"]["daily_accumulators"] = daily.fields
        if scaffold.first_step_restart_state.stomate_input is not None:
            season = scaffold.first_step_restart_state.stomate_readstart.season_state
            fields["slowproc_stomate_previous_step_state"]["t2m_longterm"] = season.t2m_longterm
        readstart = scaffold.first_step_restart_state.stomate_readstart
        fields["slowproc_stomate_previous_step_state"].update(
            {
                "dt_days_read": float(readstart.season_state.dt_days_read),
                "resp_hetero": readstart.remainder_state.resp_hetero,
                "tsurf_year": readstart.remainder_state.tsurf_year,
                "deepC_peat": readstart.remainder_state.deepC_peat,
                "depth_deepsoil": readstart.remainder_state.depth_deepsoil,
                "t2m_14": readstart.remainder_state.t2m_14,
            }
        )
        fixed_carry = stomate_fixed_path_carry_state(
            entry_state=readstart.entry_state,
            season_state=readstart.season_state,
            gas_state=readstart.gas_state,
            remainder_state=readstart.remainder_state,
            fire_disable=True,
            ok_pc=False,
            ch4_calcul=False,
            ok_leak=True,
            ok_peat=False,
            peat_occur=False,
            dyn_peat=False,
            land_cover_change=False,
            enable_grazing=False,
            spinup_analytic=False,
            stomate_ok_dgvm=False,
            lpj_gap_const_mort=True,
        )
        fields["slowproc_stomate_previous_step_state"].update(fixed_carry.fields)

        initial_finalize = sechiba_finalize_source_state_from_restart(
            scaffold.first_step_restart_state.sechiba_restart_state
        )
        fields["sechiba_finalize_state"] = _sechiba_finalize_state_from_components(
            previous=initial_finalize,
            enerbil_local=scaffold.enerbil_first_step_local,
            diffuco_local=scaffold.diffuco_first_step_local_enerbil_precall,
            thermosoil_module=scaffold.thermosoil_first_step_module,
            condveg_module=scaffold.condveg_first_step_module,
            hydrol_module=scaffold.hydrol_first_step_module,
        )

        _write_maintenance_lai_to_previous_fields(
            fields,
            fields["slowproc_stomate_previous_step_state"],
            ok_laidev,
        )

    return _packet_from_previous_step_fields(tstep=scaffold.step.tstep, fields=fields)


def driver_previous_step_state_from_next_step_hydrol_scaffold(
    scaffold: DriverNextStepHydrolPrecallScaffold,
    *,
    ok_laidev: tuple[bool, ...] | None = None,
) -> DriverPreviousStepStatePacket:
    """Extract source-backed next-step state after a completed later SECHIBA step."""

    if not scaffold.ready_for_next_state:
        raise ValueError("next-step HYDROL scaffold must close HYDROL, CONDVEG, and THERMOSOIL before state extraction")
    fields = _empty_previous_step_state_fields()
    _add_enerbil_local_to_previous_fields(fields, scaffold.enerbil.enerbil_local)
    _add_diffuco_local_to_previous_fields(fields, scaffold.enerbil.diffuco_local_enerbil_precall)
    _add_diffuco_saved_controls_to_previous_fields(
        fields,
        precall=getattr(scaffold.enerbil.diffuco_local_enerbil_precall, "kwargs", None),
    )
    _add_thermosoil_to_previous_fields(fields, scaffold.thermosoil_module)
    _add_condveg_to_previous_fields(fields, scaffold.condveg_module)
    _add_hydrol_to_previous_fields(fields, scaffold.hydrol_module)
    prior_finalize = scaffold.enerbil.previous_state.fields_by_component.get(
        "sechiba_finalize_state"
    )
    if prior_finalize:
        fields["sechiba_finalize_state"] = _sechiba_finalize_state_from_components(
            previous=prior_finalize,
            enerbil_local=scaffold.enerbil.enerbil_local,
            diffuco_local=scaffold.enerbil.diffuco_local_enerbil_precall,
            thermosoil_module=scaffold.thermosoil_module,
            condveg_module=scaffold.condveg_module,
            hydrol_module=scaffold.hydrol_module,
        )
    prior_slowproc = scaffold.enerbil.previous_state.fields_by_component["slowproc_stomate_previous_step_state"]
    for name in (
        "veget",
        "veget_max",
        "lai",
        "frac_nobio",
        "height",
        "frac_age",
        "assim_param",
        "altmax",
        "fpeat",
        "pft_present",
        "veget_lastlight",
        "need_adjacent",
        "everywhere",
        "when_growthinit",
        "biomass",
        "leaf_frac",
        "leaf_age",
        "age",
        "sla_calc",
        "ind",
        "cn_ind",
        "co2_to_bm",
        "adapted",
        "regenerate",
        "npp_longterm",
        "turnover_daily",
        "bm_to_litter",
        "senescence",
        "turnover_time",
        "rip_time",
        "gpp_daily",
        "npp_daily",
        "resp_maint",
        "resp_growth",
        "litter_above",
        "litter_below",
        "litterpart",
        "dead_leaves",
        "carbon",
        "litter",
        "lignin_struc",
        "prod10",
        "prod100",
        "flux10",
        "flux100",
        "fuel_1hr",
        "fuel_10hr",
        "fuel_100hr",
        "fuel_1000hr",
        "carbon_32l",
        "DOC",
        "interception_storage",
        "carb_mass_total",
        "soiltile",
        "totfrac_nobio",
        "tot_bare_soil",
        "lignin_struc_above",
        "lignin_struc_below",
        "deepC_a",
        "deepC_s",
        "deepC_p",
        "soilc_total",
        "thawed_humidity",
        "depth_organic_soil",
        "resp_maint_part",
        "daily_accumulators",
        "t2m_longterm",
        "temp_growth",
        "dt_days_read",
        "resp_hetero",
        "tsurf_year",
        "deepC_peat",
        "depth_deepsoil",
        "t2m_14",
    ):
        if name in prior_slowproc:
            fields["slowproc_stomate_previous_step_state"][name] = prior_slowproc[name]
    for name in ("veget", "veget_max", "lai", "frac_nobio", "height"):
        if name in prior_slowproc:
            fields["diffuco_previous_step_state"][name] = prior_slowproc[name]
    _write_maintenance_lai_to_previous_fields(fields, prior_slowproc, ok_laidev)
    prior_therm = scaffold.enerbil.previous_state.fields_by_component.get("thermosoil_previous_step_state", {})
    for name in ("refSOC", "e_soil_lat"):
        if name in prior_therm:
            fields["thermosoil_previous_step_state"][name] = prior_therm[name]
    prior_hydrol = scaffold.enerbil.previous_state.fields_by_component.get("hydrol_previous_step_state", {})
    for name in ("free_drain_coef", "zwt_force", "njsc", "fwet_new"):
        if name in prior_hydrol:
            fields["hydrol_previous_step_state"][name] = prior_hydrol[name]
    prior_diffuco = scaffold.enerbil.previous_state.fields_by_component.get("diffuco_previous_step_state", {})
    if "soilalbedo_bg" in prior_diffuco:
        fields["diffuco_previous_step_state"]["soilalbedo_bg"] = prior_diffuco["soilalbedo_bg"]
    if "height" in prior_diffuco:
        fields["diffuco_previous_step_state"]["height"] = prior_diffuco["height"]
    for name in ("control_salinity", "control_inudate"):
        if name in prior_diffuco:
            fields["diffuco_previous_step_state"][name] = prior_diffuco[name]
    prior_driver = scaffold.enerbil.previous_state.fields_by_component.get("driver_previous_step_state", {})
    if "albedo" in prior_driver and "albedo" not in fields.get("driver_previous_step_state", {}):
        fields.setdefault("driver_previous_step_state", {})["albedo"] = prior_driver["albedo"]
    return _packet_from_previous_step_fields(tstep=scaffold.tstep, fields=fields)


def _diffuco_restart_payload_from_state_fields(
    *,
    diffuco_state: Mapping[str, object],
    hydrol_state: Mapping[str, object],
    slowproc_state: Mapping[str, object],
) -> tuple[dict[str, object], tuple[str, ...], tuple[str, ...]]:
    """Map audited previous-step fields onto DIFFUCO pre-call names."""

    payload: dict[str, object] = {}
    used: list[str] = []
    sources = (diffuco_state, hydrol_state, slowproc_state)
    for name in DIFFUCO_FIRST_STEP_PRECALL_RESTART_FIELDS:
        for source in sources:
            if name in source:
                payload[name] = source[name]
                used.append(name)
                break
    missing = tuple(field for field in DIFFUCO_FIRST_STEP_PRECALL_RESTART_FIELDS if field not in payload)
    return payload, tuple(dict.fromkeys(used)), missing


def _diffuco_restart_payload_from_previous_state(
    previous_state: DriverPreviousStepStatePacket,
) -> tuple[dict[str, object], tuple[str, ...], tuple[str, ...]]:
    return _diffuco_restart_payload_from_state_fields(
        diffuco_state=_previous_state_component_fields(previous_state, "diffuco_previous_step_state"),
        hydrol_state=_previous_state_component_fields(previous_state, "hydrol_previous_step_state"),
        slowproc_state=_previous_state_component_fields(previous_state, "slowproc_stomate_previous_step_state"),
    )


def _paper_pft14_control_inundation_from_prepared_context(
    context: Paper1961PreparedDriverContext,
    *,
    tide_height,
    biomass,
) -> DiffucoControlInundationAssembly:
    """Reuse prepared source-backed static inundation inputs for DIFFUCO.

    Fortran provenance is inherited from
    ``assemble_pft14_control_inundation_first_step``; only dynamic same-step
    ``tide_height`` and previous-STOMATE ``biomass`` are supplied here.
    """

    static = context.diffuco_inundation_static
    payload = dict(static.payload)
    missing = []
    if tide_height is None:
        missing.append("tide_height")
    else:
        payload["tide_height"] = tide_height
    if biomass is None:
        missing.append("biomass")
    else:
        payload["biomass"] = biomass
    result = None
    inputs = None
    source_inputs = [*static.source_inputs]
    if not missing:
        inputs = ControlInundateInputs(
            z_soil=payload["z_soil"],
            rprof=payload["rprof"],
            tide_height=tide_height,
            biomass=biomass,
            pft_index=_paper_mangrove_pft_index(context.run_scalars),
        )
        result = mangrove_control_inundation(
            inputs,
            control_inudate_min=context.control_inudate_min,
            agb_agr_ven_all_st=context.agb_agr_ven_all_st,
            agb_agr_ven_all_pn=context.agb_agr_ven_all_pn,
            h_agr_max_st=context.h_agr_max_st,
            h_agr_max_pn=context.h_agr_max_pn,
        )
        payload["control_inudate"] = result.control_inudate
        source_inputs.extend(("tide_height", "biomass", "control_inudate"))
    return DiffucoControlInundationAssembly(
        inputs=inputs,
        result=result,
        payload=payload,
        missing_inputs=tuple(dict.fromkeys((*static.missing_inputs, *missing))),
        source_inputs=tuple(dict.fromkeys(source_inputs)),
        provenance=static.provenance,
        notes=static.notes,
    )


def _paper_later_day_slowproc_derivvar_payload(
    slowproc_state: Mapping[str, object],
    *,
    vcmax_fix,
    height_presc,
    qsintcst,
):
    """Build DIFFUCO slowproc-derived fields after a STOMATE day boundary.

    Fortran provenance: with ``ok_stomate`` active, ``slowproc_main`` lines
    1106-1113 recompute only ``qsintmax`` from current ``veget``/``lai`` and
    do not call ``slowproc_derivvar``. Dynamic ``assim_param`` is updated by
    ``stomate.f90`` lines 4567-4569, and ``temp_growth`` by line 5049.
    """

    if not {"veget", "lai"} <= slowproc_state.keys():
        return None, {}
    derivvar = slowproc_derivvar_explicit(
        veget=slowproc_state["veget"],
        lai=slowproc_state["lai"],
        vcmax_fix=vcmax_fix,
        height_presc=height_presc,
        qsintcst=qsintcst,
    )
    payload = {
        "qsintmax": derivvar.qsintmax,
        "assim_param": slowproc_state.get("assim_param", derivvar.assim_param),
        "height": slowproc_state.get("height", derivvar.height),
        "temp_growth": slowproc_state.get("temp_growth", derivvar.temp_growth),
    }
    merged = SimpleNamespace(
        qsintmax=payload["qsintmax"],
        deadleaf_cover=derivvar.deadleaf_cover,
        assim_param=payload["assim_param"],
        height=payload["height"],
        temp_growth=payload["temp_growth"],
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 1106-1113",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 4567-4569 and 5049",
        ),
    )
    return merged, payload


def _paper_diffuco_saved_control_assemblies_from_previous_state(
    previous_state: DriverPreviousStepStatePacket,
) -> tuple[DiffucoControlSalinityAssembly, DiffucoControlInundationAssembly]:
    """Return DIFFUCO firstcall controls persisted by Fortran SAVE semantics."""

    diffuco_state = previous_state.fields_by_component.get("diffuco_previous_step_state", {})
    control_salinity = diffuco_state.get("control_salinity")
    control_inudate = diffuco_state.get("control_inudate")
    provenance = (
        "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90 module SAVE variables control_salinity/control_inudate lines 81-86",
        "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_main lines 445-622 computes them only when firstcall_diffuco is true",
        "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_main lines 622 and 2839-2840 reuses saved controls after firstcall_diffuco is false",
    )
    salinity_payload = {}
    salinity_missing = []
    if control_salinity is None:
        salinity_missing.append("control_salinity")
    else:
        salinity_payload["control_salinity"] = control_salinity
    inundation_payload = {}
    inundation_missing = []
    if control_inudate is None:
        inundation_missing.append("control_inudate")
    else:
        inundation_payload["control_inudate"] = control_inudate
    return (
        DiffucoControlSalinityAssembly(
            payload=salinity_payload,
            missing_inputs=tuple(salinity_missing),
            source_inputs=tuple(salinity_payload),
            provenance=provenance,
            notes=("Later DIFFUCO calls must reuse firstcall_diffuco SAVE state; they must not recompute from later-day biomass.",),
        ),
        DiffucoControlInundationAssembly(
            inputs=None,
            result=None,
            payload=inundation_payload,
            missing_inputs=tuple(inundation_missing),
            source_inputs=tuple(inundation_payload),
            provenance=provenance,
            notes=("Later DIFFUCO calls must reuse firstcall_diffuco SAVE state; they must not recompute from later-day biomass.",),
        ),
    )


def _paper_diffuco_day_static_cache_from_previous_state(
    context: Paper1961PreparedDriverContext,
    previous_state: DriverPreviousStepStatePacket,
    *,
    salinity,
    tide_height,
) -> DriverDiffucoDayStaticCache:
    """Build DIFFUCO day-static control inputs from the day's initial state."""

    slowproc_state = _previous_state_component_fields(previous_state, "slowproc_stomate_previous_step_state")
    biomass = slowproc_state.get("biomass")
    if biomass is not None:
        nvm = int(biomass.shape[1])
    elif "veget" in slowproc_state:
        nvm = int(slowproc_state["veget"].shape[1])
    else:
        nvm = context.nvm
    del salinity, tide_height, biomass
    control_salinity, control_inundation = _paper_diffuco_saved_control_assemblies_from_previous_state(previous_state)
    slowproc_derivvar = None
    slowproc_derivvar_payload: dict[str, object] = {}
    slowproc_derivvar, slowproc_derivvar_payload = _paper_later_day_slowproc_derivvar_payload(
        slowproc_state,
        vcmax_fix=_validated_prepared_pft_axis(
            context.vcmax_fix, context=context, nvm=nvm, name="VCMAX_FIX"
        ),
        height_presc=_validated_prepared_pft_axis(
            context.slowproc_height, context=context, nvm=nvm, name="SLOWPROC_HEIGHT"
        ),
        qsintcst=context.sechiba_qsint,
    )
    return DriverDiffucoDayStaticCache(
        nvm=nvm,
        control_salinity=control_salinity,
        control_inundation=control_inundation,
        slowproc_derivvar=slowproc_derivvar,
        slowproc_derivvar_payload=slowproc_derivvar_payload,
    )


def _paper_diffuco_day_static_cache_from_firstcall_inputs(
    context: Paper1961PreparedDriverContext,
    previous_state: DriverPreviousStepStatePacket,
    *,
    salinity,
    tide_height,
) -> DriverDiffucoDayStaticCache:
    """Build DIFFUCO SAVE-state controls for a fresh executable firstcall.

    Fortran yearly scripts restart the executable for the next forcing year.
    That resets ``firstcall_diffuco`` even though SECHIBA/STOMATE prognostic
    restart state is carried forward, so the first step of the new year must
    recompute salinity/inundation controls from the current forcing boundary.
    """

    slowproc_state = _previous_state_component_fields(previous_state, "slowproc_stomate_previous_step_state")
    biomass = slowproc_state.get("biomass")
    if biomass is not None:
        nvm = int(biomass.shape[1])
    elif "veget" in slowproc_state:
        nvm = int(slowproc_state["veget"].shape[1])
    else:
        nvm = context.nvm
    control_salinity = assemble_pft14_control_salinity_first_step(
        salinity=salinity,
        control_salinity_min=context.control_salinity_min,
    )
    control_inundation = _paper_pft14_control_inundation_from_prepared_context(
        context,
        tide_height=tide_height,
        biomass=biomass,
    )
    slowproc_derivvar, slowproc_derivvar_payload = _paper_later_day_slowproc_derivvar_payload(
        slowproc_state,
        vcmax_fix=_validated_prepared_pft_axis(
            context.vcmax_fix, context=context, nvm=nvm, name="VCMAX_FIX"
        ),
        height_presc=_validated_prepared_pft_axis(
            context.slowproc_height, context=context, nvm=nvm, name="SLOWPROC_HEIGHT"
        ),
        qsintcst=context.sechiba_qsint,
    )
    return DriverDiffucoDayStaticCache(
        nvm=nvm,
        control_salinity=control_salinity,
        control_inundation=control_inundation,
        slowproc_derivvar=slowproc_derivvar,
        slowproc_derivvar_payload=slowproc_derivvar_payload,
        provenance=(
            "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_main lines 445-622 firstcall_diffuco recomputes control_salinity/control_inudate",
            "fortran_run_scripts/paper_250919/Job0_bio lines 315-322 restarts the executable for the next forcing year",
        ),
    )


def paper_1961_next_step_diffuco_precall_scaffold(
    config_path: str | Path,
    *,
    previous_state: DriverPreviousStepStatePacket,
    year: int = 1961,
    tstep: int = 1,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    used_run_def_path: str | Path | None = None,
    prepared_context: Paper1961PreparedDriverContext | None = None,
    day_static_cache: DriverDiffucoDayStaticCache | None = None,
) -> DriverNextStepDiffucoPrecallScaffold:
    """Assemble a later-step DIFFUCO pre-call only from prior model state.

    The dynamic state payload comes from ``previous_state``. The initial
    restart is not used as a current-state alias for ``tstep>0``. A rebased
    yearly restart state may enter here at ``tstep=0`` with
    ``previous_state.tstep == -1``; this mirrors the yearly executable restart
    while still rejecting cold-start first-step use.
    """

    if int(tstep) < 0:
        raise ValueError("next-step DIFFUCO scaffold requires tstep >= 0")
    if previous_state.tstep != int(tstep) - 1:
        raise ValueError("previous_state.tstep must be exactly tstep - 1")

    context = prepared_context or prepare_paper_1961_driver_context(config_path, used_run_def_path=used_run_def_path)
    if prepared_context is not None and fixed_format_trace_dir is None and static_trace_fields is None:
        step = _paper_1961_step_bundle_from_context(prepared_context, year=year, tstep=tstep)
    else:
        step = load_paper_1961_step_bundle(
            config_path,
            year=year,
            tstep=tstep,
            run_def_path=context.run_def_path,
            reference_run_dir=context.reference_run_dir,
            fixed_format_trace_dir=fixed_format_trace_dir,
            static_trace_fields=static_trace_fields,
            domain_override=context.domain_override,
        )
    payload = build_intersurf_first_step_payload(
        step,
        driver_z0_for_wind=_driver_wind_z0_from_previous_state(previous_state),
        dt_sechiba=context.dt_sechiba,
    )
    restart_payload, used_previous, missing_previous = _diffuco_restart_payload_from_previous_state(previous_state)
    slowproc_state = _previous_state_component_fields(previous_state, "slowproc_stomate_previous_step_state")
    biomass = slowproc_state.get("biomass")
    nvm = None
    if biomass is not None:
        nvm = int(biomass.shape[1])
    elif "veget" in slowproc_state:
        nvm = int(slowproc_state["veget"].shape[1])
    if nvm is None:
        nvm = max(
            int(np.asarray(fields["veget"]).shape[1])
            for fields in previous_state.fields_by_component.values()
            if "veget" in fields
        )
    if day_static_cache is not None:
        diffuco_control_salinity = day_static_cache.control_salinity
        diffuco_control_inundation = day_static_cache.control_inundation
        slowproc_derivvar = day_static_cache.slowproc_derivvar
        slowproc_derivvar_payload = dict(day_static_cache.slowproc_derivvar_payload)
    else:
        pft_to_mtc = _validated_prepared_pft_axis(
            context.pft_to_mtc, context=context, nvm=nvm, name="PFT_TO_MTC"
        )
        hydrol_humcste = _validated_prepared_pft_axis(
            context.hydrol_humcste, context=context, nvm=nvm, name="HYDROL_HUMCSTE"
        )
        if int(tstep) == 0:
            diffuco_control_salinity = assemble_pft14_control_salinity_first_step(
                salinity=payload.salinity,
                control_salinity_min=context.control_salinity_min,
            )
            if prepared_context is not None:
                diffuco_control_inundation = _paper_pft14_control_inundation_from_prepared_context(
                    context,
                    tide_height=payload.tide_height,
                    biomass=biomass,
                )
            else:
                diffuco_control_inundation = assemble_pft14_control_inundation_first_step(
                    depth_max_h=context.hydrol_depth_max_h,
                    depth_max_t=context.depth_max_t,
                    depth_topthickness=context.depth_topthickness,
                    depth_cstthickness=context.depth_cstthickness,
                    depth_geom=context.depth_geom,
                    ratio_geom_below=context.ratio_geom_below,
                    pft_to_mtc=pft_to_mtc,
                    hydrol_humcste=hydrol_humcste,
                    tide_height=payload.tide_height,
                    biomass=biomass,
                    control_inudate_min=context.control_inudate_min,
                    agb_agr_ven_all_st=context.agb_agr_ven_all_st,
                    agb_agr_ven_all_pn=context.agb_agr_ven_all_pn,
                    h_agr_max_st=context.h_agr_max_st,
                    h_agr_max_pn=context.h_agr_max_pn,
                    pft_index=_paper_mangrove_pft_index(context.run_scalars),
                )
        else:
            (
                diffuco_control_salinity,
                diffuco_control_inundation,
            ) = _paper_diffuco_saved_control_assemblies_from_previous_state(
                previous_state
            )
        slowproc_derivvar_payload = {}
        slowproc_derivvar = None
        slowproc_derivvar, slowproc_derivvar_payload = _paper_later_day_slowproc_derivvar_payload(
            slowproc_state,
            vcmax_fix=_validated_prepared_pft_axis(
                context.vcmax_fix, context=context, nvm=nvm, name="VCMAX_FIX"
            ),
            height_presc=_validated_prepared_pft_axis(
                context.slowproc_height, context=context, nvm=nvm, name="SLOWPROC_HEIGHT"
            ),
            qsintcst=context.sechiba_qsint,
        )
    diffuco_precall = assemble_diffuco_first_step_precall_payload(
        driver_or_intersurf_payload={
            "u": payload.u,
            "v": payload.v,
            "zlev": payload.zlev,
            "temp_air": payload.temp_air,
            "qair": payload.qair,
            "pb": payload.pb,
            "swdown": payload.swdown,
            "ccanopy": payload.ccanopy,
            "precip_rain": payload.precip_rain,
            "lalo": payload.lalo,
            "neighbours": payload.neighbours,
            "resolution": payload.resolution,
        },
        restart_payload=restart_payload,
        driver_albedo_payload=None,
        slowproc_derivvar_payload=slowproc_derivvar_payload,
        diffuco_control_salinity=diffuco_control_salinity,
        diffuco_control_inundation=diffuco_control_inundation,
        ok_explicitsnow=context.ok_explicitsnow,
        river_routing=context.river_routing,
        nbp_glo=payload.kjpindex,
    )
    static_passthrough = tuple(
        field
        for field in (
            "salinity",
            "tide_height",
            "control_salinity",
            "control_inudate",
        )
        if field in diffuco_precall.control_inputs
    )
    return DriverNextStepDiffucoPrecallScaffold(
        tstep=int(tstep),
        step=step,
        payload=payload,
        previous_state=previous_state,
        diffuco_precall=diffuco_precall,
        slowproc_derivvar=slowproc_derivvar if slowproc_derivvar_payload else None,
        used_previous_state_fields=used_previous,
        static_passthrough_fields=static_passthrough,
        missing_previous_state_fields=missing_previous,
    )


def _enerbil_state_payload_from_fields(
    fields: Mapping[str, object],
) -> tuple[dict[str, object], tuple[str, ...], tuple[str, ...]]:
    required = ("temp_sol", "temp_sol_pft", "soilcap", "soilcap_pft", "soilflx", "soilflx_pft")
    payload = {name: fields[name] for name in required if name in fields}
    missing = tuple(name for name in required if name not in payload)
    return payload, tuple(payload.keys()), missing


def _enerbil_state_payload_from_previous_state(
    previous_state: DriverPreviousStepStatePacket,
) -> tuple[dict[str, object], tuple[str, ...], tuple[str, ...]]:
    return _enerbil_state_payload_from_fields(
        _previous_state_component_fields(previous_state, "enerbil_previous_step_state")
    )


def paper_1961_next_step_enerbil_precall_scaffold(
    config_path: str | Path,
    *,
    previous_state: DriverPreviousStepStatePacket,
    year: int = 1961,
    tstep: int = 1,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    used_run_def_path: str | Path | None = None,
    prepared_context: Paper1961PreparedDriverContext | None = None,
    module_jit: bool = True,
    diffuco_local_jit: bool = True,
    diffuco_day_static_cache: DriverDiffucoDayStaticCache | None = None,
) -> DriverNextStepEnerbilPrecallScaffold:
    """Assemble later-step ENERBIL pre-call from prior state and DIFFUCO state."""

    diffuco = paper_1961_next_step_diffuco_precall_scaffold(
        config_path,
        previous_state=previous_state,
        year=year,
        tstep=tstep,
        fixed_format_trace_dir=fixed_format_trace_dir,
        static_trace_fields=static_trace_fields,
        used_run_def_path=used_run_def_path,
        prepared_context=prepared_context,
        day_static_cache=diffuco_day_static_cache,
    )
    thermal_payload, used_previous, missing_previous = _enerbil_state_payload_from_previous_state(previous_state)
    driver_albedo_payload = {}
    driver_fields = _previous_state_component_fields(previous_state, "driver_previous_step_state")
    state_fields = _previous_state_component_fields(previous_state, "enerbil_previous_step_state")
    if "albedo" in driver_fields:
        driver_albedo_payload["albedo"] = driver_fields["albedo"]
    elif "albedo" in state_fields:
        driver_albedo_payload["albedo"] = state_fields["albedo"]
    context = prepared_context or prepare_paper_1961_driver_context(config_path, used_run_def_path=used_run_def_path)
    run_def_values = context.run_def_values
    diffuco_local = None
    after_diffuco_payload = diffuco.diffuco_precall.payload
    if diffuco.ready_for_local_diffuco:
        diffuco_local = run_pft14_local_enerbil_precall_from_first_step_precall(
            diffuco.diffuco_precall,
            run_def_values=run_def_values,
            pft_index=_paper_mangrove_pft_index(context.run_scalars),
            static_kwargs=context.diffuco_pft14_static_kwargs if prepared_context is not None else None,
            use_jit=diffuco_local_jit,
        )
        after_diffuco_payload = diffuco_local.result.payload.payload
    enerbil_precall = assemble_enerbil_first_step_precall_payload(
        driver_or_intersurf_payload=_enerbil_driver_coverage_payload(diffuco.payload),
        driver_albedo_payload=driver_albedo_payload,
        soil_thermal_restart_payload=thermal_payload,
        after_diffuco_payload=after_diffuco_payload,
        condveg_impaze=False,
        non_watchout_driver_executed=True,
    )
    combined_missing = tuple(dict.fromkeys((*diffuco.missing_previous_state_fields, *missing_previous)))
    enerbil_local = None
    if enerbil_precall.ok and not combined_missing:
        nvm = int(np.asarray(enerbil_precall.payload["veget_max"]).shape[1])
        enerbil_local = run_enerbil_first_step_local_from_precall(
            enerbil_precall,
            ok_laidev=_validated_prepared_pft_axis(
                context.ok_laidev, context=context, nvm=nvm, name="OK_LAIDEV"
            ),
            dt_sechiba=context.dt_sechiba,
            ok_explicitsnow=context.ok_explicitsnow,
            min_wind=context.min_wind,
            use_jit=bool(module_jit) and prepared_context is not None,
        )
    return DriverNextStepEnerbilPrecallScaffold(
        tstep=int(tstep),
        step=diffuco.step,
        payload=diffuco.payload,
        previous_state=previous_state,
        diffuco_precall=diffuco.diffuco_precall,
        slowproc_derivvar=diffuco.slowproc_derivvar,
        diffuco_local_enerbil_precall=diffuco_local,
        enerbil_precall=enerbil_precall,
        enerbil_local=enerbil_local,
        used_previous_state_fields=tuple(dict.fromkeys((*diffuco.used_previous_state_fields, *used_previous))),
        missing_previous_state_fields=combined_missing,
    )


def _hydrol_state_namespace_from_fields(fields: Mapping[str, object]) -> tuple[SimpleNamespace, tuple[str, ...]]:
    required = (
        "mc",
        "mcl",
        "water2infilt",
        "ae_ns",
        "free_drain_coef",
        "zwt_force",
        "evap_bare_lim_ns",
        "us",
        "humrel",
        "njsc",
    )
    missing = tuple(name for name in required if name not in fields)
    moistc = fields["mc"].transpose((0, 2, 1)) if "mc" in fields else None
    moistcl = fields["mcl"].transpose((0, 2, 1)) if "mcl" in fields else None
    return (
        SimpleNamespace(
            moistc=moistc,
            moistcl=moistcl,
            **{name: fields.get(name) for name in fields},
        ),
        missing,
    )


def _hydrol_state_namespace_from_previous_state(previous_state: DriverPreviousStepStatePacket) -> tuple[SimpleNamespace, tuple[str, ...]]:
    return _hydrol_state_namespace_from_fields(
        _previous_state_component_fields(previous_state, "hydrol_previous_step_state")
    )


def paper_1961_next_step_hydrol_precall_scaffold(
    config_path: str | Path,
    *,
    previous_state: DriverPreviousStepStatePacket,
    year: int = 1961,
    tstep: int = 1,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    used_run_def_path: str | Path | None = None,
    prepared_context: Paper1961PreparedDriverContext | None = None,
    hydrol_static_template: dict[str, object] | None = None,
    diffuco_day_static_cache: DriverDiffucoDayStaticCache | None = None,
    module_jit: bool = True,
    diffuco_local_jit: bool = True,
) -> DriverNextStepHydrolPrecallScaffold:
    """Assemble later-step HYDROL pre-call from local DIFFUCO/ENERBIL state."""

    enerbil = paper_1961_next_step_enerbil_precall_scaffold(
        config_path,
        previous_state=previous_state,
        year=year,
        tstep=tstep,
        fixed_format_trace_dir=fixed_format_trace_dir,
        static_trace_fields=static_trace_fields,
        used_run_def_path=used_run_def_path,
        prepared_context=prepared_context,
        module_jit=module_jit,
        diffuco_local_jit=diffuco_local_jit,
        diffuco_day_static_cache=diffuco_day_static_cache,
    )
    hydrol_state, hydrol_missing = _hydrol_state_namespace_from_previous_state(previous_state)
    if enerbil.enerbil_local is None:
        return DriverNextStepHydrolPrecallScaffold(
            tstep=int(tstep),
            enerbil=enerbil,
            hydrol_precall=None,
            hydrol_module=None,
            condveg_module=None,
            thermosoil_module=None,
            missing_previous_state_fields=tuple(dict.fromkeys((*enerbil.missing_previous_state_fields, *hydrol_missing, "enerbil_local"))),
        )

    context = prepared_context or prepare_paper_1961_driver_context(config_path, used_run_def_path=used_run_def_path)
    run_def_values = context.run_def_values
    nvm = int(np.asarray(previous_state.fields_by_component["slowproc_stomate_previous_step_state"]["lai"]).shape[1])
    cwrr_grid = context.cwrr_grid
    diaglev = context.diaglev
    hydrol_humcste = _validated_prepared_pft_axis(
        context.hydrol_humcste, context=context, nvm=nvm, name="HYDROL_HUMCSTE"
    )
    hydrol_zz_mm = context.hydrol_zz_mm
    if prepared_context is not None and hydrol_static_template is None:
        hydrol_static_template = hydrol_static_precall_template_from_slowproc(
            slowproc_restart=SimpleNamespace(**previous_state.fields_by_component["slowproc_stomate_previous_step_state"]),
            pref_soil_veg=enerbil.step.run_scalars.pref_soil_veg,
            nstm=enerbil.step.run_scalars.nstm,
            ext_coeff_vegetfrac=_validated_prepared_pft_axis(
                context.ext_coeff_vegetfrac,
                context=context,
                nvm=nvm,
                name="EXT_COEFF_VEGETFRAC",
            ),
            hydrol_njsc=getattr(hydrol_state, "njsc", None),
            refSOC_1d=getattr(hydrol_state, "refSOC_1d", None),
            use_refSOC_hydrol=parse_run_def_bool(run_def_values.get("use_refSOC_hydrol", "FALSE")),
            throughfall_by_pft=_validated_prepared_pft_axis(
                context.hydrol_throughfall_by_pft,
                context=context,
                nvm=nvm,
                name="PERCENT_THROUGHFALL_PFT",
            ),
            humcste=hydrol_humcste,
            zz_mm=hydrol_zz_mm,
            peat_hydro=context.hydrol_soil_peat_hydro,
            ok_dgvm=context.stomate_ok_dgvm,
            dt_days=context.dt_sechiba_days,
        )
    hydrol_precall = assemble_hydrol_first_step_precall_payload(
        enerbil_payload=enerbil.enerbil_local.payload,
        hydrol_restart=hydrol_state,
        slowproc_restart=SimpleNamespace(**previous_state.fields_by_component["slowproc_stomate_previous_step_state"]),
        pref_soil_veg=enerbil.step.run_scalars.pref_soil_veg,
        nstm=enerbil.step.run_scalars.nstm,
        ext_coeff_vegetfrac=_validated_prepared_pft_axis(
            context.ext_coeff_vegetfrac,
            context=context,
            nvm=nvm,
            name="EXT_COEFF_VEGETFRAC",
        ),
        precip_rain=enerbil.payload.precip_rain,
        precip_snow=enerbil.payload.precip_snow,
        diffuco_payload=_hydrol_current_diffuco_payload(
            enerbil.diffuco_precall.payload,
            enerbil.diffuco_local_enerbil_precall.result.payload.payload,
        ),
        thermosoil_restart=SimpleNamespace(
            **{
                name: previous_state.fields_by_component["thermosoil_previous_step_state"][name]
                for name in ("ptn", "gtemp", "lambda_snow", "cgrnd_snow", "dgrnd_snow")
                if name in previous_state.fields_by_component["thermosoil_previous_step_state"]
            }
        ),
        throughfall_by_pft=_validated_prepared_pft_axis(
            context.hydrol_throughfall_by_pft,
            context=context,
            nvm=nvm,
            name="PERCENT_THROUGHFALL_PFT",
        ),
        humcste=hydrol_humcste,
        zz_mm=hydrol_zz_mm,
        diaglev_m=diaglev,
        peat_hydro=context.hydrol_soil_peat_hydro,
        ok_dgvm=context.stomate_ok_dgvm,
        ok_explicitsnow=context.ok_explicitsnow,
        ok_freeze_cwrr=context.ok_freeze_cwrr,
        ok_thermodynamical_freezing=context.ok_thermodynamical_freezing,
        fr_dt=context.hydrol_fr_dt,
        froz_frac_corr=context.hydrol_froz_frac_corr,
        smtot_corr=context.hydrol_smtot_corr,
        max_froz_hydro=context.hydrol_max_froz_hydro,
        dt_days=context.dt_sechiba_days,
        static_payload_template=hydrol_static_template,
        use_jit=bool(module_jit) and prepared_context is not None,
    )
    combined_missing = tuple(dict.fromkeys((*enerbil.missing_previous_state_fields, *hydrol_missing)))
    hydrol_module = None
    if hydrol_precall.ok and not combined_missing:
        hydrol_module = run_hydrol_first_step_module_from_precall(
            hydrol_precall,
            temp_air=enerbil.payload.temp_air,
            pb=enerbil.payload.pb,
            u=enerbil.payload.u,
            v=enerbil.payload.v,
            humcste=hydrol_humcste,
            nroot_state=getattr(hydrol_state, "nroot", None),
            dz_mm=context.hydrol_dz_mm,
            dh_mm=context.hydrol_dh_mm,
            zz_mm=hydrol_zz_mm,
            altmax=previous_state.fields_by_component["slowproc_stomate_previous_step_state"].get("altmax"),
            ks=context.hydrol_cwrr_ks,
            reinf_slope=context.hydrol_reinf_slope,
            zmaxh_m=context.hydrol_depth_max_h,
            doponds=context.hydrol_do_ponds,
            peat_hydro=context.hydrol_soil_peat_hydro,
            peat_hydro_water_stress=context.hydrol_soil_peat_hydro,
            ok_freeze_cwrr=context.ok_freeze_cwrr,
            ok_pc=context.hydrol_ok_pc,
            ok_leak=context.hydrol_ok_leak,
            ok_ru2peat=context.hydrol_ok_ru2peat,
            ok_wt_ab=context.hydrol_ok_wt_ab,
            tides=context.hydrol_tides,
            agri_peat=context.hydrol_agri_peat,
            max_wt_ab=context.hydrol_max_wt_ab,
            dyn_nroot_larix=context.hydrol_dyn_nroot_larix,
            is_tree=enerbil.step.run_scalars.is_tree,
            znt=diaglev,
            run2peat=getattr(hydrol_state, "run2peat", None),
            run2man=getattr(hydrol_state, "run2man", None),
            dt_sechiba=context.dt_sechiba,
            use_jit=bool(module_jit) and prepared_context is not None,
        )
    condveg_module = None
    thermosoil_module = None
    if hydrol_module is not None and hydrol_module.ok:
        slowproc_state = previous_state.fields_by_component["slowproc_stomate_previous_step_state"]
        diffuco_state = previous_state.fields_by_component["diffuco_previous_step_state"]
        hydrol_prev = previous_state.fields_by_component["hydrol_previous_step_state"]
        therm_prev = previous_state.fields_by_component["thermosoil_previous_step_state"]
        condveg_module = run_condveg_first_step_module(
            restart_payload={
                "snow": hydrol_module.snow_state.snow,
                "snow_nobio": hydrol_module.snow_state.snow_nobio,
                "snowrho": hydrol_module.snow_state.snowrho,
                "snowdz": hydrol_module.snow_state.snowdz,
                "snow_age": hydrol_module.snow_state.snow_age,
                "snow_nobio_age": hydrol_module.snow_state.snow_nobio_age,
                "veget": slowproc_state["veget"],
                "veget_max": slowproc_state["veget_max"],
                "frac_nobio": slowproc_state["frac_nobio"],
                "height": diffuco_state["height"],
                "lai": slowproc_state["lai"],
                "soilalbedo_bg": diffuco_state["soilalbedo_bg"],
            },
            driver_payload={
                "zlev": enerbil.payload.zlev,
                "temp_air": enerbil.payload.temp_air,
                "pb": enerbil.payload.pb,
                "u": enerbil.payload.u,
                "v": enerbil.payload.v,
            },
            slowproc_payload={
                "totfrac_nobio": _slowproc_totfrac_nobio(slowproc_state),
                    "tot_bare_soil": slowproc_state["veget_max"][:, 0]
                    + (slowproc_state["veget_max"][:, 1:] - slowproc_state["veget"][:, 1:]).sum(axis=1),
                "is_tree": enerbil.step.run_scalars.is_tree,
                "z0_over_height": enerbil.step.run_scalars.z0_over_height,
                "ratio_z0m_z0h": enerbil.step.run_scalars.ratio_z0m_z0h,
            },
            hydrol_diagnostics=hydrol_module.diagnostics,
            run_def_values=run_def_values,
            ok_explicitsnow=context.ok_explicitsnow,
            impaze=context.impaze,
            rough_dyn=context.rough_dyn,
            use_jit=bool(module_jit) and prepared_context is not None,
        )
        if condveg_module.ok:
            thermosoil_module = run_thermosoil_first_step_module(
                moisture=hydrol_module.thermosoil_moisture,
                thermosoil_restart=SimpleNamespace(
                    ptn=therm_prev["ptn"],
                    cgrnd=therm_prev["cgrnd"],
                    dgrnd=therm_prev["dgrnd"],
                    cgrnd_snow=therm_prev["cgrnd_snow"],
                    dgrnd_snow=therm_prev["dgrnd_snow"],
                    lambda_snow=therm_prev["lambda_snow"],
                    gtemp=therm_prev["gtemp"],
                    temp_sol_beg=therm_prev.get("temp_sol_beg", therm_prev["gtemp"]),
                    pcapa_en=therm_prev.get("pcapa_en"),
                    refsoc=therm_prev["refSOC"],
                    shum_ngrnd_permalong=therm_prev.get("shum_ngrnd_permalong"),
                ),
                enerbil_payload=enerbil.enerbil_local.payload,
                condveg_result=condveg_module.module,
                snowrho=hydrol_module.snow_state.snowrho,
                snowtemp=hydrol_module.snow_state.snowtemp,
                snowdz=hydrol_module.snow_state.snowdz,
                njsc=hydrol_prev["njsc"],
                veget_max=slowproc_state["veget_max"],
                pb=enerbil.payload.pb,
                totfrac_nobio=_slowproc_totfrac_nobio(slowproc_state),
                dlt=cwrr_grid.dlt,
                dz1=cwrr_grid.dz1,
                zlt=cwrr_grid.zlt,
                znt=cwrr_grid.znt,
                dz5=cwrr_grid.dz5,
                dt_sechiba=context.dt_sechiba,
                ok_laidev=_validated_prepared_pft_axis(
                    context.ok_laidev, context=context, nvm=nvm, name="OK_LAIDEV"
                ),
                satsoil=context.thermosoil_satsoil,
                ok_shum_ngrnd_permalong=_effective_thermosoil_wetdiaglong(context.run_def_values),
                use_jit=bool(module_jit) and prepared_context is not None,
            )
    return DriverNextStepHydrolPrecallScaffold(
        tstep=int(tstep),
        enerbil=enerbil,
        hydrol_precall=hydrol_precall,
        hydrol_module=hydrol_module,
        condveg_module=condveg_module,
        thermosoil_module=thermosoil_module,
        missing_previous_state_fields=combined_missing,
    )


def read_driver_runtime_scalars(used_run_def_path: str | Path) -> DriverRuntimeScalars:
    """Read audited runtime scalars from the actual emitted ``used_run.def``."""

    values = parse_run_def(used_run_def_path)
    return DriverRuntimeScalars(
        dt_sechiba=parse_run_def_float(values, "DT_SECHIBA"),
        dt_stomate=parse_run_def_float(values, "DT_STOMATE"),
        stomate_ok_stomate=parse_run_def_bool(values["STOMATE_OK_STOMATE"]),
        stomate_ok_dgvm=parse_run_def_bool(values["STOMATE_OK_DGVM"]),
    )


def paper_1961_driver_cold_start_first_step_coverage(
    config_path: str | Path,
    *,
    year: int = 1961,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    used_run_def_path: str | Path | None = None,
    prepared_context: Paper1961PreparedDriverContext | None = None,
) -> ColdStartFirstStepStateCoverage:
    """Audit exact cold-start initialization coverage before first timestep.

    The returned object is intentionally not a full restart replacement. It
    records only cold-start fields with implemented Fortran-provenance sources
    and leaves every remaining initialization field explicit.
    """

    config = load_case_config(config_path)
    restart_cfg = config.get("restart_protocol", {}).get("cold_start", {})
    expected_none = {
        "RESTART_FILEIN": restart_cfg.get("RESTART_FILEIN"),
        "SECHIBA_restart_in": restart_cfg.get("SECHIBA_restart_in"),
        "STOMATE_RESTART_FILEIN": restart_cfg.get("STOMATE_RESTART_FILEIN"),
    }
    if config.get("minimal_single_case", {}).get("cold_start") is not True:
        raise ValueError("minimal_single_case.cold_start must be true for cold-start coverage")
    not_none = {key: value for key, value in expected_none.items() if str(value).strip().upper() != "NONE"}
    if not_none:
        raise ValueError(f"cold-start restart settings must be NONE, got {not_none}")

    run_def_path = (
        prepared_context.run_def_path
        if prepared_context is not None
        else Path(used_run_def_path)
        if used_run_def_path is not None
        else default_used_run_def_path(config_path)
    )
    run_def_values = prepared_context.run_def_values if prepared_context is not None else parse_run_def(run_def_path)
    dt_sechiba = (
        prepared_context.dt_sechiba
        if prepared_context is not None
        else parse_run_def_float(run_def_values, "DT_SECHIBA")
    )
    init_step = (
        prepared_context.first_step_bundle
        if prepared_context is not None and fixed_format_trace_dir is None and static_trace_fields is None
        else load_paper_1961_first_step_bundle(
            config_path,
            year=year,
            run_def_path=run_def_path,
            reference_run_dir=prepared_context.reference_run_dir if prepared_context is not None else None,
            fixed_format_trace_dir=fixed_format_trace_dir,
            static_trace_fields=static_trace_fields,
            domain_override=_domain_override_from_run_def_values(run_def_values),
        )
    )
    init_payload = build_intersurf_first_step_payload(init_step, dt_sechiba=dt_sechiba)
    step = load_paper_1961_step_bundle(
        config_path,
        year=year,
        tstep=0,
        run_def_path=run_def_path,
        reference_run_dir=prepared_context.reference_run_dir if prepared_context is not None else None,
        fixed_format_trace_dir=fixed_format_trace_dir,
        static_trace_fields=static_trace_fields,
        domain_override=_domain_override_from_run_def_values(run_def_values),
    )
    payload = build_intersurf_first_step_payload(step, driver_z0_for_wind=999999.0, dt_sechiba=dt_sechiba)
    run_scalars = read_run_scalars(config_path, run_def_path=run_def_path)
    vegetation = initialize_imposed_vegetation_state(run_scalars, npts=payload.kjpindex)
    nvm = int(run_scalars.nvm)
    fortran_pft_ids = tuple(int(value) for value in run_scalars.fortran_pft_ids)
    pft_to_mtc = parse_run_def_indexed_selection(
        run_def_values, "PFT_TO_MTC", fortran_pft_ids, dtype=int
    )
    hydrol_humcste = parse_run_def_indexed_selection(
        run_def_values, "HYDROL_HUMCSTE", fortran_pft_ids
    )
    if prepared_context is not None:
        cwrr_grid = prepared_context.cwrr_grid
        diaglev, deep_zlt, deep_znt = prepared_context.diaglev, prepared_context.zlt, prepared_context.znt
    else:
        cwrr_grid = paper_case_cwrr_vertical_soil_grid_from_used_run_def(run_def_path)
        diaglev, deep_zlt, deep_znt = paper_case_vertical_grids_from_used_run_def(run_def_path)

    diffuco_control = diffuco_control_inundation_input_coverage(
        diaglev_source_available=True,
        humcste_source_available=True,
        full_tide_height_available=payload.tide_height is not None,
        first_step_cold_start_biomass_zeroed=True,
    )
    slowproc_static = slowproc_static_entry_state(
        njsc=payload.njsc,
        soil_classif=run_def_values["SOILTYPE_CLASSIF"],
        pft_to_mtc=pft_to_mtc,
        zmaxh=parse_run_def_float(run_def_values, "DEPTH_MAX_H"),
        hydrol_humcste=hydrol_humcste,
    )
    veget_update = run_def_values.get("VEGET_UPDATE")
    if veget_update is None:
        if config.get("run_def_flags", {}).get("LAND_COVER_CHANGE") is not False:
            raise KeyError("VEGET_UPDATE")
        veget_update = "0Y"
    slowproc_no_lcc = slowproc_no_lcc_entry_state(
        veget_max=vegetation.veget_max,
        use_age_class=parse_run_def_bool(run_def_values["GLUC_USE_AGE_CLASS"]),
        veget_update=veget_update,
        map_pft_format=parse_run_def_bool(run_def_values.get("MAP_PFT_FORMAT", "TRUE")),
        impose_veg=run_scalars.impose_veg,
    )
    slowproc_fire = slowproc_fire_disabled_entry_state(
        kjpindex=payload.kjpindex,
        fire_disable=parse_run_def_bool(run_def_values["FIRE_DISABLE"]),
    )
    slowproc_dyn_peat = slowproc_dyn_peat_disabled_entry_state(
        kjpindex=payload.kjpindex,
        dyn_peat=parse_run_def_bool(run_def_values["DYN_PEAT"]),
    )
    slowproc_thermosoil = slowproc_thermosoil_entry_init_state(
        kjpindex=payload.kjpindex,
        nvm=nvm,
        znt=cwrr_grid.znt,
        zlt=cwrr_grid.zlt,
    )
    slowproc_veg_cold = slowproc_cold_start_vegetation_entry_state(
        veget_max=vegetation.veget_max,
        frac_nobio=vegetation.frac_nobio,
        pref_soil_veg=run_scalars.pref_soil_veg,
        ext_coeff_vegetfrac=_select_run_def_pft_vector(
            run_def_values, "EXT_COEFF_VEGETFRAC", run_scalars, nvm=nvm
        ),
        height_presc=_select_run_def_pft_vector(
            run_def_values, "SLOWPROC_HEIGHT", run_scalars, nvm=nvm
        ),
        nstm=run_scalars.nstm,
        nleafages=parse_run_def_int(run_def_values, "NLEAFAGES") if "NLEAFAGES" in run_def_values else 4,
        read_lai=parse_run_def_bool(run_def_values.get("READ_LAI", "FALSE")),
        ok_stomate=parse_run_def_bool(run_def_values.get("STOMATE_OK_STOMATE", "TRUE")),
    )
    slowproc_full_init = slowproc_init_pft14_explicit(
        restart={},
        veget_max_default=vegetation.veget_max,
        frac_nobio_default=vegetation.frac_nobio,
        height_presc=_select_run_def_pft_vector(
            run_def_values, "SLOWPROC_HEIGHT", run_scalars, nvm=nvm
        ),
        pref_soil_veg=run_scalars.pref_soil_veg,
        ext_coeff_vegetfrac=_select_run_def_pft_vector(
            run_def_values, "EXT_COEFF_VEGETFRAC", run_scalars, nvm=nvm
        ),
        nstm=run_scalars.nstm,
        diaglev=diaglev,
        soil_boundary={
            "soilclass": payload.soilclass,
            "clay_frac": payload.clay_frac,
            "sand_frac": payload.sand_frac,
            "silt_frac": payload.silt_frac,
            "bulk_dens": payload.bulk_dens,
            "soil_ph": payload.soil_ph,
            "poor_soils": payload.poor_soils,
        },
        salinity_data=payload.salinity,
        tide_height_data=payload.tide_height,
        impose_veg=run_scalars.impose_veg,
        impose_soilt=parse_run_def_bool(run_def_values.get("IMPOSE_SOILT", "FALSE")),
        ok_stomate=parse_run_def_bool(run_def_values.get("STOMATE_OK_STOMATE", "TRUE")),
        ok_dgvm=parse_run_def_bool(run_def_values.get("STOMATE_OK_DGVM", "FALSE")),
        map_pft_format=parse_run_def_bool(run_def_values.get("MAP_PFT_FORMAT", "TRUE")),
        use_age_class=parse_run_def_bool(run_def_values["GLUC_USE_AGE_CLASS"]),
        veget_update=veget_update,
        fire_disable=parse_run_def_bool(run_def_values["FIRE_DISABLE"]),
        dyn_peat=parse_run_def_bool(run_def_values["DYN_PEAT"]),
        hydrol_cwrr=parse_run_def_bool(run_def_values.get("HYDROL_CWRR", "TRUE")),
        get_slope=parse_run_def_bool(run_def_values.get("GET_SLOPE", "FALSE")),
        read_lai=parse_run_def_bool(run_def_values.get("READ_LAI", "FALSE")),
        read_salinity=parse_run_def_bool(run_def_values.get("READ_SALINITY", "TRUE")),
        read_tide=parse_run_def_bool(run_def_values.get("READ_TIDE", "TRUE")),
        soil_classif=run_def_values["SOILTYPE_CLASSIF"],
        nleafages=parse_run_def_int(run_def_values, "NLEAFAGES") if "NLEAFAGES" in run_def_values else 4,
        dt_stomate=parse_run_def_float(run_def_values, "DT_STOMATE"),
        precip_crit=parse_run_def_float(run_def_values, "PRECIP_CRIT"),
        PWT_lim=parse_run_def_float(run_def_values, "PWT_LIM"),
        PC_lim=parse_run_def_float(run_def_values, "PC_LIM"),
        sat_gsl=parse_run_def_float(run_def_values, "SAT_GSL"),
    )
    for name, expected, actual in (
        ("veget", slowproc_veg_cold.vegetation.veget, slowproc_full_init.veget),
        ("veget_max", slowproc_veg_cold.vegetation.veget_max, slowproc_full_init.veget_max),
        ("frac_nobio", slowproc_veg_cold.vegetation.frac_nobio, slowproc_full_init.frac_nobio),
        ("soiltile", slowproc_veg_cold.vegetation.soiltile, slowproc_full_init.soiltile),
        ("lai", slowproc_veg_cold.lai, slowproc_full_init.lai),
        ("height", slowproc_veg_cold.height, slowproc_full_init.height),
        ("frac_age", slowproc_veg_cold.frac_age, slowproc_full_init.frac_age),
    ):
        if not np.array_equal(np.asarray(expected), np.asarray(actual)):
            raise ValueError(f"composite slowproc_init disagrees with existing cold-start boundary for {name}")
    ptn_source_component = "ptn_constant_THERMOSOIL_TPRO"
    ptn_reftemp_meta = None
    if parse_run_def_bool(run_def_values.get("READ_REFTEMP", "FALSE")):
        if step.domain.resolution is None:
            raise ValueError("cold-start READ_REFTEMP requires explicit driver resolution")
        ptn_constant, ptn_reftemp_meta = read_paper_thermosoil_reftemp_static_field(
            config_path,
            lalo=step.domain.lalo,
            resolution_m=step.domain.resolution,
            ngrnd=len(cwrr_grid.znt),
            nvm=nvm,
        )
        ptn_source_component = "ptn_read_reftempfile_static_input"
    else:
        ptn_constant = thermosoil_initial_ptn_constant(
            kjpindex=payload.kjpindex,
            ngrnd=len(cwrr_grid.znt),
            nvm=nvm,
            thermosoil_tpro=float(run_def_values.get("THERMOSOIL_TPRO", 280.0)),
        )
    enerbil_surface = enerbil_cold_start_surface_state(
        qair=payload.qair,
        nvm=nvm,
        enerbil_tsurf=float(run_def_values.get("ENERBIL_TSURF", 280.0)),
    )
    hydrol_cold = hydrol_cold_start_state(
        veget_max=vegetation.veget_max,
        soiltile=vegetation.soiltile,
        pref_soil_veg=run_scalars.pref_soil_veg,
        nslm=len(cwrr_grid.znh),
        nsnow=3,
        frac_nobio=vegetation.frac_nobio,
        hydrol_moisture_content=float(run_def_values.get("HYDROL_MOISTURE_CONTENT", 0.3)),
        ok_freeze_cwrr=parse_run_def_bool(run_def_values.get("OK_FREEZE_CWRR", "TRUE")),
        zwt_force_default=float(run_def_values.get("ZWT_FORCE__00001", 1.0e20)),
        peat_hydro=PAPER_1961_HYDROL_SOIL_PEAT_HYDRO,
        peat_nodr=parse_run_def_bool(run_def_values.get("PEAT_NODR", "FALSE")),
        tides=bool(config.get("model_switches", {}).get("tides", False)),
        agri_peat=parse_run_def_bool(run_def_values.get("AGRI_PEAT", "FALSE")),
        agri_drain=parse_run_def_bool(run_def_values.get("AGRI_DRAIN", "FALSE")),
        drain_factor=float(run_def_values.get("DRAIN_FACTOR", 1.0)),
        zmaxh=parse_run_def_float(run_def_values, "DEPTH_MAX_H"),
    )
    hydrol_cold_moisture = hydrol_cold_start_thermosoil_moisture_inputs(
        hydrol_cold,
        pref_soil_veg=run_scalars.pref_soil_veg,
        dz_mm=cwrr_grid.dnh * 1000.0,
        dh_mm=cwrr_grid.dlh * 1000.0,
        njsc=payload.njsc,
        peat_hydro=PAPER_1961_HYDROL_SOIL_PEAT_HYDRO,
    )
    hydrol_waterbal_begin = hydrol_cold_start_waterbal_begin_state(
        hydrol_cold,
        veget_max=slowproc_veg_cold.vegetation.veget_max,
        soiltile=slowproc_veg_cold.vegetation.soiltile,
        dz_mm=cwrr_grid.dnh * 1000.0,
    )
    refsoc = None
    refsoc_overlap = None
    if parse_run_def_bool(run_def_values.get("use_refSOC", "FALSE")):
        if step.domain.resolution is None:
            raise ValueError("cold-start use_refSOC requires explicit driver resolution")
        refsoc, refsoc_overlap = read_paper_thermosoil_refsoc_static_field(
            config_path,
            lalo=step.domain.lalo,
            resolution_m=step.domain.resolution,
        )
    hydrol_refsoc_1d = None
    hydrol_refsoc_1d_overlap = None
    if parse_run_def_bool(run_def_values.get("use_refSOC_hydrol", "FALSE")):
        if step.domain.resolution is None:
            raise ValueError("cold-start use_refSOC_hydrol requires explicit driver resolution")
        hydrol_refsoc_1d, hydrol_refsoc_1d_overlap = read_paper_hydrol_refsoc_1d_static_field(
            config_path,
            lalo=step.domain.lalo,
            resolution_m=step.domain.resolution,
        )
    soilalbedo_bg = None
    soilalbedo_bg_meta = None
    condveg_albedo = None
    driver_swnet = None
    if parse_run_def_bool(run_def_values.get("ALB_BG_MODIS", "FALSE")):
        if step.domain.resolution is None:
            raise ValueError("cold-start ALB_BG_MODIS requires explicit driver resolution")
        soilalbedo_bg, soilalbedo_bg_meta = read_paper_condveg_background_soilalbedo(
            config_path,
            lalo=step.domain.lalo,
            resolution_m=step.domain.resolution,
        )
        snow_frac = condveg_frac_snow(
            snow=hydrol_cold.snow,
            snow_nobio=hydrol_cold.snow_nobio,
            snowrho=hydrol_cold.explicit_snow.snowrho,
            snowdz=hydrol_cold.explicit_snow.snowdz,
            ok_explicitsnow=parse_run_def_bool(run_def_values.get("OK_EXPLICITSNOW", "TRUE")),
        )
        condveg_albedo = condveg_albedo_explicit(
            veget=slowproc_veg_cold.vegetation.veget,
            veget_max=slowproc_veg_cold.vegetation.veget_max,
            drysoil_frac=np.zeros((payload.kjpindex,), dtype=np.float64),
            frac_nobio=slowproc_veg_cold.vegetation.frac_nobio,
            totfrac_nobio=slowproc_veg_cold.vegetation.totfrac_nobio,
            snow=hydrol_cold.snow,
            snow_age=hydrol_cold.snow_age,
            snow_nobio=hydrol_cold.snow_nobio,
            snow_nobio_age=hydrol_cold.snow_nobio_age,
            tot_bare_soil=slowproc_veg_cold.tot_bare_soil,
            frac_snow_veg=snow_frac.frac_snow_veg,
            frac_snow_nobio=snow_frac.frac_snow_nobio,
            impaze=parse_run_def_bool(run_def_values.get("IMPOSE_AZE", "FALSE")),
            alb_bg_modis=True,
            soilalb_bg=soilalbedo_bg,
            alb_leaf_vis=parse_run_def_indexed_selection(run_def_values, "ALB_LEAF_VIS", fortran_pft_ids),
            alb_leaf_nir=parse_run_def_indexed_selection(run_def_values, "ALB_LEAF_NIR", fortran_pft_ids),
            snowa_aged_vis=parse_run_def_indexed_selection(run_def_values, "SNOWA_AGED_VIS", fortran_pft_ids),
            snowa_aged_nir=parse_run_def_indexed_selection(run_def_values, "SNOWA_AGED_NIR", fortran_pft_ids),
            snowa_dec_vis=parse_run_def_indexed_selection(run_def_values, "SNOWA_DEC_VIS", fortran_pft_ids),
            snowa_dec_nir=parse_run_def_indexed_selection(run_def_values, "SNOWA_DEC_NIR", fortran_pft_ids),
            fixed_snow_albedo=float(run_def_values.get("CONDVEG_SNOWA", 1.0e20)),
            tcst_snowa=float(run_def_values.get("TCST_SNOWA", 10.0)),
            alb_ice=tuple(parse_run_def_float(run_def_values, f"ALB_ICE__{index:05d}") for index in range(1, 3)),
        )
        driver_swnet = np.asarray(driver_swnet_from_swdown_albedo(swdown=payload.swdown, albedo=condveg_albedo.albedo))
    stomate_cold_daily = stomate_cold_start_daily_accumulator_state(
        t2m=payload.temp_air,
        nvm=nvm,
        nslm=len(cwrr_grid.znh),
    )
    stomate_cold_season = stomate_cold_start_season_state(
        t2m=payload.temp_air,
        dt_days=parse_run_def_float(run_def_values, "DT_STOMATE") / 86400.0,
        nvm=nvm,
        nslm=len(cwrr_grid.znh),
        date=0,
    )
    stomate_cold_entry = stomate_cold_start_entry_state(
        t2m=payload.temp_air,
        nvm=nvm,
        nslm=len(cwrr_grid.znh),
        ndeep=32,
        nelements=1,
        dt_days=parse_run_def_float(run_def_values, "DT_STOMATE") / 86400.0,
        date=0,
        sla=parse_run_def_indexed_selection(run_def_values, "SLA", fortran_pft_ids),
    )
    thermosoil_cold_coef = thermosoil_cold_start_coef_closure(
        ptn=ptn_constant,
        moisture=hydrol_cold_moisture,
        temp_sol_new=enerbil_surface.temp_sol_new,
        temp_sol_new_pft=enerbil_surface.temp_sol_new_pft,
        snowdz=hydrol_cold.explicit_snow.snowdz,
        snowrho=hydrol_cold.explicit_snow.snowrho,
        snowtemp=hydrol_cold.explicit_snow.snowtemp,
        njsc=payload.njsc,
        veget_max=vegetation.veget_max,
        pb=payload.pb,
        dlt=cwrr_grid.dlt,
        dz1=cwrr_grid.dz1,
        zlt=cwrr_grid.zlt,
        znt=cwrr_grid.znt,
        dz5=cwrr_grid.dz5,
        dt_sechiba=parse_run_def_float(run_def_values, "DT_SECHIBA"),
        frac_snow_veg=np.zeros((payload.kjpindex,), dtype=np.float64),
        frac_snow_nobio=np.zeros_like(vegetation.frac_nobio),
        totfrac_nobio=vegetation.totfrac_nobio,
        refsoc=refsoc,
        use_refSOC=parse_run_def_bool(run_def_values.get("use_refSOC", "FALSE")),
        use_soilc_tempdiff=parse_run_def_bool(run_def_values.get("USE_SOILC_TEMPDIFF", "FALSE")),
        ok_laidev=np.asarray(
            [
                parse_run_def_bool(run_def_values.get(f"OK_LAIDEV__{fortran_id:05d}", "FALSE"))
                for fortran_id in fortran_pft_ids
            ],
            dtype=bool,
        ),
        shum_ngrnd_permalong=np.ones_like(ptn_constant, dtype=np.float64),
    )
    slowproc_erosion = slowproc_erosion_daily_zero_entry_state(kjpindex=payload.kjpindex)

    diffuco_control_salinity = assemble_pft14_control_salinity_first_step(
        salinity=payload.salinity,
        control_salinity_min=parse_run_def_float(run_def_values, "CONTROL_SALINITY_MIN"),
    )
    diffuco_control_inundation = assemble_pft14_control_inundation_first_step(
        depth_max_h=parse_run_def_float(run_def_values, "DEPTH_MAX_H"),
        depth_max_t=parse_run_def_float(run_def_values, "DEPTH_MAX_T"),
        depth_topthickness=parse_run_def_float(run_def_values, "DEPTH_TOPTHICK"),
        depth_cstthickness=parse_run_def_float(run_def_values, "DEPTH_CSTTHICK"),
        depth_geom=parse_run_def_float(run_def_values, "DEPTH_GEOM"),
        ratio_geom_below=parse_run_def_float(run_def_values, "RATIO_GEOM_BELOW"),
        pft_to_mtc=pft_to_mtc,
        hydrol_humcste=hydrol_humcste,
        tide_height=payload.tide_height,
        biomass=stomate_cold_entry.biomass,
        control_inudate_min=parse_run_def_float(run_def_values, "CONTROL_INUDATE_MIN"),
        agb_agr_ven_all_st=parse_run_def_float(run_def_values, "AGB_AGR_VEN_ALL_ST"),
        agb_agr_ven_all_pn=parse_run_def_float(run_def_values, "AGB_AGR_VEN_ALL_PN"),
        h_agr_max_st=parse_run_def_float(run_def_values, "H_AGR_MAX_ST"),
        h_agr_max_pn=parse_run_def_float(run_def_values, "H_AGR_MAX_PN"),
        pft_index=_paper_mangrove_pft_index(run_scalars),
    )
    slowproc_derivvar = slowproc_derivvar_explicit(
        veget=slowproc_veg_cold.vegetation.veget,
        lai=slowproc_veg_cold.lai,
        vcmax_fix=parse_run_def_indexed_selection(run_def_values, "VCMAX_FIX", fortran_pft_ids),
        height_presc=parse_run_def_indexed_selection(run_def_values, "SLOWPROC_HEIGHT", fortran_pft_ids),
        qsintcst=parse_run_def_float(run_def_values, "SECHIBA_QSINT"),
    )
    condveg_initial_surface = condveg_main_minimal(
        snow=hydrol_cold.snow,
        snow_age=hydrol_cold.snow_age,
        snow_nobio=hydrol_cold.snow_nobio,
        snow_nobio_age=hydrol_cold.snow_nobio_age,
        snowrho=hydrol_cold.explicit_snow.snowrho,
        snowdz=hydrol_cold.explicit_snow.snowdz,
        veget=slowproc_veg_cold.vegetation.veget,
        veget_max=slowproc_veg_cold.vegetation.veget_max,
        frac_nobio=slowproc_veg_cold.vegetation.frac_nobio,
        totfrac_nobio=slowproc_veg_cold.vegetation.totfrac_nobio,
        zlev=init_payload.zlev,
        height=slowproc_derivvar.height,
        temp_air=init_payload.temp_air,
        pb=init_payload.pb,
        u=init_payload.u,
        v=init_payload.v,
        lai=slowproc_veg_cold.lai,
        emis_scal=1.0,
        impaze=parse_run_def_bool(run_def_values.get("IMPOSE_AZE", "FALSE")),
        rough_dyn=parse_run_def_bool(run_def_values["ROUGH_DYN"]),
        ok_explicitsnow=parse_run_def_bool(run_def_values["OK_EXPLICITSNOW"]),
        drysoil_frac=np.full((payload.kjpindex,), 0.5, dtype=np.float64),
        tot_bare_soil=slowproc_veg_cold.tot_bare_soil,
        alb_bg_modis=parse_run_def_bool(run_def_values.get("ALB_BG_MODIS", "FALSE")),
        alb_bare_model=parse_run_def_bool(run_def_values.get("ALB_BARE_MODEL", "FALSE")),
        soilalb_bg=soilalbedo_bg,
        alb_leaf_vis=parse_run_def_indexed_selection(run_def_values, "ALB_LEAF_VIS", fortran_pft_ids),
        alb_leaf_nir=parse_run_def_indexed_selection(run_def_values, "ALB_LEAF_NIR", fortran_pft_ids),
        snowa_aged_vis=parse_run_def_indexed_selection(run_def_values, "SNOWA_AGED_VIS", fortran_pft_ids),
        snowa_aged_nir=parse_run_def_indexed_selection(run_def_values, "SNOWA_AGED_NIR", fortran_pft_ids),
        snowa_dec_vis=parse_run_def_indexed_selection(run_def_values, "SNOWA_DEC_VIS", fortran_pft_ids),
        snowa_dec_nir=parse_run_def_indexed_selection(run_def_values, "SNOWA_DEC_NIR", fortran_pft_ids),
        fixed_snow_albedo=float(run_def_values.get("CONDVEG_SNOWA", 1.0e20)),
        tcst_snowa=float(run_def_values.get("TCST_SNOWA", 10.0)),
        alb_ice=tuple(parse_run_def_float(run_def_values, f"ALB_ICE__{index:05d}") for index in range(1, 3)),
    )
    diffuco_restart_payload = {
        **enerbil_surface.as_payload(),
        "snow": hydrol_cold.snow,
        "snowdz": hydrol_cold.explicit_snow.snowdz,
        "snowrho": hydrol_cold.explicit_snow.snowrho,
        "snowtemp": hydrol_cold.explicit_snow.snowtemp,
        "snow_age": hydrol_cold.snow_age,
        "snow_nobio": hydrol_cold.snow_nobio,
        "snow_nobio_age": hydrol_cold.snow_nobio_age,
        "frac_nobio": slowproc_veg_cold.vegetation.frac_nobio,
        "veget": slowproc_veg_cold.vegetation.veget,
        "veget_max": slowproc_veg_cold.vegetation.veget_max,
        "lai": slowproc_veg_cold.lai,
        "qsintveg": hydrol_cold.qsintveg,
        "rstruct": np.broadcast_to(
            parse_run_def_indexed_selection(run_def_values, "RSTRUCT_CONST", fortran_pft_ids)[None, :],
            (payload.kjpindex, nvm),
        ),
        "roughheight": condveg_initial_surface.roughheight,
        "roughheight_pft": condveg_initial_surface.roughheight_pft,
        "z0h": condveg_initial_surface.z0h,
        "z0m": condveg_initial_surface.z0m,
        "humrel": hydrol_cold.humrel,
        "evap_bare_lim": hydrol_cold.evap_bare_lim,
        "drysoil_frac": np.full((payload.kjpindex,), 0.5, dtype=np.float64),
        "soilalbedo_bg": soilalbedo_bg,
        "height": slowproc_derivvar.height,
    }
    diffuco_precall = assemble_diffuco_first_step_precall_payload(
        driver_or_intersurf_payload={
            "u": payload.u,
            "v": payload.v,
            "zlev": payload.zlev,
            "temp_air": payload.temp_air,
            "qair": payload.qair,
            "pb": payload.pb,
            "swdown": payload.swdown,
            "ccanopy": payload.ccanopy,
            "precip_rain": payload.precip_rain,
            "lalo": payload.lalo,
            "neighbours": payload.neighbours,
            "resolution": payload.resolution,
        },
        restart_payload=diffuco_restart_payload,
        driver_albedo_payload={"albedo": condveg_initial_surface.albedo},
        slowproc_derivvar_payload={
            "qsintmax": slowproc_derivvar.qsintmax,
            "assim_param": slowproc_derivvar.assim_param,
            "height": slowproc_derivvar.height,
            "temp_growth": slowproc_derivvar.temp_growth,
        },
        diffuco_control_salinity=diffuco_control_salinity,
        diffuco_control_inundation=diffuco_control_inundation,
        ok_explicitsnow=parse_run_def_bool(run_def_values["OK_EXPLICITSNOW"]),
        river_routing=parse_run_def_bool(run_def_values["RIVER_ROUTING"]),
        nbp_glo=payload.kjpindex,
    )
    diffuco_local = None
    if diffuco_precall.ok:
        diffuco_local = run_pft14_local_enerbil_precall_from_first_step_precall(
            diffuco_precall,
            run_def_values=run_def_values,
            pft_index=_paper_mangrove_pft_index(run_scalars),
        )
    after_diffuco_payload = diffuco_precall.payload if diffuco_local is None else diffuco_local.result.payload.payload
    enerbil_precall = assemble_enerbil_first_step_precall_payload(
        driver_or_intersurf_payload=_enerbil_driver_coverage_payload(payload),
        driver_albedo_payload={"albedo": condveg_initial_surface.albedo},
        soil_thermal_restart_payload=_enerbil_soil_thermal_payload_from_thermosoil_cold_coef(thermosoil_cold_coef),
        after_diffuco_payload=after_diffuco_payload,
        condveg_impaze=parse_run_def_bool(run_def_values.get("IMPOSE_AZE", "FALSE")),
        non_watchout_driver_executed=True,
    )
    enerbil_local = None
    if enerbil_precall.ok:
        enerbil_local = run_enerbil_first_step_local_from_precall(
            enerbil_precall,
            ok_laidev=[
                parse_run_def_bool(run_def_values.get(f"OK_LAIDEV__{fortran_id:05d}", "FALSE"))
                for fortran_id in fortran_pft_ids
            ],
            dt_sechiba=parse_run_def_float(run_def_values, "DT_SECHIBA"),
            ok_explicitsnow=parse_run_def_bool(run_def_values["OK_EXPLICITSNOW"]),
            min_wind=parse_run_def_float(run_def_values, "MIN_WIND"),
        )
    hydrol_precall = None
    hydrol_module = None
    condveg_first_step_module = None
    thermosoil_first_step_module = None
    if enerbil_local is not None:
        hydrol_restart = SimpleNamespace(
            moistc=np.transpose(np.asarray(hydrol_cold.mc), (0, 2, 1)),
            moistcl=np.transpose(np.asarray(hydrol_cold.mcl), (0, 2, 1)),
            ae_ns=hydrol_cold.ae_ns,
            water2infilt=hydrol_cold.water2infilt,
            free_drain_coef=hydrol_cold.free_drain_coef,
            zwt_force=hydrol_cold.zwt_force,
            evap_bare_lim_ns=hydrol_cold.evap_bare_lim_ns,
            us=hydrol_cold.us,
            humrel=hydrol_cold.humrel,
            njsc=payload.njsc,
            profil_froz_hydro_ns=hydrol_cold.profil_froz_hydro_ns,
            temp_hydro=hydrol_cold.temp_hydro,
            stempdiag=thermosoil_cold_coef.stempdiag,
            wt_ab=hydrol_cold.wt_ab,
            wt_ab_tide=hydrol_cold.wt_ab_tide,
        )
        hydrol_precall = assemble_hydrol_first_step_precall_payload(
            enerbil_payload=enerbil_local.payload,
            hydrol_restart=hydrol_restart,
            slowproc_restart=SimpleNamespace(
                lai=slowproc_veg_cold.lai,
                frac_nobio=slowproc_veg_cold.vegetation.frac_nobio,
                veget=slowproc_veg_cold.vegetation.veget,
                veget_max=slowproc_veg_cold.vegetation.veget_max,
                soiltile=slowproc_veg_cold.vegetation.soiltile,
                totfrac_nobio=slowproc_veg_cold.vegetation.totfrac_nobio,
                tot_bare_soil=slowproc_veg_cold.tot_bare_soil,
            ),
            pref_soil_veg=run_scalars.pref_soil_veg,
            nstm=run_scalars.nstm,
            ext_coeff_vegetfrac=parse_run_def_indexed_selection(
                run_def_values, "EXT_COEFF_VEGETFRAC", fortran_pft_ids
            ),
            refSOC_1d=hydrol_refsoc_1d,
            use_refSOC_hydrol=parse_run_def_bool(run_def_values.get("use_refSOC_hydrol", "FALSE")),
            precip_rain=payload.precip_rain,
            precip_snow=payload.precip_snow,
            diffuco_payload=_hydrol_current_diffuco_payload(diffuco_precall.payload, after_diffuco_payload),
            thermosoil_restart=None,
            throughfall_by_pft=parse_run_def_indexed_selection(
                run_def_values, "PERCENT_THROUGHFALL_PFT", fortran_pft_ids
            ),
            humcste=hydrol_humcste,
            zz_mm=cwrr_grid.znh * 1000.0,
            diaglev_m=paper_case_vertical_grids_from_used_run_def(run_def_path)[0],
            peat_hydro=PAPER_1961_HYDROL_SOIL_PEAT_HYDRO,
            ok_dgvm=parse_run_def_bool(run_def_values["STOMATE_OK_DGVM"]),
            ok_explicitsnow=parse_run_def_bool(run_def_values["OK_EXPLICITSNOW"]),
            ok_freeze_cwrr=parse_run_def_bool(run_def_values["OK_FREEZE_CWRR"]),
            ok_thermodynamical_freezing=parse_run_def_bool(run_def_values["OK_THERMODYNAMICAL_FREEZING"]),
            fr_dt=parse_run_def_float(run_def_values, "FR_DT"),
            froz_frac_corr=parse_run_def_float(run_def_values, "FROZ_FRAC_CORR"),
            smtot_corr=parse_run_def_float(run_def_values, "SMTOT_CORR"),
            max_froz_hydro=parse_run_def_float(run_def_values, "MAX_FROZ_HYDRO"),
            dt_days=parse_run_def_float(run_def_values, "DT_SECHIBA") / 86400.0,
        )
        if hydrol_precall.ok:
            hydrol_module = run_hydrol_first_step_module_from_precall(
                hydrol_precall,
                temp_air=payload.temp_air,
                pb=payload.pb,
                u=payload.u,
                v=payload.v,
                humcste=hydrol_humcste,
                dz_mm=cwrr_grid.dnh * 1000.0,
                dh_mm=cwrr_grid.dlh * 1000.0,
                zz_mm=cwrr_grid.znh * 1000.0,
                altmax=stomate_cold_entry.altmax,
                ks=parse_run_def_indexed_vector(run_def_values, "CWRR_KS", 12),
                reinf_slope=[float(run_def_values.get("SLOPE", "0.0"))],
                zmaxh_m=parse_run_def_float(run_def_values, "DEPTH_MAX_H"),
                doponds=parse_run_def_bool(run_def_values["DO_PONDS"]),
                peat_hydro=PAPER_1961_HYDROL_SOIL_PEAT_HYDRO,
                peat_hydro_water_stress=PAPER_1961_HYDROL_SOIL_PEAT_HYDRO,
                ok_freeze_cwrr=parse_run_def_bool(run_def_values["OK_FREEZE_CWRR"]),
                ok_pc=parse_run_def_bool(run_def_values["OK_PC"]),
                ok_leak=parse_run_def_bool(run_def_values["OK_LEAK"]),
                ok_ru2peat=parse_run_def_bool(run_def_values["OK_RU2PEAT"]),
                ok_wt_ab=parse_run_def_bool(run_def_values["OK_WT_AB"]),
                tides=parse_run_def_bool(run_def_values["TIDES"]),
                agri_peat=parse_run_def_bool(run_def_values["AGRI_PEAT"]),
                max_wt_ab=parse_run_def_float(run_def_values, "max_wt_ab") if "max_wt_ab" in run_def_values else 100.0,
                dyn_nroot_larix=parse_run_def_bool(run_def_values["dyn_nroot_larix"]),
                is_tree=run_scalars.is_tree,
                znt=paper_case_vertical_grids_from_used_run_def(run_def_path)[0],
                run2peat=hydrol_cold.run2peat,
                run2man=hydrol_cold.run2man,
                runoff2peat=np.zeros((payload.kjpindex, run_scalars.nstm), dtype=np.float64),
                dt_sechiba=parse_run_def_float(run_def_values, "DT_SECHIBA"),
            )
            if hydrol_module.ok:
                condveg_first_step_module = run_condveg_first_step_module(
                    restart_payload={
                        "snow": hydrol_module.snow_state.snow,
                        "snow_nobio": hydrol_module.snow_state.snow_nobio,
                        "snowrho": hydrol_module.snow_state.snowrho,
                        "snowdz": hydrol_module.snow_state.snowdz,
                        "snow_age": hydrol_module.snow_state.snow_age,
                        "snow_nobio_age": hydrol_module.snow_state.snow_nobio_age,
                        "veget": slowproc_veg_cold.vegetation.veget,
                        "veget_max": slowproc_veg_cold.vegetation.veget_max,
                        "frac_nobio": slowproc_veg_cold.vegetation.frac_nobio,
                        "height": slowproc_derivvar.height,
                        "lai": slowproc_veg_cold.lai,
                        "soilalbedo_bg": soilalbedo_bg,
                    },
                    driver_payload={
                        "zlev": payload.zlev,
                        "temp_air": payload.temp_air,
                        "pb": payload.pb,
                        "u": payload.u,
                        "v": payload.v,
                    },
                    slowproc_payload={
                        "totfrac_nobio": slowproc_veg_cold.vegetation.totfrac_nobio,
                        "tot_bare_soil": slowproc_veg_cold.tot_bare_soil,
                        "is_tree": run_scalars.is_tree,
                        "z0_over_height": run_scalars.z0_over_height,
                        "ratio_z0m_z0h": run_scalars.ratio_z0m_z0h,
                    },
                    hydrol_diagnostics=hydrol_module.diagnostics,
                    run_def_values=run_def_values,
                    ok_explicitsnow=parse_run_def_bool(run_def_values["OK_EXPLICITSNOW"]),
                    impaze=parse_run_def_bool(run_def_values["IMPOSE_AZE"]),
                    rough_dyn=parse_run_def_bool(run_def_values["ROUGH_DYN"]),
                )
                thermosoil_restart = _thermosoil_restart_namespace_from_cold_start(
                    ptn=ptn_constant,
                    coef=thermosoil_cold_coef,
                    refsoc=refsoc,
                    temp_sol_beg=enerbil_surface.temp_sol_new,
                    shum_ngrnd_permalong=np.ones_like(ptn_constant, dtype=np.float64),
                    e_soil_lat=(
                        np.zeros((payload.kjpindex, nvm), dtype=np.float64)
                        if parse_run_def_bool(run_def_values["OK_ECORR"])
                        else None
                    ),
                )
                if condveg_first_step_module.ok and thermosoil_restart is not None:
                    thermosoil_first_step_module = run_thermosoil_first_step_module(
                        moisture=hydrol_module.thermosoil_moisture,
                        thermosoil_restart=thermosoil_restart,
                        enerbil_payload=enerbil_local.payload,
                        condveg_result=condveg_first_step_module.module,
                        snowrho=hydrol_module.snow_state.snowrho,
                        snowtemp=hydrol_module.snow_state.snowtemp,
                        snowdz=hydrol_module.snow_state.snowdz,
                        njsc=payload.njsc,
                        veget_max=slowproc_veg_cold.vegetation.veget_max,
                        pb=payload.pb,
                        totfrac_nobio=slowproc_veg_cold.vegetation.totfrac_nobio,
                        dlt=cwrr_grid.dlt,
                        dz1=cwrr_grid.dz1,
                        zlt=cwrr_grid.zlt,
                        znt=cwrr_grid.znt,
                        dz5=cwrr_grid.dz5,
                        dt_sechiba=parse_run_def_float(run_def_values, "DT_SECHIBA"),
                        ok_laidev=[
                            parse_run_def_bool(run_def_values.get(f"OK_LAIDEV__{fortran_id:05d}", "FALSE"))
                            for fortran_id in fortran_pft_ids
                        ],
                        satsoil=parse_run_def_bool(run_def_values["satsoil"]),
                        ok_shum_ngrnd_permalong=_effective_thermosoil_wetdiaglong(run_def_values),
                    )

    covered_diffuco = [
        "diaglev_z_soil",
        "humcste_use_rprof",
        "cold_start_biomass_zero_fallback",
        "salinity_control_first_step",
        "inundation_control_first_step",
        "rstruct_const_missing_restart_fallback",
            "roughness_from_condveg_initialize_raw_first_slab_dynamic_branch",
        "drysoil_frac_missing_restart_fallback_0p5",
    ]
    if payload.tide_height is not None:
        covered_diffuco.append("tide_height")
    if diffuco_precall.ok:
        covered_diffuco.append("diffuco_first_step_precall_closed")
    if diffuco_local is not None:
        covered_diffuco.append("diffuco_pft14_local_to_enerbil_boundary_closed")

    covered = {
        "driver": (
            "forcing_step",
            "domain_payload",
            "run_def_scalars",
            "restart_protocol_none",
            *(() if driver_swnet is None else ("swnet_from_condveg_cold_start_albedo",)),
        ),
        "slowproc": (
            "imposed_veget_max",
            "frac_nobio_zero",
            "soiltile_from_veget_max",
            "lai_zero_cold_start",
            "frac_age_first_class_cold_start",
            "height_presc_cold_start",
            "veget_from_lai_and_veget_max_cold_start",
            "tot_bare_soil_cold_start",
            "no_lcc_entry_state",
            "fire_disabled_entry_state",
            "dyn_peat_disabled_entry_state",
            "static_fc_grazing_humcste_use",
            "thermosoil_entry_init_tdeep_hsdeep_heat_zimov",
            "erosion_daily_zero_inputs",
        ),
        "diffuco": tuple(covered_diffuco),
        "enerbil": (
            "temp_sol_constant_ENERBIL_TSURF",
            "temp_sol_pft_constant_ENERBIL_TSURF",
            "temp_sol_new_initialized_from_temp_sol",
            "qsurf_initialized_from_qair",
            "evapot_zero",
            "evapot_corr_zero",
            "tsol_rad_initialized_from_temp_sol",
            "potential_surface_state_initialized",
            *(() if condveg_albedo is None else ("driver_albedo_from_condveg_cold_start",)),
            *(
                (
                    "soilcap_from_thermosoil_coef",
                    "soilcap_pft_from_thermosoil_coef",
                    "soilflx_from_thermosoil_coef",
                    "soilflx_pft_from_thermosoil_coef",
                )
                if thermosoil_cold_coef.ok
                else ()
            ),
            *(() if not enerbil_precall.ok else ("enerbil_first_step_precall_closed",)),
            *(() if enerbil_local is None else ("enerbil_first_step_local_closed",)),
        ),
        "hydrol": (
            "mc_mcl_constant_HYDROL_MOISTURE_CONTENT",
            "us_humrelv_vegstressv_zero_US_INIT",
            "humrel_zero_from_humrelv",
            "water2infilt_zero",
            "ae_ns_zero",
            "evap_bare_lim_ns_zero",
            "evap_bare_lim_zero_from_tile_sum",
            "zwt_force_default_and_zforce",
            "free_drain_coef_default_with_peat_tide_switches",
            "hydrol_snow_mass_age_nobio_zero",
            "qsintveg_zero",
            "freeze_cwrr_default_profiles",
            "peat_tide_water_table_buffers_zero",
            "resdist_initialized_from_soiltile",
            "vegtot_old_and_vegtot_from_veget_max",
            "mask_veget_and_mask_soiltile",
            "explicit_snow_initialize_defaults",
            "hydrol_var_init_thermosoil_moisture_inputs",
            "hydrol_alma_waterbal_begin_fields",
            *(() if hydrol_precall is None or not hydrol_precall.ok else ("hydrol_first_step_precall_closed",)),
            *(() if hydrol_module is None or not hydrol_module.ok else ("hydrol_first_step_module_closed",)),
        ),
        "condveg": (
            *(() if soilalbedo_bg is None else ("soilalbedo_bg_read_ALB_BG_FILE",)),
            *(() if condveg_albedo is None else ("albedo_cold_start_from_condveg_albedo",)),
            *(() if condveg_first_step_module is None or not condveg_first_step_module.ok else ("condveg_first_step_module_closed",)),
        ),
        "thermosoil": (
                    ptn_source_component,
            *(() if refsoc is None else ("refSOC_read_refSOCfile_static_input",)),
            "coefficient_recompute_contract_for_missing_restart_fields",
            *(
                (
                    "thermosoil_humlev_cold_start",
                    "thermosoil_getdiff_cold_start",
                    "cgrnd_dgrnd_non_restart_initialization",
                    "lambda_snow_non_restart_initialization",
                    "soilcap_soilflx_non_restart_initialization",
                )
                if thermosoil_cold_coef.ok
                else ()
            ),
            *(() if thermosoil_first_step_module is None or not thermosoil_first_step_module.ok else ("thermosoil_first_step_module_closed",)),
        ),
        "stomate": (
            "readstart_val_exp_fallback_entry_state",
            "readstart_val_exp_fallback_season_memory",
            "readstart_val_exp_fallback_daily_accumulators",
            "biomass_zero_fallback",
            "plant_status_and_respiration_zero_fallbacks",
            "season_memory_cold_start_initialization",
            "daily_accumulator_cold_start_initialization",
            "carbon_litter_doc_initial_state_zero_fallbacks",
        ),
    }
    missing = {
        "driver": (),
        "diffuco": tuple(
            dict.fromkeys(
                (
                    *diffuco_control.missing_inputs,
                    *diffuco_control_salinity.missing_inputs,
                    *diffuco_control_inundation.missing_inputs,
                    *diffuco_precall.missing_local_process_inputs,
                )
            )
        ),
        "enerbil": (
            *(() if condveg_albedo is not None else ("driver_albedo_non_restart_initialization",)),
            *(
                ()
                if thermosoil_cold_coef.ok
                else (
                    "soilcap_non_restart_thermosoil_coef",
                    "soilcap_pft_non_restart_thermosoil_coef",
                    "soilflx_non_restart_thermosoil_coef",
                    "soilflx_pft_non_restart_thermosoil_coef",
                )
            ),
            *enerbil_precall.missing_external_inputs,
        ),
        "hydrol": tuple(
            dict.fromkeys(
                (
                    *(("hydrol_first_step_precall_waiting_for_enerbil_local",) if hydrol_precall is None else hydrol_precall.missing_inputs),
                    *(("hydrol_first_step_module_waiting_for_precall",) if hydrol_module is None else hydrol_module.missing_inputs),
                )
            )
        ),
        "thermosoil": (
            *thermosoil_cold_coef.missing_inputs,
            *(
                ()
                if thermosoil_cold_coef.ok
                else (
                    "cgrnd_dgrnd_non_restart_initialization_waiting_for_refSOC",
                    "lambda_snow_non_restart_initialization_waiting_for_refSOC",
                    "thermosoil_coef_non_restart_soilcap_soilflx_waiting_for_refSOC",
                )
            ),
            *(("thermosoil_first_step_module_waiting_for_condveg",) if thermosoil_first_step_module is None else thermosoil_first_step_module.missing_inputs),
        ),
        "slowproc": (),
        "stomate": (),
    }

    cold_start_first_step_entry_payload = None
    if not any(missing.values()):
        nflow = parse_run_def_int(run_def_values, "NSTM")
        driver_source = stomate_driver_entry_source(payload, kjit=1)
        routing_source = stomate_no_routing_entry_source(
            kjpindex=payload.kjpindex,
            nflow=nflow,
            river_routing=parse_run_def_bool(run_def_values["RIVER_ROUTING"]),
            nbp_glo=payload.kjpindex,
        )
        thermosoil_entry = slowproc_thermosoil_entry_init_state(
            kjpindex=payload.kjpindex,
            nvm=nvm,
            znt=cwrr_grid.znt,
            zlt=cwrr_grid.zlt,
        )
        same_step_sources = _paper_first_step_same_step_entry_sources(
            enerbil=enerbil_local,
            after_diffuco_payload=after_diffuco_payload,
            hydrol=hydrol_module,
            thermosoil=thermosoil_first_step_module,
            snow_state=(
                {
                    "snow": hydrol_module.snow_state.snow,
                    "snowdz": hydrol_module.snow_state.snowdz,
                    "snowrho": hydrol_module.snow_state.snowrho,
                }
                if hydrol_module is not None
                and hydrol_module.ok
                and hydrol_module.snow_state is not None
                else None
            ),
            temp_sol=enerbil_surface.temp_sol,
            znt=cwrr_grid.znt,
            zlt=cwrr_grid.zlt,
        )
        fwet_source = {}
        if hydrol_module is not None and hydrol_module.ok and hydrol_module.diagnostics.fwet_new is not None:
            fwet_source = {"fwet_new": hydrol_module.diagnostics.fwet_new}
        elif not parse_run_def_bool(run_def_values["TOPMODEL_NEW"]):
            fwet_source = {"fwet_new": hydrol_cold.fwet_new}
        assembly = assemble_stomate_main_payload(
            driver_source,
            routing_source,
            {
                "veget": slowproc_veg_cold.vegetation.veget,
                "veget_max": slowproc_veg_cold.vegetation.veget_max,
                "lai": slowproc_veg_cold.lai,
                "frac_nobio": slowproc_veg_cold.vegetation.frac_nobio,
                "height": slowproc_derivvar.height,
                "frac_age": slowproc_veg_cold.frac_age,
                "totfrac_nobio": slowproc_veg_cold.vegetation.totfrac_nobio,
            },
            stomate_slowproc_derivvar_entry_source(slowproc_derivvar),
            stomate_slowproc_no_lcc_entry_source(slowproc_no_lcc),
            stomate_slowproc_fire_disabled_entry_source(slowproc_fire),
            stomate_slowproc_dyn_peat_disabled_entry_source(slowproc_dyn_peat),
            stomate_slowproc_thermosoil_entry_source(thermosoil_entry),
            stomate_slowproc_static_entry_source(slowproc_static),
            stomate_erosion_disabled_erodepth_entry_source(
                erosion_module=parse_run_def_bool(run_def_values["EROSION_MODULE"]),
                kjpindex=payload.kjpindex,
                nvm=nvm,
            ),
            stomate_slowproc_erosion_daily_zero_entry_source(slowproc_erosion),
            stomate_restart_entry_source(stomate_cold_entry),
            *same_step_sources,
            fwet_source,
        )
        blocking_missing = tuple(
            name
            for name in assembly.missing_inputs
            if name not in assembly.missing_by_source.get("io_handle", ())
        )
        if not blocking_missing and not assembly.partial_inputs:
            cold_start_first_step_entry_payload = dict(assembly.payload)

    initialized_payloads = {
        "vegetation": vegetation,
        "first_step_bundle": step,
        "first_step_payload": payload,
        "cwrr_grid": cwrr_grid,
        "diaglev": diaglev,
        "deep_zlt": deep_zlt,
        "deep_znt": deep_znt,
        "run_scalars": run_scalars,
        "diffuco_control_coverage": diffuco_control,
        "diffuco_control_salinity": diffuco_control_salinity,
        "diffuco_control_inundation": diffuco_control_inundation,
        "slowproc_derivvar": slowproc_derivvar,
        "condveg_initial_surface": condveg_initial_surface,
        "diffuco_first_step_precall": diffuco_precall,
        "diffuco_first_step_local_enerbil_precall": diffuco_local,
        "enerbil_first_step_precall": enerbil_precall,
        "enerbil_first_step_local": enerbil_local,
        "hydrol_first_step_precall": hydrol_precall,
        "hydrol_first_step_module": hydrol_module,
        "condveg_first_step_module": condveg_first_step_module,
        "thermosoil_first_step_module": thermosoil_first_step_module,
        "enerbil_surface_state": enerbil_surface,
        "hydrol_cold_start_state": hydrol_cold,
        "hydrol_cold_start_thermosoil_moisture": hydrol_cold_moisture,
        "hydrol_waterbal_begin": hydrol_waterbal_begin,
        "slowproc_cold_start_vegetation": slowproc_veg_cold,
        "slowproc_init_pft14": slowproc_full_init,
        "condveg_soilalbedo_bg": soilalbedo_bg,
        "condveg_soilalbedo_bg_meta": soilalbedo_bg_meta,
        "condveg_albedo": condveg_albedo,
        "driver_swnet": driver_swnet,
        "forcing_t2m_for_stomate": np.asarray(payload.temp_air),
        "stomate_cold_start_daily_accumulators": stomate_cold_daily,
        "stomate_cold_start_season_state": stomate_cold_season,
        "stomate_cold_start_entry_state": stomate_cold_entry,
        "thermosoil_refSOC": refsoc,
        "thermosoil_refSOC_overlap": refsoc_overlap,
        "hydrol_refSOC_1d": hydrol_refsoc_1d,
        "hydrol_refSOC_1d_overlap": hydrol_refsoc_1d_overlap,
        "thermosoil_cold_start_coef": thermosoil_cold_coef,
        "thermosoil_e_soil_lat": (
            np.zeros((payload.kjpindex, nvm), dtype=np.float64)
            if parse_run_def_bool(run_def_values["OK_ECORR"])
            else None
        ),
        "slowproc_static": slowproc_static,
        "slowproc_no_lcc": slowproc_no_lcc,
        "slowproc_fire_disabled": slowproc_fire,
        "slowproc_dyn_peat_disabled": slowproc_dyn_peat,
        "slowproc_thermosoil_entry_init": slowproc_thermosoil,
        "thermosoil_ptn_constant": ptn_constant,
        "thermosoil_reftemp_overlap": ptn_reftemp_meta,
        "slowproc_erosion_daily_zero": slowproc_erosion,
        "cold_start_first_step_entry_payload": cold_start_first_step_entry_payload,
    }
    if not any(missing.values()):
        initialized_payloads["cold_start_first_step_previous_state"] = _previous_step_state_from_cold_start_payloads(initialized_payloads)

    return ColdStartFirstStepStateCoverage(
        mode=COLD_START_INITIAL_STATE_MODE,
        covered_by_component=covered,
        missing_by_component=missing,
        initialized_payloads=initialized_payloads,
    )


def _stomate_entry_assembly_from_driver_scaffold(
    *,
    payload: IntersurfFirstStepPayload,
    kjit: int,
    nflow: int,
    river_routing: bool,
    nbp_glo: int,
) -> StomateMainPayloadAssembly:
    driver_source = stomate_driver_entry_source(payload, kjit=kjit)
    routing_source = stomate_no_routing_entry_source(
        kjpindex=payload.kjpindex,
        nflow=nflow,
        river_routing=river_routing,
        nbp_glo=nbp_glo,
    )
    return assemble_stomate_main_payload(driver_source, routing_source)


def _enerbil_driver_coverage_payload(payload: IntersurfFirstStepPayload) -> dict[str, object]:
    """Expose driver/intersurf fields by ENERBIL coverage names."""

    return {
        "zlev": payload.zlev,
        "lwdown": payload.lwdown,
        "swdown": payload.swdown,
        "temp_air": payload.temp_air,
        "u": payload.u,
        "v": payload.v,
        "qair": payload.qair,
        "pb": payload.pb,
        "precip_rain": payload.precip_rain,
    }


def _enerbil_soil_thermal_payload_from_thermosoil_cold_coef(thermosoil_cold_coef) -> dict[str, object]:
    """Expose cold-start THERMOSOIL coefficients by ENERBIL argument names."""

    if thermosoil_cold_coef is None or not thermosoil_cold_coef.ok:
        return {}
    return {
        "soilcap": thermosoil_cold_coef.coef.soilcap,
        "soilflx": thermosoil_cold_coef.coef.soilflx,
        "soilcap_pft": thermosoil_cold_coef.coef.soil.soilcap_pft,
        "soilflx_pft": thermosoil_cold_coef.coef.soil.soilflx_pft,
    }


def _paper_first_step_restart_entry_sources(
    *,
    restart_state: ReferenceCaseFirstStepRestartState,
    run_scalars: RunScalars,
    run_def_values: dict[str, str],
    znt,
    zlt,
) -> tuple[dict[str, object], ...]:
    """Build exact restart/init sources for first-step ``stomate_main``."""

    nvm = int(restart_state.slowproc.lai.shape[1])
    derivvar = slowproc_derivvar_explicit(
        veget=restart_state.slowproc.veget,
        lai=restart_state.slowproc.lai,
        vcmax_fix=_select_run_def_pft_vector(
            run_def_values, "VCMAX_FIX", run_scalars, nvm=nvm
        ),
        height_presc=_select_run_def_pft_vector(
            run_def_values, "SLOWPROC_HEIGHT", run_scalars, nvm=nvm
        ),
        qsintcst=parse_run_def_float(run_def_values, "SECHIBA_QSINT"),
    )
    no_lcc = slowproc_no_lcc_entry_state(
        veget_max=restart_state.slowproc.veget_max,
        use_age_class=parse_run_def_bool(run_def_values["GLUC_USE_AGE_CLASS"]),
        veget_update="0Y",
        map_pft_format=parse_run_def_bool(run_def_values.get("MAP_PFT_FORMAT", "TRUE")),
        impose_veg=run_scalars.impose_veg,
    )
    fire = slowproc_fire_disabled_entry_state(
        kjpindex=restart_state.slowproc.lai.shape[0],
        fire_disable=parse_run_def_bool(run_def_values["FIRE_DISABLE"]),
    )
    peat = slowproc_dyn_peat_disabled_entry_state(
        kjpindex=restart_state.slowproc.lai.shape[0],
        dyn_peat=parse_run_def_bool(run_def_values["DYN_PEAT"]),
    )
    thermosoil = slowproc_thermosoil_entry_init_state(
        kjpindex=restart_state.slowproc.lai.shape[0],
        nvm=nvm,
        znt=znt,
        zlt=zlt,
    )
    static = slowproc_static_entry_state(
        njsc=restart_state.sechiba_static.njsc,
        soil_classif=run_def_values["SOILTYPE_CLASSIF"],
        pft_to_mtc=_select_run_def_pft_vector(
            run_def_values, "PFT_TO_MTC", run_scalars, nvm=nvm, dtype=int
        ),
        zmaxh=parse_run_def_float(run_def_values, "DEPTH_MAX_H"),
        hydrol_humcste=_select_run_def_pft_vector(
            run_def_values, "HYDROL_HUMCSTE", run_scalars, nvm=nvm
        ),
    )
    erosion = slowproc_erosion_daily_zero_entry_state(kjpindex=restart_state.slowproc.lai.shape[0])

    return (
        stomate_slowproc_restart_entry_source(restart_state.slowproc),
        stomate_slowproc_derivvar_entry_source(derivvar),
        stomate_slowproc_no_lcc_entry_source(no_lcc),
        stomate_slowproc_fire_disabled_entry_source(fire),
        stomate_slowproc_dyn_peat_disabled_entry_source(peat),
        stomate_slowproc_thermosoil_entry_source(thermosoil),
        stomate_slowproc_static_entry_source(static),
        stomate_erosion_disabled_erodepth_entry_source(
            erosion_module=parse_run_def_bool(run_def_values["EROSION_MODULE"]),
            kjpindex=restart_state.slowproc.lai.shape[0],
            nvm=nvm,
        ),
        stomate_slowproc_erosion_daily_zero_entry_source(erosion),
        stomate_restart_entry_source(restart_state.stomate),
    )


def _paper_first_step_same_step_entry_sources(
    *,
    enerbil: EnerbilFirstStepLocalAssembly | None,
    after_diffuco_payload: dict[str, object] | None,
    hydrol: HydrolFirstStepModuleClosure | None,
    thermosoil: ThermosoilFirstStepModuleClosure | None,
    snow_state: dict[str, object] | None,
    temp_sol,
    znt,
    zlt,
) -> tuple[dict[str, object], ...]:
    """Map already closed same-step SECHIBA module outputs into STOMATE entry.

    Each source is included only after its module closure reports exact inputs.
    ``temp_sol`` is intentionally not supplied from ``temp_sol_new``: Fortran
    forwards same-step ``temp_sol`` to ``slowproc_main`` before ``sechiba_end``
    copies ``temp_sol_new`` into the next-step state.
    """

    sources: list[dict[str, object]] = []
    if enerbil is not None:
        sources.append(
            stomate_enerbil_entry_source(
                t2mdiag=enerbil.payload.get("t2mdiag"),
                evapot_corr=enerbil.payload.get("evapot_corr"),
                temp_sol=temp_sol,
            )
        )
    if after_diffuco_payload is not None:
        diffuco_source = stomate_diffuco_entry_source(gpp=after_diffuco_payload.get("gpp"))
        if diffuco_source:
            sources.append(diffuco_source)
    if hydrol is not None and hydrol.ok:
        sources.append(
            stomate_hydrol_entry_source(
                diagnostics=hydrol.diagnostics,
                outputs=hydrol.outputs,
            )
        )
    if snow_state is not None:
        snow_source = stomate_snow_entry_source(**snow_state)
        if snow_source:
            sources.append(snow_source)
    if thermosoil is not None and thermosoil.ok:
        sources.extend(
            (
                stomate_thermosoil_entry_source(
                    stempdiag=thermosoil.module.profile.stempdiag,
                    tdeep=thermosoil.module.final.deeptemp_prof,
                    hsdeep=thermosoil.module.final.deephum_prof,
                ),
                stomate_thermosoil_static_entry_source(
                    zz_deep=znt,
                    zz_coef_deep=zlt,
                ),
            )
        )
    return tuple(sources)


def _paper_next_step_entry_payload(
    scaffold: DriverNextStepHydrolPrecallScaffold,
    *,
    run_def_values: dict[str, str],
    runtime: DriverRuntimeScalars,
    znt,
    zlt,
    prepared_context: Paper1961PreparedDriverContext | None = None,
) -> dict[str, object] | None:
    """Assemble a later-step STOMATE entry payload from closed SECHIBA state."""

    if not scaffold.ready_for_next_state:
        return None
    nflow = prepared_context.nflow if prepared_context is not None else parse_run_def_int(run_def_values, "NSTM")
    river_routing = prepared_context.river_routing if prepared_context is not None else parse_run_def_bool(run_def_values["RIVER_ROUTING"])
    driver_source = stomate_driver_entry_source(scaffold.enerbil.payload, kjit=int(scaffold.tstep) + 1)
    routing_source = stomate_no_routing_entry_source(
        kjpindex=scaffold.enerbil.payload.kjpindex,
        nflow=nflow,
        river_routing=river_routing,
        nbp_glo=scaffold.enerbil.payload.kjpindex,
    )
    slowproc_state = scaffold.enerbil.previous_state.fields_by_component["slowproc_stomate_previous_step_state"]
    nvm = int(np.asarray(slowproc_state["lai"]).shape[1])
    run_scalars = scaffold.enerbil.step.run_scalars
    if prepared_context is None:
        vcmax_fix = _select_run_def_pft_vector(
            run_def_values, "VCMAX_FIX", run_scalars, nvm=nvm
        )
        height_presc = _select_run_def_pft_vector(
            run_def_values, "SLOWPROC_HEIGHT", run_scalars, nvm=nvm
        )
        qsintcst = parse_run_def_float(run_def_values, "SECHIBA_QSINT")
        pft_to_mtc = _select_run_def_pft_vector(
            run_def_values, "PFT_TO_MTC", run_scalars, nvm=nvm, dtype=int
        )
        hydrol_humcste = _select_run_def_pft_vector(
            run_def_values, "HYDROL_HUMCSTE", run_scalars, nvm=nvm
        )
        zmaxh = parse_run_def_float(run_def_values, "DEPTH_MAX_H")
        use_age_class = parse_run_def_bool(run_def_values["GLUC_USE_AGE_CLASS"])
        fire_disable = parse_run_def_bool(run_def_values["FIRE_DISABLE"])
        dyn_peat = parse_run_def_bool(run_def_values["DYN_PEAT"])
        topmodel_new = parse_run_def_bool(run_def_values["TOPMODEL_NEW"])
        erosion_module = parse_run_def_bool(run_def_values["EROSION_MODULE"])
        impose_veg = parse_run_def_bool(run_def_values.get("IMPOSE_VEG", "TRUE"))
    else:
        vcmax_fix = _validated_prepared_pft_axis(
            prepared_context.vcmax_fix,
            context=prepared_context,
            nvm=nvm,
            name="VCMAX_FIX",
        )
        height_presc = _validated_prepared_pft_axis(
            prepared_context.slowproc_height,
            context=prepared_context,
            nvm=nvm,
            name="SLOWPROC_HEIGHT",
        )
        qsintcst = prepared_context.sechiba_qsint
        pft_to_mtc = _validated_prepared_pft_axis(
            prepared_context.pft_to_mtc,
            context=prepared_context,
            nvm=nvm,
            name="PFT_TO_MTC",
        )
        hydrol_humcste = _validated_prepared_pft_axis(
            prepared_context.hydrol_humcste,
            context=prepared_context,
            nvm=nvm,
            name="HYDROL_HUMCSTE",
        )
        zmaxh = prepared_context.hydrol_depth_max_h
        use_age_class = prepared_context.gluc_use_age_class
        fire_disable = prepared_context.fire_disable
        dyn_peat = prepared_context.dyn_peat
        topmodel_new = prepared_context.topmodel_new
        erosion_module = prepared_context.erosion_module
        impose_veg = prepared_context.run_scalars.impose_veg
    derivvar = scaffold.enerbil.slowproc_derivvar
    if derivvar is None:
        derivvar = slowproc_derivvar_explicit(
            veget=slowproc_state["veget"],
            lai=slowproc_state["lai"],
            vcmax_fix=vcmax_fix,
            height_presc=height_presc,
            qsintcst=qsintcst,
        )
    no_lcc = slowproc_no_lcc_entry_state(
        veget_max=slowproc_state["veget_max"],
        use_age_class=use_age_class,
        veget_update="0Y",
        map_pft_format=parse_run_def_bool(run_def_values.get("MAP_PFT_FORMAT", "TRUE")),
        impose_veg=impose_veg,
    )
    fire = slowproc_fire_disabled_entry_state(
        kjpindex=scaffold.enerbil.payload.kjpindex,
        fire_disable=fire_disable,
    )
    peat = slowproc_dyn_peat_disabled_entry_state(
        kjpindex=scaffold.enerbil.payload.kjpindex,
        dyn_peat=dyn_peat,
    )
    thermosoil_entry = slowproc_thermosoil_entry_init_state(
        kjpindex=scaffold.enerbil.payload.kjpindex,
        nvm=nvm,
        znt=znt,
        zlt=zlt,
    )
    hydrol_state = scaffold.enerbil.previous_state.fields_by_component["hydrol_previous_step_state"]
    static = slowproc_static_entry_state(
        njsc=hydrol_state["njsc"],
        soil_classif=run_def_values["SOILTYPE_CLASSIF"],
        pft_to_mtc=pft_to_mtc,
        zmaxh=zmaxh,
        hydrol_humcste=hydrol_humcste,
    )
    erosion = slowproc_erosion_daily_zero_entry_state(kjpindex=scaffold.enerbil.payload.kjpindex)
    slowproc_restart_source = {
        "veget": slowproc_state["veget"],
        "veget_max": slowproc_state["veget_max"],
        "lai": slowproc_state["lai"],
        "frac_nobio": slowproc_state["frac_nobio"],
        "height": slowproc_state["height"],
        "frac_age": slowproc_state["frac_age"],
        "totfrac_nobio": _slowproc_totfrac_nobio(slowproc_state),
    }
    stomate_state_source = {
        "assim_param": slowproc_state["assim_param"],
        "altmax": slowproc_state["altmax"],
        "fpeat": slowproc_state["fpeat"],
        "biomass": slowproc_state["biomass"],
        "litter_above": slowproc_state["litter_above"],
        "litter_below": slowproc_state["litter_below"],
        "carbon_32l": slowproc_state["carbon_32l"],
        "DOC": slowproc_state["DOC"],
        "lignin_struc_above": slowproc_state["lignin_struc_above"],
        "lignin_struc_below": slowproc_state["lignin_struc_below"],
        "soilc_total": slowproc_state["soilc_total"],
        "thawed_humidity": slowproc_state["thawed_humidity"],
        "depth_organic_soil": slowproc_state["depth_organic_soil"],
    }
    same_step_sources = _paper_first_step_same_step_entry_sources(
        enerbil=scaffold.enerbil.enerbil_local,
        after_diffuco_payload=scaffold.enerbil.diffuco_local_enerbil_precall.result.payload.payload,
        hydrol=scaffold.hydrol_module,
        thermosoil=scaffold.thermosoil_module,
        snow_state={
            "snow": scaffold.hydrol_module.snow_state.snow,
            "snowdz": scaffold.hydrol_module.snow_state.snowdz,
            "snowrho": scaffold.hydrol_module.snow_state.snowrho,
        },
        temp_sol=scaffold.enerbil.previous_state.fields_by_component["diffuco_previous_step_state"]["temp_sol"],
        znt=znt,
        zlt=zlt,
    )
    fwet_source = {}
    if scaffold.hydrol_module.diagnostics.fwet_new is None and not topmodel_new:
        fwet_source = {"fwet_new": np.asarray(hydrol_state["fwet_new"]).reshape(-1)}
    assembly = assemble_stomate_main_payload(
        driver_source,
        routing_source,
        slowproc_restart_source,
        stomate_slowproc_derivvar_entry_source(derivvar),
        stomate_slowproc_no_lcc_entry_source(no_lcc),
        stomate_slowproc_fire_disabled_entry_source(fire),
        stomate_slowproc_dyn_peat_disabled_entry_source(peat),
        stomate_slowproc_thermosoil_entry_source(thermosoil_entry),
        stomate_slowproc_static_entry_source(static),
        stomate_erosion_disabled_erodepth_entry_source(
            erosion_module=erosion_module,
            kjpindex=scaffold.enerbil.payload.kjpindex,
            nvm=nvm,
        ),
        stomate_slowproc_erosion_daily_zero_entry_source(erosion),
        stomate_state_source,
        *same_step_sources,
        fwet_source,
    )
    blocking_missing = tuple(
        name
        for name in assembly.missing_inputs
        if name not in assembly.missing_by_source.get("io_handle", ())
    )
    if blocking_missing or assembly.partial_inputs:
        return None
    daily_required = (
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
        "fwet_new",
        "liqwt_ratio",
    )
    if any(name not in assembly.payload for name in daily_required):
        return None
    return dict(assembly.payload)


def _slowproc_totfrac_nobio(slowproc_state: Mapping[str, object]):
    if "totfrac_nobio" in slowproc_state:
        return slowproc_state["totfrac_nobio"]
    return np.sum(np.asarray(slowproc_state["frac_nobio"]), axis=1)


def _paper_next_step_runtime_entry_payload(
    scaffold: DriverNextStepHydrolPrecallScaffold,
    *,
    prepared_context: Paper1961PreparedDriverContext,
) -> dict[str, object] | None:
    """Assemble only fields consumed by the runtime daily fold/modelout path."""

    if not scaffold.ready_for_next_state:
        return None
    slowproc_state = scaffold.enerbil.previous_state.fields_by_component["slowproc_stomate_previous_step_state"]
    hydrol_state = scaffold.enerbil.previous_state.fields_by_component["hydrol_previous_step_state"]
    hydrol = scaffold.hydrol_module
    thermosoil = scaffold.thermosoil_module
    if hydrol is None or not hydrol.ok or thermosoil is None or not thermosoil.ok:
        return None
    fwet_new = hydrol.diagnostics.fwet_new
    if fwet_new is None and not prepared_context.topmodel_new:
        fwet_new = hydrol_state["fwet_new"].reshape(-1)
    if fwet_new is None:
        return None
    diffuco_local = scaffold.enerbil.diffuco_local_enerbil_precall
    if diffuco_local is None:
        return None
    diffuco_payload = diffuco_local.result.payload.payload
    hydrol_outputs = hydrol.outputs
    hydrol_diag = hydrol.diagnostics
    t2mdiag = scaffold.enerbil.enerbil_local.payload.get("t2mdiag")
    wspeed = np.maximum(
        prepared_context.min_wind,
        np.sqrt(
            np.asarray(scaffold.enerbil.payload.u) * np.asarray(scaffold.enerbil.payload.u)
            + np.asarray(scaffold.enerbil.payload.v) * np.asarray(scaffold.enerbil.payload.v)
        ),
    )
    return {
        "kjit": int(scaffold.tstep) + 1,
        "lalo": scaffold.enerbil.payload.lalo,
        "precip_rain": scaffold.enerbil.payload.precip_rain,
        "precip_snow": scaffold.enerbil.payload.precip_snow,
        "t2m": t2mdiag,
        "t2m_min": t2mdiag,
        "t2m_max": t2mdiag,
        "wspeed": wspeed,
        "pb": scaffold.enerbil.payload.pb,
        "veget": slowproc_state["veget"],
        "veget_max": slowproc_state["veget_max"],
        "totfrac_nobio": _slowproc_totfrac_nobio(slowproc_state),
        "gpp": diffuco_payload["gpp"],
        "t2mdiag": t2mdiag,
        "evapot_corr": scaffold.enerbil.enerbil_local.payload.get("evapot_corr"),
        "temp_sol": scaffold.enerbil.previous_state.fields_by_component["diffuco_previous_step_state"]["temp_sol"],
        # Fortran boundary rename: sechiba_main passes HYDROL vegstress into
        # slowproc_main's humrel argument, then slowproc_main forwards it to
        # stomate_main. HYDROL humrel remains the DIFFUCO transpiration stress.
        "humrel": hydrol_diag.vegstress,
        "shumdiag": hydrol_diag.shumdiag,
        "litterhumdiag": hydrol_diag.litterhumdiag,
        "soil_mc": hydrol_diag.mc_layh_s,
        "wat_flux": hydrol_outputs.wat_flux,
        "runoff_per_soil": hydrol_outputs.runoff_per_soil,
        "drainage_per_soil": hydrol_outputs.drainage_per_soil,
        "runoff2peat": hydrol_outputs.runoff2peat,
        "canopy2ground": hydrol_outputs.canopy2ground,
        "precip2ground": hydrol_outputs.precip2ground,
        "precip2canopy": hydrol_outputs.precip2canopy,
        "wtp": hydrol_diag.wtp,
        "fwet_new": fwet_new,
        "liqwt_ratio": hydrol_diag.liqwt_ratio,
        "snow": hydrol.snow_state.snow,
        "snowdz": hydrol.snow_state.snowdz,
        "snowrho": hydrol.snow_state.snowrho,
        "stempdiag": thermosoil.module.profile.stempdiag,
        "tdeep": thermosoil.module.final.deeptemp_prof,
        "hsdeep": thermosoil.module.final.deephum_prof,
        "zz_deep": prepared_context.cwrr_grid.znt,
        "zz_coef_deep": prepared_context.cwrr_grid.zlt,
        "tmc_topgrass": hydrol_diag.tmc_topgrass,
        "shumdiag_peat": hydrol_diag.shumdiag_peat,
        "mc_peat_above": hydrol_diag.mc_peat_above,
        "shumdiag_croppeat": hydrol_diag.shumdiag_croppeat,
        "mc_croppeat_above": hydrol_diag.mc_croppeat_above,
        "shumdiag_man": hydrol_diag.shumdiag_man,
        "mc_man_above": hydrol_diag.mc_man_above,
    }


def _paper_next_step_runtime_entry_payload_from_components(
    *,
    tstep: int,
    driver_payload: IntersurfFirstStepPayload,
    previous_state: DriverPreviousStepStatePacket,
    diffuco_local: DiffucoPFT14FirstStepLocalEnerbilPrecallAssembly | None,
    enerbil_local: EnerbilFirstStepLocalAssembly | None,
    hydrol: HydrolFirstStepModuleClosure | None,
    thermosoil: ThermosoilFirstStepModuleClosure | None,
    prepared_context: Paper1961PreparedDriverContext,
) -> dict[str, object] | None:
    """Assemble runtime STOMATE entry fields directly from closed components."""

    slowproc_state = _previous_state_component_fields(previous_state, "slowproc_stomate_previous_step_state")
    hydrol_state = _previous_state_component_fields(previous_state, "hydrol_previous_step_state")
    if (
        diffuco_local is None
        or enerbil_local is None
        or hydrol is None
        or not hydrol.ok
        or thermosoil is None
        or not thermosoil.ok
    ):
        return None
    fwet_new = hydrol.diagnostics.fwet_new
    if fwet_new is None and not prepared_context.topmodel_new:
        fwet_new = hydrol_state["fwet_new"].reshape(-1)
    if fwet_new is None:
        return None
    diffuco_payload = diffuco_local.result.payload.payload
    hydrol_outputs = hydrol.outputs
    hydrol_diag = hydrol.diagnostics
    t2mdiag = enerbil_local.payload.get("t2mdiag")
    use_jax = any(isinstance(value, jax.core.Tracer) for value in (driver_payload.u, driver_payload.v))
    array = jnp.asarray if use_jax else np.asarray
    maximum = jnp.maximum if use_jax else np.maximum
    sqrt = jnp.sqrt if use_jax else np.sqrt
    u = array(driver_payload.u)
    v = array(driver_payload.v)
    wspeed = maximum(prepared_context.min_wind, sqrt(u * u + v * v))
    return {
        "kjit": int(tstep) + 1,
        "lalo": driver_payload.lalo,
        "precip_rain": driver_payload.precip_rain,
        "precip_snow": driver_payload.precip_snow,
        "t2m": t2mdiag,
        "t2m_min": t2mdiag,
        "t2m_max": t2mdiag,
        "wspeed": wspeed,
        "pb": driver_payload.pb,
        "veget": slowproc_state["veget"],
        "veget_max": slowproc_state["veget_max"],
        "totfrac_nobio": _slowproc_totfrac_nobio(slowproc_state),
        "gpp": diffuco_payload["gpp"],
        "t2mdiag": t2mdiag,
        "evapot_corr": enerbil_local.payload.get("evapot_corr"),
        "temp_sol": _previous_state_component_fields(previous_state, "diffuco_previous_step_state")["temp_sol"],
        # Fortran boundary rename: sechiba_main passes HYDROL vegstress into
        # slowproc_main's humrel argument, then slowproc_main forwards it to
        # stomate_main. HYDROL humrel remains the DIFFUCO transpiration stress.
        "humrel": hydrol_diag.vegstress,
        "shumdiag": hydrol_diag.shumdiag,
        "litterhumdiag": hydrol_diag.litterhumdiag,
        "soil_mc": hydrol_diag.mc_layh_s,
        "wat_flux": hydrol_outputs.wat_flux,
        "runoff_per_soil": hydrol_outputs.runoff_per_soil,
        "drainage_per_soil": hydrol_outputs.drainage_per_soil,
        "runoff2peat": hydrol_outputs.runoff2peat,
        "canopy2ground": hydrol_outputs.canopy2ground,
        "precip2ground": hydrol_outputs.precip2ground,
        "precip2canopy": hydrol_outputs.precip2canopy,
        "wtp": hydrol_diag.wtp,
        "fwet_new": fwet_new,
        "liqwt_ratio": hydrol_diag.liqwt_ratio,
        "snow": hydrol.snow_state.snow,
        "snowdz": hydrol.snow_state.snowdz,
        "snowrho": hydrol.snow_state.snowrho,
        "stempdiag": thermosoil.module.profile.stempdiag,
        "tdeep": thermosoil.module.final.deeptemp_prof,
        "hsdeep": thermosoil.module.final.deephum_prof,
        "zz_deep": prepared_context.cwrr_grid.znt,
        "zz_coef_deep": prepared_context.cwrr_grid.zlt,
        "tmc_topgrass": hydrol_diag.tmc_topgrass,
        "shumdiag_peat": hydrol_diag.shumdiag_peat,
        "mc_peat_above": hydrol_diag.mc_peat_above,
        "shumdiag_croppeat": hydrol_diag.shumdiag_croppeat,
        "mc_croppeat_above": hydrol_diag.mc_croppeat_above,
        "shumdiag_man": hydrol_diag.shumdiag_man,
        "mc_man_above": hydrol_diag.mc_man_above,
    }


def _runtime_next_state_from_components(
    *,
    tstep: int,
    previous_state: DriverPreviousStepStatePacket,
    enerbil_local: EnerbilFirstStepLocalAssembly | None,
    diffuco_local: DiffucoPFT14FirstStepLocalEnerbilPrecallAssembly | None,
    diffuco_day_static_cache: DriverDiffucoDayStaticCache | None = None,
    thermosoil_module: ThermosoilFirstStepModuleClosure | None,
    condveg_module: CondvegFirstStepModuleClosure | None,
    hydrol_module: HydrolFirstStepModuleClosure | None,
    ok_laidev: tuple[bool, ...] | None = None,
) -> DriverPreviousStepStatePacket:
    """Extract source-backed next-step state from compact runtime components."""

    fields = _empty_previous_step_state_fields()
    _add_enerbil_local_to_previous_fields(fields, enerbil_local)
    _add_diffuco_local_to_previous_fields(fields, diffuco_local)
    if diffuco_day_static_cache is not None:
        _add_diffuco_saved_controls_to_previous_fields(
            fields,
            control_salinity=diffuco_day_static_cache.control_salinity,
            control_inundation=diffuco_day_static_cache.control_inundation,
        )
    _add_diffuco_saved_controls_to_previous_fields(fields, precall=getattr(diffuco_local, "kwargs", None))
    _add_thermosoil_to_previous_fields(fields, thermosoil_module)
    _add_condveg_to_previous_fields(fields, condveg_module)
    _add_hydrol_to_previous_fields(fields, hydrol_module)
    prior_finalize = _previous_state_component_fields(previous_state, "sechiba_finalize_state")
    if prior_finalize:
        fields["sechiba_finalize_state"] = _sechiba_finalize_state_from_components(
            previous=prior_finalize,
            enerbil_local=enerbil_local,
            diffuco_local=diffuco_local,
            thermosoil_module=thermosoil_module,
            condveg_module=condveg_module,
            hydrol_module=hydrol_module,
        )
    prior_slowproc = _previous_state_component_fields(previous_state, "slowproc_stomate_previous_step_state")
    for name in STOMATE_HALF_HOUR_CARRY_FIELDS:
        if name in prior_slowproc:
            fields["slowproc_stomate_previous_step_state"][name] = prior_slowproc[name]
    for name in ("veget", "veget_max", "lai", "frac_nobio", "height"):
        if name in prior_slowproc:
            fields["diffuco_previous_step_state"][name] = prior_slowproc[name]
    _write_maintenance_lai_to_previous_fields(fields, prior_slowproc, ok_laidev)
    prior_therm = _previous_state_component_fields(previous_state, "thermosoil_previous_step_state")
    for name in ("refSOC", "e_soil_lat"):
        if name in prior_therm:
            fields["thermosoil_previous_step_state"][name] = prior_therm[name]
    prior_hydrol = _previous_state_component_fields(previous_state, "hydrol_previous_step_state")
    for name in ("free_drain_coef", "zwt_force", "njsc", "fwet_new"):
        if name in prior_hydrol:
            fields["hydrol_previous_step_state"][name] = prior_hydrol[name]
    prior_diffuco = _previous_state_component_fields(previous_state, "diffuco_previous_step_state")
    if "soilalbedo_bg" in prior_diffuco:
        fields["diffuco_previous_step_state"]["soilalbedo_bg"] = prior_diffuco["soilalbedo_bg"]
    if "height" in prior_diffuco:
        fields["diffuco_previous_step_state"]["height"] = prior_diffuco["height"]
    for name in ("control_salinity", "control_inudate"):
        if name in prior_diffuco and name not in fields["diffuco_previous_step_state"]:
            fields["diffuco_previous_step_state"][name] = prior_diffuco[name]
    prior_driver = _previous_state_component_fields(previous_state, "driver_previous_step_state")
    if "albedo" in prior_driver and "albedo" not in fields.get("driver_previous_step_state", {}):
        fields.setdefault("driver_previous_step_state", {})["albedo"] = prior_driver["albedo"]
    if hasattr(previous_state, "component_fields"):
        return fast_state_from_previous_fields(
            tstep=int(tstep),
            fields_by_component=fields,
            provenance_by_component=_previous_step_state_provenance(),
        )
    return _packet_from_previous_step_fields(tstep=int(tstep), fields=fields)


def _paper_1961_next_step_runtime_result_compact(
    config_path: str | Path,
    *,
    previous_state: DriverPreviousStepStatePacket,
    year: int,
    tstep: int,
    fixed_format_trace_dir: str | Path | None,
    static_trace_fields: StaticTraceFields | None,
    used_run_def_path: str | Path | None,
    prepared_context: Paper1961PreparedDriverContext,
    hydrol_static_template: dict[str, object] | None,
    hydrol_runtime_static_tables=None,
    diffuco_day_static_cache: DriverDiffucoDayStaticCache | None,
    compiled_diffuco_day_inputs: DriverCompiledDiffucoDayInputs | None = None,
    compiled_diffuco_static_kwargs: Mapping[str, object] | None = None,
    module_jit: bool,
    diffuco_local_jit: bool,
    prebuilt_input: DriverRuntimeHalfHourInput | None = None,
    prebuilt_step: DriverStepBundle | None = None,
    prebuilt_base_payload: IntersurfFirstStepPayload | None = None,
    runtime_run_scalars: RunScalars | None = None,
    runtime_forcing=None,
) -> DriverRuntimeStepResult:
    """Advance one half-hour step without retaining audit scaffold objects."""

    if int(tstep) < 0:
        raise ValueError("next-step runtime requires tstep >= 0")
    if previous_state.tstep != int(tstep) - 1:
        raise ValueError("previous_state.tstep must be exactly tstep - 1")

    context = prepared_context
    if runtime_run_scalars is not None:
        if runtime_forcing is None or prebuilt_base_payload is None:
            raise ValueError("runtime_run_scalars requires runtime_forcing and prebuilt_base_payload")
        step = SimpleNamespace(run_scalars=runtime_run_scalars, forcing=runtime_forcing)
    elif prebuilt_input is not None:
        prebuilt_step = prebuilt_input.step
        prebuilt_base_payload = prebuilt_input.base_payload
    if runtime_run_scalars is not None:
        pass
    elif prebuilt_step is not None:
        if int(prebuilt_step.tstep) != int(tstep):
            raise ValueError("prebuilt_step.tstep must match tstep")
        step = prebuilt_step
    else:
        step = (
            _paper_1961_step_bundle_from_context(context, year=year, tstep=tstep)
            if fixed_format_trace_dir is None and static_trace_fields is None
            else load_paper_1961_step_bundle(
                config_path,
                year=year,
                tstep=tstep,
                run_def_path=context.run_def_path,
                reference_run_dir=context.reference_run_dir,
                fixed_format_trace_dir=fixed_format_trace_dir,
                static_trace_fields=static_trace_fields,
                domain_override=context.domain_override,
            )
        )
    driver_wind_z0 = _driver_wind_z0_from_previous_state(previous_state)
    payload = (
        intersurf_payload_with_driver_wind_z0(
            prebuilt_base_payload,
            forcing=step.forcing,
            driver_z0_for_wind=driver_wind_z0,
        )
        if prebuilt_base_payload is not None
        else build_intersurf_first_step_payload(
            step,
            driver_z0_for_wind=driver_wind_z0,
            dt_sechiba=context.dt_sechiba,
        )
    )
    restart_payload, used_diffuco_previous, missing_diffuco_previous = _diffuco_restart_payload_from_previous_state(previous_state)
    del used_diffuco_previous
    slowproc_state = _previous_state_component_fields(previous_state, "slowproc_stomate_previous_step_state")
    biomass = slowproc_state.get("biomass")
    if biomass is not None:
        nvm = int(biomass.shape[1])
    elif "veget" in slowproc_state:
        nvm = int(slowproc_state["veget"].shape[1])
    else:
        nvm = context.nvm

    if compiled_diffuco_day_inputs is not None:
        diffuco_control_salinity = SimpleNamespace(
            payload={"control_salinity": compiled_diffuco_day_inputs.control_salinity}
        )
        diffuco_control_inundation = SimpleNamespace(
            payload={"control_inudate": compiled_diffuco_day_inputs.control_inudate}
        )
        slowproc_derivvar = None
        slowproc_derivvar_payload = {
            "qsintmax": compiled_diffuco_day_inputs.qsintmax,
            "assim_param": compiled_diffuco_day_inputs.assim_param,
            "height": compiled_diffuco_day_inputs.height,
            "temp_growth": compiled_diffuco_day_inputs.temp_growth,
        }
        diffuco_day_static_cache = SimpleNamespace(
            control_salinity=diffuco_control_salinity,
            control_inundation=diffuco_control_inundation,
        )
    elif diffuco_day_static_cache is not None:
        diffuco_control_salinity = diffuco_day_static_cache.control_salinity
        diffuco_control_inundation = diffuco_day_static_cache.control_inundation
        slowproc_derivvar = diffuco_day_static_cache.slowproc_derivvar
        slowproc_derivvar_payload = diffuco_day_static_cache.slowproc_derivvar_payload
    else:
        diffuco_control_salinity = assemble_pft14_control_salinity_first_step(
            salinity=payload.salinity,
            control_salinity_min=context.control_salinity_min,
        )
        diffuco_control_inundation = _paper_pft14_control_inundation_from_prepared_context(
            context,
            tide_height=payload.tide_height,
            biomass=biomass,
        )
        slowproc_derivvar, slowproc_derivvar_payload = _paper_later_day_slowproc_derivvar_payload(
            slowproc_state,
            vcmax_fix=_validated_prepared_pft_axis(
                context.vcmax_fix, context=context, nvm=nvm, name="VCMAX_FIX"
            ),
            height_presc=_validated_prepared_pft_axis(
                context.slowproc_height, context=context, nvm=nvm, name="SLOWPROC_HEIGHT"
            ),
            qsintcst=context.sechiba_qsint,
        )

    diffuco_precall = assemble_diffuco_first_step_precall_payload(
        driver_or_intersurf_payload={
            "u": payload.u,
            "v": payload.v,
            "zlev": payload.zlev,
            "temp_air": payload.temp_air,
            "qair": payload.qair,
            "pb": payload.pb,
            "swdown": payload.swdown,
            "ccanopy": payload.ccanopy,
            "precip_rain": payload.precip_rain,
            "lalo": payload.lalo,
            "neighbours": payload.neighbours,
            "resolution": payload.resolution,
        },
        restart_payload=restart_payload,
        driver_albedo_payload=None,
        slowproc_derivvar_payload=slowproc_derivvar_payload,
        diffuco_control_salinity=diffuco_control_salinity,
        diffuco_control_inundation=diffuco_control_inundation,
        ok_explicitsnow=context.ok_explicitsnow,
        river_routing=context.river_routing,
        nbp_glo=payload.kjpindex,
    )
    thermal_payload, _, missing_enerbil_previous = _enerbil_state_payload_from_previous_state(previous_state)
    driver_albedo_payload = {}
    driver_fields = _previous_state_component_fields(previous_state, "driver_previous_step_state")
    state_fields = _previous_state_component_fields(previous_state, "enerbil_previous_step_state")
    if "albedo" in driver_fields:
        driver_albedo_payload["albedo"] = driver_fields["albedo"]
    elif "albedo" in state_fields:
        driver_albedo_payload["albedo"] = state_fields["albedo"]

    combined_missing = tuple(dict.fromkeys((*missing_diffuco_previous, *missing_enerbil_previous)))
    diffuco_local = None
    after_diffuco_payload = diffuco_precall.payload
    if diffuco_precall.ok and not missing_diffuco_previous:
        diffuco_local = run_pft14_local_enerbil_precall_from_first_step_precall(
            diffuco_precall,
            run_def_values=context.run_def_values,
            pft_index=_paper_mangrove_pft_index(context.run_scalars),
            static_kwargs=(
                context.diffuco_pft14_static_kwargs
                if compiled_diffuco_static_kwargs is None
                else compiled_diffuco_static_kwargs
            ),
            use_jit=diffuco_local_jit,
        )
        after_diffuco_payload = diffuco_local.result.payload.payload
    enerbil_precall = assemble_enerbil_first_step_precall_payload(
        driver_or_intersurf_payload=_enerbil_driver_coverage_payload(payload),
        driver_albedo_payload=driver_albedo_payload,
        soil_thermal_restart_payload=thermal_payload,
        after_diffuco_payload=after_diffuco_payload,
        condveg_impaze=False,
        non_watchout_driver_executed=True,
    )
    enerbil_local = None
    if enerbil_precall.ok and not combined_missing:
        enerbil_local = run_enerbil_first_step_local_from_precall(
            enerbil_precall,
            ok_laidev=_validated_prepared_pft_axis(
                context.ok_laidev, context=context, nvm=nvm, name="OK_LAIDEV"
            ),
            dt_sechiba=context.dt_sechiba,
            ok_explicitsnow=context.ok_explicitsnow,
            min_wind=context.min_wind,
            use_jit=bool(module_jit),
        )

    hydrol_state, missing_hydrol_previous = _hydrol_state_namespace_from_previous_state(previous_state)
    combined_missing = tuple(dict.fromkeys((*combined_missing, *missing_hydrol_previous)))
    hydrol_module = None
    hydrol_precall = None
    condveg_module = None
    thermosoil_module = None
    if enerbil_local is not None:
        hydrol_humcste = _validated_prepared_pft_axis(
            context.hydrol_humcste, context=context, nvm=nvm, name="HYDROL_HUMCSTE"
        )
        hydrol_zz_mm = context.hydrol_zz_mm
        if hydrol_static_template is None:
            hydrol_static_template = hydrol_static_precall_template_from_slowproc(
                slowproc_restart=SimpleNamespace(**slowproc_state),
                pref_soil_veg=step.run_scalars.pref_soil_veg,
                nstm=step.run_scalars.nstm,
            ext_coeff_vegetfrac=_validated_prepared_pft_axis(
                context.ext_coeff_vegetfrac,
                context=context,
                nvm=nvm,
                name="EXT_COEFF_VEGETFRAC",
            ),
            hydrol_njsc=getattr(hydrol_state, "njsc", None),
            refSOC_1d=getattr(hydrol_state, "refSOC_1d", None),
            use_refSOC_hydrol=parse_run_def_bool(context.run_def_values.get("use_refSOC_hydrol", "FALSE")),
            throughfall_by_pft=_validated_prepared_pft_axis(
                context.hydrol_throughfall_by_pft,
                context=context,
                nvm=nvm,
                name="PERCENT_THROUGHFALL_PFT",
            ),
            humcste=hydrol_humcste,
            zz_mm=hydrol_zz_mm,
                peat_hydro=context.hydrol_soil_peat_hydro,
                ok_dgvm=context.stomate_ok_dgvm,
                dt_days=context.dt_sechiba_days,
            )
        hydrol_precall = assemble_hydrol_first_step_precall_payload(
            enerbil_payload=enerbil_local.payload,
            hydrol_restart=hydrol_state,
            slowproc_restart=SimpleNamespace(**slowproc_state),
            pref_soil_veg=step.run_scalars.pref_soil_veg,
            nstm=step.run_scalars.nstm,
            ext_coeff_vegetfrac=_validated_prepared_pft_axis(
                context.ext_coeff_vegetfrac,
                context=context,
                nvm=nvm,
                name="EXT_COEFF_VEGETFRAC",
            ),
            precip_rain=payload.precip_rain,
            precip_snow=payload.precip_snow,
            diffuco_payload=_hydrol_current_diffuco_payload(diffuco_precall.payload, after_diffuco_payload),
            thermosoil_restart=SimpleNamespace(
                **{
                    name: _previous_state_component_fields(previous_state, "thermosoil_previous_step_state")[name]
                    for name in ("ptn", "gtemp", "lambda_snow", "cgrnd_snow", "dgrnd_snow")
                    if name in _previous_state_component_fields(previous_state, "thermosoil_previous_step_state")
                }
            ),
            throughfall_by_pft=_validated_prepared_pft_axis(
                context.hydrol_throughfall_by_pft,
                context=context,
                nvm=nvm,
                name="PERCENT_THROUGHFALL_PFT",
            ),
            humcste=hydrol_humcste,
            zz_mm=hydrol_zz_mm,
            diaglev_m=context.diaglev,
            peat_hydro=context.hydrol_soil_peat_hydro,
            ok_dgvm=context.stomate_ok_dgvm,
            ok_explicitsnow=context.ok_explicitsnow,
            ok_freeze_cwrr=context.ok_freeze_cwrr,
            ok_thermodynamical_freezing=context.ok_thermodynamical_freezing,
            fr_dt=context.hydrol_fr_dt,
            froz_frac_corr=context.hydrol_froz_frac_corr,
            smtot_corr=context.hydrol_smtot_corr,
            max_froz_hydro=context.hydrol_max_froz_hydro,
            dt_days=context.dt_sechiba_days,
            static_payload_template=hydrol_static_template,
            use_jit=bool(module_jit),
        )
        if hydrol_precall.ok and not combined_missing:
            hydrol_module = run_hydrol_first_step_module_from_precall(
                hydrol_precall,
                temp_air=payload.temp_air,
                pb=payload.pb,
                u=payload.u,
                v=payload.v,
                humcste=hydrol_humcste,
                nroot_state=getattr(hydrol_state, "nroot", None),
                dz_mm=context.hydrol_dz_mm,
                dh_mm=context.hydrol_dh_mm,
                zz_mm=hydrol_zz_mm,
                altmax=slowproc_state.get("altmax"),
                ks=context.hydrol_cwrr_ks,
                reinf_slope=context.hydrol_reinf_slope,
                zmaxh_m=context.hydrol_depth_max_h,
                doponds=context.hydrol_do_ponds,
                peat_hydro=context.hydrol_soil_peat_hydro,
                peat_hydro_water_stress=context.hydrol_soil_peat_hydro,
                ok_freeze_cwrr=context.ok_freeze_cwrr,
                ok_pc=context.hydrol_ok_pc,
                ok_leak=context.hydrol_ok_leak,
                ok_ru2peat=context.hydrol_ok_ru2peat,
                ok_wt_ab=context.hydrol_ok_wt_ab,
                tides=context.hydrol_tides,
                agri_peat=context.hydrol_agri_peat,
                max_wt_ab=context.hydrol_max_wt_ab,
                dyn_nroot_larix=context.hydrol_dyn_nroot_larix,
                is_tree=step.run_scalars.is_tree,
                znt=context.diaglev,
                run2peat=getattr(hydrol_state, "run2peat", None),
                run2man=getattr(hydrol_state, "run2man", None),
                dt_sechiba=context.dt_sechiba,
                use_jit=bool(module_jit),
                static_table_payload=hydrol_static_template,
                runtime_static_tables=hydrol_runtime_static_tables,
        )
        if hydrol_module is not None and hydrol_module.ok:
            diffuco_state = _previous_state_component_fields(previous_state, "diffuco_previous_step_state")
            hydrol_prev = _previous_state_component_fields(previous_state, "hydrol_previous_step_state")
            therm_prev = _previous_state_component_fields(previous_state, "thermosoil_previous_step_state")
            condveg_module = run_condveg_first_step_module(
                restart_payload={
                    "snow": hydrol_module.snow_state.snow,
                    "snow_nobio": hydrol_module.snow_state.snow_nobio,
                    "snowrho": hydrol_module.snow_state.snowrho,
                    "snowdz": hydrol_module.snow_state.snowdz,
                    "snow_age": hydrol_module.snow_state.snow_age,
                    "snow_nobio_age": hydrol_module.snow_state.snow_nobio_age,
                    "veget": slowproc_state["veget"],
                    "veget_max": slowproc_state["veget_max"],
                    "frac_nobio": slowproc_state["frac_nobio"],
                    "height": diffuco_state["height"],
                    "lai": slowproc_state["lai"],
                    "soilalbedo_bg": diffuco_state["soilalbedo_bg"],
                },
                driver_payload={
                    "zlev": payload.zlev,
                    "temp_air": payload.temp_air,
                    "pb": payload.pb,
                    "u": payload.u,
                    "v": payload.v,
                },
                slowproc_payload={
                    "totfrac_nobio": _slowproc_totfrac_nobio(slowproc_state),
                    "tot_bare_soil": slowproc_state["veget_max"][:, 0]
                    + (slowproc_state["veget_max"][:, 1:] - slowproc_state["veget"][:, 1:]).sum(axis=1),
                    "is_tree": step.run_scalars.is_tree,
                    "z0_over_height": step.run_scalars.z0_over_height,
                    "ratio_z0m_z0h": step.run_scalars.ratio_z0m_z0h,
                },
                hydrol_diagnostics=hydrol_module.diagnostics,
                run_def_values=context.run_def_values,
                ok_explicitsnow=context.ok_explicitsnow,
                impaze=context.impaze,
                rough_dyn=context.rough_dyn,
                use_jit=bool(module_jit),
            )
            if condveg_module.ok:
                thermosoil_module = run_thermosoil_first_step_module(
                    moisture=hydrol_module.thermosoil_moisture,
                    thermosoil_restart=SimpleNamespace(
                        ptn=therm_prev["ptn"],
                        cgrnd=therm_prev["cgrnd"],
                        dgrnd=therm_prev["dgrnd"],
                        cgrnd_snow=therm_prev["cgrnd_snow"],
                        dgrnd_snow=therm_prev["dgrnd_snow"],
                        lambda_snow=therm_prev["lambda_snow"],
                        gtemp=therm_prev["gtemp"],
                        temp_sol_beg=therm_prev.get("temp_sol_beg", therm_prev["gtemp"]),
                        pcapa_en=therm_prev.get("pcapa_en"),
                        refsoc=therm_prev["refSOC"],
                        shum_ngrnd_permalong=therm_prev.get("shum_ngrnd_permalong"),
                    ),
                    enerbil_payload=enerbil_local.payload,
                    condveg_result=condveg_module.module,
                    snowrho=hydrol_module.snow_state.snowrho,
                    snowtemp=hydrol_module.snow_state.snowtemp,
                    snowdz=hydrol_module.snow_state.snowdz,
                    njsc=hydrol_prev["njsc"],
                    veget_max=slowproc_state["veget_max"],
                    pb=payload.pb,
                    totfrac_nobio=_slowproc_totfrac_nobio(slowproc_state),
                    dlt=context.cwrr_grid.dlt,
                    dz1=context.cwrr_grid.dz1,
                    zlt=context.cwrr_grid.zlt,
                    znt=context.cwrr_grid.znt,
                    dz5=context.cwrr_grid.dz5,
                    dt_sechiba=context.dt_sechiba,
                    ok_laidev=_validated_prepared_pft_axis(
                        context.ok_laidev, context=context, nvm=nvm, name="OK_LAIDEV"
                    ),
                    satsoil=context.thermosoil_satsoil,
                    ok_shum_ngrnd_permalong=_effective_thermosoil_wetdiaglong(context.run_def_values),
                    use_jit=bool(module_jit),
                )

    ready = (
        enerbil_local is not None
        and hydrol_precall is not None
        and hydrol_precall.ok
        and hydrol_module is not None
        and hydrol_module.ok
        and condveg_module is not None
        and condveg_module.ok
        and thermosoil_module is not None
        and thermosoil_module.ok
        and not combined_missing
    )
    runtime_readiness_missing: list[str] = []
    if enerbil_local is None:
        runtime_readiness_missing.append("runtime_enerbil_local")
    if hydrol_precall is None or not hydrol_precall.ok:
        runtime_readiness_missing.append("runtime_hydrol_precall")
    if hydrol_module is None or not hydrol_module.ok:
        runtime_readiness_missing.append("runtime_hydrol_module")
        if hydrol_module is not None:
            runtime_readiness_missing.extend(
                f"runtime_hydrol_module:{name}" for name in hydrol_module.missing_inputs
            )
    if condveg_module is None or not condveg_module.ok:
        runtime_readiness_missing.append("runtime_condveg_module")
    if thermosoil_module is None or not thermosoil_module.ok:
        runtime_readiness_missing.append("runtime_thermosoil_module")
    entry_payload = (
        _paper_next_step_runtime_entry_payload_from_components(
            tstep=tstep,
            driver_payload=payload,
            previous_state=previous_state,
            diffuco_local=diffuco_local,
            enerbil_local=enerbil_local,
            hydrol=hydrol_module,
            thermosoil=thermosoil_module,
            prepared_context=context,
        )
        if ready
        else None
    )
    if entry_payload is None:
        if ready:
            runtime_readiness_missing.append("runtime_entry_payload_fields")
        missing = tuple(
            dict.fromkeys((*combined_missing, *runtime_readiness_missing, "runtime_entry_payload"))
        )
        return DriverRuntimeStepResult(
            tstep=int(tstep),
            entry_payload=None,
            next_state=None,
            metadata=None,
            missing_components=missing,
        )
    next_state = _runtime_next_state_from_components(
        tstep=tstep,
        previous_state=previous_state,
        enerbil_local=enerbil_local,
        diffuco_local=diffuco_local,
        diffuco_day_static_cache=diffuco_day_static_cache,
        thermosoil_module=thermosoil_module,
        condveg_module=condveg_module,
        hydrol_module=hydrol_module,
        ok_laidev=context.ok_laidev,
    )
    metadata = DriverRuntimeStepMetadata(
        pref_soil_veg=step.run_scalars.pref_soil_veg,
        nstm=step.run_scalars.nstm,
        is_tree=step.run_scalars.is_tree,
        is_peat=step.run_scalars.is_peat,
        lalo=payload.lalo,
        npts=payload.kjpindex,
    )
    return DriverRuntimeStepResult(
        tstep=int(tstep),
        entry_payload=entry_payload,
        next_state=next_state,
        metadata=metadata,
        missing_components=(),
    )


def paper_1961_next_step_runtime_result(
    config_path: str | Path,
    *,
    previous_state: DriverPreviousStepStatePacket,
    year: int = 1961,
    tstep: int = 1,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    used_run_def_path: str | Path | None = None,
    prepared_context: Paper1961PreparedDriverContext | None = None,
    hydrol_static_template: dict[str, object] | None = None,
    diffuco_day_static_cache: DriverDiffucoDayStaticCache | None = None,
    prebuilt_step: DriverStepBundle | None = None,
    prebuilt_base_payload: IntersurfFirstStepPayload | None = None,
    module_jit: bool = True,
    diffuco_local_jit: bool = True,
) -> DriverRuntimeStepResult:
    """Advance one half-hour step and return only runtime payload/state.

    This currently delegates to the audited next-step scaffold internally, then
    drops the debug object after extracting the modeled next state and STOMATE
    runtime entry payload. It defines the compact boundary that can later be
    replaced by a pure-array compiled state transition.
    """

    context = prepared_context or prepare_paper_1961_driver_context(config_path, used_run_def_path=used_run_def_path)
    if prepared_context is not None:
        return _paper_1961_next_step_runtime_result_compact(
            config_path,
            previous_state=previous_state,
            year=year,
            tstep=tstep,
            fixed_format_trace_dir=fixed_format_trace_dir,
            static_trace_fields=static_trace_fields,
            used_run_def_path=used_run_def_path,
            prepared_context=context,
            hydrol_static_template=hydrol_static_template,
            diffuco_day_static_cache=diffuco_day_static_cache,
            module_jit=module_jit,
            diffuco_local_jit=diffuco_local_jit,
            prebuilt_step=prebuilt_step,
            prebuilt_base_payload=prebuilt_base_payload,
        )
    scaffold = paper_1961_next_step_hydrol_precall_scaffold(
        config_path,
        previous_state=previous_state,
        year=year,
        tstep=tstep,
        fixed_format_trace_dir=fixed_format_trace_dir,
        static_trace_fields=static_trace_fields,
        used_run_def_path=context.run_def_path,
        prepared_context=context,
        hydrol_static_template=hydrol_static_template,
        diffuco_day_static_cache=diffuco_day_static_cache,
        module_jit=module_jit,
        diffuco_local_jit=diffuco_local_jit,
    )
    entry_payload = _paper_next_step_runtime_entry_payload(
        scaffold,
        prepared_context=context,
    )
    if entry_payload is None or not scaffold.ready_for_next_state:
        missing = tuple(dict.fromkeys((*scaffold.missing_previous_state_fields, "runtime_entry_payload")))
        return DriverRuntimeStepResult(
            tstep=int(tstep),
            entry_payload=None,
            next_state=None,
            metadata=None,
            missing_components=missing,
        )
    next_state = driver_previous_step_state_from_next_step_hydrol_scaffold(
        scaffold,
        ok_laidev=context.ok_laidev,
    )
    metadata = DriverRuntimeStepMetadata(
        pref_soil_veg=scaffold.enerbil.step.run_scalars.pref_soil_veg,
        nstm=scaffold.enerbil.step.run_scalars.nstm,
        is_tree=scaffold.enerbil.step.run_scalars.is_tree,
        is_peat=scaffold.enerbil.step.run_scalars.is_peat,
        lalo=scaffold.enerbil.payload.lalo,
        npts=scaffold.enerbil.payload.kjpindex,
    )
    return DriverRuntimeStepResult(
        tstep=int(tstep),
        entry_payload=entry_payload,
        next_state=next_state,
        metadata=metadata,
        missing_components=(),
    )


def _paper_first_step_fwet_entry_source(
    *,
    hydrol: HydrolFirstStepModuleClosure | None,
    restart_state: ReferenceCaseFirstStepRestartState,
    topmodel_new: bool,
) -> dict[str, object]:
    """Return the exact first-step ``fwet_new`` source when HYDROL leaves it unchanged."""

    if hydrol is not None and hydrol.ok and hydrol.diagnostics.fwet_new is not None:
        return {"fwet_new": hydrol.diagnostics.fwet_new}
    if not bool(topmodel_new) and restart_state.hydrol.fwet_new is not None:
        return {"fwet_new": restart_state.hydrol.fwet_new.reshape(-1)}
    return {}


def paper_1961_driver_timestep_scaffold(
    config_path: str | Path,
    *,
    year: int = 1961,
    tstep: int = 0,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    used_run_def_path: str | Path | None = None,
    reference_run_dir: str | Path | None = None,
    root: str | Path | None = None,
    prepared_context: Paper1961PreparedDriverContext | None = None,
) -> DriverTimestepScaffold:
    """Build strict source-backed boundary objects for one driver timestep.

    This is not a runnable model loop. It assembles the exact pieces that are
    source-backed at the driver boundary and names the remaining prognostic
    state needed before SECHIBA/STOMATE can be advanced without traces.
    """

    context = prepared_context or prepare_paper_1961_driver_context(
        config_path,
        used_run_def_path=used_run_def_path,
        reference_run_dir=reference_run_dir,
    )
    run_def_path = context.run_def_path
    runtime = context.runtime
    run_def_values = context.run_def_values
    step = (
        _paper_1961_step_bundle_from_context(context, year=year, tstep=tstep)
        if fixed_format_trace_dir is None and static_trace_fields is None
        else load_paper_1961_step_bundle(
            config_path,
            year=year,
            tstep=tstep,
            run_def_path=run_def_path,
            reference_run_dir=context.reference_run_dir,
            fixed_format_trace_dir=fixed_format_trace_dir,
            static_trace_fields=static_trace_fields,
            domain_override=context.domain_override,
        )
    )
    payload = build_intersurf_first_step_payload(step, dt_sechiba=context.dt_sechiba)
    nflow = parse_run_def_int(run_def_values, "NSTM")
    river_routing = parse_run_def_bool(run_def_values["RIVER_ROUTING"])
    driver_source = stomate_driver_entry_source(payload, kjit=int(tstep) + 1)
    routing_source = stomate_no_routing_entry_source(
        kjpindex=payload.kjpindex,
        nflow=nflow,
        river_routing=river_routing,
        nbp_glo=payload.kjpindex,
    )

    covered = [
        "driver_forcing_step",
        "intersurf_payload",
        "pft_static",
        "co2_ccanopy",
        "water_table_sequences",
        "sechiba_static_restart_anchors",
    ]
    missing = [
        "sechiba_prognostic_state_before_step",
        "diffuco_full_precall_state",
        "enerbil_surface_state_before_step",
        "hydrol_soil_water_state_before_step",
        "thermosoil_temperature_state_before_step",
        "slowproc_dynamic_surface_state_before_step",
        "stomate_daily_monthly_annual_accumulators",
        "restart_history_writers",
    ]
    notes = [
        "Same-step SECHIBA/HYDROL/THERMOSOIL outputs must be produced in Fortran order before OK_LEAK process inputs can be completed.",
    ]

    stomate_pre_step = None
    first_step_restart_state = None
    diffuco_control_coverage = None
    diffuco_control_salinity = None
    diffuco_control_assembly = None
    diffuco_first_step_precall = None
    diffuco_first_step_local_enerbil_precall = None
    enerbil_first_step_coverage = None
    enerbil_first_step_precall = None
    enerbil_first_step_local = None
    hydrol_first_step_precall = None
    hydrol_first_step_module = None
    condveg_first_step_module = None
    thermosoil_first_step_module = None
    sechiba_first_step_coverage = None
    stomate_first_step_daily_accumulation = None
    stomate_bundle_source = None
    stomate_restart_bundles = None
    if int(tstep) == 0:
        diaglev, _, _ = paper_case_vertical_grids_from_used_run_def(run_def_path)
        cwrr_grid = paper_case_cwrr_vertical_soil_grid_from_used_run_def(run_def_path)
        first_step_restart_state = context.first_step_restart_state or reference_case_first_step_restart_state(
            config_path,
            root=root,
            run_dir=context.reference_run_dir,
            run_def_path=run_def_path,
        )
        payload = build_intersurf_first_step_payload(
            step,
            driver_z0_for_wind=first_step_restart_state.driver_albedo.z0,
            dt_sechiba=runtime.dt_sechiba,
        )
        driver_source = stomate_driver_entry_source(payload, kjit=int(tstep) + 1)
        routing_source = stomate_no_routing_entry_source(
            kjpindex=payload.kjpindex,
            nflow=nflow,
            river_routing=river_routing,
            nbp_glo=payload.kjpindex,
        )
        enerbil_first_step_coverage = enerbil_first_step_input_coverage(
            driver_or_intersurf_payload=_enerbil_driver_coverage_payload(payload),
            driver_albedo_payload=first_step_restart_state.driver_albedo.as_coverage_payload(),
            soil_thermal_restart_payload=first_step_restart_state.enerbil_thermal.as_coverage_payload(),
            sechiba_var_init_executed=True,
            non_watchout_driver_executed=True,
            condveg_initialize_executed=True,
            condveg_impaze=False,
        )
        enerbil_first_step_precall = assemble_enerbil_first_step_precall_payload(
            driver_or_intersurf_payload=_enerbil_driver_coverage_payload(payload),
            driver_albedo_payload=first_step_restart_state.driver_albedo.as_coverage_payload(),
            soil_thermal_restart_payload=first_step_restart_state.enerbil_thermal.as_coverage_payload(),
            condveg_impaze=False,
            non_watchout_driver_executed=True,
        )
        diffuco_control_coverage = diffuco_control_inundation_input_coverage(
            diaglev_source_available=True,
            humcste_source_available=True,
            full_tide_height_available=payload.tide_height is not None,
            first_step_stomate_start_biomass_available=True,
            restart_biomass_available=True,
        )
        diffuco_control_salinity = assemble_pft14_control_salinity_first_step(
            salinity=payload.salinity,
            control_salinity_min=parse_run_def_float(run_def_values, "CONTROL_SALINITY_MIN"),
        )
        nvm = int(first_step_restart_state.stomate.biomass.shape[1])
        fortran_pft_ids = tuple(int(value) for value in step.run_scalars.fortran_pft_ids)
        if len(fortran_pft_ids) != nvm:
            raise ValueError("restart state PFT axis disagrees with the selected run layout")
        pft_to_mtc = parse_run_def_indexed_selection(
            run_def_values, "PFT_TO_MTC", fortran_pft_ids, dtype=int
        )
        hydrol_humcste = parse_run_def_indexed_selection(
            run_def_values, "HYDROL_HUMCSTE", fortran_pft_ids
        )
        diffuco_control_assembly = assemble_pft14_control_inundation_first_step(
            depth_max_h=parse_run_def_float(run_def_values, "DEPTH_MAX_H"),
            depth_max_t=parse_run_def_float(run_def_values, "DEPTH_MAX_T"),
            depth_topthickness=parse_run_def_float(run_def_values, "DEPTH_TOPTHICK"),
            depth_cstthickness=parse_run_def_float(run_def_values, "DEPTH_CSTTHICK"),
            depth_geom=parse_run_def_float(run_def_values, "DEPTH_GEOM"),
            ratio_geom_below=parse_run_def_float(run_def_values, "RATIO_GEOM_BELOW"),
            pft_to_mtc=pft_to_mtc,
            hydrol_humcste=hydrol_humcste,
            tide_height=payload.tide_height,
            biomass=first_step_restart_state.stomate.biomass,
            control_inudate_min=parse_run_def_float(run_def_values, "CONTROL_INUDATE_MIN"),
            agb_agr_ven_all_st=parse_run_def_float(run_def_values, "AGB_AGR_VEN_ALL_ST"),
            agb_agr_ven_all_pn=parse_run_def_float(run_def_values, "AGB_AGR_VEN_ALL_PN"),
            h_agr_max_st=parse_run_def_float(run_def_values, "H_AGR_MAX_ST"),
            h_agr_max_pn=parse_run_def_float(run_def_values, "H_AGR_MAX_PN"),
            pft_index=_paper_mangrove_pft_index(step.run_scalars),
        )
        slowproc_derivvar = slowproc_derivvar_explicit(
            veget=first_step_restart_state.slowproc.veget,
            lai=first_step_restart_state.slowproc.lai,
            vcmax_fix=parse_run_def_indexed_selection(run_def_values, "VCMAX_FIX", fortran_pft_ids),
            height_presc=parse_run_def_indexed_selection(
                run_def_values, "SLOWPROC_HEIGHT", fortran_pft_ids
            ),
            qsintcst=parse_run_def_float(run_def_values, "SECHIBA_QSINT"),
        )
        diffuco_first_step_precall = assemble_diffuco_first_step_precall_payload(
            driver_or_intersurf_payload={
                "u": payload.u,
                "v": payload.v,
                "zlev": payload.zlev,
                "temp_air": payload.temp_air,
                "qair": payload.qair,
                "pb": payload.pb,
                "swdown": payload.swdown,
                "ccanopy": payload.ccanopy,
                "precip_rain": payload.precip_rain,
                "lalo": payload.lalo,
                "neighbours": payload.neighbours,
                "resolution": payload.resolution,
            },
            restart_payload=first_step_restart_state.diffuco_precall.as_payload(),
            driver_albedo_payload=first_step_restart_state.driver_albedo.as_coverage_payload(),
            slowproc_derivvar_payload={
                "qsintmax": slowproc_derivvar.qsintmax,
                "assim_param": slowproc_derivvar.assim_param,
                "height": slowproc_derivvar.height,
                "temp_growth": slowproc_derivvar.temp_growth,
            },
            diffuco_control_salinity=diffuco_control_salinity,
            diffuco_control_inundation=diffuco_control_assembly,
            ok_explicitsnow=parse_run_def_bool(run_def_values["OK_EXPLICITSNOW"]),
            river_routing=river_routing,
            nbp_glo=payload.kjpindex,
        )
        after_diffuco_payload = diffuco_first_step_precall.payload
        if diffuco_first_step_precall.ok:
            diffuco_first_step_local_enerbil_precall = run_pft14_local_enerbil_precall_from_first_step_precall(
                diffuco_first_step_precall,
                run_def_values=run_def_values,
                pft_index=_paper_mangrove_pft_index(step.run_scalars),
            )
            after_diffuco_payload = diffuco_first_step_local_enerbil_precall.result.payload.payload
        enerbil_first_step_coverage = enerbil_first_step_input_coverage(
            after_diffuco_payload=after_diffuco_payload,
            driver_or_intersurf_payload=_enerbil_driver_coverage_payload(payload),
            driver_albedo_payload=first_step_restart_state.driver_albedo.as_coverage_payload(),
            soil_thermal_restart_payload=first_step_restart_state.enerbil_thermal.as_coverage_payload(),
            sechiba_var_init_executed=True,
            non_watchout_driver_executed=True,
            condveg_initialize_executed=True,
            condveg_impaze=False,
        )
        enerbil_first_step_precall = assemble_enerbil_first_step_precall_payload(
            driver_or_intersurf_payload=_enerbil_driver_coverage_payload(payload),
            driver_albedo_payload=first_step_restart_state.driver_albedo.as_coverage_payload(),
            soil_thermal_restart_payload=first_step_restart_state.enerbil_thermal.as_coverage_payload(),
            after_diffuco_payload=after_diffuco_payload,
            condveg_impaze=False,
            non_watchout_driver_executed=True,
        )
        if diffuco_first_step_local_enerbil_precall is not None:
            ok_laidev = [
                parse_run_def_bool(run_def_values[f"OK_LAIDEV__{fortran_id:05d}"])
                for fortran_id in fortran_pft_ids
            ]
            enerbil_first_step_local = run_enerbil_first_step_local_from_precall(
                enerbil_first_step_precall,
                ok_laidev=ok_laidev,
                dt_sechiba=parse_run_def_float(run_def_values, "DT_SECHIBA"),
                ok_explicitsnow=parse_run_def_bool(run_def_values["OK_EXPLICITSNOW"]),
                min_wind=parse_run_def_float(run_def_values, "MIN_WIND"),
            )
            enerbil_first_step_coverage = enerbil_first_step_input_coverage(
                after_diffuco_payload=after_diffuco_payload,
                driver_or_intersurf_payload=_enerbil_driver_coverage_payload(payload),
                driver_albedo_payload=first_step_restart_state.driver_albedo.as_coverage_payload(),
                soil_thermal_restart_payload=first_step_restart_state.enerbil_thermal.as_coverage_payload(),
                sechiba_var_init_executed=True,
                non_watchout_driver_executed=True,
                enerbil_begin_executed=True,
                enerbil_surftemp_executed=True,
                condveg_initialize_executed=True,
                condveg_impaze=False,
            )
        if enerbil_first_step_local is not None:
            hydrol_first_step_precall = assemble_hydrol_first_step_precall_payload(
                enerbil_payload=enerbil_first_step_local.payload,
                hydrol_restart=first_step_restart_state.hydrol,
                slowproc_restart=first_step_restart_state.slowproc,
                pref_soil_veg=step.run_scalars.pref_soil_veg,
                nstm=nflow,
                ext_coeff_vegetfrac=_select_run_def_pft_vector(
                    run_def_values, "EXT_COEFF_VEGETFRAC", step.run_scalars, nvm=nvm
                ),
                precip_rain=payload.precip_rain,
                precip_snow=payload.precip_snow,
                diffuco_payload=_hydrol_current_diffuco_payload(
                    diffuco_first_step_precall.payload if diffuco_first_step_precall is not None else None,
                    after_diffuco_payload,
                ),
                thermosoil_restart=first_step_restart_state.thermosoil,
                throughfall_by_pft=_select_run_def_pft_vector(
                    run_def_values, "PERCENT_THROUGHFALL_PFT", step.run_scalars, nvm=nvm
                ),
                humcste=_select_run_def_pft_vector(
                    run_def_values, "HYDROL_HUMCSTE", step.run_scalars, nvm=nvm
                ),
                zz_mm=cwrr_grid.znh * 1000.0,
                diaglev_m=diaglev,
                peat_hydro=PAPER_1961_HYDROL_SOIL_PEAT_HYDRO,
                ok_dgvm=parse_run_def_bool(run_def_values["STOMATE_OK_DGVM"]),
                ok_explicitsnow=parse_run_def_bool(run_def_values["OK_EXPLICITSNOW"]),
                ok_freeze_cwrr=parse_run_def_bool(run_def_values["OK_FREEZE_CWRR"]),
                ok_thermodynamical_freezing=parse_run_def_bool(run_def_values["OK_THERMODYNAMICAL_FREEZING"]),
                fr_dt=parse_run_def_float(run_def_values, "FR_DT"),
                froz_frac_corr=parse_run_def_float(run_def_values, "FROZ_FRAC_CORR"),
                smtot_corr=parse_run_def_float(run_def_values, "SMTOT_CORR"),
                max_froz_hydro=parse_run_def_float(run_def_values, "MAX_FROZ_HYDRO"),
                dt_days=parse_run_def_float(run_def_values, "DT_SECHIBA") / 86400.0,
            )
            hydrol_first_step_module = run_hydrol_first_step_module_from_precall(
                hydrol_first_step_precall,
                temp_air=payload.temp_air,
                pb=payload.pb,
                u=payload.u,
                v=payload.v,
                humcste=_select_run_def_pft_vector(
                    run_def_values, "HYDROL_HUMCSTE", step.run_scalars, nvm=nvm
                ),
                dz_mm=cwrr_grid.dnh * 1000.0,
                dh_mm=cwrr_grid.dlh * 1000.0,
                zz_mm=cwrr_grid.znh * 1000.0,
                altmax=first_step_restart_state.stomate.altmax,
                ks=parse_run_def_indexed_vector(run_def_values, "CWRR_KS", 12),
                reinf_slope=[float(run_def_values.get("SLOPE", "0.0"))],
                zmaxh_m=parse_run_def_float(run_def_values, "DEPTH_MAX_H"),
                doponds=parse_run_def_bool(run_def_values["DO_PONDS"]),
                peat_hydro=PAPER_1961_HYDROL_SOIL_PEAT_HYDRO,
                peat_hydro_water_stress=PAPER_1961_HYDROL_SOIL_PEAT_HYDRO,
                ok_freeze_cwrr=parse_run_def_bool(run_def_values["OK_FREEZE_CWRR"]),
                ok_pc=parse_run_def_bool(run_def_values["OK_PC"]),
                ok_leak=parse_run_def_bool(run_def_values["OK_LEAK"]),
                ok_ru2peat=parse_run_def_bool(run_def_values["OK_RU2PEAT"]),
                ok_wt_ab=parse_run_def_bool(run_def_values["OK_WT_AB"]),
                tides=parse_run_def_bool(run_def_values["TIDES"]),
                agri_peat=parse_run_def_bool(run_def_values["AGRI_PEAT"]),
                max_wt_ab=parse_run_def_float(run_def_values, "max_wt_ab") if "max_wt_ab" in run_def_values else 100.0,
                dyn_nroot_larix=parse_run_def_bool(run_def_values["dyn_nroot_larix"]),
                is_tree=step.run_scalars.is_tree,
                znt=diaglev,
                run2peat=first_step_restart_state.hydrol.run2peat,
                run2man=first_step_restart_state.hydrol.run2man,
                dt_sechiba=parse_run_def_float(run_def_values, "DT_SECHIBA"),
                use_jit=fixed_format_trace_dir is None and static_trace_fields is None,
            )
            condveg_first_step_module = run_condveg_first_step_module(
                restart_payload=first_step_restart_state.diffuco_precall.as_payload(),
                driver_payload={
                    "zlev": payload.zlev,
                    "temp_air": payload.temp_air,
                    "pb": payload.pb,
                    "u": payload.u,
                    "v": payload.v,
                },
                slowproc_payload={
                    "totfrac_nobio": 0.0
                    if first_step_restart_state.diffuco_precall.frac_nobio is None
                    else first_step_restart_state.diffuco_precall.frac_nobio.sum(axis=1),
                    "tot_bare_soil": first_step_restart_state.diffuco_precall.veget_max[:, 0]
                    + (
                        first_step_restart_state.diffuco_precall.veget_max[:, 1:]
                        - first_step_restart_state.diffuco_precall.veget[:, 1:]
                    ).sum(axis=1),
                    "is_tree": step.run_scalars.is_tree,
                    "z0_over_height": step.run_scalars.z0_over_height,
                    "ratio_z0m_z0h": step.run_scalars.ratio_z0m_z0h,
                },
                hydrol_diagnostics=hydrol_first_step_module.diagnostics if hydrol_first_step_module.ok else None,
                run_def_values=run_def_values,
                ok_explicitsnow=parse_run_def_bool(run_def_values["OK_EXPLICITSNOW"]),
                impaze=parse_run_def_bool(run_def_values["IMPOSE_AZE"]),
                rough_dyn=parse_run_def_bool(run_def_values["ROUGH_DYN"]),
            )
            if condveg_first_step_module.ok:
                thermosoil_first_step_module = run_thermosoil_first_step_module(
                    moisture=hydrol_first_step_module.thermosoil_moisture,
                    thermosoil_restart=first_step_restart_state.thermosoil,
                    enerbil_payload=enerbil_first_step_local.payload,
                    condveg_result=condveg_first_step_module.module,
                    snowrho=first_step_restart_state.diffuco_precall.snowrho,
                    snowtemp=first_step_restart_state.diffuco_precall.snowtemp,
                    snowdz=first_step_restart_state.diffuco_precall.snowdz,
                    njsc=first_step_restart_state.hydrol.njsc,
                    veget_max=first_step_restart_state.diffuco_precall.veget_max,
                    pb=payload.pb,
                    totfrac_nobio=first_step_restart_state.diffuco_precall.frac_nobio.sum(axis=1),
                    dlt=cwrr_grid.dlt,
                    dz1=cwrr_grid.dz1,
                    zlt=cwrr_grid.zlt,
                    znt=cwrr_grid.znt,
                    dz5=cwrr_grid.dz5,
                    dt_sechiba=parse_run_def_float(run_def_values, "DT_SECHIBA"),
                    ok_laidev=[
                        parse_run_def_bool(run_def_values[f"OK_LAIDEV__{fortran_id:05d}"])
                        for fortran_id in fortran_pft_ids
                    ],
                    satsoil=parse_run_def_bool(run_def_values["satsoil"]),
                        ok_shum_ngrnd_permalong=_effective_thermosoil_wetdiaglong(run_def_values),
                )
        sechiba_first_step_coverage = first_step_sechiba_input_coverage(
            diffuco_control=diffuco_control_coverage,
            diffuco_control_salinity=diffuco_control_salinity,
            diffuco_control_inundation=diffuco_control_assembly,
            diffuco_precall=diffuco_first_step_precall,
            diffuco_local_enerbil_precall=diffuco_first_step_local_enerbil_precall,
            enerbil=enerbil_first_step_coverage,
            hydrol_precall=hydrol_first_step_precall,
            hydrol_module=hydrol_first_step_module,
            condveg_module=condveg_first_step_module,
            thermosoil_module=thermosoil_first_step_module,
            hydrol_restart_anchors_available=True,
            condveg_initialize_available=True,
            thermosoil_restart_recurrence_available=True,
            slowproc_restart_state_available=True,
        )
        first_step_sources = _paper_first_step_restart_entry_sources(
            restart_state=first_step_restart_state,
            run_scalars=step.run_scalars,
            run_def_values=run_def_values,
            znt=cwrr_grid.znt,
            zlt=cwrr_grid.zlt,
        )
        same_step_sources = _paper_first_step_same_step_entry_sources(
            enerbil=enerbil_first_step_local,
            after_diffuco_payload=after_diffuco_payload,
            hydrol=hydrol_first_step_module,
            thermosoil=thermosoil_first_step_module,
            snow_state=(
                {
                    "snow": hydrol_first_step_module.snow_state.snow,
                    "snowdz": hydrol_first_step_module.snow_state.snowdz,
                    "snowrho": hydrol_first_step_module.snow_state.snowrho,
                }
                if hydrol_first_step_module is not None
                and hydrol_first_step_module.ok
                and hydrol_first_step_module.snow_state is not None
                else None
            ),
            temp_sol=first_step_restart_state.diffuco_precall.temp_sol,
            znt=cwrr_grid.znt,
            zlt=cwrr_grid.zlt,
        )
        stomate_entry_assembly = assemble_stomate_main_payload(
            driver_source,
            routing_source,
            *first_step_sources,
            *same_step_sources,
            _paper_first_step_fwet_entry_source(
                hydrol=hydrol_first_step_module,
                restart_state=first_step_restart_state,
                topmodel_new=parse_run_def_bool(run_def_values["TOPMODEL_NEW"]),
            ),
        )
        stomate_pre_step = paper_1961_first_step_stomate_boundary(
            config_path,
            bundle=step,
            payload=payload,
            used_run_def_path=run_def_path,
            root=root,
        )
        daily_accumulators = first_step_restart_state.stomate_readstart.daily_state
        stomate_first_step_daily_accumulation = stomate_first_step_daily_accumulation_from_entry(
            entry_payload=stomate_entry_assembly.payload,
            accumulator_state=daily_accumulators,
            dt_sechiba=runtime.dt_sechiba,
            dt_stomate=runtime.dt_stomate,
            do_slow=False,
            peat_occur=parse_run_def_bool(run_def_values["PEAT_OCCUR"]),
            date=1,
        )
        if _paper_first_step_do_slow(tstep, runtime):
            stomate_bundle_source, stomate_restart_bundles = _paper_first_step_stomate_restart_input_bundles(
                config_path=config_path,
                run_def_path=run_def_path,
                restart_state=first_step_restart_state,
                runtime=runtime,
                tstep=tstep,
                run_def_values=run_def_values,
            )
        covered.extend(
            [
                "reference_case_first_step_restart_state",
                "driver_albedo_restart_state",
                "enerbil_soil_thermal_restart_state",
                "thermosoil_restart_recurrence_state",
                "hydrol_restart_anchors",
                "slowproc_restart_stomate_entry_state",
                "stomate_restart_entry_state",
                "slowproc_no_lcc_fire_dynpeat_inactive_entry_state",
                "slowproc_static_and_vertical_entry_state",
                "diffuco_control_inundation_input_coverage",
                "diffuco_control_salinity_partial_assembly",
                "diffuco_control_inundation_partial_assembly",
                "diffuco_first_step_precall_partial_payload",
                "diffuco_first_step_local_enerbil_precall"
                if diffuco_first_step_local_enerbil_precall is not None
                else "diffuco_first_step_local_enerbil_precall_missing_controls",
                "enerbil_first_step_input_coverage",
                "enerbil_first_step_precall_partial_payload",
                "enerbil_first_step_local_step"
                if enerbil_first_step_local is not None
                else "enerbil_first_step_local_step_missing_diffuco",
                "hydrol_first_step_precall_partial_payload"
                if hydrol_first_step_precall is not None
                else "hydrol_first_step_precall_missing_enerbil",
                "hydrol_first_step_module_closure"
                if hydrol_first_step_module is not None and hydrol_first_step_module.ok
                else "hydrol_first_step_module_closure_missing_inputs",
                "condveg_first_step_module_closure"
                if condveg_first_step_module is not None and condveg_first_step_module.ok
                else "condveg_first_step_module_closure_missing_inputs",
                "thermosoil_first_step_module_closure"
                if thermosoil_first_step_module is not None and thermosoil_first_step_module.ok
                else "thermosoil_first_step_module_closure_missing_inputs",
                "same_step_sechiba_to_stomate_entry_sources"
                if same_step_sources
                else "same_step_sechiba_to_stomate_entry_sources_missing_inputs",
                "stomate_first_step_daily_accumulator_state",
                "stomate_first_step_daily_accumulation"
                if stomate_first_step_daily_accumulation.ok
                else "stomate_first_step_daily_accumulation_missing_inputs",
                "stomate_main_chain_restart_input_bundles"
                if stomate_restart_bundles is not None
                else "stomate_main_chain_restart_input_bundles_waiting_for_do_slow",
                "sechiba_first_step_input_coverage",
            ]
        )
        covered.append("stomate_restart_ok_leak_pre_step_boundary")
        missing = [
            item
            for item in missing
            if item
            not in {
                "sechiba_prognostic_state_before_step",
                "enerbil_surface_state_before_step",
                "hydrol_soil_water_state_before_step",
                "thermosoil_temperature_state_before_step",
                "slowproc_dynamic_surface_state_before_step",
                "stomate_daily_monthly_annual_accumulators",
            }
        ] + [
            "diffuco_same_step_precall_state_and_gpp",
            "enerbil_same_step_outputs_to_hydrol_slowproc",
            *([] if hydrol_first_step_module is not None and hydrol_first_step_module.ok else ["hydrol_same_step_diagnostics_to_stomate"]),
            *([] if condveg_first_step_module is not None and "albedo" in condveg_first_step_module.covered_fields else ["condveg_same_step_snow_albedo_state"]),
            *([] if thermosoil_first_step_module is not None and thermosoil_first_step_module.ok else ["thermosoil_same_step_stempdiag_to_stomate"]),
            "slowproc_post_stomate_surface_update",
            *([] if stomate_restart_bundles is not None else ["stomate_daily_process_waiting_for_do_slow"]),
            *([] if stomate_restart_bundles is None else ["stomate_lpj_process_outputs_to_state"]),
            *([] if stomate_first_step_daily_accumulation is not None and stomate_first_step_daily_accumulation.ok else ["stomate_accumulator_state_before_step"]),
        ]
    else:
        stomate_entry_assembly = assemble_stomate_main_payload(driver_source, routing_source)
        missing.append("stomate_restart_state_after_previous_step")
        notes.append("For tstep>0, restart files are no longer the current prognostic state; previous-step model state is required.")

    return DriverTimestepScaffold(
        step=step,
        payload=payload,
        runtime=runtime,
        stomate_entry_assembly=stomate_entry_assembly,
        diffuco_control_coverage=diffuco_control_coverage,
        diffuco_control_salinity=diffuco_control_salinity,
        diffuco_control_assembly=diffuco_control_assembly,
        diffuco_first_step_precall=diffuco_first_step_precall,
        diffuco_first_step_local_enerbil_precall=diffuco_first_step_local_enerbil_precall,
        enerbil_first_step_coverage=enerbil_first_step_coverage,
        enerbil_first_step_precall=enerbil_first_step_precall,
        enerbil_first_step_local=enerbil_first_step_local,
        hydrol_first_step_precall=hydrol_first_step_precall,
        hydrol_first_step_module=hydrol_first_step_module,
        condveg_first_step_module=condveg_first_step_module,
        thermosoil_first_step_module=thermosoil_first_step_module,
        sechiba_first_step_coverage=sechiba_first_step_coverage,
        stomate_first_step_daily_accumulation=stomate_first_step_daily_accumulation,
        stomate_bundle_source=stomate_bundle_source,
        stomate_restart_input_bundles=stomate_restart_bundles,
        first_step_restart_state=first_step_restart_state,
        stomate_pre_step=stomate_pre_step,
        initial_state_mode=REFERENCE_CASE_INITIAL_STATE_MODE if first_step_restart_state is not None else "previous_step_state_required",
        covered_components=tuple(dict.fromkeys(covered)),
        missing_components=tuple(dict.fromkeys(missing)),
        notes=tuple(notes),
    )


def paper_1961_driver_year_scaffold(
    config_path: str | Path,
    *,
    year: int = 1961,
    preview_nsteps: int = 2,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    used_run_def_path: str | Path | None = None,
    root: str | Path | None = None,
) -> DriverYearScaffold:
    """Preview strict timestep scaffolds in Fortran driver-loop order.

    This intentionally previews a bounded number of timesteps. It does not run
    SECHIBA/STOMATE or advance prognostic state, and therefore keeps process
    gaps explicit for every previewed step.
    """

    if preview_nsteps < 0:
        raise ValueError("preview_nsteps must be non-negative")
    total = int(load_case_config(config_path)["drivers"]["atmospheric_forcing"]["dimensions"]["tstep"])
    steps = tuple(
        paper_1961_driver_timestep_scaffold(
            config_path,
            year=year,
            tstep=tstep,
            fixed_format_trace_dir=fixed_format_trace_dir,
            static_trace_fields=static_trace_fields,
            used_run_def_path=used_run_def_path,
            root=root,
        )
        for tstep in range(min(int(preview_nsteps), total))
    )
    return DriverYearScaffold(
        year=year,
        preview_steps=steps,
        total_forcing_steps=total,
    )


def _paper_day_gap_after_scaffold(scaffold: DriverTimestepScaffold) -> tuple[DriverDayStateGap, ...]:
    if not scaffold.ready_to_run_processes:
        return (
            DriverDayStateGap(
                component=f"tstep_{scaffold.step.tstep}_entry_payload",
                fields=scaffold.missing_components,
                provenance=scaffold.provenance,
                notes=(
                    "The first incomplete timestep must be closed before later day-state recurrence can be trusted.",
                ),
            ),
        )
    return DAY1_PREVIOUS_STEP_STATE_GAPS


def _paper_day_completed_entry_payloads(scaffold: DriverTimestepScaffold) -> tuple[dict[str, object], ...]:
    daily = scaffold.stomate_first_step_daily_accumulation
    if daily is None or not daily.ok:
        return ()
    for required in ("t2m", "stempdiag", "gpp", "precip_rain", "precip_snow", "snow", "tmc_topgrass"):
        if required not in scaffold.stomate_entry_assembly.payload:
            return ()
    if scaffold.stomate_entry_assembly.partial_inputs:
        return ()
    return (dict(scaffold.stomate_entry_assembly.payload),)


def _paper_day_daily_process_from_completed_entries(
    *,
    config_path: str | Path,
    run_def_path: Path,
    restart_state: ReferenceCaseFirstStepRestartState,
    runtime: DriverRuntimeScalars,
    tstep: int,
    entry_payloads: tuple[dict[str, object], ...],
    retain_step_results: bool = True,
    use_maintenance_jit: bool = False,
    single_pass_daily_fold: bool = False,
    run_def_values: dict[str, str] | None = None,
):
    if not entry_payloads:
        return None
    steps_per_stomate = int(round(runtime.dt_stomate / runtime.dt_sechiba))
    if len(entry_payloads) != steps_per_stomate:
        return None
    source, bundles = _paper_first_step_stomate_restart_input_bundles(
        config_path=config_path,
        run_def_path=run_def_path,
        restart_state=restart_state,
        runtime=runtime,
        tstep=tstep,
        run_def_values=run_def_values,
    )
    daily_state = restart_state.stomate_readstart.daily_state
    season_state = restart_state.stomate_readstart.season_state
    values = parse_run_def(run_def_path) if run_def_values is None else run_def_values
    flags = tuple(_paper_first_step_do_slow(index, runtime) for index in range(steps_per_stomate))
    folded = stomate_daily_process_fold_from_entries(
        entry_payloads=entry_payloads,
        accumulator_state=daily_state,
        resp_maint_part_current=restart_state.stomate.resp_maint_part,
        do_slow_flags=flags,
        dt_sechiba=runtime.dt_sechiba,
        dt_stomate=runtime.dt_stomate,
        biomass=restart_state.stomate.biomass,
        t2m_longterm=season_state.t2m_longterm,
        z_soil=source.boundary.kwargs["z_soil"],
        rprof=source.boundary.kwargs["rprof"],
        sla_calc=bundles.maintenance_inputs["sla_calc"],
        coeff_maint_zero=bundles.maintenance_inputs["coeff_maint_zero"],
        maint_resp_slope=bundles.maintenance_inputs["maint_resp_slope"],
        ext_coeff=bundles.maintenance_inputs["ext_coeff"],
        is_tree=bundles.maintenance_inputs["is_tree"],
        flood_frac=entry_payloads[-1].get("flood_frac"),
        peat_occur=parse_run_def_bool(values["PEAT_OCCUR"]),
        date=1,
        retain_step_results=retain_step_results,
        use_maintenance_jit=use_maintenance_jit,
        single_pass=single_pass_daily_fold,
    )
    if not folded.ok:
        return None
    return folded


def _paper_later_day_daily_process_from_completed_entries(
    *,
    first_step: DriverNextStepHydrolPrecallScaffold | None,
    previous_state: DriverPreviousStepStatePacket,
    config_path: str | Path,
    run_def_path: Path,
    run_def_values: dict[str, str],
    runtime: DriverRuntimeScalars,
    start_tstep: int,
    entry_payloads: tuple[dict[str, object], ...],
    retain_step_results: bool = True,
    use_maintenance_jit: bool = False,
    single_pass_daily_fold: bool = False,
    use_compiled_accumulator: bool = False,
    use_numpy_accumulator: bool = False,
    prepared_context: Paper1961PreparedDriverContext | None = None,
    stomate_parameter_overrides: Mapping[str, object] | None = None,
):
    """Fold a later day's STOMATE daily accumulators from model-produced state.

    Fortran provenance: ``src_stomate/stomate.f90`` lines 3187-3267 update
    daily accumulators every half-hour; lines 4944-5000 reset them after the
    previous ``do_slow`` boundary. The initial accumulator state here must come
    from ``previous_state`` rather than the original restart.
    """

    if not entry_payloads:
        return None
    steps_per_stomate = int(round(runtime.dt_stomate / runtime.dt_sechiba))
    if len(entry_payloads) != steps_per_stomate:
        return None
    slowproc_state = previous_state.fields_by_component.get("slowproc_stomate_previous_step_state", {})
    daily_accumulators = slowproc_state.get("daily_accumulators")
    if daily_accumulators is None:
        return None
    hydrol_state = previous_state.fields_by_component.get("hydrol_previous_step_state", {})
    if "humrel" not in hydrol_state:
        return None
    nvm = int(slowproc_state["lai"].shape[1])
    npts = int(slowproc_state["lai"].shape[0])
    static = prepared_context.stomate_static if prepared_context is not None else None
    boundary = prepared_context.stomate_boundary if prepared_context is not None else None
    if static is None:
        static = paper_case_stomate_static_input_kwargs(config_path, used_run_def=run_def_path)
    parameter_overrides = stomate_parameter_overrides or {}
    if boundary is None or int(np.asarray(boundary.kwargs["rprof"]).shape[0]) != npts:
        boundary = paper_case_stomate_boundary_input_kwargs(
            config_path,
            parameters=static.parameters,
            npts=npts,
            used_run_def=run_def_path,
        )
    flags = tuple(_paper_first_step_do_slow(tstep, runtime) for tstep in range(int(start_tstep), int(start_tstep) + steps_per_stomate))
    t2m_longterm = (
        slowproc_state["t2m_longterm"]
        if "t2m_longterm" in slowproc_state
        else jnp.asarray(entry_payloads[0]["t2m"])
    )
    return stomate_daily_process_fold_from_entries(
        entry_payloads=entry_payloads,
        accumulator_state=SimpleNamespace(**daily_accumulators),
        resp_maint_part_current=slowproc_state["resp_maint_part"],
        do_slow_flags=flags,
        dt_sechiba=runtime.dt_sechiba,
        dt_stomate=runtime.dt_stomate,
        biomass=slowproc_state["biomass"],
        t2m_longterm=t2m_longterm,
        z_soil=boundary.kwargs["z_soil"],
        rprof=boundary.kwargs["rprof"],
        sla_calc=slowproc_state["sla_calc"],
        coeff_maint_zero=static.parameters.coeff_maint_zero,
        maint_resp_slope=parameter_overrides.get("maint_resp_slope", static.parameters.maint_resp_slope),
        ext_coeff=static.parameters.ext_coeff,
        is_tree=static.parameters.is_tree,
        flood_frac=entry_payloads[-1].get("flood_frac"),
        peat_occur=parse_run_def_bool(run_def_values["PEAT_OCCUR"]),
        date=slowproc_state.get("date", int(start_tstep // steps_per_stomate))
        + int(round(runtime.dt_days)),
        retain_step_results=retain_step_results,
        use_maintenance_jit=use_maintenance_jit,
        use_compiled_accumulator=use_compiled_accumulator,
        use_numpy_accumulator=use_numpy_accumulator,
        single_pass=single_pass_daily_fold,
    )


def _paper_later_day_daily_fold_prerequisite_gaps(
    *,
    previous_state: DriverPreviousStepStatePacket,
    entry_payloads: tuple[dict[str, object], ...],
    steps_per_stomate: int,
) -> tuple[str, ...]:
    """Name missing runtime inputs that otherwise collapse to a null fold."""

    gaps: list[str] = []
    if not entry_payloads:
        gaps.append("daily_process_fold_entry_payloads_empty")
    elif len(entry_payloads) != int(steps_per_stomate):
        gaps.append(f"daily_process_fold_entry_payload_count_{len(entry_payloads)}_of_{int(steps_per_stomate)}")
    slowproc_state = previous_state.fields_by_component.get("slowproc_stomate_previous_step_state", {})
    if slowproc_state.get("daily_accumulators") is None:
        gaps.append("daily_process_fold_daily_accumulators")
    hydrol_state = previous_state.fields_by_component.get("hydrol_previous_step_state", {})
    if "humrel" not in hydrol_state:
        gaps.append("daily_process_fold_hydrol_humrel")
    return tuple(gaps)


def _paper_cold_start_day_daily_process_from_completed_entries(
    *,
    first_step_coverage: ColdStartFirstStepStateCoverage,
    config_path: str | Path,
    run_def_path: Path,
    runtime: DriverRuntimeScalars,
    entry_payloads: tuple[dict[str, object], ...],
):
    """Fold first-day STOMATE daily accumulators from no-restart state."""

    if not entry_payloads:
        return None
    steps_per_stomate = int(round(runtime.dt_stomate / runtime.dt_sechiba))
    if len(entry_payloads) != steps_per_stomate:
        return None
    payloads = first_step_coverage.initialized_payloads
    previous_state = payloads.get("cold_start_first_step_previous_state")
    if previous_state is None:
        return None
    slowproc_state = previous_state.fields_by_component.get("slowproc_stomate_previous_step_state", {})
    daily_state = payloads.get("stomate_cold_start_daily_accumulators")
    if daily_state is None:
        return None
    static = paper_case_stomate_static_input_kwargs(config_path, used_run_def=run_def_path)
    boundary = paper_case_stomate_boundary_input_kwargs(
        config_path,
        parameters=static.parameters,
        npts=int(np.asarray(slowproc_state["lai"]).shape[0]),
        used_run_def=run_def_path,
    )
    flags = tuple(_paper_first_step_do_slow(index, runtime) for index in range(steps_per_stomate))
    t2m_longterm = slowproc_state["t2m_longterm"] if "t2m_longterm" in slowproc_state else np.asarray(entry_payloads[0]["t2m"])
    values = parse_run_def(run_def_path)
    folded = stomate_daily_process_fold_from_entries(
        entry_payloads=entry_payloads,
        accumulator_state=daily_state,
        resp_maint_part_current=slowproc_state["resp_maint_part"],
        do_slow_flags=flags,
        dt_sechiba=runtime.dt_sechiba,
        dt_stomate=runtime.dt_stomate,
        biomass=slowproc_state["biomass"],
        t2m_longterm=t2m_longterm,
        z_soil=boundary.kwargs["z_soil"],
        rprof=boundary.kwargs["rprof"],
        sla_calc=slowproc_state["sla_calc"],
        coeff_maint_zero=static.parameters.coeff_maint_zero,
        maint_resp_slope=static.parameters.maint_resp_slope,
        ext_coeff=static.parameters.ext_coeff,
        is_tree=static.parameters.is_tree,
        flood_frac=entry_payloads[-1].get("flood_frac"),
        peat_occur=parse_run_def_bool(values["PEAT_OCCUR"]),
        date=1,
    )
    return folded if folded.ok else None


def _paper_day_daily_carbon_boundary_from_fold(
    *,
    fold: StomateDailyProcessFold | None,
    bundles: StomateRestartInputBundles | None,
) -> DailyCarbonBoundary | None:
    """Build the first explicit daily-carbon boundary after day accumulators.

    Fortran provenance: ``src_stomate/stomate.f90::stomate_main`` lines
    3187-3267 produce daily accumulators and maintenance respiration before
    the ``StomateLpj`` call boundary at lines 4297-4335. Allocation/NPP-owned
    fields are deliberately left as explicit trace/process gaps by
    ``prepare_daily_carbon_inputs``.
    """

    if fold is None or not fold.ok or bundles is None:
        return None
    return prepare_daily_carbon_inputs(
        gpp_daily=fold.daily_fields["gpp_daily"],
        biomass=bundles.prescribe_inputs["biomass"],
        resp_maint_part=fold.daily_fields["resp_maint_part"],
        pft_present=bundles.prescribe_inputs["pft_present"],
        f_alloc=None,
    )


def _paper_day_post_npp_owned_daily_fields(bundles: StomateRestartInputBundles) -> dict[str, object]:
    """Return only daily fields consumed at the post-NPP boundary.

    Fortran provenance: ``src_stomate/stomate_lpj.f90::StomateLpj`` lines
    1118-1131 pass ``gpp_daily`` and ``resp_maint_part`` into ``npp_calc``;
    lines 1137-1557 consume ``t2m_min_daily`` in the post-NPP kill/gap/turnover
    chain. Other daily accumulators such as ``t2m_daily`` and snow diagnostics
    are intentionally excluded because they belong to separate STOMATE season,
    output, or process boundaries.
    """

    return {
        name: bundles.daily_process_inputs[name]
        for name in ("gpp_daily", "resp_maint_part", "t2m_min_daily")
        if name in bundles.daily_process_inputs
    }


def _paper_day_allocation_kwargs(
    *,
    run_def_values: dict[str, str],
    run_scalars: RunScalars,
    nvm: int,
) -> dict[str, object]:
    return {
        # Fortran provenance: constantes.f90::config_stomate_parameters
        # lines 1193-1231 read these keys through getin_p; stomate_alloc.f90
        # lines 775-812 consumes them in the allocation fractions.
        "f_fruit": parse_run_def_float(run_def_values, "F_FRUIT"),
        "ecureuil": _select_run_def_pft_vector(
            run_def_values, "ECUREUIL", run_scalars, nvm=nvm
        ),
        "alloc_sap_above_grass": parse_run_def_float(run_def_values, "ALLOC_SAP_ABOVE_GRASS"),
        "min_l_to_lsr": parse_run_def_float(run_def_values, "MIN_LTOLSR"),
        "max_l_to_lsr": parse_run_def_float(run_def_values, "MAX_LTOLSR"),
        "z_nitrogen": parse_run_def_float(run_def_values, "Z_NITROGEN"),
    }


def _paper_day_stomate_daily_carbon_from_bundles(
    bundles: StomateRestartInputBundles | None,
    *,
    run_def_values: dict[str, str],
    run_scalars: RunScalars,
    use_static_jit: bool = False,
    outer_compiled_static_dispatch: Mapping[str, object] | None = None,
) -> ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult | None:
    """Run the first-day daily STOMATE carbon chain from explicit bundles."""

    if bundles is None:
        return None
    nvm = int(bundles.prescribe_inputs["veget_max"].shape[1])
    allocation_kwargs = _paper_day_allocation_kwargs(
        run_def_values=run_def_values,
        run_scalars=run_scalars,
        nvm=nvm,
    )
    post_npp_inputs = {
        **bundles.post_npp_inputs,
        **_paper_day_post_npp_owned_daily_fields(bundles),
    }
    runner = (
        stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_outer_compiled
        if outer_compiled_static_dispatch is not None
        else stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_static_jit
        if bool(use_static_jit)
        else stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit
    )
    return runner(
        prescribe_inputs=bundles.prescribe_inputs,
        constraints_inputs=bundles.constraints_inputs,
        phenology_inputs=bundles.phenology_inputs,
        alloc_inputs=bundles.alloc_inputs,
        post_npp_inputs=post_npp_inputs,
        allocation_kwargs=allocation_kwargs,
        **(
            {"static_dispatch": outer_compiled_static_dispatch}
            if outer_compiled_static_dispatch is not None
            else {}
        ),
    )


def _paper_daily_carbon_static_dispatch(
    context: Paper1961PreparedDriverContext,
    state: DriverPreviousStepStatePacket,
) -> dict[str, object]:
    """Lift PFT dispatch controls outside a compiled multi-day transition."""

    parameters = context.stomate_static.parameters
    slowproc = state.fields_by_component["slowproc_stomate_previous_step_state"]
    active_pft_mask = tuple(
        bool(value)
        for value in np.any(np.asarray(slowproc["veget_max"]) > 0.0, axis=0)
    )
    bool_tuple = lambda values: tuple(bool(value) for value in np.asarray(values))
    float_tuple = lambda values: tuple(float(value) for value in np.asarray(values))
    return {
        "pres_ok_dgvm": bool(parameters.ok_dgvm),
        "pres_lpj_gap_const_mort": bool(parameters.lpj_gap_const_mort),
        "pres_natural": bool_tuple(parameters.natural),
        "pres_pasture": bool_tuple(parameters.pasture),
        "pres_is_tree": bool_tuple(parameters.is_tree),
        "pres_pheno_is_none": bool_tuple(parameters.pheno_is_none),
        "cons_natural": bool_tuple(parameters.natural),
        "cons_is_tree": bool_tuple(parameters.is_tree),
        "cons_is_peat": bool_tuple(parameters.is_peat),
        "cons_pheno_is_none": bool_tuple(parameters.pheno_is_none),
        "cons_tmin_crit": float_tuple(parameters.tmin_crit),
        "cons_tcm_crit": float_tuple(parameters.tcm_crit),
        "phenology_active_pft_mask": active_pft_mask,
        "pheno_model": tuple(parameters.pheno_model),
        "post_ok_dgvm": bool(parameters.ok_dgvm),
        "post_lpj_gap_const_mort": bool(parameters.lpj_gap_const_mort),
        "post_wire_vmax": True,
        "post_ok_nlim_vmax": bool(parameters.grm_n_limitation),
    }


def _paper_day_final_lai_from_daily_carbon(
    daily_carbon: ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult,
):
    """Return the most advanced source-backed LAI in the explicit day chain.

    Fortran provenance: ``src_stomate/stomate_lpj.f90::StomateLpj`` lines
    1264-1276 update ``lai`` in turnover; lines 1550-1557 run the final
    ``setlai`` before ``vmax``. The explicit result carries that unconditional
    final ``setlai`` result even when ``vmax`` itself is not wired.
    """

    if daily_carbon.post_npp.lai_after_setlai is not None:
        return daily_carbon.post_npp.lai_after_setlai
    return daily_carbon.post_npp.turnover.lai


def _paper_day_slowproc_surface_update(
    *,
    first: DriverTimestepScaffold,
    run_def_values: dict[str, str],
    daily_carbon: ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult | None,
):
    """Run the day-end SLOWPROC surface update after explicit STOMATE state.

    Fortran provenance: ``src_stomate/stomate.f90`` lines 4297-4335 pass
    mutable ``lai``/``veget_cov_max`` through ``StomateLpj``; afterwards
    ``src_sechiba/slowproc.f90::slowproc_main`` line 1103 calls
    ``slowproc_veget`` and lines 1116-1121 compute ``tot_bare_soil``.
    """

    if daily_carbon is None:
        return None
    restart = first.first_step_restart_state
    if restart is None:
        return None
    lai = _paper_day_final_lai_from_daily_carbon(daily_carbon)
    veget_max = (
        daily_carbon.post_npp.cover.veget_max
        if daily_carbon.post_npp.cover is not None
        else first.payload.veget_max
    )
    nvm = int(lai.shape[1])
    return slowproc_surface_update_explicit(
        lai=lai,
        frac_nobio=restart.slowproc.frac_nobio,
        veget_max=veget_max,
        pref_soil_veg=first.step.run_scalars.pref_soil_veg,
        ext_coeff_vegetfrac=_select_run_def_pft_vector(
            run_def_values, "EXT_COEFF_VEGETFRAC", first.step.run_scalars, nvm=nvm
        ),
        nstm=first.step.run_scalars.nstm,
        ok_dgvm=parse_run_def_bool(run_def_values["STOMATE_OK_DGVM"]),
    )


def _paper_later_day_slowproc_surface_update(
    *,
    previous_state: DriverPreviousStepStatePacket,
    run_def_values: dict[str, str],
    daily_carbon: ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult | None,
    step: DriverNextStepHydrolPrecallScaffold,
):
    """Run day-end SLOWPROC surface update for a later model-produced day."""

    if daily_carbon is None:
        return None
    slowproc_state = previous_state.fields_by_component.get("slowproc_stomate_previous_step_state", {})
    lai = _paper_day_final_lai_from_daily_carbon(daily_carbon)
    veget_max = (
        daily_carbon.post_npp.cover.veget_max
        if daily_carbon.post_npp.cover is not None
        else slowproc_state["veget_max"]
    )
    nvm = int(lai.shape[1])
    return slowproc_surface_update_explicit(
        lai=lai,
        frac_nobio=slowproc_state["frac_nobio"],
        veget_max=veget_max,
        pref_soil_veg=step.enerbil.step.run_scalars.pref_soil_veg,
        ext_coeff_vegetfrac=_select_run_def_pft_vector(
            run_def_values,
            "EXT_COEFF_VEGETFRAC",
            step.enerbil.step.run_scalars,
            nvm=nvm,
        ),
        nstm=step.enerbil.step.run_scalars.nstm,
        ok_dgvm=parse_run_def_bool(run_def_values["STOMATE_OK_DGVM"]),
    )


def _paper_later_day_slowproc_surface_update_from_runtime(
    *,
    previous_state: DriverPreviousStepStatePacket,
    run_def_values: dict[str, str],
    daily_carbon: ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult | None,
    metadata: DriverRuntimeStepMetadata,
    run_scalars: RunScalars,
):
    if daily_carbon is None:
        return None
    slowproc_state = previous_state.fields_by_component.get("slowproc_stomate_previous_step_state", {})
    lai = _paper_day_final_lai_from_daily_carbon(daily_carbon)
    veget_max = (
        daily_carbon.post_npp.cover.veget_max
        if daily_carbon.post_npp.cover is not None
        else slowproc_state["veget_max"]
    )
    nvm = int(lai.shape[1])
    return slowproc_surface_update_explicit(
        lai=lai,
        frac_nobio=slowproc_state["frac_nobio"],
        veget_max=veget_max,
        pref_soil_veg=metadata.pref_soil_veg,
        ext_coeff_vegetfrac=_select_run_def_pft_vector(
            run_def_values, "EXT_COEFF_VEGETFRAC", run_scalars, nvm=nvm
        ),
        nstm=metadata.nstm,
        ok_dgvm=parse_run_def_bool(run_def_values["STOMATE_OK_DGVM"]),
    )


def _paper_ok_leak_sechiba_adapter(
    *,
    hydrol: dict[str, object],
    therm: dict[str, object],
    veget_max,
    final_entry_payload: dict[str, object] | None,
):
    """Expose previous-state day fields through the SECHIBA result interface.

    Driver day scaffolds store same-day outputs in component dictionaries,
    while the source-backed OK_LEAK composition layer consumes the local
    ``SechibaExplicitCoupledStepResult`` interface. This adapter is deliberately
    data-only: it maps already-produced HYDROL/THERMOSOIL/SLOWPROC fields
    without computing any process values.
    """

    payload = {} if final_entry_payload is None else dict(final_entry_payload)
    return SimpleNamespace(
        hydrol_diagnostics=SimpleNamespace(
            mc_layh_s=hydrol["soil_mc"],
            shumdiag_peat=hydrol.get("shumdiag_peat"),
        ),
        hydrol_outputs=SimpleNamespace(
            wat_flux=hydrol["wat_flux"],
            runoff_per_soil=hydrol["runoff_per_soil"],
            drainage_per_soil=hydrol["drainage_per_soil"],
            runoff2peat=hydrol["runoff2peat"],
            canopy2ground=hydrol["canopy2ground"],
            precip2ground=hydrol["precip2ground"],
            precip2canopy=hydrol["precip2canopy"],
        ),
        thermosoil_payload={
            "deeptemp_prof": therm["tdeep"],
            "deephum_prof": therm["hsdeep"],
        },
        slowproc=SimpleNamespace(
            vegetation=SimpleNamespace(
                veget_max=veget_max,
            )
        ),
        diffuco_payload={
            "temp_sol": payload.get("temp_sol"),
        },
        enerbil_payload={},
    )


def _paper_ok_leak_boundary_from_sources(
    *,
    base,
    previous_state: DriverPreviousStepStatePacket | None,
    runtime: DriverRuntimeScalars,
    run_def_values: dict[str, str],
    daily_carbon: ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult | None,
    bundles: StomateRestartInputBundles | None,
    final_entry_payload: dict[str, object] | None,
    veget_max,
    pref_soil_veg,
    is_tree,
    is_peat,
    restart_altmax,
    restart_fixed_cryoturbation_depth,
    npts: int,
    dayno: int,
) -> tuple[StomateOkLeakBoundaryInputs | None, tuple[DriverDayStateGap, ...]]:
    """Assemble OK_LEAK inputs from explicit day-boundary sources."""

    gaps: list[DriverDayStateGap] = []
    if base is None or previous_state is None or daily_carbon is None or bundles is None:
        return None, (
            DriverDayStateGap(
                component="stomate_ok_leak_boundary",
                fields=("pre_step_boundary", "previous_step_state", "daily_carbon", "stomate_bundles"),
                provenance=("fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 3288-3489",),
                notes=("OK_LEAK waits for explicit STOMATE carbon and same-day SECHIBA boundary state.",),
            ),
        )

    hydrol = previous_state.fields_by_component.get("hydrol_previous_step_state", {})
    therm = previous_state.fields_by_component.get("thermosoil_previous_step_state", {})
    perma_peat_enabled = parse_run_def_bool(run_def_values["PERMA_PEAT"])
    hydrol_required = (
        "soil_mc",
        "wat_flux",
        "runoff_per_soil",
        "drainage_per_soil",
        "runoff2peat",
        "canopy2ground",
        "precip2ground",
        "precip2canopy",
        *(("shumdiag_peat",) if perma_peat_enabled else ()),
    )
    missing_hydrol = tuple(name for name in hydrol_required if name not in hydrol)
    missing_therm = tuple(name for name in ("tdeep", "hsdeep") if name not in therm)
    if missing_hydrol:
        gaps.append(
            DriverDayStateGap(
                component="stomate_ok_leak_hydrol_boundary",
                fields=missing_hydrol,
                provenance=("fortran_source/ORCHIDEE/src_hydrol/hydrol.f90::hydrol_main downstream outputs to stomate_main",),
            )
        )
    if missing_therm:
        gaps.append(
            DriverDayStateGap(
                component="stomate_ok_leak_thermosoil_boundary",
                fields=missing_therm,
                provenance=("fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_main lines 1021-1025",),
            )
        )
    if gaps:
        return None, tuple(gaps)

    frozen_respiration_func = parse_run_def_int(run_def_values, "frozen_respiration_func")
    adapter = _paper_ok_leak_sechiba_adapter(
        hydrol=hydrol,
        therm=therm,
        veget_max=veget_max,
        final_entry_payload=final_entry_payload,
    )
    same_step = stomate_same_step_ok_leak_boundary_inputs(
        sechiba=adapter,
        post_npp=daily_carbon.post_npp,
        dt_sechiba=runtime.dt_sechiba,
        entry_payload=final_entry_payload,
        pre_step_boundary=base,
        ok_leak_inputs={
            "rprof": bundles.maintenance_inputs["rprof"],
            "veget_max": veget_max,
            "sla_calc": bundles.alloc_inputs["sla_calc"],
        },
        litter_controls_boundary={
            "pref_soil_veg": pref_soil_veg,
            "frozen_respiration_func": frozen_respiration_func,
            "moist_func_moyano": parse_run_def_bool(run_def_values["MOIST_FUNC_MOYANO"]),
        },
        tf_doc_boundary={
            "ok_tf_doc": parse_run_def_bool(run_def_values["TF_DOC"]),
            "dt_days": runtime.dt_days,
            "conc_doc_rain": parse_run_def_float(run_def_values, "CONC_DOC_RAIN"),
        },
        perma_peat_boundary=(
            {
                "peat_bulk_density": PAPER_CASE_PEAT_BULK_DENSITY,
                "perma_peat": True,
                "frac1": parse_run_def_float(run_def_values, "FRAC1"),
                "frac2": parse_run_def_float(run_def_values, "FRAC2"),
            }
            if perma_peat_enabled
            else None
        ),
        active_layer_boundary={
            "restart_altmax": restart_altmax,
            "restart_fixed_cryoturbation_depth": restart_fixed_cryoturbation_depth,
            "dayno": dayno,
            "firstcall_soilcarbon": (int(dayno) == 1),
        },
        soilwater_boundary={"sro_bottom": int(base.ok_leak_inputs["nslm"])},
        permafrost_activity_boundary={
            "frozen_respiration_func": frozen_respiration_func,
            "perma_peat": perma_peat_enabled,
            **({} if perma_peat_enabled else {"mc_peat": hydrol["soil_mc"][:, :, 0]}),
            **({"is_peat": is_peat} if perma_peat_enabled else {}),
        },
        soil_mc_32l_boundary={},
        doc_transport_boundary=None if "flux_red" in base.ok_leak_inputs else {},
        is_tree=is_tree,
    )
    return same_step.boundary_inputs, tuple(gaps)


def _paper_day_ok_leak_boundary(
    *,
    first: DriverTimestepScaffold,
    previous_state: DriverPreviousStepStatePacket | None,
    runtime: DriverRuntimeScalars,
    run_def_values: dict[str, str],
    daily_carbon: ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult | None,
    bundles: StomateRestartInputBundles | None,
    final_entry_payload: dict[str, object] | None,
) -> tuple[StomateOkLeakBoundaryInputs | None, tuple[DriverDayStateGap, ...]]:
    """Assemble first-day OK_LEAK explicit inputs without fabricating branches.

    Fortran provenance: pre-step restart/static/vertical inputs come from
    ``stomate_pre_step_ok_leak_boundary_inputs``; same-day hydrology and
    thermosoil fields are produced before ``stomate_main`` calls OK_LEAK at
    ``src_stomate/stomate.f90`` lines 3288-3489. PERMA_PEAT remains a gap here
    until its peat bulk-density source is audited.
    """

    if first.stomate_pre_step is None:
        return None, (
            DriverDayStateGap(
                component="stomate_ok_leak_boundary",
                fields=("pre_step_boundary",),
                provenance=("fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 3288-3489",),
            ),
        )
    final_payload = dict(first.stomate_entry_assembly.payload)
    if final_entry_payload is not None:
        final_payload.update(final_entry_payload)
    return _paper_ok_leak_boundary_from_sources(
        base=first.stomate_pre_step.pre_step.boundary_inputs,
        previous_state=previous_state,
        runtime=runtime,
        run_def_values=run_def_values,
        daily_carbon=daily_carbon,
        bundles=bundles,
        final_entry_payload=final_payload,
        veget_max=first.payload.veget_max,
        pref_soil_veg=first.step.run_scalars.pref_soil_veg,
        is_tree=first.step.run_scalars.is_tree,
        is_peat=first.step.run_scalars.is_peat,
        restart_altmax=first.first_step_restart_state.stomate.altmax,
        restart_fixed_cryoturbation_depth=first.first_step_restart_state.stomate.fixed_cryoturbation_depth,
        npts=first.payload.kjpindex,
        dayno=1,
    )


def _paper_cold_start_ok_leak_boundary(
    *,
    first_step_coverage: ColdStartFirstStepStateCoverage,
    previous_state: DriverPreviousStepStatePacket | None,
    config_path: str | Path,
    run_def_path: Path,
    runtime: DriverRuntimeScalars,
    run_def_values: dict[str, str],
    daily_carbon: ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult | None,
    bundles: StomateRestartInputBundles | None,
    final_entry_payload: dict[str, object] | None,
) -> tuple[StomateOkLeakBoundaryInputs | None, tuple[DriverDayStateGap, ...]]:
    """Assemble cold-start OK_LEAK boundary after the first daily carbon chain."""

    payloads = first_step_coverage.initialized_payloads
    entry_state = payloads.get("stomate_cold_start_entry_state")
    slowproc_veg = payloads.get("slowproc_cold_start_vegetation")
    first_payload = payloads.get("first_step_payload")
    cwrr_grid = payloads.get("cwrr_grid")
    diaglev = payloads.get("diaglev")
    deep_zlt = payloads.get("deep_zlt")
    deep_znt = payloads.get("deep_znt")
    run_scalars_payload = payloads.get("run_scalars")
    if entry_state is None or slowproc_veg is None or first_payload is None or diaglev is None or deep_zlt is None or deep_znt is None:
        return None, (
            DriverDayStateGap(
                component="cold_start_ok_leak_boundary",
                fields=tuple(
                    name
                    for name, value in (
                        ("stomate_cold_start_entry_state", entry_state),
                        ("slowproc_cold_start_vegetation", slowproc_veg),
                        ("first_step_payload", first_payload),
                        ("diaglev", diaglev),
                        ("deep_zlt", deep_zlt),
                        ("deep_znt", deep_znt),
                    )
                    if value is None
                ),
                provenance=first_step_coverage.provenance,
            ),
        )
    run_scalars = run_scalars_payload if run_scalars_payload is not None else read_run_scalars(config_path, run_def_path=run_def_path)
    static = paper_case_stomate_static_input_kwargs(config_path, used_run_def=run_def_path)
    base = _paper_ok_leak_pre_step_boundary_from_entry_state(
        config_path=config_path,
        run_def_path=run_def_path,
        run_def_values=run_def_values,
        entry_state=entry_state,
        driver_payload=first_payload,
        run_scalars=run_scalars,
        diaglev=diaglev,
        deep_zlt=deep_zlt,
        deep_znt=deep_znt,
        kjit=1,
        nflow=run_scalars.nstm,
        final_entry_payload=final_entry_payload,
    )
    veget_max = slowproc_veg.vegetation.veget_max
    return _paper_ok_leak_boundary_from_sources(
        base=base,
        previous_state=previous_state,
        runtime=runtime,
        run_def_values=run_def_values,
        daily_carbon=daily_carbon,
        bundles=bundles,
        final_entry_payload=final_entry_payload,
        veget_max=veget_max,
        pref_soil_veg=run_scalars.pref_soil_veg,
        is_tree=static.parameters.is_tree,
        is_peat=static.parameters.is_peat,
        restart_altmax=entry_state.altmax,
        restart_fixed_cryoturbation_depth=entry_state.fixed_cryoturbation_depth,
        npts=int(np.asarray(entry_state.biomass).shape[0]),
        dayno=1,
    )


def _paper_day_ok_leak_from_boundary(
    *,
    daily_carbon: ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult | None,
    boundary: StomateOkLeakBoundaryInputs | None,
    runtime: DriverRuntimeScalars,
) -> ExplicitOkLeakFromPostNppResult | None:
    if daily_carbon is None or boundary is None:
        return None
    ok_inputs = dict(boundary.ok_leak_inputs)
    flood_root_radia = ok_inputs.get("flood_root_radia")
    if flood_root_radia is None:
        flood_root_radia = np.zeros_like(np.asarray(ok_inputs["veget_max"]))
    return stomate_ok_leak_from_post_npp_explicit(
        post_npp=daily_carbon.post_npp,
        resp_maint_part_radia=daily_carbon.post_npp.daily_carbon.boundary.resp_maint_part,
        flood_root_radia=flood_root_radia,
        soil_mc=ok_inputs.pop("soil_mc"),
        dt_sechiba=runtime.dt_sechiba,
        **ok_inputs,
    )


_OK_LEAK_HALF_HOUR_STATE_FIELDS = (
    "litter_above",
    "litter_below",
    "lignin_struc_above",
    "lignin_struc_below",
    "litterpart",
    "dead_leaves",
    "fuel_1hr",
    "fuel_10hr",
    "fuel_100hr",
    "fuel_1000hr",
    "carbon_32l",
    "DOC",
    "interception_storage",
)


def _paper_half_hour_ok_leak_state_updates(ok_leak) -> dict[str, object]:
    """Return the live STOMATE state written by one ``littercalc_leak`` call.

    Fortran calls ``littercalc_leak`` on every ``dt_sechiba`` entry before the
    daily ``StomateLpj`` gate (``stomate.f90`` lines 3288-3489 and 3589).
    These are therefore half-hour carry fields, rather than daily outputs.
    """

    litter = ok_leak.littercalc
    soilcarbon = ok_leak.soilcarbon
    return {
        "litter_above": litter.litter_above,
        "litter_below": litter.litter_below,
        "lignin_struc_above": litter.lignin_struc_above,
        "lignin_struc_below": litter.lignin_struc_below,
        "litterpart": litter.litterpart,
        "dead_leaves": litter.dead_leaves,
        "fuel_1hr": litter.fuel.fuel_1hr,
        "fuel_10hr": litter.fuel.fuel_10hr,
        "fuel_100hr": litter.fuel.fuel_100hr,
        "fuel_1000hr": litter.fuel.fuel_1000hr,
        "carbon_32l": soilcarbon.carbon_32l,
        "DOC": soilcarbon.doc,
        "interception_storage": ok_leak.interception_storage,
    }


def _paper_previous_state_with_stomate_updates(
    previous_state: DriverPreviousStepStatePacket,
    updates: Mapping[str, object],
) -> DriverPreviousStepStatePacket:
    """Overlay live STOMATE inout state without changing the driver timestep."""

    fields = {
        component: dict(component_fields)
        for component, component_fields in previous_state.fields_by_component.items()
    }
    fields.setdefault("slowproc_stomate_previous_step_state", {}).update(dict(updates))
    provenance = {
        component: tuple(component_provenance)
        for component, component_provenance in previous_state.provenance_by_component.items()
    }
    provenance["slowproc_stomate_previous_step_state"] = tuple(
        dict.fromkeys(
            (
                *provenance.get("slowproc_stomate_previous_step_state", ()),
                "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 3288-3489 half-hour OK_LEAK state writeback",
            )
        )
    )
    return DriverPreviousStepStatePacket(
        tstep=previous_state.tstep,
        fields_by_component=fields,
        provenance_by_component=provenance,
    )


class DriverCompiledOkLeakCarry(NamedTuple):
    litter_above: object
    litter_below: object
    lignin_struc_above: object
    lignin_struc_below: object
    litterpart: object
    dead_leaves: object
    fuel_1hr: object
    fuel_10hr: object
    fuel_100hr: object
    fuel_1000hr: object
    carbon_32l: object
    doc: object
    interception_storage: object


class DriverCompiledOkLeakStepInputs(NamedTuple):
    soil_mc: object
    wat_flux: object
    runoff_per_soil: object
    drainage_per_soil: object
    runoff2peat: object
    canopy2ground: object
    precip2ground: object
    precip2canopy: object
    temp_sol: object
    tdeep: object
    hsdeep: object
    shumdiag_peat: object
    resp_maint_part_radia: object


_COMPILED_OK_LEAK_DYNAMIC_NAMES = frozenset(
    {
        "litter_above",
        "litter_below",
        "lignin_struc_above",
        "lignin_struc_below",
        "litterpart",
        "dead_leaves",
        "fuel_1hr",
        "fuel_10hr",
        "fuel_100hr",
        "fuel_1000hr",
        "carbon_32l",
        "doc",
        "interception_storage",
        "bm_to_litter",
        "turnover",
        "soil_mc",
        "soil_mc_32l",
        "wat_flux",
        "soilwater_31mm",
        "runoff_per_soil",
        "drainage_per_soil",
        "runoff2peat",
        "canopy2ground",
        "doc_precip2ground",
        "doc_precip2canopy",
        "dry_dep_canopy",
        "fbact_litter",
        "fbact_soilcarbon",
        "fbact_doc",
        "tprof",
        "control_temp_above",
        "control_moist_above",
        "soil_mc_top_by_pft",
        "resp_maint_part_radia",
        "flood_root_radia",
        "dt_days",
    }
)


def _compiled_ok_leak_carry_from_mapping(values: Mapping[str, object]) -> DriverCompiledOkLeakCarry:
    return DriverCompiledOkLeakCarry(
        litter_above=values["litter_above"],
        litter_below=values["litter_below"],
        lignin_struc_above=values["lignin_struc_above"],
        lignin_struc_below=values["lignin_struc_below"],
        litterpart=values["litterpart"],
        dead_leaves=values["dead_leaves"],
        fuel_1hr=values["fuel_1hr"],
        fuel_10hr=values["fuel_10hr"],
        fuel_100hr=values["fuel_100hr"],
        fuel_1000hr=values["fuel_1000hr"],
        carbon_32l=values["carbon_32l"],
        doc=values["DOC" if "DOC" in values else "doc"],
        interception_storage=values["interception_storage"],
    )


def _compiled_ok_leak_updates(carry: DriverCompiledOkLeakCarry) -> dict[str, object]:
    values = carry._asdict()
    values["DOC"] = values.pop("doc")
    return values


def _select_compiled_ok_leak_scan_outputs(outputs, *, retain_step_results: bool):
    """Keep full scan diagnostics only when an explicit audit requests them."""

    if retain_step_results:
        return outputs
    return jax.tree_util.tree_map(lambda value: value[-1], outputs)


def _paper_compiled_ok_leak_fold(
    *,
    initial: DriverCompiledOkLeakCarry,
    steps: DriverCompiledOkLeakStepInputs,
    static_inputs: Mapping[str, object],
    turnover_daily,
    bm_to_litter_daily,
    biomass,
    veget_max,
    sla_calc,
    rprof,
    pref_soil_veg_fortran,
    is_tree,
    is_peat,
    nslm: int,
    ndeep: int,
    dt_sechiba: float,
    tf_doc_dt_days: float,
    frozen_respiration_func: int,
    moist_func_moyano: bool,
    ok_tf_doc: bool,
    perma_peat: bool,
    conc_doc_rain: float,
    retain_step_results: bool = False,
):
    """Compile the 48 source-ordered half-hour OK_LEAK state transitions."""

    nslm = int(nslm)
    ndeep = int(ndeep)
    sro_bottom = nslm
    dt_days = float(dt_sechiba) / 86400.0
    pref_zero = jnp.asarray(pref_soil_veg_fortran, dtype=jnp.int32) - 1
    z_soil = static_inputs["z_soil"]
    zi_soil = static_inputs["zi_soil"]
    poor_soils = static_inputs["poor_soils"]
    flood_frac = static_inputs["flood_frac"]
    turnover = jnp.asarray(turnover_daily) * dt_days
    bm_to_litter = jnp.asarray(bm_to_litter_daily) * dt_days
    static_ok_inputs = {
        name: value
        for name, value in static_inputs.items()
        if name not in _COMPILED_OK_LEAK_DYNAMIC_NAMES
        and name not in {"nslm", "ndeep", "perma_peat"}
    }

    layer_thickness = jnp.diff(jnp.asarray(z_soil)[: sro_bottom + 1])
    def body(carry: DriverCompiledOkLeakCarry, step: DriverCompiledOkLeakStepInputs):
        controls = littercalc_aboveground_controls(
            step.temp_sol,
            step.soil_mc,
            z_soil,
            pref_zero,
            frozen_respiration_func=frozen_respiration_func,
            moist_func_moyano=moist_func_moyano,
            zz_coef_deep=static_inputs.get("zf_soil_b"),
            bulk_dens=static_inputs.get("bulk_dens"),
            clay=static_inputs.get("clay"),
            carbon_32l=carry.carbon_32l,
            veget_max=veget_max,
        )
        tf_doc = soilcarbon_leak_tf_doc_inputs(
            step.precip2ground,
            step.precip2canopy,
            veget_max,
            biomass,
            is_tree,
            ok_tf_doc=ok_tf_doc,
            dt_days=tf_doc_dt_days,
            conc_doc_rain=conc_doc_rain,
        )
        if perma_peat:
            repeated = jnp.repeat(step.shumdiag_peat[:, -1:], ndeep - nslm, axis=1)
            mc_peat = jnp.concatenate((step.shumdiag_peat, repeated), axis=1)
        else:
            mc_peat = step.soil_mc[:, :, 0]
        activity = stomate_permafrost_decomposition_controls(
            step.tdeep - 273.15,
            step.hsdeep,
            zi_soil,
            mc_peat,
            poor_soils,
            frozen_respiration_func=frozen_respiration_func,
            perma_peat=perma_peat,
            agri_peat=False,
            is_peat=is_peat if perma_peat else None,
            tau_peat=static_inputs.get("tau_peat", 3.1536e8),
            z_tau=static_inputs.get("z_tau", 1.0e6),
        )
        soil_mc_32l = stomate_soil_mc_32l(step.soil_mc, ndeep=ndeep, nslm=nslm)
        soilwater_31mm = jnp.sum(
            step.soil_mc[:, :sro_bottom, :] * layer_thickness[None, :, None],
            axis=1,
        )
        _, flood_root_radia = sum_resp_maint_radia(
            step.resp_maint_part_radia,
            flood_frac=flood_frac,
        )
        ok_args = dict(static_ok_inputs)
        ok_args.update(
            carry._asdict(),
            bm_to_litter=bm_to_litter,
            turnover=turnover,
            rprof=rprof,
            veget_max=veget_max,
            sla_calc=sla_calc,
            control_temp_above=controls.control_temp_above,
            control_moist_above=controls.control_moist_above,
            soil_mc_top_by_pft=controls.soil_mc_top_by_pft,
            carbon_32l=carry.carbon_32l,
            doc=carry.doc,
            doc_precip2ground=tf_doc.doc_precip2ground,
            doc_precip2canopy=tf_doc.doc_precip2canopy,
            dry_dep_canopy=tf_doc.dry_dep_canopy,
            interception_storage=carry.interception_storage,
            canopy2ground=step.canopy2ground,
            fbact_litter=activity.prmfrst_soilc_tempctrl,
            fbact_soilcarbon=activity.prmfrst_soilc_tempctrl,
            fbact_doc=activity.prmfrst_soilc_tempctrl_doc,
            soil_mc=step.soil_mc,
            soil_mc_32l=soil_mc_32l,
            wat_flux=step.wat_flux,
            soilwater_31mm=soilwater_31mm,
            runoff_per_soil=step.runoff_per_soil,
            drainage_per_soil=step.drainage_per_soil,
            runoff2peat=step.runoff2peat,
            tprof=step.tdeep,
            resp_maint_part_radia=step.resp_maint_part_radia,
            flood_root_radia=flood_root_radia,
            dt_days=dt_days,
            nslm=nslm,
            ndeep=ndeep,
            sro_bottom=sro_bottom,
            perma_peat=perma_peat,
        )
        result = stomate_ok_leak_explicit(**ok_args)
        updates = _paper_half_hour_ok_leak_state_updates(result)
        next_carry = _compiled_ok_leak_carry_from_mapping(updates)
        return next_carry, result

    final_carry, outputs = jax.lax.scan(body, initial, steps)
    selected_outputs = _select_compiled_ok_leak_scan_outputs(
        outputs,
        retain_step_results=retain_step_results,
    )
    return final_carry, selected_outputs


_paper_compiled_ok_leak_fold_jit = jax.jit(
    _paper_compiled_ok_leak_fold,
    static_argnames=(
        "nslm",
        "ndeep",
        "dt_sechiba",
        "tf_doc_dt_days",
        "frozen_respiration_func",
        "moist_func_moyano",
        "ok_tf_doc",
        "perma_peat",
        "conc_doc_rain",
        "retain_step_results",
    ),
)


def _paper_half_hour_ok_leak_fold_from_entries(
    *,
    entry_payloads: tuple[dict[str, object], ...],
    maintenance_step_results: tuple[object, ...] = (),
    maintenance_resp_parts=None,
    initial_state: DriverPreviousStepStatePacket,
    pre_step_boundary: StomateOkLeakBoundaryInputs,
    runtime: DriverRuntimeScalars,
    run_def_values: dict[str, str],
    rprof,
    pref_soil_veg,
    is_tree,
    is_peat,
    dayno: int,
    use_compiled_ok_leak: bool = False,
    compiled_entry_stacks: Mapping[str, object] | None = None,
    capture_compiled_driver_steps: bool = False,
) -> tuple[object, dict[str, object]] | tuple[
    object, dict[str, object], DriverCompiledOkLeakStepInputs
]:
    """Advance the OK_LEAK state on each SECHIBA entry of one STOMATE day.

    The previous day's ``turnover_daily`` and ``bm_to_litter`` remain fixed
    throughout this fold. ``StomateLpj`` replaces them only after the final
    half-hour OK_LEAK call, for use by the next day. This directly follows
    ``stomate_main`` source order at lines 3288-3489 and 3589-4364.
    """

    if maintenance_resp_parts is None and len(entry_payloads) != len(maintenance_step_results):
        raise ValueError("OK_LEAK half-hour fold requires one maintenance result per entry payload")
    if maintenance_resp_parts is not None and int(maintenance_resp_parts.shape[0]) != len(entry_payloads):
        raise ValueError("OK_LEAK compiled maintenance stack must match entry payload count")
    if not entry_payloads:
        raise ValueError("OK_LEAK half-hour fold requires at least one entry payload")
    if capture_compiled_driver_steps and not use_compiled_ok_leak:
        raise ValueError("OK_LEAK driver-step capture requires the compiled fold")

    state = initial_state.fields_by_component["slowproc_stomate_previous_step_state"]
    required = (*_OK_LEAK_HALF_HOUR_STATE_FIELDS, "turnover_daily", "bm_to_litter", "biomass", "veget_max", "sla_calc")
    missing = tuple(name for name in required if name not in state)
    if missing:
        raise ValueError(f"OK_LEAK half-hour fold is missing carried STOMATE state: {missing}")

    current = {name: state[name] for name in _OK_LEAK_HALF_HOUR_STATE_FIELDS}
    last_result = None
    maintenance_items = (
        tuple(maintenance_step_results)
        if maintenance_resp_parts is None
        else (None,) * len(entry_payloads)
    )
    for step_index, (payload, maintenance) in enumerate(
        zip(entry_payloads, maintenance_items, strict=True)
    ):
        maintenance_part = (
            maintenance.resp_maint_part
            if maintenance_resp_parts is None
            else maintenance_resp_parts[step_index]
        )
        step_base_inputs = dict(pre_step_boundary.ok_leak_inputs)
        step_base_inputs.update(
            {("doc" if name == "DOC" else name): value for name, value in current.items()}
        )
        step_boundary = StomateOkLeakBoundaryInputs(
            ok_leak_inputs=step_base_inputs,
            output_inputs=pre_step_boundary.output_inputs,
            provenance=pre_step_boundary.provenance,
        )
        hydrol = {
            "soil_mc": payload["soil_mc"],
            "wat_flux": payload["wat_flux"],
            "runoff_per_soil": payload["runoff_per_soil"],
            "drainage_per_soil": payload["drainage_per_soil"],
            "runoff2peat": payload["runoff2peat"],
            "canopy2ground": payload["canopy2ground"],
            "precip2ground": payload["precip2ground"],
            "precip2canopy": payload["precip2canopy"],
            "shumdiag_peat": payload.get("shumdiag_peat"),
        }
        sechiba = _paper_ok_leak_sechiba_adapter(
            hydrol=hydrol,
            therm={"tdeep": payload["tdeep"], "hsdeep": payload["hsdeep"]},
            veget_max=state["veget_max"],
            final_entry_payload=payload,
        )
        post_npp = SimpleNamespace(turnover=SimpleNamespace(biomass=state["biomass"]))
        same_step = stomate_same_step_ok_leak_boundary_inputs(
            sechiba=sechiba,
            post_npp=post_npp,
            dt_sechiba=runtime.dt_sechiba,
            entry_payload=payload,
            pre_step_boundary=step_boundary,
            ok_leak_inputs={
                "rprof": rprof,
                "veget_max": state["veget_max"],
                "sla_calc": state["sla_calc"],
            },
            litter_controls_boundary={
                "pref_soil_veg": pref_soil_veg,
                "frozen_respiration_func": parse_run_def_int(run_def_values, "frozen_respiration_func"),
                "moist_func_moyano": parse_run_def_bool(run_def_values["MOIST_FUNC_MOYANO"]),
            },
            tf_doc_boundary={
                "ok_tf_doc": parse_run_def_bool(run_def_values["TF_DOC"]),
                "dt_days": runtime.dt_days,
                "conc_doc_rain": parse_run_def_float(run_def_values, "CONC_DOC_RAIN"),
                "interception_storage": current["interception_storage"],
            },
            perma_peat_boundary=(
                {
                    "peat_bulk_density": PAPER_CASE_PEAT_BULK_DENSITY,
                    "perma_peat": True,
                    "frac1": parse_run_def_float(run_def_values, "FRAC1"),
                    "frac2": parse_run_def_float(run_def_values, "FRAC2"),
                }
                if parse_run_def_bool(run_def_values["PERMA_PEAT"])
                else None
            ),
            active_layer_boundary={
                "restart_altmax": state["altmax"],
                "restart_fixed_cryoturbation_depth": state["fixed_cryoturbation_depth"],
                "dayno": dayno,
                "firstcall_soilcarbon": False,
            },
            soilwater_boundary={"sro_bottom": int(step_base_inputs["nslm"])},
            permafrost_activity_boundary={
                "frozen_respiration_func": parse_run_def_int(run_def_values, "frozen_respiration_func"),
                "perma_peat": parse_run_def_bool(run_def_values["PERMA_PEAT"]),
                **({} if parse_run_def_bool(run_def_values["PERMA_PEAT"]) else {"mc_peat": payload["soil_mc"][:, :, 0]}),
                **({"is_peat": is_peat} if parse_run_def_bool(run_def_values["PERMA_PEAT"]) else {}),
            },
            soil_mc_32l_boundary={},
            doc_transport_boundary=None if "flux_red" in step_base_inputs else {},
            is_tree=is_tree,
        )
        prep = stomate_littercalc_entry_prep(
            payload["soil_mc"],
            state["turnover_daily"],
            state["bm_to_litter"],
            dt_sechiba=runtime.dt_sechiba,
            ndeep=int(step_base_inputs["ndeep"]),
            nslm=int(step_base_inputs["nslm"]),
        )
        ok_args = dict(same_step.boundary_inputs.ok_leak_inputs)
        _, flood_root_radia = sum_resp_maint_radia(
            maintenance_part,
            flood_frac=ok_args["flood_frac"],
        )
        ok_args.update(
            {
                **{name: value for name, value in current.items() if name != "DOC"},
                "doc": current["DOC"],
                "bm_to_litter": prep.bm_to_littercalc,
                "turnover": prep.turnover_littercalc,
                "soil_mc": payload["soil_mc"],
                "soil_mc_32l": prep.soil_mc_32l,
                "resp_maint_part_radia": maintenance_part,
                "flood_root_radia": flood_root_radia,
                "dt_days": runtime.dt_sechiba / 86400.0,
            }
        )
        if use_compiled_ok_leak and step_index == 0:
            def stacked(name: str):
                if compiled_entry_stacks is not None:
                    return compiled_entry_stacks[name]
                return jnp.stack([item[name] for item in entry_payloads])

            series = DriverCompiledOkLeakStepInputs(
                soil_mc=stacked("soil_mc"),
                wat_flux=stacked("wat_flux"),
                runoff_per_soil=stacked("runoff_per_soil"),
                drainage_per_soil=stacked("drainage_per_soil"),
                runoff2peat=stacked("runoff2peat"),
                canopy2ground=stacked("canopy2ground"),
                precip2ground=stacked("precip2ground"),
                precip2canopy=stacked("precip2canopy"),
                temp_sol=stacked("temp_sol"),
                tdeep=stacked("tdeep"),
                hsdeep=stacked("hsdeep"),
                shumdiag_peat=stacked("shumdiag_peat"),
                resp_maint_part_radia=jnp.stack(
                    [item.resp_maint_part for item in maintenance_step_results]
                )
                if maintenance_resp_parts is None
                else jnp.asarray(
                    maintenance_resp_parts
                ),
            )
            static_inputs = {
                name: value
                for name, value in ok_args.items()
                if name not in _COMPILED_OK_LEAK_DYNAMIC_NAMES
                and name not in {"nslm", "ndeep"}
            }
            final_carry, last_result = _paper_compiled_ok_leak_fold_jit(
                initial=_compiled_ok_leak_carry_from_mapping(current),
                steps=series,
                static_inputs=static_inputs,
                turnover_daily=state["turnover_daily"],
                bm_to_litter_daily=state["bm_to_litter"],
                biomass=state["biomass"],
                veget_max=state["veget_max"],
                sla_calc=state["sla_calc"],
                rprof=rprof,
                pref_soil_veg_fortran=pref_soil_veg,
                is_tree=is_tree,
                is_peat=is_peat,
                nslm=int(ok_args["nslm"]),
                ndeep=int(ok_args["ndeep"]),
                dt_sechiba=runtime.dt_sechiba,
                tf_doc_dt_days=runtime.dt_days,
                frozen_respiration_func=parse_run_def_int(
                    run_def_values, "frozen_respiration_func"
                ),
                moist_func_moyano=parse_run_def_bool(
                    run_def_values["MOIST_FUNC_MOYANO"]
                ),
                ok_tf_doc=parse_run_def_bool(run_def_values["TF_DOC"]),
                perma_peat=parse_run_def_bool(run_def_values["PERMA_PEAT"]),
                conc_doc_rain=parse_run_def_float(run_def_values, "CONC_DOC_RAIN"),
            )
            updates = _compiled_ok_leak_updates(final_carry)
            if capture_compiled_driver_steps:
                return last_result, updates, series
            return last_result, updates
        last_result = stomate_ok_leak_explicit(**ok_args)
        current = _paper_half_hour_ok_leak_state_updates(last_result)

    if last_result is None:
        raise RuntimeError("OK_LEAK half-hour fold did not execute")
    return last_result, current


PAPER_DAY_ZERO_DAILY_RESET_FIELDS = (
    # Fortran provenance: src_stomate/stomate.f90::stomate_main lines 4944-4969.
    "humrel_daily",
    "litterhum_daily",
    "t2m_daily",
    "wspeed_daily",
    "tsurf_daily",
    "tsoil_daily",
    "soilhum_daily",
    "precip_daily",
    "gpp_daily",
    "resp_maint_part",
    "snowfall_daily",
    "snowmass_daily",
    "tmc_topgrass_daily",
    "tdeep_daily",
    "hsdeep_daily",
    "prmfrst_soilc_tempctrl_daily",
    "prmfrst_soilc_tempctrl_doc_daily",
    "snow_daily",
    "pb_pa_daily",
    "temp_sol_daily",
    "snowdz_daily",
    "snowrho_daily",
    # Fortran provenance: src_stomate/stomate.f90::stomate_main lines 4994-5000.
    "fwet_daily",
    "liqwt_daily",
)


def _paper_day_outputs_from_ok_leak(
    *,
    daily_fold: StomateDailyProcessFold | None,
    daily_carbon: ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult | None,
    ok_leak: ExplicitOkLeakFromPostNppResult | None,
    boundary: StomateOkLeakBoundaryInputs | None,
) -> ExplicitStomateLpjOutputsResult | None:
    """Build end-of-``StomateLpj`` diagnostics after OK_LEAK.

    Fortran provenance: ``src_stomate/stomate_lpj.f90::StomateLpj`` lines
    1578-1663 compute live/litter/soil/DOC pools and carbon mass diagnostics;
    lines 1809-1880 and 2200-2246 write the paper history/modelout fields.
    """

    if daily_fold is None or daily_carbon is None or ok_leak is None or boundary is None:
        return None
    output_inputs = dict(boundary.output_inputs)
    return stomate_lpj_outputs_from_post_npp_ok_leak_explicit(
        post_npp=daily_carbon.post_npp,
        ok_leak=ok_leak,
        veget_max=output_inputs["veget_max"],
        z_soil=output_inputs["z_soil"],
        zf_soil=output_inputs["zf_soil"],
        carb_mass_total_old=output_inputs["carb_mass_total_old"],
        gpp_daily=daily_fold.daily_fields["gpp_daily"],
        prod10_total=output_inputs.get("prod10_total", 0.0),
        prod100_total=output_inputs.get("prod100_total", 0.0),
        contfrac=output_inputs.get("contfrac", 1.0),
        one_day=output_inputs.get("one_day", 86400.0),
    )


def _paper_day_daily_reset_fields(
    *,
    daily_fold: StomateDailyProcessFold | None,
    do_slow: bool,
) -> dict[str, object] | None:
    """Return modeled daily accumulator state immediately after do_slow reset.

    Fortran provenance: ``src_stomate/stomate.f90::stomate_main`` lines
    4944-5000. ``t2m_min_daily``/``t2m_max_daily`` are extrema sentinels
    (``large_value`` and ``-large_value`` from
    ``src_parameters/constantes_var.f90`` line 178), while the listed modeled
    accumulator fields are reset to zero.
    """

    if daily_fold is None or not daily_fold.ok:
        return None
    zero_names = tuple(name for name in PAPER_DAY_ZERO_DAILY_RESET_FIELDS if name in daily_fold.daily_fields)
    result = reset_daily_on_slow(daily_fold.daily_fields, do_slow, zero_names)
    if do_slow and "t2m_min_daily" in result:
        result["t2m_min_daily"] = jnp.full_like(result["t2m_min_daily"], 1.0e33)
    if do_slow and "t2m_max_daily" in result:
        result["t2m_max_daily"] = jnp.full_like(result["t2m_max_daily"], -1.0e33)
    return result


STOMATE_DAY_STATE_PERSISTED_ENTRY_FIELDS = (
    "assim_param",
    "fixed_cryoturbation_depth",
    "fpeat",
    "veget_lastlight",
    "need_adjacent",
    "carbon",
    "litter",
    "lignin_struc",
    "prod10",
    "prod100",
    "flux10",
    "flux100",
    "deepC_a",
    "deepC_s",
    "deepC_p",
    "soilc_total",
    "thawed_humidity",
    "depth_organic_soil",
)

STOMATE_DAY_SEASON_MEMORY_FIELDS = tuple(
    name for name in SeasonMemoryState._fields if name in StomateRestartSeasonState._fields
)
STOMATE_DAY_SEASON_ANNUAL_FIELDS = (
    "gpp_week",
    "maxmoiavail_lastyear",
    "maxmoiavail_thisyear",
    "minmoiavail_lastyear",
    "minmoiavail_thisyear",
    "maxgppweek_lastyear",
    "maxgppweek_thisyear",
    "gdd0_lastyear",
    "gdd0_thisyear",
    "precip_lastyear",
    "precip_thisyear",
    "lm_thisyearmax",
    "maxfpc_lastyear",
    "maxfpc_thisyear",
)
STOMATE_DAY_SEASON_BIOMETEOROLOGY_FIELDS = (
    "gdd_m5_dormance",
    "gdd_midwinter",
    "ncd_dormance",
    "ngd_minus5",
    "time_hum_min",
    "hum_min_dormance",
)
STOMATE_DAY_SEASON_PERSISTED_FIELDS = (
    "begin_leaves",
    "gdd_init_date",
    "date",
)
STOMATE_DAY_SEASON_STATE_FIELDS = (
    *STOMATE_DAY_SEASON_MEMORY_FIELDS,
    *STOMATE_DAY_SEASON_ANNUAL_FIELDS,
    *STOMATE_DAY_SEASON_BIOMETEOROLOGY_FIELDS,
    *STOMATE_DAY_SEASON_PERSISTED_FIELDS,
)

STOMATE_DAY_END_EXPLICIT_WRITEBACK_FIELDS = (
    "pft_present", "everywhere", "when_growthinit", "biomass", "leaf_frac",
    "frac_age", "height",
    "leaf_age", "age", "sla_calc", "ind", "cn_ind", "co2_to_bm", "adapted",
    "regenerate", "begin_leaves", "npp_longterm", "turnover_longterm",
    "lm_lastyearmax", "turnover_daily", "bm_to_litter", "senescence",
    "turnover_time", "rip_time", "gpp_daily", "npp_daily", "resp_maint",
    "resp_growth", "resp_maint_part", "litter_above", "litter_below",
    "litterpart", "dead_leaves", "fuel_1hr", "fuel_10hr", "fuel_100hr",
    "fuel_1000hr", "lignin_struc_above", "lignin_struc_below", "carbon_32l",
    "DOC", "interception_storage", "carb_mass_total", "lai", "frac_nobio",
    "veget_max", "veget", "soiltile", "totfrac_nobio", "tot_bare_soil",
    "altmax", "fixed_cryoturbation_depth", "assim_param", "temp_growth",
    "fpeat", "soilc_total", "thawed_humidity", "depth_organic_soil",
    "dt_days_read", "resp_hetero", "tsurf_year", "deepC_peat",
    "depth_deepsoil", "t2m_14",
)

STOMATE_DAY_END_WRITEBACK_FIELDS = tuple(
    dict.fromkeys(
        (
            "daily_accumulators",
            *STOMATE_DAY_END_EXPLICIT_WRITEBACK_FIELDS,
            *STOMATE_DAY_STATE_PERSISTED_ENTRY_FIELDS,
            *STOMATE_DAY_SEASON_STATE_FIELDS,
            *STOMATE_FIXED_PATH_CARRY_FIELDS,
        )
    )
)


def _paper_day_dynamic_assim_param_from_daily_carbon(
    daily_carbon: ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult,
):
    """Return STOMATE-updated photosynthesis parameters for next DIFFUCO.

    Fortran provenance: ``src_stomate/stomate.f90`` lines 4567-4569 zero
    ``assim_param(:,:,ivcmax)`` and then writes ``vcmax(:,j)`` for ``j=2,nvm``
    after ``StomateLpj`` has called ``stomate_vmax::vmax``.
    """

    vmax = daily_carbon.post_npp.vmax
    if vmax is None:
        return None
    vcmax = jnp.asarray(vmax.vcmax)
    return vcmax[:, :, None]


def _paper_day_temp_growth_from_season_fields(season_memory_fields: Mapping[str, object] | None):
    """Return day-end ``temp_growth`` from updated monthly temperature.

    Fortran provenance: ``src_stomate/stomate.f90`` line 5049 assigns
    ``temp_growth(:)=t2m_month(:)-tp_00`` after the daily STOMATE update.
    """

    if season_memory_fields is None or "t2m_month" not in season_memory_fields:
        return None
    return jnp.asarray(season_memory_fields["t2m_month"]) - 273.15


def paper_day_active_restart_producer_fields(
    *,
    ok_leak: ExplicitOkLeakFromPostNppResult,
    previous_slowproc: Mapping[str, object],
    dt_days: float,
) -> dict[str, object]:
    """Return non-season active-path restart producers at the day boundary."""

    if previous_slowproc.get("depth_deepsoil") is None:
        raise ValueError("day-end restart writeback requires source-backed depth_deepsoil carry state")
    perma_peat = ok_leak.ok_leak.soilcarbon.perma_peat
    if perma_peat is None:
        if previous_slowproc.get("deepC_peat") is None:
            raise ValueError("inactive PERMA_PEAT writeback requires source-backed deepC_peat carry state")
        deepc_peat = previous_slowproc["deepC_peat"]
    else:
        deepc_peat = perma_peat.deepc_peat
    return {
        # stomate_io.f90::writerestart lines 2181-2185 writes current
        # dt_days, not the value read from the input restart.
        "dt_days_read": float(dt_days),
        # stomate.f90::stomate_main line 4958 resets resp_hetero_d before
        # writerestart receives it at lines 5467 and 5496.
        "resp_hetero": jnp.zeros_like(ok_leak.ok_leak.soilcarbon.resp_hetero_soil),
        "deepC_peat": deepc_peat,
        # slowproc_initialize lines 278-290 receives stomate_initialize's
        # INTENT(out) ddeepsoil. No slowproc_main assignment follows;
        # slowproc_finalize lines 1158-1164/1275-1281 passes the same
        # source object to stomate_finalize writer at stomate.f90 5496.
        "depth_deepsoil": previous_slowproc["depth_deepsoil"],
    }


def _paper_day_first_stomate_state_packet(
    *,
    daily_carbon: ExplicitDailyCarbonPrescribeConstraintsAllocKillGapTurnoverResult | None,
    ok_leak: ExplicitOkLeakFromPostNppResult | None,
    outputs: ExplicitStomateLpjOutputsResult | None,
    daily_reset_fields: dict[str, object] | None,
    ok_leak_boundary: StomateOkLeakBoundaryInputs | None = None,
    season_memory_fields: dict[str, object] | None = None,
    slowproc_surface_update=None,
    previous_state: DriverPreviousStepStatePacket | None = None,
    dt_days: float | None = None,
) -> DriverFirstDayStomateStatePacket | None:
    """Collect only first-day STOMATE fields updated by explicit processes.

    This is not a complete STOMATE restart image. It is the source-backed state
    subset produced after the first daily ``StomateLpj`` and OK_LEAK calls.
    """

    if daily_carbon is None or ok_leak is None or outputs is None or daily_reset_fields is None:
        return None
    post = daily_carbon.post_npp
    daily = post.daily_carbon
    final_leaf_age = post.vmax.leaf_age if post.vmax is not None else post.turnover.leaf_age
    final_leaf_frac = post.vmax.leaf_frac if post.vmax is not None else post.turnover.leaf_frac
    final_crown = (
        post.crown_after_establish
        if post.crown_after_establish is not None
        else post.crown_after_npp
    )
    fields: dict[str, object] = {
        "pft_present": daily_carbon.prescribe.pft_present,
        "everywhere": daily_carbon.prescribe.everywhere,
        "when_growthinit": post.kill_after_gap.when_growthinit,
        "biomass": post.turnover.biomass,
        "leaf_frac": final_leaf_frac,
        # LAI_MAP is false in the paper protocol, so stomate_main copies the
        # final leaf fractions into the slowproc-managed age fractions.
        "frac_age": final_leaf_frac,
        "height": final_crown.height,
        "leaf_age": final_leaf_age,
        "age": post.turnover.age,
        "sla_calc": daily.age_sla.sla_calc,
        "ind": post.kill_after_gap.ind,
        "cn_ind": post.kill_after_gap.cn_ind,
        "co2_to_bm": daily_carbon.prescribe.co2_to_bm,
        "adapted": daily_carbon.constraints.adapted,
        "regenerate": daily_carbon.constraints.regenerate,
        "begin_leaves": (
            daily_carbon.phenology.begin_leaves
            if daily_carbon.phenology is not None
            else previous_state.fields_by_component.get("slowproc_stomate_previous_step_state", {}).get("begin_leaves")
            if previous_state is not None
            else None
        ),
        "npp_longterm": post.kill_after_gap.npp_longterm,
        "turnover_daily": post.turnover.turnover,
        "bm_to_litter": post.bm_to_litter,
        "senescence": post.turnover.senescence,
        "turnover_time": post.turnover.turnover_time,
        "rip_time": post.kill_after_gap.rip_time,
        "gpp_daily": daily_reset_fields["gpp_daily"],
        "npp_daily": daily.npp_update.npp,
        "resp_maint": daily.npp_update.resp_maint,
        "resp_growth": daily.npp_update.resp_growth,
        "resp_maint_part": daily_reset_fields["resp_maint_part"],
        "litter_above": ok_leak.ok_leak.littercalc.litter_above,
        "litter_below": ok_leak.ok_leak.littercalc.litter_below,
        "lignin_struc_above": ok_leak.ok_leak.littercalc.lignin_struc_above,
        "lignin_struc_below": ok_leak.ok_leak.littercalc.lignin_struc_below,
        "litterpart": ok_leak.ok_leak.littercalc.litterpart,
        "dead_leaves": ok_leak.ok_leak.littercalc.dead_leaves,
        "fuel_1hr": ok_leak.ok_leak.littercalc.fuel.fuel_1hr,
        "fuel_10hr": ok_leak.ok_leak.littercalc.fuel.fuel_10hr,
        "fuel_100hr": ok_leak.ok_leak.littercalc.fuel.fuel_100hr,
        "fuel_1000hr": ok_leak.ok_leak.littercalc.fuel.fuel_1000hr,
        "carbon_32l": ok_leak.ok_leak.soilcarbon.carbon_32l,
        "DOC": ok_leak.ok_leak.soilcarbon.doc,
        "interception_storage": ok_leak.ok_leak.interception_storage,
        "carb_mass_total": outputs.output_diagnostics.carb_mass_total,
        "daily_accumulators": dict(daily_reset_fields),
    }
    if ok_leak_boundary is not None and "altmax" in ok_leak_boundary.output_inputs:
        fields["altmax"] = ok_leak_boundary.output_inputs["altmax"]
    if ok_leak_boundary is not None and "fixed_cryoturbation_depth" in ok_leak_boundary.ok_leak_inputs:
        fields["fixed_cryoturbation_depth"] = ok_leak_boundary.ok_leak_inputs["fixed_cryoturbation_depth"]
    if season_memory_fields is not None:
        for name, value in season_memory_fields.items():
            fields[name] = value
    elif previous_state is not None:
        previous_slowproc = previous_state.fields_by_component.get("slowproc_stomate_previous_step_state", {})
        for name in ("t2m_longterm",):
            if name in previous_slowproc:
                fields[name] = previous_slowproc[name]
    if previous_state is not None:
        previous_slowproc = previous_state.fields_by_component.get("slowproc_stomate_previous_step_state", {})
        for name in STOMATE_DAY_STATE_PERSISTED_ENTRY_FIELDS:
            if name in previous_slowproc:
                fields[name] = previous_slowproc[name]
        if "gdd_init_date" in previous_slowproc:
            fields["gdd_init_date"] = previous_slowproc["gdd_init_date"]
    else:
        previous_slowproc = {}

    if dt_days is None:
        raise ValueError("day-end restart writeback requires the active STOMATE dt_days")
    fields.update(
        paper_day_active_restart_producer_fields(
            ok_leak=ok_leak,
            previous_slowproc=previous_slowproc,
            dt_days=dt_days,
        )
    )
    dynamic_assim_param = _paper_day_dynamic_assim_param_from_daily_carbon(daily_carbon)
    if dynamic_assim_param is not None:
        fields["assim_param"] = dynamic_assim_param
    dynamic_temp_growth = _paper_day_temp_growth_from_season_fields(season_memory_fields)
    if dynamic_temp_growth is not None:
        fields["temp_growth"] = dynamic_temp_growth
    if slowproc_surface_update is not None:
        fields.update(
            {
                "lai": _paper_day_final_lai_from_daily_carbon(daily_carbon),
                "frac_nobio": slowproc_surface_update.vegetation.frac_nobio,
                "veget_max": slowproc_surface_update.vegetation.veget_max,
                "veget": slowproc_surface_update.vegetation.veget,
                "soiltile": slowproc_surface_update.vegetation.soiltile,
                "totfrac_nobio": slowproc_surface_update.vegetation.totfrac_nobio,
                "tot_bare_soil": slowproc_surface_update.tot_bare_soil,
            }
        )
    provenance_by_field: dict[str, tuple[str, ...]] = {}
    carbon_prov = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj lines 1070-1557",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 4225-4364",
    )
    ok_leak_prov = (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 3288-3489",
        "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90::littercalc_leak lines 1950-2799",
        "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::soilcarbon_leak lines 1176-2303",
    )
    output_prov = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj lines 1578-1663",
    )
    reset_prov = (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 4944-5000",
    )
    slowproc_surface_prov = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj lines 1264-1276 and 1550-1557",
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 1103-1121",
    )
    for name in fields:
        if name in {
            "litter_above",
            "litter_below",
            "litterpart",
            "dead_leaves",
            "fuel_1hr",
            "fuel_10hr",
            "fuel_100hr",
            "fuel_1000hr",
            "lignin_struc_above",
            "lignin_struc_below",
            "carbon_32l",
            "DOC",
            "interception_storage",
            "altmax",
            "fixed_cryoturbation_depth",
        }:
            provenance_by_field[name] = ok_leak_prov
        elif name in STOMATE_DAY_STATE_PERSISTED_ENTRY_FIELDS:
            provenance_by_field[name] = (
                "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 2409-2448 and 2899-3522 keep STOMATE inout state live across calls",
                "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90 lines 973-1121 carries slowproc/STOMATE state to the next SECHIBA entry",
            )
        elif name == "begin_leaves":
            provenance_by_field[name] = (
                "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj lines 1068-1102 calls phenology before allocation",
                "fortran_source/ORCHIDEE/src_stomate/stomate_phenology.f90::phenology lines 356-563 updates begin_leaves for the audited active path",
            )
        elif name == "frac_age":
            provenance_by_field[name] = (
                "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 4201-4218 copies leaf_frac to frac_age when LAI_MAP=false",
                "fortran_run_scripts/paper_250919/run.def.vn line 85 sets LAI_MAP=n",
            )
        elif name == "height":
            provenance_by_field[name] = (
                "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 4310-4368 passes height as StomateLpj inout state",
                "fortran_source/ORCHIDEE/src_stomate/lpj_crown.f90::crown lines 78-201 updates tree height",
            )
        elif name in {"leaf_age", "leaf_frac"} and post.vmax is not None:
            provenance_by_field[name] = (
                "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj lines 1556-1559 calls stomate_vmax::vmax after turnover and setlai",
                "fortran_source/ORCHIDEE/src_stomate/stomate_vmax.f90::vmax lines 105-363 updates leaf_age, leaf_frac, and vcmax as inout state",
            )
        elif name == "gdd_init_date":
            provenance_by_field[name] = (
                "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 539-543 updates gdd_init_date from downward_solar_flux",
                "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 4221-4251 passes gdd_init_date as season inout state",
            )
        elif name == "date":
            provenance_by_field[name] = (
                "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90 lines 656-662 increments date by NINT(dt_days) at LastTsDay",
                "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90 lines 2187-2195 writes date to restart",
            )
        elif name in {"carb_mass_total"}:
            provenance_by_field[name] = output_prov
        elif name in {"gpp_daily", "resp_maint_part", "daily_accumulators"}:
            provenance_by_field[name] = reset_prov
        elif name == "resp_hetero":
            provenance_by_field[name] = (
                "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3364, 3378, 3476, 3561, and 3648 accumulate resp_hetero_d",
                "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main line 4958 resets resp_hetero_d before writerestart line 5467",
            )
        elif name == "dt_days_read":
            provenance_by_field[name] = (
                "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_finalize lines 5445-5496 passes current dt_days to writerestart",
                "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::writerestart lines 2181-2185 writes dt_days",
            )
        elif name == "deepC_peat":
            provenance_by_field[name] = (
                "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::soilcarbon_leak lines 1375-1437 recomputes and redistributes deepC_peat when perma_peat",
            )
        elif name == "depth_deepsoil":
            provenance_by_field[name] = (
                "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_initialize lines 278-290 receives stomate_initialize INTENT(out) depth_deepsoil",
                "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_finalize lines 1158-1164 and 1275-1281 carries it unchanged to stomate_finalize",
                "source audit: slowproc_main contains no post-initialize assignment to depth_deepsoil",
            )
        elif name in STOMATE_DAY_SEASON_MEMORY_FIELDS:
            provenance_by_field[name] = (
                "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 4221-4251 calls season before StomateLpj",
                "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 577-772 updates season memory",
            )
        elif name in STOMATE_DAY_SEASON_ANNUAL_FIELDS or name in {"turnover_longterm", "lm_lastyearmax"}:
            provenance_by_field[name] = (
                "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 4221-4251 calls season before StomateLpj",
                "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 1147-1580 updates annual and long-term season state",
            )
        elif name in STOMATE_DAY_SEASON_BIOMETEOROLOGY_FIELDS:
            provenance_by_field[name] = (
                "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 4221-4251 calls season before StomateLpj",
                "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 848-1131 updates dormancy/GDD season memory",
            )
        elif name in {"lai", "frac_nobio", "veget_max", "veget", "soiltile", "totfrac_nobio", "tot_bare_soil"}:
            provenance_by_field[name] = slowproc_surface_prov
        else:
            provenance_by_field[name] = carbon_prov
    return DriverFirstDayStomateStatePacket(
        day_index=1,
        fields=fields,
        provenance_by_field=provenance_by_field,
    )


def _paper_day_end_state_packet(
    *,
    previous_state: DriverPreviousStepStatePacket | None,
    stomate_state: DriverFirstDayStomateStatePacket | None,
) -> DriverPreviousStepStatePacket | None:
    """Merge first-day STOMATE state into the last half-hour SECHIBA state."""

    if previous_state is None or stomate_state is None:
        return None
    fields = {
        component: dict(component_fields)
        for component, component_fields in previous_state.fields_by_component.items()
    }
    provenance = {
        component: tuple(component_provenance)
        for component, component_provenance in previous_state.provenance_by_component.items()
    }
    slowproc = fields.setdefault("slowproc_stomate_previous_step_state", {})
    for name, value in stomate_state.fields.items():
        if name == "daily_accumulators":
            slowproc[name] = value
        elif (
            name in STOMATE_DAY_END_EXPLICIT_WRITEBACK_FIELDS
            or name in STOMATE_DAY_STATE_PERSISTED_ENTRY_FIELDS
            or name in STOMATE_DAY_SEASON_STATE_FIELDS
        ):
            slowproc[name] = value
            if name in {"lai", "frac_nobio", "veget_max", "veget", "tot_bare_soil"}:
                fields.setdefault("diffuco_previous_step_state", {})[name] = value
    if "sechiba_finalize_state" in fields:
        fields["sechiba_finalize_state"] = _sechiba_finalize_state_after_slowproc(
            fields["sechiba_finalize_state"], slowproc
        )
    provenance["slowproc_stomate_previous_step_state"] = tuple(
        dict.fromkeys(
            (
                *provenance.get("slowproc_stomate_previous_step_state", ()),
                "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 4225-4364 daily STOMATE state update",
                "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 3288-3489 OK_LEAK state update",
                "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 4944-5000 daily accumulator reset",
                "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90 lines 1103-1121 day-end surface update",
            )
        )
    )
    if any(name in fields.get("diffuco_previous_step_state", {}) for name in ("lai", "veget", "veget_max", "tot_bare_soil")):
        provenance["diffuco_previous_step_state"] = tuple(
            dict.fromkeys(
                (
                    *provenance.get("diffuco_previous_step_state", ()),
                    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90 lines 1103-1121 day-end DIFFUCO surface boundary",
                )
            )
        )
    return DriverPreviousStepStatePacket(
        tstep=previous_state.tstep,
        fields_by_component=fields,
        provenance_by_component=provenance,
    )


def paper_1961_driver_day_scaffold(
    config_path: str | Path,
    *,
    year: int = 1961,
    start_tstep: int = 0,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    used_run_def_path: str | Path | None = None,
    reference_run_dir: str | Path | None = None,
    root: str | Path | None = None,
    prepared_context: Paper1961PreparedDriverContext | None = None,
    module_jit: bool = True,
    diffuco_local_jit: bool = True,
    retain_stomate_step_results: bool = True,
    maintenance_local_jit: bool = False,
    single_pass_daily_fold: bool = False,
    use_numpy_accumulator: bool = False,
    use_static_jit_daily_carbon: bool = False,
) -> DriverDayScaffold:
    """Build a strict first-day scaffold up to the first real state gap.

    The first STOMATE day spans ``dt_stomate / dt_sechiba`` driver steps. For
    the paper case this is 48 half-hour calls, with ``do_slow`` true only at
    ``tstep=47``. This helper refuses to reuse the initial restart as the
    previous-step state for later half-hour calls.
    """

    context = prepared_context or prepare_paper_1961_driver_context(
        config_path,
        used_run_def_path=used_run_def_path,
        reference_run_dir=reference_run_dir,
    )
    run_def_path = context.run_def_path
    runtime = context.runtime
    run_def_values = context.run_def_values
    cwrr_grid = context.cwrr_grid
    steps_per_stomate = int(round(runtime.dt_stomate / runtime.dt_sechiba))
    if steps_per_stomate <= 0:
        raise ValueError("dt_stomate / dt_sechiba must be positive")
    if int(start_tstep) != 0:
        raise ValueError("current paper-case day scaffold is anchored to start_tstep=0")

    first = paper_1961_driver_timestep_scaffold(
        config_path,
        year=year,
        tstep=int(start_tstep),
        fixed_format_trace_dir=fixed_format_trace_dir,
        static_trace_fields=static_trace_fields,
        used_run_def_path=run_def_path,
        root=root,
        prepared_context=context,
    )
    preview_steps = [first]
    completed_payloads = list(_paper_day_completed_entry_payloads(first))
    previous_state = driver_previous_step_state_from_timestep_scaffold(
        first,
        ok_laidev=context.ok_laidev,
    )
    if first.first_step_restart_state is not None:
        restart_stomate = first.first_step_restart_state.stomate._asdict()
        carried = previous_state.fields_by_component.get("slowproc_stomate_previous_step_state", {})
        missing_restart_carry = {
            name: value
            for name, value in restart_stomate.items()
            if name in (*STOMATE_HALF_HOUR_CARRY_FIELDS, "fixed_cryoturbation_depth") and name not in carried
        }
        if missing_restart_carry:
            previous_state = _paper_previous_state_with_stomate_updates(
                previous_state,
                missing_restart_carry,
            )
    # STOMATE is not advanced by the SECHIBA-only first step. Keep this
    # boundary so OK_LEAK can replay its 48 source-order calls after the
    # complete day's SECHIBA payloads are available.
    stomate_half_hour_initial_state = previous_state
    stopped_at_tstep: int | None = None
    state_gaps: tuple[DriverDayStateGap, ...] = ()

    if not completed_payloads:
        stopped_at_tstep = int(start_tstep)
        state_gaps = _paper_day_gap_after_scaffold(first)
    else:
        day_hydrol_static_template = None
        if context is not None:
            first_hydrol_state, _ = _hydrol_state_namespace_from_previous_state(previous_state)
            day_nvm = int(np.asarray(previous_state.fields_by_component["slowproc_stomate_previous_step_state"]["lai"]).shape[1])
            day_hydrol_static_template = hydrol_static_precall_template_from_slowproc(
                slowproc_restart=SimpleNamespace(**previous_state.fields_by_component["slowproc_stomate_previous_step_state"]),
                pref_soil_veg=first.step.run_scalars.pref_soil_veg,
                nstm=first.step.run_scalars.nstm,
                ext_coeff_vegetfrac=_select_run_def_pft_vector(
                    run_def_values,
                    "EXT_COEFF_VEGETFRAC",
                    first.step.run_scalars,
                    nvm=day_nvm,
                ),
                hydrol_njsc=getattr(first_hydrol_state, "njsc", None),
                refSOC_1d=getattr(first_hydrol_state, "refSOC_1d", None),
                use_refSOC_hydrol=parse_run_def_bool(run_def_values.get("use_refSOC_hydrol", "FALSE")),
                throughfall_by_pft=_select_run_def_pft_vector(
                    run_def_values,
                    "PERCENT_THROUGHFALL_PFT",
                    first.step.run_scalars,
                    nvm=day_nvm,
                ),
                humcste=_select_run_def_pft_vector(
                    run_def_values,
                    "HYDROL_HUMCSTE",
                    first.step.run_scalars,
                    nvm=day_nvm,
                ),
                zz_mm=context.cwrr_grid.znh * 1000.0,
                peat_hydro=PAPER_1961_HYDROL_SOIL_PEAT_HYDRO,
                ok_dgvm=parse_run_def_bool(run_def_values["STOMATE_OK_DGVM"]),
                dt_days=runtime.dt_sechiba / 86400.0,
            )
        for next_step in range(int(start_tstep) + 1, steps_per_stomate):
            step_scaffold = paper_1961_next_step_hydrol_precall_scaffold(
                config_path,
                year=year,
                tstep=next_step,
                fixed_format_trace_dir=fixed_format_trace_dir,
                static_trace_fields=static_trace_fields,
                used_run_def_path=run_def_path,
                previous_state=previous_state,
                prepared_context=context,
                hydrol_static_template=day_hydrol_static_template,
                module_jit=module_jit,
                diffuco_local_jit=diffuco_local_jit,
            )
            entry_payload = _paper_next_step_entry_payload(
                step_scaffold,
                run_def_values=run_def_values,
                runtime=runtime,
                znt=context.cwrr_grid.znt,
                zlt=context.cwrr_grid.zlt,
                prepared_context=context,
            )
            if entry_payload is None:
                stopped_at_tstep = next_step
                state_gaps = _filter_day_state_gaps(DAY1_PREVIOUS_STEP_STATE_GAPS, previous_state)
                break
            completed_payloads.append(entry_payload)
            previous_state = driver_previous_step_state_from_next_step_hydrol_scaffold(
                step_scaffold,
                ok_laidev=context.ok_laidev,
            )
        else:
            stopped_at_tstep = None
            state_gaps = _filter_day_state_gaps(DAY1_PREVIOUS_STEP_STATE_GAPS, previous_state)

    do_slow_flags = tuple(_paper_first_step_do_slow(tstep, runtime) for tstep in range(int(start_tstep), steps_per_stomate))
    daily_fold = None
    daily_carbon_boundary = None
    stomate_bundles = None
    stomate_daily_carbon = None
    ok_leak_boundary = None
    ok_leak_gaps: tuple[DriverDayStateGap, ...] = ()
    stomate_ok_leak = None
    stomate_outputs = None
    daily_reset_fields = None
    slowproc_surface_update = None
    first_day_stomate_state = None
    first_day_end_state = None
    if first.first_step_restart_state is not None:
        daily_fold = _paper_day_daily_process_from_completed_entries(
            config_path=config_path,
            run_def_path=run_def_path,
            restart_state=first.first_step_restart_state,
            runtime=runtime,
            tstep=steps_per_stomate - 1,
            entry_payloads=completed_payloads,
            retain_step_results=True,
            use_maintenance_jit=maintenance_local_jit,
            single_pass_daily_fold=single_pass_daily_fold,
            run_def_values=run_def_values,
        )
        if daily_fold is not None:
            stomate_bundle_source, stomate_bundles = _paper_first_step_stomate_restart_input_bundles(
                config_path=config_path,
                run_def_path=run_def_path,
                restart_state=first.first_step_restart_state,
                runtime=runtime,
                tstep=steps_per_stomate - 1,
                daily_fields=daily_fold.daily_fields,
                run_def_values=run_def_values,
            )
            if (
                stomate_bundles is not None
                and first.stomate_pre_step is not None
                and len(daily_fold.maintenance.step_results) == steps_per_stomate
            ):
                half_hour_ok_leak, half_hour_updates = _paper_half_hour_ok_leak_fold_from_entries(
                    entry_payloads=tuple(completed_payloads),
                    maintenance_step_results=daily_fold.maintenance.step_results,
                    initial_state=stomate_half_hour_initial_state,
                    pre_step_boundary=first.stomate_pre_step.pre_step.boundary_inputs,
                    runtime=runtime,
                    run_def_values=run_def_values,
                    rprof=stomate_bundles.maintenance_inputs["rprof"],
                    pref_soil_veg=first.step.run_scalars.pref_soil_veg,
                    is_tree=first.step.run_scalars.is_tree,
                    is_peat=first.step.run_scalars.is_peat,
                    dayno=1,
                )
                state_after_ok_leak = _paper_previous_state_with_stomate_updates(
                    stomate_half_hour_initial_state,
                    half_hour_updates,
                )
                updated_restart_state = replace(
                    first.first_step_restart_state,
                    stomate=first.first_step_restart_state.stomate._replace(**half_hour_updates),
                )
                stomate_bundle_source, stomate_bundles = _paper_first_step_stomate_restart_input_bundles(
                    config_path=config_path,
                    run_def_path=run_def_path,
                    restart_state=updated_restart_state,
                    runtime=runtime,
                    tstep=steps_per_stomate - 1,
                    daily_fields=daily_fold.daily_fields,
                    run_def_values=run_def_values,
                )
                stomate_ok_leak = ExplicitOkLeakFromPostNppResult(
                    prep=SimpleNamespace(provenance=("stomate.f90 lines 3288-3489",)),
                    ok_leak=half_hour_ok_leak,
                    requires_trace=(),
                )
            else:
                state_after_ok_leak = previous_state
            daily_carbon_boundary = _paper_day_daily_carbon_boundary_from_fold(
                fold=daily_fold,
                bundles=stomate_bundles,
            )
            stomate_daily_carbon = _paper_day_stomate_daily_carbon_from_bundles(
                stomate_bundles,
                run_def_values=run_def_values,
                run_scalars=first.step.run_scalars,
                use_static_jit=use_static_jit_daily_carbon,
            )
            ok_leak_boundary = first.stomate_pre_step.pre_step.boundary_inputs if first.stomate_pre_step is not None else None
            if stomate_ok_leak is None:
                ok_leak_boundary, ok_leak_gaps = _paper_day_ok_leak_boundary(
                    first=first,
                    previous_state=previous_state,
                    runtime=runtime,
                    run_def_values=run_def_values,
                    daily_carbon=stomate_daily_carbon,
                    bundles=stomate_bundles,
                    final_entry_payload=completed_payloads[-1] if completed_payloads else None,
                )
                if ok_leak_boundary is not None and not ok_leak_gaps:
                    stomate_ok_leak = _paper_day_ok_leak_from_boundary(
                        daily_carbon=stomate_daily_carbon,
                        boundary=ok_leak_boundary,
                        runtime=runtime,
                    )
            if ok_leak_boundary is not None and not ok_leak_gaps and stomate_ok_leak is not None:
                stomate_outputs = _paper_day_outputs_from_ok_leak(
                    daily_fold=daily_fold,
                    daily_carbon=stomate_daily_carbon,
                    ok_leak=stomate_ok_leak,
                    boundary=ok_leak_boundary,
                )
                daily_reset_fields = _paper_day_daily_reset_fields(
                    daily_fold=daily_fold,
                    do_slow=do_slow_flags[-1],
                )
                slowproc_surface_update = _paper_day_slowproc_surface_update(
                    first=first,
                    run_def_values=run_def_values,
                    daily_carbon=stomate_daily_carbon,
                )
                season_memory_fields = _paper_day_season_fields_from_bundle_source(
                    stomate_bundle_source,
                    season_state=first.first_step_restart_state.stomate_readstart.season_state,
                    prior_fields=previous_state.fields_by_component["slowproc_stomate_previous_step_state"],
                    dt_days=runtime.dt_days,
                    tsurf_daily=daily_fold.daily_fields["tsurf_daily"],
                    firstcall=True,
                    latitude=first.payload.lalo[:, 0],
                    julian_diff=1.0,
                )
                first_day_stomate_state = _paper_day_first_stomate_state_packet(
                    daily_carbon=stomate_daily_carbon,
                    ok_leak=stomate_ok_leak,
                    outputs=stomate_outputs,
                    daily_reset_fields=daily_reset_fields,
                    ok_leak_boundary=ok_leak_boundary,
                    season_memory_fields=season_memory_fields,
                    slowproc_surface_update=slowproc_surface_update,
                    previous_state=state_after_ok_leak,
                    dt_days=runtime.dt_days,
                )
                first_day_end_state = _paper_day_end_state_packet(
                    previous_state=previous_state,
                    stomate_state=first_day_stomate_state,
                )

    return DriverDayScaffold(
        year=year,
        start_tstep=int(start_tstep),
        steps_per_stomate=steps_per_stomate,
        initial_state_mode=first.initial_state_mode,
        preview_steps=tuple(preview_steps),
        do_slow_flags=do_slow_flags,
        completed_entry_payloads=tuple(completed_payloads),
        previous_step_state=previous_state,
        stopped_at_tstep=stopped_at_tstep,
        state_gaps=state_gaps,
        daily_process_ready=daily_fold is not None and not state_gaps,
        daily_process_fold=daily_fold,
        daily_carbon_boundary=daily_carbon_boundary,
        stomate_restart_input_bundles=stomate_bundles,
        stomate_daily_carbon=stomate_daily_carbon,
        ok_leak_boundary_inputs=ok_leak_boundary,
        ok_leak_boundary_gaps=ok_leak_gaps,
        stomate_ok_leak=stomate_ok_leak,
        stomate_outputs=stomate_outputs,
        daily_reset_fields=daily_reset_fields,
        first_day_stomate_state=first_day_stomate_state,
        first_day_end_state=first_day_end_state,
    )


def paper_1961_driver_later_day_scaffold(
    config_path: str | Path,
    *,
    previous_state: DriverPreviousStepStatePacket,
    year: int = 1961,
    start_tstep: int = 48,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    used_run_def_path: str | Path | None = None,
    reference_run_dir: str | Path | None = None,
    prepared_context: Paper1961PreparedDriverContext | None = None,
    module_jit: bool = True,
    diffuco_local_jit: bool = True,
    retain_stomate_step_results: bool = True,
    maintenance_local_jit: bool = False,
    single_pass_daily_fold: bool = False,
    runtime_entry_payloads: bool = False,
) -> DriverLaterDayScaffold:
    """Advance one later STOMATE day from model-produced previous state.

    Fortran provenance follows the same driver/SECHIBA/STOMATE accumulator
    order as ``paper_1961_driver_day_scaffold``. Unlike the first-day helper,
    this starts from ``previous_state`` and therefore refuses to alias the
    original restart as later prognostic state.
    """

    context = prepared_context or prepare_paper_1961_driver_context(
        config_path,
        used_run_def_path=used_run_def_path,
        reference_run_dir=reference_run_dir,
    )
    run_def_path = context.run_def_path
    runtime = context.runtime
    run_def_values = context.run_def_values
    cwrr_grid = context.cwrr_grid
    steps_per_stomate = int(round(runtime.dt_stomate / runtime.dt_sechiba))
    if steps_per_stomate <= 0:
        raise ValueError("dt_stomate / dt_sechiba must be positive")
    start = int(start_tstep)
    if start <= 0 or start % steps_per_stomate != 0:
        raise ValueError("later-day scaffold start_tstep must be a positive do_slow-aligned day boundary")

    completed_payloads: list[dict[str, object]] = []
    current_state = previous_state
    first_step_scaffold: DriverNextStepHydrolPrecallScaffold | None = None
    stopped_at_tstep: int | None = None
    state_gaps: tuple[DriverDayStateGap, ...] = ()
    day_hydrol_static_template = None
    day_diffuco_static_cache = None
    if context is not None:
        first_hydrol_state, _ = _hydrol_state_namespace_from_previous_state(current_state)
        day_nvm = int(np.asarray(current_state.fields_by_component["slowproc_stomate_previous_step_state"]["lai"]).shape[1])
        run_scalars = context.first_step_bundle.run_scalars if context.first_step_bundle is not None else None
        if run_scalars is None:
            run_scalars = load_paper_1961_step_bundle(
                config_path,
                year=year,
                tstep=start,
                run_def_path=run_def_path,
                reference_run_dir=context.reference_run_dir,
                domain_override=context.domain_override,
            ).run_scalars
        day_hydrol_static_template = hydrol_static_precall_template_from_slowproc(
            slowproc_restart=SimpleNamespace(**current_state.fields_by_component["slowproc_stomate_previous_step_state"]),
            pref_soil_veg=run_scalars.pref_soil_veg,
            nstm=run_scalars.nstm,
            ext_coeff_vegetfrac=_select_run_def_pft_vector(
                run_def_values, "EXT_COEFF_VEGETFRAC", run_scalars, nvm=day_nvm
            ),
            hydrol_njsc=getattr(first_hydrol_state, "njsc", None),
            refSOC_1d=getattr(first_hydrol_state, "refSOC_1d", None),
            use_refSOC_hydrol=parse_run_def_bool(run_def_values.get("use_refSOC_hydrol", "FALSE")),
            throughfall_by_pft=_select_run_def_pft_vector(
                run_def_values, "PERCENT_THROUGHFALL_PFT", run_scalars, nvm=day_nvm
            ),
            humcste=_select_run_def_pft_vector(
                run_def_values, "HYDROL_HUMCSTE", run_scalars, nvm=day_nvm
            ),
            zz_mm=cwrr_grid.znh * 1000.0,
            peat_hydro=PAPER_1961_HYDROL_SOIL_PEAT_HYDRO,
            ok_dgvm=parse_run_def_bool(run_def_values["STOMATE_OK_DGVM"]),
            dt_days=runtime.dt_sechiba / 86400.0,
        )
        if prepared_context is not None and context.first_step_bundle is not None:
            day_diffuco_static_cache = _paper_diffuco_day_static_cache_from_previous_state(
                context,
                current_state,
                salinity=context.first_step_bundle.static_trace_fields.salinity,
                tide_height=context.first_step_bundle.static_trace_fields.tide_height,
            )

    for tstep in range(start, start + steps_per_stomate):
        step_scaffold = paper_1961_next_step_hydrol_precall_scaffold(
            config_path,
            year=year,
            tstep=tstep,
            fixed_format_trace_dir=fixed_format_trace_dir,
            static_trace_fields=static_trace_fields,
            used_run_def_path=run_def_path,
            previous_state=current_state,
            prepared_context=context,
            hydrol_static_template=day_hydrol_static_template,
            diffuco_day_static_cache=day_diffuco_static_cache,
            module_jit=module_jit,
            diffuco_local_jit=diffuco_local_jit,
        )
        if first_step_scaffold is None:
            first_step_scaffold = step_scaffold
        if bool(runtime_entry_payloads) and prepared_context is not None:
            entry_payload = _paper_next_step_runtime_entry_payload(
                step_scaffold,
                prepared_context=context,
            )
        else:
            entry_payload = _paper_next_step_entry_payload(
                step_scaffold,
                run_def_values=run_def_values,
                runtime=runtime,
                znt=cwrr_grid.znt,
                zlt=cwrr_grid.zlt,
                prepared_context=context,
            )
        if entry_payload is None:
            stopped_at_tstep = tstep
            state_gaps = _filter_day_state_gaps(DAY1_PREVIOUS_STEP_STATE_GAPS, current_state)
            break
        completed_payloads.append(entry_payload)
        current_state = driver_previous_step_state_from_next_step_hydrol_scaffold(
            step_scaffold,
            ok_laidev=context.ok_laidev,
        )
    else:
        stopped_at_tstep = None
        state_gaps = _filter_day_state_gaps(DAY1_PREVIOUS_STEP_STATE_GAPS, current_state)

    daily_fold = None
    if first_step_scaffold is not None and stopped_at_tstep is None and not state_gaps:
        daily_fold = _paper_later_day_daily_process_from_completed_entries(
            first_step=first_step_scaffold,
            previous_state=previous_state,
            config_path=config_path,
            run_def_path=run_def_path,
            run_def_values=run_def_values,
            runtime=runtime,
            start_tstep=start,
            entry_payloads=tuple(completed_payloads),
            retain_step_results=retain_stomate_step_results,
            use_maintenance_jit=maintenance_local_jit,
            single_pass_daily_fold=single_pass_daily_fold,
            prepared_context=context,
        )
    stomate_bundles = None
    stomate_bundle_source = None
    stomate_bundle_gaps: tuple[DriverDayStateGap, ...] = ()
    stomate_daily_carbon = None
    ok_leak_boundary = None
    ok_leak_gaps: tuple[DriverDayStateGap, ...] = ()
    stomate_ok_leak = None
    stomate_outputs = None
    daily_reset_fields = None
    slowproc_surface_update = None
    day_stomate_state = None
    day_end_state = None
    if first_step_scaffold is not None and daily_fold is not None and daily_fold.ok:
        restart_state = context.first_step_restart_state or reference_case_first_step_restart_state(
            config_path,
            root=Path(config_path).resolve().parents[1],
            run_dir=context.reference_run_dir,
            run_def_path=run_def_path,
        )
        stomate_bundle_source, stomate_bundles, stomate_bundle_gaps = _paper_later_day_stomate_input_bundles(
            config_path=config_path,
            run_def_path=run_def_path,
            restart_state=restart_state,
            previous_state=previous_state,
            runtime=runtime,
            tstep=start + steps_per_stomate - 1,
            daily_fold=daily_fold,
            firstcall_season=int(start) == 0,
            run_def_values=run_def_values,
        )
        if stomate_bundles is not None and not stomate_bundle_gaps:
            stomate_daily_carbon = _paper_day_stomate_daily_carbon_from_bundles(
                stomate_bundles,
                run_def_values=run_def_values,
                run_scalars=first_step_scaffold.enerbil.step.run_scalars,
            )
            pre_step_boundary = context.first_step_stomate_boundary
            if pre_step_boundary is None:
                first_step_bundle = load_paper_1961_first_step_bundle(
                    config_path,
                    year=year,
                    run_def_path=run_def_path,
                    reference_run_dir=context.reference_run_dir,
                    fixed_format_trace_dir=fixed_format_trace_dir,
                    static_trace_fields=static_trace_fields,
                    domain_override=context.domain_override,
                )
                pre_step_boundary = paper_1961_first_step_stomate_boundary(
                    config_path,
                    bundle=first_step_bundle,
                    used_run_def_path=run_def_path,
                    root=Path(config_path).resolve().parents[1],
                    run_dir=context.reference_run_dir,
                )
            slowproc_state = previous_state.fields_by_component["slowproc_stomate_previous_step_state"]
            ok_leak_boundary, ok_leak_gaps = _paper_ok_leak_boundary_from_sources(
                base=pre_step_boundary.pre_step.boundary_inputs,
                previous_state=current_state,
                runtime=runtime,
                run_def_values=run_def_values,
                daily_carbon=stomate_daily_carbon,
                bundles=stomate_bundles,
                final_entry_payload=completed_payloads[-1] if completed_payloads else None,
                veget_max=slowproc_state["veget_max"],
                pref_soil_veg=first_step_scaffold.enerbil.step.run_scalars.pref_soil_veg,
                is_tree=first_step_scaffold.enerbil.step.run_scalars.is_tree,
                is_peat=first_step_scaffold.enerbil.step.run_scalars.is_peat,
                restart_altmax=slowproc_state["altmax"],
                restart_fixed_cryoturbation_depth=slowproc_state["fixed_cryoturbation_depth"],
                npts=first_step_scaffold.enerbil.payload.kjpindex,
                dayno=int(start // steps_per_stomate) + 1,
            )
            if ok_leak_boundary is not None and not ok_leak_gaps:
                stomate_ok_leak = _paper_day_ok_leak_from_boundary(
                    daily_carbon=stomate_daily_carbon,
                    boundary=ok_leak_boundary,
                    runtime=runtime,
                )
                stomate_outputs = _paper_day_outputs_from_ok_leak(
                    daily_fold=daily_fold,
                    daily_carbon=stomate_daily_carbon,
                    ok_leak=stomate_ok_leak,
                    boundary=ok_leak_boundary,
                )
                daily_reset_fields = _paper_day_daily_reset_fields(
                    daily_fold=daily_fold,
                    do_slow=True,
                )
                slowproc_surface_update = _paper_later_day_slowproc_surface_update(
                    previous_state=previous_state,
                    run_def_values=run_def_values,
                    daily_carbon=stomate_daily_carbon,
                    step=first_step_scaffold,
                )
                season_memory_fields = _paper_day_season_fields_from_bundle_source(
                    stomate_bundle_source,
                    prior_fields=previous_state.fields_by_component["slowproc_stomate_previous_step_state"],
                    dt_days=runtime.dt_days,
                    tsurf_daily=daily_fold.daily_fields["tsurf_daily"],
                    firstcall=False,
                )
                if season_memory_fields is not None and "gdd_init_date" in previous_state.fields_by_component["slowproc_stomate_previous_step_state"]:
                    season_memory_fields["gdd_init_date"] = season_update_gdd_init_date(
                        previous_state.fields_by_component["slowproc_stomate_previous_step_state"]["gdd_init_date"],
                        latitude=first_step_scaffold.enerbil.payload.lalo[:, 0],
                        julian_diff=float(int(start // steps_per_stomate) + 1),
                    )
                day_stomate_state = _paper_day_first_stomate_state_packet(
                    daily_carbon=stomate_daily_carbon,
                    ok_leak=stomate_ok_leak,
                    outputs=stomate_outputs,
                    daily_reset_fields=daily_reset_fields,
                    ok_leak_boundary=ok_leak_boundary,
                    season_memory_fields=season_memory_fields,
                    slowproc_surface_update=slowproc_surface_update,
                    previous_state=previous_state,
                    dt_days=runtime.dt_days,
                )
                day_end_state = _paper_day_end_state_packet(
                    previous_state=current_state,
                    stomate_state=day_stomate_state,
                )

    return DriverLaterDayScaffold(
        year=year,
        start_tstep=start,
        steps_per_stomate=steps_per_stomate,
        initial_state_mode=f"model_produced_from_{REFERENCE_CASE_INITIAL_STATE_MODE}",
        input_previous_state=previous_state,
        completed_entry_payloads=tuple(completed_payloads),
        previous_step_state=current_state,
        stopped_at_tstep=stopped_at_tstep,
        state_gaps=state_gaps,
        daily_process_ready=daily_fold is not None and stopped_at_tstep is None and not state_gaps,
        daily_process_fold=daily_fold,
        stomate_restart_input_bundles=stomate_bundles,
        stomate_bundle_gaps=stomate_bundle_gaps,
        stomate_daily_carbon=stomate_daily_carbon,
        ok_leak_boundary_inputs=ok_leak_boundary,
        ok_leak_boundary_gaps=ok_leak_gaps,
        stomate_ok_leak=stomate_ok_leak,
        stomate_outputs=stomate_outputs,
        daily_reset_fields=daily_reset_fields,
        day_stomate_state=day_stomate_state,
        day_end_state=day_end_state,
    )


def _runtime_missing_from_gaps(
    *gap_groups: tuple[DriverDayStateGap, ...],
    extra: tuple[str, ...] = (),
) -> tuple[str, ...]:
    missing: list[str] = list(extra)
    for gaps in gap_groups:
        for gap in gaps:
            missing.append(gap.component)
            missing.extend(f"{gap.component}:{field}" for field in gap.fields)
    return tuple(dict.fromkeys(missing))


def _paper_1961_later_day_transition_inputs(
    *,
    context: Paper1961PreparedDriverContext,
    current_state: DriverPreviousStepStatePacket,
    year: int,
    start: int,
    steps_per_stomate: int,
    fixed_format_trace_dir: str | Path | None,
    static_trace_fields: StaticTraceFields | None,
    prebuild_day_payloads: bool,
    prebound_hydrol_runtime_static_tables: HydrolRuntimeStaticTables | None = None,
    compiled_forcing_series: DriverCompiledHalfHourForcing | None = None,
    compiled_landpoint_payload: Mapping[str, object] | None = None,
) -> DriverRuntimeDayTransitionInputs:
    """Bind per-day static inputs before the half-hour transition."""

    run_def_values = context.run_def_values
    cwrr_grid = context.cwrr_grid
    first_hydrol_state, _ = _hydrol_state_namespace_from_previous_state(current_state)
    day_nvm = int(
        current_state.fields_by_component["slowproc_stomate_previous_step_state"]["lai"].shape[1]
    )
    can_prebuild_day_steps = (
        compiled_forcing_series is None
        and
        bool(prebuild_day_payloads)
        and fixed_format_trace_dir is None
        and static_trace_fields is None
        and context.first_step_bundle is not None
    )
    half_hour_inputs: tuple[DriverRuntimeHalfHourInput, ...] = ()
    compiled_base_payload_template = None
    if compiled_forcing_series is not None:
        day_start_bundle = _paper_1961_step_bundle_from_context(
            context,
            year=year,
            tstep=48,
        )
        compiled_base_payload_template = build_intersurf_first_step_payload(
            day_start_bundle,
            driver_z0_for_wind=0.1,
            dt_sechiba=context.dt_sechiba,
        )
        if compiled_landpoint_payload is not None:
            compiled_base_payload_template = replace(
                compiled_base_payload_template,
                **dict(compiled_landpoint_payload),
            )
    elif can_prebuild_day_steps:
        day_step_bundles = tuple(
            _paper_1961_step_bundle_from_context(context, year=year, tstep=tstep)
            for tstep in range(int(start), int(start) + int(steps_per_stomate))
        )
        half_hour_inputs = tuple(
            DriverRuntimeHalfHourInput(
                step=step,
                base_payload=build_intersurf_first_step_payload(
                    step,
                    driver_z0_for_wind=0.1,
                    dt_sechiba=context.dt_sechiba,
                ),
            )
            for step in day_step_bundles
        )
        day_start_bundle = half_hour_inputs[0].step
        compiled_base_payload_template = half_hour_inputs[0].base_payload
    else:
        day_start_bundle = _paper_1961_step_bundle_from_context(context, year=year, tstep=start)
    run_scalars = day_start_bundle.run_scalars
    hydrol_static_template = hydrol_static_precall_template_from_slowproc(
        slowproc_restart=SimpleNamespace(**current_state.fields_by_component["slowproc_stomate_previous_step_state"]),
        pref_soil_veg=run_scalars.pref_soil_veg,
        nstm=run_scalars.nstm,
        ext_coeff_vegetfrac=context.ext_coeff_vegetfrac[:day_nvm],
        hydrol_njsc=getattr(first_hydrol_state, "njsc", None),
        refSOC_1d=getattr(first_hydrol_state, "refSOC_1d", None),
        use_refSOC_hydrol=parse_run_def_bool(run_def_values.get("use_refSOC_hydrol", "FALSE")),
        throughfall_by_pft=context.hydrol_throughfall_by_pft[:day_nvm],
        humcste=context.hydrol_humcste[:day_nvm],
        zz_mm=cwrr_grid.znh * 1000.0,
        peat_hydro=PAPER_1961_HYDROL_SOIL_PEAT_HYDRO,
        ok_dgvm=context.stomate_ok_dgvm,
        dt_days=context.runtime.dt_sechiba / 86400.0,
    )
    hydrol_runtime_static_tables = prebound_hydrol_runtime_static_tables
    if hydrol_runtime_static_tables is None:
        hydrol_runtime_static_tables = hydrol_runtime_static_tables_from_payload(
            hydrol_static_template,
            ks=context.hydrol_cwrr_ks,
            peat_hydro=PAPER_1961_HYDROL_SOIL_PEAT_HYDRO,
        )
    diffuco_day_static_cache = None
    if context.first_step_bundle is not None:
        if int(start) == 0:
            day_start_payload = (
                intersurf_payload_with_driver_wind_z0(
                    half_hour_inputs[0].base_payload,
                    forcing=day_start_bundle.forcing,
                    driver_z0_for_wind=_driver_wind_z0_from_previous_state(current_state),
                )
                if half_hour_inputs
                else build_intersurf_first_step_payload(
                    day_start_bundle,
                    driver_z0_for_wind=_driver_wind_z0_from_previous_state(current_state),
                    dt_sechiba=context.dt_sechiba,
                )
            )
            diffuco_day_static_cache = _paper_diffuco_day_static_cache_from_firstcall_inputs(
                context,
                current_state,
                salinity=day_start_payload.salinity,
                tide_height=day_start_payload.tide_height,
            )
        else:
            diffuco_day_static_cache = _paper_diffuco_day_static_cache_from_previous_state(
                context,
                current_state,
                salinity=(
                    compiled_forcing_series.salinity[0]
                    if compiled_forcing_series is not None
                    else context.first_step_bundle.static_trace_fields.salinity
                ),
                tide_height=(
                    compiled_forcing_series.tide_height[0]
                    if compiled_forcing_series is not None
                    else context.first_step_bundle.static_trace_fields.tide_height
                ),
            )

    return DriverRuntimeDayTransitionInputs(
        year=int(year),
        start=int(start),
        steps_per_stomate=int(steps_per_stomate),
        run_def_path=context.run_def_path,
        day_start_bundle=day_start_bundle,
        hydrol_static_template=hydrol_static_template,
        hydrol_runtime_static_tables=hydrol_runtime_static_tables,
        diffuco_day_static_cache=diffuco_day_static_cache,
        half_hour_inputs=half_hour_inputs,
        compiled_forcing_series=compiled_forcing_series,
        compiled_base_payload_template=compiled_base_payload_template,
    )


_COMPILED_STOMATE_ENTRY_FIELDS = (
    "lalo",
    "precip_rain",
    "precip_snow",
    "t2m",
    "t2m_min",
    "t2m_max",
    "wspeed",
    "pb",
    "veget",
    "veget_max",
    "totfrac_nobio",
    "gpp",
    "t2mdiag",
    "evapot_corr",
    "temp_sol",
    "humrel",
    "shumdiag",
    "litterhumdiag",
    "soil_mc",
    "wat_flux",
    "runoff_per_soil",
    "drainage_per_soil",
    "runoff2peat",
    "canopy2ground",
    "precip2ground",
    "precip2canopy",
    "wtp",
    "fwet_new",
    "liqwt_ratio",
    "snow",
    "snowdz",
    "snowrho",
    "stempdiag",
    "tdeep",
    "hsdeep",
    "zz_deep",
    "zz_coef_deep",
    "tmc_topgrass",
    "shumdiag_peat",
    "mc_peat_above",
    "shumdiag_croppeat",
    "mc_croppeat_above",
    "shumdiag_man",
    "mc_man_above",
)
_COMPILED_HALF_HOUR_FORCING_FIELDS = frozenset(DriverCompiledHalfHourForcing._fields)
_COMPILED_DRIVER_METADATA_FIELDS = frozenset({"soilclass_sub_index", "soilclass_sub_area"})
_COMPILED_INTERSURF_PAYLOAD_FIELDS = tuple(
    field.name
    for field in dataclass_fields(IntersurfFirstStepPayload)
    if field.name
    not in {"kjpindex", "nbindex", "missing_fields"}
    | _COMPILED_HALF_HOUR_FORCING_FIELDS
    | _COMPILED_DRIVER_METADATA_FIELDS
)
assert not (_COMPILED_HALF_HOUR_FORCING_FIELDS & set(_COMPILED_INTERSURF_PAYLOAD_FIELDS))
_COMPILED_SECHIBA_SCAN_CACHE: dict[object, object] = {}
_COMPILED_LATER_DAY_BLOCK_CACHE: dict[object, object] = {}


def _compiled_half_hour_forcing(item: DriverRuntimeHalfHourInput) -> DriverCompiledHalfHourForcing:
    payload = item.base_payload
    forcing = item.step.forcing
    return DriverCompiledHalfHourForcing(
        zlev=payload.zlev,
        zlevuv=payload.zlevuv,
        u=forcing.u,
        v=forcing.v,
        qair=payload.qair,
        temp_air=payload.temp_air,
        pb=payload.pb,
        precip_rain=payload.precip_rain,
        precip_snow=payload.precip_snow,
        lwdown=payload.lwdown,
        swdown=payload.swdown,
        ccanopy=payload.ccanopy,
        salinity=payload.salinity,
        tide_height=payload.tide_height,
    )


def _paper_compiled_forcing_day(
    context: Paper1961PreparedDriverContext,
    *,
    year: int,
    start_tstep: int,
    steps_per_stomate: int,
) -> DriverCompiledHalfHourForcing:
    """Stack one forcing day without constructing state-dependent process inputs."""

    first = context.first_step_bundle
    if first is None:
        raise ValueError("compiled forcing assembly requires a prepared first-step bundle")
    co2_ppm = read_annual_co2(context.config_path, year)
    ccanopy = ccanopy_from_co2(co2_ppm, first.domain.nbindex)
    salinity = first.static_trace_fields.salinity
    tide_height = first.static_trace_fields.tide_height
    if salinity is None or tide_height is None:
        raise ValueError("compiled forcing assembly requires salinity and tide fields")

    def land_vector(value):
        return np.asarray(value, dtype=np.float64).reshape(-1)

    def forcing_at(tstep: int) -> DriverCompiledHalfHourForcing:
        forcing = read_forcing_model_step_cached(
            context.config_path,
            domain=first.domain,
            year=year,
            model_tstep=tstep,
            split=context.forcing_split,
            nb_spread=context.forcing_nb_spread,
        )
        return DriverCompiledHalfHourForcing(
            zlev=land_vector(forcing.zlev),
            zlevuv=land_vector(forcing.zlevuv),
            u=forcing.u,
            v=forcing.v,
            qair=land_vector(forcing.qair),
            temp_air=land_vector(forcing.temp_air),
            pb=land_vector(forcing.pb) / 100.0,
            precip_rain=(
                land_vector(forcing.precip_rain) * float(context.dt_sechiba)
            ),
            precip_snow=(
                land_vector(forcing.precip_snow) * float(context.dt_sechiba)
            ),
            lwdown=land_vector(forcing.lwdown),
            swdown=land_vector(forcing.swdown),
            ccanopy=ccanopy,
            salinity=salinity,
            tide_height=tide_height,
        )

    half_hour_forcing = tuple(
        forcing_at(tstep)
        for tstep in range(
            int(start_tstep),
            int(start_tstep) + int(steps_per_stomate),
        )
    )
    return jax.tree_util.tree_map(
        lambda *values: np.stack(tuple(np.asarray(value) for value in values)),
        *half_hour_forcing,
    )


def _compiled_diffuco_day_inputs(cache: DriverDiffucoDayStaticCache) -> DriverCompiledDiffucoDayInputs:
    payload = cache.slowproc_derivvar_payload
    return DriverCompiledDiffucoDayInputs(
        control_salinity=cache.control_salinity.payload["control_salinity"],
        control_inudate=cache.control_inundation.payload["control_inudate"],
        qsintmax=payload["qsintmax"],
        assim_param=payload["assim_param"],
        height=payload["height"],
        temp_growth=payload["temp_growth"],
    )


def _compiled_landpoint_payload(payload: IntersurfFirstStepPayload) -> dict[str, object]:
    return {name: getattr(payload, name) for name in _COMPILED_INTERSURF_PAYLOAD_FIELDS}


def _compiled_hydrol_table_arrays(tables: HydrolRuntimeStaticTables) -> DriverCompiledHydrolTableArrays:
    mineral = tables.mineral
    return DriverCompiledHydrolTableArrays(
        mc_lin=mineral.mc_lin,
        k_lin=mineral.k_lin,
        a_lin=mineral.a_lin,
        b_lin=mineral.b_lin,
        d_lin=mineral.d_lin,
        kfact=mineral.kfact,
        nfact=mineral.nfact,
        afact=mineral.afact,
        nvan_mod=mineral.nvan_mod,
        avan_mod=mineral.avan_mod,
    )


def _compiled_diffuco_parameter_values(
    context: Paper1961PreparedDriverContext,
) -> DriverCompiledDiffucoParameterValues:
    parameters = context.diffuco_pft14_static_kwargs["trans_co2_pft_params"]
    return DriverCompiledDiffucoParameterValues(
        trans_co2={name: value for name, value in parameters.items() if name != "ok_laidev"},
        rstruct_const=context.diffuco_pft14_static_kwargs["rstruct_const"],
    )


def _compiled_diffuco_static_kwargs(
    context: Paper1961PreparedDriverContext,
    values: DriverCompiledDiffucoParameterValues,
) -> dict[str, object]:
    """Rebuild DIFFUCO kwargs with dynamic continuous point parameters."""

    kwargs = {
        name: value
        for name, value in context.diffuco_pft14_static_kwargs.items()
        if name not in {"trans_co2_pft_params", "rstruct_const"}
    }
    kwargs["trans_co2_pft_params"] = dict(values.trans_co2)
    kwargs["rstruct_const"] = values.rstruct_const
    return kwargs


def _compiled_stomate_parameter_values(
    context: Paper1961PreparedDriverContext,
) -> DriverCompiledStomateParameterValues:
    """Return the run.def-controlled continuous STOMATE vectors for a point."""

    parameters = context.stomate_static.parameters
    return DriverCompiledStomateParameterValues(
        alloc_min=parameters.alloc_min,
        residence_time=parameters.residence_time,
        vcmax25=parameters.vcmax25,
        maint_resp_slope=parameters.maint_resp_slope,
    )


def _stomate_parameter_override_mapping(values: DriverCompiledStomateParameterValues) -> dict[str, object]:
    return values._asdict()


def _paper_compiled_sechiba_scan(
    *,
    config_path: str | Path,
    context: Paper1961PreparedDriverContext,
    base_payload_template: IntersurfFirstStepPayload,
    run_scalars: RunScalars,
    runtime_spec,
    hydrol_static_template: Mapping[str, object],
    hydrol_runtime_static_tables,
):
    hydrol_dynamic_names = tuple(
        name
        for name in hydrol_static_template
        if not name.startswith("_") and name not in {"pref_soil_veg", "peat_tiles"}
    )
    diffuco_parameters = context.diffuco_pft14_static_kwargs["trans_co2_pft_params"]
    diffuco_dynamic_names = tuple(name for name in diffuco_parameters if name != "ok_laidev")
    cache_key = (
        runtime_spec.components,
        runtime_spec.field_names_by_component,
        int(context.nvm),
        int(run_scalars.nstm),
        float(context.dt_sechiba),
        bool(context.ok_explicitsnow),
        bool(context.ok_freeze_cwrr),
        bool(context.ok_thermodynamical_freezing),
        _effective_thermosoil_wetdiaglong(context.run_def_values),
        bool(context.hydrol_soil_peat_hydro),
        bool(context.peat_hydro),
        bool(context.hydrol_ok_pc),
        bool(context.hydrol_ok_leak),
        bool(context.hydrol_tides),
        tuple(bool(value) for value in context.ok_laidev),
        tuple(int(value) for value in np.asarray(run_scalars.pref_soil_veg).tolist()),
        tuple(bool(value) for value in np.asarray(run_scalars.is_tree).tolist()),
        int(hydrol_runtime_static_tables.mineral.imin),
        int(hydrol_runtime_static_tables.mineral.imax),
        hydrol_dynamic_names,
        diffuco_dynamic_names,
    )
    cached = _COMPILED_SECHIBA_SCAN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    static_hydrol_template = {
        name: value
        for name, value in hydrol_static_template.items()
        if name in {"pref_soil_veg", "peat_tiles"}
    }
    static_diffuco_kwargs = {
        name: value
        for name, value in context.diffuco_pft14_static_kwargs.items()
        if name not in {"trans_co2_pft_params", "rstruct_const"}
    }
    static_diffuco_parameters = {
        name: value
        for name, value in diffuco_parameters.items()
        if name not in diffuco_dynamic_names
    }
    mineral_imin = int(hydrol_runtime_static_tables.mineral.imin)
    mineral_imax = int(hydrol_runtime_static_tables.mineral.imax)

    def transition(
        values_by_component,
        forcing,
        landpoint_payload,
        hydrol_values,
        hydrol_table_arrays,
        diffuco_day_inputs,
        diffuco_parameter_values,
    ):
        dynamic_state = DriverFastStateBundle(
            tstep=47,
            values_by_component=values_by_component,
            spec=runtime_spec,
        )
        payload = replace(
            base_payload_template,
            **landpoint_payload,
            zlev=forcing.zlev,
            zlevuv=forcing.zlevuv,
            u=forcing.u,
            v=forcing.v,
            qair=forcing.qair,
            temp_air=forcing.temp_air,
            pb=forcing.pb,
            precip_rain=forcing.precip_rain,
            precip_snow=forcing.precip_snow,
            lwdown=forcing.lwdown,
            swdown=forcing.swdown,
            ccanopy=forcing.ccanopy,
            salinity=forcing.salinity,
            tide_height=forcing.tide_height,
        )
        runtime_forcing = SimpleNamespace(
            zlev=forcing.zlev,
            zlevuv=forcing.zlevuv,
            u=forcing.u,
            v=forcing.v,
        )
        hydrol_template = dict(static_hydrol_template)
        hydrol_template.update(zip(hydrol_dynamic_names, hydrol_values, strict=True))
        mineral_tables = MineralCWRRTables(
            **hydrol_table_arrays._asdict(),
            imin=mineral_imin,
            imax=mineral_imax,
        )
        runtime_static_tables = HydrolRuntimeStaticTables(mineral=mineral_tables, peat=None)
        trans_co2_parameters = dict(static_diffuco_parameters)
        trans_co2_parameters.update(diffuco_parameter_values.trans_co2)
        diffuco_static_kwargs = dict(static_diffuco_kwargs)
        diffuco_static_kwargs["trans_co2_pft_params"] = trans_co2_parameters
        diffuco_static_kwargs["rstruct_const"] = diffuco_parameter_values.rstruct_const
        result = _paper_1961_next_step_runtime_result_compact(
            config_path,
            previous_state=dynamic_state,
            year=1961,
            tstep=48,
            fixed_format_trace_dir=None,
            static_trace_fields=None,
            used_run_def_path=context.run_def_path,
            prepared_context=context,
            hydrol_static_template=hydrol_template,
            hydrol_runtime_static_tables=runtime_static_tables,
            diffuco_day_static_cache=None,
            compiled_diffuco_day_inputs=diffuco_day_inputs,
            compiled_diffuco_static_kwargs=diffuco_static_kwargs,
            module_jit=True,
            diffuco_local_jit=True,
            prebuilt_base_payload=payload,
            runtime_run_scalars=run_scalars,
            runtime_forcing=runtime_forcing,
        )
        entry_values = tuple(result.entry_payload[name] for name in _COMPILED_STOMATE_ENTRY_FIELDS)
        return result.next_state.values_by_component, entry_values

    def scan_transition(
        initial_values,
        forcing_series,
        landpoint_payload,
        hydrol_values,
        hydrol_table_arrays,
        diffuco_day_inputs,
        diffuco_parameter_values,
    ):
        def body(carry, forcing):
            return transition(
                carry,
                forcing,
                landpoint_payload,
                hydrol_values,
                hydrol_table_arrays,
                diffuco_day_inputs,
                diffuco_parameter_values,
            )

        return jax.lax.scan(body, initial_values, forcing_series)

    compiled = jax.jit(scan_transition)
    if len(_COMPILED_SECHIBA_SCAN_CACHE) >= 8:
        _COMPILED_SECHIBA_SCAN_CACHE.pop(next(iter(_COMPILED_SECHIBA_SCAN_CACHE)))
    _COMPILED_SECHIBA_SCAN_CACHE[cache_key] = compiled
    return compiled


def _paper_1961_later_day_half_hour_transition(
    config_path: str | Path,
    *,
    context: Paper1961PreparedDriverContext,
    initial_state: DriverPreviousStepStatePacket,
    day_inputs: DriverRuntimeDayTransitionInputs,
    fixed_format_trace_dir: str | Path | None,
    static_trace_fields: StaticTraceFields | None,
    module_jit: bool = True,
    diffuco_local_jit: bool = True,
    use_fast_state_loop: bool = False,
    use_compiled_sechiba_day: bool = False,
    materialize_compiled_entries: bool = True,
    compiled_diffuco_parameter_values: DriverCompiledDiffucoParameterValues | None = None,
) -> DriverRuntimeDayStepTransition:
    """Advance the 48 half-hour SECHIBA/STOMATE-entry steps for one day.

    This is the concrete day-level transition boundary for the compact runtime:
    per-day static caches are bound once by the caller, while the loop only
    advances dynamic previous-step state in Fortran driver order.
    """

    completed_payloads: list[dict[str, object]] = []
    current_state: object = initial_state
    first_step_metadata: DriverRuntimeStepMetadata | None = None
    stopped_at_tstep: int | None = None
    state_gaps: tuple[DriverDayStateGap, ...] = ()
    current_loop_state = (
        fast_state_from_previous_packet(initial_state)
        if bool(use_fast_state_loop)
        else initial_state
    )
    can_compile_day = (
        bool(use_compiled_sechiba_day)
        and bool(module_jit)
        and bool(diffuco_local_jit)
        and fixed_format_trace_dir is None
        and static_trace_fields is None
        and (
            day_inputs.compiled_forcing_series is not None
            or len(day_inputs.half_hour_inputs) == day_inputs.steps_per_stomate
        )
        and day_inputs.steps_per_stomate == 48
        and day_inputs.diffuco_day_static_cache is not None
        and day_inputs.hydrol_static_template is not None
        and day_inputs.hydrol_runtime_static_tables is not None
        and not bool(context.hydrol_soil_peat_hydro)
        and not bool(use_fast_state_loop)
    )
    if can_compile_day:
        first_tstep = day_inputs.start
        first_compiled_forcing = (
            None
            if day_inputs.compiled_forcing_series is None
            else jax.tree_util.tree_map(
                lambda values: values[0],
                day_inputs.compiled_forcing_series,
            )
        )
        base_payload_template = (
            day_inputs.compiled_base_payload_template
            if day_inputs.compiled_base_payload_template is not None
            else day_inputs.half_hour_inputs[0].base_payload
        )
        first_compiled_payload = (
            None
            if first_compiled_forcing is None
            else replace(
                base_payload_template,
                zlev=first_compiled_forcing.zlev,
                zlevuv=first_compiled_forcing.zlevuv,
                u=first_compiled_forcing.u,
                v=first_compiled_forcing.v,
                qair=first_compiled_forcing.qair,
                temp_air=first_compiled_forcing.temp_air,
                pb=first_compiled_forcing.pb,
                precip_rain=first_compiled_forcing.precip_rain,
                precip_snow=first_compiled_forcing.precip_snow,
                lwdown=first_compiled_forcing.lwdown,
                swdown=first_compiled_forcing.swdown,
                ccanopy=first_compiled_forcing.ccanopy,
                salinity=first_compiled_forcing.salinity,
                tide_height=first_compiled_forcing.tide_height,
            )
        )
        first_step = _paper_1961_next_step_runtime_result_compact(
            config_path,
            year=day_inputs.year,
            tstep=first_tstep,
            fixed_format_trace_dir=None,
            static_trace_fields=None,
            used_run_def_path=day_inputs.run_def_path,
            previous_state=initial_state,
            prepared_context=context,
            hydrol_static_template=day_inputs.hydrol_static_template,
            hydrol_runtime_static_tables=day_inputs.hydrol_runtime_static_tables,
            diffuco_day_static_cache=day_inputs.diffuco_day_static_cache,
            prebuilt_input=(
                None
                if first_compiled_forcing is not None
                else day_inputs.half_hour_inputs[0]
            ),
            prebuilt_base_payload=first_compiled_payload,
            runtime_run_scalars=(
                day_inputs.day_start_bundle.run_scalars
                if first_compiled_forcing is not None
                else None
            ),
            runtime_forcing=(
                SimpleNamespace(
                    zlev=first_compiled_forcing.zlev,
                    zlevuv=first_compiled_forcing.zlevuv,
                    u=first_compiled_forcing.u,
                    v=first_compiled_forcing.v,
                )
                if first_compiled_forcing is not None
                else None
            ),
            compiled_diffuco_static_kwargs=(
                None
                if compiled_diffuco_parameter_values is None
                else _compiled_diffuco_static_kwargs(
                    context, compiled_diffuco_parameter_values
                )
            ),
            module_jit=True,
            diffuco_local_jit=True,
        )
        if first_step.ok:
            runtime_state = fast_state_from_previous_packet(first_step.next_state)
            forcing_series = (
                jax.tree_util.tree_map(
                    lambda values: values[1:],
                    day_inputs.compiled_forcing_series,
                )
                if day_inputs.compiled_forcing_series is not None
                else jax.tree_util.tree_map(
                    lambda *values: np.stack(tuple(np.asarray(value) for value in values)),
                    *tuple(
                        _compiled_half_hour_forcing(item)
                        for item in day_inputs.half_hour_inputs[1:]
                    ),
                )
            )
            hydrol_dynamic_names = tuple(
                name
                for name in day_inputs.hydrol_static_template
                if not name.startswith("_") and name not in {"pref_soil_veg", "peat_tiles"}
            )
            hydrol_values = tuple(
                day_inputs.hydrol_static_template[name]
                for name in hydrol_dynamic_names
            )
            landpoint_payload = _compiled_landpoint_payload(base_payload_template)
            hydrol_table_arrays = _compiled_hydrol_table_arrays(day_inputs.hydrol_runtime_static_tables)
            diffuco_day_inputs = _compiled_diffuco_day_inputs(day_inputs.diffuco_day_static_cache)
            diffuco_parameter_values = (
                _compiled_diffuco_parameter_values(context)
                if compiled_diffuco_parameter_values is None
                else compiled_diffuco_parameter_values
            )
            compiled_scan = _paper_compiled_sechiba_scan(
                config_path=config_path,
                context=context,
                base_payload_template=base_payload_template,
                run_scalars=day_inputs.day_start_bundle.run_scalars,
                runtime_spec=runtime_state.spec,
                hydrol_static_template=day_inputs.hydrol_static_template,
                hydrol_runtime_static_tables=day_inputs.hydrol_runtime_static_tables,
            )
            final_values, stacked_entries = compiled_scan(
                runtime_state.values_by_component,
                forcing_series,
                landpoint_payload,
                hydrol_values,
                hydrol_table_arrays,
                diffuco_day_inputs,
                diffuco_parameter_values,
            )
            compiled_entry_stacks = {
                name: jnp.concatenate(
                    (
                        jnp.asarray(first_step.entry_payload[name])[None, ...],
                        values,
                    ),
                    axis=0,
                )
                for name, values in zip(
                    _COMPILED_STOMATE_ENTRY_FIELDS,
                    stacked_entries,
                    strict=True,
                )
            }
            if materialize_compiled_entries:
                stacked_entries = jax.tree_util.tree_map(np.asarray, stacked_entries)
            completed_payloads = [first_step.entry_payload]
            for offset in range(day_inputs.steps_per_stomate - 1):
                values = tuple(value[offset] for value in stacked_entries)
                payload = {
                    "kjit": first_tstep + offset + 2,
                    **dict(zip(_COMPILED_STOMATE_ENTRY_FIELDS, values, strict=True)),
                }
                completed_payloads.append(payload)
            final_state = previous_packet_from_fast_state(
                DriverFastStateBundle(
                    tstep=day_inputs.start + day_inputs.steps_per_stomate - 1,
                    values_by_component=final_values,
                    spec=runtime_state.spec,
                )
            )
            return DriverRuntimeDayStepTransition(
                completed_entry_payloads=tuple(completed_payloads),
                current_state=final_state,
                first_step_metadata=first_step.metadata,
                stopped_at_tstep=None,
                state_gaps=(),
                compiled_entry_stacks=compiled_entry_stacks,
            )
    for tstep in range(day_inputs.start, day_inputs.start + day_inputs.steps_per_stomate):
        current_state = current_loop_state
        step_offset = int(tstep) - day_inputs.start
        runtime_step = _paper_1961_next_step_runtime_result_compact(
            config_path,
            year=day_inputs.year,
            tstep=tstep,
            fixed_format_trace_dir=fixed_format_trace_dir,
            static_trace_fields=static_trace_fields,
            used_run_def_path=day_inputs.run_def_path,
            previous_state=current_state,
            prepared_context=context,
            hydrol_static_template=day_inputs.hydrol_static_template,
            hydrol_runtime_static_tables=day_inputs.hydrol_runtime_static_tables,
            diffuco_day_static_cache=day_inputs.diffuco_day_static_cache,
            prebuilt_input=day_inputs.half_hour_inputs[step_offset] if day_inputs.half_hour_inputs else None,
            module_jit=module_jit,
            diffuco_local_jit=diffuco_local_jit,
        )
        if first_step_metadata is None and runtime_step.metadata is not None:
            first_step_metadata = runtime_step.metadata
        if not runtime_step.ok:
            stopped_at_tstep = int(tstep)
            state_gaps = (
                DriverDayStateGap(
                    component=f"tstep_{int(tstep)}_runtime_step",
                    fields=tuple(runtime_step.missing_components),
                    provenance=(
                        "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 839-908",
                        "compact JAX driver runtime step diagnostics",
                    ),
                ),
                *_filter_day_state_gaps(DAY1_PREVIOUS_STEP_STATE_GAPS, current_state),
            )
            break
        completed_payloads.append(runtime_step.entry_payload)
        current_state = runtime_step.next_state
        if bool(use_fast_state_loop):
            current_loop_state = (
                current_state
                if hasattr(current_state, "component_fields")
                else fast_state_from_previous_packet(current_state)
            )
        else:
            current_loop_state = current_state
    else:
        stopped_at_tstep = None
        state_gaps = _filter_day_state_gaps(DAY1_PREVIOUS_STEP_STATE_GAPS, current_state)

    return DriverRuntimeDayStepTransition(
        completed_entry_payloads=tuple(completed_payloads),
        current_state=current_state,
        first_step_metadata=first_step_metadata,
        stopped_at_tstep=stopped_at_tstep,
        state_gaps=state_gaps,
    )


def paper_1961_driver_later_day_runtime_result(
    config_path: str | Path,
    *,
    previous_state: DriverPreviousStepStatePacket,
    day_index: int,
    year: int = 1961,
    start_tstep: int = 48,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    used_run_def_path: str | Path | None = None,
    reference_run_dir: str | Path | None = None,
    prepared_context: Paper1961PreparedDriverContext | None = None,
    module_jit: bool = True,
    diffuco_local_jit: bool = True,
    maintenance_local_jit: bool = False,
    single_pass_daily_fold: bool = False,
    use_compiled_accumulator: bool = False,
    use_numpy_accumulator: bool = False,
    retain_stomate_step_results: bool = False,
    use_fast_state_loop: bool = False,
    use_static_jit_daily_carbon: bool = False,
    prebuild_day_payloads: bool = True,
    use_compiled_sechiba_day: bool = False,
    prebound_hydrol_runtime_static_tables: HydrolRuntimeStaticTables | None = None,
    materialize_compiled_entries: bool = True,
    outer_compiled_daily_carbon_dispatch: Mapping[str, object] | None = None,
    compiled_forcing_series: DriverCompiledHalfHourForcing | None = None,
    model_day_number=None,
    compiled_stomate_parameter_values: DriverCompiledStomateParameterValues | None = None,
    compiled_landpoint_payload: Mapping[str, object] | None = None,
    compiled_stomate_restart_template: StomateRestartEntryState | None = None,
    compiled_stomate_season_template: StomateRestartSeasonState | None = None,
    compiled_diffuco_parameter_values: DriverCompiledDiffucoParameterValues | None = None,
    capture_pre_daily_training_boundary: bool = False,
) -> DriverRuntimeDayResult:
    """Advance one later day and retain only runtime outputs.

    This is the compact counterpart to ``paper_1961_driver_later_day_scaffold``.
    It keeps the same audited process order and source-backed kernels, while
    avoiding retention of completed half-hour payloads and daily debug bundles
    in long validation runs.
    """

    context = prepared_context or prepare_paper_1961_driver_context(
        config_path,
        used_run_def_path=used_run_def_path,
        reference_run_dir=reference_run_dir,
    )
    stomate_parameter_overrides = (
        None
        if compiled_stomate_parameter_values is None
        else _stomate_parameter_override_mapping(compiled_stomate_parameter_values)
    )
    run_def_path = context.run_def_path
    runtime = context.runtime
    run_def_values = context.run_def_values
    cwrr_grid = context.cwrr_grid
    steps_per_stomate = int(round(runtime.dt_stomate / runtime.dt_sechiba))
    if steps_per_stomate <= 0:
        raise ValueError("dt_stomate / dt_sechiba must be positive")
    start = int(start_tstep)
    science_day_number = (
        int(start // steps_per_stomate) + 1
        if model_day_number is None
        else model_day_number
    )
    if start < 0 or start % steps_per_stomate != 0:
        raise ValueError("later-day runtime start_tstep must be a non-negative do_slow-aligned day boundary")
    if start == 0 and previous_state.tstep != -1:
        raise ValueError("start_tstep=0 requires a rebased restart state with tstep=-1")
    if start > 0 and previous_state.tstep != start - 1:
        raise ValueError("later-day runtime previous_state.tstep must be start_tstep - 1")

    current_state = previous_state
    day_transition_inputs = _paper_1961_later_day_transition_inputs(
        context=context,
        current_state=current_state,
        year=year,
        start=start,
        steps_per_stomate=steps_per_stomate,
        fixed_format_trace_dir=fixed_format_trace_dir,
        static_trace_fields=static_trace_fields,
        prebuild_day_payloads=prebuild_day_payloads,
        prebound_hydrol_runtime_static_tables=prebound_hydrol_runtime_static_tables,
        compiled_forcing_series=compiled_forcing_series,
        compiled_landpoint_payload=compiled_landpoint_payload,
    )
    run_scalars = day_transition_inputs.day_start_bundle.run_scalars
    half_hour_transition = _paper_1961_later_day_half_hour_transition(
        config_path,
        context=context,
        initial_state=current_state,
        day_inputs=day_transition_inputs,
        fixed_format_trace_dir=fixed_format_trace_dir,
        static_trace_fields=static_trace_fields,
        module_jit=module_jit,
        diffuco_local_jit=diffuco_local_jit,
        use_fast_state_loop=use_fast_state_loop,
        use_compiled_sechiba_day=use_compiled_sechiba_day,
        materialize_compiled_entries=materialize_compiled_entries,
        compiled_diffuco_parameter_values=compiled_diffuco_parameter_values,
    )
    completed_payloads = half_hour_transition.completed_entry_payloads
    current_state = half_hour_transition.current_state
    first_step_metadata = half_hour_transition.first_step_metadata
    stopped_at_tstep = half_hour_transition.stopped_at_tstep
    state_gaps = half_hour_transition.state_gaps

    daily_fold = None
    daily_fold_prerequisite_gaps = _paper_later_day_daily_fold_prerequisite_gaps(
        previous_state=previous_state,
        entry_payloads=completed_payloads,
        steps_per_stomate=steps_per_stomate,
    )
    if first_step_metadata is not None and stopped_at_tstep is None and not state_gaps:
        daily_fold = _paper_later_day_daily_process_from_completed_entries(
            first_step=None,
            previous_state=previous_state,
            config_path=config_path,
            run_def_path=run_def_path,
            run_def_values=run_def_values,
            runtime=runtime,
            start_tstep=start,
            entry_payloads=completed_payloads,
            retain_step_results=not use_compiled_sechiba_day,
            use_maintenance_jit=maintenance_local_jit,
            single_pass_daily_fold=single_pass_daily_fold,
            use_compiled_accumulator=(use_compiled_accumulator or use_compiled_sechiba_day),
            use_numpy_accumulator=use_numpy_accumulator,
            prepared_context=context,
            stomate_parameter_overrides=stomate_parameter_overrides,
        )
    if first_step_metadata is None or daily_fold is None or not daily_fold.ok:
        missing = _runtime_missing_from_gaps(
            state_gaps,
            extra=(daily_fold_prerequisite_gaps or ("daily_process_fold",))
            if daily_fold is None
            else daily_fold.missing_inputs,
        )
        return DriverRuntimeDayResult(
            year=year,
            day_index=int(day_index),
            start_tstep=start,
            steps_per_stomate=steps_per_stomate,
            daily_modelout=None,
            day_end_state=None,
            stopped_at_tstep=stopped_at_tstep,
            missing_components=missing,
            daily_process_fold=daily_fold,
        )

    restart_state = context.first_step_restart_state or reference_case_first_step_restart_state(
        config_path,
        root=Path(config_path).resolve().parents[1],
        run_dir=context.reference_run_dir,
        run_def_path=run_def_path,
    )
    entry_state, entry_state_gaps = _paper_later_day_stomate_entry_state_from_previous(
        template=(
            restart_state.stomate
            if compiled_stomate_restart_template is None
            else compiled_stomate_restart_template
        ),
        previous_state=previous_state,
    )
    if entry_state is None or entry_state_gaps:
        return DriverRuntimeDayResult(
            year=year,
            day_index=int(day_index),
            start_tstep=start,
            steps_per_stomate=steps_per_stomate,
            daily_modelout=None,
            day_end_state=None,
            stopped_at_tstep=stopped_at_tstep,
            missing_components=_runtime_missing_from_gaps(state_gaps, entry_state_gaps),
            daily_process_fold=daily_fold,
        )
    base_boundary = _paper_ok_leak_pre_step_boundary_from_entry_state(
        config_path=config_path,
        run_def_path=run_def_path,
        run_def_values=run_def_values,
        entry_state=entry_state,
        driver_payload=(
            day_transition_inputs.half_hour_inputs[0].base_payload
            if day_transition_inputs.half_hour_inputs
            else (
                day_transition_inputs.compiled_base_payload_template
                if day_transition_inputs.compiled_base_payload_template is not None
                else build_intersurf_first_step_payload(
                    day_transition_inputs.day_start_bundle,
                    dt_sechiba=context.dt_sechiba,
                )
            )
        ),
        run_scalars=run_scalars,
        diaglev=context.diaglev,
        deep_zlt=context.zlt,
        deep_znt=context.znt,
        kjit=start + 1,
        nflow=run_scalars.nstm,
        final_entry_payload=completed_payloads[-1] if completed_payloads else None,
    )
    if not use_compiled_sechiba_day and len(daily_fold.maintenance.step_results) != steps_per_stomate:
        return DriverRuntimeDayResult(
            year=year,
            day_index=int(day_index),
            start_tstep=start,
            steps_per_stomate=steps_per_stomate,
            daily_modelout=None,
            day_end_state=None,
            stopped_at_tstep=stopped_at_tstep,
            missing_components=_runtime_missing_from_gaps(
                state_gaps,
                extra=("half_hour_ok_leak_maintenance_step_results",),
            ),
            daily_process_fold=daily_fold,
        )
    stomate_static = context.stomate_static or paper_case_stomate_static_input_kwargs(
        config_path,
        used_run_def=run_def_path,
    )
    boundary_template = context.stomate_boundary
    if boundary_template is None or int(np.asarray(boundary_template.kwargs["rprof"]).shape[0]) != first_step_metadata.npts:
        boundary_template = paper_case_stomate_boundary_input_kwargs(
            config_path,
            parameters=stomate_static.parameters,
            npts=first_step_metadata.npts,
            used_run_def=run_def_path,
        )
    maintenance_resp_parts = None
    if use_compiled_sechiba_day:
        slowproc_state = previous_state.fields_by_component["slowproc_stomate_previous_step_state"]
        maintenance_kwargs = dict(
            biomass=slowproc_state["biomass"],
            t2m_longterm=slowproc_state.get("t2m_longterm", completed_payloads[0]["t2m"]),
            z_soil=boundary_template.kwargs["z_soil"],
            rprof=boundary_template.kwargs["rprof"],
            sla_calc=slowproc_state["sla_calc"],
            coeff_maint_zero=stomate_static.parameters.coeff_maint_zero,
            maint_resp_slope=(
                stomate_static.parameters.maint_resp_slope
                if stomate_parameter_overrides is None
                else stomate_parameter_overrides["maint_resp_slope"]
            ),
            ext_coeff=stomate_static.parameters.ext_coeff,
            is_tree=stomate_static.parameters.is_tree,
            dt_sechiba=runtime.dt_sechiba,
        )
        if half_hour_transition.compiled_entry_stacks is None:
            maintenance_resp_parts = stomate_maintenance_respiration_parts_from_entries(
                entry_payloads=completed_payloads,
                **maintenance_kwargs,
            )
        else:
            maintenance_resp_parts = stomate_maintenance_respiration_parts_from_stacks(
                t2m_stack=half_hour_transition.compiled_entry_stacks["t2m"],
                stempdiag_stack=half_hour_transition.compiled_entry_stacks["stempdiag"],
                **maintenance_kwargs,
            )
    ok_leak_fold = _paper_half_hour_ok_leak_fold_from_entries(
        entry_payloads=completed_payloads,
        maintenance_step_results=daily_fold.maintenance.step_results,
        maintenance_resp_parts=maintenance_resp_parts,
        initial_state=previous_state,
        pre_step_boundary=base_boundary,
        runtime=runtime,
        run_def_values=run_def_values,
        rprof=boundary_template.kwargs["rprof"],
        pref_soil_veg=first_step_metadata.pref_soil_veg,
        is_tree=first_step_metadata.is_tree,
        is_peat=first_step_metadata.is_peat,
        dayno=science_day_number,
        use_compiled_ok_leak=use_compiled_sechiba_day,
        compiled_entry_stacks=half_hour_transition.compiled_entry_stacks,
        capture_compiled_driver_steps=capture_pre_daily_training_boundary,
    )
    if capture_pre_daily_training_boundary:
        half_hour_ok_leak, half_hour_updates, ok_leak_driver_steps = ok_leak_fold
    else:
        half_hour_ok_leak, half_hour_updates = ok_leak_fold
        ok_leak_driver_steps = None
    state_after_ok_leak = _paper_previous_state_with_stomate_updates(previous_state, half_hour_updates)
    stomate_bundle_source, stomate_bundles, stomate_bundle_gaps = _paper_later_day_stomate_input_bundles(
        config_path=config_path,
        run_def_path=run_def_path,
        restart_state=restart_state,
        previous_state=state_after_ok_leak,
        runtime=runtime,
        tstep=start + steps_per_stomate - 1,
        daily_fold=daily_fold,
        firstcall_season=int(start) == 0,
        run_def_values=run_def_values,
        model_day_number=science_day_number,
        stomate_parameter_overrides=stomate_parameter_overrides,
        stomate_restart_template=compiled_stomate_restart_template,
        stomate_season_template=compiled_stomate_season_template,
    )
    if stomate_bundles is None or stomate_bundle_gaps:
        return DriverRuntimeDayResult(
            year=year,
            day_index=int(day_index),
            start_tstep=start,
            steps_per_stomate=steps_per_stomate,
            daily_modelout=None,
            day_end_state=None,
            stopped_at_tstep=stopped_at_tstep,
            missing_components=_runtime_missing_from_gaps(state_gaps, stomate_bundle_gaps),
            daily_process_fold=daily_fold,
        )
    stomate_daily_carbon = _paper_day_stomate_daily_carbon_from_bundles(
        stomate_bundles,
        run_def_values=run_def_values,
        run_scalars=context.run_scalars,
        use_static_jit=use_static_jit_daily_carbon,
        outer_compiled_static_dispatch=outer_compiled_daily_carbon_dispatch,
    )
    # ``StomateLpj`` has now installed the following day's daily fluxes. The
    # OK_LEAK result remains the final half-hour state from the current day.
    stomate_ok_leak = ExplicitOkLeakFromPostNppResult(
        prep=SimpleNamespace(provenance=("stomate.f90 lines 3288-3489",)),
        ok_leak=half_hour_ok_leak,
        requires_trace=(),
    )
    stomate_outputs = _paper_day_outputs_from_ok_leak(
        daily_fold=daily_fold,
        daily_carbon=stomate_daily_carbon,
        ok_leak=stomate_ok_leak,
        boundary=base_boundary,
    )
    daily_reset_fields = _paper_day_daily_reset_fields(
        daily_fold=daily_fold,
        do_slow=True,
    )
    slowproc_surface_update = _paper_later_day_slowproc_surface_update_from_runtime(
        previous_state=previous_state,
        run_def_values=run_def_values,
        daily_carbon=stomate_daily_carbon,
        metadata=first_step_metadata,
        run_scalars=context.run_scalars,
    )
    season_memory_fields = _paper_day_season_fields_from_bundle_source(
        stomate_bundle_source,
        prior_fields=previous_state.fields_by_component["slowproc_stomate_previous_step_state"],
        dt_days=runtime.dt_days,
        tsurf_daily=daily_fold.daily_fields["tsurf_daily"],
        firstcall=False,
    )
    if season_memory_fields is not None and "gdd_init_date" in previous_state.fields_by_component["slowproc_stomate_previous_step_state"]:
        season_memory_fields["gdd_init_date"] = season_update_gdd_init_date(
            previous_state.fields_by_component["slowproc_stomate_previous_step_state"]["gdd_init_date"],
            latitude=first_step_metadata.lalo[:, 0],
            julian_diff=science_day_number,
        )
    day_stomate_state = _paper_day_first_stomate_state_packet(
        daily_carbon=stomate_daily_carbon,
        ok_leak=stomate_ok_leak,
        outputs=stomate_outputs,
        daily_reset_fields=daily_reset_fields,
        ok_leak_boundary=base_boundary,
        season_memory_fields=season_memory_fields,
        slowproc_surface_update=slowproc_surface_update,
        previous_state=state_after_ok_leak,
        dt_days=runtime.dt_days,
    )
    day_end_state = _paper_day_end_state_packet(
        previous_state=current_state,
        stomate_state=day_stomate_state,
    )
    modelout = _driver_modelout_from_outputs(
        stomate_outputs,
        day_index=int(day_index),
        start_tstep=start,
        steps_per_stomate=steps_per_stomate,
    )
    missing = ()
    if modelout is None:
        missing = ("modelout",)
    if day_end_state is None:
        missing = (*missing, "day_end_state")
    pre_daily_training_boundary = None
    if capture_pre_daily_training_boundary:
        perma_peat = half_hour_ok_leak.soilcarbon.perma_peat
        if perma_peat is None:
            raise RuntimeError(
                "compiled PFT14 training capture requires the active deepC_peat boundary"
            )
        pre_daily_training_boundary = DriverPreDailyTrainingBoundary(
            half_hour_state_values=fast_state_from_previous_packet(
                half_hour_transition.current_state
            ).values_by_component,
            daily_fields=dict(daily_fold.daily_fields),
            ok_leak_updates=dict(half_hour_updates),
            ok_leak_driver_steps=ok_leak_driver_steps,
            deepc_peat=perma_peat.deepc_peat,
            final_diagnostics={
                "t2mdiag": completed_payloads[-1]["t2mdiag"],
                "temp_sol": completed_payloads[-1]["temp_sol"],
            },
        )
    return DriverRuntimeDayResult(
        year=year,
        day_index=int(day_index),
        start_tstep=start,
        steps_per_stomate=steps_per_stomate,
        daily_modelout=modelout,
        day_end_state=day_end_state,
        stopped_at_tstep=stopped_at_tstep,
        missing_components=tuple(dict.fromkeys(missing)),
        daily_process_fold=daily_fold,
        pre_daily_training_boundary=pre_daily_training_boundary,
    )


def _paper_compiled_later_day_block_executable(
    config_path: str | Path,
    *,
    context: Paper1961PreparedDriverContext,
    initial_state: DriverPreviousStepStatePacket,
    year: int,
    block_forcing: DriverCompiledHalfHourForcing,
    block_day_numbers,
    prebound_hydrol_runtime_static_tables: HydrolRuntimeStaticTables,
    daily_carbon_dispatch: Mapping[str, object],
    stomate_parameter_values: DriverCompiledStomateParameterValues | None = None,
    capture_pre_daily_training_boundaries: bool = False,
    training_output_projector=None,
    training_output_projector_key: str | None = None,
):
    """Compile one reusable block of complete later-day state transitions."""

    if training_output_projector is not None:
        if not capture_pre_daily_training_boundaries:
            raise ValueError("training projection requires boundary capture")
        if not training_output_projector_key:
            raise ValueError("training projection requires a stable cache key")

    initial = fast_state_from_previous_packet(initial_state)
    if stomate_parameter_values is None:
        stomate_parameter_values = _compiled_stomate_parameter_values(context)
    block_size = int(np.asarray(block_day_numbers).shape[0])
    mineral_imin = int(prebound_hydrol_runtime_static_tables.mineral.imin)
    mineral_imax = int(prebound_hydrol_runtime_static_tables.mineral.imax)
    stomate_season_provenance = (
        context.first_step_restart_state.stomate_readstart.season_state.provenance
    )
    cache_key = (
        str(context.config_path.resolve()),
        block_size,
        mineral_imin,
        mineral_imax,
        initial.spec.components,
        initial.spec.field_names_by_component,
        bool(capture_pre_daily_training_boundaries),
        training_output_projector_key,
    )
    cached = _COMPILED_LATER_DAY_BLOCK_CACHE.get(cache_key)
    if cached is not None:
        return cached, initial.spec

    def transition(
        values_by_component,
        forcing_days,
        science_day_numbers,
        stomate_parameters,
        hydrol_table_arrays,
        landpoint_payload,
        stomate_restart_template,
        stomate_season_values,
        diffuco_parameter_values,
    ):
        def body(current_values, inputs):
            forcing, science_day_number = inputs
            state = DriverFastStateBundle(
                tstep=47,
                values_by_component=current_values,
                spec=initial.spec,
            )
            day = paper_1961_driver_later_day_runtime_result(
                config_path,
                previous_state=state,
                day_index=2,
                # All year-varying driver inputs, including CO2, are carried
                # by ``forcing``. Keep metadata static so one executable can
                # serve every forcing year with the same state layout.
                year=1961,
                start_tstep=48,
                used_run_def_path=context.run_def_path,
                prepared_context=context,
                module_jit=True,
                diffuco_local_jit=True,
                use_static_jit_daily_carbon=False,
                prebuild_day_payloads=True,
                use_compiled_sechiba_day=True,
                prebound_hydrol_runtime_static_tables=HydrolRuntimeStaticTables(
                    mineral=MineralCWRRTables(
                        **hydrol_table_arrays._asdict(),
                        imin=mineral_imin,
                        imax=mineral_imax,
                    ),
                    peat=None,
                ),
                materialize_compiled_entries=False,
                outer_compiled_daily_carbon_dispatch=daily_carbon_dispatch,
                compiled_forcing_series=forcing,
                model_day_number=science_day_number,
                compiled_stomate_parameter_values=stomate_parameters,
                compiled_landpoint_payload=landpoint_payload,
                compiled_stomate_restart_template=stomate_restart_template,
                compiled_stomate_season_template=StomateRestartSeasonState(
                    **stomate_season_values,
                    provenance=stomate_season_provenance,
                ),
                compiled_diffuco_parameter_values=diffuco_parameter_values,
                capture_pre_daily_training_boundary=(
                    capture_pre_daily_training_boundaries
                ),
            )
            packet = day.day_end_state
            next_values = tuple(
                tuple(
                    packet.fields_by_component[component][name]
                    for name in field_names
                )
                for component, field_names in zip(
                    initial.spec.components,
                    initial.spec.field_names_by_component,
                    strict=True,
                )
            )
            outputs = (
                day.daily_modelout.modelout_fields,
                day.daily_modelout.modelout,
            )
            if capture_pre_daily_training_boundaries:
                if training_output_projector is None:
                    outputs = (
                        day.pre_daily_training_boundary,
                        next_values,
                    )
                else:
                    outputs = training_output_projector(
                        current_values,
                        day.pre_daily_training_boundary,
                    )
            return next_values, outputs

        return jax.lax.scan(
            body,
            values_by_component,
            (forcing_days, science_day_numbers),
        )

    executable = jax.jit(transition).lower(
        initial.values_by_component,
        block_forcing,
        block_day_numbers,
        stomate_parameter_values,
        _compiled_hydrol_table_arrays(prebound_hydrol_runtime_static_tables),
        _compiled_landpoint_payload(
            _paper_1961_later_day_transition_inputs(
                context=context,
                current_state=initial_state,
                year=year,
                start=48,
                steps_per_stomate=int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba)),
                fixed_format_trace_dir=None,
                static_trace_fields=None,
                prebuild_day_payloads=True,
            ).compiled_base_payload_template
        ),
        context.first_step_restart_state.stomate,
        {
            name: value
            for name, value in context.first_step_restart_state.stomate_readstart.season_state._asdict().items()
            if name != "provenance"
        },
        _compiled_diffuco_parameter_values(context),
    ).compile()
    if len(_COMPILED_LATER_DAY_BLOCK_CACHE) >= 8:
        _COMPILED_LATER_DAY_BLOCK_CACHE.pop(
            next(iter(_COMPILED_LATER_DAY_BLOCK_CACHE))
        )
    _COMPILED_LATER_DAY_BLOCK_CACHE[cache_key] = executable
    return executable, initial.spec


def _paper_run_compiled_later_day_blocks(
    config_path: str | Path,
    *,
    context: Paper1961PreparedDriverContext,
    previous_state: DriverPreviousStepStatePacket,
    year: int,
    ndays: int,
    steps_per_stomate: int,
    block_size: int,
    daily_modelout: list[DriverDailyModelout],
) -> tuple[DriverPreviousStepStatePacket, int]:
    """Advance all complete later-day blocks and return the tail start day."""

    next_day_index = 2
    if next_day_index + int(block_size) - 1 > int(ndays):
        return previous_state, next_day_index
    first_block_inputs = _paper_1961_later_day_transition_inputs(
        context=context,
        current_state=previous_state,
        year=year,
        start=steps_per_stomate,
        steps_per_stomate=steps_per_stomate,
        fixed_format_trace_dir=None,
        static_trace_fields=None,
        prebuild_day_payloads=True,
    )
    prebound_tables = first_block_inputs.hydrol_runtime_static_tables
    daily_carbon_dispatch = _paper_daily_carbon_static_dispatch(
        context,
        previous_state,
    )
    stomate_parameter_values = _compiled_stomate_parameter_values(context)
    landpoint_payload = _compiled_landpoint_payload(
        first_block_inputs.compiled_base_payload_template
    )
    stomate_restart_template = context.first_step_restart_state.stomate
    stomate_season_values = {
        name: value
        for name, value in context.first_step_restart_state.stomate_readstart.season_state._asdict().items()
        if name != "provenance"
    }
    diffuco_parameter_values = _compiled_diffuco_parameter_values(context)
    executable = None
    runtime_spec = None
    while next_day_index + int(block_size) - 1 <= int(ndays):
        block_day_indices = tuple(
            range(next_day_index, next_day_index + int(block_size))
        )
        forcing_days = tuple(
            _paper_compiled_forcing_day(
                context,
                year=year,
                start_tstep=(day_index - 1) * steps_per_stomate,
                steps_per_stomate=steps_per_stomate,
            )
            for day_index in block_day_indices
        )
        block_forcing = jax.tree_util.tree_map(
            lambda *values: np.stack(values),
            *forcing_days,
        )
        block_day_numbers = np.asarray(block_day_indices, dtype=np.int32)
        if executable is None:
            executable, runtime_spec = _paper_compiled_later_day_block_executable(
                config_path,
                context=context,
                initial_state=previous_state,
                year=year,
                block_forcing=block_forcing,
                block_day_numbers=block_day_numbers,
                prebound_hydrol_runtime_static_tables=prebound_tables,
                daily_carbon_dispatch=daily_carbon_dispatch,
                stomate_parameter_values=stomate_parameter_values,
            )
        initial_values = fast_state_from_previous_packet(
            previous_state
        ).values_by_component
        final_values, stacked_outputs = executable(
            initial_values,
            block_forcing,
            block_day_numbers,
            stomate_parameter_values,
            _compiled_hydrol_table_arrays(prebound_tables),
            landpoint_payload,
            stomate_restart_template,
            stomate_season_values,
            diffuco_parameter_values,
        )
        stacked_fields, stacked_modelout = stacked_outputs
        for offset, day_index in enumerate(block_day_indices):
            daily_modelout.append(
                DriverDailyModelout(
                    day_index=day_index,
                    start_tstep=(day_index - 1) * steps_per_stomate,
                    end_tstep=day_index * steps_per_stomate - 1,
                    modelout_fields=jax.tree_util.tree_map(
                        lambda value: value[offset],
                        stacked_fields,
                    ),
                    modelout=jax.tree_util.tree_map(
                        lambda value: value[offset],
                        stacked_modelout,
                    ),
                )
            )
        previous_state = previous_packet_from_fast_state(
            DriverFastStateBundle(
                tstep=block_day_indices[-1] * steps_per_stomate - 1,
                values_by_component=final_values,
                spec=runtime_spec,
            )
        )
        next_day_index += int(block_size)
    return previous_state, next_day_index


def paper_1961_driver_restart_year_start_day_result(
    config_path: str | Path,
    *,
    previous_year_end_state: DriverPreviousStepStatePacket | None,
    year: int,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    used_run_def_path: str | Path | None = None,
    prepared_context: Paper1961PreparedDriverContext | None = None,
    module_jit: bool = True,
    diffuco_local_jit: bool = True,
    single_pass_daily_fold: bool = False,
    use_numpy_accumulator: bool = False,
    use_static_jit_daily_carbon: bool = False,
    retain_stomate_step_results: bool = False,
    prebuild_day_payloads: bool = True,
    use_compiled_sechiba_day: bool = False,
) -> DriverRuntimeDayResult:
    """Advance day 1 of a new forcing year from a previous year-end state.

    Fortran provenance: `fortran_run_scripts/paper_250919/Job0_bio` lines
    315-322 rolls restart files from the completed year into the next yearly
    executable run. The driver timestep counter restarts at zero while the
    SECHIBA/STOMATE dynamic state is carried forward.
    """

    context = prepared_context or prepare_paper_1961_driver_context(config_path, used_run_def_path=used_run_def_path)
    steps_per_stomate = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    gaps = driver_year_handoff_state_gaps(previous_year_end_state)
    if gaps or previous_year_end_state is None:
        return DriverRuntimeDayResult(
            year=int(year),
            day_index=1,
            start_tstep=0,
            steps_per_stomate=steps_per_stomate,
            daily_modelout=None,
            day_end_state=None,
            stopped_at_tstep=0,
            missing_components=_runtime_missing_from_gaps(gaps, extra=("year_handoff_state",)),
            daily_process_fold=None,
        )
    return paper_1961_driver_later_day_runtime_result(
        config_path,
        previous_state=rebase_driver_state_for_year_start(previous_year_end_state),
        day_index=1,
        year=year,
        start_tstep=0,
        fixed_format_trace_dir=fixed_format_trace_dir,
        static_trace_fields=static_trace_fields,
        used_run_def_path=context.run_def_path,
        prepared_context=context,
        module_jit=module_jit,
        diffuco_local_jit=diffuco_local_jit,
        single_pass_daily_fold=single_pass_daily_fold,
        use_numpy_accumulator=use_numpy_accumulator,
        use_static_jit_daily_carbon=use_static_jit_daily_carbon,
        retain_stomate_step_results=retain_stomate_step_results,
        prebuild_day_payloads=prebuild_day_payloads,
        use_compiled_sechiba_day=use_compiled_sechiba_day,
    )


def paper_1961_driver_restart_year_multiday_modelout_lite_run(
    config_path: str | Path,
    *,
    previous_year_end_state: DriverPreviousStepStatePacket | None,
    ndays: int,
    year: int,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    used_run_def_path: str | Path | None = None,
    reference_run_dir: str | Path | None = None,
    module_jit: bool = True,
    diffuco_local_jit: bool = True,
    single_pass_daily_fold: bool = False,
    use_numpy_accumulator: bool = False,
    use_static_jit_daily_carbon: bool = False,
    prebuild_day_payloads: bool = True,
    use_compiled_sechiba_day: bool = False,
    compiled_day_block_size: int = 0,
) -> DriverLiteMultiDayModeloutRun:
    """Run a new forcing year from a previous year-end dynamic state."""

    if int(ndays) < 0:
        raise ValueError("ndays must be non-negative")
    if int(compiled_day_block_size) < 0:
        raise ValueError("compiled_day_block_size must be non-negative")
    if int(compiled_day_block_size) == 1:
        raise ValueError("compiled_day_block_size must be zero or at least two")
    if int(compiled_day_block_size) > 0 and not (
        bool(use_compiled_sechiba_day)
        and bool(module_jit)
        and bool(diffuco_local_jit)
        and fixed_format_trace_dir is None
        and static_trace_fields is None
    ):
        raise ValueError(
            "compiled day blocks require compiled later days, JIT, and no trace overrides"
        )
    context = prepare_paper_1961_driver_context(
        config_path,
        used_run_def_path=used_run_def_path,
        reference_run_dir=reference_run_dir,
    )
    steps_per_stomate = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    initial_state_mode = "restart_from_previous_year_state"
    if int(ndays) == 0:
        return DriverLiteMultiDayModeloutRun(
            year=int(year),
            requested_days=0,
            steps_per_stomate=steps_per_stomate,
            initial_state_mode=initial_state_mode,
            daily_modelout=(),
            stopped_day_index=None,
            _missing_components=(),
            last_day_end_state=previous_year_end_state,
        )

    daily_modelout: list[DriverDailyModelout] = []
    stopped_day_index: int | None = None
    missing_components: tuple[str, ...] = ()

    first = paper_1961_driver_restart_year_start_day_result(
        config_path,
        previous_year_end_state=previous_year_end_state,
        year=year,
        fixed_format_trace_dir=fixed_format_trace_dir,
        static_trace_fields=static_trace_fields,
        used_run_def_path=context.run_def_path,
        prepared_context=context,
        module_jit=module_jit,
        diffuco_local_jit=diffuco_local_jit,
        single_pass_daily_fold=single_pass_daily_fold,
        use_numpy_accumulator=use_numpy_accumulator,
        use_static_jit_daily_carbon=use_static_jit_daily_carbon,
        prebuild_day_payloads=prebuild_day_payloads,
        use_compiled_sechiba_day=use_compiled_sechiba_day,
        retain_stomate_step_results=False,
    )
    previous_state = first.day_end_state
    if not first.ready_for_day_end_state:
        stopped_day_index = 1
        missing_components = first.missing_components
    else:
        daily_modelout.append(first.daily_modelout)

    next_day_index = 2
    if (
        stopped_day_index is None
        and int(compiled_day_block_size) > 0
        and previous_state is not None
        and 1 + int(compiled_day_block_size) <= int(ndays)
    ):
        previous_state, next_day_index = _paper_run_compiled_later_day_blocks(
            config_path,
            context=context,
            previous_state=previous_state,
            year=year,
            ndays=int(ndays),
            steps_per_stomate=steps_per_stomate,
            block_size=int(compiled_day_block_size),
            daily_modelout=daily_modelout,
        )

    if stopped_day_index is None:
        for day_index in range(next_day_index, int(ndays) + 1):
            day = paper_1961_driver_later_day_runtime_result(
                config_path,
                previous_state=previous_state,
                day_index=day_index,
                year=year,
                start_tstep=(day_index - 1) * steps_per_stomate,
                fixed_format_trace_dir=fixed_format_trace_dir,
                static_trace_fields=static_trace_fields,
                used_run_def_path=context.run_def_path,
                prepared_context=context,
                module_jit=module_jit,
                diffuco_local_jit=diffuco_local_jit,
                single_pass_daily_fold=single_pass_daily_fold,
                use_numpy_accumulator=use_numpy_accumulator,
                use_static_jit_daily_carbon=use_static_jit_daily_carbon,
                prebuild_day_payloads=prebuild_day_payloads,
                use_compiled_sechiba_day=use_compiled_sechiba_day,
                retain_stomate_step_results=False,
            )
            if not day.ready_for_day_end_state:
                stopped_day_index = day_index
                missing_components = day.missing_components
                break
            daily_modelout.append(day.daily_modelout)
            previous_state = day.day_end_state

    return DriverLiteMultiDayModeloutRun(
        year=int(year),
        requested_days=int(ndays),
        steps_per_stomate=steps_per_stomate,
        initial_state_mode=initial_state_mode,
        daily_modelout=tuple(daily_modelout),
        stopped_day_index=stopped_day_index,
        _missing_components=tuple(
            dict.fromkeys(
                f"day_{stopped_day_index}:{item}" for item in missing_components
            )
        )
        if stopped_day_index is not None
        else (),
        last_day_end_state=previous_state,
    )


def paper_1961_driver_cold_start_multiday_modelout_lite_run(
    config_path: str | Path,
    *,
    ndays: int,
    year: int = 1961,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    used_run_def_path: str | Path | None = None,
    reference_run_dir: str | Path | None = None,
    module_jit: bool = True,
    diffuco_local_jit: bool = True,
    single_pass_daily_fold: bool = False,
    use_numpy_accumulator: bool = False,
    use_static_jit_daily_carbon: bool = False,
    prebuild_day_payloads: bool = True,
    use_compiled_sechiba_day: bool = False,
    compiled_day_block_size: int = 0,
) -> DriverLiteMultiDayModeloutRun:
    """Run a compact no-restart paper-case chain through daily modelout.

    Fortran provenance follows the no-restart initialization path in
    ``sechiba_initialize`` and then the same driver/SECHIBA/STOMATE day order
    as ``paper_1961_driver_multiday_modelout_lite_run``. This runner keeps
    cold-start state sources separate from reference restart sources.
    """

    if int(ndays) < 0:
        raise ValueError("ndays must be non-negative")
    if int(compiled_day_block_size) < 0:
        raise ValueError("compiled_day_block_size must be non-negative")
    if int(compiled_day_block_size) == 1:
        raise ValueError("compiled_day_block_size must be zero or at least two")
    if int(compiled_day_block_size) > 0 and not (
        bool(use_compiled_sechiba_day)
        and bool(module_jit)
        and bool(diffuco_local_jit)
        and fixed_format_trace_dir is None
        and static_trace_fields is None
    ):
        raise ValueError(
            "compiled day blocks require compiled later days, JIT, and no trace overrides"
        )
    context = prepare_paper_1961_driver_context(
        config_path,
        used_run_def_path=used_run_def_path,
        reference_run_dir=reference_run_dir,
    )
    runtime = context.runtime
    steps_per_stomate = int(round(runtime.dt_stomate / runtime.dt_sechiba))
    if int(ndays) == 0:
        return DriverLiteMultiDayModeloutRun(
            year=year,
            requested_days=0,
            steps_per_stomate=steps_per_stomate,
            initial_state_mode=COLD_START_INITIAL_STATE_MODE,
            daily_modelout=(),
            stopped_day_index=None,
            _missing_components=(),
            last_day_end_state=None,
        )

    daily_modelout: list[DriverDailyModelout] = []
    stopped_day_index: int | None = None
    missing_components: tuple[str, ...] = ()

    first = paper_1961_driver_cold_start_day_scaffold(
        config_path,
        year=year,
        fixed_format_trace_dir=fixed_format_trace_dir,
        static_trace_fields=static_trace_fields,
        used_run_def_path=context.run_def_path,
        prepared_context=context,
    )
    previous_state = first.first_day_end_state
    first_output = _driver_modelout_from_outputs(
        first.stomate_outputs,
        day_index=1,
        start_tstep=0,
        steps_per_stomate=steps_per_stomate,
    )
    if first_output is None or previous_state is None or not first.ready_for_first_day_end_state:
        stopped_day_index = 1
        missing_components = first.missing_components
    else:
        daily_modelout.append(first_output)

    next_day_index = 2
    if (
        stopped_day_index is None
        and int(compiled_day_block_size) > 0
        and previous_state is not None
        and 1 + int(compiled_day_block_size) <= int(ndays)
    ):
        previous_state, next_day_index = _paper_run_compiled_later_day_blocks(
            config_path,
            context=context,
            previous_state=previous_state,
            year=year,
            ndays=int(ndays),
            steps_per_stomate=steps_per_stomate,
            block_size=int(compiled_day_block_size),
            daily_modelout=daily_modelout,
        )

    if stopped_day_index is None:
        for day_index in range(next_day_index, int(ndays) + 1):
            day = paper_1961_driver_later_day_runtime_result(
                config_path,
                previous_state=previous_state,
                day_index=day_index,
                year=year,
                start_tstep=(day_index - 1) * steps_per_stomate,
                fixed_format_trace_dir=fixed_format_trace_dir,
                static_trace_fields=static_trace_fields,
                used_run_def_path=context.run_def_path,
                prepared_context=context,
                module_jit=module_jit,
                diffuco_local_jit=diffuco_local_jit,
                single_pass_daily_fold=single_pass_daily_fold,
                use_numpy_accumulator=use_numpy_accumulator,
                use_static_jit_daily_carbon=use_static_jit_daily_carbon,
                prebuild_day_payloads=prebuild_day_payloads,
                use_compiled_sechiba_day=use_compiled_sechiba_day,
                retain_stomate_step_results=False,
            )
            if not day.ready_for_day_end_state:
                stopped_day_index = day_index
                missing_components = day.missing_components
                break
            daily_modelout.append(day.daily_modelout)
            previous_state = day.day_end_state

    return DriverLiteMultiDayModeloutRun(
        year=year,
        requested_days=int(ndays),
        steps_per_stomate=steps_per_stomate,
        initial_state_mode=COLD_START_INITIAL_STATE_MODE,
        daily_modelout=tuple(daily_modelout),
        stopped_day_index=stopped_day_index,
        _missing_components=tuple(
            dict.fromkeys(
                f"day_{stopped_day_index}:{item}" for item in missing_components
            )
        )
        if stopped_day_index is not None
        else (),
        last_day_end_state=previous_state,
    )


def paper_1961_driver_cold_start_runtime_day_result(
    config_path: str | Path,
    *,
    day_index: int,
    year: int = 1961,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    used_run_def_path: str | Path | None = None,
    module_jit: bool = True,
    diffuco_local_jit: bool = True,
    single_pass_daily_fold: bool = False,
    retain_stomate_step_results: bool = True,
) -> DriverColdStartDayScaffold | DriverRuntimeDayResult:
    """Return one cold-start day object with daily-fold diagnostics retained."""

    if int(day_index) < 1:
        raise ValueError("day_index must be positive")
    context = prepare_paper_1961_driver_context(config_path, used_run_def_path=used_run_def_path)
    if int(day_index) == 1:
        return paper_1961_driver_cold_start_day_scaffold(
            config_path,
            year=year,
            fixed_format_trace_dir=fixed_format_trace_dir,
            static_trace_fields=static_trace_fields,
            used_run_def_path=context.run_def_path,
            prepared_context=context,
            module_jit=module_jit,
            diffuco_local_jit=diffuco_local_jit,
        )
    runtime = context.runtime
    steps_per_stomate = int(round(runtime.dt_stomate / runtime.dt_sechiba))
    first = paper_1961_driver_cold_start_day_scaffold(
        config_path,
        year=year,
        fixed_format_trace_dir=fixed_format_trace_dir,
        static_trace_fields=static_trace_fields,
        used_run_def_path=context.run_def_path,
        prepared_context=context,
        module_jit=module_jit,
        diffuco_local_jit=diffuco_local_jit,
    )
    previous_state = first.first_day_end_state
    if previous_state is None or not first.ready_for_first_day_end_state:
        return DriverRuntimeDayResult(
            year=year,
            day_index=int(day_index),
            start_tstep=0,
            steps_per_stomate=steps_per_stomate,
            daily_modelout=None,
            day_end_state=None,
            stopped_at_tstep=0,
            missing_components=first.missing_components,
        )
    day: DriverRuntimeDayResult | None = None
    for current_day in range(2, int(day_index) + 1):
        day = paper_1961_driver_later_day_runtime_result(
            config_path,
            previous_state=previous_state,
            day_index=current_day,
            year=year,
            start_tstep=(current_day - 1) * steps_per_stomate,
            fixed_format_trace_dir=fixed_format_trace_dir,
            static_trace_fields=static_trace_fields,
            used_run_def_path=context.run_def_path,
            prepared_context=context,
            module_jit=module_jit,
            diffuco_local_jit=diffuco_local_jit,
            single_pass_daily_fold=single_pass_daily_fold,
            retain_stomate_step_results=retain_stomate_step_results,
        )
        if not day.ready_for_day_end_state:
            return day
        previous_state = day.day_end_state
    return day


def paper_1961_driver_cold_start_day_scaffold(
    config_path: str | Path,
    *,
    year: int = 1961,
    max_steps: int | None = None,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    used_run_def_path: str | Path | None = None,
    reference_run_dir: str | Path | None = None,
    prepared_context: Paper1961PreparedDriverContext | None = None,
    module_jit: bool = True,
    diffuco_local_jit: bool = True,
) -> DriverColdStartDayScaffold:
    """Advance the first paper-case day from no-restart initialization."""

    context = prepared_context or prepare_paper_1961_driver_context(
        config_path,
        used_run_def_path=used_run_def_path,
        reference_run_dir=reference_run_dir,
    )
    run_def_path = context.run_def_path
    runtime = context.runtime
    steps_per_stomate = int(round(runtime.dt_stomate / runtime.dt_sechiba))
    if steps_per_stomate <= 0:
        raise ValueError("dt_stomate / dt_sechiba must be positive")
    stop_step = steps_per_stomate if max_steps is None else min(int(max_steps), steps_per_stomate)
    if stop_step < 1:
        raise ValueError("max_steps must be positive")
    run_def_values = context.run_def_values

    first = paper_1961_driver_cold_start_first_step_coverage(
        config_path,
        year=year,
        fixed_format_trace_dir=fixed_format_trace_dir,
        static_trace_fields=static_trace_fields,
        used_run_def_path=run_def_path,
        prepared_context=context,
    )
    completed_payloads: list[dict[str, object]] = []
    previous_state = first.initialized_payloads.get("cold_start_first_step_previous_state")
    stopped_at_tstep: int | None = None
    state_gaps: tuple[DriverDayStateGap, ...] = ()
    if previous_state is None:
        stopped_at_tstep = 0
        state_gaps = (
            DriverDayStateGap(
                component="cold_start_first_step_previous_state",
                fields=first.missing_components,
                provenance=first.provenance,
            ),
        )
    else:
        completed_payloads.append(dict(first.initialized_payloads["cold_start_first_step_entry_payload"]))
        cold_entry_state = first.initialized_payloads.get("stomate_cold_start_entry_state")
        if cold_entry_state is not None:
            carried = previous_state.fields_by_component.get("slowproc_stomate_previous_step_state", {})
            missing_cold_carry = {
                name: value
                for name, value in cold_entry_state._asdict().items()
                if name in (*STOMATE_HALF_HOUR_CARRY_FIELDS, "fixed_cryoturbation_depth") and name not in carried
            }
            if missing_cold_carry:
                previous_state = _paper_previous_state_with_stomate_updates(
                    previous_state,
                    missing_cold_carry,
                )
        stomate_half_hour_initial_state = previous_state
        for next_step in range(1, stop_step):
            step_scaffold = paper_1961_next_step_hydrol_precall_scaffold(
                config_path,
                year=year,
                tstep=next_step,
                fixed_format_trace_dir=fixed_format_trace_dir,
                static_trace_fields=static_trace_fields,
                used_run_def_path=run_def_path,
                previous_state=previous_state,
                prepared_context=context,
                module_jit=module_jit,
                diffuco_local_jit=diffuco_local_jit,
            )
            entry_payload = _paper_next_step_entry_payload(
                step_scaffold,
                run_def_values=run_def_values,
                runtime=runtime,
                znt=context.cwrr_grid.znt,
                zlt=context.cwrr_grid.zlt,
            )
            if entry_payload is None:
                stopped_at_tstep = next_step
                state_gaps = _filter_day_state_gaps(DAY1_PREVIOUS_STEP_STATE_GAPS, previous_state)
                break
            completed_payloads.append(entry_payload)
            previous_state = driver_previous_step_state_from_next_step_hydrol_scaffold(
                step_scaffold,
                ok_laidev=context.ok_laidev,
            )
        else:
            stopped_at_tstep = None
            state_gaps = _filter_day_state_gaps(DAY1_PREVIOUS_STEP_STATE_GAPS, previous_state)

    daily_fold = _paper_cold_start_day_daily_process_from_completed_entries(
        first_step_coverage=first,
        config_path=config_path,
        run_def_path=run_def_path,
        runtime=runtime,
        entry_payloads=tuple(completed_payloads),
    )
    if max_steps is not None and stop_step < steps_per_stomate:
        daily_fold = None
    if daily_fold is None and max_steps is None and stopped_at_tstep is None and not state_gaps:
        state_gaps = (
            DriverDayStateGap(
                component="cold_start_stomate_daily_process_fold",
                fields=("daily_accumulators", "maintenance_inputs"),
                provenance=("fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 3187-3267",),
            ),
        )
    stomate_bundles = None
    stomate_bundle_source = None
    stomate_bundle_gaps: tuple[DriverDayStateGap, ...] = ()
    stomate_daily_carbon = None
    ok_leak_boundary = None
    ok_leak_gaps: tuple[DriverDayStateGap, ...] = ()
    stomate_ok_leak = None
    stomate_outputs = None
    daily_reset_fields = None
    slowproc_surface_update = None
    first_day_stomate_state = None
    first_day_end_state = None
    if max_steps is None and daily_fold is not None and stopped_at_tstep is None and not state_gaps:
        stomate_bundle_source, stomate_bundles, stomate_bundle_gaps = _paper_cold_start_stomate_input_bundles(
            config_path=config_path,
            run_def_path=run_def_path,
            first_step_coverage=first,
            runtime=runtime,
            tstep=steps_per_stomate - 1,
            daily_fold=daily_fold,
            run_def_values=run_def_values,
        )
        cold_payloads = first.initialized_payloads
        cold_entry_state = cold_payloads.get("stomate_cold_start_entry_state")
        first_payload = cold_payloads.get("first_step_payload")
        run_scalars = cold_payloads.get("run_scalars") or context.run_scalars
        if (
            stomate_bundles is not None
            and cold_entry_state is not None
            and first_payload is not None
            and len(daily_fold.maintenance.step_results) == steps_per_stomate
        ):
            base_boundary = _paper_ok_leak_pre_step_boundary_from_entry_state(
                config_path=config_path,
                run_def_path=run_def_path,
                run_def_values=run_def_values,
                entry_state=cold_entry_state,
                driver_payload=first_payload,
                run_scalars=run_scalars,
                diaglev=cold_payloads["diaglev"],
                deep_zlt=cold_payloads["deep_zlt"],
                deep_znt=cold_payloads["deep_znt"],
                kjit=1,
                nflow=run_scalars.nstm,
                final_entry_payload=completed_payloads[-1] if completed_payloads else None,
            )
            cold_vegetation = cold_payloads["slowproc_cold_start_vegetation"].vegetation
            half_hour_ok_leak, half_hour_updates = _paper_half_hour_ok_leak_fold_from_entries(
                entry_payloads=tuple(completed_payloads),
                maintenance_step_results=daily_fold.maintenance.step_results,
                initial_state=stomate_half_hour_initial_state,
                pre_step_boundary=base_boundary,
                runtime=runtime,
                run_def_values=run_def_values,
                rprof=stomate_bundles.maintenance_inputs["rprof"],
                pref_soil_veg=run_scalars.pref_soil_veg,
                is_tree=run_scalars.is_tree,
                is_peat=run_scalars.is_peat,
                dayno=1,
            )
            updated_entry_state = cold_entry_state._replace(**half_hour_updates)
            initial_slowproc = stomate_half_hour_initial_state.fields_by_component["slowproc_stomate_previous_step_state"]
            stomate_bundles = stomate_restart_input_bundles(
                state=updated_entry_state,
                veget_max=cold_vegetation.veget_max,
                lai=initial_slowproc["lai"],
                cn_ind=_stomate_lpj_initial_cn_ind(updated_entry_state),
                dt_days=runtime.dt_days,
                dt_sechiba=runtime.dt_sechiba,
                dt_stomate=runtime.dt_stomate,
                do_slow=True,
                stomate_restart_none=True,
                daily_fields=daily_fold.daily_fields,
                **stomate_bundle_source.kwargs,
            )
            stomate_daily_carbon = _paper_day_stomate_daily_carbon_from_bundles(
                stomate_bundles,
                run_def_values=run_def_values,
                run_scalars=run_scalars,
            )
            ok_leak_boundary = base_boundary
            stomate_ok_leak = ExplicitOkLeakFromPostNppResult(
                prep=SimpleNamespace(provenance=("stomate.f90 lines 3288-3489",)),
                ok_leak=half_hour_ok_leak,
                requires_trace=(),
            )
            previous_state = _paper_previous_state_with_stomate_updates(
                previous_state,
                {
                    **half_hour_updates,
                    "fixed_cryoturbation_depth": stomate_half_hour_initial_state.fields_by_component[
                        "slowproc_stomate_previous_step_state"
                    ]["fixed_cryoturbation_depth"],
                },
            )
        else:
            stomate_daily_carbon = _paper_day_stomate_daily_carbon_from_bundles(
                stomate_bundles,
                run_def_values=run_def_values,
                run_scalars=run_scalars,
            )
            ok_leak_boundary, ok_leak_gaps = _paper_cold_start_ok_leak_boundary(
                first_step_coverage=first,
                previous_state=previous_state,
                config_path=config_path,
                run_def_path=run_def_path,
                runtime=runtime,
                run_def_values=run_def_values,
                daily_carbon=stomate_daily_carbon,
                bundles=stomate_bundles,
                final_entry_payload=completed_payloads[-1] if completed_payloads else None,
            )
        if ok_leak_boundary is not None and not ok_leak_gaps:
            if stomate_ok_leak is None:
                stomate_ok_leak = _paper_day_ok_leak_from_boundary(
                    daily_carbon=stomate_daily_carbon,
                    boundary=ok_leak_boundary,
                    runtime=runtime,
                )
            stomate_outputs = _paper_day_outputs_from_ok_leak(
                daily_fold=daily_fold,
                daily_carbon=stomate_daily_carbon,
                ok_leak=stomate_ok_leak,
                boundary=ok_leak_boundary,
            )
            daily_reset_fields = _paper_day_daily_reset_fields(
                daily_fold=daily_fold,
                do_slow=True,
            )
            slowproc_state = previous_state.fields_by_component["slowproc_stomate_previous_step_state"]
            lai = _paper_day_final_lai_from_daily_carbon(stomate_daily_carbon)
            veget_max = (
                stomate_daily_carbon.post_npp.cover.veget_max
                if stomate_daily_carbon.post_npp.cover is not None
                else slowproc_state["veget_max"]
            )
            nvm = int(lai.shape[1])
            slowproc_surface_update = slowproc_surface_update_explicit(
                lai=lai,
                frac_nobio=slowproc_state["frac_nobio"],
                veget_max=veget_max,
                pref_soil_veg=run_scalars.pref_soil_veg,
                ext_coeff_vegetfrac=_select_run_def_pft_vector(
                    run_def_values, "EXT_COEFF_VEGETFRAC", run_scalars, nvm=nvm
                ),
                nstm=context.nflow,
                ok_dgvm=parse_run_def_bool(run_def_values["STOMATE_OK_DGVM"]),
            )
            season_memory_fields = _paper_day_season_fields_from_bundle_source(
                stomate_bundle_source,
                season_state=first.initialized_payloads["stomate_cold_start_season_state"],
                prior_fields=previous_state.fields_by_component["slowproc_stomate_previous_step_state"],
                dt_days=runtime.dt_days,
                tsurf_daily=daily_fold.daily_fields["tsurf_daily"],
                firstcall=True,
                latitude=context.first_step_bundle.domain.lalo[:, 0],
                julian_diff=1.0,
            )
            first_day_stomate_state = _paper_day_first_stomate_state_packet(
                daily_carbon=stomate_daily_carbon,
                ok_leak=stomate_ok_leak,
                outputs=stomate_outputs,
                daily_reset_fields=daily_reset_fields,
                ok_leak_boundary=ok_leak_boundary,
                season_memory_fields=season_memory_fields,
                slowproc_surface_update=slowproc_surface_update,
                previous_state=previous_state,
                dt_days=runtime.dt_days,
            )
            first_day_end_state = _paper_day_end_state_packet(
                previous_state=previous_state,
                stomate_state=first_day_stomate_state,
            )
    return DriverColdStartDayScaffold(
        year=year,
        start_tstep=0,
        steps_per_stomate=steps_per_stomate,
        initial_state_mode=COLD_START_INITIAL_STATE_MODE,
        first_step_coverage=first,
        completed_entry_payloads=tuple(completed_payloads),
        completed_steps=len(completed_payloads),
        previous_step_state=previous_state,
        stopped_at_tstep=stopped_at_tstep,
        state_gaps=state_gaps,
        daily_process_fold=daily_fold,
        daily_process_ready=daily_fold is not None and stopped_at_tstep is None and not state_gaps,
        stomate_restart_input_bundles=stomate_bundles,
        stomate_bundle_gaps=stomate_bundle_gaps,
        stomate_daily_carbon=stomate_daily_carbon,
        ok_leak_boundary_inputs=ok_leak_boundary,
        ok_leak_boundary_gaps=ok_leak_gaps,
        stomate_ok_leak=stomate_ok_leak,
        stomate_outputs=stomate_outputs,
        daily_reset_fields=daily_reset_fields,
        first_day_stomate_state=first_day_stomate_state,
        first_day_end_state=first_day_end_state,
    )


def paper_1961_driver_multiday_scaffold(
    config_path: str | Path,
    *,
    ndays: int,
    year: int = 1961,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    used_run_def_path: str | Path | None = None,
    reference_run_dir: str | Path | None = None,
    root: str | Path | None = None,
    module_jit: bool = True,
    diffuco_local_jit: bool = True,
) -> DriverMultiDayScaffold:
    """Advance consecutive paper-case days until a source-backed gap appears.

    The first day is anchored to the restart-backed driver boundary. Every
    later day starts from the previous day's model-produced day-end state.
    """

    if int(ndays) < 0:
        raise ValueError("ndays must be non-negative")
    context = prepare_paper_1961_driver_context(
        config_path,
        used_run_def_path=used_run_def_path,
        reference_run_dir=reference_run_dir,
    )
    run_def_path = context.run_def_path
    runtime = context.runtime
    steps_per_stomate = int(round(runtime.dt_stomate / runtime.dt_sechiba))
    if int(ndays) == 0:
        return DriverMultiDayScaffold(
            year=year,
            requested_days=0,
            steps_per_stomate=steps_per_stomate,
            initial_state_mode=REFERENCE_CASE_INITIAL_STATE_MODE,
            days=(),
            stopped_day_index=None,
        )

    days: list[DriverDayScaffold | DriverLaterDayScaffold] = []
    first = paper_1961_driver_day_scaffold(
        config_path,
        year=year,
        start_tstep=0,
        fixed_format_trace_dir=fixed_format_trace_dir,
        static_trace_fields=static_trace_fields,
        used_run_def_path=run_def_path,
        root=root,
        prepared_context=context,
        module_jit=module_jit,
        diffuco_local_jit=diffuco_local_jit,
        retain_stomate_step_results=False,
        single_pass_daily_fold=True,
    )
    days.append(first)
    previous_state = first.first_day_end_state
    stopped_day_index = None
    if previous_state is None or not first.ready_for_first_day_end_state:
        stopped_day_index = 1
    else:
        for day_index in range(2, int(ndays) + 1):
            day = paper_1961_driver_later_day_scaffold(
                config_path,
                year=year,
                start_tstep=(day_index - 1) * steps_per_stomate,
                fixed_format_trace_dir=fixed_format_trace_dir,
                static_trace_fields=static_trace_fields,
                used_run_def_path=run_def_path,
                previous_state=previous_state,
                prepared_context=context,
                module_jit=module_jit,
                diffuco_local_jit=diffuco_local_jit,
                retain_stomate_step_results=False,
                single_pass_daily_fold=True,
            )
            days.append(day)
            if not day.ready_for_day_end_state or day.day_end_state is None:
                stopped_day_index = day_index
                break
            previous_state = day.day_end_state

    return DriverMultiDayScaffold(
        year=year,
        requested_days=int(ndays),
        steps_per_stomate=steps_per_stomate,
        initial_state_mode=first.initial_state_mode,
        days=tuple(days),
        stopped_day_index=stopped_day_index,
    )


def paper_1961_driver_multiday_modelout_run(
    config_path: str | Path,
    *,
    ndays: int,
    year: int = 1961,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    used_run_def_path: str | Path | None = None,
    reference_run_dir: str | Path | None = None,
    root: str | Path | None = None,
    module_jit: bool = True,
    diffuco_local_jit: bool = True,
) -> DriverMultiDayModeloutRun:
    """Run the closed paper-case daily chain through STOMATE modelout fields.

    Fortran provenance follows ``paper_1961_driver_multiday_scaffold`` for the
    driver/SECHIBA/STOMATE order, then ``stomate_lpj.f90`` lines 2200-2246 for
    the history fields used by the paper modelout formula. This wrapper does
    not fill missing days or trace-dependent outputs; incomplete days remain
    visible through ``scaffold.missing_components``.
    """

    scaffold = paper_1961_driver_multiday_scaffold(
        config_path,
        ndays=ndays,
        year=year,
        fixed_format_trace_dir=fixed_format_trace_dir,
        static_trace_fields=static_trace_fields,
        used_run_def_path=used_run_def_path,
        reference_run_dir=reference_run_dir,
        root=root,
        module_jit=module_jit,
        diffuco_local_jit=diffuco_local_jit,
    )
    daily_modelout: list[DriverDailyModelout] = []
    for index, day in enumerate(scaffold.days, start=1):
        output = _driver_day_modelout(day, day_index=index)
        if output is None:
            break
        daily_modelout.append(output)
    return DriverMultiDayModeloutRun(
        scaffold=scaffold,
        daily_modelout=tuple(daily_modelout),
    )


def paper_1961_driver_multiday_modelout_lite_run(
    config_path: str | Path,
    *,
    ndays: int,
    year: int = 1961,
    fixed_format_trace_dir: str | Path | None = None,
    static_trace_fields: StaticTraceFields | None = None,
    used_run_def_path: str | Path | None = None,
    reference_run_dir: str | Path | None = None,
    root: str | Path | None = None,
    module_jit: bool = True,
    diffuco_local_jit: bool = True,
    single_pass_daily_fold: bool = False,
    use_compiled_accumulator: bool = False,
    use_numpy_accumulator: bool = False,
    compact_later_days: bool = False,
    use_fast_state_loop: bool = False,
    use_static_jit_daily_carbon: bool = False,
    prebuild_day_payloads: bool = True,
    use_compiled_sechiba_day: bool = False,
    compiled_day_block_size: int = 0,
) -> DriverLiteMultiDayModeloutRun:
    """Run daily modelout without retaining completed day scaffolds.

    This is a runtime wrapper only: first-day and later-day process closure is
    delegated to the same audited scaffold builders as the full multiday path.
    It exists to keep long sequence validation results small while preserving
    the Fortran driver order and source-backed process functions.
    """

    if int(ndays) < 0:
        raise ValueError("ndays must be non-negative")
    if int(compiled_day_block_size) < 0:
        raise ValueError("compiled_day_block_size must be non-negative")
    if int(compiled_day_block_size) == 1:
        raise ValueError("compiled_day_block_size must be zero or at least two")
    if int(compiled_day_block_size) > 0 and not (
        bool(compact_later_days)
        and bool(use_compiled_sechiba_day)
        and bool(module_jit)
        and bool(diffuco_local_jit)
        and fixed_format_trace_dir is None
        and static_trace_fields is None
    ):
        raise ValueError(
            "compiled day blocks require compact compiled later days, JIT, and no trace overrides"
        )
    context = prepare_paper_1961_driver_context(
        config_path,
        used_run_def_path=used_run_def_path,
        reference_run_dir=reference_run_dir,
    )
    run_def_path = context.run_def_path
    runtime = context.runtime
    steps_per_stomate = int(round(runtime.dt_stomate / runtime.dt_sechiba))
    if int(ndays) == 0:
        return DriverLiteMultiDayModeloutRun(
            year=year,
            requested_days=0,
            steps_per_stomate=steps_per_stomate,
            initial_state_mode=REFERENCE_CASE_INITIAL_STATE_MODE,
            daily_modelout=(),
            stopped_day_index=None,
            _missing_components=(),
            last_day_end_state=None,
        )

    daily_modelout: list[DriverDailyModelout] = []
    stopped_day_index: int | None = None
    missing_components: tuple[str, ...] = ()

    first = paper_1961_driver_day_scaffold(
        config_path,
        year=year,
        start_tstep=0,
        fixed_format_trace_dir=fixed_format_trace_dir,
        static_trace_fields=static_trace_fields,
        used_run_def_path=run_def_path,
        root=root,
        prepared_context=context,
        module_jit=module_jit,
        diffuco_local_jit=diffuco_local_jit,
        retain_stomate_step_results=False,
        single_pass_daily_fold=single_pass_daily_fold,
        use_static_jit_daily_carbon=use_static_jit_daily_carbon,
    )
    initial_state_mode = first.initial_state_mode
    previous_state = first.first_day_end_state
    output = _driver_day_modelout(first, day_index=1)
    if output is None or previous_state is None or not first.ready_for_first_day_end_state:
        stopped_day_index = 1
        missing_components = first.missing_components
    else:
        daily_modelout.append(output)

    next_day_index = 2
    if (
        stopped_day_index is None
        and int(compiled_day_block_size) > 0
        and previous_state is not None
        and 1 + int(compiled_day_block_size) <= int(ndays)
    ):
        previous_state, next_day_index = _paper_run_compiled_later_day_blocks(
            config_path,
            context=context,
            previous_state=previous_state,
            year=year,
            ndays=int(ndays),
            steps_per_stomate=steps_per_stomate,
            block_size=int(compiled_day_block_size),
            daily_modelout=daily_modelout,
        )

    if stopped_day_index is None:
        for day_index in range(next_day_index, int(ndays) + 1):
            if bool(compact_later_days):
                day = paper_1961_driver_later_day_runtime_result(
                    config_path,
                    year=year,
                    day_index=day_index,
                    start_tstep=(day_index - 1) * steps_per_stomate,
                    fixed_format_trace_dir=fixed_format_trace_dir,
                    static_trace_fields=static_trace_fields,
                    used_run_def_path=run_def_path,
                    previous_state=previous_state,
                    prepared_context=context,
                    module_jit=module_jit,
                    diffuco_local_jit=diffuco_local_jit,
                    single_pass_daily_fold=single_pass_daily_fold,
                    use_compiled_accumulator=use_compiled_accumulator,
                    use_numpy_accumulator=use_numpy_accumulator,
                    use_fast_state_loop=use_fast_state_loop,
                    use_static_jit_daily_carbon=use_static_jit_daily_carbon,
                    prebuild_day_payloads=prebuild_day_payloads,
                    use_compiled_sechiba_day=use_compiled_sechiba_day,
                )
                if not day.ready_for_day_end_state:
                    stopped_day_index = day_index
                    missing_components = day.missing_components
                    break
                daily_modelout.append(day.daily_modelout)
                previous_state = day.day_end_state
            else:
                day = paper_1961_driver_later_day_scaffold(
                    config_path,
                    year=year,
                    start_tstep=(day_index - 1) * steps_per_stomate,
                    fixed_format_trace_dir=fixed_format_trace_dir,
                    static_trace_fields=static_trace_fields,
                    used_run_def_path=run_def_path,
                    previous_state=previous_state,
                    prepared_context=context,
                    module_jit=module_jit,
                    diffuco_local_jit=diffuco_local_jit,
                    retain_stomate_step_results=False,
                    single_pass_daily_fold=single_pass_daily_fold,
                    runtime_entry_payloads=True,
                )
                output = _driver_day_modelout(day, day_index=day_index)
                if output is None or day.day_end_state is None or not day.ready_for_day_end_state:
                    stopped_day_index = day_index
                    missing_components = day.missing_components
                    break
                daily_modelout.append(output)
                previous_state = day.day_end_state

    return DriverLiteMultiDayModeloutRun(
        year=year,
        requested_days=int(ndays),
        steps_per_stomate=steps_per_stomate,
        initial_state_mode=initial_state_mode,
        daily_modelout=tuple(daily_modelout),
        stopped_day_index=stopped_day_index,
        _missing_components=tuple(
            dict.fromkeys(
                f"day_{stopped_day_index}:{item}" for item in missing_components
            )
        )
        if stopped_day_index is not None
        else (),
        last_day_end_state=previous_state,
    )
