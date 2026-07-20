"""Aggregate first-step restart state for strict driver orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from netCDF4 import Dataset

from jax_orchidee.driver.restart import SechibaStaticRestart, read_sechiba_static_restart, reference_restart_path
from jax_orchidee.sechiba.enerbil import (
    DriverAlbedoRestart,
    EnerbilSoilThermalStateRestart,
    read_driver_albedo_restart,
    read_enerbil_soil_thermal_state_restart,
)
from jax_orchidee.sechiba.diffuco import DiffucoFirstStepRestartState, read_diffuco_first_step_restart_state
from jax_orchidee.sechiba.hydrol_reference import HydrolRestartAnchors, read_hydrol_restart_anchors
from jax_orchidee.sechiba.slowproc import SlowprocRestartEntryState, read_slowproc_restart_entry_state
from jax_orchidee.sechiba.thermosoil import ThermosoilRestartState, read_thermosoil_restart_state
from jax_orchidee.sechiba.restart_io import SechibaRestartState, read_sechiba_restart_state
from jax_orchidee.stomate.reference import (
    StomateRestartEntryState,
    find_stomate_reference_files,
    read_stomate_restart_entry_state,
    read_stomate_daily_accumulator_state,
    read_stomate_restart_season_state,
)
from jax_orchidee.stomate.restart_io import (
    StomateReadstartStates,
    read_stomate_readstart_states_from_template,
)


FIRST_STEP_RESTART_STATE_PROVENANCE = (
    "fortran_run_scripts/paper_250919/Job0_bio lines 277-322",
    "reference/case_001_071 run.def lines 264-268",
    "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 1139-1164",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_initialize lines 668-684",
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90 restart reads/writes lines 2720-2963 and 1727-1747",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_init lines 1685-1707",
    "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 919-940, 1201-1232, 1466-1471, and 1512-1563",
)

REFERENCE_CASE_INITIAL_STATE_MODE = "restart_backed_reference_case"
COLD_START_INITIAL_STATE_MODE = "cold_start_no_restart"


@dataclass(frozen=True)
class ReferenceCaseFirstStepRestartState:
    """Exact restart-backed state available before the first reference step."""

    run_dir: Path
    driver_start: Path
    sechiba_start: Path
    stomate_input: Path
    driver_albedo: DriverAlbedoRestart
    sechiba_static: SechibaStaticRestart
    diffuco_precall: DiffucoFirstStepRestartState
    hydrol: HydrolRestartAnchors
    enerbil_thermal: EnerbilSoilThermalStateRestart
    thermosoil: ThermosoilRestartState
    slowproc: SlowprocRestartEntryState
    stomate: StomateRestartEntryState
    stomate_readstart: StomateReadstartStates
    sechiba_restart_state: SechibaRestartState
    provenance: tuple[str, ...] = FIRST_STEP_RESTART_STATE_PROVENANCE


def reference_case_first_step_restart_state(
    config_path: str | Path,
    *,
    root: str | Path | None = None,
    run_dir: str | Path | None = None,
    stomate_filename: str = "stomate_start.nc",
) -> ReferenceCaseFirstStepRestartState:
    """Read exact first-step restart state from the local paper reference.

    The paper driver uses ``driver_start.nc``, ``sechiba_start.nc``, and
    ``stomate_start.nc`` after the first loop year. This helper aggregates
    those restart readers for first-step orchestration coverage only. It does
    not advance prognostic state, substitute restart for later timesteps, or
    synthesize absent restart variables.
    """

    root_path = Path(root) if root is not None else Path(config_path).resolve().parents[1]
    if run_dir is None:
        driver_start = reference_restart_path(config_path, "driver_start.nc")
        sechiba_start = reference_restart_path(config_path, "sechiba_start.nc")
        stomate_files = find_stomate_reference_files(root_path)
        resolved_run_dir = stomate_files.run_dir
        stomate_input = resolved_run_dir / stomate_filename
    else:
        run_dir_path = Path(run_dir)
        driver_start = run_dir_path / "driver_start.nc"
        sechiba_start = run_dir_path / "sechiba_start.nc"
        stomate_input = run_dir_path / stomate_filename
        for path in (driver_start, sechiba_start, stomate_input):
            if not path.exists():
                raise FileNotFoundError(path)
        resolved_run_dir = run_dir_path
    if not stomate_input.exists():
        raise FileNotFoundError(stomate_input)

    stomate = read_stomate_restart_entry_state(stomate_input)
    stomate_daily = read_stomate_daily_accumulator_state(stomate_input)
    stomate_season = read_stomate_restart_season_state(stomate_input)
    with Dataset(stomate_input) as dataset:
        nvert = int(dataset.variables["uo_0"].shape[1])
        months_num = int(dataset.variables["fwet_series"].shape[1])
        nbpools = int(dataset.variables["MatrixV"].shape[1])
        nsnow = int(dataset.variables["O2_snow"].shape[2])
    readstart = read_stomate_readstart_states_from_template(
        stomate_input,
        t2m=stomate_daily.t2m_daily,
        nvm=stomate.age.shape[1],
        nslm=stomate_season.tsoil_month.shape[1],
        ndeep=stomate.carbon_32l.shape[3],
        nsnow=nsnow,
        nvert=nvert,
        months_num=months_num,
        ncarb=stomate.carbon.shape[1],
        nlitt=stomate.litter.shape[1],
        nbpools=nbpools,
    )

    return ReferenceCaseFirstStepRestartState(
        run_dir=resolved_run_dir,
        driver_start=driver_start,
        sechiba_start=sechiba_start,
        stomate_input=stomate_input,
        driver_albedo=read_driver_albedo_restart(driver_start),
        sechiba_static=read_sechiba_static_restart(sechiba_start),
        diffuco_precall=read_diffuco_first_step_restart_state(sechiba_start),
        hydrol=read_hydrol_restart_anchors(sechiba_start),
        enerbil_thermal=read_enerbil_soil_thermal_state_restart(sechiba_start),
        thermosoil=read_thermosoil_restart_state(sechiba_start),
        slowproc=read_slowproc_restart_entry_state(sechiba_start),
        stomate=stomate,
        stomate_readstart=readstart,
        sechiba_restart_state=read_sechiba_restart_state(sechiba_start),
    )
