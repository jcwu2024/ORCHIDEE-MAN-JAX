"""Independent three-file restart bundle for the paper landpoint protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from jax_orchidee.driver.dim2 import Dim2DriverRestart
from jax_orchidee.driver.restart_export import stomate_full_restart_states_from_day_end_packet
from jax_orchidee.driver.restart_file_io import write_dim2_driver_restart
from jax_orchidee.driver.source_clock import (
    PREFIX,
    SourceClockRuntime,
    physical_state_with_source_clock,
    read_source_clock_restart_bundle,
    validate_source_clock_restart_packet,
)
from jax_orchidee.parameters.pft_catalog import PFTRunLayout
from jax_orchidee.sechiba.restart_export import paper_sechiba_restart_state_from_finalize_state
from jax_orchidee.sechiba.restart_io import SechibaRestartState, write_sechiba_restart_state
from jax_orchidee.stomate.restart_io import (
    StomateReadstartStates,
    StomateRestartPhysicalState,
    StomateRestartWriteReport,
    write_stomate_full_writerestart_states,
)


@dataclass(frozen=True)
class PaperRestartBundleState:
    """Complete normalized scientific state for one year handoff."""

    driver: Dim2DriverRestart
    sechiba: SechibaRestartState
    stomate: StomateReadstartStates
    source_clock_runtime: SourceClockRuntime | None = None


@dataclass(frozen=True)
class PaperRestartBundlePhysicalState:
    """Physical coordinate/time state for each independent restart file."""

    driver: StomateRestartPhysicalState
    sechiba: StomateRestartPhysicalState
    stomate: StomateRestartPhysicalState


@dataclass(frozen=True)
class PaperRestartBundleWriteReport:
    """Strict reports for a complete driver/SECHIBA/STOMATE handoff."""

    output_directory: Path
    driver: StomateRestartWriteReport
    sechiba: StomateRestartWriteReport
    stomate: StomateRestartWriteReport

    @property
    def output_paths(self) -> tuple[Path, Path, Path]:
        return (
            self.driver.output_path,
            self.sechiba.output_path,
            self.stomate.output_path,
        )


def paper_restart_bundle_state_from_day_end_packet(
    packet: object,
    *,
    base_stomate: StomateReadstartStates,
    kjit: int,
    use_refsoc_hydrol: bool = False,
) -> PaperRestartBundleState:
    """Convert a complete production day/year-end packet to three restart states.

    The packet must contain the dedicated 93-field ``sechiba_finalize_state``
    and all 13 driver restart fields. Enabling ``use_refsoc_hydrol`` requires
    the additional source-conditional ``refSOC_1d`` field, without changing the
    default 93-field denominator. STOMATE state uses the audited day-end
    merge contract from ``driver.restart_export``.
    """

    components = getattr(packet, "fields_by_component", None)
    if components is None:
        raise TypeError("packet must expose fields_by_component")
    runtime = validate_source_clock_restart_packet(packet, source_istp=kjit)
    driver_fields = components.get("driver_previous_step_state", {})
    driver_names = tuple(Dim2DriverRestart.__dataclass_fields__)
    missing_driver = tuple(name for name in driver_names if name not in driver_fields)
    if missing_driver:
        raise ValueError(f"day-end packet is missing driver restart fields: {missing_driver}")
    finalize = components.get("sechiba_finalize_state", {})
    if bool(use_refsoc_hydrol) != ("refSOC_1d" in finalize):
        raise ValueError(
            "conditional refSOC_1d state must be present exactly when use_refsoc_hydrol is enabled"
        )
    base_count = len(finalize) - int("refSOC_1d" in finalize)
    if base_count != 93:
        raise ValueError(
            f"day-end packet SECHIBA base finalize denominator must be 93, got {base_count}"
        )
    slowproc = components.get("slowproc_stomate_previous_step_state", {})
    stomate, _ = stomate_full_restart_states_from_day_end_packet(
        slowproc, base_states=base_stomate,
    )
    return PaperRestartBundleState(
        driver=Dim2DriverRestart(**{name: driver_fields[name] for name in driver_names}),
        sechiba=paper_sechiba_restart_state_from_finalize_state(
            finalize, kjit=int(kjit), use_refsoc_hydrol=use_refsoc_hydrol,
        ),
        stomate=stomate,
        source_clock_runtime=runtime,
    )


def read_restart_physical_state(path: str | Path) -> StomateRestartPhysicalState:
    """Read the five IOIPSL physical fields without touching science state."""

    with Dataset(path) as dataset:
        fields = {
            name: np.asarray(dataset.variables[name][:])
            for name in ("nav_lon", "nav_lat", "nav_lev", "time", "time_steps")
        }
        attributes = {
            name: dataset.getncattr(name)
            for name in dataset.ncattrs()
        }
        variable_attributes = {
            name: {key: dataset.variables[name].getncattr(key)
                   for key in dataset.variables[name].ncattrs() if key != "_FillValue"}
            for name in fields
        }
    return StomateRestartPhysicalState(
        **fields,
        index_g=(1,),
        global_attributes=attributes,
        variable_attributes=variable_attributes,
    )


def write_paper_restart_start_bundle(
    output_directory: str | Path,
    *,
    state: PaperRestartBundleState,
    physical_state: PaperRestartBundlePhysicalState,
    pft_layout: PFTRunLayout,
) -> PaperRestartBundleWriteReport:
    """Write the three files consumed by the next paper-protocol year.

    The paper job renames each component ``*_restart.nc`` to ``*_start.nc``
    before the following executable year. This API writes the consumer names
    directly and refuses to overwrite an existing handoff.

    Fortran provenance: ``dim2_driver.f90`` lines 1411-1429,
    ``sechiba.f90::sechiba_finalize`` lines 1800-1923, and
    ``stomate.f90::stomate_finalize`` lines 5445-5496.
    """

    if not isinstance(state, PaperRestartBundleState):
        raise TypeError("state must be a PaperRestartBundleState")
    if not isinstance(physical_state, PaperRestartBundlePhysicalState):
        raise TypeError("physical_state must be a PaperRestartBundlePhysicalState")
    if not isinstance(pft_layout, PFTRunLayout):
        raise TypeError("pft_layout must be a PFTRunLayout")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "driver": output / "driver_start.nc",
        "sechiba": output / "sechiba_start.nc",
        "stomate": output / "stomate_start.nc",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(existing[0])
    if state.source_clock_runtime is not None:
        physical_state = PaperRestartBundlePhysicalState(**{
            role: physical_state_with_source_clock(getattr(physical_state, role), state.source_clock_runtime)
            for role in paths
        })

    driver_report = write_dim2_driver_restart(
        paths["driver"], state=state.driver, physical_state=physical_state.driver
    )
    sechiba_report = write_sechiba_restart_state(
        paths["sechiba"],
        state=state.sechiba,
        physical_state=physical_state.sechiba,
        pft_layout=pft_layout,
    )
    stomate = state.stomate
    stomate_report = write_stomate_full_writerestart_states(
        paths["stomate"],
        physical_state=physical_state.stomate,
        entry_state=stomate.entry_state,
        season_state=stomate.season_state,
        daily_state=stomate.daily_state,
        gas_state=stomate.gas_state,
        remainder_state=stomate.remainder_state,
        pft_layout=pft_layout,
    )
    # Legacy science-only writers remain explicit compatibility APIs. Whenever
    # any versioned clock is supplied, all three actual files must agree.
    if state.source_clock_runtime is not None or any(
        any(name.startswith(PREFIX) for name in (getattr(physical_state, role).global_attributes or {}))
        for role in paths
    ):
        read_source_clock_restart_bundle(paths)
    return PaperRestartBundleWriteReport(
        output_directory=output,
        driver=driver_report,
        sechiba=sechiba_report,
        stomate=stomate_report,
    )
