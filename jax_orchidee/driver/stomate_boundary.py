"""Driver/restart assembly for the first STOMATE OK_LEAK boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jax_orchidee.coupled import StomatePreStepOkLeakBoundaryInputs, stomate_pre_step_ok_leak_boundary_inputs
from jax_orchidee.driver.bundle import DriverFirstStepBundle, DriverStepBundle
from jax_orchidee.driver.init import parse_run_def, parse_run_def_bool, parse_run_def_float, parse_run_def_int
from jax_orchidee.driver.restart import reference_case_output_dir
from jax_orchidee.driver.sechiba_boundary import IntersurfFirstStepPayload, build_intersurf_first_step_payload
from jax_orchidee.parameters.pft_catalog import load_pft_catalog, read_pft_layout_from_netcdf
from jax_orchidee.driver.domain import load_case_config
from jax_orchidee.sechiba.diffuco import cwrr_diaglev_from_vertical_soil_params, cwrr_vertical_soil_grid_from_params
from jax_orchidee.stomate.reference import (
    StomateReferenceFiles,
    find_stomate_reference_files,
    read_stomate_restart_entry_state,
    stomate_reference_files_from_run_dir,
)


FIRST_STEP_STOMATE_BOUNDARY_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_parameters/control.f90::control_initialize lines 103-104 and 584-603",
    "fortran_source/ORCHIDEE/src_parameters/vertical_soil.f90::vertical_soil_init lines 175-263 and 331-454",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 973-1023",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 2430-2554, 3293-3391",
    "outputs/server_1961_trace_full_20260623/run/used_run.def lines 109-111 and 261-283",
)


@dataclass(frozen=True)
class PaperFirstStepStomateBoundary:
    """Reusable source-backed first-step STOMATE boundary assembly."""

    payload: IntersurfFirstStepPayload
    stomate_files: StomateReferenceFiles
    used_run_def_path: Path
    diaglev: np.ndarray
    zz_coef_deep: np.ndarray
    zz_deep: np.ndarray
    nflow: int
    river_routing: bool
    pre_step: StomatePreStepOkLeakBoundaryInputs
    provenance: tuple[str, ...] = FIRST_STEP_STOMATE_BOUNDARY_PROVENANCE


def default_used_run_def_path(config_path: str | Path) -> Path:
    """Resolve the archived run-definition emitted by the audited server run."""

    path = Path(config_path).resolve().parents[1] / "outputs" / "server_1961_trace_full_20260623" / "run" / "used_run.def"
    if path.exists():
        return path
    return reference_case_output_dir(config_path) / "used_run.def"


def paper_case_vertical_grids_from_used_run_def(used_run_def_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build CWRR/STOMATE vertical grids from the actual used ``run.def``.

    Fortran provenance: ``vertical_soil.f90::vertical_soil_init`` lines
    175-263 read the depth parameters and lines 331-454 build hydrology
    ``znh`` plus thermal/deep-carbon ``znt``/``zlt``. ``control.f90`` lines
    584-603 copies ``znt(1:nslm)`` into the CWRR ``diaglev`` vector, while
    ``slowproc.f90`` lines 437-442 and ``stomate_soilcarbon.f90`` consume the
    full ``ndeep`` deep-carbon ``zz_deep``/``zz_coef_deep`` vectors.
    """

    values = parse_run_def(used_run_def_path)
    grid = cwrr_vertical_soil_grid_from_params(
        depth_max_h=parse_run_def_float(values, "DEPTH_MAX_H"),
        depth_max_t=parse_run_def_float(values, "DEPTH_MAX_T"),
        depth_topthickness=parse_run_def_float(values, "DEPTH_TOPTHICK"),
        depth_cstthickness=parse_run_def_float(values, "DEPTH_CSTTHICK"),
        depth_geom=parse_run_def_float(values, "DEPTH_GEOM"),
        ratio_geom_below=parse_run_def_float(values, "RATIO_GEOM_BELOW"),
    )
    return (
        np.asarray(grid.diaglev, dtype=np.float64),
        np.asarray(grid.zlt, dtype=np.float64),
        np.asarray(grid.znt, dtype=np.float64),
    )


def paper_case_cwrr_vertical_soil_grid_from_used_run_def(used_run_def_path: str | Path):
    """Build full CWRR vertical-soil grid arrays from ``used_run.def``.

    Fortran provenance: ``vertical_soil.f90::vertical_soil_init`` lines
    268-363 define HYDROL ``znh``, ``dnh``, and ``dlh``; lines 377-447 define
    thermal ``znt``/``zlt``/``dlt``. This helper exposes the full grid when a
    caller needs HYDROL internode distances rather than only ``diaglev``.
    """

    values = parse_run_def(used_run_def_path)
    return cwrr_vertical_soil_grid_from_params(
        depth_max_h=parse_run_def_float(values, "DEPTH_MAX_H"),
        depth_max_t=parse_run_def_float(values, "DEPTH_MAX_T"),
        depth_topthickness=parse_run_def_float(values, "DEPTH_TOPTHICK"),
        depth_cstthickness=parse_run_def_float(values, "DEPTH_CSTTHICK"),
        depth_geom=parse_run_def_float(values, "DEPTH_GEOM"),
        ratio_geom_below=parse_run_def_float(values, "RATIO_GEOM_BELOW"),
    )


def paper_1961_first_step_stomate_boundary(
    config_path: str | Path,
    *,
    bundle: DriverFirstStepBundle | DriverStepBundle,
    payload: IntersurfFirstStepPayload | None = None,
    used_run_def_path: str | Path | None = None,
    root: str | Path | None = None,
    run_dir: str | Path | None = None,
    kjit: int = 1,
    nbp_glo: int | None = None,
) -> PaperFirstStepStomateBoundary:
    """Assemble the source-backed first-step STOMATE OK_LEAK boundary.

    This function is a composition point, not a process kernel. It reads the
    actual archived run-definition for vertical-grid and routing scalars, reads
    STOMATE restart state, and delegates boundary construction to
    ``stomate_pre_step_ok_leak_boundary_inputs``. Same-step HYDROL,
    THERMOSOIL, and litter-control fields remain absent until the ordered
    SECHIBA -> STOMATE step produces them.
    """

    if isinstance(bundle, DriverStepBundle) and bundle.tstep != 0:
        raise ValueError("first-step STOMATE restart boundary is only valid for tstep=0")
    payload = payload or build_intersurf_first_step_payload(bundle)
    run_def_path = Path(used_run_def_path) if used_run_def_path is not None else default_used_run_def_path(config_path)
    values = parse_run_def(run_def_path)
    diaglev, zz_coef_deep, zz_deep = paper_case_vertical_grids_from_used_run_def(run_def_path)
    stomate_files = (
        stomate_reference_files_from_run_dir(run_dir)
        if run_dir is not None
        else find_stomate_reference_files(Path(root) if root is not None else Path(config_path).resolve().parents[1])
    )
    catalog = load_pft_catalog(load_case_config(config_path)["pft_catalog"]["path"])
    source_pft_layout = read_pft_layout_from_netcdf(
        stomate_files.restart,
        catalog,
        legacy_layout_id="paper_250919_legacy14",
    )
    state = read_stomate_restart_entry_state(
        stomate_files.restart,
        source_pft_layout=source_pft_layout,
        target_pft_layout=bundle.run_scalars.pft_layout,
    )
    nflow = parse_run_def_int(values, "NSTM")
    river_routing = parse_run_def_bool(values["RIVER_ROUTING"])

    pre_step = stomate_pre_step_ok_leak_boundary_inputs(
        state=state,
        driver_payload=payload,
        run_scalars=bundle.run_scalars,
        diaglev=diaglev,
        zz_coef_deep=zz_coef_deep,
        zz_deep=zz_deep,
        kjit=kjit,
        nflow=nflow,
        river_routing=river_routing,
        nbp_glo=payload.kjpindex if nbp_glo is None else nbp_glo,
    )
    return PaperFirstStepStomateBoundary(
        payload=payload,
        stomate_files=stomate_files,
        used_run_def_path=run_def_path,
        diaglev=diaglev,
        zz_coef_deep=zz_coef_deep,
        zz_deep=zz_deep,
        nflow=nflow,
        river_routing=river_routing,
        pre_step=pre_step,
    )
