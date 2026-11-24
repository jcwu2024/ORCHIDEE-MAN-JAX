"""Aggregate first-step restart state for strict driver orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from netCDF4 import Dataset

from jax_orchidee.driver.domain import load_case_config
from jax_orchidee.driver.init import parse_run_def_bool, read_run_scalars
from jax_orchidee.driver.restart import SechibaStaticRestart, read_sechiba_static_restart, reference_restart_path
from jax_orchidee.driver.source_clock import PREFIX, RestartClockBundle, read_source_clock_restart_bundle
from jax_orchidee.parameters.pft_catalog import (
    PFTRunLayout,
    load_pft_catalog,
    read_pft_layout_from_netcdf,
)
from jax_orchidee.sechiba.diffuco import DiffucoFirstStepRestartState, read_diffuco_first_step_restart_state
from jax_orchidee.sechiba.enerbil import (
    DriverAlbedoRestart,
    EnerbilSoilThermalStateRestart,
    read_driver_albedo_restart,
    read_enerbil_soil_thermal_state_restart,
)
from jax_orchidee.sechiba.hydrol_reference import HydrolRestartAnchors, read_hydrol_restart_anchors
from jax_orchidee.sechiba.restart_io import SechibaRestartState, read_sechiba_init_restart_state
from jax_orchidee.sechiba.slowproc import SlowprocRestartEntryState, read_slowproc_restart_entry_state
from jax_orchidee.sechiba.thermosoil import ThermosoilRestartState, read_thermosoil_restart_state
from jax_orchidee.stomate.reference import (
    DEFAULT_THAWED_HUMIDITY,
    StomateRestartEntryState,
    find_stomate_reference_files,
    read_stomate_daily_accumulator_state,
    read_stomate_restart_entry_state,
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


def readstart_thawed_humidity_controls(run_def_values: Mapping[str, str]) -> tuple[bool, float]:
    """Resolve the optional STOMATE readstart reset (stomate_io.f90:425,1201-1213)."""
    values = {key.upper(): value for key, value in run_def_values.items()}
    reset = parse_run_def_bool(values.get("RESET_THAWED_HUMIDITY", "FALSE"))
    if reset and "THAWED_HUMIDITY_INPUT" in values:
        token = str(values["THAWED_HUMIDITY_INPUT"]).upper().replace("D", "E")
        return reset, float(token)
    return reset, DEFAULT_THAWED_HUMIDITY


def resolve_hydrol_initialization_refsoc_1d(
    config_path: str | Path,
    *,
    run_def_values: Mapping[str, object],
    npts: int,
    restart_path: str | Path | None = None,
    restored_refSOC_1d=None,
    domain=None,
) -> np.ndarray | None:
    """Bind the source HYDROL static input at a new initialization boundary.

    Fortran provenance: hydrol.f90::hydrol_initialize 625-636 reads the
    configured restart refSOC_1d before hydrol_init/var_init 659-676. Only
    ALL(refSOC_1d == val_exp) invokes read_refSOC_1dfile 9721-9847; a partial
    sentinel vector is retained. hydrol_finalize 1748 conditionally writes
    this separate point input. Mineral parameter vectors and SOC-adjusted
    effective mcs/mcf/mcw are different objects.

    An explicit restored vector is authoritative for an in-memory new-native
    run. Otherwise read the selected restart file, then use the existing
    static reader only for an absent/all-missing field. Continuous execution
    and complete checkpoint decoding must not call this initialization owner.
    """
    from jax_orchidee.driver.init import parse_run_def_bool
    from jax_orchidee.driver.static import read_paper_hydrol_refsoc_1d_static_field
    from jax_orchidee.sechiba.hydrol import HYDROL_INIT_RESTART_EXPVAL

    if not parse_run_def_bool(run_def_values.get("use_refSOC_hydrol", "FALSE")):
        return None
    if int(npts) != npts or int(npts) < 1:
        raise ValueError("HYDROL refSOC_1d requires positive integer npts")
    npts = int(npts)

    def point_vector(value):
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (npts,):
            raise ValueError(f"HYDROL refSOC_1d must have shape ({npts},), got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("HYDROL refSOC_1d must contain finite values")
        return array

    refsoc = None
    if restored_refSOC_1d is not None:
        refsoc = point_vector(restored_refSOC_1d)
    elif restart_path is not None:
        with Dataset(restart_path) as dataset:
            if "refSOC_1d" in dataset.variables:
                variable = dataset.variables["refSOC_1d"]
                # Read actual stored values, preserving the selected source's
                # 999999 sentinel rather than applying masked-array defaults.
                variable.set_auto_maskandscale(False)
                stored = np.asarray(variable[:], dtype=np.float64)
                if variable.dimensions and variable.dimensions[0] == "time":
                    stored = stored[0]
                refsoc = point_vector(stored.reshape(-1))
    if refsoc is not None and not np.all(refsoc == HYDROL_INIT_RESTART_EXPVAL):
        return refsoc
    if domain is None or getattr(domain, "resolution", None) is None:
        raise ValueError("HYDROL refSOC_1d static fallback requires explicit driver domain/resolution")
    refsoc, _ = read_paper_hydrol_refsoc_1d_static_field(
        config_path, lalo=domain.lalo, resolution_m=domain.resolution,
    )
    return point_vector(refsoc)


def restart_thermosoil_initialization_supported(context: object) -> bool:
    """Limit the new lifecycle reconstruction to its audited source path."""
    from jax_orchidee.driver.init import parse_run_def_bool
    values = context.run_def_values
    flags_enabled = (
        parse_run_def_bool(values.get("OK_EXPLICITSNOW", "FALSE"))
        and parse_run_def_bool(values.get("USE_SOILC_TEMPDIFF", "FALSE"))
        # thermosoil.f90:362-371 initializes TRUE before optional use_refSOC.
        and parse_run_def_bool(values.get("use_refSOC", "TRUE"))
    )
    # thermosoil_initialize 337-374 reads the soil-carbon switches only
    # inside this source gate; raw run.def flags alone do not activate them.
    freeze_thermix = parse_run_def_bool(values.get(
        "OK_FREEZE_THERMIX", values.get("OK_FREEZE", "FALSE")
    ))
    return flags_enabled and (
        (freeze_thermix and context.hydrol_ok_pc) or context.hydrol_ok_leak
    )


def restart_thermosoil_initial_fields(
    source: Mapping[str, object], *, context: object, payload: object, altmax: object
) -> dict[str, object]:
    """Rebuild nonpersisted thermal memory before the first restarted step.

    This composes existing kernels only. Fortran provenance:
    sechiba_initialize 640-643, 703-729; enerbil_initialize 241,259-261;
    hydrol_var_init 3946-4644; thermosoil_initialize 572-573,719-744;
    thermosoil_var_init 1309-1320. The complete restart coefficient group is
    retained; pcapa_en/temp_sol_beg are not written at finalize 1122-1126.
    """
    from jax_orchidee.driver.init import parse_run_def_bool
    from jax_orchidee.sechiba.hydrol import (
        hydrol_cold_start_thermosoil_moisture_inputs,
        hydrol_depth_parameters_from_run_def,
        hydrol_soil_parameters_from_run_def,
        hydrol_var_init,
    )
    from jax_orchidee.sechiba.thermosoil import (
        thermosoil_diaglev_from_ptn,
        thermosoil_getdiff_explicit,
        thermosoil_getdiff_thinsnow,
        thermosoil_humlev,
        thermosoil_soilc_tempdiff_fraction,
    )
    values = context.run_def_values
    flag = lambda name, default: parse_run_def_bool(values.get(name, default))
    if not restart_thermosoil_initialization_supported(context):
        raise ValueError("restart thermal initialization requires the audited explicit-snow/refSOC path")
    freeze_thermix = flag("OK_FREEZE_THERMIX", "TRUE" if flag("OK_FREEZE", "FALSE") else "FALSE")
    effective_wetdiag = flag("OK_WETDIAGLONG", "FALSE") or (
        freeze_thermix and context.hydrol_ok_pc
    ) or context.hydrol_ok_leak
    if not effective_wetdiag:
        raise ValueError("restart thermal initialization requires effective long-term wet diagnostics")
    soil = getattr(context, "hydrol_soil_parameters", None)
    if soil is None:
        soil = hydrol_soil_parameters_from_run_def(values)
    depth = getattr(context, "hydrol_depth_parameters", None)
    if depth is None:
        depth = hydrol_depth_parameters_from_run_def(values)
    hydro = hydrol_var_init(
        veget=source["veget"], veget_max=source["veget_max"],
        soiltile=context.first_step_bundle.vegetation.soiltile,
        njsc=source["njsc"], znh_m=context.cwrr_grid.znh,
        dnh_m=context.cwrr_grid.dnh, dlh_m=context.cwrr_grid.dlh,
        humcste=context.hydrol_humcste,
        altmax=altmax,
        mc=source["mc"], mcl=source["mcl"],
        water2infilt=source["water2infilt"], resdist=source["resdist"],
        vegtot_old=source["vegtot_old"], qsintveg=source["qsintveg"],
        drysoil_frac=source["drysoil_frac"],
        soil_parameters=soil,
        depth_parameters=depth,
        refSOC_1d=source.get("refSOC_1d"),
        use_refSOC_hydrol=flag("use_refSOC_hydrol", "FALSE"),
        ok_pc=context.hydrol_ok_pc, ok_leak=context.hydrol_ok_leak,
        peat_hydro=context.hydrol_soil_peat_hydro,
        tides=context.hydrol_tides, agri_peat=context.hydrol_agri_peat,
        ok_freeze_cwrr=context.ok_freeze_cwrr,
        zmaxh=context.hydrol_depth_max_h,
    )
    moisture = hydrol_cold_start_thermosoil_moisture_inputs(
        hydro, pref_soil_veg=context.run_scalars.pref_soil_veg,
        dz_mm=context.hydrol_dz_mm, dh_mm=context.hydrol_dh_mm,
        njsc=source["njsc"], mcs=hydro.mcs,
        soil_parameters=soil,
        peat_hydro=context.hydrol_soil_peat_hydro,
    )
    humlev = thermosoil_humlev(
        **{name: getattr(moisture, name) for name in (
            "shumdiag_perma", "mc_layh", "mcl_layh", "tmc_layh",
            "mc_layh_pft", "mcl_layh_pft", "tmc_layh_pft",
        )},
        znt=context.cwrr_grid.znt, zlt=context.cwrr_grid.zlt,
        dz5=context.cwrr_grid.dz5, satsoil=context.thermosoil_satsoil,
    )
    zx1, _ = thermosoil_soilc_tempdiff_fraction(refsoc=source["refsoc"], nvm=context.nvm)
    getdiff = thermosoil_getdiff_explicit(
        ptn=source["ptn"], njsc=source["njsc"], veget_max=source["veget_max"],
        shum_ngrnd_permalong=source["shum_ngrnd_permalong"],
        mc_layt=humlev.mc_layt, tmc_layt_pft=humlev.tmc_layt_pft,
        snowrho=source["snowrho"], snowtemp=source["snowtemp"], pb=payload.pb,
        dlt=context.cwrr_grid.dlt, zx1=zx1, ok_laidev=context.ok_laidev,
        ok_freeze_thermix=freeze_thermix,
        brk_flag=int(values.get("BEDROCK_FLAG", "0")),
        nslm=len(context.cwrr_grid.znh),
    )
    thin = thermosoil_getdiff_thinsnow(
        ptn=source["ptn"], shum_ngrnd_permalong=source["shum_ngrnd_permalong"],
        snowdz=source["snowdz"], pcapa=getdiff.pcapa,
        pcapa_en=getdiff.pcapa_en, pkappa=getdiff.pkappa,
        profil_froz=getdiff.profil_froz,
        veget_mask_2d=np.ones_like(np.asarray(source["veget_max"]), dtype=bool),
        zlt=context.cwrr_grid.zlt,
    )
    return {
        "temp_sol_beg": source["temp_sol"],
        "pcapa_en": thin.pcapa_en,
        "stempdiag": thermosoil_diaglev_from_ptn(
            ptn=source["ptn"], veget_max=source["veget_max"],
            nslm=len(context.cwrr_grid.znh),
        ),
    }


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
    pft_layout: PFTRunLayout
    stomate_source_pft_layout: PFTRunLayout
    sechiba_source_pft_layout: PFTRunLayout
    provenance: tuple[str, ...] = FIRST_STEP_RESTART_STATE_PROVENANCE
    source_clock: RestartClockBundle | None = None

    @property
    def source_clock_runtime(self):
        return None if self.source_clock is None else self.source_clock.runtime


def reference_case_first_step_restart_state(
    config_path: str | Path,
    *,
    root: str | Path | None = None,
    run_dir: str | Path | None = None,
    run_def_path: str | Path | None = None,
    stomate_filename: str = "stomate_start.nc",
    source_clock_legacy_policy: str | None = None,
) -> ReferenceCaseFirstStepRestartState:
    """Read exact first-step restart state from the local paper reference.

    The paper driver uses ``driver_start.nc``, ``sechiba_start.nc``, and
    ``stomate_start.nc`` after the first loop year. This helper aggregates
    those readers for the new-run initialization phase. The complete SECHIBA
    snapshot applies source-defined HYDROL US_INIT/vegstress initialization before
    its first main call; the other readers remain raw diagnostic anchors.
    No model timestep runs here. Complete runtime checkpoint recovery and
    in-memory continuation have separate entry points.
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

    paths = (driver_start, sechiba_start, stomate_input)
    versioned = False
    for path in paths:
        with Dataset(path) as dataset:
            versioned = versioned or any(name.startswith(PREFIX) for name in dataset.ncattrs())
    source_clock = None
    if versioned or source_clock_legacy_policy is not None:
        source_clock = read_source_clock_restart_bundle(
            paths, legacy_policy=source_clock_legacy_policy or "reject"
        )

    config = load_case_config(config_path)
    catalog = load_pft_catalog(config["pft_catalog"]["path"])
    run_scalars = read_run_scalars(config_path, run_def_path)
    target_pft_layout = run_scalars.pft_layout
    legacy_layout_id = "paper_250919_legacy14"
    stomate_source_pft_layout = read_pft_layout_from_netcdf(
        stomate_input,
        catalog,
        legacy_layout_id=legacy_layout_id,
    )
    sechiba_source_pft_layout = read_pft_layout_from_netcdf(
        sechiba_start,
        catalog,
        legacy_layout_id=legacy_layout_id,
    )

    stomate = read_stomate_restart_entry_state(
        stomate_input,
        source_pft_layout=stomate_source_pft_layout,
        target_pft_layout=target_pft_layout,
        allow_missing_thawed_humidity=True,
    )
    stomate_daily = read_stomate_daily_accumulator_state(
        stomate_input,
        source_pft_layout=stomate_source_pft_layout,
        target_pft_layout=target_pft_layout,
    )
    stomate_season = read_stomate_restart_season_state(
        stomate_input,
        source_pft_layout=stomate_source_pft_layout,
        target_pft_layout=target_pft_layout,
    )
    with Dataset(stomate_input) as dataset:
        nvert = int(dataset.variables["uo_0"].shape[1])
        months_num = int(dataset.variables["fwet_series"].shape[1])
        nbpools = int(dataset.variables["MatrixV"].shape[1])
        nsnow = int(dataset.variables["O2_snow"].shape[2])
    # Reuse the same configured season scales as the daily coupling. This
    # local import avoids making the restart-state module depend on coupling
    # initialization order. stomate_data.f90:700 converts the coefficient to
    # days; stomate_io.f90:620-637 uses that cap for missing/sentinel tau.
    from jax_orchidee.coupled import paper_case_season_time_scales

    season_scales = paper_case_season_time_scales(
        config_path, used_run_def=run_def_path
    )
    reset_thawed_humidity, thawed_humidity_input = readstart_thawed_humidity_controls(run_scalars.raw_run_def)
    readstart = read_stomate_readstart_states_from_template(
        stomate_input,
        t2m=stomate_daily.t2m_daily,
        nvm=stomate_source_pft_layout.n_pft,
        nslm=stomate_season.tsoil_month.shape[1],
        ndeep=stomate.carbon_32l.shape[3],
        nsnow=nsnow,
        nvert=nvert,
        months_num=months_num,
        ncarb=stomate.carbon.shape[1],
        nlitt=stomate.litter.shape[1],
        nbpools=nbpools,
        tau_longterm_max=season_scales.tau_longterm_max,
        reset_thawed_humidity=reset_thawed_humidity,
        thawed_humidity_input=thawed_humidity_input,
        source_pft_layout=stomate_source_pft_layout,
        target_pft_layout=target_pft_layout,
    )

    return ReferenceCaseFirstStepRestartState(
        run_dir=resolved_run_dir,
        driver_start=driver_start,
        sechiba_start=sechiba_start,
        stomate_input=stomate_input,
        driver_albedo=read_driver_albedo_restart(driver_start),
        sechiba_static=read_sechiba_static_restart(
            sechiba_start,
            source_pft_layout=sechiba_source_pft_layout,
            target_pft_layout=target_pft_layout,
        ),
        diffuco_precall=read_diffuco_first_step_restart_state(
            sechiba_start,
            source_pft_layout=sechiba_source_pft_layout,
            target_pft_layout=target_pft_layout,
        ),
        hydrol=read_hydrol_restart_anchors(
            sechiba_start,
            source_pft_layout=sechiba_source_pft_layout,
            target_pft_layout=target_pft_layout,
        ),
        enerbil_thermal=read_enerbil_soil_thermal_state_restart(
            sechiba_start,
            source_pft_layout=sechiba_source_pft_layout,
            target_pft_layout=target_pft_layout,
        ),
        thermosoil=read_thermosoil_restart_state(
            sechiba_start,
            source_pft_layout=sechiba_source_pft_layout,
            target_pft_layout=target_pft_layout,
        ),
        slowproc=read_slowproc_restart_entry_state(
            sechiba_start,
            source_pft_layout=sechiba_source_pft_layout,
            target_pft_layout=target_pft_layout,
        ),
        # Cold/default handling belongs to STOMATE readstart (1201-1213).
        # The warm driver also uses this entry directly at the next boundary.
        stomate=stomate._replace(thawed_humidity=readstart.entry_state.thawed_humidity),
        stomate_readstart=readstart,
        sechiba_restart_state=read_sechiba_init_restart_state(
            sechiba_start,
            npts=int(stomate.biomass.shape[0]),
            nvm=target_pft_layout.n_pft,
            pref_soil_veg=run_scalars.pref_soil_veg,
            nstm=run_scalars.nstm,
            nslm=int(stomate_season.tsoil_month.shape[1]),
            run_def_values=run_scalars.raw_run_def,
            source_pft_layout=sechiba_source_pft_layout,
            target_pft_layout=target_pft_layout,
        ),
        pft_layout=target_pft_layout,
        stomate_source_pft_layout=stomate_source_pft_layout,
        sechiba_source_pft_layout=sechiba_source_pft_layout,
        source_clock=source_clock,
    )
