"""Source-backed explicit coupling from SECHIBA payloads into STOMATE.

This module is intentionally a composition layer. It wires already implemented
SECHIBA and STOMATE helpers in Fortran call order and refuses to synthesize
missing driver, restart, season, or process state.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Mapping, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from jax_orchidee.driver.init import RunScalars
from jax_orchidee.driver.sechiba_boundary import IntersurfFirstStepPayload
from jax_orchidee.sechiba.sechiba_step import (
    SechibaExplicitCoupledStepResult,
    SechibaExplicitLocalDiffucoStepResult,
    sechiba_explicit_coupled_step_from_local_diffuco,
)
from jax_orchidee.sechiba.diffuco import (
    humcste_from_pft_to_mtc,
    humcste_use_from_humcste,
    rprof_from_humcste_use,
    z_soil_from_cwrr_vertical_soil_params,
)
from jax_orchidee.sechiba.slowproc import SlowprocRestartEntryState
from jax_orchidee.stomate.entry import (
    StomateMainPayloadAssembly,
    assemble_stomate_main_payload,
    stomate_driver_entry_source,
    stomate_no_routing_entry_source,
)
from jax_orchidee.stomate.daily_inputs import MIN_STOMATE
from jax_orchidee.stomate.integration import (
    ExplicitDailyCarbonScheduledGppMaintenancePrescribeConstraintsAllocKillGapTurnoverResult,
    ExplicitOkLeakFromPostNppResult,
    ExplicitStomateLpjOutputsResult,
    stomate_daily_scheduled_gpp_maintenance_prescribe_constraints_alloc_kill_gap_turnover_explicit,
    stomate_lpj_outputs_from_post_npp_ok_leak_explicit,
    stomate_ok_leak_from_post_npp_explicit,
)
from jax_orchidee.stomate.carbon_kernels import (
    DEFAULT_UNDEF,
    ICARBON,
    ICARBRES,
    IAGRHRTPN,
    IAGRHRTST,
    IAGRSAPPN,
    IAGRSAPST,
    IFRUIT,
    IHEARTABOVE,
    IHEARTBELOW,
    ILEAF,
    IROOT,
    ISAPABOVE,
    ISAPBELOW,
    NPARTS,
    littercalc_aboveground_controls,
    stomate_soil_mc_32l,
)
from jax_orchidee.stomate.permafrost import stomate_permafrost_decomposition_controls
from jax_orchidee.stomate.parameters import (
    PaperCaseStomateParameterBundle,
    load_paper_case_stomate_parameter_bundle,
    read_run_def_values,
)
from jax_orchidee.stomate.reference import StomateDailyAccumulatorState, StomateRestartEntryState, StomateRestartSeasonState
from jax_orchidee.stomate.season import (
    SEASON_MEMORY_PROVENANCE,
    SeasonAnnualState,
    SeasonAnnualStepResult,
    SeasonBiometeorologyState,
    SeasonBiometeorologyStepResult,
    SeasonMemoryState,
    SeasonMemoryStepResult,
    SeasonTimeScales,
    season_annual_step,
    season_biometeorology_step,
    season_memory_step,
)
from jax_orchidee.stomate.soilcarbon_kernels import (
    altcalc_doc,
    soilcarbon_leak_tf_doc_inputs,
    soilcarbon_perma_peat_cmax,
)


SECHIBA_STOMATE_EXPLICIT_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 997-1216",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 973-1023",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3020-3267 and 4297-4335",
    "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj lines 944-1557",
)

SECHIBA_STOMATE_OK_LEAK_EXPLICIT_PROVENANCE = (
    *SECHIBA_STOMATE_EXPLICIT_PROVENANCE,
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3288-3489",
    "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90::littercalc_leak lines 1950-2799",
    "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::soilcarbon_leak lines 1176-2303",
)

SECHIBA_STOMATE_OUTPUT_EXPLICIT_PROVENANCE = (
    *SECHIBA_STOMATE_OK_LEAK_EXPLICIT_PROVENANCE,
    "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj lines 1578-1663, 1677-1699, and 2200-2246",
    "fortran_run_scripts/paper_250919/c2.4_Model_run_functions_sensitivity.py lines 50, 60-76, and 115-118",
)

STOMATE_SAME_STEP_OK_LEAK_BOUNDARY_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3034-3124 and 3270-3489",
    "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90::littercalc_leak lines 1645-2799",
    "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::soilcarbon_leak lines 1008-2303",
)

SECHIBA_TO_STOMATE_REQUIRED_FIELDS = (
    "gpp",
    "veget_max",
    "totfrac_nobio",
    "stempdiag",
    "temp_sol",
    "evapot_corr",
    "t2mdiag",
    "humrel",
    "litterhumdiag",
    "shumdiag",
    "soil_mc",
    "wat_flux",
)

SECHIBA_TO_OK_LEAK_HYDROL_FIELDS = (
    "soil_mc",
    "wat_flux",
    "runoff_per_soil",
    "drainage_per_soil",
    "runoff2peat",
    "canopy2ground",
)


class SechibaStomateExplicitResult(NamedTuple):
    """Result of the current explicit SECHIBA -> STOMATE main-chain coupling."""

    sechiba: SechibaExplicitCoupledStepResult
    entry_payload: StomateMainPayloadAssembly
    stomate: ExplicitDailyCarbonScheduledGppMaintenancePrescribeConstraintsAllocKillGapTurnoverResult
    missing_entry_inputs: tuple[str, ...]
    partial_entry_inputs: tuple[str, ...]
    provenance: tuple[str, ...] = SECHIBA_STOMATE_EXPLICIT_PROVENANCE
    notes: tuple[str, ...] = (
        "This helper wires exact local SECHIBA outputs into the closed PFT14 STOMATE GPP/maintenance/prescribe/constraints/phenology(none)/alloc/post-NPP chain.",
        "It does not read driver forcing, restart state, season accumulators, or parameter tables by itself.",
        "All missing STOMATE state remains explicit in the caller-owned input dictionaries.",
    )


class LocalDiffucoSechibaStomateExplicitResult(NamedTuple):
    """Result from local DIFFUCO through SECHIBA into the STOMATE main chain."""

    local_sechiba: SechibaExplicitLocalDiffucoStepResult
    coupled: SechibaStomateExplicitResult
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 997-1216",
        "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_main lines 629-710",
        *SECHIBA_STOMATE_EXPLICIT_PROVENANCE[1:],
    )
    notes: tuple[str, ...] = (
        "This is the current highest local explicit closure: local PFT DIFFUCO, SECHIBA downstream modules, and STOMATE PFT14 main-chain composition.",
        "Driver forcing readers, static/restart assembly, season accumulators, and full yearly scheduling remain explicit upstream responsibilities.",
    )


class SechibaStomateOkLeakExplicitResult(NamedTuple):
    """Result of current SECHIBA -> STOMATE carbon -> OK_LEAK composition."""

    coupled: SechibaStomateExplicitResult
    ok_leak: ExplicitOkLeakFromPostNppResult
    provenance: tuple[str, ...] = SECHIBA_STOMATE_OK_LEAK_EXPLICIT_PROVENANCE
    notes: tuple[str, ...] = (
        "This composes the already closed local STOMATE post-NPP chain with the audited OK_LEAK litter/soilcarbon/DOC adapter.",
        "OK_LEAK process-boundary inputs that are not produced by the current SECHIBA/STOMATE chain remain explicit caller inputs.",
        "Maintenance respiration is consumed from the upstream STOMATE result and is not recomputed.",
    )


class SechibaStomateOutputExplicitResult(NamedTuple):
    """Result of current SECHIBA -> STOMATE -> OK_LEAK -> output mapping."""

    ok_leak_chain: SechibaStomateOkLeakExplicitResult
    outputs: ExplicitStomateLpjOutputsResult
    provenance: tuple[str, ...] = SECHIBA_STOMATE_OUTPUT_EXPLICIT_PROVENANCE
    notes: tuple[str, ...] = (
        "This is the current highest local explicit composition to paper modelout fields.",
        "It still requires explicit driver/static/restart/season and OK_LEAK boundary inputs not produced by the local chain.",
    )


class StomateRestartInputBundles(NamedTuple):
    """Restart-derived STOMATE explicit-chain input dictionaries."""

    prescribe_inputs: dict[str, object]
    constraints_inputs: dict[str, object]
    phenology_inputs: dict[str, object]
    alloc_inputs: dict[str, object]
    post_npp_inputs: dict[str, object]
    maintenance_inputs: dict[str, object]
    daily_process_inputs: dict[str, object]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 583-598",
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 843-911",
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 920-1023",
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 1147-1150 and 1599-1606",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3211-3213 and 4360-4364 pass snow/grass daily accumulators into StomateLpj",
    )


class PaperCaseStomateStaticInputKwargs(NamedTuple):
    """Static STOMATE keyword values loaded from paper-case source files."""

    kwargs: dict[str, object]
    parameters: PaperCaseStomateParameterBundle
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_parameters/pft_parameters.f90::config_stomate_pft_parameters lines 3410-3459, 4155-4242, 4394-4670",
        "fortran_source/ORCHIDEE/src_stomate/stomate_data.f90::stomate_data lines 151-178 and 585-587",
        "fortran_source/ORCHIDEE/src_parameters/constantes_mtc.f90 lines 76-84, 91-96, 140-143, 495-555, and 790-797",
        "outputs/server_1961_trace_full_20260623/run/used_run.def materialized getin_p defaults",
        "fortran_run_scripts/paper_250919/sen_reference_arg2_1.0_001.0-071.0/I10/S2_63.206_0.0876_0.2019_50.658/run.def_63.206_0.0876_0.2019_50.658 selected sensitivity overrides",
    )


class StomateRestartSeasonInputKwargs(NamedTuple):
    """Dynamic season-memory keyword values read from STOMATE restart."""

    kwargs: dict[str, object]
    season: StomateRestartSeasonState
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 600-826",
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::writerestart lines 2419-2567",
        "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season lines 396-772 and 837-955",
    )


class StomateSeasonMemoryInputKwargs(NamedTuple):
    """Dynamic season-memory keyword values after one source-backed season step."""

    kwargs: dict[str, object]
    step: SeasonMemoryStepResult
    provenance: tuple[str, ...] = (
        *SEASON_MEMORY_PROVENANCE,
        "jax_orchidee.coupled.stomate_season_memory_input_kwargs maps updated season memory into the explicit STOMATE chain without filling absent nonlocal inputs",
    )


class StomateSeasonAnnualInputKwargs(NamedTuple):
    """Updated long-term/annual season values for the explicit STOMATE chain."""

    kwargs: dict[str, object]
    step: SeasonAnnualStepResult
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season sections 12-21",
        "jax_orchidee.coupled.stomate_season_annual_input_kwargs maps source-backed annual/long-term season state without filling phenology-only inputs",
    )


class StomateSeasonBiometeorologyInputKwargs(NamedTuple):
    """Updated dormancy/GDD season values for future phenology consumption."""

    kwargs: dict[str, object]
    step: SeasonBiometeorologyStepResult
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season sections 7-11",
        "jax_orchidee.coupled.stomate_season_biometeorology_input_kwargs maps source-backed dormancy/GDD memory without running phenology onset",
    )


class PaperCaseStomatePreStepInputKwargs(NamedTuple):
    """Merged source-backed static and stepped season kwargs before STOMATE chain."""

    kwargs: dict[str, object]
    static: PaperCaseStomateStaticInputKwargs
    memory: StomateSeasonMemoryInputKwargs
    annual: StomateSeasonAnnualInputKwargs
    biometeorology: StomateSeasonBiometeorologyInputKwargs
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 4221-4251 calls season before StomateLpj",
        "jax_orchidee.coupled.paper_case_stomate_pre_step_input_kwargs composes audited static parameters and season-memory updates",
    )


class PaperCaseStomateBoundaryInputKwargs(NamedTuple):
    """Source-backed vertical/root boundary kwargs for the STOMATE chain."""

    kwargs: dict[str, object]
    humcste: object
    humcste_use: object
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_parameters/vertical_soil.f90::vertical_soil_init lines 185-263 and 331-444",
        "fortran_source/ORCHIDEE/src_parameters/control.f90::control_initialize lines 584-603",
        "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_init lines 4093-4107",
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 984-990",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 2907-2912",
    )


class PaperCaseStomateDataInputKwargs(NamedTuple):
    """Source-backed ``stomate_data``/``stomate_init`` parameter inputs."""

    kwargs: dict[str, object]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_parameters/constantes.f90::config_stomate_parameters lines 1246-1340 and 1432-1566",
        "fortran_source/ORCHIDEE/src_stomate/stomate_data.f90::data lines 261-346 and 385-397",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_init lines 8154-8157 and 8303-8306",
    )


class PaperCaseInactiveCropInputKwargs(NamedTuple):
    """Neutral crop-state kwargs for the paper case when STICS is inactive."""

    kwargs: dict[str, object]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_init lines 8338-8493 initializes STICS only when ANY(ok_LAIdev)",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3931-4201 enters driver_stics only for ok_LAIdev(j)",
        "fortran_source/ORCHIDEE/src_stomate/stomate_turnover.f90::turn lines 469-477 consumes nrec only inside ok_LAIdev(j)",
    )


class PaperCaseStomateBundleSourceKwargs(NamedTuple):
    """Merged kwargs for ``stomate_restart_input_bundles`` from audited sources."""

    kwargs: dict[str, object]
    pre_step: PaperCaseStomatePreStepInputKwargs
    boundary: PaperCaseStomateBoundaryInputKwargs
    data: PaperCaseStomateDataInputKwargs
    inactive_crop: PaperCaseInactiveCropInputKwargs
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 4221-4335",
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart daily/season state",
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_init lines 1685-1707 reads height",
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 984-990 builds rprof",
        "fortran_source/ORCHIDEE/src_stomate/stomate_data.f90::data lines 261-397 builds bm_sapl/maxdia",
        "fortran_source/ORCHIDEE/src_stomate/stomate_turnover.f90::turn lines 469-477 makes nrec inactive when ok_LAIdev is false",
    )


class StomateRestartOkLeakStateInputs(NamedTuple):
    """Restart-derived OK_LEAK litter/soil-carbon state inputs."""

    ok_leak_state_inputs: dict[str, object]
    output_inputs: dict[str, object]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 1045-1098",
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 1104-1115 and 1152-1156",
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 1512-1563",
        "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj lines 1578-1663",
    )


class StomateStaticRoutingOkLeakInputs(NamedTuple):
    """Driver/static plus no-routing OK_LEAK boundary inputs."""

    ok_leak_inputs: dict[str, object]
    output_inputs: dict[str, object]
    entry_source: dict[str, object]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1135 and 1184-1216",
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_initialize lines 749-769",
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1227-1259",
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 973-1023",
    )


class StomateVerticalOkLeakInputs(NamedTuple):
    """Source-backed vertical-grid inputs for OK_LEAK and output diagnostics."""

    ok_leak_inputs: dict[str, object]
    output_inputs: dict[str, object]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_parameters/control.f90::control_initialize lines 584-603",
        "fortran_source/ORCHIDEE/src_parameters/vertical_soil.f90::vertical_soil_init lines 331-454",
        "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90::littercalc_leak lines 2027-2030 and 1947",
        "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj lines 906-907 and 1578-1663",
    )


class StomateOkLeakBoundaryInputs(NamedTuple):
    """Merged source-backed OK_LEAK and output boundary dictionaries."""

    ok_leak_inputs: dict[str, object]
    output_inputs: dict[str, object]
    provenance: tuple[str, ...]


class StomatePreStepOkLeakBoundaryInputs(NamedTuple):
    """Source-backed OK_LEAK/output boundary bundle available before SECHIBA runs."""

    boundary_inputs: StomateOkLeakBoundaryInputs
    restart_state: StomateRestartOkLeakStateInputs
    pft_static: StomatePftStaticOkLeakInputs
    static_routing: StomateStaticRoutingOkLeakInputs
    vertical: StomateVerticalOkLeakInputs
    doc_transport: StomateDocTransportOkLeakInputs | None
    provenance: tuple[str, ...]


class StomateSameStepOkLeakBoundaryInputs(NamedTuple):
    """OK_LEAK/output boundary values derivable after the SECHIBA step."""

    boundary_inputs: StomateOkLeakBoundaryInputs
    pre_step: StomateOkLeakBoundaryInputs | None
    pft_static: object | None
    litter_controls: StomateLitterControlsOkLeakInputs | None
    tf_doc: StomateTfDocOkLeakInputs | None
    perma_peat: StomatePermaPeatOkLeakInputs | None
    active_layer: StomateActiveLayerOkLeakInputs | None
    soilwater: StomateSoilwaterOkLeakInputs | None
    permafrost_activity: StomatePermafrostActivityOkLeakInputs | None
    soil_mc_32l: StomateSoilMc32lOkLeakInputs | None
    doc_transport: StomateDocTransportOkLeakInputs | None
    provenance: tuple[str, ...] = STOMATE_SAME_STEP_OK_LEAK_BOUNDARY_PROVENANCE


class StomatePftStaticOkLeakInputs(NamedTuple):
    """Driver/PFT-parameter static inputs for OK_LEAK and soilcarbon."""

    ok_leak_inputs: dict[str, object]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_parameters/pft_parameters.f90 lines 109-123",
        "fortran_source/ORCHIDEE/src_parameters/pft_parameters.f90 lines 247-293",
        "fortran_source/ORCHIDEE/src_parameters/pft_parameters.f90 lines 3011-3051 and 3433-3441",
        "fortran_source/ORCHIDEE/src_parameters/constantes_mtc.f90 lines 135-139, 250-252, 264-266, and 279-281",
    )


class StomateLitterControlsOkLeakInputs(NamedTuple):
    """Computed OK_LEAK aboveground litter controls."""

    ok_leak_inputs: dict[str, object]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90::littercalc_leak lines 2408-2414",
        "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90::littercalc_leak lines 2482-2497",
        "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90::control_moist_func_moyano lines 2904-3037",
    )


class StomateTfDocOkLeakInputs(NamedTuple):
    """Computed TF-DOC deposition and restart canopy-storage inputs."""

    ok_leak_inputs: dict[str, object]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_main lines 1030-1035 and hydrol_canop lines 4993-5103",
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 1009-1010",
        "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::soilcarbon_leak lines 1206-1233 and 1497-1522",
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 1567-1571",
    )


class StomatePermaPeatOkLeakInputs(NamedTuple):
    """Computed PERMA_PEAT Cmax and branch-control inputs."""

    ok_leak_inputs: dict[str, object]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::soilcarbon_leak lines 1008, 1078-1087, and 1375-1437",
        "configs/orchidee_man_250919.yaml run_def_flags PERMA_PEAT and structural_overrides FRAC1/FRAC2",
    )


class StomateActiveLayerOkLeakInputs(NamedTuple):
    """Computed active-layer state for soilcarbon/cryoturbation boundaries."""

    ok_leak_inputs: dict[str, object]
    output_inputs: dict[str, object]
    state_updates: dict[str, object]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::soilcarbon_leak lines 1037-1043 and 1171-1172",
        "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::altcalc_DOC lines 2438-2578",
    )


class StomateSoilwaterOkLeakInputs(NamedTuple):
    """Computed shallow-soil water storage for ``soilcarbon_leak``."""

    ok_leak_inputs: dict[str, object]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::soilcarbon_leak lines 1482-1486",
    )


class StomatePermafrostActivityOkLeakInputs(NamedTuple):
    """Computed microactem activity controls for OK_LEAK litter/soil carbon."""

    ok_leak_inputs: dict[str, object]
    output_inputs: dict[str, object]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3034-3124",
        "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90::littercalc_leak lines 1645-1647 and 1731-1734",
        "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::soilcarbon_leak lines 641-657 and 1176-1193",
        "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90::microactem lines 2468-2703",
    )


class StomateSoilMc32lOkLeakInputs(NamedTuple):
    """Computed deep soil-moisture field for OK_LEAK."""

    ok_leak_inputs: dict[str, object]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3270-3275",
    )


class StomateMcPeatBoundaryInputs(NamedTuple):
    """Computed peat moisture lookup source for ``microactem``."""

    mc_peat: np.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3040-3052",
    )


class StomateDocTransportOkLeakInputs(NamedTuple):
    """Computed DOC transport controls for ``soilcarbon_leak``."""

    ok_leak_inputs: dict[str, object]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_parameters/constantes_var.f90 lines 1396-1400",
        "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::soilcarbon_leak lines 1204-1205",
    )


class SechibaLdDocRoutingOkLeakInputs(NamedTuple):
    """Computed long-distance DOC routing inputs for STOMATE."""

    ok_leak_inputs: dict[str, object]
    entry_source: dict[str, object]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1251-1259",
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 1008-1010",
        "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::soilcarbon_leak lines 1528-1557",
    )


def _require_fields(payload: Mapping[str, object], names: tuple[str, ...], context: str) -> None:
    missing = tuple(name for name in names if name not in payload or payload[name] is None)
    if missing:
        raise ValueError(f"missing {context} fields: {missing}")


def _entry_source_from_sechiba(
    sechiba: SechibaExplicitCoupledStepResult,
    *,
    extra_entry_sources: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    if sechiba.slowproc is None:
        raise ValueError("SECHIBA result must include slowproc output for veget/veget_max/totfrac_nobio")
    source = {
        "gpp": sechiba.diffuco_payload["gpp"],
        "veget": sechiba.slowproc.vegetation.veget,
        "veget_max": sechiba.slowproc.vegetation.veget_max,
        "totfrac_nobio": sechiba.slowproc.vegetation.totfrac_nobio,
        "evapot_corr": sechiba.enerbil_payload["evapot_corr"],
        "stempdiag": sechiba.thermosoil_payload["stempdiag"],
        # Fortran boundary rename: sechiba_main passes HYDROL vegstress into
        # slowproc_main's humrel argument, then slowproc_main forwards it to
        # stomate_main.
        "humrel": sechiba.hydrol_diagnostics.vegstress,
        "litterhumdiag": sechiba.hydrol_diagnostics.litterhumdiag,
        "shumdiag": sechiba.hydrol_diagnostics.shumdiag_perma,
        "soil_mc": sechiba.hydrol_diagnostics.mc_layh_s,
        "wat_flux": sechiba.hydrol_outputs.wat_flux,
    }
    if "t2mdiag" in sechiba.enerbil_payload:
        source["t2mdiag"] = sechiba.enerbil_payload["t2mdiag"]
    if "temp_sol" in sechiba.diffuco_payload:
        source["temp_sol"] = sechiba.diffuco_payload["temp_sol"]
    for extra in extra_entry_sources:
        overlap = set(source).intersection(extra)
        if overlap:
            raise ValueError(f"duplicate SECHIBA/STOMATE entry fields: {tuple(sorted(overlap))}")
        source.update(extra)
    _require_fields(source, SECHIBA_TO_STOMATE_REQUIRED_FIELDS, "SECHIBA-to-STOMATE boundary")
    return source


def _ok_leak_hydrol_source_from_sechiba(sechiba: SechibaExplicitCoupledStepResult) -> dict[str, object]:
    """Map local HYDROL outputs to exact OK_LEAK water-boundary arguments."""

    source = {
        "soil_mc": sechiba.hydrol_diagnostics.mc_layh_s,
        "wat_flux": sechiba.hydrol_outputs.wat_flux,
        "runoff_per_soil": sechiba.hydrol_outputs.runoff_per_soil,
        "drainage_per_soil": sechiba.hydrol_outputs.drainage_per_soil,
        "runoff2peat": sechiba.hydrol_outputs.runoff2peat,
        "canopy2ground": sechiba.hydrol_outputs.canopy2ground,
    }
    _require_fields(source, SECHIBA_TO_OK_LEAK_HYDROL_FIELDS, "SECHIBA-to-OK_LEAK HYDROL boundary")
    return source


def _merge_disjoint_sources(*sources: Mapping[str, object]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for source in sources:
        overlap = set(merged).intersection(source)
        if overlap:
            raise ValueError(f"duplicate explicit source fields: {tuple(sorted(overlap))}")
        merged.update(source)
    return merged


def stomate_ok_leak_boundary_inputs(
    *,
    restart_state: StomateRestartOkLeakStateInputs,
    pft_static: StomatePftStaticOkLeakInputs | None = None,
    static_routing: StomateStaticRoutingOkLeakInputs | None = None,
    vertical: StomateVerticalOkLeakInputs | None = None,
    litter_controls: StomateLitterControlsOkLeakInputs | None = None,
    tf_doc: StomateTfDocOkLeakInputs | None = None,
    perma_peat: StomatePermaPeatOkLeakInputs | None = None,
    active_layer: StomateActiveLayerOkLeakInputs | None = None,
    soilwater: StomateSoilwaterOkLeakInputs | None = None,
    permafrost_activity: StomatePermafrostActivityOkLeakInputs | None = None,
    soil_mc_32l: StomateSoilMc32lOkLeakInputs | None = None,
    ld_doc_routing: SechibaLdDocRoutingOkLeakInputs | None = None,
    doc_transport: StomateDocTransportOkLeakInputs | None = None,
    extra_ok_leak_inputs: Mapping[str, object] | None = None,
    extra_output_inputs: Mapping[str, object] | None = None,
) -> StomateOkLeakBoundaryInputs:
    """Merge source-backed OK_LEAK boundary bundles without duplicate fields.

    This is a composition helper only. It preserves the provenance of each
    source bundle and refuses overlapping fields so callers cannot silently mix
    restart, driver/static, vertical-grid, and process inputs for the same
    Fortran argument.
    """

    ok_sources: list[Mapping[str, object]] = [restart_state.ok_leak_state_inputs]
    output_sources: list[Mapping[str, object]] = [restart_state.output_inputs]
    provenance = list(restart_state.provenance)
    if pft_static is not None:
        ok_sources.append(pft_static.ok_leak_inputs)
        provenance.extend(pft_static.provenance)
    if static_routing is not None:
        ok_sources.append(static_routing.ok_leak_inputs)
        output_sources.append(static_routing.output_inputs)
        provenance.extend(static_routing.provenance)
    if vertical is not None:
        ok_sources.append(vertical.ok_leak_inputs)
        output_sources.append(vertical.output_inputs)
        provenance.extend(vertical.provenance)
    if litter_controls is not None:
        ok_sources.append(litter_controls.ok_leak_inputs)
        provenance.extend(litter_controls.provenance)
    if tf_doc is not None:
        for index, source in enumerate(ok_sources):
            if "interception_storage" in source:
                replacement = dict(source)
                replacement.pop("interception_storage")
                ok_sources[index] = replacement
                break
        ok_sources.append(tf_doc.ok_leak_inputs)
        provenance.extend(tf_doc.provenance)
    if perma_peat is not None:
        ok_sources.append(perma_peat.ok_leak_inputs)
        provenance.extend(perma_peat.provenance)
    if active_layer is not None:
        ok_sources.append(active_layer.ok_leak_inputs)
        output_sources.append(active_layer.output_inputs)
        provenance.extend(active_layer.provenance)
    if soilwater is not None:
        ok_sources.append(soilwater.ok_leak_inputs)
        provenance.extend(soilwater.provenance)
    if permafrost_activity is not None:
        ok_sources.append(permafrost_activity.ok_leak_inputs)
        output_sources.append(permafrost_activity.output_inputs)
        provenance.extend(permafrost_activity.provenance)
    if soil_mc_32l is not None:
        ok_sources.append(soil_mc_32l.ok_leak_inputs)
        provenance.extend(soil_mc_32l.provenance)
    if ld_doc_routing is not None:
        for index, source in enumerate(ok_sources):
            replacement = dict(source)
            removed = False
            for name in ("doc_to_topsoil", "doc_to_subsoil"):
                if name in replacement:
                    replacement.pop(name)
                    removed = True
            if removed:
                ok_sources[index] = replacement
        ok_sources.append(ld_doc_routing.ok_leak_inputs)
        provenance.extend(ld_doc_routing.provenance)
    if doc_transport is not None:
        ok_sources.append(doc_transport.ok_leak_inputs)
        provenance.extend(doc_transport.provenance)
    if extra_ok_leak_inputs is not None:
        ok_sources.append(extra_ok_leak_inputs)
    if extra_output_inputs is not None:
        output_sources.append(extra_output_inputs)
    return StomateOkLeakBoundaryInputs(
        ok_leak_inputs=_merge_disjoint_sources(*ok_sources),
        output_inputs=_merge_disjoint_sources(*output_sources),
        provenance=tuple(dict.fromkeys(provenance)),
    )


def stomate_pft_static_ok_leak_inputs(
    *,
    run_scalars: RunScalars | None = None,
    pref_soil_veg=None,
    natural=None,
    is_peat=None,
    is_c4=None,
) -> StomatePftStaticOkLeakInputs:
    """Build static PFT OK_LEAK inputs from the Fortran PFT parameter path.

    ``pref_soil_veg`` is stored as Fortran one-based tile indices in
    ``pft_parameters`` and converted here to the zero-based convention used by
    the JAX soilcarbon kernels. The boolean PFT flags are passed through from
    the same PFT/MTC mapping and optional run.def override path.
    """

    if run_scalars is not None:
        if any(value is not None for value in (pref_soil_veg, natural, is_peat, is_c4)):
            raise ValueError("run_scalars cannot be combined with explicit PFT static arrays")
        pref_source = run_scalars.pref_soil_veg
        natural_source = run_scalars.natural
        is_peat_source = run_scalars.is_peat
        is_c4_source = run_scalars.is_c4
    else:
        missing = tuple(
            name
            for name, value in {
                "pref_soil_veg": pref_soil_veg,
                "natural": natural,
                "is_peat": is_peat,
                "is_c4": is_c4,
            }.items()
            if value is None
        )
        if missing:
            raise ValueError(f"PFT static boundary requires {missing}")
        pref_source = pref_soil_veg
        natural_source = natural
        is_peat_source = is_peat
        is_c4_source = is_c4

    pref = np.asarray(pref_source, dtype=np.int32)
    natural_arr = np.asarray(natural_source, dtype=bool)
    is_peat_arr = np.asarray(is_peat_source, dtype=bool)
    is_c4_arr = np.asarray(is_c4_source, dtype=bool)
    if pref.ndim != 1 or pref.size == 0:
        raise ValueError("pref_soil_veg must be a non-empty 1D Fortran one-based vector")
    nvm = pref.shape[0]
    if natural_arr.shape != (nvm,) or is_peat_arr.shape != (nvm,) or is_c4_arr.shape != (nvm,):
        raise ValueError("natural, is_peat, and is_c4 must have shape (nvm,)")
    if np.any(pref < 1):
        raise ValueError("pref_soil_veg must use Fortran one-based soil tile indices")
    return StomatePftStaticOkLeakInputs(
        ok_leak_inputs={
            "pref_soil_veg": pref - 1,
            "natural": natural_arr,
            "is_peat": is_peat_arr,
            "is_c4": is_c4_arr,
        },
    )


def stomate_soilwater_31mm_ok_leak_inputs(
    *,
    soil_mc,
    z_soil,
    sro_bottom,
) -> StomateSoilwaterOkLeakInputs:
    """Compute ``soilwater_31mm`` exactly as ``soilcarbon_leak`` does.

    Fortran first zeros ``soilwater_31mm`` and then sums CWRR soil moisture
    over layers ``1:sro_bottom`` weighted by layer thickness
    ``z_soil(l)-z_soil(l-1)``. ``z_soil`` must include the Fortran zero upper
    boundary at Python index 0.
    """

    soil = jnp.asarray(soil_mc, dtype=jnp.float64)
    z = jnp.asarray(z_soil, dtype=jnp.float64)
    bottom = int(sro_bottom)
    if soil.ndim != 3:
        raise ValueError("soil_mc must have shape (npts, nslm, nstm)")
    if z.ndim != 1:
        raise ValueError("z_soil must be a 1D vector including the zero upper boundary")
    if bottom < 1:
        raise ValueError("sro_bottom must be at least 1")
    if bottom > soil.shape[1]:
        raise ValueError("sro_bottom cannot exceed the soil_mc layer dimension")
    if z.size <= bottom:
        raise ValueError("z_soil must include entries through sro_bottom")
    layer_thickness = jnp.diff(z[: bottom + 1])
    if isinstance(z_soil, np.ndarray) and np.any(np.diff(z_soil[: bottom + 1]) < 0.0):
        raise ValueError("z_soil depths must be nondecreasing through sro_bottom")
    soilwater_31mm = jnp.sum(
        soil[:, :bottom, :] * layer_thickness[None, :, None],
        axis=1,
    )
    return StomateSoilwaterOkLeakInputs(
        ok_leak_inputs={
            "soilwater_31mm": soilwater_31mm,
        },
    )


def stomate_permafrost_activity_ok_leak_inputs(
    *,
    tdeep,
    hsdeep,
    zz_deep,
    mc_peat,
    poor_soils,
    frozen_respiration_func,
    perma_peat: bool,
    agri_peat: bool = False,
    is_peat=None,
    tau_peat=3.1536e8,
    z_tau=1.0e6,
    flux_tot_coeff=(1.2, 1.4, 0.75),
    one_day=86400.0,
) -> StomatePermafrostActivityOkLeakInputs:
    """Compute OK_LEAK activity controls from ``stomate_main`` microactem calls.

    ``stomate_main`` converts deep temperatures to Celsius, calls
    ``microactem`` twice, converts the returned seconds to day-rate controls,
    applies the poor-soils factor, and then passes the same ``fbact`` to
    ``littercalc_leak`` and ``soilcarbon_leak`` while passing ``fbact_doc`` to
    ``soilcarbon_leak``.
    """

    tdeep_arr = jnp.asarray(tdeep, dtype=jnp.float64)
    hsdeep_arr = jnp.asarray(hsdeep, dtype=jnp.float64)
    zz_arr = jnp.asarray(zz_deep, dtype=jnp.float64)
    mc_peat_arr = jnp.asarray(mc_peat, dtype=jnp.float64)
    poor_arr = jnp.asarray(poor_soils, dtype=jnp.float64)
    if tdeep_arr.ndim != 3:
        raise ValueError("tdeep must have shape (npts, ndeep, nvm)")
    if hsdeep_arr.shape != tdeep_arr.shape:
        raise ValueError("hsdeep must match tdeep shape")
    npts, ndeep, nvm = tdeep_arr.shape
    if zz_arr.shape != (ndeep,):
        raise ValueError("zz_deep must have shape (ndeep,)")
    if mc_peat_arr.shape != (npts, ndeep):
        raise ValueError("mc_peat must have shape (npts, ndeep)")
    if poor_arr.shape != (npts,):
        raise ValueError("poor_soils must have shape (npts,)")
    if perma_peat:
        peat_mask = jnp.asarray(is_peat, dtype=bool) if is_peat is not None else None
        if peat_mask is None or peat_mask.shape != (nvm,):
            raise ValueError("is_peat must have shape (nvm,) when perma_peat is true")

    controls = stomate_permafrost_decomposition_controls(
        tdeep_arr - 273.15,
        hsdeep_arr,
        zz_arr,
        mc_peat_arr,
        poor_arr,
        frozen_respiration_func=int(frozen_respiration_func),
        perma_peat=bool(perma_peat),
        agri_peat=bool(agri_peat),
        is_peat=is_peat,
        tau_peat=tau_peat,
        z_tau=z_tau,
        flux_tot_coeff=flux_tot_coeff,
        one_day=one_day,
    )
    fbact = controls.prmfrst_soilc_tempctrl
    fbact_doc = controls.prmfrst_soilc_tempctrl_doc
    return StomatePermafrostActivityOkLeakInputs(
        ok_leak_inputs={
            "fbact_litter": fbact,
            "fbact_soilcarbon": fbact,
            "fbact_doc": fbact_doc,
            "tprof": tdeep_arr,
        },
        output_inputs={
            "tprof": tdeep_arr,
        },
    )


def stomate_mc_peat_boundary_inputs(
    *,
    shumdiag_peat,
    ndeep,
    nslm,
) -> StomateMcPeatBoundaryInputs:
    """Build ``mc_peat`` from HYDROL ``shumdiag_peat`` as ``stomate_main`` does.

    The source sorts a copy but assigns from the original ``shumdiag_peat``:
    layers ``1:nslm`` copy directly and deeper layers repeat layer ``nslm``.
    """

    shum = jnp.asarray(shumdiag_peat, dtype=jnp.float64)
    deep = int(ndeep)
    soil_layers = int(nslm)
    if shum.ndim != 2:
        raise ValueError("shumdiag_peat must have shape (npts, nslm)")
    if soil_layers < 1:
        raise ValueError("nslm must be at least 1")
    if shum.shape[1] != soil_layers:
        raise ValueError("shumdiag_peat layer axis must match nslm")
    if deep < soil_layers:
        raise ValueError("ndeep must be greater than or equal to nslm")
    if deep == soil_layers:
        mc_peat = shum
    else:
        repeated = jnp.repeat(shum[:, -1:], deep - soil_layers, axis=1)
        mc_peat = jnp.concatenate((shum, repeated), axis=1)
    return StomateMcPeatBoundaryInputs(mc_peat=mc_peat)


def stomate_doc_transport_ok_leak_inputs(
    *,
    poor_soils,
    flux_red_sro=0.2,
) -> StomateDocTransportOkLeakInputs:
    """Compute DOC runoff/transport reduction factor from poor-soils state."""

    poor = jnp.asarray(poor_soils, dtype=jnp.float64)
    if poor.ndim != 1:
        raise ValueError("poor_soils must have shape (npts,)")
    flux_red = jnp.asarray(flux_red_sro, dtype=jnp.float64) * (1.0 - poor) + poor
    return StomateDocTransportOkLeakInputs(
        ok_leak_inputs={
            "flux_red": flux_red,
        },
    )


def sechiba_ld_doc_routing_ok_leak_inputs(
    *,
    reinfiltration,
    irrigation,
    returnflow,
    river_routing: bool,
    dt_sechiba,
    one_day=86400.0,
) -> SechibaLdDocRoutingOkLeakInputs:
    """Map SECHIBA long-distance DOC routing terms to STOMATE inputs.

    Fortran provenance: ``src_sechiba/sechiba.f90`` ``sechiba_main`` lines
    1251-1259. The source computes daily DOC inputs from routing fluxes only
    when ``river_routing`` is true; otherwise both STOMATE-bound arrays are
    explicitly zeroed.
    """

    reinf = np.asarray(reinfiltration, dtype=np.float64)
    irrig = np.asarray(irrigation, dtype=np.float64)
    ret = np.asarray(returnflow, dtype=np.float64)
    if reinf.ndim != 2 or irrig.shape != reinf.shape or ret.shape != reinf.shape:
        raise ValueError("reinfiltration, irrigation, and returnflow must share shape (npts, nflow)")
    scale = np.asarray(1000.0 * one_day, dtype=np.float64) / np.asarray(dt_sechiba, dtype=np.float64)
    if bool(river_routing):
        doc_to_topsoil = (reinf + irrig) * scale
        doc_to_subsoil = ret * scale
    else:
        doc_to_topsoil = np.zeros_like(reinf)
        doc_to_subsoil = np.zeros_like(ret)
    entry_source = {
        "DOC_to_topsoil": doc_to_topsoil,
        "DOC_to_subsoil": doc_to_subsoil,
    }
    return SechibaLdDocRoutingOkLeakInputs(
        ok_leak_inputs={
            "doc_to_topsoil": doc_to_topsoil,
            "doc_to_subsoil": doc_to_subsoil,
        },
        entry_source=entry_source,
    )


def stomate_soil_mc_32l_ok_leak_inputs(
    *,
    soil_mc,
    ndeep,
    nslm,
) -> StomateSoilMc32lOkLeakInputs:
    """Compute ``soil_mc_32l`` from HYDROL soil moisture in Fortran order."""

    return StomateSoilMc32lOkLeakInputs(
        ok_leak_inputs={
            "soil_mc_32l": stomate_soil_mc_32l(soil_mc, ndeep=int(ndeep), nslm=int(nslm)),
        },
    )


def stomate_litter_controls_ok_leak_inputs(
    *,
    tsurf,
    soil_mc,
    z_soil,
    pref_soil_veg,
    frozen_respiration_func,
    moist_func_moyano: bool = False,
    zz_coef_deep=None,
    bulk_dens=None,
    clay=None,
    carbon_32l=None,
    veget_max=None,
) -> StomateLitterControlsOkLeakInputs:
    """Compute aboveground litter controls for the OK_LEAK boundary.

    Inputs are source-backed boundary values already available in the current
    composition: ``tsurf``/``temp_sol``, HYDROL ``soil_mc``, vertical
    ``z_soil``, and Fortran one-based ``pref_soil_veg`` from driver/static
    initialization. The helper converts the soil-tile map to the zero-based
    convention expected by the audited STOMATE kernel.
    """

    pref = np.asarray(pref_soil_veg, dtype=np.int32)
    if pref.ndim != 1 or pref.size == 0:
        raise ValueError("pref_soil_veg must be a non-empty 1D Fortran one-based vector")
    if np.any(pref < 1):
        raise ValueError("pref_soil_veg must use Fortran one-based soil tile indices")
    controls = littercalc_aboveground_controls(
        tsurf=tsurf,
        soil_mc=soil_mc,
        z_soil=z_soil,
        pref_soil_veg=pref - 1,
        frozen_respiration_func=int(frozen_respiration_func),
        moist_func_moyano=moist_func_moyano,
        zz_coef_deep=zz_coef_deep,
        bulk_dens=bulk_dens,
        clay=clay,
        carbon_32l=carbon_32l,
        veget_max=veget_max,
    )
    return StomateLitterControlsOkLeakInputs(
        ok_leak_inputs={
            "control_temp_above": controls.control_temp_above,
            "control_moist_above": controls.control_moist_above,
            "soil_mc_top_by_pft": controls.soil_mc_top_by_pft,
        },
    )


def stomate_tf_doc_ok_leak_inputs(
    *,
    precip2ground,
    precip2canopy,
    biomass,
    veget_max,
    is_tree,
    interception_storage,
    ok_tf_doc: bool,
    dt_days,
    conc_doc_rain=3.02,
    doc_incr_per_leaf_m2=0.00092,
) -> StomateTfDocOkLeakInputs:
    """Compute source-backed TF-DOC inputs for ``soilcarbon_leak``.

    HYDROL supplies ``precip2ground`` and ``precip2canopy``; restart supplies
    ``interception_storage`` for carbon; STOMATE carbon state supplies
    ``biomass`` and slowproc vegetation supplies ``veget_max``. The helper
    mirrors the Fortran branch for ``OK_TF_DOC`` and preserves restart canopy
    storage instead of replacing it with a synthetic zero.
    """

    storage = jnp.asarray(interception_storage, dtype=jnp.float64)
    if storage.ndim != 3:
        raise ValueError("interception_storage must have shape (npts, nvm, nelements)")
    tf_doc = soilcarbon_leak_tf_doc_inputs(
        precip2ground=precip2ground,
        precip2canopy=precip2canopy,
        veget_max=veget_max,
        biomass=biomass,
        is_tree=is_tree,
        ok_tf_doc=ok_tf_doc,
        dt_days=dt_days,
        conc_doc_rain=conc_doc_rain,
        doc_incr_per_leaf_m2=doc_incr_per_leaf_m2,
    )
    dry_dep_canopy = tf_doc.dry_dep_canopy
    if storage.shape != dry_dep_canopy.shape:
        raise ValueError("interception_storage must match TF-DOC element shape")
    return StomateTfDocOkLeakInputs(
        ok_leak_inputs={
            "doc_precip2ground": tf_doc.doc_precip2ground,
            "doc_precip2canopy": tf_doc.doc_precip2canopy,
            "dry_dep_canopy": tf_doc.dry_dep_canopy,
            "interception_storage": storage,
        },
    )


def stomate_perma_peat_ok_leak_inputs(
    *,
    peat_bulk_density,
    zf_soil_b,
    npts: int,
    nvm: int,
    perma_peat: bool,
    frac1=None,
    frac2=None,
    min_stomate=None,
) -> StomatePermaPeatOkLeakInputs:
    """Compute source-backed PERMA_PEAT inputs for ``soilcarbon_leak``.

    The Fortran first-call block computes ``Cmax`` from peat bulk density and
    ``zf_soil_B``. The same source currently sets ``veget_mask_2d = .TRUE.``
    rather than deriving it from vegetation cover, so this helper mirrors that
    mask exactly for the PERMA_PEAT branch.
    """

    if npts <= 0 or nvm <= 0:
        raise ValueError("npts and nvm must be positive")
    zf = jnp.asarray(zf_soil_b, dtype=jnp.float64)
    peat_bd = jnp.asarray(peat_bulk_density, dtype=jnp.float64)
    if zf.ndim != 1 or peat_bd.ndim != 1:
        raise ValueError("peat_bulk_density and zf_soil_b must be 1D")
    cmax = soilcarbon_perma_peat_cmax(peat_bd, zf)
    ok_leak_inputs: dict[str, object] = {
        "perma_peat": bool(perma_peat),
        "cmax_peat": cmax,
        "perma_peat_veget_mask": jnp.ones((int(npts), int(nvm)), dtype=bool),
    }
    if frac1 is not None:
        ok_leak_inputs["frac1"] = frac1
    if frac2 is not None:
        ok_leak_inputs["frac2"] = frac2
    if min_stomate is not None:
        ok_leak_inputs["min_stomate"] = min_stomate
    return StomatePermaPeatOkLeakInputs(ok_leak_inputs=ok_leak_inputs)


def stomate_active_layer_ok_leak_inputs(
    *,
    tprof,
    zi_soil,
    restart_altmax,
    restart_fixed_cryoturbation_depth,
    dayno,
    firstcall_soilcarbon: bool,
    newaltcalc: bool = False,
    soilc_isspinup: bool = False,
    npts: int | None = None,
    nvm: int | None = None,
) -> StomateActiveLayerOkLeakInputs:
    """Compute active-layer state and cryoturbation boundary inputs.

    For a first soilcarbon call, Fortran initializes ``altmax_ind`` and
    ``altmax_lastyear`` from restart ``altmax`` before the ordinary per-step
    ``altcalc_DOC`` update. ``veget_mask_2d`` is currently initialized as all
    true in the source, so this helper mirrors that exact mask.
    """

    tprof_arr = jnp.asarray(tprof, dtype=jnp.float64)
    zi_arr = jnp.asarray(zi_soil, dtype=jnp.float64)
    altmax_arr = jnp.asarray(restart_altmax, dtype=jnp.float64)
    fixed_arr = jnp.asarray(restart_fixed_cryoturbation_depth, dtype=jnp.float64)
    if tprof_arr.ndim != 3:
        raise ValueError("tprof must have shape (npts, ndeep, nvm)")
    inferred_npts, ndeep, inferred_nvm = tprof_arr.shape
    if npts is not None and int(npts) != inferred_npts:
        raise ValueError("npts must match tprof")
    if nvm is not None and int(nvm) != inferred_nvm:
        raise ValueError("nvm must match tprof")
    if zi_arr.ndim != 1 or zi_arr.shape[0] != ndeep:
        raise ValueError("zi_soil must have length ndeep for altcalc_DOC")
    if altmax_arr.shape != (inferred_npts, inferred_nvm) or fixed_arr.shape != (inferred_npts, inferred_nvm):
        raise ValueError("restart altmax and fixed_cryoturbation_depth must have shape (npts, nvm)")

    veget_mask = jnp.ones((inferred_npts, inferred_nvm), dtype=bool)
    zero_ind = jnp.zeros((inferred_npts, inferred_nvm), dtype=jnp.int32)
    if firstcall_soilcarbon:
        initialized = altcalc_doc(
            tprof_arr,
            zi_arr,
            altmax_arr,
            zero_ind,
            jnp.zeros_like(altmax_arr),
            zero_ind,
            veget_mask,
            firstcall=True,
            newaltcalc=newaltcalc,
            dayno=dayno,
            soilc_isspinup=soilc_isspinup,
        )
        altmax = initialized.altmax
        altmax_ind = initialized.altmax_ind
        altmax_lastyear = initialized.altmax_lastyear
        altmax_ind_lastyear = initialized.altmax_ind_lastyear
    else:
        altmax = altmax_arr
        altmax_ind = zero_ind
        altmax_lastyear = altmax_arr
        altmax_ind_lastyear = zero_ind

    current = altcalc_doc(
        tprof_arr,
        zi_arr,
        altmax,
        altmax_ind,
        altmax_lastyear,
        altmax_ind_lastyear,
        veget_mask,
        firstcall=False,
        newaltcalc=newaltcalc,
        dayno=dayno,
        soilc_isspinup=soilc_isspinup,
    )
    return StomateActiveLayerOkLeakInputs(
        ok_leak_inputs={
            "altmax_ind": current.altmax_ind,
            "altmax_lastyear": current.altmax_lastyear,
            "fixed_cryoturbation_depth": fixed_arr,
            "veget_mask": veget_mask,
            "zi_soil": zi_arr,
        },
        output_inputs={
            "altmax": current.altmax,
        },
        state_updates={
            "alt": current.alt,
            "alt_ind": current.alt_ind,
            "altmax": current.altmax,
            "altmax_ind": current.altmax_ind,
            "altmax_lastyear": current.altmax_lastyear,
            "altmax_ind_lastyear": current.altmax_ind_lastyear,
        },
    )


def stomate_static_routing_ok_leak_inputs(
    *,
    driver_payload: IntersurfFirstStepPayload,
    kjit,
    nflow,
    river_routing,
    nbp_glo,
    t2mdiag=None,
    temp_sol=None,
    min_wind=0.1,
    **driver_entry_kwargs,
) -> StomateStaticRoutingOkLeakInputs:
    """Map driver/static and no-routing source truth to OK_LEAK inputs.

    Fortran provenance: ``sechiba_main`` lines 1135 and 1184-1216 pass
    driver/static fields to ``slowproc_main``; the no-routing branch in
    ``sechiba_initialize`` lines 749-769 and ``sechiba_main`` lines 1227-1259
    supplies zero DOC routing, ``flood_frac``, and ``fastr`` only when routing
    is inactive or single-point. This helper exposes the same source-backed
    fields under OK_LEAK/output argument names and refuses missing static
    values.
    """

    entry_source = stomate_driver_entry_source(
        driver_payload,
        kjit=kjit,
        t2mdiag=t2mdiag,
        temp_sol=temp_sol,
        min_wind=min_wind,
        **driver_entry_kwargs,
    )
    _require_fields(entry_source, ("clay", "bulk_dens", "soil_ph", "poor_soils"), "driver/static OK_LEAK boundary")
    routing_source = stomate_no_routing_entry_source(
        kjpindex=driver_payload.kjpindex,
        nflow=nflow,
        river_routing=river_routing,
        nbp_glo=nbp_glo,
    )
    ok_leak_inputs = {
        "clay": entry_source["clay"],
        "bulk_dens": entry_source["bulk_dens"],
        "poor_soils": entry_source["poor_soils"],
        "doc_to_topsoil": routing_source["DOC_to_topsoil"],
        "doc_to_subsoil": routing_source["DOC_to_subsoil"],
        "flood_frac": routing_source["flood_frac"],
        "fastr": routing_source["fastr"],
    }
    output_inputs = {
        "contfrac": entry_source["contfrac"],
    }
    return StomateStaticRoutingOkLeakInputs(
        ok_leak_inputs=ok_leak_inputs,
        output_inputs=output_inputs,
        entry_source={**entry_source, **routing_source},
    )


def stomate_vertical_ok_leak_inputs(
    *,
    diaglev,
    zz_coef_deep,
    zz_deep=None,
) -> StomateVerticalOkLeakInputs:
    """Map audited vertical arrays to OK_LEAK and output argument names.

    ``diaglev`` is the CWRR diagnostic soil-depth vector from
    ``control_initialize``; ``zz_coef_deep`` is the deep-carbon lower-interface
    vector passed through ``slowproc_main`` into STOMATE. This helper does not
    construct those arrays. It only adds the Fortran zero upper boundary used
    by litter/soilcarbon code and exposes the same source arrays under the
    argument names consumed by the current composition.
    """

    diaglev_arr = np.asarray(diaglev, dtype=np.float64)
    zz_coef_arr = np.asarray(zz_coef_deep, dtype=np.float64)
    zz_deep_arr = None if zz_deep is None else np.asarray(zz_deep, dtype=np.float64)
    if diaglev_arr.ndim != 1 or diaglev_arr.size == 0:
        raise ValueError("diaglev must be a non-empty 1D vector")
    if zz_coef_arr.ndim != 1 or zz_coef_arr.size == 0:
        raise ValueError("zz_coef_deep must be a non-empty 1D vector")
    if zz_deep_arr is not None and (zz_deep_arr.ndim != 1 or zz_deep_arr.size != zz_coef_arr.size):
        raise ValueError("zz_deep must be a 1D vector with the same length as zz_coef_deep")
    z_soil = np.concatenate((np.asarray([0.0], dtype=np.float64), diaglev_arr))
    zf_soil_b = np.concatenate((np.asarray([0.0], dtype=np.float64), zz_coef_arr))
    ok_leak_inputs = {
        "z_soil": z_soil,
        "zf_soil_b": zf_soil_b,
        "nslm": int(diaglev_arr.size),
        "ndeep": int(zz_coef_arr.size),
    }
    if zz_deep_arr is not None:
        ok_leak_inputs["zi_soil"] = zz_deep_arr
    return StomateVerticalOkLeakInputs(
        ok_leak_inputs=ok_leak_inputs,
        output_inputs={
            "z_soil": diaglev_arr,
            "zf_soil": zf_soil_b,
        },
    )


def stomate_pre_step_ok_leak_boundary_inputs(
    *,
    state: StomateRestartEntryState,
    driver_payload: IntersurfFirstStepPayload,
    run_scalars: RunScalars,
    diaglev,
    zz_coef_deep,
    kjit,
    nflow,
    river_routing,
    nbp_glo,
    z_soil_output=None,
    zf_soil_output=None,
    carb_mass_total_old=None,
    prod10_total=None,
    prod100_total=None,
    one_day=86400.0,
    zz_deep=None,
    t2mdiag=None,
    temp_sol=None,
    min_wind=0.1,
    doc_transport_flux_red_sro=0.2,
    **driver_entry_kwargs,
) -> StomatePreStepOkLeakBoundaryInputs:
    """Build the source-backed OK_LEAK/output boundary known before a step.

    This composes only inputs available from restart, driver/static routing,
    PFT parameter initialization, and audited vertical grids. It deliberately
    does not compute same-step HYDROL/THERMOSOIL-derived fields such as
    ``soilwater_31mm``, ``soil_mc_32l``, ``fbact_*``, or litter controls; those
    must remain in the ordered SECHIBA -> STOMATE composition.
    """

    restart = stomate_restart_ok_leak_state_inputs(
        state=state,
        veget_max=driver_payload.veget_max,
        z_soil=z_soil_output,
        zf_soil=zf_soil_output,
        carb_mass_total_old=carb_mass_total_old,
        prod10_total=prod10_total,
        prod100_total=prod100_total,
        one_day=one_day,
    )
    pft_static = stomate_pft_static_ok_leak_inputs(run_scalars=run_scalars)
    static_routing = stomate_static_routing_ok_leak_inputs(
        driver_payload=driver_payload,
        kjit=kjit,
        nflow=nflow,
        river_routing=river_routing,
        nbp_glo=nbp_glo,
        t2mdiag=t2mdiag,
        temp_sol=temp_sol,
        min_wind=min_wind,
        **driver_entry_kwargs,
    )
    vertical = stomate_vertical_ok_leak_inputs(
        diaglev=diaglev,
        zz_coef_deep=zz_coef_deep,
        zz_deep=zz_deep,
    )
    doc_transport = None
    if driver_payload.poor_soils is not None:
        doc_transport = stomate_doc_transport_ok_leak_inputs(
            poor_soils=driver_payload.poor_soils,
            flux_red_sro=doc_transport_flux_red_sro,
        )
    boundary = stomate_ok_leak_boundary_inputs(
        restart_state=restart,
        pft_static=pft_static,
        static_routing=static_routing,
        vertical=vertical,
        doc_transport=doc_transport,
    )
    provenance = tuple(
        dict.fromkeys(
            (
                *restart.provenance,
                *pft_static.provenance,
                *static_routing.provenance,
                *vertical.provenance,
                *((doc_transport.provenance) if doc_transport is not None else ()),
            )
        )
    )
    return StomatePreStepOkLeakBoundaryInputs(
        boundary_inputs=boundary,
        restart_state=restart,
        pft_static=pft_static,
        static_routing=static_routing,
        vertical=vertical,
        doc_transport=doc_transport,
        provenance=provenance,
    )


def stomate_same_step_ok_leak_boundary_inputs(
    *,
    sechiba: SechibaExplicitCoupledStepResult,
    post_npp,
    dt_sechiba,
    entry_payload: Mapping[str, object] | None = None,
    pre_step_boundary: StomateOkLeakBoundaryInputs | None = None,
    ok_leak_inputs: Mapping[str, object] | None = None,
    pft_static_boundary: Mapping[str, object] | None = None,
    litter_controls_boundary: Mapping[str, object] | None = None,
    tf_doc_boundary: Mapping[str, object] | None = None,
    perma_peat_boundary: Mapping[str, object] | None = None,
    active_layer_boundary: Mapping[str, object] | None = None,
    soilwater_boundary: Mapping[str, object] | None = None,
    permafrost_activity_boundary: Mapping[str, object] | None = None,
    soil_mc_32l_boundary: Mapping[str, object] | None = None,
    doc_transport_boundary: Mapping[str, object] | None = None,
    is_tree=None,
    one_day=86400.0,
) -> StomateSameStepOkLeakBoundaryInputs:
    """Build same-step OK_LEAK boundary values from SECHIBA/STOMATE outputs.

    This is the ordered post-SECHIBA counterpart to
    ``stomate_pre_step_ok_leak_boundary_inputs``. It derives only quantities
    whose Fortran sources are already represented by local helpers:
    HYDROL water fields, THERMOSOIL deep profiles, STOMATE post-NPP biomass,
    litter controls, TF-DOC, active-layer state, PERMA_PEAT limits, shallow
    soilwater, ``soil_mc_32l``, and DOC transport factors. Branch controls
    remain explicit keyword dictionaries so inactive or unsupported branches
    cannot be hidden by guessed defaults.
    """

    base_ok_inputs = {}
    base_output_inputs = {}
    base_provenance: tuple[str, ...] = ()
    if pre_step_boundary is not None:
        base_ok_inputs = dict(pre_step_boundary.ok_leak_inputs)
        base_output_inputs = dict(pre_step_boundary.output_inputs)
        base_provenance = pre_step_boundary.provenance
    if ok_leak_inputs is not None:
        base_ok_inputs = _merge_disjoint_sources(base_ok_inputs, dict(ok_leak_inputs))
    entry_source = {} if entry_payload is None else dict(entry_payload)

    ok_leak_hydrol_source = _ok_leak_hydrol_source_from_sechiba(sechiba)
    npts = sechiba.hydrol_outputs.wat_flux.shape[0]
    if is_tree is None:
        is_tree = base_ok_inputs.get("is_tree")
    if is_tree is None:
        raise ValueError("same-step OK_LEAK boundary requires explicit is_tree or pre-step/static source")
    nvm = is_tree.shape[0]

    pft_static = None
    pft_static_source = None
    if pft_static_boundary is not None:
        pft_static_source = dict(pft_static_boundary)
        pft_static = stomate_pft_static_ok_leak_inputs(**pft_static_source)

    litter_controls = None
    if litter_controls_boundary is not None:
        boundary = dict(litter_controls_boundary)
        if "frozen_respiration_func" not in boundary:
            raise ValueError("litter controls boundary requires frozen_respiration_func")
        z_source = boundary.pop("z_soil", base_ok_inputs.get("z_soil"))
        if z_source is None:
            raise ValueError("litter controls boundary requires z_soil from OK_LEAK vertical inputs")
        pref_source = boundary.pop("pref_soil_veg", None)
        if pref_source is None and pft_static_source is not None:
            if "run_scalars" in pft_static_source:
                pref_source = pft_static_source["run_scalars"].pref_soil_veg
            else:
                pref_source = pft_static_source.get("pref_soil_veg")
        if pref_source is None:
            pref_source = base_ok_inputs.get("pref_soil_veg")
        if pref_source is None:
            raise ValueError("litter controls boundary requires Fortran one-based pref_soil_veg or pre-step/static source")
        tsurf_source = boundary.pop("tsurf", base_ok_inputs.get("tsurf"))
        if tsurf_source is None:
            tsurf_source = entry_source.get("temp_sol")
        if tsurf_source is None:
            tsurf_source = sechiba.diffuco_payload.get("temp_sol", sechiba.enerbil_payload.get("temp_sol"))
        if tsurf_source is None:
            raise ValueError("litter controls boundary requires tsurf or SECHIBA temp_sol")
        litter_controls = stomate_litter_controls_ok_leak_inputs(
            tsurf=tsurf_source,
            soil_mc=ok_leak_hydrol_source["soil_mc"],
            z_soil=z_source,
            pref_soil_veg=pref_source,
            frozen_respiration_func=boundary.pop("frozen_respiration_func"),
            zz_coef_deep=boundary.pop("zz_coef_deep", base_ok_inputs.get("zf_soil_b")),
            bulk_dens=boundary.pop("bulk_dens", base_ok_inputs.get("bulk_dens")),
            clay=boundary.pop("clay", base_ok_inputs.get("clay")),
            carbon_32l=boundary.pop("carbon_32l", base_ok_inputs.get("carbon_32l")),
            veget_max=boundary.pop("veget_max", sechiba.slowproc.vegetation.veget_max if sechiba.slowproc is not None else None),
            **boundary,
        )

    tf_doc = None
    if tf_doc_boundary is not None:
        if sechiba.slowproc is None:
            raise ValueError("SECHIBA result must include slowproc output for TF-DOC veget_max")
        boundary = dict(tf_doc_boundary)
        storage = boundary.pop("interception_storage", base_ok_inputs.get("interception_storage"))
        ok_tf_doc = boundary.pop("ok_tf_doc", None)
        if storage is None or ok_tf_doc is None:
            missing = tuple(name for name, value in {"interception_storage": storage, "ok_tf_doc": ok_tf_doc}.items() if value is None)
            raise ValueError(f"TF-DOC boundary requires {missing}")
        dt_days = boundary.pop("dt_days", np.asarray(dt_sechiba) / np.asarray(one_day))
        tf_doc = stomate_tf_doc_ok_leak_inputs(
            precip2ground=sechiba.hydrol_outputs.precip2ground,
            precip2canopy=sechiba.hydrol_outputs.precip2canopy,
            biomass=post_npp.turnover.biomass,
            veget_max=sechiba.slowproc.vegetation.veget_max,
            is_tree=is_tree,
            interception_storage=storage,
            ok_tf_doc=bool(ok_tf_doc),
            dt_days=dt_days,
            **boundary,
        )

    perma_peat = None
    if perma_peat_boundary is not None:
        boundary = dict(perma_peat_boundary)
        missing = tuple(name for name in ("peat_bulk_density", "perma_peat") if name not in boundary)
        if missing:
            raise ValueError(f"PERMA_PEAT boundary requires {missing}")
        if "zf_soil_b" not in base_ok_inputs:
            raise ValueError("PERMA_PEAT boundary requires zf_soil_b from OK_LEAK vertical inputs")
        perma_peat = stomate_perma_peat_ok_leak_inputs(
            peat_bulk_density=boundary.pop("peat_bulk_density"),
            zf_soil_b=base_ok_inputs["zf_soil_b"],
            npts=npts,
            nvm=nvm,
            perma_peat=bool(boundary.pop("perma_peat")),
            **boundary,
        )

    soilwater = None
    if soilwater_boundary is not None:
        boundary = dict(soilwater_boundary)
        z_source = boundary.pop("z_soil", base_ok_inputs.get("z_soil"))
        sro_source = boundary.pop("sro_bottom", base_ok_inputs.get("sro_bottom"))
        if z_source is None or sro_source is None:
            raise ValueError("soilwater boundary requires z_soil and sro_bottom from OK_LEAK inputs")
        if boundary:
            raise ValueError(f"unsupported soilwater boundary fields: {tuple(sorted(boundary))}")
        soilwater = stomate_soilwater_31mm_ok_leak_inputs(
            soil_mc=ok_leak_hydrol_source["soil_mc"],
            z_soil=z_source,
            sro_bottom=sro_source,
        )

    permafrost_activity = None
    if permafrost_activity_boundary is not None:
        boundary = dict(permafrost_activity_boundary)
        missing = tuple(name for name in ("frozen_respiration_func", "perma_peat") if name not in boundary)
        if missing:
            raise ValueError(f"permafrost activity boundary requires {missing}")
        perma_peat_flag = bool(boundary.pop("perma_peat"))
        tdeep_source = boundary.pop("tdeep", sechiba.thermosoil_payload.get("deeptemp_prof"))
        hsdeep_source = boundary.pop("hsdeep", sechiba.thermosoil_payload.get("deephum_prof"))
        zz_source = boundary.pop("zz_deep", base_ok_inputs.get("zi_soil", base_ok_inputs.get("zz_deep")))
        poor_source = boundary.pop("poor_soils", base_ok_inputs.get("poor_soils"))
        mc_peat_source = boundary.pop("mc_peat", None)
        if tdeep_source is None or hsdeep_source is None:
            raise ValueError("permafrost activity boundary requires tdeep/hsdeep or SECHIBA thermosoil deeptemp_prof/deephum_prof")
        if zz_source is None:
            raise ValueError("permafrost activity boundary requires zz_deep or zi_soil from vertical inputs")
        if poor_source is None:
            raise ValueError("permafrost activity boundary requires poor_soils from static inputs")
        if mc_peat_source is None:
            if not perma_peat_flag:
                raise ValueError("permafrost activity boundary requires mc_peat when perma_peat is false")
            nslm_source = base_ok_inputs.get("nslm")
            ndeep_source = base_ok_inputs.get("ndeep")
            if nslm_source is None or ndeep_source is None:
                raise ValueError("permafrost activity mc_peat derivation requires nslm and ndeep from OK_LEAK inputs")
            mc_peat_source = stomate_mc_peat_boundary_inputs(
                shumdiag_peat=sechiba.hydrol_diagnostics.shumdiag_peat,
                nslm=nslm_source,
                ndeep=ndeep_source,
            ).mc_peat
        permafrost_activity = stomate_permafrost_activity_ok_leak_inputs(
            tdeep=tdeep_source,
            hsdeep=hsdeep_source,
            zz_deep=zz_source,
            mc_peat=mc_peat_source,
            poor_soils=poor_source,
            frozen_respiration_func=boundary.pop("frozen_respiration_func"),
            perma_peat=perma_peat_flag,
            **boundary,
        )

    active_layer = None
    active_layer_source = dict(base_ok_inputs)
    if permafrost_activity is not None:
        active_layer_source = _merge_disjoint_sources(active_layer_source, permafrost_activity.ok_leak_inputs)
    if active_layer_boundary is not None:
        boundary = dict(active_layer_boundary)
        missing = tuple(
            name
            for name in ("restart_altmax", "restart_fixed_cryoturbation_depth", "dayno", "firstcall_soilcarbon")
            if name not in boundary
        )
        if missing:
            raise ValueError(f"active-layer boundary requires {missing}")
        if "tprof" not in active_layer_source:
            raise ValueError("active-layer boundary requires tprof from OK_LEAK inputs")
        zi_source = boundary.pop("zi_soil", active_layer_source.get("zi_soil"))
        if zi_source is None:
            raise ValueError("active-layer boundary requires zi_soil from vertical inputs")
        active_layer = stomate_active_layer_ok_leak_inputs(
            tprof=active_layer_source["tprof"],
            zi_soil=zi_source,
            restart_altmax=boundary.pop("restart_altmax"),
            restart_fixed_cryoturbation_depth=boundary.pop("restart_fixed_cryoturbation_depth"),
            dayno=boundary.pop("dayno"),
            firstcall_soilcarbon=bool(boundary.pop("firstcall_soilcarbon")),
            npts=npts,
            nvm=nvm,
            **boundary,
        )

    soil_mc_32l = None
    if soil_mc_32l_boundary is not None:
        boundary = dict(soil_mc_32l_boundary)
        nslm_source = boundary.pop("nslm", base_ok_inputs.get("nslm"))
        ndeep_source = boundary.pop("ndeep", base_ok_inputs.get("ndeep"))
        if nslm_source is None or ndeep_source is None:
            raise ValueError("soil_mc_32l boundary requires nslm and ndeep from OK_LEAK inputs")
        if boundary:
            raise ValueError(f"unsupported soil_mc_32l boundary fields: {tuple(sorted(boundary))}")
        soil_mc_32l = stomate_soil_mc_32l_ok_leak_inputs(
            soil_mc=ok_leak_hydrol_source["soil_mc"],
            nslm=nslm_source,
            ndeep=ndeep_source,
        )

    doc_transport = None
    if doc_transport_boundary is not None:
        boundary = dict(doc_transport_boundary)
        poor_source = boundary.pop("poor_soils", base_ok_inputs.get("poor_soils"))
        if poor_source is None:
            raise ValueError("DOC transport boundary requires poor_soils from static inputs")
        doc_transport = stomate_doc_transport_ok_leak_inputs(
            poor_soils=poor_source,
            **boundary,
        )

    def boundary_values_match(left, right) -> bool:
        if isinstance(left, jax.core.Tracer) or isinstance(right, jax.core.Tracer):
            return True
        return bool(np.allclose(np.asarray(left), np.asarray(right)))

    final_base_ok_inputs = dict(base_ok_inputs)
    for name, value in ok_leak_hydrol_source.items():
        if name in final_base_ok_inputs:
            if not boundary_values_match(final_base_ok_inputs[name], value):
                raise ValueError(f"{name} from same-step HYDROL boundary does not match pre-merged OK_LEAK input")
            final_base_ok_inputs.pop(name)

    ok_sources: list[Mapping[str, object]] = [ok_leak_hydrol_source]
    if pft_static is not None:
        ok_sources.append(pft_static.ok_leak_inputs)
    for bundle in (
        litter_controls,
        tf_doc,
        perma_peat,
        active_layer,
        soilwater,
        permafrost_activity,
        soil_mc_32l,
        doc_transport,
    ):
        if bundle is None:
            continue
        ok_inputs = dict(bundle.ok_leak_inputs)
        for duplicate_allowed in ("tprof", "zi_soil", "interception_storage"):
            if duplicate_allowed in ok_inputs and duplicate_allowed in final_base_ok_inputs:
                if not boundary_values_match(ok_inputs[duplicate_allowed], final_base_ok_inputs[duplicate_allowed]):
                    raise ValueError(f"{duplicate_allowed} from same-step boundary does not match pre-step OK_LEAK input")
                ok_inputs.pop(duplicate_allowed)
        ok_sources.append(ok_inputs)
    ok_sources.append(final_base_ok_inputs)

    output_sources: list[Mapping[str, object]] = [base_output_inputs]
    if active_layer is not None:
        output_sources.append(active_layer.output_inputs)
    if permafrost_activity is not None:
        activity_output = dict(permafrost_activity.output_inputs)
        if "tprof" in activity_output and "tprof" in base_output_inputs:
            if not boundary_values_match(activity_output["tprof"], base_output_inputs["tprof"]):
                raise ValueError("tprof from same-step output boundary does not match pre-step output input")
            activity_output.pop("tprof")
        output_sources.append(activity_output)

    boundary_inputs = StomateOkLeakBoundaryInputs(
        ok_leak_inputs=_merge_disjoint_sources(*ok_sources),
        output_inputs=_merge_disjoint_sources(*output_sources),
        provenance=tuple(
            dict.fromkeys(
                    (
                        *base_provenance,
                    *STOMATE_SAME_STEP_OK_LEAK_BOUNDARY_PROVENANCE,
                    *((pft_static.provenance) if pft_static is not None else ()),
                    *((litter_controls.provenance) if litter_controls is not None else ()),
                    *((tf_doc.provenance) if tf_doc is not None else ()),
                    *((perma_peat.provenance) if perma_peat is not None else ()),
                    *((active_layer.provenance) if active_layer is not None else ()),
                    *((soilwater.provenance) if soilwater is not None else ()),
                    *((permafrost_activity.provenance) if permafrost_activity is not None else ()),
                    *((soil_mc_32l.provenance) if soil_mc_32l is not None else ()),
                    *((doc_transport.provenance) if doc_transport is not None else ()),
                )
            )
        ),
    )
    return StomateSameStepOkLeakBoundaryInputs(
        boundary_inputs=boundary_inputs,
        pre_step=pre_step_boundary,
        pft_static=pft_static,
        litter_controls=litter_controls,
        tf_doc=tf_doc,
        perma_peat=perma_peat,
        active_layer=active_layer,
        soilwater=soilwater,
        permafrost_activity=permafrost_activity,
        soil_mc_32l=soil_mc_32l,
        doc_transport=doc_transport,
    )


def sechiba_stomate_pft14_explicit_step(
    *,
    sechiba: SechibaExplicitCoupledStepResult,
    driver_entry_source: Mapping[str, object],
    extra_entry_sources: tuple[Mapping[str, object], ...] = (),
    gpp_daily_current,
    dt_sechiba,
    dt_stomate,
    do_slow,
    t2m,
    t2m_longterm,
    z_soil,
    rprof,
    sla_calc,
    coeff_maint_zero,
    maint_resp_slope,
    ext_coeff,
    is_tree,
    resp_maint_part_current,
    prescribe_inputs: Mapping[str, object],
    constraints_inputs: Mapping[str, object],
    alloc_inputs: Mapping[str, object],
    post_npp_inputs: Mapping[str, object],
    phenology_inputs: Mapping[str, object] | None = None,
    constraints_kwargs: Mapping[str, object] | None = None,
    phenology_kwargs: Mapping[str, object] | None = None,
    allocation_kwargs: Mapping[str, object] | None = None,
    flood_frac=None,
    one_day=86400.0,
) -> SechibaStomateExplicitResult:
    """Run the currently closed explicit SECHIBA -> STOMATE PFT14 chain.

    Fortran provenance: ``sechiba_main`` calls the local SECHIBA processes at
    lines 997-1216, ``slowproc_main`` forwards the final boundary into
    ``stomate_main`` at lines 973-1023, and ``stomate_main`` schedules GPP and
    maintenance at lines 3020-3267 before calling ``StomateLpj`` at lines
    4297-4335. The STOMATE process order inside ``StomateLpj`` follows
    ``stomate_lpj.f90`` lines 944-1557.

    The caller must still supply exact restart/season/PFT state dictionaries.
    This helper only derives the boundary fields that are already local SECHIBA
    outputs.
    """

    sechiba_entry_source = _entry_source_from_sechiba(
        sechiba,
        extra_entry_sources=extra_entry_sources,
    )
    entry_payload = assemble_stomate_main_payload(driver_entry_source, sechiba_entry_source)
    stomate = stomate_daily_scheduled_gpp_maintenance_prescribe_constraints_alloc_kill_gap_turnover_explicit(
        gpp=sechiba_entry_source["gpp"],
        gpp_daily_current=gpp_daily_current,
        veget_max=sechiba_entry_source["veget_max"],
        totfrac_nobio=sechiba_entry_source["totfrac_nobio"],
        dt_sechiba=dt_sechiba,
        dt_stomate=dt_stomate,
        do_slow=do_slow,
        t2m=t2m,
        t2m_longterm=t2m_longterm,
        stempdiag=sechiba_entry_source["stempdiag"],
        z_soil=z_soil,
        rprof=rprof,
        sla_calc=sla_calc,
        coeff_maint_zero=coeff_maint_zero,
        maint_resp_slope=maint_resp_slope,
        ext_coeff=ext_coeff,
        is_tree=is_tree,
        resp_maint_part_current=resp_maint_part_current,
        prescribe_inputs={**dict(prescribe_inputs), "veget_max": sechiba_entry_source["veget_max"]},
        constraints_inputs=constraints_inputs,
        phenology_inputs=phenology_inputs,
        alloc_inputs=alloc_inputs,
        post_npp_inputs=post_npp_inputs,
        flood_frac=sechiba_entry_source.get("flood_frac") if flood_frac is None else flood_frac,
        constraints_kwargs=constraints_kwargs,
        phenology_kwargs=phenology_kwargs,
        allocation_kwargs=allocation_kwargs,
        one_day=one_day,
    )
    return SechibaStomateExplicitResult(
        sechiba=sechiba,
        entry_payload=entry_payload,
        stomate=stomate,
        missing_entry_inputs=entry_payload.missing_inputs,
        partial_entry_inputs=entry_payload.partial_inputs,
    )


def sechiba_stomate_ok_leak_explicit_step(
    *,
    sechiba: SechibaExplicitCoupledStepResult,
    ok_leak_inputs: Mapping[str, object] | None = None,
    boundary_inputs: StomateOkLeakBoundaryInputs | None = None,
    pft_static_boundary: Mapping[str, object] | None = None,
    litter_controls_boundary: Mapping[str, object] | None = None,
    tf_doc_boundary: Mapping[str, object] | None = None,
    perma_peat_boundary: Mapping[str, object] | None = None,
    active_layer_boundary: Mapping[str, object] | None = None,
    soilwater_boundary: Mapping[str, object] | None = None,
    permafrost_activity_boundary: Mapping[str, object] | None = None,
    soil_mc_32l_boundary: Mapping[str, object] | None = None,
    doc_transport_boundary: Mapping[str, object] | None = None,
    **stomate_kwargs,
) -> SechibaStomateOkLeakExplicitResult:
    """Run current SECHIBA -> STOMATE carbon chain and feed OK_LEAK.

    Fortran provenance follows ``sechiba_stomate_pft14_explicit_step`` through
    ``StomateLpj`` and then ``stomate_main`` lines 3288-3489 for
    ``turnover_daily``/``bm_to_litter`` scaling, ``littercalc_leak``,
    ``soilcarbon_leak``, and DOC aggregation. This helper wires only exact
    upstream outputs and keeps the remaining OK_LEAK boundary values explicit.
    """

    if boundary_inputs is not None:
        ok_leak_inputs = _merge_disjoint_sources(
            boundary_inputs.ok_leak_inputs,
            {} if ok_leak_inputs is None else ok_leak_inputs,
        )
    elif ok_leak_inputs is None:
        ok_leak_inputs = {}
    else:
        ok_leak_inputs = dict(ok_leak_inputs)

    coupled = sechiba_stomate_pft14_explicit_step(sechiba=sechiba, **stomate_kwargs)
    post_npp = coupled.stomate.scheduled_gpp_chain.chain.post_npp
    same_step = stomate_same_step_ok_leak_boundary_inputs(
        sechiba=sechiba,
        post_npp=post_npp,
        dt_sechiba=stomate_kwargs["dt_sechiba"],
        entry_payload=coupled.entry_payload.payload,
        ok_leak_inputs=ok_leak_inputs,
        pft_static_boundary=pft_static_boundary,
        litter_controls_boundary=litter_controls_boundary,
        tf_doc_boundary=tf_doc_boundary,
        perma_peat_boundary=perma_peat_boundary,
        active_layer_boundary=active_layer_boundary,
        soilwater_boundary=soilwater_boundary,
        permafrost_activity_boundary=permafrost_activity_boundary,
        soil_mc_32l_boundary=soil_mc_32l_boundary,
        doc_transport_boundary=doc_transport_boundary,
        is_tree=stomate_kwargs["is_tree"],
        one_day=stomate_kwargs.get("one_day", 86400.0),
    )
    ok_args = dict(same_step.boundary_inputs.ok_leak_inputs)
    ok_leak = stomate_ok_leak_from_post_npp_explicit(
        post_npp=post_npp,
        resp_maint_part_radia=coupled.stomate.maintenance.resp_maint_part,
        flood_root_radia=coupled.stomate.flood_root_radia,
        soil_mc=ok_args.pop("soil_mc"),
        dt_sechiba=stomate_kwargs["dt_sechiba"],
        **ok_args,
    )
    return SechibaStomateOkLeakExplicitResult(
        coupled=coupled,
        ok_leak=ok_leak,
    )


def sechiba_stomate_output_explicit_step(
    *,
    sechiba: SechibaExplicitCoupledStepResult,
    ok_leak_inputs: Mapping[str, object] | None = None,
    output_inputs: Mapping[str, object] | None = None,
    boundary_inputs: StomateOkLeakBoundaryInputs | None = None,
    pft_static_boundary: Mapping[str, object] | None = None,
    litter_controls_boundary: Mapping[str, object] | None = None,
    tf_doc_boundary: Mapping[str, object] | None = None,
    perma_peat_boundary: Mapping[str, object] | None = None,
    active_layer_boundary: Mapping[str, object] | None = None,
    soilwater_boundary: Mapping[str, object] | None = None,
    permafrost_activity_boundary: Mapping[str, object] | None = None,
    soil_mc_32l_boundary: Mapping[str, object] | None = None,
    doc_transport_boundary: Mapping[str, object] | None = None,
    **stomate_kwargs,
) -> SechibaStomateOutputExplicitResult:
    """Run the current explicit chain through STOMATE output/modelout fields."""

    if boundary_inputs is not None:
        ok_leak_inputs = _merge_disjoint_sources(
            boundary_inputs.ok_leak_inputs,
            {} if ok_leak_inputs is None else ok_leak_inputs,
        )
        output_inputs = _merge_disjoint_sources(
            boundary_inputs.output_inputs,
            {} if output_inputs is None else output_inputs,
        )
    elif ok_leak_inputs is None or output_inputs is None:
        missing = tuple(
            name
            for name, value in {
                "ok_leak_inputs": ok_leak_inputs,
                "output_inputs": output_inputs,
            }.items()
            if value is None
        )
        raise ValueError(f"missing output-chain boundary inputs: {missing}")
    else:
        ok_leak_inputs = dict(ok_leak_inputs)
        output_inputs = dict(output_inputs)

    if "veget_max" in output_inputs:
        raise ValueError("output_inputs must not supply veget_max; it is sourced from SECHIBA slowproc")
    if sechiba.slowproc is None:
        raise ValueError("SECHIBA result must include slowproc output for output veget_max")
    ok_leak_chain = sechiba_stomate_ok_leak_explicit_step(
        sechiba=sechiba,
        ok_leak_inputs=ok_leak_inputs,
        boundary_inputs=None,
        pft_static_boundary=pft_static_boundary,
        litter_controls_boundary=litter_controls_boundary,
        tf_doc_boundary=tf_doc_boundary,
        perma_peat_boundary=perma_peat_boundary,
        active_layer_boundary=active_layer_boundary,
        soilwater_boundary=soilwater_boundary,
        permafrost_activity_boundary=permafrost_activity_boundary,
        soil_mc_32l_boundary=soil_mc_32l_boundary,
        doc_transport_boundary=doc_transport_boundary,
        **stomate_kwargs,
    )
    post_npp = ok_leak_chain.coupled.stomate.scheduled_gpp_chain.chain.post_npp
    outputs = stomate_lpj_outputs_from_post_npp_ok_leak_explicit(
        post_npp=post_npp,
        ok_leak=ok_leak_chain.ok_leak,
        gpp_daily=ok_leak_chain.coupled.stomate.scheduled_gpp_chain.gpp_daily,
        veget_max=sechiba.slowproc.vegetation.veget_max,
        **dict(output_inputs),
    )
    return SechibaStomateOutputExplicitResult(
        ok_leak_chain=ok_leak_chain,
        outputs=outputs,
    )


def stomate_restart_ok_leak_state_inputs(
    *,
    state: StomateRestartEntryState,
    veget_max,
    z_soil=None,
    zf_soil=None,
    carb_mass_total_old=None,
    prod10_total=None,
    prod100_total=None,
    contfrac=None,
    one_day=None,
) -> StomateRestartOkLeakStateInputs:
    """Map restart litter/soil-carbon state to OK_LEAK and output inputs.

    Fortran provenance: ``stomate_io.f90::readstart`` lines 1045-1098 read
    ``litterpart``, ``dead_leaves``, and ``fuel_*`` fire fuel state; lines
    1104-1115 and 1152-1156 read product pools and ``carb_mass_total``; lines
    1512-1563 read
    ``litter_above``, ``litter_below``, ``lignin_struc_above``,
    ``lig_struc_be``, ``carbon_32l_*``, ``freedoc``, and ``adsdoc``. Output
    diagnostics consume those same state arrays in ``stomate_lpj.f90`` lines
    1578-1663.
    """

    ok_leak_state_inputs = {
        "litter_above": state.litter_above,
        "litter_below": state.litter_below,
        "lignin_struc_above": state.lignin_struc_above,
        "lignin_struc_below": state.lignin_struc_below,
        "litterpart": state.litterpart,
        "dead_leaves": state.dead_leaves,
        "fuel_1hr": state.fuel_1hr,
        "fuel_10hr": state.fuel_10hr,
        "fuel_100hr": state.fuel_100hr,
        "fuel_1000hr": state.fuel_1000hr,
        "carbon_32l": state.carbon_32l,
        "doc": state.DOC,
        "interception_storage": state.interception_storage,
    }
    output_inputs = {
        "veget_max": veget_max,
        "carb_mass_total_old": state.carb_mass_total if carb_mass_total_old is None else carb_mass_total_old,
        "prod10_total": state.prod10_total if prod10_total is None else prod10_total,
        "prod100_total": state.prod100_total if prod100_total is None else prod100_total,
    }
    if contfrac is not None:
        output_inputs["contfrac"] = contfrac
    if one_day is not None:
        output_inputs["one_day"] = one_day
    if z_soil is not None:
        output_inputs["z_soil"] = z_soil
    if zf_soil is not None:
        output_inputs["zf_soil"] = zf_soil
    return StomateRestartOkLeakStateInputs(
        ok_leak_state_inputs=ok_leak_state_inputs,
        output_inputs=output_inputs,
    )


@lru_cache(maxsize=8)
def _paper_case_stomate_static_input_kwargs_cached(
    config_path: str,
    used_run_def: str | None,
) -> PaperCaseStomateStaticInputKwargs:
    """Return source-backed static kwargs for ``stomate_restart_input_bundles``.

    This helper deliberately excludes climate, season accumulators, restart
    state, and driver diagnostics. It only provides PFT/run-definition values
    that Fortran initializes through ``pft_parameters`` and ``stomate_data``.
    """

    params = load_paper_case_stomate_parameter_bundle(config_path, used_run_def=used_run_def)
    kwargs = {
        "natural": params.natural,
        "pasture": params.pasture,
        "is_tree": params.is_tree,
        "is_peat": params.is_peat,
        "pheno_is_none": params.pheno_is_none,
        "pheno_model": params.pheno_model,
        "ok_laidev": params.ok_laidev,
        "r0": params.r0,
        "s0": params.s0,
        "ext_coeff": params.ext_coeff,
        "lai_max": params.lai_max,
        "lai_max_to_happy": params.lai_max_to_happy,
        "tau_leafinit": params.tau_leafinit,
        "alloc_min": params.alloc_min,
        "alloc_max": params.alloc_max,
        "demi_alloc": params.demi_alloc,
        "alloc_agr_st": params.alloc_agr_st,
        "alloc_agr_pn": params.alloc_agr_pn,
        "frac_growthresp": params.frac_growthresp,
        "senescence_type": params.senescence_type,
        "is_grassland_manag": params.is_grassland_manag,
        "availability_fact": params.availability_fact,
        "residence_time": params.residence_time,
        "leaf_tab": params.leaf_tab,
        "pheno_type": params.pheno_type,
        "tmin_crit": params.tmin_crit,
        "tcm_crit": params.tcm_crit,
        "min_leaf_age_for_senescence": params.min_leaf_age_for_senescence,
        "gdd_senescence": params.gdd_senescence,
        "senescence_temp": params.senescence_temp,
        "hum_frac": params.hum_frac,
        "senescence_hum": params.senescence_hum,
        "nosenescence_hum": params.nosenescence_hum,
        "max_turnover_time": params.max_turnover_time,
        "min_turnover_time": params.min_turnover_time,
        "leaffall": params.leaffall,
        "leafagecrit": params.leafagecrit,
        "lai_initmin": params.lai_initmin,
        "tau_fruit": params.tau_fruit,
        "tau_sap": params.tau_sap,
        "sla_max": params.sla_max,
        "sla_min": params.sla_min,
        "vcmax25": params.vcmax25,
        "leaf_timecst": params.leaf_timecst,
        "grm_n_limitation": params.grm_n_limitation,
        "lpj_gap_const_mort": params.lpj_gap_const_mort,
        "ok_dgvm": params.ok_dgvm,
        "coeff_maint_zero": params.coeff_maint_zero,
        "maint_resp_slope": params.maint_resp_slope,
    }
    return PaperCaseStomateStaticInputKwargs(kwargs=kwargs, parameters=params)


def paper_case_stomate_static_input_kwargs(
    config_path,
    *,
    used_run_def=None,
) -> PaperCaseStomateStaticInputKwargs:
    """Return source-backed static kwargs for ``stomate_restart_input_bundles``.

    This public wrapper preserves the original API while caching immutable
    paper-case parameter inputs by resolved config/run-definition paths.
    """

    config_key = str(Path(config_path).resolve())
    run_def_key = None if used_run_def is None else str(Path(used_run_def).resolve())
    return _paper_case_stomate_static_input_kwargs_cached(config_key, run_def_key)


def stomate_restart_season_input_kwargs(
    season: StomateRestartSeasonState,
) -> StomateRestartSeasonInputKwargs:
    """Return restart-backed dynamic kwargs for the explicit STOMATE chain.

    Only variables present in the restart state are mapped. The helper does not
    fabricate absent climatology/management inputs such as
    ``maxmoiavail_lastyear``, ``minmoiavail_lastyear``, ``herbivores``, or
    ``nrec``.
    """

    kwargs = {
        "t2m_month": season.t2m_month,
        "tseason": season.tseason,
        "moiavail_week": season.moiavail_week,
        "tsoil_month": season.tsoil_month,
        "soilhum_month": season.soilhum_month,
        "tmin_spring_time": season.tmin_spring_time,
        "t2m_longterm": season.t2m_longterm,
        "t2m_week": season.t2m_week,
        "gdd_from_growthinit": season.gdd_from_growthinit,
    }
    return StomateRestartSeasonInputKwargs(kwargs=kwargs, season=season)


def _run_def_float_with_default(values: Mapping[str, str], key: str, default: float) -> float:
    raw = values.get(key)
    if raw is None:
        return float(default)
    return float(str(raw).replace("D", "E").replace("d", "e"))


def _run_def_required_float(values: Mapping[str, str], key: str) -> float:
    raw = values[key]
    return float(str(raw).replace("D", "E").replace("d", "e"))


def _run_def_indexed_vector(values: Mapping[str, str], key: str, count: int, *, dtype=np.float64) -> np.ndarray:
    return np.asarray([_run_def_required_float(values, f"{key}__{idx:05d}") for idx in range(1, int(count) + 1)], dtype=dtype)


def _paper_case_used_run_def_path(config_path, used_run_def=None):
    from jax_orchidee.driver.domain import load_case_config
    from jax_orchidee.stomate.parameters import USED_RUN_DEF

    root = Path(load_case_config(config_path)["paths"]["workspace_root"])
    return root / (USED_RUN_DEF if used_run_def is None else Path(used_run_def))


@lru_cache(maxsize=8)
def _paper_case_season_time_scales_cached(config_path: str, used_run_def: str | None) -> SeasonTimeScales:
    """Load source-backed ``stomate_season`` time constants for the paper case.

    Fortran provenance: ``constantes.f90`` lines 1351-1397, 1423-1429, and
    2077-2101 read the run.def keys; ``stomate_data.f90`` line 700 derives
    ``tau_longterm_max`` as ``coeff_tau_longterm * one_year``.
    """

    values = read_run_def_values(_paper_case_used_run_def_path(config_path, used_run_def))
    return SeasonTimeScales(
        tau_hum_month=_run_def_float_with_default(values, "TAU_HUM_MONTH", 20.0),
        tau_hum_week=_run_def_float_with_default(values, "TAU_HUM_WEEK", 7.0),
        tau_t2m_month=_run_def_float_with_default(values, "TAU_T2M_MONTH", 20.0),
        tau_t2m_week=_run_def_float_with_default(values, "TAU_T2M_WEEK", 7.0),
        tau_tsoil_month=_run_def_float_with_default(values, "TAU_TSOIL_MONTH", 20.0),
        tau_soilhum_month=_run_def_float_with_default(values, "TAU_SOILHUM_MONTH", 20.0),
        tau_gpp_week=_run_def_float_with_default(values, "TAU_GPP_WEEK", 7.0),
        tau_gdd=_run_def_float_with_default(values, "TAU_GDD", 40.0),
        tau_ngd=_run_def_float_with_default(values, "TAU_NGD", 50.0),
        tau_climatology=_run_def_float_with_default(values, "TAU_CLIMATOLOGY", 20.0),
        coeff_tau_longterm=_run_def_float_with_default(values, "COEFF_TAU_LONGTERM", 3.0),
        tlong_ref_min=_run_def_float_with_default(values, "TLONG_REF_MIN", 253.1),
        tlong_ref_max=_run_def_float_with_default(values, "TLONG_REF_MAX", 303.1),
    )


def paper_case_season_time_scales(
    config_path,
    *,
    used_run_def=None,
) -> SeasonTimeScales:
    """Load source-backed ``stomate_season`` time constants for the paper case."""

    config_key = str(Path(config_path).resolve())
    run_def_key = None if used_run_def is None else str(Path(used_run_def).resolve())
    return _paper_case_season_time_scales_cached(config_key, run_def_key)


def stomate_restart_season_memory_state(
    season: StomateRestartSeasonState,
    *,
    tau_longterm,
) -> SeasonMemoryState:
    """Build the explicit step state from restart-backed season memory.

    ``tau_longterm`` is required because it is a Fortran SAVE scalar used by
    ``stomate_season::season`` line 626, not a field present in the restart
    reader currently audited for this project.
    """

    return SeasonMemoryState(
        moiavail_month=season.moiavail_month,
        moiavail_week=season.moiavail_week,
        t2m_longterm=season.t2m_longterm,
        tau_longterm=tau_longterm,
        t2m_month=season.t2m_month,
        t2m_week=season.t2m_week,
        tseason=season.tseason,
        tseason_length=season.tseason_length,
        tseason_tmp=season.tseason_tmp,
        tmin_spring_time=season.tmin_spring_time,
        onset_date=season.onset_date,
        tsoil_month=season.tsoil_month,
        soilhum_month=season.soilhum_month,
        gdd_from_growthinit=season.gdd_from_growthinit,
    )


def stomate_season_memory_input_kwargs(
    *,
    season: StomateRestartSeasonState,
    daily: StomateDailyAccumulatorState,
    entry_state: StomateRestartEntryState,
    parameters: PaperCaseStomateParameterBundle,
    dt_days,
    tau_longterm,
    julian_diff,
    end_of_year=False,
    time_scales: SeasonTimeScales | None = None,
    firstcall=False,
) -> StomateSeasonMemoryInputKwargs:
    """Advance restart season memory one step and map it into STOMATE kwargs.

    This closes the local ``season`` memory subset available before the carbon
    chain. Inputs that belong to phenology/yearly climatology/herbivory remain
    explicit elsewhere rather than being guessed here.
    """

    state = stomate_restart_season_memory_state(season, tau_longterm=tau_longterm)
    step = season_memory_step(
        state,
        dt_days=dt_days,
        end_of_year=end_of_year,
        moiavail_daily=daily.humrel_daily,
        t2m_daily=daily.t2m_daily,
        tsoil_daily=daily.tsoil_daily,
        soilhum_daily=daily.soilhum_daily,
        begin_leaves=season.begin_leaves,
        julian_diff=julian_diff,
        when_growthinit=entry_state.when_growthinit,
        natural=parameters.natural,
        leaf_tab=parameters.leaf_tab,
        pheno_type=parameters.pheno_type,
        pft_to_mtc=parameters.pft_to_mtc,
        time_scales=time_scales,
        firstcall=firstcall,
    )
    updated = step.state
    kwargs = {
        "t2m_month": updated.t2m_month,
        "tseason": updated.tseason,
        "moiavail_week": updated.moiavail_week,
        "tsoil_month": updated.tsoil_month,
        "soilhum_month": updated.soilhum_month,
        "tmin_spring_time": updated.tmin_spring_time,
        "t2m_longterm": updated.t2m_longterm,
        "t2m_week": updated.t2m_week,
        "gdd_from_growthinit": updated.gdd_from_growthinit,
    }
    return StomateSeasonMemoryInputKwargs(kwargs=kwargs, step=step)


def stomate_restart_season_annual_state(
    *,
    season: StomateRestartSeasonState,
    entry_state: StomateRestartEntryState,
) -> SeasonAnnualState:
    """Build section 12-21 season annual state from audited restart readers."""

    return SeasonAnnualState(
        npp_longterm=entry_state.npp_longterm,
        turnover_longterm=entry_state.turnover_longterm,
        gpp_week=season.gpp_week,
        maxmoiavail_lastyear=season.maxmoiavail_lastyear,
        maxmoiavail_thisyear=season.maxmoiavail_thisyear,
        minmoiavail_lastyear=season.minmoiavail_lastyear,
        minmoiavail_thisyear=season.minmoiavail_thisyear,
        maxgppweek_lastyear=season.maxgppweek_lastyear,
        maxgppweek_thisyear=season.maxgppweek_thisyear,
        gdd0_lastyear=season.gdd0_lastyear,
        gdd0_thisyear=season.gdd0_thisyear,
        precip_lastyear=season.precip_lastyear,
        precip_thisyear=season.precip_thisyear,
        lm_lastyearmax=entry_state.lm_lastyearmax,
        lm_thisyearmax=season.lm_thisyearmax,
        maxfpc_lastyear=season.maxfpc_lastyear,
        maxfpc_thisyear=season.maxfpc_thisyear,
    )


def stomate_season_annual_input_kwargs(
    *,
    season: StomateRestartSeasonState,
    daily: StomateDailyAccumulatorState,
    entry_state: StomateRestartEntryState,
    parameters: PaperCaseStomateParameterBundle,
    veget,
    veget_max,
    dt_days,
    tau_longterm,
    end_of_year=False,
    firstcall=False,
    time_scales: SeasonTimeScales | None = None,
    hvc1=0.019,
    hvc2=1.38,
    leaf_frac_hvc=0.33,
    green_age_ever=2.0,
    green_age_dec=0.5,
) -> StomateSeasonAnnualInputKwargs:
    """Advance ``season`` section 12-21 and expose values used downstream."""

    state = stomate_restart_season_annual_state(season=season, entry_state=entry_state)
    step = season_annual_step(
        state,
        dt_days=dt_days,
        tau_longterm=tau_longterm,
        end_of_year=end_of_year,
        firstcall=firstcall,
        veget=veget,
        veget_max=veget_max,
        moiavail_daily=daily.humrel_daily,
        t2m_daily=daily.t2m_daily,
        precip_daily=daily.precip_daily,
        biomass=entry_state.biomass,
        npp_daily=entry_state.npp_daily,
        turnover_daily=entry_state.turnover_daily,
        gpp_daily=daily.gpp_daily,
        natural=parameters.natural,
        pasture=parameters.pasture,
        leaflife_tab=parameters.residence_time,
        pheno_model=parameters.pheno_model,
        hvc1=hvc1,
        hvc2=hvc2,
        leaf_frac_hvc=leaf_frac_hvc,
        green_age_ever=green_age_ever,
        green_age_dec=green_age_dec,
        ok_stomate=True,
        ok_dgvm=parameters.ok_dgvm,
        time_scales=time_scales,
    )
    updated = step.state
    kwargs = {
        "npp_longterm": updated.npp_longterm,
        "turnover_longterm": updated.turnover_longterm,
        "gpp_week": updated.gpp_week,
        "lm_lastyearmax": updated.lm_lastyearmax,
        "maxfpc_lastyear": updated.maxfpc_lastyear,
        "maxfpc_thisyear": updated.maxfpc_thisyear,
        "herbivores": step.herbivores,
    }
    return StomateSeasonAnnualInputKwargs(kwargs=kwargs, step=step)


def stomate_restart_season_biometeorology_state(
    season: StomateRestartSeasonState,
) -> SeasonBiometeorologyState:
    """Build section 7-11 biometeorological state from restart memory."""

    return SeasonBiometeorologyState(
        gdd_m5_dormance=season.gdd_m5_dormance,
        gdd_midwinter=season.gdd_midwinter,
        ncd_dormance=season.ncd_dormance,
        ngd_minus5=season.ngd_minus5,
        time_hum_min=season.time_hum_min,
        hum_min_dormance=season.hum_min_dormance,
    )


def stomate_season_biometeorology_input_kwargs(
    *,
    season: StomateRestartSeasonState,
    daily: StomateDailyAccumulatorState,
    entry_state: StomateRestartEntryState,
    parameters: PaperCaseStomateParameterBundle,
    t2m_month,
    t2m_week,
    t2m_longterm,
    moiavail_month,
    dt_days,
    julian_diff,
    firstcall=False,
    time_scales: SeasonTimeScales | None = None,
) -> StomateSeasonBiometeorologyInputKwargs:
    """Advance ``season`` section 7-11 from restart/daily/static inputs."""

    state = stomate_restart_season_biometeorology_state(season)
    step = season_biometeorology_step(
        state,
        dt_days=dt_days,
        julian_diff=julian_diff,
        t2m_daily=daily.t2m_daily,
        t2m_month=t2m_month,
        t2m_week=t2m_week,
        t2m_longterm=t2m_longterm,
        moiavail_month=moiavail_month,
        when_growthinit=entry_state.when_growthinit,
        gdd_init_date=season.gdd_init_date,
        pheno_gdd_crit=parameters.pheno_gdd_crit,
        ncdgdd_temp=parameters.ncdgdd_temp,
        hum_min_time=parameters.hum_min_time,
        firstcall=firstcall,
        time_scales=time_scales,
    )
    updated = step.state
    kwargs = {
        "gdd_m5_dormance": updated.gdd_m5_dormance,
        "gdd_midwinter": updated.gdd_midwinter,
        "ncd_dormance": updated.ncd_dormance,
        "ngd_minus5": updated.ngd_minus5,
        "time_hum_min": updated.time_hum_min,
        "hum_min_dormance": updated.hum_min_dormance,
    }
    return StomateSeasonBiometeorologyInputKwargs(kwargs=kwargs, step=step)


def paper_case_stomate_pre_step_input_kwargs(
    *,
    config_path,
    season: StomateRestartSeasonState,
    daily: StomateDailyAccumulatorState,
    entry_state: StomateRestartEntryState,
    veget,
    veget_max,
    dt_days,
    tau_longterm=None,
    julian_diff,
    end_of_year=False,
    firstcall_season=False,
    used_run_def=None,
    run_def_values=None,
    parameter_overrides: Mapping[str, object] | None = None,
) -> PaperCaseStomatePreStepInputKwargs:
    """Compose source-backed static parameters and one local ``season`` step.

    The returned kwargs can be passed to ``stomate_restart_input_bundles`` along
    with remaining driver/restart values. ``tau_longterm`` defaults to the
    audited restart scalar read by ``stomate_io.f90::readstart``.
    """

    static = paper_case_stomate_static_input_kwargs(config_path, used_run_def=used_run_def)
    if parameter_overrides:
        # Compiled complete-day transitions supply continuous landpoint values
        # here. Branch topology remains static; no scientific formula moves
        # out of the source-backed parameter bundle.
        parameters = replace(static.parameters, **dict(parameter_overrides))
        kwargs = dict(static.kwargs)
        kwargs.update({name: getattr(parameters, name) for name in parameter_overrides})
        static = static._replace(kwargs=kwargs, parameters=parameters)
    scales = paper_case_season_time_scales(config_path, used_run_def=used_run_def)
    run_def_values = (
        read_run_def_values(_paper_case_used_run_def_path(config_path, used_run_def))
        if run_def_values is None
        else run_def_values
    )
    tau_longterm_value = season.tau_longterm if tau_longterm is None else tau_longterm
    memory = stomate_season_memory_input_kwargs(
        season=season,
        daily=daily,
        entry_state=entry_state,
        parameters=static.parameters,
        dt_days=dt_days,
        tau_longterm=tau_longterm_value,
        julian_diff=julian_diff,
        end_of_year=end_of_year,
        time_scales=scales,
        firstcall=firstcall_season,
    )
    annual = stomate_season_annual_input_kwargs(
        season=season,
        daily=daily,
        entry_state=entry_state,
        parameters=static.parameters,
        veget=veget,
        veget_max=veget_max,
        dt_days=dt_days,
        tau_longterm=memory.step.state.tau_longterm,
        end_of_year=end_of_year,
        firstcall=firstcall_season,
        time_scales=scales,
        hvc1=_run_def_float_with_default(run_def_values, "HVC1", 0.019),
        hvc2=_run_def_float_with_default(run_def_values, "HVC2", 1.38),
        leaf_frac_hvc=_run_def_float_with_default(run_def_values, "LEAF_FRAC_HVC", 0.33),
        green_age_ever=_run_def_float_with_default(run_def_values, "GREEN_AGE_EVER", 2.0),
        green_age_dec=_run_def_float_with_default(run_def_values, "GREEN_AGE_DEC", 0.5),
    )
    biometeorology = stomate_season_biometeorology_input_kwargs(
        season=season,
        daily=daily,
        entry_state=entry_state,
        parameters=static.parameters,
        t2m_month=memory.step.state.t2m_month,
        t2m_week=memory.step.state.t2m_week,
        t2m_longterm=memory.step.state.t2m_longterm,
        moiavail_month=memory.step.state.moiavail_month,
        dt_days=dt_days,
        julian_diff=julian_diff,
        firstcall=firstcall_season,
        time_scales=scales,
    )
    kwargs = {
        **static.kwargs,
        **memory.kwargs,
        "npp_longterm": annual.kwargs["npp_longterm"],
        "turnover_longterm": annual.kwargs["turnover_longterm"],
        "lm_lastyearmax": annual.kwargs["lm_lastyearmax"],
        "herbivores": annual.kwargs["herbivores"],
    }
    return PaperCaseStomatePreStepInputKwargs(
        kwargs=kwargs,
        static=static,
        memory=memory,
        annual=annual,
        biometeorology=biometeorology,
    )


def paper_case_stomate_boundary_input_kwargs(
    config_path,
    *,
    parameters: PaperCaseStomateParameterBundle | None = None,
    npts: int,
    used_run_def=None,
    run_def_values=None,
) -> PaperCaseStomateBoundaryInputKwargs:
    """Build STOMATE ``z_soil`` and ``rprof`` from audited paper-case sources.

    Fortran provenance: ``vertical_soil_init`` reads the CWRR depth parameters
    and builds the thermal ``znt`` grid, ``control_initialize`` copies
    ``znt(1:nslm)`` to ``diaglev``, ``hydrol_init`` broadcasts
    ``HYDROL_HUMCSTE`` to ``humcste_use``, and ``stomate_main``/``sechiba_main``
    set ``rprof=1/humcste_use`` before carbon and DIFFUCO calls.
    """

    params = parameters if parameters is not None else load_paper_case_stomate_parameter_bundle(config_path, used_run_def=used_run_def)
    values = read_run_def_values(_paper_case_used_run_def_path(config_path, used_run_def)) if run_def_values is None else run_def_values
    z_soil = z_soil_from_cwrr_vertical_soil_params(
        depth_max_h=_run_def_required_float(values, "DEPTH_MAX_H"),
        depth_max_t=_run_def_required_float(values, "DEPTH_MAX_T"),
        depth_topthickness=_run_def_required_float(values, "DEPTH_TOPTHICK"),
        depth_cstthickness=_run_def_required_float(values, "DEPTH_CSTTHICK"),
        depth_geom=_run_def_required_float(values, "DEPTH_GEOM"),
        ratio_geom_below=_run_def_required_float(values, "RATIO_GEOM_BELOW"),
    )
    humcste = humcste_from_pft_to_mtc(
        params.pft_to_mtc,
        zmaxh=_run_def_required_float(values, "DEPTH_MAX_H"),
        hydrol_humcste=_run_def_indexed_vector(values, "HYDROL_HUMCSTE", params.nvm),
    )
    humcste_use = humcste_use_from_humcste(humcste, npts=npts)
    rprof = rprof_from_humcste_use(humcste_use)
    return PaperCaseStomateBoundaryInputKwargs(
        kwargs={
            "z_soil": z_soil,
            "rprof": rprof,
        },
        humcste=humcste,
        humcste_use=humcste_use,
    )


def paper_case_stomate_data_input_kwargs(
    config_path,
    *,
    parameters: PaperCaseStomateParameterBundle | None = None,
    npts: int,
    used_run_def=None,
    run_def_values=None,
) -> PaperCaseStomateDataInputKwargs:
    """Build source-backed ``bm_sapl``, ``maxdia``, and initial ``sla_age1``.

    The sapling and critical-diameter formulas follow ``stomate_data.f90``
    lines 261-346 and 385-397. ``sla_age1`` follows ``stomate_init`` lines
    8303-8306, which initializes each point/PFT to ``sla_max(j)``.
    """

    params = parameters if parameters is not None else load_paper_case_stomate_parameter_bundle(config_path, used_run_def=used_run_def)
    values = read_run_def_values(_paper_case_used_run_def_path(config_path, used_run_def)) if run_def_values is None else run_def_values
    nvm = int(params.nvm)
    npts = int(npts)
    if npts <= 0:
        raise ValueError("npts must be positive")

    pipe_tune1 = _run_def_required_float(values, "PIPE_TUNE1")
    pipe_tune2 = _run_def_required_float(values, "PIPE_TUNE2")
    pipe_tune3 = _run_def_required_float(values, "PIPE_TUNE3")
    pipe_tune4 = _run_def_required_float(values, "PIPE_TUNE4")
    pipe_density = _run_def_required_float(values, "PIPE_DENSITY")
    pipe_k1 = _run_def_required_float(values, "PIPE_K1")
    alpha_grass = _run_def_required_float(values, "ALPHA_GRASS")
    alpha_tree = _run_def_required_float(values, "ALPHA_TREE")
    mass_ratio_heart_sap = _run_def_required_float(values, "MASS_RATIO_HEART_SAP")
    bm_sapl_carbres = _run_def_required_float(values, "BM_SAPL_CARBRES")
    bm_sapl_sapabove = _run_def_required_float(values, "BM_SAPL_SAPABOVE")
    bm_sapl_heartabove = _run_def_required_float(values, "BM_SAPL_HEARTABOVE")
    bm_sapl_heartbelow = _run_def_required_float(values, "BM_SAPL_HEARTBELOW")
    init_sapl_mass_leaf_nat = _run_def_required_float(values, "INIT_SAPL_MASS_LEAF_NAT")
    init_sapl_mass_leaf_agri = _run_def_required_float(values, "INIT_SAPL_MASS_LEAF_AGRI")
    init_sapl_mass_carbres = _run_def_required_float(values, "INIT_SAPL_MASS_CARBRES")
    init_sapl_mass_root = _run_def_required_float(values, "INIT_SAPL_MASS_ROOT")
    init_sapl_mass_fruit = _run_def_required_float(values, "INIT_SAPL_MASS_FRUIT")
    dia_coeff = _run_def_indexed_vector(values, "DIA_COEFF", 2)
    maxdia_coeff = _run_def_indexed_vector(values, "MAXDIA_COEFF", 2)
    bm_sapl_leaf = _run_def_indexed_vector(values, "BM_SAPL_LEAF", 4)
    sla_vector = _run_def_indexed_vector(values, "SLA", nvm)
    use_age_class = str(values["GLUC_USE_AGE_CLASS"]).strip().strip(".").upper() in {"T", "TRUE", "Y", "YES", "1"}

    bm_sapl = np.zeros((nvm, NPARTS, 1), dtype=np.float64)
    maxdia = np.full(nvm, DEFAULT_UNDEF, dtype=np.float64)
    for j in range(nvm):
        sla = float(sla_vector[j])
        if bool(params.is_tree[j]):
            alpha = alpha_tree
            leaf = (
                (
                    bm_sapl_leaf[0]
                    * pipe_tune1
                    * (mass_ratio_heart_sap * bm_sapl_leaf[1] * sla / (np.pi * pipe_k1)) ** bm_sapl_leaf[2]
                )
                / sla
            ) ** bm_sapl_leaf[3]
            bm_sapl[j, ILEAF, ICARBON] = leaf
            bm_sapl[j, ICARBRES, ICARBON] = bm_sapl_carbres * leaf if int(params.pheno_type[j]) != 1 else 0.0
            csa_sap = leaf / (pipe_k1 / sla)
            dia = (mass_ratio_heart_sap * csa_sap * dia_coeff[0] / np.pi) ** dia_coeff[1]
            bm_sapl[j, ISAPABOVE, ICARBON] = bm_sapl_sapabove * pipe_density * csa_sap * pipe_tune2 * dia**pipe_tune3
            bm_sapl[j, ISAPBELOW, ICARBON] = bm_sapl[j, ISAPABOVE, ICARBON]
            bm_sapl[j, IHEARTABOVE, ICARBON] = bm_sapl_heartabove * bm_sapl[j, ISAPABOVE, ICARBON]
            bm_sapl[j, IHEARTBELOW, ICARBON] = bm_sapl_heartbelow * bm_sapl[j, ISAPBELOW, ICARBON]
            maxdia[j] = (
                pipe_tune4 / ((pipe_tune2 * pipe_tune3) / (maxdia_coeff[0] ** pipe_tune3))
            ) ** (1.0 / (pipe_tune3 - 1.0)) * maxdia_coeff[1]
        else:
            alpha = alpha_grass
            if bool(params.ok_laidev[j]):
                if bool(params.natural[j]):
                    raise ValueError("Fortran stomate_data rejects PFTs that are both ok_LAIdev and natural")
                leaf = 0.0
                bm_sapl[j, ICARBRES, ICARBON] = 0.0
            else:
                leaf = (init_sapl_mass_leaf_nat if (bool(params.natural[j]) or bool(params.is_grassland_manag[j])) else init_sapl_mass_leaf_agri) / sla
                bm_sapl[j, ICARBRES, ICARBON] = init_sapl_mass_carbres * leaf
            bm_sapl[j, ILEAF, ICARBON] = leaf
            maxdia[j] = DEFAULT_UNDEF

        bm_sapl[j, IROOT, ICARBON] = init_sapl_mass_root * (1.0 / alpha) * bm_sapl[j, ILEAF, ICARBON]
        bm_sapl[j, IFRUIT, ICARBON] = init_sapl_mass_fruit * bm_sapl[j, ILEAF, ICARBON]
        bm_sapl[j, IAGRSAPST, ICARBON] = 0.0
        bm_sapl[j, IAGRSAPPN, ICARBON] = 0.0
        bm_sapl[j, IAGRHRTST, ICARBON] = 0.0
        bm_sapl[j, IAGRHRTPN, ICARBON] = 0.0

    if (not bool(params.ok_dgvm)) and use_age_class:
        bm_sapl[:, :, ICARBON] *= 0.05

    sla_age1 = np.broadcast_to(np.asarray(params.sla_max, dtype=np.float64), (npts, nvm)).copy()
    return PaperCaseStomateDataInputKwargs(
        kwargs={
            "bm_sapl": bm_sapl,
            "bm_sapl_rescale": _run_def_required_float(values, "BM_SAPL_RESCALE"),
            "maxdia": maxdia,
            "sla_age1": sla_age1,
        }
    )


def paper_case_inactive_crop_input_kwargs(
    *,
    parameters: PaperCaseStomateParameterBundle,
    npts: int,
) -> PaperCaseInactiveCropInputKwargs:
    """Return neutral crop-only state when all ``ok_LAIdev`` flags are false.

    This does not implement STICS. It only materializes the inactive-branch
    value needed by downstream turnover when the audited paper-case run has no
    active ``ok_LAIdev`` PFTs.
    """

    if bool(np.any(parameters.ok_laidev)):
        raise NotImplementedError("nrec cannot be source-backed as zero when any ok_LAIdev PFT is active; implement the STICS crop state first")
    npts = int(npts)
    if npts <= 0:
        raise ValueError("npts must be positive")
    return PaperCaseInactiveCropInputKwargs(
        kwargs={
            "nrec": np.zeros((npts, int(parameters.nvm)), dtype=np.int32),
        }
    )


def paper_case_stomate_bundle_source_kwargs(
    *,
    config_path,
    season: StomateRestartSeasonState,
    daily: StomateDailyAccumulatorState,
    entry_state: StomateRestartEntryState,
    slowproc_state: SlowprocRestartEntryState,
    veget,
    veget_max,
    dt_days,
    tau_longterm=None,
    julian_diff,
    end_of_year=False,
    firstcall_season=False,
    used_run_def=None,
    run_def_values=None,
    parameter_overrides: Mapping[str, object] | None = None,
) -> PaperCaseStomateBundleSourceKwargs:
    """Return source-backed kwargs accepted by ``stomate_restart_input_bundles``.

    Source-backed values for sapling biomass, diameter, vertical grid, root
    profile, inactive crop counters, SLA age-1 state, and herbivore forcing are
    included. Remaining driver scheduling values such as ``do_slow`` and
    process-local ``cn_ind`` must still be supplied at the call site.
    """

    pre_step = paper_case_stomate_pre_step_input_kwargs(
        config_path=config_path,
        season=season,
        daily=daily,
        entry_state=entry_state,
        veget=veget,
        veget_max=veget_max,
        dt_days=dt_days,
        tau_longterm=tau_longterm,
        julian_diff=julian_diff,
        end_of_year=end_of_year,
        firstcall_season=firstcall_season,
        used_run_def=used_run_def,
        run_def_values=run_def_values,
        parameter_overrides=parameter_overrides,
    )
    boundary = paper_case_stomate_boundary_input_kwargs(
        config_path,
        parameters=pre_step.static.parameters,
        npts=entry_state.biomass.shape[0],
        used_run_def=used_run_def,
        run_def_values=run_def_values,
    )
    data = paper_case_stomate_data_input_kwargs(
        config_path,
        parameters=pre_step.static.parameters,
        npts=entry_state.biomass.shape[0],
        used_run_def=used_run_def,
        run_def_values=run_def_values,
    )
    inactive_crop = paper_case_inactive_crop_input_kwargs(
        parameters=pre_step.static.parameters,
        npts=entry_state.biomass.shape[0],
    )
    kwargs = {
        **pre_step.kwargs,
        **boundary.kwargs,
        **data.kwargs,
        **inactive_crop.kwargs,
        "height": slowproc_state.height,
        "t2m_min_daily": daily.t2m_min_daily,
        "t2m": daily.t2m_daily,
        "maxmoiavail_lastyear": season.maxmoiavail_lastyear,
        "minmoiavail_lastyear": season.minmoiavail_lastyear,
    }
    return PaperCaseStomateBundleSourceKwargs(
        kwargs=kwargs,
        pre_step=pre_step,
        boundary=boundary,
        data=data,
        inactive_crop=inactive_crop,
    )


def stomate_restart_input_bundles(
    *,
    state: StomateRestartEntryState,
    veget_max,
    lai,
    cn_ind,
    dt_days,
    natural,
    pasture,
    is_tree,
    is_peat,
    pheno_is_none,
    bm_sapl,
    maxdia,
    t2m_month,
    t2m_min_daily,
    tseason,
    tmin_crit,
    tcm_crit,
    pheno_model,
    moiavail_week,
    tsoil_month,
    soilhum_month,
    z_soil,
    ok_laidev,
    r0,
    s0,
    ext_coeff,
    lai_max,
    lai_max_to_happy,
    tau_leafinit,
    alloc_min,
    alloc_max,
    demi_alloc,
    alloc_agr_st,
    alloc_agr_pn,
    frac_growthresp,
    height,
    tmin_spring_time,
    herbivores,
    maxmoiavail_lastyear,
    minmoiavail_lastyear,
    t2m_longterm,
    t2m_week,
    gdd_from_growthinit,
    nrec,
    senescence_type,
    is_grassland_manag,
    availability_fact,
    residence_time,
    leaf_tab,
    pheno_type,
    min_leaf_age_for_senescence,
    gdd_senescence,
    senescence_temp,
    hum_frac,
    senescence_hum,
    nosenescence_hum,
    max_turnover_time,
    min_turnover_time,
    leaffall,
    leafagecrit,
    lai_initmin,
    tau_fruit,
    tau_sap,
    sla_age1,
    sla_max,
    sla_min,
    vcmax25,
    leaf_timecst,
    grm_n_limitation,
    lpj_gap_const_mort,
    ok_dgvm,
    t2m,
    rprof,
    coeff_maint_zero,
    maint_resp_slope,
    dt_sechiba,
    dt_stomate,
    do_slow,
    stomate_restart_none=False,
    daily_fields=None,
    npp_longterm=None,
    turnover_longterm=None,
    lm_lastyearmax=None,
    gpp_week=None,
    n_limfert=None,
    bm_sapl_rescale=1.0,
) -> StomateRestartInputBundles:
    """Build explicit STOMATE chain dictionaries from restart-backed state.

    Fortran provenance: all state fields copied from ``state`` are read in
    ``stomate_io.f90::readstart`` at the line spans listed on
    ``StomateRestartInputBundles``. This helper does not invent climate,
    season, PFT-parameter, or driver values; those remain required keyword
    arguments.
    """

    npp_longterm_value = state.npp_longterm if npp_longterm is None else npp_longterm
    turnover_longterm_value = state.turnover_longterm if turnover_longterm is None else turnover_longterm
    lm_lastyearmax_value = state.lm_lastyearmax if lm_lastyearmax is None else lm_lastyearmax
    daily_fields = {} if daily_fields is None else dict(daily_fields)
    gpp_daily_value = daily_fields.get("gpp_daily", state.gpp_daily)
    t2m_daily_value = daily_fields.get("t2m_daily", t2m)
    t2m_min_daily_value = daily_fields.get("t2m_min_daily", t2m_min_daily)
    snowfall_daily_value = daily_fields.get("snowfall_daily")
    snowmass_daily_value = daily_fields.get("snowmass_daily")
    tmc_topgrass_daily_value = daily_fields.get("tmc_topgrass_daily")
    resp_maint_part_value = daily_fields.get("resp_maint_part", state.resp_maint_part)
    resp_maint_radia_value = daily_fields.get("resp_maint_radia")
    flood_root_radia_value = daily_fields.get("flood_root_radia")
    npts, nvm = state.biomass.shape[:2]
    if bool(grm_n_limitation):
        raise NotImplementedError(
            "GRM_N_LIMITATION=True requires source-backed N_limfert wiring before stomate_vmax can be enabled"
        )
    if n_limfert is None:
        ok_laidev_array = jnp.asarray(ok_laidev, dtype=bool)
        if not isinstance(ok_laidev_array, jax.core.Tracer) and bool(
            jnp.any(ok_laidev_array)
        ):
            raise NotImplementedError(
                "ok_LAIdev=True makes stomate_vmax consume N_limfert; provide the audited crop/N limitation state first"
            )
        n_limfert_value = jnp.ones((npts, nvm), dtype=state.biomass.dtype)
    else:
        n_limfert_value = n_limfert
    # StomateLpj clears these work arrays on entry before prescribe/gap/turnover
    # rebuild them for the current day.
    co2_to_bm_entry = jnp.zeros_like(state.co2_to_bm)
    bm_to_litter_entry = jnp.zeros_like(state.bm_to_litter)

    prescribe_inputs = {
        "veget_max": veget_max,
        "dt_days": dt_days,
        "pft_present": state.pft_present,
        "everywhere": state.everywhere,
        "when_growthinit": state.when_growthinit,
        "biomass": state.biomass,
        "leaf_frac": state.leaf_frac,
        "ind": state.ind,
        "cn_ind": cn_ind,
        "co2_to_bm": co2_to_bm_entry,
        "natural": natural,
        "pasture": pasture,
        "is_tree": is_tree,
        "bm_sapl": bm_sapl,
        "bm_sapl_rescale": bm_sapl_rescale,
        "maxdia": maxdia,
        "pheno_is_none": pheno_is_none,
        "ok_dgvm": ok_dgvm,
        "lpj_gap_const_mort": lpj_gap_const_mort,
        "min_stomate": MIN_STOMATE,
    }
    if bool(stomate_restart_none):
        prescribe_inputs["stomate_restart_none"] = True
    constraints_inputs = {
        "t2m_month": t2m_month,
        "t2m_min_daily": t2m_min_daily_value,
        "adapted": state.adapted,
        "regenerate": state.regenerate,
        "tseason": tseason,
        "natural": natural,
        "is_tree": is_tree,
        "is_peat": is_peat,
        "pheno_is_none": pheno_is_none,
        "tmin_crit": tmin_crit,
        "tcm_crit": tcm_crit,
        "dt_days": dt_days,
    }
    phenology_inputs = {
        "leaf_age": state.leaf_age,
        "pheno_model": pheno_model,
        "dt_days": dt_days,
    }
    alloc_inputs = {
        "lai": lai,
        "senescence": state.senescence,
        "moiavail_week": moiavail_week,
        "tsoil_month": tsoil_month,
        "soilhum_month": soilhum_month,
        "age": state.age,
        "leaf_age": state.leaf_age,
        "z_soil": z_soil,
        "sla_calc": state.sla_calc,
        "natural": natural,
        "pasture": pasture,
        "is_tree": is_tree,
        "ok_LAIdev": ok_laidev,
        "r0": r0,
        "s0": s0,
        "ext_coeff": ext_coeff,
        "lai_max": lai_max,
        "lai_max_to_happy": lai_max_to_happy,
        "tau_leafinit": tau_leafinit,
        "alloc_min": alloc_min,
        "alloc_max": alloc_max,
        "demi_alloc": demi_alloc,
        "alloc_agr_st": alloc_agr_st,
        "alloc_agr_pn": alloc_agr_pn,
        "dt_days": dt_days,
        "min_stomate": MIN_STOMATE,
    }
    post_npp_inputs = {
        "frac_growthresp": frac_growthresp,
        "npp_longterm": npp_longterm_value,
        "turnover_longterm": turnover_longterm_value,
        "lm_lastyearmax": lm_lastyearmax_value,
        "bm_to_litter": bm_to_litter_entry,
        "senescence": state.senescence,
        "rip_time": state.rip_time,
        "height": height,
        "t2m_min_daily": t2m_min_daily_value,
        "tmin_spring_time": tmin_spring_time,
        "herbivores": herbivores,
        "maxmoiavail_lastyear": maxmoiavail_lastyear,
        "minmoiavail_lastyear": minmoiavail_lastyear,
        "moiavail_week": moiavail_week,
        "t2m_longterm": t2m_longterm,
        "t2m_month": t2m_month,
        "t2m_week": t2m_week,
        "veget_max": veget_max,
        "gdd_from_growthinit": gdd_from_growthinit,
        "age": state.age,
        "turnover_time": state.turnover_time,
        "nrec": nrec,
        "sla_calc": state.sla_calc,
        "senescence_type": senescence_type,
        "is_tree": is_tree,
        "natural": natural,
        "pasture": pasture,
        "is_grassland_manag": is_grassland_manag,
        "ok_laidev": ok_laidev,
        "availability_fact": availability_fact,
        "residence_time": residence_time,
        "tmin_crit": tmin_crit,
        "leaf_tab": leaf_tab,
        "pheno_type": pheno_type,
        "maxdia": maxdia,
        "min_leaf_age_for_senescence": min_leaf_age_for_senescence,
        "gdd_senescence": gdd_senescence,
        "senescence_temp": senescence_temp,
        "hum_frac": hum_frac,
        "senescence_hum": senescence_hum,
        "nosenescence_hum": nosenescence_hum,
        "max_turnover_time": max_turnover_time,
        "min_turnover_time": min_turnover_time,
        "leaffall": leaffall,
        "lai_max": lai_max,
        "leafagecrit": leafagecrit,
        "lai_initmin": lai_initmin,
        "tau_fruit": tau_fruit,
        "tau_sap": tau_sap,
        "sla_age1": sla_age1,
        "sla_max": sla_max,
        "sla_min": sla_min,
        "lpj_gap_const_mort": lpj_gap_const_mort,
        "ok_dgvm": ok_dgvm,
        "dt_days": dt_days,
        "min_stomate": MIN_STOMATE,
        "vcmax25": vcmax25,
        "n_limfert": n_limfert_value,
        "leaf_timecst": leaf_timecst,
        "vmax_leafagecrit": leafagecrit,
        "vmax_pheno_type": pheno_type,
        "vmax_leaf_tab": leaf_tab,
        "vmax_ok_laidev": ok_laidev,
        "wire_vmax": True,
        "ok_nlim_vmax": grm_n_limitation,
    }
    maintenance_inputs = {
        "gpp_daily_current": state.gpp_daily,
        "dt_sechiba": dt_sechiba,
        "dt_stomate": dt_stomate,
        "do_slow": do_slow,
        "t2m": t2m,
        "t2m_longterm": t2m_longterm,
        "z_soil": z_soil,
        "rprof": rprof,
        "sla_calc": state.sla_calc,
        "coeff_maint_zero": coeff_maint_zero,
        "maint_resp_slope": maint_resp_slope,
        "ext_coeff": ext_coeff,
        "is_tree": is_tree,
        "resp_maint_part_current": state.resp_maint_part,
    }
    daily_process_inputs = {
        "gpp_daily": gpp_daily_value,
        "t2m_daily": t2m_daily_value,
        "t2m_min_daily": t2m_min_daily_value,
        "resp_maint_part": resp_maint_part_value,
    }
    if resp_maint_radia_value is not None:
        daily_process_inputs["resp_maint_radia"] = resp_maint_radia_value
    if flood_root_radia_value is not None:
        daily_process_inputs["flood_root_radia"] = flood_root_radia_value
    if snowfall_daily_value is not None:
        daily_process_inputs["snowfall_daily"] = snowfall_daily_value
    if snowmass_daily_value is not None:
        daily_process_inputs["snowmass_daily"] = snowmass_daily_value
    if tmc_topgrass_daily_value is not None:
        daily_process_inputs["tmc_topgrass_daily"] = tmc_topgrass_daily_value
    return StomateRestartInputBundles(
        prescribe_inputs=prescribe_inputs,
        constraints_inputs=constraints_inputs,
        phenology_inputs=phenology_inputs,
        alloc_inputs=alloc_inputs,
        post_npp_inputs=post_npp_inputs,
        maintenance_inputs=maintenance_inputs,
        daily_process_inputs=daily_process_inputs,
    )


def local_diffuco_sechiba_stomate_pft14_explicit_step(
    *,
    diffuco_inputs: Mapping[str, object],
    pft_output_backgrounds: Mapping[str, object],
    diffuco_passthrough: Mapping[str, object] | None = None,
    enerbil_inputs: Mapping[str, object],
    hydrol_inputs: Mapping[str, object],
    hydrol_diagnostic_inputs: Mapping[str, object],
    condveg_inputs: Mapping[str, object],
    thermosoil_inputs: Mapping[str, object],
    slowproc_inputs: Mapping[str, object],
    driver_entry_source: Mapping[str, object],
    extra_entry_sources: tuple[Mapping[str, object], ...] = (),
    gpp_daily_current=None,
    dt_sechiba=None,
    dt_stomate=None,
    do_slow=None,
    t2m=None,
    t2m_longterm=None,
    z_soil=None,
    rprof=None,
    sla_calc=None,
    coeff_maint_zero=None,
    maint_resp_slope=None,
    ext_coeff=None,
    is_tree=None,
    resp_maint_part_current=None,
    prescribe_inputs: Mapping[str, object] | None = None,
    constraints_inputs: Mapping[str, object] | None = None,
    alloc_inputs: Mapping[str, object] | None = None,
    post_npp_inputs: Mapping[str, object] | None = None,
    phenology_inputs: Mapping[str, object] | None = None,
    constraints_kwargs: Mapping[str, object] | None = None,
    phenology_kwargs: Mapping[str, object] | None = None,
    allocation_kwargs: Mapping[str, object] | None = None,
    flood_frac=None,
    one_day=86400.0,
) -> LocalDiffucoSechibaStomateExplicitResult:
    """Run the current explicit local DIFFUCO -> SECHIBA -> STOMATE chain.

    Fortran provenance: ``sechiba_main`` calls ``diffuco_main`` before
    ``enerbil_main`` at lines 997-1019, then HYDROL/CONDVEG/THERMOSOIL/
    SLOWPROC through lines 1049-1216. The resulting slowproc boundary is wired
    into ``stomate_main``/``StomateLpj`` by
    ``sechiba_stomate_pft14_explicit_step``.
    """

    missing_stomate = tuple(
        name
        for name, value in {
            "gpp_daily_current": gpp_daily_current,
            "dt_sechiba": dt_sechiba,
            "dt_stomate": dt_stomate,
            "do_slow": do_slow,
            "t2m": t2m,
            "t2m_longterm": t2m_longterm,
            "z_soil": z_soil,
            "rprof": rprof,
            "sla_calc": sla_calc,
            "coeff_maint_zero": coeff_maint_zero,
            "maint_resp_slope": maint_resp_slope,
            "ext_coeff": ext_coeff,
            "is_tree": is_tree,
            "resp_maint_part_current": resp_maint_part_current,
            "prescribe_inputs": prescribe_inputs,
            "constraints_inputs": constraints_inputs,
            "alloc_inputs": alloc_inputs,
            "post_npp_inputs": post_npp_inputs,
        }.items()
        if value is None
    )
    if missing_stomate:
        raise ValueError(f"missing STOMATE explicit-chain inputs: {missing_stomate}")

    local_sechiba = sechiba_explicit_coupled_step_from_local_diffuco(
        diffuco_inputs=diffuco_inputs,
        pft_output_backgrounds=pft_output_backgrounds,
        diffuco_passthrough=diffuco_passthrough,
        enerbil_inputs=enerbil_inputs,
        hydrol_inputs=hydrol_inputs,
        hydrol_diagnostic_inputs=hydrol_diagnostic_inputs,
        condveg_inputs=condveg_inputs,
        thermosoil_inputs=thermosoil_inputs,
        slowproc_inputs=slowproc_inputs,
    )
    coupled = sechiba_stomate_pft14_explicit_step(
        sechiba=local_sechiba.step,
        driver_entry_source=driver_entry_source,
        extra_entry_sources=extra_entry_sources,
        gpp_daily_current=gpp_daily_current,
        dt_sechiba=dt_sechiba,
        dt_stomate=dt_stomate,
        do_slow=do_slow,
        t2m=t2m,
        t2m_longterm=t2m_longterm,
        z_soil=z_soil,
        rprof=rprof,
        sla_calc=sla_calc,
        coeff_maint_zero=coeff_maint_zero,
        maint_resp_slope=maint_resp_slope,
        ext_coeff=ext_coeff,
        is_tree=is_tree,
        resp_maint_part_current=resp_maint_part_current,
        prescribe_inputs=prescribe_inputs,
        constraints_inputs=constraints_inputs,
        phenology_inputs=phenology_inputs,
        alloc_inputs=alloc_inputs,
        post_npp_inputs=post_npp_inputs,
        flood_frac=flood_frac,
        constraints_kwargs=constraints_kwargs,
        phenology_kwargs=phenology_kwargs,
        allocation_kwargs=allocation_kwargs,
        one_day=one_day,
    )
    return LocalDiffucoSechibaStomateExplicitResult(
        local_sechiba=local_sechiba,
        coupled=coupled,
    )
