"""STOMATE entry payload contracts backed by the slowproc bridge trace.

This module only describes and reads explicit values at the
`slowproc_main -> stomate_main` boundary. It does not implement STOMATE
processes and does not derive missing state from nearby sentinels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jax_orchidee.driver.sechiba_boundary import IntersurfFirstStepPayload
from jax_orchidee.sechiba.hydrol import HydrolModuleDiagnostics, HydrolModuleOutputs
from jax_orchidee.sechiba.slowproc import (
    SlowprocDynPeatDisabledEntryState,
    SlowprocDerivvarResult,
    SlowprocErosionDailyZeroEntryState,
    SlowprocFireDisabledEntryState,
    SlowprocNoLccEntryState,
    SlowprocRestartEntryState,
    SlowprocStaticEntryState,
    SlowprocThermosoilEntryInitState,
)
from jax_orchidee.stomate.reference import StomateRestartEntryState
from jax_orchidee.stomate.daily_inputs import daily_input_availability_from_entry
from jax_orchidee.trace.server_1961 import TRACE_ROOT, read_server_records


ENTRY_TRACE_NAME = "sechiba_bridge_slowproc"
ENTRY_TRACE_TAG = "before_stomate_main"
PFT14_FORTRAN_INDEX = 14

ENTRY_PROVENANCE = (
    "Trace provenance: outputs/server_1961_bridge_trace_20260624/traces/"
    "orchjax_sechiba_bridge_slowproc_trace.txt:before_stomate_main.",
    "Trace contract: docs/source_audits/server_1961_bridge_trace_patch_plan.md#before_stomate_main.",
    "Fortran source truth: src_sechiba/slowproc.f90, slowproc_main lines 973-985 "
    "passes the final slowproc boundary into stomate_main.",
    "Fortran source truth: src_stomate/stomate.f90, stomate_main signature "
    "lines 2409-2448 and input declarations lines 2483-2495, 2623-2626.",
)

DAILY_ACCUMULATION_PROVENANCE = (
    "Fortran source truth: src_stomate/stomate.f90, stomate_main lines 2918, "
    "3022-3031, and 3198-3208.",
    "JAX helper: jax_orchidee.stomate.daily.stomate_accumulate_daily implements "
    "stomate_accu scheduling only; callers must supply exact increments.",
)

ENTRY_NORMALIZATION_PROVENANCE = (
    "Fortran source truth: src_stomate/stomate.f90, stomate_main lines "
    "2921-2931 normalizes veget and veget_max with current totfrac_nobio.",
    "Fortran source truth: src_stomate/stomate.f90, stomate_main lines "
    "2980-2990 normalizes vegetnew_firstday with totfrac_nobio_new on date==1.",
    "Fortran source truth: src_stomate/stomate.f90, stomate_main lines "
    "2993-3016 normalizes veget_max_new with totfrac_nobio_new for land-cover "
    "change and dynamic peat branches.",
    "Fortran source truth: src_sechiba/slowproc.f90, slowproc_main lines "
    "1116-1121 computes tot_bare_soil separately; it must not substitute for "
    "totfrac_nobio or totfrac_nobio_new.",
)

PROCESS_BOUNDARY_PROVENANCE = (
    "Fortran source truth: src_stomate/stomate.f90, stomate_main maintenance "
    "call and accumulation lines 3244-3267.",
    "Fortran source truth: src_stomate/stomate.f90, stomate_main StomateLpj "
    "call boundary lines 4297-4335.",
    "Audit: docs/source_audits/stomate_daily_trace_contract.md and "
    "docs/source_audits/stomate_phase1c_state_contract.md.",
)

LITTERCALC_ENTRY_PROVENANCE = (
    "Fortran source truth: src_stomate/stomate.f90, stomate_main lines "
    "3270-3275 constructs soil_mc_32l from soil_mc.",
    "Fortran source truth: src_stomate/stomate.f90, stomate_main lines "
    "3288-3293 scales turnover_daily and bm_to_litter before littercalc_leak.",
    "Fortran source truth: src_stomate/stomate_litter.f90, littercalc_leak "
    "signature lines 1637-1644 consumes those prepared inputs.",
)

SOILCARBON_LEAK_PROVENANCE = (
    "Fortran source truth: src_stomate/stomate.f90, stomate_main lines "
    "3384-3403 calls soilcarbon_leak on the OK_LEAK path.",
    "Fortran source truth: src_stomate/stomate.f90, stomate_main lines "
    "3415-3458 aggregates DOC_EXP and DIC exports.",
    "Fortran source truth: src_stomate/stomate_soilcarbon.f90, "
    "soilcarbon_leak signature lines 641-657 and TF-DOC lines 1206-1233.",
    "Fortran source truth: src_stomate/stomate_soilcarbon.f90, "
    "altcalc_DOC lines 2438-2578.",
)

PERMAFROST_CONTROL_PROVENANCE = (
    "Fortran source truth: src_stomate/stomate.f90, stomate_main lines "
    "3034-3124 prepares peat moisture/deep humidity, calls microactem, and "
    "converts returned time constants into day^-1 controls.",
    "Fortran source truth: src_stomate/stomate_permafrost_soilcarbon.f90, "
    "microactem lines 2468-2703 computes temperature and moisture controls.",
    "Fortran source truth: src_parameters/constantes_soil_var.f90 lines "
    "174-175 defines tau_peat and z_tau defaults; paper run overrides are in "
    "configs/orchidee_man_250919.yaml structural_overrides.",
)

DEEP_CARBON_PROVENANCE = (
    "Fortran source truth: src_stomate/stomate.f90, stomate_main lines "
    "3336-3383 and 3628-3655 call deep_carbcycle under OK_PC.",
    "Fortran source truth: src_stomate/stomate_permafrost_soilcarbon.f90, "
    "carbinput lines 3184-3410, permafrost_decomp lines 3920-4403, and "
    "calc_vert_int_soil_carbon lines 4425-4463.",
)

DAILY_SCHEDULING_PROVENANCE = (
    "Fortran source truth: src_stomate/stomate.f90, stomate_main lines "
    "3128-3168 computes EndOfMonth, st2m, ugdh, and uphoi.",
    "Fortran source truth: src_stomate/stomate.f90, stomate_main lines "
    "3187-3240 accumulates crop, forcing, snow, peat, permafrost, extrema, "
    "and wind daily fields.",
    "Fortran source truth: src_stomate/stomate.f90, stomate_accu_r1d/r2d/r3d "
    "lines 9341-9411 defines the accumulation/mean formula.",
)

FIRST_STEP_PFT14_ENTRY_FIELDS = (
    "gpp",
    "lai",
    "veget",
    "veget_max",
    "t2m",
    "temp_sol",
    "humrel",
    "litterhumdiag",
    "precip",
    "precip_rain",
    "precip_snow",
    "snow",
    "wtp",
    "fwet_new",
    "soil_mc_top_tile",
    "wat_flux_top_tile",
)


STOMATE_MAIN_ARGUMENT_PROVENANCE = (
    "Fortran source truth: src_sechiba/slowproc.f90, slowproc_main lines "
    "973-1023 calls stomate_main.",
    "Fortran source truth: src_stomate/stomate.f90, stomate_main signature "
    "lines 2409-2448 and declarations lines 2462-2743.",
    "Trace provenance: orchjax_sechiba_bridge_slowproc_trace.txt:"
    "before_stomate_main.",
)


@dataclass(frozen=True)
class EntryField:
    """One requested entry field and whether it is explicit in the trace."""

    name: str
    status: str
    trace_fields: tuple[str, ...]
    value: object | None
    provenance: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class HelperInputContract:
    """Mapping from entry payload fields to an existing STOMATE helper input."""

    helper: str
    helper_input: str
    status: str
    entry_fields: tuple[str, ...]
    provenance: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class StomateMainArgumentContract:
    """One argument in ``slowproc_main``'s call to ``stomate_main``."""

    position: int
    name: str
    direction: str
    source: str
    status: str
    trace_fields: tuple[str, ...]
    provenance: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class StomateMainPayloadAssembly:
    """Explicit payload assembled for the ``stomate_main`` entry boundary."""

    payload: dict[str, object]
    contracts: tuple[StomateMainArgumentContract, ...]
    covered_inputs: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    partial_inputs: tuple[str, ...]
    output_arguments: tuple[str, ...]
    missing_by_source: dict[str, tuple[str, ...]]
    partial_by_source: dict[str, tuple[str, ...]]
    provenance: tuple[str, ...] = STOMATE_MAIN_ARGUMENT_PROVENANCE

    @property
    def ready(self) -> bool:
        return not self.missing_inputs and not self.partial_inputs

    def blocking_missing_inputs(self, *, ignored_sources: tuple[str, ...] = ("io_handle",)) -> tuple[str, ...]:
        """Return missing pre-entry inputs that are not deliberately external.

        Fortran provenance: IO handles are the history/restart identifiers in
        `src_stomate/stomate.f90::stomate_main` lines 2423-2427. They are not
        ecological process state and are supplied by the driver/output layer,
        so process-closure checks may ignore the `io_handle` source while still
        reporting any missing model-state inputs.
        """

        ignored = set(ignored_sources)
        ignored_inputs = {
            name
            for source, names in self.missing_by_source.items()
            if source in ignored
            for name in names
        }
        return tuple(name for name in self.missing_inputs if name not in ignored_inputs)


@dataclass(frozen=True)
class StomateEntryPayload:
    """One parsed PFT14 `before_stomate_main` bridge trace record."""

    record: dict[str, object]
    provenance: tuple[str, ...] = ENTRY_PROVENANCE

    @property
    def kjit(self) -> int:
        return int(self.record["kjit"])

    @property
    def ji(self) -> int:
        return int(self.record["ji"])

    @property
    def jv(self) -> int:
        return int(self.record["jv"])

    @property
    def available_fields(self) -> tuple[str, ...]:
        return tuple(sorted(key for key in self.record if key not in {"tag", "start_line"}))

    def require(self, name: str) -> object:
        """Return an explicit trace field or fail without deriving it."""

        if name not in self.record:
            raise KeyError(f"STOMATE entry trace has no explicit field {name!r}")
        return self.record[name]


# Names are intentionally kept in the Fortran call order from slowproc_main.
_STOMATE_MAIN_ARGUMENTS: tuple[tuple[str, str, str], ...] = (
    ("kjit", "input", "driver_time"),
    ("kjpij", "input", "driver_domain"),
    ("kjpindex", "input", "driver_domain"),
    ("index", "input", "driver_domain"),
    ("lalo", "input", "driver_static"),
    ("neighbours", "input", "driver_static"),
    ("resolution", "input", "driver_static"),
    ("contfrac", "input", "driver_static"),
    ("totfrac_nobio", "input", "slowproc_main"),
    ("clay", "input", "driver_static"),
    ("t2m", "input", "forcing"),
    ("t2m_min", "input", "forcing"),
    ("temp_sol", "input", "enerbil"),
    ("stempdiag", "input", "thermosoil"),
    ("humrel", "input", "hydrol"),
    ("shumdiag", "input", "hydrol"),
    ("litterhumdiag", "input", "hydrol"),
    ("precip_rain", "input", "forcing"),
    ("precip_snow", "input", "forcing"),
    ("wspeed", "input", "forcing"),
    ("lightn", "input", "slowproc_read_data"),
    ("popd", "inout", "slowproc_read_annual"),
    ("read_observed_ba", "input", "run_control"),
    ("observed_ba", "input", "slowproc_read_data"),
    ("humign", "inout", "slowproc_read_annual"),
    ("read_cf_fine", "input", "run_control"),
    ("cf_fine", "input", "slowproc_read_data"),
    ("read_cf_coarse", "input", "run_control"),
    ("cf_coarse", "input", "slowproc_read_data"),
    ("read_ratio_flag", "input", "run_control"),
    ("ratio_flag", "input", "slowproc_read_data"),
    ("read_ratio", "input", "run_control"),
    ("ratio", "input", "slowproc_read_data"),
    ("gpp", "input", "diffuco"),
    ("deadleaf_cover", "inout", "slowproc_stomate_state"),
    ("assim_param", "inout", "slowproc_stomate_state"),
    ("lai", "inout", "slowproc_stomate_state"),
    ("frac_age", "inout", "slowproc_stomate_state"),
    ("height", "inout", "slowproc_stomate_state"),
    ("veget", "input", "slowproc_veget"),
    ("veget_max", "inout", "slowproc_veget"),
    ("veget_max_new", "input", "slowproc_land_cover"),
    ("vegetnew_firstday", "inout", "slowproc_land_cover"),
    ("totfrac_nobio_new", "input", "slowproc_main"),
    ("glccNetLCC", "inout", "slowproc_land_cover"),
    ("glccSecondShift", "inout", "slowproc_land_cover"),
    ("glccPrimaryShift", "inout", "slowproc_land_cover"),
    ("harvest_matrix", "inout", "slowproc_land_cover"),
    ("bound_spa", "inout", "slowproc_land_cover"),
    ("hist_id", "input", "io_handle"),
    ("hist2_id", "input", "io_handle"),
    ("rest_id_stom", "input", "io_handle"),
    ("hist_id_stom", "input", "io_handle"),
    ("hist_id_stom_IPCC", "input", "io_handle"),
    ("co2_flux", "output", "stomate_output"),
    ("fco2_lu", "output", "stomate_output"),
    ("resp_maint", "output", "stomate_output"),
    ("resp_hetero", "output", "stomate_output"),
    ("resp_growth", "output", "stomate_output"),
    ("temp_growth", "output", "stomate_output"),
    ("swdown", "input", "forcing"),
    ("t2m_max", "input", "forcing"),
    ("evapot_corr", "input", "enerbil"),
    ("tdeep", "input", "thermosoil"),
    ("hsdeep", "input", "thermosoil"),
    ("snow", "input", "hydrol_condveg"),
    ("heat_Zimov", "output", "stomate_output"),
    ("pb", "input", "forcing"),
    ("sfluxCH4_deep", "output", "stomate_output"),
    ("sfluxCO2_deep", "output", "stomate_output"),
    ("thawed_humidity", "input", "thermosoil"),
    ("depth_organic_soil", "input", "driver_static"),
    ("zz_deep", "input", "thermosoil_static"),
    ("zz_coef_deep", "input", "thermosoil_static"),
    ("soilc_total", "input", "stomate_state"),
    ("snowdz", "input", "hydrol_condveg"),
    ("snowrho", "input", "condveg"),
    ("EndOfYear", "input", "driver_time"),
    ("f_rot_sech", "output", "stomate_output"),
    ("rot_cmd", "output", "stomate_output"),
    ("t2mdiag", "input", "enerbil"),
    ("tmc_topgrass", "input", "hydrol"),
    ("fc_grazing", "input", "driver_static"),
    ("humcste_use", "input", "driver_static"),
    ("altmax", "inout", "stomate_state"),
    ("wtp", "input", "hydrol"),
    ("fwet_new", "input", "hydrol"),
    ("fpeat", "inout", "slowproc_peat_state"),
    ("shumdiag_peat", "input", "hydrol"),
    ("mc_peat_above", "input", "hydrol"),
    ("peatC", "output", "stomate_output"),
    ("sat_duration", "input", "slowproc_peat_state"),
    ("liqwt_ratio", "input", "hydrol"),
    ("veget_max_adjusted", "output", "stomate_output"),
    ("shumdiag_croppeat", "input", "hydrol"),
    ("mc_croppeat_above", "input", "hydrol"),
    ("shumdiag_man", "input", "hydrol"),
    ("mc_man_above", "input", "hydrol"),
    ("soil_mc", "input", "hydrol"),
    ("wat_flux", "input", "hydrol"),
    ("bulk_dens", "inout", "driver_static"),
    ("soil_ph", "input", "driver_static"),
    ("poor_soils", "inout", "driver_static"),
    ("drainage_per_soil", "input", "hydrol"),
    ("runoff_per_soil", "input", "hydrol"),
    ("runoff2peat", "input", "hydrol"),
    ("DOC_EXP_agg", "output", "stomate_output"),
    ("DOC_to_topsoil", "input", "hydrol"),
    ("DOC_to_subsoil", "input", "hydrol"),
    ("flood_frac", "input", "hydrol"),
    ("precip2canopy", "input", "hydrol"),
    ("precip2ground", "input", "hydrol"),
    ("canopy2ground", "input", "hydrol"),
    ("fastr", "input", "hydrol"),
    ("biomass", "inout", "stomate_state"),
    ("litter_above", "inout", "stomate_state"),
    ("litter_below", "inout", "stomate_state"),
    ("carbon_32l", "inout", "stomate_state"),
    ("DOC", "inout", "stomate_state"),
    ("lignin_struc_above", "inout", "stomate_state"),
    ("lignin_struc_below", "inout", "stomate_state"),
    ("erodepth", "input", "hydrol"),
    ("sed_deposition_d", "input", "hydrol"),
    ("poc_deposition_d", "input", "hydrol"),
)


_ENTRY_TRACE_FIELDS_BY_ARGUMENT: dict[str, tuple[str, ...]] = {
    "kjit": ("kjit",),
    "t2m": ("t2m",),
    "t2m_min": ("t2m_min",),
    "temp_sol": ("temp_sol",),
    "humrel": ("humrel",),
    "litterhumdiag": ("litterhumdiag",),
    "precip_rain": ("precip_rain",),
    "precip_snow": ("precip_snow",),
    "wspeed": ("wspeed",),
    "lightn": ("lightn",),
    "popd": ("popd",),
    "humign": ("humign",),
    "gpp": ("gpp",),
    "lai": ("lai",),
    "veget": ("veget",),
    "veget_max": ("veget_max",),
    "veget_max_new": ("veget_max_new",),
    "t2m_max": ("t2mdiag",),
    "evapot_corr": ("evapot_corr",),
    "tdeep": ("tdeep",),
    "hsdeep": ("hsdeep_long",),
    "snow": ("snow",),
    "snowdz": ("snowdz_1",),
    "snowrho": ("snowrho_1",),
    "t2mdiag": ("t2mdiag",),
    "wtp": ("wtp",),
    "fwet_new": ("fwet_new",),
    "fpeat": ("fpeat",),
    "shumdiag_peat": ("shumdiag_peat_1",),
    "mc_peat_above": ("mc_peat_above",),
    "liqwt_ratio": ("liqwt_ratio",),
    "shumdiag_croppeat": ("shumdiag_croppeat_1",),
    "mc_croppeat_above": ("mc_croppeat_above",),
    "shumdiag_man": ("shumdiag_man_1",),
    "mc_man_above": ("mc_man_above",),
    "soil_mc": ("soil_mc_top_tile",),
    "wat_flux": ("wat_flux_top_tile",),
    "drainage_per_soil": ("drainage_per_soil_tile",),
    "runoff_per_soil": ("runoff_per_soil_tile",),
    "runoff2peat": ("runoff2peat_tile",),
    "flood_frac": ("flood_frac",),
    "precip2canopy": ("precip2canopy",),
    "precip2ground": ("precip2ground",),
    "canopy2ground": ("canopy2ground",),
}


_PARTIAL_TRACE_ARGUMENT_NOTES = {
    "shumdiag_peat": "Trace contains layer-1 sentinel only, not the full nslm array.",
    "shumdiag_croppeat": "Trace contains layer-1 sentinel only, not the full nslm array.",
    "shumdiag_man": "Trace contains layer-1 sentinel only, not the full nslm array.",
    "soil_mc": "Trace contains the top-layer/tile sentinel only, not the full nslm by nstm array.",
    "wat_flux": "Trace contains the top-layer/tile sentinel only, not the full nslm by nstm array.",
    "drainage_per_soil": "Trace contains the selected tile sentinel only, not the full nstm array.",
    "runoff_per_soil": "Trace contains the selected tile sentinel only, not the full nstm array.",
    "runoff2peat": "Trace contains the selected tile sentinel only, not the full nstm array.",
    "snowdz": "Trace contains snow layer 1 only, not the full nsnow array.",
    "snowrho": "Trace contains snow layer 1 only, not the full nsnow array.",
}


def read_first_pft14_entry(
    *,
    root: str | Path = TRACE_ROOT,
    scan_limit: int = 256,
) -> StomateEntryPayload:
    """Read the first PFT14 `before_stomate_main` payload.

    The scan is based on `read_server_records`, and returns only an explicit
    bridge-trace record. It does not synthesize `precip`, `gpp_d`,
    `stempdiag`, `shumdiag`, or any STOMATE process state.
    """

    rows = read_server_records(ENTRY_TRACE_NAME, tags=ENTRY_TRACE_TAG, limit=scan_limit, root=root)
    for row in rows:
        if row.get("jv") == PFT14_FORTRAN_INDEX:
            return StomateEntryPayload(row)
    raise ValueError(f"No PFT{PFT14_FORTRAN_INDEX} {ENTRY_TRACE_TAG!r} record found")


def stomate_main_argument_contract(
    payload: StomateEntryPayload | None = None,
) -> tuple[StomateMainArgumentContract, ...]:
    """Audit ``slowproc_main``'s full ``stomate_main`` argument boundary.

    The returned order is the exact Fortran call order. ``covered`` means the
    first-step PFT14 entry trace has explicit scalar/PFT14 values for the
    argument. ``partial`` means the trace has only sentinels for a larger
    Fortran array. ``missing`` means the trace has no explicit value and callers
    must provide it from source-backed driver/static/restart/module state.
    ``output`` arguments are produced by STOMATE and are not pre-entry inputs.
    """

    record = {} if payload is None else payload.record
    result: list[StomateMainArgumentContract] = []
    for position, (name, direction, source) in enumerate(_STOMATE_MAIN_ARGUMENTS, start=1):
        trace_fields = _ENTRY_TRACE_FIELDS_BY_ARGUMENT.get(name, ())
        if direction == "output":
            status = "output"
            notes = "Produced by stomate_main; not a pre-entry input."
        elif name in _PARTIAL_TRACE_ARGUMENT_NOTES:
            status = "partial" if all(field in record for field in trace_fields) else "missing"
            notes = _PARTIAL_TRACE_ARGUMENT_NOTES[name]
        elif trace_fields:
            status = "covered" if all(field in record for field in trace_fields) else "missing"
            notes = ""
        else:
            status = "missing"
            notes = "No explicit before_stomate_main trace field; must be supplied from audited source-backed state."
        result.append(
            StomateMainArgumentContract(
                position=position,
                name=name,
                direction=direction,
                source=source,
                status=status,
                trace_fields=trace_fields,
                provenance=STOMATE_MAIN_ARGUMENT_PROVENANCE,
                notes=notes,
            )
        )
    return tuple(result)


def stomate_main_entry_gaps(
    contracts: tuple[StomateMainArgumentContract, ...],
) -> tuple[StomateMainArgumentContract, ...]:
    """Return pre-entry arguments not fully covered by the bridge trace."""

    return tuple(
        contract
        for contract in contracts
        if contract.direction != "output" and contract.status != "covered"
    )


def assemble_stomate_main_payload(
    *sources: dict[str, object],
) -> StomateMainPayloadAssembly:
    """Assemble explicit inputs for ``stomate_main`` without inferring values.

    Source mappings are merged left-to-right, so later sources may replace
    earlier values. Only exact Fortran argument names are accepted. A field is
    covered when its argument name is present in the merged mapping; trace
    sentinel aliases such as ``soil_mc_top_tile`` are deliberately ignored here.
    """

    merged: dict[str, object] = {}
    for source in sources:
        merged.update(dict(source))

    payload: dict[str, object] = {}
    contracts: list[StomateMainArgumentContract] = []
    covered: list[str] = []
    missing: list[str] = []
    partial: list[str] = []
    outputs: list[str] = []
    missing_by_source: dict[str, list[str]] = {}
    partial_by_source: dict[str, list[str]] = {}

    for position, (name, direction, source) in enumerate(_STOMATE_MAIN_ARGUMENTS, start=1):
        if direction == "output":
            status = "output"
            notes = "Produced by stomate_main; not a pre-entry input."
            outputs.append(name)
        elif name in merged:
            status = "covered"
            notes = ""
            payload[name] = merged[name]
            covered.append(name)
        else:
            status = "missing"
            notes = "No exact explicit value supplied for this Fortran argument."
            missing.append(name)
            missing_by_source.setdefault(source, []).append(name)
        contracts.append(
            StomateMainArgumentContract(
                position=position,
                name=name,
                direction=direction,
                source=source,
                status=status,
                trace_fields=(),
                provenance=STOMATE_MAIN_ARGUMENT_PROVENANCE,
                notes=notes,
            )
        )

    return StomateMainPayloadAssembly(
        payload=payload,
        contracts=tuple(contracts),
        covered_inputs=tuple(covered),
        missing_inputs=tuple(missing),
        partial_inputs=tuple(partial),
        output_arguments=tuple(outputs),
        missing_by_source={key: tuple(value) for key, value in missing_by_source.items()},
        partial_by_source={key: tuple(value) for key, value in partial_by_source.items()},
    )


def stomate_driver_entry_source(
    driver_payload: IntersurfFirstStepPayload,
    *,
    kjit,
    t2mdiag=None,
    temp_sol=None,
    min_wind=0.1,
    date0=None,
    end_of_year=False,
    hist_id=None,
    hist2_id=None,
    rest_id_stom=None,
    hist_id_stom=None,
    hist_id_stom_IPCC=None,
) -> dict[str, object]:
    """Map verified driver/SECHIBA boundary fields to STOMATE entry names.

    Fortran provenance: ``sechiba.f90::sechiba_main`` lines 1135 and
    1184-1216 compute ``wspeed`` and pass driver/static/forcing fields into
    ``slowproc_main``; ``slowproc.f90::slowproc_main`` lines 973-1023 forwards
    those fields to ``stomate_main``. Optional IO handles and ``date0`` are
    included only when supplied explicitly.
    """

    out: dict[str, object] = {
        "kjit": kjit,
        "kjpij": int(driver_payload.nbindex),
        "kjpindex": int(driver_payload.kjpindex),
        "index": driver_payload.kindex,
        "lalo": driver_payload.lalo,
        "contfrac": driver_payload.contfrac,
        "precip_rain": driver_payload.precip_rain,
        "precip_snow": driver_payload.precip_snow,
        "swdown": driver_payload.swdown,
        "pb": driver_payload.pb,
        "wspeed": np.maximum(
            float(min_wind),
            np.sqrt(np.asarray(driver_payload.u) * np.asarray(driver_payload.u) + np.asarray(driver_payload.v) * np.asarray(driver_payload.v)),
        ),
        "EndOfYear": bool(end_of_year),
    }
    if driver_payload.resolution is not None:
        out["resolution"] = driver_payload.resolution
    if driver_payload.neighbours is not None:
        out["neighbours"] = driver_payload.neighbours
    if driver_payload.clay_frac is not None:
        out["clay"] = driver_payload.clay_frac
    if driver_payload.bulk_dens is not None:
        out["bulk_dens"] = driver_payload.bulk_dens
    if driver_payload.soil_ph is not None:
        out["soil_ph"] = driver_payload.soil_ph
    if driver_payload.poor_soils is not None:
        out["poor_soils"] = driver_payload.poor_soils
    if t2mdiag is not None:
        out["t2m"] = t2mdiag
        out["t2m_min"] = t2mdiag
        out["t2m_max"] = t2mdiag
        out["t2mdiag"] = t2mdiag
    if temp_sol is not None:
        out["temp_sol"] = temp_sol
    for name, value in (
        ("date0", date0),
        ("hist_id", hist_id),
        ("hist2_id", hist2_id),
        ("rest_id_stom", rest_id_stom),
        ("hist_id_stom", hist_id_stom),
        ("hist_id_stom_IPCC", hist_id_stom_IPCC),
    ):
        if value is not None:
            out[name] = value
    return out


def stomate_enerbil_entry_source(
    *,
    t2mdiag=None,
    evapot_corr=None,
    temp_sol=None,
) -> dict[str, object]:
    """Map ENERBIL/surface-temperature fields to STOMATE entry names.

    Fortran provenance: ``sechiba.f90::sechiba_main`` lines 1013-1019 computes
    ``t2mdiag`` and ``evapot_corr`` in ``enerbil_main``; lines 1184-1198 pass
    ``t2mdiag``, ``temp_sol``, and ``evapot_corr`` into ``slowproc_main``.
    ``sechiba_end`` lines 3133-3135 copies ``temp_sol_new`` to ``temp_sol``
    only after the same-step slowproc/STOMATE call, so callers must pass an
    explicit same-step ``temp_sol`` source rather than substituting
    ``temp_sol_new``.
    """

    out: dict[str, object] = {}
    if t2mdiag is not None:
        out["t2m"] = t2mdiag
        out["t2m_min"] = t2mdiag
        out["t2m_max"] = t2mdiag
        out["t2mdiag"] = t2mdiag
    if evapot_corr is not None:
        out["evapot_corr"] = evapot_corr
    if temp_sol is not None:
        out["temp_sol"] = temp_sol
    return out


def stomate_diffuco_entry_source(
    *,
    gpp=None,
    **_ignored,
) -> dict[str, object]:
    """Map exact DIFFUCO exports to ``stomate_main`` entry names.

    Fortran provenance: ``src_sechiba/sechiba.f90::sechiba_main`` lines
    997-1005 calls ``diffuco_main`` with ``gpp`` as an output and lines
    1184-1192 forwards that same ``gpp`` into ``slowproc_main``; the
    ``slowproc_main`` call at lines 973-981 forwards it into
    ``stomate_main``.
    """

    out: dict[str, object] = {}
    if gpp is not None:
        out["gpp"] = gpp
    return out


def stomate_hydrol_entry_source(
    *,
    diagnostics: HydrolModuleDiagnostics,
    outputs: HydrolModuleOutputs,
    wat_flux=None,
    flood_frac_plus_streamfl_frac=None,
) -> dict[str, object]:
    """Map source-backed HYDROL exports to ``stomate_main`` argument names.

    Fortran provenance: ``sechiba.f90::sechiba_main`` lines 1184-1216 passes
    HYDROL diagnostics and soil-tile flux arrays into ``slowproc_main``;
    the first moisture-stress argument is ``vegstress`` in ``sechiba_main``
    line 1187, renamed to ``humrel`` by ``slowproc_main`` lines 343-399,
    then forwarded to ``stomate_main`` at lines 973-985. ``wat_flux`` is
    included only when the caller supplies the exact ``hydrol.f90`` ``qflux``
    export assigned at line 7363.
    ``flood_frac`` is included only when the caller supplies the exact
    ``sechiba.f90`` ``flood_frac + streamfl_frac`` expression from line 1212.
    """

    out = {
        "humrel": diagnostics.vegstress,
        "shumdiag": diagnostics.shumdiag,
        "litterhumdiag": diagnostics.litterhumdiag,
        "tmc_topgrass": diagnostics.tmc_topgrass,
        "shumdiag_peat": diagnostics.shumdiag_peat,
        "mc_peat_above": diagnostics.mc_peat_above,
        "shumdiag_croppeat": diagnostics.shumdiag_croppeat,
        "mc_croppeat_above": diagnostics.mc_croppeat_above,
        "shumdiag_man": diagnostics.shumdiag_man,
        "mc_man_above": diagnostics.mc_man_above,
        "wtp": diagnostics.wtp,
        "liqwt_ratio": diagnostics.liqwt_ratio,
        "soil_mc": diagnostics.mc_layh_s,
        "drainage_per_soil": outputs.drainage_per_soil,
        "runoff_per_soil": outputs.runoff_per_soil,
        "runoff2peat": outputs.runoff2peat,
        "precip2canopy": outputs.precip2canopy,
        "precip2ground": outputs.precip2ground,
        "canopy2ground": outputs.canopy2ground,
    }
    if wat_flux is None and hasattr(outputs, "wat_flux"):
        wat_flux = outputs.wat_flux
    if wat_flux is not None:
        out["wat_flux"] = wat_flux
    if getattr(diagnostics, "fwet_new", None) is not None:
        out["fwet_new"] = diagnostics.fwet_new
    if flood_frac_plus_streamfl_frac is not None:
        out["flood_frac"] = flood_frac_plus_streamfl_frac
    return out


def stomate_snow_entry_source(
    *,
    snow=None,
    snowdz=None,
    snowrho=None,
    **_ignored,
) -> dict[str, object]:
    """Map exact same-step snow state to ``stomate_main`` argument names.

    Fortran provenance: ``src_sechiba/sechiba.f90::sechiba_main`` lines
    1085-1089 passes ``snowdz``/``snowrho`` to ``condveg_main`` and lines
    1184-1201 forwards ``snow``, ``snowdz``, and ``snowrho`` through
    ``slowproc_main``; ``src_sechiba/slowproc.f90`` lines 990-994 forwards
    the same arrays into ``stomate_main``.
    """

    out: dict[str, object] = {}
    if snow is not None:
        out["snow"] = snow
    if snowdz is not None:
        out["snowdz"] = snowdz
    if snowrho is not None:
        out["snowrho"] = snowrho
    return out


def stomate_no_routing_entry_source(
    *,
    kjpindex,
    nflow,
    river_routing,
    nbp_glo,
    dtype=np.float64,
) -> dict[str, object]:
    """Map the no-routing SECHIBA branch into STOMATE entry arguments.

    Fortran provenance: ``sechiba.f90::sechiba_initialize`` lines 749-769
    zeroes routing state when ``.NOT.(river_routing .AND. nbp_glo .GT. 1)``;
    ``sechiba.f90::sechiba_main`` lines 1227-1259 applies the same no-routing
    branch after ``slowproc_main`` and zeroes DOC routing inputs. The
    ``slowproc_main`` call at lines 1184-1216 therefore receives zero
    ``flood_frac + streamfl_frac``, ``DOC_to_topsoil``, ``DOC_to_subsoil``, and
    ``fastr`` for single-point/no-routing runs.
    """

    if bool(river_routing) and int(nbp_glo) > 1:
        raise ValueError("routing state must be supplied explicitly when river_routing and nbp_glo > 1")
    kjpindex = int(kjpindex)
    nflow = int(nflow)
    return {
        "DOC_to_topsoil": np.zeros((kjpindex, nflow), dtype=dtype),
        "DOC_to_subsoil": np.zeros((kjpindex, nflow), dtype=dtype),
        "flood_frac": np.zeros((kjpindex,), dtype=dtype),
        "fastr": np.zeros((kjpindex,), dtype=dtype),
    }


def stomate_restart_entry_source(
    state: StomateRestartEntryState,
) -> dict[str, object]:
    """Map restart-backed STOMATE state to exact ``stomate_main`` arguments.

    Fortran provenance: ``src_stomate/stomate_io.f90::readstart`` lines
    919-940, 1227-1232, 1466-1471, 1512-1563, and 1737-1743 read the restart
    variables; ``src_sechiba/slowproc.f90::slowproc_main`` lines 973-1013
    passes the corresponding state arrays into ``stomate_main``. This mapper
    deliberately exposes only exact ``stomate_main`` argument names. Restart
    fields such as ``resp_maint_part``, ``leaf_age``, ``leaf_frac``, ``age``,
    ``sla_calc``, and ``PFTpresent`` remain available on ``state`` for later
    STOMATE process kernels, but are not call-boundary arguments here.
    """

    return {
        "assim_param": state.assim_param,
        "altmax": state.altmax,
        "fpeat": state.fpeat,
        "biomass": state.biomass,
        "litter_above": state.litter_above,
        "litter_below": state.litter_below,
        "carbon_32l": state.carbon_32l,
        "DOC": state.DOC,
        "lignin_struc_above": state.lignin_struc_above,
        "lignin_struc_below": state.lignin_struc_below,
        "soilc_total": state.soilc_total,
        "thawed_humidity": state.thawed_humidity,
        "depth_organic_soil": state.depth_organic_soil,
    }


def stomate_slowproc_restart_entry_source(
    state: SlowprocRestartEntryState,
) -> dict[str, object]:
    """Map SECHIBA restart SLOWPROC state to ``stomate_main`` arguments.

    Fortran provenance: ``src_sechiba/slowproc.f90::slowproc_init`` lines
    1685-1707 reads ``lai``, ``height``, and ``frac_age`` from the SECHIBA
    restart; ``slowproc_main`` lines 973-985 passes these arrays, together
    with ``veget`` and ``veget_max``, into ``stomate_main``. This source does
    not synthesize missing restart fallbacks.
    """

    return {
        "lai": state.lai,
        "frac_age": state.frac_age,
        "height": state.height,
        "veget": state.veget,
        "veget_max": state.veget_max,
        "totfrac_nobio": np.sum(state.frac_nobio, axis=1),
    }


def stomate_slowproc_derivvar_entry_source(
    derivvar: SlowprocDerivvarResult,
    *,
    include_assim_param=False,
) -> dict[str, object]:
    """Map closed ``slowproc_derivvar`` outputs to STOMATE entry names.

    Fortran provenance: ``src_sechiba/slowproc.f90::slowproc_derivvar`` lines
    2625-2673. ``assim_param`` is included only on explicit request because
    STOMATE restart ``readstart`` is a more direct source when available.
    """

    out: dict[str, object] = {
        "deadleaf_cover": derivvar.deadleaf_cover,
        "height": derivvar.height,
    }
    if include_assim_param:
        out["assim_param"] = derivvar.assim_param
    return out


def stomate_slowproc_no_lcc_entry_source(
    state: SlowprocNoLccEntryState,
) -> dict[str, object]:
    """Map explicit no-LCC SLOWPROC state to STOMATE entry names.

    Fortran provenance: ``src_sechiba/slowproc.f90::slowproc_init`` lines
    1449-1452 and 1518-1536; ``slowproc_main`` lines 943-955.
    """

    return {
        "veget_max_new": state.veget_max_new,
        "vegetnew_firstday": state.vegetnew_firstday,
        "totfrac_nobio_new": state.totfrac_nobio_new,
        "glccNetLCC": state.glccNetLCC,
        "glccSecondShift": state.glccSecondShift,
        "glccPrimaryShift": state.glccPrimaryShift,
        "harvest_matrix": state.harvest_matrix,
        "bound_spa": state.bound_spa,
    }


def stomate_slowproc_fire_disabled_entry_source(
    state: SlowprocFireDisabledEntryState,
) -> dict[str, object]:
    """Map ``FIRE_DISABLE=y`` SPITFIRE no-op state to STOMATE entry names.

    Fortran provenance: ``src_sechiba/slowproc.f90::slowproc_init`` lines
    1509-1576 and 2281-2381; ``slowproc_main`` lines 925-933 and 973-979.
    """

    return {
        "lightn": state.lightn,
        "popd": state.popd,
        "read_observed_ba": state.read_observed_ba,
        "observed_ba": state.observed_ba,
        "humign": state.humign,
        "read_cf_fine": state.read_cf_fine,
        "cf_fine": state.cf_fine,
        "read_cf_coarse": state.read_cf_coarse,
        "cf_coarse": state.cf_coarse,
        "read_ratio_flag": state.read_ratio_flag,
        "ratio_flag": state.ratio_flag,
        "read_ratio": state.read_ratio,
        "ratio": state.ratio,
    }


def stomate_slowproc_dyn_peat_disabled_entry_source(
    state: SlowprocDynPeatDisabledEntryState,
) -> dict[str, object]:
    """Map branch-inactive ``DYN_PEAT=n`` peat input to STOMATE entry names.

    Fortran provenance: ``src_sechiba/slowproc.f90::slowproc_main`` lines
    616-646 and 850-918; ``src_stomate/stomate.f90::stomate_main`` lines
    3005-3016 and 3618-3628.
    """

    return {
        "sat_duration": state.sat_duration,
    }


def stomate_thermosoil_entry_source(
    *,
    stempdiag=None,
    tdeep=None,
    hsdeep=None,
    thawed_humidity=None,
    **_ignored,
) -> dict[str, object]:
    """Map explicit THERMOSOIL outputs to STOMATE entry names.

    Fortran provenance: ``src_sechiba/sechiba.f90::sechiba_main`` lines
    1109-1118 calls ``thermosoil_main`` and lines 1197-1201 forwards
    thermosoil fields through ``slowproc_main``; ``src_sechiba/slowproc.f90``
    lines 990-994 forwards the same fields into ``stomate_main``. This mapper
    only accepts caller-supplied explicit values and ignores nearby thermosoil
    diagnostics that are not ``stomate_main`` arguments.
    """

    out: dict[str, object] = {}
    for name, value in (
        ("stempdiag", stempdiag),
        ("tdeep", tdeep),
        ("hsdeep", hsdeep),
        ("thawed_humidity", thawed_humidity),
    ):
        if value is not None:
            out[name] = value
    return out


def stomate_thermosoil_static_entry_source(
    *,
    zz_deep=None,
    zz_coef_deep=None,
    **_ignored,
) -> dict[str, object]:
    """Map explicit deep thermodynamic vertical levels to STOMATE names.

    Fortran provenance: ``src_sechiba/sechiba.f90::sechiba_init`` lines
    2808-2809 sets ``zz_deep=znt`` and ``zz_coef_deep=zlt``; ``slowproc_main``
    lines 993-994 forwards them into ``stomate_main``.
    """

    out: dict[str, object] = {}
    if zz_deep is not None:
        out["zz_deep"] = zz_deep
    if zz_coef_deep is not None:
        out["zz_coef_deep"] = zz_coef_deep
    return out


def stomate_erosion_entry_source(
    *,
    erodepth=None,
    sed_deposition_d=None,
    poc_deposition_d=None,
    **_ignored,
) -> dict[str, object]:
    """Map explicit erosion boundary fields to STOMATE entry names.

    Fortran provenance: ``src_sechiba/erosion.f90::erosion_main`` lines
    163-230 declares these boundary fields and ``src_sechiba/sechiba.f90``
    lines 1214-1224 wires them around the STOMATE/erosion sequence. This
    helper does not derive erosion state from sediment export diagnostics.
    """

    out: dict[str, object] = {}
    for name, value in (
        ("erodepth", erodepth),
        ("sed_deposition_d", sed_deposition_d),
        ("poc_deposition_d", poc_deposition_d),
    ):
        if value is not None:
            out[name] = value
    return out


def stomate_erosion_disabled_erodepth_entry_source(
    *,
    erosion_module: bool,
    kjpindex: int,
    nvm: int,
    dtype=np.float64,
) -> dict[str, object]:
    """Map audited disabled-erosion ``erodepth`` into STOMATE entry.

    Fortran provenance: ``src_sechiba/sechiba.f90::sechiba_main`` lines
    1184-1216 passes ``erodepth`` into ``slowproc_main`` before the optional
    ``erosion_main`` call at lines 1219-1224. The paper run has
    ``EROSION_MODULE=FALSE`` in ``used_run.def``. Audited trace
    ``outputs/server_1961_erodepth_trace_20260628_0001/``
    ``orchjax_sechiba_bridge_erodepth_trace.txt`` records
    ``before_slowproc_erodepth`` for kjit 1-4 with
    ``erodepth(1,14)=MINVAL=MAXVAL=SUM(ABS)=0`` and shape ``(1,14)``.
    """

    if bool(erosion_module):
        raise ValueError("erodepth must be supplied from erosion_main when EROSION_MODULE is enabled")
    return {"erodepth": np.zeros((int(kjpindex), int(nvm)), dtype=dtype)}


def stomate_slowproc_thermosoil_entry_source(
    state: SlowprocThermosoilEntryInitState,
) -> dict[str, object]:
    """Map source-backed THERMOSOIL state to STOMATE entry names.

    Fortran provenance: ``src_sechiba/sechiba.f90::sechiba_init`` lines
    2704-2715 and 2808-2809 initialize ``tdeep``, ``hsdeep``,
    ``zz_deep`` and ``zz_coef_deep``; ``sechiba_main`` lines
    1198-1201 and ``slowproc_main`` lines 991-994 forward them into
    ``stomate_main``. This mapper deliberately omits ``heat_Zimov`` and
    ``sflux*`` because ``stomate_main`` declares them ``INTENT(out)``.
    """

    return {
        "tdeep": state.tdeep,
        "hsdeep": state.hsdeep,
        "zz_deep": state.zz_deep,
        "zz_coef_deep": state.zz_coef_deep,
    }


def stomate_slowproc_static_entry_source(
    state: SlowprocStaticEntryState,
) -> dict[str, object]:
    """Map source-backed SLOWPROC static fields to STOMATE entry names.

    Fortran provenance: ``src_sechiba/slowproc.f90::slowproc_soilt`` lines
    2190-2216 constructs ``fc_grazing``; ``src_sechiba/hydrol.f90`` lines
    4093-4107 constructs ``humcste_use``. ``slowproc_main`` lines 991-997
    forwards both fields into ``stomate_main``.
    """

    return {
        "fc_grazing": state.fc_grazing,
        "humcste_use": state.humcste_use,
    }


def stomate_slowproc_erosion_daily_zero_entry_source(
    state: SlowprocErosionDailyZeroEntryState,
) -> dict[str, object]:
    """Map zero daily erosion deposition accumulators to STOMATE entry names.

    Fortran provenance: ``src_sechiba/sechiba.f90::sechiba_init`` lines
    2461-2467 and ``src_sechiba/erosion.f90::erosion_main`` lines 633-653.
    ``erodepth`` is intentionally omitted because this source snapshot does
    not provide a normal-path zero initialization before the STOMATE call.
    """

    return {
        "sed_deposition_d": state.sed_deposition_d,
        "poc_deposition_d": state.poc_deposition_d,
    }


def first_step_pft14_entry_fields(payload: StomateEntryPayload) -> tuple[EntryField, ...]:
    """Describe requested first-step PFT14 entry fields.

    `precip` is reported as covered because Fortran's local increment is a
    source-audited function of entry fields `precip_rain`, `precip_snow`, and
    `dt_sechiba`; the trace still does not contain a literal `precip` field.
    """

    daily_inputs = daily_input_availability_from_entry(payload.record)
    result: list[EntryField] = []
    for name in FIRST_STEP_PFT14_ENTRY_FIELDS:
        if name in payload.record:
            result.append(
                EntryField(
                    name=name,
                    status="covered",
                    trace_fields=(name,),
                    value=payload.record[name],
                    provenance=ENTRY_PROVENANCE,
                )
            )
        elif name == "precip":
            result.append(
                EntryField(
                    name=name,
                    status=daily_inputs.precip_status,
                    trace_fields=daily_inputs.precip_fields,
                    value=None,
                    provenance=DAILY_ACCUMULATION_PROVENANCE,
                    notes=(
                        "The entry trace has no literal precip field, but "
                        "stomate_precip_increment closes Fortran's local "
                        "increment from traced rain, snow, and dt_sechiba."
                    ),
                )
            )
        else:
            result.append(
                EntryField(
                    name=name,
                    status="missing",
                    trace_fields=(),
                    value=None,
                    provenance=ENTRY_PROVENANCE,
                )
            )
    return tuple(result)


def daily_accumulation_entry_contract() -> tuple[HelperInputContract, ...]:
    """Map entry fields to `stomate_accumulate_daily` increments.

    Status meanings:
    `covered` means the exact increment is explicit in the entry payload;
    `partial` means related entry fields are present but a Fortran local or
    full array is still missing; `missing` means the helper input is not in the
    entry payload.
    """

    return (
        HelperInputContract(
            "stomate_accumulate_daily",
            "humrel -> humrel_daily",
            "covered",
            ("humrel",),
            DAILY_ACCUMULATION_PROVENANCE,
        ),
        HelperInputContract(
            "stomate_accumulate_daily",
            "litterhumdiag -> litterhum_daily",
            "covered",
            ("litterhumdiag",),
            DAILY_ACCUMULATION_PROVENANCE,
        ),
        HelperInputContract(
            "stomate_accumulate_daily",
            "t2m -> t2m_daily",
            "covered",
            ("t2m",),
            DAILY_ACCUMULATION_PROVENANCE,
        ),
        HelperInputContract(
            "stomate_accumulate_daily",
            "temp_sol -> tsurf_daily",
            "covered",
            ("temp_sol",),
            DAILY_ACCUMULATION_PROVENANCE,
        ),
        HelperInputContract(
            "stomate_accumulate_daily",
            "stempdiag -> tsoil_daily",
            "missing",
            (),
            DAILY_ACCUMULATION_PROVENANCE,
            "No full stempdiag[npts,nslm] array is present in before_stomate_main.",
        ),
        HelperInputContract(
            "stomate_accumulate_daily",
            "shumdiag -> soilhum_daily",
            "missing",
            ("shumdiag_peat_1", "shumdiag_croppeat_1", "shumdiag_man_1", "soil_mc_top_tile"),
            DAILY_ACCUMULATION_PROVENANCE,
            "Only hydrology/peat/tide sentinels are present; they are not the full shumdiag array.",
        ),
        HelperInputContract(
            "stomate_accumulate_daily",
            "precip -> precip_daily",
            "covered",
            ("precip_rain", "precip_snow", "dt_sechiba"),
            DAILY_ACCUMULATION_PROVENANCE,
            "Fortran line 2918 computes precip exactly from traced rain, snow, and dt_sechiba.",
        ),
        HelperInputContract(
            "stomate_gpp_daily_increment / stomate_accumulate_daily / scheduled-GPP STOMATE wrapper",
            "gpp_d -> gpp_daily",
            "partial",
            ("gpp", "veget_max", "dt_sechiba", "totfrac_nobio"),
            DAILY_ACCUMULATION_PROVENANCE,
            (
                "The current before_stomate_main trace has gpp, veget_max, and "
                "dt_sechiba, but lacks totfrac_nobio. Fortran lines 2921-2931 "
                "and 3020-3032 require totfrac_nobio to build veget_cov_max; "
                "tot_bare_soil is a distinct slowproc field and is not a substitute. "
                "When audited totfrac_nobio is supplied, the scheduled-GPP wrapper "
                "can feed source-backed gpp_daily into the explicit STOMATE chain."
            ),
        ),
    )


def entry_normalization_contract() -> tuple[HelperInputContract, ...]:
    """Map entry fields to STOMATE's pre-accumulation normalization helpers."""

    return (
        HelperInputContract(
            "stomate_veget_cover_fractions",
            "veget, veget_max -> veget_cov, veget_cov_max",
            "partial",
            ("veget", "veget_max", "totfrac_nobio"),
            ENTRY_NORMALIZATION_PROVENANCE,
            (
                "The current before_stomate_main trace has veget and veget_max "
                "but lacks current totfrac_nobio. An audited driver/static "
                "totfrac_nobio may be supplied explicitly; tot_bare_soil is not "
                "a substitute."
            ),
        ),
        HelperInputContract(
            "stomate_vegetnew_firstday",
            "vegetnew_firstday -> normalized vegetnew_firstday",
            "missing",
            ("vegetnew_firstday", "totfrac_nobio_new", "date"),
            ENTRY_NORMALIZATION_PROVENANCE,
            (
                "The current before_stomate_main trace has neither "
                "vegetnew_firstday nor totfrac_nobio_new/date for the date==1 "
                "branch."
            ),
        ),
        HelperInputContract(
            "stomate_veget_cov_max_new",
            "veget_max_new -> veget_cov_max_new",
            "partial",
            ("veget_max_new", "totfrac_nobio_new", "do_now_stomate_lcchange"),
            ENTRY_NORMALIZATION_PROVENANCE,
            (
                "The trace has veget_max_new but lacks totfrac_nobio_new and "
                "the branch flag. Current totfrac_nobio and tot_bare_soil must "
                "not be used for this next-year normalization."
            ),
        ),
    )


def entry_local_prep_contract() -> tuple[HelperInputContract, ...]:
    """Map entry fields to the grouped ``stomate_main`` section-3 prep."""

    return (
        HelperInputContract(
            "stomate_entry_local_prep_explicit",
            "precip_rain, precip_snow, dt_sechiba -> precip",
            "covered",
            ("precip_rain", "precip_snow", "dt_sechiba"),
            DAILY_ACCUMULATION_PROVENANCE,
            "Fortran line 2918 computes precip exactly from rain, snow, and dt_sechiba.",
        ),
        HelperInputContract(
            "stomate_entry_local_prep_explicit",
            "veget, veget_max, totfrac_nobio -> veget_cov, veget_cov_max",
            "partial",
            ("veget", "veget_max", "totfrac_nobio"),
            ENTRY_NORMALIZATION_PROVENANCE,
            (
                "The bridge trace has veget and veget_max; current "
                "totfrac_nobio must be supplied from audited slowproc/driver "
                "state. tot_bare_soil is a separate slowproc diagnostic."
            ),
        ),
        HelperInputContract(
            "stomate_entry_local_prep_explicit",
            "glccNetLCC, glccSecondShift, glccPrimaryShift, harvest_matrix, totfrac_nobio",
            "partial",
            (
                "glccNetLCC",
                "glccSecondShift",
                "glccPrimaryShift",
                "harvest_matrix",
                "totfrac_nobio",
            ),
            ENTRY_NORMALIZATION_PROVENANCE,
            (
                "Fortran lines 2935-2977 normalize these inout land-cover "
                "matrices with current totfrac_nobio; the denominator must "
                "not be inferred from tot_bare_soil."
            ),
        ),
        HelperInputContract(
            "stomate_entry_local_prep_explicit",
            "date, vegetnew_firstday, totfrac_nobio_new -> vegetnew_firstday",
            "missing",
            ("date", "vegetnew_firstday", "totfrac_nobio_new"),
            ENTRY_NORMALIZATION_PROVENANCE,
            "Fortran lines 2980-2990 run only on date == 1 and require next-year non-biological fraction.",
        ),
        HelperInputContract(
            "stomate_entry_local_prep_explicit",
            "veget_max_new, totfrac_nobio_new, do_now_stomate_lcchange/dyn_peat/update_peatfrac -> veget_cov_max_new",
            "partial",
            ("veget_max_new", "totfrac_nobio_new", "do_now_stomate_lcchange", "dyn_peat", "update_peatfrac"),
            ENTRY_NORMALIZATION_PROVENANCE,
            (
                "Fortran lines 2993-3016 run under land-cover-change or "
                "dynamic-peat branch flags and require totfrac_nobio_new."
            ),
        ),
        HelperInputContract(
            "stomate_entry_local_prep_explicit",
            "gpp, veget_cov_max, dt_sechiba -> gpp_d",
            "partial",
            ("gpp", "veget_max", "totfrac_nobio", "dt_sechiba"),
            DAILY_ACCUMULATION_PROVENANCE,
            (
                "Fortran lines 3020-3032 use veget_cov_max from the same "
                "section-3 prep; current totfrac_nobio is still required."
            ),
        ),
    )


def daily_scheduling_contract() -> tuple[HelperInputContract, ...]:
    """Map entry/local values to STOMATE's first daily scheduling block."""

    return (
        HelperInputContract(
            "stomate_end_of_month",
            "day, sec, dt_sechiba -> EndOfMonth",
            "covered",
            ("day", "sec", "dt_sechiba"),
            DAILY_SCHEDULING_PROVENANCE,
            "Fortran lines 3128-3134 use only current calendar day, seconds, and dt_sechiba.",
        ),
        HelperInputContract(
            "stomate_instant_daily_prep",
            "t2m, ok_LAIdev, SP_tdmin, SP_tdmax, swdown -> st2m, ugdh, uphoi",
            "partial",
            ("t2m", "ok_LAIdev", "SP_tdmin", "SP_tdmax", "swdown"),
            DAILY_SCHEDULING_PROVENANCE,
            (
                "PFT14 normally has ok_LAIdev false, but the generic crop "
                "branch is exact when these parameter arrays and swdown are "
                "supplied explicitly."
            ),
        ),
        HelperInputContract(
            "stomate_accumulate_named_daily",
            "explicit increments -> daily accumulators",
            "partial",
            (
                "do_slow",
                "dt_sechiba",
                "dt_stomate",
                "humrel",
                "litterhumdiag",
                "t2m",
                "temp_sol",
                "stempdiag",
                "shumdiag",
                "precip",
                "gpp_d",
                "precip_snow",
                "snow",
                "tmc_topgrass",
                "wspeed",
            ),
            DAILY_SCHEDULING_PROVENANCE,
            (
                "The helper implements exact stomate_accu semantics, but "
                "callers must provide each accumulator and full-array "
                "increment explicitly; no missing daily field is fabricated."
            ),
        ),
        HelperInputContract(
            "stomate_update_temperature_extrema_daily",
            "t2m_min, t2m_max -> t2m_min_daily, t2m_max_daily",
            "covered",
            ("t2m_min", "t2m_max", "t2m_min_daily", "t2m_max_daily"),
            DAILY_SCHEDULING_PROVENANCE,
            "Fortran lines 3235-3238 apply elementwise MIN/MAX updates.",
        ),
    )


def littercalc_entry_contract() -> tuple[HelperInputContract, ...]:
    """Map STOMATE state into the local ``littercalc_leak`` entry bridge."""

    return (
        HelperInputContract(
            "stomate_littercalc_entry_prep",
            "soil_mc -> soil_mc_32l",
            "partial",
            ("soil_mc",),
            LITTERCALC_ENTRY_PROVENANCE,
            (
                "The helper exactly expands a full hydrology soil_mc[npts,nslm,nstm] "
                "array to ndeep layers; the first-step trace currently has only a top-layer sentinel."
            ),
        ),
        HelperInputContract(
            "stomate_littercalc_entry_prep",
            "turnover_daily, bm_to_litter, dt_sechiba -> turnover_littercalc, bm_to_littercalc",
            "covered",
            ("turnover_daily", "bm_to_litter", "dt_sechiba"),
            LITTERCALC_ENTRY_PROVENANCE,
            "The dt_sechiba/one_day scaling is source-closed; turnover_daily and bm_to_litter remain upstream STOMATE state inputs.",
        ),
        HelperInputContract(
            "littercalc_leak",
            "prepared littercalc inputs and litter/carbon state",
            "partial",
            (),
            LITTERCALC_ENTRY_PROVENANCE,
            (
                "Source-closed helper kernels now cover litterfrac/frac_soil/tau tables, "
                "fbact_met/fbact_str, section-2 litter and lignin additions, aboveground "
                "and belowground decay fluxes, SPITFIRE aboveground fuel add/sync, "
                "and deadleaf cover. "
                "The composed littercalc_leak_core_with_controls wrapper is available "
                "for explicit controls and can compute ordinary and Moyano aboveground "
                "temperature/moisture controls. Full littercalc_leak closure still needs "
                "crop control interpolation branches and mass-balance/output bookkeeping."
            ),
        ),
    )


def soilcarbon_leak_contract() -> tuple[HelperInputContract, ...]:
    """Map the OK_LEAK path after ``littercalc_leak`` into helper kernels."""

    return (
        HelperInputContract(
            "altcalc_doc",
            "tprof, zprof, altmax -> alt, alt_ind, altmax",
            "partial",
            ("tprof", "zz_coef_deep", "altmax", "veget_mask_2d"),
            SOILCARBON_LEAK_PROVENANCE,
            (
                "Source-closed helper covers the standard and newaltcalc active-layer "
                "index/depth updates plus first-call altmax bookkeeping. It still needs "
                "integration with soilcarbon_leak state/restart scheduling."
            ),
        ),
        HelperInputContract(
            "soilcarbon_leak_tf_doc_inputs",
            "precip2ground, precip2canopy, biomass, veget_max -> TF-DOC inputs",
            "covered",
            ("precip2ground", "precip2canopy", "biomass", "veget_max", "is_tree", "ok_TF_DOC"),
            SOILCARBON_LEAK_PROVENANCE,
            "Implements soilcarbon_leak TF-DOC initialization and ok_TF_DOC gating.",
        ),
        HelperInputContract(
            "soilcarbon_leak_doc_export_aggregate",
            "DOC_EXP and respiration/water terms -> DOC_EXP_agg",
            "covered",
            (
                "DOC_EXP",
                "veget_max",
                "resp_hetero_litter",
                "resp_hetero_soil",
                "resp_hetero_flood",
                "resp_flood_soil",
                "runoff_per_soil",
                "drainage_per_soil",
                "pref_soil_veg",
                "flood_frac",
            ),
            SOILCARBON_LEAK_PROVENANCE,
            "Covers labile/refractory DOC aggregation, NaN skipping, bare-soil skip, and DIC runoff/drain/flood terms.",
        ),
        HelperInputContract(
            "soilcarbon_cryoturbation_coefficients / diffuse",
            "altmax, vegetation mask, soil grid, C/DOC/litter state -> cryoturbated C/DOC/litter state",
            "partial",
            (
                "altmax_ind",
                "altmax_lastyear",
                "fixed_cryoturbation_depth",
                "veget_mask_2d",
                "zi_soil",
                "zf_soil_B",
                "carbon_32l",
                "DOC",
                "litter_below",
            ),
            SOILCARBON_LEAK_PROVENANCE,
            (
                "Covers cryoturbate_doc_POC action coefficients and diffuse "
                "kernel lines 2881-3112 and 3123-3329, including old/new "
                "cryoturbation profiles, bioturbation profile, xc/xd, "
                "alpha/beta recurrence, and carbon-only C/DOC/litter state "
                "updates. It remains partial because the full soilcarbon_leak "
                "call path still has to schedule saved coefficients around "
                "daily OK_LEAK state and PERMA_PEAT branch controls."
            ),
        ),
        HelperInputContract(
            "soilcarbon_perma_peat_cmax / redistribute",
            "peat bulk density, C pools, peat mask, soil grid -> Cmax/deepC_peat/peat_OLT",
            "covered",
            (
                "peat_bulk_density",
                "carbon_32l",
                "is_peat",
                "veget_mask_2d",
                "zf_soil_B",
                "frac1",
                "frac2",
            ),
            SOILCARBON_LEAK_PROVENANCE,
            (
                "Covers PERMA_PEAT Cmax and redistribution branch lines "
                "1078-1087 and 1375-1437, including sequential excess-carbon "
                "transfer to the next layer and peat_OLT calculation."
            ),
        ),
        HelperInputContract(
            "soilcarbon_leak_activity_factors / frac_carb / doc_inputs / decompose_update / transport_diffusion_export",
            "DOC controls, inputs, POC/DOC decomposition, adsorption, water transport, diffusion, export -> updated carbon/DOC pools",
            "partial",
            (
                "fbact_doc",
                "fbact",
                "clay",
                "bulk_density",
                "soilcarbon_input_DOC",
                "DOC_to_topsoil",
                "DOC_to_subsoil",
                "DOC",
                "carbon_32l",
                "litter_above",
                "litter_below",
                "lignin_struc_above",
                "lignin_struc_below",
                "soil_mc_32l",
                "soil_mc",
                "wat_flux",
                "soilwater_31mm",
                "runoff_per_soil",
                "drainage_per_soil",
                "runoff2peat",
                "fastr",
                "pref_soil_veg",
                "flood_frac",
                "wet_dep_flood",
                "floodcarbon_input",
                "natural",
                "is_peat",
                "is_c4",
            ),
            SOILCARBON_LEAK_PROVENANCE,
            (
                "Covers source lines 1176-1193, 1289-1302, and 1528-2303, "
                "including DOC activity factors, leak-path frac_carb, carbon-only "
                "DOC inputs, LOM construction, priming/no-priming POC fluxes, DOC "
                "decomposition, CUE respiration, pool updates, adsorption/desorption, "
                "water-flux transport, DOC diffusion, runoff/drain/flood export, DOC_EXP, "
                "and DOC pool subtraction. It can call the explicit cryoturbation "
                "cycle and PERMA_PEAT redistribution branch when saved coefficients "
                "and active-layer/peat state are supplied. Full soilcarbon_leak "
                "closure still needs complete stomate_main OK_LEAK state scheduling."
            ),
        ),
    )


def permafrost_control_contract() -> tuple[HelperInputContract, ...]:
    """Map STOMATE entry state into the ``microactem`` permafrost controls."""

    return (
        HelperInputContract(
            "stomate_permafrost_decomposition_controls",
            "tdeep, hsdeep, shumdiag_peat, is_peat, tau_peat, z_tau -> prmfrst_soilc_tempctrl",
            "partial",
            ("tdeep", "hsdeep", "shumdiag_peat", "is_peat", "tau_peat", "z_tau", "poor_soils"),
            PERMAFROST_CONTROL_PROVENANCE,
            (
                "The microactem kernel is source-closed, but the composed "
                "stomate_main block still requires full deep thermodynamic "
                "arrays and explicit peat/PFT parameters."
            ),
        ),
        HelperInputContract(
            "microactem",
            "temp_celsius, moist_in, mc_peat -> fbact_seconds",
            "covered",
            ("temp_celsius", "moist_in", "mc_peat", "frozen_respiration_func", "limit_decomp_moisture"),
            PERMAFROST_CONTROL_PROVENANCE,
            "Pure local kernel from stomate_permafrost_soilcarbon.f90 lines 2468-2703.",
        ),
        HelperInputContract(
            "deep_carbon_core",
            "deepC_a/s/p, fbact_out, gas state, soil carbon input, peat mask, soil grid -> updated OK_PC deep carbon",
            "partial",
            (
                "deepC_a",
                "deepC_s",
                "deepC_p",
                "soilc_in",
                "fbact_out",
                "O2_soil",
                "CH4_soil",
                "totporO2_soil",
                "totporCH4_soil",
                "veget_mask_2d",
                "zf_soil",
                "zi_soil",
                "snowdz",
                "snowrho",
                "is_peat",
            ),
            PERMAFROST_CONTROL_PROVENANCE + DEEP_CARBON_PROVENANCE,
            (
                "The composed local OK_PC core now runs carbinput carbon-input "
                "depth distribution, permafrost_decomp oxic/methane pool "
                "transfer with optional O2 limitation and PERMA_PEAT redistribution, "
                "methane/CO2 flux aggregation, optional deep-carbon cryoturbation "
                "diffuse/coefficients, and calc_vert_int_soil_carbon full/surface "
                "integration in Fortran order. Local kernels also cover OK_PC "
                "snowlevels, snow_interpol, get_gasdiff, soil_gasdiff_coeff/diff, "
                "traMplan, ebullition, altcalc/root-depth, and yedoma reset. "
                "The explicit OK_PC daily adapter wires those kernels in "
                "deep_carbcycle source order. Restart gas reading plus a "
                "firstcall SAVE-sidecar initializer now build snow geometry, "
                "gas porosity/diffusivity, initial gasdiff coefficients, and "
                "active-layer/root state. A restart plus SAVE-sidecar handoff "
                "adapter writes deepC/carbon/soilc_total/altmax, gas/snow/SAVE "
                "coefficients, heat_Zimov, and deep CH4/CO2 flux diagnostics "
                "across consecutive days. The firstcall restart-gas wrapper "
                "now composes restart gas -> firstcall sidecar -> Day1 "
                "deep_carbcycle for a local OK_PC end-to-end micro-case; "
                "remaining work is numerical comparison against a Fortran "
                "OK_PC trace if this branch becomes an enabled target."
            ),
        ),
    )


def ok_leak_integration_contract() -> tuple[HelperInputContract, ...]:
    """Track the explicit OK_LEAK composition adapter."""

    return (
        HelperInputContract(
            "stomate_ok_leak_with_maintenance_explicit",
            "explicit maintenance/prep + littercalc + soilcarbon_leak + DOC aggregation inputs -> OK_LEAK outputs",
            "partial",
            (
                "biomass",
                "t2m",
                "t2m_longterm",
                "stempdiag",
                "rprof",
                "sla_calc",
                "soil_mc",
                "turnover_daily",
                "bm_to_litter",
                "litter/soilcarbon/hydrology boundary state",
            ),
            PROCESS_BOUNDARY_PROVENANCE + LITTERCALC_ENTRY_PROVENANCE + SOILCARBON_LEAK_PROVENANCE,
            (
                "Composes stomate_main lines 3244-3293 with the explicit OK_LEAK "
                "adapter: maintenance respiration, resp_maint_radia/flood_root_radia, "
                "resp_maint_part accumulation, soil_mc_32l construction, and "
                "turnover/bm_to_litter scaling. It remains partial because phenology, "
                "allocation, turn-process state wiring, and full state scheduling remain "
                "separate source-backed processes."
            ),
        ),
        HelperInputContract(
            "stomate_ok_leak_explicit",
            "explicit littercalc + TF-DOC + soilcarbon_leak + DOC aggregation inputs -> OK_LEAK outputs",
            "partial",
            (
                "littercalc state",
                "bm_to_litter",
                "turnover",
                "environment controls",
                "soilcarbon state",
                "HYDROL water flux/export inputs",
                "TF-DOC canopy inputs",
                "resp_maint_part_radia",
                "flood_root_radia",
            ),
            LITTERCALC_ENTRY_PROVENANCE + SOILCARBON_LEAK_PROVENANCE + PROCESS_BOUNDARY_PROVENANCE,
            (
                "Composes audited littercalc_leak, TF-DOC ground transfer, "
                "soilcarbon_leak_core_step, optional explicit cryoturbation cycle, "
                "optional explicit PERMA_PEAT redistribution, and DOC aggregation "
                "from explicit process-boundary inputs. It does not derive phenology, "
                "allocation, turnover, hydrology, maintenance respiration, PERMA_PEAT, "
                "or cryoturbation state."
            ),
        ),
    )


def helper_entry_contract() -> tuple[HelperInputContract, ...]:
    """Map entry payload coverage to current daily/NPP/maintenance helpers."""

    process_contracts = (
        HelperInputContract(
            "maintenance_respiration",
            "t2m",
            "covered",
            ("t2m",),
            PROCESS_BOUNDARY_PROVENANCE,
        ),
        HelperInputContract(
            "maintenance_respiration",
            "biomass, t2m_longterm, stempdiag, rprof, sla_calc, parameters",
            "missing",
            (),
            PROCESS_BOUNDARY_PROVENANCE,
            "These are required by the existing helper and are not supplied by the entry payload.",
        ),
        HelperInputContract(
            "prepare_daily_carbon_inputs",
            "gpp_daily",
            "missing",
            ("gpp",),
            PROCESS_BOUNDARY_PROVENANCE,
            "Raw entry gpp is not the accumulated gpp_daily state.",
        ),
        HelperInputContract(
            "prepare_daily_carbon_inputs",
            "biomass",
            "missing",
            (),
            PROCESS_BOUNDARY_PROVENANCE,
        ),
        HelperInputContract(
            "maintenance_respiration / scheduled-GPP+maintenance STOMATE wrapper",
            "resp_maint_part",
            "partial",
            ("biomass", "t2m", "t2m_longterm", "stempdiag", "rprof", "sla_calc", "maintenance parameters"),
            PROCESS_BOUNDARY_PROVENANCE,
            (
                "The source-backed maintenance wrapper can compute and accumulate "
                "resp_maint_part before the explicit STOMATE chain when full "
                "maintenance inputs and parameters are supplied. The entry trace "
                "still lacks full biomass/stempdiag/PFT parameter coverage."
            ),
        ),
        HelperInputContract(
            "prepare_daily_carbon_inputs / stomate_daily_carbon_explicit",
            "f_alloc",
            "missing",
            (),
            PROCESS_BOUNDARY_PROVENANCE,
            (
                "The low-level explicit NPP adapter still requires f_alloc. "
                "For source-backed composition, use allocation_step or the "
                "prescribe/alloc wrappers, which derive f_alloc from "
                "stomate_alloc.f90 rather than a trace."
            ),
        ),
        HelperInputContract(
            "scheduled-GPP+maintenance wrapper / prescribe_step / constraints_step / phenology_step / allocation_step",
            "gpp/maintenance -> prescribe -> constraints -> phenology(none) -> alloc -> f_alloc/regenerate/post-NPP state",
            "partial",
            (),
            PROCESS_BOUNDARY_PROVENANCE
            + (
                "src_stomate/stomate.f90, stomate_main, lines 3020-3032 and 3208",
                "src_stomate/stomate.f90, stomate_main, lines 3244-3267",
                "src_stomate/stomate_lpj.f90, stomate_lpj, lines 944-960 and 1068-1102",
                "src_stomate/stomate_prescribe.f90, prescribe, lines 126-346",
                "src_stomate/lpj_constraints.f90, constraints, lines 99-242",
                "src_stomate/stomate_phenology.f90, phenology, lines 319-563",
                "src_stomate/stomate_alloc.f90, alloc, lines 280-817",
            ),
            (
                "The static/cold-start prescribe path, climatic adaptation/"
                "regeneration memory, active-PFT pheno_model='none' scheduling "
                "shell, non-crop allocation path, GPP daily scheduling, and "
                "maintenance respiration accumulation are now source-backed and "
                "can feed gpp_daily/resp_maint_part/regenerate/f_alloc into the "
                "post-NPP chain without trace. Full entry closure still needs "
                "non-none phenology onset models for other active PFTs, season "
                "accumulators, pftinout scheduling, and complete upstream state "
                "assembly."
            ),
        ),
        HelperInputContract(
            "stomate_daily_carbon_explicit",
            "pft_present, frac_growthresp, optional leaf_age/leaf_frac/age/SLA state",
            "missing",
            (),
            PROCESS_BOUNDARY_PROVENANCE,
            (
                "The adapter can now compose npp_calc biomass/NPP algebra with "
                "leaf_age, leaf_frac, age, and SLA bookkeeping when all explicit "
                "state and PFT parameters are supplied. The entry payload still "
                "does not provide those upstream state arrays directly."
            ),
        ),
        HelperInputContract(
            "stomate_daily_carbon_turnover_explicit",
            "npp_calc age/SLA state + stomate_turnover::turn explicit climate/season/PFT inputs -> turnover_daily and updated biomass state",
            "partial",
            (),
            PROCESS_BOUNDARY_PROVENANCE
            + (
                "src_stomate/stomate_lpj.f90, stomate_lpj, lines 1118-1131 and 1260-1268",
                "src_stomate/stomate_turnover.f90, turn, lines 169-947",
            ),
            (
                "The NPP-to-turnover adapter is source-closed when all explicit "
                "season, climate, leaf-age, and PFT parameter arrays are supplied. "
                "The entry payload still does not derive those arrays from full "
                "STOMATE scheduling, season, phenology, allocation, gap/light, "
                "or establishment state."
            ),
        ),
        HelperInputContract(
            "stomate_daily_carbon_gap_turnover_explicit",
            "npp_calc age/SLA state + lpj_gap + stomate_turnover::turn explicit inputs -> bm_to_litter, turnover_daily, mortality, and updated biomass state",
            "partial",
            (),
            PROCESS_BOUNDARY_PROVENANCE
            + (
                "src_stomate/stomate_lpj.f90, stomate_lpj, lines 1118-1131, 1237-1241, and 1260-1268",
                "src_stomate/lpj_gap.f90, gap, lines 117-364",
                "src_stomate/stomate_turnover.f90, turn, lines 169-947",
            ),
            (
                "The explicit adapter now preserves the post-NPP Fortran order "
                "through gap mortality and turnover. It remains partial because "
                "full STOMATE entry still needs source-backed scheduling for "
                "kill/crown/fire/light/establish/cover/vmax and the seasonal "
                "state arrays consumed by these processes."
            ),
        ),
        HelperInputContract(
            "stomate_daily_carbon_kill_gap_turnover_explicit",
            "npp_calc age/SLA state + lpj_kill/crown/gap/turn/light/establish/cover/setlai/vmax explicit inputs -> post-NPP vcmax boundary state",
            "partial",
            (),
            PROCESS_BOUNDARY_PROVENANCE
            + (
                "src_stomate/stomate_lpj.f90, stomate_lpj, lines 1118-1148, 1237-1250, 1260-1268, 1292-1307, 1300-1318, 1372-1379, and 1552-1557",
                "src_stomate/lpj_kill.f90, kill, lines 63-272",
                "src_stomate/lpj_crown.f90, crown, lines 78-201",
                "src_stomate/lpj_gap.f90, gap, lines 117-364",
                "src_stomate/stomate_turnover.f90, turn, lines 169-947",
                "src_stomate/lpj_light.f90, light, lines 103-648",
                "src_stomate/lpj_establish.f90, establish, lines 100-855",
                "src_stomate/lpj_cover.f90, cover, lines 73-387",
                "src_stomate/stomate_lpj.f90, harvest, lines 2502-2566",
                "src_stomate/stomate_lai.f90, setlai, lines 58-88",
                "src_stomate/stomate_vmax.f90, vmax, lines 105-363",
            ),
            (
                "This adapter preserves the implemented post-NPP Fortran order "
                "through kill, crown recalculation, gap mortality, kill, and "
                "turnover, and the DGVM light-competition branch followed by "
                "kill(light), establishment, and post-establishment crown "
                "recalculation, ordinary non-peat cover redistribution, final "
                "HARVEST_AGRI turnover reduction, LAI recalculation, and "
                "stomate_vmax leaf-age/vcmax update, plus the "
                "end-of-StomateLpj diagnostic pool preparation, using explicit "
                "state arrays. It remains partial because fire/spitfire, peat "
                "cover, land-cover-change branches, "
                "and full STOMATE scheduling are not yet wired."
            ),
        ),
        HelperInputContract(
            "harvest_agri_step",
            "veget_max, turnover_daily, natural, is_peat, frac_turnover_daily -> turnover_daily, harvest_above",
            "covered",
            ("veget_max", "turnover_daily", "natural", "is_peat", "frac_turnover_daily"),
            PROCESS_BOUNDARY_PROVENANCE
            + (
                "src_stomate/stomate_lpj.f90, StomateLpj calls harvest, lines 1389-1392",
                "src_stomate/stomate_lpj.f90, harvest, lines 2502-2566",
                "src_parameters/constantes_var.f90, harvest_agri and frac_turnover_daily, lines 477-478 and 1253-1254",
            ),
            (
                "Covers the active HARVEST_AGRI branch for non-natural, "
                "non-peat PFTs, including unchanged bm_to_litter and "
                "veget_max-weighted harvest_above accumulation."
            ),
        ),
        HelperInputContract(
            "vmax_step",
            "leaf_age, leaf_frac, Vcmax25, N_limfert, leaf_timecst, leafagecrit, pheno_type, leaf_tab, ok_LAIdev -> vcmax",
            "partial",
            (),
            PROCESS_BOUNDARY_PROVENANCE
            + (
                "src_stomate/stomate_lpj.f90, stomate_lpj, lines 1552-1557",
                "src_stomate/stomate_vmax.f90, vmax, lines 105-363",
                "src_parameters/constantes_var.f90, vmax scalar defaults, lines 1417-1431",
                "src_parameters/pft_parameters.f90, Vcmax25/leaf_timecst/leafagecrit/phenology reads, lines 3240-3459 and 4430-4582",
            ),
            (
                "The local kernel is source-closed and updates leaf_age, "
                "leaf_frac, and vcmax exactly from explicit PFT parameter "
                "arrays. It remains partial at entry-contract level because "
                "full STOMATE must still supply those arrays from audited "
                "parameter/restart/scheduling state."
            ),
        ),
        HelperInputContract(
            "stomate_lpj_output_diagnostics",
            "biomass, turnover_daily, bm_to_litter, litter, carbon_32l, DOC, veget_max, soil grids, product pools -> modelout diagnostic pools",
            "partial",
            (),
            PROCESS_BOUNDARY_PROVENANCE
            + (
                "src_stomate/stomate_lpj.f90, StomateLpj, lines 1578-1663",
                "src_stomate/stomate_lpj.f90, XIOS/modelout fields, lines 1667-1807 and 1889-1917",
                "src_stomate/stomate_lpj.f90, history writes, lines 2200-2246",
                "fortran_run_scripts/paper_250919/c2.4_Model_run_functions_sensitivity.py, modelout formula, lines 50, 60-76, and 115-118",
            ),
            (
                "Source-closed helper now prepares tot_litter_carb, "
                "tot_soil_carb, tot_litter_soil_carb, tot_live_biomass, "
                "tot_turnover, tot_bm_to_litter, carb_mass_total/variation, "
                "and carbon_32l pftmean/concentration diagnostics. It remains "
                "The paper modelout mapper can now consume current explicit "
                "StomateLpj biomass/GPP/NPP state without reading history "
                "NetCDF. It remains partial because branch-specific XIOS/IPCC "
                "fields outside the paper modelout columns are still separate."
            ),
        ),
        HelperInputContract(
            "establishment_rates_step / establishment_biomass_step",
            "lpj_establish rate sections + sapling biomass accounting with explicit DGVM/static boundary inputs -> d_ind, IND_ESTAB, biomass, leaf-age, co2_to_bm, woodmass_ind",
            "partial",
            (),
            PROCESS_BOUNDARY_PROVENANCE
            + (
                "src_stomate/lpj_establish.f90, establish, lines 100-855",
                "src_parameters/constantes_var.f90, treat_expansion, lines 473-474",
                "src_parameters/constantes_var.f90, establishment parameters, lines 922-938 and 1104-1109",
                "src_stomate/stomate_data.f90, regenerate_crit, line 70",
            ),
            (
                "These helpers close the source-backed establishment-rate, "
                "individual-increment, sapling biomass, leaf-age, age, co2_to_bm, "
                "and woodmass_ind calculations. They remain partial only in the "
                "sense that full STOMATE entry still supplies their boundary "
                "inputs explicitly rather than deriving all upstream scheduling "
                "and state internally."
            ),
        ),
    )
    return (
        entry_local_prep_contract()
        + daily_scheduling_contract()
        + permafrost_control_contract()
        + littercalc_entry_contract()
        + soilcarbon_leak_contract()
        + ok_leak_integration_contract()
        + entry_normalization_contract()
        + daily_accumulation_entry_contract()
        + process_contracts
    )


def contract_gaps(contracts: tuple[HelperInputContract, ...]) -> tuple[HelperInputContract, ...]:
    """Return every contract entry that is not fully covered."""

    return tuple(contract for contract in contracts if contract.status != "covered")
