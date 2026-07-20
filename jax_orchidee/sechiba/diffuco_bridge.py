"""DIFFUCO boundary contract and trace helpers for the PFT14 paper-case path.

This module records the Fortran source boundary and reads explicit bridge
trace fields needed to close DIFFUCO without guessing process values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from jax_orchidee.trace.server_1961 import TRACE_ROOT, find_server_record, read_server_records, trace_available


@dataclass(frozen=True)
class DiffucoBoundaryField:
    """One field needed at or after the `diffuco_main` boundary."""

    name: str
    shape: str
    direction: str
    role: str
    fortran_provenance: tuple[str, ...]
    consumers: tuple[str, ...]
    trace_point: str

    @property
    def requires_trace(self) -> bool:
        """All fields in this contract need a named DIFFUCO trace."""

        return True


@dataclass(frozen=True)
class DiffucoValidation:
    """Validation result for a DIFFUCO boundary payload."""

    available_fields: frozenset[str]
    missing_fields: tuple[str, ...]
    requires_trace: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_fields


@dataclass(frozen=True)
class DiffucoTraceValidation:
    """Trace coverage result for the audited DIFFUCO bridge payload."""

    trace_name: str
    tag: str
    available_columns: frozenset[str]
    covered_contract_fields: tuple[str, ...]
    missing_contract_fields: tuple[str, ...]
    uncovered_internal_diagnostics: tuple[str, ...]
    active_branch_checks: Mapping[str, bool]
    provenance: tuple[str, ...]

    @property
    def parse_and_coverage_ok(self) -> bool:
        """True when trace columns cover every trace-backed contract field."""

        return not self.missing_contract_fields and all(self.active_branch_checks.values())

    @property
    def complete(self) -> bool:
        """True only when no known DIFFUCO diagnostic remains untraced."""

        return self.parse_and_coverage_ok and not self.uncovered_internal_diagnostics


@dataclass(frozen=True)
class DiffucoParityReadiness:
    """Readiness report for strict JAX-vs-Fortran DIFFUCO numeric parity."""

    available_fields: frozenset[str]
    missing_fields: tuple[str, ...]
    trace_sources: Mapping[str, str]
    provenance: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.missing_fields


@dataclass(frozen=True)
class DiffucoModuleClosureValidation:
    """Executable status for the PFT14 DIFFUCO module-level closure."""

    trans_co2_active_record_found: bool
    after_main_active_record_found: bool
    overlap_kjit: tuple[int, ...]
    overlap_fields: tuple[str, ...]
    overlap_mismatches: Mapping[int, tuple[str, ...]]
    missing_active_after_main_fields: tuple[str, ...]
    provenance: tuple[str, ...]

    @property
    def trans_co2_active_closed(self) -> bool:
        """The active C3 photosynthesis/resistance kernel has strict trace parity."""

        return self.trans_co2_active_record_found

    @property
    def inactive_main_boundary_closed(self) -> bool:
        """The overlap between ``diffuco_trans_co2`` and ``after_diffuco_main`` is clean."""

        return bool(self.overlap_kjit) and not any(self.overlap_mismatches.values())

    @property
    def active_main_boundary_closed(self) -> bool:
        """The active ``diffuco_main`` output boundary has an active trace row."""

        return self.after_main_active_record_found and not self.missing_active_after_main_fields

    @property
    def module_closed(self) -> bool:
        """True only when active photosynthesis and active main-boundary checks both close."""

        return self.trans_co2_active_closed and self.inactive_main_boundary_closed and self.active_main_boundary_closed


ACTIVE_PAPER_CASE_BRANCHES = {
    "co2_photosynthesis": (
        True,
        "STOMATE_OK_CO2=y in outputs/server_1961_trace_full_20260623/run/run.def "
        "lines 61-62; intersurf.f90 reads STOMATE_OK_CO2 at lines 2152-2161; "
        "diffuco.f90::diffuco_main selects diffuco_trans_co2 when ok_co2 at lines 665-671.",
    ),
    "mangrove_salinity_tide_control": (
        True,
        "READ_SALINITY=TRUE, READ_TIDE=TRUE, and tides=y in "
        "outputs/server_1961_trace_full_20260623/run/run.def lines 14-17 and 190-191; "
        "diffuco.f90::diffuco_main applies salinity/inundation controls at lines 445-622 "
        "and scales PFT14 assimilation at lines 2834-2844.",
    ),
    "bvoc_chemistry": (
        False,
        "CHEMISTRY_BVOC=FALSE in outputs/server_1961_trace_full_20260623/run/used_run.def "
        "lines 209-211; diffuco.f90 calls chemistry_bvoc only when ok_bvoc at lines 686-692.",
    ),
    "explicit_snow": (
        True,
        "OK_EXPLICITSNOW=y in outputs/server_1961_trace_full_20260623/run/run.def line 69; "
        "sechiba.f90 skips enerbil_fusion when ok_explicitsnow is true at lines 1077-1081.",
    ),
    "cwrr_hydrology": (
        True,
        "HYDROL_CWRR=y in outputs/server_1961_trace_full_20260623/run/run.def line 70; "
        "sechiba.f90 selects hydrol_main at lines 1046-1072.",
    ),
    "choisnel_hydrology": (
        False,
        "HYDROL_CWRR=y in outputs/server_1961_trace_full_20260623/run/run.def line 70; "
        "sechiba.f90 calls hydrolc_main only when .NOT. hydrol_cwrr at lines 1027-1040.",
    ),
    "peat_carbon_ok_peat_ok_pc": (
        False,
        "OK_PEAT=n and OK_PC=n in outputs/server_1961_trace_full_20260623/run/run.def "
        "lines 202-203.",
    ),
    "leak_carbon": (
        True,
        "OK_LEAK=y in outputs/server_1961_trace_full_20260623/run/run.def line 220; "
        "sechiba.f90 passes MICT-leak water fields to slowproc_main at lines 1211-1213.",
    ),
    "floodplain_routing": (
        False,
        "DO_FLOODPLAINS=n and DO_FLOODINFILT=n in "
        "outputs/server_1961_trace_full_20260623/run/run.def lines 223-226.",
    ),
}


TRACE_METADATA_FIELDS = frozenset(("tag", "start_line"))

DIFFUCO_AFTER_MAIN_TRACE = "sechiba_bridge_diffuco"
DIFFUCO_AFTER_MAIN_TAG = "after_diffuco_main"
DIFFUCO_ACTIVE_AFTER_MAIN_TRACE = "sechiba_bridge_diffuco_active"
DIFFUCO_ACTIVE_AFTER_MAIN_TAG = "after_diffuco_main_active_pft14"

DIFFUCO_TRANS_CO2_TRACE = "diffuco_trans_co2"
DIFFUCO_TRANS_CO2_PARITY_TAG = "after_diffuco_trans_co2_pft14"

DIFFUCO_TRANS_CO2_PARITY_REQUIRED_FIELDS = (
    "swdown",
    "pb",
    "qsurf",
    "qsatt",
    "t2m",
    "temp_growth",
    "ca",
    "vcmax",
    "humrel",
    "veget",
    "veget_max",
    "lai",
    "qsintveg",
    "qsintmax",
    "vbeta23",
    "q_cdrag",
    "q_cdrag_pft",
    "wind",
    "control_salinity",
    "control_inudate",
    "gpp",
    "gsmean",
    "rveget",
    "rstruct",
    "cimean",
    "vbeta3",
    "vbeta3pot",
)

DIFFUCO_BETA_CLOSURE_PARITY_REQUIRED_FIELDS = (
    "vbeta",
    "vbeta_pft",
    "vbeta1",
    "vbeta2",
    "vbeta3",
    "vbeta3pot",
    "vbeta4",
    "vbeta4_pft",
    "humrel",
    "qsintmax",
    "veget",
    "veget_max",
    "lai",
    "evap_bare_lim",
    "tot_bare_soil",
)

DIFFUCO_PFT14_FIRST_TIMESTEP_CRITICAL_FIELDS = (
    "gpp",
    "salinity",
    "tide_height_1",
    "veget_max",
    "lai",
    "temp_sol",
    "evapot_corr",
)

DIFFUCO_TRANS_TO_AFTER_MAIN_OVERLAP_FIELDS = (
    "gpp",
    "gsmean",
    "rveget",
    "rstruct",
    "cimean",
    "vbeta3",
    "vbeta3pot",
    "q_cdrag",
    "q_cdrag_pft",
    "humrel",
    "qsintveg",
    "qsintmax",
    "veget",
    "veget_max",
    "lai",
    "qsurf",
)


DIFFUCO_MINIMAL_FIELDS = (
    DiffucoBoundaryField(
        "gpp",
        "(npts,nvm); audit PFT14",
        "output",
        "same-step GPP entering slowproc_main and STOMATE daily accumulation",
        (
            "src_sechiba/sechiba.f90::sechiba_main passes gpp from diffuco_main to slowproc_main at lines 997-1005 and 1184-1192.",
            "src_sechiba/diffuco.f90::diffuco_main declares gpp output at lines 391-405.",
            "src_sechiba/diffuco.f90::diffuco_main calls diffuco_trans_co2 at lines 665-671.",
            "src_sechiba/diffuco.f90::diffuco_trans_co2 scales PFT14 assimtot by salinity/inundation at lines 2834-2844.",
            "src_sechiba/diffuco.f90::diffuco_trans_co2 assigns gpp at lines 2890-2897.",
            "src_stomate/stomate.f90::stomate_main accumulates gpp_d at lines 3198-3208.",
        ),
        ("slowproc_main", "stomate_main", "stomate_lpj/modelout GPP"),
        "after_diffuco_main",
    ),
    DiffucoBoundaryField(
        "gsmean",
        "(npts,nvm); audit PFT14",
        "output",
        "mean stomatal conductance to CO2 used to audit photosynthesis-resistance coupling",
        (
            "src_sechiba/diffuco.f90::diffuco_main declares gsmean output at lines 391-405.",
            "src_sechiba/diffuco.f90::diffuco_trans_co2 declares gsmean output at lines 2080-2088.",
            "src_sechiba/diffuco.f90::diffuco_trans_co2 assigns gsmean at lines 2884-2885.",
        ),
        ("diagnostics", "DIFFUCO parity audit"),
        "after_diffuco_main",
    ),
    DiffucoBoundaryField(
        "rveget",
        "(npts,nvm); audit PFT14",
        "output",
        "canopy stomatal resistance behind transpiration beta",
        (
            "src_sechiba/diffuco.f90::diffuco_main declares rveget output at lines 391-405.",
            "src_sechiba/diffuco.f90::diffuco_trans_co2 declares rveget output at lines 2080-2088.",
            "src_sechiba/diffuco.f90::diffuco_trans_co2 assigns rveget at lines 2915-2921.",
        ),
        ("diagnostics", "DIFFUCO transpiration beta audit"),
        "after_diffuco_main",
    ),
    DiffucoBoundaryField(
        "rstruct",
        "(npts,nvm); audit PFT14",
        "inout/output",
        "structural resistance entering interception and transpiration beta",
        (
            "src_sechiba/diffuco.f90::diffuco_main declares rstruct output at lines 391-405.",
            "src_sechiba/diffuco.f90::diffuco_inter consumes rstruct at lines 1426-1452.",
            "src_sechiba/diffuco.f90::diffuco_trans_co2 assigns rstruct at lines 2925-2931.",
        ),
        ("diffuco_inter", "diagnostics", "DIFFUCO transpiration beta audit"),
        "after_diffuco_main",
    ),
    DiffucoBoundaryField(
        "cimean",
        "(npts,nvm); audit PFT14",
        "output",
        "mean leaf intercellular CO2 concentration",
        (
            "src_sechiba/diffuco.f90::diffuco_main declares cimean output at lines 391-405.",
            "src_sechiba/diffuco.f90::diffuco_trans_co2 declares cimean output at lines 2080-2088.",
            "src_sechiba/diffuco.f90::diffuco_trans_co2 assigns cimean at lines 2887-2895.",
        ),
        ("diagnostics", "DIFFUCO photosynthesis audit"),
        "after_diffuco_main",
    ),
    DiffucoBoundaryField(
        "vbeta3",
        "(npts,nvm); audit PFT14",
        "output",
        "transpiration beta consumed immediately by ENERBIL",
        (
            "src_sechiba/diffuco.f90::diffuco_main declares vbeta3 output at lines 391-405.",
            "src_sechiba/diffuco.f90::diffuco_trans_co2 computes vbeta3 at lines 2946-2969.",
            "src_sechiba/sechiba.f90::sechiba_main passes vbeta3 to enerbil_main at lines 1013-1019.",
        ),
        ("enerbil_main",),
        "after_diffuco_main",
    ),
    DiffucoBoundaryField(
        "vbeta3pot",
        "(npts,nvm); audit PFT14",
        "output",
        "potential transpiration beta consumed by ENERBIL and HYDROL via transpot",
        (
            "src_sechiba/diffuco.f90::diffuco_main declares vbeta3pot output at lines 391-405.",
            "src_sechiba/diffuco.f90::diffuco_trans_co2 computes vbeta3pot at lines 2968-2969.",
            "src_sechiba/sechiba.f90::sechiba_main passes vbeta3pot to enerbil_main at lines 1013-1019.",
            "src_sechiba/sechiba.f90::sechiba_main passes transpot onward to hydrol_main at lines 1049-1055.",
        ),
        ("enerbil_main", "hydrol_main via transpot"),
        "after_diffuco_main",
    ),
    DiffucoBoundaryField(
        "vbeta2",
        "(npts,nvm); audit PFT14",
        "output",
        "interception-loss beta consumed by ENERBIL",
        (
            "src_sechiba/diffuco.f90::diffuco_main declares vbeta2 output at lines 391-405.",
            "src_sechiba/diffuco.f90::diffuco_inter computes vbeta2 at lines 1426-1540.",
            "src_sechiba/sechiba.f90::sechiba_main passes vbeta2 to enerbil_main at lines 1013-1019.",
        ),
        ("enerbil_main",),
        "after_diffuco_main",
    ),
    DiffucoBoundaryField(
        "vbeta/vbeta_pft/vbeta1/vbeta4/vbeta4_pft/vbeta5",
        "(npts) and (npts,nvm)",
        "output",
        "trace-written evaporation/sublimation/flood/bare-soil beta bundle consumed by ENERBIL",
        (
            "src_sechiba/diffuco.f90::diffuco_main declares beta outputs at lines 391-405.",
            "src_sechiba/diffuco.f90::diffuco_snow and diffuco_flood are called at lines 647-655.",
            "src_sechiba/diffuco.f90::diffuco_bare is called at lines 698-700.",
            "src_sechiba/diffuco.f90::diffuco_comb combines beta fields at lines 709-710 and 3256-3265.",
            "src_sechiba/sechiba.f90::sechiba_main passes these fields to enerbil_main at lines 1013-1019.",
            "outputs/server_trace_patch/apply_bridge_trace_patch.py writes vbeta, vbeta_pft, vbeta1, vbeta4, vbeta4_pft, and vbeta5, but not valpha.",
        ),
        ("enerbil_main",),
        "after_diffuco_main",
    ),
    DiffucoBoundaryField(
        "q_cdrag/q_cdrag_pft",
        "(npts) and (npts,nvm)",
        "inout",
        "surface drag products used by DIFFUCO and ENERBIL",
        (
            "src_sechiba/diffuco.f90::diffuco_main declares q_cdrag and q_cdrag_pft inout at lines 407-411.",
            "src_sechiba/diffuco.f90::diffuco_aero call updates drag fields at lines 629-637.",
            "src_sechiba/sechiba.f90::sechiba_main passes tq_cdrag/tq_cdrag_pft to enerbil_main at lines 1013-1019.",
        ),
        ("enerbil_main", "diagnostics"),
        "after_diffuco_main",
    ),
    DiffucoBoundaryField(
        "control_salinity/control_inudate",
        "(npts)",
        "internal diagnostic",
        "PFT14 mangrove controls applied to assimtot before GPP conversion",
        (
            "src_sechiba/diffuco.f90 declares control_salinity/control_inudate at lines 81-83.",
            "src_sechiba/diffuco.f90::diffuco_main initializes and computes controls at lines 445-622.",
            "src_sechiba/diffuco.f90::diffuco_trans_co2 applies controls when jv==14 at lines 2834-2844.",
            "src_sechiba/diffuco.f90::diffuco_main sends controls to XIOS at lines 722-723.",
        ),
        ("DIFFUCO PFT14 GPP audit",),
        "after_diffuco_main",
    ),
    DiffucoBoundaryField(
        "humrel",
        "(npts,nvm); audit PFT14",
        "input/inout",
        "HYDROL water stress input that may be zeroed by diffuco_comb in saturated-air cases",
        (
            "src_sechiba/diffuco.f90::diffuco_main declares humrel inout at lines 407-410.",
            "src_sechiba/diffuco.f90::diffuco_inter and diffuco_trans_co2 consume humrel at lines 659-671.",
            "src_sechiba/diffuco.f90::diffuco_comb may update humrel with vbeta fields at lines 3070-3108 and 3230-3239.",
            "src_sechiba/sechiba.f90::sechiba_main passes humrel onward to enerbil_main, hydrol_main, and slowproc_main at lines 1013-1019, 1049-1055, and 1184-1188.",
        ),
        ("enerbil_main", "hydrol_main", "slowproc_main"),
        "before_and_after_diffuco_main",
    ),
)


DIFFUCO_AFTER_MAIN_COVERAGE_COLUMNS = {
    "gpp": ("gpp",),
    "gsmean": ("gsmean",),
    "rveget": ("rveget",),
    "rstruct": ("rstruct",),
    "cimean": ("cimean",),
    "vbeta3": ("vbeta3",),
    "vbeta3pot": ("vbeta3pot",),
    "vbeta2": ("vbeta2",),
    "vbeta/vbeta_pft/vbeta1/vbeta4/vbeta4_pft/vbeta5": (
        "vbeta",
        "vbeta_pft",
        "vbeta1",
        "vbeta4",
        "vbeta4_pft",
        "vbeta5",
    ),
    "q_cdrag/q_cdrag_pft": ("q_cdrag", "q_cdrag_pft"),
    "humrel": ("humrel",),
}

DIFFUCO_UNCOVERED_INTERNAL_DIAGNOSTICS = (
    "control_salinity/control_inudate",
)

DIFFUCO_AFTER_MAIN_TRACE_PROVENANCE = (
    "outputs/server_1961_bridge_trace_20260624/traces/orchjax_sechiba_bridge_diffuco_trace.txt:after_diffuco_main",
    "outputs/server_trace_patch/apply_bridge_trace_patch.py writes after_diffuco_main immediately after sechiba.f90::sechiba_main CALL diffuco_main.",
    "docs/source_audits/server_1961_bridge_trace_patch_plan.md#after_diffuco_main records the field order and Fortran boundary.",
    "Fortran source truth: src_sechiba/sechiba.f90::sechiba_main lines 997-1005; src_sechiba/diffuco.f90::diffuco_main boundary.",
)


def minimal_diffuco_field_names() -> tuple[str, ...]:
    """Return the minimal named DIFFUCO boundary fields for PFT14."""

    return tuple(field.name for field in DIFFUCO_MINIMAL_FIELDS)


def required_after_diffuco_trace_fields() -> tuple[str, ...]:
    """Return fields that need a named after_diffuco_main trace record."""

    return tuple(field.name for field in DIFFUCO_MINIMAL_FIELDS if field.trace_point == "after_diffuco_main")


def diffuco_field(name: str) -> DiffucoBoundaryField:
    """Return one DIFFUCO boundary field by name."""

    for field in DIFFUCO_MINIMAL_FIELDS:
        if field.name == name:
            return field
    raise KeyError(f"unknown DIFFUCO boundary field: {name}")


def validate_diffuco_boundary(payload: Mapping[str, object] | Iterable[str]) -> DiffucoValidation:
    """Validate advertised DIFFUCO fields without deriving missing values."""

    if isinstance(payload, Mapping):
        available = frozenset(str(key) for key in payload.keys())
    else:
        available = frozenset(str(item) for item in payload)

    required = minimal_diffuco_field_names()
    return DiffucoValidation(
        available_fields=available,
        missing_fields=tuple(name for name in required if name not in available),
        requires_trace=required,
    )


def read_diffuco_after_main_records(
    *,
    limit: int | None = 1,
    root: str | Path = TRACE_ROOT,
) -> tuple[dict[str, object], ...]:
    """Read parsed DIFFUCO `after_diffuco_main` bridge records.

    This delegates arity validation to the fixed trace schema. No DIFFUCO
    process values are computed or inferred here.
    """

    return read_server_records(DIFFUCO_AFTER_MAIN_TRACE, tags=DIFFUCO_AFTER_MAIN_TAG, limit=limit, root=root)


def find_diffuco_after_main_payload(
    *,
    kjit: int = 1,
    ji: int = 1,
    jv: int = 14,
    root: str | Path = TRACE_ROOT,
    scan_limit: int | None = 4,
) -> dict[str, object]:
    """Return one audited PFT14 DIFFUCO bridge payload from the server trace."""

    payload = find_server_record(
        DIFFUCO_AFTER_MAIN_TRACE,
        tag=DIFFUCO_AFTER_MAIN_TAG,
        criteria={"kjit": kjit, "ji": ji, "jv": jv},
        scan_limit=scan_limit,
        root=root,
    )
    if payload is None:
        raise LookupError(f"no DIFFUCO after_diffuco_main trace for kjit={kjit}, ji={ji}, jv={jv}")
    return payload


def first_pft14_diffuco_after_main_payload(*, root: str | Path = TRACE_ROOT) -> dict[str, object]:
    """Return the first-step PFT14 DIFFUCO bridge payload used for parity tests."""

    return find_diffuco_after_main_payload(kjit=1, ji=1, jv=14, root=root, scan_limit=4)


def read_diffuco_trans_co2_records(
    *,
    limit: int | None = 1,
    root: str | Path = TRACE_ROOT,
) -> tuple[dict[str, object], ...]:
    """Read parsed PFT14 ``diffuco_trans_co2`` parity records.

    These rows come from the supplemental server trace tag
    ``after_diffuco_trans_co2_pft14`` and expose same-timestep dynamic inputs
    and outputs without reconstructing them locally.
    """

    return read_server_records(DIFFUCO_TRANS_CO2_TRACE, tags=DIFFUCO_TRANS_CO2_PARITY_TAG, limit=limit, root=root)


def find_diffuco_trans_co2_payload(
    *,
    kjit: int = 1,
    ji: int = 1,
    jv: int = 14,
    root: str | Path = TRACE_ROOT,
    scan_limit: int | None = 4,
) -> dict[str, object]:
    """Return one traced PFT14 ``diffuco_trans_co2`` payload for strict parity."""

    payload = find_server_record(
        DIFFUCO_TRANS_CO2_TRACE,
        tag=DIFFUCO_TRANS_CO2_PARITY_TAG,
        criteria={"kjit": kjit, "ji": ji, "jv": jv},
        scan_limit=scan_limit,
        root=root,
    )
    if payload is None:
        raise LookupError(f"no DIFFUCO trans_co2 trace for kjit={kjit}, ji={ji}, jv={jv}")
    return payload


def first_pft14_diffuco_trans_co2_payload(*, root: str | Path = TRACE_ROOT) -> dict[str, object]:
    """Return the first-step PFT14 ``diffuco_trans_co2`` strict-parity payload."""

    return find_diffuco_trans_co2_payload(kjit=1, ji=1, jv=14, root=root, scan_limit=4)


def first_active_pft14_diffuco_trans_co2_payload(*, root: str | Path = TRACE_ROOT) -> dict[str, object]:
    """Return the first active PFT14 ``diffuco_trans_co2`` trace payload."""

    for row in read_diffuco_trans_co2_records(limit=None, root=root):
        if _is_active_pft14_row(row):
            return row
    raise LookupError("no active PFT14 diffuco_trans_co2 trace row found")


def first_active_pft14_diffuco_after_main_payload(*, root: str | Path = TRACE_ROOT) -> dict[str, object] | None:
    """Return the first active PFT14 ``after_diffuco_main`` row, if copied."""

    if trace_available(DIFFUCO_ACTIVE_AFTER_MAIN_TRACE, root=root):
        for row in read_server_records(
            DIFFUCO_ACTIVE_AFTER_MAIN_TRACE,
            tags=DIFFUCO_ACTIVE_AFTER_MAIN_TAG,
            limit=None,
            root=root,
        ):
            if _is_active_pft14_row(row):
                return row
    if trace_available(DIFFUCO_AFTER_MAIN_TRACE, root=root):
        for row in read_diffuco_after_main_records(limit=None, root=root):
            if _is_active_pft14_row(row):
                return row
    return None


def validate_diffuco_trans_to_after_main_overlap(
    *,
    kjit_values: Iterable[int] = (1, 2, 3, 4),
    trans_root: str | Path = TRACE_ROOT,
    after_root: str | Path = TRACE_ROOT,
    fields: tuple[str, ...] = DIFFUCO_TRANS_TO_AFTER_MAIN_OVERLAP_FIELDS,
    rtol: float = 1.0e-12,
    atol: float = 0.0,
) -> Mapping[int, tuple[str, ...]]:
    """Compare same-step ``trans_co2`` values against ``after_diffuco_main``.

    Fortran provenance: ``src_sechiba/diffuco.f90::diffuco_main`` calls
    ``diffuco_trans_co2`` at lines 665-671, then ``diffuco_bare`` and
    ``diffuco_comb`` at lines 698-710. The fields in
    ``DIFFUCO_TRANS_TO_AFTER_MAIN_OVERLAP_FIELDS`` should either be preserved
    to the SECHIBA boundary or, for beta fields, be identically zero in the
    inactive overlap rows currently available in the copied bridge trace.
    """

    mismatches: dict[int, tuple[str, ...]] = {}
    for kjit in tuple(int(value) for value in kjit_values):
        trans = find_diffuco_trans_co2_payload(kjit=kjit, root=trans_root, scan_limit=None)
        after = find_diffuco_after_main_payload(kjit=kjit, root=after_root, scan_limit=None)
        bad_fields: list[str] = []
        for field in fields:
            left = trans.get(field)
            right = after.get(field)
            if not (_is_finite_number(left) and _is_finite_number(right)):
                bad_fields.append(field)
                continue
            tolerance = atol + rtol * max(abs(float(left)), abs(float(right)))
            if abs(float(left) - float(right)) > tolerance:
                bad_fields.append(field)
        mismatches[kjit] = tuple(bad_fields)
    return mismatches


def validate_diffuco_module_closure(
    *,
    trans_root: str | Path = TRACE_ROOT,
    after_root: str | Path = TRACE_ROOT,
    overlap_kjit: tuple[int, ...] = (1, 2, 3, 4),
) -> DiffucoModuleClosureValidation:
    """Return the current executable closure status for PFT14 DIFFUCO.

    This deliberately does not infer an active ``after_diffuco_main`` row from
    the active ``diffuco_trans_co2`` trace. The former is a SECHIBA boundary
    record after ``diffuco_bare`` and ``diffuco_comb``; the latter is inside the
    photosynthesis subroutine.
    """

    try:
        first_active_pft14_diffuco_trans_co2_payload(root=trans_root)
        trans_active = True
    except LookupError:
        trans_active = False

    active_after = first_active_pft14_diffuco_after_main_payload(root=after_root)
    missing_active_fields: tuple[str, ...]
    if active_after is None:
        after_active = False
        missing_active_fields = tuple(
            field
            for field in (
                "after_diffuco_main.active_row",
                "vbeta",
                "vbeta_pft",
                "vbeta1",
                "vbeta2",
                "vbeta3",
                "vbeta3pot",
                "vbeta4",
                "vbeta4_pft",
                "vbeta5",
            )
        )
    else:
        after_active = True
        missing_active_fields = tuple(
            field
            for field in DIFFUCO_TRANS_TO_AFTER_MAIN_OVERLAP_FIELDS
            if field not in active_after or not _is_finite_number(active_after[field])
        )

    return DiffucoModuleClosureValidation(
        trans_co2_active_record_found=trans_active,
        after_main_active_record_found=after_active,
        overlap_kjit=overlap_kjit,
        overlap_fields=DIFFUCO_TRANS_TO_AFTER_MAIN_OVERLAP_FIELDS,
        overlap_mismatches=validate_diffuco_trans_to_after_main_overlap(
            kjit_values=overlap_kjit,
            trans_root=trans_root,
            after_root=after_root,
        ),
        missing_active_after_main_fields=missing_active_fields,
        provenance=(
            "src_sechiba/diffuco.f90::diffuco_main lines 665-710 order trans_co2, bare, then comb.",
            "src_sechiba/sechiba.f90::sechiba_main lines 997-1005 emits after_diffuco_main only after diffuco_main returns.",
            "outputs/server_1961_diffuco_active_trace_20260624_181143 contains active diffuco_trans_co2 rows.",
            "outputs/server_1961_bridge_trace_20260624 contains after_diffuco_main rows only for kjit<=4, all with lai=0/gpp=0.",
        ),
    )


def validate_diffuco_after_main_trace_payload(
    payload: Mapping[str, object] | None = None,
    *,
    root: str | Path = TRACE_ROOT,
) -> DiffucoTraceValidation:
    """Validate trace coverage for the audited DIFFUCO after-boundary payload.

    The bridge trace covers DIFFUCO outputs and selected active-branch inputs at
    `after_diffuco_main`. It intentionally does not synthesize the internal
    mangrove controls, because this trace package did not write them.
    """

    row = dict(payload) if payload is not None else first_pft14_diffuco_after_main_payload(root=root)
    available = frozenset(key for key in row if key not in TRACE_METADATA_FIELDS)

    covered: list[str] = []
    missing: list[str] = []
    for field in DIFFUCO_MINIMAL_FIELDS:
        if field.name in DIFFUCO_UNCOVERED_INTERNAL_DIAGNOSTICS:
            continue
        columns = DIFFUCO_AFTER_MAIN_COVERAGE_COLUMNS.get(field.name)
        if columns is None:
            missing.append(field.name)
            continue
        if all(column in available for column in columns):
            covered.append(field.name)
        else:
            missing.append(field.name)

    active_branch_checks = {
        "tag_is_after_diffuco_main": row.get("tag") == DIFFUCO_AFTER_MAIN_TAG,
        "first_pft14_index": row.get("kjit") == 1 and row.get("ji") == 1 and row.get("jv") == 14,
        "pft14_pref_soil_veg_is_4": row.get("jst_pref") == 4,
        "pft14_vegetation_is_active": row.get("veget") == 1.0 and row.get("veget_max") == 1.0,
        "salinity_trace_value_present": _is_finite_number(row.get("salinity")),
        "tide_trace_value_present": _is_finite_number(row.get("tide_height_1")),
    }

    return DiffucoTraceValidation(
        trace_name=DIFFUCO_AFTER_MAIN_TRACE,
        tag=DIFFUCO_AFTER_MAIN_TAG,
        available_columns=available,
        covered_contract_fields=tuple(covered),
        missing_contract_fields=tuple(missing),
        uncovered_internal_diagnostics=DIFFUCO_UNCOVERED_INTERNAL_DIAGNOSTICS,
        active_branch_checks=active_branch_checks,
        provenance=DIFFUCO_AFTER_MAIN_TRACE_PROVENANCE,
    )


def diffuco_numeric_parity_required_fields() -> tuple[str, ...]:
    """Return same-step fields required for strict PFT14 DIFFUCO parity.

    These are fields needed to rerun the local source-backed
    ``diffuco_trans_co2_c3_pft_explicit`` and
    ``diffuco_pft14_c3_beta_closure_explicit`` wrappers against the same
    Fortran timestep. Static PFT parameters may come from the audited
    ``used_run.def``; dynamic process values must come from a trace at the
    DIFFUCO call or inside ``diffuco_trans_co2``.
    """

    return tuple(dict.fromkeys(DIFFUCO_TRANS_CO2_PARITY_REQUIRED_FIELDS + DIFFUCO_BETA_CLOSURE_PARITY_REQUIRED_FIELDS))


def diffuco_numeric_parity_readiness(
    records: Iterable[Mapping[str, object]] | None = None,
    *,
    root: str | Path = TRACE_ROOT,
) -> DiffucoParityReadiness:
    """Report whether existing traces can support strict DIFFUCO parity.

    The current bridge package has an ``after_diffuco_main`` output-boundary
    record, plus driver/slowproc context records. That is useful, but it is not
    sufficient to reconstruct the exact same ``diffuco_trans_co2`` call. This
    function makes the gap executable rather than relying on memory or nearby
    substitutes.
    """

    if records is None:
        records_list: list[Mapping[str, object]] = [
            first_pft14_diffuco_after_main_payload(root=root),
            find_server_record("intersurf_main", tag="main", criteria={"tstep": 1, "ik": 1}, scan_limit=4, root=root)
            or {},
            find_server_record(
                "sechiba_bridge_slowproc",
                tag="before_slowproc_main",
                criteria={"kjit": 1, "ji": 1, "jv": 14},
                scan_limit=4,
                root=root,
            )
            or {},
            find_server_record(
                "sechiba_bridge_hydrol",
                tag="after_hydrol_main_pft",
                criteria={"kjit": 1, "ji": 1, "jv": 14},
                scan_limit=4,
                root=root,
            )
            or {},
        ]
        if trace_available(DIFFUCO_TRANS_CO2_TRACE, root=root):
            records_list.append(first_pft14_diffuco_trans_co2_payload(root=root))
        records = tuple(records_list)

    available: set[str] = set()
    trace_sources: dict[str, str] = {}
    for row in records:
        source = str(row.get("tag", "record"))
        for key, value in row.items():
            if key in TRACE_METADATA_FIELDS:
                continue
            if _is_finite_number(value) or isinstance(value, bool):
                available.add(str(key))
                trace_sources.setdefault(str(key), source)

    required = diffuco_numeric_parity_required_fields()
    return DiffucoParityReadiness(
        available_fields=frozenset(available),
        missing_fields=tuple(field for field in required if field not in available),
        trace_sources=trace_sources,
        provenance=(
            "Strict DIFFUCO numeric parity requires same-timestep dynamic inputs to diffuco_trans_co2 and diffuco_comb.",
            "Existing after_diffuco_main bridge rows are after diffuco_main and omit internal trans_co2 inputs such as qsatt, Ca, vcmax, vbeta23, and controls.",
            "Next trace tag should be outputs/server_trace_patch.bridge_trace_patch_plan:after_diffuco_trans_co2_pft14.",
        ),
    )


def first_pft14_diffuco_critical_values(*, root: str | Path = TRACE_ROOT) -> dict[str, object]:
    """Return trace-read PFT14 first-step values used by DIFFUCO parity tests."""

    payload = first_pft14_diffuco_after_main_payload(root=root)
    return {field: payload[field] for field in DIFFUCO_PFT14_FIRST_TIMESTEP_CRITICAL_FIELDS}


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _is_active_pft14_row(row: Mapping[str, object]) -> bool:
    if row.get("ji") != 1 or row.get("jv") != 14:
        return False
    lai_active = _is_finite_number(row.get("lai")) and float(row["lai"]) > 0.01
    gpp_active = _is_finite_number(row.get("gpp")) and float(row["gpp"]) > 0.0
    return bool(lai_active or gpp_active)
