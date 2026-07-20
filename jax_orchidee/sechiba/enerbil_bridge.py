"""ENERBIL boundary contract for the PFT14 paper-case SECHIBA path.

This module is a source-audit scaffold only. It declares the fields that cross
the `enerbil_main` boundary and never computes or fills missing values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from jax_orchidee.trace.server_1961 import find_server_record, read_server_records


@dataclass(frozen=True)
class EnerbilBoundaryField:
    """One field required at the `enerbil_main` process boundary."""

    name: str
    shape: str
    direction: str
    role: str
    consumers: tuple[str, ...]
    fortran_provenance: tuple[str, ...]
    trace_status: str

    @property
    def requires_trace(self) -> bool:
        """Return True when local traces do not close this field exactly."""

        return self.trace_status != "covered"


@dataclass(frozen=True)
class EnerbilBoundaryValidation:
    """Validation result for an advertised ENERBIL boundary payload."""

    available_fields: frozenset[str]
    missing_fields: tuple[str, ...]
    missing_trace_fields: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return True when all declared boundary fields are present."""

        return not self.missing_fields


@dataclass(frozen=True)
class EnerbilTraceCoverage:
    """Trace coverage for fields read from `after_enerbil_main`."""

    trace_name: str
    tag: str
    criteria: Mapping[str, object]
    covered_fields: frozenset[str]
    missing_fields: tuple[str, ...]
    provenance: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return True when the bridge trace covers every required field."""

        return not self.missing_fields


@dataclass(frozen=True)
class EnerbilActiveModuleClosure:
    """Readiness of active PFT14 ENERBIL module closure traces."""

    before_trace_name: str
    after_trace_name: str
    criteria: Mapping[str, object]
    required_pre_fields: tuple[str, ...]
    required_after_fields: tuple[str, ...]
    available_pre_fields: frozenset[str]
    available_after_fields: frozenset[str]
    missing_pre_fields: tuple[str, ...]
    missing_after_fields: tuple[str, ...]
    provenance: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return True when active pre-call and after-call traces are complete."""

        return not self.missing_pre_fields and not self.missing_after_fields


@dataclass(frozen=True)
class EnerbilToHydrolBoundaryField:
    """One ENERBIL output or inout field consumed by `hydrol_main`."""

    name: str
    shape: str
    direction_from_enerbil: str
    hydrol_role: str
    fortran_provenance: tuple[str, ...]
    after_enerbil_trace_status: str

    @property
    def covered_by_after_enerbil_trace(self) -> bool:
        """Return True when `after_enerbil_main` currently includes the field."""

        return self.after_enerbil_trace_status == "covered"


@dataclass(frozen=True)
class EnerbilToHydrolBoundaryValidation:
    """Name-only validation for the ENERBIL-to-HYDROL boundary payload."""

    required_fields: tuple[str, ...]
    available_fields: frozenset[str]
    covered_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]
    explicit_snow_required_fields: tuple[str, ...]
    missing_explicit_snow_fields: tuple[str, ...]
    provenance: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return True when all fields HYDROL needs after ENERBIL are present."""

        return not self.missing_fields


ENERBIL_CALL_ORDER_PROVENANCE = (
    "src_sechiba/sechiba.f90::sechiba_main calls diffuco_main at lines 997-1005, "
    "enerbil_main at lines 1013-1019, CWRR hydrol_main at lines 1049-1072, "
    "condveg_main at lines 1085-1091, thermosoil_main at lines 1109-1118, "
    "and slowproc_main at lines 1184-1216.",
)

ENERBIL_AFTER_MAIN_TRACE_NAME = "sechiba_bridge_enerbil"
ENERBIL_AFTER_MAIN_TAG = "after_enerbil_main"
ENERBIL_ACTIVE_TRACE_NAME = "sechiba_bridge_enerbil_active"
ENERBIL_ACTIVE_BEFORE_TAG = "before_enerbil_main_active_pft14"
ENERBIL_ACTIVE_AFTER_TAG = "after_enerbil_main_active_pft14"
ENERBIL_POTTEMP_TRACE_NAME = "enerbil_pottemp_active"
ENERBIL_POTTEMP_BEFORE_TAG = "before_enerbil_pottemp_active"
ENERBIL_POTTEMP_AFTER_TAG = "after_enerbil_pottemp_active"

ENERBIL_AFTER_MAIN_TRACE_PROVENANCE = (
    "outputs/server_1961_bridge_trace_20260624/traces/orchjax_sechiba_bridge_enerbil_trace.txt:after_enerbil_main",
    "docs/source_audits/server_1961_bridge_trace_patch_plan.md#after_enerbil_main",
    "src_sechiba/sechiba.f90::sechiba_main after CALL enerbil_main at lines 1013-1019.",
    "src_sechiba/enerbil.f90::enerbil_main process boundary at lines 390-471.",
)

ENERBIL_AFTER_MAIN_TRACE_FIELDS = (
    "transpir",
    "transpot",
    "vevapwet",
    "vevapnu",
    "vevapnu_pft",
    "vevapsno",
    "vevapflo",
    "evapot",
    "evapot_corr",
    "temp_sol",
    "temp_sol_pft",
    "temp_sol_new",
    "temp_sol_new_pft",
    "qsurf",
    "t2mdiag",
)

ENERBIL_ACTIVE_PRECALL_FIELDS = (
    "lai",
    "gpp",
    "veget_max",
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
    "valpha",
    "vbeta",
    "vbeta_pft",
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
    "humrel",
    "precip_rain",
    "snowdz_1",
    "pgflux",
    "temp_sol",
    "temp_sol_pft",
    "qsurf",
    "evapot",
    "evapot_corr",
)

ENERBIL_ACTIVE_AFTER_FIELDS = (
    "lai",
    "gpp",
    "veget_max",
    "transpir",
    "transpot",
    "vevapwet",
    "vevapnu",
    "vevapnu_pft",
    "vevapsno",
    "vevapflo",
    "vevapp",
    "evapot",
    "evapot_corr",
    "temp_sol",
    "temp_sol_pft",
    "temp_sol_new",
    "temp_sol_new_pft",
    "qsurf",
    "t2mdiag",
    "fluxsens",
    "fluxlat",
    "pgflux",
    "temp_sol_add",
    "soilcap",
    "soilcap_pft",
    "soilflx",
    "soilflx_pft",
    "snowdz_1",
    "precip_rain",
)

ENERBIL_ACTIVE_TRACE_PROVENANCE = (
    "outputs/server_trace_patch/apply_enerbil_active_trace_patch.py",
    "src_sechiba/sechiba.f90::sechiba_main after diffuco_main and before enerbil_main at lines 997-1019.",
    "src_sechiba/sechiba.f90::sechiba_main after enerbil_main and before hydrol_main at lines 1013-1049.",
    "src_sechiba/enerbil.f90::enerbil_main process boundary at lines 390-598.",
)

ENERBIL_TO_HYDROL_EXPLICIT_SNOW_FIELDS = ("pgflux", "temp_sol_add")

ENERBIL_TO_HYDROL_BOUNDARY_PROVENANCE = (
    "src_sechiba/enerbil.f90::enerbil_main declares HYDROL-facing water flux outputs "
    "and snow energy fields at lines 449-472.",
    "src_sechiba/enerbil.f90::enerbil_flux updates pgflux/temp_sol_add for explicit snow "
    "at lines 1541-1588.",
    "src_sechiba/enerbil.f90::enerbil_evapveg computes vevapsno, vevapnu, vevapnu_pft, "
    "vevapflo, vevapwet, transpir, and transpot at lines 1756-1833.",
    "src_sechiba/sechiba.f90::sechiba_main passes ENERBIL outputs into hydrol_main "
    "at lines 1049-1064.",
    "src_sechiba/hydrol.f90::hydrol_main declares ENERBIL-derived water and evaporation "
    "inputs at lines 984-1002.",
    "src_sechiba/hydrol.f90::hydrol_main declares pgflux input at line 1010.",
    "src_sechiba/hydrol.f90::hydrol_main declares evapot/evapot_penm and vevapflo "
    "at lines 1064-1067.",
    "src_sechiba/hydrol.f90::hydrol_main declares temp_sol_add inout at line 1090.",
    "src_sechiba/hydrol.f90::hydrol_main passes temp_sol_new, pgflux, vevapsno, "
    "and temp_sol_add through explicit-snow handling at lines 1182-1196.",
    "src_sechiba/hydrol.f90::hydrol_main passes vevapwet/vevapflo to canopy and flood "
    "updates at lines 1232-1238.",
    "src_sechiba/hydrol.f90::hydrol_main passes transpir, vevapnu, vevapnu_pft, evapot, "
    "and evapot_penm to hydrol_soil at lines 1278-1284.",
)

ENERBIL_TO_HYDROL_REQUIRED_FIELDS = (
    EnerbilToHydrolBoundaryField(
        "temp_sol_new",
        "(npts)",
        "output",
        "surface temperature passed to explicit-snow handling",
        (
            "src_sechiba/enerbil.f90::enerbil_main declares temp_sol_new output at lines 456-459.",
            "src_sechiba/sechiba.f90::sechiba_main passes temp_sol_new to hydrol_main at lines 1049-1064.",
            "src_sechiba/hydrol.f90::hydrol_main passes temp_sol_new to explicitsnow_main at lines 1182-1196.",
        ),
        "covered",
    ),
    EnerbilToHydrolBoundaryField(
        "transpir",
        "(npts,nvm)",
        "output",
        "PFT transpiration passed into the soil water solve",
        (
            "src_sechiba/enerbil.f90::enerbil_main declares transpir output at lines 449-455.",
            "src_sechiba/enerbil.f90::enerbil_evapveg computes transpir at lines 1756-1833.",
            "src_sechiba/hydrol.f90::hydrol_main passes transpir to hydrol_soil at lines 1278-1284.",
        ),
        "covered",
    ),
    EnerbilToHydrolBoundaryField(
        "transpot",
        "(npts,nvm)",
        "output",
        "PFT potential transpiration used by HYDROL stress and water context",
        (
            "src_sechiba/enerbil.f90::enerbil_main declares transpot output at lines 449-455.",
            "src_sechiba/enerbil.f90::enerbil_evapveg computes transpot at lines 1756-1833.",
            "src_sechiba/hydrol.f90::hydrol_main declares transpot input at line 985.",
        ),
        "covered",
    ),
    EnerbilToHydrolBoundaryField(
        "vevapwet",
        "(npts,nvm)",
        "output",
        "interception evaporation consumed by canopy storage update",
        (
            "src_sechiba/enerbil.f90::enerbil_main declares vevapwet output at lines 449-455.",
            "src_sechiba/enerbil.f90::enerbil_evapveg computes vevapwet at lines 1756-1833.",
            "src_sechiba/hydrol.f90::hydrol_main passes vevapwet to hydrol_canop at lines 1232-1238.",
        ),
        "covered",
    ),
    EnerbilToHydrolBoundaryField(
        "vevapnu",
        "(npts)",
        "output",
        "grid-cell bare-soil evaporation passed into soil water solve",
        (
            "src_sechiba/enerbil.f90::enerbil_main declares vevapnu output at lines 449-455.",
            "src_sechiba/enerbil.f90::enerbil_evapveg computes vevapnu at lines 1756-1833.",
            "src_sechiba/hydrol.f90::hydrol_main passes vevapnu to hydrol_soil at lines 1278-1284.",
        ),
        "covered",
    ),
    EnerbilToHydrolBoundaryField(
        "vevapnu_pft",
        "(npts,nvm)",
        "output",
        "PFT bare-soil evaporation passed into soil water solve",
        (
            "src_sechiba/enerbil.f90::enerbil_main declares vevapnu_pft output at lines 449-455.",
            "src_sechiba/enerbil.f90::enerbil_evapveg computes vevapnu_pft at lines 1756-1833.",
            "src_sechiba/hydrol.f90::hydrol_main passes vevapnu_pft to hydrol_soil at lines 1278-1284.",
        ),
        "covered",
    ),
    EnerbilToHydrolBoundaryField(
        "vevapsno",
        "(npts)",
        "output",
        "snow sublimation passed into explicit-snow or bucket-snow handling",
        (
            "src_sechiba/enerbil.f90::enerbil_main declares vevapsno output at lines 449-455.",
            "src_sechiba/enerbil.f90::enerbil_evapveg computes vevapsno at lines 1756-1833.",
            "src_sechiba/hydrol.f90::hydrol_main passes vevapsno through snow handling at lines 1182-1196.",
        ),
        "covered",
    ),
    EnerbilToHydrolBoundaryField(
        "vevapflo",
        "(npts)",
        "output",
        "floodplain evaporation constrained by HYDROL flood reservoir",
        (
            "src_sechiba/enerbil.f90::enerbil_main declares vevapflo output at lines 449-455.",
            "src_sechiba/enerbil.f90::enerbil_evapveg computes vevapflo at lines 1756-1833.",
            "src_sechiba/hydrol.f90::hydrol_main passes vevapflo to hydrol_flood at lines 1232-1238.",
        ),
        "covered",
    ),
    EnerbilToHydrolBoundaryField(
        "evapot",
        "(npts)",
        "inout",
        "uncorrected potential evaporation passed into soil water solve",
        (
            "src_sechiba/enerbil.f90::enerbil_main declares evapot inout at line 463.",
            "src_sechiba/enerbil.f90::enerbil_flux computes evapot in the line 1541-1588 span.",
            "src_sechiba/hydrol.f90::hydrol_main passes evapot to hydrol_soil at lines 1278-1284.",
        ),
        "covered",
    ),
    EnerbilToHydrolBoundaryField(
        "evapot_corr",
        "(npts)",
        "inout",
        "corrected potential evaporation passed to HYDROL as evapot_penm",
        (
            "src_sechiba/enerbil.f90::enerbil_main declares evapot_corr inout at line 464.",
            "src_sechiba/sechiba.f90::sechiba_main passes evapot_corr into hydrol_main at lines 1049-1064.",
            "src_sechiba/hydrol.f90::hydrol_main declares evapot_penm at lines 1064-1067 "
            "and passes it to hydrol_soil at lines 1278-1284.",
        ),
        "covered",
    ),
    EnerbilToHydrolBoundaryField(
        "pgflux",
        "(npts)",
        "inout",
        "explicit-snow net energy into snowpack required by HYDROL",
        (
            "src_sechiba/enerbil.f90::enerbil_main declares pgflux inout at line 472.",
            "src_sechiba/enerbil.f90::enerbil_flux updates pgflux for explicit snow at lines 1541-1588.",
            "src_sechiba/hydrol.f90::hydrol_main declares pgflux input at line 1010 "
            "and passes it to explicitsnow_main at lines 1182-1196.",
        ),
        "missing_from_after_enerbil_main_trace",
    ),
    EnerbilToHydrolBoundaryField(
        "temp_sol_add",
        "(npts)",
        "output",
        "explicit-snow melt-energy temperature increment required by HYDROL",
        (
            "src_sechiba/enerbil.f90::enerbil_main declares temp_sol_add output at line 459.",
            "src_sechiba/enerbil.f90::enerbil_flux updates temp_sol_add for explicit snow at lines 1541-1588.",
            "src_sechiba/hydrol.f90::hydrol_main declares temp_sol_add inout at line 1090 "
            "and passes it to explicitsnow_main at lines 1182-1196.",
        ),
        "missing_from_after_enerbil_main_trace",
    ),
)


ENERBIL_REQUIRED_FIELDS = (
    EnerbilBoundaryField(
        "temp_sol_new",
        "(npts)",
        "output",
        "new grid-cell surface temperature from the implicit energy balance",
        ("hydrol_main snow/soil water update", "thermosoil_main surface thermal forcing"),
        (
            "src_sechiba/enerbil.f90::enerbil_main declares temp_sol_new output at lines 456-459.",
            "src_sechiba/enerbil.f90::enerbil_surftemp computes temp_sol_new at lines 927-1245; "
            "assignment is at line 1203.",
            "src_sechiba/sechiba.f90::sechiba_main passes temp_sol_new to hydrol_main at lines 1049-1055 "
            "and thermosoil_main at lines 1109-1118.",
        ),
        "missing",
    ),
    EnerbilBoundaryField(
        "temp_sol_new_pft",
        "(npts,nvm)",
        "output",
        "new PFT-resolved surface temperature for thermosoil PFT heat state",
        ("thermosoil_main",),
        (
            "src_sechiba/enerbil.f90::enerbil_main declares temp_sol_new_pft output at lines 456-459.",
            "src_sechiba/enerbil.f90::enerbil_surftemp computes temp_sol_new_pft at lines 1204-1212.",
            "src_sechiba/sechiba.f90::sechiba_main passes temp_sol_new_pft to thermosoil_main at lines 1109-1118.",
        ),
        "missing",
    ),
    EnerbilBoundaryField(
        "qsurf",
        "(npts)",
        "inout",
        "surface specific humidity updated from the new saturated surface humidity",
        ("next diffuco_main", "driver/restart diagnostics"),
        (
            "src_sechiba/enerbil.f90::enerbil_main declares qsurf inout at line 467.",
            "src_sechiba/enerbil.f90::enerbil_flux computes qsurf at lines 1481-1492.",
            "src_sechiba/diffuco.f90::diffuco_main declares qsurf input at lines 307-341; "
            "src_sechiba/sechiba.f90 passes qsurf into diffuco_main at lines 997-1002.",
        ),
        "missing",
    ),
    EnerbilBoundaryField(
        "evapot",
        "(npts)",
        "inout",
        "uncorrected potential bare-soil evaporation used by HYDROL and diagnostics",
        ("hydrol_main", "modelout/history", "dynpeat_PWT branch when active"),
        (
            "src_sechiba/enerbil.f90::enerbil_main declares evapot inout at line 463.",
            "src_sechiba/enerbil.f90::enerbil_flux computes evapot at line 1545.",
            "src_sechiba/sechiba.f90::sechiba_main passes evapot to hydrol_main at lines 1049-1055.",
            "src_sechiba/sechiba.f90::sechiba_main uses evapot in dynpeat_PWT peat_PET code at lines 1139-1159.",
        ),
        "missing",
    ),
    EnerbilBoundaryField(
        "evapot_corr",
        "(npts)",
        "inout",
        "Milly/Penman-corrected potential evaporation passed to HYDROL and daily forcing",
        ("hydrol_main", "slowproc_main/STOMATE daily forcing", "modelout/history"),
        (
            "src_sechiba/enerbil.f90::enerbil_main declares evapot_corr inout at line 464.",
            "src_sechiba/enerbil.f90::enerbil_flux computes evapot_corr at lines 1613-1640.",
            "src_sechiba/sechiba.f90::sechiba_main passes evapot_corr to hydrol_main at lines 1049-1055 "
            "and slowproc_main at lines 1196-1198.",
        ),
        "missing",
    ),
    EnerbilBoundaryField(
        "transpir",
        "(npts,nvm)",
        "output",
        "PFT transpiration, consumed immediately by HYDROL water balance",
        ("hydrol_main",),
        (
            "src_sechiba/enerbil.f90::enerbil_main declares transpir output at lines 449-455.",
            "src_sechiba/enerbil.f90::enerbil_evapveg computes transpir at lines 1807-1833.",
            "src_sechiba/sechiba.f90::sechiba_main passes transpir to hydrol_main at lines 1049-1055.",
            "src_sechiba/hydrol.f90::hydrol_main passes transpir into hydrol_soil at lines 1278-1284.",
        ),
        "missing",
    ),
    EnerbilBoundaryField(
        "transpot",
        "(npts,nvm)",
        "output",
        "PFT potential transpiration, used by HYDROL stress/irrigation/routing context",
        ("hydrol_main", "routing_main when active"),
        (
            "src_sechiba/enerbil.f90::enerbil_main declares transpot output at lines 449-455.",
            "src_sechiba/enerbil.f90::enerbil_evapveg computes transpot at lines 1807-1833.",
            "src_sechiba/sechiba.f90::sechiba_main passes transpot to hydrol_main at lines 1049-1055.",
            "src_sechiba/sechiba.f90::sechiba_main passes transpot to routing_main when "
            "river_routing .AND. nbp_glo .GT. 1 at lines 1227-1234.",
        ),
        "missing",
    ),
    EnerbilBoundaryField(
        "vevapnu",
        "(npts)",
        "output",
        "grid-cell bare soil evaporation, modified by HYDROL after ENERBIL",
        ("hydrol_main",),
        (
            "src_sechiba/enerbil.f90::enerbil_main declares vevapnu output at lines 449-455.",
            "src_sechiba/enerbil.f90::enerbil_evapveg computes vevapnu at lines 1762-1781.",
            "src_sechiba/sechiba.f90::sechiba_main passes vevapnu to hydrol_main at lines 1049-1055.",
            "src_sechiba/hydrol.f90::hydrol_split_soil consumes vevapnu at lines 8603-8730.",
        ),
        "missing",
    ),
    EnerbilBoundaryField(
        "vevapnu_pft",
        "(npts,nvm)",
        "output",
        "PFT bare soil evaporation used to split soil evaporation over soil tiles",
        ("hydrol_main",),
        (
            "src_sechiba/enerbil.f90::enerbil_main declares vevapnu_pft output at lines 449-455.",
            "src_sechiba/enerbil.f90::enerbil_evapveg computes vevapnu_pft at lines 1768-1781.",
            "src_sechiba/sechiba.f90::sechiba_main passes vevapnu_pft to hydrol_main at lines 1049-1055.",
            "src_sechiba/hydrol.f90::hydrol_split_soil consumes vevapnu_pft at lines 8603-8730.",
        ),
        "missing",
    ),
    EnerbilBoundaryField(
        "vevapwet",
        "(npts,nvm)",
        "output",
        "PFT interception evaporation, consumed by HYDROL canopy reservoir update",
        ("hydrol_main/hydrol_canop",),
        (
            "src_sechiba/enerbil.f90::enerbil_main declares vevapwet output at lines 449-455.",
            "src_sechiba/enerbil.f90::enerbil_evapveg computes vevapwet at lines 1807-1833.",
            "src_sechiba/sechiba.f90::sechiba_main passes vevapwet to hydrol_main at lines 1049-1055.",
            "src_sechiba/hydrol.f90::hydrol_main passes vevapwet to hydrol_canop at lines 1232-1234.",
        ),
        "missing",
    ),
    EnerbilBoundaryField(
        "vevapsno",
        "(npts)",
        "output",
        "snow sublimation, consumed by explicit snow or bucket snow update",
        ("hydrol_main snow branch",),
        (
            "src_sechiba/enerbil.f90::enerbil_main declares vevapsno output at lines 449-455.",
            "src_sechiba/enerbil.f90::enerbil_evapveg computes vevapsno at lines 1756-1761.",
            "src_sechiba/hydrol.f90::hydrol_main passes vevapsno to explicitsnow_main at lines 1177-1190 "
            "or hydrol_snow at lines 1192-1197.",
        ),
        "missing",
    ),
    EnerbilBoundaryField(
        "vevapflo",
        "(npts)",
        "output",
        "floodplain evaporation, active formula output but constrained by HYDROL flood reservoir",
        ("hydrol_main/hydrol_flood",),
        (
            "src_sechiba/enerbil.f90::enerbil_main declares vevapflo output at lines 449-455.",
            "src_sechiba/enerbil.f90::enerbil_evapveg computes vevapflo at lines 1783-1787.",
            "src_sechiba/hydrol.f90::hydrol_main passes vevapflo to hydrol_flood at lines 1237-1238.",
        ),
        "missing",
    ),
    EnerbilBoundaryField(
        "temp_sol",
        "(npts)",
        "inout",
        "current surface temperature retained for slowproc/STOMATE forcing and restart",
        ("slowproc_main", "next diffuco_main", "next enerbil_main"),
        (
            "src_sechiba/enerbil.f90::enerbil_main declares temp_sol inout at line 465.",
            "src_sechiba/enerbil.f90::enerbil_flux uses temp_sol to compute longwave/radiative temperature "
            "at lines 1462-1479.",
            "src_sechiba/sechiba.f90::sechiba_main passes temp_sol to slowproc_main at lines 1184-1188.",
        ),
        "missing",
    ),
    EnerbilBoundaryField(
        "temp_sol_pft",
        "(npts,nvm)",
        "inout",
        "PFT current surface temperature used by ENERBIL and next timestep",
        ("next diffuco_main", "next enerbil_main"),
        (
            "src_sechiba/enerbil.f90::enerbil_main declares temp_sol_pft inout at line 466.",
            "src_sechiba/enerbil.f90::enerbil_begin uses temp_sol_pft in the call at lines 488-489.",
            "src_sechiba/sechiba.f90::sechiba_main passes temp_sol_pft into enerbil_main at lines 1013-1019.",
        ),
        "missing",
    ),
    EnerbilBoundaryField(
        "t2mdiag",
        "(npts)",
        "output",
        "2 m diagnostic air temperature passed into daily slow processes",
        ("slowproc_main/STOMATE daily forcing",),
        (
            "src_sechiba/enerbil.f90::enerbil_main declares t2mdiag output at line 456.",
            "src_sechiba/enerbil.f90::enerbil_main calls enerbil_t2mdiag at lines 547-551.",
            "src_sechiba/enerbil.f90::enerbil_t2mdiag is defined at lines 1993-2021.",
            "src_sechiba/sechiba.f90::sechiba_main passes t2mdiag to slowproc_main at lines 1184-1198.",
        ),
        "missing",
    ),
)


def enerbil_boundary_field_names() -> tuple[str, ...]:
    """Return the minimal ENERBIL boundary field names for the PFT14 path."""

    return tuple(field.name for field in ENERBIL_REQUIRED_FIELDS)


def enerbil_boundary_field(name: str) -> EnerbilBoundaryField:
    """Return one ENERBIL boundary field by name."""

    for field in ENERBIL_REQUIRED_FIELDS:
        if field.name == name:
            return field
    raise KeyError(f"unknown ENERBIL boundary field: {name}")


def enerbil_required_trace_fields() -> tuple[str, ...]:
    """Return ENERBIL fields that still require exact after-call trace support."""

    return tuple(field.name for field in ENERBIL_REQUIRED_FIELDS if field.requires_trace)


def enerbil_after_main_trace_fields() -> tuple[str, ...]:
    """Return required ENERBIL fields covered by `after_enerbil_main`.

    The values are read from the audited bridge trace only. This helper does
    not imply that the local ENERBIL formulas have been implemented.
    """

    required = frozenset(enerbil_boundary_field_names())
    return tuple(name for name in ENERBIL_AFTER_MAIN_TRACE_FIELDS if name in required)


def enerbil_active_precall_trace_fields() -> tuple[str, ...]:
    """Return planned active pre-call fields for PFT14 ENERBIL closure."""

    return ENERBIL_ACTIVE_PRECALL_FIELDS


def enerbil_active_after_trace_fields() -> tuple[str, ...]:
    """Return planned active after-call fields for PFT14 ENERBIL closure."""

    return ENERBIL_ACTIVE_AFTER_FIELDS


def enerbil_to_hydrol_boundary_field_names() -> tuple[str, ...]:
    """Return the exact ENERBIL fields consumed by `hydrol_main` afterward."""

    return tuple(field.name for field in ENERBIL_TO_HYDROL_REQUIRED_FIELDS)


def enerbil_to_hydrol_boundary_field(name: str) -> EnerbilToHydrolBoundaryField:
    """Return one ENERBIL-to-HYDROL boundary field by name."""

    for field in ENERBIL_TO_HYDROL_REQUIRED_FIELDS:
        if field.name == name:
            return field
    raise KeyError(f"unknown ENERBIL-to-HYDROL boundary field: {name}")


def enerbil_to_hydrol_after_main_trace_fields() -> tuple[str, ...]:
    """Return HYDROL boundary fields currently present in `after_enerbil_main`."""

    trace_fields = frozenset(ENERBIL_AFTER_MAIN_TRACE_FIELDS)
    return tuple(name for name in enerbil_to_hydrol_boundary_field_names() if name in trace_fields)


def enerbil_to_hydrol_missing_after_main_trace_fields() -> tuple[str, ...]:
    """Return HYDROL boundary fields absent from the current ENERBIL trace."""

    covered = frozenset(enerbil_to_hydrol_after_main_trace_fields())
    return tuple(name for name in enerbil_to_hydrol_boundary_field_names() if name not in covered)


def read_enerbil_after_main_payload(
    *,
    kjit: int = 1,
    ji: int = 1,
    jv: int = 14,
    root: str | Path | None = None,
) -> dict[str, object]:
    """Read the PFT-specific `after_enerbil_main` payload from bridge trace.

    Defaults select the first paper-case timestep for land point 1 and PFT14.
    Values are returned exactly as parsed by
    :mod:`jax_orchidee.trace.server_1961`, including non-required diagnostics
    such as `soilcap_pft`.
    """

    criteria = {"kjit": kjit, "ji": ji, "jv": jv}
    kwargs: dict[str, object] = {
        "tag": ENERBIL_AFTER_MAIN_TAG,
        "criteria": criteria,
        "scan_limit": 64,
    }
    if root is not None:
        kwargs["root"] = root

    row = find_server_record(ENERBIL_AFTER_MAIN_TRACE_NAME, **kwargs)
    if row is None:
        raise LookupError(
            "missing after_enerbil_main trace record for "
            f"kjit={kjit}, ji={ji}, jv={jv}"
        )
    return dict(row)


def enerbil_after_main_trace_coverage(
    payload: Mapping[str, object] | None = None,
    *,
    kjit: int = 1,
    ji: int = 1,
    jv: int = 14,
    root: str | Path | None = None,
) -> EnerbilTraceCoverage:
    """Report which ENERBIL required fields are covered by bridge trace."""

    trace_payload = (
        read_enerbil_after_main_payload(kjit=kjit, ji=ji, jv=jv, root=root)
        if payload is None
        else payload
    )
    covered = frozenset(
        name for name in enerbil_boundary_field_names() if name in trace_payload
    )
    missing = tuple(name for name in enerbil_boundary_field_names() if name not in covered)
    return EnerbilTraceCoverage(
        trace_name=ENERBIL_AFTER_MAIN_TRACE_NAME,
        tag=ENERBIL_AFTER_MAIN_TAG,
        criteria={"kjit": kjit, "ji": ji, "jv": jv},
        covered_fields=covered,
        missing_fields=missing,
        provenance=ENERBIL_AFTER_MAIN_TRACE_PROVENANCE,
    )


def read_enerbil_after_main_records(
    *,
    limit: int | None = None,
    root: str | Path | None = None,
) -> tuple[dict[str, object], ...]:
    """Read parsed `after_enerbil_main` records from the bridge trace."""

    kwargs: dict[str, object] = {"tags": ENERBIL_AFTER_MAIN_TAG, "limit": limit}
    if root is not None:
        kwargs["root"] = root
    return read_server_records(ENERBIL_AFTER_MAIN_TRACE_NAME, **kwargs)


def read_enerbil_active_records(
    *,
    tag: str | None = None,
    limit: int | None = None,
    root: str | Path | None = None,
) -> tuple[dict[str, object], ...]:
    """Read planned active ENERBIL records when the trace package is present."""

    kwargs: dict[str, object] = {"limit": limit}
    if tag is not None:
        kwargs["tags"] = tag
    if root is not None:
        kwargs["root"] = root
    return read_server_records(ENERBIL_ACTIVE_TRACE_NAME, **kwargs)


def read_enerbil_active_payload(
    *,
    tag: str,
    kjit: int,
    ji: int = 1,
    jv: int = 14,
    root: str | Path | None = None,
) -> dict[str, object]:
    """Read one active ENERBIL PFT14 payload by exact boundary tag."""

    criteria = {"kjit": kjit, "ji": ji, "jv": jv}
    kwargs: dict[str, object] = {
        "tag": tag,
        "criteria": criteria,
        "scan_limit": 4096,
    }
    if root is not None:
        kwargs["root"] = root
    row = find_server_record(ENERBIL_ACTIVE_TRACE_NAME, **kwargs)
    if row is None:
        raise LookupError(f"missing {tag} trace record for kjit={kjit}, ji={ji}, jv={jv}")
    return dict(row)


def read_enerbil_pottemp_active_records(
    *,
    tag: str | None = None,
    limit: int | None = None,
    root: str | Path | None = None,
) -> tuple[dict[str, object], ...]:
    """Read active ``enerbil_pottemp`` records when the trace is present."""

    kwargs: dict[str, object] = {"limit": limit}
    if tag is not None:
        kwargs["tags"] = tag
    if root is not None:
        kwargs["root"] = root
    return read_server_records(ENERBIL_POTTEMP_TRACE_NAME, **kwargs)


def read_enerbil_pottemp_active_payload(
    *,
    tag: str,
    kjit: int,
    ji: int = 1,
    root: str | Path | None = None,
) -> dict[str, object]:
    """Read one active ``enerbil_pottemp`` payload by exact trace tag."""

    criteria = {"kjit": kjit, "ji": ji}
    kwargs: dict[str, object] = {
        "tag": tag,
        "criteria": criteria,
        "scan_limit": 4096,
    }
    if root is not None:
        kwargs["root"] = root
    row = find_server_record(ENERBIL_POTTEMP_TRACE_NAME, **kwargs)
    if row is None:
        raise LookupError(f"missing {tag} trace record for kjit={kjit}, ji={ji}")
    return dict(row)


def validate_enerbil_active_module_closure(
    *,
    before_payload: Mapping[str, object] | Iterable[str] | None = None,
    after_payload: Mapping[str, object] | Iterable[str] | None = None,
    kjit: int | None = None,
    ji: int = 1,
    jv: int = 14,
    root: str | Path | None = None,
) -> EnerbilActiveModuleClosure:
    """Validate active PFT14 ENERBIL pre-call and after-call trace coverage.

    This is a trace-contract validator only. It does not compute ENERBIL
    values or back-fill missing pre-call state from after-call outputs.
    """

    if before_payload is None and kjit is not None:
        before_payload = read_enerbil_active_payload(
            tag=ENERBIL_ACTIVE_BEFORE_TAG,
            kjit=kjit,
            ji=ji,
            jv=jv,
            root=root,
        )
    if after_payload is None and kjit is not None:
        after_payload = read_enerbil_active_payload(
            tag=ENERBIL_ACTIVE_AFTER_TAG,
            kjit=kjit,
            ji=ji,
            jv=jv,
            root=root,
        )

    before_fields = _available_fields(before_payload)
    after_fields = _available_fields(after_payload)
    missing_pre = tuple(field for field in ENERBIL_ACTIVE_PRECALL_FIELDS if field not in before_fields)
    missing_after = tuple(field for field in ENERBIL_ACTIVE_AFTER_FIELDS if field not in after_fields)
    criteria = {"ji": ji, "jv": jv}
    if kjit is not None:
        criteria = {"kjit": kjit, **criteria}
    return EnerbilActiveModuleClosure(
        before_trace_name=ENERBIL_ACTIVE_TRACE_NAME,
        after_trace_name=ENERBIL_ACTIVE_TRACE_NAME,
        criteria=criteria,
        required_pre_fields=ENERBIL_ACTIVE_PRECALL_FIELDS,
        required_after_fields=ENERBIL_ACTIVE_AFTER_FIELDS,
        available_pre_fields=before_fields,
        available_after_fields=after_fields,
        missing_pre_fields=missing_pre,
        missing_after_fields=missing_after,
        provenance=ENERBIL_ACTIVE_TRACE_PROVENANCE,
        notes=(
            "before_enerbil_main_active_pft14 is the only same-call source for ENERBIL pre-call swnet, soilflx*, soilcap*, pgflux, and driver coefficients in this closure contract.",
            "after_enerbil_main_active_pft14 verifies ENERBIL outputs consumed by HYDROL, including pgflux and temp_sol_add for the explicit-snow boundary.",
            "The active predicate is auditable through lai, gpp, and veget_max fields in both records.",
        ),
    )


def validate_enerbil_boundary_payload(
    payload: Mapping[str, object] | Iterable[str],
) -> EnerbilBoundaryValidation:
    """Validate that a payload advertises all ENERBIL boundary fields.

    Validation is name-only. It intentionally does not synthesize any missing
    process value.
    """

    if isinstance(payload, Mapping):
        available = frozenset(str(key) for key in payload.keys())
    else:
        available = frozenset(str(item) for item in payload)

    required = enerbil_boundary_field_names()
    missing = tuple(name for name in required if name not in available)

    return EnerbilBoundaryValidation(
        available_fields=available,
        missing_fields=missing,
        missing_trace_fields=enerbil_required_trace_fields(),
    )


def _available_fields(payload: Mapping[str, object] | Iterable[str] | None) -> frozenset[str]:
    if payload is None:
        return frozenset()
    if isinstance(payload, Mapping):
        return frozenset(str(key) for key in payload.keys())
    return frozenset(str(item) for item in payload)


def validate_enerbil_to_hydrol_boundary_payload(
    payload: Mapping[str, object] | Iterable[str],
) -> EnerbilToHydrolBoundaryValidation:
    """Validate the ENERBIL payload needed by the following `hydrol_main` call.

    Validation is name-only and does not synthesize missing process values.
    `pgflux` and `temp_sol_add` are explicit-snow HYDROL inputs and are required
    even though the current `after_enerbil_main` trace does not contain them.
    """

    if isinstance(payload, Mapping):
        available = frozenset(str(key) for key in payload.keys())
    else:
        available = frozenset(str(item) for item in payload)

    required = enerbil_to_hydrol_boundary_field_names()
    covered = tuple(name for name in required if name in available)
    missing = tuple(name for name in required if name not in available)
    missing_explicit_snow = tuple(
        name for name in ENERBIL_TO_HYDROL_EXPLICIT_SNOW_FIELDS if name not in available
    )

    return EnerbilToHydrolBoundaryValidation(
        required_fields=required,
        available_fields=available,
        covered_fields=covered,
        missing_fields=missing,
        explicit_snow_required_fields=ENERBIL_TO_HYDROL_EXPLICIT_SNOW_FIELDS,
        missing_explicit_snow_fields=missing_explicit_snow,
        provenance=ENERBIL_TO_HYDROL_BOUNDARY_PROVENANCE,
    )
