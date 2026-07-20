"""SECHIBA bridge field contract for the PFT14 paper-case path.

This module declares boundary fields only. It deliberately does not compute
missing SECHIBA state or fill trace gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from jax_orchidee.trace.server_1961 import find_server_record


@dataclass(frozen=True)
class BridgeField:
    """One required field crossing a SECHIBA process boundary."""

    name: str
    shape: str
    producer: str
    consumers: tuple[str, ...]
    fortran_provenance: tuple[str, ...]
    required_for: tuple[str, ...]
    server_1961_trace: str
    aliases: tuple[str, ...] = ()

    @property
    def requires_trace(self) -> bool:
        """Return True when current server_1961 traces do not close this field."""

        return self.server_1961_trace in {"partial", "missing"}

    @property
    def accepted_names(self) -> tuple[str, ...]:
        """Return canonical and source/JAX aliases accepted at this boundary."""

        return (self.name, *self.aliases)


@dataclass(frozen=True)
class BridgeGroup:
    """Named group of bridge fields sharing one process boundary."""

    name: str
    description: str
    fields: tuple[BridgeField, ...]

    @property
    def requires_trace(self) -> tuple[str, ...]:
        """Names of fields that still need exact Fortran trace support."""

        return tuple(field.name for field in self.fields if field.requires_trace)


@dataclass(frozen=True)
class BridgeValidation:
    """Grouped bridge payload validation result."""

    available_fields: frozenset[str]
    missing_by_group: dict[str, tuple[str, ...]]
    requires_trace_by_group: dict[str, tuple[str, ...]]

    @property
    def ok(self) -> bool:
        """Return True when all requested bridge fields are present."""

        return all(not missing for missing in self.missing_by_group.values())


@dataclass(frozen=True)
class BridgeTraceMapping:
    """One trace-backed equality across the HYDROL/slowproc/STOMATE boundary."""

    name: str
    source_record: str
    source_tag: str
    source_field: str
    target_record: str
    target_tag: str
    target_field: str
    fortran_provenance: tuple[str, ...]


@dataclass(frozen=True)
class BridgeTraceSlice:
    """PFT14 bridge trace records needed for one HYDROL continuity slice."""

    pft: dict[str, object]
    layer: dict[str, object]
    tile_layer: dict[str, object]
    tile: dict[str, object]
    before_slowproc: dict[str, object]
    before_stomate: dict[str, object]
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class BridgeTracePair:
    """Parsed source/target values for one bridge trace mapping."""

    mapping: BridgeTraceMapping
    source_value: object
    target_value: object


CALL_ORDER_PROVENANCE = (
    "src_sechiba/intersurf.f90::intersurf_main_2d lines 458-761; "
    "calls sechiba_main at lines 640-648.",
    "src_sechiba/sechiba.f90::sechiba_main lines 997-1216; active order is "
    "diffuco_main, enerbil_main, hydrol_main, condveg_main, thermosoil_main, "
    "slowproc_main.",
)


HYDROL_STOMATE_FIELDS = (
    BridgeField(
        "humrel",
        "(npts,nvm)",
        "hydrol_main/hydrol_soil",
        ("slowproc_main", "stomate_main daily accumulation", "next diffuco_main", "next enerbil_main"),
        (
            "src_sechiba/hydrol.f90::hydrol_main lines 938-1090; humrel INTENT(inout) at line 1063.",
            "src_sechiba/sechiba.f90::sechiba_main passes humrel to slowproc_main at lines 1184-1188.",
            "src_stomate/stomate.f90::stomate_main accumulates humrel at lines 3198-3208.",
            "src_sechiba/diffuco.f90::diffuco_main consumes/modifies humrel at lines 306-410.",
            "src_sechiba/enerbil.f90::enerbil_main consumes humrel at lines 390-445.",
        ),
        ("STOMATE moisture availability", "next-step diffuco/enerbil vegetation resistance"),
        "covered",
    ),
    BridgeField(
        "vegstress",
        "(npts,nvm)",
        "hydrol_main/hydrol_soil",
        ("slowproc_main",),
        (
            "src_sechiba/hydrol.f90::hydrol_main lines 1041-1043 declares vegstress output.",
            "src_sechiba/sechiba.f90::sechiba_main passes vegstress to slowproc_main at lines 1184-1188.",
            "src_sechiba/slowproc.f90::slowproc_main does not pass vegstress into stomate_main at lines 973-985.",
        ),
        ("slowproc water-stress handoff before STOMATE entry",),
        "covered",
    ),
    BridgeField(
        "shumdiag",
        "(npts,nslm)",
        "hydrol_main/hydrol_soil",
        ("slowproc_main", "stomate_main daily soilhum"),
        (
            "src_sechiba/hydrol.f90::hydrol_main declares shumdiag output at lines 1041-1047.",
            "src_sechiba/sechiba.f90::sechiba_main passes shumdiag to slowproc_main at lines 1184-1188.",
            "src_stomate/stomate.f90::stomate_main accumulates shumdiag into soilhum_daily at lines 3201-3208.",
        ),
        ("STOMATE soil humidity daily/monthly state",),
        "partial",
    ),
    BridgeField(
        "litterhumdiag",
        "(npts)",
        "hydrol_main/hydrol_soil",
        ("slowproc_main", "stomate_main daily litter humidity"),
        (
            "src_sechiba/hydrol.f90::hydrol_main declares litterhumdiag output at lines 1047-1049.",
            "src_stomate/stomate.f90::stomate_main accumulates litterhumdiag at lines 3201-3203.",
        ),
        ("STOMATE litter humidity daily state",),
        "covered",
    ),
    BridgeField(
        "soil_mc",
        "(npts,nslm,nstm)",
        "hydrol_main/hydrol_soil",
        ("slowproc_main", "stomate_soilcarbon::soilcarbon_leak"),
        (
            "src_sechiba/hydrol.f90::hydrol_main declares soil_mc output at lines 1030-1038.",
            "src_sechiba/hydrol.f90::hydrol_main passes soil_mc through hydrol_soil at lines 1278-1296.",
            "src_stomate/stomate_soilcarbon.f90::soilcarbon_leak consumes soil_mc at lines 651-705.",
            "src_stomate/stomate_soilcarbon.f90 uses soil_mc in DOC water normalization at lines 1844-1888.",
        ),
        ("MICT-leak soil carbon/DOC water content",),
        "covered",
    ),
    BridgeField(
        "wat_flux",
        "(npts,nslm,nstm)",
        "hydrol_main/hydrol_soil",
        ("slowproc_main", "stomate_soilcarbon::soilcarbon_leak"),
        (
            "src_sechiba/hydrol.f90::hydrol_main declares wat_flux output at lines 1030-1038.",
            "src_stomate/stomate_soilcarbon.f90::soilcarbon_leak consumes wat_flux at lines 686-689.",
            "src_stomate/stomate_soilcarbon.f90 computes DOC_FLUX from wat_flux at lines 1869-1888.",
        ),
        ("MICT-leak vertical DOC transport",),
        "covered",
    ),
    BridgeField(
        "runoff_per_soil",
        "(npts,nstm)",
        "hydrol_main/hydrol_soil",
        ("slowproc_main", "stomate_soilcarbon::soilcarbon_leak"),
        (
            "src_sechiba/hydrol.f90::hydrol_main declares runoff_per_soil output at lines 1036-1038.",
            "src_stomate/stomate_soilcarbon.f90::soilcarbon_leak consumes runoff_per_soil at lines 686-689.",
            "src_stomate/stomate_soilcarbon.f90 computes DOC_RUN from runoff_per_soil at lines 2146-2182.",
        ),
        ("MICT-leak runoff DOC export",),
        "covered",
    ),
    BridgeField(
        "drainage_per_soil",
        "(npts,nstm)",
        "hydrol_main/hydrol_soil",
        ("slowproc_main", "stomate_soilcarbon::soilcarbon_leak"),
        (
            "src_sechiba/hydrol.f90::hydrol_main declares drainage_per_soil output at lines 1036-1038.",
            "src_stomate/stomate_soilcarbon.f90::soilcarbon_leak consumes drainage_per_soil at lines 686-689.",
            "src_stomate/stomate_soilcarbon.f90 computes DOC_DRAIN from drainage_per_soil at lines 2205-2210.",
        ),
        ("MICT-leak drainage DOC export",),
        "covered",
    ),
    BridgeField(
        "runoff2peat",
        "(npts,nstm)",
        "hydrol_main/hydrol_soil",
        ("slowproc_main", "stomate_soilcarbon::soilcarbon_leak"),
        (
            "src_sechiba/hydrol.f90::hydrol_main declares runoff2peat output at lines 1036-1038.",
            "src_sechiba/sechiba.f90::sechiba_main passes runoff2peat to slowproc_main at lines 1184-1216.",
            "src_stomate/stomate_soilcarbon.f90::soilcarbon_leak consumes runoff2peat at lines 641-688.",
        ),
        ("MICT-leak mineral-to-peat runoff DOC transfer",),
        "covered",
    ),
    BridgeField(
        "precip2canopy",
        "(npts,nvm)",
        "hydrol_canop",
        ("slowproc_main", "stomate_soilcarbon::soilcarbon_leak"),
        (
            "src_sechiba/hydrol.f90::hydrol_main declares precip2canopy output at lines 1030-1035.",
            "src_sechiba/hydrol.f90::hydrol_main calls hydrol_canop at lines 1232-1235.",
            "src_stomate/stomate_soilcarbon.f90::soilcarbon_leak consumes precip2canopy at lines 699-704.",
        ),
        ("MICT-leak canopy wet deposition",),
        "covered",
    ),
    BridgeField(
        "precip2ground",
        "(npts,nvm)",
        "hydrol_canop",
        ("slowproc_main", "stomate_soilcarbon::soilcarbon_leak"),
        (
            "src_sechiba/hydrol.f90::hydrol_main declares precip2ground output at lines 1030-1035.",
            "src_sechiba/hydrol.f90::hydrol_main calls hydrol_canop at lines 1232-1235.",
            "src_stomate/stomate_soilcarbon.f90::soilcarbon_leak consumes precip2ground at lines 699-704.",
        ),
        ("MICT-leak ground wet deposition",),
        "covered",
    ),
    BridgeField(
        "canopy2ground",
        "(npts,nvm)",
        "hydrol_canop",
        ("slowproc_main", "stomate_soilcarbon::soilcarbon_leak"),
        (
            "src_sechiba/hydrol.f90::hydrol_main declares canopy2ground output at lines 1030-1035.",
            "src_sechiba/hydrol.f90::hydrol_main calls hydrol_canop at lines 1232-1235.",
            "src_stomate/stomate_soilcarbon.f90 computes DOC_canopy2ground from canopy2ground at lines 1497-1512.",
        ),
        ("MICT-leak canopy DOC transfer",),
        "covered",
    ),
    BridgeField(
        "wtp",
        "(npts)",
        "hydrol_main",
        ("slowproc_main", "stomate_main peatland/tide inputs"),
        (
            "src_sechiba/sechiba.f90::sechiba_main passes wtp to slowproc_main at lines 1184-1216.",
            "src_sechiba/slowproc.f90::slowproc_main passes wtp to stomate_main at lines 973-985.",
            "src_stomate/stomate.f90::stomate_main declares wtp input near lines 2409-2468.",
        ),
        ("STOMATE peatland/tide water-table state",),
        "covered",
    ),
    BridgeField(
        "fwet_new",
        "(npts)",
        "hydrol_main",
        ("slowproc_main", "stomate_main peatland/tide inputs"),
        (
            "src_sechiba/sechiba.f90::sechiba_main passes fwet_new to slowproc_main at lines 1184-1216.",
            "src_sechiba/slowproc.f90::slowproc_main passes fwet_new to stomate_main at lines 973-985.",
            "src_stomate/stomate.f90::stomate_main declares fwet_new input near lines 2409-2468.",
        ),
        ("STOMATE wetland fraction state",),
        "covered",
    ),
    BridgeField(
        "mc_peat_above",
        "(npts)",
        "hydrol_main",
        ("slowproc_main", "stomate_main peatland inputs"),
        (
            "src_sechiba/sechiba.f90::sechiba_main passes mc_peat_above to slowproc_main at lines 1184-1216.",
            "src_sechiba/slowproc.f90::slowproc_main passes mc_peat_above to stomate_main at lines 973-985.",
            "src_stomate/stomate.f90::stomate_main declares mc_peat_above input near lines 2409-2468.",
        ),
        ("STOMATE peat above-water moisture state",),
        "covered",
    ),
    BridgeField(
        "liqwt_ratio",
        "(npts)",
        "hydrol_main",
        ("slowproc_main", "stomate_main peatland/tide inputs"),
        (
            "src_sechiba/sechiba.f90::sechiba_main passes liqwt_ratio to slowproc_main at lines 1184-1216.",
            "src_sechiba/slowproc.f90::slowproc_main passes liqwt_ratio to stomate_main at lines 973-985.",
            "src_stomate/stomate.f90::stomate_main declares liqwt_ratio input near lines 2409-2468.",
        ),
        ("STOMATE liquid water table ratio state",),
        "covered",
    ),
    BridgeField(
        "mc_man_above",
        "(npts)",
        "hydrol_main",
        ("slowproc_main", "stomate_main mangrove/tide inputs"),
        (
            "src_sechiba/sechiba.f90::sechiba_main passes mc_man_above to slowproc_main at lines 1184-1216.",
            "src_sechiba/slowproc.f90::slowproc_main passes mc_man_above to stomate_main at lines 973-985.",
            "src_stomate/stomate.f90::stomate_main declares mc_man_above input near lines 2409-2468.",
        ),
        ("STOMATE mangrove above-water moisture state",),
        "covered",
    ),
)


HYDROL_THERMOSOIL_FIELDS = (
    BridgeField(
        "shumdiag_perma",
        "(npts,nslm)",
        "hydrol_main/hydrol_soil",
        ("thermosoil_humlev", "thermosoil_getdiff", "slowproc_main via updated stempdiag"),
        (
            "src_sechiba/hydrol.f90::hydrol_main declares shumdiag_perma output at lines 1041-1047.",
            "src_sechiba/sechiba.f90::sechiba_main passes shumdiag_perma to thermosoil_main at lines 1109-1118.",
            "src_sechiba/thermosoil.f90::thermosoil_humlev consumes shumdiag_perma at lines 2258-2399.",
        ),
        ("THERMOSOIL humidity interpolation and permafrost humidity profile",),
        "source-covered",
    ),
    BridgeField(
        "mc_layh",
        "(npts,nslm)",
        "hydrol_main/hydrol_soil",
        ("thermosoil_humlev", "thermosoil_getdiff"),
        (
            "src_sechiba/hydrol.f90::hydrol_main declares mc_layh output at lines 1085-1089.",
            "src_sechiba/sechiba.f90::sechiba_main calls thermosoil_main with mc_layh at lines 1109-1118.",
            "src_sechiba/thermosoil.f90::thermosoil_humlev consumes mc_layh at lines 2258-2399.",
        ),
        ("THERMOSOIL total volumetric moisture at hydrology nodes",),
        "source-covered",
    ),
    BridgeField(
        "mcl_layh",
        "(npts,nslm)",
        "hydrol_main/hydrol_soil",
        ("thermosoil_humlev", "thermosoil_getdiff"),
        (
            "src_sechiba/hydrol.f90::hydrol_main declares mcl_layh output at lines 1085-1089.",
            "src_sechiba/sechiba.f90::sechiba_main calls thermosoil_main with mcl_layh at lines 1109-1118.",
            "src_sechiba/thermosoil.f90::thermosoil_humlev consumes mcl_layh at lines 2258-2399.",
        ),
        ("THERMOSOIL liquid volumetric moisture at hydrology nodes",),
        "source-covered",
    ),
    BridgeField(
        "tmc_layh",
        "(npts,nslm)",
        "hydrol_main/hydrol_soil",
        ("thermosoil_humlev", "thermosoil_getdiff"),
        (
            "src_sechiba/hydrol.f90::hydrol_main declares soilmoist_out at lines 1085-1089.",
            "src_sechiba/sechiba.f90::sechiba_main passes soilmoist to thermosoil_main at lines 1109-1118.",
            "src_sechiba/thermosoil.f90::thermosoil_main passes it as tmc_layh to thermosoil_humlev at lines 893-894.",
            "src_sechiba/thermosoil.f90::thermosoil_humlev consumes tmc_layh at lines 2258-2399.",
        ),
        ("THERMOSOIL total water mass at hydrology nodes",),
        "source-covered",
        ("soilmoist",),
    ),
    BridgeField(
        "mc_layh_pft",
        "(npts,nslm,nvm)",
        "sechiba_main maps HYDROL preferred soil tiles",
        ("thermosoil_humlev", "thermosoil_getdiff"),
        (
            "src_sechiba/sechiba.f90::sechiba_main maps HYDROL moisture to PFT arrays at lines 1093-1102.",
            "src_sechiba/sechiba.f90::sechiba_main calls thermosoil_main with moisture arrays at lines 1109-1118.",
            "src_sechiba/thermosoil.f90::thermosoil_humlev consumes mc_layh_pft at lines 2258-2399.",
        ),
        ("PFT-resolved THERMOSOIL total volumetric moisture",),
        "source-covered",
    ),
    BridgeField(
        "mcl_layh_pft",
        "(npts,nslm,nvm)",
        "sechiba_main maps HYDROL preferred soil tiles",
        ("thermosoil_humlev", "thermosoil_getdiff"),
        (
            "src_sechiba/sechiba.f90::sechiba_main maps HYDROL moisture to PFT arrays at lines 1093-1102.",
            "src_sechiba/sechiba.f90::sechiba_main calls thermosoil_main with moisture arrays at lines 1109-1118.",
            "src_sechiba/thermosoil.f90::thermosoil_humlev consumes mcl_layh_pft at lines 2258-2399.",
        ),
        ("PFT-resolved THERMOSOIL liquid volumetric moisture",),
        "source-covered",
    ),
    BridgeField(
        "tmc_layh_pft",
        "(npts,nslm,nvm)",
        "sechiba_main maps HYDROL preferred soil tiles",
        ("thermosoil_humlev", "thermosoil_getdiff"),
        (
            "src_sechiba/sechiba.f90::sechiba_main maps soilmoist_pft from HYDROL soilmoist at lines 1093-1102.",
            "src_sechiba/sechiba.f90::sechiba_main passes soilmoist_pft to thermosoil_main at lines 1109-1118.",
            "src_sechiba/thermosoil.f90::thermosoil_main passes it as tmc_layh_pft to thermosoil_humlev at lines 893-894.",
            "src_sechiba/thermosoil.f90::thermosoil_humlev consumes tmc_layh_pft at lines 2258-2399.",
        ),
        ("PFT-resolved THERMOSOIL total water mass",),
        "source-covered",
        ("soilmoist_pft",),
    ),
)


ENERBIL_CONDVEG_THERMOSOIL_FIELDS = (
    BridgeField(
        "temp_sol_new",
        "(npts)",
        "enerbil_main",
        ("hydrol_main explicit snow", "thermosoil_profile", "thermosoil_coef"),
        (
            "src_sechiba/enerbil.f90::enerbil_main declares temp_sol_new output at lines 456-459.",
            "src_sechiba/sechiba.f90::sechiba_main passes temp_sol_new to hydrol_main at lines 1049-1055.",
            "src_sechiba/sechiba.f90::sechiba_main passes temp_sol_new to thermosoil_main at lines 1109-1118.",
            "src_sechiba/thermosoil.f90::thermosoil_profile consumes temp_sol_new at lines 1761-1821.",
            "src_sechiba/thermosoil.f90::thermosoil_coef consumes temp_sol_new at lines 1518-1721.",
        ),
        ("THERMOSOIL surface-temperature forcing",),
        "covered",
    ),
    BridgeField(
        "temp_sol_new_pft",
        "(npts,nvm)",
        "enerbil_main",
        ("thermosoil_profile", "thermosoil_coef"),
        (
            "src_sechiba/enerbil.f90::enerbil_main declares temp_sol_new_pft output at lines 456-459.",
            "src_sechiba/sechiba.f90::sechiba_main passes temp_sol_new_pft to thermosoil_main at lines 1109-1118.",
            "src_sechiba/thermosoil.f90::thermosoil_profile consumes temp_sol_new_pft at lines 1761-1821.",
            "src_sechiba/thermosoil.f90::thermosoil_coef consumes temp_sol_new_pft at lines 1518-1721.",
        ),
        ("PFT-resolved THERMOSOIL surface-temperature forcing",),
        "covered",
    ),
    BridgeField(
        "snowdz/snowrho/snowtemp",
        "(npts,nsnow)",
        "hydrol_main/explicitsnow_main",
        ("thermosoil_profile", "thermosoil_getdiff", "thermosoil_coef", "slowproc_main"),
        (
            "src_sechiba/hydrol.f90::hydrol_main snow state INTENT(inout) at lines 1069-1083.",
            "src_sechiba/sechiba.f90::sechiba_main passes snowdz, snowrho, snowtemp to thermosoil_main at lines 1109-1118.",
            "src_sechiba/thermosoil.f90::thermosoil_profile consumes snowtemp at lines 1761-1814.",
            "src_sechiba/thermosoil.f90::thermosoil_getdiff consumes snowrho/snowtemp at lines 2566-2863.",
            "src_sechiba/thermosoil.f90::thermosoil_coef consumes snowdz/snowrho/snowtemp at lines 1633-1721.",
        ),
        ("active explicit-snow THERMOSOIL thermal profile and coefficients",),
        "source-covered",
        ("snowdz", "snowrho", "snowtemp"),
    ),
    BridgeField(
        "frac_snow_veg/frac_snow_nobio/totfrac_nobio",
        "(npts[,nnobio])",
        "condveg_main and land-state fractions",
        ("thermosoil_profile", "thermosoil_coef"),
        (
            "src_sechiba/sechiba.f90::sechiba_main calls condveg_main at lines 1085-1091.",
            "src_sechiba/sechiba.f90::sechiba_main passes frac_snow_veg, frac_snow_nobio, and totfrac_nobio to thermosoil_main at lines 1109-1118.",
            "src_sechiba/thermosoil.f90::thermosoil_profile uses these snow fractions at lines 1807-1814.",
            "src_sechiba/thermosoil.f90::thermosoil_coef blends snow and no-snow coefficients at lines 1715-1720.",
        ),
        ("explicit-snow surface-temperature and coefficient blending",),
        "source-covered",
        ("frac_snow_veg", "frac_snow_nobio", "totfrac_nobio"),
    ),
    BridgeField(
        "ptn/cgrnd/dgrnd/cgrnd_snow/dgrnd_snow/lambda_snow",
        "(npts,ngrnd,nvm) plus snow coefficients",
        "previous thermosoil_main/restart",
        ("thermosoil_profile", "thermosoil_coef", "restart"),
        (
            "src_sechiba/thermosoil.f90::thermosoil_profile uses prior cgrnd/dgrnd and ptn at lines 1761-1851.",
            "src_sechiba/thermosoil.f90::thermosoil_coef recomputes cgrnd/dgrnd and snow coefficients at lines 1518-1721.",
            "src_sechiba/thermosoil.f90::thermosoil_finalize writes ptn, cgrnd, dgrnd, cgrnd_snow, dgrnd_snow, and lambda_snow to restart at lines 1051-1134.",
        ),
        ("THERMOSOIL recurrence state, not a tunable bridge value",),
        "upstream-required",
        ("ptn", "cgrnd", "dgrnd", "cgrnd_snow", "dgrnd_snow", "lambda_snow"),
    ),
    BridgeField(
        "soilc_total/refSOC/zx1",
        "(npts,ngrnd,nvm) or (npts,ngrnd)",
        "slowproc_main/restart organic carbon state",
        ("thermosoil_getdiff",),
        (
            "src_sechiba/sechiba.f90::sechiba_main passes soilc_total to thermosoil_main at lines 1109-1118.",
            "src_sechiba/thermosoil.f90::thermosoil_getdiff builds organic/mineral fractions from soilc_total or refSOC at lines 2673-2689.",
        ),
        ("soil organic fraction for thermal capacity/conductivity",),
        "upstream-required",
        ("soilc_total", "refSOC", "zx1"),
    ),
)


THERMOSOIL_SLOWPROC_ENERBIL_FIELDS = (
    BridgeField(
        "stempdiag",
        "(npts,nslm)",
        "thermosoil_profile/thermosoil_diaglev",
        ("slowproc_main", "stomate_main daily soil temperature", "slowproc_lai"),
        (
            "src_sechiba/thermosoil.f90::thermosoil_profile calls thermosoil_diaglev at lines 1847-1851.",
            "src_sechiba/thermosoil.f90::thermosoil_diaglev computes stempdiag at lines 3484-3513.",
            "src_sechiba/sechiba.f90::sechiba_main passes stempdiag to slowproc_main at lines 1184-1188.",
            "src_sechiba/slowproc.f90::slowproc_main consumes stempdiag at lines 403, 973-985, and 1082.",
        ),
        ("STOMATE soil-temperature daily state and LAI temperature response",),
        "source-covered",
    ),
    BridgeField(
        "soilcap/soilcap_pft",
        "(npts[,nvm])",
        "thermosoil_coef",
        ("next enerbil_main", "restart"),
        (
            "src_sechiba/thermosoil.f90::thermosoil_coef computes soilcap and soilcap_pft at lines 1518-1725.",
            "src_sechiba/thermosoil.f90::thermosoil_finalize writes soilcap and soilcap_pft to restart at lines 1118-1126.",
            "src_sechiba/enerbil.f90::enerbil_main consumes soilcap and soilcap_pft at lines 390-471.",
        ),
        ("next-step implicit surface energy balance",),
        "source-covered",
        ("soilcap", "soilcap_pft"),
    ),
    BridgeField(
        "soilflx/soilflx_pft",
        "(npts[,nvm])",
        "thermosoil_coef",
        ("next enerbil_main", "restart", "history Qg"),
        (
            "src_sechiba/thermosoil.f90::thermosoil_coef computes soilflx and soilflx_pft at lines 1518-1725.",
            "src_sechiba/thermosoil.f90::thermosoil_main writes Qg from soilflx at lines 946-967.",
            "src_sechiba/thermosoil.f90::thermosoil_finalize writes soilflx and soilflx_pft to restart at lines 1127-1130.",
            "src_sechiba/enerbil.f90::enerbil_main consumes soilflx and soilflx_pft at lines 390-471.",
        ),
        ("next-step ground heat-flux term in ENERBIL",),
        "source-covered",
        ("soilflx", "soilflx_pft"),
    ),
    BridgeField(
        "gtemp/ptnlev1",
        "(npts)",
        "thermosoil_main final state",
        ("hydrol_main explicit snow", "restart", "diagnostics"),
        (
            "src_sechiba/thermosoil.f90::thermosoil_main computes gtemp and ptnlev1 at lines 1011-1020.",
            "src_sechiba/thermosoil.f90::thermosoil_finalize writes gtemp to restart at lines 1114-1117.",
            "src_sechiba/sechiba.f90::sechiba_main passes gtemp into hydrol_main at lines 1049-1072.",
        ),
        ("ground temperature state shared with snow/hydrology",),
        "source-covered",
        ("gtemp", "ptnlev1"),
    ),
    BridgeField(
        "ptn_pftmean/pkappa_pftmean/deephum_prof/deeptemp_prof",
        "(npts,ngrnd)",
        "thermosoil_main final state",
        ("diagnostics", "permafrost/STOMATE deep profiles", "restart"),
        (
            "src_sechiba/thermosoil.f90::thermosoil_main recomputes ptn_pftmean and pkappa_pftmean at lines 1011-1017.",
            "src_sechiba/thermosoil.f90::thermosoil_main assigns deephum_prof and deeptemp_prof at lines 1021-1025.",
        ),
        ("deep soil temperature/moisture profile handoff",),
        "source-covered",
        ("ptn_pftmean", "pkappa_pftmean", "deephum_prof", "deeptemp_prof"),
    ),
)


HYDROL_NEXT_SECHIBA_FIELDS = (
    BridgeField(
        "qsintveg",
        "(npts,nvm)",
        "hydrol_canop",
        ("next diffuco_main", "next hydrol_main", "restart"),
        (
            "src_sechiba/hydrol.f90::hydrol_main treats qsintveg as INTENT(inout) at lines 1057-1060.",
            "src_sechiba/hydrol.f90::hydrol_main calls hydrol_canop at lines 1232-1235.",
            "src_sechiba/diffuco.f90::diffuco_main consumes qsintveg at lines 306-368 and 657-671.",
        ),
        ("next-step interception and transpiration resistance",),
        "missing",
    ),
    BridgeField(
        "snow/snow_nobio/snowdz/snowrho",
        "(npts[,nnobio|nsnow])",
        "hydrol_snow or explicitsnow_main",
        ("condveg_main", "thermosoil_main", "slowproc_main", "next diffuco_main", "next enerbil_main"),
        (
            "src_sechiba/hydrol.f90::hydrol_main snow state INTENT(inout) at lines 1069-1083.",
            "src_sechiba/sechiba.f90::sechiba_main passes snow to condveg_main at lines 1085-1091.",
            "src_sechiba/sechiba.f90::sechiba_main passes snow/snowdz/snowrho to slowproc_main at lines 1197-1201.",
            "src_sechiba/diffuco.f90::diffuco_main consumes snow at lines 345-359 and 647-655.",
            "src_sechiba/enerbil.f90::enerbil_main consumes snowdz at lines 444-445 and 543-545.",
        ),
        ("surface coefficients", "STOMATE snow daily state", "next-step diffusion/energy"),
        "missing",
    ),
    BridgeField(
        "evap_bare_lim/drysoil_frac/k_litt",
        "(npts)",
        "hydrol_main/hydrol_soil",
        ("next diffuco_main", "condveg_main"),
        (
            "src_sechiba/hydrol.f90::hydrol_main outputs drysoil_frac, k_litt and updates evap_bare_lim at lines 1041-1063.",
            "src_sechiba/sechiba.f90::sechiba_main passes drysoil_frac to condveg_main at lines 1085-1091.",
            "src_sechiba/diffuco.f90::diffuco_main consumes evap_bare_lim at lines 349-356 and 698-710.",
        ),
        ("next-step bare soil evaporation resistance", "surface dry-soil diagnostics"),
        "missing",
    ),
)


DIFFUCO_STOMATE_FIELDS = (
    BridgeField(
        "gpp",
        "(npts,nvm)",
        "diffuco_main/diffuco_trans_co2",
        ("slowproc_main", "stomate_main daily GPP", "modelout GPP source"),
        (
            "src_sechiba/diffuco.f90::diffuco_main declares gpp output at lines 389-405.",
            "src_sechiba/diffuco.f90::diffuco_main calls diffuco_trans_co2 at lines 665-671.",
            "src_sechiba/diffuco.f90::diffuco_trans_co2 computes gpp at lines 2890-2897.",
            "src_stomate/stomate.f90::stomate_main accumulates gpp_d into gpp_daily at lines 3198-3208.",
            "src_stomate/stomate_lpj.f90 writes GPP at lines 1678 and 2203.",
        ),
        ("STOMATE NPP and modelout GPP",),
        "covered",
    ),
    BridgeField(
        "gsmean/rveget/rstruct/cimean",
        "(npts,nvm)",
        "diffuco_main/diffuco_trans_co2",
        ("diagnostics", "next energy resistance context"),
        (
            "src_sechiba/diffuco.f90::diffuco_main declares these outputs at lines 391-405.",
            "src_sechiba/diffuco.f90::diffuco_trans_co2 returns them at lines 2018-2088.",
            "outputs/server_1961_bridge_trace_20260624/traces/orchjax_sechiba_bridge_diffuco_trace.txt:"
            "after_diffuco_main covers these PFT14 boundary diagnostics.",
        ),
        ("photosynthesis/transpiration diagnostics",),
        "covered",
    ),
)


ENERBIL_HYDROL_SLOWPROC_FIELDS = (
    BridgeField(
        "transpir/transpot/vevapnu/vevapwet/vevapsno/vevapflo",
        "(npts[,nvm])",
        "enerbil_main/enerbil_evapveg",
        ("hydrol_main", "slowproc_main via hydrol response"),
        (
            "src_sechiba/enerbil.f90::enerbil_main declares evaporation/transpiration outputs at lines 447-455.",
            "src_sechiba/enerbil.f90::enerbil_main calls enerbil_evapveg at lines 541-545.",
            "src_sechiba/sechiba.f90::sechiba_main passes these fields into hydrol_main at lines 1049-1055.",
            "outputs/server_1961_bridge_trace_20260624/traces/orchjax_sechiba_bridge_enerbil_trace.txt:"
            "after_enerbil_main covers these PFT14 boundary fields.",
        ),
        ("HYDROL water budget and stress",),
        "covered",
    ),
    BridgeField(
        "temp_sol/temp_sol_new/temp_sol_pft/qsurf/t2mdiag/evapot_corr",
        "(npts[,nvm])",
        "enerbil_main",
        ("hydrol_main", "thermosoil_main", "slowproc_main", "next diffuco_main"),
        (
            "src_sechiba/enerbil.f90::enerbil_main declares modified/output state at lines 456-471.",
            "src_sechiba/enerbil.f90::enerbil_main computes surface state at lines 507-545.",
            "src_sechiba/sechiba.f90::sechiba_main passes temp_sol_new to hydrol_main at lines 1049-1055.",
            "src_sechiba/sechiba.f90::sechiba_main passes temp_sol and evapot_corr to slowproc_main at lines 1184-1198.",
            "outputs/server_1961_bridge_trace_20260624/traces/orchjax_sechiba_bridge_enerbil_trace.txt:"
            "after_enerbil_main covers these PFT14 boundary fields.",
        ),
        ("HYDROL snow/soil water update", "STOMATE temperature/PET daily state"),
        "covered",
    ),
)


BRIDGE_GROUPS = (
    BridgeGroup(
        "hydrol_to_stomate",
        "HYDROL outputs that enter slowproc_main/stomate_main or MICT-leak soilcarbon.",
        HYDROL_STOMATE_FIELDS,
    ),
    BridgeGroup(
        "hydrol_to_thermosoil",
        "HYDROL moisture fields required before thermosoil updates thermal and STOMATE soil-temperature inputs.",
        HYDROL_THERMOSOIL_FIELDS,
    ),
    BridgeGroup(
        "enerbil_condveg_to_thermosoil",
        "Surface temperature, snow-fraction, snowpack, restart, and organic-carbon inputs required by THERMOSOIL.",
        ENERBIL_CONDVEG_THERMOSOIL_FIELDS,
    ),
    BridgeGroup(
        "thermosoil_to_slowproc_enerbil",
        "THERMOSOIL outputs consumed by slowproc/STOMATE, restart, HYDROL snow context, and next ENERBIL.",
        THERMOSOIL_SLOWPROC_ENERBIL_FIELDS,
    ),
    BridgeGroup(
        "hydrol_next_sechiba",
        "HYDROL state that feeds the next SECHIBA timestep and surface coefficient updates.",
        HYDROL_NEXT_SECHIBA_FIELDS,
    ),
    BridgeGroup(
        "diffuco_to_stomate",
        "DIFFUCO photosynthesis outputs needed before STOMATE/modelout can close.",
        DIFFUCO_STOMATE_FIELDS,
    ),
    BridgeGroup(
        "enerbil_to_hydrol_slowproc",
        "ENERBIL fields that must exist before HYDROL and STOMATE daily forcing can be exact.",
        ENERBIL_HYDROL_SLOWPROC_FIELDS,
    ),
)


HYDROL_BRIDGE_TRACE_PROVENANCE = (
    "outputs/server_1961_bridge_trace_20260624/traces/orchjax_sechiba_bridge_hydrol_trace.txt:"
    "after_hydrol_main_pft/after_hydrol_main_layer/after_hydrol_main_tile_layer/after_hydrol_main_tile",
    "outputs/server_1961_bridge_trace_20260624/traces/orchjax_sechiba_bridge_slowproc_trace.txt:"
    "before_slowproc_main/before_stomate_main",
    "docs/source_audits/server_1961_bridge_trace_patch_plan.md#hydrol-and-slowproc-bridge-records",
    "src_sechiba/sechiba.f90::sechiba_main lines 1049-1072 calls hydrol_main, "
    "lines 1093-1102 maps PFT moisture, and lines 1184-1216 calls slowproc_main.",
    "src_sechiba/slowproc.f90::slowproc_main lines 973-985 calls stomate_main.",
)

HYDROL_TO_SLOWPROC_TRACE_MAPPINGS = (
    BridgeTraceMapping(
        "humrel",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_pft",
        "humrel",
        "sechiba_bridge_slowproc",
        "before_slowproc_main",
        "humrel",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "vegstress",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_pft",
        "vegstress",
        "sechiba_bridge_slowproc",
        "before_slowproc_main",
        "vegstress",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "litterhumdiag",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_pft",
        "litterhumdiag",
        "sechiba_bridge_slowproc",
        "before_slowproc_main",
        "litterhumdiag",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "wtp",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_pft",
        "wtp",
        "sechiba_bridge_slowproc",
        "before_slowproc_main",
        "wtp",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "fwet_new",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_pft",
        "fwet_new",
        "sechiba_bridge_slowproc",
        "before_slowproc_main",
        "fwet_new",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "mc_peat_above",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_pft",
        "mc_peat_above",
        "sechiba_bridge_slowproc",
        "before_slowproc_main",
        "mc_peat_above",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "liqwt_ratio",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_pft",
        "liqwt_ratio",
        "sechiba_bridge_slowproc",
        "before_slowproc_main",
        "liqwt_ratio",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "mc_man_above",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_pft",
        "mc_man_above",
        "sechiba_bridge_slowproc",
        "before_slowproc_main",
        "mc_man_above",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
)

HYDROL_TO_STOMATE_TRACE_MAPPINGS = (
    BridgeTraceMapping(
        "humrel",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_pft",
        "humrel",
        "sechiba_bridge_slowproc",
        "before_stomate_main",
        "humrel",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "litterhumdiag",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_pft",
        "litterhumdiag",
        "sechiba_bridge_slowproc",
        "before_stomate_main",
        "litterhumdiag",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "wtp",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_pft",
        "wtp",
        "sechiba_bridge_slowproc",
        "before_stomate_main",
        "wtp",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "fwet_new",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_pft",
        "fwet_new",
        "sechiba_bridge_slowproc",
        "before_stomate_main",
        "fwet_new",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "mc_peat_above",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_pft",
        "mc_peat_above",
        "sechiba_bridge_slowproc",
        "before_stomate_main",
        "mc_peat_above",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "liqwt_ratio",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_pft",
        "liqwt_ratio",
        "sechiba_bridge_slowproc",
        "before_stomate_main",
        "liqwt_ratio",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "mc_man_above",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_pft",
        "mc_man_above",
        "sechiba_bridge_slowproc",
        "before_stomate_main",
        "mc_man_above",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "precip2canopy",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_pft",
        "precip2canopy",
        "sechiba_bridge_slowproc",
        "before_stomate_main",
        "precip2canopy",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "precip2ground",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_pft",
        "precip2ground",
        "sechiba_bridge_slowproc",
        "before_stomate_main",
        "precip2ground",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "canopy2ground",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_pft",
        "canopy2ground",
        "sechiba_bridge_slowproc",
        "before_stomate_main",
        "canopy2ground",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "soil_mc_top_tile",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_tile_layer",
        "soil_mc",
        "sechiba_bridge_slowproc",
        "before_stomate_main",
        "soil_mc_top_tile",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "wat_flux_top_tile",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_tile_layer",
        "wat_flux",
        "sechiba_bridge_slowproc",
        "before_stomate_main",
        "wat_flux_top_tile",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "drainage_per_soil_tile",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_tile",
        "drainage_per_soil",
        "sechiba_bridge_slowproc",
        "before_stomate_main",
        "drainage_per_soil_tile",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "runoff_per_soil_tile",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_tile",
        "runoff_per_soil",
        "sechiba_bridge_slowproc",
        "before_stomate_main",
        "runoff_per_soil_tile",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
    BridgeTraceMapping(
        "runoff2peat_tile",
        "sechiba_bridge_hydrol",
        "after_hydrol_main_tile",
        "runoff2peat",
        "sechiba_bridge_slowproc",
        "before_stomate_main",
        "runoff2peat_tile",
        HYDROL_BRIDGE_TRACE_PROVENANCE,
    ),
)


def bridge_group_names() -> tuple[str, ...]:
    """Return available SECHIBA bridge group names."""

    return tuple(group.name for group in BRIDGE_GROUPS)


def bridge_group(name: str) -> BridgeGroup:
    """Return one bridge group by name."""

    for group in BRIDGE_GROUPS:
        if group.name == name:
            return group
    raise KeyError(f"unknown SECHIBA bridge group: {name}")


def required_bridge_fields(groups: Iterable[str] | None = None) -> tuple[str, ...]:
    """Return required field names for selected bridge groups."""

    selected = tuple(groups) if groups is not None else bridge_group_names()
    names: list[str] = []
    seen: set[str] = set()
    for group_name in selected:
        for field in bridge_group(group_name).fields:
            if field.name not in seen:
                seen.add(field.name)
                names.append(field.name)
    return tuple(names)


def required_trace_fields(groups: Iterable[str] | None = None) -> tuple[str, ...]:
    """Return bridge fields still blocked by missing or partial trace coverage."""

    selected = tuple(groups) if groups is not None else bridge_group_names()
    names: list[str] = []
    seen: set[str] = set()
    for group_name in selected:
        for field in bridge_group(group_name).fields:
            if field.requires_trace and field.name not in seen:
                seen.add(field.name)
                names.append(field.name)
    return tuple(names)


def hydrol_bridge_trace_mappings(*, target: str = "stomate") -> tuple[BridgeTraceMapping, ...]:
    """Return trace-backed HYDROL continuity mappings.

    Trace provenance: mappings use only the 2026-06-24 `sechiba_bridge_*`
    package records listed in `HYDROL_BRIDGE_TRACE_PROVENANCE`. They do not
    compare against older HYDROL internal traces and do not compute values.
    """

    if target == "slowproc":
        return HYDROL_TO_SLOWPROC_TRACE_MAPPINGS
    if target == "stomate":
        return HYDROL_TO_STOMATE_TRACE_MAPPINGS
    if target == "all":
        return HYDROL_TO_SLOWPROC_TRACE_MAPPINGS + HYDROL_TO_STOMATE_TRACE_MAPPINGS
    raise KeyError(f"unknown HYDROL bridge mapping target: {target}")


def _find_bridge_record(
    name: str,
    *,
    tag: str,
    criteria: Mapping[str, object],
    root: str | Path | None,
    scan_limit: int,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "tag": tag,
        "criteria": dict(criteria),
        "scan_limit": scan_limit,
    }
    if root is not None:
        kwargs["root"] = root
    row = find_server_record(name, **kwargs)
    if row is None:
        raise LookupError(f"missing {name}:{tag} bridge trace record for {dict(criteria)}")
    return dict(row)


def read_hydrol_bridge_trace_slice(
    *,
    kjit: int = 1,
    ji: int = 1,
    jv: int = 14,
    jst: int = 4,
    jsl: int = 1,
    root: str | Path | None = None,
) -> BridgeTraceSlice:
    """Read the PFT/tile/layer records needed to validate HYDROL continuity.

    Defaults select the first 1961 paper-case step, land point 1, PFT14, its
    audited preferred soil tile 4, and top layer 1. Values come from
    `sechiba_bridge_hydrol` and `sechiba_bridge_slowproc` only.
    """

    return BridgeTraceSlice(
        pft=_find_bridge_record(
            "sechiba_bridge_hydrol",
            tag="after_hydrol_main_pft",
            criteria={"kjit": kjit, "ji": ji, "jv": jv},
            root=root,
            scan_limit=128,
        ),
        layer=_find_bridge_record(
            "sechiba_bridge_hydrol",
            tag="after_hydrol_main_layer",
            criteria={"kjit": kjit, "ji": ji, "jv": jv, "jsl": jsl},
            root=root,
            scan_limit=512,
        ),
        tile_layer=_find_bridge_record(
            "sechiba_bridge_hydrol",
            tag="after_hydrol_main_tile_layer",
            criteria={"kjit": kjit, "ji": ji, "jv": jv, "jst": jst, "jsl": jsl},
            root=root,
            scan_limit=512,
        ),
        tile=_find_bridge_record(
            "sechiba_bridge_hydrol",
            tag="after_hydrol_main_tile",
            criteria={"kjit": kjit, "ji": ji, "jv": jv, "jst": jst},
            root=root,
            scan_limit=512,
        ),
        before_slowproc=_find_bridge_record(
            "sechiba_bridge_slowproc",
            tag="before_slowproc_main",
            criteria={"kjit": kjit, "ji": ji, "jv": jv},
            root=root,
            scan_limit=128,
        ),
        before_stomate=_find_bridge_record(
            "sechiba_bridge_slowproc",
            tag="before_stomate_main",
            criteria={"kjit": kjit, "ji": ji, "jv": jv},
            root=root,
            scan_limit=128,
        ),
        provenance=HYDROL_BRIDGE_TRACE_PROVENANCE,
    )


def hydrol_bridge_trace_pairs(
    trace_slice: BridgeTraceSlice,
    *,
    target: str = "stomate",
) -> tuple[BridgeTracePair, ...]:
    """Return source/target values for HYDROL bridge trace equality checks."""

    records = {
        ("sechiba_bridge_hydrol", "after_hydrol_main_pft"): trace_slice.pft,
        ("sechiba_bridge_hydrol", "after_hydrol_main_layer"): trace_slice.layer,
        ("sechiba_bridge_hydrol", "after_hydrol_main_tile_layer"): trace_slice.tile_layer,
        ("sechiba_bridge_hydrol", "after_hydrol_main_tile"): trace_slice.tile,
        ("sechiba_bridge_slowproc", "before_slowproc_main"): trace_slice.before_slowproc,
        ("sechiba_bridge_slowproc", "before_stomate_main"): trace_slice.before_stomate,
    }

    pairs: list[BridgeTracePair] = []
    for mapping in hydrol_bridge_trace_mappings(target=target):
        source = records[(mapping.source_record, mapping.source_tag)]
        target_row = records[(mapping.target_record, mapping.target_tag)]
        pairs.append(
            BridgeTracePair(
                mapping=mapping,
                source_value=source[mapping.source_field],
                target_value=target_row[mapping.target_field],
            )
        )
    return tuple(pairs)


def validate_bridge_payload(
    payload: Mapping[str, object] | Iterable[str],
    *,
    groups: Iterable[str] | None = None,
) -> BridgeValidation:
    """Validate that a payload advertises selected bridge fields.

    Fortran provenance is held on each `BridgeField`. This validator checks
    names only and never derives missing SECHIBA process values.
    """

    if isinstance(payload, Mapping):
        available = frozenset(str(key) for key in payload.keys())
    else:
        available = frozenset(str(item) for item in payload)

    selected = tuple(groups) if groups is not None else bridge_group_names()
    missing_by_group: dict[str, tuple[str, ...]] = {}
    requires_trace_by_group: dict[str, tuple[str, ...]] = {}
    for group_name in selected:
        group = bridge_group(group_name)
        missing_by_group[group.name] = tuple(
            field.name for field in group.fields if not any(name in available for name in field.accepted_names)
        )
        requires_trace_by_group[group.name] = group.requires_trace

    return BridgeValidation(
        available_fields=available,
        missing_by_group=missing_by_group,
        requires_trace_by_group=requires_trace_by_group,
    )
