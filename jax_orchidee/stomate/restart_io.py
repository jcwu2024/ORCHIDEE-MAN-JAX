"""STOMATE restart serialization and audited standalone file construction.

The strict inverse state transforms can update a Fortran template or populate
an independently created paper-protocol file from the audited NetCDF schema.
"""

from __future__ import annotations

import shutil
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from netCDF4 import Dataset

from jax_orchidee.parameters.pft_catalog import (
    PFTRunLayout,
    pft_layout_netcdf_attributes,
)
from jax_orchidee.stomate.reference import (
    STOMATE_DAILY_PFT_AXES,
    STOMATE_ENTRY_PFT_AXES,
    STOMATE_GAS_PFT_AXES,
    STOMATE_SEASON_PFT_AXES,
    StomateDailyAccumulatorState,
    StomateOkPcRestartGasState,
    StomateReadstartRemainderState,
    StomateRestartEntryState,
    StomateRestartSeasonState,
    read_restart_assim_param,
    read_restart_biomass_carbon,
    read_restart_deep_carbon_pool,
    read_restart_grid_field,
    read_restart_interception_storage,
    read_restart_leaf_age_field,
    read_restart_legacy_carbon,
    read_restart_lignin_struc_below,
    read_restart_lignin_struc_legacy,
    read_restart_litter_above,
    read_restart_litter_below,
    read_restart_litter_fuel_field,
    read_restart_litter_legacy,
    read_restart_litter_pft_field,
    read_restart_maint_resp_part,
    read_restart_pft_field,
    read_restart_pft_pool_field,
    read_restart_pft_vertical_field,
    read_restart_product_pool,
    read_restart_scalar,
    read_restart_soil_layer_field,
    remap_stomate_state_by_pft_id,
    stomate_cold_start_daily_accumulator_state,
    stomate_cold_start_entry_state,
    stomate_cold_start_season_state,
)


from jax_orchidee.stomate.reference import (
    _read_restart_carbon_32l_pool,
    _read_restart_doc_pool,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STOMATE_RESTART_SCHEMA = (
    ROOT / "docs" / "source_audits" / "stomate_restart_netcdf_schema.json"
)
# IOIPSL generated two independent dimension names for the same PFT axis,
# depending on whether PFT precedes or follows another scientific dimension.
# Both STOMATE and SECHIBA paper schemas use these exact names.
PAPER_RESTART_PFT_DIMENSIONS = frozenset({"z_a", "l_d"})
STOMATE_REMAINDER_PFT_AXES = {
    **{
        name: 1
        for name in (
            "fireindex", "firelitter", "resp_hetero", "co2_fire",
            "carbon_acro", "carbon_cato", "depth_deepsoil", "wshtotsum",
            "sr_ugb", "nb_ani", "grazed_frac", "import_yield",
            "nb_grazingdays", "MatrixV", "Vector_U", "previous_stock",
            "current_stock",
        )
    },
    "deepC_peat": 2,
    "carbon_save": 2,
    "litter_not_avail": 2,
}


@dataclass(frozen=True)
class StomateRestartWriteReport:
    """Fields handled by one strict template-backed write."""

    output_path: Path
    written_fields: tuple[str, ...]
    validated_derived_fields: tuple[str, ...]
    unsupported_fields: tuple[str, ...]


@dataclass(frozen=True)
class StomateRestartPhysicalState:
    """Physical axes owned by IOIPSL outside ``stomate_io::writerestart``.

    The paper production protocol runs every landpoint independently, hence
    ``nav_lon`` and ``nav_lat`` are one-cell grids and ``index_g`` is exactly
    the Fortran one-based identity mapping ``(1,)``.
    """

    nav_lon: object
    nav_lat: object
    nav_lev: object
    time: object
    time_steps: object
    index_g: tuple[int, ...] = (1,)
    global_attributes: Mapping[str, object] | None = None


@dataclass(frozen=True)
class StomateReadstartReport:
    """Field-level result of the audited ``readstart`` transition.

    Provenance is ``stomate_io.f90::readstart`` lines 501-1743.
    """

    restart_fields: tuple[str, ...]
    defaulted_fields: tuple[str, ...]
    mixed_sentinel_fields: tuple[str, ...]
    source_undefined_fields: tuple[str, ...]
    provenance_by_field: Mapping[str, str]
    uncovered_reachable_fields: tuple[str, ...]
    out_of_span_fields: tuple[str, ...]

    @property
    def implemented(self) -> bool:
        return not self.uncovered_reachable_fields and not self.out_of_span_fields


@dataclass(frozen=True)
class StomateReadstartStates:
    """Existing normalized state contracts after Fortran restart defaults."""

    entry_state: StomateRestartEntryState
    season_state: StomateRestartSeasonState
    daily_state: StomateDailyAccumulatorState
    gas_state: StomateOkPcRestartGasState
    remainder_state: StomateReadstartRemainderState
    report: StomateReadstartReport


@dataclass(frozen=True)
class StomateWriterestartIndexLabels:
    """Index suffixes selected before the Fortran restart writes.

    Provenance: ``stomate_io.f90::writerestart`` lines 2141-2171.
    The indices are Fortran one-based constants. An index not represented by
    the corresponding structural constants is a fatal source error.
    """

    litter: tuple[str, ...]
    level: tuple[str, ...]
    element: tuple[str, ...]
    pools: tuple[str, ...]


@dataclass(frozen=True)
class StomateWriterestartFieldLedger:
    """Normalized state coverage of the template-backed writerestart owner."""

    serialized_fields: tuple[str, ...]
    validated_not_serialized_fields: tuple[str, ...]
    source_local_not_restart_fields: tuple[str, ...]
    independent_netcdf_boundaries: tuple[str, ...]

    @property
    def normalized_state_fields(self) -> tuple[str, ...]:
        return tuple(sorted((*self.serialized_fields, *self.validated_not_serialized_fields)))


@dataclass(frozen=True)
class StomateFixedPathCarryState:
    """Restart fields unchanged by the fixed paper STOMATE path.

    This is an executable state-transition owner, not a default-value owner.
    Values are retained from the normalized day-start state only after the
    structural switches below prove that their Fortran writers are not called.
    """

    fields: Mapping[str, object]
    provenance_by_field: Mapping[str, tuple[str, ...]]


STOMATE_FIXED_PATH_CARRY_FIELDS = (
    "veget_lastlight", "need_adjacent",
    "carbon", "litter", "lignin_struc",
    "prod10", "prod100", "flux10", "flux100",
    "fpeat",
    "deepC_a", "deepC_s", "deepC_p",
    "thawed_humidity", "depth_organic_soil",
    "date",
    "O2_soil", "CH4_soil", "O2_snow", "CH4_snow",
    "fireindex", "firelitter", "co2_fire", "ni_acc",
    "uo_0", "uold2_0", "uo_wet1", "uold2_wet1", "uo_wet2", "uold2_wet2",
    "uo_wet3", "uold2_wet3", "uo_wet4", "uold2_wet4",
    "height_acro", "carbon_acro", "carbon_cato",
    "fwet_month", "liqwt_month", "liqwt_max", "fwet_series",
    "carbon_save", "deepC_a_save", "deepC_s_save", "deepC_p_save", "delta_fsave",
    "wshtotsum", "sr_ugb", "nb_ani", "grazed_frac", "import_yield",
    "litter_not_avail", "nb_grazingdays", "after_snow", "after_wet", "wet1day", "wet2day",
    "Global_years", "nbp_sum", "nbp_flux", "ok_equilibrium",
    "MatrixV", "Vector_U", "previous_stock", "current_stock",
)


_FIXED_PATH_CARRY_PROVENANCE = {
    "dgvm_light": (
        "configs/orchidee_man_250919.yaml run_def_flags.STOMATE_OK_DGVM=false",
        "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj/light lines 1277-1297",
    ),
    "dgvm_establish": (
        "configs/orchidee_man_250919.yaml run_def_flags.STOMATE_OK_DGVM=false and LPJ_GAP_CONST_MORT=true",
        "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj/establish lines 1299-1320",
    ),
    "ok_leak_legacy": (
        "configs/orchidee_man_250919.yaml run_def_flags.OK_LEAK=true",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3288-3565 selects littercalc_leak/soilcarbon_leak instead of legacy carbon/litter state",
    ),
    "no_lcc_products": (
        "configs/orchidee_man_250919.yaml run_def_flags.LAND_COVER_CHANGE=false",
        "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj/lcchange lines 1413-1543",
    ),
    "no_permafrost_or_lcc": (
        "configs/orchidee_man_250919.yaml run_def_flags.OK_PC=false and LAND_COVER_CHANGE=false",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3634-3665",
        "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj/lcchange lines 1413-1543",
    ),
    "no_permafrost_inputs": (
        "configs/orchidee_man_250919.yaml run_def_flags.OK_PC=false",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3634-3665",
    ),
    "date": (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 869-872 declares date as SAVE state",
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 501-506 and writerestart lines 2193-2195",
        "source audit: stomate.f90 reads date at lines 2899 and 2980 and contains no post-read assignment",
    ),
    "ok_pc": (
        "configs/orchidee_man_250919.yaml run_def_flags.OK_PC=false",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3342-3370 and 3634-3665",
    ),
    "fire": (
        "configs/orchidee_man_250919.yaml run_def_flags.FIRE_DISABLE=true",
        "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90 lines 1184-1221 gates SPITFIRE state updates",
    ),
    "wetland_ch4": (
        "fortran_run_scripts/paper_250919/run.def.vn line 96 sets CH4_CALCUL=n",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3671-3742 gates wetland CH4 state updates",
    ),
    "legacy_peat": (
        "configs/orchidee_man_250919.yaml run_def_flags.OK_LEAK=true and OK_PEAT=false",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3342-3512 selects OK_LEAK instead of legacy peat soilcarbon",
    ),
    "peat_occurrence": (
        "configs/orchidee_man_250919.yaml run_def_flags.PEAT_OCCUR=false",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3593-3630 gates wetland-fraction memories",
    ),
    "dynamic_peat": (
        "configs/orchidee_man_250919.yaml run_def_flags.DYN_PEAT=false and LAND_COVER_CHANGE=false",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3006-3017 gates dynamic peat fraction updates",
        "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90 lines 3089-3187 updates saved peat pools only during cover change",
    ),
    "grazing": (
        "fortran_run_scripts/paper_250919/run.def.vn line 93 sets ENABLE_GRAZING=n",
        "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj lines 1323-1356 gates grassland-management state updates",
    ),
    "spinup": (
        "paper run keeps SPINUP_ANALYTIC=false",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 4938-4941 and 5122-5226 gates analytic-spinup state updates",
    ),
}


def stomate_fixed_path_carry_state(
    *,
    entry_state: StomateRestartEntryState,
    season_state: StomateRestartSeasonState,
    gas_state: StomateOkPcRestartGasState,
    remainder_state: StomateReadstartRemainderState,
    fire_disable: bool,
    ok_pc: bool,
    ch4_calcul: bool,
    ok_leak: bool,
    ok_peat: bool,
    peat_occur: bool,
    dyn_peat: bool,
    land_cover_change: bool,
    enable_grazing: bool,
    spinup_analytic: bool,
    stomate_ok_dgvm: bool,
    lpj_gap_const_mort: bool,
) -> StomateFixedPathCarryState:
    """Carry only fields whose update owners are unreachable in this path."""

    actual = {
        "fire_disable": bool(fire_disable),
        "ok_pc": bool(ok_pc),
        "ch4_calcul": bool(ch4_calcul),
        "ok_leak": bool(ok_leak),
        "ok_peat": bool(ok_peat),
        "peat_occur": bool(peat_occur),
        "dyn_peat": bool(dyn_peat),
        "land_cover_change": bool(land_cover_change),
        "enable_grazing": bool(enable_grazing),
        "spinup_analytic": bool(spinup_analytic),
        "stomate_ok_dgvm": bool(stomate_ok_dgvm),
        "lpj_gap_const_mort": bool(lpj_gap_const_mort),
    }
    expected = {
        "fire_disable": True,
        "ok_pc": False,
        "ch4_calcul": False,
        "ok_leak": True,
        "ok_peat": False,
        "peat_occur": False,
        "dyn_peat": False,
        "land_cover_change": False,
        "enable_grazing": False,
        "spinup_analytic": False,
        "stomate_ok_dgvm": False,
        "lpj_gap_const_mort": True,
    }
    mismatches = tuple(name for name, value in actual.items() if value != expected[name])
    if mismatches:
        raise ValueError(
            "fixed-path carry is invalid when structural update owners are reachable: "
            + ", ".join(mismatches)
        )

    source = {
        **{name: getattr(entry_state, name) for name in entry_state._fields},
        "date": season_state.date,
        **{name: getattr(gas_state, name) for name in gas_state._fields if name != "provenance"},
        **{name: getattr(remainder_state, name) for name in remainder_state._fields if name != "provenance"},
    }
    missing = tuple(name for name in STOMATE_FIXED_PATH_CARRY_FIELDS if source.get(name) is None)
    if missing:
        raise ValueError("fixed-path carry requires defined restart values: " + ", ".join(missing))

    groups = {
        "veget_lastlight": "dgvm_light",
        "need_adjacent": "dgvm_establish",
        **{name: "ok_leak_legacy" for name in ("carbon", "litter", "lignin_struc")},
        **{name: "no_lcc_products" for name in ("prod10", "prod100", "flux10", "flux100")},
        "fpeat": "peat_occurrence",
        **{name: "no_permafrost_or_lcc" for name in ("deepC_a", "deepC_s", "deepC_p")},
        **{name: "no_permafrost_inputs" for name in ("thawed_humidity", "depth_organic_soil")},
        **{name: "date" for name in ("date",)},
        **{name: "ok_pc" for name in ("O2_soil", "CH4_soil", "O2_snow", "CH4_snow")},
        **{name: "fire" for name in ("fireindex", "firelitter", "co2_fire", "ni_acc")},
        **{name: "wetland_ch4" for name in (
            "uo_0", "uold2_0", "uo_wet1", "uold2_wet1", "uo_wet2", "uold2_wet2",
            "uo_wet3", "uold2_wet3", "uo_wet4", "uold2_wet4",
        )},
        **{name: "legacy_peat" for name in ("height_acro", "carbon_acro", "carbon_cato")},
        **{name: "peat_occurrence" for name in ("fwet_month", "liqwt_month", "liqwt_max", "fwet_series")},
        **{name: "dynamic_peat" for name in (
            "carbon_save", "deepC_a_save", "deepC_s_save", "deepC_p_save", "delta_fsave",
        )},
        **{name: "grazing" for name in (
            "wshtotsum", "sr_ugb", "nb_ani", "grazed_frac", "import_yield", "litter_not_avail",
            "nb_grazingdays", "after_snow", "after_wet", "wet1day", "wet2day",
        )},
        **{name: "spinup" for name in (
            "Global_years", "nbp_sum", "nbp_flux", "ok_equilibrium",
            "MatrixV", "Vector_U", "previous_stock", "current_stock",
        )},
    }
    fields = {name: source[name] for name in STOMATE_FIXED_PATH_CARRY_FIELDS}
    provenance = {name: _FIXED_PATH_CARRY_PROVENANCE[groups[name]] for name in fields}
    return StomateFixedPathCarryState(fields=fields, provenance_by_field=provenance)


def writerestart_index_labels(
    *,
    nlitt: int,
    nlevs: int,
    nelements: int,
    imetabolic: int = 1,
    istructural: int = 2,
    iabove: int = 1,
    ibelow: int = 2,
    icarbon: int = 1,
) -> StomateWriterestartIndexLabels:
    """Reproduce the source-order label selection in ``writerestart``.

    This owns the ten active arms at lines 2142, 2144, 2152, 2154, and
    2162. It deliberately uses explicit source-order tests rather than a
    permissive lookup so duplicate or missing structural indices have the
    same first-match/error behavior as the Fortran ``IF/ELSEIF`` chains.
    """

    litter: list[str] = []
    for index in range(1, int(nlitt) + 1):
        if index == int(imetabolic):
            litter.append("met")
        elif index == int(istructural):
            litter.append("str")
        else:
            raise ValueError(f"writerestart cannot define litter_str for Fortran index {index}")

    level: list[str] = []
    for index in range(1, int(nlevs) + 1):
        if index == int(iabove):
            level.append("ab")
        elif index == int(ibelow):
            level.append("be")
        else:
            raise ValueError(f"writerestart cannot define level_str for Fortran index {index}")

    element: list[str] = []
    for index in range(1, int(nelements) + 1):
        if index == int(icarbon):
            element.append("")
        else:
            raise ValueError(f"writerestart cannot define element_str for Fortran index {index}")

    return StomateWriterestartIndexLabels(
        litter=tuple(litter),
        level=tuple(level),
        element=tuple(element),
        pools=("str_ab", "str_be", "met_ab", "met_be", "actif ", "slow  ", "passif"),
    )


_PFT_FIELDS: Mapping[str, str] = {
    "gpp_daily": "gpp_daily",
    "npp_daily": "npp_daily",
    "resp_maint": "resp_maint",
    "resp_growth": "resp_growth",
    "age": "age",
    "sla_calc": "sla_calc",
    "pft_present": "PFTpresent",
    "ind": "ind",
    "adapted": "adapted",
    "regenerate": "regenerate",
    "npp_longterm": "npp_longterm",
    "lm_lastyearmax": "lm_lastyearmax",
    "turnover_time": "turnover_time",
    "senescence": "senescence",
    "when_growthinit": "when_growthinit",
    "co2_to_bm": "co2_to_bm_dgvm",
    "veget_lastlight": "veget_lastlight",
    "everywhere": "everywhere",
    "need_adjacent": "need_adjacent",
    "rip_time": "RIP_time",
    "altmax": "altmax",
    "fixed_cryoturbation_depth": "fixed_cryoturb_depth",
    "lignin_struc_above": "lignin_struc_above",
}

_GRID_FIELDS: Mapping[str, str] = {
    "carb_mass_total": "carb_mass_total",
    "fpeat": "fpeat",
    "thawed_humidity": "thawed_humidity",
    "depth_organic_soil": "depth_organic_soil",
}

_PFT_POOL_FIELDS: Mapping[str, str] = {
    "biomass": "biomass",
    "turnover_daily": "turnover_daily",
    "turnover_longterm": "turnover_longterm",
    "bm_to_litter": "bm_to_litter",
}

_DIRECT_FIELDS = frozenset(
    {
        *_PFT_FIELDS,
        *_GRID_FIELDS,
        *_PFT_POOL_FIELDS,
        "resp_maint_part",
        "leaf_age",
        "leaf_frac",
        "litterpart",
        "dead_leaves",
        "carbon",
        "litter",
        "lignin_struc",
        "fuel_1hr",
        "fuel_10hr",
        "fuel_100hr",
        "fuel_1000hr",
        "prod10",
        "prod100",
        "flux10",
        "flux100",
        "assim_param",
        "litter_above",
        "litter_below",
        "carbon_32l",
        "DOC",
        "interception_storage",
        "lignin_struc_below",
        "deepC_a",
        "deepC_s",
        "deepC_p",
    }
)
_DERIVED_FIELDS = frozenset({"prod10_total", "prod100_total", "soilc_total"})
_BOOL_FIELDS = frozenset({"pft_present", "senescence", "need_adjacent", "begin_leaves"})

_SEASON_SCALAR_FIELDS: Mapping[str, str] = {
    "dt_days_read": "dt_days",
    "date": "date",
    "tau_longterm": "tau_longterm",
}
_SEASON_PFT_FIELDS: Mapping[str, str] = {
    "moiavail_month": "moiavail_month",
    "moiavail_week": "moiavail_week",
    "gdd_m5_dormance": "gdd_m5_dormance",
    "gdd_from_growthinit": "gdd_from_growthinit",
    "gdd_midwinter": "gdd_midwinter",
    "ncd_dormance": "ncd_dormance",
    "ngd_minus5": "ngd_minus5",
    "time_hum_min": "time_hum_min",
    "hum_min_dormance": "hum_min_dormance",
    "tmin_spring_time": "Tmin_spring_time",
    "begin_leaves": "begin_leaves",
    "onset_date": "onset_date",
    "gpp_week": "gpp_week",
    "maxmoiavail_lastyear": "maxmoistr_last",
    "maxmoiavail_thisyear": "maxmoistr_this",
    "minmoiavail_lastyear": "minmoistr_last",
    "minmoiavail_thisyear": "minmoistr_this",
    "maxgppweek_lastyear": "maxgppweek_lastyear",
    "maxgppweek_thisyear": "maxgppweek_thisyear",
    "maxfpc_lastyear": "maxfpc_lastyear",
    "maxfpc_thisyear": "maxfpc_thisyear",
    "lm_thisyearmax": "lm_thisyearmax",
}
_SEASON_GRID_FIELDS: Mapping[str, str] = {
    "t2m_longterm": "t2m_longterm",
    "t2m_month": "t2m_month",
    "t2m_week": "t2m_week",
    "tseason": "Tseason",
    "tseason_length": "Tseason_length",
    "tseason_tmp": "Tseason_tmp",
    "gdd0_lastyear": "gdd0_lastyear",
    "gdd0_thisyear": "gdd0_thisyear",
    "precip_lastyear": "precip_lastyear",
    "precip_thisyear": "precip_thisyear",
}
_SEASON_LAYER_FIELDS: Mapping[str, str] = {
    "tsoil_month": "tsoil_month",
    "soilhum_month": "soilhum_month",
    "gdd_init_date": "gdd_init_date",
}
_SEASON_FIELDS = frozenset(
    {
        *_SEASON_SCALAR_FIELDS,
        *_SEASON_PFT_FIELDS,
        *_SEASON_GRID_FIELDS,
        *_SEASON_LAYER_FIELDS,
    }
)

_DAILY_PFT_FIELDS: Mapping[str, str] = {
    "humrel_daily": "moiavail_daily",
    "gpp_daily": "gpp_daily",
}
_DAILY_GRID_FIELDS: Mapping[str, str] = {
    "litterhum_daily": "litterhum_daily",
    "t2m_daily": "t2m_daily",
    "t2m_min_daily": "t2m_min_daily",
    "t2m_max_daily": "t2m_max_daily",
    "wspeed_daily": "wspeed_daily",
    "tsurf_daily": "tsurf_daily",
    "precip_daily": "precip_daily",
    "fwet_daily": "fwet_daily",
    "liqwt_daily": "liqwt_daily",
}
_DAILY_LAYER_FIELDS: Mapping[str, str] = {
    "tsoil_daily": "tsoil_daily",
    "soilhum_daily": "soilhum_daily",
}
_DAILY_RUNTIME_FIELDS = frozenset({"snowfall_daily", "snowmass_daily", "tmc_topgrass_daily"})
_DAILY_DIRECT_FIELDS = frozenset({*_DAILY_PFT_FIELDS, *_DAILY_GRID_FIELDS, *_DAILY_LAYER_FIELDS})
_DAILY_FIELDS = _DAILY_DIRECT_FIELDS | _DAILY_RUNTIME_FIELDS

_GAS_WRITE_FIELDS = frozenset(
    name for name in StomateOkPcRestartGasState._fields if name != "provenance"
)
_REMAINDER_SOURCE_LOCAL_FIELDS = frozenset(
    {"read_input_thawed_humidity", "read_input_depth_organic_soil"}
)
_REMAINDER_WRITE_FIELDS = frozenset(
    name
    for name in StomateReadstartRemainderState._fields
    if name != "provenance" and name not in _REMAINDER_SOURCE_LOCAL_FIELDS
)
_FULL_WRITERESTART_SERIALIZED_FIELDS = (
    _DIRECT_FIELDS | _SEASON_FIELDS | _DAILY_DIRECT_FIELDS | _GAS_WRITE_FIELDS | _REMAINDER_WRITE_FIELDS
)

_READSTART_REMAINDER_FIELDS = (
    # Direct, unconditional reads added to the remainder normalized contract.
    # Source: stomate_io.f90 lines 707-717,
    # 979-1001, 1075-1079, 1245-1509, and 1535-1732.
    "fireindex",
    "firelitter",
    "resp_hetero",
    "co2_fire",
    "ni_acc",
    "read_input_thawed_humidity",
    "read_input_depth_organic_soil",
    "uo_0",
    "uold2_0",
    "uo_wet1",
    "uold2_wet1",
    "uo_wet2",
    "uold2_wet2",
    "uo_wet3",
    "uold2_wet3",
    "uo_wet4",
    "uold2_wet4",
    "tsurf_year",
    "height_acro",
    "carbon_acro",
    "carbon_cato",
    "fwet_month",
    "liqwt_month",
    "liqwt_max",
    "fwet_series",
    "deepC_peat",
    "carbon_save",
    "deepC_a_save",
    "deepC_s_save",
    "deepC_p_save",
    "delta_fsave",
    "depth_deepsoil",
    "wshtotsum",
    "sr_ugb",
    "nb_ani",
    "grazed_frac",
    "import_yield",
    "t2m_14",
    "litter_not_avail",
    "nb_grazingdays",
    "after_snow",
    "after_wet",
    "wet1day",
    "wet2day",
    "Global_years",
    "nbp_sum",
    "nbp_flux",
    "ok_equilibrium",
    "MatrixV",
    "Vector_U",
    "previous_stock",
    "current_stock",
)

_REMAINDER_LINES: Mapping[str, str] = {
    "fireindex": "707-711", "firelitter": "713-717",
    "resp_hetero": "979-983", "co2_fire": "997-1001", "ni_acc": "1075-1079",
    "read_input_thawed_humidity": "1201-1215",
    "read_input_depth_organic_soil": "1218-1225",
    "uo_0": "1247-1259", "uold2_0": "1261-1273",
    "uo_wet1": "1275-1287", "uold2_wet1": "1289-1301",
    "uo_wet2": "1303-1315", "uold2_wet2": "1317-1329",
    "uo_wet3": "1331-1343", "uold2_wet3": "1345-1357",
    "uo_wet4": "1359-1371", "uold2_wet4": "1373-1385",
    "tsurf_year": "1387-1391", "height_acro": "1394-1400",
    "carbon_acro": "1402-1408", "carbon_cato": "1410-1416",
    "fwet_month": "1434-1440", "liqwt_month": "1442-1448",
    "liqwt_max": "1450-1456", "fwet_series": "1458-1464",
    "deepC_peat": "1474-1478", "carbon_save": "1480-1484",
    "deepC_a_save": "1487-1491", "deepC_s_save": "1493-1497",
    "deepC_p_save": "1499-1503", "delta_fsave": "1505-1509",
    "depth_deepsoil": "1535-1538", "wshtotsum": "1587-1591",
    "sr_ugb": "1593-1597", "nb_ani": "1610-1614",
    "grazed_frac": "1616-1620", "import_yield": "1622-1626",
    "t2m_14": "1629-1633", "litter_not_avail": "1636-1639",
    "nb_grazingdays": "1641-1645", "after_snow": "1648-1652",
    "after_wet": "1654-1658", "wet1day": "1660-1664", "wet2day": "1666-1670",
    "Global_years": "1675-1679", "nbp_sum": "1681-1685", "nbp_flux": "1687-1691",
    "ok_equilibrium": "1694-1703", "MatrixV": "1705-1714",
    "Vector_U": "1716-1720", "previous_stock": "1722-1726",
    "current_stock": "1728-1732",
}


def _readstart_provenance(field: str) -> str:
    if field in _DAILY_FIELDS or field in {"npp_daily", "turnover_daily"}:
        lines = "501-596"
    elif field in _SEASON_FIELDS:
        lines = "501-839"
    elif field in {"litter_above", "litter_below", "lignin_struc_above", "lignin_struc_below", "carbon_32l", "DOC", "interception_storage"}:
        lines = "1512-1571"
    elif field == "sla_calc":
        lines = "1599-1608"
    else:
        lines = "843-1239"
    return f"fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines {lines}"


def _readstart_field(path: Path, field: str, source_name: str) -> np.ndarray | float:
    if field == "biomass":
        return read_restart_biomass_carbon(path)
    if field == "resp_maint_part":
        return read_restart_maint_resp_part(path)
    if field in {"turnover_daily", "turnover_longterm", "bm_to_litter"}:
        return read_restart_pft_pool_field(path, source_name)
    if field in {"leaf_age", "leaf_frac"}:
        return read_restart_leaf_age_field(path, source_name)
    if field in {"litterpart", "dead_leaves"}:
        return read_restart_litter_pft_field(path, source_name)
    if field == "carbon":
        return read_restart_legacy_carbon(path)
    if field == "litter":
        return read_restart_litter_legacy(path)
    if field == "lignin_struc":
        return read_restart_lignin_struc_legacy(path)
    if field.startswith("fuel_"):
        return read_restart_litter_fuel_field(path, source_name)
    if field in {"prod10", "prod100", "flux10", "flux100"}:
        return read_restart_product_pool(path, source_name, int(source_name.removeprefix("prod").removeprefix("flux")) + (1 if source_name.startswith("prod") else 0))
    if field == "assim_param":
        return read_restart_assim_param(path)
    if field == "litter_above":
        return read_restart_litter_above(path)
    if field == "litter_below":
        return read_restart_litter_below(path)
    if field == "interception_storage":
        return read_restart_interception_storage(path)
    if field == "lignin_struc_below":
        return read_restart_lignin_struc_below(path)
    if field in {"deepC_a", "deepC_s", "deepC_p"}:
        return read_restart_deep_carbon_pool(path, source_name)
    if field in _SEASON_SCALAR_FIELDS:
        return read_restart_scalar(path, source_name)
    if field in _SEASON_LAYER_FIELDS or field in _DAILY_LAYER_FIELDS:
        return read_restart_soil_layer_field(path, source_name)
    if field in _GRID_FIELDS or field in _SEASON_GRID_FIELDS or field in _DAILY_GRID_FIELDS:
        return read_restart_grid_field(path, source_name)
    return read_restart_pft_field(path, source_name)


def _resolve_readstart_value(value: object, fallback: object, *, val_exp: float) -> tuple[object, str]:
    array = np.asarray(value)
    expected = np.asarray(fallback)
    if array.shape != expected.shape:
        raise ValueError(f"restart/default shape mismatch: restart={array.shape}, expected={expected.shape}")
    sentinel = array == float(val_exp)
    if np.all(sentinel):
        return fallback, "default"
    if np.any(sentinel):
        return value, "mixed"
    return value, "restart"


def _read_fortran_axes_field(path: Path, name: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    """Normalize IOIPSL ``(time,reversed Fortran axes,y,x)`` storage."""

    with Dataset(path) as dataset:
        field = np.asarray(dataset.variables[name][:])
    fortran_rank = len(expected_shape) - 1
    if field.ndim != fortran_rank + 3 or field.shape[0] != 1:
        raise ValueError(
            f"{name} must have time, {fortran_rank} reversed Fortran axes, y, x; got {field.shape}"
        )
    selected = field[0]
    order = (selected.ndim - 2, selected.ndim - 1, *range(selected.ndim - 3, -1, -1))
    normalized = np.transpose(selected, order).reshape(expected_shape)
    return normalized


def read_stomate_readstart_states_from_template(
    path: str | Path,
    *,
    t2m: object,
    nvm: int,
    nslm: int,
    source_pft_layout: PFTRunLayout | None = None,
    target_pft_layout: PFTRunLayout | None = None,
    dt_days_default: float = 1.0,
    date_default: int = 0,
    ndeep: int = 32,
    nelements: int = 1,
    val_exp: float = 999999.0,
    large_value: float = 1.0e33,
    undef: float = -9999.0,
    gdd_crit_estab: float = 150.0,
    precip_crit: float = 100.0,
    tau_longterm_max: float = 3.0,
    sla: object | None = None,
    thawed_humidity_input: float = 1.0e20,
    reset_thawed_humidity: bool = False,
    nsnow: int = 3,
    o2_init_conc: float = 0.0,
    ch4_init_conc: float = 0.0,
    nvert: int = 171,
    ns: int = 151,
    scmax: float = 500.0,
    ch4atmo_conc: float = 0.0017,
    months_num: int = 360,
    ncarb: int = 3,
    nlitt: int = 2,
    nbpools: int = 7,
    undef_sechiba: float = 1.0e20,
) -> StomateReadstartStates:
    """Apply audited Fortran restart/default semantics to existing states.

    The NetCDF file remains the physical I/O boundary. Axis transforms and
    packed aliases reuse the existing reference readers. Missing variables and
    arrays containing only ``val_exp`` take the exact whole-array defaults in
    ``readstart``; partially initialized arrays retain their sentinel entries,
    matching Fortran's ``IF (ALL(... == val_exp))`` behavior.

    The remainder state closes the additional 52-field audit list in source
    order.  A ``None`` value is a typed source-undefined contract, not a
    numerical default.
    """

    restart_path = Path(path)
    if not restart_path.is_file():
        raise FileNotFoundError(restart_path)
    t2m_array = np.asarray(t2m, dtype=np.float64)
    if t2m_array.ndim != 1:
        raise ValueError("t2m must have shape (npts,)")

    daily = stomate_cold_start_daily_accumulator_state(
        t2m=t2m_array,
        nvm=nvm,
        nslm=nslm,
        large_value=large_value,
    )._asdict()
    season = stomate_cold_start_season_state(
        t2m=t2m_array,
        dt_days=dt_days_default,
        nvm=nvm,
        nslm=nslm,
        date=date_default,
        large_value=large_value,
        undef=undef,
        gdd_crit_estab=gdd_crit_estab,
        precip_crit=precip_crit,
    )._asdict()
    entry = stomate_cold_start_entry_state(
        t2m=t2m_array,
        nvm=nvm,
        nslm=nslm,
        ndeep=ndeep,
        nelements=nelements,
        dt_days=dt_days_default,
        date=date_default,
        sla=sla,
        thawed_humidity_input=thawed_humidity_input,
        large_value=large_value,
    )._asdict()
    daily.pop("provenance", None)
    season.pop("provenance", None)

    with Dataset(restart_path) as dataset:
        present = frozenset(dataset.variables)

    restart_fields: set[str] = set()
    defaulted_fields: set[str] = set(_DAILY_RUNTIME_FIELDS)
    mixed_fields: set[str] = set()
    source_undefined_fields: set[str] = set()

    def record(field: str, status: str) -> None:
        target = restart_fields if status == "restart" else mixed_fields if status == "mixed" else defaulted_fields
        target.add(field)

    def overlay(target: dict[str, object], field: str, source_name: str) -> None:
        if source_name not in present:
            record(field, "default")
            return
        value, status = _resolve_readstart_value(
            _readstart_field(restart_path, field, source_name),
            target[field],
            val_exp=val_exp,
        )
        if field in _BOOL_FIELDS:
            value = np.asarray(value, dtype=np.float64) >= 0.5
        target[field] = value
        record(field, status)

    entry_sources = {
        **_PFT_FIELDS,
        **_GRID_FIELDS,
        **_PFT_POOL_FIELDS,
        "resp_maint_part": "maint_resp",
        "leaf_age": "leaf_age",
        "leaf_frac": "leaf_frac",
        "litterpart": "litterpart",
        "dead_leaves": "dead_leaves",
        "carbon": "carbon",
        "litter": "litter",
        "lignin_struc": "lignin_struc",
        "fuel_1hr": "fuel_1hr",
        "fuel_10hr": "fuel_10hr",
        "fuel_100hr": "fuel_100hr",
        "fuel_1000hr": "fuel_1000hr",
        "prod10": "prod10",
        "prod100": "prod100",
        "flux10": "flux10",
        "flux100": "flux100",
        "assim_param": "assim_param",
        "litter_above": "litter_above",
        "litter_below": "litter_below",
        "interception_storage": "interception_storage",
        "lignin_struc_below": "lig_struc_be",
        "deepC_a": "deepC_a",
        "deepC_s": "deepC_s",
        "deepC_p": "deepC_p",
    }
    for field, source_name in entry_sources.items():
        if field != "thawed_humidity" or not reset_thawed_humidity:
            overlay(entry, field, source_name)
    if reset_thawed_humidity:
        entry["thawed_humidity"] = np.full(t2m_array.size, thawed_humidity_input, dtype=np.float64)
        record("thawed_humidity", "default")

    for field, source_name in {
        **_SEASON_SCALAR_FIELDS,
        **_SEASON_PFT_FIELDS,
        **_SEASON_GRID_FIELDS,
        **_SEASON_LAYER_FIELDS,
    }.items():
        if field not in {"tau_longterm", "gdd_init_date"}:
            overlay(season, field, source_name)

    for field, source_name in {
        **_DAILY_PFT_FIELDS,
        **_DAILY_GRID_FIELDS,
        **_DAILY_LAYER_FIELDS,
    }.items():
        overlay(daily, field, source_name)

    # readstart lines 516-520 default only column 1; column 2 remains val_exp.
    gdd_fallback = np.full((t2m_array.size, 2), float(val_exp), dtype=np.float64)
    gdd_fallback[:, 0] = 365.0
    season["gdd_init_date"] = gdd_fallback
    if "gdd_init_date" in present:
        value, status = _resolve_readstart_value(
            read_restart_soil_layer_field(restart_path, "gdd_init_date"),
            np.full_like(gdd_fallback, val_exp),
            val_exp=val_exp,
        )
        value_array = np.asarray(value).copy()
        if np.all(value_array[:, 0] == val_exp):
            value_array[:, 0] = 365.0
            status = "default" if np.all(value_array[:, 1] == val_exp) else "mixed"
        season["gdd_init_date"] = value_array
        record("gdd_init_date", status)
    else:
        record("gdd_init_date", "default")

    # readstart lines 620-637 couple tau_longterm to t2m_longterm presence.
    if "t2m_longterm" in defaulted_fields:
        season["tau_longterm"] = 2.0
        record("tau_longterm", "default")
    elif "tau_longterm" not in present:
        season["tau_longterm"] = float(tau_longterm_max)
        record("tau_longterm", "default")
    else:
        tau = read_restart_scalar(restart_path, "tau_longterm")
        if tau == float(val_exp):
            season["tau_longterm"] = float(tau_longterm_max)
            record("tau_longterm", "default")
        else:
            season["tau_longterm"] = tau
            record("tau_longterm", "restart")

    # Preserve the source alias packing while allowing individual aliases to
    # be absent, as restget_p leaves only that destination slice at val_exp.
    carbon_32l = np.full_like(np.asarray(entry["carbon_32l"]), val_exp, dtype=np.float64)
    for index, name in enumerate(("carbon_32l_a", "carbon_32l_s", "carbon_32l_p")):
        if name in present:
            carbon_32l[:, index] = _read_restart_carbon_32l_pool(restart_path, name)
    entry["carbon_32l"], status = _resolve_readstart_value(
        carbon_32l, entry["carbon_32l"], val_exp=val_exp
    )
    record("carbon_32l", status)

    doc = np.full_like(np.asarray(entry["DOC"]), val_exp, dtype=np.float64)
    for index, name in enumerate(("freedoc", "adsdoc")):
        if name in present:
            doc[:, :, :, index] = _read_restart_doc_pool(restart_path, name)
    entry["DOC"], status = _resolve_readstart_value(doc, entry["DOC"], val_exp=val_exp)
    record("DOC", status)

    entry["prod10_total"] = np.sum(np.asarray(entry["prod10"]), axis=(1, 2))
    entry["prod100_total"] = np.sum(np.asarray(entry["prod100"]), axis=(1, 2))
    entry["soilc_total"] = np.asarray(entry["deepC_a"]) + np.asarray(entry["deepC_s"]) + np.asarray(entry["deepC_p"])
    for field, source in (("prod10_total", "prod10"), ("prod100_total", "prod100"), ("soilc_total", "deepC_a")):
        record(field, "default" if source in defaulted_fields else "mixed" if source in mixed_fields else "restart")

    # gpp_daily is one source variable shared by both public state contracts.
    daily["gpp_daily"] = entry["gpp_daily"]
    season["dt_days_read"] = float(season["dt_days_read"])
    season["date"] = int(round(float(season["date"])))

    fields = set(entry) | set(season) | set(daily)
    provenance = {field: _readstart_provenance(field) for field in fields}
    provenance["prod10_total"] = "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90 lines 1545-1548"
    provenance["prod100_total"] = provenance["prod10_total"]
    provenance["soilc_total"] = "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_init lines 1586-1589"
    for field in _DAILY_RUNTIME_FIELDS:
        provenance[field] = "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 8181-8321"

    gas: dict[str, np.ndarray] = {}
    gas_specs = {
        "O2_soil": ((t2m_array.size, int(ndeep), int(nvm)), float(o2_init_conc)),
        "CH4_soil": ((t2m_array.size, int(ndeep), int(nvm)), float(ch4_init_conc)),
        "O2_snow": ((t2m_array.size, int(nsnow), int(nvm)), float(o2_init_conc)),
        "CH4_snow": ((t2m_array.size, int(nsnow), int(nvm)), float(ch4_init_conc)),
    }
    for field, (shape, initial_value) in gas_specs.items():
        fallback = np.full(shape, initial_value, dtype=np.float64)
        if field in present:
            gas[field], status = _resolve_readstart_value(
                read_restart_pft_vertical_field(restart_path, field),
                fallback,
                val_exp=val_exp,
            )
            record(field, status)
        else:
            gas[field] = fallback
            record(field, "default")
        provenance[field] = (
            "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 1177-1199"
        )

    npts = t2m_array.size
    zero_grid = np.zeros((npts,), dtype=np.float64)
    zero_pft = np.zeros((npts, int(nvm)), dtype=np.float64)
    remainder: dict[str, object] = {}

    def remainder_overlay(field: str, shape: tuple[int, ...], fallback: object) -> None:
        if field not in present:
            remainder[field] = fallback
            record(field, "default")
            return
        value = _read_fortran_axes_field(restart_path, field, shape)
        remainder[field], status = _resolve_readstart_value(value, fallback, val_exp=val_exp)
        record(field, status)

    # Fire, respiration, and SPITFIRE accumulator: lines 707-717, 979-1001,
    # and 1075-1079. All use whole-array zero fallback.
    for field in ("fireindex", "firelitter", "resp_hetero", "co2_fire"):
        remainder_overlay(field, zero_pft.shape, zero_pft.copy())
    remainder_overlay("ni_acc", zero_grid.shape, zero_grid.copy())

    # These locals are assigned only on the source fallback branch. They are
    # not read from the restart or exposed as readstart output arguments.
    if not reset_thawed_humidity and "thawed_humidity" in defaulted_fields:
        remainder["read_input_thawed_humidity"] = True
        record("read_input_thawed_humidity", "default")
    else:
        remainder["read_input_thawed_humidity"] = None
        source_undefined_fields.add("read_input_thawed_humidity")
    if "depth_organic_soil" in defaulted_fields:
        remainder["read_input_depth_organic_soil"] = True
        record("read_input_depth_organic_soil", "default")
    else:
        remainder["read_input_depth_organic_soil"] = None
        source_undefined_fields.add("read_input_depth_organic_soil")

    # Wetland CH4 lines 1245-1385. The first pair uses ns; wetland pairs use
    # ns-10. Fortran nivo is one-based, hence the Python slice endpoints.
    dry_ch4 = np.full((npts, int(nvert)), float(ch4atmo_conc), dtype=np.float64)
    dry_ch4[:, : int(ns)] = float(scmax)
    wet_ch4 = np.full((npts, int(nvert)), float(ch4atmo_conc), dtype=np.float64)
    wet_ch4[:, : max(int(ns) - 10, 0)] = float(scmax)
    for field in ("uo_0", "uold2_0"):
        remainder_overlay(field, dry_ch4.shape, dry_ch4.copy())
    for field in (
        "uo_wet1", "uold2_wet1", "uo_wet2", "uold2_wet2",
        "uo_wet3", "uold2_wet3", "uo_wet4", "uold2_wet4",
    ):
        remainder_overlay(field, wet_ch4.shape, wet_ch4.copy())

    remainder_overlay("tsurf_year", zero_grid.shape, t2m_array.copy())
    remainder_overlay("height_acro", zero_grid.shape, zero_grid.copy())
    for field in ("carbon_acro", "carbon_cato"):
        remainder_overlay(field, zero_pft.shape, zero_pft.copy())
    for field in ("fwet_month", "liqwt_month", "liqwt_max"):
        remainder_overlay(field, zero_grid.shape, zero_grid.copy())
    remainder_overlay(
        "fwet_series", (npts, int(months_num)), np.zeros((npts, int(months_num)), dtype=np.float64)
    )

    remainder_overlay(
        "deepC_peat", (npts, int(ndeep), int(nvm)),
        np.zeros((npts, int(ndeep), int(nvm)), dtype=np.float64),
    )
    remainder_overlay(
        "carbon_save", (npts, int(ncarb), int(nvm)),
        np.zeros((npts, int(ncarb), int(nvm)), dtype=np.float64),
    )
    for field in ("deepC_a_save", "deepC_s_save", "deepC_p_save"):
        remainder_overlay(field, (npts, int(ndeep)), np.zeros((npts, int(ndeep)), dtype=np.float64))
    remainder_overlay("delta_fsave", zero_grid.shape, zero_grid.copy())

    # Unlike every neighboring array, depth_deepsoil is not set to val_exp
    # before restget_p (lines 1535-1538). Missing-variable behavior is therefore
    # undefined in the source; an actual all-val_exp variable still triggers 0.
    if "depth_deepsoil" not in present:
        remainder["depth_deepsoil"] = None
        source_undefined_fields.add("depth_deepsoil")
    else:
        value = _read_fortran_axes_field(restart_path, "depth_deepsoil", zero_pft.shape)
        remainder["depth_deepsoil"], status = _resolve_readstart_value(
            value, zero_pft.copy(), val_exp=val_exp
        )
        record("depth_deepsoil", status)

    for field in ("wshtotsum", "sr_ugb", "nb_ani", "grazed_frac", "import_yield"):
        remainder_overlay(field, zero_pft.shape, zero_pft.copy())
    remainder_overlay("t2m_14", zero_grid.shape, t2m_array.copy())
    remainder_overlay(
        "litter_not_avail", (npts, int(nlitt), int(nvm)),
        np.zeros((npts, int(nlitt), int(nvm)), dtype=np.float64),
    )
    remainder_overlay("nb_grazingdays", zero_pft.shape, zero_pft.copy())
    for field in ("after_snow", "after_wet"):
        remainder_overlay(field, zero_grid.shape, zero_grid.copy())
    for field in ("wet1day", "wet2day"):
        remainder_overlay(field, zero_grid.shape, np.full((npts,), 6.0, dtype=np.float64))

    # Global_years uses restget_p's scalar default directly, without an ALL
    # sentinel test (lines 1677-1679).
    if "Global_years" in present:
        remainder["Global_years"] = int(round(read_restart_scalar(restart_path, "Global_years")))
        record("Global_years", "restart")
    else:
        remainder["Global_years"] = 0
        record("Global_years", "default")
    for field in ("nbp_sum", "nbp_flux"):
        remainder_overlay(field, zero_grid.shape, zero_grid.copy())
    remainder_overlay("ok_equilibrium", zero_grid.shape, zero_grid.copy())
    remainder["ok_equilibrium"] = np.asarray(remainder["ok_equilibrium"], dtype=np.float64) >= 0.5

    matrix_shape = (npts, int(nvm), int(nbpools), int(nbpools))
    matrix_fallback = np.zeros(matrix_shape, dtype=np.float64)
    diagonal = np.arange(int(nbpools))
    matrix_fallback[:, :, diagonal, diagonal] = 1.0
    remainder_overlay("MatrixV", matrix_shape, matrix_fallback)
    spinup_shape = (npts, int(nvm), int(nbpools))
    remainder_overlay("Vector_U", spinup_shape, np.zeros(spinup_shape, dtype=np.float64))
    remainder_overlay(
        "previous_stock", spinup_shape, np.full(spinup_shape, float(undef_sechiba), dtype=np.float64)
    )
    remainder_overlay("current_stock", spinup_shape, np.zeros(spinup_shape, dtype=np.float64))

    for field, lines in _REMAINDER_LINES.items():
        provenance[field] = (
            f"fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines {lines}"
        )

    states = StomateReadstartStates(
        entry_state=StomateRestartEntryState(**entry),
        season_state=StomateRestartSeasonState(**season),
        daily_state=StomateDailyAccumulatorState(**daily),
        gas_state=StomateOkPcRestartGasState(**gas),
        remainder_state=StomateReadstartRemainderState(**remainder),
        report=StomateReadstartReport(
            restart_fields=tuple(sorted(restart_fields)),
            defaulted_fields=tuple(sorted(defaulted_fields)),
            mixed_sentinel_fields=tuple(sorted(mixed_fields)),
            source_undefined_fields=tuple(sorted(source_undefined_fields)),
            provenance_by_field=provenance,
            uncovered_reachable_fields=(),
            out_of_span_fields=(),
        ),
    )
    if (source_pft_layout is None) != (target_pft_layout is None):
        raise ValueError("source and target PFT layouts must be provided together")
    if source_pft_layout is None or target_pft_layout is None:
        return states
    if int(nvm) != source_pft_layout.n_pft:
        raise ValueError("readstart nvm must match the declared source PFT layout")
    return StomateReadstartStates(
        entry_state=remap_stomate_state_by_pft_id(
            states.entry_state,
            pft_axes=STOMATE_ENTRY_PFT_AXES,
            source_pft_layout=source_pft_layout,
            target_pft_layout=target_pft_layout,
        ),
        season_state=remap_stomate_state_by_pft_id(
            states.season_state,
            pft_axes=STOMATE_SEASON_PFT_AXES,
            source_pft_layout=source_pft_layout,
            target_pft_layout=target_pft_layout,
        ),
        daily_state=remap_stomate_state_by_pft_id(
            states.daily_state,
            pft_axes=STOMATE_DAILY_PFT_AXES,
            source_pft_layout=source_pft_layout,
            target_pft_layout=target_pft_layout,
        ),
        gas_state=remap_stomate_state_by_pft_id(
            states.gas_state,
            pft_axes=STOMATE_GAS_PFT_AXES,
            source_pft_layout=source_pft_layout,
            target_pft_layout=target_pft_layout,
        ),
        remainder_state=remap_stomate_state_by_pft_id(
            states.remainder_state,
            pft_axes=STOMATE_REMAINDER_PFT_AXES,
            source_pft_layout=source_pft_layout,
            target_pft_layout=target_pft_layout,
        ),
        report=states.report,
    )


def _shape_error(field: str, expected: tuple[int, ...], actual: tuple[int, ...]) -> ValueError:
    return ValueError(f"{field} normalized shape must be {expected}, got {actual}")


def _require_shape(field: str, value: np.ndarray, expected: tuple[int, ...]) -> None:
    if value.shape != expected:
        raise _shape_error(field, expected, value.shape)


def _grid_shape(variable: object) -> tuple[int, int]:
    shape = tuple(int(size) for size in variable.shape)
    if len(shape) < 3 or shape[0] != 1:
        raise ValueError(f"template variable must have one time record and y/x axes, got {shape}")
    return shape[-2], shape[-1]


def _assign_selected(variable: object, field: str, selected: np.ndarray) -> None:
    target_shape = tuple(int(size) for size in variable.shape[1:])
    if selected.shape != target_shape:
        raise ValueError(
            f"{field} inverse transform produced {selected.shape}, "
            f"but template variable {variable.name} requires {target_shape}"
        )
    variable[0] = selected


def _inverse_point_axes(
    variable: object,
    field: str,
    value: object,
    normalized_tail: tuple[int, ...],
    selected_permutation: tuple[int, ...],
) -> None:
    array = np.asarray(value)
    ny, nx = _grid_shape(variable)
    _require_shape(field, array, (ny * nx, *normalized_tail))
    expanded = array.reshape(ny, nx, *normalized_tail)
    _assign_selected(variable, field, np.transpose(expanded, selected_permutation))


def _write_pft(variable: object, field: str, value: object) -> None:
    nvm = int(variable.shape[1])
    array = np.asarray(value)
    if field in _BOOL_FIELDS:
        array = np.where(array.astype(bool), 1.0, 0.0)
    _inverse_point_axes(variable, field, array, (nvm,), (2, 0, 1))


def _write_grid(variable: object, field: str, value: object) -> None:
    _inverse_point_axes(variable, field, value, (), (0, 1))


def _write_pft_pool(variable: object, field: str, value: object) -> None:
    nelements, nparts, nvm = map(int, variable.shape[1:4])
    if nelements != 1:
        raise ValueError(f"{field} writer only has the normalized carbon element, template has {nelements} elements")
    _inverse_point_axes(
        variable,
        field,
        value,
        (nvm, nparts, nelements),
        (4, 3, 2, 0, 1),
    )


def _write_maint_resp(variable: object, field: str, value: object) -> None:
    nparts, nvm = map(int, variable.shape[1:3])
    _inverse_point_axes(variable, field, value, (nvm, nparts), (3, 2, 0, 1))


def _write_component_pft(variable: object, field: str, value: object) -> None:
    ncomponent, nvm = map(int, variable.shape[1:3])
    _inverse_point_axes(variable, field, value, (nvm, ncomponent), (3, 2, 0, 1))


def _write_legacy_carbon(variable: object, field: str, value: object) -> None:
    nvm, ncarb = map(int, variable.shape[1:3])
    _inverse_point_axes(variable, field, value, (ncarb, nvm), (3, 2, 0, 1))


def _write_litter(variable: object, field: str, value: object) -> None:
    nelements, nlitt, nvm, nlevs = map(int, variable.shape[1:5])
    _inverse_point_axes(
        variable,
        field,
        value,
        (nlitt, nvm, nlevs, nelements),
        (5, 2, 3, 4, 0, 1),
    )


def _write_fuel(variable: object, field: str, value: object) -> None:
    nelements, nlitt, nvm = map(int, variable.shape[1:4])
    _inverse_point_axes(
        variable,
        field,
        value,
        (nvm, nlitt, nelements),
        (4, 3, 2, 0, 1),
    )


def _write_product(variable: object, field: str, value: object) -> None:
    nwp, nage = map(int, variable.shape[1:3])
    _inverse_point_axes(variable, field, value, (nage, nwp), (3, 2, 0, 1))


def _write_litter_above(variable: object, field: str, value: object) -> None:
    nelements, nvm, nlitt = map(int, variable.shape[1:4])
    _inverse_point_axes(
        variable,
        field,
        value,
        (nlitt, nvm, nelements),
        (4, 3, 2, 0, 1),
    )


def _write_litter_below(variable: object, field: str, value: object) -> None:
    nelements, ndeep, nvm, nlitt = map(int, variable.shape[1:5])
    _inverse_point_axes(
        variable,
        field,
        value,
        (nlitt, nvm, ndeep, nelements),
        (5, 4, 3, 2, 0, 1),
    )


def _write_carbon_32l(dataset: Dataset, field: str, value: object) -> None:
    array = np.asarray(value)
    aliases = ("carbon_32l_a", "carbon_32l_s", "carbon_32l_p")
    if array.ndim != 4 or array.shape[1] != len(aliases):
        raise ValueError(f"{field} must have shape (npts, 3, nvm, ndeep), got {array.shape}")
    for index, name in enumerate(aliases):
        variable = dataset.variables[name]
        ndeep, nvm = map(int, variable.shape[1:3])
        _inverse_point_axes(variable, field, array[:, index], (nvm, ndeep), (3, 2, 0, 1))


def _write_doc(dataset: Dataset, field: str, value: object) -> None:
    array = np.asarray(value)
    aliases = ("freedoc", "adsdoc")
    if array.ndim != 6 or array.shape[3] != len(aliases):
        raise ValueError(
            f"{field} must have shape (npts, nvm, ndeep, 2, npool, nelements), got {array.shape}"
        )
    for index, name in enumerate(aliases):
        variable = dataset.variables[name]
        nelements, npool, ndeep, nvm = map(int, variable.shape[1:5])
        _inverse_point_axes(
            variable,
            field,
            array[:, :, :, index],
            (nvm, ndeep, npool, nelements),
            (5, 4, 3, 2, 0, 1),
        )


def _write_interception(variable: object, field: str, value: object) -> None:
    array = np.asarray(value)
    nvm = int(variable.shape[1])
    ny, nx = _grid_shape(variable)
    _require_shape(field, array, (ny * nx, nvm, 1))
    _write_pft(variable, field, array[:, :, 0])


def _write_deep_carbon(variable: object, field: str, value: object) -> None:
    nvm, ndeep = map(int, variable.shape[1:3])
    _inverse_point_axes(variable, field, value, (ndeep, nvm), (3, 2, 0, 1))


def _write_scalar(variable: object, field: str, value: object) -> None:
    if int(variable.size) != 1:
        raise ValueError(f"{field} template variable must contain exactly one value, got shape {variable.shape}")
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{field} normalized value must be scalar, got shape {array.shape}")
    variable[:] = array.reshape(-1)[0]


def _write_layer(variable: object, field: str, value: object) -> None:
    nlayer = int(variable.shape[1])
    _inverse_point_axes(variable, field, value, (nlayer,), (2, 0, 1))


def _write_fortran_axes_field(variable: object, field: str, value: object) -> None:
    """Invert ``_read_fortran_axes_field`` for arbitrary source-declared axes."""

    storage_shape = tuple(int(size) for size in variable.shape)
    if len(storage_shape) < 3 or storage_shape[0] != 1:
        raise ValueError(
            f"{field} template variable must have time, reversed Fortran axes, y, x; "
            f"got {storage_shape}"
        )
    ny, nx = storage_shape[-2:]
    normalized_tail = tuple(reversed(storage_shape[1:-2]))
    array = np.asarray(value)
    _require_shape(field, array, (ny * nx, *normalized_tail))
    expanded = array.reshape(ny, nx, *normalized_tail)
    tail_axes = tuple(range(2, expanded.ndim))
    _assign_selected(variable, field, np.transpose(expanded, (*reversed(tail_axes), 0, 1)))


def _write_gas_dataset(dataset: Dataset, state: StomateOkPcRestartGasState) -> None:
    _require_state_mapping(state, _GAS_WRITE_FIELDS, "OK_PC gas")
    _require_variables(dataset, set(_GAS_WRITE_FIELDS))
    for field in sorted(_GAS_WRITE_FIELDS):
        _write_fortran_axes_field(dataset.variables[field], field, getattr(state, field))


def _write_remainder_dataset(dataset: Dataset, state: StomateReadstartRemainderState) -> None:
    expected = _REMAINDER_WRITE_FIELDS | _REMAINDER_SOURCE_LOCAL_FIELDS
    _require_state_mapping(state, expected, "readstart remainder")
    _require_variables(dataset, set(_REMAINDER_WRITE_FIELDS))
    for field in sorted(_REMAINDER_WRITE_FIELDS):
        value = getattr(state, field)
        if value is None:
            raise ValueError(
                f"writerestart requires a defined scientific value for {field}; "
                "None only represents source-undefined readstart behavior"
            )
        variable = dataset.variables[field]
        if field == "Global_years":
            _write_scalar(variable, field, value)
            continue
        if field == "ok_equilibrium":
            value = np.where(np.asarray(value, dtype=bool), 1.0, 0.0)
        _write_fortran_axes_field(variable, field, value)


def _copy_template(template_path: str | Path, output_path: str | Path) -> tuple[Path, Path]:
    template = Path(template_path)
    output = Path(output_path)
    if not template.is_file():
        raise FileNotFoundError(template)
    if template.resolve() == output.resolve():
        raise ValueError("output_path must differ from template_path")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template, output)
    return template, output


def create_stomate_restart_skeleton_from_schema(
    output_path: str | Path,
    physical_state: StomateRestartPhysicalState,
    *,
    schema_path: str | Path = DEFAULT_STOMATE_RESTART_SCHEMA,
    pft_layout: PFTRunLayout | None = None,
) -> Path:
    """Create the complete paper-protocol NetCDF structure without a template.

    The schema is an audited extraction of the IOIPSL file definition used by
    the Fortran paper run. Scientific variables are left at NetCDF fill values
    and must subsequently be populated by the strict writerestart owner.

    Provenance: ``stomate_io.f90::writerestart`` lines 1751-2944 and the
    IOIPSL ``restini/restput_p`` physical-file boundary recorded in the schema.
    """

    if not isinstance(physical_state, StomateRestartPhysicalState):
        raise TypeError("physical_state must be a StomateRestartPhysicalState")
    if tuple(physical_state.index_g) != (1,):
        raise ValueError(
            "paper single-landpoint restart requires Fortran one-based index_g=(1,)"
        )
    schema_file = Path(schema_path)
    if not schema_file.is_file():
        raise FileNotFoundError(schema_file)
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    if schema.get("schema_version") != 1:
        raise ValueError("unsupported STOMATE restart schema version")
    scatter = schema.get("single_landpoint_scatter", {})
    if scatter.get("index_g") != [1] or scatter.get("grid_shape") != [1, 1]:
        raise ValueError("restart schema does not declare the paper identity scatter")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with Dataset(output, "w", format=str(schema["file_format"])) as dataset:
        if pft_layout is not None:
            missing_pft_dimensions = PAPER_RESTART_PFT_DIMENSIONS - set(schema["dimensions"])
            if missing_pft_dimensions:
                raise ValueError(
                    "restart schema is missing source PFT dimensions: "
                    + ", ".join(sorted(missing_pft_dimensions))
                )
        for name, declaration in schema["dimensions"].items():
            if name in PAPER_RESTART_PFT_DIMENSIONS and pft_layout is not None:
                if declaration["unlimited"]:
                    raise ValueError(f"restart PFT dimension {name} cannot be unlimited")
                size = pft_layout.n_pft
            else:
                size = None if declaration["unlimited"] else int(declaration["size"])
            dataset.createDimension(name, size)

        global_attributes = dict(schema.get("global_attributes", {}))
        global_attributes["file_name"] = output.name
        if physical_state.global_attributes is not None:
            global_attributes.update(dict(physical_state.global_attributes))
        if pft_layout is not None:
            global_attributes.update(pft_layout_netcdf_attributes(pft_layout))
        dataset.setncatts(global_attributes)

        for name, declaration in schema["variables"].items():
            create_kwargs: dict[str, object] = {}
            if "fill_value" in declaration:
                create_kwargs["fill_value"] = declaration["fill_value"]
            if declaration.get("chunking") == "contiguous":
                create_kwargs["contiguous"] = True
            endian = declaration.get("endian")
            if endian in {"little", "big", "native"}:
                create_kwargs["endian"] = endian
            variable = dataset.createVariable(
                name,
                declaration["dtype"],
                tuple(declaration["dimensions"]),
                **create_kwargs,
            )
            variable.setncatts(dict(declaration.get("attributes", {})))

        physical_values = {
            "nav_lon": physical_state.nav_lon,
            "nav_lat": physical_state.nav_lat,
            "nav_lev": physical_state.nav_lev,
            "time": physical_state.time,
            "time_steps": physical_state.time_steps,
        }
        expected_physical = set(schema["physical_variables"])
        if set(physical_values) != expected_physical:
            raise ValueError("restart physical variable contract is inconsistent")
        for name, value in physical_values.items():
            variable = dataset.variables[name]
            array = np.asarray(value)
            if array.shape != variable.shape:
                raise _shape_error(name, tuple(variable.shape), array.shape)
            variable[:] = array
    return output


def _require_variables(dataset: Dataset, names: set[str]) -> None:
    missing = tuple(sorted(names - dataset.variables.keys()))
    if missing:
        raise KeyError(f"restart template is missing required variables: {missing}")


def _state_field_names(state: object) -> frozenset[str]:
    return frozenset(name for name in state._fields if name != "provenance")


def _require_state_mapping(state: object, expected: frozenset[str], contract: str) -> None:
    actual = _state_field_names(state)
    unsupported = tuple(sorted(actual - expected))
    missing = tuple(sorted(expected - actual))
    if unsupported or missing:
        raise ValueError(
            f"{contract} restart field mapping is incomplete: unsupported={unsupported}, missing={missing}"
        )


def _validate_daily_runtime_fields(state: StomateDailyAccumulatorState) -> None:
    for field in sorted(_DAILY_RUNTIME_FIELDS):
        value = np.asarray(getattr(state, field))
        if not np.array_equal(value, np.zeros_like(value)):
            raise ValueError(
                f"runtime-initialized field {field} has no restart variable and must be exactly zero"
            )


def _write_season_dataset(dataset: Dataset, state: StomateRestartSeasonState) -> None:
    _require_variables(
        dataset,
        set(_SEASON_SCALAR_FIELDS.values())
        | set(_SEASON_PFT_FIELDS.values())
        | set(_SEASON_GRID_FIELDS.values())
        | set(_SEASON_LAYER_FIELDS.values()),
    )
    for field, name in _SEASON_SCALAR_FIELDS.items():
        _write_scalar(dataset.variables[name], field, getattr(state, field))
    for field, name in _SEASON_PFT_FIELDS.items():
        _write_pft(dataset.variables[name], field, getattr(state, field))
    for field, name in _SEASON_GRID_FIELDS.items():
        _write_grid(dataset.variables[name], field, getattr(state, field))
    for field, name in _SEASON_LAYER_FIELDS.items():
        _write_layer(dataset.variables[name], field, getattr(state, field))


def _write_daily_dataset(dataset: Dataset, state: StomateDailyAccumulatorState) -> None:
    _require_variables(
        dataset,
        set(_DAILY_PFT_FIELDS.values())
        | set(_DAILY_GRID_FIELDS.values())
        | set(_DAILY_LAYER_FIELDS.values()),
    )
    for field, name in _DAILY_PFT_FIELDS.items():
        _write_pft(dataset.variables[name], field, getattr(state, field))
    for field, name in _DAILY_GRID_FIELDS.items():
        _write_grid(dataset.variables[name], field, getattr(state, field))
    for field, name in _DAILY_LAYER_FIELDS.items():
        _write_layer(dataset.variables[name], field, getattr(state, field))


def _validate_derived(state: StomateRestartEntryState) -> None:
    checks = {
        "prod10_total": np.sum(np.asarray(state.prod10), axis=(1, 2)),
        "prod100_total": np.sum(np.asarray(state.prod100), axis=(1, 2)),
        "soilc_total": np.asarray(state.deepC_a) + np.asarray(state.deepC_s) + np.asarray(state.deepC_p),
    }
    for field, expected in checks.items():
        actual = np.asarray(getattr(state, field))
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"derived restart field {field} cannot be written independently and is inconsistent "
                "with its source fields"
            )


def _require_complete_mapping(state: StomateRestartEntryState) -> None:
    state_fields = frozenset(state._fields)
    unsupported = tuple(sorted(state_fields - _DIRECT_FIELDS - _DERIVED_FIELDS))
    missing = tuple(sorted((_DIRECT_FIELDS | _DERIVED_FIELDS) - state_fields))
    if unsupported or missing:
        raise ValueError(f"restart field mapping is incomplete: unsupported={unsupported}, missing={missing}")


def write_stomate_restart_entry_state_from_template(
    template_path: str | Path,
    output_path: str | Path,
    state: StomateRestartEntryState,
) -> StomateRestartWriteReport:
    """Copy a real restart template and strictly serialize normalized state.

    Every field exposed by :func:`read_stomate_restart_entry_state` is either
    written through an exact inverse axis transform or verified as an exact
    derivation of written fields. Missing variables and unknown state fields
    are errors. The output remains template-backed and is not a standalone
    implementation of Fortran ``writerestart``.
    """

    if not isinstance(state, StomateRestartEntryState):
        raise TypeError("state must be a StomateRestartEntryState")
    _require_complete_mapping(state)
    _validate_derived(state)

    template = Path(template_path)
    output = Path(output_path)
    if not template.is_file():
        raise FileNotFoundError(template)
    if template.resolve() == output.resolve():
        raise ValueError("output_path must differ from template_path")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template, output)

    with Dataset(output, "r+") as dataset:
        required_variables = {
            *_PFT_FIELDS.values(),
            *_GRID_FIELDS.values(),
            *_PFT_POOL_FIELDS.values(),
            "maint_resp",
            "leaf_age",
            "leaf_frac",
            "litterpart",
            "dead_leaves",
            "carbon",
            "litter",
            "lignin_struc",
            "fuel_1hr",
            "fuel_10hr",
            "fuel_100hr",
            "fuel_1000hr",
            "prod10",
            "prod100",
            "flux10",
            "flux100",
            "assim_param",
            "litter_above",
            "litter_below",
            "carbon_32l_a",
            "carbon_32l_s",
            "carbon_32l_p",
            "freedoc",
            "adsdoc",
            "interception_storage",
            "lig_struc_be",
            "deepC_a",
            "deepC_s",
            "deepC_p",
        }
        missing_variables = tuple(sorted(required_variables - dataset.variables.keys()))
        if missing_variables:
            raise KeyError(f"restart template is missing required variables: {missing_variables}")

        for field, name in _PFT_FIELDS.items():
            _write_pft(dataset.variables[name], field, getattr(state, field))
        for field, name in _GRID_FIELDS.items():
            _write_grid(dataset.variables[name], field, getattr(state, field))
        for field, name in _PFT_POOL_FIELDS.items():
            _write_pft_pool(dataset.variables[name], field, getattr(state, field))

        _write_maint_resp(dataset.variables["maint_resp"], "resp_maint_part", state.resp_maint_part)
        for field in ("leaf_age", "leaf_frac", "litterpart", "dead_leaves", "lignin_struc"):
            _write_component_pft(dataset.variables[field], field, getattr(state, field))
        _write_legacy_carbon(dataset.variables["carbon"], "carbon", state.carbon)
        _write_litter(dataset.variables["litter"], "litter", state.litter)
        for field in ("fuel_1hr", "fuel_10hr", "fuel_100hr", "fuel_1000hr"):
            _write_fuel(dataset.variables[field], field, getattr(state, field))
        for field in ("prod10", "prod100", "flux10", "flux100"):
            _write_product(dataset.variables[field], field, getattr(state, field))
        _write_component_pft(dataset.variables["assim_param"], "assim_param", state.assim_param)
        _write_litter_above(dataset.variables["litter_above"], "litter_above", state.litter_above)
        _write_litter_below(dataset.variables["litter_below"], "litter_below", state.litter_below)
        _write_carbon_32l(dataset, "carbon_32l", state.carbon_32l)
        _write_doc(dataset, "DOC", state.DOC)
        _write_interception(
            dataset.variables["interception_storage"],
            "interception_storage",
            state.interception_storage,
        )
        _write_component_pft(
            dataset.variables["lig_struc_be"],
            "lignin_struc_below",
            state.lignin_struc_below,
        )
        for field in ("deepC_a", "deepC_s", "deepC_p"):
            _write_deep_carbon(dataset.variables[field], field, getattr(state, field))

    return StomateRestartWriteReport(
        output_path=output,
        written_fields=tuple(sorted(_DIRECT_FIELDS)),
        validated_derived_fields=tuple(sorted(_DERIVED_FIELDS)),
        unsupported_fields=(),
    )


def write_stomate_restart_season_state_from_template(
    template_path: str | Path,
    output_path: str | Path,
    state: StomateRestartSeasonState,
) -> StomateRestartWriteReport:
    """Copy a Fortran template and serialize all season-reader fields."""

    if not isinstance(state, StomateRestartSeasonState):
        raise TypeError("state must be a StomateRestartSeasonState")
    _require_state_mapping(state, _SEASON_FIELDS, "season")
    _, output = _copy_template(template_path, output_path)
    with Dataset(output, "r+") as dataset:
        _write_season_dataset(dataset, state)
    return StomateRestartWriteReport(
        output_path=output,
        written_fields=tuple(sorted(_SEASON_FIELDS)),
        validated_derived_fields=(),
        unsupported_fields=(),
    )


def write_stomate_daily_accumulator_state_from_template(
    template_path: str | Path,
    output_path: str | Path,
    state: StomateDailyAccumulatorState,
) -> StomateRestartWriteReport:
    """Copy a Fortran template and serialize all restart-backed daily fields.

    ``snowfall_daily``, ``snowmass_daily``, and ``tmc_topgrass_daily`` are
    initialized to zero by STOMATE rather than read from NetCDF. They are
    therefore required to be exactly zero and are reported as validated, not
    written.
    """

    if not isinstance(state, StomateDailyAccumulatorState):
        raise TypeError("state must be a StomateDailyAccumulatorState")
    _require_state_mapping(state, _DAILY_FIELDS, "daily accumulator")
    _validate_daily_runtime_fields(state)
    _, output = _copy_template(template_path, output_path)
    with Dataset(output, "r+") as dataset:
        _write_daily_dataset(dataset, state)
    return StomateRestartWriteReport(
        output_path=output,
        written_fields=tuple(sorted(_DAILY_DIRECT_FIELDS)),
        validated_derived_fields=tuple(sorted(_DAILY_RUNTIME_FIELDS)),
        unsupported_fields=(),
    )


def write_stomate_restart_states_from_template(
    template_path: str | Path,
    output_path: str | Path,
    *,
    entry_state: StomateRestartEntryState,
    season_state: StomateRestartSeasonState,
    daily_state: StomateDailyAccumulatorState,
) -> StomateRestartWriteReport:
    """Write all three public STOMATE restart-reader contracts to one copy.

    The shared ``gpp_daily`` field must agree between entry and daily states.
    All writes retain the same template dependency and do not constitute an
    independent implementation of Fortran ``writerestart``.
    """

    if not isinstance(entry_state, StomateRestartEntryState):
        raise TypeError("entry_state must be a StomateRestartEntryState")
    if not isinstance(season_state, StomateRestartSeasonState):
        raise TypeError("season_state must be a StomateRestartSeasonState")
    if not isinstance(daily_state, StomateDailyAccumulatorState):
        raise TypeError("daily_state must be a StomateDailyAccumulatorState")
    _require_complete_mapping(entry_state)
    _validate_derived(entry_state)
    _require_state_mapping(season_state, _SEASON_FIELDS, "season")
    _require_state_mapping(daily_state, _DAILY_FIELDS, "daily accumulator")
    _validate_daily_runtime_fields(daily_state)
    if not np.array_equal(np.asarray(entry_state.gpp_daily), np.asarray(daily_state.gpp_daily)):
        raise ValueError("shared restart field gpp_daily differs between entry_state and daily_state")

    # The entry writer creates the sole template copy and writes its complete
    # contract. Season and daily state are then added to that same file.
    entry_report = write_stomate_restart_entry_state_from_template(
        template_path,
        output_path,
        entry_state,
    )
    output = entry_report.output_path
    with Dataset(output, "r+") as dataset:
        _write_season_dataset(dataset, season_state)
        _write_daily_dataset(dataset, daily_state)

    written = _DIRECT_FIELDS | _SEASON_FIELDS | _DAILY_DIRECT_FIELDS
    validated = _DERIVED_FIELDS | _DAILY_RUNTIME_FIELDS
    return StomateRestartWriteReport(
        output_path=output,
        written_fields=tuple(sorted(written)),
        validated_derived_fields=tuple(sorted(validated)),
        unsupported_fields=(),
    )


def write_stomate_full_writerestart_states_from_template(
    template_path: str | Path,
    output_path: str | Path,
    *,
    entry_state: StomateRestartEntryState,
    season_state: StomateRestartSeasonState,
    daily_state: StomateDailyAccumulatorState,
    gas_state: StomateOkPcRestartGasState,
    remainder_state: StomateReadstartRemainderState,
) -> StomateRestartWriteReport:
    """Serialize every normalized state written by Fortran ``writerestart``.

    Scientific state selection, logical-to-real encoding, aliases, and inverse
    Fortran axis order are owned here. The output still starts from an existing
    NetCDF template: file definition, IOIPSL scatter semantics, metadata, and
    parallel side effects remain explicit external boundaries.

    Provenance: ``stomate_io.f90::writerestart`` lines 1751-2944.
    """

    _require_state_mapping(gas_state, _GAS_WRITE_FIELDS, "OK_PC gas")
    _require_state_mapping(
        remainder_state,
        _REMAINDER_WRITE_FIELDS | _REMAINDER_SOURCE_LOCAL_FIELDS,
        "readstart remainder",
    )
    undefined = tuple(
        sorted(field for field in _REMAINDER_WRITE_FIELDS if getattr(remainder_state, field) is None)
    )
    if undefined:
        raise ValueError(f"writerestart scientific state is undefined for fields: {undefined}")

    litter = np.asarray(entry_state.litter)
    if litter.ndim != 5:
        raise ValueError(
            "litter must have normalized shape (npts,nlitt,nvm,nlevs,nelements) "
            f"to establish writerestart labels, got {litter.shape}"
        )
    writerestart_index_labels(
        nlitt=litter.shape[1],
        nlevs=litter.shape[3],
        nelements=litter.shape[4],
    )

    base_report = write_stomate_restart_states_from_template(
        template_path,
        output_path,
        entry_state=entry_state,
        season_state=season_state,
        daily_state=daily_state,
    )
    with Dataset(base_report.output_path, "r+") as dataset:
        _write_gas_dataset(dataset, gas_state)
        _write_remainder_dataset(dataset, remainder_state)

    return StomateRestartWriteReport(
        output_path=base_report.output_path,
        written_fields=tuple(sorted(_FULL_WRITERESTART_SERIALIZED_FIELDS)),
        validated_derived_fields=tuple(sorted(_DERIVED_FIELDS | _DAILY_RUNTIME_FIELDS)),
        unsupported_fields=(),
    )


def write_stomate_full_writerestart_states(
    output_path: str | Path,
    *,
    physical_state: StomateRestartPhysicalState,
    entry_state: StomateRestartEntryState,
    season_state: StomateRestartSeasonState,
    daily_state: StomateDailyAccumulatorState,
    gas_state: StomateOkPcRestartGasState,
    remainder_state: StomateReadstartRemainderState,
    schema_path: str | Path = DEFAULT_STOMATE_RESTART_SCHEMA,
    pft_layout: PFTRunLayout | None = None,
) -> StomateRestartWriteReport:
    """Construct and populate a paper-protocol STOMATE restart independently.

    No Fortran restart file is read or copied at runtime. The temporary blank
    file is defined from the audited schema, populated with the physical
    coordinate/time variables, and then passed through the same strict inverse
    state transforms as the template-backed compatibility path.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(
        prefix="stomate_restart_schema_", dir=output.parent
    ) as temporary_directory:
        skeleton = Path(temporary_directory) / "stomate_skeleton.nc"
        create_stomate_restart_skeleton_from_schema(
            skeleton,
            physical_state,
            schema_path=schema_path,
            pft_layout=pft_layout,
        )
        return write_stomate_full_writerestart_states_from_template(
            skeleton,
            output,
            entry_state=entry_state,
            season_state=season_state,
            daily_state=daily_state,
            gas_state=gas_state,
            remainder_state=remainder_state,
        )


def stomate_writerestart_field_ledger() -> StomateWriterestartFieldLedger:
    """Return complete state coverage for the paper single-landpoint writer."""

    return StomateWriterestartFieldLedger(
        serialized_fields=tuple(sorted(_FULL_WRITERESTART_SERIALIZED_FIELDS)),
        validated_not_serialized_fields=tuple(sorted(_DERIVED_FIELDS | _DAILY_RUNTIME_FIELDS)),
        source_local_not_restart_fields=tuple(sorted(_REMAINDER_SOURCE_LOCAL_FIELDS)),
        independent_netcdf_boundaries=(),
    )


def template_backed_restart_field_names() -> tuple[str, ...]:
    """Return every reader field covered by strict write or validation."""

    return tuple(
        sorted(
            _DIRECT_FIELDS
            | _DERIVED_FIELDS
            | _SEASON_FIELDS
            | _DAILY_DIRECT_FIELDS
            | _DAILY_RUNTIME_FIELDS
        )
    )


def template_backed_restart_written_field_names() -> tuple[str, ...]:
    """Return reader fields serialized into template NetCDF variables."""

    return tuple(sorted(_DIRECT_FIELDS | _SEASON_FIELDS | _DAILY_DIRECT_FIELDS))


def template_backed_restart_validated_field_names() -> tuple[str, ...]:
    """Return derived/runtime fields validated rather than independently written."""

    return tuple(sorted(_DERIVED_FIELDS | _DAILY_RUNTIME_FIELDS))
