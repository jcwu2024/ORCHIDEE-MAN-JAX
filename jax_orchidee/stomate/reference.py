"""Reference STOMATE restart/history readers for Phase 1C.

These helpers are validation-boundary utilities. They read local NetCDF
reference files, report variable metadata, and normalize axes for already
audited kernels; they do not infer parameters or implement STOMATE process
logic.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile
from typing import Mapping, NamedTuple

import numpy as np
from netCDF4 import Dataset

from jax_orchidee.driver.reference_layout import LEGACY_LANDPOINT_ID, resolve_paper_landpoint_reference
from jax_orchidee.stomate.carbon_kernels import (
    ICARBON,
    IAGRSAPPN,
    IAGRSAPST,
    IAGRHRTPN,
    IAGRHRTST,
    ICARBRES,
    IFRUIT,
    IHEARTABOVE,
    IHEARTBELOW,
    ILEAF,
    IROOT,
    ISAPABOVE,
    ISAPBELOW,
    NCARB,
    NLITT,
    NLEAFAGES,
    NPOOL,
    NPARTS,
)
from jax_orchidee.stomate.modelout import MODEL_OUTPUT_FIELD_NAMES, PAPER_MODEL_PFT14_INDEX, history_year_for_age
from jax_orchidee.stomate.soilcarbon_kernels import NDOC


REFERENCE_RUN_PATTERN = (
    "reference/case_001_071/OUT/orc_calibrate_250919_sen/arg2_1.0/"
    "001.0-071.0/I10/S2_63.206_0.0876_0.2019_50.658"
)

RESTART_STATE_FIELDS = (
    "biomass",
    "maint_resp",
    "resp_maint",
    "resp_growth",
    "gpp_daily",
    "npp_daily",
    "PFTpresent",
    "sla_calc",
    "t2m_daily",
    "tsoil_daily",
    "t2m_longterm",
    "moiavail_week",
    "tsoil_month",
    "soilhum_month",
    "leaf_age",
    "leaf_frac",
    "age",
)

RESTART_ENTRY_STATE_FIELDS = (
    "biomass",
    "maint_resp",
    "gpp_daily",
    "npp_daily",
    "turnover_daily",
    "resp_maint",
    "resp_growth",
    "leaf_age",
    "leaf_frac",
    "age",
    "sla_calc",
    "PFTpresent",
    "ind",
    "adapted",
    "regenerate",
    "npp_longterm",
    "lm_lastyearmax",
    "turnover_time",
    "turnover_longterm",
    "senescence",
    "when_growthinit",
    "co2_to_bm_dgvm",
    "everywhere",
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
    "bm_to_litter",
    "carb_mass_total",
    "RIP_time",
    "assim_param",
    "altmax",
    "fixed_cryoturb_depth",
    "fpeat",
    "litter_above",
    "litter_below",
    "lignin_struc_above",
    "lig_struc_be",
    "carbon_32l_a",
    "carbon_32l_s",
    "carbon_32l_p",
    "freedoc",
    "adsdoc",
    "interception_storage",
    "deepC_a",
    "deepC_s",
    "deepC_p",
    "thawed_humidity",
    "depth_organic_soil",
)

OK_PC_RESTART_GAS_FIELDS = (
    "O2_soil",
    "CH4_soil",
    "O2_snow",
    "CH4_snow",
)

HISTORY_VALIDATION_FIELDS = MODEL_OUTPUT_FIELD_NAMES + (
    "LAI",
    "AGE",
    "VEGET_MAX",
    "MAINT_RESP",
    "GROWTH_RESP",
    "MAINT_RESP_AGRSAPST",
    "MAINT_RESP_AGRSAPPN",
    "MAINT_RESP_AGRHRTST",
    "MAINT_RESP_AGRHRTPN",
    "BM_ALLOC_LEAF",
    "BM_ALLOC_SAP_AB",
    "BM_ALLOC_SAP_BE",
    "BM_ALLOC_ROOT",
    "BM_ALLOC_FRUIT",
    "BM_ALLOC_RES",
    "NPP_ABOVE",
    "NPP_BELOW",
)

HISTORY_POOL_TO_INDEX = {
    "LEAF_M": ILEAF,
    "SAP_M_AB": ISAPABOVE,
    "SAP_M_BE": ISAPBELOW,
    "HEART_M_AB": IHEARTABOVE,
    "HEART_M_BE": IHEARTBELOW,
    "ROOT_M": IROOT,
    "FRUIT_M": IFRUIT,
    "RESERVE_M": ICARBRES,
    "AGR_SAP_ST_M": IAGRSAPST,
    "AGR_SAP_PN_M": IAGRSAPPN,
    "AGR_HRT_ST_M": IAGRHRTST,
    "AGR_HRT_PN_M": IAGRHRTPN,
}


@dataclass(frozen=True)
class NetCDFVariableInfo:
    """Lightweight metadata for one NetCDF variable."""

    name: str
    dimensions: tuple[str, ...]
    shape: tuple[int, ...]
    units: str | None = None
    long_name: str | None = None


@dataclass(frozen=True)
class StomateReferenceFiles:
    """Local reference STOMATE files for the paper-case run."""

    run_dir: Path
    start: Path
    restart: Path
    histories: tuple[Path, ...]


@dataclass(frozen=True)
class PaperModeloutCsvRow:
    """One paper modelout CSV target row.

    Provenance: ``fortran_run_scripts/paper_250919/``
    ``c2.4_Model_run_functions_sensitivity.py`` lines 48-76 and 115-118 read
    ``stomate_history_(1961 + age - 1).nc`` at ``[0, 13, 0, 0]`` and write the
    ``AGB_model``, ``BGB_model``, ``GPP_model``, and ``NPP_model`` columns.
    """

    i_grid: str
    age: int
    target_year: int
    AGB_model: float
    BGB_model: float
    GPP_model: float
    NPP_model: float

    @property
    def values(self) -> dict[str, float]:
        return {
            "AGB_model": self.AGB_model,
            "BGB_model": self.BGB_model,
            "GPP_model": self.GPP_model,
            "NPP_model": self.NPP_model,
        }


class RestartCarbonReadiness(NamedTuple):
    """Restart state arrays normalized for explicit carbon adapter readiness."""

    biomass: np.ndarray
    maint_resp_part: np.ndarray
    gpp_daily: np.ndarray
    npp_daily: np.ndarray
    resp_maint: np.ndarray
    resp_growth: np.ndarray
    pft_present: np.ndarray


class ModeloutHistoryComparison(NamedTuple):
    """Field-wise comparison between a JAX daily modelout and history truth.

    Reference/history validation boundary only. The comparison reads the
    archived Fortran history fields at the paper PFT14 point and compares them
    to already-produced JAX modelout fields; it does not compute or infer any
    STOMATE process state.
    """

    history_path: Path
    fields: tuple[str, ...]
    jax_values: dict[str, float]
    history_values: dict[str, float]
    abs_differences: dict[str, float]
    max_abs_difference: float

    @property
    def ok(self) -> bool:
        return self.max_abs_difference == 0.0


class StomateDailyAccumulatorState(NamedTuple):
    """Restart-backed daily accumulators used by ``stomate_main`` section 4."""

    humrel_daily: np.ndarray
    litterhum_daily: np.ndarray
    t2m_daily: np.ndarray
    t2m_min_daily: np.ndarray
    t2m_max_daily: np.ndarray
    wspeed_daily: np.ndarray
    tsurf_daily: np.ndarray
    tsoil_daily: np.ndarray
    soilhum_daily: np.ndarray
    precip_daily: np.ndarray
    gpp_daily: np.ndarray
    snowfall_daily: np.ndarray
    snowmass_daily: np.ndarray
    tmc_topgrass_daily: np.ndarray
    fwet_daily: np.ndarray
    liqwt_daily: np.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart reads daily restart accumulators",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3187-3240 consumes these accumulators",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 8181-8321 allocates and initializes snowfall_daily, snowmass_daily, and tmc_topgrass_daily to zero",
    )


class StomateRestartEntryState(NamedTuple):
    """Restart fields normalized to Fortran ``stomate_main`` axes."""

    biomass: np.ndarray
    resp_maint_part: np.ndarray
    gpp_daily: np.ndarray
    npp_daily: np.ndarray
    turnover_daily: np.ndarray
    resp_maint: np.ndarray
    resp_growth: np.ndarray
    leaf_age: np.ndarray
    leaf_frac: np.ndarray
    age: np.ndarray
    sla_calc: np.ndarray
    pft_present: np.ndarray
    ind: np.ndarray
    adapted: np.ndarray
    regenerate: np.ndarray
    npp_longterm: np.ndarray
    lm_lastyearmax: np.ndarray
    turnover_time: np.ndarray
    turnover_longterm: np.ndarray
    senescence: np.ndarray
    when_growthinit: np.ndarray
    co2_to_bm: np.ndarray
    veget_lastlight: np.ndarray
    everywhere: np.ndarray
    need_adjacent: np.ndarray
    litterpart: np.ndarray
    dead_leaves: np.ndarray
    carbon: np.ndarray
    litter: np.ndarray
    lignin_struc: np.ndarray
    fuel_1hr: np.ndarray
    fuel_10hr: np.ndarray
    fuel_100hr: np.ndarray
    fuel_1000hr: np.ndarray
    prod10: np.ndarray
    prod100: np.ndarray
    flux10: np.ndarray
    flux100: np.ndarray
    prod10_total: np.ndarray
    prod100_total: np.ndarray
    bm_to_litter: np.ndarray
    carb_mass_total: np.ndarray
    rip_time: np.ndarray
    assim_param: np.ndarray
    altmax: np.ndarray
    fixed_cryoturbation_depth: np.ndarray
    fpeat: np.ndarray
    litter_above: np.ndarray
    litter_below: np.ndarray
    carbon_32l: np.ndarray
    DOC: np.ndarray
    interception_storage: np.ndarray
    lignin_struc_above: np.ndarray
    lignin_struc_below: np.ndarray
    deepC_a: np.ndarray
    deepC_s: np.ndarray
    deepC_p: np.ndarray
    soilc_total: np.ndarray
    thawed_humidity: np.ndarray
    depth_organic_soil: np.ndarray


class StomateOkPcRestartGasState(NamedTuple):
    """OK_PC gas restart fields normalized to ``deep_carbcycle`` axes."""

    O2_soil: np.ndarray
    CH4_soil: np.ndarray
    O2_snow: np.ndarray
    CH4_snow: np.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 1177-1199",
        "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90::deep_carbcycle lines 186-235",
    )


class StomateReadstartRemainderState(NamedTuple):
    """Remaining ``readstart`` outputs normalized to their Fortran axes.

    Array axes follow the declarations in ``stomate_io.f90::readstart`` lines
    175-342.  ``None`` is intentional for the three paths whose Fortran local
    or destination is not initialized when no fallback assignment executes.
    """

    fireindex: np.ndarray
    firelitter: np.ndarray
    resp_hetero: np.ndarray
    co2_fire: np.ndarray
    ni_acc: np.ndarray
    read_input_thawed_humidity: bool | None
    read_input_depth_organic_soil: bool | None
    uo_0: np.ndarray
    uold2_0: np.ndarray
    uo_wet1: np.ndarray
    uold2_wet1: np.ndarray
    uo_wet2: np.ndarray
    uold2_wet2: np.ndarray
    uo_wet3: np.ndarray
    uold2_wet3: np.ndarray
    uo_wet4: np.ndarray
    uold2_wet4: np.ndarray
    tsurf_year: np.ndarray
    height_acro: np.ndarray
    carbon_acro: np.ndarray
    carbon_cato: np.ndarray
    fwet_month: np.ndarray
    liqwt_month: np.ndarray
    liqwt_max: np.ndarray
    fwet_series: np.ndarray
    deepC_peat: np.ndarray
    carbon_save: np.ndarray
    deepC_a_save: np.ndarray
    deepC_s_save: np.ndarray
    deepC_p_save: np.ndarray
    delta_fsave: np.ndarray
    depth_deepsoil: np.ndarray | None
    wshtotsum: np.ndarray
    sr_ugb: np.ndarray
    nb_ani: np.ndarray
    grazed_frac: np.ndarray
    import_yield: np.ndarray
    t2m_14: np.ndarray
    litter_not_avail: np.ndarray
    nb_grazingdays: np.ndarray
    after_snow: np.ndarray
    after_wet: np.ndarray
    wet1day: np.ndarray
    wet2day: np.ndarray
    Global_years: int
    nbp_sum: np.ndarray
    nbp_flux: np.ndarray
    ok_equilibrium: np.ndarray
    MatrixV: np.ndarray
    Vector_U: np.ndarray
    previous_stock: np.ndarray
    current_stock: np.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 707-717, 979-1001, and 1075-1079",
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 1201-1225 and 1245-1509",
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 1535-1538 and 1587-1732",
    )


class StomateRestartSeasonState(NamedTuple):
    """Restart-backed STOMATE season memory normalized to Fortran axes."""

    dt_days_read: float
    date: int
    tau_longterm: float
    moiavail_month: np.ndarray
    moiavail_week: np.ndarray
    t2m_longterm: np.ndarray
    t2m_month: np.ndarray
    t2m_week: np.ndarray
    tsoil_month: np.ndarray
    soilhum_month: np.ndarray
    gdd_m5_dormance: np.ndarray
    gdd_from_growthinit: np.ndarray
    gdd_midwinter: np.ndarray
    ncd_dormance: np.ndarray
    ngd_minus5: np.ndarray
    gdd_init_date: np.ndarray
    time_hum_min: np.ndarray
    hum_min_dormance: np.ndarray
    tseason: np.ndarray
    tseason_length: np.ndarray
    tseason_tmp: np.ndarray
    tmin_spring_time: np.ndarray
    begin_leaves: np.ndarray
    onset_date: np.ndarray
    gpp_week: np.ndarray
    maxmoiavail_lastyear: np.ndarray
    maxmoiavail_thisyear: np.ndarray
    minmoiavail_lastyear: np.ndarray
    minmoiavail_thisyear: np.ndarray
    maxgppweek_lastyear: np.ndarray
    maxgppweek_thisyear: np.ndarray
    gdd0_lastyear: np.ndarray
    gdd0_thisyear: np.ndarray
    precip_lastyear: np.ndarray
    precip_thisyear: np.ndarray
    maxfpc_lastyear: np.ndarray
    maxfpc_thisyear: np.ndarray
    lm_thisyearmax: np.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 501-505 reads dt_days/date restart scalars",
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 613-637 reads t2m_longterm and tau_longterm",
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 600-826 reads season climate/memory restart state",
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 719-792 reads yearly moisture/GPP/GDD0/precipitation season statistics",
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 516-520 and 796-839 reads GDD and humidity dormancy memory",
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 647-670 reads Tseason and Tmin_spring_time",
        "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::writerestart lines 2419-2567 writes the same season state variables",
        "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90::season uses these fields before phenology/allocation/turnover",
    )


STOMATE_COLD_START_READSTART_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 501-637 handles dt_days/date, daily accumulators, t2m_longterm, and tau_longterm fallbacks",
    "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 641-839 handles season-memory val_exp fallbacks",
    "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 843-1041 handles plant status, productivity, biomass, respiration, phenology, and RIP_time fallbacks",
    "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90::readstart lines 1063-1163 and 1509-1569 handles litter, wood-product, deep carbon, litter_above/below, carbon_32l, DOC, and TF-DOC fallbacks",
)


def _shape_1d(npts: int) -> tuple[int]:
    return (int(npts),)


def _shape_pft(npts: int, nvm: int) -> tuple[int, int]:
    return (int(npts), int(nvm))


def _zeros(shape, *, dtype=np.float64) -> np.ndarray:
    return np.zeros(tuple(int(axis) for axis in shape), dtype=dtype)


def _full(shape, value, *, dtype=np.float64) -> np.ndarray:
    return np.full(tuple(int(axis) for axis in shape), value, dtype=dtype)


def _tile_grid_to_layers(value, nslm: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("t2m must have shape (npts,)")
    return np.repeat(arr[:, None], int(nslm), axis=1)


def stomate_cold_start_daily_accumulator_state(
    *,
    t2m,
    nvm: int,
    nslm: int,
    large_value: float = 1.0e33,
) -> StomateDailyAccumulatorState:
    """Return no-restart daily accumulator fallbacks from ``readstart``.

    This is not a daily process update. It only mirrors the values assigned
    when ``restget_p`` leaves each field at ``val_exp`` during a cold start.
    """

    t2m = np.asarray(t2m, dtype=np.float64)
    if t2m.ndim != 1:
        raise ValueError("t2m must have shape (npts,)")
    npts = t2m.shape[0]
    return StomateDailyAccumulatorState(
        humrel_daily=_zeros((npts, nvm)),
        litterhum_daily=_zeros((npts,)),
        t2m_daily=_zeros((npts,)),
        t2m_min_daily=_full((npts,), large_value),
        t2m_max_daily=_full((npts,), -large_value),
        wspeed_daily=_full((npts,), large_value),
        tsurf_daily=t2m.copy(),
        tsoil_daily=_zeros((npts, nslm)),
        soilhum_daily=_zeros((npts, nslm)),
        precip_daily=_zeros((npts,)),
        gpp_daily=_zeros((npts, nvm)),
        snowfall_daily=_zeros((npts,)),
        snowmass_daily=_zeros((npts,)),
        tmc_topgrass_daily=_zeros((npts,)),
        fwet_daily=_zeros((npts,)),
        liqwt_daily=_zeros((npts,)),
        provenance=STOMATE_COLD_START_READSTART_PROVENANCE
        + (
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 8181-8321 allocates snowfall_daily, snowmass_daily, and tmc_topgrass_daily to zero",
        ),
    )


def stomate_cold_start_season_state(
    *,
    t2m,
    dt_days: float,
    nvm: int,
    nslm: int,
    date: int = 0,
    large_value: float = 1.0e33,
    undef: float = -9999.0,
    gdd_crit_estab: float = 150.0,
    precip_crit: float = 100.0,
) -> StomateRestartSeasonState:
    """Return no-restart season-memory fallbacks from ``readstart``."""

    t2m = np.asarray(t2m, dtype=np.float64)
    if t2m.ndim != 1:
        raise ValueError("t2m must have shape (npts,)")
    npts = t2m.shape[0]
    gdd_init_date = _zeros((npts, 2))
    gdd_init_date[:, 0] = 365.0
    return StomateRestartSeasonState(
        dt_days_read=float(dt_days),
        date=int(date),
        tau_longterm=2.0,
        moiavail_month=_zeros((npts, nvm)),
        moiavail_week=_zeros((npts, nvm)),
        t2m_longterm=t2m.copy(),
        t2m_month=t2m.copy(),
        t2m_week=t2m.copy(),
        tsoil_month=_tile_grid_to_layers(t2m, nslm),
        soilhum_month=_zeros((npts, nslm)),
        gdd_m5_dormance=_full((npts, nvm), undef),
        gdd_from_growthinit=_zeros((npts, nvm)),
        gdd_midwinter=_full((npts, nvm), undef),
        ncd_dormance=_full((npts, nvm), undef),
        ngd_minus5=_zeros((npts, nvm)),
        gdd_init_date=gdd_init_date,
        time_hum_min=_full((npts, nvm), undef),
        hum_min_dormance=_full((npts, nvm), undef),
        tseason=t2m.copy(),
        tseason_length=t2m.copy(),
        tseason_tmp=t2m.copy(),
        tmin_spring_time=_zeros((npts, nvm)),
        begin_leaves=_zeros((npts, nvm), dtype=bool),
        onset_date=_zeros((npts, nvm)),
        gpp_week=_zeros((npts, nvm)),
        maxmoiavail_lastyear=_zeros((npts, nvm)),
        maxmoiavail_thisyear=_zeros((npts, nvm)),
        minmoiavail_lastyear=_full((npts, nvm), 1.0),
        minmoiavail_thisyear=_full((npts, nvm), 1.0),
        maxgppweek_lastyear=_zeros((npts, nvm)),
        maxgppweek_thisyear=_zeros((npts, nvm)),
        gdd0_lastyear=_full((npts,), gdd_crit_estab),
        gdd0_thisyear=_zeros((npts,)),
        precip_lastyear=_full((npts,), precip_crit),
        precip_thisyear=_zeros((npts,)),
        maxfpc_lastyear=_zeros((npts, nvm)),
        maxfpc_thisyear=_zeros((npts, nvm)),
        lm_thisyearmax=_zeros((npts, nvm)),
        provenance=STOMATE_COLD_START_READSTART_PROVENANCE,
    )


def stomate_cold_start_entry_state(
    *,
    t2m,
    nvm: int,
    nslm: int,
    ndeep: int = 32,
    nelements: int = 1,
    dt_days: float = 1.0,
    date: int = 0,
    sla=None,
    thawed_humidity_input: float = 1.0e20,
    large_value: float = 1.0e33,
) -> StomateRestartEntryState:
    """Return no-restart STOMATE entry state fallbacks from ``readstart``.

    The object covers the state fields already normalized by
    ``StomateRestartEntryState``. Optional wetland-CH4 and crop-management
    arrays that are not part of this entry-state contract remain outside this
    helper and must be tracked in the branch ledger before use.
    """

    t2m = np.asarray(t2m, dtype=np.float64)
    if t2m.ndim != 1:
        raise ValueError("t2m must have shape (npts,)")
    npts = t2m.shape[0]
    nvm = int(nvm)
    nslm = int(nslm)
    ndeep = int(ndeep)
    nelements = int(nelements)
    if npts < 1 or nvm < 1 or nslm < 1 or ndeep < 1 or nelements < 1:
        raise ValueError("npts, nvm, nslm, ndeep, and nelements must be positive")
    if sla is None:
        sla_calc = _zeros((npts, nvm))
    else:
        sla_arr = np.asarray(sla, dtype=np.float64)
        if sla_arr.shape != (nvm,):
            raise ValueError("sla must have shape (nvm,)")
        sla_calc = np.broadcast_to(sla_arr[None, :], (npts, nvm)).copy()
    daily = stomate_cold_start_daily_accumulator_state(t2m=t2m, nvm=nvm, nslm=nslm, large_value=large_value)
    return StomateRestartEntryState(
        biomass=_zeros((npts, nvm, NPARTS, nelements)),
        resp_maint_part=_zeros((npts, nvm, NPARTS)),
        gpp_daily=daily.gpp_daily,
        npp_daily=_zeros((npts, nvm)),
        turnover_daily=_zeros((npts, nvm, NPARTS, nelements)),
        resp_maint=_zeros((npts, nvm)),
        resp_growth=_zeros((npts, nvm)),
        leaf_age=_zeros((npts, nvm, NLEAFAGES)),
        leaf_frac=_zeros((npts, nvm, NLEAFAGES)),
        age=_zeros((npts, nvm)),
        sla_calc=sla_calc,
        pft_present=_zeros((npts, nvm), dtype=bool),
        ind=_zeros((npts, nvm)),
        adapted=_zeros((npts, nvm)),
        regenerate=_zeros((npts, nvm)),
        npp_longterm=_zeros((npts, nvm)),
        lm_lastyearmax=_zeros((npts, nvm)),
        turnover_time=_full((npts, nvm), 100.0),
        turnover_longterm=_zeros((npts, nvm, NPARTS, nelements)),
        senescence=_zeros((npts, nvm), dtype=bool),
        when_growthinit=_zeros((npts, nvm)),
        co2_to_bm=_zeros((npts, nvm)),
        veget_lastlight=_zeros((npts, nvm)),
        everywhere=_zeros((npts, nvm)),
        need_adjacent=_zeros((npts, nvm), dtype=bool),
        litterpart=_zeros((npts, nvm, NLITT)),
        dead_leaves=_zeros((npts, nvm, NLITT)),
        carbon=_zeros((npts, NCARB, nvm)),
        litter=_zeros((npts, NLITT, nvm, 2, nelements)),
        lignin_struc=_zeros((npts, nvm, 2)),
        fuel_1hr=_zeros((npts, nvm, NLITT, nelements)),
        fuel_10hr=_zeros((npts, nvm, NLITT, nelements)),
        fuel_100hr=_zeros((npts, nvm, NLITT, nelements)),
        fuel_1000hr=_zeros((npts, nvm, NLITT, nelements)),
        prod10=_zeros((npts, 11, 2)),
        prod100=_zeros((npts, 101, 2)),
        flux10=_zeros((npts, 10, 2)),
        flux100=_zeros((npts, 100, 2)),
        prod10_total=_zeros((npts,)),
        prod100_total=_zeros((npts,)),
        bm_to_litter=_zeros((npts, nvm, NPARTS, nelements)),
        carb_mass_total=_zeros((npts,)),
        rip_time=_full((npts, nvm), large_value),
        assim_param=_zeros((npts, nvm, nelements)),
        altmax=_zeros((npts, nvm)),
        fixed_cryoturbation_depth=_zeros((npts, nvm)),
        fpeat=_zeros((npts,)),
        litter_above=_zeros((npts, NLITT, nvm, nelements)),
        litter_below=_zeros((npts, NLITT, nvm, ndeep, nelements)),
        carbon_32l=_zeros((npts, NCARB, nvm, ndeep)),
        DOC=_zeros((npts, nvm, ndeep, NDOC, NPOOL, nelements)),
        interception_storage=_zeros((npts, nvm, nelements)),
        lignin_struc_above=_zeros((npts, nvm)),
        lignin_struc_below=_zeros((npts, nvm, ndeep)),
        deepC_a=_zeros((npts, ndeep, nvm)),
        deepC_s=_zeros((npts, ndeep, nvm)),
        deepC_p=_zeros((npts, ndeep, nvm)),
        soilc_total=_zeros((npts, ndeep, nvm)),
        thawed_humidity=_full((npts,), thawed_humidity_input),
        depth_organic_soil=_zeros((npts,)),
    )


def find_stomate_reference_files(root: str | Path, landpoint_id: str = LEGACY_LANDPOINT_ID) -> StomateReferenceFiles:
    """Locate local STOMATE reference files.

    Reference/history reader validation boundary: file discovery only.
    """

    reference = resolve_paper_landpoint_reference(root, landpoint_id)
    if reference.output_dir is not None:
        run_dir = reference.output_dir
    else:
        if landpoint_id != LEGACY_LANDPOINT_ID:
            raise FileNotFoundError(f"No STOMATE reference output directory found for landpoint {landpoint_id}")
        root = Path(root)
        run_dir = root / REFERENCE_RUN_PATTERN
        if not run_dir.exists():
            matches = sorted(root.glob("reference/case_001_071/**/stomate_restart.nc"))
            if not matches:
                raise FileNotFoundError("No stomate_restart.nc found under reference/case_001_071")
            run_dir = matches[0].parent

    return stomate_reference_files_from_run_dir(run_dir)


def stomate_reference_files_from_run_dir(run_dir: str | Path) -> StomateReferenceFiles:
    """Validate STOMATE reference files in an explicit case run directory."""

    run_dir = Path(run_dir)
    start = run_dir / "stomate_start.nc"
    restart = run_dir / "stomate_restart.nc"
    histories = tuple(sorted(run_dir.glob("stomate_history_*.nc")))
    missing = [path for path in (start, restart) if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required STOMATE reference files: {joined}")
    if not histories:
        raise FileNotFoundError(f"No stomate_history_*.nc files found in {run_dir}")
    return StomateReferenceFiles(run_dir=run_dir, start=start, restart=restart, histories=histories)


def default_paper_modelout_csv_path(root: str | Path, landpoint_id: str = LEGACY_LANDPOINT_ID) -> Path:
    """Return the archived paper CSV path used by the sensitivity script."""

    reference = resolve_paper_landpoint_reference(root, landpoint_id)
    if reference.modelout_csv is not None:
        return reference.modelout_csv
    if reference.modelout_csv_zip is not None:
        raise FileNotFoundError(
            "modelout CSV is present only inside a zip archive; "
            f"use read_paper_modelout_csv(root=..., landpoint_id='{landpoint_id}') "
            "or materialize the reference script package first: "
            f"{reference.modelout_csv_zip.archive}!{reference.modelout_csv_zip.member}"
        )
    raise FileNotFoundError(f"No paper modelout CSV found for landpoint {landpoint_id}")


def _read_paper_modelout_csv_from_zip(root: str | Path, landpoint_id: str) -> tuple[PaperModeloutCsvRow, ...] | None:
    reference = resolve_paper_landpoint_reference(root, landpoint_id)
    if reference.modelout_csv_zip is None:
        return None
    rows: list[PaperModeloutCsvRow] = []
    with ZipFile(reference.modelout_csv_zip.archive) as zip_file:
        with zip_file.open(reference.modelout_csv_zip.member) as raw_handle:
            lines = (line.decode("utf-8") for line in raw_handle)
            for raw in csv.DictReader(lines):
                age = int(float(raw["age"]))
                rows.append(
                    PaperModeloutCsvRow(
                        i_grid=str(raw["Igrid"]),
                        age=age,
                        target_year=history_year_for_age(age),
                        AGB_model=float(raw["AGB_model"]),
                        BGB_model=float(raw["BGB_model"]),
                        GPP_model=float(raw["GPP_model"]),
                        NPP_model=float(raw["NPP_model"]),
                    )
                )
    return tuple(rows)


def read_paper_modelout_csv(
    path: str | Path | None = None,
    *,
    root: str | Path | None = None,
    landpoint_id: str = LEGACY_LANDPOINT_ID,
) -> tuple[PaperModeloutCsvRow, ...]:
    """Read archived paper modelout rows with their target history years.

    This is a validation-boundary helper only. It mirrors the paper script
    row/year protocol and never computes ecological process state.
    """

    if path is None:
        if root is None:
            raise ValueError("root is required when path is not supplied")
        zipped_rows = _read_paper_modelout_csv_from_zip(root, landpoint_id)
        if zipped_rows is not None:
            return zipped_rows
        csv_path = default_paper_modelout_csv_path(root, landpoint_id)
    else:
        csv_path = Path(path)
    rows: list[PaperModeloutCsvRow] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            age = int(float(raw["age"]))
            rows.append(
                PaperModeloutCsvRow(
                    i_grid=str(raw["Igrid"]),
                    age=age,
                    target_year=history_year_for_age(age),
                    AGB_model=float(raw["AGB_model"]),
                    BGB_model=float(raw["BGB_model"]),
                    GPP_model=float(raw["GPP_model"]),
                    NPP_model=float(raw["NPP_model"]),
                )
            )
    return tuple(rows)


def paper_modelout_csv_targets_by_year(
    path: str | Path | None = None,
    *,
    root: str | Path | None = None,
    landpoint_id: str = LEGACY_LANDPOINT_ID,
) -> dict[int, tuple[PaperModeloutCsvRow, ...]]:
    """Group paper CSV targets by the STOMATE history year they validate."""

    grouped: dict[int, list[PaperModeloutCsvRow]] = {}
    for row in read_paper_modelout_csv(path, root=root, landpoint_id=landpoint_id):
        grouped.setdefault(row.target_year, []).append(row)
    return {year: tuple(rows) for year, rows in grouped.items()}


def inventory_netcdf(path: str | Path) -> dict[str, NetCDFVariableInfo]:
    """Return variable metadata without reading full arrays.

    Reference/history reader validation boundary: NetCDF metadata inventory.
    """

    info: dict[str, NetCDFVariableInfo] = {}
    with Dataset(path) as dataset:
        for name, variable in dataset.variables.items():
            info[name] = NetCDFVariableInfo(
                name=name,
                dimensions=tuple(variable.dimensions),
                shape=tuple(int(size) for size in variable.shape),
                units=getattr(variable, "units", None),
                long_name=getattr(variable, "long_name", None),
            )
    return info


def variable_presence(path: str | Path, names: tuple[str, ...]) -> dict[str, bool]:
    """Report exact variable presence for a NetCDF file."""

    inventory = inventory_netcdf(path)
    return {name: name in inventory for name in names}


@lru_cache(maxsize=1024)
def _read_variables_cached(path: str, names: tuple[str, ...]) -> tuple[tuple[str, np.ndarray], ...]:
    """Read NetCDF variables once per file/name tuple for audit scaffolds."""

    with Dataset(path) as dataset:
        missing = [name for name in names if name not in dataset.variables]
        if missing:
            joined = ", ".join(missing)
            raise KeyError(f"Missing variables in {path}: {joined}")
        return tuple((name, np.asarray(dataset.variables[name][:])) for name in names)


def read_variables(path: str | Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Read explicitly selected NetCDF variables as NumPy arrays."""

    resolved = str(Path(path).resolve())
    cached = _read_variables_cached(resolved, tuple(names))
    return {name: np.array(value, copy=True) for name, value in cached}


def read_restart_biomass_carbon(path: str | Path) -> np.ndarray:
    """Read restart biomass carbon as `(npts, nvm, nparts, nelements)`.

    Reference/history reader validation boundary with Fortran pool provenance:
    restart variable `biomass(time,m_a,l_a,z_a,y,x)` stores element, pool, PFT,
    and grid axes. Pool order follows `src_parameters/constantes_var.f90`,
    lines 196-208.
    """

    fields = read_variables(path, ("biomass",))
    biomass = fields["biomass"]
    if biomass.ndim != 6:
        raise ValueError("restart biomass must have shape (time, element, pool, pft, y, x)")
    if biomass.shape[2] != NPARTS:
        raise ValueError(f"restart biomass pool axis must have length {NPARTS}")
    selected = biomass[0]
    normalized = np.transpose(selected, (3, 4, 2, 1, 0)).reshape(
        selected.shape[3] * selected.shape[4],
        selected.shape[2],
        selected.shape[1],
        selected.shape[0],
    )
    return normalized[:, :, :, [ICARBON]]


def read_restart_pft_field(path: str | Path, name: str) -> np.ndarray:
    """Read restart `(time,pft,y,x)` field as `(npts,nvm)`."""

    fields = read_variables(path, (name,))
    field = fields[name]
    if field.ndim != 4:
        raise ValueError(f"{name} must have shape (time, pft, y, x)")
    selected = field[0]
    return np.transpose(selected, (1, 2, 0)).reshape(selected.shape[1] * selected.shape[2], selected.shape[0])


def read_restart_grid_field(path: str | Path, name: str) -> np.ndarray:
    """Read restart ``(time,y,x)`` field as ``(npts,)``."""

    fields = read_variables(path, (name,))
    field = fields[name]
    if field.ndim != 3:
        raise ValueError(f"{name} must have shape (time, y, x)")
    return field[0].reshape(field.shape[1] * field.shape[2])


def read_restart_scalar(path: str | Path, name: str) -> float:
    """Read restart scalar variables such as ``dt_days`` or ``tau_longterm``."""

    fields = read_variables(path, (name,))
    field = np.asarray(fields[name])
    if field.size != 1:
        raise ValueError(f"{name} must be a scalar restart variable")
    return float(field.reshape(-1)[0])


def read_restart_soil_layer_field(path: str | Path, name: str) -> np.ndarray:
    """Read restart ``(time,layer,y,x)`` field as ``(npts,nlayers)``."""

    fields = read_variables(path, (name,))
    field = fields[name]
    if field.ndim != 4:
        raise ValueError(f"{name} must have shape (time, layer, y, x)")
    selected = field[0]
    return np.transpose(selected, (1, 2, 0)).reshape(selected.shape[1] * selected.shape[2], selected.shape[0])


def read_restart_leaf_age_field(path: str | Path, name: str) -> np.ndarray:
    """Read restart ``(time,leaf_age,pft,y,x)`` as ``(npts,nvm,nleafages)``.

    Fortran provenance: ``src_stomate/stomate_io.f90::readstart`` lines
    930-940 reads ``leaf_age`` and ``leaf_frac`` with dimensions
    ``(nbp_glo,nvm,nleafages)``.
    """

    fields = read_variables(path, (name,))
    field = fields[name]
    if field.ndim != 5:
        raise ValueError(f"{name} must have shape (time, leaf_age, pft, y, x)")
    selected = field[0]
    return np.transpose(selected, (2, 3, 1, 0)).reshape(
        selected.shape[2] * selected.shape[3],
        selected.shape[1],
        selected.shape[0],
    )


def read_restart_assim_param(path: str | Path) -> np.ndarray:
    """Read restart ``assim_param`` as ``(npts,nvm,npco2)``.

    Fortran provenance: ``src_stomate/stomate_io.f90::readstart`` lines
    1737-1743 reads ``assim_param`` with dimensions
    ``(nbp_glo,nvm,npco2)`` when the restart variable is present.
    """

    fields = read_variables(path, ("assim_param",))
    field = fields["assim_param"]
    if field.ndim != 5:
        raise ValueError("assim_param must have shape (time, npco2, pft, y, x)")
    selected = field[0]
    return np.transpose(selected, (2, 3, 1, 0)).reshape(
        selected.shape[2] * selected.shape[3],
        selected.shape[1],
        selected.shape[0],
    )


def read_restart_maint_resp_part(path: str | Path) -> np.ndarray:
    """Read restart `maint_resp(time,pool,pft,y,x)` as `(npts,nvm,nparts)`."""

    fields = read_variables(path, ("maint_resp",))
    field = fields["maint_resp"]
    if field.ndim != 5:
        raise ValueError("maint_resp must have shape (time, pool, pft, y, x)")
    if field.shape[1] != NPARTS:
        raise ValueError(f"maint_resp pool axis must have length {NPARTS}")
    selected = field[0]
    return np.transpose(selected, (2, 3, 1, 0)).reshape(
        selected.shape[2] * selected.shape[3],
        selected.shape[1],
        selected.shape[0],
    )


def read_restart_pft_bool_field(path: str | Path, name: str) -> np.ndarray:
    """Read restart logical field stored as ``(time,pft,y,x)`` real flags."""

    return read_restart_pft_field(path, name).astype(np.float64) >= 0.5


def encode_restart_pft_bool_field(value: np.ndarray) -> np.ndarray:
    """Encode a PFT logical field as Fortran restart real flags.

    Fortran provenance: ``src_stomate/stomate_io.f90::writerestart`` converts
    logical PFT fields to real restart variables with ``WHERE`` assignments;
    ``need_adjacent`` is written as one/zero at lines 2707-2714. The matching
    read path converts restart values back to logical with ``>= .5`` at lines
    1025-1034.
    """

    value = np.asarray(value, dtype=bool)
    if value.ndim != 2:
        raise ValueError("PFT logical restart fields must have shape (npts, nvm)")
    return np.where(value, 1.0, 0.0).astype(np.float64)


def read_restart_pft_pool_field(path: str | Path, name: str) -> np.ndarray:
    """Read restart ``(time,element,pool,pft,y,x)`` as ``(npts,nvm,nparts,1)``.

    Fortran provenance: ``src_stomate/stomate_io.f90::readstart`` reads
    ``turnover_daily``, ``turnover_longterm``, and ``bm_to_litter`` with
    ``(nbp_glo,nvm,nparts,nelements)`` axes at lines 594-598, 908-911, and
    1147-1150.
    """

    fields = read_variables(path, (name,))
    field = fields[name]
    if field.ndim != 6:
        raise ValueError(f"{name} must have shape (time, element, pool, pft, y, x)")
    if field.shape[2] != NPARTS:
        raise ValueError(f"{name} pool axis must have length {NPARTS}")
    selected = field[0]
    normalized = np.transpose(selected, (3, 4, 2, 1, 0)).reshape(
        selected.shape[3] * selected.shape[4],
        selected.shape[2],
        selected.shape[1],
        selected.shape[0],
    )
    return normalized[:, :, :, [ICARBON]]


def read_restart_litter_above(path: str | Path) -> np.ndarray:
    """Read restart ``litter_above`` as ``(npts,nlitt,nvm,nelements)``.

    Fortran provenance: ``src_stomate/stomate_io.f90::readstart`` lines
    1512-1515 reads ``litter_above`` with dimensions
    ``(nbp_glo,nlitt,nvm,nelements)``.
    """

    fields = read_variables(path, ("litter_above",))
    field = fields["litter_above"]
    if field.ndim != 6:
        raise ValueError("litter_above must have shape (time, element, pft, nlitt, y, x)")
    selected = field[0]
    return np.transpose(selected, (3, 4, 2, 1, 0)).reshape(
        selected.shape[3] * selected.shape[4],
        selected.shape[2],
        selected.shape[1],
        selected.shape[0],
    )


def read_restart_litter_pft_field(path: str | Path, name: str) -> np.ndarray:
    """Read restart ``(time,nlitt,pft,y,x)`` as ``(npts,nvm,nlitt)``.

    Fortran provenance: ``src_stomate/stomate_io.f90::readstart`` lines
    1045-1060 reads ``litterpart`` and ``dead_leaves`` with dimensions
    ``(nbp_glo,nvm,nlitt)``.
    """

    fields = read_variables(path, (name,))
    field = fields[name]
    if field.ndim != 5:
        raise ValueError(f"{name} must have shape (time, nlitt, pft, y, x)")
    selected = field[0]
    return np.transpose(selected, (2, 3, 1, 0)).reshape(
        selected.shape[2] * selected.shape[3],
        selected.shape[1],
        selected.shape[0],
    )


def read_restart_legacy_carbon(path: str | Path) -> np.ndarray:
    """Read restart legacy ``carbon`` as ``(npts,ncarb,nvm)``.

    Fortran provenance: ``src_stomate/stomate_io.f90::readstart`` lines
    1062-1066 reads ``carbon(nbp_glo,ncarb,nvm)`` from the restart and falls
    back to zero only when absent. This is distinct from the OK_LEAK
    ``carbon_32l(nbp_glo,ncarb,nvm,ndeep)`` state read at lines 1540-1551.
    """

    fields = read_variables(path, ("carbon",))
    field = fields["carbon"]
    if field.ndim != 5:
        raise ValueError("carbon must have shape (time, pft, ncarb, y, x)")
    selected = field[0]
    if selected.shape[1] != NCARB:
        raise ValueError(f"carbon pool axis must have length {NCARB}")
    return np.transpose(selected, (2, 3, 1, 0)).reshape(
        selected.shape[2] * selected.shape[3],
        selected.shape[1],
        selected.shape[0],
    )


def read_restart_litter_legacy(path: str | Path) -> np.ndarray:
    """Read restart ``litter`` as ``(npts,nlitt,nvm,nlevs,nelements)``.

    Fortran provenance: ``src_stomate/stomate_io.f90::readstart`` lines
    1051-1055 reads legacy ``litter(nbp_glo,nlitt,nvm,nlevs,nelements)``.
    This is the two-level litter pool consumed by SPITFIRE, grazing, and LCC;
    it is distinct from the OK_LEAK ``litter_above``/``litter_below`` pools.
    """

    fields = read_variables(path, ("litter",))
    field = fields["litter"]
    if field.ndim != 7:
        raise ValueError("litter must have shape (time, element, nlitt, pft, nlevs, y, x)")
    selected = field[0]
    return np.transpose(selected, (4, 5, 1, 2, 3, 0)).reshape(
        selected.shape[4] * selected.shape[5],
        selected.shape[1],
        selected.shape[2],
        selected.shape[3],
        selected.shape[0],
    )


def read_restart_lignin_struc_legacy(path: str | Path) -> np.ndarray:
    """Read restart ``lignin_struc`` as ``(npts,nvm,nlevs)``.

    Fortran provenance: ``src_stomate/stomate_io.f90::readstart`` lines
    1069-1074 reads legacy ``lignin_struc(nbp_glo,nvm,nlevs)`` used with the
    two-level ``litter`` pool.
    """

    fields = read_variables(path, ("lignin_struc",))
    field = fields["lignin_struc"]
    if field.ndim != 5:
        raise ValueError("lignin_struc must have shape (time, nlevs, pft, y, x)")
    selected = field[0]
    return np.transpose(selected, (2, 3, 1, 0)).reshape(
        selected.shape[2] * selected.shape[3],
        selected.shape[1],
        selected.shape[0],
    )


def read_restart_litter_fuel_field(path: str | Path, name: str) -> np.ndarray:
    """Read restart fuel field as ``(npts,nvm,nlitt,nelements)``.

    Fortran provenance: ``src_stomate/stomate_io.f90::readstart`` lines
    1081-1098 reads ``fuel_1hr``, ``fuel_10hr``, ``fuel_100hr``, and
    ``fuel_1000hr`` with dimensions ``(nbp_glo,nvm,nlitt,nelements)``.
    """

    fields = read_variables(path, (name,))
    field = fields[name]
    if field.ndim != 6:
        raise ValueError(f"{name} must have shape (time, element, nlitt, pft, y, x)")
    selected = field[0]
    return np.transpose(selected, (3, 4, 2, 1, 0)).reshape(
        selected.shape[3] * selected.shape[4],
        selected.shape[2],
        selected.shape[1],
        selected.shape[0],
    )


def read_restart_product_pool(path: str | Path, name: str, age_len: int) -> np.ndarray:
    """Read restart product/flux pools as ``(npts, age, nwp)``.

    Fortran provenance: ``src_stomate/stomate_io.f90::readstart`` lines
    1104-1129 reads ``prod10(nbp_glo,11,nwp)``,
    ``prod100(nbp_glo,101,nwp)``, ``flux10(nbp_glo,10,nwp)``, and
    ``flux100(nbp_glo,100,nwp)``.
    """

    fields = read_variables(path, (name,))
    field = fields[name]
    if field.ndim != 5:
        raise ValueError(f"{name} must have shape (time, pool, age, y, x)")
    if field.shape[2] != int(age_len):
        raise ValueError(f"{name} age axis must have length {int(age_len)}")
    selected = field[0]
    return np.transpose(selected, (2, 3, 1, 0)).reshape(
        selected.shape[2] * selected.shape[3],
        selected.shape[1],
        selected.shape[0],
    )


def read_restart_product_total(path: str | Path, name: str) -> np.ndarray:
    """Read restart ``prod10``/``prod100`` and sum to ``(npts,)`` totals.

    Fortran provenance: ``src_stomate/stomate_io.f90::readstart`` lines
    1104-1115 reads ``prod10(nbp_glo,11,nwp)`` and
    ``prod100(nbp_glo,101,nwp)``. ``src_stomate/stomate_lpj.f90`` lines
    1545-1548 computes ``SUM(SUM(prod*, dim=2), dim=2)`` before output
    diagnostics.
    """

    age_len = 11 if name == "prod10" else 101 if name == "prod100" else None
    if age_len is None:
        raise ValueError("product total is defined only for prod10 and prod100")
    by_point = read_restart_product_pool(path, name, age_len)
    return np.sum(by_point, axis=(1, 2))


def read_restart_litter_below(path: str | Path) -> np.ndarray:
    """Read restart ``litter_below`` as ``(npts,nlitt,nvm,ndeep,nelements)``.

    Fortran provenance: ``src_stomate/stomate_io.f90::readstart`` lines
    1517-1520 reads ``litter_below`` with dimensions
    ``(nbp_glo,nlitt,nvm,ndeep,nelements)``.
    """

    fields = read_variables(path, ("litter_below",))
    field = fields["litter_below"]
    if field.ndim != 7:
        raise ValueError("litter_below must have shape (time, element, ndeep, pft, nlitt, y, x)")
    selected = field[0]
    return np.transpose(selected, (4, 5, 3, 2, 1, 0)).reshape(
        selected.shape[4] * selected.shape[5],
        selected.shape[3],
        selected.shape[2],
        selected.shape[1],
        selected.shape[0],
    )


def read_restart_lignin_struc_below(path: str | Path) -> np.ndarray:
    """Read restart ``lig_struc_be`` as ``(npts,nvm,ndeep)``.

    Fortran provenance: ``src_stomate/stomate_io.f90::readstart`` lines
    1529-1533 maps restart variable ``lig_struc_be`` to
    ``lignin_struc_below(nbp_glo,nvm,ndeep)``.
    """

    fields = read_variables(path, ("lig_struc_be",))
    field = fields["lig_struc_be"]
    if field.ndim != 5:
        raise ValueError("lig_struc_be must have shape (time, ndeep, pft, y, x)")
    selected = field[0]
    return np.transpose(selected, (2, 3, 1, 0)).reshape(
        selected.shape[2] * selected.shape[3],
        selected.shape[1],
        selected.shape[0],
    )


def _read_restart_carbon_32l_pool(path: str | Path, name: str) -> np.ndarray:
    fields = read_variables(path, (name,))
    field = fields[name]
    if field.ndim != 5:
        raise ValueError(f"{name} must have shape (time, ndeep, pft, y, x)")
    selected = field[0]
    return np.transpose(selected, (2, 3, 1, 0)).reshape(
        selected.shape[2] * selected.shape[3],
        selected.shape[1],
        selected.shape[0],
    )


def read_restart_carbon_32l(path: str | Path) -> np.ndarray:
    """Read restart soil carbon as ``(npts,ncarb,nvm,ndeep)``.

    Fortran provenance: ``src_stomate/stomate_io.f90::readstart`` lines
    1540-1551 maps ``carbon_32l_a/s/p`` into active, slow, and passive
    ``carbon_32l(nbp_glo,ncarb,nvm,ndeep)`` pools.
    """

    active = _read_restart_carbon_32l_pool(path, "carbon_32l_a")
    slow = _read_restart_carbon_32l_pool(path, "carbon_32l_s")
    passive = _read_restart_carbon_32l_pool(path, "carbon_32l_p")
    return np.stack((active, slow, passive), axis=1)


def _read_restart_doc_pool(path: str | Path, name: str) -> np.ndarray:
    fields = read_variables(path, (name,))
    field = fields[name]
    if field.ndim != 7:
        raise ValueError(f"{name} must have shape (time, element, pool, ndeep, pft, y, x)")
    selected = field[0]
    return np.transpose(selected, (4, 5, 3, 2, 1, 0)).reshape(
        selected.shape[4] * selected.shape[5],
        selected.shape[3],
        selected.shape[2],
        selected.shape[1],
        selected.shape[0],
    )


def read_restart_doc(path: str | Path) -> np.ndarray:
    """Read restart DOC as ``(npts,nvm,ndeep,ndoc,npool,nelements)``.

    Fortran provenance: ``src_stomate/stomate_io.f90::readstart`` lines
    1553-1563 maps ``freedoc`` and ``adsdoc`` into ``DOC`` free and adsorbed
    pools. DOC pool indices follow ``src_parameters/constantes_var.f90`` lines
    259-261.
    """

    freedoc = _read_restart_doc_pool(path, "freedoc")
    adsdoc = _read_restart_doc_pool(path, "adsdoc")
    return np.stack((freedoc, adsdoc), axis=3)


def read_restart_interception_storage(path: str | Path) -> np.ndarray:
    """Read restart TF-DOC canopy storage as ``(npts,nvm,1)``.

    Fortran provenance: ``src_stomate/stomate_io.f90::readstart`` lines
    1567-1571 reads ``interception_storage(:,:,icarbon)`` from restart
    variable ``interception_storage`` and falls back to zero only when the
    restart sentinel is present.
    """

    fields = read_variables(path, ("interception_storage",))
    field = fields["interception_storage"]
    if field.ndim != 4:
        raise ValueError("interception_storage must have shape (time, pft, y, x)")
    selected = field[0]
    normalized = np.transpose(selected, (1, 2, 0)).reshape(
        selected.shape[1] * selected.shape[2],
        selected.shape[0],
    )
    return normalized[:, :, None]


def read_restart_deep_carbon_pool(path: str | Path, name: str) -> np.ndarray:
    """Read one ``deepC_*`` restart pool as ``(npts,ndeep,nvm)``.

    Fortran provenance: ``src_stomate/stomate_io.f90::readstart`` lines
    1159-1175 reads ``deepC_a``, ``deepC_s``, and ``deepC_p`` as
    ``(nbp_glo,ndeep,nvm)`` with zero fallback when absent.
    """

    fields = read_variables(path, (name,))
    field = fields[name]
    if field.ndim != 5:
        raise ValueError(f"{name} must have shape (time, pft, ndeep, y, x)")
    selected = field[0]
    return np.transpose(selected, (2, 3, 1, 0)).reshape(
        selected.shape[2] * selected.shape[3],
        selected.shape[1],
        selected.shape[0],
    )


def read_restart_pft_vertical_field(path: str | Path, name: str) -> np.ndarray:
    """Read restart ``(time,pft,vertical,y,x)`` as ``(npts,vertical,nvm)``."""

    fields = read_variables(path, (name,))
    field = fields[name]
    if field.ndim != 5:
        raise ValueError(f"{name} must have shape (time, pft, vertical, y, x)")
    selected = field[0]
    return np.transpose(selected, (2, 3, 1, 0)).reshape(
        selected.shape[2] * selected.shape[3],
        selected.shape[1],
        selected.shape[0],
    )


def read_restart_deep_carbon_total(path: str | Path) -> np.ndarray:
    """Read ``deepC_a/s/p`` restart pools as ``soilc_total(npts,ndeep,nvm)``.

    Fortran provenance: ``src_stomate/stomate_io.f90::readstart`` lines
    1159-1175 reads ``deepC_a``, ``deepC_s``, and ``deepC_p``. Then
    ``src_stomate/stomate.f90::stomate_init`` lines 1586-1589 sets
    ``soilc_total(:,:,:) = deepC_a + deepC_s + deepC_p`` before the first
    ``stomate_main`` call.
    """

    pools = [read_restart_deep_carbon_pool(path, name) for name in ("deepC_a", "deepC_s", "deepC_p")]
    return pools[0] + pools[1] + pools[2]


def read_stomate_ok_pc_restart_gas_state(path: str | Path) -> StomateOkPcRestartGasState:
    """Read OK_PC soil/snow gas restart state without inferring SAVE fields.

    Fortran provenance: ``src_stomate/stomate_io.f90::readstart`` lines
    1177-1199 reads ``O2_soil``, ``CH4_soil``, ``O2_snow``, and
    ``CH4_snow``. Runtime gas-diffusion coefficients, snow-grid geometry,
    porosity/diffusivity arrays, and active-layer indices are module SAVE
    state initialized inside ``deep_carbcycle`` firstcall, not restart-file
    variables, so this reader deliberately exposes only restart-backed gas
    concentrations.
    """

    return StomateOkPcRestartGasState(
        O2_soil=read_restart_pft_vertical_field(path, "O2_soil"),
        CH4_soil=read_restart_pft_vertical_field(path, "CH4_soil"),
        O2_snow=read_restart_pft_vertical_field(path, "O2_snow"),
        CH4_snow=read_restart_pft_vertical_field(path, "CH4_snow"),
    )


def read_stomate_restart_entry_state(path: str | Path) -> StomateRestartEntryState:
    """Read restart fields needed at the ``slowproc_main -> stomate_main`` edge.

    This is a restart/state boundary reader, not a process implementation.
    It only normalizes variables explicitly present in the restart file to the
    Fortran axes used by ``stomate_main``. Fortran provenance: restart reads in
    ``src_stomate/stomate_io.f90::readstart`` lines 919-940, 1201-1225,
    1227-1232, 1466-1471, and 1512-1563; the corresponding ``stomate_main`` call is
    ``src_sechiba/slowproc.f90`` lines 973-1013.
    """

    prod10 = read_restart_product_pool(path, "prod10", 11)
    prod100 = read_restart_product_pool(path, "prod100", 101)
    flux10 = read_restart_product_pool(path, "flux10", 10)
    flux100 = read_restart_product_pool(path, "flux100", 100)

    deepC_a = read_restart_deep_carbon_pool(path, "deepC_a")
    deepC_s = read_restart_deep_carbon_pool(path, "deepC_s")
    deepC_p = read_restart_deep_carbon_pool(path, "deepC_p")

    return StomateRestartEntryState(
        biomass=read_restart_biomass_carbon(path),
        resp_maint_part=read_restart_maint_resp_part(path),
        gpp_daily=read_restart_pft_field(path, "gpp_daily"),
        npp_daily=read_restart_pft_field(path, "npp_daily"),
        turnover_daily=read_restart_pft_pool_field(path, "turnover_daily"),
        resp_maint=read_restart_pft_field(path, "resp_maint"),
        resp_growth=read_restart_pft_field(path, "resp_growth"),
        leaf_age=read_restart_leaf_age_field(path, "leaf_age"),
        leaf_frac=read_restart_leaf_age_field(path, "leaf_frac"),
        age=read_restart_pft_field(path, "age"),
        sla_calc=read_restart_pft_field(path, "sla_calc"),
        pft_present=read_restart_pft_field(path, "PFTpresent").astype(bool),
        ind=read_restart_pft_field(path, "ind"),
        adapted=read_restart_pft_field(path, "adapted"),
        regenerate=read_restart_pft_field(path, "regenerate"),
        npp_longterm=read_restart_pft_field(path, "npp_longterm"),
        lm_lastyearmax=read_restart_pft_field(path, "lm_lastyearmax"),
        turnover_time=read_restart_pft_field(path, "turnover_time"),
        turnover_longterm=read_restart_pft_pool_field(path, "turnover_longterm"),
        senescence=read_restart_pft_bool_field(path, "senescence"),
        when_growthinit=read_restart_pft_field(path, "when_growthinit"),
        co2_to_bm=read_restart_pft_field(path, "co2_to_bm_dgvm"),
        veget_lastlight=read_restart_pft_field(path, "veget_lastlight"),
        everywhere=read_restart_pft_field(path, "everywhere"),
        need_adjacent=read_restart_pft_bool_field(path, "need_adjacent"),
        litterpart=read_restart_litter_pft_field(path, "litterpart"),
        dead_leaves=read_restart_litter_pft_field(path, "dead_leaves"),
        carbon=read_restart_legacy_carbon(path),
        litter=read_restart_litter_legacy(path),
        lignin_struc=read_restart_lignin_struc_legacy(path),
        fuel_1hr=read_restart_litter_fuel_field(path, "fuel_1hr"),
        fuel_10hr=read_restart_litter_fuel_field(path, "fuel_10hr"),
        fuel_100hr=read_restart_litter_fuel_field(path, "fuel_100hr"),
        fuel_1000hr=read_restart_litter_fuel_field(path, "fuel_1000hr"),
        prod10=prod10,
        prod100=prod100,
        flux10=flux10,
        flux100=flux100,
        prod10_total=np.sum(prod10, axis=(1, 2)),
        prod100_total=np.sum(prod100, axis=(1, 2)),
        bm_to_litter=read_restart_pft_pool_field(path, "bm_to_litter"),
        carb_mass_total=read_restart_grid_field(path, "carb_mass_total"),
        rip_time=read_restart_pft_field(path, "RIP_time"),
        assim_param=read_restart_assim_param(path),
        altmax=read_restart_pft_field(path, "altmax"),
        fixed_cryoturbation_depth=read_restart_pft_field(path, "fixed_cryoturb_depth"),
        fpeat=read_restart_grid_field(path, "fpeat"),
        litter_above=read_restart_litter_above(path),
        litter_below=read_restart_litter_below(path),
        carbon_32l=read_restart_carbon_32l(path),
        DOC=read_restart_doc(path),
        interception_storage=read_restart_interception_storage(path),
        lignin_struc_above=read_restart_pft_field(path, "lignin_struc_above"),
        lignin_struc_below=read_restart_lignin_struc_below(path),
        deepC_a=deepC_a,
        deepC_s=deepC_s,
        deepC_p=deepC_p,
        soilc_total=deepC_a + deepC_s + deepC_p,
        thawed_humidity=read_restart_grid_field(path, "thawed_humidity"),
        depth_organic_soil=read_restart_grid_field(path, "depth_organic_soil"),
    )


def read_stomate_restart_season_state(path: str | Path) -> StomateRestartSeasonState:
    """Read restart-backed season memory used before STOMATE carbon processes.

    This is a restart/state boundary reader. It exposes variables that
    ``stomate_io.f90::readstart`` reads into season/phenology/allocation memory;
    it does not advance the ``season`` subroutine or fill variables absent from
    the restart file.
    """

    return StomateRestartSeasonState(
        dt_days_read=read_restart_scalar(path, "dt_days"),
        date=int(round(read_restart_scalar(path, "date"))),
        tau_longterm=read_restart_scalar(path, "tau_longterm"),
        moiavail_month=read_restart_pft_field(path, "moiavail_month"),
        moiavail_week=read_restart_pft_field(path, "moiavail_week"),
        t2m_longterm=read_restart_grid_field(path, "t2m_longterm"),
        t2m_month=read_restart_grid_field(path, "t2m_month"),
        t2m_week=read_restart_grid_field(path, "t2m_week"),
        tsoil_month=read_restart_soil_layer_field(path, "tsoil_month"),
        soilhum_month=read_restart_soil_layer_field(path, "soilhum_month"),
        gdd_m5_dormance=read_restart_pft_field(path, "gdd_m5_dormance"),
        gdd_from_growthinit=read_restart_pft_field(path, "gdd_from_growthinit"),
        gdd_midwinter=read_restart_pft_field(path, "gdd_midwinter"),
        ncd_dormance=read_restart_pft_field(path, "ncd_dormance"),
        ngd_minus5=read_restart_pft_field(path, "ngd_minus5"),
        gdd_init_date=read_restart_soil_layer_field(path, "gdd_init_date"),
        time_hum_min=read_restart_pft_field(path, "time_hum_min"),
        hum_min_dormance=read_restart_pft_field(path, "hum_min_dormance"),
        tseason=read_restart_grid_field(path, "Tseason"),
        tseason_length=read_restart_grid_field(path, "Tseason_length"),
        tseason_tmp=read_restart_grid_field(path, "Tseason_tmp"),
        tmin_spring_time=read_restart_pft_field(path, "Tmin_spring_time"),
        begin_leaves=read_restart_pft_bool_field(path, "begin_leaves"),
        onset_date=read_restart_pft_field(path, "onset_date"),
        gpp_week=read_restart_pft_field(path, "gpp_week"),
        maxmoiavail_lastyear=read_restart_pft_field(path, "maxmoistr_last"),
        maxmoiavail_thisyear=read_restart_pft_field(path, "maxmoistr_this"),
        minmoiavail_lastyear=read_restart_pft_field(path, "minmoistr_last"),
        minmoiavail_thisyear=read_restart_pft_field(path, "minmoistr_this"),
        maxgppweek_lastyear=read_restart_pft_field(path, "maxgppweek_lastyear"),
        maxgppweek_thisyear=read_restart_pft_field(path, "maxgppweek_thisyear"),
        gdd0_lastyear=read_restart_grid_field(path, "gdd0_lastyear"),
        gdd0_thisyear=read_restart_grid_field(path, "gdd0_thisyear"),
        precip_lastyear=read_restart_grid_field(path, "precip_lastyear"),
        precip_thisyear=read_restart_grid_field(path, "precip_thisyear"),
        maxfpc_lastyear=read_restart_pft_field(path, "maxfpc_lastyear"),
        maxfpc_thisyear=read_restart_pft_field(path, "maxfpc_thisyear"),
        lm_thisyearmax=read_restart_pft_field(path, "lm_thisyearmax"),
    )


def read_stomate_daily_accumulator_state(path: str | Path) -> StomateDailyAccumulatorState:
    """Read restart-backed daily accumulators for the first ``stomate_main`` call.

    This is a restart/state boundary reader. It does not reset or advance
    accumulators; the caller must apply ``stomate_accu`` semantics explicitly.
    """

    t2m_daily = read_restart_grid_field(path, "t2m_daily")
    return StomateDailyAccumulatorState(
        humrel_daily=read_restart_pft_field(path, "moiavail_daily"),
        litterhum_daily=read_restart_grid_field(path, "litterhum_daily"),
        t2m_daily=t2m_daily,
        t2m_min_daily=read_restart_grid_field(path, "t2m_min_daily"),
        t2m_max_daily=read_restart_grid_field(path, "t2m_max_daily"),
        wspeed_daily=read_restart_grid_field(path, "wspeed_daily"),
        tsurf_daily=read_restart_grid_field(path, "tsurf_daily"),
        tsoil_daily=read_restart_soil_layer_field(path, "tsoil_daily"),
        soilhum_daily=read_restart_soil_layer_field(path, "soilhum_daily"),
        precip_daily=read_restart_grid_field(path, "precip_daily"),
        gpp_daily=read_restart_pft_field(path, "gpp_daily"),
        snowfall_daily=np.zeros_like(t2m_daily),
        snowmass_daily=np.zeros_like(t2m_daily),
        tmc_topgrass_daily=np.zeros_like(t2m_daily),
        fwet_daily=read_restart_grid_field(path, "fwet_daily"),
        liqwt_daily=read_restart_grid_field(path, "liqwt_daily"),
    )


def pack_history_modelout_fields(path: str | Path) -> dict[str, np.ndarray]:
    """Read real history fields required by the audited modelout mapper.

    Reference/history reader validation boundary. Formula application remains
    in `stomate.modelout.compute_modelout_from_fields`.
    """

    return read_variables(path, MODEL_OUTPUT_FIELD_NAMES)


def compare_modelout_fields_to_history_point(
    modelout_fields: Mapping[str, object],
    history_path: str | Path,
    *,
    field_names: tuple[str, ...] = MODEL_OUTPUT_FIELD_NAMES,
    time_index: int = 0,
    pft_index: int = PAPER_MODEL_PFT14_INDEX,
    lat_index: int = 0,
    lon_index: int = 0,
) -> ModeloutHistoryComparison:
    """Compare produced modelout fields with one Fortran history point.

    Fortran/history provenance: history fields are written by
    `fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj` lines
    2200-2246 and selected by the paper modelout script
    `fortran_run_scripts/paper_250919/c2.4_Model_run_functions_sensitivity.py`
    lines 60-73. The default `pft_index=13` is PFT14.
    """

    selected = tuple(field_names)
    missing = tuple(name for name in selected if name not in modelout_fields)
    if missing:
        joined = ", ".join(missing)
        raise KeyError(f"Missing JAX modelout fields: {joined}")
    history = read_variables(history_path, selected)

    jax_values: dict[str, float] = {}
    history_values: dict[str, float] = {}
    abs_differences: dict[str, float] = {}
    for name in selected:
        jax_arr = np.asarray(modelout_fields[name])
        if jax_arr.ndim == 2:
            jax_value = float(jax_arr[0, pft_index])
        elif jax_arr.ndim == 4:
            jax_value = float(jax_arr[time_index, pft_index, lat_index, lon_index])
        else:
            raise ValueError(f"{name} JAX field must have shape (npts,nvm) or (time,veget,lat,lon)")
        history_arr = np.asarray(history[name])
        if history_arr.ndim != 4:
            raise ValueError(f"{name} history field must have shape (time,veget,lat,lon)")
        history_value = float(history_arr[time_index, pft_index, lat_index, lon_index])
        jax_values[name] = jax_value
        history_values[name] = history_value
        abs_differences[name] = abs(jax_value - history_value)

    max_abs = max(abs_differences.values(), default=0.0)
    return ModeloutHistoryComparison(
        history_path=Path(history_path),
        fields=selected,
        jax_values=jax_values,
        history_values=history_values,
        abs_differences=abs_differences,
        max_abs_difference=max_abs,
    )


def read_history_pools_as_biomass(path: str | Path) -> np.ndarray:
    """Pack history pool outputs into `(time, nvm, nparts, lat, lon)`.

    Reference/history reader validation boundary with Fortran pool provenance:
    field-to-pool mapping follows `src_parameters/constantes_var.f90`, lines
    196-208. Missing history pool fields are left as zero because history does
    not expose every restart/process state field; callers must not treat this
    as full restart truth.
    """

    inventory = inventory_netcdf(path)
    first_name = next(name for name in HISTORY_POOL_TO_INDEX if name in inventory)
    first_shape = inventory[first_name].shape
    pools = np.zeros(first_shape[:2] + (NPARTS,) + first_shape[2:], dtype=np.float64)
    present_names = tuple(name for name in HISTORY_POOL_TO_INDEX if name in inventory)
    fields = read_variables(path, present_names)
    for name, pool_index in HISTORY_POOL_TO_INDEX.items():
        if name in fields:
            pools[:, :, pool_index, :, :] = fields[name]
    return pools


def read_restart_carbon_readiness(path: str | Path) -> RestartCarbonReadiness:
    """Read restart carbon fields needed to shape-check explicit adapters.

    Reference/history reader validation boundary with Fortran provenance:
    fields mirror `StomateLpj`/`npp_calc` boundary names in
    `src_stomate/stomate_lpj.f90` lines 1115-1131 and restart pool order from
    `src_parameters/constantes_var.f90` lines 196-208. These are restart state
    fields only; they are not before/after process traces.
    """

    return RestartCarbonReadiness(
        biomass=read_restart_biomass_carbon(path),
        maint_resp_part=read_restart_maint_resp_part(path),
        gpp_daily=read_restart_pft_field(path, "gpp_daily"),
        npp_daily=read_restart_pft_field(path, "npp_daily"),
        resp_maint=read_restart_pft_field(path, "resp_maint"),
        resp_growth=read_restart_pft_field(path, "resp_growth"),
        pft_present=read_restart_pft_field(path, "PFTpresent").astype(bool),
    )
