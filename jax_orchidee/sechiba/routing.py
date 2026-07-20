"""Source-backed SECHIBA routing micro-kernels.

These helpers intentionally cover only the audited boundaries of
``routing.f90::routing_main`` and ``routing.f90::routing_flow``. The
source-order wrapper below wires audited routing-flow kernels without
inventing missing state.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


IH2O = 0
IDOCL = 1
IDOCR = 2
ICO2AQ = 3
IPOCA = 4
IPOCS = 5
IPOCP = 6
ICLAYSED = 7
ISILTSED = 8
ISANDSED = 9

IRUNOFF = 0
IFLOODED = 1
IDRAINAGE = 2

IACTIVE = 0
ISLOW = 1
IPASSIVE = 2
IC_CLAY = 0
IC_SILT = 1
IC_SAND = 2

IDOCLABILE = 0
IDOCSTABLE = 1

IFASTR = 0
ISLOWR = 1
ISTREAMR = 2
IFLOODR = 3
IPONDR = 4
NAQSYS = 5


class RoutingAccumulatorState(NamedTuple):
    floodout_mean: np.ndarray
    precip_mean: np.ndarray
    runoff_mean: np.ndarray
    drainage_mean: np.ndarray
    temp_sol_mean: np.ndarray
    transpot_mean: np.ndarray
    totnobio_mean: np.ndarray
    k_litt_mean: np.ndarray
    humrel_mean: np.ndarray
    vegtot_mean: np.ndarray
    time_counter: float


class RoutingAccumulatorStep(NamedTuple):
    state: RoutingAccumulatorState
    due: bool
    zero_outputs: dict[str, np.ndarray]
    flow_input: np.ndarray | None
    flood_inp: np.ndarray | None
    stream_inp: np.ndarray | None


class RoutingLakeStep(NamedTuple):
    lake_reservoir: np.ndarray
    lakeinflow: np.ndarray
    return_lakes: np.ndarray
    lake_diag: np.ndarray


class RoutingReservoirOutflowStep(NamedTuple):
    fast_flow: np.ndarray
    slow_flow: np.ndarray
    stream_flow: np.ndarray
    fast_reservoir: np.ndarray
    slow_reservoir: np.ndarray
    stream_reservoir: np.ndarray
    qflow_avebas: np.ndarray
    stream_resavebas: np.ndarray
    stream_damavailbas: np.ndarray


class RoutingDailyScaledOutputs(NamedTuple):
    returnflow: np.ndarray
    reinfiltration: np.ndarray
    irrigation: np.ndarray
    sed_depositiontot: np.ndarray
    poc_depositiontot: np.ndarray
    riverflow: np.ndarray
    coastalflow: np.ndarray
    hydrographs: np.ndarray
    slowflow_diag: np.ndarray


class RoutingReturnReinfiltrationStep(NamedTuple):
    returnflow: np.ndarray
    reinfiltration: np.ndarray


class RoutingFlowDiagnosticsStep(NamedTuple):
    delsurfstor: np.ndarray
    netflow_fast_diag: np.ndarray
    netflow_slow_diag: np.ndarray
    netflow_stream_diag: np.ndarray
    hydrographs: np.ndarray
    slowflow_diag: np.ndarray
    fast_diag: np.ndarray
    slow_diag: np.ndarray
    stream_diag: np.ndarray
    flood_diag: np.ndarray
    pond_diag: np.ndarray
    irrigation: np.ndarray
    sed_deposition: np.ndarray
    poc_deposition: np.ndarray
    riv_erodep_sed: np.ndarray
    riv_erodep_poc: np.ndarray
    rivchannel_deposition: np.ndarray
    lakeinflow: np.ndarray
    coastalflow: np.ndarray
    riverflow: np.ndarray
    stream_inflow: np.ndarray
    stream_outflow: np.ndarray
    flood_daily: np.ndarray
    flood_res: np.ndarray
    fastr: np.ndarray
    poc_co2_rivbed: np.ndarray
    poc_doc_rivbed: np.ndarray


class RoutingLakeOverflowStep(NamedTuple):
    lake_reservoir: np.ndarray
    coastalflow: np.ndarray
    lake_overflow: np.ndarray
    lake_overflow_coast: np.ndarray
    total_lake_overflow: np.ndarray


class RoutingReservoirUpdateStep(NamedTuple):
    fast_reservoir: np.ndarray
    slow_reservoir: np.ndarray
    stream_reservoir: np.ndarray
    flood_reservoir: np.ndarray
    pond_reservoir: np.ndarray
    streamb_inflow: np.ndarray
    return_swamp: np.ndarray
    totflood: np.ndarray


class RoutingAreaFractionsStep(NamedTuple):
    stream_area_act: np.ndarray
    stream_area_bas: np.ndarray
    streamfl_frac: np.ndarray
    streamfl_frac_bas: np.ndarray
    stream_seddep: np.ndarray
    rivbed2fld_sed: np.ndarray
    rivbed2fld_poc: np.ndarray
    flood_frac: np.ndarray
    flood_frac_bas: np.ndarray
    flood_height: np.ndarray
    pond_frac: np.ndarray


class RoutingPondFluxStep(NamedTuple):
    fast_flow: np.ndarray
    pond_reservoir: np.ndarray
    pond_inflow: np.ndarray
    pond_drainage: np.ndarray
    flood_dep_sed: np.ndarray
    flood_dep_poc: np.ndarray


class RoutingSwampFloodStep(NamedTuple):
    streamr50th_bas: np.ndarray
    stream_flow50th: np.ndarray
    return_swamp: np.ndarray
    floods: np.ndarray
    flood_reservoir: np.ndarray
    flood_dep_sed: np.ndarray
    flood_dep_poc: np.ndarray


class RoutingFloodplainFluxStep(NamedTuple):
    flood_reservoir: np.ndarray
    stream_reservoir: np.ndarray
    flood_drainage: np.ndarray
    flood_flow: np.ndarray
    flood_dep_sed: np.ndarray
    flood_dep_poc: np.ndarray


class RoutingFloodPondInputStep(NamedTuple):
    flood_reservoir: np.ndarray
    pond_reservoir: np.ndarray
    flood_dep_sed: np.ndarray
    flood_dep_poc: np.ndarray
    flood_inp_bas: np.ndarray
    stream_inp_bas: np.ndarray


class RoutingStreamErosionStep(NamedTuple):
    stream_reservoir: np.ndarray
    stream_flow: np.ndarray
    stream_seddep: np.ndarray
    stream_erodep: np.ndarray
    sedcap: np.ndarray
    stream_damavailbas: np.ndarray
    stream_damavail: np.ndarray
    carbon_32l: np.ndarray


class RoutingPOCDecomposition(NamedTuple):
    poc_reservoir: np.ndarray
    flux_poc2doc: np.ndarray
    flux_poc2co2: float


class RoutingCO2ChemistryStep(NamedTuple):
    fast_reservoir: np.ndarray
    slow_reservoir: np.ndarray
    stream_reservoir: np.ndarray
    flood_reservoir: np.ndarray
    pond_reservoir: np.ndarray
    stream_seddep: np.ndarray
    stream_erodep: np.ndarray
    flood_drainage: np.ndarray
    flood_dep_sed: np.ndarray
    flood_dep_poc: np.ndarray
    return_swamp: np.ndarray
    fco2_aq: np.ndarray
    pco2_aq: np.ndarray
    poc_co2_aq: np.ndarray
    poc_doc_aq: np.ndarray
    poc_co2_rivbed: np.ndarray
    poc_doc_rivbed: np.ndarray
    pco2_aq_bas: np.ndarray
    t_water: np.ndarray
    k_co2: np.ndarray
    schmitt: np.ndarray


class RoutingIrrigationStep(NamedTuple):
    fast_reservoir: np.ndarray
    slow_reservoir: np.ndarray
    stream_reservoir: np.ndarray
    irrig_netereq: np.ndarray
    irrig_needs: np.ndarray
    irrig_actual: np.ndarray
    irrig_deficit: np.ndarray
    irrig_adduct: np.ndarray


class RoutingTransportStep(NamedTuple):
    transport: np.ndarray
    water_balance: np.ndarray
    carbon_balance: np.ndarray


class RoutingFlowState(NamedTuple):
    fast_reservoir: np.ndarray
    slow_reservoir: np.ndarray
    stream_reservoir: np.ndarray
    flood_reservoir: np.ndarray
    pond_reservoir: np.ndarray
    lake_reservoir: np.ndarray
    stream_seddep: np.ndarray
    stream_damavail: np.ndarray
    carbon_32l: np.ndarray
    flood_frac: np.ndarray
    flood_frac_bas: np.ndarray
    pond_frac: np.ndarray
    streamfl_frac: np.ndarray
    streamfl_frac_bas: np.ndarray
    flood_diag: np.ndarray
    pond_diag: np.ndarray
    lake_diag: np.ndarray


class RoutingFlowStepResult(NamedTuple):
    state: RoutingFlowState
    outflow: RoutingReservoirOutflowStep
    stream_erosion: RoutingStreamErosionStep
    flood_pond_input: RoutingFloodPondInputStep
    floodplain: RoutingFloodplainFluxStep
    pond: RoutingPondFluxStep
    transport: RoutingTransportStep
    swamp_flood: RoutingSwampFloodStep
    reservoir_update: RoutingReservoirUpdateStep
    area_fractions: RoutingAreaFractionsStep
    chemistry: RoutingCO2ChemistryStep
    return_reinfiltration: RoutingReturnReinfiltrationStep
    irrigation: RoutingIrrigationStep
    diagnostics: RoutingFlowDiagnosticsStep
    lake_overflow: RoutingLakeOverflowStep


class RoutingDailyBoundaryResult(NamedTuple):
    accumulator_step: RoutingAccumulatorStep
    accumulator_state: RoutingAccumulatorState
    flow_state: RoutingFlowState
    flow_step: RoutingFlowStepResult | None
    lake_step: RoutingLakeStep | None
    scaled_outputs: RoutingDailyScaledOutputs | None
    zero_outputs: dict[str, np.ndarray]


class RoutingInitializeMapFlags(NamedTuple):
    init_irrig: bool
    init_flood: bool
    init_swamp: bool
    init_streamsurf: bool


class RoutingIrrigmapPreprocessed(NamedTuple):
    irrigated_frac: np.ndarray
    flood_fracmax: np.ndarray
    headw_frac: np.ndarray
    stream_frac: np.ndarray


class RoutingIrrigmapAggregated(NamedTuple):
    irrigated: np.ndarray
    floodplains: np.ndarray
    swamp: np.ndarray
    headw_area: np.ndarray
    stream_area: np.ndarray
    streamr10th: np.ndarray
    streamr50th: np.ndarray
    streamr90th: np.ndarray
    qflow_ave: np.ndarray
    stream_resave: np.ndarray
    floodh90th: np.ndarray


class RoutingSortCoord(NamedTuple):
    coords: np.ndarray
    nb_out: int


class RoutingGetGrid(NamedTuple):
    nbi: int
    nbj: int
    trip_bx: np.ndarray
    basin_bx: np.ndarray
    area_bx: np.ndarray
    topoind_bx: np.ndarray
    gravel_bx: np.ndarray
    drainarea_bx: np.ndarray
    hierarchy_bx: np.ndarray
    lon_bx: np.ndarray
    lat_bx: np.ndarray
    max_basins: float


class RoutingFindBasins(NamedTuple):
    trip: np.ndarray
    basin: np.ndarray
    nb_basin: int
    basin_inbxid: np.ndarray
    basin_sz: np.ndarray
    basin_bxout: np.ndarray
    basin_pts: np.ndarray
    coast_pts: np.ndarray


class RoutingFindRout(NamedTuple):
    nbout: int
    outflow: np.ndarray
    trip_flow: np.ndarray
    outsz: np.ndarray


class RoutingGlobalize(NamedTuple):
    basin_count: np.ndarray
    basin_area: np.ndarray
    basin_hierarchy: np.ndarray
    basin_topoind: np.ndarray
    basin_gravel: np.ndarray
    basin_drainarea: np.ndarray
    basin_id: np.ndarray
    basin_flowdir: np.ndarray
    outflow_grid: np.ndarray
    nbcoastal: np.ndarray
    coastal_basin: np.ndarray


class RoutingLinkup(NamedTuple):
    outflow_grid: np.ndarray
    outflow_basin: np.ndarray
    inflow_number: np.ndarray
    inflow_grid: np.ndarray
    inflow_basin: np.ndarray


class RoutingFetch(NamedTuple):
    basin_area: np.ndarray
    outflow_grid: np.ndarray
    fetch_basin: np.ndarray


class RoutingTruncateFinalized(NamedTuple):
    routing_area: np.ndarray
    topo_resid: np.ndarray
    basgravel: np.ndarray
    basdrainarea: np.ndarray
    global_basinid: np.ndarray
    route_togrid: np.ndarray
    route_tobasin: np.ndarray


class RoutingKillBas(NamedTuple):
    basin_count: np.ndarray
    basin_area: np.ndarray
    basin_topoind: np.ndarray
    basin_gravel: np.ndarray
    basin_drainarea: np.ndarray
    fetch_basin: np.ndarray
    basin_id: np.ndarray
    basin_flowdir: np.ndarray
    outflow_grid: np.ndarray
    outflow_basin: np.ndarray
    inflow_number: np.ndarray
    inflow_grid: np.ndarray
    inflow_basin: np.ndarray


class RoutingBasinsPostAggregate(NamedTuple):
    routing_area: np.ndarray
    topo_resid: np.ndarray
    basgravel: np.ndarray
    basdrainarea: np.ndarray
    global_basinid: np.ndarray
    route_togrid: np.ndarray
    route_tobasin: np.ndarray
    basin_count: np.ndarray


class RoutingMapFields(NamedTuple):
    lon_rel: np.ndarray
    lat_rel: np.ndarray
    trip: np.ndarray
    basins: np.ndarray
    topoindex: np.ndarray
    gravel: np.ndarray
    drainarea: np.ndarray


class RoutingSimplify(NamedTuple):
    trip: np.ndarray


class RoutingCutBasin(NamedTuple):
    trip: np.ndarray
    nb: int
    bname: np.ndarray
    sz: np.ndarray
    pts: np.ndarray


def routing_zero_outputs(npts: int, nflow: int, nctext: int = 3, ncarb: int = 3) -> dict[str, np.ndarray]:
    """Return the pre-routing zero outputs from ``routing_main`` lines 1000-1011."""

    return {
        "returnflow": np.zeros((npts, nflow), dtype=np.float64),
        "reinfiltration": np.zeros((npts, nflow), dtype=np.float64),
        "irrigation": np.zeros((npts, nflow), dtype=np.float64),
        "riverflow": np.zeros((npts, nflow), dtype=np.float64),
        "coastalflow": np.zeros((npts, nflow), dtype=np.float64),
        "sed_depositiontot": np.zeros((npts, nctext), dtype=np.float64),
        "poc_depositiontot": np.zeros((npts, ncarb), dtype=np.float64),
        "stream_inflow": np.zeros((npts, nflow), dtype=np.float64),
        "stream_outflow": np.zeros((npts, nflow), dtype=np.float64),
        "stream_frac": np.zeros((npts,), dtype=np.float64),
        "flood_res": np.zeros((npts,), dtype=np.float64),
        "fastr": np.zeros((npts,), dtype=np.float64),
    }


def routing_setvar_no_keyword(var, default, *, val_exp=999999.0):
    """Apply the Fortran ``setvar_p(..., 'NO_KEYWORD', default)`` fallback.

    Fortran provenance: ``sechiba_io.f90`` scalar/array `setvar` overloads
    lines 244-441. Array overloads replace with the default only when every
    element equals `val_exp`; partial exceptional values are preserved.
    """

    arr = np.asarray(var, dtype=np.float64).copy()
    if np.all(arr == float(val_exp)):
        return np.zeros_like(arr, dtype=np.float64) + np.asarray(default, dtype=np.float64)
    return arr


def routing_init_restart_defaults(
    restart: dict[str, np.ndarray] | None,
    *,
    npts: int,
    nbas: int,
    nflow: int,
    routing_area,
    doirrigation: bool = False,
    val_exp=999999.0,
    undef_sechiba=1.0e20,
) -> dict[str, np.ndarray]:
    """Build source-backed routing restart defaults without doing file I/O.

    Fortran provenance: ``routing.f90::routing_init`` lines 2163-2459 and
    2530-2710. The source reads restart variables, applies
    ``setvar_p(..., 'NO_KEYWORD', default)`` fallbacks, forces
    ``stream_seddep(:,:,ih2o:ico2aq)=0``, zeros drainage/precipitation slots
    above `ico2aq`, keeps map/forcing climatology fields at `undef_sechiba`
    when missing, and derives reservoir diagnostics by dividing basin sums by
    total routing area.
    """

    restart = {} if restart is None else restart
    area = np.asarray(routing_area, dtype=np.float64)
    if area.shape != (npts, nbas):
        raise ValueError("routing_area must have shape (npts,nbas)")
    total_area = np.sum(area, axis=1)
    if np.any(total_area <= 0.0):
        raise ValueError("routing_init restart defaults require positive total routing area")

    def read(name: str, shape: tuple[int, ...], default: float, *, force_zero: bool = False) -> np.ndarray:
        if force_zero:
            return np.zeros(shape, dtype=np.float64)
        value = np.asarray(restart.get(name, np.full(shape, float(val_exp), dtype=np.float64)), dtype=np.float64)
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        return routing_setvar_no_keyword(value, default, val_exp=val_exp)

    state: dict[str, np.ndarray] = {}
    basin_flow = (npts, nbas, nflow)
    grid_flow = (npts, nflow)
    for name in ("fast_reservoir", "slow_reservoir", "stream_reservoir", "stream_seddep", "flood_reservoir"):
        state[name] = read(name, basin_flow, 0.0)
    state["stream_seddep"][:, :, IH2O : ICO2AQ + 1] = 0.0

    for name in (
        "lake_reservoir",
        "pond_reservoir",
        "lakeinflow_mean",
        "returnflow_mean",
        "reinfiltration_mean",
        "riverflow_mean",
        "coastalflow_mean",
        "hydrographs",
    ):
        state[name] = read(name, grid_flow, 0.0)
    state["irrigation_mean"] = read("irrigation_mean", grid_flow, 0.0, force_zero=not bool(doirrigation))

    for name in ("poc_deposition_mean", "riv_erodep_poc", "rivbed2fld_poc"):
        state[name] = read(name, (npts, 3), 0.0)
    for name in ("sed_deposition_mean", "riv_erodep_sed", "rivbed2fld_sed"):
        state[name] = read(name, (npts, 3), 0.0)
    state["rivchannel_deposition"] = read("rivchannel_deposition", grid_flow, 0.0)
    state["stream_inflow"] = read("stream_inflow", grid_flow, 0.0)
    state["stream_outflow"] = read("stream_outflow", grid_flow, 0.0)
    state["flood_daily"] = read("flood_daily", (npts,), 0.0)

    for name in ("flood_frac_bas", "streamfl_frac_bas", "stream_area_bas"):
        state[name] = read(name, (npts, nbas), 0.0)
    for name in ("flood_height", "pond_frac", "flood_frac", "streamfl_frac", "flood_res", "fastr", "slowflow_diag", "floodout_mean"):
        state[name] = read(name, (npts,), 0.0)
    for name in ("runoff_mean", "drainage_mean", "precip_mean"):
        state[name] = read(name, grid_flow, 0.0)
    state["drainage_mean"][:, ICO2AQ + 1 : nflow] = 0.0
    state["precip_mean"][:, ICO2AQ + 1 : nflow] = 0.0

    for name in ("transpot_mean", "k_litt_mean", "temp_sol_mean", "totnobio_mean"):
        state[name] = read(name, (npts,), 0.0)
    state["humrel_mean"] = read("humrel_mean", (npts,), 1.0)
    state["vegtot_mean"] = read("vegtot_mean", (npts,), 1.0)

    for name in (
        "irrigated",
        "floodplains",
        "swamp",
        "headw_area",
        "stream_area",
        "streamr10th",
        "streamr50th",
        "streamr90th",
        "floodh90th",
        "qflow_ave",
        "stream_damavail",
        "damreservoir_tot",
        "stream_resave",
    ):
        state[name] = read(name, (npts,), float(undef_sechiba))

    state["fast_diag"] = np.sum(state["fast_reservoir"], axis=1) / total_area[:, None]
    state["slow_diag"] = np.sum(state["slow_reservoir"], axis=1) / total_area[:, None]
    state["stream_diag"] = np.sum(state["stream_reservoir"], axis=1) / total_area[:, None]
    state["flood_diag"] = np.sum(state["flood_reservoir"], axis=1) / total_area[:, None]
    state["lake_diag"] = state["lake_reservoir"] / total_area[:, None]
    state["pond_diag"] = np.zeros(grid_flow, dtype=np.float64)
    state["delsurfstor"] = np.zeros((npts,), dtype=np.float64)
    state["fco2_aq"] = np.zeros((npts, NAQSYS), dtype=np.float64)
    state["pco2_aq"] = np.zeros((npts, NAQSYS), dtype=np.float64)
    state["poc_co2_aq"] = np.zeros((npts, NAQSYS), dtype=np.float64)
    state["poc_doc_aq"] = np.zeros((npts, NAQSYS), dtype=np.float64)
    state["poc_co2_rivbed"] = np.zeros((npts,), dtype=np.float64)
    state["poc_doc_rivbed"] = np.zeros((npts, 2), dtype=np.float64)
    state["flow_input"] = np.zeros((npts, 3, nflow), dtype=np.float64)
    return state


def routing_initialize_map_flags(
    *,
    irrigated,
    floodplains,
    swamp,
    headw_area,
    stream_area,
    streamr50th,
    do_irrigation: bool,
    do_floodplains: bool,
    doswamps: bool,
    ok_doc: bool,
    undef_sechiba=1.0e20,
) -> RoutingInitializeMapFlags:
    """Return the map interpolation flags from ``routing_initialize``.

    Fortran provenance: ``routing.f90::routing_initialize`` lines 594-621.
    The source re-reads the irrigation/flood/swamp/stream-surface maps only
    when enabled switches see missing restart fields. The floodplain branch
    also forces stream-surface initialization when `streamr50th` is nonpositive.
    """

    irrigated = np.asarray(irrigated, dtype=np.float64)
    floodplains = np.asarray(floodplains, dtype=np.float64)
    swamp = np.asarray(swamp, dtype=np.float64)
    headw_area = np.asarray(headw_area, dtype=np.float64)
    stream_area = np.asarray(stream_area, dtype=np.float64)
    streamr50th = np.asarray(streamr50th, dtype=np.float64)
    init_irrig = False
    init_flood = False
    init_swamp = False
    init_streamsurf = False

    if bool(do_irrigation) and np.count_nonzero(irrigated >= float(undef_sechiba) - 1.0) > 0:
        init_irrig = True
    if bool(do_floodplains):
        if np.count_nonzero(floodplains >= float(undef_sechiba) - 1.0) > 0:
            init_flood = True
        if np.max(streamr50th) <= 0.0:
            init_flood = True
            init_streamsurf = True
    if bool(doswamps) and np.count_nonzero(swamp >= float(undef_sechiba) - 1.0) > 0:
        init_swamp = True
    if bool(ok_doc) and np.count_nonzero((headw_area + stream_area) >= float(undef_sechiba) - 1.0) > 0:
        init_streamsurf = True

    return RoutingInitializeMapFlags(
        init_irrig=init_irrig,
        init_flood=init_flood,
        init_swamp=init_swamp,
        init_streamsurf=init_streamsurf,
    )


def routing_irrigmap_preprocess(
    *,
    irrigated_frac,
    flood_fracmax,
    headw_frac,
    stream_frac,
    undef_sechiba=1.0e20,
) -> RoutingIrrigmapPreprocessed:
    """Apply the map percent-to-fraction thresholds from ``routing_irrigmap``.

    Fortran provenance: ``routing.f90::routing_irrigmap`` lines 10885-10911.
    Valid irrigation values are divided by 100 and values below 0.005 become
    zero. Valid flood-type values are divided by 100 and values below 0.0001
    become zero. Valid headwater and stream fractions are divided by 100 and
    floored to 0.0001. Missing values near `undef_sechiba` are preserved.
    """

    irrig = np.asarray(irrigated_frac, dtype=np.float64).copy()
    flood = np.asarray(flood_fracmax, dtype=np.float64).copy()
    headw = np.asarray(headw_frac, dtype=np.float64).copy()
    stream = np.asarray(stream_frac, dtype=np.float64).copy()
    if flood.ndim != 3 or flood.shape[2] < 6:
        raise ValueError("flood_fracmax must have shape (iml,jml,ntype>=6)")

    valid = irrig < float(undef_sechiba) - 1.0
    irrig[valid] = irrig[valid] / 100.0
    irrig[valid & (irrig < 0.005)] = 0.0

    valid = flood < float(undef_sechiba) - 1.0
    flood[valid] = flood[valid] / 100.0
    flood[valid & (flood < 0.0001)] = 0.0

    valid = headw < float(undef_sechiba) - 1.0
    headw[valid] = headw[valid] / 100.0
    headw[valid & (headw < 0.0001)] = 0.0001

    valid = stream < float(undef_sechiba) - 1.0
    stream[valid] = stream[valid] / 100.0
    stream[valid & (stream < 0.0001)] = 0.0001

    return RoutingIrrigmapPreprocessed(
        irrigated_frac=irrig,
        flood_fracmax=flood,
        headw_frac=headw,
        stream_frac=stream,
    )


def routing_irrigmap_aggregate(
    *,
    irrsub_area,
    irrsub_index,
    resolution,
    contfrac,
    irrigated_frac,
    flood_fracmax,
    headw_frac,
    stream_frac,
    streamr10th_mm,
    streamr50th_mm,
    streamr90th_mm,
    qflow_aveb,
    stream_resaveb,
    floodh90th_mm,
    init_irrig: bool,
    init_flood: bool,
    init_swamp: bool,
    init_streamsurf: bool,
    dostreamswell: bool,
    new_flood_scheme: bool,
    ok_damreservoir: bool,
    irrigated=None,
    floodplains=None,
    swamp=None,
    headw_area=None,
    stream_area=None,
    streamr10th=None,
    streamr50th=None,
    streamr90th=None,
    qflow_ave=None,
    stream_resave=None,
    floodh90th=None,
    streamfld_scaler=1.0,
    floodcri=0.0,
    min_sechiba=1.0e-8,
    undef_sechiba=1.0e20,
    index_base: int = 1,
) -> RoutingIrrigmapAggregated:
    """Aggregate preprocessed irrigation/flood/stream maps to model land points.

    Fortran provenance: ``routing.f90::routing_irrigmap`` lines 11012-11204.
    This helper starts after `aggregate_p`: negative sub-areas are zeroed, the
    first `COUNT(irrsub_area(ib,:) > zero)` entries are accumulated in source
    order, init flags control which output fields are overwritten, floodplains
    are forced to exceed stream area by `min_sechiba` when stream surfaces are
    initialized, and `qflow_ave`/`stream_resave` are summed without area weights
    as in the source.
    """

    sub_area = np.asarray(irrsub_area, dtype=np.float64).copy()
    sub_area[sub_area < 0.0] = 0.0
    sub_index = np.asarray(irrsub_index, dtype=np.int64)
    resolution = np.asarray(resolution, dtype=np.float64)
    contfrac = np.asarray(contfrac, dtype=np.float64)
    irrig_frac = np.asarray(irrigated_frac, dtype=np.float64)
    flood_frac = np.asarray(flood_fracmax, dtype=np.float64)
    headw_frac = np.asarray(headw_frac, dtype=np.float64)
    stream_frac = np.asarray(stream_frac, dtype=np.float64)
    streamr10th_mm = np.asarray(streamr10th_mm, dtype=np.float64)
    streamr50th_mm = np.asarray(streamr50th_mm, dtype=np.float64)
    streamr90th_mm = np.asarray(streamr90th_mm, dtype=np.float64)
    qflow_aveb = np.asarray(qflow_aveb, dtype=np.float64)
    stream_resaveb = np.asarray(stream_resaveb, dtype=np.float64)
    floodh90th_mm = np.asarray(floodh90th_mm, dtype=np.float64)

    if sub_area.ndim != 2 or sub_index.shape != sub_area.shape + (2,):
        raise ValueError("irrsub_area must be (npts,nbpmax) and irrsub_index (npts,nbpmax,2)")
    npts = sub_area.shape[0]
    if resolution.shape != (npts, 2) or contfrac.shape != (npts,):
        raise ValueError("resolution must be (npts,2) and contfrac must be (npts,)")
    if flood_frac.ndim != 3 or flood_frac.shape[2] < 6:
        raise ValueError("flood_fracmax must have shape (iml,jml,ntype>=6)")

    def out(value):
        if value is None:
            return np.zeros((npts,), dtype=np.float64)
        arr = np.asarray(value, dtype=np.float64).copy()
        if arr.shape != (npts,):
            raise ValueError("initial map outputs must have shape (npts,)")
        return arr

    out_irrigated = out(irrigated)
    out_floodplains = out(floodplains)
    out_swamp = out(swamp)
    out_headw_area = out(headw_area)
    out_stream_area = out(stream_area)
    out_streamr10th = out(streamr10th)
    out_streamr50th = out(streamr50th)
    out_streamr90th = out(streamr90th)
    out_qflow_ave = out(qflow_ave)
    out_stream_resave = out(stream_resave)
    out_floodh90th = out(floodh90th)

    ilake, idam, iflood, iswamp, isal, ipond = range(6)
    del ilake, ipond
    valid_limit = float(undef_sechiba) - 1.0

    for ib in range(npts):
        cell_area = resolution[ib, 0] * resolution[ib, 1] * contfrac[ib]
        if cell_area <= 0.0 and (init_irrig or init_flood or init_swamp or init_streamsurf):
            raise ValueError("routing_irrigmap aggregation requires positive grid-cell land area")

        area_irrig = 0.0
        area_flood = np.zeros((6,), dtype=np.float64)
        area_stream = 0.0
        area_headw = 0.0
        streamr50th_kg = 0.0
        streamr10th_kg = 0.0
        streamr90th_kg = 0.0
        floodh90th_loc = 0.0
        qflow_aveg = 0.0
        stream_resaveg = 0.0

        npositive = int(np.count_nonzero(sub_area[ib, :] > 0.0))
        for fopt in range(npositive):
            ip = int(sub_index[ib, fopt, 0]) - int(index_base)
            jp = int(sub_index[ib, fopt, 1]) - int(index_base)
            if ip < 0 or jp < 0 or ip >= irrig_frac.shape[0] or jp >= irrig_frac.shape[1]:
                raise ValueError("irrsub_index points outside map arrays")
            area = sub_area[ib, fopt]

            if irrig_frac[ip, jp] < valid_limit:
                area_irrig += area * irrig_frac[ip, jp]
            for itype in range(6):
                if flood_frac[ip, jp, itype] < valid_limit:
                    area_flood[itype] += area * flood_frac[ip, jp, itype]
            if headw_frac[ip, jp] < valid_limit:
                area_headw += area * headw_frac[ip, jp]
            if stream_frac[ip, jp] < valid_limit:
                area_stream += area * stream_frac[ip, jp]
            if bool(dostreamswell):
                if streamr10th_mm[ip, jp] < valid_limit:
                    streamr10th_kg += area * streamr10th_mm[ip, jp]
                if streamr90th_mm[ip, jp] < valid_limit:
                    streamr90th_kg += area * streamr90th_mm[ip, jp] * float(streamfld_scaler)
            if bool(new_flood_scheme):
                if streamr50th_mm[ip, jp] < valid_limit:
                    streamr50th_kg += area * streamr50th_mm[ip, jp] * float(streamfld_scaler)
                if floodh90th_mm[ip, jp] < valid_limit:
                    floodh90th_loc += area * floodh90th_mm[ip, jp]
            if qflow_aveb[ip, jp] < valid_limit:
                qflow_aveg += qflow_aveb[ip, jp]
            if stream_resaveb[ip, jp] < valid_limit:
                stream_resaveg += stream_resaveb[ip, jp]

        if bool(init_irrig):
            out_irrigated[ib] = min(area_irrig, cell_area)
            if out_irrigated[ib] < 0.0:
                raise ValueError("routing_irrigmap produced negative irrigated area")

        if bool(init_flood):
            out_floodplains[ib] = min(area_flood[iflood] + area_flood[idam] + area_flood[isal], cell_area)
            if out_floodplains[ib] < 0.0:
                raise ValueError("routing_irrigmap produced negative floodplain area")
            if bool(new_flood_scheme):
                out_floodh90th[ib] = max(floodh90th_loc / cell_area, float(floodcri))
            else:
                out_floodh90th[ib] = 0.0
            if out_floodh90th[ib] < 0.0:
                raise ValueError("routing_irrigmap produced negative floodh90th")

        if bool(init_swamp):
            out_swamp[ib] = min(area_flood[iswamp], cell_area)
            if out_swamp[ib] < 0.0:
                raise ValueError("routing_irrigmap produced negative swamp area")

        if bool(init_streamsurf):
            out_headw_area[ib] = min(area_headw, cell_area)
            out_stream_area[ib] = min(area_stream, cell_area)
            if out_headw_area[ib] < 0.0 or out_stream_area[ib] < 0.0:
                raise ValueError("routing_irrigmap produced negative stream surface area")
            out_floodplains[ib] = max(out_floodplains[ib], out_stream_area[ib] + float(min_sechiba))
            if bool(new_flood_scheme):
                out_streamr50th[ib] = max(streamr50th_kg, 0.0)
            else:
                out_streamr50th[ib] = 0.0
            if bool(dostreamswell):
                out_streamr90th[ib] = max(streamr90th_kg, 0.0)
                out_streamr10th[ib] = max(streamr10th_kg, 0.0)
            else:
                out_streamr90th[ib] = 0.0
                out_streamr10th[ib] = 0.0
            out_qflow_ave[ib] = max(qflow_aveg, 0.0)
            out_stream_resave[ib] = max(stream_resaveg, 0.0) if bool(ok_damreservoir) else 1.0

    return RoutingIrrigmapAggregated(
        irrigated=out_irrigated,
        floodplains=out_floodplains,
        swamp=out_swamp,
        headw_area=out_headw_area,
        stream_area=out_stream_area,
        streamr10th=out_streamr10th,
        streamr50th=out_streamr50th,
        streamr90th=out_streamr90th,
        qflow_ave=out_qflow_ave,
        stream_resave=out_stream_resave,
        floodh90th=out_floodh90th,
    )


def routing_initialize_stream_fraction(*, stream_area, routing_area):
    """Compute initialization `stream_frac` from stream area and routing area.

    Fortran provenance: ``routing.f90::routing_initialize`` lines 641-647.
    The fraction uses `stream_area` only and is capped at one; zero total
    routing area gives a zero fraction.
    """

    stream_area = np.asarray(stream_area, dtype=np.float64)
    routing_area = np.asarray(routing_area, dtype=np.float64)
    if routing_area.ndim != 2 or routing_area.shape[0] != stream_area.shape[0]:
        raise ValueError("routing_area must be (npts,nbas) and match stream_area")
    total_area = np.sum(routing_area, axis=1)
    frac = np.zeros_like(stream_area, dtype=np.float64)
    active = total_area > 0.0
    frac[active] = np.minimum(stream_area[active] / total_area[active], 1.0)
    return frac


def routing_sortcoord(coords, direction: str, *, undef_sechiba=1.0e20) -> RoutingSortCoord:
    """Compress and sort grid coordinates like ``routing_sortcoord``.

    Fortran provenance: ``routing.f90::routing_sortcoord`` lines 7728-7813.
    Duplicate coordinates are removed by the source shifting loop, longitude
    sorting uses modulo-360 when any absolute longitude exceeds 160 degrees,
    and unused tail slots are filled with zero.
    """

    out = np.asarray(coords, dtype=np.float64).copy()
    if out.ndim != 1:
        raise ValueError("coords must be one-dimensional")
    nb_in = out.shape[0]
    nb_out = nb_in
    ipos = 0
    while ipos < nb_in - 1:
        if out[ipos + 1] != float(undef_sechiba):
            if np.count_nonzero(out[ipos:nb_out] == out[ipos]) > 1:
                out[ipos : nb_out - 1] = out[ipos + 1 : nb_out]
                out[nb_out - 1 : nb_in] = float(undef_sechiba)
                nb_out -= 1
            else:
                ipos += 1
        else:
            break

    direction = str(direction)
    if direction.startswith(("WE", "EW")):
        sort_values = out[:nb_out].copy()
        if np.max(np.abs(out[:nb_out])) > 160.0:
            sort_values = np.mod(sort_values + 360.0, 360.0)
    elif direction.startswith(("NS", "SN")):
        sort_values = out[:nb_out].copy()
    else:
        raise ValueError("routing_sortcoord direction must start with WE, EW, NS, or SN")

    if direction.startswith(("WE", "SN")):
        order = np.argsort(sort_values, kind="stable")
    elif direction.startswith(("EW", "NS")):
        order = np.argsort(-sort_values, kind="stable")
    else:
        raise ValueError("routing_sortcoord direction must start with WE, EW, NS, or SN")

    sorted_coords = out.copy()
    sorted_coords[:nb_out] = out[:nb_out][order]
    if nb_out < nb_in:
        sorted_coords[nb_out:nb_in] = 0.0
    return RoutingSortCoord(coords=sorted_coords, nb_out=int(nb_out))


def read_routing_map_fields(path) -> RoutingMapFields:
    """Read `routing.nc` fields in Fortran `(longitude, latitude)` order.

    Fortran provenance: ``routing.f90::routing_basins`` lines 7108-7183:
    `flininfo`/`flinopen` establish the high-resolution dimensions and
    longitude/latitude arrays, then `flinget` reads `trip`, `basins`,
    `topoind`, `gravel`, and `drainage_area`. This reader is intentionally
    strict: missing variables are reported instead of filled with defaults.
    """

    import xarray as xr

    required = ("nav_lon", "nav_lat", "trip", "basins", "topoind", "gravel", "drainage_area")
    with xr.open_dataset(path, decode_times=False) as ds:
        missing = [name for name in required if name not in ds]
        if missing:
            raise ValueError(f"routing map is missing required variables: {', '.join(missing)}")

        def as_fortran_2d(name: str) -> np.ndarray:
            values = np.asarray(ds[name].values, dtype=np.float64)
            if values.ndim != 2:
                raise ValueError(f"routing map variable {name} must be two-dimensional")
            return values.T

        return RoutingMapFields(
            lon_rel=as_fortran_2d("nav_lon"),
            lat_rel=as_fortran_2d("nav_lat"),
            trip=as_fortran_2d("trip"),
            basins=as_fortran_2d("basins"),
            topoindex=as_fortran_2d("topoind"),
            gravel=as_fortran_2d("gravel"),
            drainarea=as_fortran_2d("drainage_area"),
        )


def routing_map_fields_for_domain(*, river_routing: bool, nbp_glo: int, routing_file) -> RoutingMapFields | None:
    """Load routing map fields only when the Fortran routing branch is active.

    Fortran provenance: ``sechiba.f90`` lines 1227-1234 call
    ``routing_main`` only when `river_routing .AND. nbp_glo .GT. 1`, while the
    no-routing path remains active for the paper single land point even when
    `RIVER_ROUTING=TRUE`. When the branch is active this helper enforces the
    strict `routing.f90::routing_basins` NetCDF contract through
    `read_routing_map_fields`.
    """

    if bool(river_routing) and int(nbp_glo) > 1:
        return read_routing_map_fields(routing_file)
    return None


def routing_hierarchy(*, trip, topoindex, undef_sechiba=1.0e20) -> np.ndarray:
    """Compute cumulative topographic hierarchy along high-resolution trips.

    Fortran provenance: ``routing.f90::routing_hierarchy`` lines 8539-8627.
    For each finite routing cell the source follows directions 1-8, wrapping
    only the first grid dimension, accumulates `topoindex` along the flowline,
    rejects very large topographic indices, rejects decreasing cumulative
    hierarchy, and raises when the path does not reach an outlet before the
    source loop limit.
    """

    trip_arr = np.asarray(trip, dtype=np.float64)
    topo = np.asarray(topoindex, dtype=np.float64)
    if trip_arr.shape != topo.shape or trip_arr.ndim != 2:
        raise ValueError("trip and topoindex must be two-dimensional arrays with the same shape")
    iml, jml = trip_arr.shape
    hierarchy = np.full_like(topo, float(undef_sechiba), dtype=np.float64)
    inc = np.asarray(
        [
            [0, -1],
            [1, -1],
            [1, 0],
            [1, 1],
            [0, 1],
            [-1, 1],
            [-1, 0],
            [-1, -1],
        ],
        dtype=np.int64,
    )

    for ip in range(iml):
        for jp in range(jml):
            if trip_arr[ip, jp] < float(undef_sechiba):
                ntripi = ip
                ntripj = jp
                trp = int(np.rint(trip_arr[ip, jp]))
                cnt = 1
                if topo[ip, jp] > 1.0e10:
                    raise ValueError("routing_hierarchy found too large topographic index")
                topohier = float(topo[ip, jp])
                while trp > 0 and trp < 9 and cnt < iml * jml:
                    cnt += 1
                    ntripi += int(inc[trp - 1, 0])
                    if ntripi < 0:
                        ntripi = iml - 1
                    if ntripi >= iml:
                        ntripi = 0
                    ntripj += int(inc[trp - 1, 1])
                    if ntripj < 0 or ntripj >= jml:
                        raise ValueError("routing_hierarchy trip path left the latitude dimension")
                    topohier_old = topohier
                    topohier += float(topo[ntripi, ntripj])
                    if topohier_old > topohier:
                        raise ValueError("routing_hierarchy cumulative topography decreased")
                    trp = int(np.rint(trip_arr[ntripi, ntripj]))
                if cnt == iml * jml:
                    raise ValueError("routing_hierarchy could not route point")
                hierarchy[ip, jp] = topohier

    return hierarchy


def routing_getgrid_from_subgrid(
    *,
    ib: int,
    sub_pts,
    sub_index,
    sub_area,
    lon_rel,
    lat_rel,
    lalo,
    resolution,
    contfrac,
    trip,
    basins,
    topoindex,
    gravel,
    drainarea,
    hierarchy,
    max_basins: float,
    min_topoind: float,
    index_base: int = 1,
    undef_sechiba=1.0e20,
    undef_int=999999999,
) -> RoutingGetGrid:
    """Extract and boundary-tag one coarse routing grid box.

    Fortran provenance: ``routing.f90::routing_getgrid`` lines 7547-7709. This
    helper starts from explicit `aggregate` outputs: it orders source cells west
    to east and north to south, maps each source cell to its local box
    coordinate, invents a coastal cell when `sub_pts(ib)==0`, tags edge flows
    leaving the coarse cell as 101-108, and normalizes diagonal corner outflows
    to the cardinal boundary direction used by later basin linking.
    """

    sub_pts = np.asarray(sub_pts, dtype=np.int64)
    sub_index = np.asarray(sub_index, dtype=np.int64)
    sub_area = np.asarray(sub_area, dtype=np.float64)
    lon_rel = np.asarray(lon_rel, dtype=np.float64)
    lat_rel = np.asarray(lat_rel, dtype=np.float64)
    lalo = np.asarray(lalo, dtype=np.float64)
    resolution = np.asarray(resolution, dtype=np.float64)
    contfrac = np.asarray(contfrac, dtype=np.float64)
    trip = np.asarray(trip, dtype=np.float64)
    basins = np.asarray(basins, dtype=np.float64)
    topoindex = np.asarray(topoindex, dtype=np.float64)
    gravel = np.asarray(gravel, dtype=np.float64)
    drainarea = np.asarray(drainarea, dtype=np.float64)
    hierarchy = np.asarray(hierarchy, dtype=np.float64)

    ib0 = int(ib) - int(index_base)
    if ib0 < 0 or ib0 >= sub_pts.shape[0]:
        raise ValueError("ib points outside sub_pts")
    nsrc = int(sub_pts[ib0])

    if nsrc > 0:
        lon_values = np.zeros((nsrc,), dtype=np.float64)
        lat_values = np.zeros((nsrc,), dtype=np.float64)
        for ipt in range(nsrc):
            im = int(sub_index[ib0, ipt, 0]) - int(index_base)
            jm = int(sub_index[ib0, ipt, 1]) - int(index_base)
            lon_values[ipt] = lon_rel[im, jm]
            lat_values[ipt] = lat_rel[im, jm]

        lon_sorted = routing_sortcoord(lon_values, "WE", undef_sechiba=undef_sechiba)
        lat_sorted = routing_sortcoord(lat_values, "NS", undef_sechiba=undef_sechiba)
        nbi = lon_sorted.nb_out
        nbj = lat_sorted.nb_out

        trip_bx = np.full((nbi, nbj), int(undef_int), dtype=np.int64)
        basin_bx = np.full((nbi, nbj), int(undef_int), dtype=np.int64)
        area_bx = np.full((nbi, nbj), float(undef_sechiba), dtype=np.float64)
        topoind_bx = np.full((nbi, nbj), float(undef_sechiba), dtype=np.float64)
        gravel_bx = np.full((nbi, nbj), float(undef_sechiba), dtype=np.float64)
        drainarea_bx = np.full((nbi, nbj), float(undef_sechiba), dtype=np.float64)
        hierarchy_bx = np.full((nbi, nbj), float(undef_sechiba), dtype=np.float64)
        lon_bx = np.zeros((nbi, nbj), dtype=np.float64)
        lat_bx = np.zeros((nbi, nbj), dtype=np.float64)

        for ipt in range(nsrc):
            im = int(sub_index[ib0, ipt, 0]) - int(index_base)
            jm = int(sub_index[ib0, ipt, 1]) - int(index_base)
            iloc = int(np.argmin(np.abs(lon_sorted.coords[:nbi] - lon_rel[im, jm])))
            jloc = int(np.argmin(np.abs(lat_sorted.coords[:nbj] - lat_rel[im, jm])))
            trip_bx[iloc, jloc] = int(np.rint(trip[im, jm]))
            basin_bx[iloc, jloc] = int(np.rint(basins[im, jm]))
            area_bx[iloc, jloc] = sub_area[ib0, ipt]
            topoind_bx[iloc, jloc] = topoindex[im, jm]
            gravel_bx[iloc, jloc] = gravel[im, jm]
            drainarea_bx[iloc, jloc] = drainarea[im, jm]
            hierarchy_bx[iloc, jloc] = hierarchy[im, jm]
            lon_bx[iloc, jloc] = lon_rel[im, jm]
            lat_bx[iloc, jloc] = lat_rel[im, jm]
    else:
        nbi = 1
        nbj = 1
        trip_bx = np.asarray([[98]], dtype=np.int64)
        basin_bx = np.asarray([[int(np.rint(max_basins + 1.0))]], dtype=np.int64)
        max_basins = max_basins + 1.0
        area_bx = np.asarray([[resolution[ib0, 0] * resolution[ib0, 1] * contfrac[ib0]]], dtype=np.float64)
        topoind_bx = np.asarray([[float(min_topoind)]], dtype=np.float64)
        gravel_bx = np.asarray([[0.0]], dtype=np.float64)
        drainarea_bx = np.asarray([[0.0]], dtype=np.float64)
        hierarchy_bx = np.asarray([[float(min_topoind)]], dtype=np.float64)
        lon_bx = np.asarray([[lalo[ib0, 1]]], dtype=np.float64)
        lat_bx = np.asarray([[lalo[ib0, 0]]], dtype=np.float64)

    for jp in range(nbj):
        if trip_bx[0, jp] in (8, 7, 6):
            trip_bx[0, jp] += 100
        if trip_bx[nbi - 1, jp] in (2, 3, 4):
            trip_bx[nbi - 1, jp] += 100
    for ip in range(nbi):
        if trip_bx[ip, 0] in (8, 1, 2):
            trip_bx[ip, 0] += 100
        if trip_bx[ip, nbj - 1] in (6, 5, 4):
            trip_bx[ip, nbj - 1] += 100

    if trip_bx[0, 0] == 102:
        trip_bx[0, 0] = 101
    if trip_bx[nbi - 1, 0] == 108:
        trip_bx[nbi - 1, 0] = 101
    for ip in range(1, nbi - 1):
        if trip_bx[ip, 0] in (108, 102):
            trip_bx[ip, 0] = 101

    if trip_bx[0, nbj - 1] == 104:
        trip_bx[0, nbj - 1] = 105
    if trip_bx[nbi - 1, nbj - 1] == 106:
        trip_bx[nbi - 1, nbj - 1] = 105
    for ip in range(1, nbi - 1):
        if trip_bx[ip, nbj - 1] in (104, 106):
            trip_bx[ip, nbj - 1] = 105

    if trip_bx[nbi - 1, 0] == 104:
        trip_bx[nbi - 1, 0] = 103
    if trip_bx[nbi - 1, nbj - 1] == 102:
        trip_bx[nbi - 1, nbj - 1] = 103
    for jp in range(1, nbj - 1):
        if trip_bx[nbi - 1, jp] in (104, 102):
            trip_bx[nbi - 1, jp] = 103

    if trip_bx[0, 0] == 106:
        trip_bx[0, 0] = 107
    if trip_bx[0, nbj - 1] == 108:
        trip_bx[0, nbj - 1] = 107
    for jp in range(1, nbj - 1):
        if trip_bx[0, jp] in (106, 108):
            trip_bx[0, jp] = 107

    return RoutingGetGrid(
        nbi=int(nbi),
        nbj=int(nbj),
        trip_bx=trip_bx,
        basin_bx=basin_bx,
        area_bx=area_bx,
        topoind_bx=topoind_bx,
        gravel_bx=gravel_bx,
        drainarea_bx=drainarea_bx,
        hierarchy_bx=hierarchy_bx,
        lon_bx=lon_bx,
        lat_bx=lat_bx,
        max_basins=float(max_basins),
    )


def routing_findbasins_simple(
    *,
    trip,
    basin,
    undef_int=999999999,
) -> RoutingFindBasins:
    """Find local routing basins for source cases that need no split/merge solve.

    Fortran provenance: ``routing.f90::routing_findbasins`` lines 7862-8121.
    This source-backed subset covers initial basin collection, singleton river
    outflow-to-ocean conversion and coastal singleton merging, outflow direction
    retention, size sorting, and the source sanity check. If a multi-cell basin
    has multiple outflows and would require `routing_simplify` or
    `routing_cutbasin`, this helper raises explicitly instead of approximating.
    """

    trip_arr = np.asarray(trip, dtype=np.int64).copy()
    basin_arr = np.asarray(basin, dtype=np.int64).copy()
    if trip_arr.shape != basin_arr.shape or trip_arr.ndim != 2:
        raise ValueError("trip and basin must be two-dimensional arrays with the same shape")
    nbi, nbj = trip_arr.shape
    max_pts = nbi * nbj

    bname: list[int] = []
    pts: list[list[tuple[int, int]]] = []
    nbout: list[int] = []
    for ip in range(nbi):
        for jp in range(nbj):
            if basin_arr[ip, jp] < int(undef_int):
                bval = int(basin_arr[ip, jp])
                if bval not in bname:
                    bname.append(bval)
                    pts.append([])
                    nbout.append(0)
                ibas = bname.index(bval)
                pts[ibas].append((ip + 1, jp + 1))
                if trip_arr[ip, jp] >= 97:
                    nbout[ibas] += 1

    coast_pts = np.full((max_pts,), int(undef_int), dtype=np.int64)
    trans: list[int] = []
    for idx, basin_points in enumerate(pts):
        if len(basin_points) == 1:
            ip, jp = basin_points[0]
            if trip_arr[ip - 1, jp - 1] == 99:
                trans.append(idx)
                trip_arr[ip - 1, jp - 1] = 98

    if len(trans) > 1:
        ipb = trans[0]
        coast_pts[len(pts[ipb]) - 1] = bname[ipb]
        bname[ipb] = -1
        for idx in trans[1:]:
            coast_pts[len(pts[ipb])] = bname[idx]
            pts[ipb].append(pts[idx][0])
            pts[idx] = []
            nbout[idx] = 0

    for idx, basin_points in enumerate(pts):
        if len(basin_points) > 1 and nbout[idx] > 1:
            outflows = [int(trip_arr[ip - 1, jp - 1]) for ip, jp in basin_points if trip_arr[ip - 1, jp - 1] >= 97]
            if (outflows.count(99) + outflows.count(98)) > 1:
                for ip, jp in basin_points:
                    if trip_arr[ip - 1, jp - 1] == 99:
                        trip_arr[ip - 1, jp - 1] = 98
                outflows = [int(trip_arr[ip - 1, jp - 1]) for ip, jp in basin_points if trip_arr[ip - 1, jp - 1] >= 97]
            if np.count_nonzero(np.asarray(outflows, dtype=np.int64) > 100) >= 1:
                raise NotImplementedError("routing_findbasins requires routing_simplify for redundant boundary outflows")
            if np.count_nonzero(np.asarray(outflows, dtype=np.int64) >= 97) > 1:
                raise NotImplementedError("routing_findbasins requires routing_cutbasin for multiple basin outflows")

    active_indices = [idx for idx, basin_points in enumerate(pts) if len(basin_points) > 0]
    outdir: dict[int, int] = {}
    for idx in active_indices:
        outflows = [int(trip_arr[ip - 1, jp - 1]) for ip, jp in pts[idx]]
        direction = max(outflows)
        if direction >= 97:
            outdir[idx] = direction - 100
        else:
            raise ValueError("routing_findbasins could not find a basin outflow >= 97")

    if np.count_nonzero((trip_arr >= 97) & (trip_arr < int(undef_int))) < len(active_indices):
        raise ValueError("routing_findbasins has fewer outflow points than basins")

    remaining = {idx: len(pts[idx]) for idx in active_indices}
    sorted_indices: list[int] = []
    for _ in active_indices:
        idx = max(remaining, key=lambda item: remaining[item])
        sorted_indices.append(idx)
        del remaining[idx]

    nb_basin = len(sorted_indices)
    basin_inbxid = np.zeros((nb_basin,), dtype=np.int64)
    basin_sz = np.zeros((nb_basin,), dtype=np.int64)
    basin_bxout = np.zeros((nb_basin,), dtype=np.int64)
    basin_pts = np.zeros((nb_basin, max_pts, 2), dtype=np.int64)
    for out_idx, idx in enumerate(sorted_indices):
        basin_inbxid[out_idx] = bname[idx]
        basin_sz[out_idx] = len(pts[idx])
        basin_bxout[out_idx] = outdir[idx]
        for point_idx, (ip, jp) in enumerate(pts[idx]):
            basin_pts[out_idx, point_idx, 0] = ip
            basin_pts[out_idx, point_idx, 1] = jp

    return RoutingFindBasins(
        trip=trip_arr,
        basin=basin_arr,
        nb_basin=int(nb_basin),
        basin_inbxid=basin_inbxid,
        basin_sz=basin_sz,
        basin_bxout=basin_bxout,
        basin_pts=basin_pts,
        coast_pts=coast_pts,
    )


def routing_findrout(*, trip, basin_sz: int, basinid: int) -> RoutingFindRout:
    """Follow local trip directions to each basin outflow.

    Fortran provenance: ``routing.f90::routing_findrout`` lines 8647-8758.
    The source treats cells with `trip > 9` as outflow points, initializes
    `trip_flow` for every positive-trip cell to itself, follows directions 1-8
    until an outflow marker is reached, and checks that the sum of upstream
    counts behind all outflow points equals `basin_sz`.
    """

    trip_arr = np.asarray(trip, dtype=np.int64)
    if trip_arr.ndim != 2:
        raise ValueError("trip must be two-dimensional")
    nbi, nbj = trip_arr.shape
    inc = np.asarray(
        [
            [0, -1],
            [1, -1],
            [1, 0],
            [1, 1],
            [0, 1],
            [-1, 1],
            [-1, 0],
            [-1, -1],
        ],
        dtype=np.int64,
    )

    outflows: list[tuple[int, int]] = []
    trip_flow = np.zeros((nbi, nbj, 2), dtype=np.int64)
    for ip in range(nbi):
        for jp in range(nbj):
            if trip_arr[ip, jp] > 9:
                outflows.append((ip + 1, jp + 1))
            if trip_arr[ip, jp] > 0:
                trip_flow[ip, jp, 0] = ip + 1
                trip_flow[ip, jp, 1] = jp + 1

    for ip in range(nbi):
        for jp in range(nbj):
            if trip_flow[ip, jp, 0] + trip_flow[ip, jp, 1] > 0:
                cur_i = int(trip_flow[ip, jp, 0]) - 1
                cur_j = int(trip_flow[ip, jp, 1]) - 1
                trp = int(trip_arr[cur_i, cur_j])
                cnt = 0
                while trp > 0 and trp < 9 and cnt < nbi * nbj:
                    cnt += 1
                    cur_i += int(inc[trp - 1, 0])
                    cur_j += int(inc[trp - 1, 1])
                    if cur_i < 0 or cur_j < 0 or cur_i >= nbi or cur_j >= nbj:
                        raise ValueError("routing_findrout trip path left the local grid before an outflow marker")
                    trip_flow[ip, jp, 0] = cur_i + 1
                    trip_flow[ip, jp, 1] = cur_j + 1
                    trp = int(trip_arr[cur_i, cur_j])
                if cnt == nbi * nbj:
                    raise ValueError("routing_findrout could not route point without cycling")

    nbout = len(outflows)
    outflow = np.zeros((nbout, 2), dtype=np.int64)
    outsz = np.zeros((nbout,), dtype=np.int64)
    totsz = 0
    for idx, (oi, oj) in enumerate(outflows):
        outflow[idx, :] = [oi, oj]
        outsz[idx] = int(np.count_nonzero((trip_flow[:, :, 0] == oi) & (trip_flow[:, :, 1] == oj)))
        totsz += int(outsz[idx])
    if int(basin_sz) != totsz:
        raise ValueError(f"routing_findrout water got lost for basin {basinid}: basin_sz={basin_sz}, routed={totsz}")

    return RoutingFindRout(
        nbout=int(nbout),
        outflow=outflow,
        trip_flow=trip_flow,
        outsz=outsz,
    )


def routing_globalize_one_grid(
    *,
    ib: int,
    neighbours,
    area_bx,
    trip_bx,
    hierarchy_bx,
    topoind_bx,
    gravel_bx,
    drainarea_bx,
    min_topoind: float,
    nb_basin: int,
    basin_inbxid,
    basin_sz,
    basin_pts,
    basin_bxout,
    coast_pts,
    basin_count,
    basin_area,
    basin_hierarchy,
    basin_topoind,
    basin_gravel,
    basin_drainarea,
    basin_id,
    basin_flowdir,
    outflow_grid,
    nbcoastal,
    coastal_basin,
    index_base: int = 1,
) -> RoutingGlobalize:
    """Globalize local basin descriptors for one grid cell.

    Fortran provenance: ``routing.f90::routing_globalize`` lines 8780-8934.
    The source appends local basin IDs to the grid basin table, transfers the
    synthetic coastal basin member list, sums local sub-cell area, averages
    topoind/gravel/drainage by basin point count, uses hierarchy at the outflow
    point (`hierar_method='OUTP'`), resets hierarchy/topoind to `min_topoind`
    for ocean/return outflows, and maps positive outflow directions through the
    `neighbours` table while preserving negative outflow codes.
    """

    ib0 = int(ib) - int(index_base)
    neighbours = np.asarray(neighbours, dtype=np.int64)
    area_bx = np.asarray(area_bx, dtype=np.float64)
    trip_bx = np.asarray(trip_bx, dtype=np.int64)
    hierarchy_bx = np.asarray(hierarchy_bx, dtype=np.float64)
    topoind_bx = np.asarray(topoind_bx, dtype=np.float64)
    gravel_bx = np.asarray(gravel_bx, dtype=np.float64)
    drainarea_bx = np.asarray(drainarea_bx, dtype=np.float64)
    basin_inbxid = np.asarray(basin_inbxid, dtype=np.int64)
    basin_sz = np.asarray(basin_sz, dtype=np.int64)
    basin_pts = np.asarray(basin_pts, dtype=np.int64)
    basin_bxout = np.asarray(basin_bxout, dtype=np.int64)
    coast_pts = np.asarray(coast_pts, dtype=np.int64)
    out_basin_count = np.asarray(basin_count, dtype=np.int64).copy()
    out_basin_area = np.asarray(basin_area, dtype=np.float64).copy()
    out_basin_hierarchy = np.asarray(basin_hierarchy, dtype=np.float64).copy()
    out_basin_topoind = np.asarray(basin_topoind, dtype=np.float64).copy()
    out_basin_gravel = np.asarray(basin_gravel, dtype=np.float64).copy()
    out_basin_drainarea = np.asarray(basin_drainarea, dtype=np.float64).copy()
    out_basin_id = np.asarray(basin_id, dtype=np.int64).copy()
    out_basin_flowdir = np.asarray(basin_flowdir, dtype=np.int64).copy()
    out_outflow_grid = np.asarray(outflow_grid, dtype=np.int64).copy()
    out_nbcoastal = np.asarray(nbcoastal, dtype=np.int64).copy()
    out_coastal_basin = np.asarray(coastal_basin, dtype=np.int64).copy()

    if out_basin_count[ib0] != 0:
        raise ValueError("routing_globalize_one_grid expects basin_count for this grid to be zero")
    nwbas = out_basin_area.shape[1]
    if int(nb_basin) > nwbas:
        raise ValueError("routing_globalize_one_grid basin_count exceeds allocated nwbas")

    for ij in range(int(nb_basin)):
        out_basin_count[ib0] += 1
        slot = int(out_basin_count[ib0]) - 1
        out_basin_id[ib0, slot] = basin_inbxid[ij]

        if out_basin_id[ib0, slot] < 0:
            out_nbcoastal[ib0] = basin_sz[ij]
            out_coastal_basin[ib0, : out_nbcoastal[ib0]] = coast_pts[: out_nbcoastal[ib0]]

        out_basin_area[ib0, ij] = 0.0
        out_basin_hierarchy[ib0, ij] = 0.0
        out_basin_topoind[ib0, ij] = 0.0
        out_basin_gravel[ib0, ij] = 0.0
        out_basin_drainarea[ib0, ij] = 0.0
        for iz in range(int(basin_sz[ij])):
            ip = int(basin_pts[ij, iz, 0]) - int(index_base)
            jp = int(basin_pts[ij, iz, 1]) - int(index_base)
            out_basin_area[ib0, ij] += area_bx[ip, jp]
            out_basin_topoind[ib0, ij] += topoind_bx[ip, jp]
            out_basin_gravel[ib0, ij] += gravel_bx[ip, jp]
            out_basin_drainarea[ib0, ij] += drainarea_bx[ip, jp]
            if trip_bx[ip, jp] > 100:
                out_basin_hierarchy[ib0, ij] = hierarchy_bx[ip, jp]

        out_basin_topoind[ib0, ij] /= float(basin_sz[ij])
        out_basin_gravel[ib0, ij] /= float(basin_sz[ij])
        out_basin_drainarea[ib0, ij] /= float(basin_sz[ij])
        if basin_bxout[ij] < 0:
            out_basin_hierarchy[ib0, ij] = float(min_topoind)
            out_basin_topoind[ib0, ij] = float(min_topoind)

        out_basin_flowdir[ib0, ij] = basin_bxout[ij]
        if basin_bxout[ij] > 0:
            out_outflow_grid[ib0, ij] = neighbours[ib0, int(basin_bxout[ij]) - 1]
        else:
            out_outflow_grid[ib0, ij] = basin_bxout[ij]

    return RoutingGlobalize(
        basin_count=out_basin_count,
        basin_area=out_basin_area,
        basin_hierarchy=out_basin_hierarchy,
        basin_topoind=out_basin_topoind,
        basin_gravel=out_basin_gravel,
        basin_drainarea=out_basin_drainarea,
        basin_id=out_basin_id,
        basin_flowdir=out_basin_flowdir,
        outflow_grid=out_outflow_grid,
        nbcoastal=out_nbcoastal,
        coastal_basin=out_coastal_basin,
    )


def routing_linkup(
    *,
    contfrac,
    neighbours,
    basin_count,
    basin_area,
    basin_id,
    basin_flowdir,
    basin_hierarchy,
    outflow_grid,
    nbcoastal,
    coastal_basin,
    invented_basins: float,
    undef_int=999999999,
) -> RoutingLinkup:
    """Connect globalized routing basins and build inflow tables.

    Fortran provenance: ``routing.f90::routing_linkup`` lines 8957-9413. The
    implementation preserves the source search order: direct target-grid basin
    matching, +/- one-neighbor fallback, same-grid fallback, coastal fallback,
    inflow table updates, and final existence checks. It raises explicit
    errors for the same failure conditions that call `ipslerr_p` in Fortran.
    """

    contfrac = np.asarray(contfrac, dtype=np.float64)
    neighbours = np.asarray(neighbours, dtype=np.int64)
    basin_count = np.asarray(basin_count, dtype=np.int64)
    basin_area = np.asarray(basin_area, dtype=np.float64)
    basin_id = np.asarray(basin_id, dtype=np.int64)
    basin_flowdir = np.asarray(basin_flowdir, dtype=np.int64)
    basin_hierarchy = np.asarray(basin_hierarchy, dtype=np.float64)
    out_grid = np.asarray(outflow_grid, dtype=np.int64).copy()
    nbcoastal = np.asarray(nbcoastal, dtype=np.int64)
    coastal_basin = np.asarray(coastal_basin, dtype=np.int64)

    nbpt, nwbas = basin_id.shape
    nbvmax = nwbas
    out_basin = np.full((nbpt, nwbas), int(undef_int), dtype=np.int64)
    inflow_number = np.zeros((nbpt, nwbas), dtype=np.int64)
    inflow_grid = np.zeros((nbpt, nwbas, nbvmax), dtype=np.int64)
    inflow_basin = np.zeros((nbpt, nwbas, nbvmax), dtype=np.int64)

    def same_or_adjacent(target_dir: int, source_dir: int) -> bool:
        return (
            ((target_dir + 1 - 1) % 8) + 1 == source_dir
            or target_dir == source_dir
            or ((target_dir + 7 - 1) % 8) + 1 == source_dir
        )

    def angle_ok(target_dir: int, source_dir: int) -> bool:
        angle = (target_dir - source_dir + 8) % 8
        if angle >= 4:
            angle -= 8
        return abs(angle) <= 1

    def add_inflow(target_grid0: int, target_basin0: int, source_grid0: int, source_basin0: int) -> None:
        inflow_number[target_grid0, target_basin0] += 1
        pos = int(inflow_number[target_grid0, target_basin0]) - 1
        if pos >= nbvmax:
            raise ValueError("routing_linkup inflow table exceeds allocated nbvmax")
        inflow_grid[target_grid0, target_basin0, pos] = source_grid0 + 1
        inflow_basin[target_grid0, target_basin0, pos] = source_basin0 + 1

    for sp0 in range(nbpt):
        for sb0 in range(int(basin_count[sp0])):
            inp = int(out_grid[sp0, sb0])
            bid = int(basin_id[sp0, sb0])

            if inp > 0:
                inp0 = inp - 1
                fbas: list[int] = []
                fbas_hierarchy: list[float] = []
                for sbl0 in range(int(basin_count[inp0])):
                    if basin_id[inp0, sbl0] > 0:
                        if basin_id[inp0, sbl0] == bid or basin_id[inp0, sbl0] > float(invented_basins):
                            fbas.append(sbl0)
                            fbas_hierarchy.append(float(basin_hierarchy[inp0, sbl0]))
                    else:
                        if np.count_nonzero(coastal_basin[inp0, : int(nbcoastal[inp0])] == bid) > 0:
                            fbas.append(sbl0)
                            fbas_hierarchy.append(float(basin_hierarchy[inp0, sbl0]))

                if fbas:
                    sbl0 = fbas[int(np.argmin(np.asarray(fbas_hierarchy, dtype=np.float64)))]
                    bop0 = None
                    if basin_hierarchy[inp0, sbl0] <= basin_hierarchy[sp0, sb0]:
                        if basin_hierarchy[inp0, sbl0] < basin_hierarchy[sp0, sb0]:
                            bop0 = sbl0
                        elif same_or_adjacent(int(basin_flowdir[inp0, sbl0]), int(basin_flowdir[sp0, sb0])):
                            bop0 = sbl0
                    if bop0 is not None:
                        out_basin[sp0, sb0] = bop0 + 1
                        add_inflow(inp0, bop0, sp0, sb0)

            if out_basin[sp0, sb0] == int(undef_int) and basin_flowdir[sp0, sb0] > 0:
                flowdir = int(basin_flowdir[sp0, sb0])
                dp1i = ((flowdir + 1 - 1) % 8) + 1
                dp1 = int(neighbours[sp0, dp1i - 1])
                dm1i = ((flowdir + 7 - 1) % 8) + 1
                dm1 = int(neighbours[sp0, dm1i - 1])

                bp1 = -1
                if dp1 > 0:
                    dp10 = dp1 - 1
                    for sbl0 in range(int(basin_count[dp10])):
                        if (
                            basin_id[dp10, sbl0] == bid
                            and basin_hierarchy[sp0, sb0] >= basin_hierarchy[dp10, sbl0]
                            and bp1 < 0
                        ):
                            if basin_hierarchy[sp0, sb0] > basin_hierarchy[dp10, sbl0]:
                                bp1 = sbl0 + 1
                            elif angle_ok(int(basin_flowdir[dp10, sbl0]), flowdir):
                                bp1 = sbl0 + 1

                bm1 = -1
                if dm1 > 0:
                    dm10 = dm1 - 1
                    for sbl0 in range(int(basin_count[dm10])):
                        if (
                            basin_id[dm10, sbl0] == bid
                            and basin_hierarchy[sp0, sb0] >= basin_hierarchy[dm10, sbl0]
                            and bm1 < 0
                        ):
                            if basin_hierarchy[sp0, sb0] > basin_hierarchy[dm10, sbl0]:
                                bm1 = sbl0 + 1
                            elif angle_ok(int(basin_flowdir[dm10, sbl0]), flowdir):
                                bm1 = sbl0 + 1

                outdp1 = int(undef_int)
                if dp1 > 0 and bp1 > 0:
                    dp10 = dp1 - 1
                    bp10 = bp1 - 1
                    if basin_flowdir[dp10, bp10] > 0:
                        outdp1 = int(neighbours[dp10, int(basin_flowdir[dp10, bp10]) - 1])
                        if outdp1 == sp0 + 1:
                            outdp1 = int(undef_int)
                    else:
                        outdp1 = nbpt + 1

                outdm1 = int(undef_int)
                if dm1 > 0 and bm1 > 0:
                    dm10 = dm1 - 1
                    bm10 = bm1 - 1
                    if basin_flowdir[dm10, bm10] > 0:
                        outdm1 = int(neighbours[dm10, int(basin_flowdir[dm10, bm10]) - 1])
                        if outdm1 == sp0 + 1:
                            outdm1 = int(undef_int)
                    else:
                        outdm1 = nbpt + 1

                dop = int(undef_int)
                bop = int(undef_int)
                if outdp1 < int(undef_int) and outdm1 < int(undef_int):
                    if basin_area[dp1 - 1, bp1 - 1] < basin_area[dm1 - 1, bm1 - 1]:
                        dop, bop = dp1, bp1
                    else:
                        dop, bop = dm1, bm1
                elif outdp1 < int(undef_int):
                    dop, bop = dp1, bp1
                elif outdm1 < int(undef_int):
                    dop, bop = dm1, bm1
                else:
                    if out_grid[sp0, sb0] < 0 or dm1 < 0 or dp1 < 0:
                        dop = -1
                    elif bp1 < 0 and bm1 < 0:
                        if np.count_nonzero(basin_id[sp0, : int(basin_count[sp0])] == bid) >= 2:
                            for sbl0 in range(int(basin_count[sp0])):
                                if sbl0 != sb0 and basin_id[sp0, sbl0] == bid:
                                    if (
                                        basin_hierarchy[sp0, sb0] >= basin_hierarchy[sp0, sbl0]
                                        or (basin_flowdir[sp0, sbl0] < dm1i and basin_flowdir[sp0, sbl0] > dp1i)
                                    ):
                                        dop, bop = sp0 + 1, sbl0 + 1

                    if dop == int(undef_int) and bop == int(undef_int) and contfrac[sp0] > 0.01:
                        raise ValueError("routing_linkup could not find valid direction for basin outflow")

                if dop > 0 and dop != int(undef_int):
                    out_grid[sp0, sb0] = dop
                    out_basin[sp0, sb0] = bop
                    add_inflow(dop - 1, bop - 1, sp0, sb0)
                else:
                    out_grid[sp0, sb0] = -2
                    out_basin[sp0, sb0] = int(undef_int)

            if out_grid[sp0, sb0] > 0 and out_basin[sp0, sb0] == int(undef_int) and basin_flowdir[sp0, sb0] > 0:
                for sbl0 in range(int(basin_count[sp0])):
                    if (
                        sbl0 != sb0
                        and basin_id[sp0, sbl0] == bid
                        and basin_hierarchy[sp0, sb0] > basin_hierarchy[sp0, sbl0]
                    ):
                        out_basin[sp0, sb0] = sbl0 + 1
                        add_inflow(sp0, sbl0, sp0, sb0)

            if out_grid[sp0, sb0] > 0 and out_basin[sp0, sb0] == int(undef_int) and basin_flowdir[sp0, sb0] > 0:
                raise ValueError("routing_linkup could not find the basin into which we need to flow")

    for sp0 in range(nbpt):
        for sb0 in range(int(basin_count[sp0])):
            inp = int(out_grid[sp0, sb0])
            sbl = int(out_basin[sp0, sb0])
            if inp >= 0:
                if inp < 1 or inp > nbpt or basin_count[inp - 1] < sbl:
                    raise ValueError("routing_linkup outflow basin does not exist")

    return RoutingLinkup(
        outflow_grid=out_grid,
        outflow_basin=out_basin,
        inflow_number=inflow_number,
        inflow_grid=inflow_grid,
        inflow_basin=inflow_basin,
    )


def routing_fetch(
    *,
    resolution,
    contfrac,
    basin_count,
    basin_area,
    basin_id,
    outflow_grid,
    outflow_basin,
    num_largest: int,
) -> RoutingFetch:
    """Compute upstream fetch and river/coastal outlet classes.

    Fortran provenance: ``routing.f90::routing_fetch`` lines 9435-9555. The
    source normalizes basin areas to the land area of each grid cell, adds each
    basin area to every downstream basin along its outflow chain, converts
    existing river outlets (`-1`) to coastal (`-2`), then marks the
    `num_largest` coastal outlets by fetch as river outlets.
    """

    resolution = np.asarray(resolution, dtype=np.float64)
    contfrac = np.asarray(contfrac, dtype=np.float64)
    basin_count = np.asarray(basin_count, dtype=np.int64)
    basin_id = np.asarray(basin_id, dtype=np.int64)
    area = np.asarray(basin_area, dtype=np.float64).copy()
    out_grid = np.asarray(outflow_grid, dtype=np.int64).copy()
    out_basin = np.asarray(outflow_basin, dtype=np.int64)
    nbpt, nwbas = area.shape
    if resolution.shape != (nbpt, 2) or contfrac.shape != (nbpt,):
        raise ValueError("resolution must be (nbpt,2) and contfrac must be (nbpt,)")

    for ib in range(nbpt):
        count = int(basin_count[ib])
        if count <= 0:
            continue
        totbasins = float(np.sum(area[ib, :count]))
        if totbasins <= 0.0:
            raise ValueError("routing_fetch requires positive total basin area per active grid")
        contarea = resolution[ib, 0] * resolution[ib, 1] * contfrac[ib]
        area[ib, :count] = area[ib, :count] / totbasins * contarea

    fetch = np.zeros_like(area, dtype=np.float64)
    for ib in range(nbpt):
        for ij in range(int(basin_count[ib])):
            fetch[ib, ij] += area[ib, ij]
            igrif = int(out_grid[ib, ij])
            ibasf = int(out_basin[ib, ij])
            itt = 0
            while igrif > 0:
                ig0 = igrif - 1
                ib0 = ibasf - 1
                if ig0 < 0 or ig0 >= nbpt or ib0 < 0 or ib0 >= nwbas:
                    raise ValueError("routing_fetch outflow chain points outside basin arrays")
                fetch[ig0, ib0] += area[ib, ij]
                next_grid = int(out_grid[ig0, ib0])
                ibasf = int(out_basin[ig0, ib0])
                igrif = next_grid
                itt += 1
                if itt > 500:
                    raise ValueError(f"routing_fetch did not converge for grid {ib + 1} basin {ij + 1}")

    outlet_fetch: list[tuple[float, int, int]] = []
    for ib in range(nbpt):
        for ij in range(int(basin_count[ib])):
            if out_grid[ib, ij] == -1:
                out_grid[ib, ij] = -2
            if out_grid[ib, ij] == -2:
                outlet_fetch.append((float(fetch[ib, ij]), ib, ij))

    for _ in range(min(int(num_largest), len(outlet_fetch))):
        idx = max(range(len(outlet_fetch)), key=lambda pos: outlet_fetch[pos][0])
        _, ib, ij = outlet_fetch[idx]
        out_grid[ib, ij] = -1
        outlet_fetch[idx] = (0.0, ib, ij)

    return RoutingFetch(
        basin_area=area,
        outflow_grid=out_grid,
        fetch_basin=fetch,
    )


def routing_truncate_finalize_no_reduction(
    *,
    resolution,
    contfrac,
    basin_count,
    basin_area,
    basin_topoind,
    basin_gravel,
    basin_drainarea,
    basin_id,
    outflow_grid,
    outflow_basin,
    nbasmax: int,
    num_largest: int,
) -> RoutingTruncateFinalized:
    """Finalize routing arrays when no basin-count reduction is needed.

    Fortran provenance: ``routing.f90::routing_truncate`` lines 9917-10067.
    This helper covers the final writeback path after the source truncation
    loop has already made `basin_count <= nbasmax`: initialization of inactive
    basin slots, transfer of basin fields, conversion of negative outflow-grid
    conventions to `route_tobasin = nbasmax+1/2`, route consistency checks,
    grid-area correction of `routing_area`, and marking the `num_largest`
    outlet basins as river outflows (`nbasmax+3`). Inputs still needing basin
    reduction are rejected rather than approximated.
    """

    resolution = np.asarray(resolution, dtype=np.float64)
    contfrac = np.asarray(contfrac, dtype=np.float64)
    basin_count = np.asarray(basin_count, dtype=np.int64)
    basin_area = np.asarray(basin_area, dtype=np.float64)
    basin_topoind = np.asarray(basin_topoind, dtype=np.float64)
    basin_gravel = np.asarray(basin_gravel, dtype=np.float64)
    basin_drainarea = np.asarray(basin_drainarea, dtype=np.float64)
    basin_id = np.asarray(basin_id, dtype=np.int64)
    outflow_grid = np.asarray(outflow_grid, dtype=np.int64)
    outflow_basin = np.asarray(outflow_basin, dtype=np.int64)

    nbpt = basin_count.shape[0]
    nbasmax = int(nbasmax)
    if np.any(basin_count > nbasmax):
        raise NotImplementedError("routing_truncate basin reduction requires routing_killbas")
    if resolution.shape != (nbpt, 2) or contfrac.shape != (nbpt,):
        raise ValueError("resolution must be (nbpt,2) and contfrac must be (nbpt,)")

    routing_area = np.zeros((nbpt, nbasmax), dtype=np.float64)
    topo_resid = np.zeros((nbpt, nbasmax), dtype=np.float64)
    basgravel = np.zeros((nbpt, nbasmax), dtype=np.float64)
    basdrainarea = np.zeros((nbpt, nbasmax), dtype=np.float64)
    global_basinid = np.zeros((nbpt, nbasmax), dtype=np.int64)
    route_togrid = np.zeros((nbpt, nbasmax), dtype=np.int64)
    route_tobasin = np.zeros((nbpt, nbasmax), dtype=np.int64)
    for ib in range(nbpt):
        route_togrid[ib, :] = ib + 1

    for ib in range(nbpt):
        for ij in range(int(basin_count[ib])):
            routing_area[ib, ij] = basin_area[ib, ij]
            topo_resid[ib, ij] = basin_topoind[ib, ij]
            basgravel[ib, ij] = basin_gravel[ib, ij]
            basdrainarea[ib, ij] = basin_drainarea[ib, ij]
            global_basinid[ib, ij] = basin_id[ib, ij]
            route_togrid[ib, ij] = outflow_grid[ib, ij]
            route_tobasin[ib, ij] = outflow_basin[ib, ij]

    for ib in range(nbpt):
        for ij in range(int(basin_count[ib])):
            if route_togrid[ib, ij] == -1:
                route_tobasin[ib, ij] = nbasmax + 2
                route_togrid[ib, ij] = ib + 1
            elif route_togrid[ib, ij] == -2:
                route_tobasin[ib, ij] = nbasmax + 2
                route_togrid[ib, ij] = ib + 1
            elif route_togrid[ib, ij] == -3:
                route_tobasin[ib, ij] = nbasmax + 1
                route_togrid[ib, ij] = ib + 1

    for ib in range(nbpt):
        for ij in range(int(basin_count[ib])):
            ibf = int(route_togrid[ib, ij]) - 1
            ijf = int(route_tobasin[ib, ij])
            if ibf < 0 or ibf >= nbpt:
                raise ValueError("routing_truncate route_togrid points outside domain")
            if ijf > basin_count[ibf] and ijf <= nbasmax:
                raise ValueError("routing_truncate route_tobasin points to a missing basin")

    floflo = np.zeros((nbpt, nbasmax), dtype=np.float64)
    gridarea = contfrac * resolution[:, 0] * resolution[:, 1]
    gridbasinarea = np.sum(routing_area, axis=1)
    for ib in range(nbpt):
        for ij in range(int(basin_count[ib])):
            cnt = 0
            igrif = ib + 1
            ibasf = ij + 1
            bold = ij + 1
            while ibasf <= nbasmax and cnt < nbasmax * nbpt:
                cnt += 1
                pold = igrif
                bold = ibasf
                igrif = int(route_togrid[pold - 1, bold - 1])
                ibasf = int(route_tobasin[pold - 1, bold - 1])
                if ibasf > basin_count[igrif - 1] and ibasf <= nbasmax:
                    raise ValueError("routing_truncate route chain points to a missing basin")
            if ibasf > nbasmax:
                floflo[igrif - 1, bold - 1] += routing_area[ib, ij]
            else:
                raise ValueError("routing_truncate flow did not end in an outlet code")

    for ib in range(nbpt):
        if gridbasinarea[ib] > 0.0:
            routing_area[ib, :] *= gridarea[ib] / gridbasinarea[ib]
        else:
            raise ValueError("routing_truncate gridbasinarea must be positive")

    largest: list[tuple[int, int]] = []
    floflo_work = floflo.copy()
    pickmax = min(200, nbpt * nbasmax)
    for _ in range(pickmax):
        flat = int(np.argmax(floflo_work))
        ff = np.unravel_index(flat, floflo_work.shape)
        if route_tobasin[ff] > nbasmax:
            largest.append((int(ff[0]), int(ff[1])))
        floflo_work[ff] = 0.0
        if np.all(floflo_work <= 0.0):
            break
    if len(largest) < int(num_largest):
        raise ValueError("routing_truncate not enough outlet basins for num_largest")
    for ib, ij in largest[: int(num_largest)]:
        route_tobasin[ib, ij] = nbasmax + 3

    return RoutingTruncateFinalized(
        routing_area=routing_area,
        topo_resid=topo_resid,
        basgravel=basgravel,
        basdrainarea=basdrainarea,
        global_basinid=global_basinid,
        route_togrid=route_togrid,
        route_tobasin=route_tobasin,
    )


def routing_killbas(
    *,
    ib: int,
    tokill: int,
    totakeover: int,
    basin_count,
    basin_area,
    basin_topoind,
    basin_gravel,
    basin_drainarea,
    fetch_basin,
    basin_id,
    basin_flowdir,
    outflow_grid,
    outflow_basin,
    inflow_number,
    inflow_grid,
    inflow_basin,
    index_base: int = 1,
) -> RoutingKillBas:
    """Merge one routing basin into another and shift basin tables.

    Fortran provenance: ``routing.f90::routing_killbas`` lines 10091-10256.
    The source adds killed-basin area/fetch to the takeover basin, updates
    downstream fetch along both old and new paths, redirects inflows to the
    killed basin, removes the killed basin from downstream inflow tables, shifts
    basin fields left, fixes external references to shifted basin numbers, and
    decrements `basin_count`.
    """

    ib0 = int(ib) - int(index_base)
    kill0 = int(tokill) - int(index_base)
    take0 = int(totakeover) - int(index_base)
    out_basin_count = np.asarray(basin_count, dtype=np.int64).copy()
    out_basin_area = np.asarray(basin_area, dtype=np.float64).copy()
    out_basin_topoind = np.asarray(basin_topoind, dtype=np.float64).copy()
    out_basin_gravel = np.asarray(basin_gravel, dtype=np.float64).copy()
    out_basin_drainarea = np.asarray(basin_drainarea, dtype=np.float64).copy()
    out_fetch = np.asarray(fetch_basin, dtype=np.float64).copy()
    out_basin_id = np.asarray(basin_id, dtype=np.int64).copy()
    out_basin_flowdir = np.asarray(basin_flowdir, dtype=np.int64).copy()
    out_grid = np.asarray(outflow_grid, dtype=np.int64).copy()
    out_basin = np.asarray(outflow_basin, dtype=np.int64).copy()
    out_inflow_number = np.asarray(inflow_number, dtype=np.int64).copy()
    out_inflow_grid = np.asarray(inflow_grid, dtype=np.int64).copy()
    out_inflow_basin = np.asarray(inflow_basin, dtype=np.int64).copy()
    nwbas = out_basin_area.shape[1]
    count = int(out_basin_count[ib0])
    if not (0 <= kill0 < count and 0 <= take0 < count and kill0 != take0):
        raise ValueError("routing_killbas requires distinct valid tokill/totakeover basins")

    killed_fetch = float(out_fetch[ib0, kill0])
    out_basin_area[ib0, take0] += out_basin_area[ib0, kill0]
    out_basin_topoind[ib0, take0] = (out_basin_topoind[ib0, take0] + out_basin_topoind[ib0, kill0]) / 2.0
    out_basin_gravel[ib0, take0] = (out_basin_gravel[ib0, take0] + out_basin_gravel[ib0, kill0]) / 2.0
    out_basin_drainarea[ib0, take0] = (out_basin_drainarea[ib0, take0] + out_basin_drainarea[ib0, kill0]) / 2.0
    out_fetch[ib0, take0] += killed_fetch

    igrif = int(out_grid[ib0, take0])
    ibasf = int(out_basin[ib0, take0])
    while igrif > 0:
        out_fetch[igrif - 1, ibasf - 1] += killed_fetch
        next_grid = int(out_grid[igrif - 1, ibasf - 1])
        ibasf = int(out_basin[igrif - 1, ibasf - 1])
        igrif = next_grid

    igrif = int(out_grid[ib0, kill0])
    ibasf = int(out_basin[ib0, kill0])
    while igrif > 0:
        out_fetch[igrif - 1, ibasf - 1] -= killed_fetch
        next_grid = int(out_grid[igrif - 1, ibasf - 1])
        ibasf = int(out_basin[igrif - 1, ibasf - 1])
        igrif = next_grid

    for pos in range(int(out_inflow_number[ib0, kill0])):
        src_grid0 = int(out_inflow_grid[ib0, kill0, pos]) - 1
        src_basin0 = int(out_inflow_basin[ib0, kill0, pos]) - 1
        out_basin[src_grid0, src_basin0] = take0 + 1
        out_inflow_number[ib0, take0] += 1
        new_pos = int(out_inflow_number[ib0, take0]) - 1
        out_inflow_grid[ib0, take0, new_pos] = src_grid0 + 1
        out_inflow_basin[ib0, take0, new_pos] = src_basin0 + 1

    if out_grid[ib0, kill0] > 0:
        ing0 = int(out_grid[ib0, kill0]) - 1
        inb0 = int(out_basin[ib0, kill0]) - 1
        found = False
        n_in = int(out_inflow_number[ing0, inb0])
        keep_grid: list[int] = []
        keep_basin: list[int] = []
        for pos in range(n_in):
            if out_inflow_grid[ing0, inb0, pos] == ib0 + 1 and out_inflow_basin[ing0, inb0, pos] == kill0 + 1:
                found = True
            else:
                keep_grid.append(int(out_inflow_grid[ing0, inb0, pos]))
                keep_basin.append(int(out_inflow_basin[ing0, inb0, pos]))
        if not found:
            raise ValueError("routing_killbas did not find killed basin in downstream inflow table")
        out_inflow_number[ing0, inb0] -= 1
        out_inflow_grid[ing0, inb0, :n_in] = 0
        out_inflow_basin[ing0, inb0, :n_in] = 0
        out_inflow_grid[ing0, inb0, : len(keep_grid)] = keep_grid
        out_inflow_basin[ing0, inb0, : len(keep_basin)] = keep_basin

    if kill0 + 1 < count:
        sl_src = slice(kill0 + 1, count)
        sl_dst = slice(kill0, count - 1)
        out_basin_id[ib0, sl_dst] = out_basin_id[ib0, sl_src]
        out_basin_flowdir[ib0, sl_dst] = out_basin_flowdir[ib0, sl_src]
        out_basin_area[ib0, sl_dst] = out_basin_area[ib0, sl_src]
        out_basin_topoind[ib0, sl_dst] = out_basin_topoind[ib0, sl_src]
        out_basin_gravel[ib0, sl_dst] = out_basin_gravel[ib0, sl_src]
        out_basin_drainarea[ib0, sl_dst] = out_basin_drainarea[ib0, sl_src]
        out_fetch[ib0, sl_dst] = out_fetch[ib0, sl_src]

    out_basin_area[ib0, count - 1 : nwbas] = 0.0
    out_basin_topoind[ib0, count - 1 : nwbas] = 0.0
    out_basin_gravel[ib0, count - 1 : nwbas] = 0.0
    out_basin_drainarea[ib0, count - 1 : nwbas] = 0.0
    out_fetch[ib0, count - 1 : nwbas] = 0.0

    for ibs0 in range(kill0 + 1, count):
        ing = int(out_grid[ib0, ibs0])
        inb = int(out_basin[ib0, ibs0])
        if ing > 0:
            for pos in range(int(out_inflow_number[ing - 1, inb - 1])):
                if out_inflow_grid[ing - 1, inb - 1, pos] == ib0 + 1 and out_inflow_basin[ing - 1, inb - 1, pos] == ibs0 + 1:
                    out_inflow_basin[ing - 1, inb - 1, pos] = ibs0

    if kill0 + 1 < count:
        out_grid[ib0, kill0 : count - 1] = out_grid[ib0, kill0 + 1 : count]
        out_basin[ib0, kill0 : count - 1] = out_basin[ib0, kill0 + 1 : count]

    for ibs0 in range(kill0 + 1, count):
        for pos in range(int(out_inflow_number[ib0, ibs0])):
            src_grid0 = int(out_inflow_grid[ib0, ibs0, pos]) - 1
            src_basin0 = int(out_inflow_basin[ib0, ibs0, pos]) - 1
            out_basin[src_grid0, src_basin0] = ibs0

    if kill0 + 1 < count:
        for src in range(kill0 + 1, count):
            dst = src - 1
            n_move = int(out_inflow_number[ib0, src])
            out_inflow_grid[ib0, dst, :n_move] = out_inflow_grid[ib0, src, :n_move]
            out_inflow_basin[ib0, dst, :n_move] = out_inflow_basin[ib0, src, :n_move]
            out_inflow_number[ib0, dst] = n_move

    out_inflow_number[ib0, count - 1 : nwbas] = 0
    out_inflow_grid[ib0, count - 1 : nwbas, :] = 0
    out_inflow_basin[ib0, count - 1 : nwbas, :] = 0
    out_basin_count[ib0] -= 1

    return RoutingKillBas(
        basin_count=out_basin_count,
        basin_area=out_basin_area,
        basin_topoind=out_basin_topoind,
        basin_gravel=out_basin_gravel,
        basin_drainarea=out_basin_drainarea,
        fetch_basin=out_fetch,
        basin_id=out_basin_id,
        basin_flowdir=out_basin_flowdir,
        outflow_grid=out_grid,
        outflow_basin=out_basin,
        inflow_number=out_inflow_number,
        inflow_grid=out_inflow_grid,
        inflow_basin=out_inflow_basin,
    )


def routing_truncate_reduce_to_nbasmax(
    *,
    basin_count,
    basin_area,
    basin_topoind,
    basin_gravel,
    basin_drainarea,
    fetch_basin,
    basin_id,
    basin_flowdir,
    outflow_grid,
    outflow_basin,
    inflow_number,
    inflow_grid,
    inflow_basin,
    nbasmax: int,
) -> RoutingKillBas:
    """Reduce basin tables to `nbasmax` using the source selection rules.

    Fortran provenance: ``routing.f90::routing_truncate`` lines 9652-9912 plus
    `routing_killbas` lines 10091-10256. This helper performs only the basin
    reduction phase; final route-array writeback remains in
    `routing_truncate_finalize_no_reduction`.
    """

    out_basin_count = np.asarray(basin_count, dtype=np.int64).copy()
    out_basin_area = np.asarray(basin_area, dtype=np.float64).copy()
    out_basin_topoind = np.asarray(basin_topoind, dtype=np.float64).copy()
    out_basin_gravel = np.asarray(basin_gravel, dtype=np.float64).copy()
    out_basin_drainarea = np.asarray(basin_drainarea, dtype=np.float64).copy()
    out_fetch = np.asarray(fetch_basin, dtype=np.float64).copy()
    out_basin_id = np.asarray(basin_id, dtype=np.int64).copy()
    out_basin_flowdir = np.asarray(basin_flowdir, dtype=np.int64).copy()
    out_grid = np.asarray(outflow_grid, dtype=np.int64).copy()
    out_basin = np.asarray(outflow_basin, dtype=np.int64).copy()
    out_inflow_number = np.asarray(inflow_number, dtype=np.int64).copy()
    out_inflow_grid = np.asarray(inflow_grid, dtype=np.int64).copy()
    out_inflow_basin = np.asarray(inflow_basin, dtype=np.int64).copy()
    nbpt = out_basin_count.shape[0]
    nbasmax = int(nbasmax)

    def apply_kill(grid0: int, kill0: int, take0: int) -> None:
        nonlocal out_basin_count, out_basin_area, out_basin_topoind, out_basin_gravel
        nonlocal out_basin_drainarea, out_fetch, out_basin_id, out_basin_flowdir
        nonlocal out_grid, out_basin, out_inflow_number, out_inflow_grid, out_inflow_basin
        killed = routing_killbas(
            ib=grid0 + 1,
            tokill=kill0 + 1,
            totakeover=take0 + 1,
            basin_count=out_basin_count,
            basin_area=out_basin_area,
            basin_topoind=out_basin_topoind,
            basin_gravel=out_basin_gravel,
            basin_drainarea=out_basin_drainarea,
            fetch_basin=out_fetch,
            basin_id=out_basin_id,
            basin_flowdir=out_basin_flowdir,
            outflow_grid=out_grid,
            outflow_basin=out_basin,
            inflow_number=out_inflow_number,
            inflow_grid=out_inflow_grid,
            inflow_basin=out_inflow_basin,
        )
        out_basin_count = killed.basin_count
        out_basin_area = killed.basin_area
        out_basin_topoind = killed.basin_topoind
        out_basin_gravel = killed.basin_gravel
        out_basin_drainarea = killed.basin_drainarea
        out_fetch = killed.fetch_basin
        out_basin_id = killed.basin_id
        out_basin_flowdir = killed.basin_flowdir
        out_grid = killed.outflow_grid
        out_basin = killed.outflow_basin
        out_inflow_number = killed.inflow_number
        out_inflow_grid = killed.inflow_grid
        out_inflow_basin = killed.inflow_basin

    max_iter = max(0, int(np.max(out_basin_count)) - nbasmax + 3)
    for _ in range(max_iter):
        for ib in [idx for idx in range(nbpt) if out_basin_count[idx] > nbasmax]:
            count = int(out_basin_count[ib])
            kbas = -1
            sbas = -1

            if (np.count_nonzero(out_grid[ib, :count] == -2) > 1 or np.count_nonzero(out_grid[ib, :count] == -3) > 1):
                obj = -2 if np.count_nonzero(out_grid[ib, :count] == -2) > 1 else -3
                candidates = [idx for idx in range(count) if out_grid[ib, idx] == obj]
                positive = [idx for idx in candidates if out_fetch[ib, idx] > 0.0]
                if positive:
                    sbas = max(positive, key=lambda idx: out_fetch[ib, idx])
                    kbas = min(positive, key=lambda idx: out_fetch[ib, idx])

            if kbas < 0 or sbas < 0:
                groups: dict[int, list[int]] = {}
                for idx in range(count):
                    if out_grid[ib, idx] > 0:
                        groups.setdefault(int(out_grid[ib, idx]), []).append(idx)
                groups = {key: value for key, value in groups.items() if len(value) > 1}
                if groups:
                    chosen = max(groups.values(), key=len)
                    positive = [idx for idx in chosen if out_fetch[ib, idx] > 0.0]
                    if positive:
                        sbas = max(positive, key=lambda idx: out_fetch[ib, idx])
                        kbas = min(positive, key=lambda idx: out_fetch[ib, idx])

            if kbas < 0 or sbas < 0:
                groups: dict[int, list[int]] = {}
                for idx in range(count):
                    groups.setdefault(int(out_basin_id[ib, idx]), []).append(idx)
                groups = {key: value for key, value in groups.items() if len(value) > 1}
                if groups:
                    chosen = max(groups.values(), key=len)
                    ocean = [idx for idx in chosen if out_grid[ib, idx] < 0]
                    if ocean:
                        sbas = ocean[0]
                        positive = [idx for idx in chosen if idx != sbas and out_fetch[ib, idx] > 0.0]
                        if positive:
                            kbas = min(positive, key=lambda idx: out_fetch[ib, idx])
                    else:
                        positive = [idx for idx in chosen if out_fetch[ib, idx] > 0.0]
                        if positive:
                            sbas = max(positive, key=lambda idx: out_fetch[ib, idx])
                            kbas = min(positive, key=lambda idx: out_fetch[ib, idx])

            if kbas >= 0 and sbas >= 0 and kbas != sbas:
                apply_kill(ib, kbas, sbas)

    if np.any(out_basin_count > nbasmax):
        over = np.count_nonzero(out_basin_count > nbasmax)
        if (over / max(nbpt, 1)) * 100.0 > 5.0:
            raise ValueError("routing_truncate could not reduce enough basins without brutal method")
        for ib in range(nbpt):
            while out_basin_count[ib] > nbasmax:
                count = int(out_basin_count[ib])
                positive = [idx for idx in range(count) if out_fetch[ib, idx] > 0.0]
                if not positive:
                    raise ValueError("routing_truncate hammer found no positive-fetch basin")
                sbas = max(positive, key=lambda idx: out_fetch[ib, idx])
                kbas = min(positive, key=lambda idx: out_fetch[ib, idx])
                if kbas == sbas:
                    raise ValueError("routing_truncate hammer cannot choose distinct basins")
                apply_kill(ib, kbas, sbas)

    return RoutingKillBas(
        basin_count=out_basin_count,
        basin_area=out_basin_area,
        basin_topoind=out_basin_topoind,
        basin_gravel=out_basin_gravel,
        basin_drainarea=out_basin_drainarea,
        fetch_basin=out_fetch,
        basin_id=out_basin_id,
        basin_flowdir=out_basin_flowdir,
        outflow_grid=out_grid,
        outflow_basin=out_basin,
        inflow_number=out_inflow_number,
        inflow_grid=out_inflow_grid,
        inflow_basin=out_inflow_basin,
    )


def routing_basins_post_aggregate(
    *,
    neighbours,
    resolution,
    contfrac,
    lalo,
    sub_index,
    sub_area,
    lon_rel,
    lat_rel,
    trip,
    basins,
    topoindex,
    gravel,
    drainarea,
    hierarchy=None,
    nbasmax: int,
    num_largest: int,
    min_topoind: float,
    invented_basins: float | None = None,
) -> RoutingBasinsPostAggregate:
    """Build routing topology from explicit post-`aggregate` source-grid cells.

    Fortran provenance: ``routing.f90::routing_basins`` lines 7277-7501,
    `routing_getgrid`, `routing_findbasins`, `routing_globalize`,
    `routing_linkup`, `routing_fetch`, and `routing_truncate`. This wrapper
    starts after the source `aggregate` call: it zeroes negative sub-areas,
    extracts each coarse grid, handles the non-splitting `routing_findbasins`
    path, globalizes basins, links global outflows, computes fetch, reduces to
    `nbasmax`, and finalizes route arrays. Local multi-outflow cases requiring
    `routing_simplify`/`routing_cutbasin` are still surfaced by
    `routing_findbasins_simple` guards.
    """

    neighbours = np.asarray(neighbours, dtype=np.int64)
    resolution = np.asarray(resolution, dtype=np.float64)
    contfrac = np.asarray(contfrac, dtype=np.float64)
    lalo = np.asarray(lalo, dtype=np.float64)
    sub_index = np.asarray(sub_index, dtype=np.int64)
    sub_area = np.asarray(sub_area, dtype=np.float64).copy()
    sub_area[sub_area < 0.0] = 0.0
    trip_arr = np.asarray(trip, dtype=np.float64)
    basins_arr = np.asarray(basins, dtype=np.float64)
    topoindex_arr = np.asarray(topoindex, dtype=np.float64)
    if hierarchy is None:
        hierarchy_arr = routing_hierarchy(trip=trip_arr, topoindex=topoindex_arr)
    else:
        hierarchy_arr = np.asarray(hierarchy, dtype=np.float64)
    nbpt = sub_area.shape[0]
    if neighbours.shape[0] != nbpt or resolution.shape != (nbpt, 2) or contfrac.shape != (nbpt,) or lalo.shape != (nbpt, 2):
        raise ValueError("routing_basins_post_aggregate domain arrays have inconsistent shapes")

    sub_pts = np.count_nonzero(sub_area > 0.0, axis=1)
    nwbas = max(int(nbasmax), int(np.max(sub_pts)) if sub_pts.size else 0, neighbours.shape[1] + 1, 1)
    basin_count = np.zeros((nbpt,), dtype=np.int64)
    basin_area = np.zeros((nbpt, nwbas), dtype=np.float64)
    basin_hierarchy = np.zeros((nbpt, nwbas), dtype=np.float64)
    basin_topoind = np.zeros((nbpt, nwbas), dtype=np.float64)
    basin_gravel = np.zeros((nbpt, nwbas), dtype=np.float64)
    basin_drainarea = np.zeros((nbpt, nwbas), dtype=np.float64)
    basin_id = np.zeros((nbpt, nwbas), dtype=np.int64)
    basin_flowdir = np.zeros((nbpt, nwbas), dtype=np.int64)
    outflow_grid = np.zeros((nbpt, nwbas), dtype=np.int64)
    nbcoastal = np.zeros((nbpt,), dtype=np.int64)
    coastal_basin = np.zeros((nbpt, nwbas), dtype=np.int64)

    if invented_basins is None:
        valid_basins = basins_arr[basins_arr < 1.0e10]
        invented_basins = float(np.max(valid_basins)) if valid_basins.size else 0.0
    max_basins = float(invented_basins)

    for ib in range(nbpt):
        grid = routing_getgrid_from_subgrid(
            ib=ib + 1,
            sub_pts=sub_pts,
            sub_index=sub_index,
            sub_area=sub_area,
            lon_rel=lon_rel,
            lat_rel=lat_rel,
            lalo=lalo,
            resolution=resolution,
            contfrac=contfrac,
            trip=trip_arr,
            basins=basins_arr,
            topoindex=topoindex_arr,
            gravel=gravel,
            drainarea=drainarea,
            hierarchy=hierarchy_arr,
            max_basins=max_basins,
            min_topoind=min_topoind,
        )
        max_basins = grid.max_basins
        found = routing_findbasins_simple(trip=grid.trip_bx, basin=grid.basin_bx)
        glob = routing_globalize_one_grid(
            ib=ib + 1,
            neighbours=neighbours,
            area_bx=grid.area_bx,
            trip_bx=found.trip,
            hierarchy_bx=grid.hierarchy_bx,
            topoind_bx=grid.topoind_bx,
            gravel_bx=grid.gravel_bx,
            drainarea_bx=grid.drainarea_bx,
            min_topoind=min_topoind,
            nb_basin=found.nb_basin,
            basin_inbxid=found.basin_inbxid,
            basin_sz=found.basin_sz,
            basin_pts=found.basin_pts,
            basin_bxout=found.basin_bxout,
            coast_pts=found.coast_pts,
            basin_count=basin_count,
            basin_area=basin_area,
            basin_hierarchy=basin_hierarchy,
            basin_topoind=basin_topoind,
            basin_gravel=basin_gravel,
            basin_drainarea=basin_drainarea,
            basin_id=basin_id,
            basin_flowdir=basin_flowdir,
            outflow_grid=outflow_grid,
            nbcoastal=nbcoastal,
            coastal_basin=coastal_basin,
        )
        basin_count = glob.basin_count
        basin_area = glob.basin_area
        basin_hierarchy = glob.basin_hierarchy
        basin_topoind = glob.basin_topoind
        basin_gravel = glob.basin_gravel
        basin_drainarea = glob.basin_drainarea
        basin_id = glob.basin_id
        basin_flowdir = glob.basin_flowdir
        outflow_grid = glob.outflow_grid
        nbcoastal = glob.nbcoastal
        coastal_basin = glob.coastal_basin

    linked = routing_linkup(
        contfrac=contfrac,
        neighbours=neighbours,
        basin_count=basin_count,
        basin_area=basin_area,
        basin_id=basin_id,
        basin_flowdir=basin_flowdir,
        basin_hierarchy=basin_hierarchy,
        outflow_grid=outflow_grid,
        nbcoastal=nbcoastal,
        coastal_basin=coastal_basin,
        invented_basins=float(invented_basins),
    )
    fetched = routing_fetch(
        resolution=resolution,
        contfrac=contfrac,
        basin_count=basin_count,
        basin_area=basin_area,
        basin_id=basin_id,
        outflow_grid=linked.outflow_grid,
        outflow_basin=linked.outflow_basin,
        num_largest=num_largest,
    )
    reduced = routing_truncate_reduce_to_nbasmax(
        basin_count=basin_count,
        basin_area=fetched.basin_area,
        basin_topoind=basin_topoind,
        basin_gravel=basin_gravel,
        basin_drainarea=basin_drainarea,
        fetch_basin=fetched.fetch_basin,
        basin_id=basin_id,
        basin_flowdir=basin_flowdir,
        outflow_grid=fetched.outflow_grid,
        outflow_basin=linked.outflow_basin,
        inflow_number=linked.inflow_number,
        inflow_grid=linked.inflow_grid,
        inflow_basin=linked.inflow_basin,
        nbasmax=nbasmax,
    )
    finalized = routing_truncate_finalize_no_reduction(
        resolution=resolution,
        contfrac=contfrac,
        basin_count=reduced.basin_count,
        basin_area=reduced.basin_area,
        basin_topoind=reduced.basin_topoind,
        basin_gravel=reduced.basin_gravel,
        basin_drainarea=reduced.basin_drainarea,
        basin_id=reduced.basin_id,
        outflow_grid=reduced.outflow_grid,
        outflow_basin=reduced.outflow_basin,
        nbasmax=nbasmax,
        num_largest=num_largest,
    )
    return RoutingBasinsPostAggregate(
        routing_area=finalized.routing_area,
        topo_resid=finalized.topo_resid,
        basgravel=finalized.basgravel,
        basdrainarea=finalized.basdrainarea,
        global_basinid=finalized.global_basinid,
        route_togrid=finalized.route_togrid,
        route_tobasin=finalized.route_tobasin,
        basin_count=reduced.basin_count,
    )


def routing_basins_from_map_fields(
    *,
    neighbours,
    resolution,
    contfrac,
    lalo,
    map_fields: RoutingMapFields,
    nbasmax: int,
    num_largest: int,
    min_topoind: float | None = None,
    invented_basins: float | None = None,
    undef_sechiba=1.0e20,
) -> RoutingBasinsPostAggregate:
    """Build routing topology from complete high-resolution routing map fields.

    Fortran provenance: ``routing.f90::routing_basins`` lines 7108-7501 plus
    ``interpol_help.f90::aggregate_2d`` lines 46-466. This wrapper starts
    after the NetCDF variables are already loaded, builds the source mask from
    finite `trip`, computes `sub_index/sub_area` through the audited regular
    lon-lat overlap helper, derives `min_topoind`/`invented_basins` from the
    source fields when not supplied, and then delegates to the post-aggregate
    topology chain. It remains strict about complete source fields; no gravel
    or drainage defaults are fabricated.
    """

    from jax_orchidee.driver.static import aggregate_2d_overlap

    trip = np.asarray(map_fields.trip, dtype=np.float64)
    topoindex = np.asarray(map_fields.topoindex, dtype=np.float64)
    basins = np.asarray(map_fields.basins, dtype=np.float64)
    mask = trip < float(undef_sechiba)
    sub_index, sub_area = aggregate_2d_overlap(
        lalo,
        resolution,
        map_fields.lon_rel,
        map_fields.lat_rel,
        mask,
    )
    if min_topoind is None:
        valid_topo = topoindex[(trip < 1.0e10) & (topoindex < float(undef_sechiba) - 1.0)]
        if valid_topo.size == 0:
            raise ValueError("routing_basins_from_map_fields cannot derive min_topoind from empty valid topoindex")
        min_topoind = float(np.min(valid_topo))
    if invented_basins is None:
        valid_basins = basins[basins < 1.0e10]
        invented_basins = float(np.max(valid_basins)) if valid_basins.size else 0.0

    return routing_basins_post_aggregate(
        neighbours=neighbours,
        resolution=resolution,
        contfrac=contfrac,
        lalo=lalo,
        sub_index=sub_index,
        sub_area=sub_area,
        lon_rel=map_fields.lon_rel,
        lat_rel=map_fields.lat_rel,
        trip=trip,
        basins=basins,
        topoindex=topoindex,
        gravel=map_fields.gravel,
        drainarea=map_fields.drainarea,
        hierarchy=None,
        nbasmax=nbasmax,
        num_largest=num_largest,
        min_topoind=float(min_topoind),
        invented_basins=float(invented_basins),
    )


def routing_simplify(*, trip, basin, hierarchy, basin_inbxid: int) -> RoutingSimplify:
    """Simplify redundant same-border outflows for one local basin.

    Fortran provenance: ``routing.f90::routing_simplify`` lines 8143-8311.
    The source isolates one basin, uses `routing_findrout` to assign each cell
    to an outflow point, and for duplicate cardinal border outflows
    101/103/105/107 redirects the sub-basin with the largest outflow hierarchy
    toward a neighboring sub-basin with the same border outflow, preserving the
    source search order over directions 1-8.
    """

    trip_arr = np.asarray(trip, dtype=np.int64).copy()
    basin_arr = np.asarray(basin, dtype=np.int64)
    hierarchy = np.asarray(hierarchy, dtype=np.float64)
    if trip_arr.shape != basin_arr.shape or trip_arr.shape != hierarchy.shape:
        raise ValueError("trip, basin, and hierarchy must have the same two-dimensional shape")
    nbi, nbj = trip_arr.shape
    inc = np.asarray(
        [
            [0, -1],
            [1, -1],
            [1, 0],
            [1, 1],
            [0, 1],
            [-1, 1],
            [-1, 0],
            [-1, -1],
        ],
        dtype=np.int64,
    )
    trip_tmp = -np.ones_like(trip_arr, dtype=np.int64)
    basin_sz = 0
    for ip in range(nbi):
        for jp in range(nbj):
            if basin_arr[ip, jp] == int(basin_inbxid):
                trip_tmp[ip, jp] = trip_arr[ip, jp]
                basin_sz += 1

    route = routing_findrout(trip=trip_tmp, basin_sz=basin_sz, basinid=int(basin_inbxid))
    trip_flow = route.trip_flow.copy()
    outflow = route.outflow

    for iborder in range(101, 108, 2):
        icc = int(np.count_nonzero(trip_tmp == iborder)) - 1
        while icc > 0:
            todopt: list[int] = []
            todohi: list[float] = []
            for iout in range(route.nbout):
                oi = int(outflow[iout, 0]) - 1
                oj = int(outflow[iout, 1]) - 1
                if trip_tmp[oi, oj] == iborder:
                    todopt.append(iout)
                    todohi.append(float(hierarchy[oi, oj]))
            if len(todopt) < 2:
                break
            ismall = todopt[int(np.argmax(np.asarray(todohi, dtype=np.float64)))]
            small_i = int(outflow[ismall, 0])
            small_j = int(outflow[ismall, 1])

            changed = False
            for ip in range(nbi):
                for jp in range(nbj):
                    if trip_flow[ip, jp, 0] == small_i and trip_flow[ip, jp, 1] == small_j:
                        not_found = True
                        for ib in range(len(todopt)):
                            if ib == int(np.argmax(np.asarray(todohi, dtype=np.float64))):
                                continue
                            ibas = todopt[ib]
                            target_i = int(outflow[ibas, 0])
                            target_j = int(outflow[ibas, 1])
                            for idir in range(8):
                                iip = ip + int(inc[idir, 0])
                                jjp = jp + int(inc[idir, 1])
                                if 0 <= iip < nbi and 0 <= jjp < nbj and not_found:
                                    if trip_flow[iip, jjp, 0] == target_i and trip_flow[iip, jjp, 1] == target_j:
                                        trip_flow[ip, jp, 0] = target_i
                                        trip_flow[ip, jp, 1] = target_j
                                        trip_tmp[ip, jp] = idir + 1
                                        not_found = False
                                        changed = True
                        if not_found and trip_tmp[ip, jp] == iborder:
                            raise ValueError("routing_simplify could not redirect duplicate border outflow")
            if not changed:
                raise ValueError("routing_simplify made no progress on duplicate border outflows")
            icc -= 1

    out_trip = trip_arr.copy()
    mask = trip_tmp > 0
    out_trip[mask] = trip_tmp[mask]
    return RoutingSimplify(trip=out_trip)


def routing_cutbasin(
    *,
    trip,
    basin,
    basin_inbxid: int,
    nbbasins: int,
    nbasmax: int,
    nb: int = 0,
) -> RoutingCutBasin:
    """Split one local basin into sub-basins by outflow point.

    Fortran provenance: ``routing.f90::routing_cutbasin`` lines 8331-8519.
    This helper follows the source's `routing_findrout` assignment, optionally
    absorbs one-cell boundary outflow fragments when the basin count is already
    at `nbasmax`, and emits one new basin per remaining outflow point.
    """

    trip_arr = np.asarray(trip, dtype=np.int64).copy()
    basin_arr = np.asarray(basin, dtype=np.int64)
    if trip_arr.shape != basin_arr.shape or trip_arr.ndim != 2:
        raise ValueError("trip and basin must be two-dimensional arrays with the same shape")
    nbi, nbj = trip_arr.shape
    inc = np.asarray(
        [
            [0, -1],
            [1, -1],
            [1, 0],
            [1, 1],
            [0, 1],
            [-1, 1],
            [-1, 0],
            [-1, -1],
        ],
        dtype=np.int64,
    )

    trip_tmp = -np.ones_like(trip_arr, dtype=np.int64)
    basin_sz = 0
    for ip in range(nbi):
        for jp in range(nbj):
            if basin_arr[ip, jp] == int(basin_inbxid):
                trip_tmp[ip, jp] = trip_arr[ip, jp]
                basin_sz += 1

    route = routing_findrout(trip=trip_tmp, basin_sz=basin_sz, basinid=int(basin_inbxid))
    trip_flow = route.trip_flow.copy()
    outflow = route.outflow.copy()
    outsz = route.outsz.copy()
    nbout = route.nbout

    if int(nbbasins) >= int(nbasmax):
        for ib in range(nbout):
            oi = int(outflow[ib, 0]) - 1
            oj = int(outflow[ib, 1]) - 1
            if outsz[ib] == 1 and trip_arr[oi, oj] > 99:
                not_found = True
                for idir in range(8):
                    iip = oi + int(inc[idir, 0])
                    jjp = oj + int(inc[idir, 1])
                    if 0 <= iip < nbi and 0 <= jjp < nbj and not_found:
                        if trip_tmp[iip, jjp] > 100:
                            not_found = False
                            trip_arr[oi, oj] = idir + 1
                            trip_tmp[oi, oj] = idir + 1
                            outsz[ib] = 0
                            for ibb in range(nbout):
                                if iip == outflow[ibb, 0] - 1 and jjp == outflow[ibb, 1] - 1:
                                    outsz[ibb] += 1
                                    trip_flow[oi, oj, 0] = outflow[ibb, 0]
                                    trip_flow[oi, oj, 1] = outflow[ibb, 1]

    active_outflows = [idx for idx in range(nbout) if outsz[idx] > 0]
    max_new = max(1, len(active_outflows))
    bname = np.zeros((int(nb) + max_new,), dtype=np.int64)
    sz = np.zeros((int(nb) + max_new,), dtype=np.int64)
    pts = np.zeros((int(nb) + max_new, max(nbi * nbj, 1), 2), dtype=np.int64)
    cur_nb = int(nb)
    if len(active_outflows) > 1:
        nb_in = cur_nb
        for ib in active_outflows:
            cur_nb += 1
            if cur_nb > nbi * nbj:
                raise ValueError("routing_cutbasin generated too many basins")
            out_i = int(outflow[ib, 0])
            out_j = int(outflow[ib, 1])
            bname[cur_nb - 1] = int(basin_inbxid)
            for ip in range(nbi):
                for jp in range(nbj):
                    if trip_flow[ip, jp, 0] + trip_flow[ip, jp, 0] > 0 and trip_flow[ip, jp, 0] == out_i and trip_flow[ip, jp, 1] == out_j:
                        pos = int(sz[cur_nb - 1])
                        pts[cur_nb - 1, pos, 0] = ip + 1
                        pts[cur_nb - 1, pos, 1] = jp + 1
                        sz[cur_nb - 1] += 1
        if int(np.sum(sz[nb_in:cur_nb])) != basin_sz:
            raise ValueError("routing_cutbasin lost points while splitting the basin")

    return RoutingCutBasin(
        trip=trip_arr,
        nb=int(cur_nb),
        bname=bname[:cur_nb],
        sz=sz[:cur_nb],
        pts=pts[:cur_nb, :, :],
    )


def routing_lake_step(
    *,
    lake_reservoir,
    lakeinflow,
    routing_area,
    humrel,
    dt_routing,
    doswamps: bool,
    ok_doc: bool = True,
    maxevap_lake=7.5 / 86400.0,
) -> RoutingLakeStep:
    """Run the source ``routing_lake`` update for explicit local state.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/routing.f90``,
    subroutine ``routing_lake`` lines 5390-5448. The source adds
    ``lakeinflow`` to ``lake_reservoir``, optionally returns lake water to the
    soil when ``doswamps`` is true, transports non-water species in proportion
    to lake reservoir concentration when ``ok_doc`` is true, then scales
    ``return_lakes``, ``lake_diag``, and remaining ``lakeinflow`` by total
    routing area.
    """

    reservoir = np.asarray(lake_reservoir, dtype=np.float64).copy()
    inflow = np.asarray(lakeinflow, dtype=np.float64).copy()
    area = np.asarray(routing_area, dtype=np.float64)
    humrel = np.asarray(humrel, dtype=np.float64)
    if reservoir.ndim != 2 or inflow.shape != reservoir.shape:
        raise ValueError("lake_reservoir and lakeinflow must share shape (npts,nflow)")
    npts, nflow = reservoir.shape
    if area.ndim != 2 or area.shape[0] != npts:
        raise ValueError("routing_area must have shape (npts,nbas)")
    if humrel.shape != (npts,):
        raise ValueError("humrel must have shape (npts,)")
    total_area = np.sum(area, axis=1)
    if np.any(total_area <= 0.0):
        raise ValueError("routing_lake requires positive total routing area for each point")

    return_lakes = np.zeros_like(reservoir)
    reservoir += inflow
    if bool(doswamps):
        refill = np.maximum(0.0, float(maxevap_lake) * (1.0 - humrel) * float(dt_routing) * total_area)
        return_lakes[:, IH2O] = np.minimum(refill, reservoir[:, IH2O])
        if bool(ok_doc):
            active = reservoir[:, IH2O] > 0.0
            return_lakes[active, IH2O + 1 : nflow] = (
                return_lakes[active, IH2O, None]
                * reservoir[active, IH2O + 1 : nflow]
                / reservoir[active, IH2O, None]
            )
        reservoir -= return_lakes
        return_lakes = return_lakes / total_area[:, None]

    lake_diag = reservoir / total_area[:, None]
    inflow = inflow / total_area[:, None]
    return RoutingLakeStep(
        lake_reservoir=reservoir,
        lakeinflow=inflow,
        return_lakes=return_lakes,
        lake_diag=lake_diag,
    )


def routing_reservoir_outflow_step(
    *,
    fast_reservoir,
    slow_reservoir,
    stream_reservoir,
    topo_resid,
    route_tobasin,
    qflow_ave,
    stream_resave,
    stream_damavail,
    dt_routing,
    fast_tcst=3.0,
    slow_tcst=25.0,
    stream_tcst=0.24,
    flow_coef=1.01,
    one_day=86400.0,
    min_sechiba=1.0e-8,
) -> RoutingReservoirOutflowStep:
    """Compute fast/slow/stream reservoir outflows for explicit state.

    Fortran provenance: ``routing.f90::routing_flow`` lines 3253-3330 and
    3352-3388. The source first partitions stream climatology by each basin's
    share of upstream stream water, then computes water outflows for active
    routed basins. Non-water fast/slow outflows follow the corresponding water
    outflow concentration for all transported species; stream non-water outflow
    is applied only through `ico2aq`.
    """

    fast = np.asarray(fast_reservoir, dtype=np.float64).copy()
    slow = np.asarray(slow_reservoir, dtype=np.float64).copy()
    stream = np.asarray(stream_reservoir, dtype=np.float64).copy()
    topo = np.asarray(topo_resid, dtype=np.float64)
    route = np.asarray(route_tobasin)
    qflow_ave = np.asarray(qflow_ave, dtype=np.float64)
    stream_resave = np.asarray(stream_resave, dtype=np.float64)
    stream_damavail = np.asarray(stream_damavail, dtype=np.float64)
    if fast.ndim != 3 or slow.shape != fast.shape or stream.shape != fast.shape:
        raise ValueError("fast_reservoir, slow_reservoir, and stream_reservoir must share shape (npts,nbas,nflow)")
    npts, nbas, nflow = fast.shape
    if topo.shape != (npts, nbas) or route.shape != (npts, nbas):
        raise ValueError("topo_resid and route_tobasin must have shape (npts,nbas)")
    if qflow_ave.shape != (npts,) or stream_resave.shape != (npts,) or stream_damavail.shape != (npts,):
        raise ValueError("qflow_ave, stream_resave, and stream_damavail must have shape (npts,)")

    qflow_avebas = np.zeros((npts, nbas), dtype=np.float64)
    stream_resavebas = np.zeros((npts, nbas), dtype=np.float64)
    stream_damavailbas = np.zeros((npts, nbas), dtype=np.float64)
    tot_upstream = np.sum(stream[:, :, IH2O], axis=1)
    for ig in range(npts):
        for ib in range(nbas):
            if tot_upstream[ig] > min_sechiba:
                share = stream[ig, ib, IH2O] / tot_upstream[ig]
                qflow_avebas[ig, ib] = qflow_ave[ig] * share
                stream_resavebas[ig, ib] = stream_resave[ig] * share
                stream_damavailbas[ig, ib] = stream_damavail[ig] * share
            else:
                qflow_avebas[ig, ib] = 0.0
                stream_resavebas[ig, ib] = stream[ig, ib, IH2O]
                stream_damavailbas[ig, ib] = 0.0

    fast_flow = np.zeros_like(fast)
    slow_flow = np.zeros_like(slow)
    stream_flow = np.zeros_like(stream)
    dt_factor = float(one_day) / float(dt_routing)
    for ig in range(npts):
        for ib in range(nbas):
            if route[ig, ib] > 0:
                fast_denom = (topo[ig, ib] / 1000.0) * float(fast_tcst) * dt_factor
                slow_denom = (topo[ig, ib] / 1000.0) * float(slow_tcst) * dt_factor
                fast_raw = min((fast[ig, ib, IH2O] ** float(flow_coef)) / fast_denom, fast[ig, ib, IH2O] - min_sechiba)
                slow_raw = min((slow[ig, ib, IH2O] ** float(flow_coef)) / slow_denom, slow[ig, ib, IH2O] - min_sechiba)
                fast_flow[ig, ib, IH2O] = max(fast_raw, 0.0)
                slow_flow[ig, ib, IH2O] = max(slow_raw, 0.0)
                if stream_resavebas[ig, ib] > 0.0:
                    stream_factor = stream_resavebas[ig, ib] / (stream_damavailbas[ig, ib] + stream_resavebas[ig, ib])
                    stream_denom = (topo[ig, ib] / 1000.0) * float(stream_tcst) * stream_factor * dt_factor
                    stream_raw = min(
                        (stream[ig, ib, IH2O] ** float(flow_coef)) / stream_denom,
                        stream[ig, ib, IH2O] - min_sechiba,
                    )
                else:
                    stream_raw = 0.0
                stream_flow[ig, ib, IH2O] = max(stream_raw, 0.0)

    for ib in range(nbas):
        for iflow in range(IH2O + 1, nflow):
            active_fast = fast[:, ib, IH2O] > min_sechiba
            fast_flow[active_fast, ib, iflow] = (
                fast[active_fast, ib, iflow] * fast_flow[active_fast, ib, IH2O] / fast[active_fast, ib, IH2O]
            )
            fast[:, ib, iflow] -= fast_flow[:, ib, iflow]

            active_slow = slow[:, ib, IH2O] > min_sechiba
            slow_flow[active_slow, ib, iflow] = (
                slow[active_slow, ib, iflow] * slow_flow[active_slow, ib, IH2O] / slow[active_slow, ib, IH2O]
            )
            slow[:, ib, iflow] -= slow_flow[:, ib, iflow]

            if iflow <= ICO2AQ:
                active_stream = stream[:, ib, IH2O] > min_sechiba
                stream_flow[active_stream, ib, iflow] = (
                    stream[active_stream, ib, iflow]
                    * stream_flow[active_stream, ib, IH2O]
                    / stream[active_stream, ib, IH2O]
                )
                stream[:, ib, iflow] -= stream_flow[:, ib, iflow]

    return RoutingReservoirOutflowStep(
        fast_flow=fast_flow,
        slow_flow=slow_flow,
        stream_flow=stream_flow,
        fast_reservoir=fast,
        slow_reservoir=slow,
        stream_reservoir=stream,
        qflow_avebas=qflow_avebas,
        stream_resavebas=stream_resavebas,
        stream_damavailbas=stream_damavailbas,
    )


def routing_daily_scaled_outputs(
    *,
    returnflow_mean,
    reinfiltration_mean,
    irrigation_mean,
    sed_deposition_mean,
    poc_deposition_mean,
    rivbed2fld_sed,
    rivbed2fld_poc,
    riverflow_mean,
    coastalflow_mean,
    hydrographs,
    slowflow_diag,
    dt_routing,
    dt_sechiba,
    mille=1000.0,
) -> RoutingDailyScaledOutputs:
    """Scale daily routing means back to SECHIBA-timestep outputs.

    Fortran provenance: ``routing.f90::routing_main`` lines 1115-1135 and
    1139-1140. Water/matter return terms are scaled by
    ``dt_sechiba/dt_routing``; river/coastal/hydrograph water-volume terms are
    additionally divided by ``mille``; sediment and POC deposition outputs add
    the river-bed-to-floodplain transfer after the same timestep scaling.
    """

    scale = float(dt_sechiba) / float(dt_routing)
    volume_scale = scale / float(mille)
    returnflow = np.asarray(returnflow_mean, dtype=np.float64) * scale
    reinfiltration = np.asarray(reinfiltration_mean, dtype=np.float64) * scale
    irrigation = np.asarray(irrigation_mean, dtype=np.float64) * scale
    sed = np.asarray(sed_deposition_mean, dtype=np.float64) * scale
    poc = np.asarray(poc_deposition_mean, dtype=np.float64) * scale
    rivbed_sed = np.asarray(rivbed2fld_sed, dtype=np.float64) * scale
    rivbed_poc = np.asarray(rivbed2fld_poc, dtype=np.float64) * scale
    if sed.shape != rivbed_sed.shape:
        raise ValueError("sed_deposition_mean and rivbed2fld_sed must share shape")
    if poc.shape != rivbed_poc.shape:
        raise ValueError("poc_deposition_mean and rivbed2fld_poc must share shape")
    return RoutingDailyScaledOutputs(
        returnflow=returnflow,
        reinfiltration=reinfiltration,
        irrigation=irrigation,
        sed_depositiontot=sed + rivbed_sed,
        poc_depositiontot=poc + rivbed_poc,
        riverflow=np.asarray(riverflow_mean, dtype=np.float64) * volume_scale,
        coastalflow=np.asarray(coastalflow_mean, dtype=np.float64) * volume_scale,
        hydrographs=np.asarray(hydrographs, dtype=np.float64) * volume_scale,
        slowflow_diag=np.asarray(slowflow_diag, dtype=np.float64) * scale,
    )


def routing_return_reinfiltration_step(
    *,
    return_swamp,
    pond_drainage,
    flood_drainage,
    routing_area,
    do_floodplains: bool,
    doswamps: bool,
    doponds: bool,
) -> RoutingReturnReinfiltrationStep:
    """Aggregate routing returnflow and reinfiltration to grid-cell fluxes.

    Fortran provenance: ``routing.f90::routing_flow`` lines 4940-4970. The
    source initializes both outputs to zero, sums swamp returnflow and
    pond/flood drainage over basins only for transported species through
    ``ico2aq``, and divides those slots by total routing area when any of
    ``do_floodplains``, ``doswamps``, or ``doponds`` is active. Otherwise both
    outputs remain zero.
    """

    ret_swamp = np.asarray(return_swamp, dtype=np.float64)
    pond = np.asarray(pond_drainage, dtype=np.float64)
    flood = np.asarray(flood_drainage, dtype=np.float64)
    area = np.asarray(routing_area, dtype=np.float64)
    if ret_swamp.ndim != 3 or pond.shape != ret_swamp.shape or flood.shape != ret_swamp.shape:
        raise ValueError("return_swamp, pond_drainage, and flood_drainage must share shape (npts,nbas,nflow)")
    npts, _nbas, nflow = ret_swamp.shape
    if area.ndim != 2 or area.shape[0] != npts:
        raise ValueError("routing_area must have shape (npts,nbas)")
    total_area = np.sum(area, axis=1)
    if np.any(total_area <= 0.0):
        raise ValueError("routing return/reinfiltration requires positive total routing area")
    returnflow = np.zeros((npts, nflow), dtype=np.float64)
    reinfiltration = np.zeros((npts, nflow), dtype=np.float64)
    if bool(do_floodplains) or bool(doswamps) or bool(doponds):
        last = min(ICO2AQ + 1, nflow)
        returnflow[:, :last] = np.sum(ret_swamp[:, :, :last], axis=1) / total_area[:, None]
        reinfiltration[:, :last] = np.sum(pond[:, :, :last] + flood[:, :, :last], axis=1) / total_area[:, None]
    return RoutingReturnReinfiltrationStep(returnflow=returnflow, reinfiltration=reinfiltration)


def routing_flow_diagnostics_step(
    *,
    runoff,
    drainage,
    routing_area,
    fast_flow,
    slow_flow,
    stream_flow,
    flood_flow,
    pond_inflow,
    transport,
    return_swamp,
    floods,
    previous_flood_diag,
    previous_pond_diag,
    lake_diag,
    hydrodiag,
    fast_reservoir,
    slow_reservoir,
    stream_reservoir,
    flood_reservoir,
    pond_reservoir,
    irrig_actual,
    irrig_adduct,
    flood_dep_sed,
    flood_dep_poc,
    stream_erodep,
    stream_seddep,
    streamb_inflow,
    poc_co2_rivbed,
    poc_doc_rivbed,
    undef_sechiba=1.0e20,
) -> RoutingFlowDiagnosticsStep:
    """Aggregate end-of-``routing_flow`` diagnostics and lateral outputs.

    Fortran provenance: ``routing.f90::routing_flow`` lines 5219-5329. The
    source forms netflow diagnostics, resets daily diagnostic outputs, sums
    basin reservoirs/flows/deposition terms to grid cells, divides grid-cell
    state diagnostics by total routing area, clips floodplain deposition and
    river-bed POC transformation diagnostics, copies lake/coastal/river
    transport outlet slots, then sets ``flood_res`` and ``fastr``.
    """

    runoff = np.asarray(runoff, dtype=np.float64)
    drainage = np.asarray(drainage, dtype=np.float64)
    area = np.asarray(routing_area, dtype=np.float64)
    fast_flow = np.asarray(fast_flow, dtype=np.float64)
    slow_flow = np.asarray(slow_flow, dtype=np.float64)
    stream_flow = np.asarray(stream_flow, dtype=np.float64)
    flood_flow = np.asarray(flood_flow, dtype=np.float64)
    pond_inflow = np.asarray(pond_inflow, dtype=np.float64)
    transport = np.asarray(transport, dtype=np.float64)
    ret_swamp = np.asarray(return_swamp, dtype=np.float64)
    floods = np.asarray(floods, dtype=np.float64)
    fast_res = np.asarray(fast_reservoir, dtype=np.float64)
    slow_res = np.asarray(slow_reservoir, dtype=np.float64)
    stream_res = np.asarray(stream_reservoir, dtype=np.float64)
    flood_reservoir = np.asarray(flood_reservoir, dtype=np.float64)
    pond_res = np.asarray(pond_reservoir, dtype=np.float64)
    hydrodiag = np.asarray(hydrodiag)
    irrig_actual = np.asarray(irrig_actual, dtype=np.float64)
    irrig_adduct = np.asarray(irrig_adduct, dtype=np.float64)
    flood_dep_sed = np.asarray(flood_dep_sed, dtype=np.float64)
    flood_dep_poc = np.asarray(flood_dep_poc, dtype=np.float64)
    stream_erodep = np.asarray(stream_erodep, dtype=np.float64)
    stream_seddep = np.asarray(stream_seddep, dtype=np.float64)
    streamb_inflow = np.asarray(streamb_inflow, dtype=np.float64)
    old_flood = np.asarray(previous_flood_diag, dtype=np.float64)
    old_pond = np.asarray(previous_pond_diag, dtype=np.float64)
    lake_diag = np.asarray(lake_diag, dtype=np.float64)
    poc_co2 = np.asarray(poc_co2_rivbed, dtype=np.float64)
    poc_doc = np.asarray(poc_doc_rivbed, dtype=np.float64)

    if runoff.ndim != 2 or drainage.shape != runoff.shape:
        raise ValueError("runoff and drainage must share shape (npts,nflow)")
    npts, nflow = runoff.shape
    if area.ndim != 2 or area.shape[0] != npts:
        raise ValueError("routing_area must have shape (npts,nbas)")
    nbas = area.shape[1]
    basin_shape = (npts, nbas, nflow)
    for name, arr in (
        ("fast_flow", fast_flow),
        ("slow_flow", slow_flow),
        ("stream_flow", stream_flow),
        ("flood_flow", flood_flow),
        ("pond_inflow", pond_inflow),
        ("return_swamp", ret_swamp),
        ("floods", floods),
        ("fast_reservoir", fast_res),
        ("slow_reservoir", slow_res),
        ("stream_reservoir", stream_res),
        ("flood_reservoir", flood_reservoir),
        ("irrig_actual", irrig_actual),
        ("irrig_adduct", irrig_adduct),
        ("stream_erodep", stream_erodep),
        ("streamb_inflow", streamb_inflow),
    ):
        if arr.shape != basin_shape:
            raise ValueError(f"{name} must have shape (npts,nbas,nflow)")
    if transport.shape[0] != npts or transport.shape[1] < nbas + 3 or transport.shape[2] != nflow:
        raise ValueError("transport must have shape (npts,nbas+3-or-more,nflow)")
    if hydrodiag.shape != basin_shape:
        raise ValueError("hydrodiag must have shape (npts,nbas,nflow)")
    if old_flood.shape != (npts, nflow) or old_pond.shape != (npts, nflow) or lake_diag.shape != (npts, nflow):
        raise ValueError("previous_flood_diag, previous_pond_diag, and lake_diag must have shape (npts,nflow)")
    if pond_res.shape != (npts, nflow):
        raise ValueError("pond_reservoir must have shape (npts,nflow)")
    if flood_dep_sed.shape != (npts, nbas, 3) or flood_dep_poc.shape != (npts, nbas, 3):
        raise ValueError("flood_dep_sed and flood_dep_poc must have shape (npts,nbas,3)")
    if stream_seddep.shape != (npts, nbas, nflow):
        raise ValueError("stream_seddep must have shape (npts,nbas,nflow)")
    if poc_co2.shape != (npts,) or poc_doc.ndim != 2 or poc_doc.shape[0] != npts:
        raise ValueError("poc_co2_rivbed must be (npts,) and poc_doc_rivbed must be (npts,ndoc)")
    if nflow <= ISANDSED:
        raise ValueError("routing diagnostics with sediment/POC require nflow >= 10")

    total_area = np.sum(area, axis=1)
    if np.any(total_area <= 0.0):
        raise ValueError("routing diagnostics require positive total routing area")

    netflow_fast_diag = np.sum(runoff[:, None, :] * area[:, :, None] - fast_flow - pond_inflow, axis=1)
    netflow_slow_diag = np.sum(drainage[:, None, :] * area[:, :, None] - slow_flow, axis=1)
    netflow_stream_diag = np.sum(flood_flow + transport[:, :nbas, :] - stream_flow - ret_swamp - floods, axis=1)
    netflow_fast_diag = netflow_fast_diag / total_area[:, None]
    netflow_slow_diag = netflow_slow_diag / total_area[:, None]
    netflow_stream_diag = netflow_stream_diag / total_area[:, None]

    delsurfstor = -old_flood[:, IH2O] - old_pond[:, IH2O] - lake_diag[:, IH2O]
    hydrographs = np.zeros((npts, nflow), dtype=np.float64)
    slowflow_diag = np.zeros((npts,), dtype=np.float64)
    for ig in range(npts):
        for ib in range(nbas):
            if hydrodiag[ig, ib, IH2O] > 0:
                hydrographs[ig, :] += fast_flow[ig, ib, :] + slow_flow[ig, ib, :] + stream_flow[ig, ib, :]
                slowflow_diag[ig] += slow_flow[ig, ib, IH2O]

    fast_diag = np.sum(fast_res, axis=1) / total_area[:, None]
    slow_diag = np.sum(slow_res, axis=1) / total_area[:, None]
    stream_diag = np.sum(stream_res, axis=1) / total_area[:, None]
    flood_diag = np.sum(flood_reservoir, axis=1) / total_area[:, None]
    pond_diag = pond_res / total_area[:, None]
    irrigation = np.sum(irrig_actual + irrig_adduct, axis=1) / total_area[:, None]
    sed_deposition = np.clip(np.sum(flood_dep_sed, axis=1) / total_area[:, None], 0.0, float(undef_sechiba))
    poc_deposition = np.clip(np.sum(flood_dep_poc, axis=1) / total_area[:, None], 0.0, float(undef_sechiba))
    poc_co2 = np.clip(1.0e3 * poc_co2 / total_area, 0.0, float(undef_sechiba))
    poc_doc = np.clip(1.0e3 * poc_doc / total_area[:, None], 0.0, float(undef_sechiba))
    riv_erodep_sed = np.sum(stream_erodep[:, :, ICLAYSED : ISANDSED + 1], axis=1)
    riv_erodep_poc = np.sum(stream_erodep[:, :, IPOCA : IPOCP + 1], axis=1)
    rivchannel_deposition = np.sum(stream_seddep, axis=1)
    stream_inflow = np.sum(streamb_inflow, axis=1)
    stream_outflow = np.sum(stream_flow, axis=1)
    flood_daily = np.sum(floods[:, :, IH2O], axis=1)
    lakeinflow = transport[:, nbas, :].copy()
    coastalflow = transport[:, nbas + 1, :].copy()
    riverflow = transport[:, nbas + 2, :].copy()
    flood_res = flood_diag[:, IH2O] + pond_diag[:, IH2O]
    fastr = fast_diag[:, IH2O] + slow_diag[:, IH2O]

    return RoutingFlowDiagnosticsStep(
        delsurfstor=delsurfstor,
        netflow_fast_diag=netflow_fast_diag,
        netflow_slow_diag=netflow_slow_diag,
        netflow_stream_diag=netflow_stream_diag,
        hydrographs=hydrographs,
        slowflow_diag=slowflow_diag,
        fast_diag=fast_diag,
        slow_diag=slow_diag,
        stream_diag=stream_diag,
        flood_diag=flood_diag,
        pond_diag=pond_diag,
        irrigation=irrigation,
        sed_deposition=sed_deposition,
        poc_deposition=poc_deposition,
        riv_erodep_sed=riv_erodep_sed,
        riv_erodep_poc=riv_erodep_poc,
        rivchannel_deposition=rivchannel_deposition,
        lakeinflow=lakeinflow,
        coastalflow=coastalflow,
        riverflow=riverflow,
        stream_inflow=stream_inflow,
        stream_outflow=stream_outflow,
        flood_daily=flood_daily,
        flood_res=flood_res,
        fastr=fastr,
        poc_co2_rivbed=poc_co2,
        poc_doc_rivbed=poc_doc,
    )


def routing_lake_overflow_step(
    *,
    lake_reservoir,
    coastalflow,
    routing_area,
    mask_coast,
    nb_coast_gridcells,
    max_lake_reservoir=7000.0,
    min_sechiba=1.0e-8,
) -> RoutingLakeOverflowStep:
    """Apply the ``routing_flow`` lake-reservoir cap and coastal redistribution.

    Fortran provenance: ``routing.f90::routing_flow`` lines 5331-5366.
    Water above ``max_lake_reservoir * totarea`` is removed from each grid
    cell's lake reservoir; transported species follow lake concentration when
    water storage is positive. The global overflow sum is then distributed
    uniformly to coastal grid cells through ``mask_coast`` when at least one
    coastal grid cell exists.
    """

    reservoir = np.asarray(lake_reservoir, dtype=np.float64).copy()
    coast = np.asarray(coastalflow, dtype=np.float64).copy()
    area = np.asarray(routing_area, dtype=np.float64)
    mask = np.asarray(mask_coast, dtype=np.float64)
    if reservoir.ndim != 2 or coast.shape != reservoir.shape:
        raise ValueError("lake_reservoir and coastalflow must share shape (npts,nflow)")
    npts, nflow = reservoir.shape
    if area.ndim != 2 or area.shape[0] != npts:
        raise ValueError("routing_area must have shape (npts,nbas)")
    if mask.shape != (npts,):
        raise ValueError("mask_coast must have shape (npts,)")
    total_area = np.sum(area, axis=1)
    if np.any(total_area <= 0.0):
        raise ValueError("routing lake overflow requires positive total routing area")

    overflow = np.zeros_like(reservoir)
    overflow[:, IH2O] = np.maximum(0.0, reservoir[:, IH2O] - float(max_lake_reservoir) * total_area)
    active = reservoir[:, IH2O] > float(min_sechiba)
    overflow[active, IH2O + 1 : nflow] = (
        overflow[active, IH2O, None] * reservoir[active, IH2O + 1 : nflow] / reservoir[active, IH2O, None]
    )
    reservoir -= overflow
    total_overflow = np.sum(overflow, axis=0)
    overflow_coast = np.zeros_like(reservoir)
    if int(nb_coast_gridcells) >= 1:
        overflow_coast = total_overflow[None, :] / float(nb_coast_gridcells) * mask[:, None]
        coast += overflow_coast

    return RoutingLakeOverflowStep(
        lake_reservoir=reservoir,
        coastalflow=coast,
        lake_overflow=overflow,
        lake_overflow_coast=overflow_coast,
        total_lake_overflow=total_overflow,
    )


def routing_reservoir_update_step(
    *,
    runoff,
    drainage,
    routing_area,
    fast_reservoir,
    slow_reservoir,
    stream_reservoir,
    flood_reservoir,
    pond_reservoir,
    transport,
    return_swamp,
    floods,
    flood_flow,
    pond_inflow,
    pond_drainage,
    min_sechiba=1.0e-8,
) -> RoutingReservoirUpdateStep:
    """Apply the main ``routing_flow`` reservoir writeback and correction pass.

    Fortran provenance: ``routing.f90::routing_flow`` lines 4208-4372. The
    source adds runoff/drainage/transport/flood/pond terms to their reservoirs,
    writes ``streamb_inflow``, clears POC/sediment slots in ``return_swamp``,
    cascades negative water-storage corrections from flood to stream to fast
    to slow reservoirs, errors on materially negative slow reservoirs, and
    recomputes ``totflood`` after the update.
    """

    runoff = np.asarray(runoff, dtype=np.float64)
    drainage = np.asarray(drainage, dtype=np.float64)
    area = np.asarray(routing_area, dtype=np.float64)
    fast = np.asarray(fast_reservoir, dtype=np.float64).copy()
    slow = np.asarray(slow_reservoir, dtype=np.float64).copy()
    stream = np.asarray(stream_reservoir, dtype=np.float64).copy()
    flood = np.asarray(flood_reservoir, dtype=np.float64).copy()
    pond = np.asarray(pond_reservoir, dtype=np.float64).copy()
    transport = np.asarray(transport, dtype=np.float64)
    ret_swamp = np.asarray(return_swamp, dtype=np.float64).copy()
    floods = np.asarray(floods, dtype=np.float64)
    flood_flow = np.asarray(flood_flow, dtype=np.float64)
    pond_inflow = np.asarray(pond_inflow, dtype=np.float64)
    pond_drainage = np.asarray(pond_drainage, dtype=np.float64)

    if runoff.ndim != 2 or drainage.shape != runoff.shape:
        raise ValueError("runoff and drainage must share shape (npts,nflow)")
    npts, nflow = runoff.shape
    if area.ndim != 2 or area.shape[0] != npts:
        raise ValueError("routing_area must have shape (npts,nbas)")
    nbas = area.shape[1]
    basin_shape = (npts, nbas, nflow)
    for name, arr in (
        ("fast_reservoir", fast),
        ("slow_reservoir", slow),
        ("stream_reservoir", stream),
        ("flood_reservoir", flood),
        ("return_swamp", ret_swamp),
        ("floods", floods),
        ("flood_flow", flood_flow),
        ("pond_inflow", pond_inflow),
        ("pond_drainage", pond_drainage),
    ):
        if arr.shape != basin_shape:
            raise ValueError(f"{name} must have shape (npts,nbas,nflow)")
    if pond.shape != (npts, nflow):
        raise ValueError("pond_reservoir must have shape (npts,nflow)")
    if transport.shape[0] != npts or transport.shape[1] < nbas or transport.shape[2] != nflow:
        raise ValueError("transport must have shape (npts,nbas-or-more,nflow)")
    if nflow <= ISANDSED:
        raise ValueError("routing reservoir update requires nflow >= 10")

    streamb_inflow = np.zeros(basin_shape, dtype=np.float64)
    for ig in range(npts):
        for ib in range(nbas):
            fast[ig, ib, :] = fast[ig, ib, :] + runoff[ig, :] * area[ig, ib]
            slow[ig, ib, :] = slow[ig, ib, :] + drainage[ig, :] * area[ig, ib]
            stream[ig, ib, :] = stream[ig, ib, :] + transport[ig, ib, :] - ret_swamp[ig, ib, :] - floods[ig, ib, :]
            flood[ig, ib, :] = flood[ig, ib, :] + floods[ig, ib, :]
            pond[ig, :] = pond[ig, :] + pond_inflow[ig, ib, :] - pond_drainage[ig, ib, :]
            streamb_inflow[ig, ib, :] = transport[ig, ib, :] + flood_flow[ig, ib, :]
            ret_swamp[ig, ib, IPOCA : ISANDSED + 1] = 0.0

            if flood[ig, ib, IH2O] < 0.0:
                stream[ig, ib, :] = stream[ig, ib, :] + flood[ig, ib, :]
                flood[ig, ib, :] = 0.0
            if stream[ig, ib, IH2O] < 0.0:
                fast[ig, ib, :] = fast[ig, ib, :] + stream[ig, ib, :]
                stream[ig, ib, :] = 0.0
            if fast[ig, ib, IH2O] < 0.0:
                slow[ig, ib, :] = slow[ig, ib, :] + fast[ig, ib, :]
                fast[ig, ib, :] = 0.0
            if slow[ig, ib, IH2O] < -float(min_sechiba):
                raise ValueError("routing_flow negative slow_reservoir after reservoir correction")

    totflood = np.sum(flood, axis=1)
    return RoutingReservoirUpdateStep(
        fast_reservoir=fast,
        slow_reservoir=slow,
        stream_reservoir=stream,
        flood_reservoir=flood,
        pond_reservoir=pond,
        streamb_inflow=streamb_inflow,
        return_swamp=ret_swamp,
        totflood=totflood,
    )


def routing_area_fractions_step(
    *,
    stream_reservoir,
    routing_area,
    totflood,
    flood_reservoir,
    pond_reservoir,
    stream_seddep,
    vegtot,
    bulkdens,
    stream_area,
    headw_area,
    streamr10th,
    streamr90th,
    floodplains,
    floodh90th,
    dostreamswell: bool,
    do_floodplains: bool,
    doponds: bool,
    limit_rivdepos: bool,
    beta=2.0,
    betap=0.5,
    pondcri=2000.0,
    maxdep_rivsed=10.0,
    min_sechiba=1.0e-8,
) -> RoutingAreaFractionsStep:
    """Compute routing stream/flood/pond fractions and optional river-bed limit.

    Fortran provenance: ``routing.f90::routing_flow`` lines 4374-4500. The
    source partitions swollen stream area among basins by square-root stream
    storage, optionally limits accumulated river-bed sediment, computes
    `streamfl_frac` consistency fields, then computes `flood_frac`,
    `flood_frac_bas`, `flood_height`, and pond contribution to `flood_frac`.

    Note: lines 4438-4442 write both sediment and POC transfer terms into
    `rivbed2fld_sed` index ranges in the source. This helper preserves that
    source behavior and leaves `rivbed2fld_poc` zero for this local boundary.
    """

    stream = np.asarray(stream_reservoir, dtype=np.float64)
    area = np.asarray(routing_area, dtype=np.float64)
    totflood = np.asarray(totflood, dtype=np.float64)
    flood_res = np.asarray(flood_reservoir, dtype=np.float64)
    pond = np.asarray(pond_reservoir, dtype=np.float64)
    seddep = np.asarray(stream_seddep, dtype=np.float64).copy()
    vegtot = np.asarray(vegtot, dtype=np.float64)
    bulkdens = np.asarray(bulkdens, dtype=np.float64)
    stream_area = np.asarray(stream_area, dtype=np.float64)
    headw_area = np.asarray(headw_area, dtype=np.float64)
    streamr10th = np.asarray(streamr10th, dtype=np.float64)
    streamr90th = np.asarray(streamr90th, dtype=np.float64)
    floodplains = np.asarray(floodplains, dtype=np.float64)
    floodh90th = np.asarray(floodh90th, dtype=np.float64)

    if stream.ndim != 3:
        raise ValueError("stream_reservoir must have shape (npts,nbas,nflow)")
    npts, nbas, nflow = stream.shape
    if area.shape != (npts, nbas):
        raise ValueError("routing_area must have shape (npts,nbas)")
    for name, arr in (
        ("flood_reservoir", flood_res),
        ("stream_seddep", seddep),
    ):
        if arr.shape != (npts, nbas, nflow):
            raise ValueError(f"{name} must have shape (npts,nbas,nflow)")
    if totflood.shape != (npts, nflow) or pond.shape != (npts, nflow):
        raise ValueError("totflood and pond_reservoir must have shape (npts,nflow)")
    for name, arr in (
        ("vegtot", vegtot),
        ("bulkdens", bulkdens),
        ("stream_area", stream_area),
        ("headw_area", headw_area),
        ("streamr10th", streamr10th),
        ("streamr90th", streamr90th),
        ("floodplains", floodplains),
        ("floodh90th", floodh90th),
    ):
        if arr.shape != (npts,):
            raise ValueError(f"{name} must have shape (npts,)")
    if nflow <= ISANDSED:
        raise ValueError("routing area fractions require nflow >= 10")
    total_area = np.sum(area, axis=1)
    if np.any(total_area <= 0.0):
        raise ValueError("routing area fractions require positive total routing area")

    stream_area_act = np.zeros((npts,), dtype=np.float64)
    stream_area_bas = np.zeros((npts, nbas), dtype=np.float64)
    streamfl_frac = np.zeros((npts,), dtype=np.float64)
    streamfl_frac_bas = np.zeros((npts, nbas), dtype=np.float64)
    rivbed2fld_sed = np.zeros((npts, 3), dtype=np.float64)
    rivbed2fld_poc = np.zeros((npts, 3), dtype=np.float64)
    flood_frac = np.zeros((npts,), dtype=np.float64)
    flood_frac_bas = np.zeros((npts, nbas), dtype=np.float64)
    flood_height = np.zeros((npts,), dtype=np.float64)
    pond_frac = np.zeros((npts,), dtype=np.float64)
    sqrt_stream = np.sqrt(np.maximum(stream[:, :, IH2O], 0.0))
    tot_upstream2 = np.sum(sqrt_stream, axis=1)

    for ig in range(npts):
        stream_sum = np.sum(stream[ig, :, IH2O])
        if bool(dostreamswell):
            if stream_sum > 0.0:
                if stream_sum <= streamr10th[ig]:
                    stream_area_act[ig] = stream_area[ig] + headw_area[ig]
                elif stream_sum >= streamr90th[ig]:
                    stream_area_act[ig] = stream_area[ig] * 1.1 + headw_area[ig] * 1.2
                elif streamr90th[ig] - streamr10th[ig] > 0.0:
                    ratio = (stream_sum - streamr10th[ig]) / (streamr90th[ig] - streamr10th[ig])
                    stream_area_act[ig] = stream_area[ig] * (1.0 + 0.1 * ratio) + headw_area[ig] * (1.0 + 0.2 * ratio)
                else:
                    stream_area_act[ig] = 0.0
            else:
                stream_area_act[ig] = 0.0
        else:
            stream_area_act[ig] = stream_area[ig] + headw_area[ig]
        streamfl_frac[ig] = max((stream_area_act[ig] - stream_area[ig] - headw_area[ig]) / total_area[ig], 0.0)

        for ib in range(nbas):
            if tot_upstream2[ig] > 0.0 and area[ig, ib] > 0.0:
                stream_area_bas[ig, ib] = stream_area_act[ig] * sqrt_stream[ig, ib] / tot_upstream2[ig]
                swollen_area_bas = (stream_area_act[ig] - stream_area[ig] - headw_area[ig]) * sqrt_stream[ig, ib] / tot_upstream2[ig]
                streamfl_frac_bas[ig, ib] = max(swollen_area_bas / area[ig, ib], 0.0)

            if bool(limit_rivdepos):
                sed_sum = np.sum(seddep[ig, ib, ICLAYSED : ISANDSED + 1])
                if vegtot[ig] * stream_area_bas[ig, ib] > min_sechiba and sed_sum > min_sechiba:
                    riv_seddep = sed_sum / bulkdens[ig] / (vegtot[ig] * stream_area_bas[ig, ib])
                    if riv_seddep > maxdep_rivsed and total_area[ig] > min_sechiba:
                        excess = (riv_seddep - maxdep_rivsed) / riv_seddep
                        rivbed2fld_sed[ig, IC_CLAY : IC_SAND + 1] = (
                            1.0e3
                            * seddep[ig, ib, ICLAYSED : ISANDSED + 1]
                            / (vegtot[ig] * total_area[ig])
                            * excess
                        )
                        rivbed2fld_sed[ig, IACTIVE : IPASSIVE + 1] = (
                            1.0e3 * seddep[ig, ib, IPOCA : IPOCP + 1] / (vegtot[ig] * total_area[ig]) * excess
                        )
                        seddep[ig, ib, IPOCA : ISANDSED + 1] = maxdep_rivsed / riv_seddep * seddep[ig, ib, IPOCA : ISANDSED + 1]

        if abs(streamfl_frac[ig] - np.sum(streamfl_frac_bas[ig, :] * area[ig, :]) / total_area[ig]) > min_sechiba:
            raise ValueError("routing_flow streamfl_frac basin distribution mismatch")

    if bool(do_floodplains) or bool(doponds):
        for ig in range(npts):
            if totflood[ig, IH2O] > min_sechiba and floodh90th[ig] > min_sechiba:
                flood_frac_pot = (totflood[ig, IH2O] / (total_area[ig] * floodh90th[ig] / (float(beta) + 1.0))) ** (
                    float(beta) / (float(beta) + 1.0)
                )
                flood_frac[ig] = min(floodplains[ig] / total_area[ig], flood_frac_pot)
                for ib in range(nbas):
                    if area[ig, ib] > min_sechiba:
                        flood_frac_bas[ig, ib] = flood_frac[ig] * (
                            flood_res[ig, ib, IH2O] / totflood[ig, IH2O]
                        ) / (area[ig, ib] / total_area[ig])
                flood_height[ig] = (
                    (float(beta) / (float(beta) + 1.0)) * floodh90th[ig] * flood_frac[ig] ** (1.0 / float(beta))
                    + totflood[ig, IH2O] / (total_area[ig] * flood_frac[ig])
                )
                pond_frac[ig] = min(
                    1.0 - flood_frac[ig],
                    ((float(betap) + 1.0) * pond[ig, IH2O] / (float(pondcri) * total_area[ig]))
                    ** (float(betap) / (float(betap) + 1.0)),
                )
                flood_frac[ig] = min(flood_frac[ig] + pond_frac[ig], 1.0 - streamfl_frac[ig])

    return RoutingAreaFractionsStep(
        stream_area_act=stream_area_act,
        stream_area_bas=stream_area_bas,
        streamfl_frac=streamfl_frac,
        streamfl_frac_bas=streamfl_frac_bas,
        stream_seddep=seddep,
        rivbed2fld_sed=rivbed2fld_sed,
        rivbed2fld_poc=rivbed2fld_poc,
        flood_frac=flood_frac,
        flood_frac_bas=flood_frac_bas,
        flood_height=flood_height,
        pond_frac=pond_frac,
    )


def routing_pond_flux_step(
    *,
    fast_flow,
    pond_reservoir,
    routing_area,
    pond_frac,
    k_litt,
    reinf_slope,
    flood_dep_sed,
    flood_dep_poc,
    doponds: bool,
    dt_routing,
    one_day=86400.0,
    min_sechiba=1.0e-8,
) -> RoutingPondFluxStep:
    """Compute pond drainage, pond inflow, and trapped POC/sediment deposition.

    Fortran provenance: ``routing.f90::routing_flow`` lines 3924-3993. With
    `doponds`, water drainage is limited by the pond store share and
    `pond_frac * routing_area * k_litt * dt_routing/one_day`; DOC/CO2 drainage
    follows pond concentration; a `reinf_slope` share of every `fast_flow`
    species becomes pond inflow; POC/sediment pond inflow is immediately added
    to floodplain deposition and zeroed in `pond_inflow`. Without `doponds`,
    pond inflow/drainage and the pond reservoir are reset to zero.
    """

    fast = np.asarray(fast_flow, dtype=np.float64).copy()
    pond = np.asarray(pond_reservoir, dtype=np.float64).copy()
    area = np.asarray(routing_area, dtype=np.float64)
    pond_frac = np.asarray(pond_frac, dtype=np.float64)
    k_litt = np.asarray(k_litt, dtype=np.float64)
    reinf_slope = np.asarray(reinf_slope, dtype=np.float64)
    dep_sed = np.asarray(flood_dep_sed, dtype=np.float64).copy()
    dep_poc = np.asarray(flood_dep_poc, dtype=np.float64).copy()

    if fast.ndim != 3:
        raise ValueError("fast_flow must have shape (npts,nbas,nflow)")
    npts, nbas, nflow = fast.shape
    if area.shape != (npts, nbas):
        raise ValueError("routing_area must have shape (npts,nbas)")
    if pond.shape != (npts, nflow):
        raise ValueError("pond_reservoir must have shape (npts,nflow)")
    for name, arr in (("pond_frac", pond_frac), ("k_litt", k_litt), ("reinf_slope", reinf_slope)):
        if arr.shape != (npts,):
            raise ValueError(f"{name} must have shape (npts,)")
    if dep_sed.shape != (npts, nbas, 3) or dep_poc.shape != (npts, nbas, 3):
        raise ValueError("flood_dep_sed and flood_dep_poc must have shape (npts,nbas,3)")
    if nflow <= ISANDSED:
        raise ValueError("routing pond flux requires nflow >= 10")
    total_area = np.sum(area, axis=1)
    if np.any(total_area <= 0.0):
        raise ValueError("routing pond flux requires positive total routing area")

    pond_inflow = np.zeros_like(fast)
    pond_drainage = np.zeros_like(fast)
    if bool(doponds):
        for ig in range(npts):
            for ib in range(nbas):
                pond_drainage[ig, ib, IH2O] = min(
                    pond[ig, IH2O] * area[ig, ib] / total_area[ig],
                    pond_frac[ig] * area[ig, ib] * k_litt[ig] * float(dt_routing) / float(one_day),
                )
                for iflow in range(IH2O + 1, min(ICO2AQ + 1, nflow)):
                    if pond[ig, IH2O] > min_sechiba:
                        pond_drainage[ig, ib, iflow] = pond_drainage[ig, ib, IH2O] * pond[ig, iflow] / pond[ig, IH2O]
                    else:
                        pond_drainage[ig, ib, iflow] = pond[ig, iflow]
                pond_inflow[ig, ib, :] = fast[ig, ib, :] * reinf_slope[ig]
                fast[ig, ib, :] = fast[ig, ib, :] - pond_inflow[ig, ib, :]

        dep_poc[:, :, IACTIVE] += pond_inflow[:, :, IPOCA]
        dep_poc[:, :, ISLOW] += pond_inflow[:, :, IPOCS]
        dep_poc[:, :, IPASSIVE] += pond_inflow[:, :, IPOCP]
        dep_sed[:, :, IC_CLAY] += pond_inflow[:, :, ICLAYSED]
        dep_sed[:, :, IC_SILT] += pond_inflow[:, :, ISILTSED]
        dep_sed[:, :, IC_SAND] += pond_inflow[:, :, ISANDSED]
        pond_inflow[:, :, IPOCA : ISANDSED + 1] = 0.0
    else:
        pond = np.zeros_like(pond)

    return RoutingPondFluxStep(
        fast_flow=fast,
        pond_reservoir=pond,
        pond_inflow=pond_inflow,
        pond_drainage=pond_drainage,
        flood_dep_sed=dep_sed,
        flood_dep_poc=dep_poc,
    )


def routing_swamp_flood_step(
    *,
    stream_reservoir,
    transport,
    routing_area,
    route_tobasin,
    topo_resid,
    streamr50th,
    floodtemp,
    swamp,
    floodplains,
    stream_area,
    flood_reservoir,
    flood_dep_sed,
    flood_dep_poc,
    doswamps: bool,
    do_floodplains: bool,
    new_flood_scheme: bool,
    dt_routing,
    swamp_cst=0.2,
    stream_tcst=0.24,
    one_day=86400.0,
    tp_00=273.15,
    min_sechiba=1.0e-8,
) -> RoutingSwampFloodStep:
    """Compute stream 50th-flow, swamp returnflow, and floodplain inflow.

    Fortran provenance: ``routing.f90::routing_flow`` lines 4074-4206. The
    source partitions `streamr50th` by stream water storage, computes routed
    50th-percentile stream flow, optionally withdraws excess transport to
    swamps while depositing POC/sediment, and computes new-flood-scheme
    floodplain inflows. With `new_flood_scheme` false under `do_floodplains`,
    `floods` and `flood_reservoir` are reset to zero as in the source.
    """

    stream = np.asarray(stream_reservoir, dtype=np.float64)
    transport = np.asarray(transport, dtype=np.float64)
    area = np.asarray(routing_area, dtype=np.float64)
    route = np.asarray(route_tobasin)
    topo = np.asarray(topo_resid, dtype=np.float64)
    streamr50th = np.asarray(streamr50th, dtype=np.float64)
    floodtemp = np.asarray(floodtemp, dtype=np.float64)
    swamp = np.asarray(swamp, dtype=np.float64)
    floodplains = np.asarray(floodplains, dtype=np.float64)
    stream_area = np.asarray(stream_area, dtype=np.float64)
    flood_res = np.asarray(flood_reservoir, dtype=np.float64).copy()
    dep_sed = np.asarray(flood_dep_sed, dtype=np.float64).copy()
    dep_poc = np.asarray(flood_dep_poc, dtype=np.float64).copy()

    if stream.ndim != 3:
        raise ValueError("stream_reservoir must have shape (npts,nbas,nflow)")
    npts, nbas, nflow = stream.shape
    if transport.shape[0] != npts or transport.shape[1] < nbas or transport.shape[2] != nflow:
        raise ValueError("transport must have shape (npts,nbas-or-more,nflow)")
    for name, arr in (("routing_area", area), ("topo_resid", topo)):
        if arr.shape != (npts, nbas):
            raise ValueError(f"{name} must have shape (npts,nbas)")
    if route.shape != (npts, nbas):
        raise ValueError("route_tobasin must have shape (npts,nbas)")
    for name, arr in (
        ("streamr50th", streamr50th),
        ("floodtemp", floodtemp),
        ("swamp", swamp),
        ("floodplains", floodplains),
        ("stream_area", stream_area),
    ):
        if arr.shape != (npts,):
            raise ValueError(f"{name} must have shape (npts,)")
    if flood_res.shape != (npts, nbas, nflow):
        raise ValueError("flood_reservoir must have shape (npts,nbas,nflow)")
    if dep_sed.shape != (npts, nbas, 3) or dep_poc.shape != (npts, nbas, 3):
        raise ValueError("flood_dep_sed and flood_dep_poc must have shape (npts,nbas,3)")
    if nflow <= ISANDSED:
        raise ValueError("routing swamp/flood step requires nflow >= 10")

    streamr50th_bas = np.zeros((npts, nbas), dtype=np.float64)
    stream_flow50th = np.zeros((npts, nbas), dtype=np.float64)
    return_swamp = np.zeros((npts, nbas, nflow), dtype=np.float64)
    floods = np.zeros((npts, nbas, nflow), dtype=np.float64)
    if bool(new_flood_scheme):
        tot_upstream = np.sum(stream[:, :, IH2O], axis=1)
        for ig in range(npts):
            for ib in range(nbas):
                if tot_upstream[ig] > min_sechiba:
                    streamr50th_bas[ig, ib] = streamr50th[ig] * stream[ig, ib, IH2O] / tot_upstream[ig]
                else:
                    streamr50th_bas[ig, ib] = streamr50th[ig]
                if route[ig, ib] > 0:
                    flow = min(
                        streamr50th_bas[ig, ib] / ((topo[ig, ib] / 1000.0) * float(stream_tcst) * float(one_day) / float(dt_routing)),
                        streamr50th_bas[ig, ib] - min_sechiba,
                    )
                    stream_flow50th[ig, ib] = max(flow, 0.0)
                else:
                    stream_flow50th[ig, ib] = 0.0

        if bool(doswamps):
            tobeflooded = swamp.copy()
            for ib in range(nbas):
                for ig in range(npts):
                    if transport[ig, ib, IH2O] > stream_flow50th[ig, ib]:
                        potflood_water = transport[ig, ib, IH2O] - stream_flow50th[ig, ib]
                        potflood = np.zeros((nflow,), dtype=np.float64)
                        potflood[IH2O] = potflood_water
                        if transport[ig, ib, IH2O] != 0.0:
                            potflood[IH2O + 1 : nflow] = potflood_water / transport[ig, ib, IH2O] * transport[ig, ib, IH2O + 1 : nflow]
                        if tobeflooded[ig] > 0.0 and potflood_water > 0.0 and floodtemp[ig] > tp_00:
                            if area[ig, ib] > tobeflooded[ig]:
                                floodindex = tobeflooded[ig] / area[ig, ib]
                            else:
                                floodindex = 1.0
                            return_swamp[ig, ib, :] = float(swamp_cst) * potflood * floodindex
                            tobeflooded[ig] = tobeflooded[ig] - area[ig, ib]
                            dep_poc[ig, ib, IACTIVE] += return_swamp[ig, ib, IPOCA]
                            dep_poc[ig, ib, ISLOW] += return_swamp[ig, ib, IPOCS]
                            dep_poc[ig, ib, IPASSIVE] += return_swamp[ig, ib, IPOCP]
                            dep_sed[ig, ib, IC_CLAY] += return_swamp[ig, ib, ICLAYSED]
                            dep_sed[ig, ib, IC_SILT] += return_swamp[ig, ib, ISILTSED]
                            dep_sed[ig, ib, IC_SAND] += return_swamp[ig, ib, ISANDSED]
                            return_swamp[ig, ib, IPOCA : ISANDSED + 1] = 0.0

        if bool(do_floodplains):
            for ig in range(npts):
                for ib in range(nbas):
                    if floodplains[ig] > min_sechiba and stream_flow50th[ig, ib] > min_sechiba:
                        frac_floodplain = max(floodplains[ig] - stream_area[ig], 0.0) / floodplains[ig]
                        floods[ig, ib, IH2O] = max(
                            (transport[ig, ib, IH2O] - return_swamp[ig, ib, IH2O] - stream_flow50th[ig, ib])
                            * frac_floodplain,
                            0.0,
                        )
                        for iflow in range(IH2O + 1, nflow):
                            if transport[ig, ib, IH2O] > min_sechiba:
                                floods[ig, ib, iflow] = floods[ig, ib, IH2O] * transport[ig, ib, iflow] / transport[ig, ib, IH2O]
                            else:
                                floods[ig, ib, iflow] = 0.0
    elif bool(do_floodplains):
        floods[:, :, :] = 0.0
        flood_res[:, :, :] = 0.0

    return RoutingSwampFloodStep(
        streamr50th_bas=streamr50th_bas,
        stream_flow50th=stream_flow50th,
        return_swamp=return_swamp,
        floods=floods,
        flood_reservoir=flood_res,
        flood_dep_sed=dep_sed,
        flood_dep_poc=dep_poc,
    )


def routing_flood_pond_input_step(
    *,
    flood_reservoir,
    pond_reservoir,
    floodout,
    flood_inp,
    stream_inp,
    routing_area,
    flood_frac,
    flood_frac_bas,
    pond_frac,
    streamfl_frac,
    streamfl_frac_bas,
    do_floodplains: bool,
    doponds: bool,
    dostreamswell: bool,
    min_sechiba=1.0e-8,
    min_concensed=1.0e-20,
) -> RoutingFloodPondInputStep:
    """Apply flood/pond vertical fluxes and distribute DOC/CO2 inputs by basin.

    Fortran provenance: ``routing.f90::routing_flow`` lines 3558-3681. The
    source zeroes floodplain deposition and basin input buffers, optionally
    deposits tiny particulate floodplain stores, applies floodplain/pond net
    evaporation from ``floodout``, partitions non-water flood inputs over
    ``flood_frac_bas``, and partitions non-water stream inputs over
    ``streamfl_frac_bas`` when stream swelling is active.
    """

    flood = np.asarray(flood_reservoir, dtype=np.float64).copy()
    pond = np.asarray(pond_reservoir, dtype=np.float64).copy()
    floodout = np.asarray(floodout, dtype=np.float64)
    flood_inp = np.asarray(flood_inp, dtype=np.float64)
    stream_inp = np.asarray(stream_inp, dtype=np.float64)
    area = np.asarray(routing_area, dtype=np.float64)
    flood_frac = np.asarray(flood_frac, dtype=np.float64)
    flood_frac_bas = np.asarray(flood_frac_bas, dtype=np.float64)
    pond_frac = np.asarray(pond_frac, dtype=np.float64)
    streamfl_frac = np.asarray(streamfl_frac, dtype=np.float64)
    streamfl_frac_bas = np.asarray(streamfl_frac_bas, dtype=np.float64)

    if flood.ndim != 3:
        raise ValueError("flood_reservoir must have shape (npts,nbas,nflow)")
    npts, nbas, nflow = flood.shape
    if pond.shape != (npts, nflow):
        raise ValueError("pond_reservoir must have shape (npts,nflow)")
    if flood_inp.shape != (npts, nflow) or stream_inp.shape != (npts, nflow):
        raise ValueError("flood_inp and stream_inp must have shape (npts,nflow)")
    for name, arr in (
        ("routing_area", area),
        ("flood_frac_bas", flood_frac_bas),
        ("streamfl_frac_bas", streamfl_frac_bas),
    ):
        if arr.shape != (npts, nbas):
            raise ValueError(f"{name} must have shape (npts,nbas)")
    for name, arr in (
        ("floodout", floodout),
        ("flood_frac", flood_frac),
        ("pond_frac", pond_frac),
        ("streamfl_frac", streamfl_frac),
    ):
        if arr.shape != (npts,):
            raise ValueError(f"{name} must have shape (npts,)")
    if nflow <= ISANDSED:
        raise ValueError("routing flood/pond input requires nflow >= 10")

    flood_dep_sed = np.zeros((npts, nbas, 3), dtype=np.float64)
    flood_dep_poc = np.zeros((npts, nbas, 3), dtype=np.float64)
    flood_inp_bas = np.zeros_like(flood)
    stream_inp_bas = np.zeros_like(flood)
    total_area = np.sum(area, axis=1)
    if np.any(total_area <= 0.0):
        raise ValueError("routing flood/pond input requires positive total routing area")

    if bool(do_floodplains) or bool(doponds):
        for ig in range(npts):
            for ib in range(nbas):
                for iflow, dep_idx, dep_arr in (
                    (ICLAYSED, IC_CLAY, flood_dep_sed),
                    (ISILTSED, IC_SILT, flood_dep_sed),
                    (ISANDSED, IC_SAND, flood_dep_sed),
                    (IPOCA, IACTIVE, flood_dep_poc),
                    (IPOCS, ISLOW, flood_dep_poc),
                    (IPOCP, IPASSIVE, flood_dep_poc),
                ):
                    if flood[ig, ib, iflow] < float(min_concensed):
                        dep_arr[ig, ib, dep_idx] += flood[ig, ib, iflow]
                        flood[ig, ib, iflow] = 0.0

            if flood_frac[ig] > min_sechiba:
                flow = min(
                    floodout[ig] * total_area[ig] * pond_frac[ig] / flood_frac[ig],
                    pond[ig, IH2O] + np.sum(flood[ig, :, IH2O]),
                )
                pondex = max(flow - pond[ig, IH2O], 0.0)
                pond[ig, IH2O] = pond[ig, IH2O] - (flow - pondex)
                pond_excessflow = np.zeros((nbas, nflow), dtype=np.float64)
                denom = flood_frac[ig] - pond_frac[ig]
                for ib in range(nbas):
                    if denom != 0.0:
                        pond_excessflow[ib, IH2O] = min(
                            pondex * flood_frac_bas[ig, ib] / denom,
                            flood[ig, ib, IH2O],
                        )
                    else:
                        pond_excessflow[ib, IH2O] = 0.0
                    pondex = pondex - pond_excessflow[ib, IH2O]
                if pondex > min_sechiba:
                    raise ValueError("routing_flow unable to redistribute excess pond outflow")

                for ib in range(nbas):
                    basin_flow = (
                        floodout[ig] * area[ig, ib] * flood_frac_bas[ig, ib] / flood_frac[ig]
                        + pond_excessflow[ib, IH2O]
                    )
                    if basin_flow > 0.0 and flood[ig, ib, IH2O] > min_sechiba:
                        ratio = basin_flow / flood[ig, ib, IH2O]
                        flood_dep_sed[ig, ib, IC_CLAY] += flood[ig, ib, ICLAYSED] * ratio
                        flood_dep_sed[ig, ib, IC_SILT] += flood[ig, ib, ISILTSED] * ratio
                        flood_dep_sed[ig, ib, IC_SAND] += flood[ig, ib, ISANDSED] * ratio
                        flood_dep_poc[ig, ib, IACTIVE] += flood[ig, ib, IPOCA] * ratio
                        flood_dep_poc[ig, ib, ISLOW] += flood[ig, ib, IPOCS] * ratio
                        flood_dep_poc[ig, ib, IPASSIVE] += flood[ig, ib, IPOCP] * ratio
                        flood[ig, ib, IPOCA : ISANDSED + 1] *= 1.0 - ratio
                    elif flood[ig, ib, IH2O] < min_sechiba:
                        flood_dep_sed[ig, ib, IC_CLAY] += flood[ig, ib, ICLAYSED]
                        flood_dep_sed[ig, ib, IC_SILT] += flood[ig, ib, ISILTSED]
                        flood_dep_sed[ig, ib, IC_SAND] += flood[ig, ib, ISANDSED]
                        flood_dep_poc[ig, ib, IACTIVE] += flood[ig, ib, IPOCA]
                        flood_dep_poc[ig, ib, ISLOW] += flood[ig, ib, IPOCS]
                        flood_dep_poc[ig, ib, IPASSIVE] += flood[ig, ib, IPOCP]
                        flood[ig, ib, IPOCA : ISANDSED + 1] = 0.0
                    flood[ig, ib, IH2O] = flood[ig, ib, IH2O] - basin_flow
                    if flood[ig, ib, IH2O] < min_sechiba:
                        flood[ig, ib, IH2O] = 0.0

                if pond[ig, IH2O] < min_sechiba:
                    pond[ig, IH2O] = 0.0

                for iflow in range(IH2O + 1, nflow):
                    dissolved_flow = -flood_inp[ig, iflow] * total_area[ig] * pond_frac[ig] / flood_frac[ig]
                    pond[ig, iflow] = pond[ig, iflow] - dissolved_flow
                    for ib in range(nbas):
                        flood_inp_bas[ig, ib, iflow] = (
                            flood_inp[ig, iflow] * area[ig, ib] * flood_frac_bas[ig, ib] / flood_frac[ig]
                        )

    if bool(dostreamswell):
        for ig in range(npts):
            if streamfl_frac[ig] > min_sechiba:
                for iflow in range(IH2O + 1, nflow):
                    for ib in range(nbas):
                        stream_inp_bas[ig, ib, iflow] = (
                            stream_inp[ig, iflow] * area[ig, ib] * streamfl_frac_bas[ig, ib] / streamfl_frac[ig]
                        )

    return RoutingFloodPondInputStep(
        flood_reservoir=flood,
        pond_reservoir=pond,
        flood_dep_sed=flood_dep_sed,
        flood_dep_poc=flood_dep_poc,
        flood_inp_bas=flood_inp_bas,
        stream_inp_bas=stream_inp_bas,
    )


def routing_floodplain_flux_step(
    *,
    flood_reservoir,
    stream_reservoir,
    routing_area,
    flood_frac,
    flood_frac_bas,
    k_litt,
    route_tobasin,
    topo_resid,
    flood_dep_sed,
    flood_dep_poc,
    do_floodplains: bool,
    dofloodinfilt: bool,
    dt_routing,
    flood_tcst=4.0,
    flow_coef=1.01,
    coef_seddep_flood=(0.5, 0.8, 1.0),
    one_day=86400.0,
    min_sechiba=1.0e-8,
) -> RoutingFloodplainFluxStep:
    """Compute floodplain drainage, lateral outflow, and particulate deposition.

    Fortran provenance: ``routing.f90::routing_flow`` lines 3686-3915. The
    source optionally computes floodplain drainage from litter conductivity,
    transports DOC/CO2 with drainage by concentration, deposits/removes
    POC/sediment with drainage, computes lateral floodplain outflow through
    `flood_tcst`, deposits floodplain POC/sediment, limits remaining particulate
    concentration to 10% of floodplain water, zeroes unsupported inactive-route
    particulate stores as written in the source, and adds `flood_flow` back to
    `stream_reservoir`.
    """

    flood = np.asarray(flood_reservoir, dtype=np.float64).copy()
    stream = np.asarray(stream_reservoir, dtype=np.float64).copy()
    area = np.asarray(routing_area, dtype=np.float64)
    flood_frac = np.asarray(flood_frac, dtype=np.float64)
    flood_frac_bas = np.asarray(flood_frac_bas, dtype=np.float64)
    k_litt = np.asarray(k_litt, dtype=np.float64)
    route = np.asarray(route_tobasin)
    topo = np.asarray(topo_resid, dtype=np.float64)
    dep_sed = np.asarray(flood_dep_sed, dtype=np.float64).copy()
    dep_poc = np.asarray(flood_dep_poc, dtype=np.float64).copy()
    coef = np.asarray(coef_seddep_flood, dtype=np.float64)

    if flood.ndim != 3 or stream.shape != flood.shape:
        raise ValueError("flood_reservoir and stream_reservoir must share shape (npts,nbas,nflow)")
    npts, nbas, nflow = flood.shape
    for name, arr in (("routing_area", area), ("flood_frac_bas", flood_frac_bas), ("topo_resid", topo)):
        if arr.shape != (npts, nbas):
            raise ValueError(f"{name} must have shape (npts,nbas)")
    if route.shape != (npts, nbas):
        raise ValueError("route_tobasin must have shape (npts,nbas)")
    for name, arr in (("flood_frac", flood_frac), ("k_litt", k_litt)):
        if arr.shape != (npts,):
            raise ValueError(f"{name} must have shape (npts,)")
    if dep_sed.shape != (npts, nbas, 3) or dep_poc.shape != (npts, nbas, 3):
        raise ValueError("flood_dep_sed and flood_dep_poc must have shape (npts,nbas,3)")
    if coef.shape != (3,):
        raise ValueError("coef_seddep_flood must have three textural coefficients")
    if nflow <= ISANDSED:
        raise ValueError("routing floodplain flux requires nflow >= 10")

    flood_drainage = np.zeros_like(flood)
    flood_flow = np.zeros_like(flood)
    if bool(do_floodplains):
        if bool(dofloodinfilt):
            for ig in range(npts):
                for ib in range(nbas):
                    flood_drainage[ig, ib, IH2O] = max(
                        0.0,
                        min(
                            flood[ig, ib, IH2O],
                            flood_frac[ig] * area[ig, ib] * k_litt[ig] * float(dt_routing) / float(one_day),
                        ),
                    )
                    if flood[ig, ib, IH2O] > min_sechiba:
                        ratio = flood_drainage[ig, ib, IH2O] / flood[ig, ib, IH2O]
                        for iflow in range(IH2O + 1, min(ICO2AQ + 1, nflow)):
                            flood_drainage[ig, ib, iflow] = flood[ig, ib, iflow] * ratio
                        dep_sed[ig, ib, IC_CLAY] += flood[ig, ib, ICLAYSED] * ratio
                        dep_sed[ig, ib, IC_SILT] += flood[ig, ib, ISILTSED] * ratio
                        dep_sed[ig, ib, IC_SAND] += flood[ig, ib, ISANDSED] * ratio
                        dep_poc[ig, ib, IACTIVE] += flood[ig, ib, IPOCA] * ratio
                        dep_poc[ig, ib, ISLOW] += flood[ig, ib, IPOCS] * ratio
                        dep_poc[ig, ib, IPASSIVE] += flood[ig, ib, IPOCP] * ratio
                        flood[ig, ib, IPOCA : ISANDSED + 1] *= 1.0 - ratio
                    else:
                        for iflow in range(IH2O + 1, min(ICO2AQ + 1, nflow)):
                            flood_drainage[ig, ib, iflow] = flood[ig, ib, iflow]
                        dep_sed[ig, ib, IC_CLAY] += flood[ig, ib, ICLAYSED]
                        dep_sed[ig, ib, IC_SILT] += flood[ig, ib, ISILTSED]
                        dep_sed[ig, ib, IC_SAND] += flood[ig, ib, ISANDSED]
                        dep_poc[ig, ib, IACTIVE] += flood[ig, ib, IPOCA]
                        dep_poc[ig, ib, ISLOW] += flood[ig, ib, IPOCS]
                        dep_poc[ig, ib, IPASSIVE] += flood[ig, ib, IPOCP]
                        flood[ig, ib, IPOCA : ISANDSED + 1] = 0.0
                    flood[ig, ib, IH2O : ICO2AQ + 1] -= flood_drainage[ig, ib, IH2O : ICO2AQ + 1]

        for ig in range(npts):
            for ib in range(nbas):
                for iflow in range(IH2O, min(ICO2AQ + 1, nflow)):
                    if route[ig, ib] > 0:
                        if flood_frac_bas[ig, ib] > min_sechiba:
                            flow = min(
                                (flood[ig, ib, iflow] ** float(flow_coef))
                                / (
                                    (topo[ig, ib] / 1000.0)
                                    * float(flood_tcst)
                                    * flood_frac_bas[ig, ib]
                                    * float(one_day)
                                    / float(dt_routing)
                                ),
                                flood[ig, ib, iflow],
                            )
                        else:
                            flow = 0.0
                        flood_flow[ig, ib, iflow] = flow
                    else:
                        flood_flow[ig, ib, iflow] = 0.0
                    if iflow > IH2O:
                        flood[ig, ib, iflow] -= flood_flow[ig, ib, iflow]

                if route[ig, ib] > 0:
                    if flood[ig, ib, IH2O] > min_sechiba:
                        dep_sed[ig, ib, IC_CLAY] += coef[IC_CLAY] * flood[ig, ib, ICLAYSED]
                        dep_sed[ig, ib, IC_SILT] += coef[IC_SILT] * flood[ig, ib, ISILTSED]
                        dep_sed[ig, ib, IC_SAND] += coef[IC_SAND] * flood[ig, ib, ISANDSED]
                        dep_poc[ig, ib, IACTIVE] += coef[IC_CLAY] * flood[ig, ib, IPOCA]
                        dep_poc[ig, ib, ISLOW] += coef[IC_CLAY] * flood[ig, ib, IPOCS]
                        dep_poc[ig, ib, IPASSIVE] += coef[IC_CLAY] * flood[ig, ib, IPOCP]
                        flood[ig, ib, ICLAYSED] *= 1.0 - coef[IC_CLAY]
                        flood[ig, ib, ISILTSED] *= 1.0 - coef[IC_SILT]
                        flood[ig, ib, ISANDSED] *= 1.0 - coef[IC_SAND]
                        flood[ig, ib, IPOCA : IPOCP + 1] *= 1.0 - coef[IC_CLAY]
                        flood_flow[ig, ib, IPOCA : ISANDSED + 1] = np.minimum(
                            flood[ig, ib, IPOCA : ISANDSED + 1] * flood_flow[ig, ib, IH2O] / flood[ig, ib, IH2O],
                            flood[ig, ib, IPOCA : ISANDSED + 1],
                        )
                        flood[ig, ib, IPOCA : ISANDSED + 1] -= flood_flow[ig, ib, IPOCA : ISANDSED + 1]
                    else:
                        flood_flow[ig, ib, IPOCA : ISANDSED + 1] = 0.0
                        dep_sed[ig, ib, IC_CLAY] += flood[ig, ib, ICLAYSED]
                        dep_sed[ig, ib, IC_SILT] += flood[ig, ib, ISILTSED]
                        dep_sed[ig, ib, IC_SAND] += flood[ig, ib, ISANDSED]
                        dep_poc[ig, ib, IACTIVE] += flood[ig, ib, IPOCA]
                        dep_poc[ig, ib, ISLOW] += flood[ig, ib, IPOCS]
                        dep_poc[ig, ib, IPASSIVE] += flood[ig, ib, IPOCP]
                        flood[ig, ib, IPOCA : ISANDSED + 1] = 0.0
                else:
                    flood_flow[ig, ib, IPOCA : ISANDSED + 1] = 0.0
                    flood[ig, ib, IPOCA : ISANDSED + 1] = 0.0

                flood[ig, ib, IH2O] -= flood_flow[ig, ib, IH2O]
                if flood[ig, ib, IH2O] <= 0.0:
                    dep_sed[ig, ib, IC_CLAY] += flood[ig, ib, ICLAYSED]
                    dep_sed[ig, ib, IC_SILT] += flood[ig, ib, ISILTSED]
                    dep_sed[ig, ib, IC_SAND] += flood[ig, ib, ISANDSED]
                    dep_poc[ig, ib, IACTIVE] += flood[ig, ib, IPOCA]
                    dep_poc[ig, ib, ISLOW] += flood[ig, ib, IPOCS]
                    dep_poc[ig, ib, IPASSIVE] += flood[ig, ib, IPOCP]
                    flood[ig, ib, IPOCA : ISANDSED + 1] = 0.0
                else:
                    cap = 0.1 * flood[ig, ib, IH2O]
                    for iflow, dep_idx, dep_arr in (
                        (ICLAYSED, IC_CLAY, dep_sed),
                        (ISILTSED, IC_SILT, dep_sed),
                        (ISANDSED, IC_SAND, dep_sed),
                        (IPOCA, IACTIVE, dep_poc),
                        (IPOCS, ISLOW, dep_poc),
                        (IPOCP, IPASSIVE, dep_poc),
                    ):
                        if flood[ig, ib, iflow] > cap:
                            dep_arr[ig, ib, dep_idx] += flood[ig, ib, iflow] - cap
                            flood[ig, ib, iflow] = cap
    else:
        flood_drainage[:, :, :] = 0.0
        flood_flow[:, :, :] = 0.0
        flood[:, :, :] = 0.0

    stream = stream + flood_flow
    return RoutingFloodplainFluxStep(
        flood_reservoir=flood,
        stream_reservoir=stream,
        flood_drainage=flood_drainage,
        flood_flow=flood_flow,
        flood_dep_sed=dep_sed,
        flood_dep_poc=dep_poc,
    )


def routing_transport_between_basins_step(
    *,
    fast_flow,
    slow_flow,
    stream_flow,
    route_togrid,
    route_tobasin,
    transport=None,
    check_riverbal: bool = False,
    water_balance=None,
    carbon_balance=None,
) -> RoutingTransportStep:
    """Route fast/slow/stream outflows to downstream grid/basin transport slots.

    Fortran provenance: ``routing.f90::routing_flow`` lines 4003-4061. The
    source gathers flows globally, adds `fast_flow + slow_flow + stream_flow`
    to `transport_glo(route_togrid, route_tobasin, :)`, scatters back, and
    optionally updates water/carbon balance diagnostics from local transport
    minus local outflows. This helper applies the same indexing to explicit
    arrays; route indices are zero-based Python equivalents of the Fortran
    routing targets.
    """

    fast = np.asarray(fast_flow, dtype=np.float64)
    slow = np.asarray(slow_flow, dtype=np.float64)
    stream = np.asarray(stream_flow, dtype=np.float64)
    rtg = np.asarray(route_togrid)
    rtb = np.asarray(route_tobasin)
    if fast.ndim != 3 or slow.shape != fast.shape or stream.shape != fast.shape:
        raise ValueError("fast_flow, slow_flow, and stream_flow must share shape (npts,nbas,nflow)")
    npts, nbas, nflow = fast.shape
    if rtg.shape != (npts, nbas) or rtb.shape != (npts, nbas):
        raise ValueError("route_togrid and route_tobasin must have shape (npts,nbas)")
    if transport is None:
        ntransport_basins = max(nbas, int(np.max(rtb)) + 1 if rtb.size else nbas)
        out = np.zeros((npts, ntransport_basins, nflow), dtype=np.float64)
    else:
        out = np.asarray(transport, dtype=np.float64).copy()
        if out.ndim != 3 or out.shape[2] != nflow:
            raise ValueError("transport must have shape (ntransport_pts,ntransport_basins,nflow)")
    if np.any(rtg < 0) or np.any(rtg >= out.shape[0]) or np.any(rtb < 0) or np.any(rtb >= out.shape[1]):
        raise ValueError("route_togrid/route_tobasin contain targets outside transport shape")

    for ig in range(npts):
        for ib in range(nbas):
            out[int(rtg[ig, ib]), int(rtb[ig, ib]), :] += fast[ig, ib, :] + slow[ig, ib, :] + stream[ig, ib, :]

    if check_riverbal:
        if out.shape[0] != npts:
            raise ValueError("check_riverbal requires local transport first axis to match flow npts")
        if water_balance is None:
            wb = np.zeros((npts,), dtype=np.float64)
        else:
            wb = np.asarray(water_balance, dtype=np.float64).copy()
            if wb.shape != (npts,):
                raise ValueError("water_balance must have shape (npts,)")
        if carbon_balance is None:
            cb = np.zeros((npts,), dtype=np.float64)
        else:
            cb = np.asarray(carbon_balance, dtype=np.float64).copy()
            if cb.shape != (npts,):
                raise ValueError("carbon_balance must have shape (npts,)")
        wb += np.sum(out[:, :, IH2O], axis=1) - np.sum(fast[:, :, IH2O] + slow[:, :, IH2O] + stream[:, :, IH2O], axis=1)
        carbon_stop = min(IPOCP + 1, nflow)
        if carbon_stop > IH2O + 1:
            cb += np.sum(out[:, :, IH2O + 1 : carbon_stop], axis=(1, 2)) - np.sum(
                fast[:, :, IH2O + 1 : carbon_stop]
                + slow[:, :, IH2O + 1 : carbon_stop]
                + stream[:, :, IH2O + 1 : carbon_stop],
                axis=(1, 2),
            )
    else:
        wb = np.zeros((npts,), dtype=np.float64) if water_balance is None else np.asarray(water_balance, dtype=np.float64).copy()
        cb = np.zeros((npts,), dtype=np.float64) if carbon_balance is None else np.asarray(carbon_balance, dtype=np.float64).copy()

    return RoutingTransportStep(transport=out, water_balance=wb, carbon_balance=cb)


def _routing_nlybiotur(zz_deep: np.ndarray, bioturbation_depth: float) -> int:
    nlybiotur: int | None = None
    ndeep = zz_deep.shape[0]
    for ig_fortran in range(2, ndeep + 1):
        if zz_deep[ig_fortran - 2] < bioturbation_depth and zz_deep[ig_fortran - 1] >= bioturbation_depth:
            nlybiotur = min(max(ig_fortran, 2), ndeep - 1)
    if nlybiotur is None:
        raise ValueError("zz_deep must bracket bioturbation_depth as in routing_flow lines 3202-3206")
    return nlybiotur


def routing_stream_erosion_step(
    *,
    stream_reservoir,
    stream_flow,
    stream_seddep,
    qflow_avebas,
    basdrainarea,
    basgravel,
    stream_damavailbas,
    routing_area,
    bulkdens,
    zz_deep,
    veget_max,
    carbon_32l,
    dt_routing,
    coef_stc=(1.2e-5, 5.0e-6, 2.5e-6),
    coef_seddep=(0.1, 0.2, 0.5),
    coef_redet=(0.5, 0.5, 0.5),
    coef_chanero=(1.0e-3, 1.0e-3, 1.0e-3),
    stream_power=0.30,
    basarea_power=0.50,
    season_power=1.5,
    maxfrac_bankero=1.0e-14,
    bioturbation_depth=2.0,
    min_sechiba=1.0e-8,
) -> RoutingStreamErosionStep:
    """Apply stream sediment, POC, and bank-erosion updates.

    Fortran provenance: ``routing.f90::routing_flow`` lines 3390-3539, plus
    the local ``nlybiotur`` setup at lines 3202-3206. The source computes
    Method-3 WBMsed capacity, deposits excess suspended sediment, re-erodes
    previously deposited bed material when capacity exceeds carried sediment,
    erodes channel/bank material when bed deposits are insufficient, updates
    `carbon_32l` for vegetated PFTs, reduces dam capacity by daily sediment
    volume, and subtracts stream water outflow after the particulate loop.
    """

    stream = np.asarray(stream_reservoir, dtype=np.float64).copy()
    flow = np.asarray(stream_flow, dtype=np.float64).copy()
    seddep = np.asarray(stream_seddep, dtype=np.float64).copy()
    qflow = np.asarray(qflow_avebas, dtype=np.float64)
    basin_area = np.asarray(basdrainarea, dtype=np.float64)
    gravel = np.asarray(basgravel, dtype=np.float64)
    damavail = np.asarray(stream_damavailbas, dtype=np.float64).copy()
    area = np.asarray(routing_area, dtype=np.float64)
    bulkdens = np.asarray(bulkdens, dtype=np.float64)
    zz_deep = np.asarray(zz_deep, dtype=np.float64)
    veget_max = np.asarray(veget_max, dtype=np.float64)
    carbon = np.asarray(carbon_32l, dtype=np.float64).copy()
    coef_stc = np.asarray(coef_stc, dtype=np.float64)
    coef_seddep = np.asarray(coef_seddep, dtype=np.float64)
    coef_redet = np.asarray(coef_redet, dtype=np.float64)
    coef_chanero = np.asarray(coef_chanero, dtype=np.float64)

    if stream.ndim != 3 or flow.shape != stream.shape or seddep.shape != stream.shape:
        raise ValueError("stream_reservoir, stream_flow, and stream_seddep must share shape (npts,nbas,nflow)")
    npts, nbas, nflow = stream.shape
    for name, arr in (
        ("qflow_avebas", qflow),
        ("basdrainarea", basin_area),
        ("basgravel", gravel),
        ("stream_damavailbas", damavail),
        ("routing_area", area),
    ):
        if arr.shape != (npts, nbas):
            raise ValueError(f"{name} must have shape (npts,nbas)")
    if bulkdens.shape != (npts,):
        raise ValueError("bulkdens must have shape (npts,)")
    if zz_deep.ndim != 1 or zz_deep.shape[0] < 3:
        raise ValueError("zz_deep must be a one-dimensional soil-depth array")
    if veget_max.ndim != 2 or veget_max.shape[0] != npts:
        raise ValueError("veget_max must have shape (npts,nvm)")
    if carbon.ndim != 4 or carbon.shape[0] != npts or carbon.shape[1] < 3 or carbon.shape[2] != veget_max.shape[1]:
        raise ValueError("carbon_32l must have shape (npts,ncarb>=3,nvm,ndeep)")
    if carbon.shape[3] < zz_deep.shape[0]:
        raise ValueError("carbon_32l depth axis must cover zz_deep")
    if nflow <= ISANDSED:
        raise ValueError("routing stream erosion requires nflow >= 10")
    for name, arr in (
        ("coef_stc", coef_stc),
        ("coef_seddep", coef_seddep),
        ("coef_redet", coef_redet),
        ("coef_chanero", coef_chanero),
    ):
        if arr.shape != (3,):
            raise ValueError(f"{name} must have three textural coefficients")
    if np.any(area <= 0.0) or np.any(bulkdens <= 0.0):
        raise ValueError("routing_area and bulkdens must be positive")

    nlybiotur = _routing_nlybiotur(zz_deep, float(bioturbation_depth))
    bioturb_depth = zz_deep[nlybiotur - 1]
    total_area = np.sum(area, axis=1)
    sedcap = np.zeros((npts, nbas, 3), dtype=np.float64)
    erodep = np.zeros_like(stream)

    for ib in range(nbas):
        qh2o = 1.0e-3 * flow[:, ib, IH2O]
        qph2o = qh2o / float(dt_routing)
        for ig in range(npts):
            if stream[ig, ib, IH2O] <= min_sechiba:
                continue
            frac_out = min(flow[ig, ib, IH2O] / stream[ig, ib, IH2O], 1.0)
            for ie in range(3):
                sed_idx = ICLAYSED + ie
                if qflow[ig, ib] <= min_sechiba:
                    cap = 0.0
                else:
                    season_expon = float(season_power) - max(0.145 * np.log10(basin_area[ig, ib]), 0.8)
                    cap = (
                        coef_stc[ie]
                        * (qflow[ig, ib] ** float(stream_power))
                        * (basin_area[ig, ib] ** float(basarea_power))
                        * ((qph2o[ig] / qflow[ig, ib]) ** season_expon)
                        * 1.0e3
                        * float(dt_routing)
                    )
                sedcap[ig, ib, ie] = cap
                if frac_out <= min_sechiba:
                    continue

                if cap > 0.0 and frac_out * stream[ig, ib, sed_idx] >= cap:
                    erodep[ig, ib, sed_idx] = coef_seddep[ie] * (stream[ig, ib, sed_idx] - cap / frac_out)
                    if ie == IC_CLAY:
                        erodep[ig, ib, IPOCA : IPOCP + 1] = (
                            stream[ig, ib, IPOCA : IPOCP + 1]
                            * erodep[ig, ib, sed_idx]
                            / stream[ig, ib, sed_idx]
                        )
                        seddep[ig, ib, IPOCA : IPOCP + 1] += erodep[ig, ib, IPOCA : IPOCP + 1]
                        stream[ig, ib, IPOCA : IPOCP + 1] = np.maximum(
                            stream[ig, ib, IPOCA : IPOCP + 1] - erodep[ig, ib, IPOCA : IPOCP + 1],
                            0.0,
                        )
                        flow[ig, ib, IPOCA : IPOCP + 1] = frac_out * stream[ig, ib, IPOCA : IPOCP + 1]
                        stream[ig, ib, IPOCA : IPOCP + 1] = (1.0 - frac_out) * stream[ig, ib, IPOCA : IPOCP + 1]
                    seddep[ig, ib, sed_idx] += erodep[ig, ib, sed_idx]
                    stream[ig, ib, sed_idx] = max(stream[ig, ib, sed_idx] - erodep[ig, ib, sed_idx], 0.0)
                    flow[ig, ib, sed_idx] = frac_out * stream[ig, ib, sed_idx]
                    stream[ig, ib, sed_idx] = (1.0 - frac_out) * stream[ig, ib, sed_idx]
                elif cap > 0.0:
                    perosed = coef_redet[ie] * (frac_out * stream[ig, ib, sed_idx] - cap)
                    if -perosed <= frac_out * seddep[ig, ib, sed_idx] and (ie != IC_CLAY or seddep[ig, ib, sed_idx] > 0.0):
                        erodep[ig, ib, sed_idx] = perosed / frac_out
                        if ie == IC_CLAY:
                            erodep[ig, ib, IPOCA : IPOCP + 1] = (
                                seddep[ig, ib, IPOCA : IPOCP + 1]
                                * erodep[ig, ib, sed_idx]
                                / seddep[ig, ib, sed_idx]
                            )
                            seddep[ig, ib, IPOCA : IPOCP + 1] += erodep[ig, ib, IPOCA : IPOCP + 1]
                            flow[ig, ib, IPOCA : IPOCP + 1] = np.maximum(
                                frac_out * (stream[ig, ib, IPOCA : IPOCP + 1] - erodep[ig, ib, IPOCA : IPOCP + 1]),
                                0.0,
                            )
                            stream[ig, ib, IPOCA : IPOCP + 1] = np.maximum(
                                (1.0 - frac_out)
                                * (stream[ig, ib, IPOCA : IPOCP + 1] - erodep[ig, ib, IPOCA : IPOCP + 1]),
                                0.0,
                            )
                        seddep[ig, ib, sed_idx] += erodep[ig, ib, sed_idx]
                        flow[ig, ib, sed_idx] = max(frac_out * (stream[ig, ib, sed_idx] - erodep[ig, ib, sed_idx]), 0.0)
                        stream[ig, ib, sed_idx] = max(
                            (1.0 - frac_out) * (stream[ig, ib, sed_idx] - erodep[ig, ib, sed_idx]),
                            0.0,
                        )
                    elif -perosed > frac_out * seddep[ig, ib, sed_idx]:
                        shortage = perosed / frac_out + seddep[ig, ib, sed_idx]
                        erodep[ig, ib, sed_idx] = -seddep[ig, ib, sed_idx] + coef_chanero[ie] * (1.0 - gravel[ig, ib]) * shortage
                        if ie == IC_CLAY:
                            for im in range(veget_max.shape[1]):
                                if veget_max[ig, im] > min_sechiba:
                                    area_rivero = (
                                        -coef_chanero[ie]
                                        * (1.0 - gravel[ig, ib])
                                        * shortage
                                        * veget_max[ig, im]
                                        / bulkdens[ig]
                                        / bioturb_depth
                                    )
                                    raw_frac = area_rivero / (veget_max[ig, im] * total_area[ig])
                                    frac_rivero = min(float(maxfrac_bankero), raw_frac)
                                    area_rivero = area_rivero * (
                                        frac_rivero / (raw_frac + min_sechiba)
                                    )
                                    if im > 0:
                                        erodep[ig, ib, IPOCA] -= 1.0e-3 * np.sum(carbon[ig, IACTIVE, im, :nlybiotur]) * area_rivero
                                        erodep[ig, ib, IPOCS] -= 1.0e-3 * np.sum(carbon[ig, ISLOW, im, :nlybiotur]) * area_rivero
                                        erodep[ig, ib, IPOCP] -= 1.0e-3 * np.sum(carbon[ig, IPASSIVE, im, :nlybiotur]) * area_rivero
                                        carbon[ig, IACTIVE : IPASSIVE + 1, im, :nlybiotur] = (
                                            (1.0 - frac_rivero) * carbon[ig, IACTIVE : IPASSIVE + 1, im, :nlybiotur]
                                        )
                            erodep[ig, ib, IPOCA : IPOCP + 1] = -seddep[ig, ib, IPOCA : IPOCP + 1] + erodep[
                                ig, ib, IPOCA : IPOCP + 1
                            ]
                            seddep[ig, ib, IPOCA : IPOCP + 1] = 0.0
                            flow[ig, ib, IPOCA : IPOCP + 1] = np.maximum(
                                frac_out * (stream[ig, ib, IPOCA : IPOCP + 1] - erodep[ig, ib, IPOCA : IPOCP + 1]),
                                0.0,
                            )
                            stream[ig, ib, IPOCA : IPOCP + 1] = np.maximum(
                                (1.0 - frac_out)
                                * (stream[ig, ib, IPOCA : IPOCP + 1] - erodep[ig, ib, IPOCA : IPOCP + 1]),
                                0.0,
                            )
                        seddep[ig, ib, sed_idx] = 0.0
                        flow[ig, ib, sed_idx] = max(frac_out * (stream[ig, ib, sed_idx] - erodep[ig, ib, sed_idx]), 0.0)
                        stream[ig, ib, sed_idx] = max(
                            (1.0 - frac_out) * (stream[ig, ib, sed_idx] - erodep[ig, ib, sed_idx]),
                            0.0,
                        )

            vseddep_daily = np.sum(erodep[ig, ib, ICLAYSED : ISANDSED + 1]) / bulkdens[ig]
            damavail[ig, ib] = max(0.0, damavail[ig, ib] - vseddep_daily)

    stream[:, :, IH2O] -= flow[:, :, IH2O]
    return RoutingStreamErosionStep(
        stream_reservoir=stream,
        stream_flow=flow,
        stream_seddep=seddep,
        stream_erodep=erodep,
        sedcap=sedcap,
        stream_damavailbas=damavail,
        stream_damavail=np.sum(damavail, axis=1),
        carbon_32l=carbon,
    )


def routing_poc_fraction_matrix(
    *,
    frac_carb_ap=0.004,
    frac_carb_sa=0.42,
    frac_carb_pa=0.45,
    metabolic_ref_frac=0.85,
) -> np.ndarray:
    """Return routing POC transfer fractions from ``routing_flow`` lines 4514-4526."""

    frac = np.zeros((3, 3), dtype=np.float64)
    frac[IACTIVE, IPASSIVE] = float(frac_carb_ap) / (1.0 - float(metabolic_ref_frac))
    frac[IACTIVE, ISLOW] = 1.0 - frac[IACTIVE, IPASSIVE]
    frac[ISLOW, IACTIVE] = float(frac_carb_sa)
    frac[ISLOW, IPASSIVE] = 1.0 - frac[ISLOW, IACTIVE]
    frac[IPASSIVE, IACTIVE] = float(frac_carb_pa)
    frac[IPASSIVE, ISLOW] = 1.0 - frac[IPASSIVE, IACTIVE]
    return frac


def routing_poc_decomposition(
    *,
    poc_reservoir,
    poc_dec_rate,
    frac_pocpool,
    f_socdoc=0.03,
    cue=0.3,
    min_sechiba=1.0e-8,
) -> RoutingPOCDecomposition:
    """Decompose a three-pool POC reservoir.

    Fortran provenance: ``routing.f90::poc_decomposition`` lines 11688-11737.
    The source decays each positive POC pool, partitions active losses to
    labile DOC, slow+passive losses to stable DOC, sends `(1-CUE)` of the
    non-DOC loss to CO2, and redistributes `CUE` among POC pools using
    `frac_pocpool(origin,destination)`.
    """

    poc = np.asarray(poc_reservoir, dtype=np.float64).copy()
    dec = np.asarray(poc_dec_rate, dtype=np.float64)
    frac = np.asarray(frac_pocpool, dtype=np.float64)
    if poc.shape != (3,) or dec.shape != (3,) or frac.shape != (3, 3):
        raise ValueError("POC decomposition requires poc(3), poc_dec_rate(3), and frac_pocpool(3,3)")

    poc_flux = np.where(poc > min_sechiba, dec * poc, 0.0)
    poc = poc - poc_flux
    flux_doc = np.zeros((2,), dtype=np.float64)
    flux_doc[IDOCLABILE] = poc_flux[IACTIVE] * float(f_socdoc)
    flux_doc[IDOCSTABLE] = (poc_flux[ISLOW] + poc_flux[IPASSIVE]) * float(f_socdoc)
    poc_flux = poc_flux * (1.0 - float(f_socdoc))
    flux_co2 = (1.0 - float(cue)) * np.sum(poc_flux)
    poc[IACTIVE] += frac[ISLOW, IACTIVE] * float(cue) * poc_flux[ISLOW] + frac[IPASSIVE, IACTIVE] * float(cue) * poc_flux[IPASSIVE]
    poc[ISLOW] += frac[IACTIVE, ISLOW] * float(cue) * poc_flux[IACTIVE] + frac[IPASSIVE, ISLOW] * float(cue) * poc_flux[IPASSIVE]
    poc[IPASSIVE] += frac[IACTIVE, IPASSIVE] * float(cue) * poc_flux[IACTIVE] + frac[ISLOW, IPASSIVE] * float(cue) * poc_flux[ISLOW]
    return RoutingPOCDecomposition(poc_reservoir=poc, flux_poc2doc=flux_doc, flux_poc2co2=float(flux_co2))


def _routing_co2_auxiliary(temp_sol: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_water = 6.13 + 0.80 * (temp_sol - 273.15)
    k_co2 = 10.0 ** (-(-2.22e-06 * (t_water**3) - 1.91e-05 * (t_water**2) + 1.63e-02 * t_water + 1.11))
    schmitt = 1911.0 - 118.11 * t_water + 3.453 * (t_water**2) - 0.0413 * (t_water**3)
    return t_water, k_co2, schmitt


def _routing_segmented_fco2(
    *,
    store_co2: float,
    water: float,
    k_co2: float,
    pco2_atm: float,
    k_act: float,
    surface_area: float,
    nstep_fco2: int,
    decompos_step: float,
    poc2co2_step: float,
    msmlr_c: float,
    mille: float,
) -> tuple[float, float, float]:
    fco2_ac = 0.0
    pco2_avg = 0.0
    store = float(store_co2)
    for _istep in range(int(nstep_fco2)):
        pco2_wa = ((store / water) / msmlr_c) / k_co2
        excess = k_co2 * (pco2_wa - 1.0e-06 * pco2_atm) * msmlr_c
        fco2 = max(min(excess * k_act * surface_area * mille / float(nstep_fco2), excess * water), 0.0)
        fco2_ac += fco2
        pco2_avg += pco2_wa
        store = store - fco2 + decompos_step + poc2co2_step
    return store, fco2_ac, pco2_avg / float(nstep_fco2)


def routing_co2_chemistry_step(
    *,
    fast_reservoir,
    slow_reservoir,
    stream_reservoir,
    flood_reservoir,
    pond_reservoir,
    stream_seddep,
    stream_erodep,
    flood_drainage,
    flood_dep_sed,
    flood_dep_poc,
    return_swamp,
    flood_inp_bas,
    stream_inp_bas,
    routing_area,
    flood_frac_bas,
    stream_area,
    stream_area_bas,
    pond_frac,
    temp_sol,
    swamp,
    floodplains,
    ok_doc: bool = True,
    doswamps: bool = False,
    do_floodplains: bool = False,
    nstep_fco2: int = 240,
    pco2_atm=285.0,
    docfast_dec=0.3,
    docslow_dec=0.01,
    poca_dec=0.0090,
    pocs_dec=0.0025,
    ratio_stream_sed=0.56,
    k_stream=3.0,
    k_flood=3.0,
    msmlr_c=12.011e-03,
    frac_pocpool=None,
    f_socdoc=0.03,
    cue=0.3,
    mille=1000.0,
    min_sechiba=1.0e-8,
) -> RoutingCO2ChemistryStep:
    """Run routing DOC/POC/CO2 chemistry and CO2 evasion for explicit state.

    Fortran provenance: ``routing.f90::routing_flow`` lines 4502-4929 and
    ``poc_decomposition`` lines 11688-11737. The helper follows the source
    order for floodplains, stream water, deposited river-bed POC, fast, slow,
    and pond reservoirs, then water-weights basin pCO2 and converts atm to
    micro-atm.
    """

    fast = np.asarray(fast_reservoir, dtype=np.float64).copy()
    slow = np.asarray(slow_reservoir, dtype=np.float64).copy()
    stream = np.asarray(stream_reservoir, dtype=np.float64).copy()
    flood = np.asarray(flood_reservoir, dtype=np.float64).copy()
    pond = np.asarray(pond_reservoir, dtype=np.float64).copy()
    seddep = np.asarray(stream_seddep, dtype=np.float64).copy()
    erodep = np.asarray(stream_erodep, dtype=np.float64).copy()
    drainage = np.asarray(flood_drainage, dtype=np.float64).copy()
    dep_sed = np.asarray(flood_dep_sed, dtype=np.float64).copy()
    dep_poc = np.asarray(flood_dep_poc, dtype=np.float64).copy()
    ret_swamp = np.asarray(return_swamp, dtype=np.float64).copy()
    flood_inp = np.asarray(flood_inp_bas, dtype=np.float64)
    stream_inp = np.asarray(stream_inp_bas, dtype=np.float64)
    area = np.asarray(routing_area, dtype=np.float64)
    flood_area = np.asarray(flood_frac_bas, dtype=np.float64)
    stream_area = np.asarray(stream_area, dtype=np.float64)
    stream_area_bas = np.asarray(stream_area_bas, dtype=np.float64)
    pond_frac = np.asarray(pond_frac, dtype=np.float64)
    temp_sol = np.asarray(temp_sol, dtype=np.float64)
    swamp = np.asarray(swamp, dtype=np.float64)
    floodplains = np.asarray(floodplains, dtype=np.float64)
    frac = routing_poc_fraction_matrix() if frac_pocpool is None else np.asarray(frac_pocpool, dtype=np.float64)

    if fast.ndim != 3 or slow.shape != fast.shape or stream.shape != fast.shape or flood.shape != fast.shape:
        raise ValueError("fast/slow/stream/flood reservoirs must share shape (npts,nbas,nflow)")
    npts, nbas, nflow = fast.shape
    if nflow <= ISANDSED:
        raise ValueError("routing CO2 chemistry requires nflow >= 10")
    if pond.shape != (npts, nflow):
        raise ValueError("pond_reservoir must have shape (npts,nflow)")
    for name, arr in (
        ("stream_seddep", seddep),
        ("stream_erodep", erodep),
        ("flood_drainage", drainage),
        ("return_swamp", ret_swamp),
        ("flood_inp_bas", flood_inp),
        ("stream_inp_bas", stream_inp),
    ):
        if arr.shape != fast.shape:
            raise ValueError(f"{name} must have shape (npts,nbas,nflow)")
    for name, arr in (("flood_dep_sed", dep_sed), ("flood_dep_poc", dep_poc)):
        if arr.shape != (npts, nbas, 3):
            raise ValueError(f"{name} must have shape (npts,nbas,3)")
    for name, arr in (("routing_area", area), ("flood_frac_bas", flood_area), ("stream_area_bas", stream_area_bas)):
        if arr.shape != (npts, nbas):
            raise ValueError(f"{name} must have shape (npts,nbas)")
    for name, arr in (
        ("stream_area", stream_area),
        ("pond_frac", pond_frac),
        ("temp_sol", temp_sol),
        ("swamp", swamp),
        ("floodplains", floodplains),
    ):
        if arr.shape != (npts,):
            raise ValueError(f"{name} must have shape (npts,)")
    if frac.shape != (3, 3):
        raise ValueError("frac_pocpool must have shape (3,3)")
    if int(nstep_fco2) <= 0:
        raise ValueError("nstep_fco2 must be positive")

    fco2_aq = np.zeros((npts, NAQSYS), dtype=np.float64)
    pco2_aq = np.zeros((npts, NAQSYS), dtype=np.float64)
    poc_co2_aq = np.zeros((npts, NAQSYS), dtype=np.float64)
    poc_doc_aq = np.zeros((npts, NAQSYS), dtype=np.float64)
    poc_co2_rivbed = np.zeros((npts,), dtype=np.float64)
    poc_doc_rivbed = np.zeros((npts, 2), dtype=np.float64)
    pco2_bas = np.zeros((npts, nbas, NAQSYS), dtype=np.float64)
    t_water, k_co2, schmitt = _routing_co2_auxiliary(temp_sol)

    if not bool(ok_doc):
        return RoutingCO2ChemistryStep(
            fast_reservoir=fast,
            slow_reservoir=slow,
            stream_reservoir=stream,
            flood_reservoir=flood,
            pond_reservoir=pond,
            stream_seddep=seddep,
            stream_erodep=erodep,
            flood_drainage=drainage,
            flood_dep_sed=dep_sed,
            flood_dep_poc=dep_poc,
            return_swamp=ret_swamp,
            fco2_aq=fco2_aq,
            pco2_aq=pco2_aq,
            poc_co2_aq=poc_co2_aq,
            poc_doc_aq=poc_doc_aq,
            poc_co2_rivbed=poc_co2_rivbed,
            poc_doc_rivbed=poc_doc_rivbed,
            pco2_aq_bas=pco2_bas,
            t_water=t_water,
            k_co2=k_co2,
            schmitt=schmitt,
        )

    total_area = np.sum(area, axis=1)
    for ig in range(npts):
        temp_factor = 1.073 ** (t_water[ig] - 28.0)
        fast_dec_act = float(docfast_dec) * temp_factor
        slow_dec_act = float(docslow_dec) * temp_factor
        poc_dec_act = np.asarray([float(poca_dec), float(pocs_dec), float(poca_dec)], dtype=np.float64) * temp_factor
        for ib in range(nbas):
            flood[ig, ib, IDOCL : IDOCR + 1] += flood_inp[ig, ib, IDOCL : IDOCR + 1]
            if flood[ig, ib, IH2O] > min_sechiba:
                decompos = fast_dec_act * flood[ig, ib, IDOCL] + slow_dec_act * flood[ig, ib, IDOCR] + flood_inp[ig, ib, ICO2AQ]
                if bool(doswamps):
                    if floodplains[ig] <= min_sechiba:
                        raise ValueError("doswamps flood CO2 branch requires positive floodplains area")
                    k_act = (swamp[ig] / floodplains[ig] * float(k_flood) + (1.0 - swamp[ig] / floodplains[ig]) * float(k_stream))
                else:
                    k_act = float(k_stream)
                k_act *= (600.0 / schmitt[ig]) ** 0.5
                poc = routing_poc_decomposition(
                    poc_reservoir=flood[ig, ib, IPOCA : IPOCP + 1],
                    poc_dec_rate=poc_dec_act,
                    frac_pocpool=frac,
                    f_socdoc=f_socdoc,
                    cue=cue,
                    min_sechiba=min_sechiba,
                )
                flood[ig, ib, IPOCA : IPOCP + 1] = poc.poc_reservoir
                store, fco2, pco2 = _routing_segmented_fco2(
                    store_co2=flood[ig, ib, ICO2AQ],
                    water=flood[ig, ib, IH2O],
                    k_co2=k_co2[ig],
                    pco2_atm=float(pco2_atm),
                    k_act=k_act,
                    surface_area=flood_area[ig, ib] * area[ig, ib],
                    nstep_fco2=int(nstep_fco2),
                    decompos_step=decompos / float(nstep_fco2),
                    poc2co2_step=poc.flux_poc2co2 / float(nstep_fco2),
                    msmlr_c=float(msmlr_c),
                    mille=float(mille),
                )
                flood[ig, ib, IDOCL] = (1.0 - fast_dec_act) * flood[ig, ib, IDOCL] + poc.flux_poc2doc[IDOCLABILE]
                flood[ig, ib, IDOCR] = (1.0 - slow_dec_act) * flood[ig, ib, IDOCR] + poc.flux_poc2doc[IDOCSTABLE]
                flood[ig, ib, ICO2AQ] = store
                fco2_aq[ig, IFLOODR] += fco2
                pco2_bas[ig, ib, IFLOODR] = pco2
                poc_co2_aq[ig, IFLOODR] += poc.flux_poc2co2
                poc_doc_aq[ig, IFLOODR] += np.sum(poc.flux_poc2doc)
            else:
                drainage[ig, ib, IH2O : ICO2AQ + 1] += flood[ig, ib, IH2O : ICO2AQ + 1]
                dep_poc[ig, ib, :] += flood[ig, ib, IPOCA : IPOCP + 1]
                dep_sed[ig, ib, :] += flood[ig, ib, ICLAYSED : ISANDSED + 1]
                flood[ig, ib, :] = 0.0
                drainage[ig, ib, ICO2AQ] += flood_inp[ig, ib, ICO2AQ]

            stream[ig, ib, IDOCL : IDOCR + 1] += stream_inp[ig, ib, IDOCL : IDOCR + 1]
            stream_active = stream[ig, ib, IH2O] > min_sechiba and stream_area[ig] > 0.0
            if stream_active:
                decompos = fast_dec_act * stream[ig, ib, IDOCL] + slow_dec_act * stream[ig, ib, IDOCR] + stream_inp[ig, ib, ICO2AQ]
                k_act = float(k_stream) * (600.0 / schmitt[ig]) ** 0.5
                poc = routing_poc_decomposition(
                    poc_reservoir=stream[ig, ib, IPOCA : IPOCP + 1],
                    poc_dec_rate=poc_dec_act,
                    frac_pocpool=frac,
                    f_socdoc=f_socdoc,
                    cue=cue,
                    min_sechiba=min_sechiba,
                )
                stream[ig, ib, IPOCA : IPOCP + 1] = poc.poc_reservoir
                store, fco2, pco2 = _routing_segmented_fco2(
                    store_co2=stream[ig, ib, ICO2AQ],
                    water=stream[ig, ib, IH2O],
                    k_co2=k_co2[ig],
                    pco2_atm=float(pco2_atm),
                    k_act=k_act,
                    surface_area=stream_area_bas[ig, ib],
                    nstep_fco2=int(nstep_fco2),
                    decompos_step=decompos / float(nstep_fco2),
                    poc2co2_step=poc.flux_poc2co2 / float(nstep_fco2),
                    msmlr_c=float(msmlr_c),
                    mille=float(mille),
                )
                stream[ig, ib, IDOCL] = (1.0 - fast_dec_act) * stream[ig, ib, IDOCL] + poc.flux_poc2doc[IDOCLABILE]
                stream[ig, ib, IDOCR] = (1.0 - slow_dec_act) * stream[ig, ib, IDOCR] + poc.flux_poc2doc[IDOCSTABLE]
                stream[ig, ib, ICO2AQ] = store
                fco2_aq[ig, ISTREAMR] += fco2
                pco2_bas[ig, ib, ISTREAMR] = pco2
                poc_co2_aq[ig, ISTREAMR] += poc.flux_poc2co2
                poc_doc_aq[ig, ISTREAMR] += np.sum(poc.flux_poc2doc)
            elif bool(doswamps):
                ret_swamp[ig, ib, IH2O : ICO2AQ + 1] += stream[ig, ib, IH2O : ICO2AQ + 1]
                erodep[ig, ib, IPOCA : ISANDSED + 1] += stream[ig, ib, IPOCA : ISANDSED + 1]
                stream[ig, ib, :] = 0.0
                ret_swamp[ig, ib, ICO2AQ] += stream_inp[ig, ib, ICO2AQ]
            elif bool(do_floodplains):
                drainage[ig, ib, IH2O : ICO2AQ + 1] += stream[ig, ib, IH2O : ICO2AQ + 1]
                erodep[ig, ib, IPOCA : ISANDSED + 1] += stream[ig, ib, IPOCA : ISANDSED + 1]
                stream[ig, ib, :] = 0.0
                drainage[ig, ib, ICO2AQ] += stream_inp[ig, ib, ICO2AQ]

            if stream[ig, ib, IH2O] > min_sechiba and stream_area[ig] > 0.0:
                poc = routing_poc_decomposition(
                    poc_reservoir=seddep[ig, ib, IPOCA : IPOCP + 1],
                    poc_dec_rate=float(ratio_stream_sed) * poc_dec_act,
                    frac_pocpool=frac,
                    f_socdoc=f_socdoc,
                    cue=cue,
                    min_sechiba=min_sechiba,
                )
                seddep[ig, ib, IPOCA : IPOCP + 1] = poc.poc_reservoir
                stream[ig, ib, IDOCL] += poc.flux_poc2doc[IDOCLABILE]
                stream[ig, ib, IDOCR] += poc.flux_poc2doc[IDOCSTABLE]
                stream[ig, ib, ICO2AQ] += poc.flux_poc2co2
                erodep[ig, ib, IDOCL] -= poc.flux_poc2doc[IDOCLABILE]
                erodep[ig, ib, IDOCR] -= poc.flux_poc2doc[IDOCSTABLE]
                erodep[ig, ib, ICO2AQ] -= poc.flux_poc2co2
                poc_co2_rivbed[ig] += poc.flux_poc2co2
                poc_doc_rivbed[ig, IDOCLABILE] += poc.flux_poc2doc[IDOCLABILE]
                poc_doc_rivbed[ig, IDOCSTABLE] += poc.flux_poc2doc[IDOCSTABLE]

            if fast[ig, ib, IH2O] > min_sechiba:
                decompos = fast_dec_act * fast[ig, ib, IDOCL] + slow_dec_act * fast[ig, ib, IDOCR]
                poc = routing_poc_decomposition(
                    poc_reservoir=fast[ig, ib, IPOCA : IPOCP + 1],
                    poc_dec_rate=poc_dec_act,
                    frac_pocpool=frac,
                    f_socdoc=f_socdoc,
                    cue=cue,
                    min_sechiba=min_sechiba,
                )
                fast[ig, ib, IPOCA : IPOCP + 1] = poc.poc_reservoir
                store = fast[ig, ib, ICO2AQ] + decompos + poc.flux_poc2co2
                pco2 = ((store / fast[ig, ib, IH2O]) / float(msmlr_c)) / k_co2[ig]
                excess = k_co2[ig] * (pco2 - 1.0e-06 * float(pco2_atm)) * float(msmlr_c)
                fco2 = max(excess * fast[ig, ib, IH2O], 0.0)
                fast[ig, ib, IDOCL] = (1.0 - fast_dec_act) * fast[ig, ib, IDOCL] + poc.flux_poc2doc[IDOCLABILE]
                fast[ig, ib, IDOCR] = (1.0 - slow_dec_act) * fast[ig, ib, IDOCR] + poc.flux_poc2doc[IDOCSTABLE]
                fast[ig, ib, ICO2AQ] = store - fco2
                fco2_aq[ig, IFASTR] += fco2
                pco2_bas[ig, ib, IFASTR] = pco2
                poc_co2_aq[ig, IFASTR] += poc.flux_poc2co2
                poc_doc_aq[ig, IFASTR] += np.sum(poc.flux_poc2doc)
            else:
                fast[ig, ib, :] = 0.0

            if slow[ig, ib, IH2O] > min_sechiba:
                decompos = fast_dec_act * slow[ig, ib, IDOCL] + slow_dec_act * slow[ig, ib, IDOCR]
                slow[ig, ib, IDOCL] = (1.0 - fast_dec_act) * slow[ig, ib, IDOCL]
                slow[ig, ib, IDOCR] = (1.0 - slow_dec_act) * slow[ig, ib, IDOCR]
                slow[ig, ib, ICO2AQ] += decompos
                pco2_bas[ig, ib, ISLOWR] = ((slow[ig, ib, ICO2AQ] / slow[ig, ib, IH2O]) / float(msmlr_c)) / k_co2[ig]
            else:
                slow[ig, ib, :] = 0.0

        if pond[ig, IH2O] > min_sechiba:
            decompos = fast_dec_act * pond[ig, IDOCL] + slow_dec_act * pond[ig, IDOCR]
            k_act = float(k_stream) * (600.0 / schmitt[ig]) ** 0.5
            poc = routing_poc_decomposition(
                poc_reservoir=pond[ig, IPOCA : IPOCP + 1],
                poc_dec_rate=poc_dec_act,
                frac_pocpool=frac,
                f_socdoc=f_socdoc,
                cue=cue,
                min_sechiba=min_sechiba,
            )
            pond[ig, IPOCA : IPOCP + 1] = poc.poc_reservoir
            store, fco2, pco2 = _routing_segmented_fco2(
                store_co2=pond[ig, ICO2AQ],
                water=pond[ig, IH2O],
                k_co2=k_co2[ig],
                pco2_atm=float(pco2_atm),
                k_act=k_act,
                surface_area=total_area[ig] * pond_frac[ig],
                nstep_fco2=int(nstep_fco2),
                decompos_step=decompos / float(nstep_fco2),
                poc2co2_step=poc.flux_poc2co2 / float(nstep_fco2),
                msmlr_c=float(msmlr_c),
                mille=float(mille),
            )
            pond[ig, IDOCL] = (1.0 - fast_dec_act) * pond[ig, IDOCL] + poc.flux_poc2doc[IDOCLABILE]
            pond[ig, IDOCR] = (1.0 - slow_dec_act) * pond[ig, IDOCR] + poc.flux_poc2doc[IDOCSTABLE]
            pond[ig, ICO2AQ] = store
            fco2_aq[ig, IPONDR] = fco2
            pco2_aq[ig, IPONDR] = pco2
            poc_co2_aq[ig, IPONDR] += poc.flux_poc2co2
            poc_doc_aq[ig, IPONDR] += np.sum(poc.flux_poc2doc)

    for ig in range(npts):
        for aq_idx, reservoir in ((IFLOODR, flood), (ISTREAMR, stream), (IFASTR, fast), (ISLOWR, slow)):
            water = np.sum(reservoir[ig, :, IH2O])
            if water > min_sechiba:
                pco2_aq[ig, aq_idx] = np.sum(pco2_bas[ig, :, aq_idx] * reservoir[ig, :, IH2O]) / water
    pco2_aq *= 1.0e6

    return RoutingCO2ChemistryStep(
        fast_reservoir=fast,
        slow_reservoir=slow,
        stream_reservoir=stream,
        flood_reservoir=flood,
        pond_reservoir=pond,
        stream_seddep=seddep,
        stream_erodep=erodep,
        flood_drainage=drainage,
        flood_dep_sed=dep_sed,
        flood_dep_poc=dep_poc,
        return_swamp=ret_swamp,
        fco2_aq=fco2_aq,
        pco2_aq=pco2_aq,
        poc_co2_aq=poc_co2_aq,
        poc_doc_aq=poc_doc_aq,
        poc_co2_rivbed=poc_co2_rivbed,
        poc_doc_rivbed=poc_doc_rivbed,
        pco2_aq_bas=pco2_bas,
        t_water=t_water,
        k_co2=k_co2,
        schmitt=schmitt,
    )


def routing_irrigation_step(
    *,
    fast_reservoir,
    slow_reservoir,
    stream_reservoir,
    routing_area,
    irrigated,
    vegtot,
    humrel,
    runoff,
    precip,
    reinfiltration,
    transpot_mean,
    do_irrigation: bool,
    ok_doc: bool = True,
    crop_coef=1.0,
    resolution=None,
    neighbours=None,
    min_sechiba=1.0e-8,
) -> RoutingIrrigationStep:
    """Withdraw routing water for irrigation and optional adduction.

    Fortran provenance: ``routing.f90::routing_flow`` lines 4975-5205. The
    source computes net crop irrigation requirement, withdraws water first
    from stream, then fast, then slow reservoirs within each basin, scales
    transported matter by the remaining water in each reservoir when `ok_doc`
    is true, draws remaining deficits from the fullest stream reservoir in the
    same grid cell, and optionally imports water from neighboring grid cells
    when grid resolution is below 100 km.
    """

    fast = np.asarray(fast_reservoir, dtype=np.float64).copy()
    slow = np.asarray(slow_reservoir, dtype=np.float64).copy()
    stream = np.asarray(stream_reservoir, dtype=np.float64).copy()
    area = np.asarray(routing_area, dtype=np.float64)
    irrigated = np.asarray(irrigated, dtype=np.float64)
    vegtot = np.asarray(vegtot, dtype=np.float64)
    humrel = np.asarray(humrel, dtype=np.float64)
    runoff = np.asarray(runoff, dtype=np.float64)
    precip = np.asarray(precip, dtype=np.float64)
    reinfiltration = np.asarray(reinfiltration, dtype=np.float64)
    transpot_mean = np.asarray(transpot_mean, dtype=np.float64)

    if fast.ndim != 3 or slow.shape != fast.shape or stream.shape != fast.shape:
        raise ValueError("fast/slow/stream reservoirs must share shape (npts,nbas,nflow)")
    npts, nbas, nflow = fast.shape
    if nflow <= IH2O:
        raise ValueError("routing irrigation requires at least a water slot")
    if area.shape != (npts, nbas):
        raise ValueError("routing_area must have shape (npts,nbas)")
    for name, arr in (
        ("irrigated", irrigated),
        ("vegtot", vegtot),
        ("humrel", humrel),
        ("transpot_mean", transpot_mean),
    ):
        if arr.shape != (npts,):
            raise ValueError(f"{name} must have shape (npts,)")
    for name, arr in (("runoff", runoff), ("precip", precip), ("reinfiltration", reinfiltration)):
        if arr.shape != (npts, nflow):
            raise ValueError(f"{name} must have shape (npts,nflow)")
    total_area = np.sum(area, axis=1)
    if np.any(total_area <= 0.0):
        raise ValueError("routing irrigation requires positive total routing area")

    irrig_netereq = np.zeros((npts,), dtype=np.float64)
    irrig_needs = np.zeros((npts, nbas), dtype=np.float64)
    irrig_actual = np.zeros((npts, nbas, nflow), dtype=np.float64)
    irrig_deficit = np.zeros((npts, nbas), dtype=np.float64)
    irrig_adduct = np.zeros((npts, nbas, nflow), dtype=np.float64)

    if not bool(do_irrigation):
        return RoutingIrrigationStep(
            fast_reservoir=fast,
            slow_reservoir=slow,
            stream_reservoir=stream,
            irrig_netereq=irrig_netereq,
            irrig_needs=irrig_needs,
            irrig_actual=irrig_actual,
            irrig_deficit=irrig_deficit,
            irrig_adduct=irrig_adduct,
        )

    for ig in range(npts):
        if vegtot[ig] > min_sechiba and humrel[ig] < 1.0 - min_sechiba and runoff[ig, IH2O] < min_sechiba:
            irrig_netereq[ig] = (irrigated[ig] / total_area[ig]) * max(
                0.0,
                float(crop_coef) * transpot_mean[ig] - (precip[ig, IH2O] + reinfiltration[ig, IH2O]),
            )

        for ib in range(nbas):
            if area[ig, ib] <= 0.0:
                continue
            irrig_needs[ig, ib] = irrig_netereq[ig] * area[ig, ib]
            irrig_actual[ig, ib, IH2O] = min(
                irrig_needs[ig, ib],
                stream[ig, ib, IH2O] + fast[ig, ib, IH2O] + slow[ig, ib, IH2O],
            )
            slow_prec = slow[ig, ib, :].copy()
            fast_prec = fast[ig, ib, :].copy()
            stream_prec = stream[ig, ib, :].copy()

            stream_short = min(0.0, stream[ig, ib, IH2O] - irrig_actual[ig, ib, IH2O])
            fast_short = min(0.0, fast[ig, ib, IH2O] + stream_short)
            slow[ig, ib, IH2O] = max(0.0, slow[ig, ib, IH2O] + fast_short)
            fast[ig, ib, IH2O] = max(0.0, fast[ig, ib, IH2O] + stream_short)
            stream[ig, ib, IH2O] = max(0.0, stream[ig, ib, IH2O] - irrig_actual[ig, ib, IH2O])

            if bool(ok_doc) and nflow > IH2O + 1:
                if slow_prec[IH2O] > 0.0:
                    slow[ig, ib, IH2O + 1 : nflow] = slow[ig, ib, IH2O] * slow[ig, ib, IH2O + 1 : nflow] / slow_prec[IH2O]
                else:
                    slow[ig, ib, IH2O + 1 : nflow] = 0.0
                if fast_prec[IH2O] > 0.0:
                    fast[ig, ib, IH2O + 1 : nflow] = fast[ig, ib, IH2O] * fast[ig, ib, IH2O + 1 : nflow] / fast_prec[IH2O]
                else:
                    fast[ig, ib, IH2O + 1 : nflow] = 0.0
                if stream_prec[IH2O] > 0.0:
                    stream[ig, ib, IH2O + 1 : nflow] = (
                        stream[ig, ib, IH2O] * stream[ig, ib, IH2O + 1 : nflow] / stream_prec[IH2O]
                    )
                else:
                    stream[ig, ib, IH2O + 1 : nflow] = 0.0
                irrig_actual[ig, ib, IH2O + 1 : nflow] = (
                    slow_prec[IH2O + 1 : nflow]
                    - slow[ig, ib, IH2O + 1 : nflow]
                    + fast_prec[IH2O + 1 : nflow]
                    - fast[ig, ib, IH2O + 1 : nflow]
                    + stream_prec[IH2O + 1 : nflow]
                    - stream[ig, ib, IH2O + 1 : nflow]
                )
            irrig_deficit[ig, ib] = irrig_needs[ig, ib] - irrig_actual[ig, ib, IH2O]

        for ib in range(nbas):
            stream_tot = np.sum(stream[ig, :, IH2O])
            while irrig_deficit[ig, ib] > min_sechiba and stream_tot > min_sechiba:
                ib2 = int(np.argmax(stream[ig, :, IH2O]))
                irrig_adduct[ig, ib, IH2O] = min(irrig_deficit[ig, ib], stream[ig, ib2, IH2O])
                stream_prec = stream[ig, ib2, :].copy()
                stream[ig, ib2, IH2O] = stream[ig, ib2, IH2O] - irrig_adduct[ig, ib, IH2O]
                if bool(ok_doc) and nflow > IH2O + 1:
                    if stream_prec[IH2O] > 0.0:
                        irrig_adduct[ig, ib, IH2O + 1 : nflow] = stream[ig, ib2, IH2O + 1 : nflow] * (
                            1.0 - stream[ig, ib2, IH2O] / stream_prec[IH2O]
                        )
                        stream[ig, ib2, IH2O + 1 : nflow] = (
                            stream[ig, ib2, IH2O + 1 : nflow] - irrig_adduct[ig, ib, IH2O + 1 : nflow]
                        )
                    else:
                        stream[ig, ib2, IH2O + 1 : nflow] = 0.0
                        irrig_adduct[ig, ib, IH2O + 1 : nflow] = 0.0
                irrig_deficit[ig, ib] = irrig_deficit[ig, ib] - irrig_adduct[ig, ib, IH2O]
                stream_tot = np.sum(stream[ig, :, IH2O])

    if resolution is not None or neighbours is not None:
        if resolution is None or neighbours is None:
            raise ValueError("resolution and neighbours must be provided together for neighbor irrigation adduction")
        resolution = np.asarray(resolution, dtype=np.float64)
        neighbours = np.asarray(neighbours)
        if resolution.shape != (npts, 2) or neighbours.ndim != 2 or neighbours.shape[0] != npts:
            raise ValueError("resolution must be (npts,2) and neighbours must be (npts,nneigh)")
        if np.any(neighbours < 0) or np.any(neighbours >= npts):
            raise ValueError("neighbours contain grid indices outside local/global state")
        for ig in range(npts):
            if resolution[ig, 0] >= 100000.0 or resolution[ig, 1] >= 100000.0:
                continue
            for ib in range(nbas):
                if irrig_deficit[ig, ib] <= min_sechiba:
                    continue
                best_grid = int(neighbours[ig, 0])
                best_basin = 0
                best_water = -np.inf
                for ig2 in neighbours[ig, :]:
                    for ib2 in range(nbas):
                        if stream[int(ig2), ib2, IH2O] > best_water:
                            best_water = stream[int(ig2), ib2, IH2O]
                            best_grid = int(ig2)
                            best_basin = ib2
                if area[best_grid, best_basin] > 0.0 and stream[best_grid, best_basin, IH2O] > 0.0:
                    adduction = np.zeros((nflow,), dtype=np.float64)
                    adduction[IH2O] = min(irrig_deficit[ig, ib], stream[best_grid, best_basin, IH2O])
                    if bool(ok_doc) and nflow > IH2O + 1:
                        if stream[best_grid, best_basin, IH2O] > 0.0:
                            adduction[IH2O + 1 : nflow] = (
                                adduction[IH2O]
                                * stream[best_grid, best_basin, IH2O + 1 : nflow]
                                / stream[best_grid, best_basin, IH2O]
                            )
                        else:
                            adduction[IH2O + 1 : nflow] = 0.0
                    stream[best_grid, best_basin, :] = stream[best_grid, best_basin, :] - adduction
                    irrig_deficit[ig, ib] = irrig_deficit[ig, ib] - adduction[IH2O]
                    irrig_adduct[ig, ib, :] = irrig_adduct[ig, ib, :] + adduction

    return RoutingIrrigationStep(
        fast_reservoir=fast,
        slow_reservoir=slow,
        stream_reservoir=stream,
        irrig_netereq=irrig_netereq,
        irrig_needs=irrig_needs,
        irrig_actual=irrig_actual,
        irrig_deficit=irrig_deficit,
        irrig_adduct=irrig_adduct,
    )


def _routing_fortran_routes_to_python(route_togrid: np.ndarray, route_tobasin: np.ndarray, npts: int, nbas: int) -> tuple[np.ndarray, np.ndarray]:
    rtg = np.asarray(route_togrid, dtype=np.int64)
    rtb = np.asarray(route_tobasin, dtype=np.int64)
    if rtg.shape != (npts, nbas) or rtb.shape != (npts, nbas):
        raise ValueError("route_togrid and route_tobasin must have shape (npts,nbas)")
    if np.any(rtg < 1) or np.any(rtg > npts):
        raise ValueError("Fortran route_togrid targets must be 1..npts")
    if np.any(rtb < 1) or np.any(rtb > nbas + 3):
        raise ValueError("Fortran route_tobasin targets must be 1..nbas+3")
    return rtg - 1, rtb - 1


def routing_flow_step(
    *,
    state: RoutingFlowState,
    runoff,
    drainage,
    floodout,
    precip,
    flood_inp,
    stream_inp,
    routing_area,
    topo_resid,
    route_togrid,
    route_tobasin,
    qflow_ave,
    stream_resave,
    basdrainarea,
    basgravel,
    bulkdens,
    zz_deep,
    veget_max,
    vegtot,
    totnobio,
    transpot_mean,
    humrel,
    k_litt,
    floodtemp,
    temp_sol,
    reinf_slope,
    irrigated,
    stream_area,
    headw_area,
    streamr10th,
    streamr50th,
    streamr90th,
    floodplains,
    floodh90th,
    swamp,
    hydrodiag,
    mask_coast,
    nb_coast_gridcells,
    dt_routing,
    do_floodplains: bool,
    dofloodinfilt: bool,
    doponds: bool,
    doswamps: bool,
    dostreamswell: bool,
    new_flood_scheme: bool,
    do_irrigation: bool,
    ok_doc: bool,
    limit_rivdepos: bool = False,
    check_riverbal: bool = False,
    water_balance=None,
    carbon_balance=None,
    resolution=None,
    neighbours=None,
    max_lake_reservoir=7000.0,
    min_sechiba=1.0e-8,
) -> RoutingFlowStepResult:
    """Run the audited ``routing_flow`` day sequence for explicit state.

    Fortran provenance: ``routing.f90::routing_flow`` lines 3180-5366,
    plus the vertical flood/pond input block at lines 3558-3681. This wrapper
    only wires source-backed kernels in source order. It accepts Fortran-style
    1-based ``route_togrid`` and ``route_tobasin`` arrays, including outlet
    codes ``nbas+1:nbas+3``, and converts them only for the local transport
    helper.
    """

    fast = np.asarray(state.fast_reservoir, dtype=np.float64)
    slow = np.asarray(state.slow_reservoir, dtype=np.float64)
    stream = np.asarray(state.stream_reservoir, dtype=np.float64)
    flood = np.asarray(state.flood_reservoir, dtype=np.float64)
    if fast.ndim != 3 or slow.shape != fast.shape or stream.shape != fast.shape or flood.shape != fast.shape:
        raise ValueError("state fast/slow/stream/flood reservoirs must share shape (npts,nbas,nflow)")
    npts, nbas, nflow = fast.shape
    route_togrid_py, route_tobasin_py = _routing_fortran_routes_to_python(route_togrid, route_tobasin, npts, nbas)
    transport_shape = (npts, nbas + 3, nflow)

    outflow = routing_reservoir_outflow_step(
        fast_reservoir=fast,
        slow_reservoir=slow,
        stream_reservoir=stream,
        topo_resid=topo_resid,
        route_tobasin=route_tobasin,
        qflow_ave=qflow_ave,
        stream_resave=stream_resave,
        stream_damavail=state.stream_damavail,
        dt_routing=dt_routing,
        min_sechiba=min_sechiba,
    )

    stream_erosion = routing_stream_erosion_step(
        stream_reservoir=outflow.stream_reservoir,
        stream_flow=outflow.stream_flow,
        stream_seddep=state.stream_seddep,
        qflow_avebas=outflow.qflow_avebas,
        basdrainarea=basdrainarea,
        basgravel=basgravel,
        stream_damavailbas=outflow.stream_damavailbas,
        routing_area=routing_area,
        bulkdens=bulkdens,
        zz_deep=zz_deep,
        veget_max=veget_max,
        carbon_32l=state.carbon_32l,
        dt_routing=dt_routing,
        min_sechiba=min_sechiba,
    )

    flood_pond_input = routing_flood_pond_input_step(
        flood_reservoir=flood,
        pond_reservoir=state.pond_reservoir,
        floodout=floodout,
        flood_inp=flood_inp,
        stream_inp=stream_inp,
        routing_area=routing_area,
        flood_frac=state.flood_frac,
        flood_frac_bas=state.flood_frac_bas,
        pond_frac=state.pond_frac,
        streamfl_frac=state.streamfl_frac,
        streamfl_frac_bas=state.streamfl_frac_bas,
        do_floodplains=do_floodplains,
        doponds=doponds,
        dostreamswell=dostreamswell,
        min_sechiba=min_sechiba,
    )

    floodplain = routing_floodplain_flux_step(
        flood_reservoir=flood_pond_input.flood_reservoir,
        stream_reservoir=stream_erosion.stream_reservoir,
        routing_area=routing_area,
        flood_frac=state.flood_frac,
        flood_frac_bas=state.flood_frac_bas,
        k_litt=k_litt,
        route_tobasin=route_tobasin,
        topo_resid=topo_resid,
        flood_dep_sed=flood_pond_input.flood_dep_sed,
        flood_dep_poc=flood_pond_input.flood_dep_poc,
        do_floodplains=do_floodplains,
        dofloodinfilt=dofloodinfilt,
        dt_routing=dt_routing,
        min_sechiba=min_sechiba,
    )

    pond = routing_pond_flux_step(
        fast_flow=outflow.fast_flow,
        pond_reservoir=flood_pond_input.pond_reservoir,
        routing_area=routing_area,
        pond_frac=state.pond_frac,
        k_litt=k_litt,
        reinf_slope=reinf_slope,
        flood_dep_sed=floodplain.flood_dep_sed,
        flood_dep_poc=floodplain.flood_dep_poc,
        doponds=doponds,
        dt_routing=dt_routing,
        min_sechiba=min_sechiba,
    )

    transport = routing_transport_between_basins_step(
        fast_flow=pond.fast_flow,
        slow_flow=outflow.slow_flow,
        stream_flow=stream_erosion.stream_flow,
        route_togrid=route_togrid_py,
        route_tobasin=route_tobasin_py,
        transport=np.zeros(transport_shape, dtype=np.float64),
        check_riverbal=check_riverbal,
        water_balance=water_balance,
        carbon_balance=carbon_balance,
    )

    swamp_flood = routing_swamp_flood_step(
        stream_reservoir=floodplain.stream_reservoir,
        transport=transport.transport,
        routing_area=routing_area,
        route_tobasin=route_tobasin,
        topo_resid=topo_resid,
        streamr50th=streamr50th,
        floodtemp=floodtemp,
        swamp=swamp,
        floodplains=floodplains,
        stream_area=stream_area,
        flood_reservoir=floodplain.flood_reservoir,
        flood_dep_sed=pond.flood_dep_sed,
        flood_dep_poc=pond.flood_dep_poc,
        doswamps=doswamps,
        do_floodplains=do_floodplains,
        new_flood_scheme=new_flood_scheme,
        dt_routing=dt_routing,
        min_sechiba=min_sechiba,
    )

    reservoir_update = routing_reservoir_update_step(
        runoff=runoff,
        drainage=drainage,
        routing_area=routing_area,
        fast_reservoir=outflow.fast_reservoir,
        slow_reservoir=outflow.slow_reservoir,
        stream_reservoir=floodplain.stream_reservoir,
        flood_reservoir=swamp_flood.flood_reservoir,
        pond_reservoir=pond.pond_reservoir,
        transport=transport.transport,
        return_swamp=swamp_flood.return_swamp,
        floods=swamp_flood.floods,
        flood_flow=floodplain.flood_flow,
        pond_inflow=pond.pond_inflow,
        pond_drainage=pond.pond_drainage,
        min_sechiba=min_sechiba,
    )

    area_fractions = routing_area_fractions_step(
        stream_reservoir=reservoir_update.stream_reservoir,
        routing_area=routing_area,
        totflood=reservoir_update.totflood,
        flood_reservoir=reservoir_update.flood_reservoir,
        pond_reservoir=reservoir_update.pond_reservoir,
        stream_seddep=stream_erosion.stream_seddep,
        vegtot=vegtot,
        bulkdens=bulkdens,
        stream_area=stream_area,
        headw_area=headw_area,
        streamr10th=streamr10th,
        streamr90th=streamr90th,
        floodplains=floodplains,
        floodh90th=floodh90th,
        dostreamswell=dostreamswell,
        do_floodplains=do_floodplains,
        doponds=doponds,
        limit_rivdepos=limit_rivdepos,
        min_sechiba=min_sechiba,
    )

    chemistry = routing_co2_chemistry_step(
        fast_reservoir=reservoir_update.fast_reservoir,
        slow_reservoir=reservoir_update.slow_reservoir,
        stream_reservoir=reservoir_update.stream_reservoir,
        flood_reservoir=reservoir_update.flood_reservoir,
        pond_reservoir=reservoir_update.pond_reservoir,
        stream_seddep=area_fractions.stream_seddep,
        stream_erodep=stream_erosion.stream_erodep,
        flood_drainage=floodplain.flood_drainage,
        flood_dep_sed=swamp_flood.flood_dep_sed,
        flood_dep_poc=swamp_flood.flood_dep_poc,
        return_swamp=reservoir_update.return_swamp,
        flood_inp_bas=flood_pond_input.flood_inp_bas,
        stream_inp_bas=flood_pond_input.stream_inp_bas,
        routing_area=routing_area,
        flood_frac_bas=area_fractions.flood_frac_bas,
        stream_area=stream_area,
        stream_area_bas=area_fractions.stream_area_bas,
        pond_frac=area_fractions.pond_frac,
        temp_sol=temp_sol,
        swamp=swamp,
        floodplains=floodplains,
        ok_doc=ok_doc,
        doswamps=doswamps,
        do_floodplains=do_floodplains,
        min_sechiba=min_sechiba,
    )

    return_reinfiltration = routing_return_reinfiltration_step(
        return_swamp=chemistry.return_swamp,
        pond_drainage=pond.pond_drainage,
        flood_drainage=chemistry.flood_drainage,
        routing_area=routing_area,
        do_floodplains=do_floodplains,
        doswamps=doswamps,
        doponds=doponds,
    )

    irrigation = routing_irrigation_step(
        fast_reservoir=chemistry.fast_reservoir,
        slow_reservoir=chemistry.slow_reservoir,
        stream_reservoir=chemistry.stream_reservoir,
        routing_area=routing_area,
        irrigated=irrigated,
        vegtot=vegtot,
        humrel=humrel,
        runoff=runoff,
        precip=precip,
        reinfiltration=return_reinfiltration.reinfiltration,
        transpot_mean=transpot_mean,
        do_irrigation=do_irrigation,
        ok_doc=ok_doc,
        resolution=resolution,
        neighbours=neighbours,
        min_sechiba=min_sechiba,
    )

    diagnostics = routing_flow_diagnostics_step(
        runoff=runoff,
        drainage=drainage,
        routing_area=routing_area,
        fast_flow=pond.fast_flow,
        slow_flow=outflow.slow_flow,
        stream_flow=stream_erosion.stream_flow,
        flood_flow=floodplain.flood_flow,
        pond_inflow=pond.pond_inflow,
        transport=transport.transport,
        return_swamp=chemistry.return_swamp,
        floods=swamp_flood.floods,
        previous_flood_diag=state.flood_diag,
        previous_pond_diag=state.pond_diag,
        lake_diag=state.lake_diag,
        hydrodiag=hydrodiag,
        fast_reservoir=irrigation.fast_reservoir,
        slow_reservoir=irrigation.slow_reservoir,
        stream_reservoir=irrigation.stream_reservoir,
        flood_reservoir=chemistry.flood_reservoir,
        pond_reservoir=chemistry.pond_reservoir,
        irrig_actual=irrigation.irrig_actual,
        irrig_adduct=irrigation.irrig_adduct,
        flood_dep_sed=chemistry.flood_dep_sed,
        flood_dep_poc=chemistry.flood_dep_poc,
        stream_erodep=chemistry.stream_erodep,
        stream_seddep=chemistry.stream_seddep,
        streamb_inflow=reservoir_update.streamb_inflow,
        poc_co2_rivbed=chemistry.poc_co2_rivbed,
        poc_doc_rivbed=chemistry.poc_doc_rivbed,
    )

    lake_overflow = routing_lake_overflow_step(
        lake_reservoir=state.lake_reservoir,
        coastalflow=diagnostics.coastalflow,
        routing_area=routing_area,
        mask_coast=mask_coast,
        nb_coast_gridcells=nb_coast_gridcells,
        max_lake_reservoir=max_lake_reservoir,
        min_sechiba=min_sechiba,
    )

    new_state = RoutingFlowState(
        fast_reservoir=irrigation.fast_reservoir,
        slow_reservoir=irrigation.slow_reservoir,
        stream_reservoir=irrigation.stream_reservoir,
        flood_reservoir=chemistry.flood_reservoir,
        pond_reservoir=chemistry.pond_reservoir,
        lake_reservoir=lake_overflow.lake_reservoir,
        stream_seddep=chemistry.stream_seddep,
        stream_damavail=stream_erosion.stream_damavail,
        carbon_32l=stream_erosion.carbon_32l,
        flood_frac=area_fractions.flood_frac,
        flood_frac_bas=area_fractions.flood_frac_bas,
        pond_frac=area_fractions.pond_frac,
        streamfl_frac=area_fractions.streamfl_frac,
        streamfl_frac_bas=area_fractions.streamfl_frac_bas,
        flood_diag=diagnostics.flood_diag,
        pond_diag=diagnostics.pond_diag,
        lake_diag=state.lake_diag,
    )

    return RoutingFlowStepResult(
        state=new_state,
        outflow=outflow,
        stream_erosion=stream_erosion,
        flood_pond_input=flood_pond_input,
        floodplain=floodplain,
        pond=pond,
        transport=transport,
        swamp_flood=swamp_flood,
        reservoir_update=reservoir_update,
        area_fractions=area_fractions,
        chemistry=chemistry,
        return_reinfiltration=return_reinfiltration,
        irrigation=irrigation,
        diagnostics=diagnostics,
        lake_overflow=lake_overflow,
    )


def routing_accumulator_step(
    *,
    state: RoutingAccumulatorState,
    floodout,
    precip_rain,
    runoff,
    drainage,
    temp_sol,
    veget_max,
    transpot,
    totfrac_nobio,
    k_litt,
    humrel,
    is_tree,
    dt_sechiba,
    dt_routing,
    flood_frac=None,
    streamfl_frac=None,
    doc_exp_agg=None,
    doc_ero_agg=None,
    poc_exp_agg=None,
    sed_exp_agg=None,
    ok_doc: bool = False,
    ibare_sechiba: int = 0,
    min_sechiba: float = 1.0e-8,
) -> RoutingAccumulatorStep:
    """Advance ``routing_main`` accumulators before the reservoir network.

    Fortran provenance: ``fortran_source/ORCHIDEE/src_sechiba/routing.f90``,
    subroutine ``routing_main`` lines 912-1007 accumulate routing means, update
    ``time_counter``, and zero all routing outputs before the daily routing
    gate. Lines 1014-1029 build ``flow_input``, ``flood_inp``, and
    ``stream_inp`` when ``NINT(time_counter) >= NINT(dt_routing)``.
    """

    floodout_mean = np.asarray(state.floodout_mean, dtype=np.float64).copy()
    precip_mean = np.asarray(state.precip_mean, dtype=np.float64).copy()
    runoff_mean = np.asarray(state.runoff_mean, dtype=np.float64).copy()
    drainage_mean = np.asarray(state.drainage_mean, dtype=np.float64).copy()
    temp_sol_mean = np.asarray(state.temp_sol_mean, dtype=np.float64).copy()
    transpot_mean = np.asarray(state.transpot_mean, dtype=np.float64).copy()
    totnobio_mean = np.asarray(state.totnobio_mean, dtype=np.float64).copy()
    k_litt_mean = np.asarray(state.k_litt_mean, dtype=np.float64).copy()
    humrel_mean = np.asarray(state.humrel_mean, dtype=np.float64).copy()
    vegtot_mean = np.asarray(state.vegtot_mean, dtype=np.float64).copy()

    npts, nflow = precip_mean.shape
    for name, arr in (
        ("runoff_mean", runoff_mean),
        ("drainage_mean", drainage_mean),
    ):
        if arr.shape != (npts, nflow):
            raise ValueError(f"{name} must match precip_mean shape")

    floodout = np.asarray(floodout, dtype=np.float64)
    precip_rain = np.asarray(precip_rain, dtype=np.float64)
    runoff = np.asarray(runoff, dtype=np.float64)
    drainage = np.asarray(drainage, dtype=np.float64)
    temp_sol = np.asarray(temp_sol, dtype=np.float64)
    veget_max = np.asarray(veget_max, dtype=np.float64)
    transpot = np.asarray(transpot, dtype=np.float64)
    totfrac_nobio = np.asarray(totfrac_nobio, dtype=np.float64)
    k_litt = np.asarray(k_litt, dtype=np.float64)
    humrel = np.asarray(humrel, dtype=np.float64)
    is_tree = np.asarray(is_tree, dtype=bool)
    if not (
        floodout.shape == precip_rain.shape == runoff.shape == drainage.shape == temp_sol.shape == (npts,)
    ):
        raise ValueError("routing scalar inputs must have shape (npts,)")
    if veget_max.shape != transpot.shape or humrel.shape != veget_max.shape or veget_max.shape[0] != npts:
        raise ValueError("veget_max, transpot, and humrel must share shape (npts,nvm)")
    nvm = veget_max.shape[1]
    if is_tree.shape != (nvm,):
        raise ValueError("is_tree must have shape (nvm,)")
    if totfrac_nobio.shape != (npts,) or k_litt.shape != (npts,):
        raise ValueError("totfrac_nobio and k_litt must have shape (npts,)")

    dt_scale = float(dt_sechiba) / float(dt_routing)
    floodout_mean += floodout
    precip_mean[:, IH2O] += precip_rain
    runoff_mean[:, IH2O] += runoff
    drainage_mean[:, IH2O] += drainage
    temp_sol_mean += temp_sol * dt_scale

    if ok_doc:
        if nflow <= ISANDSED:
            raise ValueError("ok_doc routing accumulation requires nflow >= 10")
        if doc_exp_agg is None or doc_ero_agg is None or poc_exp_agg is None or sed_exp_agg is None:
            raise ValueError("ok_doc routing accumulation requires DOC/POC/SED aggregate inputs")
        doc_exp = np.asarray(doc_exp_agg, dtype=np.float64)
        doc_ero = np.asarray(doc_ero_agg, dtype=np.float64)
        poc_exp = np.asarray(poc_exp_agg, dtype=np.float64)
        sed_exp = np.asarray(sed_exp_agg, dtype=np.float64)
        if doc_exp.shape[0] != npts or doc_exp.shape[1] <= IDRAINAGE or doc_exp.shape[2] < nflow:
            raise ValueError("doc_exp_agg must have shape (npts,nexp,nflow) covering runoff/flooded/drainage")
        if doc_ero.shape != (npts, 3) or poc_exp.shape != (npts, 3) or sed_exp.shape != (npts, 3):
            raise ValueError("doc_ero_agg, poc_exp_agg, and sed_exp_agg must have three carbon/textural columns")
        for iflow in range(IH2O + 1, ICO2AQ + 1):
            drainage_mean[:, iflow] += doc_exp[:, IDRAINAGE, iflow] * 1.0e-3
            runoff_mean[:, iflow] += doc_exp[:, IRUNOFF, iflow] * 1.0e-3
            precip_mean[:, iflow] += doc_exp[:, IFLOODED, iflow] * 1.0e-3
        runoff_mean[:, IDOCL] += doc_ero[:, IACTIVE] * 1.0e-3
        runoff_mean[:, IDOCR] += (doc_ero[:, ISLOW] + doc_ero[:, IPASSIVE]) * 1.0e-3
        runoff_mean[:, IPOCA] += poc_exp[:, IACTIVE] * 1.0e-3
        runoff_mean[:, IPOCS] += poc_exp[:, ISLOW] * 1.0e-3
        runoff_mean[:, IPOCP] += poc_exp[:, IPASSIVE] * 1.0e-3
        runoff_mean[:, ICLAYSED] += sed_exp[:, IC_CLAY] * 1.0e-3
        runoff_mean[:, ISILTSED] += sed_exp[:, IC_SILT] * 1.0e-3
        runoff_mean[:, ISANDSED] += sed_exp[:, IC_SAND] * 1.0e-3
        precip_mean[:, IPOCA : ISANDSED + 1] = 0.0
        drainage_mean[:, IPOCA : ISANDSED + 1] = 0.0
        runoff_mean = np.nan_to_num(runoff_mean, nan=0.0)
        precip_mean = np.nan_to_num(precip_mean, nan=0.0)
        drainage_mean = np.nan_to_num(drainage_mean, nan=0.0)

    nonwoody_mask = np.ones((nvm,), dtype=bool)
    nonwoody_mask[int(ibare_sechiba)] = False
    nonwoody_mask &= ~is_tree
    tot_vegfrac_nowoody = np.sum(veget_max[:, nonwoody_mask], axis=1)
    for ig in range(npts):
        if tot_vegfrac_nowoody[ig] > min_sechiba:
            transpot_mean[ig] += np.sum(
                transpot[ig, nonwoody_mask] * veget_max[ig, nonwoody_mask] / tot_vegfrac_nowoody[ig]
            )
        else:
            pft_sum = np.sum(veget_max[ig, 1:nvm])
            if pft_sum > min_sechiba:
                transpot_mean[ig] += np.sum(transpot[ig, 1:nvm] * veget_max[ig, 1:nvm] / pft_sum)

    totnobio_mean += totfrac_nobio * dt_scale
    k_litt_mean += k_litt * dt_scale
    humrel_mean += np.sum(humrel[:, 1:nvm] * veget_max[:, 1:nvm], axis=1) * dt_scale
    vegtot_mean += np.sum(veget_max[:, 1:nvm], axis=1) * dt_scale

    time_counter = float(state.time_counter) + float(dt_sechiba)
    due = int(np.rint(time_counter)) >= int(np.rint(float(dt_routing)))
    zero_outputs = routing_zero_outputs(npts, nflow)
    flow_input = None
    flood_inp = None
    stream_inp = None
    if due:
        if flood_frac is None or streamfl_frac is None:
            raise ValueError("flood_frac and streamfl_frac are required when routing reaches dt_routing")
        flood_frac = np.asarray(flood_frac, dtype=np.float64)
        streamfl_frac = np.asarray(streamfl_frac, dtype=np.float64)
        if flood_frac.shape != (npts,) or streamfl_frac.shape != (npts,):
            raise ValueError("flood_frac and streamfl_frac must have shape (npts,)")
        flow_input = np.zeros((npts, 3, nflow), dtype=np.float64)
        flow_input[:, IRUNOFF, :] = runoff_mean * 1.0e3
        flow_input[:, IDRAINAGE, :] = drainage_mean * 1.0e3
        flow_input[:, IFLOODED, :] = precip_mean * 1.0e3
        denom = flood_frac + streamfl_frac
        flood_inp = np.zeros((npts, nflow), dtype=np.float64)
        stream_inp = np.zeros((npts, nflow), dtype=np.float64)
        active = denom > 0.0
        flood_inp[active, :] = precip_mean[active, :] * (flood_frac[active] / denom[active])[:, None]
        stream_inp[active, :] = precip_mean[active, :] - flood_inp[active, :]

    return RoutingAccumulatorStep(
        state=RoutingAccumulatorState(
            floodout_mean=floodout_mean,
            precip_mean=precip_mean,
            runoff_mean=runoff_mean,
            drainage_mean=drainage_mean,
            temp_sol_mean=temp_sol_mean,
            transpot_mean=transpot_mean,
            totnobio_mean=totnobio_mean,
            k_litt_mean=k_litt_mean,
            humrel_mean=humrel_mean,
            vegtot_mean=vegtot_mean,
            time_counter=time_counter,
        ),
        due=due,
        zero_outputs=zero_outputs,
        flow_input=flow_input,
        flood_inp=flood_inp,
        stream_inp=stream_inp,
    )


def routing_daily_boundary_step(
    *,
    accumulator_state: RoutingAccumulatorState,
    flow_state: RoutingFlowState,
    floodout,
    precip_rain,
    runoff,
    drainage,
    temp_sol,
    veget_max,
    transpot,
    totfrac_nobio,
    k_litt,
    humrel,
    is_tree,
    routing_area,
    topo_resid,
    route_togrid,
    route_tobasin,
    qflow_ave,
    stream_resave,
    basdrainarea,
    basgravel,
    bulkdens,
    zz_deep,
    floodtemp,
    reinf_slope,
    irrigated,
    stream_area,
    headw_area,
    streamr10th,
    streamr50th,
    streamr90th,
    floodplains,
    floodh90th,
    swamp,
    hydrodiag,
    mask_coast,
    nb_coast_gridcells,
    dt_sechiba,
    dt_routing,
    do_floodplains: bool,
    dofloodinfilt: bool,
    doponds: bool,
    doswamps: bool,
    dostreamswell: bool,
    new_flood_scheme: bool,
    do_irrigation: bool,
    ok_doc: bool,
    limit_rivdepos: bool = False,
    check_riverbal: bool = False,
    water_balance=None,
    carbon_balance=None,
    resolution=None,
    neighbours=None,
    max_lake_reservoir=7000.0,
    doc_exp_agg=None,
    doc_ero_agg=None,
    poc_exp_agg=None,
    sed_exp_agg=None,
    ibare_sechiba: int = 0,
    min_sechiba: float = 1.0e-8,
) -> RoutingDailyBoundaryResult:
    """Advance the ``routing_main`` daily gate around ``routing_flow``.

    Fortran provenance: ``routing.f90::routing_main`` lines 912-1140. The
    source accumulates sub-daily means, gates on ``time_counter >= dt_routing``,
    calls ``routing_flow``, then ``routing_lake``, adds lake returnflow,
    resets daily accumulators, and scales routing fluxes back to the SECHIBA
    timestep.
    """

    acc_step = routing_accumulator_step(
        state=accumulator_state,
        floodout=floodout,
        precip_rain=precip_rain,
        runoff=runoff,
        drainage=drainage,
        temp_sol=temp_sol,
        veget_max=veget_max,
        transpot=transpot,
        totfrac_nobio=totfrac_nobio,
        k_litt=k_litt,
        humrel=humrel,
        is_tree=is_tree,
        dt_sechiba=dt_sechiba,
        dt_routing=dt_routing,
        flood_frac=flow_state.flood_frac,
        streamfl_frac=flow_state.streamfl_frac,
        doc_exp_agg=doc_exp_agg,
        doc_ero_agg=doc_ero_agg,
        poc_exp_agg=poc_exp_agg,
        sed_exp_agg=sed_exp_agg,
        ok_doc=ok_doc,
        ibare_sechiba=ibare_sechiba,
        min_sechiba=min_sechiba,
    )

    if not acc_step.due:
        return RoutingDailyBoundaryResult(
            accumulator_step=acc_step,
            accumulator_state=acc_step.state,
            flow_state=flow_state,
            flow_step=None,
            lake_step=None,
            scaled_outputs=None,
            zero_outputs=acc_step.zero_outputs,
        )

    if acc_step.flood_inp is None or acc_step.stream_inp is None:
        raise ValueError("routing daily boundary reached due gate without flood/stream inputs")

    flow_step = routing_flow_step(
        state=flow_state,
        runoff=acc_step.state.runoff_mean,
        drainage=acc_step.state.drainage_mean,
        floodout=acc_step.state.floodout_mean,
        precip=acc_step.state.precip_mean,
        flood_inp=acc_step.flood_inp,
        stream_inp=acc_step.stream_inp,
        routing_area=routing_area,
        topo_resid=topo_resid,
        route_togrid=route_togrid,
        route_tobasin=route_tobasin,
        qflow_ave=qflow_ave,
        stream_resave=stream_resave,
        basdrainarea=basdrainarea,
        basgravel=basgravel,
        bulkdens=bulkdens,
        zz_deep=zz_deep,
        veget_max=veget_max,
        vegtot=acc_step.state.vegtot_mean,
        totnobio=acc_step.state.totnobio_mean,
        transpot_mean=acc_step.state.transpot_mean,
        humrel=acc_step.state.humrel_mean,
        k_litt=acc_step.state.k_litt_mean,
        floodtemp=floodtemp,
        temp_sol=acc_step.state.temp_sol_mean,
        reinf_slope=reinf_slope,
        irrigated=irrigated,
        stream_area=stream_area,
        headw_area=headw_area,
        streamr10th=streamr10th,
        streamr50th=streamr50th,
        streamr90th=streamr90th,
        floodplains=floodplains,
        floodh90th=floodh90th,
        swamp=swamp,
        hydrodiag=hydrodiag,
        mask_coast=mask_coast,
        nb_coast_gridcells=nb_coast_gridcells,
        dt_routing=dt_routing,
        do_floodplains=do_floodplains,
        dofloodinfilt=dofloodinfilt,
        doponds=doponds,
        doswamps=doswamps,
        dostreamswell=dostreamswell,
        new_flood_scheme=new_flood_scheme,
        do_irrigation=do_irrigation,
        ok_doc=ok_doc,
        limit_rivdepos=limit_rivdepos,
        check_riverbal=check_riverbal,
        water_balance=water_balance,
        carbon_balance=carbon_balance,
        resolution=resolution,
        neighbours=neighbours,
        max_lake_reservoir=max_lake_reservoir,
        min_sechiba=min_sechiba,
    )

    lake_step = routing_lake_step(
        lake_reservoir=flow_step.state.lake_reservoir,
        lakeinflow=flow_step.diagnostics.lakeinflow,
        routing_area=routing_area,
        humrel=acc_step.state.humrel_mean,
        dt_routing=dt_routing,
        doswamps=doswamps,
        ok_doc=ok_doc,
    )
    returnflow_mean = flow_step.return_reinfiltration.returnflow + lake_step.return_lakes
    scaled = routing_daily_scaled_outputs(
        returnflow_mean=returnflow_mean,
        reinfiltration_mean=flow_step.return_reinfiltration.reinfiltration,
        irrigation_mean=flow_step.diagnostics.irrigation,
        sed_deposition_mean=flow_step.diagnostics.sed_deposition,
        poc_deposition_mean=flow_step.diagnostics.poc_deposition,
        rivbed2fld_sed=flow_step.area_fractions.rivbed2fld_sed,
        rivbed2fld_poc=flow_step.area_fractions.rivbed2fld_poc,
        riverflow_mean=flow_step.diagnostics.riverflow,
        coastalflow_mean=flow_step.lake_overflow.coastalflow,
        hydrographs=flow_step.diagnostics.hydrographs,
        slowflow_diag=flow_step.diagnostics.slowflow_diag,
        dt_routing=dt_routing,
        dt_sechiba=dt_sechiba,
    )
    npts, nflow = acc_step.state.precip_mean.shape
    reset_acc = RoutingAccumulatorState(
        floodout_mean=np.zeros((npts,), dtype=np.float64),
        precip_mean=np.zeros((npts, nflow), dtype=np.float64),
        runoff_mean=np.zeros((npts, nflow), dtype=np.float64),
        drainage_mean=np.zeros((npts, nflow), dtype=np.float64),
        temp_sol_mean=np.zeros((npts,), dtype=np.float64),
        transpot_mean=np.zeros((npts,), dtype=np.float64),
        totnobio_mean=np.zeros((npts,), dtype=np.float64),
        k_litt_mean=np.zeros((npts,), dtype=np.float64),
        humrel_mean=np.zeros((npts,), dtype=np.float64),
        vegtot_mean=np.zeros((npts,), dtype=np.float64),
        time_counter=0.0,
    )
    next_flow_state = flow_step.state._replace(
        lake_reservoir=lake_step.lake_reservoir,
        lake_diag=lake_step.lake_diag,
    )
    return RoutingDailyBoundaryResult(
        accumulator_step=acc_step,
        accumulator_state=reset_acc,
        flow_state=next_flow_state,
        flow_step=flow_step,
        lake_step=lake_step,
        scaled_outputs=scaled,
        zero_outputs=acc_step.zero_outputs,
    )
