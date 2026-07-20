from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.routing import (  # noqa: E402
    ICLAYSED,
    IACTIVE,
    IC_CLAY,
    IC_SAND,
    IC_SILT,
    ICO2AQ,
    IDOCL,
    IDOCLABILE,
    IDOCR,
    IDOCSTABLE,
    IDRAINAGE,
    IFASTR,
    IFLOODED,
    IFLOODR,
    IH2O,
    IPASSIVE,
    IPONDR,
    IPOCA,
    IPOCP,
    IPOCS,
    RoutingMapFields,
    IRUNOFF,
    ISANDSED,
    ISILTSED,
    ISLOW,
    ISLOWR,
    ISTREAMR,
    RoutingAccumulatorState,
    RoutingFlowState,
    read_routing_map_fields,
    routing_map_fields_for_domain,
    routing_accumulator_step,
    routing_area_fractions_step,
    routing_basins_post_aggregate,
    routing_basins_from_map_fields,
    routing_co2_chemistry_step,
    routing_cutbasin,
    routing_daily_scaled_outputs,
    routing_daily_boundary_step,
    routing_fetch,
    routing_findbasins_simple,
    routing_findrout,
    routing_flow_step,
    routing_flood_pond_input_step,
    routing_floodplain_flux_step,
    routing_flow_diagnostics_step,
    routing_getgrid_from_subgrid,
    routing_globalize_one_grid,
    routing_hierarchy,
    routing_irrigation_step,
    routing_initialize_map_flags,
    routing_initialize_stream_fraction,
    routing_init_restart_defaults,
    routing_irrigmap_aggregate,
    routing_irrigmap_preprocess,
    routing_killbas,
    routing_lake_step,
    routing_lake_overflow_step,
    routing_linkup,
    routing_poc_decomposition,
    routing_poc_fraction_matrix,
    routing_pond_flux_step,
    routing_reservoir_outflow_step,
    routing_reservoir_update_step,
    routing_return_reinfiltration_step,
    routing_setvar_no_keyword,
    routing_simplify,
    routing_sortcoord,
    routing_swamp_flood_step,
    routing_stream_erosion_step,
    routing_transport_between_basins_step,
    routing_truncate_finalize_no_reduction,
    routing_truncate_reduce_to_nbasmax,
)


def _state(npts=2, nflow=10):
    return RoutingAccumulatorState(
        floodout_mean=np.zeros(npts, dtype=np.float64),
        precip_mean=np.zeros((npts, nflow), dtype=np.float64),
        runoff_mean=np.zeros((npts, nflow), dtype=np.float64),
        drainage_mean=np.zeros((npts, nflow), dtype=np.float64),
        temp_sol_mean=np.zeros(npts, dtype=np.float64),
        transpot_mean=np.zeros(npts, dtype=np.float64),
        totnobio_mean=np.zeros(npts, dtype=np.float64),
        k_litt_mean=np.zeros(npts, dtype=np.float64),
        humrel_mean=np.zeros(npts, dtype=np.float64),
        vegtot_mean=np.zeros(npts, dtype=np.float64),
        time_counter=0.0,
    )


def test_routing_accumulator_before_daily_gate_updates_means_and_keeps_outputs_zero():
    veget_max = np.asarray(
        [
            [0.0, 0.20, 0.30],
            [0.0, 0.00, 0.50],
        ],
        dtype=np.float64,
    )
    transpot = np.asarray(
        [
            [0.0, 2.0, 4.0],
            [0.0, 8.0, 6.0],
        ],
        dtype=np.float64,
    )
    humrel = np.asarray(
        [
            [0.0, 0.5, 0.7],
            [0.0, 0.2, 0.9],
        ],
        dtype=np.float64,
    )

    step = routing_accumulator_step(
        state=_state(),
        floodout=np.asarray([0.01, 0.02], dtype=np.float64),
        precip_rain=np.asarray([1.0, 2.0], dtype=np.float64),
        runoff=np.asarray([0.3, 0.4], dtype=np.float64),
        drainage=np.asarray([0.5, 0.6], dtype=np.float64),
        temp_sol=np.asarray([290.0, 291.0], dtype=np.float64),
        veget_max=veget_max,
        transpot=transpot,
        totfrac_nobio=np.asarray([0.1, 0.2], dtype=np.float64),
        k_litt=np.asarray([0.01, 0.02], dtype=np.float64),
        humrel=humrel,
        is_tree=np.asarray([False, False, False]),
        dt_sechiba=1800.0,
        dt_routing=86400.0,
    )

    assert not step.due
    assert step.flow_input is None
    for value in step.zero_outputs.values():
        np.testing.assert_allclose(value, 0.0)
    np.testing.assert_allclose(step.state.floodout_mean, [0.01, 0.02])
    np.testing.assert_allclose(step.state.precip_mean[:, IH2O], [1.0, 2.0])
    np.testing.assert_allclose(step.state.runoff_mean[:, IH2O], [0.3, 0.4])
    np.testing.assert_allclose(step.state.drainage_mean[:, IH2O], [0.5, 0.6])
    np.testing.assert_allclose(step.state.temp_sol_mean, np.asarray([290.0, 291.0]) * 1800.0 / 86400.0)
    np.testing.assert_allclose(step.state.transpot_mean, [(2.0 * 0.20 + 4.0 * 0.30) / 0.50, 6.0])
    np.testing.assert_allclose(step.state.totnobio_mean, np.asarray([0.1, 0.2]) * 1800.0 / 86400.0)
    np.testing.assert_allclose(step.state.k_litt_mean, np.asarray([0.01, 0.02]) * 1800.0 / 86400.0)
    np.testing.assert_allclose(
        step.state.humrel_mean,
        np.asarray([0.5 * 0.20 + 0.7 * 0.30, 0.9 * 0.50]) * 1800.0 / 86400.0,
    )
    np.testing.assert_allclose(step.state.vegtot_mean, np.asarray([0.50, 0.50]) * 1800.0 / 86400.0)


def test_routing_accumulator_daily_gate_builds_flow_input_and_doc_splits():
    npts = 1
    nflow = 10
    doc_exp = np.zeros((npts, 3, nflow), dtype=np.float64)
    doc_exp[0, IRUNOFF, IDOCL] = 2.0
    doc_exp[0, IRUNOFF, IDOCR] = 3.0
    doc_exp[0, IDRAINAGE, ICO2AQ] = 4.0
    doc_exp[0, IFLOODED, IDOCL] = 5.0
    doc_ero = np.asarray([[7.0, 11.0, 13.0]], dtype=np.float64)
    poc_exp = np.asarray([[17.0, 19.0, 23.0]], dtype=np.float64)
    sed_exp = np.asarray([[29.0, 31.0, 37.0]], dtype=np.float64)

    step = routing_accumulator_step(
        state=_state(npts=npts, nflow=nflow),
        floodout=np.asarray([0.0], dtype=np.float64),
        precip_rain=np.asarray([1.0], dtype=np.float64),
        runoff=np.asarray([0.2], dtype=np.float64),
        drainage=np.asarray([0.3], dtype=np.float64),
        temp_sol=np.asarray([280.0], dtype=np.float64),
        veget_max=np.asarray([[0.0, 0.4]], dtype=np.float64),
        transpot=np.asarray([[0.0, 1.0]], dtype=np.float64),
        totfrac_nobio=np.asarray([0.0], dtype=np.float64),
        k_litt=np.asarray([0.0], dtype=np.float64),
        humrel=np.asarray([[0.0, 0.5]], dtype=np.float64),
        is_tree=np.asarray([False, False]),
        dt_sechiba=86400.0,
        dt_routing=86400.0,
        flood_frac=np.asarray([0.25], dtype=np.float64),
        streamfl_frac=np.asarray([0.75], dtype=np.float64),
        doc_exp_agg=doc_exp,
        doc_ero_agg=doc_ero,
        poc_exp_agg=poc_exp,
        sed_exp_agg=sed_exp,
        ok_doc=True,
    )

    assert step.due
    assert step.flow_input is not None
    np.testing.assert_allclose(step.flow_input[0, IRUNOFF, IH2O], 0.2 * 1.0e3)
    np.testing.assert_allclose(step.flow_input[0, IDRAINAGE, IH2O], 0.3 * 1.0e3)
    np.testing.assert_allclose(step.flow_input[0, IFLOODED, IH2O], 1.0 * 1.0e3)
    np.testing.assert_allclose(step.flow_input[0, IRUNOFF, IDOCL], (2.0 + 7.0) * 1.0e-3 * 1.0e3)
    np.testing.assert_allclose(step.flow_input[0, IRUNOFF, IDOCR], (3.0 + 11.0 + 13.0) * 1.0e-3 * 1.0e3)
    np.testing.assert_allclose(step.flow_input[0, IDRAINAGE, ICO2AQ], 4.0)
    np.testing.assert_allclose(step.flow_input[0, IRUNOFF, IPOCA], 17.0)
    np.testing.assert_allclose(step.flow_input[0, IRUNOFF, IPOCS], 19.0)
    np.testing.assert_allclose(step.flow_input[0, IRUNOFF, IPOCP], 23.0)
    np.testing.assert_allclose(step.flow_input[0, IRUNOFF, ICLAYSED], 29.0)
    np.testing.assert_allclose(step.flow_input[0, IRUNOFF, ISILTSED], 31.0)
    np.testing.assert_allclose(step.flow_input[0, IRUNOFF, ISANDSED], 37.0)
    np.testing.assert_allclose(step.flow_input[0, IFLOODED, IPOCA : ISANDSED + 1], 0.0)
    np.testing.assert_allclose(step.flow_input[0, IDRAINAGE, IPOCA : ISANDSED + 1], 0.0)
    np.testing.assert_allclose(step.flood_inp[0, IH2O], 0.25)
    np.testing.assert_allclose(step.stream_inp[0, IH2O], 0.75)
    np.testing.assert_allclose(step.flood_inp[0, IDOCL], 5.0e-3 * 0.25)
    np.testing.assert_allclose(step.stream_inp[0, IDOCL], 5.0e-3 * 0.75)


def test_routing_accumulator_ok_doc_refuses_short_flow_axis():
    with pytest.raises(ValueError, match="nflow >= 10"):
        routing_accumulator_step(
            state=_state(npts=1, nflow=4),
            floodout=np.zeros(1, dtype=np.float64),
            precip_rain=np.zeros(1, dtype=np.float64),
            runoff=np.zeros(1, dtype=np.float64),
            drainage=np.zeros(1, dtype=np.float64),
            temp_sol=np.zeros(1, dtype=np.float64),
            veget_max=np.zeros((1, 2), dtype=np.float64),
            transpot=np.zeros((1, 2), dtype=np.float64),
            totfrac_nobio=np.zeros(1, dtype=np.float64),
            k_litt=np.zeros(1, dtype=np.float64),
            humrel=np.zeros((1, 2), dtype=np.float64),
            is_tree=np.asarray([False, False]),
            dt_sechiba=1800.0,
            dt_routing=86400.0,
            ok_doc=True,
        )


def test_routing_lake_step_returns_water_and_doc_by_lake_concentration():
    lake_reservoir = np.asarray([[100.0, 10.0, 20.0, 30.0]], dtype=np.float64)
    lakeinflow = np.asarray([[20.0, 2.0, 4.0, 6.0]], dtype=np.float64)
    routing_area = np.asarray([[1000.0, 500.0]], dtype=np.float64)
    humrel = np.asarray([0.50], dtype=np.float64)

    result = routing_lake_step(
        lake_reservoir=lake_reservoir,
        lakeinflow=lakeinflow,
        routing_area=routing_area,
        humrel=humrel,
        dt_routing=86400.0,
        doswamps=True,
        ok_doc=True,
        maxevap_lake=7.5 / 86400.0,
    )

    total_area = routing_area.sum(axis=1)[0]
    reservoir_after_inflow = lake_reservoir + lakeinflow
    refill = max(0.0, (7.5 / 86400.0) * (1.0 - humrel[0]) * 86400.0 * total_area)
    returned_water = min(refill, reservoir_after_inflow[0, IH2O])
    expected_return = np.zeros_like(lake_reservoir)
    expected_return[0, IH2O] = returned_water
    expected_return[0, IH2O + 1 :] = returned_water * reservoir_after_inflow[0, IH2O + 1 :] / reservoir_after_inflow[0, IH2O]
    expected_reservoir = reservoir_after_inflow - expected_return

    np.testing.assert_allclose(result.return_lakes, expected_return / total_area)
    np.testing.assert_allclose(result.lake_reservoir, expected_reservoir)
    np.testing.assert_allclose(result.lake_diag, expected_reservoir / total_area)
    np.testing.assert_allclose(result.lakeinflow, lakeinflow / total_area)


def test_routing_lake_step_without_swamps_only_updates_reservoir_and_scaled_diagnostics():
    lake_reservoir = np.asarray([[5.0, 1.0]], dtype=np.float64)
    lakeinflow = np.asarray([[2.0, 0.5]], dtype=np.float64)
    routing_area = np.asarray([[10.0]], dtype=np.float64)

    result = routing_lake_step(
        lake_reservoir=lake_reservoir,
        lakeinflow=lakeinflow,
        routing_area=routing_area,
        humrel=np.asarray([0.0], dtype=np.float64),
        dt_routing=86400.0,
        doswamps=False,
        ok_doc=True,
    )

    np.testing.assert_allclose(result.return_lakes, [[0.0, 0.0]])
    np.testing.assert_allclose(result.lake_reservoir, lake_reservoir + lakeinflow)
    np.testing.assert_allclose(result.lake_diag, (lake_reservoir + lakeinflow) / 10.0)
    np.testing.assert_allclose(result.lakeinflow, lakeinflow / 10.0)


def test_routing_reservoir_outflow_step_matches_source_water_and_concentration_rules():
    fast = np.zeros((1, 2, 5), dtype=np.float64)
    slow = np.zeros_like(fast)
    stream = np.zeros_like(fast)
    fast[0, 0, :] = [100.0, 10.0, 20.0, 30.0, 40.0]
    slow[0, 0, :] = [80.0, 8.0, 16.0, 24.0, 32.0]
    stream[0, 0, :] = [50.0, 5.0, 10.0, 15.0, 20.0]
    stream[0, 1, IH2O] = 50.0
    topo = np.asarray([[2000.0, 1000.0]], dtype=np.float64)
    route_tobasin = np.asarray([[1, 1]], dtype=np.int32)

    result = routing_reservoir_outflow_step(
        fast_reservoir=fast,
        slow_reservoir=slow,
        stream_reservoir=stream,
        topo_resid=topo,
        route_tobasin=route_tobasin,
        qflow_ave=np.asarray([12.0], dtype=np.float64),
        stream_resave=np.asarray([6.0], dtype=np.float64),
        stream_damavail=np.asarray([4.0], dtype=np.float64),
        dt_routing=86400.0,
        fast_tcst=3.0,
        slow_tcst=25.0,
        stream_tcst=0.24,
        flow_coef=1.01,
    )

    expected_qflow_share = 12.0 * 0.5
    expected_resave_share = 6.0 * 0.5
    expected_dam_share = 4.0 * 0.5
    np.testing.assert_allclose(result.qflow_avebas[0, 0], expected_qflow_share)
    np.testing.assert_allclose(result.stream_resavebas[0, 0], expected_resave_share)
    np.testing.assert_allclose(result.stream_damavailbas[0, 0], expected_dam_share)

    expected_fast_water = min(100.0**1.01 / ((2000.0 / 1000.0) * 3.0), 100.0 - 1.0e-8)
    expected_slow_water = min(80.0**1.01 / ((2000.0 / 1000.0) * 25.0), 80.0 - 1.0e-8)
    stream_factor = expected_resave_share / (expected_dam_share + expected_resave_share)
    expected_stream_water = min(50.0**1.01 / ((2000.0 / 1000.0) * 0.24 * stream_factor), 50.0 - 1.0e-8)
    np.testing.assert_allclose(result.fast_flow[0, 0, IH2O], expected_fast_water)
    np.testing.assert_allclose(result.slow_flow[0, 0, IH2O], expected_slow_water)
    np.testing.assert_allclose(result.stream_flow[0, 0, IH2O], expected_stream_water)
    np.testing.assert_allclose(result.fast_flow[0, 0, IDOCL], 10.0 * expected_fast_water / 100.0)
    np.testing.assert_allclose(result.slow_flow[0, 0, IDOCR], 16.0 * expected_slow_water / 80.0)
    np.testing.assert_allclose(result.stream_flow[0, 0, ICO2AQ], 15.0 * expected_stream_water / 50.0)
    np.testing.assert_allclose(result.stream_flow[0, 0, IPOCA], 0.0)
    np.testing.assert_allclose(result.fast_reservoir[0, 0, IDOCL], 10.0 - result.fast_flow[0, 0, IDOCL])
    np.testing.assert_allclose(result.slow_reservoir[0, 0, IDOCR], 16.0 - result.slow_flow[0, 0, IDOCR])
    np.testing.assert_allclose(result.stream_reservoir[0, 0, ICO2AQ], 15.0 - result.stream_flow[0, 0, ICO2AQ])


def test_routing_reservoir_outflow_step_zeroes_inactive_route_basins():
    fast = np.asarray([[[10.0, 1.0]]], dtype=np.float64)
    slow = np.asarray([[[20.0, 2.0]]], dtype=np.float64)
    stream = np.asarray([[[30.0, 3.0]]], dtype=np.float64)

    result = routing_reservoir_outflow_step(
        fast_reservoir=fast,
        slow_reservoir=slow,
        stream_reservoir=stream,
        topo_resid=np.asarray([[1000.0]], dtype=np.float64),
        route_tobasin=np.asarray([[0]], dtype=np.int32),
        qflow_ave=np.asarray([0.0], dtype=np.float64),
        stream_resave=np.asarray([0.0], dtype=np.float64),
        stream_damavail=np.asarray([0.0], dtype=np.float64),
        dt_routing=86400.0,
    )

    np.testing.assert_allclose(result.fast_flow, 0.0)
    np.testing.assert_allclose(result.slow_flow, 0.0)
    np.testing.assert_allclose(result.stream_flow, 0.0)
    np.testing.assert_allclose(result.fast_reservoir, fast)
    np.testing.assert_allclose(result.slow_reservoir, slow)
    np.testing.assert_allclose(result.stream_reservoir, stream)


def test_routing_flood_pond_input_step_applies_vertical_flux_and_basin_inputs():
    nflow = 10
    flood = np.zeros((1, 2, nflow), dtype=np.float64)
    flood[0, 0, IH2O] = 40.0
    flood[0, 1, IH2O] = 60.0
    flood[0, 0, IPOCA : ISANDSED + 1] = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    pond = np.zeros((1, nflow), dtype=np.float64)
    pond[0, IH2O] = 10.0
    routing_area = np.asarray([[100.0, 300.0]], dtype=np.float64)
    flood_frac = np.asarray([0.4], dtype=np.float64)
    flood_frac_bas = np.asarray([[0.1, 0.1]], dtype=np.float64)
    pond_frac = np.asarray([0.1], dtype=np.float64)
    flood_inp = np.zeros((1, nflow), dtype=np.float64)
    stream_inp = np.zeros((1, nflow), dtype=np.float64)
    flood_inp[0, IDOCL] = 0.2
    stream_inp[0, IDOCR] = 0.3

    result = routing_flood_pond_input_step(
        flood_reservoir=flood,
        pond_reservoir=pond,
        floodout=np.asarray([0.04], dtype=np.float64),
        flood_inp=flood_inp,
        stream_inp=stream_inp,
        routing_area=routing_area,
        flood_frac=flood_frac,
        flood_frac_bas=flood_frac_bas,
        pond_frac=pond_frac,
        streamfl_frac=np.asarray([0.2], dtype=np.float64),
        streamfl_frac_bas=np.asarray([[0.05, 0.05]], dtype=np.float64),
        do_floodplains=True,
        doponds=False,
        dostreamswell=True,
    )

    # Fortran lines 3597-3624 remove pond water first, then floodplain water
    # proportional to basin flood fractions.
    np.testing.assert_allclose(result.pond_reservoir[0, IH2O], 6.0)
    np.testing.assert_allclose(result.flood_reservoir[0, :, IH2O], [39.0, 57.0])
    np.testing.assert_allclose(result.flood_dep_poc[0, 0, :], [0.1, 0.125, 0.15])
    np.testing.assert_allclose(result.flood_dep_sed[0, 0, :], [0.175, 0.2, 0.225])
    np.testing.assert_allclose(result.flood_inp_bas[0, :, IDOCL], [5.0, 15.0])
    np.testing.assert_allclose(result.stream_inp_bas[0, :, IDOCR], [7.5, 22.5])


def test_routing_daily_scaled_outputs_match_fortran_unit_conversion():
    result = routing_daily_scaled_outputs(
        returnflow_mean=np.asarray([[10.0, 20.0]], dtype=np.float64),
        reinfiltration_mean=np.asarray([[30.0, 40.0]], dtype=np.float64),
        irrigation_mean=np.asarray([[50.0, 60.0]], dtype=np.float64),
        sed_deposition_mean=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        poc_deposition_mean=np.asarray([[4.0, 5.0, 6.0]], dtype=np.float64),
        rivbed2fld_sed=np.asarray([[0.5, 0.6, 0.7]], dtype=np.float64),
        rivbed2fld_poc=np.asarray([[0.8, 0.9, 1.0]], dtype=np.float64),
        riverflow_mean=np.asarray([[7000.0, 8000.0]], dtype=np.float64),
        coastalflow_mean=np.asarray([[9000.0, 10000.0]], dtype=np.float64),
        hydrographs=np.asarray([[11000.0, 12000.0]], dtype=np.float64),
        slowflow_diag=np.asarray([13000.0], dtype=np.float64),
        dt_routing=86400.0,
        dt_sechiba=1800.0,
    )

    scale = 1800.0 / 86400.0
    np.testing.assert_allclose(result.returnflow, [[10.0 * scale, 20.0 * scale]])
    np.testing.assert_allclose(result.reinfiltration, [[30.0 * scale, 40.0 * scale]])
    np.testing.assert_allclose(result.irrigation, [[50.0 * scale, 60.0 * scale]])
    np.testing.assert_allclose(result.sed_depositiontot, np.asarray([[1.5, 2.6, 3.7]]) * scale)
    np.testing.assert_allclose(result.poc_depositiontot, np.asarray([[4.8, 5.9, 7.0]]) * scale)
    np.testing.assert_allclose(result.riverflow, np.asarray([[7000.0, 8000.0]]) * scale / 1000.0)
    np.testing.assert_allclose(result.coastalflow, np.asarray([[9000.0, 10000.0]]) * scale / 1000.0)
    np.testing.assert_allclose(result.hydrographs, np.asarray([[11000.0, 12000.0]]) * scale / 1000.0)
    np.testing.assert_allclose(result.slowflow_diag, [13000.0 * scale])


def test_routing_return_reinfiltration_step_sums_only_water_doc_co2_slots():
    return_swamp = np.zeros((1, 2, 6), dtype=np.float64)
    pond_drainage = np.zeros_like(return_swamp)
    flood_drainage = np.zeros_like(return_swamp)
    return_swamp[0, 0, :6] = [10.0, 1.0, 2.0, 3.0, 99.0, 88.0]
    return_swamp[0, 1, :6] = [20.0, 4.0, 5.0, 6.0, 77.0, 66.0]
    pond_drainage[0, 0, :6] = [1.0, 0.1, 0.2, 0.3, 9.0, 8.0]
    flood_drainage[0, 1, :6] = [2.0, 0.4, 0.5, 0.6, 7.0, 6.0]

    result = routing_return_reinfiltration_step(
        return_swamp=return_swamp,
        pond_drainage=pond_drainage,
        flood_drainage=flood_drainage,
        routing_area=np.asarray([[100.0, 200.0]], dtype=np.float64),
        do_floodplains=True,
        doswamps=False,
        doponds=False,
    )

    total_area = 300.0
    np.testing.assert_allclose(result.returnflow[0, : ICO2AQ + 1], [30.0 / total_area, 5.0 / total_area, 7.0 / total_area, 9.0 / total_area])
    np.testing.assert_allclose(
        result.reinfiltration[0, : ICO2AQ + 1],
        [3.0 / total_area, 0.5 / total_area, 0.7 / total_area, 0.9 / total_area],
    )
    np.testing.assert_allclose(result.returnflow[0, IPOCA:], 0.0)
    np.testing.assert_allclose(result.reinfiltration[0, IPOCA:], 0.0)


def test_routing_return_reinfiltration_step_zero_when_all_surface_switches_disabled():
    result = routing_return_reinfiltration_step(
        return_swamp=np.ones((1, 1, 4), dtype=np.float64),
        pond_drainage=np.ones((1, 1, 4), dtype=np.float64),
        flood_drainage=np.ones((1, 1, 4), dtype=np.float64),
        routing_area=np.asarray([[10.0]], dtype=np.float64),
        do_floodplains=False,
        doswamps=False,
        doponds=False,
    )

    np.testing.assert_allclose(result.returnflow, 0.0)
    np.testing.assert_allclose(result.reinfiltration, 0.0)


def test_routing_flow_diagnostics_step_matches_fortran_gridcell_aggregation():
    npts, nbas, nflow = 1, 2, 10
    routing_area = np.asarray([[100.0, 300.0]], dtype=np.float64)
    runoff = np.full((npts, nflow), 2.0, dtype=np.float64)
    drainage = np.full((npts, nflow), 3.0, dtype=np.float64)
    fast_flow = np.full((npts, nbas, nflow), 10.0, dtype=np.float64)
    slow_flow = np.full((npts, nbas, nflow), 20.0, dtype=np.float64)
    stream_flow = np.full((npts, nbas, nflow), 30.0, dtype=np.float64)
    flood_flow = np.full((npts, nbas, nflow), 40.0, dtype=np.float64)
    pond_inflow = np.full((npts, nbas, nflow), 5.0, dtype=np.float64)
    transport = np.zeros((npts, nbas + 3, nflow), dtype=np.float64)
    transport[:, :nbas, :] = 50.0
    transport[:, nbas, :] = 60.0
    transport[:, nbas + 1, :] = 70.0
    transport[:, nbas + 2, :] = 80.0
    return_swamp = np.full((npts, nbas, nflow), 6.0, dtype=np.float64)
    floods = np.full((npts, nbas, nflow), 7.0, dtype=np.float64)
    hydrodiag = np.zeros((npts, nbas, nflow), dtype=np.int32)
    hydrodiag[0, 0, IH2O] = 1

    fast_reservoir = np.zeros((npts, nbas, nflow), dtype=np.float64)
    slow_reservoir = np.zeros_like(fast_reservoir)
    stream_reservoir = np.zeros_like(fast_reservoir)
    flood_reservoir = np.zeros_like(fast_reservoir)
    fast_reservoir[0, :, IH2O] = [100.0, 300.0]
    slow_reservoir[0, :, IH2O] = [50.0, 150.0]
    stream_reservoir[0, :, IDOCL] = [8.0, 12.0]
    flood_reservoir[0, :, IH2O] = [20.0, 60.0]
    pond_reservoir = np.zeros((npts, nflow), dtype=np.float64)
    pond_reservoir[0, IH2O] = 120.0

    irrig_actual = np.full((npts, nbas, nflow), 4.0, dtype=np.float64)
    irrig_adduct = np.full((npts, nbas, nflow), 1.0, dtype=np.float64)
    flood_dep_sed = np.asarray([[[4000.0, -4.0, 8000.0], [0.0, 0.0, 0.0]]], dtype=np.float64)
    flood_dep_poc = np.asarray([[[-1.0, 2000.0, 8000.0], [0.0, 0.0, 0.0]]], dtype=np.float64)
    stream_erodep = np.zeros((npts, nbas, nflow), dtype=np.float64)
    stream_erodep[0, :, IPOCA : IPOCP + 1] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    stream_erodep[0, :, ICLAYSED : ISANDSED + 1] = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    stream_seddep = np.full((npts, nbas, nflow), 0.5, dtype=np.float64)
    streamb_inflow = np.full((npts, nbas, nflow), 9.0, dtype=np.float64)

    result = routing_flow_diagnostics_step(
        runoff=runoff,
        drainage=drainage,
        routing_area=routing_area,
        fast_flow=fast_flow,
        slow_flow=slow_flow,
        stream_flow=stream_flow,
        flood_flow=flood_flow,
        pond_inflow=pond_inflow,
        transport=transport,
        return_swamp=return_swamp,
        floods=floods,
        previous_flood_diag=np.asarray([[2.0] + [0.0] * 9], dtype=np.float64),
        previous_pond_diag=np.asarray([[3.0] + [0.0] * 9], dtype=np.float64),
        lake_diag=np.asarray([[4.0] + [0.0] * 9], dtype=np.float64),
        hydrodiag=hydrodiag,
        fast_reservoir=fast_reservoir,
        slow_reservoir=slow_reservoir,
        stream_reservoir=stream_reservoir,
        flood_reservoir=flood_reservoir,
        pond_reservoir=pond_reservoir,
        irrig_actual=irrig_actual,
        irrig_adduct=irrig_adduct,
        flood_dep_sed=flood_dep_sed,
        flood_dep_poc=flood_dep_poc,
        stream_erodep=stream_erodep,
        stream_seddep=stream_seddep,
        streamb_inflow=streamb_inflow,
        poc_co2_rivbed=np.asarray([8.0], dtype=np.float64),
        poc_doc_rivbed=np.asarray([[2.0, 6.0]], dtype=np.float64),
        undef_sechiba=10.0,
    )

    total_area = 400.0
    np.testing.assert_allclose(result.delsurfstor, [-9.0])
    np.testing.assert_allclose(result.netflow_fast_diag, np.full((1, nflow), (2.0 * total_area - 2.0 * 10.0 - 2.0 * 5.0) / total_area))
    np.testing.assert_allclose(result.netflow_slow_diag, np.full((1, nflow), (3.0 * total_area - 2.0 * 20.0) / total_area))
    np.testing.assert_allclose(result.netflow_stream_diag, np.full((1, nflow), 2.0 * (40.0 + 50.0 - 30.0 - 6.0 - 7.0) / total_area))
    np.testing.assert_allclose(result.hydrographs, np.full((1, nflow), 60.0))
    np.testing.assert_allclose(result.slowflow_diag, [20.0])
    np.testing.assert_allclose(result.fast_diag[0, IH2O], 1.0)
    np.testing.assert_allclose(result.slow_diag[0, IH2O], 0.5)
    np.testing.assert_allclose(result.stream_diag[0, IDOCL], 0.05)
    np.testing.assert_allclose(result.flood_diag[0, IH2O], 0.2)
    np.testing.assert_allclose(result.pond_diag[0, IH2O], 0.3)
    np.testing.assert_allclose(result.flood_res, [0.5])
    np.testing.assert_allclose(result.fastr, [1.5])
    np.testing.assert_allclose(result.irrigation, np.full((1, nflow), 10.0 / total_area))
    np.testing.assert_allclose(result.sed_deposition, [[10.0, 0.0, 10.0]])
    np.testing.assert_allclose(result.poc_deposition, [[0.0, 5.0, 10.0]])
    np.testing.assert_allclose(result.riv_erodep_poc, [[5.0, 7.0, 9.0]])
    np.testing.assert_allclose(result.riv_erodep_sed, [[17.0, 19.0, 21.0]])
    np.testing.assert_allclose(result.rivchannel_deposition, np.full((1, nflow), 1.0))
    np.testing.assert_allclose(result.stream_inflow, np.full((1, nflow), 18.0))
    np.testing.assert_allclose(result.stream_outflow, np.full((1, nflow), 60.0))
    np.testing.assert_allclose(result.flood_daily, [14.0])
    np.testing.assert_allclose(result.lakeinflow, np.full((1, nflow), 60.0))
    np.testing.assert_allclose(result.coastalflow, np.full((1, nflow), 70.0))
    np.testing.assert_allclose(result.riverflow, np.full((1, nflow), 80.0))
    np.testing.assert_allclose(result.poc_co2_rivbed, [10.0])
    np.testing.assert_allclose(result.poc_doc_rivbed, [[5.0, 10.0]])


def test_routing_lake_overflow_step_caps_lakes_and_adds_global_overflow_to_coast():
    lake_reservoir = np.asarray(
        [
            [70.0, 7.0, 14.0, 21.0],
            [80.0, 8.0, 16.0, 24.0],
        ],
        dtype=np.float64,
    )
    coastalflow = np.zeros_like(lake_reservoir)

    result = routing_lake_overflow_step(
        lake_reservoir=lake_reservoir,
        coastalflow=coastalflow,
        routing_area=np.asarray([[10.0], [20.0]], dtype=np.float64),
        mask_coast=np.asarray([1.0, 0.0], dtype=np.float64),
        nb_coast_gridcells=1,
        max_lake_reservoir=5.0,
    )

    expected_overflow = np.asarray([[20.0, 2.0, 4.0, 6.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(result.lake_overflow, expected_overflow)
    np.testing.assert_allclose(result.lake_reservoir, lake_reservoir - expected_overflow)
    np.testing.assert_allclose(result.total_lake_overflow, expected_overflow.sum(axis=0))
    np.testing.assert_allclose(result.lake_overflow_coast, expected_overflow)
    np.testing.assert_allclose(result.coastalflow, expected_overflow)


def test_routing_lake_overflow_step_keeps_coastalflow_when_no_coastal_cell():
    result = routing_lake_overflow_step(
        lake_reservoir=np.asarray([[20.0, 2.0]], dtype=np.float64),
        coastalflow=np.asarray([[5.0, 6.0]], dtype=np.float64),
        routing_area=np.asarray([[1.0]], dtype=np.float64),
        mask_coast=np.asarray([0.0], dtype=np.float64),
        nb_coast_gridcells=0,
        max_lake_reservoir=10.0,
    )

    np.testing.assert_allclose(result.lake_overflow, [[10.0, 1.0]])
    np.testing.assert_allclose(result.lake_reservoir, [[10.0, 1.0]])
    np.testing.assert_allclose(result.lake_overflow_coast, [[0.0, 0.0]])
    np.testing.assert_allclose(result.coastalflow, [[5.0, 6.0]])


def test_routing_reservoir_update_step_applies_source_writeback_and_totflood():
    nflow = 10
    fast = np.full((1, 1, nflow), 10.0, dtype=np.float64)
    slow = np.full((1, 1, nflow), 5.0, dtype=np.float64)
    stream = np.full((1, 1, nflow), 20.0, dtype=np.float64)
    flood = np.full((1, 1, nflow), 30.0, dtype=np.float64)
    pond = np.full((1, nflow), 40.0, dtype=np.float64)
    return_swamp = np.full((1, 1, nflow), 1.0, dtype=np.float64)
    floods = np.full((1, 1, nflow), 2.0, dtype=np.float64)
    flood_flow = np.full((1, 1, nflow), 3.0, dtype=np.float64)
    pond_inflow = np.full((1, 1, nflow), 4.0, dtype=np.float64)
    pond_drainage = np.full((1, 1, nflow), 6.0, dtype=np.float64)

    result = routing_reservoir_update_step(
        runoff=np.full((1, nflow), 2.0, dtype=np.float64),
        drainage=np.full((1, nflow), 3.0, dtype=np.float64),
        routing_area=np.asarray([[100.0]], dtype=np.float64),
        fast_reservoir=fast,
        slow_reservoir=slow,
        stream_reservoir=stream,
        flood_reservoir=flood,
        pond_reservoir=pond,
        transport=np.full((1, 1, nflow), 7.0, dtype=np.float64),
        return_swamp=return_swamp,
        floods=floods,
        flood_flow=flood_flow,
        pond_inflow=pond_inflow,
        pond_drainage=pond_drainage,
    )

    np.testing.assert_allclose(result.fast_reservoir, 210.0)
    np.testing.assert_allclose(result.slow_reservoir, 305.0)
    np.testing.assert_allclose(result.stream_reservoir, 24.0)
    np.testing.assert_allclose(result.flood_reservoir, 32.0)
    np.testing.assert_allclose(result.pond_reservoir, 38.0)
    np.testing.assert_allclose(result.streamb_inflow, 10.0)
    np.testing.assert_allclose(result.return_swamp[:, :, IH2O:IPOCA], 1.0)
    np.testing.assert_allclose(result.return_swamp[:, :, IPOCA : ISANDSED + 1], 0.0)
    np.testing.assert_allclose(result.totflood, np.full((1, nflow), 32.0))


def test_routing_reservoir_update_step_cascades_negative_water_like_fortran():
    nflow = 10
    flood = np.zeros((1, 1, nflow), dtype=np.float64)
    flood[0, 0, :] = [-5.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0]
    slow = np.zeros_like(flood)
    slow[0, 0, IH2O] = 10.0

    result = routing_reservoir_update_step(
        runoff=np.zeros((1, nflow), dtype=np.float64),
        drainage=np.zeros((1, nflow), dtype=np.float64),
        routing_area=np.asarray([[1.0]], dtype=np.float64),
        fast_reservoir=np.zeros_like(flood),
        slow_reservoir=slow,
        stream_reservoir=np.zeros_like(flood),
        flood_reservoir=flood,
        pond_reservoir=np.zeros((1, nflow), dtype=np.float64),
        transport=np.zeros_like(flood),
        return_swamp=np.zeros_like(flood),
        floods=np.zeros_like(flood),
        flood_flow=np.zeros_like(flood),
        pond_inflow=np.zeros_like(flood),
        pond_drainage=np.zeros_like(flood),
    )

    np.testing.assert_allclose(result.flood_reservoir, 0.0)
    np.testing.assert_allclose(result.stream_reservoir, 0.0)
    np.testing.assert_allclose(result.fast_reservoir, 0.0)
    np.testing.assert_allclose(result.slow_reservoir[0, 0, IH2O], 5.0)
    np.testing.assert_allclose(result.slow_reservoir[0, 0, IDOCL], -1.0)


def test_routing_reservoir_update_step_refuses_materially_negative_slow_reservoir():
    nflow = 10
    flood = np.zeros((1, 1, nflow), dtype=np.float64)
    flood[0, 0, IH2O] = -5.0
    slow = np.zeros_like(flood)
    slow[0, 0, IH2O] = 1.0

    with pytest.raises(ValueError, match="negative slow_reservoir"):
        routing_reservoir_update_step(
            runoff=np.zeros((1, nflow), dtype=np.float64),
            drainage=np.zeros((1, nflow), dtype=np.float64),
            routing_area=np.asarray([[1.0]], dtype=np.float64),
            fast_reservoir=np.zeros_like(flood),
            slow_reservoir=slow,
            stream_reservoir=np.zeros_like(flood),
            flood_reservoir=flood,
            pond_reservoir=np.zeros((1, nflow), dtype=np.float64),
            transport=np.zeros_like(flood),
            return_swamp=np.zeros_like(flood),
            floods=np.zeros_like(flood),
            flood_flow=np.zeros_like(flood),
            pond_inflow=np.zeros_like(flood),
            pond_drainage=np.zeros_like(flood),
        )


def test_routing_area_fractions_step_matches_stream_swell_and_flood_formulas():
    nflow = 10
    stream = np.zeros((1, 2, nflow), dtype=np.float64)
    stream[0, :, IH2O] = [25.0, 75.0]
    flood = np.zeros_like(stream)
    flood[0, :, IH2O] = [50.0, 150.0]
    totflood = np.zeros((1, nflow), dtype=np.float64)
    totflood[0, IH2O] = 200.0
    pond = np.zeros((1, nflow), dtype=np.float64)
    pond[0, IH2O] = 1000.0

    result = routing_area_fractions_step(
        stream_reservoir=stream,
        routing_area=np.asarray([[400.0, 600.0]], dtype=np.float64),
        totflood=totflood,
        flood_reservoir=flood,
        pond_reservoir=pond,
        stream_seddep=np.zeros_like(stream),
        vegtot=np.asarray([0.5], dtype=np.float64),
        bulkdens=np.asarray([1.0], dtype=np.float64),
        stream_area=np.asarray([10.0], dtype=np.float64),
        headw_area=np.asarray([20.0], dtype=np.float64),
        streamr10th=np.asarray([50.0], dtype=np.float64),
        streamr90th=np.asarray([150.0], dtype=np.float64),
        floodplains=np.asarray([300.0], dtype=np.float64),
        floodh90th=np.asarray([0.5], dtype=np.float64),
        dostreamswell=True,
        do_floodplains=True,
        doponds=True,
        limit_rivdepos=False,
    )

    sqrt_stream = np.sqrt(np.asarray([25.0, 75.0]))
    sqrt_sum = sqrt_stream.sum()
    stream_area_act = 10.0 * (1.0 + 0.1 * 0.5) + 20.0 * (1.0 + 0.2 * 0.5)
    np.testing.assert_allclose(result.stream_area_act, [stream_area_act])
    np.testing.assert_allclose(result.stream_area_bas, [stream_area_act * sqrt_stream / sqrt_sum])
    np.testing.assert_allclose(result.streamfl_frac, [(stream_area_act - 30.0) / 1000.0])
    np.testing.assert_allclose(result.streamfl_frac_bas, [((stream_area_act - 30.0) * sqrt_stream / sqrt_sum) / [400.0, 600.0]])

    flood_frac_pre_pond = 300.0 / 1000.0
    pond_frac = ((0.5 + 1.0) * 1000.0 / (2000.0 * 1000.0)) ** (0.5 / 1.5)
    expected_flood_height = (2.0 / 3.0) * 0.5 * flood_frac_pre_pond ** 0.5 + 200.0 / (1000.0 * flood_frac_pre_pond)
    np.testing.assert_allclose(result.flood_frac_bas, [[0.3 * (50.0 / 200.0) / 0.4, 0.3 * (150.0 / 200.0) / 0.6]])
    np.testing.assert_allclose(result.flood_height, [expected_flood_height])
    np.testing.assert_allclose(result.pond_frac, [pond_frac])
    np.testing.assert_allclose(result.flood_frac, [flood_frac_pre_pond + pond_frac])


def test_routing_area_fractions_step_preserves_source_rivbed_limit_assignment():
    nflow = 10
    stream = np.zeros((1, 1, nflow), dtype=np.float64)
    stream[0, 0, IH2O] = 16.0
    stream_seddep = np.zeros_like(stream)
    stream_seddep[0, 0, IPOCA : IPOCP + 1] = [10.0, 20.0, 30.0]
    stream_seddep[0, 0, ICLAYSED : ISANDSED + 1] = [100.0, 100.0, 100.0]

    result = routing_area_fractions_step(
        stream_reservoir=stream,
        routing_area=np.asarray([[100.0]], dtype=np.float64),
        totflood=np.zeros((1, nflow), dtype=np.float64),
        flood_reservoir=np.zeros_like(stream),
        pond_reservoir=np.zeros((1, nflow), dtype=np.float64),
        stream_seddep=stream_seddep,
        vegtot=np.asarray([0.5], dtype=np.float64),
        bulkdens=np.asarray([1.0], dtype=np.float64),
        stream_area=np.asarray([20.0], dtype=np.float64),
        headw_area=np.asarray([0.0], dtype=np.float64),
        streamr10th=np.asarray([0.0], dtype=np.float64),
        streamr90th=np.asarray([0.0], dtype=np.float64),
        floodplains=np.asarray([0.0], dtype=np.float64),
        floodh90th=np.asarray([0.0], dtype=np.float64),
        dostreamswell=False,
        do_floodplains=False,
        doponds=False,
        limit_rivdepos=True,
        maxdep_rivsed=10.0,
    )

    riv_seddep = 300.0 / (0.5 * 20.0)
    excess = (riv_seddep - 10.0) / riv_seddep
    # The Fortran source writes the POC expression into rivbed2fld_sed(1:3),
    # overwriting the preceding sediment expression; rivbed2fld_poc is untouched.
    np.testing.assert_allclose(result.rivbed2fld_sed, [1.0e3 * np.asarray([10.0, 20.0, 30.0]) / (0.5 * 100.0) * excess])
    np.testing.assert_allclose(result.rivbed2fld_poc, 0.0)
    np.testing.assert_allclose(result.stream_seddep[0, 0, IPOCA : ISANDSED + 1], stream_seddep[0, 0, IPOCA : ISANDSED + 1] * (10.0 / riv_seddep))


def test_routing_pond_flux_step_matches_drainage_inflow_and_deposition_rules():
    nflow = 10
    fast_flow = np.zeros((1, 2, nflow), dtype=np.float64)
    fast_flow[0, 0, :] = np.arange(10.0, 20.0)
    fast_flow[0, 1, :] = np.arange(20.0, 30.0)
    pond_reservoir = np.asarray([[100.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]], dtype=np.float64)

    result = routing_pond_flux_step(
        fast_flow=fast_flow,
        pond_reservoir=pond_reservoir,
        routing_area=np.asarray([[100.0, 300.0]], dtype=np.float64),
        pond_frac=np.asarray([0.2], dtype=np.float64),
        k_litt=np.asarray([2.0], dtype=np.float64),
        reinf_slope=np.asarray([0.25], dtype=np.float64),
        flood_dep_sed=np.zeros((1, 2, 3), dtype=np.float64),
        flood_dep_poc=np.zeros((1, 2, 3), dtype=np.float64),
        doponds=True,
        dt_routing=86400.0,
    )

    expected_drainage_water = np.asarray([25.0, 75.0])
    np.testing.assert_allclose(result.pond_drainage[0, :, IH2O], expected_drainage_water)
    np.testing.assert_allclose(result.pond_drainage[0, :, IDOCL], expected_drainage_water * 10.0 / 100.0)
    np.testing.assert_allclose(result.pond_drainage[0, :, IDOCR], expected_drainage_water * 20.0 / 100.0)
    np.testing.assert_allclose(result.pond_drainage[0, :, ICO2AQ], expected_drainage_water * 30.0 / 100.0)
    np.testing.assert_allclose(result.pond_inflow[:, :, :IPOCA], fast_flow[:, :, :IPOCA] * 0.25)
    np.testing.assert_allclose(result.pond_inflow[:, :, IPOCA : ISANDSED + 1], 0.0)
    np.testing.assert_allclose(result.fast_flow, fast_flow * 0.75)
    np.testing.assert_allclose(result.flood_dep_poc[0, :, IACTIVE], fast_flow[0, :, IPOCA] * 0.25)
    np.testing.assert_allclose(result.flood_dep_poc[0, :, ISLOW], fast_flow[0, :, IPOCS] * 0.25)
    np.testing.assert_allclose(result.flood_dep_poc[0, :, IPASSIVE], fast_flow[0, :, IPOCP] * 0.25)
    np.testing.assert_allclose(result.flood_dep_sed[0, :, IC_CLAY], fast_flow[0, :, ICLAYSED] * 0.25)
    np.testing.assert_allclose(result.flood_dep_sed[0, :, IC_SILT], fast_flow[0, :, ISILTSED] * 0.25)
    np.testing.assert_allclose(result.flood_dep_sed[0, :, IC_SAND], fast_flow[0, :, ISANDSED] * 0.25)


def test_routing_pond_flux_step_zeroes_pond_state_when_disabled():
    result = routing_pond_flux_step(
        fast_flow=np.ones((1, 1, 10), dtype=np.float64),
        pond_reservoir=np.ones((1, 10), dtype=np.float64),
        routing_area=np.asarray([[10.0]], dtype=np.float64),
        pond_frac=np.asarray([0.1], dtype=np.float64),
        k_litt=np.asarray([1.0], dtype=np.float64),
        reinf_slope=np.asarray([0.5], dtype=np.float64),
        flood_dep_sed=np.ones((1, 1, 3), dtype=np.float64),
        flood_dep_poc=np.ones((1, 1, 3), dtype=np.float64),
        doponds=False,
        dt_routing=86400.0,
    )

    np.testing.assert_allclose(result.fast_flow, 1.0)
    np.testing.assert_allclose(result.pond_reservoir, 0.0)
    np.testing.assert_allclose(result.pond_inflow, 0.0)
    np.testing.assert_allclose(result.pond_drainage, 0.0)
    np.testing.assert_allclose(result.flood_dep_sed, 1.0)
    np.testing.assert_allclose(result.flood_dep_poc, 1.0)


def test_routing_swamp_flood_step_matches_new_scheme_return_and_flooding():
    nflow = 10
    stream = np.zeros((1, 1, nflow), dtype=np.float64)
    stream[0, 0, IH2O] = 100.0
    transport = np.zeros_like(stream)
    transport[0, 0, :] = [100.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]

    result = routing_swamp_flood_step(
        stream_reservoir=stream,
        transport=transport,
        routing_area=np.asarray([[100.0]], dtype=np.float64),
        route_tobasin=np.asarray([[1]], dtype=np.int32),
        topo_resid=np.asarray([[1000.0]], dtype=np.float64),
        streamr50th=np.asarray([50.0], dtype=np.float64),
        floodtemp=np.asarray([280.0], dtype=np.float64),
        swamp=np.asarray([50.0], dtype=np.float64),
        floodplains=np.asarray([100.0], dtype=np.float64),
        stream_area=np.asarray([20.0], dtype=np.float64),
        flood_reservoir=np.ones_like(stream),
        flood_dep_sed=np.zeros((1, 1, 3), dtype=np.float64),
        flood_dep_poc=np.zeros((1, 1, 3), dtype=np.float64),
        doswamps=True,
        do_floodplains=True,
        new_flood_scheme=True,
        dt_routing=86400.0,
        swamp_cst=0.2,
        stream_tcst=0.5,
    )

    stream_flow50 = min(50.0 / (1.0 * 0.5), 50.0 - 1.0e-8)
    potflood_water = 100.0 - stream_flow50
    floodindex = 50.0 / 100.0
    expected_return_water = 0.2 * potflood_water * floodindex
    expected_return = expected_return_water / 100.0 * transport[0, 0, :]
    expected_return[IH2O] = expected_return_water
    expected_flood_water = (100.0 - expected_return_water - stream_flow50) * ((100.0 - 20.0) / 100.0)

    np.testing.assert_allclose(result.streamr50th_bas, [[50.0]])
    np.testing.assert_allclose(result.stream_flow50th, [[stream_flow50]])
    np.testing.assert_allclose(result.return_swamp[0, 0, :IPOCA], expected_return[:IPOCA])
    np.testing.assert_allclose(result.return_swamp[0, 0, IPOCA : ISANDSED + 1], 0.0)
    np.testing.assert_allclose(result.floods[0, 0, IH2O], expected_flood_water)
    np.testing.assert_allclose(result.floods[0, 0, IDOCL], expected_flood_water * 10.0 / 100.0)
    np.testing.assert_allclose(result.flood_dep_poc[0, 0, IACTIVE], expected_return[IPOCA])
    np.testing.assert_allclose(result.flood_dep_poc[0, 0, ISLOW], expected_return[IPOCS])
    np.testing.assert_allclose(result.flood_dep_poc[0, 0, IPASSIVE], expected_return[IPOCP])
    np.testing.assert_allclose(result.flood_dep_sed[0, 0, IC_CLAY], expected_return[ICLAYSED])
    np.testing.assert_allclose(result.flood_dep_sed[0, 0, IC_SILT], expected_return[ISILTSED])
    np.testing.assert_allclose(result.flood_dep_sed[0, 0, IC_SAND], expected_return[ISANDSED])


def test_routing_swamp_flood_step_old_scheme_zeroes_floods_and_reservoir():
    result = routing_swamp_flood_step(
        stream_reservoir=np.ones((1, 1, 10), dtype=np.float64),
        transport=np.ones((1, 1, 10), dtype=np.float64),
        routing_area=np.asarray([[10.0]], dtype=np.float64),
        route_tobasin=np.asarray([[1]], dtype=np.int32),
        topo_resid=np.asarray([[1000.0]], dtype=np.float64),
        streamr50th=np.asarray([1.0], dtype=np.float64),
        floodtemp=np.asarray([280.0], dtype=np.float64),
        swamp=np.asarray([1.0], dtype=np.float64),
        floodplains=np.asarray([1.0], dtype=np.float64),
        stream_area=np.asarray([0.0], dtype=np.float64),
        flood_reservoir=np.ones((1, 1, 10), dtype=np.float64),
        flood_dep_sed=np.ones((1, 1, 3), dtype=np.float64),
        flood_dep_poc=np.ones((1, 1, 3), dtype=np.float64),
        doswamps=True,
        do_floodplains=True,
        new_flood_scheme=False,
        dt_routing=86400.0,
    )

    np.testing.assert_allclose(result.streamr50th_bas, 0.0)
    np.testing.assert_allclose(result.stream_flow50th, 0.0)
    np.testing.assert_allclose(result.return_swamp, 0.0)
    np.testing.assert_allclose(result.floods, 0.0)
    np.testing.assert_allclose(result.flood_reservoir, 0.0)
    np.testing.assert_allclose(result.flood_dep_sed, 1.0)
    np.testing.assert_allclose(result.flood_dep_poc, 1.0)


def test_routing_floodplain_flux_step_matches_drainage_lateral_flow_and_caps():
    nflow = 10
    flood = np.zeros((1, 1, nflow), dtype=np.float64)
    flood[0, 0, :] = [100.0, 10.0, 20.0, 30.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]

    result = routing_floodplain_flux_step(
        flood_reservoir=flood,
        stream_reservoir=np.zeros_like(flood),
        routing_area=np.asarray([[100.0]], dtype=np.float64),
        flood_frac=np.asarray([0.2], dtype=np.float64),
        flood_frac_bas=np.asarray([[0.5]], dtype=np.float64),
        k_litt=np.asarray([1.0], dtype=np.float64),
        route_tobasin=np.asarray([[1]], dtype=np.int32),
        topo_resid=np.asarray([[1000.0]], dtype=np.float64),
        flood_dep_sed=np.zeros((1, 1, 3), dtype=np.float64),
        flood_dep_poc=np.zeros((1, 1, 3), dtype=np.float64),
        do_floodplains=True,
        dofloodinfilt=True,
        dt_routing=86400.0,
        flood_tcst=4.0,
        flow_coef=1.0,
        coef_seddep_flood=(0.5, 0.8, 1.0),
    )

    drainage_water = 20.0
    ratio_drain = drainage_water / 100.0
    water_after_drain = 80.0
    docl_after_drain = 10.0 * (1.0 - ratio_drain)
    lateral_water = water_after_drain / (1.0 * 4.0 * 0.5)
    final_water = water_after_drain - lateral_water
    cap = 0.1 * final_water

    sediment_after_drain = np.asarray([110.0, 120.0, 130.0]) * (1.0 - ratio_drain)
    poc_after_drain = np.asarray([80.0, 90.0, 100.0]) * (1.0 - ratio_drain)
    sed_dep_after_drain = np.asarray([110.0, 120.0, 130.0]) * ratio_drain
    poc_dep_after_drain = np.asarray([80.0, 90.0, 100.0]) * ratio_drain

    sed_after_coef = sediment_after_drain * np.asarray([0.5, 0.2, 0.0])
    poc_after_coef = poc_after_drain * 0.5
    sed_flow = sed_after_coef * lateral_water / water_after_drain
    poc_flow = poc_after_coef * lateral_water / water_after_drain
    sed_after_flow = sed_after_coef - sed_flow
    poc_after_flow = poc_after_coef - poc_flow
    sed_excess = np.maximum(sed_after_flow - cap, 0.0)
    poc_excess = np.maximum(poc_after_flow - cap, 0.0)

    np.testing.assert_allclose(result.flood_drainage[0, 0, IH2O], drainage_water)
    np.testing.assert_allclose(result.flood_drainage[0, 0, IDOCL], 10.0 * ratio_drain)
    np.testing.assert_allclose(result.flood_flow[0, 0, IH2O], lateral_water)
    np.testing.assert_allclose(result.flood_flow[0, 0, IDOCL], docl_after_drain / (1.0 * 4.0 * 0.5))
    np.testing.assert_allclose(result.stream_reservoir, result.flood_flow)
    np.testing.assert_allclose(result.flood_reservoir[0, 0, IH2O], final_water)
    np.testing.assert_allclose(result.flood_reservoir[0, 0, ICLAYSED : ISANDSED + 1], np.minimum(sed_after_flow, cap))
    np.testing.assert_allclose(result.flood_reservoir[0, 0, IPOCA : IPOCP + 1], np.minimum(poc_after_flow, cap))
    np.testing.assert_allclose(
        result.flood_dep_sed[0, 0, :],
        sed_dep_after_drain + np.asarray([0.5, 0.8, 1.0]) * sediment_after_drain + sed_excess,
    )
    np.testing.assert_allclose(
        result.flood_dep_poc[0, 0, :],
        poc_dep_after_drain + 0.5 * poc_after_drain + poc_excess,
    )


def test_routing_floodplain_flux_step_zeroes_when_floodplains_disabled():
    result = routing_floodplain_flux_step(
        flood_reservoir=np.ones((1, 1, 10), dtype=np.float64),
        stream_reservoir=np.ones((1, 1, 10), dtype=np.float64),
        routing_area=np.asarray([[1.0]], dtype=np.float64),
        flood_frac=np.asarray([0.1], dtype=np.float64),
        flood_frac_bas=np.asarray([[0.1]], dtype=np.float64),
        k_litt=np.asarray([1.0], dtype=np.float64),
        route_tobasin=np.asarray([[1]], dtype=np.int32),
        topo_resid=np.asarray([[1000.0]], dtype=np.float64),
        flood_dep_sed=np.ones((1, 1, 3), dtype=np.float64),
        flood_dep_poc=np.ones((1, 1, 3), dtype=np.float64),
        do_floodplains=False,
        dofloodinfilt=True,
        dt_routing=86400.0,
    )

    np.testing.assert_allclose(result.flood_reservoir, 0.0)
    np.testing.assert_allclose(result.flood_drainage, 0.0)
    np.testing.assert_allclose(result.flood_flow, 0.0)
    np.testing.assert_allclose(result.stream_reservoir, 1.0)
    np.testing.assert_allclose(result.flood_dep_sed, 1.0)
    np.testing.assert_allclose(result.flood_dep_poc, 1.0)


def test_routing_floodplain_flux_step_inactive_route_zeroes_particulate_store_like_source():
    nflow = 10
    flood = np.zeros((1, 1, nflow), dtype=np.float64)
    flood[0, 0, :] = [10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    result = routing_floodplain_flux_step(
        flood_reservoir=flood,
        stream_reservoir=np.zeros_like(flood),
        routing_area=np.asarray([[1.0]], dtype=np.float64),
        flood_frac=np.asarray([0.0], dtype=np.float64),
        flood_frac_bas=np.asarray([[0.0]], dtype=np.float64),
        k_litt=np.asarray([0.0], dtype=np.float64),
        route_tobasin=np.asarray([[0]], dtype=np.int32),
        topo_resid=np.asarray([[1000.0]], dtype=np.float64),
        flood_dep_sed=np.zeros((1, 1, 3), dtype=np.float64),
        flood_dep_poc=np.zeros((1, 1, 3), dtype=np.float64),
        do_floodplains=True,
        dofloodinfilt=False,
        dt_routing=86400.0,
    )

    np.testing.assert_allclose(result.flood_flow, 0.0)
    np.testing.assert_allclose(result.flood_reservoir[0, 0, IH2O : ICO2AQ + 1], flood[0, 0, IH2O : ICO2AQ + 1])
    np.testing.assert_allclose(result.flood_reservoir[0, 0, IPOCA : ISANDSED + 1], 0.0)
    np.testing.assert_allclose(result.flood_dep_sed, 0.0)
    np.testing.assert_allclose(result.flood_dep_poc, 0.0)


def test_routing_transport_between_basins_step_routes_combined_outflow_to_targets():
    fast = np.zeros((2, 2, 4), dtype=np.float64)
    slow = np.zeros_like(fast)
    stream = np.zeros_like(fast)
    fast[0, 0, :] = [1.0, 2.0, 3.0, 4.0]
    slow[0, 0, :] = [10.0, 20.0, 30.0, 40.0]
    stream[0, 0, :] = [100.0, 200.0, 300.0, 400.0]
    fast[1, 1, :] = [5.0, 6.0, 7.0, 8.0]

    result = routing_transport_between_basins_step(
        fast_flow=fast,
        slow_flow=slow,
        stream_flow=stream,
        route_togrid=np.asarray([[1, 0], [0, 1]], dtype=np.int32),
        route_tobasin=np.asarray([[1, 0], [0, 0]], dtype=np.int32),
        transport=np.zeros((2, 2, 4), dtype=np.float64),
    )

    expected = np.zeros((2, 2, 4), dtype=np.float64)
    expected[1, 1, :] += fast[0, 0, :] + slow[0, 0, :] + stream[0, 0, :]
    expected[1, 0, :] += fast[1, 1, :]
    np.testing.assert_allclose(result.transport, expected)
    np.testing.assert_allclose(result.water_balance, 0.0)
    np.testing.assert_allclose(result.carbon_balance, 0.0)


def test_routing_transport_between_basins_step_updates_optional_balance_diagnostics():
    fast = np.asarray([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float64)
    slow = np.asarray([[[10.0, 20.0, 30.0, 40.0, 50.0]]], dtype=np.float64)
    stream = np.asarray([[[100.0, 200.0, 300.0, 400.0, 500.0]]], dtype=np.float64)
    existing_transport = np.asarray([[[7.0, 11.0, 13.0, 17.0, 19.0]]], dtype=np.float64)

    result = routing_transport_between_basins_step(
        fast_flow=fast,
        slow_flow=slow,
        stream_flow=stream,
        route_togrid=np.asarray([[0]], dtype=np.int32),
        route_tobasin=np.asarray([[0]], dtype=np.int32),
        transport=existing_transport,
        check_riverbal=True,
        water_balance=np.asarray([2.0], dtype=np.float64),
        carbon_balance=np.asarray([3.0], dtype=np.float64),
    )

    routed = fast + slow + stream
    expected_transport = existing_transport + routed
    np.testing.assert_allclose(result.transport, expected_transport)
    np.testing.assert_allclose(result.water_balance, [2.0 + expected_transport[0, :, IH2O].sum() - routed[0, :, IH2O].sum()])
    np.testing.assert_allclose(
        result.carbon_balance,
        [3.0 + expected_transport[0, :, IH2O + 1 : IPOCP + 1].sum() - routed[0, :, IH2O + 1 : IPOCP + 1].sum()],
    )


def test_routing_transport_between_basins_step_refuses_out_of_range_target():
    with pytest.raises(ValueError, match="outside transport shape"):
        routing_transport_between_basins_step(
            fast_flow=np.zeros((1, 1, 2), dtype=np.float64),
            slow_flow=np.zeros((1, 1, 2), dtype=np.float64),
            stream_flow=np.zeros((1, 1, 2), dtype=np.float64),
            route_togrid=np.asarray([[2]], dtype=np.int32),
            route_tobasin=np.asarray([[0]], dtype=np.int32),
            transport=np.zeros((1, 1, 2), dtype=np.float64),
        )


def test_routing_flow_step_wires_disabled_surface_sequence_and_fortran_outlet_route():
    npts, nbas, nflow = 1, 1, 10
    fast = np.zeros((npts, nbas, nflow), dtype=np.float64)
    slow = np.zeros_like(fast)
    stream = np.zeros_like(fast)
    fast[0, 0, IH2O] = 10.0
    slow[0, 0, IH2O] = 20.0
    stream[0, 0, IH2O] = 30.0
    stream[0, 0, IPOCA] = 4.0
    stream[0, 0, ICLAYSED] = 10.0
    state = RoutingFlowState(
        fast_reservoir=fast,
        slow_reservoir=slow,
        stream_reservoir=stream,
        flood_reservoir=np.zeros_like(fast),
        pond_reservoir=np.zeros((npts, nflow), dtype=np.float64),
        lake_reservoir=np.zeros((npts, nflow), dtype=np.float64),
        stream_seddep=np.zeros_like(fast),
        stream_damavail=np.asarray([2.0], dtype=np.float64),
        carbon_32l=np.zeros((npts, 3, 2, 3), dtype=np.float64),
        flood_frac=np.zeros((npts,), dtype=np.float64),
        flood_frac_bas=np.zeros((npts, nbas), dtype=np.float64),
        pond_frac=np.zeros((npts,), dtype=np.float64),
        streamfl_frac=np.zeros((npts,), dtype=np.float64),
        streamfl_frac_bas=np.zeros((npts, nbas), dtype=np.float64),
        flood_diag=np.zeros((npts, nflow), dtype=np.float64),
        pond_diag=np.zeros((npts, nflow), dtype=np.float64),
        lake_diag=np.zeros((npts, nflow), dtype=np.float64),
    )

    result = routing_flow_step(
        state=state,
        runoff=np.zeros((npts, nflow), dtype=np.float64),
        drainage=np.zeros((npts, nflow), dtype=np.float64),
        floodout=np.zeros((npts,), dtype=np.float64),
        precip=np.zeros((npts, nflow), dtype=np.float64),
        flood_inp=np.zeros((npts, nflow), dtype=np.float64),
        stream_inp=np.zeros((npts, nflow), dtype=np.float64),
        routing_area=np.full((npts, nbas), 100.0, dtype=np.float64),
        topo_resid=np.full((npts, nbas), 1000.0, dtype=np.float64),
        route_togrid=np.asarray([[1]], dtype=np.int32),
        route_tobasin=np.asarray([[nbas + 3]], dtype=np.int32),
        qflow_ave=np.asarray([1.0], dtype=np.float64),
        stream_resave=np.asarray([6.0], dtype=np.float64),
        basdrainarea=np.ones((npts, nbas), dtype=np.float64),
        basgravel=np.ones((npts, nbas), dtype=np.float64),
        bulkdens=np.full((npts,), 1000.0, dtype=np.float64),
        zz_deep=np.asarray([0.5, 2.0, 3.0], dtype=np.float64),
        veget_max=np.asarray([[0.0, 1.0]], dtype=np.float64),
        vegtot=np.asarray([1.0], dtype=np.float64),
        totnobio=np.asarray([0.0], dtype=np.float64),
        transpot_mean=np.asarray([0.0], dtype=np.float64),
        humrel=np.asarray([1.0], dtype=np.float64),
        k_litt=np.asarray([0.0], dtype=np.float64),
        floodtemp=np.asarray([280.0], dtype=np.float64),
        temp_sol=np.asarray([280.0], dtype=np.float64),
        reinf_slope=np.asarray([0.0], dtype=np.float64),
        irrigated=np.asarray([0.0], dtype=np.float64),
        stream_area=np.asarray([1.0], dtype=np.float64),
        headw_area=np.asarray([0.0], dtype=np.float64),
        streamr10th=np.asarray([0.0], dtype=np.float64),
        streamr50th=np.asarray([0.0], dtype=np.float64),
        streamr90th=np.asarray([1.0], dtype=np.float64),
        floodplains=np.asarray([0.0], dtype=np.float64),
        floodh90th=np.asarray([0.0], dtype=np.float64),
        swamp=np.asarray([0.0], dtype=np.float64),
        hydrodiag=np.ones((npts, nbas, nflow), dtype=np.int32),
        mask_coast=np.asarray([0.0], dtype=np.float64),
        nb_coast_gridcells=0,
        dt_routing=86400.0,
        do_floodplains=False,
        dofloodinfilt=False,
        doponds=False,
        doswamps=False,
        dostreamswell=False,
        new_flood_scheme=False,
        do_irrigation=False,
        ok_doc=False,
    )

    routed = result.outflow.fast_flow + result.outflow.slow_flow + result.stream_erosion.stream_flow
    assert result.stream_erosion.stream_flow[0, 0, ICLAYSED] > result.outflow.stream_flow[0, 0, ICLAYSED]
    np.testing.assert_allclose(result.transport.transport[0, nbas + 2, :], routed[0, 0, :])
    np.testing.assert_allclose(result.diagnostics.riverflow, routed[:, 0, :])
    np.testing.assert_allclose(result.return_reinfiltration.returnflow, 0.0)
    np.testing.assert_allclose(result.return_reinfiltration.reinfiltration, 0.0)
    np.testing.assert_allclose(result.state.flood_frac, 0.0)
    np.testing.assert_allclose(result.state.pond_frac, 0.0)
    np.testing.assert_allclose(result.lake_overflow.coastalflow, 0.0)


def _routing_flow_step_base_case(npts=1, nbas=1, nflow=10):
    state = RoutingFlowState(
        fast_reservoir=np.zeros((npts, nbas, nflow), dtype=np.float64),
        slow_reservoir=np.zeros((npts, nbas, nflow), dtype=np.float64),
        stream_reservoir=np.zeros((npts, nbas, nflow), dtype=np.float64),
        flood_reservoir=np.zeros((npts, nbas, nflow), dtype=np.float64),
        pond_reservoir=np.zeros((npts, nflow), dtype=np.float64),
        lake_reservoir=np.zeros((npts, nflow), dtype=np.float64),
        stream_seddep=np.zeros((npts, nbas, nflow), dtype=np.float64),
        stream_damavail=np.zeros((npts,), dtype=np.float64),
        carbon_32l=np.zeros((npts, 3, 2, 3), dtype=np.float64),
        flood_frac=np.zeros((npts,), dtype=np.float64),
        flood_frac_bas=np.zeros((npts, nbas), dtype=np.float64),
        pond_frac=np.zeros((npts,), dtype=np.float64),
        streamfl_frac=np.zeros((npts,), dtype=np.float64),
        streamfl_frac_bas=np.zeros((npts, nbas), dtype=np.float64),
        flood_diag=np.zeros((npts, nflow), dtype=np.float64),
        pond_diag=np.zeros((npts, nflow), dtype=np.float64),
        lake_diag=np.zeros((npts, nflow), dtype=np.float64),
    )
    kwargs = dict(
        state=state,
        runoff=np.zeros((npts, nflow), dtype=np.float64),
        drainage=np.zeros((npts, nflow), dtype=np.float64),
        floodout=np.zeros((npts,), dtype=np.float64),
        precip=np.zeros((npts, nflow), dtype=np.float64),
        flood_inp=np.zeros((npts, nflow), dtype=np.float64),
        stream_inp=np.zeros((npts, nflow), dtype=np.float64),
        routing_area=np.full((npts, nbas), 100.0, dtype=np.float64),
        topo_resid=np.full((npts, nbas), 1000.0, dtype=np.float64),
        route_togrid=np.repeat(np.arange(1, npts + 1, dtype=np.int32)[:, None], nbas, axis=1),
        route_tobasin=np.full((npts, nbas), nbas + 3, dtype=np.int32),
        qflow_ave=np.zeros((npts,), dtype=np.float64),
        stream_resave=np.zeros((npts,), dtype=np.float64),
        basdrainarea=np.ones((npts, nbas), dtype=np.float64),
        basgravel=np.ones((npts, nbas), dtype=np.float64),
        bulkdens=np.full((npts,), 1000.0, dtype=np.float64),
        zz_deep=np.asarray([0.5, 2.0, 3.0], dtype=np.float64),
        veget_max=np.tile(np.asarray([[0.0, 1.0]], dtype=np.float64), (npts, 1)),
        vegtot=np.ones((npts,), dtype=np.float64),
        totnobio=np.zeros((npts,), dtype=np.float64),
        transpot_mean=np.zeros((npts,), dtype=np.float64),
        humrel=np.ones((npts,), dtype=np.float64),
        k_litt=np.zeros((npts,), dtype=np.float64),
        floodtemp=np.full((npts,), 280.0, dtype=np.float64),
        temp_sol=np.full((npts,), 280.0, dtype=np.float64),
        reinf_slope=np.zeros((npts,), dtype=np.float64),
        irrigated=np.zeros((npts,), dtype=np.float64),
        stream_area=np.ones((npts,), dtype=np.float64),
        headw_area=np.zeros((npts,), dtype=np.float64),
        streamr10th=np.zeros((npts,), dtype=np.float64),
        streamr50th=np.zeros((npts,), dtype=np.float64),
        streamr90th=np.ones((npts,), dtype=np.float64),
        floodplains=np.zeros((npts,), dtype=np.float64),
        floodh90th=np.zeros((npts,), dtype=np.float64),
        swamp=np.zeros((npts,), dtype=np.float64),
        hydrodiag=np.ones((npts, nbas, nflow), dtype=np.int32),
        mask_coast=np.zeros((npts,), dtype=np.float64),
        nb_coast_gridcells=0,
        dt_routing=86400.0,
        do_floodplains=False,
        dofloodinfilt=False,
        doponds=False,
        doswamps=False,
        dostreamswell=False,
        new_flood_scheme=False,
        do_irrigation=False,
        ok_doc=False,
    )
    return kwargs


def test_routing_flow_step_composes_active_floodplain_and_pond_branches():
    kwargs = _routing_flow_step_base_case()
    state = kwargs["state"]._replace(
        fast_reservoir=kwargs["state"].fast_reservoir.copy(),
        flood_reservoir=kwargs["state"].flood_reservoir.copy(),
        pond_reservoir=kwargs["state"].pond_reservoir.copy(),
        flood_frac=np.asarray([0.4], dtype=np.float64),
        flood_frac_bas=np.asarray([[0.4]], dtype=np.float64),
        pond_frac=np.asarray([0.1], dtype=np.float64),
    )
    state.fast_reservoir[0, 0, IH2O] = 10.0
    state.fast_reservoir[0, 0, IPOCA] = 2.0
    state.flood_reservoir[0, 0, IH2O] = 30.0
    state.flood_reservoir[0, 0, IPOCA : ISANDSED + 1] = [3.0, 0.0, 0.0, 4.0, 0.0, 0.0]
    state.pond_reservoir[0, IH2O] = 20.0
    flood_inp = kwargs["flood_inp"].copy()
    flood_inp[0, IDOCL] = 0.2
    kwargs.update(
        state=state,
        flood_inp=flood_inp,
        k_litt=np.asarray([1.0], dtype=np.float64),
        do_floodplains=True,
        doponds=True,
    )

    result = routing_flow_step(**kwargs)

    np.testing.assert_allclose(result.flood_pond_input.flood_inp_bas[0, 0, IDOCL], 20.0)
    assert result.floodplain.flood_flow[0, 0, IH2O] > 0.0
    np.testing.assert_allclose(result.pond.pond_drainage[0, 0, IH2O], 10.0)
    np.testing.assert_allclose(result.return_reinfiltration.reinfiltration[0, IH2O], 0.1)
    assert result.diagnostics.flood_res[0] >= 0.0


def test_routing_flow_step_applies_lake_overflow_after_diagnostics():
    kwargs = _routing_flow_step_base_case()
    state = kwargs["state"]._replace(lake_reservoir=np.asarray([[80.0, 8.0] + [0.0] * 8], dtype=np.float64))
    kwargs.update(
        state=state,
        routing_area=np.asarray([[10.0]], dtype=np.float64),
        mask_coast=np.asarray([1.0], dtype=np.float64),
        nb_coast_gridcells=1,
        max_lake_reservoir=5.0,
    )

    result = routing_flow_step(**kwargs)

    np.testing.assert_allclose(result.lake_overflow.lake_overflow[0, IH2O], 30.0)
    np.testing.assert_allclose(result.lake_overflow.lake_overflow[0, IDOCL], 3.0)
    np.testing.assert_allclose(result.state.lake_reservoir[0, IH2O], 50.0)
    np.testing.assert_allclose(result.lake_overflow.coastalflow[0, IH2O], 30.0)


def test_routing_flow_step_requires_fortran_one_based_routes():
    kwargs = _routing_flow_step_base_case()
    kwargs["route_togrid"] = np.asarray([[0]], dtype=np.int32)

    with pytest.raises(ValueError, match="Fortran route_togrid"):
        routing_flow_step(**kwargs)


def test_routing_flow_step_irrigation_uses_precip_not_flood_input():
    kwargs = _routing_flow_step_base_case()
    state = kwargs["state"]._replace(stream_reservoir=kwargs["state"].stream_reservoir.copy())
    state.stream_reservoir[0, 0, IH2O] = 100.0
    precip = kwargs["precip"].copy()
    flood_inp = kwargs["flood_inp"].copy()
    precip[0, IH2O] = 0.25
    flood_inp[0, IH2O] = 1000.0
    kwargs.update(
        state=state,
        precip=precip,
        flood_inp=flood_inp,
        transpot_mean=np.asarray([2.0], dtype=np.float64),
        humrel=np.asarray([0.5], dtype=np.float64),
        irrigated=np.asarray([100.0], dtype=np.float64),
        do_irrigation=True,
    )

    result = routing_flow_step(**kwargs)

    expected_netereq = 2.0 - 0.25
    expected_basin_need = expected_netereq * 100.0
    expected_actual = min(expected_basin_need, 100.0)
    np.testing.assert_allclose(result.irrigation.irrig_netereq[0], expected_netereq)
    np.testing.assert_allclose(result.irrigation.irrig_actual[0, 0, IH2O], expected_actual)
    np.testing.assert_allclose(result.diagnostics.irrigation[0, IH2O], expected_actual / 100.0)
    np.testing.assert_allclose(result.state.stream_reservoir[0, 0, IH2O], 100.0 - expected_actual)


def test_routing_flow_step_composes_new_flood_scheme_swamp_and_flood_branches():
    kwargs = _routing_flow_step_base_case()
    state = kwargs["state"]._replace(
        fast_reservoir=kwargs["state"].fast_reservoir.copy(),
        stream_reservoir=kwargs["state"].stream_reservoir.copy(),
    )
    state.fast_reservoir[0, 0, IH2O] = 100.0
    state.stream_reservoir[0, 0, IH2O] = 10.0
    kwargs.update(
        state=state,
        route_tobasin=np.asarray([[1]], dtype=np.int32),
        streamr50th=np.asarray([1.0], dtype=np.float64),
        swamp=np.asarray([50.0], dtype=np.float64),
        floodplains=np.asarray([100.0], dtype=np.float64),
        stream_area=np.asarray([0.0], dtype=np.float64),
        do_floodplains=True,
        doswamps=True,
        new_flood_scheme=True,
    )

    result = routing_flow_step(**kwargs)

    assert result.transport.transport[0, 0, IH2O] > result.swamp_flood.stream_flow50th[0, 0]
    assert result.swamp_flood.return_swamp[0, 0, IH2O] > 0.0
    assert result.swamp_flood.floods[0, 0, IH2O] > 0.0
    np.testing.assert_allclose(
        result.reservoir_update.stream_reservoir[0, 0, IH2O],
        state.stream_reservoir[0, 0, IH2O] + result.swamp_flood.stream_flow50th[0, 0],
    )


def test_routing_flow_step_composes_doc_chemistry_active_stream_path():
    kwargs = _routing_flow_step_base_case()
    stream = kwargs["state"].stream_reservoir.copy()
    stream_seddep = kwargs["state"].stream_seddep.copy()
    stream[0, 0, IH2O] = 100.0
    stream[0, 0, IDOCL] = 1.0
    stream[0, 0, IDOCR] = 2.0
    stream[0, 0, ICO2AQ] = 3.0
    stream[0, 0, IPOCA : IPOCP + 1] = [0.5, 0.4, 0.3]
    stream_seddep[0, 0, IPOCA : IPOCP + 1] = [1.0, 0.8, 0.6]
    state = kwargs["state"]._replace(stream_reservoir=stream, stream_seddep=stream_seddep)
    kwargs.update(
        state=state,
        stream_area=np.asarray([10.0], dtype=np.float64),
        ok_doc=True,
    )

    result = routing_flow_step(**kwargs)

    assert np.isfinite(result.chemistry.pco2_aq[0, ISTREAMR])
    assert result.chemistry.pco2_aq[0, ISTREAMR] > 0.0
    assert result.chemistry.poc_co2_rivbed[0] > 0.0
    assert result.chemistry.stream_reservoir[0, 0, IDOCL] != stream[0, 0, IDOCL]


def test_routing_flow_step_composes_neighbor_irrigation_adduction():
    kwargs = _routing_flow_step_base_case(npts=2)
    state = kwargs["state"]._replace(stream_reservoir=kwargs["state"].stream_reservoir.copy())
    state.stream_reservoir[1, 0, IH2O] = 50.0
    kwargs.update(
        state=state,
        transpot_mean=np.asarray([2.0, 0.0], dtype=np.float64),
        humrel=np.asarray([0.5, 1.0], dtype=np.float64),
        irrigated=np.asarray([100.0, 0.0], dtype=np.float64),
        do_irrigation=True,
        resolution=np.asarray([[50000.0, 50000.0], [50000.0, 50000.0]], dtype=np.float64),
        neighbours=np.asarray([[1], [0]], dtype=np.int32),
    )

    result = routing_flow_step(**kwargs)

    np.testing.assert_allclose(result.irrigation.irrig_actual[0, 0, IH2O], 0.0)
    np.testing.assert_allclose(result.irrigation.irrig_adduct[0, 0, IH2O], 50.0)
    np.testing.assert_allclose(result.state.stream_reservoir[1, 0, IH2O], 0.0)
    np.testing.assert_allclose(result.diagnostics.irrigation[0, IH2O], 0.5)


def test_routing_flow_step_wires_river_balance_diagnostics():
    kwargs = _routing_flow_step_base_case()
    state = kwargs["state"]._replace(fast_reservoir=kwargs["state"].fast_reservoir.copy())
    state.fast_reservoir[0, 0, IH2O] = 10.0
    state.fast_reservoir[0, 0, IDOCL] = 1.0
    kwargs.update(
        state=state,
        check_riverbal=True,
        water_balance=np.asarray([2.0], dtype=np.float64),
        carbon_balance=np.asarray([3.0], dtype=np.float64),
    )

    result = routing_flow_step(**kwargs)

    assert result.outflow.fast_flow[0, 0, IH2O] > 0.0
    np.testing.assert_allclose(result.transport.water_balance, [2.0])
    np.testing.assert_allclose(result.transport.carbon_balance, [3.0])


def _routing_daily_boundary_base_case():
    flow = _routing_flow_step_base_case()
    npts, nbas, nflow = flow["state"].fast_reservoir.shape
    daily = dict(
        accumulator_state=_state(npts=npts, nflow=nflow),
        flow_state=flow["state"],
        floodout=np.zeros((npts,), dtype=np.float64),
        precip_rain=np.zeros((npts,), dtype=np.float64),
        runoff=np.zeros((npts,), dtype=np.float64),
        drainage=np.zeros((npts,), dtype=np.float64),
        temp_sol=np.full((npts,), 280.0, dtype=np.float64),
        veget_max=flow["veget_max"],
        transpot=np.zeros_like(flow["veget_max"]),
        totfrac_nobio=np.zeros((npts,), dtype=np.float64),
        k_litt=np.zeros((npts,), dtype=np.float64),
        humrel=np.ones_like(flow["veget_max"]),
        is_tree=np.asarray([False, False]),
        routing_area=flow["routing_area"],
        topo_resid=flow["topo_resid"],
        route_togrid=flow["route_togrid"],
        route_tobasin=flow["route_tobasin"],
        qflow_ave=flow["qflow_ave"],
        stream_resave=flow["stream_resave"],
        basdrainarea=flow["basdrainarea"],
        basgravel=flow["basgravel"],
        bulkdens=flow["bulkdens"],
        zz_deep=flow["zz_deep"],
        floodtemp=flow["floodtemp"],
        reinf_slope=flow["reinf_slope"],
        irrigated=flow["irrigated"],
        stream_area=flow["stream_area"],
        headw_area=flow["headw_area"],
        streamr10th=flow["streamr10th"],
        streamr50th=flow["streamr50th"],
        streamr90th=flow["streamr90th"],
        floodplains=flow["floodplains"],
        floodh90th=flow["floodh90th"],
        swamp=flow["swamp"],
        hydrodiag=flow["hydrodiag"],
        mask_coast=flow["mask_coast"],
        nb_coast_gridcells=flow["nb_coast_gridcells"],
        dt_sechiba=1800.0,
        dt_routing=86400.0,
        do_floodplains=flow["do_floodplains"],
        dofloodinfilt=flow["dofloodinfilt"],
        doponds=flow["doponds"],
        doswamps=flow["doswamps"],
        dostreamswell=flow["dostreamswell"],
        new_flood_scheme=flow["new_flood_scheme"],
        do_irrigation=flow["do_irrigation"],
        ok_doc=flow["ok_doc"],
    )
    return daily


def test_routing_daily_boundary_before_gate_only_accumulates_and_returns_zero_outputs():
    kwargs = _routing_daily_boundary_base_case()
    kwargs.update(
        precip_rain=np.asarray([1.0], dtype=np.float64),
        runoff=np.asarray([0.2], dtype=np.float64),
        drainage=np.asarray([0.3], dtype=np.float64),
    )

    result = routing_daily_boundary_step(**kwargs)

    assert not result.accumulator_step.due
    assert result.flow_step is None
    assert result.lake_step is None
    assert result.scaled_outputs is None
    np.testing.assert_allclose(result.accumulator_state.precip_mean[0, IH2O], 1.0)
    np.testing.assert_allclose(result.flow_state.fast_reservoir, kwargs["flow_state"].fast_reservoir)
    for value in result.zero_outputs.values():
        np.testing.assert_allclose(value, 0.0)


def test_routing_daily_boundary_due_runs_flow_lake_scales_outputs_and_resets_accumulators():
    kwargs = _routing_daily_boundary_base_case()
    flow_state = kwargs["flow_state"]._replace(fast_reservoir=kwargs["flow_state"].fast_reservoir.copy())
    flow_state.fast_reservoir[0, 0, IH2O] = 10.0
    kwargs.update(
        flow_state=flow_state,
        route_tobasin=np.asarray([[1 + 1]], dtype=np.int32),
        hydrodiag=np.ones((1, 1, 10), dtype=np.int32),
        dt_sechiba=86400.0,
        dt_routing=86400.0,
    )

    result = routing_daily_boundary_step(**kwargs)

    assert result.accumulator_step.due
    assert result.flow_step is not None
    assert result.lake_step is not None
    assert result.scaled_outputs is not None
    np.testing.assert_allclose(result.accumulator_state.time_counter, 0.0)
    np.testing.assert_allclose(result.accumulator_state.runoff_mean, 0.0)
    np.testing.assert_allclose(result.flow_state.lake_reservoir, result.lake_step.lake_reservoir)
    np.testing.assert_allclose(result.flow_state.lake_diag, result.lake_step.lake_diag)
    np.testing.assert_allclose(result.lake_step.lake_reservoir[0, IH2O], result.flow_step.diagnostics.lakeinflow[0, IH2O])
    np.testing.assert_allclose(result.scaled_outputs.hydrographs[0, IH2O], result.flow_step.diagnostics.hydrographs[0, IH2O] / 1000.0)


def test_routing_daily_boundary_combined_active_state_writebacks_remain_coherent():
    kwargs = _routing_daily_boundary_base_case()
    doc_exp = np.zeros((1, 3, 10), dtype=np.float64)
    doc_exp[0, IRUNOFF, IDOCL] = 1.0
    doc_exp[0, IRUNOFF, IDOCR] = 1.5
    doc_exp[0, IDRAINAGE, ICO2AQ] = 2.0
    doc_exp[0, IFLOODED, IDOCL] = 2.5
    doc_ero = np.asarray([[0.5, 0.25, 0.125]], dtype=np.float64)
    poc_exp = np.asarray([[0.8, 0.6, 0.4]], dtype=np.float64)
    sed_exp = np.asarray([[0.3, 0.2, 0.1]], dtype=np.float64)
    flow_state = kwargs["flow_state"]._replace(
        fast_reservoir=kwargs["flow_state"].fast_reservoir.copy(),
        stream_reservoir=kwargs["flow_state"].stream_reservoir.copy(),
        flood_reservoir=kwargs["flow_state"].flood_reservoir.copy(),
        pond_reservoir=kwargs["flow_state"].pond_reservoir.copy(),
        lake_reservoir=kwargs["flow_state"].lake_reservoir.copy(),
        stream_seddep=kwargs["flow_state"].stream_seddep.copy(),
        flood_frac=np.asarray([0.3], dtype=np.float64),
        flood_frac_bas=np.asarray([[0.3]], dtype=np.float64),
        pond_frac=np.asarray([0.1], dtype=np.float64),
    )
    flow_state.fast_reservoir[0, 0, IH2O] = 80.0
    flow_state.fast_reservoir[0, 0, IPOCA] = 2.0
    flow_state.stream_reservoir[0, 0, IH2O] = 100.0
    flow_state.stream_reservoir[0, 0, IDOCL : ICO2AQ + 1] = [1.0, 2.0, 3.0]
    flow_state.stream_reservoir[0, 0, IPOCA : IPOCP + 1] = [0.5, 0.4, 0.3]
    flow_state.stream_seddep[0, 0, IPOCA : IPOCP + 1] = [1.0, 0.8, 0.6]
    flow_state.flood_reservoir[0, 0, IH2O] = 20.0
    flow_state.flood_reservoir[0, 0, IPOCA] = 1.0
    flow_state.pond_reservoir[0, IH2O] = 10.0
    flow_state.lake_reservoir[0, IH2O] = 800.0
    kwargs.update(
        flow_state=flow_state,
        route_tobasin=np.asarray([[1]], dtype=np.int32),
        runoff=np.asarray([0.0], dtype=np.float64),
        transpot=np.asarray([[0.0, 2.0]], dtype=np.float64),
        humrel=np.asarray([[0.0, 0.5]], dtype=np.float64),
        k_litt=np.asarray([1.0], dtype=np.float64),
        irrigated=np.asarray([100.0], dtype=np.float64),
        stream_area=np.asarray([1.0], dtype=np.float64),
        streamr50th=np.asarray([1.0], dtype=np.float64),
        floodplains=np.asarray([100.0], dtype=np.float64),
        floodh90th=np.asarray([5.0], dtype=np.float64),
        swamp=np.asarray([50.0], dtype=np.float64),
        mask_coast=np.asarray([1.0], dtype=np.float64),
        nb_coast_gridcells=1,
        dt_sechiba=86400.0,
        dt_routing=86400.0,
        do_floodplains=True,
        dofloodinfilt=True,
        doponds=True,
        doswamps=True,
        new_flood_scheme=True,
        do_irrigation=True,
        ok_doc=True,
        max_lake_reservoir=5.0,
        doc_exp_agg=doc_exp,
        doc_ero_agg=doc_ero,
        poc_exp_agg=poc_exp,
        sed_exp_agg=sed_exp,
    )

    result = routing_daily_boundary_step(**kwargs)

    assert result.flow_step is not None
    assert result.lake_step is not None
    assert result.scaled_outputs is not None
    np.testing.assert_allclose(result.accumulator_state.time_counter, 0.0)
    assert np.all(result.flow_state.fast_reservoir[:, :, IH2O] >= 0.0)
    assert np.all(result.flow_state.stream_reservoir[:, :, IH2O] >= 0.0)
    assert np.all(result.flow_state.flood_reservoir[:, :, IH2O] >= 0.0)
    assert np.all(result.flow_state.pond_reservoir[:, IH2O] >= 0.0)
    assert result.flow_step.swamp_flood.return_swamp[0, 0, IH2O] >= 0.0
    assert result.flow_step.chemistry.poc_co2_rivbed[0] > 0.0
    assert result.flow_step.irrigation.irrig_actual[0, 0, IH2O] > 0.0
    assert result.flow_step.lake_overflow.coastalflow[0, IH2O] > 0.0
    assert result.scaled_outputs.coastalflow[0, IH2O] == result.flow_step.lake_overflow.coastalflow[0, IH2O] / 1000.0


def test_routing_stream_erosion_step_deposits_excess_sediment_and_poc_like_source():
    stream = np.zeros((1, 1, 10), dtype=np.float64)
    stream[0, 0, IH2O] = 2000.0
    stream[0, 0, IPOCA : IPOCP + 1] = [4.0, 5.0, 6.0]
    stream[0, 0, ICLAYSED : ISANDSED + 1] = [10.0, 20.0, 30.0]
    flow = np.zeros_like(stream)
    flow[0, 0, IH2O] = 1000.0

    result = routing_stream_erosion_step(
        stream_reservoir=stream,
        stream_flow=flow,
        stream_seddep=np.zeros_like(stream),
        qflow_avebas=np.asarray([[1.0]], dtype=np.float64),
        basdrainarea=np.asarray([[1.0]], dtype=np.float64),
        basgravel=np.asarray([[0.0]], dtype=np.float64),
        stream_damavailbas=np.asarray([[100.0]], dtype=np.float64),
        routing_area=np.asarray([[10.0]], dtype=np.float64),
        bulkdens=np.asarray([2.0], dtype=np.float64),
        zz_deep=np.asarray([0.5, 2.0, 3.0], dtype=np.float64),
        veget_max=np.asarray([[0.0, 1.0]], dtype=np.float64),
        carbon_32l=np.zeros((1, 3, 2, 3), dtype=np.float64),
        dt_routing=1.0,
        coef_stc=(0.001, 0.001, 0.001),
        coef_seddep=(0.1, 0.2, 0.5),
    )

    frac_out = 0.5
    sedcap = np.ones(3, dtype=np.float64)
    erodep_sed = np.asarray(
        [
            0.1 * (10.0 - sedcap[0] / frac_out),
            0.2 * (20.0 - sedcap[1] / frac_out),
            0.5 * (30.0 - sedcap[2] / frac_out),
        ],
        dtype=np.float64,
    )
    erodep_poc = np.asarray([4.0, 5.0, 6.0], dtype=np.float64) * erodep_sed[0] / 10.0
    expected_sed_after_dep = np.asarray([10.0, 20.0, 30.0], dtype=np.float64) - erodep_sed
    expected_poc_after_dep = np.asarray([4.0, 5.0, 6.0], dtype=np.float64) - erodep_poc

    np.testing.assert_allclose(result.sedcap[0, 0, :], sedcap)
    np.testing.assert_allclose(result.stream_erodep[0, 0, ICLAYSED : ISANDSED + 1], erodep_sed)
    np.testing.assert_allclose(result.stream_erodep[0, 0, IPOCA : IPOCP + 1], erodep_poc)
    np.testing.assert_allclose(result.stream_seddep[0, 0, ICLAYSED : ISANDSED + 1], erodep_sed)
    np.testing.assert_allclose(result.stream_seddep[0, 0, IPOCA : IPOCP + 1], erodep_poc)
    np.testing.assert_allclose(result.stream_flow[0, 0, ICLAYSED : ISANDSED + 1], frac_out * expected_sed_after_dep)
    np.testing.assert_allclose(result.stream_flow[0, 0, IPOCA : IPOCP + 1], frac_out * expected_poc_after_dep)
    np.testing.assert_allclose(result.stream_reservoir[0, 0, ICLAYSED : ISANDSED + 1], (1.0 - frac_out) * expected_sed_after_dep)
    np.testing.assert_allclose(result.stream_reservoir[0, 0, IPOCA : IPOCP + 1], (1.0 - frac_out) * expected_poc_after_dep)
    np.testing.assert_allclose(result.stream_reservoir[0, 0, IH2O], 1000.0)
    np.testing.assert_allclose(result.stream_damavailbas, [[100.0 - erodep_sed.sum() / 2.0]])


def test_routing_stream_erosion_step_reerodes_bed_deposits_with_poc_following_clay():
    stream = np.zeros((1, 1, 10), dtype=np.float64)
    stream[0, 0, IH2O] = 2000.0
    stream[0, 0, IPOCA : IPOCP + 1] = [3.0, 4.0, 5.0]
    stream[0, 0, ICLAYSED] = 1.0
    flow = np.zeros_like(stream)
    flow[0, 0, IH2O] = 1000.0
    seddep = np.zeros_like(stream)
    seddep[0, 0, ICLAYSED] = 2.0
    seddep[0, 0, IPOCA : IPOCP + 1] = [2.0, 4.0, 6.0]

    result = routing_stream_erosion_step(
        stream_reservoir=stream,
        stream_flow=flow,
        stream_seddep=seddep,
        qflow_avebas=np.asarray([[1.0]], dtype=np.float64),
        basdrainarea=np.asarray([[1.0]], dtype=np.float64),
        basgravel=np.asarray([[0.0]], dtype=np.float64),
        stream_damavailbas=np.asarray([[10.0]], dtype=np.float64),
        routing_area=np.asarray([[10.0]], dtype=np.float64),
        bulkdens=np.asarray([1.0], dtype=np.float64),
        zz_deep=np.asarray([0.5, 2.0, 3.0], dtype=np.float64),
        veget_max=np.asarray([[0.0, 1.0]], dtype=np.float64),
        carbon_32l=np.zeros((1, 3, 2, 3), dtype=np.float64),
        dt_routing=1.0,
        coef_stc=(0.001, 0.0, 0.0),
        coef_redet=(0.5, 0.5, 0.5),
    )

    frac_out = 0.5
    perosed = 0.5 * (frac_out * 1.0 - 1.0)
    erodep_clay = perosed / frac_out
    erodep_poc = seddep[0, 0, IPOCA : IPOCP + 1] * erodep_clay / seddep[0, 0, ICLAYSED]

    np.testing.assert_allclose(result.stream_erodep[0, 0, ICLAYSED], erodep_clay)
    np.testing.assert_allclose(result.stream_erodep[0, 0, IPOCA : IPOCP + 1], erodep_poc)
    np.testing.assert_allclose(result.stream_seddep[0, 0, ICLAYSED], seddep[0, 0, ICLAYSED] + erodep_clay)
    np.testing.assert_allclose(result.stream_seddep[0, 0, IPOCA : IPOCP + 1], seddep[0, 0, IPOCA : IPOCP + 1] + erodep_poc)
    np.testing.assert_allclose(result.stream_flow[0, 0, ICLAYSED], frac_out * (1.0 - erodep_clay))
    np.testing.assert_allclose(result.stream_reservoir[0, 0, ICLAYSED], (1.0 - frac_out) * (1.0 - erodep_clay))
    np.testing.assert_allclose(
        result.stream_flow[0, 0, IPOCA : IPOCP + 1],
        frac_out * (stream[0, 0, IPOCA : IPOCP + 1] - erodep_poc),
    )
    np.testing.assert_allclose(
        result.stream_reservoir[0, 0, IPOCA : IPOCP + 1],
        (1.0 - frac_out) * (stream[0, 0, IPOCA : IPOCP + 1] - erodep_poc),
    )
    np.testing.assert_allclose(result.stream_damavailbas, [[10.0 - erodep_clay]])


def test_routing_stream_erosion_step_erodes_banks_and_updates_soil_carbon():
    stream = np.zeros((1, 1, 10), dtype=np.float64)
    stream[0, 0, IH2O] = 2000.0
    stream[0, 0, IPOCA : IPOCP + 1] = [3.0, 4.0, 5.0]
    stream[0, 0, ICLAYSED] = 1.0
    flow = np.zeros_like(stream)
    flow[0, 0, IH2O] = 1000.0
    seddep = np.zeros_like(stream)
    seddep[0, 0, ICLAYSED] = 0.2
    seddep[0, 0, IPOCA : IPOCP + 1] = [0.3, 0.4, 0.5]
    carbon = np.zeros((1, 3, 2, 3), dtype=np.float64)
    carbon[0, IACTIVE, 1, :2] = [100.0, 200.0]
    carbon[0, ISLOW, 1, :2] = [300.0, 400.0]
    carbon[0, IPASSIVE, 1, :2] = [500.0, 600.0]

    result = routing_stream_erosion_step(
        stream_reservoir=stream,
        stream_flow=flow,
        stream_seddep=seddep,
        qflow_avebas=np.asarray([[1.0]], dtype=np.float64),
        basdrainarea=np.asarray([[1.0]], dtype=np.float64),
        basgravel=np.asarray([[0.0]], dtype=np.float64),
        stream_damavailbas=np.asarray([[10.0]], dtype=np.float64),
        routing_area=np.asarray([[10.0]], dtype=np.float64),
        bulkdens=np.asarray([1.0], dtype=np.float64),
        zz_deep=np.asarray([0.5, 2.0, 3.0], dtype=np.float64),
        veget_max=np.asarray([[0.0, 0.5]], dtype=np.float64),
        carbon_32l=carbon,
        dt_routing=1.0,
        coef_stc=(0.01, 0.0, 0.0),
        coef_redet=(0.5, 0.5, 0.5),
        coef_chanero=(0.001, 0.001, 0.001),
        maxfrac_bankero=1.0,
    )

    frac_out = 0.5
    cap = 10.0
    perosed = 0.5 * (frac_out * 1.0 - cap)
    shortage = perosed / frac_out + 0.2
    erodep_clay = -0.2 + 0.001 * shortage
    raw_area = -0.001 * shortage * 0.5 / 1.0 / 2.0
    raw_frac = raw_area / (0.5 * 10.0)
    area_rivero = raw_area * (raw_frac / (raw_frac + 1.0e-8))
    expected_poc_erodep = -seddep[0, 0, IPOCA : IPOCP + 1] - 1.0e-3 * np.asarray([300.0, 700.0, 1100.0]) * area_rivero

    np.testing.assert_allclose(result.stream_erodep[0, 0, ICLAYSED], erodep_clay)
    np.testing.assert_allclose(result.stream_erodep[0, 0, IPOCA : IPOCP + 1], expected_poc_erodep)
    np.testing.assert_allclose(result.stream_seddep[0, 0, ICLAYSED], 0.0)
    np.testing.assert_allclose(result.stream_seddep[0, 0, IPOCA : IPOCP + 1], 0.0)
    np.testing.assert_allclose(result.carbon_32l[0, IACTIVE, 1, :2], carbon[0, IACTIVE, 1, :2] * (1.0 - raw_frac))
    np.testing.assert_allclose(result.carbon_32l[0, ISLOW, 1, :2], carbon[0, ISLOW, 1, :2] * (1.0 - raw_frac))
    np.testing.assert_allclose(result.carbon_32l[0, IPASSIVE, 1, :2], carbon[0, IPASSIVE, 1, :2] * (1.0 - raw_frac))
    np.testing.assert_allclose(result.stream_flow[0, 0, ICLAYSED], frac_out * (1.0 - erodep_clay))
    np.testing.assert_allclose(result.stream_reservoir[0, 0, ICLAYSED], (1.0 - frac_out) * (1.0 - erodep_clay))


def _co2_state():
    shape3 = (1, 1, 10)
    return {
        "fast_reservoir": np.zeros(shape3, dtype=np.float64),
        "slow_reservoir": np.zeros(shape3, dtype=np.float64),
        "stream_reservoir": np.zeros(shape3, dtype=np.float64),
        "flood_reservoir": np.zeros(shape3, dtype=np.float64),
        "pond_reservoir": np.zeros((1, 10), dtype=np.float64),
        "stream_seddep": np.zeros(shape3, dtype=np.float64),
        "stream_erodep": np.zeros(shape3, dtype=np.float64),
        "flood_drainage": np.zeros(shape3, dtype=np.float64),
        "flood_dep_sed": np.zeros((1, 1, 3), dtype=np.float64),
        "flood_dep_poc": np.zeros((1, 1, 3), dtype=np.float64),
        "return_swamp": np.zeros(shape3, dtype=np.float64),
        "flood_inp_bas": np.zeros(shape3, dtype=np.float64),
        "stream_inp_bas": np.zeros(shape3, dtype=np.float64),
        "routing_area": np.asarray([[10.0]], dtype=np.float64),
        "flood_frac_bas": np.asarray([[0.5]], dtype=np.float64),
        "stream_area": np.asarray([5.0], dtype=np.float64),
        "stream_area_bas": np.asarray([[2.0]], dtype=np.float64),
        "pond_frac": np.asarray([0.25], dtype=np.float64),
        "temp_sol": np.asarray([273.15 + (28.0 - 6.13) / 0.8], dtype=np.float64),
        "swamp": np.asarray([0.0], dtype=np.float64),
        "floodplains": np.asarray([1.0], dtype=np.float64),
    }


def test_routing_poc_decomposition_matches_source_pool_transfers():
    frac = routing_poc_fraction_matrix(
        frac_carb_ap=0.004,
        frac_carb_sa=0.42,
        frac_carb_pa=0.45,
        metabolic_ref_frac=0.85,
    )
    result = routing_poc_decomposition(
        poc_reservoir=np.asarray([100.0, 50.0, 25.0], dtype=np.float64),
        poc_dec_rate=np.asarray([0.10, 0.20, 0.40], dtype=np.float64),
        frac_pocpool=frac,
        f_socdoc=0.03,
        cue=0.3,
    )

    poc_flux = np.asarray([10.0, 10.0, 10.0], dtype=np.float64)
    after_decay = np.asarray([100.0, 50.0, 25.0], dtype=np.float64) - poc_flux
    non_doc = poc_flux * 0.97
    expected = after_decay.copy()
    expected[IACTIVE] += frac[ISLOW, IACTIVE] * 0.3 * non_doc[ISLOW] + frac[IPASSIVE, IACTIVE] * 0.3 * non_doc[IPASSIVE]
    expected[ISLOW] += frac[IACTIVE, ISLOW] * 0.3 * non_doc[IACTIVE] + frac[IPASSIVE, ISLOW] * 0.3 * non_doc[IPASSIVE]
    expected[IPASSIVE] += frac[IACTIVE, IPASSIVE] * 0.3 * non_doc[IACTIVE] + frac[ISLOW, IPASSIVE] * 0.3 * non_doc[ISLOW]

    np.testing.assert_allclose(result.flux_poc2doc, [10.0 * 0.03, (10.0 + 10.0) * 0.03])
    np.testing.assert_allclose(result.flux_poc2co2, 0.7 * non_doc.sum())
    np.testing.assert_allclose(result.poc_reservoir, expected)


def test_routing_co2_chemistry_fast_slow_and_riverbed_branches():
    state = _co2_state()
    state["fast_reservoir"][0, 0, [IH2O, IDOCL, IDOCR, ICO2AQ]] = [1000.0, 10.0, 20.0, 5.0]
    state["fast_reservoir"][0, 0, IPOCA : IPOCP + 1] = [100.0, 50.0, 25.0]
    state["slow_reservoir"][0, 0, [IH2O, IDOCL, IDOCR, ICO2AQ]] = [1000.0, 10.0, 20.0, 5.0]
    state["stream_reservoir"][0, 0, IH2O] = 1000.0
    state["stream_seddep"][0, 0, IPOCA : IPOCP + 1] = [10.0, 20.0, 30.0]

    result = routing_co2_chemistry_step(
        **state,
        nstep_fco2=1,
        pco2_atm=0.0,
    )

    frac = routing_poc_fraction_matrix()
    fast_poc = routing_poc_decomposition(
        poc_reservoir=np.asarray([100.0, 50.0, 25.0], dtype=np.float64),
        poc_dec_rate=np.asarray([0.0090, 0.0025, 0.0090], dtype=np.float64),
        frac_pocpool=frac,
    )
    rivbed_poc = routing_poc_decomposition(
        poc_reservoir=np.asarray([10.0, 20.0, 30.0], dtype=np.float64),
        poc_dec_rate=0.56 * np.asarray([0.0090, 0.0025, 0.0090], dtype=np.float64),
        frac_pocpool=frac,
    )
    fast_doc_decomp = 0.3 * 10.0 + 0.01 * 20.0
    fast_store = 5.0 + fast_doc_decomp + fast_poc.flux_poc2co2
    slow_store = 5.0 + fast_doc_decomp

    np.testing.assert_allclose(result.fast_reservoir[0, 0, IDOCL], 0.7 * 10.0 + fast_poc.flux_poc2doc[IDOCLABILE])
    np.testing.assert_allclose(result.fast_reservoir[0, 0, IDOCR], 0.99 * 20.0 + fast_poc.flux_poc2doc[IDOCSTABLE])
    np.testing.assert_allclose(result.fast_reservoir[0, 0, ICO2AQ], 0.0, atol=1e-14)
    np.testing.assert_allclose(result.fco2_aq[0, IFASTR], fast_store)
    np.testing.assert_allclose(result.poc_co2_aq[0, IFASTR], fast_poc.flux_poc2co2)
    np.testing.assert_allclose(result.poc_doc_aq[0, IFASTR], fast_poc.flux_poc2doc.sum())

    np.testing.assert_allclose(result.slow_reservoir[0, 0, IDOCL], 0.7 * 10.0)
    np.testing.assert_allclose(result.slow_reservoir[0, 0, IDOCR], 0.99 * 20.0)
    np.testing.assert_allclose(result.slow_reservoir[0, 0, ICO2AQ], slow_store)
    np.testing.assert_allclose(result.stream_seddep[0, 0, IPOCA : IPOCP + 1], rivbed_poc.poc_reservoir)
    np.testing.assert_allclose(result.stream_reservoir[0, 0, IDOCL], rivbed_poc.flux_poc2doc[IDOCLABILE])
    np.testing.assert_allclose(result.stream_reservoir[0, 0, IDOCR], rivbed_poc.flux_poc2doc[IDOCSTABLE])
    np.testing.assert_allclose(result.stream_reservoir[0, 0, ICO2AQ], rivbed_poc.flux_poc2co2)
    np.testing.assert_allclose(result.stream_erodep[0, 0, IDOCL], -rivbed_poc.flux_poc2doc[IDOCLABILE])
    np.testing.assert_allclose(result.stream_erodep[0, 0, IDOCR], -rivbed_poc.flux_poc2doc[IDOCSTABLE])
    np.testing.assert_allclose(result.stream_erodep[0, 0, ICO2AQ], -rivbed_poc.flux_poc2co2)
    np.testing.assert_allclose(result.poc_co2_rivbed[0], rivbed_poc.flux_poc2co2)
    np.testing.assert_allclose(result.poc_doc_rivbed[0, :], rivbed_poc.flux_poc2doc)


def test_routing_co2_chemistry_flood_and_pond_segmented_evasion():
    state = _co2_state()
    state["flood_reservoir"][0, 0, [IH2O, IDOCL, IDOCR, ICO2AQ]] = [1000.0, 10.0, 20.0, 5.0]
    state["flood_reservoir"][0, 0, IPOCA : IPOCP + 1] = [30.0, 40.0, 50.0]
    state["flood_inp_bas"][0, 0, [IDOCL, IDOCR, ICO2AQ]] = [1.0, 2.0, 3.0]
    state["pond_reservoir"][0, [IH2O, IDOCL, IDOCR, ICO2AQ]] = [1000.0, 10.0, 20.0, 5.0]
    state["pond_reservoir"][0, IPOCA : IPOCP + 1] = [30.0, 40.0, 50.0]

    result = routing_co2_chemistry_step(
        **state,
        nstep_fco2=1,
        pco2_atm=0.0,
    )

    frac = routing_poc_fraction_matrix()
    poc = routing_poc_decomposition(
        poc_reservoir=np.asarray([30.0, 40.0, 50.0], dtype=np.float64),
        poc_dec_rate=np.asarray([0.0090, 0.0025, 0.0090], dtype=np.float64),
        frac_pocpool=frac,
    )
    flood_doc_decomp = 0.3 * 11.0 + 0.01 * 22.0 + 3.0
    pond_doc_decomp = 0.3 * 10.0 + 0.01 * 20.0

    np.testing.assert_allclose(result.fco2_aq[0, IFLOODR], 5.0)
    np.testing.assert_allclose(result.flood_reservoir[0, 0, ICO2AQ], flood_doc_decomp + poc.flux_poc2co2)
    np.testing.assert_allclose(result.flood_reservoir[0, 0, IDOCL], 0.7 * 11.0 + poc.flux_poc2doc[IDOCLABILE])
    np.testing.assert_allclose(result.flood_reservoir[0, 0, IDOCR], 0.99 * 22.0 + poc.flux_poc2doc[IDOCSTABLE])
    np.testing.assert_allclose(result.poc_co2_aq[0, IFLOODR], poc.flux_poc2co2)
    np.testing.assert_allclose(result.poc_doc_aq[0, IFLOODR], poc.flux_poc2doc.sum())

    np.testing.assert_allclose(result.fco2_aq[0, IPONDR], 5.0)
    np.testing.assert_allclose(result.pond_reservoir[0, ICO2AQ], pond_doc_decomp + poc.flux_poc2co2)
    np.testing.assert_allclose(result.pond_reservoir[0, IDOCL], 0.7 * 10.0 + poc.flux_poc2doc[IDOCLABILE])
    np.testing.assert_allclose(result.pond_reservoir[0, IDOCR], 0.99 * 20.0 + poc.flux_poc2doc[IDOCSTABLE])
    np.testing.assert_allclose(result.poc_co2_aq[0, IPONDR], poc.flux_poc2co2)
    np.testing.assert_allclose(result.poc_doc_aq[0, IPONDR], poc.flux_poc2doc.sum())


def test_routing_co2_chemistry_inactive_flood_and_stream_return_stores():
    state = _co2_state()
    state["stream_area"][:] = 0.0
    state["flood_reservoir"][0, 0, :] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    state["flood_inp_bas"][0, 0, ICO2AQ] = 10.0
    state["stream_reservoir"][0, 0, :] = [100.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    state["stream_inp_bas"][0, 0, [IDOCL, IDOCR, ICO2AQ]] = [1.0, 2.0, 20.0]

    result = routing_co2_chemistry_step(
        **state,
        nstep_fco2=1,
        doswamps=True,
    )

    np.testing.assert_allclose(result.flood_reservoir, 0.0)
    np.testing.assert_allclose(result.flood_drainage[0, 0, IH2O : ICO2AQ + 1], [0.0, 1.0, 2.0, 13.0])
    np.testing.assert_allclose(result.flood_dep_poc[0, 0, :], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(result.flood_dep_sed[0, 0, :], [7.0, 8.0, 9.0])
    np.testing.assert_allclose(result.stream_reservoir, 0.0)
    np.testing.assert_allclose(result.return_swamp[0, 0, IH2O : ICO2AQ + 1], [100.0, 12.0, 14.0, 33.0])
    np.testing.assert_allclose(result.stream_erodep[0, 0, IPOCA : ISANDSED + 1], [14.0, 15.0, 16.0, 17.0, 18.0, 19.0])


def test_routing_irrigation_step_disabled_leaves_reservoirs_and_fluxes_zero():
    fast = np.ones((1, 1, 4), dtype=np.float64)
    slow = 2.0 * fast
    stream = 3.0 * fast

    result = routing_irrigation_step(
        fast_reservoir=fast,
        slow_reservoir=slow,
        stream_reservoir=stream,
        routing_area=np.asarray([[1.0]], dtype=np.float64),
        irrigated=np.asarray([1.0], dtype=np.float64),
        vegtot=np.asarray([1.0], dtype=np.float64),
        humrel=np.asarray([0.0], dtype=np.float64),
        runoff=np.zeros((1, 4), dtype=np.float64),
        precip=np.zeros((1, 4), dtype=np.float64),
        reinfiltration=np.zeros((1, 4), dtype=np.float64),
        transpot_mean=np.asarray([10.0], dtype=np.float64),
        do_irrigation=False,
    )

    np.testing.assert_allclose(result.fast_reservoir, fast)
    np.testing.assert_allclose(result.slow_reservoir, slow)
    np.testing.assert_allclose(result.stream_reservoir, stream)
    np.testing.assert_allclose(result.irrig_netereq, 0.0)
    np.testing.assert_allclose(result.irrig_actual, 0.0)
    np.testing.assert_allclose(result.irrig_adduct, 0.0)


def test_routing_irrigation_step_withdraws_stream_fast_slow_and_scales_matter():
    fast = np.zeros((1, 1, 4), dtype=np.float64)
    slow = np.zeros_like(fast)
    stream = np.zeros_like(fast)
    stream[0, 0, :] = [50.0, 5.0, 10.0, 15.0]
    fast[0, 0, :] = [40.0, 8.0, 12.0, 16.0]
    slow[0, 0, :] = [100.0, 20.0, 30.0, 40.0]

    result = routing_irrigation_step(
        fast_reservoir=fast,
        slow_reservoir=slow,
        stream_reservoir=stream,
        routing_area=np.asarray([[1.0]], dtype=np.float64),
        irrigated=np.asarray([1.0], dtype=np.float64),
        vegtot=np.asarray([1.0], dtype=np.float64),
        humrel=np.asarray([0.5], dtype=np.float64),
        runoff=np.zeros((1, 4), dtype=np.float64),
        precip=np.zeros((1, 4), dtype=np.float64),
        reinfiltration=np.zeros((1, 4), dtype=np.float64),
        transpot_mean=np.asarray([120.0], dtype=np.float64),
        do_irrigation=True,
        ok_doc=True,
    )

    expected_slow = slow.copy()
    expected_slow[0, 0, IH2O] = 70.0
    expected_slow[0, 0, IH2O + 1 :] = 70.0 * slow[0, 0, IH2O + 1 :] / 100.0
    np.testing.assert_allclose(result.irrig_netereq, [120.0])
    np.testing.assert_allclose(result.irrig_needs, [[120.0]])
    np.testing.assert_allclose(result.irrig_actual[0, 0, IH2O], 120.0)
    np.testing.assert_allclose(result.stream_reservoir, 0.0)
    np.testing.assert_allclose(result.fast_reservoir, 0.0)
    np.testing.assert_allclose(result.slow_reservoir, expected_slow)
    np.testing.assert_allclose(result.irrig_actual[0, 0, IH2O + 1 :], (stream + fast + slow - expected_slow)[0, 0, IH2O + 1 :])
    np.testing.assert_allclose(result.irrig_deficit, 0.0)


def test_routing_irrigation_step_same_grid_adduction_uses_fullest_stream_basin():
    fast = np.zeros((1, 2, 4), dtype=np.float64)
    slow = np.zeros_like(fast)
    stream = np.zeros_like(fast)
    stream[0, 1, :] = [80.0, 8.0, 16.0, 24.0]

    result = routing_irrigation_step(
        fast_reservoir=fast,
        slow_reservoir=slow,
        stream_reservoir=stream,
        routing_area=np.asarray([[1.0, 0.0]], dtype=np.float64),
        irrigated=np.asarray([1.0], dtype=np.float64),
        vegtot=np.asarray([1.0], dtype=np.float64),
        humrel=np.asarray([0.5], dtype=np.float64),
        runoff=np.zeros((1, 4), dtype=np.float64),
        precip=np.zeros((1, 4), dtype=np.float64),
        reinfiltration=np.zeros((1, 4), dtype=np.float64),
        transpot_mean=np.asarray([100.0], dtype=np.float64),
        do_irrigation=True,
        ok_doc=True,
    )

    np.testing.assert_allclose(result.irrig_actual, 0.0)
    np.testing.assert_allclose(result.irrig_adduct[0, 0, :], [80.0, 8.0, 16.0, 24.0])
    np.testing.assert_allclose(result.irrig_deficit[0, 0], 20.0)
    np.testing.assert_allclose(result.stream_reservoir[0, 1, :], 0.0)


def test_routing_irrigation_step_neighbor_adduction_adds_remote_stream_water():
    fast = np.zeros((2, 1, 4), dtype=np.float64)
    slow = np.zeros_like(fast)
    stream = np.zeros_like(fast)
    stream[1, 0, :] = [70.0, 7.0, 14.0, 21.0]

    result = routing_irrigation_step(
        fast_reservoir=fast,
        slow_reservoir=slow,
        stream_reservoir=stream,
        routing_area=np.asarray([[1.0], [1.0]], dtype=np.float64),
        irrigated=np.asarray([1.0, 0.0], dtype=np.float64),
        vegtot=np.asarray([1.0, 1.0], dtype=np.float64),
        humrel=np.asarray([0.5, 0.5], dtype=np.float64),
        runoff=np.zeros((2, 4), dtype=np.float64),
        precip=np.zeros((2, 4), dtype=np.float64),
        reinfiltration=np.zeros((2, 4), dtype=np.float64),
        transpot_mean=np.asarray([100.0, 0.0], dtype=np.float64),
        do_irrigation=True,
        ok_doc=True,
        resolution=np.asarray([[50000.0, 50000.0], [50000.0, 50000.0]], dtype=np.float64),
        neighbours=np.asarray([[1], [0]], dtype=np.int32),
    )

    np.testing.assert_allclose(result.irrig_adduct[0, 0, :], [70.0, 7.0, 14.0, 21.0])
    np.testing.assert_allclose(result.irrig_deficit[0, 0], 30.0)
    np.testing.assert_allclose(result.stream_reservoir[1, 0, :], 0.0)


def test_routing_setvar_no_keyword_replaces_only_all_missing_arrays():
    val_exp = 999999.0
    np.testing.assert_allclose(
        routing_setvar_no_keyword(np.full((2,), val_exp), 3.0, val_exp=val_exp),
        [3.0, 3.0],
    )
    partial = np.asarray([val_exp, 4.0], dtype=np.float64)
    np.testing.assert_allclose(
        routing_setvar_no_keyword(partial, 3.0, val_exp=val_exp),
        partial,
    )


def test_routing_init_restart_defaults_match_source_cold_start_groups():
    val_exp = 999999.0
    state = routing_init_restart_defaults(
        None,
        npts=1,
        nbas=2,
        nflow=10,
        routing_area=np.asarray([[10.0, 30.0]], dtype=np.float64),
        doirrigation=False,
        val_exp=val_exp,
        undef_sechiba=1.0e20,
    )

    np.testing.assert_allclose(state["fast_reservoir"], 0.0)
    np.testing.assert_allclose(state["stream_seddep"], 0.0)
    np.testing.assert_allclose(state["irrigation_mean"], 0.0)
    np.testing.assert_allclose(state["drainage_mean"][:, ICO2AQ + 1 :], 0.0)
    np.testing.assert_allclose(state["precip_mean"][:, ICO2AQ + 1 :], 0.0)
    np.testing.assert_allclose(state["humrel_mean"], [1.0])
    np.testing.assert_allclose(state["vegtot_mean"], [1.0])
    np.testing.assert_allclose(state["irrigated"], [1.0e20])
    np.testing.assert_allclose(state["stream_area"], [1.0e20])
    np.testing.assert_allclose(state["qflow_ave"], [1.0e20])
    np.testing.assert_allclose(state["fco2_aq"], 0.0)
    np.testing.assert_allclose(state["flow_input"], 0.0)


def test_routing_init_restart_defaults_preserve_restart_values_and_derive_diagnostics():
    fast = np.zeros((1, 2, 10), dtype=np.float64)
    flood = np.zeros_like(fast)
    lake = np.zeros((1, 10), dtype=np.float64)
    stream_seddep = np.full((1, 2, 10), 9.0, dtype=np.float64)
    fast[0, :, IH2O] = [10.0, 30.0]
    flood[0, :, IH2O] = [2.0, 6.0]
    lake[0, IH2O] = 80.0
    stream_seddep[0, :, IH2O : ICO2AQ + 1] = [1.0, 2.0, 3.0, 4.0]
    restart = {
        "fast_reservoir": fast,
        "flood_reservoir": flood,
        "lake_reservoir": lake,
        "stream_seddep": stream_seddep,
        "irrigation_mean": np.ones((1, 10), dtype=np.float64),
        "drainage_mean": np.ones((1, 10), dtype=np.float64),
        "precip_mean": 2.0 * np.ones((1, 10), dtype=np.float64),
        "irrigated": np.asarray([12.0], dtype=np.float64),
    }

    state = routing_init_restart_defaults(
        restart,
        npts=1,
        nbas=2,
        nflow=10,
        routing_area=np.asarray([[10.0, 30.0]], dtype=np.float64),
        doirrigation=True,
    )

    np.testing.assert_allclose(state["fast_reservoir"], fast)
    np.testing.assert_allclose(state["flood_reservoir"], flood)
    np.testing.assert_allclose(state["lake_reservoir"], lake)
    np.testing.assert_allclose(state["stream_seddep"][0, :, IH2O : ICO2AQ + 1], 0.0)
    np.testing.assert_allclose(state["stream_seddep"][0, :, IPOCA:], 9.0)
    np.testing.assert_allclose(state["irrigation_mean"], 1.0)
    np.testing.assert_allclose(state["drainage_mean"][:, : ICO2AQ + 1], 1.0)
    np.testing.assert_allclose(state["drainage_mean"][:, ICO2AQ + 1 :], 0.0)
    np.testing.assert_allclose(state["precip_mean"][:, : ICO2AQ + 1], 2.0)
    np.testing.assert_allclose(state["precip_mean"][:, ICO2AQ + 1 :], 0.0)
    np.testing.assert_allclose(state["irrigated"], [12.0])
    np.testing.assert_allclose(state["fast_diag"][0, IH2O], 1.0)
    np.testing.assert_allclose(state["flood_diag"][0, IH2O], 0.2)
    np.testing.assert_allclose(state["lake_diag"][0, IH2O], 2.0)


def test_routing_initialize_map_flags_match_missing_restart_rules():
    undef = 1.0e20

    flags = routing_initialize_map_flags(
        irrigated=np.asarray([undef], dtype=np.float64),
        floodplains=np.asarray([10.0], dtype=np.float64),
        swamp=np.asarray([undef], dtype=np.float64),
        headw_area=np.asarray([2.0], dtype=np.float64),
        stream_area=np.asarray([3.0], dtype=np.float64),
        streamr50th=np.asarray([0.0], dtype=np.float64),
        do_irrigation=True,
        do_floodplains=True,
        doswamps=True,
        ok_doc=False,
        undef_sechiba=undef,
    )

    assert flags.init_irrig
    assert flags.init_flood
    assert flags.init_swamp
    assert flags.init_streamsurf

    flags = routing_initialize_map_flags(
        irrigated=np.asarray([undef], dtype=np.float64),
        floodplains=np.asarray([undef], dtype=np.float64),
        swamp=np.asarray([undef], dtype=np.float64),
        headw_area=np.asarray([undef], dtype=np.float64),
        stream_area=np.asarray([undef], dtype=np.float64),
        streamr50th=np.asarray([5.0], dtype=np.float64),
        do_irrigation=False,
        do_floodplains=False,
        doswamps=False,
        ok_doc=True,
        undef_sechiba=undef,
    )

    assert not flags.init_irrig
    assert not flags.init_flood
    assert not flags.init_swamp
    assert flags.init_streamsurf


def test_routing_irrigmap_preprocess_thresholds_percent_maps_and_preserves_missing():
    undef = 1.0e20
    flood = np.zeros((1, 2, 6), dtype=np.float64)
    flood[0, 0, :] = [0.005, 0.02, undef, 1.0, 0.009, 50.0]
    flood[0, 1, :] = [0.02, 0.009, 1.0, 2.0, 3.0, 4.0]

    result = routing_irrigmap_preprocess(
        irrigated_frac=np.asarray([[0.40, 2.0]], dtype=np.float64),
        flood_fracmax=flood,
        headw_frac=np.asarray([[0.005, undef]], dtype=np.float64),
        stream_frac=np.asarray([[0.02, 1.0]], dtype=np.float64),
        undef_sechiba=undef,
    )

    np.testing.assert_allclose(result.irrigated_frac, [[0.0, 0.02]])
    np.testing.assert_allclose(result.flood_fracmax[0, 0, 0], 0.0)
    np.testing.assert_allclose(result.flood_fracmax[0, 0, 1], 0.0002)
    np.testing.assert_allclose(result.flood_fracmax[0, 0, 2], undef)
    np.testing.assert_allclose(result.flood_fracmax[0, 1, 1], 0.0)
    np.testing.assert_allclose(result.headw_frac, [[0.0001, undef]])
    np.testing.assert_allclose(result.stream_frac, [[0.0002, 0.01]])


def test_routing_irrigmap_aggregate_matches_source_flags_and_area_rules():
    maps = routing_irrigmap_preprocess(
        irrigated_frac=np.asarray([[50.0, 1.0], [0.1, 25.0]], dtype=np.float64),
        flood_fracmax=np.asarray(
            [
                [
                    [0.0, 10.0, 20.0, 5.0, 30.0, 0.0],
                    [0.0, 1.0, 2.0, 40.0, 3.0, 0.0],
                ],
                [
                    [0.0, 7.0, 8.0, 9.0, 10.0, 0.0],
                    [0.0, 11.0, 12.0, 13.0, 14.0, 0.0],
                ],
            ],
            dtype=np.float64,
        ),
        headw_frac=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        stream_frac=np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64),
    )

    result = routing_irrigmap_aggregate(
        irrsub_area=np.asarray([[100.0, 50.0, 0.0]], dtype=np.float64),
        irrsub_index=np.asarray([[[1, 1], [2, 2], [1, 2]]], dtype=np.int32),
        resolution=np.asarray([[10.0, 10.0]], dtype=np.float64),
        contfrac=np.asarray([1.0], dtype=np.float64),
        irrigated_frac=maps.irrigated_frac,
        flood_fracmax=maps.flood_fracmax,
        headw_frac=maps.headw_frac,
        stream_frac=maps.stream_frac,
        streamr10th_mm=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        streamr50th_mm=np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
        streamr90th_mm=np.asarray([[100.0, 200.0], [300.0, 400.0]], dtype=np.float64),
        qflow_aveb=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        stream_resaveb=np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64),
        floodh90th_mm=np.asarray([[50.0, 100.0], [150.0, 200.0]], dtype=np.float64),
        init_irrig=True,
        init_flood=True,
        init_swamp=True,
        init_streamsurf=True,
        dostreamswell=True,
        new_flood_scheme=True,
        ok_damreservoir=True,
        streamfld_scaler=2.0,
        floodcri=25.0,
        min_sechiba=0.001,
    )

    np.testing.assert_allclose(result.irrigated, [62.5])
    np.testing.assert_allclose(result.floodplains, [78.5])
    np.testing.assert_allclose(result.swamp, [11.5])
    np.testing.assert_allclose(result.headw_area, [3.0])
    np.testing.assert_allclose(result.stream_area, [9.0])
    np.testing.assert_allclose(result.streamr10th, [300.0])
    np.testing.assert_allclose(result.streamr50th, [6000.0])
    np.testing.assert_allclose(result.streamr90th, [60000.0])
    np.testing.assert_allclose(result.qflow_ave, [5.0])
    np.testing.assert_allclose(result.stream_resave, [13.0])
    np.testing.assert_allclose(result.floodh90th, [150.0])


def test_routing_irrigmap_aggregate_zeroes_negative_subarea_before_source_order_count():
    result = routing_irrigmap_aggregate(
        irrsub_area=np.asarray([[100.0, -5.0, 50.0]], dtype=np.float64),
        irrsub_index=np.asarray([[[1, 1], [1, 2], [1, 3]]], dtype=np.int32),
        resolution=np.asarray([[100.0, 100.0]], dtype=np.float64),
        contfrac=np.asarray([1.0], dtype=np.float64),
        irrigated_frac=np.asarray([[0.1, 1000.0, 10.0]], dtype=np.float64),
        flood_fracmax=np.zeros((1, 3, 6), dtype=np.float64),
        headw_frac=np.zeros((1, 3), dtype=np.float64),
        stream_frac=np.zeros((1, 3), dtype=np.float64),
        streamr10th_mm=np.zeros((1, 3), dtype=np.float64),
        streamr50th_mm=np.zeros((1, 3), dtype=np.float64),
        streamr90th_mm=np.zeros((1, 3), dtype=np.float64),
        qflow_aveb=np.zeros((1, 3), dtype=np.float64),
        stream_resaveb=np.zeros((1, 3), dtype=np.float64),
        floodh90th_mm=np.zeros((1, 3), dtype=np.float64),
        init_irrig=True,
        init_flood=False,
        init_swamp=False,
        init_streamsurf=False,
        dostreamswell=False,
        new_flood_scheme=False,
        ok_damreservoir=False,
    )

    np.testing.assert_allclose(result.irrigated, [10.0])


def test_routing_irrigmap_aggregate_preserves_noninitialized_outputs_and_dam_switch():
    old = np.asarray([9.0], dtype=np.float64)
    result = routing_irrigmap_aggregate(
        irrsub_area=np.asarray([[1.0]], dtype=np.float64),
        irrsub_index=np.asarray([[[1, 1]]], dtype=np.int32),
        resolution=np.asarray([[10.0, 10.0]], dtype=np.float64),
        contfrac=np.asarray([1.0], dtype=np.float64),
        irrigated_frac=np.asarray([[0.5]], dtype=np.float64),
        flood_fracmax=np.zeros((1, 1, 6), dtype=np.float64),
        headw_frac=np.asarray([[0.01]], dtype=np.float64),
        stream_frac=np.asarray([[0.02]], dtype=np.float64),
        streamr10th_mm=np.asarray([[1.0]], dtype=np.float64),
        streamr50th_mm=np.asarray([[2.0]], dtype=np.float64),
        streamr90th_mm=np.asarray([[3.0]], dtype=np.float64),
        qflow_aveb=np.asarray([[4.0]], dtype=np.float64),
        stream_resaveb=np.asarray([[5.0]], dtype=np.float64),
        floodh90th_mm=np.asarray([[6.0]], dtype=np.float64),
        init_irrig=False,
        init_flood=False,
        init_swamp=False,
        init_streamsurf=True,
        dostreamswell=False,
        new_flood_scheme=False,
        ok_damreservoir=False,
        irrigated=old,
        floodplains=old,
        swamp=old,
        streamr10th=old,
        streamr50th=old,
        streamr90th=old,
        stream_resave=old,
        floodh90th=old,
        min_sechiba=0.001,
    )

    np.testing.assert_allclose(result.irrigated, [9.0])
    np.testing.assert_allclose(result.swamp, [9.0])
    np.testing.assert_allclose(result.headw_area, [0.01])
    np.testing.assert_allclose(result.stream_area, [0.02])
    np.testing.assert_allclose(result.floodplains, [9.0])
    np.testing.assert_allclose(result.streamr10th, [0.0])
    np.testing.assert_allclose(result.streamr50th, [0.0])
    np.testing.assert_allclose(result.streamr90th, [0.0])
    np.testing.assert_allclose(result.stream_resave, [1.0])
    np.testing.assert_allclose(result.floodh90th, [9.0])


def test_routing_initialize_stream_fraction_uses_stream_area_only_and_caps():
    result = routing_initialize_stream_fraction(
        stream_area=np.asarray([5.0, 100.0, 3.0], dtype=np.float64),
        routing_area=np.asarray([[10.0, 30.0], [10.0, 20.0], [0.0, 0.0]], dtype=np.float64),
    )

    np.testing.assert_allclose(result, [0.125, 1.0, 0.0])


def test_read_routing_map_fields_requires_source_variables_and_transposes_to_fortran_order(tmp_path):
    path = tmp_path / "routing.nc"
    base = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    xr.Dataset(
        data_vars={
            "nav_lon": (("y", "x"), base + 10.0),
            "nav_lat": (("y", "x"), base + 20.0),
            "trip": (("y", "x"), base + 30.0),
            "basins": (("y", "x"), base + 40.0),
            "topoind": (("y", "x"), base + 50.0),
            "gravel": (("y", "x"), base + 60.0),
            "drainage_area": (("y", "x"), base + 70.0),
        }
    ).to_netcdf(path)

    result = read_routing_map_fields(path)

    np.testing.assert_allclose(result.lon_rel, (base + 10.0).T)
    np.testing.assert_allclose(result.lat_rel, (base + 20.0).T)
    np.testing.assert_allclose(result.trip, (base + 30.0).T)
    np.testing.assert_allclose(result.basins, (base + 40.0).T)
    np.testing.assert_allclose(result.topoindex, (base + 50.0).T)
    np.testing.assert_allclose(result.gravel, (base + 60.0).T)
    np.testing.assert_allclose(result.drainarea, (base + 70.0).T)


def test_read_routing_map_fields_refuses_missing_fortran_variables(tmp_path):
    path = tmp_path / "routing.nc"
    base = np.ones((1, 1), dtype=np.float32)
    xr.Dataset(
        data_vars={
            "nav_lon": (("y", "x"), base),
            "nav_lat": (("y", "x"), base),
            "trip": (("y", "x"), base),
            "basins": (("y", "x"), base),
            "topoind": (("y", "x"), base),
        }
    ).to_netcdf(path)

    with pytest.raises(ValueError, match="gravel, drainage_area"):
        read_routing_map_fields(path)


def test_routing_map_fields_for_domain_reads_only_when_global_routing_branch_active(tmp_path):
    missing_path = tmp_path / "missing-routing.nc"

    assert routing_map_fields_for_domain(river_routing=True, nbp_glo=1, routing_file=missing_path) is None
    assert routing_map_fields_for_domain(river_routing=False, nbp_glo=3, routing_file=missing_path) is None

    path = tmp_path / "routing.nc"
    base = np.ones((1, 1), dtype=np.float32)
    xr.Dataset(
        data_vars={
            "nav_lon": (("y", "x"), base),
            "nav_lat": (("y", "x"), base),
            "trip": (("y", "x"), base),
            "basins": (("y", "x"), base),
            "topoind": (("y", "x"), base),
        }
    ).to_netcdf(path)

    with pytest.raises(ValueError, match="gravel, drainage_area"):
        routing_map_fields_for_domain(river_routing=True, nbp_glo=2, routing_file=path)


def test_routing_sortcoord_compresses_duplicates_and_uses_periodic_longitudes():
    result = routing_sortcoord(np.asarray([170.0, -170.0, 170.0, 10.0], dtype=np.float64), "WE")

    assert result.nb_out == 3
    np.testing.assert_allclose(result.coords, [10.0, 170.0, -170.0, 0.0])

    result = routing_sortcoord(np.asarray([0.0, 2.0, 1.0, 2.0], dtype=np.float64), "NS")

    assert result.nb_out == 3
    np.testing.assert_allclose(result.coords, [2.0, 1.0, 0.0, 0.0])


def test_routing_hierarchy_accumulates_topo_and_wraps_longitude_like_source():
    result = routing_hierarchy(
        trip=np.asarray([[99.0], [1.0e20], [98.0], [98.0], [3.0]], dtype=np.float64),
        topoindex=np.asarray([[1.0], [999.0], [4.0], [8.0], [5.0]], dtype=np.float64),
    )

    np.testing.assert_allclose(result[[0, 2, 3, 4], 0], [1.0, 4.0, 8.0, 6.0])
    assert result[1, 0] == 1.0e20


def test_routing_hierarchy_refuses_unrouted_cycle():
    with pytest.raises(ValueError, match="could not route point"):
        routing_hierarchy(
            trip=np.asarray([[3.0], [3.0], [3.0]], dtype=np.float64),
            topoindex=np.asarray([[1.0], [2.0], [3.0]], dtype=np.float64),
        )


def test_routing_getgrid_from_subgrid_orders_cells_and_tags_boundary_outflows():
    lon_rel = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    lat_rel = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    trip = np.asarray([[2.0, 8.0], [4.0, 6.0]], dtype=np.float64)
    basins = np.asarray([[11.0, 12.0], [13.0, 14.0]], dtype=np.float64)
    topo = np.asarray([[101.0, 102.0], [103.0, 104.0]], dtype=np.float64)
    gravel = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    drainarea = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    hierarchy = np.asarray([[1001.0, 1002.0], [1003.0, 1004.0]], dtype=np.float64)

    result = routing_getgrid_from_subgrid(
        ib=1,
        sub_pts=np.asarray([4], dtype=np.int32),
        sub_index=np.asarray([[[2, 2], [1, 1], [1, 2], [2, 1]]], dtype=np.int32),
        sub_area=np.asarray([[4.0, 1.0, 2.0, 3.0]], dtype=np.float64),
        lon_rel=lon_rel,
        lat_rel=lat_rel,
        lalo=np.asarray([[0.5, 0.5]], dtype=np.float64),
        resolution=np.asarray([[10.0, 10.0]], dtype=np.float64),
        contfrac=np.asarray([1.0], dtype=np.float64),
        trip=trip,
        basins=basins,
        topoindex=topo,
        gravel=gravel,
        drainarea=drainarea,
        hierarchy=hierarchy,
        max_basins=20.0,
        min_topoind=5.0,
    )

    assert result.nbi == 2
    assert result.nbj == 2
    np.testing.assert_array_equal(result.trip_bx, [[101, 107], [103, 105]])
    np.testing.assert_array_equal(result.basin_bx, [[11, 12], [13, 14]])
    np.testing.assert_allclose(result.area_bx, [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(result.topoind_bx, [[101.0, 102.0], [103.0, 104.0]])
    np.testing.assert_allclose(result.gravel_bx, [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(result.drainarea_bx, [[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_allclose(result.hierarchy_bx, [[1001.0, 1002.0], [1003.0, 1004.0]])
    np.testing.assert_allclose(result.lon_bx, [[0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(result.lat_bx, [[1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(result.max_basins, 20.0)


def test_routing_getgrid_from_subgrid_invents_coastal_cell_when_no_source_points():
    result = routing_getgrid_from_subgrid(
        ib=1,
        sub_pts=np.asarray([0], dtype=np.int32),
        sub_index=np.zeros((1, 1, 2), dtype=np.int32),
        sub_area=np.zeros((1, 1), dtype=np.float64),
        lon_rel=np.zeros((1, 1), dtype=np.float64),
        lat_rel=np.zeros((1, 1), dtype=np.float64),
        lalo=np.asarray([[12.0, 34.0]], dtype=np.float64),
        resolution=np.asarray([[10.0, 20.0]], dtype=np.float64),
        contfrac=np.asarray([0.5], dtype=np.float64),
        trip=np.zeros((1, 1), dtype=np.float64),
        basins=np.zeros((1, 1), dtype=np.float64),
        topoindex=np.zeros((1, 1), dtype=np.float64),
        gravel=np.zeros((1, 1), dtype=np.float64),
        drainarea=np.zeros((1, 1), dtype=np.float64),
        hierarchy=np.zeros((1, 1), dtype=np.float64),
        max_basins=100.0,
        min_topoind=7.0,
    )

    assert result.nbi == 1
    assert result.nbj == 1
    np.testing.assert_array_equal(result.trip_bx, [[98]])
    np.testing.assert_array_equal(result.basin_bx, [[101]])
    np.testing.assert_allclose(result.area_bx, [[100.0]])
    np.testing.assert_allclose(result.topoind_bx, [[7.0]])
    np.testing.assert_allclose(result.gravel_bx, [[0.0]])
    np.testing.assert_allclose(result.drainarea_bx, [[0.0]])
    np.testing.assert_allclose(result.hierarchy_bx, [[7.0]])
    np.testing.assert_allclose(result.lon_bx, [[34.0]])
    np.testing.assert_allclose(result.lat_bx, [[12.0]])
    np.testing.assert_allclose(result.max_basins, 101.0)


def test_routing_findbasins_simple_collects_single_outflow_basins_by_size():
    result = routing_findbasins_simple(
        trip=np.asarray([[3, 103], [99, 5]], dtype=np.int64),
        basin=np.asarray([[1, 1], [2, 2]], dtype=np.int64),
    )

    assert result.nb_basin == 2
    np.testing.assert_array_equal(result.basin_inbxid, [1, 2])
    np.testing.assert_array_equal(result.basin_sz, [2, 2])
    np.testing.assert_array_equal(result.basin_bxout, [3, -1])
    np.testing.assert_array_equal(result.basin_pts[0, :2, :], [[1, 1], [1, 2]])
    np.testing.assert_array_equal(result.basin_pts[1, :2, :], [[2, 1], [2, 2]])
    np.testing.assert_array_equal(result.trip, [[3, 103], [99, 5]])


def test_routing_findbasins_simple_merges_singleton_ocean_points_into_coastal_basin():
    result = routing_findbasins_simple(
        trip=np.asarray([[99], [99]], dtype=np.int64),
        basin=np.asarray([[10], [20]], dtype=np.int64),
    )

    assert result.nb_basin == 1
    np.testing.assert_array_equal(result.trip, [[98], [98]])
    np.testing.assert_array_equal(result.basin_inbxid, [-1])
    np.testing.assert_array_equal(result.basin_sz, [2])
    np.testing.assert_array_equal(result.basin_bxout, [-2])
    np.testing.assert_array_equal(result.basin_pts[0, :2, :], [[1, 1], [2, 1]])
    np.testing.assert_array_equal(result.coast_pts[:2], [10, 20])


def test_routing_findbasins_simple_refuses_unimplemented_multi_outflow_topology():
    with pytest.raises(NotImplementedError, match="routing_simplify"):
        routing_findbasins_simple(
            trip=np.asarray([[101, 103]], dtype=np.int64),
            basin=np.asarray([[1, 1]], dtype=np.int64),
        )


def test_routing_findrout_follows_trip_paths_to_each_outflow():
    result = routing_findrout(
        trip=np.asarray([[5, 99], [5, 103]], dtype=np.int64),
        basin_sz=4,
        basinid=77,
    )

    assert result.nbout == 2
    np.testing.assert_array_equal(result.outflow, [[1, 2], [2, 2]])
    np.testing.assert_array_equal(result.outsz, [2, 2])
    np.testing.assert_array_equal(
        result.trip_flow[:, :, 0],
        [[1, 1], [2, 2]],
    )
    np.testing.assert_array_equal(
        result.trip_flow[:, :, 1],
        [[2, 2], [2, 2]],
    )


def test_routing_findrout_refuses_cycles_and_size_mismatch():
    with pytest.raises(ValueError, match="could not route"):
        routing_findrout(
            trip=np.asarray([[3], [7]], dtype=np.int64),
            basin_sz=2,
            basinid=1,
        )

    with pytest.raises(ValueError, match="water got lost"):
        routing_findrout(
            trip=np.asarray([[99, 0]], dtype=np.int64),
            basin_sz=2,
            basinid=1,
        )


def _globalize_blank_state(npts=1, nwbas=3):
    return dict(
        basin_count=np.zeros(npts, dtype=np.int64),
        basin_area=np.zeros((npts, nwbas), dtype=np.float64),
        basin_hierarchy=np.zeros((npts, nwbas), dtype=np.float64),
        basin_topoind=np.zeros((npts, nwbas), dtype=np.float64),
        basin_gravel=np.zeros((npts, nwbas), dtype=np.float64),
        basin_drainarea=np.zeros((npts, nwbas), dtype=np.float64),
        basin_id=np.zeros((npts, nwbas), dtype=np.int64),
        basin_flowdir=np.zeros((npts, nwbas), dtype=np.int64),
        outflow_grid=np.zeros((npts, nwbas), dtype=np.int64),
        nbcoastal=np.zeros(npts, dtype=np.int64),
        coastal_basin=np.zeros((npts, nwbas), dtype=np.int64),
    )


def test_routing_globalize_one_grid_sums_basin_fields_and_maps_positive_outflow():
    state = _globalize_blank_state()
    result = routing_globalize_one_grid(
        ib=1,
        neighbours=np.asarray([[10, 11, 12, 13, 14, 15, 16, 17]], dtype=np.int64),
        area_bx=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        trip_bx=np.asarray([[5, 103], [5, 1]], dtype=np.int64),
        hierarchy_bx=np.asarray([[11.0, 22.0], [33.0, 44.0]], dtype=np.float64),
        topoind_bx=np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
        gravel_bx=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        drainarea_bx=np.asarray([[100.0, 200.0], [300.0, 400.0]], dtype=np.float64),
        min_topoind=7.0,
        nb_basin=1,
        basin_inbxid=np.asarray([55], dtype=np.int64),
        basin_sz=np.asarray([3], dtype=np.int64),
        basin_pts=np.asarray([[[1, 1], [1, 2], [2, 1]]], dtype=np.int64),
        basin_bxout=np.asarray([3], dtype=np.int64),
        coast_pts=np.zeros(3, dtype=np.int64),
        **state,
    )

    np.testing.assert_array_equal(result.basin_count, [1])
    np.testing.assert_array_equal(result.basin_id[:, :1], [[55]])
    np.testing.assert_allclose(result.basin_area[:, :1], [[6.0]])
    np.testing.assert_allclose(result.basin_hierarchy[:, :1], [[22.0]])
    np.testing.assert_allclose(result.basin_topoind[:, :1], [[20.0]])
    np.testing.assert_allclose(result.basin_gravel[:, :1], [[0.2]])
    np.testing.assert_allclose(result.basin_drainarea[:, :1], [[200.0]])
    np.testing.assert_array_equal(result.basin_flowdir[:, :1], [[3]])
    np.testing.assert_array_equal(result.outflow_grid[:, :1], [[12]])


def test_routing_globalize_one_grid_transfers_coastal_basin_and_resets_negative_hierarchy():
    state = _globalize_blank_state()
    result = routing_globalize_one_grid(
        ib=1,
        neighbours=np.asarray([[10, 11, 12, 13, 14, 15, 16, 17]], dtype=np.int64),
        area_bx=np.asarray([[1.0], [2.0]], dtype=np.float64),
        trip_bx=np.asarray([[98], [98]], dtype=np.int64),
        hierarchy_bx=np.asarray([[11.0], [22.0]], dtype=np.float64),
        topoind_bx=np.asarray([[10.0], [20.0]], dtype=np.float64),
        gravel_bx=np.asarray([[0.1], [0.2]], dtype=np.float64),
        drainarea_bx=np.asarray([[100.0], [200.0]], dtype=np.float64),
        min_topoind=7.0,
        nb_basin=1,
        basin_inbxid=np.asarray([-1], dtype=np.int64),
        basin_sz=np.asarray([2], dtype=np.int64),
        basin_pts=np.asarray([[[1, 1], [2, 1]]], dtype=np.int64),
        basin_bxout=np.asarray([-2], dtype=np.int64),
        coast_pts=np.asarray([10, 20, 0], dtype=np.int64),
        **state,
    )

    np.testing.assert_array_equal(result.basin_id[:, :1], [[-1]])
    np.testing.assert_array_equal(result.nbcoastal, [2])
    np.testing.assert_array_equal(result.coastal_basin[:, :2], [[10, 20]])
    np.testing.assert_allclose(result.basin_area[:, :1], [[3.0]])
    np.testing.assert_allclose(result.basin_hierarchy[:, :1], [[7.0]])
    np.testing.assert_allclose(result.basin_topoind[:, :1], [[7.0]])
    np.testing.assert_allclose(result.basin_gravel[:, :1], [[0.15]])
    np.testing.assert_allclose(result.basin_drainarea[:, :1], [[150.0]])
    np.testing.assert_array_equal(result.basin_flowdir[:, :1], [[-2]])
    np.testing.assert_array_equal(result.outflow_grid[:, :1], [[-2]])


def test_routing_linkup_direct_target_basin_updates_outflow_and_inflow_tables():
    result = routing_linkup(
        contfrac=np.asarray([1.0, 1.0], dtype=np.float64),
        neighbours=np.asarray([[0, 2, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0]], dtype=np.int64),
        basin_count=np.asarray([1, 1], dtype=np.int64),
        basin_area=np.asarray([[10.0], [20.0]], dtype=np.float64),
        basin_id=np.asarray([[5], [5]], dtype=np.int64),
        basin_flowdir=np.asarray([[2], [2]], dtype=np.int64),
        basin_hierarchy=np.asarray([[20.0], [10.0]], dtype=np.float64),
        outflow_grid=np.asarray([[2], [-2]], dtype=np.int64),
        nbcoastal=np.asarray([0, 0], dtype=np.int64),
        coastal_basin=np.zeros((2, 1), dtype=np.int64),
        invented_basins=100.0,
    )

    np.testing.assert_array_equal(result.outflow_grid, [[2], [-2]])
    np.testing.assert_array_equal(result.outflow_basin, [[1], [999999999]])
    np.testing.assert_array_equal(result.inflow_number, [[0], [1]])
    np.testing.assert_array_equal(result.inflow_grid[1, 0, :1], [1])
    np.testing.assert_array_equal(result.inflow_basin[1, 0, :1], [1])


def test_routing_linkup_allows_equal_hierarchy_when_directions_are_adjacent():
    result = routing_linkup(
        contfrac=np.asarray([1.0, 1.0], dtype=np.float64),
        neighbours=np.asarray([[0, 2, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0]], dtype=np.int64),
        basin_count=np.asarray([1, 1], dtype=np.int64),
        basin_area=np.asarray([[10.0], [20.0]], dtype=np.float64),
        basin_id=np.asarray([[5], [5]], dtype=np.int64),
        basin_flowdir=np.asarray([[2], [3]], dtype=np.int64),
        basin_hierarchy=np.asarray([[10.0], [10.0]], dtype=np.float64),
        outflow_grid=np.asarray([[2], [-2]], dtype=np.int64),
        nbcoastal=np.asarray([0, 0], dtype=np.int64),
        coastal_basin=np.zeros((2, 1), dtype=np.int64),
        invented_basins=100.0,
    )

    np.testing.assert_array_equal(result.outflow_basin, [[1], [999999999]])
    np.testing.assert_array_equal(result.inflow_number, [[0], [1]])


def test_routing_linkup_falls_back_to_coastal_when_neighbor_is_ocean():
    result = routing_linkup(
        contfrac=np.asarray([1.0], dtype=np.float64),
        neighbours=np.asarray([[0, -1, 0, 0, 0, 0, 0, 0]], dtype=np.int64),
        basin_count=np.asarray([1], dtype=np.int64),
        basin_area=np.asarray([[10.0]], dtype=np.float64),
        basin_id=np.asarray([[5]], dtype=np.int64),
        basin_flowdir=np.asarray([[1]], dtype=np.int64),
        basin_hierarchy=np.asarray([[10.0]], dtype=np.float64),
        outflow_grid=np.asarray([[-2]], dtype=np.int64),
        nbcoastal=np.asarray([0], dtype=np.int64),
        coastal_basin=np.zeros((1, 1), dtype=np.int64),
        invented_basins=100.0,
    )

    np.testing.assert_array_equal(result.outflow_grid, [[-2]])
    np.testing.assert_array_equal(result.outflow_basin, [[999999999]])


def test_routing_fetch_normalizes_area_accumulates_upstream_and_marks_largest_river():
    result = routing_fetch(
        resolution=np.asarray([[10.0, 10.0], [20.0, 10.0]], dtype=np.float64),
        contfrac=np.asarray([1.0, 0.5], dtype=np.float64),
        basin_count=np.asarray([1, 2], dtype=np.int64),
        basin_area=np.asarray([[2.0, 0.0], [1.0, 3.0]], dtype=np.float64),
        basin_id=np.asarray([[1, 0], [1, 2]], dtype=np.int64),
        outflow_grid=np.asarray([[2, 0], [-2, -1]], dtype=np.int64),
        outflow_basin=np.asarray([[1, 999999999], [999999999, 999999999]], dtype=np.int64),
        num_largest=1,
    )

    np.testing.assert_allclose(result.basin_area, [[100.0, 0.0], [25.0, 75.0]])
    np.testing.assert_allclose(result.fetch_basin, [[100.0, 0.0], [125.0, 75.0]])
    np.testing.assert_array_equal(result.outflow_grid, [[2, 0], [-1, -2]])


def test_routing_fetch_refuses_outflow_cycles():
    with pytest.raises(ValueError, match="did not converge"):
        routing_fetch(
            resolution=np.asarray([[10.0, 10.0]], dtype=np.float64),
            contfrac=np.asarray([1.0], dtype=np.float64),
            basin_count=np.asarray([1], dtype=np.int64),
            basin_area=np.asarray([[1.0]], dtype=np.float64),
            basin_id=np.asarray([[1]], dtype=np.int64),
            outflow_grid=np.asarray([[1]], dtype=np.int64),
            outflow_basin=np.asarray([[1]], dtype=np.int64),
            num_largest=1,
        )


def test_routing_truncate_finalize_no_reduction_writes_route_arrays_and_scales_area():
    result = routing_truncate_finalize_no_reduction(
        resolution=np.asarray([[10.0, 10.0], [20.0, 10.0]], dtype=np.float64),
        contfrac=np.asarray([1.0, 0.5], dtype=np.float64),
        basin_count=np.asarray([2, 1], dtype=np.int64),
        basin_area=np.asarray([[2.0, 3.0], [4.0, 0.0]], dtype=np.float64),
        basin_topoind=np.asarray([[10.0, 20.0], [30.0, 0.0]], dtype=np.float64),
        basin_gravel=np.asarray([[0.1, 0.2], [0.3, 0.0]], dtype=np.float64),
        basin_drainarea=np.asarray([[100.0, 200.0], [300.0, 0.0]], dtype=np.float64),
        basin_id=np.asarray([[11, 12], [13, 0]], dtype=np.int64),
        outflow_grid=np.asarray([[2, -2], [-3, 0]], dtype=np.int64),
        outflow_basin=np.asarray([[1, 999999999], [999999999, 0]], dtype=np.int64),
        nbasmax=2,
        num_largest=1,
    )

    np.testing.assert_allclose(result.routing_area, [[40.0, 60.0], [100.0, 0.0]])
    np.testing.assert_allclose(result.topo_resid, [[10.0, 20.0], [30.0, 0.0]])
    np.testing.assert_allclose(result.basgravel, [[0.1, 0.2], [0.3, 0.0]])
    np.testing.assert_allclose(result.basdrainarea, [[100.0, 200.0], [300.0, 0.0]])
    np.testing.assert_array_equal(result.global_basinid, [[11, 12], [13, 0]])
    np.testing.assert_array_equal(result.route_togrid, [[2, 1], [2, 2]])
    np.testing.assert_array_equal(result.route_tobasin, [[1, 4], [5, 0]])


def test_routing_truncate_finalize_no_reduction_refuses_unreduced_basin_counts():
    with pytest.raises(NotImplementedError, match="routing_killbas"):
        routing_truncate_finalize_no_reduction(
            resolution=np.asarray([[10.0, 10.0]], dtype=np.float64),
            contfrac=np.asarray([1.0], dtype=np.float64),
            basin_count=np.asarray([3], dtype=np.int64),
            basin_area=np.ones((1, 3), dtype=np.float64),
            basin_topoind=np.ones((1, 3), dtype=np.float64),
            basin_gravel=np.ones((1, 3), dtype=np.float64),
            basin_drainarea=np.ones((1, 3), dtype=np.float64),
            basin_id=np.ones((1, 3), dtype=np.int64),
            outflow_grid=-2 * np.ones((1, 3), dtype=np.int64),
            outflow_basin=999999999 * np.ones((1, 3), dtype=np.int64),
            nbasmax=2,
            num_largest=1,
        )


def test_routing_killbas_merges_last_basin_into_takeover_and_reduces_count():
    result = routing_killbas(
        ib=1,
        tokill=3,
        totakeover=1,
        basin_count=np.asarray([3], dtype=np.int64),
        basin_area=np.asarray([[10.0, 5.0, 7.0]], dtype=np.float64),
        basin_topoind=np.asarray([[100.0, 200.0, 300.0]], dtype=np.float64),
        basin_gravel=np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
        basin_drainarea=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        fetch_basin=np.asarray([[100.0, 50.0, 70.0]], dtype=np.float64),
        basin_id=np.asarray([[11, 12, 13]], dtype=np.int64),
        basin_flowdir=np.asarray([[-2, -2, -2]], dtype=np.int64),
        outflow_grid=np.asarray([[-2, -2, -2]], dtype=np.int64),
        outflow_basin=999999999 * np.ones((1, 3), dtype=np.int64),
        inflow_number=np.zeros((1, 3), dtype=np.int64),
        inflow_grid=np.zeros((1, 3, 3), dtype=np.int64),
        inflow_basin=np.zeros((1, 3, 3), dtype=np.int64),
    )

    np.testing.assert_array_equal(result.basin_count, [2])
    np.testing.assert_allclose(result.basin_area[0, :3], [17.0, 5.0, 0.0])
    np.testing.assert_allclose(result.basin_topoind[0, :3], [200.0, 200.0, 0.0])
    np.testing.assert_allclose(result.basin_gravel[0, :3], [0.2, 0.2, 0.0])
    np.testing.assert_allclose(result.basin_drainarea[0, :3], [2.0, 2.0, 0.0])
    np.testing.assert_allclose(result.fetch_basin[0, :3], [170.0, 50.0, 0.0])
    np.testing.assert_array_equal(result.basin_id[0, :2], [11, 12])


def test_routing_killbas_middle_shift_updates_downstream_inflow_references():
    inflow_number = np.zeros((2, 3), dtype=np.int64)
    inflow_grid = np.zeros((2, 3, 3), dtype=np.int64)
    inflow_basin = np.zeros((2, 3, 3), dtype=np.int64)
    inflow_number[1, 0] = 1
    inflow_grid[1, 0, 0] = 1
    inflow_basin[1, 0, 0] = 3

    result = routing_killbas(
        ib=1,
        tokill=2,
        totakeover=1,
        basin_count=np.asarray([3, 1], dtype=np.int64),
        basin_area=np.asarray([[10.0, 5.0, 7.0], [20.0, 0.0, 0.0]], dtype=np.float64),
        basin_topoind=np.asarray([[100.0, 200.0, 300.0], [400.0, 0.0, 0.0]], dtype=np.float64),
        basin_gravel=np.asarray([[0.1, 0.2, 0.3], [0.4, 0.0, 0.0]], dtype=np.float64),
        basin_drainarea=np.asarray([[1.0, 2.0, 3.0], [4.0, 0.0, 0.0]], dtype=np.float64),
        fetch_basin=np.asarray([[100.0, 50.0, 70.0], [170.0, 0.0, 0.0]], dtype=np.float64),
        basin_id=np.asarray([[11, 12, 13], [13, 0, 0]], dtype=np.int64),
        basin_flowdir=np.asarray([[-2, -2, 2], [-2, 0, 0]], dtype=np.int64),
        outflow_grid=np.asarray([[-2, -2, 2], [-2, 0, 0]], dtype=np.int64),
        outflow_basin=np.asarray([[999999999, 999999999, 1], [999999999, 0, 0]], dtype=np.int64),
        inflow_number=inflow_number,
        inflow_grid=inflow_grid,
        inflow_basin=inflow_basin,
    )

    np.testing.assert_array_equal(result.basin_count, [2, 1])
    np.testing.assert_array_equal(result.basin_id[0, :2], [11, 13])
    np.testing.assert_array_equal(result.outflow_grid[0, :2], [-2, 2])
    np.testing.assert_array_equal(result.outflow_basin[0, :2], [999999999, 1])
    np.testing.assert_array_equal(result.inflow_basin[1, 0, :1], [2])


def test_routing_killbas_redirects_inflows_to_takeover_basin():
    inflow_number = np.zeros((2, 2), dtype=np.int64)
    inflow_grid = np.zeros((2, 2, 2), dtype=np.int64)
    inflow_basin = np.zeros((2, 2, 2), dtype=np.int64)
    inflow_number[0, 1] = 1
    inflow_grid[0, 1, 0] = 2
    inflow_basin[0, 1, 0] = 1

    result = routing_killbas(
        ib=1,
        tokill=2,
        totakeover=1,
        basin_count=np.asarray([2, 1], dtype=np.int64),
        basin_area=np.asarray([[10.0, 5.0], [4.0, 0.0]], dtype=np.float64),
        basin_topoind=np.asarray([[100.0, 200.0], [50.0, 0.0]], dtype=np.float64),
        basin_gravel=np.asarray([[0.1, 0.2], [0.3, 0.0]], dtype=np.float64),
        basin_drainarea=np.asarray([[1.0, 2.0], [3.0, 0.0]], dtype=np.float64),
        fetch_basin=np.asarray([[10.0, 5.0], [4.0, 0.0]], dtype=np.float64),
        basin_id=np.asarray([[11, 12], [20, 0]], dtype=np.int64),
        basin_flowdir=np.asarray([[-2, -2], [1, 0]], dtype=np.int64),
        outflow_grid=np.asarray([[-2, -2], [1, 0]], dtype=np.int64),
        outflow_basin=np.asarray([[999999999, 999999999], [2, 0]], dtype=np.int64),
        inflow_number=inflow_number,
        inflow_grid=inflow_grid,
        inflow_basin=inflow_basin,
    )

    np.testing.assert_array_equal(result.basin_count, [1, 1])
    np.testing.assert_array_equal(result.outflow_basin[1, 0], 1)
    np.testing.assert_array_equal(result.inflow_number[0, 0], 1)
    np.testing.assert_array_equal(result.inflow_grid[0, 0, :1], [2])
    np.testing.assert_array_equal(result.inflow_basin[0, 0, :1], [1])


def test_routing_killbas_updates_fetch_on_new_and_old_downstream_paths():
    inflow_number = np.zeros((3, 2), dtype=np.int64)
    inflow_grid = np.zeros((3, 2, 2), dtype=np.int64)
    inflow_basin = np.zeros((3, 2, 2), dtype=np.int64)
    inflow_number[2, 0] = 1
    inflow_grid[2, 0, 0] = 1
    inflow_basin[2, 0, 0] = 2

    result = routing_killbas(
        ib=1,
        tokill=2,
        totakeover=1,
        basin_count=np.asarray([2, 1, 1], dtype=np.int64),
        basin_area=np.asarray([[10.0, 5.0], [20.0, 0.0], [30.0, 0.0]], dtype=np.float64),
        basin_topoind=np.asarray([[100.0, 200.0], [50.0, 0.0], [60.0, 0.0]], dtype=np.float64),
        basin_gravel=np.asarray([[0.1, 0.2], [0.3, 0.0], [0.4, 0.0]], dtype=np.float64),
        basin_drainarea=np.asarray([[1.0, 2.0], [3.0, 0.0], [4.0, 0.0]], dtype=np.float64),
        fetch_basin=np.asarray([[10.0, 5.0], [100.0, 0.0], [50.0, 0.0]], dtype=np.float64),
        basin_id=np.asarray([[11, 12], [21, 0], [31, 0]], dtype=np.int64),
        basin_flowdir=np.asarray([[3, 5], [-2, 0], [-2, 0]], dtype=np.int64),
        outflow_grid=np.asarray([[2, 3], [-2, 0], [-2, 0]], dtype=np.int64),
        outflow_basin=np.asarray([[1, 1], [999999999, 0], [999999999, 0]], dtype=np.int64),
        inflow_number=inflow_number,
        inflow_grid=inflow_grid,
        inflow_basin=inflow_basin,
    )

    np.testing.assert_allclose(result.fetch_basin[:, 0], [15.0, 105.0, 45.0])
    np.testing.assert_array_equal(result.inflow_number[2, 0], 0)


def test_routing_killbas_redirects_sources_into_shifted_basin_number():
    inflow_number = np.zeros((2, 3), dtype=np.int64)
    inflow_grid = np.zeros((2, 3, 3), dtype=np.int64)
    inflow_basin = np.zeros((2, 3, 3), dtype=np.int64)
    inflow_number[0, 2] = 1
    inflow_grid[0, 2, 0] = 2
    inflow_basin[0, 2, 0] = 1

    result = routing_killbas(
        ib=1,
        tokill=2,
        totakeover=1,
        basin_count=np.asarray([3, 1], dtype=np.int64),
        basin_area=np.asarray([[10.0, 5.0, 7.0], [20.0, 0.0, 0.0]], dtype=np.float64),
        basin_topoind=np.asarray([[100.0, 200.0, 300.0], [400.0, 0.0, 0.0]], dtype=np.float64),
        basin_gravel=np.asarray([[0.1, 0.2, 0.3], [0.4, 0.0, 0.0]], dtype=np.float64),
        basin_drainarea=np.asarray([[1.0, 2.0, 3.0], [4.0, 0.0, 0.0]], dtype=np.float64),
        fetch_basin=np.asarray([[100.0, 50.0, 70.0], [20.0, 0.0, 0.0]], dtype=np.float64),
        basin_id=np.asarray([[11, 12, 13], [21, 0, 0]], dtype=np.int64),
        basin_flowdir=np.asarray([[-2, -2, -2], [1, 0, 0]], dtype=np.int64),
        outflow_grid=np.asarray([[-2, -2, -2], [1, 0, 0]], dtype=np.int64),
        outflow_basin=np.asarray([[999999999, 999999999, 999999999], [3, 0, 0]], dtype=np.int64),
        inflow_number=inflow_number,
        inflow_grid=inflow_grid,
        inflow_basin=inflow_basin,
    )

    np.testing.assert_array_equal(result.basin_id[0, :2], [11, 13])
    np.testing.assert_array_equal(result.outflow_basin[1, 0], 2)
    np.testing.assert_array_equal(result.inflow_number[0, 1], 1)
    np.testing.assert_array_equal(result.inflow_grid[0, 1, :1], [2])
    np.testing.assert_array_equal(result.inflow_basin[0, 1, :1], [1])


def test_routing_truncate_reduce_to_nbasmax_merges_smallest_coastal_into_largest():
    result = routing_truncate_reduce_to_nbasmax(
        basin_count=np.asarray([3], dtype=np.int64),
        basin_area=np.asarray([[100.0, 10.0, 50.0]], dtype=np.float64),
        basin_topoind=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        basin_gravel=np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
        basin_drainarea=np.asarray([[10.0, 20.0, 30.0]], dtype=np.float64),
        fetch_basin=np.asarray([[100.0, 10.0, 50.0]], dtype=np.float64),
        basin_id=np.asarray([[11, 12, 13]], dtype=np.int64),
        basin_flowdir=np.asarray([[-2, -2, -2]], dtype=np.int64),
        outflow_grid=np.asarray([[-2, -2, -2]], dtype=np.int64),
        outflow_basin=999999999 * np.ones((1, 3), dtype=np.int64),
        inflow_number=np.zeros((1, 3), dtype=np.int64),
        inflow_grid=np.zeros((1, 3, 3), dtype=np.int64),
        inflow_basin=np.zeros((1, 3, 3), dtype=np.int64),
        nbasmax=2,
    )

    np.testing.assert_array_equal(result.basin_count, [2])
    np.testing.assert_allclose(result.basin_area[0, :2], [110.0, 50.0])
    np.testing.assert_allclose(result.fetch_basin[0, :2], [110.0, 50.0])
    np.testing.assert_array_equal(result.basin_id[0, :2], [11, 13])


def test_routing_simplify_redirects_duplicate_border_outflow_to_neighbor_subbasin():
    result = routing_simplify(
        trip=np.asarray([[101, 1], [101, 1]], dtype=np.int64),
        basin=np.asarray([[7, 7], [7, 7]], dtype=np.int64),
        hierarchy=np.asarray([[10.0, 11.0], [20.0, 21.0]], dtype=np.float64),
        basin_inbxid=7,
    )

    np.testing.assert_array_equal(result.trip, [[101, 1], [6, 1]])


def test_routing_cutbasin_splits_local_basin_by_outflow_points():
    result = routing_cutbasin(
        trip=np.asarray([[101], [103]], dtype=np.int64),
        basin=np.asarray([[7], [7]], dtype=np.int64),
        basin_inbxid=7,
        nbbasins=1,
        nbasmax=3,
    )

    assert result.nb == 2
    np.testing.assert_array_equal(result.bname, [7, 7])
    np.testing.assert_array_equal(result.sz, [1, 1])
    np.testing.assert_array_equal(result.pts[:, 0, :], [[1, 1], [2, 1]])


def test_routing_basins_post_aggregate_invented_coastal_cell_runs_to_final_route_arrays():
    result = routing_basins_post_aggregate(
        neighbours=np.zeros((1, 8), dtype=np.int64),
        resolution=np.asarray([[10.0, 20.0]], dtype=np.float64),
        contfrac=np.asarray([0.5], dtype=np.float64),
        lalo=np.asarray([[12.0, 34.0]], dtype=np.float64),
        sub_index=np.zeros((1, 1, 2), dtype=np.int64),
        sub_area=np.zeros((1, 1), dtype=np.float64),
        lon_rel=np.asarray([[34.0]], dtype=np.float64),
        lat_rel=np.asarray([[12.0]], dtype=np.float64),
        trip=np.asarray([[98.0]], dtype=np.float64),
        basins=np.asarray([[1.0]], dtype=np.float64),
        topoindex=np.asarray([[7.0]], dtype=np.float64),
        gravel=np.asarray([[0.0]], dtype=np.float64),
        drainarea=np.asarray([[0.0]], dtype=np.float64),
        hierarchy=np.asarray([[7.0]], dtype=np.float64),
        nbasmax=1,
        num_largest=1,
        min_topoind=7.0,
        invented_basins=1.0,
    )

    np.testing.assert_array_equal(result.basin_count, [1])
    np.testing.assert_allclose(result.routing_area, [[100.0]])
    np.testing.assert_allclose(result.topo_resid, [[7.0]])
    np.testing.assert_allclose(result.basgravel, [[0.0]])
    np.testing.assert_allclose(result.basdrainarea, [[0.0]])
    np.testing.assert_array_equal(result.global_basinid, [[2]])
    np.testing.assert_array_equal(result.route_togrid, [[1]])
    np.testing.assert_array_equal(result.route_tobasin, [[4]])


def test_routing_basins_post_aggregate_links_real_source_cells_to_coastal_outlet():
    result = routing_basins_post_aggregate(
        neighbours=np.asarray([[0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.int64),
        resolution=np.asarray([[10.0, 10.0], [10.0, 10.0]], dtype=np.float64),
        contfrac=np.asarray([1.0, 1.0], dtype=np.float64),
        lalo=np.asarray([[0.0, 10.0], [0.0, 20.0]], dtype=np.float64),
        sub_index=np.asarray([[[1, 1]], [[2, 1]]], dtype=np.int64),
        sub_area=np.asarray([[100.0], [100.0]], dtype=np.float64),
        lon_rel=np.asarray([[10.0], [20.0], [30.0]], dtype=np.float64),
        lat_rel=np.asarray([[0.0], [0.0], [0.0]], dtype=np.float64),
        trip=np.asarray([[3.0], [99.0], [98.0]], dtype=np.float64),
        basins=np.asarray([[10.0], [10.0], [99.0]], dtype=np.float64),
        topoindex=np.asarray([[12.0], [9.0], [1.0]], dtype=np.float64),
        gravel=np.asarray([[0.2], [0.4], [0.0]], dtype=np.float64),
        drainarea=np.asarray([[3.0], [5.0], [0.0]], dtype=np.float64),
        nbasmax=1,
        num_largest=1,
        min_topoind=7.0,
        invented_basins=10.0,
    )

    np.testing.assert_array_equal(result.basin_count, [1, 1])
    np.testing.assert_allclose(result.routing_area, [[100.0], [100.0]])
    np.testing.assert_allclose(result.topo_resid, [[12.0], [7.0]])
    np.testing.assert_allclose(result.basgravel, [[0.2], [0.4]])
    np.testing.assert_allclose(result.basdrainarea, [[3.0], [5.0]])
    np.testing.assert_array_equal(result.global_basinid, [[10], [10]])
    np.testing.assert_array_equal(result.route_togrid, [[2], [2]])
    np.testing.assert_array_equal(result.route_tobasin, [[1], [4]])


def test_routing_basins_from_map_fields_aggregates_complete_map_then_links_topology():
    lon_rel = np.asarray([[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]], dtype=np.float64)
    lat_rel = np.asarray([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]], dtype=np.float64)
    trip = 98.0 * np.ones((3, 2), dtype=np.float64)
    trip[0, 0] = 3.0
    trip[1, 0] = 99.0
    basins = 99.0 * np.ones((3, 2), dtype=np.float64)
    basins[0, 0] = 10.0
    basins[1, 0] = 10.0
    topo = np.ones((3, 2), dtype=np.float64)
    topo[0, 0] = 12.0
    topo[1, 0] = 9.0
    gravel = np.zeros((3, 2), dtype=np.float64)
    gravel[0, 0] = 0.2
    gravel[1, 0] = 0.4
    drainarea = np.zeros((3, 2), dtype=np.float64)
    drainarea[0, 0] = 3.0
    drainarea[1, 0] = 5.0

    result = routing_basins_from_map_fields(
        neighbours=np.asarray([[0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.int64),
        resolution=np.asarray([[90_000.0, 90_000.0], [90_000.0, 90_000.0]], dtype=np.float64),
        contfrac=np.asarray([1.0, 1.0], dtype=np.float64),
        lalo=np.asarray([[0.5, 10.0], [0.5, 11.0]], dtype=np.float64),
        map_fields=RoutingMapFields(
            lon_rel=lon_rel,
            lat_rel=lat_rel,
            trip=trip,
            basins=basins,
            topoindex=topo,
            gravel=gravel,
            drainarea=drainarea,
        ),
        nbasmax=1,
        num_largest=1,
        min_topoind=1.0,
        invented_basins=10.0,
    )

    np.testing.assert_array_equal(result.basin_count, [1, 1])
    np.testing.assert_array_equal(result.global_basinid, [[10], [10]])
    np.testing.assert_array_equal(result.route_togrid, [[2], [2]])
    np.testing.assert_array_equal(result.route_tobasin, [[1], [4]])
    assert result.routing_area[0, 0] > 0.0
    assert result.routing_area[1, 0] > 0.0
