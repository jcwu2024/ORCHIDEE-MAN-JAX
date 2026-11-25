"""Bounded later-day OK_LEAK state-carry regression.

The shared half-hour loop is real; only the component result is a controlled
boundary so the test isolates pool ownership and carry order.
"""
from __future__ import annotations

from collections import namedtuple
from types import SimpleNamespace as NS

import numpy as np
import pytest

from jax_orchidee.driver import orchestration as orch

POOL_FIELDS = (
    "litter_above", "litter_below", "lignin_struc_above", "lignin_struc_below",
    "litterpart", "dead_leaves", "fuel_1hr", "fuel_10hr", "fuel_100hr",
    "fuel_1000hr", "carbon_32l", "DOC", "interception_storage",
)


def _packet(pool_value: float) -> orch.DriverPreviousStepStatePacket:
    slow = {name: np.full((1, 1), pool_value, dtype=np.float64) for name in POOL_FIELDS}
    slow.update(
        turnover_daily=np.full((1, 1), 2.0), bm_to_litter=np.full((1, 1), 3.0),
        biomass=np.full((1, 1, 1, 1), 4.0), veget_max=np.ones((1, 1)),
        sla_calc=np.ones((1, 1)), altmax=np.zeros((1, 1)),
        lai=np.ones((1, 1)),
        fixed_cryoturbation_depth=np.zeros((1,)),
    )
    for name in orch.LATER_DAY_STOMATE_ENTRY_STATE_OVERRIDES:
        slow.setdefault(name, np.full((1, 1), pool_value, dtype=np.float64))
    fields = {"slowproc_stomate_previous_step_state": slow}
    for gap in orch.DAY1_PREVIOUS_STEP_STATE_GAPS:
        component = fields.setdefault(gap.component, {})
        for name in gap.fields:
            component.setdefault(name, np.zeros((1, 1), dtype=np.float64))
    return orch.DriverPreviousStepStatePacket(
        tstep=47,
        fields_by_component=fields,
        provenance_by_component={},
    )


def _result_from(values):
    next_values = {name: np.asarray(value) + 1.0 for name, value in values.items()}
    litter = NS(
        litter_above=next_values["litter_above"],
        lignin_struc_above=next_values["lignin_struc_above"],
        lignin_struc_below=next_values["lignin_struc_below"],
        litterpart=next_values["litterpart"], dead_leaves=next_values["dead_leaves"],
        fuel=NS(**{f"fuel_{hours}hr": next_values[f"fuel_{hours}hr"]
                   for hours in (1, 10, 100, 1000)}),
    )
    soil = NS(
        litter_below=next_values["litter_below"],
        carbon_32l=next_values["carbon_32l"], doc=next_values["DOC"],
    )
    return NS(littercalc=litter, soilcarbon=soil,
              interception_storage=next_values["interception_storage"],
              bulk_dens=np.full((1, 1), 8.0))


@pytest.mark.parametrize("start_pool", [0.0, 41.0], ids=["zero", "distinct-day-end"])
def test_ok_leak_half_hour_fold_carries_day_start_packet_stocks_and_return_updates(
    monkeypatch, start_pool,
):
    initial = _packet(start_pool)
    payloads = tuple({
        "soil_mc": np.full((1, 1, 1), 0.2 + index), "wat_flux": np.zeros((1, 1)),
        "runoff_per_soil": np.zeros((1, 1)), "drainage_per_soil": np.zeros((1, 1)),
        "runoff2peat": np.zeros((1,)), "canopy2ground": np.zeros((1, 1)),
        "precip2ground": np.zeros((1,)), "precip2canopy": np.zeros((1, 1)),
        "tdeep": np.full((1, 1), 270.0), "hsdeep": np.zeros((1, 1)),
        "shumdiag_peat": None, "source_clock": NS(month=2, day=index + 1),
    } for index in range(2))
    maintenance = tuple(NS(resp_maint_part=np.full((1, 1), index + 0.5)) for index in range(2))
    base_inputs = {name: np.zeros((1, 1)) for name in POOL_FIELDS}
    base_inputs.update(bulk_dens=np.full((1, 1), 8.0), flood_frac=np.zeros((1, 1)),
                       nslm=1, ndeep=1, zi_soil=np.zeros((1, 1)), veget_mask=np.ones((1, 1)))
    pre_step = NS(ok_leak_inputs=base_inputs, output_inputs={}, provenance=())
    calls = []
    prep_calls = []

    monkeypatch.setattr(orch, "_paper_ok_leak_sechiba_adapter", lambda **kwargs: NS())
    monkeypatch.setattr(orch, "stomate_same_step_ok_leak_boundary_inputs", lambda **kwargs: NS(
        boundary_inputs=NS(ok_leak_inputs={**base_inputs, **kwargs["ok_leak_inputs"]}),
        active_layer=NS(state_updates={"altmax": np.zeros((1, 1))}),
    ))
    def litter_prep(*args, **kwargs):
        prep_calls.append((np.asarray(args[1]).copy(), np.asarray(args[2]).copy()))
        return NS(bm_to_littercalc=np.zeros((1, 1)), turnover_littercalc=np.zeros((1, 1)),
                  soil_mc_32l=np.zeros((1, 1)))

    monkeypatch.setattr(orch, "stomate_littercalc_entry_prep", litter_prep)
    monkeypatch.setattr(orch, "sum_resp_maint_radia", lambda *args, **kwargs: (None, np.zeros((1, 1))))
    monkeypatch.setattr(orch, "AltCalcDocResult", lambda **kwargs: NS(**kwargs))

    def ok_leak(**kwargs):
        received = {name: np.asarray(kwargs["doc"] if name == "DOC" else kwargs[name]).copy()
                    for name in POOL_FIELDS if name != "DOC"}
        received["DOC"] = np.asarray(kwargs["doc"]).copy()
        calls.append({"pools": received})
        return _result_from(received)

    monkeypatch.setattr(orch, "stomate_ok_leak_explicit", ok_leak)
    result, updates = orch._paper_half_hour_ok_leak_fold_from_entries(
        entry_payloads=payloads, maintenance_step_results=maintenance,
        initial_state=initial, pre_step_boundary=pre_step,
        runtime=NS(dt_sechiba=43200.0),
        run_def_values={"frozen_respiration_func": "0", "MOIST_FUNC_MOYANO": "FALSE",
                        "TF_DOC": "FALSE", "PERMA_PEAT": "FALSE", "CONC_DOC_RAIN": "0"},
        rprof=np.ones((1, 1)), pref_soil_veg=np.asarray([1]),
        is_tree=np.asarray([False]), is_peat=np.asarray([False]), dayno=2,
    )

    assert len(calls) == 2
    for name in POOL_FIELDS:
        np.testing.assert_array_equal(calls[0]["pools"][name], np.full((1, 1), start_pool))
        np.testing.assert_array_equal(calls[1]["pools"][name], np.full((1, 1), start_pool + 1.0))
    assert len(prep_calls) == 2
    for turnover, bm_to_litter in prep_calls:
        np.testing.assert_array_equal(turnover, [[2.0]])
        np.testing.assert_array_equal(bm_to_litter, [[3.0]])
    for name in POOL_FIELDS:
        np.testing.assert_array_equal(updates[name], np.full((1, 1), start_pool + 2.0))
    assert result.bulk_dens[0, 0] == 8.0


@pytest.mark.parametrize("case", ["updated", "disabled", "missing-dynamic"])
@pytest.mark.parametrize("route", ["direct", "full", "lite"])
def test_later_day_scaffold_routes_previous_packet_through_real_half_hour_fold(monkeypatch, case, route):
    """Exercise the production day dispatcher while stubbing only physics kernels."""
    initial = _packet(41.0)
    run_scalars = NS(
        pref_soil_veg=np.asarray([1]), nstm=1,
        is_tree=np.asarray([False]), is_peat=np.asarray([False]),
    )
    driver_payload = NS(kjpindex=1, lalo=np.zeros((1, 2)))
    scaffold = NS(enerbil=NS(payload=driver_payload, step=NS(run_scalars=run_scalars)))
    runtime = NS(dt_stomate=86400.0, dt_sechiba=1800.0, dt_days=1.0)
    run_def_values = {
        "STOMATE_OK_STOMATE": "TRUE", "OK_LEAK": "TRUE", "PERMA_PEAT": "FALSE",
        "MOIST_FUNC_MOYANO": "FALSE", "TF_DOC": "FALSE", "CONC_DOC_RAIN": "0",
        "frozen_respiration_func": "0", "RIVER_ROUTING": "FALSE", "MIN_WIND": "0",
        "STOMATE_OK_DGVM": "TRUE", "use_refSOC_hydrol": "FALSE",
    }
    if case == "disabled":
        run_def_values.update(STOMATE_OK_STOMATE="FALSE", OK_LEAK="FALSE")
    base_inputs = {name: np.zeros((1, 1)) for name in POOL_FIELDS}
    base_inputs.update(
        bulk_dens=np.full((1, 1), 8.0), flood_frac=np.zeros((1, 1)),
        nslm=1, ndeep=1, zi_soil=np.zeros((1, 1)), veget_mask=np.ones((1, 1)),
    )
    pre_step = orch.StomateOkLeakBoundaryInputs(base_inputs, {}, ("synthetic boundary",))
    payloads = []
    for index in range(48):
        payloads.append({
            "soil_mc": np.full((1, 1, 1), 0.2), "wat_flux": np.zeros((1, 1)),
            "runoff_per_soil": np.zeros((1, 1)), "drainage_per_soil": np.zeros((1, 1)),
            "runoff2peat": np.zeros((1,)), "canopy2ground": np.zeros((1, 1)),
            "precip2ground": np.zeros((1,)), "precip2canopy": np.zeros((1, 1)),
            "tdeep": np.full((1, 1), 270.0), "hsdeep": np.zeros((1, 1)),
            "temp_sol": np.full((1, 1), 270.0), "shumdiag_peat": None,
            "source_clock": NS(month=2, day=1 + index // 48, julian_diff=0.0),
        })
    if case == "missing-dynamic":
        payloads[3]["temp_sol"] = None
    maintenance = tuple(NS(resp_maint_part=np.full((1, 1), 0.5)) for _ in payloads)
    FoldPart = namedtuple("FoldPart", "step_results")
    DailyFold = namedtuple("DailyFold", "ok maintenance accumulator daily_fields")
    calls = []
    packet_at_bundle = []
    context = NS(
        run_def_path="fixture.def", run_def_values=run_def_values, runtime=runtime,
        cwrr_grid=NS(znh=np.zeros((1, 1)), znt=np.zeros((1, 1)), zlt=np.zeros((1, 1))),
        first_step_bundle=None, first_step_restart_state=NS(stomate=None),
        source_clock_setup=None, reference_run_dir="fixture", domain_override=None,
        diaglev=0, zlt=np.zeros((1, 1)), znt=np.zeros((1, 1)),
        stomate_static=NS(parameters=NS()),
        stomate_boundary=NS(kwargs={"rprof": np.ones((1, 1))}),
        ok_laidev=np.asarray([False]),
    )
    EntryTemplate = namedtuple("EntryTemplate", orch.LATER_DAY_STOMATE_ENTRY_STATE_OVERRIDES)
    context.first_step_restart_state = NS(
        stomate=EntryTemplate(*(np.zeros((1, 1)) for _ in orch.LATER_DAY_STOMATE_ENTRY_STATE_OVERRIDES))
    )
    monkeypatch.setattr(orch, "_require_source_day_alignment", lambda *a, **k: None)
    monkeypatch.setattr(orch, "_source_clock_setup", lambda *a, **k: NS(binding=NS(calendar="noleap")))
    monkeypatch.setattr(orch, "_driver_source_clock_payload", lambda *a, **k: NS(
        month=2, day=1, julian_diff=0.0))
    monkeypatch.setattr(orch, "_hydrol_state_namespace_from_previous_state", lambda *a: (NS(), None))
    monkeypatch.setattr(orch, "_prepared_hydrol_soil_parameters", lambda *a: NS())
    monkeypatch.setattr(orch, "hydrol_static_precall_template_from_slowproc", lambda **k: NS())
    monkeypatch.setattr(orch, "_select_run_def_pft_vector", lambda *a, **k: np.ones((1,)))
    monkeypatch.setattr(orch, "load_paper_1961_step_bundle", lambda *a, **k: NS(run_scalars=run_scalars))
    monkeypatch.setattr(orch, "paper_1961_next_step_hydrol_precall_scaffold", lambda *a, **k: scaffold)
    next_payload = iter(payloads)

    def next_entry(*args, **kwargs):
        return next(next_payload)

    monkeypatch.setattr(orch, "_paper_next_step_entry_payload", next_entry)
    monkeypatch.setattr(orch, "_paper_next_step_runtime_entry_payload", next_entry)
    monkeypatch.setattr(orch, "driver_previous_step_state_from_next_step_hydrol_scaffold", lambda *a, **k: initial)
    monkeypatch.setattr(orch, "_calendar_snapshot_before_stomate", lambda previous, current: previous)
    monkeypatch.setattr(orch, "_paper_later_day_daily_process_from_completed_entries",
        lambda **k: DailyFold(True, FoldPart(maintenance), FoldPart(maintenance), {}))
    monkeypatch.setattr(orch, "_paper_ok_leak_sechiba_adapter", lambda **k: NS())
    monkeypatch.setattr(orch, "stomate_pre_step_ok_leak_boundary_inputs",
        lambda **k: NS(boundary_inputs=pre_step))
    monkeypatch.setattr(orch, "stomate_same_step_ok_leak_boundary_inputs", lambda **k: NS(
        boundary_inputs=orch.StomateOkLeakBoundaryInputs(
            {**base_inputs, **k["ok_leak_inputs"]}, {}, ("synthetic same-step boundary",)),
        active_layer=NS(state_updates={"altmax": np.zeros((1, 1))}),
    ))
    monkeypatch.setattr(orch, "stomate_littercalc_entry_prep", lambda *a, **k: NS(
        bm_to_littercalc=np.zeros((1, 1)), turnover_littercalc=np.zeros((1, 1)),
        soil_mc_32l=np.zeros((1, 1))))
    monkeypatch.setattr(orch, "sum_resp_maint_radia", lambda *a, **k: (None, np.zeros((1, 1))))
    monkeypatch.setattr(orch, "AltCalcDocResult", lambda **k: NS(**k))

    def ok_leak(**kwargs):
        received = {name: np.asarray(kwargs["doc"] if name == "DOC" else kwargs[name]).copy()
                    for name in POOL_FIELDS}
        calls.append(received)
        return _result_from(received)

    monkeypatch.setattr(orch, "stomate_ok_leak_explicit", ok_leak)
    def stop_at_daily_bundle(**kwargs):
        packet_at_bundle.append(kwargs["previous_state"])
        return None, None, (orch.DriverDayStateGap("test", ("stop after fold",), ()),)
    monkeypatch.setattr(orch, "_paper_later_day_stomate_input_bundles", stop_at_daily_bundle)

    if route == "direct":
        result = orch.paper_1961_driver_later_day_scaffold(
            "fixture.yml", previous_state=initial, year=1961, start_tstep=48,
            prepared_context=context, module_jit=False, retain_stomate_step_results=False,
        )
    else:
        monkeypatch.setattr(orch, "prepare_paper_1961_driver_context", lambda *a, **k: context)
        monkeypatch.setattr(orch, "_paper_context_after_initial_main", lambda value: value)
        first_day = NS(first_day_end_state=initial, ready_for_first_day_end_state=True,
                       initial_state_mode="synthetic-first-day")
        monkeypatch.setattr(orch, "paper_1961_driver_day_scaffold", lambda *a, **k: first_day)
        monkeypatch.setattr(orch, "_driver_day_modelout", lambda day, *, day_index: NS(day_index=day_index))
        actual_later_day = orch.paper_1961_driver_later_day_scaffold
        later_results = []

        def observe_later_day(*args, **kwargs):
            day = actual_later_day(*args, **kwargs)
            later_results.append(day)
            return day

        monkeypatch.setattr(orch, "paper_1961_driver_later_day_scaffold", observe_later_day)
        if route == "full":
            result = orch.paper_1961_driver_multiday_modelout_run("fixture.yml", ndays=2)
            assert isinstance(result, orch.DriverMultiDayModeloutRun)
            assert result.scaffold.requested_days == 2
            assert len(result.scaffold.days) == 2
            assert result.scaffold.days[1] is not None
            assert len(result.daily_modelout) == 2
        else:
            result = orch.paper_1961_driver_multiday_modelout_lite_run("fixture.yml", ndays=2)
            assert isinstance(result, orch.DriverLiteMultiDayModeloutRun)
            assert result.requested_days == 2
            assert result.stopped_day_index == 2
            assert len(result.daily_modelout) == 1
        assert len(later_results) == 1
        result = later_results[0]

    if case == "disabled":
        assert calls == []
        assert result.stomate_ok_leak is None
        assert result.stomate_bundle_gaps[0].component == "stomate_disabled"
    elif case == "missing-dynamic":
        assert calls == []
        assert any(gap.component == "stomate_ok_leak_tstep_51" and
                   gap.fields == ("temp_sol",) for gap in result.ok_leak_boundary_gaps)
        assert result.stomate_bundle_gaps == ()
        assert result.stomate_ok_leak is None
    else:
        assert len(calls) == 48, (result.stopped_at_tstep, result.state_gaps,
                                  result.daily_process_ready, result.daily_process_fold,
                                  result.stomate_bundle_gaps)
        assert not result.state_gaps
        np.testing.assert_array_equal(calls[0]["DOC"], [[41.0]])
        np.testing.assert_array_equal(calls[1]["DOC"], [[42.0]])
        np.testing.assert_array_equal(calls[-1]["DOC"], [[88.0]])
        assert result.stomate_ok_leak is not None
        assert result.ok_leak_boundary_inputs is not None
        assert len(packet_at_bundle) == 1
        np.testing.assert_array_equal(packet_at_bundle[0].fields_by_component[
            "slowproc_stomate_previous_step_state"]["DOC"], [[89.0]])
        assert result.daily_process_fold is not None
        assert result.daily_process_fold.maintenance.step_results == ()
