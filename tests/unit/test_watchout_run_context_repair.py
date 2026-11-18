"""Bounded R1/R2/R5 source-phase regressions; no physical step/day executes.

DIM2 415-422 is a raw preread; 841-848 precedes initialize1111-1125.
DIFFUCO167-172 selects one SAVE drag flag, with an explicit override.
DIM2 1007-1008 supplies zero drag outside WATCHOUT;1163 is first-main only.
Daily physics, thermal initialization and source-clock progression below are
explicit delegates. The public callers, real forcing reads and kwargs adapter
remain active. Zero drag stops before any physical divide.
"""
from __future__ import annotations

import hashlib
from dataclasses import make_dataclass, replace
from types import SimpleNamespace as NS

import numpy as np
import pytest
import xarray as xr

from jax_orchidee.driver import orchestration as o
from jax_orchidee.driver.domain import read_forcing_first_step
from jax_orchidee.driver.sechiba_boundary import build_intersurf_first_step_payload
from jax_orchidee.sechiba import diffuco
from jax_orchidee.sechiba import hydrol as h
from tests.unit.test_parameter_context_repairs import _materialized_values
from tests.unit.test_parameter_context_repairs import prepared_context as parameter_context
from tests.unit.test_watchout_drag_transport import _precall
from tests.unit.test_watchout_paper_compiled_transport import _clock_payload, _day_cache, _prepared_case, _previous

EVIDENCE = {"inputs": [], "cached": [], "adapter": [], "restart": [], "cold": []}


def _case(tmp_path, monkeypatch, *, watchout=True, override=None):
    context = _prepared_case(tmp_path, monkeypatch, watchout=watchout, coupled=False)
    # Independent counterexample: hourly records1..24 = [0,.02,...,.02,.04].
    # Source split2 gives first initialize .5*.04+.5*0=.02; next is 0.
    path = tmp_path / "forcing_1961.nc"
    with xr.open_dataset(path) as opened:
        dataset = opened.load()
    dataset["cdrag"].values[:] = .02
    dataset["cdrag"].values[0] = 0.
    dataset["cdrag"].values[-1] = .04
    dataset.to_netcdf(path, mode="w")
    dataset.close()
    context.first_step_bundle.forcing = read_forcing_first_step(context.config_path, domain=context.first_step_bundle.domain)
    EVIDENCE["inputs"].append({"watchout": watchout, "override": override,
        "netcdf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_record1_drag": np.asarray(context.first_step_bundle.forcing.cdrag).tolist()
            if context.first_step_bundle.forcing.cdrag is not None else None})
    if override is not None:
        context.run_def_values["CDRAG_from_GCM"] = str(override)
    return context


def _immutable(context):
    kind = make_dataclass("PortableWatchoutRepairContext", [(name, object) for name in vars(context)], frozen=True)
    return kind(**vars(context))


@pytest.mark.parametrize("override,expected", [(None, True), (False, False), (True, True)])
def test_cached_preparation_uses_actual_initialize_substep(tmp_path, monkeypatch, override, expected):
    source = _case(tmp_path, monkeypatch, override=override)
    original = {name: getattr(o, name) for name in (
        "load_case_config", "paper_forcing_split_from_config", "paper_forcing_nb_spread")}
    build = parameter_context.__wrapped__(monkeypatch, tmp_path)
    values = _materialized_values()
    if override is not None:
        values["CDRAG_from_GCM"] = str(override)
    scalars = o.read_run_scalars()
    scalars.raw_run_def = values
    source.first_step_bundle.run_scalars = scalars
    monkeypatch.setattr(o, "load_paper_1961_first_step_bundle", lambda *a, **kw: source.first_step_bundle)
    for name, value in original.items():
        monkeypatch.setattr(o, name, value)
    context = build(values)
    actual = context.diffuco_pft14_static_kwargs["ldq_cdrag_from_gcm"]
    EVIDENCE["cached"].append({"override": override, "raw_record1_drag": 0.,
        "expected_initialize_drag": .02, "actual_static_mode": bool(actual)})
    assert bool(actual) is expected


class _KwargsBoundaryReached(Exception):
    pass


@pytest.mark.parametrize("watchout,override,expected", [
    (True, None, True), (True, False, False), (True, True, True), (False, True, True),
], ids=["R1-default", "R1-explicit-false", "R1-explicit-true", "R5-nonwatch-explicit-true"])
def test_public_fallback_keeps_initialize_mode_at_actual_kwargs_boundary(
    tmp_path, monkeypatch, watchout, override, expected
):
    context = _case(tmp_path, monkeypatch, watchout=watchout, override=override)
    context.diffuco_pft14_static_kwargs.pop("ldq_cdrag_from_gcm")
    source = _precall().payload
    monkeypatch.setattr(o, "_diffuco_restart_payload_from_previous_state", lambda previous: (source, (), ()))
    monkeypatch.setattr(o, "_enerbil_state_payload_from_previous_state", lambda previous: ({}, (), ()))
    observed = []

    def stop_before_physics(**kwargs):
        observed.append(kwargs)
        raise _KwargsBoundaryReached

    monkeypatch.setattr(diffuco, "diffuco_pft14_local_enerbil_precall_explicit", stop_before_physics)
    for tstep, expected_drag in [(0, .02), (1, 0.)]:
        with pytest.raises(_KwargsBoundaryReached):
            o.paper_1961_next_step_enerbil_precall_scaffold(
                context.config_path, previous_state=_previous(tstep - 1), tstep=tstep,
                prepared_context=context, diffuco_day_static_cache=_day_cache(),
                module_jit=False, diffuco_local_jit=False,
            )
        actual = observed[-1]
        EVIDENCE["adapter"].append({"watchout": watchout, "override": override, "tstep": tstep,
            "actual_static_mode": actual["ldq_cdrag_from_gcm"],
            "actual_q_cdrag": None if "q_cdrag" not in actual else np.asarray(actual["q_cdrag"]).tolist(),
            "physics_calls": 0})
        assert actual["ldq_cdrag_from_gcm"] is expected
        if expected:
            np.testing.assert_array_equal(actual["q_cdrag"], [expected_drag if watchout else 0.])
        else:
            assert "q_cdrag" not in actual


class _ColdInitializationInputReached(Exception):
    pass


def test_cold_coverage_initialize_payload_uses_actual_substep(tmp_path, monkeypatch):
    context = _case(tmp_path, monkeypatch)
    config = o.load_case_config(context.config_path)
    config.update(minimal_single_case={"cold_start": True}, restart_protocol={"cold_start": {
        "RESTART_FILEIN": "NONE", "SECHIBA_restart_in": "NONE", "STOMATE_RESTART_FILEIN": "NONE"}})
    monkeypatch.setattr(o, "load_case_config", lambda *a: config)
    monkeypatch.setattr(o, "_fresh_cold_thermosoil_controls", lambda *a: NS())
    monkeypatch.setattr(o, "_fresh_cold_thermosoil_missing_inputs", lambda *a, **kw: ())
    monkeypatch.setattr(o, "_prepared_hydrol_soil_parameters", lambda *a: NS())
    monkeypatch.setattr(o, "load_paper_1961_step_bundle", lambda *a, **kw:
        o._paper_1961_step_bundle_from_context(context, year=kw["year"], tstep=kw["tstep"]))
    original = o.build_intersurf_first_step_payload
    payloads = []

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        payloads.append(result)
        return result

    def stop(*a, **kw):
        raise _ColdInitializationInputReached

    monkeypatch.setattr(o, "build_intersurf_first_step_payload", observe)
    monkeypatch.setattr(o, "read_run_scalars", stop)
    with pytest.raises(_ColdInitializationInputReached):
        o.paper_1961_driver_cold_start_first_step_coverage(context.config_path, prepared_context=context)
    EVIDENCE["cold"].append({"boundary": "initialize-input", "actual_drag": np.asarray(payloads[0].cdrag).tolist(),
        "expected_drag": .02, "physical_initialization_calls": 0})
    np.testing.assert_array_equal(payloads[0].cdrag, [.02])


def _restart_delegates(monkeypatch):
    monkeypatch.setattr(o, "driver_year_handoff_state_gaps", lambda packet: ())
    monkeypatch.setattr(o, "_source_clock_setup", lambda context: None)
    monkeypatch.setattr(o, "source_clock_setup_from_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(o, "rebase_driver_state_for_year_start", lambda previous, **kw: replace(previous, tstep=-1))
    monkeypatch.setattr(o, "_require_source_day_alignment", lambda *args, **kwargs: None)
    monkeypatch.setattr(h, "hydrol_init_us_state", lambda **kw: NS(us=kw["us"], humrelv=np.ones((1, 14))))
    monkeypatch.setattr(h, "hydrol_init_vegstress", lambda **kw: kw["vegstress"])
    monkeypatch.setattr(o, "resolve_hydrol_initialization_refsoc_1d", lambda *args, **kwargs: None)
    monkeypatch.setattr(o, "restart_thermosoil_initialization_supported", lambda context: False)


def _handoff_packet():
    packet = _previous(47)
    packet.fields_by_component.update({
        "sechiba_finalize_state": {"us": np.zeros((1, 14, 6, 11)), "vegstress": np.ones((1, 14)), "njsc": np.array([2.])},
        "hydrol_previous_step_state": {}, "thermosoil_previous_step_state": {},
    })
    return packet


def _day_result(kwargs):
    day = kwargs["day_index"]
    return o.DriverRuntimeDayResult(year=1961, day_index=day, start_tstep=(day - 1) * 48,
        steps_per_stomate=48, daily_modelout=NS(day_index=day),
        day_end_state=replace(kwargs["previous_state"], tstep=day * 48 - 1),
        stopped_at_tstep=None, missing_components=())


@pytest.mark.parametrize("blocks", [False, True], ids=["day2", "compiled-block"])
def test_restart_outer_retains_admitted_mode_and_retires_phase(tmp_path, monkeypatch, blocks):
    context = _immutable(_case(tmp_path, monkeypatch))
    _restart_delegates(monkeypatch)
    monkeypatch.setattr(o, "prepare_paper_1961_driver_context", lambda *a, **kw: context)
    observed = []

    def daily(*args, **kwargs):
        observed.append((kwargs["day_index"], kwargs["prepared_context"]))
        return _day_result(kwargs)

    def block(*args, **kwargs):
        observed.append((2, kwargs["context"]))
        kwargs["daily_modelout"].extend((NS(day_index=2), NS(day_index=3)))
        return replace(kwargs["previous_state"], tstep=143), 4

    monkeypatch.setattr(o, "paper_1961_driver_later_day_runtime_result", daily)
    monkeypatch.setattr(o, "_paper_run_compiled_later_day_blocks", block)
    result = o.paper_1961_driver_restart_year_multiday_modelout_lite_run(
        context.config_path, previous_year_end_state=_handoff_packet(), year=1961,
        ndays=3 if blocks else 2, use_compiled_sechiba_day=blocks, compiled_day_block_size=2 if blocks else 0,
    )
    data = [{"day": day, "actual_mode": o._paper_context_drag_mode(ctx),
             "actual_phase": ctx.watchout_initialize_main_tstep} for day, ctx in observed]
    EVIDENCE["restart"].append({"blocks": blocks, "observations": data, "physical_days": 0})
    assert result.ready_for_requested_days
    assert data == [{"day": 1, "actual_mode": True, "actual_phase": 0},
                    {"day": 2, "actual_mode": True, "actual_phase": None}]
    assert result.prepared_context is observed[-1][1]
    assert context.diffuco_pft14_static_kwargs["ldq_cdrag_from_gcm"] is False
    assert context.watchout_initialize_main_tstep is None


@pytest.mark.parametrize("entry", ["year-start", "outer"])
def test_continuous_outer_preserves_mode_and_does_not_replay_first_main(tmp_path, monkeypatch, entry):
    context = replace(_immutable(_case(tmp_path, monkeypatch)), watchout_initialize_main_tstep=0)
    _restart_delegates(monkeypatch)
    observed = []

    def daily(*args, **kwargs):
        ctx = kwargs["prepared_context"]
        payload = build_intersurf_first_step_payload(o._paper_1961_step_bundle_from_context(ctx, year=1961, tstep=0))
        energy = o._enerbil_driver_coverage_payload(payload, initializing=o._paper_watchout_initializing(ctx, 0),
            driver_albedo_payload={"albedo": np.array([[.3, .5]])})
        observed.append({"actual_mode": o._paper_context_drag_mode(ctx), "actual_phase": ctx.watchout_initialize_main_tstep,
                         "actual_swnet": np.asarray(energy["swnet"]).tolist(), "file_swnet": np.asarray(payload.swnet).tolist()})
        return _day_result(kwargs)

    monkeypatch.setattr(o, "paper_1961_driver_later_day_runtime_result", daily)
    # Compatibility makes the old route observable: it lacked this public
    # continuous contract, so its TypeError is retained as an API failure.
    if entry == "year-start":
        result = o.paper_1961_driver_restart_year_start_day_result(
            context.config_path, previous_year_end_state=_handoff_packet(), year=1961,
            prepared_context=context, run_mode="continuous",
        )
    else:
        result = o.paper_1961_driver_restart_year_multiday_modelout_lite_run(
            context.config_path, previous_year_end_state=_handoff_packet(), year=1961, ndays=1,
            prepared_context=context, run_mode="continuous",
        )
    EVIDENCE["restart"].append({"entry": entry, "continuous": observed, "physical_days": 0})
    assert observed == [{"actual_mode": False, "actual_phase": None, "actual_swnet": [127.], "file_swnet": [127.]}]
    assert result.prepared_context.watchout_initialize_main_tstep is None
    assert context.watchout_initialize_main_tstep == 0


@pytest.mark.parametrize("outer", ["multiday", "compiled-block", "runtime-day"])
def test_cold_real_admission_result_reaches_outer_later_day(tmp_path, monkeypatch, outer):
    context = _immutable(_case(tmp_path, monkeypatch))
    monkeypatch.setattr(o, "prepare_paper_1961_driver_context", lambda *a, **kw: context)
    monkeypatch.setattr(o, "_source_clock_setup", lambda context: NS(runtime=None))
    monkeypatch.setattr(o, "_require_source_day_alignment", lambda *a, **kw: None)
    monkeypatch.setattr(o, "_driver_source_clock_payload", _clock_payload)
    monkeypatch.setattr(o, "_filter_day_state_gaps", lambda *a: ())
    monkeypatch.setattr(o, "_calendar_snapshot_before_stomate", lambda *a: None)
    monkeypatch.setattr(o, "_paper_cold_start_day_daily_process_from_completed_entries", lambda **kw: None)
    coverage_contexts = []

    def first_coverage(*args, **kwargs):
        coverage_contexts.append(kwargs["prepared_context"])
        return NS(initialized_payloads={"cold_start_first_step_previous_state": _previous(0),
            "cold_start_first_step_entry_payload": {}}, missing_components=(), provenance=())

    monkeypatch.setattr(o, "paper_1961_driver_cold_start_first_step_coverage", first_coverage)
    real_cold = o.paper_1961_driver_cold_start_day_scaffold
    first_results = []

    def delegated_cold(*args, **kwargs):
        # Real admission, first-entry bookkeeping and returned context; all
        # physical state/fold results are delegates. No full day runs.
        result = real_cold(*args, **kwargs, max_steps=1)
        first_results.append(result)
        return replace(result, completed_steps=48, daily_process_ready=True,
            first_day_end_state=_previous(47), stomate_outputs=NS())

    monkeypatch.setattr(o, "paper_1961_driver_cold_start_day_scaffold", delegated_cold)
    monkeypatch.setattr(o, "_driver_modelout_from_outputs", lambda *a, **kw: NS(day_index=1))
    observed = []

    def daily(*args, **kwargs):
        observed.append(kwargs["prepared_context"])
        return _day_result(kwargs)

    def block(*args, **kwargs):
        observed.append(kwargs["context"])
        kwargs["daily_modelout"].extend((NS(day_index=2), NS(day_index=3)))
        return _previous(143), 4

    monkeypatch.setattr(o, "paper_1961_driver_later_day_runtime_result", daily)
    monkeypatch.setattr(o, "_paper_run_compiled_later_day_blocks", block)
    if outer == "runtime-day":
        o.paper_1961_driver_cold_start_runtime_day_result(context.config_path, day_index=2)
    else:
        o.paper_1961_driver_cold_start_multiday_modelout_lite_run(context.config_path,
            ndays=3 if outer == "compiled-block" else 2, use_compiled_sechiba_day=outer == "compiled-block",
            compiled_day_block_size=2 if outer == "compiled-block" else 0)
    data = {"outer": outer, "admitted_mode": o._paper_context_drag_mode(coverage_contexts[0]),
            "admitted_phase": coverage_contexts[0].watchout_initialize_main_tstep,
            "later_mode": o._paper_context_drag_mode(observed[0]),
            "later_phase": observed[0].watchout_initialize_main_tstep, "physical_days": 0}
    EVIDENCE["cold"].append(data)
    assert data == {"outer": outer, "admitted_mode": True, "admitted_phase": 0,
                    "later_mode": True, "later_phase": None, "physical_days": 0}
    assert first_results[0].prepared_context is observed[0]
