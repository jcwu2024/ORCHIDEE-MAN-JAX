"""Synthetic invocation checks for the production source-calendar adapters.

No physical delegate is executed. The small JAX probes below compile only
calendar/metadata transport; they are not physical-model or scientific gates.
Fortran owner: slowproc.f90::slowproc_main 556-610.
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import fields, make_dataclass, replace
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_orchidee.driver import orchestration as orch
from jax_orchidee.driver.fast_state import (
    fast_state_from_previous_packet,
    previous_packet_from_fast_state,
)
from jax_orchidee.driver.source_clock import (
    SourceClockBinding,
    SourceClockError,
    SourceClockRuntime,
    SourceClockSetup,
)
from jax_orchidee.sechiba.ioipsl_calendar import IOIPSLCalendar


SLOW = "slowproc_stomate_previous_step_state"
FINAL = "sechiba_finalize_state"


def _context(*, date=(2001, 1, 1), source_start=1, **controls):
    provider = IOIPSLCalendar("noleap")
    date0 = provider.ymds2ju(*date, 0.0)
    lock = IOIPSLCalendar.from_scalar_startdate("noleap", date0)
    binding = SourceClockBinding("synthetic-adapter-run", "noleap", date0, 1800.,
                                lock.startdate_day, lock.startdate_second, 0)
    runtime = SourceClockRuntime(binding, np.int64(source_start - 1), np.int64(0),
                                 np.int64(source_start))
    return NS(source_clock_setup=SourceClockSetup(binding, runtime, None, None, None),
              run_def_values=controls, dt_sechiba=1800., config_path=Path("synthetic.yml"),
              runtime=NS(dt_stomate=86400., dt_sechiba=1800., dt_days=1.),
              run_def_path=Path("synthetic.def"), cwrr_grid=NS())


def _packet(context, *, tstep=-1):
    slow = dict(growth_day=jnp.asarray([3., 8., 1.]), peatC_ok=jnp.asarray([0., 0., 1.]),
                temp_growth=jnp.asarray([5.1, 5., 4.9]), peatC=jnp.asarray([25., 24.9, 26.]),
                carbon=jnp.asarray([11., 12., 13.]))
    return orch.DriverPreviousStepStatePacket(
        tstep, {SLOW: dict(slow), FINAL: dict(slow)},
        {SLOW: ("synthetic source-calendar input",), FINAL: ("synthetic mirror",)},
        context.source_clock_setup.runtime,
    )


def _payload(context, state, tstep, **overrides):
    clock = orch._driver_source_clock_payload(context, state.source_clock_runtime, tstep=tstep)
    return clock._replace(**overrides)


def _apply(context, state, tstep, **overrides):
    return orch._apply_driver_source_calendar(
        state, context=context, source_clock=_payload(context, state, tstep, **overrides))


def _mirrors(state, growth, peat):
    for component in (SLOW, FINAL):
        np.testing.assert_array_equal(state.fields_by_component[component]["growth_day"], growth)
        np.testing.assert_array_equal(state.fields_by_component[component]["peatC_ok"], peat)


def _replace_extraction_delegates(monkeypatch):
    calls = []
    for name in (
        "_add_enerbil_local_to_previous_fields", "_add_diffuco_local_to_previous_fields",
        "_add_diffuco_saved_controls_to_previous_fields", "_add_thermosoil_to_previous_fields",
        "_add_condveg_to_previous_fields", "_add_hydrol_to_previous_fields",
    ):
        def spy(*args, _name=name, **kwargs):
            calls.append(_name)
        monkeypatch.setattr(orch, name, spy)
    monkeypatch.setattr(orch, "_write_maintenance_lai_to_previous_fields", lambda *a, **k: None)
    monkeypatch.setattr(orch, "_sechiba_finalize_state_from_components",
                        lambda *, previous, **kwargs: dict(previous))
    return calls


def test_jan1_initialization_first_active_and_reobservation_use_actual_packet_adapter():
    context = _context()
    initial = _packet(context)
    inactive = _apply(context, initial, 0, active=False)
    _mirrors(inactive, [3., 8., 1.], [0., 0., 1.])
    assert int(inactive.source_clock_runtime.last_applied_istp) == 0
    first = _apply(context, inactive, 0)
    _mirrors(first, [0., 0., 0.], [0., 0., 1.])
    repeated = _apply(context, first, 0)
    _mirrors(repeated, [0., 0., 0.], [0., 0., 1.])
    assert int(repeated.source_clock_runtime.last_applied_istp) == 1
    _mirrors(initial, [3., 8., 1.], [0., 0., 1.])


@pytest.mark.parametrize("dynpeat,judge,expected", [
    (False, False, [1., 1., 1.]), (False, True, [1., 1., 1.]),
    (True, False, [0., 0., 1.]), (True, True, [1., 0., 1.]),
])
@pytest.mark.parametrize("fast", [False, True], ids=["packet", "fast-codec"])
def test_jan2_actual_packet_and_fast_adapters_cover_nondefault_limit_equality(dynpeat, judge, expected, fast):
    context = _context(date=(2001, 1, 2), dYnPeAt_Pc=str(dynpeat), JUDGE_PC=str(judge), PC_LIM="25")
    state = _packet(context)
    if fast:
        state = fast_state_from_previous_packet(state)
    result = _apply(context, state, 0)
    _mirrors(result, [3., 8., 1.], expected)
    if fast:
        result = previous_packet_from_fast_state(result)
        assert result.source_clock_runtime.binding == context.source_clock_setup.binding
        _mirrors(result, [3., 8., 1.], expected)


@pytest.mark.parametrize("date,source_start,growth,peat", [
    ((2001, 1, 1), 48, [4., 8., 1.], [0., 0., 1.]),
    ((2001, 1, 2), 1, [3., 8., 1.], [1., 0., 1.]),
], ids=["midnight-pre-temperature", "jan2-pre-peat-carbon"])
def test_actual_dayend_merge_keeps_pre_stomate_calendar_and_post_stomate_science(monkeypatch, date, source_start, growth, peat):
    context = _context(date=date, source_start=source_start, DYNPEAT_PC="TRUE", JUDGE_PC="TRUE", PC_LIM="25")
    previous = _packet(context)
    midnight = _apply(context, previous, 0)
    _mirrors(midnight, growth, peat)
    # The changed temperature/peat inputs expose an erroneously delayed update.
    post = orch.DriverFirstDayStomateStatePacket(
        day_index=1, fields={"growth_day": jnp.ones(3) * 999., "peatC_ok": jnp.ones(3) * 9.,
                             "temp_growth": jnp.asarray([0., 9., 9.]), "peatC": jnp.ones(3) * 999.,
                             "carbon": jnp.asarray([101., 102., 103.])}, provenance_by_field={})
    monkeypatch.setattr(orch, "_sechiba_finalize_state_after_slowproc",
                        lambda previous, slowproc: {**previous, **slowproc})
    merged = orch._paper_day_end_state_packet(previous_state=midnight, stomate_state=post)
    _mirrors(merged, growth, peat)
    np.testing.assert_array_equal(merged.fields_by_component[SLOW]["carbon"], [101., 102., 103.])
    np.testing.assert_array_equal(merged.fields_by_component[SLOW]["temp_growth"], [0., 9., 9.])
    assert int(merged.source_clock_runtime.last_applied_istp) == source_start
    repeated = _apply(context, merged, 0)
    _mirrors(repeated, growth, peat)


def test_jan1_midnight_rebase_then_next_active_keeps_source_frame():
    context = _context(date=(2000, 12, 31), source_start=48)
    midnight = _apply(context, _packet(context), 0)
    clock = _payload(context, midnight, 0)
    assert (int(clock.year), int(clock.month), int(clock.day), float(clock.second)) == (2001, 1, 1, 0.)
    _mirrors(midnight, [4., 8., 1.], [0., 0., 1.])
    rebased = orch.rebase_driver_state_for_year_start(midnight)
    assert rebased.tstep == -1
    assert rebased.source_clock_runtime.binding == midnight.source_clock_runtime.binding
    first = _apply(context, rebased, 0)
    assert int(first.source_clock_runtime.last_applied_istp) == 49
    _mirrors(first, [0., 0., 0.], [0., 0., 1.])


def test_partial_source_day_restart_is_explicitly_rejected_by_complete_day_adapter():
    context = _context(source_start=17)
    state = _packet(context)
    assert int(_payload(context, state, 0).istp) == 17
    assert float(_payload(context, state, 0).second) == 17 * 1800.
    with pytest.raises(SourceClockError, match="partial source-day"):
        orch._require_source_day_alignment(context, state, start=0)


def test_calendar_snapshot_exposes_updated_mirrors_without_overwriting_daily_science():
    context = _context(source_start=48)
    before = _packet(context)
    current = _apply(context, before, 0)
    current.fields_by_component[SLOW]["carbon"] = jnp.ones(3) * 999.
    snapshot = orch._calendar_snapshot_before_stomate(before, current)
    _mirrors(snapshot, [4., 8., 1.], [0., 0., 1.])
    np.testing.assert_array_equal(snapshot.fields_by_component[SLOW]["carbon"], [11., 12., 13.])
    assert snapshot.source_clock_runtime is current.source_clock_runtime


def test_missing_binding_conflicting_mirrors_and_discontinuous_payload_are_rejected():
    context = _context()
    state = _packet(context)
    with pytest.raises(SourceClockError, match="declared"):
        orch._driver_source_clock_payload(NS(), None, tstep=0)
    state.fields_by_component[FINAL]["growth_day"] = jnp.ones(3) * 777.
    with pytest.raises(SourceClockError, match="mirrors conflict"):
        _apply(context, state, 0)
    with pytest.raises(SourceClockError, match="consecutive"):
        _apply(context, _packet(context), 1)


def test_real_jax_scan_transports_fast_codec_dynamic_dates_and_once_only_cursor():
    context = _context(date=(2000, 12, 31), source_start=48)
    initial = fast_state_from_previous_packet(_packet(context))
    # Source48 is midnight/year end,49 is first active/reset;50 is ordinary.
    payloads = [orch._driver_source_clock_payload(context, initial.source_clock_runtime, tstep=i)
                for i in range(3)]
    stacked = jax.tree_util.tree_map(lambda *x: jnp.asarray(x), *payloads)

    def body(state, clock):
        updated = orch._apply_driver_source_calendar(state, context=context, source_clock=clock)
        return updated, updated.component_fields(SLOW)["growth_day"]

    executable = jax.jit(lambda state, clocks: jax.lax.scan(body, state, clocks))
    compiled, growth = executable(initial, stacked)
    np.testing.assert_array_equal(growth, [[4., 8., 1.], [0., 0., 0.], [0., 0., 0.]])
    ordinary = initial
    for clock in payloads:
        ordinary, _ = body(ordinary, clock)
    for left, right in zip(jax.tree_util.tree_leaves(compiled), jax.tree_util.tree_leaves(ordinary), strict=True):
        np.testing.assert_array_equal(left, right)
    second, _ = executable(initial, stacked)
    assert second.source_clock_runtime.binding is context.source_clock_setup.binding
    assert executable._cache_size() == 1
    assert int(second.source_clock_runtime.last_applied_istp) == 50
    roundtrip = previous_packet_from_fast_state(compiled)
    assert set(roundtrip.fields_by_component[FINAL]) == set(_packet(context).fields_by_component[FINAL])
    assert not any("clock" in name for name in roundtrip.fields_by_component[FINAL])


@pytest.mark.parametrize("fast", [False, True], ids=["packet", "fast"])
def test_actual_compact_state_builder_applies_calendar_once_with_physics_spies(monkeypatch, fast):
    context = _context(date=(2001, 1, 2), DYNPEAT_PC="TRUE", JUDGE_PC="TRUE", PC_LIM="25")
    state = _packet(context)
    if fast:
        state = fast_state_from_previous_packet(state)
    calls = _replace_extraction_delegates(monkeypatch)
    result = orch._runtime_next_state_from_components(
        tstep=0, previous_state=state, enerbil_local=None, diffuco_local=None,
        thermosoil_module=None, condveg_module=None, hydrol_module=None,
        source_clock_context=context, source_clock=_payload(context, state, 0))
    assert len(calls) == 6
    assert hasattr(result, "component_fields") == fast
    _mirrors(result, [3., 8., 1.], [1., 0., 1.])
    assert int(result.source_clock_runtime.last_applied_istp) == 1


def test_actual_later_scaffold_state_builder_uses_prior_runtime_offset(monkeypatch):
    context = _context(source_start=17)
    state = _packet(context)
    calls = _replace_extraction_delegates(monkeypatch)
    scaffold = NS(ready_for_next_state=True, tstep=0, source_clock_context=context,
                  enerbil=NS(enerbil_local=None, diffuco_local_enerbil_precall=None,
                             previous_state=state, payload=_payload_template()),
                  thermosoil_module=None, condveg_module=None, hydrol_module=None)
    result = orch.driver_previous_step_state_from_next_step_hydrol_scaffold(scaffold)
    assert len(calls) == 6
    _mirrors(result, [3., 8., 1.], [0., 0., 1.])
    assert int(result.source_clock_runtime.last_applied_istp) == 17
    assert float(_payload(context, result, 0).second) == 17 * 1800.


def test_actual_first_day_scaffold_builder_and_repeated_summary_do_not_reapply(monkeypatch):
    context = _context()
    seed = _packet(context)
    _replace_extraction_delegates(monkeypatch)
    def seed_fields(fields, *args, **kwargs):
        fields[SLOW].update(seed.fields_by_component[SLOW])
        fields[FINAL] = dict(seed.fields_by_component[FINAL])
    monkeypatch.setattr(orch, "_add_enerbil_local_to_previous_fields", seed_fields)
    scaffold = NS(step=NS(tstep=0), source_clock_context=context, payload=_payload_template(),
                  enerbil_first_step_local=None, diffuco_first_step_local_enerbil_precall=None,
                  diffuco_control_salinity=None, diffuco_control_assembly=None,
                  diffuco_first_step_precall=None, thermosoil_first_step_module=None,
                  condveg_first_step_module=None, hydrol_first_step_module=None,
                  first_step_restart_state=None)
    first = orch.driver_previous_step_state_from_timestep_scaffold(scaffold)
    summary = orch.driver_previous_step_state_from_timestep_scaffold(scaffold)
    _mirrors(first, [0., 0., 0.], [0., 0., 1.])
    _mirrors(summary, [0., 0., 0.], [0., 0., 1.])
    assert int(first.source_clock_runtime.last_applied_istp) == 1


def test_actual_cold_payload_builder_distinguishes_initialization_from_active(monkeypatch):
    context = _context()
    _replace_extraction_delegates(monkeypatch)
    v = jnp.ones((3, 2))
    land = jnp.ones(3)
    peat_values = {name: land for name in (
        "peatPET_lastyear", "GSL", "peatPET_thisyear", "precipitation_lastsummer",
        "precipitation_thissummer", "summerpet_long", "summerp_long")}
    peat_values.update(growth_day=jnp.asarray([3., 8., 1.]), peatC=jnp.asarray([25., 24.9, 26.]),
                       peatC_ok=jnp.asarray([0., 0., 1.]))
    slow = NS(**{name: v for name in ("veget", "veget_max", "lai", "frac_nobio", "frac_age", "height")},
              **{name: land for name in ("njsc", "reinf_slope", "clayfraction", "sandfraction",
                                        "soil_ph", "poor_soils", "bulk_density")}, peat=NS(**peat_values))
    hydrol = NS(**{name: land for name in (
        "free_drain_coef", "zwt_force", "fwet_out", "fwet_new", "liqwt_ratio", "wtp", "run2peat",
        "run2man", "wt_ab", "wt_ab_tide", "resdist", "vegtot_old")})
    payloads = {"slowproc_init_pft14": slow, "hydrol_cold_start_state": hydrol,
                "enerbil_surface_state": NS(q_sol_pot=land, temp_sol_pot=land),
                "slowproc_derivvar": NS(assim_param=v, deadleaf_cover=land, temp_growth=jnp.asarray([5.1, 5., 4.9])),
                "condveg_soilalbedo_bg": v, "thermosoil_refSOC": v,
                "enerbil_first_step_local": None, "diffuco_first_step_local_enerbil_precall": None,
                "thermosoil_first_step_module": None, "condveg_first_step_module": None,
                "hydrol_first_step_module": None}
    def finalized(*, previous, **kwargs):
        return {name: land if value is None else value for name, value in previous.items()}
    monkeypatch.setattr(orch, "_sechiba_finalize_state_from_components", finalized)
    inactive = orch._previous_step_state_from_cold_start_payloads(
        payloads, source_clock_context=context, active=False)
    active = orch._previous_step_state_from_cold_start_payloads(payloads, source_clock_context=context)
    _mirrors(inactive, [3., 8., 1.], [0., 0., 1.])
    _mirrors(active, [0., 0., 0.], [0., 0., 1.])
    assert len(active.fields_by_component[FINAL]) == 93
    assert int(inactive.source_clock_runtime.last_applied_istp) == 0
    assert int(active.source_clock_runtime.last_applied_istp) == 1


def _payload_template():
    from jax_orchidee.driver.sechiba_boundary import IntersurfFirstStepPayload
    values = {field.name: jnp.ones(3) for field in fields(IntersurfFirstStepPayload)}
    values.update(kjpindex=3, nbindex=3, missing_fields=(), is_watchout=False,
                  **{name: None for name in orch.WATCHOUT_COUPLING_FIELDS}, veget=jnp.ones((3, 2)),
                  veget_max=jnp.ones((3, 2)), lalo=jnp.ones((3, 2)))
    return IntersurfFirstStepPayload(**values)


def _compiled_fixture(monkeypatch, *, source_start=48, date=(2000, 12, 31)):
    context = _context(source_start=source_start, date=date)
    context.__dict__.update(
        nvm=2, run_def_path=Path("synthetic.def"), ok_laidev=(False, False),
        diffuco_pft14_static_kwargs={"trans_co2_pft_params": {"ok_laidev": (False, False)}, "rstruct_const": 1.},
        **{name: False for name in (
            "ok_explicitsnow", "ok_freeze_cwrr", "ok_thermodynamical_freezing",
            "hydrol_soil_peat_hydro", "peat_hydro", "hydrol_ok_pc", "hydrol_ok_leak", "hydrol_tides")})
    scalars = NS(nstm=1, pref_soil_veg=np.asarray([1, 1]), is_tree=np.asarray([False, False]))
    tables = NS(mineral=NS(imin=0, imax=1))
    arrays = orch.DriverCompiledHydrolTableArrays(*(jnp.ones((1,)) for _ in range(10)))
    diffuco = orch.DriverCompiledDiffucoParameterValues({}, jnp.asarray(1.))
    monkeypatch.setattr(orch, "resolve_condveg_snowcri_alb", lambda *a, **k: 1.)
    monkeypatch.setattr(orch, "resolve_condveg_source_parameters", lambda *a, **k: ())
    monkeypatch.setattr(orch, "_effective_thermosoil_wetdiaglong", lambda *a, **k: False)
    monkeypatch.setattr(orch, "_compiled_hydrol_table_arrays", lambda *a: arrays)
    monkeypatch.setattr(orch, "_compiled_diffuco_parameter_values", lambda *a: diffuco)
    return context, scalars, tables, arrays, diffuco


def _forcing_series(context, state, count, *, start=0):
    result = []
    for tstep in range(start, start + count):
        clock = _payload(context, state, tstep)
        payload = _payload_template()
        step = NS(tstep=tstep, forcing=NS(u=payload.u, v=payload.v))
        item = orch.DriverRuntimeHalfHourInput(step, payload, clock)
        result.append(orch._compiled_half_hour_forcing(item))
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *result)


def test_actual_compiled_scan_builder_executes_dynamic_clock_only_delegate_and_reuses_cache(monkeypatch):
    context, scalars, tables, arrays, diffuco = _compiled_fixture(monkeypatch)
    initial = fast_state_from_previous_packet(_packet(context))
    series = _forcing_series(context, initial, 3)
    calls = []
    def clock_only(config_path, *, previous_state, prepared_context, source_clock, year, tstep, **kwargs):
        calls.append("calendar-only-runtime")
        # These assertions become numeric outputs, so hardcoded1961/48 fails
        # without Python branching on a tracer.
        state = orch._apply_driver_source_calendar(previous_state, context=prepared_context, source_clock=source_clock)
        values = {name: jnp.asarray(year + tstep) for name in orch._COMPILED_STOMATE_ENTRY_FIELDS}
        return NS(next_state=state, entry_payload=values)
    monkeypatch.setattr(orch, "_paper_1961_next_step_runtime_result_compact", clock_only)
    kwargs = dict(config_path="synthetic.yml", context=context, base_payload_template=_payload_template(),
                  run_scalars=scalars, runtime_spec=initial.spec, hydrol_static_template={},
                  hydrol_runtime_static_tables=tables)
    scan = orch._paper_compiled_sechiba_scan(**kwargs)
    assert orch._paper_compiled_sechiba_scan(**kwargs) is scan
    compiled, entries, flux = scan(initial, series, {}, (), arrays, None, diffuco)
    assert flux is None
    _mirrors(compiled, [0., 0., 0.], [0., 0., 1.])
    np.testing.assert_array_equal(entries[0], [2001, 2002, 2003])
    assert int(compiled.source_clock_runtime.last_applied_istp) == 50
    ordinary = initial
    for i in range(3):
        clock = jax.tree_util.tree_map(lambda values: values[i], series.source_clock)
        ordinary = clock_only("synthetic.yml", previous_state=ordinary, prepared_context=context,
                              source_clock=clock, year=clock.year, tstep=clock.local_tstep).next_state
    for left, right in zip(jax.tree_util.tree_leaves(compiled), jax.tree_util.tree_leaves(ordinary), strict=True):
        np.testing.assert_array_equal(left, right)
    trace_calls = len(calls)
    reused, _, _ = scan(initial, series, {}, (), arrays, None, diffuco)
    assert int(reused.source_clock_runtime.last_applied_istp) == 50
    assert len(calls) == trace_calls
    assert calls  # Physical kernels remain absent; only this replacement ran.
    # Reuse this same executable/context on genuinely different calendar dates.
    next_day = replace(initial, tstep=48, source_clock_runtime=replace(
        initial.source_clock_runtime, last_applied_istp=np.int64(96)))
    jan2_series = _forcing_series(context, next_day, 3, start=49)
    assert int(jan2_series.source_clock.day[0]) == 2
    jan2, jan2_entries, _ = scan(next_day, jan2_series, {}, (), arrays, None, diffuco)
    assert len(calls) == trace_calls
    _mirrors(jan2, [3., 8., 1.], [1., 1., 1.])
    np.testing.assert_array_equal(jan2_entries[0], [2050, 2051, 2052])
    new_binding = replace(context.source_clock_setup.binding, run_id="different-synthetic-run")
    new_context = NS(**vars(context))
    new_context.source_clock_setup = replace(context.source_clock_setup, binding=new_binding,
        runtime=replace(context.source_clock_setup.runtime, binding=new_binding))
    assert orch._paper_compiled_sechiba_scan(**{**kwargs, "context": new_context}) is not scan


@pytest.mark.parametrize("fast", [False, True], ids=["ordinary-packet", "compact-fast"])
def test_actual_later_halfhour_adapter_transports_nontrivial_source_offset(monkeypatch, fast):
    context = _context(source_start=17)
    context.hydrol_soil_peat_hydro = False
    initial = _packet(context)
    inputs = tuple(orch.DriverRuntimeHalfHourInput(NS(tstep=i), _payload_template(),
                   _payload(context, initial, i)) for i in range(3))
    called = []
    def clock_only(config_path, *, previous_state, prepared_context, tstep, prebuilt_input, **kwargs):
        called.append(int(tstep))
        state = orch._apply_driver_source_calendar(previous_state, context=prepared_context,
                                                   source_clock=prebuilt_input.source_clock)
        return NS(ok=True, next_state=state, metadata=NS(synthetic=True), entry_payload={"kjit": tstep + 1})
    monkeypatch.setattr(orch, "_paper_1961_next_step_runtime_result_compact", clock_only)
    monkeypatch.setattr(orch, "_filter_day_state_gaps", lambda *a: ())
    day = orch.DriverRuntimeDayTransitionInputs(
        year=1961, start=0, steps_per_stomate=3, run_def_path=Path("synthetic.def"),
        day_start_bundle=NS(), hydrol_static_template=None, hydrol_runtime_static_tables=None,
        diffuco_day_static_cache=None, half_hour_inputs=inputs)
    result = orch._paper_1961_later_day_half_hour_transition(
        "synthetic.yml", context=context, initial_state=initial, day_inputs=day,
        fixed_format_trace_dir=None, static_trace_fields=None, module_jit=False,
        diffuco_local_jit=False, use_fast_state_loop=fast)
    assert called == [0, 1, 2]
    assert len(result.completed_entry_payloads) == 3
    assert int(result.current_state.source_clock_runtime.last_applied_istp) == 19
    _mirrors(result.current_state, [3., 8., 1.], [0., 0., 1.])


def _compact_physics_spies(monkeypatch, context, state):
    """Replace every physical producer while retaining the full compact caller."""
    _replace_extraction_delegates(monkeypatch)
    land, veg = jnp.ones(3), jnp.ones((3, 2))
    context.__dict__.update(**{name: False for name in (
        "river_routing", "ok_explicitsnow", "hydrol_soil_peat_hydro", "stomate_ok_dgvm",
        "ok_freeze_cwrr", "ok_thermodynamical_freezing", "hydrol_do_ponds", "hydrol_ok_pc",
        "hydrol_ok_leak", "hydrol_ok_ru2peat", "hydrol_ok_wt_ab", "hydrol_tides", "hydrol_agri_peat",
        "hydrol_dyn_nroot_larix", "impaze", "rough_dyn")})
    context.__dict__.update(**{name: land for name in (
        "hydrol_humcste", "hydrol_zz_mm", "ext_coeff_vegetfrac", "hydrol_throughfall_by_pft",
        "hydrol_dh_mm", "diaglev", "hydrol_dz_mm", "hydrol_cwrr_ks", "hydrol_reinf_slope",
        "hydrol_depth_max_h", "hydrol_fr_dt", "hydrol_froz_frac_corr", "hydrol_smtot_corr",
        "hydrol_max_froz_hydro", "hydrol_max_wt_ab", "thermosoil_satsoil")})
    context.__dict__.update(nvm=2, ok_laidev=(False, False), dt_sechiba_days=1800. / 86400.,
                            min_wind=0., diffuco_pft14_static_kwargs={},
                            cwrr_grid=NS(**{name: land for name in ("dlt", "dz1", "zlt", "znt", "dz5")}))
    state.fields_by_component[SLOW].update(veget=veg, veget_max=veg, lai=veg, frac_nobio=veg,
                                            height=veg, totfrac_nobio=land)
    state.fields_by_component["diffuco_previous_step_state"] = {"height": veg, "soilalbedo_bg": veg}
    state.fields_by_component["thermosoil_previous_step_state"] = {
        name: land for name in ("ptn", "cgrnd", "dgrnd", "cgrnd_snow", "dgrnd_snow", "lambda_snow", "gtemp", "refSOC")}
    state.fields_by_component["hydrol_previous_step_state"] = {"njsc": land}
    calls = []
    def record(name, result):
        def spy(*args, **kwargs):
            calls.append(name)
            return result
        monkeypatch.setattr(orch, name, spy)
    monkeypatch.setattr(orch, "_driver_wind_z0_from_previous_state", lambda *a: .1)
    monkeypatch.setattr(orch, "intersurf_payload_with_driver_wind_z0", lambda payload, **k: payload)
    monkeypatch.setattr(orch, "_diffuco_restart_payload_from_previous_state", lambda *a: ({}, (), ()))
    monkeypatch.setattr(orch, "_enerbil_state_payload_from_previous_state", lambda *a: ({}, (), ()))
    monkeypatch.setattr(orch, "_hydrol_state_namespace_from_previous_state", lambda *a: (NS(), ()))
    monkeypatch.setattr(orch, "_paper_mangrove_pft_index", lambda *a: 1)
    monkeypatch.setattr(orch, "_validated_prepared_pft_axis", lambda value, **k: value)
    monkeypatch.setattr(orch, "_enerbil_driver_coverage_payload", lambda *a, **kwargs: {})
    monkeypatch.setattr(orch, "_hydrol_current_diffuco_payload", lambda *a: {})
    diffuco = NS(result=NS(payload=NS(payload={})))
    record("assemble_diffuco_first_step_precall_payload", NS(ok=True, payload={}))
    record("run_pft14_local_enerbil_precall_from_first_step_precall", diffuco)
    record("assemble_enerbil_first_step_precall_payload", NS(ok=True))
    record("run_enerbil_first_step_local_from_precall", NS(payload={}))
    record("assemble_hydrol_first_step_precall_payload", NS(ok=True))
    snow = NS(**{name: land for name in ("snow", "snow_nobio", "snowrho", "snowdz", "snow_age", "snow_nobio_age", "snowtemp")})
    record("run_hydrol_first_step_module_from_precall", NS(ok=True, snow_state=snow,
                                                           diagnostics=NS(), thermosoil_moisture=land))
    record("run_condveg_first_step_module", NS(ok=True, module=NS()))
    record("run_thermosoil_first_step_module", NS(ok=True))
    record("_paper_next_step_runtime_entry_payload_from_components", {"kjit": 1})
    return calls


def test_actual_complete_compact_function_invokes_calendar_builder_with_all_physics_replaced(monkeypatch):
    context = _context(date=(2001, 1, 2), DYNPEAT_PC="TRUE", JUDGE_PC="TRUE", PC_LIM="25")
    state = _packet(context)
    calls = _compact_physics_spies(monkeypatch, context, state)
    scalars = NS(nstm=1, pref_soil_veg=np.asarray([1, 1]), is_tree=np.asarray([False, False]),
                 is_peat=np.asarray([False, False]), z0_over_height=1., ratio_z0m_z0h=1.)
    context.run_scalars = scalars
    step = NS(tstep=0, run_scalars=scalars, forcing=NS())
    half_hour = orch.DriverRuntimeHalfHourInput(step, _payload_template(), _payload(context, state, 0))
    result = orch._paper_1961_next_step_runtime_result_compact(
        "synthetic.yml", previous_state=state, year=1961, tstep=0, fixed_format_trace_dir=None,
        static_trace_fields=None, used_run_def_path=None, prepared_context=context,
        hydrol_static_template={}, diffuco_day_static_cache=NS(control_salinity=None,
            control_inundation=None, slowproc_derivvar=None, slowproc_derivvar_payload={}),
        module_jit=False, diffuco_local_jit=False, prebuilt_input=half_hour)
    assert result.ok
    assert len(calls) == 9
    _mirrors(result.next_state, [3., 8., 1.], [1., 0., 1.])
    assert int(result.next_state.source_clock_runtime.last_applied_istp) == 1


def test_actual_ordinary_runtime_wrapper_and_builder_execute_with_scaffold_spy(monkeypatch):
    context = _context(source_start=48)
    context.run_def_path, context.ok_laidev = Path("synthetic.def"), (False, False)
    state = _packet(context)
    _replace_extraction_delegates(monkeypatch)
    scalars = NS(pref_soil_veg=jnp.asarray([1, 1]), nstm=1,
                 is_tree=jnp.asarray([False, False]), is_peat=jnp.asarray([False, False]))
    scaffold = NS(ready_for_next_state=True, tstep=0, source_clock_context=context,
                  enerbil=NS(enerbil_local=None, diffuco_local_enerbil_precall=None,
                             previous_state=state, step=NS(run_scalars=scalars), payload=_payload_template()),
                  thermosoil_module=None, condveg_module=None, hydrol_module=None)
    monkeypatch.setattr(orch, "prepare_paper_1961_driver_context", lambda *a, **k: context)
    monkeypatch.setattr(orch, "paper_1961_next_step_hydrol_precall_scaffold", lambda *a, **k: scaffold)
    monkeypatch.setattr(orch, "_paper_next_step_runtime_entry_payload", lambda *a, **k: {"kjit": 1})
    result = orch.paper_1961_next_step_runtime_result("synthetic.yml", previous_state=state, tstep=0,
                                                   module_jit=False, diffuco_local_jit=False)
    assert result.ok
    _mirrors(result.next_state, [4., 8., 1.], [0., 0., 1.])
    assert int(result.next_state.source_clock_runtime.last_applied_istp) == 48


def test_actual_compiled_block_builder_and_executable_carry_clock_and_reuse_context(monkeypatch):
    context, scalars, tables, arrays, diffuco = _compiled_fixture(monkeypatch, source_start=1)
    context.run_def_values.update(DYNPEAT_PC="TRUE", JUDGE_PC="TRUE", PC_LIM="25")
    context.runtime = NS(dt_stomate=86400., dt_sechiba=1800.)
    context.first_step_restart_state = NS(stomate=jnp.asarray(0.),
        stomate_readstart=NS(season_state=NS(provenance=(), _asdict=lambda: {"provenance": ()})))
    monkeypatch.setattr(orch, "StomateRestartSeasonState", lambda **kwargs: NS(**kwargs))
    monkeypatch.setattr(orch, "_compiled_landpoint_payload", lambda *a: {})
    monkeypatch.setattr(orch, "_paper_1961_later_day_transition_inputs",
                        lambda **kwargs: NS(compiled_base_payload_template=_payload_template()))
    trace_calls = []
    def calendar_day(config_path, *, previous_state, prepared_context, compiled_forcing_series,
                     model_day_number, **kwargs):
        trace_calls.append("calendar-only-day")
        def body(state, clock):
            next_state = orch._apply_driver_source_calendar(state, context=prepared_context, source_clock=clock)
            return next_state, None
        state, _ = jax.lax.scan(body, previous_state, compiled_forcing_series.source_clock)
        return NS(day_end_state=state, daily_modelout=NS(
            modelout_fields={"growth_day": state.component_fields(SLOW)["growth_day"]},
            modelout=jnp.asarray(model_day_number)))
    monkeypatch.setattr(orch, "paper_1961_driver_later_day_runtime_result", calendar_day)
    initial_packet = _packet(context)
    initial = fast_state_from_previous_packet(initial_packet)
    series = _forcing_series(context, initial, 144)
    blocks = jax.tree_util.tree_map(lambda a: a.reshape((3, 48) + a.shape[1:]), series)
    day_numbers = jnp.asarray([366, 367, 368])
    kwargs = dict(config_path="synthetic.yml", context=context, initial_state=initial_packet,
                  year=1961, block_forcing=blocks, block_day_numbers=day_numbers,
                  prebound_hydrol_runtime_static_tables=tables, daily_carbon_dispatch={},
                  stomate_parameter_values=None)
    # No scientific parameter producer is entered by the calendar-only lower.
    monkeypatch.setattr(orch, "_compiled_stomate_parameter_values", lambda *a: None)
    executable, spec = orch._paper_compiled_later_day_block_executable(**kwargs)
    again, again_spec = orch._paper_compiled_later_day_block_executable(**kwargs)
    assert again is executable and again_spec == spec
    result, outputs = executable(initial, blocks, day_numbers, None, arrays, {}, jnp.asarray(0.), {}, diffuco)
    assert int(result.tstep) == 143
    assert int(result.source_clock_runtime.last_applied_istp) == 144
    _mirrors(result, [2., 0., 0.], [1., 0., 1.])
    np.testing.assert_array_equal(outputs[0]["growth_day"], [[4., 8., 1.], [1., 0., 0.], [2., 0., 0.]])
    np.testing.assert_array_equal(outputs[1], day_numbers)
    ordinary = initial
    for i in range(144):
        clock = jax.tree_util.tree_map(lambda a: a[i], series.source_clock)
        ordinary = orch._apply_driver_source_calendar(ordinary, context=context, source_clock=clock)
    for left, right in zip(jax.tree_util.tree_leaves(result), jax.tree_util.tree_leaves(ordinary), strict=True):
        np.testing.assert_array_equal(left, right)
    before_reuse = len(trace_calls)
    reused, _ = executable(initial, blocks, day_numbers, None, arrays, {}, jnp.asarray(0.), {}, diffuco)
    assert len(trace_calls) == before_reuse
    assert int(reused.source_clock_runtime.last_applied_istp) == 144
    later_initial = replace(result, tstep=143)
    later_series = _forcing_series(context, later_initial, 144, start=144)
    later_blocks = jax.tree_util.tree_map(lambda a: a.reshape((3, 48) + a.shape[1:]), later_series)
    assert int(later_blocks.source_clock.day[0, 0]) == 3
    later_packet = previous_packet_from_fast_state(later_initial)
    later_executable, _ = orch._paper_compiled_later_day_block_executable(
        **{**kwargs, "initial_state": later_packet, "block_forcing": later_blocks})
    assert later_executable is executable
    later, _ = later_executable(later_initial, later_blocks, day_numbers, None, arrays, {}, jnp.asarray(0.), {}, diffuco)
    assert len(trace_calls) == before_reuse
    assert int(later.source_clock_runtime.last_applied_istp) == 288
    _mirrors(later, [5., 0., 0.], [1., 0., 1.])


def test_single_compiled_day_rejects_two_consecutive_days_of_forcing(monkeypatch):
    context, _, _, _, _ = _compiled_fixture(monkeypatch, source_start=1)
    state = _packet(context)
    forcing = _forcing_series(context, state, 96)

    def unexpected_assembly(**kwargs):
        pytest.fail("The single-day entry admitted two complete days of forcing")

    monkeypatch.setattr(orch, "_paper_1961_later_day_transition_inputs", unexpected_assembly)
    with pytest.raises(SourceClockError, match="one complete source day"):
        orch.paper_1961_driver_later_day_runtime_result(
            "synthetic.yml", previous_state=state, day_index=2, start_tstep=0,
            prepared_context=context, use_compiled_sechiba_day=True, compiled_forcing_series=forcing,
        )


def test_actual_compiled_block_builder_rejects_source17_local0_rows_before_lower(monkeypatch):
    context, _, tables, _, _ = _compiled_fixture(monkeypatch, source_start=17)
    state = _packet(context)
    series = _forcing_series(context, state, 48)
    block = jax.tree_util.tree_map(lambda a: a[None, ...], series)
    day_numbers = jnp.asarray([2])
    with pytest.raises(SourceClockError, match="not aligned with source day"):
        orch._paper_compiled_later_day_block_executable(
            "synthetic.yml", context=context, initial_state=state, year=1961,
            block_forcing=block, block_day_numbers=day_numbers,
            prebound_hydrol_runtime_static_tables=tables, daily_carbon_dispatch={})


@pytest.mark.parametrize("mode,run_id,expected_cursor,keep_nroot", [
    ("continuous", None, 49, True), ("new_native_run", "synthetic-new-run", 49, False),
])
@pytest.mark.parametrize("partial", [False, True], ids=["whole-source-day", "partial-source-day"])
def test_actual_restart_year_wrapper_binds_changed_forcing_with_real_metadata_factory(
    monkeypatch, tmp_path, mode, run_id, expected_cursor, keep_nroot, partial,
):
    from netCDF4 import Dataset
    from jax_orchidee.driver import domain
    old_start = 49 if partial and mode == "continuous" else 48
    old_context = _context(date=(2000, 12, 31), source_start=old_start)
    state = _apply(old_context, _packet(old_context), 0)
    state.fields_by_component["hydrol_previous_step_state"] = {"nroot": jnp.asarray([42.])}
    # A new native year initializes US and vegstress from the restored fields.
    state.fields_by_component[FINAL].update(us=jnp.zeros((3, 2, 1, 1)), vegstress=jnp.zeros((3, 2)))
    forcing = tmp_path / "new-year-forcing.nc"
    with Dataset(forcing, "w") as dataset:
        dataset.createDimension("time", 96)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "seconds since 2001-01-01 00:00:00"
        time.calendar = "noleap"
        time.time_origin = "2001-01-01 00:00:00"
        time[:] = np.arange(96) * 1800.
    values = dict(RESTART_FILEIN="NONE", SECHIBA_restart_in="NONE", STOMATE_RESTART_FILEIN="NONE",
                  DT_SECHIBA="1800")
    if partial and mode == "new_native_run":
        values["TIME_SKIP"] = "1"
    run_scalars = NS(dt_sechiba=1800., pref_soil_veg=np.asarray([1, 1]))
    context_values = dict(source_clock_setup=old_context.source_clock_setup, config_path=tmp_path / "synthetic.yml",
                          run_def_path=tmp_path / "synthetic.def", run_def_values=values,
                          first_step_bundle=NS(year=2001, run_scalars=run_scalars),
                          first_step_restart_state=None, run_scalars=run_scalars, hydrol_tides=False,
                          watchout_initialize_main_tstep=None,
                          diffuco_pft14_static_kwargs={"ldq_cdrag_from_gcm": True},
                          reference_run_dir=None, dt_sechiba=1800.,
                          runtime=NS(dt_stomate=86400., dt_sechiba=1800.))
    Context = make_dataclass("SyntheticRestartContext", [(name, object) for name in context_values], frozen=True)
    context = Context(**context_values)
    # Only file selection is synthetic: the actual config->metadata->DIM2
    # lifecycle factory, rebase and restart-year caller all execute.
    monkeypatch.setattr(domain, "load_case_config", lambda path: NS())
    monkeypatch.setattr(domain, "forcing_file_for_year", lambda config, year: forcing)
    monkeypatch.setattr(orch, "driver_year_handoff_state_gaps", lambda *a: ())
    monkeypatch.setattr(orch, "restore_sechiba_integer_source_fields", lambda fields, **k: dict(fields))
    step_reads = []

    def first_step_bundle(context, *, year, tstep):
        step_reads.append((year, tstep))
        return NS(forcing=NS(is_watchout=False), domain=NS())

    monkeypatch.setattr(orch, "_paper_1961_step_bundle_from_context", first_step_bundle)
    monkeypatch.setattr(orch, "build_intersurf_first_step_payload", lambda *a, **k: _payload_template())
    monkeypatch.setattr(orch, "_driver_wind_z0_from_previous_state", lambda *a: .1)
    monkeypatch.setattr(orch, "restart_thermosoil_initialization_supported", lambda *a: False)
    def forbidden_thermal(*a, **k):
        raise AssertionError("calendar-only restart adapter must not execute physical thermal initialization")
    monkeypatch.setattr(orch, "restart_thermosoil_initial_fields", forbidden_thermal)
    observed = []
    def calendar_only_day(config_path, *, previous_state, prepared_context, year, **kwargs):
        observed.append((previous_state, prepared_context))
        return orch.DriverRuntimeDayResult(
            year=year, day_index=1, start_tstep=0, steps_per_stomate=48,
            daily_modelout=orch.DriverDailyModelout(1, 0, 47, {}, {}),
            day_end_state=_apply(prepared_context, previous_state, 0),
            stopped_at_tstep=None, missing_components=(),
        )
    monkeypatch.setattr(orch, "paper_1961_driver_later_day_runtime_result", calendar_only_day)

    def restart():
        return orch.paper_1961_driver_restart_year_start_day_result(
            context.config_path, previous_year_end_state=state, year=2001, prepared_context=context,
            run_mode=mode, run_id=run_id, module_jit=False, diffuco_local_jit=False)

    if partial:
        with pytest.raises(SourceClockError, match="partial source-day"):
            restart()
        assert step_reads == []
        assert observed == []
        return

    day = restart()
    assert day.ready_for_day_end_state and day.prepared_context.watchout_initialize_main_tstep is None
    result = day.day_end_state
    assert len(observed) == 1
    assert step_reads == [(2001, 0)]
    rebased, rebound_context = observed[0]
    assert rebound_context is not context
    assert rebound_context.source_clock_setup.forcing.origin_ymds == (2001, 1, 1, 0.)
    assert rebound_context.diffuco_pft14_static_kwargs["ldq_cdrag_from_gcm"] == (mode == "continuous")
    assert ("nroot" in rebased.fields_by_component["hydrol_previous_step_state"]) == keep_nroot
    assert int(result.source_clock_runtime.last_applied_istp) == expected_cursor
    assert (result.source_clock_runtime.binding == state.source_clock_runtime.binding) == (mode == "continuous")
    _mirrors(result, [0., 0., 0.], [0., 0., 1.])


@pytest.mark.parametrize(("prepared_year", "mode", "skip", "first_index", "rejection"), [
    (None, "new_native_run", "0", 0, None),
    (1961, "new_native_run", "0", 0, None),
    (1962, "new_native_run", "4", 48, None),
    (1961, "continuous", "0", 0, None),
    (1961, "continuous", "1", None, "skips/reverses"),
    (None, "new_native_run", "1", None, "partial source-day"),
])
def test_memory_year_entry_binds_current_file_and_previous_cursor(
    monkeypatch, tmp_path, prepared_year, mode, skip, first_index, rejection,
):
    from netCDF4 import Dataset
    from jax_orchidee.driver import domain
    from jax_orchidee.driver.source_clock import source_clock_setup_from_config

    paths = {}
    for year in (1961, 1962):
        paths[year] = tmp_path / f"forcing-{year}.nc"
        with Dataset(paths[year], "w") as dataset:
            dataset.createDimension("time", 16)
            time = dataset.createVariable("time", "f8", ("time",))
            time.units = f"seconds since {year}-01-01 00:00:00"
            time.calendar = "noleap"
            time[:] = (np.arange(16) + 1) * 21600.0
    monkeypatch.setattr(domain, "load_case_config", lambda path: {})
    monkeypatch.setattr(domain, "forcing_file_for_year", lambda config, year: paths[year])

    cursor = 365 * 48
    previous = _packet(_context(date=(1961, 1, 1), source_start=cursor + 1))
    values = dict(RESTART_FILEIN="NONE", SECHIBA_restart_in="NONE",
                  STOMATE_RESTART_FILEIN="NONE", DT_SECHIBA="1800", TIME_SKIP=skip)
    config_path = tmp_path / "case.yaml"

    def prepare(path, *, year=1961, **kwargs):
        fields = dict(
            config_path=path, run_def_path=tmp_path / "used_run.def",
            run_def_values=values, first_step_bundle=NS(year=year),
            reference_run_dir=None, hydrol_tides=False, watchout_initialize_main_tstep=None,
            run_scalars=NS(pref_soil_veg=np.asarray([1, 1])),
            runtime=NS(dt_stomate=86400.0, dt_sechiba=1800.0),
        )
        setup = source_clock_setup_from_config(path, run_def_values=values,
            first_step_bundle=fields["first_step_bundle"])
        if year == 1961:
            setup = replace(setup, binding=previous.source_clock_runtime.binding,
                            runtime=previous.source_clock_runtime)
        fields["source_clock_setup"] = setup
        return make_dataclass("MemoryYearContext", [(key, object) for key in fields])(**fields)

    class ReachedStep(Exception):
        pass

    def first_step(context, *, year, tstep):
        assert rejection is None
        setup = context.source_clock_setup
        assert (year, tstep, setup.forcing_year) == (1962, 0, 1962)
        assert setup.forcing.origin_ymds == (1962, 1, 1, 0.0)
        assert setup.time_control.for_offset == -1460
        assert setup.time_control.istp_old == cursor
        assert setup.time_control.istp == cursor + 1 + first_index
        assert setup.binding.clock_at_istp(setup.time_control.istp).year == 1962
        assert (setup.binding == previous.source_clock_runtime.binding) == (mode == "continuous")
        assert [orch._paper_forcing_model_tstep(context, year=year, tstep=step)
                for step in (0, 1)] == [first_index, first_index + 1]
        raise ReachedStep

    monkeypatch.setattr(orch, "prepare_paper_1961_driver_context", prepare)
    monkeypatch.setattr(orch, "driver_year_handoff_state_gaps", lambda state: ())
    monkeypatch.setattr(orch, "restore_sechiba_integer_source_fields", lambda fields, **kwargs: dict(fields))
    monkeypatch.setattr(orch, "_paper_1961_step_bundle_from_context", first_step)
    monkeypatch.setattr(orch, "readstart_thawed_humidity_controls", lambda values: (False, 0.5))
    previous.fields_by_component[FINAL].update(us=jnp.zeros((3, 2, 1, 1)), vegstress=jnp.zeros((3, 2)))
    context = None if prepared_year is None else prepare(config_path, year=prepared_year)
    expected = (pytest.raises(ReachedStep) if rejection is None
                else pytest.raises(SourceClockError, match=rejection))
    with expected:
        orch.paper_1961_driver_restart_year_multiday_modelout_lite_run(
            config_path, previous_year_end_state=previous, ndays=1, year=1962,
            prepared_context=context, run_mode=mode, module_jit=False, diffuco_local_jit=False,
        )


@pytest.mark.parametrize("supply_context", (False, True))
def test_native_year_rejects_conflicting_file_and_memory_restart_times(monkeypatch, supply_context):
    context = _context(date=(1961, 1, 1), source_start=17521)
    setup = context.source_clock_setup
    file_clock = NS(calendar="noleap", date0=setup.binding.date0, itau=17520, dt=1800.0)
    context.source_clock_setup = replace(setup, forcing_year=1962, restart=NS(driver=file_clock))
    packet = _packet(context)
    packet = replace(packet, source_clock_runtime=replace(packet.source_clock_runtime,
        last_applied_istp=np.int64(17568), anchor_source_istp=np.int64(17569)))
    monkeypatch.setattr(orch, "prepare_paper_1961_driver_context", lambda *a, **k: context)
    monkeypatch.setattr(orch, "driver_year_handoff_state_gaps", lambda *a: ())
    with pytest.raises(SourceClockError, match="restart files and memory checkpoint"):
        orch.paper_1961_driver_restart_year_start_day_result(
            context.config_path, previous_year_end_state=packet, year=1962,
            prepared_context=context if supply_context else None,
        )


@pytest.mark.parametrize("route, multiday, first_istp, error", [
    ("legacy", False, None, "versioned source clock"),
    ("legacy", True, None, "versioned source clock"),
    ("cold-context", False, None, "versioned source clock"),
    ("skip", True, None, "versioned source clock"),
    ("files", False, 17521, None),
    ("prepared-files", True, 17521, None),
    ("reused-file-id", False, None, "requires a new run_id"),
    ("spliced-file-context", True, None, "prepared year restart clock is inconsistent"),
    ("files-reset", False, 1, None),
    ("prepared-cursor", False, 17521, None),
    ("prepared-run-id", True, 17521, None),
    ("inconsistent-cursor", False, None, "prepared year restart clock is inconsistent"),
    ("inconsistent-skip", False, None, "prepared year restart clock is inconsistent"),
    ("reset", True, 1, None),
    ("typed", False, 17521, None),
    ("conflicting-typed", False, None, "local tstep conflicts with source clock cursor"),
    ("continuous", True, 17521, None),
    ("legacy-continuous", True, None, "continuous year restart requires"),
])
def test_annual_entry_requires_authoritative_restart_chronology(
    monkeypatch, tmp_path, route, multiday, first_istp, error,
):
    from netCDF4 import Dataset

    from jax_orchidee.driver import domain
    from jax_orchidee.driver.source_clock import source_clock_setup_from_config
    from tests.unit.test_driver_source_clock_io import _write_restart

    forcing = tmp_path / "forcing.nc"
    with Dataset(forcing, "w") as dataset:
        dataset.createDimension("time", 16)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units, time.calendar = "seconds since 1962-01-01 00:00:00", "noleap"
        time[:] = (np.arange(16) + 1) * 21600.0
    previous = _packet(_context(date=(1961, 1, 1)), tstep=17519)
    cursor = replace(previous.source_clock_runtime, last_applied_istp=np.int64(17520))
    clocked = route in {"typed", "continuous", "conflicting-typed"}
    if route == "conflicting-typed":
        cursor = replace(cursor, last_applied_istp=np.int64(0))
    previous = replace(previous, source_clock_runtime=cursor if clocked else None)
    previous.fields_by_component[FINAL].update(us=jnp.zeros((3, 2, 1, 1)), vegstress=jnp.zeros((3, 2)))
    values = {"DT_SECHIBA": "1800"}
    for key, name in (("RESTART_FILEIN", "driver"), ("SECHIBA_restart_in", "sechiba"),
                      ("STOMATE_RESTART_FILEIN", "stomate")):
        values[key] = str(_write_restart(tmp_path / f"{name}.nc", cursor)) if route in {
            "files", "prepared-files", "reused-file-id", "spliced-file-context", "files-reset",
        } else "NONE"
    if route == "skip":
        values["TIME_SKIP"] = "4"
    if route in {"reset", "files-reset"}:
        values["DRIVER_reset_time"] = "TRUE"
    monkeypatch.setattr(domain, "load_case_config", lambda path: {})
    monkeypatch.setattr(domain, "forcing_file_for_year", lambda config, year: forcing)
    config_path = tmp_path / "case.yaml"
    fields = dict(config_path=config_path, run_def_path=tmp_path / "used_run.def", run_def_values=values,
                  first_step_bundle=NS(year=1962), reference_run_dir=None,
                  hydrol_tides=False, hydrol_tide_run=None, watchout_initialize_main_tstep=None,
                  run_scalars=NS(pref_soil_veg=np.asarray([1, 1])), diffuco_pft14_static_kwargs={},
                  runtime=NS(dt_stomate=86400.0, dt_sechiba=1800.0))
    setup = source_clock_setup_from_config(
        config_path, run_def_values=values, first_step_bundle=fields["first_step_bundle"],
    )
    if route == "spliced-file-context":
        cold = source_clock_setup_from_config(
            config_path, first_step_bundle=fields["first_step_bundle"], run_def_values={
                "DT_SECHIBA": "1800", "RESTART_FILEIN": "NONE",
                "SECHIBA_restart_in": "NONE", "STOMATE_RESTART_FILEIN": "NONE",
            },
        )
        setup = replace(cold, restart=setup.restart)
    if route in {"prepared-cursor", "prepared-run-id", "inconsistent-cursor", "inconsistent-skip"}:
        clock_values = {**values, "TIME_SKIP": "4"} if route == "inconsistent-skip" else values
        setup = source_clock_setup_from_config(
            config_path, run_def_values=clock_values, first_step_bundle=fields["first_step_bundle"],
            previous_runtime=cursor, restart_from_memory=True,
        )
    if route == "inconsistent-cursor":
        setup = replace(setup, runtime=replace(setup.runtime, anchor_source_istp=np.int64(17569)))
    context = make_dataclass("AnnualAdmissionFixture", [(key, object) for key in (*fields, "source_clock_setup")])(
        **fields, source_clock_setup=setup,
    )
    monkeypatch.setattr(orch, "prepare_paper_1961_driver_context", lambda *a, **k: context)
    monkeypatch.setattr(orch, "driver_year_handoff_state_gaps", lambda state: ())
    if error:
        def fail_before_physics(*args, **kwargs):
            pytest.fail("unsupported annual restart reached physical initialization")
        monkeypatch.setattr(orch, "restore_sechiba_integer_source_fields", fail_before_physics)
    else:
        monkeypatch.setattr(orch, "restore_sechiba_integer_source_fields", lambda fields, **kwargs: dict(fields))
    class ReachedStep(Exception):
        pass

    def first_step(context, *, year, tstep):
        actual = context.source_clock_setup
        assert actual.time_control.istp == first_istp
        assert orch._paper_forcing_model_tstep(context, year=year, tstep=tstep) == 0
        assert (actual.binding == cursor.binding) == (route == "continuous")
        if route == "prepared-run-id":
            assert actual.binding.run_id == "caller-chosen-id"
        raise ReachedStep

    monkeypatch.setattr(orch, "_paper_1961_step_bundle_from_context", first_step)
    options = dict(previous_year_end_state=previous, year=1962)
    if route not in {"legacy", "skip", "files", "reused-file-id", "files-reset",
                     "reset", "typed", "conflicting-typed"}:
        options["prepared_context"] = context
    if route in {"continuous", "legacy-continuous"}:
        options["run_mode"] = "continuous"
    if route == "prepared-run-id":
        options["run_id"] = "caller-chosen-id"
    if route == "reused-file-id":
        options["run_id"] = cursor.binding.run_id
    expected = pytest.raises(SourceClockError, match=error) if error else pytest.raises(ReachedStep)
    with expected:
        if multiday:
            orch.paper_1961_driver_restart_year_multiday_modelout_lite_run(config_path, ndays=1, **options)
        else:
            orch.paper_1961_driver_restart_year_start_day_result(config_path, **options)


@pytest.mark.parametrize("store_liquid", (False, True))
def test_day_fold_restores_wetness_fields_omitted_from_legacy_packet(monkeypatch, store_liquid):
    npts, nvm = 2, 14
    t2m = jnp.zeros(npts, dtype=jnp.float64)
    wetness = {"t2m_daily": t2m}
    if store_liquid:
        wetness["liqwt_daily"] = jnp.asarray([0.1875, 0.6875])
    previous = orch.DriverPreviousStepStatePacket(
        tstep=47, provenance_by_component={},
        fields_by_component={
            SLOW: dict(daily_accumulators=wetness, lai=jnp.zeros((npts, nvm)),
                       biomass=None, sla_calc=None, resp_maint_part=None),
            "hydrol_previous_step_state": {"humrel": None},
        },
    )
    parameters = NS(coeff_maint_zero=None, maint_resp_slope=None, ext_coeff=None, is_tree=None)
    context = NS(stomate_static=NS(parameters=parameters),
                 stomate_boundary=NS(kwargs={"rprof": jnp.zeros((npts, 2)), "z_soil": None}))

    def fold(*, accumulator_state, do_slow_flags, **kwargs):
        assert do_slow_flags == (False,) * 47 + (True,)
        return accumulator_state

    monkeypatch.setattr(orch, "stomate_daily_process_fold_from_entries", fold)
    result = orch._paper_later_day_daily_process_from_completed_entries(
        first_step=None, previous_state=previous, config_path="fixture.yaml",
        run_def_path=Path("fixture.def"), run_def_values={"PEAT_OCCUR": "FALSE"},
        runtime=NS(dt_stomate=86400.0, dt_sechiba=1800.0, dt_days=1.0), start_tstep=48,
        entry_payloads=({"t2m": t2m},) * 48, prepared_context=context,
        compiled_complete_source_day=True,
    )
    np.testing.assert_array_equal(result.fwet_daily, t2m)
    np.testing.assert_array_equal(result.liqwt_daily, wetness.get("liqwt_daily", t2m))
    assert "fwet_daily" not in wetness


@pytest.mark.parametrize("entry", [
    "paper_1961_driver_cold_start_day_scaffold", "paper_1961_driver_day_scaffold",
    "paper_1961_driver_later_day_scaffold",
])
def test_actual_whole_day_scaffolds_reject_partial_source_day_before_physics(monkeypatch, entry):
    context = _context(source_start=17)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("physical delegate entered")
        raise AssertionError("partial source day must be rejected before physical assembly")
    for name in ("paper_1961_driver_cold_start_first_step_coverage", "paper_1961_driver_timestep_scaffold",
                 "paper_1961_next_step_hydrol_precall_scaffold", "_hydrol_state_namespace_from_previous_state",
                 "hydrol_static_precall_template_from_slowproc", "_paper_compiled_sechiba_scan"):
        monkeypatch.setattr(orch, name, forbidden)
    if entry == "paper_1961_driver_cold_start_day_scaffold":
        monkeypatch.setattr(orch, "_paper_1961_step_bundle_from_context", forbidden)
    kwargs = dict(prepared_context=context)
    if entry == "paper_1961_driver_later_day_scaffold":
        runtime = replace(context.source_clock_setup.runtime, anchor_local_tstep=np.int64(48))
        context.source_clock_setup = replace(context.source_clock_setup, runtime=runtime)
        kwargs.update(previous_state=_packet(context, tstep=47), start_tstep=48)
    with pytest.raises(SourceClockError, match="partial source-day"):
        getattr(orch, entry)("synthetic.yml", **kwargs)
    assert calls == []


def test_actual_restart_year_adapter_rejects_partial_source_day_before_thermal_or_compile(monkeypatch, tmp_path):
    from netCDF4 import Dataset

    from jax_orchidee.driver.source_clock import source_clock_from_inputs

    context = _context(source_start=17)
    prior = _packet(context)
    prior = replace(prior, source_clock_runtime=replace(prior.source_clock_runtime,
                    binding=replace(prior.source_clock_runtime.binding, run_id="previous-native-run")))
    forcing = tmp_path / "forcing.nc"
    with Dataset(forcing, "w") as dataset:
        dataset.createDimension("time", 96)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units, time.calendar = "seconds since 2001-01-01 00:00:00", "noleap"
        time[:] = (np.arange(96) + 1) * 1800.0
    context.source_clock_setup = replace(
        source_clock_from_inputs(forcing, dt_sechiba=1800., run_id="partial-native-run",
                                 previous_runtime=prior.source_clock_runtime, restart_from_memory=True),
        forcing_year=2001,
    )
    monkeypatch.setattr(orch, "driver_year_handoff_state_gaps", lambda *args: ())
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("physical delegate entered")
        raise AssertionError("partial source day must be rejected before restart physical assembly")
    for name in ("restore_sechiba_integer_source_fields", "_paper_1961_step_bundle_from_context",
                 "restart_thermosoil_initial_fields", "paper_1961_driver_later_day_runtime_result",
                 "_paper_compiled_later_day_block_executable"):
        monkeypatch.setattr(orch, name, forbidden)
    with pytest.raises(SourceClockError, match="partial source-day"):
        orch.paper_1961_driver_restart_year_start_day_result(
            "synthetic.yml", previous_year_end_state=prior, year=2001, prepared_context=context)
    assert calls == []


@pytest.mark.parametrize("adapter", ["cold", "first-restart", "later"])
def test_actual_stomate_argument_builders_use_source_midnight_not_local_day_labels(monkeypatch, adapter):
    context = _context(date=(2000, 12, 31), source_start=48)
    before = _packet(context)
    current = _apply(context, before, 0)
    slow = current.fields_by_component[SLOW]
    land, veg = jnp.ones(3), jnp.ones((3, 2))
    for name in ("lai", "height", "frac_age", "veget", "veget_max", "frac_nobio", "cn_ind"):
        slow[name] = veg
    for name in (*orch.LATER_DAY_STOMATE_ENTRY_STATE_OVERRIDES, *orch.STOMATE_DAY_SEASON_STATE_FIELDS):
        slow.setdefault(name, land)
    Entry = namedtuple("SyntheticStomateEntry", orch.LATER_DAY_STOMATE_ENTRY_STATE_OVERRIDES)
    Season = namedtuple("SyntheticStomateSeason", orch.STOMATE_DAY_SEASON_STATE_FIELDS)
    entry = Entry(*(slow[name] for name in Entry._fields))
    season = Season(*(slow[name] for name in Season._fields))
    # Stale initialization counters must be overlaid before the actual source
    # argument producer receives them, rather than after its physics returns.
    stale = {**slow, "growth_day": jnp.ones(3) * 999., "peatC_ok": jnp.ones(3) * 9.}
    Slow = namedtuple("SyntheticSlowprocInput", stale)
    restart = NS(stomate=entry, slowproc=Slow(**stale),
                 stomate_readstart=NS(season_state=season, daily_state=NS(_asdict=lambda: {})))
    clock = _payload(context, current, 0)
    assert (int(clock.month), int(clock.day), float(clock.second)) == (1, 1, 0.)
    recorded = []
    bundle_recorded = []
    def source_spy(**kwargs):
        recorded.append(kwargs)
        return NS(kwargs={})
    monkeypatch.setattr(orch, "paper_case_stomate_bundle_source_kwargs", source_spy)
    monkeypatch.setattr(orch, "stomate_restart_input_bundles", lambda **kwargs: bundle_recorded.append(kwargs) or NS())
    monkeypatch.setattr(orch, "_stomate_lpj_initial_cn_ind", lambda *a, **k: veg)
    common = dict(config_path="synthetic.yml", run_def_path=Path("synthetic.def"),
                  runtime=context.runtime, tstep=999, source_clock=clock, run_def_values={})
    daily = NS(ok=True, daily_fields={"daily_marker": land})
    if adapter == "first-restart":
        orch._paper_first_step_stomate_restart_input_bundles(
            **common, restart_state=restart, calendar_state=current, daily_fields=daily.daily_fields)
    elif adapter == "cold":
        stale_previous = replace(before, fields_by_component={SLOW: stale})
        coverage = NS(provenance=("synthetic cold source",), initialized_payloads={
            "stomate_cold_start_entry_state": entry, "stomate_cold_start_gas_state": NS(),
            "stomate_cold_start_season_state": season, "stomate_cold_start_daily_accumulators": NS(),
            "slowproc_cold_start_vegetation": NS(vegetation=NS(veget=veg, veget_max=veg)),
            "cold_start_first_step_previous_state": stale_previous})
        _, _, gaps = orch._paper_cold_start_stomate_input_bundles(
            **common, first_step_coverage=coverage, calendar_state=current, daily_fold=daily)
        assert gaps == ()
    else:
        _, _, gaps = orch._paper_later_day_stomate_input_bundles(
            **common, restart_state=restart, previous_state=current, daily_fold=daily, model_day_number=12345)
        assert gaps == ()
    assert len(recorded) == len(bundle_recorded) == 1
    arguments = recorded[0]
    assert bool(np.asarray(arguments["end_of_year"])) is True
    np.testing.assert_array_equal(arguments["julian_diff"], clock.julian_diff)
    assert float(arguments["julian_diff"]) != 12345.
    np.testing.assert_array_equal(arguments["slowproc_state"].growth_day, [4., 8., 1.])
    np.testing.assert_array_equal(arguments["slowproc_state"].peatC_ok, [0., 0., 1.])
    assert bool(np.asarray(bundle_recorded[0]["do_slow"])) is True


def test_real_jit_actual_later_argument_builder_keeps_bound_calendar_values_dynamic(monkeypatch):
    """Regress host NumPy/bool conversions inside the actual argument caller."""
    context = _context(date=(2000, 12, 31), source_start=48)
    state = _packet(context)
    slow = state.fields_by_component[SLOW]
    land, veg = jnp.ones(3), jnp.ones((3, 2))
    for name in ("lai", "height", "frac_age", "veget", "veget_max", "frac_nobio", "cn_ind"):
        slow[name] = veg
    for name in (*orch.LATER_DAY_STOMATE_ENTRY_STATE_OVERRIDES, *orch.STOMATE_DAY_SEASON_STATE_FIELDS):
        slow.setdefault(name, land)
    Entry = namedtuple("SyntheticJitStomateEntry", orch.LATER_DAY_STOMATE_ENTRY_STATE_OVERRIDES)
    Season = namedtuple("SyntheticJitStomateSeason", orch.STOMATE_DAY_SEASON_STATE_FIELDS)
    restart = NS(stomate=Entry(*(slow[name] for name in Entry._fields)),
                 stomate_readstart=NS(season_state=Season(*(slow[name] for name in Season._fields))))
    traces = []
    def source_spy(**kwargs):
        traces.append("calendar arguments only")
        return NS(kwargs={"bound_julian_diff": kwargs["julian_diff"],
                          "bound_end_of_year": kwargs["end_of_year"]})
    monkeypatch.setattr(orch, "paper_case_stomate_bundle_source_kwargs", source_spy)
    monkeypatch.setattr(orch, "stomate_restart_input_bundles", lambda **kwargs: NS())
    def actual_builder(clock):
        source, _, gaps = orch._paper_later_day_stomate_input_bundles(
            config_path="synthetic.yml", run_def_path=Path("synthetic.def"),
            restart_state=restart, previous_state=state, runtime=context.runtime,
            tstep=999, source_clock=clock, daily_fold=NS(ok=True, daily_fields={}),
            model_day_number=12345, run_def_values={})
        assert gaps == ()
        return (jnp.asarray(source.kwargs["bound_julian_diff"]),
                jnp.asarray(source.kwargs["bound_end_of_year"]))
    # Both numeric payloads are admitted by the actual host clock owner. The
    # same binding spans Jan1 and Jan2 midnight without a static date argument.
    year_midnight = _payload(context, state, 0)
    ordinary_midnight = _payload(context, state, 48)
    assert (int(year_midnight.day), int(ordinary_midnight.day)) == (1, 2)
    assert float(year_midnight.second) == float(ordinary_midnight.second) == 0.
    compiled = jax.jit(actual_builder)
    year_values = compiled(year_midnight)
    ordinary_values = compiled(ordinary_midnight)
    np.testing.assert_array_equal(year_values[0], year_midnight.julian_diff)
    np.testing.assert_array_equal(ordinary_values[0], ordinary_midnight.julian_diff)
    assert bool(year_values[1]) is True
    assert bool(ordinary_values[1]) is False
    assert len(traces) == 1
    assert compiled._cache_size() == 1


@pytest.mark.parametrize("date,growth,peat", [
    ((2001, 1, 1), [0., 0., 0.], [0., 0., 1.]),
    ((2001, 1, 2), [3., 8., 1.], [1., 0., 1.]),
], ids=["jan1-first-active", "jan2-first-active"])
def test_complete_saved_b5_adapter_reproduces_stale_calendar_then_current_builder_repairs(
    monkeypatch, date, growth, peat,
):
    context = _context(date=date, DYNPEAT_PC="TRUE", JUDGE_PC="TRUE", PC_LIM="25")
    state = _packet(context)
    calls = _replace_extraction_delegates(monkeypatch)
    fixture = Path(__file__).parent / "fixtures" / "calendar_carry_b5_adapter.py"
    saved_source = fixture.read_bytes()
    source_lf = saved_source.replace(b"\r\n", b"\n")
    assert hashlib.sha256(source_lf).hexdigest() == "1bc43d3e894a4cc1a16adb37bc18558cdc7470a349247d6b3b1efc8665c38213"
    # Execute the complete saved function against today's replaced extraction
    # globals. No historical module top-level or physical kernel is imported.
    namespace = dict(vars(orch))
    exec(compile(saved_source.decode("utf-8"), str(fixture), "exec"), namespace)
    baseline = namespace["_runtime_next_state_from_components"]
    kwargs = dict(tstep=0, previous_state=state, enerbil_local=None, diffuco_local=None,
                  thermosoil_module=None, condveg_module=None, hydrol_module=None)
    old_result = baseline(**kwargs)
    assert len(calls) == 6
    assert "growth_day" not in old_result.fields_by_component[SLOW]
    assert "peatC_ok" not in old_result.fields_by_component[SLOW]
    np.testing.assert_array_equal(old_result.fields_by_component[FINAL]["growth_day"], [3., 8., 1.])
    np.testing.assert_array_equal(old_result.fields_by_component[FINAL]["peatC_ok"], [0., 0., 1.])
    assert old_result.source_clock_runtime is None
    current = orch._runtime_next_state_from_components(
        **kwargs, source_clock_context=context, source_clock=_payload(context, state, 0))
    assert len(calls) == 12
    _mirrors(current, growth, peat)
    assert int(current.source_clock_runtime.last_applied_istp) == 1
    _mirrors(state, [3., 8., 1.], [0., 0., 1.])


@pytest.mark.parametrize("adapter", ["compiled-day", "compiled-block"])
def test_actual_compiled_adapters_reject_halfday_stomate_before_physics_or_lower(monkeypatch, adapter):
    context, _, tables, _, _ = _compiled_fixture(monkeypatch, source_start=1)
    context.runtime = NS(dt_stomate=43200., dt_sechiba=1800., dt_days=.5)
    state = _packet(context)
    series = _forcing_series(context, state, 48)
    block = jax.tree_util.tree_map(lambda a: a[None, ...], series)
    day_numbers = jnp.asarray([2])
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("physical assembly or JIT lower entered")
        raise AssertionError("half-day STOMATE must be rejected before physical assembly or lower")
    for name in ("_paper_1961_later_day_transition_inputs", "_paper_1961_later_day_half_hour_transition",
                 "_compiled_stomate_parameter_values", "_compiled_hydrol_table_arrays"):
        monkeypatch.setattr(orch, name, forbidden)
    monkeypatch.setattr(orch.jax, "jit", forbidden)
    with pytest.raises(SourceClockError, match="DT_STOMATE equal to the declared source day"):
        if adapter == "compiled-day":
            orch.paper_1961_driver_later_day_runtime_result(
                "synthetic.yml", previous_state=state, day_index=2, year=1961, start_tstep=0,
                prepared_context=context, use_compiled_sechiba_day=True, compiled_forcing_series=series)
        else:
            orch._paper_compiled_later_day_block_executable(
                "synthetic.yml", context=context, initial_state=state, year=1961,
                block_forcing=block, block_day_numbers=day_numbers,
                prebound_hydrol_runtime_static_tables=tables, daily_carbon_dispatch={})
    assert calls == []
