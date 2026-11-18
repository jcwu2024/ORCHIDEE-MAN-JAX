"""Portable paper public-entry and real narrow-JIT WATCHOUT transport checks.

Files, substeps, bundles, schema, adapter and drag are real. Supplied previous
science state, downstream beta/payload packaging, energy assembly, and compact
physical transition are explicit delegates; no model step or day executes.
"""

from __future__ import annotations

from dataclasses import make_dataclass, replace
from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_orchidee.driver import orchestration as o
from jax_orchidee.driver.domain import read_forcing_first_step
from jax_orchidee.driver.sechiba_boundary import build_intersurf_first_step_payload
from jax_orchidee.sechiba import hydrol as h
from tests.unit.test_dim2_driver import _bundle
from tests.unit.test_watchout_domain_transport import FIXTURE, _case
from tests.unit.test_watchout_drag_transport import _precall, _static_kwargs
from tests.unit.test_watchout_drag_transport import beta_delegate as beta_delegate

WATCH = ("eair", "swnet", "cdrag", "petAcoef", "peqAcoef", "petBcoef", "peqBcoef")
EVIDENCE = {"public": [], "jit": {}, "year": []}


def _prepared_case(tmp_path, monkeypatch, *, daily=False, watchout=True, coupled=False):
    config, domain = _case(tmp_path, daily=daily, watchout=watchout)
    spec = FIXTURE["daily" if daily else "ordinary"]
    first = _bundle(0, npts=1)
    first.year = 1961
    first.water_table = None
    first.domain = domain
    first.forcing = read_forcing_first_step(config, domain=domain)
    first.static_trace_fields.salinity = np.zeros(1)
    first.static_trace_fields.tide_height = np.zeros(1)
    first.run_scalars = NS(nvm=14, nstm=3, fortran_pft_ids=tuple(range(1, 15)),
                           pref_soil_veg=np.ones(14, dtype=int), is_tree=np.zeros(14, dtype=bool),
                           pft_layout=NS(index_for_id=lambda pft_id: 13))
    static = {**_static_kwargs(), "dt_sechiba": spec["dt_sechiba"], "ldq_cdrag_from_gcm": coupled}
    context = NS(
        config_path=config, run_def_path=tmp_path / "unused.def", run_def_values={},
        first_step_bundle=first, first_step_restart_state=None, reference_run_dir=None, domain_override={},
        source_clock_setup=None,
        forcing_split=spec["split"], forcing_nb_spread=spec["nb_spread"],
        dt_sechiba=spec["dt_sechiba"], runtime=NS(dt_sechiba=spec["dt_sechiba"], dt_stomate=86400.),
        run_scalars=first.run_scalars, nvm=14, diffuco_pft14_static_kwargs=static,
        ok_explicitsnow=False, river_routing=False, ok_freeze_cwrr=False, ok_thermodynamical_freezing=False,
        hydrol_soil_peat_hydro=False, peat_hydro=False, hydrol_ok_pc=False, hydrol_ok_leak=False,
        hydrol_tides=False, ok_laidev=tuple(np.ones(14, dtype=bool)), watchout_initialize_main_tstep=None,
    )
    monkeypatch.setattr(o, "read_annual_co2", lambda *args: FIXTURE["driver"]["annual_atmco2"])
    return context


def _previous(tstep):
    return o.DriverPreviousStepStatePacket(tstep, {
        "slowproc_stomate_previous_step_state": {"biomass": np.zeros((1, 14, 8))},
        "driver_previous_step_state": {"z0": np.asarray([0.1]), "albedo": np.asarray([[0.3, 0.5]])},
    }, {})


def _day_cache():
    source = _precall().payload
    return NS(
        control_salinity=NS(payload={"control_salinity": np.ones(1)}),
        control_inundation=NS(payload={"control_inudate": np.ones(1)}),
        slowproc_derivvar=None,
        slowproc_derivvar_payload={name: source[name] for name in ("qsintmax", "assim_param", "height", "temp_growth")},
    )


@pytest.mark.parametrize("daily", [False, True], ids=["ordinary", "daily"])
@pytest.mark.parametrize("coupled", [False, True, None], ids=["local-drag", "coupled-drag", "initial-substep-default"])
def test_actual_paper_public_entry_keeps_file_tuple_and_static_drag(
    tmp_path, monkeypatch, beta_delegate, daily, coupled
):
    context = _prepared_case(tmp_path, monkeypatch, daily=daily, coupled=coupled)
    expected_coupled = True if coupled is None else coupled
    if coupled is None:
        context.diffuco_pft14_static_kwargs.pop("ldq_cdrag_from_gcm")
    source = _precall().payload
    monkeypatch.setattr(o, "_diffuco_restart_payload_from_previous_state", lambda previous: (source, (), ()))
    monkeypatch.setattr(o, "_enerbil_state_payload_from_previous_state", lambda previous: ({}, (), ()))
    observed = []

    def energy_assembly(**kwargs):
        observed.append(kwargs)
        return NS(ok=False, payload=kwargs["driver_or_intersurf_payload"])

    monkeypatch.setattr(o, "assemble_enerbil_first_step_precall_payload", energy_assembly)
    cases = FIXTURE["daily" if daily else "ordinary"]["cases"]
    for case in cases:
        tstep = case["model_tstep"]
        result = o.paper_1961_next_step_enerbil_precall_scaffold(
            context.config_path, previous_state=_previous(tstep - 1), tstep=tstep,
            prepared_context=context, diffuco_day_static_cache=_day_cache(),
            module_jit=False, diffuco_local_jit=False,
        )
        expected = case["expected"]
        for name in WATCH:
            np.testing.assert_allclose(getattr(result.payload, name), [expected[name]], rtol=2e-14, atol=1e-13)
        np.testing.assert_array_equal(result.payload.ccanopy, [FIXTURE["driver"]["expected_ccanopy"]])
        energy = observed[-1]["driver_or_intersurf_payload"]
        np.testing.assert_allclose(energy["epot_air"], [expected["eair"]], rtol=2e-14)
        np.testing.assert_allclose(energy["swnet"], [expected["swnet"]], rtol=2e-14)
        for name in WATCH[3:]:
            np.testing.assert_allclose(energy[name], [expected[name]], rtol=2e-14)
        assert observed[-1]["non_watchout_driver_executed"] is False
        local = result.diffuco_local_enerbil_precall
        np.testing.assert_allclose(result.diffuco_precall.payload["swnet"], [expected["swnet"]], rtol=2e-14)
        np.testing.assert_allclose(local.result.payload.payload["swnet"], [expected["swnet"]], rtol=2e-14)
        assert local.kwargs["ldq_cdrag_from_gcm"] is expected_coupled
        if expected_coupled:
            np.testing.assert_allclose(local.kwargs["q_cdrag"], [expected["cdrag"]], rtol=2e-14)
            np.testing.assert_allclose(local.result.boundary.drag.q_cdrag_pft, np.full((1, 14), expected["cdrag"]), rtol=2e-14)
        else:
            assert "q_cdrag" not in local.kwargs
        EVIDENCE["public"].append({
            "family": "daily" if daily else "ordinary", "mode": coupled, "tstep_label": tstep,
            "saved_reference": case["saved_reference"],
            "actual_tuple": {name: np.asarray(getattr(result.payload, name)).tolist() for name in WATCH},
            "actual_ccanopy": np.asarray(result.payload.ccanopy).tolist(),
            "actual_static_drag": local.kwargs["ldq_cdrag_from_gcm"],
            "actual_drag": np.asarray(local.result.boundary.drag.q_cdrag).tolist(),
            "actual_raero": np.asarray(local.result.boundary.drag.raero).tolist(),
            "actual_diffuco_swnet": np.asarray(result.diffuco_precall.payload["swnet"]).tolist(),
            "actual_energy_swnet": np.asarray(energy["swnet"]).tolist(),
        })
    assert len(beta_delegate) == len(cases)


def test_actual_paper_first_main_exception_uses_restored_albedo(tmp_path, monkeypatch, beta_delegate):
    context = _prepared_case(tmp_path, monkeypatch, coupled=True)
    context.watchout_initialize_main_tstep = 0
    source = _precall().payload
    monkeypatch.setattr(o, "_diffuco_restart_payload_from_previous_state", lambda previous: (source, (), ()))
    monkeypatch.setattr(o, "_enerbil_state_payload_from_previous_state", lambda previous: ({}, (), ()))
    observed = []

    def energy_assembly(**kwargs):
        observed.append(kwargs)
        return NS(ok=False, payload=kwargs["driver_or_intersurf_payload"])

    monkeypatch.setattr(o, "assemble_enerbil_first_step_precall_payload", energy_assembly)
    result = o.paper_1961_next_step_enerbil_precall_scaffold(
        context.config_path, previous_state=_previous(-1), tstep=0, prepared_context=context,
        diffuco_day_static_cache=_day_cache(), module_jit=False, diffuco_local_jit=False,
    )
    # The raw preread is separate from the actual interpolated initialize
    # input. Main source1163 then applies even in WATCHOUT.
    initial_payload = build_intersurf_first_step_payload(o._paper_1961_step_bundle_from_context(context, year=1961, tstep=0))
    np.testing.assert_array_equal(initial_payload.swnet, [FIXTURE["ordinary"]["cases"][0]["expected"]["swnet"]])
    expected = (1.0 - (0.3 + 0.5) / 2.0) * result.payload.swdown
    np.testing.assert_allclose(observed[-1]["driver_or_intersurf_payload"]["swnet"], expected, rtol=2e-14)
    np.testing.assert_allclose(result.diffuco_precall.payload["swnet"], expected, rtol=2e-14)
    assert not np.array_equal(expected, result.payload.swnet)
    EVIDENCE["first_main"] = {"initialize_file_swnet": np.asarray(initial_payload.swnet).tolist(),
                             "ordinary_file_swnet": np.asarray(result.payload.swnet).tolist(),
                             "actual_first_main_swnet": np.asarray(observed[-1]["driver_or_intersurf_payload"]["swnet"]).tolist(),
                             "actual_diffuco_swnet": np.asarray(result.diffuco_precall.payload["swnet"]).tolist()}


def _clock_payload(context, runtime, *, tstep):
    return o.DriverSourceClockPayload(
        local_tstep=np.asarray(tstep), istp=np.asarray(tstep + 1), year=np.asarray(1961),
        month=np.asarray(1), day=np.asarray(1), second=np.asarray(context.dt_sechiba * (tstep + 1)),
        active=np.asarray(False), julian_diff=np.asarray(0.),
    )


def _scan_context(monkeypatch):
    monkeypatch.setattr(o, "_COMPILED_SECHIBA_SCAN_CACHE", {})
    monkeypatch.setattr(o, "_source_clock_setup", lambda context: NS(binding=NS(binding_key=("watchout-bounded",))))
    monkeypatch.setattr(o, "_validate_compiled_source_clock_series", lambda *args: None)
    monkeypatch.setattr(o, "_driver_source_clock_payload", _clock_payload)
    monkeypatch.setattr(o, "_hydrol_runtime_static_tables_from_compiled_arrays", lambda *args, **kwargs: None)
    traces = []

    def compact_delegate(*args, **kwargs):
        payload = kwargs["prebuilt_base_payload"]
        traces.append(payload.is_watchout)
        leaves = ([jnp.ravel(getattr(payload, name))[0] for name in WATCH] if payload.is_watchout else [0.] * len(WATCH))
        energy = o._enerbil_driver_coverage_payload(
            payload, initializing=o._paper_watchout_initializing(kwargs["prepared_context"], kwargs["tstep"]),
            driver_albedo_payload={"albedo": jnp.asarray([[0.3, 0.5]])},
        )
        value = jnp.stack((*leaves, jnp.asarray(payload.is_watchout, dtype=float),
                           jnp.asarray(o._paper_context_drag_mode(kwargs["prepared_context"], payload), dtype=float),
                           jnp.ravel(energy.get("swnet", jnp.zeros(1)))[0]))
        return NS(next_state=value, entry_payload={name: value for name in o._COMPILED_STOMATE_ENTRY_FIELDS})

    monkeypatch.setattr(o, "_paper_1961_next_step_runtime_result_compact", compact_delegate)
    return traces


def _scan_build(context, payload):
    return o._paper_compiled_sechiba_scan(
        config_path=context.config_path, context=context, base_payload_template=payload,
        run_scalars=context.run_scalars, runtime_spec=NS(components=(), field_names_by_component=()),
        hydrol_static_template={}, hydrol_runtime_static_tables=NS(mineral=NS(imin=1, imax=2)),
    )


def _scan_evaluate(executable, forcing):
    state, output, flux = executable(
        jnp.zeros(10), forcing, {}, (), (), None,
        o.DriverCompiledDiffucoParameterValues({}, jnp.ones(14)),
    )
    assert flux is None
    np.testing.assert_array_equal(output[0][0], state)
    return np.asarray(state)


def test_actual_one_item_jit_scan_keeps_tuple_dynamic_and_separates_none_cache(tmp_path, monkeypatch):
    assert not jax.config.jax_disable_jit
    traces = _scan_context(monkeypatch)
    watch_dir, offline_dir = tmp_path / "watch", tmp_path / "offline"
    watch_dir.mkdir()
    offline_dir.mkdir()
    context = _prepared_case(watch_dir, monkeypatch, coupled=False)
    offline = _prepared_case(offline_dir, monkeypatch, watchout=False, coupled=False)
    first = o._paper_1961_step_bundle_from_context(context, year=1961, tstep=0)
    second = o._paper_1961_step_bundle_from_context(context, year=1961, tstep=3)
    a = build_intersurf_first_step_payload(first)
    b = build_intersurf_first_step_payload(second)
    plain = build_intersurf_first_step_payload(offline.first_step_bundle)
    assert not (set(WATCH) & set(o._compiled_landpoint_payload(a)))
    one = o._paper_compiled_forcing_day(context, year=1961, start_tstep=0, steps_per_stomate=1)
    two = o._paper_compiled_forcing_day(context, year=1961, start_tstep=3, steps_per_stomate=1)
    nonwatch = o._paper_compiled_forcing_day(offline, year=1961, start_tstep=0, steps_per_stomate=1)
    assert one.watchout is not None and two.watchout is not None and nonwatch.watchout is None
    executable = _scan_build(context, a)
    assert executable is _scan_build(context, b)
    dynamic_outputs = []
    for forcing, case in ((one, FIXTURE["ordinary"]["cases"][0]), (two, FIXTURE["ordinary"]["cases"][-1])):
        expected = [case["expected"][name] for name in WATCH] + [1., 0., case["expected"]["swnet"]]
        actual = _scan_evaluate(executable, forcing)
        dynamic_outputs.append(actual.tolist())
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-13)
    assert traces == [True]  # The second same-shape record was dynamic, not captured/retraced.
    offline_executable = _scan_build(context, plain)
    assert offline_executable is not executable
    np.testing.assert_array_equal(_scan_evaluate(offline_executable, nonwatch), np.zeros(10))
    assert traces == [True, False]
    # Different admitted immutable static mode produces a distinct closure.
    changed = NS(**{**vars(context), "watchout_initialize_main_tstep": 0,
                    "diffuco_pft14_static_kwargs": {**context.diffuco_pft14_static_kwargs, "ldq_cdrag_from_gcm": True}})
    changed_executable = _scan_build(changed, a)
    assert changed_executable is not executable
    changed_one = _scan_evaluate(changed_executable, one)
    assert changed_one[-2] == 1.
    np.testing.assert_allclose(changed_one[-1], 0.6 * np.asarray(one.swdown).reshape(-1)[0], rtol=2e-14)
    changed_two = _scan_evaluate(changed_executable, two)
    assert changed_two[-2] == 1.
    np.testing.assert_allclose(changed_two[-1], FIXTURE["ordinary"]["cases"][-1]["expected"]["swnet"], rtol=2e-14)
    assert traces == [True, False, True]
    EVIDENCE["jit"] = {"dynamic_same_shape_outputs": dynamic_outputs, "none_mode_output": [0.] * 10,
                       "changed_static_phase_first_output": changed_one.tolist(),
                       "changed_static_phase_later_output": changed_two.tolist(), "actual_trace_mode_sequence": traces,
                       "same_context_same_shape_executable_reused": True,
                       "watchout_none_mode_executables_separate": True, "different_immutable_context_executable_separate": True}


@pytest.mark.parametrize("mode", ["new_native_run", "continuous"])
def test_actual_restart_year_public_entry_binds_only_new_native_watchout_phase(tmp_path, monkeypatch, mode):
    context_ns = _prepared_case(tmp_path, monkeypatch, coupled=False)
    # The year-start entry uses dataclass replace for immutable context admission.
    context_type = make_dataclass("PortableWatchoutYearContext", [(name, object) for name in vars(context_ns)])
    context = context_type(**vars(context_ns))
    packet = _previous(47)
    packet.fields_by_component.update({
        "sechiba_finalize_state": {"us": np.zeros((1, 14, 6, 11)), "vegstress": np.ones((1, 14)), "njsc": np.asarray([2.])},
        "hydrol_previous_step_state": {}, "thermosoil_previous_step_state": {},
    })
    monkeypatch.setattr(o, "driver_year_handoff_state_gaps", lambda packet: ())
    monkeypatch.setattr(o, "_source_clock_setup", lambda context: None)
    monkeypatch.setattr(o, "source_clock_setup_from_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(o, "rebase_driver_state_for_year_start", lambda previous, **kwargs: replace(previous, tstep=-1))
    monkeypatch.setattr(o, "_require_source_day_alignment", lambda *args, **kwargs: None)
    monkeypatch.setattr(h, "hydrol_init_us_state", lambda **kwargs: NS(us=kwargs["us"], humrelv=np.ones((1, 14))))
    monkeypatch.setattr(h, "hydrol_init_vegstress", lambda **kwargs: kwargs["vegstress"])
    monkeypatch.setattr(o, "resolve_hydrol_initialization_refsoc_1d", lambda *args, **kwargs: None)
    monkeypatch.setattr(o, "restart_thermosoil_initialization_supported", lambda context: False)
    observed = []

    def later_day_delegate(*args, **kwargs):
        observed.append(kwargs)
        return o.DriverRuntimeDayResult(
            year=1961, day_index=1, start_tstep=0, steps_per_stomate=48,
            daily_modelout=NS(), day_end_state=kwargs["previous_state"],
            stopped_at_tstep=None, missing_components=(),
        )

    monkeypatch.setattr(o, "paper_1961_driver_later_day_runtime_result", later_day_delegate)
    result = o.paper_1961_driver_restart_year_start_day_result(
        context.config_path, previous_year_end_state=packet, year=1961, prepared_context=context, run_mode=mode,
    )
    admitted = observed[-1]["prepared_context"]
    assert result.day_end_state.tstep == -1
    assert admitted.watchout_initialize_main_tstep == (0 if mode == "new_native_run" else None)
    assert o._paper_context_drag_mode(admitted) is (mode == "new_native_run")
    assert context.watchout_initialize_main_tstep is None
    assert context.diffuco_pft14_static_kwargs["ldq_cdrag_from_gcm"] is False
    EVIDENCE["year"].append({"mode": mode, "actual_phase": admitted.watchout_initialize_main_tstep,
                             "actual_drag_mode": o._paper_context_drag_mode(admitted), "caller_context_unchanged": True})


@pytest.mark.parametrize("entry", ["public-later", "compiled-day"])
def test_actual_paper_entries_keep_source_relaxation_fatal(tmp_path, monkeypatch, entry):
    context = _prepared_case(tmp_path, monkeypatch)
    context.run_def_values["RELAXATION"] = "TRUE"
    with pytest.raises(NotImplementedError, match="RELAXATION"):
        if entry == "compiled-day":
            o._paper_compiled_forcing_day(context, year=1961, start_tstep=0, steps_per_stomate=1)
        else:
            o.paper_1961_next_step_enerbil_precall_scaffold(
                context.config_path, previous_state=_previous(0), tstep=1, prepared_context=context,
                diffuco_day_static_cache=_day_cache(), module_jit=False, diffuco_local_jit=False,
            )


class _FirstCompiledConsumerReached(Exception):
    """Stop at the actual first compact consumer before any physics executes."""


@pytest.mark.parametrize("initial_watchout,current_watchout", [(True, True), (True, False), (False, True)],
                         ids=["new-record", "clear-watchout", "enable-watchout"])
def test_actual_compiled_first_consumer_uses_current_optional_watchout(
    tmp_path, monkeypatch, initial_watchout, current_watchout
):
    initial_dir, current_dir = tmp_path / "initial", tmp_path / "current"
    initial_dir.mkdir()
    current_dir.mkdir()
    context = _prepared_case(initial_dir, monkeypatch, watchout=initial_watchout)
    current_context = _prepared_case(current_dir, monkeypatch, watchout=current_watchout)
    context.cwrr_grid = NS(znh=np.asarray([0.01, 0.1]))
    context.ext_coeff_vegetfrac = np.ones(14)
    context.hydrol_throughfall_by_pft = np.ones(14)
    context.hydrol_humcste = np.ones(14)
    context.stomate_ok_dgvm = False
    monkeypatch.setattr(o, "_driver_source_clock_payload", _clock_payload)
    forcing = o._paper_compiled_forcing_day(current_context, year=1961, start_tstep=3, steps_per_stomate=1)
    previous = _previous(2)
    previous.fields_by_component["slowproc_stomate_previous_step_state"]["lai"] = np.zeros((1, 14))
    # Only per-day science preparation is delegated. Both production entry
    # functions, their template selection, and first payload reconstruction run.
    monkeypatch.setattr(o, "_hydrol_state_namespace_from_previous_state", lambda state: (NS(njsc=np.asarray([2])), ()))
    monkeypatch.setattr(o, "_prepared_hydrol_soil_parameters", lambda context: NS())
    monkeypatch.setattr(o, "hydrol_static_precall_template_from_slowproc", lambda **kwargs: {})
    monkeypatch.setattr(o, "_paper_diffuco_day_static_cache_from_firstcall_inputs", lambda *args, **kwargs: _day_cache())
    monkeypatch.setattr(o, "_paper_diffuco_day_static_cache_from_previous_state", lambda *args, **kwargs: _day_cache())
    day_inputs = o._paper_1961_later_day_transition_inputs(
        context=context, current_state=previous, year=1961, start=3,
        # The complete-day dispatch flag is selected, then execution stops at
        # its first consumer. One forcing item is sufficient at this boundary;
        # no remaining scan item or daily fold is entered or certified.
        steps_per_stomate=48, fixed_format_trace_dir=None, static_trace_fields=None,
        prebuild_day_payloads=True, prebound_hydrol_runtime_static_tables=NS(),
        compiled_forcing_series=forcing,
    )
    template = day_inputs.compiled_base_payload_template
    assert template.is_watchout is initial_watchout
    observed = []

    def first_consumer(*args, **kwargs):
        observed.append(kwargs)
        raise _FirstCompiledConsumerReached

    monkeypatch.setattr(o, "_paper_1961_next_step_runtime_result_compact", first_consumer)
    with pytest.raises(_FirstCompiledConsumerReached):
        o._paper_1961_later_day_half_hour_transition(
            context.config_path, context=context, initial_state=previous, day_inputs=day_inputs,
            fixed_format_trace_dir=None, static_trace_fields=None,
            module_jit=True, diffuco_local_jit=True, use_compiled_sechiba_day=True,
        )
    assert len(observed) == 1
    payload = observed[0]["prebuilt_base_payload"]
    expected = FIXTURE["ordinary"]["cases"][-1]["expected"]
    EVIDENCE.setdefault("first_compiled_consumer", []).append({
        "initial_watchout": initial_watchout, "current_watchout": current_watchout,
        "template_tuple": {name: None if getattr(template, name) is None else np.asarray(getattr(template, name)).tolist() for name in WATCH},
        "actual_is_watchout": payload.is_watchout,
        "actual_tuple": {name: None if getattr(payload, name) is None else np.asarray(getattr(payload, name)).tolist() for name in WATCH},
        "expected_tuple": {name: [expected[name]] if current_watchout else None for name in WATCH},
        "actual_ccanopy": np.asarray(payload.ccanopy).tolist(),
        "source_clock_local_tstep": int(observed[0]["source_clock"].local_tstep),
        "actual_consumer_calls": len(observed), "physics_or_remaining_scan_calls": 0,
    })
    np.testing.assert_array_equal(payload.ccanopy, [FIXTURE["driver"]["expected_ccanopy"]])
    assert payload.is_watchout is current_watchout
    for name in WATCH:
        if current_watchout:
            np.testing.assert_allclose(getattr(payload, name), [expected[name]], rtol=2e-14, atol=1e-13, err_msg=name)
        else:
            assert getattr(payload, name) is None
