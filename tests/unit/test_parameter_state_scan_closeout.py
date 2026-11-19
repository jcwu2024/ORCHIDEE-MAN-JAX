"""Real narrow compiled scan/cache boundaries; unrelated physics is a spy.

Fortran HYDROL globals are read in hydrol_init (FROZ_FRAC_CORR:
hydrol.f90:1907-1914) and consumed by hydrol_soil (8455). A compiled closure must keep its admitted
immutable context. The single synthetic scan item below executes no model
step, daily fold, or physical component. Source-clock admission has separate
accepted tests and is replaced at this interface boundary.
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_orchidee.driver import orchestration as o
from jax_orchidee.sechiba import hydrol as h
from jax_orchidee.sechiba.thermosoil import thermosoil_frozen_profile_mean


@pytest.fixture
def scan_harness(monkeypatch):
    monkeypatch.setattr(o, "_COMPILED_SECHIBA_SCAN_CACHE", {})
    monkeypatch.setattr(o, "_source_clock_setup", lambda context: NS(binding=NS(binding_key=("bounded-scan",))))
    monkeypatch.setattr(o, "_validate_compiled_source_clock_series", lambda *a, **kw: None)

    def physical_spy(*args, **kw):
        # Read the same caller-owned static/dynamic projections that the real
        # compact entry consumes. Keep this a boundary witness, not physics.
        value = jnp.stack((
            jnp.asarray(kw["prepared_context"].hydrol_froz_frac_corr),
            jnp.asarray(kw["compiled_diffuco_static_kwargs"]["laimax"]),
            kw["hydrol_runtime_static_tables"].soil_parameters.ks[0],
            kw["compiled_diffuco_static_kwargs"]["trans_co2_pft_params"]["a1"][0],
        ))
        result = kw["previous_state"] + value
        return NS(next_state=result, entry_payload={name: result for name in o._COMPILED_STOMATE_ENTRY_FIELDS})

    monkeypatch.setattr(o, "_paper_1961_next_step_runtime_result_compact", physical_spy)
    payload = o.IntersurfFirstStepPayload(**{
        field.name: (() if field.name == "missing_fields" else
                     1 if field.name in {"kjpindex", "nbindex"} else np.zeros(1))
        for field in fields(o.IntersurfFirstStepPayload)
    })
    scalars = NS(nstm=3, pref_soil_veg=np.array([1]), is_tree=np.array([False]))
    spec = NS(components=(), field_names_by_component=())
    clock = o.DriverSourceClockPayload(*(
        jnp.asarray([value]) for value in (1, 1, 1961, 1, 1, 1800., True, 0.)
    ))
    forcing = o.DriverCompiledHalfHourForcing(
        **{name: jnp.ones((1, 1)) for name in o.DriverCompiledHalfHourForcing._fields
           if name not in {"source_clock", "watchout"}},
        source_clock=clock,
    )

    def context(correction=2., laimax=.25):
        return NS(
            config_path=Path("portable-scan.json"), run_def_path=Path(f"immutable-{correction}.def"),
            nvm=1, dt_sechiba=1800., ok_explicitsnow=True, ok_freeze_cwrr=False,
            ok_thermodynamical_freezing=False, hydrol_soil_peat_hydro=False,
            peat_hydro=False, hydrol_ok_pc=False, hydrol_ok_leak=False,
            hydrol_tides=False, ok_laidev=(False,), run_def_values={},
            hydrol_froz_frac_corr=correction,
            diffuco_pft14_static_kwargs={"trans_co2_pft_params": {"a1": np.array([.5])}, "laimax": laimax},
        )

    def tables(ks=100.):
        arrays = {name: jnp.ones(1) for name in o.DriverCompiledHydrolTableArrays._fields
                  if name not in {"soil_parameters", "depth_parameters"}}
        return h.HydrolRuntimeStaticTables(
            mineral=h.MineralCWRRTables(**arrays, imin=1, imax=2), peat=None,
            soil_parameters=h.hydrol_soil_parameters({"ks": np.full(12, ks)}),
        )

    def build(ctx, runtime):
        return o._paper_compiled_sechiba_scan(
            config_path=ctx.config_path, context=ctx, base_payload_template=payload,
            run_scalars=scalars, runtime_spec=spec, hydrol_static_template={},
            hydrol_runtime_static_tables=runtime,
        )

    def evaluate(executable, runtime, a1=7.):
        result, outputs, flux = executable(
            jnp.zeros(4), forcing, {}, (), o._compiled_hydrol_table_arrays(runtime), None,
            o.DriverCompiledDiffucoParameterValues({"a1": jnp.array([a1])}, jnp.array([1.])),
        )
        assert flux is None
        np.testing.assert_array_equal(outputs[0][0], result)
        return np.asarray(result)

    return NS(context=context, tables=tables, build=build, evaluate=evaluate)


def test_distinct_contexts_keep_static_values_in_real_narrow_jit(scan_harness):
    assert not jax.config.jax_disable_jit
    a, b = scan_harness.context(), scan_harness.context(3., .75)
    runtime = scan_harness.tables()
    executable_a = scan_harness.build(a, runtime)
    executable_b = scan_harness.build(b, runtime)
    np.testing.assert_array_equal(scan_harness.evaluate(executable_a, runtime), [2., .25, 100., 7.])
    np.testing.assert_array_equal(scan_harness.evaluate(executable_b, runtime), [3., .75, 100., 7.])
    assert executable_a is scan_harness.build(a, runtime)
    assert executable_b is scan_harness.build(b, runtime)
    assert executable_a is not executable_b


def test_same_context_accepts_same_shape_dynamic_values_without_stale_capture(scan_harness):
    context = scan_harness.context()
    a, b = scan_harness.tables(), scan_harness.tables(150.)
    executable = scan_harness.build(context, a)
    assert executable is scan_harness.build(context, b)
    np.testing.assert_array_equal(scan_harness.evaluate(executable, a), [2., .25, 100., 7.])
    np.testing.assert_array_equal(scan_harness.evaluate(executable, b, a1=9.), [2., .25, 150., 9.])


def test_actual_scan_rejects_thermal_source_error(monkeypatch, scan_harness):
    """Execute the real scan/error owner; all other physics is a boundary spy."""
    def guarded_component(*args, **kwargs):
        # A1 is a dynamic boundary input. Only one source guard executes;
        # no model step or daily fold is evaluated by this test.
        value = kwargs["compiled_diffuco_static_kwargs"]["trans_co2_pft_params"]["a1"][0]
        thermosoil_frozen_profile_mean(jnp.stack((jnp.asarray(0.), value))[None, None, :])
        state = kwargs["previous_state"]
        return NS(next_state=state, entry_payload={name: state for name in o._COMPILED_STOMATE_ENTRY_FIELDS})

    monkeypatch.setattr(o, "_paper_1961_next_step_runtime_result_compact", guarded_component)
    runtime = scan_harness.tables()
    executable = scan_harness.build(scan_harness.context(), runtime)
    np.testing.assert_array_equal(scan_harness.evaluate(executable, runtime, a1=0.), np.zeros(4))
    with pytest.raises(ValueError, match="profil_froz differs across PFTs"):
        scan_harness.evaluate(executable, runtime, a1=.5)


def test_actual_block_cache_specializes_static_mask_dispatch(monkeypatch, tmp_path, scan_harness):
    """One real compiled scan item; the daily physical delegate is a witness."""
    monkeypatch.setattr(o, "_COMPILED_LATER_DAY_BLOCK_CACHE", {})
    monkeypatch.setattr(o, "_paper_1961_later_day_transition_inputs",
                        lambda **kwargs: NS(compiled_base_payload_template={}))
    monkeypatch.setattr(o, "_compiled_landpoint_payload", lambda payload: {})
    monkeypatch.setattr(o, "_compiled_diffuco_parameter_values", lambda context: ())
    monkeypatch.setattr(o, "StomateRestartSeasonState", lambda **kwargs: NS(**kwargs))

    def physical_spy(*args, **kwargs):
        previous = kwargs["previous_state"]
        dispatch = kwargs["outer_compiled_daily_carbon_dispatch"]
        source_value = kwargs["compiled_forcing_series"] - 1.
        thermosoil_frozen_profile_mean(jnp.stack((jnp.asarray(0.), source_value))[None, None, :])
        value = previous.values_by_component[0][0] + float(dispatch["phenology_active_pft_mask"][0])
        return NS(day_end_state=NS(tstep=previous.tstep + 1,
                                  fields_by_component={"witness": {"x": value}}, source_clock_runtime=None),
                  daily_modelout=NS(modelout_fields={"x": value}, modelout=value))

    monkeypatch.setattr(o, "paper_1961_driver_later_day_runtime_result", physical_spy)
    context = scan_harness.context()
    context.config_path = tmp_path / "immutable.json"
    context.runtime = NS(dt_stomate=1800., dt_sechiba=1800.)
    season = NS(provenance=(), _asdict=lambda: {"provenance": ()})
    context.first_step_restart_state = NS(stomate=(), stomate_readstart=NS(season_state=season))
    packet = o.DriverPreviousStepStatePacket(0, {"witness": {"x": jnp.zeros(1)}}, {})
    runtime = scan_harness.tables()
    forcing, days = jnp.ones(1), jnp.asarray([2], dtype=jnp.int32)

    def build(enabled):
        return o._paper_compiled_later_day_block_executable(
            context.config_path, context=context, initial_state=packet, year=1961,
            block_forcing=forcing, block_day_numbers=days,
            prebound_hydrol_runtime_static_tables=runtime,
            daily_carbon_dispatch={"phenology_active_pft_mask": (enabled,)},
            stomate_parameter_values=(),
        )[0]

    def evaluate(executable):
        state, _ = executable(o.fast_state_from_previous_packet(packet), forcing, days, (),
                              o._compiled_hydrol_table_arrays(runtime), {}, (), {}, ())
        return np.asarray(state.values_by_component[0][0])

    enabled, disabled = build(True), build(False)
    np.testing.assert_array_equal(evaluate(enabled), [1.])
    np.testing.assert_array_equal(evaluate(disabled), [0.])
    assert enabled is build(True) and disabled is build(False)
    assert enabled is not disabled
    with pytest.raises(ValueError, match="profil_froz differs across PFTs"):
        enabled(o.fast_state_from_previous_packet(packet), jnp.zeros_like(forcing), days, (),
                o._compiled_hydrol_table_arrays(runtime), {}, (), {}, ())
