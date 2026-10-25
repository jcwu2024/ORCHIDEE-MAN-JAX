"""S22-CONDVEG-Z0: self-contained source-reduced parameter-binding checks.

Fortran: constantes.f90:483-489 reads Z0_BARE; condveg.f90 dynamic
roughness:1581-1600,1666-1673,1702. Bare cover one, zero snow/nobio
reduces z0m exactly to the supplied bare roughness. No saved outputs/oracle
files or exploratory fixtures are inputs to these regressions.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jax_orchidee.sechiba.condveg import (
    condveg_initialize,
    condveg_main_minimal,
    run_condveg_first_step_module,
)


def _bare_inputs():
    veget = np.zeros((1, 14), dtype=np.float64)
    veget[0, 0] = 1.0
    restart = {
        "snow": np.zeros(1), "snow_nobio": np.zeros((1, 1)),
        "snowrho": np.full((1, 3), 100.0), "snowdz": np.zeros((1, 3)),
        "veget": veget, "veget_max": veget.copy(),
        "frac_nobio": np.zeros((1, 1)), "height": np.ones((1, 14)),
        "lai": np.zeros((1, 14)), "snow_age": np.zeros(1),
        "snow_nobio_age": np.zeros(1), "soilalbedo_bg": np.array([[0.1, 0.2]]),
    }
    driver = {"zlev": np.array([10.0]), "temp_air": np.array([285.0]),
              "pb": np.array([1013.0]), "u": np.array([2.0]), "v": np.zeros(1)}
    slowproc = {"totfrac_nobio": np.zeros(1), "tot_bare_soil": np.ones(1)}
    return restart, driver, slowproc


def _parameters(z0_bare=None):
    # Explicit valid albedo inputs ensure use_jit=True reaches the real
    # compiled CONDVEG component rather than the wrapper's eager fallback.
    values = {"ALB_BG_MODIS": "TRUE", "ALB_BARE_MODEL": "FALSE",
              "CONDVEG_SNOWA": "1e20", "TCST_SNOWA": "10",
              "ALB_ICE__00001": "0.6", "ALB_ICE__00002": "0.2"}
    for name, value in (("ALB_LEAF_VIS", 0.1), ("ALB_LEAF_NIR", 0.2),
                        ("SNOWA_AGED_VIS", 0.6), ("SNOWA_AGED_NIR", 0.2),
                        ("SNOWA_DEC_VIS", 0.1), ("SNOWA_DEC_NIR", 0.1)):
        values.update({f"{name}__{i:05d}": str(value) for i in range(1, 15)})
    if z0_bare is not None:
        values["Z0_BARE"] = str(z0_bare)
    return values


def _expected_heat_roughness(z0_bare):
    # Independent scalar reduction of source lines1590-1600,1703;
    # literal unchanged source constants, not another JAX helper as oracle.
    friction_velocity = 0.41 * 2.0 / np.log(10.0 / z0_bare)
    reynolds = z0_bare * friction_velocity / (1.327e-5 * (285.0 / 273.15) ** 1.81)
    return z0_bare / np.exp(2.46 * reynolds ** 0.25 - np.log(7.4))


@pytest.mark.parametrize("use_jit", [False, True], ids=["eager", "narrow-jit"])
@pytest.mark.parametrize("z0_bare", [None, 0.01, 0.02], ids=["source-default", "explicit-default", "nondefault"])
def test_actual_wrapper_honors_bare_roughness_source_parameter(z0_bare, use_jit):
    restart, driver, slowproc = _bare_inputs()
    closure = run_condveg_first_step_module(
        restart_payload=restart, driver_payload=driver, slowproc_payload=slowproc,
        hydrol_diagnostics=SimpleNamespace(drysoil_frac=np.array([0.5])),
        run_def_values=_parameters(z0_bare), use_jit=use_jit,
    )
    assert closure.ok and closure.module.albedo is not None
    expected = 0.01 if z0_bare is None else z0_bare
    np.testing.assert_allclose(closure.module.z0m, [expected], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(closure.module.z0h, [_expected_heat_roughness(expected)], rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(closure.module.roughheight, [0.0])


def _cache_context(z0_bare):
    return SimpleNamespace(
        nvm=14, dt_sechiba=1800.0, ok_explicitsnow=True,
        ok_freeze_cwrr=False, ok_thermodynamical_freezing=False,
        hydrol_soil_peat_hydro=False, peat_hydro=False, hydrol_ok_pc=False,
        hydrol_ok_leak=False, hydrol_tides=False, ok_laidev=np.zeros(14, dtype=bool),
        run_def_values={"Z0_BARE": str(z0_bare)},
        diffuco_pft14_static_kwargs={"trans_co2_pft_params": {}},
        config_path=Path("self-contained-stage23-config.json"),
        runtime=SimpleNamespace(dt_stomate=1800.0, dt_sechiba=1800.0),
        first_step_restart_state=SimpleNamespace(
            stomate=(), stomate_readstart=SimpleNamespace(
                season_state=SimpleNamespace(provenance=(), _asdict=lambda: {"provenance": ()}))),
    )


@pytest.mark.parametrize("entry", ["sechiba-scan", "later-day-block"])
def test_real_compiled_cache_construction_isolates_nondefault_bare_roughness(monkeypatch, entry):
    """Actual cache constructors; stub compilation, never execute a model scan.

    This proves parameter-sensitive cache selection only. The separate real
    CONDVEG eager/JIT tests above provide the physical numerical evidence.
    """
    from jax_orchidee.driver import orchestration as orch

    class ConstructionOnlyJit:
        def lower(self, *args):
            return self

        def compile(self):
            return object()

    monkeypatch.setattr(orch.jax, "jit", lambda function: ConstructionOnlyJit())
    spec = SimpleNamespace(components=(), field_names_by_component=())
    tables = SimpleNamespace(mineral=SimpleNamespace(imin=0, imax=1))
    if entry == "sechiba-scan":
        monkeypatch.setattr(orch, "_COMPILED_SECHIBA_SCAN_CACHE", {})
        scalars = SimpleNamespace(nstm=3, pref_soil_veg=np.array([1]), is_tree=np.zeros(14, dtype=bool))

        def build(context):
            return orch._paper_compiled_sechiba_scan(
                config_path=context.config_path, context=context, base_payload_template=None,
                run_scalars=scalars, runtime_spec=spec, hydrol_static_template={},
                hydrol_runtime_static_tables=tables,
            )
    else:
        monkeypatch.setattr(orch, "_COMPILED_LATER_DAY_BLOCK_CACHE", {})
        monkeypatch.setattr(orch, "fast_state_from_previous_packet", lambda packet: packet)
        monkeypatch.setattr(orch, "_compiled_hydrol_table_arrays", lambda tables: ())
        monkeypatch.setattr(orch, "_compiled_landpoint_payload", lambda payload: ())
        monkeypatch.setattr(orch, "_compiled_diffuco_parameter_values", lambda context: ())
        monkeypatch.setattr(orch, "_paper_1961_later_day_transition_inputs", lambda **kwargs:
                            SimpleNamespace(compiled_base_payload_template=None))
        initial = SimpleNamespace(spec=spec, values_by_component=())

        def build(context):
            executable, actual_spec = orch._paper_compiled_later_day_block_executable(
                context.config_path, context=context, initial_state=initial, year=1961,
                block_forcing=(), block_day_numbers=np.array([1]),
                prebound_hydrol_runtime_static_tables=tables,
                daily_carbon_dispatch={}, stomate_parameter_values=(),
            )
            assert actual_spec is spec
            return executable

    default = build(_cache_context(0.01))
    assert build(_cache_context("0.010")) is default
    nondefault = build(_cache_context(0.02))
    assert nondefault is not default, "different captured Z0_BARE must not reuse default executable"
    assert build(_cache_context(0.02)) is nondefault


@pytest.mark.parametrize("z0_bare", [0.01, 0.02])
def test_main_and_missing_restart_initialization_bind_bare_roughness(z0_bare):
    restart, driver, slowproc = _bare_inputs()
    main_inputs = {**restart, **driver, **slowproc}
    soilalbedo = main_inputs.pop("soilalbedo_bg")
    # Main roughness dispatch accepts explicit state without albedo; keep the
    # separate initialization albedo inputs fully valid and source-defined.
    for name in ("snow_age", "snow_nobio_age", "tot_bare_soil"):
        main_inputs.pop(name)
    result = condveg_main_minimal(**main_inputs, emis_scal=1.0, z0_bare=z0_bare)
    np.testing.assert_allclose(result.z0m, [z0_bare], rtol=1e-12, atol=1e-14)
    initialize_inputs = {**restart, **driver, **slowproc}
    initialize_inputs.pop("soilalbedo_bg")
    initialize_inputs.update(
        l_first=True, drysoil_frac=np.array([0.5]), z0_bare=z0_bare,
        alb_leaf_vis=np.full(14, 0.1), alb_leaf_nir=np.full(14, 0.2),
        snowa_aged_vis=np.full(14, 0.6), snowa_aged_nir=np.full(14, 0.2),
        snowa_dec_vis=np.full(14, 0.1), snowa_dec_nir=np.full(14, 0.1),
    )
    fields = {"soilalbedo_bg": soilalbedo}
    initialized = condveg_initialize(**initialize_inputs, restart_fields=fields)
    assert initialized.recomputed_roughness
    np.testing.assert_allclose(initialized.z0m, [z0_bare], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(initialized.z0h, [_expected_heat_roughness(z0_bare)], rtol=1e-12, atol=1e-14)
    # Source initialize retains defined restart roughness regardless of the
    # current configured fallback parameter; parameter binding must not turn
    # into an unconditional prognostic/restart rewrite.
    fields.update(z0m=np.array([0.3]), z0h=np.array([0.04]),
                  roughheight=np.array([0.2]), roughheight_pft=np.full((1, 14), 0.2))
    preserved = condveg_initialize(**initialize_inputs, restart_fields=fields)
    assert not preserved.recomputed_roughness
    np.testing.assert_array_equal(preserved.z0m, [0.3])
    np.testing.assert_array_equal(preserved.z0h, [0.04])


@pytest.mark.parametrize("producer", ["ordinary", "compact", "cold"])
def test_current_driver_call_expressions_bind_nondefault_parameter(producer):
    """Replay real source call expressions with self-contained component data.

    This is argument-assembly evidence, not full-driver control-flow/model
    acceptance. The actual selected CONDVEG callable executes numerically.
    """
    from jax_orchidee.driver import orchestration as orch

    restart, driver, slowproc = _bare_inputs()
    values = {**_parameters(0.02), "IMPOSE_AZE": "FALSE", "ROUGH_DYN": "TRUE", "OK_EXPLICITSNOW": "TRUE"}
    scalars = SimpleNamespace(is_tree=np.zeros(14, dtype=bool),
                              z0_over_height=np.full(14, 0.0625), ratio_z0m_z0h=np.full(14, 10.0))
    payload = SimpleNamespace(**driver, kjpindex=1)
    diagnostics = SimpleNamespace(drysoil_frac=np.array([0.5]))
    snow_state = SimpleNamespace(**{key: restart[key] for key in (
        "snow", "snow_nobio", "snowrho", "snowdz", "snow_age", "snow_nobio_age")})
    hydrol_module = SimpleNamespace(snow_state=snow_state, diagnostics=diagnostics)
    slowproc_state = {key: restart[key] for key in ("veget", "veget_max", "frac_nobio", "lai")}
    diffuco_state = {key: restart[key] for key in ("height", "soilalbedo_bg")}
    env = dict(vars(orch))
    env.update(
        np=np, run_def_values=values, payload=payload, init_payload=payload,
        module_jit=False, prepared_context=object(), hydrol_module=hydrol_module,
        context=SimpleNamespace(run_def_values=values, ok_explicitsnow=True, impaze=False, rough_dyn=True),
        step=SimpleNamespace(run_scalars=scalars),
        enerbil=SimpleNamespace(payload=payload, step=SimpleNamespace(run_scalars=scalars)),
        slowproc_state=slowproc_state, diffuco_state=diffuco_state,
        slowproc_derivvar=SimpleNamespace(height=restart["height"]),
        slowproc_veg_cold=SimpleNamespace(
            vegetation=SimpleNamespace(**{**slowproc_state, "totfrac_nobio": slowproc["totfrac_nobio"]}),
            lai=restart["lai"], tot_bare_soil=slowproc["tot_bare_soil"]),
        hydrol_cold=SimpleNamespace(**{key: restart[key] for key in (
            "snow", "snow_age", "snow_nobio", "snow_nobio_age")}, explicit_snow=snow_state),
        soilalbedo_bg=restart["soilalbedo_bg"], fortran_pft_ids=tuple(range(1, 15)),
    )
    function_name, called_name = {
        "ordinary": ("paper_1961_next_step_hydrol_precall_scaffold", "run_condveg_first_step_module"),
        "compact": ("_paper_1961_next_step_runtime_result_compact", "run_condveg_first_step_module"),
        "cold": ("paper_1961_driver_cold_start_first_step_coverage", "condveg_main_minimal"),
    }[producer]
    tree = ast.parse(Path(orch.__file__).read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == called_name]
    assert len(calls) == 1
    result = eval(compile(ast.Expression(calls[0]), orch.__file__, "eval"), env)
    module = result if producer == "cold" else result.module
    assert module is not None
    np.testing.assert_allclose(module.z0m, [0.02], rtol=1e-12, atol=1e-14)
