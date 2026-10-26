"""Stage25 D: SNOWCRI raw/effective binding with independent source fractions.

Fortran source: control.f90:455,495-503,550-555 and
intersurf.f90:2267,2330-2335 initialize SECHIBA and gate parameter reads;
constantes_var.f90:129,697 defaults IMPOSE_PARAM true/SNOWCRI_ALB10;
constantes.f90:507-513 reads the active scalar; condveg.f90:893 gives
positive snow_nobio/(positive snow_nobio+snowcri_alb).

One point/14PFT, valid bare+nobio area, no ignored-output fixtures. Raw literal
run.def parsing, wrapper numerics, source-call assembly and cache-construction
domains remain distinct; no full driver execution or model compilation.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jax_orchidee.driver.init import parse_run_def
from jax_orchidee.sechiba.condveg import (
    CondvegInitializeBoundaryError,
    condveg_initialize,
    condveg_main_minimal,
    resolve_condveg_snowcri_alb,
    run_condveg_first_step_module,
)


def _inputs():
    veget = np.zeros((1, 14), dtype=np.float64)
    veget[0, 0] = 0.8
    restart = {
        "snow": np.zeros(1), "snow_nobio": np.array([[10.0]]),
        "snowrho": np.full((1, 3), 100.0), "snowdz": np.zeros((1, 3)),
        "veget": veget, "veget_max": veget.copy(), "frac_nobio": np.array([[0.2]]),
        "height": np.ones((1, 14)), "lai": np.zeros((1, 14)),
        "snow_age": np.zeros(1), "snow_nobio_age": np.zeros(1),
        "soilalbedo_bg": np.array([[0.1, 0.2]]),
    }
    driver = {"zlev": np.array([10.0]), "temp_air": np.array([285.0]),
              "pb": np.array([1013.0]), "u": np.array([2.0]), "v": np.zeros(1)}
    slowproc = {"totfrac_nobio": np.array([0.2]), "tot_bare_soil": np.array([0.8])}
    return restart, driver, slowproc


def _parameters(overrides=None):
    values = {"ALB_BG_MODIS": "TRUE", "ALB_BARE_MODEL": "FALSE",
              "CONDVEG_SNOWA": "1e20", "TCST_SNOWA": "10",
              "ALB_ICE__00001": "0.6", "ALB_ICE__00002": "0.2",
              "OK_EXPLICITSNOW": "TRUE", "IMPOSE_AZE": "FALSE", "ROUGH_DYN": "TRUE"}
    for name, value in (("ALB_LEAF_VIS", 0.1), ("ALB_LEAF_NIR", 0.2),
                        ("SNOWA_AGED_VIS", 0.6), ("SNOWA_AGED_NIR", 0.2),
                        ("SNOWA_DEC_VIS", 0.1), ("SNOWA_DEC_NIR", 0.1)):
        values.update({f"{name}__{i:05d}": str(value) for i in range(1, 15)})
    values.update({} if overrides is None else overrides)
    return values


def _wrapper(values, **kwargs):
    restart, driver, slowproc = _inputs()
    return run_condveg_first_step_module(
        restart_payload=restart, driver_payload=driver, slowproc_payload=slowproc,
        hydrol_diagnostics=SimpleNamespace(drysoil_frac=np.array([0.5])),
        run_def_values=values, **kwargs,
    )


@pytest.mark.parametrize("use_jit", [False, True], ids=["eager", "narrow-jit"])
@pytest.mark.parametrize(("raw", "source_effective"), [
    pytest.param({"IMPOSE_PARAM": "TRUE", "SNOWCRI_ALB": "20"}, 20.0, id="gate-on20"),
    pytest.param({"SNOWCRI_ALB": "20"}, 20.0, id="missing-impose-source-default-true"),
    pytest.param({"IMPOSE_PARAM": "FALSE", "SNOWCRI_ALB": "20"}, 10.0, id="gate-off-ignore20"),
    pytest.param({"IMPOSE_PARAM": ".FALSE.", "SNOWCRI_ALB": "not-read"}, 10.0, id="gate-off-ignore-invalid-raw"),
    pytest.param({"IMPOSE_PARAM": "TRUE"}, 10.0, id="gate-on-missing-scalar-default10"),
    pytest.param({}, 10.0, id="missing-gate-and-scalar-source-defaults"),
])
def test_raw_run_def_to_sechiba_wrapper_fraction(tmp_path, raw, source_effective, use_jit):
    # The real wrapper is already a SECHIBA entry: source control455 and
    # intersurf2267 set ok_sechiba=true; it is not a raw OK_SECHIBA switch.
    run_def = tmp_path / "run.def"
    run_def.write_text("\n".join(f"{key} = {value}" for key, value in _parameters(raw).items())+"\n", encoding="utf-8")
    values = parse_run_def(run_def)
    closure = _wrapper(values, use_jit=use_jit)
    assert closure.ok and closure.module.albedo is not None
    # Literal source arithmetic with independently specified effective values;
    # never call the implementation resolver to construct an expected value.
    expected = 10.0 / (10.0 + source_effective)
    np.testing.assert_allclose(closure.module.frac_snow_nobio, [[expected]], rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(closure.module.frac_snow_veg, [0.0])


@pytest.mark.parametrize("raw,ok_sechiba,expected", [
    ({"IMPOSE_PARAM": "not-read", "SNOWCRI_ALB": "not-read"}, False, 10.0),
    ({"IMPOSE_PARAM": "FALSE", "SNOWCRI_ALB": "not-read"}, True, 10.0),
    ({"IMPOSE_PARAM": ".F.", "SNOWCRI_ALB": "not-read"}, True, 10.0),
    ({"IMPOSE_PARAM": "N", "SNOWCRI_ALB": "not-read"}, True, 10.0),
    ({"IMPOSE_PARAM": "0", "SNOWCRI_ALB": "not-read"}, True, 10.0),
    ({"IMPOSE_PARAM": ".T.", "SNOWCRI_ALB": "20"}, True, 20.0),
    ({"IMPOSE_PARAM": "Y", "SNOWCRI_ALB": "20"}, True, 20.0),
    ({"IMPOSE_PARAM": "1", "SNOWCRI_ALB": "20"}, True, 20.0),
    ({"SNOWCRI_ALB": "20", "OK_SECHIBA": "FALSE"}, True, 20.0),
    ({}, True, 10.0),
])
def test_source_gated_effective_scalar(raw, ok_sechiba, expected):
    assert resolve_condveg_snowcri_alb(raw, ok_sechiba=ok_sechiba) == expected


def test_resolver_requires_initialized_state_and_does_not_hide_active_invalid_scalar():
    with pytest.raises(TypeError):
        resolve_condveg_snowcri_alb({})
    with pytest.raises(ValueError):
        resolve_condveg_snowcri_alb({"SNOWCRI_ALB": "not-a-scalar"}, ok_sechiba=True)


def test_revision_strict_impose_rejects_unknown_active_literal():
    # D-R1 is the established boolean input contract, not a Fortran numeric
    # counterexample. The scalar is valid so only the gate can reject input.
    with pytest.raises(ValueError, match="boolean"):
        resolve_condveg_snowcri_alb(
            {"IMPOSE_PARAM": "typo", "SNOWCRI_ALB": "20"}, ok_sechiba=True
        )


@pytest.mark.parametrize("literal,expected", [
    (" .YeS. ", 20.0), (" .nO. ", 10.0), (True, 20.0), (False, 10.0),
])
def test_revision_strict_impose_recognized_controls(literal, expected):
    raw_scalar = "20" if expected == 20.0 else "not-read"
    assert resolve_condveg_snowcri_alb(
        {"IMPOSE_PARAM": literal, "SNOWCRI_ALB": raw_scalar}, ok_sechiba=True
    ) == expected


def test_revision_initialized_off_never_reads_raw_mapping():
    class RawReadForbidden(dict):
        def get(self, *args, **kwargs):
            raise AssertionError("inactive initialized SECHIBA must not read raw settings")

    assert resolve_condveg_snowcri_alb(RawReadForbidden(), ok_sechiba=False) == 10.0


@pytest.mark.parametrize("use_jit", [False, True])
def test_explicit_effective_wrapper_api_bypasses_raw_mapping(use_jit):
    closure = _wrapper(_parameters({"IMPOSE_PARAM": "FALSE", "SNOWCRI_ALB": "not-read"}),
                       snowcri_alb=20.0, use_jit=use_jit)
    np.testing.assert_allclose(closure.module.frac_snow_nobio, [[1.0/3.0]], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("effective", [10.0, 20.0])
def test_direct_main_and_initialize_forward_effective_scalar(effective):
    restart, driver, slowproc = _inputs()
    main = {**restart, **driver, **slowproc}
    soilalbedo = main.pop("soilalbedo_bg")
    # Exercise the public fraction/emissivity/roughness subset without
    # requesting an albedo update; initialization below supplies full albedo.
    for name in ("tot_bare_soil", "snow_age", "snow_nobio_age"):
        main.pop(name)
    result = condveg_main_minimal(**main, emis_scal=1.0, snowcri_alb=effective)
    np.testing.assert_allclose(result.frac_snow_nobio, [[10/(10+effective)]], rtol=1e-12, atol=1e-14)
    initialize = {**restart, **driver, **slowproc}
    initialize.pop("soilalbedo_bg")
    initialize.update(
        l_first=True, drysoil_frac=np.array([0.5]), snowcri_alb=effective,
        alb_leaf_vis=np.full(14, 0.1), alb_leaf_nir=np.full(14, 0.2),
        snowa_aged_vis=np.full(14, 0.6), snowa_aged_nir=np.full(14, 0.2),
        snowa_dec_vis=np.full(14, 0.1), snowa_dec_nir=np.full(14, 0.1),
    )
    fields = {"soilalbedo_bg": soilalbedo}
    initialized = condveg_initialize(**initialize, restart_fields=fields)
    np.testing.assert_allclose(initialized.frac_snow_nobio, [[10/(10+effective)]], rtol=1e-12, atol=1e-14)
    fields.update(z0m=np.array([0.3]), z0h=np.array([0.04]),
                  roughheight=np.array([0.2]), roughheight_pft=np.full((1, 14), 0.2))
    preserved = condveg_initialize(**initialize, restart_fields=fields)
    assert not preserved.recomputed_roughness
    np.testing.assert_array_equal(preserved.z0m, [0.3])
    np.testing.assert_allclose(preserved.frac_snow_nobio, [[10/(10+effective)]], rtol=1e-12, atol=1e-14)
    # Effective scalar forwarding must not admit target-external initializer modes.
    with pytest.raises(CondvegInitializeBoundaryError):
        condveg_initialize(**{**initialize, "rough_dyn": False}, restart_fields=fields)


def _cache_context(raw):
    return SimpleNamespace(
        nvm=14, dt_sechiba=1800.0, ok_explicitsnow=True,
        ok_freeze_cwrr=False, ok_thermodynamical_freezing=False,
        hydrol_soil_peat_hydro=False, peat_hydro=False, hydrol_ok_pc=False,
        hydrol_ok_leak=False, hydrol_tides=False, ok_laidev=np.zeros(14, dtype=bool),
        run_def_values=dict(raw),
        diffuco_pft14_static_kwargs={"trans_co2_pft_params": {}},
        config_path=Path("self-contained-stage25-config.json"),
        runtime=SimpleNamespace(dt_stomate=1800.0, dt_sechiba=1800.0),
        first_step_restart_state=SimpleNamespace(
            stomate=(), stomate_readstart=SimpleNamespace(
                season_state=SimpleNamespace(provenance=(), _asdict=lambda: {"provenance": ()}))),
    )


@pytest.mark.parametrize("entry", ["sechiba-scan", "later-day-block"])
def test_real_cache_construction_uses_effective_snowcri(monkeypatch, entry):
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

    default = build(_cache_context({}))
    for raw in ({"SNOWCRI_ALB": "10"},
                {"IMPOSE_PARAM": "FALSE", "SNOWCRI_ALB": "20"},
                {"IMPOSE_PARAM": ".FALSE.", "SNOWCRI_ALB": "30"},
                {"IMPOSE_PARAM": "FALSE", "SNOWCRI_ALB": "not-read"}):
        assert build(_cache_context(raw)) is default
    nondefault = build(_cache_context({"SNOWCRI_ALB": "20"}))
    assert nondefault is not default, "different effective SNOWCRI_ALB must isolate executable"
    assert build(_cache_context({"IMPOSE_PARAM": "TRUE", "SNOWCRI_ALB": "20.0"})) is nondefault


@pytest.mark.parametrize("producer", ["ordinary", "compact", "cold-wrapper", "cold-main", "cold-frac"])
@pytest.mark.parametrize("raw,expected", [
    ({"SNOWCRI_ALB": "20"}, 1.0/3.0),
    ({"IMPOSE_PARAM": "FALSE", "SNOWCRI_ALB": "not-read"}, 0.5),
    ({}, 0.5),
], ids=["source-default-impose-active20", "inactive-invalid-raw", "default10"])
def test_current_driver_call_expressions_bind_effective_snowcri(producer, raw, expected):
    """Replay real source call expressions with self-contained component data.

    This is argument-assembly evidence, not full-driver control-flow/model
    acceptance. The actual selected CONDVEG callable executes numerically.
    """
    from jax_orchidee.driver import orchestration as orch

    restart, driver, slowproc = _inputs()
    values = _parameters(raw)
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
        np=np, run_def_values=values, payload=payload, init_payload=payload, run_scalars=scalars,
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
        "cold-wrapper": ("paper_1961_driver_cold_start_first_step_coverage", "run_condveg_first_step_module"),
        "cold-main": ("paper_1961_driver_cold_start_first_step_coverage", "condveg_main_minimal"),
        "cold-frac": ("paper_1961_driver_cold_start_first_step_coverage", "condveg_frac_snow"),
    }[producer]
    tree = ast.parse(Path(orch.__file__).read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == called_name]
    assert len(calls) == 1
    result = eval(compile(ast.Expression(calls[0]), orch.__file__, "eval"), env)
    module = result if producer in {"cold-main", "cold-frac"} else result.module
    assert module is not None
    np.testing.assert_allclose(module.frac_snow_nobio, [[expected]], rtol=1e-12, atol=1e-14)
