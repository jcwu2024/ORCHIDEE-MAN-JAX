"""Stage26 A: independent scalar reductions at supported CONDVEG entries.

Source: constantes.f90:447-497; constantes_var.f90:460,688-694;
constantes_soil.f90:70,157-176; constantes_soil_var.f90:97;
condveg.f90:878-893,1590-1600,1679-1702,1706-1718.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from jax_orchidee.driver.init import parse_run_def
from jax_orchidee.sechiba.condveg import run_condveg_first_step_module


def _fixture(kind):
    veget = np.zeros((1, 14))
    veget[0, 0] = 1.0
    nobio = np.zeros((1, 1))
    if kind == "HEIGHT_DISPLACEMENT":
        veget[0, 0], veget[0, 1] = 0.0, 1.0
    if kind == "Z0_ICE":
        veget[:] = 0.0
        nobio[:] = 1.0
    restart = dict(snow=np.array([10.0 if kind == "SNOW_DENSITY" else 0.0]),
                   snow_nobio=np.array([[10.0]]), snowrho=np.full((1, 3), 200.0),
                   snowdz=np.full((1, 3), 0.01 if kind == "explicit-density" else 0.0),
                   veget=veget, veget_max=veget.copy(), frac_nobio=nobio,
                   height=np.full((1, 14), 2.0), lai=np.ones((1, 14)),
                   snow_age=np.zeros(1), snow_nobio_age=np.zeros(1),
                   soilalbedo_bg=np.array([[0.1, 0.2]]))
    driver = dict(zlev=np.array([10.0]), temp_air=np.array([285.0]),
                  pb=np.array([1013.0]), u=np.array([0.05]), v=np.zeros(1))
    slowproc = dict(totfrac_nobio=nobio[:, 0], tot_bare_soil=veget[:, 0])
    return restart, driver, slowproc


def _values(raw):
    values = {"ALB_BG_MODIS": "TRUE", "ALB_BARE_MODEL": "FALSE",
              "CONDVEG_SNOWA": "1e20", "TCST_SNOWA": "10",
              "ALB_ICE__00001": "0.6", "ALB_ICE__00002": "0.2"}
    for name, value in (("ALB_LEAF_VIS", 0.1), ("ALB_LEAF_NIR", 0.2),
                        ("SNOWA_AGED_VIS", 0.6), ("SNOWA_AGED_NIR", 0.2),
                        ("SNOWA_DEC_VIS", 0.1), ("SNOWA_DEC_NIR", 0.1)):
        values.update({f"{name}__{i:05d}": str(value) for i in range(1, 15)})
    return {**values, **raw}


def _run(tmp_path, kind, raw, use_jit, *, explicit=None):
    path = tmp_path / "run.def"
    path.write_text("\n".join(f"{key} = {value}" for key, value in _values(raw).items()), encoding="utf-8")
    restart, driver, slowproc = _fixture(kind)
    closure = run_condveg_first_step_module(
        restart_payload=restart, driver_payload=driver, slowproc_payload=slowproc,
        hydrol_diagnostics=SimpleNamespace(drysoil_frac=np.array([0.5])),
        run_def_values=parse_run_def(path),
        ok_explicitsnow=kind != "SNOW_DENSITY" if explicit is None else explicit,
        use_jit=use_jit,
    )
    assert closure.ok and closure.module.albedo is not None
    return closure.module


def _source_heat_roughness(z0, wind):
    # Source scalar reduction, no implementation helper/resolver as oracle.
    friction = 0.41 * max(wind, 0.05) / np.log(10.0 / z0)
    reynolds = z0 * friction / (1.327e-5 * (285.0/273.15)**1.81)
    return z0 / np.exp(2.46 * reynolds**0.25 - np.log(7.4))


@pytest.mark.parametrize("use_jit", [False, True], ids=["eager", "narrow-jit"])
@pytest.mark.parametrize("gate", ["active", "default", "inactive-invalid"])
@pytest.mark.parametrize("name,default,override", [
    ("MIN_WIND", 0.1, 0.4), ("HEIGHT_DISPLACEMENT", 0.66, 0.5),
    ("Z0_ICE", 0.001, 0.002), ("SNOW_DENSITY", 330.0, 660.0),
])
def test_raw_run_def_supported_wrapper_parameter(tmp_path, name, default, override, gate, use_jit):
    raw = {name: str(override)} if gate == "active" else {}
    if gate == "inactive-invalid":
        raw = {"IMPOSE_PARAM": ".FALSE.", name: "not-read"}
    effective = override if gate == "active" else default
    module = _run(tmp_path, name, raw, use_jit)
    if name == "MIN_WIND":
        actual, expected = module.z0h, [_source_heat_roughness(0.01, effective)]
    elif name == "HEIGHT_DISPLACEMENT":
        actual, expected = module.roughheight, [2.0 * (1.0-effective)]
        np.testing.assert_allclose(module.roughheight_pft[:, 1], expected, rtol=1e-12, atol=1e-14)
    elif name == "Z0_ICE":
        actual, expected = module.z0m, [effective]
        np.testing.assert_allclose(module.z0h, [_source_heat_roughness(effective, 0.1)], rtol=1e-12, atol=1e-14)
    else:
        actual, expected = module.frac_snow_veg, [10.0/(10.0+10.0*effective/100.0)]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(module.frac_snow_nobio, [[0.5]], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("use_jit", [False, True])
def test_snow_density_override_has_no_effect_on_explicit_fraction(tmp_path, use_jit):
    module = _run(tmp_path, "explicit-density", {"SNOW_DENSITY": "660"}, use_jit)
    np.testing.assert_allclose(module.frac_snow_veg, [np.tanh(0.03/(0.025*(200/50)))], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(module.frac_snow_nobio, [[0.5]], rtol=1e-12, atol=1e-14)


# After-only Stage26 acceptance domains; original26 cases above are unchanged.
import ast
from pathlib import Path


def _assert_parameter(module, name, effective):
    if name == "MIN_WIND":
        actual, expected = module.z0h, [_source_heat_roughness(0.01, effective)]
    elif name == "HEIGHT_DISPLACEMENT":
        actual, expected = module.roughheight, [2*(1-effective)]
        np.testing.assert_allclose(module.roughheight_pft[:, 1], expected, rtol=1e-12, atol=1e-14)
    elif name == "Z0_ICE":
        actual, expected = module.z0m, [effective]
        np.testing.assert_allclose(module.z0h, [_source_heat_roughness(effective, 0.1)], rtol=1e-12, atol=1e-14)
    else:
        actual, expected = module.frac_snow_veg, [10/(10+10*effective/100)]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_effective_resolver_gates_defaults_and_density_source_guard():
    from jax_orchidee.sechiba.condveg import resolve_condveg_source_parameters as resolve
    defaults = (0.1, 0.66, 0.001, 330.0)
    raw = dict(MIN_WIND=".4", HEIGHT_DISPLACEMENT=".5", Z0_ICE=".002", SNOW_DENSITY="660")
    assert resolve({}, ok_sechiba=True) == defaults
    assert resolve(raw, ok_sechiba=True) == (0.4, 0.5, 0.002, 660.0)
    assert resolve({**raw, "OK_SECHIBA": "FALSE"}, ok_sechiba=True) == (0.4, 0.5, 0.002, 660.0)
    invalid = {key: "not-read" for key in raw}
    assert resolve({**invalid, "IMPOSE_PARAM": ".NO."}, ok_sechiba=True) == defaults
    class Unreadable(dict):
        def get(self, *args, **kwargs):
            raise AssertionError("initialized-off reads no raw key")
    assert resolve(Unreadable(), ok_sechiba=False) == defaults
    with pytest.raises(ValueError, match="boolean"):
        resolve({"IMPOSE_PARAM": "typo"}, ok_sechiba=True)
    for density in ("0", "-1"):
        with pytest.raises(ValueError, match="SNOW_DENSITY.*positive"):
            resolve({"SNOW_DENSITY": density}, ok_sechiba=True)


@pytest.mark.parametrize("name,effective", [("MIN_WIND", .4), ("HEIGHT_DISPLACEMENT", .5),
                                           ("Z0_ICE", .002), ("SNOW_DENSITY", 660.0)])
def test_main_initialize_explicit_effective_scalars_and_restart(name, effective):
    from jax_orchidee.sechiba.condveg import condveg_main_minimal, condveg_initialize, CondvegInitializeBoundaryError
    restart, driver, slowproc = _fixture(name)
    kwargs = dict(min_wind=.1, height_displacement=.66, z0_ice=.001, sn_dens=330.)
    kwargs[{"MIN_WIND": "min_wind", "HEIGHT_DISPLACEMENT": "height_displacement",
            "Z0_ICE": "z0_ice", "SNOW_DENSITY": "sn_dens"}[name]] = effective
    kwargs["ok_explicitsnow"] = name != "SNOW_DENSITY"
    inputs = {**restart, **driver, **slowproc}
    for key in ("soilalbedo_bg", "snow_age", "snow_nobio_age", "tot_bare_soil"):
        inputs.pop(key)
    _assert_parameter(condveg_main_minimal(**inputs, **kwargs, emis_scal=1.), name, effective)
    initialize = {**restart, **driver, **slowproc, **kwargs}
    soil = initialize.pop("soilalbedo_bg")
    initialize.update(l_first=True, drysoil_frac=np.array([.5]),
        alb_leaf_vis=np.full(14,.1), alb_leaf_nir=np.full(14,.2),
        snowa_aged_vis=np.full(14,.6), snowa_aged_nir=np.full(14,.2),
        snowa_dec_vis=np.full(14,.1), snowa_dec_nir=np.full(14,.1))
    fields = {"soilalbedo_bg": soil}
    initialized = condveg_initialize(**initialize, restart_fields=fields)
    assert initialized.recomputed_roughness
    _assert_parameter(initialized, name, effective)
    fields.update(z0m=np.array([.3]), z0h=np.array([.04]), roughheight=np.array([.2]), roughheight_pft=np.full((1,14),.2))
    preserved = condveg_initialize(**initialize, restart_fields=fields)
    assert not preserved.recomputed_roughness
    np.testing.assert_array_equal(preserved.z0m, [.3])
    np.testing.assert_array_equal(preserved.z0h, [.04])
    if name == "SNOW_DENSITY":
        _assert_parameter(preserved, name, effective)
    with pytest.raises(CondvegInitializeBoundaryError):
        condveg_initialize(**{**initialize, "rough_dyn": False}, restart_fields=fields)


@pytest.mark.parametrize("use_jit", [False, True])
def test_wrapper_explicit_effective_scalars_ignore_their_invalid_raw(use_jit):
    restart, driver, slowproc = _fixture("MIN_WIND")
    closure = run_condveg_first_step_module(restart_payload=restart, driver_payload=driver,
        slowproc_payload=slowproc, hydrol_diagnostics=SimpleNamespace(drysoil_frac=np.array([.5])),
        run_def_values=_values({"IMPOSE_PARAM": "typo", "MIN_WIND": "not-read", "HEIGHT_DISPLACEMENT": "not-read",
                               "Z0_ICE": "not-read", "SNOW_DENSITY": "not-read", "SNOWCRI_ALB": "not-read"}),
        snowcri_alb=10., min_wind=.4, height_displacement=.5, z0_ice=.002, sn_dens=660., use_jit=use_jit)
    assert closure.ok and closure.module.albedo is not None
    _assert_parameter(closure.module, "MIN_WIND", .4)


def _cache_context(raw):
    return SimpleNamespace(
        nvm=14, dt_sechiba=1800.0, ok_explicitsnow=True,
        ok_freeze_cwrr=False, ok_thermodynamical_freezing=False,
        hydrol_soil_peat_hydro=False, peat_hydro=False, hydrol_ok_pc=False,
        hydrol_ok_leak=False, hydrol_tides=False, ok_laidev=np.zeros(14, dtype=bool),
        run_def_values=dict(raw),
        diffuco_pft14_static_kwargs={"trans_co2_pft_params": {}},
        config_path=Path("self-contained-stage26-config.json"),
        runtime=SimpleNamespace(dt_stomate=1800.0, dt_sechiba=1800.0),
        first_step_restart_state=SimpleNamespace(
            stomate=(), stomate_readstart=SimpleNamespace(
                season_state=SimpleNamespace(provenance=(), _asdict=lambda: {"provenance": ()}))),
    )


@pytest.mark.parametrize("entry", ["sechiba-scan", "later-day-block"])
def test_actual_cache_constructors_capture_four_effective_parameters(monkeypatch, entry):
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
    names = {"MIN_WIND": ".4", "HEIGHT_DISPLACEMENT": ".5", "Z0_ICE": ".002", "SNOW_DENSITY": "660"}
    assert build(_cache_context({"IMPOSE_PARAM": "FALSE", **{name: "not-read" for name in names}})) is default
    for name, value in names.items():
        distinct = build(_cache_context({name: value}))
        assert distinct is not default, name
        assert build(_cache_context({name: str(float(value))})) is distinct


@pytest.mark.parametrize("producer", ["ordinary", "compact", "cold-wrapper", "cold-main", "cold-frac"])
@pytest.mark.parametrize("name,effective", [("MIN_WIND", .4), ("HEIGHT_DISPLACEMENT", .5),
                                           ("Z0_ICE", .002), ("SNOW_DENSITY", 660.),
                                           ("SNOW_DENSITY", 330.)])
def test_actual_driver_call_expressions_bind_four_parameters(producer, name, effective):
    """Replay real source call expressions with self-contained component data.

    This is argument-assembly evidence, not full-driver control-flow/model
    acceptance. The actual selected CONDVEG callable executes numerically.
    """
    from jax_orchidee.driver import orchestration as orch

    restart, driver, slowproc = _fixture(name)
    raw = {name: str(effective)} if effective != 330. else {"IMPOSE_PARAM": "FALSE", name: "not-read"}
    values = {**_values(raw), "IMPOSE_AZE": "FALSE", "ROUGH_DYN": "TRUE", "OK_EXPLICITSNOW": "FALSE" if name == "SNOW_DENSITY" else "TRUE"}
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
        context=SimpleNamespace(run_def_values=values, ok_explicitsnow=name != "SNOW_DENSITY", impaze=False, rough_dyn=True),
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
    if producer == "cold-frac" and name != "SNOW_DENSITY":
        # This source call produces snow fractions only; roughness parameters
        # have no consumer here. All four actual wrapper/main calls above do.
        np.testing.assert_array_equal(module.frac_snow_veg, [0.0])
    else:
        _assert_parameter(module, name, effective)
