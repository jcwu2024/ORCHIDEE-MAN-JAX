"""Cold input interfaces only: no physical scaffold/kernel/JIT is executed.

Source: thermosoil.f90:202-204,338-371,534-567,2631-2687; control261-278.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from jax_orchidee.driver import orchestration as o

ROOT = Path(__file__).resolve().parents[2]
RED_CONTROLS = []
COUNTS = {"static_reader_spy_calls": 0, "coefficient_expression_spy_calls": 0}


def values(**overrides):
    result = dict(OK_PC="FALSE", OK_LEAK="TRUE", OK_FREEZE_THERMIX="TRUE",
                  OK_EXPLICITSNOW="TRUE", USE_SOILC_TEMPDIFF="TRUE", satsoil="FALSE")
    result.update(overrides)
    return result


def controls(**overrides):
    return o._fresh_cold_thermosoil_controls(values(**overrides))


def function_tree(source=None):
    tree = ast.parse(source or (ROOT / "jax_orchidee/driver/orchestration.py").read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "paper_1961_driver_cold_start_first_step_coverage")


def actual_call(name):
    return next(n for n in ast.walk(function_tree()) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == name)


def expression(node, env):
    return eval(compile(ast.Expression(node), "<actual-cold-caller-expression>", "eval"), env)


@pytest.mark.parametrize("pc,leak,freeze,gate", [
    (False, False, False, False), (True, False, False, False),
    (False, True, False, True), (True, True, False, True),
    (False, False, True, False), (True, False, True, True),
    (False, True, True, True), (True, True, True, True),
])
def test_fresh_module_source_gate_and_conditional_refsoc_read(pc, leak, freeze, gate):
    c = controls(OK_PC=str(pc), OK_LEAK=str(leak), OK_FREEZE_THERMIX=str(freeze), use_refSOC="FALSE")
    assert c.source_config_gate is gate
    assert c.use_soilc_tempdiff is True
    assert c.use_refsoc is (not gate)  # raw FALSE is read only inside G and soilC.


@pytest.mark.parametrize("soilc,raw_ref,expected_soilc,expected_ref", [
    (None, None, False, True), (None, "FALSE", False, True),
    ("FALSE", "not-a-boolean", False, True),
    ("TRUE", None, True, True), ("TRUE", "TRUE", True, True),
    ("TRUE", "FALSE", True, False),
])
def test_inside_gate_source_defaults_are_not_a_boolean_conjunction(soilc, raw_ref, expected_soilc, expected_ref):
    v = values()
    v.pop("USE_SOILC_TEMPDIFF")
    if soilc is not None:
        v["USE_SOILC_TEMPDIFF"] = soilc
    if raw_ref is not None:
        v["use_refSOC"] = raw_ref
    c = o._fresh_cold_thermosoil_controls(v)
    assert (c.use_soilc_tempdiff, c.use_refsoc) == (expected_soilc, expected_ref)


def test_outside_gate_preserves_fresh_module_defaults_and_ignores_unread_invalid_values():
    c = controls(OK_PC="FALSE", OK_LEAK="FALSE", USE_SOILC_TEMPDIFF="invalid",
                 USE_TOPORGANICLAYER_TEMPDIFF="invalid", use_refSOC="invalid")
    assert not c.source_config_gate
    assert c.use_soilc_tempdiff and c.use_refsoc
    assert not c.use_toporganiclayer_tempdiff


def test_source_soilcarbon_suppresses_toporganic_and_specific_freeze_overrides_general():
    c = controls(USE_TOPORGANICLAYER_TEMPDIFF="TRUE", OK_FREEZE="FALSE")
    assert not c.use_toporganiclayer_tempdiff and c.ok_freeze_thermix
    c = controls(OK_FREEZE="TRUE", OK_FREEZE_THERMIX="FALSE")
    assert not c.ok_freeze_thermix


@pytest.mark.parametrize("general,expected", [(None, False), ("FALSE", False), ("TRUE", True)])
def test_missing_specific_freeze_uses_source_general_default(general, expected):
    v = values()
    v.pop("OK_FREEZE_THERMIX")
    if general is not None:
        v["OK_FREEZE"] = general
    assert o._fresh_cold_thermosoil_controls(v).ok_freeze_thermix is expected


@pytest.mark.parametrize("key", ["OK_PC", "OK_LEAK"])
def test_materialized_driver_keys_remain_required(key):
    v = values()
    v.pop(key)
    with pytest.raises(KeyError, match=key):
        o._fresh_cold_thermosoil_controls(v)


def domain(resolution=True):
    return NS(lalo=np.array([[319.0, 57.0]]), resolution=np.array([[1000.0, 2000.0]]) if resolution else None)


def reader(monkeypatch, result=None, error=None):
    calls = []
    result = result if result is not None else np.array([[1.25, 2.5, 4.0]])
    metadata = {"refsoc_sub_area": np.array([[1.0, 2.0]])}

    def spy(config, **kwargs):
        COUNTS["static_reader_spy_calls"] += 1
        calls.append((config, kwargs))
        if error is not None:
            raise error
        return result, metadata

    monkeypatch.setattr(o, "read_paper_thermosoil_refsoc_static_field", spy)
    return calls, result, metadata


@pytest.mark.parametrize("raw", [None, "TRUE"])
def test_absent_and_true_bind_existing_static_reader_without_fabricating_data(monkeypatch, raw):
    c = controls(**({} if raw is None else {"use_refSOC": raw}))
    calls, result, metadata = reader(monkeypatch)
    d = domain()
    actual, overlap = o._fresh_cold_thermosoil_refsoc_input("case.yaml", controls=c, domain=d, npts=1, ngrnd=3)
    assert actual is result and overlap is metadata
    assert len(calls) == 1 and calls[0][0] == "case.yaml"
    assert calls[0][1]["lalo"] is d.lalo and calls[0][1]["resolution_m"] is d.resolution


def test_source_read_uses_refsoc_alone_even_when_soilcarbon_is_disabled(monkeypatch):
    c = controls(USE_SOILC_TEMPDIFF="FALSE", use_refSOC="FALSE")
    assert not c.use_soilc_tempdiff and c.use_refsoc
    calls, _, _ = reader(monkeypatch)
    o._fresh_cold_thermosoil_refsoc_input("case.yaml", controls=c, domain=domain(), npts=1, ngrnd=3)
    assert len(calls) == 1
    assert "mineral_thermal_fraction_consumer_binding" in o._fresh_cold_thermosoil_missing_inputs(c)


def test_inactive_refsoc_path_does_not_require_geometry_or_read_external_data(monkeypatch):
    calls, _, _ = reader(monkeypatch)
    assert o._fresh_cold_thermosoil_refsoc_input(
        "missing.yaml", controls=controls(use_refSOC="FALSE"), domain=NS(), npts=1, ngrnd=3,
    ) == (None, None)
    assert not calls


def test_active_absent_key_is_distinct_from_missing_resolution(monkeypatch):
    calls, _, _ = reader(monkeypatch)
    with pytest.raises(ValueError, match="explicit driver resolution"):
        o._fresh_cold_thermosoil_refsoc_input("case.yaml", controls=controls(), domain=domain(False), npts=1, ngrnd=3)
    assert not calls


@pytest.mark.parametrize("error", [KeyError("refsoc_3d_file"), FileNotFoundError("missing refSOC.nc"), KeyError("soil_organic_carbon")])
def test_static_reader_missing_config_file_variable_errors_are_preserved(monkeypatch, error):
    reader(monkeypatch, error=error)
    with pytest.raises(type(error)) as caught:
        o._fresh_cold_thermosoil_refsoc_input("case.yaml", controls=controls(), domain=domain(), npts=1, ngrnd=3)
    assert caught.value is error


@pytest.mark.parametrize("bad,reason", [(np.zeros((3, 1)), "shape"), (np.zeros((1, 2)), "shape"),
                                          (np.array([[1.0, np.nan, 2.0]]), "finite"),
                                          (np.array([[1.0, np.inf, 2.0]]), "finite")])
def test_static_shape_and_nonfinite_errors_precede_any_coefficient_consumer(monkeypatch, bad, reason):
    reader(monkeypatch, result=bad)
    with pytest.raises(ValueError, match=reason):
        o._fresh_cold_thermosoil_refsoc_input("case.yaml", controls=controls(), domain=domain(), npts=1, ngrnd=3)


@pytest.mark.parametrize("override,bound", [(str(ROOT / "fixtures/refSOC.nc"), True),
                                             (str(ROOT / "fixtures/other.nc"), False), ("refSOC.nc", False)])
def test_explicit_filename_must_match_existing_absolute_config_identity(monkeypatch, override, bound):
    configured = str(ROOT / "fixtures/refSOC.nc")
    c = controls(SOIL_REFSOC_FILE=override)
    monkeypatch.setattr(o, "load_case_config", lambda _: {"static_inputs": {"refsoc_3d_file": configured}})
    calls, _, _ = reader(monkeypatch)
    if bound:
        assert not o._fresh_cold_thermosoil_missing_inputs(c, configured_refsoc_file=configured)
        o._fresh_cold_thermosoil_refsoc_input("case.yaml", controls=c, domain=domain(), npts=1, ngrnd=3)
        assert len(calls) == 1
    else:
        assert "SOIL_REFSOC_FILE_config_static_identity_binding" in o._fresh_cold_thermosoil_missing_inputs(c, configured_refsoc_file=configured)
        with pytest.raises(ValueError, match="not bound"):
            o._fresh_cold_thermosoil_refsoc_input("case.yaml", controls=c, domain=domain(), npts=1, ngrnd=3)
        assert not calls


@pytest.mark.parametrize("overrides,reason", [
    ({"use_refSOC": "FALSE"}, "modeled_soilc_total_or_zx1_consumer_binding"),
    ({"USE_SOILC_TEMPDIFF": "FALSE"}, "mineral_thermal_fraction_consumer_binding"),
    ({"USE_SOILC_TEMPDIFF": "FALSE", "USE_TOPORGANICLAYER_TEMPDIFF": "TRUE"}, "toporganiclayer_consumer_binding"),
    ({"OK_EXPLICITSNOW": "FALSE"}, "non_explicit_snow_cold_coefficient_binding"),
    ({"OK_FREEZE_THERMIX": "FALSE"}, "nonfreezing_first_step_consumer_binding"),
    ({"BEDROCK_FLAG": "1"}, "bedrock_first_step_consumer_binding"),
    ({"OK_PC": "FALSE", "OK_LEAK": "FALSE"}, "source_freeze_effective_PC_or_LEAK_gate"),
    ({"SOIL_REFSOC_FILE": "custom.nc"}, "SOIL_REFSOC_FILE_config_static_identity_binding"),
])
def test_actual_cold_entry_fails_closed_before_loading_any_physical_inputs(monkeypatch, overrides, reason):
    config = {"minimal_single_case": {"cold_start": True}, "restart_protocol": {"cold_start": {
        "RESTART_FILEIN": "NONE", "SECHIBA_restart_in": "NONE", "STOMATE_RESTART_FILEIN": "NONE"}}}
    monkeypatch.setattr(o, "load_case_config", lambda _: config)
    entered = []

    def forbidden(*args, **kwargs):
        entered.append(True)
        raise AssertionError("physical/input bundle must not be entered")

    monkeypatch.setattr(o, "load_paper_1961_first_step_bundle", forbidden)
    monkeypatch.setattr(o, "load_paper_1961_step_bundle", forbidden)
    context = NS(run_def_path=ROOT / "not-read.def", run_def_values=values(**overrides))
    result = o.paper_1961_driver_cold_start_first_step_coverage("not-read.yaml", prepared_context=context)
    assert not result.ready_for_cold_start_first_step and not entered
    assert reason in result.missing_by_component["thermosoil"]
    assert result.covered_by_component == {}
    assert "Source-valid" in result.notes[0] and "external" in result.notes[0]


@pytest.mark.parametrize("raw_ref", [None, "TRUE"])
def test_actual_cold_reader_and_coefficient_keyword_expressions_share_resolved_controls(monkeypatch, raw_ref):
    c = controls(satsoil="TRUE", **({} if raw_ref is None else {"use_refSOC": raw_ref}))
    calls, result, _ = reader(monkeypatch)
    env = {"_fresh_cold_thermosoil_refsoc_input": o._fresh_cold_thermosoil_refsoc_input,
           "config_path": "case.yaml", "cold_thermal_controls": c, "step": NS(domain=domain()),
           "payload": NS(kjpindex=1), "cwrr_grid": NS(znt=np.array([0.1, 0.3, 0.7])), "len": len}
    bound, _ = expression(actual_call("_fresh_cold_thermosoil_refsoc_input"), env)
    env.update(refsoc=bound)
    call = actual_call("thermosoil_cold_start_coef_closure")
    wanted = {"refsoc", "use_refSOC", "use_soilc_tempdiff", "ok_freeze_thermix", "satsoil", "brk_flag"}
    kwargs = {kw.arg: expression(kw.value, env) for kw in call.keywords if kw.arg in wanted}
    COUNTS["coefficient_expression_spy_calls"] += 1
    assert kwargs == dict(refsoc=result, use_refSOC=True, use_soilc_tempdiff=True,
                         ok_freeze_thermix=True, satsoil=True, brk_flag=0)
    first = actual_call("run_thermosoil_first_step_module")
    assert expression(next(kw.value for kw in first.keywords if kw.arg == "satsoil"), env) is True
    assert len(calls) == 1


def test_actual_namespace_and_carry_assignments_preserve_sourced_refsoc_identity():
    sourced = np.array([[2.0, 5.0, 9.0]])
    coef = NS(ok=True, coef=NS(soil=NS(cgrnd=object(), dgrnd=object()), cgrnd_snow=object(),
                             dgrnd_snow=object(), lambda_snow=object()), getdiff=NS(pcapa_en=object()))
    namespace = o._thermosoil_restart_namespace_from_cold_start(
        ptn=np.zeros((1, 3, 2)), coef=coef, refsoc=sourced, temp_sol_beg=np.array([280.0]),
        shum_ngrnd_permalong=np.ones((1, 3, 2)),
    )
    assert namespace.refsoc is sourced and namespace.pcapa_en is coef.getdiff.pcapa_en
    source = ast.parse((ROOT / "jax_orchidee/driver/orchestration.py").read_text(encoding="utf-8"))
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "_previous_step_state_from_cold_start_payloads")
    mirror_if = next(n for n in function.body if isinstance(n, ast.If)
                     and "payloads.get('thermosoil_refSOC')" in ast.unparse(n.test))
    finalize_if = next(n for n in function.body if isinstance(n, ast.If) and ast.unparse(n.test) == "refsoc is not None")
    env = {"payloads": {"thermosoil_refSOC": sourced}, "fields": {"thermosoil_previous_step_state": {}},
           "refsoc": sourced, "carry": {}}
    exec(compile(ast.Module(body=[mirror_if, finalize_if], type_ignores=[]), "<actual-cold-carry>", "exec"), env)
    assert env["fields"]["thermosoil_previous_step_state"]["refSOC"] is sourced
    assert env["carry"]["refsoc"] is sourced


def test_actual_cold_day_caller_maps_partial_input_coverage_to_stopped_gap_without_indexing_payloads():
    tree = ast.parse((ROOT / "jax_orchidee/driver/orchestration.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "paper_1961_driver_cold_start_day_scaffold")
    guard = next(n for n in fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == "previous_state is None")
    first = o.ColdStartFirstStepStateCoverage(
        mode=o.COLD_START_INITIAL_STATE_MODE, covered_by_component={},
        missing_by_component={"thermosoil": ("modeled_soilc_total_or_zx1_consumer_binding",)},
        initialized_payloads={},
    )
    env = {"previous_state": None, "first": first, "DriverDayStateGap": o.DriverDayStateGap}
    exec(compile(ast.Module(body=[guard], type_ignores=[]), "<actual-cold-day-missing-input-guard>", "exec"), env)
    assert env["stopped_at_tstep"] == 0
    assert env["state_gaps"][0].fields == first.missing_components
    # The selected actual daily adapter also exits before accessing any payload.
    assert o._paper_cold_start_day_daily_process_from_completed_entries(
        first_step_coverage=first, config_path="not-read.yaml", run_def_path=Path("not-read.def"),
        runtime=NS(), entry_payloads=(),
    ) is None


@pytest.mark.parametrize("raw_ref,old_reads,old_flag", [(None, 0, False), ("TRUE", 1, True)])
def test_saved_baseline_control_contrasts_actual_old_adapter_with_final(monkeypatch, raw_ref, old_reads, old_flag):
    baseline_path = os.environ.get("ORCJAX_COLD_BASELINE_SOURCE")
    if baseline_path is None:
        pytest.skip("Saved baseline AST controls run through the package run-checks.py")
    baseline = Path(baseline_path).read_text(encoding="utf-8")
    fn = function_tree(baseline)
    read_if = next(n for n in fn.body if isinstance(n, ast.If)
                   and "run_def_values.get('use_refSOC', 'FALSE')" in ast.unparse(n.test))
    closure = next(n for n in ast.walk(fn) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id == "thermosoil_cold_start_coef_closure")
    old_kw = next(kw.value for kw in closure.keywords if kw.arg == "use_refSOC")
    v = values(**({} if raw_ref is None else {"use_refSOC": raw_ref}))
    calls, _, _ = reader(monkeypatch)
    env = {"run_def_values": v, "parse_run_def_bool": o.parse_run_def_bool, "config_path": "case.yaml",
           "step": NS(domain=domain()), "read_paper_thermosoil_refsoc_static_field": o.read_paper_thermosoil_refsoc_static_field}
    exec(compile(ast.Module(body=[read_if], type_ignores=[]), "<saved-old-cold-static-branch>", "exec"), env)
    assert len(calls) == old_reads and expression(old_kw, env) is old_flag
    c = o._fresh_cold_thermosoil_controls(v)
    o._fresh_cold_thermosoil_refsoc_input("case.yaml", controls=c, domain=domain(), npts=1, ngrnd=3)
    assert c.use_refsoc is True and len(calls) == old_reads + 1
    RED_CONTROLS.append({"raw_use_refSOC": raw_ref, "old_static_reads": old_reads,
                         "old_closure_flag": old_flag, "new_static_reads": 1, "new_closure_flag": c.use_refsoc,
                         "old_behavior_contradicts_active_source_default": raw_ref is None,
                         "method": "saved baseline actual static-branch and keyword AST; physical delegates replaced"})
