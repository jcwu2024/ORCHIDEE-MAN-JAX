"""Native first-consumer counterexample and bounded lifecycle seams.

The numerical fixture comes from the original Fortran micro-driver, not a
Python reimplementation. AST execution here checks actual production call
operands and the year-start thermal block, without advancing a model step.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from jax_orchidee.sechiba import thermosoil as thermal

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/unit/fixtures/thermosoil_energy_first_consumer.json"


def fixture():
    return json.loads(FIXTURE.read_text())


def energy_operands(caller: str):
    record = fixture()
    tree = ast.parse(Path(thermal.__file__).read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == caller)
    call = next(n for n in ast.walk(function) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "thermosoil_energy_diagnostics")
    capacity = np.asarray(record["pcapa_en_level1"])[:, None, :]
    namespace = vars(thermal).copy()
    namespace.update(temp_sol_new=np.asarray(record["temp_sol_new"]),
        temp_sol_beg=np.asarray(record["temp_sol_beg"]), soilcap_initial=np.asarray(record["soilcap"]),
        energy_pcapa_en=capacity, pcapa_en_previous=capacity, getdiff=SimpleNamespace(pcapa_en=capacity),
        veget_max=np.asarray(record["veget_max"]), profile=SimpleNamespace(ptn=np.full((1, 1, 14), 280.)),
        thermosoil_energy_diagnostics=lambda **kwargs: kwargs)
    return eval(compile(ast.Expression(call), str(thermal.__file__), "eval"), namespace)


@pytest.mark.parametrize("caller", ["thermosoil_explicit_step", "thermosoil_no_explicit_snow_step"])
@pytest.mark.parametrize("use_jit", [False, True])
def test_actual_production_energy_call_matches_original_fortran(caller, use_jit):
    record = fixture()
    operands = energy_operands(caller)
    kernel = jax.jit(thermal.thermosoil_energy_diagnostics) if use_jit else thermal.thermosoil_energy_diagnostics
    actual = kernel(**operands)
    np.testing.assert_allclose(actual.surfheat_incr, record["expected_surfheat_incr"], rtol=1.e-12, atol=1.e-12)
    np.testing.assert_allclose(actual.coldcont_incr, record["expected_coldcont_incr"], rtol=1.e-12, atol=1.e-12)
    np.testing.assert_array_equal(actual.temp_sol_beg, operands["temp_sol_new"])
    np.testing.assert_array_equal(actual.ptn_beg, operands["ptn"])


def test_saved_raw_weight_counterexample_remains_reproducible():
    record = fixture()
    operands = energy_operands("thermosoil_explicit_step")
    operands["veget_max"] = np.asarray(record["veget_max"])
    actual = thermal.thermosoil_energy_diagnostics(**operands)
    np.testing.assert_array_equal(actual.surfheat_incr, record["before_surfheat_incr"])
    np.testing.assert_array_equal(actual.coldcont_incr, record["before_coldcont_incr"])
    assert not np.array_equal(actual.surfheat_incr, record["expected_surfheat_incr"])


def test_counterexample_is_bound_to_current_original_fortran():
    record = fixture()
    assert hashlib.sha256((ROOT / record["source_file"]).read_bytes()).hexdigest() == record["source_sha256"]
    lines = (ROOT / record["source_file"]).read_text(encoding="utf-8").splitlines()
    assert "soilcap,  veget_max_bg" in lines[912]


@pytest.mark.parametrize("run_mode,supported,initializes", [
    ("new_native_run", True, True), ("new_native_run", False, False),
    ("continuous", True, False),
])
def test_actual_year_start_thermal_block_obeys_source_lifecycle(run_mode, supported, initializes):
    from jax_orchidee.driver import orchestration as orch
    tree = ast.parse(Path(orch.__file__).read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "paper_1961_driver_restart_year_start_day_result")
    block = next(n for n in function.body if isinstance(n, ast.If)
                 and "restart_thermosoil_initialization_supported" in ast.unparse(n.test))
    old_capacity, old_temperature, old_hydro = object(), object(), object()
    new_capacity, new_temperature, new_diagnostic = object(), object(), object()
    source, altmax, payload, context = object(), object(), object(), object()
    fields = {"sechiba_finalize_state": source,
        "slowproc_stomate_previous_step_state": {"altmax": altmax},
        "thermosoil_previous_step_state": {"pcapa_en": old_capacity, "temp_sol_beg": old_temperature},
        "hydrol_previous_step_state": {"temp_hydro": old_hydro}}
    calls = []

    def initialize(actual_source, **kwargs):
        calls.append((actual_source, kwargs))
        return {"pcapa_en": new_capacity, "temp_sol_beg": new_temperature, "stempdiag": new_diagnostic}

    namespace = dict(run_mode=run_mode, context=context, payload=payload,
        rebased=SimpleNamespace(fields_by_component=fields),
        restart_thermosoil_initialization_supported=lambda actual: supported,
        restart_thermosoil_initial_fields=initialize)
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(orch.__file__), "exec"), namespace)
    if initializes:
        assert calls == [(source, {"context": context, "payload": payload, "altmax": altmax})]
        assert fields["thermosoil_previous_step_state"]["pcapa_en"] is new_capacity
        assert fields["thermosoil_previous_step_state"]["temp_sol_beg"] is new_temperature
        assert fields["thermosoil_previous_step_state"]["stempdiag"] is new_diagnostic
        assert "temp_hydro" not in fields["hydrol_previous_step_state"]
    else:
        assert calls == []
        assert fields["thermosoil_previous_step_state"]["pcapa_en"] is old_capacity
        assert fields["thermosoil_previous_step_state"]["temp_sol_beg"] is old_temperature
        assert fields["hydrol_previous_step_state"]["temp_hydro"] is old_hydro
