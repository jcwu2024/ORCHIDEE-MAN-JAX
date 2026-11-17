"""Independent post-profile Fortran coefficient regression at actual callers.

The fixture holds original-Fortran producer operands and outputs. Evaluating
the production call expression retains its real weight selection and kernel,
without recompiling the already validated complete thermal graph in unit tests.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from jax_orchidee.sechiba import thermosoil as thermal

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/unit/fixtures/thermosoil_full_component_coef.json"


def fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def decode(record):
    if "columns" in record:
        result = np.asarray(record["columns"])[record["column_indices"]].T[None, :, :]
    else:
        result = np.asarray(record["dense"])
    assert list(result.shape) == record["shape"]
    return result


def actual_coefficient_call(case, *, preserve_old_raw_weights=False):
    source = Path(thermal.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    caller = "thermosoil_explicit_step" if case["explicit_snow"] else "thermosoil_no_explicit_snow_step"
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == caller)
    call = next(
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id in ("thermosoil_coef_explicit_snow", "thermosoil_coef_no_explicit_snow")
    )
    operands = {name: decode(value) for name, value in case["operands"].items()}
    namespace = vars(thermal).copy()
    namespace.update(
        operands,
        profile=NS(ptn=operands["ptn"]),
        precoef_final=NS(ptn_pftmean=operands["ptn_pftmean"]),
        getdiff=NS(**operands),
        dt_sechiba=case["dt_sechiba"],
        lambda_thermal=case["lambda_thermal"],
    )
    if preserve_old_raw_weights:
        # Preserved baseline operand counterexample; expected values are still
        # the independent Fortran outputs, never duplicated coefficient logic.
        next(k for k in call.keywords if k.arg == "veget_max").value = ast.Name(id="veget_max", ctx=ast.Load())
        ast.fix_missing_locations(call)
    return eval(compile(ast.Expression(call), str(thermal.__file__), "eval"), namespace)


@pytest.mark.parametrize("case_index", [0, 1])
def test_actual_final_coefficient_call_matches_original_fortran(case_index):
    record = fixture()
    case = record["cases"][case_index]
    result = actual_coefficient_call(case)
    for name, expected in case["expected"].items():
        np.testing.assert_allclose(
            getattr(result, name), expected, rtol=record["rtol"], atol=record["atol"], err_msg=name
        )


@pytest.mark.parametrize("case_index", [0, 1])
def test_pre_review_raw_coefficient_weight_counterexample_is_retained(case_index):
    record = fixture()
    case = record["cases"][case_index]
    result = actual_coefficient_call(case, preserve_old_raw_weights=True)
    for name, expected in case["baseline"].items():
        np.testing.assert_allclose(
            getattr(result, name), expected, rtol=record["rtol"], atol=record["atol"], err_msg=name
        )
    assert not np.allclose(result.soilcap, case["expected"]["soilcap"], rtol=record["rtol"], atol=record["atol"])


def test_reference_is_bound_to_original_fortran_final_coefficient_call():
    record = fixture()
    source = ROOT / record["source_file"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record["source_sha256"]
    lines = source.read_text(encoding="utf-8").splitlines()
    start, end = record["source_coef_call_lines"]
    call = "\n".join(lines[start - 1 : end])
    assert "CALL thermosoil_coef" in call
    assert "organic_layer_thick, soilc_total, veget_max_bg, snowdz" in call
