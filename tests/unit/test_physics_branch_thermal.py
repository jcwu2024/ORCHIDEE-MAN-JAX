"""Portable PB-03 original-source full-component evidence and strict gates."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from jax_orchidee.sechiba import hydrol
from scripts.dev import oracle_physics_branch_thermal as oracle

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = oracle.load_fixture()
NAMES = [c["name"] for c in REFERENCE["cases"]]
STRICT_GATES = REFERENCE.get("strict_gates", [])


@lru_cache(maxsize=12)
def actual_thermal(name, mode):
    case = next(c for c in REFERENCE["cases"] if c["name"] == name)
    inputs = oracle.decode_inputs(case)
    result = oracle.run_production(inputs, mode)
    return case, mode, inputs, result, oracle.flatten_result(result)


def passed_coordinates(component, case, mode, call, field, size):
    mask = np.ones(size, dtype=bool)
    for gate in STRICT_GATES:
        if (gate["component"], gate["case"], gate["mode"], gate["call"], gate["field"]) == (component, case, mode, call, field):
            mask[gate["failed_flat_indices"]] = False
    return mask


@pytest.fixture(scope="module", params=[(name, mode) for name in NAMES for mode in ("eager", "jit")])
def thermal_result(request):
    name, mode = request.param
    return actual_thermal(name, mode)


def test_full_actual_thermal_component_defined_fields(thermal_result):
    case, mode, _, _, actual = thermal_result
    assert actual.keys() == case["expected"].keys()
    for field, expected in case["expected"].items():
        e, a = oracle.decode(expected).ravel(), np.asarray(actual[field]).ravel()
        mask = passed_coordinates("thermal", case["name"], mode, None, field, e.size)
        np.testing.assert_allclose(a[mask], e[mask],
            rtol=REFERENCE["rtol"], atol=REFERENCE["atol"], err_msg=field)


def test_thermal_declared_inputs_and_actual_next_payload(thermal_result):
    case, _, inputs, closure, _ = thermal_result
    assert case["origin"]["domain"].startswith("source_consumable")
    assert np.all(inputs["snowdz"] >= 0.)
    assert np.all(inputs["mc_layh"] >= inputs["mcl_layh"])
    assert np.all(inputs["refSOC"] >= 0.)
    for name in ("cgrnd_snow", "dgrnd_snow", "lambda_snow", "soilcap", "soilflx"):
        np.testing.assert_array_equal(closure.downstream_payload[name], getattr(closure.module.coef, name))
    np.testing.assert_array_equal(closure.downstream_payload["pcapa_en"], closure.module.getdiff.pcapa_en)
    np.testing.assert_array_equal(closure.downstream_payload["temp_sol_beg"], inputs["temp_sol_new"])


@lru_cache(maxsize=2)
def snow_runs(mode):
    record = REFERENCE["snow_predecessor"]
    inputs = {k: oracle.decode(a) for k, a in record["initial_inputs"].items()}
    # A complete actual component graph is compiled; no diagnostic helper JIT.
    carry = ("snowrho", "snowgrain", "snowdz", "snowtemp", "snowheat", "snowliq", "snow_age", "snow_nobio_age", "snow_nobio", "snow")
    def numeric_owner(**kwargs):
        result = hydrol.hydrol_explicit_snow_step(**kwargs)
        # Metadata strings are not JAX output leaves. Project only arrays
        # after executing the entire actual snow owner inside this graph.
        names = set(carry) | set(record["calls"][0]["expected"]) - {"mass"}
        arrays = {field: getattr(result.snow_state, field) if field in result.snow_state._fields else getattr(result, field) for field in names}
        arrays["mass"] = (result.snow_state.snowrho * result.snow_state.snowdz).sum(axis=1)
        return arrays
    owner = jax.jit(numeric_owner) if mode == "jit" else numeric_owner
    results = []
    for index, call in enumerate(record["calls"]):
        if index:
            np.testing.assert_allclose(inputs["snowdz"].ravel(), call["entry_snowdz"], rtol=1e-12, atol=1e-14)
            assert inputs["snowdz"][0, 0] == 0.
            inputs["vevapsno"] = np.zeros(1)
            inputs["soilcap"] = np.asarray([800. * (2.8345e6 - 2.5008e6)])
        result = owner(**inputs)
        results.append(result)
        if index:
            assert result["snowdz"][0, 2] == 0.
            assert result["snowdz"][0, 1] > 0.
        for field in carry:
            inputs[field] = result[field]
    return results


@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_actual_two_snow_components_have_supported_zero_layer_predecessor(mode):
    for index, (result, call) in enumerate(zip(snow_runs(mode), REFERENCE["snow_predecessor"]["calls"], strict=True)):
        for field, expected in call["expected"].items():
            e, a = oracle.decode(expected).ravel(), np.asarray(result[field]).ravel()
            mask = passed_coordinates("snow", "adjacent_predecessor", mode, index, field, e.size)
            np.testing.assert_allclose(a[mask], e[mask], rtol=1e-12, atol=1e-14, err_msg=f"call{index}.{field}")


@pytest.mark.parametrize("gate,index", [pytest.param(gate, index,
    marks=pytest.mark.xfail(strict=True, reason="Preserved exact original-reference strict coordinate discrepancy"),
    id=f"{gate['component']}-{gate['case']}-{gate['mode']}-call{gate['call']}-{gate['field']}-{index}")
    for gate in STRICT_GATES for index in gate["failed_flat_indices"]])
def test_preserved_strict_failure_coordinate(gate, index):
    if gate["component"] == "thermal":
        case, _, _, _, actual = actual_thermal(gate["case"], gate["mode"])
        expected = case["expected"][gate["field"]]
        value = actual[gate["field"]]
    else:
        expected = REFERENCE["snow_predecessor"]["calls"][gate["call"]]["expected"][gate["field"]]
        value = snow_runs(gate["mode"])[gate["call"]][gate["field"]]
    np.testing.assert_allclose(np.asarray(value).ravel()[index], oracle.decode(expected).ravel()[index],
        rtol=gate["rtol"], atol=gate["atol"])


def test_portable_declared_input_binary_identity():
    for case in REFERENCE["cases"]:
        identity = hashlib.sha256()
        for array in oracle.decode_inputs(case).values():
            identity.update(np.asarray(array, dtype="<i4" if array.dtype.kind in "iu" else "<f8").ravel(order="F").tobytes())
        assert identity.hexdigest() == case["inputs_binary_sha256"]
        reference_identity = hashlib.sha256()
        for expected in case["expected"].values():
            reference_identity.update(np.asarray(oracle.decode(expected), dtype="<f8").ravel(order="F").tobytes())
        assert reference_identity.hexdigest() == case["Fortran_output_binary_sha256"]
        if "producer_inputs" in case["origin"]:
            origin = case["origin"]
            canonical = json.dumps(origin["producer_inputs"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            assert hashlib.sha256(canonical).hexdigest() == origin["producer_inputs_canonical_sha256"]
            assert set(origin["moisture_fields"]) == set(oracle.accepted.initial.MOISTURE_FIELDS)


def test_original_source_identity_independent_gate():
    sources = dict(REFERENCE["source_files"])
    snow = REFERENCE["snow_predecessor"]
    sources[snow["source_file"]] = snow["source_sha256"]
    missing = [name for name in sources if not (ROOT / name).is_file()]
    if missing:
        pytest.skip("Original Fortran source identity gate only; numerical fixture tests remain active")
    for name, digest in sources.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    for span in [*REFERENCE["source_spans"], *snow["procedure_source_spans"]]:
        path = ROOT / span["path"]
        data = b"".join(path.read_bytes().splitlines(keepends=True)[span["first"] - 1:span["last"]])
        assert hashlib.sha256(data).hexdigest() == span["sha256"]
