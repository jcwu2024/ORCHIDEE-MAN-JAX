"""Finite PB-03 HYDROL gates, including individually visible strict residuals.

All numerical inputs and original-Fortran expectations are versioned. Only the
separate source-identity test needs the ignored original source checkout.
"""

import inspect
import json
from functools import lru_cache

import jax
import numpy as np
import pytest

from scripts.dev import oracle_physics_branch_hydrol as oracle

REFERENCE = oracle.load_fixture()
CASES = {case["name"]: case for case in REFERENCE["cases"]}
VALID = [name for name, case in CASES.items() if str(case["inputs"]["case_id"]) in REFERENCE["expected"]]
MODES = ("eager", "jit")
RESIDUALS = REFERENCE.get("strict_residuals", {})
RESIDUAL_COORDINATES = [
    pytest.param(
        mode,
        name,
        field,
        index,
        marks=pytest.mark.xfail(
            strict=True, reason="Recorded strict original-Fortran residual; acceptance remains false"
        ),
    )
    for mode, cases in RESIDUALS.items()
    for name, fields in cases.items()
    for field, record in fields.items()
    for index in record["flat_indices"]
]


@lru_cache(maxsize=None)
def actual_case(mode, name):
    assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64
    if mode == "jit":
        assert not jax.config.jax_disable_jit
    closure = oracle.run_production(CASES[name]["inputs"], mode)
    assert closure.ok, closure.missing_inputs
    return closure, oracle.flatten_result(closure)


def test_declared_inputs_and_reference_partition():
    normalized = json.loads(json.dumps(oracle.make_cases(), default=lambda value: value.tolist()))
    assert normalized == REFERENCE["cases"]
    assert len(CASES) == 22 and len(VALID) == 19
    assert REFERENCE["source_failed_cases"] == [118, 121, 122]
    assert {int(key) for key in REFERENCE["expected"]}.isdisjoint(REFERENCE["source_failed_cases"])
    assert REFERENCE["dimensions"] == {"npts": 1, "nvm": 14, "nstm": 6, "nslm": 11}
    assert REFERENCE["rtol"] == oracle.RTOL and REFERENCE["atol"] == oracle.ATOL
    assert set(RESIDUALS) == set(MODES)
    assert all(CASES[name]["inputs"]["check_cwrr"] is False for name in VALID)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", VALID)
def test_all_other_coordinates_match_original_fortran(mode, name):
    _, actual = actual_case(mode, name)
    expected = REFERENCE["expected"][str(CASES[name]["inputs"]["case_id"])]
    residuals = RESIDUALS.get(mode, {}).get(name, {})
    for field, values in expected.items():
        wanted = np.asarray(values)
        got = np.asarray(actual[field]).ravel(order="C")
        assert got.shape == wanted.shape, field
        ordinary = np.ones(wanted.shape, dtype=bool)
        ordinary[residuals.get(field, {}).get("flat_indices", [])] = False
        np.testing.assert_allclose(
            got[ordinary], wanted[ordinary], rtol=oracle.RTOL, atol=oracle.ATOL, err_msg=f"{mode}/{name}/{field}"
        )


@pytest.mark.parametrize("mode,name,field,index", RESIDUAL_COORDINATES)
def test_each_unresolved_coordinate(mode, name, field, index):
    _, actual = actual_case(mode, name)
    expected = REFERENCE["expected"][str(CASES[name]["inputs"]["case_id"])][field][index]
    got = np.asarray(actual[field]).ravel(order="C")[index]
    np.testing.assert_allclose(got, expected, rtol=oracle.RTOL, atol=oracle.ATOL)


def test_false_flux_gate_and_explicit_tide_forcing():
    closure, _ = actual_case("eager", "tide_falling")
    assert closure.module_inputs["check_cwrr"] is False
    assert closure.module_inputs["wtp_tide"] == 0.24
    assert closure.module_inputs["dwtp_tide"] == -0.045
    np.testing.assert_array_equal(closure.outputs.wat_flux, np.zeros((1, 11, 6)))
    assert inspect.signature(oracle.h.hydrol_module_explicit_step).parameters["check_cwrr"].default is False
    assert inspect.signature(oracle.h.hydrol_soil_all_tiles_explicit_step).parameters["check_cwrr"].default is True


def test_tide_missing_forcing_refuses_execution():
    closure, _ = actual_case("eager", "tide_falling")
    payload = dict(closure.module_inputs)
    payload.pop("wtp_tide")
    payload.pop("dwtp_tide")
    # The returned closure lists selected module fields, while this strict
    # wrapper also needs the original zero-snow precall state.
    payload.update(
        precip_snow=np.zeros(1),
        vevapsno=np.zeros(1),
        snow=np.zeros(1),
        snowdz=np.zeros((1, 3)),
        snowrho=np.full((1, 3), 100.0),
        snowtemp=np.full((1, 3), 273.15),
        snow_age=np.zeros(1),
        snow_nobio=np.zeros((1, 1)),
        snow_nobio_age=np.zeros((1, 1)),
        frac_nobio=np.zeros((1, 1)),
        totfrac_nobio=np.zeros(1),
    )
    precall = oracle.h.HydrolFirstStepPrecallAssembly(payload, (), None, (), (), (), (), (), (), ())
    result = oracle.h.run_hydrol_first_step_module_from_precall(
        precall,
        humcste=np.ones(14),
        nroot_state=closure.diagnostics.nroot,
        dz_mm=payload["dz_mm"],
        dh_mm=payload["dh_mm"],
        zz_mm=payload["zz_mm"],
        reinf_slope=np.array([0.17]),
        tides=True,
        runtime_static_tables=oracle.h.HydrolRuntimeStaticTables(
            payload["mineral_tables"], payload["peat_tables"], payload["soil_parameters"]
        ),
    )
    assert not result.ok and result.module is None
    assert {"wtp_tide", "dwtp_tide"} <= set(result.missing_inputs)


def test_column_integral_preserves_one_dimensional_api():
    one = np.array([0.2, 0.25, 0.3])
    dz = np.array([2.0, 2.0, 4.0])
    np.testing.assert_array_equal(
        oracle.h.hydrol_total_moisture_content(one, dz), oracle.h.hydrol_total_moisture_content(one[None, :], dz)
    )
    with pytest.raises(ValueError, match="at least two"):
        oracle.h.hydrol_total_moisture_content(np.ones(1), np.ones(1))
    with pytest.raises(ValueError, match="length must match"):
        oracle.h.hydrol_total_moisture_content(one, np.ones(2))


def test_original_source_identity_when_available(tmp_path):
    identity = REFERENCE["reference_identity"]
    needed = [
        oracle.SOURCE,
        oracle.ROOT / identity["coupling"]["source"],
        oracle.ROOT / identity["harness_constant_sources"]["source"],
        oracle.ROOT / identity["check_cwrr_control"]["source"],
    ]
    if not all(path.is_file() for path in needed):
        pytest.skip("Ignored original source checkout is unavailable; numerical gates still run")
    assert oracle.compose_reference(tmp_path / "reference.f90") == identity
