"""Bounded adjacent-component and shared writer checks, without a model step.

The thermal/snow numerical evidence lives in each lane's original-source
package. These checks execute real components and shared writers to verify
phase-specific values reach the next consumer and the independent driver IO.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import jax
import numpy as np
import pytest

from jax_orchidee.driver import orchestration as driver
from jax_orchidee.driver.dim2 import Dim2DriverRestart
from jax_orchidee.driver.restart_file_io import (
    DRIVER_RESTART_VARIABLES,
    read_dim2_driver_restart,
    write_dim2_driver_restart,
)
from jax_orchidee.sechiba import hydrol, thermosoil
from jax_orchidee.stomate.restart_io import StomateRestartPhysicalState

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "outputs/reviews/main-physics-20261116/thermal-snow-coupling-batch"
REFERENCE = json.loads((ROOT / "tests/fixtures/coupling_peq_nonzero_reference.json").read_text(encoding="utf-8"))


def load_script(name):
    # Existing maintained oracle scripts import their common utilities by name.
    import sys
    script_dir = str(ROOT / "scripts/dev")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/dev" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_case(name="positive-dt2700"):
    return next(case for case in REFERENCE["cases"] if case["name"] == name)


def source_return_payload(case):
    p = case["inputs"]
    return {
        "vevapp": np.asarray(case["outputs"]["vevapp_amount"]),
        "dt_sechiba": p["dt_sechiba"],
        "temp_sol_new": np.asarray(case["outputs"]["temp_sol_new"]),
        "temp_sol_new_pft": np.asarray(p["psold_pft"]) / 1004.675,
        "qsurf": np.full((2,), .006), "evapot": np.zeros(2), "evapot_corr": np.zeros(2),
        "fluxsens": np.asarray([12., 23.]), "zlev": np.asarray([2., 3.]),
        "qair": np.asarray(p["peqBcoef"]), "temp_air": np.asarray([280., 282.]),
        "rau": np.asarray(p["rau"]),
        **{name: np.asarray(p[name]) for name in ("petAcoef", "petBcoef", "peqAcoef", "peqBcoef")},
    }


@pytest.mark.parametrize("runtime", ["eager", "jit"])
def test_source_amount_to_rate_at_actual_shared_writer(runtime):
    case = source_case()
    payload = source_return_payload(case)

    def write(amount):
        fields = driver._empty_previous_step_state_fields()
        driver._add_enerbil_local_to_previous_fields(fields, NS(payload={**payload, "vevapp": amount}))
        return fields["driver_previous_step_state"]["vevapp"]

    kernel = jax.jit(write) if runtime == "jit" else write
    actual = kernel(payload["vevapp"])
    np.testing.assert_allclose(actual, case["outputs"]["vevapp_rate"], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(payload["vevapp"], case["outputs"]["vevapp_amount"])


def test_saved_old_writer_exposes_source_rate_counterexample():
    path = PACKAGE / "baseline/jax_orchidee/driver/orchestration.py"
    old_tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_add_enerbil_local_to_previous_fields")
    namespace = vars(driver).copy()
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    case = source_case()
    payload = source_return_payload(case)
    old_fields = driver._empty_previous_step_state_fields()
    namespace[function.name](old_fields, NS(payload=payload))
    before = np.asarray(old_fields["driver_previous_step_state"]["vevapp"])
    expected = np.asarray(case["outputs"]["vevapp_rate"])
    np.testing.assert_array_equal(before, case["outputs"]["vevapp_amount"])
    assert not np.allclose(before, expected, rtol=1e-12, atol=1e-12)
    print("JOINT_RATE " + json.dumps({"old_writer": before.tolist(), "source_rate": expected.tolist(),
        "dt": payload["dt_sechiba"], "old_status": "failed_expected_rate", "historical_module_executed": False}))


def test_actual_driver_codec_preserves_rate_and_nonzero_tuple(tmp_path):
    case = source_case()
    payload = source_return_payload(case)
    fields = driver._empty_previous_step_state_fields()
    driver._add_enerbil_local_to_previous_fields(fields, NS(payload=payload))
    # The maintained file codec declares the paper single-landpoint schema.
    # Round-trip each independent source point within that supported schema.
    physical = StomateRestartPhysicalState(
        nav_lon=np.asarray([[0.]]), nav_lat=np.asarray([[0.]]), nav_lev=np.asarray([0.]),
        time=np.asarray([0.]), time_steps=np.asarray([payload["dt_sechiba"]]),
    )
    for point in range(2):
        selection = slice(point, point + 1)
        state_values = {name: np.zeros(1) for name in DRIVER_RESTART_VARIABLES}
        state_values.update({name: fields["driver_previous_step_state"][name][selection]
                             for name in state_values if name in fields["driver_previous_step_state"]})
        state = Dim2DriverRestart(**state_values)
        path = tmp_path / f"driver-rate-{point}.nc"
        write_dim2_driver_restart(path, state=state, physical_state=physical)
        restored = read_dim2_driver_restart(path)
        np.testing.assert_allclose(restored.vevapp, case["outputs"]["vevapp_rate"][selection], rtol=1e-12, atol=1e-12)
        for name in ("petAcoef", "petBcoef", "peqAcoef", "peqBcoef"):
            np.testing.assert_array_equal(getattr(restored, name), payload[name][selection])


@pytest.mark.parametrize("case_name", ["explicit_sparse", "explicit_cold", "bucket_thin"])
def test_actual_thermal_carry_is_consumed_by_adjacent_component(case_name, monkeypatch):
    oracle = load_script("oracle_thermosoil_full_component")
    values = oracle.read_npz(PACKAGE / "thermal/attempt02" / case_name / "inputs.npz")
    kwargs = oracle.owner_kwargs(values)
    owner = thermosoil.run_thermosoil_first_step_module(**kwargs, use_jit=False)
    assert owner.ok
    fields = driver._empty_previous_step_state_fields()
    driver._add_thermosoil_to_previous_fields(fields, owner)
    carry = fields["thermosoil_previous_step_state"]
    energy = fields["enerbil_previous_step_state"]
    np.testing.assert_array_equal(carry["pcapa_en"], owner.module.getdiff.pcapa_en)
    np.testing.assert_array_equal(carry["temp_sol_beg"], owner.module.energy.temp_sol_beg)
    np.testing.assert_array_equal(carry["ptn"], owner.module.profile.ptn)
    np.testing.assert_array_equal(carry["stempdiag"], owner.module.profile.stempdiag)
    for name in ("soilcap", "soilflx", "soilcap_pft", "soilflx_pft"):
        np.testing.assert_array_equal(energy[name], owner.downstream_payload[name])
    following = copy.copy(kwargs)
    following["thermosoil_restart"] = NS(**{**vars(kwargs["thermosoil_restart"]), **carry})
    following["enerbil_payload"] = {**kwargs["enerbil_payload"], "soilcap": energy["soilcap"],
        "temp_sol_new": np.asarray(owner.module.energy.temp_sol_beg) + .125}
    delegate_name = "thermosoil_explicit_step" if kwargs["ok_explicitsnow"] else "thermosoil_no_explicit_snow_step"
    delegate = getattr(thermosoil, delegate_name)
    observed_boundaries = []

    def observed(**boundary):
        observed_boundaries.append(boundary)
        for name in ("pcapa_en_previous", "temp_sol_beg"):
            np.testing.assert_array_equal(boundary[name], carry[name.replace("_previous", "")])
        np.testing.assert_array_equal(boundary["soilcap_initial"], energy["soilcap"])
        return delegate(**boundary)

    with monkeypatch.context() as patch:
        patch.setattr(thermosoil, delegate_name, observed)
        if case_name == "explicit_cold":
            # This synthetic second input reaches the existing cross-PFT
            # freeze guard; it does not establish a legal cold trajectory.
            with pytest.raises(ValueError, match="profil_froz differs across PFTs"):
                thermosoil.run_thermosoil_first_step_module(**following, use_jit=False)
            assert len(observed_boundaries) == 1
            print("JOINT_THERMAL " + json.dumps({"case": case_name, "actual_first_component": 1,
                "next_boundary_observed": True, "full_model_steps": 0,
                "status": "carry_passed_second_input_rejected_by_existing_freeze_guard"}))
            return
        next_owner = thermosoil.run_thermosoil_first_step_module(**following, use_jit=False)
    assert len(observed_boundaries) == 1
    assert next_owner.ok
    for name in ("pcapa_en_previous", "temp_sol_beg"):
        np.testing.assert_array_equal(next_owner.boundary_payload[name], carry[name.replace("_previous", "")])
    np.testing.assert_array_equal(next_owner.boundary_payload["soilcap_initial"], energy["soilcap"])
    increment = np.asarray(energy["soilcap"]) * .125
    np.testing.assert_allclose(next_owner.module.energy.surfheat_incr + next_owner.module.energy.coldcont_incr,
                               increment, rtol=1e-12, atol=1e-12)
    print("JOINT_THERMAL " + json.dumps({"case": case_name, "adjacent_components": 2,
        "explicit_snow": bool(values["controls"][0]), "full_model_steps": 0, "status": "passed"}))


def test_actual_snow_output_reaches_thermal_boundary_and_carry():
    snow_oracle = load_script("oracle_snow_maxmass_domain")
    thermal_oracle = load_script("oracle_thermosoil_full_component")
    snow_inputs = snow_oracle.path_inputs_from_reference(PACKAGE / "snow/attempt02", "deep_regular_remap")
    snow_owner = hydrol.hydrol_explicit_snow_step(**snow_inputs)
    snow = snow_owner.snow_state
    values = thermal_oracle.read_npz(PACKAGE / "thermal/attempt02/explicit_sparse/inputs.npz")
    kwargs = thermal_oracle.owner_kwargs(values)
    kwargs.update({name: getattr(snow, name) for name in ("snow", "snowrho", "snowdz", "snowtemp")})
    owner = thermosoil.run_thermosoil_first_step_module(**kwargs, use_jit=False)
    assert owner.ok
    for name in ("snowrho", "snowdz", "snowtemp"):
        np.testing.assert_array_equal(owner.boundary_payload[name], getattr(snow, name))
    fields = driver._empty_previous_step_state_fields()
    driver._add_thermosoil_to_previous_fields(fields, owner)
    np.testing.assert_array_equal(fields["thermosoil_previous_step_state"]["ptn"], owner.module.profile.ptn)
    assert np.all(np.isfinite(np.asarray(owner.module.profile.ptn)))
    print("JOINT_SNOW " + json.dumps({"source_case": "deep_regular_remap", "actual_snow_and_thermal_calls": 2,
        "scope": "array transfer and carry, not independent coupled numeric oracle or natural trajectory", "status": "passed"}))


def test_sechiba_finalize_keeps_internal_water_amount():
    # Reuse the existing complete synthetic boundary owners only to exercise the
    # actual 93-field finalizer. No physical timestep is driven by this test.
    from tests.unit.test_driver_snow_temperature_clamp import _components, _thermal_closure
    closure, payload = _thermal_closure([280., 281.], [[0., 0., 0.], [0., 0., 0.]])
    local, diffuco, hydro, condveg, previous = _components(closure, payload)
    amount = np.asarray([.27, -.54])
    local.payload["vevapp"] = amount
    finalized = driver._sechiba_finalize_state_from_components(
        previous=previous.fields_by_component["sechiba_finalize_state"], enerbil_local=local,
        diffuco_local=diffuco, thermosoil_module=closure, condveg_module=condveg, hydrol_module=hydro,
    )
    np.testing.assert_array_equal(finalized["vevapp"], amount)
    fields = driver._empty_previous_step_state_fields()
    driver._add_enerbil_local_to_previous_fields(fields, local)
    np.testing.assert_allclose(fields["driver_previous_step_state"]["vevapp"], [.00015, -.0003], rtol=1e-12, atol=1e-12)
