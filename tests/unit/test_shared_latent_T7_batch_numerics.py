"""Small affine oracle and fail-closed captured-operand admission checks."""
import copy
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.hpc import probe_shared_latent_T7_batch_numerics as probe
from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import _linear


def test_native_mapped_ad_matches_independent_complete_affine_gradients():
    bundle = probe.validate_operands(probe.synthetic_operands())
    params = {k: jnp.asarray(bundle[k]) for k in ("kernel", "bias")}
    values = tuple(jnp.asarray(bundle[k]) for k in ("x", "direction", "cotangent", "mask"))
    _, batched, mapped, _ = probe.make_routes(jax, _linear)
    oracle = probe.affine_oracle(bundle)
    for route in (batched, mapped):
        result = jax.device_get(jax.jit(route)(params, *values))
        assert probe.compare_results(result, oracle)["passed"]
        assert np.all(result["tangent"][:3] == 0)
        assert np.max(np.abs(result["tangent"][3])) > 0.1
        assert np.count_nonzero(result["primal"][~result["mask"]]) > 0
        directions = probe.independent_direction_checks(bundle, result)
        assert directions["actual_JVP_VJP_comparison"]["passed"]
        assert directions["JVP_oracle_comparison"]["passed"]
        assert directions["VJP_oracle_comparison"]["passed"]


def test_operand_round_trip_and_captured_identity_rejection(tmp_path):
    bundle = probe.synthetic_operands()
    path = tmp_path / "operands.npz"
    probe.save_operands(path, bundle)
    loaded = probe.load_operands(path)
    for key in probe.ARRAY_KEYS:
        np.testing.assert_array_equal(loaded[key], bundle[key])
    with pytest.raises(FileExistsError):
        probe.save_operands(path, bundle)
    captured = copy.deepcopy(bundle)
    captured["metadata"]["origin"] = "captured_component_operands"
    with pytest.raises(ValueError, match="source identities"):
        probe.validate_operands(captured)
    captured["metadata"].update(parameter_file_sha256=probe.PARAMETERS_SHA256,
        prepared_context_sha256=probe.CONTEXT_SHA256, artifact_ids=probe.ARTIFACT_IDS)
    with pytest.raises(ValueError, match="provenance field"):
        probe.validate_operands(captured)


def test_complete_arrays_strict_failure_is_retained_and_input_errors_rejected():
    bundle = probe.synthetic_operands()
    result = probe.affine_oracle(bundle)
    changed = copy.deepcopy(result)
    changed["parameter_gradient"]["kernel"][0, 0] += 1.0e-3
    assert not probe.compare_results(changed, result)["passed"]
    bad = copy.deepcopy(bundle)
    bad["direction"][0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="sample-3"):
        probe.validate_operands(bad)
    bad = copy.deepcopy(bundle)
    bad["mask"] = bad["mask"].astype(np.float32)
    with pytest.raises(ValueError, match="Boolean"):
        probe.validate_operands(bad)
    bad = copy.deepcopy(bundle)
    bad["x"][3, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite FP32"):
        probe.validate_operands(bad)


def test_failed_middle_round_is_preserved_when_the_last_round_passes():
    bundle = probe.synthetic_operands()
    oracle = probe.affine_oracle(bundle)
    rows = [{"route": route, "cycle": cycle, "result": copy.deepcopy(oracle)}
            for cycle in range(3) for route in ("b1", "vmap", "map")]
    rows[4]["result"]["parameter_gradient"]["kernel"][0, 0] += 1.0e-3
    acceptance = probe.evaluate_measurements(bundle, rows)
    assert not acceptance["passed"]
    assert acceptance["measurement_count"] == 9
    assert len(acceptance["within_route_repeat_comparisons"]) == 6
    middle = next(r for r in acceptance["oracle_comparisons"] if r["cycle"] == 1 and r["route"] == "vmap")
    last = next(r for r in acceptance["oracle_comparisons"] if r["cycle"] == 2 and r["route"] == "vmap")
    assert not middle["passed"] and last["passed"]


def test_actual_jvp_and_vjp_ends_are_checked_separately_from_the_oracle():
    bundle = probe.synthetic_operands()
    result = probe.affine_oracle(bundle)
    row, column = np.argwhere(bundle["mask"][3])[0]
    result["tangent"][3, row, column] += 1.0
    checks = probe.independent_direction_checks(bundle, result)
    assert checks["actual_JVP_end"] != checks["actual_VJP_end"]
    assert not checks["actual_JVP_VJP_comparison"]["passed"]
    assert not checks["JVP_oracle_comparison"]["passed"]
    assert checks["VJP_oracle_comparison"]["passed"]


def test_nonfinite_middle_round_saves_raw_arrays_and_serializable_failure(tmp_path, monkeypatch):
    bundle = probe.synthetic_operands()
    oracle = probe.affine_oracle(bundle)
    rows = [{"route": route, "cycle": cycle, "result": copy.deepcopy(oracle)}
            for cycle in range(4) for route in ("b1", "vmap", "map")]
    failed = rows[4]["result"]
    row, column = np.argwhere(bundle["mask"][3])[0]
    failed["primal"][3, 0, 0] = np.nan  # Raw inactive values must still fail.
    failed["tangent"][3, row, column] = np.inf
    failed["input_gradient"][3, 0, 0] = np.nan
    failed["parameter_gradient"]["kernel"][0, 0] = -np.inf
    failed["loss"][...] = np.nan
    result_path = tmp_path / "failed.results.npz"
    report = probe.save_measurement_evidence(result_path, bundle, rows)
    assert report["status"] == "numerical_failed" and not report["passed"]
    assert report["preserved_array_count"] == 91
    middle = next(r for r in report["oracle_comparisons"] if r["cycle"] == 1 and r["route"] == "vmap")
    for key in ("primal", "tangent", "input_gradient", "parameter_gradient.kernel", "loss"):
        comparison = middle["comparisons"][key]
        assert not comparison["passed"] and comparison["actual_nonfinite_coordinates"] == 1
        assert comparison["max_absolute_error"] is None
        assert comparison["max_error_to_allowance"] is None
    direction = next(r for r in report["independent_direction_checks"] if r["cycle"] == 1 and r["route"] == "vmap")
    assert direction["actual_pairing_nonfinite_endpoints"] == 2
    assert direction["actual_JVP_end"] is None and direction["actual_VJP_end"] is None
    assert not direction["actual_JVP_VJP_comparison"]["passed"]
    assert any(not r["passed"] for r in report["route_comparisons"])
    assert any(not r["passed"] for r in report["within_route_repeat_comparisons"])
    assert next(r for r in report["oracle_comparisons"] if r["cycle"] == 3 and r["route"] == "vmap")["passed"]
    report.update(backend="cpu", provenance=bundle["metadata"])
    report_path = tmp_path / "failed.json"
    report_path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "numerical_failed"
    with np.load(result_path, allow_pickle=False) as saved:
        for key, array in probe.result_arrays(failed).items():
            np.testing.assert_array_equal(saved["cycle1.vmap." + key], array)
    overflow = probe.compare_arrays(np.finfo(np.float64).max, -np.finfo(np.float64).max)
    assert not overflow["passed"] and overflow["derived_nonfinite_coordinates"] == 1
    assert overflow["max_absolute_error"] is None
    json.dumps(overflow, allow_nan=False)
    monkeypatch.setattr(probe, "run", lambda args: report)
    monkeypatch.setattr(sys, "argv", ["probe", "--output", str(report_path)])
    assert probe.main() == 3
