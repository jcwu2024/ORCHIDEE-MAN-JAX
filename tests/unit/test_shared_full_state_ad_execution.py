"""CPU-only boundary tests; no real neural model compilation or GPU required."""
from collections import namedtuple
from copy import deepcopy
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from scripts.hpc.shared_full_state_ad_execution import (
    CANDIDATE_ARCHITECTURE, CORRECTED_PERFORMANCE_DIAGNOSTIC,
    FullStateADEvaluator, load_full_state_executable,
    state_shape, validate_candidate_admission, validate_full_state_abi,
    validate_original_admission, validate_corrected_performance_diagnostic,
)

Inputs = namedtuple("Inputs", "continuous_state")
Samples = namedtuple("Samples", "inputs")


def candidate(component="T7"):
    identity = {"candidate_config": {"latent_reader_execution": "field", "decoder_architecture": "latent_reader"},
                "candidate_source_sha256": {"source": "hash"}, "source_reuse_assembler_sha256": "hash",
                "source_reuse_projection_sha256": "hash", "source_head": "head", "input_source_method": {"method": 1}}
    return {"status": "passed", "component": component, "architecture": CANDIDATE_ARCHITECTURE,
            "identity": identity, "native_AD_argnums": [0, 1], "precision": "FP32_neural_FP64_physics",
            "initialization": "fresh_public_whole_tree", "training_updates": 0,
            "parameter_count": 66123713, "parameter_leaves": 355, "executable": "full.pkl",
            "validation": {"gradient_coordinates": 66123713, "gradient_leaves": 355,
                           "gradient_nonzero": 1, "gradient_norm": 1., "hard_counts": {"budget_failure": 0},
                           "initial_state_shape": list(state_shape(component)), "initial_state_finite": True,
                           "initial_state_nonzero": 1}}


@pytest.mark.parametrize("field,value", [("status", "failed"), ("component", "T30"),
    ("native_AD_argnums", [0]), ("parameter_count", 64884481), ("architecture", "original")])
def test_candidate_requires_own_full_admission(field, value):
    report = candidate()
    identity = deepcopy(report["identity"])
    report[field] = value
    with pytest.raises(ValueError):
        validate_candidate_admission(report, "T7", identity)


def test_candidate_rejects_failed_state_and_physical_gates():
    for field, value in (("initial_state_finite", False), ("initial_state_shape", [1]),
                         ("initial_state_nonzero", 0), ("gradient_norm", np.nan),
                         ("hard_counts", {"mask_mismatch": 1})):
        report = candidate()
        report["validation"][field] = value
        with pytest.raises(ValueError):
            validate_candidate_admission(report, "T7", report["identity"])


def test_actual_original_reports_and_parameter_only_rejection():
    root = Path(__file__).resolve().parents[2] / "outputs/diagnostics/shared-latent-reader-20261004/original-admissions"
    names = {"daily": "shared-chunk-vmap-daily-dual-v4-20261001-reference.json",
             "T7": "shared-recurrent-checkpoint-policy-20260930-v2-T7-reference.json",
             "T30": "shared-original-t30-dual-reference-20261001.json"}
    for component, name in names.items():
        report = json.loads((root / name).read_text())
        identity = report["identity"]
        reference = {"source_git_head": identity.get("source_head", identity.get("source_git_head")),
                     "method": identity["method"]}
        validate_original_admission(report, component, reference)
        bad = deepcopy(report)
        bad.pop("parameter_and_initial_state_gradients", None)
        bad["scope"] = "parameters_only"
        with pytest.raises(ValueError):
            validate_original_admission(bad, component, reference)
        wrong = dict(reference, source_git_head="other")
        with pytest.raises(ValueError):
            validate_original_admission(report, component, wrong)
        if component == "daily":
            bad = deepcopy(report)
            bad["variant"] = "candidate_v4"
            with pytest.raises(ValueError):
                validate_original_admission(bad, component, reference)


@pytest.mark.parametrize("component", ["daily", "T7", "T30"])
def test_forwarding_and_complete_state_retention(component):
    parameters = {"weight": np.ones(2, dtype=np.float32)}
    state = np.arange(np.prod(state_shape(component)), dtype=np.float64).reshape(state_shape(component))
    canonical = (Samples(Inputs(state)),) if component == "daily" else (state, np.ones(1), np.ones(1), np.ones(1), np.ones(1))
    seen = []
    gradient = {"weight": np.ones(2, dtype=np.float32)}
    def full(*args):
        seen.append(args)
        return ((np.float64(3), {"hard": {"ok": 0}}), (gradient, state + len(seen)))
    evaluator = FullStateADEvaluator(jax, full, component)
    count = 8 if component == "daily" else 1
    for index in range(count):
        value, actual_gradient = evaluator(parameters, *canonical)
        assert actual_gradient is gradient
        assert value[0] == 3
        assert seen[-1][0] is parameters and seen[-1][1] is state
        assert seen[-1][2:] == canonical if component == "daily" else seen[-1][2:] == canonical[1:]
    with pytest.raises(ValueError, match="consumed/reset"):
        evaluator(parameters, *canonical)
    records = evaluator.take_records()
    assert len(records) == count
    for index, record in enumerate(records):
        np.testing.assert_array_equal(record["initial_state_gradient"], state + index + 1)
    evaluator(parameters, *canonical)
    evaluator.reset()
    assert evaluator.take_records() == ()


@pytest.mark.parametrize("bad", ["parameter_only", "wrong_shape", "wrong_dtype", "nonfinite"])
def test_rejects_incomplete_or_invalid_state_result(bad):
    p = {"weight": np.ones(2, dtype=np.float32)}
    state = np.ones(3854, dtype=np.float64)
    gradient = np.ones_like(state)
    if bad == "wrong_shape": gradient = gradient[:1]
    if bad == "wrong_dtype": gradient = gradient.astype(np.float32)
    if bad == "nonfinite": gradient[0] = np.nan
    result = ((1., {}), p if bad == "parameter_only" else (p, gradient))
    evaluator = FullStateADEvaluator(jax, lambda *args: result, "T7")
    with pytest.raises(ValueError):
        evaluator(p, state, state, state, state, state)
    assert evaluator.take_records() == ()


def abi_fixture():
    arguments = ({"weight": np.ones(2, np.float32)}, np.ones(3854, np.float64),
                 np.ones(1), np.ones(1), np.ones(1), np.ones(1))
    info = jax.tree_util.tree_map(lambda value: SimpleNamespace(_aval=SimpleNamespace(shape=value.shape, dtype=value.dtype)), (arguments, {}))
    executable = SimpleNamespace(args_info=info)
    return arguments, executable, (b"binary", jax.tree_util.tree_structure((arguments, {})), "output_tree")


def test_native_cpu_ad_keeps_full_state_derivative_in_canonical_view():
    import jax.numpy as jnp
    parameters = {"weight": jnp.asarray([2.], dtype=jnp.float32)}
    state = jnp.arange(3854, dtype=jnp.float32) / 3854
    def objective(p, initial, a, b, c, d):
        return jnp.sum(initial ** 2) * p["weight"][0], {"hard": {"ok": jnp.asarray(0)}}
    full = jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)
    evaluator = FullStateADEvaluator(jax, full, "T7")
    (value, _), gradient = evaluator(parameters, state, state, state, state, state)
    np.testing.assert_allclose(gradient["weight"], np.sum(np.asarray(state) ** 2), rtol=1e-6)
    np.testing.assert_allclose(evaluator.take_records()[0]["initial_state_gradient"], 4 * np.asarray(state))
    assert np.isfinite(float(value))


def test_explicit_loader_checks_sidecar_component_identity_abi_and_memory(tmp_path):
    args, exe, serialized = abi_fixture()
    report = candidate()
    path = tmp_path / "full.pkl"
    path.write_bytes(pickle.dumps(serialized))
    sidecar = path.with_suffix(".pkl.identity.json")
    sidecar.write_text(json.dumps({"identity": report["identity"], "component": "T7"}))
    calls = []
    def deserialize(*saved):
        calls.append(saved)
        return exe
    kwargs = dict(component="T7", admission=report, architecture=CANDIDATE_ARCHITECTURE,
                  expected_identity=report["identity"], deserialize=deserialize,
                  memory_sizes=lambda executable: {"estimated_live_bytes": 100})
    actual, meta = load_full_state_executable(jax, path, args, **kwargs)
    assert actual is exe and meta["model_compiles"] == 0 and len(calls) == 1
    for payload in ({"identity": {}, "component": "T7"}, {"identity": report["identity"], "component": "T30"}):
        sidecar.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="sidecar"):
            load_full_state_executable(jax, path, args, **kwargs)
    sidecar.write_text(json.dumps({"identity": report["identity"], "component": "T7"}))
    with pytest.raises(MemoryError):
        load_full_state_executable(jax, path, args, **dict(kwargs, memory_sizes=lambda _: {"estimated_live_bytes": 25*2**30}))
    bad_args = (args[0], args[1].astype(np.float32), *args[2:])
    with pytest.raises(ValueError, match="shape/dtype"):
        validate_full_state_abi(jax, serialized, exe, bad_args, component="T7")
    with pytest.raises(ValueError, match="treedef"):
        validate_full_state_abi(jax, (b"", jax.tree_util.tree_structure(args), ""), exe, args, component="T7")


def corrected_report(component="T7", architecture="candidate"):
    candidate_id = candidate(component)["identity"]
    method = {"daily_protocol": {"seed": 3}}
    expected_architecture = CANDIDATE_ARCHITECTURE if architecture == "candidate" else "original"
    identity = {
        "source_git_head": "head", "horizon": int(component[1:]),
        "turnover_sha256": "turnover", "native_AD_argnums": [0, 1],
        "training_updates": 0, "precision": "FP32_neural_FP64_physics",
        "architecture": expected_architecture,
        "parameter_count": 66123713 if architecture == "candidate" else 64884481,
        "parameter_leaves": 355 if architecture == "candidate" else 300,
        "method": method, "parameter_provenance": {"identity": candidate_id},
    }
    validation = {
        "gradient_coordinates": identity["parameter_count"],
        "gradient_leaves": identity["parameter_leaves"],
        "gradient_nonzero": 1, "gradient_norm": 1., "hard_counts": {"budget_failure": 0},
        "initial_state_shape": list(state_shape(component)),
        "initial_state_finite": True, "initial_state_nonzero": 1,
    }
    repeat = {
        "all_parameter_gradients": {"passed": True},
        "forward_and_gates": {"passed": True},
        "all_initial_state_gradients": {"passed": False, "failed": 1},
    }
    return {
        "status": "numerical_failed", "component": component,
        "native_AD_argnums": [0, 1], "training_updates": 0,
        "precision": "FP32_neural_FP64_physics", "scientific_acceptance": False,
        "model_compiles": {expected_architecture: 0},
        "identities": {expected_architecture: identity},
        "cold_measurements": {expected_architecture: {"executable": "full.pkl"}},
        "results": [
            {"architecture": expected_architecture, "label": "cold-first", "validation": validation},
            {"architecture": expected_architecture, "label": "pair1", "validation": validation,
             "same_architecture_repeat": repeat},
        ],
        "numerical_failures": [{"scope": "initial_state_repeat", "failed": 1}],
    }, candidate_id, method, identity


@pytest.mark.parametrize("architecture", ["original", "candidate"])
def test_corrected_performance_view_preserves_failed_status_and_limits_failure_scope(architecture):
    name = "candidate" if architecture == "candidate" else "original"
    report, candidate_id, method, identity = corrected_report(architecture=name)
    expected_reference = {"source_git_head": "head", "method": method}
    view = validate_corrected_performance_diagnostic(
        report, "T7", identity["architecture"],
        expected_identity=candidate_id if name == "candidate" else None,
        expected_reference=expected_reference, expected_turnover_sha256="turnover",
        expected_executable="full.pkl")
    assert view["report_status"] == report["status"] == "numerical_failed"
    assert view["scientific_acceptance"] is False
    assert view["state_repeat_failure_records"]
    report["results"][1]["same_architecture_repeat"]["forward_and_gates"]["passed"] = False
    with pytest.raises(ValueError, match="parameter/forward/loss"):
        validate_corrected_performance_diagnostic(
            report, "T7", identity["architecture"],
            expected_identity=candidate_id if name == "candidate" else None,
            expected_reference=expected_reference, expected_turnover_sha256="turnover",
            expected_executable="full.pkl")
    with pytest.raises(ValueError, match="admission"):
        validate_candidate_admission(report, "T7", candidate_id)


def test_corrected_loader_binds_saved_report_path_and_sidecar(tmp_path):
    args, exe, serialized = abi_fixture()
    report, candidate_id, method, identity = corrected_report()
    path = tmp_path / "full.pkl"
    path.write_bytes(pickle.dumps(serialized))
    path.with_suffix(".pkl.identity.json").write_text(
        json.dumps({"identity": identity, "component": "T7"}))
    report["cold_measurements"][CANDIDATE_ARCHITECTURE]["executable"] = str(path)
    kwargs = dict(component="T7", admission=report, architecture=CANDIDATE_ARCHITECTURE,
        expected_identity=candidate_id, expected_reference={"source_git_head": "head", "method": method},
        admission_mode=CORRECTED_PERFORMANCE_DIAGNOSTIC, expected_turnover_sha256="turnover",
        deserialize=lambda *_: exe, memory_sizes=lambda _: {"estimated_live_bytes": 100})
    loaded, metadata = load_full_state_executable(jax, path, args, **kwargs)
    assert loaded is exe
    assert metadata["report_status"] == "numerical_failed"
    assert metadata["scientific_acceptance"] is False
    report["cold_measurements"][CANDIDATE_ARCHITECTURE]["executable"] = "other.pkl"
    with pytest.raises(ValueError, match="saved executable path"):
        load_full_state_executable(jax, path, args, **kwargs)
