"""Small CPU AD/AdamW and real private-pipe checks for the full-state screen."""
from collections import namedtuple
from copy import deepcopy
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.hpc.probe_shared_latent_dual import (
    MANIFEST_SCHEMA, SerialFullStateExecutor, attach_states, compare_complete,
    parser, read_manifest, stateful_branch, validate_args, validate_bundle, worker_class,
    require_memory, forbid_model_compile, validate_manifest_asset_readiness,
    changed_weight_timing_summary,
)
from scripts.hpc.probe_shared_latent_dual_worker import evaluate_command
from scripts.hpc.shared_full_state_ad_execution import FullStateADEvaluator
from scripts.hpc.probe_shared_serialized_dual import RemoteDualExecutor
from scripts.hpc.probe_shared_dual import make_microbatch_daily_branch, make_recurrent_branch, make_component_combiner
from scripts.hpc.run_shared_learning import make_tree_operations
from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
from research.daily_coarse_graining.daily_operator.target_oriented_training import initialize_target_oriented_adamw

Inputs = namedtuple("Inputs", "continuous_state")
Samples = namedtuple("Samples", "inputs")
Case = namedtuple("Case", "initial inputs days targets extra")


def corrected_report(component, architecture, method=None):
    from scripts.hpc.shared_full_state_ad_execution import CANDIDATE_ARCHITECTURE

    method = method or {"daily_protocol": {"seed": 3}}
    candidate_identity = {"seed": 41, "architecture": CANDIDATE_ARCHITECTURE}
    architecture_key = CANDIDATE_ARCHITECTURE if architecture == "candidate" else "original"
    identity = {
        "source_git_head": "head", "turnover_sha256": "turnover",
        "native_AD_argnums": [0, 1], "training_updates": 0,
        "precision": "FP32_neural_FP64_physics", "architecture": architecture_key,
        "horizon": int(component[1:]), "parameter_count": 66123713 if architecture == "candidate" else 64884481,
        "parameter_leaves": 355 if architecture == "candidate" else 300,
    }
    if architecture == "candidate":
        identity["parameter_provenance"] = {"identity": candidate_identity}
    else:
        identity["method"] = method
    validation = {
        "gradient_coordinates": identity["parameter_count"],
        "gradient_leaves": identity["parameter_leaves"],
        "gradient_nonzero": 1, "gradient_norm": 1.0,
        "initial_state_shape": [3854], "initial_state_finite": True,
        "initial_state_nonzero": 1, "hard_counts": {"physical": 0},
    }
    repeat = {
        "all_parameter_gradients": {"passed": True},
        "forward_and_gates": {"passed": True}, "loss_close": True,
        "all_initial_state_gradients": {"passed": False},
    }
    rows = [
        {"architecture": architecture_key, "label": "cold-first", "validation": validation},
        {"architecture": architecture_key, "label": "warm-repeat", "validation": validation,
         "same_architecture_repeat": repeat},
    ]
    return ({"status": "numerical_failed", "component": component,
        "native_AD_argnums": [0, 1], "training_updates": 0,
        "precision": "FP32_neural_FP64_physics", "scientific_acceptance": False,
        "model_compiles": {architecture_key: 0}, "identities": {architecture_key: identity},
        "cold_measurements": {architecture_key: {"executable": None}}, "results": rows},
        candidate_identity if architecture == "candidate" else None, method, identity)


def corrected_leaf_daily_fixture(tmp_path, method, original_checkpoint):
    executable = tmp_path / "leaf-full-state-cuda-20261005-full-state.executable.pkl"
    report_path = tmp_path / "leaf-full-state-cuda-20261005-full-state.json"
    verification_path = tmp_path / "leaf-full-state-cuda-20261005.acceptance.json"
    identity = {"source_git_head": "head", "turnover_sha256": "turnover",
        "native_AD_argnums": [0, 1], "parameters": 64884481, "parameter_leaves": 300,
        "precision": "FP32_neural_FP64_physical", "loss": "unchanged_public_joint_daily_mean4",
        "prepared_context_identity": {"method": method, "artifact_id": 742,
            "daily_rows": 4, "initial_checkpoint": original_checkpoint}}
    shapes = [[4, 2], [4, 66], [4, 5], [4, 142], [4, 1222], [4, 466], [4, 3854]]
    rows = [{"schedule": schedule, "repeat": repeat, "validation": {
        "finite": True, "physical_hard_counts_zero": True,
        "hard_counts": {"mask_mismatch": 0, "state_nonfinite": 0},
        "parameter_gradient_nonzero": 1, "initial_state_gradient_nonzero": 100,
        "state_diagnostic_group_shapes": {"expected": shapes, "predicted": shapes}}}
        for schedule in ("old", "new") for repeat in (1, 2)]
    report = {"schema_version": "shared_leaf_corrected_full_state_cuda_v1",
        "status": "completed", "scientific_acceptance": False, "adoption": False,
        "finite_and_physical_passed": True, "meaningful_gradients_passed": True,
        "observed_peak_memory_passed": True, "same_input_strict_parity": False,
        "identity": identity, "results": rows}
    state = {"coordinates": 15416, "failed": 7, "passed": False}
    regression = {"coordinates": 15416, "failed": 34, "passed": False}
    verification = {"schema_version": "shared_leaf_cuda_preserved_result_inspection_v1",
        "status": "inspected", "scientific_acceptance": False, "GPU_executed": False,
        "original_forward_parameter_passed": True, "repeat_forward_parameter_passed": True,
        "repeat_all_state_passed": False,
        "corrected_finite_physical_reported": True, "source_corrected_report": str(report_path),
        "repeat_comparisons": {name: {"all_parameter_gradients": {"passed": True},
            "forward_and_gates": {"passed": True}, "all_initial_state_gradients": state}
            for name in ("old", "new")},
        "original_regression": {name: {"state_gradient_changes": regression}
            for name in ("old", "new")}}
    executable.write_bytes(b"saved executable placeholder")
    report_path.write_text(json.dumps(report))
    verification_path.write_text(json.dumps(verification))
    Path(str(executable) + ".identity.json").write_text(json.dumps({"identity": identity, "component": "daily"}))
    return {"report": str(report_path), "executable": str(executable),
            "verification_report": str(verification_path)}, identity


def tiny_full_evaluators():
    def daily(parameters, initial, samples):
        value = parameters["weight"][0] ** 2 * jnp.mean(initial[:, 0])
        return value, {"hard": {"ok": jnp.int32(0)}, "flux": value / 3, "stock": value / 3}
    def recurrent(parameters, initial, a, b, c, d):
        value = parameters["weight"][0] ** 2 * jnp.mean(initial)
        return value, {"hard": {"ok": jnp.int32(0)}, "flux": value / 2, "stock": value / 2}
    return {name: FullStateADEvaluator(jax, jax.value_and_grad(daily if name == "daily" else recurrent,
                argnums=(0, 1), has_aux=True), name) for name in ("daily", "T7", "T30")}


def test_public_serial_and_remote_dual_changing_current_parameters_and_all_states():
    device = jax.local_devices(backend="cpu")[0]
    parameters = {"weight": jnp.array([2.], dtype=jnp.float32)}
    optimizer = initialize_target_oriented_adamw(parameters)
    operations = make_tree_operations(jax, jnp, 32)
    update = make_update({"optimizer": {"learning_rate": .02, "clip_norm": 1., "beta1": .9,
        "beta2": .999, "epsilon": 1e-8, "weight_decay": 0.}})
    evaluators = tiny_full_evaluators()
    serial = SerialFullStateExecutor(jax=jax, jnp=jnp, parameters=parameters, optimizer=optimizer,
        evaluators=evaluators, operations=operations, update=update)
    worker_evaluators = tiny_full_evaluators()
    thirty = stateful_branch(make_recurrent_branch(jax=jax, device=device,
        evaluators={30: worker_evaluators["T30"]}, operations=operations, horizons=(30,),
        preserve_auxiliary=True), worker_evaluators["T30"])
    seen = []
    class Worker:
        def evaluate(self, jax_module, current, cases):
            command = {"kind": "evaluate", "parameters": jax_module.device_get(current),
                       "cases": jax_module.device_get(cases), "parameter_version": self.parameter_version}
            seen.append((command["parameter_version"], command["parameters"]["weight"].copy()))
            response = evaluate_command(jax_module, device, thirty, command)
            self.last_parameter_version = response["executed_parameter_version"]
            self.last_device_memory = {}
            return response["result"], response["recurrent_seconds"], response["worker_total_seconds"]
    daily = stateful_branch(make_microbatch_daily_branch(jax=jax, jnp=jnp, device=device,
        evaluator=evaluators["daily"], operations=operations, microbatch_size=4,
        expected_count=32, preserve_auxiliary=True), evaluators["daily"], daily=True)
    seven = stateful_branch(make_recurrent_branch(jax=jax, device=device, evaluators={7: evaluators["T7"]},
        operations=operations, horizons=(7,), preserve_auxiliary=True), evaluators["T7"])
    samples = Samples(Inputs(jnp.broadcast_to(jnp.arange(1, 33, dtype=jnp.float32)[:, None], (32, 3854))))
    cases = {h: (jnp.ones(3854, dtype=jnp.float32) * h, *[jnp.ones(1)] * 4) for h in (7, 30)}
    with RemoteDualExecutor(jax=jax, parameters=parameters, optimizer=optimizer, daily_branch=daily,
        worker=Worker(), combine=make_component_combiner(jax), update=update, finite=operations["finite"],
        local_recurrent_branch=seven, preserve_components=True, audit_parameter_versions=True) as dual:
        for index in range(2):
            sr, sv = serial.step(daily_payload=samples, recurrent_payload=cases, persist=True)
            dr, dv = dual.step(daily_payload=samples, recurrent_payload=cases, persist=True)
            comparison = compare_complete(jax, sv, dv)
            assert comparison["strict_parity_passed"]
            assert dr["branch_parameter_versions"] == {"daily": index, "T7": index, "T30": index}
            assert int(serial.optimizer.step) == int(dual.optimizer.step) == index + 1
            component = dv["components"]["daily"]
            assert component["full_state_ad_calls"] == 8
            assert sr["initial_state_AD"]["daily"]["shape"] == [32, 3854]
            assert sr["initial_state_AD"]["daily"]["finite"] is True
            assert sr["initial_state_AD"]["daily"]["nonzero"] == 32
            assert component["initial_state_gradient"].shape == (32, 3854)
            np.testing.assert_allclose(component["initial_state_gradient"],
                component["initial_state_gradients_micro4_mean"].reshape(32, 3854) / 8)
            # Derivative of daily32 mean wrt each row's first coordinate.
            np.testing.assert_allclose(component["initial_state_gradient"][:, 0], seen[-1][1][0] ** 2 / 32)
            bad = {**dv, "components": deepcopy(dv["components"])}
            bad["components"]["T30"]["initial_state_gradient"][0] += 1
            with pytest.raises(ValueError, match="parity"):
                compare_complete(jax, sv, bad)
    assert [version for version, _ in seen] == [0, 1]
    assert not np.array_equal(seen[0][1], seen[1][1])


def test_real_private_pipe_forwards_current_parameters_and_version(tmp_path):
    with worker_class()(tmp_path / "unused.json", "candidate", cuda_device="1",
            parameters={"weight": np.array([1.])}, case=(1,), identity={}, echo_test=True) as worker:
        for version in (0, 1):
            worker.parameter_version = version
            params = {"weight": np.array([1. + version])}
            result, _, _ = worker.evaluate(jax, params, {30: (np.array([7. + version]),)})
            np.testing.assert_array_equal(result["parameters"]["weight"], params["weight"])
            assert result["parameter_version"] == worker.last_parameter_version == version
            np.testing.assert_array_equal(result["cases"][30][0], [7. + version])


def test_worker_rejects_wrong_horizon_or_missing_version_before_executing():
    for command in ({"kind": "evaluate", "cases": {7: ()}, "parameter_version": 0},
                    {"kind": "evaluate", "cases": {30: ()}, "parameter_version": None}):
        with pytest.raises(ValueError):
            evaluate_command(jax, None, lambda *_: pytest.fail("must not execute"), command)


def test_worker_initializes_backend_before_unpickling_prepared_case(monkeypatch):
    import scripts.hpc.probe_shared_latent_dual_worker as worker_module
    import scripts.hpc.probe_shared_dual as branches
    import scripts.hpc.run_shared_learning as learning
    events = []
    fake_jax = SimpleNamespace(config=SimpleNamespace(update=lambda *_: None), device_put=lambda value, *_: value)
    device = SimpleNamespace(platform="gpu")
    def initialize():
        assert events[:2] == ["manifest", "readiness"]
        events.append("backend")
        return fake_jax, (device,)
    def overlays():
        assert events[:3] == ["manifest", "readiness", "backend"]
        events.append("overlays")
    def read_manifest(_):
        events.append("manifest")
        return {"admission_mode": "corrected_performance_diagnostic"}
    def readiness(_):
        events.append("readiness")
        return {"status": "ready_for_saved_graph_load"}
    parent = SimpleNamespace(load_adjacent=lambda _: SimpleNamespace(initialize_gpu_backend=initialize),
        load_runtime_overlays=overlays, read_manifest=read_manifest,
        validate_manifest_asset_readiness=readiness,
        load_components=lambda *args, **kwargs: ({"T30": object()}, {"T30": {}}),
        stateful_branch=lambda branch, evaluator: branch)
    monkeypatch.setattr(worker_module, "load_parent", lambda: parent)
    monkeypatch.setattr(learning, "make_tree_operations", lambda *_: {})
    monkeypatch.setattr(branches, "make_recurrent_branch", lambda **_: None)
    commands = iter([{"kind": "initialize", "parameters": {}, "case": (), "identity": {}}, {"kind": "close"}])
    def receive():
        assert events[:4] == ["manifest", "readiness", "backend", "overlays"]
        events.append("receive")
        return next(commands)
    sent = []
    worker_module.serve(SimpleNamespace(manifest=Path("unused.json"), architecture="candidate"), receive, sent.append)
    assert [message["status"] for message in sent] == ["ready", "closed"]


def test_worker_rejects_missing_daily_assets_before_gpu_initialization(monkeypatch):
    import scripts.hpc.probe_shared_latent_dual_worker as worker_module
    events = []
    def initialize():
        events.append("gpu")
        pytest.fail("GPU backend must not initialize before corrected daily assets exist")
    parent = SimpleNamespace(
        read_manifest=lambda _: {"admission_mode": "corrected_performance_diagnostic"},
        validate_manifest_asset_readiness=lambda _: {
            "status": "blocked_missing_corrected_daily_graphs",
            "missing_corrected_daily_graphs": [{"architecture": "original"}, {"architecture": "candidate"}],
        },
        load_adjacent=lambda _: SimpleNamespace(initialize_gpu_backend=initialize),
    )
    monkeypatch.setattr(worker_module, "load_parent", lambda: parent)
    with pytest.raises(FileNotFoundError, match="blocked_missing_corrected_daily_graphs"):
        worker_module.serve(SimpleNamespace(manifest=Path("unused.json")), lambda: pytest.fail("must not receive"), lambda _: None)
    assert events == []


@pytest.mark.parametrize("wrong", ["parameter_version", "architecture"])
def test_parent_rejects_worker_response_for_other_version_or_architecture(wrong):
    import io
    import pickle
    owner = worker_class()
    worker = owner.__new__(owner)
    worker.metadata = {}
    worker.active_architecture = "candidate"
    worker.parameter_version = 1
    response = {"status": "passed", "executed_parameter_version": 99 if wrong == "parameter_version" else 1,
        "executed_architecture": "original" if wrong == "architecture" else "candidate", "result": {30: {}},
        "recurrent_seconds": 0., "worker_total_seconds": 0.}
    worker.process = SimpleNamespace(stdin=io.BytesIO(), stdout=io.BytesIO(pickle.dumps(response)), poll=lambda: 0)
    with pytest.raises(RuntimeError, match="version differs|different architecture"):
        worker.evaluate(jax, {"weight": np.ones(1)}, {30: (np.ones(1),)})


def test_real_compile_guard_forbids_model_lookup_but_releases_helper_jit():
    lowered = jax.jit(lambda x: x * 1.23789).lower(jnp.ones(3))
    with forbid_model_compile():
        with pytest.raises(RuntimeError, match="model compilation"):
            lowered.compile()
    executable = lowered.compile()
    np.testing.assert_allclose(executable(jnp.ones(3)), np.full(3, 1.23789), rtol=1e-6)


@pytest.mark.parametrize("memory", [None, {}, {"peak_bytes_in_use": 25 * 2**30}, {"peak_bytes_in_use": np.nan}])
def test_observed_memory_must_exist_and_fit_physical_card(memory):
    with pytest.raises(MemoryError):
        require_memory(memory, label="CPU_fake_process")


def test_full_state_call_counts_reject_incomplete_update():
    evaluator = SimpleNamespace(take_records=lambda: ({"initial_state_gradient": np.ones((4, 3854))},))
    with pytest.raises(ValueError, match="call count"):
        attach_states({}, evaluator, daily=True)


def manifest_fixture(tmp_path):
    return {"schema_version": MANIFEST_SCHEMA, "reference": {"source_git_head": "head",
        "method": {"training_plan": {"training_artifact_ids": [1, 2]}, "daily_protocol": {"optimizer": {}}}},
        "input_bundle": str(tmp_path / "bundle.pkl"), "original_checkpoint": str(tmp_path / "original.pkl"),
        "candidate_parameters": str(tmp_path / "candidate.pkl"), "architectures": {
            name: {component: {"report": "report.json", "executable": "full.pkl"} for component in ("daily", "T7", "T30")}
            for name in ("original", "candidate")}}


def test_manifest_requires_both_explicit_full_component_routes(tmp_path):
    manifest = manifest_fixture(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert read_manifest(path) == manifest
    for bad in ({**manifest, "schema_version": "old_parameter_only"},
                {**manifest, "architectures": {"original": manifest["architectures"]["original"]}}):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            read_manifest(path)


def test_corrected_manifest_requires_context_identity_bound_to_profile(tmp_path):
    from scripts.hpc.probe_shared_latent_dual import CORRECTED_PERFORMANCE_DIAGNOSTIC

    manifest = manifest_fixture(tmp_path)
    manifest["admission_mode"] = CORRECTED_PERFORMANCE_DIAGNOSTIC
    manifest["scientific_acceptance"] = False
    manifest["reference"]["source_git_head"] = "head"
    manifest["performance_diagnostic"] = {
        "source_git_head": "head", "turnover_sha256": "turnover",
        "method_sha256": hashlib.sha256(json.dumps(manifest["reference"]["method"],
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "prepared_context_identity": {
            "method_sha256": hashlib.sha256(json.dumps(manifest["reference"]["method"],
                sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "artifact_id": 1, "daily_rows": 4,
            "initial_checkpoint": manifest["original_checkpoint"],
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert read_manifest(path)["performance_diagnostic"]["prepared_context_identity"] == \
        manifest["performance_diagnostic"]["prepared_context_identity"]
    manifest["performance_diagnostic"]["prepared_context_identity"]["artifact_id"] = 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="prepared-context identity"):
        read_manifest(path)


def test_corrected_manifest_readiness_admits_original_leaf_and_blocks_candidate_daily(tmp_path):
    from scripts.hpc.probe_shared_latent_dual import CORRECTED_PERFORMANCE_DIAGNOSTIC
    from scripts.hpc.shared_full_state_ad_execution import CANDIDATE_ARCHITECTURE

    method = {"daily_protocol": {"seed": 3}, "training_plan": {"training_artifact_ids": [742, 275]}}
    reference = {"source_git_head": "head", "method": method}
    architectures = {name: {} for name in ("original", "candidate")}
    candidate_identity = None
    checkpoint = tmp_path / "original.pkl"
    original_daily, _ = corrected_leaf_daily_fixture(tmp_path, method, str(checkpoint))
    for component in ("T7", "T30"):
        original_report, _, _, original_identity = corrected_report(component, "original", method)
        candidate_report, candidate_identity, _, candidate_identity_record = corrected_report(component, "candidate", method)
        candidate_identity = candidate_identity
        report = dict(original_report)
        report["identities"] = {"original": original_identity,
                                CANDIDATE_ARCHITECTURE: candidate_identity_record}
        report["model_compiles"] = {"original": 0, CANDIDATE_ARCHITECTURE: 0}
        report["cold_measurements"] = {
            "original": {"executable": str(tmp_path / f"{component}-original.pkl")},
            CANDIDATE_ARCHITECTURE: {"executable": str(tmp_path / f"{component}-candidate.pkl")},
        }
        report["results"] = original_report["results"] + candidate_report["results"]
        report_path = tmp_path / f"{component}.json"
        report_path.write_text(json.dumps(report))
        for name, arch in (("original", "original"), ("candidate", CANDIDATE_ARCHITECTURE)):
            binary = tmp_path / f"{component}-{name}.pkl"
            binary.write_bytes(b"saved")
            binary.with_suffix(binary.suffix + ".identity.json").write_text(json.dumps({
                "identity": report["identities"][arch], "component": component}))
            architectures[name][component] = {"report": str(report_path), "executable": str(binary)}
    candidate_parameters = tmp_path / "candidate.pkl"
    candidate_parameters.write_bytes(pickle.dumps({"identity": candidate_identity}))
    architectures["original"]["daily"] = original_daily
    architectures["candidate"]["daily"] = {
        "report": str(tmp_path / "daily-candidate.json"),
        "executable": str(tmp_path / "daily-candidate.pkl"),
    }
    bundle = tmp_path / "inputs.pkl"
    bundle.write_bytes(b"inputs")
    checkpoint.write_bytes(b"checkpoint")
    manifest = {
        "admission_mode": CORRECTED_PERFORMANCE_DIAGNOSTIC,
        "scientific_acceptance": False,
        "reference": reference,
        "performance_diagnostic": {"turnover_sha256": "turnover",
            "method_sha256": hashlib.sha256(json.dumps(method, sort_keys=True,
                separators=(",", ":"), default=str).encode()).hexdigest(),
            "prepared_context_identity": {
                "method_sha256": hashlib.sha256(json.dumps(method, sort_keys=True,
                    separators=(",", ":"), default=str).encode()).hexdigest(),
                "artifact_id": 742, "daily_rows": 4,
                "initial_checkpoint": str(checkpoint)}},
        "input_bundle": str(bundle), "original_checkpoint": str(checkpoint),
        "candidate_parameters": str(candidate_parameters), "architectures": architectures,
    }
    result = validate_manifest_asset_readiness(manifest)
    assert result["status"] == "blocked_missing_corrected_daily_graphs"
    daily_blockers = [row for row in result["missing_corrected_daily_graphs"] if isinstance(row, dict)]
    assert {row["architecture"] for row in daily_blockers} == {"candidate"}
    assert result["gpu_backend_initialized"] is False
    assert result["optimizer_updates"] == 0
    assert result["corrected_component_identity_checks"]["daily"]["original"][
        "prepared_context_identity"] == manifest["performance_diagnostic"]["prepared_context_identity"]
    assert result["corrected_component_identity_checks"]["daily"]["original"]["status"] == "validated"
    assert result["corrected_component_identity_checks"]["daily"]["original"]["report_status"] == "numerical_failed"
    repeat_failures = result["corrected_component_identity_checks"]["daily"]["original"]["state_repeat_failure_records"]
    assert [failure["all_initial_state_gradients"]["failed"] for failure in repeat_failures] == [7, 7]
    regression = result["corrected_component_identity_checks"]["daily"]["original"]["source_corrected_vs_original_regression"]
    assert [record["state_gradient_changes"]["failed"] for record in regression] == [34, 34]
    assert result["corrected_component_identity_checks"]["daily"]["candidate"]["status"] == "missing_saved_asset"
    assert result["corrected_component_identity_checks"]["T7"]["original"]["status"] == "validated"
    assert result["corrected_component_identity_checks"]["T30"]["candidate"]["status"] == "validated"


def test_corrected_leaf_daily_loader_uses_source_schema_and_checks_saved_abi(tmp_path):
    from scripts.hpc.probe_shared_latent_dual import CORRECTED_PERFORMANCE_DIAGNOSTIC
    from scripts.hpc.shared_full_state_ad_execution import load_full_state_executable

    method = {"daily_protocol": {"seed": 3}, "training_plan": {"training_artifact_ids": [742, 275]}}
    checkpoint = tmp_path / "original.pkl"
    item, identity = corrected_leaf_daily_fixture(tmp_path, method, str(checkpoint))
    report = json.loads(Path(item["report"]).read_text())
    verification = json.loads(Path(item["verification_report"]).read_text())
    initial = np.ones((4, 3854), dtype=np.float32)
    samples = Samples(Inputs(initial))
    arguments = ({"weight": np.ones(1, dtype=np.float32)}, initial, samples)
    supplied = (arguments, {})
    treedef = jax.tree_util.tree_structure(supplied)
    infos = jax.tree_util.tree_map(lambda value: SimpleNamespace(
        _aval=SimpleNamespace(shape=value.shape, dtype=value.dtype)), supplied)
    executable = SimpleNamespace(args_info=infos)
    with Path(item["executable"]).open("wb") as stream:
        pickle.dump((None, treedef, None), stream)
    expected_method_sha = hashlib.sha256(json.dumps(method, sort_keys=True,
        separators=(",", ":"), default=str).encode()).hexdigest()
    expected_context = {"method_sha256": expected_method_sha, "artifact_id": 742,
        "daily_rows": 4, "initial_checkpoint": str(checkpoint)}
    loaded, metadata = load_full_state_executable(jax, item["executable"], arguments,
        component="daily", admission=item["report"], architecture="original",
        expected_reference={"source_git_head": "head", "method": method},
        admission_mode=CORRECTED_PERFORMANCE_DIAGNOSTIC,
        expected_turnover_sha256="turnover", corrected_daily_verification=verification,
        expected_method_sha256=expected_method_sha,
        expected_prepared_context_identity=expected_context,
        deserialize=lambda *_: executable, memory_sizes=lambda _: {"estimated_live_bytes": 1})
    assert loaded is executable
    assert metadata["report_status"] == "numerical_failed"
    assert metadata["identity"] == identity
    assert metadata["full_state_abi_validated"] is True
    with pytest.raises(ValueError, match="identity/schema"):
        load_full_state_executable(jax, item["executable"], arguments,
            component="daily", admission=item["report"], architecture="original",
            expected_reference={"source_git_head": "head", "method": method},
            admission_mode=CORRECTED_PERFORMANCE_DIAGNOSTIC,
            expected_turnover_sha256="turnover", corrected_daily_verification=verification,
            expected_method_sha256=expected_method_sha,
            expected_prepared_context_identity={**expected_context, "artifact_id": 743},
            deserialize=lambda *_: executable, memory_sizes=lambda _: {"estimated_live_bytes": 1})


def test_corrected_daily_candidate_keeps_global_state_failure_without_selected_failure(tmp_path):
    from scripts.hpc.shared_full_state_ad_execution import (
        CANDIDATE_ARCHITECTURE, validate_corrected_performance_diagnostic,
    )

    context = {"method_sha256": "method", "artifact_id": 742,
        "daily_rows": 4, "initial_checkpoint": "/runtime/update-000000.pkl"}
    parameters = {"seed": 41}
    identity = {"source_git_head": "head", "turnover_sha256": "turnover",
        "native_AD_argnums": [0, 1], "training_updates": 0,
        "precision": "FP32_neural_FP64_physics", "architecture": CANDIDATE_ARCHITECTURE,
        "parameter_count": 66123713, "parameter_leaves": 355,
        "prepared_context_identity": context,
        "parameter_provenance": {"identity": parameters}}
    validation = {"gradient_coordinates": 66123713, "gradient_leaves": 355,
        "gradient_nonzero": 1, "gradient_norm": 1.,
        "initial_state_shape": [4, 3854], "initial_state_finite": True,
        "initial_state_nonzero": 1, "hard_counts": {"physical": 0}}
    candidate_repeat = {"all_parameter_gradients": {"passed": True},
        "forward_and_gates": {"passed": True}, "loss_close": True,
        "all_initial_state_gradients": {"passed": True, "coordinates": 15416, "failed": 0}}
    source_failure = {"architecture": "original", "component": "daily",
        "schedule": "old", "all_initial_state_gradients": {
            "passed": False, "coordinates": 15416, "failed": 3}}
    current_failure = {"architecture": "original", "component": "daily", "label": "pair1",
        "comparison": {"forward_and_gates": {"passed": True},
            "all_parameter_gradients": {"passed": True},
            "all_initial_state_gradients": {"passed": False, "coordinates": 15416, "failed": 2}}}
    report = {"schema_version": "shared_corrected_latent_reader_daily_pair_v1",
        "status": "numerical_failed", "component": "daily", "native_AD_argnums": [0, 1],
        "training_updates": 0, "precision": "FP32_neural_FP64_physics",
        "scientific_acceptance": False, "model_compiles": {"candidate": 0},
        "identities": {"candidate": identity},
        "cold_measurements": {"candidate": {"executable": str(tmp_path / "candidate.executable.pkl")}},
        "results": [
            {"architecture": "candidate", "label": "cold-first", "validation": validation},
            {"architecture": "candidate", "label": "pair1", "validation": validation,
                "same_architecture_repeat": candidate_repeat},
        ],
        "numerical_failures": [current_failure],
        "inherited_source_strict_state_failures": [source_failure],
        "current_pair_state_parity_passed": False, "current_pair_state_failure_count": 1,
    }
    validated = validate_corrected_performance_diagnostic(
        report, "daily", CANDIDATE_ARCHITECTURE, expected_identity=parameters,
        expected_reference={"source_git_head": "head"}, expected_turnover_sha256="turnover",
        expected_executable=str(tmp_path / "candidate.executable.pkl"),
        expected_prepared_context_identity=context)
    assert validated["report_status"] == "numerical_failed"
    assert validated["state_repeat_failure_records"] == []
    assert validated["current_pair_state_failure_count"] == 1
    assert len(validated["inherited_source_strict_state_failures"]) == 1
    bad = deepcopy(report)
    bad["numerical_failures"][0]["comparison"]["forward_and_gates"]["passed"] = False
    with pytest.raises(ValueError, match="non-state"):
        validate_corrected_performance_diagnostic(
            bad, "daily", CANDIDATE_ARCHITECTURE, expected_identity=parameters,
            expected_reference={"source_git_head": "head"}, expected_turnover_sha256="turnover",
            expected_executable=str(tmp_path / "candidate.executable.pkl"),
            expected_prepared_context_identity=context)


def test_actual_cli_bounds_and_overwrite_are_checked_before_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    flags = ["--manifest", str(tmp_path / "manifest.json"), "--output", str(tmp_path / "result.json"),
             "--worker-cuda-device", "1"]
    args = parser().parse_args(flags)
    assert args.repeats == 3 and args.changing_updates == 4 and args.equivalence_updates == 2
    validate_args(args)
    for extra in (["--repeats", "2"], ["--changing-updates", "17"], ["--changing-updates", "-1"],
                  ["--changing-updates", "8"], ["--repeats", "7"], ["--equivalence-updates", "4"]):
        with pytest.raises(ValueError):
            validate_args(parser().parse_args(flags + extra))
    args.output.write_text("already exists")
    with pytest.raises(FileExistsError):
        validate_args(args)


def test_bundle_requires_eight_distinct_micro4_and_declared_cases(tmp_path):
    manifest = manifest_fixture(tmp_path)
    daily = Samples(Inputs(np.broadcast_to(np.arange(32, dtype=np.float32)[:, None], (32, 3854))))
    cases = {h: Case(np.ones(3854), np.ones(1), (Inputs(np.ones((h, 3854))),), np.ones(1), np.ones(1)) for h in (7, 30)}
    payload = {"daily": daily, "cases": cases}
    bundle = {"schema_version": "shared_serialized_dual_inputs_v1", **manifest["reference"],
        "initial_checkpoint": manifest["original_checkpoint"], "artifact_ids": (1, 2), "payloads": [payload, payload]}
    assert len(validate_bundle(jax, bundle, manifest)["daily_micro4_sha256"][0]) == 8
    repeated = {**payload, "daily": Samples(Inputs(np.ones((32, 3854))))}
    with pytest.raises(ValueError, match="eight distinct"):
        validate_bundle(jax, {**bundle, "payloads": [repeated, payload]}, manifest)


@pytest.mark.parametrize("fault", ["missing_original", "wrong_version", "different_payload", "nonfinite_time"])
def test_changed_weight_timing_requires_matched_full_architecture_pairs(fault):
    rows = [{"chain_step": 1, "architecture": name, "artifact_id": 1,
             "input_parameter_version": 0, "worker_executed_parameter_version": 0,
             "complete_seconds": 2. if name == "original" else 1.}
            for name in ("original", "candidate")]
    if fault == "missing_original": rows = rows[1:]
    if fault == "wrong_version": rows[1]["worker_executed_parameter_version"] = 1
    if fault == "different_payload": rows[1]["artifact_id"] = 2
    if fault == "nonfinite_time": rows[1]["complete_seconds"] = np.nan
    with pytest.raises(ValueError):
        changed_weight_timing_summary(rows)


@pytest.mark.parametrize("benefit,changed_benefit,performance_steps,boundary_artifact,diagnostic_mode,state_failure", [
    (True, True, 4, False, False, False), (False, False, 4, False, False, False),
    (True, False, 4, False, False, False), (True, True, 2, False, False, False),
    (True, True, 3, False, False, False), (True, False, 4, True, False, False),
    (False, False, 4, False, True, True)])
def test_run_orchestration_matches_fixed_work_and_gates_only_candidate_chains(
        tmp_path, monkeypatch, benefit, changed_benefit, performance_steps, boundary_artifact,
        diagnostic_mode, state_failure):
    """Exercise actual run control flow with CPU fake binaries/runtime/clock."""
    import pickle
    import scripts.hpc.probe_shared_latent_dual as probe
    import scripts.hpc.probe_shared_dual as branches
    import scripts.hpc.probe_shared_serialized_dual as remote
    import scripts.hpc.run_shared_learning as learning
    from scripts.hpc.probe_shared_latent_dual import CORRECTED_PERFORMANCE_DIAGNOSTIC
    from research.daily_coarse_graining.daily_operator import fresh_start_training
    manifest = manifest_fixture(tmp_path)
    if diagnostic_mode:
        manifest.update(admission_mode=CORRECTED_PERFORMANCE_DIAGNOSTIC, scientific_acceptance=False,
            performance_diagnostic={"source_git_head": "head", "turnover_sha256": "turnover"})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    args = parser().parse_args(["--manifest", str(path), "--output", str(tmp_path / "result.json"),
        "--worker-cuda-device", "1", "--changing-updates", str(performance_steps)])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    clock = SimpleNamespace(value=0.)
    monkeypatch.setattr(probe.time, "perf_counter", lambda: clock.value)
    device = SimpleNamespace(platform="gpu", memory_stats=lambda: {"peak_bytes_in_use": 17})
    fake_jax = SimpleNamespace(config=SimpleNamespace(update=lambda *_: None),
        device_put=lambda value, *_: value, device_get=lambda value: value, block_until_ready=lambda value: value,
        tree_util=jax.tree_util)
    runtime = SimpleNamespace(initialize_gpu_backend=lambda: (fake_jax, (device,)))
    monkeypatch.setattr(probe, "load_adjacent", lambda name: runtime)
    monkeypatch.setattr(probe, "load_runtime_overlays", lambda: None)
    monkeypatch.setattr(probe, "validate_manifest_asset_readiness", lambda _: {
        "status": "ready_for_saved_graph_load", "gpu_backend_initialized": False})
    payload = {"daily": np.ones((32, 1)), "cases": {7: (), 30: ()}}
    monkeypatch.setattr(probe, "load_pickle", lambda _: {"payloads": [payload, payload]})
    monkeypatch.setattr(probe, "validate_bundle", lambda *_: {"artifact_ids": [1, 2]})
    Optimizer = namedtuple("FakeOptimizer", "step")
    monkeypatch.setattr(probe, "initial_architecture", lambda jax, manifest, architecture:
        ({"weight": np.array([1. if architecture == "original" else 2.], dtype=np.float32)}, Optimizer(0), {}))
    monkeypatch.setattr(probe, "load_components", lambda *args, **kwargs:
        ({name: SimpleNamespace(component=name) for name in kwargs["components"]}, {}))
    monkeypatch.setattr(learning, "make_tree_operations", lambda *_: {"finite": lambda _: True})
    monkeypatch.setattr(fresh_start_training, "make_update", lambda *_: None)
    monkeypatch.setattr(branches, "make_component_combiner", lambda *_: None)
    monkeypatch.setattr(branches, "make_microbatch_daily_branch", lambda **_: None)
    monkeypatch.setattr(branches, "make_recurrent_branch", lambda **_: None)
    class Worker:
        def __init__(self, manifest, architecture, **kwargs):
            self.last_device_memory = {"peak_bytes_in_use": 11}
            self.select(architecture)
        def select(self, architecture, **kwargs):
            self.active_architecture = architecture
            self.metadata = {"architecture": architecture, "executables": {"T30": {"source_executable": "full.pkl"}}}
            return 0.
        def close(self): pass
    monkeypatch.setattr(probe, "worker_class", lambda: Worker)
    events = []
    class Executor:
        mode = "serial"
        def __init__(self, *, parameters, optimizer, worker=None, **kwargs):
            self.parameters, self.optimizer, self.worker = parameters, optimizer, worker
            self.daily_branch = kwargs.get("daily_branch")
            self.local_recurrent_branch = kwargs.get("local_recurrent_branch")
            self.architecture = "original" if parameters["weight"][0] == 1 else "candidate"
        def step(self, *, daily_payload, recurrent_payload, persist):
            before = self.optimizer.step
            if self.worker is not None:
                assert self.worker.active_architecture == self.architecture
                self.worker.last_parameter_version = before
            events.append((self.architecture, self.mode, persist, before))
            faster = changed_benefit if persist else benefit
            if boundary_artifact and persist and self.mode == "dual":
                # Four original times [100,2,1,1] give all median1.5,
                # after-first median1. Candidate times all1: only the
                # disclosed first boundary makes the all-pair gate pass.
                clock.value += [100., 2., 1., 1.][before] if self.architecture == "original" else 1.
            else:
                clock.value += (2. if self.architecture == "original" else 1.) if faster else (1. if self.architecture == "original" else 2.)
            next_params = {"weight": self.parameters["weight"] * np.float32(.99)}
            next_optimizer = Optimizer(before + 1)
            components = {}
            for name in ("daily", "T7", "T30"):
                array = np.ones((32, 3854) if name == "daily" else (3854,), dtype=np.float32)
                if state_failure and self.mode == "dual" and before == 0:
                    array = array.copy()
                    if array.ndim == 2:
                        array[0, 0] += 0.01
                    else:
                        array[0] += 0.01
                components[name] = {"loss": 1., "gradient": {"weight": np.ones(1, dtype=np.float32)},
                    "initial_state_gradient": array, "initial_state_AD": {"shape": list(array.shape),
                    "dtype": "float32", "finite": True, "nonzero": array.size, "coordinates": array.size}}
            values = {"parameters": next_params, "optimizer": next_optimizer,
                "gradient": {"weight": np.ones(1, dtype=np.float32)}, "components": components}
            record = {"loss": {"joint": 3.}, "timing_seconds": {"total": 1.}, "worker_request_response_timings": {},
                "all_hard_counts_zero": True}
            probe.record_state_ad(record, values)
            if persist:
                self.parameters, self.optimizer = next_params, next_optimizer
            return record, values
        def close(self): pass
    class DualExecutor(Executor):
        mode = "dual"
    monkeypatch.setattr(probe, "SerialFullStateExecutor", Executor)
    monkeypatch.setattr(remote, "RemoteDualExecutor", DualExecutor)
    # Local optimizer namedtuple is not pickleable; artifact writer still uses
    # actual atomic writes after converting only this test fixture's class.
    real_atomic = probe.atomic_pickle
    def atomic(path, value):
        if "optimizer" in value:
            value = {**value, "optimizer": tuple(value["optimizer"])}
        real_atomic(path, value)
    monkeypatch.setattr(probe, "atomic_pickle", atomic)
    report = probe.run(args)
    assert report["fixed_screen_passed"] is benefit
    assert report["status"] == ("numerical_failed" if state_failure else "passed")
    assert [event[:2] for event in events[4:10]] == [("original", "dual"), ("candidate", "dual"),
        ("candidate", "dual"), ("original", "dual"), ("original", "dual"), ("candidate", "dual")]
    retained = [event for event in events if event[2]]
    expected_performance = [(architecture, "dual", True, index) for index in range(performance_steps)
        for architecture in (("original", "candidate") if index % 2 == 0 else ("candidate", "original"))]
    expected_equivalence = ([
        ("original", "serial", True, 0), ("original", "dual", True, 0),
        ("original", "serial", True, 1), ("original", "dual", True, 1),
        ("candidate", "serial", True, 0), ("candidate", "dual", True, 0),
        ("candidate", "serial", True, 1), ("candidate", "dual", True, 1)] if diagnostic_mode else [
        ("candidate", "serial", True, 0), ("candidate", "dual", True, 0),
        ("candidate", "serial", True, 1), ("candidate", "dual", True, 1)])
    should_measure = benefit or diagnostic_mode
    assert retained == (expected_performance + expected_equivalence if should_measure else [])
    assert report["candidate_actual_retained_optimizer_updates"] == (performance_steps + 4 if should_measure else 0)
    assert report["original_actual_retained_optimizer_updates"] == (performance_steps + (4 if diagnostic_mode else 0) if should_measure else 0)
    assert report["candidate_optimizer_calls_completed"] == report["candidate_optimizer_calls_attempted"] == (9 + performance_steps if should_measure else 5)
    assert report["original_optimizer_calls_completed"] == report["original_optimizer_calls_attempted"] == ((9 if diagnostic_mode else 5) + performance_steps if should_measure else 5)
    assert report["final_changed_weight_speed_acceptance_passed"] is (
        benefit and changed_benefit and performance_steps >= 3 and not state_failure)
    if diagnostic_mode:
        assert report["executed_dual_performance_steps_per_architecture"] == performance_steps
        assert report["executed_original_equivalence_steps_per_chain"] == 2
        assert report["fixed_screen_gate_applies_to_followup_measurements"] is False
        assert len(report["strict_state_repeat_failures"]) >= 1
    if boundary_artifact:
        assert report["changed_weight_timing"]["all_paired_complete_updates"]["original_over_candidate"] == 1.5
        assert report["changed_weight_timing"]["post_first_step_changed_weights"]["original_over_candidate"] == 1.
        assert report["changed_weight_speed_gate_passed"] is False
        assert report["final_speed_acceptance_conditions"]["post_first_steady_complete_update_median_ratio_at_least1_5"] is False
    assert Path(report["state_diagnostics_artifact"]).exists()
    if should_measure:
        assert report["changed_weight_timing"]["all_paired_complete_updates"]["paired_samples"] == performance_steps
        assert report["changed_weight_timing"]["post_first_step_changed_weights"]["pair_indices"] == list(range(2, performance_steps + 1))
        for name, artifact in report["retained_diagnostic_chains"].items():
            with Path(artifact).open("rb") as handle:
                saved = pickle.load(handle)
            assert saved["completed_diagnostic_updates"] == (performance_steps if "performance" in name else 2)
            assert saved["scientific_acceptance"] is False
