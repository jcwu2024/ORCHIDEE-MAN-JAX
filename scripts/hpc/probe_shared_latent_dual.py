"""Saved full-state AD: matched original/candidate dual complete-update screen.

The manifest explicitly binds each architecture's daily/T7/T30 full native AD
reports and binaries. This is a bounded technical diagnostic, not training or
scientific acceptance. Both architectures advance independent full-state dual
performance chains after screening; candidate serial/dual equivalence uses a
separate pair of fresh diagnostic chains.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

MANIFEST_SCHEMA = "shared_latent_dual_manifest_v1"
RTOL, ATOL = 1e-5, 1e-6
MEMORY_LIMIT = 24 * 2**30
CORRECTED_PERFORMANCE_DIAGNOSTIC = "corrected_performance_diagnostic"


class FullStateParityFailure(ValueError):
    def __init__(self, comparisons):
        super().__init__("same-architecture serial/dual full-state parity failed")
        self.comparisons = comparisons


def load_adjacent(name):
    """Select staged diagnostics explicitly over the immutable source checkout."""
    root = os.environ.get("ORCHIDEE_REPO_ROOT", str(Path(__file__).resolve().parents[2]))
    if root not in sys.path:
        sys.path.insert(0, root)
    module_name = "scripts.hpc." + name
    path = Path(__file__).with_name(name + ".py")
    existing = sys.modules.get(module_name)
    if existing is not None and Path(existing.__file__).resolve() == path.resolve():
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_runtime_overlays():
    # Called only after initialize_gpu_backend in real execution.
    for name in ("training_artifact_io", "shared_daily_microbatch", "shared_full_state_ad_execution",
                 "probe_shared_dual", "probe_shared_serialized_updates", "probe_shared_serialized_dual"):
        load_adjacent(name)


@contextmanager
def forbid_model_compile():
    """Only direct full binary loading/replay is guarded; helpers may compile."""
    from jax._src import compiler
    original = compiler.compile_or_get_cached
    def forbidden(*args, **kwargs):
        raise RuntimeError("saved full-state AD attempted model compilation")
    compiler.compile_or_get_cached = forbidden
    try:
        yield
    finally:
        compiler.compile_or_get_cached = original


def read_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("explicit full-state dual manifest schema required")
    reference = manifest.get("reference", {})
    if not reference.get("source_git_head"):
        raise ValueError("manifest original method/source reference required")
    for name in ("input_bundle", "original_checkpoint", "candidate_parameters"):
        if not manifest.get(name):
            raise ValueError(f"manifest {name} required")
    if set(manifest.get("architectures", {})) != {"original", "candidate"}:
        raise ValueError("manifest must explicitly declare original and candidate")
    mode = manifest.get("admission_mode", "scientific")
    if mode not in ("scientific", CORRECTED_PERFORMANCE_DIAGNOSTIC):
        raise ValueError("unsupported explicit full-state admission mode")
    if mode == CORRECTED_PERFORMANCE_DIAGNOSTIC:
        diagnostic = manifest.get("performance_diagnostic", {})
        if (manifest.get("scientific_acceptance") is not False
                or diagnostic.get("source_git_head") != reference["source_git_head"]
                or not diagnostic.get("turnover_sha256")):
            raise ValueError("corrected performance route must be explicitly non-scientific and identity-bound")
        method = reference.get("method")
        if method is None:
            t7_report = Path(manifest["architectures"]["original"]["T7"]["report"])
            if t7_report.is_file():
                saved = json.loads(t7_report.read_text(encoding="utf-8"))
                method = saved.get("identities", {}).get("original", {}).get("method")
                encoded = json.dumps(method, sort_keys=True, separators=(",", ":")).encode()
                if hashlib.sha256(encoded).hexdigest() != diagnostic.get("method_sha256"):
                    raise ValueError("corrected T7 method identity differs from pinned manifest")
                reference["method"] = method
        if not method:
            raise ValueError("corrected prepared-context identity requires the pinned method report")
        prepared = diagnostic.get("prepared_context_identity")
        if (not isinstance(prepared, dict)
                or set(prepared) != {"method_sha256", "artifact_id", "daily_rows", "initial_checkpoint"}
                or prepared.get("method_sha256") != diagnostic.get("method_sha256")
                or prepared.get("artifact_id") != method.get("training_plan", {}).get(
                    "training_artifact_ids", [None])[0]
                or prepared.get("daily_rows") != 4
                or prepared.get("initial_checkpoint") != manifest.get("original_checkpoint")):
            raise ValueError("corrected prepared-context identity must bind method, artifact, four rows and checkpoint")
    elif not reference.get("method"):
        raise ValueError("manifest original method/source reference required")
    for architecture, components in manifest["architectures"].items():
        if set(components) != {"daily", "T7", "T30"}:
            raise ValueError(f"{architecture} must declare every full-state component")
        for component, item in components.items():
            if not item.get("report") or not item.get("executable"):
                raise ValueError(f"{architecture}/{component} report and executable required")
    return manifest


def validate_manifest_asset_readiness(manifest):
    """CPU-only asset check; never initializes JAX or compiles a model."""
    required = [manifest["input_bundle"], manifest["original_checkpoint"], manifest["candidate_parameters"]]
    missing_daily = []
    for components in manifest["architectures"].values():
        for component, item in components.items():
            required.extend((item["report"], item["executable"]))
            if item.get("verification_report"):
                required.append(item["verification_report"])
            if manifest.get("admission_mode") == CORRECTED_PERFORMANCE_DIAGNOSTIC:
                required.append(str(item["executable"]) + ".identity.json")
            if component == "daily":
                for path in (item["report"], item["executable"], item.get("verification_report")):
                    if not path:
                        continue
                    if not Path(path).is_file():
                        missing_daily.append(str(Path(path)))
    missing = sorted({str(Path(path)) for path in required if not Path(path).is_file()})
    corrected_checks = {}
    if manifest.get("admission_mode") == CORRECTED_PERFORMANCE_DIAGNOSTIC:
        corrected_checks = validate_corrected_component_assets(manifest)
        for architecture in ("original", "candidate"):
            daily = corrected_checks.get("daily", {}).get(architecture, {})
            if daily.get("status") != "validated":
                missing_daily.append({"architecture": architecture,
                    "report": manifest["architectures"][architecture]["daily"]["report"],
                    "executable": manifest["architectures"][architecture]["daily"]["executable"],
                    "reason": daily.get("error", daily.get("status", "unavailable"))})
    all_corrected = all(
        corrected_checks.get(component, {}).get(architecture, {}).get("status") == "validated"
        for component in ("daily", "T7", "T30")
        for architecture in ("original", "candidate")) if corrected_checks else True
    if not all_corrected:
        status = "blocked_missing_corrected_daily_graphs" if missing_daily else "blocked_invalid_corrected_assets"
    else:
        status = "blocked_missing_assets" if missing else "ready_for_saved_graph_load"
    return {"status": status,
            "admission_mode": manifest.get("admission_mode", "scientific"),
            "scientific_acceptance": False, "model_gradient_compilations": 0,
            "gpu_backend_initialized": False, "optimizer_updates": 0,
            "missing_assets": missing, "missing_corrected_daily_graphs": missing_daily,
            "corrected_component_identity_checks": corrected_checks}


def validate_corrected_component_assets(manifest):
    """Read-only identity/sidecar check for corrected daily, T7 and T30 graphs."""
    adapter = load_adjacent("shared_full_state_ad_execution")
    CANDIDATE_ARCHITECTURE = adapter.CANDIDATE_ARCHITECTURE

    try:
        with Path(manifest["candidate_parameters"]).open("rb") as stream:
            candidate = pickle.load(stream)
        candidate_identity = candidate["identity"]
    except (OSError, KeyError, pickle.UnpicklingError) as error:
        return {component: {architecture: {"status": "blocked",
                    "error": f"candidate parameter identity unavailable: {error}"}
                    for architecture in ("original", "candidate")}
                for component in ("daily", "T7", "T30")}
    checked = {}
    for component in ("daily", "T7", "T30"):
        checked[component] = {}
        for name, architecture in (("original", "original"),
                                   ("candidate", CANDIDATE_ARCHITECTURE)):
            item = manifest["architectures"][name][component]
            report_path, binary_path = Path(item["report"]), Path(item["executable"])
            sidecar_path = Path(str(binary_path) + ".identity.json")
            verification_path = Path(item["verification_report"]) if item.get("verification_report") else None
            required_paths = [report_path, binary_path, sidecar_path]
            if verification_path is not None:
                required_paths.append(verification_path)
            if not all(path.is_file() for path in required_paths):
                checked[component][name] = {"status": "missing_saved_asset"}
                continue
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                if component == "daily" and name == "original" and verification_path is not None:
                    verification_report = json.loads(verification_path.read_text(encoding="utf-8"))
                    diagnostic = adapter.validate_corrected_leaf_daily_diagnostic(
                        report, verification_report, sidecar,
                        expected_reference=manifest["reference"],
                        expected_turnover_sha256=manifest["performance_diagnostic"]["turnover_sha256"],
                        expected_method_sha256=manifest["performance_diagnostic"]["method_sha256"],
                        expected_executable=binary_path,
                        expected_prepared_context_identity=manifest["performance_diagnostic"][
                            "prepared_context_identity"])
                    checked[component][name] = {"status": "validated",
                        "report_status": diagnostic["report_status"],
                        "source_report_status": diagnostic["source_report_status"],
                        "scientific_acceptance": False,
                        "validated_calls": diagnostic["validated_calls"],
                        "prepared_context_identity": manifest["performance_diagnostic"][
                            "prepared_context_identity"],
                        "state_repeat_failure_records": diagnostic["state_repeat_failure_records"],
                        "source_corrected_vs_original_regression": diagnostic[
                            "source_corrected_vs_original_regression"],
                        "executable": str(binary_path)}
                    continue
                diagnostic = adapter.validate_corrected_performance_diagnostic(
                    report, component, architecture,
                    expected_identity=candidate_identity if name == "candidate" else None,
                    expected_reference=manifest["reference"],
                    expected_turnover_sha256=manifest["performance_diagnostic"]["turnover_sha256"],
                    expected_executable=binary_path,
                    expected_prepared_context_identity=(manifest["performance_diagnostic"].get(
                        "prepared_context_identity") if name == "candidate" and component == "daily" else None))
                if (sidecar.get("component") != component
                        or adapter.json_identity(sidecar.get("identity"))
                        != adapter.json_identity(diagnostic["identity"])):
                    raise ValueError("executable sidecar identity differs")
                checked[component][name] = {"status": "validated",
                    "report_status": diagnostic["report_status"], "scientific_acceptance": False,
                    "validated_calls": diagnostic["validated_calls"],
                    "prepared_context_identity": (manifest["performance_diagnostic"][
                        "prepared_context_identity"] if name == "candidate" and component == "daily" else None),
                    "current_pair_state_parity_passed": diagnostic[
                        "current_pair_state_parity_passed"],
                    "current_pair_state_failure_count": diagnostic[
                        "current_pair_state_failure_count"],
                    "state_repeat_failure_records": diagnostic["state_repeat_failure_records"],
                    "executable": str(binary_path)}
            except (OSError, ValueError, KeyError, TypeError) as error:
                checked[component][name] = {"status": "invalid_saved_identity", "error": str(error)}
    return checked


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def validate_bundle(jax, bundle, manifest):
    reference = manifest["reference"]
    if (bundle.get("schema_version") != "shared_serialized_dual_inputs_v1"
            or bundle.get("source_git_head") != reference["source_git_head"]
            or bundle.get("method") != reference["method"]
            or Path(bundle.get("initial_checkpoint", "")).name != Path(manifest["original_checkpoint"]).name):
        raise ValueError("bundle original method/source/checkpoint differs")
    ids = tuple(bundle.get("artifact_ids", ()))
    expected = tuple(reference["method"]["training_plan"]["training_artifact_ids"][:2])
    payloads = bundle.get("payloads", ())
    if len(ids) != 2 or len(set(ids)) != 2 or ids != expected or len(payloads) != 2:
        raise ValueError("two declared TRAIN artifact payloads required")
    fingerprints = []
    for payload in payloads:
        leaves = jax.tree_util.tree_leaves(payload["daily"])
        if not leaves or any(np.shape(value)[0] != 32 for value in leaves):
            raise ValueError("daily workload requires exactly 32 rows on every leaf")
        hashes = []
        for start in range(0, 32, 4):
            digest = hashlib.sha256()
            for value in leaves:
                array = np.ascontiguousarray(value[start:start + 4])
                digest.update(str((array.shape, array.dtype.str)).encode())
                digest.update(array.tobytes())
            hashes.append(digest.hexdigest())
        if len(set(hashes)) != 8:
            raise ValueError("daily32 must contain eight distinct actual micro4 chunks")
        for horizon in (7, 30):
            case = payload["cases"][horizon]
            if len(case) != 5 or np.shape(case[0]) != (3854,):
                raise ValueError("full recurrent five-input initial-state case required")
            if np.shape(case.days[0].continuous_state)[0] != horizon:
                raise ValueError("recurrent case horizon differs")
        fingerprints.append(hashes)
    return {"artifact_ids": ids, "daily_micro4_sha256": fingerprints}


def initial_architecture(jax, manifest, architecture):
    from scripts.hpc.run_shared_learning import load_learning_checkpoint
    from research.daily_coarse_graining.daily_operator.target_oriented_training import initialize_target_oriented_adamw
    from scripts.hpc.shared_full_state_ad_execution import json_identity
    if architecture == "original":
        saved = load_learning_checkpoint(Path(manifest["original_checkpoint"]), expected_method=manifest["reference"]["method"])
        if saved["completed_updates"] != 0:
            raise ValueError("original baseline must be update0")
        parameters, optimizer, identity = saved["parameters"], saved["optimizer"], None
        count, leaves = 64884481, 300
    elif architecture == "candidate":
        saved = load_pickle(manifest["candidate_parameters"])
        parameters, identity = saved["parameters"], saved["identity"]
        if (identity.get("source_head") != manifest["reference"]["source_git_head"]
                or json_identity(identity.get("input_source_method")) != json_identity(manifest["reference"]["method"])
                or identity.get("seed") != manifest["reference"]["method"]["daily_protocol"]["seed"]
                or identity.get("artifact_id") != manifest["reference"]["method"]["training_plan"]["training_artifact_ids"][0]):
            raise ValueError("candidate source inputs/method differ from original workload")
        optimizer = initialize_target_oriented_adamw(jax.device_put(parameters))
        count, leaves = 66123713, 355
    else:
        raise ValueError("unsupported architecture")
    arrays = jax.tree_util.tree_leaves(parameters)
    if (len(arrays) != leaves or sum(np.size(value) for value in arrays) != count
            or any(np.asarray(value).dtype != np.dtype("float32") or not np.all(np.isfinite(value)) for value in arrays)):
        raise ValueError("fresh architecture FP32 parameter tree invalid")
    return jax.device_put(parameters), jax.device_put(optimizer), identity


def load_components(jax, manifest, architecture, parameters, payload, identity, *, components):
    from scripts.hpc.shared_full_state_ad_execution import (
        CANDIDATE_ARCHITECTURE, FullStateADEvaluator, load_full_state_executable)
    from scripts.hpc.run_shared_learning import memory_sizes
    result, metadata = {}, {}
    for component in components:
        began = time.perf_counter()
        item = manifest["architectures"][architecture][component]
        if component == "daily":
            sample = jax.tree_util.tree_map(lambda value: value[:4], payload["daily"])
            arguments = (parameters, sample.inputs.continuous_state, sample)
        else:
            arguments = (parameters, *payload["cases"][int(component[1:])])
        executable, metadata[component] = load_full_state_executable(jax, item["executable"], arguments,
            component=component, admission=item["report"],
            architecture=CANDIDATE_ARCHITECTURE if architecture == "candidate" else "original",
            expected_identity=identity, expected_reference=manifest["reference"],
            admission_mode=manifest.get("admission_mode", "scientific"),
            expected_turnover_sha256=manifest.get("performance_diagnostic", {}).get("turnover_sha256"),
            corrected_daily_verification=item.get("verification_report"),
            expected_method_sha256=manifest.get("performance_diagnostic", {}).get("method_sha256"),
            expected_prepared_context_identity=manifest.get("performance_diagnostic", {}).get(
                "prepared_context_identity"),
            memory_sizes=memory_sizes, compile_guard=forbid_model_compile,
            memory_limit_bytes=MEMORY_LIMIT)
        metadata[component]["source_executable"] = item["executable"]
        metadata[component]["load_and_ABI_seconds"] = time.perf_counter() - began
        result[component] = FullStateADEvaluator(jax, executable, component, compile_guard=forbid_model_compile)
    return result, metadata


def attach_states(component, evaluator, *, daily=False):
    """Preserve raw micro4 mean AD and the corresponding full daily32 mean AD."""
    records = evaluator.take_records()
    expected = 8 if daily else 1
    if len(records) != expected:
        raise ValueError("complete update state AD call count differs")
    gradients = [record["initial_state_gradient"] for record in records]
    if daily:
        raw = np.stack(gradients)
        component["initial_state_gradients_micro4_mean"] = raw
        component["initial_state_gradient"] = np.concatenate(gradients, axis=0) / 8
    else:
        component["initial_state_gradient"] = gradients[0]
    component["full_state_ad_calls"] = len(records)
    state = component["initial_state_gradient"]
    component["initial_state_AD"] = {"shape": list(state.shape), "dtype": str(state.dtype),
        "finite": bool(np.all(np.isfinite(state))), "nonzero": int(np.count_nonzero(state)),
        "coordinates": int(state.size)}
    return component


def stateful_branch(branch, evaluator, *, daily=False):
    def call(*args):
        began = time.perf_counter()
        evaluator.reset()
        result = branch(*args)
        if daily:
            attach_states(result, evaluator, daily=True)
            result["seconds"] = time.perf_counter() - began
            return result
        components, seconds = result
        attach_states(components[int(evaluator.component[1:])], evaluator)
        elapsed = time.perf_counter() - began
        components[int(evaluator.component[1:])]["seconds"] = elapsed
        return components, elapsed
    return call


def comparable_components(components, *, include_state=True, only_state=False):
    state_keys = ("initial_state_gradient", "initial_state_gradients_micro4_mean")
    keys = state_keys if only_state else ("gradient", "loss", "norm", "hard", "auxiliary", "full_state_ad_calls")
    if include_state and not only_state:
        keys += state_keys
    return {name: {key: value for key, value in component.items() if key in keys}
            for name, component in components.items()}


def state_snapshot(values):
    return {name: {key: value for key, value in component.items()
                   if key in ("loss", "auxiliary", "initial_state_gradient", "initial_state_gradients_micro4_mean", "full_state_ad_calls", "initial_state_AD")}
            for name, component in values["components"].items()}


def record_state_ad(record, values):
    record["initial_state_AD"] = {name: component["initial_state_AD"]
        for name, component in values["components"].items()}
    return record


def compare_complete(jax, serial, dual, *, allow_state_value_failure=False):
    from scripts.hpc.probe_shared_execution import compare_trees
    comparisons = {name: compare_trees(jax, serial[name], dual[name], rtol=RTOL, atol=ATOL)
                   for name in ("gradient", "parameters", "optimizer")}
    comparisons["components_except_state_gradients"] = compare_trees(jax,
        comparable_components(serial["components"], include_state=False),
        comparable_components(dual["components"], include_state=False), rtol=RTOL, atol=ATOL)
    comparisons["raw_state_gradients"] = compare_trees(jax,
        comparable_components(serial["components"], only_state=True),
        comparable_components(dual["components"], only_state=True), rtol=RTOL, atol=ATOL)
    comparisons["raw_state_gradients_finite"] = {"passed": all(
        np.all(np.isfinite(np.asarray(leaf)))
        for values in (serial, dual)
        for component in values["components"].values()
        for key in ("initial_state_gradient", "initial_state_gradients_micro4_mean")
        if key in component
        for leaf in jax.tree_util.tree_leaves(component[key]))}
    summaries = [{name: {key: value for key, value in component["initial_state_AD"].items() if key != "nonzero"}
                  for name, component in values["components"].items()}
                 for values in (serial, dual)]
    comparisons["initial_state_AD_summary"] = {"passed": summaries[0] == summaries[1],
        "scope": "exact shape/dtype/finite/coordinate metadata; nonzero counts reported, raw arrays use declared tolerance"}
    comparisons["non_state_parity_passed"] = all(
        value["passed"] for name, value in comparisons.items()
        if name not in ("raw_state_gradients", "raw_state_gradients_finite"))
    comparisons["state_value_failure_only"] = bool(
        comparisons["non_state_parity_passed"]
        and comparisons["raw_state_gradients_finite"]["passed"]
        and not comparisons["raw_state_gradients"]["passed"])
    comparisons["strict_parity_passed"] = all(
        value["passed"] for name, value in comparisons.items()
        if name not in ("non_state_parity_passed", "state_value_failure_only", "strict_parity_passed"))
    if (not comparisons["strict_parity_passed"]
            and not (allow_state_value_failure and comparisons["state_value_failure_only"])):
        raise FullStateParityFailure(comparisons)
    return comparisons


class SerialFullStateExecutor:
    """Public execute_learning_update with bounded component/state capture."""
    def __init__(self, *, jax, jnp, parameters, optimizer, evaluators, operations, update):
        from scripts.hpc.shared_daily_microbatch import make_daily_accumulator
        self.jax, self.parameters, self.optimizer = jax, parameters, optimizer
        self.evaluators, self.operations, self.update = evaluators, operations, update
        self.accumulate = make_daily_accumulator(jax=jax, jnp=jnp, microbatch_size=4,
            evaluator=evaluators["daily"], expected_count=32)

    def step(self, *, daily_payload, recurrent_payload, persist):
        from scripts.hpc.run_shared_learning import execute_learning_update
        began = time.perf_counter()
        captured = {}
        for evaluator in self.evaluators.values():
            evaluator.reset()
        def capture(name, output):
            (loss, auxiliary), gradient = output
            captured[name] = {"loss": float(loss), "auxiliary": self.jax.device_get(auxiliary),
                "gradient": gradient, "norm": float(self.operations["norm"](gradient)),
                "hard": {key: int(np.asarray(value)) for key, value in auxiliary["hard"].items()}}
            return output
        def daily(parameters, samples):
            return capture("daily", self.accumulate(parameters, samples))
        evaluators = {name: (lambda parameters, *case, name=name:
            capture(name, self.evaluators[name](parameters, *case))) for name in ("T7", "T30")}
        combined = {}
        operations = dict(self.operations)
        def combine(*gradients):
            combined["gradient"] = self.operations["combine_daily_mean"](*gradients)
            return combined["gradient"]
        operations["combine_daily_mean"] = combine
        parameters, optimizer, record = execute_learning_update(jax=self.jax,
            parameters=self.parameters, optimizer=self.optimizer, daily_samples=daily_payload,
            seven_case=recurrent_payload[7], thirty_case=recurrent_payload[30],
            evaluators=evaluators, operations=operations, update=self.update, daily_accumulator=daily)
        self.jax.block_until_ready((parameters, optimizer, combined["gradient"]))
        if int(np.asarray(optimizer.step)) != int(np.asarray(self.optimizer.step)) + 1:
            raise ValueError("serial optimizer version progression differs")
        for name in ("daily", "T7", "T30"):
            attach_states(captured[name], self.evaluators[name], daily=name == "daily")
        values = {"parameters": parameters, "optimizer": optimizer,
                  "gradient": combined["gradient"], "components": captured}
        if persist:
            self.parameters, self.optimizer = parameters, optimizer
        record["timing_seconds"]["total"] = time.perf_counter() - began
        return record_state_ad(record, values), values


class TransientThirtyEvaluator:
    """Release serial-only T30 binary after each public serial gate.

    Loading is outside matched dual proposal timing. Its overhead belongs to
    serial validation and is reported there, never credited as throughput.
    """
    component = "T30"
    def __init__(self, jax, manifest, architecture, identity):
        self.jax, self.manifest, self.architecture, self.identity = jax, manifest, architecture, identity
        self.records = ()
        self.last_metadata = None

    def reset(self):
        self.records = ()

    def take_records(self):
        records, self.records = self.records, ()
        return records

    def __call__(self, parameters, *case):
        evaluators, metadata = load_components(self.jax, self.manifest, self.architecture, parameters,
            {"cases": {30: case}}, self.identity, components=("T30",))
        evaluator = evaluators["T30"]
        output = evaluator(parameters, *case)
        self.records = evaluator.take_records()
        self.last_metadata = metadata["T30"]
        return output


def worker_class():
    from scripts.hpc.probe_shared_serialized_dual import RecurrentWorker
    class FullStateWorker(RecurrentWorker):
        def __init__(self, manifest, architecture, *, cuda_device, parameters, case, identity, echo_test=False):
            if not cuda_device or "," in cuda_device:
                raise ValueError("worker needs one explicit CUDA selector")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = cuda_device
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            command = [sys.executable, "-B", str(Path(__file__).with_name("probe_shared_latent_dual_worker.py")),
                "--manifest", str(manifest), "--architecture", architecture]
            if echo_test:
                command.append("--echo-test")
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None, env=environment, bufsize=0,
                cwd=os.environ.get("ORCHIDEE_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
            self.closed = False
            self.parameter_version = 0
            pickle.dump({"kind": "initialize", "parameters": parameters, "case": case, "identity": identity},
                        self.process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
            self.process.stdin.flush()
            ready = self._receive()
            if ready.get("status") != "ready":
                self.close()
                raise RuntimeError(f"full-state worker startup failed: {ready}")
            self.metadata = ready
            self.active_architecture = architecture
            self.architecture_metadata = {architecture: ready}

        def select(self, architecture, *, parameters=None, case=None, identity=None):
            if architecture == self.active_architecture:
                return 0.
            began = time.perf_counter()
            message = {"kind": "select", "architecture": architecture}
            if architecture not in self.architecture_metadata:
                if parameters is None or case is None:
                    raise ValueError("first architecture selection requires full initial ABI arguments")
                message.update(parameters=parameters, case=case, identity=identity)
            pickle.dump(message, self.process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
            self.process.stdin.flush()
            ready = self._receive()
            if ready.get("status") != "ready" or ready.get("architecture") != architecture:
                raise RuntimeError(f"full-state worker architecture selection failed: {ready}")
            self.metadata = ready
            self.active_architecture = architecture
            self.architecture_metadata[architecture] = ready
            return time.perf_counter() - began

        def _receive(self):
            response = super()._receive()
            if response.get("status") == "passed" and self.metadata.get("mode") != "echo_test":
                if response.get("executed_architecture") != self.active_architecture:
                    raise RuntimeError("worker executed a different architecture")
            return response
    return FullStateWorker


def require_memory(memory, *, label):
    peak = (memory or {}).get("peak_bytes_in_use")
    if peak is None or not np.isfinite(peak) or peak > MEMORY_LIMIT:
        raise MemoryError(f"{label} observed process memory peak missing/exceeds24GiB")


def changed_weight_timing_summary(rows):
    """Report all paired steps and the post-first-step subset independently."""
    pairs = {}
    for row in rows:
        pair = pairs.setdefault(row["chain_step"], {})
        if row["architecture"] in pair:
            raise ValueError("duplicate architecture in changed-weight timing pair")
        pair[row["architecture"]] = row
    indices = sorted(pairs)
    if indices != list(range(1, len(indices) + 1)):
        raise ValueError("changed-weight timing step indices are not contiguous")
    for index, pair in pairs.items():
        if set(pair) != {"original", "candidate"}:
            raise ValueError("changed-weight timing requires both full-state dual architectures")
        if len({item["artifact_id"] for item in pair.values()}) != 1:
            raise ValueError("changed-weight timing pair payload differs")
        for row in pair.values():
            if row["input_parameter_version"] != index - 1 or row["worker_executed_parameter_version"] != index - 1:
                raise ValueError("changed-weight timing component parameter version differs")
            if not np.isfinite(row["complete_seconds"]) or row["complete_seconds"] <= 0:
                raise ValueError("changed-weight timing must be finite and positive")
    def summarize(selected):
        seconds = {name: [pairs[index][name]["complete_seconds"] for index in selected]
                   for name in ("original", "candidate")}
        medians = {name: float(np.median(value)) if value else None for name, value in seconds.items()}
        ratio = medians["original"] / medians["candidate"] if selected else None
        return {"pair_indices": selected, "paired_samples": len(selected), "seconds": seconds,
                "medians": medians, "original_over_candidate": ratio,
                "two_card_hours": {name: 2 * sum(value) / 3600 for name, value in seconds.items()}}
    return {"paired_complete_steps": len(indices), "minimum_three_pairs_complete": len(indices) >= 3,
            "first_step_warm_boundary_index": indices[0] if indices else None,
            "first_step_input_parameter_version": 0 if indices else None,
            "all_paired_complete_updates": summarize(indices),
            "post_first_step_changed_weights": summarize(indices[1:]),
            "post_first_stability_status": ("at_least_three_post_first_pairs" if len(indices) >= 4
                else "limited_two_post_first_pairs" if len(indices) == 3 else "bounded_less_than_two_post_first_pairs"),
            "warm_boundary_policy": "first retained update uses update0 weights after fixed warmup; all and post-first timings reported separately"}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--worker-cuda-device", required=True)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--changing-updates", type=int, default=4,
        help="paired original/candidate dual retained steps; first step is disclosed as warm boundary")
    result.add_argument("--equivalence-updates", type=int, default=2,
        help="independent fresh candidate serial/dual retained equivalence steps")
    return result


def validate_args(args):
    if args.repeats < 3 or not 0 <= args.changing_updates <= 16 or not 0 <= args.equivalence_updates <= 16:
        raise ValueError(">=3 matched pairs and bounded changing steps per chain required")
    if 2 + args.repeats + args.changing_updates + 2 * args.equivalence_updates > 16:
        raise ValueError("candidate16-call cap includes update0/fixed proposals, dual performance and independent equivalence chains")
    if args.output.exists():
        raise FileExistsError(args.output)
    for suffix in (".state-diagnostics.pkl", ".candidate-serial-diagnostic.pkl", ".candidate-dual-diagnostic.pkl",
                   ".original-dual-performance.pkl", ".candidate-dual-performance.pkl",
                   ".original-serial-diagnostic.pkl", ".original-dual-diagnostic.pkl"):
        if args.output.with_suffix(suffix).exists():
            raise FileExistsError(args.output.with_suffix(suffix))
    selector = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not selector or "," in selector or "," in args.worker_cuda_device or selector == args.worker_cuda_device:
        raise ValueError("distinct explicit single parent/worker CUDA selectors required")


def atomic_pickle(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args):
    started = time.perf_counter()
    validate_args(args)
    manifest = read_manifest(args.manifest)
    readiness = validate_manifest_asset_readiness(manifest)
    if readiness["status"] != "ready_for_saved_graph_load":
        raise FileNotFoundError(json.dumps(readiness, sort_keys=True))
    runtime = load_adjacent("orcjax_gpu_runtime")
    jax, devices = runtime.initialize_gpu_backend()
    jax.config.update("jax_enable_x64", True)
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("parent requires exactly one visible GPU")
    import jax.numpy as jnp
    load_runtime_overlays()
    from scripts.hpc.training_artifact_io import _atomic_json
    from scripts.hpc import run_shared_learning as learning
    from scripts.hpc.probe_shared_serialized_dual import RemoteDualExecutor
    from scripts.hpc.probe_shared_dual import make_microbatch_daily_branch, make_recurrent_branch, make_component_combiner
    from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
    bundle = load_pickle(manifest["input_bundle"])
    workload = validate_bundle(jax, bundle, manifest)
    payloads = bundle["payloads"]
    device = devices[0]
    report = {"status": "running", "scope": "original_candidate_matched_dual_full_parameter_and_initial_state_AD",
        "scientific_acceptance": False, "production_adoption": False, "formal_training_updates": 0,
        "model_gradient_compilations": 0, "gpu_count": 2, "rtol": RTOL, "atol": ATOL,
        "workload": workload, "schedule": "parent daily32 eight distinct micro4 + T7; worker T30",
        "state_gradient_policy": "raw micro4 mean derivatives retained; concatenate eight then divide8 for daily32 mean",
        "manifest": str(args.manifest), "parent_cuda_selector": os.environ["CUDA_VISIBLE_DEVICES"],
        "worker_cuda_selector": args.worker_cuda_device, "sealed_test_access": False, "teacher_regeneration": False,
        "candidate_actual_retained_optimizer_updates": 0, "reference": manifest["reference"],
        "corrected_component_identity_checks": readiness.get("corrected_component_identity_checks", {})}
    corrected_mode = manifest.get("admission_mode") == CORRECTED_PERFORMANCE_DIAGNOSTIC
    inherited_state_failures = []
    if corrected_mode:
        for component, architectures in readiness.get("corrected_component_identity_checks", {}).items():
            for architecture, check in architectures.items():
                for failure in check.get("state_repeat_failure_records", ()):
                    inherited_state_failures.append({"architecture": architecture,
                        "component": component, "source_report_status": check.get("report_status"),
                        "source_state_repeat_failure": failure})
    report["inherited_strict_state_repeat_failures"] = inherited_state_failures
    report["strict_state_repeat_failures"] = list(inherited_state_failures)
    report.update(candidate_optimizer_calls_attempted=0, candidate_optimizer_calls_completed=0,
        original_optimizer_calls_attempted=0, original_optimizer_calls_completed=0,
        original_actual_retained_optimizer_updates=0,
        declared_original_optimizer_call_upper_bound=(2 + args.repeats + args.changing_updates
            + (2 * args.equivalence_updates if corrected_mode else 0)),
        declared_candidate_optimizer_call_upper_bound=2 + args.repeats + args.changing_updates + 2 * args.equivalence_updates,
        candidate_call_cap_scope="discarded update0/fixed proposals + retained dual performance + independent serial/dual equivalence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    resources = []
    executors, serials, metadata = {}, {}, {}
    state_diagnostics = {"update0": {}, "dual_performance_changing": [],
                         "original_equivalence": [], "candidate_equivalence": []}
    worker = None
    equivalence_duals = {}
    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **fields)
        _atomic_json(args.output, report)
        print(f"latent_dual stage={stage}", flush=True)
    def persist_comparison(serial_values, dual_values, architecture, stage):
        diagnostic_mode = manifest.get("admission_mode") == CORRECTED_PERFORMANCE_DIAGNOSTIC
        comparison = compare_complete(jax, serial_values, dual_values,
            allow_state_value_failure=diagnostic_mode)
        if comparison["state_value_failure_only"]:
            report.setdefault("strict_state_repeat_failures", []).append({
                "architecture": architecture, "stage": stage,
                "raw_state_gradients": comparison["raw_state_gradients"]})
        return comparison
    def step(executor, architecture, **kwargs):
        report[architecture + "_optimizer_calls_attempted"] += 1
        if report[architecture + "_optimizer_calls_attempted"] > 16:
            raise ValueError(architecture + " total diagnostic optimizer-call cap exceeded")
        result = executor.step(**kwargs)
        report[architecture + "_optimizer_calls_completed"] += 1
        if kwargs.get("persist"):
            report[architecture + "_actual_retained_optimizer_updates"] += 1
        return result
    def save_chains():
        paths, steps = {}, {}
        chains = (("original_dual_performance", executors.get("original"), ".original-dual-performance.pkl"),
                  ("candidate_dual_performance", executors.get("candidate"), ".candidate-dual-performance.pkl"),
                  ("original_serial_equivalence", serials.get("original"), ".original-serial-diagnostic.pkl"),
                  ("original_dual_equivalence", equivalence_duals.get("original"), ".original-dual-diagnostic.pkl"),
                  ("candidate_serial_equivalence", serials.get("candidate"), ".candidate-serial-diagnostic.pkl"),
                  ("candidate_dual_equivalence", equivalence_duals.get("candidate"), ".candidate-dual-diagnostic.pkl"))
        for name, executor, suffix in chains:
            if executor is None:
                continue
            completed = int(np.asarray(executor.optimizer.step))
            steps[name] = completed
            if not completed:
                continue
            path = args.output.with_suffix(suffix)
            if not path.exists():
                atomic_pickle(path, {"scope": "full_state_diagnostic_not_formal_training", "chain": name, "manifest": manifest,
                    "parameters": jax.device_get(executor.parameters), "optimizer": jax.device_get(executor.optimizer),
                    "completed_diagnostic_updates": completed,
                    "scientific_acceptance": False, "production_adoption": False})
            paths[name] = str(path)
        report["retained_diagnostic_chains"] = paths
        report["final_diagnostic_optimizer_steps"] = steps
        report["candidate_actual_retained_optimizer_updates"] = sum(value for name, value in steps.items() if name.startswith("candidate_"))
        report["original_actual_retained_optimizer_updates"] = sum(
            value for name, value in steps.items() if name.startswith("original_"))
    def save_states():
        state_path = args.output.with_suffix(".state-diagnostics.pkl")
        if not state_path.exists():
            atomic_pickle(state_path, {"scope": "complete_native_initial_state_AD_diagnostic_snapshots",
                "manifest": manifest, "diagnostics": state_diagnostics})
        report["state_diagnostics_artifact"] = str(state_path)
    try:
        emit("load_full_state_binaries")
        operations = learning.make_tree_operations(jax, jnp, 32)
        update = make_update(manifest["reference"]["method"]["daily_protocol"])
        combine = make_component_combiner(jax)
        for architecture in ("original", "candidate"):
            parameters, optimizer, identity = initial_architecture(jax, manifest, architecture)
            local_payload = jax.device_put(payloads[0], device)
            evaluators, metadata[architecture] = load_components(jax, manifest, architecture, parameters,
                local_payload, identity, components=("daily", "T7"))
            evaluators["T30"] = TransientThirtyEvaluator(jax, manifest, architecture, identity)
            serials[architecture] = SerialFullStateExecutor(jax=jax, jnp=jnp, parameters=parameters,
                optimizer=optimizer, evaluators=evaluators, operations=operations, update=update)
            if worker is None:
                worker_started = time.perf_counter()
                worker = worker_class()(args.manifest, architecture, cuda_device=args.worker_cuda_device,
                    parameters=jax.device_get(parameters), case=payloads[0]["cases"][30], identity=identity)
                resources.append(worker)
                report["initial_worker_startup_seconds"] = time.perf_counter() - worker_started
            else:
                report.setdefault("architecture_selection_seconds", []).append(worker.select(architecture,
                    parameters=jax.device_get(parameters), case=payloads[0]["cases"][30], identity=identity))
            daily = stateful_branch(make_microbatch_daily_branch(jax=jax, jnp=jnp, device=device,
                evaluator=evaluators["daily"], operations=operations, microbatch_size=4,
                expected_count=32, preserve_auxiliary=True), evaluators["daily"], daily=True)
            seven = stateful_branch(make_recurrent_branch(jax=jax, device=device,
                evaluators={7: evaluators["T7"]}, operations=operations, horizons=(7,),
                preserve_auxiliary=True), evaluators["T7"])
            executor = RemoteDualExecutor(jax=jax, parameters=parameters, optimizer=optimizer,
                daily_branch=daily, worker=worker, combine=combine, update=update,
                finite=operations["finite"], local_recurrent_branch=seven,
                preserve_components=True, audit_parameter_versions=True)
            resources.append(executor)
            executors[architecture] = executor
            metadata[architecture]["worker"] = worker.metadata
            if (worker.metadata.get("architecture") != architecture
                    or worker.metadata["executables"]["T30"]["source_executable"] != manifest["architectures"][architecture]["T30"]["executable"]):
                raise ValueError("worker full-state architecture/executable association differs")
            # One same-architecture public serial/dual full proposal gate at update0.
            serial_record, serial_values = step(serials[architecture], architecture, daily_payload=local_payload["daily"],
                recurrent_payload=local_payload["cases"], persist=False)
            dual_record, dual_values = step(executor, architecture, daily_payload=local_payload["daily"],
                recurrent_payload=local_payload["cases"], persist=False)
            record_state_ad(dual_record, dual_values)
            report.setdefault("same_architecture_update0", {})[architecture] = persist_comparison(
                serial_values, dual_values, architecture, "update0")
            report.setdefault("update0_initial_state_AD", {})[architecture] = {
                "serial": serial_record["initial_state_AD"], "dual": dual_record["initial_state_AD"]}
            state_diagnostics["update0"][architecture] = {
                "serial": state_snapshot(serial_values), "dual": state_snapshot(dual_values)}
            metadata[architecture]["serial_T30"] = evaluators["T30"].last_metadata
            if architecture == "original" and manifest.get("admission_mode") != CORRECTED_PERFORMANCE_DIAGNOSTIC:
                del serials[architecture]
            del serial_values, dual_values
            emit("architecture_ready", executable_metadata=metadata)
        report["graphs_and_update0_parity_ready_seconds"] = time.perf_counter() - started
        report["prior_warm_coverage"] = {
            "each_architecture_public_serial_full_state_update0_proposals": 1,
            "each_architecture_dual_full_state_update0_proposals": 1,
            "public_finite_norm_accumulation_combine_and_AdamW_helpers": "exercised by these complete update0 proposals",
            "external_runtime_preflight_report": manifest.get("runtime_preflight_report"),
            "runtime_preflight_execution": "owned by execution wrapper; runner does not rerun or reset its budget"}
        timings = {"original": [], "candidate": []}
        fixed_rows = []
        for repeat in range(args.repeats):
            for architecture in (("original", "candidate") if repeat % 2 == 0 else ("candidate", "original")):
                payload = jax.device_put(payloads[0], device)
                # One worker context; selection/loading is explicit startup
                # overhead, outside complete current-tree proposal timing.
                report.setdefault("architecture_selection_seconds", []).append(worker.select(architecture))
                began = time.perf_counter()
                record, values = step(executors[architecture], architecture, daily_payload=payload["daily"],
                    recurrent_payload=payload["cases"], persist=False)
                record_state_ad(record, values)
                jax.block_until_ready((values["parameters"], values["optimizer"]))
                seconds = time.perf_counter() - began
                timings[architecture].append(seconds)
                if "time_to_first_valid_matched_measurement_seconds" not in report:
                    report["time_to_first_valid_matched_measurement_seconds"] = time.perf_counter() - started
                fixed_rows.append({"repeat": repeat, "architecture": architecture, "complete_seconds": seconds,
                    "loss": record["loss"], "timing_seconds": record["timing_seconds"],
                    "initial_state_AD": record["initial_state_AD"],
                    "worker_request_response_timings": record["worker_request_response_timings"]})
                del values
                require_memory(device.memory_stats(), label="parent")
                require_memory(executors[architecture].worker.last_device_memory, label="worker")
                emit("matched_complete_proposal", fixed_rows=fixed_rows)
        medians = {name: float(np.median(values)) for name, values in timings.items()}
        speedup = medians["original"] / medians["candidate"]
        report.update(fixed_screen_complete_update_seconds=timings, fixed_screen_complete_update_medians=medians,
            fixed_screen_same_workload_speedup=speedup, fixed_screen_passed=speedup >= 1.5,
            fixed_screen_two_card_hours={name: 2 * sum(values) / 3600 for name, values in timings.items()},
            fixed_screen_scope="cheap >=1.5 fixed-update0 proposal screen; not final changed-weight speed acceptance")
        report["discarded_optimizer_proposals"] = {name: args.repeats + 2 for name in ("original", "candidate")}
        report["prior_warm_coverage"]["each_architecture_fixed_update0_matched_proposals_before_changing"] = args.repeats
        emit("fixed_screen_measured")
        performance_rows, equivalence_rows = [], []
        diagnostic_mode = manifest.get("admission_mode") == CORRECTED_PERFORMANCE_DIAGNOSTIC
        measurement_admitted = speedup >= 1.5 or diagnostic_mode
        changing_updates = args.changing_updates if measurement_admitted else 0
        equivalence_updates = args.equivalence_updates if measurement_admitted else 0
        report["requested_dual_performance_steps_per_architecture"] = args.changing_updates
        report["executed_dual_performance_steps_per_architecture"] = changing_updates
        report["requested_candidate_equivalence_steps_per_chain"] = args.equivalence_updates
        report["executed_candidate_equivalence_steps_per_chain"] = equivalence_updates
        report["measurement_admission_reason"] = (
            "explicit_corrected_performance_diagnostic" if diagnostic_mode else
            "fixed_screen_at_least_1_5x" if measurement_admitted else
            "fixed_screen_below_1_5x")
        report["fixed_screen_gate_applies_to_followup_measurements"] = not diagnostic_mode
        if manifest.get("admission_mode") == CORRECTED_PERFORMANCE_DIAGNOSTIC:
            report["requested_original_equivalence_steps_per_chain"] = args.equivalence_updates
            report["executed_original_equivalence_steps_per_chain"] = equivalence_updates
            report["serial_dual_equivalence_scope"] = (
                "independent same-architecture serial/dual parameter-version chains for original and candidate; "
                "cross-architecture gradient trees are not compared")
        # Both original and candidate advance saved full-state dual binaries.
        # Historical parameter-only changing evidence is not substituted here.
        for index in range(changing_updates):
            payload = jax.device_put(payloads[index % len(payloads)], device)
            for architecture in (("original", "candidate") if index % 2 == 0 else ("candidate", "original")):
                executor = executors[architecture]
                report.setdefault("architecture_selection_seconds", []).append(worker.select(architecture))
                before = int(np.asarray(executor.optimizer.step))
                began = time.perf_counter()
                record, values = step(executor, architecture, daily_payload=payload["daily"],
                    recurrent_payload=payload["cases"], persist=True)
                record_state_ad(record, values)
                jax.block_until_ready((values["parameters"], values["optimizer"]))
                elapsed = time.perf_counter() - began
                state_diagnostics["dual_performance_changing"].append({"chain_step": index + 1,
                    "architecture": architecture, "components": state_snapshot(values)})
                parent_memory = device.memory_stats()
                worker_memory = worker.last_device_memory
                require_memory(parent_memory, label="parent")
                require_memory(worker_memory, label="shared worker")
                performance_rows.append({"chain_step": index + 1, "architecture": architecture,
                    "artifact_id": workload["artifact_ids"][index % 2], "input_parameter_version": before,
                    "worker_executed_parameter_version": worker.last_parameter_version,
                    "complete_seconds": elapsed, "first_step_warm_boundary": index == 0, "record": record,
                    "parameter_and_optimizer_finite": True,
                    "physical_hard_counts_zero": record["all_hard_counts_zero"],
                    "parent_device_memory": parent_memory, "worker_device_memory": worker_memory})
                del values
                emit("full_state_dual_performance_step", changed_weight_performance_rows=performance_rows)
        report["changed_weight_timing"] = changed_weight_timing_summary(performance_rows)
        timing_complete = report["changed_weight_timing"]["minimum_three_pairs_complete"]
        changed_ratio = report["changed_weight_timing"]["all_paired_complete_updates"]["original_over_candidate"]
        steady_ratio = report["changed_weight_timing"]["post_first_step_changed_weights"]["original_over_candidate"]
        report["changed_weight_speed_gate_passed"] = bool(
            timing_complete and changed_ratio >= 1.5 and steady_ratio is not None and steady_ratio >= 1.5)
        report["changed_weight_timing_status"] = "complete_minimum_three_pairs" if timing_complete else "bounded_incomplete_minimum_three_pairs"
        emit("changed_weight_timing_measured")
        # Independent candidate serial/dual equivalence starts afresh from the
        # untouched update0 serial chain, rather than the performance weights.
        equivalence_architectures = (("original", "candidate")
            if manifest.get("admission_mode") == CORRECTED_PERFORMANCE_DIAGNOSTIC else ("candidate",))
        equivalence_rows = {architecture: [] for architecture in equivalence_architectures}
        if equivalence_updates:
            for architecture in equivalence_architectures:
                serial = serials[architecture]
                performance = executors[architecture]
                equivalence_dual = RemoteDualExecutor(jax=jax, parameters=serial.parameters,
                    optimizer=serial.optimizer, daily_branch=performance.daily_branch, worker=worker,
                    combine=combine, update=update, finite=operations["finite"],
                    local_recurrent_branch=performance.local_recurrent_branch,
                    preserve_components=True, audit_parameter_versions=True)
                resources.append(equivalence_dual)
                equivalence_duals[architecture] = equivalence_dual
                report.setdefault("architecture_selection_seconds", []).append(worker.select(architecture))
                for index in range(equivalence_updates):
                    payload = jax.device_put(payloads[index % len(payloads)], device)
                    before = int(np.asarray(serial.optimizer.step))
                    serial_record, serial_values = step(serial, architecture,
                        daily_payload=payload["daily"], recurrent_payload=payload["cases"], persist=True)
                    dual_record, dual_values = step(equivalence_dual, architecture,
                        daily_payload=payload["daily"], recurrent_payload=payload["cases"], persist=True)
                    record_state_ad(dual_record, dual_values)
                    state_diagnostics[architecture + "_equivalence"].append({"chain_step": index + 1,
                        "serial": state_snapshot(serial_values), "dual": state_snapshot(dual_values)})
                    comparison = persist_comparison(serial_values, dual_values, architecture,
                        architecture + "_independent_serial_dual_step_" + str(index + 1))
                    equivalence_rows[architecture].append({"architecture": architecture,
                        "chain_step": index + 1, "artifact_id": workload["artifact_ids"][index % 2],
                        "input_parameter_version": before,
                        "worker_executed_parameter_version": worker.last_parameter_version,
                        "serial": serial_record, "dual": dual_record, "comparison": comparison})
                    del serial_values, dual_values
                    require_memory(device.memory_stats(), label="parent")
                    require_memory(worker.last_device_memory, label="shared worker")
                    emit(architecture + "_independent_equivalence_step",
                         architecture_equivalence_rows=equivalence_rows)
        report["independent_candidate_changing_equivalence_passed"] = len(equivalence_rows) >= 2
        # Legacy route uses a list of candidate replay rows; corrected route
        # preserves both architecture-specific lists without cross-comparison.
        if manifest.get("admission_mode") == CORRECTED_PERFORMANCE_DIAGNOSTIC:
            report["independent_original_changing_equivalence_passed"] = (
                len(equivalence_rows.get("original", ())) >= 2)
            report["independent_candidate_changing_equivalence_passed"] = (
                len(equivalence_rows.get("candidate", ())) >= 2)
            replay_passed = all(
                len(equivalence_rows.get(name, ())) >= 2
                and all(row["comparison"]["strict_parity_passed"]
                        for row in equivalence_rows[name])
                for name in ("original", "candidate"))
        else:
            replay_passed = len(equivalence_rows.get("candidate", ())) >= 2
        report["same_architecture_serial_dual_replay_passed"] = replay_passed
        state_repeat_passed = not report.get("strict_state_repeat_failures")
        report["strict_state_repeat_admission_passed"] = state_repeat_passed
        report["final_changed_weight_speed_acceptance_passed"] = bool(
            report["changed_weight_speed_gate_passed"] and replay_passed and state_repeat_passed)
        report["final_speed_acceptance_conditions"] = {
            "fixed_update0_screen_ratio_at_least1_5": speedup >= 1.5,
            "at_least_three_paired_full_state_dual_retained_steps": timing_complete,
            "changed_weight_complete_update_median_ratio_at_least1_5": report["changed_weight_speed_gate_passed"],
            "all_paired_complete_update_median_ratio_at_least1_5": bool(changed_ratio is not None and changed_ratio >= 1.5),
            "post_first_steady_complete_update_median_ratio_at_least1_5": bool(steady_ratio is not None and steady_ratio >= 1.5),
            "independent_candidate_serial_dual_at_least_two_changing_steps":
                len(equivalence_rows.get("candidate", ())) >= 2,
            "independent_original_serial_dual_at_least_two_changing_steps":
                (len(equivalence_rows.get("original", ())) >= 2
                 if manifest.get("admission_mode") == CORRECTED_PERFORMANCE_DIAGNOSTIC else None),
            "all_run_ad_physical_memory_and_version_gates": True,
            "first_step_boundary_and_post_first_sample_count_disclosed": True}
        save_chains()
        save_states()
        report["parent_device_memory"] = device.memory_stats()
        report["worker_device_memory"] = {"shared_worker_process": worker.last_device_memory}
        require_memory(report["parent_device_memory"], label="parent")
        for name, memory in report["worker_device_memory"].items():
            require_memory(memory, label=name + " worker")
        final_status = "numerical_failed" if not state_repeat_passed else "passed"
        emit("completed", status=final_status, changed_weight_performance_rows=performance_rows,
             architecture_equivalence_rows=equivalence_rows)
        return report
    except Exception as error:
        report["final_changed_weight_speed_acceptance_passed"] = False
        if isinstance(error, FullStateParityFailure):
            report["failed_same_architecture_comparisons"] = error.comparisons
        try:
            save_chains()
            save_states()
        except Exception as save_error:
            report["diagnostic_preservation_error"] = {"type": type(save_error).__name__, "error": str(save_error)}
        emit("exception", status="error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for resource in reversed(resources):
            resource.close()


if __name__ == "__main__":
    run(parser().parse_args())
