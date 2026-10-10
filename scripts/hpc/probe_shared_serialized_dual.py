"""Measure complete serial and dual updates with the trusted split executables.

This diagnostic runs the unchanged two-artifact learning bootstrap and update
contract. It never lowers or compiles a model gradient; only small tree/update
operations may be JIT-compiled. A 64-row result combines two distinct real
32-row artifacts and is a fixed-weight proposal, not the formal batch recipe.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from scripts.hpc.checked_recurrent_execution import device_helper_cache
from scripts.hpc.probe_shared_execution import compare_trees
from scripts.hpc.probe_shared_split_profile import concatenate_daily


def _load_adjacent_overlay(name):
    """Load diagnostics absent or older in the immutable source checkout."""
    module_name = f"scripts.hpc.{name}"
    path = Path(__file__).with_name(name + ".py").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"required diagnostic overlay missing: {path}")
    existing = sys.modules.get(module_name)
    if existing is not None and Path(existing.__file__).resolve() == path:
        return
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[module_name]
        raise


try:
    from scripts.hpc.training_artifact_io import _atomic_json
except ModuleNotFoundError as error:
    if error.name != "scripts.hpc.training_artifact_io":
        raise
    _load_adjacent_overlay("training_artifact_io")
    from scripts.hpc.training_artifact_io import _atomic_json

_load_adjacent_overlay("probe_shared_dual")
_load_adjacent_overlay("probe_shared_serialized_updates")
_dual_module = sys.modules["scripts.hpc.probe_shared_dual"]
_serialized_module = sys.modules["scripts.hpc.probe_shared_serialized_updates"]
_serial_mixed = _dual_module._serial_mixed
dual_cost_comparison = _dual_module.dual_cost_comparison
make_component_combiner = _dual_module.make_component_combiner
make_microbatch_daily_branch = _dual_module.make_microbatch_daily_branch
make_recurrent_branch = _dual_module.make_recurrent_branch
compare_update_reports = _serialized_module.compare_update_reports
load_evaluators = _serialized_module.load_evaluators
require_reference = _serialized_module.require_reference
require_same_method = _serialized_module.require_same_method
admission_report_paths = _serialized_module.admission_report_paths
validate_admissions = _serialized_module.validate_admissions

RTOL = 1e-5
ATOL = 1e-6


def require_distinct_batches(jax, identifiers, left, right, *, rows=32):
    """Reject nominal 64-row coverage made from one artifact or wrong shapes."""
    if len(identifiers) != 2 or identifiers[0] == identifiers[1]:
        raise ValueError("64-row proposal requires two distinct artifact IDs")
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if left_tree != right_tree or not left_leaves:
        raise ValueError("daily artifact batches have incompatible trees")
    if any(np.shape(a)[0] != rows or np.shape(b)[0] != rows for a, b in zip(left_leaves, right_leaves)):
        raise ValueError("64-row proposal requires two complete real 32-row batches")


def _source_head():
    from scripts.hpc.run_shared_learning import ROOT

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _reference_and_profile(args):
    profile_bytes = args.source_profile.read_bytes()
    profile = json.loads(profile_bytes)
    reference = json.loads(Path(profile["technical_update_report"]).read_text(encoding="utf-8"))
    require_reference(profile, reference)
    return profile, reference, hashlib.sha256(profile_bytes).hexdigest()


def _initial_checkpoint(reference):
    candidates = [row for row in reference["checkpoints"]
                  if row["update"] == 0 and row["retained_milestone"]]
    if len(candidates) != 1:
        raise ValueError("reference must identify exactly one retained update-0 checkpoint")
    return Path(candidates[0]["path"])


def _validate_bundle(bundle, *, profile_sha256, reference, source_head):
    if bundle.get("schema_version") != "shared_serialized_dual_inputs_v1":
        raise ValueError("unrecognized compact diagnostic bundle")
    if (bundle["source_git_head"] != source_head
            or bundle["source_git_head"] != reference["source_git_head"]
            or bundle["profile_sha256"] != profile_sha256
            or bundle["method"] != reference["method"]):
        raise ValueError("compact bundle source/profile/method differs from reference")
    ids = tuple(reference["method"]["training_plan"]["training_artifact_ids"][:2])
    if tuple(bundle["artifact_ids"]) != ids or len(set(ids)) != 2:
        raise ValueError("compact bundle does not contain the two declared artifacts")
    if Path(bundle["initial_checkpoint"]).resolve() != _initial_checkpoint(reference).resolve():
        raise ValueError("compact bundle update-0 checkpoint differs from reference")
    if len(bundle["payloads"]) != 2:
        raise ValueError("compact bundle needs exactly two prepared payloads")
    return ids


def _prepare_compact_bundle(args, emit):
    """One-time free-CPU preparation; never initializes or compiles a model gradient."""
    from scripts.hpc import run_shared_learning as learning

    profile, reference, profile_sha256 = _reference_and_profile(args)
    source_head = _source_head()
    if source_head != reference["source_git_head"]:
        raise ValueError("prepare bundle in the original source checkout")
    if args.input_bundle.is_file():
        with args.input_bundle.open("rb") as handle:
            existing = pickle.load(handle)
        ids = _validate_bundle(existing, profile_sha256=profile_sha256,
                               reference=reference, source_head=source_head)
        return {"status": "passed", "bundle": str(args.input_bundle),
                "artifact_ids": ids, "source_git_head": source_head,
                "reused_existing_bundle": True}
    emit("load_learning_configuration")
    from research.daily_coarse_graining.daily_operator.shared_persistent_outputs import (
        build_shared_persistent_coordinates,
    )
    from research.daily_coarse_graining.daily_operator.shared_process_head import (
        build_shared_process_head,
    )
    from research.daily_coarse_graining.daily_operator.shared_recurrent_training import prepare_shared_window
    from scripts.hpc.shared_learning_plan import load_learning_configuration
    from scripts.hpc.shared_transformer_runtime import (
        _load_prepared_artifacts,
        load_shared_runtime,
        validate_shared_artifact_metadata,
    )
    loaded = load_learning_configuration(
        args.protocol, args.joint_admission_report,
        args.prepared_manifest, args.validation_manifest,
    )
    if learning._method_identity(loaded, technical_updates=2) != reference["method"]:
        raise ValueError("preparation arguments differ from the trusted technical method")
    emit("learning_configuration_validated")
    ids = tuple(reference["method"]["training_plan"]["training_artifact_ids"][:2])
    checkpoint = _initial_checkpoint(reference)
    learning.load_learning_checkpoint(checkpoint, expected_method=reference["method"])
    args.cpu_forward_only = True
    emit("load_shared_runtime", bootstrap_artifact_id=ids[0])
    runtime, parent, template, first, _ = load_shared_runtime(args, ids[:1])
    emit("shared_runtime_loaded")
    second, manifest = _load_prepared_artifacts(args.prepared_manifest, ids[1:])
    if manifest["identity"] != reference["method"]["training_plan"]["prepared_manifest_identity"]:
        raise ValueError("prepared manifest identity differs from reference")
    validate_shared_artifact_metadata(runtime["index"], ids[1:], second)
    coordinates = build_shared_persistent_coordinates(
        parent.learned_schema, parent.contract,
        runtime["base_statistics"].arrays["state_delta"],
    )
    architecture = reference["method"]["daily_protocol"]["architecture"]
    head = build_shared_process_head(
        parent, template, config=learning.shared_config(architecture),
        thermal_transform=architecture["thermal_transform"],
        persistent_coordinates=coordinates, source_bare_cimean=True,
    )
    import jax

    emit("prepare_windows", artifact_ids=ids)
    payloads = []
    window_index = reference["method"]["configuration"]["recurrent_window_index"]
    for artifact in (*first, *second):
        daily = artifact.batch.one_day
        if np.shape(daily.inputs.continuous_state)[0] != 32:
            raise ValueError("prepared artifact must contain exactly 32 daily rows")
        cases = {
            horizon: prepare_shared_window(
                head, getattr(artifact.batch, "seven_day" if horizon == 7 else "thirty_day"),
                window_index=window_index, horizon=horizon,
            )
            for horizon in (7, 30)
        }
        payloads.append({"daily": jax.device_get(daily),
                         "cases": jax.device_get(cases)})
    require_distinct_batches(jax, ids, payloads[0]["daily"], payloads[1]["daily"])
    emit("windows_prepared")
    bundle = {
        "schema_version": "shared_serialized_dual_inputs_v1",
        "source_git_head": source_head, "profile_sha256": profile_sha256,
        "method": reference["method"], "artifact_ids": ids,
        "initial_checkpoint": str(checkpoint.resolve()), "payloads": payloads,
    }
    _validate_bundle(bundle, profile_sha256=profile_sha256,
                     reference=reference, source_head=source_head)
    args.input_bundle.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="shared-dual-inputs-", dir=args.input_bundle.parent) as staging:
        temporary = Path(staging) / args.input_bundle.name
        with temporary.open("wb") as handle:
            pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(args.input_bundle)
    return {"status": "passed", "bundle": str(args.input_bundle),
            "artifact_ids": ids, "source_git_head": source_head}


def prepare_compact_bundle(args):
    started, cpu_started = time.perf_counter(), time.process_time()
    report = {"status": "running", "scope": "compact_real_input_preparation",
              "source_profile": str(args.source_profile),
              "input_bundle": str(args.input_bundle),
              "scientific_acceptance": False}

    def emit(stage, **values):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started,
                      cpu_seconds=time.process_time() - cpu_started, **values)
        _atomic_json(args.output, report)
        print(f"serialized_dual_prepare stage={stage} seconds={report['elapsed_seconds']:.1f}",
              flush=True)

    try:
        emit("validate_reference")
        result = _prepare_compact_bundle(args, emit)
        emit("completed", **result)
        return report
    except Exception as error:
        emit("exception", status="error", error_type=type(error).__name__, error=str(error))
        raise


def _device_tree(jax, tree, device):
    with jax.default_device(device), device_helper_cache(device):
        result = jax.device_put(tree, device)
        jax.block_until_ready(result)
        return result


class RecurrentWorker:
    """Private inherited pipes, no listening port and no per-update files."""

    def __init__(self, profile_path, *, cuda_device, memory_limit_bytes, echo_test=False,
                 admission_reports=None):
        if not cuda_device or "," in cuda_device:
            raise ValueError("worker needs exactly one physical CUDA device selector")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = cuda_device
        command = [
            sys.executable, str(Path(__file__).with_name("probe_shared_serialized_dual_worker.py")),
            "--source-profile", str(profile_path),
            "--memory-limit-bytes", str(memory_limit_bytes),
        ]
        if echo_test:
            command.append("--echo-test")
        for name, flag in (("daily", "daily"), ("T7", "seven"), ("T30", "thirty")):
            if name in (admission_reports or {}):
                command.extend((f"--{flag}-admission-report", str(admission_reports[name])))
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            env=environment, bufsize=0, cwd=Path(__file__).resolve().parents[2],
        )
        self.closed = False
        ready = self._receive()
        if ready.get("status") != "ready":
            self.close()
            raise RuntimeError(f"recurrent worker startup failed: {ready}")
        self.metadata = ready

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _receive(self):
        try:
            return pickle.load(self.process.stdout)
        except (EOFError, OSError) as error:
            raise RuntimeError(
                f"recurrent worker pipe closed (exit={self.process.poll()})"
            ) from error

    def evaluate(self, jax, parameters, cases):
        # Host transfer, serialization, worker GPU upload/execution and the
        # gradient response are all inside the measured complete update.
        began = time.perf_counter()
        message = {
            "kind": "evaluate", "parameters": jax.device_get(parameters),
            "cases": jax.device_get(cases),
            "parameter_version": getattr(self, "parameter_version", None),
        }
        host_seconds = time.perf_counter() - began
        write_started = time.perf_counter()
        pickle.dump(message, self.process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
        self.process.stdin.flush()
        write_seconds = time.perf_counter() - write_started
        receive_started = time.perf_counter()
        response = self._receive()
        if response.get("status") != "passed":
            raise RuntimeError(f"recurrent worker evaluation failed: {response}")
        if (message["parameter_version"] is not None
                and response.get("executed_parameter_version") != message["parameter_version"]):
            raise RuntimeError("recurrent worker executed parameter version differs")
        self.last_parameter_version = response.get("executed_parameter_version")
        self.last_evaluation_metrics = {
            "parent_device_get": host_seconds,
            "request_pickle_write": write_seconds,
            "response_wait_read_unpickle": time.perf_counter() - receive_started,
            "request_response_total": time.perf_counter() - began,
            **response.get("timing_seconds", {}),
        }
        self.last_device_memory = response.get("device_memory", {})
        if self.metadata.get("mode") != "echo_test" and set(response["result"]) != set(cases):
            raise ValueError("worker returned a different recurrent horizon set")
        return response["result"], response["recurrent_seconds"], response["worker_total_seconds"]

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            try:
                pickle.dump({"kind": "close"}, self.process.stdin)
                self.process.stdin.flush()
                self._receive()
            except (BrokenPipeError, RuntimeError):
                pass
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()
        self.process.wait(timeout=30)


class RemoteDualExecutor:
    """Overlap local daily AD with remote saved recurrent AD, then update once."""

    def __init__(self, *, jax, parameters, optimizer, daily_branch, worker,
                 combine, update, finite, local_recurrent_branch=None,
                 preserve_components=False, audit_parameter_versions=False):
        self.jax = jax
        self.parameters = parameters
        self.optimizer = optimizer
        self.daily_branch = daily_branch
        self.worker = worker
        self.combine = combine
        self.update = update
        self.finite = finite
        self.local_recurrent_branch = local_recurrent_branch
        self.preserve_components = preserve_components
        self.audit_parameter_versions = audit_parameter_versions
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="remote_recurrent")

    def close(self):
        self.pool.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def step(self, *, daily_payload, recurrent_payload, persist):
        began = time.perf_counter()
        parameter_version = int(np.asarray(self.optimizer.step))
        if persist and self.audit_parameter_versions:
            self.worker.parameter_version = parameter_version
        if set(recurrent_payload) != {7, 30}:
            raise ValueError("complete proposal requires exactly T7 and T30")
        remote_cases = ({30: recurrent_payload[30]} if self.local_recurrent_branch
                        else recurrent_payload)
        future = self.pool.submit(
            self.worker.evaluate, self.jax, self.parameters, remote_cases
        )
        try:
            daily = self.daily_branch(self.parameters, daily_payload)
            local = {}
            local_seconds = 0.0
            if self.local_recurrent_branch is not None:
                local, local_seconds = self.local_recurrent_branch(
                    self.parameters, {7: recurrent_payload[7]})
                if set(local) != {7}:
                    raise ValueError("local branch must return exactly T7")
        except BaseException:
            future.result()
            raise
        recurrent, recurrent_seconds, worker_total_seconds = future.result()
        if set(recurrent) != set(remote_cases):
            raise ValueError("remote branch returned wrong horizon set")
        recurrent = {**local, **recurrent}
        branch_seconds = time.perf_counter() - began
        hard = {"daily": daily["hard"],
                "T7": recurrent[7]["hard"], "T30": recurrent[30]["hard"]}
        if any(int(value) != 0 for item in hard.values() for value in item.values()):
            raise ValueError("remote dual branch returned nonzero physical counters")
        merge_started = time.perf_counter()
        seven = self.jax.device_put(recurrent[7]["gradient"])
        thirty = self.jax.device_put(recurrent[30]["gradient"])
        self.jax.block_until_ready((seven, thirty))
        gradient_upload_seconds = time.perf_counter() - merge_started
        combine_started = time.perf_counter()
        mixed = self.combine(daily["gradient"], seven, thirty)
        self.jax.block_until_ready(mixed)
        combine_seconds = time.perf_counter() - combine_started
        losses = {"daily": daily["loss"], "T7": recurrent[7]["loss"],
                  "T30": recurrent[30]["loss"]}
        norms = {"daily": daily["norm"], "T7": recurrent[7]["norm"],
                 "T30": recurrent[30]["norm"]}
        if (not all(np.isfinite(value) for value in losses.values())
                or not all(np.isfinite(value) and value >= 0 for value in norms.values())
                or not bool(np.asarray(self.finite(mixed)))):
            raise ValueError("remote dual component or mixed gradient is nonfinite")
        update_started = time.perf_counter()
        proposal, next_optimizer, gradient_norm, clip_scale = self.update(
            self.parameters, mixed, self.optimizer
        )
        self.jax.block_until_ready((proposal, next_optimizer, gradient_norm, clip_scale))
        optimizer_seconds = time.perf_counter() - update_started
        if (not bool(np.asarray(self.finite((proposal, next_optimizer))))
                or not np.isfinite(float(gradient_norm))
                or not 0.0 < float(clip_scale) <= 1.0
                or int(np.asarray(next_optimizer.step))
                != int(np.asarray(self.optimizer.step)) + 1):
            raise ValueError("remote dual AdamW proposal is invalid")
        merge_seconds = time.perf_counter() - merge_started
        if persist:
            self.parameters, self.optimizer = proposal, next_optimizer
        record = {
            "loss": {**losses, "joint": sum(losses.values())},
            "gradient_norms": norms,
            "optimizer_gradient_norm": float(gradient_norm),
            "clip_scale": float(clip_scale),
            "optimizer_step": int(np.asarray(next_optimizer.step)),
            "hard": hard, "all_hard_counts_zero": True,
            "timing_seconds": {
                "daily_microbatch": daily["seconds"],
                "parent_T7": local_seconds,
                "recurrent_worker_total": worker_total_seconds,
                "recurrent_gpu_branch": recurrent_seconds,
                "branches_elapsed_including_ipc": branch_seconds,
                "merge_checks_optimizer": merge_seconds,
                "returned_gradient_device_put_and_sync": gradient_upload_seconds,
                "component_merge_and_sync": combine_seconds,
                "adamw_and_sync": optimizer_seconds,
                "total": time.perf_counter() - began,
            },
        }
        record["worker_request_response_timings"] = getattr(self.worker, "last_evaluation_metrics", {})
        record["worker_device_memory"] = getattr(self.worker, "last_device_memory", {})
        if persist and self.audit_parameter_versions:
            record["branch_parameter_versions"] = {
                "daily": parameter_version, "T7": parameter_version,
                "T30": getattr(self.worker, "last_parameter_version", None)}
            if not self.local_recurrent_branch:
                record["branch_parameter_versions"]["T7"] = getattr(self.worker, "last_parameter_version", None)
            if any(value != parameter_version for value in record["branch_parameter_versions"].values()):
                raise ValueError("changing update component parameter version differs")
        record["component_seconds"] = {
            "daily": daily["seconds"],
            "T7": recurrent[7].get("seconds"),
            "T30": recurrent[30].get("seconds"),
        }
        total = record["timing_seconds"]["total"]
        parent_busy = daily["seconds"] + local_seconds
        record["card_seconds"] = {
            "card0_synchronized_branch": parent_busy,
            "card1_synchronized_branch": recurrent_seconds,
            "card0_branch_wait_or_host": max(0.0, branch_seconds-parent_busy),
            "card1_other_or_wait": max(0.0, total-recurrent_seconds),
            "two_cards_elapsed": 2.0*total,
            "scope": "blocked branch wall proxies, not GPU utilization; host/IPC/check time is not pure transfer",
        }
        values = {"gradient": mixed, "parameters": proposal, "optimizer": next_optimizer}
        if self.preserve_components:
            values["components"] = {"daily": daily, "T7": recurrent[7], "T30": recurrent[30]}
        return record, values


def compare_proposal(jax, dual_values, serial_values):
    comparisons = {
        name: compare_trees(jax, dual_values[name], serial_values[name], rtol=RTOL, atol=ATOL)
        for name in ("gradient", "parameters", "optimizer")
    }
    return comparisons


def measure_complete_updates(*, jax, serial_once, dual, daily,
                             secondary_cases, repeats, serial_gradient=None):
    """Time complete proposals at the same weights, with blocked interleaving."""
    if repeats < 3:
        raise ValueError("complete-update timing requires at least three repeats")
    serial_once()
    dual.step(daily_payload=daily, recurrent_payload=secondary_cases, persist=False)
    serial_rows, dual_rows = [], []
    serial_breakdown, dual_breakdown = [], []
    serial_result = dual_result = dual_record = None
    for index in range(repeats):
        order = ("serial", "dual") if index % 2 == 0 else ("dual", "serial")
        for mode in order:
            began = time.perf_counter()
            if mode == "serial":
                serial_result = serial_once()
                jax.block_until_ready(serial_result[:2])
                serial_rows.append(time.perf_counter() - began)
                serial_breakdown.append({
                    ("daily_microbatch" if key == "daily_32_microbatch_4" else key): value
                    for key, value in serial_result[2].get("timing_seconds", {}).items()
                })
            else:
                dual_record, dual_result = dual.step(
                    daily_payload=daily, recurrent_payload=secondary_cases,
                    persist=False,
                )
                dual_rows.append(time.perf_counter() - began)
                dual_breakdown.append(dict(dual_record.get("timing_seconds", {})))
    serial_median = float(np.median(serial_rows))
    dual_median = float(np.median(dual_rows))
    result = {
        "serial_seconds": serial_rows, "dual_seconds": dual_rows,
        "serial_median_seconds": serial_median, "dual_median_seconds": dual_median,
        "comparison": dual_cost_comparison(serial_median, dual_median),
        "timing_scope": "complete_gradient_parameter_and_gradient_ipc_physical_checks_adamw",
        "interleaved": True,
        "serial_timing_breakdown_seconds": serial_breakdown,
        "dual_timing_breakdown_seconds": dual_breakdown,
    }
    if serial_gradient is not None:
        serial_values = {
            "gradient": serial_gradient,
            "parameters": serial_result[0], "optimizer": serial_result[1],
        }
        result["full_tree_comparisons"] = compare_proposal(jax, dual_result, serial_values)
        result["joint_loss_close"] = bool(np.isclose(
            dual_record["loss"]["joint"], serial_result[2]["loss"]["joint"],
            rtol=RTOL, atol=ATOL,
        ))
        result["optimizer_scalars_close"] = all(
            np.isclose(dual_record[key], serial_result[2][key], rtol=RTOL, atol=ATOL)
            for key in ("optimizer_gradient_norm", "clip_scale")
        )
        result["status"] = (
            "passed" if result["joint_loss_close"] and result["optimizer_scalars_close"] and all(
                value["passed"] for value in result["full_tree_comparisons"].values()
            ) else "rejected"
        )
    return result


def _compact_learning_run(args, *, bundle, reference):
    """Feed the original diagnostic hooks from saved real inputs and update-0."""
    import jax
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
    from scripts.hpc import run_shared_learning as learning
    from scripts.hpc.shared_daily_microbatch import make_daily_accumulator

    device, = jax.local_devices()
    if device.platform != "gpu":
        raise ValueError("compact replay requires one visible GPU")
    require_distinct_batches(
        jax, bundle["artifact_ids"],
        bundle["payloads"][0]["daily"], bundle["payloads"][1]["daily"],
    )
    if any(set(payload["cases"]) != {7, 30} for payload in bundle["payloads"]):
        raise ValueError("compact bundle lacks a prepared T7 or T30 case")
    checkpoint = learning.load_learning_checkpoint(
        Path(bundle["initial_checkpoint"]), expected_method=reference["method"]
    )
    if checkpoint["completed_updates"] != 0:
        raise ValueError("compact replay requires the trusted fresh update-0 checkpoint")
    with jax.default_device(device), device_helper_cache(device):
        parameters, optimizer = jax.device_put(
            (checkpoint["parameters"], checkpoint["optimizer"]), device
        )
        operations = learning.make_tree_operations(jax, jnp, 32)
    if not bool(np.asarray(operations["finite"]((parameters, optimizer)))):
        raise ValueError("compact replay update-0 state is nonfinite")
    report = {
        "status": "running", "technical_updates": 2, "completed_updates": 0,
        "source_git_head": bundle["source_git_head"], "method": bundle["method"],
        "updates": [], "monitors": [], "checkpoints": [],
        "scientific_acceptance": False,
        "classification": "compact_saved_executable_two_update_diagnostic",
        "input_bundle": str(args.input_bundle),
    }
    _atomic_json(args.output, report)
    config = reference["method"]["configuration"]
    first = bundle["payloads"][0]
    evaluators, metadata = learning.compile_gradient_evaluators(
        jax=jax, parameters=parameters, daily_objective=None,
        daily_sample=jax.tree_util.tree_map(lambda value: value[:4], first["daily"]),
        recurrent_objective=None, seven_case=first["cases"][7],
        thirty_case=first["cases"][30],
        memory_limit_bytes=int(config["gradient_memory_limit_gib"]) * 2**30,
        emit=lambda *_a, **_kw: None, daily_microbatch_size=4,
    )
    report["compiled_gradient_graphs"] = metadata
    accumulator = make_daily_accumulator(
        jax=jax, jnp=jnp, microbatch_size=4,
        evaluator=evaluators["daily"], expected_count=32,
    )
    update = make_update(reference["method"]["daily_protocol"])
    for index, (identifier, payload) in enumerate(
        zip(bundle["artifact_ids"], bundle["payloads"], strict=True)
    ):
        with jax.default_device(device), device_helper_cache(device):
            daily, cases = jax.device_put((payload["daily"], payload["cases"]), device)
            began = time.perf_counter()
            parameters, optimizer, row = learning.execute_learning_update(
                jax=jax, parameters=parameters, optimizer=optimizer,
                daily_samples=daily, seven_case=cases[7], thirty_case=cases[30],
                evaluators=evaluators, operations=operations, update=update,
                daily_accumulator=accumulator,
            )
            jax.block_until_ready((parameters, optimizer))
        report["updates"].append({
            "update": index + 1, "artifact_id": identifier, **row,
            "wall_seconds": time.perf_counter() - began,
            "device_memory": device.memory_stats(),
        })
        report["completed_updates"] = index + 1
        _atomic_json(args.output, report)
    checkpoint_path = learning.checkpoint_path(args.output, 2)
    learning.save_learning_checkpoint(
        checkpoint_path, jax=jax, parameters=parameters, optimizer=optimizer,
        completed_updates=2, method=bundle["method"],
        updates=report["updates"], monitors=(),
    )
    report.update(
        status="passed", checkpoint=str(checkpoint_path),
        warm_second_update_wall_seconds=report["updates"][1]["wall_seconds"],
        device_peak_bytes_in_use=(device.memory_stats() or {}).get("peak_bytes_in_use"),
    )
    _atomic_json(args.output, report)
    return report


def inspect_inputs(args):
    """CPU-only admission/bundle check, without reading executable payloads."""
    profile, reference, digest = _reference_and_profile(args)
    if args.input_bundle is None:
        raise ValueError("--inspect-only requires the original --input-bundle")
    with args.input_bundle.open("rb") as handle:
        bundle = pickle.load(handle)
    ids = _validate_bundle(bundle, profile_sha256=digest, reference=reference,
                           source_head=_source_head())
    import jax
    require_distinct_batches(jax, ids, *(payload["daily"] for payload in bundle["payloads"]))
    for payload in bundle["payloads"]:
        if set(payload["cases"]) != {7, 30}:
            raise ValueError("bundle requires one T7 and one T30 case")
        for horizon, case in payload["cases"].items():
            if (len(case) != 5 or np.shape(case.days[0].continuous_state)[0] != horizon
                    or np.shape(case.targets["state"])[0] != horizon):
                raise ValueError("bundle recurrent case horizon differs")
    admissions = validate_admissions(profile, reference, admission_report_paths(args),
                                     profile_sha256=digest)
    result = {"status": "passed", "scope": "cpu_saved_combined_admission_inspection",
              "source_profile": str(args.source_profile), "source_profile_sha256": digest,
              "source_git_head": reference["source_git_head"],
              "input_bundle": str(args.input_bundle), "artifact_ids": ids,
              "admitted_overrides": admissions, "model_gradient_compilations": 0,
              "gpu_executables_loaded": False, "teacher_preparation": False,
              "daily_proposal_counts": getattr(args, "daily_counts", [32, 64]),
              "scientific_acceptance": False}
    _atomic_json(args.output, result)
    return result


def run(args):
    from scripts.hpc import run_shared_learning as learning
    from scripts.hpc.probe_shared_compile_cache import _require_supported_jax

    if args.technical_updates != 2 or args.prepare_only or args.resume_checkpoint:
        raise ValueError("serialized dual probe requires fresh --technical-updates 2")
    if args.repeats < 3:
        raise ValueError("at least three complete-update timing repeats are required")
    if not args.worker_cuda_device:
        raise ValueError("replay requires --worker-cuda-device")
    profile, reference, profile_sha256 = _reference_and_profile(args)
    admission_reports = admission_report_paths(args)
    admissions = validate_admissions(profile, reference, admission_reports,
                                     profile_sha256=profile_sha256)
    daily_counts = getattr(args, "daily_counts", [32, 64])
    bundle = None
    if getattr(args, "input_bundle", None) is not None:
        with args.input_bundle.open("rb") as handle:
            bundle = pickle.load(handle)
        _validate_bundle(bundle, profile_sha256=profile_sha256,
                         reference=reference, source_head=_source_head())
    output = args.output
    update_output = output.with_suffix(".updates.json")
    if update_output.resolve() == Path(profile["technical_update_report"]).resolve():
        raise ValueError("probe output must preserve the reference report/checkpoints")
    report = {
        "status": "running", "scope": "saved_executable_dual_complete_update_probe",
        "source_profile": str(args.source_profile),
        "scientific_acceptance": False, "retained_formal_training_updates": 0,
        "comparison_rtol": RTOL, "comparison_atol": ATOL,
        "comparison_tolerance_source": "shared_execution_benchmark_v1.json",
        "timings": {}, "changing_data": [],
        "source_profile_sha256": profile_sha256, "admitted_overrides": admissions,
        "model_gradient_compilations": 0,
        "candidate_vs_original_gradient_evidence": (
            "three_admitted_full_component_gradient_gates_at_fresh_update0_artifact742; "
            "two_changing_updates_full_parameters_and_adamw_vs_original_checkpoint"
            if admissions else "original_saved_executables"),
    }
    started = time.perf_counter()
    original_compile = learning.compile_gradient_evaluators
    original_execute = learning.execute_learning_update
    original_run = learning.run
    context = {"batches": [], "cases": []}

    def emit(stage, **values):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **values)
        _atomic_json(output, report)
        print(f"serialized_dual stage={stage} seconds={report['elapsed_seconds']:.1f}", flush=True)

    def load_instead_of_compile(**kwargs):
        jax = kwargs["jax"]
        _require_supported_jax(jax)
        devices = tuple(jax.local_devices())
        if len(devices) != 1 or devices[0].platform != "gpu":
            raise ValueError("parent requires one visible GPU; worker owns the other")
        parent_selector = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not parent_selector or "," in parent_selector or parent_selector == args.worker_cuda_device:
            raise ValueError("parent and worker must select different single physical GPUs")
        if kwargs["daily_microbatch_size"] != 4:
            raise ValueError("saved daily executable requires microbatch4")
        current = json.loads(update_output.read_text(encoding="utf-8"))
        require_same_method(current, reference)
        from jax.experimental.serialize_executable import deserialize_and_load

        primary = devices[0]
        context.update(jax=jax, device=primary, memory_limit=kwargs["memory_limit_bytes"])
        with jax.default_device(primary), device_helper_cache(primary):
            primary_evaluators, primary_metadata = load_evaluators(
                profile, deserialize=deserialize_and_load,
                memory_sizes=learning.memory_sizes,
                memory_limit_bytes=kwargs["memory_limit_bytes"], emit=emit,
                overrides=admissions,
            )
        emit("start_recurrent_worker", worker_cuda_device=args.worker_cuda_device)
        context["worker"] = RecurrentWorker(
            args.source_profile, cuda_device=args.worker_cuda_device,
            memory_limit_bytes=kwargs["memory_limit_bytes"],
            admission_reports=admission_reports,
        )
        if context["worker"].metadata.get("admitted_overrides", {}) != admissions:
            raise ValueError("worker admission provenance differs from parent")
        for name in ("T7", "T30"):
            expected = Path(admissions.get(name, profile["executables"][name])["executable"]).resolve()
            actual = Path(context["worker"].metadata["executables"][name]["source_executable"]).resolve()
            if actual != expected:
                raise ValueError(f"worker {name} binary differs from parent selection")
        report["worker"] = context["worker"].metadata
        report["worker"]["physical_cuda_selector"] = args.worker_cuda_device
        report["worker"]["transport"] = "private_persistent_pickle_pipes_no_disk_per_update"
        report["worker"]["parameter_policy"] = "full_parameter_host_transfer_each_update"
        report["native_retarget_supported"] = False
        emit("gradients_ready_no_model_compile")
        return primary_evaluators, primary_metadata

    def execute_compared(**kwargs):
        jax = kwargs["jax"]
        import jax.numpy as jnp

        primary = context["device"]
        if context.get("dual") is None:
            context["initial_parameters"] = kwargs["parameters"]
            context["initial_optimizer"] = kwargs["optimizer"]
            context["update"] = kwargs["update"]
            context["evaluators"] = kwargs["evaluators"]
            operations = kwargs["operations"]
            with jax.default_device(primary), device_helper_cache(primary):
                combine = make_component_combiner(jax)
            context["combine"] = combine
            context["primary_operations32"] = operations
            context["dual"] = RemoteDualExecutor(
                jax=jax, parameters=kwargs["parameters"], optimizer=kwargs["optimizer"],
                daily_branch=make_microbatch_daily_branch(
                    jax=jax, jnp=jnp, device=primary,
                    evaluator=kwargs["evaluators"]["daily"], operations=operations,
                    microbatch_size=4, expected_count=32,
                ),
                worker=context["worker"], combine=combine,
                update=kwargs["update"], finite=operations["finite"],
            )
        dual = context["dual"]
        if len(context["batches"]) >= 2:
            raise ValueError("serialized dual probe accepts exactly two changing artifacts")
        batch = kwargs["daily_samples"]
        primary_cases = {7: kwargs["seven_case"], 30: kwargs["thirty_case"]}
        remote_cases = primary_cases
        if not context["batches"]:
            context["serial_parameters"] = kwargs["parameters"]
            context["serial_optimizer"] = kwargs["optimizer"]
        serial_kwargs = {
            **kwargs, "parameters": context["serial_parameters"],
            "optimizer": context["serial_optimizer"],
        }
        emit("serial_changing_update", update=len(context["batches"]) + 1)
        serial_began = time.perf_counter()
        serial_parameters, serial_optimizer, serial_row = original_execute(**serial_kwargs)
        jax.block_until_ready((serial_parameters, serial_optimizer))
        serial_seconds = time.perf_counter() - serial_began
        # The runner does not expose its mixed gradient. Reuse identical saved
        # evaluators at the same serial start weights for a full-tree comparison.
        primary_daily = make_microbatch_daily_branch(
            jax=jax, jnp=jnp, device=primary, evaluator=kwargs["evaluators"]["daily"],
            operations=kwargs["operations"], microbatch_size=4, expected_count=32,
        )
        primary_recurrent = make_recurrent_branch(
            jax=jax, device=primary,
            evaluators={7: kwargs["evaluators"]["T7"],
                        30: kwargs["evaluators"]["T30"]},
            operations=kwargs["operations"],
        )
        serial_gradient, _ = _serial_mixed(
            parameters=context["serial_parameters"], daily_payload=batch,
            recurrent_payload=primary_cases, daily_branch=primary_daily,
            recurrent_branch=primary_recurrent, combine=context["combine"],
        )
        emit("dual_changing_update", update=len(context["batches"]) + 1)
        dual_began = time.perf_counter()
        dual_record, dual_values = dual.step(
            daily_payload=batch, recurrent_payload=remote_cases, persist=True,
        )
        dual_seconds = time.perf_counter() - dual_began
        comparisons = compare_proposal(jax, dual_values, {
            "gradient": serial_gradient, "parameters": serial_parameters,
            "optimizer": serial_optimizer,
        })
        scalar_close = bool(np.isclose(
            dual_record["loss"]["joint"], serial_row["loss"]["joint"],
            rtol=RTOL, atol=ATOL,
        ))
        optimizer_scalars_close = all(
            np.isclose(dual_record[key], serial_row[key], rtol=RTOL, atol=ATOL)
            for key in ("optimizer_gradient_norm", "clip_scale")
        )
        passed = scalar_close and optimizer_scalars_close and all(
            row["passed"] for row in comparisons.values()
        )
        index = len(context["batches"])
        context["batches"].append(batch)
        context["cases"].append(primary_cases)
        context["serial_parameters"] = serial_parameters
        context["serial_optimizer"] = serial_optimizer
        report["changing_data"].append({
            "update": index + 1, "status": "passed" if passed else "rejected",
            "serial_wall_seconds": serial_seconds, "dual_wall_seconds": dual_seconds,
            "comparisons": comparisons, "joint_loss_close": scalar_close,
            "optimizer_scalars_close": optimizer_scalars_close,
            "serial_optimizer_step": int(np.asarray(serial_optimizer.step)),
            "dual_optimizer_step": dual_record["optimizer_step"],
        })
        emit("changing_update_compared", update=index + 1)
        if not passed:
            raise ValueError("dual changing-data full update differs from serial")
        # The unchanged runner owns checkpoint persistence. Its report's loss
        # and science terms are supplied by the checked serial computation.
        return dual_values["parameters"], dual_values["optimizer"], {
            **serial_row, "timing_seconds": dual_record["timing_seconds"],
            "serialized_dual_wall_seconds": dual_seconds,
        }

    learning.compile_gradient_evaluators = load_instead_of_compile
    learning.execute_learning_update = execute_compared
    if bundle is not None:
        learning.run = lambda compact_args: _compact_learning_run(
            compact_args, bundle=bundle, reference=reference
        )
    args.output = update_output
    try:
        emit("run_two_real_artifacts")
        outcome = learning.run(args)
        report["technical_update_report"] = str(update_output)
        if outcome["status"] != "passed":
            emit("rejected", status="rejected", reason="technical_updates_failed")
            return report
        jax = context["jax"]
        import jax.numpy as jnp

        primary = context["device"]
        identifiers = [row["artifact_id"] for row in outcome["updates"]]
        require_distinct_batches(jax, identifiers, *context["batches"])
        report["artifact_ids"] = identifiers
        report["technical_update_comparison"] = compare_update_reports(
            jax, outcome, reference, rtol=RTOL, atol=ATOL,
        )
        saved = learning.load_learning_checkpoint(
            Path(reference["checkpoint"]), expected_method=reference["method"]
        )
        replay = learning.load_learning_checkpoint(
            Path(outcome["checkpoint"]), expected_method=reference["method"]
        )
        report["reference_checkpoint_comparisons"] = {
            name: compare_trees(jax, replay[key], saved[key], rtol=RTOL, atol=ATOL)
            for name, key in (("parameters", "parameters"), ("full_adamw_state", "optimizer"))
        }
        emit("changing_updates_and_reference_compared")
        if not report["technical_update_comparison"]["passed"] or not all(
            row["passed"] for row in report["reference_checkpoint_comparisons"].values()
        ):
            raise ValueError("serialized dual technical replay differs from trusted reference")
        context["dual"].close()
        context["dual"] = None

        # Fixed-weight proposals use the same fresh initial parameters and
        # optimizer for both modes. No proposal is retained as formal training.
        parameters = context["initial_parameters"]
        optimizer = context["initial_optimizer"]
        evaluators = context["evaluators"]
        cases0 = context["cases"][0]
        batches = {32: context["batches"][0]}
        if 64 in daily_counts:
            batches[64] = _device_tree(
                jax, concatenate_daily(jax, *context["batches"]), primary
            )
        batches = {count: batches[count] for count in daily_counts}
        for count, batch in batches.items():
            emit("measure_complete_update", daily_rows=count)
            operations = learning.make_tree_operations(jax, jnp, count)
            with jax.default_device(primary), device_helper_cache(primary):
                combine = make_component_combiner(jax)
            daily_branch = make_microbatch_daily_branch(
                jax=jax, jnp=jnp, device=primary, evaluator=evaluators["daily"],
                operations=operations, microbatch_size=4, expected_count=count,
            )
            recurrent_primary = make_recurrent_branch(
                jax=jax, device=primary,
                evaluators={7: evaluators["T7"], 30: evaluators["T30"]},
                operations=operations,
            )
            from scripts.hpc.shared_daily_microbatch import make_daily_accumulator

            accumulate = make_daily_accumulator(
                jax=jax, jnp=jnp, microbatch_size=4,
                evaluator=evaluators["daily"], expected_count=count,
            )

            def serial_once():
                with jax.default_device(primary), device_helper_cache(primary):
                    return original_execute(
                        jax=jax, parameters=parameters, optimizer=optimizer,
                        daily_samples=batch, seven_case=cases0[7],
                        thirty_case=cases0[30], evaluators=evaluators,
                        operations=operations, update=context["update"],
                        daily_accumulator=accumulate,
                    )

            with RemoteDualExecutor(
                jax=jax, parameters=parameters, optimizer=optimizer,
                daily_branch=daily_branch, worker=context["worker"],
                combine=combine, update=context["update"], finite=operations["finite"],
            ) as dual:
                serial_gradient, _ = _serial_mixed(
                    parameters=parameters, daily_payload=batch,
                    recurrent_payload=cases0, daily_branch=daily_branch,
                    recurrent_branch=recurrent_primary, combine=combine,
                )
                result = measure_complete_updates(
                    jax=jax, serial_once=serial_once, dual=dual, daily=batch,
                    secondary_cases=cases0,
                    repeats=args.repeats, serial_gradient=serial_gradient,
                )
            result["daily_rows_per_update"] = count
            result["sampling_scope"] = (
                "unchanged_daily32_plus_one_T7_one_T30" if count == 32 else
                "daily64_mean_plus_one_T7_one_T30_fixed_weight_proposal_not_learning_acceptance"
            )
            result["distinct_daily_artifact_ids"] = identifiers[:count // 32]
            result["recurrent_case_artifact_id"] = identifiers[0]
            result["unique_daily_row_coverage_claimed"] = False
            result["weight_policy"] = "same_fresh_initial_weights_discard_each_proposal"
            result["serial_daily_rows_per_second"] = count / result["serial_median_seconds"]
            result["dual_daily_rows_per_second"] = count / result["dual_median_seconds"]
            report["timings"][str(count)] = result
            emit("complete_update_measured", daily_rows=count, timing_status=result["status"])
            if result["status"] != "passed":
                raise ValueError(f"serialized dual {count}-row proposal differs from serial")
        report["parent_device_memory"] = primary.memory_stats()
        emit("completed", status="passed", retained_formal_training_updates=0,
             final_batch_recipe_unchanged=True)
        return report
    except Exception as error:
        emit("exception", status="error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if context.get("dual") is not None:
            context["dual"].close()
        if context.get("worker") is not None:
            context["worker"].close()
        learning.compile_gradient_evaluators = original_compile
        learning.execute_learning_update = original_execute
        learning.run = original_run
        args.output = output


def compare_schedule_pair(jax, old_record, old_values, new_record, new_values):
    """Compare full proposals and every retained component, outside timing."""
    checks = compare_proposal(jax, new_values, old_values)
    components = {}
    for name in ("daily", "T7", "T30"):
        old = old_values["components"][name]
        new = new_values["components"][name]
        components[name] = {
            "gradient": compare_trees(jax, new["gradient"], old["gradient"], rtol=RTOL, atol=ATOL),
            "auxiliary": compare_trees(jax, new["auxiliary"], old["auxiliary"], rtol=RTOL, atol=ATOL),
            "loss_close": bool(np.isclose(new["loss"], old["loss"], rtol=RTOL, atol=ATOL)),
            "norm_close": bool(np.isclose(new["norm"], old["norm"], rtol=RTOL, atol=ATOL)),
            "physical_counters_equal": new["hard"] == old["hard"],
        }
    scalar_checks = {
        key: bool(np.isclose(new_record[key], old_record[key], rtol=RTOL, atol=ATOL))
        for key in ("optimizer_gradient_norm", "clip_scale", "optimizer_step")
    }
    joint_close = bool(np.isclose(new_record["loss"]["joint"], old_record["loss"]["joint"], rtol=RTOL, atol=ATOL))
    passed = all(item["passed"] for item in checks.values()) and all(scalar_checks.values()) and joint_close
    passed = passed and all(
        item["gradient"]["passed"] and item["auxiliary"]["passed"] and item["loss_close"]
        and item["norm_close"] and item["physical_counters_equal"]
        for item in components.values()
    )
    return {"status": "passed" if passed else "rejected", "full_tree_comparisons": checks,
            "components": components, "optimizer_scalars_close": scalar_checks,
            "joint_loss_close": joint_close}


def measure_schedule_pairs(*, jax, old, new, daily, cases, repeats, emit, output=None):
    """Warm both schedules, then discard every alternating fixed-state proposal."""
    if repeats < 3:
        raise ValueError("schedule comparison requires at least three paired repeats")
    initial = [(executor.parameters, executor.optimizer) for executor in (old, new)]
    rows = []
    for name, executor in (("old", old), ("new", new)):
        emit("schedule_warmup", schedule=name)
        executor.step(daily_payload=daily, recurrent_payload=cases, persist=False)
    for index in range(repeats):
        order = ("old", "new") if index % 2 == 0 else ("new", "old")
        actual = {}
        for name in order:
            emit("schedule_proposal", pair=index+1, schedule=name)
            executor = old if name == "old" else new
            record, values = executor.step(daily_payload=daily, recurrent_payload=cases, persist=False)
            jax.block_until_ready((values["gradient"], values["parameters"], values["optimizer"]))
            actual[name] = (record, values)
        comparison = compare_schedule_pair(jax, *actual["old"], *actual["new"])
        row = {"pair": index+1, "order": list(order), "old": actual["old"][0],
               "new": actual["new"][0], "comparison": comparison}
        row["old_over_new_wall_ratio"] = row["old"]["timing_seconds"]["total"] / row["new"]["timing_seconds"]["total"]
        if index == 0 and output is not None:
            path = output.with_suffix(".auxiliary.pkl")
            aux = {mode: {name: component["auxiliary"] for name, component in values["components"].items()}
                   for mode, (_, values) in actual.items()}
            with path.open("wb") as handle:
                pickle.dump(jax.device_get(aux), handle, protocol=pickle.HIGHEST_PROTOCOL)
            row["retained_actual_auxiliary"] = str(path)
            row["auxiliary_leaf_shapes"] = {
                name: [{"shape": list(np.shape(value)), "dtype": str(np.asarray(value).dtype)}
                       for value in jax.tree_util.tree_leaves(aux["old"][name])]
                for name in ("daily", "T7", "T30")
            }
            row["auxiliary_top_level_keys"] = {name: list(aux["old"][name]) for name in ("daily", "T7", "T30")}
        rows.append(row)
        emit("schedule_pair_compared", paired_rows=rows)
        for executor, (parameters, optimizer) in zip((old, new), initial, strict=True):
            if executor.parameters is not parameters or executor.optimizer is not optimizer:
                raise ValueError("fixed proposal advanced parameter/AdamW state")
        if comparison["status"] != "passed":
            raise ValueError("schedule full proposal or retained component differs")
        del actual
    medians = {name: float(np.median([row[name]["timing_seconds"]["total"] for row in rows]))
               for name in ("old", "new")}
    gain = medians["old"] / medians["new"]
    return {"paired_rows": rows, "median_seconds": medians,
            "old_over_new_throughput_ratio": gain, "throughput_gain_fraction": gain-1.0,
            "wall_reduction_fraction": 1.0-1.0/gain,
            "median_card_seconds_per_proposal": {name: 2*value for name, value in medians.items()},
            "weight_policy": "identical_update0_parameter_and_AdamW_objects_all_proposals_discarded",
            "proposals_per_schedule_including_warmup": repeats+1,
            "completed_changing_updates": 0}


def measure_changing_schedule_pairs(*, jax, old, new, payloads, updates, emit, output):
    """Advance two independent diagnostic chains through the public AdamW."""
    if not 1 <= updates <= 16 or not payloads:
        raise ValueError("changing schedule chains require 1..16 actual updates")
    initial_step = int(np.asarray(old.optimizer.step))
    if initial_step != 0 or int(np.asarray(new.optimizer.step)) != 0:
        raise ValueError("changing schedules require the same original update0")
    rows = []
    for index in range(updates):
        payload = payloads[index % len(payloads)]
        order = ("old", "new") if index % 2 == 0 else ("new", "old")
        actual = {}
        previous = {name: executor.parameters for name, executor in (("old", old), ("new", new))}
        for name in order:
            executor = old if name == "old" else new
            emit("changing_schedule_update", update=index+1, schedule=name)
            actual[name] = executor.step(daily_payload=payload["daily"], recurrent_payload=payload["cases"], persist=True)
            jax.block_until_ready((executor.parameters, executor.optimizer))
        comparison = compare_schedule_pair(jax, *actual["old"], *actual["new"])
        advanced = {}
        for name, executor in (("old", old), ("new", new)):
            changed = any(not np.array_equal(np.asarray(a), np.asarray(b))
                for a, b in zip(jax.tree_util.tree_leaves(previous[name]),
                    jax.tree_util.tree_leaves(executor.parameters), strict=True))
            step = int(np.asarray(executor.optimizer.step))
            if not changed or step != index+1:
                raise ValueError("changing schedule parameter/AdamW state did not advance")
            advanced[name] = {"parameters_changed": changed, "optimizer_step": step,
                "all_component_parameter_versions": actual[name][0]["branch_parameter_versions"],
                "version_evidence": "worker echoes executed request version; local branches receive current executor tree directly"}
        rows.append({"update": index+1, "payload_index": index % len(payloads), "order": list(order),
            "old": actual["old"][0], "new": actual["new"][0], "comparison": comparison,
            "state_progression": advanced})
        emit("changing_schedule_pair_compared", changing_rows=rows)
        if comparison["status"] != "passed":
            raise ValueError("changing schedule full proposal/components differ")
        del actual, previous
    retained = {}
    for name, executor in (("old", old), ("new", new)):
        path = output.with_suffix(f".{name}.diagnostic-chain.pkl")
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(jax.device_get({"parameters": executor.parameters, "optimizer": executor.optimizer,
                "completed_updates": updates, "purpose": "changing_schedule_diagnostic_only"}), handle,
                protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
        retained[name] = str(path)
    return {"completed_changing_updates": updates, "changing_rows": rows,
        "retained_diagnostic_chains": retained, "final_optimizer_steps": {"old": updates, "new": updates},
        "weight_policy": "two independent chains from same original update0; each public AdamW result feeds the next update",
        "actual_updates_all_chains": 2*updates}


def run_schedule_comparison(args):
    """Load update-0/binaries directly; never enter a learning/bootstrap replay."""
    if args.input_bundle is None or args.daily_counts != [32] or args.repeats < 3:
        raise ValueError("comparison-only requires compact bundle, daily-counts 32 and >=3 pairs")
    if args.prepare_only or args.resume_checkpoint or args.prepare_input_bundle:
        raise ValueError("comparison-only cannot prepare or resume training")
    started = time.perf_counter()
    report = {"status": "running", "scope": "fixed_parameter_old_vs_daily_T7_T30_dual_schedule",
              "scientific_acceptance": False, "completed_changing_updates": 0,
              "retained_formal_training_updates": 0, "model_gradient_compilations": 0,
              "comparison_rtol": RTOL, "comparison_atol": ATOL,
              "state_AD_scope": "existing saved parameter-gradient binaries and retained auxiliary outputs; no new independent initial-state AD certification",
              "old_schedule": "parent GPU0 daily32; worker GPU1 T7+T30",
              "new_schedule": "parent GPU0 daily32 then T7; worker GPU1 T30",
              "workload": "daily32/eight distinct micro4 + full one T7 + full one T30",
              "loss_weight_policy": "existing already-weighted saved binaries; original daily/T7/T30 combiner order",
              "native_retarget_supported": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    worker = None
    executors = []

    def emit(stage, **values):
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-started, **values)
        _atomic_json(args.output, report)
        print(f"schedule_comparison stage={stage} seconds={report['elapsed_seconds']:.1f}", flush=True)

    try:
        emit("validate_reference_and_bundle")
        profile, reference, digest = _reference_and_profile(args)
        paths = admission_report_paths(args)
        admissions = validate_admissions(profile, reference, paths, profile_sha256=digest)
        with args.input_bundle.open("rb") as handle:
            bundle = pickle.load(handle)
        ids = _validate_bundle(bundle, profile_sha256=digest, reference=reference, source_head=_source_head())
        from scripts.hpc import run_shared_learning as learning
        from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
        from scripts.hpc.probe_shared_compile_cache import _require_supported_jax
        import jax
        import jax.numpy as jnp
        from jax.experimental.serialize_executable import deserialize_and_load
        _require_supported_jax(jax)
        device, = jax.local_devices()
        selector = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if device.platform != "gpu" or not selector or "," in selector or not args.worker_cuda_device or "," in args.worker_cuda_device or selector == args.worker_cuda_device:
            raise ValueError("comparison requires distinct single physical GPU parent/worker selectors")
        require_distinct_batches(jax, ids, *(payload["daily"] for payload in bundle["payloads"]))
        checkpoint = learning.load_learning_checkpoint(Path(bundle["initial_checkpoint"]), expected_method=reference["method"])
        if checkpoint["completed_updates"] != 0:
            raise ValueError("comparison requires original update-0")
        limit = int(reference["method"]["configuration"]["gradient_memory_limit_gib"])*2**30
        report.update(source_git_head=bundle["source_git_head"], source_profile=str(args.source_profile),
                      source_profile_sha256=digest, input_bundle=str(args.input_bundle), artifact_ids=ids,
                      tested_artifact_id=ids[0], initial_checkpoint=bundle["initial_checkpoint"],
                      admitted_overrides=admissions, parent_device=str(device), parent_cuda_selector=selector,
                      worker_cuda_selector=args.worker_cuda_device, memory_limit_bytes=limit)
        with jax.default_device(device), device_helper_cache(device):
            parameters, optimizer = _device_tree(jax, (checkpoint["parameters"], checkpoint["optimizer"]), device)
            daily, cases = _device_tree(jax, (bundle["payloads"][0]["daily"], bundle["payloads"][0]["cases"]), device)
            operations = learning.make_tree_operations(jax, jnp, 32)
            update = make_update(reference["method"]["daily_protocol"])
            combine = make_component_combiner(jax)
            evaluators, metadata = load_evaluators(profile, deserialize=deserialize_and_load,
                memory_sizes=learning.memory_sizes, memory_limit_bytes=limit, emit=emit, overrides=admissions)
        changing_payloads = bundle["payloads"]
        del checkpoint, bundle
        report["parent_executables"] = metadata
        emit("start_worker")
        worker = RecurrentWorker(args.source_profile, cuda_device=args.worker_cuda_device,
                                 memory_limit_bytes=limit, admission_reports=paths)
        if worker.metadata.get("admitted_overrides", {}) != admissions:
            raise ValueError("worker admission provenance differs")
        for name in ("T7", "T30"):
            if Path(worker.metadata["executables"][name]["source_executable"]).resolve() != Path(metadata[name]["source_executable"]).resolve():
                raise ValueError("worker/parent executable selection differs")
        report["worker"] = worker.metadata
        emit("graphs_ready_no_model_compile", graphs_ready_seconds=time.perf_counter()-started)
        daily_branch = make_microbatch_daily_branch(jax=jax, jnp=jnp, device=device,
            evaluator=evaluators["daily"], operations=operations, microbatch_size=4,
            expected_count=32, preserve_auxiliary=True)
        seven_branch = make_recurrent_branch(jax=jax, device=device, evaluators={7: evaluators["T7"]},
            operations=operations, horizons=(7,), preserve_auxiliary=True)
        for local in (None, seven_branch):
            executors.append(RemoteDualExecutor(jax=jax, parameters=parameters, optimizer=optimizer,
                daily_branch=daily_branch, worker=worker, combine=combine, update=update,
                finite=operations["finite"], local_recurrent_branch=local, preserve_components=True))
        if getattr(args, "schedule_changing_updates", 0):
            for executor in executors:
                executor.audit_parameter_versions = True
        changing_updates = getattr(args, "schedule_changing_updates", 0)
        if changing_updates:
            report["scope"] = "changing_parameter_old_vs_daily_T7_T30_dual_schedule"
            report.update(measure_changing_schedule_pairs(jax=jax, old=executors[0], new=executors[1],
                payloads=changing_payloads, updates=changing_updates, emit=emit, output=args.output))
        else:
            report.update(measure_schedule_pairs(jax=jax, old=executors[0], new=executors[1],
                daily=daily, cases=cases, repeats=args.repeats, emit=emit, output=args.output))
        report["parent_device_memory"] = device.memory_stats()
        report["worker_device_memory"] = getattr(worker, "last_device_memory", {})
        for name in ("parent_device_memory", "worker_device_memory"):
            if (report[name] or {}).get("peak_bytes_in_use", 0) > limit:
                raise ValueError(f"{name} exceeds unchanged memory limit")
        emit("completed", status="passed", initial_optimizer_step=int(np.asarray(optimizer.step)),
             final_optimizer_step=int(np.asarray(executors[0].optimizer.step)))
        return report
    except Exception as error:
        emit("exception", status="error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for executor in executors:
            executor.close()
        if worker is not None:
            worker.close()


def parser():
    from scripts.hpc.run_shared_learning import parser as base_parser

    result = base_parser()
    result.description = __doc__
    result.add_argument("--source-profile", type=Path, required=True)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--worker-cuda-device",
                        help="physical CUDA selector hidden from the parent process")
    result.add_argument("--input-bundle", type=Path,
                        help="CPU-prepared real input bundle for fast diagnostic replay")
    result.add_argument("--prepare-input-bundle", action="store_true",
                        help="prepare --input-bundle once on free CPU; do not run updates")
    _serialized_module.add_admission_arguments(result)
    result.add_argument("--daily-counts", type=int, nargs="+", choices=(32, 64), default=[32, 64])
    result.add_argument("--inspect-only", action="store_true")
    result.add_argument("--comparison-only", action="store_true",
                        help="compare fixed old/new two-GPU schedules; skip changing updates and checkpoint persistence")
    result.add_argument("--schedule-changing-updates", type=int, choices=range(1, 17), default=0,
                        help="with comparison-only, advance two original diagnostic chains 1..16 updates instead of fixed proposals")
    return result


def run_worker_smoke(output, worker_cuda_device):
    """Tiny same-binary, two-process recurrent proof without Teacher inputs."""
    import jax
    import jax.numpy as jnp
    from jax.experimental.serialize_executable import serialize

    from scripts.hpc.probe_shared_compile_cache import _require_supported_jax

    _require_supported_jax(jax)
    devices = tuple(jax.local_devices())
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("smoke parent requires exactly one visible GPU")

    def objective(parameters, sample):
        residual = parameters["weight"] * sample - parameters["target"]
        loss = jnp.sum(residual * residual)
        return loss, {"hard": {"synthetic_failure": jnp.asarray(0, dtype=jnp.int32)}}

    report = {"status": "running", "scope": "tiny_saved_recurrent_worker_same_binary",
              "parent_device": str(devices[0]), "worker_cuda_selector": worker_cuda_device,
              "full_model_loaded": False, "teacher_data_loaded": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, report)
    with tempfile.TemporaryDirectory(prefix="serialized-dual-smoke-", dir=output.parent) as temp:
        temporary = Path(temp)
        parameters = {"weight": jnp.asarray([0.3, -0.7], dtype=jnp.float32),
                      "target": jnp.asarray([0.2, 0.4], dtype=jnp.float32)}
        samples = {
            7: jnp.asarray([0.8, -0.1], dtype=jnp.float32),
            30: jnp.asarray([-0.2, 0.9], dtype=jnp.float32),
        }
        profile = {"executables": {}}
        compiled = {}
        for horizon in (7, 30):
            executable = jax.jit(jax.value_and_grad(objective, has_aux=True)).lower(
                parameters, samples[horizon]
            ).compile()
            path = temporary / f"T{horizon}.executable.pkl"
            with path.open("wb") as handle:
                pickle.dump(serialize(executable), handle, protocol=pickle.HIGHEST_PROTOCOL)
            profile["executables"][f"T{horizon}"] = {"executable": str(path)}
            compiled[horizon] = executable
        profile_path = temporary / "profile.json"
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
        report["stage"] = "tiny_binaries_saved"
        _atomic_json(output, report)
        with RecurrentWorker(
            profile_path, cuda_device=worker_cuda_device, memory_limit_bytes=2**30
        ) as worker:
            report["worker"] = worker.metadata
            report["stage"] = "tiny_worker_loaded"
            _atomic_json(output, report)
            comparisons = []
            for index in range(2):
                current = jax.tree_util.tree_map(
                    lambda value: value + index * 0.05, parameters
                )
                cases = {horizon: (samples[horizon] + index * 0.1,)
                         for horizon in (7, 30)}
                actual, recurrent_seconds, worker_total = worker.evaluate(jax, current, cases)
                for horizon in (7, 30):
                    expected = compiled[horizon](current, *cases[horizon])
                    expected_gradient = jax.device_get(expected[1])
                    np.testing.assert_allclose(
                        actual[horizon]["gradient"]["weight"],
                        expected_gradient["weight"], rtol=RTOL, atol=ATOL,
                    )
                    np.testing.assert_allclose(
                        actual[horizon]["gradient"]["target"],
                        expected_gradient["target"], rtol=RTOL, atol=ATOL,
                    )
                    np.testing.assert_allclose(
                        actual[horizon]["loss"], float(expected[0][0]),
                        rtol=RTOL, atol=ATOL,
                    )
                    if any(actual[horizon]["hard"].values()):
                        raise ValueError("tiny worker returned nonzero physical counters")
                comparisons.append({"round": index + 1, "status": "passed",
                                    "worker_recurrent_seconds": recurrent_seconds,
                                    "worker_total_seconds": worker_total})
                report["comparisons"] = comparisons
                report["stage"] = "tiny_round_compared"
                _atomic_json(output, report)
    report.update(stage="completed", status="passed")
    _atomic_json(output, report)
    return report


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--smoke-worker":
        if len(sys.argv) != 5 or sys.argv[2] != "--output":
            raise SystemExit(
                "usage: probe_shared_serialized_dual.py --smoke-worker "
                "--output PATH WORKER_CUDA_DEVICE"
            )
        result = run_worker_smoke(Path(sys.argv[3]), sys.argv[4])
    else:
        args = parser().parse_args()
        if args.schedule_changing_updates and not args.comparison_only:
            raise SystemExit("--schedule-changing-updates requires --comparison-only")
        if args.inspect_only:
            result = inspect_inputs(args)
        elif args.comparison_only:
            result = run_schedule_comparison(args)
        elif args.prepare_input_bundle:
            if args.input_bundle is None:
                raise SystemExit("--prepare-input-bundle requires --input-bundle")
            result = prepare_compact_bundle(args)
        else:
            result = run(args)
    raise SystemExit(0 if result["status"] == "passed" else 1)
