"""Original-only saved full-state AD, old versus new two-card changing updates.

Old: parent daily32, worker T7+T30. New: parent daily32+T7, worker T30.
Each chain starts at the admitted original public AdamW update0. Strict replay
failures remain failures while finite, physically valid bounded timing proceeds.
No model gradient, loss, cotangent, or physical process is reconstructed here.
"""
from __future__ import annotations

import time
PROCESS_STARTED = time.monotonic()

import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading
from contextlib import contextmanager

import numpy as np

SCHEMA = "shared_original_full_state_schedule_manifest_v1"
RTOL, ATOL = 1e-5, 1e-6
MEMORY_LIMIT = 24 * 2**30


def load_adjacent(name):
    root = os.environ.get("ORCHIDEE_REPO_ROOT", str(Path(__file__).resolve().parents[2]))
    if root not in sys.path:
        sys.path.insert(0, root)
    canonical = "scripts.hpc." + name
    path = Path(__file__).with_name(name + ".py")
    existing = sys.modules.get(canonical)
    if existing is not None and Path(existing.__file__).resolve() == path.resolve():
        return existing
    spec = importlib.util.spec_from_file_location(canonical, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[canonical] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(canonical, None)
        raise
    return module


def load_overlays():
    for name in ("training_artifact_io", "shared_daily_microbatch", "shared_full_state_ad_execution",
                 "probe_shared_execution", "probe_shared_dual", "probe_shared_serialized_updates", "probe_shared_serialized_dual"):
        load_adjacent(name)


@contextmanager
def forbid_model_compile():
    from jax._src import compiler
    original = compiler.compile_or_get_cached
    def forbidden(*args, **kwargs):
        raise RuntimeError("original full-state saved replay attempted model compilation")
    compiler.compile_or_get_cached = forbidden
    try:
        yield
    finally:
        compiler.compile_or_get_cached = original


def read_manifest(path):
    """Original-only interface; candidate/latent manifests are rejected."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("explicit original-only full-state manifest required")
    if any(key in manifest for key in ("candidate_parameters", "architectures", "candidate")):
        raise ValueError("candidate routes are not part of original schedule comparison")
    for name in ("source_profile", "input_bundle", "original_checkpoint"):
        if not manifest.get(name):
            raise ValueError(f"manifest {name} required")
    if set(manifest.get("components", {})) != {"daily", "T7", "T30"}:
        raise ValueError("original daily/T7/T30 reports and full executables required")
    for item in manifest["components"].values():
        if not item.get("report") or not item.get("executable"):
            raise ValueError("each original component needs explicit report/executable")
    if "resume" in manifest:
        resume = manifest["resume"]
        if (set(resume) != {"old", "new", "completed_pairs", "prior_report"}
                or type(resume["completed_pairs"]) is not int or resume["completed_pairs"] < 1
                or not all(resume.get(key) for key in ("old", "new", "prior_report"))
                or resume["old"] == resume["new"]):
            raise ValueError("explicit independent old/new resume paths, paired step and prior report required")
    return manifest


def read_reference(manifest):
    profile = json.loads(Path(manifest["source_profile"]).read_text(encoding="utf-8"))
    reference = json.loads(Path(profile["technical_update_report"]).read_text(encoding="utf-8"))
    # Keep the metadata path independent of training/JAX imports. These are
    # the public serialized reference's exact completed-profile gates.
    if profile.get("status") != "passed" or set(profile.get("executables", {})) != {"daily", "T7", "T30"}:
        raise ValueError("requires a completed three-executable profile")
    if reference.get("status") != "passed" or reference.get("completed_updates") != 2:
        raise ValueError("requires a completed two-update reference")
    if reference.get("technical_updates") != 2:
        raise ValueError("reference must be a technical run, not formal learning")
    return profile, reference


def validate_arguments(args, *, start_step=0, previous_attempts=0):
    if not 1 <= args.updates <= 16 or args.warmup_proposals not in (0, 1):
        raise ValueError("1..16 retained updates and zero/one warmup proposal per chain required")
    if previous_attempts + args.updates + args.warmup_proposals > 16:
        raise ValueError("per-chain16 attempt cap includes discarded warmup proposals")
    external_cap = getattr(args, "external_wall_cap_seconds", 500)
    if (external_cap not in (500, 900) or not 30 <= args.wall_seconds <= external_cap
            or (external_cap == 900 and (start_step < 1 or args.wall_seconds > 880))
            or not 10 <= args.save_reserve_seconds < args.wall_seconds):
        raise ValueError("wall cap30..500s, or explicit resumed900s external/<=880s internal, with save reserve>=10s required")
    selectors = (args.parent_cuda_device, args.worker_cuda_device)
    if any(not selector or "," in selector or any(char.isspace() for char in selector) for selector in selectors) or selectors[0] == selectors[1]:
        raise ValueError("two distinct single CUDA selectors required")
    for path in (args.output, args.output.with_suffix(".timeout.json"), args.output.with_suffix(".state-diagnostics.pkl"),
                 args.output.with_suffix(".old-diagnostic.pkl"), args.output.with_suffix(".new-diagnostic.pkl"),
                 *(args.output.with_suffix(f".pair-{index:03d}-gradients.pkl") for index in range(start_step+1, start_step+args.updates+1))):
        if path.exists():
            raise FileExistsError(path)


def validate_resume_metadata(manifest, reference):
    """Pure JSON provenance gates; array identity is checked only at load."""
    resume = manifest.get("resume")
    if resume is None: return None
    for key in ("old", "new", "prior_report"):
        if not Path(resume[key]).is_file(): raise FileNotFoundError(resume[key])
    prior = json.loads(Path(resume["prior_report"]).read_text(encoding="utf-8"))
    step = resume["completed_pairs"]
    if (prior.get("scope") != "original_only_full_parameter_and_initial_state_AD_changing_schedule_comparison"
            or prior.get("source_git_head") != reference["source_git_head"]
            or prior.get("old_schedule") != "parent daily32; worker T7+T30"
            or prior.get("new_schedule") != "parent daily32+T7; worker T30"
            or prior.get("rtol") != RTOL or prior.get("atol") != ATOL
            or prior.get("model_gradient_compilations") != 0
            or prior.get("adoption") is not False or prior.get("scientific_acceptance") is not False
            or prior.get("final_optimizer_steps") != {"old": step, "new": step}
            or prior.get("retained_diagnostic_chains") != {key: resume[key] for key in ("old", "new")}
            or type(prior.get("strict_parity_passed")) is not bool):
        raise ValueError("resume prior report source/schedule/retained chain/step identity differs")
    rows = prior.get("changing_pairs", [])
    if not rows or rows[-1].get("chain_step") != step:
        raise ValueError("resume requires a complete paired step, not partial chain advancement")
    last = rows[-1]
    ids = reference["method"]["training_plan"]["training_artifact_ids"][:2]
    if prior.get("workload", {}).get("artifact_ids") != ids or last.get("artifact_id") != ids[(step-1) % len(ids)]:
        raise ValueError("resume prior artifact cursor differs from original compact inputs")
    for name, item in manifest["components"].items():
        admission = prior.get("admissions", {}).get(name, {})
        if (admission.get("full_state_admission") is not True or admission.get("report") != item["report"]
                or admission.get("executable") != item["executable"]
                or admission.get("identity", {}).get("method") != reference["method"]):
            raise ValueError("resume prior original full-state asset/method association differs")
    attempts = prior.get("optimizer_attempts", {})
    if set(attempts) != {"old", "new"} or any(type(value) is not int or not step <= value <= 16 for value in attempts.values()):
        raise ValueError("resume prior optimizer attempt accounting unavailable/differs")
    for name in ("old", "new"):
        item = last[name]
        if (item.get("input_parameter_version") != step-1 or item.get("optimizer_step") != step
                or item.get("physical_hard_counts_zero") is not True
                or item.get("parameter_and_optimizer_finite") is not True
                or item.get("record", {}).get("branch_parameter_versions") != {key: step-1 for key in ("daily", "T7", "T30")}):
            raise ValueError("resume prior paired update/version/finite/physical identity differs")
    for key in ("gradient", "parameters", "optimizer"):
        if last["strict_comparison"]["comparisons"][key].get("passed") is not True:
            raise ValueError("resume prior full parameter/update parity is not admitted")
    return {"start_step": step, "prior_report": resume["prior_report"],
            "prior_strict_parity_passed": prior["strict_parity_passed"],
            "prior_strict_comparison": last["strict_comparison"],
            "chain_paths": {key: resume[key] for key in ("old", "new")},
            "prior_optimizer_attempts": attempts,
            "array_identity_verified": False}


def validate_resume_chain(jax, saved, *, manifest, schedule, original_parameters, original_optimizer):
    """Bind independent saved weights/moments to exact original provenance."""
    step = manifest["resume"]["completed_pairs"]
    original_manifest = {key: value for key, value in manifest.items() if key != "resume"}
    saved_manifest = {key: value for key, value in saved.get("manifest", {}).items() if key != "resume"}
    if (saved.get("scope") != "original_full_state_schedule_diagnostic_not_formal_training"
            or saved.get("schedule") != schedule or saved_manifest != original_manifest
            or saved.get("completed_diagnostic_updates") != step
            or saved.get("adoption") is not False or saved.get("scientific_acceptance") is not False):
        raise ValueError("resume snapshot manifest/schedule/step/provenance differs")
    parameters, optimizer = saved["parameters"], saved["optimizer"]
    if (type(optimizer) is not type(original_optimizer)
            or np.shape(optimizer.step) != () or np.asarray(optimizer.step).dtype != np.dtype("int32")
            or int(np.asarray(optimizer.step)) != step):
        raise ValueError("resume public AdamW identity/step differs")
    for name, actual, expected in (("parameters", parameters, original_parameters),
                                   ("optimizer", optimizer, original_optimizer)):
        if jax.tree_util.tree_structure(actual) != jax.tree_util.tree_structure(expected):
            raise ValueError("resume "+name+" tree differs")
        for value, baseline in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True):
            if (np.shape(value) != np.shape(baseline) or np.asarray(value).dtype != np.asarray(baseline).dtype
                    or not np.all(np.isfinite(np.asarray(value)))):
                raise ValueError("resume "+name+" shape/dtype/finite differs")
    if any(np.any(np.asarray(value) < 0) for value in jax.tree_util.tree_leaves(optimizer.second)):
        raise ValueError("resume AdamW second moments must be nonnegative")
    return parameters, optimizer


def validate_original_assets(manifest, reference):
    from scripts.hpc.shared_full_state_ad_execution import validate_original_admission
    admissions = {}
    for name, item in manifest["components"].items():
        report = json.loads(Path(item["report"]).read_text(encoding="utf-8"))
        validate_original_admission(report, name, reference)
        declared = report.get("reference_executable", report.get("executable"))
        if not declared or Path(declared).name != Path(item["executable"]).name:
            raise ValueError("original admitted executable association differs")
        if not Path(item["executable"]).is_file():
            raise FileNotFoundError(item["executable"])
        admissions[name] = {"report": item["report"], "executable": item["executable"],
                            "identity": report["identity"], "full_state_admission": True}
    for name in ("input_bundle", "original_checkpoint"):
        if not Path(manifest[name]).is_file():
            raise FileNotFoundError(manifest[name])
    return admissions


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def validate_bundle(jax, bundle, manifest, reference):
    if (bundle.get("schema_version") != "shared_serialized_dual_inputs_v1"
            or bundle.get("source_git_head") != reference["source_git_head"]
            or bundle.get("method") != reference["method"]
            or Path(bundle.get("initial_checkpoint", "")).name != Path(manifest["original_checkpoint"]).name):
        raise ValueError("original compact bundle source/method/checkpoint differs")
    ids = tuple(bundle.get("artifact_ids", ()))
    expected = tuple(reference["method"]["training_plan"]["training_artifact_ids"][:2])
    payloads = bundle.get("payloads", ())
    if ids != expected or len(set(ids)) != 2 or len(payloads) != 2:
        raise ValueError("two original declared TRAIN artifacts required")
    import hashlib
    fingerprints = []
    for payload in payloads:
        leaves = jax.tree_util.tree_leaves(payload["daily"])
        if not leaves or any(not np.shape(value) or np.shape(value)[0] != 32 for value in leaves):
            raise ValueError("each daily leaf must contain exactly32 rows")
        hashes = []
        for start in range(0, 32, 4):
            digest = hashlib.sha256()
            for value in leaves:
                array = np.ascontiguousarray(value[start:start+4])
                digest.update(str((array.shape, array.dtype.str)).encode())
                digest.update(array.tobytes())
            hashes.append(digest.hexdigest())
        if len(set(hashes)) != 8:
            raise ValueError("eight distinct actual daily micro4 chunks required")
        for horizon in (7, 30):
            case = payload["cases"][horizon]
            if len(case) != 5 or np.shape(case[0]) != (3854,) or np.shape(case.days[0].continuous_state)[0] != horizon:
                raise ValueError("original full-state recurrent case/horizon differs")
        fingerprints.append(hashes)
    return {"artifact_ids": ids, "daily_micro4_sha256": fingerprints}


def load_evaluators(jax, manifest, reference, parameters, payload, components):
    from scripts.hpc.shared_full_state_ad_execution import FullStateADEvaluator, load_full_state_executable
    from scripts.hpc.run_shared_learning import memory_sizes
    evaluators, metadata = {}, {}
    for name in components:
        began = time.monotonic()
        if name == "daily":
            sample = jax.tree_util.tree_map(lambda value: value[:4], payload["daily"])
            arguments = (parameters, sample.inputs.continuous_state, sample)
        else:
            arguments = (parameters, *payload["cases"][int(name[1:])])
        item = manifest["components"][name]
        executable, metadata[name] = load_full_state_executable(jax, item["executable"], arguments,
            component=name, admission=item["report"], architecture="original", expected_reference=reference,
            memory_sizes=memory_sizes, compile_guard=forbid_model_compile, memory_limit_bytes=MEMORY_LIMIT)
        metadata[name].update(source_executable=item["executable"], load_ABI_seconds=time.monotonic()-began)
        evaluators[name] = FullStateADEvaluator(jax, executable, name, compile_guard=forbid_model_compile)
    return evaluators, metadata


def attach_state(component, evaluator):
    records = evaluator.take_records()
    daily = evaluator.component == "daily"
    if len(records) != (8 if daily else 1):
        raise ValueError("complete full-state AD call count differs")
    arrays = [record["initial_state_gradient"] for record in records]
    if daily:
        component["initial_state_gradients_micro4_mean"] = np.stack(arrays)
        state = np.concatenate(arrays, axis=0) / 8
    else:
        state = arrays[0]
    component["initial_state_gradient"] = state
    component["initial_state_AD"] = {"shape": list(state.shape), "dtype": str(state.dtype),
        "finite": bool(np.all(np.isfinite(state))), "nonzero": int(np.count_nonzero(state)), "coordinates": int(state.size)}
    component["full_state_ad_calls"] = len(records)


def full_state_branch(branch, evaluators, *, daily=False):
    def call(*arguments):
        began = time.monotonic()
        for evaluator in evaluators.values():
            evaluator.reset()
        result = branch(*arguments)
        if daily:
            attach_state(result, evaluators["daily"])
            result["seconds"] = time.monotonic()-began
            return result
        components, _seconds = result
        for horizon, component in components.items():
            attach_state(component, evaluators[f"T{horizon}"])
        return components, time.monotonic()-began
    return call


def state_snapshot(values):
    return {name: {key: component[key] for key in ("loss", "auxiliary", "initial_state_gradient", "initial_state_AD", "full_state_ad_calls")}
            | ({"initial_state_gradients_micro4_mean": component["initial_state_gradients_micro4_mean"]} if name == "daily" else {})
            for name, component in values["components"].items()}


def compare_pair(jax, old, new):
    """Keep numerical strict failures as data, without permitting adoption."""
    from scripts.hpc.probe_shared_execution import compare_trees
    def compare(left, right):
        return compare_trees(jax, left, right, rtol=RTOL, atol=ATOL)
    comparisons = {key: compare(old[key], new[key]) for key in ("gradient", "parameters", "optimizer")}
    for name in ("daily", "T7", "T30"):
        left, right = old["components"][name], new["components"][name]
        comparisons[name] = {
            "loss_auxiliary_and_parameter_gradient": compare(
                {key: left[key] for key in ("loss", "auxiliary", "gradient", "hard", "norm")},
                {key: right[key] for key in ("loss", "auxiliary", "gradient", "hard", "norm")}),
            "initial_state_gradient": compare(left["initial_state_gradient"], right["initial_state_gradient"])}
        if name == "daily":
            comparisons[name]["raw_micro4_initial_state_gradients"] = compare(
                left["initial_state_gradients_micro4_mean"], right["initial_state_gradients_micro4_mean"])
    passed = all(item["passed"] for item in comparisons.values() if "passed" in item)
    passed &= all(item["passed"] for name in ("daily", "T7", "T30") for item in comparisons[name].values())
    return {"passed": bool(passed), "comparisons": comparisons,
            "failure_policy": "strict failures retained; finite/physical valid timings may continue; adoption remains false"}


def require_memory(memory, label):
    peak = (memory or {}).get("peak_bytes_in_use")
    if peak is None or not np.isfinite(peak) or not 0 <= peak <= MEMORY_LIMIT:
        raise MemoryError(label + " process aggregate observed memory unavailable/exceeds24GiB")


def atomic_pickle(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parameter_snapshot(jax, values, comparison=None):
    """Owned host arrays for every parameter gradient, and failed/partial states.

    Eight original FP32 gradient trees cost about 2.076 GB per full pair.
    This transfer and persistence is outside complete-update timing but inside
    the wall deadline. Input states never enter the optimizer.
    """
    def host(tree):
        return jax.tree_util.tree_map(lambda value: np.array(jax.device_get(value), copy=True), tree)
    failed_update = comparison is None or any(not comparison["comparisons"][key]["passed"]
        for key in ("gradient", "parameters", "optimizer"))
    return {name: {"mixed_gradient": host(item["gradient"]),
                  "component_gradients": {key: host(component["gradient"])
                                          for key, component in item["components"].items()},
                  "component_initial_state_gradients": {
                      key: host(component["initial_state_gradient"])
                      for key, component in item["components"].items()},
                  **({"daily_initial_state_gradients_micro4_mean":
                      host(item["components"]["daily"]["initial_state_gradients_micro4_mean"])}
                     if "daily" in item["components"] else {}),
                  **({"parameters": host(item["parameters"]), "optimizer": host(item["optimizer"])}
                     if failed_update else {})}
            for name, item in values.items()}


class ExecutionDeadline:
    """A process deadline inside the external allocation, never a GPU budget."""
    def __init__(self, args, *, started=PROCESS_STARTED, clock=time.monotonic, exit_process=os._exit):
        self.args, self.started, self.clock, self.exit_process = args, started, clock, exit_process
        self.process = None
        self.snapshot = {"status": "running", "scientific_acceptance": False, "adoption": False}
        self.timer = None

    def elapsed(self): return self.clock()-self.started

    def capture(self, report):
        self.snapshot = json.loads(json.dumps(report, default=lambda value: np.asarray(value).tolist()))

    def register(self, process): self.process = process

    def may_start_update(self):
        return self.elapsed() < self.args.wall_seconds-self.args.save_reserve_seconds

    def may_start_pair(self, previous_pair_seconds):
        # Use measured AD/IPC/comparison/download/save cost, with headroom.
        required = 1.25*previous_pair_seconds+5.0
        return self.elapsed()+required < self.args.wall_seconds-self.args.save_reserve_seconds

    def expire(self):
        report = dict(self.snapshot)
        cleanup_errors, removed = [], []
        if self.process is not None:
            try:
                if self.process.poll() is None: self.process.kill()
                self.process.wait(timeout=.5)
            except Exception as error:
                cleanup_errors.append({"scope": "worker_kill_wait", "error": str(error)})
        for item in report.get("owned_temporary_artifacts", []):
            path = Path(item)
            # Only this process's explicitly tracked pickle temporaries.
            if (path.parent.resolve() != self.args.output.parent.resolve()
                    or not path.name.startswith("."+self.args.output.stem+".")
                    or not path.name.endswith(f".{os.getpid()}.tmp")):
                cleanup_errors.append({"path": str(path), "error": "refused nonexclusive temporary path"})
                continue
            try:
                if path.exists():
                    path.unlink()
                    removed.append(str(path))
            except Exception as error:
                cleanup_errors.append({"path": str(path), "error": str(error)})
        report.update(status="strict_failed" if report.get("strict_parity_passed") is False else "incomplete_timeout",
            hard_deadline_reached=True, execution_elapsed_seconds=self.elapsed(), adoption=False, production_adoption=False,
            scientific_acceptance=False, timeout_scope="all startup/load/update/comparison/save/exit within process deadline",
            timeout_removed_temporary_artifacts=removed, timeout_cleanup_errors=cleanup_errors)
        from scripts.hpc.training_artifact_io import _atomic_json
        # Use a separate JSON temporary filename so a simultaneous parent
        # stage write cannot collide with the watchdog's atomic write.
        timeout_path = self.args.output.with_suffix(".timeout.json")
        try:
            _atomic_json(timeout_path, report)
            os.replace(timeout_path, self.args.output)
        finally:
            self.exit_process(124)

    def __enter__(self):
        remaining = max(.01, self.args.wall_seconds-self.elapsed())
        self.timer = threading.Timer(remaining, self.expire)
        self.timer.daemon = True
        self.timer.start()
        return self

    def __exit__(self, *_):
        if self.timer is not None: self.timer.cancel()


def worker_class():
    from scripts.hpc.probe_shared_serialized_dual import RecurrentWorker
    class OriginalFullStateWorker(RecurrentWorker):
        def __init__(self, manifest, *, cuda_device, parameters, cases, deadline=None, echo_test=False, parameter_version=0):
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=cuda_device, PYTHONDONTWRITEBYTECODE="1")
            command = [sys.executable, "-B", str(Path(__file__).with_name("probe_shared_original_full_state_schedule_worker.py")),
                       "--manifest", str(manifest)]
            if echo_test: command.append("--echo-test")
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
                env=environment, bufsize=0, cwd=os.environ.get("ORCHIDEE_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
            self.closed, self.parameter_version = False, parameter_version
            if deadline is not None: deadline.register(self.process)
            pickle.dump({"kind": "initialize", "parameters": parameters, "cases": cases,
                         "parameter_version": parameter_version}, self.process.stdin,
                        protocol=pickle.HIGHEST_PROTOCOL)
            self.process.stdin.flush()
            ready = self._receive()
            if ready.get("status") != "ready":
                self.close()
                raise RuntimeError(f"original full-state worker startup failed: {ready}")
            self.metadata = ready
    return OriginalFullStateWorker


def timing_summary(rows):
    indices = [row["chain_step"] for row in rows]
    def selected(chosen):
        values = {name: [row[name]["complete_seconds"] for row in chosen] for name in ("old", "new")}
        medians = {name: float(np.median(value)) if value else None for name, value in values.items()}
        return {"pair_indices": [row["chain_step"] for row in chosen], "paired_samples": len(chosen),
                "seconds": values, "medians": medians,
                "old_over_new": medians["old"]/medians["new"] if chosen else None,
                "two_card_hours": {name: 2*sum(value)/3600 for name, value in values.items()}}
    return {"completed_pairs": len(rows), "minimum_three_pairs_complete": len(rows) >= 3,
            "minimum_three_post_first_pairs_complete": len(rows[1:]) >= 3,
            "first_step_boundary_index": indices[0] if indices else None,
            "all_pairs": selected(rows), "post_first_pairs": selected(rows[1:]),
            "requested_scope": "first pair includes helper warm boundary unless discarded warmup explicitly requested"}


def execute_pairs(*, jax, old, new, payloads, artifact_ids, args, deadline, emit, save_states, save_pair):
    rows, diagnostics, attempts = [], [], dict(getattr(args, "prior_optimizer_attempts", {"old": 0, "new": 0}))
    strict = getattr(args, "initial_strict_parity_passed", True)
    start_step = getattr(args, "start_step", 0)
    def proposal(name, executor, payload, persist):
        attempts[name] += 1
        if attempts[name] > 16: raise ValueError("per-chain optimizer-attempt cap exceeded")
        emit("optimizer_attempt", optimizer_attempts=dict(attempts), active_schedule=name,
             active_parameter_version=int(np.asarray(executor.optimizer.step)))
        began = time.monotonic()
        executor.worker.parameter_version = int(np.asarray(executor.optimizer.step))
        record, values = executor.step(daily_payload=payload["daily"], recurrent_payload=payload["cases"], persist=persist)
        jax.block_until_ready((values["parameters"], values["optimizer"]))
        elapsed = time.monotonic()-began
        record["initial_state_AD"] = {key: item["initial_state_AD"] for key, item in values["components"].items()}
        return {"complete_seconds": elapsed, "record": record}, values
    for _ in range(args.warmup_proposals):
        for name, executor in (("old", old), ("new", new)):
            if not deadline.may_start_update(): return rows, diagnostics, attempts, strict
            _record, values = proposal(name, executor, jax.device_put(payloads[start_step % len(payloads)]), False)
            del values
    for index in range(start_step, start_step+args.updates):
        if not deadline.may_start_update(): break
        if rows and hasattr(deadline, "may_start_pair"):
            estimate = max(row["pair_elapsed_including_compare_and_save_seconds"] for row in rows)
            if not deadline.may_start_pair(estimate):
                emit("next_pair_predicted_budget_stop", previous_pair_max_seconds=estimate,
                     conservative_next_pair_seconds=1.25*estimate+5.)
                break
        pair_started = time.monotonic()
        payload = jax.device_put(payloads[index % len(payloads)])
        pair, values = {"chain_step": index+1, "artifact_id": artifact_ids[index % len(artifact_ids)]}, {}
        for name in (("old", "new") if index % 2 == 0 else ("new", "old")):
            if not deadline.may_start_update():
                if values:
                    pair["parameter_gradient_artifact"] = save_pair(index+1, values, None)
                emit("partial_pair_budget_stop", partial_pair=pair, optimizer_attempts=dict(attempts),
                     retained_chain_steps={"old": int(np.asarray(old.optimizer.step)),
                                           "new": int(np.asarray(new.optimizer.step))})
                return rows, diagnostics, attempts, bool(strict)
            executor = old if name == "old" else new
            if int(np.asarray(executor.optimizer.step)) != index:
                raise ValueError("fresh independent schedule chain version differs")
            try:
                pair[name], values[name] = proposal(name, executor, payload, True)
            except Exception:
                if values:
                    pair["parameter_gradient_artifact"] = save_pair(index+1, values, None)
                    emit("partial_pair_execution_error", partial_pair=pair, optimizer_attempts=dict(attempts))
                raise
            if int(np.asarray(executor.optimizer.step)) != index+1:
                raise ValueError("public AdamW chain failed to advance")
            versions = pair[name]["record"]["branch_parameter_versions"]
            if any(version != index for version in versions.values()):
                raise ValueError("executed schedule component parameter version differs")
            pair[name].update(input_parameter_version=index, optimizer_step=index+1,
                parent_device_memory=old.jax.local_devices()[0].memory_stats(),
                worker_device_memory=executor.worker.last_device_memory,
                parameter_and_optimizer_finite=True, physical_hard_counts_zero=pair[name]["record"]["all_hard_counts_zero"])
            require_memory(pair[name]["parent_device_memory"], "parent")
            require_memory(pair[name]["worker_device_memory"], "worker T7+T30 resident")
        diagnostics.append({"chain_step": index+1, "artifact_id": pair["artifact_id"],
            "old": state_snapshot(values["old"]), "new": state_snapshot(values["new"])})
        # Preserve full states before the large parameter-tree comparison.
        # A watchdog during comparison leaves raw evidence for this pair.
        save_states(diagnostics)
        pair["strict_comparison"] = compare_pair(jax, values["old"], values["new"])
        strict &= pair["strict_comparison"]["passed"]
        # Make strict failures sticky before budgeted multi-GB persistence.
        emit("pair_comparison_completed", strict_parity_passed=bool(strict),
             pair_awaiting_gradient_save=pair)
        pair["parameter_gradient_artifact"] = save_pair(index+1, values, pair["strict_comparison"])
        pair["pair_elapsed_including_compare_and_save_seconds"] = time.monotonic()-pair_started
        rows.append(pair)
        del values
        emit("changing_pair_completed", changing_pairs=rows, optimizer_attempts=dict(attempts),
             strict_parity_passed=bool(strict), timing=timing_summary(rows))
    return rows, diagnostics, attempts, bool(strict)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--parent-cuda-device", required=True)
    result.add_argument("--worker-cuda-device", required=True)
    result.add_argument("--updates", type=int, default=4)
    result.add_argument("--warmup-proposals", type=int, default=0)
    result.add_argument("--wall-seconds", type=float, default=480)
    result.add_argument("--external-wall-cap-seconds", type=int, choices=(500, 900), default=500,
                        help="external owner's cap; interface declaration does not authorize an allocation")
    result.add_argument("--save-reserve-seconds", type=float, default=30)
    result.add_argument("--dry-run", action="store_true")
    return result


def dry_run(args):
    manifest = read_manifest(args.manifest)
    # Metadata-only validation: no bundle/checkpoint unpickling, JAX backend,
    # binary deserialization, helper JIT or GPU invocation.
    load_adjacent("shared_full_state_ad_execution")
    _profile, reference = read_reference(manifest)
    resume = validate_resume_metadata(manifest, reference)
    validate_arguments(args, start_step=resume["start_step"] if resume else 0,
                       previous_attempts=max(resume["prior_optimizer_attempts"].values()) if resume else 0)
    admissions = validate_original_assets(manifest, reference)
    return {"status": "dry_run_passed", "scope": "original_only_full_state_schedule_metadata",
            "source_git_head": reference["source_git_head"], "parent_cuda_selector": args.parent_cuda_device,
            "worker_cuda_selector": args.worker_cuda_device, "updates_per_chain": args.updates,
            "optimizer_attempt_upper_bound_per_chain": args.updates+args.warmup_proposals+(max(resume["prior_optimizer_attempts"].values()) if resume else 0),
            "full_parameter_gradient_array_bytes_per_pair": 2*4*64884481*4,
            "full_parameter_gradient_array_bytes_requested_pairs": args.updates*2*4*64884481*4,
            "wall_cap_seconds": args.wall_seconds, "two_card_gpu_seconds_process_cap": 2*args.wall_seconds,
            "external_wall_cap_seconds": args.external_wall_cap_seconds,
            "external_allocation_authority": "HPC owner's explicit funding/allocation; cap option does not grant authorization or transfer daily-only balance",
            "resume": resume, "end_step_if_completed": (resume["start_step"] if resume else 0)+args.updates,
            "components": {name: {"report": item["report"], "executable": item["executable"]} for name, item in admissions.items()},
            "input_bundle": manifest["input_bundle"], "original_checkpoint": manifest["original_checkpoint"],
            "GPU_executed": False, "scientific_acceptance": False, "adoption": False}


def run(args):
    manifest = read_manifest(args.manifest)
    _profile, reference = read_reference(manifest)
    resume = validate_resume_metadata(manifest, reference)
    args.start_step = resume["start_step"] if resume else 0
    args.initial_strict_parity_passed = resume["prior_strict_parity_passed"] if resume else True
    args.prior_optimizer_attempts = resume["prior_optimizer_attempts"] if resume else {"old": 0, "new": 0}
    validate_arguments(args, start_step=args.start_step, previous_attempts=max(args.prior_optimizer_attempts.values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    load_adjacent("training_artifact_io")
    from scripts.hpc.training_artifact_io import _atomic_json
    report = {"status": "running", "scope": "original_only_full_parameter_and_initial_state_AD_changing_schedule_comparison",
        "scientific_acceptance": False, "production_adoption": False, "adoption": False, "formal_training_updates": 0,
        "model_gradient_compilations": 0, "rtol": RTOL, "atol": ATOL, "gpu_count": 2,
        "wall_cap_seconds": args.wall_seconds, "save_exit_reserve_seconds": args.save_reserve_seconds,
        "old_schedule": "parent daily32; worker T7+T30", "new_schedule": "parent daily32+T7; worker T30",
        "strict_parity_passed": args.initial_strict_parity_passed, "requested_pairs": args.updates,
        "resume": resume, "starting_optimizer_step": args.start_step,
        "external_wall_cap_seconds": args.external_wall_cap_seconds,
        "prior_strict_state_replay_evidence": manifest.get("prior_strict_state_replay_evidence", []),
        "self_replay_certification": "not established by cross-schedule comparison; prior genuine strict failures are not superseded",
        "external_budget_authority": "HPC owner explicit allocation/funding scope; extra daily-only allowance excluded; cap option grants no authorization"}
    resources, executors = [], {}
    with ExecutionDeadline(args) as deadline:
        def emit(stage, **fields):
            elapsed = deadline.elapsed()
            report.update(stage=stage, execution_elapsed_seconds=elapsed, **fields)
            event = {"stage": stage, "execution_elapsed_seconds": elapsed}
            for key in ("active_schedule", "active_parameter_version", "pending_array_artifact",
                        "latest_parameter_gradient_artifact", "retained_chain_artifact", "current_chain_step"):
                if key in fields: event[key] = fields[key]
            report.setdefault("stage_events", []).append(event)
            deadline.capture(report)
            _atomic_json(args.output, report)
            print("original_full_state_schedule stage="+stage, flush=True)
        def save_states(diagnostics):
            path = args.output.with_suffix(".state-diagnostics.pkl")
            track_save("save_full_state_arrays", path)
            atomic_pickle(path, {"scope": "raw_full_state_AD_original_old_new", "manifest": manifest,
                "daily_state_policy": "retain micro4 mean derivatives; concatenate8 then divide8 for daily32 mean",
                "diagnostics": diagnostics, "adoption": False})
            report["state_diagnostics_artifact"] = str(path)
            emit("full_state_arrays_saved")
        def track_save(stage, path):
            temporary = str(path.with_name(f".{path.name}.{os.getpid()}.tmp"))
            owned = report.setdefault("owned_temporary_artifacts", [])
            if temporary not in owned: owned.append(temporary)
            emit(stage, pending_array_artifact=str(path), pending_array_temporary=temporary,
                 temporary_cleanup_scope="only listed current-process temporary paths; preserve completed artifacts")
        def save_chains():
            paths, steps = {}, {}
            for name, executor in executors.items():
                step = int(np.asarray(executor.optimizer.step))
                steps[name] = step
                if step:
                    path = args.output.with_suffix(f".{name}-diagnostic.pkl")
                    track_save("save_retained_"+name+"_chain", path)
                    atomic_pickle(path, {"scope": "original_full_state_schedule_diagnostic_not_formal_training",
                        "schedule": name, "manifest": manifest, "completed_diagnostic_updates": step,
                        "parameters": jax.device_get(executor.parameters), "optimizer": jax.device_get(executor.optimizer),
                        "adoption": False, "scientific_acceptance": False})
                    paths[name] = str(path)
                    emit("retained_"+name+"_chain_saved", retained_chain_artifact=str(path), current_chain_step=step)
            report.update(retained_diagnostic_chains=paths, final_optimizer_steps=steps)
        def save_pair(index, values, comparison):
            path = args.output.with_suffix(f".pair-{index:03d}-gradients.pkl")
            track_save("save_pair_full_parameter_arrays", path)
            arrays = parameter_snapshot(jax, values, comparison)
            emit("pair_parameter_arrays_downloaded", current_chain_step=index)
            atomic_pickle(path, {"scope": "full_parameter_gradient_original_schedule_pair",
                "chain_step": index, "manifest": manifest, "complete_pair": set(values) == {"old", "new"},
                "strict_comparison": comparison, "arrays": arrays,
                "adoption": False, "scientific_acceptance": False})
            del arrays
            emit("pair_full_parameter_arrays_saved", latest_parameter_gradient_artifact=str(path))
            return str(path)
        try:
            emit("initialize_backend")
            existing_selector = os.environ.get("CUDA_VISIBLE_DEVICES")
            if existing_selector and existing_selector != args.parent_cuda_device:
                raise ValueError("parent CUDA environment differs from explicit selector")
            os.environ["CUDA_VISIBLE_DEVICES"] = args.parent_cuda_device
            jax, devices = load_adjacent("orcjax_gpu_runtime").initialize_gpu_backend()
            jax.config.update("jax_enable_x64", True)
            if len(devices) != 1 or devices[0].platform != "gpu": raise ValueError("one visible parent GPU required")
            import jax.numpy as jnp
            load_overlays()
            from scripts.hpc import run_shared_learning as learning
            from scripts.hpc.probe_shared_serialized_dual import RemoteDualExecutor
            from scripts.hpc.probe_shared_dual import make_microbatch_daily_branch, make_recurrent_branch, make_component_combiner
            from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
            _profile, reference = read_reference(manifest)
            admissions = validate_original_assets(manifest, reference)
            bundle = load_pickle(manifest["input_bundle"])
            workload = validate_bundle(jax, bundle, manifest, reference)
            checkpoint = learning.load_learning_checkpoint(Path(manifest["original_checkpoint"]), expected_method=reference["method"])
            if checkpoint["completed_updates"] != 0: raise ValueError("original checkpoint must be update0")
            leaves = jax.tree_util.tree_leaves(checkpoint["parameters"])
            if (len(leaves) != 300 or sum(np.size(value) for value in leaves) != 64884481
                    or any(np.asarray(value).dtype != np.dtype("float32") or not np.all(np.isfinite(value)) for value in leaves)):
                raise ValueError("original full FP32 parameter tree differs")
            chains = {}
            for name in ("old", "new"):
                if resume:
                    saved = load_pickle(manifest["resume"][name])
                    host_chain = validate_resume_chain(jax, saved, manifest=manifest, schedule=name,
                        original_parameters=checkpoint["parameters"], original_optimizer=checkpoint["optimizer"])
                    chains[name] = jax.device_put(host_chain)
                    del saved, host_chain
                else:
                    chains[name] = jax.device_put((checkpoint["parameters"], checkpoint["optimizer"]))
            if resume:
                resume["array_identity_verified"] = True
            parameters, optimizer = chains["old"]
            del checkpoint, leaves
            payloads = bundle["payloads"]
            local = jax.device_put(payloads[args.start_step % len(payloads)])
            operations = learning.make_tree_operations(jax, jnp, 32)
            update, combine = make_update(reference["method"]["daily_protocol"]), make_component_combiner(jax)
            evaluators, metadata = load_evaluators(jax, manifest, reference, parameters, local, ("daily", "T7"))
            require_memory(devices[0].memory_stats(), "parent daily+T7 executables resident before first update")
            emit("start_original_full_state_worker", workload=workload, source_git_head=reference["source_git_head"],
                 admissions=admissions, parent_executables=metadata)
            worker = worker_class()(args.manifest, cuda_device=args.worker_cuda_device,
                parameters=jax.device_get(parameters), cases=payloads[args.start_step % len(payloads)]["cases"],
                deadline=deadline, parameter_version=args.start_step)
            resources.append(worker)
            for name in ("T7", "T30"):
                if worker.metadata["executables"][name]["source_executable"] != manifest["components"][name]["executable"]:
                    raise ValueError("worker original executable association differs")
            if worker.metadata.get("initial_parameter_version") != args.start_step:
                raise ValueError("worker admitted initial parameter version differs")
            daily = full_state_branch(make_microbatch_daily_branch(jax=jax, jnp=jnp, device=devices[0],
                evaluator=evaluators["daily"], operations=operations, microbatch_size=4, expected_count=32,
                preserve_auxiliary=True), {"daily": evaluators["daily"]}, daily=True)
            seven = full_state_branch(make_recurrent_branch(jax=jax, device=devices[0],
                evaluators={7: evaluators["T7"]}, operations=operations, horizons=(7,), preserve_auxiliary=True),
                {"T7": evaluators["T7"]})
            for name, local_recurrent in (("old", None), ("new", seven)):
                executor = RemoteDualExecutor(jax=jax, parameters=chains[name][0], optimizer=chains[name][1],
                    daily_branch=daily, worker=worker, combine=combine, update=update, finite=operations["finite"],
                    local_recurrent_branch=local_recurrent, preserve_components=True, audit_parameter_versions=True)
                executors[name] = executor
                resources.append(executor)
            # Executors own their independent current chains; do not retain
            # extra references to both resume starting trees for the whole run.
            del chains, parameters, optimizer, local
            emit("ready_no_model_compile", worker=worker.metadata, warmup_proposals_per_chain=args.warmup_proposals,
                 helper_warm_policy="default no discarded warmup; first changing pair is explicit helper/weight boundary")
            rows, diagnostics, attempts, strict = execute_pairs(jax=jax, old=executors["old"], new=executors["new"],
                payloads=payloads, artifact_ids=workload["artifact_ids"], args=args, deadline=deadline,
                emit=emit, save_states=save_states, save_pair=save_pair)
            summary = timing_summary(rows)
            save_chains()
            if not diagnostics: save_states([])
            report.update(strict_parity_passed=strict, changing_pairs=rows, timing=summary, optimizer_attempts=attempts,
                requested_pairs_complete=len(rows) == args.updates, parent_device_memory=devices[0].memory_stats(),
                worker_device_memory=worker.last_device_memory)
            require_memory(report["parent_device_memory"], "parent")
            require_memory(report["worker_device_memory"], "worker aggregate")
            status = ("strict_failed" if not strict else "passed"
                if summary["minimum_three_pairs_complete"] and len(rows) == args.updates
                    and (not resume or summary["minimum_three_post_first_pairs_complete"]) else "incomplete")
            emit("completed", status=status)
        except Exception as error:
            report.update(status="strict_failed" if report.get("strict_parity_passed") is False else "error",
                          error_type=type(error).__name__, error=str(error))
            try: save_chains()
            except Exception as save_error: report["preservation_error"] = str(save_error)
            emit("exception")
            raise
        finally:
            emit("close_resources_started")
            for resource in reversed(resources): resource.close()
            emit("close_resources_completed")
    return report


if __name__ == "__main__":
    args = parser().parse_args()
    result = dry_run(args) if args.dry_run else run(args)
    if args.dry_run: print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] in ("passed", "dry_run_passed") else 1)
