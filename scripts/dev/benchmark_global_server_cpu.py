"""Allocated server CPU scalar, persistent pool and batch full-physics benchmark.

The science is exclusively the existing audited transition. Physical execution
requires paid Slurm admission; --dry-run and --synthetic-smoke are non-model.
Resource affinity and thread environments are set before importing JAX.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import pickle
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESS_STARTED = time.perf_counter()
sys.path.insert(0, str(ROOT))

from scripts.dev.global_cpu_child_environment import snapshot as child_snapshot
from scripts.dev.global_cpu_child_environment import validate_environment
from scripts.dev.global_server_resources import admit, bind, physical_cores, thread_affinity


def seal_candidate_validation_arrays(output_dir, identity, order, permuted, replaced, materialized, *, jax, np):
    import hashlib

    from scripts.dev.probe_global_batch_cache import atomic_bytes

    evidence = {"schema": "global_cpu48_candidate_validation_arrays_v1",
        "candidate_identity": identity, "permutation_index": order,
        "replacement_index": list(range(len(order) - 1)) + [0],
        "permutation": permuted, "replacement": replaced, "frozen_reset": materialized}
    for name in ("permutation", "replacement", "frozen_reset"):
        evidence[name] = jax.tree_util.tree_map(
            lambda value: np.asarray(jax.device_get(value)), evidence[name])
    raw = pickle.dumps(evidence, protocol=pickle.HIGHEST_PROTOCOL)
    evidence_path = Path(output_dir) / "candidate_validation_arrays.pkl"
    atomic_bytes(evidence_path, raw)
    return {"path": str(evidence_path), "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw), "schema": evidence["schema"]}


def record_candidate_batch_isolation(recorder, size, isolation_reference, permuted, replaced, *, jax, np):
    order = list(reversed(range(size)))
    replacement_index = list(range(size - 1)) + [0]
    recorder.gate(f"B{size}_server_batch_cross_group_permutation", permuted,
        jax.tree_util.tree_map(lambda value: np.asarray(value)[np.asarray(order)], isolation_reference))
    recorder.gate(f"B{size}_server_batch_replacement_isolation", replaced,
        jax.tree_util.tree_map(lambda value: np.asarray(value)[np.asarray(replacement_index)], isolation_reference))


def record_candidate_batch_reset(recorder, size, output, materialized):
    recorder.gate(f"B{size}_batch_frozen_reset_repeat", materialized, output)


def seal_candidate_batch_arrays(output_dir, identity, size, permuted, replaced, materialized, *, jax, np):
    order = list(reversed(range(size)))
    return seal_candidate_validation_arrays(output_dir, identity, order, permuted,
        replaced, materialized, jax=jax, np=np)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=["scalar", "pool", "batch"], required=True)
    result.add_argument("--cpu-budget", type=int, choices=[1, 4, 48], required=True)
    result.add_argument("--sizes", type=int, nargs="+", choices=[16, 64], default=[16, 64])
    result.add_argument("--prepared-input", type=Path)
    result.add_argument("--gpu-reference-dir", type=Path)
    result.add_argument("--expected-gpu-reference-sha256", help="Optional exact sealed GPU reference artifact identity; CPU48 coordinator supplies it.")
    result.add_argument("--cpu-reference-dir", type=Path)
    result.add_argument("--validation-mode", choices=["strict", "exploratory"], default="strict")
    result.add_argument("--cpu-reference-role", choices=["accepted", "cpu-internal"], default="accepted")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--config", type=Path, default=ROOT / "configs/orchidee_man_250919.yaml")
    result.add_argument("--truth-root", type=Path, default=ROOT)
    result.add_argument("--baseline-ref", default="133e35aec7b83b9d79888c46092f64d518e97299")
    result.add_argument("--warm-repeats", type=int, default=5)
    result.add_argument("--cache-mode", choices=["off", "persistent"], default="off")
    result.add_argument("--cache-root", type=Path)
    result.add_argument("--memory-guard-gib", type=float, help="Host MemAvailable guard; scalar/batch1/4 24GiB, batch48 64GiB, pool96GiB.")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--synthetic-smoke", action="store_true")
    result.add_argument("--synthetic-stdlib-only", action="store_true", help="Pool protocol smoke without even a toy JAX compile.")
    result.add_argument("--require-child-environment", action="store_true", help="Require exact coordinator thread environment before JAX.")
    result.add_argument("--cpu-reference-admission-record", type=Path)
    result.add_argument("--expected-cpu-reference-admission-sha256")
    return result


def validate(args):
    admission = getattr(args, "cpu_reference_admission_record", None)
    admission_sha = getattr(args, "expected_cpu_reference_admission_sha256", None)
    if admission is not None or admission_sha is not None:
        import re
        if (admission is None or not re.fullmatch(r"[0-9a-f]{64}", admission_sha or "") or
                args.mode != "batch" or args.cpu_budget != 48 or args.cpu_reference_dir is None):
            raise ValueError("external scalar admission requires batch48, directory, record and exact SHA256")
    if args.validation_mode == "strict" and args.cpu_reference_role == "cpu-internal":
        raise ValueError("cpu-internal reference role requires exploratory validation")
    if args.mode == "scalar" and args.cpu_budget != 1 or args.mode == "pool" and args.cpu_budget != 4:
        raise ValueError("scalar requires budget1; pool requires budget4")
    if args.cpu_budget == 48 and args.sizes != [64]:
        raise ValueError("batch48 requires native FULL64 unique-point workload, sizes64 only")
    if args.warm_repeats < 5 or args.sizes != sorted(set(args.sizes)):
        raise ValueError("at least five repeats and ascending unique16/64 sizes required")
    if not args.synthetic_smoke and (args.prepared_input is None or args.gpu_reference_dir is None):
        raise ValueError("physical execution requires frozen prepared input and sealed GPU numerical reference")
    if not args.synthetic_smoke and args.mode != "scalar" and args.cpu_reference_dir is None:
        raise ValueError("pool/batch acceptance requires --cpu-reference-dir with complete server scalar stages")
    if args.memory_guard_gib is not None and (not math.isfinite(args.memory_guard_gib) or args.memory_guard_gib <= 0):
        raise ValueError("positive host memory guard required")
    expected = getattr(args, "expected_gpu_reference_sha256", None)
    if expected is not None and (len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected)):
        raise ValueError("expected GPU reference SHA256 must be exactly64 lowercase hexadecimal characters")


def memory_guard(gib, path=Path("/proc/meminfo")):
    available = next(int(line.split()[1]) * 1024 for line in path.read_text().splitlines() if line.startswith("MemAvailable:"))
    result = {"MemAvailable_bytes": available, "conservative_required_bytes": int(gib * 1024**3),
              "scope": "host RSS estimate, not a Slurm whole-node memory reservation or OOM guarantee"}
    if available < result["conservative_required_bytes"]:
        raise RuntimeError(f"insufficient host MemAvailable before compilation: {result}")
    return result


def memory_snapshot(label, proc_root=Path("/proc")):
    """Phase observations only; neither a reservation nor sampled peak proof."""
    result = {"phase": label, "available": False, "RSS_bytes": None,
              "peak_RSS_bytes": None, "MemAvailable_bytes": None,
              "scope": "instantaneous process RSS and host MemAvailable; process HWM when provided by Linux"}
    try:
        for path, names in ((proc_root / "self" / "status", {"VmRSS": "RSS_bytes", "VmHWM": "peak_RSS_bytes"}),
                            (proc_root / "meminfo", {"MemAvailable": "MemAvailable_bytes"})):
            for line in path.read_text().splitlines():
                key, _, payload = line.partition(":")
                if key in names:
                    parts = payload.split()
                    if len(parts) != 2 or parts[1] != "kB":
                        raise ValueError("unexpected Linux memory units")
                    result[names[key]] = int(parts[0]) * 1024
        result["available"] = all(result[key] is not None for key in ("RSS_bytes", "peak_RSS_bytes", "MemAvailable_bytes"))
    except (OSError, ValueError) as exc:
        result["unavailable_reason"] = str(exc)
    return result


def process_cpu_sample(before, after, wall, step_budget, parent_budget, *, mode):
    """OS process CPU time includes its threads; a pool controller excludes workers."""
    user, system = after[0] - before[0], after[1] - before[1]
    if (not math.isfinite(wall) or wall <= 0 or step_budget <= 0 or parent_budget < step_budget
            or not all(math.isfinite(value) and value >= 0 for value in (user, system))):
        raise RuntimeError("invalid process CPU timing observation")
    busy = (user + system) / wall
    return {"wall_seconds": wall, "process_CPU_user_seconds_delta": user,
            "process_CPU_system_seconds_delta": system,
            "process_CPU_user_system_seconds_delta": user + system,
            "busy_core_equivalents": busy, "step_CPU_budget": step_budget,
            "parent_CPU_budget": parent_budget, "step_utilization_fraction": busy / step_budget,
            "parent_utilization_fraction": busy / parent_budget,
            "scope": "controller process only; excludes pool worker CPU time" if mode == "pool" else "single process including all backend threads",
            "claim": "observed CPU-time/wall ratio only; not proof of parallel efficiency or 48 effective threads"}


def measured_run(runner, size, resources, mode, *, wrapper=False):
    before = os.times()
    started = time.perf_counter()
    value = runner.run(size, wrapper=wrapper)
    wall = time.perf_counter() - started
    sample = process_cpu_sample(before, os.times(), wall,
        resources.get("step_CPU_budget", resources.get("allocated_CPU_request_count", 1)),
        resources.get("parent_allocated_CPU_request_count", resources.get("allocated_CPU_request_count", 1)), mode=mode)
    return value, sample


def stack_outputs(values, jax, np):
    return jax.tree_util.tree_map(lambda *parts: np.stack(parts), *values)


class SingleRunner:
    def __init__(self, args, recorder, lanes, second, scale):
        self.args, self.recorder, self.lanes, self.second, self.scale = args, recorder, lanes, second, scale
        self.programs, self.pairs, self.outputs = {}, {}, None
        if args.mode == "scalar":
            recorder.result.setdefault("runtime_observations", {})["scalar_pre_compile"] = memory_snapshot("scalar_pre_compile")
            self.programs[1] = scale.base.independent_scalar_executable(lanes[0], "server_scalar_once", recorder)
            self.pairs[1] = scale.jax.device_put([(lane.arguments, other) for lane, other in zip(lanes, second, strict=True)])
            scale.base.sync(self.pairs[1])
            recorder.result["thread_affinity_after_compile"] = thread_affinity(scale.base.affinity())
            recorder.result["runtime_observations"]["scalar_post_compile"] = memory_snapshot("scalar_post_compile")

    def prepare(self, size):
        if self.args.mode == "batch" and size not in self.programs:
            self.programs[size], self.pairs[size] = self.scale.compile_batch(
                self.lanes[:size], self.second[:size], f"server_batch_B{size}", self.recorder)
            self.recorder.result["thread_affinity_after_compile"] = thread_affinity(self.scale.base.affinity())

    def run(self, size, *, wrapper=False):
        base = self.scale.base
        if self.args.mode == "scalar":
            pairs = self.pairs[1][:size]
            if wrapper:
                pairs = [(lane.arguments, other) for lane, other in zip(self.lanes[:size], self.second[:size], strict=True)]
            self.outputs = [base.run_two(self.programs[1], *pair) for pair in pairs]
        else:
            pairs = self.pairs[size]
            if wrapper:
                pairs = base.batch.stack_lanes(self.lanes[:size]), base.stack_second(self.lanes[:size], self.second[:size])
            self.outputs = base.run_two(self.programs[size], *pairs)
        if wrapper:
            return self.collect(size)

    def collect(self, size):
        values = self.scale.jax.device_get(self.outputs)
        return stack_outputs(values, self.scale.jax, self.scale.np) if self.args.mode == "scalar" else values

    def records(self):
        return []

    def close(self):
        pass


def pool_worker(connection, cpu, args, assigned):
    """Spawn bootstrap has no JAX imports; one resident scalar compile per worker."""
    if args.synthetic_smoke:
        return synthetic_pool_worker(connection, cpu, assigned, args)
    recorder = None
    try:
        bind([cpu], one_thread=True)
        os.environ["ORCHJAX_GLOBAL_BATCH_PROCESS_START"] = str(time.perf_counter())
        from scripts.dev import benchmark_global_scale as scale
        from scripts.dev.probe_global_batch_cache import CacheEvents, configure_cache

        scale.jax.config.update("jax_enable_x64", True)
        policy = configure_cache(args.cache_mode, args.cache_root)
        recorder = scale.base.Recorder(args.output_dir / "workers" / f"cpu{cpu}")
        recorder.cache_observer = CacheEvents()
        lanes, second = scale.read_prepared(args.prepared_input, args.config)
        scale.screen_inputs(lanes, second)
        executable = scale.base.independent_scalar_executable(lanes[assigned[0]], "pool_scalar_once", recorder)
        pairs = scale.jax.device_put({index: (lanes[index].arguments, second[index]) for index in assigned})
        scale.base.sync(pairs)
        metadata = {"pid": os.getpid(), "cpu": cpu, "assigned_point_indices": assigned,
                    "scalar_compile_count": 1, "environment": scale.base.metadata(args), "cache_policy": policy,
                    "thread_affinity_after_compile": thread_affinity([cpu])}
        connection.send({"ready": metadata})
        outputs = {}
        while True:
            message = connection.recv()
            command = message["command"]
            if command == "close":
                connection.send({"closed": True})
                break
            if command in {"run", "wrapper"}:
                chosen = [index for index in assigned if index < message["size"]]
                active_pairs = pairs
                if command == "wrapper":
                    active_pairs = scale.jax.device_put({index: (lanes[index].arguments, second[index]) for index in chosen})
                    scale.base.sync(active_pairs)
                outputs = {index: scale.base.run_two(executable, *active_pairs[index]) for index in chosen}
                # run_two synchronizes every full leaf/effect; retain complete outputs resident.
                if command == "run":
                    connection.send({"completed_point_indices": chosen})
                else:
                    connection.send({"outputs": scale.jax.device_get(outputs)})
            elif command == "outputs":
                connection.send({"outputs": scale.jax.device_get(outputs)})
            elif command == "metadata":
                connection.send({"metadata": {**metadata, "process_CPU_user_system_seconds": sum(os.times()[:2]),
                    "thread_affinity_after_warm": thread_affinity([cpu]),
                    "peak_RSS_bytes": scale.base.linux_ru_maxrss_bytes(), "timing": recorder.result["timing"],
                    "compilations": recorder.result.get("compilations", {})}})
            else:
                raise ValueError("unknown worker command")
    except BaseException as exc:
        connection.send({"error": str(exc), "traceback": traceback.format_exc()})
    finally:
        if recorder is not None:
            recorder.stop.set()
            recorder.flush()
            recorder.cache_observer.close()
        connection.close()


class PoolRunner:
    def __init__(self, args, recorder, lanes, second, scale, cpus):
        self.args, self.scale, self.channels, self.processes, self.outputs = args, scale, [], [], None
        context = multiprocessing.get_context("spawn")
        try:
            for position, cpu in enumerate(cpus):
                parent, child = context.Pipe()
                assigned = list(range(position, len(lanes), len(cpus)))
                process = context.Process(target=pool_worker, args=(child, cpu, args, assigned))
                process.start()
                child.close()
                self.channels.append(parent)
                self.processes.append(process)
            recorder.result["pool_workers"] = self.receive("ready")
            recorder.phase("persistent_pool_ready", worker_count=len(cpus), compile_count_per_worker=1)
        except BaseException:
            self.close()
            raise

    def receive(self, key):
        result = []
        for channel in self.channels:
            message = channel.recv()
            if "error" in message:
                raise RuntimeError(message["error"] + "\n" + message.get("traceback", ""))
            result.append(message[key])
        return result

    def prepare(self, size):
        pass

    def run(self, size, *, wrapper=False):
        for channel in self.channels:
            channel.send({"command": "wrapper" if wrapper else "run", "size": size})
        if wrapper:
            self.outputs = self.receive("outputs")
            return self.assemble(size)
        completed = self.receive("completed_point_indices")
        if sorted(index for group in completed for index in group) != list(range(size)):
            raise ValueError("persistent pool did not synchronize every unique point")

    def assemble(self, size):
        if sum(len(group) for group in self.outputs) != size:
            raise ValueError("pool output materialization duplicated or omitted a point")
        values = {index: value for group in self.outputs for index, value in group.items()}
        if sorted(values) != list(range(size)):
            raise ValueError("pool full output materialization omitted a point")
        return stack_outputs([values[index] for index in range(size)], self.scale.jax, self.scale.np)

    def collect(self, size):
        for channel in self.channels:
            channel.send({"command": "outputs"})
        self.outputs = self.receive("outputs")
        return self.assemble(size)

    def records(self):
        for channel in self.channels:
            channel.send({"command": "metadata"})
        return self.receive("metadata")

    def close(self):
        for channel, process in zip(self.channels, self.processes):
            if process.is_alive():
                try:
                    channel.send({"command": "close"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        for channel in self.channels:
            channel.close()


def prepare_references(args, recorder, lanes, second, scale, resources):
    """Shared production input/identity/reference admission, before runner creation.

    This reads saved arrays and screens inputs. It never creates, lowers,
    compiles or calls a physical transition.
    """
    from scripts.dev.global_server_reference import load_reference

    base = scale.base
    screen = scale.screen_inputs(lanes, second)
    if args.cpu_budget == 48 and len(lanes) != 64:
        raise ValueError("batch48 requires exactly64 native unique prepared lanes")
    base.legacy_factory(args.baseline_ref)
    recorder.sentinel_paths = base.roughheight_sentinel_contract(lanes[0])
    identity = base.stage_identity(args, lanes, second, recorder.result["environment"], include_git_head=True)
    validation_mode = getattr(args, "validation_mode", "strict")
    cpu_reference_role = getattr(args, "cpu_reference_role", "accepted")
    if validation_mode == "strict" and cpu_reference_role == "cpu-internal":
        raise ValueError("cpu-internal reference role requires exploratory validation")
    identity.update(contract="server-CPU-same-workload-v1", precision_policy=scale.PRECISION_POLICY,
                    mode=args.mode, CPU_budget=args.cpu_budget, validation_mode=validation_mode,
                    cpu_reference_role=cpu_reference_role)
    if "parent_allocated_CPU_request_count" in resources:
        identity.update(parent_allocated_CPU_request_count=resources["parent_allocated_CPU_request_count"],
                        billing_CPU_request_count=resources["billing_CPU_request_count"])
    reference, origin = load_reference(args.gpu_reference_dir, identity)
    expected = getattr(args, "expected_gpu_reference_sha256", None)
    if expected is not None and origin["sha256"] != expected:
        raise ValueError("sealed GPU reference SHA256 differs from explicitly expected artifact")
    identity["numerical_reference_sha256"] = origin["sha256"]
    recorder.result["reference_preparation_gpu_loaded"] = True
    cpu_references = {}
    if args.cpu_reference_dir is not None:
        candidate_provenance = None
        admission_record = getattr(args, "cpu_reference_admission_record", None)
        if admission_record is not None:
            from scripts.dev.global_cpu_candidate_identity import candidate_identity
            candidate_provenance = resources.get("candidate_provenance") or candidate_identity(args, ROOT)
            identity["candidate_provenance"] = candidate_provenance
            recorder.result["reference_preparation_identity"] = {
                "baseline_ref": {"candidate": candidate_provenance.get("baseline_ref"),
                                 "stage": identity.get("baseline_ref")},
                "science_sha256": {"candidate": candidate_provenance.get("certified_science_sha256"),
                                   "stage": identity.get("science_sha256")},
                "gpu_reference_sha256": {"candidate": candidate_provenance.get("gpu_reference_sha256"),
                                         "stage": identity.get("numerical_reference_sha256")},
                "execution_commit": {"candidate": candidate_provenance.get("execution_commit"),
                                     "stage": identity.get("runtime", {}).get("git_head")}}
        for size in args.sizes:
            admission_args = {} if admission_record is None else {
                "cpu_reference_admission_record": admission_record,
                "expected_cpu_reference_admission_sha256": args.expected_cpu_reference_admission_sha256,
                "candidate_provenance": candidate_provenance}
            cpu_references[size] = load_reference(args.cpu_reference_dir, identity,
                name=f"B{size}_server_scalar_v1", points=size, cpu_reference_role=cpu_reference_role,
                **admission_args)
    recorder.result["reference_preparation_cpu_loaded"] = sorted(cpu_references)
    return identity, screen, reference, origin, cpu_references, validation_mode, cpu_reference_role


def benchmark(args, recorder, lanes, second, scale, resources):
    from scripts.dev.global_batch_stage_store import StageStore
    from scripts.dev.global_server_reference import finite_tolerance_failure
    from scripts.dev.probe_global_batch_cache import atomic_bytes

    base, jax, np = scale.base, scale.jax, scale.np
    identity, screen, reference, origin, cpu_references, validation_mode, cpu_reference_role = prepare_references(
        args, recorder, lanes, second, scale, resources)
    store = StageStore(args.output_dir, identity)
    recorder.result.update(stages=store.manifest["stages"], compatibility_audit=screen,
        validation_mode=validation_mode, cpu_reference_role=cpu_reference_role,
        GPU_numerical_reference_origin=origin, server_resources=resources,
        required_stages=[f"B{size}_server_{args.mode}_v1" for size in args.sizes])
    runner = PoolRunner(args, recorder, lanes, second, scale, resources["selected_logical_cpus"]) if args.mode == "pool" else SingleRunner(args, recorder, lanes, second, scale)
    def gpu_gate(name, actual, expected):
        if validation_mode == "strict":
            recorder.gate(name, actual, expected)
            return recorder.result["gates"][-1]
        entry = {"name": name, **base.compare(actual, expected, sentinel_paths=recorder.sentinel_paths),
                 "gate_role": "GPU_cross_backend_parity"}
        recorder.result["gates"].append(entry)
        recorder.flush()
        return entry
    completed_records = []
    try:
        for size in args.sizes:
            def operation(size=size):
                gate_start = len(recorder.result["gates"])
                started = time.perf_counter()
                observations = {"pre_compile": memory_snapshot("pre_compile")}
                runner.prepare(size)
                observations["post_compile"] = memory_snapshot("post_compile")
                observations["thread_affinity_after_compile"] = recorder.result.get("thread_affinity_after_compile")
                _, first_sample = measured_run(runner, size, resources, args.mode)
                recorder.result["timing"][f"B{size}_first_call_seconds"] = first_sample["wall_seconds"]
                observations["first_call_process_CPU"] = first_sample
                observations["first_call_complete"] = memory_snapshot("first_call_complete")
                output = runner.collect(size)
                if validation_mode == "exploratory":
                    raw_path = args.output_dir / f"raw_B{size}_{args.mode}_before_parity.pkl"
                    atomic_bytes(raw_path, pickle.dumps({"identity": identity,
                        "status": "raw_before_validation", "evidence_class": "raw_output_only",
                        "outputs": output, "measurement_status": "not_measured"}, protocol=pickle.HIGHEST_PROTOCOL))
                    recorder.phase(f"B{size}_{args.mode}_raw_output_saved_before_parity", path=str(raw_path))
                expected = jax.tree_util.tree_map(lambda value: value[:size], reference)
                gpu_gates = []
                for index, lane in enumerate(lanes[:size]):
                    gpu_gates.append(gpu_gate(f"B{size}_{args.mode}_lane_{index}_{lane.landpoint_id}_GPU_reference",
                        jax.tree_util.tree_map(lambda value: value[index], output),
                        jax.tree_util.tree_map(lambda value: value[index], expected)))
                if validation_mode == "exploratory":
                    hard = [gate for gate in gpu_gates if not gate["passed"] and not finite_tolerance_failure(gate)]
                    if hard:
                        raise AssertionError(f"hard gate failed: {hard[0]['name']}; all per-point GPU gates recorded; see report.json")
                    for gate in gpu_gates:
                        if not gate["passed"]:
                            gate["exploratory_continuation"] = True
                    recorder.flush()
                if args.mode == "batch":
                    isolation_reference = expected if validation_mode == "strict" else output
                    order = list(reversed(range(size)))
                    shuffled = [lanes[index] for index in order]
                    permuted = base.run_two(runner.programs[size], base.batch.stack_lanes(shuffled),
                        base.stack_second(shuffled, [second[index] for index in order]))
                    changed = lanes[:size - 1] + [lanes[0]]
                    replaced = base.run_two(runner.programs[size], base.batch.stack_lanes(changed),
                        base.stack_second(changed, second[:size - 1] + [second[0]]))
                    record_candidate_batch_isolation(recorder, size, isolation_reference,
                        permuted, replaced, jax=jax, np=np)
                else:
                    order = permuted = replaced = None
                if args.cpu_reference_dir is not None:
                    scalar, cpu_origin = cpu_references[size]
                    recorder.gate(f"B{size}_{args.mode}_cross_mode_server_scalar", output, scalar)
                else:
                    cpu_origin = None
                core, wrapper, core_CPU, wrapper_CPU = [], [], [], []
                for _ in range(args.warm_repeats):
                    _, sample = measured_run(runner, size, resources, args.mode)
                    core.append(sample["wall_seconds"])
                    core_CPU.append(sample)
                    materialized, sample = measured_run(runner, size, resources, args.mode, wrapper=True)
                    wrapper.append(sample["wall_seconds"])
                    wrapper_CPU.append(sample)
                observations["warm_complete"] = memory_snapshot("warm_complete")
                if args.mode == "batch":
                    record_candidate_batch_reset(recorder, size, output, materialized)
                    validation_arrays = (seal_candidate_batch_arrays(args.output_dir, identity, size,
                        permuted, replaced, materialized, jax=jax, np=np)
                        if getattr(args, "cpu_reference_admission_record", None) is not None else None)
                else:
                    recorder.gate(f"B{size}_{args.mode}_frozen_reset_repeat", materialized, output)
                    validation_arrays = None
                metrics = {f"B{size}_warm_frozen_state_core": base.metrics(core, size * 14),
                           f"B{size}_preprepared_wrapper_stack_transfer_materialize": base.metrics(wrapper, size * 14)}
                metrics[f"B{size}_warm_frozen_state_core"]["process_CPU_samples"] = core_CPU
                metrics[f"B{size}_preprepared_wrapper_stack_transfer_materialize"]["process_CPU_samples"] = wrapper_CPU
                recorder.result["timing"].update(metrics)
                workers = runner.records()
                controller_threads = thread_affinity(base.affinity())
                observations["thread_affinity_after_warm"] = controller_threads
                recorder.result.setdefault("runtime_observations", {})[f"B{size}"] = observations
                stage_gates = recorder.result["gates"][gate_start:]
                return {"outputs": output, "measurements": metrics}, {
                    "measurement_status": "complete",
                    "evidence_class": "exploratory_performance" if validation_mode == "exploratory" else "accepted",
                    "strict_numerical_passed": all(gate["passed"] is True for gate in stage_gates),
                    "provenance": {"point_ids": [lane.landpoint_id for lane in lanes[:size]],
                        "unique_real_points": size, "full_two_block_workload": True,
                        "full_two_block_parity": all(gate["passed"] is True for gate in gpu_gates), "full_outputs": True,
                        "frozen_reset_repeat": True, "cross_mode_scalar_parity": cpu_origin is not None,
                        "batch_permutation_and_replacement": args.mode == "batch",
                        "five_or_more_synchronized_repeats": True, "precision_policy": scale.PRECISION_POLICY,
                        "blocks": [[2, 8], [9, 15]], "half_hours_per_day": 48,
                        "server_resources": resources, "GPU_numerical_reference": origin,
                        "runtime_observations": observations,
                        "CPU_scalar_reference": cpu_origin, "worker_records": workers,
                        "CPU_scalar_reference_role": "cpu_internal_comparison" if cpu_origin is not None and cpu_reference_role == "cpu-internal" else "accepted" if cpu_origin is not None else None,
                        "controller_thread_affinity_after_warm": controller_threads,
                         "validation_arrays": validation_arrays,
                        "core_boundary": "actual group wall includes dispatch/barrier; all workers synchronize and retain every output; excludes materialization/serialization/IPC of full outputs",
                        "wrapper_boundary": "fresh preprepared host inputs/device transfer plus complete computation, full host output materialization and pool serialization/IPC",
                        "scalar_compile_policy": "once per persistent scalar worker, reused across distinct points and sizes"},
                    "stage_seconds": time.perf_counter() - started, "gates": recorder.result["gates"][gate_start:],
                    "timing": metrics, "execution_origin": {"available": True, "environment": recorder.result["environment"],
                        "pid": os.getpid(), "process_start_monotonic": base.PROCESS_STARTED,
                        "report": str(args.output_dir / "report.json")}}
            (value, record), reused = store.run(f"B{size}_server_{args.mode}_v1", operation)
            completed_records.append(record)
            base.record_timing_origin(recorder.result, f"B{size}_server_{args.mode}_v1", record, reused=reused)
            recorder.phase(f"B{size}_{args.mode}_checkpoint_complete", reused=reused)
        recorder.result["all_required_stages_complete"] = True
        recorder.result["measurement_status"] = "complete"
        recorder.result["evidence_class"] = "exploratory_performance" if validation_mode == "exploratory" else "accepted"
        recorder.result["strict_numerical_passed"] = all(record["strict_numerical_passed"] for record in completed_records)
        recorder.result["worker_resources"] = runner.records()
    finally:
        runner.close()


def synthetic_smoke(args):
    # Explicit non-model protocol/affinity smoke: optional toy JAX, no model/data calls.
    values = {size: sum(index * 14 for index in range(size)) for size in args.sizes}
    worker_records = []
    if args.mode == "pool":
        allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(args.cpu_budget))
        if hasattr(os, "sched_getaffinity"):
            topology = physical_cores(allowed)
            cpus, seen = [], set()
            for record in topology:
                key = record["package_id"], record["physical_core_id"]
                if key not in seen:
                    cpus.append(record["logical_cpu"])
                    seen.add(key)
            cpus = cpus[:args.cpu_budget]
            if len(cpus) != args.cpu_budget:
                raise RuntimeError("synthetic pool requires four available unique physical cores")
        else:
            cpus = allowed[:args.cpu_budget]
        from types import SimpleNamespace

        tree = SimpleNamespace(tree_map=lambda function, *parts: function(*parts))
        toy_scale = SimpleNamespace(jax=SimpleNamespace(tree_util=tree), np=SimpleNamespace(stack=list))
        recorder = SimpleNamespace(result={}, phase=lambda *positional, **keywords: None)
        runner = PoolRunner(args, recorder, list(range(max(args.sizes))), None, toy_scale, cpus)
        try:
            for size in args.sizes:
                for _ in range(args.warm_repeats):
                    runner.run(size)
                result = runner.collect(size)
                if result != [index * 14 for index in range(size)]:
                    raise AssertionError("synthetic complete point ordering/retention mismatch")
            worker_records = runner.records()
        finally:
            runner.close()
    return {"status": "synthetic_nonmodel_smoke", "mode": args.mode, "CPU_budget": args.cpu_budget,
            "model_execution": False, "JAX_imported": "jax" in sys.modules, "toy_checksums": values,
            "worker_records": worker_records, "performance_acceptance": False}


def synthetic_pool_worker(connection, cpu, assigned, args):
    try:
        if hasattr(os, "sched_setaffinity"):
            bind([cpu], one_thread=True)
            affinity = sorted(os.sched_getaffinity(0))
        else:
            affinity = None
        executable, runtime, threads = None, None, None
        if not args.synthetic_stdlib_only:
            os.environ["JAX_PLATFORMS"] = "cpu"
            os.environ["JAX_ENABLE_X64"] = "true"
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ[name] = "1"
            import jax
            import jax.numpy as jnp

            executable = jax.jit(lambda value: value * 14).lower(jnp.asarray(0.0, dtype=jnp.float64)).compile()
            runtime = {"jax": jax.__version__, "devices": [str(device) for device in jax.devices()]}
            threads = thread_affinity([cpu]) if affinity is not None else None
        metadata = {"pid": os.getpid(), "cpu": cpu, "affinity": affinity,
                    "JAX_imported": "jax" in sys.modules, "assigned_point_indices": assigned,
                    "runtime": runtime, "toy_compile_count": 1 if executable is not None else 0,
                    "thread_affinity_after_compile": threads,
                    "scope": "non-model toy only; affinity verified only when Linux sched API available"}
        connection.send({"ready": metadata})
        outputs = {}
        while True:
            message = connection.recv()
            if message["command"] == "close":
                break
            if message["command"] in {"run", "wrapper"}:
                chosen = [index for index in assigned if index < message["size"]]
                outputs = {index: index * 14 for index in chosen}
                if executable is not None:
                    outputs = {index: float(executable(jnp.asarray(float(index), dtype=jnp.float64)).block_until_ready())
                               for index in chosen}
                connection.send({"completed_point_indices": chosen} if message["command"] == "run" else {"outputs": outputs})
            elif message["command"] == "outputs":
                connection.send({"outputs": outputs})
            elif message["command"] == "metadata":
                connection.send({"metadata": {**metadata,
                    "thread_affinity_after_warm": thread_affinity([cpu]) if affinity is not None else None}})
    finally:
        connection.close()


def main():
    args = parser().parse_args()
    validate(args)
    if args.dry_run or args.synthetic_smoke:
        if args.require_child_environment:
            validate_environment(args.cpu_budget)
        result = synthetic_smoke(args) if args.synthetic_smoke else {
            "status": "dry_run", "mode": args.mode, "CPU_budget": args.cpu_budget, "sizes": args.sizes,
            "warm_repeats": args.warm_repeats, "physical_performance_requires": "bound cnall/cnmix Slurm allocation",
            "model_execution": False, "JAX_imported": "jax" in sys.modules,
            "CPU_process_policy": "one native CPU process/device; lane count is not CPU count" if args.mode == "batch" else args.mode,
            "parent_CPU_count_declared": os.environ.get("ORCHJAX_PARENT_CPU_COUNT"),
            "billing_scope": "full current parent allocation; actual Slurm admission required"}
        result["actual_child_prejax_receipt"] = child_snapshot(args.cpu_budget, status="nonmodel_cli_only")
        result.update(validation_mode=args.validation_mode, cpu_reference_role=args.cpu_reference_role,
                      expected_gpu_reference_sha256=args.expected_gpu_reference_sha256)
        print(json.dumps(result, indent=2))
        return 0
    try:
        if args.require_child_environment:
            validate_environment(args.cpu_budget)
        resources = admit(args.mode, args.cpu_budget)
        default_guard = 96 if args.mode == "pool" else 64 if args.cpu_budget == 48 else 24
        resources["host_memory_admission"] = memory_guard(args.memory_guard_gib or default_guard)
        cpus = resources["selected_logical_cpus"]
        bind(cpus[:1] if args.mode in {"scalar", "pool"} else cpus, one_thread=args.mode != "batch")
        if args.require_child_environment:
            validate_environment(args.cpu_budget)
        if args.cpu_reference_admission_record is not None:
            from scripts.dev.global_cpu_candidate_identity import candidate_identity
            from scripts.dev.global_server_reference import validate_cpu_reference_admission
            resources["candidate_provenance"] = candidate_identity(args, ROOT)
            resources["external_scalar_reference_admission"] = validate_cpu_reference_admission(
                args.cpu_reference_admission_record, args.expected_cpu_reference_admission_sha256,
                args.cpu_reference_dir, expected_source=resources["candidate_provenance"])
    except Exception as exc:
        print(json.dumps(child_snapshot(args.cpu_budget, status="resource_admission_failed", error=str(exc))),
              file=sys.stderr, flush=True)
        raise
    receipt = child_snapshot(args.cpu_budget, status="resource_admission_passed")
    receipt.update(selected_physical_cores=resources["selected_logical_cpus"],
                   allocation=resources["Slurm"], allocation_admitted=True)
    print(json.dumps(receipt), file=sys.stderr, flush=True)
    resources["actual_child_prejax_receipt"] = receipt
    resources["thread_environment_before_JAX"] = {name: os.environ.get(name) for name in (
        "XLA_FLAGS", "JAX_PLATFORMS", "JAX_ENABLE_X64", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    resources["execution_policy"] = {
        "native_CPU_device_policy": "single CPU device; no force48 logical devices",
        "batch48_BLAS_policy": "single-thread BLAS/NUMEXPR; OMP48 and native Eigen multithreading" if args.cpu_budget == 48 else None,
        "effective_48_thread_claim": False}
    os.environ["ORCHJAX_GLOBAL_BATCH_PROCESS_START"] = str(PROCESS_STARTED)
    os.environ.setdefault("ORCHIDEE_REPO_ROOT", str(ROOT))
    os.environ.setdefault("ORCHIDEE_DATA_ROOT", str(args.truth_root / "data"))
    os.environ.setdefault("ORCHIDEE_REFERENCE_ROOT", str(args.truth_root / "reference"))
    os.environ["ORCHIDEE_OUTPUT_ROOT"] = str(args.output_dir.resolve())
    from scripts.dev import benchmark_global_scale as scale
    from scripts.dev.global_batch_stage_store import StageStore
    from scripts.dev.probe_global_batch_cache import CacheEvents, configure_cache

    scale.jax.config.update("jax_enable_x64", True)
    args.output_dir = args.output_dir.resolve()
    for source in (args.prepared_input.parent, args.gpu_reference_dir, args.cpu_reference_dir):
        if source is not None:
            StageStore.validate_source_destination(source, args.output_dir)
    policy = configure_cache(args.cache_mode, args.cache_root)
    recorder = scale.base.Recorder(args.output_dir)
    recorder.cache_observer = CacheEvents()
    previous = {}
    def stopped(number, frame):
        raise InterruptedError(f"signal {number}; sealed CPU checkpoints retained")
    for number in (signal.SIGTERM, signal.SIGINT):
        previous[number] = signal.signal(number, stopped)
    started, code = time.perf_counter(), 0
    try:
        recorder.result["environment"] = scale.base.metadata(args)
        recorder.result["environment"]["explicit_cache_policy"] = policy
        runtime = recorder.result["environment"]
        if runtime["jax"] != "0.4.38" or runtime["jaxlib"] != "0.4.38" or runtime["gpu_count"]:
            raise RuntimeError("physical server CPU benchmark requires accepted CPU-only JAX/jaxlib0.4.38")
        if args.cpu_budget == 48 and (len(runtime.get("device_details", [])) != 1 or
                runtime["device_details"][0].get("platform") != "cpu"):
            raise RuntimeError("batch48 requires exactly one native CPU device")
        lanes, second = scale.read_prepared(args.prepared_input, args.config)
        benchmark(args, recorder, lanes, second, scale, resources)
        recorder.result["status"] = ("completed_exploratory_server_CPU_bounded_forward"
            if args.validation_mode == "exploratory" else "passed_server_CPU_bounded_forward")
    except (InterruptedError, KeyboardInterrupt) as exc:
        code = 2
        recorder.result.update(status="stopped", message=str(exc))
    except Exception as exc:
        code = 1
        recorder.result.update(status="failed", message=str(exc), traceback=traceback.format_exc())
    finally:
        recorder.result.update(process_CPU_user_system_seconds=sum(os.times()[:2]),
            benchmark_wall_seconds=time.perf_counter() - started,
            allocated_CPU_request_core_hours=args.cpu_budget * (time.perf_counter() - started) / 3600,
            allocated_parent_CPU_request_core_hours=resources["billing_CPU_request_count"] * (time.perf_counter() - started) / 3600,
            CPU_core_hours_scope="allocated_CPU_request_core_hours is step-only; allocated_parent_CPU_request_core_hours uses all billable parent CPUs; external full-job accounting remains authoritative",
            process_wall_seconds_including_import_setup=time.perf_counter() - PROCESS_STARTED,
            process_wall_scope="script entry through final report preparation; excludes Slurm startup/final filesystem flush; external job accounting required for actual full-job cost",
            server_resources=resources)
        recorder.stop.set()
        recorder.flush()
        recorder.cache_observer.close()
        for number, handler in previous.items():
            signal.signal(number, handler)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
