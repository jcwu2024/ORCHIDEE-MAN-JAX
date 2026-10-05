"""Allocated server CPU scalar, persistent pool and batch full-physics benchmark.

The science is exclusively the existing audited transition. Physical execution
requires paid Slurm admission; --dry-run and --synthetic-smoke are non-model.
Resource affinity and thread environments are set before importing JAX.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESS_STARTED = time.perf_counter()
sys.path.insert(0, str(ROOT))

from scripts.dev.global_server_resources import admit, bind, physical_cores, thread_affinity


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=["scalar", "pool", "batch"], required=True)
    result.add_argument("--cpu-budget", type=int, choices=[1, 4], required=True)
    result.add_argument("--sizes", type=int, nargs="+", choices=[16, 64], default=[16, 64])
    result.add_argument("--prepared-input", type=Path)
    result.add_argument("--gpu-reference-dir", type=Path)
    result.add_argument("--cpu-reference-dir", type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--config", type=Path, default=ROOT / "configs/orchidee_man_250919.yaml")
    result.add_argument("--truth-root", type=Path, default=ROOT)
    result.add_argument("--baseline-ref", default="133e35aec7b83b9d79888c46092f64d518e97299")
    result.add_argument("--warm-repeats", type=int, default=5)
    result.add_argument("--cache-mode", choices=["off", "persistent"], default="off")
    result.add_argument("--cache-root", type=Path)
    result.add_argument("--memory-guard-gib", type=float, help="Conservative host MemAvailable guard; pool default60GiB.")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--synthetic-smoke", action="store_true")
    result.add_argument("--synthetic-stdlib-only", action="store_true", help="Pool protocol smoke without even a toy JAX compile.")
    return result


def validate(args):
    if args.mode == "scalar" and args.cpu_budget != 1 or args.mode == "pool" and args.cpu_budget != 4:
        raise ValueError("scalar requires budget1; pool requires budget4")
    if args.warm_repeats < 5 or args.sizes != sorted(set(args.sizes)):
        raise ValueError("at least five repeats and ascending unique16/64 sizes required")
    if not args.synthetic_smoke and (args.prepared_input is None or args.gpu_reference_dir is None):
        raise ValueError("physical execution requires frozen prepared input and sealed GPU numerical reference")
    if not args.synthetic_smoke and args.mode != "scalar" and args.cpu_reference_dir is None:
        raise ValueError("pool/batch acceptance requires --cpu-reference-dir with complete server scalar stages")
    if args.memory_guard_gib is not None and args.memory_guard_gib <= 0:
        raise ValueError("positive host memory guard required")


def memory_guard(gib, path=Path("/proc/meminfo")):
    available = next(int(line.split()[1]) * 1024 for line in path.read_text().splitlines() if line.startswith("MemAvailable:"))
    result = {"MemAvailable_bytes": available, "conservative_required_bytes": int(gib * 1024**3),
              "scope": "host RSS estimate, not a Slurm whole-node memory reservation or OOM guarantee"}
    if available < result["conservative_required_bytes"]:
        raise RuntimeError(f"insufficient host MemAvailable before compilation: {result}")
    return result


def stack_outputs(values, jax, np):
    return jax.tree_util.tree_map(lambda *parts: np.stack(parts), *values)


class SingleRunner:
    def __init__(self, args, recorder, lanes, second, scale):
        self.args, self.recorder, self.lanes, self.second, self.scale = args, recorder, lanes, second, scale
        self.programs, self.pairs, self.outputs = {}, {}, None
        if args.mode == "scalar":
            self.programs[1] = scale.base.independent_scalar_executable(lanes[0], "server_scalar_once", recorder)
            self.pairs[1] = scale.jax.device_put([(lane.arguments, other) for lane, other in zip(lanes, second, strict=True)])
            scale.base.sync(self.pairs[1])
            recorder.result["thread_affinity_after_compile"] = thread_affinity(scale.base.affinity())

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


def benchmark(args, recorder, lanes, second, scale, resources):
    from scripts.dev.global_batch_stage_store import StageStore
    from scripts.dev.global_server_reference import load_reference

    base, jax, np = scale.base, scale.jax, scale.np
    screen = scale.screen_inputs(lanes, second)
    base.legacy_factory(args.baseline_ref)
    recorder.sentinel_paths = base.roughheight_sentinel_contract(lanes[0])
    identity = base.stage_identity(args, lanes, second, recorder.result["environment"])
    identity.update(contract="server-CPU-same-workload-v1", precision_policy=scale.PRECISION_POLICY,
                    mode=args.mode, CPU_budget=args.cpu_budget)
    reference, origin = load_reference(args.gpu_reference_dir, identity)
    identity["numerical_reference_sha256"] = origin["sha256"]
    store = StageStore(args.output_dir, identity)
    recorder.result.update(stages=store.manifest["stages"], compatibility_audit=screen,
        GPU_numerical_reference_origin=origin, server_resources=resources,
        required_stages=[f"B{size}_server_{args.mode}_v1" for size in args.sizes])
    runner = PoolRunner(args, recorder, lanes, second, scale, resources["selected_logical_cpus"]) if args.mode == "pool" else SingleRunner(args, recorder, lanes, second, scale)
    try:
        for size in args.sizes:
            def operation(size=size):
                gate_start = len(recorder.result["gates"])
                started = time.perf_counter()
                runner.prepare(size)
                first_started = time.perf_counter()
                runner.run(size)
                recorder.result["timing"][f"B{size}_first_call_seconds"] = time.perf_counter() - first_started
                output = runner.collect(size)
                expected = jax.tree_util.tree_map(lambda value: value[:size], reference)
                for index, lane in enumerate(lanes[:size]):
                    recorder.gate(f"B{size}_{args.mode}_lane_{index}_{lane.landpoint_id}_GPU_reference",
                        jax.tree_util.tree_map(lambda value: value[index], output),
                        jax.tree_util.tree_map(lambda value: value[index], expected))
                if args.mode == "batch":
                    order = list(reversed(range(size)))
                    shuffled = [lanes[index] for index in order]
                    permuted = base.run_two(runner.programs[size], base.batch.stack_lanes(shuffled),
                        base.stack_second(shuffled, [second[index] for index in order]))
                    recorder.gate(f"B{size}_server_batch_cross_group_permutation", permuted,
                        jax.tree_util.tree_map(lambda value: value[np.asarray(order)], expected))
                    changed = lanes[:size - 1] + [lanes[0]]
                    replaced = base.run_two(runner.programs[size], base.batch.stack_lanes(changed),
                        base.stack_second(changed, second[:size - 1] + [second[0]]))
                    recorder.gate(f"B{size}_server_batch_replacement_isolation", replaced,
                        jax.tree_util.tree_map(lambda value: value[np.asarray(list(range(size - 1)) + [0])], expected))
                if args.cpu_reference_dir is not None:
                    scalar, cpu_origin = load_reference(args.cpu_reference_dir, identity,
                        name=f"B{size}_server_scalar_v1", points=size)
                    recorder.gate(f"B{size}_{args.mode}_cross_mode_server_scalar", output, scalar)
                else:
                    cpu_origin = None
                core, wrapper = [], []
                for _ in range(args.warm_repeats):
                    measured = time.perf_counter()
                    runner.run(size)
                    core.append(time.perf_counter() - measured)
                    measured = time.perf_counter()
                    materialized = runner.run(size, wrapper=True)
                    wrapper.append(time.perf_counter() - measured)
                recorder.gate(f"B{size}_{args.mode}_frozen_reset_repeat", materialized, output)
                metrics = {f"B{size}_warm_frozen_state_core": base.metrics(core, size * 14),
                           f"B{size}_preprepared_wrapper_stack_transfer_materialize": base.metrics(wrapper, size * 14)}
                recorder.result["timing"].update(metrics)
                workers = runner.records()
                controller_threads = thread_affinity(base.affinity())
                return {"outputs": output, "measurements": metrics}, {
                    "provenance": {"point_ids": [lane.landpoint_id for lane in lanes[:size]],
                        "unique_real_points": size, "full_two_block_parity": True, "full_outputs": True,
                        "frozen_reset_repeat": True, "cross_mode_scalar_parity": args.mode == "scalar" or cpu_origin is not None,
                        "batch_permutation_and_replacement": args.mode == "batch",
                        "five_or_more_synchronized_repeats": True, "precision_policy": scale.PRECISION_POLICY,
                        "blocks": [[2, 8], [9, 15]], "half_hours_per_day": 48,
                        "server_resources": resources, "GPU_numerical_reference": origin,
                        "CPU_scalar_reference": cpu_origin, "worker_records": workers,
                        "controller_thread_affinity_after_warm": controller_threads,
                        "core_boundary": "actual group wall includes dispatch/barrier; all workers synchronize and retain every output; excludes materialization/serialization/IPC of full outputs",
                        "wrapper_boundary": "fresh preprepared host inputs/device transfer plus complete computation, full host output materialization and pool serialization/IPC",
                        "scalar_compile_policy": "once per persistent scalar worker, reused across distinct points and sizes"},
                    "stage_seconds": time.perf_counter() - started, "gates": recorder.result["gates"][gate_start:],
                    "timing": metrics, "execution_origin": {"available": True, "environment": recorder.result["environment"],
                        "pid": os.getpid(), "process_start_monotonic": base.PROCESS_STARTED,
                        "report": str(args.output_dir / "report.json")}}
            (value, record), reused = store.run(f"B{size}_server_{args.mode}_v1", operation)
            base.record_timing_origin(recorder.result, f"B{size}_server_{args.mode}_v1", record, reused=reused)
            recorder.phase(f"B{size}_{args.mode}_checkpoint_complete", reused=reused)
        recorder.result["all_required_stages_complete"] = True
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
        result = synthetic_smoke(args) if args.synthetic_smoke else {
            "status": "dry_run", "mode": args.mode, "CPU_budget": args.cpu_budget, "sizes": args.sizes,
            "warm_repeats": args.warm_repeats, "physical_performance_requires": "bound cnall/cnmix Slurm allocation",
            "model_execution": False, "JAX_imported": "jax" in sys.modules}
        print(json.dumps(result, indent=2))
        return 0
    resources = admit(args.mode, args.cpu_budget)
    resources["host_memory_admission"] = memory_guard(args.memory_guard_gib or (60 if args.mode == "pool" else 20))
    cpus = resources["selected_logical_cpus"]
    bind(cpus[:1] if args.mode in {"scalar", "pool"} else cpus, one_thread=args.mode != "batch")
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
        lanes, second = scale.read_prepared(args.prepared_input, args.config)
        benchmark(args, recorder, lanes, second, scale, resources)
        recorder.result["status"] = "passed_server_CPU_bounded_forward"
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
