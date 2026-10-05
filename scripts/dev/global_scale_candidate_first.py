"""Candidate-first GPU measurements using sealed FULL64 numerical references.

No physics implementation lives here. The unchanged scalar transition factory
retains its audited Fortran provenance in the driver; this module changes only
measurement order and checkpoint boundaries.
"""
from __future__ import annotations

import os
import time

import jax
import numpy as np

from scripts.dev import benchmark_global_scale as scale
from scripts.dev.global_batch_stage_store import StageStore
from scripts.dev.global_server_reference import load_reference
from scripts.dev.global_server_resources import thread_affinity


def candidate_first_benchmark(args, recorder, lanes, second):
    base = scale.base
    screen = scale.screen_inputs(lanes, second)
    if max(args.sizes) > len(lanes):
        raise ValueError("requested candidate exceeds complete prepared lanes")
    base.legacy_factory(args.baseline_ref)
    recorder.sentinel_paths = base.roughheight_sentinel_contract(lanes[0])
    identity = base.stage_identity(args, lanes, second, recorder.result["environment"])
    identity.update(contract="server-candidate-first-full-physics-v1", precision_policy=scale.PRECISION_POLICY,
                    measurement_order="candidate_checkpoint_before_optional_B4_compile")
    reference, origin = load_reference(args.full_reference_dir, identity)
    # New namespace: neither read nor write old v1 scale stages as new measurements.
    StageStore.validate_source_destination(args.full_reference_dir, args.output_dir)
    identity["numerical_reference_sha256"] = origin["sha256"]
    store = StageStore(args.output_dir, identity)
    recorder.result.update(compatibility_audit=screen, reference_origin=origin, stages=store.manifest["stages"],
                           requested_sizes=list(args.sizes), execution_sizes=list(args.sizes),
                           execution_order="candidate-first", optional_B4_baseline_requested=args.measure_b4_baseline)
    candidates = [f"B{size}_candidate_first_v1" for size in args.sizes]
    recorder.result["required_stages"] = candidates

    def run_stage(name, operation):
        def execute():
            gate_start = len(recorder.result["gates"])
            old_timing = dict(recorder.result["timing"])
            old_compilation = dict(recorder.result.get("compilations", {}))
            started = time.perf_counter()
            value, proof = operation()
            return value, {
                "provenance": proof, "stage_seconds": time.perf_counter() - started,
                "execution_origin": {"available": True, "environment": recorder.result["environment"],
                    "pid": os.getpid(), "process_start_monotonic": base.PROCESS_STARTED,
                    "window_identity": os.environ.get("ORCHJAX_GLOBAL_BATCH_WINDOW_ID"),
                    "report": str(recorder.window_report) if recorder.window_report else str(args.output_dir / "report.json")},
                "gates": recorder.result["gates"][gate_start:],
                "timing": {key: value for key, value in recorder.result["timing"].items() if old_timing.get(key) != value},
                "compilations": {key: value for key, value in recorder.result.get("compilations", {}).items()
                                 if old_compilation.get(key) != value}}
        (value, record), reused = store.run(name, execute)
        base.record_timing_origin(recorder.result, name, record, reused=reused)
        recorder.phase(name + ("_reused" if reused else "_complete"))
        return value

    def candidate(size):
        chosen, others = lanes[:size], second[:size]
        expected = jax.tree_util.tree_map(lambda value: value[:size], reference)
        host_inputs = base.batch.stack_lanes(chosen), base.stack_second(chosen, others)
        preflight = scale.memory_preflight(size, host_inputs,
            jax.tree_util.tree_map(lambda value: value[:4], reference), {}, recorder=recorder)
        recorder.phase(f"B{size}_candidate_memory_preflight", preflight=preflight)
        executable, inputs = scale.compile_batch(chosen, others, f"B{size}_candidate", recorder)
        analysis = scale.executable_memory(executable)
        threads_after_compile = thread_affinity(recorder.result["server_resources"]["original_affinity"])
        # Refine memory admission with this actual candidate's compiler estimate.
        postcompile = scale.memory_preflight(size, host_inputs,
            jax.tree_util.tree_map(lambda value: value[:4], reference), {size: analysis}, recorder=recorder)
        started = time.perf_counter()
        output = base.run_two(executable, *inputs)
        recorder.result["timing"][f"B{size}_first_call_seconds"] = time.perf_counter() - started
        for index, lane in enumerate(chosen):
            recorder.gate(f"B{size}_lane_{index}_{lane.landpoint_id}_both_blocks",
                          jax.tree_util.tree_map(lambda value: value[index], output),
                          jax.tree_util.tree_map(lambda value: value[index], expected))
        order = list(reversed(range(size)))
        shuffled = [chosen[index] for index in order]
        permuted = base.run_two(executable, base.batch.stack_lanes(shuffled),
                              base.stack_second(shuffled, [others[index] for index in order]))
        recorder.gate(f"B{size}_cross_group_permutation", permuted,
                      jax.tree_util.tree_map(lambda value: value[np.asarray(order)], expected))
        changed = chosen[:-1] + [chosen[0]]
        replaced = base.run_two(executable, base.batch.stack_lanes(changed),
                               base.stack_second(changed, others[:-1] + [others[0]]))
        recorder.gate(f"B{size}_replacement_all_lanes_isolation", replaced,
                      jax.tree_util.tree_map(lambda value: value[np.asarray(list(range(size - 1)) + [0])], expected))
        core, wrapper = [], []
        for _ in range(args.warm_repeats):
            started = time.perf_counter()
            base.run_two(executable, *inputs)
            core.append(time.perf_counter() - started)
            started = time.perf_counter()
            wrapped = base.run_two(executable, base.batch.stack_lanes(chosen), base.stack_second(chosen, others))
            jax.device_get(wrapped)
            wrapper.append(time.perf_counter() - started)
        measured = {f"B{size}_warm_frozen_state_core": base.metrics(core, size * 14),
                    f"B{size}_preprepared_wrapper_stack_transfer_materialize": base.metrics(wrapper, size * 14)}
        recorder.result["timing"].update(measured)
        return {"outputs": jax.device_get(output), "measurements": measured}, {
            "point_ids": [lane.landpoint_id for lane in chosen], "unique_real_points": size,
            "full_two_block_parity": True, "cross_group_permutation": True, "replacement_isolation": True,
            "five_or_more_synchronized_repeats": True, "precision_policy": scale.PRECISION_POLICY,
            "blocks": [[2, 8], [9, 15]], "half_hours_per_day": 48, "full_outputs": True,
            "memory_preflight": preflight, "postcompile_memory_preflight": postcompile,
            "executable_memory_analysis": analysis, "reference_origin": origin,
            "server_resources": recorder.result["server_resources"],
            "thread_affinity_after_compile": threads_after_compile,
            "thread_affinity_after_warm": thread_affinity(recorder.result["server_resources"]["original_affinity"]),
            "measurement_boundary": "candidate-only warm measurement; no reused B4 timing or scalar extrapolation"}

    # Each accepted candidate is sealed immediately; no B4 executable exists yet.
    for size in args.sizes:
        run_stage(f"B{size}_candidate_first_v1", lambda size=size: candidate(size))
    recorder.result["all_required_stages_complete"] = all(
        store.manifest["stages"].get(name, {}).get("status") == "complete" for name in candidates)
    recorder.flush()
    if args.measure_b4_baseline:
        def optional_baseline():
            executable, _ = scale.compile_batch(lanes[:4], second[:4], "B4_resident_after_candidates", recorder)
            measured = {}
            for size in args.sizes:
                expected = jax.tree_util.tree_map(lambda value: value[:size], reference)
                pairs = jax.device_put([(base.batch.stack_lanes(lanes[start:start + 4]),
                                        base.stack_second(lanes[start:start + 4], second[start:start + 4]))
                                       for start in range(0, size, 4)])
                base.sync(pairs)
                for index, pair in enumerate(pairs):
                    recorder.gate(f"B{size}_optional_actual_B4_group_{index}", base.run_two(executable, *pair),
                                  jax.tree_util.tree_map(lambda value: value[index * 4:(index + 1) * 4], expected))
                samples = []
                for _ in range(args.warm_repeats):
                    started = time.perf_counter()
                    for pair in pairs:
                        base.run_two(executable, *pair)
                    samples.append(time.perf_counter() - started)
                measured[f"B{size}_new_actual_B4_group_serial"] = base.metrics(samples, size * 14)
            recorder.result["timing"].update(measured)
            return {"measurements": measured}, {"point_ids": [lane.landpoint_id for lane in lanes],
                "own_inputs": True, "role": "optional newly measured baseline after sealed candidate checkpoints"}
        run_stage("optional_new_B4_comparison_v1", optional_baseline)
