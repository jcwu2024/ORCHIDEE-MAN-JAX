"""Strict paired full derivative replay of existing recurrent GPU binaries.

No model lowering or compilation. Self comparisons diagnose execution noise;
initial and seven paired full derivative comparisons determine admission.
"""
from __future__ import annotations

import json
import hashlib
import pickle
from pathlib import Path
import time

import numpy as np

from probe_shared_normalized_keys_recurrent import load, make_parser, validate_inputs


def run(args):
    started = time.perf_counter()
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax, devices = initialize_gpu_backend()
    from jax.experimental.serialize_executable import deserialize_and_load
    from scripts.hpc.run_shared_learning import load_learning_checkpoint, memory_sizes
    jax.config.update("jax_enable_x64", True)
    owner = load(args.patch_owner, "scripts.hpc.probe_shared_expert_real_daily")
    gates = load(args.gate_owner, "chunk_batch_full_replay_gates")
    forward = load(args.forward_probe, "scripts.hpc.probe_shared_forward_reverse")
    if args.output.exists():
        raise FileExistsError("preserve prior replay result")
    report = {"status": "running", "scope": "saved_recurrent_binaries_seven_complete_derivative_pairs",
              "rtol": 1e-5, "atol": 1e-6, "horizon": args.horizon, "gpu_count": 1,
              "model_gradient_compilations": 0, "training_updates": 0,
              "production_adoption": False, "self_replays_are_diagnostic": True}
    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-started, **fields)
        owner.atomic_json(args.output, report)
        print(stage, flush=True)
    def checks(actual, expected):
        return {"loss_and_auxiliary": owner.compare_trees(jax, actual[0], expected[0]),
                "all_parameter_gradients": owner.compare_trees(jax, actual[1][0], expected[1][0]),
                "initial_state_gradient": owner.compare_trees(jax, actual[1][1], expected[1][1])}
    def parameter_leaf_checks(actual, expected):
        left, left_tree = jax.tree_util.tree_flatten_with_path(actual[1][0])
        right, right_tree = jax.tree_util.tree_flatten_with_path(expected[1][0])
        if left_tree != right_tree:
            raise ValueError("parameter leaf path mismatch")
        rows = []
        for (path, value), (other_path, other) in zip(left, right, strict=True):
            if path != other_path:
                raise ValueError("parameter leaf identity mismatch")
            rows.append({"path": jax.tree_util.keystr(path), "shape": list(np.shape(value)),
                         "dtype": str(np.asarray(value).dtype), **owner.compare_trees(jax, value, other)})
        if len(rows) != 300 or sum(row["coordinates"] for row in rows) != 64884481:
            raise ValueError("incomplete parameter leaf replay evidence")
        return rows
    def valid(value):
        result = gates.validate_result(jax, (value[0], value[1][0]), checkpoint["parameters"])
        state = np.asarray(value[1][1])
        if state.shape != np.shape(case[0]) or state.size != 3854 or not np.all(np.isfinite(state)) or not np.any(state):
            raise gates.NumericalFailure("invalid complete initial-state gradient")
        if result["gradient_coordinates"] != 64884481 or len(result["hard_counts"]) != 14:
            raise ValueError("wrong complete recurrent replay scope")
        return result | {"initial_state_gradient_coordinates": state.size,
                         "initial_state_gradient_nonzero": int(np.count_nonzero(state)),
                         "initial_state_gradient_finite": True}
    try:
        emit("starting")
        with gates.deadline(args.process_seconds, "whole_replay", emit):
            candidate = json.loads(args.candidate_report.read_text())
            original = json.loads(args.reference_report.read_text())
            variant = "expert5_whole_chunk_axis_vmap" if args.execution_variant == "whole-chunk-vmap" else "expert5_preserve128_learned_shapes"
            if (candidate.get("horizon") != args.horizon
                    or candidate.get("execution_variant") != variant
                    or candidate.get("precision") != "unchanged_FP32_neural_FP64_physical"
                    or original.get("status") != "passed"
                    or original.get("baseline_identity", original.get("identity")) != candidate.get("baseline_identity")):
                raise ValueError("saved candidate/reference scope or identity differs")
            report.update(identity=candidate["identity"], baseline_identity=candidate["baseline_identity"],
                          candidate_report=str(args.candidate_report), reference_report=str(args.reference_report))
            gates.require_parity(candidate["comparisons"])
            rows = candidate.get("initial_state_pair_replay", [])
            if len(rows) != 7 or any(not row["passed"] for row in rows):
                raise gates.NumericalFailure("existing paired state gate already failed")
            if len(devices) != 1 or jax.default_backend() != "gpu":
                raise ValueError("requires one actual GPU")
            report["runtime_preflight"] = gates.validate_runtime_report(args.runtime_preflight, jax.__version__)
            _profile, reference, bundle, _context, checkpoint_path, _admissions = forward._assets(args)
            identity = candidate["baseline_identity"]
            if (identity["source_head"] != reference["source_git_head"]
                    or identity["method"] != reference["method"]
                    or identity["artifact_id"] != int(bundle["artifact_ids"][0])
                    or identity["horizon"] != args.horizon
                    or identity["checkpoint"] != str(checkpoint_path.resolve())
                    or identity["jax_version"] != jax.__version__
                    or identity["source_profile_sha256"] != hashlib.sha256(args.source_profile.read_bytes()).hexdigest()):
                raise ValueError("replay declared input identity differs from compiled candidate")
            report["replay_source"] = {"path": str(Path(__file__).resolve()),
                                       "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            checkpoint = load_learning_checkpoint(checkpoint_path, expected_method=reference["method"])
            if checkpoint["completed_updates"] != 0:
                raise ValueError("requires declared update0")
            case = bundle["payloads"][0]["cases"][args.horizon]
            arguments = (jax.device_put(checkpoint["parameters"]), *jax.device_put(case))
            jax.block_until_ready(arguments)
            paths = {"default": args.reference_executable, "candidate": Path(candidate["executable"])}
            binaries, memory = {}, {}
            with gates.forbid_field_compile():
                for name, path in paths.items():
                    with path.open("rb") as stream:
                        serialized = pickle.load(stream)
                    binary = deserialize_and_load(*serialized)
                    validate_inputs(jax, serialized, binary, arguments)
                    del serialized
                    memory[name] = memory_sizes(binary)
                    if memory[name]["estimated_live_bytes"] > 24 * 2**30:
                        raise MemoryError("saved binary exceeds24GiB")
                    binaries[name] = binary
                initial = {name: jax.device_get(jax.block_until_ready(binary(*arguments))) for name, binary in binaries.items()}
            initial_checks = checks(initial["candidate"], initial["default"])
            validation = {name: valid(value) for name, value in initial.items()}
            emit("initial_compared", comparisons=initial_checks, validation=validation,
                 initial_parameter_leaf_comparisons=parameter_leaf_checks(initial["candidate"], initial["default"]),
                 binaries={name: str(path) for name, path in paths.items()}, compiled_memory=memory)
            full_pairs, self_pairs = [], {name: [] for name in binaries}
            leaf_pairs = []
            state_self = {name: [] for name in binaries}
            times = {name: [] for name in binaries}
            any_failed = not all(row["passed"] for row in initial_checks.values())
            if any_failed:
                for name, value in initial.items():
                    owner.atomic_pickle(args.output.with_suffix(f".initial.{name}.failed_full_result.pkl"), value)
            for repeat in range(7):
                order = ("default", "candidate") if repeat % 2 == 0 else ("candidate", "default")
                values = {}
                for name in order:
                    with gates.deadline(120, "warm_"+name, emit), gates.forbid_field_compile():
                        began = time.perf_counter()
                        value = jax.block_until_ready(binaries[name](*arguments))
                        times[name].append(time.perf_counter()-began)
                    values[name] = jax.device_get(value)
                    validation[name] = valid(values[name])
                    self_pairs[name].append(checks(values[name], initial[name]))
                    state_self[name].append(self_pairs[name][-1]["initial_state_gradient"])
                    owner.atomic_pickle(args.output.with_suffix(f".{name}.repeat{repeat}.initial_state_vjp.pkl"), values[name][1][1])
                    del value
                full_pairs.append(checks(values["candidate"], values["default"]))
                leaf_pairs.append(parameter_leaf_checks(values["candidate"], values["default"]))
                failed = not all(row["passed"] for row in full_pairs[-1].values())
                any_failed |= failed
                if failed:
                    for name, value in values.items():
                        owner.atomic_pickle(args.output.with_suffix(f".repeat{repeat}.{name}.failed_full_result.pkl"), value)
                emit(f"paired_replay_{repeat}", full_derivative_pair_replay=full_pairs,
                     parameter_leaf_pair_replay=leaf_pairs,
                     full_derivative_self_replay=self_pairs, initial_state_self_replay=state_self,
                     matched_warm_seconds=times, validation=validation)
                stats = jax.local_devices()[0].memory_stats() or {}
                if stats.get("peak_bytes_in_use", 24*2**30+1) > 24*2**30:
                    raise MemoryError("replay peak unavailable/exceeds24GiB")
                del values
            stats = jax.local_devices()[0].memory_stats() or {}
            if stats.get("peak_bytes_in_use", 24*2**30+1) > 24*2**30:
                raise MemoryError("replay peak unavailable/exceeds24GiB")
            emit("completed", status="numerical_failed" if any_failed else "passed", device_memory=stats,
                 default_over_candidate_median_ratio=float(np.median(times["default"])/np.median(times["candidate"])))
            if any_failed:
                raise gates.NumericalFailure("strict paired full derivative replay failed")
    except BaseException as error:
        emit("exception", status="resource_failed" if isinstance(error, MemoryError) else "numerical_failed" if isinstance(error, gates.NumericalFailure) else "error",
             error=str(error), error_type=type(error).__name__)
        raise


def main():
    parser = make_parser()
    parser.add_argument("--candidate-report", type=Path, required=True)
    args = parser.parse_args()
    args.component = f"T{args.horizon}"
    run(args)


if __name__ == "__main__":
    main()
