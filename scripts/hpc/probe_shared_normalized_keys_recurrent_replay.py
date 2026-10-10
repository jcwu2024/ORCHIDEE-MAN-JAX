"""Bounded saved recurrent binaries: raw state-gradient repeat evidence only."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import time

import numpy as np

from probe_shared_normalized_keys_recurrent import load, validate_inputs


def run(args):
    started = time.perf_counter()
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax, devices = initialize_gpu_backend()
    jax.config.update("jax_enable_x64", True)
    from jax.experimental.serialize_executable import deserialize_and_load
    from scripts.hpc.run_shared_learning import load_learning_checkpoint

    owner = load(args.patch_owner, "scripts.hpc.probe_shared_expert_real_daily")
    gates = load(args.gate_owner, "normalized_recurrent_replay_gates")
    forward = load(args.forward_probe, "scripts.hpc.probe_shared_forward_reverse")
    analyzer = load(Path(__file__).with_name("analyze_shared_normalized_key_results.py"), "normalized_recurrent_leaf_analyzer")
    prior = json.loads(args.recurrent_report.read_text())
    if args.output.exists() or prior["horizon"] != args.horizon or len(devices) != 1:
        raise ValueError("preserve output and require matching one-GPU recurrent case")
    gates.validate_runtime_report(args.runtime_preflight, jax.__version__)
    _profile, reference, bundle, context, checkpoint_path, _admissions = forward._assets(args)
    state_leaves = context["head"].contract.state_leaves
    def compare_state(actual, expected):
        return analyzer.annotate_state(analyzer.compare(actual, expected), state_leaves)
    checkpoint = load_learning_checkpoint(checkpoint_path, expected_method=reference["method"])
    if checkpoint["completed_updates"] != 0:
        raise ValueError("replay requires retained update0")
    case = bundle["payloads"][0]["cases"][args.horizon]
    arguments = (jax.device_put(checkpoint["parameters"]), *jax.device_put(case))
    jax.block_until_ready(arguments)
    paths = {
        "default": (args.reference_executable, args.recurrent_report.with_suffix(".reference-replay.pkl")),
        "normalized": (Path(prior["executable"]), Path(prior["result_artifact"])),
    }
    binaries, originals = {}, {}
    for name, (binary_path, result_path) in paths.items():
        with result_path.open("rb") as stream:
            artifact = pickle.load(stream)
        if artifact["identity"] != prior["identity"]:
            raise ValueError("saved main-process result identity drift")
        originals[name] = np.asarray(artifact["result"][1][1])
        del artifact
        with binary_path.open("rb") as stream:
            serialized = pickle.load(stream)
        with gates.forbid_field_compile():
            binaries[name] = deserialize_and_load(*serialized)
        validate_inputs(jax, serialized, binaries[name], arguments)
        del serialized
    report = {"status": "running", "scope": "saved_recurrent_binary_state_gradient_stability",
              "source_report": str(args.recurrent_report), "horizon": args.horizon,
              "identity": prior["identity"], "rtol": 1e-5, "atol": 1e-6,
              "new_model_gradient_compilations": 0, "gpu_count": 1,
              "production_adoption": False, "training_updates": 0, "repeats": []}
    first = {}
    owner.atomic_json(args.output, report)
    for repeat in range(args.repeats):
        order = ("default", "normalized") if repeat % 2 == 0 else ("normalized", "default")
        states, records = {}, {}
        for name in order:
            with gates.deadline(120, "saved_replay_" + name, lambda stage, **kw: None), gates.forbid_field_compile():
                began = time.perf_counter()
                value = binaries[name](*arguments)
                jax.block_until_ready(value)
                seconds = time.perf_counter() - began
            state = np.asarray(jax.device_get(value[1][1]))
            if state.shape != (3854,) or not np.all(np.isfinite(state)) or not np.any(state):
                raise gates.NumericalFailure("nonfinite/meaningless full initial-state replay")
            loss, auxiliary = jax.device_get(value[0])
            hard = {key: int(np.asarray(count)) for key, count in auxiliary["hard"].items()}
            if not np.isfinite(loss) or len(hard) != 14 or any(hard.values()):
                raise gates.NumericalFailure("saved replay scalar/physical gate failed")
            memory = jax.local_devices()[0].memory_stats() or {}
            if memory.get("peak_bytes_in_use", 24 * 2**30 + 1) > 24 * 2**30:
                raise MemoryError("saved replay process peak unavailable/exceeds24GiB")
            states[name] = state
            first.setdefault(name, state)
            records[name] = {
                "seconds": seconds,
                "loss": float(loss), "hard_counts": hard,
                "against_same_process_first": compare_state(state, first[name]),
                "against_main_process_saved": compare_state(state, originals[name]),
                "against_main_process_default": compare_state(state, originals["default"]),
                "state_gradient": state.tolist(),
            }
            del value
        report["repeats"].append({"repeat": repeat, "order": order, "arms": records,
                                  "paired_normalized_against_default": compare_state(states["normalized"], states["default"])})
        report["elapsed_seconds"] = time.perf_counter() - started
        owner.atomic_json(args.output, report)
        print(f"recurrent_saved_pair_{repeat}", flush=True)
    report.update(status="diagnostic_completed", elapsed_seconds=time.perf_counter() - started,
                  device_memory=jax.local_devices()[0].memory_stats())
    owner.atomic_json(args.output, report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--horizon", type=int, choices=(7, 30), required=True)
    p.add_argument("--repeats", type=int, default=7)
    for name in ("output", "recurrent-report", "reference-executable", "patch-owner", "gate-owner", "forward-probe",
                 "source-profile", "input-bundle", "prepared-context", "daily-admission-report", "seven-admission-report",
                 "thirty-admission-report", "support-owner-file", "runtime-preflight"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.repeats != 7:
        raise ValueError("this bounded diagnosis declares exactly seven alternating pairs")
    args.component = f"T{args.horizon}"
    run(args)


if __name__ == "__main__":
    main()
