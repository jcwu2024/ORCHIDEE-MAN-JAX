"""Bounded selective native-checkpoint policy test on saved T7/T30 inputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

RTOL, ATOL = 1e-5, 1e-6
POLICY = "dots_with_no_batch_dims_saveable"
MEMORY_LIMIT = 24 * 2**30


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load diagnostic overlay: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PolicyJax:
    """Replace only checkpoint policy on one bound RolloutLoss instance."""

    def __init__(self, jax, policy):
        self._jax = jax
        self._policy = policy

    def __getattr__(self, name):
        return getattr(self._jax, name)

    def checkpoint(self, fun):
        return self._jax.checkpoint(fun, policy=self._policy)


def matched_summary(reference: list[float], candidate: list[float]) -> dict:
    if len(reference) < 3 or len(reference) != len(candidate):
        raise ValueError("need at least three matched pairs")
    old, new = float(np.median(reference)), float(np.median(candidate))
    if min(old, new) <= 0:
        raise ValueError("timings must be positive")
    return {"reference_median_seconds": old, "candidate_median_seconds": new,
            "speed_ratio": old / new, "wall_reduction_fraction": 1 - new / old}


def _hard(result) -> dict[str, int]:
    hard = result[0][1]["hard"]
    return {name: int(np.asarray(value)) for name, value in hard.items()}


def _identity(args, assets, jax):
    reference = assets[1]
    return {
        "source_head": reference["source_git_head"],
        "source_profile_sha256": hashlib.sha256(args.source_profile.read_bytes()).hexdigest(),
        "method": reference["method"],
        "artifact_id": int(assets[2]["artifact_ids"][0]),
        "horizon": args.horizon,
        "checkpoint": str(assets[4].resolve()),
        "jax_version": jax.__version__,
        "admitted_executable": assets[5][f"T{args.horizon}"]["executable"],
        "candidate_decoder_sha256": hashlib.sha256(args.candidate_module_file.read_bytes()).hexdigest(),
        "candidate_backbone_sha256": hashlib.sha256(args.backbone_module_file.read_bytes()).hexdigest(),
    }


def run(args, report):
    import jax
    from jax.experimental.serialize_executable import deserialize_and_load, serialize
    from research.daily_coarse_graining.daily_operator import shared_process_head
    from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_recurrent_objective
    from scripts.hpc.probe_shared_execution import compare_trees
    from scripts.hpc.run_shared_learning import load_learning_checkpoint
    from scripts.hpc.run_shared_transformer_admission import memory_sizes

    parent = _load_module(args.forward_probe, "scripts.hpc.probe_shared_forward_reverse")
    assets = parent._assets(args)
    profile, reference, bundle, context, checkpoint, admissions = assets
    identity = _identity(args, assets, jax)
    report.update(stage="assets_validated", identity=identity)
    _write(args.output, report)
    state = load_learning_checkpoint(checkpoint, expected_method=reference["method"])
    if state["completed_updates"] != 0:
        raise ValueError("requires fixed update-0 parameters")
    parameters = jax.device_put(state["parameters"])
    case = bundle["payloads"][0]["cases"][args.horizon]
    if len(case) != 5 or np.shape(case.days[0].continuous_state)[0] != args.horizon:
        raise ValueError("saved recurrent horizon differs")
    call_args = (parameters, *jax.device_put(case))
    patch_owner = _load_module(args.patch_owner, "scripts.hpc.probe_shared_expert_real_daily")
    adapter = patch_owner.load_staged_adapter(args.candidate_module_file)
    backbone = patch_owner.load_backbone_assembler(args.backbone_module_file)
    rollout, objective = make_shared_recurrent_objective(
        context["head"], context["template"], context["state_scales"], context["science_scales"])
    if args.stage in ("candidate", "screen"):
        policy = getattr(jax.checkpoint_policies, POLICY)
        rollout.jax = PolicyJax(jax, policy)
    backward = jax.jit(jax.value_and_grad(
        objective, argnums=(0 if args.stage == "screen" else (0, 1)), has_aux=True))
    jax.block_until_ready(call_args)
    report["stage"] = "lowering"
    _write(args.output, report)
    started = time.perf_counter()
    with patch_owner.candidate_process_head_patches(shared_process_head, adapter, backbone):
        lowered = backward.lower(*call_args)
    report.update(stage="compiling", lower_seconds=time.perf_counter() - started)
    _write(args.output, report)
    started = time.perf_counter()
    compiled = lowered.compile()
    report.update(stage="compiled", compile_seconds=time.perf_counter() - started,
                  compiled_memory=memory_sizes(compiled))
    _write(args.output, report)
    if report["compiled_memory"]["estimated_live_bytes"] > MEMORY_LIMIT:
        raise MemoryError("compiled estimated live bytes exceed 24 GiB")
    if args.stage != "reference":
        candidate_executable = args.output.with_suffix(".executable.pkl")
        _pickle(candidate_executable, serialize(compiled))
        report.update(stage="candidate_executable_saved",
                      candidate_executable=str(candidate_executable))
        _write(args.output, report)
    result = compiled(*call_args)
    jax.block_until_ready(result)
    actual_peak = (jax.local_devices()[0].memory_stats() or {}).get("peak_bytes_in_use")
    report.update(stage="first_result", actual_peak_bytes_before_other_executable=actual_peak)
    _write(args.output, report)
    if actual_peak is None or int(actual_peak) > MEMORY_LIMIT:
        raise MemoryError("observed device peak exceeds 24 GiB")
    host_result = jax.device_get(result)
    if args.stage != "reference":
        candidate_result = args.output.with_suffix(".result.pkl")
        _pickle(candidate_result, {"identity": identity, "result": host_result})
        report.update(stage="candidate_result_saved", candidate_result=str(candidate_result))
        _write(args.output, report)
    hard = _hard(host_result)
    if any(hard.values()):
        raise ValueError("recurrent physical hard counts are nonzero")
    report["hard_counts"] = hard
    _write(args.output, report)
    if args.stage == "screen":
        with Path(admissions[f"T{args.horizon}"]["executable"]).open("rb") as stream:
            admitted = deserialize_and_load(*pickle.load(stream))
        trusted_result = admitted(*call_args)
        jax.block_until_ready(trusted_result)
        comparisons = {
            "loss_and_auxiliary": compare_trees(jax, host_result[0],
                                                 jax.device_get(trusted_result[0]),
                                                 rtol=RTOL, atol=ATOL),
            "all_parameter_gradients": compare_trees(jax, host_result[1],
                                                     jax.device_get(trusted_result[1]),
                                                     rtol=RTOL, atol=ATOL),
        }
        if not all(row["passed"] for row in comparisons.values()):
            raise ValueError("selective policy failed admitted parameter-gradient parity")
        reference_times, candidate_times = [], []
        for index in range(args.repeats):
            for name in (("reference", "candidate") if index % 2 == 0 else ("candidate", "reference")):
                jax.block_until_ready(call_args)
                began = time.perf_counter()
                value = (admitted if name == "reference" else compiled)(*call_args)
                jax.block_until_ready(value)
                (reference_times if name == "reference" else candidate_times).append(time.perf_counter() - began)
        report.update(status="passed", stage="completed", comparisons=comparisons,
                      reference_seconds=reference_times, candidate_seconds=candidate_times,
                      timing=matched_summary(reference_times, candidate_times),
                      device_memory_after_both_executables=jax.local_devices()[0].memory_stats(),
                      peak_memory_semantics="candidate first-result peak excludes reference executable; later peak is process-cumulative",
                      initial_state_gradient_checked=False,
                      retained_updates=0, scientific_acceptance=False)
        _write(args.output, report)
        return
    if args.stage == "reference":
        # Keep this expensive complete reference before loading any other graph.
        _pickle(args.reference_result, {"identity": identity, "result": host_result})
        _pickle(args.reference_executable, serialize(compiled))
        report.update(stage="reference_saved", reference_result=str(args.reference_result),
                      reference_executable=str(args.reference_executable))
        _write(args.output, report)
        with Path(admissions[f"T{args.horizon}"]["executable"]).open("rb") as stream:
            admitted = deserialize_and_load(*pickle.load(stream))
        admitted_result = admitted(*call_args)
        jax.block_until_ready(admitted_result)
        checks = {
            "loss_and_auxiliary": compare_trees(jax, host_result[0], jax.device_get(admitted_result[0]),
                                                 rtol=RTOL, atol=ATOL),
            "all_parameter_gradients": compare_trees(jax, host_result[1][0],
                                                     jax.device_get(admitted_result[1]),
                                                     rtol=RTOL, atol=ATOL),
        }
        state_gradient = np.asarray(host_result[1][1])
        state_finite = bool(np.all(np.isfinite(state_gradient)))
        if not all(row["passed"] for row in checks.values()) or not state_finite:
            raise ValueError("dual reference differs from admitted parameter-gradient executable")
        report.update(status="passed", stage="completed", comparisons=checks,
                      initial_state_gradient_coordinates=int(state_gradient.size),
                      initial_state_gradient_finite=state_finite,
                      initial_state_gradient_nonzero=int(np.count_nonzero(state_gradient)))
        _write(args.output, report)
        return

    reference_report = json.loads(args.reference_report.read_text(encoding="utf-8"))
    if (reference_report.get("status") != "passed"
            or reference_report.get("stage") != "completed"
            or reference_report.get("identity") != identity
            or not reference_report.get("initial_state_gradient_finite")
            or any(reference_report.get("hard_counts", {}).values())
            or set(reference_report.get("comparisons", {})) != {"loss_and_auxiliary", "all_parameter_gradients"}
            or not all(row.get("passed") for row in reference_report["comparisons"].values())):
        raise ValueError("dual reference report has not passed admission")
    with args.reference_result.open("rb") as stream:
        saved = pickle.load(stream)
    if saved.get("identity") != identity:
        raise ValueError("saved dual reference identity differs")
    old = saved["result"]
    checks = {
        "loss_and_auxiliary": compare_trees(jax, host_result[0], old[0], rtol=RTOL, atol=ATOL),
        "all_parameter_gradients": compare_trees(jax, host_result[1][0], old[1][0], rtol=RTOL, atol=ATOL),
        "initial_state_gradient": compare_trees(jax, host_result[1][1], old[1][1], rtol=RTOL, atol=ATOL),
    }
    report.update(stage="parity_checked", comparisons=checks)
    _write(args.output, report)
    if not all(row["passed"] for row in checks.values()):
        raise ValueError("selective checkpoint policy failed complete derivative parity")
    report.update(stage="parity_passed")
    _write(args.output, report)
    with args.reference_executable.open("rb") as stream:
        trusted = deserialize_and_load(*pickle.load(stream))
    trusted_result = trusted(*call_args)
    jax.block_until_ready(trusted_result)
    reference_times, candidate_times = [], []
    for index in range(args.repeats):
        for name in (("reference", "candidate") if index % 2 == 0 else ("candidate", "reference")):
            jax.block_until_ready(call_args)
            began = time.perf_counter()
            value = (trusted if name == "reference" else compiled)(*call_args)
            jax.block_until_ready(value)
            (reference_times if name == "reference" else candidate_times).append(time.perf_counter() - began)
    report.update(status="passed", stage="completed", reference_seconds=reference_times,
                  candidate_seconds=candidate_times,
                  timing=matched_summary(reference_times, candidate_times),
                  device_memory_after_both_executables=jax.local_devices()[0].memory_stats(),
                  peak_memory_semantics="candidate first-result peak excludes reference executable; later peak is process-cumulative",
                  retained_updates=0, scientific_acceptance=False)
    _write(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("reference", "candidate", "screen"), required=True)
    parser.add_argument("--horizon", type=int, choices=(7, 30), required=True)
    for name in ("source_profile", "input_bundle", "prepared_context", "daily_admission_report",
                 "seven_admission_report", "thirty_admission_report", "patch_owner",
                 "support_owner_file", "candidate_module_file", "backbone_module_file",
                 "forward_probe", "reference_result", "reference_executable", "reference_report", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    if len({args.reference_result.resolve(), args.reference_executable.resolve(), args.output.resolve()}) != 3:
        parser.error("reference artifacts and report paths must differ")
    report = {"status": "running", "stage": "starting", "mode": args.stage,
              "horizon": args.horizon, "candidate_policy": POLICY,
              "parameter_and_initial_state_gradients": args.stage != "screen",
              "memory_limit_bytes": MEMORY_LIMIT}
    _write(args.output, report)
    try:
        run(args, report)
    except BaseException as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error))
        _write(args.output, report)
        raise


if __name__ == "__main__":
    main()
