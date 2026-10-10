"""Bounded forward-only versus native-value-and-grad component timing.

The difference in timings is marginal AD execution cost, not an isolated
reverse pass: native checkpointing may recompute forward operations.
"""
from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SOURCE_HEAD = "c599236bcf57b745e9bafed678298010f4c3b58f"
RTOL, ATOL = 1e-5, 1e-6


def timing_summary(forward: list[float], combined: list[float]) -> dict:
    if len(forward) < 3 or len(forward) != len(combined):
        raise ValueError("need at least three matched forward/combined pairs")
    forward_median = float(np.median(forward))
    combined_median = float(np.median(combined))
    if forward_median <= 0 or combined_median <= 0:
        raise ValueError("timings must be positive")
    marginal = combined_median - forward_median
    return {
        "forward_median_seconds": forward_median,
        "combined_median_seconds": combined_median,
        "marginal_ad_seconds": marginal,
        "forward_fraction_of_combined": forward_median / combined_median,
        "marginal_ad_fraction_of_combined": marginal / combined_median,
        "interpretation": "separate compiled graphs; marginal AD includes rematerialized forward work",
    }


def _write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _assets(args):
    import jax

    root = Path(os.environ["ORCHIDEE_REPO_ROOT"])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != SOURCE_HEAD:
        raise ValueError("requires the immutable c599236 checkout")
    profile_bytes = args.source_profile.read_bytes()
    profile = json.loads(profile_bytes)
    reference = json.loads(Path(profile["technical_update_report"]).read_text(encoding="utf-8"))
    if profile.get("status") != "passed" or reference.get("status") != "passed":
        raise ValueError("source profile/reference is not admitted")
    if reference["source_git_head"] != head or set(profile["executables"]) != {"daily", "T7", "T30"}:
        raise ValueError("source profile differs from immutable checkout or component set")
    with args.input_bundle.open("rb") as stream:
        bundle = pickle.load(stream)
    artifact_ids = tuple(reference["method"]["training_plan"]["training_artifact_ids"][:2])
    checkpoint = next(Path(row["path"]) for row in reference["checkpoints"]
                      if row["update"] == 0 and row["retained_milestone"])
    if (bundle.get("schema_version") != "shared_serialized_dual_inputs_v1"
            or bundle.get("source_git_head") != head
            or bundle.get("profile_sha256") != hashlib.sha256(profile_bytes).hexdigest()
            or bundle.get("method") != reference["method"]
            or tuple(bundle.get("artifact_ids", ())) != artifact_ids
            or Path(bundle.get("initial_checkpoint", "")).resolve() != checkpoint.resolve()):
        raise ValueError("saved input bundle differs from profile/method/checkpoint")
    with args.prepared_context.open("rb") as stream:
        context = pickle.load(stream)
    identity = {
        "schema_version": "shared_expert_real_daily_context_v1",
        "source_git_head": head,
        "method": reference["method"],
        "initial_checkpoint": str(checkpoint.resolve()),
        "artifact_id": int(artifact_ids[0]),
        "daily_rows": 4,
    }
    if context.get("identity") != identity:
        raise ValueError("prepared context differs from profile/method/checkpoint")
    if Path(context["daily_executable"]).resolve() != Path(profile["executables"]["daily"]["executable"]).resolve():
        raise ValueError("prepared context daily executable differs from profile")
    if len(jax.local_devices()) != 1 or jax.local_devices()[0].platform != "gpu":
        raise ValueError("requires exactly one visible GPU")
    owner = _load_module(args.support_owner_file, "scripts.hpc.probe_shared_serialized_dual")
    admissions = owner.validate_admissions(profile, reference, {
        "daily": args.daily_admission_report,
        "T7": args.seven_admission_report,
        "T30": args.thirty_admission_report,
    }, profile_sha256=hashlib.sha256(profile_bytes).hexdigest())
    return profile, reference, bundle, context, checkpoint, admissions


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load diagnostic overlay: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run(args, report):
    import jax
    import jax.numpy as jnp
    from jax.experimental.serialize_executable import deserialize_and_load
    from research.daily_coarse_graining.daily_operator.shared_recurrent_training import (
        make_shared_joint_daily_objective, make_shared_recurrent_objective,
    )
    from research.daily_coarse_graining.daily_operator import shared_process_head
    from scripts.hpc.run_shared_learning import load_learning_checkpoint
    from scripts.hpc.shared_daily_microbatch import make_microbatch_objective
    from scripts.hpc.probe_shared_execution import compare_trees

    profile, reference, bundle, context, checkpoint, admissions = _assets(args)
    report.update(stage="assets_validated", source_head=SOURCE_HEAD,
                  source_profile_sha256=hashlib.sha256(args.source_profile.read_bytes()).hexdigest(),
                  artifact_id=int(bundle["artifact_ids"][0]),
                  admitted_executable=admissions[args.component]["executable"],
                  admission_report=admissions[args.component]["admission_report"])
    _write(args.output, report)
    state = load_learning_checkpoint(checkpoint, expected_method=reference["method"])
    if state["completed_updates"] != 0:
        raise ValueError("requires the update-0 fixed-weight diagnostic checkpoint")
    parameters = jax.device_put(state["parameters"])
    payload = bundle["payloads"][0]
    if args.component == "daily":
        objective = make_shared_joint_daily_objective(
            context["runtime"], context["head"], context["template"],
            context["state_scales"], context["science_scales"])
        objective = make_microbatch_objective(jax, jnp, objective)
        daily = payload["daily"]
        leaves = jax.tree_util.tree_leaves(daily)
        if not leaves or any(np.shape(x)[0] != 32 for x in leaves):
            raise ValueError("saved daily batch is not 32 rows")
        sample = jax.device_put(jax.tree_util.tree_map(lambda x: np.asarray(x)[:4].copy(), daily))
        call_args = (parameters, sample)
    else:
        _rollout, objective = make_shared_recurrent_objective(
            context["head"], context["template"], context["state_scales"],
            context["science_scales"])
        horizon = int(args.component[1:])
        case = payload["cases"][horizon]
        if len(case) != 5 or np.shape(case.days[0].continuous_state)[0] != horizon:
            raise ValueError("saved recurrent case has wrong horizon")
        call_args = (parameters, *jax.device_put(case))
    path = Path(admissions[args.component]["executable"])
    with path.open("rb") as stream:
        combined = deserialize_and_load(*pickle.load(stream))
    jax.block_until_ready(call_args)
    combined_result = combined(*call_args)
    jax.block_until_ready(combined_result)
    forward = jax.jit(objective)
    report["stage"] = "forward_lowering"
    _write(args.output, report)
    started = time.perf_counter()
    owner = _load_module(args.patch_owner, "scripts.hpc.probe_shared_expert_real_daily")
    adapter = owner.load_staged_adapter(args.candidate_module_file)
    backbone = owner.load_backbone_assembler(args.backbone_module_file)
    with owner.candidate_process_head_patches(shared_process_head, adapter, backbone):
        lowered = forward.lower(*call_args)
    report.update(stage="forward_compiling", lower_seconds=time.perf_counter() - started)
    _write(args.output, report)
    started = time.perf_counter()
    forward = lowered.compile()
    report.update(stage="forward_compiled", compile_seconds=time.perf_counter() - started)
    _write(args.output, report)
    forward_result = forward(*call_args)
    jax.block_until_ready(forward_result)
    value_check = compare_trees(jax, jax.device_get(forward_result),
                                jax.device_get(combined_result[0]), rtol=RTOL, atol=ATOL)
    hard = jax.device_get(forward_result[1]["hard"])
    hard_counts = {key: int(np.asarray(value)) for key, value in hard.items()}
    if not value_check["passed"] or any(hard_counts.values()):
        raise ValueError("forward-only value/auxiliary or physical parity failed")
    report.update(stage="parity_passed", comparison=value_check, hard_counts=hard_counts)
    _write(args.output, report)
    forward_times, combined_times = [], []
    for index in range(args.repeats):
        order = ("forward", "combined") if index % 2 == 0 else ("combined", "forward")
        for name in order:
            jax.block_until_ready(call_args)
            began = time.perf_counter()
            result = (forward if name == "forward" else combined)(*call_args)
            jax.block_until_ready(result)
            (forward_times if name == "forward" else combined_times).append(time.perf_counter() - began)
    report.update(status="passed", stage="completed", comparison=value_check,
                  forward_seconds=forward_times, combined_seconds=combined_times,
                  timing=timing_summary(forward_times, combined_times),
                  device_memory=jax.local_devices()[0].memory_stats(),
                  gradient_reference="saved combined executable; full gradient output synchronized",
                  retained_updates=0, scientific_acceptance=False)
    _write(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("daily", "T7", "T30"), required=True)
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--input-bundle", type=Path, required=True)
    parser.add_argument("--prepared-context", type=Path, required=True)
    parser.add_argument("--daily-admission-report", type=Path, required=True)
    parser.add_argument("--seven-admission-report", type=Path, required=True)
    parser.add_argument("--thirty-admission-report", type=Path, required=True)
    parser.add_argument("--patch-owner", type=Path, required=True)
    parser.add_argument("--support-owner-file", type=Path, required=True)
    parser.add_argument("--candidate-module-file", type=Path, required=True)
    parser.add_argument("--backbone-module-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    report = {"status": "running", "stage": "starting", "component": args.component,
              "repeats": args.repeats, "timing_semantics": "matched forward-only versus combined marginal AD",
              "model_gradient_compilations": 0, "forward_compilations": 1}
    _write(args.output, report)
    try:
        _run(args, report)
    except BaseException as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error))
        _write(args.output, report)
        raise


if __name__ == "__main__":
    main()
