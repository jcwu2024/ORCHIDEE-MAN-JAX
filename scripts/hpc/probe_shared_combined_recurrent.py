"""Bounded real T7/T30 gradient parity for expert128 plus source reuse.

One selected native-checkpointed recurrent component is compared at the
trusted fresh update-0 weights. This does not apply AdamW or accept training.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(os.environ.get("ORCHIDEE_REPO_ROOT", Path(__file__).resolve().parents[2]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MEMORY_LIMIT_BYTES = 24 * 2**30


def load_daily_probe(path: Path) -> Any:
    """Load the staged helper, absent from the immutable source checkout."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"required staged daily probe absent: {path}")
    name = "scripts.hpc.probe_shared_expert_real_daily"
    existing = sys.modules.get(name)
    if existing is not None and Path(existing.__file__).resolve() == path:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load staged daily probe: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def inspect_assets(args: Any, daily_probe: Any) -> dict[str, Any]:
    """Validate already prepared real inputs; never rebuild Teacher data."""

    profile, reference = daily_probe.trusted_profile(args)
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if current_head != reference["source_git_head"]:
        raise ValueError("recurrent replay requires the original immutable source checkout")
    horizon = int(args.horizon)
    if horizon not in (7, 30):
        raise ValueError("recurrent diagnostic horizon must be 7 or 30")
    key = f"T{horizon}"
    if set(profile["executables"]) != {"daily", "T7", "T30"}:
        raise ValueError("trusted profile must identify all three component executables")
    executable = Path(profile["executables"][key]["executable"]).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"trusted {key} executable absent: {executable}")

    profile_sha256 = hashlib.sha256(args.source_profile.read_bytes()).hexdigest()
    with args.input_bundle.open("rb") as stream:
        bundle = pickle.load(stream)
    artifact_ids = tuple(reference["method"]["training_plan"]["training_artifact_ids"][:2])
    checkpoint = daily_probe.initial_checkpoint(reference).resolve()
    if (
        bundle.get("schema_version") != "shared_serialized_dual_inputs_v1"
        or bundle.get("source_git_head") != current_head
        or bundle.get("profile_sha256") != profile_sha256
        or bundle.get("method") != reference["method"]
        or tuple(bundle.get("artifact_ids", ())) != artifact_ids
        or len(set(artifact_ids)) != 2
        or len(bundle.get("payloads", ())) != 2
        or Path(bundle.get("initial_checkpoint", "")).resolve() != checkpoint
    ):
        raise ValueError("compact input bundle differs from trusted profile/method/checkpoint")
    case = bundle["payloads"][0]["cases"][horizon]
    if len(case) != 5 or np.shape(case.days[0].continuous_state)[0] != horizon:
        raise ValueError(f"compact bundle {key} case does not have the declared horizon")
    if np.shape(case.targets["state"])[0] != horizon:
        raise ValueError(f"compact bundle {key} targets do not have the declared horizon")

    with args.prepared_context.open("rb") as stream:
        context = pickle.load(stream)
    expected_identity = daily_probe.context_identity(reference, checkpoint, artifact_ids[0])
    if context.get("identity") != expected_identity:
        raise ValueError("daily prepared context differs from trusted method/checkpoint/artifact")
    expected_daily = Path(profile["executables"]["daily"]["executable"]).resolve()
    if Path(context["daily_executable"]).resolve() != expected_daily:
        raise ValueError("daily prepared context executable differs from trusted profile")
    return {
        "profile": profile,
        "reference": reference,
        "context": context,
        "case": case,
        "executable": executable,
        "checkpoint": checkpoint,
        "artifact_id": artifact_ids[0],
        "profile_sha256": profile_sha256,
    }


def result_identity(assets: Mapping[str, Any], horizon: int, jax_version: str) -> dict[str, Any]:
    return assets["context"]["identity"] | {
        "horizon": horizon,
        "jax_version": jax_version,
        "source_profile_sha256": assets["profile_sha256"],
        "executable": str(assets["executable"]),
    }


def load_reference_result(path: Path, identity: Mapping[str, Any]) -> tuple[Any, list[float]]:
    with path.open("rb") as stream:
        saved = pickle.load(stream)
    if saved.get("identity") != identity:
        raise ValueError("reused recurrent reference result identity differs")
    return saved["result"], saved.get("warm_seconds", [])


def benchmark_recurrent(
    executable: Any, parameters: Any, case: Any, repeats: int, *, on_first_result=None
) -> tuple[Any, list[float]]:
    import jax

    if len(case) != 5 or repeats < 1:
        raise ValueError("recurrent benchmark requires a five-part case and positive repeats")
    result = executable(parameters, *case)
    jax.block_until_ready(result)
    if on_first_result is not None:
        on_first_result(result)
    times = []
    for _ in range(repeats):
        began = time.perf_counter()
        timed = executable(parameters, *case)
        jax.block_until_ready(timed)
        times.append(time.perf_counter() - began)
    return result, times


def lower_and_compile_recurrent(backward: Any, parameters: Any, case: Any, reference_path: Path, emit, patches):
    if not reference_path.is_file():
        raise FileNotFoundError("trusted recurrent result must be saved before candidate compilation")
    with patches:
        began = time.perf_counter()
        lowered = backward.lower(parameters, *case)
        lower_seconds = time.perf_counter() - began
        emit("candidate_before_backend_compile", lower_seconds=lower_seconds)
        began = time.perf_counter()
        candidate = lowered.compile()
        compile_seconds = time.perf_counter() - began
    return candidate, lower_seconds, compile_seconds


def compare_candidate(
    daily_probe: Any,
    jax: Any,
    candidate_result: Any,
    reference_result: Any,
) -> tuple[dict[str, Any], dict[str, int], bool]:
    candidate_hard = daily_probe.hard_counts(candidate_result[0][1])
    comparisons = {
        "loss": daily_probe.compare_trees(jax, candidate_result[0][0], reference_result[0][0]),
        "auxiliary": daily_probe.compare_trees(jax, candidate_result[0][1], reference_result[0][1]),
        "all_parameter_gradients": daily_probe.compare_trees(jax, candidate_result[1], reference_result[1]),
    }
    passed = all(row["passed"] for row in comparisons.values()) and not any(candidate_hard.values())
    return comparisons, candidate_hard, passed


def gpu_run(args: Any, daily_probe: Any, assets: Mapping[str, Any], emit) -> dict[str, Any]:
    import jax
    from jax.experimental.serialize_executable import deserialize_and_load, serialize

    from research.daily_coarse_graining.daily_operator import shared_process_head
    from research.daily_coarse_graining.daily_operator.shared_recurrent_training import (
        make_shared_recurrent_objective,
    )
    from scripts.hpc.probe_shared_compile_cache import _require_supported_jax
    from scripts.hpc.run_shared_learning import load_learning_checkpoint
    from scripts.hpc.run_shared_transformer_admission import memory_sizes

    devices = tuple(jax.local_devices())
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("select exactly one visible GPU for saved recurrent executable compatibility")
    _require_supported_jax(jax)
    if args.candidate_module_file is None or args.backbone_module_file is None:
        raise ValueError("GPU mode requires both staged decoder and backbone module files")
    checkpoint = load_learning_checkpoint(assets["checkpoint"], expected_method=assets["reference"]["method"])
    if checkpoint["completed_updates"] != 0:
        raise ValueError("recurrent diagnostic requires the fresh update-0 checkpoint")
    parameters = jax.device_put(checkpoint["parameters"])
    expected_count = int(assets["reference"]["method"]["daily_protocol"]["architecture"]["parameter_count"])
    actual_count = sum(np.size(value) for value in jax.tree_util.tree_leaves(parameters))
    if actual_count != expected_count:
        raise ValueError("update-0 parameter tree count differs from declared architecture")
    case = jax.device_put(assets["case"])
    declared_limit = int(assets["reference"]["method"]["configuration"]["gradient_memory_limit_gib"]) * 2**30
    limit = min(MEMORY_LIMIT_BYTES, declared_limit)
    identity = result_identity(assets, int(args.horizon), jax.__version__)
    reference_path = args.reference_result or args.output.with_suffix(".reference.pkl")
    candidate_path = args.output.with_suffix(".candidate.pkl")
    candidate_executable_path = args.output.with_suffix(".candidate.executable.pkl")
    if len({path.resolve() for path in (reference_path, candidate_path, candidate_executable_path)}) != 3:
        raise ValueError("reference, candidate result, and candidate executable paths must differ")
    if args.reference_result is not None:
        reference_result, reference_times = load_reference_result(reference_path, identity)
        emit("trusted_recurrent_reference_reused", reference_artifact=str(reference_path))
    else:
        emit("load_trusted_recurrent_executable", executable=str(assets["executable"]))
        with assets["executable"].open("rb") as stream:
            serialized = pickle.load(stream)
        trusted = deserialize_and_load(*serialized)
        del serialized
        reference_memory = memory_sizes(trusted)
        if reference_memory["estimated_live_bytes"] > limit:
            raise ValueError("trusted recurrent executable exceeds declared 24-GiB memory bound")
        emit("trusted_recurrent_executable_loaded", reference_memory=reference_memory)

        def preserve_reference(result):
            nonlocal reference_result
            reference_result = jax.device_get(result)
            daily_probe.atomic_pickle(
                reference_path, {"identity": identity, "result": reference_result, "warm_seconds": []}
            )
            emit("trusted_recurrent_reference_saved", reference_artifact=str(reference_path))

        result, reference_times = benchmark_recurrent(
            trusted, parameters, case, args.repeats, on_first_result=preserve_reference
        )
        del trusted, result
    reference_hard = daily_probe.hard_counts(reference_result[0][1])
    if any(reference_hard.values()):
        raise ValueError(f"trusted recurrent reference hard gate failed: {reference_hard}")

    adapter = daily_probe.load_staged_adapter(args.candidate_module_file)
    backbone = daily_probe.load_backbone_assembler(args.backbone_module_file)
    context = assets["context"]
    _rollout, objective = make_shared_recurrent_objective(
        context["head"], context["template"], context["state_scales"], context["science_scales"]
    )
    backward = jax.jit(jax.value_and_grad(objective, has_aux=True))
    emit("candidate_before_lower")
    candidate, lower_seconds, compile_seconds = lower_and_compile_recurrent(
        backward,
        parameters,
        case,
        reference_path,
        emit,
        daily_probe.candidate_process_head_patches(shared_process_head, adapter, backbone),
    )
    candidate_memory = memory_sizes(candidate)
    emit("candidate_compiled", compile_seconds=compile_seconds, candidate_memory=candidate_memory)
    if candidate_memory["estimated_live_bytes"] > limit:
        raise ValueError("candidate recurrent executable exceeds declared 24-GiB memory bound")
    daily_probe.atomic_pickle(candidate_executable_path, serialize(candidate))
    emit("candidate_executable_saved", candidate_executable=str(candidate_executable_path))

    candidate_result = None

    def preserve_candidate(result):
        nonlocal candidate_result
        candidate_result = jax.device_get(result)
        daily_probe.atomic_pickle(
            candidate_path, {"identity": identity, "result": candidate_result, "warm_seconds": []}
        )
        emit("candidate_result_saved", candidate_artifact=str(candidate_path))

    result, candidate_times = benchmark_recurrent(
        candidate, parameters, case, args.repeats, on_first_result=preserve_candidate
    )
    comparisons, candidate_hard, passed = compare_candidate(daily_probe, jax, candidate_result, reference_result)
    return {
        "status": "passed" if passed else "parity_failed",
        "horizon": int(args.horizon),
        "artifact_id": assets["artifact_id"],
        "comparisons": comparisons,
        "all_parameter_gradient_coordinates": comparisons["all_parameter_gradients"]["coordinates"],
        "reference_hard": reference_hard,
        "candidate_hard": candidate_hard,
        "reference_warm_seconds": reference_times,
        "candidate_warm_seconds": candidate_times,
        "reference_artifact": str(reference_path),
        "candidate_artifact": str(candidate_path),
        "candidate_executable": str(candidate_executable_path),
        "candidate_memory": candidate_memory,
        "lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "memory_limit_bytes": limit,
        "scientific_acceptance": False,
    }


def run(args: Any) -> dict[str, Any]:
    daily_probe = load_daily_probe(Path(__file__).with_name("probe_shared_expert_real_daily.py"))
    if getattr(args, "prepare_only", False) or getattr(args, "reuse_reference", False):
        raise ValueError("use --inspect-only or --reference-result for the recurrent probe")
    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "running",
        "scope": "actual_canonical_recurrent_single_horizon_no_daily_no_optimizer",
        "horizon": int(args.horizon),
        "scientific_acceptance": False,
        "rtol": daily_probe.RTOL,
        "atol": daily_probe.ATOL,
    }

    def emit(stage: str, **fields: Any) -> None:
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **fields)
        daily_probe.atomic_json(args.output, report)
        print(
            f"shared_combined_recurrent T{args.horizon} stage={stage} seconds={report['elapsed_seconds']:.1f}",
            flush=True,
        )

    try:
        emit("inspect_assets")
        assets = inspect_assets(args, daily_probe)
        if args.inspect_only:
            from research.daily_coarse_graining.daily_operator.shared_recurrent_training import (
                make_shared_recurrent_objective,
            )

            context = assets["context"]
            _rollout, objective = make_shared_recurrent_objective(
                context["head"], context["template"], context["state_scales"], context["science_scales"]
            )
            if not callable(objective):
                raise ValueError("prepared recurrent objective did not bind")
            if args.candidate_module_file is not None or args.backbone_module_file is not None:
                if args.candidate_module_file is None or args.backbone_module_file is None:
                    raise ValueError("inspect mode needs both staged decoder and backbone module files")
                daily_probe.load_staged_adapter(args.candidate_module_file)
                daily_probe.load_backbone_assembler(args.backbone_module_file)
            result = {
                "status": "inspected",
                "horizon": int(args.horizon),
                "artifact_id": assets["artifact_id"],
                "trusted_executable": str(assets["executable"]),
                "checkpoint": str(assets["checkpoint"]),
                "source_profile_sha256": assets["profile_sha256"],
                "recurrent_objective_bound": True,
                "staged_adapters_loaded": args.candidate_module_file is not None,
                "gpu_gradient_compiled": False,
            }
        else:
            emit("gpu_start")
            result = gpu_run(args, daily_probe, assets, emit)
        emit("completed", **result)
        return report
    except Exception as error:
        emit("exception", status="error", error_type=type(error).__name__, error=str(error))
        raise


def parser():
    daily_probe = load_daily_probe(Path(__file__).with_name("probe_shared_expert_real_daily.py"))
    result = daily_probe.parser()
    result.description = __doc__
    result.add_argument("--horizon", type=int, choices=(7, 30), required=True)
    result.add_argument("--inspect-only", action="store_true")
    return result


if __name__ == "__main__":
    outcome = run(parser().parse_args())
    raise SystemExit(0 if outcome["status"] in {"passed", "inspected"} else 1)
