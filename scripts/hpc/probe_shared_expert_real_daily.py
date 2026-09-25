"""Compare expert128 against the trusted canonical daily micro4 gradient.

This is a diagnostic daily contribution only. It does not execute T7/T30,
compose a full joint update, apply AdamW, or establish scientific acceptance.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(os.environ.get("ORCHIDEE_REPO_ROOT", Path(__file__).resolve().parents[2]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RTOL = 1e-5
ATOL = 1e-6


def atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def trusted_profile(args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = json.loads(args.source_profile.read_text(encoding="utf-8"))
    reference = json.loads(Path(profile["technical_update_report"]).read_text(encoding="utf-8"))
    if profile.get("status") != "passed" or "daily" not in profile.get("executables", {}):
        raise ValueError("trusted profile lacks a completed daily executable")
    if reference.get("status") != "passed" or reference.get("completed_updates") != 2:
        raise ValueError("trusted reference is not the completed two-update technical run")
    if reference.get("technical_updates") != 2 or reference.get("method") is None:
        raise ValueError("trusted reference method is missing")
    return profile, reference


def initial_checkpoint(reference: Mapping[str, Any]) -> Path:
    rows = [row for row in reference["checkpoints"] if row["update"] == 0 and row["retained_milestone"]]
    if len(rows) != 1:
        raise ValueError("reference needs exactly one retained update-0 checkpoint")
    return Path(rows[0]["path"])


def minimal_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only values read by the canonical daily objective and scale check."""

    return {key: runtime[key] for key in ("loss_scales", "base_statistics", "process_statistics")}


def slice_daily_micro4(jax: Any, daily: Any) -> Any:
    leaves = jax.tree_util.tree_leaves(daily)
    if not leaves or any(not np.shape(value) or int(np.shape(value)[0]) != 32 for value in leaves):
        raise ValueError("compact bundle daily batch is not the declared 32-row pytree")
    return jax.tree_util.tree_map(lambda value: np.asarray(value)[:4].copy(), daily)


def context_identity(reference: Mapping[str, Any], checkpoint: Path, artifact_id: int) -> dict[str, Any]:
    return {
        "schema_version": "shared_expert_real_daily_context_v1",
        "source_git_head": reference["source_git_head"],
        "method": reference["method"],
        "initial_checkpoint": str(checkpoint.resolve()),
        "artifact_id": int(artifact_id),
        "daily_rows": 4,
    }


def prepare(args: Any, emit) -> dict[str, Any]:
    from scripts.hpc import run_shared_learning as learning
    from scripts.hpc.shared_learning_plan import load_learning_configuration
    from scripts.hpc.shared_transformer_runtime import load_shared_runtime

    profile, reference = trusted_profile(args)
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if current_head != reference["source_git_head"]:
        raise ValueError("CPU preparation must use the original immutable source checkout")
    loaded = load_learning_configuration(
        args.protocol, args.joint_admission_report, args.prepared_manifest, args.validation_manifest
    )
    if learning._method_identity(loaded, technical_updates=2) != reference["method"]:
        raise ValueError("preparation arguments differ from trusted technical method")
    with args.input_bundle.open("rb") as stream:
        bundle = pickle.load(stream)
    artifact_ids = tuple(reference["method"]["training_plan"]["training_artifact_ids"][:2])
    checkpoint = initial_checkpoint(reference)
    if (
        bundle.get("schema_version") != "shared_serialized_dual_inputs_v1"
        or bundle.get("source_git_head") != current_head
        or bundle.get("method") != reference["method"]
        or tuple(bundle.get("artifact_ids", ())) != artifact_ids
        or len(bundle.get("payloads", ())) != 2
        or Path(bundle.get("initial_checkpoint", "")).resolve() != checkpoint.resolve()
    ):
        raise ValueError("compact input bundle differs from trusted method/checkpoint/artifacts")
    daily4 = slice_daily_micro4(__import__("jax"), bundle["payloads"][0]["daily"])
    emit("load_shared_runtime", artifact_id=artifact_ids[0])
    args.cpu_forward_only = True
    runtime, parent, template, _artifacts, _metadata = load_shared_runtime(args, artifact_ids[:1])
    from research.daily_coarse_graining.daily_operator.shared_persistent_outputs import (
        build_shared_persistent_coordinates,
    )
    from research.daily_coarse_graining.daily_operator.shared_process_head import build_shared_process_head
    from research.daily_coarse_graining.daily_operator.shared_training import derive_shared_state_change_scales

    coordinates = build_shared_persistent_coordinates(
        parent.learned_schema, parent.contract, runtime["base_statistics"].arrays["state_delta"]
    )
    architecture = reference["method"]["daily_protocol"]["architecture"]
    head = build_shared_process_head(
        parent,
        template,
        config=learning.shared_config(architecture),
        thermal_transform=architecture["thermal_transform"],
        persistent_coordinates=coordinates,
        source_bare_cimean=True,
    )
    reduced_runtime = minimal_runtime(runtime)
    state_scales = derive_shared_state_change_scales(reduced_runtime, head)
    identity = context_identity(reference, checkpoint, artifact_ids[0])
    context = {
        "identity": identity,
        "runtime": reduced_runtime,
        "head": head,
        "template": template,
        "state_scales": state_scales,
        "science_scales": loaded["admission"]["scales"],
        "daily4": daily4,
        "daily_executable": str(Path(profile["executables"]["daily"]["executable"]).resolve()),
    }
    emit("save_prepared_context")
    atomic_pickle(args.prepared_context, context)
    with args.prepared_context.open("rb") as stream:
        restored = pickle.load(stream)
    from research.daily_coarse_graining.daily_operator.shared_recurrent_training import (
        make_shared_joint_daily_objective,
    )

    if restored["identity"] != identity:
        raise ValueError("prepared context round-trip changed method identity")
    make_shared_joint_daily_objective(
        restored["runtime"],
        restored["head"],
        restored["template"],
        restored["state_scales"],
        restored["science_scales"],
    )
    emit("prepared_context_roundtrip_and_daily_objective_bound")
    return {
        "status": "prepared",
        "prepared_context": str(args.prepared_context),
        "prepared_context_bytes": args.prepared_context.stat().st_size,
        "artifact_id": artifact_ids[0],
        "daily_rows": 4,
        "daily_executable": context["daily_executable"],
        "checkpoint": str(checkpoint),
        "model_initialized": False,
        "gpu_gradient_compiled": False,
    }


def load_staged_adapter(path: Path):
    """Load only the three explicitly staged diagnostic siblings in order."""

    package = "research.daily_coarse_graining.daily_operator"
    for stem in ("shared_decoder_dense", "shared_decoder_macro_packed", "shared_decoder_expert_execution"):
        file = path.with_name(stem + ".py")
        if not file.is_file():
            raise FileNotFoundError(f"required staged diagnostic module absent: {file}")
        name = f"{package}.{stem}"
        spec = importlib.util.spec_from_file_location(name, file)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load staged diagnostic module: {file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{package}.shared_decoder_expert_execution"].shared_transformer_raw_outputs_expert128


def load_backbone_assembler(path: Path):
    """Load the explicitly staged source-reuse context assembler."""

    package = "research.daily_coarse_graining.daily_operator"
    helper_path = path.with_name("shared_backbone_projection_reuse.py")
    if not helper_path.is_file():
        raise FileNotFoundError(f"required staged backbone helper absent: {helper_path}")
    helper_name = f"{package}.shared_backbone_projection_reuse"
    helper_spec = importlib.util.spec_from_file_location(helper_name, helper_path)
    if helper_spec is None or helper_spec.loader is None:
        raise ValueError(f"cannot load staged backbone helper: {helper_path}")
    helper_module = importlib.util.module_from_spec(helper_spec)
    sys.modules[helper_name] = helper_module
    helper_spec.loader.exec_module(helper_module)

    name = f"{package}.shared_backbone_execution"
    if not path.is_file():
        raise FileNotFoundError(f"required staged backbone module absent: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load staged backbone module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    function = getattr(module, "assemble_shared_transformer_context_with_source_reuse", None)
    if not callable(function):
        raise ValueError(f"staged backbone module lacks source-reuse assembler: {path}")
    return function


@contextmanager
def candidate_process_head_patches(shared_process_head: Any, adapter: Any, backbone_assembler: Any = None):
    original_raw = shared_process_head.shared_transformer_raw_outputs
    original_assembler = shared_process_head.assemble_shared_transformer_context
    if original_raw is adapter:
        raise ValueError("diagnostic adapter already installed in production process-head module")
    if backbone_assembler is not None and original_assembler is backbone_assembler:
        raise ValueError("diagnostic backbone assembler already installed in production process-head module")
    try:
        shared_process_head.shared_transformer_raw_outputs = adapter
        if backbone_assembler is not None:
            shared_process_head.assemble_shared_transformer_context = backbone_assembler
        yield
    finally:
        shared_process_head.shared_transformer_raw_outputs = original_raw
        shared_process_head.assemble_shared_transformer_context = original_assembler


def lower_and_compile_candidate(backward: Any, parameters: Any, daily4: Any, emit, patches: Any):
    with patches:
        began = time.perf_counter()
        lowered = backward.lower(parameters, daily4)
        lower_seconds = time.perf_counter() - began
        emit("candidate_before_backend_compile", lower_seconds=lower_seconds)
        began = time.perf_counter()
        candidate = lowered.compile()
        compile_seconds = time.perf_counter() - began
    return candidate, lower_seconds, compile_seconds


def load_result_artifact(path: Path, expected_identity: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{role} result artifact not found: {path}")
    with path.open("rb") as stream:
        saved = pickle.load(stream)
    if not isinstance(saved, Mapping) or saved.get("identity") != expected_identity:
        raise ValueError(f"{role} result artifact method/checkpoint/input identity differs")
    return dict(saved)


def load_expert128_comparison(path: Path, expected_identity: Mapping[str, Any]) -> dict[str, Any]:
    saved = load_result_artifact(path, expected_identity, role="expert128 comparison")
    if saved.get("variant", "expert128") != "expert128":
        raise ValueError("comparison result is not the expert128 variant")
    if any(hard_counts(saved["result"][0][1]).values()):
        raise ValueError("expert128 comparison result hard gate failed")
    return saved


def check_artifact_output_collisions(inputs: tuple[Path | None, ...], outputs: tuple[Path, ...]) -> None:
    output_paths = {path.resolve() for path in outputs}
    collisions = [str(path) for path in inputs if path is not None and path.resolve() in output_paths]
    if collisions:
        raise ValueError(f"refusing to overwrite input result artifact(s): {', '.join(collisions)}")


def compare_trees(jax: Any, actual: Any, reference: Any) -> dict[str, Any]:
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    reference_leaves, reference_tree = jax.tree_util.tree_flatten(reference)
    if actual_tree != reference_tree or len(actual_leaves) != len(reference_leaves):
        raise ValueError("candidate and reference result trees differ")
    coordinates = failed = 0
    max_abs = sum_squared_difference = sum_squared_reference = 0.0
    nonfinite_actual = nonfinite_reference = 0
    for got, want in zip(actual_leaves, reference_leaves, strict=True):
        a, b = np.asarray(got), np.asarray(want)
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("candidate and reference result leaf shape/dtype differ")
        finite_a, finite_b = np.isfinite(a), np.isfinite(b)
        both_finite = finite_a & finite_b
        nonfinite_actual += int(np.count_nonzero(~finite_a))
        nonfinite_reference += int(np.count_nonzero(~finite_b))
        failed += int(np.count_nonzero(~(both_finite & np.isclose(a, b, rtol=RTOL, atol=ATOL))))
        coordinates += b.size
        a64 = a[both_finite].astype(np.float64)
        b64 = b[both_finite].astype(np.float64)
        delta = a64 - b64
        if delta.size:
            max_abs = max(max_abs, float(np.max(np.abs(delta))))
        sum_squared_difference += float(np.sum(delta * delta))
        sum_squared_reference += float(np.sum(b64 * b64))
    relative_l2 = float(np.sqrt(sum_squared_difference / max(sum_squared_reference, 1e-300)))
    valid_error_metrics = (
        not (nonfinite_actual or nonfinite_reference) and np.isfinite(max_abs) and np.isfinite(relative_l2)
    )
    return {
        "passed": failed == 0,
        "coordinates": coordinates,
        "failed": failed,
        "nonfinite_actual": nonfinite_actual,
        "nonfinite_reference": nonfinite_reference,
        "max_abs": max_abs if valid_error_metrics else None,
        "relative_l2": relative_l2 if valid_error_metrics else None,
    }


def hard_counts(auxiliary: Mapping[str, Any]) -> dict[str, int]:
    hard = auxiliary.get("hard")
    if not isinstance(hard, Mapping):
        raise ValueError("canonical daily auxiliary lacks hard-count mapping")
    return {name: int(np.asarray(value)) for name, value in hard.items()}


def benchmark(executable: Any, parameters: Any, daily4: Any, repeats: int) -> tuple[Any, list[float]]:
    import jax

    result = executable(parameters, daily4)
    jax.block_until_ready(result)
    times = []
    for _ in range(repeats):
        began = time.perf_counter()
        timed = executable(parameters, daily4)
        jax.block_until_ready(timed)
        times.append(time.perf_counter() - began)
    return result, times


def gpu_run(args: Any, emit) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from jax.experimental.serialize_executable import deserialize_and_load, serialize

    from research.daily_coarse_graining.daily_operator import shared_process_head
    from research.daily_coarse_graining.daily_operator.shared_recurrent_training import (
        make_shared_joint_daily_objective,
    )
    from scripts.hpc.probe_shared_compile_cache import _require_supported_jax
    from scripts.hpc.run_shared_learning import load_learning_checkpoint
    from scripts.hpc.run_shared_transformer_admission import memory_sizes
    from scripts.hpc.shared_daily_microbatch import make_microbatch_objective

    if args.require_gpu and (len(jax.local_devices()) != 1 or jax.local_devices()[0].platform != "gpu"):
        raise RuntimeError("select exactly one visible GPU for saved executable compatibility")
    _require_supported_jax(jax)
    profile, reference = trusted_profile(args)
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if current_head != reference["source_git_head"]:
        raise ValueError("GPU replay requires original immutable source checkout")
    with args.prepared_context.open("rb") as stream:
        context = pickle.load(stream)
    artifact_id = int(reference["method"]["training_plan"]["training_artifact_ids"][0])
    expected_identity = context_identity(reference, initial_checkpoint(reference), artifact_id)
    if context.get("identity") != expected_identity:
        raise ValueError("prepared context method/checkpoint/artifact identity differs")
    expected_executable = Path(profile["executables"]["daily"]["executable"]).resolve()
    if Path(context["daily_executable"]).resolve() != expected_executable:
        raise ValueError("prepared context daily executable differs from trusted profile")
    checkpoint = load_learning_checkpoint(
        Path(expected_identity["initial_checkpoint"]), expected_method=reference["method"]
    )
    parameters = jax.device_put(checkpoint["parameters"])
    parameter_count = sum(np.size(value) for value in jax.tree_util.tree_leaves(parameters))
    if parameter_count != int(reference["method"]["daily_protocol"]["architecture"]["parameter_count"]):
        raise ValueError("update-0 parameter tree count differs from declared architecture")
    daily4 = jax.device_put(context["daily4"])
    limit = int(reference["method"]["configuration"]["gradient_memory_limit_gib"]) * 2**30
    default_reference_path = args.output.with_suffix(".reference.pkl")
    reference_path = args.reference_result or default_reference_path
    candidate_path = args.output.with_suffix(".candidate.pkl")
    candidate_executable_path = args.output.with_suffix(".candidate.executable.pkl")
    result_identity = expected_identity | {"jax_version": jax.__version__, "executable": str(expected_executable)}
    output_paths = [args.output, candidate_path, candidate_executable_path]
    if args.reference_result is None and not args.reuse_reference:
        output_paths.append(default_reference_path)
    check_artifact_output_collisions((args.reference_result, args.comparison_result), tuple(output_paths))
    if args.reference_result is not None:
        saved = load_result_artifact(reference_path, result_identity, role="canonical reference")
        reference_result = saved["result"]
        reference_times = saved.get("warm_seconds", [])
        emit("explicit_daily_reference_reused", reference_artifact=str(reference_path))
    elif args.reuse_reference:
        saved = load_result_artifact(reference_path, result_identity, role="canonical reference")
        reference_result = saved["result"]
        reference_times = saved.get("warm_seconds", [])
        emit("trusted_daily_reference_reused", reference_artifact=str(reference_path))
    else:
        emit("load_trusted_daily_executable", executable=str(expected_executable))
        with expected_executable.open("rb") as stream:
            serialized = pickle.load(stream)
        began = time.perf_counter()
        trusted = deserialize_and_load(*serialized)
        load_seconds = time.perf_counter() - began
        del serialized
        reference_memory = memory_sizes(trusted)
        if reference_memory["estimated_live_bytes"] > limit:
            raise ValueError("trusted daily executable exceeds declared memory bound")
        emit("trusted_daily_executable_loaded", deserialize_seconds=load_seconds, reference_memory=reference_memory)
        result, reference_times = benchmark(trusted, parameters, daily4, args.repeats)
        reference_result = jax.device_get(result)
        reference_hard = hard_counts(reference_result[0][1])
        if any(reference_hard.values()):
            raise ValueError(f"trusted daily reference hard gate failed: {reference_hard}")
        emit("save_trusted_daily_reference", hard_counts=reference_hard)
        atomic_pickle(
            reference_path,
            {
                "identity": result_identity,
                "variant": "canonical_reference",
                "result": reference_result,
                "warm_seconds": reference_times,
            },
        )
        del trusted, result
    if any(hard_counts(reference_result[0][1]).values()):
        raise ValueError("saved daily reference hard gate failed")

    comparison_saved = None
    if args.comparison_result is not None:
        comparison_saved = load_expert128_comparison(args.comparison_result, result_identity)

    adapter = load_staged_adapter(args.candidate_module_file)
    backbone_assembler = (
        load_backbone_assembler(args.backbone_module_file) if args.backbone_module_file is not None else None
    )
    variant = "expert128+backbone_source_reuse" if backbone_assembler is not None else "expert128"
    daily_objective = make_shared_joint_daily_objective(
        context["runtime"],
        context["head"],
        context["template"],
        context["state_scales"],
        context["science_scales"],
    )
    microbatch_objective = make_microbatch_objective(jax, jnp, daily_objective)
    backward = jax.jit(jax.value_and_grad(microbatch_objective, has_aux=True))
    emit("candidate_before_lower")
    candidate, lower_seconds, compile_seconds = lower_and_compile_candidate(
        backward,
        parameters,
        daily4,
        emit,
        candidate_process_head_patches(shared_process_head, adapter, backbone_assembler),
    )
    candidate_memory = memory_sizes(candidate)
    emit("candidate_compiled", compile_seconds=compile_seconds, candidate_memory=candidate_memory)
    if candidate_memory["estimated_live_bytes"] > limit:
        raise ValueError("candidate daily gradient exceeds declared memory bound")
    emit("save_candidate_executable")
    atomic_pickle(
        candidate_executable_path,
        serialize(candidate),
    )
    result, candidate_times = benchmark(candidate, parameters, daily4, args.repeats)
    candidate_result = jax.device_get(result)
    emit("save_candidate_result")
    atomic_pickle(
        candidate_path,
        {
            "identity": result_identity,
            "variant": variant,
            "result": candidate_result,
            "warm_seconds": candidate_times,
        },
    )
    reference_hard = hard_counts(reference_result[0][1])
    candidate_hard = hard_counts(candidate_result[0][1])
    comparisons = {
        "loss": compare_trees(jax, candidate_result[0][0], reference_result[0][0]),
        "auxiliary": compare_trees(jax, candidate_result[0][1], reference_result[0][1]),
        "all_parameter_gradients": compare_trees(jax, candidate_result[1], reference_result[1]),
    }
    passed = all(value["passed"] for value in comparisons.values()) and not any(candidate_hard.values())
    comparison_vs_expert128 = None
    if comparison_saved is not None:
        expert128_result = comparison_saved["result"]
        comparison_vs_expert128 = {
            "artifact": str(args.comparison_result),
            "legacy_variant_metadata": "variant" not in comparison_saved,
            "provenance": (
                "explicit_legacy_expert128_artifact_without_variant_tag"
                if "variant" not in comparison_saved
                else "explicit_expert128_artifact"
            ),
            "warm_seconds_separate_run": comparison_saved.get("warm_seconds", []),
            "candidate_warm_seconds_separate_run": candidate_times,
            "timing_note": "separate runs; not a matched fresh speed comparison",
            "comparisons": {
                "loss": compare_trees(jax, candidate_result[0][0], expert128_result[0][0]),
                "auxiliary": compare_trees(jax, candidate_result[0][1], expert128_result[0][1]),
                "all_parameter_gradients": compare_trees(jax, candidate_result[1], expert128_result[1]),
            },
        }
    return {
        "status": "passed" if passed else "parity_failed",
        "scope": "actual_canonical_daily_micro4_only_no_T7_T30_no_optimizer",
        "all_parameter_gradient_coordinates": comparisons["all_parameter_gradients"]["coordinates"],
        "comparisons": comparisons,
        "variant": variant,
        "identity": result_identity,
        "comparison_vs_expert128": comparison_vs_expert128,
        "reference_hard": reference_hard,
        "candidate_hard": candidate_hard,
        "reference_warm_seconds": reference_times,
        "candidate_warm_seconds": candidate_times,
        "reference_artifact": str(reference_path),
        "candidate_artifact": str(candidate_path),
        "candidate_executable": str(candidate_executable_path),
        "candidate_memory": candidate_memory,
        "scientific_acceptance": False,
    }


def run(args: Any) -> dict[str, Any]:
    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "running",
        "scope": "actual_canonical_daily_micro4_only_no_T7_T30_no_optimizer",
        "scientific_acceptance": False,
        "rtol": RTOL,
        "atol": ATOL,
    }

    def emit(stage: str, **fields: Any) -> None:
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **fields)
        atomic_json(args.output, report)
        print(f"shared_expert_real_daily stage={stage} seconds={report['elapsed_seconds']:.1f}", flush=True)

    try:
        if args.prepare_only:
            emit("prepare_start")
            result = prepare(args, emit)
        else:
            if args.candidate_module_file is None:
                raise ValueError("GPU mode requires explicit --candidate-module-file")
            emit("gpu_start")
            result = gpu_run(args, emit)
        emit("completed", **result)
        return report
    except Exception as error:
        emit("exception", status="error", error_type=type(error).__name__, error=str(error))
        raise


def parser():
    from scripts.hpc.run_shared_learning import parser as learning_parser

    result = learning_parser()
    result.add_argument("--source-profile", type=Path, required=True)
    result.add_argument("--input-bundle", type=Path, required=True)
    result.add_argument("--prepared-context", type=Path, required=True)
    result.add_argument("--candidate-module-file", type=Path)
    result.add_argument("--backbone-module-file", type=Path)
    result.add_argument("--require-gpu", action="store_true")
    result.add_argument("--reuse-reference", action="store_true")
    result.add_argument("--reference-result", type=Path)
    result.add_argument("--comparison-result", type=Path)
    result.add_argument("--repeats", type=int, default=3)
    return result


if __name__ == "__main__":
    outcome = run(parser().parse_args())
    raise SystemExit(0 if outcome["status"] in {"passed", "prepared"} else 1)
