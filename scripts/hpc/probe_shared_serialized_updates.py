"""Replay two unchanged technical updates using trusted saved executables.

No model graph is lowered here. The original learning runner still owns data,
fresh initialization, physical checks, gradient composition and AdamW updates.
This bounded probe requires the same source checkout and method as its saved
reference; it is not a general cross-version executable cache.
"""
from __future__ import annotations

import json
import hashlib
import pickle
import time
from pathlib import Path

import numpy as np

from scripts.hpc.probe_shared_execution import compare_trees
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json


def require_reference(profile, reference):
    if profile.get("status") != "passed" or set(profile.get("executables", {})) != {"daily", "T7", "T30"}:
        raise ValueError("requires a completed three-executable profile")
    if reference.get("status") != "passed" or reference.get("completed_updates") != 2:
        raise ValueError("requires a completed two-update reference")
    if reference.get("technical_updates") != 2:
        raise ValueError("reference must be a technical run, not formal learning")


def require_same_method(current, reference):
    if current["method"] != reference["method"]:
        raise ValueError("saved executables belong to a different method/data/initialization")
    if current["source_git_head"] != reference["source_git_head"]:
        raise ValueError("bounded serialized replay requires the original source checkout")


def admission_report_paths(args):
    paths = {name: getattr(args, attribute, None) for name, attribute in (
        ("daily", "daily_admission_report"), ("T7", "seven_admission_report"),
        ("T30", "thirty_admission_report"))}
    if any(paths.values()) and not all(paths.values()):
        raise ValueError("combined overrides require all three admission reports")
    return paths if all(paths.values()) else {}


def add_admission_arguments(parser):
    for flag in ("daily", "seven", "thirty"):
        parser.add_argument(f"--{flag}-admission-report", type=Path)


def validate_admissions(profile, reference, report_paths, *, profile_sha256):
    """Bind admitted binaries to the original source, method and fresh inputs."""
    if not report_paths:
        return {}
    if set(report_paths) != {"daily", "T7", "T30"}:
        raise ValueError("combined overrides require daily, T7 and T30")
    checkpoints = [row["path"] for row in reference["checkpoints"]
                   if row["update"] == 0 and row["retained_milestone"]]
    if len(checkpoints) != 1:
        raise ValueError("admissions require exactly one retained update-0 checkpoint")
    base = {"schema_version": "shared_expert_real_daily_context_v1",
            "source_git_head": reference["source_git_head"], "method": reference["method"],
            "initial_checkpoint": str(Path(checkpoints[0]).resolve()),
            "artifact_id": int(reference["method"]["training_plan"]["training_artifact_ids"][0]),
            "daily_rows": 4}
    admissions = {}
    for name, report_path in report_paths.items():
        raw = Path(report_path).read_bytes()
        report = json.loads(raw)
        scope = ("actual_canonical_daily_micro4_only_no_T7_T30_no_optimizer" if name == "daily"
                 else "actual_canonical_recurrent_single_horizon_no_daily_no_optimizer")
        if (report.get("status") != "passed" or report.get("scope") != scope
                or report.get("scientific_acceptance") is not False
                or report.get("rtol") != 1e-5 or report.get("atol") != 1e-6):
            raise ValueError(f"{name} admission status/scope/tolerance differs")
        comparisons = report.get("comparisons", {})
        if (set(comparisons) != {"loss", "auxiliary", "all_parameter_gradients"}
                or any(row.get("passed") is not True or row.get("failed") != 0
                       or row.get("nonfinite_actual") != 0 or row.get("nonfinite_reference") != 0
                       for row in comparisons.values())
                or comparisons["all_parameter_gradients"].get("coordinates") != 64884481
                or report.get("all_parameter_gradient_coordinates") != 64884481
                or not report.get("reference_hard") or not report.get("candidate_hard")
                or any(report["reference_hard"].values()) or any(report["candidate_hard"].values())):
            raise ValueError(f"{name} admission gradient/physical gates failed")
        if name == "daily":
            if report.get("variant") != "expert128+backbone_source_reuse":
                raise ValueError("daily admission is not the combined variant")
            identity = report.get("identity", {})
        else:
            if report.get("horizon") != int(name[1:]) or report.get("artifact_id") != base["artifact_id"]:
                raise ValueError(f"{name} admission horizon/artifact differs")
            # Recurrent reports predate inline identities; their preserved result
            # carries the exact method/profile/checkpoint identity instead.
            with Path(report["candidate_artifact"]).open("rb") as handle:
                saved = pickle.load(handle)
            identity = saved.get("identity", {})
            del saved
            if (identity.get("horizon") != int(name[1:])
                    or identity.get("source_profile_sha256") != profile_sha256):
                raise ValueError(f"{name} admission profile/horizon identity differs")
        original = str(Path(profile["executables"][name]["executable"]).resolve())
        if (any(identity.get(key) != value for key, value in base.items())
                or identity.get("jax_version") != "0.4.38"
                or str(Path(identity.get("executable", "")).resolve()) != original):
            raise ValueError(f"{name} admission original method/checkpoint/executable differs")
        path = Path(report["candidate_executable"])
        if not path.is_file():
            raise FileNotFoundError(f"admitted {name} binary absent: {path}")
        admissions[name] = {"executable": str(path), "admission_report": str(report_path),
                            "admission_report_sha256": hashlib.sha256(raw).hexdigest(),
                            "original_executable": original, "identity": identity}
    return admissions


def load_evaluators(profile, *, deserialize, memory_sizes, memory_limit_bytes, emit,
                    overrides=None, components=("daily", "T7", "T30")):
    """Load one existing executable per component, without a compile fallback."""
    evaluators, metadata = {}, {}
    for name in components:
        emit("load_saved_gradient", component=name)
        began = time.perf_counter()
        cpu_began = time.process_time()
        selected = (overrides or {}).get(name, profile["executables"][name])
        path = Path(selected["executable"])
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        read_seconds = time.perf_counter() - began
        read_cpu_seconds = time.process_time() - cpu_began
        emit("deserialize_saved_gradient", component=name,
             executable_read_seconds=read_seconds,
             executable_read_cpu_seconds=read_cpu_seconds)
        began = time.perf_counter()
        cpu_began = time.process_time()
        executable = deserialize(*payload)
        load_seconds = time.perf_counter() - began
        load_cpu_seconds = time.process_time() - cpu_began
        del payload
        memory = memory_sizes(executable)
        if memory["estimated_live_bytes"] > memory_limit_bytes:
            raise ValueError(f"saved {name} executable exceeds declared memory limit")
        evaluators[name] = executable
        metadata[name] = {
            "source_executable": str(path),
            "read_seconds": read_seconds,
            "read_cpu_seconds": read_cpu_seconds,
            "deserialize_seconds": load_seconds,
            "deserialize_cpu_seconds": load_cpu_seconds,
            "memory": memory,
            "cache_policy": "trusted_serialized_executable_no_trace_or_lower",
            "admission": (overrides or {}).get(name),
        }
        emit("saved_gradient_loaded", component=name, compiled_gradient_graphs=metadata)
    return evaluators, metadata


def numerical_update_rows(outcome):
    return [
        {key: row[key] for key in (
            "update", "artifact_id", "loss", "gradient_norms", "optimizer_gradient_norm", "clip_scale"
        )}
        for row in outcome["updates"]
    ]


def compare_update_reports(jax, current, reference, *, rtol, atol):
    # JSON does not retain NumPy scalar dtypes. Compare scalar record values
    # in one host dtype, while checkpoint tensor comparisons remain strict.
    def normalize(rows):
        return jax.tree_util.tree_map(lambda value: np.asarray(value, dtype=np.float64), rows)
    return compare_trees(
        jax, normalize(numerical_update_rows(current)), normalize(numerical_update_rows(reference)),
        rtol=rtol, atol=atol,
    )


def run(args):
    from scripts.hpc import run_shared_learning as learning
    from scripts.hpc.probe_shared_compile_cache import _require_supported_jax

    if args.technical_updates != 2 or args.prepare_only or args.resume_checkpoint:
        raise ValueError("serialized replay requires fresh --technical-updates 2")
    profile = json.loads(args.source_profile.read_text(encoding="utf-8"))
    reference = json.loads(Path(profile["technical_update_report"]).read_text(encoding="utf-8"))
    require_reference(profile, reference)
    admissions = validate_admissions(profile, reference, admission_report_paths(args),
                                     profile_sha256=hashlib.sha256(args.source_profile.read_bytes()).hexdigest())
    output = args.output
    update_output = output.with_suffix(".updates.json")
    if update_output.resolve() == Path(profile["technical_update_report"]).resolve():
        raise ValueError("replay output must preserve the reference report/checkpoints")
    report = {
        "status": "running", "scope": "same_method_serialized_two_update_replay",
        "source_profile": str(args.source_profile), "scientific_acceptance": False,
        "comparison_rtol": 1e-5, "comparison_atol": 1e-6,
        "comparison_tolerance_source": "shared_execution_benchmark_v1.json",
        "admitted_overrides": admissions,
    }
    started = time.perf_counter()
    original_compile = learning.compile_gradient_evaluators
    context = {}

    def emit(stage, **values):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **values)
        _atomic_json(output, report)
        print(f"serialized_updates stage={stage} seconds={report['elapsed_seconds']:.1f}", flush=True)

    def load_instead_of_compile(**kwargs):
        jax = kwargs["jax"]
        _require_supported_jax(jax)
        if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
            raise ValueError("select one visible GPU matching saved executable layout")
        if kwargs["daily_microbatch_size"] != 4:
            raise ValueError("saved profile is daily micro4")
        current = json.loads(update_output.read_text(encoding="utf-8"))
        require_same_method(current, reference)
        from jax.experimental.serialize_executable import deserialize_and_load

        context["jax"] = jax
        evaluators, metadata = load_evaluators(
            profile, deserialize=deserialize_and_load, memory_sizes=learning.memory_sizes,
            memory_limit_bytes=kwargs["memory_limit_bytes"], emit=emit,
            overrides=admissions,
        )
        emit("gradients_ready", gradient_ready_seconds=time.perf_counter() - started)
        return evaluators, metadata

    learning.compile_gradient_evaluators = load_instead_of_compile
    args.output = update_output
    try:
        emit("run_unchanged_training")
        outcome = learning.run(args)
        report["technical_update_report"] = str(update_output)
        if outcome["status"] != "passed":
            emit("rejected", status="rejected", reason="technical_updates_failed")
            return report
        emit("compare_reference", update_run_seconds=time.perf_counter() - started)
        jax = context["jax"]
        saved = learning.load_learning_checkpoint(Path(reference["checkpoint"]), expected_method=reference["method"])
        replay = learning.load_learning_checkpoint(Path(outcome["checkpoint"]), expected_method=reference["method"])
        tolerance = dict(rtol=report["comparison_rtol"], atol=report["comparison_atol"])
        comparisons = {
            "update_losses_gradient_norms": compare_update_reports(jax, outcome, reference, **tolerance),
            "all_parameters": compare_trees(jax, replay["parameters"], saved["parameters"], **tolerance),
            "full_adamw_state": compare_trees(jax, replay["optimizer"], saved["optimizer"], **tolerance),
        }
        emit(
            "completed", status="passed" if all(row["passed"] for row in comparisons.values()) else "rejected",
            comparisons=comparisons, completed_updates=outcome["completed_updates"],
            warm_second_update_wall_seconds=outcome["warm_second_update_wall_seconds"],
            device_peak_bytes_in_use=outcome["device_peak_bytes_in_use"],
            checkpoint_selection="diagnostic_replay_not_formal_training",
        )
        return report
    except Exception as error:
        emit("exception", status="error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        learning.compile_gradient_evaluators = original_compile
        args.output = output


def parser():
    from scripts.hpc.run_shared_learning import parser as base_parser

    result = base_parser()
    result.add_argument("--source-profile", type=Path, required=True)
    add_admission_arguments(result)
    return result


if __name__ == "__main__":
    result = run(parser().parse_args())
    raise SystemExit(0 if result["status"] == "passed" else 1)
