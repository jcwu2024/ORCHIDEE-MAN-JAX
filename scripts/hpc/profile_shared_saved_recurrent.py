"""Profile passed saved recurrent T7 executables without compiling another graph."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(os.environ.get("ORCHIDEE_REPO_ROOT", Path(__file__).resolve().parents[2]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_GRADIENT_COORDINATES = 64_884_481
RTOL = 1e-5
ATOL = 1e-6


def load_daily_probe(path: Path):
    name = "scripts.hpc._saved_profile_daily_probe"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load staged daily probe module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_admission(admission: Mapping[str, Any]) -> None:
    if admission.get("status") != "passed" or int(admission.get("horizon", -1)) != 7:
        raise ValueError("profiling requires the passed combined recurrent T7 admission")
    if int(admission.get("all_parameter_gradient_coordinates", -1)) != EXPECTED_GRADIENT_COORDINATES:
        raise ValueError("T7 admission gradient coordinate count differs from the saved profile scope")
    if float(admission.get("rtol", RTOL)) != RTOL or float(admission.get("atol", ATOL)) != ATOL:
        raise ValueError("T7 admission parity tolerances differ from the strict saved-executable gate")
    comparisons = admission.get("comparisons", {})
    required = ("loss", "auxiliary", "all_parameter_gradients")
    if any(name not in comparisons or not comparisons[name].get("passed") for name in required):
        raise ValueError("T7 admission is missing a passing loss/auxiliary/full-gradient comparison")
    for name in ("reference_hard", "candidate_hard"):
        hard = admission.get(name)
        if not isinstance(hard, Mapping) or any(int(value) != 0 for value in hard.values()):
            raise ValueError(f"T7 admission {name} physical gate is absent or nonzero")


def prepare_assets(args: Any, jax: Any, daily_probe: Any) -> dict[str, Any]:
    admission = json.loads(args.admission_report.read_text(encoding="utf-8"))
    validate_admission(admission)
    with args.prepared_context.open("rb") as stream:
        context = pickle.load(stream)
    with args.input_bundle.open("rb") as stream:
        bundle = pickle.load(stream)
    identity = context.get("identity")
    if not isinstance(identity, Mapping) or identity.get("method") is None:
        raise ValueError("saved daily context lacks method/checkpoint identity")
    method = identity["method"]
    artifact_ids = tuple(method["training_plan"]["training_artifact_ids"][:2])
    checkpoint_path = Path(identity["initial_checkpoint"]).resolve()
    if (
        identity.get("source_git_head") != bundle.get("source_git_head")
        or bundle.get("method") != method
        or tuple(bundle.get("artifact_ids", ())) != artifact_ids
        or int(identity.get("artifact_id", -1)) != int(artifact_ids[0])
        or Path(bundle.get("initial_checkpoint", "")).resolve() != checkpoint_path
        or len(bundle.get("payloads", ())) != 2
    ):
        raise ValueError("saved T7 inputs differ from the context method/checkpoint/artifact identity")
    case = bundle["payloads"][0]["cases"][7]
    if len(case) != 5 or np.shape(case.days[0].continuous_state)[0] != 7:
        raise ValueError("saved recurrent case is not the declared five-field T7 case")
    if np.shape(case.targets["state"])[0] != 7:
        raise ValueError("saved recurrent T7 targets have a different horizon")

    from scripts.hpc.run_shared_learning import load_learning_checkpoint

    checkpoint = load_learning_checkpoint(checkpoint_path, expected_method=method)
    if int(checkpoint.get("completed_updates", -1)) != 0:
        raise ValueError("saved recurrent profile requires the fresh update-0 checkpoint")
    parameters = jax.device_put(checkpoint["parameters"])
    expected_count = int(method["daily_protocol"]["architecture"]["parameter_count"])
    parameter_count = sum(np.size(value) for value in jax.tree_util.tree_leaves(parameters))
    if parameter_count != expected_count:
        raise ValueError("update-0 parameter tree differs from declared architecture")

    paths = {
        "original": Path(admission["executable"]).resolve(),
        "expert128_backbone_source_reuse": Path(admission["candidate_executable"]).resolve(),
    }
    if paths["original"] == paths["expert128_backbone_source_reuse"]:
        raise ValueError("admission original and candidate executable paths must differ")
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"saved {name} executable not found: {path}")

    output_paths = {args.output.resolve(), args.output.with_suffix(".traces").resolve()}
    input_paths = {
        args.admission_report.resolve(),
        args.input_bundle.resolve(),
        args.prepared_context.resolve(),
        checkpoint_path,
        *paths.values(),
    }
    if output_paths & input_paths:
        raise ValueError("profile report/trace output would overwrite a saved input asset")
    return {
        "admission": admission,
        "context": context,
        "identity": identity,
        "method": method,
        "checkpoint": checkpoint_path,
        "parameters": parameters,
        "parameter_count": parameter_count,
        "case": jax.device_put(case),
        "executables": paths,
    }


def all_finite(jax: Any, tree: Any) -> bool:
    return all(np.all(np.isfinite(np.asarray(value))) for value in jax.tree_util.tree_leaves(tree))


def run(args: Any) -> dict[str, Any]:
    import jax
    from jax.experimental.serialize_executable import deserialize_and_load

    from scripts.hpc.probe_shared_compile_cache import _require_supported_jax

    _require_supported_jax(jax)
    devices = tuple(jax.local_devices())
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("select exactly one visible GPU for saved executable compatibility")
    daily_probe_path = Path(__file__).with_name("probe_shared_expert_real_daily.py")
    if not daily_probe_path.is_file():
        raise FileNotFoundError(f"stage the matching daily probe beside the profiler: {daily_probe_path}")
    daily_probe = load_daily_probe(daily_probe_path)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "running",
        "scope": "saved_recurrent_T7_profile_no_gradient_compile_no_update",
        "horizon": 7,
        "variant_pair": ["original", "expert128_backbone_source_reuse"],
        "parameter_gradient_coordinates": EXPECTED_GRADIENT_COORDINATES,
        "rtol": RTOL,
        "atol": ATOL,
        "jax_version": jax.__version__,
        "device": str(devices[0]),
        "variants": {},
    }

    def emit(stage: str, **fields: Any) -> None:
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **fields)
        write_report(args.output, report)
        print(f"saved_recurrent_profile stage={stage} seconds={report['elapsed_seconds']:.1f}", flush=True)

    try:
        assets = prepare_assets(args, jax, daily_probe)
        report.update(
            identity=assets["identity"],
            checkpoint=str(assets["checkpoint"]),
            parameter_count=assets["parameter_count"],
            admission_report=str(args.admission_report),
            input_bundle=str(args.input_bundle),
            prepared_context=str(args.prepared_context),
        )
        parameters, case = assets["parameters"], assets["case"]
        host_results = {}
        for name, path in assets["executables"].items():
            emit(f"load_{name}", executable=str(path))
            with path.open("rb") as stream:
                serialized = pickle.load(stream)
            began = time.perf_counter()
            executable = deserialize_and_load(*serialized)
            load_seconds = time.perf_counter() - began
            del serialized
            metadata = {"load_seconds": load_seconds, "executable": str(path)}

            evaluated = executable(parameters, *case)
            jax.block_until_ready(evaluated)
            host_result = jax.device_get(evaluated)
            (loss, auxiliary), gradient = host_result
            hard = daily_probe.hard_counts(auxiliary)
            if not np.isfinite(np.asarray(loss)).all() or any(hard.values()):
                raise ValueError(f"saved {name} T7 loss/physical validation failed: {hard}")
            if not all_finite(jax, host_result):
                raise ValueError(f"saved {name} T7 result contains nonfinite auxiliary/gradient values")
            gradient_coordinates = sum(np.size(value) for value in jax.tree_util.tree_leaves(gradient))
            if gradient_coordinates != EXPECTED_GRADIENT_COORDINATES:
                raise ValueError(f"saved {name} T7 gradient coordinate count differs from admission")
            metadata.update(loss=float(np.asarray(loss)), hard_counts=hard, gradient_coordinates=gradient_coordinates)
            warm_seconds = []
            for _ in range(3):
                began = time.perf_counter()
                timed = executable(parameters, *case)
                jax.block_until_ready(timed)
                warm_seconds.append(time.perf_counter() - began)
                del timed
            metadata["warm_seconds"] = warm_seconds
            trace_dir = args.output.with_suffix(".traces") / name
            with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
                with jax.profiler.TraceAnnotation(f"saved_recurrent_T7_{name}"):
                    traced = executable(parameters, *case)
                    jax.block_until_ready(traced)
            del traced
            metadata["trace_directory"] = str(trace_dir)
            report["variants"][name] = metadata
            host_results[name] = host_result
            del evaluated, executable, gradient, auxiliary, loss
            emit(f"trace_{name}_done")

        original, candidate = host_results["original"], host_results["expert128_backbone_source_reuse"]
        comparisons = {
            "loss": daily_probe.compare_trees(jax, candidate[0][0], original[0][0]),
            "auxiliary": daily_probe.compare_trees(jax, candidate[0][1], original[0][1]),
            "all_parameter_gradients": daily_probe.compare_trees(jax, candidate[1], original[1]),
        }
        parity_passed = all(value["passed"] for value in comparisons.values())
        report["comparisons"] = comparisons
        report["strict_reference_parity_passed"] = parity_passed
        report["sum_warm_seconds_are_separate_runs"] = True
        report["trace_limitation"] = (
            "Use device event categories and concurrent interval summaries; overlapping summed kernel times "
            "are not exclusive module costs. No optimized HLO text was exported."
        )
        report["status"] = "passed" if parity_passed else "parity_failed"
        emit("completed")
        return report
    except Exception as error:
        emit("exception", status="error", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-report", type=Path, required=True)
    parser.add_argument("--input-bundle", type=Path, required=True)
    parser.add_argument("--prepared-context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    raise SystemExit(0 if result["status"] == "passed" else 1)
