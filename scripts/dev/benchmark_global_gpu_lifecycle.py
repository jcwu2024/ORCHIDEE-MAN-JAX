"""Dedicated B1024-only, fresh-process lifecycle diagnostic for the declared new science.

Dry runs and module import do not import JAX, read prepared/reference pickles,
query a GPU, or execute a physical model. Real execution requires separate job
authorization and the canonical GPU entry.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.dev import benchmark_global_gpu_capacity as capacity  # noqa: E402
from scripts.dev import global_lifecycle_science as science  # noqa: E402
from scripts.dev.global_batch_stage_store import StageStore  # noqa: E402
from scripts.dev.validate_global_gpu_lifecycle import (  # noqa: E402
    compare_reference_lanes,
    native_schema,
    validate_stage_receipt,
)

CONTRACT, PROCESS_STRATEGY, STAGE_NAME = science.CONTRACT, science.PROCESS_STRATEGY, science.STAGE_NAME
HISTORY_MANIFEST_SHA256 = "4789e03be463f8c2f559b9d7a319cd1850462836a7ba904d4396ec83797bfd32"
HISTORY_REPORT_SHA256 = "0bc42a9d6a1f5d76168937bfe4feb2b1b6d10846c839a23b5e2c57342d6a57e7"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-input", "full-reference-dir", "memory-baseline-dir", "b1024-evidence-dir", "output-dir"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--config", type=Path, default=ROOT / "configs/orchidee_man_250919.yaml")
    result.add_argument("--asset-root", type=Path)
    result.add_argument("--warm-repeats", type=int, default=5)
    result.add_argument("--budget-seconds", type=int, default=7200)
    result.add_argument("--stage-shape", type=int, choices=(1024,), required=True)
    result.add_argument("--measurement-strategy", choices=science.STRATEGIES, required=True)
    result.add_argument("--expected-code-sha", required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def validate_args(args):
    if args.warm_repeats != 5 or args.budget_seconds != 7200 or args.stage_shape != 1024:
        raise ValueError("only fixed B1024/2048 lanes/five repeats/7200s lifecycle contract admitted")
    science.measurement_fields(args.measurement_strategy)
    args.baseline_ref = science.BASELINE_REF
    for source in (
        args.full_reference_dir,
        args.memory_baseline_dir,
        args.b1024_evidence_dir,
        args.prepared_input.parent,
    ):
        StageStore.validate_source_destination(source, args.output_dir)


def historical_memory_admission(args, identity):
    """Freeze historical metadata hashes and stream the checkpoint; never load old arrays."""
    directory = args.b1024_evidence_dir
    manifest_path, report_path = directory / "stage_manifest.json", directory / "report.json"
    if (
        science.digest_file(manifest_path)[0] != HISTORY_MANIFEST_SHA256
        or science.digest_file(report_path)[0] != HISTORY_REPORT_SHA256
    ):
        raise ValueError("historical B1024 immutable metadata hashes differ")
    manifest, report = json.loads(manifest_path.read_text()), json.loads(report_path.read_text())
    stage_name = "capacity_B1024_fixed1024_v1"
    stage = manifest["stages"][stage_name]
    old = manifest["identity"]
    for key, expected in {
        "science_sha256": science.OLD_SCIENCE_SHA256,
        "lanes": identity["lanes"],
        "gate": science.GATE,
        "precision_policy": science.PRECISION_POLICY,
        "numerical_reference_sha256": science.REFERENCE_SHA256,
    }.items():
        if old.get(key) != expected:
            raise ValueError(f"historical memory evidence identity differs: {key}")
    path = directory / f"stage_{stage_name}.pkl"
    if science.digest_file(path) != (capacity.B1024_SHA256, capacity.B1024_BYTES):
        raise ValueError("historical B1024 immutable checkpoint hash/bytes differ")
    analysis = stage.get("provenance", {}).get("executable_memory_analysis", {})
    if (
        stage.get("status") != "complete"
        or stage.get("sha256") != capacity.B1024_SHA256
        or report.get("stages", {}).get(stage_name) != stage
        or analysis.get("available") is not True
        or type(analysis.get("temp_size_in_bytes")) is not int
        or analysis["temp_size_in_bytes"] < 0
    ):
        raise ValueError("historical B1024 sealed compiler memory metadata incomplete")
    # B64 metadata is admitted separately but cannot replace the frozen B1024
    # temp/RSS bounds or influence the fixed2048 preallocation budget.
    baseline_manifest, baseline_stage, baseline_analysis = capacity.baseline_memory(args.memory_baseline_dir)
    for key in ("lanes", "science_sha256", "baseline_ref", "precision_policy", "gate"):
        if baseline_manifest["identity"].get(key) != old.get(key):
            raise ValueError(f"B64 historical memory source mismatch: {key}")
    return analysis.copy(), {
        "path": str(path),
        "sha256": capacity.B1024_SHA256,
        "bytes": capacity.B1024_BYTES,
        "manifest_sha256": HISTORY_MANIFEST_SHA256,
        "report_sha256": HISTORY_REPORT_SHA256,
        "role": "immutable compiler-temp/RSS admission only; no historical arrays/timings loaded",
        "B64_memory_origin": {
            "directory": str(args.memory_baseline_dir),
            "stage_sha256": baseline_stage["sha256"],
            "analysis": baseline_analysis,
        },
    }


def runtime_admission(scale, environment, resources):
    if (
        scale.jax.__version__ != "0.4.38"
        or len(scale.jax.devices()) != 1
        or scale.jax.devices()[0].platform != "gpu"
        or "V100" not in scale.jax.devices()[0].device_kind
        or not scale.jax.config.jax_enable_x64
        or resources.get("allocated_CPU_request_count") != 4
    ):
        raise ValueError("canonical single V100/JAX0.4.38/x64/four auxiliary CPUs required")
    if (
        os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR") not in (None, "")
        or os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() != "false"
    ):
        raise ValueError("unchanged default allocator with PREALLOCATE=false required")
    physical = scale.visible_gpu_memory()
    if (
        not physical.get("applicable")
        or physical.get("uuid") != environment.get("physical_gpu", {}).get("uuid")
        or physical.get("total_bytes", 0) < 30 * 1024**3
    ):
        raise ValueError("actual launcher-selected GPU UUID/total snapshot required")
    environment["physical_gpu"] = {**environment["physical_gpu"], "total_bytes": physical["total_bytes"]}
    environment["env"]["XLA_PYTHON_CLIENT_ALLOCATOR"] = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR")
    return physical


def build_identity(args, lanes, second, environment, resources, physical, source, origins, base):
    runtime_keys = ("python", "jax", "jaxlib", "numpy", "device_details", "x64_enabled", "backend_platform_version")
    compiler_keys = ("XLA_FLAGS", "JAX_PLATFORMS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    window = float(os.environ["ORCHJAX_GPU_WINDOW_START_EPOCH"])
    if not math.isfinite(window) or window <= 0:
        raise ValueError("positive finite shared allocation epoch required")
    return {
        **source,
        "contract": CONTRACT,
        "process_strategy": PROCESS_STRATEGY,
        "stage_name": STAGE_NAME,
        "stage_shape": 1024,
        "total_lanes": 2048,
        "candidate_shapes": [1024],
        "unique_real_points": 64,
        "warm_repeats": 5,
        "budget_seconds": 7200,
        "baseline_ref": science.BASELINE_REF,
        "lanes": [
            {"landpoint_id": lane.landpoint_id, "inputs_sha256": base.semantic_reference_identity(args, lane, other)}
            for lane, other in zip(lanes, second, strict=True)
        ],
        "lane_origins": origins,
        "runtime": {key: environment[key] for key in runtime_keys},
        "compiler_environment": {key: environment["env"].get(key) for key in compiler_keys},
        "gate": science.GATE,
        "precision_policy": science.PRECISION_POLICY,
        "comparison_reference": science.comparison_admission(),
        **science.measurement_fields(args.measurement_strategy),
        "run_binding": {
            "Slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": environment["hostname"],
            "physical_gpu": {key: physical[key] for key in ("uuid", "total_bytes")},
            "server_resources": capacity.stable_resource_identity(resources),
            "allocator": "default",
            "preallocate": False,
            "window_start_epoch": window,
        },
    }


def memory_observer(recorder, scale, selected):
    def observe(event, state):
        physical = scale.visible_gpu_memory()
        if not physical.get("applicable") or any(physical.get(key) != selected[key] for key in ("uuid", "total_bytes")):
            raise ValueError("lifecycle memory observation selected GPU UUID/total changed")
        record = {
            "event": event,
            **state,
            "host_RSS_bytes": scale.base.rss_bytes(),
            "physical_gpu": physical,
            "peak_unavailable": True,
            "scope": "logical roots and discrete selected-card snapshot outside warm timer",
        }
        recorder.result.setdefault("lifecycle_memory_observations", []).append(record)
        recorder.phase("lifecycle_memory_" + event, **record)

    return observe


class ValidationCapture:
    """Actual handoff arguments captured in initial calls; warm calls use original AOT."""

    def __init__(self, jax):
        self.jax, self.arrays, self.pending, self.first_state = jax, {}, [], None
        self.active = True
        self.materialization_seconds = 0.0

    def executable_factory(self, executable):
        def checked(*arguments):
            value = executable(*arguments)
            if self.active:
                if self.first_state is None:
                    self.first_state = value[0]
                else:
                    self.pending.append({"first_output_state": self.first_state, "second_input_state": arguments[0]})
                    self.first_state = None
            return value

        return checked

    def sink(self, event, host_tree):
        self.arrays[event] = host_tree
        if event == "initial_output":
            started = time.perf_counter()
            if self.first_state is not None or len(self.pending) != 2:
                raise ValueError("initial continuous-state trace requires exactly two complete two-block calls")
            self.arrays["handoffs"] = self.jax.device_get(self.pending)
            self.pending.clear()
            self.active = False
            self.materialization_seconds += time.perf_counter() - started


def restore_runtime_reference_containers(tree):
    """Restore the exact native NamedTuple class after runtime admission only.

    The CPU-only unpickler uses an inert DTO to avoid importing scientific/JAX
    modules. JAX treedef includes the class identity, so the runtime comparator
    needs the production container, with every array object left untouched.
    """
    from jax_orchidee.stomate.modelout import ModeloutResult

    def restore(value):
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            if type(value) not in {science.ModeloutResult, ModeloutResult} or value._fields != ModeloutResult._fields:
                raise ValueError("runtime reference contains undeclared NamedTuple schema")
            return ModeloutResult(*(restore(field) for field in value))
        if isinstance(value, dict):
            return {key: restore(field) for key, field in value.items()}
        if type(value) in (tuple, list):
            return type(value)(restore(field) for field in value)
        return value

    return restore(tree)


def execute_stage(args, recorder, lanes, second, scale, source):
    base = scale.base
    environment, resources = recorder.result["environment"], recorder.result["server_resources"]
    physical = runtime_admission(scale, environment, resources)
    expanded, others, origins = capacity.capacity_layout(lanes, second, 2048)
    identity = build_identity(args, lanes, second, environment, resources, physical, source, origins, base)
    science.validate_identity(identity, strategy=args.measurement_strategy, expected_code_sha=args.expected_code_sha)
    started = time.perf_counter()
    reference, reference_origin = science.load_comparison_reference(args.full_reference_dir, identity)
    reference = restore_runtime_reference_containers(reference)
    memory, historical_origin = historical_memory_admission(args, identity)
    recorder.result.update(
        identity=identity,
        reference_origin=reference_origin,
        historical_memory_origin=historical_origin,
        reference_memory_reload_seconds=time.perf_counter() - started,
        scientific_acceptance=False,
    )
    remaining = args.budget_seconds - (time.time() - identity["run_binding"]["window_start_epoch"])
    if remaining < 3600:
        raise TimeoutError("at least3600s allocation time remaining required before this arm")
    recorder.sentinel_paths = reference_origin["sentinel_paths"]
    if base.roughheight_sentinel_contract(lanes[0]) != recorder.sentinel_paths:
        raise ValueError("current source-derived bare-soil sentinel differs from frozen reference contract")
    store = StageStore(args.output_dir, identity)
    recorder.result["stages"] = store.manifest["stages"]
    if store.manifest["stages"]:
        raise ValueError("lifecycle diagnostics prohibit resume/reused stage or mixed process samples")
    capture = ValidationCapture(scale.jax)

    def validation_sink(event, host_tree):
        capture.sink(event, host_tree)
        if event == "initial_output":
            try:
                native_schema(host_tree, reference, 2048)
                compare_reference_lanes(host_tree, reference, recorder.sentinel_paths)
            except Exception as exc:
                # The old capacity gate permits finite exploratory mismatch.
                # This new entry is strict before any warm sample and preserves
                # the actual failed arrays independently of a complete stage.
                import pickle

                from scripts.dev.probe_global_batch_cache import atomic_bytes

                failed = args.output_dir / "failed_initial_arrays.pkl"
                atomic_bytes(
                    failed,
                    pickle.dumps({"identity": identity, "initial_output": host_tree}, protocol=pickle.HIGHEST_PROTOCOL),
                )
                recorder.result["initial_validation_failure"] = {
                    "path": str(failed),
                    "sha256": science.digest_file(failed)[0],
                    "message": str(exc),
                }
                recorder.flush()
                raise

    def operation():
        value, record = capacity.measure_shape(
            1024,
            args,
            recorder,
            expanded,
            others,
            reference,
            scale,
            {1024: memory},
            2048,
            memory_observer=memory_observer(recorder, scale, physical),
            validation_sink=validation_sink,
            validation_executable_factory=capture.executable_factory,
        )
        if record["strict_numerical_passed"] is not True:
            raise ValueError("new core failed strict frozen FULL64 comparison; no sealed timing admission")
        value["validation_arrays"] = capture.arrays
        record["provenance"].update(
            process_strategy=PROCESS_STRATEGY,
            full_output_native_schema=native_schema(value["outputs"], reference, 2048),
            continuous_state_binding=science.run_two_binding(ROOT),
            admission_compiler_memory_records={"1024": memory},
            memory_observations=recorder.result["lifecycle_memory_observations"],
        )
        record["startup"]["validation_materialization_seconds_outside_warm_timers"] = capture.materialization_seconds
        record["execution_origin"]["process_start_epoch"] = float(
            os.environ["ORCHJAX_GLOBAL_BATCH_PROCESS_START_EPOCH"]
        )
        return value, record

    (_value, _record), reused = store.run(STAGE_NAME, operation)
    if reused:
        raise ValueError("fresh lifecycle execution cannot reuse completed measurements")
    recorder.result.update(
        stages=store.manifest["stages"], completed_shapes=[1024], status="completed_lifecycle_diagnostic"
    )
    recorder.flush()
    receipt = validate_stage_receipt(
        args.output_dir,
        strategy=args.measurement_strategy,
        expected_code_sha=args.expected_code_sha,
        reference_dir=args.full_reference_dir,
    )
    recorder.result["array_acceptance_receipt"] = {
        key: receipt[key] for key in ("sha256", "bytes", "array_acceptance_path", "array_acceptance_sha256")
    }
    recorder.flush()
    return receipt


def main():
    args = parser().parse_args()
    validate_args(args)
    source = science.source_identity(ROOT, expected_code_sha=args.expected_code_sha, require_clean=not args.dry_run)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "contract": CONTRACT,
                    "process_strategy": PROCESS_STRATEGY,
                    "stage_name": STAGE_NAME,
                    "stage_shape": 1024,
                    "candidate_shapes": [1024],
                    "total_lanes": 2048,
                    "unique_real_points": 64,
                    "lane_days": 28672,
                    "warm_repeats": 5,
                    "budget_seconds": 7200,
                    **science.measurement_fields(args.measurement_strategy),
                    "execution_source_identity": source,
                    "comparison_reference": science.comparison_admission(),
                    "model_execution": False,
                    "JAX_imported": "jax" in sys.modules,
                    "scientific_acceptance": False,
                    "execution_requires_clean_checkout": True,
                    "historical_timings_reused": False,
                }
            )
        )
        return 0
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("fresh-process lifecycle arm requires a new empty output directory")
    from scripts.dev import benchmark_global_scale as scale
    from scripts.dev.probe_global_batch_cache import configure_cache

    scale.jax.config.update("jax_enable_x64", True)
    configure_cache("off", None)
    recorder = scale.base.Recorder(args.output_dir)
    recorder.result.update(
        environment=scale.base.metadata(args), server_resources=json.loads(os.environ["ORCHJAX_SERVER_RESOURCE_PLAN"])
    )
    previous = {}

    def stopped(number, frame):
        raise InterruptedError(f"signal{number}; partial lifecycle evidence retained")

    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.signal(number, stopped)
    code = 0
    try:
        started = time.perf_counter()
        lanes, second = scale.read_prepared(args.prepared_input, args.config)
        recorder.result["prepared_input_reload_seconds"] = time.perf_counter() - started
        recorder.result["input_compatibility_audit"] = scale.screen_inputs(lanes, second)
        execute_stage(args, recorder, lanes, second, scale, source)
    except (InterruptedError, KeyboardInterrupt) as exc:
        code = 2
        recorder.result.update(status="stopped", message=str(exc))
    except Exception as exc:
        code = 1
        recorder.result.update(status="failed", message=str(exc), traceback=traceback.format_exc())
    finally:
        recorder.stop.set()
        recorder.flush()
        for number, handler in previous.items():
            signal.signal(number, handler)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
