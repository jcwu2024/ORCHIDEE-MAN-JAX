"""CPU-only, sequential fresh-process capacity dispatch; never initialize JAX."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.dev.benchmark_global_gpu_capacity import (  # noqa: E402
    FIXED2048_CONTRACT,
    PROCESS_STRATEGY_2048,
    lane_metrics,
)


def digest_file(path):
    digest = hashlib.sha256()
    count = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            count += len(block)
    return digest.hexdigest(), count


def validate_stage_receipt(directory, shape, *, common_identity=None):
    """Strict CPU receipt validation without pickle, arrays, model or GPU imports."""
    directory = Path(directory)
    manifest_path, report_path = directory / "stage_manifest.json", directory / "report.json"
    manifest = json.loads(manifest_path.read_text())
    report = json.loads(report_path.read_text())
    identity = manifest["identity"]
    name = f"capacity_B{shape}_fixed2048_fresh_v1"
    stage = manifest.get("stages", {}).get(name, {})
    if (identity.get("contract") != FIXED2048_CONTRACT or identity.get("process_strategy") != PROCESS_STRATEGY_2048
            or identity.get("stage_shape") != shape or identity.get("total_lanes") != 2048
            or identity.get("candidate_shapes") != [1024, 2048] or identity.get("warm_repeats") != 5
            or identity.get("budget_seconds") != 7200):
        raise ValueError("fresh2048 stage identity mismatch")
    common = {key: value for key, value in identity.items() if key != "stage_shape"}
    if common_identity is not None and common != common_identity:
        raise ValueError("fresh2048 common input/science/runtime/job/node/GPU/allocator identity mismatch")
    gates = stage.get("gates", [])
    if (stage.get("status") != "complete" or stage.get("measurement_status") != "complete"
            or stage.get("strict_numerical_passed") is not True or len(gates) != 2051
            or any(g.get("passed") is not True for g in gates)):
        raise ValueError("fresh2048 strict complete2051 gates required")
    expected_names = [f"capacity_B{shape}_lane_{i}_source_{i % 64}_GPU_reference" for i in range(2048)]
    expected_names += [f"capacity_B{shape}_{suffix}" for suffix in ("frozen_reset_repeat", "permutation", "replacement_isolation")]
    if [g.get("name") for g in gates] != expected_names:
        raise ValueError("fresh2048 numerical gate order/name mismatch")
    timing = stage.get("timing", {})
    if set(timing) != {"core", "wrapper"}:
        raise ValueError("fresh2048 complete core/wrapper timing receipts required")
    for metric in timing.values():
        if metric != {**metric, **lane_metrics(metric["samples_seconds"], 2048)}:
            raise ValueError("fresh2048 five synchronized full-workload timing summaries mismatch")
    provenance = stage.get("provenance", {})
    for key, expected in {"total_lanes": 2048, "lane_days": 28672, "unique_real_points": 64,
        "chunk_size": shape, "chunks_per_collection": 2048 // shape, "blocks": [[2, 8], [9, 15]],
        "half_hours_per_day": 48, "full_outputs": True, "full_two_block_workload": True,
        "process_strategy": PROCESS_STRATEGY_2048}.items():
        if provenance.get(key) != expected:
            raise ValueError(f"fresh2048 workload mismatch: {key}")
    schema = provenance.get("full_output_native_schema", [])
    if len(schema) != 760 or any(item.get("shape", [None])[0] != 2048 or not item.get("dtype") for item in schema):
        raise ValueError("fresh2048 full760 native output schema required")
    memory = provenance.get("executable_memory_analysis", {})
    preflight = provenance.get("memory_preflight", {})
    if (memory.get("available") is not True or type(memory.get("temp_size_in_bytes")) is not int
            or memory["temp_size_in_bytes"] < 0 or type(provenance.get("host_peak_RSS_bytes")) is not int
            or provenance["host_peak_RSS_bytes"] <= 0 or any(type(preflight.get(key)) is not int or preflight[key] <= 0
                for key in ("resident_2048_input_bytes", "full_2048_output_bytes"))):
        raise ValueError("fresh2048 actual compiler/host-RSS/stack-byte receipt required")
    origin = stage.get("execution_origin", {})
    epoch = origin.get("process_start_epoch")
    if (type(origin.get("pid")) is not int or origin["pid"] <= 0 or type(epoch) not in (int, float)
            or not math.isfinite(epoch) or epoch <= 0):
        raise ValueError("fresh2048 actual worker PID/process-start origin required")
    path = directory / f"stage_{name}.pkl"
    if stage.get("path") != path.name:
        raise ValueError("fresh2048 sealed stage path mismatch")
    sha, count = digest_file(path)
    if stage.get("sha256") != sha or stage.get("checkpoint_io", {}).get("bytes") != count:
        raise ValueError("fresh2048 checkpoint full checksum/bytes mismatch")
    if (report.get("status") != "completed_exploratory_capacity_screen" or report.get("completed_shapes") != [shape]
            or report.get("stages", {}).get(name) != stage or report.get("environment") != origin.get("environment")):
        raise ValueError("fresh2048 child report/manifest/origin mismatch")
    return {"identity": identity, "common_identity": common, "stage": stage,
        "path": str(path), "sha256": sha, "bytes": count, "execution_origin": origin,
        "manifest_sha256": digest_file(manifest_path)[0], "report_sha256": digest_file(report_path)[0]}


def atomic_report(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def child_command(command, output_dir, shape):
    result = list(command)
    for flag in ("--stage-shape", "--control-stage-dir"):
        if flag in result:
            raise ValueError("coordinator owns child shape/control arguments")
    if "--fixed-2048" not in result or result.count("--output-dir") != 1:
        raise ValueError("exact fixed2048 child command with one output-dir required")
    index = result.index("--output-dir")
    result[index + 1] = str(Path(output_dir) / f"B{shape}")
    result += ["--stage-shape", str(shape)]
    if shape == 2048:
        result += ["--control-stage-dir", str(Path(output_dir) / "B1024")]
    return result


def candidate_resource_decision(receipt, remaining, snapshot):
    proof = receipt["stage"]["provenance"]
    preflight = proof["memory_preflight"]
    memory = proof["executable_memory_analysis"]
    reasons = []
    if any(type(snapshot.get(key)) is not int or snapshot[key] < 0 for key in (
            "GPU_total_bytes", "GPU_free_bytes", "MemAvailable_bytes")):
        return {"permitted": False, "reasons": ["actual nonnegative integer physical memory snapshots required"]}
    temp = memory.get("temp_size_in_bytes") if memory.get("available") is True else None
    if type(temp) is int:
        historical_bound = proof.get("admission_compiler_memory_records", {}).get("1024", {})
        if historical_bound.get("available") is True:
            temp = max(temp, historical_bound["temp_size_in_bytes"])
    estimated = None if type(temp) is not int or temp < 0 else (
        2 * preflight["resident_2048_input_bytes"] + 3 * preflight["full_2048_output_bytes"] + 2 * temp)
    host_required = 2 * max(27865956352, proof["host_peak_RSS_bytes"]) + 8 * 1024**3
    if remaining < 3600:
        reasons.append("at least3600s remaining required; not a completion guarantee")
    if (estimated is None or snapshot["GPU_free_bytes"] < estimated + 2 * 1024**3
            or snapshot["GPU_total_bytes"] < 30 * 1024**3):
        reasons.append("candidate projected buffers plus2GiB selected GPU slack required")
    if snapshot["MemAvailable_bytes"] < host_required:
        reasons.append("conservative process-RSS projection plus8GiB host slack required")
    if snapshot.get("GPU_uuid") != receipt["identity"]["run_binding"]["physical_gpu"]["uuid"]:
        reasons.append("same selected physical GPU UUID as completed control required")
    return {"permitted": not reasons, "reasons": reasons, "remaining_seconds": remaining,
        "minimum_remaining_seconds": 3600, "estimated_device_bytes": estimated,
        "conservative_host_required_bytes": host_required, "physical_snapshot": snapshot,
        "requires_performance_improvement": False,
        "scope": "control actual-stack collection bytes plus scaled actual compiler temporary; candidate repeats actual-stack admission in fresh worker; planning not peak/reservation/OOM guarantee"}


def physical_snapshot():
    uuid = os.environ["ORCHJAX_PHYSICAL_GPU_UUID"]
    output = subprocess.check_output(["nvidia-smi", "-i", uuid, "--query-gpu=memory.total,memory.free",
        "--format=csv,noheader,nounits"], text=True)
    total, free = [int(part.strip()) * 1024**2 for part in output.strip().split(",")]
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:"))
    return {"GPU_uuid": uuid, "GPU_total_bytes": total, "GPU_free_bytes": free, "MemAvailable_bytes": available}


def coordinate(args, *, run_child=None, snapshot=physical_snapshot):
    output = Path(args.output_dir)
    path = output / "orchestration.json"
    commands = {str(shape): child_command(args.child_command, output, shape) for shape in (1024, 2048)}
    if args.budget_seconds != 7200:
        raise ValueError("immutable7200s coordinator window required")
    if args.dry_run:
        return {"status": "dry_run", "contract": FIXED2048_CONTRACT, "process_strategy": PROCESS_STRATEGY_2048,
            "commands": commands, "budget_seconds": 7200, "minimum_remaining_seconds": 3600,
            "JAX_imported": "jax" in sys.modules, "model_execution": False}
    if path.exists() or any((output / f"B{shape}").exists() for shape in (1024, 2048)):
        raise ValueError("fresh2048 coordinator destination is single-use; no retries or old-stage reuse")
    start = float(os.environ.get("ORCHJAX_GPU_WINDOW_START_EPOCH", time.time()))
    if not math.isfinite(start) or start <= 0:
        raise ValueError("finite positive shared allocation-window epoch required")
    os.environ.setdefault("ORCHJAX_GPU_WINDOW_START_EPOCH", str(start))
    report = {"contract": FIXED2048_CONTRACT, "process_strategy": PROCESS_STRATEGY_2048,
        "coordinator_pid": os.getpid(), "coordinator_start_epoch": time.time(), "window_start_epoch": start,
        "budget_seconds": 7200, "status": "running", "children": {}, "completed_shapes": [],
        "comparison_shapes": [], "scientific_acceptance": False, "JAX_imported": "jax" in sys.modules}
    atomic_report(path, report)
    control = None
    for shape in (1024, 2048):
        remaining = 7200 - (time.time() - start)
        decision = {"permitted": remaining >= 3600, "remaining_seconds": remaining,
            "minimum_remaining_seconds": 3600, "reasons": [] if remaining >= 3600 else ["at least3600s remaining required"]}
        if shape == 2048:
            try:
                decision = candidate_resource_decision(control, remaining, snapshot())
            except Exception as exc:
                decision = {"permitted": False, "reasons": [f"resource snapshot failed: {exc}"]}
        report[f"B{shape}_dispatch_decision"] = decision
        atomic_report(path, report)
        if not decision["permitted"]:
            report["status"] = "candidate_denied" if shape == 2048 else "control_denied"
            break
        child = {"command": commands[str(shape)], "dispatch_start_epoch": time.time()}
        report["children"][str(shape)] = child
        atomic_report(path, report)
        try:
            if run_child is None:
                process = subprocess.Popen(child["command"], start_new_session=True)
                child["command_process_pid"] = process.pid
                atomic_report(path, report)
                child["returncode"] = process.wait(timeout=max(1, remaining))
            else:
                child.update(run_child(shape, child["command"], remaining))
            child["exit_epoch"] = time.time()
            child["dispatch_wall_seconds"] = child["exit_epoch"] - child["dispatch_start_epoch"]
            if child["returncode"] != 0:
                raise RuntimeError(f"child B{shape} exited{child['returncode']}")
            if (type(child.get("command_process_pid")) is not int or child["command_process_pid"] <= 0
                    or any(type(child[key]) not in (int, float) or not math.isfinite(child[key]) or child[key] <= 0
                        for key in ("dispatch_start_epoch", "exit_epoch"))
                    or child["exit_epoch"] < child["dispatch_start_epoch"]):
                raise ValueError("positive command-process PID and finite ordered dispatch/exit epochs required")
            receipt = validate_stage_receipt(output / f"B{shape}", shape,
                common_identity=None if control is None else control["common_identity"])
            if receipt["identity"]["run_binding"].get("window_start_epoch") != start:
                raise ValueError("child allocation-window epoch differs from coordinator")
            worker_start = receipt["execution_origin"]["process_start_epoch"]
            if not child["dispatch_start_epoch"] <= worker_start <= child["exit_epoch"]:
                raise ValueError("actual worker startup must lie within command dispatch/exit interval")
            if control is not None and (receipt["execution_origin"]["pid"] == control["execution_origin"]["pid"]
                    or receipt["execution_origin"]["process_start_epoch"] <= control["execution_origin"]["process_start_epoch"]
                    or receipt["execution_origin"]["process_start_epoch"] < report["children"]["1024"]["exit_epoch"]):
                raise ValueError("candidate requires distinct fresh worker PID/start after control process exit")
            child["worker_execution_origin"] = receipt["execution_origin"]
            child["sealed_receipt"] = {key: receipt[key] for key in ("path", "sha256", "bytes", "manifest_sha256", "report_sha256")}
            report["completed_shapes"].append(shape)
            report["comparison_shapes"] = list(report["completed_shapes"])
            if shape == 1024:
                control = receipt
        except BaseException as exc:
            # A timed-out active step is locally terminated/reaped, never scancelled.
            if run_child is None and "process" in locals() and process.poll() is None:
                if hasattr(os, "killpg"):
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if hasattr(os, "killpg"):
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
            child.update(error_type=type(exc).__name__, error=str(exc), exit_epoch=time.time())
            report["status"] = "candidate_failed" if shape == 2048 else "control_failed"
            break
        atomic_report(path, report)
    else:
        report["status"] = "completed_exploratory_capacity_screen"
    report["orchestration_wall_seconds"] = time.time() - report["coordinator_start_epoch"]
    atomic_report(path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=7200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("child_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.child_command[:1] == ["--"]:
        args.child_command = args.child_command[1:]
    def stopped(number, frame):
        raise InterruptedError(f"coordinator signal{number}; sealed control preserved")
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, stopped)
    report = coordinate(args)
    print(json.dumps(report, allow_nan=False))
    return 1 if report["status"] in ("control_failed", "control_denied", "candidate_failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
