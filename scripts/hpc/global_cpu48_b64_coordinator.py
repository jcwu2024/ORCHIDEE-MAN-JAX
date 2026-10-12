"""Bounded scalar-reference then single-process CPU48/B64 dispatcher.

This module uses only the standard library and never imports JAX or model code.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_SHA = "a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6"
DEFAULT_SCIENCE_SHA = "11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5"


def allocated_tres_cpu_count(value: str) -> int:
    entries = [item for item in value.split(",") if item.split("=", 1)[0] == "cpu"]
    if len(entries) != 1 or re.fullmatch(r"cpu=[1-9][0-9]*", entries[0]) is None:
        raise ValueError("parent AllocTRES must contain exactly one unambiguous cpu=N entry")
    return int(entries[0].split("=", 1)[1])


def parse_tres(value: str) -> dict[str, str]:
    if not value:
        raise ValueError("Slurm allocation TRES is empty")
    result = {}
    for item in value.split(","):
        key, separator, resource_value = item.partition("=")
        if not separator or not key or not resource_value or key in result:
            raise ValueError("Slurm allocation TRES contains malformed or duplicate fields")
        result[key] = resource_value
    allocated_tres_cpu_count(value)
    for key, resource_value in result.items():
        if key.startswith("gres/gpu") and resource_value != "0":
            raise ValueError("CPU48 allocation TRES contains a GPU")
    return result


def parse_scontrol_job_record(stdout: str) -> dict[str, str]:
    lines = stdout.splitlines()
    if len(lines) != 1 or not lines[0].strip():
        raise ValueError("scontrol must return exactly one nonempty job record")
    required = {"JobId", "JobState", "Partition", "NodeList", "NumNodes", "NumTasks", "CPUs/Task", "NumCPUs"}
    tracked = required | {"AllocTRES", "TRES"}
    values = {}
    for token in lines[0].split():
        key, separator, value = token.partition("=")
        if not separator or key not in tracked:
            continue
        if key in values:
            raise ValueError(f"scontrol job record contains duplicate {key} fields")
        values[key] = value
    missing = sorted(required - values.keys())
    if missing or any(not values[name] for name in required):
        raise ValueError("scontrol job record is missing required allocation fields: " + ", ".join(missing))
    return values


def _read_scontrol_evidence(env) -> tuple[str, str, dict[str, str]]:
    paths = {
        "stdout": env.get("ORCHJAX_SLURM_JOB_RECORD_STDOUT"),
        "stderr": env.get("ORCHJAX_SLURM_JOB_RECORD_STDERR"),
        "exitcode": env.get("ORCHJAX_SLURM_JOB_RECORD_EXITCODE"),
    }
    if any(not value for value in paths.values()):
        raise ValueError("saved scontrol stdout/stderr/exitcode paths are required")
    try:
        stdout = Path(paths["stdout"]).read_text(encoding="utf-8")
        stderr = Path(paths["stderr"]).read_text(encoding="utf-8")
        exitcode_text = Path(paths["exitcode"]).read_text(encoding="ascii").strip()
    except OSError as exc:
        raise ValueError(f"saved scontrol evidence could not be read: {exc}") from exc
    if exitcode_text != "0":
        raise ValueError("scontrol show job command did not exit successfully")
    if stderr:
        raise ValueError("scontrol show job wrote stderr; refusing ambiguous allocation evidence")
    return stdout, stderr, parse_scontrol_job_record(stdout)


def validate_allocation(environ=None) -> dict:
    env = os.environ if environ is None else environ
    if (not env.get("SLURM_JOB_ID") or env.get("SLURM_JOB_PARTITION") != "cnall"
            or env.get("SLURM_NTASKS", "1") != "1"
            or env.get("SLURM_CPUS_PER_TASK") != "48"):
        raise ValueError("requires one active cnall task with exactly 48 parent CPUs")
    node_count = env.get("SLURM_NNODES", env.get("SLURM_JOB_NUM_NODES"))
    if node_count != "1":
        raise ValueError("requires exactly one allocated Slurm node")
    if env.get("ORCHJAX_PARENT_CPU_COUNT") != "48":
        raise ValueError("ORCHJAX_PARENT_CPU_COUNT must bind the full 48-CPU parent allocation")
    _, _, record = _read_scontrol_evidence(env)
    expected = {
        "JobId": env["SLURM_JOB_ID"],
        "JobState": "RUNNING",
        "Partition": "cnall",
        "NodeList": env.get("SLURM_JOB_NODELIST", ""),
        "NumNodes": "1",
        "NumTasks": "1",
        "CPUs/Task": "48",
        "NumCPUs": "48",
    }
    for field, expected_value in expected.items():
        if not expected_value or record[field] != expected_value:
            raise ValueError(f"scontrol {field} does not match the active 48-CPU job environment")
    if env.get("SLURM_JOB_NUM_NODES", "1") != "1":
        raise ValueError("SLURM_JOB_NUM_NODES must report exactly one allocated node")

    alloc_tres = record.get("AllocTRES")
    legacy_tres = record.get("TRES")
    if alloc_tres:
        tres, source = alloc_tres, "AllocTRES"
        selected = parse_tres(tres)
        if legacy_tres:
            legacy = parse_tres(legacy_tres)
            for key in selected.keys() & legacy.keys():
                if selected[key] != legacy[key]:
                    raise ValueError(f"AllocTRES and TRES disagree on shared {key} allocation")
    elif legacy_tres:
        if record["JobState"] != "RUNNING":
            raise ValueError("legacy TRES fallback is accepted only for a RUNNING job")
        tres, source = legacy_tres, "TRES"
        selected = parse_tres(tres)
    else:
        raise ValueError("scontrol job record has neither AllocTRES nor usable TRES")
    if selected.get("cpu") != "48" or selected.get("node", "1") != "1":
        raise ValueError("Slurm allocation TRES must report exactly cpu=48 and node=1")
    if env.get("SLURM_JOB_GPUS"):
        raise ValueError("CPU48 contract prohibits any allocated GPU")
    exported_tres = env.get("ORCHJAX_SLURM_ALLOC_TRES")
    exported_source = env.get("ORCHJAX_SLURM_ALLOC_TRES_SOURCE")
    if exported_tres is not None and exported_tres != tres:
        raise ValueError("exported allocation TRES differs from saved scontrol evidence")
    if exported_source is not None and exported_source != source:
        raise ValueError("exported allocation source differs from saved scontrol evidence")
    return {"job_id": env["SLURM_JOB_ID"], "partition": env["SLURM_JOB_PARTITION"],
            "nodes": 1, "tasks": 1, "parent_cpu_count": 48, "allocated_tres": tres,
            "allocated_tres_source": source, "job_state": record["JobState"],
            "scontrol_job_record": {key: record[key] for key in expected}}


def child_arguments(args, mode: str, output: Path) -> list[str]:
    common = [args.python, str(args.entry), "--mode", mode,
        "--cpu-budget", "1" if mode == "scalar" else "48", "--sizes", "64",
        "--prepared-input", str(args.prepared_input), "--gpu-reference-dir", str(args.gpu_reference_dir),
        "--expected-gpu-reference-sha256", args.expected_gpu_reference_sha256,
        "--config", str(args.config), "--warm-repeats", "5", "--validation-mode", "exploratory",
        "--cpu-reference-role", "cpu-internal",
        "--memory-guard-gib", "24" if mode == "scalar" else "64",
        "--output-dir", str(output)]
    if mode == "batch":
        common += ["--cpu-reference-dir", str(args.output_dir / "scalar")]
    return common


def step_command(args, mode: str) -> tuple[list[str], dict[str, str]]:
    cpus = 1 if mode == "scalar" else 48
    environment = {"OMP_NUM_THREADS": "1" if cpus == 1 else "48",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "true"}
    if cpus == 48:
        environment["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"
    exports = ",".join(f"{key}={value}" for key, value in environment.items())
    command = [str(args.slurm_bin / "srun"), "--ntasks=1", f"--cpus-per-task={cpus}",
        "--cpu-bind=cores", f"--export=ALL,{exports}", "/usr/bin/time", "-v"]
    command += child_arguments(args, mode, args.output_dir / ("scalar" if mode == "scalar" else "batch"))
    return command, environment


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def host_snapshot() -> dict:
    value = {"captured_epoch": time.time(), "hostname": __import__("platform").node(),
        "loadavg": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "affinity_logical_cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                value["MemAvailable_bytes"] = int(line.split()[1]) * 1024
                break
    try:
        value["uptime"] = subprocess.check_output(["uptime"], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        value["uptime_error"] = str(exc)
    return value


def run_process(command: list[str], log_path: Path, timeout_seconds: float) -> dict:
    started = time.time()
    with log_path.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "start_epoch": started}, ensure_ascii=True) + "\n")
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=max(1.0, timeout_seconds))
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = terminate_process_group(process)
        except BaseException:
            terminate_process_group(process)
            raise
        ended = time.time()
        log.write(json.dumps({"returncode": returncode, "timed_out": timed_out, "end_epoch": ended,
            "wall_seconds": ended - started}, ensure_ascii=True) + "\n")
        log.flush()
        os.fsync(log.fileno())
    return {"returncode": returncode, "timed_out": timed_out, "start_epoch": started,
        "end_epoch": ended, "wall_seconds": ended - started, "log": str(log_path)}


def terminate_process_group(process, *, grace_seconds=10) -> int:
    """Stop local srun and let Slurm reap its active step; never scancel the parent job."""
    if process.poll() is not None:
        return process.returncode
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()


def runner_dry_run(args, mode: str) -> dict:
    command = child_arguments(args, mode, args.output_dir / ("scalar" if mode == "scalar" else "batch"))
    child_env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "JAX_PLATFORMS": "cpu",
             "JAX_ENABLE_X64": "true", "OMP_NUM_THREADS": "1" if mode == "scalar" else "48",
             "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    if mode == "batch":
        child_env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"
    else:
        child_env.pop("XLA_FLAGS", None)
    completed = subprocess.run([*command, "--dry-run"], capture_output=True, text=True, check=False, env=child_env)
    if completed.returncode:
        raise RuntimeError(f"runner {mode} --dry-run failed: {completed.stderr}")
    result = json.loads(completed.stdout)
    if result.get("model_execution") is not False or result.get("JAX_imported") is not False:
        raise ValueError("runner dry-run must not import JAX or execute the model")
    return {"command": command + ["--dry-run"], "runner_result": result}


def coordinate(args, *, run_step=None, clock=time.time, snapshot=host_snapshot) -> dict:
    if args.budget_seconds != 5400 or args.reference_cap_seconds != 2700 or args.minimum_candidate_seconds != 1800:
        raise ValueError("immutable 5400/2700/1800 second run window required")
    if args.expected_gpu_reference_sha256 != DEFAULT_REFERENCE_SHA:
        raise ValueError("declared numerical GPU reference SHA differs from accepted FULL64 snapshot")
    if args.expected_science_sha256 != DEFAULT_SCIENCE_SHA:
        raise ValueError("declared science source SHA differs from accepted comparison source")
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_code_commit or ""):
        raise ValueError("a concrete 40-character execution commit is required")
    if not args.dry_run:
        code_dir = Path(args.code_dir).resolve()
        project_root = Path(args.project_root).resolve()
        if not code_dir.is_relative_to(project_root / "runtime" / "worktrees"):
            raise ValueError("CODE_DIR must remain under project runtime/worktrees")
        if Path.cwd().resolve() != code_dir:
            raise ValueError("coordinator cwd must equal the immutable CODE_DIR")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_dir, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=code_dir, text=True).strip()
        if head != args.expected_code_commit or dirty:
            raise ValueError("execution tree must be clean and at EXPECTED_CODE_COMMIT before starting")
    commands = {mode: {"argv": step_command(args, mode)[0], "environment": step_command(args, mode)[1]}
        for mode in ("scalar", "batch")}
    if args.dry_run:
        checks = {mode: runner_dry_run(args, mode) for mode in ("scalar", "batch")}
        return {"status": "dry_run", "contract": "CPU48_SINGLE_PROCESS_FULL64_B64_v2",
            "expected_execution_commit": args.expected_code_commit, "expected_science_sha256": DEFAULT_SCIENCE_SHA,
            "expected_gpu_reference_sha256": DEFAULT_REFERENCE_SHA,
            "allocation": {"partition": args.partition, "parent_cpu_count": 48,
                "wall_seconds": 5400, "primary_worst_case_cny": 3.60, "conservative_worst_case_cny": 5.04},
            "minimum_candidate_seconds": 1800, "commands": commands,
            "runner_cli_dry_runs": checks, "model_execution": False, "JAX_imported": "jax" in sys.modules}
    allocation = validate_allocation()
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(args.project_root.resolve() / "runtime/outputs/research/global_throughput/cpu48-b64-execution-v2-20261012"):
        raise ValueError("RUN_ROOT must remain in CPU48/B64 v2 outputs")
    if output.exists() and any(output.iterdir()):
        raise ValueError("single-use output directory must be new and empty")
    output.mkdir(parents=True, exist_ok=True)
    start = clock()
    report = {"contract": "CPU48_SINGLE_PROCESS_FULL64_B64_v2", "status": "running",
        "coordinator_pid": os.getpid(), "start_epoch": start, "budget_seconds": 5400,
        "reference_cap_seconds": 2700, "minimum_candidate_seconds": 1800,
        "allocation": allocation, "host_start_snapshot": snapshot(), "expected_execution_commit": args.expected_code_commit,
        "expected_science_sha256": DEFAULT_SCIENCE_SHA, "expected_gpu_reference_sha256": DEFAULT_REFERENCE_SHA,
        "commands": commands, "steps": {}, "scientific_acceptance": False,
        "performance_claim_scope": "exploratory same-workload throughput only"}
    orchestration_path = output / "orchestration.json"
    atomic_json(orchestration_path, report)
    dispatch = run_process if run_step is None else run_step
    try:
        for mode, cap in (("scalar", args.reference_cap_seconds), ("batch", None)):
            remaining = args.budget_seconds - (clock() - start)
            timeout = min(cap, remaining) if cap is not None else remaining
            if mode == "batch" and remaining < args.minimum_candidate_seconds:
                report["status"] = "candidate_skipped_insufficient_window"
                report["candidate_dispatch_decision"] = {"permitted": False, "remaining_seconds": remaining,
                    "minimum_remaining_seconds": args.minimum_candidate_seconds}
                break
            if timeout <= 0:
                report["status"] = f"{mode}_failed_window_exhausted"
                break
            report["steps"][mode] = {"status": "running", "timeout_seconds": timeout,
                "dispatch_start_epoch": clock(), "cpu_budget": 1 if mode == "scalar" else 48,
                "wrapper_workload": "one scalar FULL64 reference" if mode == "scalar" else "one native FULL64 B64 vmap",
                "command": commands[mode]["argv"], "environment": commands[mode]["environment"]}
            atomic_json(orchestration_path, report)
            result = dispatch(commands[mode]["argv"], output / f"{mode}.log", timeout)
            report["steps"][mode].update(result)
            report["steps"][mode]["status"] = "timed_out" if result["timed_out"] else (
                "completed" if result["returncode"] == 0 else "failed")
            if result["timed_out"] or result["returncode"] != 0:
                report["status"] = f"{mode}_failed"
                break
            if mode == "scalar":
                report["reference_seal"] = {"directory": str(output / "scalar"),
                    "manifest": str(output / "scalar/stage_manifest.json"),
                    "report": str(output / "scalar/report.json"),
                    "policy": "runner-verified complete B64 scalar output; consumed only through exact internal reference loader"}
            else:
                report["candidate_seal"] = {"directory": str(output / "batch"),
                    "manifest": str(output / "batch/stage_manifest.json"),
                    "report": str(output / "batch/report.json")}
            atomic_json(orchestration_path, report)
        else:
            report["status"] = "completed_exploratory_measurement"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "coordinator_failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["end_epoch"] = clock()
        report["wall_seconds"] = max(0.0, report["end_epoch"] - start)
        report["coordinator_scope_primary_cost_cny_estimate"] = 48 * report["wall_seconds"] / 3600 * 0.05
        report["coordinator_scope_conservative_cost_cny_estimate"] = 48 * report["wall_seconds"] / 3600 * 0.07
        report["coordinator_scope_cost_limitation"] = (
            "lower-scope observed coordinator interval; excludes Slurm submission/startup and final scheduler accounting")
        report["billing_verified"] = False
        atomic_json(orchestration_path, report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--project-root", type=Path, default=Path("/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX"))
    result.add_argument("--code-dir", type=Path, default=ROOT)
    result.add_argument("--python", default=sys.executable)
    result.add_argument("--entry", type=Path, default=ROOT / "scripts/dev/benchmark_global_server_cpu.py")
    result.add_argument("--expected-code-commit")
    result.add_argument("--expected-science-sha256", default=DEFAULT_SCIENCE_SHA)
    result.add_argument("--expected-gpu-reference-sha256", default=DEFAULT_REFERENCE_SHA)
    result.add_argument("--prepared-input", type=Path)
    result.add_argument("--gpu-reference-dir", type=Path)
    result.add_argument("--config", type=Path)
    result.add_argument("--partition", default="cnall")
    result.add_argument("--slurm-bin", type=Path, default=Path("/rmprog/slurm/v22.05.7/bin"))
    result.add_argument("--budget-seconds", type=int, default=5400)
    result.add_argument("--reference-cap-seconds", type=int, default=2700)
    result.add_argument("--minimum-candidate-seconds", type=int, default=1800)
    result.add_argument("--validate-allocation", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def main():
    args = parser().parse_args()
    if args.validate_allocation:
        print(json.dumps(validate_allocation(), allow_nan=False))
        return 0
    required = ("expected_code_commit", "prepared_input", "gpu_reference_dir", "config", "entry", "output_dir")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser().error("required arguments missing: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    def stopped(number, frame):
        raise InterruptedError(f"coordinator signal {number}; active local srun process group will be terminated")
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, stopped)
    report = coordinate(args)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] in {"dry_run", "completed_exploratory_measurement"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
