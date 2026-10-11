"""Coordinate the two accepted fresh-process B1024 lifecycle arms on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import posixpath
import signal
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

CONTRACT = "GPU-B1024-phased-memory-lifecycle-new-science-v1"
PROCESS_STRATEGY = "sequential_fresh_process_per_strategy"
STAGE_NAME = "capacity_B1024_lifecycle_v1"
STRATEGIES = ("phased_resident_control", "phased_release")
SAMPLING_ORDER = "five_core_then_five_wrapper"
POLICIES = {STRATEGIES[0]: "retain_core_inputs", STRATEGIES[1]: "release_core_inputs"}
ROOT = Path(__file__).resolve().parents[2]


def digest_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Journal:
    def __init__(self, output: Path):
        self.output = output
        self.path = output / "orchestration.json"
        self.events_path = output / "orchestration-events.jsonl"
        self.value: dict = {}

    def event(self, name: str, **details) -> None:
        row = {"epoch": time.time(), "event": name, **details}
        with self.events_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def flush(self, event: str, **details) -> None:
        self.value["updated_epoch"] = time.time()
        atomic_json(self.path, self.value)
        self.event(event, **details)


def _value_after(command: list[str], flag: str) -> str:
    if command.count(flag) != 1:
        raise ValueError(f"child command must contain exactly one {flag}")
    index = command.index(flag)
    if index + 1 >= len(command):
        raise ValueError(f"child command is missing a value after {flag}")
    return command[index + 1]


def _posix_path(value) -> str:
    return posixpath.normpath(os.fspath(value).replace("\\", "/"))


def _absolute_path(value: str) -> bool:
    return posixpath.isabs(value) or Path(value).is_absolute()


def validate_command_template(command: list[str], expected_code_sha: str, *, code_dir: Path,
                              full_reference_dir: Path, dry_run: bool) -> None:
    if not command or command[0].endswith(".sh"):
        raise ValueError("child must be an argv vector, not a shell command string")
    if not any(token.endswith("global_batch_gpu_entry.py") for token in command):
        raise ValueError("child must use the canonical global_batch_gpu_entry.py")
    if (os.environ.get("ORCHJAX_GLOBAL_BATCH_ENTRY") != "benchmark_global_gpu_lifecycle.py"
            or os.environ.get("SINGULARITYENV_ORCHJAX_GLOBAL_BATCH_ENTRY") != "benchmark_global_gpu_lifecycle.py"):
        raise ValueError("host and Singularity entry environment must select the lifecycle runner")
    if _value_after(command, "--stage-shape") != "1024":
        raise ValueError("lifecycle children are fixed to B1024 only")
    if _value_after(command, "--warm-repeats") != "5" or _value_after(command, "--budget-seconds") != "7200":
        raise ValueError("lifecycle children require five samples and the 7200-second allocation contract")
    if _value_after(command, "--measurement-strategy") != "__MEASUREMENT_STRATEGY__":
        raise ValueError("coordinator owns the measurement strategy argument")
    if _value_after(command, "--output-dir") != "__ARM_OUTPUT_DIR__":
        raise ValueError("coordinator owns each arm's output directory")
    if _value_after(command, "--expected-code-sha") != expected_code_sha:
        raise ValueError("child expected code pin differs from coordinator pin")
    if set(command).intersection({"--fixed-2048", "--fixed-1024", "--stage-2048", "--continue-512"}):
        raise ValueError("B1024 lifecycle coordinator rejects other capacity workflows")
    if sum(token.endswith("global_batch_gpu_entry.py") for token in command) != 1:
        raise ValueError("exactly one canonical GPU entry is required")
    entry = next(token for token in command if token.endswith("global_batch_gpu_entry.py"))
    expected_entry = posixpath.join(_posix_path(code_dir), "scripts/hpc/global_batch_gpu_entry.py")
    if _posix_path(entry) != expected_entry:
        raise ValueError("canonical GPU entry must resolve inside the declared immutable CODE_DIR")
    if _posix_path(_value_after(command, "--full-reference-dir")) != _posix_path(full_reference_dir):
        raise ValueError("both arms must use the declared frozen FULL64 reference directory")
    entry_index = command.index(entry)
    if (command[:4] != ["/rmprog/slurm/v22.05.7/bin/srun", "--ntasks=1", "--cpus-per-task=4", "--cpu-bind=cores"]
            or command[4:8] != ["/usr/bin/time", "-v", "singularity", "exec"]
            or command[8] != "--nv" or command[9] != "--bind" or command[10] in ("", "__CUDA_BINDS__")
            or (not dry_run and command[10] == "<canonical-cuda-nvml-driver-binds>")
            or command[11:13] != ["--pwd", _posix_path(code_dir)]):
        raise ValueError("child must use the accepted srun/time/Singularity profile and CODE_DIR working directory")
    if (command[13] != "/apps/soft/sif/foundationpose"
            or command[14] != "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_gpu/bin/python"
            or entry_index != 15):
        raise ValueError(f"child must use the canonical Explore1000 container, CUDA Python, and direct GPU entry; observed_python={command[14]!r}, entry_index={entry_index}")
    runner_flags = (
        "--prepared-input", "--full-reference-dir", "--memory-baseline-dir", "--b1024-evidence-dir",
        "--config", "--asset-root", "--output-dir", "--warm-repeats", "--budget-seconds",
        "--stage-shape", "--measurement-strategy", "--expected-code-sha",
    )
    values = {flag: _value_after(command, flag) for flag in runner_flags}
    expected_runner_args = []
    for flag in runner_flags:
        expected_runner_args.extend((flag, values[flag]))
    if command[entry_index + 1:] != expected_runner_args:
        raise ValueError("unknown, duplicated, or reordered lifecycle runner arguments are rejected")
    if any(not _absolute_path(values[flag]) and values[flag] != "__ARM_OUTPUT_DIR__"
           for flag in runner_flags[:7]):
        raise ValueError("all runner input, reference, and output paths must be absolute")


def child_command(template: list[str], output: Path | PurePosixPath, strategy: str) -> list[str]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unsupported lifecycle strategy {strategy!r}")
    result = list(template)
    result[result.index("__MEASUREMENT_STRATEGY__")] = strategy
    result[result.index("__ARM_OUTPUT_DIR__")] = str(output / strategy / "capacity" / "B1024")
    return result


def _validator_call(directory: Path, *, strategy: str, expected_code_sha: str,
                    common_identity: dict | None, reference_dir: Path, acceptance_path: Path | None):
    """Run saved-array validation with JAX explicitly restricted to CPU."""
    original = {key: os.environ.get(key) for key in ("JAX_PLATFORMS", "JAX_PLATFORM_NAME", "CUDA_VISIBLE_DEVICES")}
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        sys.path.insert(0, str(ROOT))
        from scripts.dev.validate_global_gpu_lifecycle import validate_stage_receipt
        return validate_stage_receipt(
            directory,
            strategy=strategy,
            expected_code_sha=expected_code_sha,
            common_identity=common_identity,
            reference_dir=reference_dir,
            acceptance_path=acceptance_path,
        )
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _identity_contract(receipt: dict, strategy: str, *, expected_code_sha: str,
                       selected_uuid: str, total_bytes: int) -> tuple[dict, dict]:
    identity = receipt.get("identity")
    common = receipt.get("common_identity")
    if not isinstance(identity, dict) or not isinstance(common, dict):
        raise ValueError("validator must return complete identity and common_identity mappings")
    expected = {
        "contract": CONTRACT,
        "process_strategy": PROCESS_STRATEGY,
        "stage_name": STAGE_NAME,
        "stage_shape": 1024,
        "total_lanes": 2048,
        "warm_repeats": 5,
        "budget_seconds": 7200,
        "candidate_shapes": [1024],
        "measurement_strategy": strategy,
        "sampling_order": SAMPLING_ORDER,
        "wrapper_resident_input_policy": POLICIES[strategy],
        "expected_code_sha": expected_code_sha,
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError(f"{strategy} identity mismatch for {key}: expected {value!r}")
    # Sampling order is a common requirement and must not be dropped from matching.
    expected_common = {key: value for key, value in identity.items()
                       if key not in ("measurement_strategy", "wrapper_resident_input_policy")}
    if common != expected_common:
        raise ValueError("common identity must exactly retain every identity field except strategy and resident-input policy")
    if common.get("sampling_order") != SAMPLING_ORDER:
        raise ValueError("common identity must retain five_core_then_five_wrapper")
    binding = identity.get("run_binding", {})
    gpu = binding.get("physical_gpu", {}) if isinstance(binding, dict) else {}
    if gpu != {"uuid": selected_uuid, "total_bytes": total_bytes}:
        raise ValueError("scientific identity physical GPU UUID/total differs from launcher snapshot")
    if binding.get("allocator") != "default" or binding.get("preallocate") is not False:
        raise ValueError("all arms require the default allocator and PREALLOCATE=false")
    resources = binding.get("server_resources", {})
    slurm = resources.get("Slurm", {}) if isinstance(resources, dict) else {}
    if (resources.get("allocated_CPU_request_count") != 4 or resources.get("CUDA_VISIBLE_DEVICES_preserved") in (None, "", "-1")
            or len(str(resources.get("CUDA_VISIBLE_DEVICES_preserved", "")).split(",")) != 1
            or slurm.get("SLURM_CPUS_PER_TASK") not in (4, "4")
            or slurm.get("SLURM_NTASKS") not in (1, "1")
            or slurm.get("SLURM_JOB_PARTITION") != "gnall"):
        raise ValueError("stable identity must prove one gnall task, four CPUs, and one preserved visible GPU")
    if (binding.get("Slurm_job_id") != slurm.get("SLURM_JOB_ID")
            or binding.get("hostname") != resources.get("hostname")):
        raise ValueError("stable Slurm job/node identity is inconsistent")
    if "V100" not in str(identity.get("runtime", {}).get("device_details", "")):
        raise ValueError("scientific runtime identity must record the selected V100 device")
    return identity, common


def _origin(receipt: dict) -> dict:
    origin = receipt.get("execution_origin")
    if not isinstance(origin, dict):
        raise ValueError("actual worker execution origin is required")
    pid, started = origin.get("pid"), origin.get("process_start_epoch")
    if type(pid) is not int or pid <= 0 or type(started) not in (int, float) or not math.isfinite(started) or started <= 0:
        raise ValueError("actual worker PID and finite positive process start epoch are required")
    return origin


def _seal(receipt: dict, directory: Path) -> dict:
    path = Path(receipt.get("path", ""))
    if not path.is_absolute() or not path.is_file():
        raise ValueError("validator did not return an existing absolute sealed stage path")
    try:
        path.relative_to(directory.resolve())
    except ValueError as exc:
        raise ValueError("sealed stage path escaped the arm output directory") from exc
    sha, size = digest_file(path)
    if receipt.get("sha256") != sha or receipt.get("bytes") != size:
        raise ValueError("sealed stage checksum or byte count changed after independent acceptance")
    acceptance = Path(receipt.get("array_acceptance_path", ""))
    if not acceptance.is_absolute() or not acceptance.is_file():
        raise ValueError("independent saved-array acceptance artifact is missing")
    try:
        acceptance.relative_to(directory.resolve())
    except ValueError as exc:
        raise ValueError("array acceptance artifact escaped the arm output directory") from exc
    acceptance_sha, _ = digest_file(acceptance)
    if receipt.get("array_acceptance_sha256") != acceptance_sha:
        raise ValueError("independent array acceptance artifact checksum differs from validator receipt")
    return {"path": str(path), "sha256": sha, "bytes": size,
            "array_acceptance_path": str(acceptance),
            "array_acceptance_sha256": acceptance_sha}


def _physical_snapshot(expected_uuid: str | None = None) -> dict:
    if expected_uuid is None:
        expected_uuid = os.environ.get("ORCHJAX_PHYSICAL_GPU_UUID")
    if not expected_uuid:
        raise ValueError("selected physical GPU UUID is unavailable")
    output = subprocess.check_output([
        "nvidia-smi", "-i", expected_uuid,
        "--query-gpu=uuid,memory.total,memory.free", "--format=csv,noheader,nounits",
    ], text=True, timeout=20)
    parts = [part.strip() for part in output.strip().split(",")]
    if len(parts) != 3:
        raise ValueError(f"unrecognized selected-GPU memory response: {output!r}")
    uuid, total_mb, free_mb = parts
    if uuid != expected_uuid:
        raise ValueError("selected physical GPU UUID changed")
    return {"uuid": uuid, "total_bytes": int(total_mb) * 1024**2,
            "free_bytes": int(free_mb) * 1024**2, "source": "nvidia-smi selected-card boundary snapshot"}


def _kill_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
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


def _throughput_comparison(control: dict, release: dict) -> dict:
    boundaries = {}
    for name in ("core", "wrapper"):
        control_metric = control.get("stage_timing", {}).get(name, {})
        release_metric = release.get("stage_timing", {}).get(name, {})
        if (control_metric.get("lane_days") != 28672 or release_metric.get("lane_days") != 28672
                or type(control_metric.get("lane_days_per_second")) not in (int, float)
                or type(release_metric.get("lane_days_per_second")) not in (int, float)
                or control_metric["lane_days_per_second"] <= 0 or release_metric["lane_days_per_second"] <= 0):
            raise ValueError(f"validated {name} timing summaries must report positive 28672-lane-day throughput")
        boundaries[name] = {
            "workload_lane_days": 28672,
            "control_median_seconds": control_metric["median_seconds"],
            "release_median_seconds": release_metric["median_seconds"],
            "control_lane_days_per_second": control_metric["lane_days_per_second"],
            "release_lane_days_per_second": release_metric["lane_days_per_second"],
            "release_over_control_throughput_ratio": (
                release_metric["lane_days_per_second"] / control_metric["lane_days_per_second"]
            ),
            "control_samples_seconds": control_metric["samples_seconds"],
            "release_samples_seconds": release_metric["samples_seconds"],
        }
    return {"contract": "same-workload-phased-lifecycle-throughput-comparison-v1",
            "workload_lane_days_per_arm": 28672,
            "arms": list(STRATEGIES),
            "common_identity": control["common_identity"],
            "arm_receipts": {name: {"stage_sha256": receipt["sealed_receipt"]["sha256"],
                                    "array_acceptance_sha256": receipt["sealed_receipt"]["array_acceptance_sha256"]}
                             for name, receipt in zip(STRATEGIES, (control, release), strict=True)},
            "boundaries": boundaries,
            "limitation": "bounded repeated-FULL64 B1024 lifecycle comparison; not independent ensemble throughput or peak VRAM"}


def coordinate(args, *, run_child=None, validate_receipt=_validator_call, snapshot=_physical_snapshot,
               clock=time.time) -> dict:
    output_value = os.fspath(args.output_dir)
    if args.dry_run and os.name == "nt" and output_value.startswith("/"):
        output = PurePosixPath(posixpath.normpath(output_value))
    else:
        output = Path(output_value).resolve()
    expected_sha = args.expected_code_sha.lower()
    if len(expected_sha) != 40 or any(char not in "0123456789abcdef" for char in expected_sha):
        raise ValueError("expected code SHA must be a full 40-character Git SHA")
    if args.budget_seconds != 7200:
        raise ValueError("lifecycle allocation window is fixed at 7200 seconds")
    if not args.full_reference_dir:
        raise ValueError("frozen FULL64 reference directory is required")
    args.code_dir = Path(args.code_dir)
    validate_command_template(args.child_command, expected_sha, code_dir=args.code_dir,
                              full_reference_dir=Path(args.full_reference_dir), dry_run=args.dry_run)
    if not args.dry_run and args.code_dir.resolve() != ROOT.resolve():
        raise ValueError("actual coordinator must execute from the exact declared immutable CODE_DIR")
    commands = {strategy: child_command(args.child_command, output, strategy) for strategy in STRATEGIES}
    if args.dry_run:
        return {"status": "dry_run", "contract": CONTRACT, "process_strategy": PROCESS_STRATEGY,
                "stage_name": STAGE_NAME, "strategies": list(STRATEGIES), "commands": commands,
                "budget_seconds": 7200, "minimum_remaining_before_second_seconds": 3600,
                "JAX_imported": "jax" in sys.modules, "model_execution": False,
                "array_validation": False}

    output.mkdir(parents=True, exist_ok=True)
    journal = Journal(output)
    if journal.path.exists() or journal.events_path.exists() or any((output / name).exists() for name in STRATEGIES):
        raise ValueError("lifecycle output root is single-use; no retries or previous-stage reuse")
    if not args.gpu_identity or not args.gpu_identity.is_file():
        raise ValueError("verified metadata-only GPU identity receipt is required")
    gpu_identity = json.loads(args.gpu_identity.read_text(encoding="utf-8"))
    if gpu_identity.get("status") != "verified_metadata_only":
        raise ValueError("GPU identity receipt is not verified metadata-only evidence")
    selected_uuid = gpu_identity.get("uuid")
    if selected_uuid != os.environ.get("ORCHJAX_PHYSICAL_GPU_UUID"):
        raise ValueError("launcher GPU UUID differs from selected physical GPU receipt")
    if (os.environ.get("ORCHJAX_GLOBAL_BATCH_ENTRY") != "benchmark_global_gpu_lifecycle.py"
            or os.environ.get("SINGULARITYENV_ORCHJAX_GLOBAL_BATCH_ENTRY") != "benchmark_global_gpu_lifecycle.py"):
        raise ValueError("lifecycle runner entry environment identity is missing")
    window_start = float(os.environ.get("ORCHJAX_GPU_WINDOW_START_EPOCH", "nan"))
    if not math.isfinite(window_start) or window_start <= 0:
        raise ValueError("finite positive allocation-window epoch required")
    journal.value = {
        "contract": CONTRACT,
        "process_strategy": PROCESS_STRATEGY,
        "stage_name": STAGE_NAME,
        "expected_code_sha": expected_sha,
        "full_reference_dir": str(args.full_reference_dir),
        "coordinator_pid": os.getpid(),
        "coordinator_start_epoch": clock(),
        "window_start_epoch": window_start,
        "budget_seconds": 7200,
        "minimum_remaining_before_second_seconds": 3600,
        "selected_gpu_identity": gpu_identity,
        "status": "running",
        "children": {},
        "comparison_allowed": False,
        "ratio_published": False,
    }
    journal.flush("coordinator_started")

    previous_receipt = None
    previous_identity = None
    previous_origin = None
    for index, strategy in enumerate(STRATEGIES):
        remaining = 7200 - (clock() - window_start)
        dispatch = {"permitted": remaining >= (3600 if index == 1 else 1),
                    "remaining_seconds": remaining,
                    "minimum_remaining_seconds": 3600 if index == 1 else 1}
        if not dispatch["permitted"]:
            journal.value["status"] = "second_arm_denied_time_guard" if index == 1 else "first_arm_denied_time_guard"
            journal.value["dispatch_decision"] = dispatch
            journal.flush("dispatch_denied", strategy=strategy, **dispatch)
            break
        try:
            physical = snapshot(selected_uuid)
            if physical.get("uuid") != selected_uuid or type(physical.get("total_bytes")) is not int or physical["total_bytes"] <= 0:
                raise ValueError("selected UUID and positive physical GPU total are required")
            if "V100" not in str(gpu_identity.get("name", "")) or not 30 * 1024**3 <= physical["total_bytes"] <= 34 * 1024**3:
                raise ValueError("approved resource contract requires an actual V100 32GB selected card")
            if previous_receipt is not None:
                if physical["uuid"] != previous_identity["physical_gpu_uuid"]:
                    raise ValueError("selected GPU UUID differs from accepted first arm")
                if physical["total_bytes"] != previous_identity["physical_gpu_total_bytes"]:
                    raise ValueError("selected GPU total memory differs from accepted first arm")
                remaining = 7200 - (clock() - window_start)
                if remaining < 3600:
                    raise ValueError("at least 3600 seconds must remain immediately before second dispatch")
            journal.value["children"][strategy] = {
                "command": commands[strategy],
                "dispatch_decision": dispatch,
                "pre_dispatch_physical_gpu": physical,
                "status": "dispatching",
            }
            journal.flush("child_dispatch_started", strategy=strategy, physical_gpu=physical)
            child = journal.value["children"][strategy]
            remaining = 7200 - (clock() - window_start)
            if remaining < (3600 if index == 1 else 1):
                raise TimeoutError("allocation time guard failed immediately before child dispatch")
            child["final_dispatch_guard"] = {"remaining_seconds": remaining,
                                             "minimum_remaining_seconds": 3600 if index == 1 else 1}
            child["dispatch_start_epoch"] = clock()
            if run_child is None:
                process = subprocess.Popen(commands[strategy], start_new_session=True)
                child["command_process_pid"] = process.pid
                journal.flush("child_process_started", strategy=strategy, pid=process.pid)
                child["returncode"] = process.wait(timeout=max(1, 7200 - (clock() - window_start)))
            else:
                child.update(run_child(strategy, commands[strategy], max(1, 7200 - (clock() - window_start))))
            child["exit_epoch"] = clock()
            if child.get("returncode") != 0:
                raise RuntimeError(f"{strategy} child exited with code {child.get('returncode')}")
            if type(child.get("command_process_pid")) is not int or child["command_process_pid"] <= 0:
                raise ValueError("fresh child command PID is required")
            dispatch_start, exit_epoch = child.get("dispatch_start_epoch"), child.get("exit_epoch")
            if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in (dispatch_start, exit_epoch)) or exit_epoch < dispatch_start:
                raise ValueError("finite ordered command dispatch/exit epochs are required")

            arm_dir = output / strategy / "capacity" / "B1024"
            receipt = validate_receipt(
                arm_dir,
                strategy=strategy,
                expected_code_sha=expected_sha,
                common_identity=previous_receipt["common_identity"] if previous_receipt else None,
                reference_dir=args.full_reference_dir,
                acceptance_path=None,
            )
            identity, common = _identity_contract(receipt, strategy, expected_code_sha=expected_sha,
                                                  selected_uuid=selected_uuid, total_bytes=physical["total_bytes"])
            if previous_receipt is not None and common != previous_receipt["common_identity"]:
                raise ValueError("second arm common identity differs from the accepted first arm")
            if index == 0:
                previous_identity = {
                    "physical_gpu_uuid": physical["uuid"],
                    "physical_gpu_total_bytes": physical["total_bytes"],
                }
            origin = _origin(receipt)
            if not dispatch_start <= origin["process_start_epoch"] <= exit_epoch:
                raise ValueError("actual worker process start must lie within its child dispatch/exit interval")
            if previous_origin is not None:
                if origin["pid"] == previous_origin["pid"]:
                    raise ValueError("each strategy requires a fresh worker PID")
                if origin["process_start_epoch"] <= previous_origin["process_start_epoch"]:
                    raise ValueError("second worker must start after the first worker")
                if origin["process_start_epoch"] < journal.value["children"][STRATEGIES[0]]["exit_epoch"]:
                    raise ValueError("second worker must start after the first command exited")
            seal = _seal(receipt, arm_dir)
            child.update(status="accepted", worker_execution_origin=origin,
                         identity=identity, common_identity=common, stage_timing=receipt["stage"].get("timing", {}),
                         sealed_receipt=seal)
            journal.value["accepted_strategies"] = [*journal.value.get("accepted_strategies", []), strategy]
            journal.flush("child_sealed_and_independently_accepted", strategy=strategy,
                          stage_sha256=seal["sha256"], acceptance_path=seal["array_acceptance_path"])
            previous_receipt = receipt
            previous_origin = origin
        except BaseException as exc:
            if run_child is None and "process" in locals():
                _kill_process(process)
            child = journal.value["children"].setdefault(strategy, {})
            child.update(status="failed", error_type=type(exc).__name__, error=str(exc), exit_epoch=clock())
            journal.value["status"] = "first_arm_failed" if index == 0 else "second_arm_failed"
            journal.flush("child_failed_preserve_outputs", strategy=strategy,
                          error_type=type(exc).__name__, error=str(exc))
            break
    else:
        try:
            comparison = _throughput_comparison(
                journal.value["children"][STRATEGIES[0]],
                journal.value["children"][STRATEGIES[1]],
            )
            atomic_json(output / "comparison.json", comparison)
            journal.value["status"] = "completed_comparison"
            journal.value["comparison_allowed"] = True
            journal.value["ratio_published"] = True
            journal.value["comparison_path"] = str(output / "comparison.json")
            journal.value["comparison"] = comparison
            journal.flush("both_arms_accepted_and_ratio_computed")
        except Exception as exc:
            journal.value["status"] = "comparison_failed"
            journal.value["comparison_error"] = f"{type(exc).__name__}: {exc}"
            journal.flush("comparison_failed_no_ratio", error=journal.value["comparison_error"])
    journal.value["orchestration_wall_seconds"] = clock() - journal.value["coordinator_start_epoch"]
    journal.flush("coordinator_finished", status=journal.value["status"])
    return journal.value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget-seconds", type=int, default=7200)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--full-reference-dir", type=Path, required=True)
    parser.add_argument("--gpu-identity", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("child_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.child_command[:1] == ["--"]:
        args.child_command = args.child_command[1:]
    try:
        report = coordinate(args)
    except Exception as exc:
        print(json.dumps({"status": "argument_or_preflight_failure", "error_type": type(exc).__name__, "error": str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(report, allow_nan=False))
    return 0 if report["status"] in ("dry_run", "completed_comparison") else 1


if __name__ == "__main__":
    raise SystemExit(main())
