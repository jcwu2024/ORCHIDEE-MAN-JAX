"""Record the committed local candidate and static future argv; execute no model."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.dev import global_lifecycle_science as science  # noqa: E402


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def checked_json(command, environment):
    completed = subprocess.run(
        command, cwd=ROOT, env=environment, text=True, encoding="utf-8", capture_output=True, check=True
    )
    value = json.loads(completed.stdout)
    if value.get("status") != "dry_run" or value.get("JAX_imported") or value.get("model_execution"):
        raise ValueError("pure local dry-run contract failed")
    return {"command": command, "returncode": completed.returncode, "report": value, "stderr": completed.stderr}


def main():
    sha = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    if branch != "research/global-throughput" or git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("candidate requires the clean fixed global research branch")
    remote = git("ls-remote", "--heads", "origin", "refs/heads/research/global-throughput")
    if remote.split()[0] != sha:
        raise ValueError("pushed branch does not match the candidate SHA")
    source = science.source_identity(ROOT, expected_code_sha=sha, require_clean=True)

    project = "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX"
    code = project + "/runtime/worktrees/global-throughput-gpu-lifecycle-v1"
    run = project + "/runtime/outputs/research/global_throughput/server-benchmark/gpu-lifecycle-20261011-<new-job>"
    log_parent = project + "/runtime/logs"
    log_pattern = log_parent + "/global_gpu_lifecycle_%j.txt"
    references = project + "/runtime/outputs/research/global_throughput/gpu-round4/model"
    arguments = [
        "--prepared-input",
        project
        + "/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl",
        "--full-reference-dir",
        references,
        "--memory-baseline-dir",
        project
        + "/runtime/outputs/research/global_throughput/server-benchmark/global-throughput-v2-2b14506/gpu-b64/candidate-first",
        "--b1024-evidence-dir",
        project + "/runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed1024-v1/capacity",
        "--config",
        code + "/configs/orchidee_man_250919.yaml",
        "--asset-root",
        project + "/runtime/outputs/research/global_throughput/round4-inputs/selected64-assets-v2",
        "--output-dir",
        "__ARM_OUTPUT_DIR__",
        "--warm-repeats",
        "5",
        "--budget-seconds",
        "7200",
        "--stage-shape",
        "1024",
        "--measurement-strategy",
        "__MEASUREMENT_STRATEGY__",
        "--expected-code-sha",
        sha,
    ]
    child = [
        "/rmprog/slurm/v22.05.7/bin/srun",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--cpu-bind=cores",
        "/usr/bin/time",
        "-v",
        "singularity",
        "exec",
        "--nv",
        "--bind",
        "<canonical-cuda-nvml-driver-binds>",
        "--pwd",
        code,
        "/apps/soft/sif/foundationpose",
        project + "/.venvs/orcjax_gpu/bin/python",
        code + "/scripts/hpc/global_batch_gpu_entry.py",
        *arguments,
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "ORCHJAX_GLOBAL_BATCH_ENTRY": "benchmark_global_gpu_lifecycle.py",
            "SINGULARITYENV_ORCHJAX_GLOBAL_BATCH_ENTRY": "benchmark_global_gpu_lifecycle.py",
        }
    )
    coordinator = checked_json(
        [
            sys.executable,
            "-B",
            str(ROOT / "scripts/hpc/global_gpu_lifecycle_coordinator.py"),
            "--output-dir",
            run,
            "--expected-code-sha",
            sha,
            "--code-dir",
            code,
            "--full-reference-dir",
            references,
            "--budget-seconds",
            "7200",
            "--dry-run",
            "--",
            *child,
        ],
        environment,
    )
    runner_checks = {}
    for strategy in science.STRATEGIES:
        arm_arguments = list(arguments)
        arm_arguments[arm_arguments.index("__ARM_OUTPUT_DIR__")] = run + "/" + strategy + "/capacity/B1024"
        arm_arguments[arm_arguments.index("__MEASUREMENT_STRATEGY__")] = strategy
        planned = coordinator["report"]["commands"][strategy]
        if planned[16:] != arm_arguments:
            raise ValueError("coordinator failed canonical future argv/path expansion")
        runner_checks[strategy] = checked_json(
            [
                sys.executable,
                "-B",
                str(ROOT / "scripts/dev/benchmark_global_gpu_lifecycle.py"),
                *arm_arguments,
                "--dry-run",
            ],
            environment,
        )

    preparation = {}
    for name in (
        "source-runner-validation.json",
        "coordinator-launcher-validation.json",
        "reference-format-compatibility.json",
        "runtime-reference-container-compatibility.json",
    ):
        digest, size = science.digest_file(HERE / name)
        preparation[name] = {"sha256": digest, "bytes": size}
    submission = [
        "/rmprog/slurm/v22.05.7/bin/sbatch",
        "--partition=gnall",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--gres=gpu:1",
        "--time=02:00:00",
        "--no-requeue",
        "--chdir=" + code,
        "--output=" + log_pattern,
        "--error=" + log_pattern,
        "--nodelist=<freshly-reviewed-node>",
        code + "/scripts/hpc/slurm_global_gpu_lifecycle.sh",
    ]
    receipt = {
        "schema": "global-gpu-lifecycle-execution-candidate-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "local_preparation_complete_execution_unapproved",
        "local_cwd": str(ROOT),
        "branch": branch,
        "code_sha": sha,
        "remote_sha": remote.split()[0],
        "remote_matches": True,
        "tracked_and_nonignored_untracked_clean": True,
        "source_identity_clean_admission": source,
        "comparison_reference": science.comparison_admission(),
        "contract": science.CONTRACT,
        "process_strategy": science.PROCESS_STRATEGY,
        "stage_name": science.STAGE_NAME,
        "preparation_evidence": preparation,
        "local_runtime": {"python": sys.executable, "JAX_imported": "jax" in sys.modules},
        "local_pure_coordinator_dry_run": coordinator,
        "local_pure_runner_dry_runs": runner_checks,
        "future_plan": {
            "code_dir": code,
            "output_root_pattern": run,
            "child_commands": coordinator["report"]["commands"],
            "cpu_parent_python": project + "/.venvs/orcjax_cpu/bin/python",
            "cpu_parent_entry": code + "/scripts/hpc/global_gpu_lifecycle_coordinator.py",
            "sbatch_argv": submission,
            "submission_environment": {"CODE_DIR": code, "EXPECTED_CODE_SHA": sha},
            "child_environment": {
                "ORCHJAX_GLOBAL_BATCH_ENTRY": "benchmark_global_gpu_lifecycle.py",
                "SINGULARITYENV_ORCHJAX_GLOBAL_BATCH_ENTRY": "benchmark_global_gpu_lifecycle.py",
                "JAX_ENABLE_X64": "true",
                "SINGULARITYENV_JAX_ENABLE_X64": "true",
                "SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "allocator": "default",
                "allocation_window_epoch": "one actual parent epoch propagated to both children",
            },
            "resources": {
                "partition": "gnall",
                "nodes": 1,
                "tasks": 1,
                "V100_32GB_count": 1,
                "Slurm_CPUs": 4,
                "wall_seconds": 7200,
                "requeue": False,
                "exclusive": False,
            },
            "cost": {
                "primary_worst_case_CNY": 4.80,
                "conservative_worst_case_CNY": 4.96,
                "verified_bill": False,
                "execution_approved": False,
            },
            "slurm_stdout": log_pattern,
            "slurm_stderr": log_pattern,
            "before_submission": {
                "shared_log_parent": log_parent,
                "required": "Verify this shared directory exists and is writable before sbatch; Slurm opens logs before the script's mkdir.",
                "performed": False,
            },
            "unresolved_target_values": [
                "freshly reviewed node",
                "canonical host CUDA/NVML bind paths",
                "new Slurm job ID",
            ],
            "deployed": False,
            "submitted": False,
        },
        "limitations": [
            "Static canonical argv validation on local Windows, not target Linux/Slurm/driver/runtime validation.",
            "Local source admission and pure CLI dry-runs read no prepared/reference arrays and execute no model/JAX/GPU.",
            "No canonical JAX0.4.38/CUDA, new-core FULL64, allocator recovery, memory reduction, or throughput acceptance.",
        ],
    }
    if receipt["local_runtime"]["JAX_imported"]:
        raise ValueError("candidate generation unexpectedly imported JAX")
    path = HERE / "EXECUTION_CANDIDATE.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "receipt": str(path),
                "code_sha": sha,
                "remote_matches": True,
                "runner_dry_runs": len(runner_checks),
                "JAX_imported": False,
                "model_execution": False,
            }
        )
    )


if __name__ == "__main__":
    main()
