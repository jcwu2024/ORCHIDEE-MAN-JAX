from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
COORDINATOR = ROOT / "scripts/hpc/global_cpu48_b64_coordinator.py"
LAUNCHER = ROOT / "scripts/hpc/slurm_global_server_cpu_48core.sh"
RUNNER = ROOT / "scripts/dev/benchmark_global_server_cpu.py"
EXPECTED_REF = "a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6"
EXPECTED_SCIENCE = "11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5"


def write_scontrol_evidence(tmp_path, record=None, *, stderr="", exitcode="0"):
    evidence = tmp_path / "scontrol-evidence"
    evidence.mkdir(exist_ok=True)
    fields = {"JobId": "synthetic-job", "JobState": "RUNNING", "Partition": "cnall",
        "NodeList": "test01", "NumNodes": "1", "NumTasks": "1", "CPUs/Task": "48",
        "NumCPUs": "48", "AllocTRES": "cpu=48,node=1,mem=192G"}
    fields.update(record or {})
    stdout_path = evidence / "stdout"
    stderr_path = evidence / "stderr"
    exitcode_path = evidence / "exitcode"
    stdout_path.write_text(" ".join(f"{key}={value}" for key, value in fields.items()) + "\n", encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    exitcode_path.write_text(exitcode + "\n", encoding="ascii")
    return {"ORCHJAX_SLURM_JOB_RECORD_STDOUT": str(stdout_path),
        "ORCHJAX_SLURM_JOB_RECORD_STDERR": str(stderr_path),
        "ORCHJAX_SLURM_JOB_RECORD_EXITCODE": str(exitcode_path)}


def allocation_env(tmp_path, *, record=None, stderr="", exitcode="0", **changes):
    env = os.environ.copy()
    env.update({"SLURM_JOB_ID": "synthetic-job", "SLURM_JOB_PARTITION": "cnall",
        "SLURM_JOB_NODELIST": "test01", "SLURM_JOB_NUM_NODES": "1",
        "SLURM_CPUS_PER_TASK": "48", "SLURM_NTASKS": "1", "SLURM_NNODES": "1",
        "ORCHJAX_PARENT_CPU_COUNT": "48",
        "ORCHJAX_SLURM_ALLOC_TRES": "cpu=48,node=1,mem=192G", "ORCHJAX_SLURM_ALLOC_TRES_SOURCE": "AllocTRES"})
    env.update(write_scontrol_evidence(tmp_path, record, stderr=stderr, exitcode=exitcode))
    env.update(changes)
    return env


def test_validate_allocation_is_a_standalone_subprocess_branch(tmp_path):
    result = subprocess.run([sys.executable, str(COORDINATOR), "--validate-allocation"],
        env=allocation_env(tmp_path), text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    allocation = json.loads(result.stdout)
    assert allocation["parent_cpu_count"] == 48
    assert allocation["allocated_tres_source"] == "AllocTRES"
    invalid = subprocess.run([sys.executable, str(COORDINATOR), "--validate-allocation"],
        env=allocation_env(tmp_path, ORCHJAX_SLURM_ALLOC_TRES="cpu=48,cpu=48"), text=True,
        capture_output=True, check=False)
    assert invalid.returncode != 0


@pytest.mark.parametrize(("record", "stderr", "exitcode", "message"), [
    ({}, "", "1", "did not exit successfully"),
    ({"JobState": "PENDING"}, "", "0", "JobState"),
    ({"AllocTRES": "cpu=47,node=1,mem=192G"}, "", "0", "exactly cpu=48"),
    ({"AllocTRES": "cpu=48,node=2,mem=192G"}, "", "0", "exactly cpu=48 and node=1"),
    ({"NumCPUs": "47"}, "", "0", "NumCPUs"),
    ({"NumTasks": "2"}, "", "0", "NumTasks"),
    ({"AllocTRES": "cpu=48,node=1", "TRES": "cpu=47,node=1"}, "", "0", "disagree"),
    ({"AllocTRES": "cpu=48,node=1,billing=48", "TRES": "cpu=48,node=1,billing=96"}, "", "0", "disagree on shared billing"),
    ({}, "scontrol warning\n", "0", "wrote stderr"),
])
def test_validate_allocation_rejects_bad_saved_scontrol_evidence(tmp_path, record, stderr, exitcode, message):
    from scripts.hpc.global_cpu48_b64_coordinator import validate_allocation

    with pytest.raises(ValueError, match=message):
        validate_allocation(allocation_env(tmp_path, record=record, stderr=stderr, exitcode=exitcode))


def test_validate_allocation_rejects_missing_and_duplicate_record_fields(tmp_path):
    from scripts.hpc.global_cpu48_b64_coordinator import validate_allocation

    env = allocation_env(tmp_path)
    Path(env["ORCHJAX_SLURM_JOB_RECORD_STDOUT"]).write_text("JobId=synthetic-job\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required"):
        validate_allocation(env)
    Path(env["ORCHJAX_SLURM_JOB_RECORD_STDOUT"]).write_text(
        "JobId=synthetic-job JobId=synthetic-job JobState=RUNNING Partition=cnall "
        "NodeList=test01 NumNodes=1 NumTasks=1 CPUs/Task=48 NumCPUs=48 AllocTRES=cpu=48,node=1\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JobId"):
        validate_allocation(env)


def test_validate_allocation_accepts_legacy_tres_only_for_running_record(tmp_path):
    from scripts.hpc.global_cpu48_b64_coordinator import validate_allocation

    allocation = validate_allocation(allocation_env(tmp_path,
        record={"AllocTRES": "", "TRES": "cpu=48,node=1,mem=192G"},
        ORCHJAX_SLURM_ALLOC_TRES="cpu=48,node=1,mem=192G", ORCHJAX_SLURM_ALLOC_TRES_SOURCE="TRES"))
    assert allocation["allocated_tres_source"] == "TRES"


def args(tmp_path, *, dry_run=False):
    return SimpleNamespace(
        budget_seconds=5400, reference_cap_seconds=2700, minimum_candidate_seconds=1800,
        expected_gpu_reference_sha256=EXPECTED_REF, expected_science_sha256=EXPECTED_SCIENCE,
        expected_code_commit="1" * 40, dry_run=dry_run, output_dir=tmp_path / "run",
        project_root=tmp_path, code_dir=ROOT, python=sys.executable, entry=RUNNER,
        prepared_input=tmp_path / "prepared.pkl", gpu_reference_dir=tmp_path / "gpu-reference",
        config=tmp_path / "config.yaml", partition="cnall", slurm_bin=Path("/slurm/bin"))


def test_step_commands_bind_one_thread_reference_and_native_single_device_batch(tmp_path):
    from scripts.hpc import global_cpu48_b64_coordinator as coordinator

    scalar, scalar_env = coordinator.step_command(args(tmp_path), "scalar")
    batch, batch_env = coordinator.step_command(args(tmp_path), "batch")
    assert "--cpus-per-task=1" in scalar and "--cpus-per-task=48" in batch
    assert "--cpu-bind=cores" in batch
    assert "SLURM_CPUS_PER_TASK" not in scalar_env and "SLURM_CPUS_PER_TASK" not in batch_env
    assert scalar_env["OMP_NUM_THREADS"] == "1"
    assert batch_env == {"OMP_NUM_THREADS": "48", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "true",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=true"}
    assert "--cpu-budget 48" in " ".join(batch)
    assert "--sizes 64" in " ".join(batch)


def test_bash_to_coordinator_to_actual_runner_dry_run_path():
    bash = __import__("shutil").which("bash")
    if not bash:
        pytest.skip("GNU Bash is required for launcher expansion")
    env = os.environ.copy()
    env.update({"EXPECTED_CODE_COMMIT": "1" * 40,
        "CODE_DIR": "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v2",
        "PYTHON": sys.executable, "ENTRY": str(RUNNER).replace("\\", "/"),
        "COORDINATOR": str(COORDINATOR).replace("\\", "/"),
        "SLURM_BIN": "/synthetic/slurm/bin",
        "PREPARED_INPUT": "/synthetic/prepared.pkl", "GPU_REFERENCE_DIR": "/synthetic/gpu-reference",
        "CONFIG": "/synthetic/config.yaml",
        "RUN_ROOT": "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-execution-v2-20261012/dry"})
    shell = subprocess.run([bash, str(LAUNCHER).replace("\\", "/"), "--dry-run"],
        env=env, text=True, capture_output=True, check=False)
    assert shell.returncode == 0, shell.stderr
    line = next(row for row in shell.stdout.splitlines() if row.startswith("coordinator="))
    import shlex
    command = shlex.split(line.removeprefix("coordinator="))
    result = subprocess.run(command, env={**env, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    dry = json.loads(result.stdout)
    assert dry["model_execution"] is False and dry["JAX_imported"] is False
    assert dry["expected_gpu_reference_sha256"] == EXPECTED_REF
    assert set(dry["runner_cli_dry_runs"]) == {"scalar", "batch"}
    scalar = dry["runner_cli_dry_runs"]["scalar"]["runner_result"]
    batch = dry["runner_cli_dry_runs"]["batch"]["runner_result"]
    assert (scalar["mode"], scalar["CPU_budget"], scalar["sizes"]) == ("scalar", 1, [64])
    assert (batch["mode"], batch["CPU_budget"], batch["sizes"]) == ("batch", 48, [64])
    assert dry["runner_cli_dry_runs"]["batch"]["command"].count("--cpu-reference-dir") == 1


def test_candidate_is_skipped_when_less_than_1800_seconds_remain(tmp_path, monkeypatch):
    from scripts.hpc import global_cpu48_b64_coordinator as coordinator

    project = tmp_path / "project"
    code_dir = project / "runtime/worktrees/cpu48"
    output = project / "runtime/outputs/research/global_throughput/cpu48-b64-execution-v2-20261012/run"
    code_dir.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(code_dir)], check=True)
    subprocess.run(["git", "config", "user.email", "codex@example.invalid"], cwd=code_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Codex Test"], cwd=code_dir, check=True)
    subprocess.run(["git", "commit", "--allow-empty", "-qm", "synthetic"], cwd=code_dir, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_dir, text=True).strip()
    monkeypatch.chdir(code_dir)
    ticks = iter([100.0, 100.0, 100.0, 3800.0, 3800.0])
    calls = []
    run_args = args(tmp_path)
    run_args.project_root = project
    run_args.code_dir = code_dir
    run_args.output_dir = output
    run_args.expected_code_commit = commit
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-job")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "cnall")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "48")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setenv("SLURM_NNODES", "1")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "test01")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")
    monkeypatch.setenv("ORCHJAX_PARENT_CPU_COUNT", "48")
    monkeypatch.setenv("ORCHJAX_SLURM_ALLOC_TRES", "cpu=48,node=1,mem=192G")
    monkeypatch.setenv("ORCHJAX_SLURM_ALLOC_TRES_SOURCE", "AllocTRES")
    for name, value in write_scontrol_evidence(tmp_path).items():
        monkeypatch.setenv(name, value)
    value = coordinator.coordinate(run_args, run_step=lambda command, log, timeout: (
        calls.append((command, timeout)) or {"returncode": 0, "timed_out": False, "wall_seconds": 10,
            "start_epoch": 100.0, "end_epoch": 110.0, "log": str(log)}),
        clock=lambda: next(ticks), snapshot=lambda: {"hostname": "synthetic"})
    assert value["status"] == "candidate_skipped_insufficient_window"
    assert len(calls) == 1 and value["steps"]["scalar"]["status"] == "completed"
