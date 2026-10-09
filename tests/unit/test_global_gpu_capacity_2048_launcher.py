"""Actual Bash direct/spooled expansion for the new fresh-process capacity contract."""
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("spooled", [False, True])
def test_fixed2048_actual_wrapper_and_coordinator_plan(tmp_path, spooled):
    bash = shutil.which("bash")
    project = subprocess.run([bash, "-c", 'cd "$1" && pwd', "path", str(tmp_path).replace("\\", "/")],
        text=True, capture_output=True, check=True).stdout.strip()
    declared = tmp_path / "runtime/worktrees/declared-code/scripts/hpc"
    declared.mkdir(parents=True)
    for name in ("slurm_global_gpu_capacity.sh", "slurm_global_gpu_capacity_2048.sh"):
        source = (ROOT / "scripts/hpc" / name).read_text().replace(
            "PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX", f"PROJECT={shlex.quote(project)}")
        (declared / name).write_text(source)
    wrapper = declared / "slurm_global_gpu_capacity_2048.sh"
    if spooled:
        spool = tmp_path / "slurm-spool/job-not-submitted"
        spool.mkdir(parents=True)
        wrapper = spool / "slurm_script"
        shutil.copyfile(declared / "slurm_global_gpu_capacity_2048.sh", wrapper)
    env = os.environ.copy()
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES", "CONTINUE_512",
                "CONTINUE_256_512", "FIXED_1024", "FIXED_2048", "OPTIONAL_B64_BASELINE", "RUN_ROOT",
                "XLA_PYTHON_CLIENT_ALLOCATOR", "SINGULARITYENV_XLA_PYTHON_CLIENT_ALLOCATOR"):
        env.pop(key, None)
    env["CODE_DIR"] = project + "/runtime/worktrees/declared-code"
    result = subprocess.run([bash, str(wrapper).replace("\\", "/"), "--dry-run"],
        env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "cpus=4 gpus=1 wall=02:00:00" in result.stdout
    assert "allocator=default preallocate=false" in result.stdout
    tokens = shlex.split(next(row for row in result.stdout.splitlines() if row.startswith("command=")).split("=", 1)[1])
    coordinator = next(i for i, token in enumerate(tokens) if token.endswith("global_gpu_capacity_2048_coordinator.py"))
    assert tokens[coordinator] == env["CODE_DIR"] + "/scripts/hpc/global_gpu_capacity_2048_coordinator.py"
    assert tokens[coordinator - 1].endswith(".venvs/orcjax_cpu/bin/python")
    command = tokens[coordinator + 1:]
    assert command[command.index("--budget-seconds") + 1] == "7200"
    assert "--fixed-2048" in command and "--b1024-evidence-dir" in command
    assert not {"--fixed-1024", "--continue-512", "--continue-256-512", "--conditional-512", "--optional-b64-baseline"}.intersection(command)
    dry = subprocess.run([sys.executable, str(ROOT / "scripts/hpc/global_gpu_capacity_2048_coordinator.py"),
        "--dry-run", *command], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert dry.returncode == 0, dry.stderr
    report = json.loads(dry.stdout)
    assert "2048" in report["contract"]
    assert not (tmp_path / "runtime/outputs").exists()


@pytest.mark.parametrize("changes", [
    {"FIXED_1024": "1"}, {"CONTINUE_512": "1"}, {"CONTINUE_256_512": "1"},
    {"OPTIONAL_B64_BASELINE": "1"}, {"CONDITIONAL_512": "1"}, {"FIXED_2048": "2"},
    {"XLA_PYTHON_CLIENT_ALLOCATOR": "platform"}, {"SINGULARITYENV_XLA_PYTHON_CLIENT_ALLOCATOR": "platform"},
])
def test_fixed2048_shared_launcher_rejects_conflicts(changes):
    env = {**os.environ, "FIXED_2048": "1", "FIXED_1024": "0", "CONTINUE_512": "0",
        "CONTINUE_256_512": "0", "OPTIONAL_B64_BASELINE": "0", "CONDITIONAL_512": "0", **changes}
    result = subprocess.run([shutil.which("bash"), str(ROOT / "scripts/hpc/slurm_global_gpu_capacity.sh").replace("\\", "/"), "--dry-run"],
        env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "command=" not in result.stdout
