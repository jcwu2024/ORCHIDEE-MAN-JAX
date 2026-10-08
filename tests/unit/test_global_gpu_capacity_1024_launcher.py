"""Actual Bash fixed1024 argument expansion; no model preparation or GPU startup."""
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
def test_fixed1024_actual_Bash_wrapper_and_Python_dryrun(tmp_path, spooled):
    bash = shutil.which("bash")
    project = subprocess.run([bash, "-c", 'cd "$1" && pwd', "path", str(tmp_path).replace("\\", "/")],
        text=True, capture_output=True, check=True).stdout.strip()
    declared = tmp_path / "runtime/worktrees/declared-code/scripts/hpc"
    declared.mkdir(parents=True)
    for name in ("slurm_global_gpu_capacity.sh", "slurm_global_gpu_capacity_1024.sh"):
        source = (ROOT / "scripts/hpc" / name).read_text().replace(
            "PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX", f"PROJECT={shlex.quote(project)}")
        (declared / name).write_text(source)
    wrapper = declared / "slurm_global_gpu_capacity_1024.sh"
    if spooled:
        spool = tmp_path / "slurm-spool/job-not-submitted"
        spool.mkdir(parents=True)
        wrapper = spool / "slurm_script"
        shutil.copyfile(declared / "slurm_global_gpu_capacity_1024.sh", wrapper)
        assert not (spool / "slurm_global_gpu_capacity.sh").exists()
    env = os.environ.copy()
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES", "CONTINUE_512",
                "CONTINUE_256_512", "FIXED_1024", "OPTIONAL_B64_BASELINE", "RUN_ROOT"):
        env.pop(key, None)
    env["CODE_DIR"] = project + "/runtime/worktrees/declared-code"
    result = subprocess.run([bash, str(wrapper).replace("\\", "/"), "--dry-run"],
        env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "cpus=4 gpus=1 wall=01:30:00" in result.stdout
    tokens = shlex.split(next(row for row in result.stdout.splitlines() if row.startswith("command=")).split("=", 1)[1])
    entry = next(i for i, token in enumerate(tokens) if token.endswith("global_batch_gpu_entry.py"))
    args = tokens[entry + 1:]
    assert tokens[entry] == env["CODE_DIR"] + "/scripts/hpc/global_batch_gpu_entry.py"
    assert "--fixed-1024" in args
    assert not {"--continue-512", "--continue-256-512", "--conditional-512", "--optional-b64-baseline"}.intersection(args)
    assert args[args.index("--b512-evidence-dir") + 1].endswith("gpu-capacity-512-15657822/capacity")
    assert args[args.index("--output-dir") + 1] == project + "/runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed1024-v1/capacity"
    dry = subprocess.run([sys.executable, str(ROOT / "scripts/dev/benchmark_global_gpu_capacity.py"), *args, "--dry-run"],
        cwd=tmp_path, text=True, capture_output=True)
    assert dry.returncode == 0, dry.stderr
    report = json.loads(dry.stdout)
    assert report["contract"] == "GPU-fixed1024-repeated-FULL64-B512-B1024-capacity-v1"
    assert report["candidates"] == [512, 1024]
    assert report["total_lanes"] == 1024 and report["lane_days"] == 14336
    assert report["budget_seconds"] == 5400
    assert report["JAX_imported"] is False and report["model_execution"] is False
    assert not (tmp_path / "runtime/outputs").exists()


@pytest.mark.parametrize("changes", [
    {"CONTINUE_512": "1"}, {"CONTINUE_256_512": "1"},
    {"OPTIONAL_B64_BASELINE": "1"}, {"CONDITIONAL_512": "1"}, {"FIXED_1024": "2"},
])
def test_shared_launcher_rejects_conflicting_fixed1024_modes(changes):
    env = {**os.environ, "FIXED_1024": "1", "CONTINUE_512": "0", "CONTINUE_256_512": "0",
        "OPTIONAL_B64_BASELINE": "0", "CONDITIONAL_512": "0", **changes}
    result = subprocess.run([shutil.which("bash"), str(ROOT / "scripts/hpc/slurm_global_gpu_capacity.sh").replace("\\", "/"), "--dry-run"],
        env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "command=" not in result.stdout
