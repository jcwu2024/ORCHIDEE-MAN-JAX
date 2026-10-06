"""Exercise Bash argument expansion and the real Python dry-run entry only."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROJECT = "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX"
CODE = PROJECT + "/runtime/worktrees/global-throughput-cpu-reader-v2"
SCALAR = PROJECT + "/runtime/outputs/research/global_throughput/server-benchmark/cpu-performance-v2-B64-scalar/scalar"


def launch(cpu_count, **settings):
    bash = shutil.which("bash")
    assert bash, "Bash is required to verify real launcher expansion"
    env = os.environ.copy()
    for key in ("CPU1_METHODS", "CPU4_METHODS", "CPU_REFERENCE_DIR", "SLURM_CPUS_PER_TASK",
                "VALIDATION_MODE", "CPU_REFERENCE_ROLE", "BENCHMARK_SIZES", "SLURM_JOB_ID"):
        env.pop(key, None)
    env.update(CODE_DIR=CODE, RUN_ROOT=PROJECT + "/runtime/outputs/research/global_throughput/server-benchmark/launcher-test",
               **settings)
    script = ROOT / f"scripts/hpc/slurm_global_server_cpu_{cpu_count}core.sh"
    return subprocess.run([bash, str(script).replace("\\", "/"), "--dry-run"],
        env=env, text=True, capture_output=True, check=False)


def commands(result):
    assert result.returncode == 0, result.stderr
    return [shlex.split(line.removeprefix("command=")) for line in result.stdout.splitlines()
            if line.startswith("command=")]


def python_dry_run(command):
    entry = next(index for index, token in enumerate(command) if token.endswith("benchmark_global_server_cpu.py"))
    assert command[entry] == CODE + "/scripts/dev/benchmark_global_server_cpu.py"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/dev/benchmark_global_server_cpu.py"),
        *command[entry + 1:], "--dry-run"], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["model_execution"] is False and value["JAX_imported"] is False
    return value


def test_selected_B64_scalar_is_exactly_one_CPU_command():
    rows = commands(launch(1, CPU1_METHODS="scalar", BENCHMARK_SIZES="64",
        VALIDATION_MODE="exploratory", CPU_REFERENCE_ROLE="cpu-internal"))
    assert len(rows) == 1
    value = python_dry_run(rows[0])
    assert value["mode"] == "scalar" and value["CPU_budget"] == 1 and value["sizes"] == [64]
    assert value["warm_repeats"] == 5 and value["validation_mode"] == "exploratory"
    assert "--cpu-reference-dir" not in rows[0]
    assert rows[0][rows[0].index("--memory-guard-gib") + 1] == "24"


@pytest.mark.parametrize("cpu_count", [1, 4])
def test_B64_batch_only_uses_external_scalar_without_running_scalar_or_pool(cpu_count):
    settings = {f"CPU{cpu_count}_METHODS": "batch", "CPU_REFERENCE_DIR": SCALAR,
                "BENCHMARK_SIZES": "64", "VALIDATION_MODE": "exploratory", "CPU_REFERENCE_ROLE": "cpu-internal"}
    rows = commands(launch(cpu_count, **settings))
    assert len(rows) == 1
    value = python_dry_run(rows[0])
    assert value["mode"] == "batch" and value["CPU_budget"] == cpu_count and value["sizes"] == [64]
    assert rows[0][rows[0].index("--cpu-reference-dir") + 1] == SCALAR
    assert rows[0][rows[0].index("--memory-guard-gib") + 1] == "24"


def test_batch_only_without_external_reference_is_rejected():
    result = launch(1, CPU1_METHODS="batch", BENCHMARK_SIZES="64")
    assert result.returncode != 0 and "external complete CPU scalar snapshot" in result.stderr
    assert not any(line.startswith("command=") for line in result.stdout.splitlines())


def test_legacy_default_scalar_then_batch_remains_strict():
    rows = commands(launch(1))
    values = [python_dry_run(row) for row in rows]
    assert [value["mode"] for value in values] == ["scalar", "batch"]
    assert all(value["validation_mode"] == "strict" and value["sizes"] == [16, 64] for value in values)
    assert rows[1][rows[1].index("--cpu-reference-dir") + 1].endswith("launcher-test/scalar")


@pytest.mark.parametrize("settings", [
    {"CPU1_METHODS": "batch scalar"},
    {"CPU1_METHODS": "scalar", "BENCHMARK_SIZES": "64 16"},
    {"CPU1_METHODS": "scalar", "VALIDATION_MODE": "strict", "CPU_REFERENCE_ROLE": "cpu-internal"},
    {"CPU1_METHODS": "scalar", "SLURM_CPUS_PER_TASK": "4"},
])
def test_invalid_launcher_choices_never_emit_execution_command(settings):
    result = launch(1, **settings)
    assert result.returncode != 0
    assert not any(line.startswith("command=") for line in result.stdout.splitlines())
