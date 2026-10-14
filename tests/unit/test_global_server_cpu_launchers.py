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
                "VALIDATION_MODE", "CPU_REFERENCE_ROLE", "BENCHMARK_SIZES", "SLURM_JOB_ID",
                "POOL_WORKERS", "POOL_START_CONCURRENCY", "POOL_START_TIMEOUT_SECONDS",
                "POOL_COMMAND_TIMEOUT_SECONDS", "POOL_MEMORY_LIMIT_GIB", "POOL_MEMORY_GUARD_GIB",
                "POOL_WORKER_ADDRESS_SPACE_GIB", "CACHE_MODE", "CACHE_ROOT",
                "CPU_REFERENCE_ADMISSION_RECORD", "EXPECTED_CPU_REFERENCE_ADMISSION_SHA256", "EXPECTED_GPU_REFERENCE_SHA256"):
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
    arguments = [sys.executable, str(ROOT / "scripts/dev/benchmark_global_server_cpu.py"),
        *command[entry + 1:], "--dry-run"]
    if any(token.endswith("global_cpu_child_environment.py") for token in command):
        arguments = [sys.executable, str(ROOT / "scripts/dev/global_cpu_child_environment.py"),
            "--thread-budget", "1", "--", *arguments]
    result = subprocess.run(arguments, text=True, capture_output=True, check=False)
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


def test_pool_default_is_one_independent_scalar_command_and_no_node_pin():
    rows = commands(launch(4, CPU_REFERENCE_DIR=SCALAR,
        CPU_REFERENCE_ADMISSION_RECORD=PROJECT + "/runtime/outputs/new-pool-scalar-admission.json",
        EXPECTED_CPU_REFERENCE_ADMISSION_SHA256="a" * 64,
        EXPECTED_GPU_REFERENCE_SHA256="b" * 64))
    assert len(rows) == 1
    value = python_dry_run(rows[0])
    assert value["mode"] == "pool" and value["CPU_budget"] == 4 and value["sizes"] == [64]
    assert value["actual_child_prejax_receipt"]["thread_budget"] == 1
    assert value["pool_plan"]["pool_workers"] == 4
    assert value["pool_plan"]["pool_start_concurrency"] == 1
    assert value["pool_plan"]["pool_memory_limit_gib"] == 64
    assert value["pool_plan"]["pool_worker_address_space_gib"] is None
    assert rows[0][rows[0].index("--cache-mode") + 1] == "persistent"
    assert rows[0][rows[0].index("--cache-root") + 1].endswith("launcher-test/scalar-worker-cache")
    assert "--nodelist" not in (ROOT / "scripts/hpc/slurm_global_server_cpu_4core.sh").read_text()


def test_pool_dry_run_command_dispatches_only_to_local_srun_and_python_stubs(tmp_path):
    # Execute the expanded command under Bash, replacing scheduler and Python
    # binaries with record-only stubs. No server connection or model import.
    row = commands(launch(4, CPU_REFERENCE_DIR=SCALAR, CPU_REFERENCE_ROLE="accepted",
        VALIDATION_MODE="strict", POOL_WORKERS="2"))[0]
    trace = tmp_path / "arguments.txt"
    srun = tmp_path / "srun-stub"
    python = tmp_path / "python-stub"
    srun.write_text('#!/bin/bash\nset -euo pipefail\n'
        '[[ $1 == --ntasks=1 && $2 == --cpus-per-task=4 && $3 == --cpu-bind=cores ]]\n'
        'shift 3\n[[ $1 == /usr/bin/time && $2 == -v ]]\nshift 2\nexec "$@"\n')
    python.write_text('#!/bin/bash\nset -euo pipefail\nprintf "%s\\n" "$@" > "$POOL_STUB_TRACE"\n')
    srun.chmod(0o755)
    python.chmod(0o755)
    command = [str(srun).replace("\\", "/") if index == 0 else
        str(python).replace("\\", "/") if value.endswith("/.venvs/orcjax_cpu/bin/python") else value
        for index, value in enumerate(row)]
    env = {**os.environ, "POOL_STUB_TRACE": str(trace).replace("\\", "/")}
    result = subprocess.run([shutil.which("bash"), "-c", "exec " + shlex.join(command)],
        env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    recorded = trace.read_text().splitlines()
    assert recorded[0].endswith("global_cpu_child_environment.py")
    assert recorded[1:4] == ["--thread-budget", "1", "--"]
    assert recorded[recorded.index("--pool-workers") + 1] == "2"
    assert recorded[recorded.index("--pool-start-concurrency") + 1] == "1"
    assert recorded[recorded.index("--pool-memory-limit-gib") + 1] == "64"


@pytest.mark.parametrize("workers", ["1", "2", "4"])
def test_selected_pool_workers_remain_inside_four_core_allocation(workers):
    rows = commands(launch(4, CPU_REFERENCE_DIR=SCALAR, CPU_REFERENCE_ROLE="accepted", VALIDATION_MODE="strict",
        POOL_WORKERS=workers, POOL_START_CONCURRENCY="1"))
    value = python_dry_run(rows[0])
    assert value["pool_plan"]["pool_workers"] == int(workers)
    assert value["CPU_budget"] == 4


@pytest.mark.parametrize("settings", [
    {"POOL_WORKERS": "5"}, {"POOL_WORKERS": "1", "POOL_START_CONCURRENCY": "2"},
    {"CPU4_METHODS": "pool batch"}, {"CPU_REFERENCE_ROLE": "cpu-internal", "VALIDATION_MODE": "exploratory"},
])
def test_invalid_pool_launch_does_not_emit_command(settings):
    result = launch(4, CPU_REFERENCE_DIR=SCALAR, **settings)
    assert result.returncode != 0
    assert not any(line.startswith("command=") for line in result.stdout.splitlines())
