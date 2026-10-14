"""Independent B1 pool protocol only; no JAX/model imports or physics execution."""
from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.dev import benchmark_global_server_cpu as cpu
from scripts.dev import global_server_resources as resources


def options(tmp_path, workers=4, **changes):
    values = dict(mode="pool", cpu_budget=4, pool_workers=workers,
        pool_start_concurrency=1, pool_start_timeout_seconds=10,
        pool_command_timeout_seconds=5, pool_memory_limit_gib=None,
        pool_worker_address_space_gib=None, synthetic_smoke=True,
        synthetic_stdlib_only=True, sizes=[16, 64], warm_repeats=5,
        output_dir=tmp_path, memory_guard_gib=None)
    values.update(changes)
    return SimpleNamespace(**values)


def toy_scale():
    return SimpleNamespace(jax=SimpleNamespace(tree_util=SimpleNamespace(
        tree_map=lambda fn, *parts: fn(*parts))), np=SimpleNamespace(stack=list))


def recorder():
    return SimpleNamespace(result={}, phase=lambda *args, **kwargs: None)


def delayed_worker(connection, core, args, assigned):
    time.sleep(30)  # parent must interrupt this owned process well before then


def duplicate_worker(connection, core, args, assigned):
    connection.send({"ready": {"pid": os.getpid()}})
    while True:
        message = connection.recv()
        if message["command"] == "close":
            return
        connection.send({"completed_point_indices": [0] * len(assigned)})
        connection.send({"worker_CPU_sample": {"process_CPU_user_system_seconds_delta": 0}})


def nonresponsive_worker(connection, core, args, assigned):
    connection.send({"ready": {"pid": os.getpid()}})
    connection.recv()
    time.sleep(30)


def pool_cpus(count):
    if hasattr(os, "sched_getaffinity"):
        # Select physical cores on Linux; Windows synthetic records affinity unavailable.
        seen, cpus = set(), []
        for item in resources.physical_cores(sorted(os.sched_getaffinity(0))):
            identity = item["package_id"], item["physical_core_id"]
            if identity not in seen:
                seen.add(identity)
                cpus.append(item["logical_cpu"])
        if len(cpus) < count:
            pytest.skip("not enough physical cores for nonmodel scheduling smoke")
        return cpus[:count]
    return list(range(count))


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_spawn_unique_points_complete_two_windows_reset_and_CPU_accounting(tmp_path, workers):
    record = recorder()
    args = options(tmp_path, workers)
    runner = cpu.PoolRunner(args, record, list(range(64)), None, toy_scale(), pool_cpus(4))
    processes = list(runner.processes)
    try:
        ready = record.result["pool_workers"]
        assert len({item["pid"] for item in ready}) == workers
        assert sorted(i for item in ready for i in item["assigned_point_indices"]) == list(range(64))
        assert all(item["JAX_imported"] is False for item in ready)
        assert record.result["pool_resource_policy"]["controller_shares_worker0_cpu"]
        for size in (16, 64):
            _, sample = cpu.measured_run(runner, size,
                {"step_CPU_budget": 4, "parent_allocated_CPU_request_count": 48}, "pool")
            original = runner.collect(size)
            assert original == [cpu.synthetic_point_output(index) for index in range(size)]
            assert len(sample["worker_samples"]) == workers
            assert sample["total_pool_CPU_user_system_seconds_delta"] >= sample["worker_process_CPU_user_system_seconds_delta"]
            assert sample["parent_CPU_budget"] == 48
            for index, output in enumerate(original):
                assert output[1][1]["daily"][0] == output[0][0]["value"] + index + 2
                assert output[1][0]["value"] == index * 100 + 7 * (index + 1) + 7 * (index + 2)
            for _ in range(5):
                runner.run(size)
                assert runner.collect(size) == original
                assert runner.run(size, wrapper=True) == original
    finally:
        runner.close()
    assert all(not process.is_alive() for process in processes)


def test_serial_ready_is_a_startup_barrier(tmp_path, monkeypatch):
    original = cpu.PoolRunner.receive
    counts = []
    def receive(self, key, **kwargs):
        result = original(self, key, **kwargs)
        if key == "ready":
            counts.append((len(self.processes), len(result)))
        return result
    monkeypatch.setattr(cpu.PoolRunner, "receive", receive)
    runner = cpu.PoolRunner(options(tmp_path), recorder(), list(range(64)), None, toy_scale(), pool_cpus(4))
    runner.close()
    assert counts == [(1, 1), (2, 1), (3, 1), (4, 1)]


def test_explicit_start_concurrency_is_bounded(tmp_path):
    record = recorder()
    runner = cpu.PoolRunner(options(tmp_path, pool_start_concurrency=2), record,
        list(range(64)), None, toy_scale(), pool_cpus(4))
    runner.close()
    assert [item["worker_indices"] for item in record.result["pool_resource_policy"]["startup_batches"]] == [[0, 1], [2, 3]]


def test_startup_timeout_reclaims_every_started_process(tmp_path, monkeypatch):
    monkeypatch.setattr(cpu, "pool_worker", delayed_worker)
    active_before = {process.pid for process in multiprocessing.active_children()}
    started = time.perf_counter()
    record = recorder()
    with pytest.raises(TimeoutError, match="startup timeout"):
        cpu.PoolRunner(options(tmp_path, pool_start_timeout_seconds=0.15), record,
            list(range(64)), None, toy_scale(), pool_cpus(4))
    assert time.perf_counter() - started < 8
    assert {process.pid for process in multiprocessing.active_children()} <= active_before
    assert "timeout" in record.result["pool_resource_failure"]["reason"]


def test_RSS_abort_reclaims_resident_workers(tmp_path, monkeypatch):
    record = recorder()
    runner = cpu.PoolRunner(options(tmp_path, workers=2), record, list(range(64)), None, toy_scale(), pool_cpus(4))
    runner.args.pool_memory_limit_gib = 1
    monkeypatch.setattr(runner, "resource_snapshot", lambda: {"summed_RSS_bytes": 2 * 1024**3})
    with pytest.raises(RuntimeError, match="RSS threshold"):
        runner.run(64)
    assert all(not process.is_alive() for process in runner.processes)
    assert "RSS threshold" in record.result["pool_resource_failure"]["reason"]


def test_command_timeout_reclaims_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(cpu, "pool_worker", nonresponsive_worker)
    runner = cpu.PoolRunner(options(tmp_path, workers=2, pool_command_timeout_seconds=0.15), recorder(),
        list(range(64)), None, toy_scale(), pool_cpus(4))
    with pytest.raises(TimeoutError, match="completed_point_indices timeout"):
        runner.run(64)
    assert all(not process.is_alive() for process in runner.processes)


def test_sampled_memory_includes_controller_workers_and_virtual_space(tmp_path):
    for pid, rss in ((101, 100), (102, 200)):
        path = tmp_path / str(pid)
        path.mkdir()
        (path / "status").write_text(f"VmRSS: {rss} kB\nVmHWM: 500 kB\nVmSize: 700 kB\n")
    result = cpu.pool_process_memory([101, 102], tmp_path)
    assert result["summed_RSS_bytes"] == 300 * 1024
    assert all(item["VmHWM_bytes"] == 500 * 1024 for item in result["processes"])
    assert all(item["VmSize_bytes"] == 700 * 1024 for item in result["processes"])


def test_duplicate_task_ack_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(cpu, "pool_worker", duplicate_worker)
    runner = cpu.PoolRunner(options(tmp_path, workers=2), recorder(), list(range(64)), None, toy_scale(), pool_cpus(4))
    try:
        with pytest.raises(ValueError, match="every unique point"):
            runner.run(64)
    finally:
        runner.close()


def test_full_output_assemble_rejects_duplicate_indices_even_when_count_matches():
    runner = cpu.PoolRunner.__new__(cpu.PoolRunner)
    runner.scale = toy_scale()
    runner.outputs = [{0: "first"}, {0: "repeated"}]
    with pytest.raises(ValueError, match="omitted a point"):
        runner.assemble(2)


def test_CLI_48_plan_is_explicit_and_does_not_import_JAX(tmp_path):
    entry = Path(cpu.__file__)
    command = [sys.executable, "-B", str(entry), "--mode", "pool", "--cpu-budget", "48",
        "--sizes", "64", "--synthetic-smoke", "--dry-run", "--output-dir", str(tmp_path)]
    missing = subprocess.run(command, capture_output=True, text=True)
    assert missing.returncode != 0 and "explicit --pool-workers" in missing.stderr
    # Drop synthetic-smoke: this is CLI planning only, no workers start.
    command.remove("--synthetic-smoke")
    command += ["--pool-workers", "4", "--prepared-input", "frozen.pkl", "--gpu-reference-dir", "gpu", "--cpu-reference-dir", "scalar"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["pool_plan"]["pool_workers"] == 4
    assert plan["pool_plan"]["pool_start_concurrency"] == 1
    assert plan["actual_child_prejax_receipt"]["thread_budget"] == 1
    assert plan["JAX_imported"] is False and plan["model_execution"] is False


@pytest.mark.parametrize("workers", [0, 5, 49])
def test_worker_count_cannot_exceed_allocation(tmp_path, workers):
    args = cpu.parser().parse_args(["--mode", "pool", "--cpu-budget", "4", "--pool-workers", str(workers),
        "--synthetic-smoke", "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="worker count"):
        cpu.validate(args)


def test_physical_pool_requires_sampled_memory_threshold_and_exact_scalar_admission(tmp_path):
    command = ["--mode", "pool", "--cpu-budget", "4", "--sizes", "64",
        "--prepared-input", "frozen.pkl", "--gpu-reference-dir", "gpu", "--cpu-reference-dir", "scalar",
        "--output-dir", str(tmp_path)]
    args = cpu.parser().parse_args(command)
    with pytest.raises(ValueError, match="pool-memory-limit-gib"):
        cpu.validate(args)
    args = cpu.parser().parse_args(command + ["--pool-memory-limit-gib", "64",
        "--validation-mode", "exploratory", "--cpu-reference-role", "cpu-internal"])
    with pytest.raises(ValueError, match="saved-scalar admission provenance"):
        cpu.validate(args)
    args = cpu.parser().parse_args(command + ["--pool-memory-limit-gib", "64",
        "--validation-mode", "exploratory", "--cpu-reference-role", "cpu-internal",
        "--cpu-reference-admission-record", "new-pool-admission.json",
        "--expected-cpu-reference-admission-sha256", "a" * 64])
    cpu.validate(args)  # syntax admission only; production verifies all saved provenance
    assert args.pool_workers == 4


def test_pool48_admits_48_distinct_physical_cores_with_parent_billing(tmp_path):
    for core in range(48):
        path = tmp_path / f"cpu{core}" / "topology"
        path.mkdir(parents=True)
        (path / "physical_package_id").write_text("0")
        (path / "core_id").write_text(str(core))
        (path / "thread_siblings_list").write_text(str(core))
    environment = {"SLURM_JOB_ID": "synthetic", "SLURM_JOB_PARTITION": "cnall",
        "SLURM_CPUS_PER_TASK": "48", "SLURM_NTASKS": "1",
        "ORCHJAX_PARENT_CPU_COUNT": "48", "ORCHJAX_SLURM_ALLOC_TRES": "cpu=48,node=1"}
    result = resources.admit("pool", 48, environ=environment, allowed=list(range(48)),
        sysfs=tmp_path, hostname="synthetic-paid-host")
    assert result["selected_logical_cpus"] == list(range(48))
    assert result["billing_CPU_request_count"] == 48
    assert "shares worker0" in result["pool_controller_policy"]
