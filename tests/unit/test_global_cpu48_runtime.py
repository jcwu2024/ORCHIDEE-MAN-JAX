"""CPU48 admission and runtime accounting tests using only mocked OS observations."""
from __future__ import annotations

import sys

import pytest

from scripts.dev import benchmark_global_server_cpu as cpu
from scripts.dev import global_server_resources as resources


def fake_topology(root, logical_to_core):
    for logical, core in enumerate(logical_to_core):
        topology = root / f"cpu{logical}" / "topology"
        topology.mkdir(parents=True)
        siblings = ",".join(str(index) for index, other in enumerate(logical_to_core) if other == core)
        (topology / "physical_package_id").write_text("0", encoding="utf-8")
        (topology / "core_id").write_text(str(core), encoding="utf-8")
        (topology / "thread_siblings_list").write_text(siblings, encoding="utf-8")


def allocation_env(*, step=48, parent=48, tres="cpu=48"):
    return {"SLURM_JOB_ID": "123", "SLURM_JOB_PARTITION": "cnmix",
        "SLURM_CPUS_PER_TASK": str(step), "SLURM_NTASKS": "1",
        "ORCHJAX_PARENT_CPU_COUNT": str(parent), "ORCHJAX_SLURM_ALLOC_TRES": tres,
        "ORCHJAX_SLURM_ALLOC_TRES_SOURCE": "scontrol show job"}


def test_batch48_admission_separates_step_and_parent_billing_counts(tmp_path):
    fake_topology(tmp_path, list(range(48)))
    report = resources.admit("batch", 48, environ=allocation_env(), allowed=range(48),
        sysfs=tmp_path, hostname="cn-paid-01")
    assert report["step_CPU_budget"] == report["allocated_CPU_request_count"] == 48
    assert report["parent_allocated_CPU_request_count"] == report["billing_CPU_request_count"] == 48
    assert len(report["selected_logical_cpus"]) == 48
    assert report["CPU_count_scope"]


def test_small_step_keeps_parent_allocation_as_billing_count(tmp_path):
    fake_topology(tmp_path, list(range(48)))
    report = resources.admit("scalar", 1, environ=allocation_env(step=1, parent=48), allowed=[0],
        sysfs=tmp_path, hostname="cn-paid-01")
    assert report["step_CPU_budget"] == report["allocated_CPU_request_count"] == 1
    assert report["parent_allocated_CPU_request_count"] == report["billing_CPU_request_count"] == 48


def test_legacy_scalar_without_parent_env_uses_tres_for_parent_billing(tmp_path):
    fake_topology(tmp_path, list(range(48)))
    env = {key: value for key, value in allocation_env(step=1, parent=48).items()
        if key != "ORCHJAX_PARENT_CPU_COUNT"}
    report = resources.admit("scalar", 1, environ=env, allowed=[0], sysfs=tmp_path, hostname="cn-paid-01")
    assert report["step_CPU_budget"] == report["allocated_CPU_request_count"] == 1
    assert report["parent_allocated_CPU_request_count"] == report["billing_CPU_request_count"] == 48


def test_legacy_pool_without_parent_env_keeps_cpu4_tres_billing(tmp_path):
    fake_topology(tmp_path, list(range(48)))
    env = {key: value for key, value in allocation_env(step=4, parent=4, tres="cpu=4").items()
        if key != "ORCHJAX_PARENT_CPU_COUNT"}
    report = resources.admit("pool", 4, environ=env, allowed=range(4), sysfs=tmp_path, hostname="cn-paid-01")
    assert report["step_CPU_budget"] == report["allocated_CPU_request_count"] == 4
    assert report["parent_allocated_CPU_request_count"] == report["billing_CPU_request_count"] == 4


@pytest.mark.parametrize(("mode", "budget", "step", "tres", "message"), [
    ("scalar", 1, 1, "cpu=048", "one unambiguous cpu=N"),
    ("pool", 4, 4, "cpu=1", "smaller than the step"),
])
def test_legacy_admission_rejects_malformed_or_undersized_existing_tres(
        tmp_path, mode, budget, step, tres, message):
    fake_topology(tmp_path, list(range(48)))
    env = {key: value for key, value in allocation_env(step=step, parent=budget, tres=tres).items()
        if key != "ORCHJAX_PARENT_CPU_COUNT"}
    with pytest.raises(RuntimeError, match=message):
        resources.admit(mode, budget, environ=env, allowed=range(budget), sysfs=tmp_path, hostname="cn-paid-01")


@pytest.mark.parametrize("tres", ["", "cpu=48-49", "xcpu=48", "cpu=48,cpu=48", "cpu=048"])
def test_parent_tres_cpu_count_requires_one_exact_integer_field(tres):
    with pytest.raises(RuntimeError, match="one unambiguous cpu=N"):
        resources.allocated_tres_cpu_count(tres)


def test_parent_tres_mismatch_and_parent_smaller_than_step_fail_closed(tmp_path):
    fake_topology(tmp_path, list(range(48)))
    with pytest.raises(RuntimeError, match="match allocated TRES"):
        resources.admit("batch", 48, environ=allocation_env(parent=48, tres="cpu=47"),
            allowed=range(48), sysfs=tmp_path, hostname="cn-paid-01")
    with pytest.raises(RuntimeError, match="enclose the step"):
        resources.admit("batch", 48, environ=allocation_env(parent=32, tres="cpu=32"),
            allowed=range(48), sysfs=tmp_path, hostname="cn-paid-01")
    with pytest.raises(RuntimeError, match="explicit parent"):
        resources.admit("batch", 48, environ={key: value for key, value in allocation_env().items()
            if key != "ORCHJAX_PARENT_CPU_COUNT"}, allowed=range(48), sysfs=tmp_path, hostname="cn-paid-01")


def test_smt_logical_cpus_do_not_satisfy_48_physical_core_request(tmp_path):
    fake_topology(tmp_path, [index // 2 for index in range(48)])
    with pytest.raises(RuntimeError, match="unique physical cores"):
        resources.admit("batch", 48, environ=allocation_env(), allowed=range(48),
            sysfs=tmp_path, hostname="cn-paid-01")


@pytest.mark.parametrize(("mode", "budget"), [("scalar", 48), ("pool", 48), ("scalar", 4), ("pool", 1)])
def test_legacy_scalar_pool_budget_contract_remains_fixed(mode, budget, tmp_path):
    fake_topology(tmp_path, list(range(48)))
    with pytest.raises(ValueError):
        resources.admit(mode, budget, environ=allocation_env(step=budget, parent=48),
            allowed=range(48), sysfs=tmp_path, hostname="cn-paid-01")


def test_batch48_is_native_full64_only_and_legacy_budgets_remain_valid(tmp_path):
    valid = cpu.parser().parse_args(["--mode", "batch", "--cpu-budget", "48", "--sizes", "64",
        "--output-dir", str(tmp_path), "--synthetic-smoke"])
    cpu.validate(valid)
    for sizes in ((16,), (16, 64)):
        args = cpu.parser().parse_args(["--mode", "batch", "--cpu-budget", "48", "--sizes",
            *map(str, sizes), "--output-dir", str(tmp_path), "--synthetic-smoke"])
        with pytest.raises(ValueError, match="sizes64 only"):
            cpu.validate(args)
    for mode, budget in (("scalar", 1), ("pool", 4), ("batch", 1), ("batch", 4)):
        args = cpu.parser().parse_args(["--mode", mode, "--cpu-budget", str(budget),
            "--output-dir", str(tmp_path), "--synthetic-smoke"])
        cpu.validate(args)
    for mode in ("scalar", "pool"):
        args = cpu.parser().parse_args(["--mode", mode, "--cpu-budget", "48", "--sizes", "64",
            "--output-dir", str(tmp_path), "--synthetic-smoke"])
        with pytest.raises(ValueError):
            cpu.validate(args)


def test_expected_gpu_reference_sha256_is_optional_for_legacy_invocations(tmp_path):
    args = cpu.parser().parse_args(["--mode", "scalar", "--cpu-budget", "1", "--output-dir", str(tmp_path),
        "--synthetic-smoke"])
    assert args.expected_gpu_reference_sha256 is None
    cpu.validate(args)


def test_expected_gpu_reference_sha256_accepts_exact_lowercase_digest(tmp_path):
    digest = "a" * 64
    args = cpu.parser().parse_args(["--mode", "batch", "--cpu-budget", "48", "--sizes", "64",
        "--output-dir", str(tmp_path), "--synthetic-smoke", "--expected-gpu-reference-sha256", digest])
    cpu.validate(args)
    assert args.expected_gpu_reference_sha256 == digest


@pytest.mark.parametrize("digest", ["a" * 63, "a" * 65, "A" * 64, "g" * 64])
def test_expected_gpu_reference_sha256_rejects_malformed_digest(tmp_path, digest):
    args = cpu.parser().parse_args(["--mode", "scalar", "--cpu-budget", "1", "--output-dir", str(tmp_path),
        "--synthetic-smoke", "--expected-gpu-reference-sha256", digest])
    with pytest.raises(ValueError, match="exactly64 lowercase hexadecimal"):
        cpu.validate(args)


def bind_env(monkeypatch, *, flags="", omp=None):
    monkeypatch.delitem(sys.modules, "jax", raising=False)
    monkeypatch.setenv("XLA_FLAGS", flags)
    if omp is None:
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    else:
        monkeypatch.setenv("OMP_NUM_THREADS", omp)
    for name in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)
    affinities = []
    monkeypatch.setattr(resources.os, "sched_setaffinity", lambda pid, mask: affinities.append((pid, set(mask))),
        raising=False)
    monkeypatch.setattr(resources.os, "sched_getaffinity", lambda pid: set(range(48)), raising=False)
    return affinities


def test_batch48_bind_uses_one_native_cpu_device_and_declared_thread_policy(monkeypatch):
    affinities = bind_env(monkeypatch, flags="--xla_force_host_platform_device_count=1 --xla_cpu_enable_fast_math=false")
    resources.bind(list(range(48)), one_thread=False)
    assert affinities == [(0, set(range(48)))]
    assert "--xla_cpu_multi_thread_eigen=true" in resources.os.environ["XLA_FLAGS"]
    assert resources.os.environ["OMP_NUM_THREADS"] == "48"
    assert resources.os.environ["MKL_NUM_THREADS"] == "1"
    assert resources.os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert resources.os.environ["NUMEXPR_NUM_THREADS"] == "1"


@pytest.mark.parametrize(("flags", "omp", "message"), [
    ("--xla_force_host_platform_device_count=2", None, "one native CPU device"),
    ("", "4", "different OMP thread budget"),
    ("--xla_cpu_multi_thread_eigen=false", None, "single-thread"),
    ("intra_op_parallelism_threads=1", None, "batch4"),
])
def test_batch48_bind_rejects_inherited_single_thread_or_multidevice_policy(monkeypatch, flags, omp, message):
    bind_env(monkeypatch, flags=flags, omp=omp)
    with pytest.raises(RuntimeError, match=message):
        resources.bind(list(range(48)), one_thread=False)


def test_memory_snapshot_reads_fake_proc_and_marks_missing_data_unavailable(tmp_path):
    (tmp_path / "self").mkdir()
    (tmp_path / "self" / "status").write_text("Name:\tpython\nVmRSS:\t120 kB\nVmHWM:\t250 kB\n", encoding="utf-8")
    (tmp_path / "meminfo").write_text("MemTotal: 4096 kB\nMemAvailable: 2048 kB\n", encoding="utf-8")
    snapshot = cpu.memory_snapshot("before", tmp_path)
    assert snapshot["available"] is True
    assert snapshot["RSS_bytes"] == 120 * 1024
    assert snapshot["peak_RSS_bytes"] == 250 * 1024
    assert snapshot["MemAvailable_bytes"] == 2048 * 1024
    (tmp_path / "meminfo").unlink()
    missing = cpu.memory_snapshot("after", tmp_path)
    assert missing["available"] is False and missing["RSS_bytes"] == 120 * 1024
    assert missing["MemAvailable_bytes"] is None and missing["unavailable_reason"]


def test_process_cpu_sample_reports_busy_cores_and_both_utilization_denominators():
    sample = cpu.process_cpu_sample((2.0, 1.0), (8.0, 4.0), 2.0, 48, 64, mode="batch")
    assert sample["process_CPU_user_seconds_delta"] == 6.0
    assert sample["process_CPU_system_seconds_delta"] == 3.0
    assert sample["busy_core_equivalents"] == 4.5
    assert sample["step_utilization_fraction"] == 4.5 / 48
    assert sample["parent_utilization_fraction"] == 4.5 / 64
    assert "not proof" in sample["claim"]
    pool = cpu.process_cpu_sample((0.0, 0.0), (2.0, 1.0), 1.0, 4, 48, mode="pool")
    assert "excludes pool worker" in pool["scope"]


@pytest.mark.parametrize("wall", [0.0, -1.0, float("inf"), float("nan")])
def test_process_cpu_sample_rejects_nonfinite_or_nonpositive_wall(wall):
    with pytest.raises(RuntimeError, match="invalid process CPU"):
        cpu.process_cpu_sample((0.0, 0.0), (1.0, 0.0), wall, 48, 48, mode="batch")


def test_process_cpu_sample_rejects_parent_budget_smaller_than_step():
    with pytest.raises(RuntimeError, match="invalid process CPU"):
        cpu.process_cpu_sample((0.0, 0.0), (1.0, 0.0), 1.0, 48, 47, mode="batch")


def test_measured_run_wraps_runner_without_changing_invocation(monkeypatch):
    times = iter([(0.0, 0.0), (3.0, 1.0)])
    clocks = iter([10.0, 12.0])
    monkeypatch.setattr(cpu.os, "times", lambda: next(times))
    monkeypatch.setattr(cpu.time, "perf_counter", lambda: next(clocks))

    class Runner:
        calls = []

        def run(self, size, *, wrapper=False):
            self.calls.append((size, wrapper))
            return "output" if wrapper else None

    runner = Runner()
    value, sample = cpu.measured_run(runner, 64,
        {"step_CPU_budget": 48, "parent_allocated_CPU_request_count": 64}, "batch", wrapper=True)
    assert value == "output" and runner.calls == [(64, True)]
    assert sample["process_CPU_user_system_seconds_delta"] == 4.0
    assert sample["step_CPU_budget"] == 48 and sample["parent_CPU_budget"] == 64
