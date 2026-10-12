"""CPU-only admission tests for the separately scoped paid affine launcher."""
from __future__ import annotations

import datetime
import ctypes
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from scripts.hpc import run_shared_latent_T7_batch_numerics_paid as paid
from scripts.hpc import shared_affine_gpu_admission as shared
from scripts.hpc import shared_affine_cuda_identity as cuda_identity
from scripts.hpc import check_daily_training_runtime as runtime_check


class FakeCudaCall:
    def __init__(self, callback, result=0):
        self.callback = callback
        self.result = result

    def __call__(self, *args):
        if self.result:
            return self.result
        return self.callback(*args)


class FakeCudaDriver:
    def __init__(self, *, count=1, uuid_bytes=None, error=None, legacy_uuid_api=False):
        uuid_bytes = uuid_bytes or bytes.fromhex("0123456789abcdef0123456789abcdef")
        self.cuInit = FakeCudaCall(lambda *_: 0, result=error == "cuInit" and 1 or 0)

        def write_count(pointer):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = count
            return 0

        def write_device(pointer, ordinal):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = ordinal
            return 0

        def write_uuid(pointer, _device):
            target = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_ubyte * 16)).contents
            target[:] = uuid_bytes
            return 0

        self.cuDeviceGetCount = FakeCudaCall(write_count, result=error == "cuDeviceGetCount" and 1 or 0)
        self.cuDeviceGet = FakeCudaCall(write_device, result=error == "cuDeviceGet" and 1 or 0)
        uuid_name = "cuDeviceGetUuid" if legacy_uuid_api else "cuDeviceGetUuid_v2"
        setattr(self, uuid_name, FakeCudaCall(write_uuid, result=error == uuid_name and 1 or 0))


def allocation_fixture(monkeypatch):
    started = int(time.time()) - 60
    monkeypatch.setattr(paid.time, "time", lambda: started + 60)
    start_text = datetime.datetime.fromtimestamp(started, datetime.timezone.utc).isoformat(timespec="seconds")
    end_text = datetime.datetime.fromtimestamp(started + 1200, datetime.timezone.utc).isoformat(timespec="seconds")
    job = paid.parse_job(
        "JobId=742001 JobState=RUNNING Partition=gnall NodeList=ibc13b03n04 "
        "NumNodes=1 NumCPUs=4 NumTasks=1 Requeue=0 TimeLimit=00:20:00 "
        "AllocTRES=cpu=4,mem=16G,node=1,gres/gpu=1,gres/gpu:v100=1 "
        "GresDetail=gpu:v100:1(IDX:3) GRES=gpu:v100:1(IDX:3) OverSubscribe=OK "
        "RunTime=00:01:00 StartTime=" + start_text + " EndTime=" + end_text
    )
    environment = {
        "SLURM_JOB_ID": "742001",
        "SLURM_CPUS_PER_TASK": "4",
        "SLURM_NTASKS": "1",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_JOB_PARTITION": "gnall",
        "SLURM_JOB_START_TIME": str(started),
        "SLURM_JOB_END_TIME": str(started + 1200),
        "SLURM_JOB_GPUS": "3",
        "ORCJAX_SLURM_VISIBLE_DEVICES": "0",
    }
    scope = {
        "resources": dict(paid.RESOURCES),
        "rates": dict(paid.RATES),
        "approved_cost_ceiling_cny": "1.00",
    }
    return scope, environment, job


def scope_json(tmp_path, monkeypatch, *, status="proposed"):
    identities = {"operand": "operand-fixture", "receipt": "receipt-fixture", "capture_source": "source-fixture"}
    source_files = {path: f"fixture-sha-{index}" for index, path in enumerate(paid.SOURCE_PATHS)}
    output = tmp_path / "paid-result.json"
    ledger = tmp_path / "paid-one-shot.json"
    scope = {
        "schema_version": paid.SCHEMA,
        "scope_id": paid.SCOPE_ID,
        "gpu_execution_contract": paid.GPU_EXECUTION_CONTRACT,
        "os_device_open_isolation_required": False,
        "status": status,
        "execution_authorized": False,
        "approval_note": "",
        "code_git_head": None,
        "source_files": source_files,
        "caps": paid.free.CAPS,
        "resources": paid.RESOURCES,
        "rates": paid.RATES,
        "quote": paid.quote(),
        "approved_cost_ceiling_cny": "1.00",
        "repeats": 3,
        "cards": 1,
        "nonexclusive": True,
        "output": str(output),
        "ledger": str(ledger),
        "management_admission": shared.MANAGEMENT,
        "shared_mutex": shared.SCOPE_LOCK.as_posix(),
        "input_sha256": identities,
    }
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(scope), encoding="utf-8")
    source_manifest_snapshot = dict(source_files)
    monkeypatch.setattr(paid.free, "admit_capture", lambda args: (identities, {
        "current_execution": {"git_head": "fixture-head"},
    }))
    monkeypatch.setattr(paid, "source_manifest", lambda **_kwargs: dict(source_manifest_snapshot))
    args = SimpleNamespace(
        operands=tmp_path / "operands.npz", capture_receipt=tmp_path / "capture.json",
        capture_source=tmp_path / "capture-source.py", scope_json=path,
        output=output, ledger=ledger, dry_run=True, shell_started_epoch=None,
    )
    return args, scope, path, output, ledger


def lock_worker(lock_path, start_event, release_event, result_queue):
    lock = shared.AffineScopeLock(lock_path, "parallel-test")
    start_event.wait(10)
    try:
        lock.acquire()
    except ValueError:
        result_queue.put("busy")
        return
    result_queue.put("acquired")
    release_event.wait(10)
    lock.release(release_verified=True)


def lock_owner_with_child(lock_path, child_dir, result_queue):
    shared.AffineScopeLock(lock_path, "owner-death-test").acquire()
    child_code = """
import os, sys, time
from pathlib import Path
root = Path(sys.argv[1])
heartbeat = root / 'heartbeat'
release = root / 'release-child'
finished = root / 'child-finished'
deadline = time.monotonic() + 10
while not release.exists() and time.monotonic() < deadline:
    heartbeat.write_text(str(time.monotonic_ns()), encoding='ascii')
    time.sleep(0.05)
finished.write_text(str(os.getpid()), encoding='ascii')
"""
    child_env = os.environ.copy()
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    child = subprocess.Popen(
        [sys.executable, "-B", "-c", child_code, child_dir],
        env=child_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    result_queue.put(child.pid)
    child.wait(timeout=10)


def test_parse_job_preserves_values_and_rejects_duplicate_keys():
    fields = paid.parse_job(
        'JobId=12 Name="bounded affine" AllocTRES=cpu=4,gres/gpu=1 GresDetail=gpu:1(IDX:3)'
    )
    assert fields == {
        "JobId": "12", "Name": "bounded affine", "AllocTRES": "cpu=4,gres/gpu=1",
        "GresDetail": "gpu:1(IDX:3)",
    }
    with pytest.raises(ValueError, match="duplicate Slurm job field"):
        paid.parse_job("JobId=12 JobId=13")


@pytest.mark.parametrize(("duration", "seconds"), [
    ("00:20:00", 1200), ("1-00:00:02", 86402), ("00:00:00", 0),
])
def test_slurm_duration_parser(duration, seconds):
    assert paid.seconds(duration) == seconds


@pytest.mark.parametrize("duration", ["UNLIMITED", "00:60:00", "00:00:60", "-1:00:00"])
def test_slurm_duration_parser_rejects_unbounded_or_malformed_values(duration):
    with pytest.raises(ValueError, match="Slurm duration"):
        paid.seconds(duration)


def test_exact_paid_allocation_and_quote_are_admitted(monkeypatch):
    scope, environment, job = allocation_fixture(monkeypatch)
    result = paid.validate_allocation(scope, environment, job, "ibc13b03n04.cluster")
    assert result["actual_NumCPUs"] == 4
    assert result["actual_AllocTRES"]["gres/gpu:v100"] == "1"
    assert result["allocation_TRES_fields"] == ["AllocTRES"]
    assert result["allocation_end_epoch"] - result["allocation_start_epoch"] == 1200
    assert result["original_visible_devices"] == "0"
    assert result["quote"]["display_ceiling_cny"] == "0.83"
    assert result["quote"]["proposed_authorization_ceiling_cny"] == "1.00"


@pytest.mark.parametrize("change", [
    "job_id", "state", "partition", "node", "cpus", "tasks", "requeue", "wall",
    "tres_gpu", "tres_cpu", "tres_typed_gpu", "hostname", "environment_cpus", "partition_environment",
    "tres_multiple_typed_gpu", "gres_fields_missing", "gres_mismatch", "gres_alias_conflict",
    "gres_alias_type_conflict", "gres_count", "gres_multiple_idx",
    "sharing_no", "sharing_yes", "runtime_reserve",
    "start_binding", "end_binding", "cost_ceiling",
])
def test_paid_allocation_rejects_scope_or_live_resource_drift(change, monkeypatch):
    scope, environment, job = allocation_fixture(monkeypatch)
    job = dict(job)
    environment = dict(environment)
    if change == "job_id":
        job["JobId"] = "742002"
    elif change == "state":
        job["JobState"] = "PENDING"
    elif change == "partition":
        job["Partition"] = "cnmix"
    elif change == "node":
        job["NodeList"] = "ibc13b03n02"
    elif change == "cpus":
        job["NumCPUs"] = "8"
    elif change == "tasks":
        job["NumTasks"] = "2"
    elif change == "requeue":
        job["Requeue"] = "1"
    elif change == "wall":
        job["TimeLimit"] = "00:30:00"
    elif change == "tres_gpu":
        job["AllocTRES"] = "cpu=4,mem=16G,node=1,gres/gpu=2"
    elif change == "tres_cpu":
        job["AllocTRES"] = "cpu=8,mem=16G,node=1,gres/gpu=1"
    elif change == "tres_typed_gpu":
        job["AllocTRES"] = "cpu=4,mem=16G,node=1,gres/gpu=1,gres/gpu:v100=2"
    elif change == "tres_multiple_typed_gpu":
        job["AllocTRES"] = "cpu=4,mem=16G,node=1,gres/gpu=1,gres/gpu:v100=1,gres/gpu:a100=1"
    elif change == "gres_fields_missing":
        job.pop("GresDetail")
        job.pop("GRES")
    elif change == "gres_mismatch":
        job["GRES"] = "gpu:v100:1(IDX:4)"
        job["GresDetail"] = "gpu:v100:1(IDX:4)"
    elif change == "gres_alias_conflict":
        job["GRES"] = "gpu:v100:1(IDX:4)"
    elif change == "gres_alias_type_conflict":
        job["GRES"] = "gpu:a100:1(IDX:3)"
    elif change == "gres_count":
        job["GRES"] = "gpu:v100:2(IDX:3)"
        job["GresDetail"] = "gpu:v100:2(IDX:3)"
    elif change == "gres_multiple_idx":
        job["GRES"] = "gpu:v100:1(IDX:3),gpu:v100:1(IDX:4)"
        job["GresDetail"] = "gpu:v100:1(IDX:3),gpu:v100:1(IDX:4)"
    elif change == "sharing_no":
        job["OverSubscribe"] = "NO"
    elif change == "sharing_yes":
        job["OverSubscribe"] = "YES"
    elif change == "hostname":
        host = "ibc13b03n02"
    elif change == "environment_cpus":
        environment["SLURM_CPUS_PER_TASK"] = "8"
    elif change == "partition_environment":
        environment["SLURM_JOB_PARTITION"] = "cnmix"
    elif change == "runtime_reserve":
        job["RunTime"] = "00:17:00"
    elif change == "start_binding":
        environment["SLURM_JOB_START_TIME"] = "1"
    elif change == "end_binding":
        environment["SLURM_JOB_END_TIME"] = str(int(environment["SLURM_JOB_END_TIME"]) - 1)
    elif change == "cost_ceiling":
        scope["approved_cost_ceiling_cny"] = "0.50"
    if change != "hostname":
        host = "ibc13b03n04"
    with pytest.raises(ValueError):
        paid.validate_allocation(scope, environment, job, host)


@pytest.mark.parametrize("tres", ["cpu=4,cpu=4,gres/gpu=1", "cpu=4,broken"])
def test_tres_parser_rejects_duplicate_or_malformed_entries(tres):
    with pytest.raises(ValueError):
        paid.tres(tres)


def test_allocated_tres_uses_legacy_tres_only_with_running_resource_records(monkeypatch):
    scope, environment, job = allocation_fixture(monkeypatch)
    legacy = dict(job)
    legacy["TRES"] = legacy.pop("AllocTRES")
    legacy.update(Nodes=paid.NODE, CPU_IDs="0-3", GRES="gpu:v100:1(IDX:3)")
    accepted, sources = paid.allocated_tres(legacy)
    assert accepted["cpu"] == "4" and accepted["gres/gpu"] == "1"
    assert sources == ["TRES"]
    validated = paid.validate_allocation(scope, environment, legacy, "ibc13b03n04.cluster")
    assert validated["allocation_TRES_fields"] == ["TRES"]
    legacy["JobState"] = "PENDING"
    with pytest.raises(ValueError, match="legacy TRES lacks actual single-node allocation detail"):
        paid.allocated_tres(legacy)


@pytest.mark.parametrize("fields", [
    {"ReqTRES": "cpu=4,node=1,gres/gpu=1"},
    {"AllocTRES": "cpu=4,node=1,gres/gpu=1", "TRES": "cpu=4,node=1,gres/gpu=2"},
])
def test_allocated_tres_rejects_missing_or_conflicting_actual_fields(fields):
    with pytest.raises(ValueError, match="actual allocated TRES|conflicting actual allocated TRES"):
        paid.allocated_tres(fields)


def test_allocated_tres_accepts_identical_aliases_and_arbitrary_gpu_type_in_synthetic_job(monkeypatch):
    scope, environment, job = allocation_fixture(monkeypatch)
    job["TRES"] = job["AllocTRES"]
    job["GresDetail"] = "gpu:synthetic_type-1:1(IDX:3)"
    job["GRES"] = "gpu:synthetic_type-1:1(IDX:3)"
    job.update(Nodes=paid.NODE, CPU_IDs="0-3")
    result = paid.validate_allocation(scope, environment, job, "ibc13b03n04.cluster")
    assert result["allocation_TRES_fields"] == ["AllocTRES", "TRES"]


def test_slurm_epoch_helper_queries_numeric_job_and_uses_scheduler_timestamps(monkeypatch, capsys):
    _, _, job = allocation_fixture(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "742001")
    monkeypatch.delenv("SLURM_JOB_START_TIME", raising=False)
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    calls = []

    def scheduler(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=" ".join(f"{key}={value}" for key, value in job.items()))

    monkeypatch.setattr(paid.subprocess, "run", scheduler)
    assert paid.scheduler_allocation_epochs() == 0
    assert calls == [(
        [paid.SLURM, "show", "job", "--details", "-o", "742001"],
        {"capture_output": True, "text": True, "timeout": 4, "check": True},
    )]
    start, end = map(int, capsys.readouterr().out.split())
    assert end - start == 1200


@pytest.mark.parametrize("job_id", [None, "", "bad-job-id"])
def test_slurm_epoch_helper_rejects_missing_or_non_numeric_job_id_before_query(monkeypatch, job_id):
    if job_id is None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    else:
        monkeypatch.setenv("SLURM_JOB_ID", job_id)
    monkeypatch.setattr(paid.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("scheduler queried"))
    with pytest.raises(ValueError, match="numeric actual Slurm job ID"):
        paid.scheduler_allocation_epochs()


def test_scheduler_allocation_timestamps_work_without_22_05_environment_tokens(monkeypatch):
    scope, environment, job = allocation_fixture(monkeypatch)
    environment.pop("SLURM_JOB_START_TIME")
    environment.pop("SLURM_JOB_END_TIME")
    result = paid.validate_allocation(scope, environment, job, "ibc13b03n04.cluster")
    assert result["allocation_clock_source"].startswith("scheduler StartTime/EndTime")


@pytest.mark.parametrize("change", [
    "bad_start", "missing_end", "bad_end", "wrong_duration", "future", "expired",
    "bad_original_start", "bad_original_end",
])
def test_scheduler_allocation_epochs_reject_malformed_or_unsafe_clocks(monkeypatch, change):
    _, environment, job = allocation_fixture(monkeypatch)
    job = dict(job)
    environment = dict(environment)
    if change not in ("bad_original_start", "bad_original_end"):
        environment.pop("SLURM_JOB_START_TIME")
        environment.pop("SLURM_JOB_END_TIME")
    if change == "bad_start":
        job["StartTime"] = "N/A"
    elif change == "missing_end":
        job.pop("EndTime")
    elif change == "bad_end":
        job["EndTime"] = "2026-99-99T22:58:11"
    elif change == "wrong_duration":
        start_epoch = int(datetime.datetime.fromisoformat(job["StartTime"]).timestamp())
        job["EndTime"] = datetime.datetime.fromtimestamp(start_epoch + 1199, datetime.timezone.utc).isoformat(timespec="seconds")
    elif change == "future":
        start_epoch = int(paid.time.time()) + 1
        job["StartTime"] = datetime.datetime.fromtimestamp(start_epoch, datetime.timezone.utc).isoformat(timespec="seconds")
        job["EndTime"] = datetime.datetime.fromtimestamp(start_epoch + 1200, datetime.timezone.utc).isoformat(timespec="seconds")
    elif change == "expired":
        start_epoch = int(paid.time.time()) - 1200
        job["StartTime"] = datetime.datetime.fromtimestamp(start_epoch, datetime.timezone.utc).isoformat(timespec="seconds")
        job["EndTime"] = datetime.datetime.fromtimestamp(start_epoch + 1200, datetime.timezone.utc).isoformat(timespec="seconds")
    elif change == "bad_original_start":
        environment["SLURM_JOB_START_TIME"] = "1700000000.5"
    elif change == "bad_original_end":
        environment["SLURM_JOB_END_TIME"] = str(int(environment["SLURM_JOB_END_TIME"]) - 1)
    with pytest.raises(ValueError):
        paid.allocation_epochs(environment, job)


@pytest.mark.parametrize("absent", ["SLURM_JOB_START_TIME", "SLURM_JOB_END_TIME"])
def test_scheduler_allocation_epoch_environment_crosschecks_are_independently_optional(monkeypatch, absent):
    scope, environment, job = allocation_fixture(monkeypatch)
    environment.pop(absent)
    result = paid.validate_allocation(scope, environment, job, "ibc13b03n04.cluster")
    assert result["allocation_end_epoch"] - result["allocation_start_epoch"] == 1200


def test_actual_slurm22_cpu_job_record_parses_and_is_rejected_for_paid_gpu_scope(monkeypatch):
    fixture = Path(__file__).resolve().parents[1] / "fixtures/slurm22/actual_cpu_job.v1.txt"
    job = paid.parse_job(fixture.read_text(encoding="ascii"))
    assert job["JobId"] == "15779975"
    assert job["JobState"] == "RUNNING"
    assert job["TRES"] == "cpu=48,node=1,billing=48"
    assert job["GRES"] == ""
    assert job["OverSubscribe"] == "OK"
    scope, environment, _ = allocation_fixture(monkeypatch)
    environment.update(
        SLURM_JOB_ID=job["JobId"], SLURM_JOB_PARTITION=job["Partition"],
        SLURM_JOB_NUM_NODES=job["NumNodes"], SLURM_CPUS_PER_TASK="48",
    )
    with pytest.raises(ValueError, match="actual Slurm job/node/resources differ"):
        paid.validate_allocation(scope, environment, job, job["NodeList"])


def test_physical_slurm_gpu_is_not_confused_with_visible_ordinal():
    environment = {"SLURM_JOB_GPUS": "3", "ORCJAX_SLURM_VISIBLE_DEVICES": "0"}
    rows = [{
        "index": "3", "uuid": "GPU-physical-three", "name": "Tesla V100-SXM2",
        "minor_number": "5", "device_major": "195", "device_minor": "5",
        "control_devices": [{"major": 195, "minor": 255}],
    }]
    mapping = paid.admit_gpu_mapping(environment, rows, "c 195:5 rwm\nc 195:255 rwm\n")
    assert mapping["gpu_host_index"] == 3
    assert mapping["gpu_uuid"] == "GPU-physical-three"
    assert mapping["original_CUDA_VISIBLE_DEVICES"] == "0"
    assert mapping["child_CUDA_VISIBLE_DEVICES"] == "GPU-physical-three"
    assert mapping["child_visible_ordinal"] == 0
    assert mapping["actual_GPU_major"] == 195
    assert mapping["actual_GPU_minor"] == 5
    assert mapping["cgroup_GPU_minors"] == [5]


@pytest.mark.parametrize(("environment", "rows", "rules"), [
    ({"SLURM_JOB_GPUS": "0,1", "ORCJAX_SLURM_VISIBLE_DEVICES": "0"},
     [{"index": "0", "uuid": "GPU-a", "name": "Tesla V100", "minor_number": "0", "device_major": "195", "device_minor": "0", "control_devices": []}], "c 195:0 rwm"),
    ({"SLURM_JOB_GPUS": "3", "ORCJAX_SLURM_VISIBLE_DEVICES": "0"},
     [{"index": "2", "uuid": "GPU-a", "name": "Tesla V100", "minor_number": "5", "device_major": "195", "device_minor": "5", "control_devices": []}], "c 195:5 rwm"),
    ({"SLURM_JOB_GPUS": "3", "ORCJAX_SLURM_VISIBLE_DEVICES": "0"},
     [{"index": "3", "uuid": "GPU-a", "name": "Tesla A100", "minor_number": "5", "device_major": "195", "device_minor": "5", "control_devices": []}], "c 195:5 rwm"),
    ({"SLURM_JOB_GPUS": "3", "ORCJAX_SLURM_VISIBLE_DEVICES": "0"},
     [{"index": "3", "uuid": "GPU-a", "name": "Tesla V100", "minor_number": "5", "device_major": "195", "device_minor": "5", "control_devices": []}], "a *:* rwm"),
    ({"SLURM_JOB_GPUS": "3", "ORCJAX_SLURM_VISIBLE_DEVICES": "GPU-wrong"},
     [{"index": "3", "uuid": "GPU-a", "name": "Tesla V100", "minor_number": "5", "device_major": "195", "device_minor": "5", "control_devices": []}], "c 195:5 rwm"),
    ({"SLURM_JOB_GPUS": "3", "ORCJAX_SLURM_VISIBLE_DEVICES": "0"},
     [{"index": "3", "uuid": "GPU-a", "name": "Tesla V100", "minor_number": "5", "device_major": "195", "device_minor": "5", "control_devices": []}], "c 195:4 rwm"),
    ({"SLURM_JOB_GPUS": "3", "ORCJAX_SLURM_VISIBLE_DEVICES": ""},
     [{"index": "3", "uuid": "GPU-a", "name": "Tesla V100", "minor_number": "5", "device_major": "195", "device_minor": "5", "control_devices": []}], "c 195:5 rwm"),
    ({"SLURM_JOB_GPUS": "3", "ORCJAX_SLURM_VISIBLE_DEVICES": "0"},
     [{"index": "3", "uuid": "GPU-a", "name": "Tesla V100", "minor_number": "4", "device_major": "195", "device_minor": "5", "control_devices": []}], "c 195:5 rwm"),
    ({"SLURM_JOB_GPUS": "3", "ORCJAX_SLURM_VISIBLE_DEVICES": "0"},
     [{"index": "3", "uuid": "GPU-a", "name": "Tesla V100", "minor_number": "5", "device_major": "195", "device_minor": "4", "control_devices": []}], "c 195:5 rwm"),
    ({"SLURM_JOB_GPUS": "3", "ORCJAX_SLURM_VISIBLE_DEVICES": "0"},
     [{"index": "3", "uuid": "GPU-a", "name": "Tesla V100", "minor_number": "5", "device_major": "195", "device_minor": "5", "control_devices": []}], "c *:5 rwm"),
    ({"SLURM_JOB_GPUS": "3", "ORCJAX_SLURM_VISIBLE_DEVICES": "0"},
     [{"index": "3", "uuid": "GPU-a", "name": "Tesla V100", "minor_number": "5", "device_major": "195", "device_minor": "5", "control_devices": []}], "c 195:* rwm"),
])
def test_gpu_admission_rejects_ambiguous_cgroup_or_mapping(environment, rows, rules):
    with pytest.raises(ValueError):
        paid.admit_gpu_mapping(environment, rows, rules)


def test_read_cgroup_rules_binds_the_expected_slurm_job(tmp_path):
    job_id = "742001"
    base = tmp_path / "devices"
    membership = "/slurm/uid_1000/job_742001/step_batch"
    devices = base / membership.lstrip("/") / "devices.list"
    devices.parent.mkdir(parents=True)
    devices.write_text("c 195:5 rwm\n", encoding="ascii")
    proc = tmp_path / "proc-self-cgroup"
    proc.write_text("5:devices:" + membership + "\n", encoding="ascii")

    path, rules = paid.read_cgroup_rules(job_id, proc_path=proc, cgroup_base=base)

    assert Path(path) == devices.resolve()
    assert rules == "c 195:5 rwm\n"
    with pytest.raises(ValueError, match="different actual Slurm job"):
        paid.read_cgroup_rules("742002", proc_path=proc, cgroup_base=base)
    with pytest.raises(ValueError, match="numeric actual Slurm job ID"):
        paid.read_cgroup_rules("not-a-job", proc_path=proc, cgroup_base=base)


def test_cuda_driver_identity_reads_uuid_without_a_gpu_runtime():
    identity = cuda_identity.driver_identity(FakeCudaDriver())
    assert identity == {
        "driver_device_count": 1,
        "driver_visible_ordinal": 0,
        "driver_gpu_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
        "driver_uuid_api": "cuDeviceGetUuid_v2",
    }
    legacy = cuda_identity.driver_identity(FakeCudaDriver(legacy_uuid_api=True))
    assert legacy["driver_uuid_api"] == "cuDeviceGetUuid"


@pytest.mark.parametrize("api", ["cuInit", "cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetUuid_v2"])
def test_cuda_driver_identity_fails_closed_on_driver_api_error(api):
    with pytest.raises(RuntimeError, match=api):
        cuda_identity.driver_identity(FakeCudaDriver(error=api))


def test_cuda_driver_identity_requires_exactly_one_device():
    with pytest.raises(ValueError, match="exactly one device"):
        cuda_identity.driver_identity(FakeCudaDriver(count=2))


def test_expected_cuda_gpu_binds_pjrt_environment_and_driver_uuid(monkeypatch):
    expected = "GPU-01234567-89ab-cdef-0123-456789abcdef"
    device = SimpleNamespace(platform="gpu", local_hardware_id=0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", expected)
    identity = {
        "driver_device_count": 1,
        "driver_visible_ordinal": 0,
        "driver_gpu_uuid": expected,
        "driver_uuid_api": "mock",
    }
    result = cuda_identity.verify_expected_gpu([device], expected, identity_provider=lambda: identity)
    assert result["passed"] is True
    assert result["pjrt_local_hardware_id"] == 0
    assert result["expected_gpu_uuid"] == expected

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        cuda_identity.verify_expected_gpu([device], expected, identity_provider=lambda: identity)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", expected)
    with pytest.raises(ValueError, match="hardware ordinal zero"):
        cuda_identity.verify_expected_gpu(
            [SimpleNamespace(platform="gpu", local_hardware_id=1)], expected,
            identity_provider=lambda: identity,
        )
    with pytest.raises(ValueError, match="driver UUID differs"):
        cuda_identity.verify_expected_gpu(
            [device], expected,
            identity_provider=lambda: dict(identity, driver_gpu_uuid="GPU-ffffffff-ffff-ffff-ffff-ffffffffffff"),
        )


def test_runtime_preflight_optional_cuda_identity_hook(tmp_path, monkeypatch):
    expected = "GPU-01234567-89ab-cdef-0123-456789abcdef"
    output = tmp_path / "runtime-check.json"
    devices = [SimpleNamespace(platform="gpu", local_hardware_id=0)]
    jax = SimpleNamespace(default_backend=lambda: "gpu", __version__="0.4.38")
    monkeypatch.setattr(runtime_check, "_initialize_runtime", lambda _backend: (jax, object(), devices))
    monkeypatch.setattr(runtime_check, "check_runtime", lambda _jax, _jnp: {"status": "passed"})
    calls = []

    def verify(observed_devices, expected_uuid, **_kwargs):
        calls.append((observed_devices, expected_uuid))
        return {"passed": True, "expected_gpu_uuid": expected_uuid}

    monkeypatch.setattr(cuda_identity, "verify_expected_gpu", verify)
    monkeypatch.delenv("ORCJAX_AFFINE_EXPECTED_GPU_UUID", raising=False)
    assert runtime_check.main(["--backend", "cpu", "--output", str(output)]) == 0
    absent = json.loads(output.read_text(encoding="utf-8"))
    assert "affine_gpu_identity" not in absent
    assert calls == []

    monkeypatch.setenv("ORCJAX_AFFINE_EXPECTED_GPU_UUID", expected)
    assert runtime_check.main(["--backend", "gpu", "--output", str(output)]) == 0
    present = json.loads(output.read_text(encoding="utf-8"))
    assert present["affine_gpu_identity"]["expected_gpu_uuid"] == expected
    assert calls == [(devices, expected)]


def test_runtime_preflight_rejects_paid_binding_on_cpu_backend(tmp_path, monkeypatch):
    output = tmp_path / "runtime-check.json"
    monkeypatch.setenv("ORCJAX_AFFINE_EXPECTED_GPU_UUID", "GPU-01234567-89ab-cdef-0123-456789abcdef")
    monkeypatch.setattr(runtime_check, "_initialize_runtime", lambda _backend: (SimpleNamespace(), object(), []))
    monkeypatch.setattr(runtime_check, "check_runtime", lambda *_: {"status": "must not run"})

    assert runtime_check.main(["--backend", "cpu", "--output", str(output)]) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["error"] == "paid physical GPU binding requires GPU backend"


def test_paid_scope_dry_run_is_unique_and_one_shot(tmp_path, monkeypatch):
    args, scope, scope_path, output, ledger = scope_json(tmp_path, monkeypatch)
    plan = paid.prepare(args)
    assert plan["GPU_executed"] is False
    assert plan["readiness"].startswith("paid CPU preparation only")
    assert plan["planned_measurements"] == 12
    assert plan["scope"]["status"] == "proposed"
    assert plan["source_files"] == scope["source_files"]
    assert plan["commands"]["probe"][-4:] == ["--repeats", "3", "--seed", "20261011"]
    assert str(args.operands.resolve()) in plan["commands"]["probe"]
    assert not ledger.exists() and not output.exists()

    ledger.write_text("already admitted", encoding="utf-8")
    with pytest.raises(FileExistsError, match="one-shot already consumed"):
        paid.prepare(args)
    ledger.unlink()
    output.with_suffix(".job-attempt.json").write_text("consumed", encoding="utf-8")
    with pytest.raises(FileExistsError, match="scheduler attempt already consumed"):
        paid.prepare(args)


def test_candidate_paid_scope_uses_posix_runtime_paths():
    scope = paid.proposed_scope()
    management = scope["management_admission"]
    assert management == shared.MANAGEMENT
    assert management["launch_owner"] == "daily-affine-execution-owner"
    assert management["paid_dispatch_only"] is True
    assert management["free_dispatch_paused"] is True
    assert management["free_approval_preserved"] is True
    assert management["known_free_checkout"] == shared.FREE_CHECKOUT.as_posix()
    assert management["known_free_head"] == shared.FREE_HEAD
    assert management["known_free_scope"] == shared.FREE_SCOPE.as_posix()
    assert management["known_free_ledger"] == shared.FREE_LEDGER.as_posix()
    for name in ("output", "ledger", "shared_mutex"):
        assert scope[name].startswith("/WORK/liwei_work/jcwu/")
        assert "\\" not in scope[name]


def free_admission_fixture(tmp_path):
    checkout = tmp_path / "known-free-checkout"
    output = tmp_path / "known-free-output"
    checkout.mkdir()
    output.mkdir()
    return checkout, output, output / "scope.v1.json", output / "free.ledger.json"


def test_read_free_admission_requires_exact_head_and_only_reports_absent_known_evidence(tmp_path, monkeypatch):
    checkout, output, scope_path, ledger_path = free_admission_fixture(tmp_path)
    calls = []

    def check_output(command, **kwargs):
        calls.append((command, kwargs))
        return shared.FREE_HEAD

    monkeypatch.setattr(shared.subprocess, "check_output", check_output)
    result = shared.read_free_admission(
        checkout=checkout, output=output, scope_path=scope_path, ledger_path=ledger_path,
    )
    assert result["status"] == "known_free_unopened_unreserved"
    assert result["read_only"] is True
    assert result["known_free_head"] == shared.FREE_HEAD
    assert result["free_approval_preserved"] is True
    assert result["free_dispatch_paused_by_owner"] is True
    assert result["absent_evidence_paths"] == [
        scope_path.as_posix(), ledger_path.as_posix(),
        (output / "result.v1.json").as_posix(),
        (output / "result.v1.outer-timeout.json").as_posix(),
        (output / "result.v1.runtime-preflight.json").as_posix(),
        (output / "result.v1.supervisor.json").as_posix(),
        (output / "result.v1.results.npz").as_posix(),
        (output / "result.v1.partial.npz").as_posix(),
    ]
    assert calls == [(["git", "rev-parse", "HEAD"], {"cwd": checkout, "timeout": 4, "text": True})]
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("evidence_name", [
    "scope.v1.json", "free.ledger.json", "result.v1.json",
    "result.v1.outer-timeout.json", "result.v1.runtime-preflight.json",
    "result.v1.supervisor.json", "result.v1.results.npz", "result.v1.partial.npz",
])
def test_read_free_admission_stops_on_any_known_evidence_without_mutation(tmp_path, monkeypatch, evidence_name):
    checkout, output, scope_path, ledger_path = free_admission_fixture(tmp_path)
    evidence = scope_path if evidence_name == scope_path.name else (
        ledger_path if evidence_name == ledger_path.name else output / evidence_name
    )
    evidence.write_bytes(b"preserve existing evidence")
    before = {path: path.read_bytes() for path in output.iterdir()}
    monkeypatch.setattr(shared.subprocess, "check_output", lambda *_args, **_kwargs: shared.FREE_HEAD)

    with pytest.raises(ValueError, match="evidence exists"):
        shared.read_free_admission(
            checkout=checkout, output=output, scope_path=scope_path, ledger_path=ledger_path,
        )
    assert {path: path.read_bytes() for path in output.iterdir()} == before


def test_read_free_admission_rejects_head_drift_without_mutation(tmp_path, monkeypatch):
    checkout, output, scope_path, ledger_path = free_admission_fixture(tmp_path)
    monkeypatch.setattr(shared.subprocess, "check_output", lambda *_args, **_kwargs: "different-head")

    with pytest.raises(ValueError, match="HEAD differs"):
        shared.read_free_admission(
            checkout=checkout, output=output, scope_path=scope_path, ledger_path=ledger_path,
        )
    assert list(output.iterdir()) == []


def test_read_free_admission_permission_denial_is_unknown_and_does_not_change_evidence(tmp_path, monkeypatch):
    checkout, output, scope_path, ledger_path = free_admission_fixture(tmp_path)
    blocked = output / "result.v1.supervisor.json"
    original_lstat = Path.lstat

    def denied_lstat(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError("synthetic evidence visibility denial")
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", denied_lstat)
    monkeypatch.setattr(shared.subprocess, "check_output", lambda *_args, **_kwargs: shared.FREE_HEAD)
    with pytest.raises(PermissionError, match="visibility denial"):
        shared.read_free_admission(
            checkout=checkout, output=output, scope_path=scope_path, ledger_path=ledger_path,
        )
    assert list(output.iterdir()) == []


def test_live_paid_launcher_checks_free_management_before_gpu_observation(monkeypatch):
    events = []

    class Lock:
        def acquire(self):
            events.append("mutex_acquired")
            return self

        def release(self, *, release_verified):
            events.append(("mutex_released", release_verified))

    def reject_free_management(**_kwargs):
        events.append("free_management_checked")
        raise ValueError("known free evidence exists")

    def gpu_query(*_args, **_kwargs):
        events.append("gpu_observed")
        raise AssertionError("GPU observation must follow free-management admission")

    monkeypatch.setattr(paid, "AffineScopeLock", lambda *_args: Lock())
    monkeypatch.setattr(paid, "read_free_admission", reject_free_management)
    monkeypatch.setattr(paid.subprocess, "run", gpu_query)
    with pytest.raises(ValueError, match="known free evidence exists"):
        paid.run_live({})
    assert events == ["mutex_acquired", "free_management_checked", ("mutex_released", True)]


def test_paid_main_preserves_job_15782177_gpu_query_rejection_and_does_not_retry(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "diagnostics"
    output_dir.mkdir()
    output = output_dir / "result.v1.json"
    ledger = output_dir / "paid.ledger.json"
    scope_path = output_dir / "scope.v1.json"
    scope_path.write_text(json.dumps({
        "schema_version": paid.SCHEMA,
        "scope_id": paid.SCOPE_ID,
        "status": "open",
        "gpu_execution_contract": paid.GPU_EXECUTION_CONTRACT,
        "os_device_open_isolation_required": False,
        "execution_authorized": True,
        "approval_note": "synthetic test only",
    }), encoding="utf-8")
    args = SimpleNamespace(dry_run=False, scope_json=scope_path, output=output, ledger=ledger)
    monkeypatch.setattr(paid.free, "parser", lambda: SimpleNamespace(parse_args=lambda _argv: args))
    monkeypatch.setattr(paid, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(paid, "OUTPUT", output)
    monkeypatch.setattr(paid, "LEDGER", ledger)
    monkeypatch.setattr(paid, "SCOPE_LOCK", tmp_path / "locks" / "paid.lock")
    monkeypatch.setattr(paid, "os", SimpleNamespace(
        name="posix", environ={"SLURM_JOB_ID": "15782177", "SLURM_JOB_GPUS": "0"},
    ))
    monkeypatch.setattr(paid, "prepare", lambda _args: {})
    monkeypatch.setattr(paid, "read_free_admission", lambda **_kwargs: {"status": "known_free_unopened_unreserved"})
    monkeypatch.setattr(paid.free, "Supervisor", lambda *_args: pytest.fail("probe supervisor must not start"))
    command = ["nvidia-smi", "-i", "0", "--query-gpu=index,uuid,name",
               "--format=csv,noheader,nounits"]
    calls = []

    def reject_gpu_query(observed_command, **kwargs):
        calls.append((observed_command, kwargs))
        raise subprocess.CalledProcessError(
            2, observed_command, output="synthetic query stdout", stderr="synthetic query stderr",
        )

    monkeypatch.setattr(paid.subprocess, "run", reject_gpu_query)
    assert paid.main([]) == 2
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "paid_admission_rejected"
    assert first["GPU_executed"] is False
    assert first["error_type"] == "CalledProcessError"
    assert first["subprocess_failure"] == {
        "command": command,
        "returncode": 2,
        "stdout": "synthetic query stdout",
        "stderr": "synthetic query stderr",
    }
    assert first["gpu_observation"]["commands"] == [{
        "status": "failed", **first["subprocess_failure"],
    }]
    assert calls == [(command, {"capture_output": True, "text": True, "timeout": 4, "check": True})]
    attempt = output.with_suffix(".job-attempt.json")
    assert json.loads(attempt.read_text(encoding="utf-8"))["retry_allowed"] is False
    assert not (tmp_path / "locks" / "paid.lock").exists()
    assert not ledger.exists()

    assert paid.main([]) == 2
    second = json.loads(capsys.readouterr().out)
    assert second["error_type"] == "FileExistsError"
    assert calls == [(command, {"capture_output": True, "text": True, "timeout": 4, "check": True})]
    assert not (tmp_path / "locks" / "paid.lock").exists()


def test_subprocess_timeout_diagnostics_preserve_command_and_partial_output():
    command = ["nvidia-smi", "--query-gpu=index"]
    error = subprocess.TimeoutExpired(command, 4, output=b"partial stdout", stderr=b"partial stderr")
    assert paid.subprocess_failure_detail(error) == {
        "command": command,
        "timeout_seconds": 4,
        "stdout": "partial stdout",
        "stderr": "partial stderr",
    }


GPU_XML = """<nvidia_smi_log><gpu id="00000000:01:00.0">
<product_name>Tesla V100-SXM2</product_name>
<uuid>GPU-physical-three</uuid><minor_number>5</minor_number>
</gpu></nvidia_smi_log>"""


def test_gpu_cli_observation_binds_uuid_xml_minor_without_assuming_index(monkeypatch):
    results = [
        subprocess.CompletedProcess([], 0, "3, GPU-physical-three, Tesla V100-SXM2\n", "csv note"),
        subprocess.CompletedProcess([], 0, GPU_XML, "xml note"),
    ]
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        return results.pop(0)

    monkeypatch.setattr(paid.subprocess, "run", fake_run)
    row, observation = paid.observe_gpu_mapping({"SLURM_JOB_GPUS": "3"})

    assert row == {"index": "3", "uuid": "GPU-physical-three", "name": "Tesla V100-SXM2", "minor_number": 5}
    assert [item[0] for item in commands] == [
        ["nvidia-smi", "-i", "3", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"],
        ["nvidia-smi", "-i", "GPU-physical-three", "-q", "-x"],
    ]
    assert all(item[1] == {"capture_output": True, "text": True, "timeout": 4, "check": True}
               for item in commands)
    assert [entry["stdout"] for entry in observation["commands"]] == [
        "3, GPU-physical-three, Tesla V100-SXM2\n", GPU_XML,
    ]
    assert [entry["stderr"] for entry in observation["commands"]] == ["csv note", "xml note"]


@pytest.mark.parametrize(("xml", "message"), [
    ("<nvidia_smi_log><gpu><product_name>Tesla V100-SXM2</product_name><uuid>GPU-physical-three</uuid></gpu></nvidia_smi_log>", "occur exactly once"),
    (GPU_XML.replace("GPU-physical-three", "GPU-wrong"), "UUID differs"),
    (GPU_XML.replace("Tesla V100-SXM2", "Tesla A100"), "product name differs"),
    (GPU_XML.replace("<minor_number>5</minor_number>", "<minor_number>N/A</minor_number>"), "minor_number"),
    (GPU_XML.replace("</gpu>", "<minor_number>6</minor_number></gpu>"), "each occur exactly once"),
    (GPU_XML.replace("</gpu>", "<uuid>GPU-conflict</uuid></gpu>"), "each occur exactly once"),
    (GPU_XML.replace("</nvidia_smi_log>", GPU_XML[GPU_XML.index("<gpu"):GPU_XML.index("</nvidia_smi_log>")] + "</nvidia_smi_log>"), "exactly one GPU"),
    (GPU_XML.replace("<nvidia_smi_log>", "<other_root>").replace("</nvidia_smi_log>", "</other_root>"), "root is not"),
    (GPU_XML.replace("</gpu>", "<wrapper><gpu><uuid>GPU-nested</uuid></gpu></wrapper></gpu>"), "exactly one GPU"),
    ("<nvidia_smi_log><gpu>", "no element found"),
])
def test_gpu_cli_observation_rejects_xml_identity_and_minor_gaps(monkeypatch, xml, message):
    results = [
        subprocess.CompletedProcess([], 0, "3, GPU-physical-three, Tesla V100-SXM2\n", "csv stderr"),
        subprocess.CompletedProcess([], 0, xml, "xml stderr"),
    ]
    monkeypatch.setattr(paid.subprocess, "run", lambda *_args, **_kwargs: results.pop(0))

    with pytest.raises(paid.GpuObservationError, match=message) as caught:
        paid.observe_gpu_mapping({"SLURM_JOB_GPUS": "3"})

    assert len(caught.value.observation["commands"]) == 2
    assert caught.value.observation["commands"][0]["stdout"].startswith("3,")
    assert caught.value.observation["commands"][1]["stdout"] == xml
    assert caught.value.observation["commands"][1]["stderr"] == "xml stderr"


def test_gpu_cli_observation_rejects_wrong_physical_index_before_uuid_query(monkeypatch):
    calls = []
    monkeypatch.setattr(paid.subprocess, "run", lambda command, **_kwargs: (
        calls.append(command) or subprocess.CompletedProcess(command, 0,
            "2, GPU-physical-two, Tesla V100-SXM2\n", "index stderr")))

    with pytest.raises(paid.GpuObservationError, match="index/UUID/name mapping") as caught:
        paid.observe_gpu_mapping({"SLURM_JOB_GPUS": "3"})

    assert calls == [["nvidia-smi", "-i", "3", "--query-gpu=index,uuid,name",
                      "--format=csv,noheader,nounits"]]
    assert len(caught.value.observation["commands"]) == 1
    assert caught.value.observation["commands"][0]["stderr"] == "index stderr"


def test_gpu_cli_xml_subprocess_failure_retains_prior_csv_and_error_output(monkeypatch):
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 0,
                "3, GPU-physical-three, Tesla V100-SXM2\n", "csv stderr")
        raise subprocess.CalledProcessError(6, command, output="xml partial", stderr="xml denied")

    monkeypatch.setattr(paid.subprocess, "run", fake_run)
    with pytest.raises(paid.GpuObservationError, match="query command failed") as caught:
        paid.observe_gpu_mapping({"SLURM_JOB_GPUS": "3"})

    assert caught.value.subprocess_failure == {
        "command": calls[1], "returncode": 6, "stdout": "xml partial", "stderr": "xml denied",
    }
    assert caught.value.observation["commands"][0]["stdout"] == "3, GPU-physical-three, Tesla V100-SXM2\n"
    assert caught.value.observation["commands"][1] == {
        "status": "failed", **caught.value.subprocess_failure,
    }


def test_paid_main_preserves_successful_gpu_queries_when_device_node_is_missing(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "diagnostics"
    output_dir.mkdir()
    output = output_dir / "result.v1.json"
    ledger = output_dir / "paid.ledger.json"
    scope_path = output_dir / "scope.v1.json"
    scope_path.write_text(json.dumps({
        "schema_version": paid.SCHEMA, "scope_id": paid.SCOPE_ID, "status": "open",
        "gpu_execution_contract": paid.GPU_EXECUTION_CONTRACT, "os_device_open_isolation_required": False,
        "execution_authorized": True, "approval_note": "synthetic test only",
    }), encoding="utf-8")
    args = SimpleNamespace(dry_run=False, scope_json=scope_path, output=output, ledger=ledger)
    monkeypatch.setattr(paid.free, "parser", lambda: SimpleNamespace(parse_args=lambda _argv: args))
    monkeypatch.setattr(paid, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(paid, "OUTPUT", output)
    monkeypatch.setattr(paid, "LEDGER", ledger)
    monkeypatch.setattr(paid, "SCOPE_LOCK", tmp_path / "locks" / "paid.lock")
    monkeypatch.setattr(paid, "os", SimpleNamespace(name="posix", environ={
        "SLURM_JOB_ID": "15782177", "SLURM_JOB_GPUS": "3", "ORCJAX_SLURM_VISIBLE_DEVICES": "0",
    }))
    monkeypatch.setattr(paid, "prepare", lambda _args: {
        "slurm_allocation": {"job": {"JobId": "15782177"}}, "scope": {},
    })
    monkeypatch.setattr(paid, "read_free_admission", lambda **_kwargs: {"status": "known_free_unopened_unreserved"})
    monkeypatch.setattr(paid.free, "Supervisor", lambda *_args: pytest.fail("probe supervisor must not start"))
    results = [
        subprocess.CompletedProcess([], 0, "3, GPU-physical-three, Tesla V100-SXM2\n", "csv stderr"),
        subprocess.CompletedProcess([], 0, GPU_XML, "xml stderr"),
    ]
    calls = []
    monkeypatch.setattr(paid.subprocess, "run", lambda command, **_kwargs: calls.append(command) or results.pop(0))
    original_stat = Path.stat

    def missing_gpu_node(path, *positional, **kwargs):
        if path.as_posix() == "/dev/nvidia5":
            raise FileNotFoundError("synthetic GPU node missing")
        return original_stat(path, *positional, **kwargs)

    monkeypatch.setattr(Path, "stat", missing_gpu_node)
    assert paid.main([]) == 2
    rejection = json.loads(capsys.readouterr().out)
    assert rejection["status"] == "paid_admission_rejected"
    assert rejection["GPU_executed"] is False
    assert rejection["error_type"] == "FileNotFoundError"
    assert [entry["stdout"] for entry in rejection["gpu_observation"]["commands"]] == [
        "3, GPU-physical-three, Tesla V100-SXM2\n", GPU_XML,
    ]
    assert [entry["stderr"] for entry in rejection["gpu_observation"]["commands"]] == [
        "csv stderr", "xml stderr",
    ]
    assert json.loads(output.with_suffix(".job-attempt.json").read_text(encoding="utf-8"))["retry_allowed"] is False
    assert calls[1] == ["nvidia-smi", "-i", "GPU-physical-three", "-q", "-x"]
    assert not (tmp_path / "locks" / "paid.lock").exists()
    assert not ledger.exists()


def test_live_paid_launcher_uses_xml_minor_and_saves_observation_before_supervisor(monkeypatch):
    paths = []
    events = []

    class Lock:
        def acquire(self):
            return self

        def release(self, *, release_verified):
            events.append(("lock_released", release_verified))

    class Supervisor:
        def __init__(self, plan):
            self.record = {}
            self.deadline = 1200
            self.clock = lambda: 0
            events.append(("supervisor", plan["slurm_gpu_mapping"]["actual_GPU_minor"],
                           plan["slurm_gpu_observation"]["commands"][1]["stdout"]))

        def run(self):
            self.record["release_verified"] = True
            return 0

    monkeypatch.setenv("SLURM_JOB_GPUS", "3")
    monkeypatch.setenv("ORCJAX_SLURM_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(paid, "os", SimpleNamespace(
        environ=os.environ, major=lambda _device: 195, minor=lambda _device: 5,
    ))
    monkeypatch.setattr(paid, "AffineScopeLock", lambda *_args: Lock())
    monkeypatch.setattr(paid, "read_free_admission", lambda **_kwargs: {"status": "known_free_unopened_unreserved"})
    monkeypatch.setattr(paid, "read_cgroup_rules", lambda _job, **_kwargs: ("/tmp/devices.list", "c 195:5 rwm\n"))
    monkeypatch.setattr(paid.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(paid.free, "Supervisor", Supervisor)
    results = [
        subprocess.CompletedProcess([], 0, "3, GPU-physical-three, Tesla V100-SXM2\n", ""),
        subprocess.CompletedProcess([], 0, GPU_XML, ""),
    ]
    monkeypatch.setattr(paid.subprocess, "run", lambda *_args, **_kwargs: results.pop(0))
    original_stat = Path.stat

    def fake_stat(path, *args, **kwargs):
        if path.as_posix() == "/dev/nvidia5":
            paths.append(path.as_posix())
            return SimpleNamespace(st_mode=paid.stat.S_IFCHR, st_rdev=1)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)
    result = paid.run_live({"slurm_allocation": {"job": {"JobId": "742001"}},
        "scope": {"gpu_execution_contract": paid.GPU_EXECUTION_CONTRACT}})

    assert result == 0
    assert paths == ["/dev/nvidia5"]
    assert events == [("supervisor", 5, GPU_XML), ("lock_released", True)]


@pytest.mark.parametrize("poison", ["missing_path", "wrong_hash"])
def test_paid_prepare_rejects_source_manifest_poison(tmp_path, monkeypatch, poison):
    args, scope, scope_path, _, _ = scope_json(tmp_path, monkeypatch)
    first_path = next(iter(scope["source_files"]))
    if poison == "missing_path":
        scope["source_files"].pop(first_path)
    else:
        scope["source_files"][first_path] = "0" * 64
    scope_path.write_text(json.dumps(scope), encoding="utf-8")

    with pytest.raises(ValueError, match="source manifest differs"):
        paid.prepare(args)


@pytest.mark.parametrize(("actual_head", "accepts"), [("fixture-head", True), ("different-head", False)])
def test_paid_candidate_nonnull_head_is_bound_to_actual_git_head(tmp_path, monkeypatch, actual_head, accepts):
    args, scope, scope_path, _, _ = scope_json(tmp_path, monkeypatch)
    scope["code_git_head"] = "fixture-head"
    scope_path.write_text(json.dumps(scope), encoding="utf-8")
    calls = []

    def check_output(command, **kwargs):
        calls.append(command)
        return actual_head

    monkeypatch.setattr(paid.subprocess, "check_output", check_output)
    if accepts:
        plan = paid.prepare(args)
        assert plan["scope"]["code_git_head"] == "fixture-head"
    else:
        with pytest.raises(ValueError, match="accepted commit binding differs"):
            paid.prepare(args)
    assert calls == [["git", "rev-parse", "HEAD"]]


def test_paid_live_preparation_mocks_exact_scontrol_details_command(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    output_dir = remote / "outputs" / "diagnostics" / paid.SCOPE_ID
    output = output_dir / "result.v1.json"
    ledger = output_dir / (paid.SCOPE_ID + ".ledger.json")
    scope_path = output_dir / "scope.v1.json"
    output_dir.mkdir(parents=True)
    environment = {
        "SLURM_JOB_ID": "742001", "SLURM_CPUS_PER_TASK": "4", "SLURM_NTASKS": "1",
        "SLURM_JOB_NUM_NODES": "1", "SLURM_JOB_PARTITION": "gnall",
        "SLURM_JOB_START_TIME": "1700000000", "SLURM_JOB_END_TIME": "1700001200",
        "SLURM_JOB_GPUS": "3", "ORCJAX_SLURM_VISIBLE_DEVICES": "0",
    }
    started = datetime.datetime.fromtimestamp(1700000000, datetime.timezone.utc).isoformat(timespec="seconds")
    ended = datetime.datetime.fromtimestamp(1700001200, datetime.timezone.utc).isoformat(timespec="seconds")
    job_text = (
        "JobId=742001 JobState=RUNNING Partition=gnall NodeList=ibc13b03n04 "
        "NumNodes=1 NumCPUs=4 NumTasks=1 Requeue=0 TimeLimit=00:20:00 "
        "AllocTRES=cpu=4,mem=16G,node=1,gres/gpu=1,gres/gpu:v100=1 "
        "GresDetail=gpu:v100:1(IDX:3) GRES=gpu:v100:1(IDX:3) OverSubscribe=OK "
        "RunTime=00:01:00 StartTime=" + started + " EndTime=" + ended
    )
    scope = paid.proposed_scope()
    fixture_identities = {
        "operand": "operand-fixture", "receipt": "receipt-fixture", "capture_source": "source-fixture",
    }
    scope.update(status="open", execution_authorized=True, approval_note="CPU mock only",
                 code_git_head="fixture-head", output=output.as_posix(), ledger=ledger.as_posix(),
                 input_sha256=fixture_identities)
    scope_path.write_text(json.dumps(scope), encoding="utf-8")
    launch_receipt = output_dir / "slurm-attempt.v1.json"
    launch_receipt.write_text(json.dumps({
        "scope_id": paid.SCOPE_ID, "job_id": "742001", "status": "scheduler_started", "retry_allowed": False,
    }), encoding="utf-8")

    monkeypatch.setattr(paid, "ROOT", remote)
    monkeypatch.setattr(paid, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(paid, "OUTPUT", output)
    monkeypatch.setattr(paid, "LEDGER", ledger)
    monkeypatch.setattr(paid.free, "REMOTE", remote)
    monkeypatch.setattr(paid.free, "admit_capture", lambda args: (
        dict(fixture_identities), {"current_execution": {"git_head": "fixture-head"}},
    ))
    monkeypatch.setattr(paid, "source_manifest", lambda **_kwargs: dict(scope["source_files"]))
    monkeypatch.setattr(paid, "os", SimpleNamespace(name="posix", environ=environment))
    monkeypatch.setattr(paid.time, "time", lambda: 1700000060)
    monkeypatch.setattr(paid.socket, "gethostname", lambda: "ibc13b03n04.cluster")
    commands = []

    def check_output(command, **kwargs):
        commands.append(command)
        if command == ["git", "rev-parse", "HEAD"]:
            return "fixture-head"
        if command == ["git", "status", "--porcelain", "--untracked-files=all"]:
            return ""
        raise AssertionError(command)

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=job_text)

    monkeypatch.setattr(paid.subprocess, "check_output", check_output)
    monkeypatch.setattr(paid.subprocess, "run", run)
    args = SimpleNamespace(
        operands=remote / "inputs" / "operands.npz", capture_receipt=remote / "inputs" / "receipt.json",
        capture_source=remote / "inputs" / "capture.py", scope_json=scope_path,
        output=output, ledger=ledger, dry_run=False, shell_started_epoch=1700000000,
    )

    plan = paid.prepare(args)

    assert plan["slurm_allocation"]["actual_AllocTRES"]["gres/gpu:v100"] == "1"
    assert commands[-1] == [paid.SLURM, "show", "job", "--details", "-o", "742001"]
    assert plan["slurm_allocation"]["job"]["GresDetail"] == "gpu:v100:1(IDX:3)"

    args.shell_started_epoch += 1
    with pytest.raises(ValueError, match="outer shell deadline differs"):
        paid.prepare(args)


def test_shared_scope_lock_is_exclusive_and_unverified_release_stays_locked(tmp_path):
    path = tmp_path / "locks" / "affine.lock"
    first = shared.AffineScopeLock(path, "scope-a").acquire()
    second = shared.AffineScopeLock(path, "scope-b")
    with pytest.raises(ValueError, match="already held or stale"):
        second.acquire()
    assert first.release(release_verified=False) is False
    assert path.is_dir()
    assert first.release(release_verified=True) is True
    assert not path.exists()


def test_shared_scope_lock_never_takes_over_a_stale_lock(tmp_path):
    path = tmp_path / "locks" / "affine.lock"
    path.mkdir(parents=True)
    owner = path / "owner.json"
    owner.write_text(json.dumps({"scope_id": "old", "pid": 1, "host": "other-host"}), encoding="utf-8")
    with pytest.raises(ValueError, match="already held or stale"):
        shared.AffineScopeLock(path, "new-scope").acquire()
    assert json.loads(owner.read_text(encoding="utf-8"))["scope_id"] == "old"


def test_shared_scope_lock_refuses_to_remove_another_nonce(tmp_path):
    path = tmp_path / "locks" / "affine.lock"
    lock = shared.AffineScopeLock(path, "scope-a").acquire()
    owner_path = path / "owner.json"
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    owner["nonce"] = "foreign-nonce"
    owner_path.write_text(json.dumps(owner), encoding="utf-8")
    with pytest.raises(ValueError, match="owner differs"):
        lock.release(release_verified=True)
    assert path.is_dir() and owner_path.is_file()


def test_shared_scope_lock_serializes_concurrent_processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    release_event = context.Event()
    results = context.Queue()
    lock_path = str(tmp_path / "scope.lock")
    workers = [context.Process(target=lock_worker,
                               args=(lock_path, start_event, release_event, results))
               for _ in range(2)]
    for worker in workers:
        worker.start()
    try:
        start_event.set()
        outcomes = [results.get(timeout=15) for _ in workers]
    finally:
        release_event.set()
        for worker in workers:
            worker.join(timeout=15)
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
    assert sorted(outcomes) == ["acquired", "busy"]
    assert all(worker.exitcode == 0 for worker in workers)
    assert not Path(lock_path).exists()


def test_shared_scope_lock_stays_held_after_owner_dies_while_child_lives(tmp_path):
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    lock_path = str(tmp_path / "scope.lock")
    child_dir = tmp_path / "child-state"
    child_dir.mkdir()
    release_child = child_dir / "release-child"
    heartbeat = child_dir / "heartbeat"
    child_finished = child_dir / "child-finished"
    owner = context.Process(target=lock_owner_with_child, args=(lock_path, str(child_dir), results))
    owner.start()
    try:
        child_pid = results.get(timeout=10)
        deadline = time.monotonic() + 5
        while not heartbeat.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert heartbeat.exists(), f"child {child_pid} did not start heartbeating"
        assert json.loads((Path(lock_path) / "owner.json").read_text(encoding="utf-8"))["scope_id"] == "owner-death-test"

        owner.terminate()
        owner.join(timeout=5)
        assert not owner.is_alive()
        assert owner.exitcode != 0
        heartbeat_before = heartbeat.read_text(encoding="ascii")
        deadline = time.monotonic() + 5
        while heartbeat.read_text(encoding="ascii") == heartbeat_before and time.monotonic() < deadline:
            time.sleep(0.05)
        assert heartbeat.read_text(encoding="ascii") != heartbeat_before

        with pytest.raises(ValueError, match="already held or stale"):
            shared.AffineScopeLock(lock_path, "contender").acquire()
        assert Path(lock_path).is_dir()
    finally:
        release_child.write_text("release", encoding="ascii")
        if owner.is_alive():
            owner.terminate()
            owner.join(timeout=5)
        deadline = time.monotonic() + 5
        while not child_finished.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
    assert child_finished.exists(), "orphan child did not exit after its release flag"
    assert Path(lock_path).is_dir(), "owner death must leave the shared mutex fail-closed"
