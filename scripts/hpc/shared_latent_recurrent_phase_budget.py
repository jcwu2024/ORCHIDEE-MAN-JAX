#!/usr/bin/env python3
"""Approval-bound cumulative free-node budget for corrected T30 phases."""
from __future__ import annotations

import argparse
import hashlib
import copy
from contextlib import ExitStack, contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid


SCHEMA = "latent_reader_t30_free_approval_v1"
SCOPE = "latent_reader_T30_free_15600_20261006"
LIMIT = 15600
FUNDING_POOLS = {"compile": 12000, "matched": 3600}
OCCUPATION_CAPS = {
    "compile-original": 4800,
    "compile-candidate": 7200,
    "matched": 3600,
}
ROOT = Path("/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX")
DIAG = ROOT / "runtime/outputs/diagnostics"
AGGREGATE_NAME = "latent-reader-T30-20261006-free-budget.json"
APPROVAL_NAME = "latent-reader-T30-20261006-free-approval.json"
RESOURCE_RECORD = {
    "host": "gln01",
    "gpus": 1,
    "cpu_affinity": "8-15",
    "openblas_threads": 1,
    "omp_threads": 4,
    "memory_gib": 64,
    "memory_enforcement": "host-availability-precheck; no RSS limit",
    "exclusive": False,
}
FINALIZE_RESERVE_SECONDS = 120
FINALIZATION_CHARGE_RESERVE_SECONDS = 60
GPU_QUERY_TIMEOUT_SECONDS = 20


def approval_template() -> dict:
    """Return an unapproved receipt template for the single T30 grant."""
    return {
        "schema_version": SCHEMA,
        "approval_id": None,
        "approved": False,
        "approved_at_utc": None,
        "human_approval_reference": None,
        "funding_type": "free",
        "funding_scope": SCOPE,
        "allowed_gpu_indices": [0, 1],
        "approved_gpu_seconds": LIMIT,
        "max_occupation_seconds": 7200,
        "funding_pools_gpu_seconds": copy.deepcopy(FUNDING_POOLS),
        "occupation_caps_seconds": copy.deepcopy(OCCUPATION_CAPS),
        "resources": copy.deepcopy(RESOURCE_RECORD),
    }


def validate_approval(receipt: dict) -> str:
    if not isinstance(receipt, dict) or receipt.get("schema_version") != SCHEMA:
        raise ValueError("unsupported or non-T30 approval receipt schema")
    if receipt.get("approved") is not True:
        raise ValueError("fresh human approval receipt must have approved=true")
    try:
        uuid.UUID(str(receipt.get("approval_id", "")))
    except (ValueError, AttributeError) as error:
        raise ValueError("approval_id must be a fresh UUID") from error
    approved_at = receipt.get("approved_at_utc")
    if not isinstance(approved_at, str):
        raise ValueError("approved_at_utc is required")
    try:
        parsed = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("approved_at_utc must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError("approved_at_utc must include a timezone")
    if not isinstance(receipt.get("human_approval_reference"), str) or not receipt["human_approval_reference"].strip():
        raise ValueError("human_approval_reference is required")
    expected = approval_template()
    for key in ("funding_type", "funding_scope", "allowed_gpu_indices", "approved_gpu_seconds", "max_occupation_seconds", "funding_pools_gpu_seconds", "occupation_caps_seconds", "resources"):
        if receipt.get(key) != expected[key]:
            raise ValueError(f"approval receipt changed required field: {key}")
    if type(receipt.get("approved_gpu_seconds")) is not int or type(receipt.get("max_occupation_seconds")) is not int:
        raise ValueError("approval limits must be integer seconds")
    return hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def phase_occupation_path(phase: str, occupation_id: str) -> str:
    if phase not in OCCUPATION_CAPS:
        raise ValueError(f"unknown occupation: {phase}")
    uuid.UUID(occupation_id)
    return f"latent-reader-T30-20261006-{phase}-{occupation_id}.occupation.json"


def _charge(row: dict) -> float:
    if row.get("status") in ("reserved", "running", "interrupted"):
        return float(row["reserved_gpu_seconds"])
    return float(row.get("charged_gpu_seconds", row.get("occupied_gpu_seconds", row["reserved_gpu_seconds"])))


def command_sha256(command: list[str]) -> str:
    return hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()


def _pool_for(phase: str) -> str:
    return "matched" if phase == "matched" else "compile"


def _summarize_budget(data: dict) -> None:
    charged = sum(_charge(item) for item in data["sessions"])
    by_pool = {pool: sum(_charge(item) for item in data["sessions"] if item.get("funding_pool") == pool)
               for pool in FUNDING_POOLS}
    data["charged_or_reserved_gpu_seconds"] = charged
    data["remaining_gpu_seconds"] = LIMIT - charged
    data["funding_pools_charged_or_reserved_gpu_seconds"] = by_pool
    data["funding_pools_remaining_gpu_seconds"] = {pool: FUNDING_POOLS[pool] - by_pool[pool]
                                                    for pool in FUNDING_POOLS}
    data["grant_within_limit"] = charged <= LIMIT and all(by_pool[pool] <= FUNDING_POOLS[pool] for pool in FUNDING_POOLS)


def _new_ledger(*, phase: str | None, receipt_hash: str, approval_id: str) -> dict:
    return {
        "schema_version": "latent_reader_t30_occupation_ledger_v1" if phase else "latent_reader_t30_cumulative_ledger_v1",
        "funding_scope": SCOPE,
        "approved_gpu_seconds": LIMIT,
        "approval_id": approval_id,
        "approval_sha256": receipt_hash,
        "phase": phase,
        "sessions": [],
    }


def _check_ledger(data: dict, *, phase: str | None, receipt_hash: str, approval_id: str) -> None:
    expected = _new_ledger(phase=phase, receipt_hash=receipt_hash, approval_id=approval_id)
    for key, value in expected.items():
        if key == "sessions":
            continue
        if data.get(key) != value:
            raise ValueError(f"ledger identity changed: {key}")
    identifiers = [row.get("id") for row in data.get("sessions", [])]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate occupation ledger session identity")
    if phase is not None and any(row.get("phase") != phase for row in data.get("sessions", [])):
        raise ValueError("occupation ledger contains a different phase")
    if phase is not None and len(data.get("sessions", [])) > 1:
        raise ValueError("one occupation ledger may describe exactly one occupation")


def _read_ledger(path: Path, *, phase: str | None, receipt_hash: str, approval_id: str) -> dict:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = _new_ledger(phase=phase, receipt_hash=receipt_hash, approval_id=approval_id)
    _check_ledger(data, phase=phase, receipt_hash=receipt_hash, approval_id=approval_id)
    return data


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def _locked_ledgers(paths: list[Path]):
    """Lock both ledgers in stable order on Windows and POSIX."""
    with ExitStack() as stack:
        for path in sorted(paths, key=lambda item: str(item.resolve())):
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(path.with_suffix(path.suffix + ".lock").open("a+b"))
            if os.name == "nt":
                import msvcrt
                lock.seek(0, os.SEEK_END)
                if lock.tell() == 0:
                    lock.write(b"0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
                stack.callback(lambda handle=lock: (handle.seek(0), msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)))
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                stack.callback(fcntl.flock, lock.fileno(), fcntl.LOCK_UN)
        yield


def _remaining(rows: list[dict], predicate, cap: int) -> float:
    used = sum(_charge(row) for row in rows if predicate(row))
    return max(0.0, float(cap) - used)


def _verify_occupation_index(aggregate_path: Path, aggregate: dict, *, receipt_hash: str, approval_id: str) -> None:
    pattern = re.compile(r"latent-reader-T30-20261006-(compile-original|compile-candidate|matched)-([0-9a-f-]{36})\.occupation\.json$")
    indexed: dict[str, dict] = {}
    for path in aggregate_path.parent.glob("latent-reader-T30-20261006-*.occupation.json"):
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError("unrecognized T30 occupation ledger present; refusing a fresh budget history")
        phase, occupation_id = match.groups()
        try:
            uuid.UUID(occupation_id)
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError, json.JSONDecodeError) as error:
            raise ValueError("unreadable T30 occupation ledger; refusing to reset accounting") from error
        _check_ledger(data, phase=phase, receipt_hash=receipt_hash, approval_id=approval_id)
        if len(data["sessions"]) != 1 or data["sessions"][0].get("id") != occupation_id:
            raise ValueError("occupation ledger identity/file name mismatch")
        indexed[occupation_id] = data["sessions"][0]
    if aggregate_path.exists() is False and indexed:
        raise ValueError("cumulative ledger is missing while occupation ledgers remain")
    aggregate_rows = {row["id"]: row for row in aggregate["sessions"]}
    if set(indexed) != set(aggregate_rows):
        raise ValueError("cumulative ledger and occupation ledger inventory differ")
    for occupation_id, row in indexed.items():
        aggregate_row = aggregate_rows[occupation_id]
        for key in ("phase", "funding_pool", "gpu_index", "reserved_gpu_seconds", "occupied_gpu_seconds",
                    "measured_gpu_seconds", "charged_gpu_seconds", "process_group_release_confirmed",
                    "gpu_release_confirmed", "gpu_release_reason", "own_process_release_confirmed", "status"):
            if row.get(key) != aggregate_row.get(key):
                raise ValueError("cumulative and occupation ledger records differ")


def _reserve(paths: tuple[Path, Path], *, phase: str, occupation_id: str, gpu_index: int, seconds: int, command: list[str],
             receipt_hash: str, approval_id: str, start_snapshot: dict) -> str:
    aggregate_path, occupation_path = paths
    session_id = occupation_id
    with _locked_ledgers([aggregate_path, occupation_path]):
        aggregate = _read_ledger(aggregate_path, phase=None, receipt_hash=receipt_hash, approval_id=approval_id)
        _verify_occupation_index(aggregate_path, aggregate, receipt_hash=receipt_hash, approval_id=approval_id)
        if occupation_path.exists():
            raise ValueError("occupation ledger already exists; retries require a fresh UUID and ledger")
        occupation = _read_ledger(occupation_path, phase=phase, receipt_hash=receipt_hash, approval_id=approval_id)
        if any(row["id"] == session_id for row in aggregate["sessions"]):
            raise ValueError("occupation ledger/id already used; retries need a fresh occupation id")
        active = [row for row in aggregate["sessions"] if row.get("status") in ("reserved", "running", "interrupted")]
        if active:
            raise ValueError("unsettled reservation exists; never reset or bypass the shared T30 ledger")
        total_left = _remaining(aggregate["sessions"], lambda _row: True, LIMIT)
        pool = _pool_for(phase)
        pool_left = _remaining(aggregate["sessions"], lambda row: row.get("funding_pool") == pool, FUNDING_POOLS[pool])
        if seconds > OCCUPATION_CAPS[phase]:
            raise ValueError(f"{phase} occupation exceeds its {OCCUPATION_CAPS[phase]} second cap")
        if seconds > total_left:
            raise ValueError(f"cumulative T30 grant exhausted: {total_left:g} seconds remain")
        if seconds > pool_left:
            raise ValueError(f"protected {pool} pool exhausted: {pool_left:g} seconds remain")
        row = {
            "id": session_id,
            "phase": phase,
            "funding_pool": pool,
            "gpu_index": gpu_index,
            "host": start_snapshot["hostname"],
            "wrapper_pid": os.getpid(),
            "session_cap_seconds": seconds,
            "reserved_gpu_seconds": seconds,
            "started_epoch": time.time(),
            "command": command,
            "start_snapshot": start_snapshot,
            "status": "reserved",
        }
        aggregate["sessions"].append(row)
        occupation["sessions"].append(dict(row))
        _summarize_budget(aggregate)
        _atomic_json(aggregate_path, aggregate)
        _atomic_json(occupation_path, occupation)
    return session_id


def _mark_running(paths: tuple[Path, Path], session_id: str, *, receipt_hash: str, approval_id: str) -> None:
    aggregate_path, occupation_path = paths
    with _locked_ledgers([aggregate_path, occupation_path]):
        aggregate = _read_ledger(aggregate_path, phase=None, receipt_hash=receipt_hash, approval_id=approval_id)
        selected = next((item for item in aggregate["sessions"] if item["id"] == session_id), None)
        if selected is None:
            raise ValueError("cumulative occupation record disappeared before process start")
        occupation = _read_ledger(occupation_path, phase=selected["phase"], receipt_hash=receipt_hash, approval_id=approval_id)
        for ledger in (aggregate, occupation):
            row = next((item for item in ledger["sessions"] if item["id"] == session_id), None)
            if row is None or row["status"] != "reserved":
                raise ValueError("reserved occupation record missing or changed")
            row["status"] = "running"
            row["process_started_epoch"] = time.time()
        _summarize_budget(aggregate)
        _atomic_json(aggregate_path, aggregate)
        _atomic_json(occupation_path, occupation)


def verify_running_occupation(paths: tuple[Path, Path], *, phase: str, occupation_id: str, gpu_index: int,
                              expected_command_sha256: str,
                              receipt_hash: str, approval_id: str) -> None:
    aggregate_path, occupation_path = paths
    with _locked_ledgers([aggregate_path, occupation_path]):
        aggregate = _read_ledger(aggregate_path, phase=None, receipt_hash=receipt_hash, approval_id=approval_id)
        _verify_occupation_index(aggregate_path, aggregate, receipt_hash=receipt_hash, approval_id=approval_id)
        aggregate_rows = [row for row in aggregate["sessions"] if row.get("id") == occupation_id]
        if len(aggregate_rows) != 1:
            raise ValueError("active occupation is missing from the cumulative ledger")
        row = aggregate_rows[0]
        occupation = _read_ledger(occupation_path, phase=phase, receipt_hash=receipt_hash, approval_id=approval_id)
        occupation_rows = [item for item in occupation["sessions"] if item.get("id") == occupation_id]
        if len(occupation_rows) != 1 or row.get("phase") != phase or row.get("gpu_index") != gpu_index:
            raise ValueError("active occupation identity or phase does not match")
        reserved_command = row.get("command", [])
        if (command_sha256(reserved_command) != expected_command_sha256 and
                command_sha256(reserved_command[:-1]) != expected_command_sha256):
            raise ValueError("internal phase command differs from the reserved command")
        if row.get("status") != "running" or occupation_rows[0].get("status") != "running":
            raise ValueError("internal phase requires a matching running budget reservation")
        if row.get("host") not in ("gln01", "ibc13b02n04") or not row.get("process_started_epoch"):
            raise ValueError("active occupation lacks a valid host or process-start record")
        guardian_pid = row.get("wrapper_pid")
        if type(guardian_pid) is not int or not _has_process_ancestor(os.getpid(), guardian_pid):
            raise ValueError("internal phase is outside the guardian-launched process tree")


def _finish(paths: tuple[Path, Path], session_id: str, *, receipt_hash: str, approval_id: str,
            elapsed: float, charged_gpu_seconds: float, status: str, exit_status: int, host_end: dict | None,
            process_group_release_confirmed: bool, gpu_release_confirmed: bool, gpu_release_reason: str) -> None:
    aggregate_path, occupation_path = paths
    with _locked_ledgers([aggregate_path, occupation_path]):
        aggregate = _read_ledger(aggregate_path, phase=None, receipt_hash=receipt_hash, approval_id=approval_id)
        row0 = next((item for item in aggregate["sessions"] if item["id"] == session_id), None)
        if row0 is None:
            raise ValueError("cumulative occupation record disappeared")
        occupation = _read_ledger(occupation_path, phase=row0["phase"], receipt_hash=receipt_hash, approval_id=approval_id)
        for ledger in (aggregate, occupation):
            row = next((item for item in ledger["sessions"] if item["id"] == session_id), None)
            if row is None or row["status"] not in ("running", "reserved"):
                raise ValueError("occupation record cannot be finalized")
            own_release = process_group_release_confirmed and gpu_release_confirmed
            final_status = status if own_release else "interrupted"
            row.update(status=final_status, ended_epoch=time.time(), occupied_seconds=elapsed,
                       occupied_gpu_seconds=elapsed, measured_gpu_seconds=elapsed,
                       charged_gpu_seconds=charged_gpu_seconds, exit_status=exit_status,
                       within_session_cap=charged_gpu_seconds <= row["session_cap_seconds"], end_snapshot=host_end,
                       process_group_release_confirmed=process_group_release_confirmed,
                       gpu_release_confirmed=gpu_release_confirmed,
                       gpu_release_reason=gpu_release_reason,
                       own_process_release_confirmed=own_release)
        _summarize_budget(aggregate)
        _atomic_json(aggregate_path, aggregate)
        _atomic_json(occupation_path, occupation)


def _run_checked(command: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


def parse_cpu_affinity(value: str) -> list[int]:
    cpus: list[int] = []
    for item in value.split(","):
        fields = item.split("-")
        if len(fields) == 1:
            cpus.append(int(fields[0]))
        elif len(fields) == 2 and int(fields[0]) <= int(fields[1]):
            cpus.extend(range(int(fields[0]), int(fields[1]) + 1))
        else:
            raise ValueError("cpu-affinity must be a comma list or ascending ranges")
    if len(cpus) != 8 or len(set(cpus)) != 8:
        raise ValueError("cpu-affinity must select exactly 8 distinct CPUs")
    return cpus


def host_snapshot(gpu_index: int, cpu_affinity: list[int]) -> dict:
    hostname = subprocess.run(["hostname", "-s"], capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    load = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    applications = _run_checked(["nvidia-smi", "-i", str(gpu_index), "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], GPU_QUERY_TIMEOUT_SECONDS)
    gpu = _run_checked(["nvidia-smi", "-i", str(gpu_index), "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], GPU_QUERY_TIMEOUT_SECONDS)
    if applications.returncode or gpu.returncode:
        raise RuntimeError("bounded target-GPU occupancy query failed")
    app_lines = [line.strip() for line in applications.stdout.splitlines() if line.strip() and "no running processes" not in line.lower()]
    gpu_fields = [field.strip() for field in gpu.stdout.strip().split(",")]
    if len(gpu_fields) != 5:
        raise RuntimeError("target-GPU snapshot has an unexpected shape")
    try:
        memory_used, memory_total = int(gpu_fields[2]), int(gpu_fields[3])
        utilization = int(gpu_fields[4])
    except ValueError as error:
        raise RuntimeError("target-GPU memory/utilization is not numeric") from error
    memory_available_gib = None
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        memory_available_gib = pages * page_size / (1024 ** 3)
    except (AttributeError, OSError, ValueError):
        pass
    return {
        "hostname": hostname,
        "available_cpu_affinity": affinity,
        "selected_cpu_affinity": cpu_affinity,
        "available_cpu_count": len(affinity),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "load_average_1_5_15": load,
        "host_available_memory_gib": memory_available_gib,
        "gpu_index": int(gpu_fields[0]),
        "gpu_name": gpu_fields[1],
        "gpu_memory_used_mib": memory_used,
        "gpu_memory_total_mib": memory_total,
        "gpu_utilization_percent": utilization,
        "gpu_compute_processes": app_lines,
    }


def validate_free_host(snapshot: dict, cpu_affinity: list[int]) -> None:
    if snapshot.get("hostname") not in ("gln01", "ibc13b02n04"):
        raise ValueError("free T30 occupation is restricted to gln01/ibc13b02n04")
    if snapshot.get("slurm_job_id"):
        raise ValueError("free T30 scope cannot run inside a paid Slurm allocation")
    if not set(cpu_affinity).issubset(snapshot.get("available_cpu_affinity", [])):
        raise ValueError("approved CPU affinity is not currently available on gln01")
    if snapshot.get("selected_cpu_affinity") != cpu_affinity:
        raise ValueError("recorded CPU affinity differs from the approved selector")
    if snapshot.get("host_available_memory_gib") is None or snapshot["host_available_memory_gib"] < RESOURCE_RECORD["memory_gib"]:
        raise ValueError("free T30 occupation requires at least 64 GiB host memory available at start")
    if snapshot.get("gpu_compute_processes"):
        raise ValueError("selected V100 has visible compute processes")
    if snapshot.get("gpu_memory_used_mib", RESOURCE_RECORD["memory_gib"] * 1024) > 100:
        raise ValueError("selected V100 memory use exceeds the idle limit")
    if snapshot.get("gpu_utilization_percent") != 0:
        raise ValueError("selected V100 utilization is not zero")
    if "V100" not in str(snapshot.get("gpu_name", "")):
        raise ValueError("selected device is not a V100")


def gpu_release_evidence(snapshot: dict, expected_gpu_index: int) -> tuple[bool, str]:
    if snapshot.get("snapshot_error"):
        return False, "final_gpu_snapshot_failed"
    if snapshot.get("gpu_index") != expected_gpu_index:
        return False, "final_gpu_index_mismatch"
    if "gpu_compute_processes" not in snapshot or not isinstance(snapshot["gpu_compute_processes"], list):
        return False, "final_gpu_process_list_unavailable"
    if snapshot["gpu_compute_processes"]:
        return False, "visible_gpu_processes_remain_or_appeared"
    memory_used = snapshot.get("gpu_memory_used_mib")
    if type(memory_used) is not int or memory_used > 100:
        return False, "final_gpu_memory_not_idle"
    utilization = snapshot.get("gpu_utilization_percent")
    if type(utilization) is not int:
        return False, "final_gpu_utilization_unavailable"
    return True, "confirmed"


def _has_process_ancestor(pid: int, ancestor_pid: int, max_depth: int = 16) -> bool:
    current = pid
    for _ in range(max_depth):
        if current == ancestor_pid:
            return True
        try:
            stat = Path(f"/proc/{current}/stat").read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return False
        fields = stat[stat.rfind(")") + 2:].split()
        if len(fields) < 2:
            return False
        try:
            parent = int(fields[1])
        except ValueError:
            return False
        if parent <= 1 or parent == current:
            return False
        current = parent
    return False


def _terminate_group(process: subprocess.Popen, grace_seconds: float = 5) -> bool:
    if os.name == "nt":
        try:
            if process.poll() is None:
                process.terminate()
        except OSError:
            pass
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        return process.poll() is not None

    def group_exists() -> bool:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        return False
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and group_exists():
        time.sleep(0.1)
    if group_exists():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            return False
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    return not group_exists()


def execute(args) -> int:
    if args.phase not in OCCUPATION_CAPS:
        raise ValueError("invalid T30 occupation phase")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("missing bounded phase command")
    receipt = json.loads(args.approval_receipt.read_text(encoding="utf-8"))
    receipt_hash = validate_approval(receipt)
    if not 120 <= args.seconds <= min(7200, OCCUPATION_CAPS[args.phase]):
        raise ValueError("occupation seconds must be between 120 and the phase cap")
    if args.seconds > receipt["occupation_caps_seconds"][args.phase]:
        raise ValueError("occupation exceeds the phase cap in the approval receipt")
    if args.aggregate_ledger.name != AGGREGATE_NAME:
        raise ValueError("wrong aggregate ledger name; T7/other grants cannot be reused")
    if args.approval_receipt.name != APPROVAL_NAME:
        raise ValueError("wrong approval receipt name")
    try:
        expected_occupation = phase_occupation_path(args.phase, args.occupation_id)
    except (ValueError, AttributeError) as error:
        raise ValueError("occupation_id must be a UUID") from error
    if args.occupation_ledger.name != expected_occupation:
        raise ValueError("wrong occupation ledger name")
    resolved_diag = DIAG.resolve()
    if any(resolved_diag not in path.resolve().parents for path in (args.aggregate_ledger, args.occupation_ledger, args.approval_receipt)):
        raise ValueError("approval and budget records must remain under project diagnostics")
    if args.aggregate_ledger.resolve() == args.occupation_ledger.resolve():
        raise ValueError("aggregate and occupation ledgers must be distinct")
    if args.gpu_index not in (0, 1):
        raise ValueError("gln01 GPU index must be 0 or 1")
    if args.gpu_index not in receipt["allowed_gpu_indices"]:
        raise ValueError("selected GPU is outside the fresh approval receipt's allowed indices")
    if args.cpu_affinity != receipt["resources"]["cpu_affinity"]:
        raise ValueError("CPU affinity differs from the fresh approval receipt")
    cpu_affinity = parse_cpu_affinity(args.cpu_affinity)
    if os.environ.get("SLURM_JOB_ID"):
        raise ValueError("free T30 occupation refuses a Slurm job allocation")
    started = time.monotonic()
    paths = (args.aggregate_ledger, args.occupation_ledger)
    start_snapshot = host_snapshot(args.gpu_index, cpu_affinity)
    session_id = _reserve(paths, phase=args.phase, occupation_id=args.occupation_id, gpu_index=args.gpu_index,
                          seconds=args.seconds, command=command,
                          receipt_hash=receipt_hash, approval_id=receipt["approval_id"], start_snapshot=start_snapshot)
    process = None
    status = 125
    interrupted = None
    process_group_release_confirmed = True
    gpu_release_confirmed = False
    gpu_release_reason = "not_checked"

    def stop(signum, _frame):
        nonlocal interrupted
        interrupted = signum
        if process is not None:
            _terminate_group(process, grace_seconds=3)

    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, stop)
    try:
        validate_free_host(start_snapshot, cpu_affinity)
        _mark_running(paths, session_id, receipt_hash=receipt_hash, approval_id=receipt["approval_id"])
        selected_cpus = ",".join(str(cpu) for cpu in start_snapshot["selected_cpu_affinity"])
        wrapped_command = ["taskset", "-c", selected_cpus, "env", "OMP_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=1", *command]
        process = subprocess.Popen(wrapped_command, start_new_session=(os.name != "nt"),
                                   creationflags=(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0))
        wait_seconds = max(0.1, args.seconds - FINALIZE_RESERVE_SECONDS - (time.monotonic() - started))
        try:
            status = process.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            _terminate_group(process, grace_seconds=5)
            status = 124
        if interrupted is not None:
            status = 128 + int(interrupted)
    except BaseException as error:
        print(f"phase guard: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        status = 125
    finally:
        if process is not None:
            process_group_release_confirmed = _terminate_group(process, grace_seconds=2)
        try:
            end_snapshot = host_snapshot(args.gpu_index, cpu_affinity)
        except BaseException as error:
            end_snapshot = {"snapshot_error": f"{type(error).__name__}: {error}"}
        elapsed = time.monotonic() - started
        gpu_release_confirmed, gpu_release_reason = gpu_release_evidence(end_snapshot, args.gpu_index)
        charged_gpu_seconds = elapsed + FINALIZATION_CHARGE_RESERVE_SECONDS
        if charged_gpu_seconds > args.seconds:
            status = 124
        _finish(paths, session_id, receipt_hash=receipt_hash, approval_id=receipt["approval_id"],
                elapsed=elapsed, charged_gpu_seconds=charged_gpu_seconds, status="finished",
                exit_status=status, host_end=end_snapshot,
                process_group_release_confirmed=process_group_release_confirmed,
                gpu_release_confirmed=gpu_release_confirmed, gpu_release_reason=gpu_release_reason)
        own_process_release_confirmed = process_group_release_confirmed and gpu_release_confirmed
        if not own_process_release_confirmed:
            status = 125
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    print(json.dumps({"session_id": session_id, "phase": args.phase, "measured_gpu_seconds": elapsed,
                      "charged_gpu_seconds": charged_gpu_seconds,
                      "occupation_status": "finished" if own_process_release_confirmed else "interrupted",
                      "exit_status": status, "own_process_release_confirmed": own_process_release_confirmed,
                      "process_group_release_confirmed": process_group_release_confirmed,
                      "gpu_release_confirmed": gpu_release_confirmed,
                      "gpu_release_reason": gpu_release_reason,
                      "approval_id": receipt["approval_id"]}, sort_keys=True), flush=True)
    return status


def verify_running(args) -> int:
    if args.aggregate_ledger.name != AGGREGATE_NAME or args.approval_receipt.name != APPROVAL_NAME:
        raise ValueError("wrong T30 approval or aggregate ledger path")
    try:
        expected_occupation = phase_occupation_path(args.phase, args.occupation_id)
    except (ValueError, AttributeError) as error:
        raise ValueError("occupation_id must be a UUID") from error
    if args.occupation_ledger.name != expected_occupation:
        raise ValueError("wrong occupation ledger name")
    resolved_diag = DIAG.resolve()
    if any(resolved_diag not in path.resolve().parents for path in
           (args.aggregate_ledger, args.occupation_ledger, args.approval_receipt)):
        raise ValueError("approval and budget records must remain under project diagnostics")
    if args.aggregate_ledger.resolve() == args.occupation_ledger.resolve():
        raise ValueError("aggregate and occupation ledgers must be distinct")
    receipt = json.loads(args.approval_receipt.read_text(encoding="utf-8"))
    receipt_hash = validate_approval(receipt)
    if type(args.gpu_index) is not int or args.gpu_index not in receipt["allowed_gpu_indices"]:
        raise ValueError("selected GPU is outside the approved indices")
    verify_running_occupation(
        (args.aggregate_ledger, args.occupation_ledger),
        phase=args.phase,
        occupation_id=args.occupation_id,
        gpu_index=args.gpu_index,
        expected_command_sha256=args.command_sha256,
        receipt_hash=receipt_hash,
        approval_id=receipt["approval_id"],
    )
    print(json.dumps({"occupation_id": args.occupation_id, "phase": args.phase, "status": "running"}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("template", help="print the unapproved receipt template")
    fingerprint = sub.add_parser("fingerprint-command", help="print the stable SHA-256 for a command vector")
    fingerprint.add_argument("command", nargs=argparse.REMAINDER)
    run = sub.add_parser("execute", help="run one approved bounded free-node occupation")
    run.add_argument("--approval-receipt", type=Path, required=True)
    run.add_argument("--aggregate-ledger", type=Path, required=True)
    run.add_argument("--occupation-ledger", type=Path, required=True)
    run.add_argument("--phase", choices=tuple(OCCUPATION_CAPS), required=True)
    run.add_argument("--occupation-id", required=True)
    run.add_argument("--seconds", type=int, required=True)
    run.add_argument("--gpu-index", type=int, required=True)
    run.add_argument("--cpu-affinity", required=True)
    run.add_argument("command", nargs=argparse.REMAINDER)
    guard = sub.add_parser("verify-running", help="verify the active reservation before internal phase execution")
    guard.add_argument("--approval-receipt", type=Path, required=True)
    guard.add_argument("--aggregate-ledger", type=Path, required=True)
    guard.add_argument("--occupation-ledger", type=Path, required=True)
    guard.add_argument("--phase", choices=tuple(OCCUPATION_CAPS), required=True)
    guard.add_argument("--occupation-id", required=True)
    guard.add_argument("--gpu-index", type=int, required=True)
    guard.add_argument("--command-sha256", required=True)
    args = parser.parse_args(argv)
    if args.action == "template":
        print(json.dumps(approval_template(), indent=2, sort_keys=True))
        return 0
    if args.action == "fingerprint-command":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            raise ValueError("command vector is required")
        print(command_sha256(command))
        return 0
    if args.action == "verify-running":
        return verify_running(args)
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
