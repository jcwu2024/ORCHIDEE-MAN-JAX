"""Single-occupation free-V100 guard for the corrected daily diagnostic.

Uses the T30 guardian's locking, host snapshot, process cleanup and GPU-release
checks, but keeps this new 7200-second grant in distinct approval/ledger files.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

try:
    from scripts.hpc import shared_latent_recurrent_phase_budget as guardian
except ModuleNotFoundError:
    import shared_latent_recurrent_phase_budget as guardian


ROOT = Path("/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX")
DIAG = ROOT / "runtime/outputs/diagnostics"
SCOPE = "shared_latent_corrected_daily_free_7200_20261007"
LIMIT = 7200
SCHEMA = "shared_latent_corrected_daily_free_approval_v1"
AGGREGATE_NAME = "latent-reader-daily-corrected-20261007-free-budget.json"
APPROVAL_NAME = "latent-reader-daily-corrected-20261007-free-approval.json"
OCCUPATION_PREFIX = "latent-reader-daily-corrected-20261007-free-"
LEDGER_SCHEMA = "shared_latent_corrected_daily_occupation_ledger_v1"
FINALIZE_RESERVE_SECONDS = 120
FINALIZATION_CHARGE_RESERVE_SECONDS = 60
CPU_AFFINITY = "8-15"
RESOURCE_RECORD = {
    "host": "gln01",
    "gpus": 1,
    "cpu_affinity": CPU_AFFINITY,
    "openblas_threads": 1,
    "omp_threads": 4,
    "memory_gib": 64,
    "memory_enforcement": "host-availability-precheck; no RSS limit",
    "exclusive": False,
}
OPERATION = {
    "component": "corrected_daily",
    "paid_fallback": False,
    "source_git_head": "c599236bcf57b745e9bafed678298010f4c3b58f",
    "candidate_compiles": 1,
    "original_compiles": 0,
    "complete_ad_calls": 8,
    "call_schedule": ["original_cold", "candidate_cold", "original_pair_1", "candidate_pair_1",
                      "candidate_pair_2", "original_pair_2", "original_pair_3", "candidate_pair_3"],
    "pairs": 3,
    "occupation_seconds": LIMIT,
    "probe_wall_seconds": 6840,
    "lower_seconds": 600,
    "backend_seconds": 4800,
    "call_seconds": 180,
    "force_compilation_parallelism": 4,
}


def approval_template() -> dict:
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
        "max_occupation_seconds": LIMIT,
        "max_occupations": 1,
        "maximum_card_hours": 2.0,
        "estimated_cost_cny": 0.0,
        "paid_fallback": False,
        "resources": RESOURCE_RECORD,
        "operation": OPERATION,
    }


def validate_approval(receipt: dict) -> str:
    if not isinstance(receipt, dict) or receipt.get("schema_version") != SCHEMA:
        raise ValueError("unsupported or non-daily approval receipt schema")
    if receipt.get("approved") is not True:
        raise ValueError("fresh human approval receipt must have approved=true")
    try:
        approval_id = str(uuid.UUID(str(receipt.get("approval_id", ""))))
    except (ValueError, AttributeError) as error:
        raise ValueError("approval_id must be a fresh UUID") from error
    if approval_id == "59a57e60-91e1-434a-9787-7189dc602e93":
        raise ValueError("the completed T30 approval identity cannot fund the daily occupation")
    approved_at = receipt.get("approved_at_utc")
    if not isinstance(approved_at, str):
        raise ValueError("approved_at_utc is required")
    try:
        parsed = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("approved_at_utc must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError("approved_at_utc must include a timezone")
    reference = receipt.get("human_approval_reference")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("human_approval_reference is required")
    expected = approval_template()
    for key in ("funding_type", "funding_scope", "allowed_gpu_indices", "approved_gpu_seconds",
                "max_occupation_seconds", "max_occupations", "maximum_card_hours", "estimated_cost_cny",
                "paid_fallback", "resources", "operation"):
        if receipt.get(key) != expected[key]:
            raise ValueError(f"approval receipt changed required field: {key}")
    return hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _occupation_name(occupation_id: str) -> str:
    uuid.UUID(occupation_id)
    return f"{OCCUPATION_PREFIX}{occupation_id}.occupation.json"


def _charge(row: dict) -> float:
    if row.get("status") in ("reserved", "running", "interrupted"):
        return float(row["reserved_gpu_seconds"])
    return float(row.get("charged_gpu_seconds", row["reserved_gpu_seconds"]))


def _new_ledger(receipt_hash: str, approval_id: str) -> dict:
    return {
        "schema_version": LEDGER_SCHEMA,
        "funding_scope": SCOPE,
        "approved_gpu_seconds": LIMIT,
        "approval_id": approval_id,
        "approval_sha256": receipt_hash,
        "sessions": [],
    }


def _check_ledger(data: dict, receipt_hash: str, approval_id: str) -> None:
    expected = _new_ledger(receipt_hash, approval_id)
    for key, value in expected.items():
        if key == "sessions":
            continue
        if data.get(key) != value:
            raise ValueError(f"daily ledger identity changed: {key}")
    rows = data.get("sessions")
    if not isinstance(rows, list) or len(rows) > 1:
        raise ValueError("daily grant permits exactly one occupation")
    if rows and rows[0].get("id") is None:
        raise ValueError("daily occupation identity is missing")


def _read_ledger(path: Path, receipt_hash: str, approval_id: str) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("unreadable daily ledger; refusing to reset accounting") from error
    else:
        data = _new_ledger(receipt_hash, approval_id)
    _check_ledger(data, receipt_hash, approval_id)
    return data


def _verify_inventory(aggregate_path: Path, aggregate: dict, receipt_hash: str, approval_id: str) -> None:
    indexed: dict[str, dict] = {}
    for path in aggregate_path.parent.glob(f"{OCCUPATION_PREFIX}*.occupation.json"):
        stem = path.name.removeprefix(OCCUPATION_PREFIX).removesuffix(".occupation.json")
        try:
            uuid.UUID(stem)
            occupation = _read_ledger(path, receipt_hash, approval_id)
        except (ValueError, OSError) as error:
            raise ValueError("unrecognized or unreadable daily occupation ledger; refusing reset") from error
        if len(occupation["sessions"]) != 1 or occupation["sessions"][0].get("id") != stem:
            raise ValueError("daily occupation ledger identity/file name mismatch")
        indexed[stem] = occupation["sessions"][0]
    if not aggregate_path.exists() and indexed:
        raise ValueError("cumulative daily ledger is missing while occupation ledgers remain")
    rows = aggregate.get("sessions", [])
    by_id = {row.get("id"): row for row in rows}
    if len(by_id) != len(rows) or set(indexed) != set(by_id):
        if indexed or rows:
            raise ValueError("cumulative daily ledger and occupation inventory differ")
    for occupation_id, row in indexed.items():
        aggregate_row = by_id[occupation_id]
        for key in ("reserved_gpu_seconds", "charged_gpu_seconds", "measured_gpu_seconds", "status",
                    "process_group_release_confirmed", "gpu_release_confirmed", "own_process_release_confirmed"):
            if row.get(key) != aggregate_row.get(key):
                raise ValueError("cumulative and occupation records differ")


def _paths(approval: Path, aggregate: Path, occupation: Path) -> tuple[Path, Path, Path]:
    resolved_diag = DIAG.resolve()
    for path in (approval, aggregate, occupation):
        if resolved_diag not in path.resolve().parents:
            raise ValueError("approval and budget records must remain under project diagnostics")
    if approval.name != APPROVAL_NAME or aggregate.name != AGGREGATE_NAME:
        raise ValueError("wrong daily approval or cumulative ledger filename")
    suffix = ".occupation.json"
    if not occupation.name.startswith(OCCUPATION_PREFIX) or not occupation.name.endswith(suffix):
        raise ValueError("wrong daily occupation ledger filename")
    occupation_id = occupation.name[len(OCCUPATION_PREFIX):-len(suffix)]
    try:
        expected_name = _occupation_name(occupation_id)
    except ValueError as error:
        raise ValueError("wrong daily occupation ledger filename") from error
    if occupation.name != expected_name:
        raise ValueError("wrong daily occupation ledger filename")
    if len({approval.resolve(), aggregate.resolve(), occupation.resolve()}) != 3:
        raise ValueError("daily approval and ledger files must be distinct")
    return approval, aggregate, occupation


def validate_execution_request(args, receipt: dict) -> str:
    """Validate the bounded request without inspecting machine or GPU state."""
    occupation_id = str(uuid.UUID(args.occupation_id))
    if args.occupation_ledger.name != _occupation_name(occupation_id):
        raise ValueError("occupation ledger filename does not match the requested UUID")
    if args.seconds != LIMIT or not 120 <= args.seconds <= receipt["max_occupation_seconds"]:
        raise ValueError("the daily grant requires one occupation capped at exactly 7200 seconds")
    if args.gpu_index not in receipt["allowed_gpu_indices"] or args.cpu_affinity != RESOURCE_RECORD["cpu_affinity"]:
        raise ValueError("selected GPU or CPU affinity differs from the fresh approval")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("missing bounded daily command")
    return occupation_id


def _reserve(paths: tuple[Path, Path], *, occupation_id: str, gpu_index: int, command: list[str],
             receipt_hash: str, approval_id: str, start_snapshot: dict) -> None:
    aggregate_path, occupation_path = paths
    with guardian._locked_ledgers([aggregate_path, occupation_path]):
        aggregate = _read_ledger(aggregate_path, receipt_hash, approval_id)
        _verify_inventory(aggregate_path, aggregate, receipt_hash, approval_id)
        if aggregate["sessions"] or occupation_path.exists():
            raise ValueError("single-use daily grant already reserved or consumed; no retry from this grant")
        occupation = _read_ledger(occupation_path, receipt_hash, approval_id)
        row = {
            "id": occupation_id,
            "funding_scope": SCOPE,
            "gpu_index": gpu_index,
            "host": start_snapshot["hostname"],
            "wrapper_pid": os.getpid(),
            "session_cap_seconds": LIMIT,
            "reserved_gpu_seconds": LIMIT,
            "started_epoch": time.time(),
            "command": command,
            "command_sha256": guardian.command_sha256(command),
            "start_snapshot": start_snapshot,
            "status": "reserved",
        }
        aggregate["sessions"].append(row)
        occupation["sessions"].append(dict(row))
        _summarize(aggregate)
        guardian._atomic_json(aggregate_path, aggregate)
        guardian._atomic_json(occupation_path, occupation)


def _summarize(data: dict) -> None:
    charged = sum(_charge(row) for row in data["sessions"])
    data["charged_or_reserved_gpu_seconds"] = charged
    data["remaining_gpu_seconds"] = LIMIT - charged
    data["grant_within_limit"] = charged <= LIMIT
    data["occupation_count"] = len(data["sessions"])


def _set_running(paths: tuple[Path, Path], occupation_id: str, receipt_hash: str, approval_id: str) -> None:
    aggregate_path, occupation_path = paths
    with guardian._locked_ledgers([aggregate_path, occupation_path]):
        aggregate = _read_ledger(aggregate_path, receipt_hash, approval_id)
        occupation = _read_ledger(occupation_path, receipt_hash, approval_id)
        for ledger in (aggregate, occupation):
            row = next((item for item in ledger["sessions"] if item["id"] == occupation_id), None)
            if row is None or row.get("status") != "reserved":
                raise ValueError("daily reserved occupation is missing or changed")
            row["status"] = "running"
            row["process_started_epoch"] = time.time()
        _summarize(aggregate)
        guardian._atomic_json(aggregate_path, aggregate)
        guardian._atomic_json(occupation_path, occupation)


def _finish(paths: tuple[Path, Path], occupation_id: str, receipt_hash: str, approval_id: str, *,
            elapsed: float, exit_status: int, end_snapshot: dict | None,
            process_group_release_confirmed: bool, gpu_release_confirmed: bool,
            gpu_release_reason: str) -> tuple[str, float]:
    aggregate_path, occupation_path = paths
    own_release = process_group_release_confirmed and gpu_release_confirmed
    charged = elapsed + FINALIZATION_CHARGE_RESERVE_SECONDS if own_release else LIMIT
    status = "finished" if own_release else "interrupted"
    if charged > LIMIT:
        status = "interrupted"
    with guardian._locked_ledgers([aggregate_path, occupation_path]):
        aggregate = _read_ledger(aggregate_path, receipt_hash, approval_id)
        occupation = _read_ledger(occupation_path, receipt_hash, approval_id)
        for ledger in (aggregate, occupation):
            row = next((item for item in ledger["sessions"] if item["id"] == occupation_id), None)
            if row is None or row.get("status") not in ("running", "reserved"):
                raise ValueError("daily occupation cannot be finalized")
            row.update(status=status, ended_epoch=time.time(), occupied_seconds=elapsed,
                       occupied_gpu_seconds=elapsed, measured_gpu_seconds=elapsed,
                       charged_gpu_seconds=charged, exit_status=exit_status,
                       within_session_cap=charged <= LIMIT, end_snapshot=end_snapshot,
                       process_group_release_confirmed=process_group_release_confirmed,
                       gpu_release_confirmed=gpu_release_confirmed,
                       gpu_release_reason=gpu_release_reason,
                       own_process_release_confirmed=own_release)
        _summarize(aggregate)
        guardian._atomic_json(aggregate_path, aggregate)
        guardian._atomic_json(occupation_path, occupation)
    return status, charged


def execute(args) -> int:
    approval_path, aggregate_path, occupation_path = _paths(args.approval_receipt, args.aggregate_ledger, args.occupation_ledger)
    receipt = json.loads(approval_path.read_text(encoding="utf-8"))
    receipt_hash = validate_approval(receipt)
    occupation_id = validate_execution_request(args, receipt)
    if os.environ.get("SLURM_JOB_ID"):
        raise ValueError("free daily occupation refuses a Slurm allocation")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    cpus = guardian.parse_cpu_affinity(args.cpu_affinity)
    start_snapshot = guardian.host_snapshot(args.gpu_index, cpus)
    guardian.validate_free_host(start_snapshot, cpus)
    paths = (aggregate_path, occupation_path)
    started = time.monotonic()
    _reserve(paths, occupation_id=occupation_id, gpu_index=args.gpu_index, command=command,
             receipt_hash=receipt_hash, approval_id=receipt["approval_id"], start_snapshot=start_snapshot)
    process = None
    exit_status = 125
    interrupted = None

    def stop(signum, _frame):
        nonlocal interrupted
        interrupted = signum
        if process is not None:
            guardian._terminate_group(process, grace_seconds=5)

    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, stop)
    try:
        _set_running(paths, occupation_id, receipt_hash, receipt["approval_id"])
        selected_cpus = ",".join(str(cpu) for cpu in cpus)
        wrapped = ["taskset", "-c", selected_cpus, "env", "OMP_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=1", *command]
        process = subprocess.Popen(wrapped, start_new_session=True)
        wait_seconds = max(0.1, args.seconds - FINALIZE_RESERVE_SECONDS - (time.monotonic() - started))
        try:
            exit_status = process.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            guardian._terminate_group(process, grace_seconds=5)
            exit_status = 124
        if interrupted is not None:
            exit_status = 128 + int(interrupted)
    except BaseException as error:
        print(f"daily phase guard: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        exit_status = 125
    finally:
        process_group_release = True if process is None else guardian._terminate_group(process, grace_seconds=2)
        try:
            end_snapshot = guardian.host_snapshot(args.gpu_index, cpus)
        except BaseException as error:
            end_snapshot = {"snapshot_error": f"{type(error).__name__}: {error}"}
        gpu_release, gpu_reason = guardian.gpu_release_evidence(end_snapshot, args.gpu_index)
        elapsed = time.monotonic() - started
        status, charged = _finish(paths, occupation_id, receipt_hash, receipt["approval_id"], elapsed=elapsed,
            exit_status=exit_status, end_snapshot=end_snapshot,
            process_group_release_confirmed=process_group_release,
            gpu_release_confirmed=gpu_release, gpu_release_reason=gpu_reason)
        if status != "finished" or charged > args.seconds:
            exit_status = 125
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    print(json.dumps({"occupation_id": occupation_id, "scope": SCOPE, "measured_gpu_seconds": elapsed,
        "charged_gpu_seconds": charged, "occupation_status": status, "exit_status": exit_status,
        "process_group_release_confirmed": process_group_release,
        "gpu_release_confirmed": gpu_release, "gpu_release_reason": gpu_reason,
        "approval_id": receipt["approval_id"]}, sort_keys=True), flush=True)
    return exit_status


def verify_running(args) -> int:
    approval_path, aggregate_path, occupation_path = _paths(args.approval_receipt, args.aggregate_ledger, args.occupation_ledger)
    receipt = json.loads(approval_path.read_text(encoding="utf-8"))
    receipt_hash = validate_approval(receipt)
    occupation_id = str(uuid.UUID(args.occupation_id))
    if occupation_path.name != _occupation_name(occupation_id):
        raise ValueError("occupation ledger filename does not match the requested UUID")
    with guardian._locked_ledgers([aggregate_path, occupation_path]):
        aggregate = _read_ledger(aggregate_path, receipt_hash, receipt["approval_id"])
        _verify_inventory(aggregate_path, aggregate, receipt_hash, receipt["approval_id"])
        occupation = _read_ledger(occupation_path, receipt_hash, receipt["approval_id"])
        row = next((item for item in aggregate["sessions"] if item.get("id") == occupation_id), None)
        local = next((item for item in occupation["sessions"] if item.get("id") == occupation_id), None)
        if row is None or local is None or row.get("status") != "running" or local.get("status") != "running":
            raise ValueError("internal daily phase requires a matching running reservation")
        reserved_command = row.get("command", [])
        if (row.get("gpu_index") != args.gpu_index or
                row.get("command_sha256") != args.command_sha256 and
                guardian.command_sha256(reserved_command[:-1]) != args.command_sha256):
            raise ValueError("internal daily phase command or GPU differs from its reservation")
        if row.get("host") not in ("gln01", "ibc13b02n04") or not row.get("process_started_epoch"):
            raise ValueError("daily reservation lacks an accepted host/process start")
        if type(row.get("wrapper_pid")) is not int or not guardian._has_process_ancestor(os.getpid(), row["wrapper_pid"]):
            raise ValueError("internal daily phase is outside the guardian process tree")
    print(json.dumps({"occupation_id": occupation_id, "scope": SCOPE, "status": "running"}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("template", help="print the unapproved fresh daily grant template")
    fingerprint = sub.add_parser("fingerprint-command", help="fingerprint a command vector")
    fingerprint.add_argument("command", nargs=argparse.REMAINDER)
    run = sub.add_parser("execute", help="run one approved daily occupation")
    run.add_argument("--approval-receipt", type=Path, required=True)
    run.add_argument("--aggregate-ledger", type=Path, required=True)
    run.add_argument("--occupation-ledger", type=Path, required=True)
    run.add_argument("--occupation-id", required=True)
    run.add_argument("--seconds", type=int, required=True)
    run.add_argument("--gpu-index", type=int, required=True)
    run.add_argument("--cpu-affinity", default=CPU_AFFINITY)
    run.add_argument("command", nargs=argparse.REMAINDER)
    verify = sub.add_parser("verify-running", help="verify the active daily reservation for its child phase")
    verify.add_argument("--approval-receipt", type=Path, required=True)
    verify.add_argument("--aggregate-ledger", type=Path, required=True)
    verify.add_argument("--occupation-ledger", type=Path, required=True)
    verify.add_argument("--occupation-id", required=True)
    verify.add_argument("--gpu-index", type=int, required=True)
    verify.add_argument("--command-sha256", required=True)
    args = parser.parse_args(argv)
    if args.action == "template":
        print(json.dumps(approval_template(), indent=2, sort_keys=True))
        return 0
    if args.action == "fingerprint-command":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            raise ValueError("command vector is required")
        print(guardian.command_sha256(command))
        return 0
    return verify_running(args) if args.action == "verify-running" else execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
