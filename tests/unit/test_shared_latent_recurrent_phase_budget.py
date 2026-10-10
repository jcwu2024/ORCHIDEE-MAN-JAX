from __future__ import annotations

from datetime import datetime, timezone
import json
import tempfile
from pathlib import Path
import uuid

import pytest

from scripts.hpc import shared_latent_recurrent_phase_budget as budget


def approved_receipt():
    receipt = budget.approval_template()
    receipt.update(
        approval_id=str(uuid.uuid4()),
        approved=True,
        approved_at_utc=datetime.now(timezone.utc).isoformat(),
        human_approval_reference="test fixture for budget validation",
    )
    return receipt


def paths(root: Path, phase: str, occupation_id: str):
    return (
        root / budget.AGGREGATE_NAME,
        root / budget.phase_occupation_path(phase, occupation_id),
    )


def reserve(root: Path, receipt: dict, phase: str, seconds: int):
    occupation_id = str(uuid.uuid4())
    aggregate, occupation = paths(root, phase, occupation_id)
    receipt_hash = budget.validate_approval(receipt)
    session_id = budget._reserve(
        (aggregate, occupation),
        phase=phase,
        occupation_id=occupation_id,
        gpu_index=0,
        seconds=seconds,
        command=["bounded-test-command"],
        receipt_hash=receipt_hash,
        approval_id=receipt["approval_id"],
        start_snapshot={"hostname": "gln01"},
    )
    return occupation_id, (aggregate, occupation), session_id, receipt_hash


def settle(receipt: dict, ledger_paths, session_id: str, receipt_hash: str, seconds: float):
    budget._mark_running(ledger_paths, session_id, receipt_hash=receipt_hash, approval_id=receipt["approval_id"])
    budget._finish(
        ledger_paths,
        session_id,
        receipt_hash=receipt_hash,
        approval_id=receipt["approval_id"],
        elapsed=seconds,
        charged_gpu_seconds=seconds,
        status="finished",
        exit_status=0,
        host_end={"hostname": "gln01"},
        process_group_release_confirmed=True,
        gpu_release_confirmed=True,
        gpu_release_reason="confirmed",
    )


def test_unapproved_template_is_rejected():
    with pytest.raises(ValueError, match="approved=true"):
        budget.validate_approval(budget.approval_template())


def test_receipt_must_preserve_protected_pool_split_and_resources():
    receipt = approved_receipt()
    assert budget.validate_approval(receipt)
    receipt["funding_pools_gpu_seconds"]["matched"] = 0
    with pytest.raises(ValueError, match="funding_pools_gpu_seconds"):
        budget.validate_approval(receipt)


def test_compile_retries_cannot_spend_the_protected_matched_pool():
    receipt = approved_receipt()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        total_compile = 0
        for phase, seconds in (("compile-original", 4800), ("compile-candidate", 7200)):
            _, ledger_paths, session_id, receipt_hash = reserve(root, receipt, phase, seconds)
            settle(receipt, ledger_paths, session_id, receipt_hash, seconds)
            total_compile += seconds
        assert total_compile == budget.FUNDING_POOLS["compile"]
        with pytest.raises(ValueError, match="protected compile pool exhausted"):
            reserve(root, receipt, "compile-original", 120)
        _, ledger_paths, session_id, receipt_hash = reserve(root, receipt, "matched", 3600)
        settle(receipt, ledger_paths, session_id, receipt_hash, 3600)


def test_unresolved_reservation_blocks_overlapping_phase_spend():
    receipt = approved_receipt()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        reserve(root, receipt, "compile-original", 120)
        with pytest.raises(ValueError, match="unsettled reservation"):
            reserve(root, receipt, "matched", 120)


def test_internal_phase_requires_matching_running_reservation_and_gpu():
    receipt = approved_receipt()
    with tempfile.TemporaryDirectory() as directory:
        _, ledger_paths, session_id, receipt_hash = reserve(Path(directory), receipt, "compile-original", 120)
        with pytest.raises(ValueError, match="running budget reservation"):
            budget.verify_running_occupation(
                ledger_paths,
                phase="compile-original",
                occupation_id=session_id,
                gpu_index=0,
                expected_command_sha256=budget.command_sha256(["bounded-test-command"]),
                receipt_hash=receipt_hash,
                approval_id=receipt["approval_id"],
            )
        budget._mark_running(ledger_paths, session_id, receipt_hash=receipt_hash, approval_id=receipt["approval_id"])
        budget.verify_running_occupation(
            ledger_paths,
            phase="compile-original",
            occupation_id=session_id,
            gpu_index=0,
            expected_command_sha256=budget.command_sha256(["bounded-test-command"]),
            receipt_hash=receipt_hash,
            approval_id=receipt["approval_id"],
        )
        with pytest.raises(ValueError, match="identity or phase"):
            budget.verify_running_occupation(
                ledger_paths,
                phase="compile-original",
                occupation_id=session_id,
                gpu_index=1,
                expected_command_sha256=budget.command_sha256(["bounded-test-command"]),
                receipt_hash=receipt_hash,
                approval_id=receipt["approval_id"],
            )
        with pytest.raises(ValueError, match="differs from the reserved command"):
            budget.verify_running_occupation(
                ledger_paths,
                phase="compile-original",
                occupation_id=session_id,
                gpu_index=0,
                expected_command_sha256=budget.command_sha256(["replayed-with-different-args"]),
                receipt_hash=receipt_hash,
                approval_id=receipt["approval_id"],
            )
        for ledger_path in ledger_paths:
            data = json.loads(ledger_path.read_text(encoding="utf-8"))
            data["sessions"][0]["wrapper_pid"] = 2_000_000_000
            ledger_path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="guardian-launched process tree"):
            budget.verify_running_occupation(
                ledger_paths,
                phase="compile-original",
                occupation_id=session_id,
                gpu_index=0,
                expected_command_sha256=budget.command_sha256(["bounded-test-command"]),
                receipt_hash=receipt_hash,
                approval_id=receipt["approval_id"],
            )


def test_unconfirmed_process_release_keeps_full_reservation_active():
    receipt = approved_receipt()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _, ledger_paths, session_id, receipt_hash = reserve(root, receipt, "compile-original", 120)
        budget._mark_running(ledger_paths, session_id, receipt_hash=receipt_hash, approval_id=receipt["approval_id"])
        budget._finish(
            ledger_paths,
            session_id,
            receipt_hash=receipt_hash,
            approval_id=receipt["approval_id"],
            elapsed=8,
            charged_gpu_seconds=68,
            status="finished",
            exit_status=0,
            host_end={"hostname": "gln01"},
            process_group_release_confirmed=False,
            gpu_release_confirmed=True,
            gpu_release_reason="confirmed",
        )
        aggregate = budget._read_ledger(ledger_paths[0], phase=None, receipt_hash=receipt_hash,
                                        approval_id=receipt["approval_id"])
        row = aggregate["sessions"][0]
        assert row["status"] == "interrupted"
        assert row["own_process_release_confirmed"] is False
        assert budget._charge(row) == 120
        with pytest.raises(ValueError, match="unsettled reservation"):
            reserve(root, receipt, "matched", 120)


def test_failed_final_gpu_snapshot_keeps_full_reservation_active():
    receipt = approved_receipt()
    ok, reason = budget.gpu_release_evidence({"snapshot_error": "NVML timeout"}, 0)
    assert ok is False
    assert reason == "final_gpu_snapshot_failed"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _, ledger_paths, session_id, receipt_hash = reserve(root, receipt, "compile-original", 120)
        budget._mark_running(ledger_paths, session_id, receipt_hash=receipt_hash, approval_id=receipt["approval_id"])
        budget._finish(
            ledger_paths,
            session_id,
            receipt_hash=receipt_hash,
            approval_id=receipt["approval_id"],
            elapsed=10,
            charged_gpu_seconds=70,
            status="finished",
            exit_status=0,
            host_end={"snapshot_error": "NVML timeout"},
            process_group_release_confirmed=True,
            gpu_release_confirmed=ok,
            gpu_release_reason=reason,
        )
        aggregate = budget._read_ledger(ledger_paths[0], phase=None, receipt_hash=receipt_hash,
                                        approval_id=receipt["approval_id"])
        row = aggregate["sessions"][0]
        assert row["status"] == "interrupted"
        assert row["process_group_release_confirmed"] is True
        assert row["gpu_release_confirmed"] is False
        assert row["gpu_release_reason"] == reason
        assert budget._charge(row) == 120
        with pytest.raises(ValueError, match="unsettled reservation"):
            reserve(root, receipt, "matched", 120)


def test_visible_gpu_pid_does_not_confirm_release_or_get_attributed():
    snapshot = {
        "gpu_index": 1,
        "gpu_compute_processes": ["8421, external-python, 512"],
        "gpu_memory_used_mib": 512,
        "gpu_utilization_percent": 4,
    }
    ok, reason = budget.gpu_release_evidence(snapshot, 1)
    assert ok is False
    assert reason == "visible_gpu_processes_remain_or_appeared"
    assert snapshot["gpu_compute_processes"] == ["8421, external-python, 512"]
    receipt = approved_receipt()
    with tempfile.TemporaryDirectory() as directory:
        _, ledger_paths, session_id, receipt_hash = reserve(Path(directory), receipt, "compile-original", 120)
        budget._mark_running(ledger_paths, session_id, receipt_hash=receipt_hash, approval_id=receipt["approval_id"])
        budget._finish(
            ledger_paths,
            session_id,
            receipt_hash=receipt_hash,
            approval_id=receipt["approval_id"],
            elapsed=12,
            charged_gpu_seconds=72,
            status="finished",
            exit_status=0,
            host_end=snapshot,
            process_group_release_confirmed=True,
            gpu_release_confirmed=ok,
            gpu_release_reason=reason,
        )
        aggregate = budget._read_ledger(ledger_paths[0], phase=None, receipt_hash=receipt_hash,
                                        approval_id=receipt["approval_id"])
        row = aggregate["sessions"][0]
        assert row["status"] == "interrupted"
        assert row["gpu_release_reason"] == reason
        assert row["end_snapshot"]["gpu_compute_processes"] == snapshot["gpu_compute_processes"]
        assert budget._charge(row) == 120


def test_zero_memory_no_pid_with_nonzero_utilization_confirms_release():
    ok, reason = budget.gpu_release_evidence(
        {
            "gpu_index": 1,
            "gpu_compute_processes": [],
            "gpu_memory_used_mib": 0,
            "gpu_utilization_percent": 25,
        },
        1,
    )
    assert ok is True
    assert reason == "confirmed"


def test_clean_group_and_gpu_release_settles_measured_plus_finalization_reserve():
    receipt = approved_receipt()
    snapshot = {
        "gpu_index": 0,
        "gpu_compute_processes": [],
        "gpu_memory_used_mib": 0,
        "gpu_utilization_percent": 0,
    }
    gpu_released, reason = budget.gpu_release_evidence(snapshot, 0)
    assert gpu_released is True
    with tempfile.TemporaryDirectory() as directory:
        _, ledger_paths, session_id, receipt_hash = reserve(Path(directory), receipt, "compile-original", 120)
        budget._mark_running(ledger_paths, session_id, receipt_hash=receipt_hash, approval_id=receipt["approval_id"])
        budget._finish(
            ledger_paths,
            session_id,
            receipt_hash=receipt_hash,
            approval_id=receipt["approval_id"],
            elapsed=10,
            charged_gpu_seconds=10 + budget.FINALIZATION_CHARGE_RESERVE_SECONDS,
            status="finished",
            exit_status=0,
            host_end=snapshot,
            process_group_release_confirmed=True,
            gpu_release_confirmed=gpu_released,
            gpu_release_reason=reason,
        )
        aggregate = budget._read_ledger(ledger_paths[0], phase=None, receipt_hash=receipt_hash,
                                        approval_id=receipt["approval_id"])
        row = aggregate["sessions"][0]
        assert row["status"] == "finished"
        assert row["measured_gpu_seconds"] == 10
        assert row["charged_gpu_seconds"] == 70
        assert budget._charge(row) == 70


def test_cpu_affinity_parser_requires_eight_distinct_ids():
    assert budget.parse_cpu_affinity("8-15") == list(range(8, 16))
    with pytest.raises(ValueError, match="exactly 8"):
        budget.parse_cpu_affinity("8-14")
    with pytest.raises(ValueError, match="exactly 8"):
        budget.parse_cpu_affinity("8-14,14")


def test_free_host_gate_checks_actual_cpu_gpu_and_memory_snapshot():
    snapshot = {
        "hostname": "ibc13b02n04",
        "slurm_job_id": None,
        "available_cpu_count": 16,
        "available_cpu_affinity": list(range(16)),
        "selected_cpu_affinity": list(range(8, 16)),
        "host_available_memory_gib": 80,
        "gpu_compute_processes": [],
        "gpu_memory_used_mib": 0,
        "gpu_utilization_percent": 0,
        "gpu_name": "Tesla V100-SXM2-32GB",
    }
    budget.validate_free_host(snapshot, list(range(8, 16)))
    snapshot["gpu_compute_processes"] = ["3012, python, 20"]
    with pytest.raises(ValueError, match="compute processes"):
        budget.validate_free_host(snapshot, list(range(8, 16)))
