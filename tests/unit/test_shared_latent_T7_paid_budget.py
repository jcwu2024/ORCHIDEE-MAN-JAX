"""CPU-only protection for paid idle checks and the proposed replacement cap."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.hpc import shared_latent_T7_paid_budget as paid


def idle_clock(monkeypatch, replies, *, query_seconds=0):
    elapsed = [0.0]
    responses = iter(replies)
    timeouts = []
    monkeypatch.setattr(paid.time, 'monotonic', lambda: elapsed[0])
    monkeypatch.setattr(paid.time, 'sleep', lambda seconds: elapsed.__setitem__(0, elapsed[0]+seconds))
    monkeypatch.setattr(paid, 'atomic', lambda *args: None)

    def query(command, *, text, timeout):
        timeouts.append(timeout)
        elapsed[0] += query_seconds
        return next(responses)

    monkeypatch.setattr(paid.subprocess, 'check_output', query)
    return elapsed, timeouts


def test_utilization_tail_must_reach_exact_zero(monkeypatch):
    clock, _ = idle_clock(monkeypatch, ['0, 5', '', '0, 2', '', '0, 0', ''])
    data = {}
    row = paid.assigned_gpu_idle('0', 'candidate-model', Path('unused'), data)
    assert row['passed'] and row['utilization_percent'] == 0
    assert clock[0] == 4
    assert [r['passed'] for r in data['assigned_GPU_idle_checks']] == [False, False, True]


def test_external_pid_fails_without_wait(monkeypatch):
    clock, _ = idle_clock(monkeypatch, ['0, 5', '12345'])
    with pytest.raises(paid.PaidResourceError):
        paid.assigned_gpu_idle('0', 'candidate-model', Path('unused'), {})
    assert clock[0] == 0


def test_utilization_wait_is_bounded_including_queries(monkeypatch):
    clock, timeouts = idle_clock(monkeypatch, ['0, 5', ''], query_seconds=9)
    with pytest.raises(paid.PaidResourceError):
        paid.assigned_gpu_idle('0', 'candidate-model', Path('unused'), {})
    assert clock[0] == 20
    assert timeouts == [10, 10]


def test_second_query_shares_remaining_deadline(monkeypatch):
    clock, timeouts = idle_clock(monkeypatch, ['0, 5', ''], query_seconds=11)
    # The stub advances past its timeout deliberately, exposing the second
    # query's reduced allowance; real subprocess.run enforces that allowance.
    with pytest.raises(paid.PaidResourceError):
        paid.assigned_gpu_idle('0', 'candidate-model', Path('unused'), {})
    assert timeouts == [10, 9]


def proposal():
    root = Path(__file__).resolve().parents[2]
    return json.loads((root/'outputs/diagnostics/shared-latent-reader-20261004/latent-reader-T7-20261005-paid-replacement.approval.json').read_text())


def test_unapproved_replacement_cannot_execute(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '999999')
    with pytest.raises(ValueError, match='approved'):
        paid.validate_approval(proposal(), 'ibc13b03n04', replacement=True)


def test_approved_receipt_requires_exact_real_job_and7140_envelope(monkeypatch):
    data = proposal()
    data.update(approved=True, approval_origin='explicit_human_user',
                approval_note='CPU fixture only', approved_at='fixture', authorized_SLURM_job_id='999999')
    monkeypatch.setenv('SLURM_JOB_ID', '999999')
    paid.validate_approval(data, 'ibc13b03n04', replacement=True)
    monkeypatch.setenv('SLURM_JOB_ID', '999998')
    with pytest.raises(ValueError, match='job ID'):
        paid.validate_approval(data, 'ibc13b03n04', replacement=True)
    monkeypatch.setenv('SLURM_JOB_ID', '999999')
    data['wall_seconds'] = 7200
    with pytest.raises(ValueError, match='wall_seconds'):
        paid.validate_approval(data, 'ibc13b03n04', replacement=True)


def test_aggregate_reserves_once_and_preserves_original(monkeypatch, tmp_path):
    if os.name == 'nt':
        # Windows checks accounting and source preservation; the actual Linux
        # flock is exercised by the target-host check, not emulated here.
        monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(LOCK_EX=1, flock=lambda *args: None))
    monkeypatch.setattr(paid, 'DIAG', tmp_path)
    monkeypatch.setenv('SLURM_JOB_ID', '999999')
    old = {'SLURM_job_id': '15580715', 'funding_scope': paid.SCOPE, 'status': 'resource_failed',
           'allocated_CPUs': 8, 'GPUs': 1, 'occupied_gpu_seconds': 26.612438168376684,
           'phases': [{'status': 'finished', 'child_release_confirmed': True}]}
    original_path = tmp_path/(paid.PREFIX+'-paid.gpu-budget.first-attempt.json')
    original_bytes = json.dumps(old).encode()
    original_path.write_bytes(original_bytes)
    aggregate_path = tmp_path/'aggregate.json'
    data = {'schema_version': 'latent_T7_paid_aggregate_budget_v1', 'original_job_id': '15580715',
            'wall_seconds_cap': 7200, 'cost_CNY_cap': 5.52, 'rate_CNY_per_hour': 2.76,
            'original_allocation_reserved_seconds': 60, 'replacement_wall_seconds_cap': 7140,
            'original_helper_ledger': str(original_path), 'original_scheduler_elapsed_seconds': 28,
            'original_helper_occupied_gpu_seconds': old['occupied_gpu_seconds'],
            'original_allocation_cost_CNY_reserve': 0.046, 'replacement_attempts': []}
    aggregate_path.write_text(json.dumps(data))
    args = SimpleNamespace(aggregate_budget=aggregate_path, approval=tmp_path/'approval.json', ledger=tmp_path/'new-ledger.json')
    paid.aggregate_replacement(args, {})
    actual = json.loads(aggregate_path.read_text())
    assert actual['total_reserved_allocation_seconds'] == 7200
    assert actual['total_reserved_cost_CNY'] == pytest.approx(5.52)
    assert original_path.read_bytes() == original_bytes
    with pytest.raises(ValueError, match='already reserved'):
        paid.aggregate_replacement(args, {})
    final = {'status': 'resource_failed', 'occupied_gpu_seconds': 100, 'runtime_within_wall': True}
    paid.aggregate_replacement(args, {}, final=final)
    assert json.loads(aggregate_path.read_text())['total_reserved_allocation_seconds'] == 7200
    assert original_path.read_bytes() == original_bytes
