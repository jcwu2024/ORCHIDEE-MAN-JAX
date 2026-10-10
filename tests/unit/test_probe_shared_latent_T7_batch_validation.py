import json
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts/hpc"))

_PROBE_SPEC = importlib.util.spec_from_file_location(
    "probe_shared_latent_T7_batch", _ROOT / "scripts/hpc/probe_shared_latent_T7_batch.py")
probe = importlib.util.module_from_spec(_PROBE_SPEC)
_PROBE_SPEC.loader.exec_module(probe)
_PAID_SPEC = importlib.util.spec_from_file_location(
    "run_shared_latent_T7_batch_paid", _ROOT / "scripts/hpc/run_shared_latent_T7_batch_paid.py")
paid = importlib.util.module_from_spec(_PAID_SPEC)
_PAID_SPEC.loader.exec_module(paid)


def limits(*, wall_seconds, paid_ledger=None):
    return SimpleNamespace(wall_seconds=wall_seconds, lower_seconds=600,
        backend_seconds=9000, call_seconds=180, load_seconds=360,
        pairs=3, post_compile_reserve_seconds=600, paid_ledger=paid_ledger)


def allocation_env(job_id="90001", node="ibc13b02n04"):
    return {"SLURM_JOB_ID": job_id, "SLURMD_NODENAME": node,
        "SLURM_JOB_PARTITION": "gnall", "SLURM_CPUS_PER_TASK": "8",
        "SLURM_NTASKS": "1", "SLURM_GPUS_ON_NODE": "1",
        "SLURM_MEM_PER_NODE": "65536", "SLURM_JOB_NUM_NODES": "1",
        "CUDA_VISIBLE_DEVICES": "0"}


def test_free_wall_cap_remains_7200_seconds():
    probe.validate_deadline_scope(limits(wall_seconds=7200))
    with pytest.raises(ValueError, match="bounded deadlines"):
        probe.validate_deadline_scope(limits(wall_seconds=7201))


def test_actual_paid_runner_argv_passes_ledger_bound_deadline(tmp_path):
    ledger_path = tmp_path / "replacement.gpu-budget.json"
    ledger = {"schema_version": "latent_T7_B4_paid_followup_v1", "approved": True,
        "funding_scope": paid.SCOPE, "SLURM_job_id": "90001",
        "node": "ibc13b02n04", "execution_node": "ibc13b02n04",
        "allocated_CPUs": 8, "GPUs": 1, "wall_seconds": 10680,
        "probe_wall_seconds": 10080, "new_B4_graph_compile_cap": 1,
        "new_B4_compile_attempt_count": 0}
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    command = paid.candidate_command(SimpleNamespace(ledger=ledger_path),
        "compile-and-compare", tmp_path / "replacement.compile-and-compare.json", 10080)
    parsed = probe.parser().parse_args(command[1:])
    probe.validate_deadline_scope(parsed, environ=allocation_env())
    assert parsed.wall_seconds == ledger["probe_wall_seconds"]
    assert parsed.paid_ledger == ledger_path


def test_replacement_runner_profile_generates_validator_accepted_full_argv(tmp_path):
    approval = paid.replacement_approval_template()
    node = paid.ELIGIBLE_NODES[0]
    ledger_path = tmp_path / (paid.REPLACEMENT_PREFIX + ".gpu-budget.json")
    ledger = paid.initial_paid_ledger_fields(approval, node, "90001")
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    command = paid.candidate_command(SimpleNamespace(ledger=ledger_path),
        "compile-and-compare", tmp_path / "replacement.compile-and-compare.json",
        ledger["probe_wall_seconds"])
    parsed = probe.parser().parse_args(command[1:])
    probe.validate_deadline_scope(parsed, environ=allocation_env(node=node))
    assert "--paid-ledger" in command
    assert parsed.wall_seconds == 10080
    assert ledger["wall_seconds"] == approval["wall_seconds"] == 10680


def test_paid_deadline_rejects_cross_job_or_overledger_window(tmp_path):
    ledger_path = tmp_path / "replacement.gpu-budget.json"
    ledger = {"schema_version": "latent_T7_B4_paid_followup_v1", "approved": True,
        "funding_scope": paid.SCOPE, "SLURM_job_id": "90001",
        "node": "ibc13b02n04", "execution_node": "ibc13b02n04",
        "allocated_CPUs": 8, "GPUs": 1, "wall_seconds": 10680,
        "probe_wall_seconds": 10080, "new_B4_graph_compile_cap": 1,
        "new_B4_compile_attempt_count": 0}
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    with pytest.raises(ValueError, match="SLURM_job_id"):
        probe.validate_deadline_scope(limits(wall_seconds=10080, paid_ledger=ledger_path),
            environ=allocation_env(job_id="90002"))
    with pytest.raises(ValueError, match="bounded deadlines"):
        probe.validate_deadline_scope(limits(wall_seconds=10081, paid_ledger=ledger_path),
            environ=allocation_env())
    ledger["funding_scope"] = "other_scope"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    with pytest.raises(ValueError, match="funding_scope"):
        probe.validate_deadline_scope(limits(wall_seconds=10080, paid_ledger=ledger_path),
            environ=allocation_env())
    ledger["funding_scope"] = paid.SCOPE
    ledger["new_B4_compile_attempt_count"] = 1
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    with pytest.raises(ValueError, match="new_B4_compile_attempt_count"):
        probe.validate_deadline_scope(limits(wall_seconds=10080, paid_ledger=ledger_path),
            environ=allocation_env())
