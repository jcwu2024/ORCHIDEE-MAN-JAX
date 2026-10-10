import importlib.util
import json
import shlex
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_RUNNER = Path(__file__).resolve().parents[2] / "scripts/hpc/run_shared_latent_T7_batch_paid.py"
_SPEC = importlib.util.spec_from_file_location("run_shared_latent_T7_batch_paid", _RUNNER)
paid = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(paid)


def test_prepared_approval_is_explicitly_unapproved_and_cost_is_exact():
    receipt = paid.approval_template()
    assert receipt["approved"] is False
    assert paid.expected_cost() == 8.28
    paid.validate_plan(receipt, approved_required=False)
    receipt["approved"] = True
    with pytest.raises(ValueError, match="approval_origin"):
        paid.validate_plan(receipt, approved_required=True)


def test_approved_receipt_requires_human_provenance_and_exact_scope():
    receipt = paid.approval_template()
    receipt.update(approved=True, approval_origin="explicit_human_user",
                   approval_note="Approved exact resources and cost", approved_at="2026-10-09T00:00:00+08:00")
    paid.validate_plan(receipt, approved_required=True)
    receipt["wall_seconds"] = 10801
    with pytest.raises(ValueError, match="wall_seconds"):
        paid.validate_plan(receipt, approved_required=True)


def test_replacement_profile_is_unapproved_bounded_and_costs_within_original_ceiling():
    receipt = paid.replacement_approval_template()
    paid.validate_plan(receipt, approved_required=False)
    assert receipt["wall_seconds"] == 10680
    assert receipt["worst_case_CNY"] == 8.19
    assert receipt["prior_charge_seconds"] == 120
    assert receipt["prior_charge_ceiling_CNY"] == 0.092
    assert receipt["aggregate_wall_seconds"] == 10800
    assert receipt["aggregate_cost_ceiling_CNY"] == 8.28
    assert tuple(receipt["eligible_nodes"]) == paid.ELIGIBLE_NODES
    assert paid.phase_caps(receipt["wall_seconds"]) == {
        "runtime_preflight": 240, "probe": 10080, "cleanup_reserve": 360}


def test_replacement_allocation_accepts_only_approved_nodes(monkeypatch):
    env = {"SLURM_JOB_ID": "90001", "SLURM_JOB_PARTITION": "gnall",
        "SLURM_CPUS_PER_TASK": "8", "SLURM_NTASKS": "1",
        "SLURM_GPUS_ON_NODE": "1", "SLURM_MEM_PER_NODE": "65536",
        "SLURM_JOB_NUM_NODES": "1", "SLURMD_NODENAME": paid.ELIGIBLE_NODES[1],
        "CUDA_VISIBLE_DEVICES": "3"}
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    node, gpu = paid.validate_allocation(None, allowed_nodes=list(paid.ELIGIBLE_NODES))
    assert (node, gpu) == (paid.ELIGIBLE_NODES[1], "3")
    monkeypatch.setenv("SLURMD_NODENAME", paid.NODE)
    with pytest.raises(ValueError, match="outside the approved eligible node set"):
        paid.validate_allocation(None, allowed_nodes=list(paid.ELIGIBLE_NODES))


def approved_receipt():
    receipt = paid.approval_template()
    receipt.update(approved=True, approval_origin="explicit_human_user",
                   approval_note="Approved exact resources and cost",
                   approved_at="2026-10-09T00:00:00+08:00")
    return receipt


def placement_receipt(approval, approval_path):
    return {
        "schema_version": paid.PLACEMENT_SCHEMA,
        "job_id": paid.JOB_ID,
        "requested_node": paid.NODE,
        "execution_node": paid.EXECUTION_NODE,
        "submit_command": paid.submit_command(paid.NODE),
        "resource_scope": paid.placement_scope(approval),
        "approval_receipt_sha256": paid.sha256(approval_path),
    }


def test_placement_receipt_binds_original_approval_and_fixed_job(tmp_path):
    approval = approved_receipt()
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(approval))
    placement = placement_receipt(approval, approval_path)
    paid.validate_placement(placement, approval, approval_path=approval_path,
                            expected_job_id=paid.JOB_ID)
    placement["resource_scope"]["wall_seconds"] = 10801
    with pytest.raises(ValueError, match="resource_scope"):
        paid.validate_placement(placement, approval, approval_path=approval_path)


def test_allocation_uses_execution_node_only_with_valid_placement(monkeypatch, tmp_path):
    approval = approved_receipt()
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(approval))
    placement = placement_receipt(approval, approval_path)
    env = {"SLURM_JOB_ID": paid.JOB_ID, "SLURM_JOB_PARTITION": "gnall",
        "SLURM_CPUS_PER_TASK": "8", "SLURM_NTASKS": "1",
        "SLURM_GPUS_ON_NODE": "1", "SLURM_MEM_PER_NODE": "65536",
        "SLURM_JOB_NUM_NODES": "1", "SLURMD_NODENAME": paid.EXECUTION_NODE,
        "CUDA_VISIBLE_DEVICES": "3"}
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    node, gpu = paid.validate_allocation(paid.NODE, placement=placement,
        approval=approval, approval_path=approval_path)
    assert (node, gpu) == (paid.EXECUTION_NODE, "3")
    with pytest.raises(ValueError, match="differs from this allocation"):
        monkeypatch.setenv("SLURM_JOB_ID", "15666050")
        paid.validate_allocation(paid.NODE, placement=placement,
            approval=approval, approval_path=approval_path)
    monkeypatch.setenv("SLURM_JOB_ID", paid.JOB_ID)
    with pytest.raises(ValueError, match="approved placement"):
        monkeypatch.setenv("SLURMD_NODENAME", paid.NODE)
        paid.validate_allocation(paid.NODE, placement=placement,
            approval=approval, approval_path=approval_path)


def test_allocation_without_placement_preserves_original_node_requirement(monkeypatch):
    env = {"SLURM_JOB_ID": paid.JOB_ID, "SLURM_JOB_PARTITION": "gnall",
        "SLURM_CPUS_PER_TASK": "8", "SLURM_NTASKS": "1",
        "SLURM_GPUS_ON_NODE": "1", "SLURM_MEM_PER_NODE": "65536",
        "SLURM_JOB_NUM_NODES": "1", "SLURMD_NODENAME": paid.NODE,
        "CUDA_VISIBLE_DEVICES": "3"}
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert paid.validate_allocation(paid.NODE) == (paid.NODE, "3")


def test_phase_budget_contains_backend_reserve_and_cleanup_within_three_hours():
    caps = paid.phase_caps()
    assert sum(caps.values()) == paid.WALL_SECONDS
    assert paid.PROBE_BACKEND_CAP + paid.POST_COMPILE_RESERVE <= caps["probe"]
    assert paid.PROBE_BACKEND_CAP == 9000
    assert paid.POST_COMPILE_RESERVE == 600


def test_probe_stage_allows_one_compile_or_saved_only_but_never_retry_after_attempt(tmp_path):
    binary = tmp_path / "batch.executable.pkl"
    attempt = binary.with_suffix(binary.suffix + ".compile-attempt.json")
    sidecar = binary.with_suffix(binary.suffix + ".identity.json")
    assert paid.select_probe_stage(binary, attempt, sidecar) == "compile-and-compare"
    attempt.write_text("{}")
    with pytest.raises(ValueError, match="recompilation forbidden"):
        paid.select_probe_stage(binary, attempt, sidecar)
    binary.write_bytes(b"saved")
    sidecar.write_text("{}")
    assert paid.select_probe_stage(binary, attempt, sidecar) == "saved-compare"


def test_saved_sidecar_without_executable_fails_closed(tmp_path):
    binary = tmp_path / "batch.executable.pkl"
    sidecar = binary.with_suffix(binary.suffix + ".identity.json")
    sidecar.write_text("{}")
    with pytest.raises(ValueError, match="without the saved executable"):
        paid.select_probe_stage(binary, binary.with_suffix(binary.suffix + ".compile-attempt.json"), sidecar)


def test_probe_summary_preserves_strict_numerical_failure_and_saved_model(tmp_path):
    executable = tmp_path / "b4.executable.pkl"
    sidecar = executable.with_suffix(executable.suffix + ".identity.json")
    executable.write_bytes(b"saved")
    sidecar.write_text("{}")
    data = {}
    outcome = paid.summarize_probe({"status": "numerical_failed", "model_compiles": {"B1": 0, "B4": 1},
        "pairs": [{"pair": 1}, {"pair": 2}, {"pair": 3}], "numerical_failures": [{"pair": 1}]},
        data, executable=executable, sidecar=sidecar)
    assert outcome == "numerical_failed"
    assert data["probe_status"] == "numerical_failed"
    assert data["probe_pairs_completed"] == 3
    assert data["probe_pairs_complete"] is True
    assert data["probe_model_saved"] is True


def test_probe_summary_classifies_inner_timeout_as_incomplete(tmp_path):
    data = {}
    outcome = paid.summarize_probe({"status": "incomplete_timeout", "pairs": [],
        "model_compiles": {"B1": 0, "B4": 1}}, data,
        executable=tmp_path / "missing.pkl", sidecar=tmp_path / "missing.json")
    assert outcome == "incomplete"
    assert data["probe_status"] == "incomplete_timeout"
    assert data["probe_pairs_completed"] == 0
    assert data["probe_pairs_complete"] is False
    assert data["probe_model_saved"] is False


def test_external_paid_helper_resource_error_keeps_resource_classification():
    class HelperResourceError(RuntimeError):
        pass

    error = HelperResourceError("GPU busy")
    assert paid.is_resource_error(error, SimpleNamespace(PaidResourceError=HelperResourceError))
    assert not paid.is_resource_error(error)


def test_free_timeout_evidence_is_read_only_and_bound(tmp_path):
    report = tmp_path / "free-report.json"
    ledger = tmp_path / "free-ledger.json"
    attempt = tmp_path / "free-attempt.json"
    report.write_text(json.dumps({"status": "incomplete_timeout", "deadline_phase": "backend_B4",
        "model_compiles": {"B1": 0, "B4": 1}, "B4_cold": None}))
    ledger.write_text(json.dumps({"limit_gpu_seconds": 7200,
        "independent_funding_scope": "latent_B4_T7_20261008", "sessions": [{"status": "finished"}]}))
    attempt.write_text(json.dumps({"full_B4_compile_attempts": 1,
        "mathematical_identity": {"schema_version": "identity"}}))
    original = [path.read_bytes() for path in (report, ledger, attempt)]
    evidence = paid.verify_free_failure(report, ledger, attempt)
    assert evidence["free_ledger_limit_preserved"] == 7200
    assert [path.read_bytes() for path in (report, ledger, attempt)] == original


def test_candidate_invocation_uses_new_paid_graph_path_and_existing_B1(tmp_path):
    args = object()
    command = paid.candidate_command(args, "compile-and-compare", tmp_path / "paid.json", 10200)
    assert command[1:3] == ["--stage", "compile-and-compare"]
    assert command[command.index("--candidate-executable") + 1] == str(paid.B1_EXECUTABLE)
    assert command[command.index("--batch-executable") + 1] == str(paid.BATCH_EXECUTABLE)
    assert command[command.index("--backend-seconds") + 1] == "9000"
    assert command[command.index("--post-compile-reserve-seconds") + 1] == "600"
    assert command[command.index("--pairs") + 1] == "3"


def test_dry_run_prints_complete_phase_argv_without_running_gpu_work(tmp_path, capsys):
    args = SimpleNamespace(approval=tmp_path / "missing-approval.json")
    assert paid.run(SimpleNamespace(dry_run=True, approval=args.approval,
                                    placement=tmp_path / "missing-placement.json")) == 0
    output = capsys.readouterr().out
    lines = output.splitlines()
    preflight_line = next(line for line in lines if line.startswith("Runtime-preflight Popen argv: "))
    probe_line = next(line for line in lines if line.startswith("Probe Popen argv: "))
    preflight_argv = shlex.split(preflight_line.removeprefix("Runtime-preflight Popen argv: "))
    probe_argv = shlex.split(probe_line.removeprefix("Probe Popen argv: "))
    assert preflight_argv[:8] == ["bash", str(paid.OVERLAY / "shared_latent_T7_batch_paid_memory.sh"),
        "0", str(paid.DIAG / (paid.PREFIX + ".runtime-preflight.memory.csv")), "--",
        "taskset", "-c", "0-7"]
    assert "env" in preflight_argv and "singularity" in preflight_argv
    assert probe_argv[:8] == ["bash", str(paid.OVERLAY / "shared_latent_T7_batch_paid_memory.sh"),
        "0", str(paid.DIAG / (paid.PREFIX + ".probe.memory.csv")), "--",
        "taskset", "-c", "0-7"]
    assert str(paid.OVERLAY / "probe_shared_latent_T7_batch.py") in probe_argv
    assert probe_argv[probe_argv.index("--pairs") + 1] == "3"
    assert probe_argv[probe_argv.index("--wall-seconds") + 1] == str(paid.PROBE_CAP)
    assert probe_argv[probe_argv.index("--backend-seconds") + 1] == str(paid.PROBE_BACKEND_CAP)
    assert probe_argv[probe_argv.index("--post-compile-reserve-seconds") + 1] == str(paid.POST_COMPILE_RESERVE)


def test_run_phase_passes_full_container_command_to_popen(monkeypatch, tmp_path):
    overlay = tmp_path / "overlay"
    diag = tmp_path / "diag"
    code = tmp_path / "code"
    report = diag / "probe.json"
    memory = diag / "unit-paid.probe.memory.csv"
    ledger = diag / "ledger.json"
    monkeypatch.setattr(paid, "OVERLAY", overlay)
    monkeypatch.setattr(paid, "DIAG", diag)
    monkeypatch.setattr(paid, "CODE", code)
    monkeypatch.setattr(paid, "PROBE_REPORT", report)
    monkeypatch.setattr(paid, "LEDGER", ledger)
    monkeypatch.setattr(paid, "PREFIX", "unit-paid")
    monkeypatch.setattr(paid.os, "getloadavg", lambda: (0.1, 0.1, 0.1), raising=False)
    monkeypatch.setattr(paid.subprocess, "check_output", lambda *args, **kwargs: "gpu snapshot")
    captured = {}

    class FakeProcess:
        pid = 123

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(paid.subprocess, "Popen", fake_popen)
    command = paid.container_base("0", "0-7", report, dry_run=True)
    command += paid.candidate_command(None, "compile-and-compare", report, 9900)
    data = {"phases": [], "wall_seconds": paid.WALL_SECONDS}
    paid.run_phase(command, stage="probe", cap=9900, gpu="0", affinity="0-7",
        ledger=ledger, data=data, paid_helpers=SimpleNamespace(shutdown_phase=lambda process: None),
        started=time.monotonic(), termination={"signal": None})
    argv = captured["argv"]
    assert argv == paid.phase_command(command, gpu="0", memory_path=memory)
    assert argv[:8] == ["bash", str(overlay / "shared_latent_T7_batch_paid_memory.sh"),
        "0", str(memory), "--", "taskset", "-c", "0-7"]
    assert argv[8:13] == ["env", "-u", "LD_LIBRARY_PATH",
        "SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs",
        "SINGULARITYENV_CUDA_VISIBLE_DEVICES=0"]
    assert str(overlay / "probe_shared_latent_T7_batch.py") in argv
    assert argv[argv.index("--pairs") + 1] == "3"
    assert argv[argv.index("--wall-seconds") + 1] == "9900"
    assert argv[argv.index("--backend-seconds") + 1] == "9000"
    assert argv[argv.index("--post-compile-reserve-seconds") + 1] == "600"
    assert captured["kwargs"]["cwd"] == code
