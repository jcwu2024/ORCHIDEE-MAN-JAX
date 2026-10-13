"""Real child exec regressions, without JAX, model data or a Slurm allocation."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.dev import global_cpu_child_environment as boundary
from scripts.hpc import global_cpu48_b64_coordinator as coordinator

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "scripts/dev/benchmark_global_server_cpu.py"
BOUNDARY = ENTRY.with_name("global_cpu_child_environment.py")
CONFLICTS = {"OMP_NUM_THREADS": "1", "OMP_THREAD_LIMIT": "1", "MKL_NUM_THREADS": "64",
             "OPENBLAS_NUM_THREADS": "64", "NUMEXPR_NUM_THREADS": "64",
             "XLA_FLAGS": "--xla_force_host_platform_device_count=48 --xla_cpu_multi_thread_eigen=false",
             "JAX_PLATFORMS": "cuda", "JAX_ENABLE_X64": "false", "JAX_DISABLE_JIT": "true",
             "JAX_NUM_CPU_DEVICES": "48", "TF_NUM_INTRAOP_THREADS": "1"}


def dirty_environment():
    return {**os.environ, **CONFLICTS, "PYTHONDONTWRITEBYTECODE": "1"}


def dry_runner(tmp_path):
    return [sys.executable, str(ENTRY), "--mode", "batch", "--cpu-budget", "48",
            "--sizes", "64", "--output-dir", str(tmp_path), "--dry-run",
            "--prepared-input", str(tmp_path / "prepared.pkl"),
            "--gpu-reference-dir", str(tmp_path / "gpu"),
            "--cpu-reference-dir", str(tmp_path / "scalar"), "--require-child-environment"]


def test_post_transport_exec_restores_actual_environment(tmp_path):
    command = boundary.wrapped_command(sys.executable, BOUNDARY, 48, dry_runner(tmp_path))
    result = subprocess.run(command, env=dirty_environment(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)["actual_child_prejax_receipt"]
    assert actual["actual_environment"] == {key: boundary.policy(48).get(key) for key in boundary.WHITELIST}
    assert actual["unexpected_policy_keys"] == []
    assert actual["inherited_transport_policy"]["environment"]["OMP_NUM_THREADS"] == "1"
    assert "JAX_DISABLE_JIT" in actual["inherited_transport_policy"]["unexpected_policy_keys"]
    assert actual["JAX_imported"] is actual["model_execution"] is False
    assert json.loads(result.stderr.splitlines()[0])["status"] == "exec_boundary_ready"


def transport(tmp_path):
    """Executing transport injects stale ALL values after Popen, then execs child."""
    shim = tmp_path / "executing_transport.py"
    shim.write_text("import os,sys,subprocess\n"
                   f"os.environ.update({CONFLICTS!r})\n"
                   "args=sys.argv[1:]\n"
                   "start=args.index('/usr/bin/time')+2\n"
                   "command=args[start:]+['--dry-run']\n"
                   "raise SystemExit(subprocess.call(command,env=dict(os.environ)))\n", encoding="utf-8")
    return shim


def test_production_run_process_executes_child_after_conflicting_transport(tmp_path, monkeypatch):
    from test_global_cpu48_b64_preparation import args

    for key, value in CONFLICTS.items():
        monkeypatch.setenv(key, value)
    command, _ = coordinator.step_command(args(tmp_path), "batch")
    command[:1] = [sys.executable, str(transport(tmp_path))]
    result = coordinator.run_process(command, tmp_path / "actual.log", 20)
    assert result["returncode"] == 0
    text = (tmp_path / "actual.log").read_text()
    assert '"OMP_NUM_THREADS": "48"' in text
    assert '"unexpected_policy_keys": []' in text
    assert '"JAX_imported": false' in text
    assert '"resource_admission_passed"' not in text  # CLI only, no physical admission.


def test_coordinate_uses_real_dispatch_and_child_boundary(tmp_path, monkeypatch):
    from test_global_cpu48_b64_preparation import args

    run_args = args(tmp_path, dry_run=True)
    # Source identity and allocation are bounded unit mocks, not a target allocation.
    run_args.dry_run = False
    project = tmp_path / "project"
    code = project / "runtime/worktrees/unit"
    code.mkdir(parents=True)
    run_args.project_root, run_args.code_dir = project, code
    run_args.output_dir = project / "runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/unit"
    monkeypatch.chdir(code)
    monkeypatch.setattr(coordinator.subprocess, "check_output", lambda command, **kwargs:
                        run_args.expected_code_commit if command[-1] == "HEAD" else "")
    monkeypatch.setattr(coordinator, "validate_allocation", lambda: {"unit_mock": True})
    original = coordinator.step_command
    shim = transport(tmp_path)
    def execute_args(args, mode):
        command, environment = original(args, mode)
        command[:1] = [sys.executable, str(shim)]
        return command, environment
    monkeypatch.setattr(coordinator, "step_command", execute_args)
    result = coordinator.coordinate(run_args, snapshot=lambda: {"scope": "unit_mock"})
    assert result["status"] == "completed_exploratory_measurement"
    for mode, budget in (("scalar", 1), ("batch", 48)):
        log = (run_args.output_dir / f"{mode}.log").read_text()
        assert f'"OMP_NUM_THREADS": "{budget}"' in log
        assert '"unexpected_policy_keys": []' in log
        assert result["steps"][mode]["returncode"] == 0
    assert result["scientific_acceptance"] is False


def test_wrong_actual_env_logs_failure_before_jax_or_slurm(tmp_path):
    command = [token for token in dry_runner(tmp_path) if token != "--dry-run"]
    result = subprocess.run(command, env=dirty_environment(), capture_output=True, text=True)
    assert result.returncode != 0
    receipt = json.loads(result.stderr.splitlines()[0])
    assert receipt["status"] == "resource_admission_failed"
    assert receipt["JAX_imported"] is False
    assert "environment mismatch" in receipt["error"]
    assert receipt["actual_environment"]["OMP_NUM_THREADS"] == "1"


def test_correct_env_without_paid_allocation_still_fails_closed(tmp_path):
    command = [token for token in dry_runner(tmp_path) if token != "--dry-run"]
    env = boundary.normalized_environment(48, dirty_environment())
    for key in list(env):
        if key.startswith("SLURM_") or key.startswith("ORCHJAX_SLURM_"):
            env.pop(key)
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    receipt = json.loads(result.stderr.splitlines()[0])
    assert receipt["JAX_imported"] is False
    assert "actual declared paid Slurm" in receipt["error"]
    assert receipt["actual_environment"]["OMP_NUM_THREADS"] == "48"


def test_batch_only_real_dispatch_never_regenerates_scalar_and_bad_seal_stops(tmp_path, monkeypatch):
    from test_global_cpu48_b64_preparation import args

    from scripts.dev import global_cpu_candidate_identity as identity_module
    from scripts.dev import global_server_reference as references

    run_args = args(tmp_path)
    project = tmp_path / "project"
    code = project / "runtime/worktrees/unit"
    code.mkdir(parents=True)
    run_args.project_root, run_args.code_dir = project, code
    run_args.output_dir = project / "runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/unit"
    run_args.execution_route = "batch-only"
    run_args.budget_seconds = 3600
    run_args.cpu_reference_dir = tmp_path / "old-scalar"
    run_args.cpu_reference_admission_record = tmp_path / "admission.json"
    run_args.expected_cpu_reference_admission_sha256 = "a" * 64
    monkeypatch.chdir(code)
    monkeypatch.setattr(coordinator.subprocess, "check_output", lambda command, **kwargs:
                        run_args.expected_code_commit if command[-1] == "HEAD" else "")
    monkeypatch.setattr(coordinator, "validate_allocation", lambda: {"unit_mock": True})
    monkeypatch.setattr(identity_module, "candidate_identity", lambda *args: {"unit_mock": True})
    monkeypatch.setattr(references, "validate_cpu_reference_admission", lambda *args, **kwargs:
                        {"scope": "unit_mock_external_seal"}, raising=False)
    original = coordinator.step_command
    shim = transport(tmp_path)
    def execute_args(args, mode):
        command, environment = original(args, mode)
        command[:1] = [sys.executable, str(shim)]
        return command, environment
    monkeypatch.setattr(coordinator, "step_command", execute_args)
    result = coordinator.coordinate(run_args, snapshot=lambda: {"scope": "unit_mock"})
    assert set(result["steps"]) == set(result["commands"]) == {"batch"}
    assert result["steps"]["batch"]["returncode"] == 0
    assert result["external_scalar_reference_admission"]["scope"] == "unit_mock_external_seal"
    assert not (run_args.output_dir / "scalar").exists()
    assert not (run_args.output_dir / "scalar.log").exists()
    run_args.output_dir = run_args.output_dir.with_name("bad-seal")
    def reject(*args, **kwargs):
        raise ValueError("seal changed")
    monkeypatch.setattr(references, "validate_cpu_reference_admission", reject)
    with __import__("pytest").raises(ValueError, match="seal changed"):
        coordinator.coordinate(run_args, snapshot=lambda: {"scope": "unit_mock"})
    assert not run_args.output_dir.exists()
