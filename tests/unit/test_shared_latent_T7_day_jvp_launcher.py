"""Fail-closed preparation, canonical paths, exit status and raw preservation."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import numpy as np
from scripts.hpc import probe_shared_latent_T7_day_jvp as probe
from scripts.hpc import run_shared_latent_T7_day_jvp as launcher


def arguments(tmp_path):
    command = launcher.child_command(launcher.parser().parse_args(["--stage", "execute", "--plan-only"]))
    args = probe.parser().parse_args(command[3:])
    args.approval_report = tmp_path/"approval.json"
    proposal = probe.proposal_facts(probe.DIAG/(probe.PREFIX+".result.json"))
    proposal.update(approved=False, approval_origin=None, approval_note="")
    args.approval_report.write_text(json.dumps(proposal))
    return args, proposal


def test_child_command_resolves_declared_checkout_and_final_caps(tmp_path):
    command = launcher.child_command(launcher.parser().parse_args(["--stage", "execute", "--dry-run"]))
    args = probe.parser().parse_args(command[3:])
    probe.validate_caps(args)
    assert Path(command[2]) == launcher.OVERLAY/"probe_shared_latent_T7_day_jvp.py"
    assert args.wall_seconds == 4800 and args.gpu_seconds_cap == 5400
    assert args.backend_seconds == 1800 and args.route_seconds == 2100
    assert args.plan_only and args.route_mode == "both" and args.b4_jvp_executable is None
    assert Path(launcher.preflight_command()[2]) == launcher.CODE/"scripts/hpc/check_daily_training_runtime.py"


@pytest.mark.parametrize("field,value", [("approved", 1), ("funding_scope", "old_scope"), ("limit_gpu_seconds", 4800), ("gpu_count", True)])
def test_preparation_rejects_invalid_approval_or_scope(tmp_path, field, value):
    args, proposal = arguments(tmp_path)
    proposal[field] = value
    args.approval_report.write_text(json.dumps(proposal))
    with pytest.raises((ValueError, PermissionError)):
        probe.validate_proposal(args, execution=False)


def test_false_template_cannot_execute_and_true_cannot_prepare(tmp_path):
    args, proposal = arguments(tmp_path)
    with pytest.raises(PermissionError):
        probe.validate_proposal(args, execution=True)
    proposal.update(approved=True, approval_origin="explicit_human_user", approval_note="test only")
    args.approval_report.write_text(json.dumps(proposal))
    with pytest.raises(ValueError, match="approved=false"):
        probe.validate_proposal(args, execution=False)


def test_child_plan_only_never_enters_assets_or_model(tmp_path, monkeypatch):
    args, _ = arguments(tmp_path)
    monkeypatch.setattr(probe, "_assets", lambda *a: pytest.fail("must not read assets"))
    result = probe.run(args)
    assert result["status"] == "plan_only" and not result["GPU_executed"]
    assert result["planned_JVP_calls"] == 24 and result["saved_executable_load_count"] == 0


@pytest.mark.parametrize("status", [1, 3, 124])
def test_launcher_returns_actual_child_exit(tmp_path, monkeypatch, status):
    monkeypatch.setattr(launcher, "CODE", tmp_path)
    monkeypatch.setattr(launcher, "OVERLAY", tmp_path)
    monkeypatch.setattr(launcher.os, "chdir", lambda path: None)
    calls = []
    def child(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=status)
    monkeypatch.setattr(launcher.subprocess, "run", child)
    assert launcher.main(["--stage", "execute", "--plan-only"]) == status
    assert len(calls) == 1 and "--plan-only" in calls[0]


def test_probe_numerical_failure_is_exit3_after_preserved_result(tmp_path, monkeypatch):
    args, _ = arguments(tmp_path)
    calls = []
    owner = SimpleNamespace(atomic_pickle=lambda path, value: calls.append((path, value)))
    failed = {"primal": "complete array", "tangent": "failed full array", "aux": "all masks"}
    probe.preserve_raw(owner, tmp_path/"raw.pkl", failed, route="B1", cycle=1, coordinate=434)
    assert calls[0][1]["result"] is failed
    monkeypatch.setattr(probe, "run", lambda args: {"status": "numerical_failed"})
    command = launcher.child_command(launcher.parser().parse_args(["--stage", "execute", "--plan-only"]))
    assert probe.main(command[3:]) == 3


def test_invalid_cumulative_caps_rejected_before_readiness(tmp_path):
    args, _ = arguments(tmp_path)
    args.backend_seconds = 900
    with pytest.raises(ValueError, match="caps"):
        probe.run(args)


def test_bash_has_no_optional_empty_array_expansion_and_preserves_budget_exit():
    source = Path(launcher.__file__).with_suffix(".sh").read_text()
    assert "RUNTIME_ARGS[@]" not in source and "REPAIR_ARGS[@]" not in source
    assert "m.main()" in source and 'exec "${WRAPPED[@]}"' in source
    assert "--seconds 5400 --cards 1" in source


@pytest.mark.parametrize("value,passed", [(0, True), (24576, True), (24577, False)])
def test_sampled_nvml_memory_gate_and_boundary(tmp_path, value, passed):
    path = tmp_path/"memory.csv"
    path.write_text(f"epoch_seconds,memory_used_mib,utilization_percent\n1,{value},0\n")
    report = probe.sampled_nvml_memory(path)
    assert report["passed"] is passed and report["sample_count"] == 1
    json.dumps(report)


@pytest.mark.parametrize("body", ["", "1,nan,0\n", "nan,1,0\n", "1,-1,0\n"])
def test_invalid_nvml_samples_fail_closed(tmp_path, body):
    path = tmp_path/"memory.csv"
    path.write_text("epoch_seconds,memory_used_mib,utilization_percent\n"+body)
    with pytest.raises(MemoryError):
        probe.sampled_nvml_memory(path)
    with pytest.raises(FileNotFoundError):
        probe.sampled_nvml_memory(tmp_path/"absent.csv")


def test_actual_first_day_cpu_readiness_preserves_every_sample3_input_leaf():
    import jax
    cases = tuple((np.full(3854, i, dtype=np.float64), {"discrete": np.array(i, dtype=np.int32)},
        {"forcing": np.full((7, 2), i, dtype=np.float64), "mask": np.full((7, 2), bool(i%2))}, {}, {})
        for i in range(4))
    stacked = jax.tree_util.tree_map(lambda *values: np.stack(values), *cases)
    report = probe.first_day_readiness(jax, {"cases": cases, "stacked": stacked})
    assert report["sample3_exact"] and report["full_state_width"] == 3854
    assert report["argument_leaf_count"] == 4
    json.dumps(report)
