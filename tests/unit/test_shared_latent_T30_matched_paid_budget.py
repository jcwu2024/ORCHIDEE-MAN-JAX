import json
from pathlib import Path

import pytest

from scripts.hpc import shared_latent_T30_matched_paid_budget as paid


def test_false_approval_template_is_not_executable():
    template = json.loads(
        (Path(__file__).resolve().parents[2] / "outputs/diagnostics/shared-latent-reader-20261004/T30-paid-matched-approval.unapproved-template.json")
        .read_text(encoding="utf-8")
    )
    assert template["approved"] is False
    with pytest.raises(ValueError, match="paid approval"):
        paid.validate_approval(template, {"binary_sha256": {}, "sidecar_sha256": {}})


def test_uuid_paths_are_fixed_and_saved_only():
    with pytest.raises(ValueError, match="UUID0001/UUID0002"):
        paid.validate_saved_assets(original=Path("wrong-original.executable.pkl"))
    with pytest.raises(FileNotFoundError, match="saved-only graph missing"):
        paid.expected_compile_identity(Path("absent.executable.pkl"), "original", {})


def test_sidecar_and_compile_report_are_mandatory(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    context = tmp_path / "context.pkl"
    input_bundle = tmp_path / "inputs.npz"
    binary = tmp_path / "original.executable.pkl"
    manifest.write_text("{}", encoding="utf-8")
    context.write_bytes(b"context")
    input_bundle.write_bytes(b"inputs")
    binary.write_bytes(b"saved graph")
    monkeypatch.setattr(paid, "MANIFEST", manifest)
    monkeypatch.setattr(paid, "PREPARED_CONTEXT", context)
    identity = {
        "architecture": "original",
        "source_git_head": paid.HEAD,
        "artifact_id": 742,
        "horizon": 30,
        "native_AD_argnums": [0, 1],
        "precision": "FP32_neural_FP64_physics",
        "turnover_base_sha256": paid.BASE_SHA,
        "turnover_sha256": paid.TURNOVER_SHA,
        "training_updates": 0,
        "jax_version": "0.4.38",
        "manifest_sha256": paid.sha256_file(manifest),
        "input_bundle_sha256": paid.sha256_file(input_bundle),
        "prepared_context_sha256": paid.sha256_file(context),
        "compiler_options": {
            "xla_gpu_enable_llvm_module_compilation_parallelism": True,
            "xla_gpu_force_compilation_parallelism": 4,
        },
        "parameter_count": 64884481,
        "parameter_leaves": 300,
        "parameter_provenance": {"completed_updates": 0, "checkpoint": "checkpoint"},
    }
    with pytest.raises(FileNotFoundError, match="identity sidecar missing"):
        paid.expected_compile_identity(binary, "original", {"input_bundle": str(input_bundle), "original_checkpoint": "checkpoint"})
    paid.sidecar_path(binary).write_text(json.dumps({"component": "T30", "identity": identity}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="compile report missing"):
        paid.expected_compile_identity(binary, "original", {"input_bundle": str(input_bundle), "original_checkpoint": "checkpoint"})


def test_compile_runtime_receipts_are_reused_and_must_match(tmp_path):
    original = tmp_path / "original-preflight.json"
    candidate = tmp_path / "candidate-preflight.json"
    receipt = {
        "status": "passed",
        "passed": True,
        "actual_backend": "gpu",
        "requested_backend": "gpu",
        "jax_version": "0.4.38",
        "devices": ["cuda:0"],
    }
    original.write_text(json.dumps(receipt), encoding="utf-8")
    candidate.write_text(json.dumps(receipt), encoding="utf-8")
    assert paid.validate_runtime_preflights(original, candidate, "0.4.38") == receipt
    receipt["devices"] = ["cuda:1"]
    candidate.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="same visible device"):
        paid.validate_runtime_preflights(original, candidate, "0.4.38")


def test_cleanup_signals_residual_owned_group_after_leader_exit(monkeypatch):
    group_members = iter([{41}, set()])
    signals = []
    monkeypatch.setattr(paid, "current_process_group_pids", lambda pgid: next(group_members))
    monkeypatch.setattr(paid.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)), raising=False)
    monkeypatch.setattr(paid.time, "sleep", lambda _: None)
    cleanup = paid.cleanup_process_group(41, term_timeout=1)
    assert cleanup["release_confirmed"]
    assert cleanup["remaining_pids"] == []
    assert signals == [(41, paid.signal.SIGTERM)]


def test_dry_run_is_matched_only_and_cost_is_separate():
    command = paid.launch_command("0", "0-7", Path("matched.json"))
    assert command[command.index("--stage") + 1] == "matched"
    assert command[command.index("--horizon") + 1] == "30"
    assert command[command.index("--pairs") + 1] == "3"
    assert command[command.index("--original-executable") + 1] == str(paid.ORIGINAL)
    assert command[command.index("--candidate-executable") + 1] == str(paid.CANDIDATE)
    assert command[command.index("-B") + 1] == str(paid.MATCHED_PROBE)
    local_probe = Path(__file__).resolve().parents[2] / "scripts/hpc/probe_shared_latent_reader_T7_pair.py"
    assert paid.sha256_file(local_probe) == paid.MATCHED_PROBE_SHA256
    assert "--training-updates" not in command
    assert paid.worst_case_cost_cny() == pytest.approx(0.92)
    assert "--nodelist=node-01" in paid.submit_command("node-01")


def test_sampled_gpu_pid_classification_is_stable_after_exit():
    snapshot = {"compute_processes": [{"pid": 12}, {"pid": 34}]}
    paid.classify_gpu_processes(snapshot, {12})
    sampled_classification = [item["classification"] for item in snapshot["compute_processes"]]
    assert [item["classification"] for item in snapshot["compute_processes"]] == [
        "owned_process_group",
        "external",
    ]
    assert [item["classification"] for item in snapshot["compute_processes"]] == sampled_classification
