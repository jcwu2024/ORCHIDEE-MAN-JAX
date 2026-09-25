"""Focused structural and failure-boundary checks for recurrent admission."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import shared_process_head
from research.daily_coarse_graining.daily_operator.shared_recurrent_training import SharedWindowCase
from scripts.hpc import probe_shared_combined_recurrent as probe


@pytest.fixture
def compact_assets(tmp_path, monkeypatch):
    daily_probe = probe.load_daily_probe(Path(probe.__file__).with_name("probe_shared_expert_real_daily.py"))
    checkpoint = tmp_path / "update0.pkl"
    checkpoint.touch()
    executables = {}
    for name in ("daily", "T7", "T30"):
        path = tmp_path / f"{name}.executable.pkl"
        path.touch()
        executables[name] = {"executable": str(path)}
    method = {"training_plan": {"training_artifact_ids": [17, 19]}}
    reference = {
        "status": "passed",
        "completed_updates": 2,
        "technical_updates": 2,
        "source_git_head": "abc",
        "method": method,
        "checkpoints": [{"update": 0, "retained_milestone": True, "path": str(checkpoint)}],
    }
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    profile = {
        "status": "passed",
        "technical_update_report": str(reference_path),
        "executables": executables,
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    def case(horizon):
        return SharedWindowCase(
            initial_continuous=np.zeros(2, dtype=np.float32),
            initial_discrete={},
            days=(SimpleNamespace(continuous_state=np.zeros((horizon, 2), dtype=np.float32)), None),
            targets={"state": np.zeros((horizon, 2), dtype=np.float32)},
            loss_context=None,
        )

    bundle_path = tmp_path / "inputs.pkl"
    with bundle_path.open("wb") as stream:
        pickle.dump(
            {
                "schema_version": "shared_serialized_dual_inputs_v1",
                "source_git_head": "abc",
                "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                "method": method,
                "artifact_ids": (17, 19),
                "initial_checkpoint": str(checkpoint.resolve()),
                "payloads": ({"cases": {7: case(7), 30: case(30)}}, {"cases": {7: case(7), 30: case(30)}}),
            },
            stream,
        )
    context_path = tmp_path / "context.pkl"
    with context_path.open("wb") as stream:
        pickle.dump(
            {
                "identity": daily_probe.context_identity(reference, checkpoint, 17),
                "daily_executable": str(Path(executables["daily"]["executable"]).resolve()),
            },
            stream,
        )
    monkeypatch.setattr(probe.subprocess, "check_output", lambda *args, **kwargs: "abc\n")
    return SimpleNamespace(
        source_profile=profile_path,
        input_bundle=bundle_path,
        prepared_context=context_path,
        checkpoint=checkpoint,
        executables=executables,
        daily_probe=daily_probe,
    )


@pytest.mark.parametrize("horizon", [7, 30])
def test_inspect_selects_exact_saved_case_and_executable(compact_assets, horizon):
    compact_assets.horizon = horizon
    selected = probe.inspect_assets(compact_assets, compact_assets.daily_probe)
    assert selected["artifact_id"] == 17
    assert selected["executable"] == Path(compact_assets.executables[f"T{horizon}"]["executable"]).resolve()
    assert len(selected["case"]) == 5
    assert selected["case"].days[0].continuous_state.shape[0] == horizon


def test_benchmark_expands_five_case_arguments():
    calls = []
    case = tuple(jnp.asarray(index, dtype=jnp.float32) for index in range(5))

    def executable(*args):
        calls.append(args)
        return jnp.asarray(1.0)

    result, seconds = probe.benchmark_recurrent(executable, "parameters", case, repeats=2)
    assert float(result) == 1.0
    assert len(seconds) == 2 and all(value >= 0 for value in seconds)
    assert len(calls) == 3
    assert all(args[0] == "parameters" and args[1:] == case for args in calls)


def test_saved_reference_guard_and_compile_exception_restore_both_process_globals(tmp_path):
    daily_probe = probe.load_daily_probe(Path(probe.__file__).with_name("probe_shared_expert_real_daily.py"))
    original_raw = shared_process_head.shared_transformer_raw_outputs
    original_assembler = shared_process_head.assemble_shared_transformer_context

    def adapter(*args):
        return None

    def backbone(*args):
        return None

    reference_path = tmp_path / "reference.pkl"
    calls = []

    class Backward:
        def lower(self, *args):
            calls.append(args)
            assert shared_process_head.shared_transformer_raw_outputs is adapter
            assert shared_process_head.assemble_shared_transformer_context is backbone
            return self

        def compile(self):
            assert reference_path.is_file()
            raise RuntimeError("synthetic backend failure")

    def patches():
        return daily_probe.candidate_process_head_patches(shared_process_head, adapter, backbone)

    with pytest.raises(FileNotFoundError, match="saved before candidate"):
        probe.lower_and_compile_recurrent(
            Backward(), "parameters", (1, 2, 3, 4, 5), reference_path, lambda *args, **kwargs: None, patches()
        )
    assert not calls
    reference_path.touch()
    with pytest.raises(RuntimeError, match="synthetic backend failure"):
        probe.lower_and_compile_recurrent(
            Backward(), "parameters", (1, 2, 3, 4, 5), reference_path, lambda *args, **kwargs: None, patches()
        )
    assert calls == [("parameters", 1, 2, 3, 4, 5)]
    assert shared_process_head.shared_transformer_raw_outputs is original_raw
    assert shared_process_head.assemble_shared_transformer_context is original_assembler


def test_candidate_result_is_saved_even_when_strict_parity_fails(tmp_path):
    daily_probe = probe.load_daily_probe(Path(probe.__file__).with_name("probe_shared_expert_real_daily.py"))
    reference = (
        (np.float32(1), {"flux": np.float32(1), "hard": {"physical": np.int32(0)}}),
        {"weight": np.asarray([1, 2], dtype=np.float32)},
    )
    candidate = (
        (np.float32(2), {"flux": np.float32(1), "hard": {"physical": np.int32(0)}}),
        {"weight": np.asarray([1, 3], dtype=np.float32)},
    )
    path = tmp_path / "candidate.pkl"
    calls = []

    def executable(*args):
        calls.append(args)
        if len(calls) > 1:
            assert path.is_file()
        return jax.tree_util.tree_map(jnp.asarray, candidate)

    def preserve(result):
        daily_probe.atomic_pickle(path, {"identity": {"horizon": 7}, "result": jax.device_get(result)})

    probe.benchmark_recurrent(executable, "parameters", (1, 2, 3, 4, 5), 1, on_first_result=preserve)
    comparisons, hard, passed = probe.compare_candidate(daily_probe, jax, candidate, reference)
    assert not passed and not comparisons["loss"]["passed"]
    assert not comparisons["all_parameter_gradients"]["passed"]
    assert hard == {"physical": 0}
    with path.open("rb") as stream:
        saved = pickle.load(stream)
    np.testing.assert_array_equal(saved["result"][1]["weight"], candidate[1]["weight"])


def test_reused_reference_requires_exact_identity(tmp_path):
    path = tmp_path / "reference.pkl"
    with path.open("wb") as stream:
        pickle.dump({"identity": {"horizon": 7}, "result": "saved"}, stream)
    assert probe.load_reference_result(path, {"horizon": 7})[0] == "saved"
    with pytest.raises(ValueError, match="identity differs"):
        probe.load_reference_result(path, {"horizon": 30})


def test_parser_expands_selected_horizon_and_staged_paths():
    required = (
        "dataset-manifest",
        "acceptance-report",
        "base-statistics",
        "process-statistics",
        "fast-statistics",
        "prepared-manifest",
        "joint-admission-report",
        "validation-manifest",
        "output",
        "source-profile",
        "input-bundle",
        "prepared-context",
    )
    argv = [part for name in required for part in (f"--{name}", f"{name}.json")]
    argv += [
        "--horizon",
        "30",
        "--inspect-only",
        "--reference-result",
        "saved-reference.pkl",
        "--candidate-module-file",
        "shared_decoder_expert_execution.py",
        "--backbone-module-file",
        "shared_backbone_execution.py",
    ]
    args = probe.parser().parse_args(argv)
    assert args.horizon == 30 and args.inspect_only
    assert args.reference_result == Path("saved-reference.pkl")
    assert args.candidate_module_file.name == "shared_decoder_expert_execution.py"
    assert args.backbone_module_file.name == "shared_backbone_execution.py"
