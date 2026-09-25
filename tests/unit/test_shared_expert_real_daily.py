"""Focused preparation and comparison checks for the real daily probe."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from scripts.hpc import probe_shared_expert_real_daily as probe


def test_cpu_preparation_uses_compact_first_four_rows_and_pickles_only_daily_context(tmp_path, monkeypatch):
    from research.daily_coarse_graining.daily_operator import (
        shared_persistent_outputs,
        shared_process_head,
        shared_recurrent_training,
        shared_training,
    )
    from scripts.hpc import run_shared_learning, shared_learning_plan, shared_transformer_runtime

    checkpoint = tmp_path / "update0.pkl"
    checkpoint.touch()
    executable = tmp_path / "daily.executable.pkl"
    executable.touch()
    method = {
        "training_plan": {"training_artifact_ids": [17, 19]},
        "daily_protocol": {"architecture": {"thermal_transform": "train_centered"}},
    }
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
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "technical_update_report": str(reference_path),
                "executables": {"daily": {"executable": str(executable)}},
            }
        ),
        encoding="utf-8",
    )
    bundle_path = tmp_path / "inputs.pkl"
    first = np.arange(64, dtype=np.float32).reshape(32, 2)
    with bundle_path.open("wb") as stream:
        pickle.dump(
            {
                "schema_version": "shared_serialized_dual_inputs_v1",
                "source_git_head": "abc",
                "method": method,
                "artifact_ids": (17, 19),
                "initial_checkpoint": str(checkpoint),
                "payloads": ({"daily": {"rows": first}}, {"daily": {"rows": first + 1}}),
            },
            stream,
        )
    runtime = {
        "loss_scales": "loss",
        "base_statistics": SimpleNamespace(arrays={"state_delta": "delta"}),
        "process_statistics": "process",
        "jax": jax,
        "unused_callback": lambda: None,
    }
    parent = SimpleNamespace(learned_schema="schema", contract="contract")
    template = SimpleNamespace(name="retained")
    calls = []
    monkeypatch.setattr(probe.subprocess, "check_output", lambda *args, **kwargs: "abc\n")
    monkeypatch.setattr(
        shared_learning_plan,
        "load_learning_configuration",
        lambda *args: {
            "admission": {"scales": {"science": 1}},
        },
    )
    monkeypatch.setattr(run_shared_learning, "_method_identity", lambda *args, **kwargs: method)
    monkeypatch.setattr(run_shared_learning, "shared_config", lambda architecture: "config")
    monkeypatch.setattr(
        shared_transformer_runtime,
        "load_shared_runtime",
        lambda args, ids: calls.append((args.cpu_forward_only, ids)) or (runtime, parent, template, (), {}),
    )
    monkeypatch.setattr(shared_persistent_outputs, "build_shared_persistent_coordinates", lambda *args: "coords")
    monkeypatch.setattr(
        shared_process_head,
        "build_shared_process_head",
        lambda *args, **kwargs: SimpleNamespace(
            config=kwargs["config"], source_bare_cimean=kwargs["source_bare_cimean"]
        ),
    )
    monkeypatch.setattr(shared_training, "derive_shared_state_change_scales", lambda runtime, head: "state")
    monkeypatch.setattr(shared_recurrent_training, "make_shared_joint_daily_objective", lambda *args: "objective")
    args = SimpleNamespace(
        source_profile=profile_path,
        input_bundle=bundle_path,
        prepared_context=tmp_path / "context.pkl",
        protocol=tmp_path / "protocol.json",
        joint_admission_report=tmp_path / "joint.json",
        prepared_manifest=tmp_path / "prepared.json",
        validation_manifest=tmp_path / "validation.json",
        cpu_forward_only=False,
    )
    # This fixture uses a synthetic source head; no real runtime or Teacher file is opened.
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    result = probe.prepare(args, lambda *args, **kwargs: None)
    assert result["status"] == "prepared"
    assert calls == [(True, (17,))]
    with args.prepared_context.open("rb") as stream:
        saved = pickle.load(stream)
    np.testing.assert_array_equal(saved["daily4"]["rows"], first[:4])
    assert set(saved["runtime"]) == {"loss_scales", "base_statistics", "process_statistics"}
    assert saved["head"].source_bare_cimean is True
    assert saved["daily_executable"] == str(executable.resolve())


def test_complete_gradient_comparison_and_hard_counts():
    reference = {"a": np.array([1.0, 0.0], np.float32), "b": (np.array([2.0], np.float32),)}
    candidate = {"a": np.array([1.0 + 1e-7, 2e-6], np.float32), "b": (np.array([2.0], np.float32),)}
    result = probe.compare_trees(jax, candidate, reference)
    assert result["coordinates"] == 3
    assert result["failed"] == 1
    assert not result["passed"]
    assert result["relative_l2"] > 0
    infinite = probe.compare_trees(jax, np.array([np.inf], np.float32), np.array([np.inf], np.float32))
    assert not infinite["passed"] and infinite["failed"] == 1
    assert infinite["relative_l2"] is None and infinite["max_abs"] is None
    assert probe.hard_counts({"hard": {"state_nonfinite": np.int32(0)}}) == {"state_nonfinite": 0}
    with pytest.raises(ValueError, match="hard-count"):
        probe.hard_counts({})


def test_explicit_staged_adapter_loader_uses_sibling_files():
    directory = Path(__file__).resolve().parents[2] / "research/daily_coarse_graining/daily_operator"
    adapter = probe.load_staged_adapter(directory / "shared_decoder_expert_execution.py")
    assert adapter.__name__ == "shared_transformer_raw_outputs_expert128"
    with pytest.raises(FileNotFoundError, match="staged diagnostic"):
        probe.load_staged_adapter(directory / "missing" / "shared_decoder_expert_execution.py")


def test_backbone_loader_and_parser_options(tmp_path, monkeypatch):
    package = "research.daily_coarse_graining.daily_operator"
    for stem in ("shared_backbone_projection_reuse", "shared_backbone_execution"):
        name = f"{package}.{stem}"
        monkeypatch.setitem(__import__("sys").modules, name, __import__("sys").modules.get(name))
    helper_path = tmp_path / "shared_backbone_projection_reuse.py"
    helper_path.write_text("VALUE = 'staged-helper'\n", encoding="utf-8")
    module_path = tmp_path / "shared_backbone_execution.py"
    module_path.write_text(
        "from .shared_backbone_projection_reuse import VALUE\n"
        "def assemble_shared_transformer_context_with_source_reuse(*args, **kwargs):\n"
        "    return VALUE\n",
        encoding="utf-8",
    )
    assembler = probe.load_backbone_assembler(module_path)
    assert assembler() == "staged-helper"
    with pytest.raises(FileNotFoundError, match="staged backbone"):
        probe.load_backbone_assembler(tmp_path / "absent.py")
    options = probe.parser()._option_string_actions
    assert {"--backbone-module-file", "--reference-result", "--comparison-result"} <= set(options)


def test_result_artifact_requires_matching_identity(tmp_path):
    artifact = tmp_path / "candidate.pkl"
    with artifact.open("wb") as stream:
        pickle.dump({"identity": {"checkpoint": "update0"}, "result": "result"}, stream)
    loaded = probe.load_result_artifact(artifact, {"checkpoint": "update0"}, role="expert128 comparison")
    assert loaded["result"] == "result"
    with pytest.raises(ValueError, match="identity differs"):
        probe.load_result_artifact(artifact, {"checkpoint": "other"}, role="expert128 comparison")
    with pytest.raises(ValueError, match="overwrite"):
        probe.check_artifact_output_collisions((artifact,), (artifact,))


def test_expert128_comparison_accepts_legacy_tag_only_with_matching_identity(tmp_path):
    artifact = tmp_path / "legacy-expert128.pkl"
    value = {"identity": {"artifact_id": 17}, "result": ((0.0, {"hard": {"count": 0}}), {})}
    with artifact.open("wb") as stream:
        pickle.dump(value, stream)
    loaded = probe.load_expert128_comparison(artifact, {"artifact_id": 17})
    assert "variant" not in loaded
    with pytest.raises(ValueError, match="identity differs"):
        probe.load_expert128_comparison(artifact, {"artifact_id": 19})
    value["variant"] = "another-candidate"
    with artifact.open("wb") as stream:
        pickle.dump(value, stream)
    with pytest.raises(ValueError, match="not the expert128"):
        probe.load_expert128_comparison(artifact, {"artifact_id": 17})


@pytest.mark.parametrize("failure", ["lower", "compile"])
def test_candidate_process_head_patches_restore_after_failure(failure):
    original_raw = object()
    original_assembler = object()
    adapter = object()
    replacement_assembler = object()
    process_head = SimpleNamespace(
        shared_transformer_raw_outputs=original_raw,
        assemble_shared_transformer_context=original_assembler,
    )

    class Lowered:
        def compile(self):
            assert process_head.shared_transformer_raw_outputs is adapter
            assert process_head.assemble_shared_transformer_context is replacement_assembler
            if failure == "compile":
                raise RuntimeError("compile failed")
            return "executable"

    class Backward:
        def lower(self, parameters, daily4):
            assert process_head.shared_transformer_raw_outputs is adapter
            assert process_head.assemble_shared_transformer_context is replacement_assembler
            if failure == "lower":
                raise RuntimeError("lower failed")
            return Lowered()

    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        probe.lower_and_compile_candidate(
            Backward(),
            "parameters",
            "daily4",
            lambda *args, **kwargs: None,
            probe.candidate_process_head_patches(process_head, adapter, replacement_assembler),
        )
    assert process_head.shared_transformer_raw_outputs is original_raw
    assert process_head.assemble_shared_transformer_context is original_assembler
