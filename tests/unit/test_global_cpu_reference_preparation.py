"""Synthetic coverage for production CPU reference identity preparation."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from scripts.dev import benchmark_global_batch as base
from scripts.dev import benchmark_global_scale as scale
from scripts.dev import benchmark_global_server_cpu as cpu
from scripts.dev.global_batch_stage_store import StageStore
from scripts.dev import global_server_reference as refs

_spec = importlib.util.spec_from_file_location(
    "global_server_benchmark_fixtures", Path(__file__).with_name("test_global_server_benchmark.py"))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)


def _args(tmp_path, *, mode="batch", cpu_budget=48, cpu_reference_dir=None,
          admission_record=None):
    return SimpleNamespace(
        baseline_ref="toy-baseline",
        warm_repeats=5,
        mode=mode,
        cpu_budget=cpu_budget,
        sizes=[64],
        validation_mode="exploratory",
        cpu_reference_role="cpu-internal",
        expected_gpu_reference_sha256="a" * 64,
        gpu_reference_dir=tmp_path / "gpu",
        cpu_reference_dir=cpu_reference_dir,
        cpu_reference_admission_record=admission_record,
        expected_cpu_reference_admission_sha256="b" * 64 if admission_record else None,
        output_dir=tmp_path / "output",
    )


def _real_metadata(args):
    jax.config.update("jax_enable_x64", True)
    return base.metadata(args)


def test_stage_identity_preserves_legacy_schema_and_opts_into_metadata_git_head(monkeypatch, tmp_path):
    lanes, second = _fixtures.toy_prepared(4)
    args = _args(tmp_path, cpu_budget=1)
    environment = _real_metadata(args)
    monkeypatch.setattr(base, "scientific_source_digest", lambda: "d" * 64)

    legacy = base.stage_identity(args, lanes, second, environment)
    server = base.stage_identity(args, lanes, second, environment, include_git_head=True)

    assert "git_head" not in legacy["runtime"]
    assert server["runtime"]["git_head"] == environment["git_head"]
    assert len(server["runtime"]["git_head"]) == 40


@pytest.mark.parametrize("mode,cpu_budget", [("batch", 48), ("pool", 4)])
def test_prepare_references_uses_git_head_before_any_runner_or_compile(monkeypatch, tmp_path, mode, cpu_budget):
    lanes, second = _fixtures.toy_prepared(64)
    admission_path = tmp_path / "admission.json"
    admission_path.write_text("{}")
    args = _args(tmp_path, cpu_reference_dir=tmp_path / "scalar", admission_record=admission_path)
    args.mode, args.cpu_budget = mode, cpu_budget
    if mode == "pool":
        args.pool_workers = 4
    environment = _real_metadata(args)
    monkeypatch.setattr(base, "scientific_source_digest", lambda: "d" * 64)
    recorder = SimpleNamespace(result={"environment": environment}, sentinel_paths=None)
    candidate = {
        "job_id": "15853683",
        "execution_commit": environment["git_head"],
        "execution_code_dir": str(base.ROOT),
        "baseline_ref": args.baseline_ref,
        "certified_science_sha256": base.scientific_source_digest(),
        "gpu_reference_sha256": args.expected_gpu_reference_sha256,
    }
    calls = []

    monkeypatch.setattr(scale, "screen_inputs", lambda *a: {"unique_real_points": len(lanes)})
    monkeypatch.setattr(base, "legacy_factory", lambda *a: None)
    monkeypatch.setattr(base, "roughheight_sentinel_contract", lambda *a: {})

    def load_reference(directory, identity, **kwargs):
        calls.append((directory, kwargs))
        if kwargs.get("name", "B4_group_references") == "B4_group_references":
            return None, {"sha256": args.expected_gpu_reference_sha256}
        return None, {"sha256": "c" * 64}

    def forbidden(*args, **kwargs):
        pytest.fail("reference preparation must stop before runner creation or compilation")

    monkeypatch.setattr(refs, "load_reference", load_reference)
    monkeypatch.setattr(base, "lower_compile", forbidden)
    monkeypatch.setattr(cpu, "SingleRunner", forbidden)
    monkeypatch.setattr(cpu, "PoolRunner", forbidden)

    prepared = cpu.prepare_references(args, recorder, lanes, second, scale,
        {"candidate_provenance": candidate})
    identity = prepared[0]

    assert identity["runtime"]["git_head"] == environment["git_head"]
    assert recorder.result["reference_preparation_identity"]["execution_commit"] == {
        "candidate": environment["git_head"], "stage": environment["git_head"]}
    assert recorder.result["reference_preparation_cpu_loaded"] == [64]
    assert len(calls) == 2
    if mode == "pool":
        assert identity["pool_execution"]["pool_workers"] == 4
        assert calls[1][1]["candidate_provenance"] == candidate


@pytest.mark.parametrize("mode,cpu_budget", [("batch", 48), ("pool", 4)])
def test_real_reference_loader_rejects_wrong_execution_commit_after_sealed_load(monkeypatch, tmp_path, mode, cpu_budget):
    lanes, second = _fixtures.toy_prepared(64)
    args = _args(tmp_path, mode="scalar", cpu_budget=1)
    environment = _real_metadata(args)
    monkeypatch.setattr(base, "scientific_source_digest", lambda: "d" * 64)
    # The synthetic sealed artifact targets the server loader's accepted runtime gate.
    environment["jax"] = environment["jaxlib"] = "0.4.38"
    source_identity = base.stage_identity(args, lanes, second, environment, include_git_head=True)
    source_identity.update(
        precision_policy=scale.PRECISION_POLICY,
        mode="scalar",
        validation_mode="exploratory",
        cpu_reference_role="accepted",
        numerical_reference_sha256=args.expected_gpu_reference_sha256,
    )

    output = (
        (({"x": np.zeros((64, 1)), "mask": np.ones((64, 1), dtype=bool)},),
         {"daily": np.zeros((64, 7, 48, 1))}),
        (({"x": np.zeros((64, 1)), "mask": np.ones((64, 1), dtype=bool)},),
         {"daily": np.zeros((64, 7, 48, 1))}),
    )
    timing = {
        "B64_warm_frozen_state_core": {"samples_seconds": [1.0] * 5, "valid_point_days": 896},
        "B64_preprepared_wrapper_stack_transfer_materialize": {
            "samples_seconds": [1.0] * 5, "valid_point_days": 896},
    }
    point_ids = [lane.landpoint_id for lane in lanes]
    gates = [{"name": f"B64_scalar_lane_{index}_{point}_GPU_reference", "passed": True,
              "atol": 1e-8, "rtol": 1e-10} for index, point in enumerate(point_ids)]
    compared_leaves = base.compare(output, output)["compared_leaves"]
    gates.append({"name": "B64_scalar_frozen_reset_repeat", "passed": True,
                  "atol": 1e-8, "rtol": 1e-10, "compared_leaves": compared_leaves})
    record = {
        "gates": gates,
        "provenance": {
            "unique_real_points": 64,
            "point_ids": point_ids,
            "precision_policy": scale.PRECISION_POLICY,
            "full_two_block_workload": True,
            "full_outputs": True,
            "blocks": [[2, 8], [9, 15]],
            "half_hours_per_day": 48,
            "full_two_block_parity": True,
            "frozen_reset_repeat": True,
            "five_or_more_synchronized_repeats": True,
        },
        "timing": timing,
        "measurement_status": "complete",
        "evidence_class": "exploratory_performance",
        "strict_numerical_passed": True,
        "execution_origin": {"available": True, "environment": environment},
    }
    value = {"outputs": output, "measurements": timing}
    directory = tmp_path / "sealed-scalar"
    store = StageStore(directory, source_identity)
    store.save("B64_server_scalar_v1", value, record)

    consumer = {**source_identity, "mode": mode, "CPU_budget": cpu_budget,
        "validation_mode": "exploratory", "cpu_reference_role": "cpu-internal"}
    candidate = {
        "job_id": "15853683",
        "execution_commit": environment["git_head"],
        "execution_code_dir": str(base.ROOT),
        "baseline_ref": source_identity["baseline_ref"],
        "certified_science_sha256": source_identity["science_sha256"],
        "gpu_reference_sha256": source_identity["numerical_reference_sha256"],
    }
    admission_path = tmp_path / "admission.json"
    admission_path.write_text(json.dumps({"synthetic": True}))

    # The immutable production admission has additional source/accounting evidence;
    # this mock isolates the real load_reference provenance guard in this synthetic test.
    monkeypatch.setattr(refs, "validate_cpu_reference_admission",
        lambda *a, **kwargs: {"source_job_id": "15820467"})
    loaded, proof = refs.load_reference(directory, consumer, name="B64_server_scalar_v1",
        points=64, cpu_reference_role="cpu-internal", cpu_reference_admission_record=admission_path,
        expected_cpu_reference_admission_sha256="b" * 64, candidate_provenance=candidate)
    assert proof["sha256"] == store.manifest["stages"]["B64_server_scalar_v1"]["sha256"]
    assert loaded[0][0][0]["x"].shape == (64, 1)

    wrong_commit = copy.deepcopy(candidate)
    wrong_commit["execution_commit"] = "0" * 40
    with pytest.raises(ValueError, match="candidate provenance differs from current execution identity"):
        refs.load_reference(directory, consumer, name="B64_server_scalar_v1", points=64,
            cpu_reference_role="cpu-internal", cpu_reference_admission_record=admission_path,
            expected_cpu_reference_admission_sha256="b" * 64, candidate_provenance=wrong_commit)
