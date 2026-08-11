from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import successor_rollout_screen as screen
from research.daily_coarse_graining.daily_operator.long_horizon_objective import MaskedTrajectory
from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from scripts.hpc.successor_rollout_screen_prepared import (
    PREPARED_ARTIFACT_SCHEMA,
    PREPARED_MANIFEST_SCHEMA,
    PreparedRolloutScreenArtifact,
    atomic_json,
    load_prepared_rollout_artifact,
    load_prepared_rollout_manifest,
    write_prepared_rollout_artifact,
)


class _Index:
    def __init__(self, references):
        self.references = tuple(references)

    def select(self, *, spatial_split=None, temporal_split=None):
        return tuple(
            item
            for item in self.references
            if (spatial_split is None or item.parent.spatial_split == spatial_split)
            and (temporal_split is None or item.parent.temporal_split == temporal_split)
        )


def _reference(slice_id: str, ordinal: int, *, days: int = 365):
    splits = {
        "temporal": ("train", "validation"),
        "spatial": ("validation", "train"),
        "joint": ("validation", "validation"),
    }
    spatial, temporal = splits[slice_id]
    parent = SimpleNamespace(
        landpoint_id=f"{slice_id}-{ordinal:02d}",
        year=1961 + ordinal,
        sha256=f"parent-{slice_id}-{ordinal}",
        spatial_split=spatial,
        temporal_split=temporal,
        path=Path(f"{slice_id}-{ordinal}.npz"),
    )
    return SimpleNamespace(
        parent=parent,
        sha256=f"typed-{slice_id}-{ordinal}",
        day_count=days,
        path=Path(f"typed-{slice_id}-{ordinal}.npz"),
    )


@pytest.fixture
def contract():
    return screen.load_successor_rollout_screen_contract()


@pytest.fixture
def index():
    return _Index(
        [_reference(slice_id, ordinal) for slice_id in screen.REQUIRED_SLICES for ordinal in range(10)]
    )


def test_contract_is_hash_frozen(contract):
    assert contract.sha256 == screen.FROZEN_SCREEN_SHA256
    assert contract.payload["data_policy"]["sealed_test_used"] is False
    assert contract.payload["boundary"]["reconstructs_48_forcing_records"] is False
    assert contract.payload["boundary"]["executes_48_step_state_scan"] is False


def test_gpu_runner_uses_only_process_local_jit_cache():
    launcher = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "hpc"
        / "run_successor_rollout_screen.sh"
    ).read_text(encoding="utf-8")
    assert "JAX_COMPILATION_CACHE_DIR" not in launcher
    assert "CACHE_ROOT" not in launcher


def test_window_plan_is_deterministic_matched_and_excludes_first_rows(contract, index):
    first = screen.build_successor_screen_window_plan(index, contract, mode="formal_bounded_screen")
    second = screen.build_successor_screen_window_plan(index, contract, mode="formal_bounded_screen")
    assert first == second
    assert first["plan_sha256"] == second["plan_sha256"]
    assert first["promotion_eligible"] is True
    for slice_id in screen.REQUIRED_SLICES:
        entries = first["slices"][slice_id]["entries"]
        assert len(entries) == 8
        assert len({entry["landpoint_id"] for entry in entries}) == 8
        assert all(len(entry["start_rows"]) == 4 for entry in entries)
        assert all(start >= 1 for entry in entries for start in entry["start_rows"])
        assert all(start + 30 <= entry["day_count"] for entry in entries for start in entry["start_rows"])
        assert all(
            abs(left - right) >= 30
            for entry in entries
            for index, left in enumerate(entry["start_rows"])
            for right in entry["start_rows"][index + 1 :]
        )


def test_free_smoke_inventory_is_not_promotion_eligible(contract, index):
    plan = screen.build_successor_screen_window_plan(index, contract, mode="free_smoke")
    assert plan["promotion_eligible"] is False
    assert sum(len(cell["entries"]) for cell in plan["slices"].values()) == 3


def test_inventory_refuses_any_test_split(contract, index):
    test_reference = _reference("temporal", 99)
    test_reference.parent.spatial_split = "test"
    bad = _Index((*index.references, test_reference))
    # It cannot enter an allowed slice merely by being present elsewhere.
    inventory = screen.resolve_successor_screen_inventory(bad, contract)
    assert all(
        "test" not in (item.parent.spatial_split, item.parent.temporal_split)
        for cell in inventory.values()
        for item in cell["references"]
    )


def test_plan_verification_fails_on_identity_or_first_row_drift(contract, index):
    plan = screen.build_successor_screen_window_plan(index, contract, mode="free_smoke")
    changed = json.loads(json.dumps(plan))
    changed["slices"]["temporal"]["entries"][0]["start_rows"] = [0]
    raw = dict(changed)
    raw.pop("plan_sha256")
    changed["plan_sha256"] = screen._canonical_sha256(raw)
    with pytest.raises(ValueError, match="window identity drift"):
        screen.verify_successor_screen_window_plan(changed, index, contract)


def _statistics(*, error: float = 1.0, bias: float = 0.01, hard: int = 0):
    flux_shape = (2, len(screen.CUMULATIVE_FLUX_FIELDS))
    stock_shape = (2, len(screen.TERMINAL_STOCK_FIELDS))
    flux_count = np.full(flux_shape, 10, dtype=np.int64)
    stock_count = np.full(stock_shape, 10, dtype=np.int64)
    hard_counts = screen.RolloutHardGateCounts(
        *(np.asarray(hard, dtype=np.int64) for _ in screen.RolloutHardGateCounts._fields)
    )
    return screen.RolloutScreenStatistics(
        np.full(flux_shape, error * error * 10),
        np.full(flux_shape, bias * 10),
        np.full(flux_shape, abs(bias) * 10),
        flux_count,
        np.full(stock_shape, error * error * 10),
        np.full(stock_shape, bias * 10),
        np.full(stock_shape, abs(bias) * 10),
        stock_count,
        hard_counts,
    )


def test_statistics_finalize_and_add_are_exact(contract):
    combined = screen.add_rollout_screen_statistics(
        _statistics(error=0.5, bias=0.02),
        _statistics(error=0.5, bias=0.02),
    )
    report = screen.finalize_rollout_screen_statistics(combined, contract=contract)
    assert report["horizons"]["7"]["primary_composite_normalized_rmse"] == pytest.approx(0.5)
    metric = report["horizons"]["30"]["keys"]["cumulative.gpp"]
    assert metric == {
        "normalized_rmse": pytest.approx(0.5),
        "normalized_signed_bias": pytest.approx(0.02),
        "normalized_mean_absolute_error": pytest.approx(0.02),
        "evaluated_values": 20,
    }


def test_cumulative_and_terminal_statistics_use_frozen_horizon_normalization():
    trajectory = MaskedTrajectory(
        value=np.asarray([[3.0, 5.0], [2.0, 4.0]]),
        defined=np.ones((2, 2), dtype=bool),
    )
    expected = MaskedTrajectory(
        value=np.asarray([[1.0, 3.0], [2.0, 2.0]]),
        defined=np.ones((2, 2), dtype=bool),
    )
    squared, signed, absolute, count = screen._trajectory_statistics(
        trajectory,
        expected,
        2.0,
        cumulative=True,
        horizon=2,
    )
    # Residuals are (4 / (2 * 2), 2 / (2 * 2)) = (1, 0.5).
    assert float(squared) == pytest.approx(1.25)
    assert float(signed) == pytest.approx(1.5)
    assert float(absolute) == pytest.approx(1.5)
    assert int(count) == 2

    squared, signed, absolute, count = screen._trajectory_statistics(
        trajectory,
        expected,
        2.0,
        cumulative=False,
        horizon=2,
    )
    assert float(squared) == pytest.approx(2.0)
    assert float(signed) == pytest.approx(2.0)
    assert float(absolute) == pytest.approx(2.0)
    assert int(count) == 2


def _arm_report(contract, arm: str, *, error: float, bias: float = 0.01, mode="formal_bounded_screen"):
    finalized = screen.finalize_rollout_screen_statistics(
        _statistics(error=error, bias=bias),
        contract=contract,
    )
    finalized["window_count"] = 32
    return {
        "schema_version": screen.ARM_REPORT_SCHEMA_VERSION,
        "status": "completed",
        "arm": arm,
        "mode": mode,
        "window_plan_sha256": "plan-hash",
        "window_count": 96,
        "slices": {slice_id: finalized for slice_id in screen.REQUIRED_SLICES},
        "sealed_test_used": False,
    }


def test_classification_passes_declared_matched_improvement(contract):
    candidate = _arm_report(contract, screen.SOURCE_ROUTED_ATTENTION, error=0.90)
    control = _arm_report(contract, screen.SOURCE_ROUTED_GRAPH_CONTROL, error=1.0)
    result = screen.classify_successor_rollout_screen(
        candidate=candidate,
        control=control,
        contract=contract,
    )
    assert result["status"] == "passed"
    assert all(item["passed"] for item in result["ratio_gates"])
    assert all(item["passed"] for item in result["bias_gates"])
    assert all(item["passed"] for item in result["hard_gates"])


def test_classification_rejects_bias_hard_and_window_mismatch(contract):
    candidate = _arm_report(contract, screen.SOURCE_ROUTED_ATTENTION, error=0.90, bias=0.11)
    control = _arm_report(contract, screen.SOURCE_ROUTED_GRAPH_CONTROL, error=1.0)
    candidate["slices"]["joint"]["hard_counts"]["budget_residual_failure"] = 1
    result = screen.classify_successor_rollout_screen(
        candidate=candidate,
        control=control,
        contract=contract,
    )
    assert result["status"] == "rejected"
    assert any(not item["passed"] for item in result["bias_gates"])
    assert any(not item["passed"] for item in result["hard_gates"])

    candidate["window_plan_sha256"] = "different"
    with pytest.raises(ValueError, match="byte-identical windows"):
        screen.classify_successor_rollout_screen(
            candidate=candidate,
            control=control,
            contract=contract,
        )


def test_classification_refuses_free_smoke(contract):
    candidate = _arm_report(
        contract,
        screen.SOURCE_ROUTED_ATTENTION,
        error=0.90,
        mode="free_smoke",
    )
    control = _arm_report(
        contract,
        screen.SOURCE_ROUTED_GRAPH_CONTROL,
        error=1.0,
        mode="free_smoke",
    )
    with pytest.raises(ValueError, match="not promotion eligible"):
        screen.classify_successor_rollout_screen(
            candidate=candidate,
            control=control,
            contract=contract,
        )


def test_checkpoint_admission_requires_successful_hash_bound_dual_fit(
    tmp_path,
    monkeypatch,
    contract,
):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    identity = {"git_head": "a" * 40}
    report = {
        "status": "passed",
        "run_kind": "production_bounded_fit",
        "checkpoint_written": True,
        "promotion_gates_passed": True,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "test_split_used": False,
        "completed_updates_per_arm": {arm: 500 for arm in screen.EXPECTED_ARMS},
        "parameter_count_per_arm": {arm: 8_722_689 for arm in screen.EXPECTED_ARMS},
        "checkpoint_set": {
            "path": str(checkpoint_dir),
            "manifest_sha256": "manifest-hash",
        },
        "identity": identity,
    }
    report_path = tmp_path / "fit_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        screen,
        "load_successor_checkpoint_set",
        lambda *args, **kwargs: {
            arm: {"parameters": {"value": np.asarray([index], dtype=np.float64)}}
            for index, arm in enumerate(screen.EXPECTED_ARMS)
        },
    )
    admitted = screen.admit_successor_screen_checkpoints(
        contract,
        fit_report_path=report_path,
        checkpoint_dir=checkpoint_dir,
        expected_fit_report_sha256=sha256_file(report_path),
        expected_checkpoint_manifest_sha256="manifest-hash",
    )
    assert set(admitted.parameters) == set(screen.EXPECTED_ARMS)
    assert admitted.evidence["optimizer_loaded_for_evaluation"] is False

    report["promotion_gates_passed"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="did not satisfy"):
        screen.admit_successor_screen_checkpoints(
            contract,
            fit_report_path=report_path,
            checkpoint_dir=checkpoint_dir,
            expected_fit_report_sha256=sha256_file(report_path),
            expected_checkpoint_manifest_sha256="manifest-hash",
        )


def test_prepared_artifact_roundtrip_and_hash_failure(tmp_path):
    artifact = PreparedRolloutScreenArtifact(
        PREPARED_ARTIFACT_SCHEMA,
        "temporal",
        {"landpoint_id": "one", "start_rows": [1]},
        "trace",
        {"batch": np.asarray([1.0])},
        {"day": np.asarray([2.0])},
    )
    artifact_path = tmp_path / "screen-001.pkl"
    record = write_prepared_rollout_artifact(artifact_path, artifact) | {
        "slice_id": "temporal",
        "window_entry": dict(artifact.window_entry),
        "trace_signature": "trace",
    }
    manifest = {
        "schema_version": PREPARED_MANIFEST_SCHEMA,
        "status": "passed",
        "screen_contract_sha256": "contract",
        "window_plan_sha256": "plan",
        "source_git_head": "b" * 40,
        "artifact_count": 1,
        "total_size_bytes": record["size_bytes"],
        "entries": [record],
    }
    manifest_path = tmp_path / "prepared_manifest.json"
    atomic_json(manifest_path, manifest)
    loaded_manifest = load_prepared_rollout_manifest(
        manifest_path,
        screen_contract_sha256="contract",
        window_plan_sha256="plan",
        source_git_head="b" * 40,
    )
    loaded = load_prepared_rollout_artifact(
        manifest_path,
        loaded_manifest["entries"][0],
        slice_id="temporal",
        window_entry=artifact.window_entry,
    )
    assert np.array_equal(loaded.batch["batch"], np.asarray([1.0]))

    artifact_path.write_bytes(artifact_path.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="hash drift"):
        load_prepared_rollout_artifact(
            manifest_path,
            loaded_manifest["entries"][0],
            slice_id="temporal",
            window_entry=artifact.window_entry,
        )
