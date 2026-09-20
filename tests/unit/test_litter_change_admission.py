from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.hpc.probe_litter_change_admission import (
    _candidate_model_protocol,
    _litter_panel_summary,
    _select_training_batches,
    _validate_admission_protocol,
    parser,
)


def _protocol():
    return {
        "schema_version": "litter_change_admission_v1",
        "updates_per_arm": 2,
        "batch_indices": [4000, 4001],
        "daily_batch": 64,
        "microbatch": 8,
        "direction_seeds": [20260920, 20260921],
        "coefficients_old": [1.0, 0.1, 0.1, 0.0],
        "coefficients_candidate": [1.0, 0.1, 0.1, 0.1],
        "litter_carbon_stock_change": {
            "weight": 0.1,
            "penalty": "pseudo_huber",
            "scale_source_pft_index": 13,
            "minimum_scale_gC_m2_day": 0.01,
        },
    }


def test_protocol_freezes_paired_geometry_and_candidate_weight():
    parsed = _validate_admission_protocol(_protocol())
    assert parsed["old"] == (1.0, 0.1, 0.1, 0.0)
    assert parsed["candidate"] == (1.0, 0.1, 0.1, 0.1)
    assert parsed["batch_indices"] == (4000, 4001)

    changed = _protocol()
    changed["coefficients_candidate"][-1] = 1.0
    with pytest.raises(ValueError, match="candidate-arm"):
        _validate_admission_protocol(changed)

    changed = _protocol()
    changed["litter_carbon_stock_change"]["minimum_scale_gC_m2_day"] = 0.001
    with pytest.raises(ValueError, match="admitted trial"):
        _validate_admission_protocol(changed)


def test_candidate_protocol_adds_only_litter_configuration_without_mutating_source():
    source = {"loss": {"litterpart_feedback_mse": 0.1}, "optimizer": {"clip_norm": 1.0}}
    candidate = _candidate_model_protocol(source, _protocol())
    assert "litter_carbon_stock_change" not in source
    assert candidate["litter_carbon_stock_change"] == _protocol()["litter_carbon_stock_change"]
    assert {key for key in candidate if candidate.get(key) != source.get(key)} == {
        "litter_carbon_stock_change"
    }

    with pytest.raises(ValueError, match="already declares"):
        _candidate_model_protocol(candidate, _protocol())


def test_batch_selection_requires_exact_source_indices_and_complete_microbatches():
    batches = []
    for index in range(4002):
        microbatches = tuple(SimpleNamespace(draw_count=8) for _ in range(8))
        batches.append(SimpleNamespace(batch_index=index, draw_count=64, microbatches=microbatches))
    selected = _select_training_batches({"batches": batches}, _protocol())
    assert [batch.batch_index for batch in selected] == [4000, 4001]

    batches[4001] = SimpleNamespace(
        batch_index=4001,
        draw_count=64,
        microbatches=tuple(SimpleNamespace(draw_count=8) for _ in range(7)),
    )
    with pytest.raises(ValueError, match="microbatch-8"):
        _select_training_batches({"batches": batches}, _protocol())


def test_litter_panel_summary_keeps_owner_deltas_scales_and_hard_counts():
    def owner(offset):
        return {
            "loss": np.array([0.2 + offset, 0.4 + offset]),
            "predicted_delta": np.array([1.0 + offset, 2.0 + offset]),
            "teacher_delta": np.array([0.5 + offset, 1.5 + offset]),
            "physical_residual": np.array([0.5, 0.5]),
            "normalized_residual": np.array([0.5, 0.5]),
            "valid": np.array([True, True]),
            "valid_count": np.array([1, 1]),
            "expected_count": np.array([1, 1]),
            "invalid_prediction_count": np.array([0, 0]),
            "invalid_target_count": np.array([0, 0]),
            "raw_scale": np.array([0.1 + offset, 0.1 + offset]),
            "effective_scale": np.array([0.1 + offset, 0.1 + offset]),
            "scale_coordinate_count": np.array([4, 4], dtype=np.int32),
        }

    summary = _litter_panel_summary(
        {
            "weight": np.array([0.1, 0.1]),
            "scale_source_pft_index": np.array([13, 13], dtype=np.int32),
            "scale_compact_pft_index": np.array([1, 1], dtype=np.int32),
            "minimum_scale_gC_m2_day": np.array([0.01, 0.01]),
            "hard_invalid_count": np.array([0, 0]),
            "hard_empty_count": np.array([0, 0]),
            "per_owner": {"aboveground": owner(0.0), "belowground": owner(0.2)},
        }
    )
    assert summary["hard_invalid_count"] == 0
    assert summary["per_owner"]["aboveground"]["predicted_delta"] == [1.0, 2.0]
    assert summary["per_owner"]["belowground"]["raw_scale_gC_m2_day"] == pytest.approx(0.3)
    assert summary["per_owner"]["aboveground"]["physical_residual_statistics"]["count"] == 2


def test_cli_requires_separate_admission_protocol_and_source_checkpoint_inputs(tmp_path):
    values = [
        "--dataset-manifest", str(tmp_path / "dataset.json"),
        "--acceptance-report", str(tmp_path / "acceptance.json"),
        "--base-statistics", str(tmp_path / "base.npz"),
        "--process-statistics", str(tmp_path / "process.npz"),
        "--fast-statistics", str(tmp_path / "fast.npz"),
        "--prepared-manifest", str(tmp_path / "prepared.json"),
        "--output", str(tmp_path / "output.json"),
        "--mixed-protocol", str(tmp_path / "mixed.json"),
        "--catalogue-report", str(tmp_path / "catalogue.json"),
        "--recurrent-schedule", str(tmp_path / "schedule.json"),
        "--selection", str(tmp_path / "selection.json"),
        "--execution-policy", str(tmp_path / "execution.json"),
        "--training-report", str(tmp_path / "training.json"),
        "--checkpoint", str(tmp_path / "source.pkl"),
        "--admission-protocol", str(tmp_path / "admission.json"),
    ]
    args = parser().parse_args(values)
    assert args.admission_protocol == tmp_path / "admission.json"
    assert args.expected_update == 4000
    assert args.samples_per_panel == 8
    assert args.checkpoint == tmp_path / "source.pkl"
