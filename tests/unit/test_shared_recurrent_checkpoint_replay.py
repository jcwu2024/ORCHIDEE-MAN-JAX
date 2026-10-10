"""Check identity admission for saved selective-checkpoint diagnostic replay."""
from argparse import Namespace
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from scripts.hpc.probe_shared_execution import compare_trees
from scripts.hpc.probe_shared_recurrent_checkpoint_replay import (
    _checked_comparison,
    _orders,
    _require_memory_peak,
    _require_physical_zero,
    _validate_assets,
)


def _fixtures(tmp_path):
    paths = {name: tmp_path / name for name in (
        "reference_result", "reference_executable", "candidate_result",
        "candidate_executable", "checkpoint")}
    args = Namespace(horizon=7, **paths)
    identity = {"source_head": "source", "source_profile_sha256": "profile",
                "method": {"configuration": {}}, "artifact_id": 742, "horizon": 7,
                "checkpoint": str(paths["checkpoint"]), "jax_version": "0.4.38"}
    reference = {"identity": identity, "status": "passed", "stage": "completed",
                 "mode": "reference", "horizon": 7, "candidate_policy": "policy",
                 "initial_state_gradient_finite": True, "hard_counts": {"hard": 0},
                 "comparisons": {name: {"passed": True} for name in (
                     "loss_and_auxiliary", "all_parameter_gradients")},
                 "reference_result": str(paths["reference_result"]),
                 "reference_executable": str(paths["reference_executable"])}
    candidate = {"identity": identity, "status": "error", "mode": "candidate",
                 "horizon": 7, "candidate_policy": "policy",
                 "parameter_and_initial_state_gradients": True,
                 "candidate_result": str(paths["candidate_result"]),
                 "candidate_executable": str(paths["candidate_executable"])}
    saved = {"identity": identity}
    case = SimpleCase([SimpleNamespace(continuous_state=np.zeros((7,)))])
    bundle = {"method": identity["method"], "profile_sha256": "profile",
              "source_git_head": "source", "artifact_ids": [742],
              "initial_checkpoint": str(paths["checkpoint"]),
              "payloads": [{"cases": {7: case}}]}
    checkpoint = {"method": identity["method"], "completed_updates": 0}
    return args, reference, candidate, saved, bundle, checkpoint


class SimpleCase:
    def __init__(self, days):
        self.days = days

    def __len__(self):
        return 5


def test_failed_parity_candidate_can_be_replayed_when_assets_match(tmp_path):
    args, reference, candidate, saved, bundle, checkpoint = _fixtures(tmp_path)
    case, identity = _validate_assets(args, reference, candidate, saved, saved,
                                      bundle, checkpoint, "profile", "0.4.38")
    assert case.days[0].continuous_state.shape == (7,)
    assert identity["artifact_id"] == 742


def test_replay_rejects_identity_or_horizon_mismatch(tmp_path):
    args, reference, candidate, saved, bundle, checkpoint = _fixtures(tmp_path)
    candidate["identity"] = {**reference["identity"], "artifact_id": 743}
    with pytest.raises(ValueError, match="identities differ"):
        _validate_assets(args, reference, candidate, saved, saved,
                         bundle, checkpoint, "profile", "0.4.38")
    candidate["identity"] = reference["identity"]
    bundle["payloads"][0]["cases"][7].days[0].continuous_state = np.zeros((30,))
    with pytest.raises(ValueError, match="wrong horizon"):
        _validate_assets(args, reference, candidate, saved, saved,
                         bundle, checkpoint, "profile", "0.4.38")


def test_interleaved_pairs_require_three_repeats():
    assert _orders(3) == [("reference", "candidate"), ("candidate", "reference"),
                          ("reference", "candidate")]
    with pytest.raises(ValueError, match="at least three"):
        _orders(2)


def test_finite_strict_mismatch_is_recorded_but_invalid_replay_stops():
    saved = np.array([1.0, 2.0], dtype=np.float64)
    changed = np.array([1.0, 2.01], dtype=np.float64)
    comparison = _checked_comparison(jax, changed, saved, compare_trees)
    assert not comparison["passed"]
    assert comparison["failed_coordinates"] == 1
    with pytest.raises(ValueError, match="shape or dtype"):
        _checked_comparison(jax, changed.astype(np.float32), saved, compare_trees)
    with pytest.raises(ValueError, match="nonfinite"):
        _checked_comparison(jax, np.array([1.0, np.nan]), saved, compare_trees)


def test_physical_and_memory_fail_closed():
    result = ((1.0, {"hard": {"water_negative": 0}}), (1.0, 2.0))
    assert _require_physical_zero(result) == {"water_negative": 0}
    with pytest.raises(ValueError, match="physical hard counts"):
        _require_physical_zero(((1.0, {"hard": {"water_negative": 1}}), (1.0, 2.0)))
    with pytest.raises(MemoryError, match="24 GiB"):
        _require_memory_peak({"peak_bytes_in_use": 25 * 2**30})
