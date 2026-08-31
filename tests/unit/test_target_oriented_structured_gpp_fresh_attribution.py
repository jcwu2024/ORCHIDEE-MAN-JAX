from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_fresh_attribution import (
    add_residual_decomposition_statistics,
    classify_fresh_attribution,
    finalize_residual_decomposition_statistics,
    load_structured_gpp_fresh_attribution_contract,
    residual_decomposition_sufficient_statistics,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "manifests" / "coarse_graining" / "target_oriented_structured_gpp_fresh_attribution_v1.json"


def test_fresh_attribution_contract_freezes_three_zero_training_modes():
    contract = load_structured_gpp_fresh_attribution_contract(CONTRACT)
    assert contract.sha256 == contract.payload["contract_sha256"]
    assert set(contract.payload["modes"]) == {
        "ordinary_free_rollout",
        "teacher_day_start_direct_gpp",
        "teacher_gpp_bridge",
    }
    assert contract.payload["execution"]["optimizer_updates"] == 0
    assert contract.payload["execution"]["architecture_edit_allowed"] is False


def test_gpp_residual_decomposition_is_exact_and_mergeable():
    statistics = residual_decomposition_sufficient_statistics(
        expected=jnp.asarray([1.0, 2.0]),
        teacher_state_predicted=jnp.asarray([2.0, 1.0]),
        free_state_predicted=jnp.asarray([3.0, 3.0]),
        defined=jnp.asarray([True, True]),
        scale=1.0,
    )
    report = finalize_residual_decomposition_statistics(statistics)
    assert report["local"]["normalized_rms"] == pytest.approx(1.0)
    assert report["state_feedback"]["normalized_rms"] == pytest.approx((2.5) ** 0.5)
    assert report["total"]["normalized_rms"] == pytest.approx((2.5) ** 0.5)
    assert report["maximum_absolute_residual_closure_error"] == 0.0
    assert report["squared_decomposition_relative_error"] == 0.0

    merged = finalize_residual_decomposition_statistics(add_residual_decomposition_statistics(statistics, statistics))
    assert merged["evaluated_values"] == 4.0
    assert merged["total"]["normalized_rms"] == report["total"]["normalized_rms"]


def _key(rmse: float, bias: float, passed: bool):
    return {
        "normalized_rmse": rmse,
        "normalized_signed_bias": bias,
        "passed": passed,
    }


def test_fresh_attribution_classifies_direct_feedback_and_bridge_evidence():
    direct = {}
    decomposition = {}
    free = {}
    bridge = {}
    for slice_id in ("temporal", "spatial", "joint"):
        direct[slice_id] = {}
        decomposition[slice_id] = {}
        free[slice_id] = {}
        bridge[slice_id] = {}
        for horizon in ("7", "30"):
            direct[slice_id][horizon] = {"passed": False}
            decomposition[slice_id][horizon] = {"state_feedback": {"rms_to_total_ratio": 0.5}}
            free[slice_id][horizon] = {
                "keys": {
                    "growth_respiration": _key(1.0, 0.2, False),
                    "npp": _key(1.2, 0.1, False),
                }
            }
            bridge[slice_id][horizon] = {
                "keys": {
                    "growth_respiration": _key(0.1, 0.01, True),
                    "npp": _key(0.1, 0.01, True),
                }
            }
    result = classify_fresh_attribution(
        direct,
        decomposition,
        free,
        bridge,
        load_structured_gpp_fresh_attribution_contract(CONTRACT),
    )
    assert result["direct_mapping_insufficient"] is True
    assert result["state_feedback_material"] is True
    assert result["teacher_gpp_sufficient_for_required_downstream"] is True
    assert result["classification"] == "direct_and_feedback_gpp_with_sufficient_downstream_cause"


def test_fresh_attribution_runner_is_read_only_and_free_gpu_only():
    runner = (ROOT / "scripts" / "hpc" / "audit_target_oriented_structured_gpp_fresh_attribution.py").read_text(
        encoding="utf-8"
    )
    launcher = (ROOT / "scripts" / "hpc" / "run_target_oriented_structured_gpp_fresh_attribution.sh").read_text(
        encoding="utf-8"
    )
    assert "rollout_with_teacher_fast_gpp" in runner
    assert "residual_decomposition_sufficient_statistics" in runner
    assert "write_structured_gpp_checkpoint" not in runner
    assert "sbatch" not in launcher
    assert "--fresh-formal-report" in launcher
    assert "--attribution-contract" in launcher
