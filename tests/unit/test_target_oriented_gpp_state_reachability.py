from __future__ import annotations

from pathlib import Path

import numpy as np

from research.daily_coarse_graining.daily_operator.target_oriented_gpp_state_reachability import (
    GPPStateReachabilityLayout,
    load_gpp_state_reachability_contract,
    summarize_gpp_state_jacobians,
)


def test_reachability_contract_is_frozen():
    contract = load_gpp_state_reachability_contract()
    assert contract.sha256 == "1bf2bbfc13c2b6b4b04d323b1a95917e3b33d501dd249f7b2fd36084307c82af"
    assert contract.payload["data"]["case_count"] == 13
    assert contract.payload["differentiated_boundary"]["input_shape"] == [1, 2]
    assert contract.payload["execution"]["optimizer_updates_authorized"] == 0


def test_exact_nonzero_union_is_grouped_by_source_field():
    layout = GPPStateReachabilityLayout(
        state_indices=np.asarray([10, 11, 20, 21], dtype=np.int64),
        field_index=np.asarray([0, 0, 1, 1], dtype=np.int64),
        field_names=("slow.a", "slow.b"),
        field_state_starts=(10, 20),
        field_state_stops=(12, 22),
        field_shapes=((2,), (2,)),
        field_axis_names=(("x",), ("x",)),
    )
    jacobians = np.zeros((2, 4, 2), dtype=np.float64)
    jacobians[0, 0, 1] = 2.0
    jacobians[1, 1, 1] = 1.0e-20
    jacobians[1, 3, 1] = -3.0
    result = summarize_gpp_state_jacobians(
        jacobians,
        layout=layout,
        case_ids=("first", "second"),
        state_scale=np.asarray([2.0, 2.0, 4.0, 4.0]),
        gpp_scale=np.asarray([1.0, 2.0]),
        normalized_report_threshold=1.0e-12,
    )

    assert result["all_finite"]
    assert result["bare_soil_maximum_absolute_derivative"] == 0.0
    assert result["union_reachable_coordinate_count"] == 3
    assert result["union_reachable_field_count"] == 2
    assert result["normalized_threshold_coordinate_count"] == 2
    assert result["exact_nonzero_below_normalized_threshold_count"] == 1
    assert result["per_case"][0]["reachable_coordinate_count"] == 1
    assert result["per_case"][1]["reachable_coordinate_count"] == 2
    assert result["reachable_fields"][0]["reachable_flat_indices"] == [0, 1]
    assert result["reachable_fields"][1]["reachable_global_state_indices"] == [21]


def test_reachability_runner_is_zero_training_and_free_gpu_only():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/hpc/audit_target_oriented_gpp_state_reachability.py").read_text(encoding="utf-8")
    launcher = (root / "scripts/hpc/run_target_oriented_gpp_state_reachability.sh").read_text(encoding="utf-8")

    assert "jax.jacfwd" in runner
    assert '"training_updates": 0' in runner
    assert '"optimizer_created": False' in runner
    assert '"checkpoint_written": False' in runner
    assert "initialize_target_oriented_adamw" not in runner
    assert "write_broad_checkpoint" not in runner
    assert "free gln01" in launcher
    assert "sbatch" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-120m}" in launcher
