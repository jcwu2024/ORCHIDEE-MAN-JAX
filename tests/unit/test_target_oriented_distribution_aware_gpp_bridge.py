from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_gpp_bridge import (
    FROZEN_CONTRACT_SHA256,
    GPP_OUTPUT_KEY,
    classify_gpp_bridge,
    compare_gpp_bridge_modes,
    load_distribution_aware_gpp_bridge_contract,
    replace_teacher_fast_gpp,
)


def _groups(rmse_by_key: Mapping[str, float]) -> list[Mapping[str, Any]]:
    contract = load_distribution_aware_gpp_bridge_contract()
    groups = []
    for update in contract.evaluation_updates:
        horizons = {}
        for horizon in ("7", "30"):
            keys = {
                name: {
                    "evaluated_values": 1.0,
                    "normalized_rmse": float(rmse_by_key.get(f"{horizon}.{name}", 1.0)),
                    "normalized_signed_bias": 0.0,
                    "rmse_maximum": 0.4,
                    "absolute_bias_maximum": 0.2,
                    "passed": float(rmse_by_key.get(f"{horizon}.{name}", 1.0)) <= 0.4,
                }
                for name in contract.reported_keys
            }
            horizons[horizon] = {"keys": keys}
        groups.append(
            {
                "update": update,
                "runtime_year": 1960 + update,
                "acceptance": horizons,
                "hard_failure_count": 0,
            }
        )
    return groups


def test_frozen_contract_loads_zero_training_gpp_boundary() -> None:
    contract = load_distribution_aware_gpp_bridge_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.evaluation_updates == (251, 358, 465, 572, 679, 786, 893, 1000)
    assert contract.downstream_keys == ("growth_respiration", "npp")
    assert contract.payload["intervention"]["learned_output_key"] == GPP_OUTPUT_KEY
    assert contract.payload["execution"]["training_updates"] == 0
    assert not contract.payload["execution"]["architecture_edit_allowed"]


def test_replacement_changes_exactly_one_learned_coordinate() -> None:
    learned = {
        GPP_OUTPUT_KEY: np.asarray([[1.0, 2.0]]),
        "daily.fast_maintenance_respiration": np.asarray([[3.0, 4.0]]),
    }
    teacher = np.asarray([[5.0, 6.0]])
    bridged = replace_teacher_fast_gpp(learned, teacher)
    np.testing.assert_array_equal(bridged[GPP_OUTPUT_KEY], teacher)
    np.testing.assert_array_equal(
        bridged["daily.fast_maintenance_respiration"],
        learned["daily.fast_maintenance_respiration"],
    )
    np.testing.assert_array_equal(learned[GPP_OUTPUT_KEY], np.asarray([[1.0, 2.0]]))


def test_classifier_requires_absolute_gates_and_large_downstream_repairs() -> None:
    contract = load_distribution_aware_gpp_bridge_contract()
    free = _groups({})
    sufficient_groups = _groups(
        {
            f"{horizon}.{name}": 0.1
            for horizon in ("7", "30")
            for name in contract.reported_keys
        }
    )
    sufficient_comparison = compare_gpp_bridge_modes(free, sufficient_groups, contract)
    sufficient = classify_gpp_bridge(
        sufficient_comparison,
        hard_failure_count=0,
        contract=contract,
    )
    assert sufficient["absolute_required_gate_passed"]
    assert sufficient["downstream_scope_gate_passed"]
    assert sufficient["classification"] == "gpp_boundary_supported_as_next_scope"

    contribution_groups = _groups({"7.growth_respiration": 0.8})
    contribution = classify_gpp_bridge(
        compare_gpp_bridge_modes(free, contribution_groups, contract),
        hard_failure_count=0,
        contract=contract,
    )
    assert not contribution["absolute_required_gate_passed"]
    assert contribution["contribution_gate_passed"]
    assert contribution["classification"] == "gpp_boundary_contributes_but_not_sufficient"

    shared = classify_gpp_bridge(
        compare_gpp_bridge_modes(free, _groups({}), contract),
        hard_failure_count=0,
        contract=contract,
    )
    assert shared["classification"] == "shared_attention_or_state_scope_required"


def test_runner_and_launcher_are_forward_only_free_gpu_diagnostics() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root / "scripts" / "hpc" / "run_target_oriented_distribution_aware_gpp_bridge.sh"
    ).read_text(encoding="utf-8")
    runner = (
        root / "scripts" / "hpc" / "audit_target_oriented_distribution_aware_gpp_bridge.py"
    ).read_text(encoding="utf-8")
    step_factory = (
        root / "scripts" / "hpc" / "run_target_oriented_full_step_preflight.py"
    ).read_text(encoding="utf-8")
    assert "gln01|ibgln01|ibc13b02n04" in launcher
    assert "sbatch" not in launcher
    assert '"optimizer_created": False' in runner
    assert '"training_updates": 0' in runner
    assert "value_and_grad" not in runner
    assert '"seven_teacher_gpp_bridge_acceptance_forward"' in step_factory
    assert '"thirty_teacher_gpp_bridge_acceptance_forward"' in step_factory
