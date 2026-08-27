from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from research.daily_coarse_graining.daily_operator.target_oriented_source_owner_state_bridge import (
    FROZEN_CONTRACT_SHA256,
    REQUIRED_REVERSAL_KEYS,
    SOURCE_OWNER_STATE_KEYS,
    classify_source_owner_state_bridge,
    load_source_owner_state_bridge_contract,
)


def _comparison(value: float) -> Mapping[str, Any]:
    return {
        "all_finite": True,
        "horizons": {
            horizon: {"keys": {name: {"median_relative_score_change": value} for name in REQUIRED_REVERSAL_KEYS}}
            for horizon in ("7", "30")
        },
    }


def test_frozen_contract_loads_exact_four_owner_boundary() -> None:
    contract = load_source_owner_state_bridge_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.state_owner_keys == SOURCE_OWNER_STATE_KEYS
    assert contract.required_reversal_keys == REQUIRED_REVERSAL_KEYS
    assert contract.payload["execution"]["training_updates"] == 0
    assert not contract.payload["execution"]["architecture_edit_allowed"]


def test_classifier_requires_all_ten_cells_for_sufficiency() -> None:
    contract = load_source_owner_state_bridge_contract()
    free = _comparison(0.1)
    teacher = _comparison(-0.1)

    sufficient = classify_source_owner_state_bridge(
        free,
        _comparison(-0.02),
        teacher,
        contract,
    )
    assert sufficient["reversed_cell_count"] == 10
    assert sufficient["classification"] == "source_owner_state_drift_sufficient"

    partial_bridge = _comparison(-0.02)
    partial_bridge["horizons"]["30"]["keys"][REQUIRED_REVERSAL_KEYS[-1]]["median_relative_score_change"] = 0.03
    partial = classify_source_owner_state_bridge(free, partial_bridge, teacher, contract)
    assert partial["reversed_cell_count"] == 9
    assert partial["classification"] == "source_owner_state_drift_contributes_but_not_sufficient"

    absent = classify_source_owner_state_bridge(
        free,
        _comparison(0.05),
        teacher,
        contract,
    )
    assert absent["reversed_cell_count"] == 0
    assert absent["classification"] == "source_owner_state_drift_not_explanatory"


def test_launcher_and_step_factory_keep_bridge_free_and_forward_only() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "scripts" / "hpc" / "run_target_oriented_source_owner_state_bridge.sh").read_text(
        encoding="utf-8"
    )
    assert "gln01|ibgln01|ibc13b02n04" in launcher
    assert "sbatch" not in launcher
    assert "audit_target_oriented_source_owner_state_bridge" in launcher
    assert "--source-coupling-report" in launcher
    source = (root / "scripts" / "hpc" / "run_target_oriented_full_step_preflight.py").read_text(encoding="utf-8")
    assert '"seven_teacher_carry_acceptance_forward"' in source
    assert '"thirty_teacher_carry_acceptance_forward"' in source


def test_runner_declares_zero_training_policy() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts" / "hpc" / "audit_target_oriented_source_owner_state_bridge.py").read_text(
        encoding="utf-8"
    )
    assert '"optimizer_created": False' in source
    assert '"training_updates": 0' in source
    assert '"architecture_changed": False' in source
    assert "value_and_grad" not in source
