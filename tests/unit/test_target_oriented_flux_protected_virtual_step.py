from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from research.daily_coarse_graining.daily_operator.target_oriented_flux_virtual_step import (
    AUDIT_UPDATES,
    DERIVED_FLUX_KEYS,
    FROZEN_CONTRACT_SHA256,
    HORIZONS,
    OPTIMIZED_FLUX_KEYS,
    classify_flux_protected_virtual_steps,
    load_target_oriented_flux_virtual_step_contract,
)

ROOT = Path(__file__).resolve().parents[2]


def _group(update: int) -> dict:
    return {
        "update": update,
        "one_day_relative_change": -2.0e-5,
        "independent_flux_relative_changes": {
            str(horizon): {key: -2.0e-4 for key in OPTIMIZED_FLUX_KEYS}
            for horizon in HORIZONS
        },
        "derived_flux_relative_changes": {
            str(horizon): {key: -2.0e-4 for key in DERIVED_FLUX_KEYS}
            for horizon in HORIZONS
        },
        "stock_relative_changes": {
            str(horizon): {
                "aboveground_biomass_carbon": -1.0e-5,
                "belowground_biomass_carbon": -1.0e-5,
                "total_live_biomass_carbon": -1.0e-5,
                "nonliving_ecosystem_carbon": -1.0e-5,
                "ecosystem_carbon": -1.0e-5,
            }
            for horizon in HORIZONS
        },
        "parent_passing_acceptance_keys_retained": True,
        "parent_failing_stock_rmse_relative_increase_max": 0.0,
        "hard_failure_count": 0,
    }


def _records() -> list[dict]:
    contract = load_target_oriented_flux_virtual_step_contract()
    return [
        {
            "step_norm": step_norm,
            "nonfinite_value_count": 0,
            "groups": [_group(update) for update in AUDIT_UPDATES],
        }
        for step_norm in contract.step_norms
    ]


def test_target_oriented_flux_virtual_step_contract_is_frozen():
    contract = load_target_oriented_flux_virtual_step_contract()

    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.step_norms == (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
    assert contract.payload["direction"]["constraint_count"] == 16
    assert contract.payload["execution"]["optimizer_updates"] == 0
    assert contract.payload["execution"]["validation_values_evaluated"] is False


def test_flux_virtual_step_classifier_selects_largest_passing_norm():
    contract = load_target_oriented_flux_virtual_step_contract()
    records = _records()
    records[-1]["groups"][0]["one_day_relative_change"] = 0.001

    result = classify_flux_protected_virtual_steps(
        {"minimum_constraint_cosine": 0.01},
        records,
        contract,
    )

    assert result["status"] == "protected_flux_virtual_step_supported"
    assert result["selected_step_norm"] == 0.03
    assert result["candidates"][-1]["passed"] is False


def test_flux_virtual_step_classifier_rejects_hidden_derived_regression():
    contract = load_target_oriented_flux_virtual_step_contract()
    records = _records()
    for record in records:
        record["groups"][0]["derived_flux_relative_changes"]["30"]["npp"] = 0.01

    result = classify_flux_protected_virtual_steps(
        {"minimum_constraint_cosine": 0.01},
        records,
        contract,
    )

    assert result["status"] == "finite_flux_step_not_protected"
    assert result["selected_step_norm"] is None


def test_flux_virtual_step_classifier_closes_when_common_direction_fails():
    contract = load_target_oriented_flux_virtual_step_contract()
    records = deepcopy(_records())

    result = classify_flux_protected_virtual_steps(
        {"minimum_constraint_cosine": -0.001},
        records,
        contract,
    )

    assert result["status"] == "common_flux_direction_not_supported"
    assert result["direction_passed"] is False


def test_flux_virtual_step_runner_is_free_zero_update_and_uses_protected_grid():
    runner = (
        ROOT / "scripts" / "hpc" / "audit_target_oriented_flux_protected_virtual_step.py"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT / "scripts" / "hpc" / "run_target_oriented_flux_protected_virtual_step.sh"
    ).read_text(encoding="utf-8")
    preflight = (
        ROOT / "scripts" / "hpc" / "run_target_oriented_full_step_preflight.py"
    ).read_text(encoding="utf-8")
    validation_loader = (
        ROOT / "scripts" / "hpc" / "target_oriented_validation_prepared.py"
    ).read_text(encoding="utf-8")

    assert "target_oriented_adamw_update" not in runner
    assert '"optimizer_updates_retained": 0' in runner
    assert '"proposed_parameters_retained": False' in runner
    assert "verify_validation_artifacts=False" in runner
    assert "seven_acceptance_forward" in preflight
    assert "verify_artifacts: bool = True" in validation_loader
    assert "gln01|ibgln01|ibc13b02n04" in launcher
    assert "sbatch" not in launcher
    assert "gnall" not in launcher
