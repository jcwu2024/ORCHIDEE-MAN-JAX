from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from research.daily_coarse_graining.daily_operator.target_oriented_maintenance_head_conditioning import (
    AUDIT_UPDATES,
    FROZEN_CONTRACT_SHA256,
    HORIZONS,
    MAINTENANCE_FIELD,
    calibrated_step_records,
    classify_maintenance_head_steps,
    load_target_oriented_maintenance_head_conditioning_contract,
    maintenance_head,
    replace_maintenance_head,
    resolve_maintenance_head_index,
)

ROOT = Path(__file__).resolve().parents[2]


def _group(update: int) -> dict:
    return {
        "update": update,
        "one_day_relative_change": -1.0e-4,
        "independent_flux_relative_changes": {
            str(horizon): {
                "gpp": 0.0,
                "maintenance_respiration": -2.0e-3,
                "growth_respiration": -1.0e-4,
            }
            for horizon in HORIZONS
        },
        "derived_flux_relative_changes": {
            str(horizon): {
                "autotrophic_respiration": -1.0e-4,
                "npp": -1.0e-4,
                "nep": -1.0e-4,
                "nee": -1.0e-4,
            }
            for horizon in HORIZONS
        },
        "stock_relative_changes": {
            str(horizon): {
                "aboveground_biomass_carbon": -1.0e-4,
                "belowground_biomass_carbon": -1.0e-4,
                "total_live_biomass_carbon": -1.0e-4,
                "nonliving_ecosystem_carbon": -1.0e-4,
                "ecosystem_carbon": -1.0e-4,
            }
            for horizon in HORIZONS
        },
        "parent_passing_acceptance_keys_retained": True,
        "parent_failing_stock_rmse_relative_increase_max": 0.0,
        "hard_failure_count": 0,
    }


def _candidates() -> list[dict]:
    contract = load_target_oriented_maintenance_head_conditioning_contract()
    return [
        {
            "predicted_fraction": fraction,
            "step_norm": 0.001 * fraction,
            "within_parameter_bound": True,
            "nonfinite_value_count": 0,
            "groups": [_group(update) for update in AUDIT_UPDATES],
        }
        for fraction in contract.predicted_fractions
    ]


def test_maintenance_head_conditioning_contract_is_frozen() -> None:
    contract = load_target_oriented_maintenance_head_conditioning_contract()

    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.predicted_fractions == (0.001, 0.003, 0.01, 0.03, 0.1)
    assert contract.payload["direction"]["head_parameter_count"] == 769
    assert contract.payload["execution"]["optimizer_updates"] == 0
    assert contract.payload["execution"]["architecture_edit_allowed"] is False


def test_maintenance_head_resolution_and_replacement_are_field_local() -> None:
    schema = SimpleNamespace(
        fields=(SimpleNamespace(name="daily.fast_gpp"), SimpleNamespace(name=MAINTENANCE_FIELD))
    )
    parameters = {
        "shared": np.array([5.0]),
        "target_oriented": {
            "primary_heads": (
                {"kernel": np.zeros((768, 1)), "bias": np.zeros((1,))},
                {"kernel": np.ones((768, 1)), "bias": np.ones((1,))},
            )
        },
    }
    index = resolve_maintenance_head_index(schema)
    original = maintenance_head(parameters, index)
    replacement = {name: np.full_like(value, 2.0) for name, value in original.items()}
    proposed = replace_maintenance_head(parameters, index, replacement)

    assert index == 1
    np.testing.assert_array_equal(proposed["shared"], parameters["shared"])
    np.testing.assert_array_equal(
        proposed["target_oriented"]["primary_heads"][0]["kernel"],
        parameters["target_oriented"]["primary_heads"][0]["kernel"],
    )
    np.testing.assert_array_equal(
        proposed["target_oriented"]["primary_heads"][1]["kernel"],
        replacement["kernel"],
    )


def test_maintenance_head_step_calibration_uses_loss_to_derivative_ratio() -> None:
    contract = load_target_oriented_maintenance_head_conditioning_contract()
    records = calibrated_step_records(
        np.full(16, 2.0),
        np.full(16, 4.0),
        contract,
    )

    assert records[0]["step_norm"] == 0.0005
    assert records[-1]["step_norm"] == 0.05
    assert records[0]["within_parameter_bound"] is True
    assert records[-1]["within_parameter_bound"] is False


def test_maintenance_head_classifier_selects_largest_protected_fraction() -> None:
    contract = load_target_oriented_maintenance_head_conditioning_contract()
    candidates = _candidates()
    candidates[-1]["groups"][0]["one_day_relative_change"] = 0.01

    result = classify_maintenance_head_steps(
        {"minimum_constraint_cosine": 0.1},
        candidates,
        contract,
    )

    assert result["classification"] == "protected_head_step_supported"
    assert result["selected_predicted_fraction"] == 0.03


def test_maintenance_head_classifier_rejects_hidden_npp_regression() -> None:
    contract = load_target_oriented_maintenance_head_conditioning_contract()
    candidates = deepcopy(_candidates())
    for candidate in candidates:
        candidate["groups"][0]["derived_flux_relative_changes"]["30"]["npp"] = 0.02

    result = classify_maintenance_head_steps(
        {"minimum_constraint_cosine": 0.1},
        candidates,
        contract,
    )

    assert result["classification"] == "finite_head_step_not_protected"
    assert result["selected_step_norm"] is None


def test_maintenance_head_classifier_stops_without_common_direction() -> None:
    contract = load_target_oriented_maintenance_head_conditioning_contract()
    result = classify_maintenance_head_steps(
        {"minimum_constraint_cosine": -0.01},
        [],
        contract,
    )

    assert result["classification"] == "common_direction_not_supported"
    assert result["candidates"] == []


def test_maintenance_head_runner_is_free_and_differentiates_only_the_head() -> None:
    preflight = (
        ROOT / "scripts" / "hpc" / "run_target_oriented_full_step_preflight.py"
    ).read_text(encoding="utf-8")
    runner = (
        ROOT
        / "scripts"
        / "hpc"
        / "audit_target_oriented_maintenance_head_conditioning.py"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT
        / "scripts"
        / "hpc"
        / "run_target_oriented_maintenance_head_conditioning_audit.sh"
    ).read_text(encoding="utf-8")

    assert '"seven_maintenance_head_gradient"' in preflight
    assert '"thirty_maintenance_head_gradient"' in preflight
    assert "jax.value_and_grad" in preflight
    assert "differentiated_head_parameter_count" in runner
    assert '"optimizer_invoked": False' in runner
    assert '"architecture_changed": False' in runner
    assert "gln01|ibgln01|ibc13b02n04" in launcher
    assert "sbatch" not in launcher
    assert "gnall" not in launcher
