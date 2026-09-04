from __future__ import annotations

from pathlib import Path

import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_localized_jump import (
    classify_material_days,
    select_largest_positive_window,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    DIRECT_SOURCE,
    SEVEN_SOURCE,
    STATE_SOURCE,
    THIRTY_SOURCE,
)
from scripts.hpc import audit_target_oriented_coupled_gpp_localized_jump as runner


def test_largest_positive_window_is_selected_from_matched_scores():
    result = select_largest_positive_window(
        [
            {"window_index": 0, "control_score": 2.0, "treatment_score": 1.0},
            {"window_index": 1, "control_score": 4.0, "treatment_score": 4.2},
            {"window_index": 2, "control_score": 3.0, "treatment_score": 3.9},
        ]
    )
    assert result["classification"] == "largest_positive_window_selected"
    assert result["selected_window_index"] == 2
    assert result["selected_absolute_displacement"] == pytest.approx(0.9)
    assert result["selected_relative_displacement"] == pytest.approx(0.3)


def test_material_day_classification_prioritizes_the_first_observed_mechanism():
    result = classify_material_days(
        [
            {
                "day_index": 0,
                "discrete_mismatch_count": 0,
                "defined_mask_mismatch_count": 0,
                "maximum_normalized_state_change": 9.0e-4,
            },
            {
                "day_index": 1,
                "discrete_mismatch_count": 0,
                "defined_mask_mismatch_count": 0,
                "maximum_normalized_state_change": 1.0e-3,
            },
            {
                "day_index": 2,
                "discrete_mismatch_count": 1,
                "defined_mask_mismatch_count": 1,
                "maximum_normalized_state_change": 0.0,
            },
        ]
    )
    assert result["classification"] == "continuous_state_change_first"
    assert result["first_material_day_index"] == 1
    assert result["first_material_reasons"] == ["continuous_state"]

    mask_first = classify_material_days(
        [
            {
                "day_index": 0,
                "discrete_mismatch_count": 0,
                "defined_mask_mismatch_count": 1,
                "maximum_normalized_state_change": 0.0,
            }
        ]
    )
    assert mask_first["classification"] == "defined_mask_change_first"


def test_direction_plan_uses_frozen_source_order_and_only_nonzero_weights():
    direct = f"{DIRECT_SOURCE}.artifact_550"
    registry = (direct, STATE_SOURCE, SEVEN_SOURCE, THIRTY_SOURCE)
    rejected = {
        "proposal": {
            "direction": {
                "source_registry": list(registry),
                "combined_gradient_norm": 0.25,
                "full_parameter_step_l2": 0.01,
                "selected_minimum_norm_direction": {
                    "normalized_simplex_weights": {
                        direct: 0.25,
                        STATE_SOURCE: 0.5,
                        SEVEN_SOURCE: 0.0,
                        THIRTY_SOURCE: 0.25,
                    }
                },
            }
        }
    }
    plan = runner._direction_reconstruction_plan(rejected)
    assert plan["source_registry"] == registry
    assert plan["nonzero_direct_updates"] == (550,)
    assert plan["nonzero_scheduled_sources"] == (STATE_SOURCE, THIRTY_SOURCE)
    assert SEVEN_SOURCE not in plan["nonzero_sources"]


def test_localized_jump_launcher_routes_all_required_options_without_sbatch():
    root = Path(__file__).resolve().parents[2]
    shared = (root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh").read_text(encoding="utf-8")
    wrapper = (root / "scripts/hpc/run_target_oriented_coupled_gpp_localized_jump.sh").read_text(encoding="utf-8")
    required_options = {
        option
        for action in runner._parser()._actions
        if action.required
        for option in action.option_strings
        if option.startswith("--")
    }
    missing = sorted(option for option in required_options if option not in shared and option not in wrapper)
    assert missing == []
    assert "sbatch" not in wrapper
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-90m}" in wrapper
    assert "scripts.hpc.audit_target_oriented_coupled_gpp_localized_jump" in shared
