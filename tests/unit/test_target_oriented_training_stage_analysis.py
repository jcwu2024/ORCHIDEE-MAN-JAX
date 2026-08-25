from __future__ import annotations

import pytest

from scripts.hpc.analyze_target_oriented_training_stage import analyze


def _validation(label: str, macro: float, *, passed: bool = False):
    key = {
        "normalized_rmse": macro,
        "normalized_signed_bias": -0.1 * macro,
        "passed": passed,
    }
    return {
        "label": label,
        "science_macro_normalized_rmse": macro,
        "absolute_short_horizon_passed": passed,
        "hard_counts": {
            "budget_failure": 0,
            "declared_dynamic_state_mask_mismatch": 12,
            "discrete_mismatch": 0,
            "prediction_nonfinite": 0,
            "state_nonfinite": 0,
            "stock_negative": 0,
            "unexpected_state_mask_mismatch": 0,
        },
        "slices": {
            "joint": {
                "7": {
                    "all_key_gates_passed": passed,
                    "keys": {"gpp": key},
                }
            }
        },
    }


def _record(update: int, value: float, *, conditioning: bool = False):
    source = {
        "base_source_encoder": {"l2_norm": value},
        "domain_refinement": {"l2_norm": value},
        "science_cross_attention": {"l2_norm": value},
        "science_token_mixer": {"l2_norm": value},
        "primary_heads": {"l2_norm": 2.0 * value},
        "state_heads": {"l2_norm": 3.0 * value},
        "auxiliary_heads": {"l2_norm": 0.0},
    }
    return {
        "update": update,
        "runtime_year": 2000 + update % 2,
        "loss": 4.0 * value,
        "components": {
            "one_day_state_anchor": value,
            "seven_day_science": value,
            "thirty_day_science": value,
            "necessary_process_supervision": value,
        },
        "component_details": {
            "one_day_state_anchor": {"group_loss": {"carbon": value}},
            "seven_day_science": {
                "flux_mse": value,
                "flux_bias_squared": value,
                "stock_mse": value,
                "stock_bias_squared": value,
            },
            "thirty_day_science": {
                "flux_mse": value,
                "flux_bias_squared": value,
                "stock_mse": value,
                "stock_bias_squared": value,
            },
        },
        "gradient_conditioning": (
            {
                "one_day_and_process": source,
                "seven_day_science": source,
                "thirty_day_science": source,
            }
            if conditioning
            else None
        ),
        "unclipped_gradient_norm": value,
        "parameter_step_norm": 0.1 * value,
        "learning_rate": 1.0e-4,
        "clipped": False,
    }


def test_analysis_uses_only_continuation_records_and_applies_five_percent_gate():
    report = {
        "identity": {"stage_start": 2},
        "policy": {"stage_end": 6},
        "initial_validation": _validation("update_2", 1.0),
        "final_validation": _validation("update_6", 0.9),
        "updates": [
            _record(1, 8.0),
            _record(2, 7.0),
            _record(3, 6.0, conditioning=True),
            _record(4, 5.0),
            _record(5, 4.0),
            _record(6, 3.0, conditioning=True),
        ],
    }

    result = analyze(report, block_size=2)

    assert result["stage"] == {"start_update": 2, "end_update": 6, "record_count": 4}
    assert len(result["training_dynamics"]["blocks"]) == 2
    assert len(result["gradient_conditioning"]) == 2
    assert result["gradient_conditioning"][0]["sources"]["seven_day_science"][
        "shared_l2_norm"
    ] == 12.0
    assert result["decision"]["relative_macro_improvement"] == pytest.approx(0.1)
    assert result["decision"]["pacing_improvement_passed"] is True
    assert result["decision"]["action"] == "review_updates_1001_to_3000_continuation"


def test_dynamic_mask_mismatch_is_reported_but_not_a_hard_failure():
    report = {
        "initial_validation": _validation("initialization", 1.0),
        "final_validation": _validation("update_1", 0.99),
        "updates": [_record(1, 1.0)],
    }

    result = analyze(report, block_size=1)

    assert result["validation_timeline"][-1]["hard_failure_count"] == 0
    assert result["decision"]["action"] == "stop_training_and_attribute_failure"


def test_parent_report_builds_initial_parent_and_continuation_timeline():
    parent = {
        "initial_validation": _validation("initialization", 1.2),
        "final_validation": _validation("update_2", 1.0),
    }
    report = {
        "identity": {"stage_start": 2},
        "policy": {"stage_end": 3},
        "initial_validation": _validation("update_2", 1.0),
        "final_validation": _validation("update_3", 0.9),
        "updates": [_record(1, 3.0), _record(2, 2.0), _record(3, 1.0)],
    }

    result = analyze(report, parent=parent, block_size=1)

    assert [value["label"] for value in result["validation_timeline"]] == [
        "initialization",
        "update_2",
        "update_3",
    ]
    assert len(result["validation_changes"]) == 2
