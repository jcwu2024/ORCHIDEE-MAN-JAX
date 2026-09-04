from __future__ import annotations

from pathlib import Path

import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    SEVEN_SOURCE,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_zero_displacement import (
    classify_zero_displacement,
)
from scripts.hpc import audit_target_oriented_coupled_gpp_zero_displacement as runner


def _records(*, zero_offset: float = 0.0, step_jump: float = 0.0) -> list[dict]:
    return [
        {
            "artifact_update": update,
            "gradient_baseline": 1.0,
            "exact_zero_forward": 1.0 + (zero_offset if index == 0 else 0.0),
            "smallest_nonzero_forward": 1.0 + (step_jump if index == 0 else 0.0),
        }
        for index, update in enumerate((550, 655, 783, 959, 302, 705, 261, 580))
    ]


def test_zero_displacement_classification_separates_baseline_mismatch_and_local_jump():
    localized = classify_zero_displacement(_records(step_jump=0.1))
    assert localized["classification"] == "localized_nonzero_step_discontinuity"
    assert localized["exact_zero_reproduces_gradient_baseline"] is True
    assert localized["dominant_positive_delta_artifact_update"] == 550
    assert localized["dominant_positive_delta_fraction"] == pytest.approx(1.0)

    mismatch = classify_zero_displacement(_records(zero_offset=0.01, step_jump=0.01))
    assert mismatch["classification"] == "gradient_baseline_not_reproduced_by_exact_zero_forward"
    assert mismatch["exact_zero_reproduces_gradient_baseline"] is False

    continuous = classify_zero_displacement(_records(step_jump=1.0e-5))
    assert continuous["classification"] == "no_material_nonzero_step_discontinuity"


def test_historical_records_uses_smallest_scale_and_requires_scheduled_order():
    updates = (1, 2)
    baseline = [
        {
            "artifact_update": update,
            "source_losses_before": {SEVEN_SOURCE: float(update)},
        }
        for update in updates
    ]

    def scale(value: float) -> dict:
        return {
            "scale": value,
            "parameter_delta_l2": value,
            "scheduled_evaluation": {
                "evaluation": {
                    "records": [
                        {
                            "artifact_update": update,
                            "source_losses": {SEVEN_SOURCE: float(update) + value},
                        }
                        for update in updates
                    ]
                }
            },
        }

    rejected = {
        "proposal": {
            "baseline": {"scheduled_artifact_records": baseline},
            "scale_evaluations": [scale(0.5), scale(0.125)],
        }
    }
    records, selected = runner._historical_records(rejected, expected_updates=updates)
    assert selected["scale"] == pytest.approx(0.125)
    assert records[0]["smallest_nonzero_forward"] == pytest.approx(1.125)

    with pytest.raises(ValueError, match="artifact order"):
        runner._historical_records(rejected, expected_updates=tuple(reversed(updates)))


def test_zero_displacement_launcher_routes_parent_artifacts_and_all_required_options():
    root = Path(__file__).resolve().parents[2]
    shared = (root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh").read_text(
        encoding="utf-8"
    )
    wrapper = (root / "scripts/hpc/run_target_oriented_coupled_gpp_zero_displacement.sh").read_text(
        encoding="utf-8"
    )
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
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-60m}" in wrapper
    assert "PERSISTENT_ARTIFACT_AWARE_RECOVERY" in shared
