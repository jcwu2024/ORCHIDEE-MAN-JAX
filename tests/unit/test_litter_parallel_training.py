from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
from research.daily_coarse_graining.daily_operator.mixed_objective import make_mixed_gradient
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    initialize_target_oriented_adamw,
)
from scripts.hpc import litter_parallel_protocol as parallel

ROOT = Path(__file__).resolve().parents[2]




def source_report(checkpoint: Path, plan_identities: dict | None = None) -> dict:
    admission_method = {
        "implementation": "litter_mixed_admission_diagnostic_v1",
        "source_checkpoint": str(checkpoint.with_name("source4000.pkl").resolve()),
        "protocol": {"schema_version": "litter_mixed_admission_v1"},
    }
    method = {
        "implementation": "litter_mixed_selected_continuation_v1",
        "parent_method": admission_method,
    }
    return {
        "schema_version": "litter_mixed_continuation_result_v1",
        "status": "completed",
        "technical_continuation_passed": True,
        "bounded_continuation_completed": True,
        "physical_monitoring_passed": True,
        "scientific_acceptance": False,
        "completed_relative_update": 100,
        "completed_absolute_update": 4100,
        "checkpoint_method": method,
        "plan_identities": plan_identities or {"schedule": "fixed"},
        "milestone_checkpoints": [{
            "relative_update": 100,
            "absolute_update": 4100,
            "path": str(checkpoint.resolve()),
        }],
        "milestone_monitors": [{
            "relative_update": 100,
            "absolute_update": 4100,
            "physical_gate_passed": True,
            "metric_population_complete": True,
            "development_monitor": {"windows": []},
            "daily_panels": {
                "fixed_training_daily": {"rows": []},
                "excluded_train_daily": {"rows": []},
            },
        }],
    }






def test_source100_method_monitor_and_plan_are_exact(tmp_path):
    checkpoint = tmp_path / "selected100.pkl"
    identities = {"schedule": "fixed"}
    report = source_report(checkpoint, identities)
    method, monitor, panels = parallel.source_continuation_method(
        report, checkpoint=checkpoint, plan_identities=identities
    )
    assert method is not report["checkpoint_method"]
    assert method == report["checkpoint_method"]
    assert monitor == {"windows": []}
    assert set(panels) == {"fixed_training_daily", "excluded_train_daily"}

    report["milestone_checkpoints"][0]["absolute_update"] = 4099
    with pytest.raises(ValueError, match="selected100 checkpoint"):
        parallel.source_continuation_method(
            report, checkpoint=checkpoint, plan_identities=identities
        )












def test_arm_d_effective_update_changes_only_learning_rate_scale_and_advances_step():
    base = {
        "optimizer": {
            "name": "AdamW",
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "learning_rate": 1e-5,
            "clip_norm": 1.0,
        }
    }
    arm_a = parallel.effective_optimizer_protocol(base, arm="A", learning_rate=1e-5)
    arm_d = parallel.effective_optimizer_protocol(base, arm="D", learning_rate=3e-6)
    parameters = {"weight": jnp.asarray([0.0, 0.0], dtype=jnp.float32)}
    gradients = {"weight": jnp.asarray([0.25, -0.5], dtype=jnp.float32)}
    state = initialize_target_oriented_adamw(parameters)
    updated_a, state_a, norm_a, clip_a = make_update(arm_a)(parameters, gradients, state)
    updated_d, state_d, norm_d, clip_d = make_update(arm_d)(parameters, gradients, state)
    delta_a = np.asarray(parameters["weight"] - updated_a["weight"])
    delta_d = np.asarray(parameters["weight"] - updated_d["weight"])
    np.testing.assert_allclose(delta_d, 0.3 * delta_a, rtol=1e-6, atol=1e-12)
    assert int(state_a.step) == int(state_d.step) == 1
    np.testing.assert_allclose(norm_a, norm_d, rtol=0, atol=0)
    np.testing.assert_allclose(clip_a, clip_d, rtol=0, atol=0)
