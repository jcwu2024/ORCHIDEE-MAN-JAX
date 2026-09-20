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
from scripts.hpc.probe_litter_mixed_admission import (
    _align_continuation_history,
    parser,
    run,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "manifests/coarse_graining/litter_parallel_training_v1.json"


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


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


@pytest.mark.parametrize(
    ("arm", "weight", "learning_rate", "t30"),
    [
        ("A", parallel.SELECTED_WEIGHT, 1e-5, 0.5),
        ("B", parallel.QUARTER_WEIGHT, 1e-5, 0.5),
        ("C", parallel.SELECTED_WEIGHT, 1e-5, 1.0),
        ("D", parallel.SELECTED_WEIGHT, 3e-6, 0.5),
    ],
)
def test_repository_protocol_freezes_each_arm(arm, weight, learning_rate, t30):
    admitted = parallel.validate_protocol(protocol(), arm)
    assert admitted["update_indices"] == tuple(range(4100, 4600))
    assert admitted["milestones"] == (200, 350, 600)
    assert admitted["selected_weight"] == weight
    assert admitted["learning_rate"] == learning_rate
    assert admitted["outer_weights"] == {"daily": 1.0, "T7": 0.5, "T30": t30}
    assert admitted["scientific_acceptance"] is False


def test_protocol_rejects_cross_arm_or_common_geometry_drift():
    changed = protocol()
    changed["arms"]["B"]["litter_weight"] = parallel.SELECTED_WEIGHT
    with pytest.raises(ValueError, match="arm B configuration"):
        parallel.validate_protocol(changed, "B")

    changed = protocol()
    changed["milestones"] = [200, 600]
    with pytest.raises(ValueError, match="milestones"):
        parallel.validate_protocol(changed, "A")


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


def test_arm_d_optimizer_is_effective_metadata_without_mutating_source():
    original = {
        "parameter_count": 109_464_375,
        "optimizer": {"name": "AdamW", "learning_rate": 1e-5, "clip_norm": 1.0},
    }
    effective = parallel.effective_optimizer_protocol(
        original, arm="D", learning_rate=3e-6
    )
    assert original["optimizer"]["learning_rate"] == 1e-5
    assert effective["optimizer"]["learning_rate"] == 3e-6
    method = parallel.checkpoint_method(
        source_checkpoint=Path("selected100.pkl"),
        source_method={"implementation": "litter_mixed_selected_continuation_v1"},
        protocol=protocol(),
        arm="D",
        effective_optimizer=effective["optimizer"],
        runner_git_head="head",
    )
    assert method["effective_optimizer"]["learning_rate"] == 3e-6
    assert method["arm"] == "D"


def test_parallel_history_alignment_uses_source100_and_milestone_before_monitor():
    updates = [{"relative_update": value} for value in range(101, 226)]
    retained, monitors, due = _align_continuation_history(
        updates,
        [{"relative_update": 200}],
        source_relative_update=100,
        completed=225,
        milestones=(200, 350, 600),
    )
    assert len(retained) == 125
    assert monitors == [{"relative_update": 200}]
    assert due is None

    retained, monitors, due = _align_continuation_history(
        updates[:100], [], source_relative_update=100, completed=200,
        milestones=(200, 350, 600),
    )
    assert due == 200
    with pytest.raises(ValueError, match="unevaluated milestone"):
        _align_continuation_history(
            updates, [], source_relative_update=100, completed=225,
            milestones=(200, 350, 600),
        )


def test_parallel_artifacts_are_arm_scoped_and_cli_accepts_bounded_probe(tmp_path):
    output = tmp_path / "report.json"
    assert parallel.artifact_path(output, arm="C", recovery=True).name.endswith(
        ".arm-C-recovery.checkpoint.pkl"
    )
    assert parallel.artifact_path(output, arm="C", relative_update=350).name.endswith(
        ".arm-C-update-000350.checkpoint.pkl"
    )
    actions = {option: action for action in parser()._actions for option in action.option_strings}
    assert actions["--parallel-arm"].choices == parallel.ARMS
    assert actions["--parallel-protocol"].type is Path
    assert actions["--source-continuation-report"].type is Path
    assert actions["--max-new-updates"].type is int


def test_parallel_options_must_be_complete_before_execution(tmp_path):
    args = type("Args", (), {
        "continuation_protocol": None,
        "source_admission_report": None,
        "parallel_protocol": PROTOCOL_PATH,
        "parallel_arm": "A",
        "source_continuation_report": None,
        "prepare_only": False,
        "max_new_updates": None,
    })()
    with pytest.raises(ValueError, match="required together"):
        run(args)


@pytest.mark.parametrize("arm", ["A", "C"])
def test_arm_outer_weights_match_one_direct_weighted_scalar_gradient(arm):
    admitted = parallel.validate_protocol(protocol(), arm)
    parameters = {"w": jnp.asarray([0.4, -0.2], dtype=jnp.float32)}

    def old(current):
        return jnp.sum((current["w"] - 0.3) ** 2)

    def litter(current):
        return jnp.sum((2.0 * current["w"] + 0.1) ** 2)

    def t7(current):
        return jnp.sum((current["w"] + 0.7) ** 2)

    def t30(current):
        return jnp.sum((3.0 * current["w"] - 0.4) ** 2)

    combine = make_mixed_gradient(jax, jnp)
    daily = combine(
        jax.grad(old)(parameters),
        jax.grad(litter)(parameters),
        1.0,
        admitted["selected_weight"],
    )
    mixed = combine(
        daily,
        jax.grad(t7)(parameters),
        admitted["outer_weights"]["daily"],
        admitted["outer_weights"]["T7"],
    )
    mixed = combine(
        mixed,
        jax.grad(t30)(parameters),
        1.0,
        admitted["outer_weights"]["T30"],
    )
    direct = jax.grad(lambda current: (
        old(current)
        + admitted["selected_weight"] * litter(current)
        + admitted["outer_weights"]["T7"] * t7(current)
        + admitted["outer_weights"]["T30"] * t30(current)
    ))(parameters)
    np.testing.assert_allclose(mixed["w"], direct["w"], rtol=1e-6, atol=1e-7)


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
