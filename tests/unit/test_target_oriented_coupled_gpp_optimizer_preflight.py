from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    DIRECT_SOURCE,
    GRADIENT_SOURCES,
    SEVEN_SOURCE,
    STATE_SOURCE,
    THIRTY_SOURCE,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_optimizer_preflight import (
    BACKOFF_FROZEN_CONTRACT_SHA256,
    FROZEN_CONTRACT_SHA256,
    four_source_macro_objectives,
    load_coupled_gpp_optimizer_preflight_contract,
    macro_source_changes,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    initialize_target_oriented_adamw,
    load_target_oriented_training_contract,
    target_oriented_adamw_update,
)
from scripts.hpc.run_target_oriented_coupled_gpp_optimizer_preflight import (
    _fixed_parameter_report,
    _mean_gradient,
    _weighted_gradient,
)


def test_coupled_gpp_optimizer_preflight_contract_freezes_macro_proposal():
    contract = load_coupled_gpp_optimizer_preflight_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.artifact_updates == (352, 353, 354, 355, 356, 357, 359, 360)
    assert contract.coefficients == {
        DIRECT_SOURCE: 1.0,
        STATE_SOURCE: 0.6524275555272667,
        SEVEN_SOURCE: 0.4072260558556481,
        THIRTY_SOURCE: 0.28249951035415655,
    }
    assert contract.clip_threshold == 12.808193184117403
    assert contract.learning_rate == 4.0e-6


def test_coupled_gpp_optimizer_backoff_contract_freezes_one_halved_proposal():
    root = Path(__file__).resolve().parents[2]
    contract = load_coupled_gpp_optimizer_preflight_contract(
        root / "manifests" / "coarse_graining" / "target_oriented_coupled_gpp_optimizer_backoff_v1.json"
    )
    assert contract.sha256 == BACKOFF_FROZEN_CONTRACT_SHA256
    assert contract.artifact_updates == (352, 353, 354, 355, 356, 357, 359, 360)
    assert contract.learning_rate == 2.0e-6
    assert contract.payload["optimizer_proposal"]["independent_proposals"] == 1
    assert contract.payload["optimizer_proposal"]["chained_updates"] == 0


def test_four_source_macro_objective_and_changes_are_exact():
    contract = load_coupled_gpp_optimizer_preflight_contract()
    before = [{name: float(index + 1) for index, name in enumerate(GRADIENT_SOURCES)}] * 2
    after = [{name: 0.5 * float(index + 1) for index, name in enumerate(GRADIENT_SOURCES)}] * 2
    changes = macro_source_changes(before, after)
    assert all(value["relative_change"] == pytest.approx(-0.5) for value in changes.values())
    values = {name: changes[name]["before"] for name in GRADIENT_SOURCES}
    objective = four_source_macro_objectives(values, contract.coefficients)
    assert objective["joint"] == pytest.approx(
        sum(contract.coefficients[name] * values[name] for name in GRADIENT_SOURCES)
    )
    with pytest.raises(ValueError, match="registry drift"):
        four_source_macro_objectives({DIRECT_SOURCE: 1.0}, contract.coefficients)


def test_fresh_adamw_proposal_jits_over_two_trainable_families():
    contract = load_coupled_gpp_optimizer_preflight_contract()
    training = load_target_oriented_training_contract()
    parameters = {
        "structured_gpp": {"value": jnp.asarray([1.0, -2.0])},
        "coupled_gpp_carbon_feedback": {"value": jnp.asarray([3.0])},
    }
    gradients = jax.tree_util.tree_map(jnp.ones_like, parameters)
    optimizer = initialize_target_oriented_adamw(parameters)
    step = jax.jit(
        lambda values, derivatives, state: target_oriented_adamw_update(
            values,
            derivatives,
            state,
            learning_rate=jnp.asarray(4.0e-6, dtype=jnp.float32),
            clip_norm=jnp.asarray(contract.clip_threshold, dtype=jnp.float32),
            contract=training,
        )
    )
    updated, state, raw_norm, clip_scale = step(parameters, gradients, optimizer)
    assert int(state.step) == 1
    assert float(raw_norm) == pytest.approx(3.0**0.5)
    assert float(clip_scale) == pytest.approx(1.0)
    assert not jnp.array_equal(updated["structured_gpp"]["value"], parameters["structured_gpp"]["value"])
    assert not jnp.array_equal(
        updated["coupled_gpp_carbon_feedback"]["value"],
        parameters["coupled_gpp_carbon_feedback"]["value"],
    )


def test_macro_gradient_helpers_average_and_weight_nested_trees():
    gradients = {
        name: {
            "left": jnp.asarray([index + 1.0, 2.0 * (index + 1.0)]),
            "right": (jnp.asarray(index + 0.5),),
        }
        for index, name in enumerate(GRADIENT_SOURCES)
    }
    mean = _mean_gradient(jax, gradients[DIRECT_SOURCE], 2)
    assert jnp.array_equal(mean["left"], jnp.asarray([0.5, 1.0]))
    assert jnp.array_equal(mean["right"][0], jnp.asarray(0.25))

    coefficients = {name: float(index + 1) for index, name in enumerate(GRADIENT_SOURCES)}
    weighted = _weighted_gradient(jax, gradients, coefficients)
    expected_left = sum(coefficients[name] * gradients[name]["left"] for name in GRADIENT_SOURCES)
    expected_right = sum(coefficients[name] * gradients[name]["right"][0] for name in GRADIENT_SOURCES)
    assert jnp.array_equal(weighted["left"], expected_left)
    assert jnp.array_equal(weighted["right"][0], expected_right)


def test_fixed_parameter_report_requires_shared_retained_arrays():
    shared = jnp.asarray([1.0, 2.0])
    before = {
        "target_oriented": {
            "structured_gpp": {"value": jnp.asarray([3.0])},
            "coupled_gpp_carbon_feedback": {"value": jnp.asarray([4.0])},
            "retained": {"value": shared},
        }
    }
    after = {
        "target_oriented": {
            "structured_gpp": {"value": jnp.asarray([5.0])},
            "coupled_gpp_carbon_feedback": {"value": jnp.asarray([6.0])},
            "retained": {"value": shared},
        }
    }
    report = _fixed_parameter_report(jax, before, after)
    assert report == {
        "same_tree_structure": True,
        "same_array_objects": True,
        "leaf_count": 1,
        "element_count": 2,
        "exactly_unchanged": True,
    }

    after["target_oriented"]["retained"]["value"] = jnp.asarray([1.0, 2.0])
    report = _fixed_parameter_report(jax, before, after)
    assert report["same_tree_structure"]
    assert not report["same_array_objects"]
    assert not report["exactly_unchanged"]


def test_optimizer_preflight_runner_is_discarded_and_has_no_training_path():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts" / "hpc" / "run_target_oriented_coupled_gpp_optimizer_preflight.py").read_text(
        encoding="utf-8"
    )
    launcher = (root / "scripts" / "hpc" / "run_target_oriented_coupled_gpp_optimizer_preflight.sh").read_text(
        encoding="utf-8"
    )
    assert "initialize_target_oriented_adamw" in runner
    assert "target_oriented_adamw_update" in runner
    assert '"optimizer_updates_persisted": 0' in runner
    assert '"checkpoint_written": False' in runner
    assert "--rejected-optimizer-preflight-report" in runner
    assert "write_structured_gpp_checkpoint" not in runner
    assert "write_structured_gpp_recovery" not in runner
    assert "sbatch" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-230m}" in launcher
