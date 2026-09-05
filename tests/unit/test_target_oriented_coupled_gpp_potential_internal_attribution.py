from __future__ import annotations

import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (
    _linear,
    _linear_parameters,
    _residual_block_parameters,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_potential_internal_attribution import (
    BLOCK_PARAMETER_GROUPS,
    COMPONENT_PARAMETER_GROUPS,
    classify_activation_precision_paths,
    classify_parameter_localization,
    replace_parameter_paths,
    summarize_trace_pair,
    traced_potential_branch,
    validate_parameter_group_coverage,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp import (
    _run_gpp_branch,
)
from scripts.hpc import (
    audit_target_oriented_coupled_gpp_potential_internal_attribution as runner,
)


def _blocks(width: int = 4, hidden: int = 8):
    return tuple(_residual_block_parameters(key, width, hidden) for key in jax.random.split(jax.random.PRNGKey(11), 2))


def test_component_groups_exactly_partition_all_potential_block_leaves():
    report = validate_parameter_group_coverage(_blocks())

    assert report["block_count"] == 2
    assert report["leaf_count"] == 32
    assert report["component_group_count"] == 16
    assert report["exact_nonoverlapping_coverage"] is True
    assert set(report["group_leaf_counts"]) == set(COMPONENT_PARAMETER_GROUPS)


def test_parameter_path_replacement_is_structural_and_order_independent():
    lower = _blocks()
    upper = tuple(
        {
            **dict(reversed(tuple(block.items()))),
            "attention": dict(reversed(tuple(block["attention"].items()))),
        }
        for block in jax.tree_util.tree_map(lambda value: value + 1.0, lower)
    )

    mixed = replace_parameter_paths(
        lower,
        upper,
        COMPONENT_PARAMETER_GROUPS["block_1.attention.query"],
    )

    np.testing.assert_array_equal(
        np.asarray(mixed[1]["attention"]["query"]["kernel"]),
        np.asarray(upper[1]["attention"]["query"]["kernel"]),
    )
    np.testing.assert_array_equal(
        np.asarray(mixed[1]["attention"]["key"]["kernel"]),
        np.asarray(lower[1]["attention"]["key"]["kernel"]),
    )


def test_parameter_localization_does_not_choose_a_leaf_for_distributed_effect():
    blocks_only = {name: 1.0 for name in BLOCK_PARAMETER_GROUPS}
    blocks_without = {name: 3.0 for name in BLOCK_PARAMETER_GROUPS}
    blocks_only["block_0"] = 3.0
    blocks_without["block_0"] = 1.0
    components_only = {name: 1.0 for name in COMPONENT_PARAMETER_GROUPS}
    components_without = {name: 3.0 for name in COMPONENT_PARAMETER_GROUPS}

    result = classify_parameter_localization(
        lower_gpp=1.0,
        upper_gpp=3.0,
        block_upper_only_gpp=blocks_only,
        block_upper_without_gpp=blocks_without,
        component_upper_only_gpp=components_only,
        component_upper_without_gpp=components_without,
    )

    assert result["classification"] == "potential_jump_requires_distributed_within_block_interaction"
    assert result["selected_blocks"] == ["block_0"]
    assert result["selected_components"] == []


def test_activation_precision_classification_separates_smoothed_and_sharp_paths():
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    production = (0.0, 0.0, 0.0, 0.0, 1.0)

    smoothed = classify_activation_precision_paths(
        fractions=fractions,
        production_gpp=production,
        full_float64_gpp=fractions,
    )
    sharp = classify_activation_precision_paths(
        fractions=fractions,
        production_gpp=production,
        full_float64_gpp=production,
    )

    assert smoothed["classification"] == "float32_internal_activation_precision_dominant_for_sampled_jump"
    assert sharp["classification"] == "potential_block_conditioning_persists_in_full_float64"


def test_float32_tracer_reproduces_existing_potential_branch_and_float64_is_finite():
    blocks = _blocks()
    output = _linear_parameters(jax.random.PRNGKey(12), 4, 1)
    query = jax.random.normal(jax.random.PRNGKey(13), (2, 4))
    bank = jax.random.normal(jax.random.PRNGKey(14), (2, 3, 4))
    mask = jnp.asarray(((True, True, False), (True, False, True)))
    active = jnp.asarray((True, True))

    traced = traced_potential_branch(
        jax,
        jnp,
        blocks,
        output,
        query,
        bank,
        mask,
        active,
        heads=2,
        dtype=jnp.float32,
    )
    expected_hidden = _run_gpp_branch(
        {},
        query,
        bank,
        mask,
        blocks,
        active=active,
        config=type("Config", (), {"attention_heads": 2})(),
    )
    expected_potential = jax.nn.softplus(_linear(output, expected_hidden)[:, 0])
    np.testing.assert_allclose(np.asarray(traced["hidden"]), np.asarray(expected_hidden), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(traced["potential"]),
        np.asarray(expected_potential),
        rtol=1e-6,
        atol=1e-7,
    )

    traced64 = traced_potential_branch(
        jax,
        jnp,
        blocks,
        output,
        query,
        bank,
        mask,
        active,
        heads=2,
        dtype=jnp.float64,
    )
    assert traced64["potential"].dtype == jnp.float64
    assert np.all(np.isfinite(np.asarray(traced64["potential"])))


def test_trace_pair_summary_reports_finite_stages_and_attention_heads():
    blocks = _blocks()
    output = _linear_parameters(jax.random.PRNGKey(15), 4, 1)
    query = jnp.ones((2, 4), dtype=jnp.float32)
    bank = jnp.ones((2, 3, 4), dtype=jnp.float32)
    mask = jnp.ones((2, 3), dtype=jnp.bool_)
    active = jnp.asarray((True, True))
    lower = traced_potential_branch(jax, jnp, blocks, output, query, bank, mask, active, heads=2, dtype=jnp.float32)
    upper = traced_potential_branch(
        jax,
        jnp,
        jax.tree_util.tree_map(lambda value: value + 1.0e-4, blocks),
        output,
        query,
        bank,
        mask,
        active,
        heads=2,
        dtype=jnp.float32,
    )

    report = summarize_trace_pair(lower, upper, pft_index=1)

    assert report["all_stages_finite"] is True
    assert len(report["attention_heads"]) == 4
    assert "block_0.attention_norm" in report["stages"]
    assert report["stages"]["potential"]["delta_rms"] >= 0.0


def test_parent_report_and_free_gln01_route_are_wired_before_reconstruction():
    root = Path(__file__).resolve().parents[2]
    parent = root / "outputs/diagnostics/target-oriented-coupled-gpp-structured-precision-587c3fd.json"
    report, digest = runner._load_parent_report(parent)
    wrapper = (root / "scripts/hpc/run_target_oriented_coupled_gpp_potential_internal_attribution.sh").read_text(
        encoding="utf-8"
    )
    shared = (root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh").read_text(encoding="utf-8")

    assert digest == runner.PARENT_REPORT_SHA256
    assert report["status"] == "passed"
    assert "free gln01" in wrapper
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-60m}" in wrapper
    assert "sbatch" not in wrapper
    assert "audit_target_oriented_coupled_gpp_potential_internal_attribution" in shared
    assert "--structured-precision-report" in shared
    required = {
        option
        for action in runner._parser()._actions
        if action.required
        for option in action.option_strings
        if option.startswith("--")
    }
    assert sorted(option for option in required if option not in shared and option not in wrapper) == []
    source = inspect.getsource(runner.run)
    assert source.index("_load_parent_report") < source.index("reconstruct_localized_jump_parameters")
    assert source.count("reconstruct_localized_jump_parameters(args, bootstrap=bootstrap)") == 1
