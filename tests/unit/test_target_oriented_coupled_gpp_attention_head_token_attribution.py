from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (
    _linear_parameters,
    _residual_block_parameters,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_attention_head_token_attribution import (
    ATTENTION_HEADS,
    DOC_BANK_INDEX,
    HEAD_WIDTH,
    LITTERPART_BANK_INDEX,
    MODEL_WIDTH,
    SCHEMA_VERSION,
    TARGET_HEAD_INDEX,
    attention_token_summary,
    classify_head_controls,
    projection_head_delta_coverage,
    replace_block_attention_head,
    replace_projection_head,
    state_summary_bank_indices,
    summarize_token_mask_controls,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_potential_internal_attribution import (
    traced_potential_branch,
)
from scripts.hpc import (
    audit_target_oriented_coupled_gpp_attention_head_token_attribution as runner,
)


def _blocks(width: int = 4, hidden: int = 8):
    return tuple(_residual_block_parameters(key, width, hidden) for key in jax.random.split(jax.random.PRNGKey(31), 2))


def test_projection_head_replacement_moves_kernel_columns_and_bias_together():
    lower = {
        "kernel": jnp.arange(32, dtype=jnp.float32).reshape((4, 8)),
        "bias": jnp.arange(8, dtype=jnp.float32),
    }
    upper = jax.tree_util.tree_map(lambda value: value + 100.0, lower)

    result = replace_projection_head(lower, upper, head=1, heads=2)

    np.testing.assert_array_equal(np.asarray(result["kernel"])[:, :4], np.asarray(lower["kernel"])[:, :4])
    np.testing.assert_array_equal(np.asarray(result["kernel"])[:, 4:], np.asarray(upper["kernel"])[:, 4:])
    np.testing.assert_array_equal(np.asarray(result["bias"])[:4], np.asarray(lower["bias"])[:4])
    np.testing.assert_array_equal(np.asarray(result["bias"])[4:], np.asarray(upper["bias"])[4:])


def test_block_head_replacement_can_move_query_without_key():
    lower = _blocks()
    upper = jax.tree_util.tree_map(lambda value: value + 1.0, lower)

    result = replace_block_attention_head(
        lower,
        upper,
        block_index=0,
        head=1,
        heads=2,
        projections=("query",),
    )

    np.testing.assert_array_equal(
        np.asarray(result[0]["attention"]["query"]["kernel"])[:, 2:],
        np.asarray(upper[0]["attention"]["query"]["kernel"])[:, 2:],
    )
    np.testing.assert_array_equal(
        np.asarray(result[0]["attention"]["key"]["kernel"]),
        np.asarray(lower[0]["attention"]["key"]["kernel"]),
    )
    np.testing.assert_array_equal(
        np.asarray(result[1]["attention"]["query"]["kernel"]),
        np.asarray(lower[1]["attention"]["query"]["kernel"]),
    )


def test_projection_head_delta_report_exactly_reconstructs_qk_deltas():
    lower = _blocks()
    upper = jax.tree_util.tree_map(lambda value: value + 0.125, lower)

    report = projection_head_delta_coverage(lower, upper, block_index=0, heads=2)

    assert report["width"] == 4
    assert report["head_count"] == 2
    assert report["head_width"] == 2
    assert report["exact_nonoverlapping_coverage"] is True
    assert set(report["heads"]) == {"head_0", "head_1"}
    assert all(value["squared_delta_reproduced"] for value in report["projections"].values())


def test_head_classification_uses_matched_sufficiency_and_necessity():
    only = {f"head_{head}": 1.0 for head in range(ATTENTION_HEADS)}
    without = {f"head_{head}": 3.0 for head in range(ATTENTION_HEADS)}
    only[f"head_{TARGET_HEAD_INDEX}"] = 3.0
    without[f"head_{TARGET_HEAD_INDEX}"] = 1.0

    result = classify_head_controls(
        lower_gpp=1.0,
        upper_gpp=3.0,
        upper_head_only_gpp=only,
        upper_without_head_gpp=without,
    )

    assert result["classification"] == "target_head_qk_pair_sufficient_and_necessary"
    assert result["target_head_sufficient"] is True
    assert result["target_head_necessary"] is True
    assert result["selected_groups"] == [f"head_{TARGET_HEAD_INDEX}"]


def test_state_summary_bank_indices_reproduce_frozen_doc_and_litterpart_positions():
    keys = [f"state_{index}" for index in range(161)]
    keys[DOC_BANK_INDEX - 45] = "slowproc_stomate_previous_step_state.DOC"
    keys[LITTERPART_BANK_INDEX - 45] = "slowproc_stomate_previous_step_state.litterpart"
    leaves = [SimpleNamespace(key=key, discrete=False) for key in keys]
    leaves.insert(3, SimpleNamespace(key="discrete_ignored", discrete=True))

    result = state_summary_bank_indices(
        leaves,
        forcing_token_count=45,
        keys=(
            "slowproc_stomate_previous_step_state.DOC",
            "slowproc_stomate_previous_step_state.litterpart",
        ),
    )

    assert result["slowproc_stomate_previous_step_state.DOC"] == DOC_BANK_INDEX
    assert result["slowproc_stomate_previous_step_state.litterpart"] == LITTERPART_BANK_INDEX


def test_diagnostic_head_mask_changes_only_declared_attention_row():
    blocks = _blocks()
    output = _linear_parameters(jax.random.PRNGKey(32), 4, 1)
    query = jax.random.normal(jax.random.PRNGKey(33), (2, 4))
    bank = jax.random.normal(jax.random.PRNGKey(34), (2, 3, 4))
    bank_mask = jnp.ones((2, 3), dtype=jnp.bool_)
    active = jnp.ones((2,), dtype=jnp.bool_)
    keep = jnp.ones((2, 2, 2, 3), dtype=jnp.bool_).at[0, 1, 1, 2].set(False)

    baseline = traced_potential_branch(
        jax,
        jnp,
        blocks,
        output,
        query,
        bank,
        bank_mask,
        active,
        heads=2,
        dtype=jnp.float32,
    )
    all_kept = traced_potential_branch(
        jax,
        jnp,
        blocks,
        output,
        query,
        bank,
        bank_mask,
        active,
        heads=2,
        dtype=jnp.float32,
        attention_head_mask=jnp.ones_like(keep),
    )
    masked = traced_potential_branch(
        jax,
        jnp,
        blocks,
        output,
        query,
        bank,
        bank_mask,
        active,
        heads=2,
        dtype=jnp.float32,
        attention_head_mask=keep,
    )

    np.testing.assert_allclose(np.asarray(all_kept["potential"]), np.asarray(baseline["potential"]))
    assert bool(np.asarray(masked["effective_attention_masks"][0])[1, 1, 0, 2]) is False
    assert float(np.asarray(masked["blocks"][0]["attention_weights"])[1, 1, 0, 2]) == 0.0
    np.testing.assert_array_equal(
        np.asarray(masked["blocks"][0]["attention_weights"])[1, 0],
        np.asarray(baseline["blocks"][0]["attention_weights"])[1, 0],
    )
    np.testing.assert_array_equal(
        np.asarray(masked["effective_attention_masks"][1]),
        np.asarray(baseline["effective_attention_masks"][1]),
    )


def test_attention_token_and_matched_mask_summaries_are_explicit():
    blocks = _blocks()
    output = _linear_parameters(jax.random.PRNGKey(35), 4, 1)
    trace = traced_potential_branch(
        jax,
        jnp,
        blocks,
        output,
        jnp.ones((2, 4), dtype=jnp.float32),
        jnp.ones((2, 3, 4), dtype=jnp.float32),
        jnp.ones((2, 3), dtype=jnp.bool_),
        jnp.ones((2,), dtype=jnp.bool_),
        heads=2,
        dtype=jnp.float32,
    )

    attention = attention_token_summary(
        trace,
        pft_index=1,
        block_index=0,
        head_index=1,
        tokens={"first": 0, "last": 2},
    )
    masks = summarize_token_mask_controls(
        lower_gpp=1.0,
        upper_gpp=3.0,
        scenarios={"remove_x": {"lower_gpp": 1.2, "upper_gpp": 1.4}},
    )

    assert attention["valid_bank_count"] == 3
    assert set(attention["tokens"]) == {"first", "last"}
    assert attention["tokens"]["first"]["rank_among_valid"] is not None
    assert np.isclose(masks["scenarios"]["remove_x"]["upper_removal_capture_fraction"], 0.8)
    assert masks["scenarios"]["remove_x"]["upper_token_descriptively_necessary"] is True


def test_contract_geometry_parent_and_free_gln01_route_are_wired_before_reconstruction():
    root = Path(__file__).resolve().parents[2]
    parent = root / "outputs/diagnostics/target-oriented-coupled-gpp-potential-internal-56ec8f2.json"
    report, digest = runner._load_parent_report(parent)
    wrapper = (root / "scripts/hpc/run_target_oriented_coupled_gpp_attention_head_token_attribution.sh").read_text(
        encoding="utf-8"
    )
    shared = (root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh").read_text(encoding="utf-8")

    assert MODEL_WIDTH == 384
    assert ATTENTION_HEADS == 8
    assert HEAD_WIDTH == 48
    assert SCHEMA_VERSION.endswith("_v1")
    assert digest == runner.PARENT_REPORT_SHA256
    assert report["status"] == "passed"
    assert "free gln01" in wrapper
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-45m}" in wrapper
    assert "sbatch" not in wrapper
    assert "POTENTIAL_INTERNAL_RESULT" in shared
    assert "--potential-internal-report" in shared
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
