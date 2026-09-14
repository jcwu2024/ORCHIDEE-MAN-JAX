"""Pure-JAX compact validity counters for recurrent rollouts."""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp

from .successor_training import extract_biomass_productivity_trajectories

PHYSICAL_COUNT_NAMES = (
    "state_nonfinite",
    "budget_failure",
    "mask_mismatch",
    "discrete_mismatch",
    "gpp_nonfinite",
    "fraction_nonfinite",
    "learned_output_nonfinite",
    "stock_negative",
    "bare_soil_gsmean_nonzero",
)


def _count(condition: Any) -> Any:
    return jnp.sum(jnp.asarray(condition, dtype=jnp.bool_), dtype=jnp.int32)


def _integer_sum(value: Any) -> Any:
    return jnp.asarray(jnp.sum(jnp.asarray(value)), dtype=jnp.int32)


def physical_counts(
    obs: Mapping[str, Any],
    targets: Mapping[str, Any],
    expected_discrete: Any,
    templates: Any,
    contract: Any,
    scales: Any,
) -> dict[str, Any]:
    """Match ``probe_public_rollout.physical_checks`` without host transfer."""

    state = jnp.asarray(obs["state"])
    observed_defined = jnp.asarray(obs["defined"], dtype=jnp.bool_)
    target_defined = jnp.asarray(targets["defined"], dtype=jnp.bool_)
    valid = observed_defined | target_defined
    budgets = jnp.asarray(obs["budgets"])
    dynamic = jnp.asarray(scales.state_declared_dynamic.other, dtype=jnp.bool_)

    discrete_mismatch = sum(
        (
            _count(jnp.asarray(actual) != jnp.asarray(expected))
            for actual, expected in zip(
                jax.tree_util.tree_leaves(obs["discrete"]),
                jax.tree_util.tree_leaves(expected_discrete),
                strict=True,
            )
        ),
        start=jnp.asarray(0, dtype=jnp.int32),
    )
    science = extract_biomass_productivity_trajectories(
        state,
        valid,
        daily_fast_gpp=obs["gpp"],
        daily_fast_gpp_defined=obs["gpp_defined"],
        pft_fraction=obs["fraction"],
        active_pft_mask=obs["active"],
        contract=contract,
    )
    gsmean = next(
        leaf
        for leaf in contract.state_leaves
        if leaf.key == "diffuco_previous_step_state.gsmean"
    )
    bare = jnp.asarray(templates.pft.pft_traits)[:, :, 0] > 0.5
    conductance = state[:, gsmean.start : gsmean.stop].reshape(
        (state.shape[0], *gsmean.shape)
    )
    stock_negative = sum(
        (_count(jnp.asarray(value.defined) & (jnp.asarray(value.value) < -1e-12)) for value in science.stocks.values()),
        start=jnp.asarray(0, dtype=jnp.int32),
    )

    return {
        "state_nonfinite": _count(valid & ~jnp.isfinite(state)),
        "budget_failure": _count(~jnp.isfinite(budgets) | (jnp.abs(budgets) > 2e-12)),
        "mask_mismatch": _count((observed_defined != target_defined) & ~dynamic[None, :]),
        "discrete_mismatch": discrete_mismatch,
        "gpp_nonfinite": _count(~jnp.isfinite(jnp.asarray(obs["gpp"]))),
        "fraction_nonfinite": _count(~jnp.isfinite(jnp.asarray(obs["fraction"]))),
        "learned_output_nonfinite": _integer_sum(obs["learned_output_nonfinite"]),
        "stock_negative": stock_negative,
        "bare_soil_gsmean_nonzero": _count(bare[:, None, :] & (conductance != 0.0)),
    }


def feedback_counts(feedback: Mapping[str, Any]) -> dict[str, Any]:
    """Match ``run_trained_forward_screen.feedback_hard_counts`` in JAX."""

    if not feedback["enabled"]:
        return {}

    hard: dict[str, Any] = {}
    canopy = feedback.get("canopy_stock_change")
    if canopy is not None:
        hard["canopy_change_invalid"] = (
            _count(~jnp.asarray(canopy["row_valid"], dtype=jnp.bool_))
            + _count(~jnp.asarray(canopy["scale_valid"], dtype=jnp.bool_))
            + _count(~jnp.asarray(canopy["scale_pft_applicable"], dtype=jnp.bool_))
        )
        hard["canopy_change_empty"] = _count(jnp.asarray(canopy["active_slot_count"]) == 0)

    litterpart = feedback.get("litterpart_feedback")
    if litterpart is not None:
        hard["litterpart_feedback_invalid"] = _integer_sum(litterpart["invalid_count"])
        hard["litterpart_feedback_empty"] = _count(jnp.asarray(litterpart["expected_count"]) == 0)
    return hard
