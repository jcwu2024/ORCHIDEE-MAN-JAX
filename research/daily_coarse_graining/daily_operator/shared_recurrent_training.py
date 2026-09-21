"""Native-checkpointed T7/T30 components for the shared daily Transformer.

The prepared windows provide exogenous daily templates and Teacher targets.
Only the first state initializes the rollout; subsequent states are produced by
the canonical transition.  This module changes neither target representations
nor the accepted one-day objective.
"""
from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .canonical_rollout import make_canonical_step, prepared_retained_days
from .recurrent_validation import physical_counts
from .rollout_loss import make_rollout_loss
from .shared_training import (
    initial_component_stock_defined,
    make_shared_daily_objective,
    shared_trajectories_with_initial_mask,
)
from .shared_trajectory_loss import (
    STOCK_KEYS,
    make_shared_trajectory_loss,
    shared_component_trajectories,
)


class SharedRecurrentLossContext(NamedTuple):
    expected_discrete: Any
    templates: Any
    initial_stock_defined: Mapping[str, Any]


class SharedWindowCase(NamedTuple):
    initial_continuous: Any
    initial_discrete: Any
    days: Any
    targets: Mapping[str, Any]
    loss_context: SharedRecurrentLossContext


def _window_slice(tree: Any, window_index: int, horizon: int) -> Any:
    return jax.tree_util.tree_map(
        lambda value: jnp.asarray(value)[window_index, :horizon],
        tree,
    )


def prepare_shared_window(
    head: Any,
    window_batch: Any,
    *,
    window_index: int,
    horizon: int,
) -> SharedWindowCase:
    """Adapt one prepared T7/T30 row without teacher-forcing later states."""

    if horizon not in (7, 30):
        raise ValueError("shared recurrent horizon must be 7 or 30 days")
    continuous_shape = np.shape(window_batch.inputs.continuous_state)
    if len(continuous_shape) != 3:
        raise ValueError("prepared recurrent state must have [window, day, state] axes")
    if not 0 <= int(window_index) < continuous_shape[0] or continuous_shape[1] < horizon:
        raise ValueError("prepared recurrent window index or horizon is out of range")

    templates = _window_slice(window_batch.inputs, int(window_index), horizon)
    day_index = np.asarray(templates.day_index).reshape(-1)
    if day_index.size != horizon or not np.all(np.diff(day_index) == 1):
        raise ValueError("prepared recurrent window must contain consecutive days")
    retained = prepared_retained_days(window_batch.day_inputs, int(window_index), horizon)
    first = jax.tree_util.tree_map(lambda value: value[0], templates)
    assembled = head.input_assembler.assemble(first)

    target_gpp = jnp.asarray(window_batch.expected_fast_gpp)[window_index, :horizon]
    target_gpp_defined = jnp.asarray(window_batch.expected_fast_gpp_defined)[
        window_index, :horizon
    ]
    initial_stock_defined = initial_component_stock_defined(
        assembled.state,
        assembled.state_defined,
        jnp.zeros_like(target_gpp[0]),
        first,
        head.contract,
    )
    if set(initial_stock_defined) != set(STOCK_KEYS):
        raise ValueError("shared recurrent initial stock mask registry drift")

    expected_discrete = _window_slice(
        window_batch.expected_next_discrete_state,
        int(window_index),
        horizon,
    )
    targets = {
        "state": jnp.asarray(window_batch.expected_next_state.other)[window_index, :horizon],
        "defined": jnp.asarray(window_batch.expected_next_state_defined.other)[
            window_index, :horizon
        ],
        "gpp": target_gpp,
        "gpp_defined": target_gpp_defined,
        "fraction": templates.pft.pft_fraction,
        "active": templates.pft.active_pft_mask,
        "initial_stock_defined": initial_stock_defined,
    }
    context = SharedRecurrentLossContext(
        expected_discrete=expected_discrete,
        templates=templates,
        initial_stock_defined=initial_stock_defined,
    )
    return SharedWindowCase(
        initial_continuous=templates.continuous_state[0],
        initial_discrete=jax.tree_util.tree_map(lambda value: value[0], templates.discrete_state),
        days=(templates, retained),
        targets=targets,
        loss_context=context,
    )


def _with_initial_stock_mask(
    values: Mapping[str, Any], initial_stock_defined: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(values)
    result["initial_stock_defined"] = initial_stock_defined
    return result


def _science_hard_counts(counts: Any) -> dict[str, Any]:
    return {
        "science_nonfinite_prediction": counts.nonfinite_defined_prediction,
        "science_nonfinite_target": counts.nonfinite_defined_target,
        "science_mask_mismatch": counts.defined_mask_mismatch,
        "science_missing_valid_term": counts.missing_valid_term,
    }


def _shared_stock_negative(trajectories: Any) -> Any:
    return sum(
        (
            jnp.sum(
                jnp.asarray(value.defined, dtype=jnp.bool_)
                & (jnp.asarray(value.value, dtype=jnp.float64) < -1.0e-12),
                dtype=jnp.int32,
            )
            for value in trajectories.stocks.values()
        ),
        start=jnp.asarray(0, dtype=jnp.int32),
    )


def make_shared_recurrent_objective(
    head: Any,
    retained_template: Any,
    state_scales: Any,
    science_scales: Any,
) -> tuple[Any, Any]:
    """Return the public rollout and its native-checkpointed T7/T30 component.

    Each horizon contributes ``(flux + stock) / 9`` to the fixed objective
    containing one daily, one T7, and one T30 component.
    """

    flux_scales = (
        science_scales["flux_scales"]
        if isinstance(science_scales, Mapping)
        else science_scales.flux_scales
    )
    stock_scales = (
        science_scales["stock_change_scales"]
        if isinstance(science_scales, Mapping)
        else science_scales.stock_change_scales
    )
    science = make_shared_trajectory_loss(
        contract=head.contract,
        flux_scales=flux_scales,
        stock_change_scales=stock_scales,
        trajectory_extractor=shared_trajectories_with_initial_mask,
    )

    def terms_for(parameters, initial_continuous, observations, targets, context):
        predicted = _with_initial_stock_mask(observations, context.initial_stock_defined)
        expected = _with_initial_stock_mask(targets, context.initial_stock_defined)
        terms = science.terms(
            parameters,
            initial_continuous,
            predicted,
            expected,
            context,
        )
        return terms, predicted, expected

    def recurrent_loss(parameters, initial_continuous, observations, targets, context):
        terms, _predicted, _expected = terms_for(
            parameters, initial_continuous, observations, targets, context
        )
        return (terms.flux + terms.stock) / 9.0

    rollout = make_rollout_loss(
        jax,
        jnp,
        make_canonical_step(head, retained_template),
        recurrent_loss,
    )

    def loss_with_aux(parameters, initial_continuous, observations, targets, context):
        terms, predicted, expected = terms_for(
            parameters, initial_continuous, observations, targets, context
        )
        hard = physical_counts(
            observations,
            targets,
            context.expected_discrete,
            context.templates,
            head.contract,
            state_scales,
        )
        hard = {**hard, **_science_hard_counts(terms.hard_counts)}
        hard["shared_stock_negative"] = _shared_stock_negative(
            shared_component_trajectories(observations, head.contract)
        )
        observed_defined = jnp.asarray(observations["defined"], dtype=jnp.bool_)
        target_defined = jnp.asarray(expected["defined"], dtype=jnp.bool_)
        dynamic = jnp.asarray(state_scales.state_declared_dynamic.other, dtype=jnp.bool_)
        declared_mask_change = jnp.sum(
            (observed_defined != target_defined) & dynamic[None, :],
            dtype=jnp.int32,
        )
        loss = (terms.flux + terms.stock) / 9.0
        auxiliary = {
            "flux": terms.flux,
            "stock": terms.stock,
            "living_stock": terms.living_stock,
            "nonliving_stock": terms.nonliving_stock,
            "hard": hard,
            "declared_mask_change": declared_mask_change,
        }
        return loss, auxiliary

    return rollout, rollout.bind_checkpointed_loss_with_aux(loss_with_aux)


def make_shared_joint_daily_objective(
    runtime: Any,
    head: Any,
    retained_template: Any,
    state_scales: Any,
    science_scales: Any,
    *,
    include_state_diagnostics: bool = False,
) -> Any:
    """Return the daily contribution to the fixed daily+T7+T30 objective."""

    daily = make_shared_daily_objective(
        runtime,
        head,
        retained_template,
        state_scales,
        science_scales,
        include_state_diagnostics=include_state_diagnostics,
    )

    def objective(parameters, sample):
        _daily_loss, auxiliary = daily(parameters, sample)
        loss = auxiliary["process_day"] / 3.0 + (
            auxiliary["flux"] + auxiliary["stock"]
        ) / 9.0
        result = dict(auxiliary)
        result["joint_daily_contribution"] = loss
        return loss, result

    return objective


__all__ = [
    "SharedRecurrentLossContext",
    "SharedWindowCase",
    "make_shared_joint_daily_objective",
    "make_shared_recurrent_objective",
    "prepare_shared_window",
]
