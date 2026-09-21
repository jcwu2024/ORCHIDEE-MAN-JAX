"""Three-group objectives for the shared daily Transformer, using native AD."""
from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .mechanistic_output_loss import masked_mechanistic_output_loss
from .model import bind_daily_operator_dynamic_retained_inputs
from .shared_trajectory_loss import (
    STOCK_KEYS,
    SharedComponentTrajectories,
    make_shared_trajectory_loss,
    shared_component_trajectories,
)
from .target_oriented_contract import necessary_process_supervision_loss
from .target_oriented_training import (
    derive_target_oriented_scales,
    one_day_state_anchor_from_sufficient_statistics,
    one_day_state_sufficient_statistics,
)


def one_day_observation(state, defined, gpp, gpp_defined, inputs, *, initial_stock_defined=None):
    """The public rollout observation convention with a singleton day axis."""
    observation = {
        "state": state.other[None, :],
        "defined": defined.other[None, :],
        "gpp": gpp[None, ...],
        "gpp_defined": gpp_defined[None, ...],
        "fraction": inputs.pft.pft_fraction[None, ...],
        "active": inputs.pft.active_pft_mask[None, ...],
    }
    if initial_stock_defined is not None:
        if set(initial_stock_defined) != set(STOCK_KEYS):
            raise ValueError("initial stock-defined registry drift")
        observation["initial_stock_defined"] = initial_stock_defined
    return observation


def initial_component_stock_defined(state, defined, gpp, inputs, contract):
    """Project the common day-start masks onto the seven stock quantities."""
    initial = one_day_observation(
        state,
        defined,
        jnp.zeros_like(gpp),
        jnp.zeros_like(gpp, dtype=jnp.bool_),
        inputs,
    )
    return {
        key: value.defined
        for key, value in shared_component_trajectories(initial, contract).stocks.items()
    }


def shared_trajectories_with_initial_mask(observation, contract):
    """Require the common initial stock to be defined for change supervision."""
    trajectories = shared_component_trajectories(observation, contract)
    initial = observation.get("initial_stock_defined")
    if initial is None:
        return trajectories
    if set(initial) != set(STOCK_KEYS):
        raise ValueError("initial stock-defined registry drift")
    stocks = {}
    for key in STOCK_KEYS:
        trajectory = trajectories.stocks[key]
        initial_defined = jnp.asarray(initial[key], dtype=jnp.bool_)
        try:
            initial_defined = jnp.broadcast_to(initial_defined, jnp.asarray(trajectory.defined).shape)
        except ValueError as error:
            raise ValueError(f"initial stock-defined shape drift for {key}") from error
        stocks[key] = trajectory._replace(defined=trajectory.defined & initial_defined)
    return SharedComponentTrajectories(trajectories.fluxes, stocks)


def derive_shared_state_change_scales(runtime, head):
    """Project frozen TRAIN daily state-delta scales onto the one-day groups."""
    return derive_target_oriented_scales(
        runtime["base_statistics"].arrays["state_delta"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        head.contract,
        flux_basis=(
            "gpp",
            "maintenance_respiration",
            "growth_respiration",
            "heterotrophic_respiration",
        ),
    )


def _validate_state_change_scales(runtime, head, supplied):
    expected = derive_shared_state_change_scales(runtime, head)
    for group in supplied.state._fields:
        actual_value = np.asarray(getattr(supplied.state, group))
        expected_value = np.asarray(getattr(expected.state, group))
        if actual_value.shape != expected_value.shape or not np.array_equal(actual_value, expected_value):
            raise ValueError(f"shared daily state scale must use frozen TRAIN state_delta for {group}")
        for name in ("state_loss_eligible", "state_declared_dynamic"):
            actual_mask = np.asarray(getattr(getattr(supplied, name), group), dtype=bool)
            expected_mask = np.asarray(getattr(getattr(expected, name), group), dtype=bool)
            if actual_mask.shape != expected_mask.shape or not np.array_equal(actual_mask, expected_mask):
                raise ValueError(f"shared daily {name} drift for {group}")


def _science_scale(scales, name):
    return scales[name] if isinstance(scales, Mapping) else getattr(scales, name)


def _science_hard_failures(hard_counts):
    """Expose independent leaves so callers can require every count to be zero."""
    return {
        "science_nonfinite_prediction": hard_counts.nonfinite_defined_prediction,
        "science_nonfinite_target": hard_counts.nonfinite_defined_target,
        "science_mask_mismatch": hard_counts.defined_mask_mismatch,
        "science_missing_valid_term": hard_counts.missing_valid_term,
    }


def compose_shared_groups(process_day, daily_flux, daily_stock, recurrent_flux=(), recurrent_stock=()):
    """Equal group weights; every supplied horizon has equal within-group weight."""
    flux = (daily_flux + sum(recurrent_flux)) / (1 + len(recurrent_flux))
    stock = (daily_stock + sum(recurrent_stock)) / (1 + len(recurrent_stock))
    return (process_day + flux + stock) / 3.0


def make_shared_daily_objective(runtime, head, template, state_scales, science_scales):
    """Full canonical one-day loss; scale inputs must be frozen TRAIN statistics.

`state_scales.state` must use daily state-change scales, not standing stocks.
The process scalar uses existing source-owned typed labels and their weights.
"""
    _validate_state_change_scales(runtime, head, state_scales)
    transition = bind_daily_operator_dynamic_retained_inputs(
        input_assembler=head.input_assembler,
        process_head=head,
        retained_tail_template=template,
    )
    science = make_shared_trajectory_loss(
        contract=head.contract,
        flux_scales=_science_scale(science_scales, "flux_scales"),
        stock_change_scales=_science_scale(science_scales, "stock_change_scales"),
        trajectory_extractor=shared_trajectories_with_initial_mask,
    )

    def objective(parameters, sample):
        result = transition(parameters, sample.inputs, sample.day_inputs)
        assembled = head.input_assembler.assemble(sample.inputs)
        process = masked_mechanistic_output_loss(
            result.prediction.named_outputs, sample.process_targets, inputs=assembled,
            complete_schema=head.complete_schema, learned_schema=head.learned_schema,
            scales=runtime["loss_scales"], architecture_contract=head.architecture_contract,
            mechanism_contract=head.mechanism_contract,
            auxiliary_outputs=result.prediction.auxiliary_outputs)
        process_label = necessary_process_supervision_loss(process, head.target_contract).total
        state = one_day_state_anchor_from_sufficient_statistics(
            one_day_state_sufficient_statistics(
                result.next_state,
                sample.expected_next_state,
                result.next_state_defined,
                sample.expected_next_state_defined,
                state_scales.state,
                eligible=state_scales.state_loss_eligible,
            )
        )
        process_day = 0.5 * (process_label + state.total)
        target_gpp = sample.process_targets["daily.fast_gpp"]
        gpp = result.prediction.named_outputs["daily.fast_gpp"]
        initial_stock_defined = initial_component_stock_defined(
            assembled.state,
            assembled.state_defined,
            gpp,
            sample.inputs,
            head.contract,
        )
        predicted = one_day_observation(
            result.next_state,
            result.next_state_defined,
            gpp,
            target_gpp.defined,
            sample.inputs,
            initial_stock_defined=initial_stock_defined,
        )
        expected = one_day_observation(
            sample.expected_next_state,
            sample.expected_next_state_defined,
            target_gpp.value,
            target_gpp.defined,
            sample.inputs,
            initial_stock_defined=initial_stock_defined,
        )
        terms = science.terms(parameters, sample.inputs.continuous_state, predicted, expected, {})
        loss = compose_shared_groups(process_day, terms.flux, terms.stock)
        state_nonfinite = jnp.asarray(0, dtype=jnp.int32)
        mask_mismatch = jnp.asarray(0, dtype=jnp.int32)
        declared_mask_change = jnp.asarray(0, dtype=jnp.int32)
        for name in result.next_state._fields:
            predicted_defined = getattr(result.next_state_defined, name)
            expected_defined = getattr(sample.expected_next_state_defined, name)
            changed = predicted_defined != expected_defined
            dynamic = getattr(state_scales.state_declared_dynamic, name)
            state_nonfinite += jnp.sum(
                (predicted_defined | expected_defined) & ~jnp.isfinite(getattr(result.next_state, name)),
                dtype=jnp.int32,
            )
            mask_mismatch += jnp.sum(changed & ~dynamic, dtype=jnp.int32)
            declared_mask_change += jnp.sum(changed & dynamic, dtype=jnp.int32)
        prediction_nonfinite = sum(
            jnp.sum(
                sample.process_targets[field.name].defined
                & ~jnp.isfinite(result.prediction.named_outputs[field.name]),
                dtype=jnp.int32,
            )
            for field in head.learned_schema.fields
        )
        budgets = jnp.stack(
            (result.diagnostics["water_budget_residual"], result.diagnostics["carbon_budget_residual"])
        )
        hard = {
            "state_nonfinite": state_nonfinite,
            "prediction_nonfinite": prediction_nonfinite,
            "budget_failure": jnp.sum(~jnp.isfinite(budgets) | (jnp.abs(budgets) > 2.0e-12)),
            "mask_mismatch": mask_mismatch,
            "discrete_mismatch": sum(
                jnp.sum(a != b)
                for a, b in zip(
                    jax.tree_util.tree_leaves(result.next_discrete_state),
                    jax.tree_util.tree_leaves(sample.expected_next_discrete_state),
                    strict=True,
                )
            ),
            "water_negative": jnp.sum(result.next_state_defined.water & (result.next_state.water < -2e-12)),
            "carbon_negative": jnp.sum(result.next_state_defined.carbon & (result.next_state.carbon < -2e-12)),
            **_science_hard_failures(terms.hard_counts),
        }
        return loss, {
            "process_day": process_day,
            "process_labels": process_label,
            "state_change_anchor": state.total,
            "flux": terms.flux,
            "stock": terms.stock,
            "science": terms,
            "hard": hard,
            "declared_mask_change": declared_mask_change,
        }

    return objective
