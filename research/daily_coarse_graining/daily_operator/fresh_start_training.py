"""Single-day full-network objective and AdamW for the fresh-start prototype."""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp

from .mechanistic_output_loss import masked_mechanistic_output_loss
from .model import bind_daily_operator_dynamic_retained_inputs
from .parameter_routing import PFT_TRAIT_BARE_SOIL
from .successor_training import extract_biomass_productivity_trajectories
from .target_oriented_contract import necessary_process_supervision_loss
from .target_oriented_training import (
    one_day_state_anchor_from_sufficient_statistics,
    one_day_state_sufficient_statistics,
    target_oriented_adamw_update,
)


def extract_one_day_trajectories(state, defined, gpp, gpp_defined, inputs, contract):
    """Add the same one-day axis to state, GPP and PFT area/mask inputs."""
    return extract_biomass_productivity_trajectories(
        state.other[None, :], defined.other[None, :],
        daily_fast_gpp=gpp[None, ...], daily_fast_gpp_defined=gpp_defined[None, ...],
        pft_fraction=inputs.pft.pft_fraction[None, ...],
        active_pft_mask=inputs.pft.active_pft_mask[None, ...], contract=contract)


def make_objective(runtime, head, template, scales, protocol, *, state_diagnostics=False):
    """Differentiate one canonical day directly, without a custom reverse scan."""
    transition = bind_daily_operator_dynamic_retained_inputs(
        input_assembler=runtime['adapter'].input_assembler, process_head=head,
        retained_tail_template=template)
    flux_names = tuple(protocol['fluxes'])
    weights = protocol['loss']
    check_gsmean = protocol.get('output_applicability', {}).get('gsmean_bare_soil_zero', False)
    if check_gsmean:
        gsmean_key = 'endpoint.fast_state.diffuco_previous_step_state.gsmean'
        gsmean_field = next(field for field in head.learned_schema.fields if field.name == gsmean_key)
        gsmean_leaf = next(leaf for leaf in head.contract.state_leaves
                           if leaf.key == 'diffuco_previous_step_state.gsmean')

    def trajectories(state, defined, gpp, gpp_defined, inputs):
        return extract_one_day_trajectories(state, defined, gpp, gpp_defined, inputs, head.contract)

    def objective(parameters, sample):
        result = transition(parameters, sample.inputs, sample.day_inputs)
        assembled = runtime['adapter'].input_assembler.assemble(sample.inputs)
        process = masked_mechanistic_output_loss(
            result.prediction.named_outputs, sample.process_targets, inputs=assembled,
            complete_schema=head.complete_schema, learned_schema=head.learned_schema,
            scales=runtime['loss_scales'], architecture_contract=head.architecture_contract,
            mechanism_contract=head.mechanism_contract, auxiliary_outputs=result.prediction.auxiliary_outputs)
        process_total = necessary_process_supervision_loss(process, head.target_contract).total
        state_stats = one_day_state_sufficient_statistics(
            result.next_state, sample.expected_next_state,
            result.next_state_defined, sample.expected_next_state_defined, scales.state,
            eligible=scales.state_loss_eligible)
        state_result = one_day_state_anchor_from_sufficient_statistics(state_stats)
        state_total = state_result.total
        target_gpp = sample.process_targets['daily.fast_gpp']
        predicted = trajectories(result.next_state, result.next_state_defined,
            result.prediction.named_outputs['daily.fast_gpp'], target_gpp.defined, sample.inputs)
        expected = trajectories(sample.expected_next_state, sample.expected_next_state_defined,
            target_gpp.value, target_gpp.defined, sample.inputs)
        flux_losses = {}
        for name in flux_names:
            a, b = predicted.fluxes[name], expected.fluxes[name]
            valid = a.defined & b.defined
            delta = (jnp.where(valid, a.value, 0.) - jnp.where(valid, b.value, 0.)) / scales.fluxes[name]
            flux_losses[name] = jnp.sum(delta * delta) / jnp.maximum(jnp.sum(valid), 1)
        flux_total = sum(flux_losses.values()) / len(flux_losses)
        loss = (weights['independent_daily_flux_mse'] * flux_total
                + weights['one_day_state_anchor'] * state_total
                + weights['necessary_process_supervision'] * process_total)
        state_nonfinite, mask_mismatch, declared_mask = 0, 0, 0
        for name in result.next_state._fields:
            valid = getattr(sample.expected_next_state_defined, name)
            actual = getattr(result.next_state_defined, name)
            changed = actual != valid
            dynamic = getattr(scales.state_declared_dynamic, name)
            state_nonfinite += jnp.sum((valid | actual) & ~jnp.isfinite(getattr(result.next_state, name)))
            mask_mismatch += jnp.sum(changed & ~dynamic)
            declared_mask += jnp.sum(changed & dynamic)
        prediction_nonfinite = sum(jnp.sum(sample.process_targets[f.name].defined
            & ~jnp.isfinite(result.prediction.named_outputs[f.name])) for f in head.learned_schema.fields)
        discrete_mismatch = sum(jnp.sum(a != b) for a, b in zip(
            jax.tree_util.tree_leaves(result.next_discrete_state),
            jax.tree_util.tree_leaves(sample.expected_next_discrete_state), strict=True))
        budgets = jnp.stack([result.diagnostics[n] for n in ('water_budget_residual', 'carbon_budget_residual')])
        hard = {'state_nonfinite': state_nonfinite, 'prediction_nonfinite': prediction_nonfinite,
                'mask_mismatch': mask_mismatch, 'discrete_mismatch': discrete_mismatch,
                'budget_failure': jnp.sum(~jnp.isfinite(budgets) | (abs(budgets) > protocol['admission']['budget_absolute_tolerance'])),
                'stock_negative': sum(jnp.sum(v.defined & (v.value < -1e-12)) for v in predicted.stocks.values())}
        if check_gsmean:
            bare = sample.inputs.pft.pft_traits[:, PFT_TRAIT_BARE_SOIL] > .5
            mask_shape = [1] * len(gsmean_field.shape)
            mask_shape[gsmean_field.pft_axis] = bare.shape[0]
            bare = bare.reshape(mask_shape)
            gsmean_output = result.prediction.named_outputs[gsmean_key]
            gsmean_state = result.next_state.other[gsmean_leaf.start:gsmean_leaf.stop].reshape(gsmean_field.shape)
            # Zero remains defined and supervised; this is a physical output
            # invariant, not a loss-eligibility exclusion.
            hard['bare_soil_gsmean_nonzero'] = (jnp.sum(bare & (gsmean_output != 0.))
                                                + jnp.sum(bare & (gsmean_state != 0.)))
        physical = {group: {name: {'predicted': field.value, 'teacher': getattr(expected, group)[name].value,
                                   'defined': field.defined & getattr(expected, group)[name].defined}
                           for name, field in getattr(predicted, group).items()} for group in ('fluxes', 'stocks')}
        # Use the identical Teacher initial state to express biomass growth.
        initial = runtime['adapter'].input_assembler.assemble(sample.inputs)
        start = trajectories(initial.state, initial.state_defined, target_gpp.value, target_gpp.defined, sample.inputs)
        for name in ('aboveground_biomass_carbon', 'belowground_biomass_carbon'):
            row = physical['stocks'][name]
            row['predicted_change'] = row['predicted'] - start.stocks[name].value
            row['teacher_change'] = row['teacher'] - start.stocks[name].value
        maintenance, growth = physical['fluxes']['maintenance_respiration'], physical['fluxes']['growth_respiration']
        physical['fluxes']['autotrophic_respiration'] = {
            'predicted': maintenance['predicted'] + growth['predicted'],
            'teacher': maintenance['teacher'] + growth['teacher'],
            'defined': maintenance['defined'] & growth['defined']}
        auxiliary = {'flux_loss': flux_total, 'state_loss': state_total, 'process_loss': process_total,
                      'flux_losses': flux_losses, 'hard': hard, 'declared_dynamic_mask_mismatch': declared_mask,
                      'budgets': budgets, 'physical': physical}
        if check_gsmean:
            auxiliary['gsmean_applicability'] = {'predicted': gsmean_output, 'next_state': gsmean_state,
                                                'bare_soil': bare}
        if state_diagnostics:
            auxiliary['state_diagnostics'] = {
                'predicted': result.next_state, 'predicted_defined': result.next_state_defined,
                'expected': sample.expected_next_state, 'expected_defined': sample.expected_next_state_defined,
                'scales': scales.state, 'eligible': scales.state_loss_eligible,
                'group_loss': state_result.group_loss, 'group_mse': state_result.group_mse,
                'group_bias_squared': state_result.group_bias_squared, 'group_count': state_result.group_count,
                'next_discrete': result.next_discrete_state,
                'definition_logits': {name: value for name, value in result.prediction.named_outputs.items()
                                      if name.endswith('.defined_logit')},
            }
        return loss, auxiliary

    return objective


def make_update(protocol):
    optimizer = protocol['optimizer']
    contract = SimpleNamespace(optimizer=optimizer)
    return jax.jit(lambda parameters, gradients, state: target_oriented_adamw_update(
        parameters, gradients, state, learning_rate=optimizer['learning_rate'],
        clip_norm=optimizer['clip_norm'], contract=contract))
