"""Full-parameter canonical daily binding for public rollout differentiation."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .model import bind_daily_operator_dynamic_retained_inputs
from .successor_training import extract_biomass_productivity_trajectories, rollout_input_from_template

FLUX_FIELDS = ('gpp', 'maintenance_respiration', 'growth_respiration', 'heterotrophic_respiration')
STOCK_FIELDS = ('aboveground_biomass_carbon', 'belowground_biomass_carbon')


def prepared_retained_days(window_inputs, window_index, horizon):
    """Broadcast the existing per-window retained bundle along its daily axis.

    Prepared window.day_inputs has a window axis, not a window/day pair.
    Calendar selection still uses each actual day's canonical input inside
    the retained process, exactly as the existing production rollout does.
    """
    return jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(jnp.asarray(value)[window_index], (horizon,)+jnp.asarray(value).shape[1:]),
        window_inputs)


def make_canonical_step(head, retained_template):
    """Use every supplied neural parameter; rebuild all state-derived PFT views."""
    transition = bind_daily_operator_dynamic_retained_inputs(
        input_assembler=head.input_assembler, process_head=head,
        retained_tail_template=retained_template)
    learned_output_names = tuple(field.name for field in head.learned_schema.fields)

    def step(parameters, state, discrete, day):
        template, retained_inputs = day
        current = rollout_input_from_template(template, state, discrete, head.contract)
        result = transition(parameters, current, retained_inputs)
        gpp = result.prediction.named_outputs['daily.fast_gpp']
        observations = {
            'gpp': gpp,
            'gpp_defined': jnp.broadcast_to(current.pft.active_pft_mask[None, :], gpp.shape),
            'fraction': current.pft.pft_fraction,
            'active': current.pft.active_pft_mask,
            'defined': result.next_state_defined.other,
            'discrete': result.next_discrete_state,
            'budgets': jnp.stack((result.diagnostics['water_budget_residual'],
                                  result.diagnostics['carbon_budget_residual'])),
            'learned_output_nonfinite': sum(
                (jnp.sum(~jnp.isfinite(result.prediction.named_outputs[name]), dtype=jnp.int32)
                 for name in learned_output_names),
                jnp.asarray(0, dtype=jnp.int32)),
        }
        return result.next_state.other, result.next_discrete_state, observations

    return step


def make_science_probe_loss(contract, scales):
    """Horizon-agnostic diagnostic loss, not the final multi-window training loss.

    Four cumulative fluxes and two final stocks use existing frozen scales.
    The mean is over active fields. A caller supplies the actual dynamic
    targets/context; current PFT area is an observation, not a fixed template.
    """
    def extract(observations):
        gpp = observations['gpp']
        return extract_biomass_productivity_trajectories(
            observations['state'], observations['defined'], daily_fast_gpp=gpp,
            daily_fast_gpp_defined=observations['gpp_defined'],
            pft_fraction=observations['fraction'], active_pft_mask=observations['active'], contract=contract)

    def loss(parameters, initial_state, observations, targets, context):
        del parameters, initial_state
        predicted, expected = extract(observations), extract(targets)
        horizon = observations['state'].shape[0]
        values, active = [], []
        for name in FLUX_FIELDS:
            a, b = predicted.fluxes[name], expected.fluxes[name]
            valid = a.defined & b.defined
            complete = jnp.all(valid, axis=0)
            residual = (jnp.sum(jnp.where(valid, a.value-b.value, 0.), axis=0)
                        / (horizon*scales.fluxes[name]))
            values.append(jnp.sum(jnp.where(complete, residual*residual, 0.))/jnp.maximum(jnp.sum(complete), 1))
            active.append(jnp.any(complete))
        for name in STOCK_FIELDS:
            a, b = predicted.stocks[name], expected.stocks[name]
            valid = a.defined[-1] & b.defined[-1]
            residual = jnp.where(valid, a.value[-1]-b.value[-1], 0.)/scales.stocks[name]
            values.append(context['stock_weight']*jnp.sum(residual*residual)/jnp.maximum(jnp.sum(valid), 1))
            active.append(jnp.any(valid))
        active = jnp.stack(active)
        return jnp.sum(jnp.where(active, jnp.stack(values), 0.))/jnp.maximum(jnp.sum(active), 1)

    return loss
