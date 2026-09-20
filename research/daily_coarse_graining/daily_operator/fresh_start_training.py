"""Single-day full-network objective and AdamW for the fresh-start prototype."""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from jax_orchidee.stomate.carbon_kernels import ICARBON

from .canopy_change_loss import canopy_carbon_stock_change_loss
from .feedback_state_loss import feedback_state_mse
from .litter_change_loss import (
    LITTER_OWNER_NAMES,
    derive_litter_change_scales,
    extract_litter_carbon_pools,
    litter_carbon_stock_change_loss,
    resolve_litter_carbon_layout,
)
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

CANOPY_STOCK_KEY = "slowproc_stomate_previous_step_state.interception_storage"
LITTERPART_FEEDBACK_KEY = "slowproc_stomate_previous_step_state.litterpart"


def _canopy_change_plan(runtime, contract, protocol):
    """Resolve an optional loss term from frozen state-delta statistics."""

    specification = protocol.get("canopy_carbon_stock_change")
    if specification is None:
        return None
    if not isinstance(specification, dict):
        raise ValueError("canopy carbon stock-change protocol must be a mapping")
    required = {"weight", "scale_source_pft_index"}
    allowed = required | {"penalty"}
    if not required.issubset(specification) or not set(specification).issubset(allowed):
        raise ValueError("canopy carbon stock-change protocol keys drift")
    weight = float(specification["weight"])
    source_pft_index = int(specification["scale_source_pft_index"])
    penalty = str(specification.get("penalty", "mse"))
    if penalty not in {"mse", "pseudo_huber"}:
        raise ValueError("canopy carbon stock-change penalty is unsupported")
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("canopy carbon stock-change weight must be finite and nonnegative")
    if weight == 0.0:
        return None

    matches = tuple(leaf for leaf in contract.state_leaves if leaf.key == CANOPY_STOCK_KEY)
    if len(matches) != 1:
        raise ValueError("canopy carbon state owner drift")
    leaf = matches[0]
    if leaf.start is None or leaf.stop is None:
        raise ValueError("canopy carbon state owner must be continuous")
    if "nvm" not in leaf.axis_names or "nelements" not in leaf.axis_names:
        raise ValueError("canopy carbon state owner requires compact PFT and element axes")
    pft_axis = leaf.axis_names.index("nvm")
    carbon_axis = leaf.axis_names.index("nelements")
    if leaf.shape[carbon_axis] <= ICARBON:
        raise ValueError("canopy carbon state owner has no carbon coordinate")
    selected_pft = tuple(int(index) for index in leaf.selected_pft_indices)
    if len(selected_pft) != leaf.shape[pft_axis]:
        raise ValueError("canopy carbon compact PFT layout drift")
    try:
        scale_pft_index = selected_pft.index(source_pft_index)
    except ValueError as error:
        raise ValueError("canopy scale source PFT is absent from the compact layout") from error

    delta_scale = np.asarray(runtime["base_statistics"].arrays["state_delta"].scale, dtype=np.float64)
    if delta_scale.shape != (contract.continuous_state_width,):
        raise ValueError("frozen state-delta scale width drift")
    field_scale = delta_scale[leaf.start : leaf.stop].reshape(leaf.shape)
    carbon_scale = np.take(field_scale, ICARBON, axis=carbon_axis)
    compact_pft_axis = pft_axis - int(carbon_axis < pft_axis)
    remaining_shape = tuple(
        size for axis, size in enumerate(carbon_scale.shape) if axis != compact_pft_axis
    )
    if any(size != 1 for size in remaining_shape):
        raise ValueError("canopy carbon loss requires singleton non-PFT state axes")
    scale_by_pft = np.moveaxis(carbon_scale, compact_pft_axis, -1).reshape(-1)
    if scale_by_pft.shape != (len(selected_pft),):
        raise ValueError("canopy carbon scale compact PFT shape drift")
    scale = float(scale_by_pft[scale_pft_index])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("frozen canopy state-delta scale must be finite and positive")
    return SimpleNamespace(
        weight=weight,
        leaf=leaf,
        pft_axis=pft_axis,
        carbon_axis=carbon_axis,
        compact_pft_axis=compact_pft_axis,
        pft_count=len(selected_pft),
        scale=scale,
        penalty=penalty,
        scale_pft_index=scale_pft_index,
        scale_source_pft_index=source_pft_index,
    )


def _compact_canopy_carbon(other, plan):
    field = jnp.asarray(other, dtype=jnp.float64)[plan.leaf.start : plan.leaf.stop]
    field = field.reshape(plan.leaf.shape)
    carbon = jnp.take(field, ICARBON, axis=plan.carbon_axis)
    return jnp.moveaxis(carbon, plan.compact_pft_axis, -1).reshape((plan.pft_count,))


def _litter_change_plan(runtime, contract, protocol):
    """Resolve the optional two-owner litter-change loss and frozen scales."""

    specification = protocol.get("litter_carbon_stock_change")
    if specification is None:
        return None
    required = {
        "weight",
        "penalty",
        "scale_source_pft_index",
        "minimum_scale_gC_m2_day",
    }
    if not isinstance(specification, dict) or set(specification) != required:
        raise ValueError("litter carbon stock-change protocol keys drift")
    weight = float(specification["weight"])
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("litter carbon stock-change weight must be finite and nonnegative")
    penalty = str(specification["penalty"])
    if penalty != "pseudo_huber":
        raise ValueError("litter carbon stock-change penalty must be pseudo_huber")
    if weight == 0.0:
        return None
    scales = derive_litter_change_scales(
        runtime["base_statistics"].arrays["state_delta"].variance,
        contract,
        scale_source_pft_index=int(specification["scale_source_pft_index"]),
        minimum_scale_gC_m2_day=float(specification["minimum_scale_gC_m2_day"]),
    )
    return SimpleNamespace(
        weight=weight,
        penalty=penalty,
        layout=resolve_litter_carbon_layout(contract),
        scales=scales,
    )


def _litterpart_feedback_plan(head, weights, *, scaling=None):
    """Resolve optional litterpart supervision with an explicit scale policy."""

    weight = float(weights.get("litterpart_feedback_mse", 0.0))
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("litterpart feedback weight must be finite and nonnegative")
    if weight == 0.0:
        return None
    if scaling is None:
        scale_policy = "input_standard_deviation"
    else:
        if not isinstance(scaling, dict) or set(scaling) != {"litterpart"}:
            raise ValueError("feedback-state scaling protocol keys drift")
        scale_policy = str(scaling["litterpart"])
        if scale_policy not in {"input_standard_deviation", "physical_unit"}:
            raise ValueError("litterpart feedback-state scaling policy is unsupported")
    matches = tuple(leaf for leaf in head.contract.state_leaves if leaf.key == LITTERPART_FEEDBACK_KEY)
    if len(matches) != 1:
        raise ValueError("litterpart feedback state owner drift")
    leaf = matches[0]
    if leaf.start is None or leaf.stop is None or "nvm" not in leaf.axis_names:
        raise ValueError("litterpart feedback requires a continuous compact PFT state leaf")
    pft_axis = leaf.axis_names.index("nvm")
    pft_count = int(leaf.shape[pft_axis])
    if len(leaf.selected_pft_indices) != pft_count:
        raise ValueError("litterpart feedback compact PFT layout drift")
    input_scale = np.asarray(head.input_normalization.continuous_scale, dtype=np.float64)
    if input_scale.shape != (head.contract.continuous_state_width,):
        raise ValueError("litterpart feedback input-normalization scale width drift")
    field_scale = input_scale[leaf.start : leaf.stop].reshape(leaf.shape)
    if scale_policy == "physical_unit":
        field_scale = np.ones_like(field_scale)
    return SimpleNamespace(
        weight=weight,
        leaf=leaf,
        pft_axis=pft_axis,
        pft_count=pft_count,
        scale=jnp.asarray(field_scale, dtype=jnp.float64),
        scale_policy=scale_policy,
    )


def _state_leaf(other, leaf):
    return jnp.asarray(other, dtype=jnp.float64)[leaf.start : leaf.stop].reshape(leaf.shape)


def _pft_eligible(active_pft_mask, plan):
    active = jnp.asarray(active_pft_mask, dtype=jnp.bool_)
    if active.shape != (plan.pft_count,):
        raise ValueError("feedback state PFT input layout drift")
    shape = [1] * len(plan.leaf.shape)
    shape[plan.pft_axis] = plan.pft_count
    return active.reshape(shape)


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
    canopy_plan = _canopy_change_plan(runtime, head.contract, protocol)
    litter_change_plan = _litter_change_plan(runtime, head.contract, protocol)
    litterpart_plan = _litterpart_feedback_plan(
        head,
        weights,
        scaling=protocol.get("feedback_state_scaling"),
    )
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
        base_loss = (weights['independent_daily_flux_mse'] * flux_total
                     + weights['one_day_state_anchor'] * state_total
                     + weights['necessary_process_supervision'] * process_total)
        canopy_result = None
        canopy_scale_pft_applicable = None
        if canopy_plan is not None:
            start_pred = _compact_canopy_carbon(assembled.state.other, canopy_plan)
            next_pred = _compact_canopy_carbon(result.next_state.other, canopy_plan)
            start_teacher = _compact_canopy_carbon(assembled.state.other, canopy_plan)
            next_teacher = _compact_canopy_carbon(sample.expected_next_state.other, canopy_plan)
            active = jnp.asarray(assembled.pft.active_pft_mask, dtype=jnp.bool_)
            bare = jnp.asarray(assembled.pft.pft_traits[:, PFT_TRAIT_BARE_SOIL] > .5, dtype=jnp.bool_)
            if active.shape != (canopy_plan.pft_count,) or bare.shape != active.shape:
                raise ValueError("canopy carbon loss PFT input layout drift")
            canopy_scale_pft_applicable = active[canopy_plan.scale_pft_index] & ~bare[canopy_plan.scale_pft_index]
            canopy_result = canopy_carbon_stock_change_loss(
                start_pred=start_pred,
                next_pred=next_pred,
                start_teacher=start_teacher,
                next_teacher=next_teacher,
                start_pred_defined=_compact_canopy_carbon(assembled.state_defined.other, canopy_plan),
                next_pred_defined=_compact_canopy_carbon(result.next_state_defined.other, canopy_plan),
                start_teacher_defined=_compact_canopy_carbon(assembled.state_defined.other, canopy_plan),
                next_teacher_defined=_compact_canopy_carbon(sample.expected_next_state_defined.other, canopy_plan),
                active_pft_mask=active,
                bare_pft_mask=bare,
                scale=jnp.asarray(canopy_plan.scale, dtype=jnp.float64),
                penalty=canopy_plan.penalty,
            )
            loss = base_loss + canopy_plan.weight * canopy_result.loss
        else:
            loss = base_loss
        litter_change_result = None
        if litter_change_plan is not None:
            litter_arguments = {
                "pft_fraction": assembled.pft.pft_fraction,
                "active_pft_mask": assembled.pft.active_pft_mask,
                "contract": head.contract,
                "layout": litter_change_plan.layout,
            }
            initial_litter = extract_litter_carbon_pools(
                assembled.state.other,
                assembled.state_defined.other,
                **litter_arguments,
            )
            predicted_litter = extract_litter_carbon_pools(
                result.next_state.other,
                result.next_state_defined.other,
                **litter_arguments,
            )
            teacher_litter = extract_litter_carbon_pools(
                sample.expected_next_state.other,
                sample.expected_next_state_defined.other,
                **litter_arguments,
            )
            litter_change_result = litter_carbon_stock_change_loss(
                initial_teacher=initial_litter,
                next_predicted=predicted_litter,
                next_teacher=teacher_litter,
                scale=jnp.asarray(litter_change_plan.scales.effective_scale, dtype=jnp.float64),
                penalty=litter_change_plan.penalty,
            )
            loss = loss + litter_change_plan.weight * litter_change_result.loss
        litterpart_result = None
        if litterpart_plan is not None:
            litterpart_result = feedback_state_mse(
                predicted=_state_leaf(result.next_state.other, litterpart_plan.leaf),
                teacher=_state_leaf(sample.expected_next_state.other, litterpart_plan.leaf),
                predicted_defined=_state_leaf(result.next_state_defined.other, litterpart_plan.leaf),
                teacher_defined=_state_leaf(sample.expected_next_state_defined.other, litterpart_plan.leaf),
                eligible=_pft_eligible(assembled.pft.active_pft_mask, litterpart_plan),
                scale=litterpart_plan.scale,
            )
            loss = loss + litterpart_plan.weight * litterpart_result.loss
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
        if canopy_result is not None:
            hard['canopy_change_invalid'] = (
                jnp.sum(~canopy_result.row_valid, dtype=jnp.int32)
                + jnp.asarray(~canopy_result.scale_valid, dtype=jnp.int32)
                + jnp.asarray(~canopy_scale_pft_applicable, dtype=jnp.int32)
            )
        if litter_change_result is not None:
            hard['litter_change_invalid'] = litter_change_result.invalid_count
            hard['litter_change_empty'] = litter_change_result.empty_count
        if litterpart_result is not None:
            hard['litterpart_feedback_invalid'] = litterpart_result.invalid_count
            hard['litterpart_feedback_empty'] = jnp.asarray(
                litterpart_result.expected_count == 0,
                dtype=jnp.int32,
            )
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
        if canopy_result is not None:
            auxiliary['canopy_stock_change'] = {
                **canopy_result._asdict(),
                'weight': jnp.asarray(canopy_plan.weight, dtype=jnp.float64),
                'scale': jnp.asarray(canopy_plan.scale, dtype=jnp.float64),
                'scale_source_pft_index': jnp.asarray(canopy_plan.scale_source_pft_index, dtype=jnp.int32),
                'scale_pft_applicable': canopy_scale_pft_applicable,
            }
        if litter_change_result is not None:
            scale_record = litter_change_plan.scales
            per_owner = {}
            for owner_index, owner_name in enumerate(LITTER_OWNER_NAMES):
                per_owner[owner_name] = {
                    'loss': litter_change_result.owner_loss[owner_index],
                    'predicted_delta': litter_change_result.predicted_delta[..., owner_index],
                    'teacher_delta': litter_change_result.teacher_delta[..., owner_index],
                    'physical_residual': litter_change_result.physical_residual[..., owner_index],
                    'normalized_residual': litter_change_result.normalized_residual[..., owner_index],
                    'valid': litter_change_result.valid[..., owner_index],
                    'valid_count': litter_change_result.valid_count[owner_index],
                    'expected_count': litter_change_result.expected_count[owner_index],
                    'invalid_prediction_count': litter_change_result.invalid_prediction_count[owner_index],
                    'invalid_target_count': litter_change_result.invalid_target_count[owner_index],
                    'raw_scale': jnp.asarray(scale_record.raw_scale[owner_index], dtype=jnp.float64),
                    'effective_scale': jnp.asarray(scale_record.effective_scale[owner_index], dtype=jnp.float64),
                    'scale_coordinate_count': jnp.asarray(scale_record.coordinate_count[owner_index], dtype=jnp.int32),
                }
            auxiliary['litter_stock_change'] = {
                'total': jnp.asarray(litter_change_plan.weight, dtype=jnp.float64) * litter_change_result.loss,
                'unweighted_loss': litter_change_result.loss,
                'weight': jnp.asarray(litter_change_plan.weight, dtype=jnp.float64),
                'scale_source_pft_index': jnp.asarray(scale_record.scale_source_pft_index, dtype=jnp.int32),
                'scale_compact_pft_index': jnp.asarray(scale_record.scale_compact_pft_index, dtype=jnp.int32),
                'minimum_scale_gC_m2_day': jnp.asarray(scale_record.minimum_scale_gC_m2_day, dtype=jnp.float64),
                'hard_invalid_count': litter_change_result.invalid_count,
                'hard_empty_count': litter_change_result.empty_count,
                'per_owner': per_owner,
            }
        if litterpart_result is not None:
            auxiliary['litterpart_feedback'] = {
                **litterpart_result._asdict(),
                'weight': jnp.asarray(litterpart_plan.weight, dtype=jnp.float64),
                'scale': litterpart_plan.scale,
            }
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
