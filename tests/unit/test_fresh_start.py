from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator import (
    bind_daily_operator,
    masked_mechanistic_output_loss,
    plumbing_test_retained_tail_adapter,
    validate_conservative_result,
)
from research.daily_coarse_graining.daily_operator.fresh_start import initialize_fresh_parameters
from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
from research.daily_coarse_graining.daily_operator.target_oriented_training import initialize_target_oriented_adamw
from tests.unit.test_daily_operator_hierarchical_coordinate_attention import _complete_prediction_for_loss
from tests.unit.test_target_oriented_coupled_gpp_conditioned_attention import conditioned_gpp_setup  # noqa: F401


def test_adam_updates_doc_respiration_and_shared_parameters_together():
    protocol = json.loads((Path(__file__).resolve().parents[2] / 'manifests/coarse_graining/fresh_start_one_day_probe_v1.json').read_text())
    parameters = {name: jnp.array([1., 2.], dtype=jnp.float32) for name in ('doc', 'respiration', 'shared', 'gpp')}
    gradients = jax.tree_util.tree_map(jnp.ones_like, parameters)
    result, state, norm, scale = make_update(protocol)(parameters, gradients, initialize_target_oriented_adamw(parameters))
    assert int(state.step) == 1 and np.isfinite(float(norm)) and 0 < float(scale) <= 1
    for name in parameters:
        assert np.all(np.asarray(result[name]) < np.asarray(parameters[name]))


def test_fresh_tree_forward_loss_gradient_and_canonical_bound_day(conditioned_gpp_setup):  # noqa: F811
    setup = dict(conditioned_gpp_setup)
    head = setup['candidate_head']
    doc = next(v for v in head.contract.state_leaves if v.key == 'slowproc_stomate_previous_step_state.DOC')
    mask = np.zeros_like(head.input_normalization.continuous_mean, dtype=bool)
    mask[doc.start:doc.stop] = True
    head = replace(head, input_normalization=head.input_normalization._replace(
        continuous_tail_threshold=16., continuous_tail_mask=mask))
    setup['candidate_head'] = head
    parameters = initialize_fresh_parameters(head, setup['assembled'], seed=20260911)
    assert sum(v.size for v in jax.tree_util.tree_leaves(parameters)) == 109464375
    assert not np.array_equal(parameters['summary_input']['kernel'], setup['candidate_parameters']['summary_input']['kernel'])
    setup['candidate_parameters'] = parameters

    def loss(values):
        context = head.assemble_context(values, setup['assembled'])
        prediction, complete = _complete_prediction_for_loss(setup, values, context)
        return masked_mechanistic_output_loss(complete, setup['targets'], inputs=setup['assembled'],
            complete_schema=setup['schema'], learned_schema=setup['learned_schema'], scales=setup['scales'],
            architecture_contract=setup['architecture_contract'], mechanism_contract=setup['mechanism_contract'],
            auxiliary_outputs=prediction.auxiliary_outputs).total

    value, gradient = jax.jit(jax.value_and_grad(loss))(parameters)
    assert np.isfinite(float(value))
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(gradient))
    for family in ('state_heads', 'distribution_aware_maintenance', 'structured_gpp'):
        assert any(np.any(np.asarray(v) != 0) for v in jax.tree_util.tree_leaves(gradient['target_oriented'][family]))
    transition = bind_daily_operator(input_assembler=setup['adapter'].input_assembler,
        process_head=head, retained_tail=plumbing_test_retained_tail_adapter())
    result = transition(parameters, setup['assembled'].canonical_input)
    validate_conservative_result(result.fast_update)
    assert abs(float(result.fast_update.water.budget_residual)) <= 2e-12
    assert abs(float(result.fast_update.carbon.budget_residual)) <= 2e-12
