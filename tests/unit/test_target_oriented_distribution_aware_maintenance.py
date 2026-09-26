from __future__ import annotations

import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import (
    MAINTENANCE_FIELD,
    MechanisticTargetOrientedDistributionAwareMaintenanceProcessHead,
    bind_daily_operator,
    distribution_aware_maintenance_corrections,
    distribution_aware_maintenance_parameter_count,
    initialize_distribution_aware_maintenance_parameters,
    load_distribution_aware_maintenance_contract,
    masked_mechanistic_output_loss,
    plumbing_test_retained_tail_adapter,
    resolve_maintenance_field_index,
    target_oriented_attention_process_outputs,
    validate_conservative_result,
)
from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (
    strong_baseline_parameter_count,
)
from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_maintenance import (
    target_oriented_distribution_aware_maintenance_process_outputs,
)
from tests.unit.test_daily_operator_hierarchical_coordinate_attention import (
    _complete_prediction_for_loss,
    _host_l2,
    _permuted_candidate_inputs,
    _production_valid_inputs,
)
from tests.unit.test_daily_operator_target_oriented_attention import (
    target_attention_setup as _target_attention_setup_fixture,
)


@pytest.fixture(scope="module")
def distribution_aware_setup():
    setup = _target_attention_setup_fixture.__wrapped__()
    parent_parameters = setup["candidate_parameters"]
    distribution_contract = load_distribution_aware_maintenance_contract()
    parameters = initialize_distribution_aware_maintenance_parameters(
        parent_parameters,
        jax.random.PRNGKey(20260828),
        schema=setup["learned_schema"],
        target_contract=setup["target_contract"],
        distribution_contract=distribution_contract,
    )
    coefficient = np.zeros((2, 12), dtype=np.float64)
    coefficient[1] = 1.0e-4
    parent = setup["candidate_head"]
    head = MechanisticTargetOrientedDistributionAwareMaintenanceProcessHead(
        contract=parent.contract,
        architecture_contract=parent.architecture_contract,
        mechanism_contract=parent.mechanism_contract,
        input_assembler=parent.input_assembler,
        complete_schema=parent.complete_schema,
        learned_schema=parent.learned_schema,
        layout=parent.layout,
        input_normalization=parent.input_normalization,
        output_normalization=parent.output_normalization,
        forcing_layout=parent.forcing_layout,
        forcing_indices=parent.forcing_indices,
        message_gate_scale=parent.message_gate_scale,
        config=parent.config,
        target_contract=parent.target_contract,
        target_config=parent.target_config,
        distribution_contract=distribution_contract,
        coeff_maint_zero=coefficient,
    )
    candidate = setup | {
        "candidate_parameters": parameters,
        "candidate_head": head,
        "parent_parameters": parent_parameters,
        "parent_head": parent,
        "distribution_contract": distribution_contract,
    }
    inputs = _production_valid_inputs(candidate)
    state = np.asarray(inputs.pft.pft_state).copy()
    state[1, 0] = 0.8
    state[1, 1] = 2.0
    element_count = (state.shape[1] - 2) // 12
    biomass = state[1, 2:].reshape((12, element_count))
    biomass[:, 0] = np.linspace(2.0, 13.0, 12)
    state[1, 2:] = biomass.reshape(-1)
    fraction = np.asarray(inputs.pft.pft_fraction).copy()
    fraction[:] = (0.2, 0.8)
    pft_parameters = np.asarray(inputs.pft.pft_parameters).copy()
    pft_parameters[1, 0] = 60.0
    active_pft = inputs.pft._replace(
        pft_state=state,
        pft_parameters=pft_parameters,
        pft_fraction=fraction,
        active_pft_mask=np.asarray([True, True]),
    )
    inputs = inputs._replace(
        pft=active_pft,
        canonical_input=inputs.canonical_input._replace(pft=active_pft),
    )
    return candidate | {"assembled": inputs}


def test_distribution_aware_contract_parameter_tree_and_24_outputs_are_exact(
    distribution_aware_setup,
):
    setup = distribution_aware_setup
    contract = setup["distribution_contract"]
    parameters = setup["candidate_parameters"]
    parent = setup["parent_parameters"]
    index = resolve_maintenance_field_index(
        setup["learned_schema"],
        setup["target_contract"],
    )

    assert contract.sha256 == "f167943cfb80902fd968259f7f2a1a852d783b151a7eedd16b3157abd1a37d2d"
    assert distribution_aware_maintenance_parameter_count(parameters) == 93_504_933
    assert parameters["target_oriented"]["primary_heads"][index] == {}
    parent_heads = parent["target_oriented"]["primary_heads"]
    candidate_heads = parameters["target_oriented"]["primary_heads"]
    assert all(candidate_heads[i] is parent_heads[i] for i in range(len(parent_heads)) if i != index)

    adapter = parameters["target_oriented"]["distribution_aware_maintenance"]
    assert strong_baseline_parameter_count(adapter) == 9_750_529
    assert adapter["context_projection"]["kernel"].shape == (384, 768)
    assert len(adapter["blocks"]) == 2
    assert adapter["blocks"][0]["attention"]["query"]["kernel"].shape == (768, 768)
    assert adapter["blocks"][0]["ffn"]["first"]["kernel"].shape == (768, 1536)
    assert adapter["output"]["kernel"].shape == (768, 1)
    assert np.any(np.asarray(adapter["output"]["kernel"]) != 0.0)
    np.testing.assert_array_equal(np.asarray(adapter["output"]["bias"]), 0.0)

    inputs = setup["assembled"]
    context = setup["candidate_head"].assemble_context(parameters, inputs)
    hidden = jax.random.normal(jax.random.PRNGKey(7), (1, 2, 12, 768))
    corrections = distribution_aware_maintenance_corrections(parameters, context, hidden)
    assert corrections.shape == (2, 12)
    assert np.all(np.isfinite(np.asarray(corrections)))
    assert np.max(np.abs(np.asarray(corrections))) <= 1.0
    assert np.ptp(np.asarray(corrections)) > 1.0e-6
    assert "mean" not in inspect.getsource(distribution_aware_maintenance_corrections)


def test_distribution_aware_forward_loss_and_adapter_gradients_are_finite(
    distribution_aware_setup,
):
    setup = distribution_aware_setup
    head = setup["candidate_head"]
    parameters = setup["candidate_parameters"]
    inputs = setup["assembled"]
    context = head.assemble_context(parameters, inputs)
    eager = head.learned_outputs(parameters, context, inputs)
    compiled = jax.jit(lambda values, encoded, assembled: head.learned_outputs(values, encoded, assembled))(
        parameters, context, inputs
    )
    jax.block_until_ready(compiled[MAINTENANCE_FIELD])
    assert all(np.all(np.isfinite(np.asarray(value))) for value in eager.values())
    assert all(np.all(np.isfinite(np.asarray(value))) for value in compiled.values())

    prediction, complete = _complete_prediction_for_loss(setup, parameters, context)
    complete_loss = masked_mechanistic_output_loss(
        complete,
        setup["targets"],
        inputs=inputs,
        complete_schema=setup["schema"],
        learned_schema=setup["learned_schema"],
        scales=setup["scales"],
        architecture_contract=setup["architecture_contract"],
        mechanism_contract=setup["mechanism_contract"],
        auxiliary_outputs=prediction.auxiliary_outputs,
    )
    assert np.isfinite(float(complete_loss.total))

    def proxy(adapter_parameters):
        target = dict(parameters["target_oriented"])
        target["distribution_aware_maintenance"] = adapter_parameters
        changed = dict(parameters, target_oriented=target)
        outputs = head.learned_outputs(changed, context, inputs)
        return jnp.sum(outputs[MAINTENANCE_FIELD])

    value, gradients = jax.value_and_grad(proxy)(parameters["target_oriented"]["distribution_aware_maintenance"])
    assert np.isfinite(float(value))
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(gradients))
    assert _host_l2(gradients["context_projection"]["kernel"]) > 0.0
    assert _host_l2(gradients["blocks"][0]["attention"]["query"]["kernel"]) > 0.0
    assert _host_l2(gradients["blocks"][1]["ffn"]["first"]["kernel"]) > 0.0
    assert _host_l2(gradients["output"]["kernel"]) > 0.0


def test_distribution_aware_preserves_parent_outputs_and_daily_boundary(
    distribution_aware_setup,
):
    setup = distribution_aware_setup
    head = setup["candidate_head"]
    parameters = setup["candidate_parameters"]
    parent = setup["parent_head"]
    parent_parameters = setup["parent_parameters"]
    inputs = setup["assembled"]
    candidate_outputs = head.learned_outputs(
        parameters,
        head.assemble_context(parameters, inputs),
        inputs,
    )
    parent_outputs = target_oriented_attention_process_outputs(
        parent_parameters,
        parent.assemble_context(parent_parameters, inputs),
        inputs,
        schema=parent.learned_schema,
        layout=parent.layout,
        input_normalization=parent.input_normalization,
        output_normalization=parent.output_normalization,
        forcing_layout=parent.forcing_layout,
        target_contract=parent.target_contract,
        message_gate_scale=parent.message_gate_scale,
        baseline_config=parent.config,
        config=parent.target_config,
    )
    for name in parent_outputs:
        if name != MAINTENANCE_FIELD:
            np.testing.assert_array_equal(np.asarray(candidate_outputs[name]), np.asarray(parent_outputs[name]))
    np.testing.assert_array_equal(np.asarray(candidate_outputs[MAINTENANCE_FIELD])[:, 0], 0.0)

    inactive_pft = inputs.pft._replace(
        active_pft_mask=np.asarray([True, False]),
        pft_fraction=np.asarray([0.2, 0.0]),
    )
    inactive = inputs._replace(
        pft=inactive_pft,
        canonical_input=inputs.canonical_input._replace(pft=inactive_pft),
    )
    inactive_outputs = head.learned_outputs(
        parameters,
        head.assemble_context(parameters, inactive),
        inactive,
    )
    np.testing.assert_array_equal(np.asarray(inactive_outputs[MAINTENANCE_FIELD])[:, 1], 0.0)

    permuted = _permuted_candidate_inputs(setup)
    permuted_outputs = head.learned_outputs(
        parameters,
        head.assemble_context(parameters, permuted),
        permuted,
    )
    np.testing.assert_allclose(
        np.asarray(permuted_outputs[MAINTENANCE_FIELD]),
        np.asarray(candidate_outputs[MAINTENANCE_FIELD])[:, ::-1],
        rtol=1.0e-6,
        atol=2.0e-12,
    )

    transition = bind_daily_operator(
        input_assembler=setup["adapter"].input_assembler,
        process_head=head,
        retained_tail=plumbing_test_retained_tail_adapter(),
    )
    result = transition(parameters, inputs.canonical_input)
    validate_conservative_result(result.fast_update)
    assert result.next_state.other.shape == (3_854,)
    assert abs(float(result.fast_update.water.budget_residual)) <= 2.0e-12
    assert abs(float(result.fast_update.carbon.budget_residual)) <= 2.0e-12
