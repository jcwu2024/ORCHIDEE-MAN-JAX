from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import (
    GPP_FIELD,
    MechanisticTargetOrientedCoupledGPPConditionedAttentionProcessHead,
    bind_daily_operator,
    bounded_cosine_attention_logits,
    build_target_oriented_coupled_gpp_conditioned_attention_head,
    conditioned_continuity_gate,
    conditioned_coupled_gpp_trainable_tree,
    coupled_gpp_conditioned_attention_parameter_count,
    coupled_gpp_conditioned_attention_temperatures,
    fixed_reference_attention_temperatures,
    initialize_coupled_gpp_conditioned_attention_parameters,
    load_coupled_gpp_conditioned_attention_contract,
    masked_mechanistic_output_loss,
    plumbing_test_retained_tail_adapter,
    replace_conditioned_coupled_gpp_trainable,
    target_oriented_coupled_gpp_carbon_feedback_process_outputs,
    target_oriented_coupled_gpp_conditioned_attention_process_outputs,
    validate_conservative_result,
)
from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (
    NeuralPrecisionPolicy,
    strong_baseline_parameter_count,
)
from tests.unit.test_daily_operator_hierarchical_coordinate_attention import (
    _complete_prediction_for_loss,
    _host_l2,
    _permuted_candidate_inputs,
)
from tests.unit.test_target_oriented_coupled_gpp_carbon_feedback import (
    coupled_gpp_setup as _coupled_gpp_setup_fixture,
)


@pytest.fixture(scope="module")
def conditioned_gpp_setup():
    parent = _coupled_gpp_setup_fixture.__wrapped__()
    contract = load_coupled_gpp_conditioned_attention_contract()
    head = build_target_oriented_coupled_gpp_conditioned_attention_head(
        parent["candidate_head"],
        conditioning_contract=contract,
    )
    parameters = initialize_coupled_gpp_conditioned_attention_parameters(
        parent["candidate_parameters"],
        conditioning_contract=contract,
        structured_config=head.structured_config,
        config=head.conditioning_config,
    )
    assert isinstance(head, MechanisticTargetOrientedCoupledGPPConditionedAttentionProcessHead)
    return parent | {
        "candidate_parameters": parameters,
        "candidate_head": head,
        "parent_parameters": parent["candidate_parameters"],
        "parent_head": parent["candidate_head"],
        "conditioning_contract": contract,
    }


def _parent_outputs_with_temperatures(setup, parameters, temperatures):
    head = setup["candidate_head"]
    inputs = setup["assembled"]
    context = head.assemble_context(parameters, inputs)
    return target_oriented_coupled_gpp_carbon_feedback_process_outputs(
        parameters,
        context,
        inputs,
        schema=head.learned_schema,
        layout=head.layout,
        structured_state_layout=head.structured_state_layout,
        carbon_state_layout=head.carbon_state_layout,
        input_normalization=head.input_normalization,
        output_normalization=head.output_normalization,
        forcing_layout=head.forcing_layout,
        target_contract=head.target_contract,
        distribution_contract=head.distribution_contract,
        coeff_maint_zero=head.coeff_maint_zero,
        ext_coeff_vegetfrac=head.reconstructor.static(inputs, "ext_coeff_vegetfrac"),
        message_gate_scale=head.message_gate_scale,
        baseline_config=head.config,
        attention_config=head.target_config,
        distribution_config=head.distribution_config,
        source_config=head.source_config,
        structured_config=head.structured_config,
        config=head.coupled_config,
        potential_attention_temperatures=temperatures,
        potential_attention_normalization_epsilon=head.conditioning_config.normalization_epsilon,
    )


def test_conditioning_contract_parameters_and_matched_reference(conditioned_gpp_setup):
    setup = conditioned_gpp_setup
    contract = setup["conditioning_contract"]
    parameters = setup["candidate_parameters"]
    head = setup["candidate_head"]

    assert contract.sha256 == "b9bdd9dde51a8f6f1de2c21cd6678f5a51232011397501459d614b549ca560d4"
    assert coupled_gpp_conditioned_attention_parameter_count(parameters) == 109_464_375
    assert strong_baseline_parameter_count(parameters["target_oriented"]["coupled_gpp_conditioning"]) == 16
    assert tuple(conditioned_coupled_gpp_trainable_tree(parameters)) == (
        "structured_gpp",
        "coupled_gpp_carbon_feedback",
        "coupled_gpp_conditioning",
    )
    assert strong_baseline_parameter_count(conditioned_coupled_gpp_trainable_tree(parameters)) == 15_960_211

    candidate = coupled_gpp_conditioned_attention_temperatures(
        parameters,
        config=head.conditioning_config,
    )
    reference = fixed_reference_attention_temperatures(config=head.conditioning_config)
    assert candidate.shape == (2, 8)
    assert np.all(np.asarray(candidate) > 0.0)
    assert np.all(np.asarray(candidate) < 12.0)
    np.testing.assert_allclose(np.asarray(candidate), np.asarray(reference), rtol=1.0e-7, atol=1.0e-7)

    fixed = replace_conditioned_coupled_gpp_trainable(parameters, {})
    rebuilt = replace_conditioned_coupled_gpp_trainable(
        fixed,
        conditioned_coupled_gpp_trainable_tree(parameters),
    )
    assert (
        rebuilt["target_oriented"]["coupled_gpp_conditioning"]
        is parameters["target_oriented"]["coupled_gpp_conditioning"]
    )


def test_full_active_head_accepts_attention_fp32_policy(conditioned_gpp_setup):
    setup = conditioned_gpp_setup
    head = setup["candidate_head"]
    policy = NeuralPrecisionPolicy(attention_scores_fp32=True)
    head = replace(head, config=replace(head.config, neural_precision=policy))

    shaped = jax.eval_shape(
        lambda values: head.learned_outputs(
            values,
            head.assemble_context(values, setup["assembled"]),
            setup["assembled"],
        ),
        setup["candidate_parameters"],
    )

    assert head.config.neural_precision == policy
    assert shaped[GPP_FIELD].shape == (1, 2)
    assert shaped[GPP_FIELD].dtype == jnp.float64


def test_bounded_cosine_logits_ignore_projected_magnitude_and_remain_bounded():
    width = 8
    parameters = {
        name: {"kernel": jnp.eye(width), "bias": jnp.zeros((width,))} for name in ("query", "key", "value", "output")
    }
    query = jnp.asarray([[[1.0, -2.0, 3.0, 4.0, -1.0, 2.0, -3.0, 0.5]]])
    bank = jnp.asarray(
        [
            [
                [2.0, 1.0, -1.0, 3.0, 4.0, -2.0, 1.0, 0.25],
                [-3.0, 2.0, 1.0, -0.5, 1.0, 2.0, 3.0, -4.0],
            ]
        ]
    )
    temperatures = jnp.asarray([4.0, 12.0])
    base = bounded_cosine_attention_logits(
        parameters,
        query,
        bank,
        heads=2,
        temperatures=temperatures,
        normalization_epsilon=1.0e-6,
    )
    scaled = bounded_cosine_attention_logits(
        parameters,
        query * 1.0e9,
        bank * 1.0e8,
        heads=2,
        temperatures=temperatures,
        normalization_epsilon=1.0e-6,
    )
    np.testing.assert_allclose(np.asarray(scaled), np.asarray(base), rtol=2.0e-6, atol=2.0e-6)
    assert float(jnp.max(jnp.abs(scaled[0, 0]))) <= 4.000001
    assert float(jnp.max(jnp.abs(scaled[0, 1]))) <= 12.000001


def test_conditioned_continuity_gate_requires_absolute_mechanism_repair():
    passed = conditioned_continuity_gate(
        parent_lower_gpp=2.0,
        parent_upper_gpp=4.0,
        candidate_lower_gpp=2.5,
        candidate_upper_gpp=2.8,
        reference_lower_gpp=2.5,
        reference_upper_gpp=2.8,
        maximum_absolute_logit=11.9,
        jump_fraction_maximum=0.25,
        logit_absolute_maximum=12.000001,
        reference_rtol=1.0e-6,
        reference_atol=1.0e-8,
    )
    failed = conditioned_continuity_gate(
        parent_lower_gpp=2.0,
        parent_upper_gpp=4.0,
        candidate_lower_gpp=2.5,
        candidate_upper_gpp=3.5,
        reference_lower_gpp=2.5,
        reference_upper_gpp=3.5,
        maximum_absolute_logit=11.9,
        jump_fraction_maximum=0.25,
        logit_absolute_maximum=12.000001,
        reference_rtol=1.0e-6,
        reference_atol=1.0e-8,
    )

    assert passed["passed"] is True
    assert passed["conditioned_jump_fraction_of_parent"] == pytest.approx(0.15)
    assert failed["passed"] is False
    assert failed["jump_reduction_gate_passed"] is False


def test_conditioned_forward_loss_and_gradients(conditioned_gpp_setup):
    setup = conditioned_gpp_setup
    head = setup["candidate_head"]
    parameters = setup["candidate_parameters"]
    inputs = setup["assembled"]
    context = head.assemble_context(parameters, inputs)
    eager = head.learned_outputs(parameters, context, inputs)
    compiled = jax.jit(lambda values: head.learned_outputs(values, context, inputs))(parameters)
    jax.block_until_ready(compiled[GPP_FIELD])
    assert all(np.all(np.isfinite(np.asarray(value))) for value in eager.values())
    assert all(np.all(np.isfinite(np.asarray(value))) for value in compiled.values())

    reference = _parent_outputs_with_temperatures(
        setup,
        parameters,
        fixed_reference_attention_temperatures(config=head.conditioning_config),
    )
    np.testing.assert_allclose(
        np.asarray(eager[GPP_FIELD]),
        np.asarray(reference[GPP_FIELD]),
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    reference_head = build_target_oriented_coupled_gpp_conditioned_attention_head(
        setup["parent_head"],
        conditioning_contract=setup["conditioning_contract"],
        fixed_temperature_reference=True,
    )
    reference_head_outputs = reference_head.learned_outputs(
        parameters,
        reference_head.assemble_context(parameters, inputs),
        inputs,
    )
    np.testing.assert_allclose(
        np.asarray(eager[GPP_FIELD]),
        np.asarray(reference_head_outputs[GPP_FIELD]),
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    parent_outputs = setup["parent_head"].learned_outputs(
        setup["parent_parameters"],
        setup["parent_head"].assemble_context(setup["parent_parameters"], inputs),
        inputs,
    )
    for name in eager:
        if name != GPP_FIELD:
            np.testing.assert_array_equal(np.asarray(eager[name]), np.asarray(parent_outputs[name]))

    prediction, complete = _complete_prediction_for_loss(setup, parameters, context)
    loss = masked_mechanistic_output_loss(
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
    assert np.isfinite(float(loss.total))

    raw = parameters["target_oriented"]["coupled_gpp_conditioning"]

    def temperature_proxy(conditioning):
        target = dict(parameters["target_oriented"], coupled_gpp_conditioning=conditioning)
        changed = dict(parameters, target_oriented=target)
        return jnp.sum(head.learned_outputs(changed, context, inputs)[GPP_FIELD])

    value, temperature_gradients = jax.value_and_grad(temperature_proxy)(raw)
    temperature_gradient = np.asarray(temperature_gradients["potential_temperature_logits"])
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(temperature_gradient))
    assert np.all(np.linalg.norm(temperature_gradient, axis=1) > 0.0)

    specialist = parameters["target_oriented"]["structured_gpp"]

    def potential_proxy(blocks):
        changed_specialist = dict(specialist, potential_blocks=blocks)
        target = dict(parameters["target_oriented"], structured_gpp=changed_specialist)
        changed = dict(parameters, target_oriented=target)
        return jnp.sum(head.learned_outputs(changed, context, inputs)[GPP_FIELD])

    _, block_gradients = jax.value_and_grad(potential_proxy)(specialist["potential_blocks"])
    for block in block_gradients:
        for projection in ("query", "key", "value", "output"):
            assert _host_l2(block["attention"][projection]["kernel"]) > 0.0


def test_conditioned_masks_permutation_and_complete_daily_budget(conditioned_gpp_setup):
    setup = conditioned_gpp_setup
    head = setup["candidate_head"]
    parameters = setup["candidate_parameters"]
    inputs = setup["assembled"]
    outputs = head.learned_outputs(parameters, head.assemble_context(parameters, inputs), inputs)

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
    np.testing.assert_array_equal(np.asarray(inactive_outputs[GPP_FIELD])[:, 1], 0.0)

    permuted = _permuted_candidate_inputs(setup)
    permuted_outputs = head.learned_outputs(
        parameters,
        head.assemble_context(parameters, permuted),
        permuted,
    )
    np.testing.assert_allclose(
        np.asarray(permuted_outputs[GPP_FIELD]),
        np.asarray(outputs[GPP_FIELD])[:, ::-1],
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


def test_conditioned_inference_has_no_subdaily_reconstruction_or_scan():
    source = inspect.getsource(target_oriented_coupled_gpp_conditioned_attention_process_outputs)
    assert "lax.scan" not in source
    assert "48" not in source
    assert "optimizer" not in source
    assert "checkpoint" not in source
